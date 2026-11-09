"""dn4g8 expert overlay from a direct-INT3 overlay (throughput; excl5 version built 23 Oct as two passes, scratch
modal_dn4.py + modal_dn4g.py): every expert's down_proj, and gate/up of LAYERS, taken back from the NVFP4 checkpoint;
served with ARC_EXPERT_INT3_FP4_DOWN=1 and ARC_EXPERT_INT3_FP4_LAYERS (build_v82 --dn4g8), so those sublayers serve
NVFP4 and the rest INT3. One pass; mirrors to the Nebius bucket with its asset manifest.

    modal run modal_dn4g8_overlay.py --src arc-expert-int3-direct-t37-all25v71b-v1 \
        --out arc-expert-int3-direct-dn4g8-t37-all25v71b-v1 --check          # CPU: two shards, verify, no write
    (--check --src arc-expert-int3-direct-t37-excl5-v1 --ref arc-expert-int3-direct-dn4g8-t37-excl5-v1: equal to the 23 Oct output)
    modal run --detach modal_dn4g8_overlay.py --src ... --out ...             # build all shards + mirror
    --base <overlay name>: take the NVFP4 tensors from that W4 overlay (a student's masked W4 GPTQ fit, e.g. p75: no
    NVFP4 checkpoint holds its experts) instead of the checkpoint; shards without swapped keys (merged dense/MTP
    shards) are copied unchanged (v9, 29 Oct).
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

PREFIX = "flash-input/datasets/richardcsaky/"
LAYERS = (20, 23, 28, 31, 32, 33, 35, 40)  # largest held-out dn4-vs-NVFP4 MoE error (tp-b2screen-all48-v1, excl5)
BASE = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"

app = modal.App("arc-dn4g8-overlay")
assets = modal.Volume.from_name("arc-prolong-assets")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
         .pip_install("torch==2.8.0", "safetensors==0.6.2", "packaging", "numpy",
                      index_url="https://download.pytorch.org/whl/cpu", extra_index_url="https://pypi.org/simple"))


def swap(key):
    m = re.search(r"layers\.(\d+)\.", key)
    return ".down_proj." in key or (m is not None and int(m.group(1)) in LAYERS
                                    and (".gate_proj." in key or ".up_proj." in key))


def convert(src_dir, out_dir, name, write, ref_dir=None, base_dir=BASE):
    import hashlib

    from safetensors import safe_open
    from safetensors.torch import save_file

    out, swapped, kept, layers = {}, 0, 0, set()
    if not (Path(base_dir) / name).exists():
        # a merged non-expert shard (student dense/MTP: its shared-expert / dense MLP down_proj are not routed experts
        # and stay as exported); an expert shard must exist in the base
        if "-experts-" in name:
            raise RuntimeError(f"{name}: expert shard missing from {base_dir}")
        if not write:
            return {"name": name, "swapped": 0, "kept": None, "layers": [], "equal_to_ref": None, "copied": True}
        import shutil
        shutil.copyfile(src_dir / name, out_dir / name)
        with open(out_dir / name, "rb") as h:
            sha = hashlib.file_digest(h, "sha256").hexdigest()
        return {"name": name, "bytes": (out_dir / name).stat().st_size, "sha256": sha, "tensors": None, "swapped": 0}
    with safe_open(str(src_dir / name), framework="pt") as fs, safe_open(str(Path(base_dir) / name), framework="pt") as fb:
        meta, base_names = fs.metadata(), set(fb.keys())
        for k in fs.keys():
            if swap(k):
                if k not in base_names:
                    raise RuntimeError(f"{name}: {k} missing from the checkpoint")
                t = fb.get_tensor(k)
                if list(t.shape) != list(fs.get_slice(k).get_shape()):
                    raise RuntimeError(f"{name}: {k} shape differs")
                out[k] = t
                swapped += 1
                layers.add(int(re.search(r"layers\.(\d+)\.", k).group(1)))
            else:
                out[k] = fs.get_tensor(k)
                kept += 1
    if swapped % 4:  # per expert: down 4 tensors (+ gate/up 8 in LAYERS)
        raise RuntimeError(f"{name}: swapped {swapped} kept {kept}")
    if not write:
        same = None
        if ref_dir is not None:  # the same transform's earlier output (e.g. the excl5 dn4g8 overlay): tensor-equal?
            import torch
            with safe_open(str(ref_dir / name), framework="pt") as fr:
                same = set(fr.keys()) == set(out) and all(torch.equal(fr.get_tensor(k), out[k]) for k in out)
        return {"name": name, "swapped": swapped, "kept": kept, "layers": sorted(layers), "equal_to_ref": same}
    save_file(out, str(out_dir / name), metadata=meta)
    with open(out_dir / name, "rb") as h:
        sha = hashlib.file_digest(h, "sha256").hexdigest()
    return {"name": name, "bytes": (out_dir / name).stat().st_size, "sha256": sha, "tensors": len(out), "swapped": swapped}


@app.function(image=image, cpu=(0.125, 8), timeout=7200, retries=0, max_containers=1, volumes={"/assets": assets},
              secrets=[modal.Secret.from_name("nebius-s3")])
def build(src: str, out: str, check: bool = False, ref: str = "", base: str = ""):
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    src_dir, out_dir = Path("/assets") / PREFIX / src, Path("/assets") / PREFIX / out
    base_dir = Path("/assets") / PREFIX / base if base else Path(BASE)
    man = json.loads((src_dir / "overlay-manifest.json").read_text())
    names = sorted(man["shards"])
    if check:
        ref_dir = Path("/assets") / PREFIX / ref if ref else None
        res = [convert(src_dir, out_dir, names[0], False, ref_dir, base_dir)]
        for n in names:
            r = convert(src_dir, out_dir, n, False, ref_dir, base_dir)
            if set(r["layers"]) & set(LAYERS):
                return res + [r]
        raise RuntimeError("no shard holds a selected layer")
    done = out_dir / "overlay-manifest.json"
    res = None
    if done.exists() and json.loads(done.read_text()).get("complete"):  # a preempted run's restart: mirror only
        shards = json.loads(done.read_text())["shards"]
    else:
        if out_dir.exists() and any(out_dir.iterdir()):
            # no complete manifest: a preempted build's partial shards (Modal restarts the input; 29 Oct v9 build hit
            # the old not-empty guard). Only this function writes here, so the partial output is dropped and rebuilt.
            import shutil
            print(f"dropping partial output {out_dir}: {sorted(q.name for q in out_dir.iterdir())[:3]}...", flush=True)
            shutil.rmtree(out_dir)
            assets.commit()
        out_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(lambda n: convert(src_dir, out_dir, n, True, None, base_dir), names))
        shards = {r["name"]: {**man["shards"][r["name"]], "bytes": r["bytes"], "sha256": r["sha256"],
                              **({"tensors": r["tensors"]} if r["tensors"] is not None else {})} for r in res}
        new = {**man, "shards": shards, "complete": True, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "settings": {**man["settings"], "source_overlay": PREFIX + src,
                            "dn4g8": f"down_proj everywhere + gate/up of layers {list(LAYERS)} from "
                                     + (f"the W4 overlay {PREFIX + base}" if base else "the NVFP4 checkpoint")}}
        done.write_text(json.dumps(new, indent=1))
        assets.commit()
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")
    subprocess.run(["rclone", "copy", str(out_dir), f"arc:{BUCKET}/assets/{PREFIX}{out}", "--transfers", "16"], check=True)
    import hashlib
    files = {}
    for f in sorted(out_dir.iterdir()):
        with open(f, "rb") as h:
            files[f.name] = {"size": f.stat().st_size, "sha256": hashlib.file_digest(h, "sha256").hexdigest()}
    bad = [n for n in shards if files.get(n, {}).get("sha256") != shards[n]["sha256"]]
    if bad or sorted(files) != sorted([*shards, "overlay-manifest.json"]):
        raise RuntimeError(f"output does not match its manifest: {bad[:5]} {sorted(set(files) ^ {*shards, 'overlay-manifest.json'})[:5]}")
    slug = (PREFIX + out).replace("/", "--") + ".json"
    Path(f"/tmp/{slug}").write_text(json.dumps({"prefix": PREFIX + out, "files": files}))
    subprocess.run(["rclone", "copyto", f"/tmp/{slug}", f"arc:{BUCKET}/assets-manifests/{slug}"], check=True)
    return {"shards": len(shards), "swapped_total": None if res is None else sum(r["swapped"] for r in res),
            "manifest_sha256": files["overlay-manifest.json"]["sha256"], "seconds": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(src: str, out: str, check: bool = False, ref: str = "", base: str = ""):
    if check:
        print(json.dumps(build.remote(src, out, True, ref, base)))
        return
    call = build.spawn(src, out, False, "", base)
    print(json.dumps({"function_call_id": call.object_id, "app_id": app.app_id}))
