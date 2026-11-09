"""Trained student's routed experts -> INT3 g16 expert overlay (arc-expert-overlay-v1, omits_pruned), the format the v9
bundle loads (scripts/dfr_exact/int3/modal_heldout_refit.py's requant output + merged dense shards + Nebius publish).

    modal run scripts/sft/modal_int3_export.py --student /runs/validation/<run>/train/ckpt-N/student \
        --mask prune-t50-r15b-nvfp4-v1.json --out arc-expert-int3-rtn-k256c268-v1 --dense sft-heal/nvfp4-dense/k256-c268

QAD's grid (qat.py int3served) applied once to the BF16 masters: arena.expert_3bit_math.rtn (int3, group 16, SEARCH),
scale_2 = amax / (4 x 448) per expert (gate and up share one, on their concatenation), packed in the NVFP4 container
by to_nvfp4. No W4 stage and no refit. Per expert, the packed dequant must equal qat.fake3 of the same weights bit for
bit (two independent paths), and every code sits on the int3 grid (verify_shard's checks). Student experts are
per-expert keys (a sharded checkpoint) or fused [kept, 2I, H] / [kept, H, I] tensors, kept experts in ascending order."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ARENA = Path("./src/arena")  # main checkout: the served int3 math (expert_3bit_overlay's)
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
OUT_ROOT = "/assets/flash-input/datasets/richardcsaky"
E, H, INTER, GROUP = 512, 2560, 640, 16
SEARCH = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
CHUNK = 16
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"
app = modal.App("sft-int3-export", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
# a student in a per-run training volume: as modal_export.py, ARC_SFT_INPUT_VOLUMES=<vol> and --student /in/<vol>/...
INPUT_VOLUMES = [v for v in (os.environ.get("ARC_SFT_INPUT_VOLUMES") or "").split(",") if v]
MOUNTS = {"/assets": assets, "/runs": runs.read_only(),
          **{f"/in/{v}": modal.Volume.from_name(v).read_only() for v in INPUT_VOLUMES}}
image = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
         .pip_install("torch==2.9.1", "safetensors", "numpy", index_url="https://download.pytorch.org/whl/cpu",
                      extra_index_url="https://pypi.org/simple")
         .add_local_file(str(HERE / "qat.py"), "/opt/sft/qat.py", copy=True)
         .add_local_file(str(HERE / "export.py"), "/opt/sft/export.py", copy=True)
         .add_local_dir(str(HERE / "prune"), "/opt/sft/prune", copy=True))
if INPUT_VOLUMES:  # the container re-imports this file: same mounts as the client's
    image = image.env({"ARC_SFT_INPUT_VOLUMES": ",".join(INPUT_VOLUMES)})
if modal.is_local():
    image = image.run_commands("mkdir -p /opt/src/arena && touch /opt/src/arena/__init__.py")
    for _n in ("expert_3bit_math.py", "expert_quant_math.py"):
        image = image.add_local_file(str(ARENA / _n), f"/opt/src/arena/{_n}", copy=True)
    image = image.add_local_file(__file__, "/root/modal_int3_export.py", copy=True)


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def cut_of(student):
    """The student's kept source layers, in order (``layer`` below is always a source id; a cut student numbers its own)."""
    return json.loads((Path(student) / "config.json").read_text())["student_cut"]["source_layers"]


def kept_of(mask_file, layer):
    spec = json.loads(Path(mask_file).read_text())
    return sorted(set(range(spec["experts"])) - {int(e) for e in spec["pruned"][str(layer)]})


def student_experts(student, layer, slots, torch):
    """{student slot: (w13 [2I, H], w2 [H, I])} in the checkpoint's dtype, per-expert or fused keys."""
    from safetensors import safe_open

    index = json.loads((Path(student) / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts."
    by = {}
    fused = pre + "gate_up_proj" in index
    names = [pre + "gate_up_proj", pre + "down_proj"] if fused else \
        [f"{pre}{s}.{p}.weight" for s in slots for p in ("gate_proj", "up_proj", "down_proj")]
    for n in names:
        f = index[n]
        by.setdefault(f if f.startswith("/") else str(Path(student) / f), []).append(n)
    got = {}
    for f, ns in by.items():
        with safe_open(f, framework="pt", device="cpu") as fh:
            for n in ns:
                got[n] = fh.get_tensor(n)
    if fused:
        gu, dn = got[pre + "gate_up_proj"], got[pre + "down_proj"]
        return {s: (gu[s], dn[s]) for s in slots}
    return {s: (torch.cat([got[f"{pre}{s}.gate_proj.weight"], got[f"{pre}{s}.up_proj.weight"]]),
                got[f"{pre}{s}.down_proj.weight"]) for s in slots}


def quantize(w, torch):
    """[n, N, K] masters -> (packed uint8, E4M3 block scales, scale_2 fp32, dequant fp32), checked bit-equal to
    qat.fake3 (the trainer's int3served forward)."""
    import sys

    sys.path[:0] = ["/opt/src", "/opt/sft"]
    import qat
    from arena import expert_3bit_math as tm
    from arena import expert_quant_math as qm

    wf = w.float()
    s2 = tm.scale2_of(wf, "int3")
    if not bool((s2 > 0).all()):
        raise RuntimeError("an all-zero expert")
    q, sc = tm.rtn(wf, s2, "int3", GROUP, SEARCH)
    packed, block = tm.to_nvfp4(q, sc, s2, GROUP)
    if not torch.equal(qm.dequantize(packed, block, s2), q):
        raise RuntimeError("packed dequant != rtn dequant")
    s2q, scq = qat.scales3_of(w)
    if not torch.equal(s2q.float(), s2.float()):
        raise RuntimeError("qat scale_2 differs")
    if not torch.equal(qat.fake3(w, (s2q, scq)), q):
        raise RuntimeError("qat.fake3 != served int3 dequant")
    return packed, block, s2, q


def grid_check(packed, torch):
    codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(*packed.shape[:-1], -1)
    mag = codes & 7
    if bool(((mag == 1) | (mag == 3) | (mag == 7)).any()):
        raise RuntimeError("values off the int3 grid")
    g = codes.reshape(*codes.shape[:-1], -1, 16)
    if bool(((g == 6).any(-1) & (g == 14).any(-1)).any()):
        raise RuntimeError("a 16-group holds both +4 and -4")


# one wave for every layer (48 at most, <= 60 per fan-out; chains run one export at a time), never preempted (1059: two
# restarts), and each layer's shards go to the bucket as soon as they are written (the driver's copy skips them)
@app.function(image=image, volumes=MOUNTS, timeout=3000, retries=0, nonpreemptible=True,
              cpu=(0.125, 8), memory=(1024, 49152), max_containers=48, secrets=[modal.Secret.from_name("nebius-s3")])
def layer_job(layer: int, student: str, mask: str, out_name: str, publish: bool = False) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(8)
    t0 = time.time()
    kept = kept_of(f"/opt/sft/prune/{mask}", layer)
    slot = {e: s for s, e in enumerate(kept)}
    base_index = json.loads((Path(MODEL_DIR) / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts."
    shards = sorted({f for k, f in base_index.items() if k.startswith(pre)})
    out_dir = Path(OUT_ROOT) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    info, err = {}, {"w13": [0.0, 0.0], "w2": [0.0, 0.0]}
    for shard in shards:
        names = [k for k, f in base_index.items() if f == shard]
        here = sorted({int(k[len(pre):].split(".")[0]) for k in names if k.startswith(pre)} & set(kept))
        tensors, new = {}, {}
        for a in range(0, len(here), CHUNK):
            part = here[a:a + CHUNK]
            ws = student_experts(student, cut_of(student).index(layer), [slot[e] for e in part], torch)
            for label, idx in (("w13", 0), ("w2", 1)):
                w = torch.stack([ws[slot[e]][idx] for e in part])
                packed, block, s2, q = quantize(w, torch)
                grid_check(packed, torch)
                err[label][0] += float((q - w.float()).pow(2).sum())
                err[label][1] += float(w.float().pow(2).sum())
                for i, e in enumerate(part):
                    new[(e, label)] = (packed[i], block[i], s2[i])
        with safe_open(str(Path(MODEL_DIR) / shard), framework="pt", device="cpu") as fh:
            metadata = fh.metadata()
            for name in names:
                original = fh.get_tensor(name)
                if not name.startswith(pre):
                    tensors[name] = original
                    continue
                expert, proj, kind = name[len(pre):].split(".")
                e = int(expert)
                if e not in slot:
                    continue  # omits_pruned
                p, b, s2 = new[(e, "w2" if proj == "down_proj" else "w13")]
                rows = {"gate_proj": slice(0, INTER), "up_proj": slice(INTER, 2 * INTER), "down_proj": slice(None)}[proj]
                if kind == "weight":
                    value = p[rows]
                elif kind == "weight_scale":
                    value = b[rows]
                elif kind == "weight_scale_2":
                    value = s2.reshape(original.shape)
                else:  # input_scale: unchanged
                    value = original
                value = value.contiguous().clone()
                if value.dtype != original.dtype:
                    value = value.view(original.dtype) if value.element_size() == original.element_size() \
                        and value.dtype in (torch.uint8, torch.float8_e4m3fn) else value.to(original.dtype)
                if value.dtype != original.dtype or tuple(value.shape) != tuple(original.shape):
                    raise RuntimeError(f"{name}: {value.dtype} {tuple(value.shape)} != {original.dtype} {tuple(original.shape)}")
                tensors[name] = value
        path = out_dir / shard
        save_file(tensors, str(path) + ".tmp", metadata=metadata)
        os.replace(str(path) + ".tmp", path)
        size = path.stat().st_size
        info[shard] = {"bytes": size, "sha256": sha256_file(path), "base_bytes": size, "tensors": len(tensors),
                       "experts": len(here)}
    assets.commit()
    if publish:
        rclone_env()
        prefix = f"flash-input/datasets/richardcsaky/{out_name}"
        for shard in info:
            subprocess.run(["rclone", "copyto", str(out_dir / shard), f"arc:{BUCKET}/assets/{prefix}/{shard}",
                            "--s3-chunk-size", "64M", "--s3-upload-concurrency", "16", "--retries", "5"], check=True)
    rel = {k: (v[0] / v[1]) ** 0.5 for k, v in err.items()}
    return {"layer": layer, "kept": len(kept), "shards": info, "rel_error_vs_bf16": rel, "seconds": round(time.time() - t0, 1)}


def rclone_env():
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")


@app.function(image=image, volumes=MOUNTS, timeout=7200, retries=0, nonpreemptible=True,
              cpu=(0.125, 16), secrets=[modal.Secret.from_name("nebius-s3")])
def driver(student: str, mask: str, out_name: str, dense: str, layers: str = "", publish: bool = True) -> dict:
    """Every layer (layer_job), then overlay-manifest.json with the dense shards merged (modal_heldout_refit
    write_manifest) and the Nebius publish (its publish: bucket assets/<prefix> + assets-manifests/<slug>.json)."""
    import shutil

    keep = cut_of(student)
    todo = [int(v) for v in layers.split(",")] if layers else keep
    t0 = time.time()
    res = list(layer_job.starmap([(l, student, mask, out_name, publish) for l in todo], return_exceptions=True))
    times = {"layers_s": round(time.time() - t0, 1)}
    errors = [f"L{l}: {type(r).__name__}: {str(r)[:300]}" for l, r in zip(todo, res) if isinstance(r, Exception)]
    if errors:
        return {"errors": errors}
    assets.reload()
    out_dir = Path(OUT_ROOT) / out_name
    shards = {}
    for r in res:
        shards.update({n: {k: v[k] for k in ("bytes", "sha256", "base_bytes", "tensors")} for n, v in r["shards"].items()})
    raw = Path(f"/opt/sft/prune/{mask}").read_bytes()
    ar = {n: sha256_file(f"/opt/src/arena/{n}") for n in ("expert_3bit_math.py", "expert_quant_math.py")}
    manifest = {"format": "arc-expert-overlay-v1", "base_model_dir": MODEL_DIR, "layers": sorted(todo), "shards": shards,
                "complete": len(shards) == 4 * len(keep) and len(todo) == len(keep), "omits_pruned": True,
                "expert_prune_sha256": hashlib.sha256(raw).hexdigest(), "mask": mask, "masters": student,
                "settings": {"grid": "int3", "group": GROUP, "method": "rtn-search", "search": list(SEARCH),
                             "source": "BF16 masters (QAD int3served grid, scripts/sft/qat.py), no W4 stage, no refit",
                             "arena_math_sha256": ar},
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "build-stats.json").write_text(json.dumps({r["layer"]: {k: r[k] for k in ("kept", "rel_error_vs_bf16",
                                                                                         "seconds")} for r in res}, indent=1))
    files = {n: {"size": v["bytes"], "sha256": v["sha256"]} for n, v in shards.items()}
    if dense:
        src = Path("/assets") / dense
        for _ in range(90):  # the dense export may run beside the layer jobs: its manifest is written last
            assets.reload()
            if (src / "overlay-manifest.json").is_file():
                break
            time.sleep(20)
        times["dense_wait_s"] = round(time.time() - t0 - times["layers_s"], 1)
        names = sorted(q.name for q in src.glob("*.safetensors"))
        if not names or any("-experts-" in n for n in names):
            raise RuntimeError(f"dense dir {src}: {names[:5]}")
        for n in names:
            shutil.copyfile(src / n, out_dir / n)
            size, digest = (out_dir / n).stat().st_size, sha256_file(out_dir / n)
            manifest["shards"][n] = {"bytes": size, "sha256": digest, "base_bytes": size}
            files[n] = {"size": size, "sha256": digest}
        manifest["dense"] = {"source": str(src), "shards": names}
    body = json.dumps(manifest, indent=1, sort_keys=True).encode()
    (out_dir / "overlay-manifest.json").write_bytes(body)
    assets.commit()
    times["manifest_s"] = round(time.time() - t0, 1)
    out = {"manifest_sha256": hashlib.sha256(body).hexdigest(), "shards": len(manifest["shards"]), "times": times,
           "rel_error_vs_bf16_mean": {k: sum(r["rel_error_vs_bf16"][k] for r in res) / len(res) for k in ("w13", "w2")},
           "experts": sum(r["kept"] for r in res)}
    if publish and manifest["complete"]:
        for q in out_dir.glob("*.json"):
            files[q.name] = {"size": q.stat().st_size, "sha256": sha256_file(q)}
        on_disk = {q.name for q in out_dir.iterdir()}
        if set(files) != on_disk:
            raise RuntimeError(f"files vs disk differ: {sorted(set(files) ^ on_disk)[:5]}")
        rclone_env()
        prefix = f"flash-input/datasets/richardcsaky/{out_name}"
        # 1059: 45 GB at ~36 MB/s (21 min) with one multipart stream per 10.7 GB dense shard: parts in parallel
        subprocess.run(["rclone", "copy", str(out_dir), f"arc:{BUCKET}/assets/{prefix}", "--size-only", "--transfers", "32",
                        "--s3-chunk-size", "64M", "--s3-upload-concurrency", "16", "--retries", "5"], check=True)
        for q in out_dir.glob("*.json"):
            subprocess.run(["rclone", "copyto", str(q), f"arc:{BUCKET}/assets/{prefix}/{q.name}"], check=True)
        slug = prefix.replace("/", "--") + ".json"
        Path(f"/tmp/{slug}").write_text(json.dumps({"prefix": prefix, "files": files}))
        subprocess.run(["rclone", "copyto", f"/tmp/{slug}", f"arc:{BUCKET}/assets-manifests/{slug}"], check=True)
        out["assets_manifest"] = slug
        out["assets_manifest_sha256"] = sha256_file(f"/tmp/{slug}")
        times["publish_s"] = round(time.time() - t0 - times["manifest_s"], 1)
    return out


@app.local_entrypoint()
def main(student: str, mask: str, out: str, dense: str = "", layers: str = "", publish: bool = True):
    print(json.dumps(driver.remote(student, mask, out, dense, layers, publish), indent=1))
