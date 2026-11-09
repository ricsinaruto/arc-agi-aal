"""CPU preflight of a merge-read bundle's weights (Modal CPU, assets volume): the bundle's own
``arena.expert_overlay.materialise`` over the base checkpoint with the bundle's manifest and mask sha (every shard
sha256-checked), then each overlay shard's safetensors header against the base shard (same dtypes and shapes; only
pruned experts missing), and per non-expert shard the tensors whose bytes differ from the base.
usage: modal run scripts/posttrain/modal_overlay_check.py --bundle DIR"""
import json
import struct
from pathlib import Path

import modal

app = modal.App("arc-rl-ovcheck")
image = modal.Image.debian_slim(python_version="3.12")
assets = modal.Volume.from_name("arc-prolong-assets")
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


@app.function(image=image, volumes={"/assets": assets}, timeout=3600)
def check(overlay_src: str, spec: dict, prune: dict, mask: dict):
    import hashlib
    import re
    import types

    mod = types.ModuleType("expert_overlay")
    exec(compile(overlay_src, "expert_overlay.py", "exec"), mod.__dict__)
    overlay_dir = Path("/assets/flash-input/datasets") / spec["dataset"]
    record = mod.materialise(MODEL_DIR, overlay_dir, Path("/tmp/farm"), spec["manifest_sha256"], check="sha256",
                             expert_prune_sha256=prune["sha256"], log=lambda *_: None)
    manifest = json.loads((overlay_dir / "overlay-manifest.json").read_text())
    pruned = {int(l): {int(e) for e in ids} for l, ids in mask["pruned"].items()}
    out = {"record": record, "shards": len(manifest["shards"]), "bad": [], "dense": {}}
    for name in sorted(manifest["shards"]):
        o_start, oh = header(overlay_dir / name)
        b_start, bh = header(Path(MODEL_DIR) / name)
        for t, meta in oh.items():
            if t not in bh or bh[t]["dtype"] != meta["dtype"] or bh[t]["shape"] != meta["shape"]:
                out["bad"].append(f"{name}:{t} {meta['dtype']}{meta['shape']} vs base {bh.get(t, {}).get('dtype')}{bh.get(t, {}).get('shape')}")
        for t in set(bh) - set(oh):
            m = re.match(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.", t)
            if not m or int(m.group(2)) not in pruned.get(int(m.group(1)), set()):
                out["bad"].append(f"{name}: {t} missing and not a pruned expert")
        if "-experts-" not in name:
            changed = []
            with open(overlay_dir / name, "rb") as fo, open(Path(MODEL_DIR) / name, "rb") as fb:
                for t, meta in oh.items():
                    a, b = meta["data_offsets"], bh[t]["data_offsets"]
                    fo.seek(o_start + a[0]); fb.seek(b_start + b[0])
                    if hashlib.sha256(fo.read(a[1] - a[0])).digest() != hashlib.sha256(fb.read(b[1] - b[0])).digest():
                        changed.append(t)
            out["dense"][name] = {"tensors": len(oh), "changed": len(changed), "sample": changed[:3]}
    out["dense_changed_total"] = sum(v["changed"] for v in out["dense"].values())
    return out


@app.local_entrypoint()
def main(bundle: str):
    b = Path(bundle)
    study = json.loads((b / "configs/comparison.json").read_text())
    server = study["server"]
    mask = json.loads((b / server["expert_prune"]["mask"]).read_text())
    assert hashlib_sha(b / server["expert_prune"]["mask"]) == server["expert_prune"]["sha256"]
    res = check.remote((b / "src/arena/expert_overlay.py").read_text(), server["expert_overlay"], server["expert_prune"], mask)
    res["bundle"] = b.name
    print(json.dumps(res, indent=1)[:6000])


def hashlib_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
