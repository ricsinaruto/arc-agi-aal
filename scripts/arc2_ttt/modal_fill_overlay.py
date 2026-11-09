"""Modal CPU: the arc2 k256 INT3 expert overlay as complete shards for the TTT trainer (HF loading needs every
expert tensor; the served overlay omits the pruned experts). Each output shard is the base shard's header verbatim
with the overlay's bytes for every tensor it has and the base's for the pruned experts (masked at the router, never
used), so it is byte-for-byte the base layout and ``arena.expert_overlay.materialise`` takes it as a plain overlay.
Checks: overlay tensors match the base's dtype/shape, the missing ones are exactly the mask's pruned experts.
usage: modal run --detach scripts/arc2_ttt/modal_fill_overlay.py"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[min(2, len(Path(__file__).resolve().parents) - 1)]  # "/" in the container
MASK = ROOT / "dist/arc2-rtx-c240-hx-base18-v3b/configs/expert-prune/prune-arc2-k256-v1.json"
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
SRC = "/assets/flash-input/datasets/richardcsaky/arc-expert-int3-refit-arc2c240-k256-v1"
SRC_SHA = "4a140457bb230010cff7092d683e7887de56510a08249a5dd6b466d5dd035305"
DEST = "/assets/flash-input/datasets/richardcsaky/arc2-ttt-k256-int3-full-v1"
app = modal.App("arc2-ttt-fill")
assets = modal.Volume.from_name("arc-prolong-assets")
image = modal.Image.debian_slim(python_version="3.12").add_local_file(str(MASK), "/w/mask.json")


def header(path):
    import struct

    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        raw = fh.read(n)
    h = json.loads(raw)
    h.pop("__metadata__", None)
    return 8 + n, h


@app.function(image=image, volumes={"/assets": assets}, cpu=(2, 8), timeout=3 * 3600, retries=0)
def fill():
    import hashlib
    import re
    from concurrent.futures import ThreadPoolExecutor

    raw = (Path(SRC) / "overlay-manifest.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "overlay manifest sha differs"
    manifest = json.loads(raw)
    mask = json.loads(Path("/w/mask.json").read_bytes())
    mask_sha = hashlib.sha256(Path("/w/mask.json").read_bytes()).hexdigest()
    assert mask_sha == manifest["expert_prune_sha256"], "mask is not the overlay's"
    pruned = {int(k): {int(e) for e in v} for k, v in mask["pruned"].items()}
    dest = Path(DEST)
    dest.mkdir(parents=True, exist_ok=True)

    def one(name):
        o_start, oh = header(Path(SRC) / name)
        b_start, bh = header(Path(MODEL_DIR) / name)
        bad, from_base = [], 0
        for t, meta in oh.items():
            if t not in bh or bh[t]["dtype"] != meta["dtype"] or bh[t]["shape"] != meta["shape"]:
                bad.append(t)
        for t in set(bh) - set(oh):
            m = re.match(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.", t)
            if not m or int(m.group(2)) not in pruned.get(int(m.group(1)), set()):
                bad.append(f"missing {t}")
            from_base += 1
        if bad:
            return name, {"bad": bad[:5]}
        sha, out = hashlib.sha256(), dest / (name + ".tmp")
        with open(Path(MODEL_DIR) / name, "rb") as fb, open(Path(SRC) / name, "rb") as fo, open(out, "wb") as fw:
            head = fb.read(b_start)
            fw.write(head)
            sha.update(head)
            for t, meta in sorted(bh.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                if t in oh:
                    a = oh[t]["data_offsets"]
                    fo.seek(o_start + a[0])
                    data = fo.read(a[1] - a[0])
                else:
                    a = meta["data_offsets"]
                    fb.seek(b_start + a[0])
                    data = fb.read(a[1] - a[0])
                fw.write(data)
                sha.update(data)
        size = out.stat().st_size
        if size != (Path(MODEL_DIR) / name).stat().st_size:
            return name, {"bad": [f"size {size}"]}
        out.rename(dest / name)
        return name, {"bytes": size, "sha256": sha.hexdigest(), "from_base": from_base, "from_overlay": len(oh)}

    with ThreadPoolExecutor(8) as pool:
        rows = dict(pool.map(one, sorted(manifest["shards"])))
    bad = {k: v for k, v in rows.items() if "bad" in v}
    if bad:
        print("FILL_RESULT " + json.dumps({"pass": False, "bad": bad}), flush=True)
        return {"pass": False, "bad": bad}
    out = {k: v for k, v in manifest.items() if k != "shards"}
    out.update(omits_pruned=False, shards={k: {"bytes": v["bytes"], "sha256": v["sha256"]} for k, v in rows.items()},
               filled={"overlay": SRC, "overlay_manifest_sha256": SRC_SHA, "base": MODEL_DIR,
                       "pruned_tensors_from_base": sum(v["from_base"] for v in rows.values())})
    text = json.dumps(out, indent=1, sort_keys=True).encode()
    (dest / "overlay-manifest.json").write_bytes(text)
    assets.commit()
    result = {"pass": True, "dest": DEST, "manifest_sha256": hashlib.sha256(text).hexdigest(), "shards": len(rows),
              "bytes": sum(v["bytes"] for v in rows.values()), "from_base": out["filled"]["pruned_tensors_from_base"]}
    print("FILL_RESULT " + json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(fill.remote()))
