"""Layers-only throughput-bench cuts of p75 c613 (greedy sets on c1028, sft-layers-p75c1028-v2): SFT's
cutovr.py + pubcut.py with all 128 experts kept; tensors byte-copied from the c613 overlay, published to the bucket.
    modal run --detach cutLayers.py --mask MASK.json"""
import hashlib
import json
import os
import re
import struct
import subprocess
from pathlib import Path

import modal

app = modal.App("sft-cut-overlay")
assets = modal.Volume.from_name("arc-prolong-assets")
image = modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"
SETS = {"d8": [10, 13, 16, 20, 21, 22, 23, 44],
        "d16": [5, 9, 10, 11, 12, 13, 16, 18, 19, 20, 21, 22, 23, 29, 39, 44]}
SRC = Path("/assets/flash-input/models/arc-heal/p75-qad-c613")
EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
GATE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def subset(src, dst, keep):
    """safetensors src -> dst with the tensors ``keep`` selects, payload bytes unchanged; returns (tensors, bytes, sha256)."""
    head, base = header(src)
    names = [k for k in head if k != "__metadata__" and keep(k)]
    names.sort(key=lambda k: head[k]["data_offsets"][0])
    out, at = {}, 0
    for k in names:
        a, b = head[k]["data_offsets"]
        out[k] = {**head[k], "data_offsets": [at, at + b - a]}
        at += b - a
    if "__metadata__" in head:
        out["__metadata__"] = head["__metadata__"]
    body = json.dumps(out, separators=(",", ":")).encode()
    body += b" " * (-(len(body) + 8) % 8)
    h = hashlib.sha256()
    with open(src, "rb") as f, open(str(dst) + ".part", "wb") as g:
        first = struct.pack("<Q", len(body)) + body
        g.write(first)
        h.update(first)
        for k in names:
            a, b = head[k]["data_offsets"]
            f.seek(base + a)
            left = b - a
            while left:
                chunk = f.read(min(left, 1 << 24))
                g.write(chunk)
                h.update(chunk)
                left -= len(chunk)
    Path(str(dst) + ".part").rename(dst)
    return len(names), len(first) + at, h.hexdigest()


def check(src, dst, sample=400):
    """Every copied tensor's header entry, and ``sample`` tensors' bytes, against the source."""
    a, ba = header(src)
    b, bb = header(dst)
    keys = [k for k in b if k != "__metadata__"]
    step = max(1, len(keys) // sample)
    with open(src, "rb") as f, open(dst, "rb") as g:
        for k in keys:
            if (a[k]["dtype"], a[k]["shape"]) != (b[k]["dtype"], b[k]["shape"]):
                raise ValueError(k)
        for k in keys[::step]:
            f.seek(ba + a[k]["data_offsets"][0])
            g.seek(bb + b[k]["data_offsets"][0])
            n = a[k]["data_offsets"][1] - a[k]["data_offsets"][0]
            if f.read(n) != g.read(n):
                raise ValueError(f"bytes differ: {k}")
    return len(keys)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def publish(prefix, src, meta):
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION, RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")
    files = {}
    for q in sorted(src.iterdir()):
        digest = sha(q)
        if q.name in meta["files"] and digest != meta["files"][q.name]["sha256"]:
            raise ValueError(f"{q.name}: on the volume {digest}, in overlay.json {meta['files'][q.name]['sha256']}")
        files[q.name] = {"size": q.stat().st_size, "sha256": digest}
    subprocess.run(["rclone", "copy", str(src), f"arc:{BUCKET}/assets/{prefix}", "--size-only", "--transfers", "2",
                    "--s3-chunk-size", "64M", "--s3-upload-concurrency", "4", "--retries", "5"], check=True)
    slug = prefix.replace("/", "--") + ".json"
    Path(f"/tmp/{slug}").write_text(json.dumps({"prefix": prefix, "files": files}))
    subprocess.run(["rclone", "copyto", f"/tmp/{slug}", f"arc:{BUCKET}/assets-manifests/{slug}"], check=True)
    return slug


@app.function(image=image, timeout=3000, retries=0, cpu=(0.125, 4), max_containers=1, volumes={"/assets": assets},
              secrets=[modal.Secret.from_name("nebius-s3")])
def cut(name: str, drop: list, mask: dict) -> dict:
    prefix = f"flash-input/models/arc-heal/p75-{name}-c613"
    DST = Path("/assets") / prefix
    meta = json.loads((SRC / "overlay.json").read_text())
    if meta["format"] != "arc-heal-overlay-v1" or meta["fmt"] != "autoround" or meta["drop_layers"]:
        raise ValueError("unexpected source overlay")
    gone = set(drop)
    kept = {int(k): set(range(mask["experts"])) - set(v) for k, v in mask["pruned"].items()}

    def keep(name):
        m = EXPERT.match(name)
        if m:
            return int(m.group(1)) not in gone and int(m.group(2)) in kept[int(m.group(1))]
        m = LAYER.match(name)
        return not (m and int(m.group(1)) in gone)

    DST.mkdir(parents=True, exist_ok=True)
    weight_map = {k: v for k, v in meta["weight_map"].items() if keep(k)}
    have = {}
    for k in weight_map:
        m = EXPERT.match(k)
        if m:
            have.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    for layer in range(48):
        want = set() if layer in gone else kept[layer]
        if have.get(layer, set()) != want:
            raise ValueError(f"layer {layer}: {len(have.get(layer, ()))} experts in the overlay, {len(want)} wanted")
    files = {}
    for file in sorted(set(meta["weight_map"].values())):
        n, size, sha = subset(SRC / file, DST / file, keep)
        files[file] = {"tensors": n, "bytes": size, "sha256": sha, "checked": check(SRC / file, DST / file)}
    if sum(v["tensors"] for v in files.values()) != len(weight_map):
        raise ValueError("shard tensors differ from the weight map")
    counts = {"experts": sum(1 for k in weight_map if EXPERT.match(k) and k.endswith(".qweight")),
              "expert_tensors": sum(1 for k in weight_map if EXPERT.match(k)),
              "router_rows_tensors": sum(1 for k in weight_map if GATE.match(k)), "tensors": len(weight_map)}
    cut_mask = {**mask, "drop_layers": sorted(drop), "derived": mask["derived"] + f"; layers-only bench cut {name}, 31 Oct"}
    body = json.dumps(cut_mask).encode()
    (DST / f"expert-prune-t75-{name}.json").write_bytes(body)
    new = {**meta, "source_overlay": str(SRC), "weight_map": weight_map, "counts": counts, "drop_layers": sorted(drop),
           "files": files, "expert_mask": f"expert-prune-t75-{name}.json"}
    raw = json.dumps(new, indent=1).encode()
    (DST / "overlay.json").write_bytes(raw)
    assets.commit()
    slug = publish(prefix, DST, new)
    return {"path": str(DST), "assets_manifest": slug, "mask": cut_mask, "overlay_sha256": hashlib.sha256(raw).hexdigest(), "mask_sha256": hashlib.sha256(body).hexdigest(), "counts": counts,
            "bytes": sum(v["bytes"] for v in files.values()), "src_bytes": sum((SRC / f).stat().st_size for f in files), "files": len(files)}


@app.local_entrypoint()
def main(mask: str):
    m = json.loads(Path(mask).read_text())
    assert m["pruned_per_layer"] == 384 and not m.get("drop_layers")
    out = {}
    for name, r in zip(SETS, cut.starmap([(n, d, m) for n, d in SETS.items()])):
        Path(f"expert-prune-t75-{name}.json").write_text(json.dumps(r.pop("mask")))
        out[name] = r
    print(json.dumps(out))
