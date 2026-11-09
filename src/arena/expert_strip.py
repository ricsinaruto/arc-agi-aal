"""Physically drop a pruning mask's routed experts from an expert overlay (throughput arm phase 5, 5 October).

Each overlay shard is rewritten at the safetensors header level (byte ranges copied, no dtype decoding) without the
tensors ``model.language_model.layers.L.mlp.experts.E.*`` of the mask's pruned (L, E); the manifest gets the new sizes
and digests, ``omits_pruned`` and the mask's sha256, so ``arena.expert_overlay.materialise`` serves it only with that
mask. Stdlib only.
"""

import hashlib
import json
import re
import struct
from pathlib import Path

from .expert_overlay import FORMAT, MANIFEST

EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def read_header(path):
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        return size, json.loads(handle.read(size))


def strip_shard(src, dst, pruned, chunk=64 << 20):
    """Write ``src`` to ``dst`` without the pruned experts' tensors; returns (kept, dropped) tensor counts."""
    size, header = read_header(src)
    meta = header.pop("__metadata__", None)
    keep = []
    dropped = 0
    for name, spec in header.items():
        m = EXPERT.match(name)
        if m and int(m.group(2)) in pruned.get(int(m.group(1)), ()):
            dropped += 1
            continue
        keep.append((spec["data_offsets"][0], name, spec))
    keep.sort()
    out, offset = {}, 0
    for start, name, spec in keep:
        length = spec["data_offsets"][1] - start
        out[name] = {**spec, "data_offsets": [offset, offset + length]}
        offset += length
    if meta is not None:
        out = {"__metadata__": meta, **out}
    raw = json.dumps(out, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    base = 8 + size
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(raw)))
        fout.write(raw)
        for start, _name, spec in keep:
            fin.seek(base + start)
            left = spec["data_offsets"][1] - start
            while left:
                block = fin.read(min(chunk, left))
                if not block:
                    raise RuntimeError(f"{src}: short read")
                fout.write(block)
                left -= len(block)
    return len(keep), dropped


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def strip_overlay(src_dir, dst_dir, mask_path, log=print):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    mask_raw = Path(mask_path).read_bytes()
    mask = json.loads(mask_raw)
    pruned = {int(layer): {int(e) for e in ids} for layer, ids in mask["pruned"].items()}
    manifest = json.loads((src_dir / MANIFEST).read_text())
    if manifest.get("format") != FORMAT or not manifest.get("complete") or manifest.get("omits_pruned"):
        raise RuntimeError("strip needs a complete overlay that keeps every expert")
    dst_dir.mkdir(parents=True, exist_ok=True)
    shards, totals = {}, {"kept": 0, "dropped": 0, "bytes_before": 0, "bytes_after": 0}
    for name in sorted(manifest["shards"]):
        kept, dropped = strip_shard(src_dir / name, dst_dir / name, pruned)
        shards[name] = {"bytes": (dst_dir / name).stat().st_size, "sha256": sha256_file(dst_dir / name)}
        totals["kept"] += kept
        totals["dropped"] += dropped
        totals["bytes_before"] += manifest["shards"][name]["bytes"]
        totals["bytes_after"] += shards[name]["bytes"]
        log(json.dumps({"shard": name, "kept": kept, "dropped": dropped}))
    out = {**manifest, "shards": shards, "omits_pruned": True,
           "expert_prune_sha256": hashlib.sha256(mask_raw).hexdigest(),
           "stripped_from": {"manifest_sha256": hashlib.sha256((src_dir / MANIFEST).read_bytes()).hexdigest(),
                             "mask": Path(mask_path).name}}
    (dst_dir / MANIFEST).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    totals["manifest_sha256"] = sha256_file(dst_dir / MANIFEST)
    return totals
