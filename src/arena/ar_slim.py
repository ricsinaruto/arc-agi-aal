"""Slim AR farm (30 Oct: the v9b Kaggle preview spent 655 s loading weights). With a heal overlay and an expert-prune mask,
every base expert tensor is either replaced by the overlay (kept experts) or dropped by the mask, yet vLLM's eager
loader reads every byte of every indexed file: 73 GiB of AR base shards for 0.9 GiB of tensors still used.
slim() removes the pruned expert keys from the index, copies the base tensors still indexed byte for byte into one
local safetensors file, and unlinks the base shards from the farm, so the load reads overlay + PLE + that file only."""
import hashlib
import json
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
REST = "model-00000-base-rest.safetensors"  # sorts first in the farm (vLLM loads files in name order)
LIMIT = 8 << 30  # more base bytes than this still indexed means the overlay does not cover what it should


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_ranges(path, entries, base):
    out = {}
    with open(path, "rb", buffering=0) as f:
        for key, (a, b) in sorted(entries.items(), key=lambda kv: kv[1][0]):
            f.seek(base + a)
            buf = bytearray(b - a)
            view, got = memoryview(buf), 0
            while got < len(buf):
                n = f.readinto(view[got:])
                if not n:
                    raise OSError(f"{path}: short read for {key}")
                got += n
            out[key] = bytes(buf)
    return out


def slim(index, root, farm, cache_dir, mask, overlay_prefix="zz-heal-overlay-"):
    """index: the farm index (mutated); root: the AR checkpoint dir; farm: the symlink farm; cache_dir: where the rest
    file is kept across build() calls; mask: the loaded expert-prune mask ({'experts', 'pruned': {layer: [ids]}})."""
    if not mask or "pruned" not in mask:
        raise ValueError("slim_base needs server.expert_prune (the mask that drops the base experts)")
    wm = index["weight_map"]
    if not any(v.startswith(overlay_prefix) for v in wm.values()):
        raise ValueError("slim_base needs a heal overlay (it replaces the kept experts)")
    base_names = {e.name for e in Path(root).iterdir()}
    pruned = {int(k): {int(e) for e in v} for k, v in mask["pruned"].items()}
    base_keys = [k for k, f in wm.items() if f in base_names]
    dropped = 0
    for k in base_keys:
        m = EXPERT.match(k)
        if m and int(m.group(2)) in pruned.get(int(m.group(1)), ()):
            del wm[k]
            dropped += 1
    rest = {k: wm[k] for k in base_keys if k in wm}
    by_file, size, meta = {}, 0, {}
    for f in sorted(set(rest.values())):
        h, base = header(Path(root) / f)
        ent = {k: h[k]["data_offsets"] for k in rest if rest[k] == f}
        by_file[f] = (ent, base)
        for k in ent:
            meta[k] = (h[k]["dtype"], h[k]["shape"])
            size += ent[k][1] - ent[k][0]
    if size > LIMIT:
        raise ValueError(f"slim_base: {size / 2**30:.1f} GiB of base tensors still indexed (limit {LIMIT >> 30} GiB)")
    ident = hashlib.sha256(json.dumps(sorted((k, rest[k], by_file[rest[k]][0][k]) for k in rest)).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, stamp = cache_dir / REST, cache_dir / (REST + ".json")
    if not (out.is_file() and stamp.is_file() and json.loads(stamp.read_text()).get("ident") == ident):
        with ThreadPoolExecutor(8) as pool:
            parts = list(pool.map(lambda f: read_ranges(Path(root) / f, *by_file[f]), sorted(by_file)))
        blobs = {k: v for p in parts for k, v in p.items()}
        hdr, off = {"__metadata__": {"format": "pt"}}, 0
        for k in sorted(blobs):
            hdr[k] = {"dtype": meta[k][0], "shape": meta[k][1], "data_offsets": [off, off + len(blobs[k])]}
            off += len(blobs[k])
        raw = json.dumps(hdr, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        tmp = out.with_suffix(".part")
        with open(tmp, "wb") as w:
            w.write(struct.pack("<Q", len(raw)))
            w.write(raw)
            for k in sorted(blobs):
                w.write(blobs[k])
        tmp.replace(out)
        stamp.write_text(json.dumps({"ident": ident, "tensors": len(blobs), "bytes": off}))
    for k in rest:
        wm[k] = REST
    (Path(farm) / REST).symlink_to(out)
    used = set(wm.values())
    unlinked = sorted(n for n in base_names if n.endswith(".safetensors") and n not in used and (Path(farm) / n).is_symlink())
    for n in unlinked:
        (Path(farm) / n).unlink()
    return {"pruned_keys_dropped": dropped, "rest_tensors": len(rest), "rest_gib": round(size / 2**30, 3),
            "rest_sources": len(by_file), "base_shards_unlinked": len(unlinked), "ident": ident[:16]}
