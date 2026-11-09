# Expert-pruned view of a per-expert safetensors checkpoint (stdlib only): pruned experts dropped, kept experts
# renumbered 0..K-1 per layer, router rows sliced to the kept experts (equivalent to masking them to -inf),
# config num_experts = K. Files without expert or router tensors are symlinked.
import json, os, re, struct, shutil
from pathlib import Path

def arc_prune_view(src, dst, mask_path, prefix="model.language_model.layers."):
    src, dst = Path(src), Path(dst)
    mask = json.loads(Path(mask_path).read_text())
    n = int(mask["experts"]); pruned = {int(l): set(v) for l, v in mask["pruned"].items()}
    keep = {l: [e for e in range(n) if e not in p] for l, p in pruned.items()}
    k = {len(v) for v in keep.values()}; assert len(k) == 1, "uneven kept counts"
    k = k.pop(); new_id = {l: {e: i for i, e in enumerate(v)} for l, v in keep.items()}
    exp = re.compile(re.escape(prefix) + r"(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
    gate = re.compile(re.escape(prefix) + r"(\d+)\.mlp\.gate\.weight$")
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map, stats = {}, {"dropped": 0, "renamed": 0, "sliced": 0, "rewritten_files": 0}
    files = sorted(set(index["weight_map"].values()))
    for name in files:
        path = src / name
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(hlen)); base = 8 + hlen
        meta = header.pop("__metadata__", None)
        plan, touched = [], False
        for key, t in header.items():
            m, g = exp.match(key), gate.match(key)
            if m and int(m.group(1)) in pruned:
                l, e = int(m.group(1)), int(m.group(2))
                if e in pruned[l]: touched = True; stats["dropped"] += 1; continue
                plan.append((f"{prefix}{l}.mlp.experts.{new_id[l][e]}.{m.group(3)}", t, None)); touched = True
                stats["renamed"] += 1
            elif g and int(g.group(1)) in pruned:
                l = int(g.group(1)); assert t["shape"][0] == n, (key, t["shape"])
                plan.append((key, t, keep[l])); touched = True; stats["sliced"] += 1
            else:
                plan.append((key, t, None))
        if not touched:
            os.symlink(path.resolve(), dst / name)
            weight_map.update({key: name for key, _, _ in plan}); continue
        stats["rewritten_files"] += 1
        out_header, off, items = {}, 0, []
        for key, t, rows in plan:
            b0, b1 = t["data_offsets"]; size = b1 - b0
            shape = list(t["shape"])
            if rows is not None:
                row = size // shape[0]; size = row * len(rows); shape[0] = len(rows)
            out_header[key] = {"dtype": t["dtype"], "shape": shape, "data_offsets": [off, off + size]}
            items.append((b0, b1, rows, (b1 - b0) // t["shape"][0] if rows is not None else 0)); off += size
            weight_map[key] = name
        if meta is not None: out_header["__metadata__"] = meta
        hb = json.dumps(out_header, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(path, "rb") as f, open(dst / name, "wb") as o:
            o.write(struct.pack("<Q", len(hb))); o.write(hb)
            for b0, b1, rows, row in items:
                if rows is None:
                    f.seek(base + b0); left = b1 - b0
                    while left:
                        chunk = f.read(min(left, 64 << 20)); o.write(chunk); left -= len(chunk)
                else:
                    for r in rows:
                        f.seek(base + b0 + r * row); o.write(f.read(row))
    for p in src.iterdir():
        if p.name in files or p.name in ("model.safetensors.index.json", "config.json"): continue
        os.symlink(p.resolve(), dst / p.name)
    (dst / "model.safetensors.index.json").write_text(json.dumps({**index, "weight_map": weight_map}, indent=1))
    cfg = json.loads((src / "config.json").read_text())
    for c in (cfg, cfg.get("text_config") or {}):
        if "num_experts" in c: c["num_experts"] = k
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    stats.update(kept=k, experts=n); return stats
