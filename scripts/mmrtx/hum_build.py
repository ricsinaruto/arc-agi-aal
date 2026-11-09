"""Humming-native low-bit expert candidate, built on the serving GPU at boot (Humming's packer is a CUDA kernel;
mmrtx has no Modal GPU allowance, so this runs inside the Nebius RTX job before scoring/serving).

python3 hum_build.py SRC DST SCHEMA_JSON [--test]
  SRC  an NVFP4 candidate dir (e.g. /w/mmrtx-models/cand-drop-k38: original NVFP4 byte copies of the kept experts)
  DST  output dir on local disk; non-expert files are symlinked, each layer's experts written as
       experts.<e>.w{1,2,3}.{weight,weight_scale} in the Humming schema (searched per-group bf16 scale)
Served with MMRTX_MOE_HUMMING=<SCHEMA_JSON> (mmrtx_patch). --test: per-tensor checks on 3 experts, nothing written.

python3 hum_build.py variant SRC HUMDIR DST LAYERS
  mixed model: experts of the layers in LAYERS (comma list) from HUMDIR (a full build), every other layer's experts
  from SRC (NVFP4) -- all symlinks; serve with MMRTX_MOE_HUMMING_LAYERS=LAYERS beside MMRTX_MOE_HUMMING."""
import json
import re
import sys
import time
from pathlib import Path

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EXPERT = re.compile(r"^(.*\.block_sparse_moe\.experts\.\d+\.w[123])\.weight_packed$")


def dequant(t, dev):
    """compressed-tensors NVFP4 {'weight_packed','weight_scale','weight_global_scale'} -> fp32 [O, I] on dev."""
    import torch
    p = t["weight_packed"].to(dev)
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(p.shape[0], -1).long()
    table = torch.tensor(E2M1 + tuple(-x for x in E2M1), device=dev)
    return table[codes] * t["weight_scale"].to(dev).float().repeat_interleave(16, 1) / t["weight_global_scale"].to(dev).float()


def quant(sch, w, search=True):
    """Humming quant_tensor layout; with search, each group's bf16 scale minimizes squared error over levels
    code - 2^(b-1) (no-zero-point uint schema: code 0 = most negative), packed with Humming's own pack op.
    Validated 25 Oct (L4): pack round trip exact, Humming's dequant of these codes == level * scale exactly."""
    import torch
    from humming import ops
    q = sch.quant_tensor(w.to(torch.bfloat16), sch, torch.bfloat16)
    q = {k: v for k, v in q.items() if v is not None and v.nelement() > 0}
    if not search:
        return q
    bits, gs = sch.b_dtype.num_bits, sch.weight_scale_group_size
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    N, K = w.shape
    g = w.float().reshape(N, K // gs, gs)
    pos, neg = g.amax(-1, keepdim=True).clamp_min(0), (-g).amax(-1, keepdim=True).clamp_min(0)
    cands = [torch.maximum(pos / hi, neg / -lo)] + [torch.maximum(pos, neg) / t * f
                                                   for t in (hi, -lo) for f in (0.7, 0.8, 0.9, 1.0)]
    best_e = best_s = best_q = None
    for c in cands:
        sc = c.clamp_min(1e-12).to(torch.bfloat16).float()
        qq = (g / sc).round().clamp(lo, hi)
        e = ((qq * sc - g) ** 2).sum(-1, keepdim=True)
        if best_e is None:
            best_e, best_s, best_q = e, sc, qq
        else:
            m = e < best_e
            best_e, best_s, best_q = torch.where(m, e, best_e), torch.where(m, sc, best_s), torch.where(m, qq, best_q)
    codes = (best_q - lo).to(torch.int32).reshape(N, K).contiguous()
    out = {"weight": ops.pack_weight(codes, bits), "weight_scale": best_s.reshape(N, K // gs).to(torch.bfloat16)}
    assert out["weight"].shape == q["weight"].shape and out["weight_scale"].shape == q["weight_scale"].shape
    return out


def rel_mse(sch, q, w):
    dq = sch.dequant_tensors({k: v.unsqueeze(0) for k, v in q.items()}).float().reshape(w.shape)
    return float(((dq - w) ** 2).sum() / (w ** 2).sum())


def main(src, dst, schema, test=False):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from humming.schema.base import BaseWeightSchema
    sch = BaseWeightSchema.from_config(json.loads(schema))
    src, dst = Path(src), Path(dst)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    bases = sorted({m.group(1) for n in index if (m := EXPERT.match(n))})
    layer_of = lambda b: int(b.split(".layers.")[1].split(".")[0])
    handles = {}

    def get(name):
        if index[name] not in handles:
            handles[index[name]] = safe_open(str(src / index[name]), framework="pt")
        return handles[index[name]].get_tensor(name)

    def load(b):
        return {k: get(f"{b}.{k}") for k in ("weight_packed", "weight_scale", "weight_global_scale")}

    if test:
        res = {}
        for b in (bases[0], bases[len(bases) // 2], bases[-1]):
            x = dequant(load(b), "cuda")
            q0, q = quant(sch, x, search=False), quant(sch, x)
            res[b.split("layers.")[1]] = {"rel_humming": round(rel_mse(sch, q0, x), 5), "rel_search": round(rel_mse(sch, q, x), 5)}
        print(json.dumps(res), flush=True)
        return res
    dst.mkdir(parents=True, exist_ok=True)
    weight_map = {n: f for n, f in index.items() if not EXPERT.match(n) and ".block_sparse_moe.experts." not in n}
    for f in set(weight_map.values()):
        (dst / f).unlink(missing_ok=True)
        (dst / f).symlink_to(src / f)
    for p in src.iterdir():
        if p.name not in set(index.values()) and p.name != "model.safetensors.index.json" and not (dst / p.name).exists():
            (dst / p.name).symlink_to(p)
    t0, rels, by_layer = time.time(), [], {}
    for b in bases:
        by_layer.setdefault(layer_of(b), []).append(b)
    for layer, bs in sorted(by_layer.items()):
        tensors = {}
        for i, b in enumerate(bs):
            x = dequant(load(b), "cuda")
            q = quant(sch, x)
            if i % 24 == 0:
                rels.append(rel_mse(sch, q, x))
            tensors.update({f"{b}.{k}": v.cpu().contiguous() for k, v in q.items()})
        fn = f"model-hum-layer{layer:02d}.safetensors"
        save_file(tensors, str(dst / fn))
        weight_map.update({n: fn for n in tensors})
        print(f"hum_build layer {layer} {len(bs)} tensors-in {time.time() - t0:.0f}s", flush=True)
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    info = {"src": str(src), "schema": json.loads(schema), "rel_mse_mean": round(sum(rels) / len(rels), 5),
            "expert_bytes": sum((dst / f"model-hum-layer{l:02d}.safetensors").stat().st_size for l in by_layer),
            "seconds": round(time.time() - t0)}
    (dst / "mmrtx-hum.json").write_text(json.dumps(info))
    print("HUM_BUILT " + json.dumps(info), flush=True)
    return info


def variant(src, hum, dst, layers):
    """Symlinked mixed model: Humming experts for `layers`, NVFP4 experts elsewhere, one shared index."""
    src, hum, dst = Path(src), Path(hum), Path(dst)
    want = {int(x) for x in str(layers).split(",") if str(x).strip()}
    si = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    hi = json.loads((hum / "model.safetensors.index.json").read_text())["weight_map"]
    layer = lambda n: int(n.split(".layers.")[1].split(".")[0]) if ".block_sparse_moe.experts." in n else None  # noqa: E731
    wm = {n: f for n, f in hi.items() if layer(n) is None or layer(n) in want}
    wm.update({n: f for n, f in si.items() if layer(n) is not None and layer(n) not in want})
    dst.mkdir(parents=True, exist_ok=True)
    for f in set(wm.values()):
        origin = hum / f if f.startswith("model-hum-") else src / f
        (dst / f).unlink(missing_ok=True)
        (dst / f).symlink_to(origin.resolve())
    for p in src.iterdir():
        if not p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json" and not (dst / p.name).exists():
            (dst / p.name).symlink_to(p)
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}, indent=1))
    return {"layers": sorted(want), "files": len(set(wm.values()))}


if __name__ == "__main__":
    if sys.argv[1] == "variant":
        print(json.dumps(variant(*sys.argv[2:6])), flush=True)
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], test="--test" in sys.argv)
