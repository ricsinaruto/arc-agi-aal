"""mmrtx KL-grid candidates (Modal CPU), all in the shipped NVFP4 container format so stock kernels serve them:
  drop   K experts/layer kept by fit mass (router rows / bias sliced) -- as prune_core;
  merge  (a1) as drop, plus each dropped expert folded into the kept expert with the most similar router row,
         weights averaged by fit mass (dequant -> average -> NVFP4 re-quant with the kept tensor's global scale);
  mixed  the top A experts stay NVFP4 (byte copies), the next B are re-quantized onto a 2-bit (or 3-bit) grid written
         as E2M1 codes with per-16 FP8 scales (memory counted at the real bits; values exact in the container).
Masks/merges use only the reference run's FIT tags (excl5 ARC-3 games + ARC-2 training tasks).

usage: modal run --detach cand_build.py --ref mmrtx-ref-v2 --spec '{"mode":"merge","k":32}' --out NAME"""
import json
import re
from pathlib import Path

import modal

import sys as _sys
_sys.path[:0] = [str(__import__('pathlib').Path(__file__).resolve().parent), '/root']
from image import IMAGE  # noqa: E402

app = modal.App("mmrtx-cand")
models = modal.Volume.from_name("mmrtx-models")
SRC = "/m/minimax-m31f"
EXPERT = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(.+)$")
ROUTER = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.(gate\.weight|e_score_correction_bias)$")
LAYERS = range(3, 60)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GRIDS = {2: ((-1.5, -0.5, 0.5, 1.5), (-2.0, -1.0, 0.0, 1.0), (-1.0, 0.0, 1.0, 2.0)),
         3: ((-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0), (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0))}


# --------------------------------------------------------------------------------------------- NVFP4 codec ----
def unpack(packed):
    """[O, I/2] uint8 -> [O, I] float32 E2M1 values (first value in the low nibble, sign in bit 3)."""
    import torch
    table = torch.tensor(E2M1)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    return torch.where(codes >= 8, -1.0, 1.0) * table[codes & 7]


def pack(vals):
    """[O, I] float32 values that are exact E2M1 numbers -> [O, I/2] uint8."""
    import torch
    table = torch.tensor(E2M1)
    mag = (vals.abs().unsqueeze(-1) - table).abs().argmin(-1)
    code = mag + 8 * torch.signbit(vals).long()
    code = code.reshape(vals.shape[0], -1, 2)
    return (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)


def dequant(t):
    """t: {'weight_packed', 'weight_scale' (fp8 [O, I/16]), 'weight_global_scale'} -> fp32 [O, I]."""
    v = unpack(t["weight_packed"])
    s = t["weight_scale"].float().repeat_interleave(16, dim=1)
    return v * s / t["weight_global_scale"].float()


def codes_with_scale(w, s, gscale):
    """E2M1 values of fp32 [O, I] at given fp8 scales [O, I/16] (round to nearest)."""
    import torch
    O, I = w.shape
    real = (s.float() / gscale).clamp_min(1e-30).unsqueeze(-1)
    x = w.reshape(O, I // 16, 16) / real
    table = torch.tensor(E2M1)
    mag = table[(x.abs().unsqueeze(-1) - table).abs().argmin(-1)]
    return (torch.sign(x) * mag).reshape(O, I)


def quant_nvfp4(w, gscale):
    """fp32 [O, I] -> (packed, fp8 scales): per 16-group the best of 5 scales (amax/6 x 1 .. 0.67), RTN."""
    import torch
    O, I = w.shape
    g = w.reshape(-1, 16)
    amax = g.abs().amax(-1).clamp_min(1e-12)
    best_err = torch.full((g.shape[0],), float("inf"))
    best_v, best_s = torch.zeros_like(g), torch.zeros(g.shape[0])  # fp8-exact values held in fp32
    table = torch.tensor(E2M1)
    for f in (1.0, 0.9, 0.8, 0.75, 2 / 3):
        s = (amax / 6.0 * f * gscale).clamp(1e-12, 448.0).to(torch.float8_e4m3fn)
        real = (s.float() / gscale).clamp_min(1e-30).unsqueeze(-1)
        x = g / real
        v = torch.sign(x) * table[(x.abs().clamp_max(6.0).unsqueeze(-1) - table).abs().argmin(-1)]
        err = ((v * real - g) ** 2).sum(-1)
        better = err < best_err
        best_v[better], best_s[better] = v[better], s.float()[better]
        best_err = torch.where(better, err, best_err)
    return pack(best_v.reshape(O, I)), best_s.to(torch.float8_e4m3fn).reshape(O, I // 16)


def quant_grid(w, gscale, bits):
    """fp32 [O, I] -> (packed, fp8 scales): each 16-group on the best of GRIDS[bits] x a searched scale, values
    written as E2M1 codes (every grid value is an E2M1 number)."""
    import torch
    O, I = w.shape
    g = w.reshape(-1, 16)
    best_err = torch.full((g.shape[0],), float("inf"))
    best_q = torch.zeros_like(g)
    best_s = torch.zeros(g.shape[0])
    amax = g.abs().amax(-1).clamp_min(1e-12)
    for grid in GRIDS[bits]:
        gt = torch.tensor(grid)
        top = max(abs(x) for x in grid)
        for f in (0.6, 0.7, 0.8, 0.9, 1.0):
            s = (amax / top * f * gscale).clamp(1e-12, 448.0).to(torch.float8_e4m3fn)
            real = (s.float() / gscale).clamp_min(1e-30)
            x = g / real.unsqueeze(-1)
            q = gt[(x.unsqueeze(-1) - gt).abs().argmin(-1)]
            err = ((q * real.unsqueeze(-1) - g) ** 2).sum(-1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q[better] = q[better]
            best_s[better] = s.float()[better]
    return pack(best_q.reshape(O, I)), best_s.to(torch.float8_e4m3fn).reshape(O, I // 16)


# ------------------------------------------------------------------------------------------------- build ----
def fit_mass(route_pt):
    import torch
    stats = torch.load(route_pt, weights_only=False)
    tags = [t for t in stats if t.split(":")[0] in ("fit3", "fit2") and stats[t][0] is not None]
    return sum(stats[t][0]["mass"] for t in tags).double()


def build(spec, route_pt, out, workers=16):
    import multiprocessing as mp
    import shutil
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    mass = fit_mass(route_pt)
    k = int(spec["k"]) if spec["mode"] in ("drop", "merge") else int(spec["a"]) + int(spec["b"])
    src, dst = Path(SRC), Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    plan = {}
    for layer in LAYERS:
        order = [int(e) for e in mass[layer].argsort(descending=True)]
        plan[layer] = {"keep": sorted(order[:k]), "order": order}
    # non-expert tensors (router rows sliced) -> one shard
    rest, weight_map = {}, {}
    by_shard = {}
    for name, shard in index.items():
        by_shard.setdefault(shard, []).append(name)
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(src / shard), framework="pt") as f:
            for name in names:
                if EXPERT.match(name):
                    continue
                t = f.get_tensor(name)
                r = ROUTER.match(name)
                if r and int(r.group(1)) in plan:
                    t = t[plan[int(r.group(1))]["keep"]].contiguous()
                rest[name] = t
    save_file(rest, str(dst / "model-rest.safetensors"))
    weight_map.update({n: "model-rest.safetensors" for n in rest})
    jobs = [(layer, plan[layer], spec, index) for layer in LAYERS]
    with mp.get_context("fork").Pool(workers) as pool:
        for layer, names in pool.imap_unordered(_layer_job, jobs):
            weight_map.update({n: f"model-layer{layer:02d}.safetensors" for n in names})
            print(f"layer {layer} done ({len(names)} tensors)", flush=True)
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    cfg = json.loads((src / "config.json").read_text())
    cfg["text_config"]["num_local_experts"] = k
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in ("model.safetensors.index.json", "config.json"):
            shutil.copy(p, dst / p.name)
        elif p.is_dir():
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
    keep_tot = torch.stack([mass[layer, plan[layer]["keep"]].sum() / mass[layer].sum() for layer in LAYERS])
    info = {"spec": spec, "k": k, "kept_mass_mean": round(float(keep_tot.mean()), 4)}
    (dst / "mmrtx-cand.json").write_text(json.dumps({"info": info, "plan": {str(a): b["keep"] for a, b in plan.items()}}))
    return info


def _expert(f_get, layer, e):
    pre = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{e}."
    return {w: {k: f_get(pre + f"{w}.{k}") for k in ("weight_packed", "weight_scale", "weight_global_scale",
                                                    "input_global_scale")} for w in ("w1", "w2", "w3")}


def _layer_job(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    layer, p, spec, index = args
    torch.set_num_threads(2)
    handles = {}

    def get(name):
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(Path(SRC) / shard), framework="pt")
        return handles[shard].get_tensor(name)

    keep, order = p["keep"], p["order"]
    mass = spec["_mass"][layer] if "_mass" in spec else None
    out = {}
    merged = {}
    if spec["mode"] == "merge":
        gate = get(f"language_model.model.layers.{layer}.block_sparse_moe.gate.weight").float()
        gn = gate / gate.norm(dim=1, keepdim=True)
        dropped = [e for e in range(gate.shape[0]) if e not in keep]
        sim = gn[dropped] @ gn[keep].T
        target = {d: keep[int(sim[i].argmax())] for i, d in enumerate(dropped)}
        for kk in keep:
            group = [kk] + [d for d in dropped if target[d] == kk]
            if len(group) > 1:
                merged[kk] = group
    m = torch.tensor(spec["mass_layer"][str(layer)]) if spec["mode"] == "merge" else None
    low = set(order[int(spec["a"]):int(spec["a"]) + int(spec["b"])]) if spec["mode"] == "mixed" else set()
    for new, e in enumerate(keep):
        ex = _expert(get, layer, e)
        pre = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{new}."
        for w, t in ex.items():
            if e in merged:
                ws = torch.tensor([float(m[g]) for g in merged[e]])
                ws = ws / ws.sum()
                acc = sum(wi * dequant(_expert(get, layer, g)[w]) for wi, g in zip(ws, merged[e]))
                packed, scale = quant_nvfp4(acc, float(t["weight_global_scale"]))
                t = {**t, "weight_packed": packed, "weight_scale": scale}
            elif e in low:
                packed, scale = quant_grid(dequant(t), float(t["weight_global_scale"]), int(spec.get("bits", 2)))
                t = {**t, "weight_packed": packed, "weight_scale": scale}
            for kname, v in t.items():
                out[pre + f"{w}.{kname}"] = v.contiguous() if hasattr(v, "contiguous") else v
    save_file(out, str(Path(spec["_dst"]) / f"model-layer{layer:02d}.safetensors"))
    return layer, list(out)


def selftest():
    """Codec identity on real tensors: unpack/pack round trip, NVFP4 re-quant of a dequantized expert gives the
    shipped codes back, grid quant error ordering (2 > 3 > 4 bits)."""
    import torch
    from safetensors import safe_open
    index = json.loads((Path(SRC) / "model.safetensors.index.json").read_text())["weight_map"]
    name = "language_model.model.layers.10.block_sparse_moe.experts.5.w1.weight_packed"
    with safe_open(str(Path(SRC) / index[name]), framework="pt") as f:
        t = {k: f.get_tensor(name.replace("weight_packed", k)) for k in ("weight_packed", "weight_scale", "weight_global_scale")}
    assert torch.equal(pack(unpack(t["weight_packed"])), t["weight_packed"]), "pack/unpack"
    w = dequant(t)
    gs = float(t["weight_global_scale"])
    same = float((codes_with_scale(w, t["weight_scale"], gs) == unpack(t["weight_packed"])).float().mean())  # values (-0 == +0)
    errs = {}
    p4, s4 = quant_nvfp4(w, gs)
    errs[4] = float(((unpack(p4) * s4.float().repeat_interleave(16, 1) / gs - w) ** 2).mean() / (w ** 2).mean())
    for bits in (3, 2):
        pb, sb = quant_grid(w, float(t["weight_global_scale"]), bits)
        wq = unpack(pb) * sb.float().repeat_interleave(16, 1) / float(t["weight_global_scale"])
        errs[bits] = float(((wq - w) ** 2).mean() / (w ** 2).mean())
    res = {"codes_at_shipped_scales_identical": round(same, 5), "rel_mse_requant4": round(errs[4], 5),
           "rel_mse_3bit": round(errs[3], 4), "rel_mse_2bit": round(errs[2], 4)}
    assert same > 0.9999 and errs[4] < 0.02 and errs[2] > errs[3] > errs[4], res
    return res


@app.function(image=IMAGE, timeout=4 * 3600, cpu=(0.125, 32), volumes={"/m": models})
def run(ref: str, spec_json: str, out: str):
    import torch
    spec = json.loads(spec_json)
    print("SELFTEST " + json.dumps(selftest()), flush=True)
    mass = fit_mass(f"/m/ref/{ref}/route.pt")
    spec["mass_layer"] = {str(layer): [float(x) for x in mass[layer]] for layer in LAYERS}
    spec["_dst"] = f"/m/{out}"
    Path(spec["_dst"]).mkdir(parents=True, exist_ok=True)
    info = build(spec, f"/m/ref/{ref}/route.pt", f"/m/{out}")
    models.commit()
    print("BUILT " + json.dumps(info), flush=True)
    return info


@app.local_entrypoint()
def main(ref: str, specs: str):
    """specs: JSON list of {"out": dir, ...spec}; one container per candidate."""
    calls = [run.spawn(ref, json.dumps(s), s["out"]).object_id for s in json.loads(specs)]
    print(json.dumps({"calls": calls, "app_id": app.app_id}))
