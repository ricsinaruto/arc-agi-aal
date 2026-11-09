#!/usr/bin/env python3
"""Phase 3, CPU: GPTQ variants on real layers (29 September; docs/validation-20260927/expert-quant.md §6).

Inputs (``runs/expert-quant/p3``, fetched from HF and the Modal volumes): the BF16 experts of a layer, the served NVFP4
experts, the stage-2 GPTQ-ARC (v1) replacement shards, the stage-1 calibration capture (ka59, lf52, m0r0; 21,914
rows per layer, 4.5x fewer than stage 2's 97,749) and the stage-2 held-out capture (cn04, sk48, wa30; 9,063 rows).

Every method is scored as the served W4A16 path computes (dequantized weights, BF16 activations, float32; Marlin
agrees within 0.1 point): the routed-expert MoE output error against the true BF16 experts on the held-out rows,
overall and on the rows' share routed to "tail" experts (those v1 left on checkpoint weights).

    <torch venv>/bin/python scripts/expert_quant_p3_cpu.py --layer 20 --out runs/expert-quant/p3/L20.json
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arena import expert_quant_bench as B  # noqa: E402
from arena import expert_quant_math as qm  # noqa: E402

P3 = Path("./runs/expert-quant/p3")


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-rows", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--methods", default="all")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    ns = types.SimpleNamespace(model_dir=str(P3 / "ckpt"), bf16_dir=str(P3 / "bf16"), deadline_seconds=1e9,
                               out=args.out + ".bench", device="cpu", capture=str(P3 / "cap" / "s1"))
    bench = B.Bench(ns)
    layer = args.layer
    t0 = time.time()
    served = bench.load_served(layer)
    gate_up, down, _ = bench.load_bf16(layer)
    bench.args.model_dir = str(P3 / "v1")
    bench.index = {k: v for k, v in bench.index.items()}  # the v1 overlay has the checkpoint's shard names
    v1 = bench.load_served(layer)
    bench.args.model_dir = str(P3 / "ckpt")
    xc, idc, wc = bench.load_capture("calib", layer)
    bench.args.capture = str(P3 / "cap")
    xe, ide, we = load_eval(torch, P3 / "cap" / "s2eval", layer)
    n = min(args.eval_rows, xe.shape[0])
    pick = torch.linspace(0, xe.shape[0] - 1, n).round().long()
    xe, ide, we = xe[pick].contiguous(), ide[pick].contiguous(), we[pick].contiguous()
    print(f"loaded in {time.time() - t0:.0f} s; calib {xc.shape[0]} rows, eval {n}", flush=True)
    s13, s2d = served["s2_gate"].float(), served["s2_down"].float()
    fallback = [e for e in range(B.E) if torch.equal(v1["w13"][e], served["w13"][e])]
    results = {"layer": layer, "calib_rows": int(xc.shape[0]), "eval_rows": int(n), "v1_fallback": len(fallback),
               "methods": {}}
    token, weight, bounds = qm.routed_rows(ide, we, range(B.E), B.E)
    eval_mass = torch.zeros(B.E)
    for e in range(B.E):
        eval_mass[e] = weight[bounds[e]:bounds[e + 1]].sum()
    results["v1_fallback_eval_mass"] = float(eval_mass[fallback].sum() / eval_mass.sum())
    ref, _ = bench.moe_reference(xe, ide, we, lambda e: (gate_up[e].float(), down[e].float()))
    tail = torch.zeros(B.E, dtype=torch.bool)
    tail[fallback] = True

    def score(name, weights, extra=None):
        deq = bench.dequant_expert(weights)
        y, _ = bench.moe_reference(xe, ide, we, deq)
        y_tail, _ = bench.moe_reference(xe, ide, we * tail[ide.long()].float(), deq)
        r_tail, _ = bench.moe_reference(xe, ide, we * tail[ide.long()].float(),
                                        lambda e: (gate_up[e].float(), down[e].float()))
        row = {"vs_bf16": qm.relative_error(y, ref), "cos": qm.cosine(y, ref),
               "tail_part_vs_bf16": qm.relative_error(y_tail, r_tail) if float(r_tail.norm()) > 0 else None,
               **(extra or {})}
        results["methods"][name] = row
        print(name, json.dumps(row), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1))

    score("checkpoint_rtn", served)
    score("v1_stage2_calib", v1)
    counts_rows = None

    def run(name, *, prior=0, damp=0.01, act_order=False, static=False, min_rows=64, fc2=True):
        nonlocal counts_rows
        t = time.time()
        h1, c1 = qm.hessians(xc, idc, wc, range(B.E), B.E)
        counts_rows = c1
        if prior:
            qm.pool_hessians(h1, c1, prior)
        keep = c1 >= min_rows if not prior else torch.ones(B.E, dtype=torch.bool)
        p13, f13 = quantize(torch, gate_up, s13, h1, c1, damp, act_order, static, args.batch)
        del h1
        p13[~keep], f13[~keep] = served["w13"][~keep], served["w13_sf"][~keep]
        new = {**served, "w13": p13, "w13_sf": f13}
        if fc2:
            gu = lambda e: qm.dequantize(p13[e:e + 1], f13[e:e + 1], s13[e:e + 1])[0]  # noqa: E731
            h2, c2 = fc2_hessians(torch, xc, idc, wc, gu)
            if prior:
                qm.pool_hessians(h2, c2, prior)
            p2, f2 = quantize(torch, down, s2d, h2, c2, damp, act_order, static, B.E)
            p2[~keep], f2[~keep] = served["w2"][~keep], served["w2_sf"][~keep]
            new.update(w2=p2, w2_sf=f2)
        score(name, new, {"seconds": round(time.time() - t, 1), "gptq_experts": int(keep.sum()),
                          "settings": {"prior": prior, "damp": damp, "act_order": act_order, "static": static,
                                       "min_rows": min_rows}})

    methods = {
        "v1_recipe_local": dict(),
        "pooled256": dict(prior=256),
        "static_pooled256": dict(prior=256, static=True),
        "actorder_pooled256": dict(prior=256, static=True, act_order=True),
        "actorder_pooled256_damp03": dict(prior=256, static=True, act_order=True, damp=0.03),
        "actorder_pooled1024": dict(prior=1024, static=True, act_order=True),
        "actorder_pooled64": dict(prior=64, static=True, act_order=True),
        "actorder_pooled1024_damp03": dict(prior=1024, static=True, act_order=True, damp=0.03),
        "actorder_pooled4096": dict(prior=4096, static=True, act_order=True),
        "actorder_pooled16384": dict(prior=16384, static=True, act_order=True),
        "actorder_shared": dict(prior=1e7, static=True, act_order=True),
    }
    chosen = methods if args.methods == "all" else {k: methods[k] for k in args.methods.split(",")}
    for name, kw in chosen.items():
        run(name, **kw)
    results["calib_rows_per_expert"] = {"median": float(counts_rows.median()), "lt64": int((counts_rows < 64).sum())}
    Path(args.out).write_text(json.dumps(results, indent=1))


def load_eval(torch, root, layer):
    import numpy as np

    meta = json.load(open(root / "meta.json"))
    rows = int(meta["rows"][str(layer)])
    x = np.fromfile(root / f"L{layer:02d}.x", dtype=np.int16).reshape(rows, B.H)
    ids = np.fromfile(root / f"L{layer:02d}.ids", dtype=np.int16).reshape(rows, B.TOPK)
    w = np.fromfile(root / f"L{layer:02d}.w", dtype=np.float32).reshape(rows, B.TOPK)
    return torch.from_numpy(x).view(torch.bfloat16), torch.from_numpy(ids.astype(np.int32)), torch.from_numpy(w)


def fc2_hessians(torch, x, ids, wts, gate_up_deq):
    token, weight, bounds = qm.routed_rows(ids, wts, range(B.E), B.E)
    out = torch.zeros((B.E, B.INTER, B.INTER))
    counts = torch.zeros(B.E)
    for e in range(B.E):
        a, b = bounds[e], bounds[e + 1]
        if a == b:
            continue
        y = x[token[a:b]].float() @ gate_up_deq(e).t()
        h = torch.nn.functional.silu(y[:, :B.INTER]) * y[:, B.INTER:]
        rows = h * weight[a:b].unsqueeze(1)
        out[e] = rows.t() @ rows
        counts[e] = float(b - a)
    return out, counts


def quantize(torch, weight, scale_2, hessian, counts, damp, act_order, static, batch):
    packs, scales = [], []
    for s in range(0, weight.shape[0], batch):
        h = hessian[s:s + batch].clone()
        qm.damp_hessian(h, counts[s:s + batch], percdamp=damp)
        if static:
            p, sc = qm.gptq_static(weight[s:s + batch], h, scale_2[s:s + batch], act_order=act_order)
        else:
            p, sc = qm.gptq(weight[s:s + batch], qm.inverse_cholesky_upper(h), scale_2[s:s + batch])
        packs.append(p)
        scales.append(sc)
    return torch.cat(packs), torch.cat(scales)


if __name__ == "__main__":
    main()
