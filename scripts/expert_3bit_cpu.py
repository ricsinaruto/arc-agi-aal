#!/usr/bin/env python3
"""3-bit routed experts, CPU quality screen on real layers (30 September; docs/validation-20260930/expert-3bit.md).

Same inputs and scoring as the phase-3 GPTQ study (``scripts/expert_quant_p3_cpu.py``; ``runs/expert-quant/p3`` in the
main checkout): BF16 experts of a layer, the stage-1 calibration capture (ka59, lf52, m0r0; 21,914 rows) and the
stage-2 held-out capture (cn04, sk48, wa30; 4,096 rows used). Every method is scored as the W4A16 / W3A16 path
computes (dequantized weights, BF16 activations, float32): the routed-expert MoE output error against the true BF16
experts on the held-out rows.

Methods: ``<fmt>_g<group>_<rtn|search|gptq>``; ``gptq`` = the phase-3 optimum (pooled-Hessian prior 4,096 rows,
act-order, static searched scales, 1% damping), FC2 Hessians from the new FC1 weights.

    python scripts/expert_3bit_cpu.py --layer 20 --out runs/expert-3bit/L20.json
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arena import expert_3bit_math as tm  # noqa: E402
from arena import expert_quant_bench as B  # noqa: E402
from arena import expert_quant_math as qm  # noqa: E402

P3 = Path("./runs/expert-quant/p3")
SEARCH = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)


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
    out = torch.zeros((B.E, B.INTER, B.INTER), device=x.device)
    counts = torch.zeros(B.E, device=x.device)
    for e in range(B.E):
        a, b = bounds[e], bounds[e + 1]
        if a == b:
            continue
        y = x[token[a:b]].float() @ gate_up_deq[e].t()
        h = torch.nn.functional.silu(y[:, :B.INTER]) * y[:, B.INTER:]
        rows = h * weight[a:b].unsqueeze(1)
        out[e] = rows.t() @ rows
        counts[e] = float(b - a)
    return out, counts


def quantize(torch, weight, hessian, counts, fmt, group, method, batch, prior):
    s2 = tm.scale2_of(weight, fmt)
    outs = []
    for s in range(0, weight.shape[0], batch):
        w = weight[s:s + batch].float()
        mult = (1.0,) if method == "rtn" else SEARCH  # signround starts from the searched scales
        deq, scales = tm.rtn(w, s2[s:s + batch], fmt, group, mult)
        if method == "gptq":
            h = hessian[s:s + batch].clone()
            qm.damp_hessian(h, counts[s:s + batch], percdamp=0.01)
            deq = tm.gptq_static(w, h, s2[s:s + batch], fmt, group, scales, act_order=True)
            del h
        elif method == "requant":
            # the production recipe (6 Oct question): GPTQ-ARC onto NVFP4 (W4A16) from BF16, then RTN-search to int3
            h = hessian[s:s + batch].clone()
            qm.damp_hessian(h, counts[s:s + batch], percdamp=0.01)
            fp4_s2 = qm.scale2_of(w)
            packed, fp4_scales = qm.gptq_static(w, h, fp4_s2, act_order=True)
            del h
            w4 = qm.dequantize(packed, fp4_scales, fp4_s2)
            deq, _ = tm.rtn(w4, tm.scale2_of(w4, fmt), fmt, group, SEARCH)
        elif method.startswith("signround"):
            iters = int(method[len("signround"):] or 200)
            deq, _ = tm.signround(w, hessian[s:s + batch], s2[s:s + batch], fmt, group, scales, iters=iters)
        outs.append(deq)
    return torch.cat(outs)


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-rows", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prior", type=float, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16-dir", default=str(P3 / "bf16"))
    ap.add_argument("--model-dir", default=str(P3 / "ckpt"))
    ap.add_argument("--v1-dir", default=str(P3 / "v1"))
    ap.add_argument("--calib-root", default=str(P3 / "cap" / "s1"), help="holds calib/ (stage-1 capture)")
    ap.add_argument("--eval-dir", default=str(P3 / "cap" / "s2eval"))
    ap.add_argument("--methods", default="int3_g16_rtn,int3_g16_search,int3_g32_search,int3_g64_search,"
                    "fp3_g16_search,int3_g16_gptq,int3_g32_gptq,fp3_g16_gptq,int3_g64_gptq")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    ns = types.SimpleNamespace(model_dir=args.model_dir, bf16_dir=args.bf16_dir, deadline_seconds=1e9,
                               out=args.out + ".bench", device=args.device, capture=args.calib_root)
    bench = B.Bench(ns)
    layer = args.layer
    t0 = time.time()
    gate_up, down, _ = bench.load_bf16(layer)
    xc, idc, wc = bench.load_capture("calib", layer)
    xe, ide, we = load_eval(torch, Path(args.eval_dir), layer)
    n = min(args.eval_rows, xe.shape[0])
    pick = torch.linspace(0, xe.shape[0] - 1, n).round().long()
    xe, ide, we = (t[pick].contiguous().to(bench.dev) for t in (xe, ide, we))
    print(f"loaded in {time.time() - t0:.0f} s; calib {xc.shape[0]} rows, eval {n}", flush=True)
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {"layer": layer, "methods": {}}
    results.update(calib_rows=int(xc.shape[0]), eval_rows=int(n), prior=args.prior)
    ref, _ = bench.moe_reference(xe, ide, we, lambda e: (gate_up[e].float(), down[e].float()))

    sources = {"": (gate_up, down)}

    def source(tag):
        """BF16 experts, or (tag "v1" / "ckpt") the served NVFP4 experts dequantized as vLLM serves them."""
        if tag not in sources:
            bench.args.model_dir = args.v1_dir if tag == "v1" else args.model_dir
            served = bench.load_served(layer)
            bench.args.model_dir = args.model_dir
            sources[tag] = (qm.dequantize(served["w13"], served["w13_sf"], served["s2_gate"]),
                            qm.dequantize(served["w2"], served["w2_sf"], served["s2_down"]))
        return sources[tag]

    for name in args.methods.split(","):
        if name in results["methods"]:
            continue
        parts = name.split("_")
        tag = parts[0] if parts[0] in ("v1", "ckpt") else ""
        if tag:
            parts = parts[1:]
        if parts == ["nvfp4", "ref"]:  # the served NVFP4 experts themselves (W4A16 path)
            gu, dn = source(tag)
            y, _ = bench.moe_reference(xe, ide, we, lambda e: (gu[e], dn[e]))
            results["methods"][name] = {"vs_bf16": qm.relative_error(y, ref), "cos": qm.cosine(y, ref),
                                        "bits_per_weight": 4.5}
            print(name, json.dumps(results["methods"][name]), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=1))
            continue
        fmt, g, method = parts
        if tag and method == "gptq":
            raise ValueError("GPTQ runs from the BF16 experts")
        src_gu, src_dn = source(tag)
        group = int(g[1:])
        t = time.time()
        h1 = c1 = None
        hess = method in ("gptq", "requant") or method.startswith("signround")
        if hess:
            h1, c1 = qm.hessians(xc, idc, wc, range(B.E), B.E)
            qm.pool_hessians(h1, c1, args.prior)
        gu = quantize(torch, src_gu, h1, c1, fmt, group, method, args.batch, args.prior)
        del h1
        if hess:
            h2, c2 = fc2_hessians(torch, xc, idc, wc, gu)
            qm.pool_hessians(h2, c2, args.prior)
        else:
            h2 = c2 = None
        dn = quantize(torch, src_dn, h2, c2, fmt, group, method, B.E if hess else args.batch, args.prior)
        del h2
        y, _ = bench.moe_reference(xe, ide, we, lambda e: (gu[e], dn[e]))
        row = {"vs_bf16": qm.relative_error(y, ref), "cos": qm.cosine(y, ref),
               "weight_rel_gate_up": qm.relative_error(gu, gate_up), "weight_rel_down": qm.relative_error(dn, down),
               "bits_per_weight": tm.bits_per_weight(group), "seconds": round(time.time() - t, 1)}
        results["methods"][name] = row
        print(name, json.dumps(row), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))
        del gu, dn, y


if __name__ == "__main__":
    main()
