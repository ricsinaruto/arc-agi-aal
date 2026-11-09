#!/usr/bin/env python3
"""Large-batch expert and prefill GEMM microbenchmark (throughput arm, 16 Oct; build track).

Runs standalone on one RTX PRO 6000 (launcher: scripts/modal_kernel_bench_large.py), on top of
kernel_bench_phase4.py (its timing, NvFp4 layer loading, CUTLASS and b12x paths). Parts:

- ``marlin``: vLLM's fused Marlin MoE at the served shape (512 experts, top-10, H 2560, I 640) for
  int4 g128 W4A16 (the AutoRound experts), int4 g128 W4A8-FP8 (VLLM_MARLIN_INPUT_DTYPE=fp8: per-token FP8
  activations on FP8 MMA) and NvFp4 g16 W4A16, M = 32-4096 tokens, uniform routing (real decode batches route
  to ~as many distinct experts as uniform: 279 of 512 at 40 tokens). Weights: one GPTQ-quantized random expert
  copied into every slot (timing reads 512 distinct copies; values do not matter). Error of each against the
  BF16-activation reference of the same quantized weights (Gaussian activations: a lower bound for A8).
- ``moe``: the phase-4 part (FlashInfer CUTLASS NvFp4 W4A4 and b12x on a real NvFp4 layer) at the same M.
- ``dense``: BF16 F.linear against FP8 W8A8 torch._scaled_mm (row-wise scales; GEMM only and with the dynamic
  per-token activation quantization) for the large projections at M = 64-4096 (prefill chunks are 4,096).
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

E, H, INTER, TOPK = 512, 2560, 640, 10
MS = [32, 64, 112, 160, 256, 400, 1024, 4096]
DENSE_SHAPES = [(16384, 2560), (13312, 2560), (2560, 6144), (10240, 2560), (2560, 2560)]
DENSE_MS = [64, 112, 256, 1024, 4096]
FP8 = torch.float8_e4m3fn


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase4", required=True)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--helper", required=True)
    ap.add_argument("--parts", default="marlin,moe,dense")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--variants", default="", help="comma-separated Marlin variants (default: all)")
    ap.add_argument("--deadline-seconds", type=float, default=600)
    args = ap.parse_args()
    P = load("kernel_bench_phase4", args.phase4)
    P.MS = list(MS)
    args.pinned = ""
    bench = P.Bench(args)
    bench.results.update({"marlin": [], "marlin_error": [], "dense": []})
    bench.env()

    def part_marlin():
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_nvfp4_like
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
        from vllm.scalar_type import scalar_types

        dev = bench.dev
        gen = torch.Generator(device="cpu").manual_seed(0)
        w1 = (torch.randn(2 * INTER, H, generator=gen) / 50).to(torch.bfloat16).to(dev)   # [2I, H] gate;up
        w2 = (torch.randn(H, INTER, generator=gen) / 50).to(torch.bfloat16).to(dev)       # [H, I]
        variants = [("int4_g128_a16", scalar_types.uint4b8, 128, None),
                    ("int4_g128_a8fp8", scalar_types.uint4b8, 128, FP8),
                    ("nvfp4_g16_a16", scalar_types.float4_e2m1f, 16, None)]
        only = args.variants
        if only:
            variants = [v for v in variants if v[0] in only.split(",")]
        x_all = (torch.randn(max(MS), H, generator=gen) / 4).to(torch.bfloat16).to(dev)
        for name, qt, group, adt in variants:
            if bench.left() < 90:
                bench.results["notes"].append(f"marlin stopped before {name}: time")
                break
            try:
                gs1 = gs2 = None
                if qt == scalar_types.float4_e2m1f:
                    r1, q1, s1, gs1 = rand_marlin_weight_nvfp4_like(w1, group, input_dtype=adt)
                    r2, q2, s2, gs2 = rand_marlin_weight_nvfp4_like(w2, group, input_dtype=adt)
                    r1, r2 = r1.T, r2.T
                else:
                    r1, q1, s1 = marlin_quantize(w1.T.contiguous(), qt, group, input_dtype=adt)
                    r2, q2, s2 = marlin_quantize(w2.T.contiguous(), qt, group, input_dtype=adt)
                    r1, r2 = r1.T, r2.T  # [2I, H], [H, I] dequantized
                rep = lambda t: t.unsqueeze(0).repeat(E, *([1] * t.dim())).contiguous()
                Q1, Q2, S1, S2 = rep(q1), rep(q2), rep(s1), rep(s2)
                G1 = None if gs1 is None else gs1.reshape(1).repeat(E).contiguous()
                G2 = None if gs2 is None else gs2.reshape(1).repeat(E).contiguous()

                def run(x, ids, wts, out=None):
                    return fused_marlin_moe(x, Q1, Q2, None, None, S1, S2, wts, ids, quant_type_id=qt.id,
                                            global_num_experts=E, global_scale1=G1, global_scale2=G2,
                                            input_dtype=adt, output=out)

                # error: all experts share the weights, so the reference is one expert's BF16-activation MLP
                x = x_all[:64]
                ids, wts = bench.routes(64, 1, "uniform", gen, dev)
                y = run(x, ids[0], wts[0] / wts[0].sum(-1, keepdim=True))
                h = x.float() @ r1.float().T
                ref = (F.silu(h[:, :INTER]) * h[:, INTER:]) @ r2.float().T
                row = {"variant": name, "rel_error_vs_bf16_act_same_weights": P.rel(y, ref),
                       "weight_rel_error": P.rel(torch.cat([r1.flatten(), r2.flatten()]),
                                                 torch.cat([w1.flatten(), w2.flatten()]))}
                bench.results["marlin_error"].append(row)
                P.log({"marlin_error": row})
                for m in MS:
                    if bench.left() < 45:
                        bench.results["notes"].append(f"marlin {name} stopped before M={m}: time")
                        break
                    x = x_all[:m].contiguous()
                    ids, wts = bench.routes(m, 16, "uniform", gen, dev)
                    outs = [torch.empty(m, H, dtype=torch.bfloat16, device=dev) for _ in range(16)]
                    row = {"variant": name, "m": m}
                    try:
                        calls = [lambda r=r: run(x, ids[r], wts[r], outs[r]) for r in range(16)]
                        row["us"], row["us_min"] = bench.graph_time(calls)
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    bench.results["marlin"].append(row)
                    P.log({"marlin": row})
                    bench.flush()
                del Q1, Q2, S1, S2
                torch.cuda.empty_cache()
            except Exception as exc:
                bench.results["errors"].append(f"marlin {name}: {type(exc).__name__}: {str(exc)[:400]}")
                P.log({"marlin_failure": name, "trace": traceback.format_exc()[-2000:]})

    def part_dense():
        dev = bench.dev
        gen = torch.Generator(device="cpu").manual_seed(1)
        for n, k in DENSE_SHAPES:
            if bench.left() < 45:
                bench.results["notes"].append(f"dense stopped before {n}x{k}: time")
                break
            copies = max(2, int((256 << 20) // (n * k * 2)) + 1)
            w = [(torch.randn(n, k, generator=gen) / 50).to(torch.bfloat16).to(dev) for _ in range(copies)]
            wq, ws = [], []
            for t in w:
                s = t.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0
                wq.append((t.float() / s).to(FP8))
                ws.append(s.T.contiguous())  # [1, N]
            for m in DENSE_MS:
                x = (torch.randn(m, k, generator=gen) / 4).to(torch.bfloat16).to(dev)
                xs = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0
                xq = (x.float() / xs).to(FP8)
                row = {"n": n, "k": k, "m": m}
                try:
                    row["bf16_us"], _ = bench.graph_time([lambda i=i: F.linear(x, w[i]) for i in range(copies)])
                    mm = lambda a, i, sa: torch._scaled_mm(a, wq[i].T, scale_a=sa, scale_b=ws[i], out_dtype=torch.bfloat16)
                    row["fp8_gemm_us"], _ = bench.graph_time([lambda i=i: mm(xq, i, xs) for i in range(copies)])

                    def dyn(i):
                        s = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8) / 448.0
                        return mm((x.float() / s).to(FP8), i, s)

                    row["fp8_dynamic_us"], _ = bench.graph_time([lambda i=i: dyn(i) for i in range(copies)])
                    ref = x.float() @ w[0].float().T
                    row["bf16_rel"] = P.rel(F.linear(x, w[0]), ref)
                    row["fp8_rel"] = P.rel(mm(xq, 0, xs), ref)
                    row["fp8_speedup"] = row["bf16_us"] / row["fp8_dynamic_us"]
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                bench.results["dense"].append(row)
                P.log({"dense": row})
                bench.flush()
            del w, wq, ws
            torch.cuda.empty_cache()

    parts = {"marlin": part_marlin, "moe": bench.part_moe, "dense": part_dense}
    from vllm.config import VllmConfig, set_current_vllm_config  # QuantFP8 (W4A8 activation quant) is a CustomOp

    for name in args.parts.split(","):
        try:
            with set_current_vllm_config(VllmConfig()):
                parts[name]()
        except Exception as exc:
            bench.results["errors"].append(f"part {name}: {type(exc).__name__}: {str(exc)[:400]}")
            P.log({"part_failure": name, "trace": traceback.format_exc()[-2000:]})
        bench.flush(True)
    bench.flush(True)


if __name__ == "__main__":
    main()
