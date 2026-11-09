#!/usr/bin/env python3
"""Whole-iteration kernel census of the decode profiles (kernel work, phase 3 sizing).

Phase 1 (`kernel_trace_table.py`) labelled only the dense GEMMs. This script puts every GPU event of
a decode iteration into a category (dense GEMM, NvFp4 expert GEMM, MoE overhead, router, HC ops, GDN,
QSA, norms/elementwise, sampler, spec-decode/mamba bookkeeping, memcpy/memset), per stream, and
measures the GPU's idle time inside each iteration (wall minus the union of all kernel intervals) and
the gaps between consecutive kernels.

Traces (main checkout, git-ignored):
  mtp3-c1   serving lab control, MTP-3, 1 stream (verify M=4, drafts 4/1/1), original kernels
  mtp3-c8   serving lab control, MTP-3, 8 streams (verify M=32), all-decode windows only
  mtp0-ref  c110 kernels run, control server (MTP-0, 1 stream, original kernels)
  mtp0-ac   c110 kernels run, A+C server (MTP-0, 1 stream, skinny GEMM + INT8 W8A16)

Iterations are cut from one main-stream `execute_context_*` GPU annotation to the next, keeping only
windows whose two annotations both have zero context (prefill) requests.

The NvFp4 experts are also sized: per MoE layer the two grouped GEMMs (gate_up, then down) are timed
separately and compared with the bytes of the distinct experts a step can touch (uniform-routing
expectation 512 * (1 - (1 - 10/512)^T) for T tokens: an upper bound, real routing is correlated).

Usage: python scripts/kernel_phase3_trace.py [--root MAIN_CHECKOUT] [--json OUT]
"""

import argparse
import collections
import glob
import gzip
import json
import re
from pathlib import Path

TRACES = {
    "mtp3-c1": ("runs/modal-validation/arena-serving-lab-v1-modal-20260927/arena/trials/lab-control/profile-c1",
                {"mode": "mtp3", "streams": 1, "tokens": 4}),
    "mtp3-c8": ("runs/modal-validation/arena-serving-lab-v1-modal-20260927/arena/trials/lab-control/profile-c8",
                {"mode": "mtp3", "streams": 8, "tokens": 32}),
    "mtp0-ref": ("runs/modal-validation/arena-duck-60m-c110-kernels-v1-modal-20260928/arena/trials/kernel-ref/bench/"
                 "profile-c1", {"mode": "mtp0", "streams": 1, "tokens": 1}),
    "mtp0-ac": ("runs/modal-validation/arena-duck-60m-c110-kernels-v1-modal-20260928/arena/trials/kernel-gate/bench/"
                "profile-c1", {"mode": "mtp0", "streams": 1, "tokens": 1}),
}

# NvFp4 routed expert (Qwen3.8-Flash-Next): gate_up 1280x2560 and down 2560x640, E2M1 + one E4M3 scale per 16.
EXPERTS, TOP_K, LAYERS = 512, 10, 48
NVFP4_BYTES_PER_PARAM = 0.5 + 1 / 16
GATE_UP_BYTES = 1280 * 2560 * NVFP4_BYTES_PER_PARAM
DOWN_BYTES = 2560 * 640 * NVFP4_BYTES_PER_PARAM
EXPERT_BYTES = GATE_UP_BYTES + DOWN_BYTES
READ_CEILING_TBPS = 1.527  # phase-1 microbenchmark, read-only reduction over 2 GiB

# (category, regex) in priority order; the first match wins.
CATEGORIES = [
    ("nvfp4_expert_gemm", r"BlockScaled"),
    ("mtp_bf16_moe", r"Fused_Moe_Kernel_sm80|MoeFCGemm|expandInputRowsKernel<__nv_bfloat16|finalizeMoeRouting"),
    ("moe_overhead", r"expandInputRowsKernel|doActivationKernel|computeStridesTmaWarpSpecialized|ExpertPrefixSum|"
                     r"cvt_fp16_to_fp4"),
    ("router_topk", r"topkGating"),
    ("dense_gemm", r"cutlass_80|gemvx|splitKreduce|CuteSkinnyGemm|skinny_gemm|_w8a16_kernel|_splitk_reduce_kernel|"
                   r"dot_kernel<|reduce_1Block_kernel|_dequant_kernel"),
    ("hyperconnection", r"^_hc_"),
    ("gdn", r"gdn_decode|gated_delta_rule|causal_conv1d|^chunk_|recompute_w_u|merge_16x16|layer_norm_fwd_kernel|"
            r"solve_tril"),
    ("qsa_attention", r"_qsa_|persistent_topk|reshape_and_cache|_fused_qk_rmsnorm_rope_gate|_expand_qsa_indices|"
                      r"_build_qsa_metadata|_indexer"),
    ("sampler", r"SoftMax|scan_innermost|DeviceScan|RadixSort|sort_postprocess|_temperature_kernel|_gumbel|_resample|"
                r"_rejection|_compute_local_logits_stats|ArgMax|_insert_resampled|_bias_kernel|fill_index_and_segment|"
                r"fill_reverse_indices|_get_num_sampled|_combine_sampled|_scatter_num_accepted|topk_topp|_min_p|"
                r"_penalt"),
    ("spec_mamba_bookkeeping", r"mamba|_get_aligned_state_indices|_update_draft_inputs|_prepare_|"
                               r"_compute_slot_mappings|_gather_block_tables|_post_update|_expand_idx_mapping|"
                               r"searchsorted|arange|elementwise_kernel_with_index"),
    ("norm", r"rms_?norm|RMSNorm|_norm_kernel|gemma_rmsnorm"),
    ("shared_expert_elementwise", r"sigmoid_kernel|mul_silu|silu_kernel"),
    ("fill_memset", r"FillFunctor|^memset32|memcpy32"),
    ("elementwise_misc", r"triton_|at::native|CatArray|indexSelect|at_cuda_detail"),
]
CATEGORIES = [(name, re.compile(rx)) for name, rx in CATEGORIES]


def category(name):
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other"


def distinct_experts(tokens):
    """Uniform independent routing: expected distinct experts per layer for `tokens` tokens (upper bound)."""
    return EXPERTS * (1 - (1 - TOP_K / EXPERTS) ** tokens)


def union_length(intervals):
    total, end = 0.0, None
    start = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            if end is not None:
                total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    if end is not None:
        total += end - start
    return total


def analyse(path, meta):
    trace = glob.glob(f"{path}/*.pt.trace.json.gz")[0]
    events = json.load(gzip.open(trace))["traceEvents"]
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    main = collections.Counter(e["args"]["stream"] for e in kernels).most_common(1)[0][0]
    ann = sorted((e for e in events if e.get("cat") == "gpu_user_annotation" and e.get("tid") == main),
                 key=lambda e: e["ts"])
    decode = re.compile(r"execute_context_0\(0\)_generation_\d+\(\d+\)$")
    wins = [(a["ts"], b["ts"]) for a, b in zip(ann, ann[1:]) if decode.match(a["name"]) and decode.match(b["name"])]
    gpu.sort(key=lambda e: e["ts"])
    n = len(wins)
    cats = collections.defaultdict(lambda: {"main_us": 0.0, "side_us": 0.0, "main_calls": 0, "side_calls": 0})
    names = collections.defaultdict(lambda: {"us": 0.0, "calls": 0, "category": None, "streams": set()})
    busy = idle = 0.0
    gaps = collections.Counter()
    gap_us = collections.Counter()
    main_busy = 0.0
    moe_gemm = collections.defaultdict(list)  # position in the layer (0 gate_up, 1 down) -> durations
    idx = 0
    for t0, t1 in wins:
        while idx < len(gpu) and gpu[idx]["ts"] < t0:
            idx += 1
        j = idx
        inside = []
        while j < len(gpu) and gpu[j]["ts"] < t1:
            inside.append(gpu[j])
            j += 1
        intervals = [(e["ts"], min(e["ts"] + e["dur"], t1)) for e in inside]
        u = union_length(intervals)
        busy += u
        idle += (t1 - t0) - u
        main_iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in inside if e["args"].get("stream") == main)
        main_busy += union_length(main_iv)
        # gaps on the union timeline
        merged = []
        for a, b in sorted(intervals):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for (a0, b0), (a1, b1) in zip(merged, merged[1:]):
            g = a1 - b0
            bucket = ("<1us" if g < 1 else "1-3us" if g < 3 else "3-10us" if g < 10
                      else "10-100us" if g < 100 else ">=100us")
            gaps[bucket] += 1
            gap_us[bucket] += g
        head = merged[0][0] - t0 if merged else t1 - t0
        gaps["window head"] += 1
        gap_us["window head"] += head
        grouped = 0
        for e in inside:
            name = e["name"]
            cat = category(name) if e["cat"] == "kernel" else ("memcpy" if e["cat"] == "gpu_memcpy" else "memset")
            side = e["args"].get("stream") != main
            c = cats[cat]
            c["side_us" if side else "main_us"] += e["dur"]
            c["side_calls" if side else "main_calls"] += 1
            key = name[:160]
            names[key]["us"] += e["dur"]
            names[key]["calls"] += 1
            names[key]["category"] = cat
            names[key]["streams"].add("side" if side else "main")
            if cat == "nvfp4_expert_gemm":
                moe_gemm[grouped % 2].append(e["dur"])
                grouped += 1
    period = sum(b - a for a, b in wins) / n
    out = {**meta, "iterations": n, "period_ms": period / 1000, "gpu_busy_ms": busy / n / 1000,
           "gpu_idle_ms": idle / n / 1000, "main_stream_busy_ms": main_busy / n / 1000,
           "categories": {k: {"main_ms": v["main_us"] / n / 1000, "side_ms": v["side_us"] / n / 1000,
                              "main_calls": v["main_calls"] / n, "side_calls": v["side_calls"] / n}
                          for k, v in sorted(cats.items(), key=lambda kv: -(kv[1]["main_us"] + kv[1]["side_us"]))},
           "gaps": {k: {"count": gaps[k] / n, "ms": gap_us[k] / n / 1000} for k in gaps},
           "kernels": [{"name": k, "category": v["category"], "streams": sorted(v["streams"]),
                        "calls_per_iter": v["calls"] / n, "us_per_call": v["us"] / v["calls"],
                        "us_per_iter": v["us"] / n}
                       for k, v in sorted(names.items(), key=lambda kv: -kv[1]["us"])]}
    # NvFp4 expert GEMM sizing (target model only: the MTP draft layer has BF16 experts)
    if moe_gemm[0]:
        forwards_per_iter = len(moe_gemm[0]) / n / LAYERS
        tokens = meta["tokens"]
        d = distinct_experts(tokens)
        gu, dn = (sum(v) / len(v) for v in (moe_gemm[0], moe_gemm[1]))
        overhead_ms = sum(out["categories"].get(c, {}).get("main_ms", 0) for c in ("moe_overhead",))
        out["experts"] = {
            "forwards_per_iter": forwards_per_iter, "tokens_per_forward": tokens,
            "distinct_experts_uniform": d, "gate_up_us": gu, "down_us": dn,
            "gemm_ms_per_iter": (gu + dn) * LAYERS * forwards_per_iter / 1000,
            "bytes_per_layer_uniform_mb": d * EXPERT_BYTES / 1e6,
            "gate_up_tbps_uniform": d * GATE_UP_BYTES / (gu * 1e-6) / 1e12,
            "down_tbps_uniform": d * DOWN_BYTES / (dn * 1e-6) / 1e12,
            "floor_us_per_layer_uniform": d * EXPERT_BYTES / (READ_CEILING_TBPS * 1e12) * 1e6,
            "measured_us_per_layer": gu + dn,
            # distinct experts implied if the GEMMs ran at the read ceiling (a lower bound on efficiency)
            "experts_at_ceiling": (gu + dn) * 1e-6 * READ_CEILING_TBPS * 1e12 / EXPERT_BYTES,
            "overhead_ms_per_iter": overhead_ms,
            "router_ms_per_iter": out["categories"].get("router_topk", {}).get("main_ms", 0),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="checkout holding runs/ (git-ignored)")
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=45)
    args = ap.parse_args()
    result = {}
    for key, (rel, meta) in TRACES.items():
        path = Path(args.root) / rel
        if not glob.glob(f"{path}/*.pt.trace.json.gz"):
            print(f"skip {key}: no trace under {path}")
            continue
        r = analyse(path, meta)
        result[key] = r
        print(f"\n## {key}: {r['iterations']} iterations, period {r['period_ms']:.2f} ms, "
              f"GPU busy {r['gpu_busy_ms']:.2f}, idle {r['gpu_idle_ms']:.2f}, "
              f"main-stream busy {r['main_stream_busy_ms']:.2f} ms")
        print("| category | main ms | side ms | main calls | side calls |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for c, v in r["categories"].items():
            print(f"| {c} | {v['main_ms']:.3f} | {v['side_ms']:.3f} | {v['main_calls']:.1f} | {v['side_calls']:.1f} |")
        print("gaps:", {k: (round(v["count"], 1), round(v["ms"], 3)) for k, v in sorted(r["gaps"].items())})
        if "experts" in r:
            print("experts:", {k: round(v, 3) for k, v in r["experts"].items()})
        for row in r["kernels"][:args.top]:
            print(f"  {row['us_per_iter']:8.1f} us/it {row['calls_per_iter']:6.1f} x {row['us_per_call']:7.2f} "
                  f"[{row['category']}/{','.join(row['streams'])}] {row['name'][:110]}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
