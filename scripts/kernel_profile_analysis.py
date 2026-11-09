#!/usr/bin/env python3
"""Per-kernel GPU time per engine step, grouped, with a bytes / FLOPs roofline (kernel-profile pilot, 30 September;
docs/validation-20260930/kernel-profile.md).

Input: the pilot's ``bench.json`` (kernel_profile points) and the torch-profiler traces next to it. Steps are cut from
one main-stream ``execute_context_*`` GPU annotation to the next; a step is decode-only when its annotation says
``execute_context_0(0)`` (no prefill request). For each trace:

- period, GPU busy (union of kernel / memcpy / memset intervals) and idle inside the steps, launch-gap histogram;
- ms per step and calls per step by group (below), top kernels;
- a byte / FLOP model per group from the model's shapes (Qwen3.8-Flash-Next, prune12, the served dtypes) and the
  point's stream count and mean context, and the achieved GB/s against the measured read ceiling (1.53 TB/s,
  kernels phase 1) and the GDDR7 peak (1.79 TB/s).

Usage: python scripts/kernel_profile_analysis.py RUN_DIR [--json OUT]
  RUN_DIR: the collected trial folder (bench.json and profile-*/ traces)
"""

import argparse
import collections
import glob
import gzip
import json
import re
from pathlib import Path

READ_CEILING = 1.53e12   # B/s, measured read-only reduction (kernels-phase1.md)
PEAK_BW = 1.79e12        # B/s, GDDR7 spec (RTX PRO 6000 Blackwell, 512-bit at 28 Gb/s)
BF16_TFLOPS = 250e12     # dense BF16 tensor (FP32 accumulate) order of magnitude on SM120; FLOPs are reported, the
#                          roofline uses bandwidth: every group here is far below the compute ridge at M <= 56.

# Model (config: RadixArk/Qwen3.8-Flash-Next-NVFP4; prune12 keeps 448 of 512 experts per layer).
H, LAYERS, GDN_LAYERS, QSA_LAYERS = 2560, 48, 36, 12
EXPERTS, TOP_K, MOE_I, SHARED_I = 448, 10, 640, 640
VOCAB = 248320
NVFP4 = 0.5 + 1 / 16                       # bytes per expert parameter (E2M1 + one E4M3 scale per 16)
EXPERT_PARAMS = 2 * MOE_I * H + H * MOE_I  # gate_up [1280, 2560] + down [2560, 640]
EXPERT_BYTES = EXPERT_PARAMS * NVFP4
INT8 = 1 + 2 / 128                         # W8A16 per-channel-group scale overhead (upper bound)
DENSE_INT8 = {  # family: params per step (int8 weights: lm_head, gdn_in_proj_qkvz, qsa_qkv, attn_out)
    "lm_head": VOCAB * H, "gdn_in_proj_qkvz": GDN_LAYERS * 16384 * H, "qsa_qkv": QSA_LAYERS * 13312 * H,
    "attn_out": LAYERS * H * 6144}
DENSE_BF16 = {  # remaining BF16 projections per step
    "hc_down_inject": 2 * LAYERS * 336 * 10240, "hc_up": 2 * LAYERS * 10240 * 320, "hc_final": 320 * 10240,
    "gdn_in_proj_ba": GDN_LAYERS * 96 * H, "router": LAYERS * EXPERTS * H,
    "shared_expert": LAYERS * 3 * SHARED_I * H, "indexer_qk": QSA_LAYERS * 640 * H}
# GDN recurrent state: 48 value heads x (128 x 128) in BF16 (mamba_ssm_cache_dtype "auto" follows the model dtype;
# the config's mamba_ssm_dtype float32 is not read by this model class), read and written every decode step. One
# layer's states for 40 streams (63 MB) fit the 128 MB L2, so the write-back overlaps later kernels.
GDN_STATE_BYTES = GDN_LAYERS * 48 * 128 * 128 * 2
GDN_CONV_BYTES = GDN_LAYERS * (2 * 16 * 128 + 48 * 128) * 3 * 2  # conv window (kernel 4 - 1), bf16
# QSA: the indexer scores the compressed keys (ratio 4, 128 dims, BF16) of the whole context every step
# (_qsa_mqa_paged_kernel); the top-2048 compressed blocks expand to 2,048 x 4 = 8,192 tokens (_expand_qsa_indices)
# whose int4 K/V pages (264 B per token and KV head, 2 KV heads) the split-K reader loads (_qsa_kv4_splitk_kernel).
INDEXER_BYTES_PER_TOKEN = QSA_LAYERS * 128 * 2 / 4
KV4_BYTES_PER_TOKEN = QSA_LAYERS * 2 * 264
INDEXER_BUDGET = 2048 * 4

GROUPS = [  # (group, regex) in priority order; the first match wins
    ("moe_expert_gemm", r"marlin_moe|Marlin<|BlockScaled|moe_wna16"),
    ("moe_glue", r"moe_sum|moe_align_block_size|count_and_sort_expert|topkGating|act_and_mul_kernel|"
                 r"expandInputRows|finalizeMoeRouting|_dequant_kernel"),
    ("dense_int8", r"_w8a16"),
    ("dense_bf16", r"cutlass_80|gemvx|gemv2T|splitKreduce|CuteSkinny|skinny_gemm|_dense_gemm|_splitk_reduce|"
                   r"dot_kernel<|reduce_1Block_kernel|_mid_m"),
    ("hyperconnection", r"^_hc_"),
    ("gdn_decode_state", r"fused_recurrent_gated_delta_rule|fused_sigmoid_gating_delta_rule_update|"
                         r"_causal_conv1d_update|gdn_decode"),
    ("gdn_prefill", r"^chunk_|chunk_gated_delta|recompute_w_u|merge_16x16|solve_tril|_causal_conv1d_fwd|"
                    r"chunk_scaled_dot_kkt"),
    ("gdn_misc", r"_fused_post_conv|layer_norm_fwd_kernel|mamba"),
    ("qsa_indexer", r"_qsa_mqa_paged|persistent_topk|_qsa_pre_indexer|_indexer|_expand_qsa_indices|mbtopk"),
    ("qsa_kv_reader", r"_qsa_kv4|_qsa_kv8|_qsa_.*splitk|_qsa_.*merge|kv4_read|_qsa_attn"),
    ("qsa_kv_writer", r"_reshape_cache|kv4_write|_qsa_kv_write|reshape_and_cache|_zero_kv_blocks"),
    ("qsa_other", r"_qsa_|_fused_qk_rmsnorm_rope_gate|_prepare_rope"),
    ("sampler", r"_temperature_kernel|_gumbel|topk_first|topk_pinned|_topk_sampler|SoftMax|RadixSort|DeviceScan|"
                r"scan_innermost|ArgMax|_penalt|_min_p|topk_topp|_resample|_arc_topk|argmax"),
    ("norm", r"rms_?norm|RMSNorm|_norm_kernel|gemma_rmsnorm"),
    ("memcpy_memset", r"^Memcpy|^Memset"),
    ("elementwise_misc", r"triton_|at::native|CatArray|indexSelect|at_cuda_detail|FillFunctor"),
]
GROUPS = [(name, re.compile(rx)) for name, rx in GROUPS]
DECODE = re.compile(r"execute_context_0\(0\)_generation_(\d+)\((\d+)\)$")
ANNOT = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")


def group(name):
    for g, rx in GROUPS:
        if rx.search(name):
            return g
    return "other"


def union(intervals):
    total, start, end = 0.0, None, None
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


def expected_unique(tokens, experts=EXPERTS, k=TOP_K):
    """Distinct experts per layer for ``tokens`` independent uniformly routed tokens (an upper bound)."""
    return experts * (1 - (1 - k / experts) ** tokens)


DENSE_BF16_SIDE = ("shared_expert",)  # runs on the side stream, overlapped with the routed experts


def byte_model(streams, context):
    """Bytes read (and for the GDN state also written) per decode step by group."""
    main_bf16 = sum(v for k, v in DENSE_BF16.items() if k not in DENSE_BF16_SIDE) * 2
    return {
        "dense_int8": sum(DENSE_INT8.values()) * INT8,
        "dense_bf16_main": main_bf16,
        "dense_bf16_side_shared_expert": DENSE_BF16["shared_expert"] * 2,
        "moe_expert_gemm_uniform": LAYERS * expected_unique(streams) * EXPERT_BYTES,
        "gdn_state_read": streams * (GDN_STATE_BYTES + GDN_CONV_BYTES),
        "gdn_state_write": streams * (GDN_STATE_BYTES + GDN_CONV_BYTES),
        "qsa_indexer": streams * context * INDEXER_BYTES_PER_TOKEN,
        "qsa_kv_reader": streams * min(context, INDEXER_BUDGET) * KV4_BYTES_PER_TOKEN,
        "sampler_logits": streams * VOCAB * 4 * 3,  # write FP32 logits, ~2 read passes
    }


def flop_model(streams, context):
    active = (sum(DENSE_INT8.values()) + sum(DENSE_BF16.values()) + LAYERS * TOP_K * EXPERT_PARAMS)
    attn = QSA_LAYERS * (4 * 128 * context / 4 + 24 * 256 * 2 * min(context, INDEXER_BUDGET)) * 2
    return {"weights": 2 * active * streams, "attention": attn * streams}


def row(ms, nbytes):
    return {"ms": round(ms, 3), "bytes_gb": round(nbytes / 1e9, 3),
            "floor_ms_at_ceiling": round(nbytes / READ_CEILING * 1000, 3),
            "achieved_tbs": round(nbytes / (ms / 1000) / 1e12, 3) if ms else None,
            "fraction_of_ceiling": round(nbytes / (ms / 1000) / READ_CEILING, 3) if ms else None,
            "fraction_of_peak": round(nbytes / (ms / 1000) / PEAK_BW, 3) if ms else None}


def roofline(decode, streams, context):
    """Achieved bandwidth per group: main-stream (critical-path) ms against the byte model."""
    model = byte_model(streams, context)
    g, side = decode["groups_ms"], decode.get("groups_side_ms", {})
    main = {k: v - side.get(k, 0.0) for k, v in g.items()}
    rows = {
        "moe_expert_gemm (bytes at uniform routing: an upper bound)": row(g.get("moe_expert_gemm", 0.0),
                                                                            model["moe_expert_gemm_uniform"]),
        "dense_int8 (lm_head, qkvz, qsa_qkv, attn_out)": row(main.get("dense_int8", 0.0), model["dense_int8"]),
        "dense_bf16 main stream (HC down/up, router, in_proj_ba, indexer q/k)": row(main.get("dense_bf16", 0.0),
                                                                                     model["dense_bf16_main"]),
        "gdn_decode_state (read; the write-back is absorbed by L2)": row(g.get("gdn_decode_state", 0.0),
                                                                          model["gdn_state_read"]),
        "qsa_indexer (compressed-key scan + top-k)": row(g.get("qsa_indexer", 0.0), model["qsa_indexer"]),
        "qsa_kv_reader (8,192 selected tokens, int4)": row(g.get("qsa_kv_reader", 0.0), model["qsa_kv_reader"]),
        "sampler": row(g.get("sampler", 0.0), model["sampler_logits"]),
    }
    moe_ms = g.get("moe_expert_gemm", 0.0)
    rows["moe_implied_distinct_experts_at_ceiling"] = round(moe_ms / 1000 * READ_CEILING / LAYERS / EXPERT_BYTES, 1)
    rows["moe_uniform_distinct_experts"] = round(expected_unique(streams), 1)
    rows["side_stream_ms"] = {k: round(v, 3) for k, v in side.items() if v > 0.005}
    rows["unmodelled_main_ms"] = {k: round(main.get(k, 0.0), 3) for k in (
        "moe_glue", "hyperconnection", "elementwise_misc", "gdn_misc", "qsa_other", "qsa_kv_writer", "other",
        "memcpy_memset", "norm", "gdn_prefill")}
    rows["flops_gflop"] = {k: round(v / 1e9, 1) for k, v in flop_model(streams, context).items()}
    read = sum(v for k, v in model.items() if k not in ("sampler_logits", "moe_expert_gemm_uniform",
                                                        "gdn_state_write"))
    implied_moe = moe_ms / 1000 * READ_CEILING  # MoE bytes if Marlin reads at the ceiling
    rows["step_floor_ms_with_implied_moe"] = round((read + implied_moe) / READ_CEILING * 1000, 2)
    rows["step_floor_gb_with_implied_moe"] = round((read + implied_moe) / 1e9, 2)
    return rows


def analyse_trace(path):
    events = json.load(gzip.open(path) if str(path).endswith(".gz") else open(path))["traceEvents"]
    gpu = sorted((e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e),
                 key=lambda e: e["ts"])
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    if not kernels:
        return None
    main = collections.Counter(e["args"].get("stream") for e in kernels).most_common(1)[0][0]
    ann = [e for e in events if e.get("cat") == "gpu_user_annotation" and ANNOT.match(e.get("name", ""))]
    # Graph-replayed steps carry the annotation on the main and the side stream: keep the busiest stream's copy.
    tid = collections.Counter(e.get("tid") for e in ann).most_common(1)[0][0] if ann else None
    ann = sorted((e for e in ann if e.get("tid") == tid), key=lambda e: e["ts"])
    steps = {"decode": [], "mixed": []}
    for a, b in zip(ann, ann[1:]):
        m = ANNOT.match(a["name"])
        kind = "decode" if m.group(1) == "0" else "mixed"
        steps[kind].append((a["ts"], b["ts"], int(m.group(3)), int(m.group(2))))
    out = {"file": str(path), "main_stream": main}
    for kind, wins in steps.items():
        if not wins:
            continue
        n = len(wins)
        cats = collections.defaultdict(lambda: [0.0, 0])
        side = collections.Counter()  # side-stream (overlapped) kernel time per group
        names = collections.defaultdict(lambda: [0.0, 0, None])
        busy = idle = 0.0
        gaps = collections.Counter()
        gap_us = collections.Counter()
        idx = 0
        for t0, t1, _, _ in wins:
            while idx < len(gpu) and gpu[idx]["ts"] < t0:
                idx += 1
            j, inside = idx, []
            while j < len(gpu) and gpu[j]["ts"] < t1:
                inside.append(gpu[j])
                j += 1
            iv = [(e["ts"], min(e["ts"] + e["dur"], t1)) for e in inside]
            u = union(iv)
            busy += u
            idle += (t1 - t0) - u
            merged = []
            for a, b in sorted(iv):
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            for (_, b0), (a1, _) in zip(merged, merged[1:]):
                g = a1 - b0
                bucket = "<2us" if g < 2 else "2-5us" if g < 5 else "5-20us" if g < 20 else "20-200us" if g < 200 \
                    else ">=200us"
                gaps[bucket] += 1
                gap_us[bucket] += g
            if merged:
                gap_us["step head"] += merged[0][0] - t0
            for e in inside:
                name = e["name"]
                g = group(name) if e["cat"] == "kernel" else "memcpy_memset"
                cats[g][0] += e["dur"]
                cats[g][1] += 1
                if e["args"].get("stream") != main:
                    side[g] += e["dur"]
                entry = names[name[:150]]
                entry[0] += e["dur"]
                entry[1] += 1
                entry[2] = g
        period = sum(t1 - t0 for t0, t1, _, _ in wins) / n
        out[kind] = {
            "steps": n, "period_ms": round(period / 1000, 3), "gpu_busy_ms": round(busy / n / 1000, 3),
            "gpu_idle_ms": round(idle / n / 1000, 3), "batch_mean": round(sum(w[2] for w in wins) / n, 2),
            "prefill_tokens_mean": round(sum(w[3] for w in wins) / n, 1),
            "groups_ms": {k: round(v[0] / n / 1000, 4) for k, v in sorted(cats.items(), key=lambda kv: -kv[1][0])},
            "groups_calls": {k: round(v[1] / n, 1) for k, v in cats.items()},
            "groups_side_ms": {k: round(v / n / 1000, 4) for k, v in side.items()},
            "gaps": {k: {"count": round(gaps[k] / n, 1), "ms": round(gap_us[k] / n / 1000, 4)} for k in gap_us},
            "top_kernels": [{"name": k, "group": v[2], "us_per_step": round(v[0] / n, 1),
                             "calls_per_step": round(v[1] / n, 2), "us_per_call": round(v[0] / v[1], 2)}
                            for k, v in sorted(names.items(), key=lambda kv: -kv[1][0])[:40]],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--json")
    args = parser.parse_args()
    root = Path(args.run_dir)
    bench = json.loads((root / "bench.json").read_text())
    kp = bench.get("kernel_profile") or {}
    result = {"points": []}
    for point in kp.get("points", []):
        prof = point.get("profile") or {}
        for trace in prof.get("traces", []):
            path = root / trace["file"]
            if not path.exists():
                candidates = glob.glob(str(root / f"profile-{point['label']}" / "*.pt.trace.json*"))
                if not candidates:
                    continue
                path = Path(candidates[0])
            analysis = analyse_trace(path)
            if analysis is None:
                continue
            entry = {"label": point["label"], "streams": point["streams"],
                     "mean_prompt_tokens": point.get("mean_prompt_tokens"), "window_step_ms": point.get("step_ms"),
                     "trace": analysis}
            if "decode" in analysis:
                entry["roofline"] = roofline(analysis["decode"], point["streams"],
                                             point.get("mean_prompt_tokens") or 0)
            result["points"].append(entry)
            d = analysis.get("decode") or {}
            print(f"{point['label']:>10}: window {point.get('step_ms')} ms/step; trace decode period "
                  f"{d.get('period_ms')} ms, busy {d.get('gpu_busy_ms')}, idle {d.get('gpu_idle_ms')}")
            for g, ms in (d.get("groups_ms") or {}).items():
                print(f"{'':>14}{g:<22}{ms:8.3f} ms  {d['groups_calls'].get(g)} calls")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
