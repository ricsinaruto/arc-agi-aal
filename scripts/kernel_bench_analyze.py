#!/usr/bin/env python3
"""Turn the phase-1 GEMM microbenchmark into per-iteration decode projections.

Inputs: the pass files written by scripts/kernel_bench_gpu.py (p1..p5 *.json) and the trace table
(docs/validation-20260927/kernels-phase1-trace.json). For each option (a per-shape kernel choice)
it sums the benchmarked per-call times over the model's calls per decode iteration and projects
iteration time and per-stream tok/s with

    T_new = T_measured - dense_bench(BF16 F.linear) + dense_bench(option),

so the benchmark's own BF16 baseline cancels its systematic offset to the traced kernels.

Usage: python scripts/kernel_bench_analyze.py --bench-dir DIR [--json OUT]
"""

import argparse
import glob
import json
import os

MS = [1, 2, 4, 6, 8, 16, 24, 32, 64]
SHAPE_BYTES = {"hc_down_inject": 336 * 10240 * 2, "hc_mixer_down": 320 * 10240 * 2, "hc_up": 10240 * 320 * 2,
               "gdn_in_qkvz": 16384 * 2560 * 2, "gdn_in_ba": 96 * 2560 * 2, "attn_out": 2560 * 6144 * 2,
               "qsa_qkv": 13312 * 2560 * 2, "qsa_index_qk": 640 * 2560 * 2, "router": 512 * 2560 * 2,
               "shared_gate_up": 1280 * 2560 * 2, "shared_down": 2560 * 640 * 2, "ple_key": 10240 * 2560 * 2,
               "h2560": 2560 * 2560 * 2, "lm_head": 248320 * 2560 * 2}

# Calls per forward of the main model (48 layers: 36 GDN + 12 QSA, 1 PLE layer, final HC mixer).
MAIN = {"hc_down_inject": 96, "hc_mixer_down": 1, "hc_up": 97, "gdn_in_qkvz": 36, "gdn_in_ba": 36,
        "attn_out": 48, "qsa_qkv": 12, "qsa_index_qk": 12, "router": 48, "shared_gate_up": 48,
        "shared_down": 48, "ple_key": 1, "h2560": 1, "lm_head": 1}
# Calls per MTP draft step (fc_embedding + fc_hidden on hc_count x rows, one QSA decoder layer with
# two HCs, the HC mixer, the draft lm_head on one row per request).
DRAFT = {"h2560": 1, "hc_down_inject": 2, "hc_mixer_down": 1, "hc_up": 3, "qsa_qkv": 1, "qsa_index_qk": 1,
         "attn_out": 1, "router": 1, "shared_gate_up": 1, "shared_down": 1}

# Measured decode iterations (serving lab v1): MTP-3 profile 21.3 ms at 1 stream and 34.2 ms at
# 8 streams; MTP-0 per-stream 71-84 tok/s at 1 stream and 52-57 at 8 streams (decode-only est.).
MTP3_T = {1: 21.3, 8: 34.2}
MTP3_TOKENS_PER_ITER = 2.6  # 1 + ~1.6 accepted drafts per iteration (lab: 1.47-1.90)
MTP0_T = {1: 1000 / 77.0, 8: 1000 / 55.0}

BF16_TODAY = ["bf16_linear"]
BF16_SKINNY = ["cute_skinny_bf16", "bf16_linear"]  # the pinned CuTe skinny GEMM at M <= 16, else cuBLAS
BF16_BEST = ["bf16_linear", "bf16_linear_altblas", "bf16_mm_kmajor", "triton_bf16", "cute_skinny_bf16"]
W8A16_FP8 = ["marlin_w8a16_fp8"]
W8A16_INT8 = ["triton_w8a16_int8"]
W8A8_FP8_DYN = ["vllm_cutlass_fp8_tensor+dynq", "vllm_cutlass_fp8_channel+dynq"]
W8A8_FP8_STATIC = ["vllm_cutlass_fp8_tensor", "vllm_cutlass_fp8_channel", "fp8_scaled_mm_tensor"]
NVFP4 = ["vllm_cutlass_nvfp4+actq"]
VOCAB32K = 32768 / 248320  # draft lm_head over a 32K frequent-token sub-vocabulary (FR-Spec style)

# option -> {role: [formats]}; per (shape, role) the format with the least time over the schedule is
# chosen (a layer stores one weight format); within a format the fastest kernel per M is used (a
# dispatch table by token count, as in the pinned low_latency_gemm plans).
OPTIONS = {
    "BF16 F.linear (today)": {"main": [BF16_TODAY], "draft": [BF16_TODAY]},
    "BF16 CuTe skinny GEMM (M<=16) else cuBLAS": {"main": [BF16_SKINNY], "draft": [BF16_SKINNY]},
    "BF16 best kernel per shape/M": {"main": [BF16_BEST], "draft": [BF16_BEST]},
    "BF16 best + draft-only W8A16 (MTP layer, draft lm_head)": {
        "main": [BF16_BEST], "draft": [BF16_BEST, W8A16_INT8, W8A16_FP8]},
    "BF16 best + draft lm_head on 32K vocab": {"main": [BF16_BEST], "draft": [BF16_BEST], "vocab": VOCAB32K},
    "BF16 best + draft W8A16 + draft 32K vocab": {
        "main": [BF16_BEST], "draft": [BF16_BEST, W8A16_INT8, W8A16_FP8], "vocab": VOCAB32K},
    "W8A16 FP8 Marlin where faster (all roles)": {"main": [BF16_BEST, W8A16_FP8], "draft": [BF16_BEST, W8A16_FP8]},
    "W8A16 INT8/FP8 where faster (all roles)": {
        "main": [BF16_BEST, W8A16_INT8, W8A16_FP8], "draft": [BF16_BEST, W8A16_INT8, W8A16_FP8]},
    "W8A8 FP8 + dynamic act quant where faster": {"main": [BF16_BEST, W8A8_FP8_DYN], "draft": [BF16_BEST, W8A8_FP8_DYN]},
    "W8A8 FP8 GEMM only (static act scale) where faster": {
        "main": [BF16_BEST, W8A8_FP8_STATIC], "draft": [BF16_BEST, W8A8_FP8_STATIC]},
    "NVFP4 W4A4 where faster": {"main": [BF16_BEST, NVFP4], "draft": [BF16_BEST, NVFP4]},
    "W8A16 INT8 everywhere (forced)": {"main": [W8A16_INT8], "draft": [W8A16_INT8]},
    "NVFP4 everywhere (forced)": {"main": [NVFP4], "draft": [NVFP4]},
}


def load(bench_dir):
    rows, meta = [], {}
    for path in sorted(glob.glob(os.path.join(bench_dir, "p*.json"))):
        data = json.load(open(path))
        tag = os.path.basename(path)[:-5]
        meta[tag] = {k: data.get(k) for k in ("env", "bandwidth", "notes", "errors", "elapsed_seconds",
                                               "finished", "act_quant", "methods")}
        rows += data.get("results", [])
    return rows, meta


def table(rows):
    t = {}
    for r in rows:
        t.setdefault((r["shape"], r["method"]), {})[r["M"]] = r
    return t


def time_at(t, shape, method, m):
    """Per-call us at M; linear interpolation between measured M, None outside the measured range."""
    pts = t.get((shape, method))
    if not pts:
        return None
    if m in pts:
        return pts[m]["us"]
    lo = max((x for x in pts if x < m), default=None)
    hi = min((x for x in pts if x > m), default=None)
    if lo is None or hi is None:
        return None
    a, b = pts[lo]["us"], pts[hi]["us"]
    return a + (b - a) * (m - lo) / (hi - lo)


def best_time(t, shape, methods, m, fallback="bf16_linear"):
    cands = [(time_at(t, shape, meth, m), meth) for meth in methods]
    cands = [c for c in cands if c[0] is not None]
    if not cands:
        base = time_at(t, shape, fallback, m)
        return (base, fallback + " (fallback)") if base is not None else (None, None)
    return min(cands)


def dense_ms(t, option, sched):
    """sched: list of (shape, M, calls, role). Returns (ms, {shape/role: [ms, format methods]})."""
    total, parts = 0.0, {}
    groups = {}
    for shape, m, calls, role in sched:
        groups.setdefault((shape, role), []).append((m, calls))
    for (shape, role), items in groups.items():
        best = None
        for fmt in option[role]:
            us_sum, used = 0.0, set()
            for m, calls in items:
                us, meth = best_time(t, shape, fmt, m)
                if us is None:
                    us_sum = None
                    break
                if shape == "lm_head" and role == "draft" and option.get("vocab"):
                    us *= option["vocab"]
                us_sum += us * calls
                used.add(meth)
            if us_sum is not None and (best is None or us_sum < best[0]):
                best = (us_sum, sorted(used))
        if best is None:
            continue
        total += best[0]
        parts[f"{shape}/{role}"] = [round(best[0] / 1000, 3), best[1]]
    return total / 1000, parts


def schedule(mode, streams):
    """(shape, M, calls, role) per decode iteration."""
    out = []
    if mode == "mtp3":
        mv = 4 * streams
        for shape, c in MAIN.items():
            out.append((shape, mv, c, "main"))
        for step_m in (mv, streams, streams):
            for shape, c in DRAFT.items():
                out.append((shape, step_m, c, "draft"))
            out.append(("h2560", min(4 * step_m, 64), 1, "draft"))  # fc_hidden on hc_count x rows
            out.append(("lm_head", streams, 1, "draft"))
    else:
        for shape, c in MAIN.items():
            out.append((shape, streams, c, "main"))
    return out


def measured_t(mode, streams):
    ref = MTP3_T if mode == "mtp3" else MTP0_T
    return ref[1] + (ref[8] - ref[1]) * (streams - 1) / 7


def project(t):
    out = {}
    for mode in ("mtp3", "mtp0"):
        for streams in (1, 2, 6, 8):
            sched = schedule(mode, streams)
            base, _ = dense_ms(t, OPTIONS["BF16 F.linear (today)"], sched)
            t_meas = measured_t(mode, streams)
            tok_iter = MTP3_TOKENS_PER_ITER if mode == "mtp3" else 1.0
            gb = sum(calls * SHAPE_BYTES[s] for s, m, calls, role in sched) / 1e9
            row = {"measured_iter_ms": t_meas, "bench_dense_bf16_ms": base, "dense_gb_bf16": gb,
                   "tok_s_per_stream_today": tok_iter / t_meas * 1000, "options": {}}
            for name, option in OPTIONS.items():
                if mode == "mtp0" and ("draft" in name.split("+")[-1] or "32K" in name):
                    continue
                dense, parts = dense_ms(t, option, sched)
                t_new = t_meas - base + dense
                row["options"][name] = {
                    "dense_ms": dense, "iter_ms": t_new, "tok_s_per_stream": tok_iter / t_new * 1000,
                    "gain": t_meas / t_new - 1, "parts": parts}
            out[f"{mode}-s{streams}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows, meta = load(args.bench_dir)
    t = table(rows)
    for tag, m in meta.items():
        print(tag, "elapsed", m["elapsed_seconds"], "finished", m["finished"], "errors", len(m["errors"] or []))
        for e in (m["errors"] or [])[:8]:
            print("   ERR", e.get("shape"), e.get("M"), e.get("method"), e.get("error", "")[:160])
        for n in (m["notes"] or [])[:8]:
            print("   NOTE", n[:200])
    bw = next((m["bandwidth"] for m in meta.values() if m.get("bandwidth")), {})
    print("bandwidth", json.dumps(bw))
    shapes = []
    for r in rows:
        if r["shape"] not in shapes:
            shapes.append(r["shape"])
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    for m in (1, 4, 8, 24):
        print(f"\n#### M = {m}: us per call (TB/s of stored weight bytes)\n")
        print("| shape | " + " | ".join(methods) + " |")
        print("| --- |" + " ---: |" * len(methods))
        for s in shapes:
            cells = []
            for meth in methods:
                r = t.get((s, meth), {}).get(m)
                cells.append("" if r is None else f"{r['us']:.1f} ({r['tbps']:.2f})")
            print(f"| {s} | " + " | ".join(cells) + " |")
    print("\n#### relative output error at M=8 (real checkpoint weights, Gaussian activations)\n")
    print("| shape | " + " | ".join(methods) + " |")
    print("| --- |" + " ---: |" * len(methods))
    for s in shapes:
        cells = []
        for meth in methods:
            r = t.get((s, meth), {}).get(8)
            cells.append("" if r is None or r.get("rel_err") is None else f"{r['rel_err'] * 100:.2f}%")
        print(f"| {s} | " + " | ".join(cells) + " |")
    proj = project(t)
    print("\n#### projections\n")
    for key, row in proj.items():
        print(f"{key}: measured {row['measured_iter_ms']:.1f} ms, bench dense BF16 {row['bench_dense_bf16_ms']:.2f} ms "
              f"for {row['dense_gb_bf16']:.2f} GB, today {row['tok_s_per_stream_today']:.0f} tok/s/stream")
        for name, o in row["options"].items():
            print(f"   {name:62s} dense {o['dense_ms']:6.2f} ms  iter {o['iter_ms']:6.2f} ms  "
                  f"{o['tok_s_per_stream']:6.1f} tok/s  {o['gain'] * 100:+5.1f}%")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"projections": proj, "bandwidth": bw,
                       "env": next((m["env"] for m in meta.values() if m.get("env")), {}),
                       "passes": {k: {"elapsed_seconds": v["elapsed_seconds"], "finished": v["finished"],
                                      "errors": v["errors"], "notes": v["notes"]} for k, v in meta.items()},
                       "act_quant": next((m["act_quant"] for m in meta.values() if m.get("act_quant")), []),
                       "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
