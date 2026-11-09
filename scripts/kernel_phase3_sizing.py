#!/usr/bin/env python3
"""Phase-3 sizing: expected ms saved per decode iteration and per-stream / aggregate tok/s for each remaining
throughput lever, on top of the A+C dense-kernel overlay (skinny GEMM + INT8 W8A16 on lm_head, GDN in_proj_qkvz,
QSA qkv, attention out).

Inputs (all CPU-side, already in the repo):
  docs/validation-20260927/kernels-phase1-bench.json   phase-1 GEMM microbenchmark rows (per shape, method, M)
  docs/validation-20260927/kernels-phase3-trace.json   whole-iteration kernel census (scripts/kernel_phase3_trace.py)

Baselines with A+C:
  MTP-0  measured (phase 2, bench decode-only): 103.2 tok/s at 1 stream, 66.3 per stream at 6; 2 streams linear.
  MTP-3  projected: lab iteration (21.3 ms at 1 stream, 34.2 at 8; linear between) minus the A+C dense saving
         from the phase-1 rows, scaled by k = measured / projected A+C saving on MTP-0 at 1 stream.
  MTP-1  projected the same way from the lab's MTP-1 points (105 tok/s at 1 stream, 63 per stream at 8).

Every lever is a saving in ms per decode iteration with a low / central / high range; the derivation of each
number is in LEVERS below and in docs/validation-20260927/kernels-phase3-sizing.md.

Usage: python scripts/kernel_phase3_sizing.py [--json OUT]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kernel_bench_analyze as p1  # noqa: E402

DOC = ROOT / "docs/validation-20260927"
STREAMS = (1, 2, 6)
INT8_FAMILIES = ("lm_head", "gdn_in_qkvz", "qsa_qkv", "attn_out")
# tokens per iteration (lab acceptance: MTP-1 0.71-0.86, MTP-3 1.35-2.04 accepted per draft)
TOKENS = {"mtp0": 1.0, "mtp1": 1.78, "mtp3": 2.6}
MTP0_AC = {1: 1000 / 103.2, 6: 1000 / 66.3}  # phase 2, A+C server, decode-only
MTP0_REF = {1: 1000 / 81.6, 6: 1000 / 56.0}  # phase 2, control server (same container)
MTP1_T = {1: 1.78 / 105 * 1000, 8: 1.78 / 63 * 1000}  # serving lab mtp1 arm (4K/16K mean)

# --------------------------------------------------------------------------- NvFp4 experts (lever 1)
EXPERT_MB = (1280 * 2560 + 2560 * 640) * (0.5 + 1 / 16) / 1e6  # 2.765 MB per routed expert
CEIL_TBPS = 1.527
US_PER_EXPERT_AT_CEIL = EXPERT_MB / CEIL_TBPS  # 1.81 us


def distinct(tokens):
    return 512 * (1 - (1 - 10 / 512) ** tokens)


def ac_dense_ms(t, sched):
    """Dense ms of the phase-2 A+C server over a phase-1 schedule: INT8 Triton W8A16 on the four families (main
    and draft roles; the draft shares the target lm_head), the CuTe skinny GEMM at M <= 16 elsewhere, cuBLAS
    otherwise (HC up stays on cuBLAS, as served)."""
    total = 0.0
    for shape, m, calls, role in sched:
        if shape in INT8_FAMILIES:
            us = p1.time_at(t, shape, "triton_w8a16_int8", m)
        elif m <= 16 and shape != "hc_up" and p1.time_at(t, shape, "cute_skinny_bf16", m) is not None:
            us = p1.time_at(t, shape, "cute_skinny_bf16", m)
        else:
            us = p1.time_at(t, shape, "bf16_linear", m)
        total += us * calls
    return total / 1000


def bf16_dense_ms(t, sched):
    return sum(p1.time_at(t, s, "bf16_linear", m) * c for s, m, c, _ in sched) / 1000


def schedule(mode, streams):
    if mode in ("mtp0", "mtp3"):
        return p1.schedule(mode, streams)
    out = [(shape, 2 * streams, c, "main") for shape, c in p1.MAIN.items()]  # MTP-1: verify 2 tokens / request
    for shape, c in p1.DRAFT.items():
        out.append((shape, 2 * streams, c, "draft"))
    out.append(("h2560", min(8 * streams, 64), 1, "draft"))
    out.append(("lm_head", streams, 1, "draft"))
    return out


def baselines(t):
    """Iteration ms with A+C for each (mode, streams)."""
    sched = schedule("mtp0", 1)
    projected_saving = bf16_dense_ms(t, sched) - ac_dense_ms(t, sched)
    measured_saving = MTP0_REF[1] - MTP0_AC[1]
    k = measured_saving / projected_saving
    out = {"calibration": {"mtp0_s1_projected_saving_ms": projected_saving,
                           "mtp0_s1_measured_saving_ms": measured_saving, "k": k}}
    for mode in ("mtp0", "mtp1", "mtp3"):
        for s in STREAMS:
            if mode == "mtp0":
                t_ac = MTP0_AC[1] + (MTP0_AC[6] - MTP0_AC[1]) * (s - 1) / 5
                src = "measured" if s in (1, 6) else "interpolated"
                t_today = MTP0_REF[1] + (MTP0_REF[6] - MTP0_REF[1]) * (s - 1) / 5
            else:
                ref = p1.MTP3_T if mode == "mtp3" else MTP1_T
                t_today = ref[1] + (ref[8] - ref[1]) * (s - 1) / 7
                sched = schedule(mode, s)
                t_ac = t_today - k * (bf16_dense_ms(t, sched) - ac_dense_ms(t, sched))
                src = "projected"
            out[f"{mode}-s{s}"] = {"today_ms": t_today, "ac_ms": t_ac, "source": src,
                                   "tok_s_per_stream": TOKENS[mode] / t_ac * 1000,
                                   "aggregate": s * TOKENS[mode] / t_ac * 1000}
    return out


# --------------------------------------------------------------------------- lever models (ms / iteration)
def lever_moe_fused(mode, s, case):
    """b12x fused NvFp4 MoE (FlashInfer 0.6.16 SM12x micro/static kernels) instead of the FlashInfer CUTLASS path.
    Saves the per-layer glue measured in the traces (prefix sums, expand, strides, activation, fill, memset:
    20.2 us/layer at M = 1, minus 3 us of PDL overlap) less the fused kernel's own fixed cost (compact pre-pass,
    internal barrier: 4 us assumed), and at 1-2 tokens the grouped GEMM's low efficiency (0.84 / 0.62 TB/s for
    gate_up / down at M = 1: 36.7 us for 27.6 MB) against the micro kernel at eta x the read ceiling."""
    glue = {"low": 9.0, "central": 13.2, "high": 16.0}[case]
    eta = {"low": 0.5, "central": 0.65, "high": 0.8}[case]
    tokens = s if mode == "mtp0" else (4 * s if mode == "mtp3" else 2 * s)
    gemm_gain = 0.0
    if tokens <= 2:
        e = distinct(tokens)
        current = 36.7 + (e - 10) * (74.3 - 36.7) / (distinct(4) - 10)  # trace: 36.7 us at M=1, 74.3 at M=4
        gemm_gain = max(0.0, current - e * US_PER_EXPERT_AT_CEIL / eta)
    return 48 * (glue + gemm_gain) / 1000


def lever_hc_fusion(mode, s, case):
    """Two fused kernels per hyper-connection mix (combine+norm+down GEMV with the silu epilogue; up GEMV with the
    gate-mix epilogue) instead of five launches: 17.1 us per mix on the M = 1 trace, 13.4 MB of BF16 weights
    (floor 8.8 us); the fused pair is taken at 11.4 us (central). Batch-independent (weight reads)."""
    per = {"low": 3.5, "central": 5.7, "high": 7.5}[case]
    # HC up calls per iteration: 97 in the target forward, 3 per MTP draft step
    count = {"mtp0": 97, "mtp1": 100, "mtp3": 106}[mode]
    return count * per / 1000


def lever_small_fusion(mode, s, case):
    """Router GEMV + softmax/top-10 in one kernel (topkGating is 6.2 us per layer at M = 1), GDN zeros + gated norm
    folded into the recurrent kernel, residual adds into neighbours."""
    return {"low": 0.1, "central": 0.2, "high": 0.3}[case] * (1.0 if mode == "mtp0" else 1.1)


def lever_ple_gap(mode, s, case):
    """The PLE CPU-offload round trip: the GPU waits at the PLE semaphore 0.68-0.90 ms per iteration (profiled
    period equals the unprofiled decode rate, so the wait is real). Shortening the round trip (busy-poll worker,
    no ZMQ hop, launch layer 0 before the wait) is assumed to recover about half of it."""
    return {"low": 0.2, "central": 0.4, "high": 0.6}[case]


def lever_sampler(mode, s, case):
    """Top-k-first sampling instead of the full 248K-vocabulary sort/scan (MTP-0: 0.28 ms span at 1 stream;
    MTP-3: 0.59 ms of sampler kernels, 0.29 ms of it one fp32 cumsum)."""
    base = {"mtp0": 0.2, "mtp1": 0.3, "mtp3": 0.45}[mode] * (1 + 0.05 * (s - 1))
    return base * {"low": 0.6, "central": 1.0, "high": 1.2}[case]


def lever_option_b(mode, s, case):
    """MTP draft head on a 32K sub-vocabulary (INT8 like the shared A+C head): per draft step the INT8 lm_head read
    (phase 1: 419 us at M = 1) becomes 32K/248K of it plus a -inf fill; plus the MTP layer's 512 BF16 experts
    stored as NvFp4 (0.44 ms of draft MoE per iteration at 1 stream, 1.85 ms at 8, cut 3.55x in bytes)."""
    if mode == "mtp0":
        return 0.0
    steps = 3 if mode == "mtp3" else 1
    head = steps * 0.419 * (1 - 32768 / 248320) - steps * 0.003
    draft_moe = {1: 0.444, 2: 0.66, 6: 1.5}[s] * (steps / 3) * (1 - 1 / 3.55)
    part = {"low": 0.7, "central": 1.0, "high": 1.0}[case]
    return head * part + draft_moe * {"low": 0.0, "central": 1.0, "high": 1.0}[case]


LEVERS = {
    "1. b12x fused NvFp4 MoE": lever_moe_fused,
    "2a. HC mix fusion (2 kernels)": lever_hc_fusion,
    "2b. router/GDN/residual fusion": lever_small_fusion,
    "5a. PLE round-trip wait": lever_ple_gap,
    "5b. top-k-first sampler": lever_sampler,
    "4. Option B (32K draft head + NvFp4 draft experts)": lever_option_b,
}


# --------------------------------------------------------------------------- lever 3: admission (draft-state memory)
# Pool geometry from the pinned vLLM (kv_cache_utils._get_kv_cache_groups_csa_linear) and the lab's ARC_CACHE_LAYOUT
# lines: one block id spans 13 (12 on MTP-0) main-KV + compressed-indexer pages; GDN (3 groups of 12 layers) and
# PLE short-conv (1 group) state pages share the pool. Align mode reserves 2 + num_spec pages per mamba group
# (MambaSpec.max_memory_usage_bytes), plus one ring-buffer block, plus cdiv(L, block) attention blocks.
# Block geometry is derived as in platforms/interface.py (non-"all" mamba mode): the GDN page (conv state
# 10240 x (3 + ns) x 2 B + recurrent state 48 x 128 x 128 x state bytes) sets block = 16 * cdiv(page, 16 * 2048)
# tokens. The lab logs confirm: MTP-3 1600 (fp32 state), MTP-1 1584 (fp32), MTP-0 800 (bf16 state).
NS = {"mtp0": 0, "mtp1": 1, "mtp3": 3}
DEFAULT_STATE_BYTES = {"mtp0": 2, "mtp1": 4, "mtp3": 4}
POOL_BYTES = {  # lab / arena KV pools at utilization 0.96 with BF16 non-expert weights
    "mtp3": 222 * 13 * (1600 * 2048 + 400 * 256),
    "mtp1": 225 * 13 * (1584 * 2048 + 396 * 256),
    "mtp0": 751 * 12 * (800 * 2048 + 200 * 256),
}
GATE = 0.9  # the arena's admission margin
GIB = 1 << 30
FREED = {  # weight bytes that can move to the KV pool
    "ac": 2.4 * GIB,  # phase 2: the A+C server holds 2.4 GiB less weight (pinned away in the phase-2 runs)
    "draft_experts_fp8": 4.69 * GIB * 0.5,  # MTP layer: 512 BF16 experts (4.69 GiB) stored as FP8
    "draft_experts_nvfp4": 4.69 * GIB * (1 - 0.5625 / 2),  # ... or as NvFp4 (0.5625 B/param)
}


def geometry(mode, state_bytes=None):
    ns = NS[mode]
    state_bytes = state_bytes or DEFAULT_STATE_BYTES[mode]
    page = 10240 * (3 + ns) * 2 + 48 * 128 * 128 * state_bytes
    tokens = 16 * -(-page // (16 * 2048))
    layers = 12 if ns == 0 else 13  # QSA layers (+ the MTP layer) sharing one block id
    return {"block_tokens": tokens, "block_bytes": layers * (tokens * 2048 + tokens // 4 * 256), "ns": ns}


def per_request_blocks(mode, context, design, state_bytes=None):
    g = geometry(mode, state_bytes)
    attention = -(-context // g["block_tokens"]) + 1  # growing main+indexer group + the raw-key ring block
    if mode == "mtp0" or design == "today":
        return attention + 4 * (2 + g["ns"])
    if design == "ple_fix":  # PLE group reserves no draft pages (they are never touched)
        return attention + 3 * (2 + g["ns"]) + 2
    if design == "replay":  # one running state per group; drafts replayed from a small input ring
        return attention + 4 * 2
    raise ValueError(design)


def fits(mode, context, design, freed=(), state_bytes=None):
    g = geometry(mode, state_bytes)
    blocks = int((POOL_BYTES[mode] + sum(FREED[f] for f in freed)) // g["block_bytes"])
    need = per_request_blocks(mode, context, design, state_bytes)
    return {"block_tokens": g["block_tokens"], "block_mb": g["block_bytes"] / 1e6, "pool_blocks": blocks,
            "gate_blocks": int(blocks * GATE), "per_request": need, "fit": int(blocks * GATE) // need}


def admission(base, stacked_saving):
    """Requests that fit at 16/32/64K, and per-stream / aggregate tok/s at min(fit, 6) streams (the arena's six
    focus games), with A+C and optionally all central ms levers."""
    rows = []
    fp8, nvfp4 = "draft_experts_fp8", "draft_experts_nvfp4"
    scenarios = [  # (mode, design, freed weights, SSM state bytes: None = the server default)
        ("mtp0", "today", (), None), ("mtp0", "today", ("ac",), None),
        ("mtp3", "today", (), None), ("mtp3", "today", ("ac",), None), ("mtp3", "ple_fix", ("ac",), None),
        ("mtp3", "today", ("ac", nvfp4), None), ("mtp3", "today", ("ac",), 2), ("mtp3", "today", ("ac", fp8), 2),
        ("mtp3", "today", ("ac", nvfp4), 2), ("mtp3", "replay", ("ac",), None),
        ("mtp3", "replay", ("ac", fp8), None), ("mtp3", "replay", ("ac", nvfp4), 2),
        ("mtp1", "today", (), None), ("mtp1", "today", ("ac",), None), ("mtp1", "today", ("ac", fp8), None),
        ("mtp1", "today", ("ac", nvfp4), None), ("mtp1", "today", ("ac",), 2), ("mtp1", "today", ("ac", fp8), 2),
        ("mtp1", "replay", ("ac", nvfp4), 2),
    ]
    for mode, design, freed, state_bytes in scenarios:
        row = {"mode": mode, "design": design, "freed": list(freed),
               "ssm_state": {None: "default", 2: "bf16", 4: "fp32"}[state_bytes]}
        for ctx in (16384, 32768, 65536):
            f = fits(mode, ctx, design, freed, state_bytes)
            s = max(1, min(f["fit"], 6))
            t1, t6 = base[f"{mode}-s1"]["ac_ms"], base[f"{mode}-s6"]["ac_ms"]
            t = t1 + (t6 - t1) * (s - 1) / 5
            t_all = t - stacked_saving(mode, s)
            row[f"{ctx // 1024}K"] = {**f, "streams": s, "iter_ms": t,
                                      "tok_s_per_stream": TOKENS[mode] / t * 1000,
                                      "aggregate": s * TOKENS[mode] / t * 1000,
                                      "all_levers_tok_s_per_stream": TOKENS[mode] / t_all * 1000,
                                      "all_levers_aggregate": s * TOKENS[mode] / t_all * 1000}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    bench = json.loads((DOC / "kernels-phase1-bench.json").read_text())
    t = p1.table(bench["rows"])
    base = baselines(t)
    print("calibration", {k: round(v, 3) for k, v in base["calibration"].items()})
    out = {"baselines": base, "levers": {}, "stacked": {}}
    print("\n| mode | streams | today ms | A+C ms | source | A+C tok/s per stream | aggregate |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: |")
    for key, row in base.items():
        if key == "calibration":
            continue
        mode, s = key.split("-s")
        print(f"| {mode} | {s} | {row['today_ms']:.2f} | {row['ac_ms']:.2f} | {row['source']} | "
              f"{row['tok_s_per_stream']:.0f} | {row['aggregate']:.0f} |")
    for name, fn in LEVERS.items():
        out["levers"][name] = {}
        print(f"\n{name}")
        for mode in ("mtp0", "mtp3", "mtp1"):
            for s in STREAMS:
                b = base[f"{mode}-s{s}"]
                cells = {}
                for case in ("low", "central", "high"):
                    saved = fn(mode, s, case)
                    t_new = b["ac_ms"] - saved
                    cells[case] = {"ms_saved": saved, "iter_ms": t_new,
                                   "tok_s_per_stream": TOKENS[mode] / t_new * 1000,
                                   "aggregate": s * TOKENS[mode] / t_new * 1000,
                                   "gain": b["ac_ms"] / t_new - 1}
                out["levers"][name][f"{mode}-s{s}"] = cells
                c = cells["central"]
                print(f"  {mode} s{s}: saves {cells['low']['ms_saved']:.2f}/{c['ms_saved']:.2f}/"
                      f"{cells['high']['ms_saved']:.2f} ms -> {c['tok_s_per_stream']:.0f} tok/s per stream "
                      f"({c['gain'] * 100:+.0f}%), aggregate {c['aggregate']:.0f}")
    print("\nall ms/iter levers stacked (central)")
    for mode in ("mtp0", "mtp3", "mtp1"):
        for s in STREAMS:
            b = base[f"{mode}-s{s}"]
            saved = sum(fn(mode, s, "central") for fn in LEVERS.values())
            t_new = b["ac_ms"] - saved
            out["stacked"][f"{mode}-s{s}"] = {"ms_saved": saved, "iter_ms": t_new,
                                             "tok_s_per_stream": TOKENS[mode] / t_new * 1000,
                                             "aggregate": s * TOKENS[mode] / t_new * 1000}
            per_stream = TOKENS[mode] / t_new * 1000
            print(f"  {mode} s{s}: {b['ac_ms']:.2f} -> {t_new:.2f} ms, {per_stream:.0f} tok/s per stream "
                  f"({b['ac_ms'] / t_new - 1:+.0%}), aggregate {s * per_stream:.0f}")
    def stacked(mode, s):
        # levers modelled at 1/2/6 streams; linear in between
        lo = max(x for x in STREAMS if x <= s)
        hi = min(x for x in STREAMS if x >= s)
        a = sum(fn(mode, lo, "central") for fn in LEVERS.values())
        b = sum(fn(mode, hi, "central") for fn in LEVERS.values())
        return a if hi == lo else a + (b - a) * (s - lo) / (hi - lo)

    out["admission"] = admission(base, stacked)
    print("\nadmission (gate 0.9): fit at 16K / 32K / 64K; at min(fit, 6) streams per-stream / aggregate tok/s "
          "(A+C; all levers)")
    for row in out["admission"]:
        cells = []
        for ctx in ("16K", "32K", "64K"):
            c = row[ctx]
            cells.append(f"{ctx}: {c['fit']} ({c['per_request']} blk of {c['gate_blocks']}) -> s{c['streams']} "
                         f"{c['tok_s_per_stream']:.0f}/{c['aggregate']:.0f}; {c['all_levers_tok_s_per_stream']:.0f}/"
                         f"{c['all_levers_aggregate']:.0f}")
        print(f"  {row['mode']} {row['design']:8s} {row['ssm_state']:7s} {'+'.join(row['freed']) or '-':28s} "
              + " | ".join(cells))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
