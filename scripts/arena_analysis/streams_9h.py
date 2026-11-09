"""More concurrent streams at shorter windows: the 9-hour hidden run and the one-hour c110 (28 September).

Drives the calibrated simulator (``schedule_sim.sweep`` through ``c110_path``) over serving profiles that trade
window length for concurrency: MTP-0 + dense kernels at 6 x 64K (measured in c110-kernels-v2), 8 / 12 x 32K and
16 x 16K (projected from the serving lab), MTP-3 phase 4 at 6 x 64K (smoke) and 10 x 32K, and an 8-bit attention
KV cache (12 x 64K on MTP-0, 10 x 64K on MTP-3; hypothetical: QSA needs a BF16 KV cache today).

A shorter window compacts more often. The recordings show no measurable per-level cost of compaction
(prefill-and-compaction.md), so the window enters only as a sensitivity parameter: a clear-rate multiplier on
every level (``efficiency``) of 1.0 / 0.9 / 0.75 for 32K and 16K windows (and 0.56 = 0.75^2 for 16K).

    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/streams_9h.py \
        --attempts <attempts-now.jsonl> <stage> [--runs N]

Stages: ``nine`` (9 h, 110 distinct games), ``nine_boot`` (23-game bootstrap), ``nine_sens`` (clear rates x0.5 /
x1.33), ``nine_halves`` (held-out halves of the 20 games), ``hour`` (one-hour c110), ``hour_boot``, ``hour_sens``.
Each writes ``<name>.config.json`` / ``.result.json`` to ``docs/validation-20260927/streams-9h/``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c110_path as cp  # noqa: E402
import schedule_sim as sim  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "validation-20260927" / "streams-9h"

# ---------------------------------------------------------------------------------------------- serving profiles
#
# Phase-2 decode per iteration on MTP-0 (serving lab 16K series and the kernel phase-2 control, which agree at
# 1 stream: 12.35 / 12.38 ms): t0(n) = 12.35 / 18.2 / 19.2 / 23.6 ms at 1 / 6 / 8 / 16 streams (16K prompts;
# 12 and 10 interpolated linearly: 21.4 / 20.3). The dense kernels (A+C) save a fixed ~2.85 ms per iteration at
# 6 streams (18.2 -> 15.35 ms, 391 tok/s) -- the saving is in the dense weight reads, which do not grow with the
# batch -- so the kernel rate at n streams is n / (t0(n) - 2.85 ms): +19% at 6, +18% at 8, +15% at 12, +14% at 16.
# The arena's phase-2 rate is 0.99 x the bench (c110-kernels-v2: 387 tok/s at 6.2 running against 391).
#
# MTP-3 phase 4 (kernels-phase4.md): 6 streams on replayed arena transcripts 81.9 per stream (492 aggregate;
# synthetic 16K 88.4 / 530). At 10 streams: the verify batch grows from 24 to 40 tokens; on the MTP-0 slope at
# >= 16 tokens (0.24 ms per token) plus a draft that grows with the batch, ~36 ms per iteration against 30.5, so
# ~70 per stream; a linear extrapolation of the 1 -> 6 stream growth gives ~57. Central 64 per stream (640).
#
# The opening (110 games, ~18 running) is measured on MTP-0 + kernels (541.6 tok/s; busy clock 31.2 tok/s per
# game) and does not depend on the phase-2 slot count; the MTP-3 opening is taken as 1.1x (MTP gains little at
# ~18 streams, and the pool is the same size). busy_p2 = per-stream rate x 63.5 / 64.5 (kernels-v2 busy clock).

T0 = {1: 12.35, 6: 18.2, 8: 19.2, 10: 20.3, 12: 21.4, 16: 23.6}
KERNEL_MS = 2.85
ARENA = 387.2 / (6 / (T0[6] - KERNEL_MS) * 1000)  # 0.99
OPEN_K, BUSY_OPEN_K = 541.6, 31.2
BUSY_RATIO = 63.5 / 64.5


def mtp0k(n):
    """Arena phase-2 tokens per wall second on MTP-0 + A+C at ``n`` streams."""
    return ARENA * n / (T0[n] - KERNEL_MS) * 1000


def profile(p2, n, opening=OPEN_K, busy_open=BUSY_OPEN_K, window=64):
    s = p2 / n
    return {"open": opening, "p2": round(p2, 1), "slots": n, "busy_open": busy_open,
            "busy_p2": round(s * BUSY_RATIO, 2), "stream": round(s, 2), "window": window}


M3_OPEN, M3_BUSY_OPEN = OPEN_K * 1.1, BUSY_OPEN_K * 1.1
PROFILES = {
    # references (c110-path-to-3pct.md)
    "MTP-3 2 slots (old base)": {**cp.PROFILES["260/2 (MTP-3, measured)"], "window": 128},
    "MTP-0 6x64K": {**cp.PROFILES["390/6 (MTP-0 64K, measured)"], "window": 64},
    # MTP-0 + dense kernels (A+C)
    "K 6x64K": {"open": OPEN_K, "p2": 387.2, "slots": 6, "busy_open": BUSY_OPEN_K, "busy_p2": 63.5, "stream": 64.5,
                "window": 64},  # measured, c110-kernels-v2
    "K 8x32K": profile(mtp0k(8), 8, window=32),  # = 8 x 64K with the pool unpinned at multiplier 1
    "K 12x32K": profile(mtp0k(12), 12, window=32),
    "K 16x16K": profile(mtp0k(16), 16, window=16),
    # MTP-3 phase 4 (bf16 state, NvFp4 draft experts, A+C)
    "M3 6x64K": profile(ARENA * 492.0, 6, M3_OPEN, M3_BUSY_OPEN),
    "M3 10x32K": profile(640.0, 10, M3_OPEN, M3_BUSY_OPEN, window=32),
}
# 8-bit attention KV (hypothetical): 12 x 64K on MTP-0 + kernels, 10 x 64K on MTP-3 phase 4. The same concurrency
# and per-stream speed as "K 12x32K" / "M3 10x32K", without the shorter window: those profiles' multiplier-1.0
# cells. Likewise 8 x 64K on MTP-0 + kernels with the pool unpinned (kernels-phase3-sizing.md) is "K 8x32K" at 1.0.
ALIASES = {"K+KV8 12x64K": "K 12x32K", "M3+KV8 10x64K": "M3 10x32K", "K 8x64K (pool unpinned)": "K 8x32K"}
MULTS = {64: (1.0,), 128: (1.0,), 32: (1.0, 0.9, 0.75), 16: (1.0, 0.9, 0.75, 0.56)}


def cells(names=None, mults=None):
    """(label, profile, window multiplier) for every profile and its window multipliers."""
    out = []
    for name, p in PROFILES.items():
        if names and name not in names:
            continue
        for m in MULTS[p["window"]]:
            if mults is not None and m not in mults and m != 1.0:
                continue
            out.append((name if m == 1.0 else f"{name} m{m:g}", p, m))
    return out


# ---------------------------------------------------------------------------------------------- policies

def policies(p, skw, which):
    """``which``: names from fixed-2x, gittins-h0.15, gittins-h0.15-open20, gittins-c110, gittins-c110-open20."""
    idx = {**skw, "opening_max_wall": 1200}
    table = {
        "fixed-2x": [*cp.fixed2x(p), skw],
        # the planned wide arms keep the mtp0-wide six-clear opening switch (kernels-v2 ran with it)
        "fixed-2x sw6": [*cp.fixed2x(p), {**skw, "switch_games": 6}],
        "gittins-h0.15":[*cp.gittins(p, hysteresis=0.15, prior_scale="fixed"), idx],
        "gittins-h0.15-open20": [*cp.gittins(p, hysteresis=0.15, prior_scale="fixed"), {**idx, "opening_games": 20}],
        "gittins-c110": [*cp.gittins(p), idx],
        "gittins-c110-open20": [*cp.gittins(p), {**idx, "opening_games": 20}],
        "known-rate": ["known-rate", {}, skw],
    }
    return {k: table[k] for k in which}


NINE_POL = ("fixed-2x", "gittins-h0.15", "gittins-h0.15-open20", "gittins-c110-open20")
HOUR_POL = ("fixed-2x", "gittins-c110", "gittins-c110-open20", "known-rate")
KEY9 = ["MTP-0 6x64K", "K 6x64K", "K 8x32K", "K 12x32K", "K 16x16K", "M3 6x64K", "M3 10x32K"]


def build(cell_list, which, share, scale=1.0, **extra):
    pol = {}
    for label, p, m in cell_list:
        skw = cp.cal_sim(p, share, efficiency=m * scale, shuffle=True, **extra)
        for k, v in policies(p, skw, which).items():
            pol[f"{label} | {k}"] = v
    return pol


def nine(runs, share):
    wall = {"wall": 32400.0, "shared": False}
    return {"nine": {**cp.CAL, "runs": runs or 16, "seed": 6, "policies": build(cells(), NINE_POL, share, **wall)}}


def nine_boot(runs, share):
    b = {"pool": sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}
    sel = cells(KEY9, mults=(0.75,))
    return {"nine-boot23": {**cp.CAL, "roster": "23", "runs": 24, "seed": 21,
                            "policies": build(sel, ("fixed-2x", "gittins-h0.15"), share, wall=32400.0, shared=False,
                                              bootstrap=b)}}


def nine_sens(runs, share):
    sel = cells(KEY9, mults=(0.75,))
    out = {}
    for s in (0.5, 1.33):
        out[f"nine-x{s:g}"] = {**cp.CAL, "runs": runs or 16, "seed": 22, "policies": build(
            sel, ("fixed-2x", "gittins-h0.15"), share, scale=s, wall=32400.0, shared=False)}
    return out


def nine_halves(runs, share):
    """The same comparison on 110 distinct games drawn from one half of the 20 source games (the four splits of
    c110-path/heldout-fits.json): does the ranking of profiles hold on games the other half never saw?"""
    fits = json.loads(Path(cp.HERE, "heldout-fits.json").read_text())
    sel = cells(KEY9, mults=(0.75,))
    out = {}
    for s, split in enumerate(fits):
        pol = {}
        for r, half in enumerate(split["halves"]):
            roster = [half[i % 10] for i in range(110)]
            for label, p, m in sel:
                skw = cp.cal_sim(p, share, efficiency=m, shuffle=True, wall=32400.0, shared=False, roster_games=roster)
                for k, v in policies(p, skw, ("fixed-2x", "gittins-h0.15")).items():
                    pol[f"H{r} {label} | {k}"] = v
        out[f"nine-halves-s{s}"] = {**cp.CAL, "runs": runs or 6, "seed": 23 + s, "halves": split["halves"],
                                    "policies": pol}
    return out


def hour(runs, share):
    return {"hour": {**cp.CAL, "runs": runs or 40, "seed": 41, "policies": build(cells(), HOUR_POL, share)}}


def hour_boot(runs, share):
    b = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": cp.BOOT_PER}
    sel = cells(KEY9, mults=(0.75,))
    return {"hour-boot": {**cp.CAL, "runs": cp.BOOT_ROSTERS * cp.BOOT_PER, "seed": 11, "policies": build(
        sel, ("fixed-2x", "gittins-c110-open20"), share, bootstrap=b)}}


def hour_sens(runs, share):
    sel = cells(KEY9, mults=(0.75,))
    out = {}
    for s in (0.5, 1.33):
        out[f"hour-x{s:g}"] = {**cp.CAL, "runs": runs or 40, "seed": 12, "policies": build(
            sel, ("fixed-2x", "gittins-c110-open20"), share, scale=s)}
    return out


# ---------------------------------------------------------------------------------------------- int8 KV (28 Sept, 2)
#
# The int8 QSA KV cache works (kv-cache-8bit.md): drift at the noise floor, decode per iteration unchanged, so a 64K
# window on int8 carries no penalty. Fit on the MTP-0 + A+C pool as served in kernels-v2 (unpinned, 875 x 800-token
# BF16 blocks): 845 int8 blocks of 1,584 tokens, 760 at the 0.9 gate; a request costs ceil((window + 256) / 1584)
# + 9 blocks (scripts/kv8_sizing.py geometry, recomputed in streams-9h.md §8): 64K 51 -> 14 fit, 48K 41 -> 18,
# 21 slots <= 36 blocks -> 42,512-token windows (41,344 if the block rule pads to 1,600 tokens).
#
# Decode beyond 16 streams: the lab's 4K series flattens (0.84 ms per extra stream from 8 to 16, 0.24 from 16 to
# 24), so the 16K series continues past 16 at KV8_SLOPE ms per stream (central 0.30; range 0.225, the 4K series'
# relative flattening, to 0.55, the 8 -> 16 slope continued). int8 prefill ran 15% slower (10.7K vs 12.9K tok/s) on
# ~12% of GPU time: x0.98 on phase 2. The opening is kept at the measured 541.6 (its prefill is 29% of GPU time, so
# -5%, but the doubled pool removes much of its LRU-eviction re-prefill, 60% of its computed tokens; net taken as 0).

KV8_SLOPE = {"central": 0.30, "low": 0.55, "high": 0.225}
KV8_PREFILL = 1 / (1 + 0.12 * (12.9 / 10.7 - 1))  # 0.976


def t0_ext(n, slope):
    if n <= 16:
        keys = sorted(T0)
        for a, b in zip(keys, keys[1:]):
            if a <= n <= b:
                return T0[a] + (T0[b] - T0[a]) * (n - a) / (b - a)
    return T0[16] + (n - 16) * slope


def kv8_rate(n, which="central"):
    return ARENA * KV8_PREFILL * n / (t0_ext(n, KV8_SLOPE[which]) - KERNEL_MS) * 1000


KV8 = {
    "K 6x64K (BF16, measured)": {**PROFILES["K 6x64K"]},
    "KV8 12x64K": profile(kv8_rate(12), 12),
    "KV8 14x64K": profile(kv8_rate(14), 14),
    "KV8 18x48K": profile(kv8_rate(18), 18, window=48),
    "KV8 21x42K": profile(kv8_rate(21), 21, window=42),
}
KV8_MULTS = {64: (1.0,), 48: (1.0, 0.95, 0.9, 0.8), 42: (1.0, 0.95, 0.9, 0.8)}
KV8_KEY = ["KV8 14x64K", "KV8 18x48K", "KV8 21x42K"]
KV8_REF = "KV8 14x64K"


def kv8_cells(mults=None):
    out = []
    for name, p in KV8.items():
        for m in KV8_MULTS[p["window"]]:
            if mults is not None and m != 1.0 and m not in mults:
                continue
            out.append((name if m == 1.0 else f"{name} m{m:g}", p, m))
    return out


def kv8_build(cell_list, which, share, scale=1.0, names=None, **extra):
    pol = {}
    for label, p, m in cell_list:
        if names and not any(label.startswith(n) for n in names):
            continue
        skw = cp.cal_sim(p, share, efficiency=m * scale, shuffle=True, **extra)
        for k, v in policies(p, skw, which).items():
            pol[f"{label} | {k}"] = v
    return pol


KV8_9H = {"wall": 32400.0, "shared": False}
ROBUST_MULTS = (0.9, 0.8)


def kv8_nine(runs, share):
    return {"kv8-nine": {**cp.CAL, "runs": runs or 16, "seed": 6, "policies": kv8_build(
        kv8_cells(), ("fixed-2x sw6", "fixed-2x", "gittins-h0.15"), share, **KV8_9H)}}


def kv8_hour(runs, share):
    return {"kv8-hour": {**cp.CAL, "runs": runs or 40, "seed": 41, "policies": kv8_build(
        kv8_cells(), ("fixed-2x sw6", "fixed-2x", "gittins-c110-open20"), share)}}


def kv8_robust(runs, share):
    sel = kv8_cells(ROBUST_MULTS)
    b9 = {"pool": sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}
    b1 = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": cp.BOOT_PER}
    pol9, pol1 = ("fixed-2x sw6", "gittins-h0.15"), ("fixed-2x sw6", "gittins-c110-open20")
    out = {"kv8-nine-boot23": {**cp.CAL, "roster": "23", "runs": 24, "seed": 21, "policies": kv8_build(
        sel, pol9, share, names=KV8_KEY, bootstrap=b9, **KV8_9H)}}
    for s in (0.5, 1.33):
        out[f"kv8-nine-x{s:g}"] = {**cp.CAL, "runs": 16, "seed": 22, "policies": kv8_build(
            sel, pol9, share, scale=s, names=KV8_KEY, **KV8_9H)}
    out["kv8-hour-boot"] = {**cp.CAL, "runs": cp.BOOT_ROSTERS * cp.BOOT_PER, "seed": 11, "policies": kv8_build(
        sel, pol1, share, names=KV8_KEY, bootstrap=b1)}
    for s in (0.5, 1.33):
        out[f"kv8-hour-x{s:g}"] = {**cp.CAL, "runs": 40, "seed": 12, "policies": kv8_build(
            sel, pol1, share, scale=s, names=KV8_KEY)}
    fits = json.loads(Path(cp.HERE, "heldout-fits.json").read_text())
    for s, split in enumerate(fits):
        pol = {}
        for r, half in enumerate(split["halves"]):
            roster = [half[i % 10] for i in range(110)]
            for k, v in kv8_build(sel, pol9, share, names=KV8_KEY, roster_games=roster, **KV8_9H).items():
                pol[f"H{r} {k}"] = v
        out[f"kv8-nine-halves-s{s}"] = {**cp.CAL, "runs": 6, "seed": 23 + s, "halves": split["halves"],
                                        "policies": pol}
    return out


def kv8_switch(runs, share):
    """One hour: the six-clear against the two-clear opening switch for fixed-2x at 6 / 14 / 21 slots, under the
    game bootstrap and clear rates x0.5 / x1.33."""
    names = ["K 6x64K (BF16, measured)", "KV8 14x64K", "KV8 21x42K"]
    sel = [c for c in kv8_cells(()) if c[0] in names]
    b1 = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": cp.BOOT_PER}
    pol = ("fixed-2x sw6", "fixed-2x")
    out = {"kv8-switch-boot": {**cp.CAL, "runs": cp.BOOT_ROSTERS * cp.BOOT_PER, "seed": 11,
                               "policies": kv8_build(sel, pol, share, bootstrap=b1)}}
    for s in (0.5, 1.33):
        out[f"kv8-switch-x{s:g}"] = {**cp.CAL, "runs": 40, "seed": 12, "policies": kv8_build(sel, pol, share, scale=s)}
    return out


def kv8_report(d):
    """Markdown tables of the int8-KV comparison: main 9 h / 1 h tables, break-even multipliers against 14 x 64K,
    and the robustness summary (paired differences against 14 x 64K, same policy)."""
    out = {}
    for name, idx in (("kv8-nine", "gittins-h0.15"), ("kv8-hour", "gittins-c110-open20")):
        r = _load(d, name)
        if r is None:
            continue
        pols = ("fixed-2x sw6", "fixed-2x", idx)
        print(f"### {name}\n\n| Profile | " + " | ".join(pols) + " | sw6 − 14x64K | index − 14x64K |")
        print("| --- | " + " | ".join("---:" for _ in pols) + " | ---: | ---: |")
        for label, p, m in kv8_cells():
            row = [f"{r[f'{label} | {q}']['mean']:.2f} ({r[f'{label} | {q}']['p10']:.1f}-{r[f'{label} | {q}']['p90']:.1f})"
                   for q in pols]
            diffs = []
            for q in (pols[0], idx):
                if label == KV8_REF:
                    diffs.append("-")
                else:
                    x = _paired(r, f"{label} | {q}", f"{KV8_REF} | {q}")
                    diffs.append(f"{x['diff']:+.2f} ± {x['se']:.2f} ({int(100 * x['p_run'])}%)")
            print(f"| {label} | " + " | ".join(row) + " | " + " | ".join(diffs) + " |")
        be = {}
        for q in pols:
            for base in ("KV8 18x48K", "KV8 21x42K"):
                pts = sorted((m, r[f"{base if m == 1.0 else f'{base} m{m:g}'} | {q}"]["mean"])
                             for m in KV8_MULTS[KV8[base]["window"]])
                target = r[f"{KV8_REF} | {q}"]["mean"]
                val = None
                for (m0, s0), (m1, s1) in zip(pts, pts[1:]):
                    if s0 <= target <= s1 and s1 > s0:
                        val = m0 + (target - s0) * (m1 - m0) / (s1 - s0)
                if val is None and target < pts[0][1]:
                    (m0, s0), (m1, s1) = pts[0], pts[1]
                    val = m0 - (s0 - target) * (m1 - m0) / (s1 - s0)
                if val is None and target > pts[-1][1]:
                    val = "> 1"
                be[f"{base} | {q}"] = val if isinstance(val, str) else round(val, 3)
        out[name] = be
        print(f"\nbreak-even multipliers vs {KV8_REF}: {json.dumps(be, indent=1)}\n")
    checks = [("9 h", "kv8-nine", None, "gittins-h0.15"), ("9 h boot23", "kv8-nine-boot23", 2, "gittins-h0.15"),
              ("9 h x0.5", "kv8-nine-x0.5", None, "gittins-h0.15"), ("9 h x1.33", "kv8-nine-x1.33", None, "gittins-h0.15"),
              ("1 h", "kv8-hour", None, "gittins-c110-open20"),
              ("1 h boot", "kv8-hour-boot", cp.BOOT_PER, "gittins-c110-open20"),
              ("1 h x0.5", "kv8-hour-x0.5", None, "gittins-c110-open20"),
              ("1 h x1.33", "kv8-hour-x1.33", None, "gittins-c110-open20")]
    print("### robustness (paired difference against 14x64K, same policy; share of rosters or runs)\n")
    labels = [c for c, _, _ in kv8_cells(ROBUST_MULTS) if c.startswith(("KV8 18", "KV8 21"))]
    print("| Check | Policy | " + " | ".join(labels) + " |\n| --- | --- | " + " | ".join("---:" for _ in labels) + " |")
    rob = {}
    for title, name, per, idx in checks:
        r = _load(d, name)
        if r is None:
            continue
        for q in ("fixed-2x sw6", idx):
            cellsx = []
            for c in labels:
                if f"{c} | {q}" not in r:
                    cellsx.append("")
                    continue
                x = _paired(r, f"{c} | {q}", f"{KV8_REF} | {q}", per)
                share = x.get("p_roster", x["p_run"])
                cellsx.append(f"{x['diff']:+.2f} ({int(100 * share)}%)")
                rob[f"{name} | {c} | {q}"] = x
            print(f"| {title} | {q} | " + " | ".join(cellsx) + " |")
    halves = {}
    for s in range(4):
        r = _load(d, f"kv8-nine-halves-s{s}")
        if r is None:
            continue
        for h in (0, 1):
            for c in labels:
                for q in ("fixed-2x sw6", "gittins-h0.15"):
                    a, b = f"H{h} {c} | {q}", f"H{h} {KV8_REF} | {q}"
                    if a in r:
                        halves.setdefault(f"{c} | {q}", []).append(_paired(r, a, b)["diff"])
    if halves:
        print("\n### held-out halves (9 h, difference against 14x64K per half)\n")
        print("| Profile | Policy | Mean | Halves positive | Min / max |\n| --- | --- | ---: | ---: | --- |")
        for k, v in halves.items():
            v = np.array(v)
            c, q = k.split(" | ")
            print(f"| {c} | {q} | {v.mean():+.2f} | {int((v > 0).sum())} of {len(v)} | {v.min():+.2f} / {v.max():+.2f} |")
    out["robust"] = rob
    out["halves"] = halves
    Path(d, "kv8-summary.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- report

def _load(d, name):
    path = Path(d, f"{name}.result.json")
    return json.loads(path.read_text()) if path.exists() else None


def _paired(r, a, b, per=None):
    x = np.array(r[a]["scores"]) - np.array(r[b]["scores"])
    out = {"diff": round(float(x.mean()), 2), "se": round(float(x.std() / np.sqrt(len(x))), 2),
           "p_run": round(float(np.mean(x > 0)), 2)}
    if per:
        out["p_roster"] = round(float(np.mean(x.reshape(-1, per).mean(axis=1) > 0)), 2)
    return out


def _split(label):
    cell, pol = label.split(" | ")
    return cell, pol


def main_table(r, ref="K 6x64K"):
    """Markdown: per cell, each policy's mean (p10/p90) and the paired difference against ``ref`` (same policy)."""
    cells_, pols = [], []
    for label in r:
        c, p = _split(label)
        cells_ += [c] if c not in cells_ else []
        pols += [p] if p not in pols else []
    lines = ["| Profile | " + " | ".join(pols) + f" | fixed-2x − {ref} fixed-2x |",
             "| --- | " + " | ".join("---:" for _ in pols) + " | ---: |"]
    summary = {}
    for c in cells_:
        row = []
        for p in pols:
            v = r[f"{c} | {p}"]
            row.append(f"{v['mean']:.2f} ({v['p10']:.1f}-{v['p90']:.1f})")
            summary[f"{c} | {p}"] = {"mean": v["mean"], "se": v["se"],
                                     "vs_ref": _paired(r, f"{c} | {p}", f"{ref} | {p}") if c != ref else None}
        d = _paired(r, f"{c} | fixed-2x", f"{ref} | fixed-2x") if c != ref else None
        dtxt = "-" if d is None else f"{d['diff']:+.2f} ± {d['se']:.2f} ({int(100 * d['p_run'])}%)"
        lines.append(f"| {c} | " + " | ".join(row) + f" | {dtxt} |")
    return "\n".join(lines), summary


def robust_table(d):
    """Markdown: every robustness check, per cell, the paired difference against K 6x64K for fixed-2x and the
    index, and the index minus fixed-2x within the cell."""
    checks = [("9 h base (distinct)", "nine", None), ("9 h bootstrap 23", "nine-boot23", 2),
              ("9 h x0.5", "nine-x0.5", None), ("9 h x1.33", "nine-x1.33", None),
              ("1 h c110", "hour", None), ("1 h bootstrap", "hour-boot", cp.BOOT_PER),
              ("1 h x0.5", "hour-x0.5", None), ("1 h x1.33", "hour-x1.33", None)]
    lines = ["| Check | Profile | fixed-2x | − K 6x64K (fixed-2x) | index | − K 6x64K (index) | index − fixed-2x |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    out = {}
    for title, name, per in checks:
        r = _load(d, name)
        if r is None:
            continue
        idx = "gittins-h0.15" if name.startswith("nine") else "gittins-c110-open20"
        for c in KEY9:
            for m in ("", " m0.75"):
                cell = c + m
                a, b = f"{cell} | fixed-2x", f"{cell} | {idx}"
                if a not in r or b not in r:
                    continue
                df = _paired(r, a, "K 6x64K | fixed-2x", per) if c != "K 6x64K" else None
                di = _paired(r, b, f"K 6x64K | {idx}", per) if c != "K 6x64K" else None
                dd = _paired(r, b, a, per)

                def fmt(x):
                    if x is None:
                        return "-"
                    share = x.get("p_roster", x["p_run"])
                    return f"{x['diff']:+.2f} ± {x['se']:.2f} ({int(100 * share)}%)"
                lines.append(f"| {title} | {cell} | {r[a]['mean']:.2f} | {fmt(df)} | {r[b]['mean']:.2f} | {fmt(di)} "
                             f"| {fmt(dd)} |")
                out[f"{name} | {cell}"] = {"fixed-2x": r[a]["mean"], "index": r[b]["mean"], "fixed_vs_K6": df,
                                           "index_vs_K6": di, "index_vs_fixed": dd}
    return "\n".join(lines), out


def halves_table(d):
    """Markdown: on each held-out half (110 distinct games from 10 source games), the paired difference of every
    cell against K 6x64K, and in how many of the 8 halves it is positive."""
    agg = {}
    for s in range(4):
        r = _load(d, f"nine-halves-s{s}")
        if r is None:
            continue
        for h in (0, 1):
            for label in r:
                if not label.startswith(f"H{h} "):
                    continue
                cell, pol = _split(label[3:])
                if cell == "K 6x64K":
                    continue
                x = _paired(r, label, f"H{h} K 6x64K | {pol}")
                agg.setdefault((cell, pol), []).append(x["diff"])
            for label in r:  # the index minus fixed-2x within each cell (K 6x64K included)
                if label.startswith(f"H{h} ") and label.endswith("| gittins-h0.15"):
                    cell = _split(label[3:])[0]
                    x = _paired(r, label, f"H{h} {cell} | fixed-2x")
                    agg.setdefault((cell, "gittins-h0.15 − fixed-2x"), []).append(x["diff"])
    lines = ["| Profile | Policy | Mean difference vs K 6x64K over halves | Halves positive | Min / max |",
             "| --- | --- | ---: | ---: | --- |"]
    for (cell, pol), v in agg.items():
        v = np.array(v)
        lines.append(f"| {cell} | {pol} | {v.mean():+.2f} | {int((v > 0).sum())} of {len(v)} | {v.min():+.2f} / "
                     f"{v.max():+.2f} |")
    return "\n".join(lines), {f"{c} | {p}": v for (c, p), v in agg.items()}


def breakeven(r, pol, refs=("K 6x64K", "M3 6x64K")):
    """The window multiplier at which a shorter-window profile ties each 64K reference (linear between the
    simulated multipliers, extrapolated from the two lowest when every multiplier still wins)."""
    out = {}
    for name, p in PROFILES.items():
        if p["window"] >= 64:
            continue
        pts = sorted((m, r[f"{name if m == 1.0 else f'{name} m{m:g}'} | {pol}"]["mean"]) for m in MULTS[p["window"]])
        for ref in refs:
            target = r[f"{ref} | {pol}"]["mean"]
            be = None
            for (m0, s0), (m1, s1) in zip(pts, pts[1:]):
                if s0 <= target <= s1:
                    be = m0 + (target - s0) * (m1 - m0) / (s1 - s0)
            if be is None and target < pts[0][1]:
                (m0, s0), (m1, s1) = pts[0], pts[1]
                be = m0 - (s0 - target) * (m1 - m0) / (s1 - s0)
            out[f"{name} vs {ref}"] = None if be is None else round(be, 2)
    return out


def report(d):
    out = {}
    for name in ("nine", "hour"):
        r = _load(d, name)
        if r is not None:
            txt, s = main_table(r)
            print(f"### {name}\n\n{txt}\n")
            out[name] = s
            idx = "gittins-h0.15" if name == "nine" else "gittins-c110-open20"
            out[f"{name} breakeven"] = {pol: breakeven(r, pol) for pol in ("fixed-2x", idx)}
            print(f"break-even multipliers: {json.dumps(out[f'{name} breakeven'], indent=1)}\n")
    txt, s = robust_table(d)
    print(f"### robustness\n\n{txt}\n")
    out["robust"] = s
    txt, s = halves_table(d)
    print(f"### halves\n\n{txt}\n")
    out["halves"] = s
    Path(d, "summary.json").write_text(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", required=True)
    parser.add_argument("stage")
    parser.add_argument("--runs", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    share = cp.opening_share()
    stages = {"nine": nine, "nine_boot": nine_boot, "nine_sens": nine_sens, "nine_halves": nine_halves,
              "hour": hour, "hour_boot": hour_boot, "hour_sens": hour_sens,
              "kv8_nine": kv8_nine, "kv8_hour": kv8_hour, "kv8_robust": kv8_robust,
              "kv8_switch": kv8_switch}
    if args.stage == "profiles":
        print(json.dumps(PROFILES, indent=1))
        return
    if args.stage == "kv8_profiles":
        rates = {n: {w: round(kv8_rate(n, w), 1) for w in KV8_SLOPE} for n in (12, 14, 16, 18, 21)}
        print(json.dumps({"profiles": KV8, "rates": rates, "prefill_factor": KV8_PREFILL}, indent=1))
        return
    if args.stage == "kv8_report":
        kv8_report(args.out_dir)
        return
    if args.stage == "report":
        report(args.out_dir)
        return
    for name, cfg in stages[args.stage](args.runs, share).items():
        if args.only and name not in args.only:
            continue
        cp.run_config(name, cfg, args.attempts, args.out_dir, args.jobs)


if __name__ == "__main__":
    main()
