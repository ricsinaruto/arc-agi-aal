"""hero-v2 variants in the calibrated simulator (29 September): phase 2 restarted on MTP-3, and 4-bit KV at 64K.

"A. Would hero-v2 do better if phase 2 ran on MTP-3 serving after the opening (a server
restart at the opening switch)? B. int4 KV fits 38 x 64K on hero-v2's pool: how does 38 x 64K compare with hero-v2's
23 x 64K?" The 1-hour c110 score and the 9-hour hidden-set score (110 distinct games) for each.

    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/sim_mtp3_kv4.py profiles
    ARC_VALIDATION_ROOT=<main checkout> uv run --extra modal modal run scripts/arena_analysis/modal_sim.py \
        --study mtp3_kv4 --stages main,robust --run-id <id> --attempts <attempts-now.jsonl>
    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/sim_mtp3_kv4.py report

Everything but the serving cells is ``sim_hero_v2.py``'s (world w1, fine start, opening shares, re-prefill charges,
fixed-2x on the token clock with the 1,920-token floor, the six-clear switch, rank[nine]). The serving model and its
derivation are in docs/validation-20260927/sim-mtp3-kv4.md §1; in short:

- Two calibrations. ``sim``: sim-hero-v2.md's rates (phase 2 A(n) = 1000 n / (8.92 + 1.046 n), 697 tok/s at 23;
  opening 564 at max_num_seqs 25). ``live``: rescaled to the hero-v2 2 h run (phase 2 770 at 23 focus streams, x1.105;
  opening 690, x1.223), and MTP-3 phase 2 x1.105 too (the phase-5 stack is +10% per iteration at six streams).
- MTP-3 phase 2 (``mtp3_rate``): a per-iteration model anchored at six streams (arena 500 tok/s, depthwise-mtp3 c110
  run), each stream emitting 1 + acceptance tokens per iteration; above six the iteration grows by the decode slope
  per stream (MoE reads saturate as on MTP-0) plus the arena's prefill/queueing overhead per generated token
  (0.616 ms = the MTP-0 arena slope 1.046 minus its decode-only 0.43), times the tokens per stream per iteration.
- MTP-3 restart: the opening on MTP-0 (hero-v2's), then ``switch_pause`` s without tokens, then N focus streams on
  MTP-3; the focus games' first admissions re-prefill (cold) as every admission does.
- MTP-3 for the whole run: the opening at the measured MTP-3 opening rate (225 tok/s, mtp3-wide-kernels) or an
  optimistic 1.1x MTP-0's (streams-9h.md's old assumption).
- int4 38 x 64K / 30 x 64K on MTP-0: the same A(n) with max_num_seqs raised (opening at max_num_seqs 40 / 32), times a
  speed factor (0.90 / 1.0 / 1.15) on both phases, and a clear-rate multiplier for int4's drift (1.0-0.80).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c110_path as cp  # noqa: E402
import schedule_sim as sim  # noqa: E402
import sim_hero_v2 as hv  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "validation-20260927" / "sim-mtp3-kv4"

# ---------------------------------------------------------------------------------------------- serving model

BASE_SLOTS, BASE_CAP = 23, 25  # hero-v2 as run: 23 x 64K, max_num_seqs 25
LIVE_P2, LIVE_OPEN = 770.0, 690.0  # hero-v2 2 h run: phase 2 744-793 (770 central), opening 677-704
CAL = {"sim": {"p2": 1.0, "open": 1.0},
       "live": {"p2": LIVE_P2 / hv.p2_rate(BASE_SLOTS), "open": LIVE_OPEN / hv.open_rate(BASE_CAP)}}

# MTP-3 phase 2: (six-stream arena tok/s, accepted per 3-token draft, decode ms per extra stream above six)
MTP3 = {"low": (418.0, 1.65, 3.0),  # the standard mtp3-wide-kernels c110 phase 2; kv8 acceptance; 1->6 slope continued
        "central": (500.0, 1.87, 1.8),  # the depthwise-mtp3 c110 runs' phase 2; phase-4 64K x 6 acceptance
        "high": (540.0, 1.87, 1.2)}  # phase-4 decode-only 64K x 6 (no arena overhead); flatter saturation
ARENA_OVH = hv.SLOPE - 0.43  # ms per generated token per stream-iteration: MTP-0 arena slope minus decode-only
MTP3_OPEN = {"measured": 225.4, "optimistic": 1.1 * hv.open_rate(BASE_CAP)}  # mtp3-wide-kernels opening; 1.1 x MTP-0
MTP3_N = (9, 12, 15)
PAUSES = (400.0, 700.0)
BREAKEVEN_RATES = (500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0)

INT4_SPEED = {"central": 1.0, "low": 0.90, "high": 1.15}
INT4_EFF = (1.0, 0.95, 0.90, 0.85, 0.80)
INT4_CELLS = {38: 40, 30: 32}  # streams -> max_num_seqs (the decode graphs capture to it)


def mtp3_rate(n, rate="central"):
    """MTP-3 phase-2 tok/s at ``n`` running (sim calibration)."""
    a6, acc, dec = MTP3[rate]
    tok = 1.0 + acc
    t6 = 6 * tok * 1000.0 / a6
    return 1000.0 * n * tok / (t6 + (dec + tok * ARENA_OVH) * (n - 6))


def mtp3_capacity():
    """64K windows of MTP-3 + int8 KV on hero-v2's pool (kv8_sizing geometry; docs §1.3)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import kv8_sizing as kv8

    gib = 1 << 30
    hero_v2 = 1335 * kv8.geometry(0, 2, "8bit", "8bit-nopad")["block_bytes"]  # the served int8 pool
    phase4 = 691 * kv8.geometry(3, 2, "bf16")["block_bytes"]
    mtp0_ac = 751 * kv8.geometry(0, 2, "bf16")["block_bytes"] + 2.4 * gib
    overhead = mtp0_ac - phase4  # MTP-3's draft weights, graphs and state against MTP-0 A+C, same lab
    out = {"hero_v2_pool_gib": round(hero_v2 / gib, 2), "mtp3_overhead_gib": round(overhead / gib, 2),
           "ratio_estimate": round(BASE_SLOTS * 9 / 14, 1)}
    for label, extra in (("central", 0.0), ("+1 GiB graphs/state", 1.0 * gib)):
        f = kv8.fit(hero_v2 - overhead - extra, 3, 2, "8bit", "8bit-nopad")
        out[label] = {k: f[k] for k in ("block_tokens", "blocks", "gate_blocks", "blocks_per_64k", "fit_64k",
                                         "fit_exact")}
    return out


def cell(cal, slots, p2, opening, pause=0.0, eff=1.0):
    return {"slots": slots, "p2": round(p2 * CAL[cal]["p2"], 1), "open": round(opening * CAL[cal]["open"], 1),
            "readmit": hv.readmit(64), "mult": eff, "pause": pause}


def base_cell(cal):
    return cell(cal, BASE_SLOTS, hv.p2_rate(BASE_SLOTS), hv.open_rate(BASE_CAP))


def restart_cell(cal, n, p2, pause):
    return cell(cal, n, p2, hv.open_rate(BASE_CAP), pause)


def whole_cell(cal, n, rate, opening):
    return cell(cal, n, mtp3_rate(n, rate), MTP3_OPEN[opening])


def int4_cell(cal, n, speed="central", eff=1.0, slope="central"):
    s = INT4_SPEED[speed]
    return cell(cal, n, s * hv.p2_rate(n, slope), s * hv.open_rate(INT4_CELLS[n], slope), eff=eff)


# ---------------------------------------------------------------------------------------------- arms


def fixed(c, share, **extra):
    skw = hv.sim_kw(c, share, switch_games=6, **extra)
    if c["pause"]:
        skw["switch_pause"] = c["pause"]
    return ["fixed-2x", dict(hv.FIXED_TOKEN_CLOCK), skw]


def rank(c, share, **extra):
    return hv.rank(c, share, hv.RANK_NINE, **extra)


def arms(cal):
    """{label: (policy kind, cell)} for one calibration."""
    out = {"base 23x64K MTP-0": ("fixed", base_cell(cal)), "rank | base 23x64K MTP-0": ("rank", base_cell(cal))}
    for n in MTP3_N:
        for rate in MTP3:
            for p in PAUSES:
                out[f"restart MTP-3 {n} {rate} pause{p:g}"] = ("fixed", restart_cell(cal, n, mtp3_rate(n, rate), p))
            out[f"whole MTP-3 {n} {rate} open-measured"] = ("fixed", whole_cell(cal, n, rate, "measured"))
        out[f"whole MTP-3 {n} central open-optimistic"] = ("fixed", whole_cell(cal, n, "central", "optimistic"))
        out[f"rank | whole MTP-3 {n} central"] = ("rank", whole_cell(cal, n, "central", "measured"))
        for r in BREAKEVEN_RATES:  # the phase-2 aggregate (sim tok/s, x the calibration) needed for the restart
            out[f"grid restart {n} pause400 A{r:g}"] = ("fixed", restart_cell(cal, n, r, 400.0))
    for r in BREAKEVEN_RATES:
        out[f"grid restart 15 pause700 A{r:g}"] = ("fixed", restart_cell(cal, 15, r, 700.0))
    for rate in ("low", "high"):
        out[f"rank | whole MTP-3 15 {rate}"] = ("rank", whole_cell(cal, 15, rate, "measured"))
    for n in INT4_CELLS:
        speeds = INT4_SPEED if n == 38 else {"central": 1.0}
        for sp in speeds:
            for e in INT4_EFF:
                out[f"int4 {n}x64K {sp} eff{e:g}"] = ("fixed", int4_cell(cal, n, sp, e))
        if n == 30:
            for sp in ("low", "high"):
                out[f"int4 30x64K {sp} eff1"] = ("fixed", int4_cell(cal, 30, sp))
        out[f"int4 {n}x64K slope-low eff1"] = ("fixed", int4_cell(cal, n, slope="low"))
        for e in (1.0, 0.95, 0.90):
            out[f"rank | int4 {n}x64K central eff{e:g}"] = ("rank", int4_cell(cal, n, "central", e))
    for sp in ("low", "high"):
        out[f"rank | int4 38x64K {sp} eff1"] = ("rank", int4_cell(cal, 38, sp))
    return out


ROBUST_ARMS = ["base 23x64K MTP-0", "rank | base 23x64K MTP-0",
               "restart MTP-3 9 central pause400", "restart MTP-3 12 central pause400",
               "restart MTP-3 15 central pause400", "restart MTP-3 15 high pause400",
               "int4 38x64K central eff1", "int4 38x64K central eff0.95", "int4 38x64K central eff0.9",
               "int4 30x64K central eff1", "rank | int4 38x64K central eff1"]


def _policy(kind, c, share, extra):
    return (fixed if kind == "fixed" else rank)(c, share, **extra)


HORIZONS = {"hour": (hv.HOUR, {}, 40), "nine": (cp.CAL, hv.NINE, 16)}


def main_stage(runs, share):
    out = {}
    for cal in CAL:
        for hz, (base, extra, nrun) in HORIZONS.items():
            pol = {label: _policy(kind, c, share, extra) for label, (kind, c) in arms(cal).items()}
            out[f"main-{cal}-{hz}"] = {**base, "runs": runs or nrun, "seed": 700 + (hz == "nine"), "policies": pol}
    return out


def robust_stage(runs, share):
    """The sim-hero-v2 checks on the key arms: game bootstrap (1 h: 20 rosters x 6; 9 h: 12 x 2) and clear rates
    x0.5 / x1.33."""
    b1 = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": cp.BOOT_PER}
    b9 = {"pool": sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}
    out = {}
    for cal in CAL:
        a = arms(cal)
        for hz, (base, extra, nrun) in HORIZONS.items():
            boot = b1 if hz == "hour" else b9
            pol = {k: _policy(*a[k], share, {**extra, "bootstrap": boot}) for k in ROBUST_ARMS}
            bcfg = hv.HOUR if hz == "hour" else {**cp.CAL, "roster": "23"}
            nb = (20 if hz == "hour" else 12) * boot["per_roster"]
            out[f"robust-{cal}-{hz}-boot"] = {**bcfg, "runs": nb, "seed": 710 + (hz == "nine"), "policies": pol}
            for s in (0.5, 1.33):
                pol = {}
                for k in ROBUST_ARMS:
                    kind, c = a[k]
                    pol[k] = _policy(kind, {**c, "mult": c["mult"] * s}, share, extra)
                out[f"robust-{cal}-{hz}-x{s:g}"] = {**base, "runs": runs or nrun, "seed": 720 + (hz == "nine"),
                                                    "policies": pol}
    return out


def smoke_stage(runs, share):
    out = {}
    a = arms("sim")
    keep = ["base 23x64K MTP-0", "restart MTP-3 15 central pause400", "int4 38x64K central eff1",
            "rank | int4 38x64K central eff1"]
    for hz, (base, extra, _) in HORIZONS.items():
        out[f"smoke-{hz}"] = {**base, "runs": runs or 2, "seed": 790 + (hz == "nine"),
                              "policies": {k: _policy(*a[k], share, extra) for k in keep}}
    return out


STAGES = {"smoke": smoke_stage, "main": main_stage, "robust": robust_stage}

# ---------------------------------------------------------------------------------------------- report


def _load(d, name):
    p = Path(d, f"{name}.result.json")
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v):
    return f"{v['mean']:.2f} ({v['p10']:.1f} / {v['p50']:.1f} / {v['p90']:.1f})"


def delta(r, a, b, per=None):
    x = hv.paired(r, a, b, per)
    share = x.get("p_roster", x["p_run"])
    return f"{x['diff']:+.2f} ± {x['se']:.2f} ({int(round(100 * share))}%)", x


def breakeven(xs, ys):
    """First zero crossing of ``ys`` (paired delta) over ``xs`` by linear interpolation; None if none in range."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 <= 0 < y1 or y0 >= 0 > y1:
            return round(x0 + (x1 - x0) * (0 - y0) / (y1 - y0), 3)
    return None


def report(d=OUT):
    summary = {"capacity": mtp3_capacity(), "cal": CAL, "rates": profiles()}
    base, rbase = "base 23x64K MTP-0", "rank | base 23x64K MTP-0"
    for cal in CAL:
        for hz in HORIZONS:
            r = _load(d, f"main-{cal}-{hz}")
            if r is None:
                continue
            a = arms(cal)
            rows = {}
            print(f"\n### main {cal} {hz}\n")
            print("| Arm | streams | phase 2 / opening tok/s | mean (p10 / p50 / p90) | − base fixed-2x | − base same policy |")
            print("| --- | ---: | --- | --- | ---: | ---: |")
            for label, (kind, c) in a.items():
                if label.startswith("grid"):
                    continue
                t1, x1 = delta(r, label, base)
                t2, x2 = delta(r, label, rbase if kind == "rank" else base)
                print(f"| {label} | {c['slots']} | {c['p2']:.0f} / {c['open']:.0f} | {fmt(r[label])} | {t1} | {t2} |")
                rows[label] = {"mean": r[label]["mean"], "p10": r[label]["p10"], "p50": r[label]["p50"],
                               "p90": r[label]["p90"], "vs_base": x1, "vs_base_same_policy": x2,
                               "slots": c["slots"], "p2": c["p2"], "open": c["open"], "levels": r[label]["levels"],
                               "switch": r[label]["switch"]}
            be = {}
            for n, p in [(n, 400.0) for n in MTP3_N] + [(15, 700.0)]:
                labels = [f"grid restart {n} pause{p:g} A{x:g}" for x in BREAKEVEN_RATES]
                ys = [hv.paired(r, lab, base)["diff"] for lab in labels]
                xs = [x * CAL[cal]["p2"] for x in BREAKEVEN_RATES]
                be[f"{n} pause{p:g}"] = {"rates": [round(x) for x in xs], "delta": ys, "breakeven_tok_s":
                                         breakeven(xs, ys)}
            for n in INT4_CELLS:
                for sp in (INT4_SPEED if n == 38 else {"central": 1.0}):
                    ys = [hv.paired(r, f"int4 {n}x64K {sp} eff{e:g}", base)["diff"] for e in INT4_EFF]
                    xs = list(INT4_EFF)[::-1]
                    be[f"int4 {n} {sp}"] = {"eff": list(INT4_EFF), "delta": ys,
                                            "breakeven_eff": breakeven(xs, ys[::-1])}
            print(f"\nbreak-even ({cal} {hz}): " + json.dumps(be))
            summary[f"main-{cal}-{hz}"] = {"arms": rows, "breakeven": be}
        for hz in HORIZONS:
            for check in ("boot", "x0.5", "x1.33"):
                r = _load(d, f"robust-{cal}-{hz}-{check}")
                if r is None:
                    continue
                per = cp.BOOT_PER if (check == "boot" and hz == "hour") else (2 if check == "boot" else None)
                out = {}
                for label in ROBUST_ARMS[1:]:
                    ref = rbase if label.startswith("rank") and label != rbase else base
                    t, x = delta(r, label, ref, per)
                    out[label] = {"mean": r[label]["mean"], "ref": ref, **x}
                out[base] = {"mean": r[base]["mean"]}
                summary[f"robust-{cal}-{hz}-{check}"] = out
                print(f"\nrobust {cal} {hz} {check}: " + "; ".join(
                    f"{k}: {v['mean']:.2f} {v.get('diff', 0):+.2f} ({v.get('p_roster', v.get('p_run', 1)):.2f})"
                    for k, v in out.items()))
    Path(d, "summary.json").write_text(json.dumps(summary, indent=1))


def profiles():
    out = {"arena_ovh_ms": round(ARENA_OVH, 3), "cal": CAL, "mtp3_capacity": None}
    out["mtp0"] = {n: round(hv.p2_rate(n), 1) for n in (6, 9, 12, 15, 23, 30, 38)}
    out["mtp0_open"] = {cap: round(hv.open_rate(cap), 1) for cap in (25, 32, 40)}
    out["mtp3"] = {rate: {n: round(mtp3_rate(n, rate), 1) for n in (6, 9, 12, 15)} for rate in MTP3}
    out["mtp3_per_stream"] = {rate: {n: round(mtp3_rate(n, rate) / n, 1) for n in (6, 9, 12, 15)} for rate in MTP3}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", default="")
    parser.add_argument("stage")
    parser.add_argument("--runs", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.stage == "profiles":
        print(json.dumps({**profiles(), "mtp3_capacity": mtp3_capacity(),
                          "cells": {cal: {k: v[1] for k, v in arms(cal).items() if not k.startswith("grid")}
                                    for cal in CAL}}, indent=1))
        return
    if args.stage == "report":
        report(args.out_dir)
        return
    share = cp.opening_share()
    for name, cfg in STAGES[args.stage](args.runs, share).items():
        if args.only and name not in args.only:
            continue
        print(f"== {name}: {len(cfg['policies'])} policies x {cfg['runs']} runs", flush=True)
        hv.run_local(name, cfg, args.attempts, args.out_dir, args.jobs)


if __name__ == "__main__":
    main()
