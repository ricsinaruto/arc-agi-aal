"""Token budgets instead of busy-second budgets: measurements and simulator comparison.

``rates``: from recorded c110 runs (``arena.json`` per trial: per-game ``charges``, ``inference_intervals``,
``tool_intervals``, scheduler events), per phase (opening / focus phase): completion tokens per busy second of one game
(the fixed-2x budget clock: admitted model-request seconds plus tool execution), completion tokens per request, the
share of completion tokens by request purpose, and what the busy clock's level-0 grants and T1 were in tokens.

``sim``: the calibrated c110 simulator (``schedule_sim`` via ``streams_9h`` / ``c110_path``) with fixed-2x on the busy
clock (the recorded ``busy_factor`` = phase-2 / opening tokens per busy second, floor 30 s at the phase-2 rate) against
fixed-2x on the token clock (``busy_factor`` 1, floor ``FLOOR_TOKENS``), at 6 x 64K, 14 x 64K and 21 x 42K, for one hour
(c110) and nine hours (110 distinct games). CPU only.

    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/token_budgets.py rates
    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/token_budgets.py \
        --attempts <attempts-now.jsonl> sim [--jobs 8]
    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/token_budgets.py report

Outputs go to ``docs/validation-20260927/token-budgets/``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "validation-20260927" / "token-budgets"
RUNS = Path("./runs")
TRIALS = {  # label -> (trial directory relative to RUNS, focus slots x window)
    "kernels-v2 (6x64K)": ("modal-validation/arena-duck-60m-c110-kernels-v2-modal-20260928/"
                           "arena-duck-60m-c110-kernels-v2-modal-20260928/arena/trials/duck-transfer", "6x64K"),
    "hero-6x64k (6x64K)": ("modal-validation/arena-duck-60m-c110-hero-6x64k-v1-modal-20260928/"
                           "arena-duck-60m-c110-hero-6x64k-v1-modal-20260928/arena/trials/duck-transfer", "6x64K"),
    "hero preview (14x64K)": ("kaggle-arena-flash-competition-hero-v1/arena/trials/duck-transfer", "14x64K"),
    "hero-21x42k (21x42K)": ("modal-validation/arena-duck-60m-c110-hero-21x42k-v1-modal-20260928/"
                             "arena-duck-60m-c110-hero-21x42k-v1-modal-20260928/arena/trials/duck-transfer", "21x42K"),
}
# The token floor (arena.scheduler.TOKEN_FLOOR): the busy clock's 30 s floor at the reference phase-2 rate of one
# focus game at six streams (64 completion tokens per busy second; kernels-v2 63.5, hero-6x64k 64).
REFERENCE_TOKENS_PER_SECOND = 64.0
FLOOR_TOKENS = 1920


def overlap(intervals, lo, hi):
    total = 0.0
    for iv in intervals:
        a, b = max(lo, iv["started_wall_seconds"]), min(hi, iv["ended_wall_seconds"])
        total += max(0.0, b - a)
    return total


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if len(x) else None


def trial_rates(trial):
    report = json.loads((trial / "arena.json").read_text())
    sched = report["scheduler"]
    switch = float(sched.get("switch_wall_seconds") or 0.0)
    wall = float(report.get("wall_seconds") or 3600.0)
    phases = {"opening": (0.0, switch), "focus": (switch, wall)}
    out = {"switch_wall_seconds": switch, "active_limit": sched.get("active_limit"), "phases": {}}
    purposes = {}
    for name, (lo, hi) in phases.items():
        tokens = busy = inference = 0.0
        per_game, per_request = [], []
        for g in report["games"]:
            charges = [c for c in g.get("charges", []) if lo < c["wall_seconds"] <= hi]
            t = sum(c["completion_tokens"] for c in charges)
            b = overlap(g.get("inference_intervals", []), lo, hi) + overlap(g.get("tool_intervals", []), lo, hi)
            tokens += t
            busy += b
            inference += overlap(g.get("inference_intervals", []), lo, hi)
            per_request += [c["completion_tokens"] for c in charges if c["purpose"] == "gameplay"]
            if b >= 30.0:
                per_game.append(t / b)
            for c in charges:
                row = purposes.setdefault(c["purpose"], [0, 0])
                row[0] += c["completion_tokens"]
                row[1] += 1
        out["phases"][name] = {
            "seconds": round(hi - lo, 1), "completion_tokens": int(tokens), "aggregate_tokens_per_second":
            round(tokens / max(hi - lo, 1e-9), 1), "busy_seconds": round(busy, 1),
            "tokens_per_busy_second": round(tokens / max(busy, 1e-9), 1),
            "tokens_per_inference_second": round(tokens / max(inference, 1e-9), 1),
            "tool_share_of_busy": round(1 - inference / max(busy, 1e-9), 3),
            "per_game_tokens_per_busy_second": {"p10": pct(per_game, 10), "p50": pct(per_game, 50),
                                                "p90": pct(per_game, 90), "games": len(per_game)},
            "gameplay_completion_per_request": {"p50": pct(per_request, 50), "p90": pct(per_request, 90),
                                                "p99": pct(per_request, 99), "requests": len(per_request)},
        }
    total = sum(v[0] for v in purposes.values())
    out["purpose_share"] = {k: {"tokens": v[0], "requests": v[1], "share": round(v[0] / max(total, 1), 4)}
                            for k, v in sorted(purposes.items())}
    # What the busy clock granted, in tokens: level-0 grants (the busy seconds at the first park) and T1.
    level0, t1 = [], []
    floored = 0
    for g in report["games"]:
        charges = g.get("charges", [])
        first_park = next((e for e in g.get("admissions", []) if e["event"] == "park" and e["levels"] == 0), None)
        if first_park is not None:
            at = first_park["wall_seconds"]
            tok = sum(c["completion_tokens"] for c in charges if c["wall_seconds"] <= at + 1e-3)
            level0.append((first_park["budget_seconds"], tok))
            floored += first_park["budget_seconds"] < 30.0
        if g.get("time_to_level"):
            first = g["time_to_level"][0]
            tok = sum(c["completion_tokens"] for c in charges if c["wall_seconds"] <= first["wall_seconds"] + 1e-3)
            t1.append((first["budget_seconds"], tok, first["wall_seconds"]))
    l0s = np.array([x[0] for x in level0])
    l0t = np.array([x[1] for x in level0])
    out["level0_first_park"] = {
        "games": len(level0), "floored_below_30s": int(floored),
        "busy_seconds": {"p10": pct(l0s, 10), "p50": pct(l0s, 50), "p90": pct(l0s, 90)},
        "completion_tokens": {"p10": pct(l0t, 10), "p50": pct(l0t, 50), "p90": pct(l0t, 90)},
        "below_floor_tokens": int((l0t < FLOOR_TOKENS).sum()),
    }
    out["t1"] = [{"busy_seconds": round(s, 1), "completion_tokens": int(t), "wall_seconds": round(w, 1),
                  "tokens_per_busy_second": round(t / max(s, 1e-9), 1), "in_opening": w <= switch}
                 for s, t, w in sorted(t1, key=lambda x: x[2])]
    return out


def rates(_args):
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for label, (rel, profile) in TRIALS.items():
        trial = RUNS / rel
        if not (trial / "arena.json").is_file():
            print("missing", label, trial)
            continue
        r = trial_rates(trial)
        r["profile"] = profile
        result[label] = r
        o, f = r["phases"]["opening"], r["phases"]["focus"]
        print(f"{label:24s} switch {r['switch_wall_seconds']:6.0f}s  opening {o['aggregate_tokens_per_second']:6.1f} tok/s, "
              f"{o['tokens_per_busy_second']:5.1f} tok/busy-s | focus {f['aggregate_tokens_per_second']:6.1f} tok/s, "
              f"{f['tokens_per_busy_second']:5.1f} tok/busy-s | factor {f['tokens_per_busy_second'] / o['tokens_per_busy_second']:.2f}"
              f" | reply p50/p90 {f['gameplay_completion_per_request']['p50']}/{f['gameplay_completion_per_request']['p90']}"
              f" | level-0 park p50 {r['level0_first_park']['completion_tokens']['p50']} tok"
              f" ({r['level0_first_park']['busy_seconds']['p50']} s), floored {r['level0_first_park']['floored_below_30s']},"
              f" < floor tokens {r['level0_first_park']['below_floor_tokens']}")
    Path(OUT, "rates.json").write_text(json.dumps(result, indent=1))


# ---------------------------------------------------------------------------------------------- simulator


SIM_RUNS = {"6x64K": "hero-6x64k (6x64K)", "14x64K": "hero preview (14x64K)", "21x42K": "hero-21x42k (21x42K)"}


def sim_profiles():
    """The three focus profiles on the hero serving (MTP-0 + A+C, int8 KV, Marlin experts), each from its recorded
    c110 hour (``rates.json``): opening and phase-2 aggregate tokens per wall second, and one game's completion tokens
    per busy second in each phase (the busy clock's exchange rate). The opening is the same 110-game opening in all
    three (~23 tokens per busy second at ~24 running)."""
    measured = json.loads(Path(OUT, "rates.json").read_text())
    out = {}
    for name, label in SIM_RUNS.items():
        r = measured[label]
        o, f = r["phases"]["opening"], r["phases"]["focus"]
        slots = int(name.split("x")[0])
        out[name] = {"open": o["aggregate_tokens_per_second"], "p2": f["aggregate_tokens_per_second"], "slots": slots,
                     "busy_open": o["tokens_per_busy_second"], "busy_p2": f["tokens_per_busy_second"],
                     "stream": round(f["aggregate_tokens_per_second"] / slots, 2), "run": label}
    return out


def fixed2x_busy(p, multiple=2.0):
    return ["fixed-2x", {"multiple": multiple, "busy_factor": p["busy_p2"] / p["busy_open"],
                         "tokens_per_busy_second": p["busy_p2"]}]


def fixed2x_tokens(multiple=2.0, floor=FLOOR_TOKENS):
    # busy_factor 1: an opening token weighs as much as a phase-2 token; the floor in tokens.
    return ["fixed-2x", {"multiple": multiple, "busy_factor": 1.0, "floor_seconds": 1.0,
                         "tokens_per_busy_second": float(floor)}]


def policies(profiles, share, wall, extra, full=True, scale=1.0):
    import c110_path as cp

    pol = {}
    for name, p in profiles.items():
        for sw in (6, 2):
            skw = cp.cal_sim(p, share, efficiency=scale, shuffle=True, wall=wall, switch_games=sw, **extra)
            tag = f"{name} sw{sw}"
            pol[f"{tag} | busy 2x"] = [*fixed2x_busy(p), skw]
            pol[f"{tag} | tokens 2x"] = [*fixed2x_tokens(), skw]
            if full and sw == 6:
                # Sensitivity: the token-clock multiple that gives the busy clock's phase-2 grant for a T1 earned in
                # the opening (2 x the busy factor), and the token floor at 0.5x / 2x.
                bf = p["busy_p2"] / p["busy_open"]
                pol[f"{tag} | tokens {2 * bf:.1f}x"] = [*fixed2x_tokens(multiple=round(2 * bf, 2)), skw]
                pol[f"{tag} | tokens 2x floor960"] = [*fixed2x_tokens(floor=960), skw]
                pol[f"{tag} | tokens 2x floor3840"] = [*fixed2x_tokens(floor=3840), skw]
    return pol


def sim_configs(runs_hour=200, runs_nine=48):
    """Paired (common random numbers) configs: main hour / nine-hour, the game bootstrap and clear rates x0.5 / x1.33."""
    import c110_path as cp

    share = cp.opening_share()
    profiles = sim_profiles()
    nine = {"shared": False}
    b1 = {"pool": cp.sim.C20, "k": 20, "n": 110, "per_roster": 6}
    b9 = {"pool": cp.sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}
    configs = {
        "tb-hour": {**cp.CAL, "runs": runs_hour, "seed": 41, "policies": policies(profiles, share, 3600.0, {})},
        "tb-nine": {**cp.CAL, "runs": runs_nine, "seed": 6, "policies": policies(profiles, share, 32400.0, nine)},
        "tb-hour-boot": {**cp.CAL, "runs": 20 * 6, "seed": 11, "policies": policies(
            profiles, share, 3600.0, {"bootstrap": b1}, full=False)},
        "tb-nine-boot23": {**cp.CAL, "roster": "23", "runs": 16 * 2, "seed": 21, "policies": policies(
            profiles, share, 32400.0, {**nine, "bootstrap": b9}, full=False)},
    }
    for s in (0.5, 1.33):
        configs[f"tb-hour-x{s:g}"] = {**cp.CAL, "runs": 100, "seed": 12, "policies": policies(
            profiles, share, 3600.0, {}, full=False, scale=s)}
        configs[f"tb-nine-x{s:g}"] = {**cp.CAL, "runs": 24, "seed": 22, "policies": policies(
            profiles, share, 32400.0, nine, full=False, scale=s)}
    for c in configs.values():
        c["profiles"] = profiles
    return configs


def sim(args):
    import c110_path as cp

    OUT.mkdir(parents=True, exist_ok=True)
    configs = sim_configs(args.runs_hour, args.runs_nine)
    for name, config in configs.items():
        if args.only and name not in args.only:
            continue
        cp.run_config(name, config, args.attempts, OUT, args.jobs)


def paired(r, a, b, per=None):
    x = np.array(r[a]["scores"]) - np.array(r[b]["scores"])
    out = {"diff": round(float(x.mean()), 3), "se": round(float(x.std(ddof=1) / np.sqrt(len(x))), 3),
           "p_better": round(float(np.mean(x > 0)), 2), "p_worse": round(float(np.mean(x < 0)), 2), "n": len(x),
           "relative": round(float(x.mean() / max(np.mean(r[b]["scores"]), 1e-9)), 3)}
    if per:
        blocks = x.reshape(-1, per).mean(axis=1)
        out["p_roster_better"] = round(float(np.mean(blocks > 0)), 2)
    return out


STAGES = {"tb-hour": None, "tb-hour-boot": 6, "tb-hour-x0.5": None, "tb-hour-x1.33": None,
          "tb-nine": None, "tb-nine-boot23": 2, "tb-nine-x0.5": None, "tb-nine-x1.33": None}


def report(_args):
    summary = {}
    for stage, per in STAGES.items():
        path = Path(OUT, f"{stage}.result.json")
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        print(f"### {stage}\n\n| Cell | Mean (p10-p90) | Levels | vs busy 2x (same profile and switch) |")
        print("| --- | ---: | ---: | ---: |")
        rows = {}
        for label, v in r.items():
            tag, pol = label.split(" | ")
            ref = f"{tag} | busy 2x"
            x = None if pol == "busy 2x" else paired(r, label, ref, per)
            d = "-" if x is None else (f"{x['diff']:+.3f} ± {x['se']:.3f} ({100 * x['relative']:+.0f}%; "
                                       f"{int(100 * x['p_better'])}% of runs better, {int(100 * x['p_worse'])}% worse"
                                       + (f"; {int(100 * x['p_roster_better'])}% of rosters" if per else "") + ")")
            print(f"| {label} | {v['mean']:.3f} ({v['p10']:.2f}-{v['p90']:.2f}) | {v['levels']:.1f} | {d} |")
            rows[label] = {"mean": v["mean"], "se": v["se"], "p10": v["p10"], "p90": v["p90"], "levels": v["levels"],
                           **({} if x is None else {"vs_busy": x})}
        summary[stage] = rows
        print()
    Path(OUT, "summary.json").write_text(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["rates", "sim", "report"])
    parser.add_argument("--attempts", default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--runs-hour", type=int, default=200)
    parser.add_argument("--runs-nine", type=int, default=48)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    {"rates": rates, "sim": sim, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
