"""9 h / 1 h simulator cells for docs/validation-20260927/hero-v2-abc-compare.md (CPU, local; relative only).

The calibrated simulator of sim-mtp3-kv4.md (schedule_sim, world "within", fine start, fixed-2x token clock, six-clear
switch, 110 distinct games for 9 h, the c110 roster for 1 h), with the serving cells set from the measured phase-2
rates of the four runs and, for the untested counts, the in-run step fit t(n) = 15.0 + 0.49 n ms with 0.081 ms per
uncached prompt token. 128K windows enter only as a slower re-admission prefill and a clear-rate multiplier.

    python scripts/arena_analysis/level_attempts.py --out <dir>/attempts-now.jsonl      # the 98 earlier runs
    python scripts/arena_analysis/abc_sim.py --attempts <dir>/attempts-now.jsonl --runs 32 --out <json>
"""
import json, sys, time, copy
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/arena_analysis"), str(ROOT / "src")]
import argparse
import schedule_sim as sim
import numpy as np
ATTEMPTS = None
cfg = json.loads((ROOT / "docs/validation-20260927/sim-mtp3-kv4/main-live-nine.config.json").read_text())
cfgh = json.loads((ROOT / "docs/validation-20260927/sim-mtp3-kv4/main-live-hour.config.json").read_text())
extra = [str(ROOT / "docs/validation-20260927/c110-path/c110-attempts.jsonl")]
WORLD = None

def world(c):
    global WORLD
    if WORLD is None:
        WORLD = sim.World.fit(sim.load_rows(str(ATTEMPTS), tuple(extra), tuple(c["exclude_runs"])), shape=c["world"])
        WORLD.fine_start = c["fine_start"]
    return WORLD

# measured phase-2 aggregates (tok/s, this doc's speed section) and the in-run fit for the new counts
# t(n) = 15.0 + 0.49 n ms, prefill 0.081 ms per uncached token; uncached prompt per running stream: 64K ~50 tok/s, 128K ~49.5
def fit_rate(n, uncached_per_stream):
    t = 15.0 + 0.49 * n
    pre = n * uncached_per_stream * 0.081 / 1000.0
    return n * 1000.0 / t * (1 - pre)

CELLS = {
    # label: slots, p2 agg, opening agg, readmit window K, efficiency multiplier
    "hero-v2 23x64K int8 (810)": (23, 810.0, 707.0, 64, 1.0),
    "A 37x64K int4 (976)": (37, 976.0, 886.0, 64, 1.0),
    "A 37x64K int4 eff0.9": (37, 976.0, 886.0, 64, 0.9),
    "44x64K int4 (fit)": (44, None, 900.0, 64, 1.0),
    "B 24x128K int4 (807) eff1.0": (24, 807.0, 718.0, 128, 1.0),
    "B 24x128K int4 eff1.1": (24, 807.0, 718.0, 128, 1.1),
    "32x128K int4 (fit) eff1.0": (32, None, 760.0, 128, 1.0),
    "32x128K int4 (fit) eff1.1": (32, None, 760.0, 128, 1.1),
}

def spec(c, label, wall_cfg):
    slots, p2, op, wk, eff = CELLS[label]
    if p2 is None:
        p2 = fit_rate(slots, 50.0 if wk == 64 else 49.5)
    base = copy.deepcopy(wall_cfg["policies"]["base 23x64K MTP-0"])
    skw = base[2]
    skw.update(agg_rate=round(p2, 1), opening_rate=op, slots=slots, efficiency=eff,
               readmit_cost=[0.383, round(0.4 * (65536 if wk == 64 else 130816) / 10700.0, 3)])
    return base, round(p2, 1)

def job(args):
    global ATTEMPTS
    label, which, runs, ATTEMPTS = args
    c = cfg if which == "nine" else cfgh
    w = world(c)
    (pol, kw, skw), p2 = spec(c, label, c)
    games = sim.ROSTERS[str(c.get("roster", "23"))]
    t = time.time()
    res = sim.run_one(w, pol, kw, skw, runs, c["seed"], games, c.get("shared", False))
    return label, which, p2, res, time.time() - t

def main():
    global ATTEMPTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", required=True)
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--walls", default="nine,hour")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ATTEMPTS = args.attempts
    items = [(l, w, args.runs, args.attempts) for w in args.walls.split(",") for l in CELLS]
    with Pool(min(8, len(items))) as p:
        out = p.map(job, items)
    scores = {(w, l): np.array(res["scores"]) for l, w, _, res, _ in out}
    base = "hero-v2 23x64K int8 (810)"
    table = []
    for l, w, p2, res, sec in out:
        d = scores[(w, l)] - scores[(w, base)]
        table.append({"wall": w, "label": l, "p2_tok_s": p2, "mean": round(float(scores[(w, l)].mean()), 2),
                      "diff_vs_hero_v2": round(float(d.mean()), 2), "se": round(float(d.std() / np.sqrt(len(d))), 2),
                      "p_gt0": round(float(np.mean(d > 0)), 2)})
        print(table[-1])
    if args.out:
        args.out.write_text(json.dumps({"runs": args.runs, "cells": {k: list(v) for k, v in CELLS.items()},
                                        "table": table}, indent=1) + "\n")


if __name__ == "__main__":
    main()
