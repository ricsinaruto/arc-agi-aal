"""hero-v2 at 64K / 48K / 32K with the most streams that fit, fixed-2x (token budgets) against a naive fair scheduler
and a continuous ranking policy, in the calibrated simulator (29 September).

"compare hero-v2 variant on 64k context (max streams), 48k context (max streams), and 32k context
(max streams); compare with a naive (e.g. fair) scheduling; we need a better token (cost) and level clear-based
continuous ranking of each game, and then update the focus slots e.g. every 1 minute. The ranking should prioritize
games that cleared levels recently (according to token count), and if token counts are similar then games with
deeper levels. Optimize in sim."

    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/sim_hero_v2.py \
        --attempts <attempts-now.jsonl> <stage> [--runs N]
    ARC_VALIDATION_ROOT=<main checkout> uv run --extra modal modal run scripts/arena_analysis/modal_sim.py \
        --study hero_v2 --stages <stage> --run-id <id> --attempts <attempts-now.jsonl>

Stages: ``profiles`` (print the serving cells), ``smoke``, ``tune1`` (random search of the ranking constants on two
held-out folds of 10 games each), ``tune2`` (the top configurations of ``tune1`` on fresh seeds, own fold and the
other fold), ``final`` (every cell x every arm on the 20-game roster, fresh seeds, 1 h c110 and 9 h distinct),
``robust`` (bootstrap rosters, clear rates x0.5 / x1.33, a lower long-attempt hazard, rate and window-penalty
sensitivity, tick period), ``report``. Results go to ``docs/validation-20260927/sim-hero-v2/``.

Serving model (the derivation is in sim-hero-v2.md §1):

- Streams that fit: a 64K / 48K / 32K request costs 51 / 41 / 30 blocks (ceil((W + 256) / 1,584) attention blocks +
  9 fixed; the hero smoke's 51 per 64K and the 21 x 42,496 run's 36 check it), at the 0.9 gate of the pool:
  hero 860 blocks (774 at the gate: 15 / 18 / 25); hero-v2 adds the host embedding (+60 blocks) to the pruned pool:
  12.5% pruning 1,245 + 60 = 1,305 (1,174: 23 / 28 / 39), 25% pruning 1,667 + 60 = 1,727 (1,554: 30 / 37 / 51);
  streams = min(fit, max_num_seqs), max_num_seqs 25 (today) or 48.
- Phase-2 tokens per wall second at n running: the real arena's phase 2 (hero-6x64k 407 at 6.3, hero-14 605 at
  14.8, hero-21x42k 694 at 22.5) is linear in n per iteration: t(n) = 8.92 + 1.046 n ms (fit within 0.1 ms), so
  A(n) = 1000 n / t(n): 25 -> 713, 30 -> 744, 37 -> 777, 48 -> 812. Past 22.5 streams the slope is unmeasured:
  ``rate`` "low" continues it x1.5 above 22.5, "high" x0.6 (towards the smoke's decode-only slope, 0.43 ms).
- The opening (110 games, up to max_num_seqs running at once, 0.97 of it on average) runs 1.255x the phase-2
  iteration time (the hero opening: 564 tok/s at 24.2 running against A(24.2) = 707): 25 -> 564, 48 -> 643.
- The window as a clear-rate multiplier (streams-9h.md §8.3; compaction summaries alone take 2.2% / ~3.7% / ~12% of
  completion tokens at 64K / 48K / 32K): 64K 1.0; 48K 0.95 (0.90-1.0); 32K 0.80 (0.70-0.90).
- An admission costs whole-server prefill: a fresh game's first prompt 4.1K tokens, a parked game's context
  0.4 x window, at 10.7K tok/s (int8 prefill): 0.38 s fresh, 2.45 / 1.84 / 1.23 s at 64K / 48K / 32K.
- Level budgets are tokens (the arena's move to token budgets, 29 September): 2 x L x T1 generated tokens, T1 = the
  game's tokens to its first clear (opening tokens included, unweighted), full regrant at every clear and
  re-admission; a game without a clear is re-admitted with the tokens it had spent when first parked (floor 2K).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c110_path as cp  # noqa: E402
import schedule_sim as sim  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "validation-20260927" / "sim-hero-v2"

# ---------------------------------------------------------------------------------------------- serving model

P2_POINTS = ((6.3, 407.0), (14.8, 605.0), (22.5, 694.0))  # (running, tok/s): hero-6x64k, hero-14 preview, hero-21x42k
OPEN_POINT = (24.2, 564.0)  # hero-6x64k opening
_n = np.array([p[0] for p in P2_POINTS])
_t = np.array([1000.0 * p[0] / p[1] for p in P2_POINTS])
SLOPE, INTERCEPT = (float(v) for v in np.polyfit(_n, _t, 1))  # 1.046 ms per stream, 8.92 ms
MEASURED_MAX = 22.5
RATE_SLOPE = {"central": 1.0, "low": 1.5, "high": 0.6}  # slope multiplier above 22.5 streams


def iteration_ms(n, rate="central"):
    extra = max(0.0, n - MEASURED_MAX) * SLOPE * (RATE_SLOPE[rate] - 1.0)
    return INTERCEPT + SLOPE * n + extra


def p2_rate(n, rate="central"):
    return 1000.0 * n / iteration_ms(n, rate)


OPEN_FACTOR = 1000.0 * OPEN_POINT[0] / OPEN_POINT[1] / iteration_ms(OPEN_POINT[0])  # 1.255


def open_rate(cap, rate="central"):
    n = 0.97 * min(cap, 110)
    return 1000.0 * n / (OPEN_FACTOR * iteration_ms(n, rate))


BLOCK_TOKENS, FIXED_BLOCKS, GATE = 1584, 9, 0.9
POOLS = {"hero": 860, "p12": 1245 + 60, "p25": 1667 + 60}
WINDOWS = {64: 65536, 48: 49152, 32: 32768}
WINDOW_MULT = {64: 1.0, 48: 0.95, 32: 0.80}
WINDOW_RANGE = {48: (0.90, 1.0), 32: (0.70, 0.90)}
FRESH_PROMPT, PREFILL_RATE = 4100.0, 10700.0


def request_blocks(window):
    return -(-(WINDOWS[window] + 256) // BLOCK_TOKENS) + FIXED_BLOCKS


def fit(pool, window):
    return int(POOLS[pool] * GATE) // request_blocks(window)


def readmit(window):
    return (round(FRESH_PROMPT / PREFILL_RATE, 3), round(0.4 * WINDOWS[window] / PREFILL_RATE, 3))


def cell(pool, cap, window, slots=None, rate="central"):
    n = slots or min(fit(pool, window), cap)
    p2 = p2_rate(n, rate)
    return {"pool": pool, "cap": cap, "window": window, "slots": n, "open": round(open_rate(cap, rate), 1),
            "p2": round(p2, 1), "stream": round(p2 / n, 2), "mult": WINDOW_MULT[window], "readmit": readmit(window)}


def cells():
    out = {"hero 14x64K (ref)": cell("hero", 25, 64, slots=14)}
    for pool in ("p12", "p25"):
        for cap in (25, 48):
            for w in (64, 48, 32):
                c = cell(pool, cap, w)
                out[f"{pool} cap{cap} {c['slots']}x{w}K"] = c
    return out


CELLS = cells()
TUNE_CELLS = ["p25 cap25 25x64K", "p25 cap48 37x48K"]

# ---------------------------------------------------------------------------------------------- arms


def sim_kw(c, share, mult=None, **extra):
    return {"opening_rate": c["open"], "agg_rate": c["p2"], "slots": c["slots"], "opening_step": 100,
            "opening_share": share, "opening_share_sd": 0.35, "readmit_cost": list(c["readmit"]),
            "efficiency": c["mult"] if mult is None else mult, "shuffle": True, **extra}


FIXED_TOKENS = {"busy_factor": 1.0, "tokens_per_busy_second": 1.0, "floor_seconds": 2000.0}
OPENINGS = {"sw2": {"switch_games": 2}, "sw6": {"switch_games": 6}, "none": {"opening_max_wall": 0.0}}


def fixed2x(c, share, switch=6, **extra):
    return ["fixed-2x", dict(FIXED_TOKENS), sim_kw(c, share, switch_games=switch, **extra)]


def fair(c, share, quantum=20e3, **extra):
    """Naive fair share: no opening, the slots rotate on a token quantum, fewest tokens first; no level budgets."""
    return ["round-robin", {"quantum": quantum}, sim_kw(c, share, opening_max_wall=0.0, **extra)]


def fair_all(c, share, **extra):
    """No scheduling: all 110 games stay in the opening for the whole run (the gate runs max_num_seqs at a time)."""
    return ["fixed-2x", dict(FIXED_TOKENS), sim_kw(c, share, switch_games=10 ** 6, **extra)]


RANK_KEYS = ("horizon", "horizon_rel", "c0", "t0", "gamma", "hysteresis", "tenure", "tick", "clears",
             "budget_multiple")


def rank(c, share, cfg, **extra):
    kw = {k: cfg[k] for k in RANK_KEYS if k in cfg}
    return ["rank", kw, sim_kw(c, share, **OPENINGS[cfg.get("opening", "sw2")], **extra)]


DEFAULT_RANK = {"horizon": 40e3, "c0": 0.5, "t0": 20e3, "gamma": 1.0, "hysteresis": 0.3, "tenure": 10e3,
                "tick": 60.0, "clears": "all", "opening": "sw2"}
SPACE = {"horizon": [10e3, 20e3, 40e3, 80e3, 160e3], "horizon_rel": [1.0, 2.0, 4.0, 8.0],
         "c0": [0.25, 0.5, 1.0], "t0": [5e3, 10e3, 20e3, 40e3], "gamma": [0.0, 0.5, 1.0, 1.5, 2.0],
         "hysteresis": [0.05, 0.2, 0.5, 1.0], "tenure": [2e3, 5e3, 10e3, 20e3], "clears": ["all", "last"],
         "opening": ["sw2", "sw6", "none"], "budget_multiple": [None, 2.0]}


def sample_configs(n, seed=29):
    rng = np.random.default_rng(seed)
    out = [dict(DEFAULT_RANK)]
    while len(out) < n:
        c = {"tick": 60.0}
        for k, vals in SPACE.items():
            if k in ("horizon", "horizon_rel"):
                continue
            c[k] = vals[int(rng.integers(len(vals)))]
        if rng.random() < 0.5:
            c["horizon"] = SPACE["horizon"][int(rng.integers(5))]
        else:
            c["horizon"], c["horizon_rel"] = 40e3, SPACE["horizon_rel"][int(rng.integers(4))]
        if c.get("budget_multiple") is None:
            c.pop("budget_multiple", None)
        if c not in out:
            out.append(c)
    return out


def tag(cfg):
    h = f"hr{cfg['horizon_rel']:g}" if cfg.get("horizon_rel") else f"H{cfg['horizon'] / 1e3:g}K"
    b = f" b{cfg['budget_multiple']:g}" if cfg.get("budget_multiple") else ""
    t = f" tick{cfg['tick']:g}" if cfg.get("tick", 60.0) != 60.0 else ""
    return (f"rank {h} c{cfg['c0']:g} t{cfg['t0'] / 1e3:g}K g{cfg['gamma']:g} h{cfg['hysteresis']:g} "
            f"q{cfg['tenure'] / 1e3:g}K {cfg['clears']} {cfg['opening']}{b}{t}")


# ---------------------------------------------------------------------------------------------- rosters

HOUR = {**cp.CAL, "roster": "c110"}  # clones share their game's difficulty (the c110 roster)
NINE = {"wall": 32400.0, "shared": False}  # 110 distinct games
FOLD_SPLIT = 2  # c110-path/heldout-fits.json split 2: two halves of 10 games, both with deep games


def folds():
    split = json.loads(Path(cp.HERE, "heldout-fits.json").read_text())[FOLD_SPLIT]["halves"]
    return {f"F{i}": [half[j % 10] for j in range(110)] for i, half in enumerate(split)}


def baselines(c, share, extra):
    return {"fixed-2x sw6": fixed2x(c, share, 6, **extra), "fixed-2x sw2": fixed2x(c, share, 2, **extra),
            "fair q10K": fair(c, share, 10e3, **extra), "fair q20K": fair(c, share, 20e3, **extra),
            "fair q40K": fair(c, share, 40e3, **extra), "fair-all": fair_all(c, share, **extra)}


# ---------------------------------------------------------------------------------------------- stages


def smoke(runs, share):
    pol = {}
    for name in ("hero 14x64K (ref)", "p25 cap48 37x48K"):
        c = CELLS[name]
        for label, spec in {"fixed-2x sw6": fixed2x(c, share, 6), "fair q20K": fair(c, share),
                            "rank default": rank(c, share, DEFAULT_RANK)}.items():
            pol[f"{name} | {label}"] = spec
    nine = {n: [a, b, {**s, **NINE}] for n, (a, b, s) in pol.items()}
    return {"smoke-hour": {**HOUR, "runs": runs or 4, "seed": 90, "policies": pol},
            "smoke-nine": {**cp.CAL, "runs": max(1, (runs or 4) // 2), "seed": 91, "policies": nine}}


N_TUNE1 = 240
TUNE_RUNS = {"hour": 24, "nine": 8}


def _fold_configs(name, cfgs, runs, seed, share, with_baselines=True):
    out = {}
    for f, roster in folds().items():
        for hz in ("hour", "nine"):
            pol = {}
            extra = {"roster_games": roster, **(NINE if hz == "nine" else {})}
            for cname in TUNE_CELLS:
                c = CELLS[cname]
                if with_baselines:
                    for k, v in baselines(c, share, extra).items():
                        pol[f"{cname} | {k}"] = v
                for cfg in cfgs:
                    pol[f"{cname} | {tag(cfg)}"] = rank(c, share, cfg, **extra)
            base = HOUR if hz == "hour" else cp.CAL
            out[f"{name}-{f}-{hz}"] = {**base, "runs": runs[hz], "seed": seed + (0 if hz == "hour" else 1),
                                       "policies": pol}
    return out


def tune1(runs, share):
    return _fold_configs("tune1", sample_configs(N_TUNE1), TUNE_RUNS, 100, share)


TOP = 10
TUNE2_RUNS = {"hour": 96, "nine": 24}


def _load(d, name):
    p = Path(d, f"{name}.result.json")
    return json.loads(p.read_text()) if p.exists() else None


def objective(r, label_tag):
    """Mean score of one configuration over the tuning cells."""
    return float(np.mean([r[f"{c} | {label_tag}"]["mean"] for c in TUNE_CELLS]))


def top_configs(d=OUT, n=TOP):
    """{(fold, horizon): [cfg, ...]}: the ``n`` best tune1 configurations of every fold and horizon, by own-fold
    mean over the tuning cells."""
    cfgs = {tag(c): c for c in sample_configs(N_TUNE1)}
    out = {}
    for f in folds():
        for hz in ("hour", "nine"):
            r = _load(d, f"tune1-{f}-{hz}")
            if r is None:
                raise SystemExit(f"tune1-{f}-{hz} missing")
            best = sorted(cfgs, key=lambda t: -objective(r, t))[:n]
            out[(f, hz)] = [cfgs[t] for t in best]
    return out


def tune2(runs, share):
    """The union of every fold's and horizon's top configurations, re-run on fresh seeds on both folds."""
    top = top_configs()
    union = []
    for v in top.values():
        for c in v:
            if c not in union:
                union.append(c)
    return _fold_configs("tune2", union, TUNE2_RUNS, 200, share)


def selected(d=OUT):
    """The chosen configurations: per (fold, horizon) the best of tune2 on its own fold (its score on the other fold is
    the held-out check), and per horizon the best mean over both folds (``global``)."""
    top = top_configs(d)
    union = []
    for v in top.values():
        for c in v:
            if c not in union:
                union.append(c)
    out = {}
    for hz in ("hour", "nine"):
        r = {f: _load(d, f"tune2-{f}-{hz}") for f in folds()}
        if any(v is None for v in r.values()):
            raise SystemExit("tune2 missing")
        for f in folds():
            best = max(union, key=lambda c: objective(r[f], tag(c)))
            out[f"{f}-{hz}"] = best
        out[f"global-{hz}"] = max(union, key=lambda c: np.mean([objective(r[f], tag(c)) for f in folds()]))
    return out


FINAL_RUNS = {"hour": 40, "nine": 16}


def final_arms(c, share, sel, extra):
    pol = {"fixed-2x sw6": fixed2x(c, share, 6, **extra), "fixed-2x sw2": fixed2x(c, share, 2, **extra),
           "fair q20K": fair(c, share, 20e3, **extra), "fair q40K": fair(c, share, 40e3, **extra),
           "fair-all": fair_all(c, share, **extra)}
    seen = {}
    for k in ("global-nine", "global-hour"):
        cfg = sel[k]
        if tag(cfg) in seen:
            continue
        seen[tag(cfg)] = k
        pol[f"rank[{k[7:]}] {tag(cfg)}"] = rank(c, share, cfg, **extra)
    return pol


def final(runs, share):
    sel = selected()
    out = {}
    for hz in ("hour", "nine"):
        pol = {}
        extra = NINE if hz == "nine" else {}
        for name, c in CELLS.items():
            for k, v in final_arms(c, share, sel, extra).items():
                pol[f"{name} | {k}"] = v
        base = HOUR if hz == "hour" else cp.CAL
        out[f"final-{hz}"] = {**base, "runs": runs or FINAL_RUNS[hz], "seed": 300 + (hz == "nine"), "policies": pol}
    return out


ROBUST_CELLS = ["hero 14x64K (ref)", "p12 cap25 23x64K", "p25 cap25 25x64K", "p25 cap48 30x64K", "p25 cap48 37x48K",
                "p25 cap48 48x32K"]
TAIL = {"from": 40000, "scale": 0.5}


def robust(runs, share):
    """Paired checks of rank against fixed-2x on the key cells: game bootstrap (20 rosters x 6 runs at 1 h, 12 x 2 at
    9 h), clear rates x0.5 / x1.33, a halved hazard beyond 40K tokens on a level (the depthwise miss), the
    unmeasured rate slope past 22.5 streams, the window penalty range, and the tick period."""
    sel = selected()
    out = {}
    b1 = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": cp.BOOT_PER}
    b9 = {"pool": sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}

    def arms(c, extra, mult=None):
        m = {} if mult is None else {"efficiency": mult}
        pol = {"fixed-2x sw6": fixed2x(c, share, 6, **extra, **m), "fixed-2x sw2": fixed2x(c, share, 2, **extra, **m),
               "fair q40K": fair(c, share, 40e3, **extra, **m)}
        for k in ("global-nine", "global-hour"):
            pol[f"rank[{k[7:]}]"] = rank(c, share, sel[k], **extra, **m)
        return pol

    for hz, base, extra, boot, nrun in (("hour", HOUR, {}, b1, 40), ("nine", {**cp.CAL, "roster": "23"}, NINE, b9, 16)):
        pol = {}
        for name in ROBUST_CELLS:
            for k, v in arms(CELLS[name], {**extra, "bootstrap": boot}).items():
                pol[f"{name} | {k}"] = v
        per = boot["per_roster"]
        out[f"robust-{hz}-boot"] = {**base, "runs": (20 if hz == "hour" else 12) * per, "seed": 400, "policies": pol}
        base2 = HOUR if hz == "hour" else cp.CAL
        for s in (0.5, 1.33):
            pol = {}
            for name in ROBUST_CELLS:
                c = CELLS[name]
                for k, v in arms(c, extra, c["mult"] * s).items():
                    pol[f"{name} | {k}"] = v
            out[f"robust-{hz}-x{s:g}"] = {**base2, "runs": nrun, "seed": 410, "policies": pol}
        pol = {}
        for name in ROBUST_CELLS:
            for k, v in arms(CELLS[name], {**extra, "tail": TAIL}).items():
                pol[f"{name} | {k}"] = v
        out[f"robust-{hz}-tail"] = {**base2, "runs": nrun, "seed": 420, "policies": pol}
        # serving sensitivity: rate slope above 22.5 streams, and the window penalty range
        pol = {}
        for name in ("p25 cap48 30x64K", "p25 cap48 37x48K", "p25 cap48 48x32K", "p12 cap48 39x32K"):
            c0 = CELLS[name]
            for rate in ("low", "high"):
                c = cell(c0["pool"], c0["cap"], c0["window"], slots=c0["slots"], rate=rate)
                for k, v in arms(c, extra).items():
                    if k in ("fixed-2x sw6", "fixed-2x sw2", "rank[nine]"):
                        pol[f"{name} rate-{rate} | {k}"] = v
            for m in WINDOW_RANGE.get(c0["window"], ()):
                for k, v in arms(c0, extra, m).items():
                    if k in ("fixed-2x sw6", "fixed-2x sw2", "rank[nine]"):
                        pol[f"{name} m{m:g} | {k}"] = v
        for name in ("p25 cap25 25x64K", "p25 cap48 37x48K"):
            c = CELLS[name]
            for tick in (30.0, 180.0, 600.0):
                cfg = {**sel["global-nine"], "tick": tick}
                pol[f"{name} | rank[nine] tick{tick:g}"] = rank(c, share, cfg, **extra)
        out[f"robust-{hz}-serving"] = {**base2, "runs": nrun, "seed": 430, "policies": pol}
    return out


TICK_CELLS = {"hero-v2 23x64K (p12, max_num_seqs 25)": cell("p12", 25, 64),
              "25x64K (p25, max_num_seqs 25)": cell("p25", 25, 64),
              "30x64K (p25, max_num_seqs 32)": cell("p25", 32, 64)}
TICKS = (0.0, 60.0, 180.0, 300.0, 600.0, 900.0)  # 0: re-rank at every simulator step (the per-request variant)
TICK_STEP = 250.0  # tokens per focus game per simulator step: ~9 s at 25 streams, about one request's reply
RANK_NINE = {"horizon": 40e3, "horizon_rel": 1.0, "c0": 0.25, "t0": 5e3, "gamma": 1.5, "hysteresis": 0.05,
             "tenure": 10e3, "clears": "all", "opening": "none"}


def tick_label(t):
    return "rank per-step" if t == 0 else f"rank tick {t:g}s"


def ticks(runs, share):
    """"if 600 s scores similarly shouldn't we use that since it will allow for longer potential
    cache reuse before swapping?" The rank[nine] constants at ticks 60-900 s and at every simulator step, on the three
    64K cells, 1 h c110 and 9 h distinct, fresh seeds and the game bootstrap; on the base seeds also a free and a 3x
    re-prefill (cache sensitivity) and the 60 s tick at the main study's 1K step."""
    b1 = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": cp.BOOT_PER}
    b9 = {"pool": sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}
    out = {}
    for hz, base, extra, boot, nrun, nboot in (
            ("hour", HOUR, {}, b1, 40, 20 * cp.BOOT_PER), ("nine", cp.CAL, NINE, b9, 16, 24)):
        for kind in ("base", "boot"):
            pol = {}
            for name, c in TICK_CELLS.items():
                ex = {**extra, "step": TICK_STEP, **({"bootstrap": boot} if kind == "boot" else {})}
                pol[f"{name} | fixed-2x sw6"] = fixed2x(c, share, 6, **ex)
                for t in TICKS:
                    pol[f"{name} | {tick_label(t)}"] = rank(c, share, {**RANK_NINE, "tick": t}, **ex)
                if kind == "base":
                    for k in (0.0, 3.0):
                        cost = [c["readmit"][0], round(c["readmit"][1] * k, 3)]
                        for t in (60.0, 600.0):
                            pol[f"{name} | {tick_label(t)} reprefill x{k:g}"] = rank(
                                c, share, {**RANK_NINE, "tick": t}, **{**ex, "readmit_cost": cost})
                    pol[f"{name} | rank tick 60s step 1K"] = rank(c, share, {**RANK_NINE, "tick": 60.0},
                                                                  **{**ex, "step": 1000.0})
            cfg = base if kind == "base" else ({**cp.CAL, "roster": "23"} if hz == "nine" else HOUR)
            out[f"tick-{hz}-{kind}"] = {**cfg, "runs": nrun if kind == "base" else nboot,
                                        "seed": 500 + 2 * (hz == "nine") + (kind == "boot"), "policies": pol}
    return out


SWITCH_CELLS = {"hero-v2 23x64K (p12, max_num_seqs 25)": TICK_CELLS["hero-v2 23x64K (p12, max_num_seqs 25)"],
                "30x64K (p25, max_num_seqs 32)": TICK_CELLS["30x64K (p25, max_num_seqs 32)"]}
# (label, simulate kwargs): distinct level-1 clears; coverage (every opened game has generated X tokens, no clear
# needed); the six-clear switch with the scheduler's wall-clock cap (opening_max_wall_seconds); no opening.
SWITCHES = {**{f"sw{k}": {"switch_games": k} for k in (1, 2, 4, 6, 10)},
            **{f"cover {x // 1000}K": {"switch_games": 0, "opening_tokens": float(x)} for x in (4000, 8000, 16000)},
            "sw6 cap 600 s": {"switch_games": 6, "opening_max_wall": 600.0},
            "sw6 cap 1200 s": {"switch_games": 6, "opening_max_wall": 1200.0},
            "cover 4K cap 1200 s": {"switch_games": 0, "opening_tokens": 4000.0, "opening_max_wall": 1200.0},
            "no opening": {"opening_max_wall": 0.0}}
FIXED_TOKEN_CLOCK = {"busy_factor": 1.0, "tokens_per_busy_second": 1.0, "floor_seconds": 1920.0}  # TOKEN_FLOOR


def switch(runs, share):
    """"is 6 level-1 clears a sensible switch condition for starting phase 2?" fixed-2x on the
    token clock (hero-v2's scheduler: 2 L T1 tokens, full regrant, level-0 grant = tokens at the first park, floor
    1,920 tokens) under every opening rule in ``SWITCHES``, at hero-v2's 23 x 64K and at 30 x 64K."""
    b1 = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": cp.BOOT_PER}
    b9 = {"pool": sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}
    out = {}
    for hz, extra, boot, nrun, nboot in (("hour", {}, b1, 40, 20 * cp.BOOT_PER), ("nine", NINE, b9, 16, 24)):
        for kind in ("base", "boot"):
            pol = {}
            for name, c in SWITCH_CELLS.items():
                for label, sw in SWITCHES.items():
                    skw = sim_kw(c, share, **extra, **sw, **({"bootstrap": boot} if kind == "boot" else {}))
                    pol[f"{name} | {label}"] = ["fixed-2x", dict(FIXED_TOKEN_CLOCK), skw]
            cfg = (HOUR if hz == "hour" else cp.CAL) if kind == "base" else (
                HOUR if hz == "hour" else {**cp.CAL, "roster": "23"})
            out[f"switch-{hz}-{kind}"] = {**cfg, "runs": nrun if kind == "base" else nboot,
                                          "seed": 600 + 2 * (hz == "nine") + (kind == "boot"), "policies": pol}
    return out


STAGES = {"smoke": smoke, "tune1": tune1, "tune2": tune2, "final": final, "robust": robust, "ticks": ticks,
          "switch": switch}

# ---------------------------------------------------------------------------------------------- output


def summarise(res):
    out = cp.summarise(res)
    if "stats" in res:
        out["stats"] = res["stats"]
    for k in ("played", "cleared1"):
        if k in res:
            v = np.array(res[k], dtype=float)
            out[k] = {"mean": round(float(v.mean()), 2), "p10": float(np.percentile(v, 10)),
                      "p90": float(np.percentile(v, 90))}
    return {**out, "scores": res["scores"]}


def write(name, config, res, out_dir):
    out = {label: summarise(r) for label, r in res.items()}
    Path(out_dir, f"{name}.config.json").write_text(json.dumps(config, indent=1))
    Path(out_dir, f"{name}.result.json").write_text(json.dumps(out, indent=1))
    for label, r in out.items():
        print(f"  {label:78s} {r['mean']:6.2f} ± {r['se']:.2f} p10/50/90 {r['p10']:.2f}/{r['p50']:.2f}/{r['p90']:.2f} "
              f"levels {r['levels']:6.1f}", flush=True)


def run_local(name, config, attempts, out_dir, jobs):
    write(name, config, sim.sweep(config, attempts, jobs), out_dir)


def paired(r, a, b, per=None):
    x = np.array(r[a]["scores"]) - np.array(r[b]["scores"])
    out = {"diff": round(float(x.mean()), 2), "se": round(float(x.std() / math.sqrt(len(x))), 2),
           "p_run": round(float(np.mean(x > 0)), 2)}
    if per:
        out["p_roster"] = round(float(np.mean(x.reshape(-1, per).mean(axis=1) > 0)), 2)
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
        print(json.dumps({"slope_ms": SLOPE, "intercept_ms": INTERCEPT, "open_factor": OPEN_FACTOR,
                          "blocks": {w: request_blocks(w) for w in WINDOWS},
                          "fit": {p: {w: fit(p, w) for w in WINDOWS} for p in POOLS}, "cells": CELLS}, indent=1))
        return
    if args.stage == "report":
        import sim_hero_v2_report
        sim_hero_v2_report.report(args.out_dir)
        return
    share = cp.opening_share()
    for name, cfg in STAGES[args.stage](args.runs, share).items():
        if args.only and name not in args.only:
            continue
        print(f"== {name}: {len(cfg['policies'])} policies x {cfg['runs']} runs", flush=True)
        run_local(name, cfg, args.attempts, args.out_dir, args.jobs)


if __name__ == "__main__":
    main()
