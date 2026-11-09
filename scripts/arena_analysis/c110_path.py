"""Calibrated c110 simulator sweeps: what reaches 3% on c110 in one hour, and the 9-hour hidden run (28 September).

Drives ``schedule_sim.sweep`` with the calibration measured from the seven real c110 runs (``c110_runs.py``,
docs/validation-20260927/c110-path/c110-summary.json):

- serving profiles with the measured opening and phase-2 rates (tokens per wall second over all games) and busy-clock
  rates (fixed-2x budgets), and projected profiles from the serving lab's per-stream rates;
- the opening's per-game token share (games with short replies, like sb26, get fewer tokens per wall second);
- the world fitted to the recorded attempts plus the c110 runs' own level attempts.

    PYTHONPATH=src uv run python scripts/arena_analysis/c110_path.py --attempts <attempts.jsonl> <stage> [--runs N]

Stages: ``calibrate``, ``path`` (throughput x slots x scheduler x efficiency), ``opening``, ``legal`` (in-run
scheduler ideas), ``ninehour``. Each writes ``<stage>-<name>.config.json`` / ``.result.json`` to ``--out-dir``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schedule_sim as sim  # noqa: E402

HERE = Path(__file__).resolve().parents[2] / "docs" / "validation-20260927" / "c110-path"
C110_ATTEMPTS = str(HERE / "c110-attempts.jsonl")
LEVEL_PROBE = "arena-duck-60m-c110-level-probe-v1-modal-20260927"

# Serving profiles. "open"/"p2": tokens per wall second over all games in the opening / in phase 2; "busy_open"/
# "busy_p2": tokens per busy second of one game (the fixed-2x budget clock); "stream": the per-stream rate the
# Gittins index plans with. mtp3-2 and mtp0-6 are measured on the real c110 runs (means over the runs of each
# serving; c110-summary.json). The others are projections from the serving lab (serving-lab-v1-result.md): MTP-0
# decode-only 440 tok/s at 8 streams, 650-680 at 16 x 16K, ~900 at 24 x 4K; the real mtp0-6 phase-2 per-stream
# rate (54) equals the lab's decode-only per-stream rate at 8 streams (52-55), so the lab's rates are used as is.
# The opening of a projected profile runs at max(measured MTP-0 opening, phase-2 rate).
PROFILES = {
    "260/2 (MTP-3, measured)": {"open": 325.0, "p2": 214.0, "slots": 2, "busy_open": 66.0, "busy_p2": 107.0,
                                "stream": 107.0},
    "390/6 (MTP-0 64K, measured)": {"open": 486.0, "p2": 323.0, "slots": 6, "busy_open": 32.4, "busy_p2": 54.5,
                                    "stream": 54.0},
    "550/12 (MTP-0 32K)": {"open": 550.0, "p2": 550.0, "slots": 12, "busy_open": 32.4, "busy_p2": 46.0,
                           "stream": 46.0},
    "700/16 (MTP-0 16K)": {"open": 700.0, "p2": 700.0, "slots": 16, "busy_open": 32.4, "busy_p2": 44.0,
                           "stream": 44.0},
    "900/24 (MTP-0 <=12K)": {"open": 900.0, "p2": 900.0, "slots": 24, "busy_open": 32.4, "busy_p2": 37.5,
                             "stream": 37.5},
}
GITTINS_C110 = {"model": "gittins", "shape": "within", "k": 0.53, "ke": 0.67, "beta": 1.14, "prior_scale": "run",
                "hysteresis": 4.0}


def opening_share(summary_path=HERE / "c110-summary.json"):
    """Mean relative opening token rate per source game over the real c110 runs (without the level probe)."""
    runs = [s for s in json.loads(Path(summary_path).read_text()) if s["label"] != "level-probe"]
    games = sorted({g for s in runs for g in s["opening_share"]})
    return {g: round(float(np.mean([s["opening_share"][g] for s in runs])), 3) for g in games}


def serving_sim(p, share=None, share_sd=0.0):
    out = {"opening_rate": p["open"], "agg_rate": p["p2"], "slots": p["slots"], "opening_step": 100}
    if share:
        out["opening_share"] = share
        out["opening_share_sd"] = share_sd
    return out


def fixed2x(p):
    return ["fixed-2x", {"busy_factor": p["busy_p2"] / p["busy_open"], "tokens_per_busy_second": p["busy_p2"]}]


def gittins(p, **extra):
    return ["index", {**GITTINS_C110, "stream_tokens_per_second": p["stream"], **extra}]


def known(p):
    return ["known-rate", {}]


def world_keys(world):
    """``w0``: the 98 earlier runs (the 27 September world); ``w1``: plus the c110 runs' attempts (not the probe)."""
    if world == "w0":
        return {}
    return {"extra_attempts": [C110_ATTEMPTS], "exclude_runs": [LEVEL_PROBE]}


def summarise(res):
    s = np.array(res["scores"])
    return {"mean": round(float(s.mean()), 3), "se": round(float(s.std() / np.sqrt(len(s))), 3),
            "p10": round(float(np.percentile(s, 10)), 3), "p50": round(float(np.percentile(s, 50)), 3),
            "p90": round(float(np.percentile(s, 90)), 3), "p_ge_3": round(float((s >= 3.0).mean()), 3),
            "levels": round(res["levels"], 1), "switch": round(res["switch"]), "per_game_levels": res["per_game_levels"]}


def run_config(name, config, attempts, out_dir, jobs):
    t = time.time()
    print(f"== {name}: {len(config['policies'])} policies x {config.get('runs')} runs", flush=True)
    res = sim.sweep(config, attempts, jobs)
    out = {label: {**summarise(r), "scores": r["scores"]} for label, r in res.items()}
    Path(out_dir, f"{name}.config.json").write_text(json.dumps(config, indent=1))
    Path(out_dir, f"{name}.result.json").write_text(json.dumps(out, indent=1))
    for label, r in out.items():
        print(f"  {label:60s} {r['mean']:5.2f} ± {r['se']:.2f}  p10/50/90 {r['p10']:.2f}/{r['p50']:.2f}/{r['p90']:.2f}"
              f"  P(>=3) {r['p_ge_3']:.2f}  levels {r['levels']:5.1f}  switch {r['switch']:5d}", flush=True)
    print(f"   ({time.time() - t:.0f} s)", flush=True)
    return out


# ------------------------------------------------------------------------------------------ stages


def calibrate(runs, share):
    """The five scheduler runs on c110 (world model and level probe are harness changes), under the old and the
    calibrated assumptions."""
    m3, m0 = PROFILES["260/2 (MTP-3, measured)"], PROFILES["390/6 (MTP-0 64K, measured)"]
    configs = {}
    # The 27 September assumptions (scheduler-c110.md b5 / c1), to reproduce the earlier predictions.
    old = {"fixed-2x @230": ["fixed-2x", {}, {"opening_rate": 280, "slots": 2}],
           "gittins-c110 @230": ["index", dict(GITTINS_C110), {"opening_rate": 280, "slots": 2, "opening_max_wall": 1200}],
           "fixed-2x sw6 @450x6": ["fixed-2x", {"busy_factor": 1.0, "tokens_per_busy_second": 70},
                                  {"opening_rate": 450, "agg_rate": 450, "slots": 6, "switch_games": 6}],
           "gittins-c110 @450x6": ["index", {**GITTINS_C110, "stream_tokens_per_second": 75.0},
                                   {"opening_rate": 450, "agg_rate": 450, "slots": 6, "opening_max_wall": 1200}]}
    for world in ("w0", "w1"):
        for sh in ("even", "share"):
            pol = {}
            for tag, p, sw in (("mtp3-2", m3, 2), ("mtp0-6", m0, None)):
                skw = serving_sim(p, share if sh == "share" else None, 0.35 if sh == "share" else 0.0)
                # The deployed index rates: 115 tok/s (gittins-c110), 75 (gittins-c110-wide).
                deployed = 115.0 if tag == "mtp3-2" else 75.0
                pol[f"fixed-2x {tag}" + (" sw6" if tag == "mtp0-6" else "")] = [
                    *fixed2x(p), {**skw, "switch_games": 2 if tag == "mtp3-2" else 6}]
                pol[f"gittins-c110 {tag}"] = [*gittins(p, stream_tokens_per_second=deployed),
                                              {**skw, "opening_max_wall": 1200}]
                pol[f"known-rate {tag}"] = ["known-rate", {}, {**skw, "switch_games": 2}]
                pol[f"oracle {tag}"] = ["oracle", {}, {**skw, "switch_games": 2}]
            configs[f"calibrate-{world}-{sh}"] = {"world": "within", "roster": "c110", "runs": runs, "seed": 1,
                                                  "shared": True, "policies": pol, **world_keys(world)}
            # The calibrated world: plus the 1K sub-band shape inside the first 5K tokens of a level.
            configs[f"calibrate-{world}-{sh}-fine"] = {**configs[f"calibrate-{world}-{sh}"], "fine_start": True}
        configs[f"calibrate-{world}-old"] = {"world": "within", "roster": "c110", "runs": runs, "seed": 1,
                                             "shared": True, "sim": {"opening_step": 100}, "policies": old,
                                             **world_keys(world)}
    return configs


CAL = {"world": "within", "roster": "c110", "shared": True, "fine_start": True, **world_keys("w1")}
EFFS = (1.0, 1.5, 2.0, 3.0)


def cal_sim(p, share, **extra):
    return {**serving_sim(p, share, 0.35), **extra}


def path(runs, share):
    """Throughput profile x scheduler x per-token efficiency, in the calibrated world (two-clear opening; the index
    keeps its 1,200 s cap). One config per profile."""
    configs = {}
    for i, (pname, p) in enumerate(PROFILES.items()):
        pol = {}
        for eff in EFFS:
            e = f"x{eff:g}"
            skw = cal_sim(p, share, efficiency=eff)
            pol[f"fixed-2x {e}"] = [*fixed2x(p), skw]
            pol[f"gittins-c110 {e}"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
            pol[f"gittins-c110 cap1/source {e}"] = [*gittins(p, per_source_cap=1), {**skw, "opening_max_wall": 1200}]
            pol[f"known-rate {e}"] = ["known-rate", {}, skw]
            pol[f"oracle {e}"] = ["oracle", {}, skw]
        configs[f"path-p{i}"] = {**CAL, "runs": runs, "seed": 2, "profile": pname, "policies": pol}
    return configs


def slots(runs, share):
    """Aggregate throughput and focus slots separated: the same phase-2 rate split over 2 / 6 / 12 / 16 streams."""
    configs = {}
    for agg in (390.0, 550.0, 700.0):
        pol = {}
        for n in (2, 6, 12, 16):
            if agg == 390.0 and n == 16:
                continue
            p = {"open": max(486.0, agg), "p2": agg, "slots": n, "busy_open": 32.4, "busy_p2": agg / n,
                 "stream": agg / n}
            for eff in (1.0, 2.0):
                skw = cal_sim(p, share, efficiency=eff)
                pol[f"fixed-2x {n} slots x{eff:g}"] = [*fixed2x(p), skw]
                pol[f"gittins-c110 {n} slots x{eff:g}"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
                pol[f"known-rate {n} slots x{eff:g}"] = ["known-rate", {}, skw]
        configs[f"slots-{int(agg)}"] = {**CAL, "runs": runs, "seed": 3, "policies": pol}
    return configs


def opening(runs, share):
    """Opening variants: the two-clear switch (uncapped / 1,200 s cap), absolute caps of 300 / 600 s, and no opening
    (phase 2 from t = 0)."""
    configs = {}
    for i, pname in enumerate(list(PROFILES)[:3]):
        p = PROFILES[pname]
        pol = {}
        for eff in (1.0, 2.0):
            for oname, okw in (("two-clear", {}), ("cap 1200", {"opening_max_wall": 1200}),
                               ("cap 600", {"opening_max_wall": 600}), ("cap 300", {"opening_max_wall": 300}),
                               ("none", {"opening_max_wall": 0.0})):
                skw = cal_sim(p, share, efficiency=eff, **okw)
                pol[f"fixed-2x {oname} x{eff:g}"] = [*fixed2x(p), skw]
                pol[f"gittins-c110 {oname} x{eff:g}"] = [*gittins(p), skw]
                pol[f"known-rate {oname} x{eff:g}"] = ["known-rate", {}, skw]
        configs[f"opening-p{i}"] = {**CAL, "runs": runs, "seed": 4, "profile": pname, "policies": pol}
    return configs


def legal(runs, share):
    """Scheduler ideas that transfer to 110 distinct games (in-run signals, generic constants), against the
    known-game (the game factor known, level factors learned) and known-rate bounds."""
    configs = {}
    for i, pname in enumerate(("390/6 (MTP-0 64K, measured)", "550/12 (MTP-0 32K)")):
        p = PROFILES[pname]
        skw = cal_sim(p, share, opening_max_wall=1200)
        pol = {
            "fixed-2x": [*fixed2x(p), cal_sim(p, share)],
            "gittins-c110 (deployed rate 75)": [*gittins(p, stream_tokens_per_second=75.0), skw],
            "gittins-c110 (measured stream rate)": [*gittins(p), skw],
            "gittins h2": [*gittins(p, hysteresis=2.0), skw],
            "gittins h8": [*gittins(p, hysteresis=8.0), skw],
            "gittins h4 quantum 20K": [*gittins(p, quantum_tokens=20000), skw],
            "gittins h4 caps 20-160K": [*gittins(p, caps=[20000, 40000, 80000, 160000]), skw],
            "gittins h4 caps 10-320K": [*gittins(p, caps=[10000, 20000, 40000, 80000, 160000, 320000]), skw],
            "gittins h4 fixed prior": [*gittins(p, prior_scale="fixed"), skw],
            "gittins h4 opening 40 games": [*gittins(p), {**skw, "opening_games": 40}],
            "gittins h4 opening 20 games": [*gittins(p), {**skw, "opening_games": 20}],
            "hazard index": ["index", {"stream_tokens_per_second": p["stream"]}, skw],
            "gittins cap1/source (c110 only)": [*gittins(p, per_source_cap=1), skw],
            "known-game": ["known-game", {}, skw],
            "known-rate": ["known-rate", {}, skw],
            "oracle": ["oracle", {}, skw],
        }
        configs[f"legal-p{i + 1}"] = {**CAL, "runs": runs, "seed": 5, "profile": pname, "policies": pol}
    return configs


def legal2(runs, share):
    """The selective opening on a shuffled roster (a random subset opens, not one clone of each source game), on
    the c110 roster (clones share their game's difficulty) and on 110 distinct draws (``shared`` false)."""
    configs = {}
    for i, pname in enumerate(("390/6 (MTP-0 64K, measured)", "550/12 (MTP-0 32K)")):
        p = PROFILES[pname]
        pol = {}
        for tag, extra in (("c110", {}), ("distinct", {"shared": False})):
            skw = cal_sim(p, share, opening_max_wall=1200, shuffle=True, **extra)
            pol[f"fixed-2x {tag}"] = [*fixed2x(p), {k: v for k, v in skw.items() if k != "opening_max_wall"}]
            pol[f"gittins-c110 {tag}"] = [*gittins(p), skw]
            for n in (20, 40, 60):
                pol[f"gittins opening {n} games {tag}"] = [*gittins(p), {**skw, "opening_games": n}]
            pol[f"gittins switch at 1st clear {tag}"] = [*gittins(p), {**skw, "switch_games": 1}]
            pol[f"known-game {tag}"] = ["known-game", {}, skw]
            pol[f"known-rate {tag}"] = ["known-rate", {}, skw]
        configs[f"legal2-p{i + 1}"] = {**CAL, "runs": runs, "seed": 7, "profile": pname, "policies": pol}
    return configs


BOOT_ROSTERS, BOOT_PER = 20, 6  # 20 bootstrap rosters x 6 runs each


def boot(runs, share):
    """Game-level bootstrap (robustness): every block of 6 runs plays 110 clones of 20 games resampled with
    replacement from the 20 c110 source games; the same 20 rosters for every policy."""
    b = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": BOOT_PER}
    configs = {}
    for i, pname in enumerate(PROFILES):
        p = PROFILES[pname]
        pol = {}
        for eff in (1.0, 2.0):
            skw = cal_sim(p, share, efficiency=eff, bootstrap=b)
            pol[f"fixed-2x x{eff:g}"] = [*fixed2x(p), skw]
            pol[f"gittins-c110 x{eff:g}"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
            pol[f"gittins-c110 cap600 x{eff:g}"] = [*gittins(p), {**skw, "opening_max_wall": 600}]
        configs[f"boot-p{i}"] = {**CAL, "runs": BOOT_ROSTERS * BOOT_PER, "seed": 11, "profile": pname, "policies": pol}
    return configs


def sens(runs, share):
    """Difficulty sensitivity (robustness): every clear rate x0.5 / 0.75 / 1.33 (the policies' constants unchanged)."""
    configs = {}
    for i, pname in enumerate(list(PROFILES)[:3]):
        p = PROFILES[pname]
        pol = {}
        for eff in (0.5, 0.75, 1.33):
            skw = cal_sim(p, share, efficiency=eff)
            pol[f"fixed-2x x{eff:g}"] = [*fixed2x(p), skw]
            pol[f"gittins-c110 x{eff:g}"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
            pol[f"gittins-c110 cap600 x{eff:g}"] = [*gittins(p), {**skw, "opening_max_wall": 600}]
            pol[f"known-rate x{eff:g}"] = ["known-rate", {}, skw]
        configs[f"sens-p{i}"] = {**CAL, "runs": runs, "seed": 12, "profile": pname, "policies": pol}
    return configs


def heldout_fits(attempts, splits=4, seed=0):
    """Per split, the two halves of the 20 c110 source games and, per half, the within-level fit on every recorded
    attempt except that half's games (the constants a Gittins index would have without ever seeing them)."""
    import within_level_hazard as wlh

    rows = sim.load_rows(attempts, [C110_ATTEMPTS], [LEVEL_PROBE])
    rng = np.random.default_rng(seed)
    out = []
    for s in range(splits):
        perm = [sim.C20[i] for i in rng.permutation(len(sim.C20))]
        halves = (sorted(perm[:10]), sorted(perm[10:]))
        fits = []
        for half in halves:
            f = wlh.fit([r for r in rows if r["game"] not in half], "hier")
            fits.append({"first": f["first"], "deep": f["deep"], "k": f["k"], "ke": f["ke"], "beta": f["beta"]})
        out.append({"halves": halves, "fits_excluding": fits})
    return out


def heldout(runs, share, attempts=None):
    """Held-out halves (robustness): on 110 clones of one half's 10 games, the Gittins index with constants fitted
    without that half (hysteresis 1 / 4 / 8, to be tuned on the other half) against the all-data constants and
    fixed-2x, at 390 tok/s x 6 slots."""
    p = PROFILES["390/6 (MTP-0 64K, measured)"]
    fits = heldout_fits(attempts)
    Path(HERE, "heldout-fits.json").write_text(json.dumps(fits, indent=1))
    configs = {}
    for s, split in enumerate(fits):
        pol = {}
        for r, half in enumerate(split["halves"]):
            roster = [half[i % 10] for i in range(110)]
            skw = cal_sim(p, share, roster_games=roster)
            pol[f"R{r} fixed-2x"] = [*fixed2x(p), skw]
            pol[f"R{r} gittins all-data h4"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
            for c, f in enumerate(split["fits_excluding"]):
                for h in (1.0, 4.0, 8.0):
                    kw = {"custom_shape": {"first": f["first"], "deep": f["deep"]}, "k": f["k"], "ke": f["ke"],
                          "beta": f["beta"], "hysteresis": h}
                    pol[f"R{r} gittins fit-excl-H{c} h{h:g}"] = [*gittins(p, **kw), {**skw, "opening_max_wall": 1200}]
        configs[f"heldout-s{s}"] = {**CAL, "runs": runs, "seed": 13 + s, "halves": split["halves"], "policies": pol}
    return configs


def ninehour(runs, share):
    """The 9-hour hidden run: 110 distinct games. ``distinct``: every c110 slot draws its own game factor from its
    source's posterior (shared false); ``population``: every game drawn from the fitted population."""
    configs = {}
    m3, m0 = PROFILES["260/2 (MTP-3, measured)"], PROFILES["390/6 (MTP-0 64K, measured)"]
    for mode in ("distinct", "population"):
        pol = {}
        for tag, p in (("MTP-3 2 slots", m3), ("MTP-0 6 slots", m0)):
            skw = cal_sim(p, share, wall=32400.0, shared=False, population=mode == "population")
            pol[f"fixed-2x {tag}"] = [*fixed2x(p), skw]
            if tag == "MTP-0 6 slots":
                pol[f"fixed-2x {tag} sw6"] = [*fixed2x(p), {**skw, "switch_games": 6}]
            pol[f"gittins-c110 {tag}"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
            pol[f"gittins within h0.15 {tag}"] = [*gittins(p, hysteresis=0.15, prior_scale="fixed"),
                                                  {**skw, "opening_max_wall": 1200}]
            pol[f"known-rate {tag}"] = ["known-rate", {}, skw]
        configs[f"ninehour-{mode}"] = {**CAL, "runs": runs, "seed": 6, "policies": pol}
    return configs


# MTP-0 with the dense-kernel overlay A+C (docs/validation-20260927/kernels-phase2.md, 28 September): decode per stream
# 56 -> 66 tok/s at 6 streams (+18%), 81.6 -> 103.2 single stream (+26%). Phase 2 scales the measured 6 x 64K arena rate
# (323, 54 per stream) by 66/55; the opening (~18 running) by a smaller +15%. Not yet measured in an arena hour.
KERNELS = {"open": 486.0 * 1.15, "p2": 323.0 * 66.0 / 55.0, "slots": 6, "busy_open": 32.4 * 1.15,
           "busy_p2": 54.5 * 66.0 / 55.0, "stream": 66.0}


def kernels(runs, share):
    """The dense-kernel MTP-0 profile: one hour on c110 (with the measured MTP-0 profile on the same seeds) and
    9 hours on 110 distinct games (seed 6, paired with ``ninehour-distinct``)."""
    m0 = PROFILES["390/6 (MTP-0 64K, measured)"]
    pol = {}
    for tag, p in (("MTP-0", m0), ("MTP-0 kernels", KERNELS)):
        for eff in EFFS:
            skw = cal_sim(p, share, efficiency=eff)
            pol[f"fixed-2x {tag} x{eff:g}"] = [*fixed2x(p), skw]
            pol[f"gittins-c110 {tag} x{eff:g}"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
            pol[f"known-rate {tag} x{eff:g}"] = ["known-rate", {}, skw]
    configs = {"kernels-1h": {**CAL, "runs": runs or 40, "seed": 41, "policies": pol}}
    skw = cal_sim(KERNELS, share, wall=32400.0, shared=False)
    configs["kernels-9h"] = {**CAL, "runs": 16, "seed": 6, "policies": {
        "fixed-2x MTP-0 kernels 6 slots": [*fixed2x(KERNELS), skw],
        "gittins-c110 MTP-0 kernels 6 slots": [*gittins(KERNELS), {**skw, "opening_max_wall": 1200}],
        "gittins within h0.15 MTP-0 kernels 6 slots": [*gittins(KERNELS, hysteresis=0.15, prior_scale="fixed"),
                                                       {**skw, "opening_max_wall": 1200}]}}
    return configs


def openrobust(runs, share):
    """Robustness of the selective opening (only the first 20 games of a shuffled roster open): game-level bootstrap
    (20 rosters x 6 runs, efficiency 1) and clear rates x0.5 / 0.75 / 1.33, for gittins-c110 and fixed-2x."""
    b = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": BOOT_PER}
    configs = {}
    for i, pname in enumerate(("390/6 (MTP-0 64K, measured)", "550/12 (MTP-0 32K)")):
        p = PROFILES[pname]
        pol = {}
        for tag, extra in (("boot", {"bootstrap": b}), ("x0.5", {"efficiency": 0.5}), ("x0.75", {"efficiency": 0.75}),
                           ("x1.33", {"efficiency": 1.33})):
            skw = cal_sim(p, share, shuffle=True, **extra)
            pol[f"fixed-2x {tag}"] = [*fixed2x(p), skw]
            pol[f"fixed-2x opening 20 {tag}"] = [*fixed2x(p), {**skw, "opening_games": 20}]
            pol[f"gittins-c110 {tag}"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
            pol[f"gittins opening 20 {tag}"] = [*gittins(p), {**skw, "opening_max_wall": 1200, "opening_games": 20}]
        configs[f"openrobust-p{i + 1}"] = {**CAL, "runs": BOOT_ROSTERS * BOOT_PER, "seed": 31, "profile": pname,
                                           "policies": pol}
    return configs


def ninehour_robust(runs, share):
    """The 9-hour comparison under a game-level bootstrap of the 23 recorded games (110 distinct games per roster:
    23 resampled games, every roster slot its own draw) and under clear rates x0.5 / x1.33."""
    m3, m0 = PROFILES["260/2 (MTP-3, measured)"], PROFILES["390/6 (MTP-0 64K, measured)"]
    configs = {}
    b = {"pool": sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}
    for tag, extra, n in (("boot23", {"bootstrap": b}, 24), ("x0.5", {"efficiency": 0.5}, runs),
                          ("x1.33", {"efficiency": 1.33}, runs)):
        pol = {}
        for ptag, p in (("MTP-3 2 slots", m3), ("MTP-0 6 slots", m0)):
            skw = cal_sim(p, share, wall=32400.0, shared=False, **extra)
            pol[f"fixed-2x {ptag}"] = [*fixed2x(p), skw]
            pol[f"gittins-c110 {ptag}"] = [*gittins(p), {**skw, "opening_max_wall": 1200}]
        roster = "23" if tag == "boot23" else "c110"
        configs[f"ninehour-{tag}"] = {**CAL, "roster": roster, "runs": n, "seed": 21, "policies": pol}
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", required=True)
    parser.add_argument("stage")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out-dir", default=str(HERE))
    args = parser.parse_args()
    share = opening_share()
    stages = {"calibrate": calibrate, "path": path, "slots": slots, "opening": opening, "legal": legal, "legal2": legal2,
              "ninehour": ninehour, "boot": boot, "sens": sens, "ninehour_robust": ninehour_robust, "openrobust": openrobust, "kernels": kernels,
              "heldout": lambda r, s: heldout(r, s, args.attempts)}
    configs = stages[args.stage](args.runs, share)
    for name, cfg in configs.items():
        if args.only and name not in args.only:
            continue
        run_config(name, cfg, args.attempts, args.out_dir, args.jobs)


if __name__ == "__main__":
    main()
