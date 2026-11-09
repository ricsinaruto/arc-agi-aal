"""Depthwise scheduling against the standard kernels-v2 policy in the calibrated simulator (28 September).

``fixed-2x-depthwise`` (``arena.depthwise``): no opening, ``slots`` slots from t = 0, level-0 games rotate on an
18K-token turn at a 32K window, a level-1 clear keeps its slot as a depth game with fixed-2x budgets, full regrant,
depth-first or fixed-2x refill order (``schedule_sim.DepthWise``). The standard policy is kernels-v2: fixed-2x after
the 110-game opening with the six-clear switch.

    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/depthwise_sim.py \
        --attempts <attempts-now.jsonl> <stage> [--runs N] [--modal]

Stages: ``validate`` (the two real depthwise runs and the two standard runs on c110), ``compare`` (one hour on c110,
three servings), ``nine`` (9 hours, 110 distinct games), ``decompose`` (opening x rules), ``robust`` (bootstrap and
clear rates x0.5 / x1.33), ``report``. Results go to ``docs/validation-20260927/depthwise-sim/``; ``--modal`` runs
the policy jobs on Modal CPUs (``modal_sim.py``) instead of the local process pool.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c110_path as cp  # noqa: E402
import schedule_sim as sim  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "validation-20260927" / "depthwise-sim"

# Serving, measured on the real c110 runs (c110_runs.py summaries; tokens per wall second over all games, busy clock per
# game). Depthwise has no opening: every slot runs at the phase-2 rate from t = 0.
SERVING = {
    # MTP-0 + A+C, 6 x 64K (c110-kernels-v2: opening 541.6 at 17.6 running, phase 2 387.2, busy 31.2 / 63.5)
    "K 6x64K": {"open": 541.6, "p2": 387.2, "slots": 6, "busy_open": 31.2, "busy_p2": 63.5, "stream": 64.5},
    # MTP-3 phase 4, 6 x 64K. Standard (c110-mtp3-wide-kernels-v1): opening 225.4 at ~11 running (drafting pins
    # state pages), phase 2 417.9, busy 20.6 / 64.3. Depthwise (the two depthwise-mtp3 runs): 497-501 tok/s over
    # the hour with no opening.
    "M3 6x64K": {"open": 225.4, "p2": 417.9, "slots": 6, "busy_open": 20.6, "busy_p2": 64.3, "stream": 69.7,
                 "depthwise_p2": 500.0},
    # MTP-0 + A+C with int8 KV, 14 x 64K (streams-9h.md §8: 688.8 tok/s phase 2, opening as kernels-v2)
    "KV8 14x64K": {"open": 541.6, "p2": 688.8, "slots": 14, "busy_open": 31.2, "busy_p2": 48.43, "stream": 49.2},
}
LEVEL0_MULTS = (1.0, 0.9, 0.75)  # the level-0 turns' 32K window, a multiplier on the first level's clear rate
DW = {"quantum": 18000.0, "multiple": 2.0, "regrant": "full"}


def standard(p, share, **extra):
    """kernels-v2: fixed-2x after the 110-game opening, six-clear switch."""
    return [*cp.fixed2x(p), {**cp.serving_sim(p, share, 0.35), "switch_games": 6, **extra}]


def depthwise(p, share, backlog, m0=1.0, **extra):
    rate = p.get("depthwise_p2", p["p2"])
    skw = {**cp.serving_sim({**p, "p2": rate}, share, 0.35), "opening_max_wall": 0.0, "first_efficiency": m0,
           **extra}
    return ["depthwise", {**DW, "backlog": backlog}, skw]


def fixed2x_no_opening(p, share, **extra):
    """fixed-2x rules with no opening (phase 2 from t = 0; every game's first grant is the 30 s floor)."""
    return [*cp.fixed2x(p), {**cp.serving_sim(p, share, 0.35), "opening_max_wall": 0.0, **extra}]


def depthwise_after_opening(p, share, backlog, **extra):
    """The depthwise rules from the six-clear switch of the standard 110-game opening (T1 weighs opening tokens at
    the fixed-2x busy factor)."""
    bf = p["busy_p2"] / p["busy_open"]
    return ["depthwise", {**DW, "backlog": backlog, "busy_factor": bf},
            {**cp.serving_sim(p, share, 0.35), "switch_games": 6, **extra}]


def cells(share, servings, m0s=LEVEL0_MULTS, **extra):
    pol = {}
    for name in servings:
        p = SERVING[name]
        pol[f"{name} | standard"] = standard(p, share, **extra)
        for m0 in m0s:
            tag = "" if m0 == 1.0 else f" l0x{m0:g}"
            pol[f"{name} | depthwise depth-first{tag}"] = depthwise(p, share, "depth-first", m0, **extra)
            pol[f"{name} | depthwise fixed-order{tag}"] = depthwise(p, share, "fixed-2x", m0, **extra)
    return pol


C110 = {**cp.CAL, "roster": "c110"}  # the real roster order (no shuffle), clones share their game's difficulty


def validate(runs, share):
    return {"dw-validate": {**C110, "runs": runs or 200, "seed": 51, "policies": cells(share, ["M3 6x64K", "K 6x64K"])}}


def deepdiag(runs, share):
    """Diagnostic: every deeper level's (L >= 2) clear rate x d, for the four real-run cells. If one d fits both the
    depthwise runs and the standard runs, the simulator's miss is a world (deep-level) miss common to both policies;
    if not, it is specific to the depthwise tenures."""
    pol = {}
    for d in (0.75, 0.6, 0.5):
        for name in ("M3 6x64K", "K 6x64K"):
            p = SERVING[name]
            s = standard(p, share)
            pol[f"{name} | standard deep{d:g}"] = [s[0], s[1], {**s[2], "efficiency": d, "first_efficiency": 1 / d}]
            if name == "M3 6x64K":
                for b, lab in (("depth-first", "depth-first"), ("fixed-2x", "fixed-order")):
                    s = depthwise(p, share, b, 1.0)
                    pol[f"{name} | depthwise {lab} deep{d:g}"] = [s[0], s[1], {**s[2], "efficiency": d,
                                                                               "first_efficiency": 1 / d}]
    return {"dw-deepdiag": {**C110, "runs": runs or 200, "seed": 51, "policies": pol}}


TAILS = {"t40x0.5": {"from": 40000, "scale": 0.5}, "t40x0.25": {"from": 40000, "scale": 0.25},
         "t20x0.5": {"from": 20000, "scale": 0.5}, "t80x0.25": {"from": 80000, "scale": 0.25}}


def taildiag(runs, share):
    """Diagnostic: the within-level hazard on long attempts (bands from 20K / 40K / 80K tokens on a level) scaled down,
    for the four real-run cells. Depthwise exercises the long-attempt tail (full regrants, 150-270K per depth game);
    the standard policy rarely does."""
    pol = {}
    for tag, tail in TAILS.items():
        for name in ("M3 6x64K", "K 6x64K"):
            p = SERVING[name]
            s = standard(p, share, tail=tail)
            pol[f"{name} | standard {tag}"] = s
            if name == "M3 6x64K":
                pol[f"{name} | depthwise depth-first {tag}"] = depthwise(p, share, "depth-first", 1.0, tail=tail)
                pol[f"{name} | depthwise fixed-order {tag}"] = depthwise(p, share, "fixed-2x", 1.0, tail=tail)
    return {"dw-taildiag": {**C110, "runs": runs or 200, "seed": 51, "policies": pol}}


def compare(runs, share):
    return {"dw-hour": {**C110, "runs": runs or 60, "seed": 52, "policies": cells(share, list(SERVING))}}


def nine(runs, share):
    return {"dw-nine": {**cp.CAL, "runs": runs or 16, "seed": 6, "policies": cells(
        share, list(SERVING), wall=32400.0, shared=False, shuffle=True)}}


def decompose(runs, share):
    """Opening (the 110-game six-clear opening, or none) x rules (fixed-2x, depthwise depth-first / fixed-order), at
    MTP-0 + A+C 6 x 64K and at KV8 14 x 64K; one hour and nine hours; first level at 64K everywhere (rules only)."""
    out = {}
    for tag, extra, n, seed in (("hour", {}, runs or 60, 53), ("nine", {"wall": 32400.0, "shared": False,
                                                                       "shuffle": True}, 16, 6)):
        pol = {}
        for name in ("K 6x64K", "KV8 14x64K"):
            p = SERVING[name]
            pol[f"{name} | opening + fixed-2x (standard)"] = standard(p, share, **extra)
            pol[f"{name} | no opening + fixed-2x"] = fixed2x_no_opening(p, share, **extra)
            for b, lab in (("depth-first", "depth-first"), ("fixed-2x", "fixed-order")):
                pol[f"{name} | opening + depthwise {lab}"] = depthwise_after_opening(p, share, b, **extra)
                pol[f"{name} | no opening + depthwise {lab}"] = depthwise(p, share, b, 1.0, **extra)
        base = {**C110} if tag == "hour" else {**cp.CAL}
        out[f"dw-decompose-{tag}"] = {**base, "runs": n, "seed": seed, "policies": pol}
    return out


def robust(runs, share):
    """Game bootstrap and clear rates x0.5 / x1.33 for standard vs depthwise (both orders; level-0 at 1.0 and 0.9),
    one hour (c110) and nine hours (distinct games), at K 6x64K and KV8 14x64K."""
    out = {}
    servings = ["K 6x64K", "KV8 14x64K"]
    b1 = {"pool": sim.C20, "k": 20, "n": 110, "per_roster": cp.BOOT_PER}
    b9 = {"pool": sim.GAMES23, "k": 23, "n": 110, "per_roster": 2}
    nine_kw = {"wall": 32400.0, "shared": False, "shuffle": True}
    out["dw-hour-boot"] = {**C110, "runs": cp.BOOT_ROSTERS * cp.BOOT_PER, "seed": 11,
                           "policies": cells(share, servings, (1.0, 0.9), bootstrap=b1)}
    out["dw-nine-boot23"] = {**cp.CAL, "roster": "23", "runs": 24, "seed": 21,
                             "policies": cells(share, servings, (1.0, 0.9), bootstrap=b9, **nine_kw)}
    for s in (0.5, 1.33):
        out[f"dw-hour-x{s:g}"] = {**C110, "runs": 40, "seed": 12,
                                  "policies": scale(cells(share, servings, (1.0, 0.9)), s)}
        out[f"dw-nine-x{s:g}"] = {**cp.CAL, "runs": 16, "seed": 22,
                                  "policies": scale(cells(share, servings, (1.0, 0.9), **nine_kw), s)}
    return out


def scale(pol, s):
    return {k: [v[0], v[1], {**v[2], "efficiency": v[2].get("efficiency", 1.0) * s}] for k, v in pol.items()}


STAGES = {"validate": validate, "deepdiag": deepdiag, "taildiag": taildiag, "compare": compare, "nine": nine, "decompose": decompose, "robust": robust}


def summarise(res):
    out = cp.summarise(res)
    for k in ("played", "cleared1"):
        if k in res:
            v = np.array(res[k])
            out[k] = {"mean": round(float(v.mean()), 1), "p10": float(np.percentile(v, 10)),
                      "p90": float(np.percentile(v, 90))}
    if "stats" in res:
        out["stats"] = res["stats"]
    return {**out, "scores": res["scores"], "played_runs": res.get("played"), "cleared1_runs": res.get("cleared1")}


def run_local(name, config, attempts, out_dir, jobs):
    res = sim.sweep(config, attempts, jobs)
    write(name, config, res, out_dir)


def write(name, config, res, out_dir):
    out = {label: summarise(r) for label, r in res.items()}
    Path(out_dir, f"{name}.config.json").write_text(json.dumps(config, indent=1))
    Path(out_dir, f"{name}.result.json").write_text(json.dumps(out, indent=1))
    for label, r in out.items():
        extra = ""
        if "played" in r:
            extra = f" played {r['played']['mean']:5.1f} L1 {r['cleared1']['mean']:4.1f}"
        print(f"  {label:58s} {r['mean']:5.2f} ± {r['se']:.2f} p10/50/90 {r['p10']:.2f}/{r['p50']:.2f}/{r['p90']:.2f}"
              f" levels {r['levels']:5.1f}{extra}", flush=True)


# ---------------------------------------------------------------------------------------------- report

REAL = {  # c110 runs (c110-results.md; played / level-1 clears / levels from the run logs)
    "M3 6x64K | depthwise depth-first": {"run": "c110-depthwise-mtp3-v1", "score": 0.65, "played": 31, "cleared1": 6,
                                         "levels": 14, "level0_share": 0.331},
    "M3 6x64K | depthwise fixed-order": {"run": "c110-depthwise-mtp3-fixed-order-v1", "score": 0.86, "played": 61,
                                         "cleared1": 11, "levels": 17, "level0_share": 0.627},
    "M3 6x64K | standard": {"run": "c110-mtp3-wide-kernels-v1", "score": 0.29, "played": 110, "cleared1": None,
                            "levels": 8},
    "K 6x64K | standard": {"run": "c110-kernels-v2", "score": 1.42, "played": 110, "cleared1": 9, "levels": 21},
}


def pct(values, x):
    v = np.array(values)
    return round(float(100 * (np.mean(v < x) + 0.5 * np.mean(v == x))), 0)


def paired(r, a, b, per=None):
    x = np.array(r[a]["scores"]) - np.array(r[b]["scores"])
    out = {"diff": round(float(x.mean()), 2), "se": round(float(x.std() / np.sqrt(len(x))), 2),
           "p_run": round(float(np.mean(x > 0)), 2)}
    if per:
        out["p_roster"] = round(float(np.mean(x.reshape(-1, per).mean(axis=1) > 0)), 2)
    return out


def report(d, world="w1"):
    suffix = "" if world == "w1" else f"-{world}"

    def load(name):
        path = Path(d, f"{name}{suffix}.result.json")
        return json.loads(path.read_text()) if path.exists() else None

    summary = {}
    r = load("dw-validate")
    if r:
        print("### validation (c110, one hour)\n")
        print("| Real run | Real score | Sim mean (p10-p90) | Real's percentile | Played real / sim (p10-p90) | "
              "Level-1 clears real / sim (p10-p90) | Levels real / sim | Level-0 token share real / sim |")
        print("| --- | ---: | --- | ---: | --- | --- | --- | --- |")
        for label, real in REAL.items():
            for tag in ("", " l0x0.9", " l0x0.75"):
                key = label + tag if "depthwise" in label else label
                if key not in r or (tag and "depthwise" not in label):
                    continue
                s = r[key]
                played = s.get("played", {})
                l1 = s.get("cleared1", {})
                share = s.get("stats", {}).get("level0_share")
                print(f"| {real['run']} ({key}) | {real['score']:.2f} | {s['mean']:.2f} ({s['p10']:.2f}-{s['p90']:.2f}) | "
                      f"{pct(s['scores'], real['score']):.0f} | {real['played']} / {played.get('mean')} "
                      f"({played.get('p10')}-{played.get('p90')}) | {real['cleared1']} / {l1.get('mean')} "
                      f"({l1.get('p10')}-{l1.get('p90')}) | {real['levels']} / {s['levels']} | "
                      f"{real.get('level0_share', '-')} / {'-' if share is None else round(share, 3)} |")
                summary[f"validate | {key}"] = {"real": real, "sim_mean": s["mean"], "pct": pct(s["scores"], real["score"])}
        print()
    for name, title in (("dw-hour", "one hour, c110"), ("dw-nine", "nine hours, 110 distinct games")):
        r = load(name)
        if not r:
            continue
        print(f"### {title}\n")
        print("| Serving | Policy | Mean (p10-p90) | − standard (paired) | Played | Level-1 clears | Level-0 share |")
        print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for label, s in r.items():
            serving, pol = label.split(" | ")
            ref = f"{serving} | standard"
            dd = "-" if label == ref else (lambda x: f"{x['diff']:+.2f} ± {x['se']:.2f} ({int(100 * x['p_run'])}%)")(
                paired(r, label, ref))
            share = s.get("stats", {}).get("level0_share")
            print(f"| {serving} | {pol} | {s['mean']:.2f} ({s['p10']:.2f}-{s['p90']:.2f}) | {dd} | "
                  f"{s.get('played', {}).get('mean')} | {s.get('cleared1', {}).get('mean')} | "
                  f"{'-' if share is None else round(share, 2)} |")
            summary[f"{name} | {label}"] = {"mean": s["mean"], "p10": s["p10"], "p90": s["p90"]}
        print()
    for name in ("dw-decompose-hour", "dw-decompose-nine"):
        r = load(name)
        if not r:
            continue
        print(f"### {name}\n\n| Serving | Opening + rules | Mean (p10-p90) | − standard |\n| --- | --- | --- | ---: |")
        for label, s in r.items():
            serving, pol = label.split(" | ")
            ref = f"{serving} | opening + fixed-2x (standard)"
            dd = "-" if label == ref else f"{paired(r, label, ref)['diff']:+.2f} ± {paired(r, label, ref)['se']:.2f}"
            print(f"| {serving} | {pol} | {s['mean']:.2f} ({s['p10']:.2f}-{s['p90']:.2f}) | {dd} |")
            summary[f"{name} | {label}"] = {"mean": s["mean"]}
        print()
    checks = [("1 h c110", "dw-hour", None), ("1 h bootstrap", "dw-hour-boot", cp.BOOT_PER),
              ("1 h x0.5", "dw-hour-x0.5", None), ("1 h x1.33", "dw-hour-x1.33", None),
              ("9 h distinct", "dw-nine", None), ("9 h bootstrap 23", "dw-nine-boot23", 2),
              ("9 h x0.5", "dw-nine-x0.5", None), ("9 h x1.33", "dw-nine-x1.33", None)]
    print("### robustness: depthwise − standard (paired; share of rosters or runs)\n")
    pols = ["depthwise depth-first", "depthwise fixed-order", "depthwise depth-first l0x0.9",
            "depthwise fixed-order l0x0.9"]
    print("| Check | Serving | standard | " + " | ".join(pols) + " |\n| --- | --- | ---: | " +
          " | ".join("---:" for _ in pols) + " |")
    for title, name, per in checks:
        r = load(name)
        if not r:
            continue
        for serving in ("K 6x64K", "KV8 14x64K"):
            ref = f"{serving} | standard"
            if ref not in r:
                continue
            row = []
            for pol in pols:
                key = f"{serving} | {pol}"
                if key not in r:
                    row.append("")
                    continue
                x = paired(r, key, ref, per)
                row.append(f"{x['diff']:+.2f} ({int(100 * x.get('p_roster', x['p_run']))}%)")
                summary[f"robust | {name} | {key}"] = x
            print(f"| {title} | {serving} | {r[ref]['mean']:.2f} | " + " | ".join(row) + " |")
    Path(d, f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))


NEW_ATTEMPTS = str(OUT / "c110-attempts-new.jsonl")


def with_world(configs, world):
    """``w1``: the calibrated world of c110-path-to-3pct.md (98 earlier runs + the first six c110 runs). ``w2``: plus
    the five c110 runs recorded since (kernels-v2, deep-explore, mtp3-wide-kernels, both depthwise-mtp3 runs;
    ``c110_runs_new.py``), so the depthwise runs' long depth attempts enter the fit (in-sample for them)."""
    if world == "w1":
        return configs
    return {f"{name}-{world}": {**cfg, "extra_attempts": [cp.C110_ATTEMPTS, NEW_ATTEMPTS]}
            for name, cfg in configs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--world", default="w1", choices=("w1", "w2"))
    parser.add_argument("stage")
    parser.add_argument("--runs", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        report(args.out_dir, args.world)
        return
    share = cp.opening_share()
    for name, cfg in with_world(STAGES[args.stage](args.runs, share), args.world).items():
        if args.only and name not in args.only:
            continue
        print(f"== {name}: {len(cfg['policies'])} policies x {cfg['runs']} runs", flush=True)
        run_local(name, cfg, args.attempts, args.out_dir, args.jobs)


if __name__ == "__main__":
    main()
