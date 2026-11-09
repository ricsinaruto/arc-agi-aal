"""The reportable numbers of docs/rsi.md 4 at a registered point (every ``heldout_ab_every`` loop GPU-h): the best
candidate against frozen gen 0 on held-out A and B, three seeds each, compared paired by level with errors clustered
by game.

Per set and candidate: RHAE (A only: B has no baselines yet) and levels cleared, each the mean over seeds per game,
then over games. Paired: the per-game difference (best - gen 0) of the seed means; its mean over the games both played
and a standard error clustered by game (sd of the per-game differences / sqrt(games): seeds and levels of one game
are not independent). By level: per (game, level) the share of seeds that cleared it; the per-game sum of the
differences, clustered the same way. 95% intervals use the t quantile for games - 1 degrees of freedom.

    python -m rsi_loop.report <checkpoint folder>      # every point's report again, from heldout/results.jsonl
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       12: 2.179, 15: 2.131, 20: 2.086, 24: 2.064, 30: 2.042, 40: 2.021, 49: 2.010, 60: 2.000}


def t95(df):
    keys = sorted(T95)
    return T95[max(k for k in keys if k <= df)] if df < 60 else 1.96


def per_game(rows, metric):
    """{game: mean over seeds of ``metric``} (rows of one run and set; infrastructure failures left out)."""
    games = {}
    for r in rows:
        if r.get("infrastructure") or r.get(metric) is None:
            continue
        games.setdefault(r["game"], []).append(float(r[metric]))
    return {g: sum(v) / len(v) for g, v in games.items()}


def level_rates(rows, totals):
    """{game: [share of seeds that cleared level 1, 2, ...]}."""
    out = {}
    for game in {r["game"] for r in rows}:
        mine = [r for r in rows if r["game"] == game and not r.get("infrastructure")]
        total = int(totals.get(game) or max([int(r.get("levels") or 0) for r in mine] or [0]))
        if mine:
            out[game] = [sum(1 for r in mine if int(r.get("levels") or 0) >= level) / len(mine)
                         for level in range(1, total + 1)]
    return out


def paired(a, b):
    """Mean of b - a over common keys, clustered standard error, 95% interval."""
    common = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in common]
    if not diffs:
        return {"games": 0}
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:  # no spread to cluster on: JSON null, never NaN
        return {"games": len(diffs), "diff": round(mean, 3), "se": None, "ci95": None}
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    half = t95(len(diffs) - 1) * se
    return {"games": len(diffs), "diff": round(mean, 3), "se": round(se, 3), "ci95": [round(mean - half, 3),
                                                                                       round(mean + half, 3)]}


def compare(base_rows, best_rows, totals, scored):
    out = {}
    for name, rows in (("gen0", base_rows), ("best", best_rows)):
        levels = per_game(rows, "levels")
        out[name] = {"games": len(levels), "seeds": len({r.get("seed") for r in rows}),
                     "levels_mean": round(sum(levels.values()) / len(levels), 3) if levels else None}
        if scored:
            score = per_game(rows, "score")
            out[name]["rhae"] = round(sum(score.values()) / len(score), 3) if score else None
    out["paired_levels"] = paired(per_game(base_rows, "levels"), per_game(best_rows, "levels"))
    if scored:
        out["paired_rhae"] = paired(per_game(base_rows, "score"), per_game(best_rows, "score"))
    base_l, best_l = level_rates(base_rows, totals), level_rates(best_rows, totals)
    by_level = {g: sum(y - x for x, y in zip(base_l[g], best_l[g])) for g in set(base_l) & set(best_l)}
    out["paired_by_level"] = paired({g: 0.0 for g in by_level}, by_level)
    return out


def point_report(point, runs, results, totals):
    """``runs``: the held-out runs of this point ({"id", "candidate", "sets", "seeds"}); gen 0's three-seed run may
    come from an earlier point (it is frozen)."""
    base = next((r for r in runs if r["candidate"] == "c0000"), None)
    best = next((r for r in runs if r["candidate"] != "c0000"), None) or base
    report = {"point_loop_gpu_hours": point, "gen0_run": base and base["id"], "best_run": best and best["id"],
              "best_candidate": best and best["candidate"], "sets": {}}
    if base is None:
        return report
    for kind in ("heldout_a", "heldout_b"):
        rows_of = lambda run: [r for r in results if r.get("run") == run["id"] and r["kind"] == kind]  # noqa: E731
        report["sets"][kind] = compare(rows_of(base), rows_of(best), totals.get(kind, {}), kind == "heldout_a")
    return report


def main():
    ckpt = Path(sys.argv[1])
    state = json.loads((ckpt / "state.json").read_text())
    results = [json.loads(line) for line in (ckpt / "heldout/results.jsonl").read_text().splitlines() if line.strip()]
    for point in state["heldout"].get("points", []):
        runs = [r for r in state["heldout"]["runs"] if r["id"] in point["runs"]]
        print(json.dumps(point_report(point["loop_hours"], runs, results, point.get("totals", {})), indent=1))


if __name__ == "__main__":
    main()
