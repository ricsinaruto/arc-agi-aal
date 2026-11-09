"""Compare the run-B + parallel_samples + action_plans run with run B (30 September; parallel-samples-plans.md).

    uv run python scripts/arena_analysis/samples_plans_compare.py [--run RUN_ID] [--json P]

Score, levels and games with levels at 30 / 60 / 90 / 120 minutes (rebuilt from each clone's clears with the
scorecard's rule, abc_load), the deep games' levels and clear times, wall time and gameplay tokens per level index,
action efficiency (baseline / actions on the cleared levels and the points lost to over-baseline levels), the engine
(tok/s, preemptions) and, for the feature run, the parallel-sample and plan totals, the observed groups' spread and a
reconciliation of the engine's generated tokens against the completed requests' usage (the aborted samples' exact
total).
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_load import Run, level_score  # noqa: E402

RUN = "arena-hero-v2-samples-plans-v1-modal-20260929"
MARKS = (1800, 3600, 5400, 7200)


def marks(run):
    rows = {}
    for t in MARKS:
        games = [g for g in run.games.values() if g.levels_at(t)]
        rows[t] = {"score": round(run.score_at(t), 3), "levels": sum(g.levels_at(t) for g in run.games.values()),
                   "games_with_levels": len(games)}
    return rows


def per_level(run):
    """Wall seconds and gameplay tokens from the previous clear (or the start) to each clear, by level index."""
    out = defaultdict(lambda: {"seconds": [], "tokens": []})
    for g in run.games.values():
        prev_wall, prev_tokens = 0.0, 0
        for clear in g.clears:
            out[clear["level"]]["seconds"].append(clear["wall"] - prev_wall)
            out[clear["level"]]["tokens"].append(clear["tokens"] - prev_tokens)
            prev_wall, prev_tokens = clear["wall"], clear["tokens"]
    table = {}
    for level in sorted(out):
        s, k = out[level]["seconds"], out[level]["tokens"]
        table[level] = {"clears": len(s), "median_seconds": round(statistics.median(s)),
                        "median_tokens": round(statistics.median(k)), "mean_tokens": round(statistics.mean(k))}
    return table


def efficiency(run):
    ratios, lost = [], 0.0
    for g in run.games.values():
        for k in range(min(len(g.clears), len(g.level_actions))):
            base, acts = g.baselines[k], g.level_actions[k]
            if acts:
                ratios.append(base / acts)
                # A level's weight in the game score is (k + 1) / (n (n + 1) / 2); points lost vs 100% on it.
                weight = (k + 1) / (g.n_levels * (g.n_levels + 1) / 2)
                lost += max(0.0, 100.0 - level_score(base, acts)) * weight / 110.0
    return {"cleared_levels": len(ratios), "median_baseline_over_actions": round(statistics.median(ratios), 3),
            "mean_baseline_over_actions": round(statistics.mean(ratios), 3),
            "over_baseline_levels": sum(1 for r in ratios if r < 1), "points_lost_over_baseline": round(lost, 3)}


def deep(run, min_levels=3):
    rows = []
    for g in sorted(run.games.values(), key=lambda g: (-len(g.clears), g.id)):
        if len(g.clears) >= min_levels:
            rows.append({"game": g.id, "family": g.family, "levels": len(g.clears),
                         "clear_walls": [round(c["wall"]) for c in g.clears]})
    return rows


def replies(run):
    gp = [r for g in run.games.values() for r in g.replies if r["purpose"] == "gameplay"]
    deep_gp = [r for r in gp if r["level"] >= 2]
    return {"gameplay_replies": len(gp), "actions_per_reply": round(sum(r.get("actions", 0) for r in gp) / len(gp), 3),
            "zero_action_share": round(sum(1 for r in gp if not r.get("actions")) / len(gp), 3),
            "tokens_per_action": round(sum(r["completion"] for r in gp) / max(1, sum(r.get("actions", 0) for r in gp)),
                                       1),
            "deep_reply_tokens_mean": round(statistics.mean(r["completion"] for r in deep_gp)) if deep_gp else None,
            "deep_reply_tokens_median": round(statistics.median(r["completion"] for r in deep_gp)) if deep_gp else None,
            "deep_reply_seconds_mean": round(statistics.mean(float(r["seconds"]) for r in deep_gp), 1)
            if deep_gp else None}


def engine(run):
    e = run.arena.get("engine") or {}
    return {k: e.get(k) for k in ("accepted_generation_tokens", "accepted_generation_tokens_per_second",
                                  "preemptions", "cached_prompt_tokens", "computed_prompt_tokens",
                                  "peak_cache_fraction", "peak_running_requests", "peak_waiting_requests")}


def observed(run):
    """Observed groups (every 8th of a game): the within-group spread of sample durations."""
    rows = []
    for gid in run.games:
        path = run.dir / "games" / gid / "log.jsonl"
        for line in path.open():
            rec = json.loads(line)
            if rec.get("kind") == "parallel_observed":
                rows.append(rec)
    clean = [r for r in rows if not r["censored"] and len(r["durations"]) >= 2]
    spreads = [statistics.pstdev(math.log(max(1, t)) for t in r["tokens"]) for r in clean if all(r["tokens"])]
    return {"observed": len(rows), "uncensored": len(clean),
            "saved_vs_median_mean": round(statistics.mean(r["saved_vs_median_seconds"] for r in clean), 2)
            if clean else None,
            "saved_vs_slowest_mean": round(statistics.mean(r["saved_vs_slowest_seconds"] for r in clean), 2)
            if clean else None,
            "within_group_log_token_sd_mean": round(statistics.mean(spreads), 3) if spreads else None}


def level_spread(run, min_levels=2):
    """The within-(game, level) spread of log reply tokens: the estimate's independent-sibling reference."""
    groups = defaultdict(list)
    for g in run.games.values():
        for r in g.replies:
            if r["purpose"] == "gameplay" and r["level"] >= min_levels and r["completion"] > 0:
                groups[(g.id, r["level"])].append(math.log(r["completion"]))
    sds = [statistics.pstdev(v) for v in groups.values() if len(v) >= 4]
    return round(statistics.mean(sds), 3) if sds else None


def reconcile(run):
    """The engine's generated tokens against every completed request's usage: the difference is what aborted requests
    decoded (siblings and primaries cut by a winner, yields, the wall end)."""
    requests = json.loads((run.dir / "requests.json").read_text())
    completed = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in requests)
    by_role = defaultdict(int)
    for r in requests:
        by_role[r.get("parallel_role") or "plain"] += (r.get("usage") or {}).get("completion_tokens") or 0
    generated = (run.arena.get("engine") or {}).get("accepted_generation_tokens")
    return {"engine_generated": generated, "completed_usage": completed,
            "aborted_exact": None if generated is None else generated - completed, "completed_by_role": dict(by_role),
            "requests_by_role": {k: sum(1 for r in requests if (r.get("parallel_role") or "plain") == k)
                                 for k in {r.get("parallel_role") or "plain" for r in requests}}}


def summarize(run, feature=False):
    out = {"run": run.run_id, "marks": marks(run), "per_level": per_level(run), "efficiency": efficiency(run),
           "deep_games": deep(run), "replies": replies(run), "engine": engine(run),
           "level_log_token_sd": level_spread(run)}
    if feature:
        out["parallel_samples"] = run.arena.get("parallel_samples")
        out["action_plans"] = run.arena.get("action_plans")
        out["observed"] = observed(run)
        out["reconcile"] = reconcile(run)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--json")
    args = ap.parse_args()
    result = {"runB": summarize(Run("runB")), "feature": summarize(Run("feature", run_id=args.run), feature=True)}
    text = json.dumps(result, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
