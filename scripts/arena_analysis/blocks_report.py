"""Summary of a hero-v2-blocks trial (CPU; docs/validation-20260927/hero-v2-blocks.md).

    uv run python scripts/arena_analysis/blocks_report.py <trial dir> [--out result.json]

Score at the 30-minute marks (the live score reads), levels, the scheduler's block events (admissions, parks, fillers,
retirements, old refills), per-game gameplay tokens and levels by role, compactions and the compaction events of the
harness (compaction_split, compaction_guard, compaction_rejected -> mechanical summary), generation throughput, the
prefix-cache share of prompt tokens, gate waits from the progress rows, and the scheduler audit.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

EVENTS = ("compaction", "compaction_split", "compaction_split_skipped", "compaction_guard", "compaction_rejected",
          "compaction_empty", "compaction_tail_cut", "cap_hit_followup", "length_truncation", "stuck_notice")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_dir")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    trial = Path(args.trial_dir)
    report = json.loads((trial / "arena.json").read_text())
    rows = [json.loads(line) for line in (trial / "progress.jsonl").read_text().splitlines() if line.strip()]
    marks = {}
    for mark in (900, 1800, 2700, 3600, 4500, 5400, 6300, 7200):
        cands = [r for r in rows if ((r.get("live") or {}).get("score") or {}).get("at_elapsed_seconds", 0) >= mark - 30]
        if cands:
            s = cands[0]["live"]["score"]
            marks[mark] = {"score": round(s["score"], 3), "at": s["at_elapsed_seconds"], "levels": s.get("levels"),
                           "games_with_levels": cands[0]["live"].get("games_with_levels")}
    gate = [r["gate"] for r in rows]
    waiting = {"rows": len(gate), "rows_waiting": sum(g["waiting"] > 0 for g in gate),
               "mean_waiting": round(statistics.mean(g["waiting"] for g in gate), 2),
               "max_waiting": max(g["waiting"] for g in gate),
               "mean_inflight": round(statistics.mean(g["inflight"] for g in gate), 2)}
    by_hour = {}
    for lo in range(0, 7200, 900):
        seg = [r for r in rows if lo <= r["elapsed_seconds"] < lo + 900]
        if seg:
            by_hour[lo] = {"inflight": round(statistics.mean(r["gate"]["inflight"] for r in seg), 1),
                           "waiting": round(statistics.mean(r["gate"]["waiting"] for r in seg), 2),
                           "reserved": round(statistics.mean(r["gate"]["reserved"] for r in seg)),
                           "tok_s_recent": round(statistics.mean(
                               ((r.get("live") or {}).get("engine") or {}).get("tokens_per_second_recent") or 0
                               for r in seg), 1)}
    blocks = report["scheduler"]["blocks"]
    games = report["games"]
    per_game = []
    events = collections.Counter()
    event_games = collections.defaultdict(set)
    for g in games:
        info = blocks["games"][g["game_id"]]
        log = trial / "games" / g["game_id"] / "log.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                if '"kind":"compaction' in line or any(f'"kind":"{k}"' in line for k in EVENTS):
                    try:
                        kind = json.loads(line).get("kind")
                    except json.JSONDecodeError:
                        continue
                    if kind in EVENTS:
                        events[kind] += 1
                        event_games[kind].add(g["game_id"])
        fixes = g.get("harness_fixes") or {}
        per_game.append({"game": g["game_id"], "home_block": info["home_block"], "role": info["role"],
                         "state": info["state"], "retired": info["retired"], "levels": g["levels"],
                         "tokens": info["tokens"], "admissions": info["admissions"],
                         "first_admit_wall": info["first_admit_wall"], "compactions": g.get("compactions"),
                         "max_prompt_tokens": g.get("max_prompt_tokens"),
                         "summary_guards": fixes.get("summary_guards"), "summary_splits": fixes.get("summary_splits")})
    started = [p for p in per_game if p["first_admit_wall"] is not None]
    engine = report["engine"]
    cached, computed = engine.get("cached_prompt_tokens") or 0, engine.get("computed_prompt_tokens") or 0
    result = {
        "end_reason": report.get("end_reason"), "wall": report.get("wall"), "status": report.get("status"),
        "score": report.get("mean_score"), "levels": report.get("levels"),
        "games_with_levels": sum(p["levels"] > 0 for p in per_game), "marks": marks,
        "scheduler": {"counts": blocks["counts"], "block": blocks["block"], "games_started": blocks["games_started"],
                      "idle_slot_seconds": blocks["idle_slot_seconds"],
                      "idle_slot_seconds_with_eligible": blocks["idle_slot_seconds_with_eligible"],
                      "decisions": [d for d in blocks["decisions"] if d["decision"] not in ("block_start",)]},
        "audit": report.get("scheduler_audit"),
        "engine": {k: engine.get(k) for k in ("accepted_generation_tokens_per_second", "preemptions",
                                              "peak_cache_fraction", "peak_running_requests", "peak_waiting_requests",
                                              "errors")},
        "prefix_cache_share": round(cached / (cached + computed), 4) if cached + computed else None,
        "gate_waiting": waiting, "by_quarter_hour": by_hour,
        "compaction_events": dict(events), "compaction_event_games": {k: sorted(v) for k, v in event_games.items()},
        "compactions": sum(p["compactions"] or 0 for p in per_game),
        "compaction_games": sum((p["compactions"] or 0) > 0 for p in per_game),
        "largest_prompt": max(p["max_prompt_tokens"] or 0 for p in per_game),
        "tokens": {"started_games": len(started), "mean": round(statistics.mean(p["tokens"] for p in started)),
                   "min": min(p["tokens"] for p in started), "max": max(p["tokens"] for p in started)},
        "levels_by_role": {r: sum(p["levels"] for p in per_game if p["role"] == r) for r in ("original", "filler", "old")},
        "per_game": sorted(per_game, key=lambda p: (-p["levels"], p["game"])),
    }
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_game",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
