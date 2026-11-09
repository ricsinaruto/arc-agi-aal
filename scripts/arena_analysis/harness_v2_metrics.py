"""Harness v2 co-metrics of one or more arena runs, from the per-game ``harness_v2`` blocks of arena.json.

Each fixed-2x-harness-v2 game row carries the counters of ``arena.harness_v2.HarnessV2`` (whether each
edit fired and what it changed). This sums them per run, split into DEV and HOLDOUT games (the holdout
split is reported, never used for tuning), and prints the derived shares:

    python scripts/arena_analysis/harness_v2_metrics.py <run>/arena/trials/duck-transfer/arena.json [...]

For the per-level-attempt view (no-action shares, post-compaction requests, HUD-band change shares), run
``failure_signatures.py --glob <run>/arena/trials/duck-transfer --report``; it reads the same logs and
counts the harness-v2 events (``v2_*`` fields).
"""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from posttrain.split import DEV, HOLDOUT  # noqa: E402

LISTS = {"edits", "clocks"}


def summarize(rows):
    total = collections.Counter()
    for row in rows:
        for key, value in (row.get("harness_v2") or {}).items():
            if key not in LISTS and isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] += value
    acts = max(1, total["actions"])
    derived = {
        "games": len(rows),
        "clock_only_share": round(total["clock_only"] / acts, 3),
        "raw_no_change_share": round(total["raw_no_change"] / acts, 3),
        "masked_no_change_share": round(total["masked_no_change"] / acts, 3),
        "nudge_follow_rate": round(total["nudges_followed"] / max(1, total["nudges"]), 3),
        "compaction_noaction_requests_per_compaction": round(
            total["compaction_noaction_requests"] / max(1, total["compactions_closed"]), 2),
        "compaction_noaction_tokens_per_compaction": round(
            total["compaction_noaction_tokens"] / max(1, total["compactions_closed"])),
        "action7_undo_share": round(total["action7_undo"] / max(1, total["action7"]), 3),
        "first_clicks_on_new_share": round(total["first_clicks_on_new"] / max(1, total["first_clicks"]), 3),
    }
    total.pop("clock_only_share", None)
    return {**derived, **dict(sorted(total.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arena_json", nargs="+")
    args = parser.parse_args()
    for path in args.arena_json:
        report = json.loads(Path(path).read_text())
        rows = [row for row in report.get("games", []) if row.get("harness_v2")]
        split = {"dev": [r for r in rows if str(r.get("game_id", ""))[:4] in DEV],
                 "holdout": [r for r in rows if str(r.get("game_id", ""))[:4] in HOLDOUT]}
        print(path)
        for name, sub in split.items():
            print(f"  {name}: {json.dumps(summarize(sub))}")


if __name__ == "__main__":
    main()
