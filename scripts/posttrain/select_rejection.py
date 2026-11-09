"""Select the most efficient verified episode clears (rejection sampling, posttrain.rejection).

Usage: uv run python scripts/posttrain/select_rejection.py
           [--segments data/posttrain/teacher-episodes/segments.jsonl ...]
           [--episodes data/posttrain/episodes] [--out data/posttrain/rejection] [--k 2]
           [--max-median-multiple 2]

Reads the segment files that ``extract_segments.py --source teacher --runs <episode runs> --out
data/posttrain/teacher-episodes`` writes and the episodes' notes (``<episodes>/batch-*/notes``). Groups
verified clears by (episode_of, tier), drops clears above 2x the group median of completion tokens,
keeps the best k by completion tokens then actions, and rejects any episode of a held-out generated
mechanic. Writes ``<out>/selected.jsonl`` (git-ignored: the selected segment records, each with a
``rejection`` field) and ``<out>/stats.json`` (tracked: per-group counts, token savings against all
clears, and the held-out check).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from posttrain import rejection  # noqa: E402
from posttrain.games import episodes  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", nargs="+", default=["data/posttrain/teacher-episodes/segments.jsonl"])
    parser.add_argument("--episodes", nargs="+", default=["data/posttrain/episodes"],
                        help="episode folders (a folder of batch-* folders, or single batches)")
    parser.add_argument("--out", default="data/posttrain/rejection")
    parser.add_argument("--k", type=int, default=rejection.DEFAULT_K)
    parser.add_argument("--max-median-multiple", type=float, default=rejection.DEFAULT_MAX_MEDIAN_MULTIPLE)
    args = parser.parse_args()
    notes = episodes.load_notes(args.episodes)
    if not notes:
        raise SystemExit(f"No episode notes under {args.episodes}")
    selected, stats = rejection.select(rejection.load_jsonl(args.segments), notes, k=args.k,
                                       max_median_multiple=args.max_median_multiple)
    stats["inputs"] = {"segments": args.segments, "episodes": args.episodes, "episode_games": len(notes)}
    rejection.write(args.out, selected, stats)
    totals, check = stats["totals"], stats["held_out_check"]
    print(f"{totals['episode_clears']} verified episode clears in {totals['groups']} groups -> {totals['selected']} "
          f"selected; {totals['completion_tokens_selected']} of {totals['completion_tokens_all_clears']} completion "
          f"tokens (saved {totals['saved_share']}); {len(stats['groups_without_clears'])} groups without a clear; "
          f"held-out check {'passed' if check['passed'] else 'FAILED: ' + str(check)}")
    if not check["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
