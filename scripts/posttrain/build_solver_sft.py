"""Build the solver SFT records (data source (a)) from the solver runs.

Usage:
  uv run python scripts/posttrain/solver_play.py            # plays every generated game first
  uv run python scripts/posttrain/build_solver_sft.py [--runs data/posttrain/solver/runs] \
      [--out data/posttrain/solver]

Each run folder ``<runs>/<game_id>/`` must hold a won game at exactly the solver's action count on
every level (re-checked here from its ``game/log.jsonl``). Writes ``<out>/records.jsonl`` (one
``posttrain.sft_records`` record per harness request of a training game; git-ignored),
``<out>/eval_records.jsonl`` (the held-out generated mechanics, ``split.GENERATED_HOLDOUT``: the
BFS-optimal next action at each request, for next-action checks; never training data) and
``<out>/stats.json`` (tracked: counts per game and level, token totals, both splits).
"""

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from posttrain.sft_records import solver_records  # noqa: E402
from posttrain.split import (  # noqa: E402
    check_eval_games,
    check_training_games,
    is_generated_holdout,
)


def solver_play_module():
    spec = importlib.util.spec_from_file_location("posttrain_solver_play", Path(__file__).with_name("solver_play.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completion_estimate(target):
    from posttrain.sft_records import CHARS_PER_TOKEN

    chars = len(target["reasoning"]) + len(target["content"]) + len(json.dumps(target["tool_calls"]))
    return int(chars / CHARS_PER_TOKEN) + 1


def build(runs_dir, games_root, *, split="train"):
    """Records and stats of the runs in one split: ``train`` (training games) or ``eval`` (the
    held-out generated mechanics)."""
    play = solver_play_module()
    runs = sorted(p for p in Path(runs_dir).iterdir() if (p / "relay").is_dir())
    if split == "eval":
        runs = [p for p in runs if is_generated_holdout(p.name)]
        check_eval_games([p.name for p in runs])
    else:
        runs = [p for p in runs if not is_generated_holdout(p.name)]
        check_training_games([p.name for p in runs])
    records, games = [], {}
    for run in runs:
        game_id = run.name
        baselines = [len(level) for level in play.load_plan(games_root, game_id)]
        counts, state = play.level_actions(run / "game" / "log.jsonl")
        if state != "WIN" or counts != baselines:
            raise RuntimeError(f"{game_id}: {state} with {counts} actions per level, solution {baselines}")
        game_records = solver_records(run, game_id, split=split)
        by_level = collections.defaultdict(lambda: {"records": 0, "prompt_tokens_estimate": 0,
                                                    "completion_tokens_estimate": 0})
        for record in game_records:
            row = by_level[record["level"]]
            row["records"] += 1
            row["prompt_tokens_estimate"] += record["prompt_tokens_estimate"]
            row["completion_tokens_estimate"] += completion_estimate(record["target"])
        games[game_id] = {
            "levels": len(baselines),
            "actions": counts,
            "baseline_actions": baselines,
            "records": len(game_records),
            "summary_records": sum(r["purpose"] == "summary" for r in game_records),
            "max_prompt_tokens_estimate": max(r["prompt_tokens_estimate"] for r in game_records),
            "per_level": {str(level): {"actions": counts[level - 1], **row}
                          for level, row in sorted(by_level.items())},
        }
        records += game_records
    stats = {
        "source": "solver",
        "split": split,
        "games": len(games),
        "records": len(records),
        "actions": sum(sum(g["actions"]) for g in games.values()),
        "prompt_tokens_estimate": sum(r["prompt_tokens_estimate"] for r in records),
        "completion_tokens_estimate": sum(completion_estimate(r["target"]) for r in records),
        "images": sum(r["images"] for r in records),
        "token_estimate": "text characters / 4.1 + 64 per board image (posttrain.sft_records)",
        "per_game": games,
    }
    return records, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="data/posttrain/solver/runs")
    parser.add_argument("--games-root", default="data/posttrain/games")
    parser.add_argument("--out", default="data/posttrain/solver")
    args = parser.parse_args()
    records, stats = build(args.runs, args.games_root)
    eval_records, eval_stats = build(args.runs, args.games_root, split="eval")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("records.jsonl", records), ("eval_records.jsonl", eval_records)):
        with open(out / name, "w") as f:
            for record in rows:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats["eval_split"] = eval_stats
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    for row in (stats, eval_stats):
        print(json.dumps({k: v for k, v in row.items() if k not in ("per_game", "eval_split")}))


if __name__ == "__main__":
    main()
