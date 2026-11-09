"""Build the AZR-style auxiliary task set (predict / reach / transition) from game code.

    uv run python scripts/posttrain/build_aux_tasks.py              # the 14 DEV games -> data/posttrain/aux
    uv run python scripts/posttrain/build_aux_tasks.py --generated  # training generated games -> aux-generated
    uv run python scripts/posttrain/build_aux_tasks.py --eval       # held-out generated games -> aux-eval

Writes ``<out>/tasks.jsonl`` (one task per line, git-ignored) and ``<out>/stats.json``. Every task passed
its self-check (the engine reference, or for ``transition`` a lookup oracle, is accepted by the task's
own verifier and a trivial wrong answer is rejected); identical prompts are dropped. Only DEV and
training generated games are accepted (``posttrain.split.check_training_games``); the held-out generated
mechanics and skins (``split.GENERATED_HOLDOUT``) are built only with ``--eval``, into their own folder
(``check_eval_games``). ``--generated`` reads the generated-games store (``data/posttrain/games``: the
games' code; their notes carry the hidden rules, the engine gives the reference answers).

Every task is flagged ``no_reasoning`` (the reference answer is the engine's, with no model reasoning), so
a mixture can weight or mask it.
"""

import argparse
import collections
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from posttrain.aux_tasks import FAMILIES, GameLibrary, generate  # noqa: E402
from posttrain.split import (  # noqa: E402
    check_eval_games,
    check_training_games,
    is_dev,
    is_generated_holdout,
    is_training_game,
)

GENERATED_ENV_DIR = REPO / "data" / "posttrain" / "games" / "environment_files"


def build_game(args):
    game_id, env_dirs, n, seed, families = args
    started = time.monotonic()
    library = GameLibrary(env_dirs)
    tasks, stats = generate(game_id, library.factory(game_id), n, seed=seed, families=families, library=library)
    return game_id, tasks, stats, round(time.monotonic() - started, 2)


def summary(values):
    if not values:
        return None
    values = sorted(values)
    return {"n": len(values), "mean": round(statistics.fmean(values), 1), "p50": values[len(values) // 2],
            "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1]}


def describe(tasks):
    by_family = collections.defaultdict(list)
    for task in tasks:
        by_family[task["family"]].append(task)
    out = {}
    for family, rows in sorted(by_family.items()):
        entry = {
            "tasks": len(rows),
            "prompt_tokens_est": summary([len(t["prompt"]) // 4 for t in rows]),
            "answer_tokens_est": summary([len(t["answer"]) // 4 for t in rows if t["answer"]]),
            "levels": dict(sorted(collections.Counter(t["level"] for t in rows).items())),
        }
        if family == "predict":
            entry["format"] = dict(collections.Counter(t["meta"]["format"] for t in rows))
            entry["n_actions"] = dict(sorted(collections.Counter(t["meta"]["n_actions"] for t in rows).items()))
            entry["no_change"] = sum(1 for t in rows if not t["meta"]["changed_cells"])
            entry["border_only_change"] = sum(1 for t in rows if t["meta"]["changed_cells"]
                                              and not t["meta"]["substantive"])
            entry["changed_cells"] = summary([t["meta"]["changed_cells"] for t in rows])
        elif family == "reach":
            entry["k"] = dict(sorted(collections.Counter(t["meta"]["k"] for t in rows).items()))
            entry["shortened"] = sum(1 for t in rows if t["meta"]["k"] < t["meta"]["walk_k"])
            entry["with_clicks"] = sum(1 for t in rows if "MOUSE" in t["answer"])
        elif family == "transition":
            entry["n_shown"] = dict(sorted(collections.Counter(t["meta"]["n_shown"] for t in rows).items()))
            entry["actions"] = dict(collections.Counter(t["verifier"]["action"] for t in rows))
            entry["held_unchanged"] = sum(t["meta"]["held_unchanged"] for t in rows)
        out[family] = entry
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-dir", action="append", default=None,
                        help="environments directory in the public layout (repeatable; default environment_files)")
    parser.add_argument("--games", nargs="*", default=None,
                        help="game ids or prefixes (default: every DEV or generated game found)")
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--families", nargs="*", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--out", default=None,
                        help="default data/posttrain/aux, aux-generated (--generated) or aux-eval (--eval)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generated", action="store_true",
                       help="the training generated games of data/posttrain/games (held-out mechanics excluded)")
    group.add_argument("--eval", action="store_true",
                       help="the held-out generated mechanics and skins only (evaluation split)")
    args = parser.parse_args(argv)

    default_dirs = [GENERATED_ENV_DIR] if (args.generated or args.eval) else [REPO / "environment_files"]
    env_dirs = tuple(Path(d).resolve() for d in (args.env_dir or default_dirs))
    library = GameLibrary(env_dirs)
    if args.games:
        games = [library.resolve(g) for g in args.games]
    elif args.eval:
        games = sorted(g for g in library.dirs() if is_generated_holdout(g))
    elif args.generated:
        games = sorted(g for g in library.dirs() if is_training_game(g) and not is_dev(g))
    else:
        games = sorted(g for g in library.dirs() if is_training_game(g))
    check = check_eval_games if args.eval else check_training_games
    check(games)
    name = "aux-eval" if args.eval else "aux-generated" if args.generated else "aux"
    args.out = args.out or str(REPO / "data" / "posttrain" / name)
    if not games:
        raise SystemExit(f"no games found in {[str(d) for d in env_dirs]}")

    started = time.monotonic()
    jobs = [(g, env_dirs, args.per_family, args.seed, tuple(args.families)) for g in games]
    results = {}
    def done(result):
        game_id, tasks, stats, seconds = result
        results[game_id] = (tasks, stats, seconds)
        made = {f: s["made"] for f, s in stats.items() if f in FAMILIES}
        print(f"{game_id}: {made} in {seconds:.1f}s", flush=True)

    if args.workers <= 1:  # in process (no pickling: also works when this file is loaded by path)
        for job in jobs:
            done(build_game(job))
    else:
        with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
            for future in as_completed([pool.submit(build_game, job) for job in jobs]):
                done(future.result())

    tasks, seen, duplicates = [], set(), 0
    per_game = {}
    for game_id in games:
        game_tasks, stats, seconds = results[game_id]
        kept = 0
        for task in game_tasks:
            if task["prompt"] in seen:
                duplicates += 1
                continue
            seen.add(task["prompt"])
            task["no_reasoning"] = True
            task["split"] = "eval" if args.eval else "train"
            tasks.append(task)
            kept += 1
        per_game[game_id] = {"seconds": seconds, "families": stats, "kept": kept}
    check({t["game"] for t in tasks})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*.jsonl\n")
    with open(out / "tasks.jsonl", "w") as handle:
        for task in tasks:
            handle.write(json.dumps(task, separators=(",", ":")) + "\n")
    counts = collections.defaultdict(dict)
    for task in tasks:
        counts[task["game"]][task["family"]] = counts[task["game"]].get(task["family"], 0) + 1
    stats = {
        "config": {"split": "eval" if args.eval else "train", "no_reasoning": True,
                   "env_dirs": [os.path.relpath(d, REPO) for d in env_dirs], "games": games,
                   "per_family": args.per_family, "families": args.families, "seed": args.seed},
        "total_tasks": len(tasks),
        "cross_game_duplicate_prompts": duplicates,
        "counts": {g: dict(sorted(c.items())) for g, c in sorted(counts.items())},
        "families": describe(tasks),
        "per_game": per_game,
        "wall_seconds": round(time.monotonic() - started, 1),
        "tasks_jsonl_bytes": (out / "tasks.jsonl").stat().st_size,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(f"{len(tasks)} tasks -> {out / 'tasks.jsonl'} ({stats['tasks_jsonl_bytes'] / 1e6:.1f} MB) "
          f"in {stats['wall_seconds']}s")


if __name__ == "__main__":
    main()
