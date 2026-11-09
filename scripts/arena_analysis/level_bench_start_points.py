"""Build the level-bench start points from recorded Flash Duck-transfer runs (27 September).

For each public game (ft09 and lp85 excluded, as on the arena roster) and each level count k that a
recorded run reached, the start point is the shortest recorded action prefix after which the game
has ``levels == k`` (k = 0: a fresh game, empty prefix). Every prefix is replayed on the local
environment files (the arena's offline arcade, ``ONLY_RESET_LEVELS``); a start point is kept only
when the replay reproduces the recorded level count and final board. A failed replay falls back to
the next-shortest recorded prefix (up to ``--tries``); every failure is recorded in the stats.

Runs: ``runs/kaggle-arena-duck-*`` and ``runs/modal-validation/arena-duck-*`` Duck-transfer trials whose
``arena.json`` names the Flash model (one-hour and 30-minute arena runs, probes). Each start point is
tagged with its ``posttrain.split`` split (DEV or HOLDOUT). The bench runs DEV by default; HOLDOUT
start points are a separate confirmation set and never feed a tuning decision.

Usage:
    uv run python scripts/arena_analysis/level_bench_start_points.py \
        --runs-root ../../runs --games-dir ../../environment_files --out data/level_bench
"""

import argparse
import collections
import datetime
import json
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from arena.level_points import check_replay, encode_action, point_id  # noqa: E402
from posttrain.split import DEV, EXCLUDED, HOLDOUT  # noqa: E402

FLASH = "Qwen/Qwen3.8-Flash-Next-NVFP4"
RUN_GLOBS = ("kaggle-arena-duck-*/arena/trials/duck-transfer", "modal-validation/arena-duck-*/arena/trials/duck-transfer")


def flash_trials(runs_root):
    trials = []
    for pattern in RUN_GLOBS:
        for trial in sorted(Path(runs_root).glob(pattern)):
            if "-lost-" in str(trial) or not (trial / "arena.json").is_file():
                continue
            try:
                report = json.loads((trial / "arena.json").read_text())
            except (OSError, ValueError):
                continue
            if (report.get("config") or {}).get("model") == FLASH:
                trials.append(trial)
    return trials


def run_name(trial):
    parts = Path(trial).parts
    i = parts.index("arena") - 1
    return parts[i]


def candidates(log_path):
    """``[(k, prefix, board_hash, total_levels, step)]``: the first observation of each level count a log
    reached through replayable actions (a non-replayable action ends the log's usable prefix)."""
    from arc_infra.perception import board_hash, decode_grid

    out, prefix, level, total = [], [], 0, None
    with open(log_path) as file:
        for line in file:
            if not line.startswith('{"kind":"observation"'):
                continue
            rec = json.loads(line)
            action = rec.get("action")
            if action is not None:
                text = encode_action(action)
                if text is None:
                    break  # explorer steps or an unknown action: later levels are not reproducible
                prefix.append(text)
            total = rec.get("total_levels", total)
            new = int(rec.get("levels", level))
            if new > level:
                if new == level + 1 and rec.get("state") == "NOT_FINISHED":
                    board = board_hash(decode_grid(rec["frames"][-1]))
                    out.append((new, list(prefix), board, total, int(rec.get("step", len(prefix)))))
                level = new
    return out


def scan(args):
    trial, game_dir = args
    try:
        return [(run_name(trial), game_dir.name, *c) for c in candidates(game_dir / "log.jsonl")]
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return [("error", game_dir.name, str(exc))]


def local_versions(games_dir):
    versions = {}
    for meta in Path(games_dir).glob("*/*/metadata.json"):
        data = json.loads(meta.read_text())
        versions[data["game_id"]] = data
    return versions


def split_of(game):
    prefix = game[:4]
    return "dev" if prefix in DEV else "holdout" if prefix in HOLDOUT else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default=str(ROOT / "runs"))
    parser.add_argument("--games-dir", default=str(ROOT / "environment_files"))
    parser.add_argument("--out", default=str(ROOT / "data/level_bench"))
    parser.add_argument("--tries", type=int, default=4, help="Recorded prefixes replayed per (game, level) before giving up")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    from arc_infra.arcade import make_arcade

    versions = local_versions(args.games_dir)
    trials = flash_trials(args.runs_root)
    jobs = [(t, g) for t in trials for g in sorted((t / "games").iterdir())
            if (g / "log.jsonl").is_file() and g.name[:4] not in EXCLUDED and split_of(g.name)]
    print(f"{len(trials)} Flash Duck-transfer trials, {len(jobs)} game logs", flush=True)
    by_point = collections.defaultdict(list)
    scan_errors = []
    with multiprocessing.Pool(args.workers) as pool:
        for rows in pool.imap_unordered(scan, jobs, chunksize=4):
            for row in rows:
                if row[0] == "error":
                    scan_errors.append({"game": row[1], "error": row[2]})
                    continue
                run, game, k, prefix, board, total, step = row
                by_point[game, k].append({"run": run, "prefix": prefix, "board_hash": board,
                                          "total_levels": total, "step": step})
    arc = make_arcade("offline", args.games_dir, Path(tempfile.mkdtemp(prefix="level-bench-")))
    card = arc.open_scorecard(tags=["level-bench-start-points"])
    games = sorted({game for game, _ in by_point} | {j[1].name for j in jobs})
    points, stats, failures = [], [], []
    for game in games:
        meta = versions.get(game)
        if meta is None:
            failures.append({"game": game, "level_index": None, "error": "game version not in the local environment files"})
            continue
        baseline = meta.get("baseline_actions") or []
        env = arc.make(game, scorecard_id=card, save_recording=False)
        total = int(env.observation_space.win_levels)
        from arc_infra.perception import board_hash
        from arena.level_points import last_board

        fresh = {"run": None, "prefix": [], "board_hash": board_hash(last_board(env.observation_space)),
                 "total_levels": total, "step": 0}
        for k in range(total):
            options = [fresh] if k == 0 else sorted(by_point.get((game, k), []), key=lambda c: (len(c["prefix"]), c["run"]))
            row = {"game": game, "split": split_of(game), "level_index": k, "level": k + 1, "recorded_runs": len(options),
                   "replays": 0, "determinism_failures": 0, "kept": False}
            for option in options[: args.tries]:
                env = arc.make(game, scorecard_id=card, save_recording=False)
                point = {"level_index": k, **option}
                row["replays"] += 1
                try:
                    ok, detail, _ = check_replay(env, point)
                except Exception as exc:  # noqa: BLE001 - a replay error is a determinism failure
                    ok, detail = False, f"{type(exc).__name__}: {exc}"
                if not ok:
                    row["determinism_failures"] += 1
                    failures.append({"game": game, "level_index": k, "run": option["run"],
                                     "prefix_actions": len(option["prefix"]), "error": detail})
                    continue
                points.append({
                    "id": point_id(game, k), "game": game, "split": row["split"], "level_index": k, "level": k + 1,
                    "total_levels": total, "baseline_actions": baseline[k] if k < len(baseline) else None,
                    "prefix_actions": len(option["prefix"]), "source_run": option["run"],
                    "source_step": option["step"], "board_hash": option["board_hash"], "prefix": option["prefix"],
                })
                row.update(kept=True, prefix_actions=len(option["prefix"]), source_run=option["run"])
                break
            stats.append(row)
        print(game, [(r["level"], r["recorded_runs"], r["kept"]) for r in stats if r["game"] == game], flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    built = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "start_points.json").write_text(json.dumps({
        "schema": 1, "built": built, "model": FLASH, "runs": len(trials), "game_logs": len(jobs),
        "excluded_games": list(EXCLUDED), "rule": ("shortest recorded prefix reaching levels == k whose offline "
                                                  "replay reproduces the level count and final board"),
        "points": points,
    }, separators=(",", ":")) + "\n")
    summary = {
        "built": built, "runs": len(trials), "game_logs": len(jobs), "points": len(points),
        "by_split": dict(collections.Counter(p["split"] for p in points)),
        "determinism_failures": len(failures), "scan_errors": scan_errors,
        "per_game": {g: {"split": split_of(g), "total_levels": len(rows),
                         "kept_levels": [r["level"] for r in rows if r["kept"]],
                         "recorded_runs": [r["recorded_runs"] for r in rows]}
                     for g in games if (rows := [r for r in stats if r["game"] == g])},
        "rows": stats, "failures": failures,
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("points", "by_split", "determinism_failures")}))


if __name__ == "__main__":
    main()
