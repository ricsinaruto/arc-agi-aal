"""Play the public Duck notebook's own TAAF benchmark loop on selected games.

This mirrors the scored notebook (wuliao0, Duck Qwen3.8 Anim Base, Version 1) cells that
build the benchmark, apply the exact public-25 solver settings, build offline ``GameAPI``
games from the competition environment files and run ``Benchmark.run`` with minimal
diagnostics. The notebook's pickled ``Benchmark``, ``HarnessSolver`` and ``KaggleTarget``
are reconstructed from ``configs/duck_public_notebook.json``; only the game list, the
per-game runtime and the soft end time differ. The notebook's vLLM watchdog is not
started: the server is shared with the duck-transfer control and a restart would break
the comparison. Run this in an interpreter whose environment carries the analyzer
settings persisted by the pinned serving script.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

TERMINAL_STATES = {"won", "gave_up", "cancelled"}


def load_config(path):
    return json.loads(Path(path).read_text())


def offline_games(env_dir):
    """The notebook's offline game list from the competition's environment files."""
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir
    )
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {env_dir}.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


def apply_public_settings(solver, seconds):
    """The notebook's exact public-25 settings, except the per-game runtime."""
    solver.max_runtime_s_per_game = float(seconds)
    solver.analyzer_timeout = 900.0
    solver.concurrency = 28
    solver.max_actions_per_game = None
    solver.save_request_logs = False


def build_benchmark(config, games, output, seconds, passes=1):
    """The notebook's ``Benchmark`` and solver from plain configuration fields."""
    import taaf.benchmark
    from inference.framework.solver import HarnessSolver

    solver = HarnessSolver(**config["solver"])
    apply_public_settings(solver, seconds)
    benchmark = taaf.benchmark.Benchmark(
        label=config["benchmark"]["label"],
        games=list(games),
        solver=solver,
        n_passes=passes,
        job_dir=Path(output),
        periodic_save_interval_s=config["benchmark"]["periodic_save_interval_s"],
    )
    return benchmark


def build_target(config):
    from taaf.deploy_kaggle import KaggleTarget

    target = KaggleTarget(**config["target"])
    target.actual_run_as_submission = False
    target.is_competition_rerun = False
    return target


def select_games(config, games_dir, selected):
    public_ids = list(config["public_game_ids"])
    unknown = sorted(set(selected) - set(public_ids))
    if unknown:
        raise ValueError(f"Not public games: {unknown}")
    games = offline_games(games_dir)
    by_id = {game.env_name: game for game in games}
    if len(by_id) != len(games):
        raise RuntimeError("The offline public game list contains duplicate IDs.")
    missing = sorted(set(selected) - set(by_id))
    if missing:
        raise RuntimeError(f"Offline public game set is missing {missing}.")
    order = [game_id for game_id in public_ids if game_id in set(selected)]
    return order, [by_id[game_id] for game_id in order]


def run_rows(runs):
    rows = []
    for run in runs:
        action_tokens = sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens
        # The pinned solver's final note includes inspection-only responses after
        # its last action. Its action history does not account for that tail.
        note = re.fullmatch(r"tokens=(\d+)", run.solver_note or "")
        analyzer_tokens = int(note[1]) if note else None
        use_analyzer = analyzer_tokens is not None and analyzer_tokens >= action_tokens
        rows.append(
            {
                "game_id": run.game_id,
                "state": run.state,
                "final_score": run.final_score,
                "levels_completed": run.levels_completed,
                "number_of_levels": run.number_of_levels,
                "actions": len(run.history),
                "actions_per_level": list(run.actions_per_level),
                "base_actions_per_level": (
                    None if run.base_actions_per_level is None else list(run.base_actions_per_level)
                ),
                "generated_tokens": analyzer_tokens if use_analyzer else action_tokens,
                "action_accounted_generated_tokens": action_tokens,
                "generated_tokens_source": "analyzer_final_note" if use_analyzer else "action_history",
                "solver_note": run.solver_note,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/duck_public_notebook.json")
    parser.add_argument("--games-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--game", action="append", help="Selected public game; all 25 if omitted")
    parser.add_argument("--passes", type=int, default=1, help="Benchmark passes over the games")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument(
        "--grace", type=float, default=90.0, help="Soft end after the per-game runtime"
    )
    args = parser.parse_args(argv)
    if args.passes < 1:
        raise ValueError("passes must be at least 1")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TAAF_RUN_AS_SUBMISSION"] = "0"
    os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1"
    os.environ["ONLY_RESET_LEVELS"] = "true"
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RECORDINGS_DIR", str(output / "server_recording"))
    # Benchmark.run() would otherwise capture git state from this checkout.
    (output / "git_status.txt").write_text(config["git_status"])
    target = build_target(config)
    selected = args.game or list(config["public_game_ids"])
    order, games = select_games(config, args.games_dir, selected)
    bm = build_benchmark(config, games, output, args.seconds, args.passes)
    started = time.time()
    soft_end = datetime.now() + timedelta(seconds=args.seconds * args.passes + args.grace)
    settings = {
        "max_runtime_s_per_game": bm.solver.max_runtime_s_per_game,
        "analyzer_timeout": bm.solver.analyzer_timeout,
        "concurrency": bm.solver.concurrency,
        "max_actions_per_game": bm.solver.max_actions_per_game,
        "save_request_logs": bm.solver.save_request_logs,
        "passes": args.passes,
        "soft_end_seconds_after_start": args.seconds * args.passes + args.grace,
        "analyzer_environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                ("LOCAL_ANALYZER_", "MULTIMODAL_", "OPENAI_BASE_URL", "OPENAI_PROVIDER")
            )
        },
    }
    print(
        json.dumps({"event": "public_trial_start", "games": order, "settings": settings}),
        flush=True,
    )
    summary = {"status": "error", "games": order, "settings": settings, "started_epoch": started}
    try:
        asyncio.run(
            bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)
        )
        bm._save_json()
        runs = list(bm.game_runs)
        rows = run_rows(runs)
        unfinished = [
            (r["game_id"], r["state"], r["final_score"])
            for r in rows
            if r["state"] not in TERMINAL_STATES or r["final_score"] is None
        ]
        crashed = [r["game_id"] for r in rows if r["state"] == "crashed"]
        summary.update(
            rows=rows,
            run_ids=[run.game_id for run in runs],
            total_actions=sum(r["actions"] for r in rows),
            mean_final_score=(sum(r["final_score"] or 0 for r in rows) / len(rows))
            if rows
            else None,
            unfinished=unfinished,
            crashed=crashed,
        )
        try:
            from inference.tools.eval import evaluate_runs, save_score_file

            evaluation = evaluate_runs([output])
            score_path = save_score_file(
                evaluation, run_dirs=[output], output_path=output / "score.json"
            )
            summary["evaluation"] = {
                "overall_score": evaluation.overall_score,
                "games": [
                    {
                        "game_id": g.game_id,
                        "average_score": g.average_score,
                        "average_levels_completed": g.average_levels_completed,
                        "total_levels": g.total_levels,
                    }
                    for g in evaluation.games
                ],
                "score_path": str(score_path),
            }
        except Exception as exc:  # noqa: BLE001 - the public scorer is reported, never required
            summary["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        if summary["run_ids"] != order * args.passes:
            raise RuntimeError(f"Public run coverage changed: {summary['run_ids']}")
        if unfinished:
            raise RuntimeError(f"Public runs did not finalize cleanly: {unfinished}.")
        if crashed:
            raise RuntimeError(f"Public runs crashed: {crashed}.")
        summary["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - recorded for the driver
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["elapsed_seconds"] = time.time() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"event": "public_trial_result", **summary}), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
