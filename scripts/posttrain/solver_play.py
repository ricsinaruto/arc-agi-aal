"""Play generated games through the arena Duck-transfer harness with the solver policy (data source (a)).

Usage:
  uv run python scripts/posttrain/solver_play.py                       # every game in the games root
  uv run python scripts/posttrain/solver_play.py --game q002-4ec7e433 [--chunk 8] [--no-reasoning]

For each game the real harness (``ArenaTransferDuckAgent``, via ``relay_play.play``) runs with the
relay transport, and ``posttrain.solver_policy.SolverPolicy`` answers every request from the game's
notes. The run folder ``<out>/<game_id>/`` holds ``relay/turn-NNNN.{request,response}.json``, the
arena game folder ``game/`` (log.jsonl, duck-transcript.txt) and ``result.json``. Afterwards the
log is checked: the game must end in WIN with exactly the solution's action count on every level
(``check`` in the printed summary). ``scripts/posttrain/build_solver_sft.py`` turns the runs into
SFT records.
"""

import argparse
import asyncio
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from posttrain.solver_policy import DEFAULT_CHUNK, SolverPolicy, load_plan, serve  # noqa: E402
from posttrain.split import (  # noqa: E402
    check_eval_games,
    check_training_games,
    is_generated_holdout,
)

HERE = Path(__file__).resolve().parent


def relay_play_module():
    spec = importlib.util.spec_from_file_location("posttrain_relay_play", HERE / "relay_play.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def level_actions(log_path):
    """Actions per cleared level from the arena log (observation records), and the final state."""
    counts, current, state, levels = [], 0, None, 0
    for line in Path(log_path).read_text().splitlines():
        record = json.loads(line)
        if record.get("kind") != "observation" or record.get("action") is None:
            continue
        current += 1
        state = record.get("state")
        if int(record.get("levels") or 0) > levels:
            levels = int(record["levels"])
            counts.append(current)
            current = 0
    return counts, state


def level_score(baseline, actions):
    return min(115.0, 100.0 * (baseline / actions) ** 2) if actions else 0.0


async def play_one(game_id, games_root, out, *, chunk=DEFAULT_CHUNK, reasoning=True, wall_seconds=3600, max_actions=400):
    plan = load_plan(games_root, game_id)
    policy = SolverPolicy(plan, chunk=chunk, reasoning=reasoning)
    relay_play = relay_play_module()
    args = argparse.Namespace(game=game_id, games_dir=str(Path(games_root) / "environment_files"), out=str(out),
                              wall_seconds=wall_seconds, max_actions=max_actions, window=131072, cap=16384,
                              poll_seconds=0.02)
    finished = asyncio.Event()
    player = asyncio.create_task(relay_play.play(args))
    server = asyncio.create_task(serve(Path(out) / "relay", policy, finished))
    done, _ = await asyncio.wait({player, server}, return_when=asyncio.FIRST_COMPLETED)
    if server in done:  # the policy failed: stop the harness
        player.cancel()
        await asyncio.gather(player, return_exceptions=True)
        server.result()
    finished.set()
    answered = await server
    result = json.loads((Path(out) / "result.json").read_text())
    counts, state = level_actions(Path(out) / "game" / "log.jsonl")
    baselines = [len(p) for p in plan]
    scores = [round(level_score(b, n), 1) for b, n in zip(baselines, counts, strict=False)]
    check = state == "WIN" and counts == baselines and all(s >= 100 for s in scores)
    return {"game_id": game_id, "status": result.get("status"), "state": state, "levels": len(counts),
            "total_levels": len(plan), "actions": counts, "baseline": baselines, "scores": scores,
            "turns": result.get("relay_turns"), "answered": answered, "check": check}


def games_in(games_root):
    return sorted(json.loads(p.read_text())["game_id"]
                  for p in (Path(games_root) / "environment_files").glob("q[0-9][0-9][0-9]/*/metadata.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-root", default="data/posttrain/games")
    parser.add_argument("--game", action="append", help="game id (repeatable); default: every game")
    parser.add_argument("--out", default="data/posttrain/solver/runs")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                        help="most solver actions per python call (0 = the rest of the level)")
    parser.add_argument("--no-reasoning", action="store_true", help="send an empty reasoning string")
    args = parser.parse_args()
    games = args.game or games_in(args.games_root)
    # Held-out generated mechanics are played too: their records form the next-action eval split.
    check_training_games([g for g in games if not is_generated_holdout(g)])
    check_eval_games([g for g in games if is_generated_holdout(g)])
    summary = []
    for game_id in games:
        out = Path(args.out) / game_id
        if out.exists():
            shutil.rmtree(out)
        row = asyncio.run(play_one(game_id, args.games_root, out, chunk=args.chunk, reasoning=not args.no_reasoning))
        print(json.dumps(row), flush=True)
        summary.append(row)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if not all(row["check"] for row in summary):
        sys.exit("some games did not reach WIN at the solver's action counts")


if __name__ == "__main__":
    main()
