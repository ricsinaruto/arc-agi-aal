"""Play one game through the arena Duck-transfer harness with a relay policy (posttrain.relay).

Usage:
  uv run python scripts/posttrain/relay_play.py --game q001-c8ba2de3 \
      [--games-dir data/posttrain/games/environment_files] [--out data/posttrain/relay/<game>-<stamp>] \
      [--wall-seconds 3600] [--max-actions 400]

The script starts the harness and waits; for every model request it writes
``<out>/relay/turn-NNNN.{request.json,view.txt}`` and waits for ``turn-NNNN.response.json``
(``scripts/posttrain/relay_answer.py`` writes one). The game folder ``<out>/game/`` has the arena
layout (log.jsonl, duck-transcript.txt, prompts/). ``<out>/result.json`` is written at the end.
The harness clocks run on virtual time (the relay's waiting time is replaced by the served model's
modelled time), so the runtime lines in the prompts look like a real run's.
"""

import argparse
import asyncio
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


async def play(args):
    from arc_agi import Arcade, OperationMode

    from arc_infra.config import Config
    from arc_infra.model import ModelClient
    from arena.clock import ActiveClock, GlobalClock
    from arena.duck import ArenaTransferDuckAgent
    from arena.gate import GateSlot, KVGate
    from arena.scheduler import GameRecord
    from duck_transfer import loader
    from posttrain.relay import RelayTransport

    out = Path(args.out)
    transport = RelayTransport(out / "relay", poll_seconds=getattr(args, "poll_seconds", 0.5))
    cfg = Config.load(str(ROOT / "configs/duck_controls.yaml"))
    cfg.max_model_len, cfg.max_output_tokens = args.window, args.cap
    cfg.reasoning_effort, cfg.temperature = "xhigh", 0.7
    cfg.gameplay_seconds = args.wall_seconds
    cfg.max_actions = args.max_actions
    loader.configure(cfg)
    client = ModelClient(cfg)
    client.client = transport
    wall = GlobalClock(args.wall_seconds, now=transport.now)
    game = GameRecord(args.game, 0, ActiveClock(wall, now=transport.now, budget_clock="busy"))
    game.ready = asyncio.Event()
    gate = KVGate(int(291_900 * 0.9), slots=8)
    client.slots = GateSlot(gate, game, cap=args.cap, window=args.window)
    os.environ["ONLY_RESET_LEVELS"] = "true"
    recordings = tempfile.mkdtemp(prefix="relay-recordings-")
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(Path(args.games_dir).resolve()),
                 recordings_dir=recordings)
    card = arc.open_scorecard(tags=["posttrain-relay"])
    agent = ArenaTransferDuckAgent(args.game, arc, card, client, cfg, out / "game",
                                   ROOT / ".analysis-venv/bin/python", transport.now() + 10 * 24 * 3600)
    agent.attach(game, window=args.window, cap=args.cap)
    wall.release()
    game.clock.release()
    (out / "relay").mkdir(parents=True, exist_ok=True)
    (out / "relay" / "STARTED").write_text(datetime.datetime.now(datetime.UTC).isoformat() + "\n")
    result = await agent.run()
    result["virtual_seconds"] = round(wall.elapsed(), 1)
    result["relay_turns"] = transport.turn
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    (out / "relay" / "DONE").write_text(json.dumps({k: result.get(k) for k in ("status", "levels", "actions", "score")}) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "levels", "actions", "score", "virtual_seconds")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--games-dir", default="data/posttrain/games/environment_files")
    parser.add_argument("--out")
    parser.add_argument("--wall-seconds", type=float, default=3600)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--window", type=int, default=131072)
    parser.add_argument("--cap", type=int, default=16384)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out = args.out or f"data/posttrain/relay/{args.game}-{stamp}"
    asyncio.run(play(args))


if __name__ == "__main__":
    main()
