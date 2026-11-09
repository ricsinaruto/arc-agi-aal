"""The game side of one attempt (fixed yardstick; root-owned, outside the proposer's reach).

Prime Agent's ``arc3_local_broker.py`` (PrimeIntellect-ai/arc-agi-3-prime-agent 398d4dd6, MIT) with three changes
(actions go through ``env.step``: the published ``env.action<N>()`` calls do not exist in arc_agi 0.9.9's local
wrapper);
every applied action is appended to a record the scorer reads (``--record``), and there is no 500-action cap (the
time slice of docs/rsi.md bounds a game; the published results of that runner exceed 500 actions). Protocol as
published: ``observe`` / ``status`` / ``act`` (1-20 actions; a batch stops at a level change, WIN, or GAME_OVER, which
consumes one counted RESET).

    python -m rsi_loop.broker --game GAME_ID --corpus DIR --socket PATH --record PATH
"""
from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
import time
from pathlib import Path
from typing import Any

MAX_BATCH = 20
VALID_ACTIONS = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}


class Broker:
    def __init__(self, corpus: Path, game_id: str, record: Path, levels: int | None = None) -> None:
        from arc_agi import Arcade, OperationMode

        self.arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(corpus))
        available = [str(e.game_id) for e in self.arcade.get_environments()]
        matches = [g for g in available if g.startswith(game_id)]
        if not matches:
            raise SystemExit(f"game {game_id!r} not in local corpus")
        self.game_id = matches[0]
        self.env = self.arcade.make(self.game_id, seed=0, save_recording=False, include_frame_data=True)
        if self.env is None:
            raise SystemExit(f"Arcade.make failed for {self.game_id}")
        self.actions, self.terminal, self.lock = 0, "ACTIVE", threading.RLock()
        self.cap = levels  # RSI loop: a game whose later levels have no verified solve ends (WIN) after this many
        self.record = record.open("a", buffering=1)
        self.last = self.env.reset()
        self._write({"event": "start", "game": self.game_id, "levels_completed": self._levels()})

    def _levels(self):
        return getattr(self.last, "levels_completed", None)

    def _state(self):
        if self.cap is not None and (self._levels() or 0) >= self.cap:
            return "WIN"
        return str(getattr(getattr(self.last, "state", None), "name", getattr(self.last, "state", "NOT_FINISHED")))

    def _write(self, row):
        self.record.write(json.dumps({"t": round(time.time(), 3), **row}) + "\n")

    def _observation(self) -> dict[str, Any]:
        frame = getattr(self.last, "frame", []) or []
        return {"game_id": self.game_id, "state": self._state(), "levels_completed": self._levels(),
                "available_actions": list(getattr(self.last, "available_actions", []) or []),
                "frame": [z.tolist() if hasattr(z, "tolist") else z for z in frame],
                "actions_taken": self.actions, "terminal": self.terminal}

    def _apply(self, name, data, counted=True) -> None:
        # The published broker calls env.action<N>(), which arc_agi 0.9.9's local wrapper does not have; step() is the
        # same action.
        from arcengine import GameAction

        if name == "RESET":
            self.last = self.env.reset()
        elif name == "ACTION6":
            self.last = self.env.step(GameAction.ACTION6, {"x": int(data["x"]), "y": int(data["y"])})
        else:
            self.last = self.env.step(GameAction[name], {})
        self.actions += 1
        self._write({"n": self.actions, "action": name, "data": data or None, "levels_completed": self._levels(),
                     "state": self._state(), "auto": not counted})

    def dispatch(self, query: dict[str, Any]) -> dict[str, Any]:
        op = query.get("op")
        if op in {"observe", "status"}:
            return {"ok": True, "value": self._observation()}
        if op != "act":
            return {"ok": False, "error": "OperationError", "message": f"unknown op {op!r}"}
        actions = query.get("actions") or []
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_BATCH:
            return {"ok": False, "error": "ProtocolError", "message": f"act takes 1..{MAX_BATCH} actions"}
        with self.lock:
            if self.terminal != "ACTIVE":
                return {"ok": True, "value": self._observation()}
            before = self._levels()
            for action in actions:
                name = action.get("name") if isinstance(action, dict) else None
                if name not in VALID_ACTIONS:
                    return {"ok": False, "error": "ProtocolError", "message": f"bad action {name!r}"}
                data = action.get("data") or {}
                if name == "ACTION6" and not all(isinstance(data.get(k), int) and 0 <= data[k] <= 63
                                                  for k in ("x", "y")):
                    return {"ok": False, "error": "ProtocolError", "message": "ACTION6 needs integer x,y in 0..63"}
                self._apply(name, data)
                state = self._state()
                if state == "GAME_OVER":  # one counted RESET, then the agent re-observes
                    self._apply("RESET", None, counted=False)
                    break
                if state == "WIN":
                    self.terminal = "WIN"
                    break
                if self._levels() != before:
                    break
        return {"ok": True, "value": self._observation()}


def serve(broker: Broker, socket_path: Path) -> None:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for line in self.rfile:
                line = line.strip()
                if not line:
                    continue
                try:
                    response = broker.dispatch(json.loads(line))
                except Exception as exc:  # noqa: BLE001 - a broker fault must be visible
                    response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
                self.wfile.write(json.dumps(response).encode() + b"\n")
                self.wfile.flush()

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True
        allow_reuse_address = True

    if socket_path.exists():
        socket_path.unlink()
    with Server(str(socket_path), Handler) as server:
        os.chmod(socket_path, 0o666)  # the directory holding it decides who reaches it
        print(f"broker ready game={broker.game_id}", flush=True)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=None)
    args = parser.parse_args()
    serve(Broker(args.corpus, args.game, args.record, args.levels), args.socket)


if __name__ == "__main__":
    main()
