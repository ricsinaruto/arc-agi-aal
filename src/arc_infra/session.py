"""One independent game session: environment, scorecard, workspace and counters."""

import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from arcengine import GameAction

from .perception import Observation, board_hash


class Workspace:
    """Append-only per-game log of observations, model traffic and agent events."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "log.jsonl"

    def append(self, kind, **payload):
        with self.log.open("a") as file:
            file.write(
                json.dumps(
                    {"kind": kind, "time": time.time(), **payload},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def observe(self, step, observation, action=None):
        data = {"step": step, "action": action, **observation.payload()}
        self.append("observation", **data)
        path = self.root / "current.json"
        temp = self.root / "current.tmp"
        temp.write_text(json.dumps(data, separators=(",", ":")))
        temp.replace(path)


class GameSession:
    """Base class for fleet agents; subclasses implement ``run``."""

    def __init__(self, game_id, arc, card, client, config, workspace, interpreter, deadline):
        self.game_id, self.arc, self.card = game_id, arc, card
        self.client, self.config = client, config
        self.workspace = Workspace(workspace)
        self.interpreter, self.deadline = interpreter, deadline
        self.env = None
        self.obs = None
        self.previous = None
        self.steps = self.turns = self.level_steps = self.resets = self.analysis_turns = 0
        self.prompt_tokens = self.completion_tokens = self.context_resets = 0
        self.context_prunes = self.context_compactions = self.compaction_calls = 0
        self.model_seconds = 0.0
        self.last_prompt_tokens = 0
        self.last_action = "RESET"
        self.no_ops = Counter()
        self.status, self.error = "starting", None
        self.started = time.monotonic()
        self.last_observation_at = self.started

    def remaining(self):
        return max(0, self.deadline - time.monotonic())

    def runtime_message(self, *, record=True):
        now = time.monotonic()
        runtime = {
            "elapsed_seconds": round(now - self.started, 1),
            "remaining_seconds": round(max(0, self.deadline - now), 1),
            "actions_executed": self.steps,
            "levels_completed": self.obs.levels if self.obs else 0,
            "seconds_since_last_observation": round(now - self.last_observation_at, 1),
            "analysis_replies_since_action": self.analysis_turns,
            "mean_model_request_seconds": round(self.model_seconds / max(1, self.turns), 1),
            "target_actions_per_900_seconds": self.config.action_target,
        }
        if record:
            self.workspace.append("model_request", step=self.steps, runtime=runtime)
        return {"role": "user", "content": "[Fresh runtime status]\n" + json.dumps(runtime)}

    def record_reply(self, reply, purpose="gameplay"):
        self.turns += 1
        self.model_seconds += reply.get("seconds", 0)
        usage = reply.get("usage", {})
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.prompt_tokens += self.last_prompt_tokens
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.workspace.append("model", step=self.steps, purpose=purpose, **reply)

    async def step(self, action):
        if self.remaining() <= 0:
            raise TimeoutError("Gameplay deadline reached")
        issued = time.monotonic()
        if action.name == "RESET":
            call = self.env.reset
            args = ()
        else:
            call = self.env.step
            args = (
                GameAction[action.name],
                {"x": action.x, "y": action.y} if action.name == "ACTION6" else {},
            )
        task = asyncio.create_task(asyncio.to_thread(call, *args))
        cancelled = False
        try:
            raw = await asyncio.shield(task)
        except asyncio.CancelledError:
            # Finish recording an already-issued action before closing its scorecard.
            raw = await task
            cancelled = True
        self.previous = self.obs
        self.obs = Observation.from_raw(raw, self.previous)
        self.last_observation_at = time.monotonic()
        self.steps += 1
        self.level_steps += 1
        if action.name == "RESET":
            self.resets += 1
            self.no_ops.clear()
        self.last_action = action.name
        if self.obs.levels != self.previous.levels:
            self.level_steps = 0
            self.no_ops.clear()
        elif board_hash(self.obs.board) == board_hash(self.previous.board):
            key = (self.obs.levels, board_hash(self.obs.board), action.identity())
            self.no_ops[key] += 1
        self.workspace.observe(self.steps, self.obs, action.payload())
        self.workspace.append(
            "action_timing",
            step=self.steps,
            issued_gameplay_seconds=issued - self.started,
            seconds=time.monotonic() - issued,
        )
        if cancelled:
            raise asyncio.CancelledError

    async def run(self):
        raise NotImplementedError

    def result(self):
        return {
            "game_id": self.game_id,
            "status": self.status,
            "error": self.error,
            "levels": self.obs.levels if self.obs else 0,
            "total_levels": self.obs.total_levels if self.obs else 0,
            "actions": self.steps,
            "resets": self.resets,
            "model_calls": self.turns,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "context_resets": self.context_resets,
            "context_prunes": self.context_prunes,
            "context_compactions": self.context_compactions,
            "compaction_calls": self.compaction_calls,
            "seconds": time.monotonic() - self.started,
        }
