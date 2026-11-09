"""One arena trial in its own process for tests/test_arena_checkpoint.py: real Duck-transfer agents, scheduler, gate
and python interpreters on toy environments and a scripted model. ``python checkpoint_driver.py OUTPUT [FEATURES]``;
the test kills it, starts it again with ``ARC_CHECKPOINT_RESUME=1`` and compares ``digest.json`` with an
uninterrupted run."""

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAMES = [f"toy{i}-0000000{i}" for i in range(4)]
ACTIONS, LATENCY = 14, 0.25
SCRIPT = ["ACTION1", None, "ACTION1", "ACTION4", "ACTION1", None, "ACTION4", "ACTION4", "ACTION1", "ACTION6"]
# Values that follow the clock, not the game: masked before two runs are compared.
TIMING = re.compile(r'("?[a-z_]*(?:seconds|elapsed|remaining|epoch|time)[a-z_]*"?[:=] ?)-?[0-9.]+')


class ToyEnv:
    """Deterministic: ACTION1 advances (a level every few presses, WIN at three), two ACTION4 in a row end the game,
    ACTION6 flips a cell."""

    def __init__(self, seed):
        self.seed, self.calls = seed, []
        self.level = self.count = 0
        self.state, self.last = "NOT_FINISHED", None
        self.board = np.zeros((64, 64), dtype=np.uint8)
        self.board[0, seed % 64] = 3
        self.observation_space = self.raw()

    def raw(self):
        return SimpleNamespace(frame=[self.board.copy()], levels_completed=self.level, win_levels=3,
                               state=SimpleNamespace(name=self.state), available_actions=[1, 4, 6], full_reset=False)

    def step(self, action, data=None):
        name = action.name
        self.calls.append(name)
        if self.state == "NOT_FINISHED":
            if name == "ACTION1":
                self.count += 1
                self.board[1 + self.count % 60, (self.seed * 7 + self.count * 3) % 64] = 1 + self.count % 14
                if self.count % (2 + self.seed % 3) == 0:
                    self.level += 1
                    self.state = "WIN" if self.level == 3 else self.state
            elif name == "ACTION4":
                self.board[63, self.count % 64] = (self.board[63, self.count % 64] + 1) % 16
                if self.last == "ACTION4":
                    self.state = "GAME_OVER"
            elif name == "ACTION6":
                self.board[data["y"], data["x"]] ^= 5
        self.last = name
        self.observation_space = self.raw()
        return self.observation_space

    def reset(self):
        self.calls.append("RESET")
        self.state, self.last = "NOT_FINISHED", None
        self.board[62, :] = (self.board[62, :] + 1) % 16
        self.observation_space = self.raw()
        return self.observation_space


class ToyArcade:
    def __init__(self):
        self.envs = {}

    def get_environments(self):
        return [SimpleNamespace(game_id=g) for g in GAMES]

    def open_scorecard(self, tags):
        return "card"

    def make(self, game_id, **kwargs):
        self.envs[game_id] = ToyEnv(GAMES.index(game_id))
        return self.envs[game_id]

    def close_scorecard(self, card):
        envs = [SimpleNamespace(id=g, score=float(self.envs[g].level)) for g in GAMES]
        return SimpleNamespace(environments=envs, model_dump=lambda **kw: {"environments": [e.id for e in envs]})


class OfflineMonitor:
    def __init__(self, output, base_url):
        self.start = None

    async def sample(self):
        return None

    async def run(self):
        await asyncio.Event().wait()

    def summary(self):
        return {"valid": True, "preemptions": 0, "accepted_generation_tokens": 10}


def scripted(n):
    """The model's reply to a request holding ``n`` assistant messages: python that keeps interpreter state."""
    step = SCRIPT[n % len(SCRIPT)]
    code = ("memo = []\n" if n == 0 else "") + "memo.append(len(memo) * 3 + 1)\n"
    if step == "ACTION6":
        code += f"action([{{'action': 'ACTION6', 'row': {n % 60 + 2}, 'col': {n * 5 % 64}}}])\n"
    elif step:
        code += f"action([{{'action': '{step}'}}])\n"
    code += "print('memo', len(memo), sum(memo))\n"
    call = {"id": f"call{n}", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}
    return {"choices": [{"finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": f"World model: step {n}.", "tool_calls": [call]}}],
            "usage": {"completion_tokens": 200 + 10 * (n % 7), "prompt_tokens": 2000 + 50 * n}}


async def model(request):
    if request.url.path.endswith("/tokenize"):
        count = 1000 + 400 * request.content.count(b'"image_url"')
        return httpx.Response(200, json={"count": count, "max_model_len": 262144})
    payload = json.loads(request.content)
    await asyncio.sleep(LATENCY)
    return httpx.Response(200, json=scripted(sum(m.get("role") == "assistant" for m in payload["messages"])))


def unordered(value):
    """Lists of records sorted: which of two games finished first follows the clock, not the games."""
    if isinstance(value, dict):
        return {key: unordered(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [unordered(item) for item in value]
        return sorted(items, key=lambda item: TIMING.sub(r"\1T", json.dumps(item, sort_keys=True))) if any(
            isinstance(item, dict) for item in items) else items
    return value


def masked(value):
    return hashlib.sha256(TIMING.sub(r"\1T", json.dumps(value, sort_keys=True, default=str)).encode()).hexdigest()[:16]


async def main(output, features):
    from arena import trial as trial_module
    from arena.scheduler import token_budget_settings

    study = json.loads((ROOT / "configs/arena.json").read_text())
    study.update(games=GAMES, wall_seconds=120.0, active_limit=3, generation_slots=2, level_budget_policy="priority",
                 backlog_policy="readmissions", budget_clock="tokens", token_budget=token_budget_settings(),
                 priority={}, competition={"max_actions_per_game": ACTIONS})
    study.update(features)
    study.pop("cost_normalization", None)
    client = httpx.AsyncClient
    httpx.AsyncClient = lambda **kwargs: client(transport=httpx.MockTransport(model), **kwargs)
    trial_module.kv_pool_tokens = lambda server_dir: 291_900
    if study.get("cache_admission") == "hybrid_blocks":  # the production gate: a measured block table
        rows = [{"tokens": n, "blocks": n // 1024 + 5, "groups": [n // 1024, 5]} for n in range(16384, 262145, 16384)]
        layout = {"schema": 1, "num_blocks": 2000, "costs": rows, "max_model_len": 262144, "lookahead": 3,
                  "rule": "test table"}
        trial_module.BlockBudget.load = classmethod(lambda cls, server_dir, margin=0.9: cls(layout, margin))
    trial_module.GpuMonitor = OfflineMonitor
    arcade = ToyArcade()
    trial = trial_module.Trial(study, next(t for t in study["trials"] if t["kind"] == "duck-transfer"), source=ROOT,
                               games_dir=None, output=output / "t", server_dir=output, base_url="http://model/v1",
                               log=lambda *a, **k: None, arcade=arcade)
    report = await trial.run()
    games = {}
    for agent in trial.duck_workers:
        history = agent._analyzer._history_messages
        games[agent.game_id] = {
            "calls": arcade.envs[agent.game_id].calls, "board": hashlib.sha256(agent.obs.board.tobytes()).hexdigest()[:16],
            "steps": agent.steps, "turns": agent.turns, "levels": agent.obs.levels, "status": agent.status,
            "error": agent.error, "resets": agent.resets, "completion_tokens": agent.completion_tokens,
            "prompt_tokens": agent.prompt_tokens, "budget_tokens": agent.game.clock.budget_tokens(),
            "charges": len(agent.game.clock.charges), "sandbox_calls": agent.sandbox.calls,
            "sandbox_restarts": agent.sandbox.restarts, "messages": len(history), "history": masked(history),
            "roles": [m.get("role") for m in history],
            "tools": [m["content"] for m in history if m.get("role") == "tool"][-2:],
            "requests": sum(not r.get("discarded_at_checkpoint") for r in agent.game.requests),
            "finished": agent.game.finished_reason,
        }
    digest = {"games": games, "status": report["status"], "end_reason": report["end_reason"],
              "task_errors": report["task_errors"], "checkpoint": report.get("checkpoint"),
              "requests": sum(r.get("error") != "discarded_at_checkpoint"
                              for r in json.loads((output / "t/requests.json").read_text())),
              "scheduler": masked(unordered(report["scheduler"]))}
    (output / "digest.json").write_text(json.dumps(digest, indent=1))
    (output / "histories.json").write_text(json.dumps(  # what a failed comparison is debugged from
        {a.game_id: [TIMING.sub(r"\1T", json.dumps(m, sort_keys=True, default=str)) for m in a._analyzer._history_messages]
         for a in trial.duck_workers}, indent=1))
    (output / "scheduler.json").write_text(TIMING.sub(r"\1T", json.dumps(unordered(report["scheduler"]), sort_keys=True, indent=1)))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))
