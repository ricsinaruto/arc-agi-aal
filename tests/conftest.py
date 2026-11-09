"""Shared fixtures: synthetic observations, a scripted environment and a Duck agent factory."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np

from arc_infra.config import Config
from arc_infra.perception import Observation

ROOT = Path(__file__).resolve().parents[1]


def observation(board=None, levels=0, state="NOT_FINISHED"):
    board = np.zeros((64, 64), dtype=np.uint8) if board is None else board
    return Observation([board], levels, 5, state, ["ACTION1", "ACTION4", "ACTION6"])


def raw(obs):
    return SimpleNamespace(
        frame=obs.frames,
        levels_completed=obs.levels,
        win_levels=obs.total_levels,
        state=SimpleNamespace(name=obs.state),
        available_actions=[1, 4, 6],
        full_reset=False,
    )


class FakeEnv:
    def __init__(self, observations):
        self.observations = iter(observations)
        self.observation_space = raw(next(self.observations))
        self.calls = []
        self.reset_count = 0

    def step(self, action, data):
        self.calls.append((action.name, data))
        self.observation_space = raw(next(self.observations))
        return self.observation_space

    def reset(self):
        self.reset_count += 1
        following = next(self.observations, None)
        if following is not None:  # a scripted post-reset board, when the test provides one
            self.observation_space = raw(following)
        return self.observation_space


def duck_agent_for(tmp_path, transport, *, levels=0, agent_class=None, tokenize=None):
    """A duck-transfer agent over a scripted environment and a fake chat transport.

    ``tokenize(payload)`` returns the fake ``/tokenize`` count (default 100) or raises.
    """
    from duck_transfer.transfer import TransferDuckAgent

    class TokenizedTransport:
        base_url = httpx.URL("http://local/v1/")

        async def post(self, url, **kwargs):
            if str(url).endswith("/tokenize"):
                count = tokenize(kwargs.get("json") or {}) if tokenize else 100
                return httpx.Response(
                    200,
                    request=httpx.Request("POST", str(url)),
                    json={"count": count, "max_model_len": 65536},
                )
            return await transport.post(url, **kwargs)

    board = np.zeros((64, 64), dtype=np.uint8)
    after = board.copy()
    after[4, 9] = 9
    env = FakeEnv([observation(board), observation(after, levels=levels)])
    cfg = Config.load(ROOT / "configs/duck_controls.yaml")
    cfg.max_actions = 1
    from arc_infra.model import ModelClient

    # The real client (count_tokens over the fake transport), one admission slot.
    client = ModelClient(cfg)
    client.client = TokenizedTransport()
    client.slots = asyncio.Semaphore(1)
    agent = (agent_class or TransferDuckAgent)(
        "synthetic",
        SimpleNamespace(make=lambda *a, **kw: env),
        "card",
        client,
        cfg,
        tmp_path,
        ROOT / ".analysis-venv/bin/python",
        time.monotonic() + 15,
    )
    return agent, env


def tool_reply(code, *, call_id="call", content="World model: probe.", usage=None):
    """One assistant reply that calls the Duck python tool with ``code``."""
    import json

    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://local/v1/chat/completions"),
        json={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "python",
                                    "arguments": json.dumps({"code": code}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": usage or {"completion_tokens": 50, "prompt_tokens": 2000},
        },
    )
