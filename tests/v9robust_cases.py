"""Fault-injection cases for v9robust.py, run by test_v9robust.py on a patched copy of the v9 bundle's src
(PYTHONPATH=<copy>/src). ROBUST=1: the patched tree; ROBUST=0: the unpatched control (a failure ends the game)."""

import asyncio
import json
import os

import httpx
import numpy as np
from conftest import FakeEnv, duck_agent_for, observation

from arena.clock import ActiveClock, GlobalClock
from arena.duck import ArenaTransferDuckAgent
from arena.gate import GateSlot, KVGate
from arena.scheduler import GameRecord

WINDOW, CAP = 131072, 16384
ROBUST = os.environ.get("ROBUST") == "1"


def reply(n):
    code = "action([{'action':'UP'}])"
    return httpx.Response(200, request=httpx.Request("POST", "http://local/v1/chat/completions"), json={
        "id": f"r{n}", "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": f"step {n}", "tool_calls": [{"id": f"c{n}", "type": "function", "function": {
                "name": "python", "arguments": json.dumps({"code": code})}}]}}],
        "usage": {"completion_tokens": 40, "prompt_tokens": 1000 + 50 * n}})


def play(tmp_path, fail, actions=4):
    """``fail(n)`` -> an exception to raise on chat request n (1-based), or None to answer."""
    sent = []

    class Transport:
        async def post(self, url, **kwargs):
            sent.append(1)
            exc = fail(len(sent))
            if exc is not None:
                raise exc
            return reply(len(sent))

    agent, _ = duck_agent_for(tmp_path, Transport(), agent_class=ArenaTransferDuckAgent)
    boards = [np.zeros((64, 64), dtype=np.uint8) for _ in range(actions + 1)]
    for k, b in enumerate(boards):
        b[k, k] = 3
    env = FakeEnv([observation(b, levels=0) for b in boards[:-1]] + [observation(boards[-1], levels=1, state="WIN")])
    agent.arc.make = lambda *a, **kw: env
    cfg = agent.config
    cfg.max_model_len, cfg.max_output_tokens, cfg.max_actions = WINDOW, CAP, 100
    wall = GlobalClock(900)
    game = GameRecord("k007", 7, ActiveClock(wall))
    game.ready = asyncio.Event()
    gate = KVGate(int(291_900 * 0.9), slots=8)
    agent.attach(game, window=WINDOW, cap=CAP)
    agent.client.slots = GateSlot(gate, game, cap=CAP, window=WINDOW)
    agent.deadline = __import__("time").monotonic() + 600
    if ROBUST:
        agent.REENTRY_SECONDS = 0.01
    wall.release()
    game.clock.release()
    return agent, env, sent


def events(agent, kind):
    path = agent.workspace.root / "log.jsonl"
    agent.workspace.flush() if hasattr(agent.workspace, "flush") else None
    return [json.loads(line) for line in path.open() if f'"kind":"{kind}"' in line] if path.exists() else []


async def test_engine_outage_mid_game(tmp_path):
    """Requests 2-4 fail (engine down); the game resumes and wins on the same env."""
    agent, env, sent = play(tmp_path, lambda n: httpx.ConnectError("engine down") if 2 <= n <= 4 else None)
    result = await agent.run()
    if not ROBUST:
        assert result["status"] == "error" and "ConnectError" in (result["error"] or "")
        assert len(env.calls) == 1  # the control ends at the first failure
        return
    assert result["error"] is None and result["status"] == "won", result
    assert len(env.calls) == 4 and len(sent) == 4 + 3
    got = events(agent, "game_reentry")
    assert [e["reentry"] for e in got] == [1, 2, 3] and all("ConnectError" in e["error"] for e in got)


async def test_reentry_is_bounded(tmp_path):
    """A permanent failure ends the game after REENTRY_MAX re-entries (no endless loop)."""
    agent, env, sent = play(tmp_path, lambda n: httpx.ConnectError("gone"))
    if ROBUST:
        agent.REENTRY_MAX = 5
    result = await agent.run()
    assert result["status"] == "error" and "ConnectError" in result["error"]
    assert len(sent) == (6 if ROBUST else 1) and not env.calls


async def test_reentry_waits_for_the_watchdog(tmp_path):
    """While the watchdog reports the engine down, the re-entered game waits; it resumes once healthy."""
    if not ROBUST:
        return
    from arena import engine_watchdog as wd

    agent, env, sent = play(tmp_path, lambda n: httpx.ConnectError("engine down") if n == 1 else None)
    wd.HEALTHY.clear()
    real_sleep = asyncio.sleep
    waits = []

    async def fast_sleep(s, *a):
        waits.append(s)
        if len(waits) == 4:
            wd.HEALTHY.set()
        await real_sleep(0)

    asyncio.sleep = fast_sleep
    try:
        result = await agent.run()
    finally:
        asyncio.sleep = real_sleep
        wd.HEALTHY.set()
    assert result["status"] == "won" and waits.count(5) >= 3  # polled the watchdog while it was down
