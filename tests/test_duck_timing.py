"""Competition-equivalent planning clock and its request-only feedback."""

import json
import time
from copy import deepcopy

import pytest
from conftest import duck_agent_for, tool_reply

from duck_transfer.timing import CompetitionClock


@pytest.mark.parametrize("sessions,elapsed,remaining", [(25, 900, 28440), (110, 3600, 28800)])
def test_competition_equivalent_budget(sessions, elapsed, remaining):
    clock = CompetitionClock(100, sessions)
    assert clock.snapshot(100)["competition_equivalent_remaining_seconds"] == 32400
    assert clock.snapshot(100 + elapsed)["competition_equivalent_remaining_seconds"] == remaining
    assert clock.snapshot(100 + 1e6)["competition_equivalent_remaining_seconds"] == 0
    assert clock.metadata()["elapsed_multiplier"] == 110 / sessions


def test_request_feedback_is_fresh_without_mutating_history():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "board"}}]},
        {"role": "assistant", "content": "inspect"},
        {"role": "tool", "content": "result", "tool_call_id": "x"},
    ]
    original = deepcopy(messages)
    clock = CompetitionClock(0, 25)
    early, late = clock.annotate(messages, 0), clock.annotate(messages, 900)
    assert messages == original
    assert len(late) == len(messages)
    assert early[1]["content"][0] == late[1]["content"][0] == messages[1]["content"][0]
    assert len(late[1]["content"]) == 2
    assert '"competition_equivalent_remaining_seconds": 28440.0' in late[1]["content"][-1]["text"]
    assert "32400.0" in json.dumps(early)
    with pytest.raises(ValueError, match="no user observation"):
        clock.annotate(messages[:1], 0)


async def test_slow_inspection_calls_keep_one_board_and_deadline(tmp_path, monkeypatch):
    payloads = []
    real_time = time

    class Clock:
        elapsed = time.monotonic()

        def monotonic(self):
            return self.elapsed

        def __getattr__(self, name):
            return getattr(real_time, name)

    clock = Clock()

    class Slots:
        async def __aenter__(self):
            clock.elapsed += 70  # Admission wait alone exceeded the public yield timer.

        async def __aexit__(self, *args):
            pass

    class Transport:
        async def post(self, url, **kwargs):
            payloads.append(deepcopy(kwargs["json"]))
            i = len(payloads)
            assert i <= 4
            assert kwargs["timeout"] <= 15  # Real budget was not extended to nine hours.
            return tool_reply(
                "print(current_frame.shape)" if i < 4
                else "print(action([{'action':'MOUSE','row':4,'col':9}]))",
                call_id=f"call-{i}", usage={"completion_tokens": 10},
            )

    agent, env = duck_agent_for(tmp_path, Transport())
    agent.client.slots = Slots()
    agent.configure_fleet_timing(clock.elapsed, 110)
    import duck_transfer.agent as bridge

    monkeypatch.setattr(bridge, "time", clock)
    monkeypatch.setattr(agent.module, "time", clock)
    # Cached import settings must not silently re-enable the old behavior.
    monkeypatch.setattr(agent.module, "_LOCAL_ANALYZER_YIELD_SECONDS", 60)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    assert len(payloads) == 4 and len(env.calls) == 1
    for i, payload in enumerate(payloads, 1):
        encoded = json.dumps(payload["messages"])
        assert encoded.count("RUNTIME PLANNING BUDGET") == 1
        assert f"{32400 - 70 * i}.0" in encoded  # Refreshed after admission.
        assert sum(
            p.get("type") == "image_url"
            for m in payload["messages"]
            if isinstance(m.get("content"), list)
            for p in m["content"]
        ) == 1
    transcript = (tmp_path / "duck-transcript.txt").read_text()
    assert transcript.count("| tool-agent ---") == 1


async def test_real_deadline_still_blocks_admitted_request(tmp_path):
    class Transport:
        async def post(self, *args, **kwargs):
            raise AssertionError("Expired run reached inference")

    agent, _ = duck_agent_for(tmp_path, Transport())
    agent.configure_fleet_timing(time.monotonic(), 110)
    agent.deadline = time.monotonic() - 1
    try:
        with pytest.raises(TimeoutError):
            await agent.chat([{"role": "user", "content": "board"}], [])
    finally:
        agent.executor.shutdown()
