import asyncio
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_baselines.agent import Agent
from qwen_baselines.bridge import Bridge
from qwen_baselines.clock import Clock
from qwen_baselines.config import Config
from qwen_baselines.evidence import Archive
from qwen_baselines.model import Conversation


class Environment:
    def __init__(self, clear_at=10000):
        self.n = 0
        self.clear_at = clear_at
        self.observation_space = self.raw()

    def raw(self):
        return SimpleNamespace(
            frame=[np.full((64, 64), self.n % 16)],
            state=SimpleNamespace(name="NOT_FINISHED"),
            levels_completed=int(self.n >= self.clear_at),
            win_levels=8,
            available_actions=[1, 2, 3, 4, 6, 7],
        )

    def reset(self):
        self.n = 0
        return self.raw()

    def step(self, action, data):
        self.n += 1
        return self.raw()


async def bridge_at(tmp_path, clear_at=10000):
    clock = Clock(600)
    archive = Archive(tmp_path)
    bridge = Bridge(Environment(clear_at), archive, clock)
    await bridge.initialize()
    clock.release()
    return bridge


def calls(*actions, content=None):
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": "preserved reasoning",
        "tool_calls": [
            {
                "id": str(i),
                "type": "function",
                "function": {"name": "act", "arguments": json.dumps({"action": a})},
            }
            for i, a in enumerate(actions)
        ],
    }


def batch(*actions, content=None):
    """One act call carrying an ordered action list."""
    items = [a if isinstance(a, dict) else {"action": a} for a in actions]
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": "preserved reasoning",
        "tool_calls": [
            {
                "id": "0",
                "type": "function",
                "function": {"name": "act", "arguments": json.dumps({"actions": items})},
            }
        ],
    }


async def test_order_action7_boundaries_and_pairing(tmp_path):
    bridge = await bridge_at(tmp_path, clear_at=2)
    # Legacy loop: every act call executes until a boundary.
    agent = Agent("A", bridge, SimpleNamespace(config=Config(actions_per_turn=None)))
    await agent.execute(calls("ACTION7", "ACTION4", "ACTION3"))
    assert bridge.actions == 2
    group = agent.conversation.groups[-1]
    results = [m for m in group if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["0", "1", "2"]
    assert json.loads(results[-1]["content"])["skipped"]
    assert group[0]["reasoning_content"] == "preserved reasoning"
    assert bridge.archive.metadata()["level_completed"]
    await bridge.begin_turn()
    await bridge.step("RESET")
    assert bridge.archive.metadata()["attempt"] == 1
    with pytest.raises(RuntimeError, match="boundary"):
        await bridge.step("ACTION1")


async def test_single_action_per_turn(tmp_path):
    bridge = await bridge_at(tmp_path)
    agent = Agent("A", bridge, SimpleNamespace(config=Config(actions_per_turn=1)))
    assert "Only one action is executed per response" in agent.conversation.system["content"]
    await agent.execute(calls("ACTION1", "ACTION2", "ACTION3"))
    assert bridge.actions == 1
    group = agent.conversation.groups[-1]
    results = [json.loads(m["content"]) for m in group if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in group if m["role"] == "tool"] == ["0", "1", "2"]
    assert results[0]["executed"] == 1 and not results[0].get("skipped")
    assert all(
        r["skipped"] and not r["executed"] and "one action" in r["reason"] for r in results[1:]
    )
    assert group[0]["reasoning_content"] == "preserved reasoning"
    assert [m["role"] for m in group] == ["assistant", "tool", "tool", "tool", "user"]
    # A boundary keeps precedence over the per-turn limit.
    bridge = await bridge_at(tmp_path / "clear", clear_at=1)
    agent = Agent("A", bridge, SimpleNamespace(config=Config(actions_per_turn=1)))
    await agent.execute(calls("ACTION1", "ACTION2"))
    reason = json.loads(agent.conversation.groups[-1][2]["content"])["reason"]
    assert "boundary" in reason
    # Baseline B is unaffected by the act limit.
    assert (
        "act takes"
        not in Agent("B", bridge, SimpleNamespace(config=Config())).conversation.system["content"]
    )


async def test_batched_act_stops_at_boundary_and_cap(tmp_path):
    bridge = await bridge_at(tmp_path, clear_at=2)
    agent = Agent("A", bridge, SimpleNamespace(config=Config(actions_per_turn=5)))
    await agent.execute(
        batch("ACTION1", {"action": "ACTION6", "x": 3, "y": 4}, "ACTION2", "ACTION3")
    )
    assert bridge.actions == 2
    group = agent.conversation.groups[-1]
    result = json.loads(group[1]["content"])
    assert result["executed"] == 2 and "boundary" in result["stopped"]
    assert (
        result["results"][0]["actions_taken"] == 1 and not result["results"][0]["level_completed"]
    )
    assert result["results"][1]["level_completed"] and result["results"][1]["action"] == {
        "action": "ACTION6",
        "x": 3,
        "y": 4,
    }
    assert all(r["skipped"] and "boundary" in r["reason"] for r in result["results"][2:])
    assert "frames" not in result["results"][0]
    # Every executed action still yields its own observation image message.
    assert [m["role"] for m in group] == ["assistant", "tool", "user", "user"]
    # The response cap counts actions across act calls; later calls are skipped.
    bridge = await bridge_at(tmp_path / "cap")
    agent = Agent("A", bridge, SimpleNamespace(config=Config(actions_per_turn=3)))
    message = batch("ACTION1", "ACTION2", "ACTION3", "ACTION4")
    message["tool_calls"].append(
        {**message["tool_calls"][0], "id": "1"},
    )
    await agent.execute(message)
    assert bridge.actions == 3
    first, second = (json.loads(m["content"]) for m in agent.conversation.groups[-1][1:3])
    assert first["executed"] == 3 and "At most 3 actions" in first["stopped"]
    assert first["results"][3]["skipped"] and second["skipped"]
    # An invalid item stops the batch and is reported in place.
    bridge = await bridge_at(tmp_path / "bad")
    agent = Agent("A", bridge, SimpleNamespace(config=Config()))
    await agent.execute(batch("ACTION1", {"action": "ACTION6", "x": 99, "y": 0}, "ACTION2"))
    result = json.loads(agent.conversation.groups[-1][1]["content"])
    assert bridge.actions == 1 and "error" in result["results"][1]
    assert result["results"][2]["skipped"] and "failed" in result["stopped"]


async def test_memory_block_is_recorded_and_carried_when_missing(tmp_path):
    bridge = await bridge_at(tmp_path)
    agent = Agent("A", bridge, SimpleNamespace(config=Config()))
    await agent.execute(batch("ACTION1", content="Probing.\nMEMORY:\nclicks toggle tiles"))
    assert agent.memory == "clicks toggle tiles"
    observation = agent.conversation.groups[-1][-1]
    assert observation["content"][0]["text"].startswith("Observation: ")
    await agent.execute(batch("ACTION2", content="Forgot the block."))
    observation = agent.conversation.groups[-1][-1]
    assert observation["content"][0]["text"] == (
        "Memory from your previous turn:\nclicks toggle tiles"
    )
    kinds = [k for (k,) in bridge.archive.db.execute("SELECT kind FROM events")]
    assert kinds.count("memory") == 1 and kinds.count("memory_missing") == 1
    # Baseline B attaches only the settled frame image to its observation.
    b = Agent("B", bridge, SimpleNamespace(config=Config()))
    content = b.conversation.groups[0][0]["content"]
    assert sum(p["type"] == "image_url" for p in content) == 1


async def test_initial_not_action_and_validation(tmp_path):
    bridge = await bridge_at(tmp_path)
    assert bridge.actions == 0 and bridge.archive.metadata()["action_id"] is None
    for args in [
        ("ACTION6", -1, 2),
        ("ACTION6", 1.5, 2),
        ("ACTION1", 1, 2),
        ("ACTION5", None, None),
    ]:
        with pytest.raises(ValueError):
            await bridge.step(*args)
    await bridge.step("ACTION6", 63, 0)
    assert bridge.archive.metadata()["action"] == {"action": "ACTION6", "x": 63, "y": 0}


async def test_issued_action_survives_cancellation(tmp_path):
    bridge = await bridge_at(tmp_path)
    original = bridge.env.step

    def slow(*args):
        time.sleep(0.05)
        return original(*args)

    bridge.env.step = slow
    task = asyncio.create_task(bridge.step("ACTION1"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bridge.actions == 1 and bridge.archive.metadata()["executed"]


async def test_archive_eviction_reopen_and_incremental_storage(tmp_path):
    bridge = await bridge_at(tmp_path)
    trace = []
    bridge.archive.db.set_trace_callback(trace.append)
    for _ in range(40):
        await bridge.step("ACTION1")
        bridge.archive.frame()
    assert len(bridge.archive.cache) == 32
    assert bridge.archive.frame(0)[0, 0] == 0
    with pytest.raises(ValueError):
        bridge.archive.frame().setflags(write=True)
    assert not any("UPDATE frames" in sql or "DELETE FROM frames" in sql for sql in trace)
    bridge.archive.close()
    archive = Archive(tmp_path)
    assert archive.count() == 41
    assert archive.transition(-1)["before"]["grid"][0][0] == 7
    archive.close()


def test_clock_actual_600_seconds():
    now = [100.0]
    clock = Clock(600, now=lambda: now[0])
    clock.release()
    now[0] = 123
    assert clock.feedback() == {
        "solve_target_seconds": 600,
        "elapsed_seconds": 23,
        "remaining_seconds": 577,
    }
    now[0] = 800
    assert clock.remaining() == 0


async def test_history_cap_trims_below_the_window(tmp_path):
    class Layer:
        config = SimpleNamespace(context_window=1000, reserve_tokens=5, history_tokens=12)

        def payload(self, messages, tools):
            return messages

        def fresh(self, messages):
            return messages

        async def count(self, messages):
            return len(messages) * 3

    observation = {"role": "user", "content": "initial observation"}
    conv = Conversation("task", observation, Archive(tmp_path))
    conv.append([calls("ACTION1"), {"role": "tool", "tool_call_id": "0", "content": "receipt"}])
    conv.append([calls("ACTION2"), {"role": "tool", "tool_call_id": "0", "content": "receipt"}])
    messages = await conv.maintain(Layer(), [])
    # 12 tokens admit four messages: system, the newest exchange and the observation.
    assert len(messages) == 4 and messages[-1] is observation
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {
        "action": "ACTION2"
    }


async def test_context_removes_complete_pairs_retains_observation(tmp_path):
    class Layer:
        config = SimpleNamespace(context_window=15, reserve_tokens=5)

        def payload(self, messages, tools):
            return messages

        def fresh(self, messages):
            return messages

        async def count(self, messages):
            return len(messages) * 3

    observation = {"role": "user", "content": "initial observation"}
    conv = Conversation("task", observation, Archive(tmp_path))
    conv.append([calls("ACTION1"), {"role": "tool", "tool_call_id": "0", "content": "receipt"}])
    messages = await conv.maintain(Layer(), [])
    assert messages == [conv.system, observation]


async def test_animation_order_terminal_reset_and_unknown_outcome(tmp_path):
    bridge = await bridge_at(tmp_path)
    raw = bridge.env.raw()
    raw.frame = [np.ones((64, 64), dtype=np.uint8), np.full((64, 64), 2, dtype=np.uint8)]
    raw.state.name = "GAME_OVER"
    bridge.env.step = lambda *args: raw
    result = await bridge.step("ACTION7")
    assert result["frame_count"] == 2 and result["settled_frame_index"] == 1
    assert bridge.archive.observation()["frames"][0][0][0] == 1
    assert bridge.archive.observation()["grid"][0][0] == 2
    await bridge.begin_turn()
    assert bridge.archive.metadata()["reset"] and bridge.actions == 2
    raw.state.name = "WIN"
    await bridge.step("ACTION7")
    with pytest.raises(RuntimeError):
        await bridge.step("ACTION1")
    await bridge.begin_turn()
    with pytest.raises(RuntimeError, match="already complete"):
        await bridge.step("RESET")
    bridge2 = await bridge_at(tmp_path / "unknown")
    bridge2.env.step = lambda *args: None
    with pytest.raises(RuntimeError, match="unknown"):
        await bridge2.step("ACTION1")
    with pytest.raises(RuntimeError, match="replay prohibited"):
        await bridge2.begin_turn()


def test_reasoning_effort_is_a_template_supported_value():
    import pytest

    from qwen_baselines.clock import Clock
    from qwen_baselines.config import Config
    from qwen_baselines.model import RequestLayer

    layer = RequestLayer(Config(), Clock(600))
    body = layer.payload([{"role": "user", "content": "x"}])
    assert body["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": False,
        "reasoning_effort": "xhigh",
    }
    legacy = RequestLayer(Config(preserve_thinking=True), Clock(600))
    assert legacy.payload([{"role": "user", "content": "x"}])["chat_template_kwargs"][
        "preserve_thinking"
    ]
    assert Config(reasoning_effort="medium").reasoning_effort == "medium"
    with pytest.raises(ValueError, match="xhigh, medium or low"):
        Config(reasoning_effort="high")


async def test_tokenize_errors_include_server_detail():
    import httpx
    import pytest

    from qwen_baselines.clock import Clock
    from qwen_baselines.config import Config
    from qwen_baselines.model import RequestLayer

    layer = RequestLayer(Config(), Clock(600))
    await layer.close()
    layer.client = httpx.AsyncClient(
        base_url="http://127.0.0.1:1234/v1/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                400, json={"error": {"message": "Invalid image data URL", "type": "BadRequest"}}
            )
        ),
    )
    with pytest.raises(RuntimeError, match="400.*Invalid image data URL"):
        await layer.count(layer.payload([{"role": "user", "content": "x"}]))
    await layer.close()


async def test_bridge_prints_live_progress_lines(tmp_path, capsys):
    from qwen_baselines.bridge import Bridge
    from qwen_baselines.clock import Clock
    from qwen_baselines.evidence import Archive

    clock = Clock(600)
    bridge = Bridge(Environment(2), Archive(tmp_path / "game"), clock, label="A game-1")
    await bridge.initialize()
    clock.release()
    assert capsys.readouterr().out == ""  # the initial observation is not an action
    await bridge.step("ACTION6", 3, 4)
    await bridge.step("ACTION1")
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[A game-1] action 1 ACTION6(x=3,y=4) -> levels 0/")
    assert "elapsed" in out[0] and "***" not in out[0]
    assert out[1].startswith("[A game-1] action 2 ACTION1 -> levels 1/")
    assert out[1].endswith("*** LEVEL CLEARED")


class StaticEnvironment(Environment):
    """A board that never changes, so every action is a no-change repeat candidate."""

    def raw(self):
        raw = super().raw()
        raw.frame = [np.zeros((64, 64), dtype=np.uint8)]
        return raw


async def test_turn_cap_no_change_repeats_and_transition_metadata(tmp_path):
    clock = Clock(600)
    archive = Archive(tmp_path)
    bridge = Bridge(
        StaticEnvironment(), archive, clock, objects=True, actions_per_turn=2, reject_repeats=True
    )
    await bridge.initialize()
    clock.release()
    assert archive.metadata()["transition"]["note"] == "initial observation"
    await bridge.step("ACTION6", 3, 4)
    meta = archive.metadata()
    assert meta["transition"]["changed_cells"] == 0 and meta["transition"]["board_changed"] is False
    assert meta["transition"]["objects"]["counts"] == {"appeared": 0, "disappeared": 0, "moved": 0}
    with pytest.raises(ValueError, match="already produced no board change"):
        await bridge.step("ACTION6", 3, 4)
    await bridge.step("ACTION1")  # a different action from the same board is fine
    with pytest.raises(ValueError, match="At most 2 actions per response"):
        await bridge.step("ACTION2")
    await bridge.begin_turn()  # a new response resets the cap, not the repeat memory
    with pytest.raises(ValueError, match="ACTION1 already produced"):
        await bridge.step("ACTION1")
    await bridge.step("RESET")  # a reset clears the repeat memory (and is a boundary)
    await bridge.begin_turn()
    await bridge.step("ACTION1")
    assert archive.metadata()["transition"]["no_change_repeat"] is False
    kinds = [
        (json.loads(p)["reason"])
        for k, p in archive.db.execute("SELECT kind,payload FROM events")
        if k == "action_rejected"
    ]
    assert kinds == ["no_change_repeat", "turn_cap", "no_change_repeat"]
    plain = await bridge_at(tmp_path / "plain")  # defaults keep the old contract for P/R/S
    await plain.step("ACTION1")
    await plain.step("ACTION1")
    assert "transition" not in plain.archive.metadata()


async def test_changing_board_transition_reports_objects(tmp_path):
    bridge = await bridge_at(tmp_path)
    bridge.objects = True
    await bridge.step("ACTION1")  # the fake board goes from all-0 to all-1
    summary = bridge.archive.metadata()["transition"]
    assert summary["changed_cells"] == 4096 and summary["change_bbox"] == [0, 0, 64, 64]
    assert summary["objects"]["counts"] == {"appeared": 0, "disappeared": 0, "moved": 0}
    assert summary["objects"]["background_reshaped"]  # whole-board recolour
    assert bridge.archive.object_view().startswith("objects: 1 (by colour: 1x1)")
