"""PRO-LONG persistent memory block (fixed-2x-memory): rewrite cadence, injection shape, token
accounting, survival across compaction, and the unchanged fixed-2x control payload."""

import copy
import itertools
import json

import numpy as np
from conftest import observation, tool_reply
from test_arena_duck import CAP, WINDOW, arena_agent, text_reply
from test_arena_fixed_2x import CONTROL

from arena.compaction import DUCK_SUMMARY_PROMPT, SUMMARY_PREAMBLE
from arena.duck import ArenaTransferDuckAgent
from arena.memory import MEMORY_COMPACTION_NOTE, MEMORY_LABEL, MEMORY_PROMPT, memory_message
from arena.variants import apply_variant

BLOCK_1 = "GOAL: reach the exit.\nVERIFIED MECHANICS: UP moves the cursor.\nPLAN: press UP."
BLOCK_2 = "GOAL: reach the exit.\nVERIFIED MECHANICS: UP moves the cursor.\nPLAN: press UP twice."


def more_observations(env, count):
    boards = []
    for i in range(count):
        board = np.zeros((64, 64), dtype=np.uint8)
        board[5, 5 + i] = 3
        boards.append(observation(board))
    env.observations = itertools.chain(env.observations, boards)


def test_memory_variant_is_one_change_on_the_fixed_2x_base():
    study = apply_variant(CONTROL, "fixed-2x-memory")
    assert study["memory_block"] == {"every_actions": 6, "request_cap_tokens": 2048}
    assert study["arena_variant"]["reference_difference"] == {
        "memory_block": {"before": None, "after": study["memory_block"]}}
    assert "cross_game_notes" not in study and study["budget_clock"] == "busy"
    assert memory_message(" x ") == {"role": "user", "content": MEMORY_LABEL + "x"}


async def test_block_is_rewritten_every_k_actions_and_leads_every_later_request(tmp_path):
    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            self.payloads.append(payload)
            n = len(self.payloads)
            if n in (2, 4):
                assert "tools" not in payload and payload["chat_template_kwargs"] == {"enable_thinking": False}
                assert payload["max_tokens"] == 2048 and payload["messages"][-1]["content"] == MEMORY_PROMPT
                if n == 2:
                    assert MEMORY_LABEL not in json.dumps(payload["messages"])
                    return text_reply(BLOCK_1, usage={"completion_tokens": 60, "prompt_tokens": 4000})
                assert payload["messages"][1] == memory_message(BLOCK_1)  # the rewrite sees its block
                return text_reply(BLOCK_2, usage={"completion_tokens": 70, "prompt_tokens": 6000})
            return tool_reply("action([{'action':'UP'}])", call_id=f"act{n}")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.attach(game, window=WINDOW, cap=CAP, memory={"every_actions": 1, "request_cap_tokens": 2048})
    agent.config.max_actions = 3
    more_observations(env, 2)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 3 and len(transport.payloads) == 5
    first, _, third, _, fifth = transport.payloads
    assert [m["role"] for m in first["messages"]] == ["system", "user"]  # no block before K actions
    assert third["messages"][1] == memory_message(BLOCK_1) and third["messages"][0] == first["messages"][0]
    assert [m["role"] for m in third["messages"]][:3] == ["system", "user", "user"] and "tools" in third
    assert fifth["messages"][1] == memory_message(BLOCK_2) and MEMORY_LABEL not in json.dumps(fifth["messages"][2:])
    stats = result["memory_block"]
    assert stats["rewrites"] == 2 and stats["failures"] == 0 and stats["seconds"] > 0
    assert stats["prompt_tokens"] == 10000 and stats["completion_tokens"] == 130
    assert stats["block_tokens"] > 0 and stats["block_chars"] == len(BLOCK_2)
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    purposes = ["gameplay", "memory", "gameplay", "memory", "gameplay"]
    assert [e["purpose"] for e in events if e["kind"] == "model"] == purposes
    assert [i["purpose"] for i in game.clock.inference_intervals] == purposes
    rewrites = [e for e in events if e["kind"] == "memory_rewrite"]
    assert [r["step"] for r in rewrites] == [1, 2] and rewrites[1]["block_chars"] == len(BLOCK_2)
    assert result["model_calls"] == 5 and gate.snapshot()["reserved"] == 0
    assert agent.carried_reserve() == 2176


async def test_no_rewrite_before_k_actions_keeps_the_control_payload(tmp_path):
    payloads = {}

    def transport_for(name):
        class Transport:
            async def post(self, url, **kwargs):
                payloads.setdefault(name, []).append(copy.deepcopy(kwargs["json"]))
                return tool_reply("action([{'action':'UP'}])", call_id="act")

        return Transport()

    control, env, game, gate = arena_agent(tmp_path / "control", transport_for("control"), ArenaTransferDuckAgent)
    await control.run()
    agent, env, game, gate = arena_agent(tmp_path / "memory", transport_for("memory"), ArenaTransferDuckAgent)
    agent.attach(game, window=WINDOW, cap=CAP, memory={"every_actions": 6, "request_cap_tokens": 2048})
    result = await agent.run()
    assert json.dumps(payloads["memory"]) == json.dumps(payloads["control"])
    assert result["memory_block"] == {"rewrites": 0, "failures": 0, "seconds": 0.0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "block_tokens": 0, "block_chars": 0}
    assert agent._analyzer._context_budget_tokens == control._analyzer._context_budget_tokens - 2176
    # Five actions since the last rewrite: still nothing to do (no request, same list).
    agent.steps, agent.memory_last_step = 5, 0
    messages = [{"role": "system", "content": "s"}]
    assert await agent.carried_messages(messages) is messages


async def test_block_survives_compaction_verbatim_and_the_summary_skips_it(tmp_path):
    block = "GOAL: reach the exit.\nREUSABLE CODE: saved_board() returns the level-1 board."

    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return tool_reply("saved_ascii = current_frame.ascii\n"
                                  "def saved_board():\n    return saved_ascii\nprint('probe')")
            if len(self.payloads) == 2:
                assert payload["tool_choice"] == "none"  # compaction shares the gameplay prefix
                assert payload["messages"][-1]["content"] == DUCK_SUMMARY_PROMPT + MEMORY_COMPACTION_NOTE
                assert payload["messages"][1] == memory_message(block)
                assert [m["role"] for m in payload["messages"]] == ["system", "user", "user", "assistant", "tool", "user"]
                return text_reply("World model: tiles. Plan: press UP.")
            return tool_reply("assert saved_board() == current_frame.ascii\naction([{'action':'UP'}])", call_id="act")

    transport = Transport()
    # A 20K window: the first tool round overflows the budget and compacts, as in test_arena_duck.
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    agent.attach(game, window=20000, cap=CAP, memory={"every_actions": 6, "request_cap_tokens": 2048})
    agent.memory_block = block  # written by an earlier rewrite
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and len(transport.payloads) == 3
    first, summary, second = transport.payloads
    assert [m["role"] for m in first["messages"]] == ["system", "user", "user"]
    assert first["messages"][1] == memory_message(block)
    roles = [m["role"] for m in second["messages"]]
    assert roles[:4] == ["system", "user", "user", "user"]
    assert second["messages"][1] == memory_message(block)  # verbatim, still first after the system prompt
    assert second["messages"][2]["content"].startswith(SUMMARY_PREAMBLE)
    assert "Plan: press UP." in second["messages"][2]["content"]
    assert second["messages"][3]["content"] == first["messages"][2]["content"]  # the observation, untouched
    assert game.compactions == 1 and result["arena_compactions"] == 1
    assert result["memory_block"]["rewrites"] == 0 and result["memory_block"]["block_chars"] == len(block)
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert [e["purpose"] for e in events if e["kind"] == "model"] == ["gameplay", "compaction", "gameplay"]
    assert [i["purpose"] for i in game.clock.inference_intervals] == ["gameplay", "summary", "gameplay"]


async def test_failed_or_empty_rewrite_keeps_the_previous_block(tmp_path):
    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(copy.deepcopy(kwargs["json"]))
            n = len(self.payloads)
            if n == 2:
                return text_reply("")  # an empty block
            if n == 4:
                import httpx
                raise httpx.ConnectError("serving hiccup")
            return tool_reply("action([{'action':'UP'}])", call_id=f"act{n}")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.attach(game, window=WINDOW, cap=CAP, memory={"every_actions": 1, "request_cap_tokens": 2048})
    agent.memory_block = "GOAL: keep me."
    agent.config.max_actions = 3
    more_observations(env, 2)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 3 and len(transport.payloads) == 5
    assert all(p["messages"][1] == memory_message("GOAL: keep me.") for p in transport.payloads)
    stats = result["memory_block"]
    assert stats["rewrites"] == 0 and stats["failures"] == 2 and stats["completion_tokens"] == 300
    assert agent.memory_last_step == 2 and gate.snapshot()["reserved"] == 0
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    errors = [e["error"] for e in events if e["kind"] == "memory_rewrite_failed"]
    assert errors[0] == "empty block" and errors[1].startswith("ConnectError")


async def test_trial_passes_the_memory_settings_to_its_duck_agents(tmp_path):
    from types import SimpleNamespace

    from conftest import ROOT
    from test_arena_core import Now, make_games

    from arena.gate import KVGate
    from arena.trial import Trial

    study = apply_variant(CONTROL, "fixed-2x-memory")
    trial = Trial(study, study["trials"][0], source=ROOT, games_dir=None, output=tmp_path / "out",
                  server_dir=tmp_path, base_url="http://model/v1")
    now = Now()
    wall, games = make_games(now, count=2)
    agents, rows, close = await trial.duck_agents(games, KVGate(1000, slots=2), 131072, wall, "card", SimpleNamespace())
    try:
        assert all(agent.memory == {"every_actions": 6, "request_cap_tokens": 2048} for agent in agents)
        assert trial.notes is None and all(agent.notes is None and game.on_park is None
                                           for agent, game in zip(agents, games, strict=True))
        assert agents[0].carried_reserve() == 2176 and agents[0].memory_block is None
    finally:
        for agent in agents:
            agent.executor.shutdown(wait=False)
        await close()
        trial.admission_log.close()
