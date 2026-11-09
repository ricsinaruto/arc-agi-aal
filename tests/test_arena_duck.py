"""Arena Duck adapters: parking at the gate, level reports, the public variant, compaction."""

import asyncio
import copy
import json

import httpx
import pytest
from conftest import ROOT, duck_agent_for, observation, tool_reply

from arena.clock import ActiveClock, GlobalClock
from arena.compaction import SUMMARY_PREAMBLE
from arena.duck import ArenaPublicDuckAgent, ArenaTransferDuckAgent
from arena.gate import GateSlot, KVGate
from arena.scheduler import GameRecord

WINDOW, CAP = 131072, 16384


def arena_agent(tmp_path, transport, agent_class, *, window=WINDOW, levels=1):
    agent, env = duck_agent_for(tmp_path, transport, levels=levels, agent_class=agent_class)
    cfg = agent.config
    cfg.max_model_len, cfg.max_output_tokens = window, CAP
    cfg.reasoning_effort, cfg.temperature = "xhigh", 0.7
    wall = GlobalClock(900)
    game = GameRecord("synthetic", 0, ActiveClock(wall))
    game.ready = asyncio.Event()
    gate = KVGate(int(291_900 * 0.9), slots=8)
    agent.attach(game, window=window, cap=CAP)
    agent.client.slots = GateSlot(gate, game, cap=CAP, window=window)
    wall.release()
    game.clock.release()
    return agent, env, game, gate


def text_reply(text, usage=None):
    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://local/v1/chat/completions"),
        json={
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": usage or {"completion_tokens": 300, "prompt_tokens": 5000},
        },
    )


async def test_transfer_agent_parks_at_the_gate_and_reports_levels(tmp_path):
    class Transport:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            payload = kwargs["json"]
            assert payload["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
            assert payload["max_tokens"] == CAP and "[RUNTIME PLANNING BUDGET" not in json.dumps(payload["messages"])
            return tool_reply("action([{'action':'UP'}])", call_id="act", usage={"completion_tokens": 50, "prompt_tokens": 7000})

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    game.clock.pause()  # parked before the first request
    task = asyncio.ensure_future(agent.run())
    await asyncio.sleep(0.3)
    assert transport.calls == 0 and not game.ready.is_set() or transport.calls == 0
    game.clock.resume()
    result = await task
    assert result["error"] is None and result["actions"] == 1 and transport.calls == 1
    assert game.ready.is_set() and game.levels == 1 and game.time_to_level[0]["level"] == 1
    assert game.time_to_level[0]["active_seconds"] >= 0 and game.time_to_level[0]["wall_seconds"] >= 0.3
    assert game.max_prompt_tokens == 7000 and game.requests and game.requests[0]["tokens"] > CAP
    assert gate.snapshot()["reserved"] == 0 and result["arena_compactions"] == 0
    assert result["agent_profile"] == "duck-transfer-arena"


async def test_public_agent_keeps_the_public_prompt_with_xhigh_cap_and_file_state(tmp_path):
    class Transport:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            payload = kwargs["json"]
            assert payload["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
            assert payload["max_tokens"] == CAP
            prompt = payload["messages"][0]["content"]
            assert "Python state persists" not in prompt  # the public system prompt, not v5's
            assert "persistent interpreter" not in payload["tools"][0]["function"]["description"]
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaPublicDuckAgent)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and transport.calls == 1
    assert result["agent_profile"] == "duck-public-arena" and game.levels == 1
    state = json.loads((tmp_path / "tool_runtime_state.json").read_text())
    assert len(state["history"]) == 2  # rewritten after the action, as the public loop does
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert not any(e["kind"] == "stuck_notice" for e in events)


@pytest.mark.parametrize('nex', [False, True])
@pytest.mark.parametrize('effort', ['high', 'none'])
async def test_compaction_replaces_the_prefix_with_a_model_written_summary(tmp_path, nex, effort):
    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])  # the analyzer extends its list in place
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return tool_reply(
                    "saved_ascii = current_frame.ascii\n"
                    "def saved_board():\n    return saved_ascii\n"
                    "print('probe')"
                )
            if len(self.payloads) == 2:
                # The compaction request shares the gameplay prefix (28 September): the first gameplay request's
                # template settings and tool list, with the tools rendered but not callable.
                first = self.payloads[0]
                assert payload["chat_template_kwargs"] == first["chat_template_kwargs"]
                assert payload["tools"] == first["tools"] and payload["tool_choice"] == "none"
                assert payload["messages"][:len(first["messages"])] == first["messages"]
                assert 2048 <= payload["max_tokens"] <= CAP
                assert payload["messages"][-1]["content"].startswith("Your context is full.")
                return text_reply("World model: tiles. Action model: UP moves. Plan: press UP.")
            return tool_reply(
                "assert saved_board() == history[0].frame.ascii == current_frame.ascii\n"
                "action([{'action':'UP'}])", call_id="act"
            )

    transport = Transport()
    # A 20K window leaves a 3,104-token budget: the first tool round overflows it.
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    if nex:
        from arc_infra.model_templates import NEX_MODEL, NEX_REVISION
        agent.config.model, agent.config.model_revision = NEX_MODEL, NEX_REVISION
        agent.config.reasoning_effort = effort
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and len(transport.payloads) == 3
    first, summary, second = transport.payloads
    if nex:
        assert first['chat_template_kwargs'] == second['chat_template_kwargs'] == {'reasoning_effort': effort, 'enable_thinking': effort != 'none'}
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert [m["role"] for m in summary["messages"]][:4] == ["system", "user", "assistant", "tool"]
    roles = [m["role"] for m in second["messages"]]
    assert roles[:3] == ["system", "user", "user"] and second["messages"][0] is not None
    assert second["messages"][1]["content"].startswith(SUMMARY_PREAMBLE)
    assert "Plan: press UP." in second["messages"][1]["content"]
    assert second["messages"][2]["content"] == first["messages"][1]["content"]  # the observation, untouched
    assert game.compactions == 1 and result["arena_compactions"] == 1 and result["context_compactions"] == 1
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    compaction = [e for e in events if e["kind"] == "compaction"]
    assert len(compaction) == 1 and compaction[0]["summary_chars"] > 0
    assert [e["purpose"] for e in events if e["kind"] == "model"] == ["gameplay", "compaction", "gameplay"]


@pytest.mark.parametrize("cancel_at_deadline", [False, True])
async def test_initial_compaction_deadline_timeout_drains_cleanly(tmp_path, cancel_at_deadline):
    class Transport:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            # real arena summary request: no gameplay request preceded it here, so no tool list to share
            assert kwargs["json"].get("tool_choice", "none") == "none"
            game.clock.global_clock.seconds = 0
            if cancel_at_deadline:
                task.cancel()
            raise httpx.ReadTimeout("expired summary")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    original_analyzer = agent.analyzer

    def with_history(loop):
        analyzer = original_analyzer(loop)
        analyzer._ensure_session(agent.state_path)
        analyzer._history_messages = [
            {"role": "user", "content": "old observation"},
            {"role": "assistant", "content": "old evidence " * 1000},
        ]
        return analyzer

    agent.analyzer = with_history
    task = asyncio.create_task(agent.run())
    result = await asyncio.wait_for(task, 3)
    assert result["status"] == "time_limit" and result["error"] is None
    assert result["arena_compactions"] == 0 and transport.calls == 1
    assert agent.worker is None and agent.chat_future is None
    assert gate.snapshot()["reserved"] == 0 and not env.calls
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert any(e["kind"] in {"worker_deadline_drained", "worker_request_cancelled"} for e in events)


def test_estimate_and_summary_prompts_are_wired(tmp_path):
    agent, env = duck_agent_for(tmp_path, None, agent_class=ArenaTransferDuckAgent)
    assert ROOT.joinpath("configs/duck_controls.yaml").is_file()
    assert observation().levels == 0
    agent.executor.shutdown()


async def test_exact_multimodal_count_happens_before_block_admission(tmp_path):
    from arena.cache import BlockBudget
    events = []

    class Transport:
        async def post(self, url, **kwargs):
            events.append('generation')
            assert [e for e in events if not e.startswith('request_')][:2] == ['count', 'admit']
            return tool_reply("action([{'action':'UP'}])", call_id='act',
                              usage={'completion_tokens': 50, 'prompt_tokens': 7000})

    agent, env, game, gate = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    layout = {'schema': 1, 'num_blocks': 100, 'max_model_len': 262144, 'lookahead': 3,
              'costs': [{'tokens': 262144, 'blocks': 20, 'groups': [20]}]}
    gate.budget = BlockBudget(layout)
    gate.capacity = 90
    gate.log = lambda e: events.append(e['event'])

    async def count(messages, deadline, **kwargs):
        events.append('count')
        assert kwargs['tools'] and kwargs['template_kwargs']['reasoning_effort'] == 'xhigh'
        assert 'image_url' in json.dumps(messages)
        return 7000

    agent.client.count_tokens = count
    result = await agent.run()
    assert result['error'] is None and result['actions'] == 1
    assert events == ['count', 'request_wait_started', 'admit', 'request_admitted',
                      'generation', 'release', 'request_context_closed']
    assert gate.reserved == 0


async def test_empty_compaction_summary_is_retried_then_replaced_by_a_mechanical_handoff(tmp_path):
    """80k-three v1: one empty summary (29 tokens, no content) ended a level-4 focus game."""
    from arena.compaction import SUMMARY_RETRY_SUFFIX

    class Transport:
        def __init__(self, empties):
            self.payloads, self.empties = [], empties

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return tool_reply("saved = current_frame.ascii\nprint('probe')")
            if payload.get("tool_choice") == "none":  # a summary request (gameplay prefix, tools not callable)
                if self.empties:
                    self.empties -= 1
                    return text_reply(None, usage={"completion_tokens": 29, "prompt_tokens": 5000})
                return text_reply("World model: tiles. Plan: press UP.")
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    # One empty reply: the retry carries the suffix and its text is used.
    transport = Transport(empties=1)
    agent, env, game, gate = arena_agent(tmp_path / "retry", transport, ArenaTransferDuckAgent, window=20000)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    summaries = [p for p in transport.payloads if p.get("tool_choice") == "none"]
    assert len(summaries) == 2 and summaries[1]["messages"][-1]["content"].endswith(SUMMARY_RETRY_SUFFIX)
    gameplay = [p for p in transport.payloads if p.get("tool_choice") != "none"]
    assert "Plan: press UP." in gameplay[-1]["messages"][1]["content"]
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    compaction = [e for e in events if e["kind"] == "compaction"]
    assert compaction[0]["attempts"] == 2 and compaction[0]["mechanical_fallback"] is False
    assert [e["attempt"] for e in events if e["kind"] == "compaction_empty"] == [1]
    assert game.compactions == 1
    # Two empty replies: the mechanical handoff keeps the game alive.
    transport = Transport(empties=2)
    agent, env, game, gate = arena_agent(tmp_path / "fallback", transport, ArenaTransferDuckAgent, window=20000)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    gameplay = [p for p in transport.payloads if p.get("tool_choice") != "none"]
    assert "summary unavailable" in gameplay[-1]["messages"][1]["content"]
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    compaction = [e for e in events if e["kind"] == "compaction"]
    assert compaction[0]["attempts"] == 2 and compaction[0]["mechanical_fallback"] is True
    assert [e["attempt"] for e in events if e["kind"] == "compaction_empty"] == [1, 2]


def test_critical_note_marks_failed_python_calls_only(tmp_path):
    class Transport:
        async def post(self, url, **kwargs):
            raise AssertionError("no request expected")

    agent, _, game, _ = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    agent.attach(game, window=WINDOW, cap=CAP, critical_errors=True)
    failed = agent.run_python_tool(lambda s, a: {"tool": "python", "error": "Traceback ...", "stdout": ""}, None, {})
    fine = agent.run_python_tool(lambda s, a: {"tool": "python", "returncode": 0, "stdout": "ok"}, None, {})
    assert failed["critical"].startswith("CRITICAL: the last step failed") and "critical" not in fine
    assert agent.result()["critical_notes"] == 1
    agent.attach(game, window=WINDOW, cap=CAP)
    assert "critical" not in agent.run_python_tool(lambda s, a: {"tool": "python", "error": "x"}, None, {})


async def test_no_think_sends_enable_thinking_false_and_the_variant_changes_only_that(tmp_path):
    from test_arena_gittins import CONTROL

    from arena.variants import apply_variant

    class Transport:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            payload = kwargs["json"]
            assert payload["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "xhigh"}
            return tool_reply("action([{'action':'UP'}])", call_id="act", usage={"completion_tokens": 50, "prompt_tokens": 7000})

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.config.enable_thinking = False
    result = await agent.run()
    assert result["error"] is None and transport.calls == 1
    study = apply_variant(CONTROL, "fixed-2x-no-think")
    control = apply_variant(CONTROL, "reclaim-fixed-2x")
    study.pop("arena_variant"), control.pop("arena_variant")
    assert {k for k in study.keys() | control.keys() if study.get(k) != control.get(k)} == {"enable_thinking"}
    assert study["enable_thinking"] is False


def test_adaptive_thinking_rule():
    from arena.adaptive import AdaptiveThinking, settings

    a = AdaptiveThinking({}, "xhigh")
    heavy, light = {"enable_thinking": True, "reasoning_effort": "xhigh"}, {"enable_thinking": False, "reasoning_effort": "xhigh"}
    assert a.decide() == heavy  # first request
    assert a.decide() == heavy  # no action since the previous request
    a.note_action(changed=True, level_completed=False, game_over=False)
    assert a.decide() == light  # a plan that works
    a.note_action(changed=True, level_completed=False, game_over=False)
    a.note_action(changed=False, level_completed=False, game_over=False)
    assert a.decide() == heavy  # surprise: an action changed nothing
    for _ in range(3):
        a.note_action(changed=True, level_completed=False, game_over=False)
        assert a.decide() == light
    a.note_action(changed=True, level_completed=False, game_over=False)
    assert a.decide() == heavy  # streak of 3 light requests
    a.note_action(changed=True, level_completed=True, game_over=False)
    assert a.decide() == heavy  # level change
    a.note_action(changed=True, level_completed=False, game_over=False)
    assert a.decide(compactions=1) == heavy  # compaction
    assert a.summary()["reasons"] == {"first": 1, "no_action": 1, "light": 4, "surprise": 2, "streak": 1, "compaction": 1}
    low = AdaptiveThinking({"light": "low"}, "xhigh")
    low.decide()
    low.note_action(changed=True, level_completed=False, game_over=False)
    assert low.decide() == {"enable_thinking": True, "reasoning_effort": "low"}
    with pytest.raises(ValueError):
        settings({"light": "off"})


async def test_adaptive_override_reaches_the_payload_and_the_logged_effort(tmp_path):
    from test_arena_gittins import CONTROL

    from arena.variants import apply_variant

    class Transport:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            payload = kwargs["json"]
            assert payload["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "xhigh"}
            return tool_reply("action([{'action':'UP'}])", call_id="act", usage={"completion_tokens": 50, "prompt_tokens": 7000})

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    from arena.adaptive import AdaptiveThinking

    agent.adaptive = AdaptiveThinking({}, "xhigh")
    agent.adaptive.requests, agent.adaptive.acted = 1, True  # as after a request whose action changed the board
    result = await agent.run()
    assert result["error"] is None and transport.calls == 1
    assert result["adaptive_thinking"]["modes"] == {"none": 1}
    logged = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()
              if json.loads(line).get("kind") == "model"]
    assert logged and logged[-1]["reasoning_effort"] == "none"
    study = apply_variant(CONTROL, "fixed-2x-adaptive-think")
    control = apply_variant(CONTROL, "reclaim-fixed-2x")
    study.pop("arena_variant"), control.pop("arena_variant")
    assert {k for k in study.keys() | control.keys() if study.get(k) != control.get(k)} == {"adaptive_thinking"}


async def test_worldmodel_notice_is_queued_by_a_level_clear(tmp_path):
    class Transport:
        async def post(self, url, **kwargs):
            return tool_reply("action([{'action':'UP'}])", call_id="act", usage={"completion_tokens": 50, "prompt_tokens": 7000})

    agent, env, game, gate = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent, levels=1)
    agent.worldmodel_notice = True
    result = await agent.run()
    assert result["error"] is None and game.levels == 1
    assert agent.worldmodel_pending == 1 and agent.turn_notice().startswith("LEVEL 1 CLEARED")
    assert agent.result()["worldmodel_notices"] == 1


async def test_a_refused_prefix_sharing_summary_falls_back_to_the_tool_less_request(tmp_path):
    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return tool_reply("saved = current_frame.ascii\nprint('probe')")
            if payload.get("tool_choice") == "none":
                return httpx.Response(400, request=httpx.Request("POST", "http://local/v1/chat/completions"),
                                      json={"error": "tool_choice none not supported"})
            if "tools" not in payload:
                assert payload["chat_template_kwargs"] == {"enable_thinking": False}
                return text_reply("World model: tiles. Plan: press UP.")
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=20000)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and game.compactions == 1
    assert [p.get("tool_choice") for p in transport.payloads][1] == "none" and "tools" not in transport.payloads[2]
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert [e["status"] for e in events if e["kind"] == "compaction_prefix_refused"] == [400]


@pytest.mark.parametrize("summary_count", [15000, 19000])
async def test_a_compaction_summary_over_the_window_is_fitted_or_mechanical_not_fatal(tmp_path, summary_count):
    """c110 hero-21x42k k035: a long reasoning reply plus its tool result overflowed the budget, the compaction
    summary took its cap from the estimate (``window - estimate``, without the rendered tools or its own prompt),
    the exact count plus that cap exceeded the window, and the admission's rejection, raised inside the
    trimmer, ended the game ("Duck analyzer could not recover the rejected request"). The summary's cap now
    fits the exact count (15,000: 5,000 tokens of room), and a summary that cannot fit even the 2,048-token
    floor (19,000: 1,000 of room) is replaced by the mechanical handoff."""
    from arena.cache import BlockBudget

    window = 20000

    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return tool_reply("saved = current_frame.ascii\nprint('probe')")
            if payload.get("tool_choice") == "none":
                return text_reply("World model: tiles. Plan: press UP.")
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=window)
    layout = {'schema': 1, 'num_blocks': 100, 'max_model_len': 262144, 'lookahead': 3,
              'costs': [{'tokens': 262144, 'blocks': 20, 'groups': [20]}]}
    gate.budget = BlockBudget(layout)
    gate.capacity = 90

    async def count(messages, deadline, **kwargs):
        summary = str(messages[-1].get("content") or "").startswith("Your context is full.")
        return summary_count if summary else 3000  # gameplay: 3,000 + the 16,384 cap fits

    agent.client.count_tokens = count
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and game.compactions == 1
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    summaries = [p for p in transport.payloads if p.get("tool_choice") == "none"]
    gameplay = [p for p in transport.payloads if p.get("tool_choice") != "none"]
    assert len(gameplay) == 2 and all(p["max_tokens"] == CAP for p in gameplay)  # gameplay requests unchanged
    assert gameplay[1]["messages"][1]["content"].startswith(SUMMARY_PREAMBLE)
    compaction = [e for e in events if e["kind"] == "compaction"]
    if summary_count == 15000:
        assert len(summaries) == 1 and summaries[0]["max_tokens"] == window - summary_count
        fitted = [e for e in events if e["kind"] == "aside_cap_fitted"]
        assert len(fitted) == 1 and fitted[0]["max_tokens"] == 5000 and fitted[0]["requested"] > 5000
        assert "Plan: press UP." in gameplay[1]["messages"][1]["content"]
        assert compaction[0]["mechanical_fallback"] is False
    else:
        assert not summaries  # rejected at admission, never sent
        assert [e["kind"] for e in events if e["kind"] in {"compaction_rejected", "compaction_empty"}] == [
            "compaction_rejected"]
        assert "summary unavailable" in gameplay[1]["messages"][1]["content"]
        assert compaction[0]["mechanical_fallback"] is True
    assert gate.reserved == 0


def test_count_key_snapshots_mutable_payload_and_ignores_mapping_order():
    from arena.duck import count_key

    payload = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "board ☀"}]}],
        "tools": [{"function": {"name": "python", "parameters": {"type": "object"}}}],
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"},
    }
    before = count_key(payload)
    cache = {before: 123}
    equivalent = copy.deepcopy(payload)
    equivalent["messages"][0] = dict(reversed(list(equivalent["messages"][0].items())))
    equivalent["chat_template_kwargs"] = dict(reversed(list(equivalent["chat_template_kwargs"].items())))
    equivalent.update(max_tokens=42, temperature=.7)
    assert cache[count_key(equivalent)] == 123
    payload["messages"][0]["content"][0]["text"] += " changed"
    assert count_key(payload) not in cache
    assert cache[count_key(equivalent)] == 123
    equivalent["tools"][0]["function"]["name"] = "other"
    assert count_key(equivalent) not in cache


@pytest.mark.parametrize("first,second", [(True, 1), (1, "1"), (1, 1.0), (0.0, -0.0), (None, "null")])
def test_count_key_distinguishes_json_scalar_encodings(first, second):
    from arena.duck import count_key

    assert count_key({"chat_template_kwargs": {"setting": first}}) != count_key(
        {"chat_template_kwargs": {"setting": second}})


@pytest.mark.parametrize("first,second", [
    ([["x"], "y"], ["x", ["y"]]), ([[], []], [[]]), ({"a": {"b": 1}}, {"a": "b", "c": 1}),
    ([], {}), ({}, None),
])
def test_count_key_keeps_container_boundaries(first, second):
    from arena.duck import count_key

    assert count_key({"messages": first}) != count_key({"messages": second})
