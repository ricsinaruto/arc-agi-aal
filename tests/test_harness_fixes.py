"""hero-v2 ``harness_fixes`` (arena.harness_fixes): one test group per sub-fix, the component and the replay."""

import asyncio
import copy
import json
import sys
import time

import httpx
import pytest
from conftest import ROOT, observation, tool_reply
from test_arena_duck import CAP, arena_agent, text_reply

from arena import harness_fixes as hf
from arena.cache import BlockBudget
from arena.clock import ActiveClock, GlobalClock
from arena.compaction import SUMMARY_PREAMBLE
from arena.duck import ArenaTransferDuckAgent

sys.path.insert(0, str(ROOT / "scripts/arena_analysis"))

LAYOUT = {'schema': 1, 'num_blocks': 100, 'max_model_len': 262144, 'lookahead': 3,
          'costs': [{'tokens': 262144, 'blocks': 20, 'groups': [20]}]}


def fixed_agent(tmp_path, transport, fixes, *, window=20000, gated=True):
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, window=window)
    agent.apply_harness_fixes(fixes)
    if gated:
        gate.budget = BlockBudget(LAYOUT)
        gate.capacity = 90
    return agent, env, game, gate


def events_of(agent, kind=None):
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    return [e for e in events if kind is None or e["kind"] == kind]


def is_summary(payload):
    return payload.get("tool_choice") == "none"


def text_of(message):
    content = message.get("content")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


# --- settings and the hero-v2 component -------------------------------------------------------------------------

def test_settings_default_off_and_list_every_sub_fix():
    assert hf.settings(None) is None and hf.settings(False) is None
    full = hf.settings(True)
    assert full["fixes"] == list(hf.FIXES) and full["summary_room_tokens"] == 4096
    assert full["stuck_turns"] == 16 and full["stuck_no_change_actions"] == 5
    assert hf.settings(["prompt_text", "exact_trigger"])["fixes"] == ["exact_trigger", "prompt_text"]
    assert hf.settings({"fixes": ["inventory"], "inventory_max_tokens": 800})["inventory_max_tokens"] == 800
    for bad in (["nope"], [], {"fixes": ["inventory"], "bogus": 1}, {"summary_room_tokens": 0}, "yes"):
        with pytest.raises(ValueError):
            hf.settings(bad)


def test_hero_v2_component_records_the_sub_fixes(monkeypatch):
    from test_hero_v2 import hero, pretend_merged

    from arena import hero_v2

    pretend_merged(monkeypatch)
    study = hero_v2.compose(hero(), {"harness_fixes": True})
    assert study["harness_fixes"] == hf.settings(True)
    record = study["hero_v2"]["components"]["harness_fixes"]
    assert record["on"] and record["fixes"] == list(hf.FIXES) and "harness_fixes" in study["hero_v2"]["on"]
    assert study["arena_variant"]["reference_difference"]["harness_fixes"] == {"before": None,
                                                                             "after": hf.settings(True)}
    some = hero_v2.compose(hero(), {"harness_fixes": ["exact_trigger", "inventory"]})
    assert some["harness_fixes"]["fixes"] == ["exact_trigger", "inventory"]
    off = hero_v2.compose(hero())
    assert "harness_fixes" not in off and "harness_fixes" in off["hero_v2"]["off"]
    with pytest.raises(ValueError):
        hero_v2.compose(hero(), {"harness_fixes": ["bogus"]})
    with pytest.raises(ValueError):
        hero_v2.compose(hero(), {"harness_fixes": "on"})


def test_a_hero_v2_bundle_writes_the_block_into_study_json(tmp_path):
    from test_hero_v2 import READY, build_args, builder

    _, study = builder().build("hero-v2-fixes", output=tmp_path / "bundle", wall_seconds=7200,
                               hero_v2={**READY, "harness_fixes": True}, **build_args())
    written = json.loads((tmp_path / "bundle/configs/comparison.json").read_text())
    assert written["harness_fixes"] == study["harness_fixes"] == hf.settings(True)
    assert written["hero_v2"]["components"]["harness_fixes"]["fixes"] == list(hf.FIXES)


# --- 1. exact-count compaction trigger ---------------------------------------------------------------------------

class ScriptedTransport:
    """First reply: a probe tool call; summaries: a handoff; afterwards: one action."""

    def __init__(self):
        self.payloads = []

    async def post(self, url, **kwargs):
        payload = copy.deepcopy(kwargs["json"])
        self.payloads.append(payload)
        if is_summary(payload):
            return text_reply("World model: tiles. Plan: press UP.")
        if len([p for p in self.payloads if not is_summary(p)]) == 1:
            return tool_reply("def saved_board():\n    '''The saved board.'''\n    return 1\nprint('probe')")
        return tool_reply("action([{'action':'UP'}])", call_id="act")


def counter(rule):
    calls = []

    async def count(messages, deadline, **kwargs):
        calls.append(messages)
        return rule(messages)

    return count, calls


@pytest.mark.parametrize("fixes", [None, ["exact_trigger"]])
async def test_exact_trigger_ignores_an_estimate_overshoot(tmp_path, fixes):
    """The 20K window leaves a 3,104-token budget: the estimate of the first tool round overflows it (the control
    compacts), while the exact count (3,000) fits, so the exact trigger does not compact; the admission reuses the
    trigger's count instead of counting again."""
    transport = ScriptedTransport()
    agent, env, game, gate = fixed_agent(tmp_path, transport, fixes)
    count, calls = counter(lambda messages: 3000)
    agent.client.count_tokens = count
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    summaries = [p for p in transport.payloads if is_summary(p)]
    if fixes is None:
        assert len(summaries) == 1 and game.compactions == 1
        return
    assert not summaries and game.compactions == 0
    stats = result["harness_fixes"]
    assert stats["exact_trigger_counts"] >= 2 and stats["admission_count_reused"] == 2
    assert len(calls) == 2  # one count per distinct gameplay payload: trigger and admission share it
    assert not events_of(agent, "exact_trigger_fallback")


async def test_exact_trigger_compacts_on_the_exact_count_and_keeps_the_budget(tmp_path):
    transport = ScriptedTransport()
    agent, env, game, gate = fixed_agent(tmp_path, transport, ["exact_trigger"])

    def rule(messages):
        if text_of(messages[-1]).startswith("Your context is full."):
            return 3600  # the summary request
        if any(text_of(m).startswith(SUMMARY_PREAMBLE) for m in messages):
            return 1200  # after the compaction
        return 3105 if len(messages) > 2 else 1000  # one token over the 3,104 budget after the tool round

    agent.client.count_tokens, _ = counter(rule)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and game.compactions == 1
    assert agent._arena_analyzer._context_budget_tokens == 20000 - CAP - 512 == 3104
    gameplay = [p for p in transport.payloads if not is_summary(p)]
    assert text_of(gameplay[-1]["messages"][1]).startswith(SUMMARY_PREAMBLE)


async def test_exact_trigger_falls_back_to_the_estimate_when_the_count_fails(tmp_path):
    transport = ScriptedTransport()
    agent, env, game, gate = fixed_agent(tmp_path, transport, ["exact_trigger"])

    async def count(messages, deadline, **kwargs):
        return 3000

    async def broken(messages, tools):
        raise httpx.ConnectError("tokenizer down")

    agent.client.count_tokens = count
    agent.exact_gameplay_tokens = broken  # the trigger's count fails; the admission counts on its own
    result = await agent.run()
    assert result["error"] is None and result["harness_fixes"]["exact_trigger_fallbacks"] >= 1
    assert events_of(agent, "exact_trigger_fallback")


# --- 2. summary-room guard ---------------------------------------------------------------------------------------

async def test_summary_guard_summarises_up_to_the_latest_observation_when_the_context_overshot(tmp_path):
    transport = ScriptedTransport()
    agent, env, game, gate = fixed_agent(tmp_path, transport, ["exact_trigger", "summary_guard"])

    def rule(messages):
        if text_of(messages[-1]).startswith("Your context is full."):
            return 1500
        if any(text_of(m).startswith(SUMMARY_PREAMBLE) for m in messages):
            return 2000
        return 17000 if len(messages) > 2 else 1000  # a long reply overshot: 17,000 + 4,096 > the 20,000 window

    agent.client.count_tokens, _ = counter(rule)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and game.compactions == 1
    summary = [p for p in transport.payloads if is_summary(p)][0]
    roles = [m["role"] for m in summary["messages"]]
    assert roles == ["system", "user", "user"]  # system, the observation, the summary prompt: no tool round
    assert summary["max_tokens"] == CAP  # 18,500 of exact room: the full cap, never the mechanical handoff
    guard = events_of(agent, "compaction_guard")
    assert len(guard) == 1 and guard[0]["prompt_tokens"] == 17000 and guard[0]["source_tokens"] == 1000
    assert guard[0]["excluded_messages"] == 2
    assert not events_of(agent, "compaction_rejected")
    assert not [e for e in events_of(agent, "compaction") if e["mechanical_fallback"]]
    gameplay = [p for p in transport.payloads if not is_summary(p)]
    assert [m["role"] for m in gameplay[-1]["messages"]] == ["system", "user", "user", "assistant", "tool"]


async def test_summary_cap_comes_from_the_exact_room_not_the_estimate(tmp_path):
    """hero-v2 2 h: the estimate-based cap gave the tightest summaries 2.6K where the exact room was 7.7K."""
    for fixes, expected in ((None, 2048), (["summary_guard"], CAP)):
        transport = ScriptedTransport()
        agent, env, game, gate = fixed_agent(tmp_path / str(bool(fixes)), transport, fixes)

        async def count(messages, deadline, **kwargs):
            return 3000

        agent.client.count_tokens = count
        agent.estimate_prompt_tokens = lambda messages, tools: 18500  # a 1.1x estimate far above the exact count
        text, data, seconds, estimate = await agent.aside_request(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "Your context is full.",
            cap=CAP, floor=2048, purpose="summary", record_purpose="compaction")
        assert text and transport.payloads[-1]["max_tokens"] == expected
        agent.executor.shutdown()


# --- 3. post-compaction inventory --------------------------------------------------------------------------------

async def test_inventory_lists_live_helpers_and_receipts_after_the_summary(tmp_path):
    import numpy as np

    class Transport(ScriptedTransport):
        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            if not is_summary(payload) and not [p for p in self.payloads if not is_summary(p)]:
                self.payloads.append(payload)
                return tool_reply("def saved_board():\n    '''The saved board.'''\n    return 1\n"
                                  "action([{'action':'UP'}])", call_id="act1")
            return await super().post(url, **kwargs)

    transport = Transport()
    agent, env, game, gate = fixed_agent(tmp_path, transport, ["inventory"], window=20000, gated=False)
    agent.config.max_actions = 2
    first = np.zeros((64, 64), dtype=np.uint8)
    first[4, 9] = 9
    second = first.copy()
    second[5, 9] = 9
    env.observations = iter([observation(first), observation(second)])
    result = await agent.run()
    assert result["error"] is None and game.compactions >= 1
    gameplay = [p for p in transport.payloads if not is_summary(p)]
    summary_text = [text_of(p["messages"][1]) for p in gameplay if text_of(p["messages"][1]).startswith(
        SUMMARY_PREAMBLE)][0]
    assert "Plan: press UP." in summary_text and "[Harness inventory]" in summary_text
    assert "- saved_board() -- The saved board." in summary_text
    assert "Last actions (newest last):" in summary_text and "- step 1 UP:" in summary_text
    inventory = events_of(agent, "compaction_inventory")
    assert inventory and inventory[0]["live"] and inventory[0]["functions"] == 1
    assert result["harness_fixes"]["inventories"] == game.compactions


def test_inventory_render_respects_the_size_cap_and_keeps_the_newest():
    data = {"functions": [[f"helper_{i}", "(board, depth=3)", "Does a thing " * 5] for i in range(300)],
            "values": [[f"var_{i}", "list", 64, "[" + "1, " * 30 + "...]"] for i in range(300)],
            "classes": ["Node"], "modules": ["math", "collections"]}
    receipts = [f"- step {i} UP: changed 12 cells in [1, 2, 3, 4]" for i in range(12)]
    text = hf.render_inventory(data, receipts, max_tokens=1536, chars_per_token=3.5)
    assert len(text) <= 1536 * 3.5
    assert "helper_299(board, depth=3)" in text and "helper_0(" not in text
    assert "more, oldest first: helper_0" in text
    assert text.startswith("[Harness inventory]") and "Functions:" in text
    small = hf.render_inventory({"functions": [["f", "(x)", ""]], "values": [["n", "int", None, "3"]]},
                                ["- step 4 UP: no change"], max_tokens=1536, chars_per_token=3.5)
    assert "- f(x)" in small and "- n: int = 3" in small and "- step 4 UP: no change" in small and "more" not in small
    assert hf.render_inventory(None, [], max_tokens=1536, chars_per_token=3.5) == hf.FRESH_SESSION


def test_receipt_lines_name_the_action_and_whether_the_board_changed():
    from inference.agent.runtime_state import Frame, HistoryEntry

    def entry(action, step):
        return HistoryEntry(action=action, frame=Frame(grid=((0,),), step=step, level=1))

    history = [entry("", 0), entry("UP", 1), entry("LEFT", 2), entry("RESET", 3), entry("DOWN", 4)]
    results = {1: {"board_changed": True, "transition": {"changed_cells": 5, "change_bbox": [1, 2, 3, 4]}},
               2: {"board_changed": False}, 3: {"executed": True}, 4: {"level_completed": True, "board_changed": True}}
    assert hf.receipt_lines(history, results, 3) == [
        "- step 2 LEFT: no change", "- step 3 RESET: level restarted", "- step 4 DOWN: level completed"]
    assert hf.receipt_lines(history, results, 12)[0] == "- step 1 UP: changed 5 cells in [1, 2, 3, 4]"


# --- 4. cap-hit follow-up --------------------------------------------------------------------------------------

def length_reply():
    return httpx.Response(200, request=httpx.Request("POST", "http://local/v1/chat/completions"), json={
        "choices": [{"finish_reason": "length",
                     "message": {"role": "assistant", "content": None, "reasoning": "Long analysis of the lattice."}}],
        "usage": {"completion_tokens": CAP, "prompt_tokens": 2000}})


@pytest.mark.parametrize("fixes", [None, ["cap_followup"]])
async def test_a_reply_cut_at_the_cap_gets_the_cut_off_follow_up(tmp_path, fixes):
    class Transport:
        payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(copy.deepcopy(kwargs["json"]))
            if len(self.payloads) == 1:
                return length_reply()
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    transport = Transport()
    transport.payloads = []
    agent, env, game, gate = fixed_agent(tmp_path, transport, fixes, window=131072, gated=False)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    second = transport.payloads[1]["messages"]
    assert second[-2]["role"] == "assistant" and "Long analysis of the lattice." in json.dumps(second[-2])
    followup = second[-1]["content"]
    if fixes is None:
        assert followup.startswith("You have not acted yet. Investigate first.")
        return
    assert followup == hf.CAP_FOLLOWUP.format(cap=CAP) and "Investigate first" not in followup
    assert "cut off at the 16,384-token output limit" in followup
    assert transport.payloads[1]["max_tokens"] == CAP  # the cap is unchanged
    assert result["harness_fixes"]["cap_followups"] == 1 and events_of(agent, "cap_hit_followup")


async def test_a_tool_call_cut_at_the_cap_gets_a_note(tmp_path):
    """hero-v2 k095: the cap cut a tool call to an empty ``code``; the tool result now says why."""

    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(copy.deepcopy(kwargs["json"]))
            if len(self.payloads) == 1:
                reply = tool_reply("", content=None)
                data = reply.json()
                data["choices"][0]["finish_reason"] = "length"
                return httpx.Response(200, request=reply.request, json=data)
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    transport = Transport()
    agent, env, game, gate = fixed_agent(tmp_path, transport, ["cap_followup"], window=131072, gated=False)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    tool = json.loads(transport.payloads[1]["messages"][-1]["content"])
    assert "non-empty" in tool["error"] and tool["note"] == hf.CAP_TOOL_NOTE.format(cap=CAP)
    assert result["harness_fixes"]["cap_tool_notes"] == 1


# --- 5. stuck notice ---------------------------------------------------------------------------------------------

class FakeTime:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_stuck_notice_reports_active_seconds_and_level_tokens_not_parked_wall_time():
    from arc_infra.config import Config

    now = FakeTime()
    wall = GlobalClock(9000, now=now)
    clock = ActiveClock(wall, now=now)
    wall.release()
    clock.release()
    rule = hf.settings(True)
    tracker = hf.ActiveStuckTracker(Config.load(ROOT / "configs/duck_controls.yaml"), active_seconds=clock.active_seconds,
                                    completion_tokens=clock.completion_tokens, rule=rule)
    clock.charge(0, 5000)  # before the level start
    tracker.reset_level()
    now.t += 60
    clock.pause()
    now.t += 5000  # parked: must not count
    clock.resume()
    now.t += 40
    clock.charge(3000, 7000)
    for _ in range(15):
        tracker.note_turn()
    assert tracker.notice(2, []) is None  # 15 turns: not due (the recorded rule fired from 8)
    tracker.note_turn()
    text = tracker.notice(2, ["UP"])
    assert "16 turns" in text and "(100s of active play)" in text and "7,000 generated tokens" in text
    assert "RESET" not in text  # a turn-triggered notice does not push RESET
    for _ in range(7):
        tracker.note_turn()
        assert tracker.notice(2, []) is None
    tracker.note_turn()
    assert tracker.notice(2, []) is not None  # every 8 turns after the first
    assert tracker.summary() == {"notices": 2, "streak": 0, "turns": 2}


def test_stuck_notice_fires_on_a_no_change_streak_with_the_reset_hint():
    from arc_infra.config import Config

    tracker = hf.ActiveStuckTracker(Config.load(ROOT / "configs/duck_controls.yaml"), active_seconds=lambda: 0.0,
                                    completion_tokens=lambda: 0, rule=hf.settings(True))
    for _ in range(4):
        tracker.note_action(False)
    assert tracker.notice(1, []) is None
    tracker.note_action(False)
    text = tracker.notice(1, ["UP", "DOWN"])
    assert "5 consecutive actions produced no board change" in text and "action(['RESET'])" in text
    for _ in range(4):
        tracker.note_action(False)
        assert tracker.notice(1, []) is None
    tracker.note_action(False)
    assert "10 consecutive" in tracker.notice(1, [])
    tracker.note_action(True)
    for _ in range(5):
        tracker.note_action(False)
    assert "5 consecutive" in tracker.notice(1, [])


def test_the_agent_uses_the_active_tracker_only_with_the_fix(tmp_path):
    from duck_transfer.feedback import StuckTracker

    agent, env, game, gate = fixed_agent(tmp_path, None, ["stuck_notice"], gated=False)
    assert isinstance(agent.stuck, hf.ActiveStuckTracker)
    other, *_ = fixed_agent(tmp_path / "control", None, None, gated=False)
    assert type(other.stuck) is StuckTracker
    with pytest.raises(ValueError, match="stuck"):
        other.apply_harness_fixes(["stuck_notice"], replaces_stuck=True)
    agent.executor.shutdown()
    other.executor.shutdown()


# --- 6. sandbox --------------------------------------------------------------------------------------------------

@pytest.fixture
def fixed_sandbox():
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import StatefulSandbox, stateful_bootstrap

    original = load().run_sandboxed_python.__globals__
    events = []
    sandbox = StatefulSandbox(stateful_bootstrap(preserve_transition_results(original["_SANDBOX_BOOTSTRAP"])),
                              color_chars=original["ARC_COLOR_CHARS"],
                              on_event=lambda kind, **data: events.append((kind, data)))
    sandbox.configure_fixes(stateful_bootstrap(preserve_transition_results(original["_SANDBOX_BOOTSTRAP"]),
                                               fixes=("modules",)), interrupt=True, grace=3)
    sandbox.events = events
    yield sandbox
    sandbox.close()


STATE = {"current_frame": {"grid": [[0] * 64] * 64, "step": 3, "level": 1}, "history": [], "valid_actions": ["UP"],
         "last_action_result": {}}


def run_in(sandbox, code, timeout=5, handler=None):
    return sandbox.run(code=code, timeout_seconds=timeout, initial_state=STATE, action_handler=handler)


def test_timeout_interrupts_the_statement_and_keeps_python_state(fixed_sandbox):
    from duck_transfer.sandbox import INTERRUPTED_AFTER_TIMEOUT, RESET_AFTER_TIMEOUT

    sandbox = fixed_sandbox
    assert run_in(sandbox, "def helper(a):\n    return a + 1\nkept = [1, 2]\nprint(helper(1))")["stdout"] == "2\n"
    pid = sandbox.process.pid
    started = time.monotonic()
    timed_out = run_in(sandbox, "partial = 'set before the loop'\nprint('started')\nwhile True:\n    pass", timeout=1)
    assert time.monotonic() - started < 3
    assert timed_out["interrupted"] and INTERRUPTED_AFTER_TIMEOUT in timed_out["error"]
    assert RESET_AFTER_TIMEOUT not in timed_out["error"] and "ToolTimeout" in timed_out["error"]
    assert timed_out["stdout"] == "started\n"  # partial output is returned
    after = run_in(sandbox, "print(helper(41), kept, partial)")
    assert after["stdout"] == "42 [1, 2] set before the loop\n" and after["error"] == ""
    assert sandbox.process.pid == pid and sandbox.restarts == 0 and sandbox.interrupts == 1
    assert [kind for kind, _ in sandbox.events if kind not in ("duck_python_heap", "duck_python_rng")] == [
        "duck_python_interrupt", "duck_python_interrupted"]


def test_an_interrupt_during_actions_never_breaks_the_protocol(fixed_sandbox):
    sandbox = fixed_sandbox
    calls = []

    def handler(actions):
        calls.append(actions)
        time.sleep(0.01)
        return {"action_result": {"executed": True, "action_num": len(calls)}, "state": STATE}

    run_in(sandbox, "n = 0", handler=handler)
    timed_out = run_in(sandbox, "while True:\n    action(['UP'])\n    n += 1", timeout=1, handler=handler)
    assert timed_out["interrupted"] and len(timed_out["action_results"]) >= 1
    after = run_in(sandbox, "print(n >= 1)", handler=handler)
    assert after["stdout"] == "True\n" and sandbox.restarts == 0


def test_a_statement_that_ignores_the_interrupt_still_falls_back_to_the_reset(fixed_sandbox):
    from duck_transfer.sandbox import RESET_AFTER_TIMEOUT

    sandbox = fixed_sandbox
    sandbox.interrupt_grace = 1
    # One long C-level call does not return to the interpreter (where the interrupt is raised) within the grace.
    code = "while True:\n    try:\n        sum(range(10**10))\n    except BaseException:\n        pass"
    timed_out = run_in(sandbox, code, timeout=1)
    assert RESET_AFTER_TIMEOUT in timed_out["error"] and sandbox.restarts == 1 and sandbox.process is None


def test_harmless_modules_and_builtins_are_allowed_and_file_access_stays_closed(fixed_sandbox):
    sandbox = fixed_sandbox
    ok = run_in(sandbox, "\n".join([
        "import hashlib, textwrap, traceback, time, pickle, inspect",
        "from time import perf_counter",
        "def helper(a, b=2):",
        "    '''Adds.'''",
        "    return a + b",
        "print(hashlib.md5(b'x').hexdigest()[:6], textwrap.shorten('a b c d', 5), perf_counter() > 0)",
        "print(pickle.loads(pickle.dumps({'a': {1, 2}, 'b': (1, [2])})))",
        "print(inspect.signature(helper), inspect.getsource(helper).splitlines()[0])",
        "print('helper' in globals(), id(helper) > 0, sorted(vars())[:1] != [])",
    ]))
    assert ok["error"] == "", ok["error"]
    assert ok["stdout"].splitlines() == ["9dd4e4 [...] True", "{'a': {1, 2}, 'b': (1, [2])}",
                                         "(a, b=2) def helper(a, b=2):", "True True True"]
    for code, message in (("open('x', 'w')", "name 'open' is not defined"),
                          ("import os", "Module 'os' is not allowed in the sandbox. Importable modules:"),
                          ("import numpy", "not installed in the sandbox"),
                          ("import time\ntime.sleep(1)", "time.sleep is disabled"),
                          ("import pickle\npickle.loads(b\"cos\\nsystem\\n(S'ls'\\ntR.\")", "may not load os.system"),
                          ("import inspect\ninspect.currentframe()", "has no attribute 'currentframe'"),
                          ("import inspect\ninspect.getsource(len)", "only for functions defined in this")):
        result = run_in(sandbox, code)
        assert message in result["error"], (code, result["error"])


def test_inventory_request_lists_definitions_without_running_code(fixed_sandbox):
    sandbox = fixed_sandbox
    assert sandbox.inventory() is None  # no interpreter yet
    ran = run_in(sandbox, "import math\nfrom collections import namedtuple\nfrom functools import lru_cache\n"
                          "def bfs(start, goal):\n    '''Shortest path.\n    more'''\n    return []\n"
                          "@lru_cache(None)\ndef cost(n):\n    return n\n"
                          "Node = namedtuple('Node', 'r c')\nwalls = {(1, 2), (3, 4)}\nlabel = 'x' * 500")
    assert ran["error"] == ""
    data = sandbox.inventory()
    assert data["functions"] == [["bfs", "(start, goal)", "Shortest path."], ["cost", "(n)", ""]]
    assert data["classes"] == ["Node"] and data["modules"] == ["math", "namedtuple", "lru_cache"]
    values = {v[0]: v for v in data["values"]}
    assert values["walls"][1:3] == ["set", 2] and len(values["label"][3]) <= 80
    assert "current_frame" not in values and "history" not in values
    assert run_in(sandbox, "print(bfs(0, 1))")["stdout"] == "[]\n"  # the request loop continues


def test_the_default_bootstrap_is_unchanged_without_the_fixes():
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import stateful_bootstrap

    original = preserve_transition_results(load().run_sandboxed_python.__globals__["_SANDBOX_BOOTSTRAP"])
    plain = stateful_bootstrap(original)
    assert "ToolTimeout" not in plain and "SHIM_MODULES" not in plain and "_inventory" not in plain
    core = stateful_bootstrap(original, fixes=())
    assert "ToolTimeout" in core and "SHIM_MODULES" not in core
    assert "SHIM_MODULES" in stateful_bootstrap(original, fixes=("modules",))


# --- 7. prompt text ----------------------------------------------------------------------------------------------

async def test_prompt_text_fixes_the_system_prompt_claims(tmp_path):
    agent, env, game, gate = fixed_agent(tmp_path, None, ["prompt_text", "sandbox", "timeout_interrupt"], gated=False)
    prompt = agent.analyzer(asyncio.get_running_loop())._system_prompt
    assert "earlier turns keep only a note that their image was omitted" not in prompt
    assert "Earlier turns, their images and your earlier replies stay in the context verbatim" in prompt
    assert "The user turn shows the world model carried from your previous turn" not in prompt
    assert "when your previous reply's text had one, the world model you carried" in prompt
    assert "hashlib" in prompt and "no sleep" in prompt and "numpy is not installed" in prompt
    assert "a timeout resets the interpreter" not in prompt and "interrupted and its earlier definitions" in prompt
    control, *_ = fixed_agent(tmp_path / "control", None, None, gated=False)
    plain = control.analyzer(asyncio.get_running_loop())._system_prompt
    assert "earlier turns keep only a note that their image was omitted" in plain
    agent.executor.shutdown()
    control.executor.shutdown()


async def test_user_prompt_drops_the_empty_world_model_line_and_uses_one_step(tmp_path):
    from inference.agent.runtime_state import Frame

    for fixes in (None, ["prompt_text"]):
        agent, env, game, gate = fixed_agent(tmp_path / str(fixes), None, fixes, gated=False)
        agent.obs = observation()
        analyzer = agent.analyzer(asyncio.get_running_loop())
        frame = Frame(grid=tuple(tuple(0 for _ in range(8)) for _ in range(8)), step=333, level=2)
        text = analyzer._build_user_prompt(333, valid_actions=["UP"], current_frame=frame, history_entries=[],
                                           previous_step_summary=None)
        if fixes is None:
            assert "end of world model." in text and "Current state: step 334, level 2" in text
        else:
            assert "end of world model." not in text and "Current state: step 333, level 2" in text
            analyzer._summarized_knowledge["world_model"] = "tiles"
            carried = analyzer._build_user_prompt(333, valid_actions=["UP"], current_frame=frame, history_entries=[],
                                                  previous_step_summary=None)
            assert "Working world model carried from earlier turns:" in carried and "end of world model." in carried
        agent.executor.shutdown()


async def test_game_over_note_no_longer_points_at_missing_mechanics(tmp_path):
    for fixes, retained in ((None, False), (["prompt_text"], False), (["prompt_text"], True)):
        agent, env, game, gate = fixed_agent(tmp_path / f"{fixes}{retained}", None, fixes, gated=False)
        analyzer = agent.analyzer(asyncio.get_running_loop())
        if retained:
            analyzer._summarized_knowledge["world_model"] = "tiles"
        analyzer._last_step_summary = {"game_over": True, "end_action_num": 7, "executed_actions": ["UP"],
                                       "level_transition": False}
        analyzer._update_summarized_knowledge_from_step_summary()
        note = analyzer._summarized_knowledge["recent_findings"]
        assert note.startswith("GAME_OVER at step 7 after actions [UP]")
        if fixes is None or retained:
            assert "The mechanics above still hold" in note
        else:
            assert "above" not in note and "What you learned about this level's mechanics still holds" in note
        agent.executor.shutdown()


# --- replay on a recorded trial ----------------------------------------------------------------------------------

def test_replay_simulation_counts_compactions_from_the_growth():
    import harness_fixes_replay as replay

    prompts = [10_000, 20_000, 30_000, 44_000, 12_000, 25_000, 40_000, 43_500]
    comps = [(4, 45_000, 12_000)]  # the real run compacted before request 4 at a would-be 45,000
    assert replay.simulate(prompts, comps, 44_000) == 1  # the recorded trigger: one compaction
    # An exact trigger at 48,640 lets 45,000 through, so the growth continues and one compaction still follows.
    assert replay.simulate(prompts, comps, 48_640) == 1
    assert replay.simulate([10_000, 20_000], [], 48_640) == 0
    # Trailing compaction after the last request (the run stopped): counted when it crosses the threshold.
    assert replay.simulate([10_000, 40_000], [(2, 47_000, None)], 44_000) == 1
    assert replay.simulate([10_000, 40_000], [(2, 47_000, None)], 48_640) == 0
    row = {"turns": 20, "tokens": 10_000, "no_effect": [True] * 5 + [False] + [True] * 3, "cleared": False}
    rule = {"turns": 16, "interval": 8, "no_change_actions": 5}
    assert replay.notices_old(row) == 4 and replay.notices_new(row, rule) == 2


# --- opt-in: summary_headroom and inventory_imports (docs/validation-20260927/compaction-overshoot-fix.md) -------

# Run A's (and B's) recorded block: ``true`` must keep producing it byte for byte.
RECORDED_BLOCK = {"fixes": ["exact_trigger", "summary_guard", "inventory", "cap_followup", "stuck_notice", "sandbox",
                            "timeout_interrupt", "prompt_text"], "summary_room_tokens": 4096,
                  "inventory_max_tokens": 1536, "inventory_receipts": 12, "stuck_turns": 16, "stuck_notice_turns": 8,
                  "stuck_no_change_actions": 5, "interrupt_grace_seconds": 3.0}
HEADROOM = ["exact_trigger", "summary_guard", "summary_headroom"]


def test_opt_in_fixes_are_named_only_and_true_keeps_the_recorded_block():
    assert json.dumps(hf.settings(True)) == json.dumps(RECORDED_BLOCK)
    assert not set(hf.OPTIONAL_FIXES) & set(hf.settings(True)["fixes"])
    full = hf.settings([*hf.FIXES, *hf.OPTIONAL_FIXES])
    assert full["fixes"] == [*hf.FIXES, *hf.OPTIONAL_FIXES]
    assert set(full) == set(RECORDED_BLOCK)  # no new thresholds
    assert hf.settings(["summary_headroom"])["fixes"] == ["summary_headroom"]
    with pytest.raises(ValueError):
        hf.settings(["inventory_imports"])  # changes the inventory: needs it


class SequenceTransport:
    """Gameplay replies in order (``act`` acts, ``probe`` runs python without acting); summaries: a handoff."""

    def __init__(self, script):
        self.script, self.payloads = list(script), []

    async def post(self, url, **kwargs):
        payload = copy.deepcopy(kwargs["json"])
        self.payloads.append(payload)
        if is_summary(payload):
            return text_reply("World model: tiles. Plan: press UP.")
        kind = self.script.pop(0)
        n = len([p for p in self.payloads if not is_summary(p)])
        if kind == "act":
            return tool_reply("action([{'action':'UP'}])", call_id=f"act{n}")
        return tool_reply(f"print('probe {n}')", call_id=f"probe{n}")


def sized(table, *, post):
    """A /tokenize rule: gameplay lists by message count (``table``), a summary request = its source + 228 (the
    prefilled summary prompt beside the gameplay count in run A), a post-compaction list = ``post``."""
    def rule(messages):
        texts = [text_of(m) for m in messages]
        if any(t.startswith(SUMMARY_PREAMBLE) for t in texts):
            return post
        if texts[-1].startswith("Your context is full."):
            return table[len(messages) - 1] + 228
        return table[len(messages)]
    return rule


def overshoot_agent(tmp_path, fixes, script, table, post):
    import numpy as np

    transport = SequenceTransport(script)
    agent, env, game, gate = fixed_agent(tmp_path, transport, fixes, window=65536)
    agent.config.max_actions = 2
    board = np.zeros((64, 64), dtype=np.uint8)
    after = board.copy()
    after[5, 9] = 9
    env.observations = iter([observation(after), observation(board)])
    agent.client.count_tokens, _ = counter(sized(table, post=post))
    return agent, transport, game


# k038 (run A): a 15,130-token reply on a 48,081 prompt (budget 48,640) acted; the next turn's list is 63,766, the
# guard's source without the new observation 63,332, and the summary request (+228) leaves 1,976 < the 2,048 floor.
K038 = {2: 48_000, 4: 63_400, 5: 63_800}


@pytest.mark.parametrize("fixes", [["exact_trigger", "summary_guard"], HEADROOM])
async def test_a_long_reply_after_a_near_threshold_prompt_leaves_room_for_the_summary(tmp_path, fixes):
    agent, transport, game = overshoot_agent(tmp_path, fixes, ["act", "act"], K038, post=16_800)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2 and game.compactions == 1
    compaction = events_of(agent, "compaction")[0]
    summaries = [p for p in transport.payloads if is_summary(p)]
    gameplay = [p for p in transport.payloads if not is_summary(p)]
    if "summary_headroom" not in fixes:  # the recorded bug: the mechanical "summary unavailable" handoff
        assert compaction["mechanical_fallback"] and events_of(agent, "compaction_rejected")
        assert events_of(agent, "compaction_guard")[0]["source_tokens"] == 63_400
        return
    assert not compaction["mechanical_fallback"] and not events_of(agent, "compaction_rejected")
    assert not events_of(agent, "compaction_guard")  # the split decided before the guard
    # The summary covers everything before the long reply and gets the full cap (48,228 + 16,384 <= 65,536).
    assert [m["role"] for m in summaries[0]["messages"]] == ["system", "user", "user"]
    assert summaries[0]["max_tokens"] == CAP
    # The long reply, its tool result and the new observation stay verbatim after the summary, in order.
    after = gameplay[-1]["messages"]
    assert [m["role"] for m in after] == ["system", "user", "assistant", "tool", "user"]
    assert text_of(after[1]).startswith(SUMMARY_PREAMBLE)
    assert after[2]["tool_calls"][0]["id"] == "act1" and after[3]["tool_call_id"] == "act1"
    split = events_of(agent, "compaction_split")
    assert len(split) == 1 and split[0]["source_tokens"] == 48_000 and split[0]["summary_room"] == 17_536
    assert split[0]["kept_messages"] == 3 and result["harness_fixes"]["summary_splits"] == 1


async def test_a_long_mid_turn_reply_keeps_the_latest_observation_verbatim(tmp_path):
    """The long reply came after the latest observation (no action yet): the summary covers up to that reply and
    the kept tail starts at the observation, as in every compaction."""
    table = {2: 30_000, 4: 45_000, 5: 46_000, 7: 47_500, 9: 63_700}
    agent, transport, game = overshoot_agent(tmp_path, HEADROOM, ["act", "probe", "probe", "act"], table,
                                             post=20_000)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2 and game.compactions == 1
    summary = [p for p in transport.payloads if is_summary(p)][0]
    assert [m["role"] for m in summary["messages"]] == ["system", "user", "assistant", "tool", "user", "assistant",
                                                         "tool", "user"]  # up to the long reply, then the prompt
    after = [p for p in transport.payloads if not is_summary(p)][-1]["messages"]
    assert [m["role"] for m in after] == ["system", "user", "user", "assistant", "tool", "assistant", "tool"]
    split = events_of(agent, "compaction_split")[0]
    assert split["source_tokens"] == 47_500 and split["kept_messages"] == 5
    assert not [e for e in events_of(agent, "compaction") if e["mechanical_fallback"]]


def test_a_cap_hit_reply_is_kept_with_its_follow_up(tmp_path):
    """k050 (run A): a 16,384-token reply cut at the cap (no tool call) on a 48,195 prompt, then the follow-up: the
    whole list is 64,657 and without the follow-up 64,587; the cut before the reply leaves 17,341."""
    agent, env, game, gate = fixed_agent(tmp_path, None, HEADROOM, window=65536, gated=False)
    call = {"id": "c1", "type": "function", "function": {"name": "python", "arguments": "{}"}}
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "board"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "assistant", "content": None, "reasoning_content": "long " * 100},
                {"role": "user", "content": hf.CAP_FOLLOWUP.format(cap=CAP)}]
    sizes = {1: 4_000, 2: 5_000, 4: 48_195, 5: 64_587, 6: 64_657}
    agent.trigger_tokens = lambda msgs, tools: sizes[len(msgs)]
    agent.estimate_prompt_tokens = lambda msgs, tools: 400 * len(msgs)  # no analyzer in this unit test
    source, tail = agent.summary_split(messages, None, 64_657, 48_640)
    assert source == messages[:4] and tail == messages[4:]  # the reply and its follow-up, verbatim
    assert events_of(agent, "compaction_split")[0]["summary_room"] == 65_536 - 48_195
    # Room enough for the whole list: the ordinary compaction (None); without the sub-fix: None.
    assert agent.summary_split(messages, None, 50_000, 48_640) is None
    agent.apply_harness_fixes(["exact_trigger", "summary_guard"])
    assert agent.summary_split(messages, None, 64_657, 48_640) is None
    agent.executor.shutdown()


def test_a_tail_too_long_for_the_budget_leaves_the_decision_to_the_guard(tmp_path):
    agent, env, game, gate = fixed_agent(tmp_path, None, HEADROOM, window=32768, gated=False)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "board"},
                {"role": "assistant", "content": "long"}, {"role": "user", "content": "follow-up"}]
    sizes = {2: 15_000, 3: 31_500}
    agent.trigger_tokens = lambda msgs, tools: sizes[len(msgs)]
    agent.estimate_prompt_tokens = lambda msgs, tools: 400 * len(msgs)  # no analyzer in this unit test
    # The 16.6K tail + a summary + the system prompt cannot fit the 15,872-token budget of a 32K window.
    assert agent.summary_split(messages, None, 31_600, 32768 - CAP - 512) is None
    assert events_of(agent, "compaction_split_skipped")
    agent.executor.shutdown()


def test_replay_rules_on_the_k038_growth():
    """The replay's rules on a synthetic k038: a 15,130-token reply on a 48,081 prompt, then 555 tokens of tool
    result and observation; the recorded guard's exact source (63,332) calibrates the guard rule."""
    import harness_fixes_replay as replay

    requests = [{"prompt": 30_000, "completion": 1_000, "user_after": False},
                {"prompt": 48_081, "completion": 15_130, "user_after": True},
                {"prompt": 5_952, "completion": 136, "user_after": False}]
    comps = [(2, 63_766, 5_952, 63_332)]
    kwargs = {"threshold": 48_640, "window": 65_536, "room": 4096, "overhead": 228}
    guard = replay.simulate_rule(requests, comps, rule="guard", **kwargs)
    assert len(guard) == 1 and guard[0]["tight"] and guard[0]["rejected"] and guard[0]["summary_room"] == 1_976
    split = replay.simulate_rule(requests, comps, rule="headroom", **kwargs)
    assert not split[0]["rejected"] and split[0]["split"] and split[0]["source"] == 48_081
    assert split[0]["summary_room"] == 65_536 - 48_081 - 228 and split[0]["extra_kept"] == 15_130
    assert split[0]["post"] == 5_952 + 15_130
    # Room enough: no rule changes anything.
    calm = [{**r, "completion": 100} for r in requests]
    assert not replay.simulate_rule(calm, [(2, 50_000, 5_952, None)], rule="headroom", **kwargs)[0]["split"]


def test_inventory_imports_lists_modules_first_and_names_the_inspect_import():
    data = {"functions": [[f"helper_{i}", "(a, b, c=3)", "does a thing"] for i in range(10)],
            "values": [[f"v{i}", "list", 12, "[(1, 2), (3, 4), (5, 6), (7, 8), ...]"] for i in range(54)],
            "classes": ["deque", "Counter"], "modules": ["re", "collections"]}
    receipts = [f"- step {i} UP: changed 4 cells in [1, 2, 3, 4]" for i in range(12)]
    # Run A's k038 shape: 54 variables fill the cap and the recorded rendering drops the modules line.
    recorded = hf.render_inventory(data, receipts, max_tokens=1536, chars_per_token=2.87)
    assert "imported modules" not in recorded and "`inspect.getsource(name)`" in recorded
    text = hf.render_inventory(data, receipts, max_tokens=1536, chars_per_token=2.87, imports=True)
    assert len(text) <= 1536 * 2.87
    lines = text.splitlines()
    assert lines[0] == hf.INVENTORY_HEADER_IMPORTS and "`import inspect` first" in lines[0]
    assert lines[1] == "- classes: deque, Counter" and lines[2] == "- imported modules and callables: re, collections"
    assert lines[3] == hf.IMPORTS_NOTE and hf.RECEIPTS_TITLE_IMPORTS in lines
    assert "- helper_9(a, b, c=3) -- does a thing" in text and "- step 11 UP:" in text
    bare = hf.render_inventory({"functions": [["f", "(x)", ""]]}, [], max_tokens=1536, chars_per_token=3.5,
                               imports=True)
    assert hf.NO_IMPORTS_NOTE in bare
    assert hf.render_inventory(None, [], max_tokens=1536, chars_per_token=3.5, imports=True) == hf.FRESH_SESSION


# --- prompt audit fixes (9 October; docs/harness-arm.md 5.7) ---------------------------------------------------------

AUDIT = ("solvable_line", "death_turn_v2", "findings_keep", "tool_call_followup", "stuck_streak")


def test_audit_fixes_are_opt_in_one_by_one():
    assert set(AUDIT) <= set(hf.OPTIONAL_FIXES) and not set(AUDIT) & set(hf.settings(True)["fixes"])
    for name in AUDIT:
        assert hf.settings([name])["fixes"] == [name]


async def test_solvable_line_is_one_system_line(tmp_path):
    agent, *_ = fixed_agent(tmp_path, None, ["prompt_text", "solvable_line"], gated=False)
    prompt = agent.analyzer(asyncio.get_running_loop())._system_prompt
    assert prompt.count(hf.SOLVABLE_LINE) == 1 and hf.SOLVABLE_ANCHOR + hf.SOLVABLE_LINE in prompt
    control, *_ = fixed_agent(tmp_path / "control", None, ["prompt_text"], gated=False)
    plain = control.analyzer(asyncio.get_running_loop())._system_prompt
    assert "can be completed with the actions offered" not in plain
    assert prompt.replace(hf.SOLVABLE_LINE, "") == plain  # nothing else moves
    agent.executor.shutdown()
    control.executor.shutdown()


async def test_findings_keep_leaves_the_models_recent_findings_alone(tmp_path):
    death = {"game_over": True, "end_action_num": 7, "executed_actions": ["UP"], "level_transition": False}
    for fixes, own, expected in ((["prompt_text"], "the door opens on SPACE", None),
                                 (["prompt_text", "findings_keep"], "the door opens on SPACE", "the door opens on SPACE"),
                                 (["prompt_text", "findings_keep"], "GAME_OVER at step 3 after actions [UP]; old", ""),
                                 (["prompt_text", "findings_keep"], "", "")):
        agent, *_ = fixed_agent(tmp_path / f"{len(fixes)}{len(own)}", None, fixes, gated=False)
        analyzer = agent.analyzer(asyncio.get_running_loop())
        analyzer._summarized_knowledge.update(world_model="tiles", recent_findings=own, current_plan="go up")
        analyzer._last_step_summary = dict(death)
        analyzer._update_summarized_knowledge_from_step_summary()
        knowledge = analyzer._summarized_knowledge
        assert knowledge["world_model"] == "tiles" and knowledge["current_plan"] == ""
        if expected is None:
            assert knowledge["recent_findings"].startswith("GAME_OVER at step 7") and "avoid that sequence" in knowledge["recent_findings"]
        else:
            assert knowledge["recent_findings"] == expected
        agent.executor.shutdown()


def test_death_text_names_a_step_budget_and_stays_one_line_after_the_first():
    first = hf.death_text([{"actions": 40, "edge": 36, "sequence": "a"}], step=41, fatal="DOWN", level=2)
    assert first.startswith("GAME_OVER at step 41 after DOWN: the attempt on level 2 ended")
    assert "lasted 40 actions; cells in the border ring changed on 36 of them." in first
    assert "step-budget bar" in first and "how many actions one attempt allows" in first and "\n" not in first
    for old in ("BAR RULE", "Do NOT", "FAILED", "falsified", "avoid that sequence"):
        assert old not in first
    quiet = hf.death_text([{"actions": 3, "edge": 3, "sequence": "a"}], step=4, fatal="UP", level=1)
    assert "lasted 3 actions. Identify the cause" in quiet
    two = [{"actions": 40, "edge": 36, "sequence": "a"}, {"actions": 41, "edge": 30, "sequence": "b"}]
    later = hf.death_text(two, step=90, fatal="LEFT", level=2)
    assert later.startswith("GAME_OVER at step 90 after LEFT: death 2 on level 2")
    assert "Attempts lasted 40, 41 actions: the same length with different actions, so this is a step budget of about 41" in later
    assert "not a fatal action" in later and len(later) < 400
    same = hf.death_text([two[0], {"actions": 40, "edge": 36, "sequence": "a"}], step=90, fatal="LEFT", level=2)
    assert "the same actions ended the same way" in same and "step budget" not in same
    other = hf.death_text([two[0], {"actions": 7, "edge": 0, "sequence": "c"}], step=90, fatal="LEFT", level=2)
    assert "Attempts lasted 40, 7 actions; identify the cause" in other
    many = hf.death_text([{"actions": n, "edge": 0, "sequence": str(n)} for n in (5, 9, 14, 20, 30)], step=1, fatal="UP", level=3)
    assert "death 5 on level 3" in many and "lasted 9, 14, 20, 30 actions" in many


def test_attempts_and_deaths_are_counted_per_level():
    import numpy as np
    from types import SimpleNamespace

    from arc_infra.plans import Action

    def obs(board, levels=0, state="NOT_FINISHED"):
        return SimpleNamespace(board=board, levels=levels, state=state)

    base = np.zeros((16, 16), dtype=np.uint8)
    bar = base.copy()
    bar[0, :3] = 5      # a border cell changes
    inner = bar.copy()
    inner[8, 8] = 2     # an interior cell changes
    owner = SimpleNamespace(attempt=None, death_log=())
    note = lambda action, before: ArenaTransferDuckAgent.note_attempt(owner, action, before)  # noqa: E731
    owner.previous, owner.obs = obs(base), obs(bar)
    note(Action("ACTION1"), 0)
    owner.previous, owner.obs = obs(bar), obs(inner)
    note(Action("ACTION6", 3, 4), 0)
    assert owner.attempt == {"actions": 2, "edge": 1, "labels": ["ACTION1", "ACTION6@3,4"]}
    owner.previous, owner.obs = obs(inner), obs(inner, state="GAME_OVER")
    note(Action("ACTION2"), 0)
    assert [(d["actions"], d["edge"]) for d in owner.death_log] == [(3, 1)]
    owner.previous, owner.obs = obs(inner, state="GAME_OVER"), obs(base)
    note(Action("RESET"), 0)
    assert owner.attempt["actions"] == 0 and len(owner.death_log) == 1
    owner.previous, owner.obs = obs(base), obs(bar, levels=1)
    note(Action("ACTION1"), 0)  # the clearing action: a new level starts clean
    assert owner.attempt["actions"] == 0 and owner.death_log == []


def death_turn(fixes, deaths):
    from types import SimpleNamespace

    from arena import dfr_harness as dfr
    from arena.duck import ArenaDuckMixin

    owner = SimpleNamespace(
        _analyzer=None, dfr=dfr.settings(True), auto_reset={"step": 41, "actions": ["UP", "DOWN"], "reset_step": 42},
        dfr_gameover_shown=None, dfr_last_action=None, dfr_opener_step=None, dfr_transients=[], dfr_stats={"gameover_turns": 0},
        dfr_noop_stops=[], dfr_guard_stops=lambda: [], dfr_gameover_images=None, fix=lambda name: name in fixes,
        fix_stats={}, death_log=deaths, obs=SimpleNamespace(levels=1), loop_gameover_trace=lambda: [],
        dfr_gameover_visuals=lambda: [])
    text = "Executed actions: UP, DOWN.\nGAME_OVER at step 41 after actions [UP, DOWN]; the level was reset.\nCurrent state: step 42, level 2."
    return ArenaDuckMixin.dfr_rewrite(owner, text, None), owner


def test_death_turn_v2_replaces_the_stacked_texts():
    from arena import dfr_harness as dfr

    old, _ = death_turn((), [{"actions": 40, "edge": 36, "sequence": "a"}])
    assert dfr.GAME_OVER_DIAGNOSIS in old and "Do NOT re-submit" in old and "The game over occurred immediately after action 'DOWN'" in old
    new, owner = death_turn(("death_turn_v2",), [{"actions": 40, "edge": 36, "sequence": "a"}])
    lines = new.split("\n")
    assert len(lines) == 3 and lines[1].startswith("GAME_OVER at step 41 after DOWN: the attempt on level 2 ended")
    assert "BAR RULE" not in new and "Do NOT re-submit" not in new and owner.fix_stats == {"death_texts": 1}
    assert owner.dfr_stats["gameover_turns"] == 1 and len(new) < len(old) / 2
    again, _ = death_turn(("death_turn_v2",), [{"actions": 40, "edge": 36, "sequence": "a"},
                                               {"actions": 40, "edge": 36, "sequence": "b"}])
    assert again.split("\n")[1].startswith("GAME_OVER at step 41 after DOWN: death 2 on level 2") and len(again) < 500


async def test_tool_call_followup_keeps_the_demand_for_a_call_only(tmp_path):
    public = hf.PUBLIC_FOLLOWUP_PREFIX + "Then investigate ... then call `action(actions)` inside Python."
    for fixes in (["prompt_text"], ["prompt_text", "tool_call_followup"]):
        agent, *_ = fixed_agent(tmp_path / str(len(fixes)), None, fixes, gated=False)
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": public}]
        out = agent.before_trim(messages)
        if len(fixes) == 1:
            assert out[-1]["content"] == public
        else:
            assert out[-1]["content"] == hf.TOOL_CALL_FOLLOWUP and "action(" not in out[-1]["content"]
            assert "not acted" not in out[-1]["content"] and "tool call" in out[-1]["content"]
            assert messages[-1]["content"] == public  # the caller's list is not mutated
        agent.executor.shutdown()


def test_stuck_streak_drops_the_turn_counter_trigger():
    from arc_infra.config import Config

    now = FakeTime()
    wall = GlobalClock(9000, now=now)
    clock = ActiveClock(wall, now=now)
    wall.release()
    clock.release()
    cfg = Config.load(ROOT / "configs/duck_controls.yaml")
    plain = hf.ActiveStuckTracker(cfg, active_seconds=clock.active_seconds, completion_tokens=clock.completion_tokens,
                                  rule=hf.settings(True))
    fixed = hf.ActiveStuckTracker(cfg, active_seconds=clock.active_seconds, completion_tokens=clock.completion_tokens,
                                  rule=hf.settings([*hf.FIXES, "stuck_streak"]))
    for tracker in (plain, fixed):
        tracker.reset_level()
        for _ in range(16):
            tracker.note_turn()
    assert plain.due() == "turns" and fixed.due() is None
    for _ in range(5):
        fixed.note_action(False)
    assert fixed.due() == "streak"
