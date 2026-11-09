"""hero-v2 ``drain_half`` (arena.context_drain): dfranzen's drop-half context drain in place of the summary compaction."""

import json
import threading

import httpx
import numpy as np
import pytest
from conftest import observation
from test_arena_duck import CAP
from test_harness_fixes import SequenceTransport, events_of, fixed_agent, is_summary, text_of

from arena import context_drain as cd
from arena.compaction import CompactionMixin

SPEC = cd.settings(True)


def msg(role, tokens, label=None, **extra):
    return {"role": role, "content": label or f"{role}-{tokens}", "tokens": tokens, **extra}


def sized(messages):
    """The synthetic estimate: every message carries its token size."""
    return sum(m.get("tokens", 0) for m in messages)


def blocks(n, *, user=1000, reply=3000, tool=1000):
    """``n`` turns of observation, assistant call and tool result."""
    out = []
    for i in range(n):
        out += [msg("user", user, f"obs{i}"), msg("assistant", reply, f"a{i}"), msg("tool", tool, f"t{i}")]
    return out


# --- settings and arithmetic ---------------------------------------------------------------------------------------

def test_settings_default_off_and_carry_dfranzens_values():
    assert cd.settings(None) is None and cd.settings(False) is None
    assert SPEC == {"drain_tokens": 59392, "turn_cap": 150, "turn_drain": 30, "coalesce": True,
                    "tool_output_tokens": 3072, "middle_truncation": True, "drop_malformed": True, "note": True}
    assert cd.settings({"turn_cap": 0, "note": False})["turn_cap"] == 0
    for bad in ("yes", {"bogus": 1}, {"coalesce": 1}, {"drain_tokens": -1}, {"turn_drain": 150},
                {"tool_output_tokens": 10}):
        with pytest.raises(ValueError):
            cd.settings(bad)


def test_the_drain_is_capped_at_half_the_budget():
    # The competition build: 130,816 - 16,384 - 512 = 113,920; 58K is above half of it.
    assert cd.drain_target(113_920, 59_392) == (56_960, 56_960)
    assert cd.drain_target(200_000, 59_392) == (59_392, 140_608)
    assert cd.drain_target(10_000, 0) == (0, 10_000)
    assert cd.drain_target(114_432, 85_824, 0.75) == (85_824, 28_608)
    assert cd.drain_target(114_432, 200_000, 0.75) == (85_824, 28_608)
    assert cd.settings({"drain_tokens": 85_824, "drain_cap_fraction": 0.75})["drain_cap_fraction"] == 0.75
    assert "drain_cap_fraction" not in cd.settings(True)
    with pytest.raises(ValueError):
        cd.settings({"drain_cap_fraction": 0.95})


# --- block boundaries ----------------------------------------------------------------------------------------------

def test_a_block_is_a_user_message_and_everything_up_to_the_next_one():
    history = [msg("user", 1, "obs0"), msg("assistant", 1), msg("tool", 1), msg("user", 1, "follow-up"),
               msg("assistant", 1), msg("user", 1, "obs1"), msg("assistant", 1)]
    assert cd.drop_oldest_block(history)
    assert [m["content"] for m in history][0] == "follow-up" and len(history) == 4
    assert cd.drop_oldest_block(history) and history[0]["content"] == "obs1"
    # The last user message is never dropped (the request would have no user query).
    assert not cd.drop_oldest_block(history) and len(history) == 2
    # An assistant head goes with its tool results only.
    history = [msg("assistant", 1), msg("tool", 1), msg("tool", 1), msg("user", 1, "obs")]
    assert cd.drop_oldest_block(history) and history == [msg("user", 1, "obs")]


def test_the_drain_drops_whole_blocks_to_the_target_and_keeps_the_latest_observation():
    history = [*blocks(12), msg("user", 1000, "obs-now"), msg("assistant", 3000, "a-now"), msg("tool", 1000)]
    count = 2000 + sized(history)  # system and tools: 2,000
    kept, info = cd.drain_history(history, count=count, budget=60_000, estimate=sized, spec=SPEC)
    assert count == 67_000 and info["trigger"] == "tokens" and info["target"] == 30_000
    # Whole blocks from the front: the kept history starts at an observation and is a suffix of the input.
    assert kept[0]["content"].startswith("obs") and kept == history[len(history) - len(kept):]
    assert 2000 + sized(kept) <= 30_000 < 2000 + sized(history[len(history) - len(kept) - 3:])
    assert kept[-3]["content"] == "obs-now" and info["blocks"] == info["messages_dropped"] // 3
    # Under the budget: nothing changes and the very same list comes back (append-only between drains).
    same, none = cd.drain_history(history, count=59_000, budget=60_000, estimate=sized, spec=SPEC)
    assert same is history and none is None


def test_the_current_turn_and_the_system_prompt_are_never_drained():
    turn = [msg("user", 500, "follow-up"), msg("assistant", 2000), msg("tool", 500)]
    history = [*blocks(4), msg("user", 1000, "Grid image obs-now"), msg("assistant", 40_000, "long"), msg("tool", 500),
               *turn]
    kept, info = cd.drain_history(history, count=2000 + sized(history), budget=60_000, estimate=sized, spec=SPEC,
                                  is_observation=lambda m: m["content"].startswith("Grid image"))
    # Only the four old blocks may go (the turn alone is above the target); the turn's observation stays even though
    # a later user message exists.
    assert kept[0]["content"] == "Grid image obs-now" and info["blocks"] == 4 and kept[-3:] == turn
    # Without the observation predicate the latest user message is the protected one (dfranzen's rule).
    kept, _ = cd.drain_history(history, count=2000 + sized(history), budget=60_000, estimate=sized, spec=SPEC)
    assert kept[0]["content"] == "follow-up"


def test_a_turn_too_long_for_the_budget_loses_its_oldest_exchange():
    history = [msg("user", 1000, "obs"), msg("assistant", 30_000, "a1"), msg("tool", 1000, "t1"),
               msg("assistant", 30_000, "a2"), msg("tool", 1000, "t2")]
    kept, info = cd.drain_history(history, count=2000 + sized(history), budget=60_000, estimate=sized, spec=SPEC)
    assert [m["content"] for m in kept] == ["obs", "a2", "t2"] and info["turn_exchanges"] == 1


def test_a_forced_drain_drains_to_the_target_even_when_the_count_fits():
    history = [*blocks(6), msg("user", 1000, "obs-now")]
    kept, info = cd.drain_history(history, count=50_000, budget=60_000, estimate=sized, spec=SPEC, force=True)
    assert info["trigger"] == "forced" and 50_000 - (sized(history) - sized(kept)) <= 30_000
    # A forced drain whose count is already under the target still drops one block (as the public retry does).
    kept, info = cd.drain_history(history, count=10_000, budget=60_000, estimate=sized, spec=SPEC, force=True)
    assert info["blocks"] == 1 and kept == history[3:]


# --- the turn cap ---------------------------------------------------------------------------------------------------

def test_the_turn_cap_fires_above_150_and_drains_to_120_at_a_user_message():
    history = [m for i in range(151) for m in (msg("user", 1, f"obs{i}"), msg("assistant", 1, f"a{i}"))]
    history.append(msg("user", 1, "obs-now"))
    kept, info = cd.drain_history(history, count=1000, budget=60_000, estimate=sized, spec=SPEC)
    assert info["trigger"] == "turns" and info["assistant_messages_kept"] == 120 and kept[0]["role"] == "user"
    assert kept[0]["content"] == "obs31" and kept[-1]["content"] == "obs-now"
    # At the cap (150) nothing happens: the cut is periodic, not continuous.
    at_cap = history[2:]
    same, none = cd.drain_history(at_cap, count=1000, budget=60_000, estimate=sized, spec=SPEC)
    assert same is at_cap and none is None
    assert cd.drain_history(history, count=1000, budget=60_000, estimate=sized,
                            spec={**SPEC, "turn_cap": 0})[1] is None


def test_coalesce_takes_the_deep_turn_cut_with_a_token_drain():
    history = [m for i in range(140) for m in (msg("user", 50, f"obs{i}"), msg("assistant", 350, f"a{i}"))]
    history.append(msg("user", 100, "obs-now"))
    count = 1000 + sized(history)  # 57,100 > 55,000: the default drain keeps ~26K = ~66 turns, below 120
    kept, info = cd.drain_history(history, count=count, budget=55_000, estimate=sized, spec=SPEC)
    assert info["blocks"] and info["turn_cap_messages"] == 0
    # A drain to just below the budget (small drain) keeps 129 turns: coalesce cuts to 120 in the same drain.
    spec = {**SPEC, "drain_tokens": 2000}
    kept, info = cd.drain_history(history, count=count, budget=55_000, estimate=sized, spec=spec)
    assert info["blocks"] and info["turn_cap_messages"] and info["assistant_messages_kept"] == 120
    kept, info = cd.drain_history(history, count=count, budget=55_000, estimate=sized, spec={**spec, "coalesce": False})
    assert info["assistant_messages_kept"] > 120 and not info["turn_cap_messages"]


# --- tool output and malformed calls --------------------------------------------------------------------------------

def test_middle_truncation_keeps_the_head_two_thirds_and_the_tail_third():
    text = "".join(f"{i:05d}" for i in range(4000))  # 20,000 characters
    out, cut = cd.middle_truncate(text, 12_288)
    head, marker_tail = out.split("\n... [truncated 7712 chars from the middle] ...\n")
    assert cut and head == text[:8192] and marker_tail == text[-4096:]
    assert cd.middle_truncate("short", 12_288) == ("short", False)


def test_malformed_calls_are_named():
    good = {"function": {"name": "python", "arguments": json.dumps({"code": "1"})}}
    bad = {"function": {"name": "python", "arguments": '{"code": "print(1'}}
    assert cd.malformed_calls({"tool_calls": [good]}) == []
    assert cd.malformed_calls({"tool_calls": [good, bad]})[0].startswith("python: invalid JSON arguments")


# --- the analyzer path ----------------------------------------------------------------------------------------------

def test_the_mixin_forces_the_drain_on_the_public_context_length_retry():
    calls = []

    class Owner:
        drain = SPEC

        def estimate_prompt_tokens(self, messages, tools):
            return 1

        def drain_context(self, analyzer, messages, tools, measure, budget, *, force=False):
            calls.append((budget, force))
            return messages

    class Analyzer(CompactionMixin):
        _context_budget_tokens = 50_000

    analyzer = Analyzer()
    analyzer._owner = Owner()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    analyzer._trim_messages_for_context(messages)
    analyzer._trim_messages_for_context(messages, extra_safety_tokens=512)  # the public loop's rejection retry
    analyzer._force_reduce_messages(messages)
    assert calls == [(50_000, False), (49_488, True), (49_488, True)]


def drain_agent(tmp_path, script, *, fixes=None, rule=None, actions=2, drain=True):
    transport = script if isinstance(script, SequenceTransport) else SequenceTransport(script)
    agent, env, game, gate = fixed_agent(tmp_path, transport, fixes, window=65536)
    agent.apply_context_drain(drain)
    agent.config.max_actions = actions
    board = np.zeros((64, 64), dtype=np.uint8)
    after = board.copy()
    after[5, 9] = 9
    env.observations = iter([observation(after), observation(board), observation(after)])
    counted = []

    async def count(messages, deadline, **kwargs):
        counted.append(messages)
        return rule(messages)

    agent.client.count_tokens = count
    hooks = []
    game.note_context_drain = lambda **kw: hooks.append({**kw, "thread": threading.current_thread().name})
    return agent, transport, game, gate, hooks, counted


def drained(messages):
    return any(text_of(m).startswith("[Context drain]") for m in messages)


COMPETITION_FIXES = ["exact_trigger", "summary_guard", "inventory", "cap_followup", "stuck_notice", "sandbox",
                     "timeout_interrupt", "prompt_text", "summary_headroom"]  # arena.campaign comp-prune25


@pytest.mark.parametrize("fixes", [["exact_trigger"], COMPETITION_FIXES])
async def test_a_drain_replaces_the_summary_and_calls_the_scheduler_hook(tmp_path, fixes):
    """64K window, 16,384 cap: budget 48,640, drain 24,320. The third request (second turn) counts 50,000 and drains:
    no summary request goes out, the first turn's block is dropped, the hook hears of it on the event loop, and the
    drained request reserves its own (smaller) count plus the cap at the gate."""
    def rule(messages):
        if drained(messages):
            return 9_000
        return 50_000 if len(messages) >= 5 else 3_000

    agent, transport, game, gate, hooks, _ = drain_agent(tmp_path, ["act", "act", "act"], fixes=fixes, rule=rule)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2
    assert not [p for p in transport.payloads if is_summary(p)] and game.compactions == 0
    first, second = transport.payloads
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "user"] and drained([second["messages"][1]])
    note = text_of(second["messages"][1])
    assert "python interpreter was not touched" in note
    assert ("inventory" in fixes) == bool(events_of(agent, "compaction_inventory"))
    assert second["messages"][2]["content"][0]["text"] != first["messages"][1]["content"][0]["text"]
    assert len(hooks) == 1 and hooks[0]["tokens_before"] == 50_000 and hooks[0]["tokens_after"] == 9_000
    assert hooks[0]["tokens_after"] < hooks[0]["tokens_before"] and hooks[0]["thread"] == threading.main_thread().name
    stats = result["context_drain"]
    assert stats["drains"] == 1 and stats["messages_dropped"] == 3 and stats["notes"] == 1
    assert stats["tokens_dropped"] == hooks[0]["tokens_before"] - hooks[0]["tokens_after"]
    event = events_of(agent, "context_drain")[0]
    assert event["trigger"] == "tokens" and event["blocks"] == 1 and event["tokens_after"] == hooks[0]["tokens_after"]
    # The KV gate admitted the drained request on its exact count plus the cap, and released everything.
    admission = events_of(agent, "exact_admission_count")[-1]
    assert admission["prompt_tokens"] == 9_000 and admission["output_allowance"] == CAP
    assert gate.snapshot()["reserved"] == 0


async def test_a_context_length_rejection_forces_the_drain_and_retries(tmp_path):
    """The estimate trigger lets the second turn through; the admission's exact count (60,000 + 16,384 > 65,536)
    rejects it, the public loop retries through the forced drain, and the retry goes out drained."""
    def rule(messages):
        if drained(messages):
            return 4_000
        return 60_000 if len(messages) >= 5 else 3_000

    agent, transport, game, gate, hooks, counted = drain_agent(tmp_path, ["act", "act"], rule=rule)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2 and len(transport.payloads) == 2
    assert len(counted) == 3 and len(counted[1]) == 5 and drained(counted[2])
    assert events_of(agent, "context_rejection") and events_of(agent, "context_drain")[0]["trigger"] == "forced"
    assert hooks[0]["tokens_before"] == 60_000 and hooks[0]["tokens_after"] < 60_000
    assert result["context_drain"]["forced"] == 1


class MalformedFirst(SequenceTransport):
    """The first reply is cut off at the cap inside its tool call (arguments not valid JSON); then the script."""

    async def post(self, url, **kwargs):
        if self.payloads:
            return await super().post(url, **kwargs)
        self.payloads.append(json.loads(json.dumps(kwargs["json"])))
        bad = {"id": "bad", "type": "function", "function": {"name": "python", "arguments": '{"code": "action(['}}
        return httpx.Response(200, request=httpx.Request("POST", "http://local/v1/chat/completions"), json={
            "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "World model: probe.",
                                                                "tool_calls": [bad]}}],
            "usage": {"completion_tokens": CAP, "prompt_tokens": 2000}})


async def test_a_malformed_call_never_reaches_history(tmp_path):
    transport = MalformedFirst(["act"])
    agent, _, game, gate, hooks, _ = drain_agent(tmp_path, transport, rule=lambda m: 3_000, actions=1)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1 and len(transport.payloads) == 2
    second = transport.payloads[1]["messages"]
    assert not any(c["id"] == "bad" for m in second for c in m.get("tool_calls") or [])
    assert second[-2]["role"] == "assistant" and "tool_calls" not in second[-2]
    assert text_of(second[-1]).startswith(cd.TRUNCATED_CALL_FOLLOWUP)
    assert result["context_drain"]["malformed_replies"] == 1 and events_of(agent, "malformed_tool_calls_dropped")


async def test_tool_outputs_follow_the_drain_settings_and_off_is_unchanged(tmp_path):
    def rule(messages):
        return 3_000

    for drain in (True, None):
        agent, transport, game, gate, hooks, _ = drain_agent(tmp_path / str(drain), ["act"], rule=rule, actions=1,
                                                             drain=drain)
        result = await agent.run()
        analyzer = agent._arena_analyzer
        prompt = transport.payloads[0]["messages"][0]["content"]
        if drain:
            assert analyzer._tool_output_tokens == 3072 and analyzer._tool_output_chars == 12_288
            assert "capped to about 3072 tokens" in prompt and "capped to about 1024 tokens" not in prompt
            assert analyzer._trim_tool_text("x" * 13_000)[0].count("from the middle") == 1
            assert result["context_drain"]["drains"] == 0
        else:
            assert analyzer._tool_output_tokens == 1024 and "capped to about 1024 tokens" in prompt
            assert "_trim_tool_text" not in analyzer.__dict__ and "context_drain" not in result
            assert agent.drain is None and not hooks


# --- the hero-v2 component -----------------------------------------------------------------------------------------

def test_hero_v2_component_records_the_drain_and_off_is_byte_identical(monkeypatch):
    from test_hero_v2 import hero, pretend_merged

    from arena import hero_v2
    from arena.campaign import components

    pretend_merged(monkeypatch)
    study = hero_v2.compose(hero(), {"drain_half": True})
    assert study["drain_half"] == SPEC and "drain_half" in study["hero_v2"]["on"]
    assert study["hero_v2"]["components"]["drain_half"]["drain_tokens"] == 59392
    assert study["arena_variant"]["reference_difference"]["drain_half"] == {"before": None, "after": SPEC}
    custom = hero_v2.compose(hero(), {"drain_half": {"note": False, "turn_cap": 0}})
    assert custom["drain_half"] == {**SPEC, "note": False, "turn_cap": 0}
    off = json.dumps(hero_v2.compose(hero()), sort_keys=True)
    assert "drain_half" not in off and json.dumps(hero_v2.compose(hero(), {"drain_half": False}), sort_keys=True) == off
    # The competition build's components (comp-prune25 + prune12 + int3 at 50 x 130,816) without the key: unchanged.
    comp = {**components("comp-prune25"), "expert_prune": "prune12", "expert_int3": 16, "focus_slots": 50,
            "density_blocks": 1700, "graph_capture": 56}
    base = json.dumps(hero_v2.compose(hero(), comp, served_blocks=1839), sort_keys=True)
    assert "drain_half" not in base
    assert json.dumps(hero_v2.compose(hero(), {**comp, "drain_half": None}, served_blocks=1839), sort_keys=True) == base
    on = hero_v2.compose(hero(), {**comp, "drain_half": True}, served_blocks=1839)
    assert on["drain_half"] == SPEC and on["hero_v2"]["layout"] == json.loads(base)["hero_v2"]["layout"]
    with pytest.raises(ValueError):
        hero_v2.compose(hero(), {"drain_half": "on"})


def test_a_hero_v2_bundle_writes_the_drain_into_study_json(tmp_path):
    from test_hero_v2 import READY, build_args, builder

    _, study = builder().build("hero-v2-drain", output=tmp_path / "bundle", wall_seconds=7200,
                               hero_v2={**READY, "drain_half": True}, **build_args())
    written = json.loads((tmp_path / "bundle/configs/comparison.json").read_text())
    assert written["drain_half"] == study["drain_half"] == SPEC


async def test_the_trial_passes_the_drain_to_its_duck_agents(tmp_path):
    from types import SimpleNamespace

    from conftest import ROOT
    from test_arena_core import Now, make_games
    from test_arena_fixed_2x import CONTROL

    from arena.gate import KVGate
    from arena.trial import Trial
    from arena.variants import apply_variant

    for drain in (SPEC, None):
        study = apply_variant(CONTROL, "fixed-2x-memory")
        if drain is not None:
            study["drain_half"] = drain
        out = tmp_path / str(bool(drain))
        trial = Trial(study, study["trials"][0], source=ROOT, games_dir=None, output=out, server_dir=tmp_path,
                      base_url="http://model/v1")
        wall, games = make_games(Now(), count=2)
        agents, rows, close = await trial.duck_agents(games, KVGate(1000, slots=2), 131072, wall, "card",
                                                      SimpleNamespace())
        try:
            assert all(agent.drain == drain for agent in agents)
            totals = trial.feature_totals([{"context_drain": agent.drain_stats} for agent in agents])
            assert (totals["context_drain"]["games"] == 2 and totals["context_drain"]["drains"] == 0) if drain \
                else totals == {}
        finally:
            for agent in agents:
                agent.executor.shutdown(wait=False)
            await close()
            trial.admission_log.close()


def test_an_under_counted_drain_drains_again_from_the_measure(tmp_path):
    """The linear estimate says twice what the measure says: the first pass stops early (the request is still over
    the budget), so the kept messages are drained again from the measure with the estimate rescaled by the measured
    drop, the note staying in front."""
    from test_arena_duck import arena_agent

    from arena.duck import ArenaTransferDuckAgent

    agent, _, game, _ = arena_agent(tmp_path, None, ArenaTransferDuckAgent, window=65536)
    agent.apply_context_drain({"drain_tokens": 1000})
    hooks = []
    game.note_context_drain = lambda **kw: hooks.append(kw)

    class Analyzer:
        @staticmethod
        def _estimate_request_input_tokens(messages, tools=None):
            return sized(messages) * 2

    def measure(messages, tools):
        return 2000 + sized(messages) + 50 * any(text_of(m).startswith("[Context drain]") for m in messages)

    history = [*blocks(20), msg("user", 1000, "obs-now")]
    messages = [{"role": "system", "content": "s"}, *history]
    out = agent.drain_context(Analyzer(), messages, None, measure, 60_000)
    assert measure(messages, None) == 103_000 and measure(out, None) <= 60_000
    assert text_of(out[1]).startswith("[Context drain]") and out[-1]["content"] == "obs-now"
    event = events_of(agent, "context_drain")[0]
    assert event["rounds"] == 2 and measure(out, None) > 50_000 and hooks == [{"tokens_before": 103_000, "tokens_after": measure(out, None)}]
    assert agent.drain_stats["drains"] == 1 and agent.drain_stats["messages_dropped"] == len(messages) - len(out) + 1
