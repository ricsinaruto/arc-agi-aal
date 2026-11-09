"""hero-v2 ``dfr_loop`` (arena.dfr_loop; v5): the drain system line, the 2,048-token yield with his stale opener, no
world model, and his action feedback; off keeps every payload, prompt and sandbox."""

import copy
import json

import httpx
import pytest
from conftest import observation, raw, tool_reply
from test_arena_duck import arena_agent
from test_dfr_harness import board, images, texts, user_turns

from arena import dfr_harness as dfr
from arena import dfr_loop as loop
from arena.duck import ArenaTransferDuckAgent

ALL = loop.settings(True)


def only(*on):
    return loop.settings({name: True for name in on})


def think_reply(tokens, call_id):
    """A reply with reasoning and no tool call."""
    return httpx.Response(200, request=httpx.Request("POST", "http://local/v1/chat/completions"), json={
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None,
                                                          "reasoning": "hmm " * 10}}],
        "usage": {"completion_tokens": tokens, "prompt_tokens": 7000}})


class Transport:
    """Replies: ``(code, completion tokens)``, or ``(None, tokens)`` for a reply without a tool call."""

    def __init__(self, *replies):
        self.replies, self.payloads = list(replies), []

    async def post(self, url, **kwargs):
        self.payloads.append(copy.deepcopy(kwargs["json"]))
        code, tokens = self.replies.pop(0) if self.replies else ("action(['UP'])", 50)
        if code is None:
            return think_reply(tokens, f"c{len(self.payloads)}")
        return tool_reply(code, call_id=f"c{len(self.payloads)}",
                          usage={"completion_tokens": tokens, "prompt_tokens": 7000})


async def run(tmp_path, boards, replies, block, *, dfr_block=True, drain=True, max_actions=None, states=None,
              levels=None):
    transport = Transport(*replies)
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    if drain:
        agent.apply_context_drain(True)
    agent.apply_dfr_harness(dfr_block)
    agent.apply_dfr_loop(block)
    obs = [observation(b, levels=(levels or {}).get(i, 0), state=(states or {}).get(i, "NOT_FINISHED"))
           for i, b in enumerate(boards)]
    env.observations = iter(obs[1:])
    env.observation_space = raw(obs[0])
    agent.config.max_actions = max_actions or len(boards) - 1
    result = await agent.run()
    return agent, transport, env, result


def tool_outputs(payload):
    return [json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool"]


def test_settings_schema():
    assert loop.settings(None) is None and loop.settings(False) is None and loop.settings({}) is None
    assert ALL == {"prompt_fix": True, "yield_tokens": 2048, "no_world_model": True, "action_feedback": True,
                   "inventory_animation": True}
    assert loop.settings({"yield_tokens": 512}) == {"prompt_fix": False, "yield_tokens": 512, "no_world_model": False,
                                                    "action_feedback": False, "inventory_animation": False}
    assert loop.settings({"yield_tokens": False}) is None
    with pytest.raises(ValueError, match="Unknown dfr_loop flags"):
        loop.settings({"memory": True})
    with pytest.raises(ValueError, match="booleans"):
        loop.settings({"prompt_fix": 1})
    with pytest.raises(ValueError):
        loop.settings({"yield_tokens": -1})


def test_hero_v2_records_the_component_only_when_on():
    from test_competition_run_b import RUN_B, SERVED, hero

    from arena import hero_v2

    off = hero_v2.compose(hero(), {**RUN_B}, served_blocks=SERVED)
    explicit = hero_v2.compose(hero(), {**RUN_B, "dfr_loop": False}, served_blocks=SERVED)
    assert json.dumps(off, sort_keys=True) == json.dumps(explicit, sort_keys=True) and "dfr_loop" not in off
    on = hero_v2.compose(hero(), {**RUN_B, "dfr_loop": True}, served_blocks=SERVED)
    assert on["dfr_loop"] == ALL and on["hero_v2"]["components"]["dfr_loop"]["on"]
    strip = {k: v for k, v in on.items() if k not in ("dfr_loop", "hero_v2", "arena_variant")}
    base = {k: v for k, v in off.items() if k not in ("hero_v2", "arena_variant")}
    assert json.dumps(strip, sort_keys=True) == json.dumps(base, sort_keys=True)
    assert hero_v2.status()["dfr_loop"]["available"]


async def test_off_keeps_the_payloads_prompts_and_sandbox(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    replies = [("print(1)", 1500), ("print(2)", 1500), ("action(['UP', 'UP'])", 50), ("action(['RIGHT'])", 50)]
    control, ct, _, cr = await run(tmp_path / "control", boards, replies, None)
    off, ot, _, orr = await run(tmp_path / "off", boards, replies, False)
    assert cr["error"] is None and orr["error"] is None and "dfr_loop" not in orr
    assert json.dumps(ct.payloads, sort_keys=True) == json.dumps(ot.payloads, sort_keys=True)
    assert control.sandbox.bootstrap == off.sandbox.bootstrap and off.dloop is None
    assert len(ct.payloads) == 3 and "Your last python snippet" not in json.dumps(ct.payloads)


async def test_prompt_fix_describes_the_drain(tmp_path):
    agent, *_ = await run(tmp_path, [board(), board((1, 1))], [], only("prompt_fix"))
    text = agent._analyzer._system_prompt
    assert loop.DRAIN_LINE in text and "context compaction replaces them" not in text
    bare, *_ = arena_agent(tmp_path / "nodrain", None, ArenaTransferDuckAgent)
    with pytest.raises(ValueError, match="drain_half"):
        bare.apply_dfr_loop(only("prompt_fix"))
    bare.executor.shutdown()


async def test_yield_after_2048_tokens_and_the_stale_opener(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    replies = [("print(1)", 1500), ("print(2)", 600),              # 2,100 tokens, nothing executed: yield
               ("action(['UP'])", 100),                            # the resumed turn acts
               ("print(3)", 2100),                                 # yield after one inspection
               ("action(['RIGHT'])", 50)]
    agent, transport, _, result = await run(tmp_path, boards, replies, {"yield_tokens": True})
    assert result["error"] is None and result["actions"] == 2
    stats = result["dfr_loop"]
    assert stats["yields"] == 2 and stats["resumes"] == 2 and stats["stale_openers"] == 1
    # Request 3 resumes: the first opener, two tool rounds, then a full new opener with the board image.
    third = user_turns(transport.payloads[2])
    assert len(third) == 2 and len(images(third[-1])) == 1
    assert texts(third[-1]).startswith("No previous sequence has been executed yet.")
    # Request 5 resumes after an executed turn: his stale lines replace the outcome block; same board, no diff image.
    opener = texts(user_turns(transport.payloads[4])[-1])
    assert opener.startswith("Your last python snippet executed NO actions (it only computed or inspected state - "
                             "that is fine).\nFor reference, the most recent executed sequence earlier this turn "
                             "was: UP.\nCurrent state: step 1, level 1.")
    assert "The code executed" not in opener and len(images(user_turns(transport.payloads[4])[-1])) == 1
    # Append-only between drains: every request extends the previous one.
    for before, after in zip(transport.payloads, transport.payloads[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]
    # A resumption is not a turn for the stuck notice.
    assert agent.stuck.turns == 2  # four invocations, two turns
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert sum(e.get("kind") == "dfr_yield" for e in events) == 2


async def test_a_yield_without_a_tool_call_says_so(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    replies = [("action(['UP'])", 100), (None, 2500), ("action(['RIGHT'])", 50)]
    agent, transport, _, result = await run(tmp_path, boards, replies, {"yield_tokens": True})
    assert result["error"] is None and result["dfr_loop"]["stale_no_tool_call"] == 1
    opener = texts(user_turns(transport.payloads[2])[-1])
    assert opener.startswith("Your previous exchange this turn produced no tool call and executed no actions.\n"
                             "For reference, the most recent executed sequence earlier this turn was: UP.")
    # No "You have not acted yet" follow-up: the budget ended the invocation first.
    assert "You have not acted yet" not in json.dumps(transport.payloads[2]["messages"])


async def test_stuck_turns_and_refusals_ignore_resumptions(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    replies = [("print(1)", 2100), ("print(2)", 2100), ("print(3)", 2100), ("action(['UP'])", 50),
               ("action(['RIGHT'])", 50)]
    agent, transport, _, result = await run(tmp_path, boards, replies, {"yield_tokens": True})
    assert result["error"] is None and result["dfr_loop"]["yields"] == 3
    control, ct, _, cr = await run(tmp_path / "c", boards, [("action(['UP'])", 50), ("action(['RIGHT'])", 50)], None)
    assert agent.stuck.turns == control.stuck.turns


def test_stale_lines_match_his_texts():
    assert loop.stale_lines({"stale": True, "executed_actions": ["A", "B"], "game_over": True,
                             "fatal_action": "B"})[1] == (
        "Reminder of the still-pending outcome from earlier this turn: the executed sequence (A, B) ended in GAME OVER "
        "after 'B'; the level was auto-reset and you are planning the fresh attempt now. Do NOT re-submit that "
        "sequence - the full diagnosis instructions were given in an earlier message this turn.")
    assert loop.stale_lines({"stale": True, "level_transition": True}) == [
        "Your last python snippet executed NO actions (it only computed or inspected state - that is fine).",
        "Reminder: the earlier sequence this turn completed the level; you are now on the new level."]


async def test_no_world_model(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    replies = [("action(['UP'])", 50), ("action(['RIGHT'])", 50)]
    agent, transport, _, result = await run(tmp_path, boards, replies, only("no_world_model"), levels={1: 1, 2: 1})
    assert result["error"] is None
    system = transport.payloads[0]["messages"][0]["content"]
    assert "BEFORE EXECUTING NEW ACTIONS" not in system and "Maintain a compact working world model" not in system
    assert "revise your approach from the result" in system and "as described by your world model" not in system
    second = texts(user_turns(transport.payloads[1])[-1])
    assert "Working world model" not in second and "end of world model" not in second
    assert dfr.LEVEL_START_USER_PROMPT in second and "Cross-level notes" not in second
    assert agent._analyzer._summarized_knowledge["cross_level_notes"] == ""
    assert agent._analyzer._summarized_knowledge["world_model"] == ""  # "World model: probe." was not parsed


async def test_no_world_model_on_the_v71p_stack(tmp_path):
    """Audit fixes + notes + drain: no block, no label sentence in the follow-up, no cross-level carry; notes stay."""
    from arena import harness_fixes as hf

    labels = "World model: walls are X.\nGoal model: reach G.\nCross-level notes: UP moves the block."

    class Labelled(Transport):
        async def post(self, url, **kwargs):
            self.payloads.append(copy.deepcopy(kwargs["json"]))
            code = self.replies.pop(0) if self.replies else "action(['UP'])"
            if code is None:
                return httpx.Response(200, request=httpx.Request("POST", "http://local/v1/chat/completions"), json={
                    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": labels}}],
                    "usage": {"completion_tokens": 50, "prompt_tokens": 7000}})
            return tool_reply(code, call_id=f"c{len(self.payloads)}", content=labels,
                              usage={"completion_tokens": 50, "prompt_tokens": 7000})

    transport = Labelled("note('rule', 'UP moves the block')\naction(['UP'])", None, "action(['RIGHT'])")
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_harness_fixes([*hf.FIXES, "solvable_line", "death_turn_v2", "findings_keep"])
    agent.apply_context_drain(True)
    agent.apply_dfr_harness(True)
    agent.apply_hx_harness({"toolbox": True, "memo": True})
    agent.apply_dfr_loop(loop.settings({"prompt_fix": True, "no_world_model": True, "inventory_animation": True}))
    obs = [observation(board()), observation(board((10, 10)), levels=1), observation(board((10, 10), (20, 20)), levels=1),
           observation(board((30, 30)), levels=1), observation(board((40, 40)), levels=1)]
    env.observations = iter(obs[1:])
    env.observation_space = raw(obs[0])
    agent.config.max_actions = 3
    result = await agent.run()
    assert result["error"] is None
    system = transport.payloads[0]["messages"][0]["content"]
    assert "BEFORE EXECUTING NEW ACTIONS" not in system and hf.SOLVABLE_LINE in system and "note(key, text)" in system
    turns = [texts(turn) for turn in user_turns(transport.payloads[-1])]
    everything = "\n".join(turns)
    for gone in ("Working world model", "end of world model", "Prior-level hypotheses", "Cross-level notes",
                 "World model:", "Goal model:"):
        assert gone not in everything, gone
    sent = [m["content"] for p in transport.payloads for m in p["messages"] if m["role"] == "user"]
    followup = next(t for t in sent if isinstance(t, str) and t.startswith(hf.PUBLIC_FOLLOWUP_PREFIX))
    assert "update lines" not in followup and "Call the `python` tool with code that inspects" in followup
    assert dfr.LEVEL_START_USER_PROMPT in turns[1] and "rule: UP moves the block" in turns[1]  # note() block stays
    assert all(value == "" for value in agent._analyzer._summarized_knowledge.values())


async def test_action_feedback(tmp_path):
    start, death, restart = board(), board((20, 20), (30, 30)), board()
    obs = [observation(start), observation(board((20, 20))), observation(death, state="GAME_OVER"),
           observation(restart), observation(board((40, 40))), observation(board((40, 40), (41, 41)))]
    code_one = ("r = action(['UP', 'RIGHT'])\nprint(sorted(r))\nprint(last_action_call_result is last_action_result,"
                " history[-1].result['automatic'])")
    code_two = "print(history[-1].action, history[-1].result['automatic'], history[-2].result['automatic'])\n" \
               "action(['UP'])"
    transport = Transport((code_one, 50), (code_two, 50))
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    agent.apply_context_drain(True)
    agent.apply_dfr_harness(True)
    agent.apply_dfr_loop(only("action_feedback"))
    env.observations = iter(obs[1:])
    env.observation_space = raw(obs[0])
    agent.config.max_actions = 5
    result = await agent.run()
    assert result["error"] is None
    out = tool_outputs(transport.payloads[1])[-1]["stdout"].splitlines()
    assert out[0] == ("[action] batch of 2 actions, 2 executed: UP, RIGHT | stop_reason: game_over")
    keys = eval(out[1])
    assert "reward" not in keys and "action_echo" not in keys
    assert {"action_trace", "fatal_action", "run_elapsed_seconds", "time_remaining_seconds",
            "gameplay_changed_per_action"} <= set(keys)
    assert out[2] == "True False"
    opener = texts(user_turns(transport.payloads[1])[-1])
    assert ("Per-action trace of the previous sequence: 1: UP -> effect; 2: RIGHT -> GAME OVER (attempt ended "
            "immediately after this action).") in opener
    assert opener.index("The game over occurred immediately after action") < opener.index("Per-action trace") \
        < opener.index("Do NOT re-submit")
    second = next(o["stdout"] for o in tool_outputs(transport.payloads[-1]) if o.get("stdout", "").startswith("RESET"))
    second = second.splitlines()
    assert second[0] == "RESET True False" and second[1] == "[action] UP -> effect"
    receipt = agent.transition_results[3]
    assert receipt["automatic"] is True and agent.transition_results[1]["automatic"] is False
    system = transport.payloads[0]["messages"][0]["content"]
    assert "- Flag semantics: `game_over` = this attempt FAILED" in system and "`last_action_result`" not in system
    assert "last_action_call_result" in system
    stats = result["dfr_loop"]
    assert stats["echoes"] >= 3 and stats["traces"] == 1 and stats["gameover_traces"] == 1


async def test_refusal_trace_and_inventory(tmp_path):
    boards = [board(), board(), board((5, 5)), board((5, 5), (6, 6))]
    replies = [("action(['UP'])", 50), ("r = action(['UP', 'RIGHT'])\nprint(r['action_trace'])", 50)]
    agent, transport, env, result = await run(tmp_path, boards, replies, only("action_feedback"), max_actions=3)
    assert result["error"] is None
    stdout = tool_outputs(transport.payloads[-1])[-1]["stdout"].splitlines()
    assert stdout[0] == "[action] batch of 2 actions, 1 executed: RIGHT"
    assert stdout[1] == str(["1: UP -> " + loop.BLOCKED_NOOP, "2: RIGHT -> effect"])
    inventory = agent.sandbox.inventory()
    assert "last_action_call_result" not in json.dumps(inventory)


async def test_trial_and_attach_wire_the_block(tmp_path):
    import inspect

    from arena import trial

    assert 'dfr_loop=self.study.get("dfr_loop") or None' in inspect.getsource(trial)
    agent, env, game, gate = arena_agent(tmp_path, None, ArenaTransferDuckAgent)
    agent.attach(game, window=agent.window, cap=agent.cap, dfr_harness=True, dfr_loop={"action_feedback": True})
    assert agent.dloop == only("action_feedback") and "_v5_inner_action" in agent.sandbox.bootstrap
    assert agent.sandbox.bootstrap.index("_DFR_FLAGS = ") < agent.sandbox.bootstrap.index("_v5_inner_action")
    agent.executor.shutdown()
    from arena.duck import ArenaPublicDuckAgent

    public, *_ = arena_agent(tmp_path / "public", None, ArenaPublicDuckAgent)
    with pytest.raises(ValueError, match="corrected Duck agent"):
        public.apply_dfr_loop(True)
    public.executor.shutdown()


def test_turn_trace_restitches_calls():
    calls = [{"executed": True, "action_trace": ["1: UP -> effect", "2: UP -> effect"]},
             {"executed": True, "executed_actions": ["LEFT"], "game_over": True, "action_display": "LEFT"}]
    assert loop.summary_extras(calls) == {
        "game_over_trace": ["1: UP -> effect", "2: UP -> effect",
                            "3: LEFT -> GAME OVER (attempt ended immediately after this action)"],
        "fatal_action": "LEFT"}
    assert loop.summary_extras(calls[:1]) == {"game_over_trace": [], "fatal_action": ""}


async def test_inventory_answers_a_lazy_animation_view(tmp_path):
    """v3/v4: a variable bound to ``last_animation_timeline`` made the drain note's inventory discard the interpreter
    (unknown_message); with ``inventory_animation`` the definitions survive."""
    from test_dfr_harness import Game, make_sandbox
    from test_dfr_harness import run as run_code

    for fixed in (False, True):
        sandbox = make_sandbox(dfr.settings({"animation": True}), True)
        sandbox.animation_handler = lambda request: {"action": "UP", "frames": 2, "steps": []}
        sandbox.inventory_answers_animation = fixed
        game = Game()
        run_code(sandbox, "tl = last_animation_timeline\ndef keep():\n    return 1", game)
        inventory = sandbox.inventory()
        if fixed:
            assert inventory is not None and sandbox.restarts == 0
            assert "keep" in json.dumps(inventory["functions"]) and "tl" in json.dumps(inventory["values"])
            assert run_code(sandbox, "print(keep())", game)["stdout"] == "1\n"
        else:
            assert inventory is None and sandbox.restarts == 1  # the v3/v4 reset
        sandbox.close()
