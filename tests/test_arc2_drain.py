"""naive12: v8.5p's context management in the conv harness (arena.context_drain drain_half, the drain
note with the python line and the harness inventory, the turn cap, the tool-output cap, drop_malformed, the loop rule)."""

import asyncio
import json

from test_arc2_conv import TASK, call, engine

from arc2 import conv, tasks
from arc2.loop import TaskRun
from arc2.repl import Repl
from arena import context_drain, loop_rule

ON = {"context_mode": "drain"}


def test_notes_are_v85p_s_with_puzzle_for_game():
    assert tasks.DRAIN_NOTE == context_drain.DRAIN_NOTE.replace("of this game", "of this puzzle")
    assert tasks.LOOP_NOTE.startswith("[Loop rule] Your last {streak} calls") and tasks.LOOP_NOTE.endswith(". ")
    assert conv.DRAIN_SPEC["drain_tokens"] == 56960 and conv.DRAIN_SPEC["turn_cap"] == 150  # sabl-v85p-c100-300k-v1
    assert conv.DRAIN_SPEC["turn_drain"] == 30 and conv.DRAIN_SPEC["inventory_tokens"] == 1536


def test_drain_drops_oldest_blocks_without_a_summary_and_keeps_reasoning():
    run = TaskRun("t#1", TASK, {"conversation": True})
    big = "y" * 36000  # ~12K tokens a reply
    eng, sent = engine([big, big, big, call("print(1)")], window=40000, cap=8000, **ON)
    asyncio.run(eng.advance(run, 10 ** 9, replies=4))
    assert not any(m.get("content") == tasks.CONV_COMPACT for s in sent for m in s["messages"])  # no summary request
    assert run.compactions == 0 and run.drains == 1
    after = sent[3]["messages"]
    assert after[:2] == run.head[:2] or after[0] == run.head[0]
    note = after[2]["content"]
    assert note.startswith("[Context drain] The oldest ") and context_drain.PYTHON_NOTE in note
    assert "[Harness inventory]" in note or "fresh" in note.lower()
    kept = after[3:]
    assert [m["role"] for m in kept] == ["assistant", "user"]  # the latest exchange, whole
    assert kept[0]["content"] == big and kept[0]["reasoning"] == "think"  # verbatim
    budget = 40000 - 8000 - 256
    assert sum(conv.approx_tokens(m) for m in after) <= budget
    run.interp.close()


def test_turn_cap_cuts_to_120_replies():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("x = 1")] * 152, window=10 ** 6, cap=8000, **ON, loop_rule=None)
    asyncio.run(eng.advance(run, 10 ** 9, replies=152))
    last = sent[-1]["messages"]
    assert sum(m["role"] == "assistant" for m in last) <= 121
    assert all(m["role"] != "tool" or prev.get("tool_calls") for prev, m in zip(last, last[1:]))  # no orphan result
    assert any(str(m.get("content") or "").startswith("[Context drain]") for m in last)
    run.interp.close()


def test_malformed_call_is_dropped_with_dfranzens_follow_up():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([{"name": "python", "arguments": "{\"code\": \"print(1)"}, call("print(1)")],
                       drop_malformed=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    history = sent[1]["messages"]
    assert history[-1] == {"role": "user", "content": context_drain.MALFORMED_CALL_FOLLOWUP.strip()}
    assert history[-2]["role"] == "assistant" and "tool_calls" not in history[-2]
    assert run.replies[0]["outcome"] == "malformed_dropped"
    run.interp.close()


def test_loop_rule_clears_the_conversation_after_20_short_calls_without_progress():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("print(1)")] * 21, window=10 ** 6, cap=8000, **ON, loop_rule=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=21))
    assert loop_rule.DEFAULTS == {"action": "fresh_context", "streak": 20, "mean_completion": 300, "similarity": 0.9}
    cleared = sent[20]["messages"]
    assert len(cleared) == 3 and cleared[2]["content"].startswith("[Loop rule] Your last 20 calls")
    assert "Your python interpreter was not touched" in cleared[2]["content"]
    assert len(sent[19]["messages"]) > 30
    run.interp.close()


def test_inventory_lists_live_definitions_outside_the_replay_history():
    repl = Repl(TASK, strict=False, call_seconds=5)
    repl.run("def area(g):\n    'cells'\n    return len(g) * len(g[0])\nseen = [1, 2, 3]")
    calls = len(repl.history)
    data = repl.inventory()
    assert ["area", "(g)", "cells"] in data["functions"]
    assert any(v[0] == "seen" and v[2] == 3 for v in data["values"])
    assert len(repl.history) == calls
    repl.close()


def test_tool_cap_line_in_the_system_prompt():
    text = tasks.conv_system({"tool_chars": 3072 * 4})
    assert "Tool output is capped to about 3072 tokens" in text
    assert "capped" not in tasks.conv_system({})


def test_playbook_is_saved_uncapped_and_shown_in_the_drain_note():
    run = TaskRun("t#1", TASK, {"conversation": True})
    big = "y" * 36000
    book = "Working model: transpose (checked). " + "x" * 6000  # past v8.5p's old 5,000 cap: no cap in style 2
    eng, sent = engine([call(f"playbook({book!r})"), big, big, big, call("print(1)")], window=40000, cap=8000,
                       memo=True, **ON)
    eng.repl = lambda r: Repl(r.task, strict=False, call_seconds=5, memo=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=5))
    assert "playbook(text)" in sent[0]["messages"][0]["content"]  # the system paragraph
    tool = next(m for m in sent[1]["messages"] if m["role"] == "tool")
    assert tool["content"].startswith("[playbook] saved, %d characters" % len(book))
    note = next(m["content"] for m in sent[-1]["messages"] if str(m.get("content") or "").startswith("[Context drain]"))
    assert tasks.PLAYBOOK_SHOWN + book in note
    run.interp.close()


def test_playbook_due_line_once_before_a_cut():
    run = TaskRun("t#1", TASK, {"conversation": True})
    mid = "y" * 27000  # ~9K tokens a reply: the third request is past 0.8 of the 31,744 budget, below it
    eng, sent = engine([mid, mid, mid, call("print(1)")], window=40000, cap=8000, memo=True, **ON)
    eng.repl = lambda r: Repl(r.task, strict=False, call_seconds=5, memo=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=4))
    dues = [i for i, s in enumerate(sent) if tasks.PLAYBOOK_OPEN_DUE in str(s["messages"][-1].get("content"))]
    assert len(dues) == 1
    run.interp.close()


def test_step_back_after_six_failing_programs_then_every_eight():
    from test_arc2_conv import WRONG
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([WRONG] * 15, step_back=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=15))
    blocks = [i for i, s in enumerate(sent) if "STEP BACK (harness check: your last" in str(s["messages"][-1].get("content"))]
    assert blocks == [6, 14]  # after the 6th failing program, again 8 replies later
    text = sent[6]["messages"][-1]["content"]
    assert "6 programs (0 pass every training pair)" in text and "best 0/2 training pairs" in text
    run.interp.close()


def test_step_back_after_twelve_replies_without_a_program():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("print(1)")] * 13, step_back=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=13))
    assert "12 replies without a new program" in sent[12]["messages"][-1]["content"]
    assert not any("STEP BACK" in str(s["messages"][-1].get("content")) for s in sent[:12])
    run.interp.close()


def test_drain_jitter_draws_a_seeded_trigger_per_drain_and_keeps_keep_tokens():
    """v52 (l4 request, port of v84-sub 77e18eef0): drain.jitter {low, high, seed}: each drain index has its own trigger,
    uniform in [low, high], sha256(seed|task|index) (same draw on a rerun), capped at the window's budget; the drain
    keeps keep_tokens; trigger and post-drain size are logged. Off by default."""
    big = "y" * 9000  # ~3K tokens a reply
    jitter = {"low": 9000, "high": 14000, "seed": 7}

    def play():
        run = TaskRun("t#1", TASK, {"conversation": True})
        eng, sent = engine([big] * 14, window=40000, cap=8000, **ON,
                           drain={"turn_cap": 0, "keep_tokens": 4000, "jitter": jitter})
        rows = []
        eng.record = rows.append
        asyncio.run(eng.advance(run, 10 ** 9, replies=14))
        run.interp.close()
        return run, rows

    run, rows = play()
    draws = [r for r in rows if r.get("event") == "drain_trigger"]
    drains = [r for r in rows if r.get("event") == "context_drain"]
    assert len(drains) >= 2 and len(draws) >= len(drains)
    assert all(9000 <= d["trigger_tokens"] <= 14000 and d["keep_tokens"] == 4000 for d in draws)
    assert [d["drain_index"] for d in draws] == list(range(len(draws)))
    for d in drains:
        assert d["trigger_tokens"] == draws[d["drain_index"]]["trigger_tokens"]
        assert d["tokens_before"] > d["trigger_tokens"] and d["tokens_after"] <= 4000 + 3500  # the latest exchange kept
    assert len({d["trigger_tokens"] for d in draws}) > 1
    assert [d["trigger_tokens"] for d in play()[1] if d.get("event") == "drain_trigger"] == \
        [d["trigger_tokens"] for d in draws]  # seeded: the rerun draws the same
    import hashlib
    digest = hashlib.sha256(b"7|t#1|0").digest()
    assert draws[0]["trigger_tokens"] == 9000 + int.from_bytes(digest[:8], "big") * 5001 // 2 ** 64
