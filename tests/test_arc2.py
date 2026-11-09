"""ARC-AGI-2 arm: the loop against a scripted model, guesses, the read-out and the sandbox."""

import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from arc2 import sandbox, tasks
from arc2.loop import (SYMMETRIES, ContextFull, Engine, TaskRun, augment_back, augment_spec, augment_view, guesses,
                       readout, roster, score, token_clock)

TASK = {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 3], [2, 4]]},
                  {"input": [[5, 6]], "output": [[5], [6]]}],
        "test": [{"input": [[7, 8, 9]], "output": [[7], [8], [9]]}]}
RIGHT = "```python\ndef transform(grid):\n    return [list(r) for r in zip(*grid)]\n```"
WRONG = "```python\ndef transform(grid):\n    print('hi')\n    return grid\n```"
OTHER = "```python\ndef transform(grid):\n    return [[c] for row in grid for c in row] if len(grid) == 1 else " \
        "[list(r) for r in zip(*grid)]\n```"


def engine(script, **kwargs):
    sent = []

    async def complete(task_id, messages, count, max_tokens, settings):
        sent.append([dict(m) for m in messages])
        content = script.pop(0)
        if isinstance(content, Exception):
            raise content
        if isinstance(content, dict):  # a `python` tool call
            return {"message": {"content": None, "reasoning": "think", "tool_calls": [{"id": "c1", "function": content}]},
                    "finish_reason": "tool_calls", "usage": {"completion_tokens": 100, "prompt_tokens": 500}}
        return {"message": {"content": content, "reasoning": "think"}, "finish_reason": "stop" if content else "length",
                "usage": {"completion_tokens": 1000, "prompt_tokens": 500}}

    async def count(messages, settings):
        return sum(len(m["content"]) for m in messages) // 2

    async def execute(code, inputs, **kw):
        return await asyncio.to_thread(sandbox.run, code, inputs, **kw)

    async def explore(cells, code, task):
        return await asyncio.to_thread(sandbox.run_cells, cells, code, task)

    return Engine(complete=complete, count=count, execute=execute, explore=explore, window=65536, cap=16384,
                  **kwargs), sent


def test_refine_then_confirm():
    run = TaskRun("t", TASK)
    eng, sent = engine([WRONG, "", RIGHT, OTHER])
    asyncio.run(eng.advance(run, 300000))
    assert run.done == "verified" and run.attempts == 2 and run.spent == 4000
    assert [r["outcome"] for r in run.replies] == ["wrong", "truncated", "verified", "verified"]
    # The refinement turn carries the reasoning, the exact differences and the print output; the confirm is fresh.
    assert sent[1][2]["reasoning"] == "think" and "(0,1): 3 not 2" in sent[1][3]["content"]
    assert "Printed by your program:\nhi" in sent[1][3]["content"] and sent[2][5]["content"] == tasks.TRUNCATED
    assert len(sent[3]) == 2
    assert score(run)[:2] == (1, 1) and score(run, 2000) == (0, 1, ["unverified"], [False])
    cont = TaskRun("t", TASK, {"truncated_continue": True})  # a cut reply goes on from its kept reasoning
    eng, sent = engine([WRONG, "", RIGHT, OTHER])
    asyncio.run(eng.advance(cont, 300000))
    assert sent[2][5]["content"] == tasks.CONTINUE and sent[2][4]["reasoning"] == "think" and cont.done == "verified"
    out = readout([run])
    assert out["final"]["score_outputs"] == 100 and out["final"]["by_source"]["verified"]["accuracy"] == 1
    assert [c["score_outputs"] for c in out["curve"]][:3] == [100, 100, 100] and out["curve"][0]["complete"]
    assert out["curve"][0]["mean_tokens"] == 4000 and score(run, 3000)[0] == 1
    assert out["where_tokens_go"]["after_first_verified"] == 1000 and out["ended"]["verified"] == 1


def test_cut_program():
    # A cut reply: one program reply with no tools at the cut cap and effort; unless it verifies, the attempt ends.
    caps = []
    run = TaskRun("t", TASK, {"cut_program": 2048, "tool": True})
    eng, sent = engine(["", WRONG, "", RIGHT, RIGHT], settings={"confirm_attempts": 1})
    inner = eng.complete

    async def spy(task_id, messages, count, max_tokens, settings):
        caps.append((max_tokens, settings.get("tool"), settings.get("effort")))
        return await inner(task_id, messages, count, max_tokens, settings)
    eng.complete = spy
    asyncio.run(eng.advance(run, 300000))
    assert [r["outcome"] for r in run.replies][:4] == ["truncated", "wrong", "truncated", "verified"]
    assert caps[0] == (16384, True, None) and caps[1] == (2048, False, "low") and caps[2][0] == 16384
    assert sent[1][-1]["content"] == tasks.TRUNCATED and len(sent[2]) == 2  # attempt 2 starts fresh
    assert run.replies[1]["attempt"] == 1 and run.replies[2]["attempt"] == 2 and not run.cut_pending
    off = TaskRun("t", TASK)  # off by default: a cut reply's follow-up keeps the full cap
    eng, sent = engine(["", WRONG, RIGHT, RIGHT])
    asyncio.run(eng.advance(off, 300000))
    assert off.replies[1]["attempt"] == 1 and off.replies[2]["attempt"] == 1


def test_settle_agree():
    # One verified program and a failed confirm: settled by default, confirms on with settle_agree until a second agrees.
    run = TaskRun("t", TASK)
    eng, _ = engine([RIGHT, WRONG], settings={"attempt_replies": 1})
    asyncio.run(eng.advance(run, 300000))
    assert run.done == "verified" and run.attempts == 2
    run = TaskRun("t", TASK, {"confirm_attempts": 3, "settle_agree": True})
    eng, _ = engine([RIGHT, WRONG, RIGHT], settings={"attempt_replies": 1})
    asyncio.run(eng.advance(run, 300000))
    assert run.done == "verified" and run.attempts == 3 and run.confirms == 2


def test_progress_weight():
    # An open task whose best program passes half the training pairs ranks as if it had spent 3/4 of its tokens.
    eng, _ = engine([], settings={"progress_weight": 0.5})
    a, b = TaskRun("a", TASK), TaskRun("b", TASK)
    a.spent = b.spent = 4000
    a.candidates.append({"passed": 1, "pairs": 2, "verified": False})
    assert eng.priority(a) == 3000 and eng.priority(b) == 4000
    assert engine([])[0].priority(a) == 4000  # off by default


def test_augment_copies():
    grid = [[1, 2, 0], [3, 4, 5]]
    for fwd, back in SYMMETRIES:
        assert back(fwd(grid)) == grid
    assert len({json.dumps(f(grid)) for f, _ in SYMMETRIES}) == 8
    for k in range(1, 6):
        sym, colours = augment_spec("t#1", k)
        assert 1 <= sym <= 7 and colours[0] == 0 and sorted(colours) == list(range(10))
        assert augment_spec("t#2", k) == (sym, colours)  # a roster copy of a task sees the same copies
        view = augment_view(TASK, "t", k)
        assert "output" not in view["test"][0]
        assert augment_back(view["train"][0]["output"], "t", k) == TASK["train"][0]["output"]
    assert augment_view(TASK, "t", 1) != augment_view(TASK, "t", 2) or augment_spec("t", 1) == augment_spec("t", 2)


def test_augment_agreement():
    a = {"code": "a", "verified": True, "passed": 2, "accuracy": 1.0, "tests": [[[1]]], "tokens_at": 1}
    b = {"code": "b", "verified": True, "passed": 2, "accuracy": 1.0, "tests": [[[2]]], "tokens_at": 2}
    c = {"code": "c", "verified": True, "passed": 2, "accuracy": 1.0, "tests": [[[2]]], "tokens_at": 3, "copy": 1}
    assert [g for g, _ in guesses([a, b], [[[0]]])[0]] == [[[1]], [[2]]]  # earliest first
    assert [g for g, _ in guesses([a, b, c], [[[0]]])[0]] == [[[2]], [[1]]]  # two copies agree on [[2]]
    eng, _ = engine([], settings={"augment": 2, "confirm_attempts": 3})
    run = TaskRun("t", TASK)
    run.first_verified_attempt, run.confirms = 1, 1
    run.candidates = [dict(b), dict(b, code="b2")]
    assert eng.phase(run) == "confirm"  # same copy agreeing does not settle
    run.candidates.append(c)
    assert eng.phase(run) == "settled"


def test_augment_attempt():
    # Attempt 2 works on copy 1: the prompt shows the copy, and its test output comes back in the task's frame.
    run = TaskRun("t", TASK)
    eng, sent = engine([WRONG, "```python\ndef transform(grid):\n    return [list(r) for r in grid]\n```"],
                       settings={"augment": 2, "attempt_replies": 1})
    asyncio.run(eng.advance(run, 2000))
    view = augment_view(TASK, "t", 1)
    assert run.attempts == 2 and run.copy == 1 and sent[1][1]["content"] == __import__("arc2.tasks").tasks.task_prompt(view)
    c = run.candidates[-1]
    assert c["copy"] == 1 and c["tests"][0] == augment_back(view["test"][0]["input"], "t", 1) == TASK["test"][0]["input"]


def test_review_turn():
    odd = RIGHT.replace("return [", "return [[0]] if len(grid[0]) == 3 else [")  # passes the pairs, wrong on the test
    run = TaskRun("t", TASK, {"name": "review", "review": True})
    eng, sent = engine([odd, RIGHT, OTHER])
    asyncio.run(eng.advance(run, 300000))
    assert [r["outcome"] for r in run.replies] == ["verified", "review_verified", "verified"] and run.done == "verified"
    assert "your program's output (1x1):\n0" in sent[1][-1]["content"] and len(sent[2]) == 2  # the confirm is fresh
    assert [bool(c.get("review")) for c in run.candidates] == [False, True, False] and run.attempts == 2
    assert score(run)[0] == 1 and score(run, 1000)[0] == 0 and score(run, 2000)[0] == 1  # the correction is guess 2
    assert "second answer is free" in tasks.review_prompt(TASK, [[[0]]], "second")
    second = TaskRun("t", TASK, {"review": True, "review_style": "second"})
    eng, sent = engine([RIGHT, "KEEP"], settings={"confirm_attempts": 0})
    asyncio.run(eng.advance(second, 300000))
    assert "first of two allowed answers" in sent[1][-1]["content"] and second.done == "verified"
    keep = TaskRun("t", TASK, {"review": True})
    eng, sent = engine([RIGHT, "KEEP", OTHER])
    asyncio.run(eng.advance(keep, 300000))
    assert [r["outcome"] for r in keep.replies] == ["verified", "review_keep", "verified"] and len(keep.candidates) == 2
    last = TaskRun("t", TASK, {"review": True})  # the review reply comes after the attempt's last allowed reply
    eng, _ = engine([WRONG, RIGHT, ""], settings={"attempt_replies": 2, "confirm_attempts": 0})
    asyncio.run(eng.advance(last, 300000))
    assert [r["outcome"] for r in last.replies] == ["wrong", "verified", "review_cut"] and last.done == "verified"

    both = TaskRun("t", TASK, {"review": True, "review_style": "check+second"})  # check, then second, one attempt
    eng, sent = engine([odd, RIGHT, OTHER], settings={"confirm_attempts": 0})
    asyncio.run(eng.advance(both, 300000))
    assert [r["outcome"] for r in both.replies] == ["verified", "review_verified", "review_verified"]
    assert "reply with exactly KEEP" in sent[1][-1]["content"] and "first of two allowed answers" in sent[2][-1]["content"]
    assert "your program's output (1x1):\n0" in sent[1][-1]["content"] and "(1x1):\n0" not in sent[2][-1]["content"]
    assert both.attempts == 1 and both.done == "verified" and [bool(c.get("review")) for c in both.candidates] == [False, True, True]
    keep2 = TaskRun("t", TASK, {"review": True, "review_style": "check+second"})  # KEEP on the check: second still asked
    eng, sent = engine([RIGHT, "KEEP", "KEEP"], settings={"confirm_attempts": 0})
    asyncio.run(eng.advance(keep2, 300000))
    assert [r["outcome"] for r in keep2.replies] == ["verified", "review_keep", "review_keep"] and len(sent) == 3
    for style in ("check", "second"):  # one style: one review turn, the attempt ends after it
        one = TaskRun("t", TASK, {"review": True, "review_style": style})
        eng, sent = engine([RIGHT, "KEEP", OTHER], settings={"confirm_attempts": 0})
        asyncio.run(eng.advance(one, 300000))
        assert [r["outcome"] for r in one.replies] == ["verified", "review_keep"] and len(sent) == 2
        assert sent[1][-1]["content"] == tasks.review_prompt(one.view, [[[7], [8], [9]]], style)

    def cand(code, grid, at, **more):
        return {"code": code, "passed": 2, "pairs": 2, "accuracy": 1, "verified": True, "tests": [grid],
                "tokens_at": at, **more}

    ranked = guesses([cand("b", [[2]], 20), cand("r", [[5]], 25, review=True), cand("c", [[3]], 30)], [[[0]]])
    assert [g for g, _ in ranked[0]] == [[[2]], [[3]]]  # an independent program before a correction


def test_python_tool_arm():
    def call(code):
        return {"name": "python", "arguments": json.dumps({"code": code})}

    run = TaskRun("t", TASK, {"name": "tool", "tool": True})
    eng, sent = engine([call("shapes = [np.array(p['input']).shape for p in train]\nprint(shapes)"),
                        call("1/0"), call("result = (shapes, len(test_inputs))"), {"name": "python", "arguments": "{"},
                        WRONG, RIGHT, call("print(1)"), call("print(2)"), OTHER],
                       settings={"tool_calls": 4})
    asyncio.run(eng.advance(run, 300000))
    assert [r["outcome"] for r in run.replies] == ["tool"] * 4 + ["wrong", "verified", "tool", "tool", "verified"]
    assert sent[0][0]["content"] == tasks.SYSTEM_TOOL and "`python` tool" in tasks.SYSTEM_TOOL
    assert sent[1][2]["tool_calls"][0]["function"]["name"] == "python" and sent[1][2]["reasoning"] == "think"
    assert sent[1][3] == {"role": "tool", "tool_call_id": "c1", "content": "[(2, 2), (1, 2)]"}
    assert sent[2][5]["content"].startswith("Error: ZeroDivisionError")  # and the failed snippet leaves no state
    assert sent[3][7]["content"] == "result = ([(2, 2), (1, 2)], 1)" and "one string argument" in sent[4][9]["content"]
    assert run.done == "verified" and run.spent == 3600 and score(run)[0] == 1
    assert len(sent[6]) == 2 and sent[7][3]["content"] == "1"  # the confirm: a fresh conversation and session
    assert tasks.chat_request("m", [], 5, temperature=1, effort="low", tools=[tasks.PYTHON_TOOL])["tool_choice"] == "auto"
    assert "tools" not in tasks.chat_request("m", [], 5, temperature=1, effort="low")
    limited = TaskRun("t", TASK, {"tool": True})  # past the limit the call is refused and counts as a reply
    eng, sent = engine([call("x = 1"), call("print(x)"), RIGHT], settings={"tool_calls": 1, "confirm_attempts": 0})
    asyncio.run(eng.advance(limited, 300000))
    assert sent[2][-1]["content"] == tasks.TOOL_LIMIT and limited.done == "verified"
    plain = TaskRun("t", TASK)  # the baseline arm: no tool offered, a call is not read
    eng, sent = engine([call("x = 1"), RIGHT, OTHER])
    asyncio.run(eng.advance(plain, 300000))
    assert sent[0][0]["content"] == tasks.SYSTEM and [r["outcome"] for r in plain.replies][0] == "no_code"
    assert sent[1][-1]["content"] == tasks.NO_CODE


def test_python_tool_fixes():
    """C1 (audit 20 Oct, setting tool_fixes): preloaded names import, more modules, a failing snippet keeps what ran
    before it, empty calls are free, output shows start and end, and no `tools` are offered after the last call."""
    def call(code):
        return {"name": "python", "arguments": json.dumps({"code": code})}

    offered = []

    async def explore(cells, code, task, **kw):
        return await asyncio.to_thread(sandbox.run_cells, cells, code, task, **kw)

    async def execute(code, inputs, **kw):
        return await asyncio.to_thread(sandbox.run, code, inputs, **kw)

    eng, sent = engine([call("import np\nimport train\nimport sys, traceback, inspect, pprint\n"
                             "print(np.array(train[0]['input']).shape)"),
                        call("def f(g):\n    return len(g)\nx = f(train[0]['input'])\n1/0\ny = 5"),
                        call(""), call("print(f([1, 2, 3]), x)\nprint('a' * 10000 + 'END')"),
                        "```python\nimport sys\ndef transform(grid):\n    return [list(r) for r in zip(*grid)]\n```"],
                       settings={"tool_calls": 3, "tool_fixes": True, "confirm_attempts": 0})
    complete = eng.complete

    async def recorded(task_id, messages, count, max_tokens, settings):
        offered.append(bool(settings.get("tool")))
        return await complete(task_id, messages, count, max_tokens, settings)

    eng.complete, eng.explore, eng.execute = recorded, explore, execute
    run = TaskRun("t", TASK, {"name": "c1", "tool": True})
    asyncio.run(eng.advance(run, 300000))
    system = sent[0][0]["content"]
    assert "numpy is preloaded as `np`" in system and "keeps what ran before the failing statement" in system
    assert "6,000 characters" in system and "leaves nothing behind" not in system
    tools = [m["content"] for m in sent[-1] if m["role"] == "tool"]
    assert tools[0] == "(2, 2)" and tools[1].startswith("Error: ZeroDivisionError") and "line 4" in tools[1]
    assert tools[2] == "Error: the call needs one string argument `code`."  # free: the next call is the third
    assert tools[3].startswith("3 2\naaa") and "characters not shown" in tools[3] and "END" in tools[3]
    assert tools[3].endswith(tasks.TOOL_LAST) and len(tools[3]) < tasks.TOOL_CHARS + 300
    assert offered == [True, True, True, True, False] and run.attempt_empty == 1 and run.attempt_tools == 3
    assert run.done == "verified" and score(run)[0] == 1  # the program's own `import sys` loads too
    # The pieces against the sandbox, and the default unchanged.
    kept = sandbox.run_cells([], "a = 1\n@dec\ndef g():\n    pass\nb = 2", TASK, fixes=True)
    assert kept["error"].startswith("NameError") and kept["kept"] == "a = 1"
    assert "kept" not in sandbox.run_cells([], "a = 1\nb = c", TASK)
    assert "ImportError" in sandbox.run_cells([], "import sys", TASK)["error"]
    assert "ImportError" in sandbox.run_cells([], "import np", TASK)["error"]
    assert not sandbox.run_cells([], "import np, train\nprint(len(train))", TASK, fixes=True).get("error")
    assert "ImportError" in sandbox.run("import sys\ndef transform(grid):\n    return grid", [[[1]]])["error"]
    long = sandbox.run_cells([], "print('b' * 5000)", TASK)
    assert long["stdout"] == "b" * 4000 and tasks.tool_text(long) == "b" * tasks.STDOUT_CHARS
    assert tasks.clip("x" * 10, 6) == "xxxx\n... [4 characters not shown] ...\nxx" and tasks.clip("xy", 6) == "xy"


def test_submission_file(tmp_path):
    from arc2 import kaggle
    from arc2.tasks import write_json
    from arc2.trial import submission

    hidden = {"train": TASK["train"], "test": [{"input": [[7, 8, 9]], "output": None}]}
    solved, open_ = TaskRun("a", hidden), TaskRun("b", hidden)
    eng, _ = engine([RIGHT, OTHER], settings={"confirm_attempts": 1})
    asyncio.run(eng.advance(solved, 300000))
    out = submission([solved, open_])
    assert out["a"] == [{"attempt_1": [[7], [8], [9]], "attempt_2": [[7], [8], [9]]}]
    assert out["b"] == kaggle.fallback({"b": hidden})["b"] == [{"attempt_1": [[7, 8, 9]], "attempt_2": [[7, 8, 9]]}]
    write_json(tmp_path / "s.json", out)
    assert json.loads((tmp_path / "s.json").read_text()) == out and not list(tmp_path.glob("*.part"))
    challenges = {"a": hidden, "b": hidden, "c": hidden}
    assert kaggle.problems(out, {"a": hidden, "b": hidden}) == [] and len(kaggle.problems(out, challenges)) == 1
    assert kaggle.problems({"a": [{"attempt_1": [[1]], "attempt_2": [[1, 2], [3]]}]}, {"a": hidden}) != []


def test_submission_merges_copies():
    """Entries ``id#1`` / ``id#2`` (trial ``copies``) pool their candidates; agreement counts per entry; one entry
    alone gives what the unmerged writer gave."""
    from arc2.trial import submission

    task = {"train": TASK["train"], "test": [{"input": [[1]], "output": None}]}

    def cand(code, grid, verified=True, copy=0, tokens=1):
        return {"code": code, "tests": [grid], "verified": verified, "passed": 2, "accuracy": 1.0, "copy": copy,
                "tokens_at": tokens}
    a, b = TaskRun("t#1", task), TaskRun("t#2", task)
    a.candidates = [cand("x", [[5]], copy=0, tokens=1), cand("y", [[6]], copy=1, tokens=2)]
    b.candidates = [cand("z", [[6]], copy=0, tokens=3), cand("w", [[7]], verified=False, copy=0, tokens=4)]
    out = submission([a, b])
    assert out == {"t": [{"attempt_1": [[6]], "attempt_2": [[5]]}]}  # [[6]] found by both entries leads
    assert submission([a]) == {"t": [{"attempt_1": [[5]], "attempt_2": [[6]]}]}  # one entry: tokens order
    single = TaskRun("u", task)
    single.candidates = [cand("x", [[5]], copy=0), cand("y", [[5]], copy=1), cand("v", [[4]], copy=2)]
    assert submission([single]) == {"u": [{"attempt_1": [[5]], "attempt_2": [[4]]}]}


def test_attempt_keeps_its_blocks_between_turns():
    import httpx

    from arc2 import trial
    from arena.gate import KVGate

    async def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 10}})

    async def play(hold):
        log = []
        gate = KVGate(100, slots=4, log=log.append)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1/")
        transport = trial.Transport(client, gate, model="m", effort="low", temperature=0.7,
                                    deadline=time.monotonic() + 60, hold=hold)
        await transport.complete("a", [], 30, 10, {})
        between = gate.reserved
        waiter = asyncio.create_task(transport.complete("b", [], 55, 10, {}))  # 65 of 100: not beside a held 40
        await asyncio.sleep(0)
        await transport.complete("a", [], 50, 10, {})  # a's next turn, 60: before the waiting b when a holds
        grown = gate.reserved
        transport.drop("a")  # a's attempt ends
        await waiter
        transport.drop("b")
        await client.aclose()
        return between, grown, gate.reserved, gate.inflight, [row["event"] for row in log if row["game"] == "a"]

    assert asyncio.run(play(False)) == (0, 0, 0, 0, ["admit", "release"] * 2)
    assert asyncio.run(play(True)) == (40, 60, 0, 0, ["admit", "grow", "release"])


def test_attempt_waits_for_blocks_with_its_own_held():
    import httpx

    from arc2 import trial
    from arena.gate import KVGate

    gates = {}

    async def handler(request):
        name = json.loads(request.content)["messages"][0]["content"]
        if name in gates:
            await gates[name].wait()
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 10}})

    async def play():
        log = []
        gate = KVGate(100, slots=4, log=log.append)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1/")
        transport = trial.Transport(client, gate, model="m", effort="low", temperature=0.7,
                                    deadline=time.monotonic() + 60, hold=True)
        def say(name):
            return [{"role": "user", "content": name}]

        await transport.complete("a", say("a"), 30, 10, {})  # a holds 40
        gates["b"] = asyncio.Event()
        b = asyncio.create_task(transport.complete("b", say("b"), 40, 10, {}))  # b in flight with 50: 90 of 100
        await asyncio.sleep(0.01)
        c = asyncio.create_task(transport.complete("c", say("c"), 20, 10, {}))  # c queues at the gate for 30
        await asyncio.sleep(0.01)
        a2 = asyncio.create_task(transport.complete("a", say("a2"), 50, 10, {}))  # a wants 60: waits, holding 40
        await asyncio.sleep(0.01)
        waiting = (gate.reserved, len(transport.growers), a2.done(), c.done())
        gates["b"].set()
        await b
        transport.drop("b")  # b's attempt ends: a grows first (60), then c fits (30)
        await asyncio.gather(a2, c)
        alone = await transport.grown(transport.held.pop("a"), 95)  # nothing in flight can free blocks: given up
        transport.drop("c")
        await client.aclose()
        events = [(row["event"], row["game"]) for row in log]
        return waiting, alone, gate.reserved, gate.inflight, events.index(("grow", "a")) < events.index(("admit", "c"))

    assert asyncio.run(play()) == ((90, 1, False, False), None, 0, 0, True)


PLE_PACK_SCRIPT = r'''
import dataclasses, sys, torch
import torch.nn.functional as F
text, anchor, replacement, SLACK = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3], int(sys.argv[4])
assert text.count(anchor) == 1
text = text.replace(anchor, replacement)
start = text.index(anchor)
body = text[start:text.index("\n    def ", text.index("def _arc_short_conv_prefill_packed("))]
space = {"torch": torch, "F": F, "NULL_BLOCK_ID": 0, "PleShortConvAttentionMetadata": object}
exec("class Layer:\n    conv_state_len = 9\n    short_conv_dilation = 3\n" + body, space)

@dataclasses.dataclass
class Meta:
    non_spec_query_start_loc: torch.Tensor
    has_initial_states_p: torch.Tensor
    max_prefill_query_len: int

def case(lengths, decodes=7, channels=8):
    torch.manual_seed(len(lengths))
    rows, tokens = len(lengths), sum(lengths)
    bounds = torch.tensor([0] + lengths).cumsum(0) + decodes
    meta = Meta(torch.cat((torch.arange(decodes), bounds)).to(torch.int32), torch.rand(rows) < 0.6, max(lengths))
    x, weights = torch.randn(tokens, channels), torch.randn(channels, 4)
    state = torch.randn(rows + 6, channels, 10)  # row 0: the null block, as in the runtime
    index = (torch.randperm(rows + 5)[:rows] + 1).to(torch.int32)
    index[min(3, rows - 1)] = 0 if rows > 3 else index[rows - 1]
    results = []
    for bounded in (False, True):
        layer, calls, cache = space["Layer"](), [], state.clone()
        inner = layer._arc_short_conv_prefill_packed
        def packed(x_p, metadata, *rest, inner=inner, calls=calls):
            calls.append((rest[3], metadata.max_prefill_query_len, x_p.shape[0]))
            return inner(x_p, metadata, *rest)
        layer._arc_short_conv_prefill_packed = packed
        call = layer._short_conv_dilated_prefill_batched if bounded else packed
        results.append((call(x, meta, cache, weights, index, rows, decodes, tokens), cache, calls))
    (out_a, state_a, calls_a), (out_b, state_b, calls_b) = results
    valid = torch.repeat_interleave(index != 0, torch.tensor(lengths))
    assert torch.allclose(out_a[valid], out_b[valid], atol=1e-5) and not out_b[~valid].any()
    assert torch.allclose(state_a, state_b, atol=1e-6)
    assert sum(c[2] for c in calls_b) == tokens
    return calls_a, calls_b

# The step that failed the 8,192-token test: 50 short rows around one 3,110-token row.
before, after = case([100] * 25 + [3110] + [97, 103] * 12 + [0])
assert before == [(51, 3110, 8010)] and len(after) > 1
assert all(rows * longest <= tokens + SLACK for rows, longest, tokens in after)
assert max(rows * longest for rows, longest, _ in after) == 3110
# The steps that ended the L4 engines: full 3,136-token chunks beside short turns (10 rows: 31,360 padded slots).
before, after = case([3136] + [40, 250, 12, 90, 300, 7, 130, 60, 71])
assert before == [(10, 3136, 4096)] and max(rows * longest for rows, longest, _ in after) == 3136
assert all(rows * longest <= tokens + SLACK for rows, longest, tokens in after)
before, after = case([1500, 1480, 20, 30, 1066])
assert max(rows * longest for rows, longest, _ in after) <= 2980 + SLACK
for seed in range(40):  # any mix of lengths in a 4,096-token step
    torch.manual_seed(1000 + seed)
    rows = int(torch.randint(2, 19, ()))
    cuts = sorted(torch.randint(0, 4097, (rows - 1,)).tolist())
    lengths = [b - a for a, b in zip([0] + cuts, cuts + [4096])]
    before, after = case(lengths)
    assert all(rows * longest <= tokens + SLACK for rows, longest, tokens in after) or len(after) == 1
    assert max(rows * longest for rows, longest, _ in after) <= 4096 + SLACK or len(after) == 1
# A tight step and a single row keep the pinned call.
assert case([400, 380, 410])[1] == [(3, 410, 1190)]
assert case([5000])[1] == [(1, 5000, 5000)]
print("OK", after)
'''


def test_ple_pack_patch_text():
    from arc2 import l4_ple_pack as pack

    stock = "class Layer:\n" + pack.ANCHOR + "        self,\n    ):\n        return 1\n"
    done = pack.patched(stock)
    assert done.count("def _short_conv_dilated_prefill_batched(") == 1 and "__SLACK__" not in done
    assert done.count("def _arc_short_conv_prefill_packed(") == 1 and f"+ {pack.SLACK}" in done
    assert pack.patched(done) == done  # once
    with pytest.raises(RuntimeError):
        pack.patched(stock.replace(pack.ANCHOR, ""))
    compile(done, "ple_layer", "exec")


@pytest.mark.skipif(not os.environ.get("ARC_PINNED_PYTHON") or not os.environ.get("ARC_PINNED_SITE"),
                    reason="needs the pinned runtime's ple_layer.py and a Python with torch")
def test_ple_pack_grouped_call_equals_the_pinned_call():
    from arc2 import l4_ple_pack as pack

    source = Path(os.environ["ARC_PINNED_SITE"]) / pack.RELATIVE
    done = subprocess.run([os.environ["ARC_PINNED_PYTHON"], "-c", PLE_PACK_SCRIPT, str(source), pack.ANCHOR,
                           pack.REPLACEMENT.replace("__SLACK__", str(pack.SLACK)), str(pack.SLACK)],
                          capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.startswith("OK"), done.stderr[-3000:]


PLE_DROP_STUB = '''
def safetensors_weights_iterator(
    hf_weights_files, local_expert_ids=None
):
    leftover_state_dict: dict[str, torch.Tensor] = {}
    for st_file in tqdm(
        hf_weights_files,
    ):
        if False:
            pass
        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
'''


def ple_drop_stream(tmp_path, files, *, offload=True, pattern="model-plefp8-"):
    """The patched stub over real files, with fadvise recorded: (events in order, logged lines)."""
    import contextlib
    import types

    from arc2 import l4_ple_drop as drop

    events, logged, opened = [], [], {}
    for name in files:
        (tmp_path / name).write_bytes(b"x" * 4096)

    def open_(path, flags):
        fd = os.open(path, flags)
        opened[fd] = os.path.basename(path)
        return fd

    fake_os = types.SimpleNamespace(environ={drop.ENV: pattern} if pattern else {}, path=os.path, open=open_,
                                    close=os.close, O_RDONLY=os.O_RDONLY, POSIX_FADV_DONTNEED=4,
                                    posix_fadvise=lambda fd, offset, length, advice: events.append(
                                        ("drop", opened[fd], offset, length, advice)))

    @contextlib.contextmanager
    def safe_open(path, framework):
        yield types.SimpleNamespace(keys=lambda: ["a", "b"],
                                    get_tensor=lambda name: (os.path.basename(path), name))

    namespace = {"os": fake_os, "json": json, "tqdm": lambda files, **_: files, "safe_open": safe_open,
                 "should_skip_weight": lambda name, ids: False, "is_offload_process": lambda: offload,
                 "logger": types.SimpleNamespace(info=lambda *a: logged.append(a[0] % a[1:]))}
    exec(compile(drop.patched(PLE_DROP_STUB), "weight_utils", "exec"), namespace)
    for name, (file, _) in namespace["safetensors_weights_iterator"]([str(tmp_path / f) for f in files]):
        events.append(("read", file, name))
    return events, logged


def test_ple_drop_patch_text():
    from arc2 import l4_ple_drop as drop

    done = drop.patched(PLE_DROP_STUB)
    assert done.count("def _arc_ple_drop(") == 1 and done.count("_arc_ple_drop(_arc_pending, st_file)") == 1
    assert drop.patched(done) == done  # once
    with pytest.raises(RuntimeError):
        drop.patched(PLE_DROP_STUB.replace("param = f.get_tensor(name)", "param = f.get_slice(name)"))
    compile(done, "weight_utils", "exec")


def test_ple_drop_drops_each_ple_shard_one_file_behind(tmp_path):
    files = ["layer-00000-experts-0000-0127.safetensors", "model-bf16-00001.safetensors",
             "model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors", "model-plefp8-00002.safetensors"]
    events, logged = ple_drop_stream(tmp_path, files)
    order = [(e[0], e[1]) for e in events]
    # a PLE shard leaves once the next file has been read through (its tensors released); the last at the end
    assert order == [("read", files[0]), ("read", files[0]), ("read", files[1]), ("read", files[1]),
                     ("read", files[2]), ("read", files[2]), ("read", files[3]), ("read", files[3]),
                     ("drop", files[2]), ("read", files[4]), ("read", files[4]), ("drop", files[3]),
                     ("drop", files[4])]
    assert all(e[2:] == (0, 0, 4) for e in events if e[0] == "drop")
    assert logged and json.loads(logged[-1].split(" ", 1)[1]) == {"files": 3, "gib": 0.0}
    # the GPU ranks (not the offload process) and a start without the option drop nothing
    for kwargs in ({"offload": False}, {"pattern": None}):
        events, logged = ple_drop_stream(tmp_path, files, **kwargs)
        assert not [e for e in events if e[0] == "drop"] and not logged


def test_cgroup_memory_reads_v2_then_v1(tmp_path):
    from arc2 import kaggle

    assert kaggle.cgroup_memory(tmp_path) == {}
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory/memory.limit_in_bytes").write_text(str(1 << 62))
    (tmp_path / "memory/memory.usage_in_bytes").write_text(str(3 << 30))
    assert kaggle.cgroup_memory(tmp_path) == {"limit_gib": None, "used_gib": 3.0}
    (tmp_path / "memory.max").write_text(f"{120 << 30}\n")
    (tmp_path / "memory.current").write_text(f"{100 << 30}\n")
    (tmp_path / "memory.stat").write_text(f"anon {5 << 30}\nfile {94 << 30}\nfile_mapped 7\nshmem 0\n")
    assert kaggle.cgroup_memory(tmp_path) == {"limit_gib": 120.0, "used_gib": 100.0, "anon_gib": 5.0,
                                              "file_gib": 94.0, "shmem_gib": 0.0}
    record = {}
    kaggle.sample_memory(record, time.time(), every=0.01, count=3)
    time.sleep(0.5)
    assert len(record["memory"]) == 3 and all("elapsed" in row for row in record["memory"])


@pytest.mark.skipif(not os.environ.get("ARC_PINNED_SITE"), reason="needs the pinned runtime's weight_utils.py")
def test_ple_drop_applies_to_the_pinned_weight_loader(tmp_path):
    import shutil

    from arc2 import l4_ple_drop as drop

    site = tmp_path / "site"
    (site / drop.RELATIVE).parent.mkdir(parents=True)
    shutil.copyfile(Path(os.environ["ARC_PINNED_SITE"]) / drop.RELATIVE, site / drop.RELATIVE)
    record = drop.apply(site, tmp_path)
    assert record["changed"] and record["original_sha256"] == (
        "25bb3393105f9a642b7fa58bc277865db9976748be320787f8db2e03b16fd667")
    assert not drop.apply(site, tmp_path)["changed"]


def test_live_row_reads_as_an_arena_progress_row():
    from arc2 import kaggle

    record = {"stage": "play-2", "attempts": [{}, {}], "choice": {"utilization": "0.82"},
              "plays": [{"seconds": 970, "code": 1}]}
    row = {"elapsed_seconds": 4387.0, "completion_tokens": 1220316, "request_errors": 0, "test_outputs": 59,
           "gate": {"capacity": 292, "reserved": 271, "inflight": 15, "waiting": 3, "slots": 18},
           "final": {"score_outputs": 100 * 16 / 59, "verified_tasks": 0.35, "by_source": {
               "fallback": {"outputs": 28, "correct": 0}, "unverified": {"outputs": 11, "correct": 0},
               "verified": {"outputs": 20, "correct": 16}}}}
    out = kaggle.live_row(record, row, 7400.0, 1.0, [(5000.0, 700000), (5700.0, 800000)])
    assert out["play_seconds"] == 5357.0 and out["restarts"] == 1 and not out["finished"]
    engine, score = out["live"]["engine"], out["live"]["score"]
    assert engine["generated_tokens"] == 1220316 and abs(engine["tokens_per_second_mean"] - 1220316 / 5357) < 1e-9
    assert abs(engine["tokens_per_second_30m"] - (1220316 - 700000) / 2400) < 1e-9 and engine["running"] == 15
    assert (score["levels"], score["decided"], score["outputs"], score["verified_right"]) == (16, 31, 59, 16)
    assert score["at_elapsed_seconds"] == out["elapsed_seconds"] == 7400.0
    # before the first play and between two plays: the clock and the stage, nothing from a stale row
    assert "score" not in kaggle.live_row({"stage": "server-1"}, {}, 300.0, 1.0)["live"]
    waiting = kaggle.live_row({**record, "stage": "restart-1"}, row, 1200.0, 1.0)
    assert waiting["play_seconds"] == 970 and waiting["live"]["engine"]["running"] == 0


def test_package_driver_needs_only_the_standard_library(tmp_path):
    """The notebook's interpreter runs ``arc2.kaggle`` before any environment exists (rehearsal 1: no httpx)."""
    import subprocess
    import sys
    from pathlib import Path

    program = """
import importlib.abc, json, sys
allowed = set(sys.stdlib_module_names) | {"arc2", "arc_infra", "arena"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] not in allowed:
            raise ImportError("not the standard library: " + name)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
from arc2 import kaggle, l4_serving
task = {"train": [], "test": [{"input": [[1]]}]}
kaggle.tasks_written = kaggle.fallback({"a": task})
from arc2.tasks import write_json
write_json(sys.argv[2], kaggle.tasks_written)
assert kaggle.problems(json.load(open(sys.argv[2])), {"a": task}) == []
assert isinstance(kaggle.health("http://127.0.0.1:9/v1"), (int, str))
l4_serving.fallback(kaggle.OPTIONS, 2), l4_serving.log_first(sys.argv[3]), l4_serving.failure(sys.argv[3], "s", "e")
print("ok")
"""
    src = Path(__file__).resolve().parents[1] / "src"
    done = subprocess.run([sys.executable, "-c", program, str(src), str(tmp_path / "s.json"), str(tmp_path)],
                          capture_output=True, text=True)
    assert done.stdout.strip() == "ok", done.stderr[-800:]


def test_replay_summary():
    from arc2.trial import replay_summary

    items = [{"id": "a", "kind": "wrong", "samples": 2}, {"id": "b", "kind": "right"}]
    rows = [{"item": "a", "kind": "wrong", "outcome": "corrected", "right": True, "changed": True, "completion_tokens": 5},
            {"item": "a", "kind": "wrong", "outcome": "keep", "completion_tokens": 1},
            {"item": "b", "kind": "right", "outcome": "corrected", "right": False, "changed": True, "completion_tokens": 2}]
    out = replay_summary(rows, items)
    assert out["wrong"]["corrected_right"] == 1 and out["wrong"]["items_with_a_right_correction"] == 1
    assert out["wrong"]["outcomes"] == {"corrected": 1, "keep": 1} and out["wrong"]["completion_tokens"] == 6
    assert out["right"]["corrected_other_output_wrong"] == 1 and out["right"]["samples"] == 1


def test_budget_tiers_errors_and_context():
    run = TaskRun("t", TASK)
    eng, _ = engine([WRONG, WRONG, RuntimeError("500"), ContextFull("too long"), WRONG],
                    settings={"attempt_replies": 3})
    asyncio.run(eng.advance(run, 1500))  # checked between replies: two replies, then the tier is spent
    assert run.spent == 2000 and run.done is None and run.messages is not None
    eng.deadline = None
    asyncio.run(eng.advance(run, 2500))
    assert run.errors == 1 and run.attempts == 2 and run.spent == 3000 and run.done is None


def test_guess_ranking_and_fallback():
    def cand(code, passed, verified, grid, at):
        return {"code": code, "passed": passed, "pairs": 2, "accuracy": passed / 2, "verified": verified,
                "tests": [grid], "tokens_at": at}

    cands = [cand("a", 1, False, [[1]], 10), cand("b", 2, True, [[2]], 20), cand("c", 2, True, [[3]], 30),
             cand("d", 2, True, [[3]], 40), cand("e", 0, False, None, 5)]
    assert guesses(cands, [[[0]]]) == [[([[3]], "verified"), ([[2]], "verified")]]  # two programs agree on [[3]]
    assert guesses(cands, [[[0]]], limit=20) == [[([[2]], "verified"), ([[1]], "unverified")]]
    assert guesses(cands, [[[0]]], limit=5) == [[([[0]], "fallback")]]


def test_text_and_judge():
    assert tasks.extract_code("a\n```python\nx=1\n```\n" + RIGHT).startswith("def transform")
    assert tasks.extract_code("no code") is None and not tasks.valid([[1, 2], [3]]) and not tasks.valid([[10]])
    assert "Training pair 2\nInput (1x2):\n5 6\nOutput (2x1):\n5\n6" in tasks.task_prompt(TASK)
    assert "Test input 1 (1x3):\n7 8 9" in tasks.task_prompt(TASK) and "[[7], [8], [9]]" not in tasks.task_prompt(TASK)
    run = sandbox.run("import os\ndef transform(grid):\n    return grid", [[[1]]])
    assert "ImportError" in run["error"]
    verdict = tasks.judge(TASK, sandbox.run("def transform(g):\n    return g[1]", [p["input"] for p in TASK["train"]]
                                            + [TASK["test"][0]["input"]]))
    assert verdict["passed"] == 0 and not verdict["verified"] and verdict["tests"] == [None]


def test_reply_cap_arm_and_winning_arm(tmp_path):
    from arc2.trial import winning_arm

    caps = []
    eng, _ = engine([RIGHT, RIGHT])

    async def complete(task_id, messages, count, max_tokens, settings, inner=eng.complete):
        caps.append(max_tokens)
        return await inner(task_id, messages, count, max_tokens, settings)

    eng.complete = complete
    asyncio.run(eng.advance(TaskRun("t@a", TASK, {"name": "a", "cap": 8192}), 500))
    asyncio.run(eng.advance(TaskRun("t@b", TASK, {"name": "b"}), 500))
    assert caps == [8192, 16384]
    study = {"trials": [{"name": "tune", "arms": [{"name": "a"}, {"name": "b", "effort": "low"}, {"name": "c"}]}]}
    rows = {"a": (40.0, 25000), "b": (50.0, 28000), "c": (50.0, 21000)}
    (tmp_path / "tune").mkdir()
    (tmp_path / "tune/report.json").write_text(json.dumps({"per_arm": {
        k: {"curve": [{"budget": 30000, "score_outputs": v[0], "mean_tokens": v[1]}]} for k, v in rows.items()}}))
    assert winning_arm(study, tmp_path, trial="tune", budget=30000)["arm"] == {"name": "c"}


def test_roster_order_clock_and_marks():
    clock = token_clock([{"completion_tokens": 100, "t0": 0, "t1": 10}, {"completion_tokens": 100, "t0": 5, "t1": 15},
                         {"completion_tokens": 7, "t0": 20, "t1": 20}])
    assert [clock(t) for t in (-1, 5, 10, 15, 19, 20, 99)] == [0, 50, 150, 200, 200, 207, 207]
    # Two tasks, one reply at a time to the lower priority: a task is never done, a verified one waits its delay.
    eng, _ = engine([RIGHT, WRONG, WRONG, RIGHT, WRONG, RIGHT, WRONG, WRONG],
                    settings={"confirm_delay": 1500, "settled_delay": 2500})
    runs, order = [TaskRun("a", TASK), TaskRun("b", TASK)], []
    for _ in range(8):
        run = min(runs, key=eng.priority)
        order.append(run.id + ":" + eng.phase(run))
        asyncio.run(eng.advance(run, float("inf"), replies=1))
    assert order == ["a:open", "b:open", "b:open", "b:open", "a:confirm", "a:confirm", "b:confirm", "a:settled"]
    assert all(r.done is None for r in runs) and [r.spent for r in runs] == [4000, 4000]
    out = readout(runs)["where_tokens_go"]["by_phase"]
    assert {k: v["replies"] for k, v in out.items()} == {"open": 4, "confirm": 3, "settled": 1}
    # The roster clock: replies laid end to end, 1,000 tokens each; b's program came with the fourth reply.
    for k, x in enumerate(sorted((x for r in runs for x in r.replies), key=lambda x: x["t1"])):
        x["t0"], x["t1"] = float(k), k + 1.0
    for r in runs:
        for c in r.candidates:
            c["t"] = next(x["t1"] for x in r.replies if (x["attempt"], x["reply"]) == (c["attempt"], c["reply"]))
    out = roster(runs, {"low": 1000, "mid": 3500, "high": 4000, "far": 20000}, cut=6.5, step=2000)
    assert [out["marks"][k]["score_outputs"] for k in ("low", "mid", "high", "far")] == [50, 50, 100, 100]
    assert [out["marks"][k]["reached"] for k in ("low", "mid", "high", "far")] == [True, True, True, False]
    assert out["tokens"] == 8000 and out["exact_through"] == 6500 and out["marks"]["mid"]["tokens_per_task"] == 1750
    assert [c["tokens"] for c in out["curve"]][:2] == [2000, 4000] and len(out["curve"]) == 10
    assert runs[1].candidates[-1]["clock"] == 7000


def test_sticky_turn_ends_with_the_attempt():
    # The worker's turn with sticky attempts (arc2.trial): replies until the attempt ends. An attempt out of
    # replies ends the turn before the next attempt's first reply, so the unsolved task goes back into roster order.
    eng, _ = engine([WRONG, WRONG, RIGHT, RIGHT, WRONG, WRONG, RIGHT, RIGHT],
                    settings={"attempt_replies": 2, "confirm_delay": 0, "settled_delay": 2500})
    runs, order = [TaskRun("a", TASK), TaskRun("b", TASK), TaskRun("c", TASK)], []
    for _ in range(5):
        run = min(runs, key=eng.priority)
        while True:
            asyncio.run(eng.advance(run, float("inf"), replies=1))
            if run.done is not None or run.messages is None:
                break
        order.append((run.id, run.attempts, len(run.replies), eng.phase(run)))
    # a: two wrong programs, its turn ends; b and c have their first attempts before a's second.
    assert order == [("a", 1, 2, "open"), ("b", 1, 1, "confirm"), ("c", 1, 1, "confirm"),
                     ("b", 2, 3, "settled"), ("c", 2, 2, "settled")]
    assert [r.spent for r in runs] == [2000, 3000, 2000]


def test_l4_profile(tmp_path):
    from arc2 import l4_serving
    from arc2.trial import block_budget

    base = {"name": "b", "server": {"mtp_tokens": 1, "max_num_seqs": 80, "dense_kernels": {"x": 1}, "stack_kernels": {},
                                    "draft_head": {}, "expert_prune": {"mask": "m"}, "serving_kernels": {"ple_uva": True}},
            "cache_experiment": {"prefix_cache": {"a": 1}}, "arc2": {"data": "d"}, "generation_slots": 80,
            "active_limit": 76, "windows": [65536],
            "trials": [{"name": "tune", "split": "training", "tasks": [str(i) for i in range(48)], "server_env": {}}]}
    study = l4_serving.l4_study(base, streams=32, kv="int4")
    server = study["server"]
    assert server["mtp_tokens"] == 0 and server["max_num_seqs"] == study["generation_slots"] == 32
    assert server["cudagraph_capture_sizes"][-1] == server["max_cudagraph_capture_size"] == 32
    assert server["serving_kernels"] == {"mamba_ssm_cache_dtype": "bfloat16", "moe_backend": "humming",
                                         "expert_int3": 16, "kv_cache_4bit": "int4"}
    assert not {"dense_kernels", "stack_kernels", "draft_head"} & set(server) and server["expert_prune"] == {"mask": "m"}
    assert "prefix_cache" not in study["cache_experiment"] and study["arc2"]["cache_layout"] == "smallest"
    assert study["trials"][0]["arms"] == [{"name": "load", "cap": 1024}] and base["server"]["mtp_tokens"] == 1
    measure = l4_serving.l4_study(base, streams=16, kv="int4", text_only=True, prefix_caching=False, inventory=True)
    assert measure["trials"][0]["server_env"]["ARENA_PREFIX_CACHING"] == "0" and measure["arc2"]["l4x4"]["text_only"]
    assert "ARENA_PREFIX_CACHING" not in study["trials"][0]["server_env"] and not study["arc2"]["l4x4"]["inventory"]
    assert l4_serving.l4_study(base, expert_parallel=True)["arc2"]["l4x4"]["expert_parallel"]
    # The profile start and the window setting: the profiler, graphs from 8 rows, the tower off; the loop's window.
    traced = l4_serving.l4_study(base, streams=32, kv="int4", text_only=True, lean_graphs=True, profile=True,
                                 probe_decode=[8, 16, 24, 32], window=49152)
    assert traced["trials"][0]["server_env"]["ARENA_TORCH_PROFILER"] == "1" and traced["windows"] == [49152]
    assert traced["arc2"]["profile_rows"] == [16, 24, 32] and traced["arc2"]["probe_decode"] == [8, 16, 24, 32]
    assert traced["arc2"]["l4x4"]["max_model_len"] == 65536 and study["windows"] == [65536]
    assert "arc2_l4_stacks" in l4_serving.plugin_files(traced["arc2"]["l4x4"])["entry_points.txt"]
    assert "ARENA_TORCH_PROFILER" not in study["trials"][0]["server_env"] and "profile_rows" not in study["arc2"]
    assert l4_serving.l4_study(base, kv="int4", grow_wait=0)["arc2"]["grow_wait"] == 0 and "grow_wait" not in study["arc2"]
    from arc2 import kaggle

    assert kaggle.OPTIONS["grow_wait"] == 0  # the package: no wait for blocks (study_for passes it to l4_study)
    from arc2 import l4_allreduce

    assert l4_allreduce.arguments(["--out", "x", "--variants"]).variants  # refused by the first profile job
    assert not l4_allreduce.arguments(["--out", "x"]).variants
    from arc2 import l4_dense_bench

    assert 2.8 < l4_dense_bench.step_bytes() / 2**30 < 2.95 and l4_dense_bench.step_bytes(8) * 2 == l4_dense_bench.step_bytes()
    assert l4_dense_bench.arguments(["--out", "x"]).rows == "16,24,32"
    from arc2 import l4_dense

    dense = l4_serving.l4_study(base, kv="int4", dense_int8=True)["arc2"]["l4x4"]
    assert dense["dense_int8"] and not study["arc2"]["l4x4"]["dense_int8"]
    assert l4_serving.l4_study(base, dense_int8=["output_head", "gdn_qkvz"])["arc2"]["l4x4"]["dense_int8"] == [
        "gdn_qkvz", "output_head"]
    lean = l4_serving.l4_study({**base, "server": {**base["server"], "max_num_batched_tokens": 4096}}, kv="int4",
                               tower_off=True, prefill_chunk=2048)
    # the loader takes 4,096 or 8,192: the chunk goes to the recipe's environment in patch_recipe, not the study
    assert lean["server"]["max_num_batched_tokens"] == 4096 and lean["arc2"]["l4x4"]["prefill_chunk"] == 2048
    assert lean["arc2"]["l4x4"]["tower_off"]
    tl = l4_serving.l4_study(base, kv="int4", tower_lean=True)["arc2"]["l4x4"]
    assert tl["tower_lean"] and not tl["tower_off"] and "arc2.l4_tower:register" not in l4_serving.plugin_files(tl).get("entry_points.txt", "")
    assert "arc2.l4_tower:register" in l4_serving.plugin_files(lean["arc2"]["l4x4"])["entry_points.txt"]
    assert l4_serving.start_fallback({"tower_off": True, "utilization": "0.93"}) == {
        "tower_off": False, "text_only": False, "embed_host": False, "utilization": "0.93"}
    with pytest.raises(ValueError):
        l4_serving.l4_study(base, prefill_chunk=64)
    reserved = l4_serving.l4_study(base, kv="int4", reply_reserve=4096)
    assert reserved["arc2"]["reply_reserve"] == 4096 and "reply_reserve" not in l4_serving.l4_study(base)["arc2"]
    with pytest.raises(ValueError):
        l4_serving.l4_study(base, reply_reserve=100)
    topk = l4_serving.l4_study(base, kv="int4", logits_topk=64)
    assert topk["arc2"]["l4x4"]["logits_topk"] == 64 and l4_serving.l4_study(base)["arc2"]["l4x4"]["logits_topk"] == 0
    assert "arc2.l4_logits:register" in l4_serving.plugin_files({"logits_topk": 64})["entry_points.txt"]
    with pytest.raises(ValueError):
        l4_serving.l4_study(base, logits_topk=8)
    assert l4_serving.state_block_tokens() == l4_serving.state_block_tokens(1) == 1600
    roster_base = {**base, "trials": [{**base["trials"][0], "arms": [{"name": "tool", "cap": 16384}]}]}
    wide = l4_serving.l4_study(roster_base, kv="int4", streams=64, tasks=48, roster={"arm": "tool", "rates": [330]})
    assert wide["generation_slots"] == 64
    small = l4_serving.l4_study(base, kv="int4", block_tokens=1600)
    assert small["arc2"]["l4x4"]["block_tokens"] == 1600 and l4_serving.l4_study(base)["arc2"]["l4x4"]["block_tokens"] is None
    long = l4_serving.l4_study(base, kv="int4", context=131072)
    assert long["windows"] == [131072] and long["arc2"]["l4x4"]["context"] == 131072
    assert l4_serving.l4_study(base)["windows"] == [65536]
    for bad in ({"kv": "int4", "block_tokens": 1536}, {"kv": "int4", "block_tokens": 1650}, {"block_tokens": 1600},
                {"context": 100000}, {"window": 98304}):
        with pytest.raises(ValueError):
            l4_serving.l4_study(base, **bad)
    assert "arc2.l4_dense:register" in l4_serving.plugin_files(dense)["entry_points.txt"]
    assert l4_dense.family("language_model.model.layers.3.hyper_connection.input_mix_weight_up") == "hc_up"
    assert l4_dense.family("model.layers.3.linear_attn.out_proj") == l4_dense.family("m.layers.7.self_attn.o_proj")
    assert l4_dense.family("lm_head") == l4_dense.family("language_model.lm_head") == "output_head"
    assert not any(l4_dense.family(name) for name in (
        "model.layers.3.linear_attn.in_proj_ba", "model.layers.7.self_attn.indexer.index_qk_proj",
        "language_model.mtp.layers.0.self_attn.o_proj", "visual.blocks.0.self_attn.qkv_proj",
        "model.layers.3.mlp.shared_expert.gate_up_proj", "mtp.lm_head"))
    assert {label for _, label in l4_dense.SUFFIXES} < {name for name, *_ in l4_dense_bench.SHAPES}
    assert l4_dense.label_selects("extra-nested_288_dense8", "hc_up") and not l4_dense.label_selects("extra-nested_288", "hc_up")
    assert l4_dense.label_selects("x_dense8-hc", "hc_up") and not l4_dense.label_selects("x_dense8-hc", "gdn_qkvz")
    assert l4_dense.label_selects("x_dense8-nohc", "gdn_qkvz") and not l4_dense.label_selects(None, "gdn_qkvz")
    keys = l4_serving.l4_study(base, kv="int4", embed_host=True, indexer_fp8=True)["server"]["serving_kernels"]
    assert keys["embed_host"] and keys["indexer_fp8"] and "embed_host" not in study["server"]["serving_kernels"]
    assert not study["arc2"]["l4x4"]["expert_parallel"]
    scored = l4_serving.l4_study({**base, "trials": [{**base["trials"][0], "arms": [{"name": "x8", "cap": 8192}]}]},
                                 kv="int4", expert_parallel=True, roster={"arm": "x8", "rates": [180, 130]},
                                 prefill_slots=2)
    tasks_n = len(scored["trials"][0]["tasks"])
    assert scored["trials"][0]["marks"] == {"130": 130 * 180 * tasks_n, "180": 180 * 180 * tasks_n}
    assert scored["trials"][0]["arms"] == [{"name": "x8", "cap": 8192}] and scored["arc2"]["prefill"]["slots"] == 2
    again = l4_serving.fallback({"expert_parallel": True, "utilization": "0.87", "streams": 32}, 3)
    assert again["utilization"] == "0.84" and again["prefill_slots"] == 3 and again["streams"] == 32
    assert l4_serving.fallback({"expert_parallel": True, "utilization": "0.87"}, 0) is None
    patched = {**base, "cache_experiment": {**base["cache_experiment"], "prefix_cache": {
        "mamba_align_fix": True, "mamba_keep_state": True, "request_end_tail": True, "spec_decode": True,
        "state_cotenant": True, "admission_tick": 2.0}}}
    kept = l4_serving.l4_study(patched, kv="int4", prefix_patches=True)["cache_experiment"]["prefix_cache"]
    assert kept == {"mamba_align_fix": True, "mamba_keep_state": True, "request_end_tail": True}
    assert "prefix_cache" not in l4_serving.l4_study(patched, kv="int4")["cache_experiment"]
    drafted = l4_serving.l4_study(patched, streams=16, kv="int4", prefix_patches=True, draft=1)
    assert drafted["server"]["mtp_tokens"] == 1 and drafted["server"]["draft_head"] == {}
    assert drafted["server"]["max_cudagraph_capture_size"] == drafted["server"]["cudagraph_capture_sizes"][-1] == 32
    assert drafted["server"]["serving_kernels"] == server["serving_kernels"] and drafted["arc2"]["l4x4"]["draft"] == 1
    assert set(drafted["cache_experiment"]["prefix_cache"]) == set(kept) | {"spec_decode"}
    assert "dense_kernels" not in drafted["server"] and study["arc2"]["l4x4"]["draft"] == 0
    mtp3 = l4_serving.l4_study(patched, streams=48, kv="int4", prefix_patches=True, draft=3, block_tokens=1664,
                               lean_graphs=True)  # MTP 3 on L4
    assert mtp3["server"]["mtp_tokens"] == 3 and mtp3["server"]["max_cudagraph_capture_size"] == 192
    assert mtp3["server"]["cudagraph_capture_sizes"][-1] == 192 and l4_serving.state_block_tokens(3) == 1664
    assert set(mtp3["cache_experiment"]["prefix_cache"]) == set(drafted["cache_experiment"]["prefix_cache"])
    with pytest.raises(ValueError):
        l4_serving.l4_study(patched, streams=48, kv="int4", draft=3, block_tokens=1600)
    with pytest.raises(ValueError):
        l4_serving.l4_study(patched, streams=80, kv="int4", draft=3)
    dense = l4_serving.candidate_study({**base, "server": {**base["server"], "max_num_batched_tokens": 4096}}, "q",
                                       streams=32, profile={"model": "m", "revision": "r", "effort": "xhigh",
                                                            "mtp_tokens": 3})
    assert dense["server"]["max_cudagraph_capture_size"] == 128 and dense["server"]["max_num_seqs"] == 32
    from arc2 import l4_inventory

    assert l4_inventory.collapse("m.layers.17.mlp.experts.w13.weight.x") == "m.layers.*.mlp.experts.w13"
    choice = {"streams": 16, "kv": "int4", "utilization": "0.90"}
    after = l4_serving.fallback(choice, 12)  # the engine died above 12 requests at once
    assert after == {"streams": 12, "kv": "int4", "utilization": "0.87", "slots": 12}
    assert l4_serving.fallback(choice, 4)["streams"] == 8 and l4_serving.fallback(choice, 4)["slots"] == 4
    assert l4_serving.fallback(choice, 0)["kv"] == "bf16" and l4_serving.fallback({**choice, "utilization": "0.88"}, 8)["utilization"] == "0.87"
    again = l4_serving.start_fallback({**choice, "text_only": True, "utilization": "0.93"})
    assert not again["text_only"] and again["utilization"] == "0.93"  # the tower first, then the share
    assert l4_serving.start_fallback(again)["utilization"] == "0.90" and l4_serving.start_fallback(choice) == choice
    fewer = l4_serving.l4_study(base, **l4_serving.fallback(choice, 4))
    assert fewer["server"]["max_num_seqs"] == 8 and fewer["generation_slots"] == fewer["active_limit"] == 4
    body = tasks.chat_request("m", [{"role": "user", "content": "x"}], 7, temperature=0.7, effort="xhigh")
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"} and not body["stream"]
    log = tmp_path / "vllm-openai-server.log"
    log.write_text("Traceback before ready\nOutOfMemoryError early\nINFO: Application startup complete.\nfine\n"
                   "[rank0] memory allocation failed with OOM\nERROR torch.OutOfMemoryError: CUDA out of memory\n")
    assert [line.split()[0] for line in l4_serving.log_first(tmp_path)] == ["[rank0]", "ERROR"]
    log.unlink()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": {
        "num_attention_heads": 24, "num_key_value_heads": 2, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "indexer_n_heads": 4, "indexer_kv_heads": 1, "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640}}))
    assert l4_serving.head_counts(config)["all"] and not l4_serving.head_counts(config, ranks=5)["all"]

    def layout(blocks):
        return {"schema": 1, "num_blocks": blocks, "max_model_len": 8192, "lookahead": 0,
                "costs": [{"tokens": 4096 * k, "blocks": k + 2, "groups": [k, 2]} for k in (1, 2)]}
    (tmp_path / "server.log").write_text("".join(f"ARC_CACHE_LAYOUT {json.dumps(layout(n))}\n" for n in (100, 90)))
    assert block_budget(tmp_path, 1.0, smallest=True).capacity == 90
    facts = l4_serving.server_facts(tmp_path, window=8192)
    assert facts["cache"]["streams_that_fit"] == 22 and facts["cache"]["num_blocks"] == [90, 100]
    try:
        block_budget(tmp_path, 1.0)
        raise AssertionError("the arena's loader refuses differing layouts")
    except RuntimeError:
        pass


def test_nested_mask_is_served_from_the_overlay_of_the_wider_one(tmp_path, monkeypatch):
    import hashlib

    import arena.expert_overlay as overlay
    from arc2 import l4_serving

    def mask(name, ids):
        path = tmp_path / "configs/expert-prune" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "arc-expert-prune-v1", "experts": 16, "fraction": len(ids) / 16,
                                    "pruned_per_layer": len(ids), "pruned": {str(k): ids for k in range(48)}}))
        return {"mask": f"configs/expert-prune/{name}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    wide, deep, other = mask("wide.json", [2, 5]), mask("deep.json", [1, 2, 5]), mask("other.json", [1, 2, 7])
    assert l4_serving.within({"0": [2, 5]}, {"0": [1, 2, 5]}) == 1 and l4_serving.within({"0": [2]}, {"0": [1]}) is None
    base, shards = tmp_path / "base", tmp_path / "overlay"
    base.mkdir()
    shards.mkdir()
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "layer.safetensors"}}))
    (base / "layer.safetensors").write_bytes(b"x" * 8)
    (shards / "layer.safetensors").write_bytes(b"y" * 5)
    (shards / overlay.MANIFEST).write_text(json.dumps({
        "format": overlay.FORMAT, "complete": True, "omits_pruned": True, "expert_prune_sha256": wide["sha256"],
        "shards": {"layer.safetensors": {"bytes": 5, "sha256": hashlib.sha256(b"y" * 5).hexdigest()}}}))
    digest = hashlib.sha256((shards / overlay.MANIFEST).read_bytes()).hexdigest()

    def farm(sha):
        return overlay.materialise(base, shards, tmp_path / "farm", digest, log=lambda *_: None, expert_prune_sha256=sha)

    monkeypatch.setattr(overlay, "materialise", overlay.materialise)  # put back after the test
    for sha in (deep["sha256"], other["sha256"]):  # the arena's loader: the overlay's own mask only
        try:
            farm(sha)
            raise AssertionError("another mask than the overlay's was taken")
        except RuntimeError:
            pass
    study = {"arc2": {"nested_mask": wide}, "server": {"expert_prune": deep}}
    assert l4_serving.nest_overlay({"arc2": {}, "server": {"expert_prune": deep}}, tmp_path) is None
    try:  # a mask that keeps an expert the overlay omits
        l4_serving.nest_overlay({"arc2": {"nested_mask": wide}, "server": {"expert_prune": other}}, tmp_path)
        raise AssertionError("a mask outside the overlay's was taken")
    except RuntimeError:
        pass
    assert l4_serving.nest_overlay(study, tmp_path) == 48
    record = farm(deep["sha256"])
    assert record["nested_mask"]["experts_pruned_beyond_the_overlay"] == 48 and record["omits_pruned"]
    assert (tmp_path / "farm/layer.safetensors").read_bytes() == b"y" * 5 and "nested_mask" not in farm(wide["sha256"])
    try:
        farm(other["sha256"])
        raise AssertionError("another mask than the served one was taken")
    except RuntimeError:
        pass


def test_l4_text_check_and_finite_report(tmp_path):
    import random

    from arc2 import l4_serving

    rng = random.Random(1)
    words = ["grid", "shape", "moves", "wall", "colour", "row", "column", "until", "then", "the", "smaller", "pair"]
    sound = " ".join(rng.choice(words) + str(rng.randint(0, 9)) for _ in range(900))
    assert l4_serving.degenerate(sound, 900) is None and l4_serving.degenerate(sound[:900], 200) is None
    assert l4_serving.degenerate(" eight" * 1200, 1200) == "loop" and l4_serving.degenerate(" eight" * 120, 120) == "loop"
    assert l4_serving.degenerate("", 3, 0) == "empty" and l4_serving.degenerate("ok", 2) is None
    assert l4_serving.plugin_files({}) == {}
    both = l4_serving.plugin_files({"inventory": True, "finite_check": True})["entry_points.txt"]
    assert "arc2.l4_inventory:register" in both and "arc2.l4_finite:register" in both
    load = {"kind": "load", "layer": 3, "tensors": {"w1": {"shape": [80, 4], "dtype": "int32", "zero": 2}}}
    event = {"kind": "moe", "t": 5.0, "step": 1, "layer": 6, "rows": 600, "bad_rows": 3, "suspects": [[71, 3, 0]]}
    later = {"kind": "input", "t": 5.5, "step": 1, "layer": 7, "rows": 600, "bad_rows": 3}
    nan = {"k_peak_to_rms": float("nan"), "layer_order": 6, "sample": 1}
    (tmp_path / "vllm-openai-server.log").write_text("\n".join([
        "(Worker_TP1_EP1 pid=9) ARC2_FINITE " + json.dumps(later), "(Worker_TP2_EP2 pid=8) ARC2_FINITE " + json.dumps(event),
        "(Worker_TP2_EP2 pid=8) ARC2_FINITE " + json.dumps(load), "(Worker_TP0_EP0 pid=7) ARC2_FINITE_ERROR \"boom\"",
        "(Worker_TP0_EP0 pid=7) ARC_KV4_ROUNDTRIP " + json.dumps(nan),
        "(Worker_TP0_EP0 pid=7) ARC_KV4_ROUNDTRIP " + json.dumps({"k_peak_to_rms": 3.2, "layer_order": 0})]))
    report = l4_serving.finite_report(tmp_path)
    assert report["moe_events"] == 1 and report["input_events"] == 1 and report["first"]["worker"] == "Worker_TP2_EP2"
    assert report["first"]["layer"] == 6 and report["load"][0]["odd"] and report["errors"]
    assert report["kv_roundtrip"] == {"lines": 2, "nan": 1, "first_layer": 6}


def test_kv4_rotation_guard(monkeypatch):
    """The 4-bit cache's rotation guard: off = the plain call; on = whole tiles and no non-finite row."""
    import sys
    import types

    torch = pytest.importorskip("torch")
    if "triton" not in sys.modules:
        try:
            import triton  # noqa: F401
        except ImportError:  # the guard is plain torch: a stand-in lets the module load
            lang = types.ModuleType("triton.language")
            lang.__getattr__ = lambda name: object
            stub = types.ModuleType("triton")
            stub.jit, stub.language = (lambda f: f), lang
            monkeypatch.setitem(sys.modules, "triton", stub)
            monkeypatch.setitem(sys.modules, "triton.language", lang)
    from arc_infra import kv4_runtime as kv4

    seen = []

    def tail_fault(x, inverse=False):  # a transform that ruins the last vector when the count is not whole tiles
        seen.append(x.shape[0])
        out = kv4.rht_reference(x, inverse)
        if x.shape[0] % kv4.ROTATION_TILE:
            out[-1, 3] = float("inf")
        return out

    x = torch.randn(37, 1, 256)
    monkeypatch.delenv(kv4.GUARD_ENV, raising=False)
    assert kv4._rotation() is kv4.plain_rht and not kv4.guard_enabled()
    assert torch.equal(kv4.plain_rht(x.bfloat16()), kv4.rht(x.bfloat16().float()))  # the call as it was
    assert not torch.isfinite(tail_fault(x.reshape(-1, 256))).all()
    monkeypatch.setenv(kv4.GUARD_ENV, "1")
    assert kv4._rotation() is kv4.guarded_rht
    out = kv4.guarded_rht(x, transform=tail_fault)
    assert seen[-1] == 48 and out.shape == x.shape and torch.isfinite(out).all()
    assert torch.allclose(out, kv4.rht_reference(x), atol=1e-4)
    assert torch.allclose(kv4.guarded_rht(out, inverse=True, transform=tail_fault) / 256, x, atol=1e-3)
    bad = torch.randn(128, 2, 256)
    bad[5, 1, 7] = float("nan")
    kv4._STATE.pop("guard", None)
    fixed = kv4.guarded_rht(bad)
    assert torch.isfinite(fixed).all() and not fixed[5, 1].any() and torch.allclose(fixed[4], kv4.rht_reference(bad[4]))
    assert kv4._STATE["guard"]["calls"] == 1 and int(kv4._STATE["guard"]["zeroed"]) == 1
    half = torch.randn(70, 6, 256).bfloat16()  # the query as the op hands it over: converted and padded in one buffer
    assert torch.equal(kv4.guarded_rht(half, transform=tail_fault), kv4.rht_reference(half.float()))
    assert seen[-1] == 432 and torch.equal(half, half.clone()) and kv4.guarded_rht(half).dtype == torch.float32


def test_reply_reserve_holds_less_than_the_cap():
    import asyncio

    from arc2 import trial

    class Gate:
        def __init__(self):
            self.asked = []

        async def acquire(self, task, tokens):
            self.asked.append(tokens)
            return trial.Reservation(task, tokens) if hasattr(trial, "Reservation") else object()

        def release(self, reservation):
            pass

    class Reply:
        status_code, text = 200, ""

        def json(self):
            return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}

        def raise_for_status(self):
            pass

    class Client:
        async def post(self, *a, **k):
            return Reply()

    for reserve, want in ((None, 1000 + 8192), (4096, 1000 + 4096), (16384, 1000 + 8192)):
        gate = Gate()
        t = trial.Transport(Client(), gate, model="m", effort="xhigh", temperature=0.7,
                            deadline=__import__("time").monotonic() + 60, reserve=reserve)
        t.release = lambda r: None
        asyncio.run(t.complete("a", [{"role": "user", "content": "x"}], 1000, 8192, {}))
        assert gate.asked == [want]


def test_block_tokens_rewrites_the_one_card_block(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from arc2 import l4_serving

    argv = ["vllm", "serve", "--tensor-parallel-size", "1", "--kv-cache-dtype", "int4_per_token_head",
            "--block-size", "3136"]
    recipe = SimpleNamespace(MAX_NUM_BATCHED_TOKENS_ENV="X", runtime_environment=lambda: ({}, {}),
                             server_command=lambda: list(argv), update_ready_identity=lambda i: i, main=lambda: None)
    monkeypatch.setenv(l4_serving.OPTIONS_ENV, json.dumps({"block_tokens": 1600}))
    l4_serving.patch_recipe(recipe, tmp_path, {})
    served = recipe.server_command()
    assert served[served.index("--block-size") + 1] == "1600" and served[served.index("--tensor-parallel-size") + 1] == "4"
    monkeypatch.setenv(l4_serving.OPTIONS_ENV, json.dumps({}))
    recipe.server_command = lambda: list(argv)
    l4_serving.patch_recipe(recipe, tmp_path, {})
    assert recipe.server_command()[-1] == "3136"


def test_carry_forward_and_other_reading():
    """carry: an attempt without a verified program ends on a short note; the next starts from the notes and the
    best program with its differences. confirm_style "other": the attempt after a verified program asks for the
    other reading."""
    run = TaskRun("t", TASK, {"carry": True, "confirm_style": "other"})
    eng, sent = engine([WRONG, WRONG, "Transposing is ruled out? Try columns.", RIGHT, OTHER],
                       settings={"attempt_replies": 2, "confirm_attempts": 1})
    asyncio.run(eng.advance(run, 300000))
    assert [r["outcome"] for r in run.replies] == ["wrong", "wrong", "note", "verified", "verified"]
    assert sent[2][-1]["content"] == tasks.NOTE and run.notes == ["Transposing is ruled out? Try columns."]
    second = sent[3][1]["content"]
    assert second.startswith(tasks.task_prompt(TASK)) and "- Transposing is ruled out" in second
    assert "passes 0 of 2 training pairs" in second and "print('hi')" in second and len(sent[3]) == 2
    third = sent[4][1]["content"]
    assert "already been found" in third and "(3x1):\n7\n8\n9" in third and run.done == "verified"
    plain = TaskRun("t", TASK)  # the defaults: no note, fresh attempts
    eng, sent = engine([WRONG, WRONG, RIGHT, OTHER], settings={"attempt_replies": 2})
    asyncio.run(eng.advance(plain, 300000))
    assert [r["outcome"] for r in plain.replies] == ["wrong", "wrong", "verified", "verified"]
    assert sent[2][1]["content"] == tasks.task_prompt(TASK) and sent[3][1]["content"] == tasks.task_prompt(TASK)


def test_arc2_prompt_setting():
    run = TaskRun("t", TASK, {"prompt": "arc2"})
    eng, sent = engine([RIGHT, OTHER])
    asyncio.run(eng.advance(run, 300000))
    assert sent[0][0]["content"] == tasks.SYSTEM + tasks.ARC2_HINT and run.done == "verified"


def test_lessons_shared_across_tasks_not_copies():
    """lessons: the reply after a task's first verified program is a lesson; attempts on other tasks start with the
    latest lessons, never the same task's other roster copy; the lesson reply is not a candidate."""
    arm = {"lessons": True}
    eng, sent = engine([RIGHT, "Check transposes first.", OTHER, RIGHT, "x", RIGHT, "y"],
                       settings={"confirm_attempts": 1})
    a = TaskRun("a#1", TASK, arm)
    asyncio.run(eng.advance(a, 300000))
    assert [r["outcome"] for r in a.replies] == ["verified", "lesson", "verified"]
    assert sent[1][-1]["content"] == tasks.LESSON and eng.lessons == [("a", "Check transposes first.")]
    assert len(a.candidates) == 2 and "Lessons noted" not in sent[2][1]["content"]
    twin = TaskRun("a#2", TASK, arm)  # the other copy of the same task
    asyncio.run(eng.advance(twin, 300000, replies=2))
    assert "Lessons noted" not in sent[3][1]["content"]
    other = TaskRun("b#1", TASK, arm)
    asyncio.run(eng.advance(other, 300000, replies=1))
    assert sent[5][1]["content"].startswith(tasks.LESSONS.format(lessons="- Check transposes first.\n- x"))


def test_lesson_after_last_reply_and_minds():
    """A program verified on the attempt's last reply still gets its lesson; minds 2 switches every second
    attempt's system prompt."""
    eng, sent = engine([WRONG, RIGHT, "lesson"], settings={"attempt_replies": 2})
    run = TaskRun("t", TASK, {"lessons": True, "carry": True})
    asyncio.run(eng.advance(run, 300000, replies=3))
    assert [r["outcome"] for r in run.replies] == ["wrong", "verified", "lesson"] and run.first_verified_attempt == 1
    eng, sent = engine([WRONG, WRONG, RIGHT, OTHER], settings={"attempt_replies": 1})
    run = TaskRun("t", TASK, {"minds": 2})
    asyncio.run(eng.advance(run, 300000))
    systems = [m[0]["content"] for m in sent]
    assert systems[0] == tasks.SYSTEM and systems[1] == tasks.SYSTEM + tasks.MIND_B and systems[2] == tasks.SYSTEM


def test_scheduler_keys_from_the_arm():
    """settled_delay / confirm_attempts set on the arm (a roster `set`) override the study's."""
    eng, _ = engine([RIGHT, OTHER, RIGHT], settings={"settled_delay": 100000, "confirm_attempts": 1})
    run = TaskRun("t", TASK, {"settled_delay": None})
    asyncio.run(eng.advance(run, 300000))
    assert run.done == "verified" and len(run.replies) == 2
    eng, _ = engine([RIGHT, OTHER, RIGHT], settings={"settled_delay": 100000, "confirm_attempts": 1})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000, replies=3))
    assert run.done is None and eng.priority(run) == run.spent + 100000


def test_helpers_preloaded():
    from arc2 import sandbox
    grid = [[0, 1, 0, 0], [0, 1, 0, 2], [0, 0, 0, 2], [3, 0, 0, 0]]
    code = ("def transform(grid):\n    obs = objects(grid)\n    big = max(obs, key=lambda o: o['size'])\n"
            "    return crop(grid, big['box']).tolist()")
    assert sandbox.run(code, [grid], strict=False, helpers=True)["outputs"] == [[[1], [1]]]
    assert "error" in sandbox.run(code, [grid], strict=False)["outputs"][0]  # off: no helpers
    out = sandbox.run_cells([], "result = (period([[1,2,1,2],[3,4,3,4]]), symmetries([[1,1],[1,1]])[:2], "
                            "complete([[1,2,1,2],[3,0,3,4]], 0).tolist(), panels([[1,5,2],[5,5,5],[3,5,4]])[3][2].tolist(),"
                            " int(enclosed([[1,1,1],[1,0,1],[1,1,1]], 0).sum()), downscale(upscale([[1,2]], 2), 2).tolist())",
                            {"train": [], "test": []}, strict=False, helpers=True)
    assert out["result"] == "((2, 2), ['identity', 'rot90'], [[1, 2, 1, 2], [3, 4, 3, 4]], [[4]], 1, [[1, 2]])", out


def test_direct_attempt():
    # Every 2nd attempt of an open task answers the grids itself (no program, no tool); its grid is a candidate
    # ranked after verified programs and before unverified ones, and the task's source label is "direct".
    run = TaskRun("t", TASK)
    eng, sent = engine([WRONG, "The rule transposes.\n```\n7\n8\n9\n```", RIGHT],
                       settings={"direct_every": 2, "attempt_replies": 1})
    asyncio.run(eng.advance(run, 300000))
    assert [r["outcome"] for r in run.replies][:3] == ["wrong", "direct", "verified"]
    assert sent[1][0]["content"] == tasks.DIRECT_SYSTEM and run.candidates[1]["direct"]
    assert run.candidates[1]["tests"] == [[[7], [8], [9]]]
    assert guesses(run.candidates[:2], [TASK["test"][0]["input"]])[0][0] == ([[7], [8], [9]], "direct")
    assert tasks.direct_grids("```\n1 2\n```\n```\nx\n```", 2) == [[[1, 2]], None]
    assert tasks.direct_grids("no grid", 1) is None


def test_direct_after():
    # An attempt that ends without a verified program gets one more reply: the test grids without a program.
    run = TaskRun("t", TASK)
    eng, sent = engine([WRONG, "```\n7\n8\n9\n```", RIGHT], settings={"direct_after": True, "attempt_replies": 1})
    asyncio.run(eng.advance(run, 300000))
    assert [r["outcome"] for r in run.replies][:3] == ["wrong", "direct", "verified"]
    assert sent[1][-1]["content"] == tasks.DIRECT_AFTER and len(sent[1]) == 5  # same conversation, then fresh
    assert run.candidates[1]["direct"] and run.candidates[1]["tests"] == [[[7], [8], [9]]]


def test_views_prompt():
    # views: object summary after each grid, changes after a same-shape pair; off by default (prompt unchanged).
    task = {"train": [{"input": [[0, 1], [0, 0]], "output": [[0, 1], [1, 0]]}], "test": [{"input": [[2, 0], [0, 0]]}]}
    plain, shown = tasks.task_prompt(task), tasks.task_prompt(task, views=True)
    assert "Objects" not in plain and plain in shown.replace("\nObjects (background 0, 1): 1@(0,1)-(0,1)x1", "", 1).replace(
        "\nObjects (background 0, 2): 1@(0,1)-(0,1)x1 1@(1,0)-(1,0)x1", "").replace(
        "\nChanges: 1 of 4 cells; 0->1 x1\nChanged cells (row,column): (1,0)", "").replace(
        "\nObjects (background 0, 1): 2@(0,0)-(0,0)x1", "")
    assert "Changes: 1 of 4 cells; 0->1 x1" in shown and "2@(0,0)-(0,0)x1" in shown


def test_direct_after_no_room():
    # No room left for the grid reply in the attempt: the next attempt is a fresh one-reply direct attempt.
    run = TaskRun("t", TASK)
    eng, sent = engine([WRONG, "```\n7\n8\n9\n```", RIGHT],
                       settings={"direct_after": True, "attempt_replies": 1, "direct_tokens": 10 ** 6})
    asyncio.run(eng.advance(run, 300000))
    assert [r["outcome"] for r in run.replies][:3] == ["wrong", "direct", "verified"]
    assert sent[1][0]["content"] == tasks.DIRECT_SYSTEM and len(sent[1]) == 2  # fresh direct conversation
    assert run.candidates[1]["direct"] and run.candidates[1]["attempt"] == 2


def test_views_matches():
    # views 2: an input object of 3+ cells found again in the output, moved and recoloured.
    a = [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]
    b = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 2, 2]]
    line = tasks.matches_view(a, b)
    assert "0 in place, 1 moved or recoloured" in line and "1@(1,0)x3 moved (+1,+1) recoloured 1->2" in line
    task = {"train": [{"input": a, "output": b}], "test": [{"input": a}]}
    assert "Object matches" in tasks.task_prompt(task, views=2) and "Object matches" not in tasks.task_prompt(task, views=True)


def test_confirm_effort():
    # confirm_effort: attempts after the first verified program are sent at that effort; open attempts are not.
    efforts = []
    run = TaskRun("t", TASK)
    eng, sent = engine([RIGHT, RIGHT], settings={"confirm_effort": "low", "attempt_replies": 1, "confirm_attempts": 1})
    inner = eng.complete

    async def spy(task_id, messages, count, max_tokens, settings):
        efforts.append(settings.get("effort"))
        return await inner(task_id, messages, count, max_tokens, settings)
    eng.complete = spy
    asyncio.run(eng.advance(run, 300000))
    assert efforts[:2] == [None, "low"]


def test_images_prompt():
    # images: the text prompt first, then one picture a training pair (input | output) and a test input.
    eng, sent = engine([RIGHT], settings={"images": True})
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    parts = sent[0][1]["content"]
    assert parts[0] == {"type": "text", "text": tasks.task_prompt(TASK)}
    urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert len(urls) == len(TASK["train"]) + len(TASK["test"])
    from arc_infra.model import png_size
    w, h = png_size(urls[0])
    g, o = TASK["train"][0]["input"], TASK["train"][0]["output"]
    cell = max(14, min(28, tasks.IMAGE_SIDE // max(len(g), len(g[0]), len(o), len(o[0]))))
    assert (w, h) == ((len(g[0]) + len(o[0]) + 1) * cell + 2, max(len(g), len(o)) * cell + 1)
    eng, sent = engine([RIGHT])
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    assert isinstance(sent[0][1]["content"], str)


def test_pictures_first_message_has_no_text_grids():
    # pictures (with images and tool): sizes only in the text, the pictures note, the tool globals named; the tool
    # still sees the exact grids; without the tool or images the setting does nothing.
    eng, sent = engine([{"name": "python", "arguments": json.dumps({"code": "print(train[0]['input'])"})}, RIGHT],
                       settings={"images": True, "tool": True, "pictures": True, "image_side": 896, "max_attempts": 1})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    parts = sent[0][1]["content"]
    text = parts[0]["text"]
    assert text == tasks.pictures_prompt(TASK) and "1 2" not in text and "Training pair 1: input 2x2 -> output" in text
    assert parts[1]["text"].strip() == tasks.PICTURES_NOTE
    assert sum(p["type"] == "image_url" for p in parts) == len(TASK["train"]) + len(TASK["test"])
    assert "[[1, 2], [3, 4]]" in sent[1][-1]["content"]  # the tool result: the exact grid
    assert run.candidates and run.candidates[-1]["verified"]
    for settings in ({"images": True, "pictures": True}, {"tool": True, "pictures": True}):
        eng, sent = engine([RIGHT], settings={**settings, "max_attempts": 1})
        asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
        first = sent[0][1]["content"]
        assert tasks.task_prompt(TASK) in (first if isinstance(first, str) else first[0]["text"])


def test_reject_known_asks_for_a_different_reading_once_per_known_hit():
    # reject_known: a verified program whose test output is a known reading stays a candidate, the attempt goes on with
    # KNOWN_READING; a new verified reading ends it; without known readings the setting does nothing.
    known = json.dumps([[[7], [8], [9]]])  # RIGHT's test output
    new_reading = ("```python\ndef transform(grid):\n    out = [list(r) for r in zip(*grid)]\n"
                   "    return out[::-1] if len(grid[0]) == 3 else out\n```")  # same pairs, another test output
    eng, sent = engine([RIGHT, new_reading], settings={"reject_known": 2, "known_k": 3, "max_attempts": 1})
    eng.pools[("", "t")] = [{"view": 0, "verified": True, "tests": known, "code": "x"}]
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert run.known == {known} and run.rejects == 1
    assert sent[1][-1]["content"] == tasks.KNOWN_READING.format(s="")
    assert [c["verified"] for c in run.candidates] == [True, True] and run.candidates[0].get("known")
    assert run.first_verified_attempt == 1 and len(sent) == 2
    eng, sent = engine([RIGHT], settings={"reject_known": 2, "max_attempts": 1})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert run.rejects == 0 and len(sent) == 1 and run.first_verified_attempt == 1


def test_score_run_writes_every_reading_view_once_and_resumes(tmp_path):
    # arc2.score against the scripted server: each (reading, view) once, the span after the empty-answer prefix,
    # a second call skips what scores.jsonl has.
    import threading
    import urllib.request
    from arc2 import preflight, score

    port = 23457
    threading.Thread(target=preflight.serve_forever, args=(str(tmp_path), port), daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    items = [{"id": "r1", "task": "t", "tests": [[[7], [8], [9]]], "right": True, "count": 3},
             {"id": "r2", "task": "t", "tests": [[[9], [8], [7]]], "right": False, "count": 1}]
    (tmp_path / "items.json").write_text(json.dumps(items))
    trial = {"score_items": "items.json", "score_source": str(tmp_path), "score_views": 2, "score_concurrency": 2}
    report = asyncio.run(score.run_scores(trial, {"t": TASK}, tmp_path, f"http://127.0.0.1:{port}/v1",
                                          time.monotonic() + 60))
    rows = [json.loads(x) for x in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert report["status"] == "complete" and report["error_count"] == 0
    assert sorted((r["reading"], r["view"]) for r in rows) == [("r1", 0), ("r1", 1), ("r2", 0), ("r2", 1)]
    assert all(r["tokens"] > 0 and r["logprob"] <= 0 for r in rows)
    again = asyncio.run(score.run_scores(trial, {"t": TASK}, tmp_path, f"http://127.0.0.1:{port}/v1",
                                         time.monotonic() + 60))
    assert again["jobs"] == 0 and again["skipped"] == 4
    view, outs = score.view_of(TASK, [[[7], [8], [9]]], "t", 1)
    sym, colours = augment_spec("t", 1)
    assert outs == [[[colours[c] for c in row] for row in SYMMETRIES[sym][0]([[7], [8], [9]])]]


def test_image_feedback_and_side():
    # image_feedback: after a wrong program, the feedback text then input | output | expected per failing pair;
    # image_side 896 doubles the cell size; review with images shows test input | output.
    eng, sent = engine([WRONG, RIGHT], settings={"images": True, "image_feedback": True, "image_side": 896})
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    second = sent[1][-1]["content"]
    assert second[0]["type"] == "text" and second[0]["text"].startswith("Your program is correct on")
    from arc_infra.model import png_size
    urls = [p["image_url"]["url"] for p in second if p["type"] == "image_url"]
    assert 1 <= len(urls) <= tasks.DETAILED_PAIRS
    g = TASK["train"][0]["input"]
    cell = max(28, min(56, 896 // max(len(g), len(g[0]))))
    assert png_size(urls[0])[1] == len(g) * cell + 1
    eng, sent = engine([WRONG, RIGHT], settings={"images": True})
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    assert isinstance(sent[1][-1]["content"], str)  # off: text only
    eng, sent = engine([RIGHT, "KEEP"], settings={"images": True, "review": True})
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    review = sent[1][-1]["content"]
    assert review[0]["type"] == "text" and sum(p["type"] == "image_url" for p in review) == len(TASK["test"])


def test_hard_effort():
    # hard_effort: attempts after the first hard_after on a task with no verified program use that effort.
    seen = []
    eng, sent = engine([WRONG, WRONG, RIGHT], settings={"hard_effort": "high", "hard_after": 1, "attempt_replies": 1})
    complete = eng.complete

    async def spy(task_id, messages, count, max_tokens, settings):
        seen.append(settings["effort"])
        return await complete(task_id, messages, count, max_tokens, settings)
    eng.complete = spy
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert seen[0] != "high" and "high" in seen[1:]


def test_image_changes():
    assert tasks.change_grid([[1, 2]], [[1, 3]]) == [[5, 3]] and tasks.change_grid([[1]], [[1, 2]]) is None
    eng, sent = engine([RIGHT], settings={"images": True, "image_changes": True})
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    parts = sent[0][1]["content"]
    assert tasks.CHANGES_NOTE in parts[1]["text"]


def test_show_pictures():
    # show: grids passed to show() in a `python` call come back as a picture in a user message; earlier cells' calls
    # are not shown again.
    out = sandbox.run_cells(["show(train[0]['input'])"], "show(train[0]['input'], train[0]['output'])\nprint(1)",
                            TASK, show=True)
    assert out["shown"] == [[TASK["train"][0]["input"], TASK["train"][0]["output"]]] and out["stdout"] == "1\n"
    assert "shown" not in sandbox.run_cells([], "show(train[0]['input'])", TASK)  # off: no pictures
    call = {"name": "python", "arguments": json.dumps({"code": "show(train[0]['input'])"})}
    eng, sent = engine([call, RIGHT], settings={"tool": True, "show": True})

    async def explore(cells, code, task, **kw):
        return await asyncio.to_thread(sandbox.run_cells, cells, code, task, **kw)

    eng.explore = explore
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    assert tasks.SHOW_NOTE in sent[0][0]["content"]
    tool, pics = sent[1][-2], sent[1][-1]
    assert tool["role"] == "tool" and pics["role"] == "user" and pics["content"][1]["type"] == "image_url"


def test_twins_share():
    # twins: a verified program found by one run moves the other out of "open"; its next attempts confirm.
    a, b = TaskRun("t", TASK), TaskRun("t~2", TASK)
    b.candidates, b.replies, b.copy_offset = a.candidates, a.replies, 1
    a.shared = b.shared = True
    eng, sent = engine([RIGHT], settings={"augment": 4})
    asyncio.run(eng.advance(a, 300000, replies=1))
    assert eng.phase(a) == "confirm" and eng.phase(b) == "confirm"
    eng.start_attempt(b)
    assert b.copy == 1
    eng.end_attempt(b, False)
    assert b.confirms == 1 and b.first_verified_attempt is None


def test_examples_prompt():
    # examples: the nearest library entries (training-task programs) come before the puzzle; none without a library.
    lib = [{"id": i, "train": TASK["train"], "program": f"def transform(grid):\n    return grid  # {i}",
            "features": tasks.task_features(TASK)} for i in ("a", "b", "c")]
    assert [e["id"] for e in tasks.pick_examples(TASK, lib, 2)] == ["a", "b"]
    eng, sent = engine([RIGHT], settings={"examples": 2})
    eng.library = lib
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    first = sent[0][1]["content"]
    assert first.startswith(tasks.EXAMPLES_HEAD) and "# b" in first and "# c" not in first
    assert first.endswith(tasks.task_prompt(TASK))


def test_repair():
    # repair: after the first verified program (one reading), one direct reply; its different grid is the second guess.
    eng, sent = engine([OTHER, "```\n7 8 9\n```", RIGHT], settings={"repair": True, "confirm_attempts": 2})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert sent[1][0]["content"] == tasks.DIRECT_SYSTEM and "the program's output" in sent[1][1]["content"]
    rep = [c for c in run.candidates if c.get("repair")]
    assert rep and rep[0]["tests"] == [[[7, 8, 9]]] and run.repaired
    picks = guesses(run.candidates, [TASK["test"][0]["input"]])[0]
    assert picks[1][1] == "direct"


def test_anchor():
    # anchor: the prompt asks for a prediction first; a ```predict block is a candidate ranked after verified ones.
    reply = "new: wider\n```predict\n7\n8\n9\n```\n" + RIGHT
    eng, sent = engine([reply], settings={"anchor": True})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert "```predict" in sent[0][1]["content"]
    pred = [c for c in run.candidates if c.get("predict")]
    assert pred and pred[0]["tests"] == [[[7], [8], [9]]]
    assert tasks.predicted_grids("```python\nx\n```", 1) is None


def test_late():
    # late_after: once the engine's tasks have spent that many tokens, replies use late_cap and late_effort.
    seen = []
    eng, sent = engine([WRONG, WRONG, RIGHT], settings={"late_after": 1500, "late_effort": "xhigh", "late_cap": 9999,
                                                        "attempt_replies": 1})
    complete = eng.complete

    async def spy(task_id, messages, count, max_tokens, settings):
        seen.append((settings["effort"], max_tokens))
        return await complete(task_id, messages, count, max_tokens, settings)
    eng.complete = spy
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    assert seen[0][0] != "xhigh" and seen[2][0] == "xhigh" and seen[2][1] <= 9999 and seen[2][1] != seen[0][1]


TASK3 = {"train": TASK["train"] + [{"input": [[1, 2, 3]], "output": [[1], [2], [3]]}], "test": TASK["test"]}
FIT2 = "```python\ndef transform(grid):\n    return [list(r) for r in zip(*grid)] if len(grid) == 1 else grid\n```"


def test_holdout():
    # holdout: the first attempt hides pair 1; a program right on the shown pairs but wrong on the hidden one is told
    # so without seeing it, and does not verify; the hidden pair never appears in the prompt.
    eng, sent = engine([FIT2, RIGHT], settings={"holdout": True})
    run = TaskRun("t", TASK3)
    asyncio.run(eng.advance(run, 300000))
    first = sent[0][1]["content"]
    assert "[[1, 2], [3, 4]]" not in first and "1 2\n3 4" not in first and "kept back" in first
    assert "kept back" in sent[1][-1]["content"] and not run.candidates[0]["verified"] and run.candidates[1]["verified"]


def test_hypotheses_and_diverse():
    eng, sent = engine([RIGHT, OTHER], settings={"hypotheses": True, "confirm_style": "diverse", "confirm_attempts": 2})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert "three different rules" in sent[0][1]["content"] and "Reading 1:" in sent[1][1]["content"]


def test_open_schedule():
    # open_schedule: an open task's attempt k uses entry k's effort and cap (the last repeats).
    seen = []
    eng, sent = engine([WRONG, WRONG, WRONG, RIGHT], settings={"open_schedule": [["low", 4000], ["high", 9000]],
                                                               "attempt_replies": 1})
    complete = eng.complete

    async def spy(task_id, messages, count, max_tokens, settings):
        seen.append((settings["effort"], max_tokens))
        return await complete(task_id, messages, count, max_tokens, settings)
    eng.complete = spy
    asyncio.run(eng.advance(TaskRun("t", TASK), 300000))
    assert seen[0] == ("low", 4000) and seen[1][0] == "high" and seen[2][0] == "high" and seen[1][1] <= 9000


def _scripted_roster(tmp_path, monkeypatch, delay="0.3"):
    """A 6-task roster bundle, a served cache layout and arc2.preflight's scripted server on a free local port."""
    import socket
    import threading
    import urllib.request

    from arc2 import preflight

    source, server = tmp_path / "bundle", tmp_path / "server"
    (source / "data/rl").mkdir(parents=True)
    (source / "data/evaluation").mkdir(parents=True)
    server.mkdir()
    ids = [f"t{i}" for i in range(6)]
    for i in ids:
        (source / f"data/evaluation/{i}.json").write_text(json.dumps(TASK))
    (source / "data/rl/tasks.jsonl").write_text(json.dumps(TASK) + "\n")  # the scripted server's answer table
    layout = {"schema": 1, "num_blocks": 100, "max_model_len": 8192, "lookahead": 0,
              "costs": [{"tokens": 4096 * k, "blocks": k + 2, "groups": [k, 2]} for k in (1, 2)]}
    (server / "server.log").write_text(f"ARC_CACHE_LAYOUT {json.dumps(layout)}\n")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("ARC2_PREFLIGHT_DELAY", delay)
    monkeypatch.delenv("ARENA_STOP_REQUEST", raising=False)
    threading.Thread(target=preflight.serve_forever, args=(source, port), daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    study = {"arc2": {"data": "data"}, "kv_margin": 0.9, "windows": [8192], "completion_cap": 2048,
             "generation_slots": 4, "active_limit": 2, "reasoning_effort": "low", "temperature": 0.7}
    trial = {"name": "h", "kind": "arc2", "split": "evaluation", "tasks": ids, "wall_seconds": 300,
             "marks": {"end": 10 ** 9}, "curve_step": 1000, "stop_grace_seconds": 60}
    return source, server, f"http://127.0.0.1:{port}/v1", study, trial


def test_run_arc2_stop_request_ends_the_roster_between_replies(tmp_path, monkeypatch):
    """Graceful stop: a stop request ends a roster trial between replies. Replies in flight finish,
    none starts after the request is seen, and report.json, tasks.jsonl (finished tries kept) and the marks are
    written as at the wall. The real run_arc2 against arc2.preflight's scripted server on this CPU."""
    import threading

    from arc2.trial import run_arc2

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch)
    out = tmp_path / "out"

    def stopper():  # the request lands once a few replies have ended
        for _ in range(1200):
            events = out / "events.jsonl"
            if events.exists() and len(events.read_text().splitlines()) >= 3:
                (out / "stop-request").write_text("{}")
                return
            time.sleep(0.05)

    threading.Thread(target=stopper, daemon=True).start()
    began = time.monotonic()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    stop = report["stop_request"]
    assert stop and stop["seen_seconds"] <= stop["ended_seconds"] < 60 and time.monotonic() - began < 90
    assert stop["cut_inflight"] == 0  # the replies in flight finished inside the grace
    replies = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    started = replies[0]["t1"] - replies[0]["t"]  # the trial's monotonic start
    assert len(replies) == report["replies"] >= 3
    assert all(r["t0"] - started <= stop["seen_seconds"] + 0.05 for r in replies)  # none started after the request
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert len(rows) == len(trial["tasks"]) and sum(len(r["candidates"]) for r in rows) == report["candidates"] > 0
    assert json.loads((out / "report.json").read_text())["stop_request"] == stop
    assert report["roster"]["marks"]["end"]["reached"] is False and report["status"] == "complete"


def test_run_arc2_fixed_attempt_paired_design(tmp_path, monkeypatch):
    """C5 (audit 20 Oct): a fixed-attempt paired trial. Every task under both arms, ``copies`` entries a task with one
    attempt each (max_attempts 1), copy k + 1 on augmented view k in both arms (rotate_copies), tiers mode to the end."""
    from arc2.trial import run_arc2

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    del trial["marks"]
    trial.update(tasks=trial["tasks"][:3], tiers=[10 ** 9], copies=3, paired=True, rotate_copies=True,
                 arms=[{"name": "base", "max_attempts": 1, "augment": 4},
                       {"name": "c1", "max_attempts": 1, "augment": 4, "tool_fixes": True}])
    out = tmp_path / "out"
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert report["status"] == "complete" and len(rows) == 3 * 3 * 2
    assert all(r["attempts"] == 1 and r["ended"] == "attempts" for r in rows)
    replies = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    views = {}
    for r in replies:
        name, arm = r["task"].split("@")
        views.setdefault(name, {})[arm] = r.get("copy", 0)
    assert len(views) == 9 and all(v["base"] == v["c1"] == int(n.split("#")[1]) - 1 for n, v in views.items())
    assert set(report["per_arm"]) == {"base", "c1"}
    # Resume: a second run skips the entries the first finished; copy_major starts every task's copy 1 first.
    done = [r["task"] for r in rows][:7]
    trial.update(skip=done, copy_major=True)
    again = tmp_path / "again"
    asyncio.run(run_arc2(study, trial, source=str(source), output=again, server_dir=server, base_url=url,
                         deadline_epoch=time.time() + 600))
    second = [json.loads(line)["task"] for line in (again / "tasks.jsonl").read_text().splitlines()]
    assert sorted(second + done) == sorted(r["task"] for r in rows)
    copies = [int(t.split("@")[0].split("#")[1]) for t in second]  # the read-out keeps the play order
    assert copies == sorted(copies)


def test_pool_picks_one_program_per_verified_reading_then_near_misses():
    """pool (item 1, 21 Oct): an attempt from ``pool_after`` sees up to k earlier programs on its view, one per
    distinct verified test reading first, then the unverified ones by pairs passed and cell accuracy."""
    def row(code, verified, tests, passed=2, accuracy=1.0, view=0):
        return {"view": view, "code": code, "passed": passed, "pairs": 2, "accuracy": accuracy, "verified": verified,
                "tests": tests}
    eng, _ = engine([])
    eng.pools[("", "t")] = [row("a1", True, "[1]"), row("a2", True, "[1]"), row("b", True, [[2]]),
                            row("far", False, "[3]", passed=0, accuracy=0.2), row("near", False, "[4]", 1, 0.9),
                            row("a1", True, "[1]"), row("v1", True, "[5]", view=1)]
    eng.pools[("pool", "t")] = [row("near2", False, "[6]", 1, 0.95)]
    run = TaskRun("t#2@pool", TASK, {"name": "pool"})
    run.attempts, run.copy = 1, 0
    picked = eng.pick_pool(run, 4)
    assert len(picked) == 4 and {p["code"] for p in picked[:2]} in ({"a1", "b"}, {"a2", "b"})
    assert [p["code"] for p in picked[2:]] == ["near2", "near"]
    assert [p["code"] for p in eng.pick_pool(run, 1)] in (["a1"], ["a2"], ["b"])
    run.copy = 1
    assert [p["code"] for p in eng.pick_pool(run, 3)] == ["v1"]
    text = tasks.pool_prompt(TASK, picked[1:3] + [dict(picked[3], code="x" * 5000)])
    assert text.startswith(tasks.task_prompt(TASK)) and "Program A (" in text and "Program C (reproduces 1 of 2 " \
        "training pairs, 90% of cells)" in text and "# ... (cut)" in text and "x" * 5001 not in text


def test_pool_seeds_and_records_programs_in_the_attempt_prompt():
    """pool with pool_after 1: the first attempt already sees the seeded programs; with pool_live the next attempt
    also sees this run's own, and every pooled candidate records how many programs it was shown."""
    eng, sent = engine([WRONG, RIGHT, OTHER], settings={"pool": 2, "pool_after": 1, "attempt_replies": 1,
                                                       "confirm_attempts": 1})
    eng.pools[("", "t")] = [{"view": 0, "code": "def transform(grid):\n    return grid[::-1]", "passed": 0, "pairs": 2,
                             "accuracy": 0.5, "verified": False, "tests": "[[[9]]]"}]
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert "Program A (reproduces 0 of 2 training pairs, 50% of cells)" in sent[0][1]["content"]
    assert "Program B" not in sent[0][1]["content"] and "Program B" in sent[1][1]["content"]
    assert "print('hi')" in sent[1][1]["content"] and run.candidates[0]["pooled"] == 1
    assert run.candidates[1]["pooled"] == 2 and len(eng.pools[("", "t")]) == 4
    plain, sent = engine([WRONG, RIGHT, OTHER], settings={"attempt_replies": 1, "confirm_attempts": 1})
    asyncio.run(plain.advance(TaskRun("t", TASK), 300000))
    assert sent[1][1]["content"] == tasks.task_prompt(TASK) and not plain.pools


def test_pool_check_style_shows_one_verified_program_and_its_test_output():
    """pool_style "check" (item 2): the attempt sees one earlier verified program (never an unverified one) with its
    test output and is asked for the most plausible reading where the test input is new."""
    eng, sent = engine([RIGHT], settings={"pool": 3, "pool_after": 1, "pool_style": "check", "pool_live": False,
                                          "attempt_replies": 1, "confirm_attempts": 0})
    eng.pools[("", "t")] = [{"view": 0, "code": "def transform(grid):\n    return [[7], [8], [9]]", "passed": 2,
                             "pairs": 2, "accuracy": 1.0, "verified": True, "tests": "[[[7], [8], [9]]]"},
                            {"view": 0, "code": "def transform(grid):\n    return grid", "passed": 1, "pairs": 2,
                             "accuracy": 0.9, "verified": False, "tests": "[[[1]]]"}]
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    first = sent[0][1]["content"]
    assert first.startswith(tasks.task_prompt(TASK)) and "return [[7], [8], [9]]" in first
    assert "(3x1):\n7\n8\n9" in first and "return grid\n" not in first and run.candidates[0]["pooled"] == 1
    eng.pools[("", "t")] = eng.pools[("", "t")][1:]  # no verified program: a plain attempt
    eng2, sent2 = engine([RIGHT], settings={"pool": 3, "pool_after": 1, "pool_style": "check", "attempt_replies": 1,
                                            "confirm_attempts": 0})
    eng2.pools[("", "t")] = eng.pools[("", "t")]
    asyncio.run(eng2.advance(TaskRun("t", TASK), 300000))
    assert sent2[0][1]["content"] == tasks.task_prompt(TASK)


def test_pool_judge_style_groups_sampled_verified_programs_by_reading():
    """pool_style "judge" (item 3): k earlier verified programs drawn at random, grouped by test reading with their
    attempt counts; the candidate records the readings it was shown."""
    def row(code, tests, verified=True):
        return {"view": 0, "code": code, "passed": 2, "pairs": 2, "accuracy": 1.0, "verified": verified, "tests": tests}
    eng, sent = engine(["RANKING: B, A\n" + RIGHT], settings={
        "pool": 4, "pool_after": 1, "pool_style": "judge", "pool_live": False, "attempt_replies": 1,
        "confirm_attempts": 0})
    eng.pools[("", "t")] = [row("def transform(grid):\n    return [[7], [8], [9]]  # a", "[[[7], [8], [9]]]"),
                            row("def transform(grid):\n    return [[7], [8], [9]]  # b", "[[[7], [8], [9]]]"),
                            row("def transform(grid):\n    return [[7], [8], [9]]  # c", "[[[7], [8], [9]]]"),
                            row("def transform(grid):\n    return [[1, 2, 3]]", "[[[1, 2, 3]]]"),
                            row("def transform(grid):\n    return grid", "[[[5]]]", verified=False)]
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    first = sent[0][1]["content"]
    assert "wrote 4 programs" in first and "their 2 different test outputs" in first and "return grid" not in first
    assert "(3 of 4 attempts)" in first and "(1 of 4 attempts)" in first and "(1x3):\n1 2 3" in first
    judged = run.candidates[0]["judged"]
    assert run.candidates[0]["pooled"] == 4 and sorted(n for _, n in judged) == [1, 3]
    assert hashlib.sha256(b"[[[1, 2, 3]]]").hexdigest()[:12] in [h for h, _ in judged]
    assert run.candidates[0]["ranking"] == [1, 0] and "RANKING: X, Y" in first


def test_two_readings_scores_the_second_program_as_a_candidate():
    """two_readings: the first message asks for two readings; a reply's last two programs are both candidates, the
    second flagged and ranked after the first; feedback and verification follow the first."""
    eng, sent = engine([WRONG + "\n" + RIGHT, RIGHT + "\n" + OTHER], settings={"two_readings": True,
                                                                          "attempt_replies": 2, "confirm_attempts": 0})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert sent[0][1]["content"].endswith(tasks.TWO_READINGS.format(s=""))
    assert [(c["verified"], bool(c.get("second"))) for c in run.candidates] == [
        (False, False), (True, True), (True, False), (True, True)]
    assert "passes 0 of 2 training pairs" in sent[1][-1]["content"] or "(0," in sent[1][-1]["content"]
    assert run.verified_outputs() == {json.dumps(run.candidates[2]["tests"])}  # a second never settles a task
    first = guesses(run.candidates, [t["input"] for t in TASK["test"]])[0]
    assert first[0] == (run.candidates[2]["tests"][0], "verified")


def test_scouts_state_rules_pick_one_and_hand_it_to_the_attempt():
    """scouts: n short code-free replies at scout_effort, one pick reply, the pick in the attempt's first message;
    every side request's reservation is dropped and its tokens count as the task's."""
    dropped = []
    eng, sent = engine(["RULE: keep the grid", "RULE: recolour 1 to 2", "Rule B fits. PICK: B", RIGHT],
                       settings={"scouts": 2, "confirm_attempts": 0, "max_attempts": 1}, drop=dropped.append)
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert sent[0][0]["content"] == tasks.SCOUT_SYSTEM and sent[0][1]["content"].endswith(tasks.SCOUT.format(s=""))
    assert "Rule A:\nRULE: keep the grid" in sent[2][1]["content"] and "Rule B:\nRULE: recolour 1 to 2" in sent[2][1]["content"]
    assert sent[3][1]["content"].endswith(tasks.SCOUT_PICK.format(rule="RULE: recolour 1 to 2"))
    assert run.scout["pick"] == 1 and run.scout["tokens"] == 3000 and run.spent == 4000
    assert sorted(dropped) == ["t~j1", "t~s1.0", "t~s1.1"]
    assert run.candidates[0]["verified"]
    from arc2.trial import efforts
    assert efforts({}, [{"scouts": 4, "effort": "medium"}]) == ["low", "medium"]


def test_trace_shows_the_failing_programs_steps_on_its_first_failing_pair():
    """trace: a program that fails a training pair gets its line trace on the first failing pair in the feedback;
    a verified program gets none."""
    steps = ("```python\nimport numpy as np\ndef transform(grid):\n    a = np.array(grid)\n    n = a.shape[0]\n"
             "    for k in range(5):\n        n += k\n    return a.tolist()\n```")
    eng, sent = engine([steps, RIGHT], settings={"trace": True, "confirm_attempts": 0})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    feedback = sent[1][-1]["content"]
    assert "steps on training pair 1" in feedback and feedback.endswith(tasks.FIX_LINE)
    assert "a = array 2x2, values [1, 2, 3, 4]" in feedback and "n = 2" in feedback
    assert "later steps on lines already shown are left out" in feedback
    assert run.candidates[-1]["verified"] and len(sent) == 2
    out = sandbox.run("def transform(grid):\n    x = 1\n    return grid[9]", [[[1]]], trace=0)
    assert "x = 1" in out["trace"] and "stopped: IndexError" in out["trace"]


def test_scouts_cut_while_thinking_give_their_last_lines_with_scout_unfinished():
    eng, sent = engine(["", "RULE: two", "PICK: A", RIGHT], settings={"scouts": 2, "max_attempts": 1,
                                                                    "scout_unfinished": True})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert run.scout["rules"] == [tasks.SCOUT_UNFINISHED + "think", "RULE: two"] and run.scout["pick"] == 0


def test_columns_add_each_grids_transposed_text():
    eng, sent = engine([RIGHT], settings={"columns": True, "max_attempts": 1})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    first = sent[0][1]["content"]
    assert "Input (2x2):\n1 2\n3 4\nThe same grid column by column (line k = column k, top to bottom):\n1 3\n2 4" in first
    assert "Test input 1 (1x3):\n7 8 9\nThe same grid column by column (line k = column k, top to bottom):\n7\n8\n9" in first


def test_scouts_without_a_pick_leave_the_attempt_plain():
    eng, sent = engine(["", "RULE: one", "no letter here", RIGHT], settings={"scouts": 2, "max_attempts": 1})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert run.scout["rules"] == ["RULE: one"] and run.scout["pick"] == 0  # one rule: no judge reply
    eng, sent = engine(["RULE: one", "RULE: two", "no letter here", RIGHT], settings={"scouts": 2, "max_attempts": 1})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000))
    assert run.scout["pick"] is None and tasks.SCOUT_PICK.split("{")[0] not in sent[3][1]["content"]


def test_run_arc2_pool_seed_in_a_fixed_attempt_trial(tmp_path, monkeypatch):
    """The item 1 paired test: one attempt per entry on view 0, the pool arm's prompt from the trial's pool_seed."""
    from arc2.trial import run_arc2

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    del trial["marks"]
    seed = {"t0": [{"view": 0, "code": "def transform(grid):\n    return grid", "passed": 0, "pairs": 2,
                    "accuracy": 0.5, "verified": False, "tests": "[[[1]]]"}]}
    (source / "seed.json").write_text(json.dumps(seed))
    trial.update(tasks=trial["tasks"][:2], tiers=[10 ** 9], copies=2, paired=True, pool_seed="seed.json",
                 arms=[{"name": "base", "max_attempts": 1},
                       {"name": "pool", "max_attempts": 1, "pool": 3, "pool_after": 1, "pool_live": False}])
    out = tmp_path / "out"
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    assert report["status"] == "complete" and set(report["per_arm"]) == {"base", "pool"}
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 2 * 2
    pooled = {r["task"]: [c.get("pooled") for c in r.get("candidates", [])] for r in rows}
    assert all(not any(v) for t, v in pooled.items() if t.endswith("@base"))
    seeded = [v for t, v in pooled.items() if t.startswith("t0#") and t.endswith("@pool")]
    assert len(seeded) == 2 and all(v and v == [1] * len(v) for v in seeded)
    assert all(not any(v) for t, v in pooled.items() if not t.startswith("t0#"))


def test_run_arc2_repair_seed_replays_one_repair_reply_per_seeded_entry(tmp_path, monkeypatch):
    """Repair replay: a seeded entry starts with the recorded program's test outputs as its one verified candidate
    and takes one repair reply; an entry without a seed ends before play."""
    from arc2.trial import run_arc2

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    del trial["marks"]
    (source / "seed.json").write_text(json.dumps({"t0#1": [[[1]]], "t1#2": [[[2]]]}))
    trial.update(tasks=trial["tasks"][:2], tiers=[10 ** 9], copies=2, repair_seed="seed.json",
                 arms=[{"name": "rep", "max_attempts": 1, "repair": True}])
    out = tmp_path / "out"
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    rows = {r["task"]: r for r in map(json.loads, (out / "tasks.jsonl").read_text().splitlines())}
    assert report["status"] == "complete" and len(rows) == 4
    assert {k for k, r in rows.items() if r["ended"] == "no_seed"} == {"t0#2", "t1#1"}
    seeded = [rows["t0#1"], rows["t1#2"]]
    assert all(r["attempts"] == 1 and r["candidates"][0].get("seed") and r["spent"] > 0 for r in seeded)


def test_run_arc2_effort_guard_refuses_a_rejected_effort_before_play(tmp_path, monkeypatch):
    """C4 effort guard (20 Oct): the 17 Oct hard-effort run sent "high" from attempt 2 and its open entries ended in
    400s. The trial now renders every effort it can send before play and refuses the run."""
    from arc2.trial import efforts, run_arc2

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    trial["arms"] = [{"name": "esc", "effort": "medium", "hard_effort": "high", "hard_after": 1,
                      "open_schedule": [["medium", 2048], ["xhigh", 2048]]}]
    assert efforts(study, trial["arms"]) == ["high", "low", "medium", "xhigh"]
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="rejects reasoning_effort 'high'"):
        asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                             deadline_epoch=time.time() + 600))
    assert not (out / "events.jsonl").exists() or not (out / "events.jsonl").read_text()  # no reply was sent


def test_api_proxy_maps_requests_and_replies():
    from arc2 import api_proxy, tasks

    body = tasks.chat_request("flash", [{"role": "system", "content": "s"},
                                        {"role": "user", "content": [{"type": "text", "text": "x" * 320},
                                                                     {"type": "image_url", "image_url": {"url": "d"}}]},
                                        {"role": "assistant", "content": "a", "reasoning": "r", "reasoning_content": "r"}],
                              1000, temperature=0.7, effort="xhigh", tools=[tasks.PYTHON_TOOL])
    up = api_proxy.upstream_body(body, "z-ai/glm-5.3-flash")
    assert up["model"] == "z-ai/glm-5.3-flash" and up["reasoning"] == {"effort": "high"}
    assert "chat_template_kwargs" not in up and "top_k" not in up and up["tools"] and up["max_tokens"] == 1000
    assert up["messages"][2] == {"role": "assistant", "content": "a", "reasoning": "r"}
    assert "reasoning" not in api_proxy.upstream_body(body, "m", keep_reasoning=False)["messages"][2]
    assert 300 <= api_proxy.estimate(body["messages"]) < 500
    reply = api_proxy.reply_body({"choices": [{"message": {"content": "c", "reasoning": "why"}}]})
    assert reply["choices"][0]["message"]["reasoning_content"] == "why"


def test_transport_does_not_send_a_reply_that_waited_at_the_gate_past_a_stop():
    """Flake of test_run_arc2_stop_request_ends_the_roster_between_replies (25 Oct): a worker passed its stop check,
    waited at the gate, and its reply started after the stop was seen. Transport checks the stop after the gate."""
    import types

    from arc2.loop import StopRequested
    from arc2.trial import Transport

    async def scenario():
        opened, posts, released = asyncio.Event(), [], []

        class Gate:
            budget, capacity, reserved, inflight, log = None, 10, 0, 0, None

            async def acquire(self, task_id, held):
                await opened.wait()
                return types.SimpleNamespace(tokens=1, game_id=task_id)

            def release(self, reservation):
                released.append(reservation)

        class Client:
            async def post(self, *args, **kwargs):
                posts.append(args)
                raise AssertionError("sent after the stop")

        transport = Transport(Client(), Gate(), model="m", effort="medium", temperature=0.7,
                              deadline=time.monotonic() + 60)
        stop = {"seen": False}
        transport.stopped = lambda: stop["seen"]
        request = asyncio.create_task(transport.complete("t#1", [{"role": "user", "content": "x"}], 10, 100, {}))
        await asyncio.sleep(0.01)  # the reply waits at the gate
        stop["seen"] = True
        opened.set()
        try:
            await request
        except StopRequested:
            return posts, released
        raise AssertionError("no StopRequested")

    posts, released = asyncio.run(scenario())
    assert posts == [] and len(released) == 1  # nothing sent, the reservation given back


def test_guesses_direct_answer_is_second_answer_only():
    """direct_answer: the direct grid takes slot 2 where slot 2 is empty or unverified, never over a
    second verified reading, never slot 1 over a harness candidate."""
    from arc2.loop import guesses

    def cand(grid, verified, passed=1, **kw):
        return {"code": str(grid) + str(verified) + str(kw), "tests": [grid], "verified": verified, "passed": passed,
                "accuracy": 0.5, "tokens_at": 1, **kw}

    A, B, D = [[1]], [[2]], [[9]]
    answer = cand(D, False, passed=0, direct=True, answer=True)
    # one verified reading: answer is slot 2
    assert guesses([cand(A, True), answer], [[[0]]])[0] == [(A, "verified"), (D, "answer")]
    # verified + unverified: answer replaces the unverified slot 2
    assert guesses([cand(A, True), cand(B, False), answer], [[[0]]])[0] == [(A, "verified"), (D, "answer")]
    # two verified readings: answer unused
    assert guesses([cand(A, True), cand(B, True), answer], [[[0]]])[0] == [(A, "verified"), (B, "verified")]
    # no harness candidate: answer alone
    assert guesses([answer], [[[0]]])[0] == [(D, "answer")]
    # only unverified: slot 1 stays the harness's, answer slot 2
    assert guesses([cand(A, False), answer], [[[0]]])[0] == [(A, "unverified"), (D, "answer")]
    # answer equal to slot 1: not repeated
    assert guesses([cand(D, True), answer], [[[0]]])[0] == [(D, "verified")]


def test_direct_answer_first_reply_then_harness_with_rotation_unchanged():
    """direct_answer: the first reply is direct (DIRECT_SYSTEM, task frame, no tool, xhigh, cap = the window's
    room); its grid is an "answer" candidate; the harness attempts after it see the same augment copies as without."""
    calls = []

    async def complete(task_id, messages, count, max_tokens, settings):
        calls.append((messages[0]["content"], max_tokens, settings.get("effort"), settings.get("tool"), count))
        content = "rule\n```\n1 3\n2 4\n```" if len(calls) == 1 else RIGHT
        return {"message": {"content": content, "reasoning": "think"}, "finish_reason": "stop",
                "usage": {"completion_tokens": 1000, "prompt_tokens": 500}}

    async def count(messages, settings):
        return 500

    async def execute(code, inputs, **kw):
        return await asyncio.to_thread(sandbox.run, code, inputs, **kw)

    eng = Engine(complete=complete, count=count, execute=execute, window=65536, cap=16384,
                 settings={"direct_answer": True, "augment": 4, "effort": "medium"})
    run = TaskRun("t", TASK)
    asyncio.run(eng.advance(run, 300000, replies=2))
    system, max_tokens, effort, tool, n = calls[0]
    assert system == tasks.DIRECT_SYSTEM and effort == "xhigh" and not tool and max_tokens == 65536 - 500
    answer = [c for c in run.candidates if c.get("answer")]
    assert len(answer) == 1 and answer[0]["tests"] == [[[1, 3], [2, 4]]] and answer[0]["direct"]
    assert calls[1][0] != tasks.DIRECT_SYSTEM and calls[1][1] == 16384 and calls[1][2] == "medium"
    assert run.copy == 0  # first harness attempt on the task's own frame, as without direct_answer


def test_direct_answer_share_copy1_answers_for_copy2_and_reserve_is_passed():
    """direct_answer_share: only copy 1 sends the direct reply (with ``gate_reserve``); copy 2 gets copy 1's grid
    as its second-answer candidate when next advanced, and its first request is a harness one."""
    calls = []

    async def complete(task_id, messages, count, max_tokens, settings):
        calls.append((task_id, messages[0]["content"] == tasks.DIRECT_SYSTEM, settings.get("gate_reserve")))
        direct = messages[0]["content"] == tasks.DIRECT_SYSTEM
        return {"message": {"content": "```\n1 3\n2 4\n```" if direct else RIGHT, "reasoning": "think"},
                "finish_reason": "stop", "usage": {"completion_tokens": 1000, "prompt_tokens": 500}}

    async def count(messages, settings):
        return 500

    async def execute(code, inputs, **kw):
        return await asyncio.to_thread(sandbox.run, code, inputs, **kw)

    eng = Engine(complete=complete, count=count, execute=execute, window=65536, cap=16384,
                 settings={"direct_answer": True, "direct_answer_share": True, "direct_answer_reserve": 32768})
    one, two = TaskRun("t#1", TASK), TaskRun("t#2", TASK)
    asyncio.run(eng.advance(one, 300000, replies=1))
    asyncio.run(eng.advance(two, 300000, replies=1))
    assert calls[0] == ("t#1", True, 32768) and calls[1] == ("t#2", False, None)
    shared = [c for c in two.candidates if c.get("answer")]
    assert len(shared) == 1 and shared[0]["shared"] and shared[0]["tests"] == [[[1, 3], [2, 4]]]


def test_transport_gate_reserve_holds_less_than_the_cap():
    import types

    from arc2.trial import Transport

    held = []

    class Gate:
        budget, capacity, reserved, inflight, log = None, 10 ** 9, 0, 0, None

        async def acquire(self, task_id, tokens):
            held.append(tokens)
            raise RuntimeError("stop here")

    transport = Transport(types.SimpleNamespace(), Gate(), model="m", effort="medium", temperature=0.7,
                          deadline=time.monotonic() + 60)
    for settings in ({}, {"gate_reserve": 32768}):
        try:
            asyncio.run(transport.complete("t#1", [{"role": "user", "content": "x"}], 1000, 120000, settings))
        except RuntimeError:
            pass
    assert held == [121000, 33768]
