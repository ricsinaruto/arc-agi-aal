"""Setting ``conversation`` (arc2.conv): one conversation and one persistent interpreter per task."""

import asyncio
import json

from arc2 import conv, sandbox, tasks
from arc2.loop import TaskRun, guesses, readout
from arc2.repl import Repl

TASK = {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 3], [2, 4]]},
                  {"input": [[5, 6]], "output": [[5], [6]]}],
        "test": [{"input": [[7, 8, 9]], "output": [[7], [8], [9]]}]}
RIGHT = "```python\ndef transform(grid):\n    return [list(r) for r in zip(*grid)]\n```"
SAME = "```python\ndef transform(grid):\n    return [list(c) for c in zip(*grid)]\n```"
WRONG = "```python\ndef transform(grid):\n    return grid\n```"


def call(code):
    return {"name": "python", "arguments": json.dumps({"code": code})}


def engine(script, window=65536, **settings):
    sent = []

    async def complete(task_id, messages, count, max_tokens, s):
        sent.append({"messages": [dict(m) for m in messages], "max_tokens": max_tokens, "settings": dict(s)})
        item = script.pop(0)
        if isinstance(item, dict):
            return {"message": {"content": None, "reasoning": "think", "tool_calls": [{"id": f"c{len(sent)}",
                                                                                       "function": item}]},
                    "finish_reason": "tool_calls", "usage": {"completion_tokens": 100, "prompt_tokens": 500}}
        return {"message": {"content": item, "reasoning": "think"}, "finish_reason": "stop" if item else "length",
                "usage": {"completion_tokens": 1000, "prompt_tokens": 500}}

    async def count(messages, s):
        return sum(conv.approx_tokens(m) for m in messages)

    async def execute(code, inputs, **kw):
        return await asyncio.to_thread(sandbox.run, code, inputs, strict=False)

    eng = conv.ConvEngine(complete=complete, count=count, execute=execute, window=window, cap=16384,
                          settings={"tool": True, "images": False, "conversation": True, **settings},
                          repl=lambda run: Repl(run.task, strict=False, call_seconds=5))
    return eng, sent


def test_conversation_programs_ledger_and_submit():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("x = len(train)\nprint(x)"), WRONG, RIGHT, SAME, call("print(x, program_2(test_inputs[0]))"),
                        call("submit(program_2(test_inputs[0]), test_inputs[0])")])
    asyncio.run(eng.advance(run, 10 ** 9))
    assert run.done == "submitted" and run.interp is None
    assert [r["outcome"] for r in run.replies] == ["tool", "wrong", "verified", "verified", "tool", "submit"]
    msgs = sent[-1]["messages"]
    assert msgs[0]["content"] == tasks.CONV_SYSTEM and len([m for m in msgs if m["role"] == "system"]) == 1
    assert sent[0]["settings"]["tool"] is True and tasks.tool_spec(sent[0]["settings"]) is tasks.CONV_TOOL
    # the interpreter persists: x from the first call, program_2 defined from the reply
    assert "2 [[7], [8], [9]]" in msgs[-1]["content"]
    texts = [m["content"] for m in msgs if m["role"] == "user" and isinstance(m["content"], str)]
    assert any("Program 3 passes" in t and "same as program 2" in t and "programs 2, 3" in t for t in texts)
    assert any(t.startswith("Program 1: ") and "1 distinct" not in t for t in texts)
    picks = guesses(run.candidates, [TASK["test"][0]["input"]])[0]
    assert picks[0][0] == [[7], [8], [9]] and picks[1][0] == [[7, 8, 9]]
    assert readout([run])["final"]["score_outputs"] == 100


def test_bad_submit_keeps_going_and_unsubmitted_task_uses_ranking():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[1, 2], [3]])"), RIGHT, ""])
    asyncio.run(eng.advance(run, 10 ** 9, replies=3))
    assert run.done is None and "Nothing was submitted" in sent[1]["messages"][-1]["content"]
    assert sent[2]["messages"][-1]["content"].startswith("Program 1 passes")
    assert run.replies[-1]["outcome"] == "truncated"
    assert guesses(run.candidates, [TASK["test"][0]["input"]])[0][0][0] == [[7], [8], [9]]
    run.interp.close()


def test_compaction_keeps_head_summary_and_recent_tail():
    run = TaskRun("t#1", TASK, {"conversation": True})
    big = "y" * 36000  # ~12K tokens a reply at 3 chars a token
    eng, sent = engine([big, big, big, "SUMMARY: tried transpose", call("print(1)")], window=40000,
                       cap=8000, compact_keep=0.10)
    asyncio.run(eng.advance(run, 10 ** 9, replies=5))
    assert run.compactions == 1 and run.replies[3]["outcome"] == "compaction"
    summary_request = sent[3]
    assert summary_request["settings"]["tool"] is False and summary_request["messages"][-1]["content"] == tasks.CONV_COMPACT
    after = sent[4]["messages"]
    assert after[:2] == run.head and "SUMMARY: tried transpose" in after[2]["content"]
    assert sum(conv.approx_tokens(m) for m in after[3:]) <= 0.10 * 40000 + 200
    run.interp.close()


def test_repl_restart_replays_clean_calls():
    r = Repl(TASK, strict=False, call_seconds=1)
    assert r.run("a = 41")["stdout"] == ""
    out = r.run("while True:\n    pass")
    assert "TimeoutError" in out["error"]
    assert r.run("print(a + 1)")["stdout"].strip() == "42"
    r.close()


def test_run_arc2_conversation_roster_on_the_scripted_server(tmp_path, monkeypatch):
    """The real run_arc2 with the conversation arm against arc2.preflight's scripted server: replies flow, a stop
    request ends the roster, every interpreter is closed and the report and tasks.jsonl are written."""
    import threading
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.1")
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False}]
    out = tmp_path / "out"
    closed = []
    real_close = Repl.close

    def spy(self):
        closed.append(1)
        real_close(self)
    monkeypatch.setattr(Repl, "close", spy)

    def stopper():
        for _ in range(1200):
            events = out / "events.jsonl"
            if events.exists() and len(events.read_text().splitlines()) >= 8:
                (out / "stop-request").write_text("{}")
                return
            time.sleep(0.05)

    threading.Thread(target=stopper, daemon=True).start()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    assert report["status"] == "complete" and report["replies"] >= 8 and not report["task_errors"]
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert len(rows) == len(trial["tasks"]) and closed
    replies = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    assert {r["attempt"] for r in replies if r.get("event") == "reply"} == {1}


def test_run_arc2_conversation_parks_at_compaction_and_resumes_with_its_interpreter(tmp_path, monkeypatch):
    """one slot, two tasks. A task keeps the slot until it compacts, then goes back to the
    backlog; the slot takes the task with the fewest tokens, and a parked task resumes later on the same interpreter
    process."""
    import threading
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    study.update(active_limit=1, windows=[5120], completion_cap=1024)
    trial["tasks"] = trial["tasks"][:2]
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False}]
    out = tmp_path / "out"
    pids, real_call = [], Repl.call

    def spy(self, request):
        reply = real_call(self, request)
        pids.append((id(self), self.process.pid, self.restarts))
        return reply
    monkeypatch.setattr(Repl, "call", spy)

    def rows():
        path = out / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def stopper():
        for _ in range(3000):
            if sum(r.get("outcome") == "compaction" for r in rows()) >= 3:
                (out / "stop-request").write_text("{}")
                return
            time.sleep(0.05)

    threading.Thread(target=stopper, daemon=True).start()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    assert report["status"] == "complete" and not report["task_errors"]
    replies = [r for r in rows() if r.get("event") == "reply"]
    # the slot changes task only right after a compaction, and each task holds it more than once
    switches = [i for i in range(1, len(replies)) if replies[i]["task"] != replies[i - 1]["task"]]
    assert len(switches) >= 2 and all(replies[i - 1]["outcome"] == "compaction" for i in switches)
    blocks = [replies[0]["task"]] + [replies[i]["task"] for i in switches]
    assert len(set(blocks)) == 2 and len(blocks) > len(set(blocks))
    # the resumed task's first request is after its own compaction: summary in the context, no restart
    resumed = blocks[2]
    first_back = replies[switches[1]]
    assert first_back["task"] == resumed and first_back["prompt_tokens"] > 0
    # one interpreter process per task across park and resume, never restarted
    by_task = {}
    for repl, pid, restarts in pids:
        by_task.setdefault(repl, set()).add(pid)
        assert restarts == 0
    assert len(by_task) == 2 and all(len(p) == 1 for p in by_task.values())
