"""Setting ``conversation`` (arc2.conv): one conversation and one persistent interpreter per task."""

import asyncio
import json
import re
from pathlib import Path

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
                          repl=lambda run: Repl(getattr(run, "shown", None) or run.task, strict=False, call_seconds=5))
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


def test_bad_arguments_nudge_after_three_and_conv1_text_when_off():
    bad = {"name": "python", "arguments": '{"code": "print(1)'}  # cut JSON: no snippet
    for fix in (True, False):
        run = TaskRun("t#1", TASK, {"conversation": True, "bad_call_fix": fix})
        eng, sent = engine([bad, bad, bad, ""])
        asyncio.run(eng.advance(run, 10 ** 9, replies=4))
        tools = [m for m in sent[-1]["messages"] if m["role"] == "tool"]
        calls = [m["tool_calls"][0]["function"]["arguments"] for m in sent[-1]["messages"] if m.get("tool_calls")]
        if fix:
            assert "received" in tools[0]["content"] and tasks.CONV_BAD_CALLS in tools[2]["content"]
            assert tasks.CONV_BAD_CALLS not in tools[1]["content"] and calls == ['{"code": ""}'] * 3
        else:
            assert [t["content"] for t in tools] == ["Error: the call needs one string argument `code`."] * 3
            assert calls == [bad["arguments"]] * 3
        run.interp.close()


def test_tail_keeps_a_clipped_call_when_the_call_alone_is_past_the_budget():
    call_msg = {"role": "assistant", "content": None, "reasoning": "r" * 40000,
                "tool_calls": [{"id": "c1", "type": "function", "function": call("print(1)")}]}
    msgs = [{"role": "user", "content": "u" * 100}, call_msg, {"role": "tool", "tool_call_id": "c1", "content": "1"}]
    kept = conv.tail(msgs, 1000)  # conv1 21897d95#1: the tool result alone, popped -> nothing kept
    assert [m["role"] for m in kept] == ["assistant", "tool"] and kept[0]["tool_calls"] == call_msg["tool_calls"]
    assert kept[0]["reasoning"].startswith("[... earlier text cut]") and conv.approx_tokens(kept[0]) < 1100
    assert conv.tail(msgs, 10 ** 6) == msgs  # room for everything: unchanged


def test_compaction_keeps_head_summary_and_recent_tail():
    run = TaskRun("t#1", TASK, {"conversation": True})
    big = "y" * 36000  # ~12K tokens a reply at 3 chars a token
    eng, sent = engine([big, big, big, "SUMMARY: tried transpose", call("print(1)")], window=40000,
                       cap=8000, compact_keep=0.10, summary_prefill="World model:")
    asyncio.run(eng.advance(run, 10 ** 9, replies=5))
    assert run.compactions == 1 and run.replies[3]["outcome"] == "compaction"
    summary_request = sent[3]  # summary_prefill: v8.5p's prefilled-empty request (play prefix, tools kept, continued)
    assert summary_request["settings"]["tool"] is not False and summary_request["settings"]["continue"] is True
    assert summary_request["messages"][-2]["content"] == tasks.CONV_COMPACT
    assert summary_request["messages"][-1] == {"role": "assistant", "content": "World model:"}
    assert summary_request["messages"][:len(sent[2]["messages"])] == sent[2]["messages"]  # extends the play prefix
    after = sent[4]["messages"]
    assert after[:2] == run.head and "World model: " in after[2]["content"] and "SUMMARY: tried transpose" in after[2]["content"]
    assert sum(conv.approx_tokens(m) for m in after[3:]) <= 0.10 * 40000 + 200
    run.interp.close()


def test_compaction_without_prefill_is_the_tool_less_request():
    run = TaskRun("t#1", TASK, {"conversation": True})
    big = "y" * 36000
    eng, sent = engine([big, big, big, "SUMMARY: tried transpose", call("print(1)")], window=40000,
                       cap=8000, compact_keep=0.10)  # default: off
    asyncio.run(eng.advance(run, 10 ** 9, replies=5))
    summary_request = sent[3]
    assert summary_request["settings"]["tool"] is False and summary_request["messages"][-1]["content"] == tasks.CONV_COMPACT
    assert "SUMMARY: tried transpose" in sent[4]["messages"][2]["content"]
    run.interp.close()


def test_repl_restart_replays_clean_calls():
    r = Repl(TASK, strict=False, call_seconds=1)
    assert r.run("a = 41")["stdout"] == ""
    out = r.run("while True:\n    pass")
    assert "TimeoutError" in out["error"]
    assert r.run("print(a + 1)")["stdout"].strip() == "42"
    r.close()


def test_repl_fit_lists_functions_that_reproduce_every_training_pair():
    """fit probe (fit1 sim): train-fitting functions with their test outputs; others, crashes and state untouched."""
    r = Repl(TASK, strict=False, call_seconds=5)
    assert r.fit(2.0) is None  # no interpreter yet
    r.run("x = 1")
    assert r.fit(2.0) == []
    r.run("def ident(g):\n    return g\ndef boom(g):\n    raise ValueError\n"
          "def flip(g):\n    return [list(c) for c in zip(*g)]\nkeep = 3")
    assert r.fit(2.0) == [{"name": "flip", "tests": [[[7], [8], [9]]]}]
    assert r.run("print(keep)")["stdout"].strip() == "3"
    r.close()


def test_repl_fit_equivariance_flags_a_train_fitting_function_that_breaks_symmetry_on_the_test_input():
    """verifier probe: a transform kept by the training pairs must be kept on the test input too."""
    r = Repl(TASK, strict=False, call_seconds=5)
    r.run("def flip(g):\n    return [list(c) for c in zip(*g)]\n"
          "def odd(g):\n    t = [list(c) for c in zip(*g)]\n    return [[0]] * 3 if g == [[7, 8, 9]] else t")
    fits = {f["name"]: f for f in r.fit(5.0, equiv=5.0)}
    assert fits["odd"]["tests"] == [[[0], [0], [0]]] and fits["flip"]["tests"] == [[[7], [8], [9]]]
    assert fits["flip"]["equiv"]["transpose"] == [True, True] and fits["odd"]["equiv"]["transpose"] == [True, False, [[[0, 0], [1, 0], [2, 0]]], [[[7], [8], [9]]]]
    assert fits["flip"]["equiv"]["rot90"] == [False, None]
    assert all(v[1] for v in fits["flip"]["equiv"].values() if v[0])
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


def test_live_feed_carries_engine_metrics(tmp_path, monkeypatch):
    """arena/live/arc2-<trial>.json has an ``engine`` dict from /metrics:
    running, waiting, kv_cache_fraction, prefix_hit_rate (30-min window) and cumulative preemptions."""
    import threading
    import time

    from arc2 import trial as trial_mod
    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.05")
    monkeypatch.setattr(trial_mod, "PROGRESS_SECONDS", 0.3)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "validation"))
    out = tmp_path / "out"
    live = tmp_path / "validation/arena/live/arc2-h.json"

    def stopper():
        for _ in range(1200):
            if live.exists() and (json.loads(live.read_text()).get("engine") or {}).get("prefix_hit_rate"):
                (out / "stop-request").write_text("{}")
                return
            time.sleep(0.05)

    threading.Thread(target=stopper, daemon=True).start()
    asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                         deadline_epoch=time.time() + 300))
    engine = json.loads(live.read_text())["engine"]
    assert engine["running"] == 0 and engine["waiting"] == 0 and engine["kv_cache_fraction"] == 0.1
    assert engine["preemptions"] == 0 and engine["prefix_hit_rate"] == 0.9


def test_engine_prefix_hit_rate_uses_the_last_30_minutes(monkeypatch):
    from arc2 import trial as trial_mod

    class Client:
        base_url = __import__("httpx").URL("http://x/v1")

        def __init__(self):
            self.text = ""

        async def get(self, url, timeout):
            return type("R", (), {"text": self.text})()

    client, history, clock = Client(), [], [0.0]
    monkeypatch.setattr(trial_mod.time, "monotonic", lambda: clock[0])

    def at(t, hits, queries):
        clock[0] = t
        client.text = f"vllm:prefix_cache_hits_total {hits}\nvllm:prefix_cache_queries_total {queries}\n"
        return asyncio.run(trial_mod.engine_metrics(client, history))["prefix_hit_rate"]

    assert at(0, 0, 0) is None
    assert at(600, 100, 1000) == 0.1
    assert at(1900, 1100, 2000) == 0.55   # base still t=0 (t=600 is not yet 30 min old)
    assert at(2500, 1100, 2000) == 1.0  # base t=600, the newest sample at least 30 min old
    assert at(4000, 1600, 3000) == 0.5  # base t=1900


def test_live_wall_from_the_control_file_capped_by_the_machine_deadline(tmp_path, monkeypatch):
    """Checklist item 5: the wall is read from the control file each tick, capped by the machine
    deadline less the drain; a later, shorter wall ends the roster early."""
    import threading
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.05")
    trial["wall_seconds"] = 8  # 3 s: a loaded CI container spent it before play (startup's effort check)
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False}]  # never ends by itself
    control = tmp_path / "control.json"
    monkeypatch.setenv("ARENA_CONTROL", str(control))
    control.write_text(json.dumps({"wall_seconds": 10000}))  # past the machine deadline: capped
    out = tmp_path / "out"

    def shorten():  # once play has run past the launch wall
        for _ in range(2400):
            events = out / "events.jsonl"
            if events.exists() and len(events.read_text().splitlines()) >= 30 and time.time() - start > 10:
                control.write_text(json.dumps({"wall_seconds": 1}))
                return
            time.sleep(0.05)

    start = time.time()
    threading.Thread(target=shorten, daemon=True).start()
    began = time.time()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 300))
    changes = report["wall"]["changes"]
    assert [c["asked"] for c in changes] == [10000, 1] and changes[0]["wall"] == 225.0 and changes[1]["wall"] == 1
    assert changes[1]["at_seconds"] > 8 and report["wall"]["seconds"] == 1  # played past the launch wall, then ended
    assert time.time() - began < 200  # not the capped 225 s


def test_provisional_submit_keeps_the_task_open_and_logs_overwrites():
    """submit() records a provisional answer and the task goes on; the latest submit ranks first;
    each overwrite logs what it did to every output."""
    run = TaskRun("t#1", TASK, {"conversation": True, "submit_mode": "provisional"})
    eng, sent = engine([call("submit([[7, 8, 9]])"), call("submit(program_1(test_inputs[0]))"), RIGHT, call("print(1)"),
                        call("submit(program_1(test_inputs[0]), [[1]])"), call("submit([[0]], [[1]])")],
                       submit_mode="provisional")
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.done is None and run.parked and run.submits == 1
    assert sent[0]["messages"][0]["content"] == tasks.CONV_SYSTEM_PROVISIONAL
    assert tasks.tool_spec(sent[0]["settings"]) is tasks.CONV_TOOL_PROVISIONAL
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=5))
    assert run.done is None and run.submits == 3
    picks = guesses(run.candidates, [TASK["test"][0]["input"]])[0]
    assert [g for g, _ in picks] == [[[0]], [[1]]]  # the latest submit's two guesses
    # the second call fails (program_1 not yet defined); submit 2 is right but so was the pick before it (the
    # verified program filled guess 2); submit 3 replaces both guesses with wrong ones: right -> wrong
    assert [(o["before"], o["after"]) for o in run.overwrites] == [([True], [True]), ([True], [False])]
    run.interp.close()


def test_reopen_mode_gives_one_check_pass_with_the_other_copys_answer():
    """R: a submit parks the task for one check pass; the pass shows a differing answer of the other copy and ends at
    its next submit."""
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[7, 8, 9]])"), call("submit([[7], [8], [9]])"), call("submit([[7], [8], [9]], [[7, 8, 9]])")],
                       submit_mode="reopen")
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    asyncio.run(eng.advance(b, 10 ** 9, replies=1))
    assert a.awaiting_reopen and b.awaiting_reopen and a.done is None
    assert eng.queue_item(a, 0) == ((1, 0), -1)
    n = eng.pick_reopen([a, b])
    assert n in (0, 1) and [a, b][n].reopened and not [a, b][n].awaiting_reopen
    text = [a, b][n].messages[-1]["content"]
    assert text.startswith("[Check pass]") and "different answer" in text
    asyncio.run(eng.advance([a, b][n], 10 ** 9, replies=1))
    assert [a, b][n].done == "submitted" and len([a, b][n].overwrites) == 1
    for r in (a, b):
        if r.interp:
            r.interp.close()


def test_reopen_order_first_submit_takes_the_longest_to_submit_first():
    a, b, c = (TaskRun(f"t#{k}", TASK, {"conversation": True, "reopen_order": "first_submit"}) for k in (1, 2, 3))
    eng, _ = engine([], submit_mode="reopen")
    for run, tokens in ((a, 20000), (b, 90000), (c, 5000)):
        run.messages, run.awaiting_reopen, run.first_submit_tokens, run.candidates = [], True, tokens, []
        run.submits = 1
    eng.known = {r.id: r for r in (a, b, c)}
    assert eng.pick_reopen([a, b, c]) == 1 and eng.pick_reopen([a, b, c]) == 0 and eng.pick_reopen([a, b, c]) == 2


def test_tail_clip_off_keeps_conv1s_empty_tail():
    call_msg = {"role": "assistant", "content": None, "reasoning": "r" * 40000,
                "tool_calls": [{"id": "c1", "type": "function", "function": call("print(1)")}]}
    msgs = [{"role": "user", "content": "u"}, call_msg, {"role": "tool", "tool_call_id": "c1", "content": "1"}]
    assert conv.tail(msgs, 1000, clip_call=False) == [] and len(conv.tail(msgs, 1000)) == 2


def test_copy3_starts_after_both_copies_end_informed_on_one_half():
    import hashlib

    eng, _ = engine([], copy3="split")
    eng.record = lambda row: None
    tasks_ = {}
    for base in ("t0", "t1", "t3", "t4"):  # sha1 halves 0, 1, 0, 1
        a, b = TaskRun(f"{base}#1", TASK, {"name": "x"}), TaskRun(f"{base}#2", TASK, {"name": "x"})
        a.candidates = [{"submitted": 1, "submit_n": 1, "tests": [[[7], [8], [9]]], "t": 1.0},
                        {"verified": True, "code": "0123456789abcdef", "program": 1, "t": 0.5, "tests": [[[1]]]}]
        a.programs = ["def transform(g):\n    return g"]  # a candidate's code is the source's hash; the source is here
        tasks_[base] = (a, b)
        assert eng.copy3_for(a, [a, b]) is None  # neither has ended
        a.done = "submitted"
        assert eng.copy3_for(a, [a, b]) is None  # the other copy is still open
        b.done = "submitted"
        third = eng.copy3_for(b, [a, b])
        informed = int(hashlib.sha1(base.encode()).hexdigest()[:8], 16) % 2 == 0
        assert third.id == f"{base}#3" and third.arm["copy3_mode"] == ("informed" if informed else "blind")
        assert eng.copy3_for(b, [a, b, third]) is None  # one third copy a task
        eng.start(third, {**loop_defaults(), "images": False, "tool": True, "conversation": True})
        text = third.messages[1]["content"]
        assert (tasks.CONV_PRIOR in text and "Attempt 1 submitted" in text and "def transform(g)" in text
                and "Attempt 2 submitted:\n(no submit)" in text) == informed
        if third.interp:
            third.interp.close()
    off = TaskRun("eeee#1", TASK, {"name": "x"})
    off.done = "submitted"
    eng2, _ = engine([])
    assert eng2.copy3_for(off, [off, TaskRun("eeee#2", TASK, {"name": "x"})]) is None


def test_run_arc2_copy3_adds_third_copies_outside_the_readout(tmp_path, monkeypatch):
    """The real run_arc2 on the scripted server (it submits a wrong answer at call 2, closing each copy): every task
    whose two copies ended gets a third copy, informed on one sha1 half; tasks.jsonl carries the thirds after the
    read-out rows, the read-out stays on the 12 entries and the roster clock counts the thirds' tokens."""
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    from arc2 import trial as trial_mod

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.02")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "1")
    monkeypatch.setattr(trial_mod, "PROGRESS_SECONDS", 0.02)  # publish while thirds are in flight (cuts(), 27 Oct)
    real_summary, with_thirds = trial_mod.summary, []

    def spy(*args, **kw):  # a publish after the first third's reply: thirds are playing (in flight between them)
        events = tmp_path / "out" / "events.jsonl"
        if events.exists() and '#3", "attempt"' in events.read_text():
            with_thirds.append(1)
        return real_summary(*args, **kw)
    monkeypatch.setattr(trial_mod, "summary", spy)
    trial.update(copies=2)
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False, "copy3": "split"}]
    out = tmp_path / "out"
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 300))
    assert report["status"] == "complete" and not report["task_errors"], report.get("error")
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    thirds = [r for r in rows if r["task"].endswith("#3")]
    assert len(rows) == 18 and len(thirds) == 6 and rows[-1]["task"].endswith("#3")
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    modes = {e["task"]: e["mode"] for e in events if e.get("event") == "copy3"}
    assert sorted(modes) == sorted(f"t{i}#3" for i in range(6)) and set(modes.values()) == {"informed", "blind"}
    assert report["extra_tokens"] == sum(r["spent"] for r in thirds) > 0
    assert round(report["roster"]["tokens"]) == sum(r["spent"] for r in rows)
    assert report["tasks"] == 12
    assert len(with_thirds) >= 5  # the live feed kept publishing while thirds played (a KeyError ended it at once)


def loop_defaults():
    from arc2.loop import DEFAULTS
    return dict(DEFAULTS)


def test_reopen_passes_zero_later_submits_keep_the_slot():
    """the first submit parks the task for its pass; the pass's submit (a later one) never swaps:
    the task keeps iterating in its slot until its next compaction."""
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    sub = call("submit([[7], [8], [9]])")
    eng, sent = engine([sub, sub, sub, sub], submit_mode="reopen", reopen_passes=0)
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    asyncio.run(eng.advance(b, 10 ** 9, replies=1))
    assert a.parked and a.awaiting_reopen and b.parked
    picked = [a, b][eng.pick_reopen([a, b])]
    picked.parked = False
    asyncio.run(eng.advance(picked, 10 ** 9, replies=2))
    assert picked.done is None and not picked.parked and not picked.awaiting_reopen and picked.submits == 3
    assert picked.messages[-1]["content"].startswith("Recorded as your provisional answer (submission 3)")
    for r in (a, b):
        if r.interp:
            r.interp.close()


def test_swap_only_at_the_first_submit_in_every_mode():
    """swap at the first submit and at a compaction, never at a later submit."""
    sub = call("submit([[7], [8], [9]])")
    for mode in ("provisional", "fresh", "reopen"):
        run = TaskRun("t#1", TASK, {"conversation": True})
        eng, _ = engine([sub, sub, sub], submit_mode=mode, reopen_passes=0, firm=True)
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        assert run.parked, mode  # first submit: leaves its slot
        run.parked, run.awaiting_reopen = False, False
        asyncio.run(eng.advance(run, 10 ** 9, replies=2))
        assert not run.parked and run.done is None and run.submits == 3, mode
        run.interp.close()


def _roster_rows(out):
    path = out / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_run_arc2_provisional_swaps_at_submit_never_ends_and_backlog_prefers_unsubmitted(tmp_path, monkeypatch):
    import threading
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "1")
    study.update(active_limit=1)
    trial["tasks"] = trial["tasks"][:3]
    trial["arms"] = [{"name": "p", "conversation": True, "tool": True, "images": False, "submit_mode": "provisional"}]
    out = tmp_path / "out"

    def stopper():
        for _ in range(3000):
            if sum(r.get("outcome") == "submit" for r in _roster_rows(out)) >= 6:
                (out / "stop-request").write_text("{}")
                return
            time.sleep(0.05)

    threading.Thread(target=stopper, daemon=True).start()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    assert report["status"] == "complete" and not report["task_errors"]
    replies = [r for r in _roster_rows(out) if r.get("event") == "reply"]
    switches = [i for i in range(1, len(replies)) if replies[i]["task"] != replies[i - 1]["task"]]
    assert all(replies[i - 1]["outcome"] in ("submit", "compaction") for i in switches)
    blocks = [replies[0]["task"]] + [replies[i]["task"] for i in switches]
    assert len(set(blocks[:3])) == 3  # every task gets its first submit before any task comes back
    assert len(blocks) > 3
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert all(r["ended"] != "submitted" for r in rows)
    assert "overwrites" in report and sum(r.get("event") == "overwrite" for r in _roster_rows(out)) >= 1


def test_run_arc2_reopen_after_the_backlog_and_end_at_the_second_submit(tmp_path, monkeypatch):
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "1")
    study.update(active_limit=1)
    trial["tasks"] = trial["tasks"][:2]
    trial["arms"] = [{"name": "r", "conversation": True, "tool": True, "images": False, "submit_mode": "reopen"}]
    out = tmp_path / "out"
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    assert report["status"] == "complete" and not report["task_errors"]
    rows = _roster_rows(out)
    first_reopen = next(i for i, r in enumerate(rows) if r.get("event") == "reopen")
    firsts = [i for i, r in enumerate(rows) if r.get("outcome") == "submit" and r.get("submit_n") == 1]
    assert len(firsts) == 2 and max(firsts) < first_reopen  # reopen only once the backlog is empty
    assert sum(r.get("event") == "reopen" for r in rows) == 2
    ended = [json.loads(line)["ended"] for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert ended == ["submitted", "submitted"]


def test_run_arc2_continued_play_keeps_closed_tasks_closed_and_numbers_submits_on(tmp_path, monkeypatch):
    """arc2.kaggle plays again after an early end (all tasks closed, or the engine died) with the trial's ``state``:
    a closed task stays closed (no reply, its answers kept); a task that had submitted gets one more submit, numbered
    after its earlier ones, which closes it. The old state (no "conv") replayed every task from submit 1."""
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "1")
    study.update(active_limit=1)
    trial["tasks"] = trial["tasks"][:2]
    trial["arms"] = [{"name": "r", "conversation": True, "tool": True, "images": False, "submit_mode": "reopen"}]
    trial["state"] = str(tmp_path / "play-state.json")

    def play(name):
        out = tmp_path / name
        report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                      deadline_epoch=time.time() + 600))
        assert report["status"] == "complete" and not report["task_errors"], {k: v for k, v in report.items() if k in ("status", "error", "reason", "failures", "task_errors", "min_verified_tasks", "verified_tasks")}
        return _roster_rows(out), [json.loads(x) for x in (out / "tasks.jsonl").read_text().splitlines()]

    rows1, ended1 = play("p1")
    assert [r["ended"] for r in ended1] == ["submitted", "submitted"]
    rows2, ended2 = play("p2")  # everything closed: nothing is played again
    assert not [r for r in rows2 if r.get("event") == "reply"]
    answers = lambda ended: [[(c["tests"], c.get("submit_n")) for c in r["candidates"]] for r in ended]  # noqa: E731
    assert answers(ended2) == answers(ended1)
    rows2b, ended2b = play("p2b")  # and again: the state p2 wrote still says closed
    assert not [r for r in rows2b if r.get("event") == "reply"] and answers(ended2b) == answers(ended1)
    # a break after each task's first submit: the next play's submit is number 2 and closes the task
    state = json.loads(Path(trial["state"]).read_text())
    for row in state.values():
        row["conv"].update(done=None, submits=1, reopened=False, awaiting_reopen=True)
        row["candidates"] = [c for c in row["candidates"] if c.get("submit_n") in (None, 1)]
    Path(trial["state"]).write_text(json.dumps(state))
    rows3, ended3 = play("p3")
    subs = [r for r in rows3 if r.get("outcome") == "submit"]
    assert len(subs) == 2 and {r["submit_n"] for r in subs} == {2}
    assert not [r for r in rows3 if r.get("event") == "reopen"]
    assert [r["ended"] for r in ended3] == ["submitted", "submitted"]


def test_run_arc2_conversation_ends_at_the_wall_not_at_the_drain_timeout(tmp_path, monkeypatch):
    """Replies in flight at the wall: the wall's cancel meets the request's deadline timeout and can arrive as
    TimeoutError, which the worker takes for the wall and goes back to the queue still cancelling. The play must end
    at the wall; before the fix it waited DRAIN_SECONDS / 2 (v51 preview: killed at wall + 120, no report)."""
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="4")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    study.update(active_limit=2)
    trial["arms"] = [{"name": "r", "conversation": True, "tool": True, "images": False, "submit_mode": "reopen"}]
    trial["wall_seconds"] = 10
    began = time.time()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=tmp_path / "out", server_dir=server,
                                  base_url=url, deadline_epoch=time.time() + 10))
    assert report["replies"] > 0 and time.time() - began < 60, time.time() - began


def test_submit_check_runs_the_function_on_train_and_rejects_with_the_differences():
    """submit_check: submit(fn) is checked on every training pair; a failing function and bare grids are refused
    unless force=True; a passing function submits its test outputs, marked verified."""
    run = TaskRun("t#1", TASK, {"conversation": True, "submit_check": True})
    eng, sent = engine([call("def bad(g):\n    return g\nsubmit(bad)"),
                        call("submit([[7], [8], [9]])"),
                        call("def good(g):\n    return [list(r) for r in zip(*g)]\nsubmit(good, [[1]])")],
                       submit_check=True)
    rows = []
    eng.record = rows.append
    asyncio.run(eng.advance(run, 10 ** 9, replies=3))
    texts = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert texts[0].startswith("submit: your function does not reproduce") and "Pair 1" in texts[0]
    assert "force" not in texts[0] and texts[1] == "submit: " + tasks.CONV_SUBMIT_GRIDS and "force" not in texts[1]
    assert "force" not in sent[0]["messages"][0]["content"]  # no force path, none in the prompt
    assert run.done == "submitted" and tasks.CONV_SUBMIT_CHECK in sent[0]["messages"][0]["content"]
    first = next(c for c in run.candidates if c.get("submitted") == 1)
    assert first["tests"] == [[[7], [8], [9]]] and first["verified"]
    assert [r.get("submit_passed") for r in rows if r.get("event") == "reply"] == [0, None, 2]


GOOD = "def good(g):\n    return [list(r) for r in zip(*g)]\n"
ALT = ("def alt(g):\n    t = [list(r) for r in zip(*g)]\n    return t[::-1] if len(g[0]) == 3 else t\n")  # fits, differs


def test_submit_check2_needs_two_functions_that_fit_and_differ_then_exits_to_chk1_after_three_refusals():
    """submit() needs two functions that both reproduce every training pair and differ on
    the test input; one function, agreeing functions or a failing second are refused with the reason; after 3
    refusals the chk1 rule applies (one verified function, unchecked second guess)."""
    run = TaskRun("t#1", TASK, {"conversation": True, "submit_check": True, "submit_check2": True})
    eng, sent = engine([call(GOOD + "submit(good)"), call(GOOD + "submit(good, lambda g: g)"),
                        call(GOOD + ALT + "submit(good, alt)")], submit_check=True, submit_check2=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=3))
    texts = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert tasks.CONV_SUBMIT_CHECK2 in sent[0]["messages"][0]["content"]
    assert texts[0] == "submit: " + tasks.CONV_SUBMIT_TWO
    assert texts[1].startswith("submit: your second function does not reproduce")
    assert run.done == "submitted" and run.check2_refusals == 2
    subs = sorted((c for c in run.candidates if c.get("submitted")), key=lambda c: c["submitted"])
    assert [c["verified"] for c in subs] == [True, True] and subs[0]["tests"] != subs[1]["tests"]
    run = TaskRun("t#1", TASK, {"conversation": True, "submit_check": True, "submit_check2": True})
    eng, sent = engine([call(GOOD + "submit(good, good)")] * 3 + [call(GOOD + "submit(good, [[1]])")],
                       submit_check=True, submit_check2=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=4))
    assert [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"][0] == "submit: " + tasks.CONV_SUBMIT_SAME
    assert run.done == "submitted" and run.check2_refusals == 3  # exit: chk1's rule
    subs = sorted((c for c in run.candidates if c.get("submitted")), key=lambda c: c["submitted"])
    assert [c["verified"] for c in subs] == [True, False]


def test_submit_check2_exits_at_the_token_budget():
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent, run.check2_from, run.check2_budget, run.check2_refusals = 1499, 1000, 500, 1
    assert not conv.ConvEngine.check2_exited(run, {})
    run.spent = 1500
    assert conv.ConvEngine.check2_exited(run, {})


def test_submit_check2_exit_knobs():  # chk4: refusal exit at 10; token exit factor, None = off
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent, run.check2_from, run.check2_budget, run.check2_refusals = 1000, 1000, 500, 3
    assert conv.ConvEngine.check2_exited(run, {"check2_max_refusals": 3})
    assert not conv.ConvEngine.check2_exited(run, {"check2_max_refusals": 10})
    run.check2_refusals = 10
    assert conv.ConvEngine.check2_exited(run, {"check2_max_refusals": 10})
    run.check2_refusals, run.spent = 1, 1500
    assert conv.ConvEngine.check2_exited(run, {"check2_max_refusals": 10, "check2_token_exit": 1.0})
    assert not conv.ConvEngine.check2_exited(run, {"check2_max_refusals": 10, "check2_token_exit": 2.0})
    assert not conv.ConvEngine.check2_exited(run, {"check2_max_refusals": 10, "check2_token_exit": None})


def test_submit_check_force_is_refused_by_default():
    """force=True passes nothing; a failing function and grids alone are refused, also with it,
    also after check2's exit (the plain check has no force either); the entry keeps working."""
    for script, knobs in (([call("submit(lambda g: g, force=True)"), call("submit([[7], [8], [9]], force=True)")], {}),
                          ([call(GOOD + "submit(good)"), call("submit(lambda g: g, force=True)")],
                           dict(submit_check2=True, check2_max_refusals=1, check2_token_exit=None))):
        run = TaskRun("t#1", TASK, {"conversation": True, "submit_check": True})
        eng, sent = engine(script, submit_check=True, **knobs)
        rows = []
        eng.record = rows.append
        asyncio.run(eng.advance(run, 10 ** 9, replies=2))
        replies = [r for r in rows if r.get("event") == "reply"]
        assert run.done is None and run.submits == 0 and not any(c.get("submitted") for c in run.candidates)
        assert replies[-1]["force_refused"] and not replies[-1]["submit_forced"]
        assert all("force" not in m["content"] for m in sent[-1]["messages"] if isinstance(m.get("content"), str))


def test_submit_check_force_submits_a_failing_function_with_allow_force():  # chk1-chk6 bundles
    run = TaskRun("t#1", TASK, {"conversation": True, "submit_check": True})
    eng, sent = engine([call("submit(lambda g: g, force=True)")], submit_check=True, allow_force=True)
    rows = []
    eng.record = rows.append
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.done == "submitted" and rows[-1]["submit_forced"] is True
    assert not next(c for c in run.candidates if c.get("submitted") == 1)["verified"]


def test_run_arc2_submit_check_on_the_scripted_server(tmp_path, monkeypatch):
    """The stub submits grids first (refused under submit_check), then its answer function (checked, accepted)."""
    import time

    from arc2.trial import run_arc2
    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.01")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "1")
    study.update(active_limit=2)
    trial["tasks"] = trial["tasks"][:2]
    trial["arms"] = [{"name": "s", "conversation": True, "tool": True, "images": False, "submit_check": True}]
    out = tmp_path / "out"
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 600))
    assert report["status"] == "complete" and not report["task_errors"]
    rows = _roster_rows(out)
    assert sum((r.get("tool_result") or "").startswith("submit: nothing was submitted") for r in rows) == 2
    assert sum(r.get("outcome") == "submit" and r.get("submit_passed") == r.get("submit_pairs") for r in rows) == 2
    ended = [json.loads(line)["ended"] for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert ended == ["submitted", "submitted"] and report["final"]["score_outputs"] == 100


def _close(*runs):
    for r in runs:
        if getattr(r, "interp", None):
            r.interp.close()


def test_churn_same_answer_three_times_goes_stale_then_closes():
    """churn rule: three identical submits in a row make the task stale (back of the queue); stale a
    second time, it is closed."""
    run = TaskRun("t#1", TASK, {"conversation": True})
    sub = call("submit([[7], [8], [9]])")
    eng, sent = engine([sub] * 6, submit_mode="provisional", firm=True, churn=True)
    for k in range(3):
        run.parked = False
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert getattr(run, "stale", False) and run.done is None and run.stale_count == 1
    assert eng.priority(run)[0] == 3  # submitted + stale: after every fresh and submitted task
    assert sent[1]["messages"][-1]["content"].startswith("Recorded as your provisional answer (submission 1). It is not final")
    for k in range(3):
        run.parked = False
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.done == "stale"
    _close(run)


def test_churn_short_replies_without_change_go_stale():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call("print(1)")] * 8 + [call("submit([[1]])")] + [call("print(1)")] * 8,
                    submit_mode="provisional", churn=True)  # 100-token tool replies
    asyncio.run(eng.advance(run, 10 ** 9, replies=8))
    assert getattr(run, "stale_count", 0) == 0  # never submitted: kept indefinitely
    asyncio.run(eng.advance(run, 10 ** 9, replies=9))
    assert run.stale_count == 1 and run.stale and run.done is None  # no swap of its own
    _close(run)


def test_open_order_first_submit_ranks_submitted_tasks_by_longest_first_submit():
    a, b, c = (TaskRun(f"t#{k}", TASK, {"conversation": True}) for k in (1, 2, 3))
    eng, _ = engine([], submit_mode="provisional", open_order="first_submit")
    for run, first, spent in ((a, 20000, 90000), (b, 90000, 95000), (c, None, 99000)):
        run.submits, run.first_submit_tokens, run.spent = (1 if first else 0), first, spent
    assert sorted((a, b, c), key=eng.priority) == [c, b, a]


def test_fresh_mode_pass_is_a_new_conversation_with_only_this_copys_program_and_answer():
    """a submit parks the task; its pass is a fresh conversation showing its own last program
    and submitted answer, never the other copy's; passes repeat until the budget ends."""
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    eng, sent = engine([WRONG, call("submit([[7, 8, 9]])"), call("submit([[1]])"), call("submit([[7], [8], [9]])")],
                       submit_mode="fresh", reopen_order="first_submit")
    asyncio.run(eng.advance(a, 10 ** 9, replies=2))
    asyncio.run(eng.advance(b, 10 ** 9, replies=1))
    assert a.awaiting_reopen and a.done is None and b.awaiting_reopen
    a.first_submit_tokens, b.first_submit_tokens = 50000, 10
    assert eng.pick_reopen([a, b]) == 0
    first = a.messages[1]["content"]
    assert len(a.messages) == 2 and tasks.CONV_FRESH in first and "return grid" in first
    assert "7 8 9" in first and "\n1\n" not in first + "\n"  # its own answer, not the other copy's [[1]]
    a.parked = False  # the worker clears it when the task leaves its slot
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert a.done is None and not a.awaiting_reopen and not a.parked and a.submits == 2  # iterates on in its slot
    _close(a, b)


def test_naive_fresh_each_reopens_fresh_after_every_submit_with_only_its_program_and_answer():
    """every submit ends the pass; the next pass is a fresh conversation whose only change
    is the earlier program and submitted answer (CONV_NAIVE); with keep_first submission 1's first guess stays guess 2."""
    inputs = [t["input"] for t in TASK["test"]]
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([WRONG, call("submit([[7, 8, 9]])"), call("submit([[1]])"), call("submit([[2]])")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True)
    asyncio.run(eng.advance(a, 10 ** 9, replies=2))
    assert a.awaiting_reopen and a.parked and a.submits == 1
    for k, grid in ((2, "\n1\n"), (3, None)):
        assert eng.pick_reopen([a]) == 0
        first = a.messages[1]["content"]
        assert len(a.messages) == 2 and tasks.CONV_NAIVE in first and tasks.CONV_FRESH not in first
        assert "return grid" in first and (k == 2 and "7 8 9" in first or k == 3 and "\n1" in first)
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        assert a.submits == k and a.awaiting_reopen and a.parked and a.done is None  # every submit ends the pass
    assert [g for g, _ in guesses(a.candidates, inputs)[0]] == [[[2]], [[7, 8, 9]]]  # S1 pinned as guess 2
    _close(a)


def test_naive_blank_pass_is_a_completely_fresh_attempt():
    """fresh_text "blank": the reopened pass is the bare task prompt (no earlier program or answer); S1 stays guess 2."""
    inputs = [t["input"] for t in TASK["test"]]
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([WRONG, call("submit([[7, 8, 9]])"), call("submit([[2]])")],
                    submit_mode="fresh", fresh_each=True, fresh_text="blank", keep_first=True)
    asyncio.run(eng.advance(a, 10 ** 9, replies=2))
    head = list(a.head)
    assert eng.pick_reopen([a]) == 0
    assert a.messages == head and tasks.CONV_NAIVE not in a.messages[1]["content"]  # the bare task prompt again
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert a.submits == 2 and a.awaiting_reopen
    assert [g for g, _ in guesses(a.candidates, inputs)[0]] == [[[2]], [[7, 8, 9]]]
    _close(a)


def test_naive_blank_view_pass_runs_on_augmented_view_k_and_maps_back():
    """fresh_view (naive5): blank pass k shows augmented view k; its submit is stored in the task's own frame."""
    from arc2.conv import to_view
    inputs = [t["input"] for t in TASK["test"]]
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([WRONG, call("submit([[7, 8, 9]])")], submit_mode="fresh", fresh_each=True,
                    fresh_text="blank", fresh_view=True, keep_first=True)
    asyncio.run(eng.advance(a, 10 ** 9, replies=2))
    assert eng.pick_reopen([a]) == 0 and a.view_k == 1 and a.shown != TASK
    assert tasks.text(a.shown["train"][0]["input"]) in a.messages[1]["content"]
    shown = to_view(a, [[2, 3]])  # the model answers in the view; the candidate is in the task's frame
    eng2, _ = engine([call(f"submit({json.dumps(shown)})")], submit_mode="fresh", fresh_each=True,
                     fresh_text="blank", fresh_view=True, keep_first=True)
    a.parked = False
    eng2.known = eng.known
    asyncio.run(eng2.advance(a, 10 ** 9, replies=1))
    assert [g for g, _ in guesses(a.candidates, inputs)[0]][0] == [[2, 3]]
    _close(a)


def test_partner_text_off_hides_the_other_copys_answer():
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("t#2", TASK, {"conversation": True})
    eng, _ = engine([call("submit([[7, 8, 9]])"), call("submit([[7], [8], [9]])")], submit_mode="reopen",
                    partner_text=False, reopen_passes=0)
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    asyncio.run(eng.advance(b, 10 ** 9, replies=1))
    n = eng.pick_reopen([a, b])
    assert [a, b][n].messages[-1]["content"] == tasks.CONV_REOPEN
    _close(a, b)


def test_balance_order_shares_extra_tokens_by_first_submit_and_no_task_holds_the_head():
    """240 submitted tasks, first submit 5K-120K, replies 300-16K. With the
    balanced key no task runs more than 3 turns in a row, and spent / first_submit ends within one maximum reply /
    the smallest first submit across tasks; longest-first alone keeps one task at the head."""
    import random

    rng = random.Random(7)
    eng, _ = engine([], submit_mode="provisional", open_order="balance")

    def simulate(key, turns=20000):
        runs = []
        for k in range(240):
            r = TaskRun(f"t{k}#1", TASK, {"conversation": True})
            r.submits, r.first_submit_tokens = 1, rng.randint(5000, 120000)
            r.spent = r.first_submit_tokens
            runs.append(r)
        streak, longest, last = 0, 0, None
        for _ in range(turns):
            head = min(runs, key=key)
            streak = streak + 1 if head is last else 1
            longest, last = max(longest, streak), head
            head.spent += rng.randint(300, 16000)
        ratios = [r.spent / r.first_submit_tokens for r in runs]
        return longest, max(ratios) - min(ratios)

    longest, spread = simulate(eng.priority)
    assert longest <= 3 and spread <= 16000 / 5000, (longest, spread)
    first_only = lambda r: -r.first_submit_tokens  # noqa: E731
    assert simulate(first_only)[0] > 1000  # longest-first: one task keeps the head
    a, b = TaskRun("a#1", TASK, {"conversation": True}), TaskRun("b#1", TASK, {"conversation": True})
    eng2, _ = engine([], submit_mode="reopen", reopen_order="balance")
    for r, first, spent in ((a, 90000, 100000), (b, 10000, 20000)):
        r.messages, r.awaiting_reopen, r.first_submit_tokens, r.spent, r.candidates, r.submits = [], True, first, spent, [], 1
    eng2.known = {r.id: r for r in (a, b)}
    assert eng2.pick_reopen([a, b]) == 0  # 1.11 < 2.0


def test_rounds_order_gives_every_task_round_k_before_any_gets_k_plus_1_hardest_first():
    """submitted tasks by (round asc, tokens-to-first-submit desc)."""
    runs = [TaskRun(f"t{k}#1", TASK, {"conversation": True}) for k in range(4)]
    eng, _ = engine([], submit_mode="provisional", open_order="rounds", swap="budget")
    for r, (rnd, first) in zip(runs, ((2, 90000), (2, 10000), (3, 99000), (1, None))):
        r.round, r.first_submit_tokens, r.submits = rnd, first, (1 if first else 0)
    assert sorted(runs, key=eng.priority) == [runs[3], runs[0], runs[1], runs[2]]


def test_budget_swap_ends_a_later_round_at_its_budget_not_at_a_compaction_and_verified_twice_swaps():
    sub = call("submit(program_1(test_inputs[0]))")
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([RIGHT, sub, call("print(1)"), call("print(2)"), sub, sub], submit_mode="provisional",
                       swap="budget", open_order="rounds", firm=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))  # program, first submit: round 1 ends
    assert run.parked and run.round == 2 and run.first_submit_tokens == 1100
    assert not eng.swaps_at_compaction(run) and eng.swaps_at_compaction(TaskRun("u#1", TASK, {}))
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # 100 tokens of 1,100: the round goes on
    assert not run.parked and run.round == 2
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))  # a verified submit, then the same verified answer again
    assert run.parked and run.round == 3  # verified twice
    run.interp.close()


def test_budget_swap_at_the_round_budget_with_compaction_first():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[7], [8], [9]])"), call("print(1)"), call("print(2)"), "summary"],
                       submit_mode="provisional", swap="budget", compact_on_swap=True, compact_keep=0.1,
                       summary_effort="low", tail_clip=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.parked and run.round == 2
    run.parked, run.first_submit_tokens = False, 150  # round budget 150: two 100-token replies reach it
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert not run.parked
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.parked and run.round == 3 and run.compactions == 1  # compacted, then parked
    run.interp.close()


def test_forced_provisional_submit_after_two_compactions():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([RIGHT, "summary one", "summary two"], submit_mode="provisional", force_submit_after=2,
                    compact_keep=0.1, summary_effort="low", tail_clip=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    s = {**eng.settings, "tool": True}
    asyncio.run(eng.compact(run, s))
    assert run.submits == 0
    asyncio.run(eng.compact(run, s))
    assert run.submits == 1 and run.first_submit_tokens is not None
    run.interp.close()


def test_budget_floor_is_one_reply_cap_and_forced_submit_after_two_compactions():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([], submit_mode="provisional", swap="budget", budget_floor=True)
    run.first_submit_tokens = 3000
    assert eng.round_budget(run, {**eng.settings, "cap": None}) == 16384
    assert eng.round_budget(run, {**eng.settings, "budget_floor": False}) == 3000


def test_rounds_same_answer_twice_swaps_without_verification():
    """any identical answer submitted twice ends the round (no verification needed)."""
    sub = call("submit([[1]])")
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([sub, call("submit([[2]])"), call("submit([[2]])")], submit_mode="provisional", swap="budget",
                    open_order="rounds", same_answer_swap=True, budget_floor=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.parked and run.round == 2
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # a different answer: the round goes on
    assert not run.parked
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # the same answer again: swap
    assert run.parked and run.round == 3
    run.interp.close()


def test_budget_swap_compacts_only_a_context_over_the_threshold():
    for over, compacted in ((10 ** 6, 0), (10, 1)):
        run = TaskRun("t#1", TASK, {"conversation": True})
        eng, _ = engine([call("submit([[1]])"), call("print(1)"), call("print(2)"), "summary"],
                        submit_mode="provisional", swap="budget", compact_on_swap=True, compact_over=over,
                        compact_keep=0.1, summary_effort="low", tail_clip=True)
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        run.parked, run.first_submit_tokens = False, 150
        asyncio.run(eng.advance(run, 10 ** 9, replies=2))
        assert run.parked and run.compactions == compacted, over
        run.interp.close()


def test_copy_views_second_copy_works_on_an_augmented_view_and_answers_map_back():
    """conv1 analysis item (28 Oct): copy 2 sees view 1 (symmetry + colours); programs and submits are stored in the
    task's own frame, so a right answer in the view is right in the read."""
    from arc2.loop import augment_view

    run = TaskRun("t#2", TASK, {"conversation": True})
    view = augment_view(TASK, "t#2", 1)
    run.view_k = 1
    right_in_view = conv.to_view(run, TASK["test"][0]["output"])
    assert right_in_view != TASK["test"][0]["output"] and conv.to_task(run, right_in_view) == TASK["test"][0]["output"]
    eng, sent = engine([call("print(test_inputs[0])"), call(f"submit({right_in_view!r})")], submit_mode="close",
                       copy_views=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    assert run.view_k == 1 and tasks.task_prompt(view) in sent[0]["messages"][1]["content"]
    assert str(view["test"][0]["input"]) in sent[1]["messages"][-1]["content"]  # the interpreter holds the view
    sub = next(c for c in run.candidates if c.get("submitted") == 1)
    assert sub["tests"] == [TASK["test"][0]["output"]] and run.done == "submitted"
    run1 = TaskRun("t#1", TASK, {"conversation": True})
    eng.start(run1, {**loop_defaults(), "images": False, "tool": True, "conversation": True, "copy_views": True})
    assert run1.view_k == 0 and run1.shown is TASK
    run1.interp.close()


def test_arc2_hint_in_the_conversation_system_prompt_and_code_now_cut_text():
    assert tasks.ARC2_HINT in tasks.conv_system({"prompt": "arc2"})
    assert tasks.ARC2_HINT not in tasks.conv_system({})
    assert tasks.CONV_SHORT_TURNS in tasks.conv_system({"short_turns": True})
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine(["", "", RIGHT], cut_mode="code_now")  # "" = a reply cut at the length limit
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.messages[-1]["content"] == tasks.CONV_CUT_CODE
    run.interp.close()


def test_round_scoped_same_answer_swap_ignores_the_round_1_submit():
    """(P4/R4: every later round ended at its first resubmit): with ``same_answer_scope`` "round" the
    round-1 answer resubmitted once in round 2 does not swap; the same answer twice inside round 2 does."""
    sub = call("submit([[1]])")
    for scope, parked_after_one in (("task", True), ("round", False)):
        run = TaskRun("t#1", TASK, {"conversation": True})
        eng, _ = engine([sub, sub, sub], submit_mode="provisional", swap="budget", open_order="rounds",
                        same_answer_swap=True, same_answer_scope=scope, budget_floor=True)
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        assert run.parked and run.round == 2
        run.parked = False
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # the round-1 answer once in round 2
        assert run.parked == parked_after_one, scope
        if not parked_after_one:
            asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # twice in round 2: swap
            assert run.parked and run.round == 3
        run.interp.close()


def test_budget_context_makes_the_round_budget_at_least_the_resumed_context():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([], submit_mode="provisional", swap="budget", budget_floor=True)
    run.first_submit_tokens, run.round_context = 3000, 40000
    s = {**eng.settings, "cap": None}
    assert eng.round_budget(run, s) == 16384
    assert eng.round_budget(run, {**s, "budget_context": True}) == 40000


def test_budget_context_is_read_from_the_resume_reply_and_reset_at_the_round_end():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call("submit([[7], [8], [9]])"), call("print(1)"), call("print(2)")], submit_mode="provisional",
                    swap="budget", budget_context=True, budget_floor=False)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.parked and run.round == 2 and run.round_context is None
    run.parked, run.first_submit_tokens = False, 150  # ttfs budget 150 would end the round after two replies
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    assert run.round_context == run.replies[1]["prompt_tokens"] > 200
    assert not run.parked and run.round == 2  # budget = the resumed context, not 150
    run.interp.close()


def test_compacted_never_submitted_task_goes_to_the_head_of_the_backlog():
    """a compaction ends round 1, and a never-submitted task that compacted is picked up next."""
    runs = [TaskRun(f"t{k}#1", TASK, {"conversation": True}) for k in range(4)]
    eng, _ = engine([], submit_mode="provisional", open_order="rounds", swap="budget", compacted_first=True)
    fresh, compacted, submitted, later = runs
    fresh.spent = 0
    compacted.spent, compacted.compactions, compacted.compacted_at = 90000, 1, 5.0
    later.spent, later.compactions, later.compacted_at = 50000, 1, 9.0
    submitted.submits, submitted.round, submitted.first_submit_tokens = 1, 2, 30000
    assert sorted(runs, key=eng.priority) == [compacted, later, fresh, submitted]
    eng2, _ = engine([], submit_mode="provisional", open_order="rounds", swap="budget")
    assert sorted(runs, key=eng2.priority)[0] is fresh  # without the setting: by tokens spent


TASK2 = {"train": TASK["train"], "test": [{"input": [[7, 8, 9]], "output": [[7], [8], [9]]},
                                          {"input": [[1, 2]], "output": [[1], [2]]}]}
NOREP = dict(submit_mode="provisional", swap="budget", open_order="rounds", same_answer_swap=True,
             same_answer_scope="round", budget_floor=True, budget_context=True, no_repeat=True, round_prompt=True)


def last_tool(run):
    return next(m["content"] for m in reversed(run.messages) if m["role"] == "tool")


def test_no_repeat_refuses_an_identical_later_submit_and_the_round_goes_on():
    """in rounds >= 2 submit() refuses an answer already submitted; the refusal does not end the round."""
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[1]])"), call("submit([[1]])"), call("submit([[2]])")], **NOREP)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.parked and run.round == 2 and last_tool(run) == tasks.CONV_RECORDED.format(n=1)
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # identical: refused, round goes on
    assert run.submits == 1 and not run.parked and run.round == 2 and run.repeats_refused == 1
    assert last_tool(run).startswith("submit: nothing was submitted: this answer is the same")
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # different: recorded, first guess; the old one second
    assert run.submits == 2 and not run.parked and last_tool(run) == tasks.CONV_NEXT.format(n=2)
    assert [g for g, _ in guesses(run.candidates, [t["input"] for t in TASK["test"]])[0]] == [[[2]], [[1]]]
    run.interp.close()


def test_no_repeat_multi_output_partial_overlap_is_accepted_full_overlap_refused():
    run = TaskRun("t#1", TASK2, {"conversation": True})
    eng, _ = engine([call("submit([[[1]], [[5]]])"), call("submit([[[1]], [[6]]])"),
                     call("submit([[[1]], [[5]]], [[[1]], [[6]]])")], **NOREP)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # output 1 repeats, output 2 is new: accepted
    assert run.submits == 2 and getattr(run, "repeats_refused", 0) == 0
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # every grid of both outputs was submitted before: refused
    assert run.submits == 2 and run.repeats_refused == 1
    assert "every output of this answer is the same" in last_tool(run)
    run.interp.close()


def test_keep_first_pins_submission_one_first_guess_as_guess_two_on_every_later_submit():
    """(p7's right->wrong): submission 1's first guess S1 stays guess 2 on every later submit, unless the
    new first guess equals it (then the model's own second guess). Patterns as observed: 2c181942#1 / 409aa875#2
    (S2, then S3: was [S3, S2]), 2c181942#2 (S2, then submit(new, strong) with strong = S1: strong was ranked below S2)."""
    inputs = [t["input"] for t in TASK["test"]]
    cases = [  # later submits after submit([[1]]) -> expected guesses after the last
        ([call("submit([[2]])"), call("submit([[3]])")], [[[3]], [[1]]], [[[3]], [[2]]]),
        ([call("submit([[2]])"), call("submit([[3]], [[1]])")], [[[3]], [[1]]], [[[3]], [[1]]]),
        ([call("submit([[2]], [[4]])"), call("submit([[3]], [[5]])")], [[[3]], [[1]]], [[[3]], [[5]]]),
        ([call("submit([[2]])"), call("submit([[1]], [[4]])")], [[[1]], [[4]]], [[[1]], [[4]]]),  # new first = S1
    ]
    for later, pinned, unpinned in cases:
        for keep, want in ((True, pinned), (False, unpinned)):
            run = TaskRun("t#1", TASK, {"conversation": True})
            eng, sent = engine([call("submit([[1]])")] + later, **{**NOREP, "keep_first": keep})
            for _ in range(1 + len(later)):
                asyncio.run(eng.advance(run, 10 ** 9, replies=1))
                run.parked = False
            assert run.submits == 3 and [g for g, _ in guesses(run.candidates, inputs)[0]] == want, (later, keep)
            if keep:
                assert "first submission's first guess" in last_tool(run)
                text = eng.round_text(run, {**(eng.settings or {}), **run.arm, "keep_first": True})
                assert "Submission 1, first guess" in text and "first submission's first guess" in text
            run.interp.close()


def test_no_repeat_is_off_by_default_and_first_submit_is_never_refused():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call("submit([[1]])"), call("submit([[1]])")], submit_mode="provisional")
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    assert run.submits == 2


def test_round_prompt_shows_earlier_answers_once_per_round_with_budget_and_no_reopen_text():
    sub = call("submit(program_1(test_inputs[0]))")
    for mode in ("provisional", "reopen"):
        run = TaskRun("t#1", TASK, {"conversation": True})
        eng, sent = engine([RIGHT, sub, call("print(1)"), call("print(2)")],
                           **{**NOREP, "submit_mode": mode, "reopen_passes": 0, "partner_text": False})
        eng.known = {run.id: run}
        asyncio.run(eng.advance(run, 10 ** 9, replies=2))
        assert run.parked and run.round == 2
        run.parked = False
        if mode == "reopen":
            assert eng.pick_reopen([run]) == 0
        asyncio.run(eng.advance(run, 10 ** 9, replies=2))
        users = [m["content"] for m in sent[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], str)]
        rounds = [u for u in users if u.startswith("[Round 2]")]
        assert len(rounds) == 1, mode  # once, not at every reply
        assert "Submission 1, first guess (produced by a program that reproduces every training pair)" in rounds[0]
        assert "16,384 tokens" in rounds[0] and not any(u.startswith("[Check pass]") for u in users)
        assert "submit again" not in json.dumps(sent[-1]["messages"])
        run.interp.close()


def test_repeat_ends_round_setting_two_identical_tries_in_a_round_end_it():
    """targets.md: a refused repeat goes on, but the same answer tried twice within the round ends it."""
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call("submit([[1]])"), call("submit([[1]])"), call("submit([[1]])")], **NOREP, repeat_ends_round=True)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # refused once (the round-1 answer): the round goes on
    assert not run.parked and run.round == 2
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))  # the same answer tried again in this round: round ends
    assert run.parked and run.round == 3 and run.submits == 1 and run.repeats_refused == 2
    run.interp.close()


def test_second_attempt_spawns_after_every_entry_started_and_merges_by_agreement_and_verification():
    """an entry that submitted after > ``second_attempt`` tokens gets one fresh, independent second
    attempt (its own run, not the roster's other copy) once every entry has started; agree -> unchanged, differ -> the
    verified one first, else the entry's first guess first and the second attempt's as guess 2."""
    inputs = [t["input"] for t in TASK["test"]]
    eng, sent = engine([call("submit([[1]])"), call("submit([[2]])"), call("submit([[1]])")],
                       second_attempt=50, second_view=True)
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("u#1", TASK, {"conversation": True})
    assert eng.seconds_due([a, b]) == []  # b has not started
    asyncio.run(eng.advance(a, 10 ** 9))
    assert a.done == "submitted" and a.first_submit_tokens > 50
    b.messages = []  # started
    (second,) = eng.seconds_due([a, b])
    assert second.id == "t#b1" and second.arm["b_attempt"] and second.messages is None and a.pair is second
    assert eng.seconds_due([a, b]) == []  # once
    asyncio.run(eng.advance(second, 10 ** 9))
    assert second.done == "submitted" and second.view_k == 1
    assert not any("[[1]]" in str(m.get("content")) and "submitted" in str(m.get("content")) for m in sent[-1]["messages"])
    g2 = next(c for c in second.candidates if c.get("submitted") == 1)["tests"][0]  # mapped back from its view
    assert g2 != [[1]]
    assert [g for g, _ in guesses(a.candidates, inputs)[0]] == [[[1]], g2]  # differ, neither verified: entry first
    next(c for c in second.candidates if c.get("submitted") == 1)["verified"] = True
    eng.merge_second(second)
    assert [g for g, _ in guesses(a.candidates, inputs)[0]] == [g2, [[1]]]  # the verified second attempt first
    assert sum(c.get("b2", False) for c in a.candidates) == 1  # redone, not stacked


def test_second_attempt_agreeing_answer_changes_nothing_and_never_submitted_entries_qualify():
    inputs = [t["input"] for t in TASK["test"]]
    eng, _ = engine([call("submit([[1]])"), call("submit([[1]])")], second_attempt=50)
    a = TaskRun("t#1", TASK, {"conversation": True})
    asyncio.run(eng.advance(a, 10 ** 9))
    (second,) = eng.seconds_due([a])
    asyncio.run(eng.advance(second, 10 ** 9))
    assert not any(c.get("b2") for c in a.candidates) and [g for g, _ in guesses(a.candidates, inputs)[0]] == [[[1]]]
    open_ = TaskRun("x#1", TASK, {"conversation": True})
    open_.messages, open_.spent, open_.submits = [], 10, 0
    assert eng.seconds_due([open_]) == []
    open_.spent = 60
    assert len(eng.seconds_due([open_])) == 1


def test_rounds_after_all_submitted_holds_later_rounds_until_every_playing_entry_has_submitted():
    eng, _ = engine([], rounds_after_all_submitted=True)
    a, b, c = (TaskRun(f"{k}#1", TASK, {"conversation": True}) for k in "abc")
    a.submits, b.submits, c.submits = 1, 0, 0
    a.done = b.done = c.done = None
    assert eng.holds_round(a, [a, b, c]) and not eng.holds_round(b, [a, b, c])  # b, c never submitted
    c.done = "errors"
    b.submits = 1
    assert not eng.holds_round(a, [a, b, c])  # every entry still playing has submitted
    off, _ = engine([])
    b.submits = 0
    assert not off.holds_round(a, [a, b, c])  # off by default


def test_multi_attempt_spawns_by_backlog_reenters_and_merges_all_attempts():
    """b2: once every entry has started, free slots start fresh attempts (attempt k on view
    k - 1) for the backlog entries (multi.pick); an entry re-enters after its attempt submits; the entry's guesses are
    the merge of all its attempts' first guesses."""
    inputs = [t["input"] for t in TASK["test"]]
    eng, _ = engine([call("submit([[1]])")] + [call("submit([[2]])")] * 3, multi_attempt=True, multi_view=True)
    a, b = TaskRun("t#1", TASK, {"conversation": True}), TaskRun("u#1", TASK, {"conversation": True})
    assert eng.attempts_due([a, b], 4) == []  # b has not started
    asyncio.run(eng.advance(a, 10 ** 9))  # a submits [[1]] (close mode: done)
    b.messages, b.spent = [], a.first_submit_tokens + 1  # b started, never submitted, spent past a's ttfs: first
    first = eng.attempts_due([a, b], 1)
    assert [r.id for r in first] == ["u#b2x1"] and first[0].entry is b
    (second,) = eng.attempts_due([a, b], 3)  # b has an attempt running: only a is in the backlog
    assert second.id == "t#b2x1" and second.arm["attempt_k"] == 2
    asyncio.run(eng.advance(second, 10 ** 9))
    assert second.view_k == 1 and second.done == "submitted"
    g2 = next(c for c in second.candidates if c.get("submitted") == 1)["tests"][0]  # its answer in the task frame
    assert [g for g, _ in guesses(a.candidates, inputs)[0]] == [[[1]], g2]  # differ, unverified: attempt 1 first
    (third,) = eng.attempts_due([a, b], 1)  # a re-enters after its attempt submitted
    assert third.id == "t#b3x1" and len(a.attempt_runs) == 3
    asyncio.run(eng.advance(third, 10 ** 9))
    g3 = next(c for c in third.candidates if c.get("submitted") == 1)["tests"][0]
    want = [g2, [[1]]] if g3 == g2 else [[[1]], g2]  # two attempts agreeing outrank the first
    assert [g for g, _ in guesses(a.candidates, inputs)[0]] == want



def test_chk5_check2_only_when_late_or_uncertain():
    """chk5: check2 applies when the entry's tokens to its first submit exceed check2_late or an
    augmentation check (the submitted function under grid transforms) failed; otherwise the chk1 rule."""
    knobs = dict(submit_check=True, submit_check2=True, check2_late=40000, check2_max_refusals=10, check2_token_exit=None)
    # early, robust (transpose commutes with transposing the grids; recolouring keeps it): one function is enough
    run = TaskRun("t#1", TASK, {"conversation": True, **knobs, "aug_check": ["transpose", "color1"]})
    eng, sent = engine([call(GOOD + "submit(good)")], **knobs, aug_check=["transpose", "color1"])
    rows = []
    eng.record = rows.append
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    reply = [r for r in rows if r.get("event") == "reply"][-1]
    assert run.done == "submitted" and reply["aug_failed"] == [] and reply["check2_reason"] is None
    assert tasks.CONV_SUBMIT_CHECK2_LATE in sent[0]["messages"][0]["content"]
    assert tasks.CONV_SUBMIT_CHECK2 not in sent[0]["messages"][0]["content"]
    # early, not robust to rotation: the entry is uncertain and check2 applies
    run = TaskRun("t#1", TASK, {"conversation": True, **knobs, "aug_check": ["rot90"]})
    eng, sent = engine([call(GOOD + "submit(good)"), call(GOOD + ALT + "submit(good, alt)")], **knobs, aug_check=["rot90"])
    rows = []
    eng.record = rows.append
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    first, second = [r for r in rows if r.get("event") == "reply"][:2]
    assert first["aug_failed"] == ["rot90"] and first["check2_reason"] == "aug" and run.uncertain
    assert [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"][0] == "submit: " + tasks.CONV_SUBMIT_TWO
    assert second.get("check2_accepted") and run.done == "submitted" and "aug_failed" not in second  # checked once
    # late: past 40,000 tokens before the first submit
    run = TaskRun("t#1", TASK, {"conversation": True, **knobs})
    eng, sent = engine([call(GOOD + "submit(good)")], **knobs)
    run.spent = 40001
    asyncio.run(eng.advance(run, 10 ** 9 + 40001, replies=1))
    assert run.done is None and run.check2_refusals == 1
    assert conv.ConvEngine.check2_due(TaskRun("t#1", TASK, {}), {}, {})  # unset: always (chk2-chk4)


def test_hyp1_triage_close_and_hypothesis_reopen():
    """hyp1 (29 Oct): a first submit within triage_close tokens ends the entry; a later one reopens in the "hyp" mode
    (the naive head, its program and answer, then CONV_HYP)."""
    knobs = dict(submit_mode="fresh", fresh_each=True, fresh_text="hyp", keep_first=True, triage_close=30000)
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[7, 8, 9]])")], **knobs)
    rows = []
    eng.record = rows.append
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert a.done == "submitted" and not a.awaiting_reopen
    assert [r for r in rows if r.get("event") == "reply"][-1].get("triage_closed")
    _close(a)
    b = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[7, 8, 9]])"), call("submit([[1]])")], **knobs)
    b.spent = 30001
    asyncio.run(eng.advance(b, 10 ** 9 + 30001, replies=1))
    assert b.done is None and b.awaiting_reopen and b.parked
    assert eng.pick_reopen([b]) == 0
    first = b.messages[1]["content"]
    assert tasks.CONV_NAIVE in first and tasks.CONV_HYP in first and "7 8 9" in first
    _close(b)


def test_bounded_pass_submits_its_best_program_and_reopens_fresh():
    """pass1 (29 Oct): a fresh-mode pass reaching pass_budget tokens without a submit submits its best program output
    (verified over wrong) and waits for a fresh pass; the next pass has its own budget."""
    inputs = [t["input"] for t in TASK["test"]]
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([RIGHT, WRONG, WRONG, call("submit([[1]])")], submit_mode="fresh", fresh_each=True,
                       fresh_text="naive", keep_first=True, pass_budget=3000)
    rows = []
    eng.record = rows.append
    asyncio.run(eng.advance(a, 10 ** 9, replies=3))
    cut = [r for r in rows if r.get("event") == "pass_cut"]
    assert len(cut) == 1 and cut[0]["submitted"] and cut[0]["verified"] and cut[0]["pass_tokens"] == 3000
    assert a.parked and a.awaiting_reopen and a.submits == 1
    assert [g for g, _ in guesses(a.candidates, inputs)[0]][0] == [[7], [8], [9]]
    assert eng.pick_reopen([a]) == 0 and a.pass_start == 3000
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert a.submits == 2 and a.awaiting_reopen and len([r for r in rows if r.get("event") == "pass_cut"]) == 1
    _close(a)


TC = dict(submit_mode="fresh", fresh_each=True, fresh_text="naive", reopen_order="try_cap", context_mode="cap",
          try_cap=True, memo=True)


def test_try_cap_needs_two_different_guesses_then_a_new_first_guess_and_keeps_s1_g1_and_s2_g1():
    """try_cap: a submit needs two different guesses; it ends the try; the next try sees the
    earlier answer marked wrong and needs a new first guess (guess 2 is free, here S1's guess 1 again);
    the second accepted submit ends the task with guess 1 = S1 guess 1, guess 2 = S2 guess 1."""
    inputs = [t["input"] for t in TASK["test"]]
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[1]], [[1]])"), call("submit([[1]])"), call("submit([[1]], [[2]])"),
                        call("submit([[2]], [[3]])"), call("submit([[7], [8], [9]], [[1]])")], **TC)
    rows = []
    eng.record = rows.append
    assert "`second` must differ from `first`." in tasks.conv_system({**eng.settings})
    asyncio.run(eng.advance(a, 10 ** 9, replies=3))
    replies = lambda: [r for r in rows if r.get("event") == "reply"]  # noqa: E731
    assert [r.get("try_refused") for r in replies()] == ["same", "second_missing", None]
    assert a.submits == 1 and a.parked and a.awaiting_reopen and a.done is None
    assert eng.pick_reopen([a]) == 0 and a.try_n == 2
    first = a.messages[1]["content"]
    assert tasks.CONV_NAIVE in first and tasks.CONV_TRY_WRONG in first and "Your submitted second guess:" in first
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=2))
    assert replies()[3]["try_refused"] == "earlier" and a.done == "submitted" and a.submits == 2
    assert [g for g, _ in guesses(a.candidates, inputs)[0]] == [[[1]], [[7], [8], [9]]]
    ends = [r for r in rows if r.get("event") == "try_end"]
    assert [(e["cause"], e["try"], e["accepted"]) for e in ends] == [("accepted", 1, 1), ("accepted", 2, 2)]
    _close(a)


def test_try_cap_a_full_context_ends_the_try_and_a_task_without_a_submit_restarts_blank():
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([WRONG] * 40, window=4000, try_min_reply=1024, **TC)
    rows = []
    eng.record = rows.append
    asyncio.run(eng.advance(a, 10 ** 9))
    assert a.parked and a.awaiting_reopen and a.done is None and a.compactions == 0
    assert all(s["max_tokens"] <= 4000 - 256 for s in sent)
    end = [r for r in rows if r.get("event") == "try_end"]
    assert len(end) == 1 and end[0]["cause"] == "cap" and end[0]["accepted"] == 0 and a.try_tokens == [end[0]["tokens"]]
    eng.pick_reopen([a])
    assert a.messages == a.head and tasks.CONV_NAIVE not in a.messages[1]["content"] and a.try_n == 2
    _close(a)


def test_try_cap_reopen_order_no_accepted_first_then_mean_tokens_to_submit():
    eng, _ = engine([], **TC)
    runs = [TaskRun(f"t{k}#1", TASK, {"conversation": True}) for k in range(4)]
    for r, (subs, tries, toks) in zip(runs, [(1, 1, [9000]), (1, 2, [30000, 2000]), (0, 3, [5, 5]), (0, 1, [80000])]):
        r.messages, r.submits, r.try_n, r.try_tokens, r.awaiting_reopen = [], subs, tries, toks, True
        r.candidates = []
    eng.refresh = lambda run, settings: None
    order = [eng.pick_reopen(runs) for _ in range(4)]
    assert order == [3, 2, 1, 0]


def test_try_cap_later_submit_takes_a_lone_new_first_guess():
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call("submit([[1]], [[2]])"), call("submit([[3]])")], **TC)
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    eng.pick_reopen([a])
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert a.done == "submitted" and a.submits == 2
    assert [g for g, _ in guesses(a.candidates, [t["input"] for t in TASK["test"]])[0]] == [[[1]], [[3]]]
    _close(a)


def test_try_cap_a_prompt_that_fills_the_window_ends_the_task_instead_of_cycling():
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([WRONG], window=1200, try_min_reply=1024, **TC)
    asyncio.run(eng.advance(a, 10 ** 9))
    assert a.done == "context_full" and not a.awaiting_reopen and not sent and a.interp is None
    _close(a)


def test_agree_close_ends_a_task_when_a_pass_repeats_an_earlier_first_guess():
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call("submit([[1]], [[2]])"), call("submit([[3]])"), call("submit([[1]], [[5]])")],
                    submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, agree_close=True)
    for k in range(3):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        if a.awaiting_reopen:
            eng.pick_reopen([a])
    assert a.done == "agreed" and a.submits == 3
    _close(a)


FLIP = "def flip(g):\n    return [list(c) for c in zip(*g)]\n"


def _first_text(messages):
    first = messages[1]["content"]
    return first if isinstance(first, str) else first[0]["text"]


def test_alt_fit_pass_asks_for_a_second_rule_once_at_the_first_train_fit():
    """altP: the tool result after the pass's first train-fitting function carries the ask, once per pass."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("x = 1"), call(FLIP), call("print(2)"), call("submit([[7], [8], [9]])"), call("y = 2")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, alt_fit="pass")
    for _ in range(5):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        if a.awaiting_reopen:
            eng.pick_reopen([a])
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    texts = [m["content"] for s in sent for m in s["messages"] if m["role"] == "tool"]
    assert sum("[A second rule] `flip`" in t for t in set(texts)) == 1
    assert not any("[A second rule]" in t for t in tools)  # the next pass has not run a tool call yet
    _close(a)


def test_alt_fit_reopen_shows_the_fitting_function_and_asks_for_a_different_rule():
    """altR: after a submit with a train-fitting function in the interpreter, the next pass's head shows its source
    and the ask; without a fit the plain naive head."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call(FLIP), call("submit([[7], [8], [9]])"), call("submit([[1], [2], [3]])")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, alt_fit="reopen")
    for _ in range(3):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        if a.awaiting_reopen:
            eng.pick_reopen([a])
    head = _first_text(sent[2]["messages"])
    assert "A function that reproduces every training pair" in head and "def flip(g)" in head
    _close(a)
    c = TaskRun("t#1", TASK, {"conversation": True})  # defined in the submit cell itself (not in the replay history)
    eng, sent = engine([call(FLIP + "submit([[7], [8], [9]])"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, alt_fit="reopen")
    for _ in range(2):
        c.parked = False
        asyncio.run(eng.advance(c, 10 ** 9, replies=1))
        if c.awaiting_reopen:
            eng.pick_reopen([c])
    assert "def flip(g)" in _first_text(sent[1]["messages"])
    a = c
    assert tasks.CONV_ALT_REOPEN.strip()[:40] in head
    _close(a)
    b = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[7], [8], [9]])"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, alt_fit="reopen")
    for _ in range(2):
        b.parked = False
        asyncio.run(eng.advance(b, 10 ** 9, replies=1))
        if b.awaiting_reopen:
            eng.pick_reopen([b])
    assert "[A second rule]" not in _first_text(sent[1]["messages"])
    _close(b)


def test_resolve_effort_lowers_effort_only_after_the_first_submit():
    """lean: pass 1's requests keep the arm's effort; every request after the first submit uses resolve_effort."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("x = 1"), call("submit([[7], [8], [9]])"), call("y = 2"), call("submit([[1], [2], [3]])")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, effort="xhigh",
                       resolve_effort="medium")
    for _ in range(4):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        if a.awaiting_reopen:
            eng.pick_reopen([a])
    assert [s["settings"]["effort"] for s in sent] == ["xhigh", "xhigh", "medium", "medium"]
    _close(a)


def test_resolve_effort_under_keeps_full_effort_after_a_late_first_submit():
    """leanS: re-solves think less only when the first submit came under resolve_effort_under tokens."""
    for under, want in ((10 ** 6, "medium"), (50, "xhigh")):
        a = TaskRun("t#1", TASK, {"conversation": True})
        eng, sent = engine([call("x = 1"), call("submit([[7], [8], [9]])"), call("y = 2")], submit_mode="fresh",
                           fresh_each=True, fresh_text="naive", keep_first=True, effort="xhigh",
                           resolve_effort="medium", resolve_effort_under=under)
        for _ in range(3):
            a.parked = False
            asyncio.run(eng.advance(a, 10 ** 9, replies=1))
            if a.awaiting_reopen:
                eng.pick_reopen([a])
        assert [s["settings"]["effort"] for s in sent] == ["xhigh", "xhigh", want]
        _close(a)


def test_resolve_cap_sets_the_reply_cap_of_requests_after_the_first_submit():
    """cap1: pass 1's replies get the arm's cap, every request after the first submit resolve_cap."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("x = 1"), call("submit([[7], [8], [9]])"), call("y = 2")], submit_mode="fresh",
                       fresh_each=True, fresh_text="naive", keep_first=True, cap=12000, resolve_cap=8000)
    for _ in range(3):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        if a.awaiting_reopen:
            eng.pick_reopen([a])
    assert [s["max_tokens"] for s in sent] == [12000, 12000, 8000]
    _close(a)


def test_resolve_temperature_applies_only_after_the_first_submit():
    """t10r: pass 1 keeps the study's temperature (none set here), re-solves get resolve_temperature."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("x = 1"), call("submit([[7], [8], [9]])"), call("y = 2")], submit_mode="fresh",
                       fresh_each=True, fresh_text="naive", keep_first=True, resolve_temperature=1.0)
    for _ in range(3):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        if a.awaiting_reopen:
            eng.pick_reopen([a])
    assert [s["settings"].get("temperature") for s in sent] == [None, None, 1.0]
    _close(a)


def test_repair_fresh_opens_pass_two_with_the_fitting_function_and_the_novelty_report():
    """rep1: after a first submit with a train-fitting function, pass 2's head is the repair text, the function and the
    novelty notes; pass 3 is the plain naive head again."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call(FLIP), call("submit([[7], [8], [9]])"), call("submit([[1], [2], [3]])"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, repair="fresh")
    for _ in range(4):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        if a.awaiting_reopen:
            eng.pick_reopen([a])
    heads = [str(s["messages"][1]["content"]) for s in sent]
    assert "[Check the test input]" in heads[2] and "def flip(" in heads[2] and "Compared with the training inputs" in heads[2]
    assert "[Check the test input]" not in heads[3] and "[Your earlier attempt]" in heads[3]
    _close(a)


def test_repair_inpass_keeps_pass_one_going_after_its_first_submit():
    """rep2: the first submit returns the edge-case check and the pass goes on; the second submit ends it."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("submit([[7], [8], [9]])"), call("submit([[1], [2], [3]])"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, repair="inpass")
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert not a.awaiting_reopen and a.submits == 1
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert any("provisional answer" in t for t in tools)
    assert a.awaiting_reopen  # the second submit ended pass 1
    firsts = [c for c in a.candidates if c.get("submitted") in (1, 1.5)]
    assert {json.dumps(c["tests"]) for c in firsts} >= {json.dumps([[[7], [8], [9]]]), json.dumps([[[1], [2], [3]]])}
    _close(a)


def test_fresh2_vote_pick_ranks_the_most_submitted_answer_then_a_different_one_and_each_try_gets_a_new_session():
    """fresh2: blank passes in a new interpreter; final guesses = most-submitted first guess (ties: latest), then the
    most-submitted different one."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    A, B = "[[7], [8], [9]]", "[[1], [2], [3]]"
    eng, sent = engine([call("keep = 1\nsubmit(" + A + ")"), call("print('kept' if 'keep' in globals() else 'fresh')\nsubmit(" + B + ")"),
                        call("submit(" + A + ")"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="blank", fresh_interp=True, pick="vote")
    first = None
    for i in range(3):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        if i == 0:
            first = a.interp
        if a.awaiting_reopen:
            eng.pick_reopen([a])
        if i == 1:  # A, B: a tie goes to the latest (B), A second
            assert [g for g, _ in guesses(a.candidates, [TASK["test"][0]["input"]])[0]] == [[[1], [2], [3]], [[7], [8], [9]]]
    assert a.interp is not first
    assert a.interp.run("print('keep' in globals())")["stdout"].strip() == "False"  # no names carried over
    assert [g for g, _ in guesses(a.candidates, [TASK["test"][0]["input"]])[0]] == [[[7], [8], [9]], [[1], [2], [3]]]
    heads = [str(s["messages"][1]["content"]) for s in sent]
    assert heads[1] == heads[0] and heads[2] == heads[0]  # blank: nothing of the earlier tries is shown
    _close(a)


def test_ind20_one_naive_reopen_then_independent_tries_with_the_vote_pick():
    """ind20: pass 2 is the naive reopen (program + answer shown, S1 pinned); pass 3 is blank in a new session and its
    submit is the vote; the fit probe marks the entry at the submit."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    A, B, C = "[[7], [8], [9]]", "[[1], [2], [3]]", "[[4], [5], [6]]"
    eng, sent = engine([call(FLIP + "submit(" + A + ")"), call("submit(" + B + ")"), call("submit(" + C + ")"),
                        call("submit(" + C + ")"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, indep_after=1,
                       reopen_order="indep")
    inp = [TASK["test"][0]["input"]]
    sessions = []
    for i in range(4):
        a.parked = False
        asyncio.run(eng.advance(a, 10 ** 9, replies=1))
        sessions.append(a.interp)
        if i == 0:
            assert a.fit_seen
        if i == 1:  # the naive reopen: latest + pinned S1, same session
            assert [g for g, _ in guesses(a.candidates, inp)[0]] == [[[1], [2], [3]], [[7], [8], [9]]]
        if i == 2:  # three different first guesses: ties to the latest, S1 drops out (keep_first off)
            assert [g for g, _ in guesses(a.candidates, inp)[0]] == [[[4], [5], [6]], [[1], [2], [3]]]
        if a.awaiting_reopen:
            eng.pick_reopen([a])
    assert [g for g, _ in guesses(a.candidates, inp)[0]][0] == [[4], [5], [6]]  # C twice
    assert sessions[1] is sessions[0] and sessions[2] is not sessions[1] and sessions[3] is not sessions[2]
    heads = [str(s["messages"][1]["content"]) for s in sent]
    assert "Your submitted first guess" in heads[1] and heads[2] == heads[0] and heads[3] == heads[0]
    _close(a)


def test_ind20_order_naive_reopens_first_then_fit_open_then_no_fit_then_repeated_answer():
    """ind20: every entry's naive reopen first; then a class's next try waits for the classes before it."""
    eng, _ = engine([], submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, indep_after=1,
                    reopen_order="indep")
    def run(name, count, fit, firsts, tokens=1000):
        r = TaskRun(name, TASK, {"conversation": True})
        r.awaiting_reopen, r.reopen_count, r.fit_seen, r.first_submit_tokens, r.submits = True, count, fit, tokens, len(firsts)
        r.candidates = [{"code": f"submit-{i + 1}-1", "tests": [g]} for i, g in enumerate(firsts)]
        eng.refresh = lambda *a, **k: None
        return r
    x, y = [[1]], [[2]]
    naive, late = run("a#1", 0, False, [x], 500), run("b#1", 0, True, [x], 900)
    fit2 = run("c#1", 2, True, [x, y, [[3]]])  # second independent try: level 2
    nofit = run("d#1", 1, False, [x, y])  # first independent try: level 2
    fit = run("e#1", 1, True, [x, y])  # level 1
    same = run("f#1", 1, True, [x, x])  # level 3
    runs = [naive, late, fit2, nofit, fit, same]
    picked = [runs[eng.pick_reopen(runs)].id[0] for _ in range(6)]
    assert picked[:3] == ["b", "a", "e"] and set(picked[3:5]) == {"c", "d"} and picked[5] == "f"
    assert eng.try_class(same) == 2 and eng.try_class(nofit) == 1 and eng.try_class(fit) == 0


def test_sweep_reports_train_fitting_variants_with_different_test_outputs():
    """Item 2: a function with a free constant: variants that still fit come back with their test outputs."""
    task = {"train": [{"input": [[1, 0]], "output": [[1, 0]]}, {"input": [[0, 1]], "output": [[0, 1]]}],
            "test": [{"input": [[5, 1]]}]}
    interp = Repl(task, strict=False, call_seconds=5)
    code = "def f(g):\n    limit = 3\n    return [[v if v < limit else 2 for v in r] for r in g]\n"
    interp.run(code)
    got = interp.sweep("f", code, 5.0)
    assert got["tried"] > 3 and [[[3, 1]]] in got["tests"] and [[[0, 1]]] in got["tests"]
    interp.close()


def test_notes1_try_ends_with_notes_and_the_next_try_opens_with_the_playbook_not_the_program():
    """notes1: a submit without a playbook rewrite asks for the notes; the next submit ends the try; the new try (new
    session) is shown the playbook, no program."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call(FLIP + "submit([[7], [8], [9]])"), call("playbook('rule: transpose (checked)')\nsubmit([[7], [8], [9]])"),
                        call("print(playbook())"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", memo=True, try_notes=True)
    eng.repl = lambda run: Repl(run.task, strict=False, call_seconds=5, memo=True)
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert not a.awaiting_reopen and a.note_asked
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert a.awaiting_reopen
    old = a.interp
    eng.pick_reopen([a])
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    head = str(sent[2]["messages"][1]["content"])
    assert "[A new try]" in head and "rule: transpose (checked)" in head and "def flip(" not in head
    assert a.interp is not old
    _close(a)


ODD = "def odd(g):\n    t = [list(c) for c in zip(*g)]\n    return [[0]] * 3 if g == [[7, 8, 9]] else t\n"


def test_vcheck_holds_the_pass_when_the_submitting_function_breaks_a_kept_symmetry():
    """vcheck: a first submit whose train-fitting function breaks a training symmetry on the test input gets the check
    text (transform and cells) and the pass goes on; a symmetric function's first submit ends the pass."""
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call(ODD + "submit(odd)"), call("submit([[7], [8], [9]])"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, vcheck=True)
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    assert not a.awaiting_reopen and a.submits == 1
    asyncio.run(eng.advance(a, 10 ** 9, replies=1))
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert any("rotation" in t and "(0, 0), (1, 0), (2, 0)" in t and "`odd`" in t for t in tools)
    assert a.awaiting_reopen
    _close(a)
    b = TaskRun("t#2", TASK, {"conversation": True})
    eng, sent = engine([call(FLIP + "submit(flip)"), call("x = 1")],
                       submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, vcheck=True)
    b.parked = False
    asyncio.run(eng.advance(b, 10 ** 9, replies=1))
    assert b.awaiting_reopen
    _close(b)


def test_tool_fix_imports_session_names_and_explains_an_empty_call():
    r = Repl(TASK, strict=False, call_seconds=5, import_globals=True)
    out = r.run("import train, test_inputs, np\nprint(len(train), np.zeros(2).shape)")
    assert out["stdout"].strip() == "2 (2,)" and not out.get("error")
    r.close()
    a = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([{"name": "python", "arguments": "{}"}, call("x = 1")], tool_fix=True)
    a.parked = False
    asyncio.run(eng.advance(a, 10 ** 9, replies=2))
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert any("arrived with no code" in t for t in tools)
    _close(a)


TWO = dict(submit_mode="fresh", fresh_each=True, fresh_text="blank", fresh_interp=True, pass_clean=True,
           pick="vote_all", submit_two=True, check2_late=20000, score_refused=True)
R1, R2, R3 = "[[7], [8], [9]]", "[[1], [2], [3]]", "[[4], [4], [4]]"


def _picks(run):
    return [g for g, _ in guesses(run.candidates, [TASK["test"][0]["input"]])[0]]


def _rows(eng):
    rows = []
    eng.record = rows.append
    return rows


def test_base13b_early_entry_needs_two_different_guesses_no_train_check_and_repeats_are_allowed():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call(f"submit({R1})"), call(f"submit({R1}, {R1})"), call(f"submit({R1}, {R2})"),
                        call(f"submit({R1}, {R2})")], **TWO)
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    assert run.submits == 0 and run.two_refusals == 2
    assert "`second` must differ from `first`." in sent[0]["messages"][0]["content"]
    assert tasks.CONV_SUBMIT_EVERY in sent[0]["messages"][0]["content"]
    assert "first guess must always be the one you are more confident in" in sent[0]["messages"][0]["content"]
    texts = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert texts[0] == "submit: " + tasks.CONV_SUBMIT_TWO_GUESSES
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    texts = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert texts[-1] == "submit: " + tasks.CONV_SUBMIT_SAME_GUESSES
    assert run.submits == 1 and run.awaiting_reopen
    run.parked = False
    eng.pick_reopen([run])
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 2  # item 4: a repeat of an earlier pass is accepted
    _close(run)


def test_base13b_late_entry_needs_two_fitting_functions_on_every_submit_with_no_exit():
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent = 20001
    bad = [call(GOOD + f"submit(good, {R2})"), call(GOOD + "submit(good)"), call(f"submit({R1}, {R2})")] * 5
    eng, sent = engine(bad + [call(GOOD + ALT + "submit(good, alt)"), call(GOOD + "submit(good, [[1], [2], [3]])"),
                              call(GOOD + ALT + "submit(good, alt)")], **TWO)
    asyncio.run(eng.advance(run, 10 ** 9, replies=15))
    assert run.submits == 0 and run.two_refusals == 15  # past chk's 10: still refused, no exit, no chk1 fallback
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 1 and run.first_submit_tokens > 20000
    run.parked = False
    eng.pick_reopen([run])
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 1 and run.two_refusals == 16  # a reopen: a bare second guess is refused as well
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 2
    _close(run)


def test_base13b_entry_past_20k_counted_to_now_before_its_first_submit():
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent = 19950
    eng, sent = engine([call(f"submit({R1})"), call(f"submit({R1}, {R2})")], **TWO)
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))  # 19,950 + 100 + 100: refused once under 20K, then past it
    assert run.submits == 0 and run.two_refusals == 2
    _close(run)


def _vote(script):
    run = TaskRun("t#1", TASK, {"conversation": True})
    n = len(script)
    eng, _ = engine(script, **TWO)
    for _i in range(n):
        run.parked = False
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        eng.pick_reopen([run])
    picks = _picks(run)
    _close(run)
    return picks


def test_base13b_vote_top_two_by_passes_ties_first_guess_then_oldest():
    A, B, C, D = R1, R2, R3, "[[5], [5], [5]]"
    a, b, c, d = (json.loads(x) for x in (A, B, C, D))
    # B in 3 passes, D in 2, C in 2 as first guess: S1's first guess A (1 pass) is not pinned
    assert _vote([call(f"submit({A}, {B})"), call(f"submit({C}, {B})"), call(f"submit({D}, {B})"), call(f"submit({C}, {D})")]) == [b, c]
    # one pass each: first guesses A, C over seconds B, D; A older
    assert _vote([call(f"submit({A}, {B})"), call(f"submit({C}, {D})")]) == [a, c]
    # 2 passes, both first guesses match, seconds differ: A, then the oldest second
    assert _vote([call(f"submit({A}, {B})"), call(f"submit({A}, {C})")]) == [a, b]
    # a grid counts once in a pass and over both guess places
    assert _vote([call(f"submit({A}, {B})"), call(f"submit({B}, {A})")]) == [a, b]
    # A and B in 2 passes each, A never a first guess
    assert _vote([call(f"submit({C}, {A})"), call(f"submit({B}, {A})"), call(f"submit({B}, {D})")]) == [b, a]


def test_base13b_vote_is_per_output():
    run = TaskRun("t#1", TASK2, {"conversation": True})
    x, y = "[[7], [8], [9]]", "[[1], [2]]"
    p, q = "[[1], [1], [1]]", "[[2], [2]]"
    eng, _ = engine([call(f"submit([{x}, {y}], [{x}, {q}])"), call(f"submit([{p}, {y}], [{x}, {q}])")], **TWO)
    for _i in range(2):
        run.parked = False
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        eng.pick_reopen([run])
    picks = guesses(run.candidates, [t["input"] for t in TASK2["test"]])
    assert [g for g, _ in picks[0]] == [json.loads(x), json.loads(p)]
    assert [g for g, _ in picks[1]] == [json.loads(y), json.loads(q)]
    _close(run)


def test_base13b_reopen_is_a_plain_repeat_of_pass_one():
    run = TaskRun("t#1", TASK, {"conversation": True})
    good = "```python\ndef transform(grid):\n    return [list(r) for r in zip(*grid)]\n```"
    eng, sent = engine([good, call("keep = 1\nsubmit([[7], [8], [9]], [[1], [2], [3]])"),
                        good, call("print(sorted(k for k in globals() if k.startswith('program') or k == 'keep'))"),
                        call("print(1)")], **TWO)
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    assert run.submits == 1 and run.awaiting_reopen
    first = sent[0]["messages"]
    run.parked = False
    eng.pick_reopen([run])
    assert run.messages == list(first)  # the head is pass 1's, byte for byte
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    tool = [m["content"] for m in sent[-1]["messages"] if m["role"] in ("tool", "user")]
    assert any(t.startswith("Program 1 passes") or "Program 1 passes" in t for t in map(str, tool))
    ledger = next(t for t in map(str, tool) if "Program 1 passes" in t)
    assert "1 distinct test output:" in ledger and "program 1" in ledger and "Its test output is a new reading" in ledger
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    tool = [str(m["content"]) for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert "['program_1']" in tool[-1]  # no `keep`, no earlier program
    assert not (run.interp.memo or {}).get("text")
    assert run.programs and len(run.programs) == 2  # the record keeps both
    _close(run)


def test_base13b_verification_stays_per_entry_on_a_blank_reopen():
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent = 20001
    eng, _ = engine([call(GOOD + ALT + "submit(good, alt)"), call(f"submit({R1}, {R2})"), call(GOOD + ALT + "submit(good, alt)")], **TWO)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    run.parked = False
    eng.pick_reopen([run])
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 1 and run.two_refusals == 1
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 2
    _close(run)


def test_base13b_never_accepted_entry_scores_its_last_refused_pair_and_an_accepted_submit_replaces_it():
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent = 20001
    eng, sent = engine([call(f"submit({R3}, {R1})"), call(f"submit({R1}, {R2})"), call(GOOD + ALT + "submit(good, alt)")], **TWO)
    asyncio.run(eng.advance(run, 10 ** 9, replies=2))
    assert run.submits == 0 and _picks(run) == [json.loads(R1), json.loads(R2)]
    assert not any("refused pair" in str(m.get("content")) for m in sent[-1]["messages"])
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 1 and not any(c.get("refused") for c in run.candidates)
    _close(run)
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent = 20001
    eng, _ = engine([call(f"submit({R1}, {R2})")], **{**TWO, "score_refused": False})
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert not any(c.get("refused") for c in run.candidates)
    _close(run)


def _turn(eng, run):
    while not getattr(run, "parked", False) and run.done is None:  # the worker's loop: one reply at a time until the task leaves its slot
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))


CAP = dict(TWO, context_mode="cap", drain=None, reopen_order="never_first")


def test_base13b_a_refusal_loop_ends_only_at_the_context_cap_and_comes_back_as_a_fresh_pass():
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent = 20001
    eng, sent = engine([call(f"submit({R1}, {R2})")] * 600, window=8192, **CAP)
    rows = _rows(eng)
    _turn(eng, run)
    ends = [r for r in rows if r.get("event") == "try_end"]
    assert run.awaiting_reopen and run.done is None and run.submits == 0 and [e["cause"] for e in ends] == ["cap"]
    assert run.two_refusals == len(sent) and len(sent) > 12  # past chk's 10: nothing but the cap ended it
    assert ends[0]["context"] + 1024 + 256 > 8192  # the room left was under the smallest reply
    grown = len(sent[-1]["messages"])
    run.parked = False
    eng.pick_reopen([run])
    assert len(run.messages) == 2 and run.reopen_count == 1  # fresh: head only, a new interpreter
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert len(sent[-1]["messages"]) == 2 < grown
    assert _picks(run) == [json.loads(R1), json.loads(R2)]  # the last refused pair stands for the entry
    _close(run)


def test_base13b_a_reply_near_the_cap_cannot_overrun_the_window():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("print(1)")] * 600, window=8192, **CAP)
    rows = _rows(eng)
    _turn(eng, run)
    counts = [sum(conv.approx_tokens(m) for m in s["messages"]) for s in sent]
    assert all(c + s["max_tokens"] + 256 <= 8192 for c, s in zip(counts, sent))
    assert min(s["max_tokens"] for s in sent) >= 1024
    assert not any(r.get("event") in ("drain", "context_drain", "compaction") for r in rows) and not run.compactions
    _close(run)


def test_base13b_entries_with_no_accepted_submit_go_first_in_the_reopen_queue():
    a, b, c = (TaskRun(f"t#{i}", TASK, {"conversation": True}) for i in (1, 2, 3))
    eng, _ = engine([], **CAP)
    a.submits, a.first_submit_tokens, a.reopen_count = 1, 90000, 0
    b.submits, b.first_submit_tokens, b.reopen_count = 0, None, 3
    c.submits, c.first_submit_tokens, c.reopen_count = 1, 30000, 0
    for r in (a, b, c):
        r.awaiting_reopen, r.candidates = True, []
        r.messages, r.head, r.programs, r.interp = [], [], [], None
    eng.settings = {**eng.settings, "fresh_interp": False}
    assert eng.pick_reopen([a, b, c]) == 1
    assert eng.pick_reopen([a, b, c]) == 0
    assert eng.pick_reopen([a, b, c]) == 2


def test_base13b_nothing_says_verification_until_a_late_entry_is_refused():
    def told(sent):
        return [str(m.get("content")) for s in sent for m in s["messages"]]

    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call(f"submit({R1})"), call(f"submit({R1}, {R1})"), call(f"submit({R1}, {R2})"), call("x = 1")], **TWO)
    asyncio.run(eng.advance(run, 10 ** 9, replies=3))
    run.parked = False
    eng.pick_reopen([run])
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    text = " ".join(told(sent)).lower()
    assert run.submits == 1 and "verif" not in text and "20,000" not in text and "two functions" not in text
    assert "playbook" not in text and "dropped" not in text
    assert "verif" not in str(tasks.tool_spec({"conversation": True, "submit_two": True})).lower()
    assert "second=None" not in str(tasks.tool_spec({"conversation": True, "submit_two": True}))
    _close(run)
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent = 20001
    eng, sent = engine([call(f"submit({R1}, {R2})"), call(GOOD + "submit(good, lambda g: g)"), call(GOOD + "submit(good, good)"),
                        call(GOOD + ALT + "submit(good, alt)"), call("x = 1")], **TWO)
    asyncio.run(eng.advance(run, 10 ** 9, replies=4))
    head = str(sent[0]["messages"][0]["content"]) + str(sent[0]["messages"][1]["content"])
    assert "verif" not in head.lower() and "two functions" not in head
    tool = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert tool[0] == "submit: " + tasks.CONV_SUBMIT_VERIFY
    assert tool[1].startswith("submit: " + tasks.CONV_SUBMIT_VERIFY + "\n\nYour second function does not reproduce")
    assert tool[2].startswith("submit: " + tasks.CONV_SUBMIT_VERIFY + "\n\nBoth functions reproduce every training pair")
    assert run.submits == 1
    run.parked = False
    eng.pick_reopen([run])
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert "verif" not in " ".join(str(m["content"]) for m in sent[-1]["messages"][:2]).lower()  # a fresh pass: nothing up front
    _close(run)


B14 = {k: v for k, v in CAP.items() if k not in ("check2_late", "score_refused")}
B14.update(no_first_line=True, agree_close=True, loop_exit=20, loop_rule=True)
TWO_OUT = {**TASK, "test": [TASK["test"][0], TASK["test"][0]]}


def _pass(eng, run):
    run.parked = False
    eng.pick_reopen([run])
    _turn(eng, run)


def test_base14_bare_grids_pass_past_20k_nothing_refused_and_no_first_line():
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.spent = 50000
    eng, sent = engine([call(f"submit({R1}, {R2})")], **B14)
    _turn(eng, run)
    assert run.submits == 1 and not getattr(run, "two_refusals", 0) and run.awaiting_reopen and run.done is None
    system = sent[0]["messages"][0]["content"]
    assert "more confident" not in system and "Every submit needs two guesses" in system
    assert "Each guess is grids or a function transform(grid)." in system
    assert not any(c.get("refused") for c in run.candidates)
    _close(run)


def test_base14_keeps_base13b_prompt_and_refusals():
    assert tasks.CONV_SUBMIT_FIRST_LINE in tasks.CONV_SUBMIT_EVERY
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call(f"submit({R1})"), call(f"submit({R1}, {R1})"), call(f"submit({R1}, {R2})")], **B14)
    _turn(eng, run)
    assert run.two_refusals == 2 and run.submits == 1  # still two different guesses
    _close(run)


def test_base14_every_entry_gets_pass_2_however_early_its_first_submit():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call(f"submit({R1}, {R2})"), call(f"submit({R2}, {R1})")], **B14)
    _turn(eng, run)
    assert run.submits == 1 and run.spent < 20000 and run.awaiting_reopen and run.done is None
    _pass(eng, run)
    assert run.submits == 2 and run.awaiting_reopen and run.done is None  # first guesses differ: pass 3
    _close(run)


def test_base14_an_entry_stops_when_pass_2_repeats_all_first_guesses():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call(f"submit({R1}, {R2})"), call(f"submit({R1}, {R3})")], **B14)
    _turn(eng, run)
    assert run.awaiting_reopen and run.done is None
    _pass(eng, run)
    assert run.submits == 2 and run.done == "agreed" and not run.awaiting_reopen


def test_base14_pass_3_only_where_a_first_guess_differs_on_one_output_and_it_stops_on_any_repeat():
    both = lambda a, b: f"[{a}, {b}]"  # noqa: E731
    run = TaskRun("t#1", TWO_OUT, {"conversation": True})
    p1, p2 = both(R1, R1), both(R1, R2)  # same on the first output, different on the second
    eng, _ = engine([call(f"submit({p1}, {both(R3, R3)})"), call(f"submit({p2}, {both(R3, R3)})"),
                     call(f"submit({p1}, {both(R3, R2)})")], **B14)
    _turn(eng, run)
    _pass(eng, run)
    assert run.submits == 2 and run.awaiting_reopen and run.done is None  # one output differs: pass 3
    _pass(eng, run)
    assert run.submits == 3 and run.done == "agreed"  # pass 3 repeats pass 1, not pass 2


def test_base14_a_capped_entry_with_no_submit_comes_back():
    run = TaskRun("t#1", TASK, {"conversation": True})
    script = [call("print(1)")] * 600
    eng, _ = engine(script, window=8192, **{**B14, "loop_rule": False})
    rows = _rows(eng)
    _turn(eng, run)
    assert [r["cause"] for r in rows if r.get("event") == "try_end"] == ["cap"]
    assert run.awaiting_reopen and run.done is None and run.submits == 0
    script[:] = [call(f"submit({R1}, {R2})")]
    _pass(eng, run)
    assert run.submits == 1 and run.awaiting_reopen and run.done is None
    _close(run)


def test_base14_twenty_bad_calls_end_the_pass_and_it_comes_back_fresh():
    run = TaskRun("t#1", TASK, {"conversation": True})
    bad = {"name": "python", "arguments": json.dumps({"code": ""})}
    eng, sent = engine([bad] * 19 + [call("x = 1")] + [bad] * 20 + [call(f"submit({R1}, {R2})")],
                       **{**B14, "loop_rule": False})
    rows = _rows(eng)
    _turn(eng, run)
    ends = [r for r in rows if r.get("event") == "try_end"]
    assert [e["cause"] for e in ends] == ["bad_calls"] and len(sent) == 40  # the streak restarted at the good call
    assert run.awaiting_reopen and run.done is None and run.bad_calls == 0
    _pass(eng, run)
    assert len(sent[-1]["messages"]) == 2 and run.submits == 1
    _close(run)


def test_base14_a_loop_rule_fire_ends_the_pass_and_a_new_pass_can_fire_again():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("print(1)")] * 45, **B14)
    rows = _rows(eng)
    _turn(eng, run)
    assert [r["cause"] for r in rows if r.get("event") == "try_end"] == ["loop"] and len(sent) == 20
    assert not getattr(run, "loop_fire", None)
    _pass(eng, run)
    assert [r["cause"] for r in rows if r.get("event") == "try_end"] == ["loop", "loop"] and len(sent) == 40
    _close(run)


def test_base14_loop_exit_off_leaves_a_fire_to_the_drain():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("print(1)")] * 25, **{**B14, "loop_exit": None})
    asyncio.run(eng.advance(run, 10 ** 9, replies=25))
    assert len(sent) == 25 and run.done is None and not run.parked  # base13b's behaviour, kept for its replays
    _close(run)


B15 = dict(submit_mode="reopen", keep_first=True, reopen_order="first_submit_asc", context_mode="drain", drain={"turn_cap": 0},
           reopen_wrong=True, second_differs=True, memo=True)


def _reopen(eng, run):
    run.parked = False
    eng.pick_reopen([run])
    _turn(eng, run)


def test_base15_one_guess_parks_the_entry_and_it_keeps_its_context():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call("x = 1"), call(f"submit({R1})"), call(f"submit({R2})")], **B15)
    _turn(eng, run)
    assert run.submits == 1 and run.parked and run.awaiting_reopen and run.done is None
    held = len(run.messages)
    assert "Recorded as your answer" in run.messages[-1]["content"]
    run.parked = False
    eng.pick_reopen([run])
    assert len(run.messages) == held + 1 and run.messages[-1]["content"] == tasks.CONV_REOPEN_WRONG
    assert "x = 1" in json.dumps(run.messages)  # nothing dropped: the same conversation
    _turn(eng, run)
    assert len(sent[-1]["messages"]) == held + 1 and run.submits == 2 and run.done == "submitted"
    _close(run)


def test_base15_second_submit_must_differ_on_some_output_and_both_first_guesses_score():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = engine([call(f"submit({R1})"), call(f"submit({R1})"), call(f"submit({R2})")], **B15)
    _turn(eng, run)
    _reopen(eng, run)
    assert run.submits == 2 and run.repeats_refused == 1
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert tools[-1].startswith("submit: nothing was submitted") and run.done == "submitted"
    assert sorted(map(str, _picks(run))) == sorted(map(str, [json.loads(R1), json.loads(R2)]))
    _close(run)


def test_base15_one_differing_output_of_two_is_a_different_answer():
    both = lambda a, b: f"[{a}, {b}]"  # noqa: E731
    run = TaskRun("t#1", TWO_OUT, {"conversation": True})
    eng, _ = engine([call(f"submit({both(R1, R1)})"), call(f"submit({both(R1, R2)})")], **B15)
    _turn(eng, run)
    _reopen(eng, run)
    assert run.submits == 2 and not getattr(run, "repeats_refused", 0)
    _close(run)


def test_base15_a_right_first_guess_stays_when_the_second_submit_carries_two_guesses():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call(f"submit({R1})"), call(f"submit({R2}, {R3})")], **B15)
    _turn(eng, run)
    _reopen(eng, run)
    assert json.loads(R1) in _picks(run) and len(_picks(run)) == 2
    _close(run)


def test_base15_a_drain_never_parks_and_an_entry_with_no_submit_stays_in():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = engine([call(f"print({i}, 'x' * 400)") for i in range(120)], window=8192, **{**B15, "loop_rule": False})
    rows = _rows(eng)
    asyncio.run(eng.advance(run, 10 ** 9, replies=120))
    assert any(r.get("event") == "context_drain" for r in rows)
    assert not run.parked and not getattr(run, "awaiting_reopen", False) and run.done is None and run.submits == 0
    _close(run)


def test_base15_reopen_order_is_shortest_first_submit_first():
    a, b, c = (TaskRun(f"t#{i}", TASK, {"conversation": True}) for i in (1, 2, 3))
    eng, _ = engine([], **B15)
    for r, t in ((a, 30000), (b, 90000), (c, 50000)):
        r.submits, r.first_submit_tokens, r.reopen_count, r.awaiting_reopen = 1, t, 0, True
        r.candidates, r.messages, r.head, r.programs, r.interp = [], [], [], [], None
    assert eng.pick_reopen([a, b, c]) == 0 and eng.pick_reopen([a, b, c]) == 2 and eng.pick_reopen([a, b, c]) == 1


ONE = {**B15, "one_answer": True, "loop_rule": True, "prompt": "arc2", "tool_chars": 3072 * 4, "drop_malformed": True,
       "bad_call_fix": True, "tool_fix": True}
TWO_WORDS = re.compile(r"\b(first|second|guess|guesses)\b|two (guess|answer)", re.I)


def _flat(m):
    c = m.get("content")
    return " ".join(p.get("text", "") for p in c) if isinstance(c, list) else str(c or "")


def _one_engine(script, **over):
    eng, sent = engine(script, **({**ONE, **over} if "open_order" not in over else over))
    eng.repl = lambda run: Repl(getattr(run, "shown", None) or run.task, strict=False, call_seconds=5, memo=True, one_answer=True)
    return eng, sent


def test_base15b_a_second_argument_is_an_error_and_nothing_is_submitted():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({R1}, {R2})"), call(f"submit({R1}, second={R2})"), call(f"submit({R1}, force=True)"),
                             call("x = 1"), call("print(submit.__doc__)"), call(f"submit({R1})")])
    _turn(eng, run)
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert run.submits == 1 and len(_picks(run)) == 1
    assert all("TypeError" in t for t in tools[:3]) and "output grid" in tools[4]
    assert "takes 1 positional argument" in tools[0] and not TWO_WORDS.search(tools[0])
    _close(run)


def test_base15b_every_served_text_names_one_answer():
    from arena.context_drain import MALFORMED_CALL_FOLLOWUP, PYTHON_NOTE, TRUNCATED_CALL_FOLLOWUP
    spec = tasks.tool_spec({**ONE, "conversation": True})["function"]
    static = {"system": tasks.conv_system(ONE), "tool": json.dumps(spec), "task": tasks.task_prompt(TASK),
              "nothing": tasks.nothing_text(ONE), "repeat": tasks.CONV_REPEAT_ONE, "reopen": tasks.CONV_REOPEN_WRONG,
              "recorded": tasks.CONV_RECORDED, "cut": tasks.CONV_CUT, "empty": tasks.CONV_EMPTY_CALL,
              "bad": tasks.CONV_BAD_CALLS, "drain": tasks.DRAIN_NOTE, "loop": tasks.LOOP_NOTE, "py": PYTHON_NOTE,
              "shown": tasks.PLAYBOOK_SHOWN, "none": tasks.PLAYBOOK_EMPTY, "due": tasks.PLAYBOOK_OPEN_DUE,
              "malformed": MALFORMED_CALL_FOLLOWUP, "truncated": TRUNCATED_CALL_FOLLOWUP, "fix": tasks.FIX_LINE}
    assert not {k: TWO_WORDS.search(v).group() for k, v in static.items() if TWO_WORDS.search(v)}
    for gone in ("Submitting ends", "end the puzzle", "may be None", "two guesses"):
        assert gone not in static["system"] + static["tool"]
    assert "submit(answer)" in static["system"] and "submit(answer)" in static["tool"]
    # a whole entry: programs, nothing, bad shape, one answer, reopen, repeat, drain, loop, then a different answer
    run = TaskRun("t#1", TASK, {"conversation": True})
    pass1 = [WRONG, RIGHT, "words only", call("submit([1, 2, 3, 4])"), call("print(submit.__doc__)"), call(f"submit({R1})")]
    eng, sent = _one_engine(pass1 + [call(f"submit({R1})")] + [call(f"print({i}, 'x' * 400)") for i in range(60)]
                            + [call(f"submit({R2})")], window=20480, loop_rule=True)
    _turn(eng, run)
    assert run.submits == 1
    run.parked = False
    eng.pick_reopen([run])
    while run.done is None:
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 2 and run.repeats_refused == 1 and len(_picks(run)) == 2
    served = {}
    for s in sent:
        for m in s["messages"]:
            if m["role"] != "assistant":
                served[_flat(m)] = m["role"]
    assert any(tasks.CONV_REOPEN_WRONG in t for t in served) and any(tasks.CONV_REPEAT_ONE in t for t in served)
    assert any("[Context drain]" in t or "[Loop rule]" in t for t in served)
    bad = {t[:160]: TWO_WORDS.search(t).group() for t in served if TWO_WORDS.search(t)}
    assert not bad
    _close(run)


def test_base15e_serves_the_exact_sentence_and_only_that_change():
    line = "Submitting ends the puzzle, so make sure you are 100% certain that your answer is correct before submitting."
    e = {**ONE, "submit_final": True}
    a, b = tasks.conv_system(ONE), tasks.conv_system(e)
    assert line not in a and b.count(line) == 1 and "Submitting records your answer." not in b
    assert b == a.replace("Submitting records your answer.", line)
    assert tasks.tool_spec({**e, "conversation": True}) == tasks.tool_spec({**ONE, "conversation": True})
    assert tasks.nothing_text(e) == tasks.nothing_text(ONE)
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({R1})")], submit_final=True)
    _turn(eng, run)
    assert _flat(sent[0]["messages"][0]).count(line) == 1
    assert run.submits == 1
    _close(run)


B15C = {**ONE, "submit_mode": "continue", "open_order": "fewest", "park_drain": True, "idle_end": 30, "idle_tokens": 128}
B15C.pop("reopen_wrong"), B15C.pop("reopen_order")
B15D = {**B15C, "recheck": True, "pick": "first_last"}
B15D.pop("second_differs")


def _all_replies(eng, run):
    n = 0
    while run.done is None and n < 400:
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        run.parked, n = False, n + 1


def _served(sent):
    return [_flat(m) for s in sent for m in s["messages"] if m["role"] != "assistant"]


def test_base15c_first_submit_stays_and_is_asked_for_a_different_answer_second_ends_it():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({R1})"), call(f"submit({R1})"), call(f"submit({R2})")], **B15C)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 1 and not run.parked and not getattr(run, "awaiting_reopen", False) and run.done is None
    assert run.messages[-1]["content"] == tasks.CONV_ALTERNATIVE
    held = len(run.messages)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 1 and run.repeats_refused == 1 and "nothing was submitted" in run.messages[-1]["content"]
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 2 and run.done == "submitted" and len(sent[-1]["messages"]) == held + 2
    assert sorted(map(str, _picks(run))) == sorted(map(str, [json.loads(R1), json.loads(R2)]))
    assert not any("wrong" in m["content"] or "guess" in m["content"] for m in sent[-1]["messages"] if m["role"] == "tool")


def test_base15c_a_drain_parks_a_submit_does_not_and_it_comes_back_without_new_text():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"print({i}, 'x' * 400)") for i in range(80)], window=20480, **{**B15C, "loop_rule": False})
    rows = _rows(eng)
    while not getattr(run, "parked", False) and len(sent) < 80:
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.parked and run.done is None and run.drains == 1 and not getattr(run, "awaiting_reopen", False)
    assert [r for r in rows if r.get("event") == "park"][0]["task"] == run.id
    held, n = list(run.messages), len(sent)
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert len(sent) == n + 1 and sent[-1]["messages"] == held
    assert eng.queue_item(run, 3) == ((0, run.spent), 3)
    _close(run)


def test_base15c_backlog_is_fewest_tokens_submitted_or_not():
    eng, _ = engine([], **B15C)
    a, b, c = (TaskRun(f"t#{i}", TASK, {"conversation": True}) for i in (1, 2, 3))
    a.spent, a.submits, b.spent, b.submits, c.spent, c.submits = 100, 1, 900, 0, 0, 0
    assert sorted((eng.queue_item(r, i) for i, r in enumerate((a, b, c)))) == [((0, 0), 2), ((0, 100), 0), ((0, 900), 1)]


def test_backstop_ends_only_an_entry_with_a_submit_and_logs_it():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call("print('ok')")] * 35 + [call(f"submit({R1})"), call("x = 1")] + [call("print('ok')")] * 40, **B15D)
    rows = _rows(eng)
    _all_replies(eng, run)
    assert run.done == "idle" and run.submits == 1 and len(run.replies) == 36
    assert [r["streak"] for r in rows if r.get("event") == "idle_end"] == [36]
    never = TaskRun("t#2", TASK, {"conversation": True})
    eng, _ = _one_engine([call("print('ok')")] * 60, **B15D)
    for _ in range(55):  # the loop rule's clear parks it on the way
        asyncio.run(eng.advance(never, 10 ** 9, replies=1))
        never.parked = False
    assert never.done is None and len(never.replies) >= 50
    _close(never)


def _d(script):
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({r})") for r in script], **B15D)
    _all_replies(eng, run)
    return run, sent


def test_base15d_three_in_a_row_ends_and_non_consecutive_does_not():
    run, sent = _d([R1, R1, R1, R2])
    assert run.done == "identical" and run.submits == 3
    run, sent = _d([R1, R2, R1, R2, R3, R3, R3])
    assert run.done == "identical" and run.submits == 7
    assert sum(tasks.CONV_RECHECK == m["content"] for m in sent[-1]["messages"] if m["role"] == "tool") == 6
    tools = " ".join(m["content"] for m in sent[-1]["messages"] if m["role"] == "tool")
    assert not any(t in tools for t in ("refused", "alternative", "wrong"))


def test_base15d_final_slots_are_the_first_and_the_last_submit_only():
    run, _ = _d([R1, R2, R3, R3, R3])
    assert sorted(map(str, _picks(run))) == sorted(map(str, [json.loads(R1), json.loads(R3)]))
    run, _ = _d([R1, R2, R1, R1, R1])  # first = last: one grid, the middle one is not a guess
    assert _picks(run) == [json.loads(R1)]


def test_base15cd_served_texts():
    for settings, one in ((B15C, tasks.CONV_ALTERNATIVE), (B15D, tasks.CONV_RECHECK)):
        system, tool = tasks.conv_system(settings), json.dumps(tasks.tool_spec({**settings, "conversation": True}))
        assert "submit(answer)" in system and "submit(answer)" in tool and not TWO_WORDS.search(system + tool)
        assert "wrong" not in one and "Submitting ends" not in system
    assert not TWO_WORDS.search(tasks.CONV_RECHECK) and "second answer" in tasks.CONV_ALTERNATIVE
    run, sent = _d([R1, R2, R2, R2])
    assert not {TWO_WORDS.search(t).group() for t in _served(sent) if TWO_WORDS.search(t)}


B15C2 = {**B15C, "differs_first": True, "recheck": "reopen", "submit_final": True, "pick": "first_last", "keep_first": True,
         "churn": False}


def _c2(script):
    eng, sent = _one_engine([call(f"submit({r})") for r in script] + [WRONG], **B15C2)
    events = []
    eng.record = events.append
    run = TaskRun("t#1", TASK, {"conversation": True})
    for _ in script:
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    return run, sent, [e for e in events if e.get("event") == "overwrite"]


def test_base15c2_refuses_only_the_first_answer_a_later_answer_may_come_back():
    run, sent, over = _c2([R1, R2, R3, R2, R1, R2, R2, R2])
    assert run.submits == 7 and run.repeats_refused == 1 and run.done is None and not run.parked
    assert [e["later_repeat"] for e in over] == [False, False, True, True, True, True]
    assert [e["new_answer"] for e in over] == [True, True, True, False, False, False]
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert tools.count(tasks.CONV_REOPEN_WRONG) == 6 and sum("nothing was submitted" in t for t in tools) == 1
    assert tasks.CONV_ALTERNATIVE not in tools


def test_base15c2_final_slots_are_the_first_and_the_last_accepted():
    run, _, _ = _c2([R1, R2, R3, R1])
    assert sorted(map(str, _picks(run))) == sorted(map(str, [json.loads(R1), json.loads(R3)]))


def test_base15c2_serves_the_line_and_reopen_text_verbatim():
    line = "Submitting ends the puzzle, so make sure you are 100% certain that your answer is correct before submitting."
    assert tasks.conv_system(B15C2).count(line) == 1 and "submit(answer)" in tasks.conv_system(B15C2)
    assert tasks.CONV_REOPEN_WRONG == ("Your last answer was slightly wrong. Come up with a different, correct answer and "
                                       "submit it. Submitting the same answer again is refused.")
    run, sent, _ = _c2([R1, R2])
    assert not {TWO_WORDS.search(t).group() for t in _served(sent) if TWO_WORDS.search(t)}


def test_base15c2_ends_only_at_the_idle_end():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = _one_engine([call(f"submit({R1})")] * 3 + [call(f"submit({R2})")] * 3 + [
        {"role": "assistant", "content": "x", "completion_tokens": 10}] * 40, **{**B15C2, "churn": False})
    n = 0
    while run.done is None and n < 60:
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        run.parked, n = False, n + 1
    assert run.done == "idle"


B15F = {**B15D, "recheck": "wrong", "submit_check": True, "submit_final": True, "idle_any": True, "fallback_last": "last"}
B16 = {**B15F, "fallback_last": "passing", "hold_one": 20261101}
FN_RIGHT = "def transform(grid):\n    return [list(r) for r in zip(*grid)]\n"
FN_ID = "def transform(grid):\n    return grid\n"
TASK3 = {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 3], [2, 4]]}, {"input": [[5, 6]], "output": [[5], [6]]},
                   {"input": [[8], [9]], "output": [[8, 9]]}],
         "test": [{"input": [[7, 8, 9]], "output": [[7], [8], [9]]}]}


def _f(script, settings=B15F, task=None):
    eng, sent = engine(script, **settings)
    eng.repl = lambda run: Repl(getattr(run, "shown", None) or run.task, strict=False, call_seconds=5, memo=True,
                                one_answer=True, programs=True)
    events = []
    eng.record = events.append
    run = TaskRun("t#1", task or TASK, {"conversation": True})
    n = 0
    while run.done is None and script and n < 40:
        n += 1
        if getattr(run, "parked", False):
            run.parked = False
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    return run, sent, events


def test_base15f_served_text_names_programs_only():
    system, tool = tasks.conv_system(B15F), json.dumps(tasks.tool_spec({**B15F, "conversation": True}))
    assert "Submitting ends the puzzle, so make sure you are 100% certain that your answer is correct before submitting." in system
    assert "submit(transform)" in system and "submit(transform)" in tool and "submit(transform)" in tasks.nothing_text(B15F)
    assert "Grids alone are refused." in system and "submit(answer)" not in system + tool
    assert not TWO_WORDS.search(system + tool) and "hidden" not in system
    assert tasks.CONV_RECHECK_WRONG == "This answer is slightly wrong. Continue iterating, and submit again after revising."
    assert "whose output is hidden" in tasks.conv_system(B16)


def test_base15f_refuses_grids_and_a_function_that_misses_a_pair():
    run, sent, _ = _f([call(f"submit({R1})"), call(FN_ID + "submit(transform)"), call("submit([1, 2])")])
    assert run.submits == 0 and run.refusals == {"grids": 1, "train": 1}
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert tasks.CONV_SUBMIT_GRIDS_ONE in tools[0] and "does not reproduce every training pair" in tools[1]
    assert not any(TWO_WORDS.search(x) for x in tools)
    run, _, _ = _f([call(f"submit({R1}, force=True)")])
    assert run.submits == 0


def test_base15f_accepted_submits_told_wrong_end_on_three_identical():
    run, sent, _ = _f([call(FN_RIGHT + "submit(transform)")] * 3 + [WRONG])
    assert run.done == "identical" and run.submits == 3
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert tools.count(tasks.CONV_RECHECK_WRONG) == 2
    assert _picks(run) == [json.loads(R1)]


def test_base15f_no_accepted_submit_scores_the_last_program():
    wrong2 = "```python\ndef transform(grid):\n    return [[1]]\n```"
    run, _, _ = _f([RIGHT, wrong2])
    assert run.submits == 0 and _picks(run) == [[[1]]]
    run, _, _ = _f([RIGHT, wrong2], B16, TASK3)
    assert _picks(run) == [json.loads(R1)]  # base16: the last that fits the visible pairs


def test_base15f_idle_end_covers_an_entry_with_no_accepted_submit():
    run = TaskRun("t#1", TASK, {"conversation": True})
    run.replies, run.submits = [{"completion_tokens": 10}], 0
    eng, _ = engine([], **B15F)
    eng.record = lambda e: None
    run.idle_streak = 29
    assert eng.idle_end(run, B15F) and run.done == "idle"
    run.done, run.idle_streak = None, 29
    assert not eng.idle_end(run, B15D)


def _png_grids(parts):
    return [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]


def test_base16_held_input_is_served_its_output_never_and_feedback_is_pass_fail():
    run, sent, events = _f([call(FN_RIGHT + "submit(transform)")] * 3 + [WRONG], B16, TASK3)
    pair = next(e for e in events if e["event"] == "held_pair")
    held = TASK3["train"][pair["index"]]
    assert pair["pairs"] == 3 and len(run.shown["train"]) == 2 and run.shown["extra"] == [held["input"]]
    assert held not in run.shown["train"] and run.submits == 3
    served = json.dumps([s["messages"] for s in sent])
    assert json.dumps(tasks.text(held["output"]))[1:-1] not in served
    assert json.dumps(f"Extra input 1 ({tasks.shape(held['input'])}), output hidden:\n{tasks.text(held['input'])}")[1:-1] in served
    assert [e["passed"] for e in events if e["event"] == "held_check"] == [True] * 3
    system = tasks.conv_system(B16)
    assert "`extra_inputs`" in system and "whose output is hidden" in system and "test_inputs`, `extra_inputs` and `np`" in system
    assert "`extra_inputs`" in json.dumps(tasks.tool_spec({**B16, "conversation": True}))
    # images: the extra input's panel is the input alone, its output's panel is nowhere
    shown = run.shown
    urls = _png_grids(tasks.image_parts(shown))
    alone = tasks.png_url(tasks.grid_pixels([held["input"]], image_side=tasks.IMAGE_SIDE))
    out = tasks.png_url(tasks.grid_pixels([held["output"]], image_side=tasks.IMAGE_SIDE))
    pair_png = tasks.png_url(tasks.grid_pixels([held["input"], held["output"]], image_side=tasks.IMAGE_SIDE))
    assert alone in urls and out not in urls and pair_png not in urls and len(urls) == 2 + 1 + 1
    # the interpreter: the input is there, the output is not, whatever the model prints
    repl = Repl(shown, strict=False, call_seconds=5, memo=True, one_answer=True, programs=True)
    seen = repl.run("print(extra_inputs); print(train); print(sorted(k for k in globals() if not k.startswith('__')))")["stdout"]
    repl.close()
    assert str(held["input"]) in seen and str(held["output"]) not in seen
    bad = ("def transform(grid):\n    if grid in extra_inputs:\n        return [[0]]\n"
           "    return [list(r) for r in zip(*grid)]\n")
    run, sent, events = _f([call(bad + "submit(transform)")], B16, TASK3)
    checks = [e for e in events if e["event"] == "held_check"]
    assert checks[0]["passed"] is False and checks[0]["right"] == [True] and run.refusals == {"held": 1} and run.submits == 0
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"] if len(sent) > 1 else []
    run, sent, _ = _f([call(bad + "submit(transform)"), WRONG], B16, TASK3)
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert tasks.CONV_HELD_FAIL in tools[0]
    assert str(held["output"]) not in tools[0] and tasks.text(held["output"]) not in tools[0]


B17 = {**B15C, "reopen_wrong": True, "submit_final": True, "differs_first": True, "fresh_tries": True, "fresh_text": "blank",
       "fresh_interp": True, "pass_clean": True, "keep_first": False, "pick": "vote_all", "idle_any": True}


def _t17(script):
    eng, sent = _one_engine(script + [WRONG], **B17)
    events = []
    eng.record = events.append
    run = TaskRun("t#1", TASK, {"conversation": True})
    for _ in script:
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        if run.parked:
            events.append({"event": "left_slot", "messages": len(run.messages), "interp": run.interp})
        run.parked = False
    return run, sent, events


def _subs(*grids):
    return [call(f"submit({g})") for g in grids]


def test_base17_the_wall_ends_a_roster_with_no_stop_and_submission_json_holds_the_last_vote(tmp_path, monkeypatch):
    """Competition run: no task ends by itself and no mark is reached; the wall ends play and the file
    holds each task's latest vote, not its latest submit."""
    import time

    import arc2.trial as trial_mod

    from test_arc2 import _scripted_roster

    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.02")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "tenth")
    file, seen = tmp_path / "submission.json", []
    trial.update(wall_seconds=6, submission=str(file), arms=[{"name": "b17", "conversation": True, "tool": True,
                                                             "images": False, **B17}])
    real = trial_mod.submission

    def spy(runs):
        seen[:] = list(runs)
        return real(runs)
    monkeypatch.setattr(trial_mod, "submission", spy)
    report = asyncio.run(trial_mod.run_arc2(study, trial, source=str(source), output=tmp_path / "out", server_dir=server,
                                            base_url=url, deadline_epoch=time.time() + 120))
    assert report["status"] == "complete" and report["wall"]["seconds"] == 6 and not report["task_errors"]
    written = json.loads(file.read_text())
    assert set(written) == set(trial["tasks"]) and {r.done for r in seen} == {None}
    assert max(r.try_n for r in seen) >= 2  # tries were still running when the wall came
    voted = differs = 0
    for run in seen:
        votes = [c for c in run.candidates if c.get("only")]
        if not votes:
            continue
        voted += 1
        last = max(c["submit_n"] for c in votes)
        one, two = (next((c["tests"] for c in votes if c["submit_n"] == last and c["submitted"] == k), None) for k in (1, 2))
        for k, row in enumerate(written[run.id]):
            assert row["attempt_1"] == one[k] and row["attempt_2"] == (two or one)[k]
        newest = max((c for c in run.candidates if c.get("submitted") and not c.get("vote")), key=lambda c: c["submit_n"])
        differs += one != newest["tests"]
    assert voted == len(seen) and differs  # the file is the vote, not the latest submit


def test_base17_two_submits_end_a_try_and_the_next_is_blank_in_a_new_interpreter():
    run, sent, events = _t17(_subs(R1, R2, R3))
    assert run.done is None and run.submits == 3 and run.try_n == 2 and run.try_base == 2
    assert [m["content"] for m in sent[1]["messages"] if m["role"] == "tool"] == [tasks.CONV_REOPEN_WRONG]
    left = next(e for e in events if e["event"] == "left_slot")
    assert left["messages"] == 2  # system and task only
    fresh = sent[2]["messages"]
    assert len(fresh) == 2 and fresh == sent[0]["messages"]
    end = next(e for e in events if e["event"] == "try_end")
    assert end["cause"] == "submits" and end["try"] == 1 and end["submits"] == 2


def test_base17_a_parked_round_1_task_with_more_tokens_beats_a_round_2_task_with_fewer():
    eng, _ = engine([], **B17)
    late, ahead, next_up = (TaskRun(f"t#{i}", TASK, {"conversation": True}) for i in (1, 2, 3))
    late.spent, ahead.spent, next_up.spent = 100000, 40000, 90000
    ahead.try_n = 2
    order = sorted(eng.queue_item(r, i) for i, r in enumerate((late, ahead, next_up)))
    assert [n for _, n in order] == [2, 0, 1]  # tokens only break ties within a round


def test_base17_second_differs_from_its_own_try_only_and_tries_may_repeat_each_other():
    run, sent, _ = _t17(_subs(R1, R2, R1, R1, R2))
    assert run.submits == 4 and run.repeats_refused == 1 and run.try_n == 3
    assert "nothing was submitted" in [m["content"] for m in sent[4]["messages"] if m["role"] == "tool"][-1]


def test_base17_final_slots_are_the_most_agreed_grids_ties_to_first_submits_then_oldest():
    run, _, _ = _t17(_subs(R1, R2, R1, R3))
    assert _picks(run) == [json.loads(R1), json.loads(R2)]  # R1 2 votes; R2 and R3 tie, neither first: oldest
    run, _, _ = _t17(_subs(R1, R2, R1, R3, R3, R2))
    assert _picks(run) == [json.loads(R1), json.loads(R3)]  # 2 votes each; R2 never a try's first
    late = [c for c in run.candidates if c.get("vote") and c["submit_at"] == 4]
    picks = guesses(run.candidates, [TASK["test"][0]["input"]], limit=late[0]["tokens_at"])[0]
    assert len(picks) == 2


def test_base17_only_submitted_grids_take_a_slot():
    run, _, _ = _t17([WRONG, call(f"submit({R1})"), "```python\ndef transform(grid):\n    return [[1]]\n```"])
    assert run.submits == 1 and _picks(run) == [json.loads(R1)]


def test_base17_idle_ends_the_try_with_or_without_a_submit_never_the_task():
    eng, sent = _one_engine([call("print(1)"), WRONG], **B17)
    run = TaskRun("t#1", TASK, {"conversation": True})
    asyncio.run(eng.advance(run, 10 ** 9, replies=0))
    run.idle_streak = 29
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.done is None and run.try_n == 2 and run.submits == 0 and run.parked and len(run.messages) == 2


B17B = {k: v for k, v in B17.items() if k not in ("reopen_wrong", "differs_first", "second_differs", "one_answer")}
B17B.update(submit_two=True, no_first_line=True)


def _t17b(script, task=TASK):
    eng, sent = engine(script + [WRONG], **B17B)
    run = TaskRun("t#1", task, {"conversation": True})
    for _ in script:
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        run.parked = False
    return run, sent


def _two(a, b=None):
    return call(f"submit({a}, {b})" if b else f"submit({a})")


def test_base17b_served_text_is_two_guesses_and_the_line():
    system, tool = tasks.conv_system(B17B), json.dumps(tasks.tool_spec({**B17B, "conversation": True}))
    assert "You have two guesses per test input" in system and "`second` must differ from `first`." in system
    assert system.count(tasks.SUBMIT_FINAL_LINE) == 1 and "Submitting ends the puzzle." not in system
    assert "submit(first, second)" in tool and "submit(first, second=None)" not in tool
    assert "different on every test input" in system and "two with the same grid on any test input" in system
    assert "submit(answer)" not in system + tool and "more confident" not in system and "Your answer counts" not in system


def test_base17b_one_accepted_submit_ends_the_try_and_a_refusal_records_nothing():
    run, sent = _t17b([_two(R1), _two(R1, R1), _two(R1, R2), _two(R1, R3)])
    assert run.submits == 2 and run.try_n == 3 and getattr(run, "two_refusals", 0) == 2 and run.done is None
    tools = [m["content"] for s in sent for m in s["messages"] if m["role"] == "tool"]
    assert any(tasks.CONV_SUBMIT_TWO_GUESSES in t for t in tools) and any(tasks.CONV_SUBMIT_SAME_OUTPUT in t for t in tools)
    assert not any("wrong" in t for t in tools) and tasks.CONV_REOPEN_WRONG not in tools
    assert len(sent[3]["messages"]) == 2  # the next try is blank


def test_base17b_guesses_must_differ_on_every_output():
    task = {"train": TASK["train"], "test": TASK["test"] + TASK["test"]}
    run, _ = _t17b([call("submit([[[1]], [[2]]], [[[3]], [[2]]])"), call("submit([[[1]], [[2]]], [[[3]], [[4]]])")], task)
    assert run.two_refusals == 1 and run.submits == 1


def test_base17b_votes_over_all_guesses_ties_to_a_tries_first_guess_then_oldest():
    run, _ = _t17b([_two(R1, R2), _two(R3, R1)])
    assert _picks(run) == [json.loads(R1), json.loads(R3)]  # R1 2 votes; R3 a first guess, R2 older but a second
    run, _ = _t17b([_two(R1, R2), _two(R2, R3), _two(R3, R1)])
    assert _picks(run) == [json.loads(R1), json.loads(R2)]  # 2 votes each, all first guesses: oldest


V56 = {**B15C, "reopen_wrong": True, "submit_final": True, "keep_first": True}


def test_v56_first_submit_stays_hears_reopen_wrong_second_ends_it_both_score():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({R1})"), call(f"submit({R1})"), call(f"submit({R2})")], **V56)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 1 and not run.parked and run.done is None and run.messages[-1]["content"] == tasks.CONV_REOPEN_WRONG
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.submits == 1 and run.repeats_refused == 1
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.done == "submitted" and run.submits == 2
    assert sorted(map(str, _picks(run))) == sorted(map(str, [json.loads(R1), json.loads(R2)]))
    system = tasks.conv_system(V56)
    assert system.count(tasks.SUBMIT_FINAL_LINE) == 1 and "submit(answer)" in system


def test_v56r_a_try_drains_on_v56s_jitter_keeps_its_tail_and_the_next_try_is_blank():
    """v56r: v56's context and drain inside base17r's fresh rounds; the drain index runs on across tries."""
    v56r = {**B17, "drain": {"turn_cap": 0, "keep_tokens": 2000, "jitter": {"low": 6000, "high": 6000, "seed": 0}},
            "loop_rule": False, "idle_end": 500}  # 100-token fake replies would end the try as idle at 30
    script = [call(f"print({i}, 'x' * 400)") for i in range(30)] + _subs(R1, R2) + [call("print(1)"), WRONG]
    eng, sent = _one_engine(script, window=20480, **v56r)
    rows = _rows(eng)
    run = TaskRun("t#1", TASK, {"conversation": True})
    while getattr(run, "try_n", 1) < 2:
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        run.parked = False
        assert len(sent) <= 32
    assert run.drains >= 1 and [r for r in rows if r.get("event") == "drain_trigger"][0]["trigger_tokens"] == 6000
    assert any(r.get("event") == "context_drain" for r in rows)
    drained, n = run.drains, len(sent)
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert sent[n]["messages"] == sent[0]["messages"] and len(sent[n]["messages"]) == 2  # blank: no drain note, no tail
    assert run.drains == drained and run.submits == 2 and run.done is None
    assert sum(1 for c in run.candidates if c.get("only") and c["submit_n"] == 2.5) == 2  # the vote of both answers
    _close(run)


B16B = {**B15C, "reopen_wrong": True, "submit_check": True, "submit_final": True, "idle_any": True, "fallback_last": "passing",
        "hold_one": 20261101, "keep_first": True}
FN_ALT = ("def transform(grid):\n    if len(grid[0]) == 3:\n        return [[1], [2], [3]]\n"
          "    return [list(r) for r in zip(*grid)]\n")


def test_base16b_second_accepted_differs_ends_it_and_both_score():
    run, sent, events = _f([call(FN_RIGHT + "submit(transform)")] * 2 + [call(FN_ALT + "submit(transform)")], B16B, TASK3)
    assert run.done == "submitted" and run.submits == 2 and run.refusals == {"repeat": 1}
    tools = [m["content"] for m in sent[-1]["messages"] if m["role"] == "tool"]
    assert tools[0] == tasks.CONV_REOPEN_WRONG and "nothing was submitted" in tools[1]
    assert sorted(map(str, _picks(run))) == sorted(map(str, [json.loads(R1), json.loads(R2)]))
    assert [(e["n"], e["passed"]) for e in events if e["event"] == "held_check"] == [(1, True), (2, True)]


def test_base16b_refusals_record_nothing_and_grids_are_refused():
    bad = ("def transform(grid):\n    if grid in extra_inputs:\n        return [[0]]\n"
           "    return [list(r) for r in zip(*grid)]\n")
    run, _, _ = _f([call(bad + "submit(transform)"), call(f"submit({R1})"), call(FN_ID + "submit(transform)"), WRONG],
                   B16B, TASK3)
    assert run.submits == 0 and run.refusals == {"held": 1, "grids": 1, "train": 1}


def test_base16b_only_accepted_answers_score_else_the_last_passing_program():
    late = FN_ALT.replace("[[1], [2], [3]]", "[[4], [4], [4]]") + "print('x')"
    run, _, _ = _f([call(FN_RIGHT + "submit(transform)"), "```python\n" + late + "\n```"], B16B, TASK3)
    assert run.submits == 1 and _picks(run) == [json.loads(R1)]
    run, _, _ = _f([call(FN_ID + "submit(transform)"), "```python\n" + late + "\n```"], B16B, TASK3)
    assert run.submits == 0 and _picks(run) == [json.loads(R3)]


def test_base16b_holds_the_same_pair_as_base16():
    pair = [next(e for e in _f([WRONG], s, TASK3)[2] if e["event"] == "held_pair")["index"] for s in (B16, B16B)]
    assert pair[0] == pair[1]


def test_base16_two_pair_task_shows_one_pair_and_one_extra_input():
    run, sent, events = _f([WRONG], B16, TASK)
    assert len(run.shown["train"]) == 1 and len(run.shown["extra"]) == 1
    assert next(e for e in events if e["event"] == "held_pair")["pairs"] == 2


def test_bf16x2_ple_queue_waits_where_the_pinned_put_failed(tmp_path, monkeypatch):
    """v3's queue.Full: a one-slot queue whose staging thread is late. The patched line waits it out."""
    import hashlib
    import queue
    import threading
    from types import SimpleNamespace

    import pytest

    from arc2 import b300_serving, ple_queue_patch
    q = queue.Queue(maxsize=1)
    q.put("previous")
    with pytest.raises(queue.Full):
        q.put_nowait("next")
    threading.Timer(.1, q.get).start()
    source = ("class C:\n    def f(self, request):\n" + ple_queue_patch.ANCHOR).encode()
    monkeypatch.setattr(ple_queue_patch, "SHA256", hashlib.sha256(source).hexdigest())
    text = ple_queue_patch.patched(source, 5).decode()
    assert "put_nowait" not in text and "timeout=5.0" in text
    namespace = {}
    exec(text, namespace)
    namespace["C"].f(SimpleNamespace(_request_queue=q), "next")
    assert q.get_nowait() == "next"
    site = tmp_path / "usr/local/lib/python3.12/dist-packages" / ple_queue_patch.RELATIVE
    site.parent.mkdir(parents=True)
    site.write_bytes(source)
    recipe = SimpleNamespace(RUNTIME_ROOT=tmp_path, verify_and_extract_runtime=lambda: "extracted", server_command=lambda: [],
                             update_ready_identity=lambda i: i, runtime_environment=lambda: ({}, {}))
    b300_serving.patch_recipe(recipe, tmp_path, "/assets/expert-bf16/x")
    assert recipe.verify_and_extract_runtime() == "extracted"
    assert "put_nowait" not in site.read_text() and f"timeout={float(b300_serving.PLE_QUEUE_WAIT)}" in site.read_text()


def test_bf16x2_recipe_serves_the_bf16_folder_on_two_cards(tmp_path):
    from types import SimpleNamespace

    from arc2.b300_serving import patch_recipe
    argv = ["python", "-m", "vllm", "serve", "/kaggle/nvfp4", "--dtype", "bfloat16", "--quantization", "modelopt_fp4",
            "--tensor-parallel-size", "1", "--moe-backend", "humming", "--port", "8000"]
    recipe = SimpleNamespace(server_command=lambda *a, **k: list(argv), update_ready_identity=lambda i: i,
                             verify_and_extract_runtime=lambda: None,
                             runtime_environment=lambda *a, **k: ({"TORCH_CUDA_ARCH_LIST": "10.0", "CUDA_VISIBLE_DEVICES": "0"}, {}))
    patch_recipe(recipe, tmp_path, "/assets/expert-bf16/x")
    out = recipe.server_command()
    assert out[out.index("serve") + 1] == "/assets/expert-bf16/x" and out[out.index("--tensor-parallel-size") + 1] == "2"
    assert "--quantization" not in out and "--moe-backend" not in out and "--dtype" in out
    env, _ = recipe.runtime_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and env["NCCL_CUMEM_HOST_ENABLE"] == "0"
    recipe.runtime_environment = None



B18 = {**ONE, "submit_final": True, "third": True}


def _sel_first(run_id, k=0):
    import hashlib
    return hashlib.sha256(f"{run_id}|{k}".encode()).digest()[0] % 2 == 1  # True: answer 3 is shown as A


def _g18(run):
    return [[g for g, _ in pick] for pick in guesses(run.candidates, [t["input"] for t in run.task["test"]])][0]


def test_base18_third_is_asked_as_the_second_and_the_selector_picks_guess_2():
    run = TaskRun("t#1", TASK, {"conversation": True})
    letter = "A" if _sel_first("t#1") else "B"  # whichever letter shows answer 3
    eng, sent = _one_engine([call(f"submit({R1})"), call(f"submit({R2})"), call(f"submit({R3})"),
                             f"Both fit some pairs.\nPICK: {letter}"], **B18)
    _turn(eng, run)
    assert run.submits == 1 and run.awaiting_reopen and run.messages[-1]["content"] == "Recorded as your answer (submission 1)."
    _reopen(eng, run)
    assert run.submits == 2 and run.awaiting_reopen and run.done is None
    assert run.messages[-1]["content"] == "Recorded as your answer (submission 2)."  # as after the first answer
    assert _g18(run) == [json.loads(R1), json.loads(R2)]
    run.parked = False
    eng.pick_reopen([run])
    assert run.messages[-1]["content"] == tasks.CONV_REOPEN_WRONG  # the same ask as for the second answer
    _turn(eng, run)
    assert run.submits == 3 and run.done is None and run.select_due == [0] and run.parked
    assert _g18(run) == [json.loads(R1), json.loads(R2)]  # until the selector answers: answer 2
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    ask = sent[-1]
    assert ask["settings"]["tool"] is False and ask["messages"][:-1] == run.head
    shown = ask["messages"][-1]["content"]
    assert "PICK: A or PICK: B" in shown and tasks.text(json.loads(R2)) in shown and tasks.text(json.loads(R3)) in shown
    assert run.done == "submitted" and _g18(run) == [json.loads(R1), json.loads(R3)]
    assert run.select_log == [{"output": 0, "letter": letter, "third_first": _sel_first("t#1"), "picked": 3}]
    assert run.replies[-1]["phase"] == "selector" and run.spent >= 4 * 100
    _close(run)


def test_base18_an_unreadable_selector_keeps_answer_2_and_no_third_keeps_answer_2():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, _ = _one_engine([call(f"submit({R1})"), call(f"submit({R2})"), call(f"submit({R3})"), "I like the first one."],
                         **B18)
    _turn(eng, run)
    _reopen(eng, run)
    run.parked = False
    eng.pick_reopen([run])
    _turn(eng, run)
    run.parked = False
    asyncio.run(eng.advance(run, 10 ** 9, replies=1))
    assert run.done == "submitted" and _g18(run) == [json.loads(R1), json.loads(R2)]
    assert run.select_log[0]["letter"] is None and run.select_log[0]["picked"] is None
    _close(run)
    other = TaskRun("t#2", TASK, {"conversation": True})
    eng, _ = _one_engine([call(f"submit({R1})"), call(f"submit({R2})")], **B18)
    _turn(eng, other)
    _reopen(eng, other)
    assert other.done is None and _g18(other) == [json.loads(R1), json.loads(R2)]  # the wall here: a1, a2
    _close(other)


def test_base18_levels_order_all_firsts_then_second_passes_then_third_passes_then_selectors():
    eng, _ = engine([], **B18)
    runs = [TaskRun(f"t#{i}", TASK, {"conversation": True}) for i in range(6)]
    for r in runs:
        r.candidates, r.messages, r.head, r.programs, r.interp = [], [], [], [], None
    o, mid2, tok2, mid3, tok3, sel = runs
    o.spent = 500000
    mid2.submits, mid2.reopen_count, mid2.spent = 1, 1, 10
    tok2.submits, tok2.reopen_count, tok2.awaiting_reopen = 1, 0, True
    mid3.submits, mid3.reopen_count, mid3.spent = 2, 2, 10
    tok3.submits, tok3.reopen_count, tok3.awaiting_reopen = 2, 1, True
    sel.submits, sel.reopen_count, sel.select_due = 3, 2, [0]
    keys = [eng.queue_item(r, n)[0] for n, r in enumerate(runs)]
    assert keys == sorted(keys) and len(set(keys)) == 6
    tok3.first_submit_tokens, tok2.first_submit_tokens = 1, 99999  # a third pass never goes before a second
    assert eng.pick_reopen(runs) == 2 and eng.pick_reopen(runs) == 4


B19 = {**ONE, "submit_final": True, "sure": True}


def test_base19_strong_ask_then_sure_ask_repeats_accepted_and_five_in_a_row_close():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({R1})"), call(f"submit({R1})"), call(f"submit({R2})"), call(f"submit({R3})"),
                             call(f"submit({R3})"), call(f"submit({R2})")] + [call(f"submit({R2})")] * 4, **B19)
    rows = _rows(eng)
    _turn(eng, run)
    assert run.submits == 1 and run.awaiting_reopen
    run.parked = False
    eng.pick_reopen([run])
    assert run.messages[-1]["content"] == tasks.CONV_WRONG_STRONG
    _turn(eng, run)
    assert run.submits == 2 and run.repeats_refused == 1  # the second answer must still differ from the first
    assert run.awaiting_reopen and run.messages[-1]["content"] == "Recorded as your answer (submission 2)."
    assert _g18(run) == [json.loads(R1), json.loads(R2)]
    for n, latest in ((3, R3), (4, R3), (5, R2)):  # later answers: repeats accepted, each followed by the same question
        run.parked = False
        eng.pick_reopen([run])
        assert run.messages[-1]["content"] == tasks.CONV_SURE
        _turn(eng, run)
        assert run.submits == n and run.done is None and _g18(run) == [json.loads(R1), json.loads(latest)]
    while run.done is None:
        run.parked = False
        eng.pick_reopen([run])
        _turn(eng, run)
    assert run.done == "identical5" and run.submits == 9  # R2 at submits 5-9: five in a row
    assert _g18(run) == [json.loads(R1), json.loads(R2)]
    logs = [r for r in rows if r.get("event") == "b19_submit"]
    assert [r["same_run"] for r in logs] == [1, 1, 2, 1, 2, 3, 4, 5] and logs[0]["right"] == [False]
    _close(run)


def test_base19_levels_keep_fewer_passes_first():
    eng, _ = engine([], **B19)
    a, b = (TaskRun(f"t#{i}", TASK, {"conversation": True}) for i in (1, 2))
    for r in (a, b):
        r.candidates, r.messages, r.head, r.programs, r.interp = [], [], [], [], None
    a.submits, a.reopen_count, a.awaiting_reopen, a.first_submit_tokens = 4, 3, True, 1
    b.submits, b.reopen_count, b.awaiting_reopen, b.first_submit_tokens = 1, 0, True, 99999
    assert eng.queue_item(b, 1)[0] < eng.queue_item(a, 0)[0] and eng.pick_reopen([a, b]) == 1


def test_base19w256_only_the_second_answer_ask_changes():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({R1})"), call(f"submit({R2})")], **{**ONE, "submit_final": True, "strong_ask": True})
    _turn(eng, run)
    run.parked = False
    eng.pick_reopen([run])
    assert run.messages[-1]["content"] == tasks.CONV_WRONG_STRONG
    _turn(eng, run)
    assert run.done == "submitted" and run.submits == 2  # the second answer still ends the task, as base15e
    _close(run)


# base20
B20 = {"sandbox_fix": True, "check_fn": True, "strong_ask": True, "reopen_picture": True, "images": True,
       "image_side": 896, "image_changes": False, "submit_final": True}


def test_base20_sandbox_fixes_imports_getsource_and_result():
    r = Repl(TASK, strict=False, call_seconds=5, sandbox_fix=True)
    try:
        out = r.run("import train, test_inputs, np\nimport np as _np\nfrom np import zeros\nprint(len(train), _np is np, "
                    "zeros(2).tolist())")
        assert "error" not in out and out["stdout"].strip() == "2 True [0.0, 0.0]"
        assert "error" not in r.run("print(result)\nresult = 5") and r.run("print(result)")["stdout"].strip() == "None"
        assert "error" not in r.run("def f(g):\n    return g\n")
        got = r.run("import inspect\nprint(inspect.getsource(f))")
        assert "error" not in got and got["stdout"].strip() == "def f(g):\n    return g"
        r.define("program_1", "def transform(grid):\n    return grid[::-1]\n")
        assert "return grid[::-1]" in r.run("import inspect\nprint(inspect.getsource(program_1))")["stdout"]
        assert r.run("x = 1\n1/0")["error"] == "ZeroDivisionError: division by zero (line 2)"  # line numbers kept
        assert "SyntaxError" in r.run("def (")["error"]
        assert "module not available" in r.run("import os")["error"]
        assert ["f", "(g)", ""] in r.inventory()["functions"]  # the model's own definitions are still inventoried
    finally:
        r.close()
    plain = Repl(TASK, strict=False, call_seconds=5)  # off: base15e's sandbox
    try:
        assert "module not available" in plain.run("import train")["error"]
        assert "NameError" in plain.run("print(result)")["error"]
        assert "check" in plain.run("check")["error"]
    finally:
        plain.close()


def test_base20_check_prints_pairs_tests_and_the_ledger():
    r = Repl(TASK, strict=False, call_seconds=5, check_fn=True)
    try:
        good = r.run("def t1(g):\n    return [list(c) for c in zip(*g)]\nprint(check(t1))")["stdout"]
        assert "pair 1: pass" in good and "pair 2: pass" in good and "test 1 output:\n7\n8\n9" in good
        assert "Distinct test outputs of functions that pass every training pair: 1" in good
        assert "#1 from t1 (check 1): this function's output above" in good and good.strip().endswith("True")
        bad = r.run("def t2(g):\n    return g\ncheck(t2)")["stdout"]
        assert "pair 1: FAIL 2 cells differ; first (row, col, got, expected): [(0, 1, 2, 3), (1, 0, 3, 2)]" in bad
        assert "pair 2: FAIL shape 1x2, expected 2x1" in bad and "0/2 training pairs pass." in bad
        assert "#1 from t1 (check 1):\ntest 1:\n7\n8\n9" in bad  # the ledger is listed after every call
        same = r.run("def t3(g):\n    return [list(r) for r in zip(*g)]\ncheck(t3)")["stdout"]
        assert "Same test outputs as an earlier passing function." in same and "#1 from t1 (check 1), t3 (check 3)" in same
        err = r.run("check(lambda g: 1/0)")["stdout"]
        assert "pair 1: error ZeroDivisionError" in err and "test 1: error ZeroDivisionError" in err
        assert "TypeError" in r.run("check([[1]])")["error"]
        assert "check" not in json.dumps(r.inventory())  # preset, not the model's
    finally:
        r.close()


def test_base20_system_text_and_tool_say_check_not_programs():
    s = {**ONE, **B20, "conversation": True}
    text, tool = tasks.conv_system(s), tasks.tool_spec({**s, "tool": True})["function"]["description"]
    assert tasks.CONV_CHECK_PARA in text and "```python" not in text and "distinct test outputs of the programs" not in text
    assert tasks.SUBMIT_FINAL_LINE in text and "call `submit(answer)` in the interpreter" in text
    base = tasks.conv_system({**ONE, "submit_final": True})
    assert text == base.replace(tasks.CONV_PROGRAM_PARAS, tasks.CONV_CHECK_PARA)  # nothing else changes
    assert "`check(fn)`" in tool and "program_N" not in tool and "submit(answer)" in tool


def test_base20_salvage_takes_the_longest_lost_code_body():
    assert tasks.salvage_code("\n\n<tool_call>\n<function=python>\n</parameter>\n<parameter=code>\nprint(1)\nprint(2)\n"
                              "</parameter>\n</parameter>\n<parameter=code>\n\n</parameter>\n</function>") == "print(1)\nprint(2)"
    assert tasks.salvage_code("<tool_call>\n<function=python>\n<tool_call>\n<function=code='''\ndef f(g):\n    return g\n"
                              "'''>\n</function>") == "def f(g):\n    return g"
    assert tasks.salvage_code("<function=python>\n<parameter=code>\n\n```python\nx = 1\n```\n</parameter>") == "x = 1"
    assert tasks.salvage_code("<tool_call>\n<function=python>\n</function>\n</tool_call>") is None
    assert tasks.salvage_code("") is None


def test_base20_an_empty_call_runs_its_salvaged_body_only_with_the_fix():
    raw = "\n\n<tool_call>\n<function=python>\n<parameter=code>\nprint(len(train) * 7)\n</parameter>\n</function>"
    for fix in (True, False):
        run = TaskRun("t#1", TASK, {"conversation": True})
        eng, sent = _one_engine([call(""), call(f"submit({R1})")], **{**ONE, **B20, "sandbox_fix": fix})
        complete = eng.complete

        async def with_raw(*a, complete=complete):
            reply = await complete(*a)
            return {**reply, "raw": raw} if len(sent) == 1 else reply

        eng.complete = with_raw
        rows = _rows(eng)
        _turn(eng, run)
        tool = [m for m in sent[1]["messages"] if m["role"] == "tool"][0]["content"]
        first = [r for r in rows if r.get("event", "reply") == "reply"][0]
        if fix:
            assert tool.strip() == "14" and first.get("salvaged") and first["tool_code"] == "print(len(train) * 7)"
            assert json.loads(sent[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"]) == {
                "code": "print(len(train) * 7)"}
        else:
            assert tool.startswith("Error:") and not first.get("salvaged")
        _close(run)


def test_base20_reopen_shows_the_strong_ask_then_each_test_input_beside_the_answer():
    """think1 lesson: an image-bearing prompt rebuild needs its own check; the reopen message carries pictures."""
    task = {**TASK, "test": TASK["test"] + [{"input": [[1, 1]], "output": [[1], [1]]}]}
    two = "[[[7], [8], [9]], [[1], [1]]]"
    run = TaskRun("t#1", task, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({two})"), call("print(1)"), call(f"submit([[[1], [2], [3]], [[2], [2]]])")],
                            **{**ONE, **B20})
    eng.repl = lambda run: Repl(getattr(run, "shown", None) or run.task, strict=False, call_seconds=5, memo=True,
                                one_answer=True, sandbox_fix=True, check_fn=True)
    _turn(eng, run)
    head = sent[0]["messages"][1]["content"]
    assert isinstance(head, list) and sum(p["type"] == "image_url" for p in head) == 4  # 2 pairs + 2 test inputs
    run.parked = False
    eng.pick_reopen([run])
    msg = run.messages[-1]["content"]
    assert isinstance(msg, list) and msg[0] == {"type": "text", "text": tasks.CONV_WRONG_STRONG}
    assert [p.get("text") for p in msg[1::2]] == [tasks.ANSWER_PICTURE_K.format(k=1), tasks.ANSWER_PICTURE_K.format(k=2)]
    want = [tasks.png_url(tasks.grid_pixels([t["input"], a], image_side=896)) for t, a in
            zip(task["test"], [[[7], [8], [9]], [[1], [1]]])]
    assert [p["image_url"]["url"] for p in msg[2::2]] == want
    _turn(eng, run)
    assert run.done == "submitted" and run.submits == 2
    later = sent[-1]["messages"]
    assert later[1]["content"] == head and any(m.get("content") == msg for m in later)  # both picture sets re-sent intact
    one = tasks.answer_parts(TASK, [[[7], [8], [9]]], 896)
    assert one[0]["text"] == tasks.ANSWER_PICTURE == "\nYour submitted answer (right) for the test input (left):"
    _close(run)


def test_base20_design_builds_its_settings():
    import importlib.util
    spec = importlib.util.spec_from_file_location("designs", Path(__file__).resolve().parents[1] / "scripts/arc2_weights/build_rtx_designs.py")
    src = (Path(__file__).resolve().parents[1] / "scripts/arc2_weights/build_rtx_designs.py").read_text()
    ns = {}
    exec(compile(src.split("HEAD_FILES = ")[0], "designs", "exec"), {"__file__": spec.origin, **ns}, ns)
    d, e = ns["DESIGNS"]["base20"], ns["DESIGNS"]["base15e256"]
    diff = {k for k in set(d) | set(e) if d.get(k) != e.get(k)}
    assert diff == {"sandbox_fix", "check_fn", "image_side", "strong_ask", "reopen_picture", "reopen_order", "cap", "_bps",
                    "_src_of"}
    assert d["reopen_order"] == "first_submit" and d["cap"] == 24576 and d["image_side"] == 896
    assert round(2180 * 0.96 / d["_bps"]) == 47


def test_base20_raw_text_detokenizes_only_an_empty_python_call():
    import time

    import httpx

    from arc2.trial import Transport
    asked = []

    def handler(request):
        asked.append(json.loads(request.content))
        return httpx.Response(200, json={"prompt": "<think>\nx\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n"
                                                   "print(1)\n</parameter>"})

    t = Transport.__new__(Transport)
    t.client, t.model, t.deadline = httpx.AsyncClient(base_url="http://x/v1/", transport=httpx.MockTransport(handler)), "m", time.monotonic() + 60
    empty = {"message": {"tool_calls": [{"id": "a", "function": {"name": "python", "arguments": "{}"}}]}, "token_ids": [1, 2]}
    full = {"message": {"tool_calls": [{"id": "a", "function": {"name": "python", "arguments": '{"code": "x=1"}'}}]},
            "token_ids": [1, 2]}
    raw = asyncio.run(t.raw_text(empty))
    assert raw == "\n\n<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>" and tasks.salvage_code(raw) == "print(1)"
    assert asked == [{"model": "m", "tokens": [1, 2]}]
    assert asyncio.run(t.raw_text(full)) is None and asyncio.run(t.raw_text({**empty, "token_ids": None})) is None
    assert len(asked) == 1


def test_restore_names_undoes_a_shadowed_submit_playbook_and_changed_train():
    """base15e240 cb2d8a2c#1: `def submit(answer): return answer` swallowed every later submit for 3,149 replies."""
    r = Repl(TASK, strict=False, call_seconds=5, memo=True, one_answer=True, restore_names=True)
    try:
        r.define("program_1", "def transform(grid):\n    return grid\n")
        out = r.run("def submit(answer):\n    return answer\ndef playbook(t=None):\n    return t\nprogram_1 = 5\n"
                    "train[0]['input'][0][0] = 9\ntest_inputs = []\nimport numpy as np")
        assert sorted(out["restored"]) == ["playbook", "program_1", "submit", "test_inputs", "train"]
        assert "[harness] `submit`" in out["stdout"] and "were restored after this call" in out["stdout"]
        got = r.run("print(train[0]['input'][0][0], len(test_inputs), program_1([[1]]), playbook('x'))")
        assert "1 1 [[1]] x" in got["stdout"] and "[playbook] saved" in got["stdout"] and "restored" not in got
        assert r.run("submit([[7], [8], [9]])")["submitted"] == [[[[7], [8], [9]]], None]  # recorded again
        assert "restored" not in r.run("x = train[0]['input']; x = 3")  # reading is fine
    finally:
        r.close()
    off = Repl(TASK, strict=False, call_seconds=5, one_answer=True)
    try:
        assert "restored" not in off.run("def submit(a):\n    return a") and "submitted" not in off.run("submit([[1]])")
    finally:
        off.close()


def test_a_model_that_redefines_submit_still_submits_and_stall_end_backstops():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call("def submit(answer):\n    return answer\nsubmit([[1]])"), call(f"submit({R1})")],
                            **{**ONE, "submit_final": True})
    eng.repl = lambda run: Repl(getattr(run, "shown", None) or run.task, strict=False, call_seconds=5, memo=True,
                                one_answer=True, restore_names=True)
    rows = _rows(eng)
    _turn(eng, run)
    assert run.submits == 1 and _picks(run) == [[[7], [8], [9]]]
    first = [r for r in rows if r.get("event", "reply") == "reply"][0]
    assert first["restored"] == ["submit"] and "[harness] `submit`" in first["tool_result"]
    _close(run)
    stuck = TaskRun("t#2", TASK, {"conversation": True})
    eng, sent = _one_engine([call("print(1)")] * 6, **{**ONE, "stall_end": 5, "loop_rule": False})
    rows = _rows(eng)
    asyncio.run(eng.advance(stuck, 10 ** 9, replies=6))
    assert stuck.done == "stalled" and len(sent) == 5 and [r for r in rows if r.get("event") == "stall_end"][0]["streak"] == 5
    _close(stuck)
    assert conv.ConvEngine(complete=None, count=None, execute=None, window=8, cap=4, settings={}).settings["stall_end"] == 200


def test_base21_is_base15e_plus_the_three_fixes():
    src = (Path(__file__).resolve().parents[1] / "scripts/arc2_weights/build_rtx_designs.py").read_text()
    ns = {}
    exec(compile(src.split("HEAD_FILES = ")[0], "designs", "exec"), {"__file__": "x"}, ns)
    d, e = ns["DESIGNS"]["base21"], ns["DESIGNS"]["base15e"]
    assert {k for k in set(d) | set(e) if d.get(k) != e.get(k)} == {"sandbox_fix", "check_fn", "restore_names", "stall_end"}
    assert d["stall_end"] == 0 and not conv.ConvEngine(complete=None, count=None, execute=None, window=8, cap=4,
                                                         settings={"stall_end": 0}).settings["stall_end"]


def test_base22_is_base15e256_plus_picture_order_sandbox_restore():
    src = (Path(__file__).resolve().parents[1] / "scripts/arc2_weights/build_rtx_designs.py").read_text()
    ns = {}
    exec(compile(src.split("HEAD_FILES = ")[0], "designs", "exec"), {"__file__": "x"}, ns)
    d, e = ns["DESIGNS"]["base22"], ns["DESIGNS"]["base15e256"]
    assert {k for k in set(d) | set(e) if d.get(k) != e.get(k)} == {
        "reopen_picture", "reopen_order", "sandbox_fix", "restore_names", "stall_end", "_src_of"}
    assert d["reopen_order"] == "first_submit" and "check_fn" not in d and "strong_ask" not in d and "cap" not in d


def test_v57240_is_base15e256_plus_the_four_v57_keys_on_c240_to_30m():
    src = (Path(__file__).resolve().parents[1] / "scripts/arc2_weights/build_rtx_designs.py").read_text()
    ns = {}
    exec(compile(src.split("HEAD_FILES = ")[0], "designs", "exec"), {"__file__": "x"}, ns)
    d, e = ns["DESIGNS"]["v57240"], ns["DESIGNS"]["base15e256"]
    assert {k for k in set(d) | set(e) if d.get(k) != e.get(k)} == {
        "reopen_picture", "sandbox_fix", "restore_names", "stall_end", "_src_of", "_copies", "_stop15", "_stop30"}
    assert d["reopen_order"] == "first_submit_asc" and d["_src_of"] == "arc2-rtx-c120-hx-base22-cpu32-v3b"
    assert "check_fn" not in d and d["_window"] == 262144


def test_base22_reopen_is_base15e_text_then_a_224_px_answer_picture():
    run = TaskRun("t#1", TASK, {"conversation": True})
    eng, sent = _one_engine([call(f"submit({R1})"), call(f"submit({R2})")],
                            **{**ONE, "submit_final": True, "reopen_picture": True, "images": True, "image_side": 224,
                               "image_changes": False, "sandbox_fix": True, "restore_names": True})
    _turn(eng, run)
    run.parked = False
    eng.pick_reopen([run])
    msg = run.messages[-1]["content"]
    assert msg[0] == {"type": "text", "text": tasks.CONV_REOPEN_WRONG} and msg[1]["text"] == tasks.ANSWER_PICTURE
    assert msg[2]["image_url"]["url"] == tasks.png_url(tasks.grid_pixels([TASK["test"][0]["input"], [[7], [8], [9]]], image_side=224))
    _turn(eng, run)
    assert run.done == "submitted" and run.submits == 2
    _close(run)
