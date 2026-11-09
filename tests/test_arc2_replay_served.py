import asyncio
import json

from arc2 import conv, served, tasks
from arc2.loop import TaskRun
from arc2.replay_served import main
from test_arc2_conv import RIGHT, TASK, WRONG, call, engine


SETTINGS = dict(submit_mode="fresh", fresh_each=True, fresh_text="naive", keep_first=True, memo=True, step_back=True)


def roundtrip(tmp_path, script, settings, mutate=None):
    """Play ``script`` live (served log on), replay its events.jsonl; both served logs' messages and the report."""
    n = len(script)
    live = served.ServedLog(tmp_path / "live", ids=False)
    eng, sent = engine(script, **settings)
    inner = eng.complete

    async def complete(task_id, messages, count, max_tokens, s):
        reply = await inner(task_id, messages, count, max_tokens, s)
        payload = tasks.chat_request("replay", messages, max_tokens, temperature=0.7, effort="xhigh",
                                     tools=[tasks.tool_spec(s)] if s.get("tool") else None, template=None)
        live.write(task_id, payload, {"choices": [{"message": reply["message"]}]}, 0, 0)
        return reply

    rows = []
    eng.complete, eng.record = complete, rows.append
    run = TaskRun("t#1", TASK, {"conversation": True})
    for _ in range(n):
        run.parked = False
        asyncio.run(eng.advance(run, 10 ** 9, replies=1))
        if run.awaiting_reopen:
            eng.pick_reopen([run])
    run.interp.close()
    if mutate is not None:
        rows = mutate(rows)
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = tmp_path / "data"
    data.mkdir()
    (data / "t.json").write_text(json.dumps(TASK))
    study = {"trials": [{"arms": [{"conversation": True}]}], "arc2": {"settings": {"tool": True, "images": False,
             "conversation": True, **settings}}, "windows": [65536], "completion_cap": 16384,
             "reasoning_effort": "xhigh", "temperature": 0.7}
    (tmp_path / "study.json").write_text(json.dumps(study))
    main([str(folder), str(tmp_path / "study.json"), str(data), str(tmp_path / "replay")])
    want = [p["messages"] for _, p in served.rebuild(tmp_path / "live")]
    got = [p["messages"] for _, p in served.rebuild(tmp_path / "replay")]
    report = json.loads((tmp_path / "replay" / "replay-report.json").read_text())["summary"]
    return want, got, report


# call ids are the server's and not recorded (the chat template does not render them)
def strip(ms, args=False):
    return [{k: v for k, v in m.items() if k not in ("tool_calls", "tool_call_id")} for m in ms] if not args else \
        [{**m, "tool_calls": [c["function"] for c in m["tool_calls"]]} if m.get("tool_calls") else
         {k: v for k, v in m.items() if k != "tool_call_id"} for m in ms]


def test_replay_rebuilds_the_served_messages_of_a_recorded_run(tmp_path):
    """A run recorded through events.jsonl only replays to the same served messages its live log wrote."""
    want, got, report = roundtrip(tmp_path, [call("x = 1\nprint(x)"), WRONG, call("submit([[1]], [[2]])"), RIGHT,
                                             call("submit([[7], [8], [9]])")], SETTINGS)
    assert len(got) == len(want) == 5 and [strip(m) for m in got] == [strip(m) for m in want]
    assert report["replayed"] == 5 and report["tool_mismatch"] == 0


def test_replay_keeps_a_long_bad_call_a_bad_call(tmp_path):
    """events cut bad_args at 2,000 characters: a long call the server parsed (no `code`) stays a bad call on
    replay (bad_call_fix: arguments {"code": ""} and the recorded error text), not a dropped malformed call."""
    bad = {"name": "python", "arguments": json.dumps({"function": "", 'code"': "x = 1\n" * 600})}
    want, got, report = roundtrip(tmp_path, [bad, call("print(2)"), call("submit([[7], [8], [9]])")],
                                  {**SETTINGS, "bad_call_fix": True, "drop_malformed": True})
    assert len(got) == len(want) == 3 and [strip(m, True) for m in got] == [strip(m, True) for m in want]
    assert report["mismatch"] == [{"task": "t#1", "reply": 1, "kind": "bad_call_forced"}]



def test_replay_legacy_swap_ends_every_reopen_pass_at_its_submit(tmp_path):
    """Runs before ce971a80a (convroms64: reopen_passes 0): every submit ended the pass and the task came back
    for another; replayed with legacy_swap the same served messages come back."""
    script = [call("submit([[1]], [[2]])"), call("submit([[7], [8], [9]])"), call("submit([[7], [8], [9]])")]
    want, got, report = roundtrip(tmp_path, script, {"submit_mode": "reopen", "reopen_passes": 0, "legacy_swap": True})
    assert len(got) == len(want) == 3 and [strip(m) for m in got] == [strip(m) for m in want]
    assert report["tool_mismatch"] == 0
    assert sum(m["content"] == "Submitted. The puzzle is finished." for m in got[-1] if m["role"] == "tool") == 2


def test_replay_rebuilds_a_bad_call_whose_arguments_were_not_recorded(tmp_path):
    """conv1's events kept no bad_args: the call is rebuilt (arguments unknown, so flagged) and the recorded error
    goes back as the tool result, not a "no call" nudge."""
    bad = {"name": "python", "arguments": json.dumps({"function": "x"})}
    drop = lambda rows: [{k: v for k, v in r.items() if k != "bad_args"} for r in rows]
    want, got, report = roundtrip(tmp_path, [bad, call("print(2)"), call("submit([[7], [8], [9]])")],
                                  {**SETTINGS, "bad_call_fix": False}, mutate=drop)
    assert len(got) == len(want) == 3 and [strip(m) for m in got] == [strip(m) for m in want]
    assert report["mismatch"] == [{"task": "t#1", "reply": 1, "kind": "bad_args_unrecorded"}]
