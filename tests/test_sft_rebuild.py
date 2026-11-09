"""scripts/sft/rebuild.py pure parts: the log's script, rollout rows, the request compare and the logged clock."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sft_rebuild", ROOT / "scripts/sft/rebuild.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)


def msg(role, text):
    return {"role": role, "content": text}


def test_script_of_splits_purposes():
    rows = [{"kind": "observation", "step": 0}, {"kind": "model", "purpose": "gameplay", "message": {"a": 1},
                                                 "finish_reason": "tool_calls", "usage": {"completion_tokens": 3}},
            {"kind": "model", "purpose": "summary", "message": {"b": 2}, "finish_reason": "stop", "usage": {}},
            {"kind": "model", "message": {"c": 3}, "finish_reason": "stop"}]
    game, other = rb.script_of(rows)
    assert [g["message"] for g in game] == [{"a": 1}, {"c": 3}] and [o["message"] for o in other] == [{"b": 2}]
    assert game[1]["usage"] == {}


def test_rollout_rows_round_trip():
    tools = [{"type": "function", "function": {"name": "python"}}]
    caps = [{"payload": {"messages": [msg("system", "s"), msg("user", "u1")], "tools": tools,
                         "chat_template_kwargs": {"x": 1}}, "reply": {"message": msg("assistant", "a1")},
             "gameplay": True, "step": 0},
            {"payload": {"messages": [msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"), msg("tool", "t")],
                         "tools": tools, "chat_template_kwargs": {"x": 1}}, "reply": {"message": msg("assistant", "a2")},
             "gameplay": True, "step": 1},
            {"payload": {"messages": [msg("system", "s"), msg("user", "drained")], "tools": None,
                         "chat_template_kwargs": {}}, "reply": {"message": msg("assistant", "sum")},
             "gameplay": False, "step": 1}]
    rows = rb.rollout_rows(caps)
    assert [r["keep"] for r in rows] == [0, 2, 1] and "tools" in rows[0] and "tools" not in rows[1] and "tools" in rows[2]
    back = rb.requests_of(rows)
    assert [b["messages"] for b in back] == [c["payload"]["messages"] for c in caps]
    assert back[1]["tools"] == tools and back[2]["tools"] is None
    # two replays: kept up to the first request where they differ, and up to each one's cut
    other = rb.rollout_rows(caps[:1] + [{**caps[1], "payload": {**caps[1]["payload"], "messages": [
        msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"), msg("tool", "t'")]}}] + caps[2:])
    assert rb.first_difference(rows, other) == 1 and rb.first_difference(rows, rows) == 3
    base = {"kept": 3, "cut": None, "whole": True, "game_replies": 2}
    out, c = rb.settle([(rows, base), (other, base)])
    assert len(out) == 1 and c["cut"] == "replicas_differ" and c["ok"] and not c["whole"]
    out, c = rb.settle([(rows, base), (rows, {**base, "kept": 2, "cut": "log:observation", "whole": False})])
    assert len(out) == 2 and c["cut"] == "log:observation" and c["ok"]
    out, c = rb.settle([(rows, base), (rows, base)])
    assert len(out) == 3 and c["whole"] and c["ok"] and c["cut"] is None


def test_compare_classes():
    def rows(*requests):
        return rb.rollout_rows([{"payload": {"messages": m, "tools": None, "chat_template_kwargs": {}},
                                 "reply": {"message": {}}, "gameplay": True, "step": i} for i, m in enumerate(requests)])

    base = [msg("system", "s"), msg("user", "Stuck check: 3 turns (12s of active play), go")]
    other = [msg("system", "s"), msg("user", "Stuck check: 3 turns (13s of active play), go")]
    tool_a = base + [msg("tool", "out 1")]
    tool_b = base + [msg("tool", "out 2")]
    got = rb.compare(rows(base, tool_a), rows(base, tool_b))
    assert got["requests"] == 2 and got["exact"] == 1 and [m["class"] for m in got["mismatches"]] == ["tool_output"]
    got = rb.compare(rows(other), rows(base))
    assert got["mismatches"][0]["class"] == "stuck_seconds" and got["mismatches"][0]["message"] == 1
    got = rb.compare(rows(base), rows(base, tool_a))
    assert got["missing"] == 1 and got["exact"] == 1


def test_logged_clock():
    rows = [{"kind": "observation", "step": 0, "time": 0.0}, {"kind": "model_request", "time": 10.0,
                                                              "runtime": {"elapsed_seconds": 1.0}},
            {"kind": "observation", "step": 1, "time": 15.0}, {"kind": "model_request", "time": 100.0,
                                                               "runtime": {"elapsed_seconds": 20.0}}]
    requests, observed = rb.timeline(rows)
    assert requests == [(10.0, 1.0), (100.0, 20.0)] and observed == {0: 0.0, 1: 15.0}
    assert rb.active_at(requests, 15.0) == pytest.approx(6.0)
    assert rb.active_at(requests, 90.0) == pytest.approx(20.0)  # parked: no more than the next request's
    assert rb.active_at(requests, 105.0) == pytest.approx(25.0)


def test_log_diff_ignores_times():
    a = [{"kind": "observation", "time": 1.0, "step": 1, "levels": 0}, {"kind": "duck_python", "result": {"x": 1}}]
    b = [{"kind": "observation", "time": 9.0, "step": 1, "levels": 0}, {"kind": "duck_python", "result": {"x": 2}}]
    d = rb.log_diff(a, b)
    assert d["observation"]["first_diff"] is None and d["duck_python"]["first_diff"] == 0
    assert d["duck_python"]["keys"] == ["result"]


def test_unsafe_code_and_hash_order():
    assert rb.CLOCK_CODE.search("t0 = time.time()") and rb.CLOCK_CODE.search("perf_counter ()")
    assert not rb.CLOCK_CODE.search("timeline(x)") and rb.RANDOM_CODE.search("random.choice(xs)")
    values = [["a", "set", 2, "{'W', 'N'}"], ["b", "set", 3, "{(1, 2), (3, 4), -5}"], ["c", "set", 1, "{'x'}"],
              ["d", "frozenset", 2, "frozenset({1, 2})"], ["e", "list", 2, "['p', 'q']"]]
    assert rb.hash_ordered({"values": values}) == ["a"] and rb.hash_ordered(None) == []
    assert rb.hash_risky({"values": values}) == ["a", "e"]
    assert rb.SET_CODE.search("seen = set()") and rb.SET_CODE.search("{c for c in row}")
    assert rb.SET_CODE.search("s = {'a', 'b'}") and not rb.SET_CODE.search("d = {'a': 1}")
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x\n" + rb.OBSERVERS + " (step 3): o: {'W': 1, 'b': 2}"}]},
            {"role": "user", "content": "nothing"}]
    lines = rb.observer_lines(msgs)
    assert len(lines) == 1 and rb.strings_ordered(lines[0]) and not rb.strings_ordered("o: [(1, 2), (3, 4)]")
