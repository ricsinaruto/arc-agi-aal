"""prime2: arc2.prime (stock Prime Agent sessions on the c240 roster)."""
import asyncio
import importlib.util
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from arc2 import prime  # noqa: E402

TASK = {"train": [{"input": [[1, 0]], "output": [[0, 1]]}],
        "test": [{"input": [[2, 0]], "output": [[0, 2]]}, {"input": [[3]], "output": [[3]]}]}


def test_parse_answer_and_right():
    good = {"answers": [[[[0, 2]], [[2, 0]]], [[3]]]}  # test 0: two candidates; test 1: one bare grid
    ans = prime.parse_answer(good, 2)
    assert ans == [[[[0, 2]], [[2, 0]]], [[[3]]]]
    assert prime.right_outputs(ans, TASK) == 2
    assert prime.parse_answer({"answers": [[[[0, 2]]]]}, 2) is None  # wrong length
    part = prime.parse_answer({"answers": [[[[0, 12]]], [[3]]]}, 2)  # invalid cell -> that output has none
    assert part == [None, [[[3]]]] and prime.right_outputs(part, TASK) == 1
    third = prime.parse_answer({"answers": [[[[9]], [[8]], [[0, 2]]], [[3]]]}, 2)  # only the first two count
    assert prime.right_outputs(third, TASK) == 1
    assert prime.parse_answer([], 2) is None


def test_arc_task_submit_roundtrip(tmp_path):
    shutil.copy(ROOT / "configs/prime2/player/workspace/arc_task.py", tmp_path / "arc_task.py")
    visible = {"train": TASK["train"], "test": [{"input": t["input"]} for t in TASK["test"]]}
    (tmp_path / "task.json").write_text(json.dumps(visible))
    spec = importlib.util.spec_from_file_location("arc_task_t", tmp_path / "arc_task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "output" not in json.dumps(mod.load()["test"])
    print(mod.submit([[[[0, 2]], [[2, 0]]], [[3]]]))
    ans = prime.parse_answer(json.loads((tmp_path / "answer.json").read_text()), 2)
    assert prime.right_outputs(ans, TASK) == 2
    with pytest.raises(ValueError):
        mod.submit([[[0, 2]]])  # one entry for two test inputs
    with pytest.raises(ValueError):
        mod.submit([[[1.5]], [[3]]])
    with pytest.raises(ValueError):
        mod.submit([[[[1]], [[2]], [[3]]], [[3]]])  # three candidates


def test_priority_gate_order_and_capacity():
    async def main():
        gate = prime.PriorityGate(capacity=10, slots=2)
        order = []

        async def req(name, key, tokens, hold):
            units = await gate.acquire(key, tokens)
            order.append(name)
            await asyncio.sleep(hold)
            gate.release(units)

        first = [asyncio.create_task(req("a", 0, 4, 0.05)), asyncio.create_task(req("b", 1, 4, 0.05))]
        await asyncio.sleep(0.01)
        later = [asyncio.create_task(req(n, k, 4, 0.01)) for n, k in (("low", 9), ("high", 2), ("mid", 5))]
        await asyncio.gather(*first, *later)
        assert order[:2] == ["a", "b"] and order[2:] == ["high", "mid", "low"]
        assert gate.peak_inflight == 2 and gate.inflight == 0 and gate.reserved == 0
        big = prime.PriorityGate(capacity=10, slots=5)  # capacity binds before slots
        u1 = await big.acquire(0, 6)
        waiter = asyncio.create_task(big.acquire(1, 6))
        await asyncio.sleep(0.01)
        assert not waiter.done()
        big.release(u1)
        await asyncio.wait_for(waiter, 1)
        with pytest.raises(ValueError):
            await big.acquire(0, 11)
        closed = prime.PriorityGate(capacity=10, slots=5)
        closed.close()
        blocked = asyncio.create_task(closed.acquire(0, 1))
        await asyncio.sleep(0.02)
        assert not blocked.done()
        blocked.cancel()
    asyncio.run(main())


def test_purpose_and_continuation():
    assert prime.purpose_of({"messages": [{"role": "system", "content": prime.SUMMARY_SYSTEM + " x"}]}) == "summary"
    assert prime.purpose_of({"messages": [{"role": "system", "content": prime.REFINE_SYSTEMS[1]}]}) == "refine"
    assert prime.purpose_of({"messages": [{"role": "system", "content": "You are Prime Agent"}]}) == "gameplay"
    p = {"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "x"},
                      {"role": "user", "content": [{"type": "text", "text": prime.CONTINUATION}]}]}
    assert prime.last_user_text(p) == prime.CONTINUATION
    assert prime.last_user_text({"messages": [{"role": "user", "content": "a"}, {"role": "assistant"},
                                              {"role": "tool", "content": "r"}]}) == ""


def test_translate_matches_base18_sampling():
    class C:
        pass
    ctx = C()
    ctx.cfg, ctx.temperature, ctx.served, ctx.cap, ctx.base_url = dict(prime.DEFAULTS), 0.7, "served", 16384, "http://x/v1"
    proxy = prime.Proxy(ctx)
    out = proxy.translate({"model": "flash", "stream": True, "reasoning_effort": "xhigh", "max_tokens": 32000,
                           "messages": []})
    assert out["chat_template_kwargs"] == {"reasoning_effort": "xhigh", "enable_thinking": True}
    assert (out["temperature"], out["top_p"], out["top_k"], out["max_tokens"], out["model"]) == (0.7, 0.95, 20, 16384, "served")
    assert out["stream_options"]["include_usage"] is True
    off = proxy.translate({"model": "flash", "stream": True, "reasoning_effort": "none", "messages": []})
    assert off["chat_template_kwargs"]["enable_thinking"] is False


def test_read_out_token_clock():
    class E:
        pass
    a, b = E(), E()
    a.task, b.task = TASK, TASK
    # a: one reply 0..10 s of 100 tokens; b: one reply 10..20 s of 100 tokens
    a.replies = [{"completion_tokens": 100, "t0": 0.0, "t1": 10.0}]
    b.replies = [{"completion_tokens": 100, "t0": 10.0, "t1": 20.0}]
    right = prime.parse_answer({"answers": [[[0, 2]], [[3]]]}, 2)
    a.answers = [{"t": 5.0, "answer": right}]  # at 50 tokens of the roster
    b.answers = [{"t": 15.0, "answer": right}, {"t": 19.0, "answer": prime.parse_answer({"answers": [[[9]], [[9]]]}, 2)}]
    out, clock = prime.read_out([a, b], {"m40": 40, "m60": 60, "m160": 160, "m200": 200}, step=100)
    m = out["marks"]
    assert (m["m40"]["right"], m["m60"]["right"], m["m160"]["right"], m["m200"]["right"]) == (0, 2, 4, 2)
    assert m["m200"]["entries_answered"] == 2 and out["tokens"] == pytest.approx(200)
    assert clock(5.0) == pytest.approx(50)


def test_settings_take_study_delays():
    cfg = prime.settings({"prime": {"sandbox": False}}, {"arc2": {"settings": {"confirm_delay": 7, "settled_delay": 9}}})
    assert (cfg["confirm_delay"], cfg["settled_delay"], cfg["sandbox"]) == (7, 9, False)
    with pytest.raises(ValueError):
        prime.settings({"prime": {"nope": 1}}, {})


def test_trial_dispatch(monkeypatch):
    from arc2 import trial as arc2_trial

    seen = {}

    async def fake(study, trial, **kw):
        seen.update(kw)
        return {"status": "complete"}
    monkeypatch.setattr(prime, "run_prime", fake)
    out = asyncio.run(arc2_trial.run_arc2({}, {"prime": {}}, source="s", output="o", server_dir="d", base_url="u",
                                          deadline_epoch=1.0))
    assert out["status"] == "complete" and seen["base_url"] == "u"


# --- end to end: the real v0.3.3 binary against the scripted server (needs PRIME2_RUNTIME: a prepared runtime) -------

RUNTIME = os.environ.get("PRIME2_RUNTIME")
V3 = {"conv": False, "images": False, "hot_limit": 64, "quantum_tokens": 40000, "release_on_compaction": False}


def _e2e(tmp_path, monkeypatch, *, fake_kw=None, prime_kw=None, tokens=12000, active=3, kill_after=None,
         kill_in_tool=False):
    import signal
    import threading

    import prime2_fake_model as fm

    monkeypatch.setattr(prime, "RUNTIME", Path(RUNTIME))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log = open(os.environ["PRIME2_FAKE_LOG"], "a") if os.environ.get("PRIME2_FAKE_LOG") else None
    fake = fm.serve_in_thread(port, log=log, **(fake_kw or {}))
    source, server, out = tmp_path / "bundle", tmp_path / "server", tmp_path / "out"
    data = source / "data/arc2/evaluation"
    data.mkdir(parents=True)
    ids = ["t1", "t2"]
    for i in ids:
        (data / f"{i}.json").write_text(json.dumps(TASK))
    shutil.copytree(ROOT / "configs/prime2/player", source / "prime/player")
    server.mkdir()
    layout = {"schema": 1, "num_blocks": 2000, "max_model_len": 262144, "lookahead": 0,
              "costs": [{"tokens": 32768 * k, "blocks": 10 * k + 2, "groups": [10 * k, 2]} for k in range(1, 9)]}
    (server / "server.log").write_text(f"ARC_CACHE_LAYOUT {json.dumps(layout)}\n")
    study = {"arc2": {"data": "data/arc2", "settings": {"confirm_delay": 20000, "settled_delay": 100000}},
             "temperature": 0.7, "completion_cap": 16384, "windows": [131072], "kv_margin": 0.96,
             "active_limit": active}
    trial = {"name": "c240", "kind": "arc2", "split": "evaluation", "tasks": ids, "copies": 2, "wall_seconds": 600,
             "marks": {"small": tokens // 2, "all": tokens},
             "prime": {"sandbox": False, "idle_seconds": 2, "start_stagger_seconds": 0.5, "answer_poll_seconds": 0.5,
                       **(prime_kw or {})}}
    monkeypatch.delenv("ARENA_STOP_REQUEST", raising=False)
    monkeypatch.setenv("ARC_REAL_NVIDIA_SMI", "true")
    killed = {}
    if kill_after:
        def killer():
            while True:
                ev = out / "events.jsonl"
                if ev.exists() and len(ev.read_text().splitlines()) >= kill_after:
                    for logf in sorted((out / "sessions").glob("*/log.jsonl")):
                        rows = [json.loads(x) for x in logf.read_text().splitlines()]
                        pid = next((r["pid"] for r in rows if r["event"] == "session_start"), None)
                        if kill_in_tool:  # only a session inside a tool cell (its last rpc event)
                            rpc = logf.parent / "rpc-events.jsonl"
                            last = rpc.read_text().splitlines()[-1:] if rpc.exists() else []
                            if not last or json.loads(last[0]).get("type") != "tool_execution_start":
                                continue
                        if pid:
                            os.killpg(pid, signal.SIGKILL)
                            killed.update(entry=logf.parent.name)
                            return
                time.sleep(0.1)
        threading.Thread(target=killer, daemon=True).start()
    began = time.time()
    report = asyncio.run(prime.run_prime(study, trial, source=source, output=out, server_dir=server,
                                         base_url=f"http://127.0.0.1:{port}/v1", deadline_epoch=time.time() + 900))
    rows = [json.loads(x) for x in (out / "tasks.jsonl").read_text().splitlines()]
    print(json.dumps({k: report.get(k) for k in ("status", "end_reason", "completion_tokens", "replies", "gate",
                                                 "task_errors")}, indent=1)[:3000], time.time() - began)
    return report, rows, fake, killed


@pytest.mark.skipif(not RUNTIME, reason="PRIME2_RUNTIME (node, prime-agent build, kernel venv) not set")
def test_e2e_real_binary(tmp_path, monkeypatch):
    report, rows, fake, _ = _e2e(tmp_path, monkeypatch, prime_kw=V3)
    kinds = {r["kind"] for r in fake.requests}
    assert report["status"] == "complete" and report["end_reason"] == "last_mark"
    assert len(rows) == 4 and all(r["answers"] for r in rows)
    assert report["roster"]["marks"]["all"]["right"] == 4  # test 1 is the identity: right in every entry
    assert {"gameplay", "child", "summary"} <= kinds, kinds
    totals = report["prime"]
    assert totals["completion_tokens"].get("summary", 0) > 0 and totals["rpc_events"].get("compaction_end", 0) >= 4
    assert report["server_generated_tokens"] == report["completion_tokens"]  # generated = charged
    assert all(r["kwargs"]["reasoning_effort"] in ("xhigh", "none") for r in fake.requests)
    assert all((r["temperature"], r["top_p"], r["top_k"], r["model"]) == (0.7, 0.95, 20, "flash-served")
               for r in fake.requests)
    assert sum(r["settled"] for r in rows) == 4


@pytest.mark.skipif(not RUNTIME, reason="PRIME2_RUNTIME (node, prime-agent build, kernel venv) not set")
@pytest.mark.parametrize("kind", ["500", "drop"])
def test_e2e_faults_hot_set(tmp_path, monkeypatch, kind):
    """Run 2's failure classes through the real binary: failing requests (vLLM's EngineDead 500, dropped
    connections) and a killed session, with a hot set smaller than the roster; the prefill limit holds."""
    report, rows, fake, killed = _e2e(
        tmp_path, monkeypatch, fake_kw={"fail_every": 7, "fail_kind": kind, "first_delay": 0.05},
        prime_kw={**V3, "hot_limit": 2, "quantum_tokens": 300, "hot_idle_seconds": 5, "prefill_slots": 2},
        tokens=6000, active=3, kill_after=40)
    gate = report["gate"]
    assert report["end_reason"] == "last_mark" and all(r["answers"] for r in rows), report["end_reason"]
    assert fake.failed > 5 and gate["dropped"] >= fake.failed // 2, (fake.failed, gate)
    assert fake.prefill_max <= 2
    assert killed and next(r for r in rows if r["task"].replace("#", "_") == killed["entry"])["restarts"] == 1
    assert gate["cold_admitted"] > 4  # entries rotated through the hot set


@pytest.mark.skipif(not RUNTIME, reason="PRIME2_RUNTIME (node, prime-agent build, kernel venv) not set")
def test_e2e_dead_server_ends_trial(tmp_path, monkeypatch):
    report, rows, fake, _ = _e2e(tmp_path, monkeypatch, fake_kw={"down_at": 30},
                                 prime_kw={**V3, "server_dead_seconds": 8}, tokens=10 ** 9)
    assert report["end_reason"] == "server_dead" and report["status"] == "error"
    assert report["play_seconds"] < 120


def _cache_sim(hot_limit, *, entries=240, slots=49, lru=100, replies=3000, quantum=2000):
    """Run 1's failure on a model server whose cache keeps the ``lru`` most recently served conversations (run 1:
    a turn hit its whole conversation when it came back within ~60 s, i.e. ~100 conversations at 49 slots): each
    entry alternates a 100-token reply and a short tool call. Returns the share of replies that found their
    conversation cached."""
    import collections
    import random

    async def main():
        gate = prime.PriorityGate(10 ** 9, slots, hot_limit=hot_limit, quantum=quantum, hot_idle=5.0)
        cache, stats, spent = collections.OrderedDict(), [0, 0], {}
        rng = random.Random(0)

        class E:
            streaming = 0

        async def run(e):
            spent[e] = 0
            while sum(stats) < replies:
                units = await gate.acquire(spent[e], 1000, entry=e)
                hit = e in cache
                stats[0 if hit else 1] += 1
                cache.pop(e, None)
                e.streaming = 1
                await asyncio.sleep(rng.uniform(0.002, 0.006))
                e.streaming = 0
                cache[e] = True
                while len(cache) > lru:
                    cache.popitem(last=False)
                spent[e] += 100
                gate.charged(e, 100)
                gate.release(units)
                await asyncio.sleep(rng.uniform(0.0, 0.002))
        jobs = [asyncio.create_task(run(E())) for _ in range(entries)]
        while sum(stats) < replies:
            await asyncio.sleep(0.01)
        gate.close()
        for j in jobs:
            j.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        return stats[0] / sum(stats), spent
    return asyncio.run(main())


def test_hot_set_keeps_conversations_cached():
    plain, _ = _cache_sim(0)
    hot, spent = _cache_sim(64)
    assert plain < 0.2 and hot > 0.85, (plain, hot)
    played = sorted(spent.values())
    assert played[len(played) // 2] > 0  # entries rotate through the hot set (quantum), the median entry played


def test_hot_set_quantum_idle_drop_and_no_idle_gpu():
    async def main():
        t = [0.0]
        gate = prime.PriorityGate(100, 2, hot_limit=2, quantum=50, hot_idle=10, now=lambda: t[0])

        class E:
            streaming = 0
        a, b, c = E(), E(), E()
        uc = await gate.acquire(0, 1, entry=c)  # c enters and stays in flight: the GPU is busy
        c.streaming = 1
        u = await gate.acquire(0, 1, entry=a)  # a enters the hot set (now full)
        wb = asyncio.create_task(gate.acquire(0, 1, entry=b))  # cold, hot set full
        await asyncio.sleep(0)
        gate.charged(a, 10)
        gate.release(u)
        await asyncio.sleep(0)
        assert not wb.done() and a in gate.hot  # a slot is free but b waits: a is still hot (tool call)
        wa = asyncio.create_task(gate.acquire(5, 1, entry=a))
        await asyncio.sleep(0)
        assert wa.done()  # a's next turn goes first
        gate.charged(a, 45)  # quantum used: a leaves, b enters at the next release
        gate.release(wa.result())
        await asyncio.sleep(0)
        assert wb.done() and b in gate.hot and a not in gate.hot
        gate.release(wb.result())
        t[0] = 20.0  # b idle past hot_idle: dropped at the next pump
        gate.pump()
        assert b not in gate.hot
        # error / exit path: drop at once
        wa2 = asyncio.create_task(gate.acquire(0, 1, entry=a))
        await asyncio.sleep(0)
        assert a in gate.hot
        gate.drop(a)
        assert a not in gate.hot and gate.dropped == 1
        gate.release(wa2.result())
        # a full hot set of quiet entries never holds the GPU idle while cold requests wait
        c.streaming = 0
        gate.release(uc)
        gate.hot = {a: {"spent": 0, "last": 19.0}, c: {"spent": 0, "last": 18.0}}
        wb2 = asyncio.create_task(gate.acquire(0, 1, entry=b))
        await asyncio.sleep(0)
        assert wb2.done() and c not in gate.hot and gate.idle_evictions == 1
    asyncio.run(main())


# --- v4: conv1's prompts, kernel globals, answer workflow and closing submit ------------------------------------------

@pytest.mark.skipif(not RUNTIME, reason="PRIME2_RUNTIME (node, prime-agent build, kernel venv) not set")
def test_e2e_conv_submit_closes_and_next_starts(tmp_path, monkeypatch):
    """The real binary through conv1's workflow: conv1's system prompt and first message (text + one picture per
    pair and test input), the kernel's globals and call limit, a stop without a program ("neither"), a program checked
    by the harness and callable as program_1, a compaction that gives the hot place up, and a submit that closes the
    session so the next entry plays (one slot, one hot place)."""
    report, rows, fake, _ = _e2e(tmp_path, monkeypatch, fake_kw={"conv": True},
                                 prime_kw={"hot_limit": 1, "call_seconds": 2, "reply_delay_seconds": 0.5},
                                 tokens=10 ** 9, active=1)
    assert report["end_reason"] == "sessions_ended" and report["status"] == "complete", report["end_reason"]
    assert [r["status"] for r in rows] == ["submitted"] * 4
    assert report["prime"]["entries_submitted"] == 4 and report["sessions_live"] == 0
    # conv1's ranking: the submit first, the verified program's reading second -> both outputs right everywhere
    assert report["final"]["right"] == 8
    assert all(r["harness_replies"].get("nothing", 0) >= 1 and r["harness_replies"].get("program") == 1 for r in rows)
    assert all(r["verified_programs"] == 1 and r["compaction_releases"] >= 1 for r in rows)
    assert report["gate"]["compaction_releases"] >= 1  # a compaction while hot gives the place up
    assert len(fake.first_requests) == 4
    assert all(system.startswith(prime.CONV_SYSTEM[:200]) and images == 3 for system, images in fake.first_requests)
    out = tmp_path / "out"
    for r in rows:
        results = []
        for f in (out / "sessions" / r["task"].replace("#", "_")).glob("session/home/.prime/agent/sessions/*.jsonl"):
            for line in f.read_text().splitlines():
                m = json.loads(line).get("message") or {}
                if m.get("role") == "toolResult":
                    results.append(" ".join(p.get("text", "") for p in m.get("content", []) if isinstance(p, dict)))
        text = "\n".join(results)
        assert "STEP0 1 2 numpy" in text and "TimeoutError" in text and "not reached\n" not in text, text[:2000]
        assert "STEP6 [[0, 2]]" in text and "submitted" in text
    closes = sorted(json.loads(x)["t"] for f in (out / "sessions").glob("*/log.jsonl")
                    for x in f.read_text().splitlines() if json.loads(x)["event"] == "closed")
    assert len(closes) == 4
    # one hot place: the last entry to start played its first reply after an earlier entry had closed
    first_reply = {}
    for line in (out / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        first_reply.setdefault(row["task"], row["t1"])
    started = max(first_reply.values())  # monotonic, like the closes' wall clock only in order: compare by order
    order = sorted(first_reply, key=first_reply.get)
    assert len(order) == 4 and report["gate"]["cold_admitted"] >= 4


def test_conv_prompts_are_conv1s():
    from arc2 import tasks

    assert prime.CONV_SYSTEM == tasks.CONV_SYSTEM.replace("`python`", "`ipython`")
    assert "```python block" in prime.CONV_SYSTEM and "`python`" not in prime.CONV_SYSTEM
    assert prime.CONV_NOTHING.startswith("Your reply had neither a `ipython` call nor a program.")


def test_hold_release_on_compaction_and_drop_on_submit():
    async def main():
        gate = prime.PriorityGate(100, 4, hot_limit=1, quantum=0, hot_idle=10 ** 6)

        class E:
            streaming = 0
        a, b, c = E(), E(), E()
        busy = await gate.acquire(0, 1)  # another request in flight all along: the no-idle-GPU rule stays out
        ua = await gate.acquire(0, 1, entry=a)
        a.streaming = 1
        wb = asyncio.create_task(gate.acquire(0, 1, entry=b))
        wc = asyncio.create_task(gate.acquire(500, 1, entry=c))
        await asyncio.sleep(0)
        gate.charged(a, 10 ** 6)  # quantum 0: a holds its place however much it plays
        a.streaming = 0
        gate.release(ua)
        await asyncio.sleep(0)
        assert a in gate.hot and not wb.done()
        ua = await gate.acquire(10 ** 6, 1, entry=a)
        gate.release(ua)
        gate.release_hot(a)  # a compacted: the cold entry with the fewest tokens (b) takes the place
        await asyncio.sleep(0)
        assert wb.done() and b in gate.hot and not wc.done() and gate.compaction_releases == 1
        gate.release(wb.result())
        gate.drop(b)  # b submitted: closed, its place goes to c
        await asyncio.sleep(0)
        assert wc.done() and c in gate.hot
        gate.release(wc.result())
        gate.release(busy)
    asyncio.run(main())


def test_read_out_conv_candidates():
    class E:
        conv = True
        task = TASK
        current_answer = prime.Entry.current_answer
        answers = []
    a = E()
    a.replies = [{"completion_tokens": 100, "t0": 0.0, "t1": 10.0}]
    a.candidates = [  # a verified program at 30 tokens (both outputs right), a submit at 80 (test 0 wrong)
        {"code": "p1", "tests": [[[0, 2]], [[3]]], "verified": True, "passed": 1, "accuracy": 1.0, "tokens_at": 0,
         "t": 3.0, "program": 1},
        {"code": "submit-1", "tests": [[[2, 0]], [[3]]], "verified": False, "passed": 0, "accuracy": 0.0,
         "tokens_at": 0, "t": 8.0, "submitted": 1}]
    out, _ = prime.read_out([a], {"m20": 20, "m50": 50, "m100": 100})
    m = out["marks"]
    assert (m["m20"]["right"], m["m50"]["right"], m["m100"]["right"]) == (1, 2, 2)  # m20: fallback guesses only
    assert (m["m50"]["entries_answered"], m["m100"]["entries_answered"]) == (0, 1)


def test_check_program_and_ledger(tmp_path):
    from arc2 import sandbox

    class Ctx:
        cfg = dict(prime.DEFAULTS, sandbox=False)
        output = tmp_path

        async def execute(self, code, inputs):
            return await asyncio.to_thread(sandbox.run, code, inputs, strict=False)
    e = prime.Entry(Ctx(), "t1#1", TASK, "prime")
    (e.dir / "session/home/.arc/programs").mkdir(parents=True)
    good = "def transform(grid):\n    return [row[::-1] for row in grid]"
    text = asyncio.run(e.check_program(good))
    assert text.startswith("Program 1 passes every training pair (`program_1` in python).\nIts test output is a new "
                           "reading:\n0 2\n\n3") and text.endswith("- reading A: program 1")
    assert (e.dir / "session/home/.arc/programs/program_1.py").read_text() == good
    again = asyncio.run(e.check_program(good + "\n"))
    assert "Its test output is the same as program 1's." in again and "reading A: programs 1, 2" in again
    bad = asyncio.run(e.check_program("def transform(grid):\n    return grid"))
    assert bad.startswith("Program 3: Your program is correct on 0 of 1 training pairs.")
    assert "Fix the rule and send the corrected program, or keep exploring in python." in bad
    assert prime.right_outputs(e.current_answer(), TASK) == 2  # two verified programs, one reading


@pytest.mark.skipif(not RUNTIME, reason="PRIME2_RUNTIME (node, prime-agent build, kernel venv) not set")
def test_e2e_conv_faults_and_killed_session(tmp_path, monkeypatch):
    """Preflight v4 C/D: a session killed mid-tool resumes (--continue) and gets the harness's replies again; with
    failing requests and a small hot set every entry still submits; the prefill limit holds."""
    report, rows, fake, killed = _e2e(
        tmp_path, monkeypatch, fake_kw={"conv": True, "fail_every": 7, "first_delay": 0.05},
        prime_kw={"hot_limit": 2, "hot_idle_seconds": 5, "prefill_slots": 2, "call_seconds": 2,
                  "reply_delay_seconds": 0.5}, tokens=10 ** 9, active=3, kill_after=4, kill_in_tool=True)
    assert report["end_reason"] == "sessions_ended", report["end_reason"]
    assert [r["status"] for r in rows] == ["submitted"] * 4
    assert killed and next(r for r in rows if r["task"].replace("#", "_") == killed["entry"])["restarts"] == 1
    assert fake.failed > 3 and fake.prefill_max <= 2


def test_cut_replies_restored_in_place_until_compaction():
    def sse(*deltas, finish="length"):
        rows = [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
        rows.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        return b"".join(b"data: " + json.dumps(r).encode() + b"\n\n" for r in rows)
    cut = prime.CutReplies()
    sys_m, u1 = {"role": "system", "content": "S"}, {"role": "user", "content": "task"}
    sent = [sys_m, u1]
    assert cut.keep(sent, sse({"reasoning_content": "think A"}, {"reasoning_content": " more"}))
    assert not cut.keep(sent, sse({"reasoning_content": "x"}, {"content": "text"}))  # has text: Pi keeps it
    assert not cut.keep(sent, sse({"reasoning_content": "x"}, {"tool_calls": [{"index": 0}]}))
    cut_msg = {"role": "user", "content": "length limit"}
    nxt = [sys_m, u1, cut_msg]  # what Pi sends next: the cut reply is gone
    assert cut.restore(nxt) == 1
    assert nxt[2] == {"role": "assistant", "content": "", "reasoning_content": "think A more"} and nxt[3] == cut_msg
    assert cut.keep(nxt, sse({"reasoning": "think B"}))  # a second cut, after the first one's user message
    later = [sys_m, u1, cut_msg, cut_msg, {"role": "assistant", "content": "", "tool_calls": []}]
    assert cut.restore(later) == 2 and [m.get("reasoning_content") or m.get("reasoning") for m in later[2:6:2]] == [
        "think A more", "think B"]
    assert cut.restore(later) == 0  # already in place
    compacted = [sys_m, {"role": "user", "content": "summary"}, cut_msg]
    assert cut.restore(compacted) == 0 and cut.kept == []  # the anchors are gone


@pytest.mark.skipif(not RUNTIME, reason="PRIME2_RUNTIME (node, prime-agent build, kernel venv) not set")
def test_e2e_conv_cut_reply_stays_in_context(tmp_path, monkeypatch):
    """v5: a reply cut at the cap (reasoning only) is back in the next request, so the scripted session moves on
    (without it the fake serves the cut again and the entry never submits)."""
    report, rows, fake, _ = _e2e(tmp_path, monkeypatch, fake_kw={"conv": True, "cut": True},
                                 prime_kw={"call_seconds": 2, "reply_delay_seconds": 0.5}, tokens=10 ** 9, active=3)
    assert report["end_reason"] == "sessions_ended", report["end_reason"]
    assert [r["status"] for r in rows] == ["submitted"] * 4 and fake.cuts == 4
    assert all(r["cut_kept"] == 1 and r["cut_restored"] >= 1 and r["harness_replies"].get("cut") == 1 for r in rows)
