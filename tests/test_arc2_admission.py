"""arc2.admission.CacheGate: admission against the server's real KV use, reserving gate when unread."""

import asyncio
import json

import pytest

from arc2.admission import CacheGate, settings


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def gate(clock, **kw):
    # reserving rule: capacity 30 units = one request of 30; the cache has 100 blocks
    return CacheGate(30, slots=8, blocks=100, now=clock, **{"target": 0.8, "max_age": 15.0, "settle": 15.0, **kw})


def test_unread_metrics_fall_back_to_the_reserving_rule():
    async def go():
        clock = Clock()
        g = gate(clock)
        first = await g.acquire("a", 30)  # budget None: units = tokens
        waiter = asyncio.ensure_future(g.acquire("b", 30))
        await asyncio.sleep(0)
        assert not waiter.done() and g.stats["reserve_admitted"] == 1  # 30 + 30 > 30: waits, as KVGate
        g.observe(None)  # /metrics unreadable: still the reserving rule
        await asyncio.sleep(0)
        assert not waiter.done()
        g.release(first)
        await waiter
    asyncio.run(go())


def test_fresh_reading_admits_on_real_kv_until_the_target():
    async def go():
        clock = Clock()
        g = gate(clock)
        g.observe({"kv_cache_fraction": 0.1, "preemptions": 0})
        held = [await g.acquire(x, 30) for x in "ab"]  # two where the reserving rule takes one
        assert len(held) == 2 and g.reserved == 60 > g.capacity and g.stats["kv_admitted"] == 2
        assert g.estimate() == pytest.approx(0.7)
        third = asyncio.ensure_future(g.acquire("c", 30))  # 0.7 + 0.3 > 0.8
        await asyncio.sleep(0)
        assert not third.done()
        third.cancel()
    asyncio.run(go())


def test_target_holds_back_and_a_new_reading_pumps_the_queue():
    async def go():
        clock = Clock()
        g = gate(clock)
        g.observe({"kv_cache_fraction": 0.5, "preemptions": 0})
        await g.acquire("a", 25)  # 0.5 + 0.25 <= 0.8
        waiter = asyncio.ensure_future(g.acquire("b", 25))  # 0.75 + 0.25 > 0.8
        await asyncio.sleep(0)
        assert not waiter.done()
        clock.t = 5.0
        g.observe({"kv_cache_fraction": 0.5, "preemptions": 0})  # "a" still pending (settle 15 s)
        await asyncio.sleep(0)
        assert not waiter.done()
        clock.t = 16.0
        g.observe({"kv_cache_fraction": 0.52, "preemptions": 0})  # "a" now in the reading; 0.52 + 0.25 fits
        await asyncio.sleep(0)
        assert waiter.done()
    asyncio.run(go())


def test_stale_reading_returns_to_the_reserving_rule():
    async def go():
        clock = Clock()
        g = gate(clock)
        g.observe({"kv_cache_fraction": 0.1, "preemptions": 0})
        await g.acquire("a", 30)
        clock.t = 16.0  # past max_age with no reading
        assert not g.fresh() and not g._fits(10)  # reserved 30 of 30
        assert g.snapshot()["fresh"] is False
    asyncio.run(go())


def test_preemption_lowers_the_target_to_the_floor():
    clock = Clock()
    g = gate(clock, floor=0.7)
    g.observe({"kv_cache_fraction": 0.3, "preemptions": 4})  # the first reading sets the base
    assert g.target == 0.8
    g.observe({"kv_cache_fraction": 0.3, "preemptions": 5})
    assert g.target == 0.75 and g.stats["target_cuts"] == 1
    g.observe({"kv_cache_fraction": 0.3, "preemptions": 9})
    g.observe({"kv_cache_fraction": 0.3, "preemptions": 12})
    assert g.target == 0.7 and g.stats["target_cuts"] == 2


def test_held_growth_counts_the_whole_next_turn_in_cache_mode():
    clock = Clock()
    g = gate(clock)
    g.observe({"kv_cache_fraction": 0.6, "preemptions": 0})
    assert g.grow_fits(20, 18) and not g.grow_fits(21, 20)  # 0.6 + 0.2 fits; 0.6 + 0.21 does not
    clock.t = 20.0
    assert g.grow_fits(30, 0) is True and g.grow_fits(31, 0) is False  # stale: reserved 0 + 30 <= 30


def test_settings_validate():
    assert settings({"mode": "kv"})["target"] == 0.8
    with pytest.raises(ValueError):
        settings({"mode": "kv", "targett": 0.7})
    with pytest.raises(ValueError):
        settings({"target": 0.4, "floor": 0.5})


def _cache_roster(tmp_path, monkeypatch, **env):
    import threading
    import time

    from test_arc2 import _scripted_roster

    from arc2.trial import run_arc2

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.2")
    study.update(generation_slots=4, active_limit=4)
    # reserving rule: one request at a time (one full 8,192-token window, 4 blocks); cache rule: 100 blocks
    study["arc2"].update(gate_match={"requests": 1, "tokens": 8192}, hold_attempt_blocks=True,
                         admission={"mode": "kv", "target": 0.8, "poll": 0.1, "max_age": 1.0, "settle": 0.3})
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False}]
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    out = tmp_path / "out"

    def stopper():
        for _ in range(1200):
            events = out / "events.jsonl"
            if events.exists() and len(events.read_text().splitlines()) >= 24:
                (out / "stop-request").write_text("{}")
                return
            time.sleep(0.05)

    threading.Thread(target=stopper, daemon=True).start()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 300))
    rows = [json.loads(line) for line in (out / "admission.jsonl").read_text().splitlines()]
    return report, rows


def test_run_arc2_cache_admission_runs_past_the_reserving_capacity(tmp_path, monkeypatch):
    """The real run_arc2 on the scripted server (KV 5% a reply in flight): the cache gate admits several requests
    where the reserving rule took one, stays under the target and logs every admission as cache-based."""
    report, rows = _cache_roster(tmp_path, monkeypatch, ARC2_PREFLIGHT_KV_PER="0.05")
    g = report["gate"]
    assert report["status"] == "complete" and not report["task_errors"]
    assert g["admission"] == "kv" and g["kv_admitted"] > 0 and g["readings"] > 0
    assert g["peak_inflight"] >= 2 and g["peak_reserved"] > g["capacity"] and g["peak_kv"] <= 0.8


def test_run_arc2_cache_admission_fails_closed_without_metrics(tmp_path, monkeypatch):
    """/metrics down: every admission is the reserving rule's, one request at a time, and the roster still plays."""
    report, rows = _cache_roster(tmp_path, monkeypatch, ARC2_PREFLIGHT_METRICS_DOWN="1")
    g = report["gate"]
    assert report["status"] == "complete" and report["replies"] >= 8 and not report["task_errors"], report.get("error")
    assert g["kv_admitted"] == 0 and g["reserve_admitted"] > 0 and g["readings"] == 0 and g["stale_reads"] > 0
    assert g["peak_reserved"] <= g["capacity"]


def test_charge_counts_the_hint_not_the_reserved_cap():
    """charge 1: an admission adds its hint (prompt + mean reply) to the estimate, not prompt + cap; queued
    requests are fitted on their own hint."""
    async def go():
        clock = Clock()
        g = gate(clock, charge=1.0)
        g.observe({"kv_cache_fraction": 0.1, "preemptions": 0})
        for x in "abcd":
            g.hint(x, 15)  # reserved 30 (prompt + cap), expected 15 (prompt + mean reply)
        held = [await g.acquire(x, 30) for x in "abcd"]  # 0.1 + 4 x 0.15 = 0.7; full cost would stop at two
        assert len(held) == 4 and g.estimate() == pytest.approx(0.7)
        g.hint("e", 15)
        fifth = asyncio.ensure_future(g.acquire("e", 30))  # 0.7 + 0.15 > 0.8: waits on its own hint
        await asyncio.sleep(0)
        assert not fifth.done()
        clock.t = 16.0  # the pending charges settle; a reading shows what they really use
        g.observe({"kv_cache_fraction": 0.6, "preemptions": 0})
        await asyncio.sleep(0)
        assert fifth.done() and g.estimate() == pytest.approx(0.75)
        assert not g.charges  # a hint is spent by its admission
    asyncio.run(go())


def test_burst_of_long_reopens_stays_under_095_and_a_preemption_backs_the_target_off():
    """A burst of long-conversation re-admissions, each pinning its whole cached prompt (the hint's prompt part)
    and growing by its real reply, with readings every 2 s that lag the admissions: the real KV fraction never
    passes 0.95, and a preemption cuts the target from 0.88 to 0.80."""
    async def go():
        clock = Clock()
        blocks = 1000
        g = CacheGate(10 ** 9, slots=120, blocks=blocks, now=clock, target=0.88, max_age=15.0, settle=15.0,
                      step=0.08, floor=0.5, charge=1.0)
        live = {}  # admitted -> (prompt blocks, reply blocks)
        prompt, reply_mean, reply_real, cap = 30, 3, 5, 16  # blocks: real replies run longer than the mean
        real = lambda: sum(p + r for p, r in live.values()) / blocks  # noqa: E731
        g.observe({"kv_cache_fraction": 0.0, "preemptions": 0})
        waiters = []
        for i in range(60):  # 60 conversations come back at once
            g.hint(i, prompt + reply_mean)
            waiters.append(asyncio.ensure_future(g.acquire(i, prompt + cap)))
        peak = 0.0
        for step in range(30):  # 60 s of 2 s readings; admitted turns pin their prompt and generate
            await asyncio.sleep(0)
            for i, w in enumerate(waiters):
                if w.done() and i not in live:
                    live[i] = (prompt, 0)
            for i, (p, r) in live.items():
                live[i] = (p, min(reply_real, r + 2))
            peak = max(peak, real())
            clock.t += 2.0
            g.observe({"kv_cache_fraction": real(), "preemptions": 0})
        assert peak <= 0.95 and len(live) >= 20, (peak, len(live))
        g.observe({"kv_cache_fraction": real(), "preemptions": 1})
        assert g.target == pytest.approx(0.80) and g.stats["target_cuts"] == 1
        for w in waiters:
            w.cancel()
    asyncio.run(go())


def test_control_file_sets_gate_knobs_live(tmp_path):
    """Checklist item 5: control.json's admission target/floor/step and slot ceiling apply at the next tick, capped
    by the worker count; the wall still reads from the same file; an unchanged file changes nothing."""
    from arc2.trial import control_gate, control_read, control_wall

    g = CacheGate(10 ** 9, slots=104, blocks=1000, target=0.88, step=0.08, floor=0.72, charge=1.0)
    path, seen = tmp_path / "control.json", [None]
    path.write_text(json.dumps({"wall_seconds": 5000, "admission": {"target": 0.7, "bogus": 1}, "slots": 500}))
    read = control_read(path, seen)
    assert control_wall(path, seen, read=read) == 5000.0
    assert control_gate(g, read, 104) == {"target": 0.7, "floor": 0.7}
    assert (g.target, g.floor, g.slots) == (0.7, 0.7, 104)
    assert control_read(path, seen) is None
    assert control_gate(g, {"slots": 64, "admission": {"target": 1.5}}, 104) == {"slots": 64} and g.target == 0.7


def test_control_file_sets_the_backlog_order_live_and_rekeys_the_queue():
    """the backlog / reopen order is a control.json knob; waiting entries are re-keyed at once."""
    from arc2 import conv
    from arc2.loop import TaskRun
    from arc2.trial import control_order

    eng = conv.ConvEngine(complete=None, count=None, execute=None, window=65536, cap=16384,
                          settings={"conversation": True, "open_order": "tokens"})
    runs = []
    for k, (first, spent) in enumerate(((90000, 120000), (10000, 30000))):
        r = TaskRun(f"t{k}#1", {"train": [], "test": []}, {})
        r.submits, r.first_submit_tokens, r.spent = 1, first, spent
        runs.append(r)
    q = asyncio.PriorityQueue()
    for n, r in enumerate(runs):
        q.put_nowait((eng.priority(r), n))
    q.put_nowait(((1, 0), -1))  # a reopen token keeps its key
    tasks_first = lambda: [n for _, n in sorted(q._queue) if n >= 0]  # noqa: E731
    assert tasks_first() == [1, 0]  # fewest tokens first
    assert control_order(eng, {"order": "balance"}, q, runs) == "balance" and eng.live_order == "balance"
    assert tasks_first() == [0, 1] and ((1, 0), -1) in q._queue  # 1.33 < 3.0
    assert control_order(eng, {"order": "balance"}, q, runs) is None and control_order(eng, {"order": "x"}, q, runs) is None



def test_overhang_gate_cuts_capacity_by_blocks_used_beyond_reservations_and_recovers():
    """Q6 (28 Oct): real use above every reservation (blocks held outside requests) shrinks the gate's capacity, so
    reservations + overhang stay within it; the cut lasts ``window`` seconds after the overhang goes."""
    from arc2.admission import OverhangGate

    async def go():
        clock = Clock()
        g = OverhangGate(100, slots=8, blocks=200, window=60.0, now=clock)
        a = await g.acquire("a", 60)
        g.observe({"kv_cache_fraction": 0.4})  # 80 blocks used, 60 reserved: overhang 20
        assert g.capacity == 80 and g.stats["peak_overhang"] == 20
        waiter = asyncio.ensure_future(g.acquire("b", 30))  # 60 + 30 > 80: waits (KVGate alone admits it)
        await asyncio.sleep(0)
        assert not waiter.done()
        clock.t = 30.0
        g.observe({"kv_cache_fraction": 0.3})  # overhang gone, but within the window: the cut stands
        await asyncio.sleep(0)
        assert not waiter.done() and g.capacity == 80
        clock.t = 91.0
        g.observe({"kv_cache_fraction": 0.3})  # window passed: full capacity, the queue pumps
        await asyncio.sleep(0)
        assert waiter.done() and g.capacity == 100
        g.observe(None)
        assert g.stats["stale_reads"] == 1 and g.capacity == 100
        g.release(a)
    asyncio.run(go())


def test_run_arc2_overhang_gate_plays_the_roster_on_the_scripted_server(tmp_path, monkeypatch):
    """The real run_arc2 with ``arc2.gate_overhang`` on the scripted server: readings arrive, the roster plays, and
    reservations never pass the (cut) capacity."""
    import threading
    import time

    from test_arc2 import _scripted_roster

    from arc2.trial import run_arc2

    monkeypatch.setenv("ARC2_PREFLIGHT_KV_PER", "0.05")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.2")
    study.update(generation_slots=4, active_limit=4)
    study["arc2"].update(hold_attempt_blocks=True, gate_overhang={"window": 1.0, "poll": 0.1})
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False}]
    out = tmp_path / "out"

    def stopper():
        for _ in range(1200):
            events = out / "events.jsonl"
            if events.exists() and len(events.read_text().splitlines()) >= 24:
                (out / "stop-request").write_text("{}")
                return
            time.sleep(0.05)

    threading.Thread(target=stopper, daemon=True).start()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 300))
    g = report["gate"]
    assert report["status"] == "complete" and not report["task_errors"]
    assert g["admission"] == "overhang" and g["readings"] > 0 and g["peak_reserved"] <= g["base_capacity"]


def test_run_arc2_second_attempt_on_the_scripted_server(tmp_path, monkeypatch):
    """B (``second_attempt``) through the real run_arc2 on the scripted server: every entry submits, then each gets one
    fresh second attempt from the token budget (read out apart, its tokens on the roster clock); the entries keep
    their own rows, the trial ends by itself."""
    import time

    from test_arc2 import _scripted_roster

    from arc2.trial import run_arc2

    monkeypatch.setenv("ARC2_PREFLIGHT_KV_PER", "0.05")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.05")
    study.update(generation_slots=4, active_limit=4)
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False, "second_attempt": 1,
                      "second_view": True}]
    trial.update(wall_seconds=45)
    out = tmp_path / "out"
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 120))
    rows = [json.loads(x) for x in (out / "tasks.jsonl").read_text().splitlines()]
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    entries = [r for r in rows if "#b" not in r["task"]]
    seconds = [r for r in rows if "#b" in r["task"]]
    assert report["status"] == "complete" and not report["task_errors"]
    assert len(entries) == len(trial["tasks"]) * int(trial.get("copies", 1)) and len(seconds) == len(entries)
    assert report["extra_tokens"] > 0 and all("#b" not in t for t in report.get("per_task", {}) or {})
    spawned = [e for e in events if e.get("event") == "second_attempt"]
    assert len(spawned) == len(entries) == len({e["task"] for e in spawned})  # once per entry, never for a second
    assert {e["second"] for e in spawned} == {r["task"] for r in seconds}
    assert any(e.get("event") == "reply" and "#b" in e["task"] for e in events)


def test_run_arc2_rounds_after_all_submitted_on_the_scripted_server(tmp_path, monkeypatch):
    """``rounds_after_all_submitted`` through the real run_arc2: no reply of a submitted task's later round starts
    before every entry has its first submit; then rounds run."""
    import threading
    import time

    from test_arc2 import _scripted_roster

    from arc2.trial import run_arc2

    monkeypatch.setenv("ARC2_PREFLIGHT_KV_PER", "0.05")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "1")
    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.05")
    study.update(generation_slots=4, active_limit=4)
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False, "submit_mode": "provisional",
                      "firm": True, "swap": "budget", "open_order": "rounds", "reopen_order": "rounds",
                      "budget_floor": True, "rounds_after_all_submitted": True}]
    trial.update(wall_seconds=60)
    out = tmp_path / "out"

    def rows():
        events = out / "events.jsonl"
        return [json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []

    def stopper():
        for _ in range(1200):
            if len(rows()) >= 120:
                (out / "stop-request").write_text("{}")
                return
            time.sleep(0.05)

    threading.Thread(target=stopper, daemon=True).start()
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 120))
    assert report["status"] == "complete" and not report["task_errors"]
    replies = [e for e in rows() if e.get("event") == "reply"]
    first = {}
    for e in replies:
        if e.get("outcome") == "submit":
            first.setdefault(e["task"], e["t1"])
    assert len(first) == len(trial["tasks"])  # every entry submitted
    all_in = max(first.values())
    later = [e for e in replies if e["task"] in first and e["t0"] > first[e["task"]]]
    assert later and all(e["t0"] >= all_in - 1e-6 for e in later)  # later rounds only after the last first submit


def test_run_arc2_multi_attempt_on_the_scripted_server(tmp_path, monkeypatch):
    """b2 through the real run_arc2: most prompts submit (close mode ends those entries), a tenth never do;
    once every entry has started, fresh attempts fill the free slots, re-entries follow, merges land in the entries'
    rows; attempts are read out apart and run the roster clock."""
    import time

    from test_arc2 import _scripted_roster

    from arc2.trial import run_arc2

    monkeypatch.setenv("ARC2_PREFLIGHT_KV_PER", "0.05")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "tenth")  # half: a rotated view keeps the stub's prompt parity, so
    # attempts of never-submitting entries (largest keys) never submit and no merge lands; tenth as pf19
    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.05")
    study.update(generation_slots=8, active_limit=8)  # free slots beside the never-submitting entries
    trial["arms"] = [{"name": "conv", "conversation": True, "tool": True, "images": False, "multi_attempt": True,
                      "multi_view": True}]
    trial.update(wall_seconds=90)  # 45 s: no attempt submitted in time on a loaded runner (-n 8)
    out = tmp_path / "out"
    report = asyncio.run(run_arc2(study, trial, source=str(source), output=out, server_dir=server, base_url=url,
                                  deadline_epoch=time.time() + 200))
    rows = [json.loads(x) for x in (out / "tasks.jsonl").read_text().splitlines()]
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    entries = [r for r in rows if "#b" not in r["task"]]
    attempts = [e for e in events if e.get("event") == "attempt"]
    assert report["status"] == "complete" and not report["task_errors"] and report["extra_tokens"] > 0
    assert len(entries) == len(trial["tasks"]) and attempts
    assert {e["attempt"] for e in attempts} == {r["task"] for r in rows if "#b" in r["task"]}
    assert all(e["task"] in {r["task"] for r in entries} for e in attempts)  # attempts belong to entries
    assert len({e["attempt"] for e in attempts}) == len(attempts)
    ks = {}
    for e in attempts:
        ks.setdefault(e["task"], []).append(e["k"])
    assert all(v == list(range(2, 2 + len(v))) for v in ks.values())  # attempt numbers run on per entry
    assert any("#b" in e.get("task", "") for e in events if e.get("event") == "reply")  # attempts played
    assert any(e.get("event") == "attempt_merge" for e in events)
    assert any(max(v) >= 3 for v in ks.values())  # re-entry


def test_run_arc2_fork_plays_only_reopen_passes_on_the_scripted_server(tmp_path, monkeypatch):
    """arc2.fork through the real run_arc2: a naive base run, its snapshot, then a fork that starts every submitted
    entry at its first reopen (no first-pass reply is played again), keeps never-submitted entries' base results, and
    runs the roster clock from the base's first-pass tokens."""
    import time

    from test_arc2 import _scripted_roster

    from arc2 import fork
    from arc2.trial import run_arc2

    monkeypatch.setenv("ARC2_PREFLIGHT_KV_PER", "0.05")
    monkeypatch.setenv("ARC2_PREFLIGHT_TOOLS", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_SUBMIT", "tenth")
    source, server, url, study, trial = _scripted_roster(tmp_path, monkeypatch, delay="0.02")
    arm = {"name": "conv", "conversation": True, "tool": True, "images": False, "submit_mode": "fresh",
           "fresh_each": True, "fresh_text": "naive", "keep_first": True, "reopen_order": "first_submit"}
    base_out = tmp_path / "base"
    asyncio.run(run_arc2(study, {**trial, "arms": [arm], "wall_seconds": 60, "marks": {"end": 40000}},
                         source=str(source), output=base_out, server_dir=server, base_url=url,
                         deadline_epoch=time.time() + 120))
    rows = [json.loads(x) for x in (base_out / "tasks.jsonl").read_text().splitlines()]
    events = [json.loads(x) for x in (base_out / "events.jsonl").read_text().splitlines()]
    snap = fork.snapshot(rows, events)
    assert any(e["submitted"] for e in snap["entries"].values())
    (source / "fork.json").write_text(json.dumps(snap))
    first = snap["first_pass_tokens"]
    out = tmp_path / "fork"
    report = asyncio.run(run_arc2(study, {**trial, "arms": [arm], "wall_seconds": 60, "fork": "fork.json",
                                          "marks": {"end": first + 20000}}, source=str(source), output=out,
                                  server_dir=server, base_url=url, deadline_epoch=time.time() + 120))
    assert report["status"] == "complete" and not report["task_errors"] and report["extra_tokens"] >= first
    ev = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    reopened, early = set(), []
    for e in ev:
        if e.get("event") == "reopen":
            reopened.add(e["task"])
        elif e.get("event", "reply") == "reply" and e.get("task") not in reopened:
            early.append(e)
    assert reopened and not early  # every reply belongs to a reopen pass
    first = {}  # each reopened entry's first pass: replies, submitted
    for e in ev:
        if e.get("event", "reply") == "reply" and e.get("task") in reopened and e["task"] not in first or (
                e.get("event", "reply") == "reply" and first.get(e.get("task"), {}).get("open")):
            row = first.setdefault(e["task"], {"replies": 0, "open": True})
            row["replies"] += 1
            if e.get("outcome") == "submit":
                row["open"] = False
    stalled = [t for t, r in first.items() if r["replies"] == 1 and r["open"]]
    assert len(stalled) <= int(study["active_limit"])  # only passes cut by the end mark (fk-naive1: 159 of 236)
    frows = {json.loads(x)["task"]: json.loads(x) for x in (out / "tasks.jsonl").read_text().splitlines()}
    for name, s in snap["entries"].items():
        if not s["submitted"]:
            assert frows[name]["ended"] == "fork_base"


def test_never_submitted_requests_are_admitted_strictly_first():
    async def go():
        clock = Clock()
        g = gate(clock)  # reserving rule (no reading): capacity 30 units
        first = await g.acquire("hold", 20)
        g.rank.update({"reopen": 1, "fresh": 0})
        reopen = asyncio.ensure_future(g.acquire("reopen", 25))  # queued first, a submitted entry's pass
        await asyncio.sleep(0)
        fresh = asyncio.ensure_future(g.acquire("fresh", 15))  # a never-submitted entry, queued later
        await asyncio.sleep(0)
        assert not reopen.done() and not fresh.done()
        g.release(first)
        await asyncio.sleep(0)
        assert fresh.done() and not reopen.done()  # rank 0 first though queued later
        small = asyncio.ensure_future(g.acquire("reopen2", 5))
        g.rank["big"] = 0
        big = asyncio.ensure_future(g.acquire("big", 30))
        await asyncio.sleep(0)
        assert not big.done() and not small.done()  # strict: no rank-1 request passes a waiting rank-0 one
        g.release(fresh.result())
        await asyncio.sleep(0)
        assert big.done() and not reopen.done() and not small.done()
        g.release(big.result())
        await asyncio.wait_for(reopen, 1)
        await asyncio.wait_for(small, 1)
    asyncio.run(go())


def test_gate_rank_marks_never_submitted_entries_only():
    from arc2.conv import gate_rank
    from arc2.loop import TaskRun
    on = {"ns_first_gate": True}
    a = TaskRun("t#1", {"train": [], "test": []}, {})
    assert gate_rank(a, on) == 0 and gate_rank(a, {"ns_first_gate": False}) == 1
    a.submits = 1
    assert gate_rank(a, on) == 1
    b = TaskRun("t#b1x1", {"train": [], "test": []}, {"attempt_k": 1})
    assert gate_rank(b, on) == 1  # b2 attempts are work on submitted entries
