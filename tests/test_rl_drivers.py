"""Multi-process arena drivers (arena.rl_drivers, mg.redesign.drivers): groups play in spawned workers, the backlog,
hook, harvest and phases stay in the controller."""
import asyncio
import hashlib
import inspect
import json
import multiprocessing
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_rl_seq import FakeLoop, harvest_ok  # noqa: F401 - harvest_ok is a fixture

from arena import level_bench, phase, rl_backlog, rl_drivers, rl_seq


def draw(*key):
    return int(hashlib.md5(":".join(map(str, key)).encode()).hexdigest()[:8], 16)


class SlowUpstream(httpx.AsyncBaseTransport):
    def __init__(self, delay=0.002):
        self.delay = delay

    async def handle_async_request(self, request):
        await asyncio.sleep(self.delay)
        return httpx.Response(200, json={"ok": True})


class FileFeed:
    """The controller's one feed.jsonl (rl_loop.Feed's file), read back by kind."""

    def __init__(self, folder):
        from arena.rl_loop import Feed

        self.inner = Feed(folder)
        self.path = Path(folder) / "feed.jsonl"

    def emit(self, kind, **data):
        return self.inner.emit(kind, **data)

    @property
    def rows(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.is_file() else []

    def of(self, kind):
        return [r for r in self.rows if r["kind"] == kind]


class DetLoop(FakeLoop):
    """FakeLoop whose tries' requests and clears are fixed by (group, try): the same in any process."""

    def __init__(self, tmp, max_steps=3, clear_p=0.3, drivers=0, delay=0.002, die=None, file_feed=False):
        super().__init__(Path(tmp), max_steps=max_steps, clear_p=clear_p)
        if file_feed:
            self.feed = FileFeed(self.output / "rl")
        if drivers:
            self.cfg["mg"]["redesign"]["drivers"] = drivers
        self.delay, self.die, self.plays = delay, die, 0
        self.driver_factory = (worker_loop, {"tmp": str(tmp), "clear_p": clear_p, "delay": delay, "die": die})
        self.driver_gate = lambda: phase.PhaseGate(inner=SlowUpstream(delay),
                                                   leader_first=self.cfg["mg"].get("leader_first"))

    async def mg_play(self, name, spec, starts, tries, *, model, seed, harvest, base_url=None):
        assert (model, harvest) in {("policy", True), ("base", False)}  # base: the control group
        self.plays += 1
        if self.die and self.die["driver"] in ("all", rl_drivers.DRIVER_INDEX) and self.plays > self.die["after"]:
            os._exit(3)
        if rl_drivers.DRIVER_INDEX is not None:
            self.feed.emit("worker_play", bench=name, pid=os.getpid())  # forwarded to the controller's feed
        client = httpx.AsyncClient(base_url=base_url or "http://s0/v1", transport=level_bench.PHASE_GATE)
        rows = []

        async def one(i):
            for _ in range(1 + draw(name, i, "n") % 3):
                await client.post("chat/completions", json={"messages": [{"role": "user", "content": name}]})
            cleared = draw(name, i) / 2 ** 32 < self.clear_p
            row = {"label": f"a{i:03d}", "outcome": "cleared" if cleared else "budget", "completion_tokens": 900,
                   "level_rows": [{"tokens": 800, "actions": 20, "score": 50.0}] if cleared else []}
            if cleared:
                row["harvest"] = {"request": 7, "step": 3, "state": "NOT_FINISHED"}
            rows.append(row)
            level_bench.PROGRESS_HOOK(name, row)

        await asyncio.gather(*(one(i) for i in range(tries)))
        await client.aclose()
        return self.output / "rl" / name, rows, {"seconds": 0.01}


def worker_loop(tmp, clear_p, delay, die):
    """What a driver process builds (rl_drivers.real_loop builds rl_loop.Loop)."""
    return DetLoop(tmp, clear_p=clear_p, delay=delay, die=die, file_feed=True)


@pytest.fixture
def spy(monkeypatch):
    """Backlog accounting per gid: accepted on_try_done indexes and full groups."""
    got = {"done": defaultdict(list), "full": {}}
    real = rl_backlog.Backlog.on_try_done

    def on_try_done(self, group_id, index, **kw):
        live = self.groups.get(group_id)
        if live is not None and index not in live["results"]:
            got["done"][group_id].append(index)
        out = real(self, group_id, index, **kw)
        if out is not None:
            got["full"][group_id] = out
        return out

    monkeypatch.setattr(rl_backlog.Backlog, "on_try_done", on_try_done)
    return got


def run(loop, gate=None, timeout=60):
    return asyncio.run(asyncio.wait_for(rl_seq.run(loop, gate=gate), timeout))


def no_children():
    multiprocessing.active_children()  # reaps finished ones
    return [p for p in multiprocessing.active_children() if p.name.startswith("rl-driver-")]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie of ours: reap it
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True


def check_accounting(loop, spy, steps):
    feed = loop.feed
    groups = {r["gid"]: r for r in feed.of("seq_group")}
    done = {r["gid"]: r for r in feed.of("group_done")}
    assert len([r for r in feed.of("step")]) == steps
    assert set(done) <= set(groups)
    for gid, full in spy["full"].items():  # every full group: each try accounted once, its row count = its tries
        assert sorted(spy["done"][gid]) == list(range(groups[gid]["tries"]))
        assert len(full["tries"]) == groups[gid]["tries"]
        assert done[gid]["tries"] == groups[gid]["tries"]
    trained = set(loop.accumulated)
    assert trained and trained <= set(done) and trained <= set(spy["full"])
    state = json.loads((Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "seq_state.json").read_text())
    return state, groups, done


def test_drivers_0_and_2_keep_the_same_backlog_accounting(tmp_path, harvest_ok, spy, monkeypatch):  # noqa: F811
    out = {}
    for n in (0, 2):
        monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / f"vo{n}"))
        spy["done"].clear()
        spy["full"].clear()
        loop = DetLoop(tmp_path / f"d{n}", max_steps=2, drivers=n, file_feed=True)
        why = run(loop, gate=None if n else phase.PhaseGate(inner=SlowUpstream()))
        assert why == "max_steps" and loop.trainer.steps == 2
        state, groups, done = check_accounting(loop, spy, steps=2)
        out[n] = (state, groups, done, loop)
    s0, s2 = out[0][0], out[2][0]
    assert (s0["step"], s0["version"]) == (s2["step"], s2["version"]) == (2, 2)
    assert set(s0["backlog"]) == set(s2["backlog"])
    loop2 = out[2][3]
    # one feed.jsonl (the controller's); the workers' rows are in it, tagged with their driver
    feeds = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("feed.jsonl"))
    assert feeds == [Path("d0/out/rl/feed.jsonl"), Path("d2/out/rl/feed.jsonl")]
    plays = loop2.feed.of("worker_play")
    assert plays and {r["driver"] for r in plays} == {0, 1} and len({r["pid"] for r in plays}) == 2
    assert all(r["pid"] != os.getpid() for r in plays)
    assert {r["driver"] for r in loop2.feed.of("seq_group")} == {0, 1}
    assert not out[0][3].feed.of("worker_play") and "driver" not in out[0][3].feed.of("seq_group")[0]
    assert not loop2.feed.of("alert") and not no_children()
    assert level_bench.PHASE_GATE is None and not phase.TIME.paused


def fake_pool(n=2, urls=("http://a/v1",)):
    sent = [[] for _ in range(n)]
    loop = SimpleNamespace(cfg={"mg": {}}, feed=SimpleNamespace(rows=[], emit=lambda kind, **kw: None),
                           driver_factory=(worker_loop, {}))
    pool = rl_drivers.DriverPool(n, loop, {})
    for i in range(n):
        ch = SimpleNamespace(send=sent[i].append, unlisten=lambda: None, close=lambda: None)
        proc = SimpleNamespace(pid=1000 + i, is_alive=lambda: True, exitcode=None, kill=lambda: None)
        pool.workers.append(rl_drivers.Worker(i, proc, ch))
    return pool, sent


def test_placement_balances_unfinished_tries_per_server():
    async def go():
        pool, _ = fake_pool(2)
        a, b = "http://a/v1", "http://b/v1"
        assert pool.assign("g1", "g1-x", 12, a) == 0
        assert pool.assign("g2", "g2-x", 12, a) == 1
        assert pool.assign("g3", "g3-x", 8, a) == 0  # tie at 12: index order
        assert pool.assign("g4", "g4-x", 12, a) == 1  # 20 vs 12
        assert pool.assign("g5", "g5-x", 12, b) == 0  # another server's load counts apart
        for _ in range(11):  # g1's tries finish on worker 0: 1 + 8 left there, 24 on worker 1
            pool.on_message(pool.workers[0], {"op": "try_done", "bench": "g1-x", "row": {}})
        assert pool.workers[0].unfinished(a) == 9 and pool.workers[1].unfinished(a) == 24
        assert pool.assign("g6", "g6-x", 12, a) == 0
        pool.workers[0].alive = False
        assert pool.assign("g7", "g7-x", 12, a) == 1
        pool.workers[1].alive = False
        assert pool.assign("g8", "g8-x", 12, a) is None
        for w in pool.workers:  # futures never awaited here
            for g in w.groups.values():
                g["future"].cancel()

    asyncio.run(go())


def test_capacity_share_scales_the_pool_and_its_gate():
    pool = level_bench.SharedSlots(10, capacity_share=0.25)
    assert pool.share(1000) == 250 and level_bench.SharedSlots(10).share(1001) == 1001
    pool.budget = SimpleNamespace(layout={"num_blocks": 100, "costs": [{"tokens": 16, "blocks": 1}]})
    assert pool.capacity_tokens() == 400
    full = level_bench.SharedSlots(10)
    full.budget = pool.budget
    assert full.capacity_tokens() == 1600
    src = inspect.getsource(level_bench.LevelBench.run)  # the shared pool and its KV gate take the share
    assert "pool.share(capacity)" in src and "KVGate(pool.share(request_capacity)" in src


async def fresh_states(pool, timeout=10.0):
    before = {w.index: (w.state or {}).get("t") for w in pool.live()}
    for w in pool.live():
        w.ch.send({"op": "ping"})
    end = time.monotonic() + timeout
    while any((w.state or {}).get("t") == before[w.index] for w in pool.live()):
        assert time.monotonic() < end, "no fresh state"
        await asyncio.sleep(0.02)
    return {w.index: w.state for w in pool.live()}


def test_workers_take_ceil_slots_a_capacity_share_and_live_knobs(tmp_path):
    async def go():
        loop = DetLoop(tmp_path, drivers=3)
        loop.cfg["mg"]["active_tries"] = 50
        pool = rl_drivers.DriverPool(3, loop, {})
        urls = ["http://a/v1", "http://b/v1"]
        try:
            assert await pool.start(urls) == 3
            states = await fresh_states(pool)
            for st in states.values():
                assert set(st["pools"]) == set(urls)
                assert all(p["slots"] == 17 and abs(p["capacity_share"] - 1 / 3) < 1e-9 for p in st["pools"].values())
            pool.knobs({"active_tries": 90})
            states = await fresh_states(pool)
            assert all(p["slots"] == 30 for st in states.values() for p in st["pools"].values())
            row = pool.pool_row(urls[0])
            assert row["slots"] == 90 and row["contexts"] == {"n": 0} and len(pool.driver_rows()) == 3
            assert loop.feed.of("seq_drivers")[0]["slots"] == 17
            pool.knobs({"hold": [urls[0]]})  # blind guard: each worker holds at its own used (none here -> 1)
            states = await fresh_states(pool)
            assert all(st["pools"][urls[0]]["slots"] == 1 and st["pools"][urls[1]]["slots"] == 30 for st in states.values())
            pool.knobs({"active_tries": 90})
            states = await fresh_states(pool)
            assert all(p["slots"] == 30 for st in states.values() for p in st["pools"].values())
        finally:
            await pool.stop()

    asyncio.run(asyncio.wait_for(go(), 60))
    assert not no_children()


def test_proxy_gate_close_returns_after_every_worker_acks():
    async def go():
        pool, sent = fake_pool(3)
        task = asyncio.ensure_future(pool.gate.close())
        await asyncio.sleep(0.01)
        assert not task.done() and [m[-1]["state"] for m in sent] == ["closed"] * 3
        epoch = sent[0][-1]["epoch"]
        pool.on_message(pool.workers[0], {"op": "gate_ack", "epoch": epoch, "aborted": 2})
        pool.on_message(pool.workers[2], {"op": "gate_ack", "epoch": epoch, "aborted": 1})
        await asyncio.sleep(0.01)
        assert not task.done()  # worker 1 has not acked
        pool.dead(pool.workers[1], "test")  # a dead worker's ack resolves (never a hang)
        assert await task == 3 and not pool.gate.is_open
        task = asyncio.ensure_future(pool.gate.open())
        await asyncio.sleep(0.01)
        assert not task.done() and len(sent[1]) == 1  # the dead worker gets nothing
        pool.on_message(pool.workers[0], {"op": "gate_ack", "epoch": epoch + 1})
        pool.on_message(pool.workers[2], {"op": "gate_ack", "epoch": epoch + 1})
        await task
        assert pool.gate.is_open

    asyncio.run(go())


def test_proxy_gate_holds_requests_while_closed_and_resends_after_open(tmp_path):
    async def go():
        loop = DetLoop(tmp_path, drivers=2, delay=0.4)
        pool = rl_drivers.DriverPool(2, loop, {})
        try:
            await pool.start(["http://s0/v1"])
            spec = {"name": "sb26-L1", "game": "sb26", "level": 1, "role": "train", "cap": 1000, "fresh": True}
            plays = []
            for k in range(4):
                gid = f"g{k}"
                pool.assign(gid, f"{gid}-sb26-L1", 12, "http://s0/v1")
                plays.append(asyncio.ensure_future(pool.play(gid, f"{gid}-sb26-L1", spec, [], 12, model="policy",
                                                             url=None)))
            await asyncio.sleep(0.2)  # every try's first request in flight
            ctl = phase.PhaseController([pool.gate], [], _noop, _noop)
            stop = await ctl.to_train()  # returns once both workers closed, aborted and drained
            assert not pool.gate.is_open and phase.TIME.paused
            first = await fresh_states(pool)
            assert all(not st["gate"]["open"] and st["gate"]["inflight"] == 0 for st in first.values())
            assert stop["aborted"] == sum(st["gate"]["aborted"] for st in first.values()) > 0
            await asyncio.sleep(0.4)
            later = await fresh_states(pool)
            assert all(later[i]["gate"]["sent"] + later[i]["gate"]["resent"]
                       == first[i]["gate"]["sent"] + first[i]["gate"]["resent"] for i in later)  # nothing sent
            assert not any(p.done() for p in plays)
            await ctl.to_play()
            got = await asyncio.wait_for(asyncio.gather(*plays), 30)
            assert all(len(rows) == 12 for _, rows, _ in got)
            end = await fresh_states(pool)
            assert sum(st["gate"]["resent"] for st in end.values()) > 0 and not phase.TIME.paused
        finally:
            await pool.stop()

    asyncio.run(asyncio.wait_for(go(), 60))
    assert not no_children()


async def _noop(*a, **kw):
    return None


def test_a_killed_worker_alerts_fails_its_groups_and_the_run_ends(tmp_path, harvest_ok, monkeypatch):  # noqa: F811
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = DetLoop(tmp_path / "a", max_steps=2, drivers=2, die={"driver": 1, "after": 1}, file_feed=True)
    assert run(loop, timeout=60) == "max_steps"
    alerts = loop.feed.of("alert")
    died = [a for a in alerts if a["why"] == "driver died"]
    assert len(died) == 1 and died[0]["driver"] == 1
    failed = [a for a in alerts if a["why"] == "seq: group play failed"]
    assert failed and all("driver 1" in a["error"] for a in failed)
    assert loop.trainer.steps == 2 and not no_children()
    # every worker dead: the run stops at once (no spin, no hang)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo2"))
    loop = DetLoop(tmp_path / "b", max_steps=3, drivers=2, die={"driver": "all", "after": 1}, file_feed=True)
    assert run(loop, timeout=60) == "drivers_dead"
    assert len([a for a in loop.feed.of("alert") if a["why"] == "driver died"]) == 2
    assert loop.feed.of("seq_end")[-1]["why"] == "drivers_dead" and not no_children()


def test_controller_cancel_leaves_no_live_worker(tmp_path, harvest_ok, monkeypatch):  # noqa: F811
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    # a run that raises in its setup (test_rl_seq's draft-off resume) leaves its hook behind: start from none
    monkeypatch.setattr(level_bench, "PROGRESS_HOOK", None)
    monkeypatch.setattr(level_bench, "PHASE_GATE", None)
    loop = DetLoop(tmp_path, max_steps=50, drivers=2, delay=0.5, file_feed=True)

    async def go():
        task = asyncio.ensure_future(rl_seq.run(loop))
        while not loop.feed.of("worker_play"):
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(go(), 60))
    pids = loop.feed.of("seq_drivers")[0]["pids"]
    assert len(pids) == 2 and not no_children() and not any(pid_alive(p) for p in pids)
    assert level_bench.PHASE_GATE is None and level_bench.PROGRESS_HOOK is None


def test_play_only_preflight_with_drivers_ends_after_max_steps_phases(tmp_path, harvest_ok, monkeypatch):  # noqa: F811
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", "{}")
    loop = DetLoop(tmp_path, max_steps=2, clear_p=0.5, drivers=2, file_feed=True)
    loop.cfg["mg"]["redesign"].update(play_only=True, admit_log_s=0.05)
    loop.cfg["mg"]["active_tries"] = 50
    why = run(loop, timeout=60)
    assert why == "max_steps" and len(loop.feed.of("seq_play_phase")) == 2 and not loop.feed.of("seq_train")
    rows = [r for r in loop.feed.of("seq_admit") if "servers" in r]
    assert rows and all(len(r["drivers"]) == 2 for r in rows)
    assert any(r["servers"][0].get("slots") == 50 for r in rows)  # 2 x ceil(50 / 2), from the workers' state rows
    dw = [r["servers"][0]["dwell"] for r in rows if "dwell" in r["servers"][0]]  # summed over the workers
    assert dw and all(set(d) == {"now", "live_s"} and set(d["now"]) == {"sending", "gate_wait", "tool", "parked", "other"}
                      for d in dw)
    assert not no_children()


def test_drivers_setting_is_checked():
    base = {"streams": 4, "max_steps": 1}
    assert rl_seq.settings({"redesign": {**base, "drivers": 2}})["drivers"] == 2
    for bad in (-1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="drivers"):
            rl_seq.settings({"redesign": {**base, "drivers": bad}})


def test_a_silent_worker_is_declared_dead_and_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_drivers, "DEAD_S", 1.0)
    monkeypatch.setattr(rl_drivers, "STATE_S", 60.0)  # one state row at start, then silence

    async def go():
        loop = DetLoop(tmp_path, drivers=1)
        dead = []
        pool = rl_drivers.DriverPool(1, loop, {})
        try:
            await pool.start(["http://s0/v1"], on_all_dead=lambda: dead.append(1))
            pid = pool.workers[0].pid
            end = time.monotonic() + 10
            while pool.workers[0].alive:
                assert time.monotonic() < end
                await asyncio.sleep(0.1)
            alert = loop.feed.of("alert")
            assert alert[0]["why"] == "driver died" and alert[0]["cause"].startswith("silent") and dead == [1]
            assert pool.assign("g1", "g1-x", 4, "http://s0/v1") is None
            return pid
        finally:
            await pool.stop()

    pid = asyncio.run(asyncio.wait_for(go(), 60))
    assert not pid_alive(pid) and not no_children()


def test_a_worker_builds_the_real_loop_from_the_controllers_arguments(tmp_path):
    """The production factory: rl_loop.Loop rebuilt in the worker from loop_args (study, trial, paths, deadline)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from golden import common

    from arena import rl_loop

    study = common.B.inline(common.config("rl1-loop-seq-real-v8")["study"])
    lp = rl_loop.Loop(study, {"name": "drv"}, source=str(tmp_path), games_dir="/g", output=str(tmp_path / "out"),
                      server_dir="/s", analysis_python="py", base_url="http://127.0.0.1:1234/v1", deadline_epoch=123.0)
    lp.gpu.stop.set()
    w = rl_drivers.real_loop(**rl_drivers.loop_args(lp))
    w.gpu.stop.set()
    assert (w.base_url, w.deadline, w.output, w.source, w.cfg) == (lp.base_url, 123.0, lp.output, lp.source, lp.cfg)
    lp.cfg["mg"]["active_tries"] = 8

    async def go():
        pool = rl_drivers.DriverPool(2, lp, {})
        assert pool.factory == "arena.rl_drivers:real_loop"
        try:
            assert await pool.start([lp.base_url]) == 2
            states = await fresh_states(pool)
            assert all(st["pools"][lp.base_url]["slots"] == 4 for st in states.values())
        finally:
            await pool.stop()

    asyncio.run(asyncio.wait_for(go(), 60))
    assert not no_children() and lp.feed.status["seq_drivers"]["live"] == 2
    assert not (tmp_path / "out" / "rl" / "feed.jsonl").read_text().count('"kind": "alert"')


def test_workers_gate_takes_mg_leader_first(tmp_path):
    async def go():
        loop = DetLoop(tmp_path, drivers=2)
        loop.cfg["mg"]["leader_first"] = {"min_chars": 1000}
        pool = rl_drivers.DriverPool(2, loop, {})
        try:
            assert pool.setup(["http://a/v1"])["mg"]["leader_first"] == {"min_chars": 1000}
            assert await pool.start(["http://a/v1"]) == 2
            states = await fresh_states(pool)
            assert all(st["gate"]["leader_first"] for st in states.values())
        finally:
            await pool.stop()

    asyncio.run(asyncio.wait_for(go(), 60))
    assert not no_children()


def test_leader_first_run_keeps_the_backlog_accounting(tmp_path, harvest_ok, spy, monkeypatch):  # noqa: F811
    """mg.leader_first through the whole seq loop with 2 drivers: held turns warm up and resend, every try lands."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = DetLoop(tmp_path / "d", max_steps=2, drivers=2, file_feed=True, delay=0.02)
    loop.cfg["mg"]["leader_first"] = {"min_chars": 1, "min_share": 0.0, "wait_s": 5}
    assert run(loop) == "max_steps" and loop.trainer.steps == 2
    state, _, _ = check_accounting(loop, spy, steps=2)
    assert (state["step"], state["version"]) == (2, 2)
    assert not loop.feed.of("alert") and not no_children()


def test_control_twin_through_drivers_stays_out_of_the_backlog(tmp_path, harvest_ok, spy, monkeypatch):  # noqa: F811
    """mg.redesign.control with 2 drivers: the base-model twin plays in a worker; the backlog never sees its tries."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = DetLoop(tmp_path / "d", max_steps=4, drivers=2, file_feed=True, clear_p=0.0)
    loop.cfg["mg"]["redesign"]["control"] = {"every": 2, "tries": 3}
    assert run(loop) == "max_steps" and loop.trainer.steps == 4
    ctl = loop.feed.of("seq_control")
    assert ctl and loop.feed.of("control_done")
    assert not any(str(g).endswith("c") for g in spy["done"]) and not any(str(g).endswith("c") for g in spy["full"])
    check_accounting(loop, spy, steps=4)
    assert not loop.feed.of("alert") and not no_children()
