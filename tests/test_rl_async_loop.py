"""v7 async loop (arena.rl_async) on the mg loop's stand-ins (test_rl_startup): groups pinned to their version,
streamed accumulate + one step per batch, versions published and served, per-step checkpoint and state, reads
pinned to their version, and a resume that continues the step count without training a group twice."""
import asyncio
import json
from pathlib import Path

import pytest

from test_rl_startup import make_loop

V7 = {"group_size": 2, "batch_groups": 2, "s_max": 8, "max_ready": 8, "ramp_s": 0.0, "in_flight": 6, "read_every": 2,
      "read_tries": 2, "k3_windows": 2, "warmup": 2, "max_steps": 3, "poll_s": 0.01,
      "serve_every": 1, "max_loras": 10}

REAL_TIME = {"test_overlap_starts_the_trainer_with_the_boot_and_gates_its_card_use",  # a real trainer subprocess
             "test_loop_lag_catches_a_blocking_call"}  # measures a real blocking call


@pytest.fixture(autouse=True)
def virtual_time(monkeypatch, request):
    """Every loop here runs on virtual time (tests/vclock.py): event order follows the fakes' delays, never CPU speed
    (25 Oct: [True] variants flaked under CI load). Tests that need the real clock are listed in REAL_TIME."""
    if request.node.originalname in REAL_TIME:
        yield
        return
    import vclock

    old = vclock.vclock(monkeypatch, vclock.fixture_modules())
    yield
    asyncio.set_event_loop_policy(old)



def v7_loop(tmp_path, monkeypatch, log, *, steps=3, play_delay=0.05, mg=None):
    loop = make_loop(tmp_path, monkeypatch, log, play_delay=play_delay,
                     mg={"async": True, "resume": None, "v7": {**V7, "max_steps": steps}, "fixed_tries": 2, "cap": 1000, **(mg or {})})
    loop.cfg["lr"] = 1e-5
    plays = []

    async def play(name, spec, starts, tries, *, model, seed, **kw):
        plays.append((name, model, seed))
        folder = loop.output / "rl" / name
        folder.mkdir(parents=True, exist_ok=True)
        await asyncio.sleep(play_delay)
        rows = [{"label": f"{name}-a{i}", "cleared": i % 2 == 0, "completion_tokens": 1000} for i in range(tries)]
        return folder, rows, {}

    loop.mg_play = play
    loop.mg_select = lambda folder, rows, name=None: ([(Path(folder), r, 1.0 if r["cleared"] else -1.0, "s")
                                                        for r in rows], {"reward_mean": 0.5})
    loop.windows = lambda folder, rows, prefix="": [{"name": prefix + r["label"], "rollout": "x"} for r in rows]
    loop.plays = plays

    async def unload(name):
        log.append(("server.unload", name))

    loop.server.unload = unload
    return loop


def ops(loop):
    return [op for op, _ in loop.trainer.ops]


def events(loop, kind):
    return [d for k, d in loop.feed.events if k == kind]


def test_live_versions_capped_late_serve(tmp_path, monkeypatch):
    """max_loras 2: a due version waits for a slot (served_late) and live versions never pass the cap."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=5, play_delay=0.03)
    loop.cfg["mg"]["v7"]["max_loras"] = 2
    fast = loop.mg_play

    async def play(name, spec, starts, tries, **kw):  # one slow group on the base holds a slot over several steps
        if name.startswith("g00001-"):
            await asyncio.sleep(0.6)
        return await fast(name, spec, starts, tries, **kw)

    loop.mg_play = play
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert not [a for a in events(loop, "alert") if "max_loras" in a.get("why", "")]
    # the base model (version 0) needs no LoRA slot (v6g's base read pinned one): the slow base
    # group no longer delays any version; adapters never pass the cap
    assert all(len([x for x in s["live_versions"] if x != 0]) <= 2 for s in events(loop, "step"))
    late = events(loop, "served_late")
    waits = events(loop, "slot_wait")
    assert {w["version"] for w in waits} >= {d["version"] for d in late}
    served = [s["version"] for s in events(loop, "step") if s["load_seconds"] is not None] + [d["version"] for d in late]
    assert sorted(served) == sorted(set(served))
    starts = [d["version"] for d in events(loop, "group_start")]
    assert starts == sorted(starts)  # new groups never go back to an older version


def test_fresh_run_steps_publishes_and_pins(tmp_path, monkeypatch):
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    steps = events(loop, "step")
    assert [s["step"] for s in steps] == [1, 2, 3] and [s["version"] for s in steps] == [1, 2, 3]
    o = ops(loop)
    assert o.count("step") == 3 and o.count("checkpoint") == 3
    assert o.count("k3_score") == 3 and all(s.get("k3") for s in steps)  # forward-only k3 after each step
    acc = [p for op, p in loop.trainer.ops if op == "accumulate"]
    assert len(acc) == 6 and [a["first"] for a in acc] == [True, False] * 3
    assert [a["lr"] for a in acc[::2]] == [0.5e-5, 1e-5, 1e-5]  # warm-up over 2 updates
    # every group plays on the version served when it was made: base, then policy-rNN
    starts = {d["gid"]: d["version"] for d in events(loop, "group_start")}
    for name, model, _ in loop.plays:
        gid = name.split("-")[0]
        if gid in starts:
            assert model == ("base" if starts[gid] == 0 else f"policy-r{starts[gid]:02d}")
    assert ("server.load", str(loop.work / "adapters" / "v0001")) in log
    # a read every 2 steps on that step's version, at the read seed
    reads = events(loop, "read")
    assert [r["version"] for r in reads] == [2] and any(n.startswith("x0002-fixed-") and m == "policy-r02"
                                                         for n, m, _ in loop.plays)
    state = json.loads((Path(loop.v7_state_dir(3)).parent.parent / "v7_state.json").read_text())
    assert state["step"] == 3 and state["book"]["lineage"] == [0, 1, 2, 3]
    assert (Path(state["checkpoint"]) / "state.json").is_file()


def test_resume_continues_without_training_a_group_twice(tmp_path, monkeypatch):
    log = []
    first = v7_loop(tmp_path, monkeypatch, log, steps=2)
    asyncio.run(first.mg_loop())
    trained1 = {d["gid"] for d in events(first, "group_done")} & set(json.loads(
        (tmp_path / "out" / "v7_state.json").read_text())["buffer"]["closed"])
    log2 = []
    second = v7_loop(tmp_path / "b", monkeypatch, log2, steps=4)  # make_loop clears the resume env
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "out"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    assert asyncio.run(second.mg_loop()) == "max_steps"
    start = events(second, "v7_start")[0]
    assert start["resume"] and start["step"] == 2 and start["version"] == 2
    assert [s["step"] for s in events(second, "step")] == [3, 4]
    assert ("load_checkpoint" in ops(second))
    acc = [p for op, p in second.trainer.ops if op == "accumulate"]
    assert [a["lr"] for a in acc[::2]] == [0.5e-5, 1e-5]  # warm-up again after the resume
    gids2 = {d["gid"] for d in events(second, "group_start")}
    assert not gids2 & trained1  # new ids continue the count; nothing trained before is made again


def test_resume_after_a_lost_container_rebuilds_the_current_adapter(tmp_path, monkeypatch):
    """A preempted leg: its adapters (local disk) and trial folders (/tmp/arena) are gone; the resume rebuilds only the
    checkpoint's version (trainer save), re-pins unfinished groups to it, finds finished groups under the root's
    arena/ and re-reads a read the restart cut."""
    import shutil

    log = []
    first = v7_loop(tmp_path, monkeypatch, log, steps=2, play_delay=0.05)
    asyncio.run(first.mg_loop())
    out = tmp_path / "out"
    state = json.loads((out / "v7_state.json").read_text())
    state["reads"]["open"] = {"2": {"version": 2, "tries": 2, "done": 1}}
    state["running"] = state["running"] or [{"gid": "g09999", "unit": next(iter(first.cfg["mg_sets"])), "start": None,
                                             "version": 1, "size": 2, "created": 0.0, "done": {}, "finished": None,
                                             "status": "running"}]
    for g in state["buffer"]["ready"][1:]:  # recorded under the dead container's /tmp/arena; the first never synced
        name = Path(g["done"]["folder"]).name
        (out / "arena" / "trials" / name).mkdir(parents=True, exist_ok=True)
        (out / "arena" / "trials" / name / "level-bench.json").write_text('{"attempts": []}')
        g["done"]["folder"] = f"/tmp/arena/trials/{name}"
    lost = state["buffer"]["ready"][0]
    lost["done"]["folder"] = "/tmp/arena/trials/" + Path(lost["done"]["folder"]).name
    (out / "v7_state.json").write_text(json.dumps(state))
    shutil.rmtree(first.work / "adapters", ignore_errors=True)
    second = v7_loop(tmp_path / "b", monkeypatch, [], steps=3, play_delay=0.05)
    monkeypatch.setenv("ARC_RL1_RESUME", str(out))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    folders, select = [], second.mg_select
    second.mg_select = lambda folder, rows, name=None: (folders.append(str(folder)), select(folder, rows, name))[1]
    second.mg_rows = lambda folder: [{"label": f"{Path(folder).name}-a{i}", "cleared": i == 0, "completion_tokens": 10}
                                     for i in range(2)]
    assert second.v7_found("/tmp/arena/trials/x") == "/tmp/arena/trials/x"  # nothing found: as recorded
    assert asyncio.run(second.mg_loop()) == "max_steps"
    start = events(second, "v7_start")[0]
    assert start["resume"] and start["version"] == 2 and start["reads_lost"] == ["2"]
    assert start["lost_ready"] == 1 and start["replayed"] == len(state["running"]) + 1
    assert start["repinned"] == {**{g["gid"]: g["version"] for g in state["running"]}, lost["gid"]: lost["version"]}
    saves = [p for op, p in second.trainer.ops if op == "save"]
    assert saves and saves[0]["path"].endswith("v0002")
    assert {d["version"] for d in events(second, "group_start")} <= {2, 3}
    assert all(m != "policy-r01" for _, m, _ in second.plays)  # the lost version is never played
    assert events(second, "read_lost") and any(r["step"] == 2 for r in events(second, "read"))
    ready = {g["gid"] for g in state["buffer"]["ready"][1:]}
    assert ready and any(second.v7["buffer"].closed.get(gid) == "consumed" for gid in ready)  # oldest first
    seen = [p for op, p in second.trainer.ops if op == "accumulate"]
    assert seen and any(f.startswith(str(out / "arena" / "trials")) for f in folders)  # mapped from /tmp/arena


@pytest.mark.parametrize("how", ["peak", "oom"])
def test_v7_window_guard_falls_back_on_peak_or_oom(tmp_path, monkeypatch, how):
    """mg.window_guard in v7 (hero 40,960 window): an accumulate past peak_gib, or one that ran out
    of memory (no in-call retry in v7), sets lvl_window to the fallback for the groups after it."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2, mg={"window_guard": {"peak_gib": 250.0, "fallback": 32768}})
    loop.cfg["lvl_window"] = 40960

    async def start_trainer(gate=None):
        from test_rl_startup import Trainer

        t = Trainer(log)
        call = t.call
        n = {"acc": 0}

        async def big(op, timeout=3600, **payload):
            out = await call(op, timeout=timeout, **payload)
            if op == "accumulate":
                n["acc"] += 1
                if n["acc"] == 1:
                    return ({**out, "card_peak_gib": 262.0} if how == "peak"
                            else {"ok": False, "error": "CUDA out of memory. Tried to allocate 2.00 GiB"})
            return out

        t.call = big
        loop.trainer = t

    loop.start_trainer = start_trainer
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    fb = events(loop, "window_fallback")
    assert len(fb) == 1 and fb[0]["why"] == how and fb[0]["window"] == 40960 and fb[0]["to"] == 32768
    assert loop.cfg["lvl_window"] == 32768


def test_v7_final_reads_are_cut_at_their_ceiling_and_kept_open(tmp_path, monkeypatch):
    """(hero-v5 A2's slow final reads): the run's last reads wait at most 3x the slowest read it
    measured (floor final_read_floor_s); past that they are cut, kept open for a resume, with an event and an alert."""
    import time as _time

    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=4)
    loop.cfg["mg"]["v7"].update(final_read_s=60.0, final_read_floor_s=0.5)
    real = loop.mg_play

    async def slow_last(name, *a, **kw):
        if name.startswith("x0004-"):  # the last read's play hangs
            await asyncio.sleep(60)
        return await real(name, *a, **kw)

    loop.mg_play = slow_last
    began = _time.time()
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert _time.time() - began < 30
    cut = events(loop, "reads_deadline")
    assert len(cut) == 1 and cut[0]["n"] == 1 and "4" in cut[0]["open"] and cut[0]["slowest_s"] is not None
    assert cut[0]["limit_s"] >= 0.5
    assert any("final reads past" in (a.get("why") or "") for a in events(loop, "alert"))
    reads = events(loop, "read")
    assert reads and all(isinstance(r.get("seconds"), float) for r in reads)


def test_failed_accumulate_resets_and_drops_the_batch(tmp_path, monkeypatch):
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2)
    orig = {}

    async def start_trainer(gate=None):
        from test_rl_startup import Trainer

        t = Trainer(log)
        call = t.call
        n = {"acc": 0}

        async def flaky(op, timeout=3600, **payload):
            out = await call(op, timeout=timeout, **payload)
            if op == "accumulate":
                n["acc"] += 1
                if n["acc"] == 2:
                    return {"ok": False, "stop": "nan_or_inf"}
            return out

        t.call = flaky
        loop.trainer = t
        orig["t"] = t

    loop.start_trainer = start_trainer
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert "acc_reset" in ops(loop)
    assert any(d.get("why") == "v7: accumulate failed; batch dropped" for d in events(loop, "alert"))
    assert [s["step"] for s in events(loop, "step")] == [1, 2]


@pytest.mark.parametrize("start_block", [False, True])
def test_mode_b_block_parks_plays_server_2_and_moves_its_tries(tmp_path, monkeypatch, start_block):
    from arena import level_bench

    monkeypatch.setattr(level_bench, "SERVER2_VERSIONS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=3,
                   mg={"timeshare": {"park_check": True}, "active_tries": 8})
    loop.cfg["mg"]["v7"]["card2"] = {"on": True, "min_block_s": 0.0, "max_block_s": 0.2, "ratio_on": 0.0,
                                     "horizon_s": 100.0, "switch_s": 0.0, "start_block": start_block}

    class Pool:
        def __init__(self):
            self.active, self.served, self.used = False, 0, 3

    class S2:
        def __init__(self):
            self.calls = []

        async def wake(self):
            self.calls.append("wake")
            return 0.1

        async def serve(self, name, path):
            self.calls.append(("serve", name))
            return 0.1

        async def sleep(self):
            self.calls.append("sleep")
            return 0.1

        def stop(self):
            self.calls.append("stop")

    async def ts_boot(sleep=True):
        loop.server2 = S2()
        level_bench.SHARED_SLOTS2 = Pool()
        return {"boot_seconds": 0.1}

    async def ts_park_check(probes):
        return {"ok": True, "identical": True}

    loop.server2 = None
    loop.server.root = "http://127.0.0.1:1234"
    loop.ts_boot, loop.ts_park_check = ts_boot, ts_park_check
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    monkeypatch.setattr(level_bench, "TRANSPORT2", None)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    on, off = events(loop, "block_on"), events(loop, "block_off")
    assert on and len(on) == len(off) and all("error" not in r for r in off)
    o = ops(loop)
    assert o.count("park") == o.count("unpark") == len(on)
    # a later block serves a version (virtual time: whether a block opens after step 0 no longer depends on CPU speed)
    assert any(r["step"] > 0 for r in on) and any(isinstance(c, tuple) and c[0] == "serve" for c in loop.server2.calls)
    assert level_bench.SERVER2_VERSIONS == {on[-1]["step"]}  # server 2 takes only the version served at its block
    assert isinstance(level_bench.SHARED_SLOTS, level_bench.SharedSlots) and level_bench.SHARED_SLOTS.slots == 8

    assert level_bench.TRANSPORT2 is not None and level_bench.TRANSPORT2.redirect  # asleep: tries go to server 1
    assert off[0]["moved_tries"] == 3 and not level_bench.SHARED_SLOTS2.active
    assert (on[0]["step"] == 0) == start_block  # start_block: card 2 plays before the first step


def test_route2_sends_server_2_requests_to_server_1_while_redirected():
    import httpx

    from arena.rl_async import Route2

    seen = []
    r = Route2("http://127.0.0.1:1234")
    r.inner = httpx.MockTransport(lambda req: seen.append(str(req.url)) or httpx.Response(200, json={}))

    async def go():
        async with httpx.AsyncClient(base_url="http://127.0.0.1:1235/", transport=r) as c:
            await c.post("v1/chat/completions", json={})
            r.redirect = True
            await c.post("v1/chat/completions", json={})

    asyncio.run(go())
    assert seen == ["http://127.0.0.1:1235/v1/chat/completions", "http://127.0.0.1:1234/v1/chat/completions"]
    assert r.moved == 1


def test_route2_moved_requests_queue_on_server_1_gate():
    import httpx

    from arena.gate import KVGate
    from arena.rl_async import Route2

    gate = KVGate(10_000, slots=1)
    inflight, peak = [0], [0]

    async def handler(req):
        inflight[0] += 1
        peak[0] = max(peak[0], inflight[0])
        await asyncio.sleep(0.02)
        inflight[0] -= 1
        return httpx.Response(200, json={})

    r = Route2("http://127.0.0.1:1234", gate=lambda: gate)
    r.inner = httpx.MockTransport(handler)
    r.redirect = True

    async def go():
        async with httpx.AsyncClient(base_url="http://127.0.0.1:1235/", transport=r) as c:
            await asyncio.gather(*[c.post("v1/chat/completions", json={"max_tokens": 100}) for _ in range(4)])

    asyncio.run(go())
    assert r.moved == 4 and peak[0] == 1 and gate.inflight == 0 and gate.reserved == 0


def test_route2_hold_keeps_server_2_requests_for_its_next_block():
    """card2.hold: a request made while server 2 sleeps waits and goes to server 2 when it reopens;
    nothing reaches server 1. release() sends a held request to server 1; a hold past hold_max_s moves too."""
    import httpx

    from arena.rl_async import Route2

    seen = []
    r = Route2("http://127.0.0.1:1234")
    r.inner = httpx.MockTransport(lambda req: seen.append(str(req.url)) or httpx.Response(200, json={}))

    async def go():
        async with httpx.AsyncClient(base_url="http://127.0.0.1:1235/", transport=r) as c:
            r.close(True)
            t = asyncio.ensure_future(c.post("v1/chat/completions", json={"model": "policy-r03"}))
            await asyncio.sleep(0.05)
            assert not seen and r.held == 1 and r.held_models == {"policy-r03"}
            r.reopen()
            await t
            r.close(True)
            t2 = asyncio.ensure_future(c.post("v1/chat/completions", json={}))
            await asyncio.sleep(0.05)
            r.release()
            await t2
            r.close(True)
            r.hold_max_s = 0.05
            await c.post("v1/chat/completions", json={})

    asyncio.run(go())
    assert seen == ["http://127.0.0.1:1235/v1/chat/completions", "http://127.0.0.1:1234/v1/chat/completions",
                    "http://127.0.0.1:1234/v1/chat/completions"]
    assert r.hold_timeouts == 1


def test_mode_b_needs_the_shared_slot_pool(tmp_path, monkeypatch):
    import pytest

    loop = v7_loop(tmp_path, monkeypatch, [], steps=1)
    loop.cfg["mg"]["v7"]["card2"] = {"on": True}
    with pytest.raises(ValueError, match="active_tries"):
        asyncio.run(loop.mg_loop())


def test_concurrent_adapter_gc_unloads_each_name_once(tmp_path, monkeypatch):
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=1)
    loop.mg_version = 3
    loop.mg_loaded = {loop.mg_name(1): "a", loop.mg_name(2): "b", loop.mg_name(3): "c"}
    loop.mg_users = {}
    calls = []

    async def unload(name):
        calls.append(name)
        await asyncio.sleep(0.01)

    loop.server.unload = unload

    async def both():
        await asyncio.gather(loop.mg_gc_names(), loop.mg_gc_names())

    asyncio.run(both())
    assert sorted(calls) == [loop.mg_name(1), loop.mg_name(2)] and list(loop.mg_loaded) == [loop.mg_name(3)]

    # a rollback re-serving a name while its unload is in flight: the load lands after the unload
    order = []
    loop.mg_loaded = {loop.mg_name(1): "a", loop.mg_name(3): "c"}

    async def unload2(name):
        order.append(("unload", name))
        await asyncio.sleep(0.05)
        order.append(("unloaded", name))

    async def load(name, path):
        order.append(("load", name))
        return 0.1

    loop.server.unload, loop.server.load = unload2, load

    async def race():
        gc = asyncio.ensure_future(loop.mg_gc_names())
        await asyncio.sleep(0.01)
        await loop.mg_serve("a2", 1)
        await gc

    asyncio.run(race())
    name = loop.mg_name(1)
    assert [o for o in order if o[1] == name] == [("unload", name), ("unloaded", name), ("load", name)]
    assert loop.mg_loaded.get(name) == "a2"  # served again; the old current (r03) is GC'd after it


def test_stop_request_ends_at_a_step_boundary_and_keeps_the_adapter(tmp_path, monkeypatch):
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=5)
    stop = tmp_path / "stop-request"
    monkeypatch.setenv("ARENA_STOP_REQUEST", str(stop))
    orig = loop.v7_step

    async def step_then_ask(step, groups):
        row = await orig(step, groups)
        if step == 2:
            stop.write_text("{}")
        return row

    loop.v7_step = step_then_ask
    adapter = tmp_path / "work" / "adapters" / "v0002"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "x").write_text("1")
    assert asyncio.run(loop.mg_loop()) == "manual"
    assert [s["step"] for s in events(loop, "step")] == [1, 2]
    kept = events(loop, "mg_adapter")
    assert kept and kept[0]["version"] == 2 and (Path(kept[0]["path"]) / "x").is_file()


def test_groups_keep_tries_and_a_replay_takes_the_oldest_version(tmp_path, monkeypatch):
    """Finished tries are kept under partial/v7 (level_bench keep_try); a replayed group that reuses an earlier
    leg's tries carries the oldest try's version (its true staleness)."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "old"))
    keep = loop.v7_keep()["try_keep"]
    assert keep == {"keep": str(tmp_path / "out" / "partial" / "v7"), "reuse": [str(tmp_path / "old" / "partial" / "v7")],
                    "snap_s": 300.0}
    loop.cfg["mg"]["try_keep"] = False
    assert loop.v7_keep() is None
    loop.cfg["mg"]["try_keep"] = True
    seen, fast, flag = [], loop.mg_play, []

    async def play(name, spec, starts, tries, **kw):
        seen.append(kw.get("own"))
        folder, rows, info = await fast(name, spec, starts, tries, **kw)
        if name.startswith("g") and kw.get("model") != "base" and not flag:  # as if both tries came from an
            flag.append(name)  # earlier leg's keep, played on the base
            rows = [{**r, "policy_version": 0} for r in rows]
        return folder, rows, info

    loop.mg_play = play
    asyncio.run(loop.mg_loop())
    assert seen and all(o and o["try_keep"]["keep"].endswith("partial/v7") for o in seen if o is not None)
    reused = events(loop, "group_reused")
    assert flag and reused and reused[0]["oldest"] == 0 and reused[0]["pinned"] > 0 and reused[0]["reused"] == 2


def test_a_mixed_replayed_group_keeps_each_tries_version_and_drops_when_stale(tmp_path, monkeypatch):
    """A replayed group with one kept try from an old version and one new try: each row keeps its own
    policy_version (TIS reads each try's served log-probs), the group's lag is the oldest try's, and past s_max the
    group drops whole like any stale group."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=5, play_delay=0.02)
    loop.cfg["mg"]["v7"]["s_max"] = 1
    fast, mixed, selected = loop.mg_play, [], {}

    async def play(name, spec, starts, tries, **kw):
        folder, rows, info = await fast(name, spec, starts, tries, **kw)
        pinned = int(kw["model"].split("-r")[1]) if kw.get("model", "base") != "base" else 0
        rows = [{**r, "policy_version": pinned} for r in rows]
        if pinned >= 3 and not mixed:  # the first try kept by an earlier leg on the base, the second new
            mixed.append(name.split("-")[0])
            rows[0]["policy_version"] = 0
        return folder, rows, info

    select = loop.mg_select

    def sel(folder, rows, name=None):
        selected[Path(folder).name.split("-")[0]] = [r.get("policy_version") for r in rows]
        return select(folder, rows, name)

    loop.mg_play, loop.mg_select = play, sel
    asyncio.run(loop.mg_loop())
    assert mixed, "no group played on version 3+"
    gid = mixed[0]
    reused = [r for r in events(loop, "group_reused") if r["gid"] == gid]
    assert reused and reused[0]["oldest"] == 0 and reused[0]["pinned"] >= 3
    assert gid not in selected  # never trained: its lag (>= 3) is past s_max 1
    drops = [d for d in loop.v7["buffer"].drops if d["gid"] == gid]
    assert loop.v7["buffer"].closed.get(gid) == "dropped" and drops and drops[0]["lag"] > 1
    assert all(len(set(v)) == 1 for v in selected.values())  # other groups: every try on its own pinned version


def bg_trainer(loop, refuse=False):
    """loop.trainer with rl1's background checkpoint: .partial until the next checkpoint or checkpoint_wait."""
    pend = []
    boot = loop.start_trainer

    async def start(*a, **kw):
        out = await boot(*a, **kw)
        wrap(loop.trainer)
        return out

    loop.start_trainer = start

    def land():
        if not pend:
            return None
        part = pend.pop()
        final = part.with_name(part.name[: -len(".partial")])
        part.rename(final)
        return {"path": str(final), "write_seconds": 0.5}

    def wrap(trainer):
        inner = trainer.call

        async def call(op, timeout=3600, **payload):
            return await bg_call(inner, op, timeout, payload)

        trainer.call = trainer.must = call

    async def bg_call(inner, op, timeout, payload):
        if op == "checkpoint" and payload.get("background"):
            loop.trainer.ops.append((op, payload))
            prior = land()
            if refuse:
                out = await inner(op, timeout=timeout, **{k: x for k, x in payload.items() if k != "background"})
                return {**out, "background_refused": {"why": "host"}}
            part = Path(payload["path"]).with_name(Path(payload["path"]).name + ".partial")
            part.mkdir(parents=True)
            (part / "state.json").write_text("{}")
            pend.append(part)
            return {"ok": True, "path": payload["path"], "background": True, **({"prior": prior} if prior else {})}
        if op == "checkpoint_wait":
            loop.trainer.ops.append((op, payload))
            done = land()
            return {"ok": True, "pending": done is not None, **(done or {})}
        return await inner(op, timeout=timeout, **payload)

    return pend


def test_async_checkpoint_writes_each_state_only_after_its_folder_lands(tmp_path, monkeypatch):
    """mg.async_checkpoint under v7: the step's checkpoint is written in the background; v7_state.json (root and the
    step's copy) follows only once s<step> exists, in step order, never naming a .partial; the run's end joins the
    last write."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=3, mg={"async_checkpoint": True})
    bg_trainer(loop)
    written = []
    put = loop.v7_put_state

    def spy(state):
        assert Path(state["checkpoint"]).is_dir() and not state["checkpoint"].endswith(".partial")
        written.append(state["step"])
        return put(state)

    loop.v7_put_state = spy
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert written == [1, 2, 3]
    steps = events(loop, "step")
    assert all(s.get("checkpoint_background") for s in steps)
    assert [c for c, _ in loop.trainer.ops].count("checkpoint_wait") == 1 and events(loop, "checkpoint_done")
    out = Path(loop.v7_state_dir(3)).parent.parent
    state = json.loads((out / "v7_state.json").read_text())
    assert state["step"] == 3 and json.loads((Path(state["checkpoint"]) / "v7_state.json").read_text())["step"] == 3
    assert not list((out / "state").glob("*.partial"))


def test_async_checkpoint_crash_mid_write_leaves_the_last_landed_state(tmp_path, monkeypatch):
    """A container lost while s<step> is still .partial: v7_state.json still names the previous complete step."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2, mg={"async_checkpoint": True})
    bg_trainer(loop)

    async def no_join():
        return None

    loop.v7_ckpt_join = no_join  # the run ends without the final join (a lost container)
    loop.v7_ckpt_watch = lambda path, every=5.0: asyncio.sleep(0)
    asyncio.run(loop.mg_loop())
    out = Path(loop.v7_state_dir(2)).parent.parent
    state = json.loads((out / "v7_state.json").read_text())
    assert state["step"] == 1 and Path(state["checkpoint"]).is_dir()
    assert (out / "state" / "s0002.partial").is_dir() and not (out / "state" / "s0002").exists()


def test_async_checkpoint_refused_writes_the_state_at_once(tmp_path, monkeypatch):
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2, mg={"async_checkpoint": True})
    bg_trainer(loop, refuse=True)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    steps = events(loop, "step")
    assert all(s.get("checkpoint_refused") and not s.get("checkpoint_background") for s in steps)
    out = Path(loop.v7_state_dir(2)).parent.parent
    assert json.loads((out / "v7_state.json").read_text())["step"] == 2


def test_fake_trainer_background_checkpoint_lands_on_join(tmp_path, monkeypatch):
    """The CPU preflight's stand-in writes a background checkpoint into .partial and renames it later; the next
    checkpoint, checkpoint_wait, load_checkpoint and exit all join it first (as rl1_trainer.ckpt_join)."""
    from arena.fake_trainer import Fake

    monkeypatch.setenv("RL1_FAKE_CKPT_SECONDS", "30")
    f = Fake(folder=tmp_path)
    a, b = tmp_path / "state" / "s0001", tmp_path / "state" / "s0002"
    out = f.run("checkpoint", {"path": str(a), "background": True})
    assert out["background"] and (tmp_path / "state" / "s0001.partial").is_dir() and not a.exists()
    out = f.run("checkpoint", {"path": str(b), "background": True})
    assert out["prior"]["path"] == str(a) and a.is_dir() and json.loads((a / "state.json").read_text())
    assert f.run("checkpoint_wait", {})["path"] == str(b) and b.is_dir()
    assert f.run("checkpoint_wait", {})["pending"] is False
    f.run("checkpoint", {"path": str(tmp_path / "state" / "s0003"), "background": True})
    assert f.run("load_checkpoint", {"path": str(b)})["updates"] == 0 and (tmp_path / "state" / "s0003").is_dir()


def test_v7_pool_takes_kv_project(tmp_path, monkeypatch):
    from arena import level_bench

    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=1, mg={"active_tries": 4, "active_kv_fill": 1.0, "kv_project": True})
    asyncio.run(loop.mg_loop())
    assert level_bench.SHARED_SLOTS.project == level_bench.KV_PROJECT


def test_mode_b_block_ends_once_end_ready_groups_wait(tmp_path, monkeypatch):
    """hero-v3 read: the start block slept its planned 1,760 s while finished groups waited for the parked trainer.
    card2.end_ready ends a block once that many groups are ready (here a 60 s plan ends in well under that)."""
    import time as _time

    from arena import level_bench

    monkeypatch.setattr(level_bench, "SERVER2_VERSIONS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2, mg={"timeshare": {"park_check": False}, "active_tries": 8})
    loop.cfg["mg"]["v7"]["card2"] = {"on": True, "min_block_s": 0.0, "max_block_s": 60.0, "ratio_on": 0.0,
                                     "horizon_s": 60.0, "switch_s": 0.0, "start_block": True, "end_ready": 1}
    loop.cfg["mg"]["v7"]["poll_s"] = 0.05

    class Pool:
        def __init__(self):
            self.active, self.served, self.used = False, 0, 0

    class S2:
        async def wake(self):
            return 0.0

        async def serve(self, name, path):
            return 0.0

        async def sleep(self):
            return 0.0

        def stop(self):
            pass

    async def ts_boot(sleep=True):
        loop.server2 = S2()
        level_bench.SHARED_SLOTS2 = Pool()
        return {"boot_seconds": 0.0}

    loop.server2 = None
    loop.server.root = "http://127.0.0.1:1234"
    loop.ts_boot = ts_boot
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    monkeypatch.setattr(level_bench, "TRANSPORT2", None)
    began = _time.time()
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    off = events(loop, "block_off")
    assert off and all(r.get("ended", "").endswith("groups ready") for r in off)
    assert _time.time() - began < 50  # no block ran its 60 s plan


def test_mode_b_never_blocks_on_a_full_buffer(tmp_path, monkeypatch):
    """xv7b pfA (envs): with max_ready finished groups waiting, a block parks the trainer while no group can be made
    for server 2 (6 blocks served 0); the trainer keeps training instead."""
    import time as _time

    from arena.async_rl import Group

    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=1)
    asyncio.run(loop.mg_loop())  # Mode B off for the run itself; the controller is driven by hand below
    v = loop.v7
    from arena.async_rl import Card2

    v["card2"] = Card2(on=True, min_block_s=0.0, ratio_on=0.0)
    v["card2"].c = 1.0
    v["card2"].observe_group(_time.time() - 10, 2)
    v["card2"].observe_group(_time.time(), 2)
    called = []

    async def block(step, seconds, probes):
        called.append(seconds)

    loop.v7_block = block
    v["buffer"].ready.clear()
    asyncio.run(loop.v7_maybe_block(1, []))
    assert called  # room in the buffer: a block
    for i in range(int(loop.v7_cfg_now["max_ready"])):
        v["buffer"].ready.append(Group(gid=f"z{i}", unit="tr", start=None, version=0, size=2, created=0.0))
    called.clear()
    asyncio.run(loop.v7_maybe_block(2, []))
    assert not called


def test_preflight_shrink_with_mode_b_caps_server_1_slots_at_in_flight(tmp_path, monkeypatch):
    """xv7c pfA (envs): server 2 takes only tries server 1 refuses; 128 slots for a shrink's 8-12 tries never fill, so
    4 blocks served 0. Shrink + card2: server 1 gets in_flight slots and the read cap follows them."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=1, mg={"active_tries": 128})
    A = {**loop.v7_cfg(), "card2": {"on": True}, "in_flight": 8}
    assert loop.v7_slots(A) == 128  # no preflight stub: the configured pool
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", json.dumps({"shrink": 1}))
    assert loop.v7_slots(A) == 8
    assert loop.v7_slots({**A, "card2": None}) == 128  # Mode B off: nothing spills, the pool stays


def test_ahead_names_the_next_groups_specs_and_trains_on_them(tmp_path, monkeypatch):
    """mg.v7.ahead (MFU-1 m1k2): an accumulate (or the k3 pass) carries the specs of the group the trainer takes next;
    that group is later accumulated with exactly those specs (prepared once: its turn credit spent once)."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=3, play_delay=0.2)
    loop.cfg["mg"]["v7"]["ahead"] = True
    loop.cfg["mg"]["v7"]["max_ready"] = 8
    prepared = []
    orig = loop.v7_prepare

    def counting(g):
        if g.gid not in loop.v7.get("prepared", {}):
            prepared.append(g.gid)
        return orig(g)

    loop.v7_prepare = counting
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    calls = [(op, p) for op, p in loop.trainer.ops if op in ("accumulate", "k3_score")]
    acc = [p for op, p in calls if op == "accumulate"]
    named = [(i, [s["name"] for s in p["ahead"]]) for i, (op, p) in enumerate(calls) if p.get("ahead")]
    assert named  # some call ran with a ready group behind it
    checked = 0
    for i, names in named:  # the next accumulate trains exactly the announced specs (none after the last step)
        nxt = next((p for op, p in calls[i + 1:] if op == "accumulate"), None)
        if nxt is not None:
            assert [s["name"] for s in nxt["attempts"]] == names
            checked += 1
    assert checked
    assert len(prepared) == len(set(prepared)) >= len(acc)  # each group prepared once (one may wait past the end)


def test_ahead_off_sends_no_ahead(tmp_path, monkeypatch):
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert not any(p.get("ahead") for op, p in loop.trainer.ops if op in ("accumulate", "k3_score"))


@pytest.mark.parametrize("reload_kl,ok", [(0.02, True), (0.03, False)])
def test_v7_reload_check_after_the_first_step(tmp_path, monkeypatch, reload_kl, ok):
    """mg.reload_check under v7 (the launch gate's in-run smoke): after the first step its checkpoint reloads and the
    probes rescore within the measured repeat floor; a mismatch stops the run."""
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2, mg={"reload_check": True})
    loop.windows = lambda folder, rows, prefix="": [{"name": "p1", "rollout": "x"}]
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(loop.output))
    import test_rl_startup

    orig = test_rl_startup.Trainer.call

    async def call(self, op, timeout=3600, **payload):
        if op == "score" and payload.get("label") == "reload":
            self.ops.append((op, payload))
            return {"ok": True, "windows": {"p1": {"kl_base": reload_kl}}}
        return await orig(self, op, timeout=timeout, **payload)

    monkeypatch.setattr(test_rl_startup.Trainer, "call", call)
    if ok:
        assert asyncio.run(loop.mg_loop()) == "max_steps"
    else:
        with pytest.raises(RuntimeError, match="reload check failed"):
            asyncio.run(loop.mg_loop())
    rows = events(loop, "reload_check")
    assert len(rows) == 1 and rows[0]["round"] == 1 and rows[0]["ok"] is ok
    loads = [p["path"] for op, p in loop.trainer.ops if op == "load_checkpoint"]
    assert loads and loads[-1].endswith("s0001")
    labels = [p.get("label") for op, p in loop.trainer.ops if op == "score"]
    assert labels[0] == "base"  # the base reference precedes every state load and the reload check


def test_v7_reload_check_resume_scores_base_first(tmp_path, monkeypatch):
    """hero-v6f2: a resumed v7 run never scored the probes' base references, so the reload check after its first
    step compared nothing and stopped the run. The base reference now comes before the resume's state load."""
    first = v7_loop(tmp_path, monkeypatch, [], steps=2)
    asyncio.run(first.mg_loop())
    loop2 = v7_loop(tmp_path / "b", monkeypatch, [], steps=4, mg={"reload_check": True})
    loop2.windows = lambda folder, rows, prefix="": [{"name": "p1", "rollout": "x"}]
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "out"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    assert asyncio.run(loop2.mg_loop()) == "max_steps"
    assert events(loop2, "v7_start")[0]["resume"]
    ops = [(op, p.get("label")) for op, p in loop2.trainer.ops if op in ("score", "load_checkpoint")]
    assert ops[0] == ("score", "base") and ("load_checkpoint", None) in ops[1:]
    rows = events(loop2, "reload_check")
    assert len(rows) == 1 and rows[0]["ok"] is True


def test_v7_reload_check_without_probes_fails_at_setup(tmp_path, monkeypatch):
    loop = v7_loop(tmp_path, monkeypatch, [], steps=2, mg={"reload_check": True})
    with pytest.raises(ValueError, match="needs probe windows"):
        asyncio.run(loop.mg_loop())
    assert not [p for op, p in loop.trainer.ops if op == "accumulate"]


def test_manual_stop_cuts_open_reads(tmp_path, monkeypatch):
    """A stop request ends the run at its step boundary without waiting for open reads (hero-v2 CPU legs waited
    45+ min); the reads stay open in the state for a resume to re-read."""
    import time as _time

    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=50, play_delay=0.05)
    stop = tmp_path / "stop-request"
    monkeypatch.setenv("ARENA_STOP_REQUEST", str(stop))
    orig = loop.v7_read

    async def slow_read(*a, **k):
        stop.write_text("{}")  # the request lands while this read is open
        await asyncio.sleep(30)
        return await orig(*a, **k)

    loop.v7_read = slow_read
    began = _time.time()
    assert asyncio.run(loop.mg_loop()) == "manual"
    assert _time.time() - began < 20
    cut = events(loop, "reads_cut")
    assert cut and cut[0]["n"] >= 1


@pytest.mark.parametrize("same", [True, False, "empty"])
def test_card2_continuous_serves_while_the_trainer_waits_mid_batch(tmp_path, monkeypatch, same):
    """with card2.continuous, card 2 serves whenever the trainer waits with
    nothing ready, mid-batch too; the batch's gradients ride the park (keep_grads) and are checked both ways. A changed
    sum of squares drops the batch and turns continuous off."""
    from arena import level_bench

    monkeypatch.setattr(level_bench, "SERVER2_VERSIONS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    log = []
    loop = v7_loop(tmp_path, monkeypatch, log, steps=2, play_delay=0.3,
                   mg={"timeshare": {"park_check": False}, "active_tries": 8})
    loop.cfg["mg"]["v7"]["card2"] = {"on": True, "min_block_s": 1e9, "max_block_s": 0.5, "ratio_on": 0.0,
                                     "switch_s": 0.0, "continuous": True, "idle_min_s": 0.05}
    loop.cfg["mg"]["v7"]["poll_s"] = 0.02
    import test_rl_startup

    inner = test_rl_startup.Trainer.call
    acc = {"calls": 0}

    async def call(self, op, timeout=3600, **payload):
        out = await inner(self, op, timeout=timeout, **payload)
        if op == "accumulate":
            acc["calls"] = 1 if payload.get("first") else acc["calls"] + 1
        if op in ("step", "acc_reset"):
            acc["calls"] = 0
        if op == "park":
            held = payload.get("keep_grads") and acc["calls"] and same != "empty"  # empty: no gradients held yet
            out = {**out, "grads_kept": {"n": 1, "sq": 2.0} if held else None}
        if op == "unpark" and acc["calls"] and same != "empty":
            out = {**out, "grads_restored": {"n": 1, "sq": 2.0 if same else 2.5}}
        return out

    monkeypatch.setattr(test_rl_startup.Trainer, "call", call)
    fast = loop.mg_play
    made = []

    async def play(name, spec, starts, tries, **kw):  # every other group slow: the trainer waits mid-batch
        made.append(name)
        if len(made) % 2 == 0:
            await asyncio.sleep(0.6)
        return await fast(name, spec, starts, tries, **kw)

    loop.mg_play = play

    class Pool:
        def __init__(self):
            self.active, self.served, self.used = False, 0, 0

    class S2:
        async def wake(self):
            return 0.0

        async def serve(self, name, path):
            return 0.0

        async def sleep(self):
            return 0.0

        def stop(self):
            pass

    async def ts_boot(sleep=True):
        loop.server2 = S2()
        level_bench.SHARED_SLOTS2 = Pool()
        return {"boot_seconds": 0.0}

    loop.server2 = None
    loop.server.root = "http://127.0.0.1:1234"
    loop.ts_boot = ts_boot
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    monkeypatch.setattr(level_bench, "TRANSPORT2", None)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    idle = [r for r in events(loop, "block_off") if r.get("idle")]
    assert idle, events(loop, "block_off")
    parks = [p for op, p in loop.trainer.ops if op == "park"]
    mid = [p for p in parks if p.get("keep_grads")]
    bal = events(loop, "balance")
    assert bal and all("card2_serve_s" in r and "trainer_wait_s" in r for r in bal)
    alerts = [a for a in events(loop, "alert") if "continuous" in str(a.get("why"))]
    if same == "empty":  # v6fR: a batch whose groups trained nothing keeps no gradients; no alert
        assert mid and not alerts and loop.v7["card2"].continuous is True
    elif same:
        assert mid and not alerts and sum(r["idle_blocks"] for r in bal) >= 1, (parks, [(k, d.get("gid") or d.get("step")) for k, d in loop.feed.events if k in ("accumulate", "block_on", "block_off", "step", "group_done")])
        assert [s for s in events(loop, "step")][-1]["step"] == 2
    else:
        assert mid and alerts and loop.v7["card2"].continuous is False
        assert any(op == "acc_reset" for op, _ in loop.trainer.ops)


def test_a_lost_base_read_is_re_read_on_version_0(tmp_path, monkeypatch):
    """hero-v6e: v6d stopped at step 1 with its step-0 base read open; the resume re-read it on version 1, so the run
    had no base read for its 2x ratio. A lost read of step 0 (mg.v7.read_base) is re-read on version 0."""
    log = []
    first = v7_loop(tmp_path, monkeypatch, log, steps=1, play_delay=0.05)
    asyncio.run(first.mg_loop())
    out = tmp_path / "out"
    state = json.loads((out / "v7_state.json").read_text())
    state["reads"]["open"] = {"0": {"version": 0, "tries": 2, "done": 0}}
    (out / "v7_state.json").write_text(json.dumps(state))
    second = v7_loop(tmp_path / "b", monkeypatch, [], steps=2, play_delay=0.05)
    second.cfg["mg"]["v7"]["read_base"] = True
    monkeypatch.setenv("ARC_RL1_RESUME", str(out))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    assert asyncio.run(second.mg_loop()) == "max_steps"
    lost = events(second, "read_lost")
    assert lost and lost[0]["base_reread"] is True
    assert any(r["step"] == 0 and r.get("version") == 0 for r in events(second, "read")), events(second, "read")


def test_a_lineage_without_a_landed_base_read_reads_the_base_once(tmp_path, monkeypatch):
    """v6e -> v6f: no read open, but the lineage's base read never landed (base_read_done absent): the resume reads
    version 0 once and records it in the state."""
    log = []
    first = v7_loop(tmp_path, monkeypatch, log, steps=1, play_delay=0.05)
    asyncio.run(first.mg_loop())
    out = tmp_path / "out"
    state = json.loads((out / "v7_state.json").read_text())
    state["reads"]["open"] = {}
    state.pop("base_read_done", None)
    (out / "v7_state.json").write_text(json.dumps(state))
    second = v7_loop(tmp_path / "b", monkeypatch, [], steps=2, play_delay=0.05)
    second.cfg["mg"]["v7"]["read_base"] = True
    monkeypatch.setenv("ARC_RL1_RESUME", str(out))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    assert asyncio.run(second.mg_loop()) == "max_steps"
    assert events(second, "base_reread")
    assert any(r["step"] == 0 and r.get("version") == 0 for r in events(second, "read"))
    assert json.loads((out / "v7_state.json").read_text())["base_read_done"] is True


def test_resume_drops_a_group_already_past_s_max_before_replaying_it(tmp_path, monkeypatch):
    """v6h: replayed groups reuse their kept tries and keep their old version; one already past s_max at the resume
    is dropped up front (v6h played 6 such groups' missing tries, then lag-dropped every replayed group)."""
    first = v7_loop(tmp_path, monkeypatch, [], steps=3, play_delay=0.05)
    asyncio.run(first.mg_loop())
    out = tmp_path / "out"
    state = json.loads((out / "v7_state.json").read_text())
    unit = next(iter(first.cfg["mg_sets"]))
    old = {"unit": unit, "start": None, "size": 2, "created": 0.0, "done": {}, "finished": None, "status": "running"}
    state["running"] = [{**old, "gid": "g09998", "version": 0}, {**old, "gid": "g09999", "version": state["step"]}]
    (out / "v7_state.json").write_text(json.dumps(state))
    second = v7_loop(tmp_path / "b", monkeypatch, [], steps=4, play_delay=0.05)
    second.cfg["mg"]["v7"]["s_max"] = 1
    monkeypatch.setenv("ARC_RL1_RESUME", str(out))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    asyncio.run(second.mg_loop())
    start = events(second, "v7_start")[0]
    assert "g09998" not in start["repinned"] and "g09999" in start["repinned"]
    stale = [d for d in events(second, "group_drop") if d.get("why") == "resume_lag"]
    assert len(stale) == start["dropped_stale"] and all(d["lag"] > 1 for d in stale)
    assert all(state["step"] - v <= 1 for v in start["repinned"].values())
    drop = [d for d in stale if d["gid"] == "g09998"]
    assert drop and drop[0]["lag"] == state["step"]
    assert not [n for n, _, _ in second.plays if n.startswith("g09998")]  # never replayed
    assert [n for n, _, _ in second.plays if n.startswith("g09999")]


def test_groups_pin_their_cap_and_a_replay_plays_at_it_after_the_set_cap_moved(tmp_path, monkeypatch):
    """v6i: running caps moved a set's cap between legs (27K -> 26K); replays took the new cap, so every kept try's
    plan digest changed and 9 of 11 groups replayed from scratch. Groups pin their effective cap at creation."""
    first = v7_loop(tmp_path, monkeypatch, [], steps=2, play_delay=0.05)
    asyncio.run(first.mg_loop())
    assert all(g.cap for g in first.v7["running"].values())  # pinned even when equal to the set's cap
    out = tmp_path / "out"
    state = json.loads((out / "v7_state.json").read_text())
    unit = next(iter(first.cfg["mg_sets"]))
    state["running"] = [{"gid": "g09999", "unit": unit, "start": None, "size": 2, "created": 0.0, "done": {},
                         "finished": None, "status": "running", "version": state["step"], "cap": 27000}]
    # saved settings the leg's must override (v6j: max_loras and s_max came from the saved leg)
    state["book"].update(max_loras=2, serve_every=3, read_slots=0)
    state["buffer"].update(s_max=1, batch_groups=7)
    state["reads"]["every"] = 9
    (out / "v7_state.json").write_text(json.dumps(state))
    second = v7_loop(tmp_path / "b", monkeypatch, [], steps=3, play_delay=0.05)
    caps = {}
    play = second.mg_play

    async def spy(name, spec, starts, tries, **kw):
        caps.setdefault(name.split("-")[0], spec.get("cap"))
        return await play(name, spec, starts, tries, **kw)

    second.mg_play = spy
    monkeypatch.setenv("ARC_RL1_RESUME", str(out))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    asyncio.run(second.mg_loop())
    assert caps.get("g09999") == 27000 and second.cfg["mg_sets"][unit].get("cap") != 27000  # not the set's cap
    assert getattr(second, "_log_mirror", None) is not None  # v7 starts the serving-log mirror (v6j: it never did)
    second._log_mirror.stop.set()
    live, leg = events(second, "v7_start")[0]["live"], second.cfg["mg"]["v7"]  # the leg's settings in force
    for key in ("max_loras", "s_max", "batch_groups", "serve_every", "read_every"):
        assert live[key] == leg[key], key
    assert live["read_slots"] == int(leg.get("read_slots", 1))


def test_an_unpinned_saved_group_replays_at_its_kept_tries_budget(tmp_path, monkeypatch):
    """Groups saved before caps were pinned (cap None): the replay takes the budget its kept tries played at."""
    first = v7_loop(tmp_path, monkeypatch, [], steps=2, play_delay=0.05)
    asyncio.run(first.mg_loop())
    out = tmp_path / "out"
    state = json.loads((out / "v7_state.json").read_text())
    unit = next(iter(first.cfg["mg_sets"]))
    state["running"] = [{"gid": "g09999", "unit": unit, "start": None, "size": 2, "created": 0.0, "done": {},
                         "finished": None, "status": "running", "version": state["step"], "cap": None}]
    (out / "v7_state.json").write_text(json.dumps(state))
    kept = out / "partial" / "v7" / f"g09999-{unit}"
    kept.mkdir(parents=True, exist_ok=True)
    (kept / "tries.jsonl").write_text(json.dumps({"label": "a000", "token_budget": 27000}) + "\n")
    second = v7_loop(tmp_path / "b", monkeypatch, [], steps=3, play_delay=0.05)
    caps, play = {}, second.mg_play

    async def spy(name, spec, starts, tries, **kw):
        caps.setdefault(name.split("-")[0], spec.get("cap"))
        return await play(name, spec, starts, tries, **kw)

    second.mg_play = spy
    monkeypatch.setenv("ARC_RL1_RESUME", str(out))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    asyncio.run(second.mg_loop())
    assert caps.get("g09999") == 27000


def test_v7_reload_check_once_skips_a_passed_trainer_key_on_the_next_leg(tmp_path, monkeypatch):
    """mg.reload_check_once: the first leg checks and marks its trainer key; a resume on the same key skips the
    check (no reload load/score), a new key checks again."""
    monkeypatch.setenv("ARC_RELOAD_CHECK_ROOT", str(tmp_path / "marks"))
    monkeypatch.setenv("ARC_RL1_TRAINER_KEY", "tk1")
    mg = {"reload_check": True, "reload_check_once": True}

    def leg(sub, steps, resume):
        loop = v7_loop(tmp_path / sub, monkeypatch, [], steps=steps, mg=mg)
        loop.windows = lambda folder, rows, prefix="": [{"name": "p1", "rollout": "x"}]
        if resume:
            monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "out"))
        monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
        assert asyncio.run(loop.mg_loop()) == "max_steps"
        return loop

    first = leg("a", 2, False)
    assert len(events(first, "reload_check")) == 1 and len(list((tmp_path / "marks").glob("*.json"))) == 1
    second = leg("b", 4, True)
    assert events(second, "v7_start")[0]["resume"]
    assert not events(second, "reload_check") and len(events(second, "reload_check_skip")) == 1
    assert not [p for op, p in second.trainer.ops if op == "score" and p.get("label") == "reload"]
    monkeypatch.setenv("ARC_RL1_TRAINER_KEY", "tk2")
    third = leg("c", 6, True)
    assert len(events(third, "reload_check")) == 1 and not events(third, "reload_check_skip")


@pytest.mark.parametrize("on", [True, False])
def test_card2_end_on_batch_ends_an_idle_block_under_its_floor_once_the_batch_is_complete(tmp_path, monkeypatch, on):
    """card2.end_on_batch (v6i: the first step waited 4 min on an idle block's 300 s floor with its last groups
    ready): ready groups that complete the batch end the block at once; off, the floor holds."""
    from arena import level_bench

    monkeypatch.setattr(level_bench, "SERVER2_VERSIONS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    loop = v7_loop(tmp_path, monkeypatch, [], steps=2, play_delay=0.3,
                   mg={"timeshare": {"park_check": False}, "active_tries": 8})
    loop.cfg["mg"]["v7"]["card2"] = {"on": True, "min_block_s": 1e9, "max_block_s": 1.5, "ratio_on": 0.0,
                                     "switch_s": 0.0, "continuous": True, "idle_min_s": 0.05,
                                     "idle_min_serve_s": 1.5, "end_on_batch": on}
    loop.cfg["mg"]["v7"]["poll_s"] = 0.02
    fast, made = loop.mg_play, []

    async def play(name, spec, starts, tries, **kw):
        made.append(name)
        if len(made) % 2 == 0:
            await asyncio.sleep(0.4)
        return await fast(name, spec, starts, tries, **kw)

    loop.mg_play = play

    class Pool:
        def __init__(self):
            self.active, self.served, self.used = False, 0, 0

    class S2:
        async def wake(self):
            return 0.0

        async def serve(self, name, path):
            return 0.0

        async def sleep(self):
            return 0.0

        def stop(self):
            pass

    async def ts_boot(sleep=True):
        loop.server2 = S2()
        level_bench.SHARED_SLOTS2 = Pool()
        return {"boot_seconds": 0.0}

    loop.server2, loop.server.root, loop.ts_boot = None, "http://127.0.0.1:1234", ts_boot
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    monkeypatch.setattr(level_bench, "TRANSPORT2", None)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    idle = [r for r in events(loop, "block_off") if r.get("idle")]
    assert idle
    early = [r for r in idle if "complete the batch" in str(r.get("ended"))]
    if on:
        assert early and all(r["seconds"] < 1.4 for r in early)
    else:
        assert not early and all(r["seconds"] >= 1.4 for r in idle)


@pytest.mark.parametrize("max_failures", [1, 3])
def test_mode_b_block_ends_when_server_2_dies_and_boots_a_fresh_one(tmp_path, monkeypatch, max_failures):
    """v6i: server 2 died mid-block and failed every try sent to it until the block ended on 'ready' groups; the
    failure then turned Mode B off for the rest of the run. A dead process ends the block at the next poll, its log
    tail is kept, and below timeshare.max_failures the next block boots a fresh server 2."""
    from arena import level_bench

    monkeypatch.setattr(level_bench, "SERVER2_VERSIONS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    loop = v7_loop(tmp_path, monkeypatch, [], steps=3, play_delay=0.2,
                   mg={"timeshare": {"park_check": False, "max_failures": max_failures}, "active_tries": 8})
    loop.cfg["mg"]["v7"]["card2"] = {"on": True, "min_block_s": 0.0, "max_block_s": 60.0, "ratio_on": 0.0,
                                     "horizon_s": 60.0, "switch_s": 0.0, "start_block": True, "end_ready": 99}
    loop.cfg["mg"]["v7"]["poll_s"] = 0.02
    boots = []

    class Pool:
        def __init__(self):
            self.active, self.served, self.used = False, 0, 0

    class Proc:
        returncode = -9

        def poll(self):
            return self.returncode

    class S2:
        def __init__(self, folder):
            self.folder, self.process, self.stopped = folder, Proc(), False
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "vllm.log").write_text("CUDA error: an illegal memory access\n")

        async def wake(self):
            return 0.0

        async def serve(self, name, path):
            return 0.0

        async def sleep(self):
            return 0.0

        def stop(self):
            self.stopped = True

    async def ts_boot(sleep=True):
        boots.append(S2(tmp_path / f"s2-{len(boots)}"))
        loop.server2 = boots[-1]
        level_bench.SHARED_SLOTS2 = Pool()
        return {"boot_seconds": 0.0}

    loop.server2, loop.server.root, loop.ts_boot = None, "http://127.0.0.1:1234", ts_boot
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    monkeypatch.setattr(level_bench, "TRANSPORT2", None)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    off = events(loop, "block_off")
    assert off and all("exited -9" in r["error"] and r["seconds"] < 5 for r in off)  # not the 60 s plan
    assert all(s.stopped for s in boots) and not level_bench.SHARED_SLOTS2.active
    assert len(boots) == len(off) == min(max_failures, len(off)) and len(off) <= max_failures
    assert "illegal memory access" in Path(off[0]["server2_log"]).read_text()
    alerts = [a["why"] for a in events(loop, "alert") if "Mode B" in a["why"]]
    assert alerts[-1].endswith("off for the rest of the run") == (len(off) == max_failures)
    assert loop.v7["card2"].on == (len(off) < max_failures)


def test_s2_health_counts_failed_and_stalled_probes():
    from arena.rl_async import S2Health, s2_progress

    page = ('# HELP x\nvllm:num_requests_running{model_name="a"} 3.0\nvllm:num_requests_running{model_name="b"} 1.0\n'
            'vllm:generation_tokens_total{model_name="a"} 100.0\nvllm:prompt_tokens_total{model_name="a"} 50.0\n')
    assert s2_progress(page) == (4.0, 150.0)
    h = S2Health(2)
    assert h.update((4.0, 150.0)) is None and h.update((4.0, 150.0)) is None  # one stalled probe
    assert h.update((4.0, 150.0)) == "4 running, no token in a probe interval"  # two in a row
    h = S2Health(2)
    assert h.update("/metrics ReadTimeout") is None and h.update((2.0, 10.0)) is None  # a good probe resets
    assert h.update((0.0, 10.0)) is None and h.update((0.0, 10.0)) is None  # idle, nothing running: healthy
    assert h.update("/metrics ConnectError") is None and h.update("/metrics ReadTimeout") == "/metrics ReadTimeout"


def test_mode_b_block_ends_when_a_live_server_2_stops_making_tokens(tmp_path, monkeypatch):
    """v6h: server 2's process stayed alive, hung in sample_tokens. Probes of /metrics that see running requests and
    no new token end the block on the max_failures path; the next block boots a fresh server 2."""
    from arena import level_bench

    monkeypatch.setattr(level_bench, "SERVER2_VERSIONS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    loop = v7_loop(tmp_path, monkeypatch, [], steps=2, play_delay=0.2,
                   mg={"timeshare": {"park_check": False, "max_failures": 5, "health_every_s": 0.01, "health_misses": 3},
                       "active_tries": 8})
    loop.cfg["mg"]["v7"]["card2"] = {"on": True, "min_block_s": 0.0, "max_block_s": 60.0, "ratio_on": 0.0,
                                     "horizon_s": 60.0, "switch_s": 0.0, "start_block": True, "end_ready": 99}
    loop.cfg["mg"]["v7"]["poll_s"] = 0.02
    boots, probes = [], []

    class Pool:
        def __init__(self):
            self.active, self.served, self.used = False, 0, 0

    class Proc:
        returncode = None

        def poll(self):
            return None

    class S2:
        root = "http://127.0.0.1:1235"

        def __init__(self, folder):
            self.folder, self.process = folder, Proc()

        async def wake(self):
            return 0.0

        async def serve(self, name, path):
            return 0.0

        async def sleep(self):
            return 0.0

        def stop(self):
            pass

    async def probe(timeout):
        probes.append(timeout)
        return (2.0, 500.0) if len(probes) % 4 else "/metrics ReadTimeout"  # stalled: tokens never move

    async def ts_boot(sleep=True):
        boots.append(S2(tmp_path / f"s2-{len(boots)}"))
        loop.server2 = boots[-1]
        level_bench.SHARED_SLOTS2 = Pool()
        return {"boot_seconds": 0.0}

    loop.server2, loop.server.root, loop.ts_boot = None, "http://127.0.0.1:1234", ts_boot
    loop.v7_server2_probe = probe
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    monkeypatch.setattr(level_bench, "TRANSPORT2", None)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    off = events(loop, "block_off")
    assert off and all("server 2 unhealthy" in r["error"] and r["seconds"] < 5 for r in off)
    assert len(boots) == len(off) and loop.v7["card2"].on and probes[0] == 20.0


@pytest.mark.parametrize("on", [True, False])
def test_timeshare_all_versions_loads_every_live_version_on_server_2(tmp_path, monkeypatch, on):
    """timeshare.all_versions (v6i/v6j: most running groups at block open were on an older version and server 2 served
    only the newest, so it took 2-7 new tries a block): each block loads the live versions (newest first, within the
    LoRA slots) so their tries can take server 2's slots on their pinned version."""
    from arena import level_bench

    monkeypatch.setattr(level_bench, "SERVER2_VERSIONS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    loop = v7_loop(tmp_path, monkeypatch, [], steps=4, play_delay=0.3,
                   mg={"timeshare": {"park_check": False, "all_versions": on}, "active_tries": 8})
    loop.cfg["mg"]["v7"]["max_loras"] = 3
    loop.cfg["mg"]["v7"]["card2"] = {"on": True, "min_block_s": 1e9, "max_block_s": 0.3, "ratio_on": 0.0,
                                     "switch_s": 0.0, "continuous": True, "idle_min_s": 0.02}
    loop.cfg["mg"]["v7"]["poll_s"] = 0.02
    seen = []

    class Pool:
        def __init__(self):
            self.active, self.served, self.used = False, 0, 0

    class S2:
        def __init__(self):
            self.loads = []

        async def wake(self):
            return 0.0

        async def serve(self, name, path):
            return 0.0

        async def load(self, name, path):
            self.loads.append(name)
            return 0.0

        async def sleep(self):
            seen.append(set(level_bench.SERVER2_VERSIONS))
            return 0.0

        def stop(self):
            pass

    async def ts_boot(sleep=True):
        loop.server2 = S2()
        level_bench.SHARED_SLOTS2 = Pool()
        return {"boot_seconds": 0.0}

    loop.server2, loop.server.root, loop.ts_boot = None, "http://127.0.0.1:1234", ts_boot
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    monkeypatch.setattr(level_bench, "TRANSPORT2", None)
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    on_rows = events(loop, "block_on")
    extra = [r.get("extra_versions") for r in on_rows]
    if on:
        assert any(extra), extra
        for r, vs in zip(on_rows, seen):
            assert set(r["extra_versions"]) <= vs and len(r["extra_versions"]) + len(r["held_versions"]) <= 2
        assert loop.server2.loads
    else:
        assert not any(extra) and not loop.server2.loads
