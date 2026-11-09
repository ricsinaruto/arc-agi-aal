"""hero-v6 stall (24 Oct): tries held back only by their bench's admission stagger are not a server queue."""
import json
import time

from arena.async_rl import Scheduler
from arena.level_bench import SharedSlots


def pool_with_three_groups(slots=80):
    p = SharedSlots(slots, stagger_s=150.0)
    now = time.time()
    for k in ("g1", "g2", "g3"):
        p.pending[k] = (0, 11)  # one try each admitted just now, 11 held by the stagger
        p.last_take[k] = now
        p.used += 1
    return p


def test_stagger_held_tries_do_not_stop_group_creation():
    p = pool_with_three_groups()
    old = sum(max(0, c) for pr, c in p.pending.values() if pr > -10 ** 9)
    s = Scheduler(group_size=12, max_ready=8, ramp_s=0.0, in_flight_target=110)
    s.ramped = True
    assert s.want(time.time(), old, 36, 0) == 0  # the old count: the stall
    assert p.queued() == 0
    assert s.want(time.time(), p.queued(), 36, 0) == 1


def test_full_pool_counts_every_waiting_try():
    p = pool_with_three_groups(slots=3)
    assert p.queued() == 33


def test_due_bench_counts_and_floor_ignored():
    p = pool_with_three_groups()
    p.last_take["g1"] = time.time() - 200  # its stagger has run out: its tries wait on the pool
    p.pending["explorer"] = (-10 ** 9 - 1, 4)
    assert p.queued() == 11


def test_kv_gate_full_counts():
    p = pool_with_three_groups()
    p.kv_fill, p.capacity = 0.9, 1000
    p.live = {object()}
    p.load = lambda: 950
    p._allowance = lambda: 0
    assert p.queued() == 33
    p.load = lambda: 100
    assert p.queued() == 0


def test_stagger_held_higher_priority_bench_does_not_block():
    """hero-v6b: the held-out read (+1e9) and the highest-cap group, each admitting one try per 150 s, held every other
    bench back."""
    p = SharedSlots(80, stagger_s=150.0)
    now = time.time()
    p.pending["read"] = (10 ** 9, 3)
    p.last_take["read"] = now  # just took: held by its stagger
    p.pending["group"] = (64000, 12)
    assert p.take("group")  # the held read does not block it
    assert not p.take("group")  # its own stagger now holds it
    p.last_take["read"] = now - 200  # the read can take: it goes first
    p.pending["g2"] = (64000, 12)
    assert not p.take("g2") and p.take("read")


def test_compacted_fake_trainer_refuses_off_the_fused_path(monkeypatch, tmp_path):
    from arena import fake_trainer, rl_loop

    monkeypatch.setenv("RL1_EXPERT_COMPACT", "1")
    cls = next(v for v in vars(fake_trainer).values() if isinstance(v, type) and hasattr(v, "FORWARD_OPS"))
    obj = cls.__new__(cls)
    with __import__("pytest").raises(RuntimeError, match="fused expert path"):
        obj.run("park_check", {"op": "park_check"})
    host = type("H", (), {})()
    host.cfg = {"trainer_env": {"RL1_EXPERT_COMPACT": "1"}, "mg": {"trainer_fast": {"fused_experts": True, "k3_hold": 16}}}
    host.trainer = type("T", (), {})()
    cls_loop = next(v for v in vars(rl_loop).values() if isinstance(v, type) and hasattr(v, "compact_always"))
    cls_loop.compact_always(host)
    assert host.trainer.always == {"fused_experts": True, "k3_hold": 16}
    host.cfg["trainer_env"] = {}
    host.trainer.always = None
    cls_loop.compact_always(host)
    assert host.trainer.always is None


def test_ramp_and_stagger_live_knobs(monkeypatch):
    from arena import level_bench, rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_knobs"))
    host = type("H", (), {})()
    host.cfg = {"mg": {"trainer_fast": {}}}
    host.mg_knobs = lambda step: None
    host.trainer = None
    host.v7_cfg_now = {"ramp_s": 90.0}
    host.v7 = {"scheduler": Scheduler(group_size=12, max_ready=8, ramp_s=90.0, in_flight_target=110), "reads": None}
    rows = []
    host.feed = type("F", (), {"emit": lambda self, kind, **kw: kind == "knobs_file" or rows.append((kind, kw))})()
    host.v7_admission_knobs = lambda step: cls.v7_admission_knobs(host, step)
    host.v7_knob_file = lambda: {"ramp_s": 30, "admit_stagger_s": 50, "in_flight": 200, "in_flight_mode": "unfinished"}
    p1, p2 = SharedSlots(80, stagger_s=150.0), SharedSlots(48, stagger_s=150.0)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", p1)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", p2)
    cls.v7_knobs(host, 3)
    assert host.v7["scheduler"].ramp_s == 30.0 and p1.stagger_s == p2.stagger_s == 50.0
    assert host.cfg["mg"]["admit_stagger_s"] == 50.0 and len(rows) == 4
    assert host.v7["scheduler"].in_flight_target == 200 and host.v7_cfg_now["in_flight"] == 200
    assert host.v7_cfg_now["in_flight_mode"] == "unfinished"
    cls.v7_knobs(host, 4)  # unchanged values: no new rows
    assert len(rows) == 4


def test_scheduler_loop_applies_admission_knobs_between_steps(monkeypatch):
    """admission knobs (in_flight, kv fill, active tries, stagger) apply within knob_poll_s from
    the scheduler loop, without a step boundary or restart; a broken knobs file alerts and play goes on."""
    import asyncio
    from types import SimpleNamespace

    from arena import level_bench, rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_admission_knobs"))
    p1, p2 = SharedSlots(80, stagger_s=150.0), SharedSlots(128, stagger_s=150.0)
    p1.kv_fill = p2.kv_fill = 0.6
    p1.project, p2.project = {"group_prefix": True}, {"group_prefix": True}
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", p1)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", p2)
    rows = []
    files = [{"in_flight": 400, "active_kv_fill": 0.8, "active_tries": 96, "admit_stagger_s": 20, "kv_engine_scale": 1.2,
              "admit_stagger2_s": 0, "card2_idle_min_serve_s": 300, "card2_idle_end_ready": 2},
             ValueError("bad")]
    sched = Scheduler(group_size=12, max_ready=8, ramp_s=1e9, in_flight_target=110)
    host = SimpleNamespace(cfg={"mg": {}}, v7_cfg_now={"poll_s": 0.01, "knob_poll_s": 0.0},
                           feed=SimpleNamespace(emit=lambda kind, **kw: kind == "knobs_file" or rows.append((kind, kw))),
                           v7={"stop": False, "running": {}, "buffer": SimpleNamespace(ready=[]), "scheduler": sched,
                               "book": SimpleNamespace(served=7),
                               "card2": SimpleNamespace(idle_end_ready=1, idle_min_serve_s=0.0, max_block_s=900.0, idle_min_s=60.0)})

    def knob_file():
        got = files.pop(0) if len(files) > 1 else files[0]
        if isinstance(got, Exception):
            host.v7["stop"] = True
            raise got
        return got

    host.v7_knob_file = knob_file
    host.v7_make_group = lambda sets, pool: None
    host.v7_unfinished = lambda g: 0
    host.v7_admission_knobs = lambda step: cls.v7_admission_knobs(host, step)
    asyncio.run(asyncio.wait_for(cls.v7_scheduler(host, None, None), 5))
    assert sched.in_flight_target == 400 and p1.kv_fill == 0.8 and p2.kv_fill == 0.6
    assert p1.slots == 96 and p1.stagger_s == 20.0 and p2.stagger_s == 0.0
    assert host.v7["card2"].idle_min_serve_s == 300.0 and host.v7["card2"].idle_end_ready == 2
    assert p1.project["engine_scale"] == p2.project["engine_scale"] == 1.2
    assert {kw.get("round") for kind, kw in rows if kind == "knobs"} == {7}
    assert any(kind == "alert" and "live knobs" in kw["why"] for kind, kw in rows)


def test_unfinished_in_flight_lets_straggler_groups_stop_blocking_creation(monkeypatch):
    """hero-v6d 10:47: 12 running groups of 12, 135 of 144 tries finished, 9 live: planned counting (144 >= 110 + 12)
    makes no group; unfinished counting (9) makes one, and max_ready still stops it at 8 ready groups."""
    from types import SimpleNamespace

    from arena import level_bench, rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_unfinished"))
    done = [12, 12, 11, 12, 12, 11, 12, 11, 12, 10, 10, 10]  # 135 finished, 9 unfinished
    groups = [SimpleNamespace(gid=f"g{i:05d}", unit="x00b-L2", size=12) for i in range(1, 13)]
    monkeypatch.setattr(level_bench, "BENCH_DONE", {f"{g.gid}-{g.unit}": d for g, d in zip(groups, done)})
    host = SimpleNamespace(v7_cfg_now={"in_flight_mode": "planned"})
    sched = Scheduler(group_size=12, max_ready=8, ramp_s=30.0, in_flight_target=110)
    sched.ramped = True
    planned = sum(cls.v7_unfinished(host, g) for g in groups)
    assert planned == 144 and not sched.want(0.0, 0, planned, 0)
    host.v7_cfg_now["in_flight_mode"] = "unfinished"
    unfinished = sum(cls.v7_unfinished(host, g) for g in groups)
    assert unfinished == 9 and sched.want(0.0, 0, unfinished, 0)
    assert not sched.want(0.0, 0, unfinished, 8)  # max_ready bounds staleness
    assert cls.v7_unfinished(host, SimpleNamespace(gid="g00099", unit="x", size=12)) == 12  # a new group: all unfinished


def test_ckpt_join_when_the_watcher_flushed_during_the_wait(tmp_path, monkeypatch):
    """v6e S leg: v7_ckpt_watch wrote the pending state while v7_ckpt_join awaited checkpoint_wait; the join then
    read a None pending and crashed the run's end."""
    import asyncio
    from types import SimpleNamespace

    from arena import rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_ckpt_join"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    ckpt = tmp_path / "s0001"
    ckpt.mkdir()
    rows = []
    host = SimpleNamespace(v7={"ckpt_pending": {"step": 1, "checkpoint": str(ckpt)}},
                           feed=SimpleNamespace(emit=lambda kind, **kw: rows.append(kind)))
    host.v7_put_state = lambda state: None
    host.v7_ckpt_flush = lambda: cls.v7_ckpt_flush(host)

    async def wait(op, **kw):
        host.v7_ckpt_flush()  # the watcher lands it meanwhile
        return {"ok": True, "seconds": 1.0}

    host.trainer = SimpleNamespace(must=wait)
    asyncio.run(cls.v7_ckpt_join(host))
    assert "alert" not in rows and rows[-1] == "checkpoint_done"


def test_engine_scale_scales_the_group_prefix_projection():
    """RL 24 Oct: the group_prefix projection counts client tokens; engine_scale converts them to engine KV (v6e
    projected 0.886 while the engine reached 0.997). Default 1.0 keeps the old gate."""
    class Try:
        next_estimate, floor_tokens, window, cap, prefix_key = 40_000, 30_000, 131072, 16384, "g"

        def progress(self):
            return 0, 0

    def pool(scale):
        p = SharedSlots(80, 0.8, project={"group_prefix": True, **({"engine_scale": scale} if scale else {})})
        p.capacity = 1_000_000
        p.enter(Try())
        return p

    assert pool(None).load() == 40_000
    assert abs(pool(1.35).load() - 54_000) < 1e-6
    p = pool(None)
    p.scale_sample(0.08)  # 80K engine tokens over 40K client tokens of every live try
    assert getattr(p, "scale_all", [None])[-1] in (None, 2.0)


def test_lr_fixed_knob_holds_the_rate_over_the_warm_up():
    from types import SimpleNamespace

    from arena import rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_lr"))
    host = SimpleNamespace(cfg={"lr": 1e-4}, v7_cfg_now={"warmup": 4}, v7={"since_resume": 1})
    host.v7_knob_file = lambda: {}
    assert abs(cls.v7_lr(host) - 5e-5) < 1e-12
    host.v7_knob_file = lambda: {"lr_fixed": 2.5e-5}
    assert cls.v7_lr(host) == 2.5e-5


def test_curriculum_live_knobs(monkeypatch):
    """Curriculum settings from the knobs file (``curriculum``: allowed keys only), one row per change."""
    from arena import level_bench, rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_knobs"))
    host = type("H", (), {})()
    host.cfg = {"mg": {}}
    host.v7_cfg_now = {}
    host.v7 = {"scheduler": Scheduler(group_size=12, max_ready=8, ramp_s=90.0, in_flight_target=110)}
    host.mg_curr = type("C", (), {"cfg": {"retire_p": 1.01, "token_share": 1.0}})()
    rows = []
    host.feed = type("F", (), {"emit": lambda self, kind, **kw: kind == "knobs_file" or rows.append((kind, kw))})()
    host.v7_knob_file = lambda: {"curriculum": {"retire_p": 0.9, "token_share": 1.0, "groups": 99}}
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    cls.v7_admission_knobs(host, 3)
    assert host.mg_curr.cfg == {"retire_p": 0.9, "token_share": 1.0}
    assert [kw["changed"] for _, kw in rows] == [{"curriculum.retire_p": [1.01, 0.9]}]
    cls.v7_admission_knobs(host, 4)
    assert len(rows) == 1


def test_v7_launcher_round_ceiling_sets_max_steps(monkeypatch):
    from arena import rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_cfg"))
    host = type("H", (), {})()
    host.cfg = {"mg": {"v7": {"max_steps": 1000}}}
    monkeypatch.delenv("ARC_RL1_MAX_STEPS", raising=False)
    monkeypatch.setenv("ARC_RL1_MAX_ROUNDS", "3")
    assert cls.v7_cfg(host)["max_steps"] == 3
    monkeypatch.setenv("ARC_RL1_MAX_STEPS", "5")
    assert cls.v7_cfg(host)["max_steps"] == 5


def test_gigpo_omega_live_knob(monkeypatch):
    from arena import level_bench, rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_knobs"))
    host = type("H", (), {})()
    host.cfg = {"mg": {"turn_credit": {"mode": "gigpo", "omega": 1.0}}}
    host.v7_cfg_now = {}
    host.v7 = {"scheduler": Scheduler(group_size=12, max_ready=8, ramp_s=90.0, in_flight_target=110)}
    rows = []
    host.feed = type("F", (), {"emit": lambda self, kind, **kw: kind == "knobs_file" or rows.append((kind, kw))})()
    host.v7_knob_file = lambda: {"gigpo_omega": 0.3}
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    cls.v7_admission_knobs(host, 3)
    cls.v7_admission_knobs(host, 4)
    assert host.cfg["mg"]["turn_credit"]["omega"] == 0.3 and len(rows) == 1


def test_oldest_group_takes_free_slots_first():
    from arena import rl_async
    from arena.level_bench import bench_priority

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_own"))
    host = type("H", (), {})()
    host.cfg = {"mg": {"try_keep": False}}
    host.v7_cfg_now = {"oldest_first": True}
    host.v7_keep = lambda: None
    group = lambda gid: type("G", (), {"gid": gid})()
    budgets = {"token_budget_by_point": {"x-game": 60000}}
    old, new = (bench_priority(budgets, cls.v7_own(host, group(g)) or {}) for g in ("g00031", "g00046"))
    read = bench_priority(budgets, {})
    assert read > old > new > -10 ** 9
    p = SharedSlots(4)
    p.pending.update({"new": (new, 3), "old": (old, 2), "read": (read, 1)})
    assert not p.take("old") and p.take("read") and not p.take("new")
    assert p.take("old") and p.take("old") and p.take("new")
    host.v7_cfg_now = {"oldest_first": False}
    assert cls.v7_own(host, group("g00031")) is None


def test_v7_v0_scan_reads_kept_version_0_clears(tmp_path, monkeypatch):
    from arena import rl_async

    from arena import rl_single

    mix = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_v0_scan"))
    single = next(v for v in vars(rl_single).values() if isinstance(v, type) and hasattr(v, "mg_v0_add"))
    cls = type("H", (mix, single), {})
    keep, old = tmp_path / "out" / "partial" / "v7", tmp_path / "leg1" / "partial" / "v7"
    rows = lambda pv: "\n".join(json.dumps({"label": f"a{i}-p{i}", "outcome": "cleared", "policy_version": pv,
                                            "level_rows": [{"level": 1, "tokens": t, "actions": 20}]})
                                for i, t in enumerate([1000, 3000])) + "\nnot json\n"
    for root, gid, pv in ((old, "g00001-y-L1", 0), (old, "g00002-z-L1", 1), (keep, "g00009-y-L1", 0)):
        (root / gid).mkdir(parents=True)
        (root / gid / "tries.jsonl").write_text(rows(pv))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "leg1"))
    host = cls.__new__(cls)
    host.cfg = {"mg": {"ref_fallback": "v0"}, "mg_sets": {"y-L1": {}, "z-L1": {}}}
    got = []
    host.feed = type("F", (), {"emit": lambda self, kind, **kw: got.append((kind, kw))})()
    import asyncio

    async def body(probes=None):  # the scan runs on a thread beside the setup body and lands before it returns
        assert got == []
        return 7

    host.v7_setup_body = body
    assert asyncio.run(host.v7_setup()) == 7
    assert got == [("ref_fallback", {"mode": "v0", "groups": 3, "units": 1, "usable": 1})]
    assert host.mg_ref_fallback("y-L1") == {"median_to_clear": 2000, "median_actions": 20}
    assert host.mg_ref_fallback("z-L1") is None  # version 1 tries never count


def test_read_share_knob_resizes_the_open_reads_gate(monkeypatch):
    """read_share is live (v6g's base read held 8 tries for 2 h): the open gate takes the new
    width at once."""
    import asyncio

    from arena import level_bench, rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_read_gate_resize"))
    host = cls.__new__(cls)
    host.cfg = {"mg": {"active_tries": 80}}
    host.v7_cfg_now = {"read_share": 0.1}
    rows = []
    host.feed = type("F", (), {"emit": lambda self, kind, **kw: kind == "knobs_file" or rows.append((kind, kw))})()
    host.v7_knob_file = lambda: {"read_share": 0.25}
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)

    async def go():
        host.v7 = {"scheduler": Scheduler(group_size=12, max_ready=8, ramp_s=0.0, in_flight_target=110),
                   "read_gate": {"cap": 8, "free": 0, "cond": asyncio.Condition()}}
        host.v7_admission_knobs(5)
        await asyncio.sleep(0)
        return host.v7["read_gate"]

    gate = asyncio.run(go())
    assert gate["cap"] == 20 and gate["free"] == 12 and host.v7_cfg_now["read_share"] == 0.25
    assert rows == [("knobs", {"round": 5, "changed": {"read_share": [0.1, 0.25]}})]


def test_s_max_knob_reaches_the_buffer_and_a_lag_drop_reports_its_tokens(monkeypatch):
    """s_max is live (v6i step 11: 3 lag drops at s_max 2); drops carry the tokens thrown away."""
    from arena import level_bench, rl_async
    from arena.async_rl import Book, Buffer, Group

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_read_gate_resize"))
    host = cls.__new__(cls)
    host.cfg = {"mg": {}}
    host.v7_cfg_now = {"s_max": 2}
    rows = []
    host.feed = type("F", (), {"emit": lambda self, kind, **kw: kind == "knobs_file" or rows.append((kind, kw))})()
    host.v7_knob_file = lambda: {"s_max": 3}
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    host.v7 = {"buffer": Buffer(2, 1)}
    host.v7_admission_knobs(7)
    assert host.v7["buffer"].s_max == 3 and ("knobs", {"round": 7, "changed": {"s_max": [2, 3]}}) in rows
    book = Book(max_loras=8, version=5)
    g = Group(gid="g1", unit="u-L2", start=None, version=1, size=2, created=0.0, tokens=41000)
    host.v7["buffer"].add(g)
    assert host.v7["buffer"].next_group(book, 0.0) == (None, None)
    assert host.v7["buffer"].drops[-1]["tokens"] == 41000


def test_pool_state_reports_slots_queue_scale_and_projection():
    from arena.level_bench import pool_state

    p = SharedSlots(80, 0.75, project={"group_prefix": True, "engine_scale": 0.6})
    row = pool_state(p)
    assert row["slots"] == 80 and row["used"] == 0 and row["queued"] == 0 and row["room"] is True
    assert row["engine_scale"] == 0.6 and row["auto_off"] is False and row["kv_fill"] == 0.75
    assert "projected" not in row  # no capacity yet
    assert pool_state(SharedSlots(48))["slots"] == 48


def test_max_ready_knob_reaches_the_scheduler(monkeypatch):
    """max_ready is live (v6j resume: reused replays filled the ready buffer and held group creation)."""
    from arena import level_bench, rl_async

    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_read_gate_resize"))
    host = cls.__new__(cls)
    host.cfg, host.v7_cfg_now = {"mg": {}}, {"max_ready": 8}
    host.feed = type("F", (), {"emit": lambda self, kind, **kw: None})()
    host.v7_knob_file = lambda: {"max_ready": 12}
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    host.v7 = {"scheduler": Scheduler(group_size=12, max_ready=8, ramp_s=0.0, in_flight_target=110)}
    host.v7_admission_knobs(3)
    assert host.v7["scheduler"].max_ready == 12 and host.v7_cfg_now["max_ready"] == 12
    assert host.v7["scheduler"].want(0.0, 0, 0, 10) == 1


def test_preflight_shrink_caps_final_reads_under_the_stall_killer(monkeypatch):
    """v6j pfR2/R3: final reads (4,500 s ceiling) outwaited the 900 s stall killer; shrink caps them at 300 s."""
    import json

    from arena import rl_async

    host = rl_async.AsyncMixin.__new__(rl_async.AsyncMixin)
    host.cfg = {"mg": {"v7": {}}}
    monkeypatch.delenv("ARC_RL1_MAX_STEPS", raising=False)
    monkeypatch.delenv("ARC_RL1_MAX_ROUNDS", raising=False)
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", json.dumps({"shrink": True}))
    cfg = host.v7_cfg()
    assert cfg["final_read_s"] == 300.0 and cfg["final_read_floor_s"] == 0.0
