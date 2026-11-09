"""card2_turn_cap (MFU-2 25 Oct): server 2's pool lets only its turn_cap earliest-entered live tries send while active;
GateSlot waits for the turn before the KV gate; active_tries2 / card2_turn_cap are live knobs."""

import asyncio
from types import SimpleNamespace

from arena import level_bench, rl_async
from arena.clock import ActiveClock, GlobalClock
from arena.gate import GateSlot, KVGate
from arena.level_bench import SharedSlots
from arena.scheduler import GameRecord


class Slot:  # a live try as the pool sees it (hashable, like GateSlot)
    next_estimate, cap = 0, 0


def test_turn_cap_counts_requests_in_flight_by_entry_order():
    pool = SharedSlots(8)
    slots = [Slot() for _ in range(4)]
    for s in slots:
        pool.used += 1
        pool.enter(s)
    assert all(pool.may_send(s) for s in slots)  # off by default
    pool.turn_cap = 2
    for s in slots[:2]:
        asyncio.run(pool.turn(s))
        assert pool.begin(s)
    assert set(pool.sending) == {id(slots[0]), id(slots[1])} and not pool.may_send(slots[2])  # 2 in flight
    pool.end(slots[1])  # a request ends: the next try may send
    assert pool.may_send(slots[2])
    pool.wanting[id(slots[3])] = pool.entered[id(slots[3])]  # two wait for one place: the earlier-entered goes
    pool.wanting[id(slots[2])] = pool.entered[id(slots[2])]
    assert pool.may_send(slots[2]) and not pool.may_send(slots[3])
    pool.wanting.clear()
    pool.give(slots[0])  # a try that ends frees its place too
    assert pool.sending == {} and pool.may_send(slots[3])
    pool.active = False  # server 2 asleep: the gate does not apply (Route2 holds or moves requests)
    pool.turn_cap, pool.sending = 1, {id(slots[2]): 1}
    assert pool.may_send(slots[3]) and pool.may_send(Slot())


def test_idle_early_tries_hold_no_turn_places():
    """v6l 05:26-05:43: 32 early tries restoring / in tool time held every place of the sticky gate and server 2 ran
    0-1 tries; a later try must send at once while none is in flight."""
    pool = SharedSlots(45)
    pool.turn_cap = 32
    early = [Slot() for _ in range(32)]
    for s in early:
        pool.used += 1
        pool.enter(s)  # in their slots, sending nothing
    late = [Slot() for _ in range(13)]
    for s in late:
        pool.used += 1
        pool.enter(s)
    asyncio.run(asyncio.wait_for(pool.turn(late[-1]), 1))
    assert pool.turn_waits == 0 and pool.sending == {}


async def test_gateslot_waits_for_its_turn_before_the_kv_gate():
    now = [0.0]
    clock = lambda: now[0]
    wall = GlobalClock(1800, now=clock)
    pool = SharedSlots(4)
    pool.turn_cap = 1
    first, second = Slot(), None
    gate = KVGate(10000, slots=4, now=clock)
    games = []
    for name in ("a", "b"):
        g = GameRecord(name, 0, ActiveClock(wall, now=clock))
        g.clock.release()
        games.append(g)
    wall.release()
    second = GateSlot(gate, games[1], cap=100, window=500)
    second.turn_pool = pool
    pool.used += 2
    pool.enter(first)
    pool.enter(second)
    pool.begin(first)  # the first try's request is in flight
    entered = asyncio.create_task(second.__aenter__())
    await asyncio.sleep(0.3)
    assert not entered.done() and gate.inflight == 0 and pool.turn_waits == 1  # waits outside the gate
    pool.end(first)  # the first try's request ends
    await asyncio.wait_for(entered, 2)
    assert gate.inflight == 1
    await second.__aexit__(None, None, None)


def test_live_knobs_set_pool2_slots_and_turn_cap(monkeypatch):
    cls = next(v for v in vars(rl_async).values() if isinstance(v, type) and hasattr(v, "v7_admission_knobs"))
    p1, p2 = SharedSlots(50), SharedSlots(45)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", p1)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", p2)
    rows = []
    host = SimpleNamespace(cfg={"mg": {}}, v7_cfg_now={},
                           feed=SimpleNamespace(emit=lambda kind, **kw: rows.append((kind, kw))),
                           v7={"scheduler": SimpleNamespace(ramp_s=0.0, in_flight_target=100),
                               "card2": SimpleNamespace(continuous=False, idle_end_ready=1, idle_min_serve_s=0.0,
                                                        max_block_s=900.0, idle_min_s=60.0, hold=True,
                                                        idle_ready_wait_s=0.0, end_on_batch=True)})
    host.v7_knob_file = lambda: {"active_tries2": 64, "card2_turn_cap": 40}
    cls.v7_admission_knobs(host, 3)
    assert p2.slots == 64 and p2.turn_cap == 40 and p1.slots == 50
    changed = [kw["changed"] for kind, kw in rows if kind == "knobs"]
    assert {"active_tries2": [45, 64]} in changed and {"card2_turn_cap": [None, 40]} in changed
    host.v7_knob_file = lambda: {"card2_turn_cap": 0}
    cls.v7_admission_knobs(host, 4)
    assert p2.turn_cap is None
    host.v7_knob_file = lambda: {"turn_auto": 0, "turn_cap": 60}
    cls.v7_admission_knobs(host, 5)
    assert host.turn_auto_live is False and p1.turn_cap == 60
    assert {"turn_auto": [True, False]} in [kw["changed"] for kind, kw in rows if kind == "knobs"]


def test_turn_auto_cuts_above_hi_and_reopens_below_lo_with_waiters():
    from arena.level_bench import SharedSlots, turn_auto_step

    p = SharedSlots(70, 0.75)
    for _ in range(60):
        p.enter(Slot())
    spec = {"hi": 0.80, "lo": 0.65, "step": 4, "min": 12, "window_s": 60, "cut_s": 10, "reopen_s": 60}
    assert turn_auto_step(p, 0.70, spec, now=0) is None and p.turn_cap is None  # in band, gate off: nothing
    assert turn_auto_step(p, 0.89, spec, now=10) == 56  # from the live count
    assert turn_auto_step(p, 0.85, spec, now=15) is None  # one cut per cut_s
    assert turn_auto_step(p, 0.85, spec, now=20) == 52
    assert turn_auto_step(p, 0.50, spec, now=40) is None  # low sample: no cut, and the 60 s max still holds 0.85
    assert turn_auto_step(p, 0.50, spec, now=75) is None  # max still 0.85 (t=20)
    assert turn_auto_step(p, 0.50, spec, now=85) == 56  # max now 0.50, 65 s since the last move: reopen one step
    assert turn_auto_step(p, 0.50, spec, now=100) is None  # under reopen_s since that move
    p.turn_cap = 14
    assert turn_auto_step(p, 0.95, spec, now=200) == 12 and turn_auto_step(p, 0.95, spec, now=300) is None  # floor
    assert turn_auto_step(p, None, spec, now=400) is None


def test_turn_auto_does_not_flap_on_noisy_kv(monkeypatch):
    """v6l 05:44-05:47: engine KV 0.12 <-> 0.90 between samples; single-sample control cut and reopened 16 times in
    3 min. On the 60 s max it only cuts while the peak stays high and never reopens inside the window."""
    from arena import level_bench
    from arena.level_bench import SharedSlots, turn_auto_step

    p = SharedSlots(100, 0.75)
    for _ in range(100):
        p.enter(Slot())
    spec = {"hi": 0.80, "lo": 0.65, "step": 4, "min": 12}
    clock = [0.0]
    monkeypatch.setattr(level_bench.time, "time", lambda: clock[0])  # the production call passes no clock
    caps = []
    for kv in [0.84, 0.18, 0.83, 0.12, 0.86, 0.3] * 6:
        caps.append(turn_auto_step(p, kv, spec))
        clock[0] += 5
    moves = [c for c in caps if c is not None]
    assert moves == sorted(moves, reverse=True)  # cuts only, no reopen while noisy peaks stay above hi


def test_server_one_pool_caps_requests_in_flight():
    from arena.level_bench import SharedSlots

    p = SharedSlots(10, None)
    slots = [Slot() for _ in range(5)]
    for s in slots:
        p.enter(s)
    p.turn_cap = 3
    for s in slots[:3]:
        asyncio.run(p.turn(s))
        p.begin(s)
    assert [p.may_send(s) for s in slots] == [True, True, True, False, False]
    p.end(slots[0])
    assert p.may_send(slots[3])


def _gate_games(names, now):
    clock = lambda: now[0]
    wall = GlobalClock(1800, now=clock)
    games = []
    for name in names:
        g = GameRecord(name, 0, ActiveClock(wall, now=clock))
        g.clock.release()
        games.append(g)
    wall.release()
    return KVGate(10000, slots=8, now=clock), games


async def test_parked_try_past_its_turn_holds_no_place():
    """MFU-2 review: a try that passed the turn but waits for its game to be active must not hold a place."""
    now = [0.0]
    gate, (ga, gb) = _gate_games(("a", "b"), now)
    pool = SharedSlots(4)
    pool.turn_cap = 1
    a, b = GateSlot(gate, ga, cap=100, window=500), GateSlot(gate, gb, cap=100, window=500)
    for s in (a, b):
        s.turn_pool = pool
        pool.used += 1
        pool.enter(s)
    ga.clock.pause()
    parked = asyncio.create_task(a.__aenter__())
    await asyncio.sleep(0.1)
    assert not parked.done() and pool.sending == {}
    await asyncio.wait_for(b.__aenter__(), 1)  # b sends while a is parked
    assert pool.sending == {id(b): 1}
    await b.__aexit__(None, None, None)
    parked.cancel()
    await asyncio.gather(parked, return_exceptions=True)
    assert pool.sending == {}


async def test_aside_requests_keep_the_place_until_the_last_ends():
    """MFU-2 review: a try's aside beside its main request; a cancelled waiting aside frees nothing."""
    now = [0.0]
    gate, (ga,) = _gate_games(("a",), now)
    pool = SharedSlots(4)
    pool.turn_cap = 1
    a, other = GateSlot(gate, ga, cap=100, window=500), Slot()
    a.turn_pool = pool
    pool.used += 2
    pool.enter(a)
    pool.enter(other)
    await a.__aenter__()
    await a.__aenter__()  # the aside goes out beside the main request
    assert pool.sending == {id(a): 2}
    await a.__aexit__(None, None, None)
    assert pool.sending == {id(a): 1} and not pool.may_send(other)
    await a.__aexit__(None, None, None)
    assert pool.sending == {}


async def test_place_ends_on_the_pool_that_counted_it():
    """MFU-2 review: route2 moves a try to another pool while its request is out; the place ends where it was taken."""
    now = [0.0]
    gate, (ga,) = _gate_games(("a",), now)
    p1, p2 = SharedSlots(4), SharedSlots(4)
    p1.turn_cap = 2
    a = GateSlot(gate, ga, cap=100, window=500)
    a.turn_pool = p1
    p1.used += 1
    p1.enter(a)
    await a.__aenter__()
    assert p1.sending == {id(a): 1}
    a.turn_pool = p2
    await a.__aexit__(None, None, None)
    assert p1.sending == {} and p2.sending == {}
