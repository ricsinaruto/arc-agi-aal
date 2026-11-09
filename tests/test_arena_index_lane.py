"""Background lane (fixed-2x-index-lane): games outside the focus slots request on leftover capacity,
never into a focus game's guard; focus requests go first; lane order; lane clears feed the index."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from test_arena_hazard_index import build, charged

from arena.gate import BackgroundLane, GateSlot, KVGate, lane_settings
from arena.scheduler import Scheduler
from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
WINDOW, CAP = 400, 50


def steady(count=6, **lane):
    """Index scheduler past the opening: g0 and g1 focus, the rest parked in the lane."""
    now, wall, games, sched = build("index", count=count, index_settings={}, background_lane=lane or {})
    for g in games:
        charged(now, g, 10, 3000)
    games[0].report_level(1, wall.elapsed())
    games[1].report_level(1, wall.elapsed())
    sched.tick()
    assert [g.state for g in games[:2]] == ["active", "active"]
    assert all(g.state == "parked" and g.lane for g in games[2:])
    return now, wall, games, sched


def gate_for(games, capacity, now, **lane):
    log = []
    gate = KVGate(capacity, slots=25, log=log.append, now=now,
                  lane=BackgroundLane(games, window=WINDOW, cap=CAP, **lane))
    return gate, log


async def test_lane_fills_idle_capacity_beside_small_focus_requests_and_focus_never_waits():
    now, wall, games, sched = steady()
    g0, g1, g2, g3, g4, _ = games
    g0.latest_prompt_tokens = g1.latest_prompt_tokens = 100  # guards: min(400, 100 + 2 * 50) = 200 each
    gate, log = gate_for(games, 800, now)
    assert gate.guard_deficit() == 400
    # Parked lane games do not wait for a focus slot: their slots enter at once on idle capacity.
    slot2, slot3 = GateSlot(gate, g2, cap=CAP, window=WINDOW), GateSlot(gate, g3, cap=CAP, window=WINDOW)
    slot2.reserve(100)
    slot3.reserve(100)
    await asyncio.wait_for(slot2.__aenter__(), 1)
    await asyncio.wait_for(slot3.__aenter__(), 1)
    assert gate.lane_inflight == 2 and gate.reserved == 300 and g2.clock.parked
    # A third lane request waits: two lane requests at most, whatever the free capacity.
    third = asyncio.create_task(gate.acquire("g4", 60))
    await asyncio.sleep(0)
    assert not third.done() and gate.reserved + 60 + gate.guard_deficit() <= gate.capacity
    assert gate.snapshot()["lane"]["lane_waiting"] == 1
    # Both focus games' next requests (within their guards) go in at once beside the full lane.
    r0 = await gate.acquire("g0", 200)
    r1 = await gate.acquire("g1", 200)
    assert gate.reserved == 700 and not r0.lane and not r1.lane
    stats = gate.snapshot()["lane"]
    assert stats["focus_wait_seconds"] == 0 and stats["focus_waits"] == 0 and stats["focus_admitted"] == 2
    assert stats["lane_admitted"] == 2 and stats["peak_lane_inflight"] == 2
    now.advance(5)
    await slot2.__aexit__(None, None, None)  # the lane request ends; the queued one takes its place
    admitted = await asyncio.wait_for(third, 1)
    assert admitted.lane and gate.lane_inflight == 2
    assert [e["game"] for e in log if e["event"] == "admit"] == ["g2", "g3", "g0", "g1", "g4"]
    assert g2.requests[-1]["lane"] is True and g2.lane_requests == 1
    snap = gate.snapshot()["lane"]
    assert snap["inflight_seconds"]["focus=2,lane=2"] == pytest.approx(5)
    for r in (r0, r1, admitted):
        gate.release(r)
    await slot3.__aexit__(None, None, None)
    assert gate.reserved == gate.inflight == gate.lane_inflight == 0 and not gate.held


async def test_lane_request_never_eats_into_a_focus_guard():
    now, wall, games, sched = steady()
    g0, g1, g2 = games[:3]
    g0.latest_prompt_tokens = g1.latest_prompt_tokens = 300  # guards: min(400, 300 + 100) = 400 each
    gate, _ = gate_for(games, 900, now)
    # Plain capacity has room (0 + 150 <= 900), but 150 + 800 of guards does not fit.
    lane = asyncio.create_task(gate.acquire("g2", 150))
    await asyncio.sleep(0)
    assert not lane.done() and gate.guard_deficit() == 800
    # A focus game's in-flight request is part of its guard, not in addition to it.
    r0 = await gate.acquire("g0", 350)
    assert gate.guard_deficit() == 50 + 400
    gate.refresh()
    await asyncio.sleep(0)
    assert not lane.done()  # 350 + 150 + 450 > 900
    # After a compaction the focus game's guard shrinks; the next tick's refresh admits the lane.
    g1.latest_prompt_tokens = 50
    gate.refresh()
    admitted = await asyncio.wait_for(lane, 1)
    assert admitted.lane and gate.reserved + gate.guard_deficit() <= gate.capacity
    # The invariant makes the guarded focus request go in at once.
    gate.release(r0)
    r0 = await gate.acquire("g0", 400)
    assert gate.snapshot()["lane"]["focus_wait_seconds"] == 0
    for r in (r0, admitted):
        gate.release(r)
    # The same arithmetic in cache blocks: the guard is the block cost of the guarded length.
    from test_arena_cache import layout

    from arena.cache import BlockBudget

    budget = BlockBudget(layout(), margin=1.0)
    blocks = KVGate(budget.capacity, slots=25, budget=budget, now=now,
                    lane=BackgroundLane(games, window=2048, cap=512))
    g0.latest_prompt_tokens, g1.latest_prompt_tokens = 500, 1500
    assert blocks.guard_deficit() == budget.cost(1524) + budget.cost(2048)


async def test_focus_requests_jump_queued_lane_requests():
    now, wall, games, sched = steady()
    g0, g1, g2 = games[:3]
    g0.latest_prompt_tokens = g1.latest_prompt_tokens = 100
    gate, log = gate_for(games, 600, now)
    holder = await gate.acquire("g0", 500)
    lane = asyncio.create_task(gate.acquire("g2", 150))  # queued first
    await asyncio.sleep(0)
    now.advance(3)
    focus = asyncio.create_task(gate.acquire("g1", 200))  # queued second: 500 + 200 > 600
    await asyncio.sleep(0)
    assert not lane.done() and not focus.done()
    now.advance(2)
    gate.release(holder)
    r1, r2 = await asyncio.wait_for(asyncio.gather(focus, lane), 1)
    assert [e["game"] for e in log if e["event"] == "admit"] == ["g0", "g1", "g2"]
    assert not r1.lane and r2.lane
    stats = gate.snapshot()["lane"]
    assert stats["focus_wait_seconds"] == pytest.approx(2) and stats["lane_wait_seconds"] == pytest.approx(5)
    # No lane request goes while a focus request waits, even when the lane rule alone would pass.
    gate.release(r2)
    blocked = asyncio.create_task(gate.acquire("g0", 450))  # beyond g0's guard: 200 + 450 > 600
    await asyncio.sleep(0)
    lane = asyncio.create_task(gate.acquire("g2", 10))  # 200 + 10 + g0's guard 200 would fit
    await asyncio.sleep(0)
    gate.refresh()
    await asyncio.sleep(0)
    assert not blocked.done() and not lane.done()
    gate.release(r1)
    r0 = await asyncio.wait_for(blocked, 1)
    assert not lane.done()  # 450 + 10 + g1's guard 200 > 600
    gate.release(r0)
    gate.release(await asyncio.wait_for(lane, 1))
    assert gate.reserved == gate.inflight == 0


async def test_lane_order_is_fewest_tokens_on_the_level_then_roster():
    now, wall, games, sched = steady(count=6, max_inflight=1)
    g0, g1, g2, g3, g4, g5 = games
    g0.latest_prompt_tokens = g1.latest_prompt_tokens = 100
    charged(now, g2, 10, 9000)
    charged(now, g3, 10, 500)
    charged(now, g4, 10, 500)  # g3 and g4 tie at 3,500 tokens on level 0 (roster order); g5 has 3,000
    gate, log = gate_for(games, 1000, now, max_inflight=1)
    admitted = []
    first = await gate.acquire("g2", 100)  # alone, it goes at once
    waiters = [asyncio.create_task(gate.acquire(g, 100)) for g in ("g4", "g3", "g5")]
    await asyncio.sleep(0)
    assert not any(w.done() for w in waiters)
    gate.release(first)
    for _ in waiters:
        await asyncio.sleep(0)
        done = [w for w in waiters if w.done() and w not in admitted]
        assert len(done) == 1  # one lane request at a time (max_inflight=1)
        admitted.append(done[0])
        gate.release(done[0].result())
    assert [e["game"] for e in log if e["event"] == "admit"] == ["g2", "g5", "g3", "g4"]


async def test_role_changes_move_queued_requests_between_the_queues():
    now, wall, games, sched = steady()
    g0, g1, g2 = games[:3]
    g0.latest_prompt_tokens = g1.latest_prompt_tokens = 100
    gate, _ = gate_for(games, 600, now)
    holder = await gate.acquire("g0", 500)
    lane = asyncio.create_task(gate.acquire("g2", 100))  # 500 + 100 + guards 200 > 600
    await asyncio.sleep(0)
    sched.park(g1, "index")
    sched.admit(g2, "index", None)  # promoted: it now waits by the plain focus rule and fits
    gate.refresh()
    admitted = await asyncio.wait_for(lane, 1)
    assert not admitted.lane and gate.lane_inflight == 0 and g1.lane and not g2.lane
    gate.release(admitted)
    gate.release(holder)


async def test_a_focus_game_parked_after_admission_requeues_in_the_lane():
    now, wall, games, sched = steady()
    g0, g1 = games[:2]
    g0.latest_prompt_tokens = g1.latest_prompt_tokens = 100
    gate, _ = gate_for(games, 1000, now)
    holder = await gate.acquire("g0", 900)
    slot = GateSlot(gate, g1, cap=CAP, window=WINDOW)
    slot.reserve(100)
    entering = asyncio.create_task(slot.__aenter__())
    await asyncio.sleep(0)
    sched.park(g1, "index")  # admitted as focus below, but it is a lane game by the time it resumes
    gate.release(holder)
    await asyncio.wait_for(entering, 1)
    assert gate.lane_inflight == 1 and slot._stack[-1].lane and g1.requests[-1]["lane"] is True
    await slot.__aexit__(None, None, None)


async def test_lane_play_is_charged_to_the_lane_game_and_a_lane_clear_wins_a_focus_slot():
    now, wall, games, sched = build("index", count=4, index_settings={}, background_lane={})
    g0, g1, g2, g3 = games
    for g, tokens in ((g0, 4000), (g1, 6000), (g2, 5000), (g3, 5000)):
        charged(now, g, 20, tokens)
    g0.report_level(1, wall.elapsed())
    g1.report_level(1, wall.elapsed())
    sched.tick()
    assert g3.lane and g3.clock.parked
    gate = KVGate(10_000, slots=25, now=now, lane=BackgroundLane(games, window=WINDOW, cap=CAP))
    slot = GateSlot(gate, g3, cap=CAP, window=WINDOW)
    for _ in range(2):
        for g in (g0, g1):
            charged(now, g, 30, 4000)  # the focus games pass their quantum without a clear
        sched.tick()
    assert (g2.state, g3.state) == ("parked", "parked") and sched.index_swaps == 0
    # One background request (1,500 completion tokens) clears g3's first level.
    slot.reserve(100)
    await slot.__aenter__()
    focus_tokens = g0.clock.completion_tokens()
    charged(now, g3, 30, 1500)
    await slot.__aexit__(None, None, None)
    g3.report_level(1, wall.elapsed())
    assert g0.clock.completion_tokens() == focus_tokens and g3.lane_tokens() == 1500
    assert sched.lane_clears[-1]["game"] == "g3" and sched.lane_clears[-1]["level_tokens"] == 6500
    sched.tick()  # the next index decision
    assert g3.state == "active" and not g3.lane and g3.admissions[-1]["reason"] == "index"
    assert g1.state == "parked" and g1.lane
    summary = sched.summary()["background_lane"]
    assert summary["promotions"][-1] == {"game": "g3", "reason": "index", "levels": 1,
                                         "wall_seconds": round(wall.elapsed(), 3), "after_lane_clear": True}
    assert summary["requests"] == 1 and summary["completion_tokens"] == 1500
    assert summary["games"] == {"g3": {"requests": 1, "completion_tokens": 1500, "clears": 1}}
    # Focus requests now: g3's next request is a focus request and settles its lane tokens.
    slot.reserve(100)
    await slot.__aenter__()
    charged(now, g3, 30, 700)
    await slot.__aexit__(None, None, None)
    assert g3.lane_tokens() == 1500 and g3.requests[-1]["lane"] is False
    g3.note_finished("won")
    sched.tick()
    assert g3.state == "finished" and not g3.lane


def test_without_the_lane_nothing_changes():
    now, wall, games, sched = build("index", count=4, index_settings={})
    for g in games:
        charged(now, g, 10, 3000)
    games[0].report_level(1, wall.elapsed())
    games[1].report_level(1, wall.elapsed())
    sched.tick()
    assert not any(g.lane for g in games) and "background_lane" not in sched.summary()
    assert "lane" not in KVGate(1000, slots=2).snapshot()
    with pytest.raises(ValueError, match="index budget policy"):
        Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions", background_lane={})
    for bad in ({"max_inflight": 0}, {"guard_caps": 0.5}, {"slots": 2}):
        with pytest.raises(ValueError):
            lane_settings(bad)


def test_variant_and_bundle(tmp_path):
    study = apply_variant(CONTROL, "fixed-2x-index-lane")
    index = apply_variant(CONTROL, "fixed-2x-index")
    assert study["background_lane"] == {"max_inflight": 2, "guard_caps": 2}
    meta = study.pop("arena_variant")
    index_meta = index.pop("arena_variant")
    assert {k for k in study.keys() | index.keys() if study.get(k) != index.get(k)} == {"background_lane"}
    assert meta["reference_variant"] == "reclaim-fixed-2x" and set(meta["reference_difference"]) == {
        "level_budget_policy", "index_scheduler", "background_lane"}
    assert "background_lane" not in index_meta["changed"]
    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    _, built = builder.build("lane-test", paired_30m=True, output=tmp_path / "lane", arena_variant="fixed-2x-index-lane",
                             opening_burst_requests=3, exclude_games=["ft09", "lp85"], wall_seconds=3600)
    assert built["background_lane"] == {"max_inflight": 2, "guard_caps": 2}
    assert built["level_budget_policy"] == "index" and "background lane" in built["level_budget_rule"]
    assert json.loads((tmp_path / "lane/configs/comparison.json").read_text())["background_lane"]["max_inflight"] == 2


async def test_trial_wires_the_lane_into_the_gate_the_scheduler_and_the_report(tmp_path, monkeypatch):
    from test_arena_trial import STUDY, FakeArcade, OfflineMonitor

    from arena import trial as trial_module
    from arena.trial import Trial

    games = STUDY["games"][:5]
    study = {**STUDY, "games": games, "wall_seconds": 1.5, "level_budget_policy": "index",
             "backlog_policy": "readmissions", "index_scheduler": {}, "background_lane": {"max_inflight": 2}}
    monkeypatch.setattr(trial_module, "make_arcade", lambda mode, games_dir, rec: FakeArcade(games))
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda server_dir: 291_900)
    monkeypatch.setattr(trial_module, "GpuMonitor", OfflineMonitor)

    class RequestingWorker:
        """Sends one small request every 20 ms through its gate slot; clears a level early."""

        def __init__(self, game, gate, window, cap):
            self.game, self.slot = game, GateSlot(gate, game, cap=cap, window=window)

        async def run(self):
            self.game.ready.set()
            while True:
                self.slot.reserve(2000)
                async with self.slot:
                    with self.game.clock.inference():
                        await asyncio.sleep(0.02)
                    self.game.clock.charge(2000, 500)
                if self.game.index in (1, 3) and self.game.levels == 0:
                    self.game.report_level(1, self.game.clock.global_clock.elapsed())

        def result(self):
            return {"game_id": self.game.game_id, "status": "time_limit", "error": None, "levels": self.game.levels,
                    "total_levels": 6, "actions": 3, "model_calls": 2, "prompt_tokens": 100, "completion_tokens": 50}

    async def fake_duck_agents(self, records, gate, window, wall, card, arc):
        assert gate.lane is not None and gate.lane.max_inflight == 2
        workers = [RequestingWorker(r, gate, window, self.cap) for r in records]
        self.config_dict = {"reasoning_effort": "xhigh", "temperature": 0.7}

        async def close():
            return None

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", fake_duck_agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t",
                  server_dir=tmp_path, base_url="http://127.0.0.1:1234/v1", log=lambda *a, **k: None)
    report = await trial.run()
    assert report["status"] == "complete" and report["scheduler_audit"]["valid"], report["scheduler_audit"]
    lane, gate = report["scheduler"]["background_lane"], report["gate"]["lane"]
    # Every completed lane request charged 500 tokens; up to two were cut at the wall.
    assert 500 * (lane["requests"] - 2) <= lane["completion_tokens"] <= 500 * lane["requests"]
    assert gate["lane_admitted"] == lane["requests"] and gate["focus_wait_seconds"] == 0
    assert any(k.endswith("lane=2") for k in gate["inflight_seconds"])
    # The parked games kept playing in the lane with their clocks parked.
    parked = [r for r in report["games"] if r["state"] == "parked"]
    assert parked and all(r["game_id"] in lane["games"] and r["parked_seconds"] > 0 for r in parked)
    with pytest.raises(ValueError, match="plain Duck-transfer"):
        await Trial({**study, "explorer": {"max_level": 0}}, study["trials"][0], source=".", games_dir="games",
                    output=tmp_path / "x", server_dir=tmp_path, base_url="http://127.0.0.1:1234/v1",
                    log=lambda *a, **k: None).run()
