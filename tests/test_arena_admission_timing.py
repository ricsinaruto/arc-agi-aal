"""Separate scheduler parking from eligible waits, including cancelled work."""

import asyncio

import pytest

from arena.clock import ActiveClock, GlobalClock
from arena.gate import GateSlot, KVGate
from arena.report import game_row
from arena.scheduler import GameRecord


def setup():
    now = [0.0]
    clock = lambda: now[0]
    wall = GlobalClock(1800, now=clock)
    game = GameRecord("game", 0, ActiveClock(wall, now=clock))
    wall.release()
    game.clock.release()
    events = []
    gate = KVGate(1000, slots=1, now=clock, log=events.append)
    return now, game, gate, GateSlot(gate, game, cap=100, window=500), events


async def test_waiting_clock_split_includes_parking_while_capacity_queued():
    now, game, gate, slot, events = setup()
    holder = await gate.acquire("holder", 500)
    entered = asyncio.create_task(slot.__aenter__())
    await asyncio.sleep(0)
    now[0] = 3
    game.clock.pause()
    now[0] = 8
    gate.release(holder)
    await asyncio.sleep(0)  # Acquires, notices parking, releases, then waits again.
    assert not entered.done() and gate.inflight == 0
    now[0] = 13
    game.clock.resume()
    await entered
    now[0] = 17
    game.clock.pause()  # Requests already issued finish even when a game is parked.
    now[0] = 19
    await slot.__aexit__(None, None, None)
    record = game.admission_attempts[0]
    assert record["status"] == "closed"
    assert record["wait_seconds"] == 13
    assert record["eligible_wait_seconds"] == 3
    assert record["parked_wait_seconds"] == 10
    assert record["context_seconds"] == 6
    assert record["context_active_seconds"] == 4
    assert record["context_parked_seconds"] == 2
    assert game.requests == [{"tokens": 100, "gate_wait_seconds": 13, "wall_seconds": 13}]
    assert gate.reserved == gate.inflight == 0
    # Event snapshots must not mutate when the attempt reaches a later state.
    starts = [e for e in events if e["event"] == "request_wait_started"]
    assert starts[0]["status"] == "waiting" and "context_seconds" not in starts[0]
    row = game_row(game, {})
    assert row["admission_timing"]["eligible_wait_seconds"] == 3
    assert row["admission_timing"]["parked_wait_seconds"] == 10


@pytest.mark.parametrize("parked", [False, True])
async def test_cancelled_wait_is_recorded_without_counting_it_as_a_request(parked):
    now, game, gate, slot, events = setup()
    holder = await gate.acquire("holder", 500)
    if parked:
        game.clock.pause()
    entered = asyncio.create_task(slot.__aenter__())
    await asyncio.sleep(0)
    now[0] = 7
    entered.cancel()
    with pytest.raises(asyncio.CancelledError):
        await entered
    record = game.admission_attempts[0]
    assert record["status"] == "cancelled_wait"
    assert record["wait_seconds"] == 7
    assert record["parked_wait_seconds"] == (7 if parked else 0)
    assert record["eligible_wait_seconds"] == (0 if parked else 7)
    assert not game.requests and "context_seconds" not in record
    assert events[-1]["event"] == "request_wait_closed"
    assert gate.inflight == 1 and not gate.queue
    gate.release(holder)


async def test_context_cancellation_records_incomplete_work_and_releases_slot():
    now, game, gate, slot, _ = setup()
    running = asyncio.Event()

    async def request():
        async with slot:
            running.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(request())
    await running.wait()
    now[0] = 9
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert game.admission_attempts[0]["status"] == "cancelled_context"
    assert game.admission_attempts[0]["context_seconds"] == 9
    assert gate.inflight == gate.reserved == 0


async def test_wait_and_context_failures_remain_failures_with_evidence():
    now, game, gate, slot, _ = setup()
    with pytest.raises(ValueError, match="endpoint"):
        async with slot:
            now[0] = 4
            raise ValueError("endpoint")
    assert game.admission_attempts[0]["status"] == "failed_context"
    assert game.admission_attempts[0]["context_seconds"] == 4
    assert gate.inflight == gate.reserved == 0

    class TooLarge:
        def cost(self, tokens):
            return 2000

    gate.budget = TooLarge()
    with pytest.raises(ValueError, match="Single request"):
        await slot.__aenter__()
    assert game.admission_attempts[-1]["status"] == "failed_wait"
    assert len(game.requests) == 1


async def test_nested_summary_and_gameplay_keep_separate_records():
    now, game, gate, slot, _ = setup()
    gate.slots = 2
    async with slot:
        now[0] = 2
        async with slot:
            now[0] = 3
        now[0] = 5
    assert [r["context_seconds"] for r in game.admission_attempts] == [5, 1]
    assert [r["attempt"] for r in game.admission_attempts] == [1, 2]
    assert gate.inflight == gate.reserved == 0


async def test_pre_barrier_wait_is_not_mislabeled_gameplay_queue_time():
    now = [0.0]
    clock = lambda: now[0]
    wall = GlobalClock(1800, now=clock)
    game = GameRecord("game", 0, ActiveClock(wall, now=clock))
    game.clock.pause()
    gate = KVGate(1000, slots=1, now=clock)
    slot = GateSlot(gate, game, cap=100, window=500)
    entering = asyncio.create_task(slot.__aenter__())
    await asyncio.sleep(0)
    now[0] = 12
    wall.release()
    game.clock.release()
    game.clock.resume()
    await entering
    await slot.__aexit__(None, None, None)
    record = game.admission_attempts[0]
    assert record["wait_seconds"] == record["unclassified_wait_seconds"] == 12
    assert record["eligible_wait_seconds"] == record["parked_wait_seconds"] == 0
