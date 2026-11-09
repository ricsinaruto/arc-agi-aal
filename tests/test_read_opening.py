"""The held-out read's opening: continuation affinity on the read's gate until the 6th distinct try
clears level 1, then FIFO for good; training benches (no study.hread) keep burst 1."""
import asyncio

from arena.gate import KVGate
from arena.level_bench import read_opening

READ = {"hread": {"protocol": "x"}, "opening_burst_requests": 3, "opening_switch_level1_games": 6}


async def contended_round(gate):
    """One request at a time fits. A holds the gate; B queues first, then A's next request: while the opening lasts
    A's continuation passes B (affinity), after it B (the FIFO head) goes first."""
    first = await gate.acquire("A", 100)
    order = []

    async def ask(game):
        r = await gate.acquire(game, 100)
        order.append(game)
        gate.release(r)

    b = asyncio.ensure_future(ask("B"))
    await asyncio.sleep(0)
    a = asyncio.ensure_future(ask("A"))
    await asyncio.sleep(0)
    gate.release(first)
    await asyncio.gather(a, b)
    return order


def test_read_opening_affinity_until_the_sixth_level1_clear():
    rows = []
    gate = KVGate(100, slots=4, log=rows.append)
    opening = read_opening(READ, gate, rows.append)
    assert gate.opening_burst_requests == 3 and gate.opening() and rows[-1]["event"] == "opening_start"
    assert asyncio.run(contended_round(gate)) == ["A", "B"] and gate.affinity_admissions == 1
    for i in range(5):
        opening.reached(f"t{i}", 1)
    opening.reached("t0", 2)  # a deeper level of a try already counted is no new game
    opening.reached("t9", 0)  # no level cleared yet
    assert gate.opening() and not any(r["event"] == "opening_end" for r in rows)
    opening.reached("t5", 1)  # the 6th distinct try
    ends = [r for r in rows if r["event"] == "opening_end"]
    assert not gate.opening() and ends == [{"event": "opening_end", "reached": 6, "switch": 6, "last": "t5"}]
    before = gate.affinity_admissions
    selections = sum(r["event"] == "opening_selection" for r in rows)
    assert asyncio.run(contended_round(gate)) == ["B", "A"] and gate.affinity_admissions == before
    assert sum(r["event"] == "opening_selection" for r in rows) == selections  # no opening rows after the end
    opening.reached("t7", 3)
    assert not gate.opening()  # no reopen


def test_training_benches_and_burst_one_keep_the_fifo_gate():
    for study in ({**READ, "hread": None}, {**READ, "opening_burst_requests": 1}, {"opening_burst_requests": 3}):
        gate = KVGate(100, slots=4)
        assert read_opening(study, gate, lambda row: None) is None
        assert gate.opening_burst_requests == 1 and not gate.opening()
        assert asyncio.run(contended_round(gate)) == ["B", "A"]
