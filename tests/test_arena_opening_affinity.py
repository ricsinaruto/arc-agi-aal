"""Opening affinity preserves capacity, bounded FIFO aging and phase-two FIFO."""
import asyncio

from arena.gate import KVGate


async def queued(gate, name, cost=10):
    task = asyncio.create_task(gate.acquire(name, cost))
    await asyncio.sleep(0)
    return task


async def test_warm_continuations_are_bounded_and_cold_head_eventually_runs():
    events = []
    gate = KVGate(100, 1, opening_burst_requests=3, opening=lambda: True, log=events.append)
    warm = await gate.acquire('warm', 10)
    cold = await queued(gate, 'cold')
    # Two continuation requests bypass the cold head. Third returns to FIFO.
    for i in range(3):
        continuation = await queued(gate, 'warm')
        gate.release(warm)
        await asyncio.sleep(0)
        if i < 2:
            warm = await continuation
            assert not cold.done()
        else:
            first_cold = await cold
            assert not continuation.done()
            gate.release(first_cold)
            gate.release(await continuation)
    assert gate.affinity_admissions == 2
    assert gate.inflight == gate.reserved == 0
    assert [e['policy'] for e in events if e['event'] == 'opening_selection'][:4] == [
        'fifo', 'continuation', 'continuation', 'fifo']


async def test_phase_switch_restores_fifo_and_cancelled_waiters_release_cleanly():
    opening = True
    gate = KVGate(100, 1, opening_burst_requests=3, opening=lambda: opening)
    a = await gate.acquire('a', 80)
    cold = await queued(gate, 'cold', 80)
    warm = await queued(gate, 'a', 80)
    opening = False
    gate.release(a)
    b = await cold
    assert not warm.done()
    warm.cancel()
    try:
        await warm
    except asyncio.CancelledError:
        pass
    gate.release(b)
    assert gate.inflight == gate.reserved == len(gate.queue) == 0
    assert gate.affinity_admissions == 0


async def test_affinity_never_overbooks_cache_and_does_not_hold_idle_tool_slots():
    gate = KVGate(100, 3, opening_burst_requests=3, opening=lambda: True)
    a = await gate.acquire('a', 60)
    b = await gate.acquire('b', 40)
    cold = await queued(gate, 'cold', 100)
    warm = await queued(gate, 'a', 70)
    gate.release(a)
    assert not cold.done() and not warm.done()
    gate.release(b)
    c = await warm
    assert gate.reserved == 70 and not cold.done()
    gate.release(c)
    gate.release(await cold)
    assert gate.peak_reserved <= 100 and gate.peak_inflight <= 3
    assert gate.reserved == 0  # no reservation held for a hypothetical next request


def test_matched_bundles_only_differ_in_opening_policy(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('builder', root / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    _, control = builder.build('clock-fifo', paired_30m=True, arena_variant='cache-reclaim',
                               output=tmp_path / 'fifo')
    _, candidate = builder.build('clock-affinity', paired_30m=True, arena_variant='cache-reclaim',
                                 opening_burst_requests=3, output=tmp_path / 'affinity')
    assert control['budget_clock'] == candidate['budget_clock'] == 'tokens'  # new builds (29 September)
    assert control['opening_burst_requests'] == 1 and candidate['opening_burst_requests'] == 3
    assert control['active_limit'] == candidate['active_limit'] == 2
    assert {k for k in control if control[k] != candidate[k]} == {'name', 'opening_burst_requests'}
