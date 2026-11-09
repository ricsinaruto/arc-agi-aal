"""The exact-gate replay (``scripts/arena_analysis/gate_replay.py``): FIFO head-of-line blocking, the slot limit, a
stream that moves on while its request is in flight, the open-loop validation of a recorded run against
``arena.cache.BlockBudget`` and ``arena.gate.KVGate``, and the phase-2 tenure extraction."""

import sys
from pathlib import Path

import pytest

from arena.cache import BlockBudget

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/arena_analysis"))
import gate_replay as gr  # noqa: E402


def test_fifo_head_of_line_blocks_a_smaller_request_behind_it():
    # capacity 10: A (6) runs; B (6) does not fit; C (2) would fit but waits behind B (FIFO, as KVGate).
    a, b, c = gr.Request(6, 10.0, at=0.0), gr.Request(6, 5.0, at=1.0), gr.Request(2, 5.0, at=2.0)
    done, series = gr.GateSim(10, 8).run([[a], [b], [c]])
    assert (a.admitted, b.admitted, c.admitted) == (0.0, 10.0, 10.0)
    assert b.wait == 9.0 and c.wait == 8.0
    assert max(s[1] for s in series) == 8  # B + C after A's release
    assert gr.time_share(series, 0.0, 15.0, lambda r, i, w: w > 0) == pytest.approx(9.0 / 15.0)


def test_slot_limit_and_closed_loop_gaps():
    # one slot: the second stream's request waits for the first's release; a stream's next request follows its
    # previous release by its gap.
    s1 = [gr.Request(1, 4.0, gap=0.0), gr.Request(1, 1.0, gap=2.0)]
    s2 = [gr.Request(1, 3.0, gap=1.0)]
    done, _ = gr.GateSim(100, 1).run([s1, s2])
    first, second = s1
    assert first.admitted == 0.0 and s2[0].admitted == 4.0  # arrived at 1, waited for the slot
    assert second.arrival == 6.0 and second.admitted == 7.0  # arrived 2 s after its release, waited for s2


def test_moves_on_overlaps_the_next_request_with_the_one_in_flight():
    # a game parked mid-request: the next game on the stream starts 1 s after the admission, beside it
    s = [gr.Request(5, 10.0, moves_on=1.0), gr.Request(5, 2.0, gap=0.0)]
    done, series = gr.GateSim(100, 4).run([s])
    assert s[1].admitted == 1.0
    assert max(x[2] for x in series) == 2  # two in flight
    assert gr.Request(5, 1.0, moves_on=3.0).moves_on is None  # a move after the release is a plain release


def layout():
    rows = [{"tokens": t, "blocks": 2 + t // 1000, "groups": [t // 1000, 2]} for t in range(1000, 65001, 1000)]
    return {"schema": 1, "num_blocks": 100, "max_model_len": 65000, "lookahead": 0, "costs": rows}


def report(budget):
    """Two games; the switch at 100 s; g1 parked at the switch and re-admitted at 150 s."""
    def attempt(n, requested, admitted, ended, waited=0.0):
        return {"attempt": n, "status": "closed", "requested_tokens": requested, "started_wall_seconds": admitted - waited,
                "eligible_wait_seconds": waited, "admitted_wall_seconds": admitted, "ended_wall_seconds": ended,
                "tokens": budget.cost(requested), "admitted_phase": "steady" if admitted > 100 else "opening"}

    def charge(end, prompt=1000, completion=500):
        return {"wall_seconds": end, "purpose": "gameplay", "prompt_tokens": prompt, "completion_tokens": completion}

    g0 = {"game_id": "g0", "index": 0, "time_to_level": [{"level": 1, "wall_seconds": 90.0, "budget_seconds": 1000}],
          "admissions": [{"event": "admit", "wall_seconds": 0.0, "levels": 0},
                         {"event": "finish", "wall_seconds": 7200.0, "levels": 1}],
          "admission_timing": {"attempts": [attempt(1, 20000, 10.0, 95.0), attempt(2, 30000, 110.0, 130.0),
                                            attempt(3, 30000, 140.0, 170.0)]},
          "charges": [charge(95.0), charge(130.0), charge(170.0)]}
    g1 = {"game_id": "g1", "index": 1, "time_to_level": [],
          "admissions": [{"event": "admit", "wall_seconds": 0.0, "levels": 0},
                         {"event": "park", "wall_seconds": 100.0, "levels": 0},
                         {"event": "admit", "wall_seconds": 150.0, "levels": 0},
                         {"event": "finish", "wall_seconds": 7200.0, "levels": 0}],
          # its second request (62 blocks) queues 5 s until g0's third (32) is released: 62 + 32 > 90
          "admission_timing": {"attempts": [attempt(1, 20000, 20.0, 60.0), attempt(2, 60000, 170.0, 200.0, waited=5.0)]},
          "charges": [charge(60.0), charge(200.0)]}
    return {"games": [g0, g1], "gate": {"slots": 4, "peak_reserved": 0},
            "scheduler": {"switch_wall_seconds": 100.0, "events": [{"wall_seconds": 7200.0}]},
            "_gate_admits": {"g1": [(170.0, 5.0)]}}


def test_validate_reproduces_the_recorded_admissions_and_extracts_tenures():
    budget = BlockBudget(layout(), margin=0.9)
    assert budget.capacity == 90 and budget.cost(20000) == 22 and budget.cost(60001) == 63
    rep = report(budget)
    out = gr.validate(rep, budget)
    assert out["cost_mismatches"] == 0 and out["phase2_requests"] == 3
    assert out["admission_time_error_seconds"]["max"] == 0.0
    tenures = gr.phase2_tenures(rep, budget)
    by_game = {t["game"]: t for t in tenures}
    assert by_game["g0"]["levels"] == 1 and by_game["g1"]["levels"] == 0
    assert [round(x[0], 3) for x in by_game["g0"]["requests"]] == [10.0, 10.0]  # gaps after the switch, release
    assert by_game["g1"]["requests"][0][:3] == (15.0, budget.cost(60000), 30.0)  # arrival 165 after the 150 admit


def test_stream_replay_runs_and_scales_holds():
    budget = BlockBudget(layout(), margin=0.9)
    tenures = gr.phase2_tenures(report(budget), budget)
    out = gr.replay_streams(tenures, budget.capacity, 3, seeds=2, start=100.0, duration=2000.0, warmup=100.0)
    assert out["max_num_seqs"] == 5 and out["hold_factor"] == pytest.approx(gr.slowdown(3), abs=1e-4)
    assert 0.0 <= out["time_share_waiting"] <= 1.0 and out["reserved_blocks"]["max"] <= budget.capacity
    assert gr.slowdown(23) == 1.0 and gr.slowdown(30) > 1.0
