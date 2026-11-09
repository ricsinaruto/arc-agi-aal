"""MFU-1 HEAL_STREAM_ALIGN: the slot picker takes the pool members whose call structure matches the lead's."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)
import heal_kd  # noqa: E402


def _queue(tmp_path, shapes):
    q = heal_kd.StreamQueue(tmp_path, 0, coalesce=True)
    q.window, q.chunk = 45056, 32768
    fresh = []
    for i, (tokens, row0) in enumerate(shapes):
        path = str(tmp_path / f"s-c{i:03d}-p0000.pt")
        q.meta[path] = {"tokens": tokens, "rows": tokens - row0, "reply_rows": 100}
        fresh.append((5, 1000.0 - i, path))  # piece 0 is the freshest: it leads
    return q, fresh


def test_lockstep_idle(tmp_path, monkeypatch):
    monkeypatch.setenv("HEAL_INERT_PREFIX", "1")
    monkeypatch.setenv("HEAL_SKIP_INERT_LOSS", "1")
    q, fresh = _queue(tmp_path, [(40000, 8000), (40000, 8000), (40000, 30000)])
    same = q.lockstep_idle([[fresh[0][2]], [fresh[1][2]]])
    diff = q.lockstep_idle([[fresh[0][2]], [fresh[2][2]]])
    assert same < 0.01 < diff


def test_align_picks_matching_shapes(tmp_path, monkeypatch):
    monkeypatch.setenv("HEAL_INERT_PREFIX", "1")
    monkeypatch.setenv("HEAL_SKIP_INERT_LOSS", "1")
    monkeypatch.setenv("HEAL_STREAM_BALANCE", "1")
    # lead: 4K prefix + 28K loss. fe-equal decoys (prefix-heavy) vs two structural matches further down the pool
    shapes = [(32768, 4096), (100000, 20000), (100000, 20000), (33000, 4300), (32500, 4000), (60000, 50000)]
    q, fresh = _queue(tmp_path, shapes)
    monkeypatch.delenv("HEAL_STREAM_ALIGN", raising=False)
    plain = q.units(fresh, 3)
    monkeypatch.setenv("HEAL_STREAM_ALIGN", "1")
    aligned = q.units(fresh, 3)
    idle = lambda units: q.lockstep_idle(units)  # noqa: E731
    assert aligned[0] == plain[0] == [fresh[0][2]]
    assert sorted(u[0] for u in aligned[1:]) == sorted([fresh[3][2], fresh[4][2]])
    assert idle(aligned) <= idle(plain)


def test_step_plan_groups_the_steps_units_by_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("HEAL_INERT_PREFIX", "1")
    monkeypatch.setenv("HEAL_SKIP_INERT_LOSS", "1")
    shapes = [(30000, 20000), (80000, 64000), (20000, 4000), (70000, 60000), (34000, 18000), (90000, 70000),
              (25000, 5000), (40000, 16000), (60000, 48000), (50000, 30000)]  # 10 fresh parts, freshest first
    q, fresh = _queue(tmp_path, shapes)

    def plain_slots():  # today's per-slot picks over the same parts, removing each slot's units
        left, out = list(fresh), []
        while len(left) >= 3:
            got = q.units(left, 3)
            out.append(got)
            used = {p for u in got for p in u}
            left = [f for f in left if f[2] not in used]
        return out

    monkeypatch.delenv("HEAL_STREAM_STEPPLAN", raising=False)
    plain = plain_slots()[:3]
    monkeypatch.setenv("HEAL_STREAM_STEPPLAN", "1")
    first = q.units(fresh, 3, max_rows=300)  # room: 3 ranks x 300 rows = 9 parts of 100 rows
    slots = [first] + [q.planned(fresh) for _ in range(len(q.plan))]
    assert q.plan == [] and len(slots) == 3
    got = sorted(p for s in slots for u in s for p in u)
    assert got == sorted(f[2] for f in fresh[:9])  # the freshest 9 parts: the same step, regrouped
    prefixes = [[q.calls([u[0]])[0][1] and 1 for u in s] for s in slots]
    assert prefixes  # every slot has calls
    span = lambda ss: sum(q.lockstep_span([q.calls(u) for u in s])[0] for s in ss)  # noqa: E731
    assert span(slots) < 0.95 * span(plain)  # 396K -> 371K fe here; c7h's mix gains more


def test_planned_drops_a_plan_with_a_vanished_piece(tmp_path, monkeypatch):
    monkeypatch.setenv("HEAL_INERT_PREFIX", "1")
    monkeypatch.setenv("HEAL_STREAM_STEPPLAN", "1")
    q, fresh = _queue(tmp_path, [(30000, 20000 - 1000 * i) for i in range(6)])
    q.units(fresh, 3, max_rows=200)
    assert len(q.plan) == 1
    gone = q.plan[0][0][0]
    assert q.planned([f for f in fresh if f[2] != gone]) is None and q.plan == []
