"""MFU-1 HEAL_STREAM_ALIGN: the slot picker takes the pool members whose call structure matches the lead's."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
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
