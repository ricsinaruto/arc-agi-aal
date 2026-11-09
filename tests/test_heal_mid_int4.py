"""HEAL_MID_INT4 (audit 24 Oct): Int4Rows puts the served int4 values (heal_delta's export rule) into the trained rows
inside the block and restores the originals bit for bit after; frozen rows never change."""
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import heal_delta  # noqa: E402
import heal_kd  # noqa: E402


def _deq_one(w, group=128):
    qw, _, sc = heal_delta.int4_rtn(w, group=min(group, w.shape[1]))
    inp8, out = qw.shape
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    codes = ((qw.unsqueeze(1) >> shifts.view(1, -1, 1)) & 15).reshape(inp8 * 8, out).float() - 8
    return (codes * sc.float().repeat_interleave(w.shape[1] // sc.shape[0], 0)).t()


class _Rows:
    def __init__(self, items):
        self.items = items


def test_int4_rows_quantizes_trained_rows_and_restores_exactly():
    os.environ.pop("HEAL_DELTA_FAST", None)
    g = torch.Generator().manual_seed(0)
    gate_up = torch.nn.Parameter(torch.randn(6, 2 * 64, 256, generator=g) * 0.02)  # [E, 2I, H]
    down = torch.nn.Parameter(torch.randn(6, 256, 64, generator=g) * 0.02)  # [E, H, I]
    before = (gate_up.detach().clone(), down.detach().clone())
    sel = torch.tensor([1, 4])
    rows = _Rows([{"p": gate_up, "rows": sel}, {"p": down, "rows": sel}])
    names = {id(gate_up): "model.layers.0.mlp.experts.gate_up_proj", id(down): "model.layers.0.mlp.experts.down_proj"}
    with heal_kd.Int4Rows(rows, names):
        for e in (1, 4):
            w = before[0][e]
            want = torch.cat([_deq_one(w[:64]), _deq_one(w[64:])], 0)
            assert torch.allclose(gate_up.data[e], want, atol=1e-6)
            assert torch.allclose(down.data[e], _deq_one(before[1][e]), atol=1e-6)
            assert not torch.equal(gate_up.data[e], w)
        for e in (0, 2, 3, 5):  # frozen rows untouched
            assert torch.equal(gate_up.data[e], before[0][e]) and torch.equal(down.data[e], before[1][e])
    assert torch.equal(gate_up.data, before[0]) and torch.equal(down.data, before[1])


def test_int4_rows_restores_after_an_error():
    p = torch.nn.Parameter(torch.randn(3, 128, 128) * 0.02)
    before = p.detach().clone()
    rows = _Rows([{"p": p, "rows": torch.tensor([0, 2])}, {"p": p, "rows": torch.tensor([], dtype=torch.long)}])
    with pytest.raises(RuntimeError):
        with heal_kd.Int4Rows(rows, {id(p): "x.down_proj"}):
            raise RuntimeError("read failed")
    assert torch.equal(p.data, before)


def test_prune_checkpoints_keeps_a_rollback_point(tmp_path, monkeypatch):
    for k, complete in ((3, True), (6, True), (9, False), (12, True), (15, True)):
        d = tmp_path / f"ckpt-{k}"
        d.mkdir()
        if complete:
            (d / "progress.json").write_text("{}")
    monkeypatch.setenv("HEAL_KEEP_CKPT", "2")
    heal_kd.prune_checkpoints(tmp_path, tmp_path / "ckpt-12")
    assert sorted(p.name for p in tmp_path.glob("ckpt-*")) == ["ckpt-12", "ckpt-15", "ckpt-6"]  # newer 15 untouched
    monkeypatch.delenv("HEAL_KEEP_CKPT")
    heal_kd.prune_checkpoints(tmp_path, tmp_path / "ckpt-15")
    assert sorted(p.name for p in tmp_path.glob("ckpt-*")) == ["ckpt-15"]  # default: newest only
