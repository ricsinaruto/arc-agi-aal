"""c7m (24 Oct): rl1_trainer.plain_causal summed the whole [queries, keys] MTP mask at once (an int64 [55020, 120556]
temporary = 49.4 GiB: CUDA OOM at step 4). Checked in row chunks: same answers, temporaries rows x keys."""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))


def _causal(q, k, dtype=torch.bool):
    m = torch.arange(k)[None, :] < (k - q + 1 + torch.arange(q))[:, None]
    return (m if dtype == torch.bool else torch.where(m, 0.0, float("-inf")).to(dtype))[None, None]


def _naive(mask):
    visible = mask if mask.dtype == torch.bool else mask == 0
    counts = visible[0, 0].sum(-1)
    pos = torch.arange(visible.shape[-1])
    return counts if torch.equal(visible[0, 0], pos[None, :] < counts[:, None]) else None


@pytest.mark.parametrize("q,k,rows", [(7, 7, 3), (50, 130, 8), (64, 64, 2048), (33, 90, 1)])
def test_chunked_plain_causal_matches_the_whole_mask_check(q, k, rows):
    import rl1_trainer

    for dtype in (torch.bool, torch.float32):
        m = _causal(q, k, dtype)
        assert torch.equal(rl1_trainer.plain_causal(m, rows=rows), _naive(m))
    bad = _causal(q, k).clone()
    bad[0, 0, q // 2, 0] = ~bad[0, 0, q // 2, 0]  # not plain causal: None either way
    assert rl1_trainer.plain_causal(bad, rows=rows) is None and _naive(bad) is None
    assert rl1_trainer.plain_causal(torch.ones(2, 1, q, k, dtype=torch.bool), rows=rows) is None


def test_chunked_plain_causal_memory_stays_rows_by_keys(tmp_path):
    """A 4000 x 60000 mask (240 MB bool): the check's peak RSS rise stays near rows x keys x 9 bytes (256 rows: 0.14
    GB), not the whole mask x 8 (1.9 GB) a single int64 sum can take."""
    code = textwrap.dedent(f"""
        import resource, sys
        sys.path.insert(0, {str(ROOT / 'scripts' / 'posttrain')!r})
        import torch, rl1_trainer
        q, k = 4000, 60000
        m = (torch.arange(k)[None, :] < (k - q + 1 + torch.arange(q))[:, None])[None, None]
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        c = rl1_trainer.plain_causal(m, rows=256)
        rise = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base
        print(int(c is not None), rise * (1 if sys.platform == "darwin" else 1024))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    ok, rise = out.stdout.split()
    assert ok == "1" and int(rise) < 0.6e9, (out.stdout, out.stderr[-500:])


def test_mask_host_check_is_chunked_and_matches(monkeypatch):
    """HEAL_MASK_HOST's one-read check (heal_kd.plain_causal_host) goes through the same row chunks."""
    import heal_kd

    monkeypatch.setenv("HEAL_MASK_HOST", "1")
    monkeypatch.setattr(heal_kd.HOST, "wait", lambda name: None, raising=False)
    check = heal_kd.plain_causal_host(lambda m: None)
    m = _causal(50, 130)
    got = check(m)
    assert torch.equal(got, _naive(m)) and got._heal_max == 130
    bad = m.clone()
    bad[0, 0, 3, 0] = ~bad[0, 0, 3, 0]
    assert check(bad) is None


def test_stream_queue_drops_pieces_over_the_unit_bound(tmp_path, monkeypatch):
    """HEAL_MAX_UNIT: a piece whose context exceeds the bound is never picked (the MTP mask and memory are sized at
    the bound in prelaunch_check)."""
    import json
    import heal_kd

    monkeypatch.setenv("HEAL_MAX_UNIT", "130816")
    for name, tokens in (("s-a-p0000", 120556), ("s-b-p0000", 140000), ("b-c-p0000", 200000)):
        (tmp_path / f"{name}.ready").write_text(json.dumps({"version": 1, "tokens": tokens}))
    q = heal_kd.StreamQueue(tmp_path, 0)
    assert [r[2].rsplit("/", 1)[-1] for r in q.ready()] == ["s-a-p0000.pt"] and q.stats["too_long"] == 2
    assert len(q.ready()) == 1 and q.stats["too_long"] == 2  # dropped once, never re-counted


def test_live_ckpt_fires_once_per_new_tag(tmp_path):
    """HEAL_LIVE_KNOBS "ckpt" (c7n): the tag at the first read only arms; each later change saves once."""
    import json
    import heal_kd

    f = tmp_path / "knobs.json"
    live = heal_kd.LiveCkpt(str(f))
    assert live.due(0) is False  # no file
    f.write_text(json.dumps({"lr": 5e-6, "ckpt": "a"}))
    assert live.due(0) is True  # first tag after an empty first read
    assert live.due(0) is False
    f.write_text(json.dumps({"ckpt": "b"}))
    assert [live.due(0), live.due(0)] == [True, False]
    armed = heal_kd.LiveCkpt(str(f))
    assert armed.due(0) is False  # present at the first read: armed only


def test_host_timers_book_peak_raises(monkeypatch):
    """HostTimers (c7n): each phase / leaf books how far it raised the device's peak allocation."""
    import heal_kd

    h = heal_kd.HostTimers()
    h.on = True
    peak = {"v": 100 * 2 ** 30}
    monkeypatch.setattr(h, "_peak", lambda: peak["v"])
    h.reset()
    with h.phase("fwd"):
        peak["v"] += 30 * 2 ** 30
        with h.leaf("mask_check"):
            peak["v"] += 2 * 2 ** 30
    with h.leaf("delta"):
        pass
    r = h.report()
    assert r["peak_raise_gib"] == {"fwd": 32.0, "mask_check": 2.0}
