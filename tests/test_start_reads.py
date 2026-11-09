"""heal_kd.start_reads_cache: a fresh start's reads are reused only for the same
student folder and read settings, and with the held-out sampled metric only when b is frozen in HEAL_HELDOUT2_BEH."""
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import heal_kd  # noqa: E402


def args(tmp, student="s"):
    (tmp / "held").mkdir(exist_ok=True)
    for i in range(3):
        (tmp / "held" / f"f{i}.pt").write_bytes(b"x")
    return types.SimpleNamespace(student=str(tmp / student), heldout=str(tmp / "held"), heldout_files=20,
                                 general="g", window=32768, chunk=8192)


def save(path, key, reads):
    path.write_text(json.dumps({"key": key, "reads": reads}))


def test_reuse_only_on_the_same_key(tmp_path, monkeypatch):
    monkeypatch.delenv("HEAL_SAMPLED_METRIC", raising=False)
    monkeypatch.setenv("HEAL_CLASS_SPLIT", "1")
    p = tmp_path / "sr.json"
    got, key = heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 20, 5)
    assert got is None and key["heldout_files"] == len(heal_kd.read_files(str(tmp_path / "held"), 20))
    save(p, key, {"heldout": {"kd": 1.0}, "mid_read0": {"step": 0, "d_fwd": 0.18}})
    got, _ = heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 20, 5)
    assert got == {"heldout": {"kd": 1.0}, "mid_read0": {"step": 0, "d_fwd": 0.18}}
    assert heal_kd.start_reads_cache(str(p), args(tmp_path, student="other"), "h2", 20, 5)[0] is None
    assert heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 10, 5)[0] is None
    other = args(tmp_path)
    other.resume = "/runs/other/ckpt"  # same student folder, other resumed weights
    assert heal_kd.start_reads_cache(str(p), other, "h2", 20, 5)[0] is None
    monkeypatch.setenv("HEAL_MID_GENERAL", "1")  # a cached mid read 0 without the general probe is not reused
    assert heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 20, 5)[0] is None
    assert heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 20, 4)[0] is None
    assert heal_kd.start_reads_cache(None, args(tmp_path), "h2", 20, 5)[0] is None


def test_sampled_metric_needs_the_frozen_b(tmp_path, monkeypatch):
    monkeypatch.setenv("HEAL_SAMPLED_METRIC", "1")
    p = tmp_path / "sr.json"
    _, key = heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 20, 5)
    save(p, key, {"heldout2": {"kd": 1.0}})
    monkeypatch.delenv("HEAL_HELDOUT2_BEH", raising=False)
    assert heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 20, 5)[0] is None
    beh = tmp_path / "beh.pt"
    monkeypatch.setenv("HEAL_HELDOUT2_BEH", str(beh))
    assert heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 20, 5)[0] is None  # no file yet
    beh.write_bytes(b"x")
    assert heal_kd.start_reads_cache(str(p), args(tmp_path), "h2", 20, 5)[0] == {"heldout2": {"kd": 1.0}}


def test_the_trainer_wires_every_start_read():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "heal_kd.py").read_text()
    for k in ('sr_cached["general_probe"]', 'sr_cached["heldout"]', 'sr_cached["heldout2"]', 'sr_cached["mid_read0"]',
              'sr_new["general_probe"]', 'sr_new["heldout"]', 'sr_new["heldout2"]', 'sr_new["mid_read0"]'):
        assert k in src, k
