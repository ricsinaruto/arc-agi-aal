"""heal_kd's resume schedule check under HEAL_REWARM: the checkpoint's last lr is compared with the writer's schedule
(its own --step0 ramp), so an exact resume of a re-warmed WSD run passes. Needs torch (Modal CPU)."""
import argparse
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import heal_kd  # noqa: E402


def test_rewarm_resume_matches_writer(monkeypatch):
    monkeypatch.setenv("HEAL_SCHEDULE_STEPS", "125")
    monkeypatch.setenv("HEAL_WSD_DECAY", "0")
    monkeypatch.setenv("HEAL_REWARM", "2")
    a = argparse.Namespace(lr=1e-5, total_steps=0, warmup=0.0, schedule="wsd", step0=0)
    written = heal_kd.lr_at(1, a)  # round A's lr at step 1 (ramped: step0 0, rewarm 2)
    b = argparse.Namespace(**{**vars(a), "step0": 2})
    assert heal_kd.lr_at(1, b) != written  # the old check compared these and refused the resume
    assert heal_kd.lr_at(1, argparse.Namespace(**{**vars(b), "step0": 0})) == written
    assert heal_kd.lr_at(2, b) < heal_kd.lr_at(4, b)  # the resumed run re-warms from its own step0


def test_heal_threads_sets_torch_threads():
    import os
    import subprocess
    src = Path(heal_kd.__file__)
    code = ("import sys, torch; sys.argv=['heal_kd','tiny','--help']; sys.path.insert(0, %r); import heal_kd\n"
            "try:\n    heal_kd.main()\nexcept SystemExit:\n    pass\nprint(torch.get_num_threads())") % str(src.parent)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "OMP_NUM_THREADS": "1", "HEAL_THREADS": "3"}).stdout.split()
    assert out and out[-1] == "3", out
