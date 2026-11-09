"""HEAL_LOAD_PREFETCH (MFU-2, 23 Oct): the student's safetensors split over the ranks (symlinks resolved, each file
read once, the same split on every rank). Needs torch (Modal CPU)."""
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
sys.path.insert(0, "/opt/heal")
import heal_kd  # noqa: E402


def test_split_covers_every_file_once(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    sizes = [5000, 300, 7000, 1200, 50, 9000]
    for i, n in enumerate(sizes):
        (base / f"model-{i:05d}.safetensors").write_bytes(b"x" * n)
    student = tmp_path / "student"
    student.mkdir()
    for i in range(4):  # the healed folder links the cut's shards (as c3's student-p75d4 links)
        os.symlink(base / f"model-{i:05d}.safetensors", student / f"model-{i:05d}.safetensors")
    for i in (4, 5):
        (student / f"trained-{i:05d}.safetensors").write_bytes((base / f"model-{i:05d}.safetensors").read_bytes())
    (student / "config.json").write_text("{}")
    for world in (1, 2, 3, 4):
        got = [heal_kd.prefetch_files(student, r, world, threads=3, chunk=256) for r in range(world)]
        assert {g[0] for g in got} == {6}
        assert sum(g[1] for g in got) == sum(sizes)  # each file once, links followed, config not read
    assert heal_kd.prefetch_files(tmp_path / "empty", 0, 2) == (0, 0)
