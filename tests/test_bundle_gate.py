"""Every launch/rl and launch/pth bundle builder runs the CI gate (ci.py gate) before it writes a bundle."""
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = re.compile(r"\b(ci_gate|bundle_gate)\.require\(")
WRITES = re.compile(r"\b(copytree|rmtree)\(")


def builders():
    return sorted(p for d in ("launch/rl", "launch/pth") for p in (ROOT / d).rglob("*.py") if WRITES.search(p.read_text()))


def test_builders_are_found():
    names = {p.name for p in builders()}
    assert {"build.py", "build_heal.py", "build_play.py", "build_train.py", "build_eval.py"} <= names


@pytest.mark.parametrize("path", builders(), ids=lambda p: str(p.relative_to(ROOT)))
def test_gate_runs_before_the_first_write(path):
    s = path.read_text()
    g = GATE.search(s)
    assert g, f"{path.name}: no CI gate (launch/bundle_gate.py)"
    assert g.start() < WRITES.search(s).start(), f"{path.name}: gate after the first bundle write"


@pytest.fixture
def gate(monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None, **kw):
        calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, RC[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.syspath_prepend(str(ROOT / "launch"))
    monkeypatch.syspath_prepend(str(ROOT / "launch/pth"))
    for m in ("bundle_gate", "ci_gate"):
        sys.modules.pop(m, None)
    return calls


RC = [0]


@pytest.mark.parametrize("module", ["bundle_gate", "ci_gate"])
def test_gate_passes_on_green_and_exits_on_block(gate, module, tmp_path):
    mod = importlib.import_module(module)
    RC[0] = 0
    mod.require(tmp_path, "rl-hero")
    cmd, cwd = gate[-1]
    assert cmd[-4:] == ["gate", "--builder", "rl-hero", "HEAD"] and cmd[1].endswith("tools/ops/ci.py")
    assert cwd == str(tmp_path)
    RC[0] = 1
    with pytest.raises(SystemExit, match="rl-hero build from .* blocked"):
        mod.require(tmp_path, "rl-hero")
    RC[0] = 0
