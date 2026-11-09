"""v17 a7 (30 Oct): served_before used outside its scope killed U20 after 36 min; the RL build refuses undefined names."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "launch" / "rl"))

import build  # noqa: E402


def test_build_gate_refuses_an_undefined_name(tmp_path, monkeypatch):
    if not Path(sys.executable).with_name("ruff").exists():
        pytest.skip("no ruff beside this python")
    for d in build.LINT_DIRS:
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "scripts/posttrain/t.py").write_text(
        "def op_update():\n    before = {}\n    return update_once()\n\n\n"
        "def update_once():\n    return {k: 1 for k in before}\n")
    with pytest.raises(SystemExit, match="F821"):
        build.undefined_names(tmp_path)
    (tmp_path / "scripts/posttrain/t.py").write_text("def f():\n    before = {}\n    return before\n")
    build.undefined_names(tmp_path)
