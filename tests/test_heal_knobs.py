"""heal_kd bench knobs (PTH 23 Oct): HEAL_CHUNK_AFTER's step rule, read from heal_kd's source (no torch import)."""
import ast
import os
from pathlib import Path

import pytest
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)

SRC = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "heal_kd.py"


def chunk_after():
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chunk_after"]
    ns = {"os": os}
    exec(compile(ast.Module(keep, []), "heal_kd_chunk_after", "exec"), ns)
    return ns["chunk_after"]


def test_chunk_after(monkeypatch):
    f = chunk_after()
    monkeypatch.delenv("HEAL_CHUNK_AFTER", raising=False)
    assert [f(s, 4096) for s in (0, 1, 9)] == [4096, 4096, 4096]
    monkeypatch.setenv("HEAL_CHUNK_AFTER", "4:16384")
    assert [f(s, 4096) for s in (1, 3, 4, 5, 100)] == [4096, 4096, 16384, 16384, 16384]
    for bad in ("4", "x:1", "0:64", "4:0", "4:64:1"):
        monkeypatch.setenv("HEAL_CHUNK_AFTER", bad)
        with pytest.raises(ValueError):
            f(1, 4096)
