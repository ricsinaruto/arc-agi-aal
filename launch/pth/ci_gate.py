"""PTH builders' import of the shared gate (launch/bundle_gate.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundle_gate import require as _require  # noqa: E402


def require(worktree, builder="pth"):
    _require(worktree, builder)
