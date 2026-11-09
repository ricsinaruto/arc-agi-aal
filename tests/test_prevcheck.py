"""prevcheck.py (30 Oct, prev9h v2 skipped its trial): the built preview passes the trial-start timing gates with the
notebook's own --max-seconds, and the v2 timing (32,400 s for a 31,800 s wall) fails them."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILT = ROOT / "dist/arena-flash-v9-prev9h"
spec = importlib.util.spec_from_file_location("prevcheck", ROOT / "docs/validation-20261009/v9/prevcheck.py")
prevcheck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prevcheck)


def copy(tmp_path):
    if not (BUILT / "experiment.ipynb").is_file():
        pytest.skip("no built preview bundle")
    out = tmp_path / "b"
    for rel in ("experiment.ipynb", "bootstrap_arena.py", "configs/comparison.json"):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BUILT / rel, out / rel)
    shutil.copytree(BUILT / "src/arena", out / "src/arena", ignore=shutil.ignore_patterns("__pycache__"))
    return out


def test_built_preview_starts_its_trial(tmp_path):
    r = prevcheck.check(copy(tmp_path))
    assert r["max_seconds"] == 33600 and r["slack_at_setup_max"] >= 0


def test_v2_timing_fails(tmp_path):
    b = copy(tmp_path)
    book = json.loads((b / "experiment.ipynb").read_text())
    for c in book["cells"]:
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        c["source"] = src.replace("'--max-seconds', '33600'", "'--max-seconds', '32400'")
    (b / "experiment.ipynb").write_text(json.dumps(book))
    with pytest.raises(AssertionError, match="bootstrap skips"):
        prevcheck.check(b)
