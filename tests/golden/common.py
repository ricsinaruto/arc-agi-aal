"""rl-min goldens: fixed outputs of the current recipe on recorded fixtures. GOLDEN_RECORD=1 rewrites
tests/golden/data/<name>.json (only for an intentional, reviewed change). Fixtures (recorded R-leg groups, 100s of MB)
live outside git in $ARC_GOLDEN_FIXTURES (default ~/arc-rl-data/rlmin/fixtures); without them the goldens skip."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("data")
FIX = Path(os.environ.get("ARC_GOLDEN_FIXTURES") or Path.home() / "arc-rl-data/rlmin/fixtures")
sys.path.insert(0, str(ROOT / "launch/rl"))
import build as B  # noqa: E402


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def fixture(name):
    p = FIX / name
    if not p.exists():
        pytest.skip(f"golden fixture {p} missing")
    return p


def check(name, value):
    """Compare ``value`` (JSON-able) with tests/golden/data/<name>.json, or record it under GOLDEN_RECORD=1."""
    value = json.loads(json.dumps(value, sort_keys=True, default=str))
    path = DATA / f"{name}.json"
    if os.environ.get("GOLDEN_RECORD") == "1":
        DATA.mkdir(exist_ok=True)
        path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    if not path.exists():  # a container without GOLDEN_RECORD (CI): print it for tools/ops/golden_pull, then fail
        print(f"GOLDEN-RECORD {name} {json.dumps(value, sort_keys=True)}")
        pytest.fail(f"golden {name}: no recorded data")
    assert value == json.loads(path.read_text())


def config(run):
    return json.loads((ROOT / "launch/rl/runs" / f"{run}.json").read_text())


def source_dir(cfg, tmp):
    """The bundle's in-repo seeds (configs/*.json) under ``tmp``: what the loop reads as ``self.source``."""
    for s in cfg["seeds"]:
        if s["path"].endswith(".json"):
            (tmp / s["dest"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / s["path"], tmp / s["dest"])
    return tmp


def loop(run, tmp):
    """The hero Loop as the bundle builds it, without a server or trainer."""
    from arena import rl_loop

    cfg = config(run)
    study = B.inline(cfg["study"])
    lp = rl_loop.Loop.__new__(rl_loop.Loop)
    lp.cfg, lp.study, lp.trial = rl_loop.settings(study["rl_loop"]), study, {}
    lp.mg_screen, lp.source = {"sets": {}}, source_dir(cfg, tmp)
    return lp
