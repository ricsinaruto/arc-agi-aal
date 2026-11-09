"""Typed launch configs: strict load (no defaults, no unknown keys, types), roster consistency, diff vs the previous
attempt from the record, and the dry run's anchor / roster / data / gate checks."""
import copy
import json
from dataclasses import dataclass

import pytest

from arc_infra.runops import launchcfg as L
from arc_infra.runops import records as R


@dataclass(frozen=True)
class Hero(L.LaunchConfig):
    bundle: str
    resume_from: tuple[str, ...]
    env: dict[str, str]
    knobs: str | None


BASE = {"run": "rl1-hero-x", "arm": "rl", "platform": "modal", "gpu": "2 x B300", "gpu_count": 2, "bound_usd": 50,
        "max_seconds": 3600, "authorization": "test", "what": "a test run",
        "anchor": {"run": "in-run", "note": "base read at step 0"},
        "roster": {"name": "held-out-10", "ids": ["tr87", "r11l"], "size": 2},
        "data": [], "bundle": "dist/x", "resume_from": [], "env": {"ARC_RL1_STALL": "1200"}, "knobs": None}


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/modal-queue.json").write_text(json.dumps({"live": [], "queued": []}))
    (tmp_path / "docs/spend.json").write_text(json.dumps({"runs": []}))
    (tmp_path / "docs/budget.json").write_text(json.dumps({"allowances": {"rl": {"gpus": {"B300": 4}}},
                                                          "caps": {"gpus": {"B300": 8}}}))
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setenv("ARC_RECORDS_NO_SYNC", "1")
    return tmp_path


def test_strict_load():
    cfg = L.from_dict(Hero, BASE)
    assert cfg.bound_usd == 50.0 and cfg.roster.ids == ("tr87", "r11l") and cfg.env["ARC_RL1_STALL"] == "1200"
    for bad, msg in (({k: v for k, v in BASE.items() if k != "knobs"}, "missing \\['knobs'\\]"),
                     ({**BASE, "lr": 1e-4}, "unknown \\['lr'\\]"),
                     ({**BASE, "gpu_count": "2"}, "gpu_count: expected int"),
                     ({**BASE, "roster": {"name": "r", "ids": ["a", "a"], "size": 2}}, "2 distinct|1 distinct"),
                     ({**BASE, "roster": {"name": "r", "ids": ["a"], "size": 2}}, "size 2 but 1 ids"),
                     ({**BASE, "anchor": {"run": "x"}}, "anchor: missing \\['note'\\]")):
        with pytest.raises(L.ConfigError, match=msg):
            L.from_dict(Hero, bad)

    @dataclass(frozen=True)
    class WithDefault(L.LaunchConfig):
        lr: float = 1e-4
    with pytest.raises(L.ConfigError, match="has a default"):
        L.from_dict(WithDefault, {**{k: v for k, v in BASE.items() if k in {f for f in L.LaunchConfig.__annotations__}},
                                  "lr": 1e-4})


def test_diff_vs_previous_attempt(root):
    cfg = L.from_dict(Hero, BASE)
    assert "(first attempt)" in L.report(cfg)
    R.reserve(cfg.run, arm="rl", platform="modal", gpu=cfg.gpu, gpu_count=2, bound_usd=1, max_seconds=1,
              authorization="t", what="t")
    L.remember(cfg)
    R.settle(cfg.run, "stopped", end_cause="bug:oom")
    nxt = L.from_dict(Hero, {**BASE, "env": {"ARC_RL1_STALL": "900"}, "resume_from": ["validation/rl1-hero-x"]})
    rep = L.report(nxt)
    assert 'env.ARC_RL1_STALL: "1200" -> "900"' in rep and 'resume_from: [] -> ["validation/rl1-hero-x"]' in rep


def test_dry_run_checks_anchor_roster_data_gate(root, tmp_path):
    good = tmp_path / "pool.json"
    good.write_text("{}")
    (tmp_path / "empty").mkdir()
    cfg = L.from_dict(Hero, {**BASE, "data": [str(good)]})
    assert L.dry_run(cfg, roster_ids=["tr87", "r11l"]) == []
    bad = L.from_dict(Hero, {**BASE, "anchor": {"run": "rl1-base-read", "note": "base"},
                             "data": [str(good), str(tmp_path / "nope"), str(tmp_path / "empty")],
                             "gpu": "9 x B300", "gpu_count": 9})
    probs = L.dry_run(bad, roster_ids=["tr87"])
    joined = "\n".join(probs)
    for want in ("roster held-out-10: config 2 ids vs bundle 1", "anchor rl1-base-read: no run record",
                 "nope: missing", "empty: empty dir", "gate: rl: 0 B300 live + 9 > its allowance 4"):
        assert want in joined, joined
    R.reserve("rl1-base-read", arm="rl", platform="modal", gpu="B300", bound_usd=1, max_seconds=1,
              authorization="t", what="t")
    assert not any(p.startswith("anchor") for p in L.dry_run(bad, gate=False))  # scheduled/running counts
    R.settle("rl1-base-read", "failed", end_cause="bug:oom")
    assert any("not completed" in p for p in L.dry_run(bad, gate=False))


def test_volume_data_paths_use_modal_listing(monkeypatch):
    calls = []

    class P:
        def __init__(self, rc, out):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def fake(cmd, **kw):
        calls.append(cmd)
        return P(0, json.dumps([{"Filename": "a", "Type": "file", "Size": 3}])) if cmd[-1] == "validation/ok" \
            else P(1, "No such file")
    monkeypatch.setattr(L.subprocess, "run", fake)
    probs = L.check_data(["arc-prolong-runs:validation/ok", "arc-prolong-runs:validation/gone"], modal_bin="m")
    assert len(probs) == 1 and "validation/gone: not found" in probs[0] and calls[0][:3] == ["m", "volume", "ls"]
    assert copy.deepcopy(BASE) == BASE
