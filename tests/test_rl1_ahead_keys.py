"""Ahead windows are the call's windows (gate legs 28 Oct: 0 of 192 ahead windows hit in every rl_seq run; the ahead
plan expanded bare attempts, the call whole_segment ones). The CPU stand-in trainer runs the real trainer's planning
on the loop's ahead file and reports the hits, so the preflight judge and the watcher see the share."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

import rl1_encode  # noqa: E402

from arena import fake_trainer  # noqa: E402


def fake_expand(spec, window):
    if not spec.get("whole"):
        return [spec]
    return [{**spec, "whole": False, "end": e, "name": f"{spec['name']}#{i}"} for i, e in enumerate((100, 200))]


def test_stand_in_reports_the_real_trainers_ahead_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(rl1_encode, "expand_one", fake_expand)
    roll = tmp_path / "rollout.jsonl"
    roll.write_text("{}\n")
    attempts = [{"name": n, "w": 1.0, "rollout": str(roll), "whole": True} for n in ("a", "b")]
    fake = fake_trainer.Fake(16384, tmp_path)
    call = {"attempts": attempts, "whole_segment": True, "signed": True, "share_prefix": False, "max_windows": 100}
    assert fake.ahead_encode(call) is None  # no ahead file: nothing to judge
    (tmp_path / "ahead-early.json").write_text(json.dumps({"token": "t", "ahead": attempts, "encode_procs": 2,
                                                           "whole_segment": True, "signed": True,
                                                           "share_prefix": False, "max_windows": 100}))
    assert fake.ahead_encode(call) == {"ahead_hits": 4, "encoded": 4}
    # the 28 Oct class: the ahead plan without the call's whole_segment -> no key matches
    (tmp_path / "ahead-early.json").write_text(json.dumps({"token": "t2", "ahead": attempts, "encode_procs": 2,
                                                           "signed": True, "share_prefix": False, "max_windows": 100}))
    assert fake.ahead_encode(call)["ahead_hits"] == 0
