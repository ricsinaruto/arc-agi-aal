"""Read selection and start-read reuse: one file per session, previous round's end reads reused only
for the same weights, files and frozen held-out b. Needs torch (Modal CPU)."""
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
sys.path.insert(0, "/opt/heal")
import heal_kd  # noqa: E402


def make(folder, names, reply):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"x" * 10)
    rows = [{"file": n, "reply_tokens": reply.get(n, 0)} for n in names]
    (folder / "index.json").write_text(json.dumps({"rows": rows}))


def test_session_select_covers_every_session(tmp_path, monkeypatch):
    names = ["k000--e010.pt", "k000--e050.pt", "k001--e020.pt", "k002--e005.pt", "k002--e030.pt", "k002--e090.pt"]
    make(tmp_path / "d", names, {"k000--e050.pt": 900, "k000--e010.pt": 100, "k002--e030.pt": 700, "k002--e090.pt": 50})
    monkeypatch.delenv("HEAL_READ_SELECT", raising=False)
    assert [f.name for f in heal_kd.read_files(tmp_path / "d", 3)] == names[:3]  # by name: k002 never read
    monkeypatch.setenv("HEAL_READ_SELECT", "session")
    got = [f.name for f in heal_kd.read_files(tmp_path / "d", 20)]
    assert got == ["k000--e050.pt", "k001--e020.pt", "k002--e030.pt"]
    assert [f.name for f in heal_kd.read_files(tmp_path / "d", 2)] == got[:2]


def test_preread_reused_only_for_same_weights_and_files(tmp_path, monkeypatch):
    held = tmp_path / "h"
    make(held, [f"h-k{i:03d}--e001.pt" for i in range(4)], {})
    held2 = tmp_path / "d"
    make(held2, [f"k{i:03d}--e001.pt" for i in range(3)], {})
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / "student").mkdir()
    ev = prev / "events.jsonl"
    ev.write_text(json.dumps({"event": "heldout", "files": 4, "kd_reply": 0.25}) + "\n"
                  + json.dumps({"event": "heldout2", "folder": str(held2), "files": 3, "kd_reply": 0.19,
                                "sampled_reverse": 0.15}) + "\n")
    args = types.SimpleNamespace(student=str(prev / "student"), heldout=str(held), heldout_files=20)
    monkeypatch.setenv("HEAL_PREREAD_FROM", str(ev))
    monkeypatch.setenv("HEAL_HELDOUT2_FILES", "20")
    monkeypatch.setenv("HEAL_SAMPLED_METRIC", "1")
    monkeypatch.delenv("HEAL_HELDOUT2_BEH", raising=False)
    assert heal_kd.preread_from(args, str(held2)) == {}  # sampled metric without a frozen b: b would differ
    monkeypatch.setenv("HEAL_HELDOUT2_BEH", str(tmp_path / "beh.pt"))
    got = heal_kd.preread_from(args, str(held2))
    assert got["heldout"]["kd_reply"] == 0.25 and got["heldout2"]["sampled_reverse"] == 0.15
    other = types.SimpleNamespace(student=str(tmp_path / "elsewhere"), heldout=str(held), heldout_files=20)
    assert heal_kd.preread_from(other, str(held2)) == {}  # other weights
    monkeypatch.setenv("HEAL_HELDOUT2_FILES", "2")
    assert "heldout2" not in heal_kd.preread_from(args, str(held2))  # other file count
