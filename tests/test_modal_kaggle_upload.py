"""scripts/modal_kaggle_upload.py reserves through the records library, never docs/spend.json (2 Nov)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("modal")
ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("modal_kaggle_upload", ROOT / "scripts/modal_kaggle_upload.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRecords:
    def __init__(self):
        self.calls = []

    def reserve(self, run, **kw):
        self.calls.append((run, kw))


def test_reserve_goes_through_records(monkeypatch, tmp_path):
    mod = load()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setenv("ARC_UPLOAD_ARM", "l4")
    monkeypatch.setenv("ARC_UPLOAD_AUTH", "owner test")
    fake = FakeRecords()
    bound = mod.reserve("x-upload", {"source": "arc-prolong-assets:/a", "dataset": "richardcsaky/d"}, records=fake)
    (run, kw), = fake.calls
    assert run == "x-upload" and kw["arm"] == "l4" and kw["gpu"] == "none" and kw["gpu_count"] == 0
    assert kw["platform"] == "modal" and kw["bound_usd"] == bound > 0 and kw["authorization"] == "owner test"
    assert kw["root"] == tmp_path and "richardcsaky/d" in kw["what"]
    assert not (tmp_path / "docs/spend.json").exists()


def test_reserve_needs_arm_and_auth(monkeypatch):
    mod = load()
    monkeypatch.delenv("ARC_UPLOAD_ARM", raising=False)
    monkeypatch.delenv("ARC_UPLOAD_AUTH", raising=False)
    with pytest.raises(SystemExit):
        mod.reserve("x", {"source": "s", "dataset": "d"}, records=FakeRecords())
