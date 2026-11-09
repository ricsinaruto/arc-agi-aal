"""A failed or empty `modal app list` is UNKNOWN, never "app gone" (24 Oct: api.modal.com unreachable from ~19:20
read as every Modal app ended; rl1-prof-m1cr1 was flagged)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from arc_infra.runops.watch import Snapshot, Watch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/ops"))
import paths  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ops_{name}", ROOT / f"tools/ops/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raises(*a, **k):
    raise subprocess.TimeoutExpired("modal app list", 60)


def _returns(rc, out):
    return lambda *a, **k: subprocess.CompletedProcess(a, rc, stdout=out, stderr="")


@pytest.mark.parametrize("fake", [_raises, _returns(1, ""), _returns(0, ""), _returns(0, "[]"), _returns(0, "{oops")])
def test_failed_or_empty_listing_is_unknown(monkeypatch, fake):
    monkeypatch.setattr(subprocess, "run", fake)
    assert paths.modal_apps("modal") is None

    watch = _load("watch")
    plan = {"run": "r", "platform": "modal", "app_id": "ap-x", "launched": 0}
    assert watch.liveness(plan) is None
    alerts = Watch(plan).check(Snapshot(now=3600, rows=[], live=watch.liveness(plan)))
    assert not [a for a in alerts if a.rule in ("unclean-end", "gone")]

    reconcile = _load("reconcile")
    assert reconcile.app_ended("ap-x") is False

    assert _load("gpu_reconcile").modal_live() is None


def test_listing_rows_normalized_and_gone_judged(monkeypatch):
    out = '[{"App ID": "ap-a", "State": "deployed", "Tasks": "1", "Created at": "2026-10-24T19:00:00+00:00"}]'
    monkeypatch.setattr(subprocess, "run", _returns(0, out))
    assert paths.modal_apps("modal") == [{"app_id": "ap-a", "state": "deployed", "tasks": "1",
                                          "created_at": "2026-10-24T19:00:00+00:00"}]
    reconcile = _load("reconcile")
    assert reconcile.app_ended("ap-a") is False and reconcile.app_ended("ap-b") is True
    watch = _load("watch")
    assert watch.liveness({"run": "r", "platform": "modal", "app_id": "ap-b"}) is False
