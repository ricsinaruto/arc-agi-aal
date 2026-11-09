"""trunk/rl copy of main's test_runops_restart rerun test (the bundle's arc_infra is trunk's)."""

import pytest

from arc_infra.runops import hostmem
from arc_infra.runops._io import read_json
from arc_infra.runops.restart import RestartBlocked, RestartGuard, classify

OK_MEM = {"total_gib": 2000.0, "avail_gib": 900.0}
THRASH_MEM = {"total_gib": 2000.0, "avail_gib": 3.0}


@pytest.fixture(autouse=True)
def mem(monkeypatch):
    box = {"m": OK_MEM}
    monkeypatch.setattr(hostmem, "sample", lambda *a, **k: dict(box["m"]))
    monkeypatch.delenv("ARC_RESTART_ACK", raising=False)
    return box


def guard(tmp_path, call="fc-1", **kw):
    return RestartGuard(tmp_path, "run-x", call_id=call, **kw)


def die(g, signal_name=None):
    """An attempt that started and was killed: no end() call."""
    g.start(checkpoint_step=0)
    if signal_name:
        g.me["signal"] = {"name": signal_name, "t": 1.0}
        g._save()


def state(tmp_path):
    return read_json(tmp_path / ".runops/run-x/attempts.json")


def test_platform_rerun_without_signal_is_a_preemption(tmp_path, mem):
    """v17 a8 (30 Oct): Modal preempted the container, no signal reached the record; the re-run of the same input
    with healthy heartbeats auto-resumes under policy resume. A relaunch (new call id) stays unknown."""
    die(guard(tmp_path, policy="resume"))
    s = guard(tmp_path, policy="resume").start(checkpoint_step=21)
    assert s.resume and s.incident["cause"].startswith("preemption (platform re-run")
    assert classify({"mem_last": OK_MEM, "mem_min": OK_MEM}, rerun=True).startswith("preemption")
    assert classify({"mem_last": THRASH_MEM, "mem_min": THRASH_MEM}, rerun=True) == "host-memory"
    assert classify({}, rerun=True).startswith("unknown")  # no heartbeat evidence
    other = tmp_path / "relaunch"
    die(guard(other, call="fc-9", policy="resume"))
    with pytest.raises(RestartBlocked) as e:
        guard(other, call="fc-10", policy="resume").start(checkpoint_step=21)
    assert e.value.incident["cause"].startswith("unknown") and not e.value.incident["auto"]
