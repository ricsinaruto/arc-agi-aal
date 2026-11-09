import os
import signal
import time
from importlib import reload

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


def test_fresh_start_and_clean_end(tmp_path):
    g = guard(tmp_path)
    with g.run() as s:
        assert (s.attempt, s.restart, s.resume, s.incident) == (1, False, False, None)
    st = state(tmp_path)
    assert st["attempts"][0]["ended"]["status"] == "ok" and st["incidents"] == []
    with guard(tmp_path, call="fc-2").run() as s:  # a deliberate relaunch after a clean end needs no ack
        assert s.attempt == 2 and not s.restart


def test_stop_policy_blocks_auto_restart(tmp_path):
    die(guard(tmp_path), "SIGINT")
    with pytest.raises(RestartBlocked) as e:
        guard(tmp_path).start(checkpoint_step=40)
    inc = e.value.incident
    assert inc["auto"] and inc["blocked"] and inc["cause"] == "preemption (SIGINT)" and "policy 'stop'" in inc["why"]
    st = state(tmp_path)
    assert [a["ended"]["status"] for a in st["attempts"]] == ["died", "blocked"]


def test_resume_policy_resumes_preemption_with_checkpoint(tmp_path):
    die(guard(tmp_path, policy="resume"), "SIGINT")
    s = guard(tmp_path, policy="resume").start(checkpoint_step=40)
    assert s.restart and s.resume and s.incident["resumed"] and not s.incident["blocked"]


def test_resume_policy_caps_resumes(tmp_path):
    for _ in range(4):  # start, two resumes, then the third restart
        g = guard(tmp_path, policy="resume", max_resumes=2)
        try:
            g.start(checkpoint_step=40)
        except RestartBlocked as e:
            assert "2 auto-resumes already" in e.incident["why"]
            break
        g.me["signal"] = {"name": "SIGINT", "t": 1.0}
        g._save()
    else:
        pytest.fail("third restart was not blocked")


def test_c7b_no_checkpoint_blocks(tmp_path, mem):
    """c7b: died during the first checkpoint copy, Modal restarted it from step 0."""
    die(guard(tmp_path, policy="resume"), "SIGINT")
    with pytest.raises(RestartBlocked) as e:
        guard(tmp_path, policy="resume").start(checkpoint_step=0)
    assert "no checkpoint" in e.value.incident["why"]


def test_host_memory_evidence_overrides_preemption_label(tmp_path, mem):
    g = guard(tmp_path, policy="resume")
    die(g, "SIGINT")
    mem["m"] = THRASH_MEM
    g.beat()  # the last heartbeat saw the host near exhaustion
    mem["m"] = OK_MEM
    with pytest.raises(RestartBlocked) as e:
        guard(tmp_path, policy="resume").start(checkpoint_step=40)
    assert e.value.incident["cause"] == "host-memory"
    assert e.value.incident["mem_last"]["avail_gib"] == 3.0


def test_no_signal_is_unknown_death():
    assert classify({"mem_last": OK_MEM}).startswith("unknown")


def test_relaunch_needs_ack(tmp_path):
    die(guard(tmp_path))
    with pytest.raises(RestartBlocked):
        guard(tmp_path).start()  # Modal's auto-restart: blocked
    with pytest.raises(RestartBlocked) as e:
        guard(tmp_path, call="fc-2").start()  # agent relaunch without reading the cause
    assert e.value.incident["dead_attempt"] == 1
    with pytest.raises(ValueError):
        guard(tmp_path, call="fc-3", ack="ok").start()
    s = guard(tmp_path, call="fc-4", ack="host RAM thrash at first ckpt copy").start(checkpoint_step=10)
    assert s.restart and s.resume and s.incident["ack"]["cause_read"].startswith("host RAM")
    assert not any(i["blocked"] for i in state(tmp_path)["incidents"])


def test_launch_ack_does_not_ack_a_later_auto_restart(tmp_path):
    g = guard(tmp_path, ack="earlier death read: preemption at step 3")
    g.start()  # first attempt of this call: nothing open, the ack is a no-op
    with pytest.raises(RestartBlocked):  # Modal re-runs the same input (same env, same call id)
        guard(tmp_path, ack="earlier death read: preemption at step 3").start(checkpoint_step=5)


def test_manual_relaunch_after_death_with_ack(tmp_path):
    die(guard(tmp_path))
    s = guard(tmp_path, call="fc-2", ack="worker lost, no signal; relaunch from ckpt").start(checkpoint_step=8)
    assert s.restart and s.resume and not s.incident["auto"]


def test_sigint_is_recorded_and_leaves_attempt_open(tmp_path):
    g = guard(tmp_path, beat_s=3600)
    with pytest.raises(KeyboardInterrupt):
        with g.run():
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(1)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    a = state(tmp_path)["attempts"][0]
    assert a["signal"]["name"] == "SIGINT" and a["ended"] is None


def test_hostmem_sample_parses_proc_and_cgroup():
    meminfo = "MemTotal: 2097152000 kB\nMemAvailable: 1048576000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
    real = reload(hostmem)  # undo the fixture's stub
    r = real.sample(meminfo, "VmRSS: 1048576 kB\n", cgroup=(900 * 1024**3, 1000 * 1024**3))
    assert r["avail_gib"] == 100.0 and r["total_gib"] == 1000.0 and r["rss_gib"] == 1.0
    assert real.tight({"total_gib": 1000.0, "avail_gib": 40.0}) and not real.tight({"total_gib": 1000.0, "avail_gib": 60.0})


def test_preflight_stand_in_resumes(tmp_path):
    g = guard(tmp_path, policy="resume")
    g.start()
    g.record_preemption()  # then os._exit in the preflight
    s = guard(tmp_path, policy="resume").start(checkpoint_step=1)
    assert s.resume and s.incident["signal"]["note"] == "preflight stand-in"


def test_hostmem_detail_reads_processes_shm_tmp_psi(tmp_path):
    real = reload(hostmem)
    proc = tmp_path / "proc"
    for pid, name, rss, pin in ((10, "python", 50 * 1024, 4 * 1024), (11, "vllm", 200 * 1024, 0)):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "status").write_text(f"Name:\t{name}\nVmRSS:\t{rss} kB\nVmPin:\t{pin} kB\nVmLck:\t0 kB\n")
    (proc / "self").mkdir()
    tmp = tmp_path / "tmp"
    (tmp / "d").mkdir(parents=True)
    (tmp / "d" / "f").write_bytes(b"x" * 1_048_576)
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.stat").write_text(f"anon {3 * 1024**3}\nfile {1024**3}\nshmem {2 * 1024**3}\n")
    (cg / "memory.pressure").write_text("some avg10=12.50 avg60=3.0 avg300=1.0 total=1\nfull avg10=0.00 avg60=0 "
                                        "avg300=0 total=0\n")
    d = real.detail(proc=str(proc), shm=str(tmp_path), tmp=str(tmp), cgroup_dir=str(cg))
    assert d["infra.host_rss_total_gib"] == round(250 / 1024, 3) and d["infra.host_pinned_gib"] == round(4 / 1024, 3)
    assert list(k for k in d if k.startswith("infra.host_rss_gib."))[0] == "infra.host_rss_gib.vllm-11"
    assert 0.0009 < d["infra.host_tmp_gib"] < 0.0011 and d["infra.host_cg_shmem_gib"] == 2.0
    assert d["infra.host_psi_some"] == 12.5 and not real.thrashing(d) and real.thrashing({"infra.host_psi_full": 11})


def test_hostguard_aborts_after_two_bad_readings(tmp_path):
    from arc_infra.runops.hostguard import HostGuard
    from arc_infra.runops.metrics import MetricsWriter, read_rows

    w = MetricsWriter(tmp_path / "m.jsonl")
    seq = iter([(100.0, 0.0), (3.0, 0.0), (90.0, 0.0), (90.0, 55.0), (90.0, 60.0), (2.0, 0.0)])
    cur = {}

    def sample():
        cur["a"], cur["p"] = next(seq)
        return {"total_gib": 1000.0, "avail_gib": cur["a"]}

    aborts = []
    g = HostGuard(w, sample=sample, detail=lambda: {"infra.host_psi_some": cur["p"], "infra.host_shm_gib": 1.5},
                  abort=aborts.append, log=None)
    for _ in range(3):
        g.read()
    assert aborts == []  # one low reading is not enough
    g.read()
    g.read()  # two thrashing readings in a row
    assert len(aborts) == 1 and "PSI some 60.0" in aborts[0]
    g.read()
    assert len(aborts) == 1  # once
    rows = read_rows(tmp_path / "m.jsonl")
    assert rows[0]["m"]["infra.host_shm_gib"] == 1.5 and any("host RAM abort" in (r.get("alert") or "") for r in rows)
