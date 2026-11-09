"""The CPU launch gate runner: parallel steps, a failing / crashing / over-budget step blocks."""
import importlib.util
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_gate", ROOT / "tools/ops/cpu_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_all_pass_in_parallel():
    t0 = time.time()
    r = gate.run_gate("r", {"a": lambda: (time.sleep(0.3), (True, "ok"))[1],
                            "b": lambda: (time.sleep(0.3), (True, "ok"))[1]}, budget=5)
    assert r["ok"] and time.time() - t0 < 0.55  # parallel, not 0.6 s


def test_failure_crash_and_budget_block():
    def boom():
        raise RuntimeError("x")
    r = gate.run_gate("r", {"fail": lambda: (False, "patch mm_warmup: renderers/base.py sha mismatch"),
                            "crash": boom, "slow": lambda: (time.sleep(2), (True, ""))[1],
                            "fine": lambda: (True, "")}, budget=0.5)
    s = r["steps"]
    assert not r["ok"] and s["fine"]["ok"]
    assert "sha mismatch" in s["fail"]["why"] and "crashed" in s["crash"]["why"] and "budget" in s["slow"]["why"]
    assert not gate.run_gate("r", {}, budget=1)["ok"]  # no steps is not a pass


def test_cmd_step_reports_last_line(tmp_path):
    ok, why = gate.step_cmd("echo checking; echo 'ARC_PATCH_CHECK FAIL mxfp8 anchor'; exit 3", str(tmp_path))
    assert not ok and why == "ARC_PATCH_CHECK FAIL mxfp8 anchor"
    assert gate.step_cmd("true", str(tmp_path)) == (True, "exit 0")


def test_a_failing_command_before_tail_fails_the_step(tmp_path):
    """25 Oct: `launch.py check ... | tail -3` printed CONFIG REFUSED and exited 1, the gate said PASS."""
    ok2, why2 = gate.step_cmd("(echo 'CONFIG REFUSED'; echo 'x: attempt 1 is still running'; exit 1) | tail -3",
                              str(tmp_path))
    assert not ok2 and why2 == "x: attempt 1 is still running"
    assert gate.step_cmd("yes | head -1", str(tmp_path)) == (True, "y")  # SIGPIPE from head is not a failure
    rec = gate.run_gate("r", {"refused": lambda: gate.step_cmd("(echo REFUSED; exit 1) | tail -1", str(tmp_path))}, 30)
    assert not rec["ok"] and not rec["steps"]["refused"]["ok"]


def test_volume_alert_at_80_percent():
    spec = importlib.util.spec_from_file_location("volume_usage", ROOT / "tools/ops/volume_usage.py")
    vu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vu)
    v = {"volume": "arc-prolong-runs", "inodes": 420_000, "bytes": 10, "inode_cap": 500_000, "byte_cap": None,
         "top": [["validation/x", 300_000, 1]]}
    assert len(vu.alerts(v)) == 1 and "84%" in vu.alerts(v)[0] and "validation/x 300,000" in vu.alerts(v)[0]
    assert vu.alerts({**v, "inodes": 399_999}) == []
    assert "byte cap" in vu.alerts({**v, "inodes": 1, "bytes": 81, "byte_cap": 100})[0]


def test_records_backup_and_restore(tmp_path, monkeypatch):
    """Hourly records backup (24 Oct): tar of docs/runs + budget.json to a remote, keep N, restore into a dir."""
    import json
    import shutil

    import pytest
    if not shutil.which("rclone"):
        pytest.skip("rclone not installed")
    spec = importlib.util.spec_from_file_location("records_backup", ROOT / "tools/ops/records_backup.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    root, rem = tmp_path / "root", tmp_path / "remote"
    (root / "docs/runs").mkdir(parents=True)
    rem.mkdir()
    for i in range(5):
        (root / f"docs/runs/r{i}.json").write_text(json.dumps({"run": f"r{i}"}))
    (root / "docs/runs/r0.json.lock").write_text("")
    (root / "docs/budget.json").write_text('{"caps": 1}')
    names = []
    for _ in range(3):
        names.append(rb.backup(root, str(rem), keep=2)[0])
        time.sleep(1.05)  # names carry the second
    assert rb.listing(str(rem)) == names[1:]
    (root / "docs/runs/r9.json").write_text("{}")  # after the last backup
    name, n = rb.restore(str(rem), None, tmp_path / "out")
    assert name == names[-1] and n == 6
    out = tmp_path / "out/docs"
    assert sorted(p.name for p in (out / "runs").iterdir()) == [f"r{i}.json" for i in range(5)]
    assert (out / "runs/r3.json").read_text() == (root / "docs/runs/r3.json").read_text()
    assert (out / "budget.json").read_text() == '{"caps": 1}'


def test_fork_report_lists_shared_functions_that_differ(tmp_path):
    spec = importlib.util.spec_from_file_location("fork_report", ROOT / "tools/ops/fork_report.py")
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    (tmp_path / "scripts/posttrain/pth").mkdir(parents=True)
    (tmp_path / "scripts/posttrain/t.py").write_text(
        "def a():\n    '''rl doc'''\n    return 1\nclass T:\n    def m(self):\n        return 2\ndef rl_only():\n    pass\n")
    (tmp_path / "scripts/posttrain/pth/t.py").write_text(
        "# OWNER: PTH\ndef a():\n    '''pth doc'''\n    return 1\nclass T:\n    def m(self):\n        return 3\n")
    r = fr.report(tmp_path)["t.py"]
    assert r["differ"] == ["T.m"] and r["same"] == 1 and r["rl_only"] == 1 and r["pth_only"] == 0
    assert fr.report(tmp_path / "none") == {}


def test_step_cmd_fails_on_a_pytest_failure_summary_even_with_exit_0(tmp_path):
    """v6m2 25 Oct: `pytest ... | tail -1` exited 0 and the gate printed PASS for "3 failed, 160 passed"."""
    ok, why = gate.step_cmd("echo '3 failed, 160 passed in 12.3s'; exit 0", str(tmp_path))
    assert not ok and "3 failed" in why
    assert gate.step_cmd("echo '1 error in 0.5s'", str(tmp_path))[0] is False
    assert gate.step_cmd("echo '160 passed in 9.1s'", str(tmp_path)) == (True, "160 passed in 9.1s")
    assert gate.step_cmd("echo 'no failed jobs'; echo '0 errors seen'", str(tmp_path))[0] is True


def test_config_step_runs_the_arms_own_checks(tmp_path, monkeypatch):
    import json

    from arc_infra.runops import launchcfg

    mod = tmp_path / "armcfg.py"
    mod.write_text("from dataclasses import dataclass\nfrom arc_infra.runops import launchcfg\n"
                   "@dataclass(frozen=True)\nclass C(launchcfg.LaunchConfig):\n    lr: float\n"
                   "def checks(cfg):\n    return [] if cfg.lr < 1e-4 else [f'lr {cfg.lr} too high']\n")
    cfg = {"run": "r1", "arm": "sft", "platform": "modal", "gpu": "B300", "gpu_count": 1, "bound_usd": 1.0,
           "max_seconds": 60, "authorization": "t", "what": "t", "anchor": {"run": "in-run", "note": ""},
           "roster": {"name": "x", "ids": [], "size": 0}, "data": [], "lr": 1e-3}
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(launchcfg, "dry_run", lambda c, **kw: [])
    ok, why = gate.step_config(str(tmp_path / "c.json"), f"{mod}:C")
    assert not ok and why == "lr 0.001 too high"
    (tmp_path / "c.json").write_text(json.dumps({**cfg, "lr": 1e-5}))
    assert gate.step_config(str(tmp_path / "c.json"), f"{mod}:C")[0]
