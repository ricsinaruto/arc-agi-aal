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
