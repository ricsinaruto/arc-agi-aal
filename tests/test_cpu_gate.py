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
