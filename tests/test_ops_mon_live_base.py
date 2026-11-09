"""mon reads each run's live folder directly (28 Oct): record live_base, else the watch plan's arena spec, else the
folder an earlier pass found; no plan and no cache = probe (None)."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_live_base_from_record_plan_or_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_OPS_SCRATCH", str(tmp_path / "ops"))
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    import paths

    monkeypatch.setattr(paths, "SCRATCH", tmp_path / "ops")
    spec = importlib.util.spec_from_file_location("mon_t", ROOT / "tools/ops/mon.py")
    mon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mon)
    monkeypatch.setattr(mon, "ROOT", str(tmp_path))
    monkeypatch.setattr(mon, "OUT", tmp_path / "mon")
    (tmp_path / "mon").mkdir()
    (tmp_path / "docs/runs").mkdir(parents=True)
    (tmp_path / "ops/watch-plans").mkdir(parents=True)
    assert mon.live_base("zz-none") is None  # probe
    (tmp_path / "ops/watch-plans/zz-rl.json").write_text(json.dumps(
        {"arena": {"sampler": "modal:arc-prolong-runs:validation/zz-rl/live/sampler.jsonl"}}))
    assert mon.live_base("zz-rl") == "validation/zz-rl/live/"
    (tmp_path / "docs/runs/zz-arena.json").write_text(json.dumps(
        {"plan": {"arena": {"progress": "modal:arc-prolong-runs:validation/zz-arena/arena/live/progress.jsonl"}}}))
    assert mon.live_base("zz-arena") == "validation/zz-arena/arena/live/"
    (tmp_path / "docs/runs/zz-tp.json").write_text(json.dumps({"live_base": "throughput/zz-tp/live/"}))
    assert mon.live_base("zz-tp") == "throughput/zz-tp/live/"
    (tmp_path / "mon/zz-cached.base").write_text("validation/zz-cached/live/\n")
    assert mon.live_base("zz-cached") == "validation/zz-cached/live/"


def test_external_calls_are_capped(tmp_path, monkeypatch):
    """28 Oct: a pass sat 7 min on volume gets; every external call in a mon pass has a hard cap and reads as failed."""
    import subprocess
    import time

    monkeypatch.setenv("ARC_OPS_SCRATCH", str(tmp_path / "ops"))
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    sys.path.insert(0, str(ROOT / "src"))
    import paths
    from arc_infra.runops import volfetch

    monkeypatch.setattr(paths, "SCRATCH", tmp_path / "ops")
    spec = importlib.util.spec_from_file_location("mon_c", ROOT / "tools/ops/mon.py")
    mon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mon)
    monkeypatch.setattr(subprocess, "run", subprocess.run)  # restored after the test
    monkeypatch.setattr(volfetch, "run", volfetch.run)
    monkeypatch.setattr(time, "sleep", time.sleep)
    monkeypatch.setattr(mon, "CALL_S", 1.0)
    monkeypatch.setattr(mon, "MIN_S", 0.5)
    mon.cap_external_calls()
    t = time.time()
    r = subprocess.run(["sleep", "5"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 124 and r.stdout == "" and time.time() - t < 3
    t = time.time()
    assert volfetch.run(["sleep", "5"], timeout=300) is None and time.time() - t < 3
    assert subprocess.run(["true"]).returncode == 0


def test_page_pass_reads_caches_never_remotes(tmp_path, monkeypatch):
    """28 Oct: the page pass reads the fetch loop's caches; missing = pending, old = 'stale since', an old listing = unknown."""
    import time

    monkeypatch.setenv("ARC_OPS_SCRATCH", str(tmp_path / "ops"))
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    import paths

    monkeypatch.setattr(paths, "SCRATCH", tmp_path / "ops")
    spec = importlib.util.spec_from_file_location("mon_s", ROOT / "tools/ops/mon.py")
    mon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mon)
    monkeypatch.setattr(mon, "SRC", tmp_path / "src")
    (tmp_path / "src").mkdir()
    row, apps = mon.cached("row"), mon.cached("modal_apps")
    assert row("r1")["state"].startswith("first read pending") and apps() is None
    now = time.time()
    (tmp_path / "src" / f"{mon._skey('row', ['r1'])}.json").write_text(json.dumps({"t": now - 10, "v": {"run": "r1", "min": 5}}))
    (tmp_path / "src" / "modal_apps.json").write_text(json.dumps({"t": now - 600, "v": [{"app_id": "a"}]}))
    mon.STALE.clear()
    assert row("r1") == {"run": "r1", "min": 5} and apps() is None  # a 10-min-old listing is unknown, never "gone"
    assert any("modal_apps(): stale since" in x for x in mon.STALE) and not any("row(r1)" in x for x in mon.STALE)
    mon.write_wanted()
    assert {w["name"] for w in json.loads((tmp_path / "src" / "wanted.json").read_text()).values()} == {"row", "modal_apps"}


def test_vm_calibration_relaunch(monkeypatch):
    import types

    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    spec = importlib.util.spec_from_file_location("mon_c", ROOT / "tools/ops/mon.py")
    mon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mon)
    cal = {"host": {"cpu_model": "Intel(R) Xeon(R) 6776P"}, "gpus": [{"uuid": "GPU-bbc6772b-3078-190f-52f6-0c887956118f"}],
           "bench": {"gpu_core_gflops": 12.5, "host_mem_copy_gbs": 80.0}, "errors": ["x"], "started_epoch": 1000.0}
    rec, saved, fetched = {"attempts": [{"n": 1, "calibration": {"old": 1}}, {"n": 2, "start_epoch": 1200}]}, [], []
    R = types.SimpleNamespace(get=lambda run: rec, out_id=lambda run: run + "-attempt2",
                              set_calibration=lambda run, data: saved.append(data))

    def cat(argv, **k):
        fetched.append(argv[2])
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(cal))

    monkeypatch.setattr(mon.subprocess, "run", cat)
    rows = [{"run": "r", "platform": "nebius"}, {"run": "m", "platform": "modal"}]
    assert mon.vm_calibration(rows, R, now=1300) == {} and len(fetched) == 2 and not saved  # the old VM's file: not this attempt's
    fetched.clear()
    assert mon.vm_calibration(rows, R, now=9000) == {} and not fetched  # past the window: no fetch every pass
    cal["started_epoch"] = 1250.0
    assert mon.vm_calibration(rows, R, now=1300)["r"] is not None and saved and len(fetched) == 1
