"""wd_nogiveup.py: the patched engine watchdog keeps restarting past MAX_RESTARTS (HEALTHY stays clear, so games
wait), and recovers when the server comes back. Runs on a copy of a bundle's engine_watchdog.py (WD_SRC env, else
the v9-sub tree)."""

import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("WD_SRC", ROOT / "dist/arena-flash-competition-v9-sub"))


def patched(tmp_path):
    if not (SRC / "src/arena/engine_watchdog.py").is_file():
        pytest.skip("no bundle watchdog to patch")
    (tmp_path / "src/arena").mkdir(parents=True)
    shutil.copyfile(SRC / "src/arena/engine_watchdog.py", tmp_path / "src/arena/engine_watchdog.py")
    subprocess.run([sys.executable, str(ROOT / "docs/validation-20261009/v9/wd_nogiveup.py"), str(tmp_path)],
                   check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location(f"wd_{uuid.uuid4().hex}", tmp_path / "src/arena/engine_watchdog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_give_up_then_recovery(tmp_path):
    wd = patched(tmp_path)
    up = {"v": False}
    wd._healthy = lambda root, timeout=5: up["v"]
    restarts = []

    def fake_restart(cfg, server_dir):
        restarts.append(time.monotonic())
        return 0, 1, "start failed"  # every restart fails

    wd._restart = fake_restart
    wd.POLL_SECONDS, wd.FAILS, wd.MAX_RESTARTS = 0.01, 2, 2
    wd.time = types.SimpleNamespace(monotonic=time.monotonic, time=time.time, sleep=lambda s: time.sleep(min(s, 0.01)))
    t = threading.Thread(target=wd._run, args=("http://x", {}, str(tmp_path), 0), daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while len(restarts) < 6 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(restarts) >= 6, "watchdog stopped restarting"  # past MAX_RESTARTS = 2
    assert t.is_alive() and not wd.HEALTHY.is_set()  # games keep waiting
    events = [e["what"] for e in wd.STATE["events"]]
    assert "gave_up" not in events and "restart_backoff" in events
    up["v"] = True
    deadline = time.monotonic() + 5
    while not wd.HEALTHY.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert wd.HEALTHY.is_set() and "healthy" in [e["what"] for e in wd.STATE["events"]]


def test_patch_text(tmp_path):
    wd_text = None
    patched(tmp_path)
    wd_text = (tmp_path / "src/arena/engine_watchdog.py").read_text()
    assert '_log("gave_up"' not in wd_text and "settle(1800.0)" not in wd_text
    assert "(attempt > 5 and HEALTHY.is_set())" in wd_text
