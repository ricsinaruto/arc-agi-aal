"""v9robust.py (v9 9 h preview): A. oom_guard victim order on a fake /proc and (Linux) on real
processes; B. game re-entry fault injection on a patched copy of the v9 bundle's src (v9robust_cases.py), with the
unpatched control. The bundle comes from V9_SRC (default dist/b2-v9-p75c268-kf-ho10x11-9h-v1-sub)."""

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "docs/validation-20261009/v9"
SRC = Path(os.environ.get("V9_SRC", ROOT / "dist/b2-v9-p75c268-kf-ho10x11-9h-v1-sub"))


def guard():
    spec = importlib.util.spec_from_file_location(f"og_{uuid.uuid4().hex}", V9 / "oom_guard_src.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_proc(root, pid, *, cwd=None, args=b"python\0x.py\0", comm="python"):
    d = root / str(pid)
    d.mkdir(parents=True)
    target = root.parent / (cwd or f"work{pid}")
    target.mkdir(exist_ok=True)
    os.symlink(target, d / "cwd")
    (d / "cmdline").write_bytes(args)
    (d / "comm").write_text(comm + "\n")
    (d / "oom_score_adj").write_text("0")
    return d


def test_guard_victim_order_on_a_fake_proc(tmp_path):
    og = guard()
    proc = tmp_path / "proc"
    og.STATE["out"] = str(tmp_path / "oom-guard.jsonl")
    trial = fake_proc(proc, 10)
    sandbox = fake_proc(proc, 11, cwd="duck_python_abc")
    api = fake_proc(proc, 12, args=b"/usr/bin/python\0-m\0vllm.entrypoints.openai.api_server\0--port\0001234\0")
    core = fake_proc(proc, 13, comm="VLLM::EngineCor")
    other = fake_proc(proc, 14, args=b"python\0-m\0arena.run\0")
    assert og.scan(proc, self_pid=10) == 4
    got = {p.name: (p / "oom_score_adj").read_text() for p in (trial, sandbox, api, core, other)}
    assert got == {"10": "-900", "11": "1000", "12": "-1000", "13": "-1000", "14": "0"}
    assert og.scan(proc, self_pid=10) == 0  # written once per pid
    shutil.rmtree(sandbox)
    fake_proc(proc, 15, cwd="duck_python_def")  # a restarted interpreter
    assert og.scan(proc, self_pid=10) == 1 and "11" not in map(str, og.STATE["set"])


def test_guard_logs_a_refusal_once(tmp_path):
    og = guard()
    proc = tmp_path / "proc"
    og.STATE["out"] = str(tmp_path / "oom-guard.jsonl")
    d = fake_proc(proc, 20, comm="VLLM::EngineCor")
    (d / "oom_score_adj").unlink()
    (d / "oom_score_adj").mkdir()  # writing fails like EACCES
    og.scan(proc, self_pid=99999)
    og.scan(proc, self_pid=99999)
    lines = (tmp_path / "oom-guard.jsonl").read_text().splitlines()
    assert sum('"refused"' in line and '"pid": 20' in line for line in lines) == 1


@pytest.mark.skipif(not Path("/proc/self/oom_score_adj").exists(), reason="Linux /proc only")
def test_guard_on_real_processes(tmp_path):
    og = guard()
    og.STATE["out"] = str(tmp_path / "oom-guard.jsonl")
    box = tmp_path / "duck_python_x1"
    box.mkdir()
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    sandbox = subprocess.Popen(sleeper, cwd=box)
    engine = subprocess.Popen([*sleeper, "vllm", "serve"])
    try:
        time.sleep(0.3)
        og.scan()
        assert Path(f"/proc/{sandbox.pid}/oom_score_adj").read_text().strip() == "1000"
        eng = Path(f"/proc/{engine.pid}/oom_score_adj").read_text().strip()
        refused = '"refused"' in (tmp_path / "oom-guard.jsonl").read_text()
        assert eng == "-1000" or (eng == "0" and refused)
    finally:
        sandbox.kill()
        engine.kill()


def patched_src(tmp_path, robust):
    if not (SRC / "src/duck_transfer/agent.py").is_file():
        pytest.skip("no v9 bundle tree")
    root = tmp_path / ("robust" if robust else "control")
    shutil.copytree(SRC / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    if robust:
        subprocess.run([sys.executable, str(V9 / "v9robust.py"), str(root)], check=True, capture_output=True)
    return root / "src"


@pytest.mark.parametrize("robust", [True, False])
def test_reentry_fault_injection(tmp_path, robust):
    src = patched_src(tmp_path, robust)
    env = {**os.environ, "PYTHONPATH": f"{src}{os.pathsep}{ROOT / 'tests'}", "ROBUST": str(int(robust))}
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests/v9robust_cases.py")],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]
    assert " passed" in run.stdout


def test_patch_shape(tmp_path):
    src = patched_src(tmp_path, True)
    agent = (src / "duck_transfer/agent.py").read_text()
    assert agent.count("_arena_reenter") == 2 and "reentries += 1" in agent
    trial = (src / "arena/trial.py").read_text()
    assert "_oom_guard.start(self.output)" in trial and (src / "arena/oom_guard.py").is_file()
