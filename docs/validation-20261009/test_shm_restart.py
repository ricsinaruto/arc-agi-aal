"""--l1-shm-fix check (25 Oct): an engine-watchdog restart leaves one /dev/shm offload region, not one per start.

    ARC_SHM_FIXED=<built bundle with --l1-shm-fix> ARC_SHM_UNFIXED=<built --l1 bundle without it> pytest -q THIS  (Linux)

The bundle's own arena.engine_watchdog restarts a fake server that maps the region exactly as vLLM's
SharedOffloadRegion does (O_EXCL creator ftruncates and populates; a start that finds the file joins it; nothing
unlinks on a crash). The engine id comes from the bundle's serving.py kv-transfer-config (a new uuid4 per start when
it sets none, as vLLM's default); the stop runs the bundle's own stop_program with only the owned-server stop
replaced by a SIGKILL of the fake server's process group. The unfixed bundle is the control: it must leak.
"""
import ast
import importlib.util
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

REGION = 8 << 20
FAKE = r'''
import http.server, os, sys
engine_id, size, shm, port, pidfile = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
path = os.path.join(shm, f"vllm_offload_{engine_id}.mmap")
try:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    os.pwrite(fd, b"x" * size, 0)  # MADV_POPULATE_WRITE: the region is resident from the start
except FileExistsError:
    fd = os.open(path, os.O_RDWR)
    assert os.fstat(fd).st_size >= size, "joined a short region"
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def log_message(self, *a):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
open(pidfile, "w").write(str(os.getpid()))
server.serve_forever()
'''
LAUNCH = r'''
import subprocess, sys, time, urllib.request
port = sys.argv[4]
subprocess.Popen([sys.executable, "-c", sys.argv[6], *sys.argv[1:6]], start_new_session=True,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(200):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); sys.exit(0)
    except Exception:
        time.sleep(0.05)
sys.exit(1)
'''


def bundle(var):
    path = os.environ.get(var)
    if not path:
        pytest.skip(f"{var} not set")
    return Path(path)


def engine_id_of(root):
    found = re.findall(r"'engine_id': '([\w.-]+)'", (root / "src/arena/serving.py").read_text())
    assert len(found) <= 1
    return found[0] if found else None


def stop_program_of(root, *, unlink=True):
    text = (root / "bootstrap_arena.py").read_text()
    match = re.search(r"stop_program = (\((?:\s*\"[^\n]*\"\s*)+\))", text)
    program = ast.literal_eval(match.group(1))
    owned = "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()"
    assert owned in program
    program = program.replace("from arc_infra.serving import load_recipe; ", "")
    program = "import os, time; " + program.replace(
        owned, "(os.killpg(int(open(sys.argv[1]).read()), 9), time.sleep(0.3))")
    if not unlink:  # a stop whose unlink failed: the fixed engine id alone must hold one region
        program = re.sub(r"; \[os\.unlink\(p\).*$", "", program)
    return program


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def regions(shm):
    files = sorted(shm.glob("vllm_offload_*.mmap"))
    return len(files), sum(f.stat().st_blocks * 512 for f in files)


def run_restarts(root, tmp_path, n, *, unlink=True):
    shm, server_dir = tmp_path / "shm", tmp_path / "serving"
    shm.mkdir()
    server_dir.mkdir()
    pidfile, port = tmp_path / "server.pid", free_port()
    fixed = engine_id_of(root)
    starter = tmp_path / "start.py"
    # the start command picks the engine id per start: the bundle's fixed id, or a fresh uuid4 (vLLM's default)
    starter.write_text(f"import sys, uuid, subprocess\nE = {fixed!r} or str(uuid.uuid4())\n"
                       f"sys.exit(subprocess.run([sys.executable, '-c', {LAUNCH!r}, E, '{REGION}', {str(shm)!r}, "
                       f"'{port}', {str(pidfile)!r}, {FAKE!r}]).returncode)\n")
    cfg = {"start": [sys.executable, str(starter)],
           "stop": [sys.executable, "-c", stop_program_of(root, unlink=unlink), str(pidfile), "unused"], "env": {"ARENA_SHM_DIR": str(shm)}}
    spec = importlib.util.spec_from_file_location(f"wd_{uuid.uuid4().hex}", root / "src/arena/engine_watchdog.py")
    wd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wd)
    wd.POLL_SECONDS = 0.2
    assert subprocess.run(cfg["start"]).returncode == 0
    assert regions(shm)[0] == 1
    threading.Thread(target=wd._run, args=(f"http://127.0.0.1:{port}", cfg, server_dir, None), daemon=True).start()
    seen = []
    for k in range(1, n + 1):
        os.killpg(int(pidfile.read_text()), signal.SIGKILL)  # the engine dies: nothing unlinks its region
        deadline = time.time() + 60
        while time.time() < deadline and not (wd.STATE["restarts"] == k and wd.HEALTHY.is_set()):
            time.sleep(0.1)
        assert wd.STATE["restarts"] == k and wd.HEALTHY.is_set(), wd.STATE["events"][-3:]
        seen.append(regions(shm))
    wd.MAX_RESTARTS = wd.STATE["restarts"]  # this watchdog gives up at the next failure instead of restarting
    os.killpg(int(pidfile.read_text()), signal.SIGKILL)
    return seen


def test_fixed_bundle_keeps_one_region(tmp_path):
    root = bundle("ARC_SHM_FIXED")
    assert engine_id_of(root) == "arc-arena-offload"
    seen = run_restarts(root, tmp_path, 3)
    assert seen == [(1, REGION)] * 3, seen


def test_fixed_engine_id_alone_reuses_the_region(tmp_path):
    seen = run_restarts(bundle("ARC_SHM_FIXED"), tmp_path, 2, unlink=False)
    assert seen == [(1, REGION)] * 2, seen


def test_unfixed_bundle_leaks_one_region_per_restart(tmp_path):  # control: the test sees the v85l1b leak
    root = bundle("ARC_SHM_UNFIXED")
    assert engine_id_of(root) is None
    seen = run_restarts(root, tmp_path, 2)
    assert seen == [(2, 2 * REGION), (3, 3 * REGION)], seen
