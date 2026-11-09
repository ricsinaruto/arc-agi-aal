"""arena.server_recycle: a play server's process session is stopped and started again from its
own argv / environment; server 1's identity file names the new root."""
import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arena import server_recycle

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc (Linux)")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_recycle_replaces_the_whole_session_and_the_identity(tmp_path):
    (tmp_path / "www" / "v1").mkdir(parents=True)
    (tmp_path / "www" / "v1" / "models").write_text("{}")
    port = free_port()
    # a root with a child in its session (vLLM: API server -> EngineCore -> workers)
    code = ("import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(600)']);"
            f"import http.server,functools;h=functools.partial(http.server.SimpleHTTPRequestHandler,directory={str(tmp_path / 'www')!r});"
            f"http.server.ThreadingHTTPServer(('127.0.0.1',{port}),h).serve_forever()")
    first = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, env={"RECYCLE_MARK": "1", "PATH": "/usr/bin:/bin"})
    ident = tmp_path / "vllm-server-identity.json"
    ident.write_text(json.dumps({"pid": first.pid, "pgid": first.pid, "sid": first.pid, "workers": []}))

    class S:
        root = f"http://127.0.0.1:{port}"
        recycle_log = tmp_path / "server.log"
        recycle_identity = ident

        def recycle_pid(self):
            return int(json.loads(ident.read_text())["pid"])

    deadline = time.time() + 20
    while time.time() < deadline and len(server_recycle.session(first.pid)) < 2:
        time.sleep(0.1)
    old = server_recycle.session(first.pid)
    assert len(old) == 2
    s = S()
    s.process = first
    row = asyncio.run(server_recycle.recycle(s, timeout=30))
    try:
        assert row["old_pid"] == first.pid and row["pid"] != first.pid and row["stopped"]["pids"] == 2
        assert not server_recycle.session(first.pid)  # every old process is gone (none left as a live zombie)
        assert json.loads(ident.read_text())["pid"] == row["pid"] and json.loads(ident.read_text())["recycled"] == 1
        env = Path(f"/proc/{row['pid']}/environ").read_bytes().decode()
        assert "RECYCLE_MARK=1" in env  # same environment
        assert row["boot_s"] is not None
    finally:
        server_recycle.stop(server_recycle.stat(row["pid"])[3], reap=s.process.poll)
