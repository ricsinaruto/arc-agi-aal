"""v38: the engine watchdog in the arc2 transport and the driver's model prefetch."""
import asyncio
import json
import time

import httpx

from arc2 import kaggle, trial
from arena import engine_watchdog
from arena.gate import KVGate


def transport(handler):
    client = httpx.AsyncClient(base_url="http://x/v1/", transport=httpx.MockTransport(handler))
    return trial.Transport(client, KVGate(10**9, slots=4), model="m", effort="xhigh", temperature=0.7,
                           deadline=time.monotonic() + 600)


def ok():
    return httpx.Response(200, json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                                     "usage": {"completion_tokens": 3}})


def test_5xx_during_outage_waits_for_the_restart(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="down") if len(calls) <= 4 else ok()

    async def settle(seconds):  # an outage the watchdog saw, then healthy
        engine_watchdog.STATE["events"].append({"what": "unhealthy"})
        return True

    monkeypatch.setitem(engine_watchdog.STATE, "enabled", True)
    monkeypatch.setitem(engine_watchdog.STATE, "events", [])
    monkeypatch.setattr(engine_watchdog, "settle", settle)
    reply = asyncio.run(transport(handler).complete("t", [{"role": "user", "content": "hi"}], 10, 100, {}))
    assert reply["message"]["content"] == "x" and len(calls) == 5  # 4 outages waited, no 5/15 s sleeps


def test_5xx_on_a_healthy_server_keeps_the_plain_retries(monkeypatch):
    calls, slept = [], []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, text="bad")

    async def settle(seconds):  # healthy, nothing new from the watchdog
        return True

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setitem(engine_watchdog.STATE, "enabled", True)
    monkeypatch.setitem(engine_watchdog.STATE, "events", [])
    monkeypatch.setattr(engine_watchdog, "settle", settle)
    monkeypatch.setattr(trial.asyncio, "sleep", sleep)
    try:
        asyncio.run(transport(handler).complete("t", [{"role": "user", "content": "hi"}], 10, 100, {}))
    except httpx.HTTPStatusError:
        pass
    assert len(calls) == 3 and slept == [5, 15]


def test_watchdog_off_is_the_old_path(monkeypatch):
    calls, slept = [], []

    def handler(request):
        calls.append(1)
        return httpx.Response(502) if len(calls) == 1 else ok()

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setitem(engine_watchdog.STATE, "enabled", False)
    monkeypatch.setattr(trial.asyncio, "sleep", sleep)
    reply = asyncio.run(transport(handler).complete("t", [{"role": "user", "content": "hi"}], 10, 100, {}))
    assert reply["finish_reason"] == "stop" and slept == [5]


def test_prefetch_reads_overlay_shards_in_place_of_the_base(tmp_path, monkeypatch):
    bundle, inputs = tmp_path / "bundle", tmp_path / "input"
    (bundle / "configs").mkdir(parents=True)
    (bundle / "configs/comparison.json").write_text(json.dumps({"server": {"expert_overlay": {"dataset": "o/ov"}}}))
    model, overlay = tmp_path / "model", inputs / "datasets/o/ov"
    model.mkdir()
    overlay.mkdir(parents=True)
    for name in ("a.safetensors", "b.safetensors", "config.json"):
        (model / name).write_bytes(b"x" * 1000)
    (overlay / "b.safetensors").write_bytes(b"y" * 3000)
    (overlay / "overlay-manifest.json").write_text(json.dumps({"shards": {"b.safetensors": {}}}))
    monkeypatch.setenv("ARC_MODEL_DIR", str(model))
    record = {}
    kaggle.prefetch(bundle, inputs, 4, record)
    deadline = time.time() + 10
    while "seconds" not in record["prefetch"] and time.time() < deadline:
        time.sleep(0.05)
    state = record["prefetch"]
    assert state["files"] == 2 and state["of"] == 2 and state["overlay_shards"] == 1 and "error" not in state
    assert abs(state["gib"] - 4000 / 2**30) < 1e-4


def test_32k_cap_in_the_probe_gate_table():
    from arc2 import l4_serving

    assert 32768 in l4_serving.REPLY_CAPS


SERVER = r'''
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"message": {"content": "back"}, "finish_reason": "stop"}], "usage": {}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


def test_dead_server_is_restarted_and_the_request_waits(tmp_path, monkeypatch):
    """The whole path on CPU: the server dies, the watchdog stops and starts it with the given commands, and a
    request sent during the outage is answered by the new server."""
    import socket
    import subprocess
    import sys

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    pidfile = tmp_path / "pid"
    start = [sys.executable, "-c", f"import subprocess,sys,time,urllib.request\n"
             f"p=subprocess.Popen([sys.executable,{str(script)!r},'{port}'],start_new_session=True)\n"
             f"open({str(pidfile)!r},'w').write(str(p.pid))\n"
             f"for _ in range(100):\n"
             f"    try: urllib.request.urlopen('http://127.0.0.1:{port}/health',timeout=1); break\n"
             f"    except Exception: time.sleep(0.1)\n"]
    stop = [sys.executable, "-c", f"import os,signal\ntry: os.kill(int(open({str(pidfile)!r}).read()),signal.SIGKILL)\n"
            f"except Exception: pass\n"]
    subprocess.run(start, check=True)
    monkeypatch.setattr(engine_watchdog, "POLL_SECONDS", 0.2)
    monkeypatch.setitem(engine_watchdog.STATE, "enabled", False)
    monkeypatch.setitem(engine_watchdog.STATE, "events", [])
    monkeypatch.setitem(engine_watchdog.STATE, "restarts", 0)
    monkeypatch.setenv(engine_watchdog.ENV, json.dumps({"start": start, "stop": stop, "env": {}}))
    original = httpx.AsyncClient.send
    try:
        assert engine_watchdog.start(f"http://127.0.0.1:{port}/v1", str(tmp_path))
        subprocess.run(stop, check=True)  # the engine dies

        async def ask():
            client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}/v1/")
            t = trial.Transport(client, KVGate(10**9, slots=4), model="m", effort="xhigh", temperature=0.7,
                                deadline=time.monotonic() + 60)
            return await t.complete("t", [{"role": "user", "content": "hi"}], 10, 100, {})

        reply = asyncio.run(asyncio.wait_for(ask(), 30))
        assert reply["message"]["content"] == "back"
        assert engine_watchdog.STATE["restarts"] == 1
        assert [e["what"] for e in engine_watchdog.STATE["events"]][:4] == ["armed", "unhealthy", "restart_begin",
                                                                           "restart_end"]
    finally:
        engine_watchdog.stop()
        time.sleep(0.5)
        httpx.AsyncClient.send = original
        subprocess.run(stop)
