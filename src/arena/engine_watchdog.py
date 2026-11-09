"""Engine watchdog (17 Oct, throughput arm): restart a dead vLLM server during a trial and let games wait for it.

The AV2C submission's engine died of a CUDA OOM (PLE short-conv prefill) and nothing restarted it: every game's
requests failed after three retries (22 s) and the score froze. Here a daemon thread in the trial process polls
``/health``; after ``FAILS`` failed polls it copies the server log aside, stops the owned server (process group) and
starts it again with the bootstrap's own serving command (``ARENA_ENGINE_WATCHDOG``: start / stop argv and the
trial's server env). ``HEALTHY`` is clear from the first failed poll until the restarted server answers; requests that
fail meanwhile (5xx or transport errors) wait for it instead of ending their game (``arena.duck`` with_server_retries).

``test_kill_seconds`` (study ``engine_watchdog``): kill the EngineCore that many seconds after play starts (test only).
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ENV = "ARENA_ENGINE_WATCHDOG"
HEALTHY = threading.Event()
HEALTHY.set()
FAILS = 3
POLL_SECONDS = 15.0
MAX_RESTARTS = 4
# How long a request waits out an outage (arc2 on four L4s, 19 Oct: a start takes 600-2,000 s there).
OUTAGE_WAIT = float(os.environ.get("ARENA_ENGINE_WATCHDOG_WAIT", "1800"))
STATE = {"enabled": False, "restarts": 0, "outage_seconds": 0.0, "events": []}


def enabled():
    return STATE["enabled"]


def _log(event, **fields):
    record = {"event": "engine_watchdog", "what": event, "epoch": round(time.time(), 1), **fields}
    STATE["events"].append(record)
    print(json.dumps(record), flush=True)
    sys.stdout.flush()


def _healthy(root, timeout=10.0):
    try:
        with urllib.request.urlopen(root + "/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - any failure is an unhealthy poll
        return False


def _restart(cfg, server_dir):
    env = {**os.environ, **{str(k): str(v) for k, v in (cfg.get("env") or {}).items()}}
    log = Path(server_dir) / "vllm-openai-server.log"
    if log.is_file():
        shutil.copyfile(log, log.with_name(f"vllm-openai-server-before-restart-{STATE['restarts']}.log"))
    # Output to files, not pipes: a server the start leaves running keeps a pipe open and run() would wait on it.
    number = STATE["restarts"]
    with open(Path(server_dir) / f"watchdog-stop-{number}.log", "w") as out:
        stop = subprocess.run(cfg["stop"], env=env, stdout=out, stderr=subprocess.STDOUT, timeout=300)
    path = Path(server_dir) / f"watchdog-start-{number}.log"
    with open(path, "w") as out:
        start = subprocess.run(cfg["start"], env=env, stdout=out, stderr=subprocess.STDOUT,
                               timeout=float(cfg.get("start_timeout", 2700)))
    return stop.returncode, start.returncode, path.read_text(errors="replace")[-1500:]


def _run(root, cfg, server_dir, kill_at):
    started = time.monotonic()
    fails, killed, outage_start = 0, False, None
    while STATE["enabled"]:  # ``stop`` ends the thread at its next poll
        if kill_at and not killed and time.monotonic() - started >= kill_at:
            killed = True
            done = subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True, text=True)
            _log("test_kill", returncode=done.returncode, play_seconds=round(time.monotonic() - started, 1))
        if _healthy(root):
            if outage_start is not None:
                STATE["outage_seconds"] += time.monotonic() - outage_start
                _log("healthy", outage_seconds=round(time.monotonic() - outage_start, 1), restarts=STATE["restarts"])
                outage_start = None
            fails = 0
            HEALTHY.set()
        else:
            fails += 1
            HEALTHY.clear()
            if outage_start is None:
                outage_start = time.monotonic()
                _log("unhealthy", play_seconds=round(time.monotonic() - started, 1))
            if fails >= FAILS:
                if STATE["restarts"] >= MAX_RESTARTS:
                    _log("gave_up", restarts=STATE["restarts"])
                    HEALTHY.set()  # games fall back to their ordinary retries
                    return
                STATE["restarts"] += 1
                t0 = time.monotonic()
                _log("restart_begin", restart=STATE["restarts"])
                try:
                    stop_rc, start_rc, tail = _restart(cfg, server_dir)
                    _log("restart_end", restart=STATE["restarts"], seconds=round(time.monotonic() - t0, 1),
                         stop_rc=stop_rc, start_rc=start_rc, stderr_tail=tail if start_rc else "")
                except Exception as exc:  # noqa: BLE001 - logged; the next polls decide
                    _log("restart_error", restart=STATE["restarts"], error=repr(exc)[:500],
                         seconds=round(time.monotonic() - t0, 1))
                fails = 0
                continue
        time.sleep(POLL_SECONDS)


def _patch_httpx(root):
    """Every request of this process to the server (chat, tokenize, metrics, cache pings) waits out an outage on a
    transport error instead of failing its game (gi3wd: one game ended on a ConnectError outside the chat retry).
    Timeouts are never retried; at most 5 attempts unless a restart happens in between."""
    import httpx

    original = httpx.AsyncClient.send
    if getattr(original, "_arc_watchdog", False):
        return

    async def send(self, request, *args, **kwargs):
        attempt, restarts = 0, STATE["restarts"]
        while True:
            try:
                return await original(self, request, *args, **kwargs)
            except httpx.TimeoutException:
                raise
            except httpx.TransportError:
                if not str(request.url).startswith(root):
                    raise
                attempt += 1
                if STATE["restarts"] != restarts:
                    attempt, restarts = 1, STATE["restarts"]
                if attempt > 5 or not await settle(OUTAGE_WAIT):
                    raise

    send._arc_watchdog = True
    httpx.AsyncClient.send = send


def start(base_url, server_dir):
    """Start the daemon thread when the bootstrap armed it (``ARENA_ENGINE_WATCHDOG``); returns whether it runs."""
    raw = os.environ.get(ENV)
    if not raw or STATE["enabled"]:
        return STATE["enabled"]
    cfg = json.loads(raw)
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    kill_at = float(cfg.get("test_kill_seconds") or 0)
    STATE["enabled"] = True
    _patch_httpx(root)
    threading.Thread(target=_run, args=(root, cfg, server_dir, kill_at), daemon=True,
                     name="engine-watchdog").start()
    _log("armed", root=root, kill_at=kill_at or None)
    return True


def stop():
    """End the polling thread at its next poll (tests; a run's thread lives as long as its process)."""
    STATE["enabled"] = False
    HEALTHY.set()


async def settle(seconds):
    """After a failed request: give the watchdog one poll to notice (it polls every ``POLL_SECONDS``), then wait out
    any outage, at most ``seconds`` in all. True when the server is healthy."""
    import asyncio

    pause = min(max(0.0, seconds), POLL_SECONDS + 5.0)
    await asyncio.sleep(pause)
    return await wait_healthy(seconds - pause)


async def wait_healthy(seconds):
    """Wait (polling) until the server is healthy again or ``seconds`` pass; True when healthy."""
    import asyncio

    deadline = time.monotonic() + max(0.0, seconds)
    while not HEALTHY.is_set():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(2.0)
    return True
