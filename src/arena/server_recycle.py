"""Play-server recycle. vLLM's host memory grows ~15 GiB per M generated tokens per server
(heap legs A/B 27 Oct: EngineCore, API server and workers; live objects that malloc_trim does not return and sleep
does not free). A recycle stops a server's whole process session and starts the same argv and environment again
(same card, port and caches) while the trainer is parked off the card (rl_seq: PhaseController.to_play), so the
next phase plays on fresh processes."""
import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def snapshot(pid):
    """(argv, env, cwd) of a live process."""
    proc = Path(f"/proc/{pid}")
    argv = [a for a in (proc / "cmdline").read_bytes().decode(errors="replace").split("\0") if a]
    env = dict(x.split("=", 1) for x in (proc / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in x)
    return argv, env, os.readlink(proc / "cwd")


def stat(pid):
    """(state, ppid, pgid, sid) from /proc/<pid>/stat, or None."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return fields[0], int(fields[1]), int(fields[2]), int(fields[3])


def session(sid):
    """Live (not zombie) pids of a process session."""
    out = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            st = stat(int(name))
            if st is not None and st[3] == sid and st[0] != "Z":
                out.append(int(name))
    return out


def rss_gib(pids):
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except OSError:
            continue
    return round(total / 2**20, 2)


def stop(sid, term_s=30.0, kill_s=30.0, reap=None):
    """SIGTERM then SIGKILL every process of the session until none is live; ``reap``: a callable that collects our
    own child's exit (its zombie would otherwise stay). {"pids", "rss_gib", "seconds", "killed"}."""
    began, pids = time.time(), session(sid)
    out = {"pids": len(pids), "rss_gib": rss_gib(pids), "killed": False}
    for sig, wait in ((signal.SIGTERM, term_s), (signal.SIGKILL, kill_s)):
        for pid in session(sid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        out["killed"] = sig == signal.SIGKILL
        deadline = time.time() + wait
        while time.time() < deadline:
            if reap is not None:
                reap()
            if not session(sid):
                out["seconds"] = round(time.time() - began, 2)
                return out
            time.sleep(0.2)
    raise RuntimeError(f"recycle: session {sid} still has {session(sid)} after SIGKILL")


async def ready(root, process, log, timeout):
    """Until GET <root>/v1/models answers 200; raise if the process exits or ``timeout`` passes."""
    import httpx

    began = time.time()
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        while time.time() - began < timeout:
            if process.poll() is not None:
                tail = Path(log).read_text(errors="replace")[-3000:] if Path(log).is_file() else ""
                raise RuntimeError(f"recycled server exited {process.returncode}: {tail}")
            try:
                if (await client.get(root + "/v1/models")).status_code == 200:
                    return round(time.time() - began, 1)
            except httpx.HTTPError:
                pass
            from arc_infra.fleet import ACTIVITY  # a booting server is progress for the stall watchdog

            ACTIVITY["last_reply"] = time.time()
            await asyncio.sleep(2)
    raise RuntimeError(f"recycled server {root} not ready in {timeout} s")


def write_identity(path, process):
    """Server 1's vllm-server-identity.json now names the new root (the run's final stop_owned_server and every reader
    of its pid find the live server)."""
    path = Path(path)
    identity = json.loads(path.read_text())
    st = stat(process.pid)
    ticks = int(Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    root = {"pid": process.pid, "start_ticks": ticks, "ppid": st[1], "pgid": st[2], "sid": st[3]}
    try:
        import hashlib

        root["cmdline_sha256"] = hashlib.sha256(Path(f"/proc/{process.pid}/cmdline").read_bytes()).hexdigest()
    except OSError:
        pass
    now = time.time()
    identity.update(pid=process.pid, start_ticks=ticks, pgid=st[2], sid=st[3], workers=[root], started_epoch=now,
                    snapshot_epoch=now, ready_epoch=now, phase="ready", recycled=int(identity.get("recycled") or 0) + 1)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


async def recycle(server, timeout=1800):
    """Stop ``server``'s process session and start it again from its own argv / environment. ``server`` gives
    ``recycle_pid()`` (its root), ``recycle_log`` (a path), optionally ``recycle_identity`` (server 1's identity file)
    and ``process`` (a Popen-like handle to reap). Returns the row for the feed."""
    pid = server.recycle_pid()
    argv, env, cwd = snapshot(pid)
    st = stat(pid)
    old = getattr(server, "process", None)
    reap = (lambda: old.poll()) if old is not None and hasattr(old, "poll") else None
    stopped = await asyncio.to_thread(stop, st[3], reap=reap)
    began = time.time()
    with open(server.recycle_log, "a") as handle:
        handle.write(f"\n==== recycle {time.strftime('%H:%M:%S', time.gmtime())} UTC ====\n")
        handle.flush()
        process = subprocess.Popen(argv, env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    server.process = process
    waited = await ready(server.root, process, server.recycle_log, timeout)
    if getattr(server, "recycle_identity", None):
        write_identity(server.recycle_identity, process)
    return {"old_pid": pid, "pid": process.pid, "stopped": stopped, "boot_s": waited,
            "seconds": round(time.time() - began + stopped.get("seconds", 0.0), 1)}
