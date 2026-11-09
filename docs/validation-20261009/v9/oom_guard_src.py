"""OOM victim order (30 Oct, owner-approved for the v9 9 h preview): under a host OOM the kernel kills a model-code
python sandbox, never the serving engine or the trial.

A daemon thread scans /proc every ``POLL_SECONDS``: sandboxes (cwd under a ``duck_python_`` tempdir) get
oom_score_adj +1000 (raising is always permitted); vLLM processes (API server and EngineCore, restarted ones too) get
-1000 and the trial process -900 (lowering needs CAP_SYS_RESOURCE: a refusal is logged once per pid, nothing else
changes). Nothing changes play unless the host runs out of memory. Events: ``<out>/oom-guard.jsonl``."""
import json
import os
import threading
import time
from pathlib import Path

POLL_SECONDS = 5.0
SANDBOX, ENGINE, TRIAL = 1000, -1000, -900
PROC = Path("/proc")
STATE = {"thread": None, "set": {}, "refused": set(), "out": None}


def _log(what, **data):
    out = STATE["out"]
    if out is None:
        return
    try:
        with open(out, "a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 1), "what": what, **data}) + "\n")
    except OSError:
        pass


def kind(pid, proc=PROC):
    """'sandbox', 'engine' or None for one /proc entry."""
    base = proc / str(pid)
    try:
        if os.readlink(base / "cwd").rsplit("/", 1)[-1].startswith("duck_python_"):
            return "sandbox"
    except OSError:
        pass
    try:
        args = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        comm = (base / "comm").read_text(errors="replace").strip()
    except OSError:
        return None
    if comm.startswith("VLLM::") or "EngineCore" in args or ("vllm" in args and (" serve" in args or "api_server" in args)):
        return "engine"
    return None


def adjust(pid, value, label, proc=PROC):
    """Write ``value`` once per pid; a refusal is logged once."""
    key = (pid, value)
    if STATE["set"].get(pid) == value or key in STATE["refused"]:
        return False
    path = proc / str(pid) / "oom_score_adj"
    try:
        path.write_text(str(value))
    except OSError as exc:
        STATE["refused"].add(key)
        _log("refused", pid=pid, kind=label, value=value, error=type(exc).__name__)
        return False
    STATE["set"][pid] = value
    _log("set", pid=pid, kind=label, value=value)
    return True


def scan(proc=PROC, self_pid=None):
    """One pass; returns the number of writes."""
    self_pid = os.getpid() if self_pid is None else self_pid
    writes = int(adjust(self_pid, TRIAL, "trial", proc))
    try:
        pids = [int(p.name) for p in proc.iterdir() if p.name.isdigit()]
    except OSError:
        return writes
    live = set(pids)
    for pid in [p for p in STATE["set"] if p not in live]:
        STATE["set"].pop(pid, None)
    for pid in pids:
        if pid == self_pid:
            continue
        k = kind(pid, proc)
        if k is not None:
            writes += int(adjust(pid, SANDBOX if k == "sandbox" else ENGINE, k, proc))
    return writes


def _run():
    while True:
        try:
            scan()
        except Exception as exc:  # never let the guard end anything
            _log("scan_error", error=repr(exc)[:300])
        time.sleep(POLL_SECONDS)


def start(out_dir):
    if STATE["thread"] is not None or not PROC.is_dir():
        return
    STATE["out"] = str(Path(out_dir) / "oom-guard.jsonl")
    STATE["thread"] = threading.Thread(target=_run, daemon=True, name="arena-oom-guard")
    STATE["thread"].start()
    _log("start", poll=POLL_SECONDS, sandbox=SANDBOX, engine=ENGINE, trial=TRIAL)
