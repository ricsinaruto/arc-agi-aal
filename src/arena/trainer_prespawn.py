"""Trainer pre-spawn (RL start-up cut (b), 22 Oct): the launcher starts the loop's trainer process at container start,
beside server 1's boot, instead of at the loop's first line (~8 min in: r4 spawned it 17:41, loaded 17:44). The loop's
``start_trainer`` adopts it when its own command and environment match (``take``), else stops it (verified) and spawns
afresh. Only the lazy time share (trainer alone on its card at start) pre-spawns.

``python -m arena.trainer_prespawn STUDY SOURCE LOG`` (run by modal_rl1.execute) spawns and returns at once.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/tmp/rl1")
RECORD = "prespawn.json"
# environment keys that may differ between the launcher's spawn and the loop's (set later for the arena/vLLM only);
# filled from the CPU leg's measured diff, each with its reason
ALLOWED = {"ARC_COMPILE_KEEP"}  # vLLM-only: the first server start keeps the restored compile graph (modal_rl1.compile_restore)


def wanted(study):
    mg = study["rl_loop"].get("mg") or {}
    return (bool(study["rl_loop"].get("dual")) and bool((mg.get("timeshare") or {}).get("lazy_boot"))
            and not mg.get("single_card") and not mg.get("bench") and not mg.get("lr_probe") and not mg.get("eval")
            and study["rl_loop"].get("mode") not in ("screen", "final"))


def command_for(cfg, source, work=WORK, gate=None):
    """The loop's trainer command and environment (rl_loop.start_trainer's)."""
    from .rl_loop import trainer_command

    command, env = trainer_command(cfg, source, work)
    if cfg["dual"]:
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(cfg["trainer_device"]))
    if gate is not None:
        env["RL1_GPU_GATE"] = str(gate)
    return command, env


def spawn(study_path, source, log, work=WORK):
    from .rl_loop import settings

    study = json.loads(Path(study_path).read_text())
    if not wanted(study):
        return None
    command, env = command_for(settings(study["rl_loop"]), source, work)
    folder = Path(work) / "trainer"
    (folder / "cmd").mkdir(parents=True, exist_ok=True)
    with open(log, "a") as out:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    (folder / RECORD).write_text(json.dumps({"pid": process.pid, "epoch": time.time(), "command": command,
                                             "env": env, "log": str(log)}))
    return process.pid


def alive(pid):
    try:  # our own child (tests; a launcher that never waits): reap it, so its exit reads as gone
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    stat = Path(f"/proc/{pid}/stat")
    if Path("/proc").is_dir():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class Adopted:
    """A pre-spawned process seen through the Popen calls rl_loop.Trainer makes (poll, returncode, terminate, wait,
    kill). Not our child: an exit reads as returncode -1."""

    def __init__(self, pid):
        self.pid, self.returncode = int(pid), None

    def poll(self):
        if self.returncode is None and not alive(self.pid):
            self.returncode = -1
        return self.returncode

    def _signal(self, sig):
        try:
            os.killpg(self.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def terminate(self):
        self._signal(signal.SIGTERM)

    def kill(self):
        self._signal(signal.SIGKILL)

    def wait(self, timeout=None):
        end = None if timeout is None else time.time() + timeout
        while self.poll() is None:
            if end is not None and time.time() > end:
                raise subprocess.TimeoutExpired(f"pid {self.pid}", timeout)
            time.sleep(0.2)
        return self.returncode


def take(folder, command, env, log=None):
    """(pid to adopt or None, feed row or None) for the trainer dir ``folder``. A mismatched or dead pre-spawn is
    stopped (verified gone) and its files removed, so a fresh trainer starts clean (a stale alive.json would read as
    ready)."""
    folder = Path(folder)
    try:
        pre = json.loads((folder / RECORD).read_text())
    except (OSError, ValueError):
        return None, None
    (folder / RECORD).unlink(missing_ok=True)
    keys = sorted(k for k in set(pre["env"]) | set(env) if pre["env"].get(k) != env.get(k))
    row = {"pid": pre["pid"], "age_seconds": round(time.time() - pre["epoch"], 1), "env_diff": keys,
           "command_equal": pre["command"] == command,
           "command_diff": [[a, b] for a, b in zip(pre["command"], command) if a != b][:6]
           + ([len(pre["command"]), len(command)] if len(pre["command"]) != len(command) else []), "log_equal": log is None or pre.get("log") == str(log)}
    why = None
    if not alive(pre["pid"]):
        why = "pre-spawned trainer not alive"
    elif pre["command"] != command:
        why = "command differs"
    elif set(keys) - ALLOWED:
        why = "environment differs"
    if why is None:
        return pre["pid"], {**row, "adopted": True, "log": pre.get("log")}
    proc = Adopted(pre["pid"])
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(30)
    for name in ("alive.json",):
        (folder / name).unlink(missing_ok=True)
    for sub in ("cmd", "done"):
        for f in (folder / sub).glob("*") if (folder / sub).is_dir() else []:
            f.unlink(missing_ok=True)
    return None, {**row, "adopted": False, "why": why}


if __name__ == "__main__":
    print(json.dumps({"event": "trainer_prespawn", "pid": spawn(sys.argv[1], sys.argv[2], sys.argv[3])}), flush=True)
