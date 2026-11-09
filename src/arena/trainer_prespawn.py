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
ALLOWED = {"ARC_COMPILE_KEEP",  # vLLM-only: the first server start keeps the restored compile graph (modal_rl1.compile_restore)
           "RL1_PRELOAD_RESERVE_GIB"}  # read once by the trainer at its start (its preload guard): set only here
GATE = "trainer-gpu-gate"  # rl_seq's defer_trainer gate file, in the loop's work dir
SERVER_BOOT_GIB = 92.4  # host GiB per play server while it boots (modal_rl1.SERVER_HOST_GIB: ple + routed + load transient)


def preload_prespawn(study, environ=None):
    """Stage H defer_trainer with preload_host: the preloading trainer starts at container start,
    so the servers' boot counts as host-load time. A resume too (real-v9 28 Oct: its trainer loaded only after both
    boots, +330 s): rl_seq opens the gate once the servers sleep (rl_seq.prespawned_gate). A drafting loop loads the
    trainer first, without the gate."""
    mg = study["rl_loop"].get("mg") or {}
    red = mg.get("redesign")
    defer = red.get("defer_trainer") if isinstance(red, dict) else None
    return (isinstance(defer, dict) and bool(defer.get("preload_host")) and bool(defer.get("template"))
            and not float(mg.get("draft_weight") or 0) > 0 and not mg.get("lr_probe"))


def wanted(study):
    return preload_prespawn(study)


def command_for(cfg, source, work=WORK, gate=None):
    """The loop's trainer command and environment (rl_loop.start_trainer's)."""
    from .rl_loop import trainer_command

    command, env = trainer_command(cfg, source, work)
    if gate is not None:
        env["RL1_GPU_GATE"] = str(gate)
        redesign = ((cfg.get("mg") or {}).get("redesign")) or {}
        defer = redesign.get("defer_trainer") if isinstance(redesign, dict) else None
        if isinstance(defer, dict) and defer.get("preload_host"):  # the model built in host RAM before the gate
            env["RL1_PRELOAD_HOST"] = "1"  # (flash_qlora_job.load); the gated part is the move to the card
            if defer.get("preload_need_gib"):  # host GiB per trainer process the guard asks for (default 400)
                env["RL1_PRELOAD_NEED_GIB"] = str(float(defer["preload_need_gib"]))
    return command, env


def spawn(study_path, source, log, work=WORK):
    from .rl_loop import settings

    study = json.loads(Path(study_path).read_text())
    if not wanted(study):
        return None
    gate = None
    if preload_prespawn(study):  # the gate path rl_seq opens at the first switch; closed until then
        gate = Path(work) / GATE
        gate.unlink(missing_ok=True)
    # the merge only: the launcher's python lacks the play stack the redesign validators import (preload pf N1:
    # arc_agi missing; the loop validates the same config itself)
    cfg = settings(study["rl_loop"], check=False)
    red = (cfg.get("mg") or {}).get("redesign")
    if isinstance(red, dict):  # rl_seq's own default before it starts the trainer: one rank per card (torchrun)
        cfg["mg"].setdefault("trainer_cards", int(red.get("cards") or 1))
    command, env = command_for(cfg, source, work, gate)
    if gate is not None and env.get("RL1_PRELOAD_HOST") == "1":  # the servers boot beside this load: their host need
        cards = int((((study["rl_loop"].get("mg") or {}).get("redesign")) or {}).get("cards") or 1)
        env["RL1_PRELOAD_RESERVE_GIB"] = str(round(cards * SERVER_BOOT_GIB, 1))
    folder = Path(work) / "trainer"
    (folder / "cmd").mkdir(parents=True, exist_ok=True)
    from arc_infra.heap_census import install_for

    install_for(command, env)  # the trainer ranks' census when trainer_env asks for it
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
