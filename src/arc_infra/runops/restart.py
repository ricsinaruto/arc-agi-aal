"""Restart guard: every restart of a run is an incident.

Modal re-runs a GPU function's input after a preemption or a lost worker (host OOM, thrash) whatever ``retries`` says,
and GPU functions cannot be non-preemptible. So each training job calls the guard at start-up. It keeps one state
file per run on the volume, ``<state_root>/.runops/<run_id>/attempts.json`` (attempts and incidents), outside the
run's own folder so the arms' "-lost-" / "preempted-" renames leave it alone.

At each start the guard records the attempt. When the last attempt has no clean end, this start is a restart: it
writes an incident with the cause the evidence supports. The evidence is the last attempt's preemption signal (SIGINT
or SIGTERM caught) and its host-memory readings (heartbeat every ``beat_s``). Modal's own label is not used; it called
c7b's host-RAM thrash "preemption". Then:

* Modal auto-restart (same function call id): proceeds only when the run opted in (``policy="resume"``), the cause is
  a preemption signal, a checkpoint past step 0 exists, and fewer than ``max_resumes`` resumes have run. Otherwise it
  raises ``RestartBlocked``. With ``retries=0`` the input then fails, so Modal does not restart it again.
* A relaunch by an agent (new call id) after a death or a blocked restart proceeds only with
  ``ack="<the cause, as read>"`` (env ``ARC_RESTART_ACK``), which is recorded on every open incident.

Use::

    guard = RestartGuard("/runs", run_id, policy="resume", commit=runs.commit)
    with guard.run(checkpoint_step=newest_ckpt_step) as start:   # raises RestartBlocked
        ...  # start.restart / start.resume / start.incident
"""
import contextlib
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import hostmem
from ._io import read_json, write_json

MIN_ACK_CHARS = 12  # an ack must say something: the cause as read, not "ok"


class RestartBlocked(RuntimeError):
    def __init__(self, incident):
        self.incident = incident
        super().__init__(f"RUNOPS restart blocked: {incident['id']} cause={incident['cause']}; {incident['why']}. "
                         "Read the cause, then relaunch with ARC_RESTART_ACK='<cause>'")


@dataclass
class Start:
    attempt: int
    restart: bool  # an earlier attempt of this run died
    resume: bool  # proceed from the checkpoint (a restart with a checkpoint past step 0)
    incident: dict | None


def _call_id():
    try:
        import modal

        return modal.current_function_call_id() or None
    except Exception:  # noqa: BLE001 - outside a Modal container
        return None


def classify(attempt, rerun=False):
    """The death cause the dead attempt's own records support. ``rerun``: Modal re-ran the same input (same call id),
    which it does after a preemption or a lost worker, never after the function raised or exited (retries=0). With
    heartbeats that never saw the host tight, that is a preemption even when no signal was recorded (v17 a8, 30 Oct:
    "Container terminated due to preemption", no SIGTERM reached the record; 537 GiB free at the last beat)."""
    if hostmem.tight(attempt.get("mem_last") or {}) or hostmem.tight(attempt.get("mem_min") or {}):
        return "host-memory"
    if attempt.get("signal"):
        return f"preemption ({attempt['signal']['name']})"
    if rerun and attempt.get("mem_last") and attempt.get("mem_min"):
        return "preemption (platform re-run, no signal recorded; host memory healthy)"
    return "unknown (no signal: OOM kill, worker loss or hard crash)"


class RestartGuard:
    def __init__(self, state_root, run_id, *, policy="stop", max_resumes=2, commit=None, ack=None, call_id=None,
                 beat_s=60.0, clock=time.time):
        if policy not in ("stop", "resume"):
            raise ValueError(f"policy {policy!r}: 'stop' or 'resume'")
        self.run_id, self.policy, self.max_resumes = run_id, policy, max_resumes
        self.path = Path(state_root) / ".runops" / run_id / "attempts.json"
        self.commit_fn = commit
        self.ack = ack if ack is not None else os.environ.get("ARC_RESTART_ACK", "")
        self.call_id = call_id if call_id is not None else _call_id()
        self.beat_s, self.clock = beat_s, clock
        self.lock = threading.RLock()
        self.state = None
        self.me = None
        self._stop = threading.Event()

    # --- state -------------------------------------------------------------------------------------------------
    def _save(self, commit=True):
        with self.lock:
            write_json(self.path, self.state)
            if commit and self.commit_fn:
                try:
                    self.commit_fn()
                except Exception as e:  # noqa: BLE001 - a commit failure must not kill the run
                    print(f"RUNOPS commit failed: {e!r}", flush=True)

    def start(self, checkpoint_step=None, pending=False):
        """Record this attempt and decide. ``checkpoint_step``: the newest complete checkpoint's step (or a callable
        returning it); None or 0 means none. ``pending``: no checkpoint, but the dead attempt saved its first update's
        groups (rl_seq seq_pending.json of step 1): a preemption then resumes from them (real-v11, 28 Oct)."""
        if callable(checkpoint_step):
            checkpoint_step = checkpoint_step()
        now = self.clock()
        with self.lock:
            state = read_json(self.path) or {"run_id": self.run_id, "attempts": [], "incidents": []}
            attempts, incidents = state["attempts"], state["incidents"]
            first_of_call = not any(a.get("call_id") == self.call_id for a in attempts if self.call_id)
            mem = hostmem.sample()
            me = {"n": len(attempts) + 1, "call_id": self.call_id, "task_id": os.environ.get("MODAL_TASK_ID"),
                  "region": os.environ.get("MODAL_REGION"), "host": socket.gethostname(), "started": now,
                  "last_beat": now, "mem_last": mem, "mem_min": mem, "signal": None, "ended": None,
                  "policy": self.policy}
            prev = attempts[-1] if attempts else None
            incident = None
            if prev and prev.get("ended") is None:
                prev["ended"] = {"t": now, "status": "died", "seen_by": me["n"]}
                auto = bool(self.call_id) and prev.get("call_id") == self.call_id
                incident = {"id": f"{self.run_id}/a{prev['n']}", "t": now, "dead_attempt": prev["n"],
                            "restart_attempt": me["n"], "auto": auto, "cause": classify(prev, rerun=auto),
                            "alive_s": round(prev.get("last_beat", now) - prev.get("started", now)),
                            "since_beat_s": round(now - prev.get("last_beat", now)),
                            "mem_last": prev.get("mem_last"), "signal": prev.get("signal"),
                            "checkpoint_step": checkpoint_step, "pending": bool(pending), "blocked": True, "ack": None}
                incidents.append(incident)
            acked = []
            if self.ack and first_of_call:  # an agent relaunch that read the cause: record it on every open incident
                if len(self.ack.strip()) < MIN_ACK_CHARS:
                    raise ValueError(f"ARC_RESTART_ACK too short ({self.ack!r}): write the cause as read")
                for inc in incidents:
                    if inc["blocked"] and not (inc is incident and inc["auto"]):
                        inc.update(blocked=False, ack={"t": now, "by_attempt": me["n"], "cause_read": self.ack})
                        acked.append(inc)
            if incident and incident["blocked"]:
                resumes = sum(1 for i in incidents if i.get("resumed"))
                if not incident["auto"]:
                    incident["why"] = "relaunch after a death without ARC_RESTART_ACK"
                elif self.policy != "resume":
                    incident["why"] = "run policy 'stop' (no auto-resume)"
                elif not incident["cause"].startswith("preemption"):
                    incident["why"] = f"cause '{incident['cause']}' is not a preemption"
                elif not checkpoint_step and not pending:
                    incident["why"] = "no checkpoint past step 0 and no saved pending groups (a resume would start over)"
                elif resumes >= self.max_resumes:
                    incident["why"] = f"{resumes} auto-resumes already (max {self.max_resumes})"
                else:
                    incident.update(blocked=False, resumed=True, why="auto-resume: preemption with a checkpoint" if
                                    checkpoint_step else "auto-resume: preemption with saved pending groups (step 0)")
            elif incident:
                incident["why"] = "acked relaunch"
            open_ = [i for i in incidents if i["blocked"]]
            if not incident:  # a relaunch after an earlier incident: the open one, else the one it acked
                incident = open_[-1] if open_ else (acked[-1] if acked else None)
            blocked = bool(open_)
            if blocked:
                me["ended"] = {"t": now, "status": "blocked", "incident": open_[-1]["id"]}
            attempts.append(me)
            self.state, self.me = state, me
            self._save()
        if incident:
            print(f"RUNOPS INCIDENT {incident['id']} cause={incident['cause']} auto={incident['auto']} "
                  f"blocked={blocked} why={incident.get('why')}", flush=True)
        if blocked:
            raise RestartBlocked(incident)
        return Start(attempt=me["n"], restart=incident is not None, resume=incident is not None and bool(checkpoint_step),
                     incident=incident)

    # --- liveness ----------------------------------------------------------------------------------------------
    def beat(self, commit=False, **extra):
        """Heartbeat with a fresh host-memory reading; ``extra`` (e.g. step, phase) lands on the attempt."""
        mem = hostmem.sample()
        with self.lock:
            me = self.me
            me["last_beat"], me["mem_last"] = self.clock(), mem
            if (mem.get("avail_gib") or 1e18) < (me["mem_min"].get("avail_gib") or 1e18):
                me["mem_min"] = mem
            me.update(extra)
            self._save(commit=commit)

    def _beat_loop(self):
        while not self._stop.wait(self.beat_s):
            try:
                self.beat()
            except Exception as e:  # noqa: BLE001
                print(f"RUNOPS heartbeat failed: {e!r}", flush=True)

    def _on_signal(self, signum, frame):
        with self.lock:
            if self.me and not self.me.get("signal"):
                self.me["signal"] = {"name": signal.Signals(signum).name, "t": self.clock()}
                self._save()
        prior = self._prior.get(signum)
        if callable(prior):
            prior(signum, frame)
        elif prior == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def watch(self):
        """Heartbeat thread plus SIGINT/SIGTERM recording (chained to the earlier handlers; main thread only)."""
        self._prior = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prior[sig] = signal.signal(sig, self._on_signal)
            except ValueError:  # not the main thread: no signal evidence, the heartbeat still runs
                pass
        threading.Thread(target=self._beat_loop, daemon=True, name="runops-beat").start()

    def end(self, status="ok"):
        """A clean end (normal return or an application error). Never called for a kill or a preemption signal."""
        self._stop.set()
        with self.lock:
            if self.me and self.me.get("ended") is None:
                self.me["ended"] = {"t": self.clock(), "status": status}
                self._save()
        print(f"RUNOPS END {status}", flush=True)  # the shared watcher's clean-end marker (when status is ok)

    @contextlib.contextmanager
    def run(self, checkpoint_step=None):
        start = self.start(checkpoint_step)
        self.watch()
        try:
            yield start
        except Exception as e:
            self.end(f"error: {type(e).__name__}: {e}"[:300])
            raise
        else:
            self.end("ok")
        # BaseException (KeyboardInterrupt from a preemption SIGINT, SystemExit) leaves the attempt open: a death
