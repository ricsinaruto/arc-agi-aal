"""Host side of the persistent python sandbox: one interpreter per game (``inference.agent.python_tool_sandbox``)
whose user definitions survive between tool calls. Each call sends only the history entries the child has not seen,
seeds ``random`` from the call, and gets ``timeout_seconds``: past it the call is interrupted (SIGUSR1) and, after the
grace, the child is killed and the next call starts a fresh one (the model is told its state was reset). ``close``
kills the child at the end of the try (SIGKILL to its process group)."""

import collections
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

RESET_AFTER_TIMEOUT = "python state was reset after a timeout (redefine helpers before reuse)"
RESET_AFTER_EXIT = "Sandbox process exited unexpectedly; python state was reset"
INTERRUPTED_AFTER_TIMEOUT = (
    "the running statement was interrupted and python state was kept (functions and variables defined so far are "
    "still available; the interrupted call may have left partial results)"
)

def _kill(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


# tool code whose output depends on wall-clock time (a restored try cannot replay it)
CLOCK_CODE = re.compile(r"\b(?:time\.(?:time|time_ns|perf_counter|perf_counter_ns|monotonic|monotonic_ns|process_time)"
                        r"|perf_counter|monotonic|process_time|datetime\.(?:now|utcnow|today)|date\.today)\s*\(")


class StatefulSandbox:
    """Drop-in for ``run_sandboxed_python`` that keeps one child alive per game."""

    def __init__(self, bootstrap, *, color_chars, on_event=None, label=""):
        self.bootstrap, self.color_chars = bootstrap, color_chars
        self.on_event, self.label = on_event, label
        self.process = None
        self.tempdir = None
        self.lines = None
        self.stderr = collections.deque(maxlen=50)
        self.calls = self.restarts = 0
        self.closed = False
        self.cpu_seconds = 1200
        self.lock = threading.Lock()
        # History entries the live child already holds (delta history).
        self.synced = 0
        # A call past its timeout is interrupted (SIGUSR1) and gets ``interrupt_grace`` seconds before the child is
        # killed (arena harness_fixes timeout_interrupt; the grace comes from ``configure_fixes``).
        self.interrupt_grace = 3.0
        self.interrupts = 0
        # arena dfr_harness (``arena.dfr_harness.with_dfr``): the host side of the child's lazy animation views
        # (``handler(request) -> payload``; None: the child never asks) and the guard refusals it reported.
        self.animation_handler = None
        self.dfr_guards = []
        # arena hx_harness toolbox (``arena.hx_harness.with_toolbox``): the child's last registry / observer report.
        self.hx_state = None

    def _call_seed(self, code, initial_state):
        """The call's ``random`` seed: a function of this game's call count, its history length and the code, so a
        restored try replays every tool call to the recorded output (R leg 25 Oct: 14 of 30 bad restored starts were
        tool code drawing from the unseeded ``random``). Code that reads the clock cannot replay: logged
        ``duck_python_clock`` (arena.restore.UNSAFE). The first call logs ``duck_python_rng`` (a seeded recording)."""
        try:
            size = len((initial_state or {}).get("history") or [])
        except TypeError:
            size = 0
        if self.calls == 1:
            self._event("duck_python_rng", seeded=1)
        if CLOCK_CODE.search(code or ""):
            self._event("duck_python_clock", call=self.calls)
        digest = hashlib.sha256(f"{self.calls}\0{size}\0{code}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def configure_fixes(self, bootstrap, *, grace=3.0):
        """The arena's harness_fixes setup before the first call: ``bootstrap`` and the interrupt grace."""
        if self.process is not None:
            raise RuntimeError("Sandbox fixes must be configured before the interpreter starts")
        self.bootstrap, self.interrupt_grace = bootstrap, float(grace)

    def _interrupt(self):
        try:
            os.kill(self.process.pid, signal.SIGUSR1)
        except (OSError, AttributeError):
            pass

    def _event(self, kind, **data):
        if self.on_event is not None:
            self.on_event(kind, **data)

    def _spawn(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="duck_python_")
        process = subprocess.Popen(
            # -s -P -S, not -I: -I implies -E, which ignores PYTHONHASHSEED below (22 Oct: v6m10 R9 restores printed
            # {'W', 'N'} vs a recorded {'N', 'W'}; the hash seed was random per sandbox). The env is built explicitly.
            [sys.executable, "-s", "-P", "-S", "-c", self.bootstrap],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=self.tempdir.name,
            env={
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                # set/dict print order is the same in every process, so a restored try replays tool output exactly
                # (pf hvpf-v1: a printed {'W', 'N'} flipped order and broke one harvested start's replay)
                "PYTHONHASHSEED": "0",
                "HOME": "/tmp",
                "TMPDIR": "/tmp",
                "PATH": os.environ.get("PATH", ""),
            },
            start_new_session=True,
        )
        lines = queue.Queue()

        def read_stdout():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        def read_stderr():
            for line in process.stderr:
                self.stderr.append(line)

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        self.process, self.lines = process, lines
        self.synced = 0

    def _delta(self, state):
        """State with only the history entries the child has not received yet."""
        state = dict(state or {})
        history = state.get("history") or []  # a list, or the host's lazy duck_transfer.agent.HistoryPayloads
        size = len(history)
        base = self.synced if size >= self.synced else 0
        state["history"], state["history_base"] = list(history[base:]), base
        self.synced = size
        return state

    def _discard(self, reason):
        process, self.process, self.lines = self.process, None, None
        self.synced = 0
        if process is not None:
            _kill(process)
            try:
                process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if self.tempdir is not None:
            self.tempdir.cleanup()
            self.tempdir = None
        if reason:
            self.restarts += 1
            self._event("duck_python_reset", reason=reason, restarts=self.restarts)

    def _send(self, payload):
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def run(self, *, code, timeout_seconds, initial_state, action_handler):
        with self.lock:
            if self.closed:
                return {"error": "Sandbox closed", "stdout": "", "action_results": []}
            self.calls += 1
            results = []
            seed = self._call_seed(code, initial_state)
            try:
                if self.process is None:
                    self._spawn()
                    self._send(
                        {
                            "code": code,
                            "timeout_seconds": timeout_seconds,
                            "sandbox_cwd": self.tempdir.name,
                            "state": self._delta(initial_state),
                            "color_chars": self.color_chars,
                            "cpu_seconds": int(self.cpu_seconds),
                            "rng_seed": seed,
                        }
                    )
                else:
                    self._send({"type": "code", "code": code, "state": self._delta(initial_state), "rng_seed": seed})
            except OSError:
                self._discard("start_failed")
                return {"error": "Sandbox process could not start.", "stdout": "", "action_results": []}
            if self.closed:
                self._discard(None)
                return {"error": "Sandbox closed", "stdout": "", "action_results": []}
            deadline = time.monotonic() + max(1, int(timeout_seconds))
            interrupted_at = None  # harness_fixes: the time the interrupt was first sent
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if (interrupted_at is None and self.process is not None
                            and self.process.poll() is None):
                        interrupted_at = time.monotonic()
                        deadline = interrupted_at + self.interrupt_grace
                        self._interrupt()
                        self._event("duck_python_interrupt", timeout=timeout_seconds)
                        continue
                    self._discard("timeout")
                    return {
                        "error": f"Tool timed out after {timeout_seconds}s; {RESET_AFTER_TIMEOUT}",
                        "stdout": "",
                        "action_results": list(results),
                    }
                try:
                    line = self.lines.get(timeout=min(remaining, 0.5) if interrupted_at is not None else remaining)
                except queue.Empty:
                    if interrupted_at is not None:
                        self._interrupt()  # a bare ``except:`` in the call may have swallowed the first one
                    continue
                if line is None:
                    self._discard(None if self.closed else "exit")
                    return {
                        "error": "Sandbox closed" if self.closed else RESET_AFTER_EXIT,
                        "stdout": "",
                        "action_results": list(results),
                    }
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._discard("invalid_response")
                    return {
                        "error": f"Sandbox process returned an invalid response; {RESET_AFTER_EXIT}",
                        "stdout": "",
                        "action_results": list(results),
                    }
                kind = str(message.get("type", "")).strip()
                if kind == "animation":
                    # arena dfr_harness: a lazy animation view asked for the last animated action's frames/timeline.
                    self._send(self._animation_reply(message, refuse=interrupted_at is not None))
                    continue
                if kind == "action" and interrupted_at is not None:
                    # Sent before the interrupt landed: refused, never executed (the child raises on the reply).
                    self._send({"type": "action_error", "error": "tool timeout: action not executed"})
                    continue
                if kind == "action":
                    try:
                        reply = action_handler(list(message.get("actions") or []))
                    except Exception:  # noqa: BLE001
                        self._send({"type": "action_error", "error": "action failed in sandbox host."})
                        continue
                    raw = reply.get("action_result") or {}
                    if isinstance(raw, dict):
                        results.append(dict(raw))
                    self._send(
                        {
                            "type": "action_result",
                            "action_result": raw,
                            "state": self._delta(reply.get("state") or {}),
                        }
                    )
                    continue
                if kind in {"final", "error"}:
                    output = {
                        "stdout": str(message.get("stdout", "") or ""),
                        "result": message.get("result"),
                        "error": str(message.get("error", "") or ""),
                        "action_results": list(message.get("action_results") or results),
                        "segmentation": message.get("segmentation"),
                        "history_size": message.get("history_size"),
                    }
                    guards = message.get("dfr_guards")
                    if guards:
                        # arena dfr_harness: stale-state / terminal refusals raised in the child (nothing executed).
                        self.dfr_guards.extend(guards)
                        self._event("duck_python_dfr_guards", guards=guards)
                    hx = message.get("hx")
                    if isinstance(hx, dict):
                        # arena hx_harness: registered tools / observers and the last observer outputs.
                        self.hx_state = hx
                    if interrupted_at is not None and message.get("history_size") != self.synced:
                        # Resend the whole history next time unless the child reports the entries the host sent
                        # (v7.1: a full resync of a 20K-step history was a ~300 MB message built under the GIL).
                        self.synced = 0
                    if interrupted_at is not None:
                        if kind == "error":
                            self.interrupts += 1
                            self._event("duck_python_interrupted", interrupts=self.interrupts,
                                        seconds=round(time.monotonic() - interrupted_at, 3))
                            detail = output["error"]
                            output["error"] = (f"Tool timed out after {timeout_seconds}s; {INTERRUPTED_AFTER_TIMEOUT}"
                                               + (f"\n{detail}" if detail else ""))
                            output["interrupted"] = True
                    return output
                self._discard("unknown_message")
                return {
                    "error": f"Sandbox process returned an unknown message type; {RESET_AFTER_EXIT}",
                    "stdout": "",
                    "action_results": list(results),
                }

    def _animation_reply(self, message, refuse=False):
        handler = self.animation_handler
        try:
            if handler is None or refuse:
                raise RuntimeError("no animation handler")
            return {"type": "animation_result", "animation": handler(dict(message.get("request") or {}))}
        except Exception:  # noqa: BLE001
            return {"type": "animation_error", "error": "Animation frames are not available."}

    def inventory(self, timeout=5.0):
        """The live user definitions (``LOOP_FIXES`` inventory request), or None when there is no interpreter or it
        cannot answer. A child that does not answer within ``timeout`` is interrupted, then discarded."""
        with self.lock:
            if self.closed or self.process is None:
                return None
            try:
                self._send({"type": "inventory"})
            except OSError:
                self._discard("inventory_failed")
                return None
            deadline = time.monotonic() + timeout
            interrupted = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not interrupted:
                        interrupted = True
                        self._interrupt()
                        deadline = time.monotonic() + self.interrupt_grace
                        continue
                    self._discard("inventory_timeout")
                    return None
                try:
                    line = self.lines.get(timeout=remaining)
                except queue.Empty:
                    continue
                if line is None:
                    self._discard(None if self.closed else "exit")
                    return None
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._discard("invalid_response")
                    return None
                if message.get("type") == "animation":  # a lazy animation view bound to a listed variable
                    self._send(self._animation_reply(message))
                    continue
                if message.get("type") != "inventory":
                    self._discard("unknown_message")
                    return None
                message.pop("type", None)
                return message

    def close(self):
        """Kill the child (unblocking a call in progress) and drop the working directory."""
        self.closed = True
        process = self.process
        if process is not None:
            _kill(process)
        # A call in progress observes EOF and discards; otherwise clean up here.
        if not self.lock.locked():
            with self.lock:
                self._discard(None)
