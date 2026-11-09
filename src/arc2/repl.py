"""Host side of the per-task persistent interpreter (``arc2.repl_worker``; setting ``conversation``).

One process lives for the task's whole conversation. A call that outlives the host timeout, or a process that dies,
is restarted and the calls that ran clean are replayed (output discarded), so definitions survive; the model is told.
"""

import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

WORKER = str(Path(__file__).with_name("repl_worker.py"))
ENV = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
RESTARTED = ("The python interpreter was restarted ({why}); the earlier calls that ran without an error were run again, "
             "so their definitions are back.")


class Repl:
    def __init__(self, task, *, python=None, strict=True, call_seconds=30, cpu_total=3600, chars=6000):
        self.start_line = json.dumps({"train": task["train"], "test_inputs": [t["input"] for t in task["test"]],
                                      "strict": strict, "call_seconds": call_seconds, "cpu_total": cpu_total,
                                      "chars": chars}) + "\n"
        self.python, self.call_seconds = python or sys.executable, int(call_seconds)
        self.history = []  # requests that ran clean, replayed after a restart
        self.process = None
        self.restarts = 0

    def _spawn(self):
        self.process = subprocess.Popen([self.python, "-I", "-B", WORKER], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd="/", env=ENV,
                                        start_new_session=True, text=True, bufsize=1)
        self.process.stdin.write(self.start_line)
        self.process.stdin.flush()

    def _kill(self):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            self.process.kill()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.process = None

    def _ask(self, request, timeout):
        if self.process is None or self.process.poll() is not None:
            raise BrokenPipeError("not running")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError
        line = self.process.stdout.readline()
        if not line:
            raise BrokenPipeError("ended")
        return json.loads(line)

    def _restart(self, why):
        self._kill()
        self.restarts += 1
        self._spawn()
        for request in self.history:
            try:
                self._ask(request, self.call_seconds + 10)
            except (TimeoutError, BrokenPipeError, OSError, ValueError):
                self._kill()
                self._spawn()
                break
        return RESTARTED.format(why=why)

    def call(self, request):
        """One request: ``{"stdout", "result"?, "error"?, "submitted"?, "notice"?}``."""
        notice = None
        if self.process is None or self.process.poll() is not None:
            if self.process is None and not self.restarts and not self.history:
                self._spawn()
            else:
                notice = self._restart("it had stopped")
        try:
            out = self._ask(request, self.call_seconds + 10)
        except TimeoutError:
            notice = self._restart(f"a call ran past {self.call_seconds} s")
            return {"stdout": "", "error": f"the call did not finish within {self.call_seconds} s", "notice": notice}
        except (BrokenPipeError, OSError, ValueError):
            notice = self._restart("the call ended the interpreter (memory or CPU limit)")
            return {"stdout": "", "error": "the call ended the interpreter (memory or CPU limit)", "notice": notice}
        if not out.get("error") and not out.get("submitted"):
            self.history.append(request)
        if notice:
            out["notice"] = notice
        return out

    def run(self, code):
        return self.call({"op": "run", "code": code})

    def define(self, name, code):
        return self.call({"op": "define", "name": name, "code": code})

    def close(self):
        self._kill()
