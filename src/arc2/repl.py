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
    def __init__(self, task, *, python=None, strict=True, call_seconds=30, cpu_total=3600, chars=6000, memo=False,
                 import_globals=False, one_answer=False, programs=False, sandbox_fix=False, check_fn=False,
                 restore_names=False):
        self.start_line = json.dumps({"train": task["train"], "test_inputs": [t["input"] for t in task["test"]],
                                      "strict": strict, "call_seconds": call_seconds, "cpu_total": cpu_total,
                                      "chars": chars, **({"memo": True} if memo else {}),
                                      **({"import_globals": True} if import_globals else {}),
                                      **({"one_answer": True} if one_answer else {}),
                                      **({"programs": True} if programs else {}),
                                      **({"sandbox_fix": True} if sandbox_fix else {}),
                                      **({"check_fn": True} if check_fn else {}),
                                      **({"restore_names": True} if restore_names else {}),
                                      **({"extra_inputs": task["extra"]} if task.get("extra") else {})}) + "\n"
        self.python, self.call_seconds = python or sys.executable, int(call_seconds)
        self.history = []  # requests that ran clean, replayed after a restart
        self.codes = []  # every code run, submit cells included (alt_fit: the source of a fitting function)
        self.memo = {"text": "", "writes": 0}  # the playbook as of the last inventory request
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
        self.codes.append(code)
        return self.call({"op": "run", "code": code})

    def inventory(self, timeout=5.0):
        """The live definitions (v8.5p's drain-note inventory), or None when the interpreter is down or slow."""
        if self.process is None or self.process.poll() is not None:
            return None
        try:
            out = self._ask({"op": "inventory"}, timeout)
            self.memo = {"text": out.get("playbook") or "", "writes": int(out.get("playbook_writes") or 0)}
            return out.get("inventory")
        except (TimeoutError, BrokenPipeError, OSError, ValueError):
            self._restart("an inventory request did not answer")
            return None

    def aug(self, transforms, seconds=20.0):
        """chk5: the last submitted function under ``transforms`` ({name: passed or None}); {} when the interpreter
        is down or slow (never replayed after a restart)."""
        if self.process is None or self.process.poll() is not None:
            return {}
        try:
            return self._ask({"op": "aug", "transforms": list(transforms), "seconds": seconds}, seconds + 10).get("aug") or {}
        except (TimeoutError, BrokenPipeError, OSError, ValueError):
            self._restart("an augmentation check did not answer")
            return {}

    def held(self, grid):
        """base16: the last submitted function's output on the held-back input (never replayed), None on any failure."""
        if self.process is None or self.process.poll() is not None:
            return None
        try:
            return self._ask({"op": "held", "grid": grid}, self.call_seconds + 10).get("held")
        except (TimeoutError, BrokenPipeError, OSError, ValueError):
            self._restart("a check did not answer")
            return None

    def fit(self, seconds=10.0, equiv=0.0):
        """The model's functions that reproduce every training pair, with their test outputs; None when the
        interpreter is down or slow."""
        if self.process is None or self.process.poll() is not None:
            return None
        try:
            return self._ask({"op": "fit", "seconds": seconds, "equiv": equiv},
                             seconds + 10 + 40 * equiv).get("fit") or []
        except (TimeoutError, BrokenPipeError, OSError, ValueError):
            self._restart("a fit probe did not answer")
            return None

    def sweep(self, name, source, seconds=10.0):
        """arc2 item 2: train-fitting variants of ``name`` with one constant or comparison changed; None when down."""
        if self.process is None or self.process.poll() is not None:
            return None
        try:
            return self._ask({"op": "sweep", "name": name, "source": source, "seconds": seconds}, seconds + 15).get("sweep")
        except (TimeoutError, BrokenPipeError, OSError, ValueError):
            self._restart("a sweep did not answer")
            return None

    def define(self, name, code):
        return self.call({"op": "define", "name": name, "code": code})

    def close(self):
        self._kill()
