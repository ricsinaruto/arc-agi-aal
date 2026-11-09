"""Child process of ``arc2.repl``: one persistent Python interpreter for one task's whole conversation (setting
``conversation``), as the ARC-AGI-3 agent's per-game interpreter.

Run by path with ``python -I``. The first stdin line is the task and the limits; each later line is one request
(``{"op": "run", "code"}`` or ``{"op": "define", "name", "code"}``) and gets one JSON line back. The guard is
``arc2.worker``'s (CPU and address-space rlimits, no file writes, an unprivileged uid when root, the import allow-list,
builtins without ``open``); each request also has a SIGALRM limit.
"""

import builtins
import contextlib
import copy
import io
import json
import os
import signal
import sys
import traceback

ALLOWED = {"numpy", "scipy", "math", "collections", "itertools", "functools", "heapq", "bisect", "statistics",
           "random", "re", "json", "operator", "fractions", "decimal", "copy", "typing", "dataclasses", "string",
           "enum", "abc", "queue", "array", "numbers", "time", "sys", "traceback", "inspect", "pprint"}


def plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value if isinstance(value, (int, str, type(None))) else repr(value)[:80]


def clip(text, limit):
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}\n... [{len(text) - limit} characters not shown] ...\n{text[len(text) - (limit - head):]}"


def where(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename in ("<python>", "<program>")]
    line = f" (line {frames[-1].lineno})" if frames else ""
    return f"{type(exc).__name__}: {str(exc)[:400]}{line}"


def main():
    real = os.fdopen(os.dup(1), "w")
    start = json.loads(sys.stdin.readline())
    import numpy as np  # before the limits and the allow-list

    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (int(start["cpu_total"]), int(start["cpu_total"]) + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        if start.get("strict", True):
            resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    except (ImportError, ValueError, OSError):  # macOS refuses RLIMIT_AS; the host's timeout still applies
        pass
    if start.get("strict", True) and hasattr(os, "geteuid") and os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED:
            raise ImportError(f"module not available here: {name} (available: numpy, scipy and the standard library)")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in vars(builtins).items() if k not in ("open", "input", "breakpoint", "exit", "quit")}
    safe["__import__"] = safe_import
    n_tests = len(start["test_inputs"])
    submitted = {}

    def submit(first, second=None):
        """Your answer: ``first`` and optionally ``second`` are your two guesses, each the output grid for the test
        input (a list with one grid per test input when there are several). It ends the puzzle."""
        guesses = []
        for guess in (first, second):
            if guess is None:
                guesses.append(None)
                continue
            guess = plain(guess)
            if n_tests == 1 and guess and isinstance(guess[0], list) and guess[0] and not isinstance(guess[0][0], list):
                guess = [guess]  # one test input: a bare grid
            if not isinstance(guess, list) or len(guess) != n_tests:
                raise ValueError(f"submit: each guess needs {n_tests} grid(s), one per test input")
            guesses.append(guess)
        submitted["guesses"] = guesses
        return "submitted"

    namespace = {"__builtins__": safe, "__name__": "__main__", "np": np, "submit": submit,
                 "train": copy.deepcopy(start["train"]), "test_inputs": copy.deepcopy(start["test_inputs"])}
    seconds, chars = int(start["call_seconds"]), int(start["chars"])

    def alarm(signum, frame):
        raise TimeoutError(f"more than {seconds} s on one call")

    signal.signal(signal.SIGALRM, alarm)
    for line in sys.stdin:
        request = json.loads(line)
        printed = io.StringIO()
        out = {}
        submitted.clear()
        try:
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
                signal.alarm(seconds)
                if request["op"] == "define":  # a reply's program, callable as ``request["name"]``
                    scope = {"__builtins__": safe, "__name__": "__program__", "np": np}
                    exec(compile(request["code"], "<program>", "exec"), scope, scope)
                    if callable(scope.get("transform")):
                        namespace[request["name"]] = scope["transform"]
                else:
                    namespace.pop("result", None)
                    exec(compile(request["code"], "<python>", "exec"), namespace, namespace)
                signal.alarm(0)
            if namespace.get("result") is not None and request["op"] == "run":
                out["result"] = clip(repr(plain(namespace["result"])), chars)
        except BaseException as exc:  # noqa: BLE001 - shown to the model
            signal.alarm(0)
            out["error"] = where(exc)
        if submitted.get("guesses") is not None:
            out["submitted"] = submitted["guesses"]
        out["stdout"] = clip(printed.getvalue(), chars)
        real.write(json.dumps(out) + "\n")
        real.flush()


if __name__ == "__main__":
    main()
