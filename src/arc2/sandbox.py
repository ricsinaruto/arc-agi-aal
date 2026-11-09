"""Host side of the program runs: one fresh ``arc2/worker.py`` child per run, killed with its group at the timeout."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

WORKER = str(Path(__file__).with_name("worker.py"))
ENV = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def run_cells(cells, code, task, *, seconds=30, cell_seconds=10, python=None, strict=True, helpers=False, show=False,
              fixes=False, chars=0):
    """One `python` tool call of the tool arm: ``cells`` (the conversation's earlier snippets that ran clean) are run
    again in a fresh child, then ``code``, with ``train`` and ``test_inputs`` preloaded: the same guard as a program
    run, and a session that persists without a process that does. ``{"stdout", "result"?}`` or ``{"error",
    "stdout"}``. ``fixes`` (setting tool_fixes, C1): ``import np`` and the like bind the preloaded names, sys,
    traceback, inspect and pprint import, a failing snippet returns ``kept`` (its statements before the failing one)
    and output is cut to ``chars`` as start and end."""
    return _child({"cells": list(cells), "code": code, "train": task["train"],
                   "test_inputs": [t["input"] for t in task["test"]], "cpu_seconds": seconds,
                   "grid_seconds": cell_seconds, "strict": strict, "helpers": helpers, "show": show,
                   **({"fixes": True, "chars": int(chars)} if fixes else {})}, seconds, python, ("stdout", "error"))


def run(code, inputs, *, seconds=30, grid_seconds=4, python=None, strict=True, helpers=False, fixes=False, trace=None):
    """``{"outputs": [grid or {"error"}], "stdout"}`` or ``{"error", "stdout"}`` when the program did not load, ran out
    of time or broke the worker. ``fixes``: the imports of ``run_cells``'s. ``trace``: the index of one input whose
    run is also returned as a line trace (``"trace"``)."""
    return _child({"code": code, "inputs": inputs, "cpu_seconds": seconds, "grid_seconds": grid_seconds,
                   "strict": strict, "helpers": helpers, **({"fixes": True} if fixes else {}),
                   **({"trace": int(trace)} if trace is not None else {})}, seconds, python,
                  ("outputs", "error"))


def _child(payload, seconds, python, keys):
    request = json.dumps(payload)
    process = subprocess.Popen([python or sys.executable, "-I", "-B", WORKER], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/", env=ENV,
                               start_new_session=True)
    try:
        out, err = process.communicate(request.encode(), timeout=seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.communicate()
        return {"error": f"the program did not finish within {seconds} s", "stdout": ""}
    try:
        result = json.loads(out.decode(errors="replace").strip().splitlines()[-1])
        if not isinstance(result, dict) or not any(key in result for key in keys):
            raise ValueError
        return result
    except (ValueError, IndexError):
        lines = err.decode(errors="replace").strip().splitlines()
        if "Error in sitecustomize" in "".join(lines[-2:-1]):  # Kaggle's own start-up noise, not the program's
            lines = lines[:-2]
        tail = lines[-1:] or ["no output"]
        return {"error": f"the program ended the interpreter (exit {process.returncode}): {tail[0][:300]}",
                "stdout": ""}


def self_test(python=None, strict=True):
    """Run before a trial: the sandbox executes a program with numpy and scipy, reports an error and survives a loop."""
    good = run("import numpy as np\nfrom scipy import ndimage\ndef transform(grid):\n    return np.array(grid).T",
               [[[1, 2], [3, 4]]], python=python, strict=strict)
    bad = run("def transform(grid):\n    return grid[5]", [[[1]]], python=python, strict=strict)
    loop = run("def transform(grid):\n    while True: pass", [[[1]]], seconds=6, grid_seconds=1, python=python,
               strict=strict)
    wrote = run("def transform(grid):\n    open('/tmp/x', 'w')", [[[1]]], python=python, strict=strict)
    checks = {"runs": good.get("outputs") == [[[1, 3], [2, 4]]],
              "error_reported": "IndexError" in str((bad.get("outputs") or [{}])[0]),
              "loop_stopped": "TimeoutError" in str(loop.get("outputs") or loop.get("error")),
              "no_open": "NameError" in str((wrote.get("outputs") or [{}])[0])}
    return {"ok": all(checks.values()), "checks": checks, "strict": strict, "detail": {"good": good, "loop": loop}}
