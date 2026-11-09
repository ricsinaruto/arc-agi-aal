# prime2 v4: the puzzle's globals in Prime's IPython kernel, as conv1's interpreter has them (arc2.repl_worker):
# `train`, `test_inputs`, `np`, `submit(first, second=None)` and `program_N` for each program the harness checked.
# Installed as an IPython startup file in the session's HOME; the task file holds no test outputs.
import json as _json
import os as _os
import signal as _signal

import numpy as np

_ARC = _os.path.join(_os.path.expanduser("~"), ".arc")
with open(_os.path.join(_ARC, "task.json")) as _f:
    _task = _json.load(_f)
train = _task["train"]
test_inputs = [t["input"] for t in _task["test"]]
_n_tests = len(test_inputs)
_CALL_SECONDS = int(_task.get("call_seconds", 30))


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


def _valid(grid):
    return (isinstance(grid, list) and 0 < len(grid) <= 30 and all(isinstance(r, list) for r in grid)
            and 0 < len(grid[0]) <= 30 and all(len(r) == len(grid[0]) for r in grid)
            and all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 9 for r in grid for c in r))


def submit(first, second=None):
    """Your answer: ``first`` and optionally ``second`` are your two guesses, each the output grid for the test
    input (a list with one grid per test input when there are several). It ends the puzzle."""
    guesses = []
    for guess in (first, second):
        if guess is None:
            guesses.append(None)
            continue
        guess = _plain(guess)
        if _n_tests == 1 and guess and isinstance(guess[0], list) and guess[0] and not isinstance(guess[0][0], list):
            guess = [guess]  # one test input: a bare grid
        if not isinstance(guess, list) or len(guess) != _n_tests:
            raise ValueError(f"submit: each guess needs {_n_tests} grid(s), one per test input")
        guesses.append(guess)
    if guesses[0] is None:
        raise ValueError("submit: the first guess is missing")
    for guess in (g for g in guesses if g is not None):
        if not all(_valid(g) for g in guess):
            raise ValueError(f"submit: each guess needs {_n_tests} valid grid(s) (lists of lists of ints 0-9, "
                             "rectangular, at most 30x30), one per test input. Nothing was submitted.")
    path = _task["answer_path"]
    with open(path + ".tmp", "w") as f:
        _json.dump({"guesses": guesses}, f)
    _os.replace(path + ".tmp", path)
    return "submitted"


_loaded = set()


def _load_programs(*_):
    folder = _os.path.join(_ARC, "programs")
    try:
        names = sorted(_os.listdir(folder))
    except OSError:
        return
    shell = get_ipython()  # noqa: F821 - IPython startup scope
    for name in names:
        if not name.endswith(".py") or name in _loaded:
            continue
        _loaded.add(name)
        scope = {"__name__": "__program__", "np": np}
        try:
            with open(_os.path.join(folder, name)) as f:
                exec(compile(f.read(), "<program>", "exec"), scope, scope)
        except BaseException:  # noqa: BLE001 - a program that does not load is not defined (as conv1)
            continue
        if callable(scope.get("transform")):
            shell.user_ns[name[:-3]] = scope["transform"]


def _alarm(signum, frame):
    raise TimeoutError(f"the call ran past {_CALL_SECONDS} s")


def _arm(*_):
    _load_programs()
    _signal.signal(_signal.SIGALRM, _alarm)
    _signal.alarm(_CALL_SECONDS)


def _disarm(*_):
    _signal.alarm(0)


get_ipython().events.register("pre_run_cell", _arm)  # noqa: F821
get_ipython().events.register("post_run_cell", _disarm)  # noqa: F821
