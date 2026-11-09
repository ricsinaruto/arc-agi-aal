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
import inspect
import io
import json
import linecache
import reprlib
import os
import signal
import sys
import time
import traceback
import types

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


def transforms():
    """chk5's augmentation set: 7 dihedral maps and 3 fixed colour permutations of 1-9 (0, the background, kept)."""
    import random

    import numpy as np
    rng, perms = random.Random(20261029), []
    for _ in range(3):
        p = list(range(1, 10))
        rng.shuffle(p)
        perms.append([0] + p)
    out = {"rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2), "rot270": lambda g: np.rot90(g, 3),
           "flipud": np.flipud, "fliplr": np.fliplr, "transpose": lambda g: np.asarray(g).T,
           "antitranspose": lambda g: np.rot90(np.asarray(g), 2).T}
    for i, p in enumerate(perms):
        out[f"color{i + 1}"] = (lambda table: (lambda g: table[np.asarray(g)]))(np.asarray(p))
    return out


def clip(text, limit):
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}\n... [{len(text) - limit} characters not shown] ...\n{text[len(text) - (limit - head):]}"


RESTORED = ("[harness] {names} {verb} provided by the harness and {was} restored after this call (anything that used "
            "your version in this call did not count); use other names for your own definitions.")


def grid_ok(grid):
    """A non-empty rectangular list of lists of ints (check_fn)."""
    return (isinstance(grid, list) and bool(grid) and all(isinstance(r, list) and r and len(r) == len(grid[0])
                                                          for r in grid) and all(type(c) is int for r in grid for c in r))


def model_code(filename):
    """The model's code: "<python>" / "<program>", or "<python-N>" / "<program-N>" under ``sandbox_fix``."""
    return str(filename).startswith(("<python", "<program"))


def renamed(code, filename):
    """``code`` and its nested code objects under ``filename`` (compiled as "<python>", so error text is unchanged)."""
    return code.replace(co_filename=filename, co_consts=tuple(
        renamed(c, filename) if isinstance(c, types.CodeType) else c for c in code.co_consts))


def where(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if model_code(f.filename)]
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
        if (not level and name in ("train", "test_inputs", "extra_inputs", "np") and name in namespace
                and (start.get("sandbox_fix") or start.get("import_globals") and not fromlist)):
            return namespace[name]  # tool_fix (30 Oct) / sandbox_fix (base20, any form): the session's own names
        if level or name.split(".")[0] not in ALLOWED:
            raise ImportError(f"module not available here: {name} (available: numpy, scipy and the standard library)")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in vars(builtins).items() if k not in ("open", "input", "breakpoint", "exit", "quit")}
    safe["__import__"] = safe_import
    n_tests = len(start["test_inputs"])
    submitted = {}

    n_train = len(start["train"])
    last = {"fn": None}  # chk5: the last submitted first function, for the ``aug`` op

    def run_fn(fn, grids):
        outs = []
        for grid in grids:
            try:
                outs.append(plain(fn(copy.deepcopy(grid))))
            except Exception as exc:  # noqa: BLE001 - reported per grid
                outs.append({"error": where(exc)})
        return outs

    def submit(first, second=None, force=False):
        """Your answer: ``first`` and optionally ``second`` are your two guesses, each the output grid for the test
        input (a list with one grid per test input when there are several), or a function ``transform(grid)``: a
        function is run on every training input first (``submit_check``) and its test outputs are the guess."""
        check = None
        if callable(first):
            last["fn"] = first
            check = run_fn(first, [p["input"] for p in start["train"]] + start["test_inputs"])
            first = check[n_train:]
        check2 = None
        if callable(second):  # also run on the training inputs (``submit_check2``: both functions are checked)
            check2 = run_fn(second, [p["input"] for p in start["train"]] + start["test_inputs"])
            second = check2[n_train:]
        submitted["check"], submitted["check2"], submitted["forced"] = check, check2, bool(force)
        guesses = []
        for guess in (first, second):
            if guess is None:
                guesses.append(None)
                continue
            guess = plain(guess)
            if n_tests == 1 and guess and isinstance(guess[0], list) and guess[0] and not isinstance(guess[0][0], list):
                guess = [guess]  # one test input: a bare grid
            if not isinstance(guess, list) or len(guess) != n_tests:
                raise ValueError("submit: pass your transform function; grids alone are refused" if start.get("programs")
                                 else f"submit: {'your answer' if start.get('one_answer') else 'each guess'} needs "
                                 f"{n_tests} grid(s), one per test input")
            guesses.append(guess)
        submitted["guesses"] = guesses
        return "submitted"

    def one_answer(answer):
        """Your answer: the output grid for the test input (a list with one grid per test input when there are
        several), or a function ``transform(grid)`` whose test outputs are the answer."""
        return submit(answer)

    one_answer.__name__ = one_answer.__qualname__ = "submit"
    if start.get("programs"):
        one_answer.__doc__ = "Your answer: a function ``transform(grid)``. Its test outputs are recorded if it passes the check."

    def held_check(grid):
        """The last submitted function on the held-back input, in a forked child: nothing it stores or prints, and no
        trace of the grid, reaches the model's interpreter."""
        read, write = os.pipe()
        child = os.fork()
        if not child:
            os.close(read)
            try:
                signal.alarm(seconds)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    value = plain(last["fn"](copy.deepcopy(grid)))
                signal.alarm(0)
                os.write(write, json.dumps(value).encode())
            finally:
                os._exit(0)
        os.close(write)
        chunks = []
        with os.fdopen(read, "rb") as pipe:
            while True:
                chunk = pipe.read(1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
        os.waitpid(child, 0)
        return json.loads(b"".join(chunks)) if chunks else None
    cells = {"n": 0}

    def source(code, base):
        """Compiled ``code``; under ``sandbox_fix`` each cell gets its own name in linecache so inspect.getsource
        works on what it defines."""
        compiled = compile(code, base, "exec")
        if not start.get("sandbox_fix"):
            return compiled
        cells["n"] += 1
        name = f"{base[:-1]}-{cells['n']}>"
        linecache.cache[name] = (len(code), None, code.splitlines(True) or ["\n"], name)
        return renamed(compiled, name)

    ledger = {"checks": 0, "outputs": {}}  # check_fn: test outputs (json) of passing functions -> their labels

    def check(fn):
        """check(transform): runs it on every training input and prints, per pair, pass or the differences, then its
        test outputs and the distinct test outputs of every function that passed all pairs so far."""
        if not callable(fn):
            raise TypeError("check(fn) takes your transform function, e.g. check(transform)")
        ledger["checks"] += 1
        label = f"{getattr(fn, '__name__', 'fn')} (check {ledger['checks']})"
        outs = run_fn(fn, [p["input"] for p in start["train"]])
        ok = 0
        for i, (pair, out) in enumerate(zip(start["train"], outs), 1):
            want = pair["output"]
            if isinstance(out, dict):
                print(f"pair {i}: error {out['error']}")
            elif not grid_ok(out):
                print(f"pair {i}: not a grid of ints: {short.repr(out)}")
            elif (len(out), len(out[0])) != (len(want), len(want[0])):
                print(f"pair {i}: FAIL shape {len(out)}x{len(out[0])}, expected {len(want)}x{len(want[0])}")
            else:
                diff = [(r, c, out[r][c], want[r][c]) for r in range(len(want)) for c in range(len(want[0]))
                        if out[r][c] != want[r][c]]
                if diff:
                    print(f"pair {i}: FAIL {len(diff)} cells differ; first (row, col, got, expected): {diff[:6]}")
                else:
                    ok += 1
                    print(f"pair {i}: pass")
        tests = run_fn(fn, start["test_inputs"])
        for k, out in enumerate(tests, 1):
            if isinstance(out, dict):
                print(f"test {k}: error {out['error']}")
            else:
                print(f"test {k} output:\n" + ("\n".join(" ".join(str(c) for c in row) for row in out)
                                               if grid_ok(out) else short.repr(out)))
        passed = ok == len(outs) and all(grid_ok(t) for t in tests)
        print(f"{ok}/{len(outs)} training pairs pass.")
        if passed:
            key = json.dumps(tests)
            if key in ledger["outputs"]:
                print("Same test outputs as an earlier passing function.")
            ledger["outputs"].setdefault(key, []).append(label)
        if ledger["outputs"]:
            print(f"Distinct test outputs of functions that pass every training pair: {len(ledger['outputs'])}")
            for j, (key, labels) in enumerate(ledger["outputs"].items(), 1):
                mine = passed and key == json.dumps(tests)
                print(f"#{j} from {', '.join(labels)}" + (": this function's output above" if mine else ":"))
                if not mine:
                    for k, out in enumerate(json.loads(key), 1):
                        print(f"test {k}:\n" + "\n".join(" ".join(str(c) for c in row) for row in out))
        return passed

    namespace = {"__builtins__": safe, "__name__": "__main__", "np": np,
                 "submit": one_answer if start.get("one_answer") else submit,
                 "train": copy.deepcopy(start["train"]), "test_inputs": copy.deepcopy(start["test_inputs"])}
    if start.get("extra_inputs"):
        namespace["extra_inputs"] = copy.deepcopy(start["extra_inputs"])
    if start.get("sandbox_fix"):
        namespace["result"] = None
    if start.get("check_fn"):
        namespace["check"] = check
    fixed = {k: namespace[k] for k in ("submit", "np", "playbook", "check") if k in namespace}  # + program_N
    pristine = {k: copy.deepcopy(namespace[k]) for k in ("train", "test_inputs", "extra_inputs") if k in namespace}

    def restore():
        """restore_names: the harness's names back after a call that rebound or changed them."""
        replaced = []
        for name, value in fixed.items():
            if namespace.get(name) is not value:
                namespace[name] = value
                replaced.append(name)
        for name, value in pristine.items():
            try:
                same = namespace.get(name) == value
            except Exception:  # noqa: BLE001 - e.g. an array's ambiguous truth value
                same = False
            if not same:
                namespace[name] = copy.deepcopy(value)
                replaced.append(name)
        return replaced

    seconds, chars = int(start["call_seconds"]), int(start["chars"])

    def alarm(signum, frame):
        raise TimeoutError(f"more than {seconds} s on one call")

    signal.signal(signal.SIGALRM, alarm)
    short = reprlib.Repr()
    short.maxstring = short.maxother = 48
    short.maxlist = short.maxtuple = short.maxset = short.maxfrozenset = short.maxdict = 4
    short.maxdeque = short.maxarray = 4
    short.maxlevel = 2
    preset = {"__builtins__", "__name__", "np", "submit", "train", "test_inputs", "extra_inputs", "result", "playbook",
              "check"}
    memo = {"text": "", "writes": 0}

    def playbook(text=None):
        """Your playbook: playbook(text) replaces it, playbook() returns it. It is what survives when the
        conversation is cut."""
        if text is not None:  # v8.5p memo_style 2 (playbook_open): no length limit
            text = str(text).strip()
            print("[playbook] saved, %d characters" % len(text))
            memo["text"] = text
            memo["writes"] += 1
        return memo["text"]

    if start.get("memo"):
        namespace["playbook"] = fixed["playbook"] = playbook

    def inventory(limit=120):  # duck_transfer.sandbox._inventory on this namespace ("<python>": the model's own code)
        functions, classes, modules, values = [], [], [], []
        for name, value in list(namespace.items()):
            if name in preset or name.startswith("__"):
                continue
            try:
                function = getattr(value, "__wrapped__", value)
                own = isinstance(function, types.FunctionType) and model_code(function.__code__.co_filename)
                if own:
                    try:
                        signature = str(inspect.signature(function))
                    except (TypeError, ValueError):
                        signature = "(...)"
                    doc = (function.__doc__ or "").strip().split("\n", 1)[0]
                    functions.append([name, signature, doc])
                elif isinstance(value, type):
                    classes.append(name)
                elif isinstance(value, types.ModuleType) or callable(value):
                    modules.append(name)
                else:
                    try:
                        size = len(value)
                    except Exception:
                        size = None
                    values.append([name, type(value).__name__, size, short.repr(value)[:80]])
            except Exception:
                values.append([name, "?", None, "<unavailable>"])
        return {"functions": functions[-limit:], "classes": classes[-limit:], "modules": modules[-limit:],
                "values": values[-limit:],
                "counts": {"functions": len(functions), "classes": len(classes), "values": len(values)}}

    def augment_check(fn, names, budget):
        """chk5: per transform name, whether ``fn`` maps every transformed training input to the transformed
        expected output (dihedral transforms; colour permutations keep 0); None when the budget ran out."""
        if not callable(fn):
            return {}
        maps = transforms()
        out, began = {}, time.monotonic()
        for name in names:
            f = maps.get(name)
            if f is None or time.monotonic() - began > budget:
                out[name] = None
                continue
            ok = True
            for pair in start["train"]:
                want = plain(f(np.asarray(pair["input"]))), plain(f(np.asarray(pair["output"])))
                signal.alarm(max(1, int(budget)))
                try:
                    got = plain(fn(copy.deepcopy(want[0])))
                except BaseException:  # noqa: BLE001 - a failing call fails the transform
                    got = None
                finally:
                    signal.alarm(0)
                if got != want[1]:
                    ok = False
                    break
            out[name] = ok
        return out

    def equivariance(fn, tests, budget):
        """Verifier probe (30 Oct): per transform T, [train_ok, test_ok]: train_ok = fn(T(x)) == T(y) on every training
        pair; test_ok (only where train_ok) = fn(T(x)) == T(fn(x)) on every test input. None past the budget."""
        out, began = {}, time.monotonic()
        for name, f in transforms().items():
            if time.monotonic() - began > budget:
                out[name] = None
                continue
            signal.alarm(max(1, int(budget)))
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    ok = all(plain(fn(plain(f(np.asarray(p["input"]))))) == plain(f(np.asarray(p["output"])))
                             for p in start["train"])
                    hold = ok and all(not isinstance(t, dict) and
                                      plain(fn(plain(f(np.asarray(x))))) == plain(f(np.asarray(t)))
                                      for x, t in zip(start["test_inputs"], tests))
            except BaseException:  # noqa: BLE001 - a failing or hanging call fails the transform
                ok, hold = False, False
            finally:
                signal.alarm(0)
            out[name] = [ok, hold if ok else None]
            if ok and not hold:  # where: per test output, the cells (in fn(x)'s frame) where T^-1(fn(T(x))) != fn(x)
                out[name].extend(equivariance_cells(fn, f, name, tests))
        return out

    def equivariance_cells(fn, f, name, tests):
        inverse = {"rot90": "rot270", "rot270": "rot90"}.get(name, name)
        back = transforms()[inverse]
        cells, grids = [], []  # grids: T^-1(fn(T(x))), the candidate the transform implies
        for x, t in zip(start["test_inputs"], tests):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    moved = np.asarray(back(np.asarray(plain(fn(plain(f(np.asarray(x))))))))
                if name.startswith("color"):  # undo the colour map on the values as well
                    table = np.asarray(f(np.arange(10)))
                    moved = np.argsort(table)[moved]
                own = np.asarray(t)
                grids.append(plain(moved))
                cells.append(None if isinstance(t, dict) or moved.shape != own.shape
                             else [[int(a), int(b)] for a, b in zip(*np.nonzero(moved != own))][:400])
            except BaseException:  # noqa: BLE001
                cells.append(None)
                grids.append(None)
        return cells, grids

    def fit_check(budget, limit=40, equiv=0.0):
        """Train-fit probe (fit1 sim, 30 Oct): every function the model defined (newest first, at most ``limit``)
        that maps each training input to its output exactly, with its test outputs (``equiv`` > 0: plus each one's
        equivariance verdicts within that many seconds). Nothing is changed."""
        fns = [(k, v) for k, v in list(namespace.items()) if callable(v) and not k.startswith("_")
               and model_code(getattr(getattr(v, "__code__", None), "co_filename", ""))]
        found, began = [], time.monotonic()
        for name, fn in fns[::-1][:limit]:
            if time.monotonic() - began > budget:
                break
            signal.alarm(max(1, int(budget)))
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    outs = run_fn(fn, [p["input"] for p in start["train"]] + start["test_inputs"])
            except BaseException:  # noqa: BLE001 - a hanging or failing function does not fit
                outs = None
            finally:
                signal.alarm(0)
            if outs is not None and all(o == p["output"] for o, p in zip(outs, start["train"])):
                found.append({"name": name, "tests": outs[n_train:]})
                if equiv:
                    found[-1]["equiv"] = equivariance(fn, outs[n_train:], equiv)
        return found

    def sweep_check(name, source, budget):
        """Parameter sweep: one constant or comparison of function ``name`` (``source``: the cell
        that defined it) changed at a time; each variant that still reproduces every training pair, with its test
        outputs. Nothing is changed in the session."""
        import ast
        tree = ast.parse(source)
        node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if node is None:
            return {"tried": 0, "tests": []}
        colours = sorted({v for p in start["train"] for g in (p["input"], p["output"]) for row in g for v in row})
        swaps = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
        sites = []
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and type(n.value) is int and -1 <= n.value <= 30:
                for value in sorted({n.value - 1, n.value + 1, *colours} - {n.value}):
                    sites.append((n, "value", value))
            elif isinstance(n, ast.Compare):
                for i, op in enumerate(n.ops):
                    if type(op) in swaps:
                        sites.append((n.ops, i, swaps[type(op)]()))
        grids = [p["input"] for p in start["train"]] + start["test_inputs"]
        found, seen, began, tried = [], set(), time.monotonic(), 0
        try:
            for owner, key, value in sites:
                if time.monotonic() - began > budget:
                    break
                signal.alarm(1)  # a changed constant can hang a loop
                old = owner[key] if isinstance(owner, list) else getattr(owner, key)
                owner.__setitem__(key, value) if isinstance(owner, list) else setattr(owner, key, value)
                tried += 1
                try:
                    space = dict(namespace)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        exec(compile(ast.Module([node], []), "<sweep>", "exec"), space)
                        outs = run_fn(space[name], grids)
                except BaseException:  # noqa: BLE001 - a failing or hanging variant does not fit
                    outs = None
                finally:
                    signal.alarm(0)
                    owner.__setitem__(key, old) if isinstance(owner, list) else setattr(owner, key, old)
                n_train = len(start["train"])
                if outs is not None and all(o == p["output"] for o, p in zip(outs, start["train"])):
                    key_out = json.dumps(outs[n_train:])
                    if key_out not in seen:
                        seen.add(key_out)
                        found.append(outs[n_train:])
        finally:
            signal.alarm(0)
        return {"tried": tried, "tests": found}

    for line in sys.stdin:
        request = json.loads(line)
        if request.get("op") == "sweep":
            real.write(json.dumps({"sweep": sweep_check(request["name"], request["source"],
                                                        float(request.get("seconds") or 10))}) + "\n")
            real.flush()
            continue
        if request.get("op") == "fit":
            real.write(json.dumps({"fit": fit_check(float(request.get("seconds") or 10),
                                                    equiv=float(request.get("equiv") or 0))}) + "\n")
            real.flush()
            continue
        if request.get("op") == "aug":  # chk5: the last submitted function under grid transforms
            real.write(json.dumps({"aug": augment_check(last["fn"], request.get("transforms") or [],
                                                        float(request.get("seconds") or 20))}) + "\n")
            real.flush()
            continue
        if request.get("op") == "held":  # base16: pass or fail on the held-back pair, the output stays on the host
            real.write(json.dumps({"held": held_check(request["grid"]) if last["fn"] is not None else None}) + "\n")
            real.flush()
            continue
        if request.get("op") == "inventory":
            try:
                real.write(json.dumps({"inventory": inventory(), "playbook": memo["text"],
                                       "playbook_writes": memo["writes"]}) + "\n")
            except Exception as exc:  # noqa: BLE001
                real.write(json.dumps({"inventory": {"error": f"{type(exc).__name__}: {exc}"[:200]}}) + "\n")
            real.flush()
            continue
        printed = io.StringIO()
        out = {}
        submitted.clear()
        try:
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
                signal.alarm(seconds)
                if request["op"] == "define":  # a reply's program, callable as ``request["name"]``
                    scope = {"__builtins__": safe, "__name__": "__program__", "np": np}
                    exec(source(request["code"], "<program>"), scope, scope)
                    if callable(scope.get("transform")):
                        namespace[request["name"]] = fixed[request["name"]] = scope["transform"]
                else:
                    if start.get("sandbox_fix"):
                        namespace["result"] = None  # base20: `result` is always defined
                    else:
                        namespace.pop("result", None)
                    exec(source(request["code"], "<python>"), namespace, namespace)
                signal.alarm(0)
            if namespace.get("result") is not None and request["op"] == "run":
                out["result"] = clip(repr(plain(namespace["result"])), chars)
        except BaseException as exc:  # noqa: BLE001 - shown to the model
            signal.alarm(0)
            out["error"] = where(exc)
        if start.get("restore_names") and request["op"] == "run":
            replaced = restore()
            if replaced:
                out["restored"] = replaced
                print(RESTORED.format(names=", ".join(f"`{n}`" for n in replaced),
                                      verb="is" if len(replaced) == 1 else "are",
                                      was="was" if len(replaced) == 1 else "were"), file=printed)
        if submitted.get("guesses") is not None:
            out["submitted"] = submitted["guesses"]
            out["submit_check"], out["forced"] = submitted.get("check"), submitted.get("forced", False)
            out["submit_check2"] = submitted.get("check2")
        out["stdout"] = clip(printed.getvalue(), chars)
        real.write(json.dumps(out) + "\n")
        real.flush()


if __name__ == "__main__":
    main()
