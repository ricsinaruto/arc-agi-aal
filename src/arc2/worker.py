"""Child process of ``arc2.sandbox``: runs one model-written ``transform`` on a list of grids.

Standalone (run by path with ``python -I``): a JSON request on stdin, one JSON line on the saved stdout. Limits, in the
order applied: CPU and address-space rlimits, no file writes, an unprivileged uid when started as root, an import
allow-list and builtins without ``open``. The same kind of guard as the Duck python tool (a restricted interpreter, not
a kernel sandbox); the run's VM has no network and is deleted after the run.
"""

import ast
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
           "enum", "abc", "queue", "array", "numbers", "time"}
STDOUT_LIMIT = 4000
EXTRA = {"sys", "traceback", "inspect", "pprint"}  # setting tool_fixes (C1): allowed as well
PRELOADED = ("np", "train", "test_inputs")  # tool_fixes: `import np` and the like bind what is already there
SHOW_CALLS, SHOW_GRIDS = 4, 4  # pictures a `python` call can show, grids a picture


def limit(seconds, strict=True):
    """``strict`` adds the address-space limit and the uid drop; the trial falls back to the rest (CPU time, no file
    writes, the timeouts) when its self-test fails with them."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        if strict:
            resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    except (ImportError, ValueError, OSError):  # macOS refuses RLIMIT_AS; the host's wall timeout still applies
        pass
    if strict and hasattr(os, "geteuid") and os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)


def plain(value):
    """numpy arrays and scalars as lists and ints; anything else unchanged (the host validates)."""
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
    """All of a short text; else its start and its end around a marker (arc2.tasks.clip)."""
    if not limit or len(text) <= limit:
        return text[:STDOUT_LIMIT] if not limit else text
    head = limit * 2 // 3
    return f"{text[:head]}\n... [{len(text) - limit} characters not shown] ...\n{text[len(text) - (limit - head):]}"


TRACE_EVENTS, TRACE_VISITS, TRACE_CHARS = 2000, 3, 3000


def summary(value):
    """A short description of a traced variable: arrays and grids by shape and colours, containers by length."""
    try:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            shape = "x".join(map(str, value.shape))
            if value.dtype == bool:
                return f"bool array {shape}, {int(value.sum())} True"
            if value.size and value.size <= 900 and value.dtype.kind in "iu":
                return f"array {shape}, values {sorted(set(value.ravel().tolist()))[:10]}"
            return f"array {shape} {value.dtype}"
        if isinstance(value, list) and value and all(isinstance(r, list) for r in value) and \
                all(isinstance(c, int) for r in value for c in r):
            return f"grid {len(value)}x{len(value[0])}, colours {sorted({c for r in value for c in r})[:10]}"
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            head = repr(value)
            return head if len(head) <= 60 else f"{type(value).__name__} of {len(value)}: {head[:50]}..."
        if isinstance(value, (int, float, str, bool)) or value is None:
            return repr(value)[:60]
        return type(value).__name__
    except Exception:  # noqa: BLE001 - a description never breaks the run
        return type(value).__name__


def traced(transform, grid):
    """``trace`` (setting, ABPR-style debugging): the program's lines on one grid in order, each with the variables
    it set or changed, a line shown at most TRACE_VISITS times (later visits counted), cut to TRACE_CHARS."""
    events, seen, visits = [], {}, {}

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "<program>":
            return None
        if event in ("line", "return") and len(events) < TRACE_EVENTS:
            key = id(frame)
            now = {k: summary(v) for k, v in frame.f_locals.items() if not k.startswith("__") and not callable(v)}
            before = seen.get(key, {})
            changed = {k: v for k, v in now.items() if before.get(k) != v}
            seen[key] = now
            if changed and frame.f_lineno is not None:
                events.append((frame.f_code.co_name, prev.get(key, frame.f_lineno), changed))
            prev[key] = frame.f_lineno
        return tracer

    prev = {}
    error = None
    sys.settrace(tracer)
    try:
        transform(copy.deepcopy(grid))
    except Exception as exc:  # noqa: BLE001 - the trace ends at the error
        error = where(exc)
    finally:
        sys.settrace(None)
    lines, hidden = [], 0
    for name, line, changed in events:
        visits[(name, line)] = visits.get((name, line), 0) + 1
        if visits[(name, line)] > TRACE_VISITS:
            hidden += 1
            continue
        lines.append(f"{name} line {line}: " + "; ".join(f"{k} = {v}" for k, v in changed.items()))
    if hidden:
        lines.append(f"({hidden} later steps on lines already shown are left out)")
    if error:
        lines.append(f"stopped: {error}")
    return clip("\n".join(lines), TRACE_CHARS)


def where(exc):
    lines = [f.lineno for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<program>"]
    return f"{type(exc).__name__}: {str(exc)[:300]}" + (f" (line {lines[-1]})" if lines else "")


def main():
    request = json.loads(sys.stdin.read())
    real = os.fdopen(os.dup(1), "w")
    import numpy as np  # before the uid drop and the import allow-list

    kit = {}
    if request.get("helpers"):  # arckit's grid helpers (setting ``helpers``), loaded by path before the limits
        kit = {"__name__": "arckit"}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arckit.py")
        with builtins.open(path) as source:
            exec(compile(source.read(), path, "exec"), kit)
        kit = {name: kit[name] for name in kit["NAMES"]}

    limit(int(request["cpu_seconds"]), request.get("strict", True))
    fixes, chars = bool(request.get("fixes")), int(request.get("chars") or 0)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if fixes and not level and name in PRELOADED and name in namespace:
            return namespace[name]
        if level or (name.split(".")[0] not in ALLOWED and not (fixes and name.split(".")[0] in EXTRA)):
            raise ImportError(f"module not available here: {name} (available: numpy, scipy and the pure standard library)")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in vars(builtins).items() if k not in ("open", "input", "breakpoint", "exit", "quit")}
    safe["__import__"] = safe_import
    namespace = {"__builtins__": safe, "__name__": "__program__", "np": np, **kit}
    printed = io.StringIO()
    result = {"outputs": []}

    def alarm(signum, frame):
        raise TimeoutError(f"more than {request['grid_seconds']} s on one {'snippet' if 'cells' in request else 'grid'}")

    signal.signal(signal.SIGALRM, alarm)
    if "cells" in request:  # the tool arm's persistent session, rebuilt: earlier snippets again, then this one
        namespace.update(train=copy.deepcopy(request["train"]), test_inputs=copy.deepcopy(request["test_inputs"]))
        shown, kept = [], [""]
        if request.get("show"):  # setting ``show``: show(*grids) puts grids side by side in a picture (this call's only)
            def show(*grids):
                if len(shown) < SHOW_CALLS and grids:
                    shown.append([plain(g) for g in grids[:SHOW_GRIDS]])
            namespace["show"] = lambda *grids: None
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for cell in request["cells"]:
                    signal.alarm(int(request["grid_seconds"]))
                    exec(compile(cell, "<earlier>", "exec"), namespace, namespace)
                    signal.alarm(0)
            namespace.pop("result", None)
            if request.get("show"):
                namespace["show"] = show
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
                signal.alarm(int(request["grid_seconds"]))
                if fixes:  # statement by statement: a failure keeps what ran before it (``kept``)
                    tree, source = ast.parse(request["code"], "<program>"), request["code"].splitlines(keepends=True)
                    for stmt in tree.body:
                        first = min([stmt.lineno] + [d.lineno for d in getattr(stmt, "decorator_list", [])])
                        kept[0] = "".join(source[:first - 1]).strip()
                        exec(compile(ast.Module([stmt], []), "<program>", "exec"), namespace, namespace)
                    kept[0] = ""
                else:
                    exec(compile(request["code"], "<program>", "exec"), namespace, namespace)
                signal.alarm(0)
            result = {"shown": shown} if shown else {}
            if namespace.get("result") is not None:
                result["result"] = clip(repr(namespace["result"]), chars)
        except BaseException as exc:  # noqa: BLE001 - shown to the model
            signal.alarm(0)
            result = {"error": where(exc)}
            if kept[0]:
                result["kept"] = kept[0]
        result["stdout"] = clip(printed.getvalue(), chars)
        real.write(json.dumps(result) + "\n")
        real.flush()
        return
    try:
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
            signal.alarm(int(request["grid_seconds"]))
            exec(compile(request["code"], "<program>", "exec"), namespace, namespace)
            signal.alarm(0)
            transform = namespace.get("transform")
            if not callable(transform):
                raise NameError("the program defines no function transform(grid)")
            for grid in request["inputs"]:
                try:
                    signal.alarm(int(request["grid_seconds"]))
                    out = plain(transform(copy.deepcopy(grid)))
                    signal.alarm(0)
                except Exception as exc:  # noqa: BLE001 - reported to the model per grid
                    signal.alarm(0)
                    out = {"error": where(exc)}
                result["outputs"].append(out)
            if request.get("trace") is not None:
                signal.alarm(int(request["grid_seconds"]) * 2)
                result["trace"] = traced(transform, request["inputs"][int(request["trace"])])
                signal.alarm(0)
    except BaseException as exc:  # noqa: BLE001 - a program that does not load, or exits
        signal.alarm(0)
        result = {"error": where(exc)}
    result["stdout"] = printed.getvalue()[:STDOUT_LIMIT]
    real.write(json.dumps(result) + "\n")
    real.flush()


if __name__ == "__main__":
    main()
