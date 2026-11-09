"""The sandbox child program (``_SANDBOX_BOOTSTRAP``): one persistent interpreter per game, driven by the host's
``duck_transfer.sandbox.StatefulSandbox`` over JSON lines. Each ``code`` request refreshes the runtime variables
(delta history: the host sends only entries the child has not seen), seeds ``random`` per call and runs the snippet
with user definitions kept; an ``inventory`` request lists them. A call past its timeout is interrupted (SIGUSR1
raises ``ToolTimeout``); ``time``, ``pickle`` and ``inspect`` are restricted shims; transitions keep every
action's result and are built incrementally. The arena's dfr_harness and hx_harness patch pinned lines of this text."""
import inspect

from inference.utils import segmentation as _segmentation

_SANDBOX_BOOTSTRAP = r'''
import builtins
import contextlib
import io
import json
import os
import sys
import traceback

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

COLOR_CHARS = ""

__SEGMENTATION_SOURCE__

_segment_layer_uncached = segment_layer
_SEGMENTATION_CACHE = {}
_SEGMENTATION_STATS = {"hits": 0, "misses": 0}


def segment_layer(layer, color_chars):
    key = (tuple(tuple(row) for row in layer), str(color_chars))
    cached = _SEGMENTATION_CACHE.get(key)
    if cached is not None:
        _SEGMENTATION_STATS["hits"] += 1
        return cached
    _SEGMENTATION_STATS["misses"] += 1
    if len(_SEGMENTATION_CACHE) >= 64:
        del _SEGMENTATION_CACHE[next(iter(_SEGMENTATION_CACHE))]
    cached = _SEGMENTATION_CACHE[key] = _segment_layer_uncached(layer, color_chars)
    return cached

HOST_STDOUT = sys.stdout

SAFE_MODULES = {
    "bisect",
    "collections",
    "copy",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
}
SAFE_BUILTINS = {
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "Exception",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "hasattr",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "TypeError",
    "type",
    "ValueError",
    "RuntimeError",
    "zip",
}


def _send(payload):
    HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    HOST_STDOUT.flush()


def _recv():
    line = sys.stdin.readline()
    if not line:
        raise EOFError("sandbox input closed")
    return json.loads(line)


class FrameView:
    def __init__(self, *, ascii, step, level, shape, grid):
        self.ascii = ascii
        self.step = step
        self.level = level
        self.shape = tuple(shape)
        self._grid = grid
        self._segmentation = None

    @property
    def segmentation(self):
        if self._segmentation is None:
            self._segmentation = segment_layer(self._grid, COLOR_CHARS)
        return self._segmentation

    def __str__(self):
        rows, cols = self.shape
        return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

    __repr__ = __str__


class HistoryEntryView:
    def __init__(self, *, action, frame):
        self.action = action
        self.frame = frame

    def __str__(self):
        return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

    __repr__ = __str__


class TransitionView:
    def __init__(self, *, action, before_frame, after_frame, result):
        self.action = action
        self.before_frame = before_frame
        self.after_frame = after_frame
        self.frame = after_frame
        self.result = dict(result) if isinstance(result, dict) else {}

    def __str__(self):
        return (
            "ActionTransitionView("
            f"action={self.action!r}, "
            f"before_frame={self.before_frame}, "
            f"after_frame={self.after_frame})"
        )

    __repr__ = __str__


def _frame_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    return FrameView(
        ascii=str(payload.get("ascii", "")),
        step=int(payload.get("step", 0)),
        level=int(payload.get("level", 0)),
        shape=payload.get("shape", [0, 0]),
        grid=payload.get("grid", []),
    )


def _history_from_payload(payload):
    items = []
    for entry in payload or []:
        if not isinstance(entry, dict):
            continue
        items.append(
            HistoryEntryView(
                action=str(entry.get("action", "")),
                frame=_frame_from_payload(entry.get("frame")),
            )
        )
        items[-1].result = dict(entry.get('result') or {})
    return items


_TV_CACHE = {"first": None, "last": None, "n": 0, "items": [], "src": [], "hist": {}}
_TV_TRACK = {}     # id(cached view) -> its index in the cache
_TV_DIRTY = set()  # cache indices to rebuild; -1: a history entry changed, rebuild all


class _TrackedResult(dict):
    """A cached view's ``result``: a mutation marks the view for a rebuild at the next refresh."""

    __slots__ = ("_tv_index",)

    def _tv_mark(self):
        _TV_DIRTY.add(getattr(self, "_tv_index", -1))

    def __setitem__(self, key, value):
        self._tv_mark()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._tv_mark()
        dict.__delitem__(self, key)

    def __ior__(self, other):
        self._tv_mark()
        return dict.__ior__(self, other)

    def pop(self, *args):
        self._tv_mark()
        return dict.pop(self, *args)

    def popitem(self):
        self._tv_mark()
        return dict.popitem(self)

    def clear(self):
        self._tv_mark()
        dict.clear(self)

    def update(self, *args, **kwargs):
        self._tv_mark()
        dict.update(self, *args, **kwargs)

    def setdefault(self, key, default=None):
        self._tv_mark()
        return dict.setdefault(self, key, default)


def _tv_setattr(self, name, value):
    index = _TV_TRACK.get(id(self))
    if index is not None:
        _TV_DIRTY.add(index)
    object.__setattr__(self, name, value)


def _tv_delattr(self, name):
    index = _TV_TRACK.get(id(self))
    if index is not None:
        _TV_DIRTY.add(index)
    object.__delattr__(self, name)


def _he_setattr(self, name, value):
    if id(self) in _TV_CACHE["hist"]:
        _TV_DIRTY.add(-1)
    object.__setattr__(self, name, value)


def _he_delattr(self, name):
    if id(self) in _TV_CACHE["hist"]:
        _TV_DIRTY.add(-1)
    object.__delattr__(self, name)


TransitionView.__setattr__, TransitionView.__delattr__ = _tv_setattr, _tv_delattr
HistoryEntryView.__setattr__, HistoryEntryView.__delattr__ = _he_setattr, _he_delattr


def _tv_build(index, history, position):
    entry = history[position]
    view = TransitionView(action=str(getattr(entry, "action", "") or "").strip(),
                          before_frame=history[position - 1].frame if position > 0 else None,
                          after_frame=entry.frame, result={})
    result = _TrackedResult(dict(getattr(entry, "result", {})))
    result._tv_index = index
    object.__setattr__(view, "result", result)
    _TV_TRACK[id(view)] = index
    return view


def _tv_reset(history):
    cache = _TV_CACHE
    cache.update(first=history[0] if history else None, last=None, n=0, items=[], src=[], hist={})
    _TV_TRACK.clear()
    _TV_DIRTY.clear()


def _transitions_incremental(history, last_action_result):
    """``transitions`` of ``history`` (each action's recorded result), the views of the retained history kept between
    refreshes; ``last_action_result`` fills the last view's empty result."""
    cache = _TV_CACHE
    count = len(history)
    if (not history or cache["first"] is not history[0] or cache["n"] > count or -1 in _TV_DIRTY
            or (cache["n"] and history[cache["n"] - 1] is not cache["last"])):
        _tv_reset(history)
    items, src = cache["items"], cache["src"]
    for index in sorted(_TV_DIRTY):
        if 0 <= index < len(items):
            _TV_TRACK.pop(id(items[index]), None)
            items[index] = _tv_build(index, history, src[index])
    _TV_DIRTY.clear()
    for position in range(cache["n"], count):
        entry = history[position]
        cache["hist"][id(entry)] = position
        if str(getattr(entry, "action", "") or "").strip():
            src.append(position)
            items.append(_tv_build(len(items), history, position))
    cache["n"], cache["last"] = count, history[-1] if history else None
    transitions = list(items)
    if transitions and isinstance(last_action_result, dict) and not transitions[-1].result:
        last = transitions[-1]  # a fresh view, as before: the override does not stick to the cached one
        transitions[-1] = TransitionView(action=last.action, before_frame=last.before_frame,
                                         after_frame=last.after_frame, result=dict(last_action_result))
    return transitions


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sanitize_exception(exc):
    extracted = traceback.extract_tb(exc.__traceback__)
    user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
    lines = ["Traceback (most recent call last):"]
    for frame in user_frames or extracted[-1:]:
        lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
    lines.append(f"{exc.__class__.__name__}: {exc}")
    return "\n".join(lines)


import inspect as _inspect_real
import reprlib as _reprlib
import signal as _signal
import types as _types


class ToolTimeout(BaseException):
    """Raised in the running statement when the host's tool timeout expires."""


_INTERRUPT = {"armed": False, "pending": False, "seconds": 30}


def _on_interrupt(signum, frame):
    if _INTERRUPT["armed"]:
        raise ToolTimeout(
            "the python call ran past the %ss tool timeout and was interrupted here" % _INTERRUPT["seconds"])
    _INTERRUPT["pending"] = True


_signal.signal(_signal.SIGUSR1, _on_interrupt)
_SOURCES = {}


def _register_source(code, text):
    stack = [code]
    while stack:
        item = stack.pop()
        _SOURCES[item] = text
        stack.extend(c for c in item.co_consts if isinstance(c, _types.CodeType))
    while len(_SOURCES) > 4096:
        del _SOURCES[next(iter(_SOURCES))]


def _getsourcelines(obj):
    obj = getattr(obj, "__func__", obj)
    obj = getattr(obj, "__wrapped__", obj)
    code = getattr(obj, "__code__", None)
    text = _SOURCES.get(code) if code is not None else None
    if text is None:
        raise OSError("source is available only for functions defined in this python session")
    lines = text.splitlines(True)
    return _inspect_real.getblock(lines[code.co_firstlineno - 1:]), code.co_firstlineno


def _getsource(obj):
    return "".join(_getsourcelines(obj)[0])


import pickle as _pickle_real
import time as _time_real


def _shim(name, doc, **attrs):
    module = _types.ModuleType(name, doc)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _no_sleep(*args, **kwargs):
    raise RuntimeError("time.sleep is disabled in the sandbox: the game advances only when you call action(...)")


_PICKLE_CLASSES = {("builtins", n) for n in ("set", "frozenset", "complex", "bytearray", "slice", "range")} | {
    ("collections", n) for n in ("deque", "Counter", "OrderedDict", "defaultdict")}


class _SafeUnpickler(_pickle_real.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _PICKLE_CLASSES:
            return super().find_class(module, name)
        raise _pickle_real.UnpicklingError("pickle.loads may not load %s.%s in the sandbox" % (module, name))


def _pickle_loads(data, **kwargs):
    return _SafeUnpickler(io.BytesIO(data), **kwargs).load()


SHIM_MODULES = {
    "time": _shim("time", "Clocks only (no sleep).", **{n: getattr(_time_real, n) for n in (
        "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "process_time",
        "process_time_ns")}, sleep=_no_sleep),
    "pickle": _shim("pickle", "dumps/loads of plain data.", dumps=_pickle_real.dumps, loads=_pickle_loads,
                    HIGHEST_PROTOCOL=_pickle_real.HIGHEST_PROTOCOL, DEFAULT_PROTOCOL=_pickle_real.DEFAULT_PROTOCOL,
                    PickleError=_pickle_real.PickleError, PicklingError=_pickle_real.PicklingError,
                    UnpicklingError=_pickle_real.UnpicklingError),
    "inspect": _shim("inspect", "Signatures and the source of functions defined in this session.",
                     getsource=_getsource, getsourcelines=_getsourcelines,
                     **{n: getattr(_inspect_real, n) for n in (
                         "signature", "Signature", "Parameter", "getdoc", "cleandoc", "getfullargspec",
                         "getmembers", "isfunction", "isclass", "ismethod", "isroutine", "isbuiltin",
                         "isgenerator", "isgeneratorfunction")}),
}
SAFE_MODULES = SAFE_MODULES | {"hashlib", "textwrap", "traceback"}
SAFE_BUILTINS = SAFE_BUILTINS | {"globals", "id", "locals", "vars"}
_NOT_INSTALLED = {"numpy", "scipy", "pandas", "sklearn", "torch", "PIL", "cv2", "networkx", "sympy"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name or "").split(".", 1)[0]
    if root in SHIM_MODULES and root == name:
        return SHIM_MODULES[root]
    if root not in SAFE_MODULES:
        why = ("is not installed in the sandbox (standard library only)" if root in _NOT_INSTALLED
               else "is not allowed in the sandbox")
        raise ImportError("Module '%s' %s. Importable modules: %s." % (
            name, why, ", ".join(sorted(SAFE_MODULES | set(SHIM_MODULES)))))
    return builtins.__import__(name, globals, locals, fromlist, level)


def _set_limits(timeout_seconds):
    if resource is None:
        return
    cpu_limit = max(1, int(timeout_seconds)) + 1
    for limit, value in (
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
        (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
        (getattr(resource, "RLIMIT_NOFILE", None), 32),
    ):
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (OSError, ValueError):
            pass


def _normalize_actions(actions):
    if isinstance(actions, str):
        items = [actions]
    elif isinstance(actions, dict):
        items = [actions]
    elif isinstance(actions, (list, tuple)):
        items = list(actions)
    else:
        raise TypeError(
            "action(actions) expects a string, an action object, or a list of action strings/objects."
        )
    if not items:
        raise ValueError("action(actions) requires at least one action.")

    normalized = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            action_name = item.strip()
            if not action_name:
                raise ValueError(f"Action {index} is empty.")
            normalized.append({"action": action_name})
            continue
        if isinstance(item, dict):
            action_name = str(item.get("action", "")).strip()
            if not action_name:
                raise ValueError(f"Action {index} is missing an `action` field.")
            entry = {"action": action_name}
            if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                raise ValueError(
                    f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                )
            if "row" in item:
                entry["row"] = item.get("row")
            if "col" in item:
                entry["col"] = item.get("col")
            normalized.append(entry)
            continue
        raise TypeError(f"Action {index} must be a string or a dict.")
    return normalized


def main():
    initial = _recv()
    global COLOR_CHARS
    COLOR_CHARS = str(initial.get("color_chars") or "")
    timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
    sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
    if sandbox_cwd:
        os.chdir(sandbox_cwd)
    _set_limits(int(initial.get("cpu_seconds", 1200)))

    action_results = []
    retained_history = []
    stdout = io.StringIO()
    runtime_globals = {
        "__builtins__": {
            name: getattr(builtins, name)
            for name in SAFE_BUILTINS
        },
        "result": None,
    }
    runtime_globals["__builtins__"]["__import__"] = _safe_import

    def _refresh_state(state_payload):
        current_frame = _frame_from_payload(state_payload.get("current_frame"))
        incoming = _history_from_payload(state_payload.get("history"))
        base = state_payload.get("history_base")
        if not base:
            retained_history[:] = incoming
        elif int(base) == len(retained_history):
            retained_history.extend(incoming)
        else:
            raise RuntimeError("history sync mismatch: %s held, base %s" % (len(retained_history), base))
        history = list(retained_history)
        last_action_result = state_payload.get("last_action_result")
        action_result = (
            dict(last_action_result) if isinstance(last_action_result, dict) else {}
        )
        transitions = _transitions_incremental(history, action_result)
        last_transition = transitions[-1] if transitions else None

        runtime_globals["current_frame"] = current_frame
        runtime_globals["latest_frame"] = current_frame
        runtime_globals["history"] = history
        runtime_globals["transitions"] = transitions
        runtime_globals["last_transition"] = last_transition
        runtime_globals["previous_frame"] = (
            last_transition.before_frame if last_transition is not None else None
        )
        runtime_globals["last_action_frame"] = (
            last_transition.after_frame if last_transition is not None else None
        )
        runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
        runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
        runtime_globals["last_action_result"] = action_result

    def action(actions):
        normalized_actions = _normalize_actions(actions)
        _send({"type": "action", "actions": normalized_actions})
        reply = _recv()
        if reply.get("type") == "action_error":
            raise RuntimeError(str(reply.get("error", "action failed")))
        if reply.get("type") != "action_result":
            raise RuntimeError("Invalid action response from sandbox host.")
        action_result = reply.get("action_result") or {}
        action_results.append(action_result)
        _refresh_state(reply.get("state") or {})
        return action_result

    runtime_globals["action"] = action

    _raw_action = action

    def action(actions):
        armed = _INTERRUPT["armed"]
        _INTERRUPT["armed"] = False
        try:
            return _raw_action(actions)
        finally:
            _INTERRUPT["armed"] = armed
            if armed and _INTERRUPT["pending"]:
                _INTERRUPT["pending"] = False
                raise ToolTimeout("the python call ran past the %ss tool timeout; interrupted after this action"
                                  % _INTERRUPT["seconds"])

    runtime_globals["action"] = action
    _INTERRUPT["seconds"] = timeout_seconds
    _RUNTIME_NAMES = {"current_frame", "latest_frame", "history", "transitions", "last_transition", "previous_frame",
                      "last_action_frame", "last_action", "valid_actions", "last_action_result", "action", "result"}
    _short = _reprlib.Repr()
    _short.maxstring = _short.maxother = 48
    _short.maxlist = _short.maxtuple = _short.maxset = _short.maxfrozenset = _short.maxdict = 4
    _short.maxdeque = _short.maxarray = 4
    _short.maxlevel = 2

    def _inventory(limit=120):
        functions, classes, modules, values = [], [], [], []
        for name, value in list(runtime_globals.items()):
            if name in _RUNTIME_NAMES or name.startswith("__"):
                continue
            try:
                function = getattr(value, "__wrapped__", value)  # functools.lru_cache/wraps around a helper
                own = isinstance(function, _types.FunctionType) and function.__code__.co_filename == "<python_tool>"
                if own:
                    try:
                        signature = str(_inspect_real.signature(function))
                    except (TypeError, ValueError):
                        signature = "(...)"
                    doc = (function.__doc__ or "").strip().split("\n", 1)[0]
                    functions.append([name, signature, doc])
                elif isinstance(value, type):
                    classes.append(name)
                elif isinstance(value, _types.ModuleType) or callable(value):
                    modules.append(name)  # imported modules and names (e.g. ``from collections import deque``)
                else:
                    try:
                        size = len(value)
                    except Exception:
                        size = None
                    values.append([name, type(value).__name__, size, _short.repr(value)[:80]])
            except Exception:
                values.append([name, "?", None, "<unavailable>"])
        return {"functions": functions[-limit:], "classes": classes[-limit:], "modules": modules[-limit:],
                "values": values[-limit:],
                "counts": {"functions": len(functions), "classes": len(classes), "values": len(values)}}

    request = initial
    while True:
        if request.get("type") == "inventory":
            _INTERRUPT["pending"] = False
            _INTERRUPT["armed"] = True
            try:
                payload = _inventory()
            except BaseException as exc:
                payload = {"error": "%s: %s" % (exc.__class__.__name__, exc)}
            finally:
                _INTERRUPT["armed"] = False
            _send({"type": "inventory", **payload})
        else:
            del action_results[:]
            stdout = io.StringIO()
            runtime_globals["result"] = None
            _refresh_state(request.get("state") or {})
            __import__("random").seed(request.get("rng_seed"))
            try:
                code = str(request.get("code", ""))
                compiled = compile(code, "<python_tool>", "exec")
                _register_source(compiled, code)
                _INTERRUPT["pending"] = False
                _INTERRUPT["armed"] = True
                try:
                    with contextlib.redirect_stdout(stdout):
                        exec(compiled, runtime_globals, runtime_globals)
                finally:
                    _INTERRUPT["armed"] = False
                _send(
                    {
                        "type": "final",
                        "stdout": stdout.getvalue(),
                        "result": _json_safe(runtime_globals.get("result")),
                        "action_results": _json_safe(action_results),
                        "segmentation": dict(_SEGMENTATION_STATS),
                        "history_size": len(retained_history),
                    }
                )
            except (Exception, ToolTimeout) as exc:
                _send(
                    {
                        "type": "error",
                        "error": _sanitize_exception(exc),
                        "stdout": stdout.getvalue(),
                        "action_results": _json_safe(action_results),
                        "interrupted": isinstance(exc, ToolTimeout),
                        "history_size": len(retained_history),
                    }
                )
        try:
            request = _recv()
        except EOFError:
            break
        if request.get("type") not in ("code", "inventory"):
            break


if __name__ == "__main__":
    main()
'''.replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation))
