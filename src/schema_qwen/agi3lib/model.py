"""Loading and running the agent-authored world model, sandboxed.

Two contracts are accepted, both recovered from the released traces:

    # stateful (most models)
    init_state(entry_grid) -> state
    predict(state, grid, action, x=None, y=None) -> (next_grid, info, next_state)

    # stateless (e.g. the sb26/su15/r11l claude runs)
    step(grid, action, x=None, y=None) -> (next_grid, info)

``info`` carries the flags ``level_up`` / ``dead`` / ``win``. The harness
injects three module globals before every call — ``CURRENT_LEVEL``,
``ENTRY_GRID`` (the current level's entry grid) and ``np`` — the complete
injection surface recovered by static analysis of all 50 released models
(no other undefined names appear in any of them).

The sandbox is a discipline boundary, not a security one: it keeps the model a
pure function of its arguments (no file or network IO, whitelisted imports
only). The builtin set matches what the traces reveal of the original — the
released models had to replace ``next()`` with a plain loop because it was
absent, so it is deliberately absent here too.
"""

ALLOWED_IMPORTS = {
    "math", "collections", "itertools", "functools", "heapq", "numpy",
}

_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "bytes", "callable", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "frozenset", "getattr", "hasattr", "hash",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "object", "ord", "pow", "print", "range", "repr", "reversed",
    "round", "set", "setattr", "slice", "sorted", "str", "sum", "tuple",
    "type", "zip",
    # exceptions the models legitimately raise/catch
    "ArithmeticError", "AssertionError", "AttributeError", "Exception",
    "IndexError", "KeyError", "LookupError", "NameError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
]


class ModelError(RuntimeError):
    """The world model raised, or returned something malformed."""


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(
            f"import of {name!r} is not allowed in the world model sandbox "
            f"(allowed: {sorted(ALLOWED_IMPORTS)})"
        )
    return __import__(name, globals, locals, fromlist, level)


def _sandbox_builtins():
    import builtins

    safe = {n: getattr(builtins, n) for n in _BUILTIN_NAMES}
    safe["__import__"] = _safe_import
    safe["__build_class__"] = builtins.__build_class__
    safe["True"] = True
    safe["False"] = False
    safe["None"] = None
    return safe


def _as_grid(obj):
    """Normalise a predicted grid (possibly numpy) to lists of ints."""
    if obj is None:
        return None
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    return [[int(v) for v in row] for row in obj]


class WorldModel:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, encoding="utf-8") as fh:
            source = fh.read()
        self.globals = {"__builtins__": _sandbox_builtins(),
                        "__name__": "world_model", "CURRENT_LEVEL": 0}
        # The original sandbox pre-injects numpy: released models use bare
        # `np.` with no import statement at all.
        try:
            import numpy

            self.globals["np"] = numpy
            self.globals["numpy"] = numpy
        except ImportError:
            pass
        try:
            exec(compile(source, self.path, "exec"), self.globals)
        except Exception as exc:  # surface load failures as model errors
            raise ModelError(f"world model failed to load: {exc!r}") from exc
        self.stateless = "step" in self.globals
        if not self.stateless:
            for name in ("init_state", "predict"):
                if name not in self.globals:
                    raise ModelError(
                        f"world model defines no {name}() (and no step())")

    def init_state(self, entry_grid, level=0):
        self.globals["CURRENT_LEVEL"] = level
        self.globals["ENTRY_GRID"] = entry_grid
        if self.stateless:
            return {}
        try:
            return self.globals["init_state"](entry_grid)
        except Exception as exc:
            raise ModelError(f"init_state raised: {exc!r}") from exc

    def predict(self, state, grid, action, x=None, y=None, level=0,
                entry_grid=None):
        """Returns (next_grid, info, next_state); info flags forced to bool."""
        self.globals["CURRENT_LEVEL"] = level
        if entry_grid is not None:
            self.globals["ENTRY_GRID"] = entry_grid
        fn = "step" if self.stateless else "predict"
        try:
            out = (self.globals["step"](grid, action, x, y)
                   if self.stateless
                   else self.globals["predict"](state, grid, action, x, y))
        except Exception as exc:
            raise ModelError(f"{fn} raised: {exc!r}") from exc
        try:
            if self.stateless:
                next_grid, info = out
                next_state = state
            else:
                next_grid, info, next_state = out
        except Exception as exc:
            raise ModelError(
                f"{fn} returned a malformed result: {type(out).__name__}"
            ) from exc
        info = info or {}
        flags = {k: bool(info.get(k)) for k in ("level_up", "dead", "win")}
        return _as_grid(next_grid), flags, next_state


def find_model(workdir, explicit=None):
    """The model file in play: explicit path, else newest world_model*.py."""
    import glob
    import os

    if explicit:
        return explicit if os.path.exists(explicit) else None
    candidates = sorted(
        glob.glob(os.path.join(str(workdir), "world_model*.py")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None
