"""Persistent Python subprocess. This file can run without importing the agent package."""

import ast
import builtins
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import traceback
from collections.abc import Sequence
from io import TextIOBase
from pathlib import Path

import numpy as np

ALLOWED = {
    "numpy",
    "math",
    "collections",
    "itertools",
    "functools",
    "heapq",
    "bisect",
    "statistics",
    "random",
    "re",
    "json",
    "operator",
    "fractions",
    "decimal",
}


def immutable(value):
    if isinstance(value, dict):
        return {
            k: np.frombuffer(np.array(v, dtype=np.uint8).tobytes(), dtype=np.uint8).reshape(64, 64)
            if k == "grid"
            else tuple(
                np.frombuffer(np.array(f, dtype=np.uint8).tobytes(), dtype=np.uint8).reshape(64, 64)
                for f in v
            )
            if k == "frames" and v and isinstance(v[0], list)
            else immutable(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [immutable(v) for v in value]
    return value


class RemoteSequence(Sequence):
    def __init__(self, name, rpc):
        self.name, self.rpc = name, rpc

    def __len__(self):
        return self.rpc("length", name=self.name)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not isinstance(index, int):
            raise TypeError("Integer index required")
        return immutable(self.rpc("item", name=self.name, index=index))


def main():
    # Load trusted modules before applying filesystem restrictions.
    sandbox_path = os.environ.pop("ARC_SANDBOX_MODULE")
    spec = importlib.util.spec_from_file_location("sandbox", sandbox_path)
    sandbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandbox)
    for module in ALLOWED:
        importlib.import_module(module)
    # Duck's segmentation helper, staged by the parent (arena.serving_core.segmentation).
    helper = {"__name__": "segmentation", "__builtins__": builtins}
    exec(
        compile(
            Path(os.environ.pop("ARC_SEGMENTATION_SOURCE")).read_text(), "segmentation.py", "exec"
        ),
        helper,
    )
    source_in, source_out = sys.stdin, sys.stdout

    def send(value):
        source_out.write(json.dumps(value) + "\n")
        source_out.flush()

    def rpc(method, **args):
        send({"type": "rpc", "method": method, **args})
        result = json.loads(source_in.readline())
        if "error" in result:
            if result.get("error_type") == "IndexError":
                raise IndexError(result["error"])
            raise RuntimeError(result["error"])
        return result["result"]

    class Output(TextIOBase):
        def write(self, value):
            for start in range(0, len(value), 16384):
                send({"type": "output", "text": value[start : start + 16384]})
            return len(value)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] not in ALLOWED or level:
            raise ImportError(f"Module unavailable: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins = {
        k: v for k, v in vars(builtins).items() if k not in {"open", "input", "breakpoint"}
    }
    safe_builtins["__import__"] = safe_import
    namespace = {"__builtins__": safe_builtins, "__name__": "__game__"}
    bindings = {
        "np": np,
        "observe": lambda: immutable(rpc("observe")),
        "step": lambda action, x=None, y=None: immutable(rpc("step", action=action, x=x, y=y)),
        "segment": helper["segment"],
        "changes": lambda: immutable(rpc("transition")),
        "previous": lambda: immutable(rpc("observe", index=-2)),
        **{name: RemoteSequence(name, rpc) for name in ("history", "transitions", "transcript")},
    }
    isolation = sandbox.restrict(read=sandbox.runtime_paths())
    send({"type": "ready", "sandbox": isolation})
    for line in source_in:
        request = json.loads(line)
        if request["type"] == "close":
            break
        namespace.update(bindings)
        namespace["current_frame"] = bindings["observe"]()
        namespace["grid"] = namespace["current_frame"]["grid"]
        error = None
        try:
            tree = ast.parse(request["code"])
            with contextlib.redirect_stdout(Output()), contextlib.redirect_stderr(Output()):
                exec(compile(tree, "<python-tool>", "exec"), namespace, namespace)
        except BaseException:
            error = traceback.format_exc()
            send({"type": "output", "text": error})
        send({"type": "done", "error": error})


if __name__ == "__main__":
    main()
