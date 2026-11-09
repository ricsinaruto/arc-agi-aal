"""Persistent Python kernel of the ``pi-rlm`` arena kind (one per game).

Runs under ``python -I`` without importing the agent package, inside the same Landlock /
namespace / chroot sandbox as the B-baseline worker. The model's code sees ``env`` (the game:
every action is an RPC to the host, which steps the serialized bridge), ``np`` and ``skills``
(the shared harness procedures, exec'd from sources the host sends). Observations stay in the
kernel (``env.history()``); the model context receives only what its code prints.
"""

import ast
import base64
import builtins
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import traceback
import types
from io import TextIOBase

import numpy as np

ALLOWED = {
    "numpy", "math", "collections", "itertools", "functools", "heapq", "bisect", "statistics",
    "random", "re", "json", "operator", "fractions", "decimal", "copy", "string", "dataclasses",
    "typing", "enum", "abc", "array", "struct", "hashlib", "textwrap", "pprint", "time",
}
HEX = "0123456789ABCDEF"


def decode_frame(text):
    frame = np.frombuffer(base64.b64decode(text), dtype=np.uint8).reshape(64, 64).copy()
    frame.setflags(write=False)
    return frame


def decode(record):
    """The host's observation record with its frames as read-only 64x64 uint8 arrays."""
    frames = tuple(decode_frame(f) for f in record.get("frames", []))
    return {**{k: v for k, v in record.items() if k != "frames"}, "frames": frames,
            "frame": frames[-1] if frames else None}


def ascii_board(frame):
    return "\n".join("".join(HEX[int(c)] for c in row) for row in np.asarray(frame))


class Env:
    """The game as an object. Every action goes through the host's serialized bridge."""

    def __init__(self, rpc):
        self._rpc = rpc
        self._history = [decode(o) for o in rpc("history")]

    def _apply(self, record):
        observation = decode(record)
        self._history.append(observation)
        return observation

    def step(self, action, x=None, y=None):
        """Execute one action (a name from legal_actions(); ACTION6 needs x=column, y=row)
        and return the resulting observation record (dict; 'frame' is the settled board)."""
        return self._apply(self._rpc("step", action=str(action), x=x, y=y))

    def reset(self):
        """RESET the current level (counts as an action)."""
        return self.step("RESET")

    def last(self):
        """The latest observation record."""
        return self._history[-1]

    def frame(self):
        """The current settled board: read-only 64x64 uint8 array, frame[y, x] in 0..15."""
        return self._history[-1]["frame"]

    def ascii(self, frame=None):
        """The board (current by default) as 64 lines of hex digits 0-F, one per row."""
        return ascii_board(self.frame() if frame is None else frame)

    def legal_actions(self):
        return list(self._history[-1]["legal_actions"])

    def levels(self):
        return int(self._history[-1]["levels"])

    def win_levels(self):
        return int(self._history[-1]["win_levels"])

    def state(self):
        """The latest record without its frames (status, levels, legal actions, outcome...)."""
        return {k: v for k, v in self._history[-1].items() if k not in ("frame", "frames")}

    def history(self):
        """Every observation of this game so far (initial state first), kept in the kernel."""
        return list(self._history)

    def __repr__(self):
        last = self.state()
        return (f"Env(actions={last.get('actions_taken')}, levels={last.get('levels')}/"
                f"{last.get('win_levels')}, status={last.get('status')!r}, "
                f"legal_actions={last.get('legal_actions')})")


def main():
    # Load trusted modules before applying filesystem restrictions.
    sandbox_path = os.environ.pop("ARC_SANDBOX_MODULE")
    spec = importlib.util.spec_from_file_location("sandbox", sandbox_path)
    sandbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandbox)
    for module in ALLOWED:
        importlib.import_module(module)
    source_in, source_out = sys.stdin, sys.stdout

    def send(value):
        source_out.write(json.dumps(value) + "\n")
        source_out.flush()

    def rpc(method, **args):
        send({"type": "rpc", "method": method, **args})
        result = json.loads(source_in.readline())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    class Output(TextIOBase):
        def write(self, value):
            for start in range(0, len(value), 16384):
                send({"type": "output", "text": value[start : start + 16384]})
            return len(value)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (name.split(".")[0] not in ALLOWED and name.split(".")[0] != "skills") or level:
            raise ImportError(f"Module unavailable: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins = {
        k: v for k, v in vars(builtins).items() if k not in {"open", "input", "breakpoint"}
    }
    safe_builtins["__import__"] = safe_import
    skills = types.ModuleType("skills")
    skills.__doc__ = "The shared harness procedures (harness/skills/<name>.py), one submodule each."
    skills.names = lambda: sorted(n for n in vars(skills) if not n.startswith("_") and n != "names")
    sys.modules["skills"] = skills

    def load_skills(sources):
        """Rebuild the skills package from the host's sources; errors keep the old module."""
        errors = {}
        for name in list(vars(skills)):
            if not name.startswith("_") and name != "names" and name not in sources:
                delattr(skills, name)
                sys.modules.pop(f"skills.{name}", None)
        for name, source in sorted(sources.items()):
            module = types.ModuleType(f"skills.{name}")
            module.__dict__["__builtins__"] = safe_builtins
            try:
                with contextlib.redirect_stdout(Output()), contextlib.redirect_stderr(Output()):
                    exec(compile(source, f"skills/{name}.py", "exec"), module.__dict__)
            except BaseException:  # noqa: BLE001 - reported to the model, never fatal
                errors[name] = traceback.format_exc(limit=-3)
                continue
            setattr(skills, name, module)
            sys.modules[f"skills.{name}"] = module
        return errors

    namespace = {"__builtins__": safe_builtins, "__name__": "__game__"}
    isolation = sandbox.restrict(read=sandbox.runtime_paths())
    env = Env(rpc)
    send({"type": "ready", "sandbox": isolation})
    for line in source_in:
        request = json.loads(line)
        if request["type"] == "close":
            break
        if request["type"] == "skills":
            send({"type": "done", "error": None, "skills": load_skills(request["sources"])})
            continue
        namespace.update(env=env, np=np, skills=skills)
        error = None
        try:
            tree = ast.parse(request["code"])
            with contextlib.redirect_stdout(Output()), contextlib.redirect_stderr(Output()):
                exec(compile(tree, "<python-tool>", "exec"), namespace, namespace)
        except BaseException:  # noqa: BLE001 - the traceback is the model's feedback
            error = traceback.format_exc()
            send({"type": "output", "text": error})
        send({"type": "done", "error": error})


if __name__ == "__main__":
    main()
