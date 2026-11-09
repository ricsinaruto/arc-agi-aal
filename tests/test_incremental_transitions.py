import json
import os
import random

import pytest

from duck_transfer.history import preserve_transition_results
from inference.agent.python_tool_sandbox import _SANDBOX_BOOTSTRAP

FRAMES = os.environ.get("ARC_TRANSITION_FRAMES")  # optional: a JSON list of recorded grids (toolbench frames.json)


def child():
    namespace = {"__name__": "child_test"}
    exec(compile(preserve_transition_results(_SANDBOX_BOOTSTRAP), "<bootstrap>", "exec"), namespace)
    return namespace


def payload(step, grids, action):
    grid = grids[step % len(grids)]
    return {"action": action, "frame": {"ascii": "\n".join("".join(str(c) for c in row) for row in grid), "step": step,
                                        "level": 1, "shape": [len(grid), len(grid[0])], "grid": grid},
            "result": {"executed": True, "action_num": step, "changed": step % 3,
                       "transition": {"cells": [step, step + 1]}} if action and step % 5 else {}}


def same(old, new):
    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert a is not b or a is new[-1]
        assert vars(a).keys() == vars(b).keys()
        for key, value in vars(a).items():
            other = getattr(b, key)
            if key.endswith("frame"):
                assert value is other
            else:
                assert value == other and (key != "result" or isinstance(other, dict))
        assert str(a) == str(b)


def grids_for_test():
    if FRAMES:
        return json.load(open(FRAMES))[:200]
    rng = random.Random(0)
    return [[[rng.randrange(10) for _ in range(8)] for _ in range(8)] for _ in range(50)]


@pytest.mark.parametrize("seed", range(6))
def test_incremental_equals_rebuild(seed):
    ns, grids, rng = child(), grids_for_test(), random.Random(seed)
    history, step = [], 0
    for refresh in range(120):
        new = []
        for _ in range(rng.choice([0, 1, 1, 3])):
            new.append(payload(step, grids, "" if step == 0 or rng.random() < 0.05 else rng.choice(["UP", "LEFT"])))
            step += 1
        if refresh and rng.random() < 0.03:  # the host resent the whole history
            history = ns["_history_from_payload"](new and [*[{"action": h.action, "frame": {"ascii": "", "step": 0,
                                                    "level": 1, "shape": [0, 0], "grid": []}} for h in history], *new]
                                                    or [])
        else:
            history = history + ns["_history_from_payload"](new)
        last = rng.choice([{}, {"executed": True, "action_num": step}])
        old = ns["_transitions_from_history"](history, dict(last))
        fresh = ns["_transitions_incremental"](history, dict(last))
        same(old, fresh)
        if fresh and rng.random() < 0.5:  # the model's code changes what it was given
            view = rng.choice(fresh)
            what = rng.randrange(6)
            if what == 0:
                view.result["note"] = 1
            elif what == 1:
                view.result.pop("executed", None)
            elif what == 2:
                view.tag = "seen"
            elif what == 3:
                view.action = "X"
            elif what == 4:
                view.result.clear()
            elif history:
                rng.choice(history).action = rng.choice(["", "DOWN"])


def test_views_are_reused_and_cheap():
    ns, grids = child(), grids_for_test()
    history = ns["_history_from_payload"]([payload(s, grids, "UP" if s else "") for s in range(300)])
    first = ns["_transitions_incremental"](history, {})
    second = ns["_transitions_incremental"](history, {})
    assert first is not second and first[0] is second[0] and len(first) == 299
    first[0].result["x"] = 1
    third = ns["_transitions_incremental"](history, {})
    assert third[0] is not first[0] and "x" not in third[0].result and third[1] is first[1]
