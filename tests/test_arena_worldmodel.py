"""World-model harness (fixed-2x-worldmodel): check_sim / plan / run_plan skills, the level-up notice, the variant."""

import json
from pathlib import Path
from types import SimpleNamespace

from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "src/arena/harnesses/worldmodel/skills/worldmodel.py"
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def frame(level, step, pos):
    return SimpleNamespace(level=level, step=step, pos=pos)


def skills(transitions, current, receipts=None):
    ns = {"transitions": transitions, "current_frame": current}
    exec(compile(SKILL.read_text(), str(SKILL), "exec"), ns)

    def action(items):
        (item,) = items
        name = item["action"]
        r, c = ns["current_frame"].pos
        dr, dc = MOVES.get(name, (0, 0))
        if name == "LEFT":
            dr, dc = 0, 0  # reality: LEFT is blocked here, the model thinks it moves
        ns["current_frame"] = frame(ns["current_frame"].level, ns["current_frame"].step + 1, (r + dr, c + dc))
        return {"executed": True, "level_completed": ns["current_frame"].pos == (0, 3)}
    ns["action"] = action
    return ns


def sim(state, action):
    dr, dc = MOVES.get(action, (0, 0))
    return (state[0] + dr, state[1] + dc)


def test_check_sim_plan_and_run_plan():
    t = SimpleNamespace
    transitions = [
        t(action="UP", before_frame=frame(0, 0, (2, 2)), after_frame=frame(0, 1, (1, 2))),
        t(action="RIGHT", before_frame=frame(0, 1, (1, 2)), after_frame=frame(0, 2, (1, 3))),
        t(action="RIGHT", before_frame=frame(0, 2, (1, 3)), after_frame=frame(1, 3, (5, 5))),  # level change: skipped
        t(action="DOWN", before_frame=frame(1, 3, (5, 5)), after_frame=frame(1, 4, (5, 5))),  # blocked: mismatch
        t(action="UP", before_frame=None, after_frame=frame(0, 0, (2, 2))),
    ]
    ns = skills(transitions, frame(1, 4, (2, 0)))
    read = lambda f: f.pos  # noqa: E731
    report = ns["check_sim"](sim, read)
    assert report.startswith("check_sim: 2/3 transitions reproduced (level 0: 2/2, level 1: 0/1)")
    assert "level 1 step 4 DOWN: predicted (6, 5), observed (5, 5)" in report
    assert "all reproduced" in ns["check_sim"](sim, read, level=0)
    p = ns["plan"](sim, (2, 0), lambda s: s == (0, 3), ["UP", "RIGHT", "DOWN", "LEFT"])
    assert len(p) == 5 and p.count("UP") == 2 and p.count("RIGHT") == 3
    assert ns["plan"](sim, (0, 0), lambda s: False, ["UP"], max_depth=3) is None
    assert ns["run_plan"](p, sim, read).startswith("run_plan: executed 5/5 actions; level completed at step 4")
    ns["current_frame"] = frame(1, 9, (1, 1))
    out = ns["run_plan"](["LEFT", "UP"], sim, read)
    assert "executed 1/2" in out and "model mismatch at step 0 (LEFT)" in out
    assert ns["_as_action"]("MOUSE(row=12, col=30)") == {"action": "ACTION6", "row": 12, "col": 30}


def test_levelup_notice_once_and_the_variant():
    from arena.duck import WORLDMODEL_NOTICE, ArenaDuckMixin

    agent = SimpleNamespace(harness_v2=None, worldmodel_pending=2, worldmodel_notices=0, steps=7,
                            workspace=SimpleNamespace(append=lambda *a, **k: None))
    first = ArenaDuckMixin.turn_notice(agent)
    assert first == WORLDMODEL_NOTICE.format(cleared=2) and first.startswith("LEVEL 2 CLEARED")
    assert ArenaDuckMixin.turn_notice(agent) is None and agent.worldmodel_notices == 1
    study = apply_variant(CONTROL, "fixed-2x-worldmodel")
    control = apply_variant(CONTROL, "reclaim-fixed-2x")
    study.pop("arena_variant"), control.pop("arena_variant")
    assert {k for k in study.keys() | control.keys() if study.get(k) != control.get(k)} == {"duck_harness", "worldmodel_notice"}
