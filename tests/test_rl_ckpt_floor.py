"""v17 a5 (30 Oct): U20's accumulate OOMed at 258.4 GiB allocated. Its 285 windows (lvl_window 47104, rebuilt on CPU
from the rollouts with rl_loop's spec rule; tests/data/rl_v17_u20_windows.json: [tail, total] per window) include
tails up to the cap, whose fully checkpointed (n=0) peak the planner cannot bring under the budget. lvl_window 38912
keeps every one of U20's 301 windows under the budget less the room, R3's tables included."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "launch" / "rl"))
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

import launch as L  # noqa: E402
import rl1_memory as m  # noqa: E402

WINDOWS = json.loads((ROOT / "tests" / "data" / "rl_v17_u20_windows.json").read_text())
BUDGET = 240.0


def n0(tail):  # the planner's own model at its floor (rl1_memory.ckpt_peak, n=0) + the R3 term
    f = L.CKPT_FLOOR
    old = dict(m.CKPT)
    try:
        m.CKPT.update(a=f["a"], margin=1.0)
        return m.ckpt_peak(tail, f["base"] + f["gap"], 0) + f["r3"]
    finally:
        m.CKPT.clear()
        m.CKPT.update(old)


def test_u20_windows_fit_under_the_cap():
    rows = WINDOWS["38912"]
    assert len(rows) == 301
    worst = max(n0(tail) for tail, _ in rows)
    assert worst <= BUDGET - L.CKPT_ROOM, worst
    assert abs(worst - L.n0_peak(max(t for t, _ in rows))) < 1e-6


def test_u20_windows_at_47104_reproduce_the_overflow():
    rows = WINDOWS["47104"]
    assert len(rows) == 285  # the early-ahead row's count for s0020
    assert sum(n0(tail) > BUDGET for tail, _ in rows) > 0


def test_launch_check_refuses_an_uncapped_window():
    import config as C

    cfg = C.load(str(ROOT / "launch" / "rl" / "runs" / "rl1-loop-seq-real-v17.json"))
    assert L.ckpt_floor_problems(cfg) == []
    cfg.study["rl_loop"]["lvl_window"] = 47104
    assert L.ckpt_floor_problems(cfg) and "47104" in L.ckpt_floor_problems(cfg)[0]
