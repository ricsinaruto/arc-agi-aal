"""Live admission knob (checklist item 5, 30 Oct): Scheduler.tick re-reads active_limit from $ARENA_KNOBS on change,
clamped to 1..KNOB_CAP; a missing or bad file leaves the limit and play alone."""
import json
import os

from test_arena_core import Now

from arena.clock import ActiveClock, GlobalClock
from arena.scheduler import GameRecord, Scheduler


def build(limit=4, count=8):
    now = Now()
    wall = GlobalClock(3600, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="busy"), total_levels=3)
             for i in range(count)]
    log = []
    return Scheduler(games, wall, log=log.append, active_limit=limit), log


def test_knob_raises_and_clamps(tmp_path, monkeypatch):
    sched, log = build()
    knobs = tmp_path / "knobs.json"
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    sched.tick()
    assert sched.active_limit == 4  # no file: unchanged
    knobs.write_text(json.dumps({"active_limit": 6}))
    sched.tick()
    assert sched.active_limit == 6 and log[-1] == {"event": "knobs", "active_limit": 6, "was": 4, "requested": 6}
    knobs.write_text(json.dumps({"active_limit": 500}))
    os.utime(knobs, (1, 2_000_000_000))
    sched.tick()
    assert sched.active_limit == Scheduler.KNOB_CAP


def test_bad_knob_file_keeps_limit(tmp_path, monkeypatch):
    sched, log = build()
    knobs = tmp_path / "knobs.json"
    knobs.write_text("{not json")
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    sched.tick()
    assert sched.active_limit == 4 and log[-1]["event"] == "knobs" and "error" in log[-1]
