"""Structured stuck procedure: same cadence as the public tracker, checklist text, collapse detector."""

from types import SimpleNamespace

import pytest

from arena.procedure import ProcedureStuckTracker, action_key
from duck_transfer.feedback import StuckTracker

CFG = SimpleNamespace(stuck_no_change_actions=12, stuck_turns=8, stuck_notice_turns=4)


def test_action_key_groups_clicks_by_region():
    assert action_key("MOUSE(row=3, col=17)") == "click region r0-7 c16-23"
    assert action_key("MOUSE(row=7, col=16)") == action_key("MOUSE(row=0, col=23)")
    assert action_key("ACTION3") == "ACTION3"


def test_notice_cadence_matches_the_public_tracker_and_text_is_the_checklist():
    public, proc = StuckTracker(CFG, now=0.0), ProcedureStuckTracker(CFG, now=0.0)
    seen = []
    for turn in range(20):
        for tracker in (public, proc):
            tracker.note_turn()
        a, b = public.notice(2, ["ACTION1"], now=10.0), proc.notice(2, ["ACTION1"], now=10.0)
        assert (a is None) == (b is None)
        if b:
            seen.append(b)
    assert len(seen) == 4 and proc.total_notices == 4
    assert "CONFIRMED, REFUTED or" in seen[0] and "Bad execution or bad idea" in seen[0]
    assert "tried on this exact board with no effect: ACTION1" in seen[0] and "COLLAPSE" not in seen[0]


def test_collapse_detector_needs_a_full_window_and_resets_per_level():
    proc = ProcedureStuckTracker(CFG, collapse_window=10, collapse_share=0.8, now=0.0)
    for _ in range(9):
        proc.note_label("MOUSE(row=1, col=1)")
    assert proc.collapse() is None
    proc.note_label("ACTION2")  # window full: 9 of 10 in one region >= 80%
    line = proc.collapse()
    assert line and "9 of your last 10 actions were click region r0-7 c0-7" in line
    for _ in range(3):
        proc.note_label("ACTION4")
    assert proc.collapse() is None  # 6 of 10 < 80%
    for _ in range(10):
        proc.note_label("MOUSE(row=2, col=2)")
    for _ in range(8):
        proc.note_turn()
    assert "COLLAPSE DETECTED" in proc.notice(1, [], now=5.0) and proc.collapses == 1
    proc.reset_level()
    assert proc.collapse() is None
    with pytest.raises(ValueError):
        ProcedureStuckTracker(CFG, collapse_window=2)
