"""Reopened phase 1: a freed slot runs the whole top tie group until a clear."""

import pytest
from test_arena_core import Now
from test_arena_pool import charge, make_pool_games

from arena.scheduler import Scheduler
from arena.scheduler_audit import audit_scheduler


def setup(mode, count=6):
    now = Now()
    wall, games = make_pool_games(now, count)
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions", reopen_phase1=mode)
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    now.advance(100)
    charge(games[0], 900)    # T1 = 10 normalized seconds
    games[0].note_level(1, wall.elapsed())
    now.advance(50)
    charge(games[1], 1900)   # T1 = 20
    games[1].note_level(1, wall.elapsed())
    sched.tick()
    for g in games:
        g.observer = sched
    return now, wall, games, sched


def test_pause_mode_holds_the_surviving_focus_game_and_resumes_it_after_the_first_clear():
    now, wall, games, sched = setup("pause")
    g0, g1 = games[0], games[1]
    assert sched.focus() == [g0, g1] and g1.clock.budget == 80
    charge(g1, 1000)  # g1 spends 11 of its 80 before the slot frees
    charge(g0, 3900)  # g0 exhausts its budget: the slot frees, the tied level-0 group reopens phase 1
    sched.tick()
    assert g0.state == "parked" and g1.state == "held" and g1.clock.parked
    group = sched.pool_games()
    assert [g.index for g in group] == [2, 3, 4, 5] and all(g.clock.budget is None and g.readmissions == 0 for g in group)
    assert g1.admissions[-1]["reason"] == "reopen_hold" and games[3].admissions[-1]["reason"] == "reopen"
    assert sched.reopen_active() and sched.reopens == 1 and sched.backlog() == [g0]
    sched.tick()
    assert sched.pool_games() == group  # nothing changes until a clear, however long it takes
    now.advance(400)
    charge(games[3], 2900)
    games[3].report_level(1, wall.elapsed())
    assert games[3].state == "active" and not games[3].pool and games[3].admissions[-1]["reason"] == "reopen_clear"
    assert games[3].clock.budget == pytest.approx(2 * 2 * games[3].t1) and games[3].readmissions == 1
    assert all(games[i].state == "parked" and games[i].admissions[-1]["reason"] == "reopen_close" and games[i].readmissions == 0
               for i in (2, 4, 5))
    assert g1.state == "active" and g1.admissions[-1]["reason"] == "reopen_resume" and g1.clock.budget == 80
    assert g1.clock.budget_spent() == pytest.approx(11)  # paused while held: the 400 s cost it nothing
    assert sorted(g.index for g in sched.focus()) == [1, 3] and not sched.reopen_active()
    summary = sched.summary()
    assert summary["reopen_phase1"] == "pause" and summary["reopens"] == 1


def test_run_mode_keeps_the_focus_game_playing_and_a_single_top_game_is_admitted_normally():
    now, wall, games, sched = setup("run")
    g0, g1 = games[0], games[1]
    charge(g0, 3900)
    sched.tick()
    assert g1.state == "active" and not g1.clock.parked and [g.index for g in sched.pool_games()] == [2, 3, 4, 5]
    charge(games[2], 900)
    games[2].report_level(1, wall.elapsed())
    assert sorted(g.index for g in sched.focus()) == [1, 2] and sched.pool_games() == []
    # Next free slot: g3, g4, g5 are still tied at the top, so phase 1 reopens for the three of them.
    charge(g1, 100000)  # exhaust g1: the slot frees again
    sched.tick()
    assert [g.index for g in sched.pool_games()] == [3, 4, 5] and sched.reopens == 2
    charge(games[5], 900)
    games[5].report_level(1, wall.elapsed())
    # g3 and g4 are still tied at the top: the next free slot reopens phase 1 for the two of them.
    charge(games[2], 100000)
    sched.tick()
    assert sched.reopens == 3 and [g.index for g in sched.pool_games()] == [3, 4]
    charge(games[3], 900)
    games[3].report_level(1, wall.elapsed())
    # g4 is now alone at the top (readmissions 0, level 0): a single game is admitted normally.
    charge(games[5], 100000)
    sched.tick()
    assert sched.reopens == 3 and games[4].state == "active" and not games[4].pool
    assert games[4].admissions[-1]["reason"] == "regrant" and sorted(g.index for g in sched.focus()) == [3, 4]


def test_a_held_game_that_clears_from_an_in_flight_request_gets_its_level_budget_on_resume():
    now, wall, games, sched = setup("pause")
    g0, g1 = games[0], games[1]
    charge(g0, 3900)
    sched.tick()
    assert g1.state == "held"
    g1.report_level(2, wall.elapsed())  # the request in flight when the hold began cleared level 2
    assert g1.admissions[-1]["event"] == "progress" and g1.state == "held"
    charge(games[4], 900)
    games[4].report_level(1, wall.elapsed())
    assert g1.state == "active" and [e["reason"] for e in g1.admissions[-2:]] == ["reopen_resume", "level_reached"]
    assert g1.clock.budget == pytest.approx(2 * 3 * g1.t1)


def test_validation_and_audit_accept_the_reopen_events():
    now = Now()
    wall, games = make_pool_games(now, 3)
    with pytest.raises(ValueError, match="reopen_phase1 must be"):
        Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions", reopen_phase1="always")
    with pytest.raises(ValueError, match="no audition pool"):
        Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions", reopen_phase1="pause", audition_pool=2)
    assert callable(audit_scheduler)
