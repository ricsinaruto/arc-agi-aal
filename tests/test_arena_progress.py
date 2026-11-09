"""Learning-progress backlog order (``backlog_policy="progress"``, ideas.md point 2)."""

import pytest
from test_arena_core import Now
from test_arena_pool import CONTROL, charge, make_pool_games

from arena.scheduler import Scheduler
from arena.variants import apply_variant


def _scheduler(count=6):
    now = Now()
    wall, games = make_pool_games(now, count)
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="progress")
    wall.release()
    for g in games:
        g.clock.release()
        g.observer = sched
    sched.start_all()
    return now, wall, games, sched


def test_progress_value_ranks_progressing_games_above_untried_ones_above_stuck_ones():
    now, wall, games, sched = _scheduler()
    untried = sched.progress_value(games[5])
    assert untried == pytest.approx((1 / 36) * (0.5 / 90))
    charge(games[0], 900)   # 10 busy seconds, one clear of eight levels
    games[0].report_level(1, wall.elapsed(), total_levels=8)
    charge(games[1], 900)   # 10 busy seconds, one clear of a six-level game: a larger share per level
    games[1].report_level(1, wall.elapsed(), total_levels=6)
    charge(games[2], 20000)  # 201 busy seconds without a clear
    for _ in range(30):
        games[3].note_action("same")  # stalled: 29 actions without a novel board
    assert sched.progress_value(games[1]) > sched.progress_value(games[0]) > untried
    assert sched.progress_value(games[2]) < untried
    assert sched.progress_value(games[3]) == pytest.approx(untried / (1 + 29 / 10))
    order = [g.game_id for g in sorted(games, key=sched.priority)]
    assert order[:2] == ["g1", "g0"] and order[2:4] == ["g4", "g5"] and order[-2:] == ["g2", "g3"]
    assert sched.summary()["backlog_policy"] == "progress"
    assert sched.summary()["progress_order"] == {"model": "levels", "prior_clears": 0.5, "prior_seconds": 90.0, "stall_scale": 10.0, "default_levels": 8, "novelty_prior": 3}


def test_progress_order_admits_the_top_value_game_and_a_parked_clear_first():
    now, wall, games, sched = _scheduler()
    now.advance(100)
    charge(games[0], 900)
    games[0].report_level(1, wall.elapsed())
    now.advance(50)
    charge(games[1], 1900)
    games[1].report_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "steady" and {g.game_id for g in sched.focus()} == {"g0", "g1"}
    charge(games[4], 3000)   # 31 busy seconds in the opening, no clear: below the untried games
    games[3].note_action("a")
    games[3].note_action("b")  # novelty does not enter the value; g2 (untried) stays first by roster
    now.advance(1)
    charge(games[0], 100000)  # exhaust g0's level-2 budget
    sched.tick()
    assert games[0].state == "parked" and games[0].admissions[-1]["reason"] == "level_budget"
    sched.tick()
    assert games[2].state == "active" and games[2].admissions[-1]["reason"] == "regrant"
    # A clear that arrives while parked still jumps the queue.
    games[4].report_level(1, wall.elapsed())
    assert games[4].pending_parked_progress
    charge(games[2], 100000)
    sched.tick()
    sched.tick()
    assert games[4].state == "active" and games[4].admissions[-1]["reason"] == "parked_progress"


def test_progress_variant_changes_only_the_backlog_order():
    base = apply_variant(CONTROL, "reclaim-fixed-2x")
    progress = apply_variant(CONTROL, "fixed-2x-progress")
    assert progress["backlog_policy"] == "progress" and base["backlog_policy"] == "readmissions"
    assert progress["budget_clock"] == "busy" and progress["level_budget_policy"] == "fixed-2x"
    assert set(progress["arena_variant"]["reference_difference"]) == {"backlog_policy"}
    diff = {k for k in base.keys() | progress.keys() if base.get(k) != progress.get(k)}
    assert diff == {"backlog_policy", "level_budget_rule", "arena_variant"}
    with pytest.raises(ValueError, match="backlog policy"):
        Scheduler([], None, backlog_policy="random")


def test_novelty_model_drops_the_level_share_and_ranks_by_distinct_boards():
    with pytest.raises(ValueError, match="progress_model"):
        Scheduler([], None, budget_policy="fixed-2x", backlog_policy="progress", progress_model="scores")
    now = Now()
    wall, games = make_pool_games(now, 6)
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="progress", progress_model="novelty")
    wall.release()
    for g in games:
        g.clock.release()
        g.observer = sched
    sched.start_all()
    for g, total in zip(games, (6, 8, 8, 8, 6, 8)):
        g.report_level(0, 0.0, total_levels=total)
    untried = sched.progress_value(games[5])
    assert untried == pytest.approx(0.5 / 90)  # no (levels+1)/(N(N+1)/2) factor: six- and eight-level games tie
    assert sched.progress_value(games[0]) == untried
    for key in "abc":
        games[1].note_action(key)  # three actions, three distinct boards: 3/(3+3) extra
    assert sched.progress_value(games[1]) == pytest.approx((0.5 + 0.5) / 90)
    for key in "abcdefghij" + "j" * 13:
        games[2].note_action(key)  # 23 actions, 10 distinct boards, 13 stalled
    assert sched.progress_value(games[2]) == pytest.approx((0.5 + 10 / 26) / 90 / (1 + 13 / 10))
    charge(games[3], 900)  # a clear still dominates: (1 + 0.5) / (10 + 90)
    games[3].report_level(1, wall.elapsed(), total_levels=8)
    assert sched.progress_value(games[3]) == pytest.approx(1.5 / 100)
    order = [g.game_id for g in sorted(games, key=sched.priority)]
    assert order == ["g3", "g1", "g0", "g4", "g5", "g2"]
    assert sched.summary()["progress_order"]["model"] == "novelty"
    study = apply_variant(CONTROL, "fixed-2x-progress-novelty")
    assert study["backlog_policy"] == "progress" and study["progress_model"] == "novelty"
    assert study["arena_variant"]["reference_difference"]["progress_model"] == {"before": None, "after": "novelty"}
