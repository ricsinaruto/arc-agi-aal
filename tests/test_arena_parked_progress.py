"""Late clears earn one resumption, without changing active grants or controls."""

import importlib.util
from pathlib import Path

import pytest
from test_arena_core import Now, make_games

from arena.analysis import allocation
from arena.scheduler import Scheduler


def opening(*, enabled=True, count=6, seconds=3600):
    now = Now()
    wall, games = make_games(now, count=count, seconds=seconds)
    scheduler = Scheduler(games, wall, prioritize_parked_progress=enabled)
    wall.release()
    for game in games:
        game.clock.release()
        game.observer = scheduler
    scheduler.start_all()
    return now, wall, games, scheduler


def focused(enabled=True):
    now, wall, games, scheduler = opening(enabled=enabled)
    now.advance(100)
    games[0].report_level(1, wall.elapsed())
    now.advance(50)
    games[1].report_level(1, wall.elapsed())
    scheduler.tick()
    return now, wall, games, scheduler


@pytest.mark.parametrize('enabled,expected', [(False, 2), (True, 4)])
def test_late_clear_waits_for_slot_and_changes_only_next_resumption(enabled, expected):
    now, wall, games, scheduler = focused(enabled)
    old_budgets = [g.clock.budget for g in games[:2]]
    now.advance(7)
    games[4].report_level(1, wall.elapsed())
    events = len(scheduler.events)
    games[4].report_level(1, wall.elapsed())  # duplicate observation is not new progress
    assert len(scheduler.events) == events
    scheduler.tick()
    assert [g.index for g in scheduler.active()] == [0, 1]
    assert [g.clock.budget for g in games[:2]] == old_budgets
    assert games[4].clock.parked and games[4].clock.active_seconds() == 150
    games[0].note_finished('won')
    scheduler.tick()
    assert [g.index for g in scheduler.active()] == [1, expected]
    assert not games[expected].pending_parked_progress
    if enabled:
        assert games[4].admissions[-1]['reason'] == 'parked_progress'
        assert games[4].clock.budget == 300  # original two-times-T1 grant
        scheduler.park(games[4], 'test_pause')
        games[4].report_level(1, wall.elapsed())
        scheduler.tick()
        assert [g.index for g in scheduler.active()] == [1, 2]
        assert not games[4].pending_parked_progress  # no permanent score-based priority


def test_additional_late_clear_can_earn_a_new_priority_after_consumption():
    now, wall, games, scheduler = focused()
    games[4].report_level(1, wall.elapsed())
    scheduler.finish(games[0], 'won')
    scheduler.tick()
    scheduler.park(games[4], 'test_pause')
    scheduler.tick()
    now.advance(5)
    games[4].report_level(2, wall.elapsed())
    assert games[4].pending_parked_progress
    scheduler.finish(games[2], 'won')
    scheduler.tick()
    assert games[4].state == 'active' and not games[4].pending_parked_progress
    assert games[4].clock.budget == 450


def test_terminal_parked_game_is_never_resumed_and_multiple_clears_do_not_stack():
    _, wall, games, scheduler = focused()
    games[4].report_level(2, wall.elapsed())
    games[5].report_level(1, wall.elapsed())
    games[4].note_finished('won')
    games[0].note_finished('won')
    scheduler.tick()
    assert games[4].state == 'finished' and not games[4].pending_parked_progress
    assert [g.index for g in scheduler.active()] == [1, 5]
    assert not any(e['reason'] == 'parked_progress' for e in games[4].admissions)


def test_priority_ties_keep_service_order_and_never_exceed_active_limit():
    _, wall, games, scheduler = focused()
    for i in (5, 4):  # notification arrival order does not replace existing tie break
        games[i].report_level(2, wall.elapsed())
    scheduler.finish(games[0], 'won')
    scheduler.tick()
    assert [g.index for g in scheduler.active()] == [1, 4]
    scheduler.finish(games[1], 'won')
    scheduler.tick()
    assert [g.index for g in scheduler.active()] == [4, 5]
    assert all(not g.pending_parked_progress for g in games)


@pytest.mark.parametrize('enabled,expected', [(False, 0), (True, 105)])
def test_rehearsal_late_lp85_clear_replay(enabled, expected):
    # Recorded boundary times from the 110-game v1 rehearsal; this tests allocation,
    # not a counterfactual model trajectory or score.
    now, wall, games, scheduler = opening(enabled=enabled, count=110)
    now.advance(1040.044)
    games[61].report_level(1, wall.elapsed())
    now.advance(1218.519 - wall.elapsed())
    games[22].report_level(1, wall.elapsed())
    scheduler.tick()
    now.advance(1225.863 - wall.elapsed())
    games[105].report_level(1, wall.elapsed())
    scheduler.tick()
    assert [g.index for g in scheduler.active()] == [22, 61]
    now.advance(2316.749 - wall.elapsed())
    games[22].note_finished('won')
    scheduler.tick()
    assert games[expected].state == 'active' and games[61].state == 'active'
    assert len(scheduler.active()) == 2
    scheduler.wall_end()
    report = {
        'wall_seconds': wall.elapsed(), 'scheduler': scheduler.summary(),
        'games': [{'game_id': g.game_id, 'admissions': g.admissions,
                   'time_to_level': g.time_to_level} for g in games],
    }
    measured = allocation(report)
    assert measured['valid'], measured['issues']
    late = measured['games'][105]
    assert late['clears_outside_eligibility'][0]['level'] == 1


def test_builder_opt_in_preserves_all_other_competition_settings(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('builder', root / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    args = dict(paired_30m=True, arena_variant='cache-reclaim', competition=True)
    _, control = builder.build('test', output=tmp_path / 'control', **args)
    _, candidate = builder.build('test', output=tmp_path / 'candidate',
                                 prioritize_parked_progress=True, **args)
    assert candidate.pop('prioritize_parked_progress') is True
    assert candidate == control
    with pytest.raises(ValueError, match='another scheduling variant'):
        builder.build('invalid', output=tmp_path / 'invalid', paired_30m=True,
                      arena_variant='grant-180s', prioritize_parked_progress=True)
