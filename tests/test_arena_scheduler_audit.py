import copy

import pytest

from arena.scheduler_audit import audit_scheduler


def event(kind, reason, wall, active, level=0, budget=None):
    return dict(event=kind, reason=reason, wall_seconds=wall, active_seconds=active,
                levels=level, budget=budget)


def fixture():
    a = dict(game_id='a', active_seconds=80, t1_seconds=10,
             time_to_level=[dict(level=1, wall_seconds=10, active_seconds=10)],
             admissions=[event('admit', 'opening_start', 0, 0),
                         event('grant', 'opening_switch', 20, 20, 1, 30),
                         event('park', 'level_budget', 50, 50, 1, 30),
                         event('admit', 'regrant', 70, 50, 1, 70),
                         event('finish', 'wall_end', 100, 80, 1)])
    b = dict(game_id='b', active_seconds=70, t1_seconds=20,
             time_to_level=[dict(level=1, wall_seconds=20, active_seconds=20),
                            dict(level=2, wall_seconds=40, active_seconds=40)],
             admissions=[event('admit', 'opening_start', 0, 0),
                         event('grant', 'opening_switch', 20, 20, 1, 40),
                         event('grant', 'level_reached', 40, 40, 2, 60),
                         event('finish', 'won', 70, 70, 2)])
    c = dict(game_id='c', active_seconds=70, t1_seconds=None, time_to_level=[],
             admissions=[event('admit', 'opening_start', 0, 0),
                         event('park', 'opening_switch', 20, 20),
                         event('admit', 'regrant', 50, 20, 0, 50),
                         event('finish', 'wall_end', 100, 70)])
    return dict(wall_seconds=100, config={'gameplay_seconds': 100}, games=[a, b, c],
                scheduler=dict(active_limit=2, opening_switch_level1_games=2,
                               switch_wall_seconds=20, max_grant_seconds=None))


def test_independent_grant_arithmetic_clocks_and_phase_limits():
    out = audit_scheduler(fixture())
    assert out['valid'], out['issues']
    assert out['checks'] == dict(games=3, admission_clocks=13, clears=3, grants=5, budget_parks=1)


@pytest.mark.parametrize('field', ['t1_seconds', 'active_seconds'])
def test_clock_mutations_fail(field):
    data = fixture()
    data['games'][0][field] += 1
    assert not audit_scheduler(data)['valid']


def test_grants_premature_parks_and_opening_trigger_are_checked():
    data = fixture()
    data['games'][0]['admissions'][1]['budget'] = 31
    out = audit_scheduler(data)
    assert any('grant at' in s for s in out['issues'])
    assert any('before exhausting' in s for s in out['issues'])
    for switch in (None, 15, 22, float('nan')):
        data = fixture()
        data['scheduler']['switch_wall_seconds'] = switch
        assert not audit_scheduler(data)['valid']


def test_missing_evidence_and_excess_concurrency_are_not_accepted():
    data = fixture()
    data['scheduler']['active_limit'] = 1
    assert any('eligible games' in s for s in audit_scheduler(data)['issues'])
    data = fixture()
    del data['games'][0]['admissions']
    assert not audit_scheduler(data)['valid']


def test_cleanup_tail_is_not_misclassified_as_gameplay_overrun():
    data = fixture()
    data['wall_seconds'] = 104
    out = audit_scheduler(data)
    assert out['valid'] and out['warnings']
    data['config']['gameplay_seconds'] = 95
    assert any('wall-end event' in s for s in audit_scheduler(data)['issues'])


def test_report_generation_automatically_rejects_invalid_timing(monkeypatch):
    from arena import report
    data = fixture()
    monkeypatch.setattr(report, 'fleet_summary', lambda rows, seconds: {
        'failed_sessions': 0, 'mean_score_equal_weight_25_games': 0})

    def make():
        return report.trial_report({}, rows=data['games'], elapsed=data['wall_seconds'],
                                   choice={}, cap=16384, config=data['config'],
                                   scheduler=data['scheduler'], gate={},
                                   engine={'valid': True, 'preemptions': 0}, task_errors=[])

    assert make()['status'] == 'complete'
    data['games'][0]['time_to_level'][0]['active_seconds'] += 1
    result = make()
    assert result['status'] == 'error' and not result['scheduler_audit']['valid']


def test_input_records_remain_immutable():
    data = fixture()
    before = copy.deepcopy(data)
    audit_scheduler(data)
    assert data == before


def test_opt_in_grant_cap_is_part_of_the_checked_specification():
    data = fixture()
    data['scheduler']['max_grant_seconds'] = 30
    for game in data['games']:
        for row in game['admissions']:
            if row['budget'] is not None:
                row['budget'] = min(row['budget'], 30)
    assert audit_scheduler(data)['valid']


def test_t1_relative_cap_and_regrant_multiple_are_modelled():
    # reclaim-grant-2t1: every steady-state grant is at most max(2*T1, 30).
    data = fixture()
    data['scheduler']['max_grant_t1_multiple'] = 2
    data['games'][1]['admissions'][2]['budget'] = 40   # level 2 control grant 3*20=60 -> 2*T1=40
    data['games'][0]['admissions'][3]['budget'] = 30   # regrant 70 -> max(2*10, 30)=30
    out = audit_scheduler(data)
    assert out['valid'], out['issues']
    data['games'][1]['admissions'][2]['budget'] = 60
    assert any('level_reached grant' in s for s in audit_scheduler(data)['issues'])
    # reclaim-regrant-2x: re-admission grants are max(2*spent, spent+30); level budgets unchanged.
    data = fixture()
    data['scheduler']['regrant_multiple'] = 2
    data['games'][0]['admissions'][3]['budget'] = 80   # spent 40 on level 1: 1.5x=70 -> 2x=80
    out = audit_scheduler(data)
    assert out['valid'], out['issues']   # game c's level-0 regrant (spent 20) stays at the +30 floor
    data['games'][0]['admissions'][3]['budget'] = 70
    assert any('regrant grant' in s for s in audit_scheduler(data)['issues'])
    for bad in ({'max_grant_t1_multiple': 0.5}, {'regrant_multiple': 0}):
        data = fixture()
        data['scheduler'].update(bad)
        assert not audit_scheduler(data)['valid']
