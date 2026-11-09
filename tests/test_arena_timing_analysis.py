import copy

import pytest

from arena.timing_analysis import admission_timing


def fixture():
    return {
        'wall_seconds': 20, 'scheduler': {'switch_wall_seconds': 10},
        'games': [{'game_id': 'one', 'admissions': [
            {'event': 'admit', 'wall_seconds': 0},
            {'event': 'park', 'wall_seconds': 10},
            {'event': 'admit', 'wall_seconds': 15},
            {'event': 'finish', 'wall_seconds': 20}],
            'time_to_level': [{'level': 1, 'wall_seconds': 12, 'active_seconds': 10}],
            'admission_timing': {'attempts': [
                {'attempt': 1, 'started_wall_seconds': 0, 'wait_ended_wall_seconds': 6,
                 'admitted_wall_seconds': 6, 'ended_wall_seconds': 12},
                {'attempt': 2, 'started_wall_seconds': 13, 'wait_ended_wall_seconds': 17,
                 'admitted_wall_seconds': 17, 'ended_wall_seconds': 21}]}}]}


def test_phase_first_clear_and_cancelled_tail_use_observed_intersections():
    out = admission_timing(fixture())
    assert out['valid']
    first = out['games'][0]['first_clear']
    assert first['eligible_seconds'] == 10
    assert first['eligible_wait_without_slot_seconds'] == 6
    assert first['eligible_slot_held_seconds'] == 4
    assert first['parked_slot_held_seconds'] == 2
    assert first['wait_fraction_of_active'] == .6
    phase2 = out['phase_totals']['phase2']
    assert phase2['eligible_seconds'] == 5
    assert phase2['eligible_wait_without_slot_seconds'] == 2
    assert phase2['eligible_slot_held_seconds'] == 3
    assert phase2['parked_slot_held_seconds'] == 2


def test_nested_requests_do_not_double_count_wall_time():
    data = fixture()
    data['games'][0]['admission_timing']['attempts'].append({
        'attempt': 3, 'started_wall_seconds': 7, 'wait_ended_wall_seconds': 9,
        'admitted_wall_seconds': 9, 'ended_wall_seconds': 11})
    out = admission_timing(data)
    assert out['valid']
    phase = out['phase_totals']['opening']
    assert phase['nested_wait_while_slot_held_seconds'] == 2
    assert phase['eligible_wait_without_slot_seconds'] == 6
    assert phase['eligible_slot_held_seconds'] == 4
    assert phase['eligible_outside_admission_seconds'] == 0


def test_unrecorded_tools_are_residual_not_gpu_service():
    data = fixture()
    data['games'][0]['admission_timing']['attempts'][0]['started_wall_seconds'] = 2
    out = admission_timing(data)
    assert out['valid']
    assert out['phase_totals']['opening']['eligible_outside_admission_seconds'] == 2
    assert 'not GPU service' in out['scope']


@pytest.mark.parametrize('fault', ['missing_end', 'negative', 'nonfinite', 'reversed',
                                  'bad_clear', 'duplicate', 'missing_attempts', 'bad_admission'])
def test_incomplete_or_inconsistent_evidence_is_not_silently_zero(fault):
    data = fixture()
    game = data['games'][0]
    a = game['admission_timing']['attempts'][0]
    if fault == 'missing_end':
        del a['ended_wall_seconds']
    elif fault == 'negative':
        a['started_wall_seconds'] = -1
    elif fault == 'nonfinite':
        a['wait_ended_wall_seconds'] = float('nan')
    elif fault == 'reversed':
        a['started_wall_seconds'] = 7
    elif fault == 'bad_clear':
        game['time_to_level'][0]['active_seconds'] = 12
    elif fault == 'duplicate':
        game['admission_timing']['attempts'].append(copy.deepcopy(a))
    elif fault == 'missing_attempts':
        del game['admission_timing']
    elif fault == 'bad_admission':
        a['admitted_wall_seconds'] = 9
    out = admission_timing(data)
    assert not out['valid'] and out['issues']
    assert out['games_covered'] == 0


def test_wait_cancelled_without_admission_and_no_phase_switch():
    data = fixture()
    data['scheduler']['switch_wall_seconds'] = None
    game = data['games'][0]
    game['admission_timing']['attempts'] = [
        {'attempt': 1, 'started_wall_seconds': 0, 'wait_ended_wall_seconds': 20}]
    game['time_to_level'] = []
    out = admission_timing(data)
    assert out['valid']
    assert list(out['phase_totals']) == ['opening']
    assert out['phase_totals']['opening']['eligible_wait_without_slot_seconds'] == 15
    assert out['phase_totals']['opening']['eligible_slot_held_seconds'] == 0


def inference_fixture():
    data = fixture()
    data['scheduler']['budget_clock'] = 'inference'
    game = data['games'][0]
    game.update(inference_seconds=7, t1_seconds=4, inference_intervals=[
        {'started_wall_seconds': 8, 'ended_wall_seconds': 12,
         'purpose': 'gameplay', 'status': 'completed'},
        {'started_wall_seconds': 17, 'ended_wall_seconds': 20,
         'purpose': 'summary', 'status': 'CancelledError'}])
    return data


def test_inference_intervals_cross_phase_and_park_without_counting_waits():
    from arena.timing_analysis import inference_timing
    data = inference_fixture()
    before = copy.deepcopy(data)
    out = inference_timing(data)
    assert out['valid'] and data == before
    assert out['games'][0]['first_clear_inference_seconds'] == 4
    assert out['phase_totals']['opening']['inference_seconds'] == 2
    assert out['phase_totals']['opening']['mean_requests_in_flight'] == .2
    assert out['phase_totals']['phase2'] == {
        'inference_seconds': 5, 'summary_seconds': 3,
        'interrupted_request_seconds': 3, 'mean_requests_in_flight': .5}


@pytest.mark.parametrize('fault', ['queue_counted', 'overlap', 'wrong_total', 'wrong_t1',
                                  'missing_intervals', 'no_admissions'])
def test_inference_accounting_rejects_queue_and_missing_or_inconsistent_evidence(fault):
    from arena.timing_analysis import inference_timing
    data = inference_fixture()
    game = data['games'][0]
    if fault == 'queue_counted':
        game['inference_intervals'][0]['started_wall_seconds'] = 4
    elif fault == 'overlap':
        game['inference_intervals'][1]['started_wall_seconds'] = 10
    elif fault == 'wrong_total':
        game['inference_seconds'] = 8
    elif fault == 'wrong_t1':
        game['t1_seconds'] = 12
    elif fault == 'missing_intervals':
        del game['inference_intervals']
    elif fault == 'no_admissions':
        game['admission_timing']['attempts'] = []
    out = inference_timing(data)
    assert not out['valid'] and out['issues']


def test_historical_inference_unavailable_is_not_zero():
    from arena.timing_analysis import inference_timing
    out = inference_timing(fixture())
    assert out['available'] is False and out['valid'] is None
    assert 'phase_totals' not in out
