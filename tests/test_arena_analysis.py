import pytest

from arena.analysis import allocation, phases, replicas, score_curve


def report(switch=3, wall=10):
    return {'wall_seconds': wall, 'scheduler': {'switch_wall_seconds': switch}}


def samples():
    # First timestamp is setup-relative; generation increments are 50 and 100.
    return [{'elapsed_seconds': t, 'metrics': {
        'vllm:generation_tokens_total': n,
        'vllm:prompt_tokens_by_source_total:local_compute': 2*n,
        'vllm:prompt_tokens_by_source_total:local_cache_hit': n,
        'vllm:num_requests_running': 2,
        'vllm:kv_cache_usage_perc': .5}}
        for t, n in [(99, 100), (5, 150), (10, 250)]]


def test_phase_interpolation_subtracts_initial_counters_and_conserves_totals():
    out = phases(report(), samples())
    assert out['valid']
    a, b = out['phases']['opening'], out['phases']['phase2']
    assert a['tokens']['generation'] == 30
    assert b['tokens']['generation'] == 120
    assert a['tokens_per_covered_second']['generation'] == 10
    assert b['tokens_per_covered_second']['generation'] == pytest.approx(120/7)
    assert a['cached_share'] == pytest.approx(1/3)
    assert out['generation_counter_scope'] == 'decode_tokens_including_interrupted_requests'


def test_missing_and_reset_counters_do_not_become_zero_or_positive_throughput():
    rows = samples()
    del rows[1]['metrics']['vllm:generation_tokens_total']
    out = phases(report(), rows)
    assert not out['valid'] and 'generation' not in out['phases']['opening']['tokens']
    rows = samples()
    rows[-1]['metrics']['vllm:generation_tokens_total'] = 10
    out = phases(report(), rows)
    assert not out['valid'] and any('reset' in s for s in out['issues'])


def test_missing_tail_is_reported_and_not_extrapolated():
    out = phases(report(wall=20), samples())
    assert not out['valid']
    phase = out['phases']['phase2']
    assert phase['covered_seconds'] == 7 and phase['wall_seconds'] == 17
    assert phase['coverage_fraction'] == pytest.approx(7/17)
    assert phase['tokens']['generation'] == 120


def test_no_switch_means_only_opening_and_late_switch_has_no_fake_data():
    assert list(phases(report(None), samples())['phases']) == ['opening']
    out = phases(report(15, 20), samples())
    assert out['phases']['phase2']['tokens'] == {}
    assert out['phases']['phase2']['covered_seconds'] == 0


def test_sglang_finished_request_scope_and_total_prompt_denominator():
    rows = [{'elapsed_seconds': t, 'values': {'sglang:generation_tokens_total': n,
        'sglang:prompt_tokens_total': n*4, 'sglang:cached_tokens_total': n*3}}
        for t, n in [(100, 10), (10, 110)]]
    out = phases(report(5), rows)
    assert out['valid'] and out['generation_counter_scope'] == 'finished_requests'
    assert out['phases']['opening']['cached_share'] == .75
    assert out['phases']['opening']['tokens']['generation'] == 50


def test_invalid_time_or_no_samples_fails_explicitly():
    assert not phases(report(), [])['valid']
    rows = samples()
    rows[-1]['elapsed_seconds'] = 2
    assert not phases(report(), rows)['valid']
    assert not phases(report(11), samples())['valid']


def test_time_weighted_concurrency_distinguishes_brief_peak_from_sustained_work():
    rows = samples()
    for row, running in zip(rows, [0, 4, 0]):
        row['metrics']['vllm:num_requests_running'] = running
    out = phases(report(5), rows)
    for phase in out['phases'].values():
        assert phase['observed_peaks']['running'] == 4
        assert phase['estimated_time_weighted_means']['running'] == 2
    del rows[1]['metrics']['vllm:num_requests_running']
    assert phases(report(5), rows)['phases']['opening']['estimated_time_weighted_means']['running'] is None


def allocation_report():
    return {**report(3), 'games': [{'game_id': 'a', 'admissions': [
        {'event': 'admit', 'wall_seconds': 0},
        {'event': 'park', 'wall_seconds': 3},
        {'event': 'admit', 'wall_seconds': 7, 'budget': 9},
        {'event': 'grant', 'wall_seconds': 8, 'budget': 2},
        {'event': 'finish', 'wall_seconds': 9}],
        'time_to_level': [{'level': 1, 'wall_seconds': 4},
                          {'level': 2, 'wall_seconds': 8}]}]}


def test_allocation_counts_only_eligible_intervals_and_detects_late_clear():
    out = allocation(allocation_report())
    assert out['valid']
    game = out['games'][0]
    assert game['eligible_seconds'] == {'opening': 3, 'phase2': 2}
    assert game['clears_outside_eligibility'] == [{'level': 1, 'wall_seconds': 4}]
    assert len(game['grants_exceeding_wall_remaining']) == 1
    assert out['mean_eligible_games']['phase2'] == pytest.approx(2/7)


def test_allocation_preserves_missing_data_and_rejects_reordered_events():
    doc = allocation_report()
    doc['games'].append({'game_id': 'missing'})
    out = allocation(doc)
    assert not out['valid'] and out['games_covered'] == 1 and out['games_expected'] == 2
    doc['games'][0]['admissions'][1]['wall_seconds'] = 20
    assert allocation(doc)['games_covered'] == 0


def test_allocation_no_switch_open_tail_and_boundary_ambiguity():
    doc = allocation_report()
    doc['scheduler']['switch_wall_seconds'] = None
    game = doc['games'][0]
    game['admissions'] = [{'event': 'admit', 'wall_seconds': 0}]
    game['time_to_level'] = [{'level': 1, 'wall_seconds': 0.001}]
    out = allocation(doc)
    assert out['mean_eligible_games'] == {'opening': 1}
    assert out['games'][0]['boundary_ambiguous_clears'] == game['time_to_level']
    assert out['games'][0]['clears_outside_eligibility'] == []


def test_allocation_same_timestamp_park_readmit_does_not_lose_time():
    doc = allocation_report()
    doc['games'][0]['admissions'][2]['wall_seconds'] = 3
    out = allocation(doc)
    assert out['games'][0]['eligible_seconds'] == {'opening': 3, 'phase2': 6}


def curve_evidence():
    # Efficient first clear compensates some of an inefficient second clear.
    doc = {'wall_seconds': 30, 'mean_score_equal_weight_games': 215/6,
           'games': [{'game_id': 'a', 'total_levels': 3, 'levels': 2,
                      'time_to_level': [{'level': 1, 'wall_seconds': 10},
                                        {'level': 2, 'wall_seconds': 20}]}]}
    card = {'environments': [{'id': 'a', 'score': 215/6, 'runs': [
        {'level_scores': [115, 50, 0], 'levels_completed': 2}]}]}
    return doc, card


def test_score_curve_preserves_efficiency_bonus_and_caps_completed_weight():
    doc, card = curve_evidence()
    out = score_curve(doc, card, checkpoints=(0, 10, 15, 20, 30, 60))
    assert out['valid']
    assert [p['score'] for p in out['checkpoints']] == pytest.approx([0, 100/6, 100/6, 215/6, 215/6])
    assert [p['clears'] for p in out['checkpoints']] == [0, 1, 1, 2, 2]
    assert [p['wall_seconds'] for p in out['events']] == [0, 10, 20, 30]


@pytest.mark.parametrize('fault', ['missing_game', 'multi_run', 'wrong_score', 'wrong_levels', 'wrong_order', 'missing_scores'])
def test_score_curve_rejects_incomplete_or_ambiguous_evidence(fault):
    doc, card = curve_evidence()
    env = card['environments'][0]
    if fault == 'missing_game':
        card['environments'] = []
    elif fault == 'multi_run':
        env['runs'].append(env['runs'][0])
    elif fault == 'wrong_score':
        env['score'] = 50
    elif fault == 'wrong_levels':
        doc['games'][0]['levels'] = 1
    elif fault == 'wrong_order':
        doc['games'][0]['time_to_level'][0]['wall_seconds'] = 25
    else:
        del env['runs'][0]['level_scores']
    assert not score_curve(doc, card)['valid']


def test_score_curve_includes_drained_action_receipts_after_report_end():
    doc, card = curve_evidence()
    doc['wall_seconds'] = 19.9
    out = score_curve(doc, card)
    assert out['valid'] and out['post_report_end_clears'] == 1
    assert out['checkpoints'][-1]['wall_seconds'] == 20


def test_replica_weighting_keeps_zero_action_copies_and_all_sources():
    # Five copies of a successful source and four of a zero-score source.
    games = [{'game_id': str(i), 'score': 100 if i == 0 else 0,
              'levels': 8 if i == 0 else 0, 'actions': 80 if i == 0 else 0}
             for i in range(9)]
    roster = [{'game_id': str(i), 'source_game_id': 'a' if i < 5 else 'b'} for i in range(9)]
    out = replicas({'games': games, 'mean_score_equal_weight_games': 100/9}, roster)
    assert out['valid']
    assert out['instance_mean_score'] == pytest.approx(100/9)
    assert out['source_balanced_mean_score'] == 10
    assert out['games'][0]['zero_action_copies'] == 4
    assert out['games'][1]['copies_with_clears'] == 0
    assert out['games'][0]['mean_score'] == 20  # Never use best-of-five score 100.
    assert not replicas({'games': games, 'mean_score_equal_weight_games': 100/9}, roster[:-1])['valid']
    assert not replicas({'games': games, 'mean_score_equal_weight_games': 100/9}, roster+[roster[0]])['valid']


def test_speculation_subtracts_setup_and_splits_rounds_without_counting_drafts_as_output():
    rows = samples()
    for row, rounds in zip(rows, [20, 30, 50]):
        row['metrics'].update({'vllm:spec_decode_num_drafts_total': rounds,
                             'vllm:spec_decode_num_draft_tokens_total': 3*rounds,
                             'vllm:spec_decode_num_accepted_tokens_total': 1.5*rounds})
    out = phases(report(5), rows)
    assert out['valid']
    a, b = out['phases']['opening'], out['phases']['phase2']
    assert a['tokens']['generation'] == 50 and b['tokens']['generation'] == 100
    assert not any(k.startswith('_') for k in a['tokens'])
    assert a['speculation']['draft_rounds'] == 10
    assert b['speculation']['draft_rounds'] == 20
    assert a['speculation']['acceptance_fraction'] == .5
    assert a['speculation']['proposed_tokens_per_round'] == 3
    assert a['speculation']['accepted_tokens_per_round'] == 1.5
    del rows[1]['metrics']['vllm:spec_decode_num_draft_tokens_total']
    invalid = phases(report(5), rows)
    assert not invalid['valid']
    assert 'acceptance_fraction' not in invalid['phases']['opening']['speculation']
