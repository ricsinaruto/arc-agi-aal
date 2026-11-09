"""Post-run phase metrics: python -m arena.analysis TRIAL_DIR [TRIAL_DIR ...].

Reads saved arena.json, telemetry.jsonl and optional scorecard.json; never submits runs.
Backend counters retain their native scope, which differs between vLLM and SGLang.
"""
import argparse
import bisect
import json
import math
from pathlib import Path

from arena.scheduler_audit import audit_scheduler
from arena.timing_analysis import admission_timing, inference_timing


def phases(report, samples):
    if len(samples) < 2:
        return {'valid': False, 'reason': 'At least two telemetry samples required'}
    sglang = 'values' in samples[0]
    field = 'values' if sglang else 'metrics'
    prefix = 'sglang:' if sglang else 'vllm:'
    # Trial takes the first counter sample before releasing the gameplay barrier,
    # then rebases the monitor clock. Its recorded setup-relative time is not play.
    times = [0.0, *[float(s['elapsed_seconds']) for s in samples[1:]]]
    if any(not math.isfinite(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        return {'valid': False, 'reason': 'Telemetry times must increase after the pre-release sample'}
    duration = float(report['wall_seconds'])
    switch = report['scheduler'].get('switch_wall_seconds')
    boundary = duration if switch is None else float(switch)
    if not 0 <= boundary <= duration or not math.isfinite(duration) or duration <= 0:
        return {'valid': False, 'reason': 'Invalid gameplay/phase timing'}
    names = {'generation': prefix + 'generation_tokens_total'}
    if sglang:
        names.update(prompt='sglang:prompt_tokens_total', cached_prompt='sglang:cached_tokens_total')
    else:
        names.update(computed_prompt='vllm:prompt_tokens_by_source_total:local_compute',
                     cached_prompt='vllm:prompt_tokens_by_source_total:local_cache_hit')
        speculation_names = {
            '_draft_rounds': 'vllm:spec_decode_num_drafts_total',
            '_draft_tokens': 'vllm:spec_decode_num_draft_tokens_total',
            '_accepted_draft_tokens': 'vllm:spec_decode_num_accepted_tokens_total',
        }
        if any(name in s.get(field, {}) for s in samples for name in speculation_names.values()):
            names.update(speculation_names)
    issues, counters = [], {}
    for label, name in names.items():
        values = [s.get(field, {}).get(name) for s in samples]
        if any(v is None or not math.isfinite(v) for v in values):
            issues.append(f'{label}: missing/non-finite counter; not treated as zero')
        elif any(a > b for a, b in zip(values, values[1:])):
            issues.append(f'{label}: counter reset; phase totals unavailable')
        else:
            counters[label] = values

    def interpolate(values, t):
        i = bisect.bisect_right(times, t) - 1
        if i == len(times) - 1:
            return values[i]
        weight = (t - times[i]) / (times[i+1] - times[i])
        return values[i] + weight * (values[i+1] - values[i])

    result = {}
    windows = [('opening', 0.0, boundary)]
    if switch is not None:
        windows.append(('phase2', boundary, duration))
    for label, start, end in windows:
        covered_end = min(end, times[-1])
        covered = max(0, covered_end - start)
        counts = {key: interpolate(v, covered_end) - interpolate(v, start)
                  for key, v in counters.items()} if covered else {}
        spec = {key.removeprefix('_'): counts.pop(key) for key in list(counts) if key.startswith('_')}
        if {'draft_rounds', 'draft_tokens', 'accepted_draft_tokens'} <= spec.keys():
            rounds, drafted, accepted = (spec[k] for k in ('draft_rounds', 'draft_tokens', 'accepted_draft_tokens'))
            if accepted > drafted:
                issues.append(f'{label}: accepted draft tokens exceed proposed tokens')
            spec.update(acceptance_fraction=accepted/drafted if drafted else None,
                        proposed_tokens_per_round=drafted/rounds if rounds else None,
                        accepted_tokens_per_round=accepted/rounds if rounds else None)
        if not sglang and {'computed_prompt', 'cached_prompt'} <= counts.keys():
            counts['prompt'] = counts['computed_prompt'] + counts['cached_prompt']
        cached_share = None
        if counts.get('prompt', 0) > 0 and 'cached_prompt' in counts:
            if 0 <= counts['cached_prompt'] <= counts['prompt']:
                cached_share = counts['cached_prompt'] / counts['prompt']
            else:
                issues.append(f'{label}: cached prompt count outside total prompt count')
        selected = [s.get(field, {}) for t, s in zip(times, samples) if start <= t <= end]
        peaks, means = {}, {}
        for key, metric in {'running': prefix + ('num_running_reqs' if sglang else 'num_requests_running'),
                            'cache_fraction': prefix + ('token_usage' if sglang else 'kv_cache_usage_perc')}.items():
            values = [s[metric] for s in selected if metric in s and math.isfinite(s[metric])]
            peaks[key] = max(values) if values else None
            full = [s.get(field, {}).get(metric) for s in samples]
            means[key] = None
            if covered and all(v is not None and math.isfinite(v) for v in full):
                knots = [start, *[t for t in times if start < t < covered_end], covered_end]
                means[key] = sum((interpolate(full, a) + interpolate(full, b)) * (b-a) / 2
                                 for a, b in zip(knots, knots[1:])) / covered
        result[label] = {'wall_seconds': end-start, 'covered_seconds': covered,
                         'coverage_fraction': covered/(end-start) if end > start else None,
                         'tokens': counts, 'tokens_per_covered_second':
                         {k: v/covered for k, v in counts.items()} if covered else {},
                         'cached_share': cached_share, 'observed_peaks': peaks,
                         'estimated_time_weighted_means': means}
        if spec:
            result[label]['speculation'] = spec
    if times[-1] < duration - .1:
        issues.append('Telemetry ends before gameplay; rates use covered seconds only')
    return {'valid': not issues, 'issues': issues, 'backend': 'sglang' if sglang else 'vllm',
            'generation_counter_scope': 'finished_requests' if sglang else 'decode_tokens_including_interrupted_requests',
            'method': 'First sample rebased to barrier zero; cumulative counters interpolated at phase boundaries; no extrapolation',
            'max_sample_gap_seconds': max(b-a for a, b in zip(times, times[1:])),
            'phases': result}


def throughput(report, phase_metrics):
    """Aggregate token throughput and clears per token for throughput arms (fixed-2x-mtp0-wide, 27 September).

    Two token scopes: ``engine`` is vLLM's generation counter over the sampled gameplay (every decoded token,
    including replies cut off at the wall), ``completed`` sums the completion tokens of finished replies in the
    game rows. Clears per 10K tokens use the completed scope (the token-efficiency convention); per-phase
    rates and streams in flight come from the telemetry phases (time-weighted mean and peak running requests).
    """
    engine = report.get('engine') or {}
    wall = float(report['wall_seconds'])
    levels, completed = report.get('levels'), report.get('completed_reply_tokens')
    sampled = engine.get('sampled_seconds')
    generated = engine.get('accepted_generation_tokens') if engine.get('valid') else None
    gate = report.get('gate') or {}
    choice = report.get('window_choice') or {}

    def per_10k(clears, tokens):
        return clears * 1e4 / tokens if clears is not None and tokens else None

    result = {
        'window': choice.get('window'), 'focus_fit': choice.get('focus_fit'),
        'phase2_limit': (report.get('scheduler') or {}).get('active_limit'),
        'engine_generation_tokens': generated,
        'engine_generation_tokens_per_hour': generated * 3600 / sampled if generated is not None and sampled else None,
        'completed_reply_tokens': completed,
        'completed_reply_tokens_per_hour': completed * 3600 / wall if completed is not None and wall > 0 else None,
        'clears': levels,
        'clears_per_10k_completed_tokens': per_10k(levels, completed),
        'clears_per_10k_engine_tokens': per_10k(levels, generated),
        'preemptions': engine.get('preemptions'),
        'peak_running_requests': engine.get('peak_running_requests'),
        'gate': {k: gate.get(k) for k in ('capacity', 'units', 'slots', 'peak_reserved', 'peak_inflight',
                                          'peak_waiting')},
        'phases': {},
    }
    for label, phase in ((phase_metrics or {}).get('phases') or {}).items():
        rates = phase.get('tokens_per_covered_second') or {}
        result['phases'][label] = {
            'wall_seconds': phase.get('wall_seconds'), 'covered_seconds': phase.get('covered_seconds'),
            'generation_tokens': (phase.get('tokens') or {}).get('generation'),
            'generation_tokens_per_second': rates.get('generation'),
            'mean_running_requests': (phase.get('estimated_time_weighted_means') or {}).get('running'),
            'peak_running_requests': (phase.get('observed_peaks') or {}).get('running'),
        }
    return result


def analyze(folder):
    folder = Path(folder)
    report = json.loads((folder/'arena.json').read_text())
    samples = [json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines() if line]
    phase_metrics = phases(report, samples)
    result = {'source': str(folder), 'trial_status': report['status'],
            'score': report.get('mean_score_equal_weight_25_games'),
            'clears': report.get('levels'), 'actions': report.get('actions'),
            'phase2_limit': report['scheduler'].get('active_limit'),
            'capacity_estimate': report['window_choice'].get('phase2'),
            'throughput': throughput(report, phase_metrics),
            'task_errors': report.get('task_errors'), 'phase_metrics': phase_metrics,
            'allocation': allocation(report), 'admission_timing': admission_timing(report),
            'inference_timing': inference_timing(report),
            'scheduler_audit': audit_scheduler(report)}
    if (folder/'scorecard.json').exists():
        result['score_curve'] = score_curve(report, json.loads((folder/'scorecard.json').read_text()))
    if (folder/'public-roster.json').exists():
        result['replicas'] = replicas(report, json.loads((folder/'public-roster.json').read_text()))
    return result


def replicas(report, roster):
    """Separate the actual instance mean from equally weighted source-game means.

    Cyclic 110-instance previews contain five copies of ten games and four copies
    of fifteen. Every copy, including zero-action copies, stays in the denominator.
    Grouping is controller-side analysis only; no cross-game evidence is exposed.
    """
    try:
        games = {g['game_id']: g for g in report['games']}
        mapping = {r['game_id']: r['source_game_id'] for r in roster}
        if (not games or len(games) != len(report['games']) or len(mapping) != len(roster)
                or games.keys() != mapping.keys()):
            raise ValueError('Replica roster and report coverage do not reconcile')
        groups = {}
        for game_id, game in games.items():
            score = float(game['score'])
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError(f'{game_id}: invalid score')
            groups.setdefault(mapping[game_id], []).append(game)
        rows = []
        for source, copies in groups.items():
            scores = [float(g['score']) for g in copies]
            rows.append({'source_game_id': source, 'copies': len(copies),
                         'mean_score': sum(scores)/len(scores), 'min_score': min(scores),
                         'max_score': max(scores), 'clears': sum(g['levels'] for g in copies),
                         'actions': sum(g['actions'] for g in copies),
                         'zero_action_copies': sum(g['actions'] == 0 for g in copies),
                         'copies_with_clears': sum(g['levels'] > 0 for g in copies),
                         'game_ids': [g['game_id'] for g in copies]})
        actual = sum(float(g['score']) for g in games.values())/len(games)
        expected = report.get('mean_score_equal_weight_games', report.get('mean_score_equal_weight_25_games'))
        if not math.isclose(actual, expected, abs_tol=1e-7):
            raise ValueError('Instance score does not reconcile with report')
        return {'valid': True, 'instances': len(games), 'source_games': len(groups),
                'instance_mean_score': actual,
                'source_balanced_mean_score': sum(r['mean_score'] for r in rows)/len(rows),
                'scope': 'All copies retained; source means are not best-of-copy scores or independent experiment repetitions',
                'games': rows}
    except (KeyError, TypeError, ValueError) as exc:
        return {'valid': False, 'reason': str(exc)}


def score_curve(report, scorecard, checkpoints=(900, 1800, 3600)):
    """Reconstruct single-run score prefixes from immutable completed-level costs.

    Reject ambiguous/missing evidence instead of assuming every clear earns 100.
    Scores use the SDK's linear level weights and completed-weight cap. This is
    retrospective analysis, not a score signal exposed to the agent.
    """
    try:
        games = {g['game_id']: g for g in report['games']}
        envs = {e['id']: e for e in scorecard['environments']}
        if (not games or len(games) != len(report['games']) or games.keys() != envs.keys()
                or len(envs) != len(scorecard['environments'])):
            raise ValueError('Game coverage or unique IDs do not reconcile')
        duration = float(report['wall_seconds'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid gameplay duration')
        evidence = {}
        event_times = {0.0, duration}
        for game_id, game in games.items():
            env = envs[game_id]
            if len(env['runs']) != 1:
                raise ValueError(f'{game_id}: multiple runs cannot be reconstructed from one clear history')
            run = env['runs'][0]
            scores = run['level_scores']
            clears = game['time_to_level']
            if (not scores or len(scores) != game['total_levels']
                    or len(clears) != game['levels'] or len(clears) != run['levels_completed']
                    or len(clears) > len(scores)
                    or any(not math.isfinite(s) or not 0 <= s <= 115 for s in scores)
                    or any(s != 0 for s in scores[len(clears):])):
                raise ValueError(f'{game_id}: level evidence does not reconcile')
            times = [float(c['wall_seconds']) for c in clears]
            if ([c['level'] for c in clears] != list(range(1, len(clears)+1))
                    or any(not math.isfinite(t) or t < 0 for t in times)
                    or any(a > b for a, b in zip(times, times[1:]))):
                raise ValueError(f'{game_id}: invalid clear ordering/timing')
            evidence[game_id] = (scores, times)
            event_times.update(times)

        def point(t):
            per_game, clears = {}, 0
            for game_id, (scores, times) in evidence.items():
                n = bisect.bisect_right(times, t)
                denominator = len(scores)*(len(scores)+1)/2
                raw = sum((i+1)*s for i, s in enumerate(scores[:n]))/denominator
                cap = 100*sum(i+1 for i, s in enumerate(scores[:n]) if s > 0)/denominator
                per_game[game_id] = min(raw, cap)
                clears += n
            return {'wall_seconds': t, 'score': sum(per_game.values())/len(games),
                    'clears': clears, 'per_game_score': per_game}

        end = max(event_times)
        final = point(end)
        for game_id, value in final['per_game_score'].items():
            if not math.isclose(value, envs[game_id]['score'], abs_tol=1e-7):
                raise ValueError(f'{game_id}: final score does not reconcile')
        expected = report.get('mean_score_equal_weight_games', report.get('mean_score_equal_weight_25_games'))
        if not math.isclose(final['score'], expected, abs_tol=1e-7):
            raise ValueError('Final suite score does not reconcile')
        targets = {float(t) for t in checkpoints if 0 <= t <= end}
        targets.add(end)
        return {'valid': True, 'scope': 'Retrospective single-run level-cost reconstruction; equal game weights',
                'post_report_end_clears': sum(t > duration for _, times in evidence.values() for t in times),
                'checkpoints': [point(t) for t in sorted(targets)],
                'events': [point(t) for t in sorted(event_times)]}
    except (KeyError, TypeError, ValueError) as exc:
        return {'valid': False, 'reason': str(exc)}


def allocation(report):
    """Reconstruct eligibility, not GPU service, from saved admission events.

    A parked request can finish and act before reaching its next admission gate.
    Such clears are reported separately; their work is not assigned to a GPU phase.
    Rounded event timestamps cannot resolve ordering of a clear at a boundary.
    """
    duration = float(report['wall_seconds'])
    switch = report['scheduler'].get('switch_wall_seconds')
    boundary = duration if switch is None else float(switch)
    if not math.isfinite(duration) or duration <= 0 or not 0 <= boundary <= duration:
        return {'valid': False, 'issues': ['Invalid gameplay/phase timing']}
    windows = {'opening': (0.0, boundary)}
    if switch is not None:
        windows['phase2'] = (boundary, duration)
    rows, issues = [], []
    games = report.get('games')
    if not games:
        return {'valid': False, 'issues': ['No per-game admission records']}
    for game in games:
        game_id = game['game_id']
        events = game.get('admissions', [])
        if not events or events[0].get('event') != 'admit':
            issues.append(f'{game_id}: missing initial admission; allocation unavailable')
            continue
        times = [float(e['wall_seconds']) for e in events]
        if (any(not math.isfinite(t) or t < 0 or t > duration + .01 for t in times)
                or any(a > b for a, b in zip(times, times[1:]))):
            issues.append(f'{game_id}: invalid admission times; allocation unavailable')
            continue
        active, start, intervals = False, None, []
        long_grants = []
        for event, raw_time in zip(events, times):
            t = min(raw_time, duration)
            kind = event['event']
            if kind == 'admit' and not active:
                active, start = True, t
            elif kind in ('park', 'finish') and active:
                intervals.append((start, t))
                active, start = False, None
            elif kind not in ('admit', 'park', 'finish', 'grant', 'progress'):
                issues.append(f'{game_id}: unknown admission event {kind}')
            budget = event.get('budget')
            # Seconds budgets only: a token budget is not comparable with the wall seconds left.
            if (switch is not None and t >= boundary and kind in ('admit', 'grant')
                    and report['scheduler'].get('budget_clock') != 'tokens'
                    and budget is not None and float(budget) > duration - t):
                long_grants.append({'wall_seconds': t, 'budget_seconds': budget,
                                    'wall_remaining_seconds': duration-t,
                                    'reason': event.get('reason')})
        if active:
            intervals.append((start, duration))
        eligible = {name: sum(max(0.0, min(b, end)-max(a, begin))
                             for a, b in intervals)
                    for name, (begin, end) in windows.items()}
        clears, ambiguous = [], []
        boundaries = [t for e, t in zip(events, times) if e['event'] in ('admit', 'park', 'finish')]
        for clear in game.get('time_to_level', []):
            t = float(clear['wall_seconds'])
            if any(abs(t-b) <= .002 for b in boundaries):
                ambiguous.append(clear)
            elif not any(a <= t < b for a, b in intervals):
                clears.append(clear)
        rows.append({'game_id': game_id, 'eligible_seconds': eligible,
                     'clears_outside_eligibility': clears, 'boundary_ambiguous_clears': ambiguous,
                     'grants_exceeding_wall_remaining': long_grants})
    totals = {name: sum(r['eligible_seconds'][name] for r in rows) for name in windows}
    return {'valid': not issues, 'issues': issues,
            'scope': 'Scheduler eligibility wall time; not GPU compute, inference service, or useful progress',
            'games_covered': len(rows), 'games_expected': len(games),
            'mean_eligible_games': {name: total/(windows[name][1]-windows[name][0])
                                    if windows[name][1] > windows[name][0] else None
                                    for name, total in totals.items()},
            'games': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trials', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps([analyze(path) for path in args.trials], indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
