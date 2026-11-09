"""Retrospective timing attribution; never changes admission or gameplay clocks."""

import math


def _merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _intersection(left, right):
    return _merged((max(a, c), min(b, d)) for a, b in left for c, d in right
                   if max(a, c) < min(b, d))


def _seconds(intervals):
    return sum(b - a for a, b in intervals)


def _time(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('non-finite or negative timestamp')
    return value


def _interval(start, end):
    a, b = _time(start), _time(end)
    if b < a:
        raise ValueError('reversed timing interval')
    return a, b


def _eligible(game, duration):
    events = game['admissions']
    if not events or events[0]['event'] != 'admit':
        raise ValueError('missing initial admission')
    intervals, start, previous = [], None, 0.0
    for event in events:
        t = _time(event['wall_seconds'])
        if t < previous or t > duration + .01:
            raise ValueError('invalid admission ordering or duration')
        previous = t
        t = min(t, duration)
        kind = event['event']
        if kind == 'admit':
            if start is not None:
                raise ValueError('admit while already eligible')
            start = t
        elif kind in ('park', 'finish'):
            if start is not None:
                intervals.append((start, t))
                start = None
        elif kind not in ('grant', 'progress'):
            raise ValueError('unknown admission event')
    if start is not None:
        intervals.append((start, duration))
    return intervals


def _attribute(eligible, waits, held, start, end):
    window = [(start, end)]
    active = _intersection(eligible, window)
    waiting = _intersection(waits, active)
    holding = _intersection(held, active)
    overlap = _intersection(waiting, holding)
    # Nested admissions can wait while an outer request still holds a slot.
    # Keep this explicit rather than counting those seconds twice as pure waiting.
    active_s, wait_s, held_s, overlap_s = map(_seconds, (active, waiting, holding, overlap))
    return {
        'eligible_seconds': active_s,
        'eligible_wait_without_slot_seconds': max(0.0, wait_s - overlap_s),
        'eligible_slot_held_seconds': held_s,
        'nested_wait_while_slot_held_seconds': overlap_s,
        'eligible_outside_admission_seconds': max(0.0, active_s - wait_s - held_s + overlap_s),
        'parked_slot_held_seconds': max(0.0, _seconds(_intersection(held, window)) - held_s),
    }


def admission_timing(report):
    """Attribute observed eligible wall time, including before each first clear.

    A held slot includes transport, backend queue/prefill/decode and response work;
    it is NOT GPU service time. Residual time includes tools and tokenization.
    Unions avoid double-counting nested requests. Missing timing is not zero work.
    """
    issues, rows = [], []
    try:
        duration = _time(report['wall_seconds'])
        switch = report['scheduler'].get('switch_wall_seconds')
        boundary = duration if switch is None else _time(switch)
        if duration <= 0 or boundary > duration:
            raise ValueError('invalid phase boundary')
        windows = {'opening': (0.0, boundary)}
        if switch is not None:
            windows['phase2'] = (boundary, duration)
        games = report['games']
        if not games:
            raise ValueError('missing games')
    except (KeyError, TypeError, ValueError) as exc:
        return {'valid': False, 'issues': [str(exc)]}
    for game in games:
        game_id = game['game_id']
        try:
            if (report['scheduler'].get('budget_policy') in ('deep-explore', 'depthwise', 'rank')
                    and not any(e['event'] == 'admit' for e in game['admissions'])):
                continue  # never reached a slot: no eligible time to attribute
            eligible = _eligible(game, duration)
            attempts = game['admission_timing']['attempts']
            if not attempts:
                raise ValueError('no admission attempts; timing unavailable')
            if len({a['attempt'] for a in attempts}) != len(attempts):
                raise ValueError('duplicate admission attempt')
            waits, held = [], []
            for attempt in attempts:
                waits.append(_interval(attempt['started_wall_seconds'],
                                       attempt['wait_ended_wall_seconds']))
                if 'admitted_wall_seconds' in attempt:
                    if abs(_time(attempt['admitted_wall_seconds']) - waits[-1][1]) > .01:
                        raise ValueError('wait end and admission do not reconcile')
                    held.append(_interval(attempt['admitted_wall_seconds'],
                                          attempt['ended_wall_seconds']))
            waits, held = _merged(waits), _merged(held)
            row = {'game_id': game_id, 'phases': {
                name: _attribute(eligible, waits, held, *window)
                for name, window in windows.items()}}
            clears = game.get('time_to_level', [])
            if clears:
                first = clears[0]
                wall = _time(first['wall_seconds'])
                active = _time(first['active_seconds'])
                if wall > duration + .01 or first['level'] != 1:
                    raise ValueError('invalid first-clear record')
                first_timing = _attribute(eligible, waits, held, 0.0, min(wall, duration))
                if abs(first_timing['eligible_seconds'] - active) > .02:
                    raise ValueError('first-clear active time does not reconcile')
                first_timing.update(wall_seconds=wall, recorded_active_seconds=active,
                                    wait_fraction_of_active=(
                                        first_timing['eligible_wait_without_slot_seconds'] / active
                                        if active else None))
                row['first_clear'] = first_timing
            rows.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f'{game_id}: {exc}')
    totals = {name: {key: sum(row['phases'][name][key] for row in rows)
                     for key in rows[0]['phases'][name]} for name in windows} if rows else {}
    return {'valid': not issues, 'issues': issues,
            'scope': 'Summed per-game eligible wall time; slot-held time is not GPU service time',
            'games_covered': len(rows), 'games_expected': len(games),
            'phase_totals': totals, 'games': rows}


def inference_timing(report):
    """Reconcile generation intervals with admission and split actual requests by phase.

    Older runs did not record these intervals. Never substitute slot-held or
    eligible time for missing inference measurements.
    """
    if report.get('scheduler', {}).get('budget_clock') not in ('inference', 'busy', 'normalized', 'tokens'):
        return {'available': False, 'valid': None,
                'reason': 'Historical run lacks the inference-budget clock'}
    issues, rows = [], []
    try:
        duration = _time(report['wall_seconds'])
        switch = report['scheduler'].get('switch_wall_seconds')
        boundary = duration if switch is None else _time(switch)
        if not 0 <= boundary <= duration or duration <= 0 or not report['games']:
            raise ValueError('Invalid trial duration or games')
        windows = {'opening': (0.0, boundary)}
        if switch is not None:
            windows['phase2'] = (boundary, duration)
    except (KeyError, TypeError, ValueError) as exc:
        return {'available': True, 'valid': False, 'issues': [str(exc)]}
    for game in report['games']:
        try:
            requests = game['inference_intervals']
            attempts = game['admission_timing']['attempts']
            held = [_interval(a['admitted_wall_seconds'], a['ended_wall_seconds'])
                    for a in attempts if 'admitted_wall_seconds' in a]
            intervals, summaries, interrupted = [], [], []
            for r in requests:
                start, end = _interval(r['started_wall_seconds'], r['ended_wall_seconds'])
                if (end > duration + .002 or (intervals and start < intervals[-1][1])
                        or not any(a - .002 <= start <= end <= b + .002 for a, b in held)):
                    raise ValueError('Inference interval outside a held admission, overlapping, or past deadline')
                intervals.append((start, end))
                if r['purpose'] == 'summary':
                    summaries.append((start, end))
                if r['status'] != 'completed':
                    interrupted.append((start, end))
            measured = _seconds(intervals)
            if abs(measured - _time(game['inference_seconds'])) > .002:
                raise ValueError('Inference total does not reconcile with request intervals')
            row = {'game_id': game['game_id'], 'requests': len(requests),
                   'inference_seconds': measured, 'phases': {}}
            for name, window in windows.items():
                elapsed = window[1] - window[0]
                seconds = _seconds(_intersection(intervals, [window]))
                row['phases'][name] = {
                    'inference_seconds': seconds,
                    'summary_seconds': _seconds(_intersection(summaries, [window])),
                    'interrupted_request_seconds': _seconds(_intersection(interrupted, [window])),
                    'request_wall_fraction': seconds / elapsed if elapsed else None,
                }
            clears = game.get('time_to_level', [])
            if clears:
                first = clears[0]
                seconds = _seconds(_intersection(intervals, [(0.0, _time(first['wall_seconds']))]))
                budget_seconds = seconds
                clock = report.get('scheduler', {}).get('budget_clock')
                if clock == 'normalized':
                    # Charged equivalent seconds of the requests completed by the clear, not latency.
                    budget_seconds = sum(float(c['cost_seconds']) for c in game.get('charges', [])
                                         if _time(c['wall_seconds']) <= _time(first['wall_seconds']) + .025)
                if clock == 'tokens':
                    # Token clock: completion tokens of the counted purposes by the clear, not seconds.
                    counted = report['scheduler']['token_budget']['purposes']
                    budget_seconds = float(sum(int(c['completion_tokens']) for c in game.get('charges', [])
                                               if c['purpose'] in counted
                                               and _time(c['wall_seconds']) <= _time(first['wall_seconds']) + .025))
                if clock in ('busy', 'normalized'):
                    # These clocks also charge tool executions (clock-recorded intervals).
                    tools = [_interval(t['started_wall_seconds'], t['ended_wall_seconds'])
                             for t in game.get('tool_intervals', [])]
                    budget_seconds += _seconds(_intersection(tools, [(0.0, _time(first['wall_seconds']))]))
                if abs(max(budget_seconds, 1.0) - _time(game['t1_seconds'])) > .025:
                    raise ValueError('First-clear inference disagrees with T1')
                row['first_clear_inference_seconds'] = seconds
            rows.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f'{game.get("game_id", "<missing>")}: {exc}')
    totals = {}
    for name, (start, end) in windows.items():
        total = sum(r['phases'][name]['inference_seconds'] for r in rows)
        totals[name] = {
            'inference_seconds': total,
            'summary_seconds': sum(r['phases'][name]['summary_seconds'] for r in rows),
            'interrupted_request_seconds': sum(r['phases'][name]['interrupted_request_seconds'] for r in rows),
            'mean_requests_in_flight': total / (end - start) if end > start else None,
        }
    return {'available': True, 'valid': not issues, 'issues': issues,
            'scope': 'Admitted generation latency, including backend/transport; not exclusive GPU time',
            'games_covered': len(rows), 'games_expected': len(report['games']),
            'phase_totals': totals, 'games': rows}
