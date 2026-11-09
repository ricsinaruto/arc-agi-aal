"""Independent checks of recorded arena clocks and the documented grant arithmetic."""

import math
import statistics

from .timing_analysis import _eligible, _intersection, _seconds


def audit_scheduler(report):
    issues, warnings = [], []
    checks = {'games': 0, 'admission_clocks': 0, 'clears': 0, 'grants': 0, 'budget_parks': 0}
    tolerance = .025  # Stored timestamps, T1 and grants are rounded to milliseconds.
    try:
        duration = float(report['wall_seconds'])
        schedule = report['scheduler']
        games = report['games']
        policy = schedule.get('budget_clock', 'eligible_wall')
        if policy not in ('inference', 'eligible_wall', 'busy', 'normalized', 'tokens'):
            raise ValueError('Unknown budget clock')
        # Token clock (29 September): the budget unit is the completion tokens of the counted request purposes;
        # the grant floor is recorded with the run (30 s on the time clocks).
        token_budget = schedule.get('token_budget')
        if (policy == 'tokens') != (token_budget is not None):
            raise ValueError('token_budget settings must accompany the token clock only')
        floor = 30.0
        purposes = ()
        if token_budget is not None:
            floor = float(token_budget['floor_tokens'])
            purposes = tuple(token_budget['purposes'])
            if not math.isfinite(floor) or floor < 1 or not purposes:
                raise ValueError('invalid token budget settings')
        normalization = schedule.get('normalization')
        if (policy == 'normalized') != (normalization is not None):
            raise ValueError('normalization parameters must accompany the normalized clock only')
        if normalization is not None:
            from .clock import Normalization

            normalization = Normalization(**normalization)
        budget_policy = schedule.get('budget_policy', 'incremental')
        if budget_policy not in ('incremental', 'fixed-2x', 'index', 'commit', 'deep-explore', 'depthwise', 'rank', 'blocks',
                                 'priority'):
            raise ValueError('Unknown budget policy')
        # Three-block schedule (hero-v2-blocks): the recorded settings (level-0 budget, block length, tenure unit).
        blocks_cfg = (schedule.get('blocks') or {}).get('settings') or {}
        if budget_policy == 'blocks' and (policy != 'tokens' or not blocks_cfg):
            raise ValueError('the blocks policy needs its settings and the token clock')
        # Priority gate (arena.priority): the recorded settings and every game's context evictions (handover points).
        priority_info = schedule.get('priority') or {}
        if budget_policy == 'priority' and not priority_info.get('settings'):
            raise ValueError('the priority policy needs its settings')
        # Depthwise (fixed-2x-depthwise): the re-admission rule and multiple recorded with the run.
        depthwise = schedule.get('depthwise') or {}
        depthwise_regrant = (depthwise.get('settings') or {}).get('regrant', 'spent')
        if budget_policy == 'depthwise' and depthwise_regrant not in ('spent', 'full'):
            raise ValueError('invalid depthwise regrant rule')
        # Clipped budget unit (fixed-2x-token-budget): T1 clipped to [low, high] x the median T1 of
        # the games whose first clear came no later than the grant.
        t1_clip = schedule.get('t1_clip')
        if t1_clip is not None and (budget_policy != 'fixed-2x' or len(t1_clip) != 2
                                    or not 0 < float(t1_clip[0]) <= 1 <= float(t1_clip[1])):
            raise ValueError('invalid T1 clip')
        first_clear_t1 = sorted(
            (float(g['time_to_level'][0]['wall_seconds']), float(g['t1_seconds']))
            for g in games if g.get('t1_seconds') is not None and g.get('time_to_level'))

        def unit(t1, wall):
            teff = t1 if t1 is None or t1_ceiling is None else min(t1, float(t1_ceiling))
            if t1_clip is None or teff is None:
                return teff
            known = [value for cleared, value in first_clear_t1 if cleared <= wall + tolerance]
            median = statistics.median(known)
            return min(max(teff, float(t1_clip[0]) * median), float(t1_clip[1]) * median)
        pool = schedule.get('audition_pool')
        pool_time_limit = schedule.get('pool_time_limit_seconds')
        if pool is not None and (budget_policy != 'fixed-2x' or type(pool) is not int or pool < 1):
            raise ValueError('invalid audition pool')
        t1_ceiling = schedule.get('t1_cap_seconds')  # fixed-2x T1 cap; distinct from the renewable multiple below
        if t1_ceiling is not None and (budget_policy != 'fixed-2x' or not math.isfinite(float(t1_ceiling)) or float(t1_ceiling) < 1):
            raise ValueError('invalid T1 cap')
        multiple = float(schedule.get('budget_multiple') or 2.0)  # fixed-2x level budget multiple * L * T1
        if not math.isfinite(multiple) or multiple <= 0:
            raise ValueError('invalid budget multiple')
        if not games or not math.isfinite(duration) or duration <= 0:
            raise ValueError('missing games or invalid duration')
        cap = schedule.get('max_grant_seconds')
        if cap is not None and (not math.isfinite(cap) or cap < 30):
            raise ValueError('invalid grant cap')
        # Renewable T1-relative cap and the re-admission multiple (24 September variants).
        t1_cap = schedule.get('max_grant_t1_multiple')
        if t1_cap is not None and (not math.isfinite(float(t1_cap)) or float(t1_cap) < 1):
            raise ValueError('invalid T1-relative grant cap')
        regrant = schedule.get('regrant_multiple', 1.5)
        if not math.isfinite(float(regrant)) or float(regrant) < 1:
            raise ValueError('invalid regrant multiple')
        switch = schedule.get('switch_wall_seconds')
        if switch is not None and (not math.isfinite(float(switch)) or not 0 <= float(switch) <= duration):
            raise ValueError('invalid opening switch timestamp')
        if len({g['game_id'] for g in games}) != len(games):
            raise ValueError('duplicate game records')
        limit = int(schedule['active_limit'])
        trigger = int(schedule.get('opening_switch_level1_games', 2))
        if limit < 1 or trigger < 1:
            raise ValueError('invalid scheduling limits')
        if pool is not None:
            limit *= pool  # focus games plus pool games per free slot
        reopen = schedule.get('reopen_phase1')
        if reopen is not None:
            limit = len(games)  # a reopened phase 1 runs its whole tie group
        target = report.get('config', {}).get('gameplay_seconds')
        if target is not None and (not math.isfinite(float(target)) or float(target) <= 0):
            raise ValueError('invalid gameplay target')
    except (KeyError, TypeError, ValueError) as exc:
        return {'valid': False, 'issues': [str(exc)], 'warnings': [], 'checks': checks}

    def equal(label, actual, expected):
        if actual is None or not math.isfinite(float(actual)) or abs(float(actual) - expected) > tolerance:
            issues.append(f'{label}: recorded {actual}, expected {expected:.6f}')

    def capped(value, t1=None):
        if cap is not None:
            value = min(value, cap)
        if t1_cap is not None and t1 is not None:
            value = min(value, max(float(t1_cap) * t1, floor))
        return value

    first_times, intervals_by_game = [], []
    for game in games:
        game_id = game.get('game_id', '<missing>')
        try:
            # Deep/explore and depthwise: with 110 games and a few slots some games never reach a slot (their only
            # event is the wall-end finish).
            unadmitted = (budget_policy in ('deep-explore', 'depthwise', 'rank', 'blocks', 'priority')
                          and not any(e['event'] == 'admit' for e in game['admissions']))
            intervals = [] if unadmitted else _eligible(game, duration)
            intervals_by_game.append(intervals)
            inference = []
            charged = []  # normalized clock: (wall, cost) per completed request; token clock: (wall, tokens)

            def charge_wall(charge):
                clipped = float(charge['wall_seconds'])
                observed = float(charge.get('observed_wall_seconds', clipped))
                # Report duration is captured before worker drain, so a late
                # response's observation time may also exceed that duration.
                if (not all(math.isfinite(v) and v >= 0 for v in (clipped, observed))
                        or clipped > observed + tolerance):
                    raise ValueError('Invalid charge observation time')
                return observed

            if policy == 'tokens':
                total = 0
                for charge in game.get('charges', []):
                    if charge['purpose'] in purposes:
                        charged.append((charge_wall(charge), int(charge['completion_tokens'])))
                        total += int(charge['completion_tokens'])
                equal(f'{game_id}: budget total', game['budget_seconds'], float(total))
            if policy == 'normalized':
                total = 0.0
                for charge in game.get('charges', []):
                    cost = normalization.cost(charge['prompt_tokens'], charge['completion_tokens'])
                    equal(f'{game_id}: charge at {charge["wall_seconds"]}', charge['cost_seconds'], cost)
                    charged.append((charge_wall(charge), cost))
                    total += cost
                tools = []
                for call in game.get('tool_intervals', []):
                    start, end = float(call['started_wall_seconds']), float(call['ended_wall_seconds'])
                    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start <= end <= duration + tolerance:
                        raise ValueError('Invalid tool intervals')
                    tools.append((start, end))
                equal(f'{game_id}: tool total', game.get('tool_seconds'), _seconds(tools))
                equal(f'{game_id}: budget total', game['budget_seconds'], total + _seconds(tools))
                inference = tools
            if policy in ('inference', 'busy'):
                previous_end = 0.0
                for request in game['inference_intervals']:
                    start, end = float(request['started_wall_seconds']), float(request['ended_wall_seconds'])
                    if not all(math.isfinite(v) for v in (start, end)) or not previous_end <= start <= end <= duration + tolerance:
                        raise ValueError('Invalid or overlapping inference intervals')
                    previous_end = end
                    inference.append((start, end))
                equal(f'{game_id}: inference total', game['inference_seconds'], _seconds(inference))
                if policy == 'busy':
                    # Tool executions never overlap the game's own requests; the busy clock is their union.
                    tools = []
                    for call in game.get('tool_intervals', []):
                        start, end = float(call['started_wall_seconds']), float(call['ended_wall_seconds'])
                        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start <= end <= duration + tolerance:
                            raise ValueError('Invalid tool intervals')
                        tools.append((start, end))
                    overlap = _seconds(_intersection(tools, inference))
                    if overlap > tolerance:
                        warnings.append(f'{game_id}: tool intervals overlap inference by {overlap:.3f}s')
                    equal(f'{game_id}: tool total', game.get('tool_seconds'), _seconds(tools))
                    inference = sorted(inference + tools)
                equal(f'{game_id}: budget total', game['budget_seconds'], _seconds(inference))
            budget_intervals = inference if policy in ('inference', 'busy', 'normalized') else intervals
            budget_key = 'budget_seconds' if policy in ('inference', 'busy', 'normalized', 'tokens') else 'active_seconds'

            def budget_at(wall):
                if policy == 'tokens':
                    return float(sum(tokens for at, tokens in charged if at <= wall))
                charges = sum(cost for at, cost in charged if at <= wall + tolerance) if policy == 'normalized' else 0.0
                return charges + _seconds(_intersection(budget_intervals, [(0.0, wall)]))

            def equal_budget(label, actual, wall):
                # Token clock: a reply charged in the same millisecond as the event may land on either side of it.
                if policy == 'tokens' and actual is not None and math.isfinite(float(actual)):
                    after = float(sum(tokens for at, tokens in charged if at < wall + 1e-3))
                    before = float(sum(tokens for at, tokens in charged if at < wall - 1e-3))
                    if any(abs(float(actual) - v) <= tolerance for v in (before, budget_at(wall), after)):
                        return
                equal(label, actual, budget_at(wall))
            clear_records = game['time_to_level']
            clear_active = {0: 0.0}
            previous_wall = 0.0
            for expected_level, clear in enumerate(clear_records, 1):
                if clear['level'] != expected_level:
                    raise ValueError('nonconsecutive level records')
                wall = float(clear['wall_seconds'])
                if not math.isfinite(wall) or wall < previous_wall or wall > duration + tolerance:
                    raise ValueError('invalid clear ordering/duration')
                previous_wall = wall
                active = _seconds(_intersection(intervals, [(0.0, wall)]))
                equal(f'{game_id}: level {expected_level} active clock', clear['active_seconds'], active)
                equal_budget(f'{game_id}: level {expected_level} budget clock', clear[budget_key], wall)
                clear_active[expected_level] = float(clear[budget_key])
                checks['clears'] += 1
            t1 = max(clear_active[1], 1.0) if clear_records else None
            if t1 is not None:
                equal(f'{game_id}: T1', game['t1_seconds'], t1)
                first_times.append(float(clear_records[0]['wall_seconds']))
            elif game.get('t1_seconds') is not None:
                raise ValueError('T1 without a first clear')
            last_grant = None
            blocks_first = None  # blocks: the game's first admission
            block_len = float(blocks_cfg.get('block_seconds') or 0.0) or 1.0
            first_park = None  # fixed-2x: a level-0 game's re-admission grant is its spend at its first park
            priority_admits = priority_handovers = 0  # priority: admissions so far; handovers (parks + kept grants)
            for event in game['admissions']:
                wall, active = float(event['wall_seconds']), float(event['active_seconds'])
                equal(f'{game_id}: {event["event"]} active clock', active,
                      _seconds(_intersection(intervals, [(0.0, wall)])))
                checks['admission_clocks'] += 1
                budget_time = float(event[budget_key])
                equal_budget(f'{game_id}: event budget clock', budget_time, wall)
                reason, kind = event['reason'], event['event']
                level = int(event['levels'])
                if level not in clear_active:
                    raise ValueError('admission references an unrecorded level')
                spent = max(0.0, budget_time - clear_active[level])
                budget = event.get('budget')
                if kind == 'admit' and reason == 'opening_start':
                    equal(f'{game_id}: release time', wall, 0.0)
                    if budget is not None:
                        raise ValueError('opening starts with a level budget')
                elif kind == 'admit' and reason in ('reopen', 'reopen_resume'):
                    if reopen is None:
                        raise ValueError('reopen admission without the reopen policy')
                    if reason == 'reopen' and budget is not None:
                        raise ValueError('a reopened phase-1 admission carries no budget')
                    checks['reopen_admissions'] = checks.get('reopen_admissions', 0) + (reason == 'reopen')
                elif kind == 'grant' and reason == 'reopen_clear':
                    if reopen is None or not level or t1 is None:
                        raise ValueError('reopen promotion needs the reopen policy and a clear')
                    teff = t1 if t1_ceiling is None else min(t1, float(t1_ceiling))
                    equal(f'{game_id}: reopen_clear grant at {wall}', budget, multiple * (level + 1) * teff)
                    last_grant = (budget_time, float(budget))
                    checks['grants'] += 1
                elif kind == 'park' and reason in ('reopen_close', 'reopen_hold'):
                    if reopen is None:
                        raise ValueError('reopen park without the reopen policy')
                elif kind == 'progress':
                    pass
                elif kind == 'admit' and reason == 'pool':
                    if pool is None or level:
                        raise ValueError('pool admission needs the pool policy and no clear')
                    if pool_time_limit is None:
                        if budget is not None:
                            raise ValueError('an untimed pool admission carries no budget')
                    else:
                        equal(f'{game_id}: pool time limit at {wall}', budget, float(pool_time_limit))
                    checks['pool_admissions'] = checks.get('pool_admissions', 0) + 1
                elif kind == 'grant' and reason == 'pool_clear':
                    if pool is None or not level or t1 is None:
                        raise ValueError('pool promotion needs the pool policy and a clear')
                    teff = t1 if t1_ceiling is None else min(t1, float(t1_ceiling))
                    equal(f'{game_id}: pool_clear grant at {wall}', budget, multiple * (level + 1) * teff)
                    last_grant = (budget_time, float(budget))
                    checks['grants'] += 1
                elif kind == 'park' and reason in ('pool_close', 'pool_stall', 'pool_limit', 'pool_timeout'):
                    if pool is None:
                        raise ValueError('pool park without the pool policy')
                elif kind in ('admit', 'grant') and budget_policy == 'index':
                    # Hazard index: admissions and the opening-switch keep carry no budget.
                    if budget is not None or reason not in ('index', 'index_fill', 'opening_switch'):
                        raise ValueError(f'index admission {reason} with budget {budget}')
                    checks['index_admissions'] = checks.get('index_admissions', 0) + 1
                elif kind in ('admit', 'grant') and budget_policy == 'rank':
                    # Continuous ranking (hero-v2-rank): admissions at the release, fills and swaps carry no budget.
                    if kind != 'admit' or budget is not None or reason not in ('rank_start', 'rank_fill', 'rank'):
                        raise ValueError(f'rank {kind} {reason} with budget {budget}')
                    if reason == 'rank_start' and abs(wall - float(switch or 0.0)) > 1.1:
                        raise ValueError('rank_start after the release')
                    checks['rank_admissions'] = checks.get('rank_admissions', 0) + 1
                elif kind == 'park' and reason == 'rank' and budget_policy != 'rank':
                    raise ValueError('rank park without the rank policy')
                elif kind in ('admit', 'grant') and budget_policy == 'blocks':
                    # Three-block schedule: admissions carry no budget; only an old refill re-admits a game.
                    if kind != 'admit' or budget is not None or reason not in ('block_start', 'filler', 'old_refill', 'retired_continue'):
                        raise ValueError(f'blocks {kind} {reason} with budget {budget}')
                    if blocks_first is None:
                        if reason in ('old_refill', 'retired_continue'):
                            raise ValueError(f'{reason} of a game that never ran')
                        blocks_first = wall
                    elif reason not in ('old_refill', 'retired_continue'):
                        raise ValueError(f'blocks re-admission {reason} (only old_refill and retired_continue re-admit)')
                    if reason == 'block_start' and wall + tolerance < int(event.get('home_block') or 0) * block_len:
                        raise ValueError('block_start before its block')
                    checks['blocks_admissions'] = checks.get('blocks_admissions', 0) + 1
                elif kind in ('park', 'progress') and reason == 'level_retired':
                    allowance = blocks_cfg.get('level_token_allowance') if budget_policy == 'blocks' else schedule.get('level_token_allowance')
                    if not allowance or spent + tolerance < allowance * (level + 1):
                        raise ValueError('independent level retirement before exhausting the level allowance')
                    checks['budget_parks'] += 1
                elif kind in ('admit', 'grant') and schedule.get('level_token_allowance'):
                    expected = schedule['level_token_allowance'] * (level + 1) - spent
                    if reason == 'opening_switch':
                        expected = max(expected, floor)
                    equal(f'{game_id}: independent level grant', budget, max(0, expected))
                    last_grant = (budget_time, float(budget))
                    checks['grants'] += 1
                elif kind == 'park' and reason in ('block_end', 'tenure_end', 'level0_retired'):
                    if budget_policy != 'blocks':
                        raise ValueError('blocks park without the blocks policy')
                    if reason == 'level0_retired' and (
                            level or budget_time + tolerance < float(blocks_cfg['level0_budget_tokens'])):
                        raise ValueError(f'level0_retired before its budget ({budget_time:.0f} tokens, level {level})')
                    if reason == 'tenure_end' and blocks_cfg.get('tenure_unit', 'wall_seconds') == 'wall_seconds' and (
                            blocks_first is None or wall - blocks_first + tolerance < block_len):
                        raise ValueError('tenure_end before one block from the first admission')
                    if reason == 'block_end':
                        # One block after the original's own first admission (its block's boundary; later for a
                        # staggered start), within one tick.
                        end = (blocks_first or 0.0) + block_len
                        if blocks_first is None or not -tolerance <= wall - end <= 1.1:
                            raise ValueError(f'block_end at {wall} is not one block after the first admission')
                    checks['blocks_parks'] = checks.get('blocks_parks', 0) + 1
                elif kind in ('admit', 'grant') and budget_policy == 'priority':
                    # Priority gate: no budgets; a game's first admission is its first window, every later one a
                    # handover or a finished game's slot; a grant is a handover the game won back (it kept the slot).
                    allowed = {'admit': ('first_window', 'handover', 'finished'), 'grant': ('handover',)}[kind]
                    if budget is not None or reason not in allowed:
                        raise ValueError(f'priority {kind} {reason} with budget {budget}')
                    if kind == 'admit':
                        if (reason == 'first_window') != (priority_admits == 0):
                            raise ValueError(f'priority admission {reason} as admission {priority_admits + 1}')
                        priority_admits += 1
                    else:
                        if not priority_admits:
                            raise ValueError('priority handover grant before any admission')
                        priority_handovers += 1
                        checks['priority_handovers'] = checks.get('priority_handovers', 0) + 1
                    checks['priority_admissions'] = checks.get('priority_admissions', 0) + (kind == 'admit')
                elif kind == 'park' and reason == 'handover':
                    if budget_policy != 'priority':
                        raise ValueError('handover park without the priority policy')
                    priority_handovers += 1
                    checks['priority_handovers'] = checks.get('priority_handovers', 0) + 1
                elif kind == 'finish' and reason == 'finished' and budget_policy not in ('blocks', 'priority'):
                    raise ValueError('finished event without the blocks or priority policy')
                elif kind in ('admit', 'grant') and budget_policy == 'commit':
                    # Momentum commitment: promotions (a lane clear or a fill) and the opening-switch keep
                    # carry no budget.
                    if budget is not None or reason not in ('lane_clear', 'lane_clear_queued', 'fill', 'opening_switch'):
                        raise ValueError(f'commit admission {reason} with budget {budget}')
                    checks['commit_admissions'] = checks.get('commit_admissions', 0) + 1
                elif kind in ('admit', 'grant') and budget_policy == 'deep-explore':
                    # Deep/explore: exploration admissions, deep promotions and quantum renewals carry no budget.
                    allowed = {'admit': ('explore_start', 'explore_rotate', 'deep_promote'),
                               'grant': ('deep_promote', 'explore_renew')}[kind]
                    if budget is not None or reason not in allowed:
                        raise ValueError(f'deep-explore {kind} {reason} with budget {budget}')
                    if reason == 'explore_start' and abs(wall - float(switch or 0.0)) > 1.1:
                        raise ValueError('explore_start after the release')
                    if reason == 'deep_promote' and not level:
                        raise ValueError('deep promotion without a clear')
                    checks['deep_explore_admissions'] = checks.get('deep_explore_admissions', 0) + 1
                elif kind == 'park' and reason in ('explore_rotate', 'explore_queue', 'deep_demote'):
                    if budget_policy != 'deep-explore':
                        raise ValueError('deep-explore park without the deep-explore policy')
                    if reason in ('explore_queue', 'deep_demote') and not level:
                        raise ValueError(f'{reason} park without a clear')
                elif kind in ('admit', 'grant') and budget_policy == 'depthwise':
                    # Depthwise: level-0 turns carry no budget; a depth game's budget is the fixed-2x allowance
                    # multiple x (level + 1) x T1 at a clear, and the recorded re-admission rule after a park.
                    allowed = {'admit': ('level0_start', 'level0_rotate', 'depth_readmit', 'depth_parked_clear'),
                               'grant': ('level0_renew', 'level0_clear', 'depth_level')}[kind]
                    if reason not in allowed:
                        raise ValueError(f'depthwise {kind} {reason}')
                    if reason.startswith('level0_') and reason != 'level0_clear':
                        if budget is not None or level:
                            raise ValueError(f'depthwise {reason} with budget {budget} at level {level}')
                        if reason == 'level0_start' and abs(wall - float(switch or 0.0)) > 1.1:
                            raise ValueError('level0_start after the release')
                    else:
                        if not level or t1 is None:
                            raise ValueError(f'depthwise {reason} without a clear')
                        full = multiple * (level + 1) * t1
                        if reason in ('level0_clear', 'depth_level') or depthwise_regrant == 'full':
                            expected = full
                        else:
                            fresh = full - spent
                            expected = fresh if fresh > 0 else max(float(regrant) * spent, spent + floor)
                        equal(f'{game_id}: {reason} grant at {wall}', budget, expected)
                        last_grant = (budget_time, float(budget))
                        checks['grants'] += 1
                    checks['depthwise_admissions'] = checks.get('depthwise_admissions', 0) + 1
                elif kind == 'park' and reason in ('level0_rotate', 'depth_park'):
                    if budget_policy != 'depthwise':
                        raise ValueError('depthwise park without the depthwise policy')
                    if reason == 'level0_rotate' and level:
                        raise ValueError('level0_rotate park after a clear')
                    if reason == 'depth_park':
                        if not level or last_grant is None:
                            raise ValueError('depth_park without a clear and a grant')
                        base, allotted = last_grant
                        if budget_time - base + tolerance < allotted:
                            raise ValueError('depth_park before exhausting its level budget')
                        checks['budget_parks'] += 1
                elif kind in ('admit', 'grant') and budget_policy == 'fixed-2x':
                    if level and t1 is None:
                        raise ValueError('level grant without T1')
                    teff = unit(t1, wall)
                    if reason == 'opening_switch':
                        expected = max(multiple * (level + 1) * teff - spent, floor)
                    elif reason in ('level_reached', 'level_budget', 'regrant', 'parked_progress',
                                    'regrant_same_tick'):
                        if level:
                            expected = multiple * (level + 1) * teff
                        else:
                            if first_park is None:
                                raise ValueError('level-0 readmission before any park')
                            expected = max(first_park, floor)
                    else:
                        raise ValueError(f'unknown grant reason {reason}')
                    equal(f'{game_id}: {reason} grant at {wall}', budget, expected)
                    last_grant = (budget_time, float(budget))
                    checks['grants'] += 1
                elif kind in ('admit', 'grant'):
                    if level and t1 is None:
                        raise ValueError('level grant without T1')
                    fresh = (level + 1) * t1 - spent if level else None
                    if reason == 'opening_switch':
                        expected = capped(max(fresh, floor), t1)
                    elif reason == 'level_reached':
                        expected = capped((level + 1) * t1, t1)
                    elif reason in ('level_budget', 'regrant', 'parked_progress'):
                        expected = capped(fresh if fresh is not None and fresh > 0
                                          else max(float(regrant) * spent, spent + floor), t1)
                    else:
                        raise ValueError(f'unknown grant reason {reason}')
                    equal(f'{game_id}: {reason} grant at {wall}', budget, expected)
                    last_grant = (budget_time, float(budget))
                    checks['grants'] += 1
                elif kind == 'park' and reason == 'index' and budget_policy != 'index':
                    raise ValueError('index park without the index policy')
                elif kind == 'park' and reason in ('stalled_level', 'commit_replaced') and budget_policy != 'commit':
                    raise ValueError('commit park without the commit policy')
                elif kind == 'park' and reason == 'level_budget':
                    if last_grant is None:
                        raise ValueError('budget park without a grant')
                    base, allotted = last_grant
                    if budget_time - base + tolerance < allotted:
                        raise ValueError('parked before exhausting its level budget')
                    checks['budget_parks'] += 1
                if kind == 'park' and first_park is None and level == 0:
                    first_park = budget_time
                if reason == 'wall_end' and target is not None and wall > float(target) + 1.1:
                    issues.append(f'{game_id}: wall-end event {wall:.3f}s exceeds {target}s target by over 1.1s')
            if budget_policy == 'priority':
                # A handover (park or kept grant) happens only at a context eviction the scheduler saw.
                evictions = ((priority_info.get('games') or {}).get(game_id) or {}).get('context_evictions', 0)
                if priority_handovers > int(evictions):
                    raise ValueError(f'{priority_handovers} priority handovers but {evictions} context evictions')
            equal(f'{game_id}: final active clock', game['active_seconds'], _seconds(intervals))
            checks['games'] += 1
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f'{game_id}: {exc}')
    opening_cap = schedule.get('opening_max_wall_seconds')
    if opening_cap is not None and (not math.isfinite(float(opening_cap)) or float(opening_cap) <= 0):
        issues.append('invalid opening cap')
        opening_cap = None
    if budget_policy in ('deep-explore', 'depthwise', 'rank', 'blocks', 'priority'):
        # No opening: the slots start at the release, so phase two is the whole run.
        if switch is None or float(switch) > 1.1:
            issues.append(f'{budget_policy} run switched at {switch}s instead of the release')
    elif len(first_times) >= trigger:
        second = sorted(first_times)[trigger - 1]
        if opening_cap is not None:
            second = min(second, float(opening_cap))
        if switch is None:
            issues.append('Opening did not switch despite enough distinct first clears')
        elif schedule.get('opening_min_tokens') is not None:
            # Broad opening: the switch also waits for every game's minimum tokens (not in these records),
            # so it may come any time after the trigger clear, and no later than the cap.
            if float(switch) < second - tolerance or (
                    opening_cap is not None and float(switch) > float(opening_cap) + 1.1):
                issues.append(f'Broad-opening switch {switch}s precedes trigger clear {second}s or passes the cap')
        elif not second - tolerance <= float(switch) <= second + 1.1:
            issues.append(f'Opening switch {switch}s does not follow trigger clear {second}s within 1.1s')
    elif switch is not None:
        if opening_cap is None or not float(opening_cap) - tolerance <= float(switch) <= float(opening_cap) + 1.1:
            issues.append('Opening switched without enough distinct first clears')
    elif opening_cap is not None and duration > float(opening_cap) + 1.1:
        issues.append('Opening did not switch at its cap')
    if switch is not None:
        switch = float(switch)
        knots = sorted({t for intervals in intervals_by_game for pair in intervals for t in pair
                        if switch + tolerance < t < duration})
        for a, b in zip([switch + tolerance, *knots], [*knots, duration]):
            midpoint = (a + b) / 2
            active = sum(any(start <= midpoint < end for start, end in intervals)
                         for intervals in intervals_by_game)
            if active > limit:
                issues.append(f'Phase two has {active} eligible games at {midpoint:.3f}s; limit {limit}')
                break
    if target is not None and duration > float(target) + .1:
        warnings.append('Report wall duration includes post-deadline monitoring/cleanup; use wall-end events for cutoff')
    return {'valid': not issues, 'issues': issues, 'warnings': warnings, 'checks': checks,
            'scope': 'Recorded clocks and request intervals, T1/grant arithmetic, opening trigger, phase-two eligibility and cutoff; not exclusive GPU service or action-issuance timing',
            'clock_semantics': policy}
