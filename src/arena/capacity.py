"""Context and concurrency admission from measured backend cache capacity."""

import re
from pathlib import Path

KV_POOL_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
WINDOWS = (131072, 98304, 65536)
MARGIN = 0.90


def parse_kv_pool(lines):
    """The last ``GPU KV cache size: N tokens`` value among the server-log lines, or None."""
    pool = None
    for line in lines:
        match = KV_POOL_RE.search(line)
        if match:
            pool = int(match.group(1).replace(",", ""))
    return pool


def server_log_lines(server_dir):
    """Every line of every ``*.log`` under the server directory, in file order."""
    lines = []
    for log in sorted(Path(server_dir).glob("*.log")):
        lines.extend(log.read_text(errors="replace").splitlines())
    return lines


def kv_pool_tokens(server_dir):
    """The served KV pool from the whole server log.

    Not ``kv_admission_summary``: that keeps only the last twelve KV-related lines, and
    vLLM's ten-second "GPU KV cache usage" throughput lines push the one "GPU KV cache
    size" line out once a trial has run (arena smoke 1: trials 3 and 4 found no pool).
    """
    pool = parse_kv_pool(server_log_lines(server_dir))
    if pool is None:
        raise RuntimeError(f"No 'GPU KV cache size' line in the server logs under {server_dir}")
    return pool


def choose_window(pool, *, active_limit=2, margin=MARGIN, windows=WINDOWS):
    """``{"window", "pool", "margin", "capacity", "rejected", "rule"}``; raises when none fits."""
    capacity = int(pool * margin)
    rejected = []
    for window in sorted(windows, reverse=True):
        if active_limit * window <= capacity:
            return {
                "window": window,
                "pool": int(pool),
                "margin": margin,
                "capacity": capacity,
                "active_limit": active_limit,
                "rejected": rejected,
                "rule": "active_limit * window <= pool * margin",
            }
        rejected.append(window)
    raise RuntimeError(
        f"No window in {windows} fits {active_limit} games in a {pool}-token pool at margin {margin}"
    )


FRESH_OVERHEAD_TOKENS = 256


def focus_fit(budget, window, active_limit, *, required=False):
    """How many full-window requests the block gate admits at once, from the served layout.

    A focus game's largest reservation is its window (prompt plus completion cap, compaction at the
    window) plus the fresh-clock/template overhead; ``budget.cost`` adds the server's own speculative
    lookahead (0 on MTP-0) and its fixed hybrid state. ``required``: raise before gameplay unless every
    focus game's full window fits beside the others (``require_focus_windows_fit``, fixed-2x-mtp0-wide),
    so a server that came up with a smaller cache (e.g. MTP-3: 199 blocks, 63 per 64K request) cannot
    silently run fewer concurrent focus games than the study names.
    """
    # A native-128K server cannot admit 128K + 256; its requests are clipped at the native window.
    request_tokens = int(window) + FRESH_OVERHEAD_TOKENS
    if budget.layout.get('max_model_len'):
        ceiling = int(budget.layout['max_model_len']) - int(budget.layout.get('lookahead', 0))
        request_tokens = min(request_tokens, ceiling)
    cost = budget.cost(request_tokens)
    fits = budget.capacity // cost
    result = {'request_tokens': request_tokens, 'cost_per_full_window_request': cost,
              'full_window_requests': fits, 'capacity': budget.capacity,
              'lookahead': budget.layout.get('lookahead'), 'active_limit': active_limit,
              'required': bool(required),
              'rule': 'floor(capacity / cost(window + 256)) >= active_limit'}
    if required and (type(active_limit) is not int or fits < active_limit):
        raise RuntimeError(
            f'{active_limit} focus windows of {window} tokens need {cost} of {budget.capacity} blocks each; '
            f'only {fits} fit the served cache (lookahead {budget.layout.get("lookahead")})')
    return result


def competition_focus_fit(budget, window, study):
    """``focus_fit`` for a study with ``focus_fit_fallback`` (competition builds): a served
    cache that fits fewer full windows than ``active_limit`` must not cost the whole run, so the study's
    ``active_limit`` drops to what fits (at least ``min_slots``, else raise before gameplay) and the result records
    the fallback. Without the key this is ``focus_fit`` with ``require_focus_windows_fit``."""
    fallback = study.get("focus_fit_fallback")
    over = study.get("focus_oversubscription")
    limit = study["active_limit"]
    if over:
        # hero-v2 follow-ups (29 September): more focus games than full windows, on purpose. The exact KV gate holds
        # requests until the pool has room (no preemption), so the fit is recorded, not enforced; a served pool below
        # ``min_full_windows`` full windows is broken and refuses before gameplay.
        if fallback or study.get("require_focus_windows_fit"):
            raise ValueError("focus_oversubscription replaces the focus fit check and its fallback")
        result = focus_fit(budget, window, limit, required=False)
        fits = result["full_window_requests"]
        floor = int(over["min_full_windows"])
        if fits < floor:
            raise RuntimeError(f"only {fits} full {window}-token windows fit the served cache; the oversubscribed "
                               f"focus needs at least {floor}")
        # The validated density (the build's gate blocks per focus stream, e.g. the gate replay's 30 streams on 1,201
        # gate blocks) holds on the served pool: a pool that came up smaller runs proportionally fewer streams.
        density = over.get("gate_blocks_per_stream")
        # Milli-block integers (the build rounds the density down to 1/1,000 block): a float quotient can land just
        # below an exact integer (1,564 / 39.1 -> 39.999...), refusing the validated pool itself.
        allowed = (limit if density is None
                   else int(budget.capacity) * 1000 // int(round(float(density) * 1000)))
        active = max(int(over.get("min_slots", floor)), min(limit, allowed))
        if active < limit:
            study["active_limit"] = active
            if study.get("rank") is not None:
                study["rank"] = {**study["rank"], "slots": active}
            result.update(active_limit=active, fallback_from=limit, density_allowed=allowed)
        result.update(oversubscribed=active > fits, oversubscription=max(0, active - fits),
                      gate_blocks_per_stream=density,
                      rule=result["rule"] + " (recorded only: focus_oversubscription, the exact KV gate; focus = "
                                            "min(target, floor(capacity / gate_blocks_per_stream)))")
        return result
    if not fallback:
        return focus_fit(budget, window, limit, required=bool(study.get("require_focus_windows_fit")))
    result = focus_fit(budget, window, limit, required=False)
    fits = result["full_window_requests"]
    if fits < limit:
        floor = int(fallback["min_slots"])
        if fits < floor:
            raise RuntimeError(f"only {fits} full {window}-token windows fit the served cache; the competition "
                               f"fallback needs at least {floor}")
        study["active_limit"] = fits
        if study.get("rank") is not None:
            study["rank"] = {**study["rank"], "slots": fits}  # the ranking's slots are the focus games
        result.update(active_limit=fits, fallback_from=limit, fallback_min_slots=floor)
    return result


def deep_explore_fit(budget, cfg, *, required=True):
    """Mixed windows (fixed-2x-deep-explore): every deep slot's full window beside every exploration slot's.

    ``deep_slots x cost(deep_window + 256) + explore_slots x cost(explore_window + 256) <= capacity`` on the served
    layout (``budget.cost`` adds the lookahead and the fixed hybrid state, as in ``focus_fit``). ``required``: raise
    before gameplay when it does not hold. A demoted deep game's one compaction request (its 128K context, sent from an
    exploration slot) is not in this sum; the gate holds it until the cache has room for it.
    """
    rows = {}
    total = 0
    for kind in ("deep", "explore"):
        window, slots = int(cfg[f"{kind}_window"]), int(cfg[f"{kind}_slots"])
        request_tokens = window + FRESH_OVERHEAD_TOKENS
        if budget.layout.get('max_model_len'):
            ceiling = int(budget.layout['max_model_len']) - int(budget.layout.get('lookahead', 0))
            request_tokens = min(request_tokens, ceiling)
        cost = budget.cost(request_tokens)
        rows[kind] = {'slots': slots, 'window': window, 'request_tokens': request_tokens,
                      'cost_per_full_window_request': cost, 'units': slots * cost}
        total += slots * cost
    result = {**rows, 'total_units': total, 'capacity': budget.capacity, 'fits': total <= budget.capacity,
              'lookahead': budget.layout.get('lookahead'), 'required': bool(required),
              'rule': 'deep_slots x cost(deep_window + 256) + explore_slots x cost(explore_window + 256) <= capacity'}
    if required and total > budget.capacity:
        raise RuntimeError(
            f"{rows['deep']['slots']} deep windows of {rows['deep']['window']} tokens ({rows['deep']['cost_per_full_window_request']} "
            f"blocks each) and {rows['explore']['slots']} exploration windows of {rows['explore']['window']} tokens "
            f"({rows['explore']['cost_per_full_window_request']} each) need {total} of {budget.capacity} blocks "
            f"(lookahead {budget.layout.get('lookahead')})")
    return result


def phase2_capacity(study, budget, window, game_count):
    """Choose concurrency from this server's resolved cache, without shortening context.

    The agent window includes its completion reserve. Fresh-clock/template overhead
    gets a further 256 tokens; cost() adds the backend's speculative lookahead/state.
    This is an admission estimate, not a claim of demonstrated zero preemptions.
    """
    policy = study.get('phase2_admission', 'fixed')
    if policy == 'fixed':
        return {'policy': policy, 'active_limit': int(study['active_limit'])}
    if policy != 'cache_capacity' or budget is None:
        raise ValueError('Cache-sized phase 2 requires a measured backend cache budget')
    # A native-128K model cannot admit 128K+256. Actual requests are hard-clipped
    # by the native server window, including output and refreshed timing.
    ceiling = int(study['server'].get('max_model_len', window + 256)) - int(budget.layout.get('lookahead', 0))
    request_tokens = min(int(window) + 256, ceiling)
    cost = budget.cost(request_tokens)
    cache_limit = budget.capacity // cost
    server = study['server']
    sequence_limit = int(server.get('max_num_seqs', study['generation_slots']))
    chosen = min(cache_limit, int(study['generation_slots']), sequence_limit, game_count)
    if chosen < 1:
        raise RuntimeError('Full agent window and request overhead cannot fit measured cache')
    return {
        'policy': policy, 'active_limit': chosen, 'capacity': budget.capacity,
        'units': budget.layout.get('units', 'cache_blocks'),
        'request_tokens': request_tokens, 'cost_per_full_window_request': cost,
        'cache_limited_games': cache_limit, 'generation_slots': study['generation_slots'],
        'server_sequence_limit': sequence_limit, 'game_count': game_count,
        'planned_cache_bytes': budget.layout.get('planned_cache_bytes'),
        'gpu_memory_utilization': server.get('gpu_memory_utilization'),
        'margin': study['kv_margin'], 'gpu_validated': False,
        'rule': 'min(games, generation slots, server sequences, floor(cache budget / full-window request cost))',
    }
