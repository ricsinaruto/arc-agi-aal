"""Real arena phase 2 vs the synthetic decode rate: where the per-stream gap goes (kernel work, phase 6; CPU only).

Inputs, per recorded arena trial (``arena/trials/duck-transfer``):
- ``telemetry.jsonl``: vLLM's cumulative counters every ~5 s. ``iteration_tokens_total_count`` counts engine steps
  (one histogram observation per step), its ``_sum`` the scheduled tokens, ``generation_tokens_total`` the decoded
  tokens and ``prompt_tokens_by_source_total:local_compute`` the prefilled (uncached) prompt tokens.
- the game logs (``games/*/log.jsonl``): every model request's end time and duration, so the HTTP in-flight count
  and each focus game's time between requests (tool execution + harness) are known. The clock offset between the
  logs and the telemetry is fitted on the cumulative prompt-token counter, as in ``prefill_attribution.py``.
- ``arena.json``: the opening switch (phase 2 starts there).
- the hero smoke's decode points (``hero-smoke-result.json``): decode-only ms per step at 1 / 6 / 14 streams.

Per 5-s interval of phase 2 the wall time is split into:
- the decode time the synthetic curve predicts for the interval's steps at their decode batch
  (``steps x t_syn(gen / steps)``);
- the remainder, fitted by non-negative least squares on (prefilled tokens, requests started, steps): the prefill
  cost per token, a fixed cost per new request (its first, mixed, eager step; the vision encoder; admission), and any
  per-step overhead beyond the synthetic.
The gap in aggregate tok/s against the synthetic rate at the phase's slot count is then attributed to
(a) fewer decoding sequences per step than slots (tool time between requests, requests queued or in prefill),
(b) prefill time, (c) per-request overhead and (d) the rest (per-step overhead, context length, fit error).

Run: uv run python scripts/arena_analysis/phase2_gap.py --out docs/validation-20260927/kernels-phase6-gap.json
"""

import argparse
import bisect
import json
from pathlib import Path

ROOT = Path('./runs')
MODAL = ROOT / 'modal-validation'
RUNS = {
    'hero-14x64k (Kaggle preview)': ROOT / 'kaggle-arena-flash-competition-hero-v1/arena/trials/duck-transfer',
    'hero-21x42k': MODAL / ('arena-duck-60m-c110-hero-21x42k-v1-modal-20260928/'
                            'arena-duck-60m-c110-hero-21x42k-v1-modal-20260928/arena/trials/duck-transfer'),
    'hero-6x64k': MODAL / ('arena-duck-60m-c110-hero-6x64k-v1-modal-20260928/'
                           'arena-duck-60m-c110-hero-6x64k-v1-modal-20260928/arena/trials/duck-transfer'),
    'kernels-v2 (6x64k, no Marlin/tiles/kv8)': MODAL / ('arena-duck-60m-c110-kernels-v2-modal-20260928/'
                                                         'arena-duck-60m-c110-kernels-v2-modal-20260928/arena/'
                                                         'trials/duck-transfer'),
}
SMOKE = Path(__file__).resolve().parents[2] / 'docs/validation-20260927/hero-smoke-result.json'
# vLLM's decode CUDA-graph sizes (served capture sizes); a decode batch pads up to the next one.
GRAPH_SIZES = (1, 2, 4, 8, 16, 24, 32)
M = 'vllm:'


def synthetic_curve(path=SMOKE):
    """Decode-only ms per step by stream count: the mean of the 16K and 64K synthetic points (hero smoke)."""
    points = json.loads(path.read_text())['points']
    by = {}
    for p in points:
        if p['kind'] == 'distinct':
            by.setdefault(p['streams'], []).append(p['ms_per_iteration'])
    return sorted((s, sum(v) / len(v)) for s, v in by.items())


def t_syn(curve, b):
    """ms per decode step at a mean batch of b sequences: piecewise linear, extrapolated above the last point with
    the slope of the last segment (flagged in the output: no synthetic point above 14 streams)."""
    if b <= curve[0][0]:
        return curve[0][1]
    for (s0, t0), (s1, t1) in zip(curve, curve[1:]):
        if b <= s1:
            return t0 + (t1 - t0) * (b - s0) / (s1 - s0)
    (s0, t0), (s1, t1) = curve[-2], curve[-1]
    return t1 + (t1 - t0) / (s1 - s0) * (b - s1)


def load_telemetry(trial):
    rows = []
    for line in (trial / 'telemetry.jsonl').open():
        r = json.loads(line)
        rows.append((r['elapsed_seconds'], r['metrics']))
    return rows


def load_admission(trial, t0):
    """The arena gate's events: HTTP in-flight steps (admit/release ``inflight_after``) and, per game, the times its
    requests were admitted (sent) and its waits started (the harness finished the previous turn's tools)."""
    inflight, admitted, waits = [], {}, {}
    for line in (trial / 'admission.jsonl').open():
        e = json.loads(line)
        t = e['epoch'] - t0
        if 'inflight_after' in e:
            inflight.append((t, e['inflight_after']))
        if e.get('event') == 'request_admitted':
            admitted.setdefault(e['game'], []).append(t)
        elif e.get('event') == 'request_wait_started':
            waits.setdefault(e['game'], []).append(t)
    return inflight, admitted, waits


def load_requests(trial, admitted):
    """Model requests from the game logs. A log's ``seconds`` includes the wait at the arena gate (a parked game's
    request waits there until it is re-admitted), so the HTTP start is the game's last admission before the end."""
    t0 = json.loads((trial / 'execution.json').read_text())['started_epoch']
    reqs = []
    for game_dir in sorted((trial / 'games').iterdir()):
        log = game_dir / 'log.jsonl'
        if not log.is_file():
            continue
        times = admitted.get(game_dir.name, [])
        for line in log.open():
            d = json.loads(line)
            if d.get('kind') != 'model':
                continue
            usage = d.get('usage') or {}
            end = d['time'] - t0
            i = bisect.bisect_right(times, end + 1e-3) - 1
            start = times[i] if i >= 0 else end - d.get('seconds', 0)
            reqs.append({'game': game_dir.name, 'purpose': d.get('purpose'), 'end': end, 'start': start,
                         'queued': d.get('seconds', 0) - (end - start),
                         'P': int(usage.get('prompt_tokens') or 0), 'G': int(usage.get('completion_tokens') or 0)})
    return sorted(reqs, key=lambda r: r['start'])


def step_mean(points, lo, hi):
    """Time-weighted mean of a step function [(t, value after t)] over [lo, hi]."""
    total, prev = 0.0, None
    for t, v in points:
        if prev is not None and t > lo and prev[0] < hi:
            total += (min(t, hi) - max(prev[0], lo)) * prev[1]
        prev = (t, v)
    if prev is not None and prev[0] < hi:
        total += (hi - max(prev[0], lo)) * prev[1]
    return total / (hi - lo)


def fit_offset(reqs, tel):
    """Telemetry time + offset = log time; fitted on the cumulative prompt-token (query) counter."""
    q0 = tel[0][1][M + 'prefix_cache_queries_total']
    starts = [r['start'] for r in reqs]
    cum = [0]
    for r in reqs:
        cum.append(cum[-1] + r['P'])

    def err(off):
        return sum(abs((m[M + 'prefix_cache_queries_total'] - q0) - cum[bisect.bisect_right(starts, t + off)])
                   for t, m in tel[1:]) / len(tel)

    return min((err(o / 10), o / 10) for o in range(-300, 301))[::-1]


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def nnls(rows, targets, iters=20000):
    """Tiny non-negative least squares (projected coordinate descent); rows are feature tuples."""
    k = len(rows[0])
    x = [0.0] * k
    gram = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    rhs = [sum(r[i] * y for r, y in zip(rows, targets)) for i in range(k)]
    for _ in range(iters):
        changed = 0.0
        for i in range(k):
            if gram[i][i] <= 0:
                continue
            g = rhs[i] - sum(gram[i][j] * x[j] for j in range(k) if j != i)
            new = max(0.0, g / gram[i][i])
            changed = max(changed, abs(new - x[i]))
            x[i] = new
        if changed < 1e-15:
            break
    return x


def analyze(name, trial, curve, settle=60.0, tail=30.0):
    tel = load_telemetry(trial)
    t0 = json.loads((trial / 'execution.json').read_text())['started_epoch']
    inflight_steps, admitted, waits = load_admission(trial, t0)
    reqs = load_requests(trial, admitted)
    offset, fit_err = fit_offset(reqs, tel)
    arena = json.loads((trial / 'arena.json').read_text())
    switch = find_key(arena, 'switch_wall_seconds')
    capacity = json.loads((trial / 'phase2-capacity.json').read_text())
    slots = capacity.get('active_limit')
    wall = find_key(arena, 'wall_seconds') or 3600
    lo, hi = switch + settle, min(wall, tel[-1][0] + offset) - tail
    intervals = []
    for (ta, ma), (tb, mb) in zip(tel, tel[1:]):
        a, b = ta + offset, tb + offset
        if a < lo or b > hi:
            continue
        d = {key: mb[M + key] - ma[M + key] for key in (
            'iteration_tokens_total_count', 'iteration_tokens_total_sum', 'generation_tokens_total',
            'prompt_tokens_by_source_total:local_compute', 'prompt_tokens_total', 'num_preemptions_total')}
        dt = b - a
        inflight = step_mean(inflight_steps, a, b)
        games_inflight = len({r['game'] for r in reqs if overlap(r['start'], r['end'], a, b) > 0})
        started = sum(1 for r in reqs if a <= r['start'] < b)
        intervals.append({'t': a, 'dt': dt, 'steps': d['iteration_tokens_total_count'],
                          'sched': d['iteration_tokens_total_sum'], 'gen': d['generation_tokens_total'],
                          'P': d['prompt_tokens_by_source_total:local_compute'],
                          'P_all': d['prompt_tokens_total'],
                          'running': (ma[M + 'num_requests_running'] + mb[M + 'num_requests_running']) / 2,
                          'waiting': (ma[M + 'num_requests_waiting'] + mb[M + 'num_requests_waiting']) / 2,
                          'inflight': inflight, 'games_inflight': games_inflight, 'started': started,
                          'preempt': d['num_preemptions_total']})
    T = sum(i['dt'] for i in intervals)
    tot = {k: sum(i[k] for i in intervals) for k in ('steps', 'sched', 'gen', 'P', 'P_all', 'started', 'preempt')}
    mean = {k: sum(i[k] * i['dt'] for i in intervals) / T for k in ('running', 'waiting', 'inflight')}
    idle = sum(i['dt'] for i in intervals if i['steps'] == 0)
    # Synthetic decode time of every step at its interval's decode batch; the rest is regressed.
    busy = [i for i in intervals if i['steps'] > 0]
    syn = [i['steps'] * t_syn(curve, i['gen'] / i['steps']) / 1000.0 for i in busy]
    rest = [i['dt'] - s for i, s in zip(busy, syn)]
    feats = [(i['P'], i['started'], i['steps']) for i in busy]
    c_p, c_req, c_step = nnls(feats, rest)
    fitted = [s + c_p * f[0] + c_req * f[1] + c_step * f[2] for s, f in zip(syn, feats)]
    # Fit quality on the non-synthetic time (dt is ~5 s in every interval, so R^2 on dt itself says nothing).
    ss_res = sum((i['dt'] - f) ** 2 for i, f in zip(busy, fitted))
    mean_rest = sum(rest) / len(rest)
    ss_tot = sum((r - mean_rest) ** 2 for r in rest)
    # Also the plain three-term fit of the whole interval (no synthetic prior): dt = a steps + b gen + c P.
    free = nnls([(i['steps'], i['gen'], i['P'], i['started']) for i in busy], [i['dt'] for i in busy])
    # Robustness: the non-synthetic time on the prefilled tokens alone (one term), and its effective rate.
    only_p = nnls([(i['P'],) for i in busy], rest)[0]
    time_syn, time_p, time_req, time_step = sum(syn), c_p * tot['P'], c_req * tot['started'], c_step * tot['steps']
    non_decode = T - idle - time_syn
    time_resid = T - idle - time_syn - time_p - time_req - time_step
    b_dec = tot['gen'] / tot['steps']
    agg = tot['gen'] / T
    syn_slots = slots / t_syn(curve, slots) * 1000.0
    syn_bdec = b_dec / t_syn(curve, b_dec) * 1000.0
    # Attribution of (syn_slots - agg): (a) fewer decoding sequences per step than slots, at the synthetic rate;
    # then, at the observed batch, the share of wall time not spent in synthetic decode steps.
    loss_batch = syn_slots - syn_bdec
    decode_share = time_syn / T
    loss_time = syn_bdec - agg  # all non-decode time at the observed batch
    parts = {'prefill': time_p, 'per_request': time_req, 'per_step_extra': time_step,
             'idle_engine': idle, 'residual': time_resid}
    share = {k: v / T for k, v in parts.items()}
    result = {
        'run': name, 'trial': str(trial), 'offset_s': offset, 'offset_fit_error_tokens': round(fit_err),
        'switch_s': switch, 'window_s': [round(lo, 1), round(hi, 1)], 'phase2_seconds': round(T, 1),
        'intervals': len(intervals), 'slots': slots,
        'gen_tok_s': round(agg, 1), 'prefill_tok_s': round(tot['P'] / T, 1),
        'prompt_tok_s_incl_cached': round(tot['P_all'] / T, 1),
        'uncached_per_generated': round(tot['P'] / tot['gen'], 2),
        'steps_per_s': round(tot['steps'] / T, 2), 'ms_per_step': round(1000 * (T - idle) / tot['steps'], 2),
        'decode_seqs_per_step': round(b_dec, 2), 'sched_tokens_per_step': round(tot['sched'] / tot['steps'], 1),
        'requests_started_per_s': round(tot['started'] / T, 3),
        'mean_running': round(mean['running'], 2), 'mean_waiting': round(mean['waiting'], 2),
        'mean_http_inflight': round(mean['inflight'], 2), 'preemptions': tot['preempt'],
        'per_running_tok_s': round(agg / mean['running'], 1) if mean['running'] else None,
        'synthetic': {'at_slots_tok_s': round(syn_slots, 1), 'at_decode_batch_tok_s': round(syn_bdec, 1),
                      'ms_per_step_at_decode_batch': round(t_syn(curve, b_dec), 2),
                      'extrapolated': slots > curve[-1][0] or b_dec > curve[-1][0]},
        'fit': {'prefill_ms_per_1k_tokens': round(c_p * 1e6, 2), 'prefill_equiv_tok_s': round(1 / c_p) if c_p else None,
                'per_request_ms': round(c_req * 1000, 2), 'per_step_extra_ms': round(c_step * 1000, 3),
                'r2_of_non_synthetic_time': round(1 - ss_res / ss_tot, 3) if ss_tot else None,
                'rmse_s_per_interval': round((ss_res / len(busy)) ** 0.5, 3),
                'prefill_only_fit_ms_per_1k': round(only_p * 1e6, 2),
                'non_decode_s_per_uncached_token_equiv_tok_s': round(tot['P'] / non_decode) if non_decode > 0 else None,
                'free_fit': {'ms_per_step': round(free[0] * 1000, 2), 'ms_per_decode_token': round(free[1] * 1000, 3),
                             'ms_per_1k_prefill': round(free[2] * 1e6, 2), 'ms_per_request': round(free[3] * 1000, 2)}},
        'time_shares': {'synthetic_decode': round(decode_share, 3), **{k: round(v, 3) for k, v in share.items()}},
        'gap_tok_s': {'total_vs_synthetic_at_slots': round(syn_slots - agg, 1),
                      'fewer_decoding_seqs_than_slots': round(loss_batch, 1),
                      'non_decode_time_at_observed_batch': round(loss_time, 1),
                      'of_which': {k: round(syn_bdec * v, 1) for k, v in share.items()}},
    }
    # Where the missing decoding sequences are: slots - decoding = (slots - HTTP in flight: tool time and harness)
    # + (in flight but not running: engine queue) + (running but not decoding: prefill in progress).
    result['slot_split'] = {
        'slots': slots,
        'not_in_flight (tool/harness time)': round(slots - mean['inflight'], 2),
        'in_flight_not_running (queued)': round(mean['inflight'] - mean['running'], 2),
        'running_not_decoding (in prefill)': round(mean['running'] - b_dec, 2),
        'decoding_per_step': round(b_dec, 2)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--settle', type=float, default=60.0)
    args = parser.parse_args()
    curve = synthetic_curve()
    results = {'synthetic_ms_per_step': curve, 'runs': []}
    for name, trial in RUNS.items():
        if not (trial / 'telemetry.jsonl').is_file():
            print(f'skip {name}: no telemetry')
            continue
        res = analyze(name, trial, curve, settle=args.settle)
        results['runs'].append(res)
        print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
