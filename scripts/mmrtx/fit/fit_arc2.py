"""minimax arm: MiniMax-M3.1-Flash on the v37 ARC-AGI-2 harness through OpenRouter (Modal CPU driver + arc2.api_proxy).
API-screen code from scripts/arc2_dots3/dots3_screen.py (arc2 arm), trimmed to the API path, plus: the window and the
proxy's reply timeout from the overrides (``max_model_len``, ``api_timeout``).

usage: SCREEN_BUNDLE=dist/arc2-mm-v1 python scripts/openrouter_cli.py -- modal run --detach scripts/arc2_minimax/mm_screen.py::api_main \
         --run R --name N --target minimax/minimax-m3.1-flash --overrides "$(cat CFG.json)" [--limit 2]"""
import json
import math
import os
import time
from pathlib import Path

import modal

app = modal.App('arc2-mmrtx-fit')
BUNDLE = Path(os.environ.get('SCREEN_BUNDLE', '/bundle')).resolve()
out_vol = modal.Volume.from_name('arc2-apiscreen', create_if_missing=True)
SECRET = modal.Secret.from_dict({'OPENROUTER_API_KEY': os.environ.get('OPENROUTER_API_KEY', '')})
API_IMAGE = (modal.Image.debian_slim(python_version='3.12').pip_install('numpy==2.2.6', 'scipy==1.15.3', 'httpx==0.28.1'))
if modal.is_local():
    API_IMAGE = (API_IMAGE.add_local_dir(str(BUNDLE), '/bundle', ignore=['wheels', 'draft-head', '**/__pycache__'])
                 .add_local_file(str(Path(__file__).resolve().parent / 'capture.py'), '/opt/mm/capture.py'))
P0, P1, BOUND, MIN_N = 0.20, 0.35, math.log(19), 38  # the arc2 screen's stop rule (off here: "stop_rule": false)


def sprt(k, n):
    """'pass' / 'miss' / None for k right of the first n entries."""
    if n < MIN_N:
        return None
    llr = k * math.log(P1 / P0) + (n - k) * math.log((1 - P1) / (1 - P0))
    return 'pass' if llr >= BOUND else 'miss' if llr <= -BOUND else None


def prefix(order, done):
    """Right2 flags of the longest finished prefix of ``order`` (``done``: entry -> bool)."""
    flags = []
    for name in order:
        if name not in done:
            break
        flags.append(done[name])
    return flags


def right2(row, task):
    from arc2.loop import guesses
    truth = [t['output'] for t in task['test']]
    picks = guesses(row['candidates'], [t['input'] for t in task['test']])
    return all(any(g == t for g, _ in picks[k]) for k, t in enumerate(truth))


def trial_for(slots, wall, limit=0, images=True, overrides=None):
    study = json.loads(Path('/bundle/configs/comparison.json').read_text())
    study['generation_slots'] = study['active_limit'] = int(slots)
    trial = dict(study['trials'][0])
    base = next((a for a in trial['arms'] if a['name'] == 'base'), trial['arms'][0])
    o = dict(overrides or {})
    trial['arms'] = [o['replace_arm']] if o.get('replace_arm') else [{**base, 'images': bool(images), **o.get('arm', {})}]
    study.update(o.get('study', {}))
    study['arc2'] = {**study['arc2'], **o.get('arc2', {})}
    trial.update(o.get('trial', {}))
    trial['wall_seconds'] = int(wall)
    trial.pop('pool_seed', None)
    trial['paired'] = False
    trial['copy_major'] = True
    if limit:
        trial['tasks'] = trial['tasks'][:int(limit)]
    study['trials'] = [trial]
    return study, trial


def finished_rows(path):
    """Finished entries of a tasks.jsonl (cut or failed entries are not finished)."""
    rows = {}
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            if line.startswith('{'):
                r = json.loads(line)
                if r.get('ended') not in (None, 'failed', 'budget'):
                    rows[r['task'].split('@')[0]] = r
    return rows


def screen_driver(out, base_url, layout_line, slots, wall, limit=0, images=True, force_stop_after=0, log=print,
                  overrides=None, prior=None):
    """run_arc2 on hard38 with the stop rule in a side thread; returns the report (+ the rule's state)."""
    import asyncio
    import sys
    import threading

    sys.path[:0] = ['/bundle/src']
    from arc2.trial import run_arc2
    from arena.early_stop import stop_paths

    out.mkdir(parents=True, exist_ok=True)
    server = out / 'server'
    server.mkdir(parents=True, exist_ok=True)
    (server / 'server.log').write_text(layout_line + '\n')
    prior = dict(prior or {})  # entry -> row from an earlier leg of the same screen (continued after its wall)
    if prior:
        overrides = {**(overrides or {}), 'trial': {**(overrides or {}).get('trial', {}), 'skip': sorted(prior)}}
    study, trial = trial_for(slots, wall, limit, images, overrides)
    data = Path('/bundle') / study['arc2']['data'] / trial['split']
    tasks = {t: json.loads((data / f'{t}.json').read_text()) for t in trial['tasks']}
    copies = int(trial.get('copies', 1))
    order = [f'{t}#{k + 1}' for k in range(copies) for t in trial['tasks']]
    state = {'decision': None, 'n': 0, 'k': 0, 'checked': 0}
    stop_file = stop_paths(out, {})[0]
    halt = threading.Event()

    def rule():
        while not halt.wait(20):
            rows = {**prior, **finished_rows(out / 'tasks.jsonl')}
            done = {n: right2(r, tasks[n.split('#')[0]]) for n, r in rows.items()}
            flags = prefix(order, done)
            state.update(n=len(flags), k=sum(flags), finished=len(done), checked=state['checked'] + 1)
            decision = sprt(sum(flags), len(flags))
            if force_stop_after and len(done) >= force_stop_after:
                decision = decision or 'forced'
            if decision and not state['decision']:
                state.update(decision=decision, at=time.time())
                stop_file.write_text(json.dumps(state))
                log(f'stop rule: {decision} at k={state["k"]} n={state["n"]}')
            (out / 'stop-rule.json').write_text(json.dumps(state))

    if (overrides or {}).get('stop_rule', True):
        threading.Thread(target=rule, daemon=True).start()
    began = time.time()
    try:
        report = asyncio.run(run_arc2(study, trial, source='/bundle', output=out, server_dir=server, base_url=base_url,
                                      deadline_epoch=time.time() + wall + 900))
    except Exception as error:  # noqa: BLE001 - recorded
        report = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'[:2000]}
    halt.set()
    report = {**(report if isinstance(report, dict) else {'report': report}), 'seconds': round(time.time() - began, 1),
              'stop_rule': state}
    (out / 'screen-report.json').write_text(json.dumps(report, indent=1, default=str))
    return report


def committer(volume, stop):
    import threading

    def loop():
        while not stop.wait(60):
            try:
                volume.commit()
            except Exception:  # noqa: BLE001 - next minute
                pass
    threading.Thread(target=loop, daemon=True).start()


def repair_args(raw):
    """A tool call's ``arguments`` as valid JSON (MiniMax v37 run: 2 tasks lost when the provider rejected a history
    holding an earlier call with malformed arguments). Valid JSON passes unchanged; JSON with raw control characters
    is re-encoded; a truncated object is closed; anything else becomes ``{"code": raw}``. Returns (arguments, how)."""
    if not isinstance(raw, str):
        return json.dumps(raw or {}), 'object'
    try:
        json.loads(raw)
        return raw, None
    except ValueError:
        pass
    try:
        return json.dumps(json.loads(raw, strict=False)), 'control-chars'
    except ValueError:
        pass
    text = raw.strip()
    if text.startswith('{'):
        for tail in ('"}', '}', '\\"}', '"]}'):
            try:
                value = json.loads(text + tail, strict=False)
                if isinstance(value, dict):
                    return json.dumps(value), 'closed'
            except ValueError:
                continue
    return json.dumps({'code': raw}), 'wrapped'


def apply_overrides(out_body, overrides, note):
    """Screen overrides on the upstream request: tool-call repair, one pinned provider, and ``api_effort`` (sent
    as reasoning.effort as is, e.g. "minimal", which the proxy's effort table would map to "high")."""
    if overrides.get('repair_tool_args', True):
        out_body['messages'] = repair_history(out_body['messages'], note)
    if overrides.get('provider'):  # one provider for every request (effort and reasoning accounting differ by provider)
        out_body['provider'] = {'order': [overrides['provider']], 'allow_fallbacks': False}
        if overrides.get('drop_top_k', True):
            out_body.pop('top_k', None)
    if overrides.get('api_effort'):
        out_body['reasoning'] = {'effort': overrides['api_effort']}
    return out_body


def repair_history(messages, log=None):
    """Every assistant tool call in the history with valid JSON arguments (the content the model sees is unchanged
    otherwise). ``log``: called with (how, raw) for each repair."""
    out = []
    for m in messages:
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            calls = []
            for c in m['tool_calls']:
                f = dict(c.get('function') or {})
                fixed, how = repair_args(f.get('arguments'))
                if how and log:
                    log(how, f.get('arguments'))
                f['arguments'] = fixed
                calls.append({**c, 'function': f})
            m = {**m, 'tool_calls': calls}
        out.append(m)
    return out


@app.function(image=API_IMAGE, timeout=4 * 3600, cpu=(0.125, 8), volumes={'/out': out_vol}, secrets=[SECRET])
def api_screen(run, name, target, overrides_json, slots=32, wall=7200, limit=0, images=True):
    import socket
    import sys
    import threading
    import urllib.request
    sys.path[:0] = ['/bundle/src']
    from arc2 import api_proxy
    from arena.stock_serving import layout_line
    out = Path('/out') / run / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'events.jsonl').exists() and (out / 'events.jsonl').stat().st_size:  # a preemption restart: the
        (out / 'restart-refused.json').write_text(json.dumps({'t': time.time()}))  # first leg's events stay;
        out_vol.commit()  # replaying from scratch would pay twice (minimax-two-20261024)
        return {'status': 'restart-refused'}
    stop = threading.Event()
    committer(out_vol, stop)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    threading.Thread(target=api_proxy.serve_forever, args=(port, target, os.environ['OPENROUTER_API_KEY']),
                     kwargs={'log_path': str(out / 'proxy.jsonl')}, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    pool = out / 'pool.log'  # a hosted model has no KV pool: a large token pool, so the gate is the slot count
    overrides = json.loads(overrides_json)
    max_len = int(overrides.get('max_model_len', 262144))
    api_proxy.MAX_LEN, api_proxy.TIMEOUT = max_len, int(overrides.get('api_timeout', 1800))
    pool.write_text('GPU KV cache size: 200,000,000 tokens\n')  # 32 slots x (prompt + 524K reply) fits
    repairs = (out / 'repairs.jsonl').open('a')
    lock = threading.Lock()

    def note(how, raw):
        with lock:
            repairs.write(json.dumps({'t': time.time(), 'how': how, 'raw': str(raw)[:4000]}) + '\n')
            repairs.flush()
    original = api_proxy.upstream_body

    def upstream_body(body, target, keep_reasoning=True):
        keep = keep_reasoning and overrides.get('keep_reasoning', True)  # False: earlier reasoning not resent
        return apply_overrides(original(body, target, keep), overrides, note)
    api_proxy.upstream_body = upstream_body
    sys.path.insert(0, '/opt/mm')
    import capture  # mmrtx: full conversations for the reference / fit sets
    rec = capture.Recorder(str(out / 'traj.jsonl.gz')).start()
    capture.install(api_proxy, rec)
    report = screen_driver(out, f'http://127.0.0.1:{port}/v1', layout_line(pool, max_len), slots, wall, limit,
                           images=images, overrides=overrides)
    rec.flush()
    report['overrides'] = overrides
    (out / 'screen-report.json').write_text(json.dumps(report, indent=1, default=str))
    stop.set()
    out_vol.commit()
    return {'status': report.get('status'), 'stop_rule': report.get('stop_rule')}


ARM_PREFIX, ARM_CAP_USD, ARM_SINCE = 'mmrtx-', 500.0, 1792841181 # mmrtx


def arm_check(run, bound):
    """Refuse a run whose id lacks the arm prefix, is already used in the ledger, or would take the minimax arm's
    rows since ARM_SINCE (actual cost when closed, else the bound) plus ``bound`` past ARM_CAP_USD."""
    import subprocess
    if not run.startswith(ARM_PREFIX):
        raise ValueError(f'run id must start with {ARM_PREFIX}')
    common = subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True,
                                     cwd='.').strip()
    rows = json.loads((Path(common).parent / 'docs/spend.json').read_text())['runs']
    own = [r for r in rows if r.get('run_id') == run]
    if any(r.get('state') != 'reserved' or r.get('app_id') for r in own):
        raise ValueError(f'{run} already used in the ledger')  # a reserved row without an app is this launch's
    arm = sum(r.get('actual_usd', r.get('charged_bound_usd', 0)) for r in rows if r.get('run_id') != run
              and str(r.get('run_id', '')).startswith(ARM_PREFIX) and (r.get('start_epoch') or 0) >= ARM_SINCE)
    if arm + bound > ARM_CAP_USD:
        raise RuntimeError(f'minimax arm cap: {arm:.2f} + {bound:.2f} > {ARM_CAP_USD}')
    return arm


@app.local_entrypoint()
def api_main(run: str, name: str, target: str, overrides: str, slots: int = 32, limit: int = 0, wall: int = 7200,
             bound: float = 0.0):
    if not bound:
        raise SystemExit('--bound USD is required (arm cap check)')
    arm = arm_check(run, bound)
    call = api_screen.spawn(run, name, target, overrides, slots, wall, limit)
    print(json.dumps({'call': call.object_id, 'arm_spent_usd': round(arm, 2)}))
