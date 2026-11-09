"""Watcher for the arc2 model screen (alert-only): every 5 min reads each model's tasks.jsonl / proxy.jsonl /
screen-report.json from the volume arc2-apiscreen. Alerts (one line, exit) on: a model with no finished entry 40 min
after launch while its proxy has replies (two signals: rows and replies); replies flat for 20 min with entries left
(two signals: proxy rows and task rows both flat); > 30% of the last 40 proxy replies non-200; any model failed;
all models ended (the read).
usage: screenwatch.py RUN LAUNCH_EPOCH MODELS [--backtest FILE]"""
import json
import subprocess
import sys
import time

M = './.venv/bin/modal'


def check(s, state, launch):
    """s: {model: {now, rows, replies, bad_last40, ended, failed}} -> alert or None."""
    out = []
    for name, x in s.items():
        st = state.setdefault(name, {'rows': -1, 'replies': -1, 'since': x['now']})
        if x.get('failed'):
            out.append(f'{name}: failed: {x["failed"]}')
            continue
        if x.get('ended'):
            continue
        if (x['rows'], x['replies']) != (st['rows'], st['replies']):
            st.update(rows=x['rows'], replies=x['replies'], since=x['now'])
        elif x['now'] - st['since'] > 1200 and x['now'] - launch > 1800:
            out.append(f'{name}: replies {x["replies"]} and entries {x["rows"]} flat for {int(x["now"] - st["since"])} s')
        if x['rows'] == 0 and x['replies'] > 0 and x['now'] - launch > 2400:
            out.append(f'{name}: no finished entry {int(x["now"] - launch)} s after launch ({x["replies"]} replies)')
        if x['bad_last40'] > 12:
            out.append(f'{name}: {x["bad_last40"]} of the last 40 proxy replies failed')
    if not out and s and all(x.get('ended') for x in s.values()):
        return 'all models ended'
    return '; '.join(out) or None


def read(run, name):
    def cat(path):
        r = subprocess.run([M, 'volume', 'get', 'arc2-apiscreen', f'{run}/{name}/{path}', '-'], capture_output=True,
                           text=True, timeout=180)
        return r.stdout if r.returncode == 0 else ''
    rows = [x for x in cat('tasks.jsonl').splitlines() if x.strip()]
    proxy = [json.loads(x) for x in cat('proxy.jsonl').splitlines() if x.strip()]
    replies = [p for p in proxy if p.get('event') == 'reply']
    rep = cat('screen-report.json')
    report = json.loads(rep) if rep.strip() else {}
    return {'now': time.time(), 'rows': len(rows), 'replies': len(replies),
            'bad_last40': sum(p.get('status') != 200 for p in replies[-40:]), 'ended': bool(report),
            'failed': report.get('error') if report.get('status') == 'failed' else None}


if __name__ == '__main__':
    RUN, LAUNCH, MODELS = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(',')
    if '--backtest' in sys.argv:
        state = {}
        for s in json.load(open(sys.argv[sys.argv.index('--backtest') + 1])):
            a = check(s, state, LAUNCH)
            if a:
                print(a)
                break
        else:
            print('quiet')
        sys.exit(0)
    state = {}
    while True:
        s = {}
        for name in MODELS:
            try:
                s[name] = read(RUN, name)
            except Exception:  # noqa: BLE001 - next pass
                pass
        a = check(s, state, LAUNCH)
        if a:
            print(a, flush=True)
            sys.exit(0)
        time.sleep(300)
