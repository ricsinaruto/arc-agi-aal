"""Alert-only watcher for the Dots3 fixed-attempt DP2 screen, every 3 min from arc2-apiscreen/<run>/dots3/.
One line, exit, on: a server exited before ready; not both ready 60 min after launch; the rate rule (20 min
after both ready: combined tok/s < 1,500 AND mean in flight < 30, two signals); tokens and replies flat 15 min after
ready; the drivers done (the read). usage: dp2watch.py RUN LAUNCH_EPOCH [--backtest FILE]"""
import json
import re
import subprocess
import sys
import time

M = './.venv/bin/modal'


def check(x, st, launch):
    """x: {now, log, a, b} with a/b = progress dicts or None."""
    log = x['log']
    if 'exited' in log and 'before ready' in log:
        return 'server exited before ready: ' + re.findall(r'vllm \d exited[^\n]*', log)[0]
    if 'drivers done' in log:
        return 'ended: ' + re.findall(r'drivers done[^\n]*', log)[0]
    ready = len(re.findall(r'vllm \d ready after', log))
    if ready < 2:
        return f'only {ready} of 2 servers ready {int(x["now"] - launch)} s after launch' if x['now'] - launch > 3600 else None
    st.setdefault('ready_at', x['now'])
    halves = [h for h in (x['a'], x['b']) if h]
    tokens = sum(sum((h.get('arm_tokens') or {}).values()) for h in halves)
    replies = sum(h.get('replies') or 0 for h in halves)
    play = max([h.get('elapsed_seconds') or 0 for h in halves] or [0])
    inflight = sum((h.get('gate') or {}).get('inflight', 0) for h in halves)
    st.setdefault('inflight', []).append(inflight)
    if (tokens, replies) != st.get('key'):
        st.update(key=(tokens, replies), since=x['now'])
    elif x['now'] - st['since'] > 900:
        return f'tokens {tokens} and replies {replies} flat {int(x["now"] - st["since"])} s'
    if play >= 1200 and not st.get('rated'):
        st['rated'] = True
        rate, mean_in = tokens / play, sum(st['inflight']) / len(st['inflight'])
        if rate < 1500 and mean_in < 30:
            return f'RATE RULE: {rate:.0f} tok/s and {mean_in:.1f} in flight at {play:.0f} s of play (stop rule: < 1,500)'
        print(f'rate check passed: {rate:.0f} tok/s, {mean_in:.1f} in flight at {play:.0f} s', flush=True)
    return None


def read(run):
    def cat(path):
        r = subprocess.run([M, 'volume', 'get', 'arc2-apiscreen', f'{run}/dots3/{path}', '-'], capture_output=True,
                           text=True, timeout=180)
        if 'rate limit' in r.stdout + r.stderr:
            raise RuntimeError('rate limited')
        return r.stdout if r.returncode == 0 else ''

    def js(t):
        return json.JSONDecoder().raw_decode(t[t.index('{'):])[0] if '{' in t else None
    return {'now': time.time(), 'log': cat('screen.log'), 'a': js(cat('a/progress.json')), 'b': js(cat('b/progress.json'))}


if __name__ == '__main__':
    RUN, LAUNCH = sys.argv[1], float(sys.argv[2])
    st = {}
    if '--backtest' in sys.argv:
        for x in json.load(open(sys.argv[sys.argv.index('--backtest') + 1])):
            a = check(x, st, LAUNCH)
            if a:
                print(a)
                break
        else:
            print('quiet')
        sys.exit()
    while True:
        try:
            a = check(read(RUN), st, LAUNCH)
        except Exception as e:  # noqa: BLE001
            a = None
            print(f'read error {e}', flush=True)
        if a:
            print(a, flush=True)
            sys.exit()
        time.sleep(180)
