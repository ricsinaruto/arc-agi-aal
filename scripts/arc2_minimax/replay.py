"""CPU replay of a v37 API screen's records: reply caps, turn caps, stop-after-verify, confirm value."""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, './.claude/worktrees/arc2/dist/arc2-apiscreen-v1/src')
from arc2.loop import guesses
DATA = Path('./.claude/worktrees/arc2/dist/arc2-rtx-hard38-v37play/data/arc2/evaluation')
d = Path(sys.argv[1]); MARK = float(sys.argv[2]) if len(sys.argv) > 2 else 4514400
rows = [json.loads(l) for l in (d / 'tasks.jsonl').read_text().splitlines() if l.startswith('{')]
ev = [json.loads(l) for l in (d / 'events.jsonl').read_text().splitlines() if l.startswith('{')]
rep = [e for e in ev if e['event'] == 'reply']
G = collections.defaultdict(list)
for e in rep: G[(e['task'], int(e['attempt']), int(e['reply']))].append(e)
for v in G.values(): v.sort(key=lambda e: float(e['t0']))
task = {r['task']: json.loads((DATA / (r['task'].split('#')[0] + '.json')).read_text()) for r in rows}
def right(t, cands, limit=None):
    tk = task[t]; picks = guesses(cands, [x['input'] for x in tk['test']], limit, 'clock')
    return sum(any(g == x['output'] for g, _ in p) for p, x in zip(picks, tk['test']))
def ok(t, c, k): return c['tests'][k] is not None and c['tests'][k] == task[t]['test'][k]['output']
NOUT = sum(len(task[r['task']]['test']) for r in rows)
C = {r['task']: r['candidates'] for r in rows}
tot = sum(e['completion_tokens'] for e in rep)
base = sum(right(t, C[t]) for t in C); atmark = sum(right(t, C[t], MARK) for t in C)
print(f'outputs {NOUT}; right final {base}, at mark {atmark}; gen tokens {tot/1e6:.2f}M; right/Mtok final {base/tot*1e6:.2f}')
L = [e for e in rep if e['finish'] == 'length']
print(f'length-cut replies {len(L)}/{len(rep)}: {sum(e["completion_tokens"] for e in L)/1e6:.2f}M ({sum(e["completion_tokens"] for e in L)/tot:.0%} of tokens)')

def sim(cap=1e9, calls=99, replies=99, keep=lambda key: True):
    """Cut each API call at cap (a cut call ends its reply with no candidate), at most `calls` calls per reply and
    `replies` replies per attempt; keep(key) drops whole replies. Later replies are left as they were."""
    alive, T = set(), 0
    for key, calls_ in G.items():
        if key[2] > replies or not keep(key): continue
        lost = False
        for i, e in enumerate(calls_):
            T += min(e['completion_tokens'], cap)
            if e['completion_tokens'] > cap or i + 1 > calls: lost = True; break
        if not lost: alive.add(key)
    s = sum(right(t, [c for c in cs if (t, c['attempt'], c['reply']) in alive]) for t, cs in C.items())
    return s, T
rc = [(t, c) for t, cs in C.items() for c in cs if any(ok(t, c, k) for k in range(len(c['tests'])))]
mx = sorted(max(e['completion_tokens'] for e in G[(t, c['attempt'], c['reply'])]) for t, c in rc)
nc = sorted(len(G[(t, c['attempt'], c['reply'])]) for t, c in rc)
print(f'right candidates {len(rc)}: largest single call in their reply {mx}; calls in their reply {nc}')
print('(a) per-call cap: cap, right, tokens M, right/Mtok')
for cap in (8192, 12288, 16384, 20480, 24576, 32768):
    s, T = sim(cap=cap); print(f'  {cap:6d} {s:3d} {T/1e6:5.2f} {s/T*1e6:5.2f}')
A = collections.defaultdict(list)
for e in rep: A[(e['task'], int(e['attempt']))].append(e)
pt = sorted(e['completion_tokens'] for e in rep); pa = sorted(sum(x['completion_tokens'] for x in v) for v in A.values())
pr = sorted(sum(x['completion_tokens'] for x in v) for v in G.values())
q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
print(f'(b) per call median {q(pt,.5)} p90 {q(pt,.9)} mean {sum(pt)/len(pt):.0f}; per reply (turn) median {q(pr,.5)} p90 {q(pr,.9)} mean {sum(pr)/len(pr):.0f} n={len(pr)}; per attempt median {q(pa,.5)} mean {sum(pa)/len(pa):.0f} n={len(pa)}')
by = collections.defaultdict(lambda: [0, 0])
for e in rep: by[(e['phase'], e['outcome'])][0] += 1; by[(e['phase'], e['outcome'])][1] += e['completion_tokens']
for k, v in sorted(by.items(), key=lambda x: -x[1][1])[:6]: print(f'  {k[0]:8s} {k[1]:9s} n={v[0]:3d} tok={v[1]/1e6:.2f}M ({v[1]/tot:.0%})')
ends = collections.Counter(G[k][-1]['outcome'] for k in G)
print('  reply endings', dict(ends))
for n in (3, 5, 8, 12):
    s, T = sim(calls=n); print(f'  calls/reply<={n}: right {s} tokens {T/1e6:.2f}M right/Mtok {s/T*1e6:.2f}')
for n in (2, 3, 4, 6):
    s, T = sim(replies=n); print(f'  replies/attempt<={n}: right {s} tokens {T/1e6:.2f}M right/Mtok {s/T*1e6:.2f}')
post = 0
for (t, a), es in A.items():
    vr = min([c['reply'] for c in C[t] if c['attempt'] == a and c['verified']], default=None)
    if vr is not None: post += sum(e['completion_tokens'] for e in es if int(e['reply']) > vr)
print(f'  tokens spent in an attempt after its first verified program: {post}')
fv_right = fv_n = 0
for t, cs in C.items():
    v = sorted([c for c in cs if c['verified']], key=lambda c: c['clock'])
    if v: fv_n += 1; fv_right += all(ok(t, v[0], k) for k in range(len(v[0]['tests'])))
print(f'(c) first verified answer right on all outputs: {fv_right}/{fv_n} tasks')
ph = {k: G[k][0]['phase'] for k in G}
s, T = sim(keep=lambda k: ph[k] == 'open'); print(f'  open phase only: right {s} tokens {T/1e6:.2f}M right/Mtok {s/T*1e6:.2f}')
s, T = sim(keep=lambda k: k[1] == 1); print(f'  attempt 1 only: right {s} tokens {T/1e6:.2f}M right/Mtok {s/T*1e6:.2f}')
for cap in (12288, 16384):
    s, T = sim(cap=cap, keep=lambda k: ph[k] == 'open'); print(f'  open only + cap {cap}: right {s} tokens {T/1e6:.2f}M right/Mtok {s/T*1e6:.2f}')
for cap in (20480,):
    s, T = sim(cap=cap, keep=lambda k: ph[k] == 'open'); print(f'  open only + cap {cap}: right {s} tokens {T/1e6:.2f}M right/Mtok {s/T*1e6:.2f}')
    s, T = sim(cap=cap); print(f'  all phases + cap {cap}: right {s} tokens {T/1e6:.2f}M right/Mtok {s/T*1e6:.2f}')
