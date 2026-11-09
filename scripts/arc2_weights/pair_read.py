"""Paired h240 read, l256 vs k256 base (pre-registered 20 Oct): right at 2.3M (and 4.6M) from the live/report marks,
and reply health through each mark from events.jsonl: no-code share, broken calls (a python call without `code`),
attempts whose first reply is no-code or broken, tool results that are errors.
usage: pair_read.py RUN_L256 RUN_BASE"""
import json, subprocess, sys

B = 'nebius-arc:arc-runs-bucket/runs/validation'
K = 2.74


def cat(run, rel):
    out = subprocess.run(['rclone', 'cat', f'{B}/{run}/arena/{rel}'], capture_output=True, text=True, timeout=600).stdout
    return out


def read(run):
    live = json.loads(cat(run, 'live/arc2-h240.json') or '{}')
    marks = {k: round(v['score_outputs'] * K) for k, v in ((live.get('roster') or {}).get('marks') or {}).items()}
    rows = [json.loads(l) for l in cat(run, 'working/trials/h240/events.jsonl').splitlines() if l.strip()]
    rows = [r for r in rows if r.get('event') == 'reply']
    out = {'marks': marks, 'tokens': sum((live.get('arm_tokens') or {}).values()), 'tok_s': live.get('tokens_per_second')}
    for name, cap in (('2.3M', 2.3e6), ('4.6M', 4.6e6)):
        rs = [r for r in rows if r.get('tokens_at', 0) <= cap]
        if not rs:
            continue
        n = len(rs)
        nocode = sum(r.get('outcome') == 'no_code' for r in rs)
        broken = sum(r.get('outcome') == 'tool' and r.get('tool_code') is None for r in rs)
        tools = [r for r in rs if r.get('outcome') == 'tool' and r.get('tool_code') is not None]
        terr = sum(str(r.get('tool_result') or '').lstrip().startswith(('Error', 'Traceback')) for r in tools)
        first = [r for r in rs if r.get('reply') == 1]
        fbad = sum(r.get('outcome') == 'no_code' or (r.get('outcome') == 'tool' and r.get('tool_code') is None) for r in first)
        out[name] = {'replies': n, 'no_code': round(nocode / n * 100, 2), 'broken': round(broken / n * 100, 2),
                     'first_bad': round(fbad / max(1, len(first)) * 100, 2), 'first_replies': len(first),
                     'tool_err': round(terr / max(1, len(tools)) * 100, 1), 'truncated': round(
                         sum(r.get('outcome') == 'truncated' for r in rs) / n * 100, 2)}
    return out


a, b = read(sys.argv[1]), read(sys.argv[2])
print(json.dumps({'l256': a, 'base': b}, indent=1))
for m, key in (('2.3M', 'v22-5k'), ('4.6M', '4.6M')):
    if key in a['marks'] and key in b['marks']:
        d = a['marks'][key] - b['marks'][key]
        print(f'{m}: right l256 {a["marks"][key]} base {b["marks"][key]} diff {d:+d}',
              'PASS' if d >= -10 else 'FAIL', '(primary)' if m == '2.3M' else '(secondary)')
if '2.3M' in a and '2.3M' in b:
    ok = all(a['2.3M'][k] <= b['2.3M'][k] + 1.0 for k in ('no_code', 'broken', 'first_bad'))
    print('reply health through 2.3M (<= base + 1 pt):', 'PASS' if ok else 'FAIL',
          {k: (a['2.3M'][k], b['2.3M'][k]) for k in ('no_code', 'broken', 'first_bad', 'tool_err', 'truncated')})
