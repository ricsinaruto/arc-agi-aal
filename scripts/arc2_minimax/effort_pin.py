import json, os, sys, urllib.request, urllib.error, concurrent.futures as cf
D = './.claude/worktrees/arc2/dist/arc2-rtx-hard38-v37play/data/arc2/evaluation/'
TASKS = sys.argv[1].split(',')
T = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
VARIANTS = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {
    'low': {'effort': 'low'}, 'minimal': {'effort': 'minimal'}, 'none': {'effort': 'none'},
    'enabled_false': {'enabled': False}, 'max2k': {'max_tokens': 2048}, 'max6k': {'max_tokens': 6144}}
def ask(arg):
    tid, name, r = arg
    task = json.load(open(D + tid + '.json'))
    prompt = ("Solve this ARC task. Reply with only the test output grid as JSON.\n"
              + json.dumps({'train': task['train'], 'test': [{'input': t['input']} for t in task['test']][:1]}))
    body = {'model': 'minimax/minimax-m3.1-flash', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 20480,
            'temperature': r.get('T', T), 'top_p': 0.95, 'usage': {'include': True}, 'reasoning': {k: v for k, v in r.items() if k != 'T'}, 'provider': {'order': ['parasail'], 'allow_fallbacks': False}}
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', json.dumps(body).encode(),
                                 {'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'], 'Content-Type': 'application/json'})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        return tid, name, 'HTTP', e.code, e.read()[:200]
    u = d.get('usage', {}); ch = d['choices'][0]; txt = ch['message'].get('content') or ''
    try:
        g = json.loads(txt[txt.index('['):txt.rindex(']') + 1]); right = g == task['test'][0]['output']
    except Exception:
        right = None
    return (tid, name, u.get('completion_tokens'), (u.get('completion_tokens_details') or {}).get('reasoning_tokens'),
            ch.get('finish_reason'), right, d.get('provider'), round(u.get('cost') or 0, 4))
jobs = [(t, n, r) for t in TASKS for n, r in VARIANTS.items()]
with cf.ThreadPoolExecutor(12) as ex:
    for r in ex.map(ask, jobs): print(r, flush=True)
