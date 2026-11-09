"""Watch the fit-data API runs: prints alerts (errors > 2%, no reply 10 min) and END lines. Every 120 s."""
import json, subprocess, sys, time
from pathlib import Path
M = './.venv/bin/modal'; D = Path(__file__).parent / 'w'; D.mkdir(exist_ok=True)
SPECS = [s.split(':') for s in sys.argv[1:]]  # vol:path:endfile
done, said = set(), set()
def get(vol, path, dest):
    return subprocess.run([M, 'volume', 'get', '--force', vol, path, str(dest)], capture_output=True).returncode == 0
def say(msg):
    if msg not in said: said.add(msg); print(msg, flush=True)
while len(done) < len(SPECS):
    for vol, path, endf in SPECS:
        if path in done: continue
        n = path.replace('/', '_')
        stats = None
        if get(vol, f'{path}/proxy.jsonl', D / f'{n}.proxy'):
            r = [json.loads(l) for l in open(D / f'{n}.proxy') if l.startswith('{')]
            r = [x for x in r if x.get('event') == 'reply']
            e = sum(x['status'] != 200 for x in r); c = sum(x.get('completion') or 0 for x in r); cost = sum(x.get('cost') or 0 for x in r)
            age = time.time() - max([x['t'] for x in r] or [time.time()])
            stats = f'replies={len(r)} errors={e} completion={c} cost=${cost:.2f}'
            if len(r) > 20 and e * 50 > len(r): say(f'ALERT {path} errors {e}/{len(r)}')
            if age > 600: print(f'ALERT {path} no reply for {int(age)} s ({stats})', flush=True)
            if cost >= 7: say(f'ALERT {path} cost ${cost:.2f}')
        if get(vol, f'{path}/{endf}', D / f'{n}.end'):
            print(f'END {path} {stats}', flush=True); done.add(path)
    if len(done) < len(SPECS): time.sleep(120)
print('ALL END', flush=True)
