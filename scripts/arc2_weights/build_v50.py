"""arc2-kaggle-v50: v49 with the reply cap 8K; nothing else changed. Options only: the v49
notebook and its runtime dataset (arc2-kaggle-v49-runtime, same bytes), roster set cap 16384 -> 8192 (the loop's and
the start probe's cap: loop.py takes settings["cap"] before the study's completion_cap, which no L4 path reads when
the arm has a cap), save run 30 -> 10 min. usage: build_v50.py"""
import difflib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path('./.claude/worktrees/arc2'); DIST = ROOT / 'dist'
src, name = DIST / 'arc2-kaggle-v49-kernel', 'arc2-kaggle-v50'
bundle = DIST / 'arc2-kaggle-v49'
nb = json.loads((src / 'submission.ipynb').read_text()); old = ''.join(nb['cells'][1]['source'])
i = old.index("'--options', '") + len("'--options', '"); j = old.index("'", i)
opts = json.loads(old[i:j])
assert opts['roster']['set']['cap'] == 16384 and opts['context'] == 131072, opts
new = {**opts, "roster": {**opts['roster'], "set": {**opts['roster']['set'], "cap": 8192}}}
code = old[:i] + json.dumps(new) + old[j:]
save = 'SAVE_TASKS, SAVE_SECONDS, RUN_SECONDS = 240, 1800, 42300  # the save run (public preview): 30 min of play'
assert code.count(save) == 1
code = code.replace(save, save.replace('240, 1800,', '240, 600,').replace('30 min of play', '10 min of play'))
compile(code, 'nb', 'exec')

check = ("import json, sys; from arc2 import kaggle, l4_serving, loop; o = json.loads(sys.argv[1]); "
         "kaggle.check_options(o); s = o['roster']['set']; bad = sorted(set(s) - set(loop.DEFAULTS)); "
         "assert not bad, bad; assert int(s['cap']) in l4_serving.REPLY_CAPS; "
         "c = {k: v for k, v in {**kaggle.OPTIONS, **o}.items() if k not in kaggle.DRIVER_KEYS}; "
         "st = l4_serving.study_for(json.loads(open(sys.argv[2]).read()), c); x = st['arc2']['l4x4']; "
         "print(json.dumps({'arm_cap': st['trials'][0]['arms'][0]['cap'], 'window': st['windows'], "
         "'context': x.get('context'), 'prune': st['server']['expert_prune']['sha256'][:8]}))")
done = subprocess.run([sys.executable, '-c', check, json.dumps(new), str(bundle / 'configs/comparison.json')],
                      env={**os.environ, 'PYTHONPATH': str(bundle / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'},
                      capture_output=True, text=True)
assert done.returncode == 0, done.stderr[-2000:]
res = json.loads(done.stdout)
assert res['arm_cap'] == 8192 and res['window'] == [131072] and res['context'] == 131072, res
assert res['prune'] == '01514464', res
print('options check:', done.stdout.strip())

diff = [l for l in difflib.unified_diff(old.splitlines(), code.splitlines(), lineterm='', n=0)
        if l[:1] in '+-' and l[:3] not in ('+++', '---')]
assert len(diff) == 4, diff  # the options line and the save-run line, each - and +
words = [w for a, b in zip(diff[0::2], diff[1::2]) for w in difflib.ndiff(a[1:].split(), b[1:].split()) if w[:1] in '+-']
print('notebook diff vs v49:', words)
assert words == ['- 16384,', '+ 8192,', '- 1800,', '+ 600,', '- 30', '+ 10'] or sorted(words) == sorted(
    ['- 16384,', '+ 8192,', '- 1800,', '+ 600,', '- 30', '+ 10']), words
nb['cells'][1]['source'] = code.splitlines(keepends=True)
k = DIST / f'{name}-kernel'; shutil.rmtree(k, ignore_errors=True); k.mkdir()
(k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
m = json.loads((src / 'kernel-metadata.json').read_text()); m.update(id='richardcsaky/' + name, title=name)
assert 'keithtyser' not in json.dumps(m)
(k / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + '\n')
(DIST / f'{name}-smoke-options.json').write_text(json.dumps(new) + '\n')
print(k, m['dataset_sources'])
