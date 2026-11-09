"""arc2-kaggle-v49: v46 (v45 = v37 weights and settings + start fixes; atomic notebook writes) with
the 128K attempt window (context 131,072: the shipped compile cache is v39's 128K one) and four harness changes in the
roster set: review "check+second" (check turn, then the free second reading), prompt "arc2" (hint v1), reject_known 3.
Runtime = v45's bundle with src/arc2/loop.py and tasks.py from the arc2 branch at HEAD (the hx c240 series' harness);
trial, sandbox, worker, the driver and the L4 stack are v45's bytes. Not v47 copies, not v48 medium. Save run (public
preview) 30 min, as v44.
usage: build_v49.py"""
import importlib.util, json, os, shutil, subprocess, sys, runpy, tarfile
from pathlib import Path
ROOT = Path('./.claude/worktrees/arc2'); DIST = ROOT / 'dist'
src, name = DIST / 'arc2-kaggle-v45', 'arc2-kaggle-v49'
HARNESS = ('src/arc2/loop.py', 'src/arc2/tasks.py')
dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', *HARNESS], capture_output=True, text=True)
assert dirty.stdout == '', 'commit the harness files first: ' + dirty.stdout
commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
out = DIST / name; shutil.rmtree(out, ignore_errors=True); out.mkdir()
for p in src.rglob('*'):
    rel = p.relative_to(src)
    if rel.parts[0] == 'dataset-upload' or '__pycache__' in rel.parts or p.is_dir(): continue
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    if rel.as_posix() in HARNESS: shutil.copyfile(ROOT / rel, out / rel)
    elif rel.as_posix() in ('configs/comparison.json', 'MANIFEST.json'): shutil.copyfile(p, out / rel)
    else: os.link(p, out / rel)
path = out / 'configs/comparison.json'; study = json.loads(path.read_text()); study['name'] = name
path.write_text(json.dumps(study, indent=2) + '\n')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/arena')]
from build_arena import manifest
idn = json.loads((out / 'MANIFEST.json').read_text())
idn.update(study=name, files=manifest(out))
idn['source'] = ('arc2-kaggle-v45 bundle + src/arc2/loop.py, tasks.py from arc2 %s (scripts/arc2_weights/build_v49.py)'
                 % commit[:8])
(out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
runpy.run_path(str(out / 'bootstrap.py'))['verify'](out)

template = DIST / 'arc2-kaggle-v46-kernel'
nb = json.loads((template / 'submission.ipynb').read_text()); code = ''.join(nb['cells'][1]['source'])
i = code.index("'--options', '") + len("'--options', '"); j = code.index("'", i)
opts = json.loads(code[i:j])
v37set = {"effort": "xhigh", "cap": 16384, "augment": 4, "confirm_attempts": 3, "images": True, "image_side": 224}
assert opts['roster'] == {"arm": "tool", "rates": [330], "set": v37set} and 'context' not in opts, opts
harness = {"review": True, "review_style": "check+second", "prompt": "arc2", "reject_known": 3}
new = {**opts, "context": 131072, "roster": {**opts['roster'], "set": {**v37set, **harness}}}
code = code[:i] + json.dumps(new) + code[j:]
save = 'SAVE_TASKS, SAVE_SECONDS, RUN_SECONDS = 240, 600, 42300  # the save run (public preview): 10 min of play'
assert code.count(save) == 1  # 30-min public preview (as v44); the placeholder save run only
code = code.replace(save, save.replace('240, 600,', '240, 1800,').replace('10 min of play', '30 min of play'))

check = ("import json, sys; from arc2 import kaggle, l4_serving, loop, tasks, trial; o = json.loads(sys.argv[1]); "
         "kaggle.check_options(o); s = o['roster']['set']; bad = sorted(set(s) - set(loop.DEFAULTS)); "
         "assert not bad, bad; assert all(st in tasks.REVIEWS for st in s['review_style'].split('+')); "
         "c = {k: v for k, v in {**kaggle.OPTIONS, **o}.items() if k not in kaggle.DRIVER_KEYS}; "
         "st = l4_serving.study_for(json.loads(open(sys.argv[2]).read()), c); x = st['arc2']['l4x4']; "
         "print(json.dumps({'ple_drop': x['ple_drop'], 'window': st['windows'], 'context': x.get('context'), "
         "'prune': st['server']['expert_prune']['sha256'][:8], 'overlay': st['server']['expert_overlay']['dataset']}))")
done = subprocess.run([sys.executable, '-c', check, json.dumps(new), str(out / 'configs/comparison.json')],
                      env={**os.environ, 'PYTHONPATH': str(out / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'},
                      capture_output=True, text=True)
assert done.returncode == 0, done.stderr[-2000:]
res = json.loads(done.stdout)
assert res['ple_drop'] is True and res['window'] == [131072] and res['context'] == 131072, res
assert res['prune'] == '01514464' and res['overlay'].endswith('arc-expert-int3-refit-arc2c240-k256-v1'), res
print('options check:', done.stdout.strip())
(DIST / f'{name}-smoke-options.json').write_text(json.dumps(new) + '\n')

spec = importlib.util.spec_from_file_location('build_arc2', ROOT / 'docs/validation-20261009/build_arc2.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
for cache in list(out.rglob('__pycache__')): shutil.rmtree(cache)
info = b.kaggle_files(out, 'tool')
with tarfile.open(out / 'dataset-upload' / f'{name}.tar.gz.blob') as archive:
    assert not [n for n in archive.getnames() if '__pycache__' in n or n.endswith('.pyc')]
for extra in ('submission.ipynb', 'kernel-metadata.json'): (out / extra).unlink()
V45_BLOB = 'e0774df49525ef46df5337d46f55de4c7fc5863abcb7d22c50cd76de9f8c6156'  # v45 runtime archive (v46 notebook)
assert code.count(V45_BLOB) == 1
code = code.replace(V45_BLOB, info['sha256']).replace('arc2-kaggle-v45-runtime', f'{name}-runtime').replace(
    'arc2-kaggle-v45.tar.gz.blob', f'{name}.tar.gz.blob')
assert 'arc2-kaggle-v45' not in code and 'def put(data)' in code and "'--copies'" not in code
compile(code, 'nb', 'exec'); nb['cells'][1]['source'] = code.splitlines(keepends=True)
k = DIST / f'{name}-kernel'; shutil.rmtree(k, ignore_errors=True); k.mkdir()
(k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
m = json.loads((template / 'kernel-metadata.json').read_text()); m.update(id='richardcsaky/' + name, title=name)
m['dataset_sources'] = [info['dataset'] if d.endswith('arc2-kaggle-v45-runtime') else d for d in m['dataset_sources']]
assert 'keithtyser' not in json.dumps(m) and info['dataset'] in m['dataset_sources']
(k / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(info, indent=1)); print(m['dataset_sources'])
