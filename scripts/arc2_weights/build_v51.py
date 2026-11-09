"""arc2-kaggle-v51: the RO harness (hx convro: one conversation per task, submit closes it,
reopen after the backlog in first-submit order, one pass) on v49's serving and competition stack. Runtime = v49's
bundle with the harness files from the arc2 branch at HEAD (loop, tasks, trial, conv, repl, repl_worker, admission,
gpu, sandbox, worker, arckit); the driver (kaggle.py), the L4 stack, weights, k256 mask and runtime are v49's bytes.
Notebook = v49's with the roster set swapped for convro's arm (reserving gate: no "admission" key; cap 16384, v49's
streams 64 / context 131072 / grow_wait 60) and a 10-min save run. usage: build_v51.py"""
import difflib, importlib.util, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path
ROOT = Path('./.claude/worktrees/arc2'); DIST = ROOT / 'dist'
src, name = DIST / 'arc2-kaggle-v49', 'arc2-kaggle-v51'
HARNESS = tuple(f'src/arc2/{f}.py' for f in ('loop', 'tasks', 'trial', 'conv', 'repl', 'repl_worker', 'admission',
                                             'gpu', 'sandbox', 'worker', 'arckit'))
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
for rel in HARNESS:
    if not (out / rel).exists(): shutil.copyfile(ROOT / rel, out / rel)
path = out / 'configs/comparison.json'; study = json.loads(path.read_text()); study['name'] = name
path.write_text(json.dumps(study, indent=2) + '\n')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/arena')]
from build_arena import manifest
idn = json.loads((out / 'MANIFEST.json').read_text())
idn.update(study=name, files=manifest(out))
idn['source'] = ('arc2-kaggle-v49 bundle + %s from arc2 %s (scripts/arc2_weights/build_v51.py)'
                 % (', '.join(Path(h).name for h in HARNESS), commit[:8]))
(out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
runpy = __import__('runpy'); runpy.run_path(str(out / 'bootstrap.py'))['verify'](out)

template = DIST / 'arc2-kaggle-v49-kernel'
nb = json.loads((template / 'submission.ipynb').read_text()); old = ''.join(nb['cells'][1]['source'])
i = old.index("'--options', '") + len("'--options', '"); j = old.index("'", i)
opts = json.loads(old[i:j])
assert opts['roster']['set']['cap'] == 16384 and opts['context'] == 131072 and opts['streams'] == 64, opts
ro = {"effort": "xhigh", "cap": 16384, "augment": 4, "confirm_attempts": 3, "images": True, "image_side": 224,
      "tool": True, "conversation": True, "submit_check": False, "submit_mode": "reopen",
      "reopen_order": "first_submit", "tail_clip": False, "bad_call_fix": True, "reopen_passes": 1}
ref = json.loads((DIST / 'arc2-rtx-c240-hx-convro-v3b/configs/comparison.json').read_text())['trials'][-1]['arms'][0]
assert {k: ref.get(k, ro[k]) for k in ro} == ro, ref  # convro's arm, key for key
new = {**opts, "roster": {**opts['roster'], "set": ro}}
code = old[:i] + json.dumps(new) + old[j:]
save = 'SAVE_TASKS, SAVE_SECONDS, RUN_SECONDS = 240, 1800, 42300  # the save run (public preview): 30 min of play'
assert code.count(save) == 1
code = code.replace(save, save.replace('240, 1800,', '240, 600,').replace('30 min of play', '10 min of play'))

check = ("import json, sys; from arc2 import kaggle, l4_serving, loop, conv, repl, admission, trial; "
         "o = json.loads(sys.argv[1]); kaggle.check_options(o); s = o['roster']['set']; "
         "bad = sorted(set(s) - set(loop.DEFAULTS)); assert not bad, bad; assert int(s['cap']) in l4_serving.REPLY_CAPS; "
         "c = {k: v for k, v in {**kaggle.OPTIONS, **o}.items() if k not in kaggle.DRIVER_KEYS}; "
         "st = l4_serving.study_for(json.loads(open(sys.argv[2]).read()), c); x = st['arc2']['l4x4']; a = st['trials'][0]['arms'][0]; "
         "assert 'admission' not in st['arc2'] and 'admission' not in st['trials'][0]; "
         "print(json.dumps({'arm': {k: a.get(k) for k in s}, 'window': st['windows'], 'context': x.get('context'), "
         "'streams': x['streams'], 'grow_wait': st['arc2'].get('grow_wait'), "
         "'prune': st['server']['expert_prune']['sha256'][:8], 'overlay': st['server']['expert_overlay']['dataset']}))")
done = subprocess.run([sys.executable, '-c', check, json.dumps(new), str(out / 'configs/comparison.json')],
                      env={**os.environ, 'PYTHONPATH': str(out / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'},
                      capture_output=True, text=True)
assert done.returncode == 0, done.stderr[-2000:]
res = json.loads(done.stdout)
assert res['arm'] == ro and res['window'] == [131072] and res['context'] == 131072, res
assert res['prune'] == '01514464' and res['overlay'].endswith('arc-expert-int3-refit-arc2c240-k256-v1'), res
print('options check:', done.stdout.strip())
(DIST / f'{name}-smoke-options.json').write_text(json.dumps(new) + '\n')

spec = importlib.util.spec_from_file_location('build_arc2', ROOT / 'docs/validation-20261009/build_arc2.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
for cache in list(out.rglob('__pycache__')): shutil.rmtree(cache)
info = b.kaggle_files(out, 'tool')
with tarfile.open(out / 'dataset-upload' / f'{name}.tar.gz.blob') as archive:
    names = archive.getnames()
    assert not [n for n in names if '__pycache__' in n or n.endswith('.pyc')]
    assert all(any(n.endswith(h) for n in names) for h in HARNESS), 'harness file missing from the blob'
for extra in ('submission.ipynb', 'kernel-metadata.json'): (out / extra).unlink()
V49_BLOB = '21db8b9fbb100a5e5ae84fabf774c2ce4dc953480a62451d134143f6ac1024bb'  # v49 runtime archive
assert code.count(V49_BLOB) == 1
code = code.replace(V49_BLOB, info['sha256']).replace('arc2-kaggle-v49-runtime', f'{name}-runtime').replace(
    'arc2-kaggle-v49.tar.gz.blob', f'{name}.tar.gz.blob')
assert 'arc2-kaggle-v49' not in code and "'--copies'" not in code
compile(code, 'nb', 'exec'); nb['cells'][1]['source'] = code.splitlines(keepends=True)
diff = [l for l in difflib.unified_diff(old.splitlines(), code.splitlines(), lineterm='', n=0)
        if l[:1] in '+-' and l[:3] not in ('+++', '---')]
print('notebook lines changed vs v49:', len(diff) // 2)
k = DIST / f'{name}-kernel'; shutil.rmtree(k, ignore_errors=True); k.mkdir()
(k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
m = json.loads((template / 'kernel-metadata.json').read_text()); m.update(id='richardcsaky/' + name, title=name)
m['dataset_sources'] = [info['dataset'] if d.endswith('arc2-kaggle-v49-runtime') else d for d in m['dataset_sources']]
assert 'keithtyser' not in json.dumps(m) and info['dataset'] in m['dataset_sources']
(k / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(info, indent=1)); print(m['dataset_sources'])
