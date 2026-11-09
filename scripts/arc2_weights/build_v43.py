"""arc2-kaggle-v43: v37's settings (xhigh, 64K, cap 16K, images 224 + tower_lean, api_servers 4,
aug4 / confirm 3) on the v42 runtime (v41 watchdog + GCS-FUSE notebook prefetch and ready timeout, v42 PLE page-cache
drop; no driver prefetch) with the layer-aware l256 mask and its c240 refit weights. Build only: no push. Scratch.
usage: build_v43.py <overlay manifest sha256>"""
import importlib.util, json, os, shutil, subprocess, sys, runpy, hashlib
from pathlib import Path
ROOT = Path('./.claude/worktrees/arc2'); DIST = ROOT / 'dist'
MASK = Path(__file__).resolve().parent / 'masks/prune-arc2-l256-v1.json'
OLD_DS, NEW_DS = 'arc-expert-int3-refit-arc2c240-k256-v1', 'arc-expert-int3-refit-arc2c240-l256-v1'
manifest_sha = sys.argv[1]
assert len(manifest_sha) == 64
src, name = DIST / 'arc2-kaggle-v42', 'arc2-kaggle-v43'
out = DIST / name; shutil.rmtree(out, ignore_errors=True); out.mkdir()
for p in src.rglob('*'):
    rel = p.relative_to(src)
    if rel.parts[0] == 'dataset-upload' or '__pycache__' in rel.parts or p.is_dir(): continue
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    if rel.as_posix() in ('configs/comparison.json', 'MANIFEST.json'): shutil.copyfile(p, out / rel)
    else: os.link(p, out / rel)
mask_rel = 'configs/expert-prune/prune-arc2-l256-v1.json'
shutil.copyfile(MASK, out / mask_rel)
mask_sha = hashlib.sha256((out / mask_rel).read_bytes()).hexdigest()
mask = json.loads((out / mask_rel).read_text())
counts = [mask['experts'] - len(v) for v in mask['pruned'].values()]
assert len(counts) == 48 and sum(counts) == 256 * 48 and all(c % 32 == 0 for c in counts), counts

path = out / 'configs/comparison.json'; study = json.loads(path.read_text()); study['name'] = name
assert study['server']['expert_overlay']['dataset'] == 'richardcsaky/' + OLD_DS
study['server']['expert_prune'] = {'mask': mask_rel, 'sha256': mask_sha}
study['server']['expert_overlay'] = {'dataset': 'richardcsaky/' + NEW_DS, 'manifest_sha256': manifest_sha, 'check': 'size'}
assert study['model_datasets'] == ['richardcsaky/' + OLD_DS]
study['model_datasets'] = ['richardcsaky/' + NEW_DS]
study['arc2']['weights'] = {'name': 'arc2c240-l256-refit', 'replaced': 'richardcsaky/' + OLD_DS}
assert OLD_DS not in json.dumps(study).replace(json.dumps(study['arc2']['weights']), '')
path.write_text(json.dumps(study, indent=2) + '\n')

sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/arena')]
from build_arena import manifest
idn = json.loads((out / 'MANIFEST.json').read_text())
idn.update(study=name, files=manifest(out))
idn['source'] = ('arc2-kaggle-v42 runtime (da839b71) + layer-aware l256 mask (sha %s) and its c240 refit overlay '
                 '(manifest %s), config only (dist/ops/build_v43.py)' % (mask_sha[:8], manifest_sha[:8]))
assert not any('__pycache__' in n or n.endswith('.pyc') for n in idn['files'])
(out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
runpy.run_path(str(out / 'bootstrap.py'))['verify'](out)

template = DIST / 'arc2-kaggle-v42-kernel'
nb = json.loads((template / 'submission.ipynb').read_text()); code = ''.join(nb['cells'][1]['source'])
i = code.index("'--options', '") + len("'--options', '"); j = code.index("'", i)
opts = json.loads(code[i:j])
assert opts['context'] == 131072 and opts['roster']['set']['cap'] == 32768 and opts['engine_watchdog'] is True
v37 = {"roster": {"arm": "tool", "rates": [330], "set": {"effort": "xhigh", "cap": 16384, "augment": 4,
       "confirm_attempts": 3, "images": True, "image_side": 224}}, "streams": 64, "tower_off": False,
       "prefill_chunk": 2048, "lean_graphs": True, "probe_decode": [16], "block_tokens": 1600, "logits_topk": 64,
       "warm_rht": True, "embed_host": True, "indexer_fp8": False, "grow_wait": 60, "tower_lean": True, "api_servers": 4}
new = {**v37, "engine_watchdog": True}  # v37 exactly + the v41 watchdog; no context key: the served 65,536 default
assert {k: v for k, v in opts.items() if k not in ('context', 'engine_watchdog', 'roster')} == \
    {k: v for k, v in v37.items() if k != 'roster'}, opts
code = code[:i] + json.dumps(new) + code[j:]

check = ("import json, sys; from arc2 import kaggle, l4_serving; o = json.loads(sys.argv[1]); kaggle.check_options(o); "
         "c = {k: v for k, v in {**kaggle.OPTIONS, **o}.items() if k not in kaggle.DRIVER_KEYS}; "
         "s = l4_serving.study_for(json.loads(open(sys.argv[2]).read()), c); x = s['arc2']['l4x4']; "
         "print(json.dumps({'ple_drop': x['ple_drop'], 'window': s['windows'], 'context': x.get('context'), "
         "'prune': s['server']['expert_prune'], 'overlay': s['server']['expert_overlay']['dataset']}))")
done = subprocess.run([sys.executable, '-c', check, json.dumps(new), str(out / 'configs/comparison.json')],
                      env={**os.environ, 'PYTHONPATH': str(out / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'},
                      capture_output=True, text=True)
assert done.returncode == 0, done.stderr[-2000:]
res = json.loads(done.stdout)
assert res['ple_drop'] is True and res['window'] == [65536] and res['overlay'].endswith(NEW_DS), res
print('options check:', done.stdout.strip())
(DIST / f'{name}-smoke-options.json').write_text(json.dumps(new) + '\n')

spec = importlib.util.spec_from_file_location('build_arc2', ROOT / 'docs/validation-20261009/build_arc2.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
for cache in list(out.rglob('__pycache__')): shutil.rmtree(cache)
info = b.kaggle_files(out, 'tool')
import tarfile
with tarfile.open(out / 'dataset-upload' / f'{name}.tar.gz.blob') as archive:
    assert not [n for n in archive.getnames() if '__pycache__' in n or n.endswith('.pyc')]
for extra in ('submission.ipynb', 'kernel-metadata.json'): (out / extra).unlink()
v42_sha = 'da839b71d0c842b162430453ff12ae8f6afefcc895d58153d88fa72b26842d2a'
assert code.count(v42_sha) == 1 and code.count(OLD_DS) == 2
code = code.replace(v42_sha, info['sha256']).replace('arc2-kaggle-v42-runtime', f'{name}-runtime').replace(
    'arc2-kaggle-v42.tar.gz.blob', f'{name}.tar.gz.blob').replace(OLD_DS, NEW_DS)
assert 'arc2-kaggle-v42' not in code and 'k256' not in code
compile(code, 'nb', 'exec'); nb['cells'][1]['source'] = code.splitlines(keepends=True)
k = DIST / f'{name}-kernel'; shutil.rmtree(k, ignore_errors=True); k.mkdir()
(k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
m = json.loads((template / 'kernel-metadata.json').read_text()); m.update(id='richardcsaky/' + name, title=name)
m['dataset_sources'] = [info['dataset'] if d.endswith('arc2-kaggle-v42-runtime') else
                        'richardcsaky/' + NEW_DS if d.endswith(OLD_DS) else d for d in m['dataset_sources']]
assert 'keithtyser' not in json.dumps(m) and info['dataset'] in m['dataset_sources'] and OLD_DS not in json.dumps(m)
(k / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(info, indent=1)); print(m['dataset_sources'])
