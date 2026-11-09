"""arc2-kaggle-v47: v46 (v45 = v37 + start fixes, atomic notebook writes) with two independent
copies of every task (``--copies 2``: entries id#1 / id#2 share the roster, so each gets about half the tokens) and the
submission merging them (arc2 eb2c81b8: pooled candidates, agreement counted per entry). Runtime = v45's bundle with
that one function patched. Build only: no push. usage: build_v47.py"""
import json, os, shutil, sys, runpy, importlib.util
from pathlib import Path
ROOT = Path('./.claude/worktrees/arc2'); DIST = ROOT / 'dist'
src, name = DIST / 'arc2-kaggle-v45', 'arc2-kaggle-v47'
out = DIST / name; shutil.rmtree(out, ignore_errors=True); out.mkdir()
for p in src.rglob('*'):
    rel = p.relative_to(src)
    if rel.parts[0] == 'dataset-upload' or '__pycache__' in rel.parts or p.is_dir(): continue
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    if rel.as_posix() in ('configs/comparison.json', 'MANIFEST.json', 'src/arc2/trial.py'): shutil.copyfile(p, out / rel)
    else: os.link(p, out / rel)
t = out / 'src/arc2/trial.py'; code = t.read_text()
old = '''    out = {}
    for run in runs:
        picks = guesses(run.candidates, [t["input"] for t in run.task["test"]])
        out.setdefault(run.id.split("@")[0].split("#")[0],
                       [{"attempt_1": pick[0][0], "attempt_2": pick[-1][0]} for pick in picks])
    return out'''
new = '''    pooled = {}  # independent copies (entries id#k) of a task: candidates pooled, agreement counted per entry
    for run in runs:
        entry = run.id.split("@")[0]
        task, cands = pooled.setdefault(entry.split("#")[0], (run.task, []))
        cands.extend({**c, "copy": (entry, c.get("copy", 0))} for c in run.candidates)
    out = {}
    for name, (task, cands) in pooled.items():
        picks = guesses(cands, [t["input"] for t in task["test"]])
        out[name] = [{"attempt_1": pick[0][0], "attempt_2": pick[-1][0]} for pick in picks]
    return out'''
assert code.count(old) == 1; t.write_text(code.replace(old, new))
path = out / 'configs/comparison.json'; study = json.loads(path.read_text()); study['name'] = name
path.write_text(json.dumps(study, indent=2) + '\n')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/arena')]
from build_arena import manifest
idn = json.loads((out / 'MANIFEST.json').read_text())
idn.update(study=name, files=manifest(out))
idn['source'] = 'arc2-kaggle-v45 bundle + trial.submission merging copies (scripts/arc2_weights/build_v47.py)'
(out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
runpy.run_path(str(out / 'bootstrap.py'))['verify'](out)
spec = importlib.util.spec_from_file_location('build_arc2', ROOT / 'docs/validation-20261009/build_arc2.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
for cache in list(out.rglob('__pycache__')): shutil.rmtree(cache)
info = b.kaggle_files(out, 'tool')
for extra in ('submission.ipynb', 'kernel-metadata.json'): (out / extra).unlink()
template = DIST / 'arc2-kaggle-v46-kernel'
nb = json.loads((template / 'submission.ipynb').read_text()); code = ''.join(nb['cells'][1]['source'])
V45_BLOB = 'e0774df49525ef46df5337d46f55de4c7fc5863abcb7d22c50cd76de9f8c6156'  # v45 runtime archive (v46 notebook)
assert code.count(V45_BLOB) == 1
code = code.replace(V45_BLOB, info['sha256']).replace('arc2-kaggle-v45-runtime', f'{name}-runtime').replace(
    'arc2-kaggle-v45.tar.gz.blob', f'{name}.tar.gz.blob')
anchor = "'--end-epoch', str(START + RUN_SECONDS), '--options',"
assert code.count(anchor) == 1
code = code.replace(anchor, "'--end-epoch', str(START + RUN_SECONDS), '--copies', '2', '--options',")
assert 'arc2-kaggle-v45' not in code and "def put(data)" in code
compile(code, 'nb', 'exec'); nb['cells'][1]['source'] = code.splitlines(keepends=True)
k = DIST / f'{name}-kernel'; shutil.rmtree(k, ignore_errors=True); k.mkdir()
(k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
m = json.loads((template / 'kernel-metadata.json').read_text()); m.update(id='richardcsaky/' + name, title=name)
m['dataset_sources'] = [info['dataset'] if d.endswith('arc2-kaggle-v45-runtime') else d for d in m['dataset_sources']]
assert 'keithtyser' not in json.dumps(m) and info['dataset'] in m['dataset_sources']
(k / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(info, indent=1)); print(m['dataset_sources'])
