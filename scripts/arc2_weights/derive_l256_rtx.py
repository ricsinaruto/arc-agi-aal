"""The paired h240 l256 arm (20 Oct): the k256 base bundle with only the served mask and overlay changed. Scratch.
usage: derive_l256_rtx.py <overlay manifest sha256>"""
import hashlib, json, os, runpy, shutil, subprocess, sys
from pathlib import Path
ROOT = Path('./.claude/worktrees/arc2'); DIST = ROOT / 'dist'
MASK = Path(__file__).resolve().parent / 'masks/prune-arc2-l256-v1.json'
OLD, NEW = 'arc-expert-int3-refit-arc2c240-k256-v1', 'arc-expert-int3-refit-arc2c240-l256-v1'
src, name = DIST / 'arc2-rtx-h240-v37set-k256-v3b', 'arc2-rtx-h240-v37set-l256-v3b'
out = DIST / name; shutil.rmtree(out, ignore_errors=True); out.mkdir()
for p in src.rglob('*'):
    rel = p.relative_to(src)
    if '__pycache__' in rel.parts or p.is_dir(): continue
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    if rel.as_posix() in ('configs/comparison.json', 'MANIFEST.json'): shutil.copyfile(p, out / rel)
    else: os.link(p, out / rel)
mask_rel = 'configs/expert-prune/prune-arc2-l256-v1.json'
shutil.copyfile(MASK, out / mask_rel)
path = out / 'configs/comparison.json'; study = json.loads(path.read_text()); study['name'] = name
assert study['server']['expert_overlay']['dataset'] == 'richardcsaky/' + OLD and study['model_datasets'] == ['richardcsaky/' + OLD]
study['server']['expert_prune'] = {'mask': mask_rel, 'sha256': hashlib.sha256((out / mask_rel).read_bytes()).hexdigest()}
study['server']['expert_overlay'] = {'dataset': 'richardcsaky/' + NEW, 'manifest_sha256': sys.argv[1], 'check': 'size'}
study['model_datasets'] = ['richardcsaky/' + NEW]
study['arc2']['weights'] = {'name': 'arc2c240-l256-refit', 'replaced': 'richardcsaky/' + OLD}
path.write_text(json.dumps(study, indent=2) + '\n')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/arena')]
from build_arena import manifest
idn = json.loads((out / 'MANIFEST.json').read_text())
idn.update(study=name, files=manifest(out))
idn['source'] = f'{src.name} with the l256 mask and overlay {NEW} (manifest {sys.argv[1][:8]}) only'
(out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
runpy.run_path(str(out / 'bootstrap.py'))['verify'](out)
a = json.loads((src / 'configs/comparison.json').read_text()); a['name'] = name
diff = [k for k in set(a) | set(study) if a.get(k) != study.get(k)]
print(name, 'top-level keys differing from the base:', sorted(diff),
      [k for k in set(a['server']) | set(study['server']) if a['server'].get(k) != study['server'].get(k)])
