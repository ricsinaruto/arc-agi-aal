"""arc2-kaggle-v46 (22 Oct): v45 unchanged (same runtime dataset, overlay, options) except the notebook's own writes of
submission.json go through a .part file and os.replace, as the driver's already do. v45's rerun ended in Kaggle's
"A system error. Please try resubmitting" (ref 57534350, 0 bytes); v44 with the same notebook scored 62.50.
Build only: no push. usage: build_v46.py"""
import json, shutil
from pathlib import Path
DIST = Path('./.claude/worktrees/arc2/dist')
src, name = DIST / 'arc2-kaggle-v45-kernel', 'arc2-kaggle-v46'
nb = json.loads((src / 'submission.ipynb').read_text()); code = ''.join(nb['cells'][1]['source'])
edits = [
    ("OUT = Path('/kaggle/working/submission.json')\n",
     "OUT = Path('/kaggle/working/submission.json')\n\n\ndef put(data):\n"
     "    # whole or not at all: a notebook cut while writing never leaves half a file\n"
     "    part = OUT.with_name(OUT.name + '.part'); part.write_text(json.dumps(data)); os.replace(part, OUT)\n"),
    ("    OUT.write_text(json.dumps(repaired({}, tests)))  # before anything that can fail\n",
     "    put(repaired({}, tests))  # before anything that can fail\n"),
    ("        OUT.write_text(json.dumps(final))\n", "        put(final)\n"),
]
for old, new in edits:
    assert code.count(old) == 1, old
    code = code.replace(old, new)
assert 'OUT.write_text' not in code
compile(code, 'nb', 'exec'); nb['cells'][1]['source'] = code.splitlines(keepends=True)
k = DIST / f'{name}-kernel'; shutil.rmtree(k, ignore_errors=True); k.mkdir()
(k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
m = json.loads((src / 'kernel-metadata.json').read_text()); m.update(id='richardcsaky/' + name, title=name)
assert 'richardcsaky/arc2-kaggle-v45-runtime' in m['dataset_sources']  # v45's runtime, unchanged
(k / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + '\n')
print(k, m['dataset_sources'])
