"""arc2-kaggle-v48: v45 (v37 + start fixes) with effort medium instead of xhigh; nothing else.
Same runtime dataset (arc2-kaggle-v45-runtime), overlay and models; notebook options differ in effort only.
usage: build_v48.py"""
import json, os, shutil, subprocess, sys
from pathlib import Path
DIST = Path('./.claude/worktrees/arc2/dist')
src, name = DIST / 'arc2-kaggle-v45-kernel', 'arc2-kaggle-v48'
nb = json.loads((src / 'submission.ipynb').read_text()); code = ''.join(nb['cells'][1]['source'])
i = code.index("'--options', '") + len("'--options', '"); j = code.index("'", i)
opts = json.loads(code[i:j]); assert opts['roster']['set']['effort'] == 'xhigh'
new = json.loads(code[i:j]); new['roster']['set']['effort'] = 'medium'
code2 = code[:i] + json.dumps(new) + code[j:]
diff = [(a, b) for a, b in zip(code.split('\n'), code2.split('\n')) if a != b]
assert len(diff) == 1 and diff[0][0].replace('"xhigh"', '"medium"') == diff[0][1], diff
compile(code2, 'nb', 'exec')
check = ("import json, sys; from arc2 import kaggle; kaggle.check_options(json.loads(sys.argv[1])); print('ok')")
done = subprocess.run([sys.executable, '-c', check, json.dumps(new)], capture_output=True, text=True,
                      env={**os.environ, 'PYTHONPATH': str(DIST / 'arc2-kaggle-v45/src'), 'PYTHONDONTWRITEBYTECODE': '1'})
assert done.returncode == 0 and done.stdout.strip() == 'ok', done.stderr[-2000:]
nb['cells'][1]['source'] = code2.splitlines(keepends=True)
k = DIST / f'{name}-kernel'; shutil.rmtree(k, ignore_errors=True); k.mkdir()
(k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
m = json.loads((src / 'kernel-metadata.json').read_text()); m.update(id='richardcsaky/' + name, title=name)
assert 'keithtyser' not in json.dumps(m)
(k / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + '\n')
print('options check ok; diff:', diff[0][1][-160:])
