"""Private Kaggle read kernels for the l4 arm (never a submission): the v49 notebook with only the save run's play
length, the driver options and (for a rebuilt bundle) the runtime dataset changed. The save run plays the public
placeholder's 240 training tasks with their solutions, so tok/s and right outputs are read alike for every variant.
usage: build_kaggle_reads.py VARIANT [--play 4500]   VARIANT: ctl | mtp1 | rk32 | car
  -> dist/l4-read-<variant>-kernel (submission.ipynb + kernel-metadata.json) and, for rk32/car, the runtime dataset
     folder dist/<bundle>/dataset-upload (pushed only on explicit approval)."""
import argparse, importlib.util, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ARC2 = Path('./.claude/worktrees/arc2')
TEMPLATE = ARC2 / 'dist/arc2-kaggle-v49-kernel'
V49_BLOB = '21db8b9fbb100a5e5ae84fabf774c2ce4dc953480a62451d134143f6ac1024bb'
BUNDLES = {'rk32': HERE / 'dist/arc2-kaggle-v49rk', 'car': HERE / 'dist/arc2-kaggle-v49car',
           'mtp1dbg': HERE / 'dist/arc2-kaggle-v49dbg', 'mtp1dbg2': HERE / 'dist/arc2-kaggle-v49dbg2', 'mtp1dbg3': HERE / 'dist/arc2-kaggle-v49dbg3', 'mtp1dbg4': HERE / 'dist/arc2-kaggle-v49dbg4', 'mtp1dbg5': HERE / 'dist/arc2-kaggle-v49dbg5', 'mtp1dbg6': HERE / 'dist/arc2-kaggle-v49dbg6', 'mtp1dbg7': HERE / 'dist/arc2-kaggle-v49dbg7', 'mtp1dbg8': HERE / 'dist/arc2-kaggle-v49dbg8', 'mtp1dbg9': HERE / 'dist/arc2-kaggle-v49dbg9', 'mtp1eag': HERE / 'dist/arc2-kaggle-v49eag', 'mtp1ple': HERE / 'dist/arc2-kaggle-v49ple'}
SAVE = 'SAVE_TASKS, SAVE_SECONDS, RUN_SECONDS = 240, 1800, 42300'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=('ctl', 'mtp1', 'mtp1dbg', 'mtp1dbg2', 'mtp1dbg3', 'mtp1dbg4', 'mtp1dbg5', 'mtp1dbg6', 'mtp1dbg7', 'mtp1dbg8', 'mtp1dbg9', 'mtp1eag', 'mtp1ple', 'rk32', 'car'))
    ap.add_argument('--play', type=int, default=4500)
    ap.add_argument('--start-limit', type=int, default=2700, help='no play by then ends the save run')
    ap.add_argument('--progress', type=int, default=300, help='status-line period, s')
    ap.add_argument('--hard', type=int, default=0, help='if set: the notebook process exits at this wall second')
    a = ap.parse_args()
    nb = json.loads((TEMPLATE / 'submission.ipynb').read_text())
    code = ''.join(nb['cells'][1]['source'])
    i = code.index("'--options', '") + len("'--options', '"); j = code.index("'", i)
    opts = json.loads((HERE / f'scripts/l4/runs/opts-{a.variant}.json').read_text())
    base = json.loads(code[i:j])
    assert {k: v for k, v in opts.items() if k in base} == base, 'run options must extend v49 options'
    code = code[:i] + json.dumps(opts) + code[j:]
    assert code.count(SAVE) == 1
    code = code.replace(SAVE, SAVE.replace('240, 1800,', f'240, {a.play},'))
    for old, new in (('PREVIEW_START_LIMIT = 2700', f'PREVIEW_START_LIMIT = {a.start_limit}'),
                     ('PROGRESS_SECONDS = 300', f'PROGRESS_SECONDS = {a.progress}')):
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    if a.hard:
        old = 'START = time.time()\n'
        assert code.count(old) == 1
        code = code.replace(old, old + (
            'import threading, shutil as _sh, glob as _gl\n'
            'def _l4_keep():  # per-reply evidence copied to the output before any hard stop\n'
            '    for f in _gl.glob("/tmp/arena/trials/*/events.jsonl") + _gl.glob("/kaggle/working/arc2/live-logs/*/events.jsonl"):\n'
            '        try:\n'
            '            d = Path("/kaggle/working/arc2/l4-events") / Path(f).parent.name; d.mkdir(parents=True, exist_ok=True)\n'
            '            _sh.copyfile(f, d / "events.jsonl")\n'
            '        except Exception as e:\n'
            '            print("l4 keep", e, flush=True)\n'
            'def _l4_loop():\n'
            '    while True:\n'
            '        time.sleep(60); _l4_keep()\n'
            'def _l4_stop():\n'
            f'    time.sleep(max(1, {a.hard} - (time.time() - START))); _l4_keep()\n'
            '    print("l4 hard stop", flush=True); os._exit(0)\n'
            'threading.Thread(target=_l4_loop, daemon=True).start()\n'
            'threading.Thread(target=_l4_stop, daemon=True).start()  # l4 read: wall cap\n'))
    meta = json.loads((TEMPLATE / 'kernel-metadata.json').read_text())
    if a.variant in BUNDLES:
        out = BUNDLES[a.variant]
        upload = out / 'dataset-upload' / 'dataset-metadata.json'
        if upload.exists():  # an uploaded runtime is kept: a rebuilt blob would not match the dataset on Kaggle
            dmeta = json.loads(upload.read_text())
            info = {'sha256': dmeta['description'].rsplit('sha256 ', 1)[1].rstrip('.'), 'dataset': dmeta['id']}
        else:
            spec = importlib.util.spec_from_file_location('build_arc2', HERE / 'docs/validation-20261009/build_arc2.py')
            b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
            for cache in list(out.rglob('__pycache__')):
                shutil.rmtree(cache)
            info = b.kaggle_files(out, 'tool')
            for extra in ('submission.ipynb', 'kernel-metadata.json'):
                (out / extra).unlink()
        assert code.count(V49_BLOB) == 1
        code = (code.replace(V49_BLOB, info['sha256']).replace('arc2-kaggle-v49-runtime', f'{out.name}-runtime')
                .replace('arc2-kaggle-v49.tar.gz.blob', f'{out.name}.tar.gz.blob'))
        meta['dataset_sources'] = [info['dataset'] if d.endswith('arc2-kaggle-v49-runtime') else d
                                   for d in meta['dataset_sources']]
    compile(code, 'nb', 'exec')
    nb['cells'][1]['source'] = code.splitlines(keepends=True)
    k = HERE / 'dist' / f'l4-read-{a.variant}-kernel'
    shutil.rmtree(k, ignore_errors=True); k.mkdir(parents=True)
    (k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    meta.update(id=f'richardcsaky/l4-read-{a.variant}', title=f'l4-read-{a.variant}')
    assert 'keithtyser' not in json.dumps(meta) and meta['competition_sources'] == ['arc-prize-2026-arc-agi-2']
    (k / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('built', k, meta['dataset_sources'])


if __name__ == '__main__':
    main()
