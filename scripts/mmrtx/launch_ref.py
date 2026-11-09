"""launch_ref.py RUN : reserve the 2 x B200 row, launch ref + CPU preflight beside it, write the watch plan."""
import json, re, subprocess, sys, time
from pathlib import Path
ROOT = Path('.'); P = str(ROOT / '.venv/bin/python'); REC = [P, str(ROOT / 'tools/ops/records.py')]
MODAL = str(ROOT / '.venv/bin/modal'); HERE = Path(__file__).resolve().parent
run = sys.argv[1]
def rec(*a):
    p = subprocess.run(REC + list(a), capture_output=True, text=True); print(a[0], a[1], '->', p.returncode, (p.stdout + p.stderr).strip()[-200:])
    if p.returncode: raise SystemExit('records failed')
def launch(entry, *args):
    p = subprocess.run([MODAL, 'run', '--detach', f'ref_job.py::{entry}', *args], cwd=HERE, capture_output=True, text=True)
    out = p.stdout + p.stderr; (HERE / f'launch-{entry}-{args[1]}.log').write_text(out)
    m = re.search(r'"app_id": "(ap-[A-Za-z0-9]+)"', out)
    if not m or p.returncode: raise SystemExit(f'{entry} launch failed: {out[-500:]}')
    return m.group(1)
now = int(time.time())
rec('reserve', run, '--arm', 'mmrtx', '--platform', 'modal', '--gpu', '2 x B200', '--gpu-count', '2', '--bound', '25',
    '--max-seconds', '10800', '--auth', 'approved the 2 x B200 reference job (~1 h, in the mmrtx)',
    '--what', 'Runs the full, unpruned MiniMax-M3.1-Flash once on 2 B200s to measure which experts carry the routing on our calibration data '
    '(sets how many experts a 1-GPU build can drop), how often its sparse attention changes the blocks it reads (sizes a host-RAM KV design), '
    'its draft-model acceptance, and reference token probabilities for later quality checks. Judged by finishing all phases.',
    '--kind', 'research-preparation', '--allowance-exempt', 'approved one 2 x B200 reference job for mmrtx (~1 h, in its), ', '--set', f'start_epoch={now}')
app = launch('main_ref', '--run', run, '--sets', 'sets-v2')
rec('ledger', run, '--set', f'app_id={app}'); rec('queue', run, '--set', f'app_id={app}')
pf = launch('main_preflight', '--run', f'{run}-pf', '--sets', 'sets-v2')
rec('cpu-app', pf)
subprocess.run(['python3', str(HERE.parent / 'watch/mkplan.py'), 'ref', run, app, str(now)], cwd=HERE.parent / 'watch', check=True)
print(json.dumps({'run': run, 'app': app, 'preflight_app': pf, 'launched': now}))
