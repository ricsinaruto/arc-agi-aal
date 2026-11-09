"""launch3 RUN BOUND | launch2 RUN BOUND | close RUN ACTUAL 'note'  (rows only via tools/ops/records.py)"""
import json, os, re, subprocess, sys, time
from pathlib import Path
ROOT = Path('.'); PY = str(ROOT / '.venv/bin/python'); REC = [PY, str(ROOT / 'tools/ops/records.py')]
HERE = Path(__file__).resolve().parent; L = HERE / 'launch'; L.mkdir(exist_ok=True)
AUTH = "mmrtx; for the two fit-data API runs "
def rec(*a):
    p = subprocess.run(REC + list(a), capture_output=True, text=True); print(' '.join(a[:2]), '->', p.returncode, (p.stdout + p.stderr).strip()[-300:])
    if p.returncode: raise SystemExit('records.py failed')
def go(run, bound, what, cmd, env):
    now = int(time.time())
    rec('reserve', run, '--arm', 'mmrtx', '--platform', 'modal+openrouter', '--gpu', 'CPU', '--bound', str(bound), '--max-seconds', '7200',
        '--auth', AUTH, '--what', what, '--kind', 'research-preparation', '--set', f'start_epoch={now}')
    p = subprocess.run(['python3', str(ROOT / 'scripts/openrouter_cli.py'), '--', str(ROOT / '.venv/bin/modal'), 'run', '--detach', *cmd],
                       cwd=HERE, env={**os.environ, **env}, capture_output=True, text=True)
    out = p.stdout + p.stderr; (L / f'{run}.log').write_text(out)
    m = re.search(r'(ap-[A-Za-z0-9]+)', out); app = m.group(1) if m else None
    print(run, 'app', app, 'rc', p.returncode, out[-400:])
    if not app or p.returncode: raise SystemExit('launch failed')
    rec('ledger', run, '--set', f'app_id={app}'); rec('queue', run, '--set', f'app_id={app}'); rec('cpu-app', app)
    (L / f'{run}.epoch').write_text(str(now))
if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'launch3':
        go(a[2], a[3], "MiniMax-M3.1-Flash via API plays the 5 calibration games (2 copies) + 5 other games (1 copy) on the v8.5p harness, "
           "128K window, saving every conversation: fit data for the expert-pruning masks and held-out data for the KL screens. Judged by "
           "data volume (~1.8M generated tokens), 0 errors.",
           ['fit_arc3.py', '--run', a[2], '--target', 'minimax/minimax-m3.1-flash', '--instances', '15', '--token-stop', '120000',
            '--wall', '3600', '--provider', 'minimax', '--effort', 'medium', '--window', '131072', '--cap', '8192', '--patch', 'concise'],
           {'MM_BUNDLE': str(ROOT / '.claude/worktrees/dfr-exact/dist/sabl-v85p-c100-300k-v1')})
    elif a[1] == 'launch2':
        go(a[2], a[3], "MiniMax-M3.1-Flash via API solves 24 ARC-AGI-2 training tasks (fit data) + 8 hard38 evaluation tasks (held-out, "
           "measurement only), text grids, saving every conversation for the pruning masks and KL screens. Judged by data volume, 0 errors.",
           ['fit_arc2.py::api_main', '--run', a[2], '--name', 'mm', '--target', 'minimax/minimax-m3.1-flash',
            '--overrides', (HERE / 'fit-arc2.json').read_text(), '--limit', '0', '--wall', '2400', '--bound', str(a[3])],
           {'SCREEN_BUNDLE': str(HERE / 'arc2-bundle')})
    else:
        rec('settle', a[2], 'complete', '--cause', 'completed', '--set', f'actual_usd={a[3]}', '--set', f'reconciliation={a[4]}')
