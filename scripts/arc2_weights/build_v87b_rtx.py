"""arc2 RO on the v87b submission serving stack. Base = the bytes of
arena-flash-competition-v87b-sub (runtime, wheels, draft head, bootstrap, serving config: MTP 2, fp8 dense, CPU
tier 32 GiB, PLE prefill bound, graphs). Declared exceptions (check_v87b_stack.py fails on anything else):
- harness: src/arc2 (arc2 HEAD), the arc2 trial-kind dispatch in src/arena/run.py, src/arena/engine_watchdog.py
  (HEAD: OUTAGE_WAIT, which the arc2 transport reads, and the start/stop pipe-hang fix), data/arc2, the study's
  arc2 section and trial, windows [131072] (RO's conversation window);
- weights: the arc2 k256 mask and its overlay replace t37 + dn4g8 (dn4g8 is a requant of the t37 overlay: its env
  and section go);
- streams (v87b's rule, its mechanism, our numbers): CacheGate primary (target 0.88, step 0.08, floor 0.72, charge
  = prompt + mean reply), v87b's reserving KVGate as the fallback on stale /metrics; ceiling 104 slots,
  max_num_seqs 112, graphs to min(256, seqs x 3);
- no time limit: wall 43,200 s and notebook wall + 4,800 s; the stop is the 20M-token mark;
- L4-match parts dropped: gate_match (41.1 x 32K cap) and the prefill throttle (2 slots x 1,500 tok/s).
Game-scheduler keys the arc2 kind never reads stay as v87b's. usage: build_v87b_rtx.py [SUFFIX]"""
import json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path('./.claude/worktrees/arc2'); DIST = ROOT / 'dist'
V87B = Path('./.claude/worktrees/v84sub/dist/arena-flash-competition-v87b-sub')
ANCHOR = DIST / 'arc2-rtx-c240-hx-convro-v3b'  # RO: arm, roster, data, mask, overlay
suffix = sys.argv[1] if len(sys.argv) > 1 else 'ro87b'
name = f'arc2-rtx-c240-hx-{suffix}-v3b'
DROP = ('dataset-upload', 'experiment.ipynb', 'kernel-metadata.json', 'build-manifest.json')
HEAD_FILES = ['src/arena/engine_watchdog.py'] + sorted(
    p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/arc2').rglob('*.py') if '__pycache__' not in p.parts)
MASK = 'configs/expert-prune/prune-arc2-k256-v1.json'
SLOTS, SEQS, WALL = 104, 112, 43200
dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', *HEAD_FILES], capture_output=True, text=True)
assert dirty.stdout == '', 'commit the harness files first: ' + dirty.stdout
commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()

out = DIST / name; shutil.rmtree(out, ignore_errors=True); out.mkdir()
for p in V87B.rglob('*'):
    rel = p.relative_to(V87B)
    if p.is_dir() or rel.parts[0] in DROP or '__pycache__' in rel.parts:
        continue
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    if rel.as_posix() in ('configs/comparison.json', 'MANIFEST.json', 'src/arena/run.py'):
        shutil.copyfile(p, out / rel)
    elif rel.as_posix() not in HEAD_FILES:
        os.link(p, out / rel)
for rel in HEAD_FILES:
    (out / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, out / rel)
for p in (ANCHOR / 'data/arc2').rglob('*'):
    if p.is_file():
        (out / p.relative_to(ANCHOR)).parent.mkdir(parents=True, exist_ok=True); os.link(p, out / p.relative_to(ANCHOR))
shutil.copyfile(ANCHOR / MASK, out / MASK)

# run.py: the arc2 trial kind, as in the v3b bundles (only this block differs from v87b)
run = out / 'src/arena/run.py'; text = run.read_text()
anchor = '        if trial["kind"] == "level-bench":\n'
block = '''        if trial["kind"] == "arc2":
            # ARC-AGI-2 arm (9 October): static tasks, verify-and-refine (src/arc2); no arcade, no scorecard.
            from arc2.trial import run_arc2

            report = asyncio.run(run_arc2(study, trial, source=args.source, output=folder, server_dir=args.server_dir,
                                          base_url=args.base_url, deadline_epoch=hard_end - 180,
                                          python=args.analysis_python))
            execution["report"] = report
            execution["status"] = report["status"]
            if report["status"] == "error":
                raise RuntimeError("arc2 trial failed: " + "; ".join(report["task_errors"])[:2000])
            raise _Done
'''
assert text.count(anchor) == 1 and 'arc2' not in text
run.write_text(text.replace(anchor, block + anchor))

study = json.loads((V87B / 'configs/comparison.json').read_text())
ro = json.loads((ANCHOR / 'configs/comparison.json').read_text())
study['name'] = name
# weights
srv = study['server']
srv['expert_prune'], srv['expert_overlay'] = ro['server']['expert_prune'], ro['server']['expert_overlay']
assert srv['expert_prune']['mask'] == MASK
study['model_datasets'] = [ro['server']['expert_overlay']['dataset'],
                           *[d for d in study['model_datasets'] if 'fp8-calib' in d]]
study.pop('dn4g8')
# streams
sizes = [s for s in srv['cudagraph_capture_sizes'] if s <= 160] + [192, 224, 256]
graphs = min(256, SEQS * (1 + int(srv['mtp_tokens'])))
srv.update(max_num_seqs=SEQS, cudagraph_capture_sizes=[s for s in sizes if s <= graphs], max_cudagraph_capture_size=graphs)
study['generation_slots'] = study['active_limit'] = SLOTS
study['windows'] = ro['windows']
# harness: RO's arc2 section without the L4-match caps, CacheGate admission
arc2 = {k: v for k, v in ro['arc2'].items() if k not in ('gate_match', 'l4_match', 'prefill', 'rtx_of')}
arc2['admission'] = {'mode': 'kv', 'target': 0.88, 'step': 0.08, 'floor': 0.72, 'charge': 1, 'max_age': 15,
                     'settle': 15, 'poll': 2}
arc2['stack'] = ('v87b-sub serving (scripts/arc2_weights/build_v87b_rtx.py, arc2 %s); exceptions in its docstring'
                 % commit[:8])
study['arc2'] = arc2
env = {k: v for k, v in study['trials'][0]['server_env'].items() if not k.startswith('ARC_EXPERT_INT3_')}
trial = {k: v for k, v in ro['trials'][0].items() if k != 'server_env'}
trial.update(server_env=env, wall_seconds=WALL,
             marks={'v22-5k': 2300000, '4.6M': 4600000, 'v13-12h': 9260000, '10M': 10000000, '14M': 14000000,
                    '15M': 15000000, '18M': 18000000, '20M': 20000000})
study['trials'] = [trial]
study['wall_seconds'], study['notebook_seconds'] = WALL, WALL + 4800
(out / 'configs/comparison.json').write_text(json.dumps(study, indent=2) + '\n')

sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'scripts/arena')]
from build_arena import manifest
idn = json.loads((V87B / 'MANIFEST.json').read_text())
idn.update(study=name, files=manifest(out),
           source=f'arena-flash-competition-v87b-sub + arc2 harness {commit[:8]} (scripts/arc2_weights/build_v87b_rtx.py)')
(out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
__import__('runpy').run_path(str(out / 'bootstrap.py'))['verify'](out)
check = subprocess.run([sys.executable, str(ROOT / 'scripts/arc2_weights/check_v87b_stack.py'), str(out)])
assert check.returncode == 0, 'stack check failed'
print(out)
