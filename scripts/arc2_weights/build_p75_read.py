"""BASE12 on c120 with a p75 checkpoint in its trained format: the v9b INT8 tree (AR W4A16 experts, heal
overlay, t75 mask, INT8 dense, int4 KV, MTP 2) with base12's arc2 harness grafted on, plus the 32 GiB CPU tier.
Window and reply cap stay base12's; streams come from the formula on this model's blocks.
c647: p50 ckpt-647 the same way on its own tree (k256c647 kf: INT3 direct experts, t50 mask, INT8 dense, int4 KV,
MTP 2, default runner, so live slots work), streams sized with the uncharged Mamba blocks in the reserve.
p50d7c294: harness exactly base15e on arc3sub's tree for the 41-layer student (mkP50d7.py: drop_layers
mask, cut config, BF16 dense, no graph capture list), streams by the formula on its predicted pool.
usage: build_p75_read.py ARM   (ARM: a key of ARMS, c647, p50d7c294, p50d7c942, v56c268 or c1147)
       build_p75_read.py p50d7ck|p75ck DATASET MANIFEST_SHA TAG   (base15e256 on an SFT checkpoint)"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'dist/arc2-rtx-c120-hx-base12-cpu32-v3b'  # the anchor's bundle: study, harness and data
READ = ROOT.parents[1] / 'worktrees/v84sub/dist/b2-v9b-p75c613-int8-check-v1'  # the trained-format serving tree
P50 = ROOT.parents[1] / 'worktrees/v84sub/dist/b2-k256c647-kf-c110-9h-v1'  # p50 c647 as arc3sub serves it
ARMS = {'c893': 'flash-input/models/arc-heal/p75-qad-c893',  # gate rehearsal only: same tree, known-good overlay
        'c1047': 'flash-input/models/arc-heal/p75-qad-c1047',
        'c1147': 'flash-input/models/arc-heal/p75-arc2-c1147'}
KV_BLOCKS = 2680  # c893 INT8 read, ARC_CACHE_LAYOUT at 64 seqs
# base12's 28.5 reserved blocks a stream on v3b's table is 57,344 tokens; the same tokens cost 31 on this layout's
# (3,200-token blocks, 14 fixed vs 3,136 and 11: MTP 2)
PER_STREAM = 31
SLOTS = round(KV_BLOCKS * 0.96 / PER_STREAM)
SEQS = SLOTS + 4
TIER_ENV = {'ARC_OFFLOAD_EAGLE_TAIL': '1', 'ARC_OFFLOAD_NO_EAGLE_POP': '1'}
ARC2_RUN = '''        if trial["kind"] == "arc2":
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

arm = sys.argv[1]
if arm == 'c647':
    READ = P50
    KV_BLOCKS = 2245  # b2-k256c268-kf-ho10x11-9h-v3 boot at 64 seqs, same tree and sizes
    # p75's 83 slots at 31: 148 preemptions, ~8.5 Mamba blocks a request held outside the gate's 3 (lessons 907): +6
    PER_STREAM = 37
    SLOTS = round(KV_BLOCKS * 0.96 / PER_STREAM)
    SEQS = SLOTS + 4
    model, name = None, 'arc2-rtx-c120-hx-base12-p50c647-cpu32-v9'
elif arm == 'p50d7c294':
    READ = P50.parent / 'b2-p50d7c294-kf-ho10x11-9h-v2'  # v2: arc3sub's PLE gate fix for the cut config
    BASE = ROOT / 'dist/arc2-rtx-c120-hx-base15e-cpu32-v3b'
    KV_BLOCKS = 2680  # SFT's prediction, no boot yet: slots go live on the measured pool
    # base15e's 32.7 reserved blocks on v3b's table cost 35.6 on this layout (as 28.5 -> 31), + the 6 Mamba blocks
    PER_STREAM = 42
    SLOTS = round(KV_BLOCKS * 0.96 / PER_STREAM)
    SEQS = SLOTS + 4
    model, name = None, 'arc2-rtx-c120-hx-base15e-p50d7c294-cpu32-v9'
elif arm == 'p50d7c942':  # the c294 read again, only the overlay changed (arc3sub's tc942 tree); 10M stop
    READ = P50.parent / 'b2-p50d7tc942-ex5x10-v1'
    BASE = ROOT / 'dist/arc2-rtx-c120-hx-base15e-cpu32-v3b'
    KV_BLOCKS = 2680
    PER_STREAM = 42
    SLOTS = round(KV_BLOCKS * 0.96 / PER_STREAM)
    SEQS = SLOTS + 4
    model, name = None, 'arc2-rtx-c120-hx-base15e-p50d7c942-cpu32-v9'
elif arm == 'v56c268':  # v56 on p50 c268 (arc3sub's tree, 2,245 blocks at its boot); reserve as base15e's
    READ = P50.parent / 'b2-k256c268-kf-ho10x11-9h-v3'
    BASE = ROOT / 'dist/arc2-rtx-c120-hx-v56c120-cpu32-v3b'
    KV_BLOCKS = 2245
    PER_STREAM = 42
    SLOTS = round(KV_BLOCKS * 0.96 / PER_STREAM)
    SEQS = SLOTS + 4
    model, name = None, 'arc2-rtx-c120-hx-v56-k256c268-cpu32-v9'
elif arm == 'c1147':  # base17 exactly on c1047's recipe; 32.7 -> 36 on this table + the 6 Mamba blocks
    BASE = ROOT / 'dist/arc2-rtx-c120-hx-base17-cpu32-v3b'
    PER_STREAM = 42
    SLOTS = round(KV_BLOCKS * 0.96 / PER_STREAM)
    SEQS = SLOTS + 4
    model, name = ARMS[arm], 'arc2-rtx-c120-hx-base17-p75c1147-cpu32-v9b'
elif arm in ('p50d7ck', 'p75ck'):  # base15e256 (256K, 15M) on each pass-end checkpoint of the SFT runs
    # ARM DATASET MANIFEST_SHA TAG: the exact INT3-RTN overlay with its BF16 dense, swapped into a tree as mkP50d7x5.py
    # does. p50d7ck: arc3sub's tc942 tree (41 layers, cut config). p75ck: the V9C p75 tree (t75 nodrop mask) with the
    # dense served BF16 as trained (no INT8 at load, as tc942's). base19w256's 40.25 reserved blocks a 256K stream on
    # v3b's table cost 44.3 here (as 32.7 -> 36), + the 6 Mamba blocks
    BASE = ROOT / 'dist/arc2-rtx-c120-hx-base15e256-cpu32-v3b'
    READ = P50.parent / ('b2-p50d7tc942-ex5x10-v1' if arm == 'p50d7ck' else 'b2-v9c-p75c893-kf-c110-9h-v1')
    DATASET, MSHA = sys.argv[2], sys.argv[3]
    # late: MSHA "late" boots while SFT's export runs (nebius_asset_late, nebius 8a3db3d5d); the
    # runner fetches the overlay when its asset manifest lands and checks every file against it; the bundle then reads
    # the overlay manifest unpinned (logged) instead of a sha fixed at build time
    LATE = MSHA == 'late'
    KV_BLOCKS = 2680 if arm == 'p50d7ck' else 2400  # p75: BF16 dense in place of INT8 (prediction; the guard holds KV)
    PER_STREAM = 50
    SLOTS = round(KV_BLOCKS * 0.96 / PER_STREAM)
    SEQS = SLOTS + 4
    model, name = None, f'arc2-rtx-c120-hx-base15e256-{sys.argv[4]}-cpu32-v9'
else:
    model, name = ARMS[arm], f'arc2-rtx-c120-hx-base12-p75{arm}-cpu32-v9b'
out = ROOT / 'dist' / name
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(READ, out, ignore=shutil.ignore_patterns('__pycache__', 'build-manifest.json'))
# the harness as base12 ran it; cache_probe: base12's proves a load from the host tier; rest: modules arc2 imports
shutil.copytree(BASE / 'src/arc2', out / 'src/arc2', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(BASE / 'data', out / 'data', ignore=shutil.ignore_patterns('__pycache__'))
if arm in ('p50d7ck', 'p75ck') and LATE:
    ov = out / 'src/arena/expert_overlay.py'
    t = ov.read_text()
    a = '''    digest = spec["manifest_sha256"]\n    if not isinstance'''
    b = '''    digest = spec["manifest_sha256"]\n    if digest == "late":  # nebius_asset_late: checked file by file against the export's asset manifest\n        return dict(spec)\n    if not isinstance'''
    c = '''    if hashlib.sha256(raw).hexdigest() != expected_sha256:'''
    d = '''    if expected_sha256 == "late":  # nebius_asset_late: the runner checked every file against the asset manifest\n        print(json.dumps({"event": "expert_overlay_late", "manifest_sha256": hashlib.sha256(raw).hexdigest()}), flush=True)\n    elif hashlib.sha256(raw).hexdigest() != expected_sha256:'''
    assert t.count(a) == 1 and t.count(c) == 1
    ov.write_text(t.replace(a, b).replace(c, d))
for rel in ('context_drain', 'engine_watchdog', 'rollout_record', 'cache_probe'):
    shutil.copyfile(BASE / f'src/arena/{rel}.py', out / f'src/arena/{rel}.py')
run = out / 'src/arena/run.py'
text = run.read_text()
anchor = '        if trial["kind"] == "level-bench":\n'
if text.count(anchor) != 1:
    raise SystemExit(f'run.py anchor found {text.count(anchor)} times')
run.write_text(text.replace(anchor, ARC2_RUN + anchor))

study = json.loads((BASE / 'configs/comparison.json').read_text())
read = json.loads((READ / 'configs/comparison.json').read_text())
if arm in ('p50d7ck', 'p75ck'):
    read['server']['expert_overlay'].update(dataset=f'richardcsaky/{DATASET}', manifest_sha256=MSHA)
    read['model_datasets'] = [f'richardcsaky/{DATASET}']
    if arm == 'p75ck':
        read['server']['dense_kernels']['int8_weights'] = []
        read['fp8_dense'] = {**read.get('fp8_dense', {}), 'families': [], 'toggle': None, 'calib': None,
                             'note': 'p75 SFT checkpoint: BF16 dense as trained, no INT8 at load, no FP8'}
for key in ('server', 'serving_profile', 'research_checkpoint', 'nebius_asset_skip', 'nebius_boot_disk_gib',
            'ple_prefill_bound', 'fp8_dense', 'hero_v2', 'arena_variant', 'model_datasets', 'tail_multidrop',
            'probe_retry', 'cache_experiment'):
    if key in read:
        study[key] = read[key]
study['name'] = name
study['cache_experiment']['cpu_cache_gib'] = 32
if arm in ('p50d7ck', 'p75ck') and LATE:
    study['nebius_asset_late'] = {f'flash-input/datasets/richardcsaky/{DATASET}': {'wait_seconds': 3600, 'gib': 34}}
if model:
    study['research_checkpoint']['heal_overlay'] = f'/assets/{model}'
    study['nebius_assets'] = [x for x in read['nebius_assets'] if 'arc-heal' not in x] + [model]
study['generation_slots'] = study['active_limit'] = SLOTS
study['focus_oversubscription'].update(focus_slots=SLOTS, gate_blocks_per_stream=PER_STREAM)
study['arc2'].pop('live_seqs', None)
study['arc2'].update(blocks_per_stream=PER_STREAM, live_slots_max=SEQS,
                     weights={'name': sys.argv[4] if arm in ('p50d7ck', 'p75ck') else {'c647': 'p50-k256-c647', 'p50d7c294': 'p50d7-c294', 'c1147': 'p75-arc2-c1147', 'v56c268': 'p50-k256-c268', 'p50d7c942': 'p50d7-tc942'}.get(arm, f'p75-qad-{arm}')},
                     kv=f'v87b mechanism: exact reserving gate, {SLOTS} slots by the formula on {KV_BLOCKS} blocks')
server = study['server']
server['max_num_seqs'] = SEQS
top = min(256, SEQS * (1 + server['mtp_tokens']))
top = max(server['max_cudagraph_capture_size'], -(-top // 8) * 8)
sizes = server['cudagraph_capture_sizes']
sizes += list(range(sizes[-1] + 8, top + 1, 8))
server['max_cudagraph_capture_size'] = top
trial = study['trials'][0]
if arm == 'p50d7c942':  # ends at the 10M mark (the c294 read ran to its 15M ceiling)
    trial['marks'] = {k: v for k, v in trial['marks'].items() if v <= 10_000_000}
trial['server_env'] = {**read['trials'][0]['server_env'], **TIER_ENV}
(out / 'configs/comparison.json').write_text(json.dumps(study, indent=2) + '\n')

manifest = json.loads((READ / 'MANIFEST.json').read_text())
manifest.update(study=name, source=f'{READ.name} + {BASE.name} harness (scripts/arc2_weights/build_p75_read.py)')
manifest['files'] = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(out.rglob('*'))
                     if p.is_file() and p.suffix != '.ipynb' and '__pycache__' not in p.parts
                     and p.name != 'MANIFEST.json'}
(out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
__import__('runpy').run_path(str(out / 'bootstrap.py'))['verify'](out)
print(out, 'slots', SLOTS, 'seqs', SEQS, 'graphs', top)
