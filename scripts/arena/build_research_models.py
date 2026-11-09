"""Build and verify compatibility pilots at the three-block serving concurrency."""
import argparse
import json
import pathlib
import runpy
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from build_arena import manifest
from arena.llamacpp_profiles import profile
from arena.research_checkpoint import REPO, REVISION


def swift_gameplay(study, replay):
    """Apply the approved context-only capacity change, retaining the actual scheduler horizon."""
    window, floor = 49152, 880
    rows = [r for r in replay['rows'] if r['window'] == window and r['streams'] == 37]
    scenarios = {(r['trace'], r['pool'], r['floor']) for r in rows}
    expected = {(t, p, f) for t in ('A', 'B') for p in ('all', 'late') for f in (0, 8800)}
    if (replay['num_blocks'] != floor or replay['margin'] != .96 or scenarios != expected
            or not all(r['passes'] and r['seeds'] >= 3 and not r['time_share_occupancy_over_pool'] for r in rows)):
        raise ValueError('Swift needs all eight measured-layout replay scenarios to pass')
    study['base_hero_v2_provenance'] = study.pop('hero_v2')
    # The base run's optional diagnostic patch is installed through its pruning overlay.
    # Swift deliberately has no such overlay; do not inherit the trial's request for that patch.
    study.pop('stall_dump_seconds', None)
    for trial in study.get('trials', []):
        trial.get('server_env', {}).pop('ARENA_STALL_DUMP', None)
    study['windows'] = [window]
    study['blocks_plan']['min_pool_blocks'] = floor
    capacity = int(floor * .96)
    study['focus_oversubscription'].update(
        full_windows_fit_projected=capacity // 25, blocks_per_full_window=25,
        gate_capacity_projected=capacity, gate_blocks_per_stream=(capacity * 1000 // 37) / 1000)
    study['early_stop'] = {'always': True, 'at_seconds': 1800, 'score_above': None, 'retry_seconds': 300}
    study['research_capacity'] = dict(measured_blocks=897, replay_floor=floor, window=window,
                                     streams=37, measured_layout='swift-cache-layout.json',
                                     replay='swift-37-capacity.json',
                                     changes=['checkpoint', 'no base pruning/overlay', 'agent context and admission density'])
    study['research_scope'] = ('Approved Swift 37-stream gameplay pilot with 49,152-token agent context, '
                               '16,384 output/reserve unchanged, first 1,800 seconds of the unchanged '
                               '32,400-second three-block scheduler. Measured 897-block pool; replay floor 880.')


def build(kind, gameplay=False, version=None):
    source = ROOT / 'dist/arena-campaign-blocks3-plans-9h-v1'
    version = version or ('v1' if gameplay else 'v2')
    target = ROOT / f'dist/research-{kind}-3blocks-{"gameplay" if gameplay else "compat"}-{version}'
    if target.exists():
        raise FileExistsError(f'Preserve uploaded bundles: {target}')
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        'dataset-upload', '*.tar.gz', '*.ipynb', 'build-manifest.json'))
    names = ('serving.py', 'research_checkpoint.py', 'models/swift15-flash-nvfp4.json') if kind == 'swift' else (
        'llamacpp.py', 'llamacpp_profiles.py', 'llamacpp_run.py')
    for name in names:
        shutil.copyfile(ROOT / 'src/arena' / name, target / 'src/arena' / name)
    path = target / 'configs/comparison.json'
    study = json.loads(path.read_text())
    study.pop('campaign', None)
    study.update(notebook_seconds=3540, kernel_timeout=3600)
    study['research_scope'] = ('Three-block serving compatibility and capacity measurement only; no gameplay claim. '
                               'Base expert overlay and pruning deliberately absent for new checkpoints.')
    if kind == 'swift':
        study['research_checkpoint'] = {'repo': REPO, 'revision': REVISION, 'path': '/assets/research-20260930/swift'}
        study['server'].pop('expert_overlay', None)
        study['server'].pop('expert_prune', None)
        if gameplay:
            replay = ROOT / 'docs/validation-20260930/swift-37-capacity.json'
            swift_gameplay(study, json.loads(replay.read_text()))
            shutil.copyfile(replay, target / replay.name)
            shutil.copyfile(ROOT / 'runs/research-swift-compat/cache-layout.json', target / 'swift-cache-layout.json')
        else:
            study['trials'] = [{'name': 'swift-compat', 'kind': 'server-check', 'wall_seconds': 120, 'competition': False}]
    else:
        name = 'flash-next-gsq-iq3s'
        prof = profile(name)
        for key in ('cache_admission', 'cache_experiment', 'isolate_trial_cache', 'focus_oversubscription'):
            study.pop(key, None)
        study.update(backend='llamacpp', generation_slots=37,
                     served_model=prof['served_model'], model_revision=prof['revision'],
                     windows=[prof['window']], reasoning_effort=prof['reasoning_effort'],
                     llamacpp={'profile': name, 'repo': prof['repo'], 'revision': prof['revision'],
                               'files': prof['files'], 'llama_cpp': prof['llama_cpp'], 'kv_unified': False,
                               'serve_check': {'depths': [1024], 'streams': [1, 4, 37], 'output_tokens': 64,
                                               'recall_depth': 2048}},
                     server={'backend': 'llamacpp', 'max_model_len': prof['window'], 'slots': 37,
                             'context': 37 * prof['window'], 'prefix_caching': True, 'mtp_tokens': 0})
        if gameplay:
            study['early_stop'] = {'always': True, 'at_seconds': 1800, 'score_above': None, 'retry_seconds': 300}
            study['llamacpp']['serve_check']['vision'] = True
            study['research_scope'] = ('User-approved 1800-second IQ3_S gameplay pilot, first block of unchanged '
                                       '32400-second three-block scheduler, with independent per-level allowances. '
                                       'No base expert pruning or overlay; llama.cpp q4_0 cache and native mmproj.')
    path.write_text(json.dumps(study, indent=1) + '\n')
    identity = json.loads((source / 'MANIFEST.json').read_text())
    identity['files'] = manifest(target)
    (target / 'MANIFEST.json').write_text(json.dumps(identity, indent=2) + '\n')
    runpy.run_path(str(target / 'bootstrap.py'))['verify'](target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('swift', 'iq3'))
    parser.add_argument('--gameplay', action='store_true')
    parser.add_argument('--version')
    args = parser.parse_args()
    print(build(args.kind, args.gameplay, args.version))
