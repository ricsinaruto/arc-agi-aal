"""Build the wide-tiles validation pilot (30 September; docs/validation-20260930/kernel-profile.md section 6).

One container, two server starts of the blocks3 campaign serving (the decode-sweep bundle: 56 sequences, graphs every
4 from 32 to 56, torch profiler configured on both):

1. ``wide-ref`` (kernel-quality reference): the serving as it runs today (phase-5 dense plan).
2. ``wide-gate`` (kernel-quality gate, report-only): the same with ``ARENA_DENSE_KERNELS`` = the served dense setting
   + ``mid_wide`` + ``mid_wide_shapes`` [[448, 2560]] (buckets 40 / 48 / 56 and the prune12 router, tuned in the
   kernel warmup). Nothing else differs between the two starts.

Each trial measures, in order:
- **drift** (arena.kernel_gate): teacher-forced ``prompt_logprobs`` on the 40 recorded texts of
  ``data/kernel_quality`` (a prefill path the wide tiles do not serve: a control), then greedy decoding of 48 prompts at
  concurrency 48 (every step a 33-56-row decode batch, the wide buckets' path) with top-20 logprobs; the reference
  repeats both at the same concurrencies (the noise bar);
- **speed** (arena.kernel_profile, real recorded contexts): R56 (cold: all 56 prefilled), R40 x 3, R24, and one torch
  trace at R40.

Pre-stated pass bar (recorded in the study as ``kernel_wide.pass_bar``):
- speed: the gate's mean R40 step (3 windows) at least 0.5 ms below the reference's, and the gap larger than both arms'
  window spread (max - min);
- drift: greedy (decode-path) mean KL(ref || gate) and top-1 disagreement each <= 1.25 x the reference's own repeat's
  (same server, same batch), and the teacher-forced control within the same bar.

  uv run python scripts/arena/build_kernel_wide.py --source <decode-sweep bundle> --contexts <contexts.json>
"""

import argparse
import hashlib
import json
import pathlib
import runpy
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from build_arena import manifest  # noqa: E402

from arc_infra import dense_kernels_patch  # noqa: E402

NAME = 'arena-kernel-wide-v1'
BASE_COMMIT = '7c701f9'  # the source the decode-sweep bundle carries for the replaced files
REPLACED = ('src/arena/throughput_bench.py', 'src/arena/kernel_gate.py', 'src/arc_infra/dense_kernels_patch.py')
ADDED = ('src/arena/kernel_profile.py', 'src/arc_infra/dense_wide_runtime.py')
DECODE = {"window_seconds": 15}
POINTS = [
    {"label": "R56", "content": "real", "streams": 56, "settle_seconds": 330, "min_remaining_seconds": 300},
    {"label": "R40-a", "content": "real", "streams": 40},
    {"label": "R40-b", "content": "real", "streams": 40},
    {"label": "R40-c", "content": "real", "streams": 40, "profile": True, "profile_seconds": 4},
    {"label": "R24", "content": "real", "streams": 24, "subset": "longest"},
]
SPEC = {"contexts_file": "bench/kernel-profile-contexts.json", "output_tokens": 2048, "synthetic_streams": 1,
        "calibrate_tokens": 24576, "window_seconds": 15, "settle_seconds": 90, "quiet_seconds": 1.0,
        "sample_seconds": 0.5, "inject_lead_seconds": 4, "load_lead_seconds": 2, "points": POINTS}
QUALITY = {"greedy": True, "gsm8k": False, "report_only": True, "greedy_concurrency": 48,
           "greedy_repeat_concurrency": 48}
GREEDY_COPIES = 3  # 16 recorded greedy prompts x 3 = 48 concurrent decodes
PASS_BAR = {
    "speed": "gate mean R40 step (3 windows) <= reference mean - 0.5 ms, and the gap > each arm's window spread",
    "drift": "greedy (decode, batch 48) mean KL(ref||gate) and top-1 disagreement <= 1.25 x the reference repeat's; "
             "teacher-forced control within the same 1.25 x bar",
}


def base_sha(relative):
    blob = subprocess.run(['git', 'show', f'{BASE_COMMIT}:{relative}'], cwd=ROOT, check=True,
                          capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


def quality_inputs():
    data = json.loads((ROOT / 'data/kernel_quality/prompts.json').read_text())
    greedy = [{**g, 'index': f"{g['index']}-{copy}"} for copy in range(GREEDY_COPIES) for g in data['greedy']]
    return {**data, 'greedy': greedy}


def build(source, contexts, target):
    if target.exists():
        raise FileExistsError(f'Preserve built bundles: {target}')
    source_study = json.loads((source / 'configs/comparison.json').read_text())
    if 'decode_sweep' not in source_study or source_study['trials'][0]['name'] != 'sweep':
        raise ValueError('The source must be the decode-sweep bundle')
    for relative in REPLACED:
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != base_sha(relative):
            raise ValueError(f'{relative} in the source bundle differs from {BASE_COMMIT}')
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        'dataset-upload', '*.tar.gz', '*.ipynb', 'build-manifest.json', 'kernel-metadata.json'))
    for relative in REPLACED + ADDED:
        shutil.copyfile(ROOT / relative, target / relative)
    data = json.loads(pathlib.Path(contexts).read_text())
    if data.get('format') != 'arc-kernel-profile-contexts-v1' or len(data['contexts']) < 56:
        raise ValueError('kernel-wide: 56 contexts in the kernel-profile v1 format are required')
    (target / 'bench').mkdir(exist_ok=True)
    shutil.copyfile(contexts, target / SPEC['contexts_file'])
    quality = target / 'configs/kernel-quality'
    quality.mkdir(parents=True, exist_ok=True)
    (quality / 'prompts.json').write_text(json.dumps(quality_inputs()))
    shutil.copyfile(ROOT / 'data/kernel_quality/gsm8k.jsonl', quality / 'gsm8k.jsonl')
    study = json.loads(json.dumps(source_study))
    served = dense_kernels_patch.validate(study['server']['dense_kernels'])
    wide = dense_kernels_patch.validate({**served, 'mid_wide': True,
                                         'mid_wide_shapes': [list(s) for s in dense_kernels_patch.PRUNED_ROUTER_SHAPES]})
    shared = dict(study['trials'][0].get('server_env') or {})
    shared['ARENA_TORCH_PROFILER'] = '1'
    bench = {"prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None, "points": [],
             "kernel_profile": SPEC}
    study['name'] = NAME
    study['trials'] = [
        {'name': 'wide-ref', 'kind': 'kernel-quality', 'role': 'reference', 'competition': False, 'wall_seconds': 600,
         'server_env': dict(shared), 'quality': dict(QUALITY), 'bench': bench},
        {'name': 'wide-gate', 'kind': 'kernel-quality', 'role': 'gate', 'reference': 'wide-ref', 'competition': False,
         'wall_seconds': 480, 'server_env': {**shared, 'ARENA_DENSE_KERNELS': json.dumps(wide, sort_keys=True)},
         'quality': dict(QUALITY), 'bench': bench}]
    tb = study['throughput_bench']
    tb.pop('stream_sweep', None)
    tb.update(prompt_tokens=[], concurrency=[], shared_prefix=None, points=[])
    study['kernel_wide'] = {
        'pass_bar': PASS_BAR, 'reference_dense_kernels': served, 'gate_dense_kernels': wide,
        'source_bundle': str(source), 'replaced': list(REPLACED), 'added': list(ADDED) + [SPEC['contexts_file']],
        'quality_inputs': {'teacher_forced': len(quality_inputs()['teacher_forced']),
                           'greedy': len(quality_inputs()['greedy']),
                           'sha256': hashlib.sha256((quality / 'prompts.json').read_bytes()).hexdigest()},
        'contexts_sha256': hashlib.sha256(pathlib.Path(contexts).read_bytes()).hexdigest()}
    study.pop('decode_sweep', None)
    study.update(notebook_seconds=2640, kernel_timeout=2700, wall_seconds=600)
    (target / 'configs/comparison.json').write_text(json.dumps(study, indent=2) + '\n')
    identity = json.loads((source / 'MANIFEST.json').read_text())
    identity['files'] = manifest(target)
    (target / 'MANIFEST.json').write_text(json.dumps(identity, indent=2) + '\n')
    runpy.run_path(str(target / 'bootstrap.py'))['verify'](target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--contexts', required=True)
    parser.add_argument('--target', default=str(ROOT / 'dist' / NAME))
    args = parser.parse_args()
    print(build(pathlib.Path(args.source), args.contexts, pathlib.Path(args.target)))
