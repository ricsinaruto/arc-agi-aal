"""Build the kernel-profile pilot bundle (30 September; docs/validation-20260930/kernel-profile.md).

The decode-sweep bundle (arena-decode-sweep-v1: the blocks3 campaign serving at 56 sequences with CUDA graphs every 4
from 32 to 56) is copied unchanged except for

- ``src/arena/kernel_profile.py`` (new) and ``src/arena/throughput_bench.py`` (its hook), checked first against the
  source they replace;
- ``bench/kernel-profile-contexts.json`` (``scripts/kernel_profile_inputs.py``);
- ``configs/comparison.json``: one serving-bench trial ``kprof`` (wall 1,000 s, so the bootstrap's per-trial limit is
  1,600 s) with the torch profiler configured (``ARENA_TORCH_PROFILER=1``), the stream sweep replaced by the
  ``kernel_profile`` spec below, and the Modal test's 2,400 s envelope.

The server (vLLM argv, kernels, overlays, prune mask, KV dtype, graph ladder) is byte-identical to the decode sweep's.

  uv run python scripts/arena/build_kernel_profile.py --source <decode-sweep bundle> --contexts <contexts.json>
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

NAME = 'arena-kernel-profile-v1'
BLOCK_TOKENS, FIXED_BLOCKS, POOL_BLOCKS = 3136, 9, 1240  # int4 blocks3 layout (decode sweep: 1,240 assumed)
DECODE = {"profile": True, "profile_seconds": 4}
POINTS = [
    # Real recorded contexts (the Duck request shape). The first point prefills all 56 prompts cold.
    {"label": "R56", "content": "real", "streams": 56, "settle_seconds": 330, **DECODE, "min_remaining_seconds": 420},
    {"label": "R40", "content": "real", "streams": 40, **DECODE},
    {"label": "R40-greedy", "content": "real", "streams": 40,
     "overrides": {"temperature": 0.0, "top_p": 1.0, "top_k": -1}},
    {"label": "R40-mixed", "content": "real", "streams": 40, "profile": True, "profile_seconds": 6,
     "inject": {"prefix_tokens": 24576, "tail_tokens": 2000, "rate": 0.6}, "min_remaining_seconds": 150},
    {"label": "R24", "content": "real", "streams": 24, "subset": "longest", **DECODE},
    {"label": "R40-cpu4", "content": "real", "streams": 40, "load": {"cpu_burners": 4}},
    {"label": "R40-cpuall", "content": "real", "streams": 40, "load": {"cpu_burners": "all"}},
    {"label": "R40-tokenize", "content": "real", "streams": 40, "load": {"tokenize_rate": 4}},
    {"label": "R40-stream", "content": "real", "streams": 40, "stream": True},
    {"label": "R40-seed", "content": "real", "streams": 40, "overrides": {"seed": 7}},
    {"label": "R40-repeat", "content": "real", "streams": 40},
    # Synthetic word-salad prompts at the same token counts (the decode sweep's bodies). Cold prefill first.
    {"label": "S40", "content": "synthetic", "streams": 40, "settle_seconds": 260, **DECODE,
     "min_remaining_seconds": 330},
    {"label": "S40-noseed", "content": "synthetic", "streams": 40, "drop": ["seed"]},
    {"label": "S33", "content": "synthetic", "streams": 33},
    {"label": "S32-greedy", "content": "synthetic", "streams": 32,
     "overrides": {"temperature": 0.0, "top_p": 1.0, "top_k": -1}},
]
SPEC = {"contexts_file": "bench/kernel-profile-contexts.json", "output_tokens": 2048, "synthetic_streams": 40,
        "calibrate_tokens": 24576, "window_seconds": 15, "settle_seconds": 90, "quiet_seconds": 1.0,
        "sample_seconds": 0.5, "inject_lead_seconds": 4, "load_lead_seconds": 2, "points": POINTS}
REPLACED = ('src/arena/throughput_bench.py',)
ADDED = ('src/arena/kernel_profile.py',)


# The source the pilot's hook was written against (main merged with the frontier branch, before the pilot).
BASE_COMMIT = '7c701f9'


def head_sha(relative):
    blob = subprocess.run(['git', 'show', f'{BASE_COMMIT}:{relative}'], cwd=ROOT, check=True,
                          capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


def build(source, contexts, target):
    if target.exists():
        raise FileExistsError(f'Preserve built bundles: {target}')
    source_study = json.loads((source / 'configs/comparison.json').read_text())
    if 'decode_sweep' not in source_study or source_study['trials'][0]['name'] != 'sweep':
        raise ValueError('The source must be the decode-sweep bundle')
    for relative in REPLACED:
        # The bundle's copy must be the committed source this worktree edits (no other drift is carried).
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != head_sha(relative):
            raise ValueError(f'{relative} in the source bundle differs from {BASE_COMMIT}; rebuild from matching source')
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        'dataset-upload', '*.tar.gz', '*.ipynb', 'build-manifest.json', 'kernel-metadata.json'))
    for relative in REPLACED + ADDED:
        shutil.copyfile(ROOT / relative, target / relative)
    data = json.loads(pathlib.Path(contexts).read_text())
    if data.get('format') != 'arc-kernel-profile-contexts-v1' or len(data['contexts']) < 56:
        raise ValueError('kernel-profile contexts: 56 contexts in the v1 format are required')
    (target / 'bench').mkdir(exist_ok=True)
    shutil.copyfile(contexts, target / SPEC['contexts_file'])
    study = json.loads(json.dumps(source_study))
    out = SPEC['output_tokens']
    local = [c['tokens_local'] for c in data['contexts']]
    need = sum(-(-(n + out) // BLOCK_TOKENS) + FIXED_BLOCKS for n in local)
    if need > int(study['kv_margin'] * POOL_BLOCKS):
        raise ValueError(f'kernel profile: {need} blocks at 56 real streams do not fit the gate')
    server_env = dict(study['trials'][0].get('server_env') or {})
    server_env['ARENA_TORCH_PROFILER'] = '1'
    study['name'] = NAME
    study['trials'] = [{'name': 'kprof', 'kind': 'serving-bench', 'competition': False, 'wall_seconds': 1000,
                        'server_env': server_env}]
    bench = study['throughput_bench']
    bench.pop('stream_sweep', None)
    bench['kernel_profile'] = SPEC
    study['kernel_profile_pilot'] = {
        'source_bundle': str(source), 'source_manifest_sha256': hashlib.sha256(
            (source / 'MANIFEST.json').read_bytes()).hexdigest(),
        'replaced': list(REPLACED), 'added': list(ADDED) + [SPEC['contexts_file']],
        'contexts_source': data['source'], 'blocks_at_56_real_streams': need,
        'server_unchanged_from': 'arena-decode-sweep-v1 (plus ARENA_TORCH_PROFILER=1: --profiler-config)'}
    study.update(notebook_seconds=2340, kernel_timeout=2400, wall_seconds=1000)
    (target / 'configs/comparison.json').write_text(json.dumps(study, indent=2) + '\n')
    identity = json.loads((source / 'MANIFEST.json').read_text())
    identity['files'] = manifest(target)
    (target / 'MANIFEST.json').write_text(json.dumps(identity, indent=2) + '\n')
    runpy.run_path(str(target / 'bootstrap.py'))['verify'](target)
    return target, need


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--contexts', required=True)
    parser.add_argument('--target', default=str(ROOT / 'dist' / NAME))
    args = parser.parse_args()
    print(build(pathlib.Path(args.source), args.contexts, pathlib.Path(args.target)))
