"""Build the speculative-decoding pilot bundles (30 September; docs/validation-20260930/spec-decode.md).

Source: the decode-sweep bundle (the frontier agent's ``--decode-sweep`` build, which is the blocks3 campaign serving
exactly: int4 KV, prefix-match-unit 16, prune12 + GPTQ-ARC + Marlin, ple_uva / embed_host, prefix cache v2, 0.96
margin; only its sequences, graphs and bench differ). Each arm changes only:

- ``server.mtp_tokens`` k (0, 1 or 2); for k > 0 the known-good draft stack: ``mamba_ssm_cache_dtype`` bfloat16 (the
  MTP config hook would otherwise make the GDN state float32; MTP-0 already runs bf16), NvFp4 draft experts and the
  16K draft vocabulary (draft-only: rejection sampling keeps the output distribution);
- 44 sequences and CUDA graphs at [1, 2, 4, 8, 16, 24, 32, 40, 44] streams x (k + 1) tokens, so 24 / 32 / 40 streams
  run unpadded in every arm and 37 pads to 40 as in blocks3;
- the bench: ``arena.spec_bench`` on recorded arena contexts (``scripts/spec_decode_inputs.py``);
- the 4-bit block follows k (``kv4_block_tokens``: 3,136 for k <= 1, 3,200 for k = 2).

    PYTHONPATH=src uv run python scripts/arena/build_spec_decode.py --mtp 1 --contexts <spec-contexts.json>
"""
import argparse
import json
import pathlib
import runpy
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from build_arena import manifest  # noqa: E402

SOURCE = pathlib.Path('./.claude/worktrees/agent-a2f5982a0052e7ded/dist/arena-decode-sweep-v1')
FILES = ('arena/spec_bench.py', 'arena/throughput_bench.py', 'arena/serving.py', 'arc_infra/metrics.py',
         'arc_infra/serving.py', 'arc_infra/serving_kernels_patch.py', 'arc_infra/expert_prune_runtime.py',
         'qwen_baselines/serving.py')
LADDER = [1, 2, 4, 8, 16, 24, 32, 40, 44]
# v3 (30 September, user: "go ahead with the proposed mtp pilots"): 30 / 36 stream graphs, clean windows (3,072-token
# forced replies, five consecutive quiet samples) and the verify-path drift check.
LADDER_V3 = [1, 2, 4, 8, 16, 24, 30, 32, 36, 40, 44]
SPEC_V3 = {
    "contexts": "bench/spec-contexts.json",
    "decode_counts": [24, 30, 32, 36, 37, 40],
    "output_tokens": 3072,
    "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20},
    "settle_seconds": 120, "quiet_samples": 5, "window_seconds": 30, "sample_seconds": 1.0,
    "drift": {"prompts": 16, "output_tokens": 256, "top_logprobs": 10, "teacher_concurrency": 4},
    "greedy": {"prompts": 16, "output_tokens": 256, "orders": ["concurrent"]},
}
PRUNE25 = {"mask": "configs/expert-prune/prune25-layer.json",
           "sha256": "6ea3dfd2373f64d09f71eea1dd3939011100917226368f61a31ee07eb4cca8cc"}
SEQUENCES = 44
SPEC = {
    "contexts": "bench/spec-contexts.json",
    "decode_counts": [24, 32, 37, 40],
    "output_tokens": 1024,
    "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20},
    "settle_seconds": 90, "window_seconds": 25, "sample_seconds": 1.0,
    "greedy": {"prompts": 16, "output_tokens": 256, "orders": ["concurrent", "sequential"]},
}


def build(mtp, contexts, version='v1', prune25=False, v3=False):
    target = ROOT / f'dist/arena-spec-sweep{"-p25" if prune25 else ""}-mtp{mtp}-{version}'
    if target.exists():
        raise FileExistsError(f'Preserve built bundles: {target}')
    shutil.copytree(SOURCE, target, ignore=shutil.ignore_patterns(
        'dataset-upload', '*.tar.gz', '*.ipynb', 'build-manifest.json'))
    for name in FILES:
        shutil.copyfile(ROOT / 'src' / name, target / 'src' / name)
    (target / 'bench').mkdir(exist_ok=True)
    shutil.copyfile(contexts, target / SPEC['contexts'])
    path = target / 'configs/comparison.json'
    study = json.loads(path.read_text())
    server = study['server']
    if (server.get('mtp_tokens') != 0 or server['serving_kernels'].get('kv_cache_4bit') != 'int4'
            or server.get('prefix_match_unit') != 16 or 'decode_sweep' not in study):
        raise ValueError('The source must be the MTP-0 int4 decode-sweep bundle')
    before = {k: server.get(k) for k in ('mtp_tokens', 'max_num_seqs', 'max_cudagraph_capture_size',
                                         'cudagraph_capture_sizes')}
    before['serving_kernels'] = dict(server['serving_kernels'])
    server['mtp_tokens'] = mtp
    server['max_num_seqs'] = study['generation_slots'] = SEQUENCES
    server['cudagraph_capture_sizes'] = [s * (mtp + 1) for s in (LADDER_V3 if v3 else LADDER)]
    if prune25:
        shutil.copyfile(ROOT / 'data/expert_prune/prune25-layer.json', target / PRUNE25['mask'])
        server['expert_prune'] = dict(PRUNE25)
    server['max_cudagraph_capture_size'] = server['cudagraph_capture_sizes'][-1]
    if mtp:
        server['serving_kernels'].update(mamba_ssm_cache_dtype='bfloat16', draft_experts='nvfp4',
                                         draft_vocab='arena16k')
    bench = study['throughput_bench']
    study['throughput_bench'] = {"prompt_tokens": [], "concurrency": [], "output_tokens": 64,
                                 "shared_prefix": None, "points": [], "sanity": bench.get("sanity"),
                                 "spec_sweep": dict(SPEC_V3 if v3 else SPEC)}
    env = dict(study['trials'][0].get('server_env') or {})
    env['ARENA_READY_GUARD'] = '180,1200'  # the MTP draft load and larger graphs lengthen start-up
    study['trials'] = [{"name": "spec", "kind": "serving-bench", "competition": False, "wall_seconds": 120,
                        "server_env": env}]
    data = json.loads(pathlib.Path(contexts).read_text())
    study.pop('decode_sweep')
    study['spec_sweep_pilot'] = {
        'mtp_tokens': mtp, 'source_bundle': str(SOURCE), 'server_before': before, 'prune25_layer': prune25,
        'version': version,
        'server_after': {k: server.get(k) for k in ('mtp_tokens', 'max_num_seqs', 'max_cudagraph_capture_size',
                                                    'cudagraph_capture_sizes', 'serving_kernels')},
        'contexts': {'source': data['source'], 'count': data['count'], 'mean_tokens': data['mean_tokens'],
                     'sha256': data['contexts_sha256']}}
    study.update(notebook_seconds=2340, kernel_timeout=2400, wall_seconds=120)
    path.write_text(json.dumps(study, indent=1) + '\n')
    identity = json.loads((SOURCE / 'MANIFEST.json').read_text())
    identity['files'] = manifest(target)
    (target / 'MANIFEST.json').write_text(json.dumps(identity, indent=2) + '\n')
    runpy.run_path(str(target / 'bootstrap.py'))['verify'](target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtp', type=int, required=True, choices=(0, 1, 2))
    parser.add_argument('--contexts', required=True)
    parser.add_argument('--version', default='v1')
    parser.add_argument('--prune25', action='store_true', help='the 25%% layer-aware pruned pool (prune25-layer)')
    parser.add_argument('--v3', action='store_true', help='clean windows + verify-path drift check')
    args = parser.parse_args()
    print(build(args.mtp, args.contexts, args.version, args.prune25, args.v3))
