"""Pinned candidate serving without Flash-specific model/PLE admission checks."""
import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from arc_infra.model_templates import text_only
from arc_infra.serving import dataset_mount, load_recipe
from arc_infra.serving_fork import (
    GPU_MEMORY_UTILIZATION_ENV,
    fork_identity,
    stage_fork_bundle,
)

from .cache import BlockBudget
from .candidate_profiles import NEX, PROFILES, from_study
from .serving import configure, lab_overrides
from .serving_core.config import Config
from .serving_core.jail import Jail, needed
from .serving_core.serving import preflight, verify_hardware, worker_probe


def checkpoint_mount(study, manifest, input_root):
    """Select an explicitly reviewed mirror; content hashes remain authoritative."""
    sources = study.get('model_sources', [])
    if not sources:
        return dataset_mount(manifest['dataset'], input_root)
    if sources != [manifest.get('model_source')]:
        raise ValueError('Candidate model source differs from reviewed inventory')
    parts = sources[0].split('/')
    if (len(parts) != 5 or any(p in ('', '.', '..') for p in parts)
            or parts[2].lower() not in ('pytorch', 'transformers') or not parts[4].isdigit()):
        raise ValueError('Expected pinned Kaggle PyTorch or Transformers model version')
    parts[2] = parts[2].lower()
    path = Path(input_root) / 'models' / Path(*parts)
    if not path.is_dir():
        raise FileNotFoundError(f'Missing pinned candidate model mount: {path}')
    return path


def verify_model(path, manifest, cache=None):
    """Check every required file against the reviewed inventory, before GPU loading.

    ``cache`` (a JSON file in the server's working directory): a restart in the same container reuses the hashes
    of files whose path, size, mtime and inode are unchanged since this container hashed them (the drift trials
    restart the server twice); anything else is hashed again."""
    path = Path(path)
    cache = None if cache is None else Path(cache)
    known = json.loads(cache.read_text()) if cache is not None and cache.is_file() else {}
    verified, reused = {}, 0
    for name, spec in manifest['files'].items():
        file = path / name
        if not file.is_file() or file.stat().st_size != spec['bytes']:
            raise RuntimeError(f'Candidate checkpoint file missing or wrong size: {name}')
        stat = file.stat()
        key = {'path': str(file.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}
        if known.get(name) == {**key, 'sha256': spec['sha256']}:
            digest, reused = spec['sha256'], reused + 1
        else:
            with file.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != spec['sha256']:
            raise RuntimeError(f'Candidate checkpoint hash mismatch: {name}')
        verified[name] = {**key, 'sha256': digest}
    if cache is not None:
        cache.write_text(json.dumps(verified, indent=1))
    index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    if not set(index.values()) <= manifest['files'].keys():
        raise RuntimeError('Checkpoint index references unverified shards')
    profile = next((p for p in PROFILES.values()
                    if (p.model, p.revision) == (manifest['repo'], manifest['revision'])), None)
    if profile is None:
        raise RuntimeError('Unreviewed candidate checkpoint identity')
    config = json.loads((path / 'config.json').read_text())
    mtp_count = sum('mtp' in name.lower() for name in index)
    if (config['architectures'] != [profile.architecture]
            or (config.get('quantization_config') or {}).get('quant_method') != profile.quant_method
            or config.get('text_config', config)['max_position_embeddings'] != (profile.declared_context or profile.native_context)
            or mtp_count != profile.mtp_tensors):
        raise RuntimeError('Candidate architecture, quantization, context or MTP inventory changed')
    if profile.quant_method is None and config.get('dtype', config.get('torch_dtype')) != 'bfloat16':
        raise RuntimeError('Expected native BF16 checkpoint; no implicit quantization')
    if profile.cli_quantization == 'modelopt_mixed' and config['quantization_config'].get('quant_algo') != 'MIXED_PRECISION':
        raise RuntimeError('Expected mixed NVFP4/FP8 checkpoint configuration')
    return {'repo': manifest['repo'], 'revision': manifest['revision'],
            'verified_files': len(manifest['files']), 'all_file_hashes_verified': True,
            **({'hashes_reused_from_this_container': reused} if cache is not None else {})}


def candidate_args(profile, kv_cache_dtype=None):
    """The profile's extra serve arguments, with its ``--kv-cache-dtype`` replaced by ``kv_cache_dtype`` when given
    (a trial's ARENA_KV_CACHE_DTYPE: the Holo4 BF16-KV drift reference on the same attention backend)."""
    args = list(profile.extra_args)
    if kv_cache_dtype is None:
        return args
    if kv_cache_dtype not in ('bfloat16', 'fp8'):
        raise ValueError('A candidate KV override is bfloat16 or fp8')
    if '--kv-cache-dtype' not in args:
        raise ValueError(f'{profile.name} pins no --kv-cache-dtype to override')
    args[args.index('--kv-cache-dtype') + 1] = kv_cache_dtype
    return args


def configure_candidate(recipe, profile=NEX, kv_cache_dtype=None):
    """Keep process ownership/runtime setup; replace only the model command identity."""
    extra = candidate_args(profile, kv_cache_dtype)
    recipe.SERVED_MODEL_NAME = profile.model
    recipe.ANALYZER_CONTEXT = profile.native_context
    os.environ[recipe.MTP_TOKENS_ENV] = str(profile.mtp_tokens)
    os.environ[recipe.MTP_INDEX_SHARE_FOR_ITERATION_ENV] = '0'
    os.environ.pop(recipe.MTP_DYNAMIC_BATCH_SCHEDULE_ENV, None)
    os.environ.pop(recipe.MOE_BACKEND_ENV, None)
    original = recipe.server_command

    def command(*args, **kwargs):
        argv = original(*args, **kwargs)
        if not profile.mtp_tokens and '--speculative-config' in argv:
            # n-gram (prompt-lookup) drafts need no MTP weights; only an MTP launch is rejected.
            spec = json.loads(argv[argv.index('--speculative-config') + 1])
            if spec.get('method') == 'mtp':
                raise RuntimeError('Pinned checkpoint has no MTP weights; speculative launch rejected')
        index = argv.index('--quantization') + 1
        if argv[index] != 'modelopt_fp4':
            raise RuntimeError('Pinned serving command changed')
        if profile.cli_quantization is None:
            del argv[index - 1:index + 1]
        else:
            argv[index] = profile.cli_quantization
        argv[argv.index('--tool-call-parser') + 1] = profile.tool_parser
        argv[argv.index('--reasoning-parser') + 1] = profile.reasoning_parser
        argv.extend(extra)
        return argv

    recipe.server_command = command


async def ready(recipe, identity, *, timeout, base_url, profile=NEX):
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        while time.monotonic() < deadline:
            if not recipe.alive(identity['pid'], identity['start_ticks']):
                raise RuntimeError('Candidate server exited during startup; inspect server.log')
            try:
                response = await client.get(base_url.rstrip('/') + '/models')
                response.raise_for_status()
                models = response.json().get('data', [])
                if len(models) != 1 or models[0].get('id') != profile.model:
                    raise RuntimeError('Candidate server returned the wrong model identity')
                if models[0].get('max_model_len') != profile.native_context:
                    raise RuntimeError(f'Candidate server did not preserve native {profile.native_context} context')
                return models
            except httpx.HTTPError:
                await asyncio.sleep(1)
    raise TimeoutError(f'{profile.model} server not ready within {timeout} seconds')


# Server-environment overrides a candidate server applies (arena.serving.lab_overrides keys).
CANDIDATE_LAB = {'kv_cache_dtype', 'ready_guard', 'prefix_cache', 'prompt_tokens_details', 'stall_dump_seconds'}


def candidate_configure_options(study, lab):
    """``arena.serving.configure`` keywords beyond the cache/graph ones: the study's prefix-cache patches and stall
    dump (hero-v2 components on a candidate), each replaceable by the trial's server environment."""
    settings = study.get('cache_experiment', {})
    options = {}
    prefix = lab['prefix_cache'] if 'prefix_cache' in lab else settings.get('prefix_cache')
    if prefix is not None:
        options['prefix_cache'] = prefix
    if lab.get('ready_guard') is not None:
        options['ready_guard_seconds'] = lab['ready_guard']
    if lab.get('prompt_tokens_details'):
        options['prompt_tokens_details'] = True
    if lab.get('stall_dump_seconds') is not None:
        options['stall_dump_seconds'] = lab['stall_dump_seconds']
    return options


def served_kv_dtype(profile, override=None):
    """The KV dtype the candidate's command asks for (``auto``: vLLM resolves it from the checkpoint)."""
    if override is not None:
        return override
    args = list(profile.extra_args)
    return args[args.index('--kv-cache-dtype') + 1] if '--kv-cache-dtype' in args else 'auto'


def start(study, working, input_root):
    profile = from_study(study)
    if study['server']['max_model_len'] != profile.native_context or study['server']['prefix_caching'] is not True:
        raise ValueError(f'Candidates require native {profile.native_context} context and prefix caching')
    working = Path(working)
    working.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {'model': profile.model, 'revision': profile.revision, 'stages': {}}
    recipe = None
    stage = 'hardware'
    try:
        record['hardware'] = verify_hardware()
        stage = 'sandbox'
        probe = worker_probe(working, Jail() if needed() else None)
        if probe.returncode or '"type": "ready"' not in probe.stdout:
            raise RuntimeError('Candidate sandbox preflight failed: ' + probe.stderr)
        record['sandbox'] = json.loads(probe.stdout.splitlines()[0])
        stage = 'checkpoint_verification'
        manifest = json.loads(profile.manifest.read_text())
        if (manifest['repo'], manifest['revision']) != (profile.model, profile.revision):
            raise ValueError('Candidate checkpoint inventory identity mismatch')
        model = checkpoint_mount(study, manifest, input_root)
        record['checkpoint'] = verify_model(model, manifest, cache=working / 'verified-checkpoint.json')
        if study.get('weights_overlay'):
            # LoRA test (docs/posttrain/lora-test-plan.md): replaced tensors from a pinned overlay dataset,
            # applied to a copy of the verified shards; every patched shard is hash-checked.
            from .weights_overlay import materialise_for_study

            model, record['weights_overlay'] = materialise_for_study(
                study['weights_overlay'], model, manifest, input_root, working)
        record['stages'][stage] = time.monotonic() - started
        stage = 'runtime_setup'
        bundle = stage_fork_bundle(input_root, working / 'fork-bundle')
        os.environ[GPU_MEMORY_UTILIZATION_ENV] = study['server']['gpu_memory_utilization']
        # A trial's server environment (arena.serving.lab_overrides): the KV dtype of a pinned-KV profile, an explicit
        # KV byte pool (the drift trials' prompt_logprobs transients), the prefix-cache patches, the ready guard,
        # exact cached-token usage and the stall-only dump. Absent: the study's own settings, as before.
        lab = lab_overrides()
        unsupported = sorted(set(lab) - CANDIDATE_LAB)
        if unsupported:
            raise ValueError(f'Candidate servers do not take these overrides: {unsupported}')
        kv_bytes = int(os.environ.get('ARENA_KV_CACHE_MEMORY_BYTES', '0'))
        settings = study.get('cache_experiment', {})
        record['server_overrides'] = {**{k: v for k, v in lab.items() if k != 'ready_guard'},
                                      **({'ready_guard': list(lab['ready_guard'])} if 'ready_guard' in lab else {}),
                                      **({'kv_cache_memory_bytes': kv_bytes} if kv_bytes else {})}
        recipe = load_recipe(working, input_root, bundle_dir=bundle,
                             setup_sha256=fork_identity()['fork_sha256'],
                             generation_slots=study['generation_slots'], prefix_caching=True,
                             kv_cache_memory_bytes=kv_bytes, kv_cache_dtype='auto',
                             max_num_batched_tokens=study['server']['max_num_batched_tokens'])
        configure(recipe, working, graph_capture_size=study['server'].get('max_cudagraph_capture_size'),
                  graph_metrics=study['server'].get('cudagraph_metrics', False),
                  reclaim_skipped_blocks=settings.get('reclaim_skipped_blocks', False),
                  **candidate_configure_options(study, lab))
        configure_candidate(recipe, profile, lab.get('kv_cache_dtype'))
        record['runtime'] = recipe.verify_and_extract_runtime(
            recipe.resolve_runtime_dir(), full_layer_hashes=False, scan_extracted_caches=False)
        # Hash-verified shared runtime files, not a candidate PLE module.
        record['unused_flash_runtime_patch'] = recipe.patch_ple_layer()
        tuning = recipe.resolve_vllm_tuning()
        env, record['runtime_environment'] = recipe.runtime_environment(
            deep_preload_validation=False, tuning=tuning)
        record['tuning'] = tuning
        record['stages'][stage] = time.monotonic() - started
        stage = 'model_startup'
        identity = recipe.start_server(model, env, tuning=tuning)
        record['models'] = asyncio.run(ready(recipe, identity,
                                            timeout=study['startup_timeout_seconds'],
                                            base_url=recipe.BASE_URL, profile=profile))
        record['server_identity'] = recipe.update_ready_identity(identity)
        stage = 'native_context_preflight'
        cfg = Config(model=profile.model, model_revision=profile.revision, reasoning_effort=profile.effort,
                     context_window=profile.native_context,
                     generation_slots=study['generation_slots'], base_url=recipe.BASE_URL)
        native = asyncio.run(preflight(cfg))
        native['profile'] = {**study['server'], 'model': profile.model,
                             'kv_cache_dtype': served_kv_dtype(profile, lab.get('kv_cache_dtype'))}
        record['native_preflight'] = native
        if text_only(profile.model):
            from .tool_preflight import check

            stage = 'native_tool_handoff'
            record['native_tool_handoff'] = asyncio.run(check(cfg, working / 'native-tool-preflight.json'))
        budget = BlockBudget.load(working, study['kv_margin'])
        (working / 'cache-layout.json').write_text(json.dumps(budget.layout, indent=2))
        record.update(status='ready', seconds=time.monotonic() - started)
    except BaseException as exc:
        record.update(status='error', failed_stage=stage, error=f'{type(exc).__name__}: {exc}')
        if recipe is not None:
            record['cleanup'] = recipe.stop_owned_server()
        raise
    finally:
        (working / 'candidate-preflight.json').write_text(json.dumps(record, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('study', 'working', 'input-root'):
        parser.add_argument('--' + arg, required=True)
    args = parser.parse_args()
    start(json.loads(Path(args.study).read_text()), args.working, args.input_root)


if __name__ == '__main__':
    main()
