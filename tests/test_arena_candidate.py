import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from test_qwen_serving import import_fork

from arc_infra.model_templates import NEX_MODEL, NEX_REVISION
from arena import candidate_serving as candidate
from arena.cache_probe import payload
from arena.candidate_profiles import (
    NEX,
    QWEN_27_FP8,
    QWEN_27_NVFP4,
    UI_MATE_9B,
    UI_MATE_27B,
    from_study,
)

ROOT = Path(__file__).resolve().parents[1]


def test_actual_pinned_command_selects_compressed_tensors_without_mtp(tmp_path, monkeypatch):
    recipe = import_fork(tmp_path, monkeypatch)
    for key, value in {'TAAF_VLLM_MAX_NUM_SEQS': '25', 'TAAF_VLLM_ENABLE_PREFIX_CACHING': '1',
                       'TAAF_VLLM_MAX_NUM_BATCHED_TOKENS': '4096',
                       'TAAF_VLLM_MTP_TOKENS': '3'}.items():
        monkeypatch.setenv(key, value)
    candidate.configure_candidate(recipe)
    tuning = recipe.resolve_vllm_tuning()
    assert not recipe.validate_vllm_tuning_document(tuning)
    assert tuning['mtp_speculative_tokens'] == 0
    command = recipe.server_command(tmp_path / 'model', tuning=tuning)
    for flag, value in {'--quantization': 'compressed-tensors', '--served-model-name': NEX_MODEL,
                        '--max-num-seqs': '25', '--max-model-len': '262144',
                        '--max-num-batched-tokens': '4096'}.items():
        assert command[command.index(flag) + 1] == value
    assert '--speculative-config' not in command and '--enable-prefix-caching' in command
    with pytest.raises(RuntimeError, match='no MTP'):
        recipe.server_command(tmp_path, tuning={**tuning, 'mtp_speculative_tokens': 3})


def test_nex_probe_uses_candidate_model_and_native_no_thinking():
    body = payload(0, 4, model=NEX_MODEL)
    assert body['model'] == NEX_MODEL
    assert body['chat_template_kwargs'] == {'reasoning_effort': 'none', 'enable_thinking': False}


@pytest.mark.parametrize('profile', [QWEN_27_NVFP4, QWEN_27_FP8])
def test_qwen27_actual_command_preserves_mtp_and_selects_its_quantization(tmp_path, monkeypatch, profile):
    recipe = import_fork(tmp_path, monkeypatch)
    candidate.configure_candidate(recipe, profile)
    tuning = recipe.resolve_vllm_tuning()
    assert not recipe.validate_vllm_tuning_document(tuning)
    command = recipe.server_command(tmp_path / 'model', tuning=tuning)
    assert command[command.index('--quantization') + 1] == profile.cli_quantization
    assert command[command.index('--served-model-name') + 1] == profile.model
    speculative = json.loads(command[command.index('--speculative-config') + 1])
    assert speculative == {'method': 'mtp', 'num_speculative_tokens': 3}


@pytest.mark.parametrize('profile', [QWEN_27_NVFP4, QWEN_27_FP8])
def test_qwen27_manifest_and_study_reject_model_drift(tmp_path, profile):
    manifest = json.loads(profile.manifest.read_text())
    assert (manifest['repo'], manifest['revision']) == (profile.model, profile.revision)
    weights = {n for n in manifest['files'] if n.endswith('.safetensors')}
    assert weights and all(len(manifest['files'][n]['sha256']) == 64 for n in weights)
    config = {'architectures': [profile.architecture],
              'quantization_config': {'quant_method': profile.quant_method, 'quant_algo': 'MIXED_PRECISION'},
              'text_config': {'max_position_embeddings': 262144}}
    index = {'weight_map': {f'mtp.layer{i}': 'a.safetensors' for i in range(profile.mtp_tensors)}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))
    (tmp_path / 'a.safetensors').write_bytes(b'fixed-test-weights')
    manifest['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in tmp_path.iterdir()}
    assert candidate.verify_model(tmp_path, manifest)['all_file_hashes_verified']
    index['weight_map'].pop('mtp.layer0')
    f = tmp_path / 'model.safetensors.index.json'
    f.write_text(json.dumps(index))
    manifest['files'][f.name] = {'bytes': f.stat().st_size, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
    with pytest.raises(RuntimeError, match='MTP inventory'):
        candidate.verify_model(tmp_path, manifest)

    suffix = 'nvfp4' if profile == QWEN_27_NVFP4 else 'fp8'
    study = json.loads((ROOT / f'configs/arena_30m_qwen27_{suffix}.json').read_text())
    assert from_study(study) == profile
    study['server']['mtp_tokens'] = 0
    with pytest.raises(ValueError, match='MTP/reasoning'):
        from_study(study)


def test_checkpoint_verification_rejects_modified_or_untracked_shards(tmp_path):
    config = {'architectures': ['Qwen3_5MoeForConditionalGeneration'],
              'quantization_config': {'quant_method': 'compressed-tensors'},
              'text_config': {'max_position_embeddings': 262144}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'layer': 'a.safetensors'}}))
    (tmp_path / 'a.safetensors').write_bytes(b'fixed')
    manifest = {'repo': NEX_MODEL, 'revision': NEX_REVISION,
                'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in tmp_path.iterdir()}}
    assert candidate.verify_model(tmp_path, manifest)['all_file_hashes_verified']
    (tmp_path / 'a.safetensors').write_bytes(b'other')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        candidate.verify_model(tmp_path, manifest)
    (tmp_path / 'a.safetensors').write_bytes(b'fixed')
    manifest['files'].pop('a.safetensors')
    with pytest.raises(RuntimeError, match='unverified shards'):
        candidate.verify_model(tmp_path, manifest)


@pytest.mark.parametrize('wrong', [False, True])
async def test_readiness_checks_candidate_identity_without_ple_log_requirements(monkeypatch, wrong):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
        200, json={'data': [{'id': 'wrong' if wrong else NEX_MODEL, 'max_model_len': 262144}]})))
    monkeypatch.setattr(candidate.httpx, 'AsyncClient', lambda **kwargs: client)
    recipe = NS(alive=lambda *args: True)
    if wrong:
        with pytest.raises(RuntimeError, match='identity'):
            await candidate.ready(recipe, {'pid': 2, 'start_ticks': 3}, timeout=1, base_url='http://test/v1')
    else:
        assert await candidate.ready(recipe, {'pid': 2, 'start_ticks': 3}, timeout=1, base_url='http://test/v1')


def test_candidate_bundle_contains_own_weights_reference_and_both_trials(tmp_path):
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build('nex-test', output=tmp_path / 'bundle', paired_30m=True, nex=True, include_pi=True)
    builder.kaggle_files(output, study, smoke=False)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['model_sources'] == []
    assert 'richardcsaky/arc-nex-n25-mini-nvfp4' in metadata['dataset_sources']
    control = json.loads((ROOT / 'configs/arena_30m.json').read_text())
    assert study['budget_clock'] == 'inference'
    assert 'queue, tokenization and tools excluded' in study['level_budget_rule']
    for key in ['games', 'wall_seconds', 'windows', 'active_limit', 'completion_cap', 'trials', 'temperature']:
        assert study[key] == control[key]
    assert study['server']['mtp_tokens'] == 0 and study['reasoning_effort'] == 'high'
    assert (output / 'src/arena/models/nex-n25-mini-nvfp4.json').is_file()
    with pytest.raises(ValueError, match='paired'):
        builder.build('bad', output=tmp_path / 'bad', nex=True)


@pytest.mark.parametrize('profile', [NEX, replace(NEX, effort='none'), QWEN_27_NVFP4, QWEN_27_FP8, UI_MATE_9B, UI_MATE_27B])
@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('graph_size', [None, 128])
def test_candidate_controller_preserves_failure_evidence_and_owned_cleanup(tmp_path, monkeypatch, fail, profile, graph_size):
    study = json.loads((ROOT / 'configs/arena_30m_nex.json').read_text())
    study.update(candidate_profile=profile.name, served_model=profile.model,
                 model_revision=profile.revision, reasoning_effort=profile.effort)
    study['server']['mtp_tokens'] = profile.mtp_tokens
    native = profile in (UI_MATE_9B, UI_MATE_27B)
    if native:
        study['trials'] = [{'name': 'native', 'kind': 'ui-mate-native'}]
    if graph_size is not None:
        study['server']['max_cudagraph_capture_size'] = graph_size
        study['server']['cudagraph_metrics'] = True
    calls = []
    recipe = NS(
        BASE_URL='http://localhost/v1',
        resolve_runtime_dir=lambda: tmp_path,
        verify_and_extract_runtime=lambda *a, **k: {'verified': True},
        patch_ple_layer=lambda: {'shared_runtime': True},
        resolve_vllm_tuning=lambda: {'mtp_speculative_tokens': profile.mtp_tokens},
        runtime_environment=lambda **k: ({}, {'verified': True}),
        start_server=lambda *a, **k: {'pid': 999, 'start_ticks': 4},
        update_ready_identity=lambda identity: {**identity, 'phase': 'ready'},
        stop_owned_server=lambda: calls.append('stop') or {'stopped': True},
    )
    monkeypatch.setattr(candidate, 'verify_hardware', lambda: 'test-only GPU')
    monkeypatch.setattr(candidate, 'needed', lambda: False)
    monkeypatch.setattr(candidate, 'worker_probe', lambda *a: NS(returncode=0, stdout='{"type": "ready"}', stderr=''))
    monkeypatch.setattr(candidate, 'dataset_mount', lambda *a: tmp_path)
    monkeypatch.setattr(candidate, 'verify_model', lambda *a, **k: {'mock': True})
    monkeypatch.setattr(candidate, 'stage_fork_bundle', lambda *a: tmp_path)
    monkeypatch.setattr(candidate, 'load_recipe', lambda *a, **k: recipe)
    def configure(*args, graph_capture_size=None, reclaim_skipped_blocks=False, graph_metrics=False, **extra):
        assert graph_capture_size == graph_size
        assert reclaim_skipped_blocks is False
        assert graph_metrics == study['server'].get('cudagraph_metrics', False)
        assert extra == {}  # no prefix cache, ready guard, token details or stall dump without a study/trial asking
        calls.append('cache')

    monkeypatch.setattr(candidate, 'configure', configure)
    monkeypatch.setattr(candidate, 'configure_candidate', lambda *a: calls.append(profile.name))
    for key in ('ARENA_KV_CACHE_DTYPE', 'ARENA_READY_GUARD', 'ARENA_STALL_DUMP', 'ARENA_PREFIX_CACHE',
                'ARENA_KV_CACHE_MEMORY_BYTES', 'ARENA_PROMPT_TOKENS_DETAILS'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(candidate.BlockBudget, 'load', lambda *a: NS(layout={'lookahead': 0}))

    async def ready(*args, **kwargs):
        if fail:
            raise TimeoutError('loading failed')
        assert kwargs['profile'] == profile
        return [{'id': profile.model}]

    async def preflight(config):
        assert config.model == profile.model and config.model_revision == profile.revision
        assert config.reasoning_effort == profile.effort
        return {'profile': {}}

    monkeypatch.setattr(candidate, 'ready', ready)
    monkeypatch.setattr(candidate, 'preflight', preflight)
    from ui_mate_actor import preflight as native_preflight

    async def handoff(config, output):
        assert native and output == tmp_path / 'native-action-preflight'
        assert config.model == profile.model
        calls.append('handoff')
        return {'status': 'passed', 'public_game_actions': 0}

    monkeypatch.setattr(native_preflight, 'check', handoff)
    if fail:
        with pytest.raises(TimeoutError):
            candidate.start(study, tmp_path, '/inputs')
    else:
        candidate.start(study, tmp_path, '/inputs')
    record = json.loads((tmp_path / 'candidate-preflight.json').read_text())
    assert calls == ['cache', profile.name, *(['stop'] if fail else ['handoff'] if native else [])]
    if fail:
        assert record['failed_stage'] == 'model_startup' and record['cleanup']['stopped']
    else:
        assert record['status'] == 'ready' and record['native_preflight']['profile']['mtp_tokens'] == profile.mtp_tokens


@pytest.mark.parametrize('profile', [QWEN_27_NVFP4, QWEN_27_FP8])
def test_qwen27_bundle_preserves_paired_control_and_own_weights(tmp_path, profile):
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build('candidate-test', output=tmp_path / 'bundle', paired_30m=True,
                                  candidate_profile=profile.name, include_pi=True)
    builder.kaggle_files(output, study, smoke=False)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    manifest = json.loads(profile.manifest.read_text())
    if profile == QWEN_27_FP8:
        assert metadata['model_sources'] == [manifest['model_source']]
        assert manifest['dataset'] not in metadata['dataset_sources']
    else:
        assert metadata['model_sources'] == []
        assert manifest['dataset'] in metadata['dataset_sources']
    control = json.loads((ROOT / 'configs/arena_30m.json').read_text())
    assert study['budget_clock'] == 'inference'
    assert 'queue, tokenization and tools excluded' in study['level_budget_rule']
    for key in ['games', 'wall_seconds', 'windows', 'active_limit', 'completion_cap', 'trials',
                'temperature', 'generation_slots', 'reasoning_effort', 'server']:
        assert study[key] == control[key]
    assert study['served_model'] == profile.model and study['model_revision'] == profile.revision


def test_verified_model_mount_is_explicit_and_versioned(tmp_path):
    from arena.candidate_serving import checkpoint_mount

    manifest = json.loads(QWEN_27_FP8.manifest.read_text())
    study = {'model_sources': [manifest['model_source']]}
    with pytest.raises(FileNotFoundError, match='Missing pinned'):
        checkpoint_mount(study, manifest, tmp_path)
    owner, model, framework, variant, version = manifest['model_source'].split('/')
    path = tmp_path / 'models' / owner / model / framework.lower() / variant / version
    path.mkdir(parents=True)
    assert checkpoint_mount(study, manifest, tmp_path) == path
    with pytest.raises(ValueError, match='reviewed inventory'):
        checkpoint_mount({'model_sources': [manifest['model_source'][:-1] + '2']}, manifest, tmp_path)
    # No opportunistic switch to a public mirror when a dataset was requested.
    with pytest.raises(FileNotFoundError, match='mounted dataset'):
        checkpoint_mount({}, manifest, tmp_path)


def test_nex_none_builder_and_profile_preserve_comparison(tmp_path):
    spec = importlib.util.spec_from_file_location('builder_none', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    _, study = builder.build('nex-none', output=tmp_path / 'bundle', paired_30m=True,
                             nex=True, nex_no_thinking=True, capacity_phase2=True)
    original = json.loads((ROOT / 'configs/arena_30m_nex.json').read_text())
    assert from_study(study) == replace(NEX, effort='none')
    assert from_study(original) == NEX
    for key in ['games', 'wall_seconds', 'windows', 'completion_cap', 'temperature', 'server']:
        assert study[key] == original[key]
    assert study['trials'] == [{'name': 'duck-transfer', 'kind': 'duck-transfer'}]
    assert study['phase2_admission'] == 'cache_capacity'
    assert study['model_template_notes']['gameplay'] == 'none'
    with pytest.raises(ValueError, match='Nex candidate'):
        builder.build('bad', nex_no_thinking=True)
    study['server']['mtp_tokens'] = 3
    with pytest.raises(ValueError, match='MTP/reasoning'):
        from_study(study)
