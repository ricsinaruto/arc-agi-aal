"""Native UI-Mate BF16 model settings for Duck/Pi; not the screenshot actor."""
import hashlib
import importlib.util
import json
from pathlib import Path

import jinja2
import pytest
from test_qwen_serving import import_fork

from arc_infra.model_templates import template_kwargs
from arena.candidate_profiles import UI_MATE_9B, UI_MATE_27B, from_study
from arena.candidate_serving import configure_candidate, verify_model
from arena.trial import duck_config, qwen_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('profile', [UI_MATE_9B, UI_MATE_27B])
def test_native_bf16_command_and_both_agent_configs(tmp_path, monkeypatch, profile):
    recipe = import_fork(tmp_path, monkeypatch)
    configure_candidate(recipe, profile)
    tuning = recipe.resolve_vllm_tuning()
    assert not recipe.validate_vllm_tuning_document(tuning)
    argv = recipe.server_command(tmp_path / 'model', tuning=tuning)
    assert '--quantization' not in argv and '--speculative-config' not in argv
    assert argv[argv.index('--dtype') + 1] == 'bfloat16'
    assert argv[argv.index('--served-model-name') + 1] == profile.model
    study = json.loads((ROOT / f'configs/arena_30m_{profile.name.replace("-", "_")}.json').read_text())
    assert from_study(study) == profile
    params = dict(window=131072, cap=16384, wall=1800, base_url='http://localhost/v1')
    duck = duck_config(study, ROOT, **params)
    pi = qwen_config(study, 'pi', **params)
    assert duck.reasoning_effort == pi.reasoning_effort == 'native'
    assert duck.model == pi.model == profile.model
    assert duck.speculative_tokens == 0


@pytest.mark.parametrize('profile', [UI_MATE_9B, UI_MATE_27B])
def test_actual_native_template_switches_and_history_difference(profile):
    manifest = json.loads(profile.manifest.read_text())
    fixture = ROOT / f'tests/fixtures/{profile.name}-chat-template.jinja'
    raw = fixture.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['files']['chat_template.jinja']['sha256']
    template = jinja2.Environment().from_string(raw.decode())
    def render(messages, kwargs):
        def fail(message):
            raise ValueError(message)
        return template.render(messages=messages, add_generation_prompt=True, raise_exception=fail, **kwargs)
    messages = [{'role': 'user', 'content': 'Board'},
                {'role': 'assistant', 'content': 'Action', 'reasoning_content': 'OLD_REASONING'},
                {'role': 'user', 'content': 'New board'}]
    gameplay = template_kwargs(profile.model, {'enable_thinking': True,
                               'reasoning_effort': 'native', 'preserve_thinking': True})
    assert 'reasoning_effort' not in gameplay
    text = render(messages, gameplay)
    assert text.endswith('<think>\n')
    assert ('OLD_REASONING' in text) == (profile == UI_MATE_27B)
    summary = template_kwargs(profile.model, {'enable_thinking': False, 'reasoning_effort': 'native'})
    assert render([{'role': 'user', 'content': 'Summarize'}], summary).endswith('<think>\n\n</think>\n\n')


@pytest.mark.parametrize('profile', [UI_MATE_9B, UI_MATE_27B])
def test_manifest_native_dtype_and_absent_mtp_are_enforced(tmp_path, profile):
    manifest = json.loads(profile.manifest.read_text())
    assert manifest['repo'] == profile.model and manifest['revision'] == profile.revision
    config = {'architectures': [profile.architecture], 'dtype': 'bfloat16',
              'text_config': {'max_position_embeddings': 262144}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'layer': 'weights'}}))
    (tmp_path / 'weights').write_bytes(b'mock-only')
    def inventory():
        return {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in tmp_path.iterdir()}
    manifest['files'] = inventory()
    assert verify_model(tmp_path, manifest)['all_file_hashes_verified']
    config['dtype'] = 'float16'
    (tmp_path / 'config.json').write_text(json.dumps(config))
    manifest['files'] = inventory()
    with pytest.raises(RuntimeError, match='native BF16'):
        verify_model(tmp_path, manifest)


@pytest.mark.parametrize('profile', [UI_MATE_9B, UI_MATE_27B])
def test_candidate_build_keeps_paired_trials_and_pinned_manifest(tmp_path, profile):
    spec = importlib.util.spec_from_file_location('ui_builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build('ui-test', output=tmp_path / 'bundle', paired_30m=True,
                                  candidate_profile=profile.name, include_pi=True)
    assert study['wall_seconds'] == 1800 and len(study['games']) == 25
    assert [t['kind'] for t in study['trials']] == ['duck-transfer', 'pi']
    assert study['server']['mtp_tokens'] == 0 and study['windows'] == [131072]
    builder.kaggle_files(output, study, smoke=False)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert study['model_datasets'][0] in metadata['dataset_sources']
    assert metadata['model_sources'] == [] and metadata['machine_shape'] == 'NvidiaRtxPro6000'
