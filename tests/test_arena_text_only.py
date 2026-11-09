import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import jinja2
import pytest

from arc_infra.model_templates import MINICPM_MODEL, request_messages, template_kwargs
from arena.cache_probe import payload
from arena.candidate_profiles import (
    MINICPM_2B,
    NEMOTRON_LIGHTNING,
    NEMOTRON_SUPER,
    ORNITH_35B,
    QWEN35_122B,
    QWEN_36_NVFP4,
)
from arena.capacity import phase2_capacity

ROOT = Path(__file__).resolve().parents[1]


def test_native_templates_count_visible_text_tools_and_thinking():
    for profile, stem in [(MINICPM_2B, 'minicpm5'), (QWEN_36_NVFP4, 'qwen36'), (NEMOTRON_SUPER, 'nemotron3-super'),
                          (NEMOTRON_LIGHTNING, 'nemotron35-lightning'), (QWEN35_122B, 'qwen35-122b'),
                          (ORNITH_35B, 'ornith15-35b')]:
        fixture = ROOT / f'tests/fixtures/{stem}-chat-template.jinja'
        inventory = json.loads(profile.manifest.read_text())
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == inventory['files']['chat_template.jinja']['sha256']
        env = jinja2.Environment()
        env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, **kwargs)
        template = env.from_string(fixture.read_text())
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'EXACT_BOARD'}]},
                    {'role': 'assistant', 'content': '', 'reasoning_content': 'EVIDENCE_THOUGHT',
                     'tool_calls': [{'function': {'name': 'python', 'arguments': {'code': 'print(1 < 2)'}}}]},
                    {'role': 'tool', 'content': 'True'}]
        messages = request_messages(profile.model, messages)
        rendered = template.render(messages=messages, tools=[{'type': 'function', 'function': {'name': 'python'}}],
                                   bos_token='<s>', add_generation_prompt=True,
                                   **template_kwargs(profile.model, {'enable_thinking': True, 'reasoning_effort': 'native'}))
        assert 'EXACT_BOARD' in rendered and 'EVIDENCE_THOUGHT' in rendered and 'print(1 < 2)' in rendered
        assert rendered.endswith('<think>\n')
        quiet = template.render(messages=messages, bos_token='<s>', add_generation_prompt=True,
                                **template_kwargs(profile.model, {'enable_thinking': False}))
        if profile in (NEMOTRON_SUPER, NEMOTRON_LIGHTNING):
            # Nemotron opens the assistant turn with an empty think block when thinking is off and
            # renders tool calls in the Qwen3-Coder XML form the pinned qwen3_coder parser reads.
            assert quiet.endswith('<|im_start|>assistant\n<think></think>')
            assert '<tool_call>\n<function=python>\n<parameter=code>\nprint(1 < 2)\n</parameter>\n</function>\n</tool_call>' in rendered
            assert '<tool_response>\nTrue\n</tool_response>' in rendered
        else:
            assert quiet.endswith('<think>\n\n</think>\n\n')


def test_text_only_normalizes_without_mutating_and_rejects_images():
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'board'}, {'type': 'text', 'text': 'receipt'}]}]
    original = copy.deepcopy(messages)
    assert request_messages(MINICPM_MODEL, messages)[0]['content'] == 'board\nreceipt'
    assert original == messages
    messages[0]['content'].append({'type': 'image_url', 'image_url': {'url': 'secret'}})
    with pytest.raises(ValueError, match='non-text evidence'):
        request_messages(MINICPM_MODEL, messages)
    assert isinstance(payload(0, 4, model=MINICPM_MODEL)['messages'][0]['content'], str)


def test_native_128k_capacity_does_not_request_more_than_native_context():
    seen = []
    def cost(tokens):
        seen.append(tokens)
        assert tokens <= 131072
        return 8192
    budget = NS(capacity=90000, cost=cost, layout={'lookahead': 0})
    study = {'phase2_admission': 'cache_capacity', 'generation_slots': 25, 'kv_margin': .9,
             'server': {'max_model_len': 131072, 'max_num_seqs': 25}}
    result = phase2_capacity(study, budget, 131072, 25)
    assert result['request_tokens'] == 131072 and result['active_limit'] == 10


async def test_actual_duck_text_observation_reaches_action_bridge(tmp_path):
    from conftest import tool_reply
    from test_arena_duck import arena_agent

    from arena.duck import ArenaTransferDuckAgent

    class Transport:
        async def post(self, url, **kwargs):
            body = kwargs['json']
            assert body['chat_template_kwargs'] == {'enable_thinking': True}
            assert all(isinstance(m.get('content', ''), (str, type(None))) for m in body['messages'])
            text = '\n'.join(m.get('content') or '' for m in body['messages'])
            assert 'Grid text of the board at step 0' in text
            assert 'No images are supplied' in text
            assert 'image_url' not in json.dumps(body)
            return tool_reply("action([{'action':'UP'}])", call_id='act')

    agent, env, game, gate = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    agent.config.model = MINICPM_MODEL
    agent.config.reasoning_effort = 'native'
    result = await agent.run()
    assert result['error'] is None and result['actions'] == 1


@pytest.mark.parametrize('profile', [MINICPM_2B, QWEN_36_NVFP4, NEMOTRON_SUPER, NEMOTRON_LIGHTNING, QWEN35_122B,
                                     ORNITH_35B])
def test_candidate_command_and_inventory(tmp_path, monkeypatch, profile):
    from test_serving_core import import_fork

    from arena.candidate_serving import configure_candidate, verify_model

    recipe = import_fork(tmp_path, monkeypatch)
    configure_candidate(recipe, profile)
    command = recipe.server_command(tmp_path, tuning=recipe.resolve_vllm_tuning())
    assert command[command.index('--max-model-len') + 1] == str(profile.native_context)
    assert command[command.index('--tool-call-parser') + 1] == profile.tool_parser
    assert command[command.index('--reasoning-parser') + 1] == profile.reasoning_parser
    if not profile.mtp_tokens:
        assert '--speculative-config' not in command
    if profile.cli_quantization is None:
        assert '--quantization' not in command
    else:
        assert command[command.index('--quantization') + 1] == profile.cli_quantization
    if profile == QWEN35_122B:
        # Plain NVFP4 checkpoint: the pinned Flash quantization flag stands; no speculation, BF16 KV pinned.
        assert command[command.index('--quantization') + 1] == 'modelopt_fp4'
        assert '--speculative-config' not in command
        assert command[command.index('--kv-cache-dtype') + 1] == 'bfloat16' and command.count('--kv-cache-dtype') == 1
        assert command[command.index('--reasoning-parser') + 1] == 'qwen3'
    if profile == ORNITH_35B:
        # The Qwen3.6 serving path (mixed NVFP4/FP8, MTP depth 3) with the checkpoint's FP8 KV scheme overridden.
        assert command[command.index('--quantization') + 1] == 'modelopt_mixed'
        assert json.loads(command[command.index('--speculative-config') + 1])['num_speculative_tokens'] == 3
        assert command[command.index('--kv-cache-dtype') + 1] == 'bfloat16' and command.count('--kv-cache-dtype') == 1
        assert command[command.index('--tool-call-parser') + 1] == 'qwen3_coder'
        assert command[command.index('--reasoning-parser') + 1] == 'qwen3'
    if profile in (NEMOTRON_SUPER, NEMOTRON_LIGHTNING):
        assert command[command.index('--attention-backend') + 1] == 'TRITON_ATTN'
        assert command[command.index('--reasoning-parser') + 1] == 'nemotron_v3'
        assert command[command.index('--max-model-len') + 1] == '262144'
        assert '--speculative-config' not in command  # both Nemotron profiles run without the MTP head
        assert command[command.index('--kv-cache-dtype') + 1] == 'bfloat16'
        assert command[command.index('--moe-backend') + 1] == 'marlin' and command.count('--moe-backend') == 1
    else:
        assert '--attention-backend' not in command
    config = {'architectures': [profile.architecture], 'dtype': 'bfloat16'}
    if profile == MINICPM_2B:
        config['max_position_embeddings'] = profile.native_context
    elif profile in (NEMOTRON_SUPER, NEMOTRON_LIGHTNING):
        config.update(max_position_embeddings=profile.declared_context or profile.native_context,
                      quantization_config={'quant_method': 'modelopt', 'quant_algo': 'MIXED_PRECISION'})
    else:
        config.update(text_config={'max_position_embeddings': profile.native_context},
                      quantization_config={'quant_method': 'modelopt',
                                           'quant_algo': 'NVFP4' if profile == QWEN35_122B else 'MIXED_PRECISION'})
    (tmp_path / 'config.json').write_text(json.dumps(config))
    index = {f'mtp.layer{i}': 'model.safetensors' for i in range(profile.mtp_tensors)}
    index['model.layer0'] = 'model.safetensors'
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
    (tmp_path / 'model.safetensors').write_bytes(b'fake-test-weights')
    files = {name: {'bytes': (tmp_path/name).stat().st_size,
                    'sha256': hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()}
             for name in ('config.json', 'model.safetensors.index.json', 'model.safetensors')}
    manifest = {'repo': profile.model, 'revision': profile.revision, 'files': files}
    assert verify_model(tmp_path, manifest)['all_file_hashes_verified']


@pytest.mark.parametrize('failure', [None, 'unparsed', 'count'])
async def test_tool_preflight_keeps_receipt_and_detects_parser_failure(tmp_path, monkeypatch, failure):
    import httpx

    from arena import tool_preflight
    from arena.serving_core.config import Config

    bodies = []
    def endpoint(request):
        body = json.loads(request.content)
        bodies.append(body)
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 123, 'max_model_len': 131072})
        assert request.url.path == '/v1/chat/completions'
        message = {'content': ''}
        if failure != 'unparsed':
            message['tool_calls'] = [{'function': {'name': 'python', 'arguments': json.dumps({'code': 'print(7)\n'})}}]
        return httpx.Response(200, json={'usage': {'prompt_tokens': 124 if failure == 'count' else 123},
                                        'choices': [{'message': message}]})
    client = httpx.AsyncClient(base_url='http://local/v1/', transport=httpx.MockTransport(endpoint))
    monkeypatch.setattr(tool_preflight.httpx, 'AsyncClient', lambda **kwargs: client)
    cfg = Config(model=MINICPM_MODEL, context_window=131072, reasoning_effort='native')
    output = tmp_path / 'probe.json'
    if failure:
        with pytest.raises(RuntimeError):
            await tool_preflight.check(cfg, output)
    else:
        assert (await tool_preflight.check(cfg, output))['status'] == 'passed'
    receipt = json.loads(output.read_text())
    assert receipt['actions_executed'] == 0 and receipt['status'] == ('error' if failure else 'passed')
    assert bodies[0]['messages'] == bodies[1]['messages']


def test_ornith_manifest_dataset_and_native_multimodal_profile(tmp_path):
    from arc_infra.model_templates import allowed_efforts, text_only
    from arena.candidate_profiles import from_study
    from arena.candidate_serving import verify_model

    inventory = json.loads(ORNITH_35B.manifest.read_text())
    assert (inventory['repo'], inventory['revision']) == (ORNITH_35B.model, ORNITH_35B.revision)
    shards = [n for n in inventory['files'] if n.endswith('.safetensors')]
    assert len(shards) == 3 and all(len(inventory['files'][n]['sha256']) == 64 for n in inventory['files'])
    assert sum(inventory['files'][n]['bytes'] for n in shards) == 23420419700
    for name in ('config.json', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                 'merges.txt', 'model.safetensors.index.json', 'hf_quant_config.json', 'preprocessor_config.json',
                 'processor_config.json', 'video_preprocessor_config.json', 'generation_config.json'):
        assert name in inventory['files']
    # Documentation, images and upload markers are not execution inputs.
    assert not any(n == 'README.md' or n.startswith(('.', 'assets/')) for n in inventory['files'])
    assert inventory.get('model_source') is None and inventory['dataset'] == 'richardcsaky/arc-ornith15-35b-a3b-nvfp4'
    # Tokenizer and processor files byte-identical to the Qwen3.6 candidate's reviewed copies.
    qwen36 = json.loads(QWEN_36_NVFP4.manifest.read_text())['files']
    for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'generation_config.json',
                 'preprocessor_config.json', 'video_preprocessor_config.json'):
        assert inventory['files'][name] == qwen36[name]
    study = json.loads((ROOT / 'configs/arena_30m_ornith15.json').read_text())
    assert from_study(study) == ORNITH_35B
    assert study['model_datasets'] == [inventory['dataset']] and study['model_sources'] == []
    assert study['server']['max_model_len'] == 262144 and study['server']['mtp_tokens'] == 3
    assert study['observation_mode'] == 'native-multimodal' and study['phase2_admission'] == 'cache_capacity'
    assert not text_only(ORNITH_35B.model) and allowed_efforts(ORNITH_35B.model) == ('native',)
    assert template_kwargs(ORNITH_35B.model, {'enable_thinking': True, 'reasoning_effort': 'native',
                                              'preserve_thinking': False}) == {'enable_thinking': True}
    assert payload(0, 4, model=ORNITH_35B.model)['chat_template_kwargs'] == {'enable_thinking': False}
    messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}]
    assert request_messages(ORNITH_35B.model, messages) is messages
    # verify_model accepts the MIXED_PRECISION checkpoint with its 785-tensor MTP head and rejects a plain NVFP4
    # config (the modelopt_mixed flag would then be wrong) or a different MTP inventory.
    (tmp_path / 'model.safetensors').write_bytes(b'w')
    for algo, mtp_count, error in (('MIXED_PRECISION', 785, None), ('MIXED_PRECISION', 19, 'MTP inventory'),
                                   ('NVFP4', 785, 'mixed NVFP4/FP8')):
        index = {f'mtp.layers.0.mlp.experts.{i}.up_proj.weight': 'model.safetensors' for i in range(mtp_count)}
        index['model.language_model.layers.0.self_attn.q_proj.weight'] = 'model.safetensors'
        (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
        (tmp_path / 'config.json').write_text(json.dumps({
            'architectures': ['Qwen3_5MoeForConditionalGeneration'], 'dtype': 'bfloat16',
            'text_config': {'max_position_embeddings': 262144},
            'quantization_config': {'quant_method': 'modelopt', 'quant_algo': algo,
                                    'kv_cache_scheme': {'dynamic': False, 'num_bits': 8, 'type': 'float'}}}))
        files = {n: {'bytes': (tmp_path / n).stat().st_size, 'sha256': hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()}
                 for n in ('config.json', 'model.safetensors.index.json', 'model.safetensors')}
        manifest = {'repo': ORNITH_35B.model, 'revision': ORNITH_35B.revision, 'files': files}
        if error is None:
            assert verify_model(tmp_path, manifest)['all_file_hashes_verified']
        else:
            with pytest.raises(RuntimeError, match=error):
                verify_model(tmp_path, manifest)


def test_ornith_template_keeps_every_reasoning_block():
    # Unlike Qwen3.6 (reasoning before the last user turn dropped unless preserve_thinking), Ornith's
    # template renders all earlier reasoning: the arena's token counts must see the same prompt.
    env = jinja2.Environment()
    env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, **kwargs)
    messages = [{'role': 'user', 'content': 'FIRST'}, {'role': 'assistant', 'content': 'a', 'reasoning_content': 'OLD_THOUGHT'},
                {'role': 'user', 'content': 'SECOND'}]
    for stem, kept in (('ornith15-35b', True), ('qwen36', False)):
        template = env.from_string((ROOT / f'tests/fixtures/{stem}-chat-template.jinja').read_text())
        rendered = template.render(messages=messages, add_generation_prompt=True,
                                   **template_kwargs(ORNITH_35B.model if kept else QWEN_36_NVFP4.model, {}))
        assert ('OLD_THOUGHT' in rendered) is kept and rendered.endswith('<|im_start|>assistant\n<think>\n')


def test_ornith_one_hour_r23_build_matches_the_base_protocol(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build('arena-duck-60m-r23-ornith-test', output=tmp_path / 'ornith', paired_30m=True,
                                  candidate_profile=ORNITH_35B.name, capacity_phase2=True, opening_burst_requests=3,
                                  exclude_games=['ft09', 'lp85'], wall_seconds=3600)
    # The one-hour base (reclaim-fixed-2x on 23 games, 3,600 s) with only model, serving and phase-2 sizing changed.
    base = json.loads((ROOT / 'configs/arena_30m.json').read_text())
    base = builder.apply_variant(base, 'reclaim-fixed-2x')
    for key in ('level_budget_policy', 'backlog_policy', 'prioritize_parked_progress'):
        assert study[key] == base[key]
    assert base['budget_clock'] == 'busy' and study['budget_clock'] == 'tokens'  # new builds: the token clock
    assert study['cache_experiment'] == base['cache_experiment']
    assert study['cache_experiment']['reclaim_skipped_blocks'] is True
    assert study['wall_seconds'] == 3600 and len(study['games']) == 23
    assert sorted(g[:4] for g in study['games_excluded']) == ['ft09', 'lp85']
    assert study['opening_burst_requests'] == 3 and study['trials'] == [{'name': 'duck-transfer', 'kind': 'duck-transfer'}]
    for key in ('windows', 'kv_margin', 'opening_switch_level1_games', 'completion_cap', 'temperature',
                'generation_slots', 'context_policy', 'cache_admission'):
        assert study[key] == base[key]
    assert study['level_budget_rule'].startswith('2*L*T1') and 'completion tokens' in study['level_budget_rule']
    assert study['phase2_admission'] == 'cache_capacity' and study['active_limit'] == 'resolved_from_cache'
    assert study['reasoning_effort'] == 'native' and study['notebook_seconds'] == 7200 and study['kernel_timeout'] == 7500
    assert (output / f'src/arena/models/{ORNITH_35B.name}.json').is_file()
    builder.kaggle_files(output, study, smoke=False)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['model_sources'] == [] and 'richardcsaky/arc-ornith15-35b-a3b-nvfp4' in metadata['dataset_sources']
    # Qwen3.6's measured layout on identical shapes (2,669 usable blocks, 139 per full 128K request) admits 19.
    budget = NS(capacity=2669, cost=lambda tokens: 139, layout={'lookahead': 3})
    assert phase2_capacity(study, budget, 131072, len(study['games']))['active_limit'] == 19
