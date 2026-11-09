import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from arena import cache_failure_runtime
from arena.cache import BlockBudget
from arena.cache_export_runtime import describe
from arena.cache_patch import ORIGINAL_SHA256, apply
from arena.gate import KVGate
from arena.serving import configure


@pytest.mark.parametrize('mtp_tokens', [1, 3])
def test_arena_launcher_forwards_requested_mtp_to_native_start(tmp_path, monkeypatch, mtp_tokens):
    from arena import serving
    config = {'generation_slots': 25, 'kv_margin': .9,
              'server': {'mtp_tokens': mtp_tokens, 'max_num_batched_tokens': 4096},
              'cache_experiment': {'cpu_cache_gib': 0, 'kv_cache_dtype': 'auto'}}
    study = tmp_path/'study.json'
    study.write_text(json.dumps(config))
    calls = []
    monkeypatch.setattr(serving, 'start', lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(serving.BlockBudget, 'load', lambda *args: NS(layout={}))
    monkeypatch.setattr('sys.argv', ['arena.serving', '--working', str(tmp_path),
                                   '--input-root', str(tmp_path), '--study', str(study)])
    serving.main()
    assert calls[0]['mtp_tokens'] == mtp_tokens
    assert calls[0]['max_num_batched_tokens'] == 4096
    assert calls[0]['generation_slots'] == 25


class Attention:
    page_size_bytes = 100
    block_size = 100

    def max_memory_usage_bytes(self, cfg):
        return ((cfg.model_config.max_model_len + 99) // 100) * 100


class State(Attention):
    def max_memory_usage_bytes(self, cfg):
        return 500


def layout():
    config = NS(model_config=NS(max_model_len=4096), speculative_config=NS(num_speculative_tokens=3))
    cache = NS(num_blocks=100, kv_cache_groups=[NS(kv_cache_spec=x, layer_names=['layer'])
                                              for x in (Attention(), State())])
    result = describe(config, cache)
    assert config.model_config.max_model_len == 4096
    return result


def test_fixed_state_cost_and_rounding_are_not_proportional_to_tokens():
    budget = BlockBudget(layout())
    assert budget.cost(10) == 16  # ceil to next exported 1024 point, 11 attention + 5 state
    assert budget.cost(1024) == 26  # speculative lookahead crosses table boundary
    assert budget.cost(4000) == 46
    with pytest.raises(ValueError, match='lookahead'):
        budget.cost(4096)
    bad = layout()
    bad['costs'][1]['blocks'] = 0
    with pytest.raises(ValueError):
        BlockBudget(bad)


async def test_gate_accounts_blocks_and_never_clamps_an_oversized_request():
    budget = BlockBudget(layout(), margin=0.3)
    gate = KVGate(budget.capacity, 25, budget=budget)
    first = await gate.acquire('first', 10)
    waiter = asyncio.create_task(gate.acquire('second', 10))
    await asyncio.sleep(0)
    assert gate.inflight == 1 and gate.reserved == 16 and not waiter.done()
    gate.release(first)
    second = await waiter
    gate.release(second)
    assert gate.inflight == gate.reserved == 0
    with pytest.raises(ValueError, match='Single request'):
        await gate.acquire('too-big', 4000)
    assert gate.snapshot()['units'] == 'cache_blocks'


async def test_cancelled_block_waiter_does_not_leak():
    budget = BlockBudget(layout(), margin=0.3)
    gate = KVGate(budget.capacity, 25, budget=budget)
    first = await gate.acquire('first', 10)
    waiter = asyncio.create_task(gate.acquire('cancel', 10))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    gate.release(first)
    assert not gate.queue and gate.reserved == 0


def test_layout_missing_or_disagreeing_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match='Missing'):
        BlockBudget.load(tmp_path)
    row = layout()
    p = tmp_path / 'server.log'
    p.write_text('worker ARC_CACHE_LAYOUT ' + json.dumps(row) + '\n')
    assert BlockBudget.load(tmp_path).capacity == 90
    row['num_blocks'] = 80
    with p.open('a') as f:
        f.write('ARC_CACHE_LAYOUT ' + json.dumps(row))
    with pytest.raises(RuntimeError, match='disagree'):
        BlockBudget.load(tmp_path)


def test_patch_rejects_changed_source(tmp_path):
    target = tmp_path / 'vllm/v1/core/kv_cache_utils.py'
    target.parent.mkdir(parents=True)
    target.write_text('not pinned')
    with pytest.raises(RuntimeError, match='unrecognized'):
        apply(tmp_path, tmp_path)
    assert target.read_text() == 'not pinned'
    assert len(ORIGINAL_SHA256) == 64


def test_offload_command_preserves_native_flags(monkeypatch, tmp_path):
    recipe = NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ['serve', '--max-num-seqs', '25'],
                RUNTIME_ROOT=tmp_path)
    from arena import serving
    patches = []
    monkeypatch.setattr(serving, 'apply', lambda *args: patches.append(args))
    configure(recipe, tmp_path, cpu_gib=16)
    recipe.verify_and_extract_runtime()
    assert patches
    command = recipe.server_command()
    assert command[:3] == ['serve', '--max-num-seqs', '25']
    config = json.loads(command[command.index('--kv-transfer-config') + 1])
    assert config['kv_connector_extra_config']['cpu_bytes_to_use'] == 16 * 1024**3
    assert config['kv_connector_extra_config']['offload_prompt_only'] is False
    assert '--no-disable-hybrid-kv-cache-manager' in command
    assert config['kv_connector'] == 'ArcOffloadingConnector' and config['kv_connector_module_path'] == 'vllm.arc_offload'
    assert (tmp_path / 'usr/local/lib/python3.12/dist-packages/vllm/arc_offload.py').is_file()


def test_graph_override_reaches_pinned_recipe_tuning_without_other_changes(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    for key in ('TAAF_KAGGLE_BUNDLE_DIR', 'TAAF_KAGGLE_WORKING_DIR', 'TAAF_KAGGLE_SETUP_ENV'):
        monkeypatch.setenv(key, str(tmp_path))
    path = Path(__file__).resolve().parents[1] / 'src/arc_infra/serving_setup.py'
    spec = importlib.util.spec_from_file_location('graph_test_recipe', path)
    serving_setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serving_setup)
    # Isolate both environment mutations; the real tuning resolver reads this key.
    monkeypatch.setenv(serving_setup.MAX_CUDAGRAPH_CAPTURE_SIZE_ENV, '32')
    monkeypatch.setenv('VLLM_SERVER_DEV_MODE', '0')
    recipe = NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ['serve'],
                RUNTIME_ROOT=tmp_path,
                MAX_CUDAGRAPH_CAPTURE_SIZE_ENV=serving_setup.MAX_CUDAGRAPH_CAPTURE_SIZE_ENV)
    control = serving_setup.resolve_vllm_tuning()
    record = configure(recipe, tmp_path, graph_capture_size=128)
    actual = serving_setup.resolve_vllm_tuning()
    assert actual == {**control, 'max_cudagraph_capture_size': 128}
    assert record['max_cudagraph_capture_size'] == 128
    assert recipe.server_command() == ['serve']  # The recipe consumes the env override.
    argv = serving_setup.server_command(tmp_path, tuning=actual)
    assert argv[argv.index('--max-cudagraph-capture-size') + 1] == '128'


@pytest.mark.parametrize('value', [True, 0, 257, '128', 128.5])
def test_invalid_graph_override_fails_before_runtime_mutation(tmp_path, value):
    with pytest.raises(ValueError, match='graph capture'):
        configure(NS(), tmp_path, graph_capture_size=value)


@pytest.mark.parametrize('enabled', [False, True])
def test_optional_graph_metrics_preserve_other_server_arguments(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv('VLLM_SERVER_DEV_MODE', '0')
    recipe = NS(verify_and_extract_runtime=lambda: {},
                server_command=lambda: ['serve', '--max-num-seqs', '25'])
    receipt = configure(recipe, tmp_path, graph_metrics=enabled)
    assert recipe.server_command() == ['serve', '--max-num-seqs', '25',
                                       *(['--cudagraph-metrics'] if enabled else [])]
    assert receipt.get('cudagraph_metrics', False) is enabled


def test_graph_metrics_do_not_silently_override_disabled_stats(tmp_path, monkeypatch):
    monkeypatch.setenv('VLLM_SERVER_DEV_MODE', '0')
    recipe = NS(verify_and_extract_runtime=lambda: {},
                server_command=lambda: ['serve', '--disable-log-stats'])
    configure(recipe, tmp_path, graph_metrics=True)
    with pytest.raises(ValueError, match='enabled log stats'):
        recipe.server_command()
    with pytest.raises(ValueError, match='boolean'):
        configure(NS(), tmp_path, graph_metrics='true')


def test_failure_snapshot_counts_shared_blocks_without_mutation():
    shared = NS(block_id=1, is_null=False, ref_cnt=2)
    null = NS(block_id=0, is_null=True, ref_cnt=100)
    private = NS(block_id=2, is_null=False, ref_cnt=1)
    mappings = {'a': [shared, null, private], 'b': [shared]}
    group = NS(req_to_blocks=mappings)
    manager = NS(coordinator=NS(single_type_managers=[group]),
                 block_pool=NS(get_num_free_blocks=lambda: 3))
    request = NS(request_id='a', status='RUNNING', num_tokens=9000, num_computed_tokens=8999)
    result = cache_failure_runtime.describe(manager, request, stage='incremental',
                                           required=5, available=1, reserved=2, watermark=0)
    assert result['groups'][0]['held_by_request'] == {'a': 2, 'b': 1}
    assert result['groups'][0]['unique_blocks'] == 2
    assert result['groups'][0]['shared_blocks'] == 1
    assert result['free_blocks'] == 3 and result['available_blocks'] == 1
    assert mappings == {'a': [shared, null, private], 'b': [shared]} and shared.ref_cnt == 2


def test_failure_diagnostics_never_propagate_errors_and_bound_snapshots(monkeypatch, capsys):
    monkeypatch.setattr(cache_failure_runtime, '_failures', 0)
    cache_failure_runtime.emit_failure(None, None)
    assert 'ARC_CACHE_DIAGNOSTIC_ERROR' in capsys.readouterr().out
    monkeypatch.setattr(cache_failure_runtime, 'describe', lambda *a, **kw: {})
    for _ in range(255):
        cache_failure_runtime.emit_failure(None, None)
    records = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 101  # failures 2..100, then 128 and 256
    assert records[-1]['snapshots_suppressed'] == 156


def test_arena_launch_keeps_backend_capacity_separate_from_regular_slots(tmp_path, monkeypatch):
    from arena import serving
    config = {'generation_slots': 26, 'kv_margin': .9,
              'server': {'max_num_seqs': 48, 'mtp_tokens': 0},
              'parallel_samples': {'max_num_seqs': 48},
              'cache_experiment': {'cpu_cache_gib': 0, 'kv_cache_dtype': 'auto'}}
    study = tmp_path / 'study.json'
    study.write_text(json.dumps(config))
    calls = []
    monkeypatch.setattr(serving, 'start', lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(serving.BlockBudget, 'load', lambda *args: NS(layout={}))
    monkeypatch.setattr('sys.argv', ['arena.serving', '--working', str(tmp_path),
                                   '--input-root', str(tmp_path), '--study', str(study)])
    serving.main()
    assert calls[0]['generation_slots'] == 48
    assert config['generation_slots'] == 26
    config['server']['max_num_seqs'] = 26
    study.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='limits disagree'):
        serving.main()
