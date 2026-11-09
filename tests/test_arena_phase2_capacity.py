import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_arena_cache import Attention

from arena.cache_export_runtime import describe
from arena.capacity import phase2_capacity

ROOT = Path(__file__).resolve().parents[1]


def study():
    return {'phase2_admission': 'cache_capacity', 'active_limit': 'resolved_from_cache',
            'generation_slots': 25, 'kv_margin': 0.9,
            'server': {'max_num_seqs': 25, 'gpu_memory_utilization': '0.96'}}


def budget(capacity, cost):
    return SimpleNamespace(capacity=capacity, cost=lambda tokens: cost,
                           layout={'planned_cache_bytes': 60 * 1024**3})


def test_smaller_model_uses_larger_measured_pool_without_shortening_context():
    plan = phase2_capacity(study(), budget(2959, 132), 131072, 25)
    assert plan['active_limit'] == 22
    assert plan['request_tokens'] == 131328  # Clock allowance, no double-counted output reserve.
    assert plan['cost_per_full_window_request'] * plan['active_limit'] <= plan['capacity']
    assert plan['cost_per_full_window_request'] * (plan['active_limit'] + 1) > plan['capacity']
    assert not plan['gpu_validated']
    assert phase2_capacity(study(), budget(199, 104), 131072, 25)['active_limit'] == 1


@pytest.mark.parametrize('pool,games,slots,sequences,expected', [
    (10000, 25, 25, 25, 25), (10000, 3, 25, 25, 3),
    (10000, 25, 8, 25, 8), (10000, 25, 25, 12, 12), (450, 25, 25, 25, 4),
])
def test_limits_respect_actual_hardware_and_roster(pool, games, slots, sequences, expected):
    cfg = study()
    cfg['generation_slots'] = slots
    cfg['server']['max_num_seqs'] = sequences
    assert phase2_capacity(cfg, budget(pool, 100), 131072, games)['active_limit'] == expected


def test_missing_capacity_fails_instead_of_silently_using_two_or_shrinking_context():
    with pytest.raises(ValueError, match='measured'):
        phase2_capacity(study(), None, 131072, 25)
    with pytest.raises(RuntimeError, match='cannot fit'):
        phase2_capacity(study(), budget(100, 101), 131072, 25)
    assert phase2_capacity({'active_limit': 2}, None, 131072, 25) == {'policy': 'fixed', 'active_limit': 2}


def test_cache_bytes_do_not_count_packed_buffer_aliases_twice():
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096), speculative_config=None)
    cache = SimpleNamespace(num_blocks=100, kv_cache_groups=[
        SimpleNamespace(kv_cache_spec=Attention(), layer_names=['a'])],
        kv_cache_tensors=[SimpleNamespace(size=1000, block_stride=100),
                          SimpleNamespace(size=1000, block_stride=100),
                          SimpleNamespace(size=200, block_stride=0)])
    assert describe(cfg, cache)['planned_cache_bytes'] == 1200


def test_bundle_selects_cache_sized_phase2_and_documents_it(tmp_path):
    spec = importlib.util.spec_from_file_location('builder', ROOT/'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, cfg = builder.build('cache-sized-test', output=tmp_path/'bundle', paired_30m=True,
                                candidate_profile='ornith15-35b-a3b-nvfp4', capacity_phase2=True)
    assert cfg['phase2_admission'] == 'cache_capacity'
    assert cfg['active_limit'] == 'resolved_from_cache'
    assert cfg['windows'] == [131072] and cfg['server']['gpu_memory_utilization'] == '0.96'
    assert [t['kind'] for t in cfg['trials']] == ['duck-transfer']
    builder.kaggle_files(output, cfg, smoke=False)
    text = (output/'experiment.ipynb').read_text()
    assert 'cache-sized phase-2 concurrency' in text
    assert 'two active games after two reach level 1' not in text


@pytest.mark.parametrize('options,expected', [
    (['--arena-30m', '--candidate', 'ornith15-35b-a3b-nvfp4'], True),
    (['--arena-30m'], True),
    (['--paired-30m', '--fixed-phase2-control'], False),
    (['--paired-30m', '--arena-variant', 'phase2-four'], False),
    (['--arena-30m', '--candidate', 'ornith15-35b-a3b-nvfp4', '--arena-variant', 'graph-128'], True),
    (['--arena-30m', '--candidate', 'ornith15-35b-a3b-nvfp4', '--arena-variant', 'graph-128',
      '--fixed-phase2-control'], False),
    (['--arena-30m', '--arena-variant', 'reclaim-graph-128'], False),
    (['--arena-30m', '--candidate', 'ornith15-35b-a3b-nvfp4', '--cudagraph-metrics'], True),
    (['--smoke'], False),
])
def test_cli_defaults_model_comparisons_to_measured_capacity(monkeypatch, tmp_path, options, expected):
    spec = importlib.util.spec_from_file_location('builder', ROOT/'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    captured = {}

    def build(name, **kwargs):
        captured.update(kwargs)
        return tmp_path, {}

    monkeypatch.setattr(builder, 'build', build)
    monkeypatch.setattr(builder, 'kaggle_files', lambda *args, **kwargs: 'test-digest')
    monkeypatch.setattr(sys, 'argv', ['build_arena.py', '--name', 'test', *options])
    builder.main()
    assert captured['capacity_phase2'] is expected
    assert captured['graph_metrics'] is ('--cudagraph-metrics' in options)


async def test_trial_installs_measured_limit_for_scheduler(tmp_path, monkeypatch):
    from test_arena_trial import STUDY, FakeArcade, FakeWorker, OfflineMonitor

    from arena import trial as module

    games = STUDY['games'][:6]
    cfg = {**STUDY, **study(), 'games': games, 'wall_seconds': 1.1,
           'cache_admission': 'hybrid_blocks'}
    measured = budget(400, 100)
    measured.layout['rule'] = 'test measured allocation'
    monkeypatch.setattr(module.BlockBudget, 'load', lambda *args: measured)
    monkeypatch.setattr(module, 'make_arcade', lambda *args: FakeArcade(games))
    monkeypatch.setattr(module, 'GpuMonitor', OfflineMonitor)

    async def agents(self, records, gate, window, *args):
        assert window == 131072 and gate.capacity == 400
        workers = [FakeWorker(g, clears_at=0.02 if g.index < 2 else None) for g in records]
        self.config_dict = {}

        async def close():
            await asyncio.gather(*(w.close() for w in workers))

        return workers, lambda: [w.result() for w in workers], close

    monkeypatch.setattr(module.Trial, 'duck_agents', agents)
    result = await module.Trial(cfg, cfg['trials'][0], source='.', games_dir='games',
                               output=tmp_path/'run', server_dir=tmp_path,
                               base_url='http://localhost', log=lambda *a, **k: None).run()
    assert result['status'] == 'complete'
    assert result['scheduler']['active_limit'] == 4
    assert result['window_choice']['window'] == 131072
    assert result['window_choice']['phase2']['cost_per_full_window_request'] == 100
    assert sum(g['state'] == 'active' for g in result['games']) == 4
