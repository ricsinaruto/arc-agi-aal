"""fixed-2x-mtp0-wide: MTP-0 serving, six focus games at a 64K window, and the capacity it relies on."""

import asyncio
import json
import math
from pathlib import Path

import pytest
from test_arena_trial import STUDY, FakeArcade, FakeWorker, OfflineMonitor

from arena import trial as trial_module
from arena.analysis import phases, throughput
from arena.cache import BlockBudget
from arena.capacity import WINDOWS, choose_window, focus_fit
from arena.gate import KVGate
from arena.trial import Trial, duck_config
from arena.variants import MTP0_WIDE, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / 'configs/arena_30m.json').read_text())


def served_layout(mtp):
    """The pinned vLLM's exported layout (``cache_export_runtime.describe``) as recorded on 27 September:
    MTP-0 (level-bench calibration) 751 blocks of 800 tokens, fixed hybrid state 1 + 4 x 2 blocks, no lookahead;
    MTP-3 (serving-lab control) 222 blocks of 1,600 tokens, fixed state 1 + 4 x 5 blocks, lookahead 3."""
    block, fixed, blocks, lookahead = {0: (800, [1, 2, 2, 2, 2], 751, 0),
                                       3: (1600, [1, 5, 5, 5, 5], 222, 3)}[mtp]
    rows = []
    for n in sorted({1, 262144, *range(1024, 262144, 1024)}):
        groups = [math.ceil(n / block), *fixed]
        rows.append({'tokens': n, 'blocks': sum(groups), 'groups': groups})
    return {'schema': 1, 'num_blocks': blocks, 'max_model_len': 262144, 'lookahead': lookahead,
            'planned_cache_bytes': None, 'costs': rows, 'rule': 'test: recorded layout shape'}


def test_variant_changes_only_serving_depth_slots_window_and_switch():
    base = apply_variant(CONTROL, 'reclaim-fixed-2x')
    wide = apply_variant(CONTROL, 'fixed-2x-mtp0-wide')
    assert wide['server'] == {**base['server'], 'mtp_tokens': 0}
    assert wide['server']['max_num_seqs'] == wide['generation_slots'] == 25
    assert wide['windows'] == [65536] and wide['active_limit'] == 6 == MTP0_WIDE['active_limit']
    assert wide['opening_switch_level1_games'] == 6 and wide['require_focus_windows_fit'] is True
    assert wide['cache_admission'] == 'hybrid_blocks' and wide['kv_margin'] == 0.9
    assert wide['cache_experiment'] == {**base['cache_experiment'], 'compaction_changed': True}
    for key in ('budget_clock', 'level_budget_policy', 'backlog_policy', 'prioritize_parked_progress',
                'reasoning_effort', 'completion_cap', 'temperature', 'level_budget_rule', 'games'):
        assert wide[key] == base[key]
    assert wide['reasoning_effort'] == 'xhigh' and wide['completion_cap'] == 16384
    meta = wide['arena_variant']
    assert meta['reference_variant'] == 'reclaim-fixed-2x' and not meta['checkpoint_and_serving_unchanged']
    assert set(meta['reference_difference']) == {'server.mtp_tokens', 'active_limit', 'opening_switch_level1_games',
                                                 'windows', 'cache_experiment.compaction_changed',
                                                 'require_focus_windows_fit'}


@pytest.mark.parametrize('server', [{'mtp_tokens': 1}, {'speculative': {'method': 'ngram_gpu'}}])
def test_variant_requires_the_mtp3_control(server):
    with pytest.raises(ValueError, match='MTP-0 wide arm'):
        apply_variant({**CONTROL, 'server': {**CONTROL['server'], **server}}, 'fixed-2x-mtp0-wide')


def test_client_lookahead_and_window_follow_the_server():
    wide = apply_variant(CONTROL, 'fixed-2x-mtp0-wide')
    cfg = duck_config(wide, ROOT, window=wide['windows'][0], cap=wide['completion_cap'], wall=3600,
                      base_url='http://model/v1')
    assert cfg.speculative_tokens == 0 and cfg.max_model_len == 65536 and cfg.max_output_tokens == 16384
    assert cfg.parallel_games == 25 and cfg.max_num_seqs >= 25
    cfg.__post_init__()  # the agent config's own validation accepts lookahead 0 at this window


def test_recorded_mtp0_layout_admits_seven_full_64k_windows():
    mtp0 = BlockBudget(served_layout(0), 0.9)
    assert mtp0.capacity == 675
    assert [mtp0.cost(n) for n in (1, 4096, 16384, 32768, 49152, 65536)] == [10, 15, 30, 50, 71, 91]
    fit = focus_fit(mtp0, 65536, 6, required=True)
    assert fit['cost_per_full_window_request'] == 93 and fit['full_window_requests'] == 7
    assert fit['lookahead'] == 0 and fit['request_tokens'] == 65536 + 256
    # The same six windows on the MTP-3 server's layout: 63 of 199 blocks each, three fit.
    mtp3 = BlockBudget(served_layout(3), 0.9)
    assert mtp3.capacity == 199 and focus_fit(mtp3, 65536, 6)['full_window_requests'] == 3
    with pytest.raises(RuntimeError, match='only 3 fit'):
        focus_fit(mtp3, 65536, 6, required=True)
    # The base (two 128K windows on MTP-3) never required it: one full window fits.
    assert focus_fit(mtp3, 131072, 2)['full_window_requests'] == 1


def test_token_pool_window_choice_resolves_64k_for_six_games():
    mtp0_pool, mtp3_pool = 584_184, 314_572  # serving lab v1 "GPU KV cache size"
    assert choose_window(mtp0_pool, active_limit=6, windows=(65536,))['window'] == 65536
    assert choose_window(mtp0_pool, active_limit=6, windows=WINDOWS)['window'] == 65536
    with pytest.raises(RuntimeError):
        choose_window(mtp3_pool, active_limit=6, windows=(65536,))


async def test_gate_admits_seven_full_windows_and_queues_the_eighth():
    budget = BlockBudget(served_layout(0), 0.9)
    gate = KVGate(budget.capacity, slots=25, budget=budget)
    held = [await gate.acquire(f'g{i}', 65536 + 256) for i in range(7)]
    assert gate.reserved == 7 * 93 <= gate.capacity and gate.inflight == 7
    waiter = asyncio.create_task(gate.acquire('g7', 65536 + 256))
    await asyncio.sleep(0)
    assert not waiter.done() and gate.snapshot()['waiting'] == 1
    gate.release(held[0])
    assert (await waiter).tokens == 93 and gate.reserved <= gate.capacity


def trial_study(tmp_path, monkeypatch, mtp):
    games = STUDY['games'][:3]
    study = {**STUDY, 'games': games, 'wall_seconds': 0.3, 'cache_admission': 'hybrid_blocks',
             'windows': [65536], 'active_limit': 6, 'opening_switch_level1_games': 6,
             'require_focus_windows_fit': True}
    budget = BlockBudget(served_layout(mtp), 0.9)
    monkeypatch.setattr(trial_module.BlockBudget, 'load', lambda *args: budget)
    monkeypatch.setattr(trial_module, 'make_arcade', lambda *a: FakeArcade(games))
    monkeypatch.setattr(trial_module, 'GpuMonitor', OfflineMonitor)
    return study, budget


async def test_trial_records_focus_fit_on_the_mtp0_layout(tmp_path, monkeypatch):
    study, budget = trial_study(tmp_path, monkeypatch, 0)

    async def agents(self, records, gate, window, wall, card, arc):
        assert window == 65536 and gate.budget is budget and gate.capacity == 675
        workers = [FakeWorker(record) for record in records]
        self.config_dict = {}

        async def close():
            pass

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, 'duck_agents', agents)
    trial = Trial(study, study['trials'][0], source='.', games_dir='games', output=tmp_path / 't',
                  server_dir=tmp_path, base_url='http://localhost/v1', log=lambda *a, **k: None)
    report = await trial.run()
    fit = report['window_choice']['focus_fit']
    assert fit['full_window_requests'] == 7 and fit['active_limit'] == 6 and fit['required'] is True
    assert report['scheduler']['active_limit'] == 6


async def test_trial_refuses_six_focus_windows_on_an_mtp3_server(tmp_path, monkeypatch):
    study, _ = trial_study(tmp_path, monkeypatch, 3)

    async def agents(*args):
        raise AssertionError('no gameplay before the capacity check')

    monkeypatch.setattr(Trial, 'duck_agents', agents)
    trial = Trial(study, study['trials'][0], source='.', games_dir='games', output=tmp_path / 't',
                  server_dir=tmp_path, base_url='http://localhost/v1', log=lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='only 3 fit'):
        await trial.run()


def test_throughput_summary_reads_tokens_per_hour_phase_rates_streams_and_clears_per_10k():
    report = {'wall_seconds': 3600, 'levels': 12, 'completed_reply_tokens': 1_200_000,
              'scheduler': {'switch_wall_seconds': 600, 'active_limit': 6},
              'engine': {'valid': True, 'sampled_seconds': 3600, 'accepted_generation_tokens': 1_440_000,
                         'preemptions': 0, 'peak_running_requests': 9},
              'gate': {'capacity': 675, 'units': 'cache_blocks', 'slots': 25, 'peak_reserved': 640,
                       'peak_inflight': 9, 'peak_waiting': 4},
              'window_choice': {'window': 65536, 'focus_fit': {'full_window_requests': 7}}}
    samples = [{'elapsed_seconds': t, 'metrics': {
        'vllm:generation_tokens_total': n, 'vllm:prompt_tokens_by_source_total:local_compute': n,
        'vllm:prompt_tokens_by_source_total:local_cache_hit': n, 'vllm:num_requests_running': running,
        'vllm:kv_cache_usage_perc': .5}}
        for t, n, running in [(99, 0, 0), (600, 240_000, 12), (3600, 1_440_000, 6)]]
    out = throughput(report, phases(report, samples))
    assert out['engine_generation_tokens_per_hour'] == 1_440_000
    assert out['completed_reply_tokens_per_hour'] == 1_200_000
    assert out['clears_per_10k_completed_tokens'] == pytest.approx(0.1)
    assert out['clears_per_10k_engine_tokens'] == pytest.approx(12e4 / 1_440_000)
    assert out['preemptions'] == 0 and out['window'] == 65536 and out['phase2_limit'] == 6
    assert out['gate']['peak_inflight'] == 9 and out['focus_fit']['full_window_requests'] == 7
    assert out['phases']['opening']['generation_tokens_per_second'] == pytest.approx(400)
    phase2 = out['phases']['phase2']
    assert phase2['generation_tokens_per_second'] == pytest.approx(400)
    assert phase2['peak_running_requests'] == 12 and phase2['mean_running_requests'] == pytest.approx(9)
    # Missing counters stay missing instead of becoming zero rates.
    empty = throughput({'wall_seconds': 3600, 'engine': {'valid': False}}, None)
    assert empty['engine_generation_tokens_per_hour'] is None and empty['clears_per_10k_completed_tokens'] is None


def test_competition_fit_falls_back_to_what_fits_instead_of_refusing():
    from arena.capacity import competition_focus_fit

    mtp0 = BlockBudget(served_layout(0), 0.9)  # seven full 64K windows fit
    study = {'active_limit': 14, 'require_focus_windows_fit': True, 'focus_fit_fallback': {'min_slots': 6}}
    fit = competition_focus_fit(mtp0, 65536, study)
    assert study['active_limit'] == 7 and fit['active_limit'] == 7 and fit['fallback_from'] == 14
    study = {'active_limit': 6, 'require_focus_windows_fit': True, 'focus_fit_fallback': {'min_slots': 6}}
    assert 'fallback_from' not in competition_focus_fit(mtp0, 65536, study) and study['active_limit'] == 6
    mtp3 = BlockBudget(served_layout(3), 0.9)  # three fit: below the floor, refuse before gameplay
    with pytest.raises(RuntimeError, match='at least 6'):
        competition_focus_fit(mtp3, 65536, {'active_limit': 14, 'focus_fit_fallback': {'min_slots': 6}})
    # Without the fallback key it is the plain required check (research arms keep refusing).
    with pytest.raises(RuntimeError, match='only 7 fit'):
        competition_focus_fit(mtp0, 65536, {'active_limit': 14, 'require_focus_windows_fit': True})


def test_21_by_42k_fits_the_served_hero_layout_and_the_duck_config():
    import json

    from arena.cache import BlockBudget
    from arena.variants import HERO_21_SLOTS, HERO_21_WINDOW

    # The hero smoke's served layout (860 blocks, lookahead 0), recorded in the repository evidence when present.
    path = ROOT / 'runs/modal-validation/arena-hero-smoke-v1-modal-20260928/arena/serving/cache-layout.json'
    if path.exists():
        budget = BlockBudget(json.loads(path.read_text()), 0.9)
        assert focus_fit(budget, HERO_21_WINDOW, HERO_21_SLOTS, required=True)['full_window_requests'] == 21
    arm = apply_variant(CONTROL, 'fixed-2x-mtp0-wide-kv8-hero-21x42k')
    cfg = duck_config(arm, ROOT, window=arm['windows'][0], cap=arm['completion_cap'], wall=3600,
                      base_url='http://model/v1')
    assert cfg.max_model_len == HERO_21_WINDOW and cfg.max_output_tokens == 16384
    cfg.__post_init__()

