import asyncio
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from arena import sglang, sglang_run
from arena.gate import KVGate
from arena.trial import duck_config

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / 'configs/arena_30m_sglang.json').read_text())
STARTUP = {'max_total_num_tokens': 824384, 'max_running_requests': 4,
           'max_mamba_cache_size': 24, 'context_len': 524288}


def test_loading_retry_preserves_full_paired_trial_and_deadline():
    from sglang_bench.profile import ServerProfile
    profile = ServerProfile.from_dict(STUDY['sglang_profile'])
    assert profile.weight_loader_disable_mmap and profile.weight_loader_threads == 2
    assert len(STUDY['games']) == 25 and STUDY['wall_seconds'] == 1800
    assert [t['kind'] for t in STUDY['trials']] == ['duck-transfer', 'pi']
    assert STUDY['startup_timeout_seconds'] == 2700
    assert STUDY['windows'] == [131072] and STUDY['active_limit'] == 2


def test_sglang_capacity_is_verified_without_context_downgrade():
    budget = sglang.TokenBudget(STARTUP, STUDY)
    assert budget.capacity == int(824384 * .9) and budget.cost(131072) == 131136
    assert budget.units == 'tokens'
    for key in STARTUP:
        with pytest.raises(RuntimeError, match='capacity differs'):
            sglang.TokenBudget({**STARTUP, key: STARTUP[key] - 1}, STUDY)
    with pytest.raises(ValueError, match='concurrency'):
        sglang.TokenBudget(STARTUP, {**STUDY, 'generation_slots': 25})
    with pytest.raises(ValueError, match='context capacity'):
        budget.cost(524288)


async def test_sglang_gate_respects_both_tokens_and_mamba_concurrency():
    budget = sglang.TokenBudget(STARTUP, STUDY)
    gate = KVGate(budget.capacity, STUDY['generation_slots'], budget=budget)
    active = [await gate.acquire(str(i), 131072) for i in range(4)]
    waiting = asyncio.create_task(gate.acquire('fifth', 131072))
    await asyncio.sleep(0)
    assert not waiting.done() and gate.snapshot()['units'] == 'tokens'
    gate.release(active.pop())
    active.append(await waiting)
    for reservation in active:
        gate.release(reservation)
    assert gate.reserved == gate.inflight == 0


def test_the_duck_agent_uses_the_sglang_alias_and_the_context_target():
    kwargs = dict(window=131072, cap=16384, wall=1800, base_url='http://127.0.0.1:8001/v1')
    duck = duck_config(STUDY, ROOT, **kwargs)
    assert duck.model == 'pennyroyal'
    assert duck.max_model_len == 131072
    assert duck.gameplay_seconds == 1800
    assert duck.reasoning_effort == 'xhigh'


async def test_sglang_isolation_waits_for_idle_and_checks_flush_body(monkeypatch):
    calls = []
    attempts = 0

    async def sleep(_):
        pass

    def endpoint(request):
        nonlocal attempts
        calls.append(request.url.path)
        if request.url.path == '/metrics':
            attempts += 1
            return httpx.Response(200, text=f'sglang:num_running_reqs {int(attempts == 1)}\nsglang:num_queue_reqs 0\n')
        return httpx.Response(200, text='Cache flushed.\nPlease check backend logs for more details.')

    client = httpx.AsyncClient(transport=httpx.MockTransport(endpoint))
    monkeypatch.setattr(sglang.httpx, 'AsyncClient', lambda **kw: client)
    monkeypatch.setattr(sglang.asyncio, 'sleep', sleep)
    result = await sglang.isolate('http://localhost/v1')
    assert result['prefix_cache_reset'] and result['attempts'] == 2
    assert calls == ['/metrics', '/metrics', '/flush_cache']


def test_sglang_arena_bundle_includes_the_pinned_runtime_reference(tmp_path):
    spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build('sglang-test', output=tmp_path / 'bundle', paired_30m=True, sglang=True)
    builder.kaggle_files(output, study, smoke=False)
    assert (output / 'src/sglang_bench/launch.py').is_file()
    for name in ('tokenize_patch.py', 'tokenize_check.py'):
        assert (output / 'src/sglang_bench' / name).is_file()
    meta = json.loads((output / 'kernel-metadata.json').read_text())
    assert meta['dataset_sources'] == [builder.DATASET, 'richardcsaky/arc-sglang-runtime']
    assert study['wall_seconds'] == 1800 and [t['kind'] for t in study['trials']] == ['duck-transfer']
    assert study['notebook_seconds'] > study['startup_timeout_seconds'] + 2400


@pytest.mark.parametrize('failure', [None, 'capacity', 'timeout', 'tokenize'])
def test_sglang_controller_owns_server_and_stops_after_trials_or_capacity_failure(tmp_path, monkeypatch, failure):
    bundle, working = tmp_path / 'bundle', tmp_path / 'work'
    (bundle / 'configs').mkdir(parents=True)
    working.mkdir()
    (bundle / 'configs/comparison.json').write_text(json.dumps(STUDY))
    args = SimpleNamespace(bundle=str(bundle), working=str(working), input_root='/inputs',
                           pi_dir='/pi', node='/node', games_dir='/games', trial=None,
                           started_epoch=time.time(), max_seconds=9000)
    for name in ('prefetch_model', 'locate', 'verify_manifest', 'extract', 'link_cudart',
                 'link_cuda_driver', 'verify_model'):
        monkeypatch.setattr(sglang_run.runtime, name, lambda *a, **k: {})
    monkeypatch.setattr(sglang_run.runtime, 'check', lambda *a: {'problems': []})
    monkeypatch.setattr(sglang_run.tokenize_patch, 'apply', lambda *a: {'mode': 'tested'})
    monkeypatch.setattr(sglang_run.launch, 'recipe_env', lambda *a, **k: {})
    monkeypatch.setattr(sglang_run.launch, 'recipe_argv', lambda *a, **k: ['sglang', 'serve'])
    monkeypatch.setattr(sglang_run.launch, 'preflight', lambda *a, **k: {})
    monkeypatch.setattr(sglang_run.launch, 'warmup', lambda *a, **k: {'ok': True})
    async def parity(*args, **kwargs):
        if failure == 'tokenize':
            raise RuntimeError('token count mismatch')
        return {'state': 'passed'}
    monkeypatch.setattr(sglang_run.tokenize_check, 'run', parity)
    server = object()
    monkeypatch.setattr(sglang_run.launch, 'start', lambda *a, **k: server)
    def ready(*args, **kwargs):
        if failure == 'timeout':
            raise TimeoutError('draft still loading')
        return {} if failure == 'capacity' else STARTUP

    monkeypatch.setattr(sglang_run.launch, 'wait_ready', ready)
    stopped, commands = [], []

    def stop(actual):
        assert actual is server
        stopped.append(True)
        return {'port_free': True}

    real_run = sglang_run.subprocess.run

    def run(command, **kwargs):
        if not (isinstance(command, list) and '--trial' in command):
            return real_run(command, **kwargs)  # global patch: another test's leaked thread (CI red 9d9a9a591)
        commands.append(command)
        assert kwargs['timeout'] == 2400
        assert command[command.index('--base-url') + 1] == 'http://127.0.0.1:8001/v1'
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sglang_run.launch, 'stop', stop)
    monkeypatch.setattr(sglang_run.subprocess, 'run', run)
    if failure is None:
        assert sglang_run.run(args) == 0
        assert [c[c.index('--trial') + 1] for c in commands] == ['duck-transfer', 'qwen-c']
        assert (working / 'setup.json').is_file()
        assert json.loads((working / 'setup.json').read_text())['tokenize_patch'] == {'mode': 'tested'}
    else:
        error, message = {'timeout': (TimeoutError, 'draft still loading'),
                          'capacity': (RuntimeError, 'capacity differs'),
                          'tokenize': (RuntimeError, 'token count mismatch')}[failure]
        with pytest.raises(error, match=message):
            sglang_run.run(args)
        assert not commands
        setup = json.loads((working / 'setup.json').read_text())
        assert setup['stage'] == 'failed'
        assert setup['failed_stage'] == {'timeout': 'server_readiness', 'capacity': 'capacity_validation',
                                         'tokenize': 'tokenize_generation_parity'}[failure]
        assert setup['error']['type'] == error.__name__
        assert 'prefetch' in setup and 'runtime_extraction' in setup
        assert all('host' in s for s in setup['stages'])
    assert stopped == [True] and (working / 'serving/stop.json').is_file()
