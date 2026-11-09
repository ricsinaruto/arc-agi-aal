"""Serving-only throughput grid: fixed request shapes, block-gated admission, no gameplay."""

import importlib.util
import json
import re
from pathlib import Path

import httpx
import pytest
from test_arena_cache import layout
from test_qwen_serving import import_fork

from arena import throughput_bench as tb
from arena.variants import NGRAM_SPECULATION

ROOT = Path(__file__).resolve().parents[1]


def test_grid_and_request_shapes():
    points = tb.grid_points(tb.DEFAULT_GRID)
    assert len(points) == 13 and points[-1]['kind'] == 'shared_prefix'
    assert {p['concurrency'] for p in points[:-1]} == {1, 2, 4, 8}
    body = tb.request_body(7, 100, output_tokens=512, forced=True)
    assert body['max_tokens'] == body['min_tokens'] == 512 and body['ignore_eos'] is True
    assert body['chat_template_kwargs'] == {'enable_thinking': True, 'reasoning_effort': 'xhigh'}
    assert body['messages'][1]['content'][1]['type'] == 'image_url'
    plain = tb.request_body(7, 100)
    assert 'min_tokens' not in plain and plain['max_tokens'] == 1
    assert tb.synthetic_text(1, 50) == tb.synthetic_text(1, 50) != tb.synthetic_text(2, 50)
    shared_a = tb.request_body(1_000_000, 100, tail_seed=1, tail_words=5)
    shared_b = tb.request_body(1_000_000, 100, tail_seed=2, tail_words=5)
    ta, tb_ = shared_a['messages'][1]['content'][0]['text'], shared_b['messages'][1]['content'][0]['text']
    assert ta != tb_ and ta.split('\nLatest step')[0] == tb_.split('\nLatest step')[0]


class FakeServer:
    """Tokenizer, chat completions, metrics and cache reset of a vLLM-like server."""

    def __init__(self):
        self.generated = 0.0
        self.prompt_compute = 0.0
        self.running = 0
        self.requests = []

    def count(self, body):
        text = json.dumps(body['messages'])
        return max(1, len(text) // 6)

    def handler(self, request):
        path = request.url.path
        if path.endswith('/tokenize'):
            body = json.loads(request.content)
            return httpx.Response(200, json={'count': self.count(body), 'max_model_len': 262144})
        if path.endswith('/chat/completions'):
            body = json.loads(request.content)
            self.requests.append(body)
            prompt = self.count(body)
            completion = body['max_tokens'] if body.get('ignore_eos') else 1
            self.generated += completion
            self.prompt_compute += prompt
            return httpx.Response(200, json={
                'usage': {'prompt_tokens': prompt, 'completion_tokens': completion},
                'choices': [{'finish_reason': 'length' if completion > 1 else 'stop', 'message': {'content': 'x'}}]})
        if path.endswith('/metrics'):
            text = (f'vllm:generation_tokens_total {self.generated}\n'
                    f'vllm:num_preemptions_total 0.0\n'
                    f'vllm:prompt_tokens_by_source_total{{source="local_compute"}} {self.prompt_compute}\n'
                    f'vllm:prompt_tokens_by_source_total{{source="local_cache_hit"}} 0.0\n'
                    f'vllm:num_requests_running {self.running}\nvllm:num_requests_waiting 0\n'
                    'vllm:kv_cache_usage_perc 0.2\n')
            return httpx.Response(200, text=text)
        if path.endswith('/reset_prefix_cache'):
            return httpx.Response(200, json={'success': True})
        raise AssertionError(path)


async def test_bench_measures_every_point_through_the_block_gate(tmp_path, monkeypatch):
    server = FakeServer()
    transport = httpx.MockTransport(server.handler)
    real_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs.setdefault('transport', transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', patched_client)
    # Sampling every 5 s would slow the test; sample on demand only.
    monkeypatch.setattr(tb.GpuMonitor, 'run', lambda self: __import__('asyncio').sleep(3600))
    (tmp_path / 'server.log').write_text('worker ARC_CACHE_LAYOUT ' + json.dumps(layout()) + '\n')
    study = {'kv_margin': 0.9, 'generation_slots': 25, 'completion_cap': 16384, 'reasoning_effort': 'xhigh',
             'temperature': 0.7, 'server': {'mtp_tokens': 3},
             'throughput_bench': {'prompt_tokens': [300, 600], 'concurrency': [1, 2], 'output_tokens': 64,
                                  'shared_prefix': {'prompt_tokens': 300, 'concurrency': 2, 'tail_tokens': 16}}}
    report = await tb.run_bench(study, {'name': 'serving-bench', 'kind': 'serving-bench'}, output=tmp_path / 'out',
                                server_dir=tmp_path, base_url='http://model/v1', transport=transport, log=lambda *a, **k: None)
    assert report['status'] == 'complete' and report['preemption_free'] is True and not report['task_errors']
    assert len(report['points']) == 5 and report['points'][-1]['kind'] == 'shared_prefix'
    for point in report['points']:
        prefill, decode = point['passes']
        assert prefill['pass'] == 'prefill' and prefill['requests'] == point['concurrency']
        assert decode['completion_tokens'] == 64 * point['concurrency'] and decode['finish_reasons'] == ['length']
        assert decode['engine_generation_tokens'] == 64 * point['concurrency']
        assert point['fits_concurrently'] >= 1 and point['cache_reset']['prefix_cache_reset'] is True
    assert (tmp_path / 'out/bench.json').is_file() and report['calibration'][300]['words'] > 0
    forced = [r for r in server.requests if r.get('ignore_eos')]
    assert len(forced) == sum(p['concurrency'] for p in report['points'])
    assert all(r['min_tokens'] == r['max_tokens'] == 64 for r in forced)


async def test_bench_records_a_failed_point_and_continues(tmp_path, monkeypatch):
    server = FakeServer()
    calls = {'n': 0}
    original = server.handler

    def flaky(request):
        if request.url.path.endswith('/chat/completions'):
            calls['n'] += 1
            if calls['n'] == 1:
                return httpx.Response(500, text='boom')
        return original(request)

    transport = httpx.MockTransport(flaky)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda *a, **k: real_client(*a, **{**k, 'transport': transport}))
    monkeypatch.setattr(tb.GpuMonitor, 'run', lambda self: __import__('asyncio').sleep(3600))
    (tmp_path / 'server.log').write_text('worker ARC_CACHE_LAYOUT ' + json.dumps(layout()) + '\n')
    study = {'kv_margin': 0.9, 'generation_slots': 25, 'completion_cap': 16384, 'reasoning_effort': 'xhigh',
             'temperature': 0.7, 'server': {}, 'throughput_bench': {'prompt_tokens': [300], 'concurrency': [1],
                                                                    'output_tokens': 8, 'shared_prefix': None}}
    report = await tb.run_bench(study, {'name': 'serving-bench', 'kind': 'serving-bench'}, output=tmp_path / 'out',
                                server_dir=tmp_path, base_url='http://model/v1', log=lambda *a, **k: None)
    assert report['status'] == 'complete'
    prefill = report['points'][0]['passes'][0]
    assert prefill['requests'] == 0 and prefill['failures'][0]['status'] == 500
    assert report['points'][0]['passes'][1]['completion_tokens'] == 8


def test_builder_makes_a_serving_bench_study_on_the_reclaimed_flash_profile(tmp_path):
    spec = importlib.util.spec_from_file_location('bench_builder', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build('serving-bench-v1', output=tmp_path / 'b', paired_30m=True,
                                  arena_variant='cache-reclaim', throughput_bench=True)
    assert study['trials'] == [{'name': 'serving-bench', 'kind': 'serving-bench'}]
    assert study['throughput_bench'] == builder.THROUGHPUT_GRID
    assert study['cache_experiment']['reclaim_skipped_blocks'] is True and study['server']['mtp_tokens'] == 3
    assert study['notebook_seconds'] == 3600 and study['kernel_timeout'] == 3900 and study['wall_seconds'] == 1500
    assert 'phase2_admission' not in study
    saved = json.loads((output / 'configs/comparison.json').read_text())
    assert saved == study and (output / 'src/arena/throughput_bench.py').is_file()
    notebook = builder.notebook(study, 'runtime.blob', 'fake-digest', smoke=False)
    assert re.search(r"'--max-seconds', '3600'", notebook['cells'][1]['source'])
    with pytest.raises(ValueError, match='serving benchmark'):
        builder.build('bad', output=tmp_path / 'c', paired_30m=True, candidate_profile='qwen38-27b-fp8',
                      throughput_bench=True)
    with pytest.raises(ValueError, match='serving benchmark'):
        builder.build('bad2', output=tmp_path / 'd', paired_30m=True, arena_variant='reclaim-phase2-four',
                      throughput_bench=True)
    # Serving-only variants may be benchmarked: the same grid on MTP-1 and graph-128 profiles.
    _, mtp1 = builder.build('bench-mtp1', output=tmp_path / 'e', paired_30m=True, arena_variant='reclaim-mtp1',
                            throughput_bench=True)
    _, graph = builder.build('bench-graph', output=tmp_path / 'f', paired_30m=True, arena_variant='reclaim-graph-128',
                             throughput_bench=True)
    assert mtp1['server']['mtp_tokens'] == 1 and mtp1['trials'][0]['kind'] == 'serving-bench'
    _, mtp4 = builder.build('bench-mtp4', output=tmp_path / 'g', paired_30m=True, arena_variant='reclaim-mtp4',
                            throughput_bench=True)
    assert mtp4['server']['mtp_tokens'] == 4 and mtp4['trials'][0]['kind'] == 'serving-bench'
    assert graph['server']['max_cudagraph_capture_size'] == 128 and graph['throughput_bench'] == builder.THROUGHPUT_GRID
    output, ngram = builder.build('bench-ngram', output=tmp_path / 'h', paired_30m=True, arena_variant='reclaim-ngram',
                                  throughput_bench=True)
    assert ngram['server']['mtp_tokens'] == 0 and ngram['server']['speculative'] == NGRAM_SPECULATION
    assert ngram['trials'][0]['kind'] == 'serving-bench' and ngram['throughput_bench'] == builder.THROUGHPUT_GRID
    assert ngram['kernel_timeout'] == 3900 and ngram['arena_variant']['name'] == 'reclaim-ngram'
    assert ngram['cache_experiment']['reclaim_skipped_blocks'] is True
    assert json.loads((output / 'configs/comparison.json').read_text()) == ngram


def test_ngram_bench_server_command_replaces_the_mtp_draft(tmp_path, monkeypatch):
    """The actual pinned command: one n-gram --speculative-config, no MTP config, flags otherwise intact."""
    from types import SimpleNamespace as NS

    from arena.candidate_profiles import NEX
    from arena.candidate_serving import configure_candidate
    from arena.serving import configure

    recipe = import_fork(tmp_path, monkeypatch)
    for key, value in {'TAAF_VLLM_MAX_NUM_SEQS': '25', 'TAAF_VLLM_ENABLE_PREFIX_CACHING': '1',
                       'TAAF_VLLM_MAX_NUM_BATCHED_TOKENS': '4096', 'TAAF_VLLM_MTP_TOKENS': '0'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(recipe, 'verify_and_extract_runtime', lambda *a, **k: {})
    recipe.ANALYZER_CONTEXT = 262144  # as qwen_baselines.serving.start sets before launching
    receipt = configure(recipe, tmp_path, reclaim_skipped_blocks=True, speculative=NGRAM_SPECULATION)
    assert receipt['speculative'] == NGRAM_SPECULATION and receipt['reclaim_skipped_blocks'] is True
    tuning = recipe.resolve_vllm_tuning()
    assert not recipe.validate_vllm_tuning_document(tuning) and tuning['mtp_speculative_tokens'] == 0
    argv = recipe.server_command(tmp_path / 'model', tuning=tuning)
    assert argv.count('--speculative-config') == 1
    assert json.loads(argv[argv.index('--speculative-config') + 1]) == NGRAM_SPECULATION
    assert not any('mtp' in arg for arg in argv)
    # The rest of the profile is the control's: async scheduling, native context, 25 slots, prefix caching.
    assert '--async-scheduling' in argv and '--enable-prefix-caching' in argv
    assert argv[argv.index('--max-model-len') + 1] == '262144' and argv[argv.index('--max-num-seqs') + 1] == '25'
    # The arena hook appends the config after the script's own tail flags, which are untouched.
    assert argv[-6:-2] == ['--no-enable-log-requests', '--disable-uvicorn-access-log', '--uvicorn-log-level', 'info']
    assert argv[-2] == '--speculative-config'
    # Launching with an MTP depth as well would produce two speculative configs: rejected.
    with pytest.raises(ValueError, match='MTP depth 0'):
        recipe.server_command(tmp_path / 'model', tuning={**tuning, 'mtp_speculative_tokens': 3})
    # The candidate MTP-weights guard lets an n-gram draft through (no MTP weights needed) and
    # still rejects an MTP launch on a checkpoint without them.
    configure_candidate(recipe, NEX)
    command = recipe.server_command(tmp_path / 'model', tuning=recipe.resolve_vllm_tuning())
    assert json.loads(command[command.index('--speculative-config') + 1]) == NGRAM_SPECULATION
    stub = NS(server_command=lambda: ['serve', '--speculative-config', '{"method":"mtp","num_speculative_tokens":3}',
                                      '--quantization', 'modelopt_fp4', '--tool-call-parser', 'x', '--reasoning-parser', 'y'],
              MTP_TOKENS_ENV='T_MTP', MTP_INDEX_SHARE_FOR_ITERATION_ENV='T_SHARE',
              MTP_DYNAMIC_BATCH_SCHEDULE_ENV='T_DYN', MOE_BACKEND_ENV='T_MOE')
    configure_candidate(stub, NEX)
    with pytest.raises(RuntimeError, match='no MTP weights'):
        stub.server_command()
    # Only the GPU n-gram proposer is allowed under the pinned --async-scheduling launch.
    for bad in ({**NGRAM_SPECULATION, 'method': 'ngram'}, {**NGRAM_SPECULATION, 'prompt_lookup_min': 5},
                {**NGRAM_SPECULATION, 'num_speculative_tokens': '3'}, {'method': 'ngram_gpu'}):
        with pytest.raises(ValueError):
            configure(NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ['serve']), tmp_path,
                      speculative=bad)


def test_transcript_replay_grid_and_bodies(tmp_path):
    from arena.variants import NGRAM8_SPECULATION
    spec = {'prompt_tokens': [], 'concurrency': [], 'output_tokens': 512, 'shared_prefix': None,
            'transcript': {'file': 'bench/transcript.txt', 'prompt_tokens': [8192, 32768], 'concurrency': [1, 2, 3]}}
    points = tb.grid_points(spec)
    assert len(points) == 6 and {p['kind'] for p in points} == {'transcript'}
    assert points[0] == {'kind': 'transcript', 'prompt_tokens': 8192, 'concurrency': 1}
    text = '\n'.join(f'line {i}: current_frame.ascii diff {i * 7}' for i in range(200))
    tail = tb.transcript_tail(text, 100)
    assert len(tail) <= 100 + 40 and tail.startswith('line ') and text.endswith(tail)
    assert tb.transcript_tail(text, 10 ** 6) == text
    body = tb.transcript_body(3, tail, output_tokens=512, forced=True)
    assert body['min_tokens'] == body['max_tokens'] == 512 and body['ignore_eos'] is True and body['seed'] == 3
    assert isinstance(body['messages'][1]['content'], str) and tail in body['messages'][1]['content']
    assert body['messages'][1]['content'].startswith('Session 3.')
    assert tb.transcript_body(4, tail)['messages'][1]['content'] != body['messages'][1]['content']
    # Speculative-decoding counters are reported per pass (zero when the server exposes none).
    summary = tb.summarize_pass('decode', 2.0, [], {'vllm:spec_decode_num_drafts_total': 10.0,
                                                    'vllm:spec_decode_num_draft_tokens_total': 30.0,
                                                    'vllm:spec_decode_num_accepted_tokens_total': 12.0},
                                {'vllm:spec_decode_num_drafts_total': 30.0, 'vllm:spec_decode_num_draft_tokens_total': 90.0,
                                 'vllm:spec_decode_num_accepted_tokens_total': 42.0})
    assert summary['spec_drafts'] == 20.0 and summary['spec_draft_tokens'] == 60.0 and summary['accepted_tokens_per_draft'] == 1.5
    assert tb.summarize_pass('decode', 1.0, [], {}, {})['accepted_tokens_per_draft'] is None
    assert NGRAM8_SPECULATION['num_speculative_tokens'] == 8 and NGRAM8_SPECULATION['prompt_lookup_max'] == 6


async def test_bench_replays_a_transcript_from_the_bundle(tmp_path, monkeypatch):
    server = FakeServer()
    transport = httpx.MockTransport(server.handler)
    real_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs.setdefault('transport', transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', patched_client)
    monkeypatch.setattr(tb.GpuMonitor, 'run', lambda self: __import__('asyncio').sleep(3600))
    (tmp_path / 'server.log').write_text('worker ARC_CACHE_LAYOUT ' + json.dumps(layout()) + '\n')
    (tmp_path / 'bench').mkdir()
    text = '\n'.join(f'--- step {i} ---\n[python]\nprint(current_frame.ascii[{i}])\n[result] row {i} ok' for i in range(400))
    (tmp_path / 'bench/transcript.txt').write_text(text)
    study = {'kv_margin': 0.9, 'generation_slots': 25, 'completion_cap': 16384, 'reasoning_effort': 'xhigh',
             'temperature': 0.7, 'server': {'mtp_tokens': 3},
             'throughput_bench': {'prompt_tokens': [], 'concurrency': [], 'output_tokens': 32, 'shared_prefix': None,
                                  'transcript': {'file': 'bench/transcript.txt', 'prompt_tokens': [300, 900], 'concurrency': [1, 2]}}}
    report = await tb.run_bench(study, {'name': 'serving-bench', 'kind': 'serving-bench'}, output=tmp_path / 'out',
                                server_dir=tmp_path, base_url='http://model/v1', transport=transport,
                                log=lambda *a, **k: None, source=tmp_path)
    assert report['status'] == 'complete' and not report['task_errors'] and len(report['points']) == 4
    for point in report['points']:
        assert point['kind'] == 'transcript' and point['words'] is None
        assert abs(point['calibration']['tokens'] - point['prompt_tokens']) <= point['prompt_tokens'] * 0.1
        prefill, decode = point['passes']
        assert prefill['requests'] == point['concurrency'] and decode['completion_tokens'] == 32 * point['concurrency']
        assert 'accepted_tokens_per_draft' in decode
    replayed = [r for r in server.requests if r['messages'][1]['content'].startswith('Session ')]
    assert replayed and all('current_frame.ascii' in r['messages'][1]['content'] for r in replayed)
    assert all(isinstance(r['messages'][1]['content'], str) for r in replayed)
    with pytest.raises(ValueError, match='source'):
        tb.Bench(study, tmp_path / 'x', server_dir=tmp_path, base_url='http://model/v1').transcript()


def test_builder_bundles_the_replay_transcript(tmp_path):
    spec = importlib.util.spec_from_file_location('bench_builder2', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from arena.variants import NGRAM8_SPECULATION
    transcript = tmp_path / 'lp85-transcript.txt'
    transcript.write_text('--- analysis_step=0 ---\n[SYSTEM PROMPT]\nYou are a coding agent.\n' * 50)
    output, study = builder.build('bench-replay', output=tmp_path / 'r', paired_30m=True, arena_variant='reclaim-ngram8',
                                  throughput_bench=True, bench_transcript=str(transcript))
    grid = study['throughput_bench']
    assert grid['prompt_tokens'] == [] and grid['shared_prefix'] is None and grid['output_tokens'] == 512
    assert grid['transcript']['file'] == 'bench/transcript.txt' and grid['transcript']['prompt_tokens'] == [8192, 32768, 65536, 98304]
    assert grid['transcript']['concurrency'] == [1, 2, 3] and grid['transcript']['chars'] == len(transcript.read_text())
    assert (output / 'bench/transcript.txt').read_text() == transcript.read_text()
    assert study['server']['mtp_tokens'] == 0 and study['server']['speculative'] == NGRAM8_SPECULATION
    assert study['arena_variant']['reference_variant'] == 'cache-reclaim'
    assert tb.grid_points(grid) == [{'kind': 'transcript', 'prompt_tokens': length, 'concurrency': c}
                                    for length in [8192, 32768, 65536, 98304] for c in [1, 2, 3]]
    _, control = builder.build('bench-replay-control', output=tmp_path / 's', paired_30m=True, arena_variant='cache-reclaim',
                               throughput_bench=True, bench_transcript=str(transcript))
    assert control['server']['mtp_tokens'] == 3 and control['throughput_bench'] == grid
    with pytest.raises(ValueError, match='serving benchmark'):
        builder.build('bad', output=tmp_path / 't', paired_30m=True, arena_variant='cache-reclaim', bench_transcript=str(transcript))


def test_lab_points_extend_the_product_grid_and_trace_summary():
    spec = {'prompt_tokens': [4096, 16384], 'concurrency': [1, 24], 'output_tokens': 512, 'shared_prefix': None,
            'transcript': None, 'points': [[65536, 1], [65536, 2]]}
    points = tb.grid_points(spec)
    assert len(points) == 6 and points[-2:] == [{'kind': 'distinct', 'prompt_tokens': 65536, 'concurrency': c}
                                                for c in (1, 2)]
    assert tb.grid_points({**spec, 'points': None}) == points[:4]
    trace = {'traceEvents': [
        {'ph': 'X', 'cat': 'kernel', 'name': 'fused_moe_cutlass_kernel', 'ts': 0, 'dur': 40},
        {'ph': 'X', 'cat': 'kernel', 'name': 'qsa_sparse_attention', 'ts': 30, 'dur': 20},  # overlaps the MoE
        {'ph': 'X', 'cat': 'kernel', 'name': 'fused_recurrent_gated_delta_rule', 'ts': 60, 'dur': 10},
        {'ph': 'X', 'cat': 'gpu_memcpy', 'name': 'Memcpy HtoD', 'ts': 90, 'dur': 5},
        {'ph': 'X', 'cat': 'kernel', 'name': 'topk_topp_sampling_from_probs', 'ts': 95, 'dur': 5},
        {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaGraphLaunch', 'ts': 0, 'dur': 2},
        {'ph': 'X', 'cat': 'cpu_op', 'name': 'aten::mm', 'ts': 0, 'dur': 100},
    ]}
    summary = tb.summarize_trace(trace)
    assert summary['kernels'] == 5 and summary['window_ms'] == 0.1
    assert summary['busy_ms'] == 0.07 and summary['busy_fraction'] == 0.7  # 0-50, 60-70, 90-100
    assert summary['families_ms'] == {'moe': 0.04, 'attention': 0.02, 'linear_attention': 0.01, 'gpu_memcpy': 0.005,
                                      'sampling': 0.005}
    assert summary['top_kernels'][0]['name'] == 'fused_moe_cutlass_kernel' and summary['launches'] == {'cudaGraphLaunch': 1}
    assert tb.summarize_trace({'traceEvents': []}) == {'kernels': 0}
    assert tb.kernel_family('void cutlass::Kernel<sm120_gemm>') == 'gemm' and tb.kernel_family('xyz') == 'other'


async def test_lab_trial_grid_kv_tokens_and_decode_profile(tmp_path, monkeypatch):
    import gzip

    server = FakeServer()
    profile_dir = tmp_path / tb.PROFILE_DIR
    calls = []
    original = server.handler

    def handler(request):
        path = request.url.path
        if path in ('/start_profile', '/stop_profile'):
            calls.append(path)
            if path == '/stop_profile':
                profile_dir.mkdir(exist_ok=True)
                name = f'rank0.{len(calls)}.pt.trace.json.gz'
                with gzip.open(profile_dir / name, 'wt') as stream:
                    json.dump({'traceEvents': [{'ph': 'X', 'cat': 'kernel', 'name': 'moe_gemm', 'ts': 0, 'dur': 10}]},
                              stream)
                (profile_dir / 'profiler_out_0.txt').write_text('table')
            return httpx.Response(200)
        return original(request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda *a, **k: real_client(*a, **{**k, 'transport': transport}))
    monkeypatch.setattr(tb.GpuMonitor, 'run', lambda self: __import__('asyncio').sleep(3600))
    (tmp_path / 'server.log').write_text('worker ARC_CACHE_LAYOUT ' + json.dumps(layout()) + '\n'
                                         'INFO GPU KV cache size: 314,572 tokens, Maximum concurrency 1.20x\n')
    study = {'kv_margin': 0.9, 'generation_slots': 25, 'completion_cap': 16384, 'reasoning_effort': 'xhigh',
             'temperature': 0.7, 'server': {'mtp_tokens': 3}, 'throughput_bench': {'prompt_tokens': [300]}}
    trial = {'name': 'lab-control', 'kind': 'serving-bench', 'server_env': {'ARENA_TORCH_PROFILER': '1'},
             'throughput_bench': {'prompt_tokens': [200], 'concurrency': [2], 'output_tokens': 16, 'shared_prefix': None,
                                  'points': [[400, 1]], 'profile': {'prompt_tokens': 200, 'concurrency': [2, 1],
                                                                    'output_tokens': 8}}}
    report = await tb.run_bench(study, trial, output=tmp_path / 'out', server_dir=tmp_path,
                                base_url='http://model/v1', log=lambda *a, **k: None)
    assert report['status'] == 'complete' and report['kv_cache_tokens'] == 314572
    assert [(p['prompt_tokens'], p['concurrency']) for p in report['points']] == [(200, 2), (400, 1)]
    assert calls == ['/start_profile', '/stop_profile'] * 2
    first, second = report['profile']
    assert first['concurrency'] == 2 and first['decode']['completion_tokens'] == 16
    assert first['traces'][0]['summary']['families_ms'] == {'moe': 0.01}
    assert (tmp_path / 'out/profile-c2/profiler_out_0.txt').read_text() == 'table'
    assert second['traces'][0]['file'] != first['traces'][0]['file'] and not list(profile_dir.glob('*.gz'))
    # Without a profiler on the server the grid stands and the profile records the refusal.
    calls.clear()
    monkeypatch.setattr(httpx, 'AsyncClient', lambda *a, **k: real_client(
        *a, **{**k, 'transport': httpx.MockTransport(lambda r: httpx.Response(404) if 'profile' in r.url.path
                                                     else original(r))}))
    report = await tb.run_bench(study, trial, output=tmp_path / 'out2', server_dir=tmp_path,
                                base_url='http://model/v1', log=lambda *a, **k: None)
    assert report['status'] == 'complete_with_errors' and len(report['points']) == 2
    assert 'start_profile 404' in report['profile'][0]['error']


def test_serving_lab_overrides_reach_the_server_command(tmp_path, monkeypatch):
    from arena import serving

    assert serving.lab_overrides({}) == {}
    assert serving.lab_overrides({'ARENA_KV_CACHE_DTYPE': 'fp8', 'ARENA_MTP_TOKENS': '0',
                                  'ARENA_MAX_CUDAGRAPH_CAPTURE_SIZE': '128', 'ARENA_PREFIX_CACHING': '0',
                                  'ARENA_TORCH_PROFILER': '1'}) == {
        'kv_cache_dtype': 'fp8', 'mtp_tokens': 0, 'graph_capture_size': 128, 'prefix_caching': False,
        'torch_profiler': True}
    for bad in ({'ARENA_KV_CACHE_DTYPE': 'fp4'}, {'ARENA_PREFIX_CACHING': 'yes'}, {'ARENA_TORCH_PROFILER': '2'}):
        with pytest.raises(ValueError):
            serving.lab_overrides(bad)
    config = {'generation_slots': 25, 'kv_margin': .9, 'server': {'mtp_tokens': 3, 'max_num_batched_tokens': 4096},
              'cache_experiment': {'cpu_cache_gib': 0, 'kv_cache_dtype': 'auto', 'reclaim_skipped_blocks': True}}
    study = tmp_path / 'study.json'
    study.write_text(json.dumps(config))
    calls = []
    monkeypatch.setattr(serving, 'start', lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(serving.BlockBudget, 'load', lambda *args: __import__('types').SimpleNamespace(layout={}))
    monkeypatch.setattr('sys.argv', ['arena.serving', '--working', str(tmp_path), '--input-root', str(tmp_path),
                                     '--study', str(study)])
    serving.main()
    assert calls[-1]['mtp_tokens'] == 3 and calls[-1]['kv_cache_dtype'] == 'auto' and calls[-1]['prefix_caching'] is None
    monkeypatch.setenv('ARENA_MTP_TOKENS', '0')
    monkeypatch.setenv('ARENA_PREFIX_CACHING', '0')
    monkeypatch.setenv('ARENA_KV_CACHE_DTYPE', 'fp8')
    serving.main()
    assert calls[-1]['mtp_tokens'] == 0 and calls[-1]['prefix_caching'] is False and calls[-1]['kv_cache_dtype'] == 'fp8'
    # The profiler flag lands once on the pinned command, pointing at the bench's trace folder.
    recipe = import_fork(tmp_path, monkeypatch)
    for key, value in {'TAAF_VLLM_MAX_NUM_SEQS': '25', 'TAAF_VLLM_ENABLE_PREFIX_CACHING': '1',
                       'TAAF_VLLM_MAX_NUM_BATCHED_TOKENS': '4096', 'TAAF_VLLM_MTP_TOKENS': '3'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(recipe, 'verify_and_extract_runtime', lambda *a, **k: {})
    recipe.ANALYZER_CONTEXT = 262144
    receipt = serving.configure(recipe, tmp_path, reclaim_skipped_blocks=True, torch_profiler=True)
    argv = recipe.server_command(tmp_path / 'model', tuning=recipe.resolve_vllm_tuning())
    config = json.loads(argv[argv.index('--profiler-config') + 1])
    assert argv.count('--profiler-config') == 1 and config == receipt['torch_profiler']
    assert config['torch_profiler_dir'] == str(tmp_path / tb.PROFILE_DIR) and config['max_iterations'] == 40
    with pytest.raises(ValueError):
        serving.configure(recipe, tmp_path, torch_profiler='1')


def test_builder_makes_the_serving_lab_study(tmp_path):
    spec = importlib.util.spec_from_file_location('bench_builder3', ROOT / 'scripts/build_arena.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build('serving-lab', output=tmp_path / 'lab', paired_30m=True, arena_variant='cache-reclaim',
                                  throughput_bench=True, serving_lab=True)
    names = [t['name'] for t in study['trials']]
    assert names[0] == 'lab-control' and names[1] == 'lab-kv-fp8' and names[-1] == 'lab-control-repeat'
    assert {t['kind'] for t in study['trials']} == {'serving-bench'} and study['serving_ab']['lab'] == 'serving-lab-v1'
    control = study['trials'][0]
    assert control['server_env'] == {'ARENA_TORCH_PROFILER': '1'} and control['throughput_bench']['profile']
    assert tb.grid_points(control['throughput_bench'])[-1] == {'kind': 'distinct', 'prompt_tokens': 65536, 'concurrency': 2}
    assert len(tb.grid_points(control['throughput_bench'])) == 14
    assert study['server']['mtp_tokens'] == 3 and study['cache_experiment']['reclaim_skipped_blocks'] is True
    assert study['notebook_seconds'] == 6600 and all(t['wall_seconds'] <= 900 for t in study['trials'])
    assert json.loads((output / 'configs/comparison.json').read_text()) == study
    with pytest.raises(ValueError, match='serving lab'):
        builder.build('lab-bad', output=tmp_path / 'bad', paired_30m=True, arena_variant='reclaim-mtp1',
                      throughput_bench=True, serving_lab=True)
    with pytest.raises(ValueError, match='--throughput-bench'):
        builder.build('lab-bad2', output=tmp_path / 'bad2', paired_30m=True, arena_variant='cache-reclaim',
                      serving_lab=True)
