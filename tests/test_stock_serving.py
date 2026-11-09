"""Stock vLLM candidate path (arena.stock_serving; Nemotron 3.5 screen)."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from arena import stock_serving
from arena.cache import BlockBudget
from arena.candidate_profiles import NEMOTRON35_BOLT, NEMOTRON35_BOLT_FP8, NEMOTRON35_SUPER_VL, PROFILES

SERVER = {'max_num_seqs': 64, 'max_num_batched_tokens': 8192, 'gpu_memory_utilization': '0.95'}


def test_profiles_are_stock_and_reviewed():
    for profile in (NEMOTRON35_SUPER_VL, NEMOTRON35_BOLT, NEMOTRON35_BOLT_FP8):
        assert PROFILES[profile.name] is profile
        assert profile.runtime == 'stock' and profile.stock_dataset and len(profile.stock_sha256) == 64
        assert dict(profile.template_extra) == {'truncate_history_thinking': False}
        assert '{"image":999,"video":0,"audio":0}' in profile.extra_args
        assert profile.manifest.is_file()
        manifest = json.loads(profile.manifest.read_text())
        assert (manifest['repo'], manifest['revision']) == (profile.model, profile.revision)
    assert dict(NEMOTRON35_BOLT.stock_env) == {'VLLM_USE_V2_MODEL_RUNNER': '1'}
    fp8 = dict(zip(NEMOTRON35_BOLT_FP8.extra_args, NEMOTRON35_BOLT_FP8.extra_args[1:]))
    assert fp8['--quantization'] == 'fp8_per_channel' and '--kv-cache-dtype' not in fp8
    assert NEMOTRON35_BOLT_FP8.quant_method is None and not NEMOTRON35_BOLT_FP8.stock_env


def test_command_has_card_flags_and_plugin(tmp_path):
    argv = stock_serving.command(Path('/tmp/stockvllm/venv/bin/python'), tmp_path, NEMOTRON35_BOLT, SERVER,
                                 '127.0.0.1', 1234)
    assert argv[:5] == ['/tmp/stockvllm/venv/bin/python', '-m', 'vllm.entrypoints.cli.main', 'serve', str(tmp_path)]
    pairs = dict(zip(argv, argv[1:]))
    assert pairs['--served-model-name'] == NEMOTRON35_BOLT.model
    assert pairs['--max-model-len'] == '262144' and pairs['--max-num-seqs'] == '64'
    assert pairs['--reasoning-parser-plugin'] == str(tmp_path / 'ultra_v3_reasoning_parser.py')
    assert pairs['--reasoning-parser'] == 'ultra_v3' and pairs['--tool-call-parser'] == 'qwen3_coder'
    assert pairs['--moe-backend'] == 'b12x' and pairs['--kv-cache-dtype'] == 'fp8'
    assert '--enable-prefix-caching' in argv and '--trust-remote-code' in argv
    sv = stock_serving.command('py', tmp_path, NEMOTRON35_SUPER_VL, SERVER, '127.0.0.1', 1234)
    assert '--reasoning-parser-plugin' not in sv and dict(zip(sv, sv[1:]))['--reasoning-parser'] == 'nemotron_v3'


def test_environment_drops_fork_settings():
    base = {'PATH': '/usr/bin', 'PYTHONPATH': '/opt/bundle/src', 'LD_LIBRARY_PATH': '/tmp/runtime/lib',
            'TAAF_VLLM_X': '1', 'ARENA_KV_CACHE_MEMORY_BYTES': '5', 'NVIDIA_VISIBLE_DEVICES': 'all'}
    env = stock_serving.environment(NEMOTRON35_BOLT, base)
    assert env['PATH'] == '/usr/bin' and env['NVIDIA_VISIBLE_DEVICES'] == 'all'
    assert not {'PYTHONPATH', 'LD_LIBRARY_PATH', 'TAAF_VLLM_X', 'ARENA_KV_CACHE_MEMORY_BYTES'} & set(env)
    assert env['HF_HUB_OFFLINE'] == '1' and env['VLLM_USE_V2_MODEL_RUNNER'] == '1'
    assert env['VLLM_SERVER_DEV_MODE'] == '1'  # /reset_prefix_cache for the cache probe


def test_layout_line_feeds_the_gate(tmp_path):
    log = tmp_path / 'vllm-openai-server.log'
    log.write_text('INFO GPU KV cache size: 1,000 tokens\nINFO later start\nINFO GPU KV cache size: 2,048,000 tokens, '
                   'Maximum concurrency for 262,144 tokens per request: 7.81x\n')
    (tmp_path / stock_serving.LAYOUT_LOG).write_text(stock_serving.layout_line(log, 262144) + '\n')
    budget = BlockBudget.load(tmp_path, 0.9)
    assert budget.layout['num_blocks'] == 128000 and budget.capacity == 115200
    assert budget.cost(1) == 1 and budget.cost(16) == 1 and budget.cost(17) == 2 and budget.cost(262144) == 16384
    with pytest.raises(ValueError):
        budget.cost(262145)
    log.write_text('nothing\n')
    with pytest.raises(RuntimeError):
        stock_serving.layout_line(log, 1024)


def test_runtime_checks_hash_and_extracts_once(tmp_path, monkeypatch):
    root = tmp_path / 'stockvllm'
    (root / 'venv/bin').mkdir(parents=True)
    (root / 'venv/bin/python').write_text('#!/bin/sh\n')
    data = tmp_path / 'input/datasets/me/arc-stock'
    data.mkdir(parents=True)
    tar = data / 'stockvllm.tar.gz'
    subprocess.run(['tar', '-czf', str(tar), '-C', str(tmp_path), 'stockvllm'], check=True)
    sha = hashlib.sha256(tar.read_bytes()).hexdigest()
    import shutil
    shutil.rmtree(root)
    monkeypatch.setattr(stock_serving, 'ROOT', root)
    monkeypatch.setattr('arc_infra.serving.dataset_mount', lambda dataset, input_root: data)
    with pytest.raises(RuntimeError, match='hash mismatch'):
        stock_serving.runtime(tmp_path / 'input', 'me/arc-stock', '0' * 64)
    python = stock_serving.runtime(tmp_path / 'input', 'me/arc-stock', sha)
    assert python == root / 'venv/bin/python' and python.is_file()
    tar.unlink()  # a second call in the same container does not read the tar again
    assert stock_serving.runtime(tmp_path / 'input', 'me/arc-stock', sha) == python
