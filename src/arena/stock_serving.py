"""Stock vLLM for candidates our Flash fork cannot serve (Nemotron 3.5 Super VL and Bolt, architecture
``nemotron_h_omni``). The runtime is one pinned folder (scripts/modal_stock_vllm.py: a standalone CPython, a venv with a
nightly vLLM wheel and b12x, the downloaded kernels), checked by sha256 and extracted at its build path (the venv names
its base interpreter by path). The serve command is the candidate's own (``stock_args``), started and stopped through the
fork recipe's process ownership (``start_server`` / ``stop_owned_server``), with an environment that carries nothing of
the fork's. Stock vLLM prints no ``ARC_CACHE_LAYOUT``, so ``layout_line`` turns its logged KV token pool into a one-group
token layout for the arena gate (``arena.cache.BlockBudget``)."""
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/tmp/stockvllm')
LAYOUT_LOG = 'stock-cache-layout.log'
STEP = 16  # tokens per row of the gate's cost table
KEEP_ENV = ('PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'TZ', 'TMPDIR', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')


def runtime(input_root, dataset, sha256):
    """The stock runtime's python, extracting ``<dataset>/stockvllm.tar.gz`` once per container after its sha256."""
    from arc_infra.serving import dataset_mount

    python = ROOT / 'venv/bin/python'
    marker = ROOT / f'.extracted-{sha256[:16]}'
    if marker.is_file() and python.exists():
        return python
    tar = Path(dataset_mount(dataset, input_root)) / 'stockvllm.tar.gz'
    with tar.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != sha256:
        raise RuntimeError(f'Stock vLLM runtime hash mismatch: {digest} != {sha256}')
    shutil.rmtree(ROOT, ignore_errors=True)
    subprocess.run(['tar', '-xzf', str(tar), '-C', str(ROOT.parent)], check=True)
    if not python.exists():
        raise RuntimeError(f'Stock vLLM runtime has no {python}')
    marker.write_text(json.dumps({'sha256': sha256, 'tar': str(tar)}))
    return python


def command(python, model_dir, profile, server, host, port):
    """The candidate's serve command on the stock runtime: the study's server sizes, prefix caching, the candidate's
    tool and reasoning parsers (a parser plugin file from the checkpoint when it ships one) and its pinned extras."""
    plugin = []
    if profile.reasoning_plugin:
        plugin = ['--reasoning-parser-plugin', str(Path(model_dir) / profile.reasoning_plugin)]
    return [str(python), '-m', 'vllm.entrypoints.cli.main', 'serve', str(model_dir),
            '--served-model-name', profile.model, '--host', str(host), '--port', str(port),
            '--trust-remote-code', '--max-model-len', str(profile.native_context),
            '--max-num-seqs', str(server['max_num_seqs']),
            '--max-num-batched-tokens', str(server['max_num_batched_tokens']),
            '--gpu-memory-utilization', str(server['gpu_memory_utilization']),
            '--enable-prefix-caching', '--enable-auto-tool-choice', '--tool-call-parser', profile.tool_parser,
            *plugin, '--reasoning-parser', profile.reasoning_parser,
            '--no-enable-log-requests', '--disable-uvicorn-access-log', *profile.extra_args]


def environment(profile, base=None):
    """A clean environment: none of the fork's PYTHONPATH, LD_LIBRARY_PATH or TAAF_/ARENA_ knobs; offline hub; the
    runtime's own caches; the candidate's pinned variables (``stock_env``)."""
    base = os.environ if base is None else base
    env = {k: v for k, v in base.items() if k in KEEP_ENV or k.startswith('NVIDIA_')}
    # VLLM_SERVER_DEV_MODE: /reset_prefix_cache (the bootstrap's cache probe clears its synthetic prefixes with it;
    # stock vLLM serves it only in dev mode: the Super VL run's probe got 404 on 19 Oct)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
               VLLM_SERVER_DEV_MODE='1',
               XDG_CACHE_HOME=str(ROOT / 'cache'), VLLM_CACHE_ROOT=str(ROOT / 'cache/vllm'),
               PYTHONNOUSERSITE='1', **dict(profile.stock_env))
    return env


def layout_line(server_log, max_model_len, step=STEP):
    """``ARC_CACHE_LAYOUT`` for the gate from stock vLLM's ``GPU KV cache size: N tokens`` (the last start's): N/step
    blocks of ``step`` tokens, a request of t tokens costing ceil(t/step). A token pool: the hybrid's per-request
    Mamba state is not in it, so vLLM's own scheduler stays the backstop (preemption) when many short requests run."""
    found = re.findall(r'GPU KV cache size: ([\d,]+) tokens', Path(server_log).read_text(errors='replace'))
    if not found:
        raise RuntimeError('Stock server logged no KV cache size')
    tokens = int(found[-1].replace(',', ''))
    rows = [{'tokens': t, 'blocks': -(-t // step), 'groups': [-(-t // step)]}
            for t in range(step, max_model_len + step, step) if t <= max_model_len]
    if rows[-1]['tokens'] != max_model_len:
        rows.append({'tokens': max_model_len, 'blocks': -(-max_model_len // step),
                     'groups': [-(-max_model_len // step)]})
    layout = {'schema': 1, 'num_blocks': tokens // step, 'max_model_len': max_model_len, 'lookahead': 0,
              'costs': rows, 'source': f'stock vLLM token pool, {tokens} tokens'}
    return 'ARC_CACHE_LAYOUT ' + json.dumps(layout, separators=(',', ':'))
