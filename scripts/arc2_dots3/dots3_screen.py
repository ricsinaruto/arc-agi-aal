"""arc2 Dots3-Note self-host screen: dots3-note-prev-fp8 on stock vLLM, 4x H200, in the exact
hard38 fixed-attempt trial (bundle known-v3b base arm, as the API screens), with a pre-registered sequential stop.

  stage      CPU: pinned FP8 checkpoint -> volume arc2-dots3:/dots3-fp8 (sizes checked against the hub listing)
  preflight  CPU (8,16): the GPU image on CPU: vLLM registry / parsers / DeepGEMM / FlashMLA import, config +
             tokenizer + chat template with tools and an image, the stop rule's unit cases, then the same driver
             (screen_driver) end to end against a hosted stand-in through arc2.api_proxy on 3 tasks with a forced
             stop, so start, stop request, drain, report and exit run on the exact code
  screen     4x H200: vLLM serve from the volume, the driver against it, stop rule, exit

usage: SCREEN_BUNDLE=DIR modal run --detach dots3_screen.py::stage_main
       SCREEN_BUNDLE=DIR python scripts/openrouter_cli.py -- modal run --detach dots3_screen.py::preflight_main --run R
       SCREEN_BUNDLE=DIR modal run --detach dots3_screen.py::screen_main --run R"""
import json
import math
import os
import time
from pathlib import Path

import modal

REPO, REVISION = 'dots-studio/dots3-note-prev-fp8', '7c14222e22423d6df6848eb0d1c5c3a88a00311a'
WEIGHTS = '/w/dots3-fp8'
SERVED = 'dots3-note-prev'
VLLM_IMAGE = 'vllm/vllm-openai:v0.32.0'
GPU = 'H200:4'

app = modal.App('arc2-dots3')
BUNDLE = Path(os.environ.get('SCREEN_BUNDLE', '/bundle')).resolve()
vol = modal.Volume.from_name('arc2-dots3', create_if_missing=True, version=2)
out_vol = modal.Volume.from_name('arc2-apiscreen', create_if_missing=True)

cpu_image = (modal.Image.debian_slim(python_version='3.12').pip_install('huggingface_hub>=0.34', 'hf_xet')
             .env({'HF_XET_HIGH_PERFORMANCE': '1'}))
gpu_image = (modal.Image.from_registry(VLLM_IMAGE).entrypoint([])
             .run_commands('ln -sf $(which python3) /usr/local/bin/python || true')
             .pip_install('httpx==0.28.1', 'scipy'))
if modal.is_local():
    gpu_image = gpu_image.add_local_dir(str(BUNDLE), '/bundle', ignore=['wheels', 'draft-head', '**/__pycache__'])
    gpu_image = gpu_image.add_local_file(str(Path(__file__).resolve().parent / 'fixedread.py'),
                                         '/fr/fixedread.py')
    gpu_image = gpu_image.add_local_file('./.claude/worktrees/arc2/dist/arc2-apiscreen-v1/src/arc2/'
                                         'api_proxy.py', '/fr/api_proxy.py')  # stand-in server for older bundles
SECRET = modal.Secret.from_dict({'OPENROUTER_API_KEY': os.environ.get('OPENROUTER_API_KEY', '')})

# --- pre-registered sequential stop (arm log 23 Oct): Wald SPRT on right2 per entry, p0 0.20 vs p1 0.35,
# alpha = beta = 0.05, on the completed prefix of the copy-major launch order, from 38 entries (copy 1: each task
# once) on; 114 entries = all, then the API screen's bars on the point estimate.
P0, P1, BOUND, MIN_N = 0.20, 0.35, math.log(19), 38


def sprt(k, n):
    """'pass' / 'miss' / None for k right of the first n entries."""
    if n < MIN_N:
        return None
    llr = k * math.log(P1 / P0) + (n - k) * math.log((1 - P1) / (1 - P0))
    return 'pass' if llr >= BOUND else 'miss' if llr <= -BOUND else None


def prefix(order, done):
    """Right2 flags of the longest finished prefix of ``order`` (``done``: entry -> bool)."""
    flags = []
    for name in order:
        if name not in done:
            break
        flags.append(done[name])
    return flags


def right2(row, task):
    from arc2.loop import guesses
    truth = [t['output'] for t in task['test']]
    picks = guesses(row['candidates'], [t['input'] for t in task['test']])
    return all(any(g == t for g, _ in picks[k]) for k, t in enumerate(truth))


def trial_for(slots, wall, limit=0, images=True, overrides=None):
    study = json.loads(Path('/bundle/configs/comparison.json').read_text())
    study['generation_slots'] = study['active_limit'] = int(slots)
    trial = dict(study['trials'][0])
    base = next((a for a in trial['arms'] if a['name'] == 'base'), trial['arms'][0])
    o = dict(overrides or {})
    trial['arms'] = [o['replace_arm']] if o.get('replace_arm') else [{**base, 'images': bool(images), **o.get('arm', {})}]
    study.update(o.get('study', {}))
    study['arc2'] = {**study['arc2'], **o.get('arc2', {})}
    trial.update(o.get('trial', {}))
    trial['wall_seconds'] = int(wall)
    trial.pop('pool_seed', None)
    trial['paired'] = False
    trial['copy_major'] = True
    if limit:
        trial['tasks'] = trial['tasks'][:int(limit)]
    study['trials'] = [trial]
    return study, trial


def patch_top_k(value):
    """The harness sends Flash's top_k 20; a model whose card sets none gets ``value`` (-1: off in vLLM)."""
    from arc2 import tasks as arc_tasks
    original = arc_tasks.chat_request

    def chat_request(*a, **k):
        body = original(*a, **k)
        body['top_k'] = value
        return body
    arc_tasks.chat_request = chat_request
    assert arc_tasks.chat_request('m', [], 8, temperature=1.0, effort='medium')['top_k'] == value


def finished_rows(path):
    """Finished entries of a tasks.jsonl (cut or failed entries are not finished)."""
    rows = {}
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            if line.startswith('{'):
                r = json.loads(line)
                if r.get('ended') not in (None, 'failed', 'budget'):
                    rows[r['task'].split('@')[0]] = r
    return rows


def screen_driver(out, base_url, layout_line, slots, wall, limit=0, images=True, force_stop_after=0, log=print,
                  overrides=None, prior=None):
    """run_arc2 on hard38 with the stop rule in a side thread; returns the report (+ the rule's state)."""
    import asyncio
    import sys
    import threading

    sys.path[:0] = ['/bundle/src']
    from arc2.trial import run_arc2
    from arena.early_stop import stop_paths

    out.mkdir(parents=True, exist_ok=True)
    server = out / 'server'
    server.mkdir(parents=True, exist_ok=True)
    (server / 'server.log').write_text(layout_line + '\n')
    prior = dict(prior or {})  # entry -> row from an earlier leg of the same screen (continued after its wall)
    if prior:
        overrides = {**(overrides or {}), 'trial': {**(overrides or {}).get('trial', {}), 'skip': sorted(prior)}}
    study, trial = trial_for(slots, wall, limit, images, overrides)
    data = Path('/bundle') / study['arc2']['data'] / trial['split']
    tasks = {t: json.loads((data / f'{t}.json').read_text()) for t in trial['tasks']}
    copies = int(trial.get('copies', 1))
    order = [f'{t}#{k + 1}' for k in range(copies) for t in trial['tasks']]
    state = {'decision': None, 'n': 0, 'k': 0, 'checked': 0}
    stop_file = stop_paths(out, {})[0]
    halt = threading.Event()

    def rule():
        while not halt.wait(20):
            rows = {**prior, **finished_rows(out / 'tasks.jsonl')}
            done = {n: right2(r, tasks[n.split('#')[0]]) for n, r in rows.items()}
            flags = prefix(order, done)
            state.update(n=len(flags), k=sum(flags), finished=len(done), checked=state['checked'] + 1)
            decision = sprt(sum(flags), len(flags))
            if force_stop_after and len(done) >= force_stop_after:
                decision = decision or 'forced'
            if decision and not state['decision']:
                state.update(decision=decision, at=time.time())
                stop_file.write_text(json.dumps(state))
                log(f'stop rule: {decision} at k={state["k"]} n={state["n"]}')
            (out / 'stop-rule.json').write_text(json.dumps(state))

    if (overrides or {}).get('stop_rule', True):
        threading.Thread(target=rule, daemon=True).start()
    began = time.time()
    try:
        report = asyncio.run(run_arc2(study, trial, source='/bundle', output=out, server_dir=server, base_url=base_url,
                                      deadline_epoch=time.time() + wall + 900))
    except Exception as error:  # noqa: BLE001 - recorded
        report = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'[:2000]}
    halt.set()
    report = {**(report if isinstance(report, dict) else {'report': report}), 'seconds': round(time.time() - began, 1),
              'stop_rule': state}
    (out / 'screen-report.json').write_text(json.dumps(report, indent=1, default=str))
    return report


def committer(volume, stop):
    import threading

    def loop():
        while not stop.wait(60):
            try:
                volume.commit()
            except Exception:  # noqa: BLE001 - next minute
                pass
    threading.Thread(target=loop, daemon=True).start()


def serve_args(port, slots, max_len=131072, model=WEIGHTS):
    return ['vllm', 'serve', model, '--served-model-name', SERVED, '--host', '127.0.0.1', '--port', str(port),
            '--tensor-parallel-size', '4', '--enable-expert-parallel',
            '--max-model-len', str(max_len), '--max-num-seqs', str(slots), '--max-num-batched-tokens', '8192',
            '--block-size', '64', '--gpu-memory-utilization', '0.92', '--enable-prefix-caching',
            '--enable-auto-tool-choice', '--tool-call-parser', 'dots', '--reasoning-parser', '{reasoning}',
            '--mm-processor-cache-gb', '4', '--limit-mm-per-prompt', '{"image": 16, "audio": 0, "video": 0}',
            '--trust-remote-code']


# --- stage -----------------------------------------------------------------------------------------------------------

@app.function(image=cpu_image, timeout=3 * 3600, cpu=(0.125, 8), volumes={'/w': vol})
def stage():
    from huggingface_hub import HfApi, snapshot_download
    began = time.time()
    snapshot_download(REPO, revision=REVISION, local_dir=WEIGHTS, max_workers=16, allow_patterns=[
        '*.safetensors', '*.json', '*.jinja', '*.txt', '*.py', 'LICENSE*', 'README.md'])
    listing = {f.path: f.size for f in HfApi().list_repo_tree(REPO, revision=REVISION, recursive=True)
               if hasattr(f, 'size') and not f.path.startswith('assets/')}
    bad = {p: (s, (Path(WEIGHTS) / p).stat().st_size if (Path(WEIGHTS) / p).is_file() else None)
           for p, s in listing.items() if not (Path(WEIGHTS) / p).is_file() or (Path(WEIGHTS) / p).stat().st_size != s}
    result = {'repo': REPO, 'revision': REVISION, 'files': len(listing), 'bytes': sum(listing.values()),
              'bad': bad, 'ok': not bad, 'seconds': round(time.time() - began, 1)}
    (Path(WEIGHTS) / 'staged.json').write_text(json.dumps(result, indent=1))
    vol.commit()
    return result


@app.local_entrypoint()
def stage_main():
    call = stage.spawn()
    print(json.dumps({'call': call.object_id}))


# --- preflight -------------------------------------------------------------------------------------------------------

def checks(log):
    """CPU checks of the serving stack: registry, parsers, kernels import, config/tokenizer/template."""
    import importlib
    result = {}
    import vllm
    result['vllm'] = vllm.__version__
    try:
        from vllm.model_executor.models import ModelRegistry
        result['arch'] = 'Dots3NoteForCausalLM' in ModelRegistry.get_supported_archs()
        result['arch_like'] = [a for a in ModelRegistry.get_supported_archs() if 'dots' in a.lower()]
    except Exception as error:  # noqa: BLE001
        result['arch'] = f'{type(error).__name__}: {error}'[:200]
    def parsers(mods, attr):
        for mod in mods:
            try:
                m = importlib.import_module(mod)
                mgr = getattr(m, attr)
                return sorted(set(getattr(mgr, 'reasoning_parsers', {}) or {}) | set(getattr(mgr, 'tool_parsers', {}) or {})
                              | set(getattr(mgr, 'lazy_parsers', {}) or {}))
            except Exception as error:  # noqa: BLE001
                last = f'{mod}: {type(error).__name__}: {error}'[:200]
        return [last]
    rnames = parsers(['vllm.reasoning', 'vllm.reasoning.abs_reasoning_parsers'], 'ReasoningParserManager')
    result['reasoning_parsers'] = rnames
    tnames = parsers(['vllm.tool_parsers', 'vllm.entrypoints.openai.tool_parsers',
                      'vllm.entrypoints.openai.tool_parsers.abstract_tool_parser'], 'ToolParserManager')
    result['tool_parsers'] = tnames
    result['tool_parser_dots'] = 'dots' in tnames
    import subprocess
    helptext = subprocess.run(['vllm', 'serve', '--help=all'], capture_output=True, text=True, timeout=300).stdout
    for flag in ('--tool-call-parser', '--reasoning-parser', '--moe-backend', '--language-model-only'):
        i = helptext.find(flag)
        result[f'help{flag}'] = helptext[i:i + 700] if i >= 0 else None
    for mod in ('deep_gemm', 'flash_mla', 'vllm.third_party.flashmla', 'vllm.attention.ops.flashmla'):
        try:
            importlib.import_module(mod)
            result[f'import:{mod}'] = True
        except Exception as error:  # noqa: BLE001
            result[f'import:{mod}'] = f'{type(error).__name__}: {error}'[:200]
    if Path(WEIGHTS, 'staged.json').is_file():
      try:
        result['staged'] = json.loads(Path(WEIGHTS, 'staged.json').read_text())
        result['staged'].pop('bad', None)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
        tools = [{'type': 'function', 'function': {'name': 'python', 'description': 'run code',
                                                   'parameters': {'type': 'object', 'properties': {
                                                       'code': {'type': 'string'}}, 'required': ['code']}}}]
        msgs = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
                                             {'type': 'text', 'text': 'hi'}]},
                {'role': 'assistant', 'content': 'x', 'reasoning_content': 'think',
                 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'python',
                                                                          'arguments': '{"code": "1"}'}}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': '1'}]
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                       reasoning_effort='high')
        result['template'] = {'chars': len(text), 'has_tool': 'python' in text, 'tail': text[-160:]}
        from vllm.config import ModelConfig
        try:
            cfg = ModelConfig(model=WEIGHTS, trust_remote_code=True, max_model_len=131072)
            result['model_config'] = {'arch': cfg.architectures, 'mm': cfg.is_multimodal_model}
        except Exception as error:  # noqa: BLE001
            result['model_config'] = f'{type(error).__name__}: {error}'[:400]
      except Exception as error:  # noqa: BLE001
        result['template'] = f'{type(error).__name__}: {error}'[:400]
    # the rule's unit cases
    assert sprt(15, 38) == 'pass' and sprt(14, 38) is None and sprt(6, 38) == 'miss' and sprt(7, 38) is None
    assert sprt(30, 37) is None and sprt(25, 76) == 'pass' and sprt(16, 76) == 'miss'
    assert prefix(['a', 'b', 'c'], {'a': True, 'c': False}) == [True]
    result['rule_cases'] = True
    log(json.dumps(result, default=str)[:3000])
    return result


@app.function(image=gpu_image, timeout=3600, cpu=(8, 16), volumes={'/w': vol, '/out': out_vol}, secrets=[SECRET])
def preflight(run, target='z-ai/glm-5.3-flash'):
    import socket
    import sys
    import threading
    import urllib.request

    sys.path[:0] = ['/bundle/src']
    out = Path('/out') / run / 'dots3-pf'
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / 'preflight.log').open('a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()
    stop = threading.Event()
    committer(out_vol, stop)
    result = {'checks': checks(log)}
    from arc2 import api_proxy
    from arena.stock_serving import layout_line
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    threading.Thread(target=api_proxy.serve_forever, args=(port, target, os.environ['OPENROUTER_API_KEY']),
                     kwargs={'log_path': str(out / 'proxy.jsonl')}, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    log_file = out / 'stock.log'
    log_file.write_text('GPU KV cache size: 4,000,000 tokens\n')
    layout = layout_line(log_file, 131072)
    report = screen_driver(out, f'http://127.0.0.1:{port}/v1', layout, slots=4, wall=1500, limit=3, images=True,
                           force_stop_after=2, log=log)
    result['driver'] = {k: report.get(k) for k in ('status', 'seconds', 'stop_rule', 'error')}
    result['serve_args'] = serve_args(8000, 48)
    result['ok'] = bool(result['checks'].get('arch') is True and result['checks'].get('tool_parser_dots')
                        and report.get('stop_rule', {}).get('decision') == 'forced' and report.get('status') != 'failed')
    (out / 'preflight-report.json').write_text(json.dumps(result, indent=1, default=str))
    stop.set()
    out_vol.commit()
    return result


@app.local_entrypoint()
def preflight_main(run: str):
    call = preflight.spawn(run)
    print(json.dumps({'call': call.object_id}))


# --- screen ----------------------------------------------------------------------------------------------------------

@app.function(image=gpu_image, gpu=GPU, timeout=4 * 3600, cpu=(0.125, 16), volumes={'/w': vol, '/out': out_vol})
def screen(run, slots=48, wall=7200, reasoning='', limit=0, name='dots3', prior_legs='', overrides_json='',
           max_len=131072, top_k=0):
    import subprocess
    import sys
    import threading
    import urllib.request

    sys.path[:0] = ['/bundle/src']
    from arena.stock_serving import layout_line
    out = Path('/out') / run / name
    out.mkdir(parents=True, exist_ok=True)
    prior = {}
    for leg in [x for x in prior_legs.split(',') if x]:
        prior.update(finished_rows(Path('/out') / run / leg / 'tasks.jsonl'))
    logf = (out / 'screen.log').open('a')

    def log(msg):
        print(msg, flush=True)
        logf.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
        logf.flush()
    stop = threading.Event()
    committer(out_vol, stop)
    staged = json.loads(Path(WEIGHTS, 'staged.json').read_text())
    assert staged['ok'], staged
    reasoning = reasoning or json.loads(os.environ.get('DOTS3_REASONING', '""')) or 'dots3'
    args = [a.replace('{reasoning}', reasoning) for a in serve_args(8000, slots, max_len)]
    overrides = json.loads(overrides_json) if overrides_json else None
    if top_k:
        patch_top_k(top_k)
    env = {**os.environ, 'VLLM_CACHE_ROOT': '/w/vllm-cache', 'HF_HUB_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
           'DG_JIT_CACHE_DIR': '/w/deepgemm-cache'}
    server_log = out / 'vllm.log'
    began = time.time()
    proc = subprocess.Popen(args, stdout=server_log.open('w'), stderr=subprocess.STDOUT, env=env)
    log(f'vllm started: {" ".join(args)}')
    while True:
        if proc.poll() is not None:
            log(f'vllm exited {proc.returncode} before ready')
            stop.set()
            out_vol.commit()
            return {'status': 'server_failed', 'code': proc.returncode}
        try:
            urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)
            break
        except OSError:
            time.sleep(5)
    log(f'vllm ready after {time.time() - began:.0f} s')
    try:
        vol.commit()  # the DeepGEMM JIT / compile caches for the next leg's boot
    except Exception:  # noqa: BLE001
        pass

    def gpu_sampler():
        rows = (out / 'gpu.jsonl').open('a')
        while not stop.wait(30):
            q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True).stdout.strip().splitlines()
            rows.write(json.dumps({'t': time.time(), 'gpu': [list(map(float, x.split(','))) for x in q]}) + '\n')
            rows.flush()
    threading.Thread(target=gpu_sampler, daemon=True).start()
    log(f'prior finished entries: {len(prior)}')
    report = screen_driver(out, 'http://127.0.0.1:8000/v1', layout_line(server_log, max_len), slots, wall, limit,
                           images=True, log=log, prior=prior, overrides=overrides)
    log(f'driver done: {report.get("status")} {json.dumps(report.get("stop_rule"))}')
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
    stop.set()
    out_vol.commit()
    return {'status': report.get('status'), 'stop_rule': report.get('stop_rule'), 'seconds': time.time() - began}


@app.local_entrypoint()
def screen_main(run: str, slots: int = 48, reasoning: str = '', wall: int = 7200, name: str = 'dots3',
                prior_legs: str = '', overrides: str = '', max_len: int = 131072, top_k: int = 0):
    call = screen.spawn(run, slots, wall, reasoning, 0, name, prior_legs, overrides, max_len, top_k)
    print(json.dumps({'call': call.object_id}))


@app.function(image=gpu_image, timeout=900, cpu=(0.125, 8))
def probe():
    """Where the DSA / DeepGEMM kernels live in this image, and the serve flags (preflight part 2)."""
    import importlib.util
    import pkgutil
    import subprocess
    import sys
    r = {}
    import vllm.third_party as tp
    r['third_party'] = [m.name for m in pkgutil.iter_modules(tp.__path__)]
    r['dist'] = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True).stdout
    r['dist'] = [x for x in r['dist'].splitlines() if any(k in x.lower() for k in ('gemm', 'mla', 'flash', 'torch', 'triton', 'vllm', 'flashinfer'))]
    for name in ('vllm.utils.deep_gemm', 'vllm.model_executor.layers.quantization.utils.fp8_utils'):
        r[name] = bool(importlib.util.find_spec(name))
    try:
        from vllm.utils import deep_gemm as dg
        r['has_deep_gemm'] = dg.has_deep_gemm() if hasattr(dg, 'has_deep_gemm') else [a for a in dir(dg) if 'deep' in a.lower()]
    except Exception as error:  # noqa: BLE001
        r['has_deep_gemm'] = f'{type(error).__name__}: {error}'[:300]
    h = subprocess.run(['vllm', 'serve', '--help=all'], capture_output=True, text=True, timeout=300)
    text = h.stdout + h.stderr
    r['help_len'] = len(text)
    for flag in ('--moe-backend', '--reasoning-parser ', '--language-model-only', '--limit-mm-per-prompt',
                 '--mm-processor-cache-gb', '--kv-cache-dtype'):
        i = text.find(flag)
        r[flag] = text[i:i + 600] if i >= 0 else None
    print(json.dumps(r, indent=1))
    return r


@app.function(image=gpu_image, timeout=900, cpu=(0.125, 8))
def probe_args():
    """Every serve flag maps to an EngineArgs / frontend field (CPU cannot build vLLM's parser: no device)."""
    import dataclasses
    import subprocess
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    fields = {f.name for f in dataclasses.fields(EngineArgs)}
    root = str(Path(vllm.__file__).parent)
    src = subprocess.run(['grep', '-rhoE', r'"--[a-z0-9-]+"|--[a-z0-9-]+', root + '/entrypoints'], capture_output=True,
                         text=True).stdout
    flags = [a for a in serve_args(8000, 48)[2:] if a.startswith('--')]
    r = {}
    for flag in flags:
        name = flag[2:].replace('-', '_')
        r[flag] = 'engine' if name in fields else ('frontend' if flag in src else 'MISSING')
    moe = subprocess.run(['grep', '-rn', 'deep_gemm', root + '/config/kernel.py', root + '/engine/arg_utils.py'],
                         capture_output=True, text=True).stdout
    r['moe_backend_choices'] = [x[:200] for x in moe.splitlines() if 'moe' in x.lower()][:8]
    print(json.dumps(r, indent=1))
    return r


# --- hosted model under the same driver (11:5x: MiniMax at medium with its own settings) ----------------

API_IMAGE = (modal.Image.debian_slim(python_version='3.12').pip_install('numpy==2.2.6', 'scipy==1.15.3', 'httpx==0.28.1'))
if modal.is_local():
    API_IMAGE = API_IMAGE.add_local_dir(str(BUNDLE), '/bundle', ignore=['wheels', 'draft-head', '**/__pycache__'])
    API_IMAGE = API_IMAGE.add_local_file(__file__, '/root/dots3_screen.py')


def repair_args(raw):
    """A tool call's ``arguments`` as valid JSON (MiniMax v37 run: 2 tasks lost when the provider rejected a history
    holding an earlier call with malformed arguments). Valid JSON passes unchanged; JSON with raw control characters
    is re-encoded; a truncated object is closed; anything else becomes ``{"code": raw}``. Returns (arguments, how)."""
    if not isinstance(raw, str):
        return json.dumps(raw or {}), 'object'
    try:
        json.loads(raw)
        return raw, None
    except ValueError:
        pass
    try:
        return json.dumps(json.loads(raw, strict=False)), 'control-chars'
    except ValueError:
        pass
    text = raw.strip()
    if text.startswith('{'):
        for tail in ('"}', '}', '\\"}', '"]}'):
            try:
                value = json.loads(text + tail, strict=False)
                if isinstance(value, dict):
                    return json.dumps(value), 'closed'
            except ValueError:
                continue
    return json.dumps({'code': raw}), 'wrapped'


def apply_overrides(out_body, overrides, note):
    """Screen overrides on the upstream request: tool-call repair, one pinned provider, and ``api_effort`` (sent
    as reasoning.effort as is, e.g. "minimal", which the proxy's effort table would map to "high")."""
    if overrides.get('repair_tool_args', True):
        out_body['messages'] = repair_history(out_body['messages'], note)
    if overrides.get('provider'):  # one provider for every request (effort and reasoning accounting differ by provider)
        out_body['provider'] = {'order': [overrides['provider']], 'allow_fallbacks': False}
        if overrides.get('drop_top_k', True):
            out_body.pop('top_k', None)
    if overrides.get('api_effort'):
        out_body['reasoning'] = {'effort': overrides['api_effort']}
    return out_body


def repair_history(messages, log=None):
    """Every assistant tool call in the history with valid JSON arguments (the content the model sees is unchanged
    otherwise). ``log``: called with (how, raw) for each repair."""
    out = []
    for m in messages:
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            calls = []
            for c in m['tool_calls']:
                f = dict(c.get('function') or {})
                fixed, how = repair_args(f.get('arguments'))
                if how and log:
                    log(how, f.get('arguments'))
                f['arguments'] = fixed
                calls.append({**c, 'function': f})
            m = {**m, 'tool_calls': calls}
        out.append(m)
    return out


@app.function(image=API_IMAGE, timeout=4 * 3600, cpu=(0.125, 8), volumes={'/out': out_vol}, secrets=[SECRET])
def api_screen(run, name, target, overrides_json, slots=32, wall=7200, limit=0, images=True):
    import socket
    import sys
    import threading
    import urllib.request
    sys.path[:0] = ['/bundle/src']
    from arc2 import api_proxy
    from arena.stock_serving import layout_line
    out = Path('/out') / run / name
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    committer(out_vol, stop)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    threading.Thread(target=api_proxy.serve_forever, args=(port, target, os.environ['OPENROUTER_API_KEY']),
                     kwargs={'log_path': str(out / 'proxy.jsonl')}, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    pool = out / 'pool.log'  # a hosted model has no KV pool: a large token pool, so the gate is the slot count
    pool.write_text('GPU KV cache size: 50,000,000 tokens\n')
    overrides = json.loads(overrides_json)
    repairs = (out / 'repairs.jsonl').open('a')
    lock = threading.Lock()

    def note(how, raw):
        with lock:
            repairs.write(json.dumps({'t': time.time(), 'how': how, 'raw': str(raw)[:4000]}) + '\n')
            repairs.flush()
    original = api_proxy.upstream_body

    def upstream_body(body, target, keep_reasoning=True):
        return apply_overrides(original(body, target, keep_reasoning), overrides, note)
    api_proxy.upstream_body = upstream_body
    report = screen_driver(out, f'http://127.0.0.1:{port}/v1', layout_line(pool, 262144), slots, wall, limit,
                           images=images, overrides=overrides)
    report['overrides'] = overrides
    (out / 'screen-report.json').write_text(json.dumps(report, indent=1, default=str))
    stop.set()
    out_vol.commit()
    return {'status': report.get('status'), 'stop_rule': report.get('stop_rule')}


@app.local_entrypoint()
def api_main(run: str, name: str, target: str, overrides: str, slots: int = 32, limit: int = 0, wall: int = 7200):
    call = api_screen.spawn(run, name, target, overrides, slots, wall, limit)
    print(json.dumps({'call': call.object_id}))


@app.function(image=gpu_image, timeout=3600, cpu=(8, 16), volumes={'/out': out_vol}, secrets=[SECRET])
def preflight_leg(run, target='z-ai/glm-5.3-flash'):
    """CPU preflight of a continued leg: the first preflight's finished rows as the prior, skip + merged rule, forced
    stop; the driver code of screen() with prior rows."""
    import socket
    import sys
    import threading
    import urllib.request
    sys.path[:0] = ['/bundle/src']
    from arc2 import api_proxy
    from arena.stock_serving import layout_line
    out = Path('/out') / run / 'leg2-pf'
    out.mkdir(parents=True, exist_ok=True)
    prior = finished_rows(Path('/out') / 'arc2-dots3-pf-v1' / 'dots3-pf' / 'tasks.jsonl')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    threading.Thread(target=api_proxy.serve_forever, args=(port, target, os.environ['OPENROUTER_API_KEY']),
                     kwargs={'log_path': str(out / 'proxy.jsonl')}, daemon=True).start()
    time.sleep(1)
    pool = out / 'pool.log'
    pool.write_text('GPU KV cache size: 4,000,000 tokens\n')
    report = screen_driver(out, f'http://127.0.0.1:{port}/v1', layout_line(pool, 131072), 4, 1500, limit=3,
                           force_stop_after=len(prior) + 1, prior=prior)
    played = {r for r in finished_rows(out / 'tasks.jsonl')}
    res = {'prior': sorted(prior), 'played_overlap': sorted(set(prior) & played), 'status': report.get('status'),
           'stop_rule': report.get('stop_rule')}
    res['ok'] = bool(prior) and not res['played_overlap'] and report.get('stop_rule', {}).get('finished', 0) > len(prior) \
        and report.get('status') != 'failed'
    (out / 'preflight-report.json').write_text(json.dumps(res, indent=1, default=str))
    out_vol.commit()
    print(json.dumps(res))
    return res


@app.local_entrypoint()
def preflight_leg_main(run: str):
    call = preflight_leg.spawn(run)
    print(json.dumps({'call': call.object_id}))


@app.function(image=gpu_image, timeout=3600, cpu=(8, 16), volumes={'/out': out_vol}, secrets=[SECRET])
def preflight_adapted(run, overrides_json, top_k=-1, target='z-ai/glm-5.3-flash'):
    """CPU preflight of the adapted Dots3 screen: overrides (cap, window, temperature) reach the trial, top_k patched,
    serve args at the larger max length parse, driver end to end with a forced stop."""
    import socket
    import sys
    import threading
    sys.path[:0] = ['/bundle/src']
    from arc2 import api_proxy
    from arena.stock_serving import layout_line
    out = Path('/out') / run / 'adapted-pf'
    out.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(overrides_json)
    patch_top_k(top_k)
    study, trial = trial_for(48, 7200, 0, True, overrides)
    res = {'arm': trial['arms'][0], 'temperature': study['temperature'], 'windows': study['windows'],
           'serve_args': serve_args(8000, 48, 262144)}
    arm = overrides.get('replace_arm') or overrides['arm']
    assert trial['arms'][0]['cap'] == arm['cap'] and study['windows'] == overrides['study']['windows']
    ruled = overrides.get('stop_rule', True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    threading.Thread(target=api_proxy.serve_forever, args=(port, target, os.environ['OPENROUTER_API_KEY']),
                     kwargs={'log_path': str(out / 'proxy.jsonl')}, daemon=True).start()
    time.sleep(1)
    pool = out / 'pool.log'
    pool.write_text('GPU KV cache size: 1,131,347 tokens\n')
    report = screen_driver(out, f'http://127.0.0.1:{port}/v1', layout_line(pool, 262144), 4, 1500, limit=2,
                           force_stop_after=1 if ruled else 0, overrides=overrides)
    tf = out / 'tasks.jsonl'
    rows = [json.loads(x) for x in tf.read_text().splitlines() if x.strip()] if tf.exists() else []
    res.update(status=report.get('status'), stop_rule=report.get('stop_rule'), error=report.get('error'),
               finished=len(rows), seconds=report.get('seconds'))
    res['ok'] = report.get('status') != 'failed' and (
        (report.get('stop_rule') or {}).get('decision') == 'forced' if ruled else len(rows) >= 1)
    (out / 'preflight-report.json').write_text(json.dumps(res, indent=1, default=str))
    out_vol.commit()
    print(json.dumps(res, default=str))
    return res


@app.local_entrypoint()
def preflight_adapted_main(run: str, overrides: str, top_k: int = -1):
    call = preflight_adapted.spawn(run, overrides, top_k)
    print(json.dumps({'call': call.object_id}))


@app.function(image=gpu_image, timeout=1800, cpu=(8, 16), volumes={'/w': vol})
def bootprobe(files=6, threads=16):
    """Boot investigation (23 Oct: weights load 393 s in v1, 1,998 s in v2): vLLM's DeepGEMM warmup knobs in this
    image, and volume read throughput for the weight shards with 1 reader vs ``threads`` readers."""
    import concurrent.futures as cf
    import re
    import vllm
    src = (Path(vllm.__file__).parent / 'envs.py').read_text()
    knobs = sorted(set(re.findall(r'"(VLLM_[A-Z_]*(?:DEEP_GEMM|DEEPGEMM|WARMUP)[A-Z_]*)"', src)))
    shards = sorted(Path(WEIGHTS).glob('*.safetensors'))
    out = {'knobs': knobs, 'shards': len(shards), 'total_gb': round(sum(p.stat().st_size for p in shards) / 1e9, 1)}

    def read(p, chunk=64 << 20):
        n = 0
        with open(p, 'rb', buffering=0) as f:
            while True:
                b = f.read(chunk)
                if not b:
                    return n
                n += len(b)
    for name, group, workers in (('serial', shards[:files], 1), ('parallel', shards[files:3 * files], threads)):
        t = time.time()
        with cf.ThreadPoolExecutor(workers) as ex:
            n = sum(ex.map(read, group))
        out[name] = {'files': len(group), 'gb': round(n / 1e9, 2), 'seconds': round(time.time() - t, 1),
                     'gb_per_s': round(n / 1e9 / (time.time() - t), 3)}
    print(json.dumps(out))
    return out


# --- fixed-attempt DP2 screen (Dots3 vs Flash, paired per task) ---------------------------
GPU8 = 'H200:8'
LOCAL = '/local/dots3-fp8'


def prefetch(src, dst, threads=16, log=print):
    """Copy the weight directory to local disk with ``threads`` parallel readers (bootprobe 23 Oct: the volume
    reads 0.67 GB/s with one reader, 1.28 GB/s with 16; v2's vLLM load ran at 0.15 GB/s for 33 min). Both servers
    then load from local disk. Returns (bytes, seconds); sizes checked."""
    import concurrent.futures as cf
    import shutil
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = [f for f in src.iterdir() if f.is_file()]
    began = time.time()

    def copy(f):
        target = dst / f.name
        with open(f, 'rb', buffering=0) as a, open(target, 'wb') as b:
            shutil.copyfileobj(a, b, 64 << 20)
        assert target.stat().st_size == f.stat().st_size, f.name
        return f.stat().st_size
    with cf.ThreadPoolExecutor(threads) as ex:
        n = sum(ex.map(copy, sorted(files, key=lambda f: -f.stat().st_size)))
    log(f'prefetch {len(files)} files {n / 1e9:.1f} GB in {time.time() - began:.0f} s')
    return n, time.time() - began


def dp2_env(gpus):
    """Environment of one TP4 server: its four GPUs, caches, and no DeepGEMM warmup (8.7-9.7 min a boot in v1/v2
    although the kernel cache was on the volume; kernels compile at first use from the cache instead)."""
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': gpus, 'VLLM_CACHE_ROOT': '/w/vllm-cache', 'HF_HUB_OFFLINE': '1',
            'VLLM_NO_USAGE_STATS': '1', 'DG_JIT_CACHE_DIR': '/w/deepgemm-cache', 'VLLM_DEEP_GEMM_WARMUP': 'skip'}


def split_overrides(overrides, half, limit=0):
    """The trial's hard38 tasks for one of the two drivers (alternating, so both get easy and hard tasks)."""
    study = json.loads(Path('/bundle/configs/comparison.json').read_text())
    tasks = study['trials'][0]['tasks'][half::2]
    tasks = tasks[:limit] if limit else tasks
    o = json.loads(json.dumps(overrides))
    o.setdefault('trial', {})['tasks'] = tasks
    return o


def merge_rows(out, halves=('a', 'b')):
    """One tasks.jsonl from the two drivers' (entries are disjoint by construction)."""
    rows, seen = [], set()
    for h in halves:
        f = Path(out) / h / 'tasks.jsonl'
        for line in (f.read_text().splitlines() if f.is_file() else []):
            if line.startswith('{'):
                r = json.loads(line)
                assert r['task'] not in seen, r['task']
                seen.add(r['task'])
                rows.append(r)
    (Path(out) / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return len(rows)


def run_drivers(out, urls, layouts, slots, wall, overrides, log=print, limit=0):
    """Two screen drivers in threads, one a server, on alternating halves of the trial's tasks."""
    import threading
    reports = {}

    def drive(i, half):
        reports[half] = screen_driver(Path(out) / half, urls[i], layouts[i], slots, wall, 0, images=True, log=log,
                                      overrides=split_overrides(overrides, i, limit))
    threads = [threading.Thread(target=drive, args=(i, h)) for i, h in enumerate(('a', 'b'))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return reports


@app.function(image=gpu_image, gpu=GPU8, timeout=4 * 3600, cpu=(0.125, 32), ephemeral_disk=600 * 1024,
              volumes={'/w': vol, '/out': out_vol})
def screen_dp2(run, overrides_json, slots=48, wall=7200, reasoning='deepseek_r1', name='dots3', max_len=262144,
               top_k=-1, threads=16):
    """Fixed-attempt Dots3 screen on 8x H200: weights prefetched to local disk, two TP4 servers (GPUs 0-3, 4-7),
    two screen drivers on alternating halves of hard38, rows merged. ``overrides_json``: the trial (copies, the
    replace_arm with max_attempts 1, arc2 reply_reserve)."""
    import subprocess
    import sys
    import threading
    import urllib.request
    sys.path[:0] = ['/bundle/src']
    from arena.stock_serving import layout_line
    out = Path('/out') / run / name
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / 'screen.log').open('a')

    def log(msg):
        print(msg, flush=True)
        logf.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
        logf.flush()
    stop = threading.Event()
    committer(out_vol, stop)
    staged = json.loads(Path(WEIGHTS, 'staged.json').read_text())
    assert staged['ok'], staged
    overrides = json.loads(overrides_json)
    patch_top_k(top_k)
    n, secs = prefetch(WEIGHTS, LOCAL, threads, log)
    began = time.time()
    procs, logs = [], []
    for i, gpus in enumerate(('0,1,2,3', '4,5,6,7')):
        args = [a.replace('{reasoning}', reasoning) for a in serve_args(8000 + i, slots, max_len, LOCAL)]
        logs.append(out / f'vllm{i}.log')
        procs.append(subprocess.Popen(args, stdout=logs[-1].open('w'), stderr=subprocess.STDOUT, env=dp2_env(gpus)))
        log(f'vllm {i} started on GPUs {gpus}: {" ".join(args)}')
    for i, proc in enumerate(procs):
        while True:
            if proc.poll() is not None:
                log(f'vllm {i} exited {proc.returncode} before ready')
                for q in procs:
                    q.kill()
                stop.set()
                out_vol.commit()
                return {'status': 'server_failed', 'server': i, 'code': proc.returncode}
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{8000 + i}/v1/models', timeout=2)
                break
            except OSError:
                time.sleep(5)
        log(f'vllm {i} ready after {time.time() - began:.0f} s')

    def gpu_sampler():
        rows = (out / 'gpu.jsonl').open('a')
        while not stop.wait(30):
            q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True).stdout.strip().splitlines()
            rows.write(json.dumps({'t': time.time(), 'gpu': [list(map(float, x.split(','))) for x in q]}) + '\n')
            rows.flush()
    threading.Thread(target=gpu_sampler, daemon=True).start()
    reports = run_drivers(out, [f'http://127.0.0.1:{8000 + i}/v1' for i in range(2)],
                          [layout_line(logs[i], max_len) for i in range(2)], slots, wall, overrides, log)
    rows = merge_rows(out)
    log(f'drivers done: {[r.get("status") for r in reports.values()]}; {rows} entries merged')
    for proc in procs:
        proc.terminate()
    stop.set()
    out_vol.commit()
    return {'status': [r.get('status') for r in reports.values()], 'entries': rows, 'prefetch_s': secs,
            'seconds': time.time() - began}


@app.local_entrypoint()
def dp2_main(run: str, overrides: str, slots: int = 48, wall: int = 7200):
    call = screen_dp2.spawn(run, overrides, slots, wall)
    print(json.dumps({'call': call.object_id}))


@app.function(image=gpu_image, timeout=3600, cpu=(8, 16), volumes={'/w': vol, '/out': out_vol}, secrets=[SECRET])
def pf_fixed(run, overrides_json, mode='dots3', copies=0, target='z-ai/glm-5.3-flash'):
    """CPU preflight of the fixed-attempt screens with this image's bundle: two hosted stand-in servers, the exact
    overrides (``copies`` > 0 lowers the copies), two drivers on 1 task each, rows merged, every entry one attempt,
    fixedread on the merged rows. ``mode`` dots3 also checks the DP2 serve commands and environments, the DeepGEMM
    warmup value against this vLLM, and the parallel prefetch on two weight shards."""
    import socket
    import sys
    import threading
    sys.path[:0] = ['/bundle/src', '/fr']
    try:
        from arc2 import api_proxy
    except ImportError:  # a bundle older than the API screens (the Flash RTX bundle): the stand-in from /fr
        import api_proxy
    from arena.stock_serving import layout_line
    import fixedread
    out = Path('/out') / run / 'pf'
    out.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(overrides_json)
    if int(copies):
        overrides.setdefault('trial', {})['copies'] = int(copies)
    res = {'mode': mode}
    if mode == 'dots3':
        import vllm
        src = (Path(vllm.__file__).parent / 'envs.py').read_text()
        i = src.index('"VLLM_DEEP_GEMM_WARMUP"')
        res['warmup_env'] = src[i:i + 400]
        assert '"skip"' in res['warmup_env'], res['warmup_env']
        args = [serve_args(8000 + k, 48, 262144, LOCAL) for k in range(2)]
        envs = [dp2_env(g) for g in ('0,1,2,3', '4,5,6,7')]
        assert [e['CUDA_VISIBLE_DEVICES'] for e in envs] == ['0,1,2,3', '4,5,6,7']
        assert all(a[2] == LOCAL and '262144' in a for a in args) and args[0][8] == '8000' and args[1][8] == '8001'
        small = sorted(Path(WEIGHTS).glob('*.safetensors'), key=lambda f: f.stat().st_size)[:2]
        tmp = Path('/tmp/pf-src')
        tmp.mkdir(exist_ok=True)
        for f in small:
            (tmp / f.name).symlink_to(f)
        n, secs = prefetch(tmp, '/tmp/pf-dst', 4)
        assert n == sum(f.stat().st_size for f in small)
        res['prefetch'] = {'gb': round(n / 1e9, 2), 'seconds': round(secs, 1)}
        res['arc2'] = overrides.get('arc2')
        assert overrides['arc2'].get('reply_reserve') == 16384
    urls = []
    for _ in range(2):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        threading.Thread(target=api_proxy.serve_forever, args=(port, target, os.environ['OPENROUTER_API_KEY']),
                         kwargs={'log_path': str(out / f'proxy{port}.jsonl')}, daemon=True).start()
        urls.append(f'http://127.0.0.1:{port}/v1')
    time.sleep(1)
    pool = out / 'pool.log'
    pool.write_text('GPU KV cache size: 1,131,347 tokens\n')
    reports = run_drivers(out, urls, [layout_line(pool, 262144)] * 2, 8, 1800, overrides, limit=1)
    rows = merge_rows(out)
    merged = [json.loads(x) for x in (out / 'tasks.jsonl').read_text().splitlines()]
    k = int(overrides['trial']['copies'])
    res.update(status=[r.get('status') for r in reports.values()], entries=rows,
               attempts=sorted({r['attempts'] for r in merged}), expected=2 * k)
    read = fixedread.read(out, out, 118800, data='/bundle/data/arc2/evaluation')
    res['read'] = read
    res['ok'] = (rows == 2 * k and res['attempts'] == [1] and all(st != 'failed' for st in res['status'])
                 and read['tasks'] == 2 and read['diff'] == 0)
    (out / 'preflight-report.json').write_text(json.dumps(res, indent=1, default=str))
    out_vol.commit()
    print(json.dumps(res, default=str))
    return json.loads(json.dumps(res, default=str))


# --- v37 roster screen on one 4x H200, throughput -------------------------------------------
# The TP4 server keeps Dots3's MLA latent KV on every GPU (1.13M-token pool, 8.5 in flight in the v37 run); DP4
# attention + EP4 experts gives each GPU its own pool (~4x the tokens). DP4 first, TP4 if DP4 does not come up.

def fast_args(layout, port, slots, max_len=262144, model=LOCAL, batched=16384, dp=4):
    """vllm serve for ``layout`` dp4 (``slots`` sequences a DP rank) or tp4 (``slots`` in all)."""
    a = serve_args(port, slots, max_len, model)
    a[a.index('--max-num-batched-tokens') + 1] = str(batched)
    if layout == 'dp4':
        a[a.index('--tensor-parallel-size') + 1] = '1'
        a += ['--data-parallel-size', str(dp)]
    return a


def fast_env():
    """Weights from local disk; DeepGEMM warmup skipped (8.7-9.7 min a boot); compile and DeepGEMM JIT caches on the
    volume, committed after ready and at the end."""
    return {**os.environ, 'VLLM_CACHE_ROOT': '/w/vllm-cache', 'HF_HUB_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
            'DG_JIT_CACHE_DIR': '/w/deepgemm-cache', 'VLLM_DEEP_GEMM_WARMUP': 'skip'}


def pool_log(server_log, ranks, dst):
    """The gate's token pool: the per-rank ``GPU KV cache size`` lines of this start summed over ``ranks`` (fewer lines
    than ranks: the smallest found times ``ranks``) -> ``dst`` as one line. Returns (total, per-rank values)."""
    import re
    found = [int(x.replace(',', '')) for x in
             re.findall(r'GPU KV cache size: ([\d,]+) tokens', Path(server_log).read_text(errors='replace'))]
    if not found:
        raise RuntimeError('server logged no KV cache size')
    vals = found[-ranks:] if len(found) >= ranks else [min(found)] * ranks
    Path(dst).write_text(f'GPU KV cache size: {sum(vals):,} tokens\n')
    return sum(vals), vals


def boot(cands, log, poll=5, ready_timeout=5400):
    """Start each candidate ``(name, args, env, log_path, url)`` in turn until one answers ``url``; a candidate that
    exits before ready (or is not ready in ``ready_timeout`` s) is killed and the next one starts.
    Returns (name, proc, seconds) or (None, None, seconds)."""
    import subprocess
    import urllib.request
    began = time.time()
    for name, args, env, path, url in cands:
        t0 = time.time()
        proc = subprocess.Popen(args, stdout=Path(path).open('w'), stderr=subprocess.STDOUT, env=env)
        log(f'vllm {name} started: {" ".join(args)}')
        while True:
            if proc.poll() is not None:
                log(f'vllm {name} exited {proc.returncode} before ready after {time.time() - t0:.0f} s')
                break
            if time.time() - t0 > ready_timeout:
                proc.kill()
                log(f'vllm {name} not ready in {ready_timeout} s; killed')
                break
            try:
                urllib.request.urlopen(url, timeout=2)
                log(f'vllm {name} ready after {time.time() - t0:.0f} s ({time.time() - began:.0f} s since boot start)')
                return name, proc, time.time() - began
            except OSError:
                time.sleep(poll)
    return None, None, time.time() - began


@app.function(image=gpu_image, gpu=GPU, timeout=4 * 3600, cpu=(0.125, 16), ephemeral_disk=450 * 1024,
              volumes={'/w': vol, '/out': out_vol})
def screen_fast(run, overrides_json, wall=10800, name='dots3', max_len=262144, top_k=-1, reasoning='deepseek_r1',
                layouts='dp4,tp4', dp_seqs=16, tp_seqs=48, batched=16384, threads=16):
    """v37 roster screen of Dots3 on 4x H200: weights prefetched to local disk with parallel readers, DP4+EP4 server
    (TP4 fallback), the gate's pool summed over DP ranks, then the screen driver (no stop rule) to the mark."""
    import subprocess
    import sys
    import threading
    sys.path[:0] = ['/bundle/src']
    from arena.stock_serving import layout_line
    out = Path('/out') / run / name
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / 'screen.log').open('a')

    def log(msg):
        print(msg, flush=True)
        logf.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
        logf.flush()
    stop = threading.Event()
    committer(out_vol, stop)
    staged = json.loads(Path(WEIGHTS, 'staged.json').read_text())
    assert staged['ok'], staged
    overrides = json.loads(overrides_json)
    patch_top_k(top_k)
    began = time.time()
    n, secs = prefetch(WEIGHTS, LOCAL, threads, log)
    cands = []
    for layout in layouts.split(','):
        args = [a.replace('{reasoning}', reasoning) for a in
                fast_args(layout, 8000, dp_seqs if layout == 'dp4' else tp_seqs, max_len, LOCAL, batched)]
        cands.append((layout, args, fast_env(), out / f'vllm-{layout}.log', 'http://127.0.0.1:8000/v1/models'))
    layout, proc, boot_s = boot(cands, log)
    if proc is None:
        stop.set()
        out_vol.commit()
        return {'status': 'server_failed', 'prefetch_s': secs}
    try:
        vol.commit()  # compile / DeepGEMM JIT caches for the next boot
    except Exception:  # noqa: BLE001
        pass
    ranks = 4 if layout == 'dp4' else 1
    total, vals = pool_log(out / f'vllm-{layout}.log', ranks, out / 'pool.log')
    slots = dp_seqs * 4 if layout == 'dp4' else tp_seqs
    log(f'layout {layout}: KV pool {total:,} tokens ({vals}); driver slots {slots}; boot {time.time() - began:.0f} s '
        f'(prefetch {secs:.0f} s)')

    def gpu_sampler():
        rows = (out / 'gpu.jsonl').open('a')
        while not stop.wait(30):
            q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True).stdout.strip().splitlines()
            rows.write(json.dumps({'t': time.time(), 'gpu': [list(map(float, x.split(','))) for x in q]}) + '\n')
            rows.flush()
    threading.Thread(target=gpu_sampler, daemon=True).start()
    report = screen_driver(out, 'http://127.0.0.1:8000/v1', layout_line(out / 'pool.log', max_len), slots, wall,
                           log=log, overrides=overrides)
    log(f'driver done: {report.get("status")}')
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        vol.commit()
    except Exception:  # noqa: BLE001
        pass
    stop.set()
    out_vol.commit()
    return {'status': report.get('status'), 'layout': layout, 'pool': total, 'boot_s': boot_s, 'prefetch_s': secs,
            'seconds': time.time() - began}


@app.local_entrypoint()
def fast_main(run: str, overrides: str, wall: int = 10800, layouts: str = 'dp4,tp4'):
    call = screen_fast.spawn(run, overrides, wall, layouts=layouts)
    print(json.dumps({'call': call.object_id}))


@app.function(image=gpu_image, timeout=3600, cpu=(8, 16), volumes={'/w': vol, '/out': out_vol}, secrets=[SECRET])
def pf_fast(run, overrides_json, target='z-ai/glm-5.3-flash'):
    """CPU preflight of screen_fast with this image and bundle: both layouts' serve args parsed by this vLLM's own
    CLI parser; DeepGEMM warmup 'skip' accepted; pool_log on 4-rank and 1-rank logs; parallel prefetch on two shards;
    boot() falls back past a server that exits to one that answers; then the driver end to end on 2 tasks with the
    exact overrides against a hosted stand-in, the gate on a DP4-sized pool."""
    import socket
    import sys
    import threading
    sys.path[:0] = ['/bundle/src']
    from arc2 import api_proxy
    from arena.stock_serving import layout_line
    out = Path('/out') / run / 'pf'
    out.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(overrides_json)
    res = {}
    import vllm
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    src = (Path(vllm.__file__).parent / 'envs.py').read_text()
    i = src.index('"VLLM_DEEP_GEMM_WARMUP"')
    assert '"skip"' in src[i:i + 400]
    parsed = {}
    for layout, seqs in (('dp4', 16), ('tp4', 48)):
        a = [x.replace('{reasoning}', 'deepseek_r1') for x in fast_args(layout, 8000, seqs)]
        ns = make_arg_parser(FlexibleArgumentParser()).parse_args(a[1:])
        parsed[layout] = {'tp': ns.tensor_parallel_size, 'dp': ns.data_parallel_size, 'ep': ns.enable_expert_parallel,
                          'seqs': ns.max_num_seqs, 'batched': ns.max_num_batched_tokens, 'len': ns.max_model_len,
                          'prefix': ns.enable_prefix_caching, 'model': getattr(ns, 'model', None) or getattr(ns, 'model_tag', None)}
    assert parsed['dp4']['tp'] == 1 and parsed['dp4']['dp'] == 4 and parsed['dp4']['ep'] and parsed['dp4']['seqs'] == 16
    assert parsed['tp4']['tp'] == 4 and parsed['tp4']['dp'] == 1 and parsed['tp4']['seqs'] == 48
    assert all(p['batched'] == 16384 and p['len'] == 262144 and p['prefix'] for p in parsed.values())
    res['parsed'] = parsed
    t4 = Path('/tmp/v4.log')
    t4.write_text('old start\nGPU KV cache size: 9 tokens\n' + ''.join(
        f'(EngineCore_DP{k} pid=1) INFO [kv_cache_utils.py:1] GPU KV cache size: {v:,} tokens\n'
        for k, v in enumerate((1_000_000, 1_000_100, 999_900, 1_000_000))))
    assert pool_log(t4, 4, '/tmp/p4.log') == (4_000_000, [1_000_000, 1_000_100, 999_900, 1_000_000])
    assert pool_log(t4, 1, '/tmp/p1.log')[0] == 1_000_000
    one = Path('/tmp/v1.log')
    one.write_text('GPU KV cache size: 1,131,347 tokens\n')
    assert pool_log(one, 4, '/tmp/p5.log')[0] == 4 * 1_131_347
    assert '4,000,000' in Path('/tmp/p4.log').read_text() and 'num_blocks' in layout_line('/tmp/p4.log', 262144)
    small = sorted(Path(WEIGHTS).glob('*.safetensors'), key=lambda f: f.stat().st_size)[:2]
    tmp = Path('/tmp/pf-src')
    tmp.mkdir(exist_ok=True)
    for f in small:
        (tmp / f.name).symlink_to(f)
    n, secs = prefetch(tmp, '/tmp/pf-dst', 4)
    assert n == sum(f.stat().st_size for f in small)
    res['prefetch'] = {'gb': round(n / 1e9, 2), 'seconds': round(secs, 1)}
    fake = Path('/tmp/fake/v1')
    fake.mkdir(parents=True, exist_ok=True)
    (fake / 'models').write_text('{"data": [{"id": "x"}]}')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        fport = probe.getsockname()[1]
    name, proc, _ = boot([('dp4', [sys.executable, '-c', 'import sys; sys.exit(3)'], dict(os.environ), '/tmp/b1.log',
                           f'http://127.0.0.1:{fport}/v1/models'),
                          ('tp4', [sys.executable, '-m', 'http.server', str(fport), '--bind', '127.0.0.1', '--directory',
                                   '/tmp/fake'], dict(os.environ), '/tmp/b2.log', f'http://127.0.0.1:{fport}/v1/models')],
                         print, poll=1, ready_timeout=60)
    assert name == 'tp4', name
    proc.kill()
    res['fallback'] = name
    study, trial = trial_for(64, 1800, 2, True, overrides)
    res['arm'], res['arc2'], res['temperature'], res['windows'] = trial['arms'][0], study['arc2'], study['temperature'], study['windows']
    assert study['arc2'].get('reply_reserve') == overrides['arc2']['reply_reserve'] and trial['arms'][0]['cap'] == 98304
    patch_top_k(-1)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    threading.Thread(target=api_proxy.serve_forever, args=(port, target, os.environ['OPENROUTER_API_KEY']),
                     kwargs={'log_path': str(out / 'proxy.jsonl')}, daemon=True).start()
    time.sleep(1)
    report = screen_driver(out, f'http://127.0.0.1:{port}/v1', layout_line('/tmp/p4.log', 262144), 8, 1800, limit=2,
                           overrides={**overrides, 'trial': {**overrides.get('trial', {}), 'marks': {'pf': 60000}}})
    tf = out / 'tasks.jsonl'
    rows = [json.loads(x) for x in tf.read_text().splitlines() if x.strip()] if tf.exists() else []
    res.update(status=report.get('status'), error=report.get('error'), finished=len(rows), seconds=report.get('seconds'),
               request_errors=report.get('request_errors'))
    res['ok'] = report.get('status') != 'failed' and len(rows) >= 1
    (out / 'preflight-report.json').write_text(json.dumps(res, indent=1, default=str))
    out_vol.commit()
    print(json.dumps(res, default=str))
    return json.loads(json.dumps(res, default=str))


@app.local_entrypoint()
def pf_fast_main(run: str, overrides: str):
    call = pf_fast.spawn(run, overrides)
    print(json.dumps({'call': call.object_id}))
