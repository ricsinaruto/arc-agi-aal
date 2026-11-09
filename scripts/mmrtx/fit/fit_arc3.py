"""ARC-3 reference read of a hosted model on v8.5p's harness: arena.run (duck-transfer, v8.5p bundle,
20 c100 instances k000-k019) on Modal CPU against arc2's api_proxy (OpenRouter). The proxy is wrapped so that the
harness's verbatim reasoning (``reasoning_content``) goes back as ``reasoning`` and the provider is pinned (with
fallbacks) for prompt caching. The KV gate reads v8.5p's served cache layout (copied into the server dir), so
admission is v8.5p's. Outputs: runs volume validation/<run>/ (live/ every 120 s, the full trial at the end).
usage: MM_BUNDLE=DIR python scripts/openrouter_cli.py -- modal run --detach mm_arena.py --run RUN --target MODEL
       [--instances 20 --token-stop 200000 --wall 23400 --provider minimax]"""
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent.parent / 'mmarc3' / 'h'
FIT = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get('MM_BUNDLE', '/bundle')).resolve()
app = modal.App('arc-mmrtx-fit3')
# mmrtx: excl5 (fit, 2 instances each by the cyclic roster) + 5 non-excl5 games (held-out, 1 each)
GAMES = ['lp85-305b61c3', 'tu93-0768757b', 'vc33-5430563c', 'ar25-0c556536', 'ft09-0d8bbf25',
         'tn36-ef4dde99', 'lf52-271a04aa', 'cn04-2fe56bfb', 'bp35-0a0ad940', 'wa30-ee6fef47']
assets = modal.Volume.from_name('arc-prolong-assets')
runs = modal.Volume.from_name('arc-prolong-runs')
image = modal.Image.debian_slim(python_version='3.12').pip_install('tokenizers==0.22.1', 'jinja2==3.1.6')
if modal.is_local():
    image = (image.add_local_file(str(HERE / 'nvidia-smi'), '/usr/local/bin/nvidia-smi', copy=True)
             .run_commands('chmod +x /usr/local/bin/nvidia-smi')
             .add_local_dir(str(BUNDLE), '/bundle', ignore=['draft-head', '**/__pycache__'])
             .add_local_file(str(HERE / 'api_proxy.py'), '/opt/mm/api_proxy.py')
             .add_local_file(str(HERE / 'layout.log'), '/opt/mm/layout.log')
             .add_local_dir(str(HERE / 'tok'), '/opt/mm/tok')
             .add_local_file(str(HERE / 'layout_v2.py'), '/opt/mm/layout_v2.py')
             .add_local_dir(str(HERE / 'patches'), '/opt/mm/patches')
             .add_local_file(str(FIT / 'capture.py'), '/opt/mm/capture.py'))
SECRET = modal.Secret.from_dict({'OPENROUTER_API_KEY': os.environ.get('OPENROUTER_API_KEY', '')})


STATE = {}


def start_proxy(port, target, key, served, provider, log_path):
    import threading
    import time
    import urllib.request

    sys.path.insert(0, '/opt/mm')
    import api_proxy

    base_body = api_proxy.upstream_body

    def upstream_body(body, target_, keep_reasoning=True):
        msgs = []
        for m in body.get('messages') or []:
            if m.get('role') == 'assistant' and m.get('reasoning_content') and not m.get('reasoning'):
                m = {**m, 'reasoning': m['reasoning_content']}
            msgs.append(m)
        out = base_body({**body, 'messages': msgs}, target_, keep_reasoning)
        if provider:
            out['provider'] = {'order': [provider], 'allow_fallbacks': True}
        return out

    api_proxy.upstream_body = upstream_body
    base_reply = api_proxy.reply_body

    def reply_body(data):
        data = base_reply(data)
        usage = data.get('usage') or {}
        with open(log_path + '.providers', 'a') as f:  # provider + cache per reply
            f.write(json.dumps({'provider': data.get('provider'), 'model': data.get('model'),
                                'cached': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                                'prompt': usage.get('prompt_tokens'), 'completion': usage.get('completion_tokens'),
                                'cost': usage.get('cost')}) + '\n')
        return data

    api_proxy.reply_body = reply_body
    import capture
    rec = capture.Recorder(log_path.replace('proxy.jsonl', 'traj.jsonl.gz')).start()
    capture.install(api_proxy, rec)
    STATE['recorder'] = rec
    threading.Thread(target=api_proxy.serve_forever, args=(port, target, key),
                     kwargs={'served': served, 'log_path': log_path}, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError('proxy did not start')


@app.function(image=image, timeout=8 * 3600, cpu=(0.125, 8), volumes={'/assets': assets, '/runs': runs},
              secrets=[SECRET])
def play(run: str, target: str, instances: int, token_stop: int, wall: int, provider: str = '', effort: str = 'medium',
         temperature: float = 1.0, top_p: float = 0.95, window: int = 1040384, cap: int = 32768,
         image_tokens: int = 531, dump: int = 0, patch: str = '', schedule: str = ''):
    import hashlib
    import shutil
    import subprocess
    import threading
    import time

    started = time.time()
    out = Path('/runs/validation') / run
    out.mkdir(parents=True, exist_ok=False)
    live = out / 'live'
    live.mkdir()
    bundle = Path('/tmp/bundle')
    shutil.copytree('/bundle', bundle)
    if patch:  # mmarc3: harness patch = [[bundle-relative file, old, new], ...], each old exactly once
        manifest = json.loads((bundle / 'MANIFEST.json').read_text())
        for rel, old, new in json.loads(Path(f'/opt/mm/patches/{patch}.json').read_text()):
            f = bundle / rel
            text = f.read_text()
            assert text.count(old) == 1, (rel, old[:80], text.count(old))
            f.write_text(text.replace(old, new))
            if rel in manifest['files']:
                manifest['files'][rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        (bundle / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    study_path = bundle / 'configs/comparison.json'
    study = json.loads(study_path.read_text())
    allg = list(study['games']) + list(study.get('games_excluded') or [])
    assert sorted(allg) == sorted(set(allg)) and set(GAMES) <= set(allg), allg
    study['games'] = list(GAMES)
    study['games_excluded'] = [g for g in allg if g not in GAMES]
    study['competition']['preview_instances'] = int(instances)
    study['token_stop'] = int(token_stop)
    study['wall_seconds'] = int(wall)
    study['competition']['preview_seconds'] = int(wall)
    # v2: MiniMax's own window (OpenRouter 1,048,576 less 8,192 headroom) and reply cap; drain at v8.5p's ratio
    # (drain budget = window - cap - 512, drain_tokens = half of it; turn cap unchanged).
    study['windows'] = [int(window)]
    study['completion_cap'] = int(cap)
    study['drain_half']['drain_tokens'] = (int(window) - int(cap) - 512) // 2
    study_path.write_text(json.dumps(study, indent=2))
    ctl = bundle / 'configs/duck_controls.yaml'
    text = ctl.read_text()
    assert text.count('image_tokens_estimate: 400') == 1
    ctl.write_text(text.replace('image_tokens_estimate: 400', f'image_tokens_estimate: {int(image_tokens)}'))
    (out / 'study-overrides.json').write_text(json.dumps({'target': target, 'instances': instances, 'token_stop':
                                                           token_stop, 'wall': wall, 'provider': provider, 'effort': effort,
                                                           'temperature': temperature, 'top_p': top_p, 'window': window,
                                                           'cap': cap, 'image_tokens': image_tokens, 'patch': patch,
                                                           'schedule': schedule}))
    working = Path('/tmp/arena')
    serving = working / 'serving'
    trials = working / 'trials'
    serving.mkdir(parents=True)
    trials.mkdir(parents=True)
    # v8.5p's served layout with the pool scaled so 20 games at the 1M window pass the gate as v1's did at 130K
    # (an API has no KV pool; admission stays every game at once).
    sys.path.insert(0, '/opt/mm')
    import layout_v2
    (serving / 'vllm-openai-server.log').write_text(layout_v2.build(Path('/opt/mm/layout.log').read_text()))
    os.environ.update(MM_EFFORT=effort, MM_TEMPERATURE=str(temperature), MM_TOP_P=str(top_p), MM_TOKENIZER_DIR='/opt/mm/tok',
                      MM_IMAGE_TOKENS=str(int(image_tokens)), MM_MAX_MODEL_LEN='1048576', MM_DUMP_BODIES=str(int(dump)),
                      MM_EFFORT_SCHEDULE=schedule)
    env = {**os.environ, 'PYTHONPATH': str(bundle / 'src'), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PIP_NO_INDEX': '1', 'MPLBACKEND': 'Agg',
           'ONLY_RESET_LEVELS': 'true'}
    sys.path.insert(0, str(bundle))
    from bootstrap import install_agent_environment

    python = install_agent_environment(bundle, working / 'agent-venv', env)
    served = 'Qwen/Qwen3.8-Flash-Next-NVFP4'
    start_proxy(1234, target, os.environ['OPENROUTER_API_KEY'], served, provider, str(out / 'proxy.jsonl'))
    command = [python, '-u', '-m', 'arena.run', '--source', str(bundle), '--study', str(study_path),
               '--output', str(trials), '--trial', 'duck-transfer', '--server-dir', str(serving),
               '--analysis-python', python, '--base-url', 'http://127.0.0.1:1234/v1',
               '--notebook-start-epoch', str(started), '--notebook-seconds', str(int(wall) + 1800),
               '--games-dir', '/assets/games']
    stop = threading.Event()

    def sync():
        t = trials / 'duck-transfer'
        for f in [*t.glob('*.json'), *t.glob('*.jsonl'), trials / 'duck-transfer-process.log']:
            if f.is_file():
                data = f.read_bytes()
                (live / f.name).write_bytes(data[-3_000_000:] if f.suffix == '.log' else data)

    def committer():
        while not stop.wait(120):
            try:
                sync()
                runs.commit()
            except Exception as exc:  # noqa: BLE001
                print(f'commit failed: {exc}', flush=True)

    threading.Thread(target=committer, daemon=True).start()
    rc = None
    try:
        with (trials / 'duck-transfer-process.log').open('wb') as log:
            rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=int(wall) + 3000).returncode
    finally:
        stop.set()
        if STATE.get('recorder'):
            STATE['recorder'].flush()
        sync()
        dest = out / 'arena' / 'trials'
        shutil.copytree(trials, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns('prompts', 'agent-venv'))
        (out / 'execution.json').write_text(json.dumps({'exit_code': rc, 'seconds': round(time.time() - started, 1)}))
        runs.commit()
    return {'run': run, 'exit_code': rc}


@app.local_entrypoint()
def main(run: str, target: str, instances: int = 20, token_stop: int = 200000, wall: int = 23400, provider: str = '',
         effort: str = 'medium', temperature: float = 1.0, top_p: float = 0.95, window: int = 1040384, cap: int = 32768,
         image_tokens: int = 531, dump: int = 0, patch: str = '', schedule: str = ''):
    call = play.spawn(run, target, instances, token_stop, wall, provider, effort, temperature, top_p, window, cap,
                      image_tokens, dump, patch, schedule)
    print(json.dumps({'run': run, 'function_call_id': call.object_id, 'app_id': app.app_id}))
