"""Verified offline native-vLLM startup followed by the selected-game fleet comparison."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from bootstrap import collect_evidence, install_agent_environment, verify


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'working', 'output', 'games-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    a = p.parse_args()
    bundle, work = Path(a.bundle), Path(a.working)
    identity = verify(bundle)
    study = json.loads((bundle / 'mxfp8-study.json').read_text())
    work.mkdir(parents=True, exist_ok=False)
    started = time.time()
    env = {**os.environ, 'ARC_VLLM_ONLINE_MXFP8': '1', 'PYTHONPATH': str(bundle / 'src')}
    python = install_agent_environment(bundle, work / 'agent-venv', env)
    result = {'state': 'setup', 'study': study, 'bundle': identity}
    try:
        subprocess.run([python, '-m', 'qwen_baselines.cli', 'serve-native', '--working',
                        str(work / 'serving'), '--input-root', '/kaggle/input',
                        '--kv-cache-memory-bytes', str(study.get('kv_cache_memory_bytes', 0))], env=env, check=True,
                       timeout=max(1, started + a.max_seconds - study['seconds'] - 100 - time.time()))
        result['setup_seconds'] = time.time() - started
        native = json.loads((work / 'serving/native-preflight.json').read_text())
        if native['profile']['kv_cache_memory_bytes'] != study.get('kv_cache_memory_bytes', 0):
            raise RuntimeError('Native serving cache allocation differs from the trial definition')
        result['native_profile'] = native['profile']
        result['patch'] = json.loads((work / 'serving/online-mxfp8-patch.json').read_text())
        logs = '\n'.join(f.read_text(errors='replace') for f in (work / 'serving').glob('*.log'))
        layers = [json.JSONDecoder().raw_decode(line.split('ARC_MXFP8_LAYER ', 1)[1])[0] for line in logs.splitlines()
                  if 'ARC_MXFP8_LAYER ' in line]
        if not layers:
            raise RuntimeError('No online MXFP8 layers were selected; refusing BF16 fallback benchmark')
        (work / 'serving/mxfp8-layers.json').write_text(json.dumps(layers, indent=2))
        command = [python, '-m', 'sglang_bench.transfer', '--backend', 'vllm', '--source', str(bundle),
                   '--output', str(work / 'trials/transfer'), '--games-dir', a.games_dir,
                   '--base-url', 'http://127.0.0.1:1234/v1', '--model', study['model'],
                   '--config', study['config'], '--seconds', str(study['seconds'])]
        for game in study['games']:
            command.extend(['--game', game])
        subprocess.run(command, env=env, check=True,
                       timeout=max(1, started + a.max_seconds - 80 - time.time()))
        result['state'] = 'complete'
    except BaseException as exc:
        result.update(state='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (work / 'setup.json').write_text(json.dumps(result, indent=2))
        collect_evidence(work, a.output)
        program = 'from arc_infra.serving import load_recipe; import sys; load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()'
        subprocess.run([python, '-c', program, str(work / 'serving'), '/kaggle/input'],
                       env=env, check=False, timeout=60)
        collect_evidence(work, a.output)


if __name__ == '__main__':
    main()
