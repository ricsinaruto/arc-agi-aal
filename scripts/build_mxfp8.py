"""Package the optional vLLM MXFP8 comparison, reusing verified arena dependencies."""

import argparse
import json
import shutil
from pathlib import Path

from build_arena import ROOT, build, manifest


def prepare(output, dependency_bundle, *, kv_cache_memory_bytes=0):
    if type(kv_cache_memory_bytes) is not int or kv_cache_memory_bytes < 0:
        raise ValueError("Cache bytes must be non-negative")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory to preserve comparison archives')
    output.mkdir(parents=True)
    for name in ('wheels', 'pi', 'node-v22.19.0-linux-x64.tar.xz', 'NODE_SHA256.txt'):
        source = Path(dependency_bundle) / name
        if source.is_dir():
            shutil.copytree(source, output / name, symlinks=True)
        else:
            shutil.copyfile(source, output / name)
    build(output.name, output=output)
    shutil.copytree(ROOT / 'src/sglang_bench', output / 'src/sglang_bench',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT / 'configs/duck_transfer_sglang.yaml', output / 'configs/duck_transfer_sglang.yaml')
    shutil.copyfile(ROOT / 'scripts/mxfp8_bootstrap.py', output / 'bootstrap_mxfp8.py')
    study = {'kv_cache_memory_bytes': kv_cache_memory_bytes, 'backend': 'vllm', 'online_mxfp8': True, 'seconds': 900,
             'games': ['ft09-0d8bbf25', 'lp85-305b61c3', 'ar25-0c556536'],
             'model': 'Qwen/Qwen3.8-Flash-Next-NVFP4', 'config': 'configs/duck_transfer_sglang.yaml'}
    (output / 'mxfp8-study.json').write_text(json.dumps(study, indent=2))
    identity = json.loads((output / 'MANIFEST.json').read_text())
    identity.update(files=manifest(output), dependencies_prepared=True,
                    dependency_bundle=str(Path(dependency_bundle).resolve()), study=study)
    (output / 'MANIFEST.json').write_text(json.dumps(identity, indent=2))
    return identity


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--dependency-bundle', required=True)
    p.add_argument("--kv-cache-memory-bytes", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(prepare(a.output, a.dependency_bundle, kv_cache_memory_bytes=a.kv_cache_memory_bytes)['study']))
