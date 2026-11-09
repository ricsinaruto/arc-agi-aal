"""Bounded Sept 30 research preparation/pilots, independently budgeted from nine-hour games."""
import json
import os
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
TIMEOUT=3600
RECORDS=ROOT/'docs/validation-20260930'
REPOS={
 'bf16':('Qwen/Qwen3.8-Flash-Next','de4b8e4d43b917e7706784d8bb445c9af86a3540'),
 'swift':('ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4','3ff0520224f264a2d0ac4ab56ece8f2f13aadb38'),
 'iq3':('ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF','ed59f92082b1e93c0e96d60a8b11aab089b52f09'),
}
LLAMA_COMMIT='72db1e02ff0d804f7553be6f01b93c2fd1fa1a6d'
CPU_KINDS=(*REPOS,'iq3-build')
app=modal.App('arc-research-pilots-20260930',include_source=False)
assets=modal.Volume.from_name('arc-prolong-assets');runs=modal.Volume.from_name('arc-prolong-runs')
cpu_image=(modal.Image.debian_slim(python_version='3.12').pip_install('huggingface_hub==1.7.1','hf_xet==1.4.2')
           .add_local_file(__file__,'/root/modal_research.py',copy=True))
gpu_image=(modal.Image.debian_slim(python_version='3.12').apt_install('build-essential').pip_install('vllm==0.30.0')
           .add_local_dir(str(ROOT/'src/research_pilots'),'/opt/research_pilots',copy=True)
           .add_local_file(str(ROOT/'data/expert_prune/prune25-layer.json'),'/opt/prune25-layer.json',copy=True)
           .add_local_file(__file__,'/root/modal_research.py',copy=True))
build_image=(modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04',add_python='3.12').entrypoint([])
             .apt_install('git','cmake','build-essential','ninja-build','ca-certificates')
             .add_local_file(__file__,'/root/modal_research.py',copy=True))


def reserve(run_id,kind):
    import fcntl
    path=ROOT/'docs/spend.json'
    gpu=kind not in CPU_KINDS
    cpu=32 if kind=='iq3-build' else 8
    ram=64 if kind=='iq3-build' else 96 if gpu else 16
    rate=(.000842 if gpu else 0)+cpu*.0000131+ram*.00000222
    bound=(TIMEOUT+60)*rate*1.15+(1.5 if not gpu else 0) # CPU prep includes a bounded storage allowance
    with path.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);d=json.load(f)
        if any(r['run_id']==run_id for r in d['runs']):raise ValueError('Already reserved')
        active=sum(r.get('kind')=='research-pilot' and r.get('state')=='reserved' for r in d['runs'])
        if gpu and active>=4:raise RuntimeError('Four research pilot reservations active; reconcile before another')
        used=d['infrastructure_reserve_usd']+sum(r['charged_bound_usd'] for r in d['runs'])
        if used+bound>d['cap_usd']:raise RuntimeError(f'Cap: {used}+{bound}>{d["cap_usd"]}')
        d['runs'].append({'run_id':run_id,'state':'reserved','kind':'research-pilot' if gpu else 'research-preparation',
                         'charged_bound_usd':bound,'max_seconds':TIMEOUT,'gpu':'RTX-PRO-6000' if gpu else None,
                         'cpu_ceiling':cpu,'memory_ceiling_gib':ram,
                         'authorization':', <=4 research GPUs, each pilot <=1 hour', 'research_kind':kind})
        f.seek(0);json.dump(d,f,indent=1);f.write('\n');f.truncate()
    return bound


@app.function(image=build_image,cpu=32,memory=65536,timeout=TIMEOUT,retries=0,volumes={'/assets':assets})
def compile_iq3(script):
    import subprocess,time
    started=time.time()
    subprocess.run(['bash','-c',script],check=True)
    assets.commit()
    return {'kind':'iq3-build','commit':LLAMA_COMMIT,'seconds':time.time()-started,
            'tarball':f'/assets/research-20260930/llama-{LLAMA_COMMIT}.tar.gz'}


@app.function(image=cpu_image,cpu=8,memory=16384,timeout=TIMEOUT,retries=0,volumes={'/assets':assets})
def prepare(kind):
    import time
    from huggingface_hub import hf_hub_download,snapshot_download
    repo,revision=REPOS[kind];root=Path('/assets/research-20260930')/kind
    root.mkdir(parents=True,exist_ok=True);started=time.time()
    if kind=='bf16':
        index=hf_hub_download(repo,'model.safetensors.index.json',revision=revision,local_dir=root)
        m=json.loads(Path(index).read_text())['weight_map']
        files=sorted({m[f'model.language_model.layers.{l}.mlp.{n}'] for l in (0,20,40)
                      for n in ('experts.gate_up_proj','experts.down_proj','gate.weight')})+['config.json']
    elif kind=='iq3':
        files=['IQ3_S/*','mmproj-Qwen3.8-Flash-Next-BF16.gguf','README.md','LICENSE*']
    else:
        files=['*.safetensors','*.json','*.jinja','*.txt','*.model','LICENSE*','README.md']
    snapshot_download(repo,revision=revision,local_dir=root,allow_patterns=files,max_workers=8)
    result={'kind':kind,'repo':repo,'revision':revision,'seconds':time.time()-started,
            'files':{str(p.relative_to(root)):p.stat().st_size for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts}}
    (root/'prepared.json').write_text(json.dumps(result,indent=2));assets.commit();return result


@app.function(image=gpu_image,gpu='RTX-PRO-6000',cpu=(.125,8),memory=(128,98304),timeout=TIMEOUT,retries=0,
              max_containers=4,volumes={'/assets':assets,'/runs':runs})
def pilot(kind,run_id):
    import sys,time,traceback
    sys.path.insert(0,'/opt');out=Path('/runs/research')/run_id;out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    try:
        import torch
        # Tiny CPU test of the actual router training operation before any GPU computation.
        from research_pilots.recovery import mixture
        torch.manual_seed(7);x=torch.randn(8,4);router=torch.nn.Parameter(torch.randn(6,4));ys=torch.randn(8,6,4)
        y=mixture(torch,x,router,ys,torch.ones(6,dtype=torch.bool),3);y.square().sum().backward()
        assert router.grad is not None and torch.isfinite(router.grad).all()
        if kind=='state-storage':
            from research_pilots.state_storage import run
            result=run(out)
        elif kind in ('prune-recovery','topk-recovery'):
            from research_pilots.recovery import run
            root=Path('/assets/research-20260930/bf16');assert (root/'prepared.json').is_file(),'BF16 preparation incomplete'
            capture='/runs/validation/arena-expert-quant-bench-v1-modal-20260928/expert-capture'
            result=run(kind,root,capture,'/opt/prune25-layer.json',out,seconds=TIMEOUT-(time.time()-started)-180)
        else:raise ValueError(kind)
        result.update(elapsed_seconds=time.time()-started,gpu=torch.cuda.get_device_name())
        (out/'execution.json').write_text(json.dumps(result,indent=2));runs.commit();return result
    except BaseException as e:
        (out/'execution.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started},indent=2));runs.commit();raise


@app.local_entrypoint()
def main(kind:str,run_id:str):
    if kind not in (*CPU_KINDS,'state-storage','prune-recovery','topk-recovery'):raise ValueError(kind)
    bound=reserve(run_id,kind)
    if kind=='iq3-build':
        from modal_llamacpp import build_script
        script=build_script({'commit':LLAMA_COMMIT,'ref':LLAMA_COMMIT},'https://github.com/ggml-org/llama.cpp',
                            f'/assets/research-20260930/llama-{LLAMA_COMMIT}.tar.gz')
        call=compile_iq3.spawn(script)
    else:
        call=prepare.spawn(kind) if kind in REPOS else pilot.spawn(kind,run_id)
    record={'run_id':run_id,'kind':kind,'function_call_id':call.object_id,'charged_bound_usd':bound,'state':'submitted'}
    RECORDS.mkdir(parents=True,exist_ok=True)
    (RECORDS/f'modal-{run_id}.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
