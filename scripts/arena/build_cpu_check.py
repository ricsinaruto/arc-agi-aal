"""Prepare a CPU-only Kaggle SDK/sandbox check over an immutable runtime bundle.

The source overlay contains only explicit agent packages, tests and configurations.
No models, credentials or host files are exposed to agent tools. GPU use is disabled.
"""
import argparse
import base64
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ('arena', 'arc_infra', 'inference', 'taaf', 'duck_transfer', 'qwen_baselines')


def build(output, name, base_blob, base_sha):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as tar:
        for folder in [*(ROOT/'src'/p for p in PACKAGES), ROOT/'tests', ROOT/'configs']:
            for file in sorted(folder.rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.pyc':
                    tar.add(file, arcname=str(file.relative_to(ROOT)))
        for name_ in ('pi-arena/agent.mjs', 'pyproject.toml'):
            tar.add(ROOT/name_, arcname=name_)
    overlay = data.getvalue()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    code = '''import base64, hashlib, io, json, os, subprocess, sys, tarfile, time, traceback
from pathlib import Path
start=time.monotonic()
result={'status':'running','gpu':False,'source_commit':COMMIT,'overlay_sha256':OVERLAY_SHA,'base_sha256':BASE_SHA}
try:
    candidates=list(Path('/kaggle/input').rglob(BASE_BLOB))
    assert len(candidates)==1, candidates
    blob=candidates[0]
    with blob.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==BASE_SHA
    bundle=Path('/tmp/arc-cpu-base'); bundle.mkdir()
    with tarfile.open(blob) as tar: tar.extractall(bundle,filter='data')
    sys.path.insert(0,str(bundle))
    from bootstrap import verify, install_agent_environment
    from bootstrap_arena import stage_pi
    result['base_manifest']=verify(bundle)
    overlay=base64.b64decode(OVERLAY)
    assert hashlib.sha256(overlay).hexdigest()==OVERLAY_SHA
    root=Path('/tmp/arc-cpu-source');root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(overlay)) as tar:tar.extractall(root,filter='data')
    work=Path('/tmp/arc-cpu-work');work.mkdir()
    env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'}
    python=install_agent_environment(bundle,work/'venv',env)
    pip_env={**env,'PYTHONPATH':str(bundle/'wheels/pip-25.2-py3-none-any.whl'),'PIP_CONFIG_FILE':os.devnull}
    subprocess.run([python,'-m','pip','install','pytest==9.0.2','pytest-asyncio==1.3.0'],env=pip_env,check=True)
    pi,node=stage_pi(bundle,work)
    (pi/'agent.mjs').write_bytes((root/'pi-arena/agent.mjs').read_bytes())
    (root/'.analysis-venv').symlink_to(work/'venv',target_is_directory=True)
    env.update(PATH=str(node.parent)+':'+os.environ['PATH'],ARC_PI_DIR=str(work/'pi'),ARC_PI_ARENA_DIR=str(pi))
    command=[python,'-m','pytest','-q','tests/test_qwen_pi.py','tests/test_arena_text_only.py','-k','not candidate_bundles and not cpu_notebook']
    run=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=600)
    result.update(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,status='passed' if run.returncode==0 else 'failed')
    print(run.stdout,run.stderr,flush=True)
except BaseException as exc:
    result.update(status='error',error=repr(exc),traceback=traceback.format_exc())
finally:
    result['seconds']=time.monotonic()-start
    Path('/kaggle/working/cpu-validation.json').write_text(json.dumps(result,indent=2))
assert result['status']=='passed', result.get('error','CPU validation failed; inspect saved report')
'''
    for key,value in [('OVERLAY_SHA', hashlib.sha256(overlay).hexdigest()), ('BASE_SHA', base_sha),
                      ('BASE_BLOB', base_blob), ('COMMIT', commit), ('OVERLAY', base64.b64encode(overlay).decode())]:
        code = code.replace(key, repr(value))
    compile(code, 'cpu-check.ipynb', 'exec')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'check.ipynb').write_text(json.dumps({'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python','version':'3.12'}},'cells':[
        {'cell_type':'markdown','id':'cpu-intro','metadata':{},'source':['# Native Linux CPU-only agent validation\nReal Pi SDK, tools, compaction, and sandbox. No GPU or game models.']},
        {'cell_type':'code','id':'cpu-tests','metadata':{},'execution_count':None,'outputs':[],'source':[code]}]},indent=1))
    (output/'kernel-metadata.json').write_text(json.dumps({'id':'richardcsaky/'+name,'title':name,
        'code_file':'check.ipynb','language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':False,'enable_internet':True,'dataset_sources':['richardcsaky/arc-arena-runtime'],
        'competition_sources':[],'model_sources':[],'kernel_sources':[]},indent=2))
    return {'source_commit':commit,'overlay_sha256':hashlib.sha256(overlay).hexdigest(),
            'overlay_bytes':len(overlay),'base_blob':base_blob,'base_sha256':base_sha}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--base-blob',required=True)
    parser.add_argument('--base-sha',required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output,args.name,args.base_blob,args.base_sha)))
