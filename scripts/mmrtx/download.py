"""mmrtx: MiniMax-M3.1-Flash (pinned sha) -> Modal volume mmrtx-models:/minimax-m31f, sizes checked against the HF tree.
usage: modal run --detach download.py"""
import modal
app = modal.App('mmrtx-download')
vol = modal.Volume.from_name('mmrtx-models', create_if_missing=True)
image = modal.Image.debian_slim(python_version='3.12').pip_install('huggingface_hub[hf_xet]==0.36.0')
REPO, SHA = 'MiniMaxAI/MiniMax-M3.1-Flash', '54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e'


@app.function(image=image, timeout=4 * 3600, cpu=(0.125, 16), volumes={'/m': vol},
              secrets=[modal.Secret.from_name('hf-token')])
def fetch():
    import json, os, time, threading
    from pathlib import Path
    from huggingface_hub import HfApi, snapshot_download
    t0 = time.time(); dest = Path('/m/minimax-m31f')
    stop = threading.Event()
    def committer():
        while not stop.wait(300):
            try: vol.commit()
            except Exception: pass
    threading.Thread(target=committer, daemon=True).start()
    snapshot_download(REPO, revision=SHA, local_dir=str(dest), max_workers=16,
                      ignore_patterns=['figures/*', 'docs/*', '*.md'])
    info = HfApi().model_info(REPO, revision=SHA, files_metadata=True)
    bad = [s.rfilename for s in info.siblings if not s.rfilename.startswith(('figures/', 'docs/')) and not s.rfilename.endswith('.md')
           and (not (dest / s.rfilename).is_file() or (s.size is not None and (dest / s.rfilename).stat().st_size != s.size))]
    out = {'sha': SHA, 'seconds': round(time.time() - t0), 'bad': bad,
           'bytes': sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())}
    (dest / 'mmrtx-download.json').write_text(json.dumps(out))
    stop.set(); vol.commit()
    print(json.dumps(out))
    if bad: raise RuntimeError(f'size mismatch: {bad[:5]}')
    return out


@app.local_entrypoint()
def main():
    call = fetch.spawn()
    print(json.dumps({'call': call.object_id, 'app_id': app.app_id})) if False else print('call', call.object_id, 'app', app.app_id)
