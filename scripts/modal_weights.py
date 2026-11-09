"""CPU-only Modal weight downloads for the llama.cpp model screens (27 September).

    ARC_LLAMACPP_PROFILE=glm53-flash-aj-iq2xxs uv run --extra modal modal run --detach \
        scripts/modal_weights.py::download_main
    ARC_LLAMACPP_PROFILE=... uv run --extra modal modal run scripts/modal_weights.py::delete_main

Files land in the ``arc-llamacpp-weights`` Volume (v2) under ``<profile>/<repo path>``, where
scripts/modal_llamacpp.py mounts them. Separate from the GPU launcher so a download never waits for
the CUDA image build. Cost: the infrastructure reserve (CPU and memory only); storage is about
/GiB-month, so a screen's weights are deleted once it is done unless the result is promising.
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
PROFILE_NAME = os.environ.get("ARC_LLAMACPP_PROFILE", "glm53-flash-aj-iq2xxs")
if modal.is_local():
    sys.path.insert(0, str(ROOT / "src"))
    from arena.llamacpp_profiles import profile

    PROFILE = profile(PROFILE_NAME)
else:
    PROFILE = {}

app = modal.App("arc-llamacpp-weights", include_source=False)
weights = modal.Volume.from_name("arc-llamacpp-weights", create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface_hub>=0.34", "hf_xet", "hf_transfer")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
         .add_local_file(__file__, "/root/modal_weights.py", copy=True))


@app.function(image=image, cpu=8, memory=8192, timeout=3 * 3600, retries=0, max_containers=1,
              volumes={"/weights": weights})
def download(name: str, repo: str, files: dict):
    """Download each file into /weights/<name>/<path> (resumable); verify byte sizes; commit."""
    import time as _time
    from pathlib import Path as _Path

    from huggingface_hub import hf_hub_download

    target = _Path("/weights") / name
    target.mkdir(parents=True, exist_ok=True)
    log = []
    for path, size in files.items():
        local = target / path
        started = _time.time()
        if not (local.is_file() and local.stat().st_size == size):
            hf_hub_download(repo, path, local_dir=str(target))
        actual = local.stat().st_size
        if actual != size:
            raise RuntimeError(f"{path}: {actual} bytes, expected {size}")
        log.append({"file": path, "bytes": actual, "seconds": round(_time.time() - started, 1)})
        print(json.dumps(log[-1]), flush=True)
        weights.commit()
    manifest = {"repo": repo, "files": files, "log": log,
                "utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())}
    (target / "download.json").write_text(json.dumps(manifest, indent=2))
    weights.commit()
    return manifest


@app.function(image=image, cpu=0.25, memory=512, timeout=600, volumes={"/weights": weights})
def delete_weights(name: str):
    import shutil as _shutil
    from pathlib import Path as _Path

    target = _Path("/weights") / name
    existed = target.exists()
    if existed:
        _shutil.rmtree(target)
    weights.commit()
    return {"deleted": str(target), "existed": existed}


@app.local_entrypoint()
def download_main():
    call = download.spawn(PROFILE_NAME, PROFILE["repo"], PROFILE["files"])
    record = ROOT / "docs/validation-20260927" / f"modal-download-{PROFILE_NAME}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {"profile": PROFILE_NAME, "repo": PROFILE["repo"], "files": PROFILE["files"],
            "function_call_id": call.object_id, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cost_accounting": "CPU-only download; infrastructure reserve"}
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


@app.local_entrypoint()
def delete_main():
    print(json.dumps(delete_weights.remote(PROFILE_NAME)))
