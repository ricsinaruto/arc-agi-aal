"""CPU-only Modal fetch of a reviewed candidate checkpoint into the assets volume (Holo4-27B, 29 September).

    ARC_VALIDATION_ROOT=. ARC_CANDIDATE=holo4-27b-nvfp4 uv run --extra modal modal run \
        --detach scripts/modal_candidate_weights.py::fetch_main --run-id <id>
    ARC_CANDIDATE=holo4-27b-nvfp4 uv run --extra modal modal run scripts/modal_candidate_weights.py::delete_main \
        --run-id <id>

The candidate's reviewed inventory (``src/arena/models/<name>.json``: repo, pinned revision, every file with its byte
size and sha256) decides what is fetched. Files land in ``arc-prolong-assets:/flash-input/datasets/<dataset>/``, which
the arena mounts as ``/kaggle/input/datasets/<dataset>`` (``arc_infra.serving.dataset_mount``); every file is hashed
after the download and the job fails on any size or sha256 mismatch (``arena.candidate_serving.verify_model`` checks the
same inventory again before every server start).

Ledger: the main checkout's docs/spend.json (ARC_VALIDATION_ROOT), reserved before the spawn with the procedure of
scripts/modal_validation.py: (timeout + 60 s) x (CPU + memory rate) x 1.15, plus the volume storage for
``STORAGE_DAYS``.
"""

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]
RECORDS = "docs/validation-20260927"
CANDIDATE = os.environ.get("ARC_CANDIDATE", "holo4-27b-nvfp4")
CPU, MEMORY_GIB, TIMEOUT = 8.0, 16, 3600
STORAGE_DAYS = 7  # kept for the smoke and the one-hour arena, then deleted unless the result is promising
AUTHORIZATION = os.environ.get("ARC_CANDIDATE_AUTHORIZATION") or ("test Holo4-27B (Hcompany/Holo4-27B-NVFP4, CC BY-NC fine for testing) with the "
                 "hero-v2 arena setup; CPU fetch of the pinned checkpoint into the assets volume (no GPU lock)")

if modal.is_local():
    MANIFEST = json.loads((REPO / "src/arena/models" / f"{CANDIDATE}.json").read_text())
else:
    MANIFEST = json.loads(Path("/root/manifest.json").read_text())
TARGET = f"/assets/flash-input/datasets/{MANIFEST['dataset']}"

app = modal.App("arc-candidate-weights", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface_hub>=0.34", "hf_xet", "hf_transfer")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
         .add_local_file(__file__, "/root/modal_candidate_weights.py", copy=True))
if modal.is_local():
    image = image.add_local_file(str(REPO / "src/arena/models" / f"{CANDIDATE}.json"), "/root/manifest.json",
                                 copy=True)


# --- pure helpers (tested) -------------------------------------------------------------------------


def storage_gib(manifest):
    return sum(spec["bytes"] for spec in manifest["files"].values()) / 2**30


def bound_usd(ledger, gib):
    rates = ledger.get("rates_per_second") or {}
    rate = CPU * rates.get("physical_cpu_core", 1.31e-05) + MEMORY_GIB * rates.get("memory_gib", 2.22e-06)
    storage = gib * ledger.get("volume_rate_usd_per_gib_month", 0.09) * STORAGE_DAYS / 30
    return (TIMEOUT + 60) * rate * 1.15 + storage, storage


def check_file(path, spec):
    """``{"file", "bytes", "sha256", "ok"}`` for one downloaded file against its inventory entry."""
    import hashlib

    path = Path(path)
    size = path.stat().st_size if path.is_file() else None
    digest = None
    if size == spec["bytes"]:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"file": path.name, "bytes": size, "sha256": digest, "ok": size == spec["bytes"] and digest == spec["sha256"]}


def reserve(run_id, gib, extra):
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        bound, storage = bound_usd(data, gib)
        committed = data["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": "weights-fetch", "charged_bound_usd": round(bound, 4),
            "max_seconds": TIMEOUT, "gpu": "none", "cpu_ceiling": CPU, "memory_ceiling_gib": MEMORY_GIB,
            "storage_bound_usd": round(storage, 4),
            "storage_note": f"{gib:.1f} GiB of {MANIFEST['repo']}@{MANIFEST['revision'][:12]} in "
                            f"arc-prolong-assets{TARGET[7:]} for {STORAGE_DAYS} days",
            "authorization": AUTHORIZATION, **extra})
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


# --- remote ----------------------------------------------------------------------------------------


@app.function(image=image, cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TIMEOUT, retries=0, max_containers=1,
              volumes={"/assets": assets})
def fetch():
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    target = Path(TARGET)
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    files = MANIFEST["files"]

    def one(name):
        spec = files[name]
        t0 = time.time()
        row = check_file(target / name, spec)
        if not row["ok"]:
            hf_hub_download(MANIFEST["repo"], name, revision=MANIFEST["revision"], local_dir=str(target))
            row = check_file(target / name, spec)
        row["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(one, sorted(files)))
    record = {"repo": MANIFEST["repo"], "revision": MANIFEST["revision"], "dataset": MANIFEST["dataset"],
              "files": rows, "all_sha256_verified": all(r["ok"] for r in rows),
              "seconds": round(time.time() - started, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (target / f"fetch-{int(started)}.json").write_text(json.dumps(record, indent=1))
    assets.commit()
    bad = [r["file"] for r in rows if not r["ok"]]
    if bad:
        raise RuntimeError(f"size/sha256 mismatch: {bad}")
    return record


@app.function(image=image, cpu=0.25, memory=512, timeout=600, volumes={"/assets": assets})
def delete_weights():
    import shutil

    target = Path(TARGET)
    existed = target.exists()
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if existed else 0
    if existed:
        shutil.rmtree(target)
    assets.commit()
    return {"deleted": TARGET, "existed": existed, "bytes": size}


@app.local_entrypoint()
def fetch_main(run_id: str):
    gib = storage_gib(MANIFEST)
    bound = reserve(run_id, gib, {"repo": MANIFEST["repo"], "revision": MANIFEST["revision"],
                                  "files": len(MANIFEST["files"])})
    record = REPO / RECORDS / f"modal-{run_id}.json"
    data = {"state": "submission_intent", "run_id": run_id, "kind": "weights-fetch", "charged_bound_usd": bound,
            "candidate": CANDIDATE, "repo": MANIFEST["repo"], "revision": MANIFEST["revision"], "gib": round(gib, 2),
            "target": f"arc-prolong-assets:{TARGET[7:]}", "ledger": str(ROOT / "docs/spend.json")}
    record.write_text(json.dumps(data, indent=2))
    call = fetch.spawn()
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


@app.local_entrypoint()
def delete_main(run_id: str):
    result = delete_weights.remote()
    record = REPO / RECORDS / f"modal-{run_id}.json"
    data = json.loads(record.read_text()) if record.exists() else {"run_id": run_id}
    data["deleted"] = result
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(result))
