"""CPU-only Modal upload of an assets-volume dataset directory as a private Kaggle dataset (29 September).

"please create a competition submission with run B" -- run B serves the GPTQ-ARC routed-expert
overlay (``arena.variants.EXPERT_OVERLAY_GPTQ_ARC``: 192 shards, 63.3 GiB), which lived only in the Modal assets volume.
A local download runs at ~9 MB/s (~2 h for 63 GB, before the upload), so the upload runs next to the volume.

Launch under the authorized Kaggle wrapper, which puts the token in the environment of this process only; the token
travels to the job as an ephemeral ``modal.Secret.from_dict`` (never a named secret, never printed, redacted from every
log line):

    ARC_VALIDATION_ROOT=. uv run --extra kaggle --extra modal \
        python ./scripts/kaggle_cli.py --python -m modal run --detach \
        scripts/modal_kaggle_upload.py::upload_main --run-id <id>

The job stages a directory of symlinks to the volume files plus ``dataset-metadata.json``, uploads the files in parallel
threads with the Kaggle client's own resumable blob upload (``KaggleApi._upload_file``), then creates the dataset
(private) with every file token, and polls the dataset status. It never writes the volume.

Record: tools/ops/records.py's library (records.reserve, gate included) in the main checkout (ARC_VALIDATION_ROOT),
before the spawn, bound (timeout + 60 s) x (CPU + memory rate) x 1.15; settle it with records.py once the dataset is ready.
Needs ARC_UPLOAD_ARM and ARC_UPLOAD_AUTH (2 Nov: was a direct docs/spend.json write, a dropped view edit).
"""

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]
RECORDS = "docs/validation-20260927"
SOURCE = os.environ.get("ARC_UPLOAD_SOURCE", "/assets/flash-input/datasets/richardcsaky/arc-expert-gptq-arc-v1")
DATASET = os.environ.get("ARC_UPLOAD_DATASET", "richardcsaky/arc-expert-gptq-arc-v1")
TITLE = os.environ.get("ARC_UPLOAD_TITLE", "ARC GPTQ-ARC routed-expert overlay v1")
CPU, MEMORY_GIB, TIMEOUT, THREADS = 8.0, 8, 3 * 3600, 8
KAGGLE_VERSION = "2.2.4"  # the local wrapper's client
AUTHORIZATION = ("competition submission with run B incl. a 30-minute public preview; the private "
                 "datasets it needs may be pushed (coordinator brief: GPTQ-ARC overlay to Kaggle via a Modal CPU job "
                 "with the authorized wrapper); CPU only, no GPU lock")

app = modal.App("arc-kaggle-upload", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(f"kaggle=={KAGGLE_VERSION}")
         .add_local_file(__file__, "/root/modal_kaggle_upload.py", copy=True))
if modal.is_local() and os.environ.get("KAGGLE_API_TOKEN"):
    SECRETS = [modal.Secret.from_dict({"KAGGLE_API_TOKEN": os.environ["KAGGLE_API_TOKEN"],
                                       "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", DATASET.split("/")[0])})]
else:
    # The container hydrates the function's objects by position: keep one secret object on both sides (the remote
    # placeholder takes the local secret's id; the first launch without it failed at hydration).
    SECRETS = [modal.Secret.from_dict({})]


def bound_usd(ledger):
    rates = ledger.get("rates_per_second") or {}
    rate = CPU * rates.get("physical_cpu_core", 1.31e-05) + MEMORY_GIB * rates.get("memory_gib", 2.22e-06)
    return (TIMEOUT + 60) * rate * 1.15


def reserve(run_id, extra, records=None):
    """The run record through arc_infra.runops.records (gate, ledger and queue rows); never docs/spend.json, which is
    a generated view. ARC_UPLOAD_ARM and ARC_UPLOAD_AUTH name the arm and explicit approval for this upload."""
    if records is None:
        from arc_infra.runops import records
    arm, auth = os.environ.get("ARC_UPLOAD_ARM"), os.environ.get("ARC_UPLOAD_AUTH")
    if not arm or not auth:
        raise SystemExit("set ARC_UPLOAD_ARM and ARC_UPLOAD_AUTH (explicit approval for this upload)")
    bound = round(bound_usd({}), 2)
    records.reserve(run_id, arm=arm, platform="modal", gpu="none", gpu_count=0, bound_usd=bound,
                    max_seconds=TIMEOUT + 60, authorization=auth, kind="kaggle-upload",
                    what=f"CPU Modal upload of {extra['source']} as the private Kaggle dataset {extra['dataset']}",
                    root=ROOT)
    return bound


@app.function(image=image, cpu=(0.125, CPU), memory=(512, MEMORY_GIB * 1024), timeout=TIMEOUT, retries=0, max_containers=1,
              volumes={"/assets": assets}, secrets=SECRETS)
def upload(source: str, dataset: str, title: str, expect: str = ""):
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        raise RuntimeError("no Kaggle token in the job environment")

    def log(text):
        print(str(text).replace(token, "[REDACTED]"), flush=True)

    os.environ["KAGGLE_CONFIG_DIR"] = tempfile.mkdtemp(prefix="kaggle-config-")
    from kaggle.api.kaggle_api_extended import ApiBlobType, KaggleApi, ResumableUploadContext

    api = KaggleApi()
    api.authenticate()
    src = Path(source)
    names = sorted(p.name for p in src.iterdir() if p.is_file())
    if expect:  # {name: {"size", "sha256"}} (an assets manifest's "files"): every byte checked before any upload
        import hashlib
        want = json.loads(expect)

        def digest(name):
            h = hashlib.sha256()
            with open(src / name, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            return name, (src / name).stat().st_size, h.hexdigest()

        with ThreadPoolExecutor(THREADS) as pool:
            got = {n: (size, h) for n, size, h in pool.map(digest, names)}
        bad = sorted(set(want) ^ set(got)) + [n for n in want if n in got and got[n] != (want[n]["size"], want[n]["sha256"])]
        log(json.dumps({"event": "byte_check", "files": len(got), "expected": len(want), "bad": bad[:20]}))
        if bad:
            raise RuntimeError(f"source differs from the expected manifest: {bad[:5]}")
    total = sum((src / n).stat().st_size for n in names)
    stage = Path(tempfile.mkdtemp(prefix="kaggle-stage-"))
    for name in names:
        (stage / name).symlink_to(src / name)
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"id": dataset, "title": title, "licenses": [{"name": "other"}]}))
    log(json.dumps({"event": "start", "dataset": dataset, "files": len(names), "bytes": total}))
    started = time.time()
    done = {"bytes": 0, "files": 0}

    original = KaggleApi.upload_files

    def parallel_upload_files(self, request, resources, folder, blob_type, upload_context, quiet=False,
                              dir_mode="skip", ignore_patterns=None):
        def one(name):
            import threading
            t0 = time.time()
            size = os.path.getsize(os.path.join(folder, name))
            # 2 Nov (l4-tc942-upload2): one ~10 GiB shard stalled 2 h+ with no error. Each try runs in a daemon thread;
            # a try past 600 s + size at 20 MB/s is abandoned and the file uploaded again (3 tries)
            result = None
            for attempt in range(3):
                box = {}
                worker = threading.Thread(daemon=True, target=lambda: box.update(r=self._upload_file(
                    name, os.path.join(folder, name), blob_type, upload_context, True, resources)))
                worker.start()
                worker.join(600 + size / 20e6)
                if "r" in box and box["r"] is not None:
                    result = box["r"]
                    break
                log(json.dumps({"event": "file_retry", "name": name, "attempt": attempt, "stalled": worker.is_alive()}))
            if result is None:
                raise RuntimeError(f"upload failed: {name}")
            done["bytes"] += size
            done["files"] += 1
            elapsed = time.time() - started
            log(json.dumps({"event": "file", "name": name, "bytes": size, "seconds": round(time.time() - t0, 1),
                            "files_done": done["files"], "gib_done": round(done["bytes"] / 2**30, 2),
                            "mb_per_s": round(done["bytes"] / 1e6 / max(elapsed, 1e-3), 1)}))
            return result

        with ThreadPoolExecutor(THREADS) as pool:
            results = list(pool.map(one, names))
        for result in results:
            request.files.append(self._new_file(result))

    # r2 (29 September): every file uploaded, then CreateDataset answered with a body that was not JSON and the job
    # died. Log the raw answer of any such call, and on a failed create poll the status (the create may land
    # asynchronously), then retry the create with the same uploaded file tokens instead of re-uploading.
    from kaggle.api.kaggle_api_extended import ApiCreateDatasetRequest
    from kagglesdk import kaggle_http_client

    prepare = kaggle_http_client.KaggleHttpClient._prepare_response

    def logged_prepare(self, response_type, http_response):
        try:
            return prepare(self, response_type, http_response)
        except Exception:
            log(json.dumps({"event": "http_error", "url": http_response.url, "status": http_response.status_code,
                            "content_type": http_response.headers.get("Content-Type"),
                            "body": http_response.text[:800]}))
            raise

    kaggle_http_client.KaggleHttpClient._prepare_response = logged_prepare

    def exists():
        try:
            return str(api.dataset_status(dataset))
        except Exception as error:  # noqa: BLE001
            return None if "403" in str(error) or "404" in str(error) else f"error: {error}"

    request = ApiCreateDatasetRequest()
    request.files = []
    with ResumableUploadContext() as upload_context:
        parallel_upload_files(api, request, None, str(stage), ApiBlobType.DATASET, upload_context)
    upload_seconds = round(time.time() - started, 1)
    log(json.dumps({"event": "uploaded", "files": len(request.files), "seconds": upload_seconds}))
    owner, slug = dataset.split("/")
    response = None
    for attempt in range(4):
        create = ApiCreateDatasetRequest()
        create.title, create.slug, create.owner_slug = title, slug, owner
        create.license_name, create.files, create.is_private = "other", request.files, True
        create.category_ids = []
        try:
            with api.build_kaggle_client() as kaggle:
                response = kaggle.datasets.dataset_api_client.create_dataset(create)
            if getattr(response, "error", None) in ("", None):
                break
            log(json.dumps({"event": "create_error", "attempt": attempt, "error": response.error}))
        except Exception as error:  # noqa: BLE001
            log(json.dumps({"event": "create_exception", "attempt": attempt, "error": f"{type(error).__name__}: "
                                                                                      f"{str(error)[:300]}"}))
        for _ in range(10):  # the create may still land: up to 10 minutes before the next attempt
            state = exists()
            log(json.dumps({"event": "exists_check", "attempt": attempt, "status": state}))
            if state is not None:
                break
            time.sleep(60)
        if state is not None:
            break
    KaggleApi.upload_files = original
    record = {"dataset": dataset, "status": getattr(response, "status", None), "error": getattr(response, "error", None),
              "url": getattr(response, "url", None), "ref": getattr(response, "ref", None), "files": len(names),
              "bytes": total, "upload_seconds": upload_seconds, "mb_per_s": round(total / 1e6 / upload_seconds, 1)}
    log(json.dumps({"event": "created", **record}))
    if record["error"]:
        raise RuntimeError(f"dataset create failed: {record['error']}")
    deadline = time.time() + 2 * 3600
    while time.time() < deadline:
        try:
            status = str(api.dataset_status(dataset))
        except Exception as error:  # noqa: BLE001 - the status endpoint can lag the create
            status = f"error: {error}"
        log(json.dumps({"event": "status", "status": status, "seconds": round(time.time() - started, 1)}))
        if status.lower().endswith("ready") or "ready" == status.lower():
            break
        if "error" in status.lower() and "http" not in status.lower():
            break
        time.sleep(60)
    record.update(final_status=status, total_seconds=round(time.time() - started, 1),
                  utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    log(json.dumps({"event": "done", **record}))
    return record


@app.local_entrypoint()
def upload_main(run_id: str):
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Run under scripts/kaggle_cli.py --python (the authorized wrapper) so the token is present")
    bound = reserve(run_id, {"source": f"arc-prolong-assets:{SOURCE[7:]}", "dataset": DATASET})
    record = REPO / RECORDS / f"modal-{run_id}.json"
    data = {"state": "submission_intent", "run_id": run_id, "kind": "kaggle-upload", "charged_bound_usd": bound,
            "source": f"arc-prolong-assets:{SOURCE[7:]}", "dataset": DATASET, "title": TITLE,
            "settle": f"tools/ops/records.py settle {run_id} completed --cause ... once the dataset is ready"}
    record.write_text(json.dumps(data, indent=2))
    expect = os.environ.get("ARC_UPLOAD_EXPECT")  # path of an assets manifest ({"files": {name: {size, sha256}}})
    expect = json.dumps(json.loads(Path(expect).read_text())["files"]) if expect else ""
    call = upload.spawn(SOURCE, DATASET, TITLE, expect)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
