"""CPU-only Modal fetch of a reviewed candidate checkpoint into the assets volume (Holo4-27B, 29 September).

    ARC_VALIDATION_ROOT=. ARC_CANDIDATE=holo4-27b-nvfp4 uv run --extra modal modal run \
        --detach scripts/modal_candidate_weights.py::fetch_main --run-id <id>
    ARC_CANDIDATE=holo4-27b-nvfp4 uv run --extra modal modal run scripts/modal_candidate_weights.py::delete_main \
        --run-id <id>

    ARC_CANDIDATE=qwen35-4b ARC_CANDIDATE_CHECK=train-probe ... fetch_main --run-id <id>   (fetch, then the CPU dry run
        of arena.train_probe_job in the serving runtime's Python: 16 CPUs, 96 GiB)

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
CHECK = os.environ.get("ARC_CANDIDATE_CHECK", "")
if CHECK not in ("", "train-probe"):
    raise ValueError("ARC_CANDIDATE_CHECK is train-probe or unset")
CPU, MEMORY_GIB, TIMEOUT = (16.0, 96, 3600) if CHECK else (8.0, 16, 3600)
RUNTIME = "/assets/flash-input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
FLA_WHEEL = "fla_core-0.5.2-py3-none-any.whl"  # docs/validation-20261011/build_rl0.py downloads and pins it
STORAGE_DAYS = 7  # kept for the smoke and the one-hour arena, then deleted unless the result is promising
AUTHORIZATION = ("test Holo4-27B (Hcompany/Holo4-27B-NVFP4, CC BY-NC fine for testing) with the "
                 "hero-v2 arena setup; CPU fetch of the pinned checkpoint into the assets volume (no GPU lock)")
if CANDIDATE == "qwen35-4b":
    RECORDS = "docs/validation-20261011"
    AUTHORIZATION = ("RL restart step 0 on Qwen/Qwen3.5-4B (BF16); CPU fetch of the pinned checkpoint "
                     "into the assets volume and the CPU dry run of the training-cost probe (no GPU)")

if CANDIDATE == "qwen36-35b-a3b-fp8":
    RECORDS = "docs/validation-20261011"
    AUTHORIZATION = ("RL step 0 on Qwen3.6-35B-A3B (\"run the same 2 tests (thinking on/off) with "
                     "qwen3.6-35b-a3b\"); CPU fetch of the pinned official FP8 checkpoint into the assets volume (no GPU)")
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
    if CHECK:
        image = (image.add_local_file(str(REPO / "src/arena/train_probe_job.py"), "/root/train_probe_job.py", copy=True)
                 .add_local_file(str(REPO / "dist/.cache" / FLA_WHEEL), "/root/" + FLA_WHEEL, copy=True))


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
        from arc_infra.runops.records import row_cost
        committed = data["infrastructure_reserve_usd"] + sum(row_cost(r)[0] for r in data["runs"])
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


FACTS = """
import json, sys, inspect, importlib.metadata as md
import torch, transformers
out = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__}
for name in ("accelerate", "einops", "triton", "safetensors"):
    try:
        out[name] = md.version(name)
    except Exception as exc:
        out[name] = repr(exc)[:120]
out["class"] = hasattr(transformers, "Qwen3_5ForConditionalGeneration")
try:
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    src = inspect.getsource(m)
    out["kernel_lines"] = [l.strip()[:160] for l in src.splitlines() if "fla" in l.split("#")[0] and "flash_attention" not in l and "flatten" not in l][:12]
    out["gradient_checkpointing"] = "gradient_checkpointing" in src
except Exception as exc:
    out["modeling_error"] = repr(exc)[:300]
try:
    import fla.ops.gated_delta_rule as g
    out["fla"] = {"file": g.__file__, "chunk": hasattr(g, "chunk_gated_delta_rule")}
except Exception as exc:
    out["fla"] = repr(exc)[:300]
print("FACTS " + json.dumps(out))
"""


def runtime_env(root, extra=()):
    site = root / "usr/local/lib/python3.12/dist-packages"
    cuda = root / "usr/local/cuda-13.0"
    libs = [cuda / "targets/x86_64-linux/lib", cuda / "lib64", site / "torch/lib", *sorted((site / "nvidia").glob("*/lib"))]
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), *map(str, extra)]), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "LD_LIBRARY_PATH": os.pathsep.join(str(p) for p in libs if p.is_dir())}


def train_probe_check(lengths=(64, 128)):
    """The serving runtime's Python packages, extracted as the server does (layers in order, whiteouts applied), then
    the probe job on CPU with the fetched weights: imports, model class, one full training step per length."""
    import shutil
    import subprocess
    import sys
    import zipfile

    root, started, result = Path("/tmp/runtime"), time.time(), {}
    root.mkdir(parents=True)
    layers = sorted(Path(RUNTIME).glob("layer-*.tar.gz*"), key=lambda p: int(p.name.split("-")[1]))
    for layer in layers:
        subprocess.run(["tar", "-xzf", str(layer), "-C", str(root), "--no-same-owner", "--exclude=*/__pycache__",
                        "--exclude=*.pyc"], check=True)
        for marker in list(root.rglob(".wh.*")):
            target = marker.parent / marker.name[4:]
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
    result["runtime"] = {"layers": [p.name[:8] for p in layers], "extract_seconds": round(time.time() - started, 1)}
    print(json.dumps(result), flush=True)
    fla = Path("/tmp/fla")
    with zipfile.ZipFile("/root/" + FLA_WHEEL) as archive:
        archive.extractall(fla)
    for name, extra in (("facts", ()), ("facts_with_fla", (fla,))):
        done = subprocess.run([sys.executable, "-c", FACTS], env=runtime_env(root, extra), capture_output=True,
                              text=True, timeout=900)
        line = next((x[6:] for x in done.stdout.splitlines() if x.startswith("FACTS ")), None)
        result[name] = json.loads(line) if line else {"returncode": done.returncode, "tail": (done.stdout + done.stderr)[-1500:]}
        print(json.dumps({name: result[name]}), flush=True)
    out = Path("/tmp/probe/fp32.json")
    job = subprocess.run([sys.executable, "-u", "/root/train_probe_job.py", "--model-dir", TARGET, "--out", str(out),
                          "--storage", "fp32", "--lengths", *map(str, lengths), "--deadline-epoch",
                          str(time.time() + 1500), "--device", "cpu"], env=runtime_env(root), capture_output=True,
                         text=True, timeout=2100)
    result["job"] = {"returncode": job.returncode, "tail": (job.stdout + job.stderr)[-3000:],
                     "report": json.loads(out.read_text()) if out.is_file() else None}
    result["seconds"] = round(time.time() - started, 1)
    print(json.dumps({"job": {k: v for k, v in result["job"].items() if k != "report"}}), flush=True)
    return result



@app.function(image=image, cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TIMEOUT, retries=0, max_containers=1,
              volumes={"/assets": assets})
def fetch(check: str = ""):
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
    if check == "train-probe":
        record["train_probe_check"] = train_probe_check()
        (target / f"train-probe-check-{int(started)}.json").write_text(json.dumps(record["train_probe_check"], indent=1))
        assets.commit()
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
            "check": CHECK or None, "cpu": CPU, "memory_gib": MEMORY_GIB,
            "target": f"arc-prolong-assets:{TARGET[7:]}", "ledger": str(ROOT / "docs/spend.json")}
    record.write_text(json.dumps(data, indent=2))
    call = fetch.spawn(CHECK)
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
