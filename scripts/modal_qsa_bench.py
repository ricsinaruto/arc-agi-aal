"""Modal launcher for the QSA prefill microbenchmark (scripts/qsa_prefill_bench.py; kernel work, phase 7).

One RTX PRO 6000 container, no server: the stock vllm==0.30.0 image (torch 2.13 / Triton 3.7.1, the serving runtime's
versions, as scripts/modal_kernel_bench.py), the pinned ``nvidia/ops/qsa.py`` (its sha256 checked here and in the
container) and ``src/arc_infra/kv8_runtime.py`` (the served ``arc_kv8``).

Usage (from the worktree; ledger in the main checkout):
  ARC_VALIDATION_ROOT=. ARC_PINNED_QSA=<pinned ops/qsa.py> uv run --extra modal modal run \
      --detach scripts/modal_qsa_bench.py --run-id qsa-prefill-bench-v1-modal-20260929
Results on the arc-prolong-runs volume under validation/<run_id>/.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]
TIMEOUT = 900
CPU_CEILING = 8
MEMORY_CEILING_GIB = 32
RECORD_FOLDER = "docs/validation-20260927"
PINNED_QSA_SHA256 = "c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f"
PINNED_QSA = Path(os.environ.get(
    "ARC_PINNED_QSA", "/tmp/arc-pinned-runtime/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"))


def ledger_rate():
    return 0.000842 + CPU_CEILING * 0.0000131 + MEMORY_CEILING_GIB * 0.00000222


def bound_usd(seconds=TIMEOUT):
    return (seconds + 60) * ledger_rate() * 1.15


app = modal.App("arc-qsa-bench", include_source=False)
runs = modal.Volume.from_name("arc-prolong-runs")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .pip_install("vllm==0.30.0")
    .env({"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(REPO / "scripts/qsa_prefill_bench.py"), "/opt/qsa_prefill_bench.py", copy=True)
    .add_local_file(str(REPO / "src/arc_infra/kv8_runtime.py"), "/opt/kv8_runtime.py", copy=True)
    .add_local_file(str(PINNED_QSA), "/opt/pinned_qsa.py", copy=True)
    # include_source=False: the container imports this launcher from /root (v1 omitted it and every container failed
    # with ModuleNotFoundError before running).
    .add_local_file(__file__, "/root/modal_qsa_bench.py", copy=True)
)


def reserve(run_id, seconds):
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        bound = bound_usd(seconds)
        from arc_infra.runops.records import row_cost
        committed = data["infrastructure_reserve_usd"] + sum(row_cost(r)[0] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": "kernel-bench", "charged_bound_usd": bound,
            "max_seconds": seconds, "gpu": "RTX-PRO-6000", "cpu_ceiling": CPU_CEILING,
            "memory_ceiling_gib": MEMORY_CEILING_GIB,
            "authorization": "(via coordinator): QSA prefill microbenchmark, test slot"})
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


@app.function(image=image, gpu="RTX-PRO-6000", cpu=(2, CPU_CEILING), memory=(16384, MEMORY_CEILING_GIB * 1024),
              timeout=TIMEOUT, retries=0, max_containers=1, scaledown_window=2, block_network=True,
              volumes={"/runs": runs})
def bench(run_id: str):
    import subprocess
    import sys

    started = time.time()
    output = Path("/runs/validation") / run_id
    if output.exists():
        output.rename(output.with_name(f"{run_id}-lost-{int(time.time())}"))
    output.mkdir(parents=True, exist_ok=False)
    identity = {"run_id": run_id, "started_epoch": started, "gpu": "RTX-PRO-6000", "function_timeout": TIMEOUT,
                "pinned_qsa_sha256": hashlib.sha256(Path("/opt/pinned_qsa.py").read_bytes()).hexdigest()}
    try:
        if identity["pinned_qsa_sha256"] != PINNED_QSA_SHA256:
            raise RuntimeError("pinned ops/qsa.py hash mismatch")
        command = [sys.executable, "-u", "/opt/qsa_prefill_bench.py", "--out", str(output / "qsa-prefill.json"),
                   "--kv8", "/opt/kv8_runtime.py", "--pinned-qsa", "/opt/pinned_qsa.py", "--budget-seconds", "170"]
        with (output / "bench.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT - 120)
        identity["exit"] = completed.returncode
    except Exception as exc:  # noqa: BLE001
        identity["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    finally:
        identity["elapsed_seconds"] = time.time() - started
        (output / "execution.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return str(output)


@app.function(image=image, cpu=1, memory=2048, timeout=300, retries=0, block_network=True)
def probe():
    """CPU-only container: this module imported in the image (v1 failed there), the bench and the pinned kernel
    files present with their hashes."""
    import importlib

    importlib.import_module("modal_qsa_bench")
    return {name: hashlib.sha256(Path(f"/opt/{name}").read_bytes()).hexdigest()
            for name in ("qsa_prefill_bench.py", "kv8_runtime.py", "pinned_qsa.py")}


@app.local_entrypoint()
def main(run_id: str = "", build_only: bool = False, probe_only: bool = False):
    if build_only:
        print("image built; no container started")
        return
    if probe_only:
        print(json.dumps({"probe": probe.remote()}))
        return
    if not run_id:
        raise ValueError("--run-id is required")
    if hashlib.sha256(PINNED_QSA.read_bytes()).hexdigest() != PINNED_QSA_SHA256:
        raise RuntimeError(f"{PINNED_QSA} is not the pinned ops/qsa.py")
    bound = reserve(run_id, TIMEOUT)
    record = REPO / RECORD_FOLDER / f"modal-{run_id}.json"
    data = {"state": "submission_intent", "run_id": run_id, "kind": "kernel-bench", "charged_bound_usd": bound,
            "platform_policy": "modal-only; test slot (/tmp/arc-gpu-lock)", "ledger": str(ROOT / "docs/spend.json")}
    record.write_text(json.dumps(data, indent=2))
    call = bench.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
