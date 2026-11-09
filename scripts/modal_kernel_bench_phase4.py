"""Modal launcher for the phase-4 kernel microbenchmark (scripts/kernel_bench_phase4.py).

One RTX PRO 6000 container, two passes (separate processes so that a crashing kernel cannot take the other
pass's results with it): ``moe,draft`` (b12x and CUTLASS fused MoE on real NvFp4 experts, then the NvFp4
draft experts) and ``hc,sampler``. Stock stack as in phase 1 (vllm==0.30.0: torch 2.13.0+cu130, triton
3.7.1), with the serving runtime's FlashInfer pinned: flashinfer-python 0.6.16.post3 and its cu130 AOT
cache (flashinfer-jit-cache, for the CUTLASS fused MoE), CuTe DSL 4.6.2. The pinned hc.py and
topk_topp_triton.py baselines come from the extracted serving runtime (ARC_PINNED_SITE).

Usage (from a worktree, ledger in the main checkout):
  ARC_VALIDATION_ROOT=. ARC_PINNED_SITE=<runtime>/usr/local/lib/python3.12/dist-packages \\
      uv run --extra modal modal run scripts/modal_kernel_bench_phase4.py --build-only
  ... modal run --detach scripts/modal_kernel_bench_phase4.py --run-id kernel-bench-phase4-v1-modal-20260928
Results on the arc-prolong-runs volume under validation/<run_id>/ (committed every 60 s).
"""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/nonexistent"))
PINNED_FILES = {"hc.py": "vllm/models/qwen3_8_flash_next/nvidia/ops/hc.py",
                "topk_topp_triton.py": "vllm/v1/sample/ops/topk_topp_triton.py"}
TIMEOUT = 780
CPU_CEILING = 8
MEMORY_CEILING_GIB = 32
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
RECORD_FOLDER = "docs/validation-20260927"
# (tag, per-pass deadline seconds, extra arguments)
PASSES = [
    ("p1-moe-draft", 400, ["--parts", "moe,draft"]),
    ("p2-hc-sampler", 170, ["--parts", "hc,sampler"]),
]

app = modal.App("arc-kernel-bench-phase4", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
_IMPORT_CHECK = (
    "import torch, triton, vllm, flashinfer; print('IMAGE torch', torch.__version__, 'triton', triton.__version__, "
    "'vllm', vllm.__version__, 'flashinfer', flashinfer.__version__)\n"
    "for mod in ['flashinfer_jit_cache', 'cutlass', 'flashinfer.fused_moe', 'vllm._custom_ops']:\n"
    "    try:\n"
    "        m = __import__(mod); print('IMPORT OK', mod, getattr(m, '__version__', ''))\n"
    "    except Exception as exc:\n"
    "        print('IMPORT FAIL', mod, type(exc).__name__, str(exc)[:200])\n"
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .pip_install("vllm==0.30.0")
    .pip_install("flashinfer-python==0.6.16.post3", "nvidia-cutlass-dsl==4.6.2", "nvidia-cutlass-dsl-libs-base==4.6.2",
                 "nvidia-cutlass-dsl-libs-core==4.6.2", "nvidia-cutlass-dsl-libs-cu13==4.6.2",
                 "apache-tvm-ffi==0.1.11", "cuda-python==13.3.1")
    .pip_install("flashinfer-jit-cache==0.6.16.post3+cu130", extra_index_url="https://flashinfer.ai/whl/cu130")
    .run_commands("python -c \"import base64; exec(base64.b64decode('"
                  + __import__("base64").b64encode(_IMPORT_CHECK.encode()).decode() + "'))\" || true")
    .env({"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1",
          "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(REPO / "scripts/kernel_bench_phase4.py"), "/opt/kernel_bench_phase4.py", copy=True)
    .add_local_file(str(REPO / "src/arc_infra/serving_kernels_runtime.py"), "/opt/serving_kernels_runtime.py", copy=True)
)
if modal.is_local():
    for name, relative in PINNED_FILES.items():
        if not (PINNED_SITE / relative).is_file():
            raise FileNotFoundError(f"Set ARC_PINNED_SITE to the extracted serving runtime (missing {relative})")
for name, relative in PINNED_FILES.items():
    image = image.add_local_file(str(PINNED_SITE / relative), f"/opt/pinned/{name}", copy=True)
image = image.add_local_file(__file__, "/root/modal_kernel_bench_phase4.py", copy=True)


def ledger_rate():
    return 0.000842 + CPU_CEILING * 0.0000131 + MEMORY_CEILING_GIB * 0.00000222


def reserve(run_id, kind, seconds):
    """Append a reservation to docs/spend.json under an exclusive lock (as scripts/modal_validation.py)."""
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        bound = (seconds + 60) * ledger_rate() * 1.15
        from arc_infra.runops.records import row_cost
        committed = data["infrastructure_reserve_usd"] + sum(row_cost(r)[0] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": kind,
            "charged_bound_usd": bound, "max_seconds": seconds,
            "gpu": "RTX-PRO-6000", "cpu_ceiling": CPU_CEILING, "memory_ceiling_gib": MEMORY_CEILING_GIB,
            "authorization": ("kernel phase 4 (MTP-3 admission route + fusions); one Modal GPU "
                              "microbenchmark <= (one GPU job at a time)"),
        })
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


@app.function(
    image=image, gpu="RTX-PRO-6000", cpu=(2, CPU_CEILING), memory=(16384, MEMORY_CEILING_GIB * 1024),
    timeout=TIMEOUT, retries=0, max_containers=1, scaledown_window=2, block_network=True,
    volumes={"/assets": assets, "/runs": runs},
)
def bench(run_id: str):
    import subprocess
    import sys
    import threading

    started = time.time()
    output = Path("/runs/validation") / run_id
    if output.exists():
        output.rename(output.with_name(f"{run_id}-lost-{int(time.time())}"))
    output.mkdir(parents=True, exist_ok=False)
    identity = {"run_id": run_id, "started_epoch": started, "gpu": "RTX-PRO-6000", "function_timeout": TIMEOUT,
                "image": "debian-slim/python3.12 + vllm==0.30.0 + flashinfer 0.6.16.post3 (+jit-cache cu130)",
                "pinned_sha256": {n: hashlib.sha256(Path(f"/opt/pinned/{n}").read_bytes()).hexdigest()
                                  for n in PINNED_FILES},
                "helper_sha256": hashlib.sha256(Path("/opt/serving_kernels_runtime.py").read_bytes()).hexdigest(),
                "passes": []}
    stop = threading.Event()

    def committer():
        while not stop.wait(60):
            try:
                runs.commit()
            except Exception as exc:
                print(f"interim volume commit failed: {exc}", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    try:
        with (output / "bench.log").open("wb") as log:
            for tag, budget, extra in PASSES:
                remaining = started + TIMEOUT - 90 - time.time()
                deadline = min(budget, remaining - 40)
                if deadline < 40:
                    identity["passes"].append({"tag": tag, "skipped": "time"})
                    continue
                command = [sys.executable, "-u", "/opt/kernel_bench_phase4.py", "--out", str(output / f"{tag}.json"),
                           "--model-dir", MODEL_DIR, "--helper", "/opt/serving_kernels_runtime.py",
                           "--pinned", "/opt/pinned", "--deadline-seconds", str(int(deadline)), *extra]
                banner = f"=== PASS {tag} deadline {int(deadline)} s at +{time.time() - started:.0f} s\n"
                sys.stdout.write(banner)
                sys.stdout.flush()
                log.write(banner.encode())
                t0 = time.time()
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                def pump(pipe=process.stdout):
                    for line in iter(pipe.readline, b""):
                        log.write(line)
                        log.flush()
                        sys.stdout.buffer.write(line)
                        sys.stdout.flush()

                reader = threading.Thread(target=pump, daemon=True)
                reader.start()
                try:
                    code = process.wait(timeout=deadline + 60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = "killed-after-timeout"
                reader.join(timeout=10)
                identity["passes"].append({"tag": tag, "exit": code, "seconds": time.time() - t0})
                runs.commit()
    finally:
        stop.set()
        identity["elapsed_seconds"] = time.time() - started
        (output / "execution.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return str(output)


@app.local_entrypoint()
def main(run_id: str = "", build_only: bool = False):
    if build_only:
        print("image built; no container started")
        return
    if not run_id:
        raise ValueError("--run-id is required")
    bound = reserve(run_id, "kernel-bench", TIMEOUT)
    record = REPO / RECORD_FOLDER / f"modal-{run_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": "submission_intent", "run_id": run_id, "kind": "kernel-bench-phase4", "charged_bound_usd": bound,
            "platform_policy": "modal-only; one Modal GPU job at a time",
            "ledger": str(ROOT / "docs/spend.json"), "passes": [p[0] for p in PASSES]}
    record.write_text(json.dumps(data, indent=2))
    call = bench.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
