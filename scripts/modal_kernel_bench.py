"""Modal launcher for the phase-1 dense-GEMM decode microbenchmark (scripts/kernel_bench_gpu.py).

One RTX PRO 6000 container runs the benchmark in five passes, each a separate process so that a
crashing kernel (illegal address, compiler abort) cannot take the earlier passes' results with it.
Stock stack: vllm==0.30.0 from PyPI, which pins torch==2.13.0 (CUDA 13) and triton 3.7.1, the
same torch/triton as the pinned serving runtime (vLLM 0.1.dev20073+g8e685d198, torch 2.13.0).

Usage (from a worktree, ledger in the main checkout):
  ARC_VALIDATION_ROOT=. uv run --extra modal modal run \
      scripts/modal_kernel_bench.py --build-only            # image build only, no container
  ARC_VALIDATION_ROOT=. uv run --extra modal modal run --detach \
      scripts/modal_kernel_bench.py --run-id kernel-bench-phase1-v1-modal-20260928
Live progress: `modal app logs <app-id>`; results on the arc-prolong-runs volume under
validation/<run_id>/ (committed every 60 s).
"""

import base64
import json
import os
import time
from pathlib import Path

import modal

# The ledger (docs/spend.json) lives in the main checkout (ARC_VALIDATION_ROOT); the launch
# record is written next to the phase-1 documents of this checkout.
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]
TIMEOUT = 1800
CPU_CEILING = 8
MEMORY_CEILING_GIB = 32
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
RECORD_FOLDER = "docs/validation-20260927"
# (tag, per-pass deadline seconds, extra arguments)
PASSES = [
    ("p1-baseline", 240, ["--bandwidth", "--methods",
                          "bf16_linear,bf16_linear_altblas,bf16_mm_kmajor,fp8_scaled_mm_tensor,fp8_scaled_mm_rowwise"]),
    ("p2-vllm-fp8", 240, ["--act-quant", "--methods",
                          "vllm_cutlass_fp8_tensor,vllm_cutlass_fp8_channel,vllm_cutlass_fp8_tensor+dynq,"
                          "vllm_cutlass_fp8_channel+dynq,marlin_w8a16_fp8"]),
    ("p3-triton", 330, ["--methods", "triton_bf16,triton_w8a16_fp8,triton_w8a16_int8"]),
    ("p4-nvfp4", 150, ["--methods", "vllm_cutlass_nvfp4,vllm_cutlass_nvfp4+actq"]),
    ("p5-skinny", 200, ["--skinny-compile-budget", "150", "--methods", "cute_skinny_bf16"]),
]

app = modal.App("arc-kernel-bench", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
_IMPORT_CHECK = (
    "import torch, triton, vllm; print('IMAGE torch', torch.__version__, 'cuda', torch.version.cuda, "
    "'triton', triton.__version__, 'vllm', vllm.__version__)\n"
    "for mod in ['vllm._custom_ops', 'vllm.model_executor.layers.quantization.utils.marlin_utils_fp8', "
    "'vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm', 'safetensors', 'cutlass', 'quack']:\n"
    "    try:\n"
    "        __import__(mod); print('IMPORT OK', mod)\n"
    "    except Exception as exc:\n"
    "        print('IMPORT FAIL', mod, type(exc).__name__, str(exc)[:200])\n"
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")  # Triton builds its launcher stub with the system C compiler
    .pip_install("vllm==0.30.0")
    # Dockerfile RUN lines are single-line: ship the import check base64-encoded (it never fails the build).
    .run_commands("python -c \"import base64; exec(base64.b64decode('"
                  + base64.b64encode(_IMPORT_CHECK.encode()).decode() + "'))\" || true")
    .env({"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1",
          "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(REPO / "scripts/kernel_bench_gpu.py"), "/opt/kernel_bench_gpu.py", copy=True)
    .add_local_file(__file__, "/root/modal_kernel_bench.py", copy=True)
)


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
        committed = data["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": kind,
            "charged_bound_usd": bound, "max_seconds": seconds,
            "gpu": "RTX-PRO-6000", "cpu_ceiling": CPU_CEILING, "memory_ceiling_gib": MEMORY_CEILING_GIB,
            "authorization": ("custom-kernel throughput work, phase 1; exactly one short "
                              "Modal GPU microbenchmark (one GPU job at a time)"),
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
    identity = {"run_id": run_id, "started_epoch": started, "gpu": "RTX-PRO-6000",
                "function_timeout": TIMEOUT, "image": "debian-slim/python3.12 + vllm==0.30.0", "passes": []}

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
                remaining = started + TIMEOUT - 150 - time.time()
                deadline = min(budget, remaining - 60)
                if deadline < 45:
                    identity["passes"].append({"tag": tag, "skipped": "time"})
                    continue
                command = [sys.executable, "-u", "/opt/kernel_bench_gpu.py", "--out", str(output / f"{tag}.json"),
                           "--model-dir", MODEL_DIR, "--deadline-seconds", str(int(deadline)), *extra]
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
                    code = process.wait(timeout=deadline + 150)
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
    data = {"state": "submission_intent", "run_id": run_id, "kind": "kernel-bench", "charged_bound_usd": bound,
            "platform_policy": "modal-only; one Modal GPU job at a time",
            "ledger": str(ROOT / "docs/spend.json"), "passes": [p[0] for p in PASSES]}
    record.write_text(json.dumps(data, indent=2))
    call = bench.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
