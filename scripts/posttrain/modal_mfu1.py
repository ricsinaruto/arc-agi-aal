"""MFU-1 (22 Oct) kernel tests and benches in the RL trainer's image: each ``;``-separated segment of ``--args`` runs
one shipped script (``rl1_hyper_test.py`` when the segment does not start with a ``.py`` name), on CPU (Triton
interpreter, small shapes) or one GPU (MFU_H_GPU, default B300). Model weights at /assets (read-only).

    uv run --extra modal modal run scripts/posttrain/modal_mfu1.py --run-id mfu-h-t1 [--on-cpu] [--args "bench 16384"]

Output: arc-prolong-runs:/validation/mfu-h/<run id>/log.txt."""
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
RL_PIP = ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0")
VENV = "/workspace/axolotl-venv/bin/python"

app = modal.App("arc-mfu1", include_source=False)
runs = modal.Volume.from_name("arc-prolong-runs")
assets = modal.Volume.from_name("arc-prolong-assets")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .run_commands(f"uv pip install --python {VENV} 'py-spy==0.4.0'")
         .env({"PYTHONUNBUFFERED": "1"}))
for command in RL_PIP:  # the RL trainer image's layers (modal_rl1.py), so the build is cached
    image = image.run_commands(f"uv pip install --python {VENV} " + " ".join(f"'{item}'" for item in command.split()))
for name in ("rl1_hyper.py", "rl1_hyper_test.py", "native_experts.py", "mfu1_probe.py", "rl1_dequant.py", "rl1_experts.py", "rl1_sparse.py",
             "rl1_sparse2.py", "bench_sparse2.py", "rl1_trainer.py"):
    image = image.add_local_file(str(HERE / name), f"/root/mfu/{name}", copy=True)
if modal.is_local():  # MFU-1 expert compaction GPU check (24 Oct); the container re-imports this file from /root
    image = image.add_local_file(str(HERE.parents[1] / "tests" / "test_rl1_expert_compact.py"),
                                 "/root/mfu/test_rl1_expert_compact.py", copy=True)
image = image.add_local_file(__file__, "/root/modal_mfu1.py", copy=True)


def body(run_id, args, env):
    out = Path("/runs/validation/mfu-h") / run_id
    out.mkdir(parents=True, exist_ok=True)
    code = 0
    with open(out / "log.txt", "a") as log:
        for one in args.split(";"):
            parts = one.split()
            script = parts.pop(0) if parts and parts[0].endswith(".py") else "rl1_hyper_test.py"
            proc = subprocess.run([VENV, "-u", f"/root/mfu/{script}", *parts], capture_output=True,
                                  text=True, env={**os.environ, "PYTHONPATH": "/root/mfu", **env})
            text = f"$ {one}\n{proc.stdout[-200000:]}{proc.stderr[-4000:]}exit {proc.returncode}\n"
            print(text, flush=True)
            log.write(text)
            code = code or proc.returncode
    runs.commit()
    return code


GPU = os.environ.get("MFU_H_GPU", "B300")  # correctness-only runs may take a small card


@app.function(image=image, gpu=GPU, timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/runs": runs, "/assets": assets})
def gpu(run_id: str, args: str):
    return body(run_id, args, {})


@app.function(image=image, timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/runs": runs, "/assets": assets})
def cpu(run_id: str, args: str):
    return body(run_id, args, {"TRITON_INTERPRET": "1"})


@app.local_entrypoint()
def main(run_id: str, args: str = "", on_cpu: bool = False):
    started = time.time()
    code = cpu.remote(run_id, args) if on_cpu else gpu.remote(run_id, args)
    print(f"exit {code} after {round(time.time() - started)} s")
