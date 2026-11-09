"""MFU stage G (RL arm, 21 Oct): the Gated DeltaNet layers on one B200. ``bench_gdn.py`` builds one real GDN layer
(weights from the Flash checkpoint, a rank-16 LoRA on its projections as the trainer has it), a 64K-token prefix state
and a 16-32K window, and times/compares the model's path against ``rl1_gdn.py`` (forward, backward, with the
checkpoint recompute the trainer runs).

    uv run --extra modal modal run --detach scripts/posttrain/modal_mfu_g.py --run-id mfu-g-v1 [--mode bench] [--cpu]

``--cpu``: the same script on CPU (source dump / tiny-shape checks under the Triton interpreter).
Output: arc-prolong-runs:/validation/mfu-g/<run id>/log.txt, ending "RUNOPS END ok" on success (watch plan "logs":
["modal:arc-prolong-runs:validation/mfu-g/<run id>/log.txt"])."""
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
RL_PIP = ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0")
VENV = "/workspace/axolotl-venv/bin/python"

app = modal.App("arc-mfu-g", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .run_commands(f"uv pip install --python {VENV} 'py-spy==0.4.0'")
         .env({"PYTHONUNBUFFERED": "1"}))
for command in RL_PIP:  # the RL trainer image's layers (modal_rl1.py), so the build is cached
    image = image.run_commands(f"uv pip install --python {VENV} " + " ".join(f"'{item}'" for item in command.split()))
for name in ("rl1_gdn.py", "bench_gdn.py", "rl1_gdn_test.py", "gdn_compare.py"):
    image = image.add_local_file(str(HERE / name), f"/root/mfu/{name}", copy=True)
image = image.add_local_file(__file__, "/root/modal_mfu_g.py", copy=True)


def body(run_id, mode, env):
    out = Path("/runs/validation/mfu-g") / run_id
    out.mkdir(parents=True, exist_ok=True)
    Path("/kaggle").mkdir(exist_ok=True)
    if not Path("/kaggle/input").exists():
        Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    log = open(out / "log.txt", "a")
    proc = subprocess.Popen([VENV, "-u", "/root/mfu/bench_gdn.py", *mode.split()], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONPATH": "/root/mfu", "MFU_G_OUT": str(out),
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **env})
    last = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
        if time.time() - last > 60:
            runs.commit()
            last = time.time()
    code = proc.wait()
    log.write(f"exit {code}\n")
    if code == 0:  # the shared watcher's clean-end marker (plan "logs": this file on the volume)
        log.write("RUNOPS END ok\n")
    log.close()
    runs.commit()
    return code


@app.function(image=image, gpu="B200", timeout=2400, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/assets": assets, "/runs": runs})
def bench(run_id: str, mode: str = "bench", fast: bool = False):
    return body(run_id, mode, {"MFU_G_FAST": "1"} if fast else {})


@app.function(image=image, gpu="B300", timeout=2400, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/assets": assets, "/runs": runs})
def bench_b300(run_id: str, mode: str = "bench", fast: bool = False):
    return body(run_id, mode, {"MFU_G_FAST": "1"} if fast else {})


@app.function(image=image, timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/assets": assets, "/runs": runs})
def cpu(run_id: str, mode: str = "src"):
    return body(run_id, mode, {"TRITON_INTERPRET": "1"} if mode.startswith("test") else {})


@app.local_entrypoint()
def main(run_id: str, mode: str = "bench", on_cpu: bool = False, fast: bool = False, gpu: str = "B200"):
    fn = bench_b300 if gpu == "B300" else bench
    print(f"exit {cpu.remote(run_id, mode) if on_cpu else fn.remote(run_id, mode, fast)}")
