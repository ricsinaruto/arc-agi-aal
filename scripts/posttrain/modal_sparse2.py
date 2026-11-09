"""MFU stage E: ``bench_sparse2.py`` on one Modal B200 (``bench``) or on CPU under the Triton interpreter
(``preflight``: the same script and kernels on tiny shapes). The RL trainer image (Axolotl venv, Triton 3.6.0).

    uv run --extra modal modal run --detach scripts/posttrain/modal_sparse2.py --run-id mfu-e-v1 [--preflight]

Streams the script's JSON lines to the log and to arc-prolong-runs:/validation/mfu-e/<run id>/bench.jsonl, which ends with
"RUNOPS END ok" on success: give the watch plan "logs": ["modal:arc-prolong-runs:validation/mfu-e/<run id>/bench.jsonl"]."""
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
RL_PIP = ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0")
VENV = "/workspace/axolotl-venv/bin/python"

app = modal.App("arc-mfu-e", include_source=False)
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .run_commands(f"uv pip install --python {VENV} 'py-spy==0.4.0'")
         .env({"PYTHONUNBUFFERED": "1"}))
for command in RL_PIP:  # the RL trainer image's layers (modal_rl1.py), so the build is cached
    image = image.run_commands(f"uv pip install --python {VENV} " + " ".join(f"'{item}'" for item in command.split()))
for name in ("rl1_sparse.py", "rl1_sparse2.py", "bench_sparse2.py"):
    image = image.add_local_file(str(HERE / name), f"/root/mfu/{name}", copy=True)
image = image.add_local_file(__file__, "/root/modal_sparse2.py", copy=True)


def body(run_id, args, env):
    import os
    out = Path("/runs/validation/mfu-e") / run_id
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "bench.jsonl", "a")
    proc = subprocess.Popen([VENV, "/root/mfu/bench_sparse2.py", *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env={**os.environ, **env})
    last = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
        if time.time() - last > 60:
            runs.commit()
            last = time.time()
    code = proc.wait()
    log.write(f'{{"exit": {code}}}\n')
    if code == 0:  # the shared watcher's clean-end marker (plan "logs": this file on the volume)
        log.write("RUNOPS END ok\n")
    log.close()
    runs.commit()
    return code


@app.function(image=image, gpu="B200", timeout=2400, retries=0, max_containers=1, volumes={"/runs": runs})
def bench(run_id: str, mode: str = "bench"):
    return body(run_id, ["--mode", *mode.split()], {})


@app.function(image=image, gpu="B300", timeout=2400, retries=0, max_containers=1, volumes={"/runs": runs})
def bench_b300(run_id: str, mode: str = "bench"):
    return body(run_id, ["--mode", *mode.split()], {})


@app.function(image=image, timeout=1800, retries=0, max_containers=1, volumes={"/runs": runs})
def preflight(run_id: str, mode: str = "test"):
    return body(run_id, ["--mode", *mode.split()], {"TRITON_INTERPRET": "1"})


@app.local_entrypoint()
def main(run_id: str, preflight_only: bool = False, mode: str = "bench", gpu: str = "B200"):
    fn = preflight if preflight_only else bench_b300 if gpu == "B300" else bench
    code = fn.remote(run_id, mode)
    print(f"exit {code}")
