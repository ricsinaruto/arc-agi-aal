"""Modal launcher: QSA indexer top-k / scoring micro-benchmark in the pinned serving runtime (throughput arm, 4 October).

One RTX PRO 6000, no server: the pinned runtime extracted from the assets volume (as scripts/modal_expert3bit.py) and
``scripts/qsa_topk_bench.py`` run with its python. Results on arc-prolong-runs under research/<run_id>/.

    ARC_VALIDATION_ROOT=. uv run --extra modal modal run --detach scripts/modal_topk_bench.py \
        --run-id tp-topk-bench-v1-modal-20261004
"""
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or HERE).resolve()
RECORDS = HERE / "docs/validation-20261003"
TIMEOUT = 1200
CPU, RAM_GIB = 8, 64
RUNTIME_DIR = "/assets/flash-input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
RT = "/tmp/qwen38-flash-next-vllm-runtime"
WHITEOUTS = {8: ("usr/local/cuda-13.0/targets/x86_64-linux/lib/.wh.libcudart.so.13.0.88",),
             17: ("tmp/.wh.requirements-cuda.txt", "tmp/.wh.common.txt")}

app = modal.App("arc-tp-topk-bench", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential")
         .add_local_file(str(HERE / "scripts/qsa_topk_bench.py"), "/opt/qsa_topk_bench.py", copy=True)
         .add_local_file(__file__, "/root/modal_topk_bench.py", copy=True))


def reserve(run_id):
    import fcntl

    rate = 0.000842 + CPU * 0.0000131 + RAM_GIB * 0.00000222
    bound = (TIMEOUT + 60) * rate * 1.15
    path = ROOT / "docs/spend.json"
    with path.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.load(f)
        if any(r["run_id"] == run_id for r in d["runs"]):
            raise ValueError("Already reserved")
        used = d["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in d["runs"])
        if used + bound > d["cap_usd"]:
            raise RuntimeError(f"Cap: {used}+{bound}>{d['cap_usd']}")
        d["runs"].append({"run_id": run_id, "state": "reserved", "kind": "research-pilot", "charged_bound_usd": bound,
                          "max_seconds": TIMEOUT, "gpu": "RTX-PRO-6000", "cpu_ceiling": CPU,
                          "memory_ceiling_gib": RAM_GIB,
                          "authorization": "throughput research arm, up to two parallel Modal GPUs "
                                           "and in total (run ids tp-*); + (arm total), 1 GPU at a time"})
        f.seek(0)
        json.dump(d, f, indent=1)
        f.write("\n")
        f.truncate()
    return bound


def extract_runtime():
    import shutil
    import subprocess

    root = Path(RT)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    manifest = json.loads((Path(RUNTIME_DIR) / "runtime-manifest.json").read_text())
    for row in sorted(manifest["selected_layers"], key=lambda r: int(r["index"])):
        layer = Path(RUNTIME_DIR) / row["file"]
        if not layer.is_file():
            layer = Path(RUNTIME_DIR) / f"{row['file']}.blob"
        subprocess.run(["tar", "-xzf", str(layer), "-C", str(root), "--no-same-owner", "--exclude=*/__pycache__",
                        "--exclude=*/__pycache__/*", "--exclude=*.pyc"], check=True)
        for rel in WHITEOUTS.get(int(row["index"]), ()):
            marker = root / rel
            target = marker.with_name(marker.name[4:])
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            marker.unlink(missing_ok=True)


def runtime_env():
    import subprocess

    site = Path(RT) / "usr/local/lib/python3.12/dist-packages"
    cuda_home = Path(RT) / "usr/local/cuda-13.0"
    driver = None
    for line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout.splitlines():
        if "libcuda.so.1" in line and "=>" in line and "stub" not in line:
            driver = Path(line.split("=>", 1)[1].strip()).resolve().parent
    libs = [cuda_home / "targets/x86_64-linux/lib", cuda_home / "lib64", site / "torch/lib",
            *sorted(p for p in (site / "nvidia").glob("*/lib") if p.is_dir()),
            *([driver] if driver else []), Path("/usr/local/nvidia/lib64"), Path("/usr/lib/x86_64-linux-gnu")]
    env = os.environ.copy()
    for d in ("/tmp/rtcache/triton", "/tmp/rttmp"):
        Path(d).mkdir(parents=True, exist_ok=True)
    env.update({"PYTHONPATH": os.pathsep.join([str(site / "nvidia_cutlass_dsl/dsl_packages"), str(site)]),
                "PATH": os.pathsep.join([str(cuda_home / "bin"), str(Path(RT) / "usr/local/bin"), env.get("PATH", "")]),
                "LD_LIBRARY_PATH": os.pathsep.join(str(p) for p in libs if p.is_dir()),
                "CUDA_HOME": str(cuda_home), "TRITON_CACHE_DIR": "/tmp/rtcache/triton", "TMPDIR": "/tmp/rttmp",
                "VLLM_NO_USAGE_STATS": "1", "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


@app.function(image=image, gpu="RTX-PRO-6000", cpu=(0.125, CPU), memory=(128, RAM_GIB * 1024), timeout=TIMEOUT,
              retries=0, max_containers=1, volumes={"/assets": assets, "/runs": runs})
def bench(run_id: str):
    import subprocess
    import sys
    import time
    import traceback

    started = time.time()
    out = Path("/runs/research") / run_id
    out.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "kind": "tp-topk-bench"}
    try:
        extract_runtime()
        record["extract_seconds"] = round(time.time() - started, 1)
        cmd = [sys.executable, "-u", "/opt/qsa_topk_bench.py", "--out", str(out / "bench.json")]
        with (out / "bench.log").open("w") as log:
            proc = subprocess.run(cmd, env=runtime_env(), stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT - 300)
        record["returncode"] = proc.returncode
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    record["elapsed_seconds"] = round(time.time() - started, 1)
    (out / "execution.json").write_text(json.dumps(record, indent=2))
    runs.commit()
    return record


@app.local_entrypoint()
def main(run_id: str):
    bound = reserve(run_id)
    call = bench.spawn(run_id)
    record = {"run_id": run_id, "kind": "tp-topk-bench", "function_call_id": call.object_id, "charged_bound_usd": bound,
              "timeout_seconds": TIMEOUT, "state": "submitted"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
