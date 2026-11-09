"""CPU-only Modal build of a stock vLLM runtime for candidate models our Flash fork cannot serve (the
Nemotron 3.5 screen; Bolt's card needs nightly 77154ca6 or newer, Super VL's names nightly 3f1bcfd8 or v0.32.0).

    ARC_VALIDATION_ROOT=. modal run --detach scripts/modal_stock_vllm.py::build_main --run-id <id>

The runtime is one folder, /tmp/stockvllm: a standalone CPython 3.12 (uv's python-build-standalone), a venv over it
with the pinned nightly vLLM wheel and b12x (Bolt's linear/MoE backend), and the kernel cache ``vllm download-kernels``
fills (the RTX run has no internet). It must be extracted at the same path (the venv names its base by path). Written
as ``stockvllm.tar.gz`` + ``stock-vllm.json`` (versions, sha256, sizes) to
``arc-prolong-assets:/flash-input/datasets/richardcsaky/arc-stock-vllm-<commit 8>/``. Ledger: the main checkout's
docs/spend.json, as scripts/modal_candidate_weights.py.
"""

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
COMMIT = "3f1bcfd82bb2e93fcd364fc7a5491718453a77e2"
WHEEL = ("https://wheels.vllm.ai/3f1bcfd82bb2e93fcd364fc7a5491718453a77e2/"
         "vllm-0.32.0rc1.dev190%2Bg3f1bcfd82-cp38-abi3-manylinux_2_28_x86_64.whl")
EXTRA = ["b12x==1.5.1"]
DATASET = f"richardcsaky/arc-stock-vllm-{COMMIT[:8]}"
TARGET = f"/assets/flash-input/datasets/{DATASET}"
CPU, MEMORY_GIB, TIMEOUT = 8.0, 32, 3600
AUTHORIZATION = ("build the Nemotron serving path (stock vLLM nightly or v0.32 in its "
                 "own venv in the RTX bundle) and screen Nemotron 3.5 Bolt and Super VL; CPU build, no GPU")

app = modal.App("arc-stock-vllm", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("pigz", "curl", "git", "ca-certificates")
         .pip_install("uv==0.9.5")
         .add_local_file(__file__, "/root/modal_stock_vllm.py", copy=True))


def run(cmd, env=None, timeout=3000):
    import subprocess

    print("$ " + " ".join(cmd), flush=True)
    done = subprocess.run(cmd, env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=timeout)
    tail = (done.stdout + done.stderr)[-6000:]
    print(tail, flush=True)
    if done.returncode:
        raise RuntimeError(f"{cmd[:3]} exited {done.returncode}: {tail[-1500:]}")
    return done.stdout


@app.function(image=image, cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TIMEOUT, retries=0, max_containers=1,
              volumes={"/assets": assets})
def build():
    import glob
    import hashlib
    import shutil

    started = time.time()
    root = Path("/tmp/stockvllm")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    run(["uv", "python", "install", "3.12", "--install-dir", str(root / "py")])
    base = Path(sorted(glob.glob(str(root / "py/cpython-3.12*/bin/python3.12")))[0])
    (root / "python").symlink_to(base.parent.parent.relative_to(root))
    run(["uv", "venv", "--seed", "--python", str(root / "python/bin/python3.12"), str(root / "venv")])
    py = str(root / "venv/bin/python")
    env = {"XDG_CACHE_HOME": str(root / "cache"), "VLLM_CACHE_ROOT": str(root / "cache/vllm"),
           "UV_CACHE_DIR": "/tmp/uv-cache"}
    run(["uv", "pip", "install", "--python", py, f"vllm @ {WHEEL}", *EXTRA], env, timeout=2400)
    versions = json.loads(run([py, "-c", (
        "import json, importlib.metadata as m; names = ['vllm', 'torch', 'triton', 'flashinfer-python', 'b12x', "
        "'transformers', 'nvidia-cutlass-dsl', 'xgrammar']; out = {}\n"
        "for n in names:\n"
        "    try: out[n] = m.version(n)\n"
        "    except Exception as e: out[n] = None\n"
        "print(json.dumps(out))")], env).strip().splitlines()[-1])
    kernels = {}
    try:
        # what `vllm download-kernels` runs (its CLI cannot start on a host without a GPU): FlashInfer's prebuilt
        # JIT cache for this CUDA, else its cubin wheel
        fenv = {**env, "FLASHINFER_DISABLE_VERSION_CHECK": "1"}
        try:
            kernels["out"] = run([py, "-m", "flashinfer", "download-kernels"], fenv, timeout=1800)[-3000:]
        except RuntimeError as exc:
            kernels["jit_cache_error"] = str(exc)[-1500:]
            kernels["out"] = run([py, "-m", "flashinfer", "install-cubin-wheel"], fenv, timeout=1800)[-3000:]
        versions.update(json.loads(run([py, "-c", (
            "import json, importlib.metadata as m; out = {}\n"
            "for n in ['flashinfer-jit-cache', 'flashinfer-cubin']:\n"
            "    try: out[n] = m.version(n)\n"
            "    except Exception: out[n] = None\n"
            "print(json.dumps(out))")], env).strip().splitlines()[-1]))
    except Exception as exc:  # noqa: BLE001 - recorded: the serve start then shows whether kernels are missing
        kernels["error"] = f"{type(exc).__name__}: {exc}"[:3000]
    shutil.rmtree(root / "cache/uv", ignore_errors=True)
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    target = Path(TARGET)
    target.mkdir(parents=True, exist_ok=True)
    tar = target / "stockvllm.tar.gz"
    run(["tar", "-I", "pigz -6", "-cf", str(tar), "-C", "/tmp", "stockvllm"], timeout=2400)
    with tar.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {"commit": COMMIT, "wheel": WHEEL, "extra": EXTRA, "versions": versions, "kernels": kernels,
              "unpacked_bytes": size, "tar_bytes": tar.stat().st_size, "tar_sha256": sha, "path": str(root),
              "python": str(base.relative_to(root)), "seconds": round(time.time() - started),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (target / "stock-vllm.json").write_text(json.dumps(record, indent=1))
    assets.commit()
    print(json.dumps({k: v for k, v in record.items() if k != "kernels"}), flush=True)
    return record


def reserve(run_id):
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        rates = data.get("rates_per_second") or {}
        rate = CPU * rates.get("physical_cpu_core", 1.31e-05) + MEMORY_GIB * rates.get("memory_gib", 2.22e-06)
        bound = (TIMEOUT + 60) * rate * 1.15 + 10 * data.get("volume_rate_usd_per_gib_month", 0.09) * 7 / 30
        committed = data["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved")
        data["runs"].append({"run_id": run_id, "state": "reserved", "kind": "runtime-build",
                             "charged_bound_usd": round(bound, 4), "max_seconds": TIMEOUT, "gpu": "none",
                             "cpu_ceiling": CPU, "memory_ceiling_gib": MEMORY_GIB, "authorization": AUTHORIZATION,
                             "target": f"arc-prolong-assets{TARGET[7:]}"})
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


@app.local_entrypoint()
def build_main(run_id: str):
    bound = reserve(run_id)
    call = build.spawn()
    print(json.dumps({"run_id": run_id, "kind": "runtime-build", "charged_bound_usd": bound,
                      "function_call_id": call.object_id, "target": f"arc-prolong-assets{TARGET[7:]}"}))
