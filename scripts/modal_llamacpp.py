"""Modal launcher for llama.cpp model screens (27 September): llama.cpp compiled for SM120 at the
profile's pinned commit (a CPU function, once per commit, into the Volume) and one GPU job per screen. Weights come from
the arc-llamacpp-weights Volume (scripts/modal_weights.py downloads them on CPU).

    # CPU only: compile llama.cpp at the profile's commit into the Volume (once) and print its build info
    ARC_LLAMACPP_PROFILE=glm53-flash-aj-iq2xxs uv run --extra modal modal run scripts/modal_llamacpp.py::build_main
    # GPU: serve check only, or serve check then the Duck-transfer probe, from a --llamacpp arena bundle
    ARC_LLAMACPP_PROFILE=... ARC_LLAMACPP_TEST=serve-check ARC_VALIDATION_BUNDLE=dist/<bundle> \
        uv run --extra modal modal run --detach scripts/modal_llamacpp.py --run-id <id>

The GPU entrypoint reserves the job's bound in the main checkout's docs/spend.json (the ledger of
record) with the procedure of scripts/modal_validation.py before spawning, and writes the launch
record (function-call id) under docs/validation-20260927/.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
LEDGER = Path(os.environ.get("ARC_SPEND_LEDGER", "./docs/spend.json"))
PROFILE_NAME = os.environ.get("ARC_LLAMACPP_PROFILE", "glm53-flash-aj-iq2xxs")
TEST = os.environ.get("ARC_LLAMACPP_TEST", "serve-check")
TESTS = {
    "research-compat": 3600,
    "research-gameplay": 3600,
    # Load (~5-10 min from the Volume), serve check (~10 min), teardown.
    "serve-check": 2400,
    # The same, then the 8-game MEDIUM probe for 3,600 s of gameplay and its grace.
    "probe-60m": 7200,
    "probe-30m": 5400,
    # 45 minutes of gameplay after about 12 minutes of setup and a trimmed serve check.
    "probe-45m": 4800,
}
if TEST not in TESTS:
    raise ValueError(f"Unknown test {TEST!r}; choose from {sorted(TESTS)}")
TIMEOUT = TESTS[TEST]
GPU_RATE = 0.000842
CPU_CEILING = 8
MEMORY_CEILING_GIB = 96

if modal.is_local():
    sys.path.insert(0, str(ROOT / "src"))
    from arena.llamacpp_profiles import LLAMA_CPP_REPO, profile

    PROFILE = profile(PROFILE_NAME)
else:
    LLAMA_CPP_REPO = os.environ.get("ARC_LLAMA_CPP_REPO", "")
    PROFILE = json.loads(base64.b64decode(os.environ.get("ARC_LLAMACPP_PROFILE_B64", "e30=")))

app = modal.App(f"arc-llamacpp-{PROFILE_NAME}", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
weights = modal.Volume.from_name("arc-llamacpp-weights", create_if_missing=True, version=2)
PROFILE_ENV = {"ARC_LLAMACPP_PROFILE": PROFILE_NAME, "ARC_LLAMACPP_TEST": TEST,
               "ARC_LLAMACPP_PROFILE_B64": base64.b64encode(json.dumps(PROFILE).encode()).decode(),
               "ARC_LLAMA_CPP_REPO": LLAMA_CPP_REPO}

def build_script(llama, repo, dest):
    """Fetch the pinned commit (the PR ref or tag; the fork as a fallback), build the CUDA binaries and
    pack build/bin with BUILD_INFO into ``dest`` (a tarball on the weights Volume)."""
    commit, ref = llama["commit"], llama["ref"]
    fetch = f"git fetch -q --depth 1 origin {ref}"
    if llama.get("fork"):
        fetch = f"({fetch} || git fetch -q --depth 1 {llama['fork']} {commit})"
    return " && ".join([
        "set -e", "rm -rf /tmp/llama.cpp", "git init -q /tmp/llama.cpp", "cd /tmp/llama.cpp",
        f"git remote add origin {repo}", fetch, f"git checkout -q {commit}", f'test "$(git rev-parse HEAD)" = {commit}',
        "cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120"
        " -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"
        # Static libraries: the first shared build packed build/bin without libllama-server-impl.so
        # (glm probe v1 failed its flag check on the GPU after 34 s).
        " -DBUILD_SHARED_LIBS=OFF",
        "cmake --build build --config Release -j$(nproc) --target llama-server llama-bench llama-batched-bench",
        "(git log -1 --format='%H %s' && nvcc --version | tail -2 && nproc) > build/bin/BUILD_INFO",
        "(find build -name '*.so*' ! -path 'build/bin/*' -exec cp -a {} build/bin/ \\; || true)",
        "(LD_LIBRARY_PATH=build/bin ldd build/bin/llama-server | tee build/bin/LDD.txt)",
        "! (grep 'not found' build/bin/LDD.txt | grep -v libcuda.so)",
        f"mkdir -p $(dirname {dest})", f"tar -C build -czf {dest}.tmp bin", f"mv {dest}.tmp {dest}",
    ])


# CUDA 12.8 devel (SM120 support; the runtime libraries llama.cpp links against) and the agent's apt
# packages. llama.cpp itself is compiled once per commit on a 32-core CPU function (compile_llama) into
# the weights Volume: the image builder compiled the CUDA kernels at about 4 files a minute.
gpu_image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
             .entrypoint([])
             .apt_install("git", "cmake", "build-essential", "ninja-build", "ca-certificates", "libgomp1",
                          "libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1")
             .env({"LLAMACPP_SERVER_BIN": "/opt/llama.cpp/bin/llama-server",
                   "LLAMACPP_BENCH_BIN": "/opt/llama.cpp/bin/llama-bench",
                   "LLAMACPP_BUILD_INFO": "/opt/llama.cpp/bin/BUILD_INFO", "LD_LIBRARY_PATH": "/opt/llama.cpp/bin",
                   **PROFILE_ENV})
             .add_local_file(__file__, "/root/modal_llamacpp.py", copy=True))
BUNDLE = None
if os.environ.get("ARC_VALIDATION_BUNDLE") or not modal.is_local():
    BUNDLE = (Path(os.environ["ARC_VALIDATION_BUNDLE"]).resolve() if modal.is_local() else Path("/opt/arc-validation"))
    gpu_image = gpu_image.add_local_dir(str(BUNDLE), "/opt/arc-validation", copy=True,
                                        ignore=["dataset-upload/**", "*.ipynb", "source.zip"])


def build_tarball(commit):
    if PROFILE_NAME == 'flash-next-gsq-iq3s':
        return f'/assets/research-20260930/llama-{commit}.tar.gz'
    return f"/weights/_llama.cpp/{commit}-static.tar.gz"


@app.function(image=gpu_image, cpu=32, memory=65536, timeout=3600, retries=0, max_containers=1,
              volumes={"/weights": weights})
def compile_llama():
    import subprocess as _subprocess

    llama = PROFILE["llama_cpp"]
    dest = build_tarball(llama["commit"])
    if not Path(dest).is_file():
        started = time.time()
        _subprocess.run(["bash", "-c", build_script(llama, LLAMA_CPP_REPO, dest)], check=True)
        weights.commit()
        print(json.dumps({"built": dest, "seconds": round(time.time() - started, 1)}), flush=True)
    _subprocess.run(["bash", "-c", f"mkdir -p /opt/llama.cpp && tar -C /opt/llama.cpp -xzf {dest}"], check=True)
    help_text = _subprocess.run([os.environ["LLAMACPP_SERVER_BIN"], "--help"], capture_output=True, text=True).stdout
    return {"tarball": dest, "bytes": Path(dest).stat().st_size,
            "build": Path(os.environ["LLAMACPP_BUILD_INFO"]).read_text(), "help_chars": len(help_text)}


@app.function(image=gpu_image, gpu="RTX-PRO-6000", cpu=(0.125, CPU_CEILING),
              memory=(1024, MEMORY_CEILING_GIB * 1024), timeout=TIMEOUT, retries=0, max_containers=1,
              scaledown_window=2, block_network=True,
              volumes={"/assets": assets, "/runs": runs, "/weights": weights})
def screen(run_id: str):
    import shutil
    import signal
    import subprocess
    import sys as _sys

    profile_name = os.environ["ARC_LLAMACPP_PROFILE"]
    test = os.environ["ARC_LLAMACPP_TEST"]
    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    if output.exists():
        output.rename(output.with_name(f"{run_id}-lost-{int(time.time())}"))
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    subprocess.run(["bash", "-c", "mkdir -p /opt/llama.cpp && tar -C /opt/llama.cpp -xzf "
                    + build_tarball(PROFILE["llama_cpp"]["commit"])], check=True)
    identity = {"run_id": run_id, "profile": profile_name, "test": test, "started_epoch": started,
                "gpu": "RTX-PRO-6000", "function_timeout": TIMEOUT,
                "llama_cpp_build": Path(os.environ["LLAMACPP_BUILD_INFO"]).read_text()}
    os.environ.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1",
                      MPLBACKEND="Agg", ONLY_RESET_LEVELS="true", PYTHONDONTWRITEBYTECODE="1",
                      LLAMACPP_WEIGHTS_DIR=('/assets/research-20260930/iq3' if profile_name == 'flash-next-gsq-iq3s'
                                           else f"/weights/{profile_name}"))
    command = [_sys.executable, str(source / "bootstrap_arena.py"), "--bundle", str(source),
               "--working", "/tmp/arena", "--output", str(output / "arena"), "--games-dir", "/assets/games",
               "--max-seconds", str(TIMEOUT - 60), "--notebook-start-epoch", str(started)]
    command += ["--serve-check-only"] if test in ("serve-check", "research-compat") else ["--trial", "duck-transfer"]
    process = None
    # Live visibility: the bootstrap's output (llama-server highlights, the serve
    # check, the trial's progress and a probe_progress line a minute) is teed to the container's stdout,
    # so `modal app logs <app>` streams it; every two minutes the setup, serving records, serve check,
    # the trial's progress files and the logs are copied to the runs volume and committed.
    import threading

    def pump(pipe, log):
        for line in iter(pipe.readline, b""):
            log.write(line)
            log.flush()
            _sys.stdout.buffer.write(line)
            _sys.stdout.flush()

    def sync_live():
        live = output / "live"
        live.mkdir(exist_ok=True)
        work = Path("/tmp/arena")
        files = [work / "setup.json", *sorted((work / "serving").glob("*")), *sorted((work / "trials").glob("*.log")),
                 *(work / "trials" / "duck-transfer" / name for name in ("progress.json", "progress.jsonl"))]
        for f in files:
            if f.is_file() and f.suffix in {".json", ".jsonl", ".log", ".md"}:
                data = f.read_bytes()
                (live / f.name).write_bytes(data[-2_000_000:] if f.suffix == ".log" else data)

    def committer(stop):
        while not stop.wait(120):
            try:
                sync_live()
                runs.commit()
            except Exception as exc:  # a failed interim commit must not end the run
                print(f"interim volume commit failed: {exc}", flush=True)

    try:
        with (output / "bootstrap.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            stop = threading.Event()
            threads = [threading.Thread(target=pump, args=(process.stdout, log), daemon=True),
                       threading.Thread(target=committer, args=(stop,), daemon=True)]
            for thread in threads:
                thread.start()
            try:
                process.wait(timeout=max(1, started + TIMEOUT - 20 - time.time()))
            finally:
                stop.set()
                threads[0].join(timeout=10)
        identity["exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"Bootstrap exited {process.returncode}; inspect {output}/bootstrap.log")
    except BaseException as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        serving = Path("/tmp/arena/serving")
        if serving.exists():
            dest = output / "arena" / "serving"
            dest.mkdir(parents=True, exist_ok=True)
            for f in serving.iterdir():
                if f.is_file() and f.suffix in {".json", ".log", ".jsonl", ".txt", ".md"}:
                    shutil.copyfile(f, dest / f.name)
        identity["elapsed_seconds"] = time.time() - started
        (output / "execution.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return str(output)


def reserve(run_id, seconds, *, ledger=LEDGER, test=TEST, profile_name=PROFILE_NAME):
    """scripts/modal_validation.py reserve(), with this function's CPU and memory ceilings."""
    import fcntl

    with Path(ledger).open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        if test.startswith('research-') and sum(r.get('kind') == 'research-pilot' and r.get('state') == 'reserved'
                                           for r in data['runs']) >= 4:
            raise RuntimeError('Four research GPUs reserved; reconcile before launching another')
        rate = GPU_RATE + CPU_CEILING * 0.0000131 + MEMORY_CEILING_GIB * 0.00000222
        bound = (seconds + 60) * rate * 1.15
        from arc_infra.runops.records import row_cost
        committed = data["infrastructure_reserve_usd"] + sum(row_cost(r)[0] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": "research-pilot" if test.startswith('research-') else "llamacpp-screen", "test": test,
            "profile": profile_name, "charged_bound_usd": bound, "max_seconds": seconds,
            "gpu": "RTX-PRO-6000", "cpu_ceiling": CPU_CEILING, "memory_ceiling_gib": MEMORY_CEILING_GIB,
            "authorization": ("<=4 research GPUs, <=1 hour pilots, " if test.startswith('research-')
                              else "screen new open models one by one with short Modal tests; Modal only"),
        })
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


@app.local_entrypoint()
def main(run_id: str):
    if BUNDLE is None:
        raise ValueError("Set ARC_VALIDATION_BUNDLE to the llama.cpp arena bundle")
    study = json.loads((BUNDLE / "configs/comparison.json").read_text())
    if study.get("backend") != "llamacpp" or study["llamacpp"]["profile"] != PROFILE_NAME:
        raise ValueError("The bundle is not a llama.cpp study for this profile")
    bound = reserve(run_id, TIMEOUT)
    record = ROOT / ("docs/validation-20260930" if TEST.startswith('research-') else "docs/validation-20260927") / f"modal-{run_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": "submission_intent", "run_id": run_id, "kind": "llamacpp-screen", "test": TEST,
            "profile": PROFILE_NAME, "charged_bound_usd": bound, "ledger": str(LEDGER),
            "platform_policy": "modal-only model screen", "bundle": str(BUNDLE)}
    record.write_text(json.dumps(data, indent=2))
    call = screen.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


@app.local_entrypoint()
def build_main():
    print(json.dumps(compile_llama.remote()))
