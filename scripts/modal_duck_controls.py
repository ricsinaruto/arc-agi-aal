"""Run a Duck control bundle on Modal overflow capacity. Importing this file launches nothing.

Only invoke after budget review; the run reserves its bound in ``docs/spend.json``.
Existing offline assets are read-only; the run gets its own output directory and one GPU.
"""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("arc-duck-controls", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = modal.Image.debian_slim(python_version="3.12").apt_install(
    "build-essential", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "ninja-build"
)
MEMORY_CEILING_GIB = 192


def offline_environment(source, inherited):
    """Do not let Modal's injected packages satisfy dependencies of a new venv."""
    return {
        **inherited,
        "PYTHONPATH": str(Path(source) / "src"),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ONLY_RESET_LEVELS": "true",
        "MPLBACKEND": "Agg",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def bundle_timeout(bundle):
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        seconds = json.loads(archive.read("study.json"))["notebook_seconds"]
    if type(seconds) is not int or not 1 <= seconds <= 5640:
        raise ValueError("Notebook budget must be an integer within 1..5640 seconds")
    return seconds + 120


def reserve(run_id, seconds, *, gpu, memory_gib):
    """Book the maximum resource ceilings before spending, including failed jobs."""
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        rate = (0.000842 if gpu else 0) + 8 * 0.0000131 + memory_gib * 0.00000222
        bound = (seconds + 60) * rate * 1.15
        from arc_infra.runops.records import row_cost
        committed = data["infrastructure_reserve_usd"] + sum(row_cost(r)[0] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} committed + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("This run already has a spend reservation")
        data["runs"].append(
            {
                "run_id": run_id,
                "state": "reserved",
                "charged_bound_usd": bound,
                "max_seconds": seconds,
                "gpu": "RTX-PRO-6000" if gpu else None,
                "cpu_ceiling": 8,
                "memory_ceiling_gib": memory_gib,
                "resource_requests": "Minimum CPU/RAM; dynamic up to recorded ceilings",
            }
        )
        file.seek(0)
        json.dump(data, file, indent=2)
        file.truncate()
    return bound


@app.function(
    image=image,
    serialized=True,
    gpu="RTX-PRO-6000",
    cpu=(0.125, 8),
    memory=(128, MEMORY_CEILING_GIB * 1024),
    timeout=3800,
    startup_timeout=600,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    block_network=True,
    volumes={"/assets": assets, "/runs": runs},
)
def benchmark(bundle, sha, run_id):
    import io
    import os
    import subprocess
    import sys
    import tarfile
    import zipfile

    def offline_environment(source, inherited):
        return {
            **inherited,
            "PYTHONPATH": str(Path(source) / "src"),
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ONLY_RESET_LEVELS": "true",
            "MPLBACKEND": "Agg",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def bundle_timeout(bundle):
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            return json.loads(archive.read("study.json"))["notebook_seconds"] + 120

    started = time.time()
    output = Path("/runs/duck-controls") / run_id
    output.mkdir(parents=True, exist_ok=False)
    source = Path("/tmp/arc-flash-next/source")
    source.mkdir(parents=True)
    execution = {
        "run_id": run_id,
        "provider": "modal",
        "source_bundle_sha256": sha,
        "container_start_epoch": started,
        "gpu": "RTX-PRO-6000",
    }
    process = None
    try:
        if not Path("/assets/flash-input/READY.json").is_file():
            raise RuntimeError("CPU preparation has not completed")
        if hashlib.sha256(bundle).hexdigest() != sha:
            raise ValueError("Source archive checksum differs")
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            for name in archive.namelist():
                if not (source / name).resolve().is_relative_to(source):
                    raise ValueError("Invalid source archive path")
            archive.extractall(source)
        manifest = json.loads((source / "build-manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Runtime source hash differs: {name}")
        (output / "source.zip").write_bytes(bundle)
        Path("/kaggle").mkdir(exist_ok=True)
        Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
        os.environ.update(offline_environment(source, os.environ))
        os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
        sys.path[:0] = [str(source / "src"), str(source / "kaggle")]
        from flash_bootstrap import prepare, start_server, stop_server

        study = json.loads((source / "study.json").read_text())
        first = study["trials"][0]
        stop_dirs = [first["server_dir"]]
        install = prepare(
            source, output, started, server=study["servers"][first["name"]], working=first["server_dir"]
        )
        # The public notebook loop imports imageio at the end of Benchmark.run().
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "--python",
                install["agent_python"],
                "install",
                "--no-index",
                "--no-compile",
                "--disable-pip-version-check",
                "--find-links",
                str(source / "wheels"),
                "imageio==2.37.0",
            ],
            check=True,
            timeout=120,
        )
        failures = []
        for index, trial in enumerate(study["trials"]):
            working = trial["server_dir"]
            if index:
                stop_dirs[:] = [working]
                start_server(source, output, study["servers"][trial["name"]], working)
            command = [
                install["agent_python"],
                "-u",
                "-m",
                "duck_controls.run",
                "--source",
                str(source),
                "--study",
                str(source / "study.json"),
                "--output",
                str(output),
                "--analysis-python",
                install["analysis_python"],
                "--notebook-start-epoch",
                str(started),
                "--trial",
                trial["name"],
                "--server-dir",
                str(output / working),
            ]
            with (output / f"benchmark-{trial['name']}.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    env={**os.environ, "PYTHONPATH": str(source / "src")},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                process.wait(timeout=max(1, bundle_timeout(bundle) - 120 - (time.time() - started)))
                if process.returncode:
                    failures.append(f"{trial['name']}: benchmark exited {process.returncode}")
            process = None
            stop_server(output, working)
            stop_dirs[:] = []
        if failures:
            raise RuntimeError("; ".join(failures))
        execution["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - recorded, then cleaned up
        execution.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None and process.poll() is None:
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if "stop_server" in locals():
            for working in stop_dirs if "stop_dirs" in locals() else ["server"]:
                try:
                    stop_server(output, working)
                except Exception as exc:  # noqa: BLE001
                    execution["cleanup_error"] = str(exc)
        execution["container_seconds"] = time.time() - started
        (output / "modal-execution.json").write_text(json.dumps(execution, indent=2))
        with tarfile.open(str(output) + ".tar.gz", "w:gz") as archive:
            archive.add(output, arcname=run_id)
        runs.commit()
    return execution


@app.local_entrypoint()
def main(bundle_dir: str, run_id: str = ""):
    bundle = (Path(bundle_dir) / "source.zip").read_bytes()
    sha = hashlib.sha256(bundle).hexdigest()
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + sha[:6]
    seconds = bundle_timeout(bundle)
    bound = reserve(run_id, seconds, gpu=True, memory_gib=MEMORY_CEILING_GIB)
    call = benchmark.spawn(bundle, sha, run_id)
    launch = {
        "run_id": run_id,
        "function_call_id": call.object_id,
        "source_bundle_sha256": sha,
        "max_seconds": seconds,
        "charged_bound_usd": bound,
        "launched_utc": datetime.now(UTC).isoformat(),
    }
    (Path(bundle_dir) / "modal-launch.json").write_text(json.dumps(launch, indent=2))
    print(json.dumps(launch))
