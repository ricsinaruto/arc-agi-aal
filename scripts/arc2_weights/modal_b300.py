"""One arc2 bundle on Modal B300 (base15b on the unpruned model). The arc2-run path of modal_validation.py
on a B300, plus what a Modal run lacks: control.json mirrored from the runs volume (live slots), RestartGuard, retries=0.

    ARC_VALIDATION_BUNDLE=$PWD/dist/<bundle> ARC2_GPU=B300 .venv/bin/modal run --detach \
        scripts/arc2_weights/modal_b300.py --run-id <run>   (reserve with records.py first)
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

GPU = os.environ.get("ARC2_GPU") or "B300:2"
TIMEOUT = int(os.environ.get("ARC2_TIMEOUT") or 14400)  # crash ceiling; the run ends on its token mark
STOP_FILE, STOP_LOCAL, CONTROL_LOCAL = "stop-request", "/tmp/arena-stop-request", "/tmp/arena-control.json"
BUNDLE = Path(os.environ.get("ARC_VALIDATION_BUNDLE") or ".").resolve() if modal.is_local() else Path("/opt/arc-validation")
RESTART = "./src/arc_infra/runops" if modal.is_local() else "/opt/guard/runops"  # main's: not in arc2 bundles
app = modal.App("arc-arc2-b300", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential", "ninja-build")
         .env({"ARC2_GPU": GPU, "ARC2_TIMEOUT": str(TIMEOUT)})
         .add_local_dir(str(BUNDLE), "/opt/arc-validation", copy=True, ignore=["dataset-upload/**", "*.ipynb", "source.zip"])
         .add_local_dir(RESTART, "/opt/guard/runops", copy=True, ignore=["__pycache__"])
         .add_local_file(__file__, "/root/modal_b300.py", copy=True))


def watch_live(run_id, output, stop):
    """Stop request and live slots from the runs volume (this container cannot reload it, and with no network the API
    can list it but not read it: slots travel in a file's name, slots-<n>-<epoch>, the newest counts)."""
    folder, seen = f"validation/{run_id}/arena/live", None
    while not stop.wait(30):
        try:
            names = {Path(e.path).name for e in runs.listdir(folder)}
            asked = sorted(([int(p[2]), int(p[1])] for p in (n.split("-") for n in names) if len(p) == 3 and p[0] == "slots"))
            if asked and asked[-1] != seen:
                seen = asked[-1]
                Path(CONTROL_LOCAL + ".tmp").write_text(json.dumps({"slots": seen[1]}))
                os.replace(CONTROL_LOCAL + ".tmp", CONTROL_LOCAL)
                print(json.dumps({"event": "control_mirrored", "slots": seen[1]}), flush=True)
            if STOP_FILE in names:
                Path(STOP_LOCAL).write_text(json.dumps({"source": folder, "epoch": round(time.time(), 1)}) + "\n")
                print(json.dumps({"event": "stop_request_seen"}), flush=True)
        except Exception as exc:  # no live folder yet, or the API is unavailable: keep polling
            print(f"live watcher: {type(exc).__name__}: {str(exc)[:160]}", flush=True)


@app.function(image=image, gpu=GPU, cpu=(0.125, 64), memory=(128, 409600), timeout=TIMEOUT, retries=0, max_containers=1,
              scaledown_window=2, block_network=True, volumes={"/assets": assets, "/runs": runs})
def run(run_id: str):
    source, output = Path("/opt/arc-validation"), Path("/runs/validation") / run_id
    sys.path.insert(0, "/opt/guard")
    from runops.restart import RestartGuard
    guard = RestartGuard("/runs", run_id, policy="stop", commit=runs.commit)
    guard.start()
    real = shutil.which("nvidia-smi")  # the serving stack checks for exactly one card; it sees GPU 0 (as RL's two-card loop)
    if real:
        shim = Path("/tmp/gpu0-bin")
        shim.mkdir(exist_ok=True)
        (shim / "nvidia-smi").write_text('#!/bin/sh\ncase " $* " in *" -i "*|*" --id"*) exec %s "$@";; esac\n'
                                         'exec %s -i 0 "$@"\n' % (real, real))
        (shim / "nvidia-smi").chmod(0o755)
        os.environ.update(PATH=f"{shim}:{os.environ['PATH']}", ARC_REAL_NVIDIA_SMI=real)
    (output / "arena/live").mkdir(parents=True, exist_ok=True)
    (output / "identity.json").write_text(json.dumps({"run_id": run_id, "gpu": GPU, "started_epoch": time.time(), "timeout": TIMEOUT}))
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ.update(PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PIP_NO_INDEX="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", ARC_FORCE_CHROOT="1",
                      ARC_VALIDATION_OUTPUT=str(output), ARENA_STOP_REQUEST=STOP_LOCAL, ARENA_CONTROL=CONTROL_LOCAL)
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
    command = [sys.executable, str(source / "bootstrap_arena.py"), "--bundle", str(source), "--working", "/tmp/arena",
               "--output", str(output / "arena"), "--games-dir", "/assets/games", "--max-seconds", str(TIMEOUT - 60)]
    stop = threading.Event()

    def committer():
        while not stop.wait(60):
            try:
                runs.commit()
            except Exception as exc:  # a failed interim commit must not end the run
                print(f"interim volume commit failed: {exc}", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    threading.Thread(target=watch_live, args=(run_id, output, stop), daemon=True).start()
    with (output / "bootstrap.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        for line in iter(process.stdout.readline, b""):
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
        code = process.wait()
    stop.set()
    runs.commit()
    if code:
        raise RuntimeError(f"bootstrap exit {code}")


@app.function(image=image, cpu=(8, 16), timeout=600, retries=0, max_containers=1, block_network=True,
              volumes={"/runs": runs})
def probe(run_id: str):
    """CPU leg of the launcher: guard start, control mirror, stop mirror, bundle compiles, the BF16 checkpoint's folder."""
    import compileall
    sys.path.insert(0, "/opt/guard")
    from runops.restart import RestartGuard
    guard = RestartGuard("/runs", run_id, policy="stop", commit=runs.commit)
    guard.start()
    output = Path("/runs/validation") / run_id
    (output / "arena/live").mkdir(parents=True, exist_ok=True)
    (output / "arena/live" / "slots-80-100").write_text("")
    (output / "arena/live" / "slots-76-200").write_text("")
    (output / "arena/live" / STOP_FILE).write_text("x")
    runs.commit()
    stop = threading.Event()
    thread = threading.Thread(target=watch_live, args=(run_id, output, stop), daemon=True)
    thread.start()
    time.sleep(70)
    stop.set()
    return {"control": Path(CONTROL_LOCAL).read_text(), "stop": Path(STOP_LOCAL).exists(),
            "compiled": compileall.compile_dir("/opt/arc-validation/src", quiet=2, force=True)}


@app.local_entrypoint()
def main(run_id: str, probe_only: bool = False):
    if probe_only:
        print(json.dumps(probe.remote(run_id)))
        return
    print(run.spawn(run_id).object_id)
