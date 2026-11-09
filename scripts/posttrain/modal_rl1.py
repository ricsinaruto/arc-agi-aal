"""Modal launcher for RL step 1 (13 October, docs/rl-plan.md): one B300 runs a ``build_rl1.py`` bundle (the arena
bootstrap: Flash NVFP4 server with LoRA, the ``rl-loop`` trial) in the pinned Axolotl image, whose venv the trial's
trainer service uses.

    ARC_RL1_BUNDLE=<bundle> ARC_VALIDATION_ROOT=. \
        uv run --extra modal modal run --detach scripts/posttrain/modal_rl1.py --run-id rl1-probe-v1
    add --dry-run for the bound and checks without spawning

Timeout = the bundle's notebook seconds + 300. Output ``arc-prolong-runs:/validation/<run id>/`` (the arena output
under ``arena/``, the loop's feed under ``arena/<trial>/rl/``). Graceful stop: put ``stop-request`` in
``validation/<run id>/arena/live/`` (``scripts/arena/request_stop.py``); the loop stops after its current round.
Receipt ``docs/validation-rl/modal-<run id>.json``; no ledger write (the operator keeps docs/spend.json).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or ".").resolve()
BUNDLE = (Path(os.environ.get("ARC_RL1_BUNDLE", "dist/rl1-probe-v1")).resolve() if modal.is_local()
          else Path("/opt/arc-validation"))
GPU, GPU_RATE, CPU, MEMORY_GIB = "B300", 0.001972, 16.0, 435  # Modal bills a B300 container 435 GiB of host memory
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
# SonicMoE on a B300 (8 October): Triton 3.6.x, cutlass-dsl 4.6.0, quack kernels, in Axolotl's venv
RL_PIP = ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0")
STOP_FILE, STOP_LOCAL, STOP_POLL = "stop-request", "/tmp/arena-stop-request", 30
AUTHORIZATION = ("LR probe 4 updates/rate (~4.6 h) and the 12.5 h loop approved; "
                 "loop with split mismatch rule, TIS cap 2, KV 56 GiB, LR 4e-5, pass 7.0; dose (all A > 0, "
                 "4 steps/round), trend read, per-step KL, split once with the noise bar; continue from v7's "
                 "round 3 for at most 6 h of loop wall (inside the 12.5 h): one signed step/round, 64 attempts, lp85 play guard; "
                 "'yes i pick (b). yes to F-small after.' (v9 bound; F-small bound)")


def bundle_study():
    return json.loads((BUNDLE / "configs/comparison.json").read_text())


STUDY = bundle_study() if (modal.is_local() and (BUNDLE / "configs/comparison.json").is_file()) else None
TIMEOUT = int(os.environ.get("ARC_RL1_TIMEOUT") or ((STUDY["notebook_seconds"] + 300) if STUDY else 46000))
# a two-card loop (server on GPU 0, trainer on GPU 1) runs on two B300s
CARDS = int(os.environ.get("ARC_RL1_CARDS") or (2 if STUDY and STUDY["rl_loop"].get("dual") else 1))
if CARDS == 2:
    GPU, GPU_RATE = "B300:2", 2 * GPU_RATE
TEST_TIMEOUT = 3600  # trainer test: load ~2-5 min, fit ~10-15 min, two updates; the rest is slack

app = modal.App("arc-rl1", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .env({"ARC_RL1_TIMEOUT": str(TIMEOUT), "ARC_RL1_CARDS": str(CARDS), "PYTHONUNBUFFERED": "1"}))
for command in RL_PIP:
    image = image.run_commands("uv pip install --python /workspace/axolotl-venv/bin/python "
                               + " ".join(f"'{item}'" for item in command.split()))
image = image.add_local_dir(str(BUNDLE), "/opt/arc-validation", copy=True,
                            ignore=["dataset-upload/**", "*.ipynb", "source.zip"]).add_local_file(
    __file__, "/root/modal_rl1.py", copy=True)


def host_cpu():
    try:
        line = next(l for l in open("/proc/cpuinfo") if l.startswith("model name"))
        return {"model": line.split(":", 1)[1].strip(), "count": os.cpu_count()}
    except Exception:  # a record field only
        return None


def bound(seconds):
    return (seconds + 60) * (GPU_RATE + CPU * 1.31e-05 + MEMORY_GIB * 2.22e-06) * 1.15


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TIMEOUT, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def run(run_id: str):
    import shutil
    import subprocess
    import threading

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    if output.exists():
        output.rename(output.with_name(f"{run_id}-lost-{int(time.time())}"))
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": GPU, "function_timeout": TIMEOUT,
                "image": AXOLOTL_IMAGE, "host_cpu": host_cpu()}
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ.update(PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1", MPLBACKEND="Agg", ONLY_RESET_LEVELS="true",
                      PYTHONDONTWRITEBYTECODE="1", ARC_FORCE_CHROOT="1", ARENA_STOP_REQUEST=STOP_LOCAL,
                      ARC_VALIDATION_OUTPUT=str(output))
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
    if CARDS == 2:  # the serving stack checks for exactly one card: it sees GPU 0 (the trainer asks for its own card)
        real = shutil.which("nvidia-smi")
        shim = Path("/tmp/gpu0-bin")
        shim.mkdir(exist_ok=True)
        (shim / "nvidia-smi").write_text('#!/bin/sh\ncase " $* " in *" -i "*|*" --id"*) exec %s "$@";; esac\n'
                                         'exec %s -i 0 "$@"\n' % (real, real))
        (shim / "nvidia-smi").chmod(0o755)
        os.environ.update(PATH=f"{shim}:{os.environ['PATH']}", ARC_REAL_NVIDIA_SMI=real)
        identity["cards"] = subprocess.run([real, "--query-gpu=index,name", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip().splitlines()
    command = [sys.executable, str(source / "bootstrap_arena.py"), "--bundle", str(source), "--working", "/tmp/arena",
               "--output", str(output / "arena"), "--games-dir", "/assets/games", "--max-seconds", str(TIMEOUT - 60),
               "--trial", "rl-loop"]

    def pump(pipe, log):
        for line in iter(pipe.readline, b""):
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()

    def committer(stop):
        # the trial writes under /tmp/arena until the bootstrap collects it at the end; the live feed is copied
        # each pass (rl1-probe-v7's watcher saw no feed for the whole run)
        live, mirror = Path("/tmp/arena/trials/rl-loop/rl"), output / "arena/trials/rl-loop/rl"
        while not stop.wait(120):
            try:
                for name in ("feed.jsonl", "status.json"):
                    if (live / name).exists():
                        mirror.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(live / name, mirror / name)
                runs.commit()
            except Exception as exc:  # a failed interim commit must not end the run
                print(f"interim volume commit failed: {exc}", flush=True)

    def stop_watcher(stop):
        folder = f"validation/{run_id}/arena/live"
        while not stop.wait(STOP_POLL):
            try:
                names = {Path(entry.path).name for entry in runs.listdir(folder)}
            except Exception:  # no live folder yet
                continue
            if STOP_FILE in names:
                Path(STOP_LOCAL).write_text(json.dumps({"source": f"{folder}/{STOP_FILE}", "epoch": time.time()}))
                print(json.dumps({"event": "stop_request_seen"}), flush=True)
                return

    process = None
    try:
        with (output / "bootstrap.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            stop = threading.Event()
            threads = [threading.Thread(target=pump, args=(process.stdout, log), daemon=True),
                       threading.Thread(target=committer, args=(stop,), daemon=True),
                       threading.Thread(target=stop_watcher, args=(stop,), daemon=True)]
            for thread in threads:
                thread.start()
            try:
                process.wait()
            finally:
                stop.set()
                threads[0].join(timeout=30)
        identity["exit_code"] = process.returncode
    except Exception as exc:  # the identity records every outcome
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


TEST_ROLLOUTS = "/runs/posttrain/rl1-test-rollouts"  # six recorded Flash level-bench attempts (rl-nt-pool-v1-third-p1)


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TEST_TIMEOUT, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def trainer_test(run_id: str):
    """Every trainer operation of the loop without the play server (``arena.rl_loop.trainer_test``)."""
    import asyncio

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    sys.path.insert(0, str(source / "src"))
    from arena import rl_loop

    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": GPU, "function_timeout": TEST_TIMEOUT,
                "host_cpu": host_cpu()}
    try:
        report = asyncio.run(rl_loop.trainer_test(rl_loop.settings(bundle_study()["rl_loop"]), source,
                                                  sorted(str(p) for p in Path(TEST_ROLLOUTS).iterdir()), output))
        identity["status"] = report.get("status", "incomplete")
        identity["failed_steps"] = [s["step"] for s in report["steps"] if not s.get("ok")]
    except Exception as exc:  # the identity records every outcome
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


SPEED_ATTEMPTS = os.environ.get("ARC_RL1_SPEED_ATTEMPTS", "")  # comma-separated rollout.jsonl paths on the runs volume


@app.function(image=image, gpu="B300", cpu=CPU, memory=MEMORY_GIB * 1024, timeout=1800, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def trainer_speed(run_id: str, attempts: str, profile: bool = False, attention: bool = False):
    """QSA indexer loop vs vectorized on recorded windows (``arena.rl_loop.trainer_speed``), one card, 30 min cap."""
    import asyncio

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    sys.path.insert(0, str(source / "src"))
    from arena import rl_loop

    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": "B300", "function_timeout": 1800}
    try:
        report = asyncio.run(rl_loop.trainer_speed(rl_loop.settings(bundle_study()["rl_loop"]), source,
                                                   attempts.split(","), output, profile=profile,
                                                   attention=attention))
        identity["status"] = report.get("status", "incomplete")
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


@app.local_entrypoint()
def main(run_id: str, dry_run: bool = False):
    if run_id.startswith(("rl1-speed-", "rl1-prof-", "rl1-attn-")):
        record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
        data = {"state": "dry_run", "run_id": run_id, "kind": run_id.rsplit("-", 1)[0], "gpu": "B300", "timeout_seconds": 1800,
                "charged_bound_usd": round(1860 * (0.001972 + CPU * 1.31e-05 + MEMORY_GIB * 2.22e-06) * 1.15, 2), "bundle": str(BUNDLE),
                "attempts": SPEED_ATTEMPTS.split(",")}
        if dry_run:
            print(json.dumps(data, indent=1))
            return
        if record.exists():
            raise SystemExit(f"{record} exists: run ids are single-use")
        call = trainer_speed.spawn(run_id, SPEED_ATTEMPTS, run_id.startswith("rl1-prof-"), run_id.startswith("rl1-attn-"))
        data.update(state="submitted", function_call_id=call.object_id)
        record.write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
        return
    if run_id.startswith("rl1-ttest-"):
        return launch_test(run_id, dry_run)
    if not run_id.startswith(("rl1-probe-", "rl1-loop-", "rl1-restore-", "rl1-final-")):
        raise ValueError("run ids start with rl1-probe-, rl1-loop- or rl1-ttest-")
    study = bundle_study()
    mode = study["rl_loop"]["mode"]
    if not run_id.startswith(f"rl1-{mode}-"):
        raise ValueError(f"the bundle's mode is {mode}")
    files = json.loads((BUNDLE / "MANIFEST.json").read_text())["files"]
    bad = [rel for rel, row in files.items() if (BUNDLE / rel).is_file() and isinstance(row, dict) and row.get(
        "sha256") and hashlib.sha256((BUNDLE / rel).read_bytes()).hexdigest() != row["sha256"]]
    if bad:
        raise SystemExit(f"bundle files changed after the build: {bad[:5]}")
    record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
    data = {"state": "dry_run", "run_id": run_id, "kind": "rl1", "mode": mode, "gpu": GPU, "timeout_seconds": TIMEOUT,
            "charged_bound_usd": round(bound(TIMEOUT), 2), "expected_hours": round(TIMEOUT / 3600, 2),
            "bundle": str(BUNDLE), "manifest_sha256": hashlib.sha256((BUNDLE / "MANIFEST.json").read_bytes()).hexdigest(),
            "image": AXOLOTL_IMAGE, "output": f"arc-prolong-runs:/validation/{run_id}", "authorization": AUTHORIZATION,
            "rl_loop": {k: study["rl_loop"][k] for k in ("mode", "rounds", "lr", "rank", "window", "probe_rates")}}
    if dry_run:
        print(json.dumps(data, indent=1))
        return
    if record.exists():
        raise SystemExit(f"{record} exists: run ids are single-use")
    record.parent.mkdir(exist_ok=True)
    data["state"] = "submission_intent"
    record.write_text(json.dumps(data, indent=2))
    call = run.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


def launch_test(run_id, dry_run):
    record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
    data = {"state": "dry_run", "run_id": run_id, "kind": "rl1-trainer-test", "gpu": GPU, "timeout_seconds": TEST_TIMEOUT,
            "charged_bound_usd": round(bound(TEST_TIMEOUT), 2), "bundle": str(BUNDLE),
            "manifest_sha256": hashlib.sha256((BUNDLE / "MANIFEST.json").read_bytes()).hexdigest(),
            "image": AXOLOTL_IMAGE, "output": f"arc-prolong-runs:/validation/{run_id}", "rollouts": TEST_ROLLOUTS,
            "authorization": AUTHORIZATION}
    if dry_run:
        print(json.dumps(data, indent=1))
        return
    if record.exists():
        raise SystemExit(f"{record} exists: run ids are single-use")
    data["state"] = "submission_intent"
    record.write_text(json.dumps(data, indent=2))
    call = trainer_test.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
