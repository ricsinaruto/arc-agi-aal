"""Modal launcher for arc2 RL (docs/arc2-arm.md, "RL with a verifiable reward"): one B300 runs a
``dist/ops/build_arc2rl.py`` bundle (the RL arm's rl1-final-mg1 stack: Flash NVFP4 server with LoRA and the trainer
service, in the pinned Axolotl image) whose ``rl-loop`` trial dispatches to ``arc2.rl_loop`` (copy of the RL arm's
scripts/posttrain/modal_rl1.py, loop runs only).

    ARC2_RL_BUNDLE=<bundle> ARC_VALIDATION_ROOT=. \
        .venv/bin/modal run --detach scripts/modal_arc2_rl.py --run-id arc2rl-pipe-v1   (--dry-run: bound + checks)

Timeout = the bundle's notebook seconds + 300. Output ``arc-prolong-runs:/validation/<run id>/``; the live report is
mirrored to ``arena/trials/rl-loop/rl/status.json``. Graceful stop: ``stop-request`` in
``validation/<run id>/arena/live/``. Resume: ``--resume /runs/validation/<earlier run>/ckpt`` (new run id). Receipt ``docs/validation-20261009/modal-<run id>.json``.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or ".").resolve()
BUNDLE = (Path(os.environ.get("ARC2_RL_BUNDLE", "dist/arc2rl-pipe-v1")).resolve() if modal.is_local()
          else Path("/opt/arc-validation"))
GPU, GPU_RATE, CPU, MEMORY_GIB = "B300", 0.001972, 16.0, 435  # Modal bills a B300 container 435 GiB of host memory
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
RL_PIP = ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0")
STOP_FILE, STOP_LOCAL, STOP_POLL = "stop-request", "/tmp/arena-stop-request", 30
AUTHORIZATION = os.environ.get("ARC2_RL_AUTHORIZATION", "")


def bundle_study():
    return json.loads((BUNDLE / "configs/comparison.json").read_text())


STUDY = bundle_study() if (modal.is_local() and (BUNDLE / "configs/comparison.json").is_file()) else None
TIMEOUT = int(os.environ.get("ARC_RL1_TIMEOUT") or ((STUDY["notebook_seconds"] + 300) if STUDY else 46000))

app = modal.App("arc2-rl", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .env({"ARC_RL1_TIMEOUT": str(TIMEOUT), "PYTHONUNBUFFERED": "1"}))
for command in RL_PIP:
    image = image.run_commands("uv pip install --python /workspace/axolotl-venv/bin/python "
                               + " ".join(f"'{item}'" for item in command.split()))
image = image.add_local_dir(str(BUNDLE), "/opt/arc-validation", copy=True,
                            ignore=["dataset-upload/**", "*.ipynb", "source.zip"]).add_local_file(
    __file__, "/root/modal_arc2_rl.py", copy=True)


def host_cpu():
    try:
        line = next(l for l in open("/proc/cpuinfo") if l.startswith("model name"))
        return {"model": line.split(":", 1)[1].strip(), "count": os.cpu_count()}
    except Exception:  # a record field only
        return None


def bound(seconds):
    return (seconds + 60) * (GPU_RATE + CPU * 1.31e-05 + MEMORY_GIB * 2.22e-06) * 1.15


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def run(run_id: str, resume: str = ""):
    return body(run_id, resume)


PREFLIGHT_CPU, PREFLIGHT_GIB, PREFLIGHT_TIMEOUT = 8.0, 32, 3600


@app.function(image=image, timeout=PREFLIGHT_TIMEOUT, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def preflight(run_id: str, resume: str = ""):
    """Launch gate: this bundle, these arguments, on a CPU container; arc2.preflight's scripted server
    and trainer stand in for the Flash server and rl1_trainer (ARC2_PREFLIGHT=1); everything else runs as on the B300."""
    os.environ["ARC2_PREFLIGHT"] = "1"
    os.environ["ARC2_PREFLIGHT_BUNDLE"] = "/opt/arc-validation"
    if run_id.startswith("arc2rl-dose-"):
        os.environ["ARC2_PREFLIGHT_TRAINER_DELAY"] = "30" if run_id.endswith("-pfb") else "0"
        return dose_body(run_id)
    if run_id.startswith("arc2rl-chunk-"):  # -pfc: slow windows, so the stop lands mid-run
        os.environ["ARC2_PREFLIGHT_TRAINER_DELAY"] = "30" if run_id.endswith("-pfc") else "0"
        return chunk_body(run_id, resume)
    if run_id.startswith("arc2rl-snr-"):  # tensor-mode stand-in (rl1_trainer's checkpoint format); -pfc: slow updates
        os.environ["ARC2_PREFLIGHT_TENSORS"] = "/workspace/axolotl-venv/bin/python"
        os.environ["ARC2_PREFLIGHT_TRAINER_DELAY"] = "20" if run_id.endswith("-pfc") else "0"
        os.environ["ARC2_PREFLIGHT_SCORE_DELAY"] = "20" if run_id.endswith("-pfc") else "0"  # dose mode has no updates
        return snr_body(run_id, resume)
    # stages A and C must still be playing when their stop arrives (the stop poller reads every 120 s)
    os.environ["ARC2_PREFLIGHT_DELAY"] = "150" if run_id.endswith(("-pfa", "-pfc")) else "0.05"
    stub = Path("/usr/local/bin/nvidia-smi")  # CPU container: hardware records and GPU samplers read a stub card
    stub.write_text("#!/usr/bin/env python3\nimport sys\nq = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--query-gpu=')), '')\n"
                    "v = {'name': 'NVIDIA B300 preflight stub', 'memory.total': '275040', 'power.limit': '1100', 'driver_version': 'preflight'}\n"
                    "print(', '.join(v.get(f.strip(), '0') for f in q.split(',')) if q else 'preflight stub')\n")
    stub.chmod(0o755)
    return body(run_id, resume)


def body(run_id, resume):
    import shutil
    import subprocess
    import threading

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    if output.exists():
        output.rename(output.with_name(f"{run_id}-lost-{int(time.time())}"))
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": None if os.environ.get("ARC2_PREFLIGHT") else GPU,
                "function_timeout": TIMEOUT, "image": AXOLOTL_IMAGE, "host_cpu": host_cpu(),
                "preflight": bool(os.environ.get("ARC2_PREFLIGHT")), "resume": resume or None}
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ.update(PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1", MPLBACKEND="Agg", ONLY_RESET_LEVELS="true",
                      PYTHONDONTWRITEBYTECODE="1", ARC_FORCE_CHROOT="1", ARENA_STOP_REQUEST=STOP_LOCAL,
                      ARC_VALIDATION_OUTPUT=str(output))
    if resume:  # an earlier run's ckpt folder on the volume (arc2.rl_loop resumes from its state.json)
        os.environ["ARC2_RL_RESUME"] = resume
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
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


SPEED_SOURCE = "/runs/validation/arc2rl-pipe-v3/arena/trials/rl-loop/rl/screen"  # recorded pool-v2 screen episodes


@app.function(image=image, gpu=GPU, timeout=1800, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def speed(run_id: str, count: int = 8):
    """Trainer only, no server (arc2.rl_loop.trainer_speed): profiled passes and lr-0 updates on recorded episodes."""
    import asyncio

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    sys.path.insert(0, str(source / "src"))
    from arc2 import rl_loop

    tasks = sorted(Path(SPEED_SOURCE).iterdir())
    picked = []
    for task in tasks:  # two tasks' episodes with a reply each: shared prompts as in a training group
        runs_ = sorted(p for p in task.glob("*/rollout.jsonl") if p.stat().st_size)
        if len(runs_) >= 4:
            picked += [str(p) for p in runs_[:4]]
        if len(picked) >= count:
            break
    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": GPU, "function_timeout": 1800, "rollouts": picked}
    try:
        cfg = {**rl_loop.DEFAULTS, **bundle_study()["arc2_rl"]}
        report = asyncio.run(rl_loop.trainer_speed(cfg, source, picked, output))
        identity["status"] = report.get("status")
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


@app.function(image=image, gpu=GPU, timeout=3600, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def dose(run_id: str):
    """P0: trainer only (arc2.rl_loop.dose_curve) on the bundle's index of recorded base episodes."""
    return dose_body(run_id)


def dose_body(run_id):
    import asyncio
    import threading

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    sys.path.insert(0, str(source / "src"))
    from arc2 import rl_loop

    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": None if os.environ.get("ARC2_PREFLIGHT") else GPU,
                "function_timeout": 3600, "preflight": bool(os.environ.get("ARC2_PREFLIGHT"))}
    stop = threading.Event()

    def committer():  # dose-v1 (19 Oct): the watcher saw no trainer.log or dose.json for 40 min without commits
        while not stop.wait(60):
            try:
                runs.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"interim volume commit failed: {exc}", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    try:
        cfg = {**rl_loop.DEFAULTS, **bundle_study()["arc2_rl"]}
        report = asyncio.run(rl_loop.dose_curve(cfg, source, source / "data/rl/dose-prior.json", output,
                                                **(cfg.get("dose") or {})))
        identity["status"] = report.get("status")
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


CHUNK_TIMEOUT = 2400


@app.function(image=image, gpu=GPU, timeout=CHUNK_TIMEOUT, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def chunk(run_id: str, resume: str = ""):
    """rl1-chunk port: trainer only (arc2.rl_loop.chunk_check) on recorded base tries, each
    window at every prefix chunk x QSA selection setting of the bundle's ``arc2_rl.chunk`` plan."""
    return chunk_body(run_id, resume)


def chunk_body(run_id, resume=""):
    import asyncio
    import threading

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    sys.path.insert(0, str(source / "src"))
    from arc2 import rl_loop

    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": None if os.environ.get("ARC2_PREFLIGHT") else GPU,
                "function_timeout": CHUNK_TIMEOUT, "preflight": bool(os.environ.get("ARC2_PREFLIGHT")),
                "resume": resume or None}
    stop = threading.Event()

    def committer():  # the watcher reads chunk.json and trainer.log mid-run
        while not stop.wait(60):
            try:
                runs.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"interim volume commit failed: {exc}", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    try:
        cfg = {**rl_loop.DEFAULTS, **bundle_study()["arc2_rl"]}
        plan = cfg["chunk"]
        rollouts = rl_loop.pick_tries(plan["source"], int(plan.get("per_task") or 2))
        identity["rollouts"] = len(rollouts)
        report = asyncio.run(rl_loop.chunk_check(cfg, source, rollouts, output, settings=plan["settings"],
                                                 spec=plan.get("spec"), resume=resume or None,
                                                 deadline=started + CHUNK_TIMEOUT - 60))
        identity["status"] = report.get("status")
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


SNR_TIMEOUT = 2700 # (2,700 + 60) s x the B300 rate x 1.15 = : the <=


@app.function(image=image, gpu=GPU, timeout=SNR_TIMEOUT, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def snr(run_id: str, resume: str = ""):
    """arc2 RL diagnostic: trainer only (arc2.rl_loop.snr_check), gradient SNR of P5's positives
    and the LoRA-scope / random-direction doses of the bundle's ``arc2_rl.snr`` plan."""
    return snr_body(run_id, resume)


def snr_body(run_id, resume=""):
    import asyncio
    import threading

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    sys.path.insert(0, str(source / "src"))
    from arc2 import rl_loop

    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": None if os.environ.get("ARC2_PREFLIGHT") else GPU,
                "function_timeout": SNR_TIMEOUT, "preflight": bool(os.environ.get("ARC2_PREFLIGHT")),
                "resume": resume or None}
    stop = threading.Event()

    def committer():  # the watcher reads snr.json, card.json and trainer.log mid-run
        while not stop.wait(60):
            try:
                runs.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"interim volume commit failed: {exc}", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    try:
        cfg = {**rl_loop.DEFAULTS, **bundle_study()["arc2_rl"]}
        report = asyncio.run(rl_loop.snr_check(cfg, source, cfg["snr"], output, resume=resume or None,
                                               deadline=started + SNR_TIMEOUT - 60))
        identity["status"] = report.get("status")
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


def server_check(study):
    """The bundle's server start path through every settings check up to the vLLM launch (server_check.py), so a
    bound the server rejects fails here, not after the image and model load (mg5f, mg5g-chk)."""
    import subprocess

    out = subprocess.run([sys.executable, str(BUNDLE / "server_check.py"), str(BUNDLE)],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit("server check failed: " + (out.stderr.strip().splitlines() or ["?"])[-1])
    tuning = json.loads(out.stdout.strip().splitlines()[-1])
    return {k: tuning.get(k) for k in ("max_num_seqs", "kv_cache_memory_bytes", "max_cudagraph_capture_size")}



def preflight_receipt(manifest_sha):
    return ROOT / "docs/validation-20261009" / f"preflight-{manifest_sha[:16]}.json"


@app.local_entrypoint()
def main(run_id: str, dry_run: bool = False, resume: str = "", preflight_run: bool = False):
    if not run_id.startswith("arc2rl-"):
        raise ValueError("run ids start with arc2rl-")
    if run_id.startswith(("arc2rl-speed-", "arc2rl-dose-", "arc2rl-chunk-", "arc2rl-snr-")) and not preflight_run:
        # trainer only: speed 1,800 s cap, dose 3,600 s, chunk CHUNK_TIMEOUT, snr SNR_TIMEOUT
        record = ROOT / "docs/validation-20261009" / f"modal-{run_id}.json"
        dose_run, chunk_run = run_id.startswith("arc2rl-dose-"), run_id.startswith("arc2rl-chunk-")
        snr_run = run_id.startswith("arc2rl-snr-")
        limit = 3600 if dose_run else CHUNK_TIMEOUT if chunk_run else SNR_TIMEOUT if snr_run else 1800
        kind = ("arc2-rl-dose" if dose_run else "arc2-rl-chunk" if chunk_run else "arc2-rl-snr" if snr_run
                else "arc2-rl-speed")
        sha = hashlib.sha256((BUNDLE / "MANIFEST.json").read_bytes()).hexdigest()
        data = {"state": "dry_run", "run_id": run_id, "kind": kind, "gpu": GPU, "timeout_seconds": limit,
                "charged_bound_usd": round(bound(limit), 2), "bundle": str(BUNDLE), "manifest_sha256": sha,
                "authorization": AUTHORIZATION, "resume": resume or None}
        if dry_run:
            print(json.dumps(data, indent=1))
            return
        if not AUTHORIZATION or record.exists():
            raise SystemExit("set ARC2_RL_AUTHORIZATION / run ids are single-use")
        gate = preflight_receipt(sha)
        if (data["charged_bound_usd"] > 10 or chunk_run or snr_run) and not (gate.exists() and json.loads(gate.read_text()).get("pass")):
            raise SystemExit(f"launch gate: no passing preflight of this bundle ({gate.name})")
        data["preflight"] = gate.name if gate.exists() else None
        data["state"] = "submission_intent"
        record.write_text(json.dumps(data, indent=2))
        call = (chunk.spawn(run_id, resume) if chunk_run else snr.spawn(run_id, resume) if snr_run
                else (dose if dose_run else speed).spawn(run_id))
        data.update(state="submitted", function_call_id=call.object_id)
        record.write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
        return
    study = bundle_study()
    if not study.get("arc2_rl"):
        raise SystemExit("not an arc2 RL bundle (no arc2_rl block)")
    files = json.loads((BUNDLE / "MANIFEST.json").read_text())["files"]
    bad = [rel for rel, digest in files.items() if (BUNDLE / rel).is_file() and isinstance(digest, str)
           and hashlib.sha256((BUNDLE / rel).read_bytes()).hexdigest() != digest]
    if bad:
        raise SystemExit(f"bundle files changed after the build: {bad[:5]}")
    checked = server_check(study)
    record = ROOT / "docs/validation-20261009" / f"modal-{run_id}.json"
    data = {"state": "dry_run", "run_id": run_id, "kind": "arc2-rl", "gpu": GPU, "timeout_seconds": TIMEOUT,
            "server_check": checked, "charged_bound_usd": round(bound(TIMEOUT), 2),
            "bundle": str(BUNDLE), "manifest_sha256": hashlib.sha256((BUNDLE / "MANIFEST.json").read_bytes()).hexdigest(),
            "image": AXOLOTL_IMAGE, "output": f"arc-prolong-runs:/validation/{run_id}", "authorization": AUTHORIZATION,
            "resume": resume or None, "arc2_rl": {k: v for k, v in study["arc2_rl"].items() if k not in ("trainer_python", "model_dir")}}
    if dry_run:
        print(json.dumps(data, indent=1))
        return
    if preflight_run:  # launch gate stage: CPU only, no authorization needed (dist/ops/preflight_rl.py drives it)
        if "-pf" not in run_id:
            raise SystemExit("preflight run ids carry -pf")
        if record.exists():
            raise SystemExit(f"{record} exists: run ids are single-use")
        data.update(kind="arc2-rl-preflight", gpu=None, timeout_seconds=PREFLIGHT_TIMEOUT, state="submission_intent",
                    charged_bound_usd=round((PREFLIGHT_TIMEOUT + 60) * (PREFLIGHT_CPU * 1.31e-05 + PREFLIGHT_GIB * 2.22e-06) * 1.15, 2))
        record.write_text(json.dumps(data, indent=2))
        call = preflight.spawn(run_id, resume)
        data.update(state="submitted", function_call_id=call.object_id)
        record.write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
        return
    gate = preflight_receipt(data["manifest_sha256"])
    if not (gate.exists() and json.loads(gate.read_text()).get("pass")):
        raise SystemExit(f"launch gate: no passing preflight of this bundle ({gate.name}); run dist/ops/preflight_rl.py")
    if not AUTHORIZATION:
        raise SystemExit("set ARC2_RL_AUTHORIZATION (who approved the B300 run, when)")
    if record.exists():
        raise SystemExit(f"{record} exists: run ids are single-use")
    data["state"] = "submission_intent"
    data["preflight"] = gate.name
    record.write_text(json.dumps(data, indent=2))
    call = run.spawn(run_id, resume)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
