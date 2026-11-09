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


AUTHORIZATION = os.environ.get("ARC_RL1_AUTHORIZATION") or AUTHORIZATION  # another arm's run names its own


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
         .apt_install("rclone", "libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
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


def bound(seconds, rate=None):
    return (seconds + 60) * ((GPU_RATE if rate is None else rate) + CPU * 1.31e-05 + MEMORY_GIB * 2.22e-06) * 1.15


def live_sample(trial, folder, started):
    """Live watcher feed (every run readable from its first minute), committed each minute:
    ``sampler.jsonl`` += nvidia-smi util / memory per card and the vLLM server's running / waiting / KV / generated
    tokens; ``progress.json``: per gate folder the attempts started, finished, cleared and their completion tokens."""
    import subprocess
    import urllib.request

    folder.mkdir(parents=True, exist_ok=True)
    row = {"t": round(time.time() - started)}
    try:
        out = subprocess.run([os.environ.get("ARC_REAL_NVIDIA_SMI", "nvidia-smi"), "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        cards = [[float(v) for v in line.split(",")] for line in out.splitlines() if line.strip()]
        row.update(util=[c[0] for c in cards], mem_gib=[round(c[1] / 1024, 1) for c in cards])
    except Exception:  # noqa: BLE001
        pass
    try:
        text = urllib.request.urlopen("http://127.0.0.1:1234/metrics", timeout=5).read().decode()
        sums = {}
        for line in text.splitlines():
            if line and line[0] != "#":
                name, _, value = line.rpartition(" ")
                key = name.split("{", 1)[0]
                if key in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                           "vllm:generation_tokens_total", "vllm:num_preemptions_total"):
                    sums[key.split(":", 1)[1]] = sums.get(key.split(":", 1)[1], 0.0) + float(value)
        row["vllm"] = sums
    except Exception:  # noqa: BLE001 - no server (yet)
        pass
    with open(folder / "sampler.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    progress = {}
    for gate in sorted(trial.glob("gate-*")):
        logs = list(gate.glob("attempts/*/log.jsonl"))
        done = cleared = tokens = 0
        open_tokens = []
        for log in logs:
            try:
                tail = log.read_bytes()[-20000:].decode(errors="ignore")
            except OSError:
                continue
            if not ('"kind": "finished"' in tail or '"kind":"finished"' in tail):
                used = 0  # an unfinished try's completion tokens so far (the watcher's wall-cut projection)
                for line in log.read_text(errors="ignore").splitlines():
                    if line.startswith('{"kind": "model"') or line.startswith('{"kind":"model"'):
                        try:
                            used += int((json.loads(line).get("usage") or {}).get("completion_tokens") or 0)
                        except ValueError:
                            pass
                open_tokens.append(used)
            if '"kind": "finished"' in tail or '"kind":"finished"' in tail:
                done += 1
                cleared += '"level_bench_clear"' in log.read_text(errors="ignore")
                for line in tail.splitlines()[::-1]:
                    if '"finished"' in line:
                        try:
                            tokens += int(json.loads(line).get("completion_tokens") or 0)
                        except ValueError:
                            pass
                        break
        progress[gate.name] = {"started": len(logs), "finished": done, "cleared": cleared,
                               "finished_completion_tokens": tokens, "open_tokens": sorted(open_tokens)}
    (folder / "progress.json").write_text(json.dumps({"t": row["t"], "gates": progress}, indent=1))


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0, max_containers=1,
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
        while not stop.wait(60):
            try:
                for name in ("feed.jsonl", "status.json"):
                    if (live / name).exists():
                        mirror.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(live / name, mirror / name)
                live_sample(live, output / "live", started)
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


@app.function(image=image, gpu=GPU, timeout=TEST_TIMEOUT, retries=0, max_containers=1,
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


@app.function(image=image, gpu="B300", timeout=1800, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def trainer_speed(run_id: str, attempts: str, profile: bool = False, attention: str = "", fit: str = ""):
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
                                                   attention=attention or None, fit=json.loads(fit) if fit else None))
        identity["status"] = report.get("status", "incomplete")
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


HEAL_TIMEOUT = 3300  # healing step 4: one B300 under 1 h
HEAL_STUDENT = "/assets/sft-heal/student-p75d24k4"


@app.function(image=image, gpu="B300", timeout=HEAL_TIMEOUT, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def heal_speed(run_id: str, max_tokens: int = 1_500_000, student_seconds: int = 600, window: int = 32768,
               mtp_weight: float = float(os.environ.get("ARC_HEAL_MTP_WEIGHT", "0.1")),
               general: int = int(os.environ.get("ARC_HEAL_GENERAL", "64")),
               k: int = int(os.environ.get("ARC_HEAL_K", "128")),
               energy: str = os.environ.get("ARC_HEAL_ENERGY", ""), only_energy: bool = False):
    """Healing trainer step 4: the teacher's top-32 files (rl1 trainer ``topk``, Flash NVFP4, adapter off) for the
    bundle's corpus segments, then the cut student's full-parameter KD on them (``heal_kd.py speed``, one rank)."""
    import asyncio
    import subprocess

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    sys.path.insert(0, str(source / "src"))
    from arena import rl_loop

    cfg = rl_loop.settings(bundle_study()["rl_loop"])
    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": "B300", "function_timeout": HEAL_TIMEOUT}
    report = {}

    def save():
        (output / "heal.json").write_text(json.dumps(report, indent=1, default=str))
        runs.commit()

    async def teacher():
        work = Path("/tmp/heal")
        command, env = rl_loop.trainer_command(cfg, source, work)
        trainer = rl_loop.Trainer(work / "trainer", command, env, output / "trainer.log")
        try:
            report["load"] = await trainer.ready(cfg["trainer_timeout"])
            if energy:  # width-cut scores first (a JSON list of rows with "ids" on the runs volume: excl5 usage set)
                rows = json.loads(Path(energy).read_text())
                report["energy"] = await trainer.call("channel_energy", timeout=1800, sequences=[r["ids"] for r in rows],
                                                      out=str(output / "channel-energy.pt"))
                save()
                if only_energy:
                    if not report["energy"].get("tokens"):
                        raise RuntimeError(f"channel energy failed: {str(report['energy'].get('error'))[:300]}")
                    return
            segments = [{"rollout": str(m.parent / "rollout.jsonl"), "name": m.parent.name}
                        for part in ("train", "probe") for m in sorted((source / "sft" / part).glob("*/meta.json"))]
            extra = {"mtp_from": HEAL_STUDENT} if mtp_weight else {}
            if general:
                extra["general"] = {"windows": general, "tokens": 2048, "out": str(output / "general")}
            topk = await trainer.call("topk", timeout=1800, segments=segments, out=str(output / "teacher"), k=k,
                                      max_tokens=max_tokens, **extra)
            report["topk"] = {k: v for k, v in topk.items() if k != "rows"}
            report["topk_rows"] = topk.get("rows")
            if not topk.get("ok", True) or not topk.get("sequences"):
                raise RuntimeError(f"teacher wrote no sequences: {str(topk.get('error'))[:300]}")
        finally:
            try:
                await asyncio.wait_for(trainer.call("exit", timeout=120), 150)
            except Exception:
                pass
            trainer.close()

    try:
        asyncio.run(teacher())
        save()
        if only_energy:
            identity["status"] = "complete"
            return identity
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
        env.update(PYTHONPATH=str(source / "rl1"), PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        command = [cfg["trainer_python"], "-m", "torch.distributed.run", "--nproc-per-node", "1",
                   str(source / "rl1/heal_kd.py"), "speed", "--student", HEAL_STUDENT, "--teacher",
                   str(output / "teacher"), "--out", str(output / "student"), "--seconds", str(student_seconds),
                   "--window", str(window), "--mtp-weight", str(mtp_weight)]
        if general:
            command += ["--general", str(output / "general")]
        if k > 32:
            command += ["--compare-k", "32", "64", str(k)]
        with open(output / "student.log", "w") as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=max(60, HEAL_TIMEOUT - (time.time() - started) - 120)).returncode
        report["student_exit"] = code
        identity["status"] = "complete" if code == 0 else "student_failed"
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        save()
    return identity


def balanced_split(specs, cards):
    """Segments to cards by estimated tokens (longest first onto the lightest card): a round-robin split left card 1
    with 3.44M of card 0's 4.11M window tokens and the schedule ran out of data (c1opd-v1). The estimate is the
    session's final context (prompt + completion of its last reply)."""
    def size(spec):
        last = None
        with open(spec["rollout"]) as f:
            for line in f:
                if line.strip():
                    last = line
        usage = (json.loads(last).get("usage") or {}) if last else {}
        return int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    parts, sizes = [[] for _ in range(cards)], [0] * cards
    for spec, n in sorted(((sp, size(sp)) for sp in specs), key=lambda x: -x[1]):
        c = sizes.index(min(sizes))
        parts[c].append(spec)
        sizes[c] += n
    return parts, sizes


def run_smoke(torchrun, kw, output, env, timeout, teacher_dirs=None):
    """Real-trainer smoke (gate): the launch's own train arguments on the real student and teacher
    files, 3 steps with a checkpoint after step 2 (A); a fresh process resumes that checkpoint and redoes step 3 (B):
    its kd / kd_reply / grad norm must match A's (exact resume); a third process loads A's saved student and optimizer
    as a next round does and takes one step (C). Verdict in smoke.json."""
    import subprocess

    import time as _t

    import os as _os
    import signal as _signal

    deadline = _t.time() + timeout
    # of the time left when each starts (A is the longest); B2 repeats B: the run-to-run noise of the resumed step
    shares = {"smoke-a": 0.5, "smoke-b": 0.4, "smoke-b2": 0.4, "smoke-c": 1.0}

    def gpu_free(limit=180):
        """Wait until no process holds card memory (smoke v4: torchrun's ranks outlived a killed parent and the next
        process OOMed at load)."""
        until = _t.time() + limit
        while _t.time() < until:
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=30).stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return True
            if not out:
                return True
            _t.sleep(5)
        return False

    def run(kw2, name):
        """One trainer process group; a hang in NCCL teardown after round_done (seen on B300) counts as finished:
        the whole group (torchrun and its ranks) is killed 120 s after round_done, and the cards must be free before
        the next process starts."""
        events = output / name / "events.jsonl"
        end, done_at = _t.time() + max(60, shares[name] * (deadline - _t.time())), None
        with open(output / f"{name}.log", "w") as log:
            proc = subprocess.Popen(torchrun(kw2, str(output / name)), env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                _t.sleep(10)
                if done_at is None and events.exists() and '"round_done"' in events.read_text():
                    done_at = _t.time()
                if (done_at and _t.time() - done_at > 120) or _t.time() > end:
                    break
            if proc.poll() is None or done_at:
                try:
                    _os.killpg(proc.pid, _signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait()
            if done_at is None and events.exists() and '"round_done"' in events.read_text():
                done_at = _t.time()
            code = proc.returncode if done_at is None or proc.returncode == 0 else 0
            if done_at is None and _t.time() > end:
                code = "timeout"
        freed = gpu_free()
        rows = [json.loads(x) for x in events.read_text().splitlines() if x.strip()] if events.exists() else []
        if not freed:
            rows.append({"event": "cards_not_freed"})
        return code, [r for r in rows if r.get("event") == "step"], rows

    staged = 0  # the teacher pass's reply positions: a batch of ~1/10 of them gives 3 steps with files left at step 2
    for folder in {Path(x) for x in (teacher_dirs or [output / "teacher"])}:
        index = folder / "index.json"
        if index.is_file():
            staged += sum(int(r.get("reply_tokens") or 0) for r in json.loads(index.read_text()).get("rows", []))
    n = int(kw.get("smoke_steps") or 3)  # A's steps; the checkpoint after step n-1, B redoes step n (2 fits 1 h)
    a = dict(kw, total_steps=n, checkpoint_at=str(n - 1), checkpoint_seconds=0, train_seconds=10 ** 6, no_save=False,
             heldout="", skip="", batch_tokens=max(8192, staged // 10))
    code_a, steps_a, _ = run(a, "smoke-a")
    ckpt = output / "smoke-a" / f"ckpt-{n - 1}"
    routing = kw.get("routing") or str(output / "smoke-a" / "routing.pt")  # B and C keep A's expert selection
    b = dict(a, student=str(ckpt / "student") if (ckpt / "student").is_dir() else a["student"], resume=str(ckpt),
             step0=n - 1, skip=str(ckpt / "progress.json"),
             checkpoint_at="", no_save=True, routing=routing)
    code_b, steps_b, _ = run(b, "smoke-b")
    code_b2, steps_b2, _ = run(b, "smoke-b2")
    c = dict(a, student=str(output / "smoke-a" / "student"), resume=str(output / "smoke-a"), step0=0, total_steps=2,
             checkpoint_at="", no_save=True, allow_unfinished=True, routing=routing)
    code_c, steps_c, _ = run(c, "smoke-c")
    a3 = next((r for r in steps_a if r["step"] == n), None)
    b3 = next((r for r in steps_b if r["step"] == n), None)
    b23 = next((r for r in steps_b2 if r["step"] == n), None)
    rel = lambda x, y: abs(x - y) / max(abs(x), abs(y), 1e-9)
    close = lambda x, y, bar: x is not None and y is not None and rel(x, y) <= bar
    # grad-norm bar: twice the run-to-run noise of the same resumed step (B vs B2), floor 1e-3
    noise = rel(b3["grad_norm"], b23["grad_norm"]) if b3 and b23 else None
    # floor 6% (opd-r1 gate, 22 Oct): one B/B2 pair under-reads the noise (A-B 5.2%, A-B2 3.1%, B-B2 2.1%, KD 2e-4)
    # floor 12% (opdS-g1c gate, 22 Oct: A-B 7.2% with B-B2 5.2%; earlier A-B 5.2%, 3.1%, 1.7%)
    gn_bar = max(2 * noise, float(os.environ.get("ARC_HEAL_GN_FLOOR", "0.12"))) if noise is not None else 0.0
    # HEAL_OPD_LOG: the step's trained objective and sampled-reverse estimate (forward-only numbers) must match too
    # bar: twice the same resumed step's run-to-run spread (B vs B2), floor ARC_HEAL_OPD_FLOOR (1e-2). Measured 22 Oct
    # (opdS-g1b / g1c gates, same flags): sampled reverse A-B 4.2e-3 / 2.2e-3, B-B2 4.7e-3 / 1.6e-4; objective A-B
    # 8.8e-5 / 2.1e-4. The estimate is IS-weighted on 160K sampled positions: its kernel noise exceeds kd's (A-B 2e-5 to
    # 2.4e-4), so one B/B2 pair under-reads it; kd and kd_reply keep their 1e-3 bars (a resume bug moves those)
    floor = float(os.environ.get("ARC_HEAL_OPD_FLOOR", "0.01"))
    opd_bar = lambda k: max(floor, 2 * rel(b3[k], b23[k])) if b3 and b23 and b3.get(k) is not None and b23.get(k) is not None else floor  # noqa: E731
    opd = all(a3.get(k) is None or close(a3[k], b3.get(k), opd_bar(k)) for k in ("opd_objective", "opd_sampled_rev")) if a3 and b3 else False
    exact = bool(a3 and b3 and close(a3["kd"], b3["kd"], 1e-3) and close(a3["kd_reply"], b3["kd_reply"], 1e-3)
                 and close(a3["grad_norm"], b3["grad_norm"], gn_bar) and a3["trained"] == b3["trained"] and opd)
    try:
        card = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=30).stdout.split()[0]) / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
        card = None
    peaks = [r.get("peak_gib") or 0.0 for r in steps_a + steps_b + steps_b2 + steps_c]
    peak_a = max((r.get("peak_gib") or 0.0 for r in steps_a), default=0.0)
    peak_c = max((r.get("peak_gib") or 0.0 for r in steps_c), default=0.0)
    # memory bars: the next round's peak within 2 GiB of A's, every stage at most 90% of the card
    memory_ok = bool(steps_c) and abs(peak_c - peak_a) <= 2.0 and (card is None or max(peaks) <= 0.9 * card)
    verdict = {"ok": code_a == 0 and code_b == 0 and code_b2 == 0 and code_c == 0 and len(steps_a) == n and exact
               and len(steps_c) >= 1 and memory_ok and all(r.get("finite") for r in steps_a + steps_b + steps_b2 + steps_c),
               "exit": [code_a, code_b, code_b2, code_c], "steps": [len(steps_a), len(steps_b), len(steps_b2), len(steps_c)],
               "a_step3": a3, "b_step3": b3, "b2_step3": b23, "grad_norm_noise": noise, "grad_norm_bar": gn_bar,
               "opd_bars": {k: opd_bar(k) for k in ("opd_objective", "opd_sampled_rev")} if a3 and b3 else None,
               "exact_resume": exact, "c_first": steps_c[0] if steps_c else None, "memory_ok": memory_ok,
               "card_gib": card, "lr_a": [r["lr"] for r in steps_a], "peak_gib": peaks}
    (output / "smoke.json").write_text(json.dumps(verdict, indent=1))
    return verdict


HEAL_ROUND_CARDS = int(os.environ.get("ARC_HEAL_CARDS", "2"))
HEAL_ROUND_TIMEOUT = int(os.environ.get("ARC_HEAL_TIMEOUT", "3300"))
# B200 bench of the heal trainer (teacher stays on B300); Modal per-card rates
HEAL_GPU = os.environ.get("ARC_HEAL_GPU", "B300")
HEAL_GPU_RATE = {"B300": GPU_RATE if CARDS == 1 else GPU_RATE / 2, "B200": 0.001736}[HEAL_GPU]


def env_pairs(spec):
    """"K=V,K=V" -> {K: V} (HEAL_* trainer flags, plus NCCL_DEBUG / NCCL_DEBUG_SUBSYS logging: MFU-1 24 Oct)."""
    out = {}
    for item in filter(None, (x.strip() for x in (spec or "").split(","))):
        k, _, v = item.partition("=")
        if not (k.startswith("HEAL_") or k in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS")):
            raise ValueError(f"trainer env {k!r}: only HEAL_* flags (and NCCL_DEBUG)")
        out[k] = v
    return out


def preempt_retry(output):
    """Create ``output``; when it already exists (Modal re-ran a preempted input), move its contents to
    ``output/preempted-<t>`` and return how to go on: the newest complete ``train/ckpt-<step>`` as resume args, and
    the round's own teacher folder when it was scored (no re-scoring). {} on a first run."""
    if not output.exists():
        output.mkdir(parents=True)
        return {}
    old = output / f"preempted-{int(time.time())}"
    old.mkdir()
    for p in list(output.iterdir()):
        if not p.name.startswith("preempted-"):
            p.rename(old / p.name)
    out = {"moved_to": str(old)}
    teachers = sorted(output.glob("preempted-*/teacher/index.json"))  # any earlier attempt's (repeat preemptions)
    if teachers:
        out["teacher_dir"] = str(teachers[-1].parent)
    ckpts = [p for p in output.glob("preempted-*/train/ckpt-*") if (p / "progress.json").is_file()]
    if ckpts:
        ck = max(ckpts, key=lambda p: int(p.name.split("-")[1]))
        progress = json.loads((ck / "progress.json").read_text())  # sharded (v25): the round's starting student
        student = ck / "student" if (ck / "student").is_dir() else progress.get("student")
        out.update(student=str(student), resume=str(ck), step0=int(ck.name.split("-")[1]),
                   skip=str(ck / "progress.json"))
        if (ck.parent / "routing.pt").is_file():
            out["routing"] = str(ck.parent / "routing.pt")
    return out


def restart_guard(output, retry, call_id=None):
    """every restart is an incident. Counts this call's attempts in ``output/attempts.json`` (Modal
    re-runs a preempted or memory-killed input under the same call id). On attempt >= 2 writes
    ``output/restart-alert.json``; with a complete checkpoint the round resumes from it, without one it stops here
    (returns the halt report) instead of training from step 0 again. ARC_HEAL_RESTART_GO=1 overrides (the agent's
    decision after reading the cause)."""
    if call_id is None:
        try:
            call_id = modal.current_function_call_id() or modal.current_input_id()
        except Exception:  # noqa: BLE001 - outside a Modal container
            call_id = None
    path = output / "attempts.json"
    # preempt_retry has just moved the last attempt's files (this book too) into the newest preempted-<t>
    books = [path] + sorted(output.glob("preempted-*/attempts.json"), key=lambda p: int(p.parent.name.split("-")[1]),
                            reverse=True)
    book = next((json.loads(b.read_text()) for b in books if b.is_file()), {})
    key = call_id or "local"
    attempt = int(book.get(key, 0)) + 1
    book[key] = attempt
    path.write_text(json.dumps(book))
    if attempt < 2:
        return None
    alert = {"t": round(time.time(), 1), "attempt": attempt, "call_id": call_id, "moved_to": retry.get("moved_to"),
             "resume": retry.get("resume"), "step0": retry.get("step0")}
    go = os.environ.get("ARC_HEAL_RESTART_GO") == "1"
    alert["action"] = "resume" if retry.get("resume") else ("from step 0 (ARC_HEAL_RESTART_GO)" if go else "halt")
    (output / "restart-alert.json").write_text(json.dumps(alert) + "\n")
    print("RESTART ALERT " + json.dumps(alert), flush=True)
    try:
        runs.commit()
    except Exception:  # noqa: BLE001 - outside a Modal container
        pass
    if retry.get("resume") or go:
        return None
    return {"halted": "restart without a complete checkpoint (no automatic step-0 reload)", **alert}


def stream_layout(cards, tranks=0):
    """A streaming round's cards: (trainer ranks, trainer devices, teacher devices, teacher spread). ARC_HEAL_TRANKS
    trainer ranks on cards 0..t-1 (0: cards - 2, the BF16 teacher's 2 cards), the teacher spread over the rest."""
    cards, t = int(cards), int(tranks or 0) or int(cards) - 2
    if not 1 <= t < cards:
        raise ValueError(f"ARC_HEAL_TRANKS {t} on {cards} cards: the trainer needs >= 1 card and the teacher >= 1")
    return t, ",".join(map(str, range(t))), ",".join(map(str, range(t, cards))), cards - t


CHAINS_OFF = ("RL1_STREAM_CHAINS > 1 is disabled: MultiForward chains 2 fails the GPU stream verify (top-1 0.94-0.96; "
              "batched MoE and the fla pin cleared, 23 Oct); use chains 1")


def stream_teacher_env(spread, pairs=""):
    """The streaming teacher's extra env from ARC_HEAL_TEACHER_ENV ("K=V,K=V"; MFU-2's one-card teacher: RL1_EXPERTS_FP8,
    RL1_STREAM_CHAINS, RL1_STREAM_PIPE). A one-card teacher needs RL1_EXPERTS_FP8=1 (BF16 Flash does not fit a card)."""
    allowed = ("RL1_EXPERTS_FP8", "RL1_STREAM_CHAINS", "RL1_STREAM_PIPE", "RL1_GDN_PIN_O", "RL1_CHAIN_LOCK",
               "ARC_TEACHER_CACHE",  # MFU-2: the versioned quack + hub cache (teacher_cache_*)
               "HEAL_STREAM_NEWEST",  # c7h 24 Oct: the freshest piece first (heal_stream._due)
               "RL1_QSA_TIES",  # c7j 24 Oct: QSA budget-edge ties to the lowest block index (rl1_trainer.TIES)
               "HEAL_VERIFY_SLICING",  # c7k 24 Oct: the stream verify's fresh-pass cuts (pieces | safe | sub)
               "HEAL_VERIFY_KL",  # c7j v7: the stream verify's KL bar (k x the bf16 call-shape floor)
               "RL1_TEACHER_RESERVE_GIB",  # c7m 24 Oct: the cache byte budget's floor of free room per piece
               "RL1_MEM_SNAPSHOT_GIB",  # c7o 25 Oct: CUDA memory history, snapshot once past this many GiB
               "HEAL_EVICT_CLOSING_FIRST",  # c7r 25 Oct: the cache budget evicts closing chains before open ones
               "HEAL_FORCE_IDS")  # 25 Oct: ids scored at every row beside the top-k ("+"-separated here)
    env = {}
    for item in (pairs or "").split(","):
        if item.strip():
            key, _, value = item.strip().partition("=")
            if key not in allowed:
                raise ValueError(f"ARC_HEAL_TEACHER_ENV: {key} is not a teacher flag ({', '.join(allowed)})")
            env[key] = value
    if int(env.get("RL1_STREAM_CHAINS") or 1) > 1:
        raise ValueError(CHAINS_OFF)
    if int(spread) == 1 and env.get("RL1_EXPERTS_FP8") != "1":
        raise ValueError("a one-card stream teacher needs RL1_EXPERTS_FP8=1 in ARC_HEAL_TEACHER_ENV")
    return env


def trainer_ranks(cards, stream_play="", tranks=0):
    """Trainer ranks of a heal round: a streaming round keeps the cards after its trainer ranks for the teacher."""
    return stream_layout(cards, tranks)[0] if stream_play else int(cards)


def heal_round_kwargs(e):
    """heal_round's keyword arguments from the ARC_HEAL_* settings (the launcher and heal_preflight.py)."""
    return {"segments": e["ARC_HEAL_SEGMENTS"], "student": e["ARC_HEAL_STUDENT"],
          "frozen_root": e.get("ARC_HEAL_FROZEN", "/assets/sft-heal/student-p50g24c3"),
          "resume": e.get("ARC_HEAL_RESUME", ""), "step0": int(e.get("ARC_HEAL_STEP0", "0")),
          "total_steps": int(e.get("ARC_HEAL_TOTAL_STEPS", "60")), "lr": float(e.get("ARC_HEAL_LR", "1e-5")),
          "general_windows": int(e.get("ARC_HEAL_GENERAL_WINDOWS", "256")),
          "routing": e.get("ARC_HEAL_ROUTING", ""), "train_seconds": int(e.get("ARC_HEAL_TRAIN_SECONDS", "1500")),
          "positions": e.get("ARC_HEAL_POSITIONS", "reply"), "teacher_dir": e.get("ARC_HEAL_TEACHER_DIR", ""),
          "teacher_tokens": int(e.get("ARC_HEAL_TEACHER_TOKENS", "0")),
          "teacher_chunk": int(e.get("ARC_HEAL_TEACHER_CHUNK", "16384")),
          "reverse": float(e.get("ARC_HEAL_REVERSE", "1.0")), "forward": float(e.get("ARC_HEAL_FORWARD", "0.1")),
          "batch_tokens": int(e.get("ARC_HEAL_BATCH_TOKENS", "500000")),
          "mtp_weight": float(e.get("ARC_HEAL_MTP_WEIGHT", "0.1")), "esft": float(e.get("ARC_HEAL_ESFT", "0.25")),
          "schedule": e.get("ARC_HEAL_SCHEDULE", "wsd"), "warmup": float(e.get("ARC_HEAL_WARMUP", "0.05")),
          "extra_teacher": e.get("ARC_HEAL_EXTRA_TEACHER", ""),
          "checkpoint_seconds": int(e.get("ARC_HEAL_CHECKPOINT_SECONDS", "1800")), "skip": e.get("ARC_HEAL_SKIP", ""),
          # the container does not see the launch env: module constants there fall back to defaults (3,300 s
          # killed opd1-v5's teardown and would have cut cut1-v1 at step ~10), so the limit travels as an argument
          "round_timeout": int(e.get("ARC_HEAL_TIMEOUT", "3300")), "cards": int(e.get("ARC_HEAL_CARDS", "2")),
          "tranks": int(e.get("ARC_HEAL_TRANKS", "0") or 0), "teacher_env": e.get("ARC_HEAL_TEACHER_ENV", ""),
          "truncate_loops": e.get("ARC_HEAL_TRUNCATE_LOOPS", "") == "1",
          "loop_share": float(e.get("ARC_HEAL_LOOP_SHARE", "0")), "seed": int(e.get("ARC_HEAL_SEED", "0")),
          "checkpoint_at": e.get("ARC_HEAL_CHECKPOINT_AT", ""), "heldout": e.get("ARC_HEAL_HELDOUT", ""),
          "heldout_files": int(e.get("ARC_HEAL_HELDOUT_FILES", "4")), "no_save": e.get("ARC_HEAL_NO_SAVE", "") == "1",
          "teacher_only": e.get("ARC_HEAL_TEACHER_ONLY", "") == "1", "smoke": e.get("ARC_HEAL_SMOKE", "") == "1",
          "teacher_model": e.get("ARC_HEAL_TEACHER_MODEL", ""), "stagger_teacher": e.get("ARC_HEAL_STAGGER", "") == "1",
          "gpu": e.get("ARC_HEAL_GPU", "B300"), "teacher_spread": int(e.get("ARC_HEAL_TEACHER_SPREAD", "1")),
          "smoke_steps": int(e.get("ARC_HEAL_SMOKE_STEPS", "3")),
          "window": int(e.get("ARC_HEAL_WINDOW", "32768")), "profile": e.get("ARC_HEAL_PROFILE", ""),
          "diag_window": int(e.get("ARC_HEAL_DIAG_WINDOW", "0")),
          "heal_env": e.get("ARC_HEAL_ENV", ""), "diag_env": e.get("ARC_HEAL_DIAG_ENV", ""),
          "gate": e.get("ARC_HEAL_GATE", "") == "1",
          # streaming OPD (22 Oct): the play runs to mirror (comma-separated), chunk size, age limit, bars
          "stream_play": e.get("ARC_HEAL_STREAM_PLAY", ""), "stream_poll": float(e.get("ARC_HEAL_STREAM_POLL", "10")),
          "stream_base": e.get("ARC_HEAL_STREAM_BASE", ""),  # base-model play runs (forward-KD rows, "b-" pieces)
          "stream_base_fill": e.get("ARC_HEAL_STREAM_BASE_FILL", ""),  # recorded base sessions (folder glob)
          "stream_chunk": int(e.get("ARC_HEAL_STREAM_CHUNK", "32768")),
          "stream_max_age": int(e.get("ARC_HEAL_STREAM_MAX_AGE", "6")),
          "stream_verify_after": int(e.get("ARC_HEAL_STREAM_VERIFY_AFTER", "4")),
          "stream_prune": e.get("ARC_HEAL_STREAM_PRUNE", ""),
          "stream_bar_step": float(e.get("ARC_HEAL_STREAM_BAR_STEP", "230")),
          "stream_bar_teacher": float(e.get("ARC_HEAL_STREAM_BAR_TEACHER", "2960")),
          "stream_bar_after": int(e.get("ARC_HEAL_STREAM_BAR_AFTER", "1000000"))}


def heal_train_args(kw, *, teacher, out, general):
    """``heal_kd.py train`` arguments of a heal_round launch (the launch and heal_preflight.py run these)."""
    args = ["train", "--student", kw["student"], "--frozen-root", kw["frozen_root"], "--teacher", teacher, "--out", out,
            "--lr", str(kw["lr"]), "--positions", kw["positions"], "--total-steps", str(kw["total_steps"]),
            "--step0", str(kw["step0"]), "--batch-tokens", str(kw["batch_tokens"]), "--esft", str(kw["esft"]),
            "--mtp-weight", str(kw["mtp_weight"]), "--reverse", str(kw["reverse"]), "--forward", str(kw["forward"]),
            "--seconds", str(kw["train_seconds"]), "--window", str(kw.get("window") or 32768), "--schedule", kw["schedule"],
            "--warmup", str(kw["warmup"])]
    if kw["general_windows"]:  # with teacher_dir: that round's general files
        args += ["--general", general, "--general-share", str(kw.get("general_share", 0.2))]
    if kw.get("routing"):
        args += ["--routing", kw["routing"]]
    if kw.get("resume"):
        args += ["--resume", kw["resume"]]
    if kw.get("checkpoint_seconds"):
        args += ["--checkpoint-seconds", str(kw["checkpoint_seconds"])]
    if kw.get("skip"):
        args += ["--skip", kw["skip"]]
    if kw.get("truncate_loops"):
        args += ["--truncate-loops"]
    if kw.get("loop_share"):
        args += ["--loop-share", str(kw["loop_share"])]
    if kw.get("seed"):
        args += ["--seed", str(kw["seed"])]
    if kw.get("checkpoint_at"):
        args += ["--checkpoint-at", str(kw["checkpoint_at"])]
    if kw.get("heldout"):
        args += ["--heldout", kw["heldout"], "--heldout-files", str(kw.get("heldout_files", 4))]
    if kw.get("no_save"):
        args += ["--no-save"]
    if kw.get("allow_unfinished"):
        args += ["--allow-unfinished-schedule"]
    return args


SC_ROOT, SC_WORK = "/runs/validation", "/tmp/stream-check"  # stream_check's dirs (the CPU gate in test_stream_pipe: tmp)


@app.function(image=image, gpu="B300", timeout=2400, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/assets": assets, "/runs": runs}, secrets=[modal.Secret.from_name("nebius-s3")])
def stream_check(run_id: str, files: str, teacher_model: str, near: str = "16384", limit: int = 40000,
                 fla_pin: str = "1,0", teacher_env: str = "RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=1,ARC_TEACHER_CACHE=1",
                 ties: str = "topk,index", extra: str = "{}"):
    """c7j diagnosis (24 Oct): the one-card stream teacher's cached continuation vs a fresh pass (rl1 op
    ``stream_check``) on real teacher files, with the round's own teacher env and quack cache. Output
    validation/<run>/stream-check.json."""
    import asyncio

    opts = json.loads(extra)  # v3: ref_single, subs, offsets, indexer, attention, state_check, kernels
    source = Path("/opt/arc-validation")
    sys.path.insert(0, str(source / "src"))
    from arena import rl_loop

    output = Path(SC_ROOT) / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cfg = rl_loop.settings(bundle_study()["rl_loop"])
    cfg["model_dir"] = teacher_model
    work = Path(SC_WORK)
    command, tenv = rl_loop.trainer_command(cfg, source, work)
    tpatch = stream_teacher_env(1, teacher_env)
    ckey, cver = teacher_cache_key(command[0])
    cdirs, seeded = teacher_cache_seed(ckey, cver)
    tenv = {**tenv, "CUDA_VISIBLE_DEVICES": "0", "RL1_PIN_QUACK": "1", "RL1_QUACK_REQUIRED": "1",
            "QUACK_CACHE_DIR": str(cdirs["quack"]), "QUACK_CACHE_AUTOTUNING": "1", **teacher_cache_env(cdirs), **tpatch}
    report = {"run_id": run_id, "started_epoch": started, "teacher_model": teacher_model, "teacher_env": tpatch,
              "cache_seeded": seeded, "opts": opts}
    sv = opts.get("stream_verify")
    if sv and sv.get("bucket_run"):  # k000 diagnosis: a recorded play session from the bucket, as the round pulls it
        import subprocess

        nebius_rclone_env()
        roll = Path(SC_WORK + "-roll")
        dst = roll / sv["bucket_run"] / sv.get("bucket_path", "")
        subprocess.run(["rclone", "copy", f"arc:{NEBIUS_BUCKET}/runs/validation/{sv['bucket_run']}/stream/trials/"
                        f"{sv.get('bucket_path', '')}", str(dst)], check=True, capture_output=True, timeout=900)
        sv["rollouts"] = str(roll)
        report["rollouts"] = sorted(str(f.relative_to(roll)) for f in roll.glob("**/rollout.jsonl"))
        if not report["rollouts"]:
            raise RuntimeError(f"no rollout.jsonl under {dst}")

    async def go():
        trainer = rl_loop.Trainer(work / "trainer", command, tenv, output / "teacher.log")
        try:
            report["load"] = (await trainer.ready(cfg["trainer_timeout"])).get("seconds")
            got = await trainer.call("stream_check", timeout=1800, files=[x for x in files.split(",") if x],
                                     near=[int(x) for x in near.split(",")], limit=limit,
                                     fla_pin=[x == "1" for x in fla_pin.split(",")], ties=ties.split(","),
                                     **opts)
            if not got.get("ok", True):  # the trainer answers an op failure instead of raising
                report["traceback"] = got.get("traceback")
                raise RuntimeError(f"op stream_check: {got.get('error')}")
            report["results"] = got.get("results")
        finally:
            try:
                await asyncio.wait_for(trainer.call("exit", timeout=120), 150)
            except Exception:  # noqa: BLE001
                pass
            trainer.close()

    try:
        asyncio.run(go())
        report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"[:2000]
    report["elapsed_seconds"] = round(time.time() - started, 1)
    (output / "stream-check.json").write_text(json.dumps(report, indent=1, default=str))
    runs.commit()
    return report


@app.function(image=image, gpu=os.environ.get("ARC_TEACHER_CHECK_GPU", "H100:2"), timeout=2700, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def teacher_check(run_id: str, segments: str, names: str, spread: int = 2, max_tokens: int = 400_000,
                  chunk: int = 4096, k: int = 128, real_model: str = "", blocking: bool = False):
    """Cheap-card check of the split teacher: a tiny real-vocabulary Flash teacher (heal_kd
    preflight-student; tokenizer and processor files from ``real_model``) runs the bundle's own rl1 trainer with
    ``--spread`` over two small cards and scores the named Stage A segments through ``op_topk`` (chunk, k as the
    launch), whole windows up to ``max_tokens``. Verdict: every requested window written or skipped by rule, the files
    readable with the heal_kd fields, layers on both cards, per-window peak per card."""
    import asyncio
    import shutil
    import subprocess

    import torch

    source = Path("/opt/arc-validation")
    sys.path.insert(0, str(source / "src"))
    from arena import rl_loop

    root, out = Path("/tmp/tc"), Path("/tmp/tc/teacher")
    root.mkdir(parents=True)
    p = subprocess.run([sys.executable, str(source / "rl1/heal_kd.py"), "preflight-student", "--out", str(root)],
                       capture_output=True, text=True, cwd=str(source / "rl1"))
    if p.returncode:
        return {"ok": False, "stage": "tiny", "stderr": p.stderr[-3000:]}
    tiny = root / "tiny-teacher"
    for f in Path(real_model).iterdir():  # tokenizer / processor / chat template; the tiny config and weights stay
        if f.is_file() and not f.name.endswith((".safetensors", ".json.index")) and f.name not in (
                "config.json", "model.safetensors.index.json", "generation_config.json"):
            shutil.copy(f, tiny / f.name)
    cfg = rl_loop.settings(bundle_study()["rl_loop"])
    cfg["model_dir"] = str(tiny)
    specs = [{"rollout": str(Path(segments) / n / "rollout.jsonl"), "name": n} for n in names.split(",")]
    need = {"ids", "top_ids", "top_logp", "tail", "reply", "image"}

    def run_once(sp, tag):
        """One trainer process (``sp`` cards) scoring the segments into /tmp/tc/<tag>/teacher."""
        work, out, log = root / tag, root / tag / "teacher", root / f"teacher-{tag}.log"
        command, env = rl_loop.trainer_command(cfg, source, work)
        env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(sp)),
               **({"CUDA_LAUNCH_BLOCKING": "1"} if blocking else {}),
               **({"RL1_PIN_QUACK": "1"} if tag == "split" else {})}  # base: unpinned, unsplit (the reference)
        if sp > 1:
            command = [*command, "--spread", str(sp)]
        trainer = rl_loop.Trainer(work / "trainer", command, env, log)
        r = {"spread": sp}

        async def go():
            try:
                r["loaded"] = await trainer.ready(cfg["trainer_timeout"])
                r["topk"] = await trainer.call("topk", timeout=2000, segments=specs, out=str(out), k=k,
                                               max_tokens=max_tokens, chunk=chunk)
            finally:
                try:
                    await asyncio.wait_for(trainer.call("exit", timeout=120), 150)
                except Exception:
                    pass
                trainer.close()

        try:
            asyncio.run(go())
        except Exception as exc:  # noqa: BLE001
            r["error"] = f"{type(exc).__name__}: {exc}"[:1500]
        ev_path = work / "trainer/trainer/events.jsonl"
        events = [json.loads(line) for line in ev_path.read_text().splitlines() if line.startswith("{")] \
            if ev_path.exists() else []
        r["spread_event"] = next(({k2: e.get(k2) for k2 in ("layers_per_device", "device_gib")} for e in events
                                  if e.get("event") == "spread"), None)
        r["windows"] = [{k2: e.get(k2) for k2 in ("segment", "tokens", "seconds", "peak_gib")} for e in events
                        if e.get("event") == "window"]
        files = sorted(out.glob("*.pt")) if out.exists() else []
        fields = sorted(torch.load(files[0], weights_only=False).keys()) if files else []
        topk = r.get("topk") or {}
        rows = topk.get("rows") or []
        r.update(files=len(files), fields=fields, rows=len(rows), skipped=sum("skipped" in x for x in rows),
                 log_tail=log.read_text(errors="ignore")[-3000:] if log.exists() else "")
        r["ok"] = bool(files) and need <= set(fields) and not r.get("error") and (
            sp == 1 or (r["spread_event"] is not None and all(n > 0 for n in r["spread_event"]["layers_per_device"])))
        return r, {f.name: f for f in files}

    def compare(a, b):
        """Split vs unsplit on the same windows: per position, the two top-k lists matched by
        token id; max |logp| difference on shared ids, top-1 agreement, mean KL(unsplit || split) over the unsplit
        top-k mass (ids missing from the split list take its tail)."""
        out = {"files": 0, "positions": 0, "max_abs_logp": 0.0, "kl_sum": 0.0, "top1_agree": 0, "ids_equal": True}
        for name in sorted(set(a) & set(b)):
            x, y = torch.load(a[name], weights_only=False), torch.load(b[name], weights_only=False)
            out["ids_equal"] &= bool(torch.equal(x["ids"], y["ids"]))
            ia, la, ib, lb = x["top_ids"].long(), x["top_logp"].float(), y["top_ids"].long(), y["top_logp"].float()
            tb = y["tail"].float()
            for s0 in range(0, ia.shape[0], 2048):
                A, B, LA, LB = ia[s0:s0 + 2048], ib[s0:s0 + 2048], la[s0:s0 + 2048], lb[s0:s0 + 2048]
                eq = A[:, :, None] == B[:, None, :]  # [n, k, k]
                hit = eq.any(-1)
                lb_on_a = torch.where(hit, (eq.float() * LB[:, None, :]).sum(-1),
                                      tb[s0:s0 + 2048, None].expand_as(LA) - torch.log(torch.tensor(float(LB.shape[1]))))
                diff = (LA - lb_on_a).abs().masked_fill(~hit, 0)
                out["max_abs_logp"] = max(out["max_abs_logp"], float(diff.max()))
                out["kl_sum"] += float((LA.exp() * (LA - lb_on_a)).sum())
                out["top1_agree"] += int((A[:, 0] == B[:, 0]).sum())
                out["positions"] += A.shape[0]
            out["files"] += 1
        n = max(out["positions"], 1)
        out["mean_kl"] = out.pop("kl_sum") / n
        out["top1_agree"] = out["top1_agree"] / n
        return out

    res = {"run_id": run_id, "gpus": torch.cuda.device_count(), "names": names, "spread": spread}
    split, split_files = run_once(spread, "split")
    res["split"] = split
    if spread > 1:
        base, base_files = run_once(1, "base")
        res["base"] = {k2: v for k2, v in base.items() if k2 != "log_tail"}
        res["compare"] = compare(base_files, split_files) if base["ok"] and split["ok"] else None
    c = res.get("compare")
    # BF16 tolerance: the same weights, a different layer placement only (no math change), so near-identical
    res["equivalent"] = spread == 1 or bool(c and c["ids_equal"] and c["files"] == split["files"] and
                                            c["mean_kl"] <= 1e-3 and c["top1_agree"] >= 0.99)
    res["ok"] = split["ok"] and res["equivalent"]
    res["log_tail"] = split["log_tail"]
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=True)
    for f in root.glob("teacher-*.log"):
        shutil.copy(f, output / f.name)
    (output / "teacher-check.json").write_text(json.dumps(res, indent=1, default=str))
    runs.commit()
    return res


def round_paths(run_id, teacher_dir, extra_teacher):
    """heal_round's teacher argument and general folder (the round and its CPU dry run share them)."""
    output = Path("/runs/validation") / run_id
    teacher_out = Path(teacher_dir.split(",")[0]) if teacher_dir else output / "teacher"  # general files: the first
    teacher_arg = ",".join(x for x in ([teacher_dir] if teacher_dir else [str(output / "teacher")]) + [extra_teacher] if x)
    return teacher_arg, str(teacher_out.parent / "general" if teacher_dir else output / "general")


@app.function(image=image, timeout=3600, retries=0, volumes={"/assets": assets, "/runs": runs})
def heal_plan(run_id: str, **kw):
    """The mandatory CPU dry run of a heal_round launch: ``heal_kd.py plan`` on the round's own
    arguments (heal_train_args), world = cards, the round's HEAL_* env, over the real files; no model, no GPU.
    A round that scores its own teacher files first is planned on its segments only."""
    import subprocess

    source = Path("/opt/arc-validation")
    if not kw.get("teacher_dir"):
        found = sorted(Path(kw["segments"]).glob("*/rollout.jsonl"))
        return {"ok": bool(found), "segments": len(found), "note": "teacher files are written in the run: not planned"}
    teacher_arg, general_dir = round_paths(run_id, kw["teacher_dir"], kw.get("extra_teacher", ""))
    out = f"/tmp/plan-{run_id}"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
    env.update(PYTHONPATH=str(source / "rl1"), PYTHONUNBUFFERED="1", **env_pairs(kw.get("heal_env", "")))
    args = heal_train_args(dict(kw, general_share=kw.get("general_share", 0.2),  # a stream round: its own schedule rule
                                allow_unfinished=kw.get("allow_unfinished") or bool(kw.get("stream_play"))),
                           teacher=teacher_arg, out=out, general=general_dir)
    sys.path.insert(0, str(source / "src"))
    from arena import rl_loop

    python = rl_loop.settings(bundle_study()["rl_loop"])["trainer_python"]  # the trainer's own interpreter
    command = [python, str(source / "rl1/heal_kd.py"), "plan", *args[1:], "--world",
               str(trainer_ranks(kw.get("cards", 2), kw.get("stream_play"), kw.get("tranks", 0)))]
    r = subprocess.run(command, env=env, capture_output=True, text=True)
    plan_file = Path(out) / "plan.json"
    plan = json.loads(plan_file.read_text()) if plan_file.is_file() else {}
    plan.pop("files_detail", None)
    return {"ok": r.returncode == 0, "plan": plan, "tail": (r.stdout + r.stderr)[-3000:]}


NEBIUS_BUCKET = "arc-runs-bucket"
QUACK_CACHE = "/runs/quack-cache/teacher-b300"  # the stream teacher's quack .o + autotune cache (persistent)
QUACK_SAVE_AFTER = 1500  # s after the teacher starts: its first pieces have compiled every kernel


# MFU-2 (23 Oct): every teacher start spent 84-120 s on its first piece (c1-c4) fetching sonic-moe's hub build and
# compiling its vendored quack; the quack cache and the hub snapshot persist under a key of the teacher venv's
# kernel/runtime versions (another venv never loads them), saved once the first pieces ran (preempted legs too)
TEACHER_CACHE = "/runs/teacher-cache"  # <key>/{quack,hub,cute,versions.json}
TEACHER_CACHE_DISTS = ("torch", "triton", "quack-kernels", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
                       "kernels", "huggingface-hub", "transformers", "axolotl", "sonicmoe", "nvidia-cuda-runtime-cu13",
                       "nvidia-cuda-nvrtc-cu13")
QUACK_SAVE_PIECES = 3  # the teacher log's stream_piece count that triggers the first save
CACHE_DIRS = ("quack", "hub", "cute")  # quack autotune picks, kernels-hub snapshots, CuTe DSL compiled kernels
CACHE_SKIP = ("*.lock", "lock", "*.incomplete", "blobs", ".locks")  # hub snapshots are copied as files: no blobs


def teacher_cache_key(python, dists=TEACHER_CACHE_DISTS, run=None):
    """(key, versions): a hash of ``python``'s version and its installed versions of ``dists`` (None: absent); no
    torch there is an error, never an empty key."""
    import subprocess

    script = ("import importlib.metadata as m, json, platform, sys\n"
              "out = {'python': sys.version.split()[0], 'machine': platform.machine()}\n"
              "for d in json.loads(sys.argv[1]):\n"
              "    try:\n        out[d] = m.version(d)\n    except m.PackageNotFoundError:\n        out[d] = None\n"
              "print('ARC_VERSIONS ' + json.dumps(out, sort_keys=True))")
    r = (run or subprocess.run)([python, "-c", script, json.dumps(list(dists))], capture_output=True, text=True,
                                timeout=120)
    line = next((x for x in r.stdout.splitlines()[::-1] if x.startswith("ARC_VERSIONS ")), None)
    if r.returncode or line is None:
        raise RuntimeError(f"teacher cache key: {python} failed: {(r.stdout + r.stderr)[-500:]}")
    versions = json.loads(line[len("ARC_VERSIONS "):])
    if not versions.get("torch"):
        raise RuntimeError(f"teacher cache key: no torch in {python}")
    return hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()[:16], versions


def teacher_cache_seed(key, versions, local="/tmp/teacher-cache", root=None):
    """Local caches (CACHE_DIRS) seeded from ``root/key`` (its versions.json must equal ``versions``, else nothing is
    seeded). Returns ({name: local dir}, seeded: bool)."""
    import shutil

    vol, local = Path(root or TEACHER_CACHE) / key, Path(local)
    dirs = {name: local / name for name in CACHE_DIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    meta = vol / "versions.json"
    seeded = meta.is_file() and json.loads(meta.read_text()) == versions
    if meta.is_file() and not seeded:
        print(f"teacher cache {vol}: versions differ, not seeded", flush=True)
    if seeded:
        for name, dst in dirs.items():
            if (vol / name).is_dir():
                shutil.copytree(vol / name, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*CACHE_SKIP))
    return dirs, bool(seeded)


def teacher_cache_env(dirs):
    """The teacher's env for the local caches, as each package reads it (sources checked in the teacher venv, 23 Oct):
    quack 0.6.1 QUACK_CACHE_DIR (autotune picks only); kernels KERNELS_CACHE (kernels/hf_hub.py CACHE_DIR, passed to
    snapshot_download; c6's HF_KERNELS_CACHE was ignored: hub 0 files); cutlass-dsl CUTE_DSL_CACHE_DIR (the JIT's
    compiled-kernel file cache, default $TMPDIR/<user>/cutlass_python_cache: lost with every container)."""
    return {"QUACK_CACHE_DIR": str(dirs["quack"]), "KERNELS_CACHE": str(dirs["hub"]),
            "CUTE_DSL_CACHE_DIR": str(dirs["cute"])}


def teacher_cache_save(key, versions, dirs, root=None, lock=None):
    """Copies the local caches to ``root/key`` (hub symlinks followed: plain files) and writes versions.json last.
    Returns the volume's file counts per cache (hub 0 after a fetch: the kernels package wrote elsewhere), or False
    on a failure (logged, never raised)."""
    import shutil
    import threading

    lock = lock or threading.Lock()
    vol = Path(root or TEACHER_CACHE) / key
    try:
        with lock:
            for name, src in dirs.items():
                if Path(src).is_dir():
                    shutil.copytree(src, vol / name, dirs_exist_ok=True, symlinks=False,
                                    ignore=shutil.ignore_patterns(*CACHE_SKIP))
            (vol / "versions.json").write_text(json.dumps(versions, sort_keys=True, indent=1))
            runs.commit()
        return {name: sum(1 for x in (vol / name).rglob("*") if x.is_file()) for name in dirs}
    except Exception as exc:  # noqa: BLE001 - a cache, not data
        print(f"teacher cache save failed: {type(exc).__name__}: {exc}"[:300], flush=True)
        return False


def teacher_cache_saver(log, save, *, pieces=QUACK_SAVE_PIECES, after=None, poll=10.0, done=None):
    """Thread body: ``save()`` once ``log`` holds ``pieces`` stream_piece events, and again ``after`` s from the start
    (kernels the later pieces compiled); stops when ``done`` (an Event) is set."""
    began, saved = time.time(), False
    while not (done is not None and done.is_set()):
        if not saved:
            try:
                saved = Path(log).read_text(errors="replace").count('"event": "stream_piece"') >= pieces
            except OSError:
                saved = False
            if saved:
                save()
        if after is not None and time.time() - began >= after:
            save()
            return
        if done is not None:
            done.wait(poll)
        else:
            time.sleep(poll)


def quack_cache_seed(local="/tmp/quack-cache"):
    """The teacher's local quack cache, seeded from the volume copy (missing: an empty cache, compiled once)."""
    import shutil

    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    if Path(QUACK_CACHE).is_dir():
        shutil.copytree(QUACK_CACHE, local, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.lock", "lock"))
    return local


TEACHER_RESTARTS = int(os.environ.get("ARC_HEAL_TEACHER_RESTARTS", "3"))  # coloc-v5 IMA: a dead teacher restarts


async def teacher_with_restarts(start, stop_file, restarts, report, *, limit=TEACHER_RESTARTS, log_tail=None):
    """Run the stream teacher; when its process dies before ``stop_file`` exists (e.g. a CUDA illegal memory
    access, coloc-v5 23 Oct), log the error with its log tail (the kernel context) in ``restarts`` and start it
    again (skip_existing: the trainers keep consuming what is already written), up to ``limit`` times.
    ``start(attempt)`` -> (trainer, coroutine factory for its stream_topk call); returns the last call's result."""
    import asyncio

    for attempt in range(limit + 1):
        trainer, run = start(attempt)
        try:
            report["stream_teacher"] = await run(trainer, attempt)
            return report["stream_teacher"]
        except Exception as exc:  # noqa: BLE001 - the restart decides
            err = f"{type(exc).__name__}: {exc}"[:1000]
            report["stream_teacher"] = {"error": err}
            tail = (log_tail(attempt) if log_tail else "")[-4000:]
            if "verify failed" in tail:  # a wrong teacher is not a crash: never restart past it (opdS-c1, 23 Oct)
                report["stream_teacher"]["verify_failed"] = True
                return report["stream_teacher"]
            if Path(stop_file).exists() or attempt >= limit:
                return report["stream_teacher"]
            restarts.append({"attempt": attempt, "at": time.strftime("%H:%M:%S", time.gmtime()), "error": err,
                             "log_tail": tail})
        finally:
            try:
                await asyncio.wait_for(trainer.call("exit", timeout=120), 150)
            except Exception:  # noqa: BLE001
                pass
            trainer.close()
    return report["stream_teacher"]


def quack_cache_save(local):
    """Copy the local quack cache to the volume (new files only; a failure is logged, never raised)."""
    import shutil

    try:
        shutil.copytree(local, QUACK_CACHE, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.lock", "lock"))
        runs.commit()
    except Exception as exc:  # noqa: BLE001 - a cache, not data
        print(f"quack cache save failed: {type(exc).__name__}: {exc}"[:300], flush=True)
STREAM_PULL_FLAGS = ["--update", "--transfers", "16", "--checkers", "32", "--retries", "2"]  # heal_preflight runs these too


NEBIUS_RCLONE = {"RCLONE_CONFIG_ARC_TYPE": "s3", "RCLONE_CONFIG_ARC_PROVIDER": "Other", "RCLONE_CONFIG_ARC_ENV_AUTH": "true",
                 "RCLONE_CONFIG_ARC_ENDPOINT": "https://storage.eu-south1.nebius.cloud",
                 "RCLONE_CONFIG_ARC_REGION": "eu-south1", "RCLONE_CONFIG_ARC_NO_CHECK_BUCKET": "true"}


def nebius_rclone_env():
    """rclone remote ``arc:`` for the Nebius bucket (credentials from the ``nebius-s3`` secret)."""
    os.environ.update(NEBIUS_RCLONE)


def follow_path(play):
    """A play's weights-follow.jsonl (heal_delta confirm)."""
    return f"arc:{NEBIUS_BUCKET}/runs/validation/{play}/arena/working/trials/duck-transfer/weights-follow.jsonl"


def stream_knobs(knobs):
    """The round's live knobs (24 Oct), from ``<run>/stream/knobs.json``: {"plays": [run, ...], "confirm": [subset],
    "lr": x, "ckpt": tag}, each optional. plays: the pulled student plays; confirm: the plays whose applied generation
    ends full versions (default every listed play); lr: the trainer's peak lr (heal_kd LiveLr, the schedule scaled by
    x / --lr); ckpt: a short tag, a new value checkpoints at the next step boundary (c7n 25 Oct; the cadence goes on).
    Returns {"plays", "confirm", "lr", "ckpt"} (None: unchanged); ValueError on anything else (the round keeps its knobs)."""
    import re
    if not isinstance(knobs, dict) or not knobs or set(knobs) - {"plays", "confirm", "lr", "ckpt"}:
        raise ValueError(f"knobs are {{'plays': [...], 'confirm': [...], 'lr': x, 'ckpt': tag}}; got {knobs!r}")
    out = {"plays": None, "confirm": None, "lr": None, "ckpt": None}
    if "ckpt" in knobs:
        if not isinstance(knobs["ckpt"], (str, int)) or isinstance(knobs["ckpt"], bool) or not re.fullmatch(
                r"[A-Za-z0-9._:-]{1,40}", str(knobs["ckpt"])):
            raise ValueError(f"ckpt must be a short tag; got {knobs!r}")
        out["ckpt"] = str(knobs["ckpt"])
    if "lr" in knobs:
        if isinstance(knobs["lr"], bool) or not isinstance(knobs["lr"], (int, float)) or not 0 < knobs["lr"] <= 1e-4:
            raise ValueError(f"lr must be a number in (0, 1e-4]; got {knobs!r}")
        out["lr"] = float(knobs["lr"])
    if "confirm" in knobs and "plays" not in knobs:
        raise ValueError(f"confirm needs plays; got {knobs!r}")
    if "plays" in knobs:
        plays = knobs["plays"]
        confirm = knobs.get("confirm", plays)
        if not isinstance(plays, list) or not isinstance(confirm, list) or not all(
                isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9._-]+", x) for x in plays + confirm) \
                or set(confirm) - set(plays):
            raise ValueError(f"plays must be run names and confirm a subset of them; got {knobs!r}")
        out["plays"], out["confirm"] = list(dict.fromkeys(plays)), list(dict.fromkeys(confirm))
    return out


def stream_trainer_env(env, *, chunks, k, max_age, deltas, run_id, prune="", devices="0,1", play="", confirm_live="",
                       live_knobs=""):
    """The streaming round's trainer env (heal_kd): its own ``arc:`` remote for the delta upload (g1l, 23 Oct: the env
    was snapshotted before nebius_rclone_env, so the first upload failed: rclone had no ``arc:``). The CPU preflight
    builds its round-a env with this function (delta_bucket check)."""
    tenv = {**env, **NEBIUS_RCLONE, "CUDA_VISIBLE_DEVICES": devices, "HEAL_STREAM": str(chunks), "HEAL_STREAM_K": str(k),
            "HEAL_STREAM_MAX_AGE": str(max_age), "HEAL_DELTA_DIR": str(deltas),
            # HEAL_DELTA_RUN (24 Oct): a resumed restart under a new run id keeps publishing where its play follows
            "HEAL_DELTA_REMOTE": f"arc:{NEBIUS_BUCKET}/runs/validation/{env.get('HEAL_DELTA_RUN') or run_id}/deltas",
            "HEAL_OPD_LOG": "1"}
    if prune:
        tenv["HEAL_DELTA_PRUNE"] = prune
    play = play.split(",")[0]  # the live play leads the list; the rest are recorded backlogs (c7b, 24 Oct)
    if play:  # full-row versions until the play has applied this trainer generation (heal_delta.play_confirm)
        tenv["HEAL_DELTA_CONFIRM"] = follow_path(play)
        if confirm_live:  # the list the round's puller rewrites from the live play knobs (heal_delta.live_confirm)
            tenv["HEAL_DELTA_CONFIRM_LIVE"] = str(confirm_live)
    if live_knobs:  # the trainer's own live knobs (lr) the round's puller writes (heal_kd LiveLr)
        tenv["HEAL_LIVE_KNOBS"] = str(live_knobs)
    tenv.setdefault("HEAL_SAMPLED", "1.0")
    return tenv


@app.function(image=image, gpu=f"{HEAL_GPU}:{HEAL_ROUND_CARDS}", timeout=HEAL_ROUND_TIMEOUT, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs}, secrets=[modal.Secret.from_name("nebius-s3")])
def heal_round(run_id: str, segments: str, student: str, frozen_root: str, resume: str = "", step0: int = 0,
               total_steps: int = 60, lr: float = 1e-5, general_windows: int = 256, k: int = 128,
               batch_tokens: int = 500_000, esft: float = 0.25, routing: str = "", mtp_weight: float = 0.1,
               reverse: float = 1.0, forward: float = 0.1, general_share: float = 0.2, train_seconds: int = 1500,
               positions: str = "reply", teacher_dir: str = "", teacher_tokens: int = 0, teacher_chunk: int = 16384,
               schedule: str = "wsd", warmup: float = 0.05, extra_teacher: str = "", checkpoint_seconds: int = 1800,
               skip: str = "", round_timeout: int = 3300, cards: int = 2, truncate_loops: bool = False,
               loop_share: float = 0.0, seed: int = 0, checkpoint_at: str = "", heldout: str = "", heldout_files: int = 4,
               no_save: bool = False, teacher_only: bool = False, smoke: bool = False, teacher_model: str = "",
               stagger_teacher: bool = False, gpu: str = "B300", teacher_spread: int = 1, smoke_steps: int = 3,
               window: int = 32768, profile: str = "", diag_window: int = 0, heal_env: str = "", diag_env: str = "",
               gate: bool = False, stream_play: str = "", stream_poll: float = 10.0, stream_chunk: int = 32768,
               stream_max_age: int = 6, stream_verify_after: int = 4, stream_prune: str = "", stream_base: str = "", stream_base_fill: str = "",
               stream_bar_step: float = 230.0, stream_bar_teacher: float = 2960.0, stream_bar_after: int = 1_000_000,
               tranks: int = 0, teacher_env: str = ""):
    """One OPD healing round on ``cards`` B300 (PTH, 18 Oct): (1) one teacher per card (rl1 trainer
    ``topk``, Flash NVFP4, adapter off; segments split round-robin; card 0 also the ``general_windows`` wikitext
    windows) scores the play in ``segments`` (a folder of ``<name>/rollout.jsonl``), top-``k`` + tail, at most
    ``teacher_tokens`` per card (0: all); ``teacher_dir``: an earlier round's teacher folder instead (no scoring);
    (2) ``heal_kd.py train`` on every card (FSDP): ``reverse`` / ``forward`` KL on ``positions`` (reply: the
    student's own replies, OPD; all: every position, the offline warm start), the general files by forward KL at
    ``general_share`` of trained positions, ESFT experts, MTP head; saves the student and the optimizer state.
    nvidia-smi every 10 s to gpu.jsonl (utilization and memory per card).

    ``stream_play`` (streaming OPD, 22 Oct): the Nebius play run whose ``stream/trials`` rollouts are mirrored here
    every ``stream_poll`` s; cards 2-3 run the BF16 teacher (``teacher_model``) as ``stream_topk`` into
    ``stream-chunks/``; cards 0-1 train on them (HEAL_STREAM) and export each step's weights to the bucket
    (``<run_id>/deltas``) for the play server; first-phase bars (trainer step compute, teacher tok/s) stop the run
    on a miss. ``teacher_dir``: the earlier round whose general files mix in."""
    import asyncio
    import subprocess
    import threading

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    retry = preempt_retry(output)  # Modal re-runs a preempted container's input (s0-b3, 21 Oct)
    halt = restart_guard(output, retry)
    if halt:
        return halt
    if retry.get("teacher_dir") and not teacher_dir:
        teacher_dir = retry["teacher_dir"]
    if retry.get("resume"):
        student, resume, step0, skip = retry["student"], retry["resume"], retry["step0"], retry["skip"]
        routing = routing or retry.get("routing", "")
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    sys.path.insert(0, str(source / "src"))
    from arena import rl_loop

    cfg = rl_loop.settings(bundle_study()["rl_loop"])
    if teacher_model:  # BF16 variant (20 Oct): e.g. /assets/expert-bf16/<rev> in place of the NVFP4 checkpoint
        cfg["model_dir"] = teacher_model
    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": f"{gpu}:{cards}", "student": student,
                "teacher_model": cfg["model_dir"], "preempt_retry": retry or None,
                "segments": segments, "step0": step0, "total_steps": total_steps, "positions": positions,
                # where Modal placed this container (c4, 23 Oct: three preemptions in 47 min; region pinning read)
                "placement": {k: os.environ.get(k) for k in ("MODAL_REGION", "MODAL_CLOUD_PROVIDER", "MODAL_TASK_ID")}}
    report = {}
    stop_smi = threading.Event()

    def smi():
        with open(output / "gpu.jsonl", "a") as f:
            while not stop_smi.wait(10):
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                         timeout=20).stdout
                    cards = [[float(x) for x in line.split(",")] for line in out.strip().splitlines()]
                    f.write(json.dumps({"t": round(time.time() - started), "epoch": round(time.time(), 1),
                                        "phase": report.get("phase"),
                                        "util": [c[1] for c in cards], "mem_gib": [round(c[2] / 1024, 1) for c in cards]})
                            + "\n")
                    f.flush()
                    tick = getattr(smi, "tick", 0) + 1
                    smi.tick = tick
                    if tick % 6 == 0:  # live watcher: gpu.jsonl, teacher logs, train/events.jsonl each minute
                        runs.commit()
                except Exception:  # noqa: BLE001
                    pass

    threading.Thread(target=smi, daemon=True).start()
    teacher_out = Path(teacher_dir.split(",")[0]) if teacher_dir else output / "teacher"  # general files: the first
    teacher_arg, general_dir = round_paths(run_id, teacher_dir, extra_teacher)

    def save():
        (output / "round.json").write_text(json.dumps(report, indent=1, default=str))
        runs.commit()

    loads_done = {}

    async def one_teacher(card, segs, general):
        work = Path(f"/tmp/heal{card}")
        if stagger_teacher and card:  # one host-memory load peak at a time (a BF16 load reads 335 GiB)
            await loads_done[card - 1].wait()
        command, env = rl_loop.trainer_command(cfg, source, work)
        # teacher_spread: one teacher's layers over that many cards
        devices = ",".join(str(card * teacher_spread + i) for i in range(teacher_spread))
        env = {**env, "CUDA_VISIBLE_DEVICES": devices, "RL1_PIN_QUACK": "1"}  # one tuned config for every last-chunk size
        if teacher_spread > 1:
            command = [*command, "--spread", str(teacher_spread)]
        trainer = rl_loop.Trainer(work / "trainer", command, env, output / f"teacher{card}.log")
        try:
            try:
                loaded = await trainer.ready(cfg["trainer_timeout"])
            finally:
                loads_done[card].set()
            extra = {"general": {"windows": general, "tokens": 2048, "out": str(output / "general")}} if general else {}
            topk = await trainer.call("topk", timeout=round_timeout, segments=segs, out=str(teacher_out),
                                      k=k, max_tokens=teacher_tokens or 10 ** 12, chunk=teacher_chunk, **extra)
            return {"card": card, "load_seconds": loaded.get("seconds"),
                    **{key: v for key, v in topk.items() if key != "rows"}}
        finally:
            try:
                await asyncio.wait_for(trainer.call("exit", timeout=120), 150)
            except Exception:
                pass
            trainer.close()

    async def teacher():
        found = sorted(Path(segments).glob("*/rollout.jsonl"))
        if smoke:
            found = found[:16]  # the smoke scores a few real segments
        specs = [{"rollout": str(m), "name": m.parent.name} for m in found]
        report["segments"] = len(specs)
        teachers = max(1, cards // teacher_spread)
        parts, sizes = balanced_split(specs, teachers)
        report["staged_estimate_per_card"] = sizes
        general = min(general_windows, 48) if smoke else general_windows  # 16 probe windows + the mix
        for c in range(teachers):
            loads_done[c] = asyncio.Event()
        results = await asyncio.gather(*[one_teacher(c, parts[c], general if c == 0 else 0)
                                         for c in range(teachers)], return_exceptions=True)
        report["topk"] = [r if isinstance(r, dict) else {"error": f"{type(r).__name__}: {r}"[:400]} for r in results]
        if not sum(r.get("sequences") or 0 for r in report["topk"]):
            raise RuntimeError(f"teacher wrote no sequences: {report['topk']}")

    try:
        if not teacher_dir:
            report["phase"] = "teacher"
            asyncio.run(teacher())
        report["teacher_seconds"] = round(time.time() - started, 1)
        if teacher_only:
            identity["status"] = "complete"
            report["phase"] = "done"
            return identity
        report["phase"] = "train"
        save()
        env = {k2: v for k2, v in os.environ.items() if not k2.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
        env.update(PYTHONPATH=str(source / "rl1"), PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        env.setdefault("HEAL_ROUTE_CACHE", "/runs/cache/routing")
        env.update(env_pairs(heal_env))  # ARC_HEAL_ENV "K=V,K=V": trainer flags (e.g. HEAL_CAPTURE_CACHE=1) for every run
        if profile:  # HEAL_PROFILE=A-B: torch.profiler over steps A..B (trainer MFU)
            env["HEAL_PROFILE"] = profile
        kw = dict(student=student, frozen_root=frozen_root, lr=lr, positions=positions, total_steps=total_steps,
                  step0=step0, batch_tokens=batch_tokens, esft=esft, mtp_weight=mtp_weight, reverse=reverse,
                  forward=forward, train_seconds=train_seconds, schedule=schedule, warmup=warmup,
                  general_windows=general_windows, general_share=general_share, routing=routing, resume=resume,
                  checkpoint_seconds=checkpoint_seconds, skip=skip, truncate_loops=truncate_loops,
                  loop_share=loop_share, seed=seed, checkpoint_at=checkpoint_at, heldout=heldout,
                  heldout_files=heldout_files, no_save=no_save, window=window)

        def torchrun(kw2, out_dir):
            return [cfg["trainer_python"], "-m", "torch.distributed.run", "--nproc-per-node",
                    str(trainer_ranks(cards, stream_play, tranks)),
                    str(source / "rl1/heal_kd.py"), *heal_train_args(kw2, teacher=teacher_arg, out=out_dir,
                                                                    general=general_dir)]
        if smoke:
            report["smoke"] = run_smoke(torchrun, dict(kw, smoke_steps=smoke_steps), output, env,
                                        round_timeout - (time.time() - started) - 120,
                                        teacher_dirs=[x for x in teacher_arg.split(",") if x])
            identity["status"] = "complete" if report["smoke"]["ok"] else "smoke_failed"
            return identity
        if gate:  # first-phase gate: the smoke on this run's own args and data; a mismatch stops the run
            report["phase"] = "gate"
            save()
            report["gate"] = run_smoke(torchrun, dict(kw, smoke_steps=smoke_steps), output,
                                       {k2: v for k2, v in env.items() if k2 != "HEAL_HELDOUT2"},
                                       round_timeout - (time.time() - started) - 120,
                                       teacher_dirs=[x for x in teacher_arg.split(",") if x])
            report["phase"] = "train"
            save()
            if not report["gate"]["ok"]:
                identity["status"] = "gate_failed"
                return identity
        if diag_window:  # trainer diagnostics: run 1 at `window` saves step 1's gradient shards (and
            # profiles HEAL_PROFILE steps); run 2 at diag_window takes step 1 on the same data and reports their cosine
            shards = output / "gradshard"
            runs_diag = [(kw, output / "train", {"HEAL_GRAD_SHARD_DUMP": str(shards)}),
                         (dict(kw, window=diag_window), output / f"diag-w{diag_window}",
                          {"HEAL_GRAD_COSINE": str(shards), **env_pairs(diag_env)})]  # diag_env: run 2 only
            report["diag_exit"] = []
            for kw2, out_dir, extra in runs_diag:
                with open(out_dir.with_suffix(".log"), "w") as log:
                    try:
                        code = subprocess.run(torchrun(kw2, str(out_dir)), env={**env, **extra}, stdout=log,
                                              stderr=subprocess.STDOUT, timeout=max(600, round_timeout -
                                                                                   (time.time() - started) - 120)).returncode
                    except subprocess.TimeoutExpired:
                        code = "timeout"
                report["diag_exit"].append(code)
                save()
            import shutil
            shutil.rmtree(shards, ignore_errors=True)
            identity["status"] = "complete" if report["diag_exit"] == [0, 0] else "diag_failed"
            return identity
        if stream_play:
            return stream_round(identity, report, save, cfg, source, output, env, kw, torchrun, retry=retry,
                                round_timeout=round_timeout - (time.time() - started) - 120, run_id=run_id,
                                play=stream_play, base=stream_base, base_fill=stream_base_fill, poll=stream_poll, chunk=stream_chunk, k=k, max_age=stream_max_age,
                                verify_after=stream_verify_after, prune=stream_prune, cards=cards, tranks=tranks,
                                teacher_env=teacher_env,
                                bars={"step_seconds": stream_bar_step, "teacher_tok_per_s": stream_bar_teacher,
                                      "teacher_after": stream_bar_after})
        command = torchrun(kw, str(output / "train"))
        report["train_timeout"] = round(max(60, round_timeout - (time.time() - started) - 120))
        save()
        with open(output / "train.log", "w") as log:
            try:
                code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=report["train_timeout"]).returncode
            except subprocess.TimeoutExpired:
                events = output / "train" / "events.jsonl"
                done = events.exists() and '"round_done"' in events.read_text()
                code = 0 if done else "timeout"  # a hang in NCCL teardown after the save is a finished round
                report["teardown_timeout"] = done
        report["train_exit"] = code
        identity["status"] = "complete" if code == 0 else "train_failed"
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_smi.set()
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        save()
    return identity


def stream_round(identity, report, save, cfg, source, output, env, kw, torchrun, *, retry, round_timeout, run_id,
                 play, poll, chunk, k, max_age, verify_after, prune, bars, base="", base_fill="", cards=4, tranks=0,
                 teacher_env=""):
    """heal_round's streaming mode (its docstring). Returns the identity."""
    import asyncio
    import signal
    import subprocess
    import threading

    # ARC_HEAL_TRANKS: trainer ranks on the first cards, the teacher on the rest (3 + 1: FP8 experts on one card)
    tr, trainer_devices, teacher_devices, spread = stream_layout(cards, tranks)
    teacher_extra = stream_teacher_env(spread, teacher_env)
    report["layout"] = {"cards": int(cards), "trainer_ranks": tr, "trainer_devices": trainer_devices,
                        "teacher_devices": teacher_devices, "teacher_spread": spread, "teacher_env": teacher_extra}

    sys.path.insert(0, str(source / "rl1"))
    import heal_stream
    from arena import rl_loop

    nebius_rclone_env()
    began = time.time()
    roll, chunks, deltas = Path("/tmp/stream-roll"), Path("/tmp/stream-chunks"), Path("/tmp/deltas")
    for d in (roll, chunks, deltas):
        d.mkdir(parents=True, exist_ok=True)
    stop_file, done = chunks / ".stop", threading.Event()
    bases = [x for x in base.split(",") if x]  # base-model play runs: their pieces are "b-" (forward-KD rows)
    plays = [x for x in play.split(",") if x] + bases  # rotation groups: one play run each (the next may start early)
    fill = base_fill  # ARC_HEAL_STREAM_BASE_FILL: recorded base play, scored when the teacher has room
    if fill:
        sys.path.insert(0, str(source / "rl1"))
        import heal_stream
        bases.append(heal_stream.stream_fill(fill, roll))
    pulls = {"ok": 0, "failed": 0}
    # live play list (24 Oct): <run>/stream/knobs.json {"plays": [...], "confirm": [...]} swaps the pulled student
    # plays and the trainer's confirm targets without a restart (launch/pth/trainer_knobs.py writes it)
    confirm_live = Path("/tmp/stream-control/confirm.json")
    confirm_live.parent.mkdir(parents=True, exist_ok=True)
    first = [x for x in play.split(",") if x][:1]
    confirm_live.write_text(json.dumps([follow_path(x) for x in first]))
    trainer_knobs = confirm_live.parent / "trainer.json"
    trainer_knobs.write_text("{}")
    knobs_seen = {"raw": None}

    def read_knobs():
        try:
            r = subprocess.run(["rclone", "cat", f"arc:{NEBIUS_BUCKET}/runs/validation/{run_id}/stream/knobs.json"],
                               capture_output=True, text=True, timeout=60)
        except subprocess.SubprocessError:
            return
        if r.returncode != 0 or not r.stdout.strip() or r.stdout == knobs_seen["raw"]:
            return
        knobs_seen["raw"] = r.stdout
        try:
            got = stream_knobs(json.loads(r.stdout))
        except ValueError as exc:
            report.setdefault("stream_knobs", []).append({"epoch": time.time(), "refused": str(exc)[:300]})
            return
        if got["plays"] is not None:
            plays[:] = got["plays"] + [x for x in base.split(",") if x]
            tmp = confirm_live.with_suffix(".tmp")
            tmp.write_text(json.dumps([follow_path(x) for x in got["confirm"]]))
            tmp.replace(confirm_live)
        tmp = trainer_knobs.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: got[k] for k in ("lr", "ckpt") if got[k] is not None}))
        tmp.replace(trainer_knobs)
        report.setdefault("stream_knobs", []).append({"epoch": time.time(), **got})

    def puller():
        while not done.wait(poll):
            read_knobs()
            for one in list(plays):
                try:
                    r = subprocess.run(["rclone", "copy", f"arc:{NEBIUS_BUCKET}/runs/validation/{one}/stream/trials",
                                        str(roll / one), *STREAM_PULL_FLAGS], capture_output=True, text=True, timeout=600)
                    pulls["ok" if r.returncode == 0 else "failed"] += 1
                except subprocess.SubprocessError:
                    pulls["failed"] += 1

    threading.Thread(target=puller, daemon=True).start()

    async def teacher():
        work = Path("/tmp/heal-stream")
        command, tenv = rl_loop.trainer_command(cfg, source, work)
        # quack's compiled kernels and autotune picks persist on the volume (g1i 23 Oct: its compile pool workers died
        # on an import and the in-process re-tune held the first piece 699 s at every teacher start)
        # (quack locks its cache files with flock: the cache runs on local disk, seeded from and saved to the volume)
        logs = lambda a: output / ("teacher-stream.log" if not a else f"teacher-stream-r{a}.log")  # noqa: E731
        # MFU-2: versioned quack + hub cache, saved after piece 3 (ARC_HEAL_TEACHER_ENV=...,ARC_TEACHER_CACHE=1)
        keyed = teacher_extra.get("ARC_TEACHER_CACHE") == "1"
        if keyed:
            ckey, cver = await asyncio.to_thread(teacher_cache_key, command[0])
            cdirs, seeded = await asyncio.to_thread(teacher_cache_seed, ckey, cver)
            report["teacher_cache"] = {"key": ckey, "seeded": seeded, "versions": cver}
            cache_lock = threading.Lock()
            save_q = lambda: teacher_cache_save(ckey, cver, cdirs, lock=cache_lock)  # noqa: E731
            threading.Thread(target=teacher_cache_saver, args=(logs(0), save_q),
                             kwargs={"after": QUACK_SAVE_AFTER, "done": done}, daemon=True).start()
            local_q, hub_env = cdirs["quack"], teacher_cache_env(cdirs)
        else:
            local_q, hub_env = quack_cache_seed(), {}
            save_q = lambda: quack_cache_save(local_q)  # noqa: E731
            threading.Thread(target=lambda: [time.sleep(QUACK_SAVE_AFTER), save_q()], daemon=True).start()
        tenv = {**tenv, "CUDA_VISIBLE_DEVICES": teacher_devices, "RL1_PIN_QUACK": "1", "RL1_QUACK_REQUIRED": "1",
                "QUACK_CACHE_DIR": str(local_q),
                "QUACK_CACHE_AUTOTUNING": "1", **hub_env, **teacher_extra}

        def start(attempt):
            return rl_loop.Trainer(work / ("trainer" if not attempt else f"trainer-r{attempt}"),
                                   [*command, *(["--spread", str(spread)] if spread > 1 else [])], tenv, logs(attempt)), run

        async def run(trainer, attempt):
            loaded = await trainer.ready(cfg["trainer_timeout"])
            report["teacher_load_seconds"] = loaded.get("seconds")
            return await trainer.call(
                "stream_topk", timeout=round_timeout, rollouts=str(roll), out=str(chunks), k=k, chunk=chunk,
                stop_file=str(stop_file), verify_after=verify_after,
                skip_existing=bool(retry.get("resume")) or attempt > 0,
                base=",".join(bases), base_share=float(env.get("HEAL_STREAM_BASE_SHARE") or 0))

        try:  # a dead teacher restarts (skip_existing); the trainer's stall ceiling ends the round if it cannot
            await teacher_with_restarts(start, stop_file, report.setdefault("teacher_restarts", []), report,
                                        log_tail=lambda a: logs(a).read_text(errors="replace") if logs(a).exists() else "")
        finally:
            saved = save_q()
            if keyed:
                report["teacher_cache"]["saved"] = saved

    teaching = threading.Thread(target=lambda: asyncio.run(teacher()), daemon=True)
    teaching.start()
    tenv = stream_trainer_env(env, chunks=chunks, k=k, max_age=max_age, deltas=deltas, run_id=run_id, prune=prune,
                              devices=trainer_devices, play=play, confirm_live=confirm_live, live_knobs=trainer_knobs)
    command = torchrun(dict(kw, allow_unfinished=True), str(output / "train"))
    command[command.index("--teacher") + 1] = ""  # the stream is the round's data
    report.update(phase="stream", stream={"play": play, "chunk": chunk, "max_age": max_age, "bars": bars})
    save()
    with open(output / "train.log", "w") as log:
        proc = subprocess.Popen(command, env=tenv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        verdict = None
        while proc.poll() is None:
            time.sleep(30)
            if verdict is None:
                verdict, read = heal_stream.stream_bars(output / "train/events.jsonl", chunks / "status.json", bars)
                report["stream_bars"] = {"verdict": verdict, **read}
                if verdict and verdict.startswith("fail"):
                    os.killpg(proc.pid, signal.SIGTERM)  # first-phase bar missed: the run stops (the agent verifies)
            if (report.get("stream_teacher") or {}).get("verify_failed") and not report.get("teacher_verify_failed"):
                report["teacher_verify_failed"] = True  # a wrong teacher stops the round at once (opdS-c1, 23 Oct)
                os.killpg(proc.pid, signal.SIGTERM)
            if (chunks / "status.json").exists():
                report["stream_status"] = json.loads((chunks / "status.json").read_text())
            report["pulls"] = dict(pulls)
            save()
            if time.time() - began > round_timeout:  # crash ceiling, far above the planned length
                os.killpg(proc.pid, signal.SIGTERM)
                report["train_timeout"] = True
        code = proc.wait()
    stop_file.write_text("stop")
    teaching.join(600)
    done.set()
    report["train_exit"] = code
    verdict = (report.get("stream_bars") or {}).get("verdict")
    identity["status"] = ("bar_failed" if verdict and verdict.startswith("fail") else
                          "complete" if code == 0 else "train_failed")
    return identity


@app.function(image=image, gpu="B300", timeout=HEAL_TIMEOUT, retries=0,
              max_containers=3, scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def heal_score(run_id: str, students: str, teacher: str, general: str, files: int = 4):
    """Damage map: ``heal_kd.py score`` on the listed student folders against existing teacher files."""
    import subprocess

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": "B300", "students": students}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
    env.update(PYTHONPATH=str(source / "rl1"), PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    command = ["/workspace/axolotl-venv/bin/python", str(source / "rl1/heal_kd.py"), "score", "--students",
               *students.split(","), "--teacher", teacher, "--general", general, "--compare-files", str(files),
               "--out", str(output), "--window", "32768"]
    try:
        with open(output / "score.log", "w") as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=HEAL_TIMEOUT - 120).returncode
        identity["status"] = "complete" if code == 0 else f"exit {code}"
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


@app.function(image=image, gpu="B300", timeout=HEAL_TIMEOUT, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def heal_layers(run_id: str, student: str, windows: str, depth: int, lazy: int, cost_full: float, sets: str):
    """Depth-cut layer choice: ``heal_kd.py layers`` (single-layer, greedy KL and KL per cost) on one student."""
    import subprocess

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": "B300", "student": student}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
    env.update(PYTHONPATH=str(source / "rl1"), PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    command = ["/workspace/axolotl-venv/bin/python", str(source / "rl1/heal_kd.py"), "layers", "--student", student,
               "--windows", windows, "--depth", str(depth), "--lazy", str(lazy), "--cost-full", str(cost_full),
               "--out", str(output), "--sets", *[x for x in sets.split(";") if x]]
    try:
        with open(output / "layers.log", "w") as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=HEAL_TIMEOUT - 120).returncode
        identity["status"] = "complete" if code == 0 else f"exit {code}"
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


@app.local_entrypoint()
def main(run_id: str, dry_run: bool = False):
    if run_id.startswith("sft-stream-check-"):  # ARC_SC_FILES (volume paths), ARC_HEAL_TEACHER_MODEL, ARC_SC_NEAR
        e = os.environ
        json.loads(e.get("ARC_SC_EXTRA", "{}"))  # fail before the spawn
        call = stream_check.spawn(run_id, e["ARC_SC_FILES"], e["ARC_HEAL_TEACHER_MODEL"], e.get("ARC_SC_NEAR", "16384"),
                                  int(e.get("ARC_SC_LIMIT", "40000")), e.get("ARC_SC_PIN", "1,0"),
                                  ties=e.get("ARC_SC_TIES", "topk,index"), extra=e.get("ARC_SC_EXTRA", "{}"),
                                  **({"teacher_env": e["ARC_SC_TENV"]} if e.get("ARC_SC_TENV") else {}))
        print("STREAM CHECK CALL", call.object_id, flush=True)
        print(json.dumps(call.get(), indent=1, default=str)[-6000:])
        return
    if run_id.startswith("sft-teacher-check-"):  # ARC_TC_SEGMENTS, ARC_TC_NAMES=a,b; ARC_HEAL_TEACHER_MODEL (tokenizer)
        e = os.environ
        call = teacher_check.spawn(run_id, e["ARC_TC_SEGMENTS"], e["ARC_TC_NAMES"], int(e.get("ARC_TC_SPREAD", "2")),
                                   int(e.get("ARC_TC_MAX_TOKENS", "400000")), int(e.get("ARC_HEAL_TEACHER_CHUNK", "4096")),
                                   128, e["ARC_HEAL_TEACHER_MODEL"], e.get("ARC_TC_BLOCKING") == "1")
        print("TEACHER CHECK CALL", call.object_id, flush=True)
        res = call.get()
        print(json.dumps({k: v for k, v in res.items() if k != "log_tail"}, indent=1, default=str))
        print("TEACHER CHECK OK" if res.get("ok") else "TEACHER CHECK FAILED\n" + str(res.get("log_tail", ""))[-2500:])
        return
    if run_id.startswith("sft-layers-"):  # ARC_LAYERS_STUDENT, ARC_LAYERS_SETS="name=i,j;..."
        record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
        if record.exists():
            raise SystemExit(f"{record} exists: run ids are single-use")
        data = {"run_id": run_id, "kind": "heal-layers", "gpu": "B300", "timeout_seconds": HEAL_TIMEOUT,
                "charged_bound_usd": round(bound(HEAL_TIMEOUT), 2), "student": os.environ["ARC_LAYERS_STUDENT"],
                "windows": os.environ.get("ARC_LAYERS_WINDOWS", "/assets/sft-heal/layer-tf-excl5.json"),
                "depth": int(os.environ.get("ARC_LAYERS_DEPTH", "16")), "lazy": int(os.environ.get("ARC_LAYERS_LAZY", "6")),
                "cost_full": float(os.environ.get("ARC_LAYERS_COST_FULL", "2.0")), "sets": os.environ.get("ARC_LAYERS_SETS", ""),
                "bundle": str(BUNDLE), "state": "dry_run"}
        if dry_run:
            print(json.dumps(data, indent=1))
            return
        call = heal_layers.spawn(run_id, data["student"], data["windows"], data["depth"], data["lazy"], data["cost_full"],
                                 data["sets"])
        data.update(state="submitted", function_call_id=call.object_id)
        record.write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
        return
    if run_id.startswith("sft-score-"):  # ARC_SCORE_STUDENTS=a,b
        record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
        if record.exists():
            raise SystemExit(f"{record} exists: run ids are single-use")
        base = "/runs/validation/sft-heal-speed-v4"
        data = {"run_id": run_id, "kind": "heal-score", "gpu": "B300", "timeout_seconds": HEAL_TIMEOUT,
                "charged_bound_usd": round(bound(HEAL_TIMEOUT), 2), "students": os.environ["ARC_SCORE_STUDENTS"],
                "teacher": os.environ.get("ARC_SCORE_TEACHER", f"{base}/teacher"),
                "general": os.environ.get("ARC_SCORE_GENERAL", f"{base}/general"),
                "files": int(os.environ.get("ARC_SCORE_FILES", "4")), "bundle": str(BUNDLE), "state": "dry_run"}
        if dry_run:
            print(json.dumps(data, indent=1))
            return
        call = heal_score.spawn(run_id, data["students"], data["teacher"], data["general"], data["files"])
        data.update(state="submitted", function_call_id=call.object_id)
        record.write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
        return
    if run_id.startswith("sft-heal-round-"):  # ARC_HEAL_* settings; see heal_round
        record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
        if record.exists():
            raise SystemExit(f"{record} exists: run ids are single-use")
        e = os.environ
        kw = heal_round_kwargs(e)
        data = {"run_id": run_id, "kind": "heal-round", "gpu": f"{HEAL_GPU}:{HEAL_ROUND_CARDS}",
                "timeout_seconds": HEAL_ROUND_TIMEOUT,
                "charged_bound_usd": round(HEAL_ROUND_CARDS * bound(HEAL_ROUND_TIMEOUT, HEAL_GPU_RATE), 2), "bundle": str(BUNDLE),
                **kw, "train_subprocess_timeout_max": kw["round_timeout"] - 120,  # in the container, from kw
                "authorization": e.get("ARC_RL1_AUTHORIZATION", ""), "state": "dry_run"}
        if dry_run:
            print(json.dumps(data, indent=1))
            return
        if e.get("ARC_HEAL_PLAN") != "skip":  # mandatory CPU dry run over the real files
            verdict = heal_plan.remote(run_id, **kw)
            data["plan"] = verdict.get("plan") or verdict
            if not verdict["ok"]:
                print(verdict.get("tail") or json.dumps(verdict))
                raise SystemExit(f"REFUSED: {run_id} plan failed: {(verdict.get('plan') or {}).get('problems')}")
            print("PLAN OK", json.dumps({k: data["plan"].get(k) for k in ("files", "steps_fit", "total_steps", "reads")}))
            if e.get("ARC_HEAL_PLAN") == "only":  # the dry run alone: nothing launched, no record
                return
        call = heal_round.spawn(run_id, **kw)
        data.update(state="submitted", function_call_id=call.object_id)
        record.write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
        return
    if run_id.startswith("sft-heal-"):
        record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
        if record.exists():
            raise SystemExit(f"{record} exists: run ids are single-use")
        data = {"run_id": run_id, "kind": "heal-speed", "gpu": "B300", "timeout_seconds": HEAL_TIMEOUT,
                "charged_bound_usd": round(bound(HEAL_TIMEOUT), 2), "bundle": str(BUNDLE), "student": HEAL_STUDENT,
                "output": f"arc-prolong-runs:/validation/{run_id}", "state": "dry_run"}
        if dry_run:
            print(json.dumps(data, indent=1))
            return
        data["energy"] = os.environ.get("ARC_HEAL_ENERGY", "")
        data["only_energy"] = os.environ.get("ARC_HEAL_ONLY_ENERGY") == "1"
        call = heal_speed.spawn(run_id, energy=data["energy"], only_energy=data["only_energy"])
        data.update(state="submitted", function_call_id=call.object_id)
        record.write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
        return
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
        modes = (os.environ.get("ARC_RL1_ATTN_MODES", "selected,masked,selected") if run_id.startswith("rl1-attn-")
                 else os.environ.get("ARC_RL1_ATTN_MODES", ""))
        data["attention_modes"] = modes
        data["fit"] = os.environ.get("ARC_RL1_SPEED_FIT", "")
        call = trainer_speed.spawn(run_id, SPEED_ATTEMPTS, run_id.startswith("rl1-prof-"), modes, data["fit"])
        data.update(state="submitted", function_call_id=call.object_id)
        record.write_text(json.dumps(data, indent=2))
        print(json.dumps(data))
        return
    if run_id.startswith("rl1-ttest-"):
        return launch_test(run_id, dry_run)
    if not run_id.startswith(("rl1-probe-", "rl1-loop-", "rl1-restore-", "rl1-final-", "rl1-screen-", "sft-t", "sft-gate-", "sft-heldout-")):
        raise ValueError("run ids start with rl1-probe-, rl1-loop- or rl1-ttest-")
    study = bundle_study()
    mode = study["rl_loop"]["mode"]
    if not run_id.startswith(f"rl1-{mode}-") and not (mode in ("sft", "final") and run_id.startswith(("sft-t", "sft-gate-", "sft-heldout-"))):
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
