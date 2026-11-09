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
import math
import os
import re
import sys
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or ".").resolve()
BUNDLE = (Path(os.environ.get("ARC_RL1_BUNDLE", "dist/rl1-probe-v1")).resolve() if modal.is_local()
          else Path("/opt/arc-validation"))
GPU = os.environ.get("ARC_RL1_GPU") or "B300"  # the loop on B200 (cost test)
GPU_RATE = {"B300": 0.001972, "B200": 0.001736}[GPU]
# Modal jobs reserve no CPU or RAM; these only size the cost bound (usage billing)
BILL_CPU, BILL_MEMORY_GIB = 16.0, 435
# exception: the RL loop and trainer B300 functions keep 16 cores (no reservation: the trainer's
# cold step 792 s vs 511 s, encode waits 27 s vs 0.6 s); pilot 1 measured mean 2.7-3.4 cores, p95 5.1-7.8. The first
# launch at 8 cores (rl1-tsmoke-q2-v4, 20 Oct 23:03) stalled in model load at 60/1500 tensors for 38 min: back to 16
# until a load at 8 is shown to work.
RL_CPU = 16.0
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
# SonicMoE on a B300 (8 October): Triton 3.6.x, cutlass-dsl 4.6.0, quack kernels, in Axolotl's venv
RL_PIP = ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0")
STOP_FILE, STOP_LOCAL, STOP_POLL = "stop-request", "/tmp/arena-stop-request", 30
KNOBS_FILE, KNOBS_LOCAL = "knobs.json", "/tmp/arena-knobs.json"  # live knobs (rl_single.mg_knobs), read per round
AUTHORIZATION = ("LR probe 4 updates/rate (~4.6 h) and the 12.5 h loop approved; "
                 "loop with split mismatch rule, TIS cap 2, KV 56 GiB, LR 4e-5, pass 7.0; dose (all A > 0, "
                 "4 steps/round), trend read, per-step KL, split once with the noise bar; continue from v7's "
                 "round 3 for at most 6 h of loop wall (inside the 12.5 h): one signed step/round, 64 attempts, lp85 play guard; "
                 "'yes i pick (b). yes to F-small after.' (v9 bound; F-small bound)")


AUTHORIZATION = os.environ.get("ARC_RL1_AUTHORIZATION") or AUTHORIZATION  # another arm's run names its own


def carry_live(lost, output):
    """A preemption restart renames the dead attempt to <base>-lost-<epoch>; its arena/live requests (knobs.json, and
    stop-request: hero-v5 S4, 24 Oct, a stop put 80 s before the restart went aside with the dead attempt and the resumed
    one never saw it) are copied to the live attempt's folder."""
    import shutil

    src = None if lost is None else Path(lost) / "arena" / "live"
    if src is None or not src.is_dir():
        return []
    (Path(output) / "arena" / "live").mkdir(parents=True, exist_ok=True)
    moved = sorted(f.name for f in src.iterdir() if f.is_file())
    for name in moved:
        shutil.copyfile(src / name, Path(output) / "arena" / "live" / name)
    return moved


def bundle_study():
    return json.loads((BUNDLE / "configs/comparison.json").read_text())


STUDY = bundle_study() if (modal.is_local() and (BUNDLE / "configs/comparison.json").is_file()) else None
TIMEOUT = int(os.environ.get("ARC_RL1_TIMEOUT") or ((STUDY["notebook_seconds"] + 300) if STUDY else 46000))
# a two-card loop (server on GPU 0, trainer on GPU 1) runs on two B300s
CARDS = int(os.environ.get("ARC_RL1_CARDS") or (2 if STUDY and STUDY["rl_loop"].get("dual") else 1))
if CARDS == 2:
    GPU, GPU_RATE = f"{GPU}:2", 2 * GPU_RATE
app = modal.App("arc-rl1", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("rclone", "libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         # the loop's stall watchdog dumps every python process (ARC_RL1_STALL_DUMP); the venv's bin is on PATH
         .run_commands("uv pip install --python /workspace/axolotl-venv/bin/python 'py-spy==0.4.0'")
         .env({"ARC_RL1_TIMEOUT": str(TIMEOUT), "ARC_RL1_CARDS": str(CARDS), "ARC_RL1_GPU": GPU.split(":")[0],
               "PYTHONUNBUFFERED": "1"}))
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


def bound(seconds, rate=None):  # rate: PTH heal rounds on B200 / per-card shares
    return (seconds + 60) * ((GPU_RATE if rate is None else rate) + BILL_CPU * 1.31e-05 + BILL_MEMORY_GIB * 2.22e-06) * 1.15


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


def stall_env(stall):
    """``stall``: "SECONDS" or "SECONDS:kill" -> the loop's stall watchdog (rl_loop.StallWatch)."""
    if stall:
        seconds, _, kill = stall.partition(":")
        os.environ["ARC_RL1_STALL_DUMP"] = seconds
        if kill == "kill":
            os.environ["ARC_RL1_STALL_KILL"] = "1"


STALL = os.environ.get("ARC_RL1_STALL", "")  # launch-time setting, e.g. "300:kill" (preflights) or "1200" (runs)


ALERT_ONLY = ("k3 over", "TIS:")  # feed alerts that report without stopping (judge: not an error)


def alert_only(e):
    why = str(e.get("why") or "")
    return why.startswith(ALERT_ONLY) or "alert only" in why


READ_KEYS = ("ARC_RL1_EVAL_ADAPTERS", "ARC_RL1_RESUME_ADAPTER", "ARC_RL1_EXPECT_EVAL",
             "ARC_RL1_MAX_STEPS", "ARC_RESTART_ACK")  # launch-time read targets (rl_loop.settings);
# EXPECT_EVAL: the eval arms the launcher resolved, asserted by the run before it loads them (mg_eval); MAX_STEPS: v7's
# step ceiling for a launch (mg.v7.max_steps otherwise; a resume extends with a new one, no rebuild)


def apply_overrides(overrides):
    for key, value in json.loads(overrides or "{}").items():
        if key not in READ_KEYS:
            raise ValueError(f"override {key} not allowed")
        os.environ[key] = value if isinstance(value, str) else json.dumps(value)


@app.function(image=image, gpu=GPU, cpu=RL_CPU, timeout=TIMEOUT, retries=0, max_containers=1,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def run(run_id: str, resume: str = "", stall: str = "", max_rounds: int = 0, overrides: str = ""):
    apply_overrides(overrides)
    if max_rounds:  # the ceiling is a launch value; an extension is a resume with a new one, no rebuild
        os.environ["ARC_RL1_MAX_ROUNDS"] = str(int(max_rounds))
    stall_env(stall)
    return execute(run_id, f"validation/{run_id}", resume)


PREFLIGHT_MEMORY_GIB = 96  # bound only
# CPU preflight legs end themselves; Modal kills one at this crash ceiling (default 45 min)
PREFLIGHT_TIMEOUT = int(os.environ.get("ARC_RL1_PREFLIGHT_TIMEOUT") or 2700)
PREFLIGHT_CPU = (8, 16)  # CPU preflights reserve 8 cores, up to 16 (unreserved: ~1 core per rank)
FAKE_SMI = r"""#!/usr/bin/env python3
# preflight stand-in for nvidia-smi: B300 readings (the run's cards: ARC_RL1_CARDS)
import os, sys
args = sys.argv[1:]
vals = {"name": ("NVIDIA B300 SXM6 AC preflight-stub", ""), "memory.total": ("275040", " MiB"), "memory.used": ("250560", " MiB"),
        "memory.free": ("24480", " MiB"), "utilization.gpu": ("85", " %"), "power.limit": ("1100.00", " W"),
        "power.draw": ("700.00", " W"), "driver_version": ("580.82.07", ""), "compute_cap": ("10.3", ""),
        "temperature.gpu": ("50", ""), "pci.bus_id": ("00000000:00:00.0", "")}
query = next((a.split("=", 1)[1] for a in args if a.startswith("--query-gpu=")), None)
if query is None:
    print("NVIDIA-SMI preflight stub"); sys.exit(0)
fmt = next((a.split("=", 1)[1] for a in args if a.startswith("--format=")), "csv")
here = os.environ.get("ARC_FAKE_SMI_DIR") or os.path.dirname(os.path.abspath(__file__))
cards = list(range(int(os.environ.get("ARC_RL1_CARDS") or 1)))
for flag in ("-i", "--id"):
    if flag in args:
        cards = [int(args[args.index(flag) + 1])]
for i in cards:
    row, used = [], os.path.join(here, f"used-{i}")
    if os.path.isfile(used):  # the fake trainer's park/unpark (arena.fake_trainer.card_used)
        u = int(open(used).read().strip())
        vals = {**vals, "memory.used": (str(u), " MiB"), "memory.free": (str(275040 - u), " MiB")}
    for field in query.split(","):
        v, unit = (str(i), "") if field == "index" else vals.get(field, ("0", ""))
        row.append(v + ("" if "nounits" in fmt else unit))
    print(", ".join(row))
"""
PREFLIGHT_TAG = os.environ.get("ARC_RL1_PREFLIGHT_TAG", "")  # several preflights of one run id (e.g. smoke, full, resume)
RESUME_FROM = os.environ.get("ARC_RL1_RESUME_FROM", "")  # earlier run dirs (volume-relative) a relaunch continues
OVERRIDES = os.environ.get("ARC_RL1_OVERRIDES", "")  # JSON {READ_KEYS: value}: a read bundle's adapters per launch
MAX_ROUNDS = int(os.environ.get("ARC_RL1_MAX_ROUNDS") or 0)  # launch-time ceiling (rounds) over the bundle's


@app.function(image=image, timeout=PREFLIGHT_TIMEOUT, retries=0, max_containers=1, cpu=PREFLIGHT_CPU,
              scaledown_window=2, volumes={"/assets": assets, "/runs": runs})
def preflight(run_id: str, spec: str, resume: str = "", tag: str = "", max_rounds: int = 0, overrides: str = ""):
    apply_overrides(overrides)
    """Launch gate: the exact bundle and arguments on a CPU container, a replay stub in place of the
    vLLM server (``arena.replay_stub``); output ``validation/preflight/<run id>``."""
    os.environ["ARC_PREFLIGHT_STUB"] = spec
    os.environ["ARC_RL1_TRIAL"] = json.loads(spec).get("trial") or "rl-loop"
    if json.loads(spec).get("patch_check"):
        os.environ["ARC_PATCH_CHECK_ONLY"] = "1"
    if max_rounds:
        os.environ["ARC_RL1_MAX_ROUNDS"] = str(int(max_rounds))
    os.environ["ARC_PREFLIGHT_FAKE_TRAINER"] = "1"  # loop bundles: the stand-in trainer (arena.fake_trainer)
    os.environ["ARC_FAKE_TRAINER_PYTHON"] = sys.executable  # the pre-spawn and the loop start the same command
    stall_env(json.loads(spec).get("stall") or "")
    return execute(run_id, f"validation/preflight/{run_id}{tag}", resume)


COMPILE_ROOTS = ("tmp/qwen38-flash-next-vllm-cache", "tmp/qwen38-flash-next-vllm-compile-cache")


def compile_key(source):
    """v5 (throughput's Kaggle restore): the archive key covers everything vLLM's own cache key
    misses: the serving config, every runtime patch file of the bundle and the pinned runtime's file list."""
    import hashlib

    h = hashlib.sha256(json.dumps(bundle_study().get("server"), sort_keys=True).encode())
    for f in sorted((source / "src" / "arc_infra").glob("*.py")) + [source / "src" / "arena" / "serving.py"]:
        h.update(f.name.encode() + f.read_bytes())
    for f in sorted(Path("/assets/flash-input").rglob("*")) if Path("/assets/flash-input").exists() else []:
        if "runtime" in str(f) and f.is_file():
            h.update(f"{f}:{f.stat().st_size}".encode())
    if os.environ.get("ARC_PREFLIGHT_STUB"):
        h.update(b"stub")
    return h.hexdigest()[:16]


def compile_restore(key, identity, keep=False, archives="/runs/compile-cache", root="/"):
    import tarfile

    archive = Path(archives) / f"{key}.tar.gz"
    if not archive.is_file():
        identity["compile_cache"] = {"key": key, "restored": False}
        return
    began = time.time()
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(COMPILE_ROOTS) and (m.isfile() or m.isdir())]
        tar.extractall(root, members=members, filter="data")
    identity["compile_cache"] = {"key": key, "restored": True, "files": len(members),
                                 "seconds": round(time.time() - began, 1)}
    if keep:  # MFU-2 (22 Oct): the first server start keeps the restored graph (arena.serving.take_compile_keep)
        os.environ["ARC_COMPILE_KEEP"] = key
        identity["compile_cache"]["keep"] = True


def compile_save(key, stop):
    """After the server is up (its preflight file) and 10 min of play (the cudagraphs and the lazily compiled
    kernels exist), archive both roots once per key."""
    import tarfile

    ready = Path("/tmp/arena/serving/vllm-server-preflight.json")
    while not ready.exists():
        if stop.wait(30):
            return
    if stop.wait(600):
        return
    archive = Path("/runs/compile-cache") / f"{key}.tar.gz"
    if archive.exists():
        return
    try:
        archive.parent.mkdir(parents=True, exist_ok=True)
        tmp = archive.with_name(f"{key}.{os.getpid()}.tmp")
        with tarfile.open(tmp, "w:gz") as tar:
            for root in COMPILE_ROOTS:
                base = Path("/") / root
                for f in base.rglob("*") if base.exists() else []:
                    if f.is_file() and not f.is_symlink() and f.stat().st_size < 2**30:
                        tar.add(f, arcname=str(f.relative_to("/")))
        os.replace(tmp, archive)
        runs.commit()
        print(json.dumps({"event": "compile_cache_saved", "key": key, "bytes": archive.stat().st_size}), flush=True)
    except Exception as exc:  # the cache never ends the run
        print(f"compile cache save failed: {exc}", flush=True)


MODEL_ROOT = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"


def boot_settings(source):
    """MFU-2 boot switches (bundle configs/boot.json; absent = all off)."""
    path = Path(source) / "configs" / "boot.json"
    return json.loads(path.read_text() or "{}") if path.is_file() else {}


def prefetch_pending(roots, identity, block=16 << 20):
    """Start-up (pfP2 profile, 22 Oct): the loop's pending-pack extraction read its round tar from the volume (~16% of
    the ~3-min setup). The resumed root's packs from its state round on are read into the page cache at container
    start, beside the server boot. Reads only."""
    began, n = time.time(), 0
    try:
        root = next((Path(r) for r in reversed(roots) if (Path(r) / "mg_state.json").is_file()), None)
        if root is None:
            return
        k = int(json.loads((root / "mg_state.json").read_text())["round"])
        for tar in sorted((root / "rounds").glob("r*.tar")):
            if int(tar.stem[1:]) >= k:
                with open(tar, "rb", buffering=0) as f:
                    while chunk := f.read(block):
                        n += len(chunk)
    except Exception as exc:  # a prefetch never fails the run
        identity["pending_prefetch_error"] = f"{type(exc).__name__}: {exc}"
    identity["pending_prefetch"] = {"gib": round(n / 2**30, 2), "seconds": round(time.time() - began, 1)}


def prefetch_weights(row, threads, root=MODEL_ROOT, limit=None, block=16 << 20):
    """MFU-2 (configs/boot.json prefetch_threads): read the model's safetensors into the page cache from container
    start, beside the bootstrap's runtime extraction, so the vLLM loads (both servers) and the trainer's read RAM
    instead of the volume (probe 22 Oct: 9P 0.8 GiB/s one stream, 1.3 GiB/s at 8+, page-cache re-read 47 GiB/s).
    Reads only; ``row`` gets the totals; ``limit`` caps the files (CPU preflight)."""
    from concurrent.futures import ThreadPoolExecutor

    began = time.time()
    files = sorted(Path(root).glob("*.safetensors"))[:limit]

    def read(path):
        n = 0
        with open(path, "rb", buffering=0) as f:
            while chunk := f.read(block):
                n += len(chunk)
        return n

    try:
        with ThreadPoolExecutor(int(threads)) as pool:
            total = sum(pool.map(read, files))
        row.update(files=len(files), gib=round(total / 2**30, 2), seconds=round(time.time() - began, 1))
    except Exception as exc:  # a speed-up only: the loads read the volume as before
        row["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"event": "weights_prefetched", **row}), flush=True)
    return row


def _natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def shadow_input(source="/assets/flash-input", target="/kaggle/input", model=MODEL_ROOT):
    """MFU-2 (configs/boot.json local_weights): ``target`` as a real tree of symlinks into ``source`` (same paths, same
    files) whose model directory is real, so ``stage_local_weights`` can point its shards at local copies one at a
    time. Returns the shadow model directory."""
    source, target = Path(source), Path(target)
    rel = Path(model).relative_to(source)
    here_src, here_dst = source, target
    here_dst.mkdir(parents=True, exist_ok=True)
    for part in list(rel.parts) + [None]:
        for entry in here_src.iterdir():
            if part is not None and entry.name == part:
                continue
            link = here_dst / entry.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(entry, target_is_directory=entry.is_dir())
        if part is None:
            return here_dst
        here_src, here_dst = here_src / part, here_dst / part
        here_dst.mkdir(exist_ok=True)


def stage_local_weights(row, threads, shadow, root=MODEL_ROOT, local="/tmp/arc-local-weights", limit=None):
    """Copy the model's safetensors to local storage in vLLM's load order (natural sort), each to a temporary name, a
    size check, then an atomic switch of its shadow symlink: a reader opens either the volume file or a complete
    byte copy (the 9P mmap read is the first boot's weight-load bottleneck: 125 s vs 56-65 s on a re-boot, mtp4).
    ``limit`` caps the files (CPU preflight)."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    began = time.time()
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(root).glob("*.safetensors"), key=lambda p: _natural(p.name))[:limit]

    def copy(src):
        tmp, dst = local / f".{src.name}.part", local / src.name
        shutil.copyfile(src, tmp)
        if tmp.stat().st_size != src.stat().st_size:
            raise OSError(f"short copy of {src.name}")
        os.replace(tmp, dst)
        link = Path(shadow) / src.name
        new = link.with_name(f".{src.name}.link")
        new.unlink(missing_ok=True)
        new.symlink_to(dst)
        os.replace(new, link)
        return dst.stat().st_size

    try:
        with ThreadPoolExecutor(int(threads)) as pool:
            total = sum(pool.map(copy, files))
        row.update(files=len(files), gib=round(total / 2**30, 2), seconds=round(time.time() - began, 1))
    except Exception as exc:  # a speed-up only: unswitched shards stay on the volume
        row["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"event": "weights_staged_local", **row}), flush=True)
    return row


def runtime_cache_save(stop, cache_dir, input_root="/kaggle/input"):
    """MFU-2 (configs/boot.json runtime_cache): once the server is up, build this runtime's cache entry if the volume
    has none (arena.runtime_cache.build: a fresh fast-mode extraction, parts, a restored-tree digest check)."""
    ready = Path("/tmp/arena/serving/vllm-server-preflight.json")
    while not ready.exists():
        if stop.wait(30):
            return
    if stop.wait(60):
        return
    try:
        sys.path.insert(0, "/opt/arc-validation/src")
        from arena import runtime_cache

        recipe, runtime_dir = runtime_cache.pinned_recipe(input_root)
        record = runtime_cache.build(recipe, runtime_dir, cache_dir)
        runs.commit()
        print(json.dumps({"event": "runtime_cache_build", **{k: v for k, v in record.items()
                                                              if k not in ("built_tree", "restored_tree")}}), flush=True)
    except Exception as exc:  # the cache never ends the run
        print(f"runtime cache build failed: {type(exc).__name__}: {exc}", flush=True)


def dead_step(folder):
    """The step (v7) or round (mg) of a dead attempt's newest loop state: RestartGuard's checkpoint past step 0."""
    for name, key in (("v7_state.json", "step"), ("mg_state.json", "round")):
        try:
            return int(json.loads((Path(folder) / name).read_text()).get(key) or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return 0


def restart_guard(rel, root="/runs"):
    """arc_infra.runops.RestartGuard from the bundle's own source, loaded alone (standard library
    only; the Modal function's python has no bundle packages). None for a bundle without it."""
    import importlib.util

    pkg = BUNDLE / "src" / "arc_infra" / "runops"
    if not (pkg / "restart.py").is_file():
        return None
    if "arc_runops" not in sys.modules:
        spec = importlib.util.spec_from_file_location("arc_runops", pkg / "__init__.py",
                                                      submodule_search_locations=[str(pkg)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["arc_runops"] = module
        spec.loader.exec_module(module)
    from arc_runops.restart import RestartGuard

    return RestartGuard(root, Path(rel).name, policy="resume", commit=runs.commit)


def execute(run_id, rel, resume=""):
    """``resume``: earlier run dirs on the volume (comma-separated, relative to the volume root) whose finished tries
    and sets the run keeps (owner gate item 4)."""
    import shutil
    import subprocess
    import threading

    source = Path("/opt/arc-validation")
    output = Path("/runs") / rel
    # every restart is an incident: auto-resume only on a preemption signal with a checkpoint past step 0, a relaunch
    # after a death only with ARC_RESTART_ACK (raises RestartBlocked; retries=0 then fails the input)
    guard = restart_guard(rel)
    guard_start = guard.start(dead_step(output) if output.exists() else 0) if guard is not None else None
    if guard is not None:
        guard.watch()
    lost = None
    if output.exists():
        lost = output.with_name(f"{output.name}-lost-{int(time.time())}")
        output.rename(lost)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {"run_id": run_id, "started_epoch": started, "gpu": GPU, "function_timeout": TIMEOUT,
                "image": AXOLOTL_IMAGE, "host_cpu": host_cpu(),
                "restart_guard": None if guard is None else {"attempt": guard_start.attempt, "restart": guard_start.restart,
                                                             "incident": (guard_start.incident or {}).get("id")}}
    boot = boot_settings(source)
    runtime_dir = None
    if boot.get("runtime_cache"):  # MFU-2: the extracted vLLM runtime from the volume (arena.runtime_cache)
        runtime_dir = "/runs/runtime-cache" + ("-preflight" if os.environ.get("ARC_PREFLIGHT_STUB") else "")
        os.environ["ARC_RUNTIME_CACHE"] = runtime_dir
    if boot.get("expert_mask_skip_read"):  # MFU-2: server.expert_mask's masked experts' packed weights never read
        os.environ["ARC_EXPERT_MASK_SKIP_READ"] = "1"  # (arc_infra.expert_mask_patch; scales still read; exact)
        identity["expert_mask_skip_read"] = True
    if boot.get("expert_mask_compact"):  # MFU-2: masked experts not allocated (expert_mask_patch COMPACT_TARGETS;
        os.environ["ARC_EXPERT_MASK_COMPACT"] = "1"  # +KV on card 1; exactness vs the masked path is GPU-checked)
        identity["expert_mask_compact"] = True
    if boot.get("load_profile"):  # MFU-2: weight-load time split per load (expert_mask_runtime.install_load_profile;
        os.environ["ARC_LOAD_PROFILE"] = "1"  # needs server.expert_mask; exact, rows in the feed's expert_mask event)
        identity["load_profile"] = True
    if boot.get("local_weights"):  # MFU-2: the shards copied to local storage, switched in one by one
        Path("/kaggle").mkdir(exist_ok=True)
        identity["local_weights"] = {"threads": int(boot["local_weights"])}
        threading.Thread(target=stage_local_weights, daemon=True, args=(
            identity["local_weights"], boot["local_weights"], shadow_input()), kwargs={
            "limit": 2 if os.environ.get("ARC_PREFLIGHT_STUB") else None}).start()
    elif boot.get("prefetch_threads"):  # MFU-2: page-cache the weights while the bootstrap installs and extracts
        identity["prefetch"] = {"threads": int(boot["prefetch_threads"])}
        threading.Thread(target=prefetch_weights, daemon=True, args=(
            identity["prefetch"], boot["prefetch_threads"]), kwargs={
            "limit": 2 if os.environ.get("ARC_PREFLIGHT_STUB") else None}).start()
    # Modal preemption (v6 21 Oct 20:37: "Container terminated due to preemption", restarted with the same input): an
    # earlier attempt that died (no modal-identity.json) with a round state (v7: step state) is resumed, not discarded; its live knobs
    # carry over (applied ones stay applied: knobs_seen is in mg_state)
    # (its kept tries count too: a preemption in the first round after a resume leaves partial/ but no mg_state; the
    # loop resumes the latest root with an mg_state and reuses same-policy tries from every root)
    if lost is not None and not (lost / "modal-identity.json").is_file() and (
            (lost / "mg_state.json").is_file() or (lost / "v7_state.json").is_file() or (lost / "partial").is_dir()):
        resume = ",".join([x for x in (resume or "").split(",") if x] + [str(lost.relative_to("/runs"))])
        identity["preempt_resume"] = str(lost)
    carry_live(lost, output)
    Path("/kaggle").mkdir(exist_ok=True)
    if not boot.get("local_weights"):  # (local_weights built /kaggle/input as a shadow tree above)
        Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ.update(PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1", MPLBACKEND="Agg", ONLY_RESET_LEVELS="true",
                      PYTHONDONTWRITEBYTECODE="1", ARC_FORCE_CHROOT="1", ARENA_STOP_REQUEST=STOP_LOCAL, ARENA_KNOBS=KNOBS_LOCAL,
                      ARC_VALIDATION_OUTPUT=str(output))
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
    probe = ((bundle_study()["rl_loop"].get("mg") or {}).get("lr_probe"))
    if probe and not os.environ.get("ARC_PREFLIGHT_STUB"):  # trainer only (P1): the stub holds the server's place, no card
        os.environ["ARC_PREFLIGHT_STUB"] = json.dumps({"corpus": [], "serving": probe["serving"], "speed": 0})
    if not (os.environ.get("ARC_PREFLIGHT_STUB") or os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER")):
        try:  # v5: a park check passed on the GPU for this trainer key is not rerun (stand-in trainers never record)
            os.environ["ARC_RL1_TRAINER_KEY"] = identity["trainer_key"] = trainer_key(source)
        except Exception as exc:  # no key: the check runs
            identity["trainer_key_error"] = f"{type(exc).__name__}: {exc}"
    if resume:
        os.environ["ARC_RL1_RESUME"] = ",".join(unpacked(f"/runs/{d.strip('/')}") for d in resume.split(","))
        threading.Thread(target=prefetch_pending, args=(os.environ["ARC_RL1_RESUME"].split(","), identity),
                         daemon=True).start()
    if os.environ.get("ARC_PREFLIGHT_STUB") and not shutil.which("nvidia-smi"):  # a CPU preflight: a stand-in B300
        fake = Path("/tmp/fake-smi")
        fake.mkdir(exist_ok=True)
        (fake / "nvidia-smi").write_text(FAKE_SMI)
        (fake / "nvidia-smi").chmod(0o755)
        os.environ["PATH"] = f"{fake}:{os.environ['PATH']}"
        os.environ["ARC_FAKE_SMI_DIR"] = str(fake)  # the fake trainer's park writes the card's memory.used here
    if CARDS == 2 and shutil.which("nvidia-smi"):  # the serving stack checks for exactly one card: it sees GPU 0 (the trainer asks for its own card)
        real = shutil.which("nvidia-smi")
        shim = Path("/tmp/gpu0-bin")
        shim.mkdir(exist_ok=True)
        (shim / "nvidia-smi").write_text('#!/bin/sh\ncase " $* " in *" -i "*|*" --id"*) exec %s "$@";; esac\n'
                                         'exec %s -i 0 "$@"\n' % (real, real))
        (shim / "nvidia-smi").chmod(0o755)
        os.environ.update(PATH=f"{shim}:{os.environ['PATH']}", ARC_REAL_NVIDIA_SMI=real)
        identity["cards"] = subprocess.run([real, "--query-gpu=index,name", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip().splitlines()
    if boot.get("trainer_prespawn"):  # start-up (b), 22 Oct: the loop's trainer loads beside server 1's boot
        try:  # (arena.trainer_prespawn; the loop adopts it only on an exact command/environment match)
            r = subprocess.run([sys.executable, "-m", "arena.trainer_prespawn", str(source / "configs" / "comparison.json"),
                                str(source), "/tmp/rl1-prespawn.log"], capture_output=True, text=True, timeout=300)
            identity["trainer_prespawn"] = ((r.stdout.strip().splitlines() or [""])[-1] or r.stderr[-400:])
        except Exception as exc:  # no pre-spawn: the loop spawns its own
            identity["trainer_prespawn"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "trainer_prespawn", "result": identity["trainer_prespawn"]}), flush=True)
    command = [sys.executable, str(source / "bootstrap_arena.py"), "--bundle", str(source), "--working", "/tmp/arena",
               "--output", str(output / "arena"), "--games-dir", "/assets/games", "--max-seconds", str(TIMEOUT - 60),
               "--trial", os.environ.get("ARC_RL1_TRIAL") or "rl-loop"]  # preflight spec "trial": e.g. a level-bench read

    def pump(pipe, log):
        for line in iter(pipe.readline, b""):
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()

    def committer(stop):
        # the trial writes under /tmp/arena until the bootstrap collects it at the end; the live feed is copied
        # each pass (rl1-probe-v7's watcher saw no feed for the whole run)
        trial = os.environ.get("ARC_RL1_TRIAL") or "rl-loop"
        live, mirror = Path(f"/tmp/arena/trials/{trial}/rl"), output / f"arena/trials/{trial}/rl"
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
        folder = f"{rel}/arena/live"
        while not stop.wait(STOP_POLL):
            try:
                names = {Path(entry.path).name for entry in runs.listdir(folder)}
            except Exception:  # no live folder yet
                continue
            if KNOBS_FILE in names:
                try:
                    data = b"".join(runs.read_file(f"{folder}/{KNOBS_FILE}"))
                    if not Path(KNOBS_LOCAL).exists() or Path(KNOBS_LOCAL).read_bytes() != data:
                        Path(KNOBS_LOCAL + ".tmp").write_bytes(data)
                        os.replace(KNOBS_LOCAL + ".tmp", KNOBS_LOCAL)
                        print(json.dumps({"event": "knobs_seen", "bytes": len(data)}), flush=True)
                except Exception as exc:
                    print(f"knobs read failed: {exc}", flush=True)
            if STOP_FILE in names:
                Path(STOP_LOCAL).write_text(json.dumps({"source": f"{folder}/{STOP_FILE}", "epoch": time.time()}))
                print(json.dumps({"event": "stop_request_seen"}), flush=True)
                return

    def sampler(stop):
        # 1-s card and CPU samples on the volume from minute one (the efficiency watcher reads util/; screen
        # mg6-5 ran blind when the external sampler was not started)
        util = output / "util"
        util.mkdir(parents=True, exist_ok=True)
        smi = os.environ.get("ARC_REAL_NVIDIA_SMI") or shutil.which("nvidia-smi") or "nvidia-smi"
        with (util / "cpulog.txt").open("a") as cpu, (util / "gpulog.txt").open("a") as gpu:
            while not stop.wait(1.0):
                try:
                    now = int(time.time())
                    stat = Path("/proc/stat").read_text().splitlines()[0]
                    try:
                        thr = " ".join(l for l in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines()
                                       if l.startswith(("usage_usec", "nr_throttled", "throttled_usec")))
                    except OSError:
                        thr = ""
                    mem = ""  # container memory (cgroup v2, else v1), GiB: the time share's host-memory headroom
                    for name in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
                        try:
                            mem = f" mem_gib {int(Path(name).read_text()) / 2**30:.1f}"
                            break
                        except (OSError, ValueError):
                            continue
                    try:  # cgroup v2: anon (not page cache; what a host snapshot adds to) and the limit
                        rows = dict(l.split() for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
                                    if l.count(" ") == 1)
                        top = Path("/sys/fs/cgroup/memory.max").read_text().strip()
                        mem += f" anon_gib {int(rows.get('anon', 0)) / 2**30:.1f}" + (
                            f" max_gib {int(top) / 2**30:.1f}" if top.isdigit() else f" max {top}")
                    except (OSError, ValueError):
                        try:  # cgroup v1 (ovf3 logged usage only, page cache included): rss and the limit
                            rows = dict(l.split() for l in Path("/sys/fs/cgroup/memory/memory.stat").read_text().splitlines()
                                        if l.count(" ") == 1)
                            top = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
                            mem += f" anon_gib {int(rows.get('total_rss', rows.get('rss', 0))) / 2**30:.1f}" + (
                                f" cache_gib {int(rows.get('total_cache', 0)) / 2**30:.1f} max_gib {top / 2**30:.1f}")
                        except (OSError, ValueError):
                            try:  # neither (gVisor): the sandbox's own /proc/meminfo
                                info = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                                        if len(l.split()) >= 2 and l.split()[1].isdigit()}
                                mem += (f" anon_gib {info.get('AnonPages', 0) / 2**20:.1f}"
                                        f" cache_gib {info.get('Cached', 0) / 2**20:.1f}")
                            except (OSError, ValueError, IndexError):
                                pass
                    cpu.write(f"{now} {stat} {thr}{mem}\n")
                    cards = subprocess.run([smi, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
                    gpu.write(" ".join(cards.split("\n")).strip() + f" {now}\n")
                    cpu.flush()
                    gpu.flush()
                except Exception:  # sampling never ends the run
                    continue

    key = None
    if (source / "configs" / "compile-cache.json").is_file():  # v5 bundles opt in
        try:
            key = compile_key(source)
            spec = json.loads((source / "configs" / "compile-cache.json").read_text() or "{}")
            compile_restore(key, identity, keep=bool(spec.get("keep_restored")))
        except Exception as exc:  # a broken archive: a cold start
            identity["compile_cache"] = {"key": key, "error": f"{type(exc).__name__}: {exc}"}
    process = None
    try:
        with (output / "bootstrap.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            stop = threading.Event()
            threads = [threading.Thread(target=pump, args=(process.stdout, log), daemon=True),
                       threading.Thread(target=committer, args=(stop,), daemon=True),
                       threading.Thread(target=stop_watcher, args=(stop,), daemon=True),
                       threading.Thread(target=sampler, args=(stop,), daemon=True)]
            if key:
                threads.append(threading.Thread(target=compile_save, args=(key, stop), daemon=True))
            if runtime_dir:
                threads.append(threading.Thread(target=runtime_cache_save, args=(stop, runtime_dir), daemon=True))
            die = (json.loads(os.environ.get("ARC_PREFLIGHT_STUB") or "{}") or {}).get("die_after_s")
            # once per leg: Modal retries the input in a new container, which then resumes the dead attempt (the real
            # preemption path) instead of dying again
            if die and rel.startswith("validation/preflight/") and lost is None:  # a preemption stand-in (RL 22 Oct)
                def killer(proc=process):  # resumes a container that died mid-round (kept tries, no final row)
                    if not stop.wait(float(die)):
                        print(json.dumps({"event": "preflight_die", "after_s": die}), flush=True)
                        if guard is not None:  # the signal evidence a real preemption leaves (auto-resume allowed)
                            guard.record_preemption()
                        runs.commit()
                        os.killpg(proc.pid, 9)
                        os._exit(137)
                threads.append(threading.Thread(target=killer, daemon=True))
            for thread in threads:
                thread.start()
            try:
                process.wait()
            finally:
                stop.set()
                threads[0].join(timeout=30)
        identity["exit_code"] = process.returncode
        if guard is not None:  # a clean end (a kill or a preemption signal never gets here)
            guard.end("ok" if process.returncode == 0 else f"exit {process.returncode}")
    except Exception as exc:  # the identity records every outcome
        identity["error"] = f"{type(exc).__name__}: {exc}"
        if guard is not None:
            guard.end(f"error: {type(exc).__name__}: {exc}"[:300])
    finally:
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        if rel.startswith("validation/preflight/"):  # the verdict's checks here, on the local tree
            try:
                local = lambda r: (Path("/runs") / r).read_text(errors="replace") if (Path("/runs") / r).is_file() else None
                study = json.loads((source / "configs" / "comparison.json").read_text())
                judged = judge(rel, local, lambda r: [q.name for q in (Path("/runs") / r).iterdir()], resume, study, identity)
                (output / JUDGED).write_text(json.dumps(judged))
            except Exception as exc:  # the local verdict still judges from the volume
                identity["judge_error"] = f"{type(exc).__name__}: {exc}"
        if rel.startswith("validation/preflight/"):  # inodes: a preflight leaves ~830 files;
            try:  # the tree goes into one tarball beside the identity (verdicts and resumes read it)
                identity["packed"] = pack(output)
            except Exception as exc:  # an unpacked tree is still readable
                identity["pack_error"] = f"{type(exc).__name__}: {exc}"
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return identity


PACK = "pack.tar.gz"


def pack(output):
    """Everything under ``output`` but the identity into ``output/pack.tar.gz``; the packed entries are removed."""
    import shutil
    import tarfile

    entries = [p for p in output.iterdir()
               if p.name not in ("modal-identity.json", PACK, PACK + ".sha256", PACK + ".part", JUDGED)]
    files = sum(len(f) + len(d) for e in entries if e.is_dir() for _, d, f in os.walk(e)) + len(entries)
    part = output / (PACK + ".part")  # xv7d pfA: preempted mid-pack, the restart read a half-written pack.tar.gz
    with tarfile.open(part, "w:gz") as tar:
        for entry in entries:
            tar.add(entry, arcname=entry.name)
    (output / (PACK + ".sha256")).write_text(sha256_file(part) + "\n")
    os.replace(part, output / PACK)  # the pack appears whole or not at all; entries go only after it
    for entry in entries:
        shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
    return {"entries": len(entries), "inodes": files, "bytes": (output / PACK).stat().st_size}


def unpacked(path):
    """A packed preflight dir as a readable tree under /tmp (same relative path, so restores still see
    /validation/preflight/ in their source); any other dir as it is."""

    folder = Path(path)
    if not (folder / PACK).is_file():
        return path
    target = Path("/tmp") / folder.relative_to("/runs")
    if not target.exists():
        import shutil

        def fetch(n):
            if n:
                try:
                    runs.reload()
                except Exception:  # open files on the mount: the copy below still reads afresh
                    pass
            local = Path("/tmp") / f"{folder.name}.{os.getpid()}.{PACK}"
            shutil.copyfile(folder / PACK, local)
            return local

        side = folder / (PACK + ".sha256")
        extract_checked(fetch, target, side.read_text().split()[0] if side.is_file() else None)
    return str(target)


def sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def extract_checked(fetch, target, want=None, tries=3):
    """``fetch(n)`` -> a fresh local copy of a pack from the volume; checked against ``want`` (its sha256 sidecar,
    packs from 22 Oct on) and extracted into ``target``. A mismatch or a corrupt archive fetches again, up to
    ``tries`` reads (22 Oct: two gzip CRC failures on reads of intact packs, pf13 and pf21)."""
    import shutil
    import tarfile
    import zlib

    last = None
    for n in range(tries):
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        local = fetch(n)
        try:
            got = sha256_file(local) if want else None
            if want and got != want:
                raise ValueError(f"sha256 {got[:12]} != {want[:12]}")
            if not want:  # no sidecar: decompress to the end so the gzip CRC is checked (tarfile stops before it)
                import gzip

                with gzip.open(local, "rb") as f:
                    while f.read(1 << 24):
                        pass
            with tarfile.open(local) as tar:
                tar.extractall(target, filter="data")
            return n + 1
        except (ValueError, EOFError, OSError, tarfile.TarError, zlib.error) as exc:  # BadGzipFile is an OSError
            last = exc
            print(f"pack read {n + 1}/{tries} of {target} failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            Path(local).unlink(missing_ok=True)
    shutil.rmtree(target, ignore_errors=True)
    raise RuntimeError(f"{target}: pack unreadable after {tries} reads: {last}")


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


def heal_restart_guard(output, retry, call_id=None):
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
               "HEAL_VERIFY_KL")  # c7j v7: the stream verify's KL bar (k x the bf16 call-shape floor)
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
              volumes={"/assets": assets, "/runs": runs})
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
    "lr": x}, each optional. plays: the pulled student plays; confirm: the plays whose applied generation ends full
    versions (default every listed play); lr: the trainer's peak lr (heal_kd LiveLr, the schedule scaled by x / --lr).
    Returns {"plays", "confirm", "lr"} (None: unchanged); ValueError on anything else (the round keeps its knobs)."""
    import re
    if not isinstance(knobs, dict) or not knobs or set(knobs) - {"plays", "confirm", "lr"}:
        raise ValueError(f"knobs are {{'plays': [...], 'confirm': [...], 'lr': x}}; got {knobs!r}")
    out = {"plays": None, "confirm": None, "lr": None}
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
    halt = heal_restart_guard(output, retry)
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
        tmp.write_text(json.dumps({"lr": got["lr"]} if got["lr"] is not None else {}))
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




def server_check(study):
    """The bundle's server start path through every settings check up to the vLLM launch (server_check.py), so a
    bound the server rejects fails here, not after the image and model load (mg5f, mg5g-chk)."""
    import subprocess

    out = subprocess.run([sys.executable, str(Path(__file__).with_name("server_check.py")), str(BUNDLE)],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit("server check failed: " + (out.stderr.strip().splitlines() or ["?"])[-1])
    tuning = json.loads(out.stdout.strip().splitlines()[-1])
    return {k: tuning.get(k) for k in ("max_num_seqs", "kv_cache_memory_bytes", "max_cudagraph_capture_size")}


CPU_SIDE = ("served_route.py")  # rl1 files off the GPU path (the R3 donor reader needs tests + CPU preflight)


def trainer_key(bundle=None, play_cap=True, gpu_only=False, part=None):
    """The real-trainer gate's key: what the trainer runs, not the loop's driver or its config. The
    bundle's trainer files (rl1/*), the loop's trainer client (``class Trainer`` and ``trainer_command`` in rl_loop.py),
    the trainer's command-line settings and the mg update arguments. Learning rate, sets, caps of plays, rules and
    rounds are left out: a config-only relaunch keeps its smoke verdict."""
    import ast

    bundle = Path(bundle or BUNDLE)
    digest = hashlib.sha256()
    for path in sorted((bundle / "rl1").glob("*.py")) if part != "settings" else ():
        if gpu_only and path.name in CPU_SIDE:
            continue
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    if part == "code":
        return digest.hexdigest()
    source = (bundle / "src/arena/rl_loop.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("Trainer", "trainer_command"):
            digest.update(ast.get_source_segment(source, node).encode() + b"\0")
    study = json.loads((bundle / "configs/comparison.json").read_text())
    cfg = json.loads(subprocess_settings(bundle, study))
    mg = cfg.get("mg") or {}
    keyed = {k: cfg.get(k) for k in ("model_dir", "trainer_python", "rank", "window", "optimizer", "loop_lora", "tis_cap",
                                      "path_chunk", "trainer_env", "attention", "l2_norm")}
    keyed["mg"] = {k: mg.get(k) for k in (("cap",) if play_cap else ()) + ("batch_tokens", "batch_max", "policy_loss", "gspo_clip")}
    if mg.get("gspo_norm"):  # pilot 1 D; absent keeps earlier bundles' keys
        keyed["mg"]["gspo_norm"] = mg["gspo_norm"]
    if mg.get("trainer_fast"):  # MFU fixes on the GPU path (21 Oct); absent keeps earlier keys
        keyed["mg"]["trainer_fast"] = mg["trainer_fast"]
    digest.update(json.dumps(keyed, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def subprocess_settings(bundle, study):
    """The bundle's own ``rl_loop.settings`` of its study, as JSON (the bundle's code, not the launcher's)."""
    import subprocess

    out = subprocess.run([sys.executable, "-c", "import json, sys; from arena import rl_loop; "
                          "print(json.dumps(rl_loop.settings(json.load(sys.stdin)['rl_loop']), default=str))"],
                         input=json.dumps(study), env={**os.environ, "PYTHONPATH": str(Path(bundle) / "src")},
                         capture_output=True, text=True, check=True)
    return out.stdout


def smoke_check(bundle=None, final=False):
    """Launch gate addition: a training launch needs a passing rl1-tsmoke verdict for its trainer key.
    A final read (``final``) runs no update, only load_checkpoint and save: a passing smoke of a bundle whose trainer
    files, client and settings are equal apart from the play cap (mg.cap) counts."""
    key = trainer_key(bundle)
    for path in sorted((ROOT / "docs/validation-rl").glob("smoke-*.json")):
        row = json.loads(path.read_text())
        if row.get("trainer_key") == key and row.get("verdict") == "pass":
            return path.name
        if final and row.get("verdict") == "pass" and row.get("bundle") and Path(row["bundle"]).is_dir() \
                and trainer_key(row["bundle"], play_cap=False) == trainer_key(bundle, play_cap=False):
            return f"{path.name} (final read: same trainer code and settings apart from the play cap)"
    for path in sorted((ROOT / "docs/validation-rl").glob("preflight-rl1-ttest-*.json")):
        # a passing trainer test (P1 probe bundles: load, update, checkpoint, park, reload, update) of the same key
        row = json.loads(path.read_text())
        if row.get("verdict") == "pass" and row.get("bundle") and Path(row["bundle"]).is_dir() \
                and trainer_key(row["bundle"]) == key:
            return f"{path.name} (trainer test of the same trainer key)"
    gpu = trainer_key(bundle, gpu_only=True)
    for path in sorted((ROOT / "docs/validation-rl").glob("smoke-*.json")) + \
            sorted((ROOT / "docs/validation-rl").glob("preflight-rl1-ttest-*.json")):
        # faster gate: a smoke is needed only for GPU-path changes; a bundle that differs from a passed
        # smoke/test only in CPU-side files (CPU_SIDE) is covered by their unit tests and the CPU preflight
        row = json.loads(path.read_text())
        if row.get("verdict") == "pass" and row.get("bundle") and Path(row["bundle"]).is_dir() \
                and trainer_key(row["bundle"], gpu_only=True) == gpu:
            return f"{path.name} (same GPU path; differs only in CPU-side {', '.join(CPU_SIDE)})"
    code = settings = None  # trainer code proven on the card by one passed smoke, the client + settings by another
    for path in sorted((ROOT / "docs/validation-rl").glob("smoke-*.json")) + \
            sorted((ROOT / "docs/validation-rl").glob("preflight-rl1-ttest-*.json")):
        row = json.loads(path.read_text())
        if row.get("verdict") != "pass" or not row.get("bundle") or not Path(row["bundle"]).is_dir():
            continue
        if code is None and trainer_key(row["bundle"], gpu_only=True, part="code") == trainer_key(bundle, gpu_only=True, part="code"):
            code = path.name
        if settings is None and trainer_key(row["bundle"], part="settings") == trainer_key(bundle, part="settings"):
            settings = path.name
    if code and settings:
        return f"trainer code as {code}, client and settings as {settings} (GPU path covered in parts)"
    study = json.loads((Path(bundle or BUNDLE) / "configs/comparison.json").read_text())
    if ((study.get("rl_loop") or {}).get("mg") or {}).get("reload_check"):
        # faster gate: the smoke as the run's own first phase: load, U1, checkpoint, reload, rescore
        # (KL-to-base within 1e-3), the next update; a mismatch stops the run (mg.reload_check)
        return f"in-run first-phase smoke (mg.reload_check) for trainer key {key[:12]}"
    raise SystemExit(f"launch gate: no passing trainer smoke for trainer key {key[:12]} (run an rl1-tsmoke-* job on this "
                     "bundle, then --verdict)")


# Settings/state-only restarts need no preflight: same code files as a bundle whose preflight
# passed, and every changed config value is a plain value change at one of these paths (a key appearing, vanishing,
# turning None/bool or changing type switches a code path and counts as a code change). Restart state is not in the
# bundle (ARC_RL1_RESUME_FROM, mg_state via the rollback tool).
SETTINGS_PATHS = (r"/name", r"/status", r"/rl_loop/lr", r"/rl_loop/rounds", r"/rl_loop/resume_(from|round|state|adapter)",
                  r"/rl_loop/mg/(kl_target|kl_floor|kl_step|kl_damage|lr_max|max_rounds|updates|active_kv_fill|"
                  r"active_tries|gate_overcommit)", r"/rl_loop/mg/kl_ramp/(start|cap|factor|hold|z|overshoot)",
                  r"/rl_loop/mg/timeshare/min_queued", r"/rl_loop/mg/behaviour_guard/[a-z_]+", r"/rl_loop/mg/warn/[a-z_]+",
                  r"/rl_loop/(update|trainer)_timeout")  # crash ceilings (v6d-r2 U9 old pass past 3600 s, 22 Oct)


def settings_diff(old, new):
    """The changed config paths of ``new`` vs ``old`` and the ones that are not plain settings changes."""
    import re

    def flat(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from flat(v, f"{path}/{k}")
        else:
            yield path, x

    a, b = dict(flat(old)), dict(flat(new))
    changed = sorted(k for k in set(a) | set(b) if a.get(k, KeyError) != b.get(k, KeyError))

    def plain(k):
        if k not in a or k not in b or not any(re.fullmatch(p, k) for p in SETTINGS_PATHS):
            return False
        x, y = a[k], b[k]
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        return (num(x) and num(y)) or (isinstance(x, str) and isinstance(y, str))

    return changed, [k for k in changed if not plain(k)]


def settings_only_gate(bundle, receipts):
    """``bundle`` passes without a preflight when a passed preflight's bundle (on disk, manifest verified) has the same
    files except configs, and the configs differ only in settings (``settings_diff``)."""
    import hashlib

    mine = json.loads((bundle / "MANIFEST.json").read_text())["files"]
    configs = {k for k in mine if k.startswith("configs/") and k.endswith(".json")}
    for path in receipts:
        row = json.loads(path.read_text())
        base = Path(row.get("bundle") or "/nonexistent")
        if row.get("verdict") != "pass" or not (base / "MANIFEST.json").is_file():
            continue
        raw = (base / "MANIFEST.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.get("manifest_sha256"):
            continue
        theirs = json.loads(raw)["files"]
        if set(theirs) != set(mine) or any(theirs[k] != mine[k] for k in mine if k not in configs):
            continue
        changed, refused = [], []
        for name in sorted(configs):
            c, r = settings_diff(json.loads((base / name).read_text()), json.loads((bundle / name).read_text()))
            changed += [f"{name}:{k}" for k in c]
            refused += r
        if not refused:
            return f"settings-only vs {path.name}: {', '.join(changed) or 'no change'}"
    return None


def reconcile_check(gpu=None, resume=None, run=None):
    """live GPUs vs queue rows and the B300 cap (main's tools/ops/gpu_reconcile.py) before any B300
    launch; refuses on an unknown app, a stale row (except the legs this launch resumes: a chained death relaunch), or
    live + this launch over the cap. Returns the tool's count line."""
    import subprocess

    gpu = gpu or GPU
    new = int(gpu.split(":")[1]) if ":" in gpu else 1
    cap = int(os.environ.get("ARC_B300_CAP") or 8)
    # the tool parses the user's modal CLI's app-list JSON (the venv's older CLI names keys differently): PATH as is
    r = subprocess.run(["python3", str(ROOT / "tools/ops/gpu_reconcile.py"), "--b300-cap", str(cap), "--scope", "modal"],
                       capture_output=True, text=True, timeout=300)
    out = r.stdout + (r.stderr[-300:] if r.returncode and "ISSUE" not in r.stdout else "")
    return reconcile_verdict(out, new, cap, resume if resume is not None else RESUME_FROM)


def reconcile_verdict(out, new, cap, resume):
    own = {r.rstrip("/").rsplit("/", 1)[-1] for r in (resume or "").split(",") if r}
    count = re.search(r"B300\+B200 live (\d+)/", out)
    if not count:
        raise SystemExit(f"launch gate: gpu_reconcile gave no count: {out[-300:]}")
    issues = [l[6:] for l in out.splitlines() if l.startswith("ISSUE ")
              and not (" says running but app " in l and l.split()[3] in own)]
    if int(count.group(1)) + new > cap:
        issues.append(f"B300 live {count.group(1)} + this launch {new} > cap {cap}")
    if issues:
        raise SystemExit("launch gate: gpu_reconcile: " + "; ".join(issues))
    return f"{count.group(0)}{cap} + {new}"


RECORD_ID = None  # the run record's id when the launch is attempt N > 1 (run_id is then <record>-attemptN)


def gate_check(manifest, run_id=None):
    return {"reservation": gate_reservation(RECORD_ID or run_id), "preflight": gate_preflight(manifest),
            "reconcile": reconcile_check()}


def _records_cli(*args):
    import subprocess

    main = ROOT.parents[2] if ROOT.parent.name == "worktrees" and ROOT.parent.parent.name == ".claude" else ROOT
    py = main / ".venv/bin/python"
    return subprocess.run([str(py if py.exists() else sys.executable), str(main / "tools/ops/records.py"), *args],
                          capture_output=True, text=True, timeout=120)


def record_attempt(run_id):
    """The run record's current attempt (records.py: a reserve of an ended run is attempt N+1), 1 without a record."""
    r = _records_cli("show", run_id)
    try:
        return int((json.loads(r.stdout) or {}).get("attempt") or 1)
    except (ValueError, TypeError, AttributeError):
        return 1


def gate_reservation(run_id):
    """The run's record must hold a fresh open reservation (records.py reserve passed the arm's allowance and the
    global caps): a refused or skipped reserve never launches (24 Oct: a GPU launch ran 40 s after a printed refusal)."""
    r = _records_cli("require", str(run_id))
    if r.returncode:
        raise SystemExit(f"launch gate: {(r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else 'records.py require failed'}")
    return "reserved"


def gate_preflight(manifest):
    """Launch gate: no B300 launch without a passing preflight of this exact bundle; the
    bundle's own settings loader refuses unknown rl_loop / mg keys before anything starts."""
    import subprocess

    for study in sorted((BUNDLE / "configs").glob("*.json")):
        if '"rl_loop"' in study.read_text():
            check = subprocess.run([sys.executable, "-c", "import json, sys; from arena import rl_loop; "
                                    "rl_loop.settings(json.load(open(sys.argv[1]))['rl_loop'])", str(study)],
                                   env={**os.environ, "PYTHONPATH": str(BUNDLE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},  # no pyc in the bundle (hero-v5 06:35)
                                   capture_output=True, text=True)
            if check.returncode:
                raise SystemExit(f"launch gate: {study.name}: {check.stderr.strip().splitlines()[-1]}")
    for path in sorted((ROOT / "docs/validation-rl").glob("preflight-*.json")):
        row = json.loads(path.read_text())
        if row.get("manifest_sha256") == manifest and row.get("verdict") == "pass":
            return path.name
    settings = settings_only_gate(BUNDLE, sorted((ROOT / "docs/validation-rl").glob("preflight-*.json")))
    if settings:
        return settings
    if os.environ.get("ARC_RL1_GATE_ALONGSIDE") == "1":  # faster gate: the CPU preflight of this exact
        # bundle runs beside the GPU boot (a failed preflight stops the run); it must already be submitted
        for path in sorted((ROOT / "docs/validation-rl").glob("preflight-*.json")):
            row = json.loads(path.read_text())
            if row.get("manifest_sha256") == manifest and row.get("state") == "submitted":
                return f"alongside:{path.name}"
    raise SystemExit(f"launch gate: no passing preflight for bundle manifest {manifest[:12]} (run with --preflight-run, "
                     "then --verdict)")


PACKS = {}  # volume dir -> local extraction of its pack.tar.gz (verdicts of packed preflights)
PACKS_ROOTS = []  # their temp dirs, removed at exit


def packed_local(rel):
    """The local file or dir for ``rel`` when it lies in a packed preflight dir, else None."""
    import tempfile

    parts = rel.split("/")
    if len(parts) < 4 or parts[:2] != ["validation", "preflight"]:
        return None
    base = "/".join(parts[:3])
    if base not in PACKS:
        PACKS[base] = None
        try:
            data = b"".join(runs.read_file(f"{base}/{PACK}"))
        except Exception:  # not packed
            return None
        where = Path(tempfile.mkdtemp(prefix="rl1-pack-"))
        if not PACKS_ROOTS:  # 22 Oct: 31 left-behind extractions held 182 GB of local disk
            import atexit
            import shutil

            atexit.register(lambda: [shutil.rmtree(r, ignore_errors=True) for r in PACKS_ROOTS])
        PACKS_ROOTS.append(where)
        (where / PACK).write_bytes(data)
        del data

        def fetch(n):
            if n:
                (where / PACK).write_bytes(b"".join(runs.read_file(f"{base}/{PACK}")))
            return where / PACK

        try:
            want = b"".join(runs.read_file(f"{base}/{PACK}.sha256")).decode().split()[0]
        except Exception:  # packed before the sidecar (22 Oct): the gzip CRC still checks the read
            want = None
        extract_checked(fetch, where / "tree", want)
        PACKS[base] = where / "tree"
    if PACKS[base] is None:
        return None
    return PACKS[base] / "/".join(parts[3:])


def read_volume(rel):
    try:
        return b"".join(runs.read_file(rel)).decode(errors="replace")
    except Exception:  # absent
        local = packed_local(rel)
        return local.read_text(errors="replace") if local is not None and local.is_file() else None


def list_names(rel):
    """Entry names of a volume dir, or of the same dir inside a packed preflight."""
    local = packed_local(rel)
    if local is not None and local.is_dir():
        return [p.name for p in local.iterdir()]
    return [Path(entry.path).name for entry in runs.listdir(rel)]


def restore_kind(r):
    """A restored start in a stub preflight: "exact" (state and prompt token ids as recorded), "state" (the recorded
    actions replayed to the recorded board; the source is a real-card recording, so the prompt digest cannot match the
    stub's pseudo ids, and harness text that depends on token counts or CPU time (compaction point, tool timeouts on the
    CPU container) may differ; token exactness is the real card's check: ttests p2a-d 10/10 exact), or None."""
    if r.get("ok") and (r.get("prompt_digest") or {}).get("same"):
        return "exact"
    if (r.get("state") or {}).get("ok") and "/validation/preflight/" not in str(r.get("source") or ""):
        return "state"
    return None


JUDGED = "preflight-judged.json"  # the in-app verdict: judged in the container before the pack


def judge(base, read, names, resume, study, identity):
    """The preflight checks of one output ``base`` (validation/preflight/<id><tag>): ``read(rel)`` a file's text or None,
    ``names(rel)`` a dir's entry names; the one rule set behind the local verdict (preflight_verdict) and the in-app
    one (execute, before the pack). Returns the verdict fields in the record's order."""
    rl = f"{base}/arena/trials/rl-loop/rl"
    checks = {}
    checks["exit_clean"] = identity.get("exit_code") == 0 and not identity.get("error")
    results = json.loads(read(f"{base}/arena/bootstrap-results.json") or "{}")
    checks["trial_complete"] = bool(results) and all(v == "complete" for v in results.values())
    feed = [json.loads(l) for l in (read(f"{rl}/feed.jsonl") or "").splitlines() if l.strip()]
    # a preempted run judged on the union of its attempts (hero-v5 A2, 24 Oct: preempted after max_steps, the resumed
    # attempt went straight to v7_end and its own feed held no step): the earlier attempt's feed and state come first
    prior = str(identity.get("preempt_resume") or "").removeprefix("/runs/").strip("/")
    if prior:
        feed = [json.loads(l) for l in (read(f"{prior}/arena/trials/rl-loop/rl/feed.jsonl") or "").splitlines()
                if l.strip()] + feed
    # alert-only dosing: k3 over target / stop and the TIS share are reports, the step was taken
    kinds = [e.get("kind") for e in feed if not (e.get("kind") == "alert" and alert_only(e))]
    checks["feed_final_no_error"] = "final" in kinds and not {"error", "alert"} & set(kinds)
    r3 = [e for e in feed if e.get("kind") == "stop" and e.get("reason") == "r3_check_done"]
    if r3:  # an R3 check run ends on purpose before its first update: its check row stands in for rounds/updates
        rows = [e for e in feed if e.get("kind") == "r3_check"]
        checks["feed_final_no_error"] = not {"error", "alert"} & set(kinds)
        checks["r3_check_ran"] = bool(rows) and all(int((rows[-1].get(m) or {}).get("windows") or 0) > 0 for m in ("off", "served"))
    screen = json.loads(read(f"{rl}/screen.json") or "null")
    restore = {"starts": 0, "bad": [], "state_only": 0}
    if screen is not None:
        sets = screen.get("sets") or {}
        # a set fed only by recordings may lack starts when no recording reached its level (envs 22 Oct: the seed
        # screen reads every level its plays reach, no jumps); every recording must then be a usable play
        specs = {x["name"]: x for x in (study["rl_loop"].get("mg") or {}).get("sets") or []}
        no_starts = {e.get("name") for e in feed if e.get("kind") == "mg_set_missing" and e.get("error") == "no starts"}
        fed = {n for n, x in specs.items() if any(str(y).startswith("record:") for y in x.get("sources") or [])}
        unreached = [n for n in screen.get("missing") or [] if n in no_starts and n in fed
                     and all(str(x).startswith("record:") for x in specs[n]["sources"])
                     and int(specs[n].get("level") or 1) >= 2]  # every play starts at L1: an L1 set is always reached
        checks["every_set_played"] = (bool(sets) and all(int(v.get("tries") or 0) > 0 for v in sets.values())
                                      and not set(screen.get("missing") or []) - set(unreached))
        records = [e for e in feed if e.get("kind") == "mg_record"]
        if records or fed:  # record-fed sets with no recording at all fail here, never pass as "unreached"
            checks["every_record_usable"] = bool(records) and all(
                int((e.get("play") or {}).get("usable") or 0) >= 1 for e in records)
        restore["unreached"] = len(unreached)
        restore["unreached_sets"] = sorted(unreached)
        for name, v in sets.items():
            bench = json.loads(read(f"{rl}/{Path(v['folder']).name}/level-bench.json") or "{}")
            for row in bench.get("attempts") or []:
                r = row.get("restore")
                if r:
                    restore["starts"] += 1
                    kind = restore_kind(r)
                    restore["state_only"] += kind == "state"
                    if kind is None:
                        restore["bad"].append([name, row.get("label"), r.get("diff_count"), (r.get("state") or {}).get("detail")])
        checks["restored_starts_exact"] = restore["starts"] > 0 and not restore["bad"]
    elif (study["rl_loop"].get("mg") or {}).get("eval"):  # an eval / joint read: every arm on every set
        ev = study["rl_loop"]["mg"]["eval"]
        sets_ = [e for e in feed if e.get("kind") == "eval_set"]
        arms = len(ev.get("adapters") or {}) + (1 if ev.get("base", True) else 0) if "adapters" in ev else 2
        checks["eval_written"] = any(e.get("kind") == "final" and e.get("stop") == "eval" for e in feed)
        checks["every_arm_read"] = len(sets_) >= arms and all(int(str(e.get("clears", "0/0")).split("/")[1]) > 0 for e in sets_)
        if "adapters" in ev:
            checks["joint_pooled"] = any(e.get("kind") == "eval" and e.get("label") == "joint_read" for e in feed)
        for name in names(rl):
            if not name.startswith(("jr-", "ev-")):
                continue
            bench = json.loads(read(f"{rl}/{name}/level-bench.json") or "{}")
            for row in bench.get("attempts") or []:
                r = row.get("restore")
                if r:
                    restore["starts"] += 1
                    kind = restore_kind(r)
                    restore["state_only"] += kind == "state"
                    if kind is None:
                        restore["bad"].append([name, row.get("label"), r.get("diff_count"), (r.get("state") or {}).get("detail")])
        checks["restored_starts_exact"] = restore["starts"] > 0 and not restore["bad"]
    elif study["rl_loop"].get("mode") == "final":  # a final read: the eval row, every final bench, its starts
        evals = [e for e in feed if e.get("kind") == "eval" and e.get("label") == "mg_final"]
        finals = (study["rl_loop"]["mg"].get("final") or {})
        checks["eval_written"] = bool(evals)
        checks["every_final_set_paired"] = bool(evals) and all(
            ((evals[-1].get(n) or {}).get("paired") or {}).get("pairs", (evals[-1].get(n) or {}).get("policy", {}).get("tries", 0))
            for n in finals)
        names = names(rl)
        for name in [n for n in names if n.startswith("final-")]:
            bench = json.loads(read(f"{rl}/{name}/level-bench.json") or "{}")
            for row in bench.get("attempts") or []:
                r = row.get("restore")
                if r:
                    restore["starts"] += 1
                    kind = restore_kind(r)
                    restore["state_only"] += kind == "state"
                    if kind is None:
                        restore["bad"].append([name, row.get("label"), r.get("diff_count"), (r.get("state") or {}).get("detail")])
        checks["restored_starts_exact"] = restore["starts"] > 0 and not restore["bad"]
    elif (study["rl_loop"].get("mg") or {}).get("async"):  # v7 (arena.rl_async): steps, not rounds
        v7 = next((e.get("cfg") for e in feed if e.get("kind") == "v7_start" and e.get("cfg")),
                  study["rl_loop"]["mg"].get("v7") or {})  # the run's own (preflight-shrunk) settings
        steps = sorted({int(e.get("version", e.get("step") or 0)): e for e in feed if e.get("kind") == "step"}.values(),
                       key=lambda e: int(e.get("version", e.get("step") or 0)))  # a resume redoes steps after its checkpoint: the last counts
        checks["feed_final_no_error"] = "v7_end" in kinds and not {"error", "alert"} & set(kinds)
        checks["steps_ok"] = bool(steps) and all(
            isinstance(e.get("grad_norm"), (int, float)) and math.isfinite(e["grad_norm"]) for e in steps) and [
            e["version"] for e in steps] == list(range(steps[0]["version"], steps[0]["version"] + len(steps)))
        saved = set()
        for where in [base] + ([prior] if prior else []):
            try:
                saved |= set(names(f"{where}/state") or [])
            except OSError:
                pass
        checks["checkpoints_saved"] = bool(steps) and f"s{int(steps[-1]['step']):04d}" in saved
        checks["adapter_served"] = any(e.get("load_seconds") is not None for e in steps) or "served_late" in kinds
        if len(steps) >= int(v7.get("read_every") or 8):
            checks["reads_landed"] = "read" in kinds or "reads_cut" in kinds  # a manual stop cuts open reads (58d786bb7)
        if (v7.get("card2") or {}).get("on"):
            offs = [e for e in feed if e.get("kind") == "block_off"]
            checks["mode_b_played"] = bool(offs) and not any(e.get("error") for e in offs) and sum(
                int(e.get("served") or 0) for e in offs) > 0
        if resume:
            checks["resumed_from_checkpoint"] = any(e.get("kind") == "v7_start" and e.get("resume") for e in feed)
        for name in names(rl):
            if not re.match(r"(g\d{5}|x\d{4})-", name):
                continue
            bench = json.loads(read(f"{rl}/{name}/level-bench.json") or "{}")
            for row in bench.get("attempts") or []:
                r = row.get("restore")
                if r:
                    restore["starts"] += 1
                    kind = restore_kind(r)
                    restore["state_only"] += kind == "state"
                    if kind is None:
                        restore["bad"].append([name, row.get("label"), r.get("diff_count"), (r.get("state") or {}).get("detail")])
        checks["restored_starts_exact"] = restore["starts"] > 0 and not restore["bad"]
    else:  # a loop: rounds, updates, checkpoints, the served adapter, restored starts in every round's benches
        rounds = [e for e in feed if e.get("kind") == "round"]
        if not r3:
            checks["rounds_played"] = len(rounds) >= 1
            checks["updates_ok"] = any(r.get("update") for r in rounds) and all(
                (r.get("update") or {}).get("ok", True) for r in rounds)
            checks["checkpoints_saved"] = any(e.get("kind") == "checkpoint" and e.get("ok") for e in feed)
            checks["adapter_served"] = any((r.get("swap") or {}).get("load_seconds") is not None for r in rounds)
            mg = study["rl_loop"].get("mg") or {}
            if resume and rounds and mg.get("read_final") and not any(r.get("update") for r in rounds):
                # a resume segment that only plays the final read (pilot 2 batch arm: R1 + U1, then the R2 read): no
                # update to check here (the first segment's preflight had it); the read is played by the restored state
                del checks["updates_ok"], checks["adapter_served"], checks["checkpoints_saved"]  # pf19b: no update, nothing to save
                checks["read_by_restored_adapter"] = any(
                    e.get("kind") == "mg_adapter" and int(e.get("played_by") or 0) >= 1 for e in feed) and all(
                    int(r.get("played_by") or 0) >= 1 for r in rounds)
        if resume:
            checks["resumed_from_checkpoint"] = any(e.get("kind") == "resume" for e in feed)
        for name in names(rl):
            if not (name[:1] == "r" and name[1:3].isdigit() and name[3:4] == "-"):
                continue
            bench = json.loads(read(f"{rl}/{name}/level-bench.json") or "{}")
            for row in bench.get("attempts") or []:
                r = row.get("restore")
                if r:
                    restore["starts"] += 1
                    kind = restore_kind(r)
                    restore["state_only"] += kind == "state"
                    if kind is None:
                        restore["bad"].append([name, row.get("label"), r.get("diff_count"), (r.get("state") or {}).get("detail")])
        checks["restored_starts_exact"] = restore["starts"] > 0 and not restore["bad"]
    log = read(f"{base}/arena/serving/vllm-openai-server.log") or ""
    stub = [l.split("STUB ", 1)[1] for l in log.splitlines() if " STUB {" in l]
    stub = json.JSONDecoder().raw_decode(stub[-1])[0] if stub else {}  # KV_STUB {...} follows on the same line
    checks["stub_clean_on_recorded_paths"] = stub.get("errors") == 0 and stub.get("exact", 0) > 0
    if resume and (study["rl_loop"].get("mg") or {}).get("eval") and stub.get("errors") == 0 \
            and stub.get("any", 0) == 0:  # an eval resumed after a finished run: every try kept, the stub serves none
        checks["stub_clean_on_recorded_paths"] = True
        checks["every_try_reused"] = True
    boot = read(f"{base}/bootstrap.log") or ""
    checks["server_stopped"] = "server_stop_timeout" not in boot and bool(boot)
    probe = [e for e in feed if e.get("kind") == "lr_probe"]
    if probe:  # a trainer-only lr_probe bundle (P1, envs nextboard dose probe): no play, no stub traffic, no rounds;
        # every arm's row ok, the first arm's in-run smoke (mg.reload_check) and the next-board reads when configured
        mg = study["rl_loop"].get("mg") or {}
        rates = [e for e in feed if e.get("kind") == "lr_probe_rate"]
        checks = {"exit_clean": checks["exit_clean"], "trial_complete": checks["trial_complete"],
                  "no_error": not {"error", "alert"} & set(kinds), "server_stopped": checks["server_stopped"],
                  "every_arm_ok": bool(rates) and all(e.get("ok") for e in rates) and not probe[-1].get("skipped"),
                  "steps_probed": bool(rates) and all(s.get("kl_base") is not None for e in rates for s in e.get("steps") or [{}])}
        if mg.get("reload_check"):
            checks["first_arm_smoke"] = bool(rates) and bool((rates[0].get("reload") or {}).get("ok")) and bool(
                (rates[0].get("reload") or {}).get("update_after_reload"))
        if mg.get("nextboard"):
            checks["nextboard_aux_and_reads"] = bool(rates) and all(
                e.get("nb_after") and all((s or {}).get("windows") for s in (e.get("nextboard") or {}).get("steps") or [None])
                for e in rates) and "nextboard_read" in kinds
    return {"verdict": "pass" if all(checks.values()) else "fail", "checks": checks, "restore": restore, "stub": stub,
            "feed_kinds": {k: kinds.count(k) for k in sorted(set(kinds))}, "elapsed_seconds": identity.get("elapsed_seconds")}


def preflight_verdict(run_id):
    """Pass/fail of a preflight from its output: clean exit, every set played, restored starts exact, stub without
    errors and on recorded paths, server stopped; written into ``docs/validation-rl/preflight-<run id>.json``. The
    container's in-app judgement (``JUDGED``) is used when present (no pack download); ARC_RL1_VERDICT_PATH=local or
    an older leg judges from the volume as before."""
    record = ROOT / "docs/validation-rl" / f"preflight-{run_id}{PREFLIGHT_TAG}.json"
    data = json.loads(record.read_text())
    base = f"validation/preflight/{run_id}{PREFLIGHT_TAG}"
    judged = None
    if os.environ.get("ARC_RL1_VERDICT_PATH") != "local":  # in-app by default: both paths matched on pfXd (22 Oct)
        try:
            judged = json.loads(b"".join(runs.read_file(f"{base}/{JUDGED}")).decode())
        except Exception:  # an older or unfinished leg: judged here
            judged = None
    if judged is None:
        identity = json.loads(read_volume(f"{base}/modal-identity.json") or "{}")
        judged = judge(base, read_volume, list_names, data.get("resume"), bundle_study(), identity)
    data.update(**judged, judged_epoch=time.time())
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps({k: data[k] for k in ("verdict", "checks", "restore", "stub", "elapsed_seconds")}, indent=1))


# launch-side ARC_RL1_* names this launcher reads; anything else set locally never reaches the container (23 Oct:
# ARC_RL1_EVAL_ADAPTERS in the launch env was dropped and a read served the bundle's adapter)
LAUNCH_ENV = {"ARC_RL1_AUTHORIZATION", "ARC_RL1_BUNDLE", "ARC_RL1_CARDS", "ARC_RL1_GATE_ALONGSIDE", "ARC_RL1_GPU",
              "ARC_RL1_MAX_ROUNDS", "ARC_RL1_OVERRIDES", "ARC_RL1_PATCH_CHECK", "ARC_RL1_RESUME_FROM", "ARC_RL1_STALL",
              "ARC_RL1_STALL_DUMP", "ARC_RL1_STALL_KILL", "ARC_RL1_TIMEOUT", "ARC_RL1_TRAINER_KEY", "ARC_RL1_VERDICT_PATH"}
RL_IDS = ("rl1-loop-", "rl1-screen-", "rl1-final-", "rl1-restore-")  # RL's run ids (PTH's sft-* dispatch above)


def launch_env_errors(environ):
    """ARC_RL1_* variables set at launch that the launcher would silently drop."""
    out = []
    for key in sorted(k for k in environ if k.startswith("ARC_RL1_")):
        if key in READ_KEYS:
            out.append(f"{key}: pass it as ARC_RL1_OVERRIDES='{{\"{key}\": ...}}'")
        elif key not in LAUNCH_ENV and not key.startswith("ARC_RL1_PREFLIGHT_"):
            out.append(f"{key}: not read by the launcher, never reaches the run")
    return out


def record_app(run_id, app_id=None, cli=None):
    """After a detached launch: the app id goes into the run record (queue and ledger rows, records.py's shared
    setter), loudly (25 Oct: PTH's sft-stream-check-eosA-v1 ran as an unrecorded B300 that took three agents to name)."""
    app_id = app.app_id if app_id is None else app_id
    cli = cli or _records_cli
    if not str(app_id or "").startswith("ap-"):
        raise SystemExit(f"APP_ID NOT RECORDED {run_id}: no app id from Modal ({app_id!r}); set it by hand: "
                         f"records.py ledger {run_id} --set app_id='\"ap-...\"'")
    r = cli("ledger", str(run_id), "--set", f'app_id="{app_id}"')
    if r.returncode:
        raise SystemExit(f"APP_ID NOT RECORDED {run_id} ({app_id}): {((r.stderr or r.stdout) or '').strip()[-300:]}")
    print(f"record {run_id}: app_id {app_id}")
    return app_id


@app.local_entrypoint()
def main(run_id: str, dry_run: bool = False, preflight_run: bool = False, verdict: bool = False):
    out = launch(run_id, dry_run, preflight_run, verdict)
    if not (dry_run or preflight_run or verdict):
        record_app(RECORD_ID or run_id)  # a launch: its record names its app
    return out


def launch(run_id, dry_run=False, preflight_run=False, verdict=False):
    # PTH (sft-heal) entry points first, unchanged; then RL's
    if run_id.startswith("sft-stream-check-"):  # ARC_SC_FILES (volume paths), ARC_HEAL_TEACHER_MODEL, ARC_SC_NEAR
        e = os.environ
        json.loads(e.get("ARC_SC_EXTRA", "{}"))  # fail before the spawn
        call = stream_check.spawn(run_id, e["ARC_SC_FILES"], e["ARC_HEAL_TEACHER_MODEL"], e.get("ARC_SC_NEAR", "16384"),
                                  int(e.get("ARC_SC_LIMIT", "40000")), e.get("ARC_SC_PIN", "1,0"),
                                  ties=e.get("ARC_SC_TIES", "topk,index"), extra=e.get("ARC_SC_EXTRA", "{}"))
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
    global RECORD_ID
    bad = launch_env_errors(os.environ)
    if bad:
        raise SystemExit("launch env: " + "; ".join(bad))
    if not (dry_run or preflight_run or verdict):  # attempt N > 1 of one record: its own receipt and volume folder
        n = record_attempt(run_id)
        if n > 1:
            RECORD_ID, run_id = run_id, f"{run_id}-attempt{n}"
            print(f"record {RECORD_ID} attempt {n}: launching as {run_id}")
    if verdict:
        return preflight_verdict(run_id)
    return launch_rl(run_id, dry_run, preflight_run)


def launch_rl(run_id, dry_run=False, preflight_run=False):
    """An RL run (RL_IDS) of the bundle: dry run, its CPU preflight (``preflight_run``), or the run itself."""
    if not run_id.startswith(RL_IDS):
        raise ValueError(f"run ids start with {', '.join(RL_IDS)}")
    study = bundle_study()
    mode = study["rl_loop"]["mode"]
    if not run_id.startswith(f"rl1-{mode}-"):
        raise ValueError(f"the bundle's mode is {mode}")
    files = json.loads((BUNDLE / "MANIFEST.json").read_text())["files"]
    bad = [rel for rel, row in files.items() if (BUNDLE / rel).is_file() and isinstance(row, dict) and row.get(
        "sha256") and hashlib.sha256((BUNDLE / rel).read_bytes()).hexdigest() != row["sha256"]]
    if bad:
        raise SystemExit(f"bundle files changed after the build: {bad[:5]}")
    checked = server_check(study)
    record = ROOT / "docs/validation-rl" / f"modal-{run_id}.json"
    data = {"state": "dry_run", "run_id": run_id, "kind": "rl1", "mode": mode, "gpu": GPU, "timeout_seconds": TIMEOUT,
            "server_check": checked,
            "charged_bound_usd": round(bound(TIMEOUT), 2), "expected_hours": round(TIMEOUT / 3600, 2),
            "bundle": str(BUNDLE), "manifest_sha256": hashlib.sha256((BUNDLE / "MANIFEST.json").read_bytes()).hexdigest(),
            "image": AXOLOTL_IMAGE, "output": f"arc-prolong-runs:/validation/{run_id}", "authorization": AUTHORIZATION,
            "rl_loop": {k: study["rl_loop"][k] for k in ("mode", "rounds", "lr", "rank", "window", "probe_rates")}}
    if dry_run:
        print(json.dumps(data, indent=1))
        return
    if preflight_run:
        pf = ROOT / "docs/validation-rl" / f"preflight-{run_id}{PREFLIGHT_TAG}.json"
        if pf.exists():
            raise SystemExit(f"{pf} exists")
        spec = {"corpus": [f"/runs/{d}" for d in os.environ["ARC_RL1_PREFLIGHT_CORPUS"].split(",")],
                "serving": "/runs/" + os.environ["ARC_RL1_PREFLIGHT_SERVING"],
                "speed": float(os.environ.get("ARC_RL1_PREFLIGHT_SPEED") or 4000), "stall": STALL,
                "shrink": {"1": True, "2": 2}.get(os.environ.get("ARC_RL1_PREFLIGHT_SHRINK") or "", False),  # 2: minimum size (22 Oct)
                "branch_from_tar": os.environ.get("ARC_RL1_PREFLIGHT_BRANCH_TAR") == "1",
                "handoff_after_round": int(os.environ["ARC_RL1_PREFLIGHT_HANDOFF"]) if os.environ.get("ARC_RL1_PREFLIGHT_HANDOFF") else None,
                "kv": os.environ.get("ARC_RL1_PREFLIGHT_KV") == "1",  # replay_stub kv mode (KV from live contexts)
                # tied group rewards get a fixed ladder so updates run (replays repeat their recordings)
                "spread": os.environ.get("ARC_RL1_PREFLIGHT_SPREAD", "1") != "0",
                "served_route": (study.get("server") or {}).get("served_route"),  # stand-in route records (c)
                "die_after_s": float(os.environ["ARC_RL1_PREFLIGHT_DIE"]) if os.environ.get("ARC_RL1_PREFLIGHT_DIE") else None,
                # patch check (24 Oct, v6f's boot death): the real serving prep with the trial's server_env and
                # ARC_SERVING_PATCH_CHECK=1 on the pinned runtime tree, then exit; no loop (bootstrap ARC_PATCH_CHECK_ONLY)
                "patch_check": os.environ.get("ARC_RL1_PATCH_CHECK") == "1",
                "trial": os.environ.get("ARC_RL1_PREFLIGHT_TRIAL") or "rl-loop"}  # the bundle's trial to run
        call = preflight.spawn(run_id, json.dumps(spec), RESUME_FROM, PREFLIGHT_TAG, MAX_ROUNDS, OVERRIDES)
        row = {"state": "submitted", "run_id": run_id, "manifest_sha256": data["manifest_sha256"], "bundle": str(BUNDLE),
               "spec": spec, "cpu": None, "memory_gib": None, "timeout_seconds": TIMEOUT,
               "charged_bound_usd": round((TIMEOUT + 60) * (BILL_CPU * 1.31e-05 + PREFLIGHT_MEMORY_GIB * 2.22e-06) * 1.15, 2),
               "output": f"arc-prolong-runs:/validation/preflight/{run_id}{PREFLIGHT_TAG}", "resume": RESUME_FROM,
               "function_call_id": call.object_id}
        pf.write_text(json.dumps(row, indent=2))
        print(json.dumps(row))
        return
    if record.exists():
        raise SystemExit(f"{record} exists: run ids are single-use")
    data["launch_gate"] = gate_check(data["manifest_sha256"], run_id)
    if (study["rl_loop"].get("mg") or {}).get("eval"):  # an eval (joint read) starts no trainer
        data["trainer_smoke"] = "not needed: mg.eval starts no trainer"
    elif study["rl_loop"].get("mode") != "screen":  # a screen starts no trainer
        data["trainer_smoke"] = smoke_check(final=study["rl_loop"].get("mode") == "final")
    record.parent.mkdir(exist_ok=True)
    data["state"] = "submission_intent"
    record.write_text(json.dumps(data, indent=2))
    data["resume"] = RESUME_FROM
    data["max_rounds_override"] = MAX_ROUNDS or None
    overrides = json.loads(OVERRIDES) if OVERRIDES else {}
    ev = (study["rl_loop"].get("mg") or {}).get("eval")
    if ev and "adapters" in ev:  # the arms this launch means to serve, asserted in the container before play
        arms = dict(overrides.get("ARC_RL1_EVAL_ADAPTERS") or ev["adapters"])
        overrides["ARC_RL1_EXPECT_EVAL"] = arms
        print(f"eval arms: {arms} + base={ev.get('base', True)}")
    data["overrides"] = overrides or None
    call = run.spawn(run_id, RESUME_FROM, STALL, MAX_ROUNDS, json.dumps(overrides) if overrides else "")
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


