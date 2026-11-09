"""ARC-AGI-2 arm, port to the competition hardware: the pinned serving runtime on Modal L4s.

``probe`` (4 x L4, no server): host facts; ``arc2.l4_probe`` in the extracted runtime (does torch cu130 run on sm_89,
read bandwidth, dense GEMMs, the fork's own lines on tensor / pipeline parallelism and MoE backends); then the routed
expert kernel benchmark ``arena.expert_3bit_bench`` on card 0 (Marlin W4A16 and Humming on the served NVFP4 experts,
Humming on the int3 grid; real weights of layers 20 and 30, real routings, 1 to 56 rows). Output:
``arc-prolong-runs:/research/<run id>/`` (execution.json, probe.json, bench.json, logs), committed every 120 s.

``serve`` (4 x L4, the real server): the bundle's study in the first L4 profile (``arc2.l4_serving``: tensor
parallel 4, pruned 3-bit experts, 64K window, no sm_120 overlays), the arena's own start-up through
``python -m arc2.l4_serving``, then the probe (``arc2.l4_serving.probe``: plain completions, then the load's
first turns at rising concurrency, every server error body kept whole), then a short roster trial as load. Reads
host memory through the start, the head counts against four ranks, the streams that fit and tokens per second.
Output: ``result.json`` (always; "served" only when the load got replies), ``probe-N.json``, ``failure.json`` /
``failure-N.json`` (stage, the recipe's failure files, the server log's first errors after readiness, its error
lines and tail: one read), ``execution.json``, the logs and ``arena/`` (serving diagnostics, the trial).

    ARC_VALIDATION_ROOT=. modal run --detach scripts/modal_arc2_l4.py::serve_main \
        --run-id arc2-l4-serve-v3-modal-20261010 [--streams 16] [--kv int4|bf16] [--utilization 0.90]
        [--tasks 48] [--wall 420]

``--text-only`` (no vision tower on the ranks), ``--no-prefix-caching`` (one state block per group; every turn
prefilled whole), ``--inventory`` (``arc2.l4_inventory``: what each rank holds, by module group). The probe also
times the decode step at 1 to ``--streams`` requests at once, and the job times an all-reduce between the cards after
the server has stopped (``arc2.l4_allreduce``). A text-only start that fails is tried once more with the tower.
``--expert-parallel``: each rank holds whole experts (start 4: the tensor-parallel expert shards load padded,
4.6 GiB a rank); a start that fails with it ends the job with the error (without it the job is start 4 again).

``score_main --run-id <id> [--arm xhigh-8k] [--rates 130,180] [--utilization 0.87] [--wall 5400]
[--no-prefix-patches]``: a scored run of the bundle's roster trial under one arm (Flash, expert parallel, the
bundle's prefix-cache patches), at most 7,200 s; no all-reduce timing. A
server that dies in play still leaves the partial score (the trial's report, or its last progress record).

``--model qwen38-27b-fp8``: the same job on a reviewed dense checkpoint (``arena.candidate_profiles``; its
files at the assets volume root, ``/assets/<name>``) through ``arena.candidate_serving``: the profile's
quantization and draft tokens, BF16 cache.

One more start in the same job when at least 25 minutes are left: after a 4-bit start that does not come up, the
start that has (BF16 cache, 16 slots, 0.92); after an engine that died in the probe, ``arc2.l4_serving.fallback``
(0.03 less of each card, no more slots than were answered). ``failure-1.json`` and ``arena-attempt-1/`` keep the
first.

The ledger is the main checkout's (``ARC_VALIDATION_ROOT``); the launch record goes to this checkout's
``docs/validation-20261009``. Run from the arc2 worktree:

    ARC_VALIDATION_ROOT=. modal run --detach scripts/modal_arc2_l4.py \
        --run-id arc2-l4-probe-v1-modal-20261009      (with ``serve`` in the file: ``modal_arc2_l4.py::main``)
"""

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or HERE).resolve()
RECORDS = HERE / "docs/validation-20261009"
# Held-out routings of layers 20 and 30 (runs/ is not in git; the 30 September capture of the 3-bit study).
CAPTURE = Path(os.environ.get("ARC2_L4_CAPTURE") or
               ROOT / ".claude/worktrees/agent-a925b9bc05b48e382/runs/expert-3bit/capture")
TIMEOUT = 1800
GPUS, CPU, RAM_GIB = 4, 8, 96
# modal.com/pricing as known to the author: L4 per second (four cost about one RTX PRO 6000, 0.000842 in
# the ledger). Not in docs/spend.json rates_per_second: check before trusting the bound.
L4_RATE = 0.000222
RUNTIME_DIR = "/assets/flash-input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
RT = "/tmp/qwen38-flash-next-vllm-runtime"
WHITEOUTS = {8: ("usr/local/cuda-13.0/targets/x86_64-linux/lib/.wh.libcudart.so.13.0.88",),
             17: ("tmp/.wh.requirements-cuda.txt", "tmp/.wh.common.txt")}  # as scripts/modal_expert3bit.py

# The server job: a built arc2 bundle (its study, wheels, pruning mask and training tasks) and this checkout's arc2
# package in front of the bundle's (the L4 switch is newer than the bundle). 16 CPUs and 128 GiB: four workers, the
# API server and the lookup-table process (47.7 GiB) on one host.
BUNDLE = Path(os.environ.get("ARC2_L4_BUNDLE") or HERE / "dist/arc2-c240-tune-v1")
SERVE_TIMEOUT, SERVE_CPU, SERVE_RAM_GIB = 3300, 16, 128
SERVER_SECONDS = 1920  # the start's own limits are 1,500 s to ready and 1,800 s in all (arc2.l4_serving.SERVER_ENV)
SECOND_START_SECONDS = 1500  # what a fallback start, a short load and the clean-up need
SCORE_TIMEOUT = 7200  # a scored roster run: about 1,250 s around the play (score-v1), up to 5,400 s of play
CACHE_TIMEOUT = 1800  # the turn-gap cache probe: ready by about 1,000 s, about 400 s of probe, the evidence
# NVIDIA driver majors on which the four workers die at start ("NCCL error: unhandled cuda error" in the first
# all-reduce; one host, one start, 10 October). Every start that served ran on 580.95.05. ``any_driver`` in a
# job's options tries the start all the same (11 October: two launches in a row were handed 610.57.04), with
# NCCL's own log, and a serve job that meets the error again stops after that one start.
# 12 October (arc2-l4-nccl610-v2): on 610 the first reduction failed in NCCL's shared-memory transport importing a
# cuMem host buffer (Cuda failure 801); NCCL_CUMEM_HOST_ENABLE=0 (arc2.l4_serving, every L4 start) reduces right at
# 100 us (16 rows). No driver is refused now; the tuple stays for a driver that cannot serve.
BAD_DRIVERS = ()
# ``--hosts n`` (package_main): a container on such a host asks for the call again (try + 1) and keeps its cards busy
# until a later try has a sound host, so the next container cannot be placed on the same cards (11 October: four of
# six launches drew 610, twice in a row from the host just freed). Hold and tries bound the cost: 3 x 900 s =.
# 12 October: the hold did not keep the next try off the host (depth-v2: four tries, all 610, each started when
# the last hold ended, for nothing); a short hold and more tries cost a bad host.
HOST_TRIES = 6
HOST_HOLD_SECONDS = 60
# The package rehearsal: about 1,300 s to the play, the play, the stop and the evidence.
PACKAGE_TIMEOUT = 10200
# A kernel smoke (v38, 19 Oct: start, probe, a short play with a test kill of the engine): bound.
SMOKE_TIMEOUT = 6000
# The whole notebook: its 41,400 s (``RUN_SECONDS`` in build_arc2's cell) and the 300 s this wrapper keeps back.
# ``ARC2_L4_FULL_SIZE=48,186`` at launch sizes the job like Kaggle's host (48 CPUs, 186 GiB in its cgroup).
FULL_TIMEOUT = 41700
FULL_CPU, FULL_RAM_GIB = (int(x) for x in (os.environ.get("ARC2_L4_FULL_SIZE") or "16,128").split(","))
LIVE_SECONDS = 300
STATE_SECONDS = 60  # the play's state to the volume (package_body)
RESUME_SECONDS = 2400  # left on the first container's clock for a later one to play: a start, 600 s, the stop
OWNER_FULL = ("\"i approve the full-length L4 run on 240 entries, and raising the limit to 0\"; "
              "12 October: \"please start a 12h run of v13 on 4 L4's with our 240 task roster.\" "
              "(ARC-AGI-2 arm, run ids arc2-)")
# The 240-entry roster for four hours (up to 4 h for a comparison that cannot test its mechanism in
# less): which tasks get a first attempt shows only with more tasks than workers for hours, not on 43 tasks.
ROSTER_TIMEOUT = int(os.environ.get("ARC2_L4_ROSTER_TIMEOUT") or 14400)  # l4 arm reads: 7,200
RULE_ROSTER = ("Working conventions: \"Up to 4 hours is pre-approved for a comparison that cannot test its mechanism "
               "in less\" (240-entry roster against the stopped full-length run at equal tokens; ARC-AGI-2 arm, "
               "run ids arc2-)")
# The path check: the venv, the start up to the first thing that needs a card, the evidence. No GPU.
PATHCHECK_TIMEOUT, PATHCHECK_RAM_GIB = 2400, 72
FAKE_SMI = r'''#!/usr/bin/env python3
import sys
VALUES = {"name": "NVIDIA L4", "memory.total": "23034", "memory.used": "1", "memory.free": "22563",
          "driver_version": "580.95.05", "compute_cap": "8.9", "utilization.gpu": "0", "temperature.gpu": "40",
          "compute_mode": "Default", "pci.bus_id": "00000000:00:0{i}.0"}
UNITS = {"memory.total": " MiB", "memory.used": " MiB", "memory.free": " MiB", "utilization.gpu": " %"}
query = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--query-gpu=")), None)
form = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--format=")), "csv")
if query is None:
    print("fake nvidia-smi: 4 x NVIDIA L4 (path check, no cards)" if not any(a.startswith("--query") for a in sys.argv[1:]) else "", end="")
    sys.exit(0)
fields = query.split(",")
if "noheader" not in form:
    print(", ".join(fields))
for i in range(4):
    row = []
    for f in fields:
        value = str(i) if f == "index" else f"GPU-fake-{i}" if f in ("uuid", "gpu_uuid") else VALUES.get(f, "0").replace("{i}", str(i))
        row.append(value + ("" if "nounits" in form else UNITS.get(f, "")))
    print(", ".join(row))
'''
# The profile start: a text-only start (a hung one prints its stacks and the tower's start follows: 1,920 + 1,000 s),
# the probe with its traces (600 s), a short load, the all-reduce variants and the evidence; about 2,400 s if clean.
PROFILE_TIMEOUT = 4800
GARBAGE_TIMEOUT = 2140 # the garbage probe: ready by about 1,000 s, about 800 s of prefill load; two
NCCL_ENV_FILE = "/tmp/arc2-l4-runtime-env.json"  # the server's environment for arc2.l4_nccl; never on the volume
# Second turns for the garbage probe: {task: {"reasoning", "content"}}, sound first replies of the RTX tune run
# (training tasks; dist/ is not in git).
TURNS = Path(os.environ.get("ARC2_L4_TURNS") or HERE / "dist/arc2-l4-turns.json")

app = modal.App("arc2-l4-20261009", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .add_local_dir(str(HERE / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
         .add_local_dir(str(CAPTURE), "/opt/capture", copy=True)
         .add_local_file(__file__, "/root/modal_arc2_l4.py", copy=True))


serve_image = (modal.Image.debian_slim(python_version="3.12")
               .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1",
                            "build-essential", "ninja-build", "procps")
               .add_local_dir(str(BUNDLE), "/opt/bundle", copy=True, ignore=["**/__pycache__/**", "dataset-upload/**"])
               .add_local_dir(str(HERE / "src/arc2"), "/opt/l4src/arc2", copy=True, ignore=["**/__pycache__/**"])
               .add_local_file(str(TURNS), "/opt/turns.json", copy=True)
               .add_local_file(__file__, "/root/modal_arc2_l4.py", copy=True))


def bound_usd(seconds=TIMEOUT, cpu=CPU, ram_gib=RAM_GIB):
    return (seconds + 60) * (GPUS * L4_RATE + cpu * 0.0000131 + ram_gib * 0.00000222) * 1.15


def reserve(run_id, kind="arc2-l4-probe", seconds=TIMEOUT, cpu=CPU, ram_gib=RAM_GIB, authorization=None):
    """The run's reservation, made beforehand with ``tools/ops/records.py reserve`` (records only through
    records.py; spend.json is a generated view). Refuses unless the run holds an open, fresh reservation whose bound
    covers this job's; returns that bound. ``kind``/``authorization`` are the reserve call's, not checked here."""
    return reserved(run_id, bound_usd(seconds, cpu, ram_gib))


def reserved(run_id, need):
    import subprocess

    tool = [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/ops/records.py")]
    done = subprocess.run([*tool, "require", run_id], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"{run_id}: no open reservation ({(done.stderr or done.stdout).strip()[-400:]}); "
                         "reserve with tools/ops/records.py reserve first")
    shown = subprocess.run([*tool, "show", run_id], capture_output=True, text=True, check=True).stdout
    bound = float(json.loads(shown)["ledger"][-1]["row"]["charged_bound_usd"])
    if bound + 1e-6 < need:
        raise SystemExit(f"{run_id}: reserved ${bound:.2f} < this job's bound ${need:.2f}")
    return bound


def extract_runtime():
    import shutil
    import subprocess
    import time

    started = time.time()
    root = Path(RT)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    manifest = json.loads((Path(RUNTIME_DIR) / "runtime-manifest.json").read_text())
    for row in sorted(manifest["selected_layers"], key=lambda r: int(r["index"])):
        layer = Path(RUNTIME_DIR) / row["file"]
        if not layer.is_file():
            layer = Path(RUNTIME_DIR) / f"{row['file']}.blob"
        if layer.stat().st_size != int(row["size"]):
            raise RuntimeError(f"wrong-sized runtime layer {layer}")
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
    return time.time() - started


def runtime_env(devices):
    """scripts/modal_expert3bit.py's runtime environment with the L4's architecture (8.9) and ``devices`` visible."""
    import subprocess

    site = Path(RT) / "usr/local/lib/python3.12/dist-packages"
    cuda_home = Path(RT) / "usr/local/cuda-13.0"
    driver = None
    out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "libcuda.so.1" in line and "=>" in line and "stub" not in line:
            driver = Path(line.split("=>", 1)[1].strip()).resolve().parent
    libs = [cuda_home / "targets/x86_64-linux/lib", cuda_home / "lib64", site / "torch/lib",
            *sorted(p for p in (site / "nvidia").glob("*/lib") if p.is_dir()),
            *([driver] if driver else []), Path("/usr/local/nvidia/lib64"), Path("/usr/lib/x86_64-linux-gnu")]
    env = os.environ.copy()
    for d in ("/tmp/rtcache", "/tmp/rtcache/triton", "/tmp/rtcache/humming", "/tmp/rttmp"):
        Path(d).mkdir(parents=True, exist_ok=True)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(site / "nvidia_cutlass_dsl/dsl_packages"), str(site), "/opt/src"]),
        "PATH": os.pathsep.join([str(cuda_home / "bin"), str(Path(RT) / "usr/local/bin"), env.get("PATH", "")]),
        "LD_LIBRARY_PATH": os.pathsep.join(str(p) for p in libs if p.is_dir()),
        "CUDA_HOME": str(cuda_home), "CUDA_PATH": str(cuda_home), "CUDACXX": str(cuda_home / "bin/nvcc"),
        "TORCH_CUDA_ARCH_LIST": "8.9", "TRITON_CACHE_DIR": "/tmp/rtcache/triton", "XDG_CACHE_HOME": "/tmp/rtcache",
        "HUMMING_CACHE_DIR": "/tmp/rtcache/humming", "TMPDIR": "/tmp/rttmp", "VLLM_NO_USAGE_STATS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False", "CUDA_VISIBLE_DEVICES": devices,
        "PYTHONDONTWRITEBYTECODE": "1"})
    return env, str(site)


def host_facts():
    import shutil
    import subprocess

    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    memory = {line.split(":")[0]: line.split(":")[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal", "MemAvailable"))}
    return {"gpus": smi.stdout.strip().splitlines(), "smi_error": smi.stderr.strip()[:300] or None, "memory": memory,
            "cpus": os.cpu_count(), "tmp_free_gib": round(shutil.disk_usage("/tmp").free / 2**30, 1)}


def second_start(out, run_id, started, volume=None, resume=False):
    """Modal starts a call again when it takes the container back (preemption; rehearsal 9: two hours in, a full
    repeat over the first container's files, and a second lifetime billed). The first container leaves a marker; a
    later one writes ``rerun.json`` and returns it: the first one's flushed files stand, and playing again takes a
    new run id. ``resume`` (``package_body``, whose driver takes up a saved state): the later one is counted in the
    marker and goes on, on the first one's clock."""
    volume, marker = volume or runs, out / "started.json"
    if not marker.is_file():
        marker.write_text(json.dumps({"run_id": run_id, "started_epoch": started, "containers": [started]}))
        volume.commit()
        return None
    first = json.loads(marker.read_text())
    if resume:
        first["containers"] = [*first.get("containers", [first["started_epoch"]]), started]
        marker.write_text(json.dumps(first))
        volume.commit()
        return {"run_id": run_id, "status": "resumed", "first": first}
    result = {"run_id": run_id, "status": "interrupted", "started_epoch": started, "first": first,
              "error": "the call's container was started a second time: not repeated"}
    (out / "rerun.json").write_text(json.dumps(result, indent=1))
    volume.commit()
    print("ARC2_L4_RERUN " + json.dumps(result), flush=True)
    return result


def wrong_host(drivers, options):
    """Not a host for this call: a driver the server cannot start on (``BAD_DRIVERS``) unless ``any_driver``; with
    ``want_driver`` (the NCCL diagnosis wants exactly such a host), any host without that driver."""
    want = options.get("want_driver")
    if want:
        return not any(d.split(".")[0] == str(want) for d in drivers)
    return any(d.split(".")[0] in BAD_DRIVERS for d in drivers) and not options.get("any_driver")


def another_host(out, run_id, options, drivers, started, attempt, respawn, volume=None):
    """A container whose host cannot serve (``BAD_DRIVERS``) hands the call on: its own file in ``hosts/``, the call
    spawned again as try ``attempt + 1``, then it holds its cards until a later try reports a sound host (or gives
    up, or ``HOST_HOLD_SECONDS``). Each container writes only its own files, so the commits cannot collide. Not the
    preemption path: a refused container leaves no ``started.json``, and a held call that Modal starts again finds
    the later try's file and returns (``package_body``)."""
    import time

    volume, hosts = volume or runs, out / "hosts"
    hosts.mkdir(parents=True, exist_ok=True)
    mine = hosts / f"try-{attempt}-bad.json"
    row = {"run_id": run_id, "try": attempt, "drivers": drivers, "started_epoch": started}
    mine.write_text(json.dumps(row))
    volume.commit()
    call = respawn(run_id, {**options, "host_try": attempt + 1})
    row["next"] = getattr(call, "object_id", None)
    print("ARC2_L4_HOST " + json.dumps(row), flush=True)
    released, errors = None, []
    # want_driver (the NCCL diagnosis hunts a bad host): holding a sound host only costs (nccl610-v1: 2 x 900 s)
    hold = 0 if options.get("want_driver") else HOST_HOLD_SECONDS
    while released is None and time.time() - started < hold:
        time.sleep(15)
        try:
            volume.reload()
            later = sorted(p.name for p in hosts.glob("try-*-ok.json") if int(p.name.split("-")[1]) > attempt)
            released = later[0] if later else "gave-up.json" if (hosts / "gave-up.json").is_file() else None
        except Exception as exc:  # noqa: BLE001 - keep holding
            errors.append(repr(exc)[:200])
    row.update(status="moved", released_by=released, seconds=round(time.time() - started), reload_errors=errors[:3])
    mine.write_text(json.dumps(row))
    volume.commit()
    return row


def run_logged(command, env, log_path, timeout):
    """The command's output to ``log_path`` and to stdout (the Modal log); the return code, or "timeout"."""
    import subprocess
    import threading

    with open(log_path, "w") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def pump():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line[:2000], end="", flush=True)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = "timeout"
        thread.join(timeout=10)
    return code


@app.function(image=image, gpu=f"L4:{GPUS}", timeout=TIMEOUT,
              retries=0, max_containers=1, volumes={"/assets": assets, "/runs": runs})
def probe(run_id: str, bench_args: list):
    import sys
    import threading
    import time
    import traceback

    started = time.time()
    out = Path("/runs/research") / run_id
    out.mkdir(parents=True, exist_ok=True)
    again = second_start(out, run_id, started)
    if again:
        return again
    record = {"run_id": run_id, "kind": "arc2-l4-probe", "gpu": f"L4:{GPUS}", "started_epoch": started}
    stop = threading.Event()

    def save():
        record["elapsed_seconds"] = time.time() - started
        (out / "execution.json").write_text(json.dumps(record, indent=2))
        runs.commit()

    def keep():
        while not stop.wait(120):
            try:
                save()
            except Exception:  # noqa: BLE001 - the next commit tries again
                pass

    threading.Thread(target=keep, daemon=True).start()
    try:
        record["host"] = host_facts()
        print(json.dumps({"event": "arc2_l4", "host": record["host"]}), flush=True)
        record["extract_seconds"] = extract_runtime()
        save()
        env, site = runtime_env(",".join(str(i) for i in range(GPUS)))
        record["probe_returncode"] = run_logged(
            [sys.executable, "-u", "-m", "arc2.l4_probe", "--site", site, "--out", str(out / "probe.json"),
             "--dump", str(out / "source")],
            env, out / "probe.log", 900)
        save()
        deadline = TIMEOUT - (time.time() - started) - 180
        env, _ = runtime_env("0")
        record["bench_returncode"] = run_logged(
            [sys.executable, "-u", "-m", "arena.expert_3bit_bench", "--model-dir", MODEL_DIR, "--capture",
             "/opt/capture", "--out", str(out / "bench.json"), "--deadline-seconds", str(deadline), "--groups", "16",
             "--low-memory", *bench_args], env, out / "bench.log", deadline + 60)
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    stop.set()
    save()
    return record


@app.function(image=image, gpu="L4", timeout=TIMEOUT,
              retries=0, max_containers=1, volumes={"/assets": assets, "/runs": runs})
def dense(run_id: str, bench_args: list):
    """``arc2.l4_dense_bench`` on one L4 in the serving runtime, no server: a step's dense projections in BF16
    and from weights of fewer bits."""
    import sys
    import time
    import traceback

    started = time.time()
    out = Path("/runs/research") / run_id
    out.mkdir(parents=True, exist_ok=True)
    again = second_start(out, run_id, started)
    if again:
        return again
    record = {"run_id": run_id, "kind": "arc2-l4-dense", "gpu": "L4", "started_epoch": started}
    try:
        record["host"] = host_facts()
        record["extract_seconds"] = extract_runtime()
        env, _ = runtime_env("0")
        record["returncode"] = run_logged([sys.executable, "-u", "-m", "arc2.l4_dense_bench", "--out",
                                           str(out / "dense.json"), *bench_args], env, out / "dense.log",
                                          TIMEOUT - (time.time() - started) - 120)
        if (out / "dense.json").is_file():
            found = json.loads((out / "dense.json").read_text())
            record.update(totals=found.get("totals"), errors=len(found.get("errors") or []), env=found.get("env"))
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    record["elapsed_seconds"] = time.time() - started
    (out / "result.json").write_text(json.dumps(record, indent=1))
    runs.commit()
    return record


@app.function(image=image, gpu=f"L4:{GPUS}", timeout=TIMEOUT,
              retries=0, max_containers=1, volumes={"/assets": assets, "/runs": runs})
def collective(run_id: str, bench_args: list):
    """``arc2.l4_hostar`` on four L4s in the serving runtime, no server: the all-reduce through host memory
    against NCCL and vLLM's one-shot kernel, eager and inside a CUDA graph, with the ranks' bits compared."""
    import sys
    import time
    import traceback

    started = time.time()
    out = Path("/runs/research") / run_id
    out.mkdir(parents=True, exist_ok=True)
    again = second_start(out, run_id, started)
    if again:
        return again
    record = {"run_id": run_id, "kind": "arc2-l4-collective", "gpu": f"L4:{GPUS}", "started_epoch": started}
    try:
        record["host"] = host_facts()
        record["extract_seconds"] = extract_runtime()
        env, _ = runtime_env(",".join(str(i) for i in range(GPUS)))
        module = "arc2.l4_hostar"
        if bench_args and bench_args[0].startswith("module:"):  # another no-model bench, e.g. module:arc2.l4_allreduce
            module, bench_args = bench_args[0][7:], bench_args[1:]
        record["module"] = module
        record["returncode"] = run_logged([sys.executable, "-u", "-m", module, "--out",
                                           str(out / "hostar.json"), *bench_args], env, out / "hostar.log",
                                          TIMEOUT - (time.time() - started) - 120)
        if (out / "hostar.json").is_file():
            found = json.loads((out / "hostar.json").read_text())
            record.update(ranks_agree=found.get("ranks_agree"), rank0=found.get("rank0"))
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    record["elapsed_seconds"] = time.time() - started
    (out / "result.json").write_text(json.dumps(record, indent=1))
    runs.commit()
    return record


@app.local_entrypoint()
def hostar_main(run_id: str, args: str = ""):
    """Four L4s, about ten minutes, no model: ``--args "--paths nccl,host,custom"``."""
    bound = reserve(run_id, "arc2-l4-collective")
    call = collective.spawn(run_id, args.split() if args else [])
    record = {"run_id": run_id, "kind": "arc2-l4-collective", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": TIMEOUT, "gpu": f"L4:{GPUS}", "state": "submitted",
              "args": args, "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def dense_main(run_id: str, args: str = ""):
    """One L4, about ten minutes: ``--args "--rows 16,24,32"``. The bound is the four-card one (reserve)."""
    bound = reserve(run_id, "arc2-l4-dense")
    call = dense.spawn(run_id, args.split() if args else [])
    record = {"run_id": run_id, "kind": "arc2-l4-dense", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": TIMEOUT, "gpu": "L4", "state": "submitted", "args": args,
              "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def main(run_id: str, args: str = ""):
    if not (CAPTURE / "meta.json").is_file():
        raise FileNotFoundError(f"routing capture not found: {CAPTURE} (set ARC2_L4_CAPTURE)")
    bound = reserve(run_id)
    call = probe.spawn(run_id, args.split() if args else [])
    record = {"run_id": run_id, "kind": "arc2-l4-probe", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": TIMEOUT, "gpu": f"L4:{GPUS}", "state": "submitted",
              "args": args, "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=SERVE_TIMEOUT, retries=0, max_containers=1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def serve(run_id: str, options: dict):
    return serve_body(run_id, options, SERVE_TIMEOUT)


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=SCORE_TIMEOUT, retries=0, max_containers=1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def score(run_id: str, options: dict):
    return serve_body(run_id, options, SCORE_TIMEOUT)


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=CACHE_TIMEOUT, retries=0, max_containers=1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def cache(run_id: str, options: dict):
    return serve_body(run_id, options, CACHE_TIMEOUT)


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=GARBAGE_TIMEOUT, retries=0, max_containers=HOST_TRIES + 1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def garbage(run_id: str, options: dict):
    return serve_body(run_id, options, GARBAGE_TIMEOUT, respawn=garbage.spawn)


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=PROFILE_TIMEOUT, retries=0, max_containers=1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def profile(run_id: str, options: dict):
    return serve_body(run_id, options, PROFILE_TIMEOUT)


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=PACKAGE_TIMEOUT, retries=0, max_containers=HOST_TRIES + 1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})  # more than one container only while a refused host is held
def package(run_id: str, options: dict):
    return package_body(run_id, options, PACKAGE_TIMEOUT, respawn=package.spawn)


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=SMOKE_TIMEOUT, retries=0, max_containers=1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def smoke(run_id: str, options: dict):
    return package_body(run_id, options, SMOKE_TIMEOUT)


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=FULL_TIMEOUT, retries=0, max_containers=1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def full(run_id: str, options: dict):
    return package_body(run_id, options, FULL_TIMEOUT)


@app.function(image=serve_image, gpu=f"L4:{GPUS}",
              timeout=ROSTER_TIMEOUT, retries=0, max_containers=1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def roster(run_id: str, options: dict):
    return package_body(run_id, options, ROSTER_TIMEOUT)


@app.function(image=serve_image, cpu=(8, 16),  # CPU preflights
              timeout=PATHCHECK_TIMEOUT, retries=0, max_containers=1, block_network=True,
              volumes={"/assets": assets, "/runs": runs})
def pathcheck(run_id: str, options: dict):
    """``package`` on a host without cards, with an ``nvidia-smi`` that reports four L4s: the real driver and the
    real server start as far as they go before something needs a GPU. For path and layout faults, for cents."""
    fake = Path("/tmp/fakebin")
    fake.mkdir()
    (fake / "nvidia-smi").write_text(FAKE_SMI)
    (fake / "nvidia-smi").chmod(0o755)
    os.environ["PATH"] = f"{fake}:{os.environ['PATH']}"
    result = package_body(run_id, options, PATHCHECK_TIMEOUT)
    if result.get("status") == "interrupted":
        return result
    # The start stops at the driver library, before the server command builds the expert checkpoint: that step on
    # its own, as arena.serving calls it, in the agent environment's Python.
    import subprocess

    study = json.loads(Path("/opt/bundle/configs/comparison-l4.json").read_text())
    if study.get("research_checkpoint"):  # the AR farm with the real headers (ar_slim reads the base tensors it keeps)
        program = (
            "import json, sys, types\nfrom pathlib import Path\nfrom arena import research_checkpoint\n"
            "from arena.serving import bundle_expert_prune\n"
            "study = json.loads(Path(sys.argv[1]).read_text())\n"
            "mask = bundle_expert_prune(study['server']['expert_prune'], Path('/opt/bundle'))\n"
            "recipe = types.SimpleNamespace(verify_and_extract_runtime=lambda *a, **k: None, SERVED_MODEL_NAME='x')\n"
            "Path('/tmp/arena/farm-check').mkdir(parents=True, exist_ok=True)\n"
            "out = research_checkpoint.configure(recipe, '/tmp/arena/farm-check', study['research_checkpoint'], mask)\n"
            "print('FARM ' + json.dumps({k: str(v)[:400] for k, v in out.items()}))\n")
        done = subprocess.run(["/tmp/arena/agent-venv/bin/python", "-c", program, "/opt/bundle/configs/comparison-l4.json"],
                              env={**os.environ, "PYTHONPATH": "/opt/bundle/src"}, capture_output=True, text=True)
        result["overlay"] = {"code": done.returncode, "out": done.stdout[-1500:], "error": done.stderr[-1500:]}
        (Path("/runs/research") / run_id / "result.json").write_text(json.dumps(result, indent=1))
        runs.commit()
        return result
    program = (
        "import json, sys\nfrom pathlib import Path\nfrom arc_infra.serving import dataset_mount\n"
        "from arena import expert_overlay\nstudy = json.loads(Path(sys.argv[1]).read_text())['server']\n"
        "spec = study['expert_overlay']\n"
        "record = expert_overlay.materialise(sys.argv[2], dataset_mount(spec['dataset'], '/kaggle/input'), "
        "Path('/tmp/arena/overlay-check'), spec['manifest_sha256'], check=spec.get('check', 'sha256'), "
        "expert_prune_sha256=study['expert_prune']['sha256'])\n"
        "print('OVERLAY ' + json.dumps({k: v for k, v in record.items() if not isinstance(v, (list, dict))}))\n")
    done = subprocess.run(["/tmp/arena/agent-venv/bin/python", "-c", program, "/opt/bundle/configs/comparison-l4.json",
                           "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"],
                          env={**os.environ, "PYTHONPATH": "/opt/bundle/src"}, capture_output=True, text=True)
    result["overlay"] = {"code": done.returncode, "out": done.stdout[-1500:], "error": done.stderr[-1500:]}
    (Path("/runs/research") / run_id / "result.json").write_text(json.dumps(result, indent=1))
    runs.commit()
    return result


def package_body(run_id, options, total, respawn=None):
    """The Kaggle package as the notebook runs it (``arc2.kaggle`` from the bundle alone, the same inputs under
    /kaggle/input), on a sample of the public evaluation tasks with their solutions: the rehearsal's score.

    Preemption: the play's state (``play-state.json``: every task's programs, tokens and phase) goes to the volume
    every minute beside the submission, and the logs every 300 s. A later container of the same call (a bundle
    whose driver has ``--resume-state``) plays on from them until the first container's end, in ``arc2-c<n>``;
    what is lost is the start, the open conversations and up to two minutes of state."""
    import random
    import sys
    import threading
    import time

    started = time.time()
    out, bundle = Path("/runs/research") / run_id, Path("/opt/bundle")
    out.mkdir(parents=True, exist_ok=True)
    # The host first (``another_host``): a refused container is not one of the run's, and leaves no marker.
    host, attempt, limit = host_facts(), int(options.get("host_try", 0)), int(options.get("hosts", 0))
    drivers = sorted({row.split(", ")[3] for row in host.get("gpus") or [] if row.count(", ") >= 3})
    suspect = any(d.split(".")[0] in BAD_DRIVERS for d in drivers)
    later = [p.name for p in (out / "hosts").glob("try-*.json") if int(p.name.split("-")[1]) > attempt]
    if later:  # this call handed the run on and was started again (its held container was taken back)
        return {"run_id": run_id, "status": "superseded", "try": attempt, "by": sorted(later)[0]}
    if suspect and not options.get("any_driver"):  # as serve_body: seconds, not two starts that cannot work
        error = None
        if respawn is not None and attempt < min(limit, HOST_TRIES):
            try:
                return another_host(out, run_id, options, drivers, started, attempt, respawn)
            except Exception as exc:  # noqa: BLE001 - the call could not be spawned again: the plain refusal
                error = repr(exc)[:300]
        result = {"run_id": run_id, "status": "failed", "error": f"host driver {drivers}: relaunch for another host",
                  "seconds": round(time.time() - started), "options": options, "host_try": attempt,
                  **({"respawn_error": error} if error else {})}
        try:  # an earlier container's play stands (a later container of a played run drew the host)
            result["play_seconds_before"] = float(json.loads((out / "played.json").read_text())["play_seconds"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if limit:
            (out / "hosts").mkdir(exist_ok=True)
            (out / "hosts" / "gave-up.json").write_text(json.dumps(result))
        (out / "result.json").write_text(json.dumps(result, indent=1))
        runs.commit()
        return result
    if limit or attempt:  # what the held containers wait for
        (out / "hosts").mkdir(exist_ok=True)
        (out / "hosts" / f"try-{attempt}-ok.json").write_text(json.dumps({"try": attempt, "drivers": drivers,
                                                                          "started_epoch": started}))
        runs.commit()
    again = second_start(out, run_id, started,
                         resume="--resume-state" in (bundle / "src/arc2/kaggle.py").read_text())
    if again and again["status"] != "resumed":
        return again
    number = len(again["first"]["containers"]) if again else 1
    end = (again["first"]["started_epoch"] if again else started) + total - 300  # every container's clock
    before = 0.0  # seconds of play in earlier containers
    if again:
        try:
            before = float(json.loads((out / "played.json").read_text())["play_seconds"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    drive = out / ("arc2" if number == 1 else f"arc2-c{number}")
    Path("/kaggle").mkdir(exist_ok=True)
    spec = json.loads((bundle / "configs/comparison.json").read_text()).get("research_checkpoint")
    if spec:  # AR + heal overlay (v55b4): the farm's inputs as the notebook links them; the overlay sits under models/
        heal = Path(spec["heal_overlay"])
        for folder, source in ((Path("/kaggle/input"), Path("/assets/flash-input")),
                               (Path("/kaggle/input/datasets"), Path("/assets/flash-input/datasets")),
                               (heal.parent, Path("/assets/flash-input/datasets") / heal.parent.name)):
            folder.mkdir()
            for child in source.iterdir():
                if not (child.name == "datasets" and folder.name == "input") and child.name != heal.parent.name:
                    (folder / child.name).symlink_to(child, target_is_directory=child.is_dir())
        heal.symlink_to("/assets/flash-input/models/arc-heal/p75-arc2-c1147", target_is_directory=True)
        Path("/tmp/arc-w4-model").symlink_to(
            "/assets/flash-input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
            target_is_directory=True)
        Path("/tmp/arc-model").symlink_to(MODEL_DIR, target_is_directory=True)
    else:
        Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    folder = bundle / "data/arc2/evaluation"
    names = sorted(random.Random(options["seed"]).sample(sorted(p.stem for p in folder.glob("*.json")),
                                                         options["tasks"]))
    tasks = {name: json.loads((folder / f"{name}.json").read_text()) for name in names}
    # ``clones``: every task that many times under its own name, as the 240 entries of the c240 roster: to the
    # driver they are 240 tasks, as the competition's are.
    tasks = {name + ("" if k == 0 else f"_{k + 1}"): task for name, task in tasks.items()
             for k in range(int(options.get("clones", 1)))}
    (out / "challenges.json").write_text(json.dumps({
        name: {"train": t["train"], "test": [{"input": x["input"]} for x in t["test"]]} for name, t in tasks.items()}))
    (out / "solutions.json").write_text(json.dumps({name: [x["output"] for x in t["test"]] for name, t in tasks.items()}))
    stop = threading.Event()
    # The live record, where tools/ops/mon.py reads every Modal run's (validation/<run>/arena/live on this volume).
    sys.path.insert(0, "/opt/l4src")
    from arc2 import kaggle

    live, history = Path("/runs/validation") / run_id / "arena/live", []

    def write(path, data):  # whole or not at all: a later container reads these
        path.with_name(path.name + ".part").write_text(json.dumps(data))
        os.replace(path.with_name(path.name + ".part"), path)

    def publish():
        def read(path):
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):  # not there yet, or caught between two writes
                return {}

        row = kaggle.live_row(read(drive / "run.json"), read(drive / "arena/live/arc2-submit.json"),
                              time.time() - started, time.time(), history)
        row.update(run=run_id, host_cpus=host.get("cpus"), container=number)
        engine = row["live"]["engine"]
        if before:  # the tasks' tokens are carried by the state; the play's seconds are added here
            row["play_seconds"] = round(row["play_seconds"] + before, 1)
            engine["tokens_per_second_mean"] = engine["generated_tokens"] / row["play_seconds"]
        history.append((row["elapsed_seconds"], engine["generated_tokens"]))
        write(out / "played.json", {"play_seconds": row["play_seconds"], "container": number,
                                    "tokens": engine["generated_tokens"], "epoch": row["epoch"]})
        live.mkdir(parents=True, exist_ok=True)
        (live / "progress.json").write_text(json.dumps(row, indent=1))
        with (live / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def logs():  # the play's reply and admission logs so far: a cancelled call writes nothing else
        import shutil

        for path in Path("/tmp/arena/trials").glob("*/*.jsonl"):
            target = drive / "live-logs" / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    def state():  # what a later container plays on from (the trial writes it whole, every minute)
        try:
            saved = json.loads(Path("/tmp/arena/play-state.json").read_text())
        except (OSError, ValueError):
            return
        write(out / "play-state.json", saved)

    def server_logs():  # the server's own logs, live (19 Oct: v38's start phases; Kaggle keeps none of a cut run)
        import shutil

        for path in Path("/tmp/arena/serving").glob("*.log"):
            target = drive / "live-logs" / "serving" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    def keep():
        rounds = 0
        while not stop.wait(STATE_SECONDS):
            rounds += 1
            try:
                state()
                server_logs()
                if rounds % (LIVE_SECONDS // STATE_SECONDS) == 0:
                    publish()
                    logs()
                runs.commit()
            except Exception as exc:  # noqa: BLE001 - the next round tries again
                print(f"ARC2_L4_LIVE {type(exc).__name__}: {exc}"[:300], flush=True)

    threading.Thread(target=keep, daemon=True).start()
    env = {**os.environ, "PYTHONPATH": "/opt/bundle/src", "PYTHONDONTWRITEBYTECODE": "1", "ARC_FORCE_CHROOT": "1",
           "LIBRARY_PATH": "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")}
    # The scored runs' layout (serve_body): the bundle at /opt/bundle, its study in configs, /tmp/arena to work in.
    play = options["play_seconds"] - before
    if again and (end - time.time() < RESUME_SECONDS or play < 600):
        code = f"not resumed: {int(end - time.time())} s left on the first container's clock, {int(play)} s of play"
    else:
        try:
            code = run_logged([sys.executable, "-u", "-m", "arc2.kaggle", "--bundle", str(bundle), "--working",
                               "/tmp/arena", "--output", str(drive), "--input-root", "/kaggle/input", "--challenges",
                               str(out / "challenges.json"), "--solutions", str(out / "solutions.json"),
                               "--submission", str(out / "submission.json"), "--end-epoch", str(end),
                               "--play-seconds", str(play),
                               *(["--resume-state", str(out / "play-state.json")] if again else []),
                               *(["--options", json.dumps(options["options"])] if options.get("options") else [])],
                              env, out / ("driver.log" if number == 1 else f"driver-c{number}.log"),
                              end + 150 - time.time())
        except BaseException:  # Modal's interrupt before it takes the machine: the state as it is, then out
            stop.set()
            try:
                state()
                publish()
                logs()
                runs.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"ARC2_L4_LIVE {type(exc).__name__}: {exc}"[:300], flush=True)
            raise
    stop.set()
    try:
        publish()
    except Exception as exc:  # noqa: BLE001 - the result below stands without it
        print(f"ARC2_L4_LIVE {type(exc).__name__}: {exc}"[:300], flush=True)
    record = json.loads((drive / "run.json").read_text()) if (drive / "run.json").is_file() else {}
    submitted = json.loads((out / "submission.json").read_text()) if (out / "submission.json").is_file() else {}
    rows = [(row, answer) for name, t in tasks.items()
            for row, answer in zip(submitted.get(name) or [{}] * len(t["test"]), (x["output"] for x in t["test"]))]
    right = sum(answer in (row.get("attempt_1"), row.get("attempt_2")) for row, answer in rows)
    status = "served" if code == 0 else "interrupted" if str(code).startswith("not resumed") else "failed"
    result = {"run_id": run_id, "status": status, "driver_exit": code,
              "seconds": round(time.time() - started), "options": options, "tasks": len(tasks), "outputs": len(rows),
              "right": right, "score_outputs": 100 * right / len(rows), "host": host, "run": record,
              "containers": number, "play_seconds_before": before}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("ARC2_L4_RESULT " + json.dumps({k: result[k] for k in ("status", "seconds", "outputs", "right")}), flush=True)
    runs.commit()
    return result


class Finished(Exception):
    """The job's work ended before the trial (the cache probe): not an error."""


def serve_body(run_id, options, total, respawn=None):
    import runpy
    import shutil
    import subprocess
    import sys
    import threading
    import time
    import traceback
    import urllib.request

    started = time.time()
    out = Path("/runs/research") / run_id
    out.mkdir(parents=True, exist_ok=True)
    attempt, limit = int(options.get("host_try", 0)), int(options.get("hosts", 0))
    if limit or attempt:  # ``--hosts``: the host before the run's marker, as package_body
        rows = host_facts().get("gpus") or []
        drivers = sorted({row.split(", ")[3] for row in rows if row.count(", ") >= 3})
        later = [p.name for p in (out / "hosts").glob("try-*.json") if int(p.name.split("-")[1]) > attempt]
        if later:
            return {"run_id": run_id, "status": "superseded", "try": attempt, "by": sorted(later)[0]}
        (out / "hosts").mkdir(exist_ok=True)
        if wrong_host(drivers, options):
            if respawn is not None and attempt < min(limit, HOST_TRIES):
                try:
                    return another_host(out, run_id, options, drivers, started, attempt, respawn)
                except Exception as exc:  # noqa: BLE001 - not spawned again: the plain refusal below
                    print("ARC2_L4_HOST " + json.dumps({"respawn_error": repr(exc)[:300]}), flush=True)
            (out / "hosts" / "gave-up.json").write_text(json.dumps({"try": attempt, "drivers": drivers}))
        else:  # what the held containers wait for
            (out / "hosts" / f"try-{attempt}-ok.json").write_text(json.dumps({"try": attempt, "drivers": drivers,
                                                                              "started_epoch": started}))
        runs.commit()
    again = second_start(out, run_id, started)
    if again:
        return again
    bundle, working = Path("/opt/bundle"), Path("/tmp/arena")
    serving = working / "serving"
    sys.path.insert(0, "/opt/l4src")
    from arc2 import l4_serving

    record = {"run_id": run_id, "kind": "arc2-l4-serve", "gpu": f"L4:{GPUS}", "started_epoch": started,
              "options": options, "stage": "host", "samples": []}
    stop = threading.Event()

    def save():
        record["elapsed_seconds"] = time.time() - started
        (out / "execution.json").write_text(json.dumps(record, indent=1))
        runs.commit()

    def metric(text, name):
        return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith(name))

    def sample():  # host memory, card memory and the server's counters through the start and the load
        row = {"t": round(time.time() - started), "stage": record["stage"]}
        try:
            memory = l4_serving.meminfo()
            row.update(available_gib=round(memory["MemAvailable"] / 2**30, 1),
                       cached_gib=round(memory.get("Cached", 0) / 2**30, 1),
                       card_used_mib=[int(c.split(",")[2]) for c in l4_serving.cards()])
            if record["stage"].startswith(("facts", "probe", "trial")):  # a served engine
                with urllib.request.urlopen("http://127.0.0.1:1234/metrics", timeout=5) as response:
                    text = response.read().decode()
                row.update(generated=metric(text, "vllm:generation_tokens_total"),
                           prompt=metric(text, "vllm:prompt_tokens_total"),
                           cache_queries=metric(text, "vllm:prefix_cache_queries_total"),
                           cache_hits=metric(text, "vllm:prefix_cache_hits_total"),
                           running=metric(text, "vllm:num_requests_running"),
                           waiting=metric(text, "vllm:num_requests_waiting"))
        except Exception as exc:  # noqa: BLE001 - a sample, not the run
            row["error"] = repr(exc)[:200]
        record["samples"].append(row)

    def keep():
        ticks = 0
        while not stop.wait(20):
            ticks += 1
            sample()
            if ticks % 6 == 0:
                try:
                    save()
                except Exception:  # noqa: BLE001 - the next commit tries again
                    pass

    def stage(name):
        record["stage"] = name
        print(json.dumps({"event": "arc2_l4_serve", "stage": name, "elapsed": round(time.time() - started)}),
              flush=True)

    def stop_server():
        program = ("from arc_infra.serving import load_recipe; import sys; "
                   "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()")
        try:
            subprocess.run([python, "-c", program, str(serving), "/kaggle/input"], env=env, check=False, timeout=150)
        except subprocess.TimeoutExpired:
            pass

    def facts():  # what the ready server's processes hold and say
        ps = subprocess.run(["ps", "-eo", "pid,ppid,rss,comm,args", "--sort=-rss"], capture_output=True, text=True)
        record["after_ready"] = {"meminfo": l4_serving.meminfo(), "cards": l4_serving.cards(),
                                 "processes": [line[:220] for line in ps.stdout.splitlines()[:14]]}
        seen = {}  # did the 3-bit switch and the four cards reach the server's processes (the repack logs nothing)
        for line in ps.stdout.splitlines()[1:14]:
            try:
                pairs = [item.decode(errors="replace").split("=", 1)
                         for item in Path(f"/proc/{line.split()[0]}/environ").read_bytes().split(b"\0") if b"=" in item]
                seen[line.split()[0]] = {key: value for key, value in pairs if key in (
                    "ARC_EXPERT_INT3", "CUDA_VISIBLE_DEVICES", "TORCH_CUDA_ARCH_LIST", "VLLM_PLE_CPU_OFFLOAD")}
            except Exception:  # noqa: BLE001 - a process that ended, or not ours to read
                pass
        record["after_ready"]["environment"] = seen
        record["server"] = l4_serving.server_facts(serving)

    threading.Thread(target=keep, daemon=True).start()
    env = {**os.environ, "PYTHONPATH": "/opt/l4src:/opt/bundle/src", "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PIP_NO_INDEX": "1",
           "MPLBACKEND": "Agg", "ARC_FORCE_CHROOT": "1", "ARC_VALIDATION_OUTPUT": str(out),
           "LIBRARY_PATH": "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")}
    python, bootstrap, error, process_log = None, None, None, None
    candidate = options.get("model")
    try:
        record["host"] = host_facts()
        record["head_counts"] = l4_serving.head_counts(
            Path(f"/assets/{candidate}/config.json") if candidate else Path(MODEL_DIR) / "config.json")
        print(json.dumps({"event": "arc2_l4_serve", "host": record["host"], "head_counts": record["head_counts"]}),
              flush=True)
        drivers = sorted({row.split(", ")[3] for row in record["host"].get("gpus") or [] if row.count(", ") >= 3})
        suspect = any(d.split(".")[0] in BAD_DRIVERS for d in drivers)
        record["driver"] = {"versions": drivers, "suspect": suspect}
        if options.get("want_driver") and wrong_host(drivers, options):
            raise RuntimeError(f"host driver {drivers}: not the {options['want_driver']} this call asked for")
        if suspect and not options.get("any_driver"):  # seconds, not a 340 s start that cannot work
            raise RuntimeError(f"host driver {drivers}: NCCL's first all-reduce fails on it (arc2-l4-attention-v2); "
                               "relaunch for another host")
        Path("/kaggle").mkdir(exist_ok=True)
        if candidate:  # its checkpoint sits at the volume root: mounted where the arena looks for the dataset
            owner, dataset = options["profile"]["dataset"].split("/")
            for folder, source in ((Path("/kaggle/input"), Path("/assets/flash-input")),
                                   (Path("/kaggle/input/datasets"), Path("/assets/flash-input/datasets")),
                                   (Path("/kaggle/input/datasets") / owner, Path("/assets/flash-input/datasets") / owner)):
                folder.mkdir()
                for child in (source.iterdir() if source.is_dir() else ()):
                    if not (child.name == "datasets" and folder.name == "input") and child.name != owner:
                        (folder / child.name).symlink_to(child, target_is_directory=child.is_dir())
            if not (Path("/kaggle/input/datasets") / owner / dataset).exists():
                (Path("/kaggle/input/datasets") / owner / dataset).symlink_to(f"/assets/{candidate}",
                                                                               target_is_directory=True)
        else:
            Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
        base = json.loads((bundle / "configs/comparison.json").read_text())
        study_path = bundle / "configs/comparison-l4.json"
        stage("venv")
        bootstrap = runpy.run_path(str(bundle / "bootstrap.py"))
        working.mkdir(parents=True)
        python = bootstrap["install_agent_environment"](bundle, working / "agent-venv", env)
        record["attempts"] = []
        choice, number, study = {k: v for k, v in options.items()
                                 if k not in ("cache_probe", "garbage", "any_driver", "hosts", "host_try", "nccl",
                                              "want_driver", "depth")}, 0, None
        while True:
            number += 1
            study = l4_serving.study_for(base, choice)
            study_path.write_text(json.dumps(study, indent=1))
            (out / f"comparison-l4-{number}.json").write_text(json.dumps(study, indent=1))
            stage(f"server-{number}")
            process_log = out / f"serving-process-{number}.log"
            begun = time.time()
            code = run_logged([python, "-u", "-m", "arc2.l4_serving", "serve-candidate" if candidate else "serve",
                               "--study", str(study_path),
                               "--working", str(serving), "--input-root", "/kaggle/input"],
                              {**env, **study["trials"][0]["server_env"], **({"NCCL_DEBUG": "INFO"} if suspect else {}),
                               **({"ARC2_L4_ENV_DUMP": NCCL_ENV_FILE} if options.get("nccl") else {})},
                              process_log, min(SERVER_SECONDS, total - 520))
            entry = {"options": choice, "code": code, "seconds": round(time.time() - begun)}
            record["attempts"].append(entry)
            record["server_seconds"] = round(time.time() - started)
            if options.get("nccl"):  # the 610 diagnosis: the start as it went, then NCCL alone in its environment
                stage("nccl")
                record["driver"]["start_code"] = code
                record["driver"]["nccl_lines"] = list(dict.fromkeys(
                    line.strip()[-300:] for log in sorted(serving.glob("*.log"))
                    for line in log.read_text(errors="replace").splitlines()
                    if "NCCL WARN" in line or "NCCL error" in line))[:20]
                stop_server()
                save()
                server_env = json.loads(Path(NCCL_ENV_FILE).read_text())
                limit = max(120, total - (time.time() - started) - 240)
                run_logged([python, "-u", "-m", "arc2.l4_nccl", "--out", str(out / "nccl.json"), "--logs",
                            str(out / "nccl-logs"), "--budget", str(int(limit - 100))],
                           {**server_env, "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                            "PYTHONPATH": os.pathsep.join([str(server_env.get("PYTHONPATH", "")), "/opt/l4src"])},
                           out / "nccl-process.log", limit)
                if not (out / "nccl.json").is_file():
                    raise RuntimeError("the NCCL diagnosis left no record; see its process log")
                record["nccl"] = json.loads((out / "nccl.json").read_text())
                raise Finished
            if code == 0:
                stage(f"facts-{number}")
                facts()
                entry["cache"] = record["server"].get("cache")
                save()
                if options.get("cache_probe"):  # turn 2 of a conversation after a pause: cached or not
                    stage("cache-probe")
                    limit = max(120, total - (time.time() - started) - 200)
                    run_logged([python, "-u", "-m", "arc2.l4_serving", "cache-probe", "--study", str(study_path),
                                "--source", str(bundle), "--output", str(out / "cache-probe.json")],
                               env, out / "cache-probe-process.log", limit)
                    if not (out / "cache-probe.json").is_file():
                        raise RuntimeError("the cache probe left no record; see its process log")
                    record["cache_probe"] = found = json.loads((out / "cache-probe.json").read_text())
                    found["kv_roundtrip"] = l4_serving.finite_report(serving)["kv_roundtrip"]
                    found["verdict"]["nan_lines"] = found["kv_roundtrip"]["nan"]
                    found["verdict"]["passed"] = found["verdict"]["passed"] and not found["kv_roundtrip"]["nan"]
                    record["server_errors_after_ready"] = l4_serving.log_first(serving)
                    raise Finished
                if options.get("depth"):  # the decode step at a play's context depth (arc2.l4_depth)
                    stage("depth")
                    limit = max(120, total - (time.time() - started) - 240)
                    run_logged([python, "-u", "-m", "arc2.l4_depth", "--study", str(study_path), "--output",
                                str(out / "depth.json"), "--depths", str(options["depth"]["depths"]), "--rows",
                                str(options["depth"]["rows"]), "--budget-seconds", str(int(limit - 60))],
                               env, out / "depth-process.log", limit)
                    record["server_errors_after_ready"] = l4_serving.log_first(serving)
                    record["allocator_retries"] = l4_serving.allocator_retries(serving)
                    if not (out / "depth.json").is_file():
                        raise RuntimeError("the depth probe left no record; see its process log")
                    record["depth"] = json.loads((out / "depth.json").read_text())
                    raise Finished
                if options.get("garbage"):  # garbage replies and non-finite expert outputs under prefill load
                    stage("garbage")
                    limit = max(120, total - (time.time() - started) - 200)
                    tokens = int(options["garbage"].get("tokens") or 200)
                    window = limit - 150 - 0.07 * tokens  # the last replies end inside the limit
                    phases = ",".join(f"{int(streams)}:{max(30, int(share * window))}"
                                      for streams, share in options["garbage"]["phases"])
                    try:
                        run_logged([python, "-u", "-m", "arc2.l4_serving", "garbage-probe", "--study",
                                    str(study_path), "--source", str(bundle), "--phases", phases, "--turns",
                                    "/opt/turns.json", "--tokens", str(tokens), "--replies",
                                    str(out / "garbage-replies.jsonl"), "--output", str(out / "garbage.json")],
                                   env, out / "garbage-process.log", limit)
                    finally:  # the server log's evidence, whatever the client did
                        record["finite"] = l4_serving.finite_report(serving)
                        save()
                    if not (out / "garbage.json").is_file():
                        raise RuntimeError("the garbage probe left no record; see its process log")
                    record["garbage"] = json.loads((out / "garbage.json").read_text())
                    record["server_errors_after_ready"] = l4_serving.log_first(serving)
                    raise Finished
                stage(f"probe-{number}")
                left = total - (time.time() - started) - 200
                limit = max(60, min(600, left - 480))
                run_logged([python, "-u", "-m", "arc2.l4_serving", "probe", "--study", str(study_path), "--source",
                            str(bundle), "--server-dir", str(serving), "--output", str(out / f"probe-{number}.json"),
                            "--budget-seconds", str(limit)], env, out / f"probe-process-{number}.log", limit + 560)
                if not (out / f"probe-{number}.json").is_file():  # the probe itself broke: the load still runs
                    entry["probe"] = record["probe"] = {"error": "no probe record; see the probe's process log"}
                    break
                entry["probe"] = record["probe"] = found = json.loads((out / f"probe-{number}.json").read_text())
                save()
                if found["health"] == 200 and found["passed"] >= 1:  # tokens came and the engine lives
                    if found["passed"] < found["top"]:  # fewer first turns at once: the prefill limiter's slots
                        choice = {**choice, "prefill_slots": found["passed"]}
                        study = l4_serving.study_for(base, choice)
                    if choice.get("sticky"):  # no worker waits at the gate in the middle of an attempt
                        cap = study["trials"][0]["arms"][0].get("cap") or study["completion_cap"]
                        entry = found["gate"]["real_loop_streams"][str(cap)]
                        fit = entry.get("second_turn_reserved", entry["second_turn"])
                        choice = {**choice, "slots": max(8, min(int(choice["streams"]), fit - 2))}
                        study = l4_serving.study_for(base, choice)
                        entry["slots"] = choice["slots"]
                    break
                problem = f"the probe failed (health {found['health']}): {found['stopped']}"
                following = l4_serving.fallback(choice, found["passed"])
            else:
                problem = f"the server start ended with {code}"
                nccl = [line.strip()[-300:] for log in sorted(serving.glob("*.log"))
                        for line in log.read_text(errors="replace").splitlines() if "NCCL error" in line][:3]
                if nccl:  # the host, not the options: no second start
                    record["driver"]["nccl_errors"] = nccl
                    problem += f" with an NCCL error on driver {drivers}"
                if candidate or nccl:
                    following = None
                elif choice.get("text_only") or choice.get("embed_host"):  # the start as the play has it
                    following = {**choice, "text_only": False, "embed_host": False}
                elif choice.get("expert_parallel"):  # without it the job would repeat start 4: stop with the error
                    following = None
                else:
                    following = ({**choice, "kv": "bf16", "streams": 16, "slots": None, "utilization": "0.92"}
                                 if choice["kv"] == "int4" else None)
            report = l4_serving.failure(serving, record["stage"], problem, process_log)
            (out / f"failure-{number}.json").write_text(json.dumps(report, indent=1))
            entry["failure"] = {key: report.get(key) for key in ("error", "native-preflight-failure.json",
                                                                 "first_errors_after_ready")}
            print("ARC2_L4_FAILURE " + json.dumps({"attempt": number, "error": problem}), flush=True)
            if number == 2 or following is None or total - (time.time() - started) < SECOND_START_SECONDS:
                raise RuntimeError(problem)
            stop_server()
            bootstrap["collect_evidence"](working, out / f"arena-attempt-{number}")
            shutil.move(str(serving), str(working / f"serving-failed-{number}"))
            choice = following
            save()
        # After the trial: stopping the server, the all-reduce timing, copying the evidence and the commit (380 s).
        left = total - (time.time() - started) - (530 if options.get("profile") else 380)
        wall = int(min(study["trials"][0]["wall_seconds"], left - 120))
        if wall < 120:
            raise RuntimeError(f"no time left for the load trial ({int(left)} s)")
        study["trials"][0]["wall_seconds"] = wall
        study_path.write_text(json.dumps(study, indent=1))
        (out / "comparison-l4-load.json").write_text(json.dumps(study, indent=1))
        stage("trial")
        process_log = out / "trial-process.log"
        (working / "trials").mkdir(parents=True, exist_ok=True)
        code = run_logged([python, "-u", "-m", "arc2.l4_serving", "load", "--study", str(study_path), "--source",
                           str(bundle), "--output", str(working / "trials/load"), "--server-dir", str(serving)],
                          env, process_log, wall + 120)
        report, cut = working / "trials/load/report.json", False
        if not report.is_file() and (working / "trials/load/progress.json").is_file():
            report, cut = working / "trials/load/progress.json", True  # the trial's process was ended: its last minute
        if report.is_file():
            data = json.loads(report.read_text())
            record["load"] = {key: data.get(key) for key in (
                "status", "play_seconds", "completion_tokens", "prompt_tokens", "tokens_per_second",
                "server_generated_tokens", "server_tokens_per_second", "replies", "candidates", "request_errors",
                "streams", "gate", "task_errors", "server_errors", "arm_tokens", "final", "ended", "roster",
                "tokens_per_task")}
            if cut:
                record["load"].update(status="cut", play_seconds=data.get("elapsed_seconds"),
                                      task_errors=[f"the trial's process ended with {code}; its last progress record"])
            events = working / "trials/load/events.jsonl"
            record["load"]["steady"] = l4_serving.steady_rate(events)
            record["load"]["prompt_cache"] = l4_serving.prompt_cache(events)
        play = [row for row in record["samples"] if row["stage"] == "trial"]
        counted = [row for row in play if "cache_queries" in row]
        if "load" in record and counted:  # the server's own counters over the play window, in tokens
            first, last = counted[0], counted[-1]
            asked, hit = last["cache_queries"] - first["cache_queries"], last["cache_hits"] - first["cache_hits"]
            record["load"]["server_window"] = {
                "seconds": last["t"] - first["t"], "generated": last["generated"] - first["generated"],
                "tokens_per_second": (last["generated"] - first["generated"]) / max(1, last["t"] - first["t"]),
                "prompt_tokens": last["prompt"] - first["prompt"], "prefix_cache_queries": asked,
                "prefix_cache_hits": hit, "hit_share": hit / asked if asked else None}
        if "load" in record:
            record["load"]["card_used_mib_peak"] = (
                [max(row["card_used_mib"][i] for row in play if "card_used_mib" in row) for i in range(GPUS)]
                if any("card_used_mib" in row for row in play) else None)
            record["load"]["allocator_retries"] = l4_serving.allocator_retries(serving)
            sample()
            record["load"]["health_after"] = record["samples"][-1].get("error") or "metrics answered"
        record["server_errors_after_ready"] = l4_serving.log_first(serving)
        if "load" not in record:
            raise RuntimeError(f"the load trial ended with {code} and left no report")
        save()
        if not options.get("roster"):  # a scored run has its numbers out here: no all-reduce timing
            stage("allreduce")  # the cards are free once the server has stopped; the runtime is still extracted
            try:
                stop_server()
                identity = json.loads((serving / "vllm-server-identity.json").read_text())
                server_env = json.loads((serving / "serving-environment.json").read_text())
                run_logged([identity["argv"][0], "-u", "-m", "arc2.l4_allreduce", "--out", str(out / "allreduce.json"),
                            *(["--variants"] if options.get("profile") else [])],
                           {**os.environ, **{k: str(v) for k, v in server_env.items()},
                            "PYTHONPATH": os.pathsep.join([str(server_env.get("PYTHONPATH", "")), "/opt/l4src"]),
                            "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, out / "allreduce.log",
                           320 if options.get("profile") else 170)
                if (out / "allreduce.json").is_file():
                    record["allreduce"] = json.loads((out / "allreduce.json").read_text())
            except Exception as exc:  # noqa: BLE001 - a measurement beside the run
                record["allreduce"] = {"error": repr(exc)[:500]}
        if not record["load"].get("replies"):  # a start that served the probe only is not a served load
            raise RuntimeError("the trial got no reply: " + "; ".join(record["server_errors_after_ready"][:3]))
        if options.get("roster") and record["load"]["status"] != "complete":  # the partial score is in the result
            raise RuntimeError(f"the roster trial ended as {record['load']['status']}: "
                               + "; ".join(map(str, (record["load"].get("task_errors") or [])[:3])))
        stage("done")
    except Finished:
        stage("done")
    except BaseException as exc:  # noqa: BLE001
        error = exc
        record["error"], record["traceback"] = repr(exc)[:2000], traceback.format_exc()[-4000:]
    failed_stage = record["stage"]
    try:
        if error is not None:
            report = l4_serving.failure(serving, failed_stage, error, process_log)
            report["traceback"] = record["traceback"]
            (out / "failure.json").write_text(json.dumps(report, indent=1))
            print("ARC2_L4_FAILURE " + json.dumps({k: report[k] for k in ("stage", "error")}), flush=True)
        if python is not None:
            stop_server()
        if bootstrap is not None:
            bootstrap["collect_evidence"](working, out / "arena")
        for traces in (serving / "torch-profile", working / "torch-profile"):  # the profiler's traces and tables
            if traces.is_dir():
                shutil.copytree(traces, out / "torch-profile", dirs_exist_ok=True)
    except BaseException as exc:  # noqa: BLE001 - the result below still goes out
        record["cleanup_error"] = repr(exc)[:500]
    stop.set()
    samples = [row for row in record["samples"] if "available_gib" in row]
    result = {"run_id": run_id, "status": "failed" if error is not None else "served", "stage": failed_stage,
              "tokens": (record.get("load") or {}).get("completion_tokens"), "probe": record.get("probe"),
              "server_errors_after_ready": record.get("server_errors_after_ready"),
              "error": record.get("error"), "seconds": round(time.time() - started), "options": options,
              "attempts": record.get("attempts"), "head_counts": record.get("head_counts"), "host": record.get("host"),
              "host_available_gib": {"start": samples[0]["available_gib"] if samples else None,
                                     "lowest": min((row["available_gib"] for row in samples), default=None)},
              "card_used_mib_peak": [max(row["card_used_mib"][i] for row in samples) for i in range(GPUS)]
              if samples else None,
              "server_seconds": record.get("server_seconds"), "after_ready": record.get("after_ready"),
              "server": record.get("server"), "load": record.get("load"), "allreduce": record.get("allreduce"),
              "driver": record.get("driver"),
              "cache_probe": record.get("cache_probe"), "garbage": record.get("garbage"),
              "finite": record.get("finite"), "nccl": record.get("nccl"), "depth": record.get("depth"),
              "allocator_retries": record.get("allocator_retries")}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("ARC2_L4_RESULT " + json.dumps({k: result[k] for k in ("status", "stage", "error", "seconds")}), flush=True)
    save()
    return result


@app.local_entrypoint()
def serve_main(run_id: str, streams: int = 16, kv: str = "int4", tasks: int = 48, wall: int = 420,
               utilization: str = "0.90", model: str = "", text_only: bool = False, prefix_caching: bool = True,
               inventory: bool = False, expert_parallel: bool = False):
    import sys

    sys.path.insert(0, str(HERE / "src"))
    from arc2 import l4_serving

    options = {"streams": streams, "kv": kv, "tasks": tasks, "wall": wall, "utilization": utilization,
               "text_only": text_only, "prefix_caching": prefix_caching, "inventory": inventory,
               "expert_parallel": expert_parallel}
    if model:  # a dense candidate checkpoint (arena.candidate_profiles) in place of Flash; BF16 cache
        options = {**options, "kv": "bf16", "model": model, "profile": l4_serving.candidate_profile(model)}
    l4_serving.study_for(json.loads((BUNDLE / "configs/comparison.json").read_text()), options)  # valid before spend
    bound = reserve(run_id, "arc2-l4-serve", SERVE_TIMEOUT, SERVE_CPU, SERVE_RAM_GIB)
    call = serve.spawn(run_id, options)
    record = {"run_id": run_id, "kind": "arc2-l4-serve", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": SERVE_TIMEOUT, "gpu": f"L4:{GPUS}", "state": "submitted",
              "options": options, "bundle": BUNDLE.name, "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def profile_main(run_id: str, utilization: str = "0.90", text_only: bool = True, wall: int = 300,
                 embed_host: bool = False, any_driver: bool = False, tower_off: bool = False):
    """One start of the play's serving with the torch profiler on: the vision tower off (does it start; its
    stacks if not), no decode graphs under 8 rows, the buffer inventory; the probe traces 40 decode steps at
    16, 24 and 32 rows and the prefill of four first turns (``torch-profile/`` in the output); after a short
    load on training tasks, peer access and the all-reduce under NCCL's settings (``allreduce.json``).
    ``embed_host``: the input embedding in host memory as well (its first start on four ranks); a start that does
    not come up is followed by one with the tower and the embedding on the cards."""
    import sys

    sys.path.insert(0, str(HERE / "src"))
    from arc2 import l4_serving

    options = {"streams": 32, "kv": "int4", "tasks": 48, "wall": wall, "utilization": utilization,
               "expert_parallel": True, "prefix_patches": True, "kv4_guard": True, "ple_pack": True,
               "prefill_slots": 2, "text_only": text_only, "lean_graphs": True, "inventory": True,
               "profile": True, "probe_decode": [8, 16, 24, 32], **({"embed_host": True} if embed_host else {}),
               **({"tower_off": True, "text_only": False} if tower_off else {})}
    l4_serving.study_for(json.loads((BUNDLE / "configs/comparison.json").read_text()), options)  # valid before spend
    options.update({"any_driver": True} if any_driver else {})  # the job's, not the study's
    bound = reserve(run_id, "arc2-l4-profile", PROFILE_TIMEOUT, SERVE_CPU, SERVE_RAM_GIB)
    call = profile.spawn(run_id, options)
    record = {"run_id": run_id, "kind": "arc2-l4-profile", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": PROFILE_TIMEOUT, "gpu": f"L4:{GPUS}",
              "state": "submitted", "options": options, "bundle": BUNDLE.name,
              "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def score_main(run_id: str, arm: str = "xhigh-8k", rates: str = "130,180", streams: int = 32,
               utilization: str = "0.85", wall: int = 5400, tasks: int = 48, prefix_patches: bool = True,
               sticky: bool = True, kv4_guard: bool = True):
    """The bundle's roster trial under one arm on the four L4s (Flash, expert parallel, 4-bit cache), scored at the
    marks of ``rates``; the result carries the play window's tok/s, the prefix-cache hit share, the cards' peak and
    the allocator's retries, and the partial score when the server or the trial ends early. ``kv4_guard`` (on):
    the 4-bit cache's rotation guard, without which 5-14% of L4 replies are garbage (arc2-l4-attention-v1/v3).
    0.85 of each card: at 0.87 with the guard a 490 MiB prefill buffer found no room after 590 s of play and the
    engine died (arc2-l4-score-v4); the guarded probe at 0.85 served."""
    import sys

    sys.path.insert(0, str(HERE / "src"))
    from arc2 import l4_serving

    options = {"streams": streams, "kv": "int4", "tasks": tasks, "wall": wall, "utilization": utilization,
               "expert_parallel": True, "roster": {"arm": arm, "rates": [int(r) for r in rates.split(",")]},
               "prefix_patches": prefix_patches, "sticky": sticky, "kv4_guard": kv4_guard}
    study = l4_serving.study_for(json.loads((BUNDLE / "configs/comparison.json").read_text()), options)
    bound = reserve(run_id, "arc2-l4-score", SCORE_TIMEOUT, SERVE_CPU, SERVE_RAM_GIB)
    call = score.spawn(run_id, options)
    record = {"run_id": run_id, "kind": "arc2-l4-score", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": SCORE_TIMEOUT, "gpu": f"L4:{GPUS}", "state": "submitted",
              "options": options, "trial": {k: study["trials"][0][k] for k in ("arms", "marks", "wall_seconds")},
              "bundle": BUNDLE.name, "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def cache_main(run_id: str, utilization: str = "0.87", streams: int = 32, prefix_patches: bool = True):
    """A short start (Flash, expert parallel, 4-bit cache) that only asks: is turn 2 of a conversation served
    from the prefix cache, right away and after 30 / 90 / 240 s with 20 other conversations generating?"""
    import sys

    sys.path.insert(0, str(HERE / "src"))
    from arc2 import l4_serving

    options = {"streams": streams, "kv": "int4", "tasks": 48, "wall": 420, "utilization": utilization,
               "expert_parallel": True, "prefix_patches": prefix_patches, "cache_probe": True}
    l4_serving.study_for(json.loads((BUNDLE / "configs/comparison.json").read_text()),
                         {k: v for k, v in options.items() if k != "cache_probe"})
    bound = reserve(run_id, "arc2-l4-cache-probe", CACHE_TIMEOUT, SERVE_CPU, SERVE_RAM_GIB)
    call = cache.spawn(run_id, options)
    record = {"run_id": run_id, "kind": "arc2-l4-cache-probe", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": CACHE_TIMEOUT, "gpu": f"L4:{GPUS}", "state": "submitted",
              "options": options, "bundle": BUNDLE.name, "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def garbage_main(run_id: str, expert_parallel: bool = True, utilization: str = "", phases: str = "",
                 site: str = "moe", tokens: int = 200, kv4_guard: bool = False, indexer_fp8: bool = False,
                 ple_pack: bool = False, options: str = "", streams: int = 0, hosts: int = 0):
    """A short start (Flash, 4-bit cache, no prefix-cache patches, ``arc2.l4_finite`` in the workers) under prefill
    load: real first turns and second turns, every reply read for garbage, every prefill forward's expert output
    checked per rank and layer. With expert parallel: the scored runs' server (32 seats, 0.87), 4 streams then 20.
    Without: tensor parallel only (16 seats, 0.90, the pool holds 4 streams). ``phases``: "streams:share,...".
    ``site`` "attention": the check sits on the 4-bit sparse attention instead of the routed experts (query,
    indices and stored scales at a non-finite row; block reuse). ``tokens``: the reply cap; every reply is kept
    whole in ``garbage-replies.jsonl``. ``kv4_guard``: the 4-bit cache's rotation guard on (``ARC_KV4_GUARD``).
    ``options``: JSON over the study's choice (``arc2.l4_serving.l4_study``), e.g. the memory-only settings to
    prove in the same start; ``streams``: the server's seats when not 32 / 16."""
    import sys

    sys.path.insert(0, str(HERE / "src"))
    from arc2 import l4_serving

    plan, extra = phases or ("4:0.3,20:0.7" if expert_parallel else "4:1"), json.loads(options) if options else {}
    options = {"streams": streams or (32 if expert_parallel else 16), "kv": "int4", "tasks": 48, "wall": 420,
               "utilization": utilization or ("0.87" if expert_parallel else "0.90"),
               "expert_parallel": expert_parallel, "prefix_patches": False,
               "finite_check": "attention" if site == "attention" else True, "kv4_guard": kv4_guard,
               **({"indexer_fp8": True} if indexer_fp8 else {}), **({"ple_pack": True} if ple_pack else {}),
               **extra,
               "garbage": {"phases": [[int(a), float(b)] for a, b in (part.split(":") for part in plan.split(","))],
                           "tokens": tokens}}
    l4_serving.study_for(json.loads((BUNDLE / "configs/comparison.json").read_text()),
                         {k: v for k, v in options.items() if k != "garbage"})
    if not TURNS.is_file():
        raise SystemExit(f"no second turns at {TURNS}")
    if not 0 <= hosts <= HOST_TRIES:
        raise SystemExit(f"--hosts is 0..{HOST_TRIES}")
    options.update({"hosts": hosts} if hosts else {})  # further hosts on a refused driver (another_host)
    bound = reserve(run_id, "arc2-l4-garbage", GARBAGE_TIMEOUT + hosts * (HOST_HOLD_SECONDS + 60), SERVE_CPU,
                    SERVE_RAM_GIB)
    call = garbage.spawn(run_id, options)
    record = {"run_id": run_id, "kind": "arc2-l4-garbage", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": GARBAGE_TIMEOUT, "gpu": f"L4:{GPUS}",
              "state": "submitted", "options": options, "bundle": BUNDLE.name,
              "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def depth_main(run_id: str, options: str = "", depths: str = "16000,32000", rows: str = "32,16", hosts: int = 0):
    """The decode step at a play's context depth (``arc2.l4_depth``): the package's server (32 seats, 0.90, the
    prefix-cache patches, the guard) with ``options`` (JSON over that choice, e.g. ``{"dense_int8": true}``), the
    probe's ladder at ``rows``, then the same rows after prompts of each of ``depths`` tokens. One start a call:
    a variant is a second call, and each carries its own short-context rows for the host."""
    import sys

    sys.path.insert(0, str(HERE / "src"))
    from arc2 import l4_serving

    chosen = {"streams": 32, "kv": "int4", "tasks": 48, "wall": 420, "utilization": "0.90", "expert_parallel": True,
              "prefix_patches": True, "kv4_guard": True, "ple_pack": True, "lean_graphs": True,
              **(json.loads(options) if options else {})}
    l4_serving.study_for(json.loads((BUNDLE / "configs/comparison.json").read_text()), chosen)
    if not 0 <= hosts <= HOST_TRIES:
        raise SystemExit(f"--hosts is 0..{HOST_TRIES}")
    chosen.update({"depth": {"depths": depths, "rows": rows}}, **({"hosts": hosts} if hosts else {}))
    # with the profiler: the profile job's limit (a 32K-token level of 48 rows and its trace passed 2,140 s, v26b)
    job, limit = (profile, PROFILE_TIMEOUT) if chosen.get("profile") else (garbage, GARBAGE_TIMEOUT)
    bound = reserve(run_id, "arc2-l4-depth", limit + hosts * (HOST_HOLD_SECONDS + 60), SERVE_CPU, SERVE_RAM_GIB)
    call = job.spawn(run_id, chosen)
    record = {"run_id": run_id, "kind": "arc2-l4-depth", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": limit, "gpu": f"L4:{GPUS}",
              "state": "submitted", "options": chosen, "bundle": BUNDLE.name,
              "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def nccl_main(run_id: str, want: str = "610", hosts: int = HOST_TRIES):
    """NCCL on a host with driver ``want`` (default 610, where the server's first all-reduce fails): the server
    start as it goes there, with NCCL's log on, then ``arc2.l4_nccl`` in the server's environment (every transport
    setting under vLLM's communicator and torch.distributed's; ``nccl.json``, ``nccl-logs/``). A host with another
    driver hands the call on (``--hosts``). No model is loaded on 610 (the workers die first): about 12 minutes."""
    options = {"streams": 32, "kv": "int4", "tasks": 48, "wall": 420, "utilization": "0.90", "expert_parallel": True,
               "prefix_patches": False, "kv4_guard": True, "ple_pack": True, "any_driver": True, "nccl": True,
               "want_driver": want}
    if not 0 <= hosts <= HOST_TRIES:
        raise SystemExit(f"--hosts is 0..{HOST_TRIES}")
    options.update({"hosts": hosts} if hosts else {})
    bound = reserve(run_id, "arc2-l4-nccl", GARBAGE_TIMEOUT + hosts * (HOST_HOLD_SECONDS + 60), SERVE_CPU,
                    SERVE_RAM_GIB)
    call = garbage.spawn(run_id, options)
    record = {"run_id": run_id, "kind": "arc2-l4-nccl", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": GARBAGE_TIMEOUT, "gpu": f"L4:{GPUS}",
              "state": "submitted", "options": options, "bundle": BUNDLE.name,
              "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def package_main(run_id: str, tasks: int = 43, play: int = 7150, seed: int = 0, options: str = "", arm: str = "tool",
                 any_driver: bool = False, hosts: int = 0):
    """The Kaggle package (``ARC2_L4_BUNDLE=dist/arc2-kaggle-v1``) on ``tasks`` public evaluation tasks, each with the
    competition's share of the play: 240 tasks in about 39,900 s of an 11.5 h notebook, so 43 in 7,150 s (under
    two hours of play)."""
    if not (BUNDLE / "src/arc2/kaggle.py").is_file():
        raise SystemExit("ARC2_L4_BUNDLE must name a bundle built with --plan=kaggle")
    extra = {**(json.loads(options) if options else {}), **({"roster": {"arm": arm, "rates": [330]}} if arm else {})}
    chosen = {"tasks": tasks, "play_seconds": play, "seed": seed, **({"options": extra} if extra else {}),
              **({"any_driver": True} if any_driver else {})}
    if not 0 <= hosts <= HOST_TRIES:
        raise SystemExit(f"--hosts is 0..{HOST_TRIES}")
    chosen.update({"hosts": hosts} if hosts else {})  # further hosts the call may ask for on a refused driver
    bound = reserve(run_id, "arc2-l4-package", PACKAGE_TIMEOUT + hosts * (HOST_HOLD_SECONDS + 60), SERVE_CPU,
                    SERVE_RAM_GIB)
    call = package.spawn(run_id, chosen)
    record = {"run_id": run_id, "kind": "arc2-l4-package", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": PACKAGE_TIMEOUT, "gpu": f"L4:{GPUS}", "state": "submitted",
              "options": chosen, "bundle": BUNDLE.name, "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def smoke_main(run_id: str, options_file: str, tasks: int = 48, play: int = 2400, seed: int = 0, clones: int = 1):
    """A kernel's options as the notebook passes them (``options_file``: JSON, roster included) on ``tasks`` public
    evaluation tasks for ``play`` seconds, in at most ``SMOKE_TIMEOUT`` (v38 smoke: server start with the prefetch,
    the probe's slots at the 32K cap, the PLE bound's line, a test kill of the engine and the watchdog's restart)."""
    if not (BUNDLE / "src/arc2/kaggle.py").is_file():
        raise SystemExit("ARC2_L4_BUNDLE must name a bundle built with --plan=kaggle")
    chosen = {"tasks": tasks, **({"clones": clones} if clones > 1 else {}), "play_seconds": play, "seed": seed,
              "options": json.loads(Path(options_file).read_text())}  # clones 2 on 120 tasks: the c240 roster
    bound = reserve(run_id, "arc2-l4-smoke", SMOKE_TIMEOUT, SERVE_CPU, SERVE_RAM_GIB,
                    os.environ.get("ARC2_L4_AUTHORIZATION") or
                    "one short Modal 4xL4 smoke of the exact v38 kernel bundle, "
                    "bound (ARC-AGI-2 arm, run ids arc2-)")
    call = smoke.spawn(run_id, chosen)
    record = {"run_id": run_id, "kind": "arc2-l4-smoke", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": SMOKE_TIMEOUT, "gpu": f"L4:{GPUS}", "state": "submitted",
              "options": chosen, "bundle": BUNDLE.name, "output": f"arc-prolong-runs:/research/{run_id}",
              "live": f"arc-prolong-runs:/validation/{run_id}/arena/live/progress.jsonl"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def full_main(run_id: str, arm: str = "tool", options: str = "", play: int = 39900):
    """The whole notebook on the c240 roster: the 120 public evaluation tasks twice (240 entries, 334 outputs),
    11.5 h in all, about 39,900 s of it play; a dead engine is restarted (``arc2.kaggle``). ``arm``: the bundle's
    arm to play (the tool arm; ``xhigh-8k``: the plain loop). Needs explicit approval (over 4 h)."""
    if not (BUNDLE / "src/arc2/kaggle.py").is_file():
        raise SystemExit("ARC2_L4_BUNDLE must name a bundle built with --plan=kaggle")
    extra = {**(json.loads(options) if options else {}), **({"roster": {"arm": arm, "rates": [330]}} if arm else {})}
    chosen = {"tasks": 120, "clones": 2, "play_seconds": play, "seed": 0, **({"options": extra} if extra else {})}
    bound = reserve(run_id, "arc2-l4-full", FULL_TIMEOUT, FULL_CPU, FULL_RAM_GIB, OWNER_FULL)
    call = full.spawn(run_id, chosen)
    record = {"run_id": run_id, "kind": "arc2-l4-full", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": FULL_TIMEOUT, "gpu": f"L4:{GPUS}", "state": "submitted",
              "cpu": FULL_CPU, "memory_gib": FULL_RAM_GIB, "options": chosen, "bundle": BUNDLE.name,
              "output": f"arc-prolong-runs:/research/{run_id}",
              "live": f"arc-prolong-runs:/validation/{run_id}/arena/live/progress.jsonl"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def roster_main(run_id: str, arm: str = "tool", options: str = "", play: int = 12600):
    """The c240 roster as ``full_main`` plays it, for four hours in all: the first 12,600 s of a full-length play,
    read against a full-length run at equal tokens (tasks started, attempts by number, outputs right)."""
    if not (BUNDLE / "src/arc2/kaggle.py").is_file():
        raise SystemExit("ARC2_L4_BUNDLE must name a bundle built with --plan=kaggle")
    extra = {**(json.loads(options) if options else {}), **({"roster": {"arm": arm, "rates": [330]}} if arm else {})}
    chosen = {"tasks": 120, "clones": 2, "play_seconds": play, "seed": 0, **({"options": extra} if extra else {})}
    bound = reserve(run_id, "arc2-l4-roster", ROSTER_TIMEOUT, FULL_CPU, FULL_RAM_GIB, RULE_ROSTER)
    call = roster.spawn(run_id, chosen)
    record = {"run_id": run_id, "kind": "arc2-l4-roster", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": ROSTER_TIMEOUT, "gpu": f"L4:{GPUS}",
              "state": "submitted", "cpu": FULL_CPU, "memory_gib": FULL_RAM_GIB, "options": chosen,
              "bundle": BUNDLE.name, "output": f"arc-prolong-runs:/research/{run_id}",
              "live": f"arc-prolong-runs:/validation/{run_id}/arena/live/progress.jsonl"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


@app.local_entrypoint()
def pathcheck_main(run_id: str, tasks: int = 4, seed: int = 0, options: str = ""):
    """The package's driver and server start on a CPU host (``pathcheck``): bound about a dollar. ``options``: the
    driver's ``--options`` (JSON), as the GPU run it checks will pass them."""
    if not (BUNDLE / "src/arc2/kaggle.py").is_file():
        raise SystemExit("ARC2_L4_BUNDLE must name a bundle built with --plan=kaggle")
    bound = reserved(run_id, (PATHCHECK_TIMEOUT + 60) * (SERVE_CPU * 0.0000131 + PATHCHECK_RAM_GIB * 0.00000222) * 1.15)
    chosen = {"tasks": tasks, "play_seconds": 120, "seed": seed, **({"options": json.loads(options)} if options else {})}
    call = pathcheck.spawn(run_id, chosen)
    record = {"run_id": run_id, "kind": "arc2-l4-pathcheck", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": PATHCHECK_TIMEOUT, "gpu": None, "state": "submitted",
              "options": chosen, "bundle": BUNDLE.name, "output": f"arc-prolong-runs:/research/{run_id}"}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
