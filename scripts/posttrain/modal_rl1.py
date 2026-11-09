"""Modal launcher for RL step 1 (13 October, docs/rl-plan.md): one B300 runs a ``build_rl1.py`` bundle (the arena
bootstrap: Flash NVFP4 server with LoRA, the ``rl-loop`` trial) in the pinned Axolotl image, whose venv the trial's
trainer service uses.

    ARC_RL1_BUNDLE=<bundle> ARC_VALIDATION_ROOT=. \
        uv run --extra modal modal run --detach scripts/posttrain/modal_rl1.py --run-id rl1-probe-v1
    add --dry-run for the bound and checks without spawning

Timeout = the bundle's notebook seconds + 300. Output ``arc-prolong-runs:/validation/<run id>/`` (the arena output
under ``arena/``, the loop's feed under ``arena/<trial>/rl/``). Graceful stop: put ``stop-request`` in
``validation/<run id>/arena/live/``; the loop stops after its current round.
Receipt ``docs/validation-rl/modal-<run id>.json``; no ledger write (the operator keeps docs/spend.json).
"""
import hashlib
import inspect
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or ".").resolve()
BUNDLE = (Path(os.environ.get("ARC_RL1_BUNDLE", "dist/rl1-probe-v1")).resolve() if modal.is_local()
          else Path("/opt/arc-validation"))
GPU, GPU_RATE = "B300", 0.001972
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
# in the container STUDY is None: the image env carries the launch-time count (the GPU 0 shim needs it)


def study_cards(study):
    """Cards of a study: mg.redesign.cards (one play server and one trainer rank per card), else 1."""
    rl = (study or {}).get("rl_loop") or {}
    red = (rl.get("mg") or {}).get("redesign")
    if isinstance(red, dict) and red.get("cards"):
        return int(red["cards"])
    return 1


CARDS = int(os.environ.get("ARC_RL1_CARDS") or study_cards(STUDY))
if CARDS not in (1, 2, 4, 8):
    raise ValueError(f"ARC_RL1_CARDS {CARDS}: 1, 2, 4 or 8 B300 in one container")
if CARDS > 1:  # stage H (26 Oct): N cards in one container (gpu B300:N), a server per card, a torchrun rank per card
    GPU, GPU_RATE = f"{GPU}:{CARDS}", CARDS * GPU_RATE
# CPU: the 16-core exception (RL_CPU) stays the reservation; N > 2 cards burst up to 8 cores per card (8 servers'
# API/engine processes, 8 trainer ranks and the harness on 16 cores would starve)
RUN_CPU = RL_CPU if CARDS <= 2 else (RL_CPU, 8.0 * CARDS)
app = modal.App("arc-rl1", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
# per-run volumes: launch.py sets ARC_RL1_VOLUME to the record's arc-run-<run> (a v2 volume at /runs) and
# ARC_RL1_INPUT_VOLUMES to the volumes the config reads at /in/<vol>/ (read-only); caches then live on the shared v2
# arc-cache at /cache. Unset: everything on arc-prolong-runs at /runs, as before.
SHARED_VOLUME = "arc-prolong-runs"
RUN_VOLUME = os.environ.get("ARC_RL1_VOLUME") or SHARED_VOLUME
PER_RUN = RUN_VOLUME != SHARED_VOLUME
INPUT_VOLUMES = [v for v in (os.environ.get("ARC_RL1_INPUT_VOLUMES") or "").split(",") if v]
CACHE_ROOT = "/cache" if PER_RUN else "/runs"
runs = modal.Volume.from_name(RUN_VOLUME, **({"create_if_missing": True, "version": 2} if PER_RUN else {}))
cache = modal.Volume.from_name("arc-cache", create_if_missing=True, version=2) if PER_RUN else None
VOLUMES = {"/assets": assets, "/runs": runs, **({"/cache": cache} if PER_RUN else {}),
           **{f"/in/{v}": modal.Volume.from_name(v).read_only() for v in INPUT_VOLUMES}}


def commit_volumes():
    """The run's volume, and on a per-run volume the cache volume (compile/trainer/runtime/boot caches)."""
    runs.commit()
    if cache is not None:
        cache.commit()


def vrel(path):
    """A volume path relative to its volume's root: /runs/x or /in/<vol>/x -> x."""
    parts = Path(path).parts
    return Path(*parts[3:]) if parts[1:2] == ("in",) else Path(path).relative_to("/runs")


def vabs(d):
    """A volume path as the container sees it: an absolute one (/in/<vol>/..., /runs/...) as given, else under /runs."""
    return d if d.startswith("/") else f"/runs/{d.strip('/')}"
PREBUILT_VENV = "/opt/agent-venv"  # stage H: the bundle's hash-locked agent venv, built once per wheel set (image layer)
VENV_BUILD = (  # a failed build leaves no venv (the bootstrap installs as before), never a failed image
    # the interpreter the function runs (add_python's uv build; pf N4 26 Oct: a venv on /usr/local/bin was refused)
    "( PY=$(ls /root/.local/share/uv/python/cpython-3.12*-linux-x86_64-gnu/bin/python3.12 2>/dev/null | head -1); "
    "PY=${PY:-/usr/local/bin/python}; "
    "$PY -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' && "
    "$PY -m venv --without-pip /opt/agent-venv && "
    "PYTHONPATH=/opt/arc-req/wheels/pip-25.2-py3-none-any.whl PIP_CONFIG_FILE=/dev/null PIP_NO_INDEX=1 "
    "/opt/agent-venv/bin/python -m pip install --no-index --require-hashes --find-links /opt/arc-req/wheels "
    "-r /opt/arc-req/requirements.txt && "
    "PYTHONPATH=/opt/arc-req/wheels/pip-25.2-py3-none-any.whl /opt/agent-venv/bin/python -m pip check && "
    "sha256sum /opt/arc-req/requirements.txt | cut -d' ' -f1 > /opt/agent-venv/.arc-requirements.sha256 ) || "
    "( rm -rf /opt/agent-venv; echo 'prebuilt agent venv skipped' )")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("rclone", "libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         # the loop's stall watchdog dumps every python process (ARC_RL1_STALL_DUMP); the venv's bin is on PATH
         .run_commands("uv pip install --python /workspace/axolotl-venv/bin/python 'py-spy==0.4.0'"))
for command in RL_PIP:
    image = image.run_commands("uv pip install --python /workspace/axolotl-venv/bin/python "
                               + " ".join(f"'{item}'" for item in command.split()))
if (BUNDLE / "requirements.txt").is_file() and (BUNDLE / "wheels").is_dir():  # cached while the wheel set holds
    image = (image.add_local_file(str(BUNDLE / "requirements.txt"), "/opt/arc-req/requirements.txt", copy=True)
             .add_local_dir(str(BUNDLE / "wheels"), "/opt/arc-req/wheels", copy=True).run_commands(VENV_BUILD))
# the launch-time env after the cached layers (stage H: before them, a new timeout or card count rebuilt every pip layer)
image = image.env({"ARC_RL1_TIMEOUT": str(TIMEOUT), "ARC_RL1_CARDS": str(CARDS), "ARC_RL1_GPU": GPU.split(":")[0],
                   "PYTHONUNBUFFERED": "1",  # the container's volume objects match the mounts (per-run only)
                   **({"ARC_RL1_VOLUME": RUN_VOLUME, "ARC_RL1_INPUT_VOLUMES": ",".join(INPUT_VOLUMES)} if PER_RUN else {})})
image = image.add_local_dir(str(BUNDLE), "/opt/arc-validation", copy=True,
                            ignore=["dataset-upload/**", "*.ipynb", "source.zip"]).add_local_file(
    __file__, "/root/modal_rl1.py", copy=True)


def host_cpu():
    try:
        line = next(l for l in open("/proc/cpuinfo") if l.startswith("model name"))
        return {"model": line.split(":", 1)[1].strip(), "count": os.cpu_count()}
    except Exception:  # a record field only
        return None


HOST_CHECK = "/tmp/rl1-host-check.json"  # rl_loop emits it as the feed row host_check
HOST_CHECK_SLOW_GIB_S = 0.5  # pinned card->host copy (T3-v3 park 1.0 GiB/s, T4-v1 0.22: setup 211 vs 487 s)
# a park is mostly pinned allocation (T4r-v2 host: copy 52 GiB/s, allocation 3.3 GiB/s; its 31 GiB cold park 29.6 s)
HOST_CHECK_SLOW_PIN_GIB_S = 1.0
HOST_CHECK_CODE = r"""
import json, os, socket, sys, time
row = {"host": socket.gethostname(), "modal": {k: v for k, v in os.environ.items() if k in
       ("MODAL_TASK_ID", "MODAL_REGION", "MODAL_CLOUD_PROVIDER", "MODAL_IMAGE_ID")}}
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no cuda")
    gib = float(sys.argv[1])
    n = int(gib * 2**30)
    t = time.time(); host = torch.empty(n, dtype=torch.uint8, pin_memory=True); row["pin_alloc_s"] = round(time.time() - t, 2)
    card = torch.ones(n, dtype=torch.uint8, device="cuda"); torch.cuda.synchronize()
    d2h, h2d = [], []
    for _ in range(3):
        t = time.time(); host.copy_(card); torch.cuda.synchronize(); d2h.append(gib / (time.time() - t))
        t = time.time(); card.copy_(host); torch.cuda.synchronize(); h2d.append(gib / (time.time() - t))
    row.update(gib=gib, d2h_gib_s=round(min(d2h), 2), d2h_best_gib_s=round(max(d2h), 2), h2d_gib_s=round(min(h2d), 2),
               gpu=torch.cuda.get_device_name(0))
    del card, host
except Exception as exc:
    row["skipped"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(row))
"""


def memory_limits(meminfo_path="/proc/meminfo", cgroup="/sys/fs/cgroup"):
    """GiB: MemTotal and MemAvailable (/proc/meminfo) and the container cgroup's memory.max and memory.current (v1's
    limit/usage files otherwise); None where unreadable or unlimited (as flash_qlora_job.host_memory)."""
    info = meminfo(meminfo_path)
    out = {"mem_total_gib": None if info.get("MemTotal") is None else round(info["MemTotal"], 1),
           "mem_available_gib": None if info.get("MemAvailable") is None else round(info["MemAvailable"], 1),
           "cgroup_max_gib": None, "cgroup_current_gib": None}
    for limit, usage in (("memory.max", "memory.current"), ("memory/memory.limit_in_bytes",
                                                            "memory/memory.usage_in_bytes")):
        try:
            top = Path(cgroup, limit).read_text().strip()
            used = int(Path(cgroup, usage).read_text().strip())
        except (OSError, ValueError):
            continue
        out["cgroup_current_gib"] = round(used / 2 ** 30, 1)
        if top.isdigit() and int(top) < 2 ** 60:
            out["cgroup_max_gib"] = round(int(top) / 2 ** 30, 1)
        break
    return out


def proc_label(cmdline, comm=""):
    """A short group name for a process: the vLLM title, else the program and its script or module."""
    args = [a for a in cmdline.split("\0") if a]
    if not args or comm.startswith("VLLM") or args[0].startswith("VLLM"):
        return comm or (args[0] if args else "?")
    prog = os.path.basename(args[0])
    if prog.startswith("python") and len(args) > 1:
        rest = [a for a in args[1:] if not a.startswith("-") or a in ("-m", "-c")]
        if rest[:1] == ["-m"] and len(rest) > 1:
            return f"{prog} -m {rest[1]}"
        if rest[:1] == ["-c"]:
            return f"{prog} -c {rest[1][:40]}" if len(rest) > 1 else f"{prog} -c"
        return f"{prog} {os.path.basename(rest[0])}" if rest else prog
    return prog


def procmem_row(proc="/proc", top=12):
    """Per-process RssAnon (GiB; VmRSS where the kernel has no RssAnon: gVisor, real-v6 27 Oct) summed by label, the
    largest processes, and the host's AnonPages / Shmem: the host plan's in-run read."""
    groups, procs, basis = {}, [], set()
    for entry in Path(proc).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            rows = {l.split(":")[0]: l.split()[1] for l in status.splitlines() if l.startswith(("RssAnon:", "VmRSS:"))}
            key = "RssAnon" if "RssAnon" in rows else "VmRSS"
            anon = int(rows.get(key, 0)) / 2**20
            basis.add(key)
            label = proc_label((entry / "cmdline").read_text(errors="replace"),
                               (entry / "comm").read_text(errors="replace").strip())
        except (OSError, ValueError, StopIteration):
            continue
        n, total = groups.get(label, (0, 0.0))
        groups[label] = (n + 1, total + anon)
        procs.append((anon, int(entry.name), label))
    procs.sort(reverse=True)
    host = {}
    try:
        info = {l.split(":")[0]: l.split()[1] for l in Path(proc, "meminfo").read_text().splitlines() if len(l.split()) >= 2}
        host = {k.lower() + "_gib": round(int(info[k]) / 2**20, 1) for k in ("AnonPages", "Shmem", "Cached") if k in info}
    except (OSError, ValueError):
        pass
    return {"t": round(time.time(), 1), "anon_gib": round(sum(a for a, _, _ in procs), 2), "basis": sorted(basis), **host,
            "groups": {k: [n, round(a, 2)] for k, (n, a) in sorted(groups.items(), key=lambda kv: -kv[1][1])
                       if a >= 0.05},
            "top": [[pid, label, round(a, 2)] for a, pid, label in procs[:top]]}


def host_check(identity, gib=4.0, path=HOST_CHECK, python=None):
    """Boot host check (T4-v1's slow host): a timed ``gib`` pinned card->host copy (worst of 3,
    plus host->card and the pin allocation) and the host's ids (hostname, Modal task, GPU UUID and PCI bus), written to
    ``path`` and the identity. ``slow``: the copy below HOST_CHECK_SLOW_GIB_S or the pinned allocation below HOST_CHECK_SLOW_PIN_GIB_S; the run
    continues either way."""
    import subprocess

    began = time.time()
    try:
        out = subprocess.run([python or sys.executable, "-c", HOST_CHECK_CODE, str(gib)], capture_output=True, text=True,
                             timeout=300)
        row = json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else {"error": out.stderr[-400:]}
    except Exception as exc:  # a record field only
        row = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        row["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except Exception:
        row["gpus"] = None
    if row.get("d2h_gib_s") is not None:
        row["pin_alloc_gib_s"] = round(row["gib"] / max(row["pin_alloc_s"], 1e-3), 2)
        row["slow"] = row["d2h_gib_s"] < HOST_CHECK_SLOW_GIB_S or row["pin_alloc_gib_s"] < HOST_CHECK_SLOW_PIN_GIB_S
    row["memory"] = memory_limits()  # H3 (26 Oct): H2 used 844 GiB of an unrecorded total; the preload guard's input
    row["seconds"] = round(time.time() - began, 1)
    identity["host_check"] = row
    Path(path).write_text(json.dumps(row))
    print(json.dumps({"event": "host_check", **row}), flush=True)
    return row


def bound(seconds):
    return (seconds + 60) * (GPU_RATE + BILL_CPU * 1.31e-05 + BILL_MEMORY_GIB * 2.22e-06) * 1.15


LAST_COMMIT = {}  # the committer's last commit_volumes() seconds (live_sample records it: a lagging feed vs a stall)
VLLM_KEYS = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "generation_tokens_total",
             "num_preemptions_total", "prefix_cache_queries_total", "prefix_cache_hits_total")  # G-long: prefix hit


def server_ports(cards=None):
    """The play servers' ports: :1234 on card 0, rl_loop.SecondServer's 1235 + i on card i."""
    return [1234] + [1235 + i for i in range(1, int(cards or CARDS))]


def vllm_metrics(text):
    """One server's /metrics: VLLM_KEYS summed over its label sets (one per engine)."""
    out = {}
    for line in text.splitlines():
        if line and line[0] != "#":
            name, _, value = line.rpartition(" ")
            key = name.split("{", 1)[0]
            if key.startswith("vllm:") and key[5:] in VLLM_KEYS:
                out[key[5:]] = out.get(key[5:], 0.0) + float(value)
    return out


def vllm_totals(servers):
    """Across servers: counts and counters summed, KV usage the mean of the servers reporting it."""
    servers = list(servers)
    out = {k: sum(s[k] for s in servers if k in s) for k in VLLM_KEYS if any(k in s for s in servers)}
    kv = [s["kv_cache_usage_perc"] for s in servers if "kv_cache_usage_perc" in s]
    if kv:
        out["kv_cache_usage_perc"] = sum(kv) / len(kv)
    return out


def live_sample(trial, folder, started, ports=None):
    """Live watcher feed (every run readable from its first minute), committed each minute:
    ``sampler.jsonl`` += nvidia-smi util / memory per card and the vLLM servers' running / waiting / KV / generated
    tokens (``vllm``: totals over the servers; ``vllm_servers``: per port); ``progress.json``: per gate folder the
    attempts started, finished, cleared and their completion tokens."""
    import subprocess
    import urllib.request

    folder.mkdir(parents=True, exist_ok=True)
    began = time.time()  # epoch: lines rows up with util/gpulog.txt; commit_s: the previous pass's volume commit
    row = {"t": round(began - started), "epoch": round(began, 1), "commit_s": LAST_COMMIT.get("seconds")}
    try:
        out = subprocess.run([os.environ.get("ARC_REAL_NVIDIA_SMI", "nvidia-smi"), "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        cards = [[float(v) for v in line.split(",")] for line in out.splitlines() if line.strip()]
        row.update(util=[c[0] for c in cards], mem_gib=[round(c[1] / 1024, 1) for c in cards])
    except Exception:  # noqa: BLE001
        pass
    servers = {}
    for port in ports or server_ports():  # H2 (26 Oct): every card's server, not :1234 only
        try:
            text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
            servers[str(port)] = vllm_metrics(text)
        except Exception:  # noqa: BLE001 - no server (yet), or asleep / restarting
            pass
    if servers:
        row["vllm"] = vllm_totals(servers.values())
        row["vllm_servers"] = servers
    row["sample_s"] = round(time.time() - began, 1)
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


READ_KEYS = ("ARC_RL1_EVAL_ADAPTERS", "ARC_RL1_RESUME_ADAPTER", "ARC_RESTART_ACK")  # launch-time read targets


def apply_overrides(overrides):
    for key, value in json.loads(overrides or "{}").items():
        if key not in READ_KEYS:
            raise ValueError(f"override {key} not allowed")
        os.environ[key] = value if isinstance(value, str) else json.dumps(value)


@app.function(image=image, gpu=GPU, cpu=RUN_CPU, timeout=TIMEOUT, retries=0, max_containers=1,
              scaledown_window=2, volumes=VOLUMES)
def run(run_id: str, resume: str = "", stall: str = "", overrides: str = "", stub: str = ""):
    apply_overrides(overrides)
    if stub:  # ARC_RL1_STUB_SERVING: the replay stub serves, the real trainer trains on this card
        os.environ["ARC_PREFLIGHT_STUB"] = stub
        os.environ["ARC_RL1_TRIAL"] = json.loads(stub).get("trial") or "rl-loop"
    stall_env(stall)
    return execute(run_id, f"validation/{run_id}", resume)


PREFLIGHT_MEMORY_GIB = 96  # bound only
# CPU preflight legs end themselves; Modal kills one at this crash ceiling (default 45 min)
PREFLIGHT_TIMEOUT = 2700
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


@app.function(image=image, timeout=PREFLIGHT_TIMEOUT, retries=0, max_containers=1, cpu=PREFLIGHT_CPU,
              scaledown_window=2, volumes=VOLUMES)
def preflight(run_id: str, spec: str, resume: str = "", tag: str = "", overrides: str = ""):
    apply_overrides(overrides)
    """Launch gate: the exact bundle and arguments on a CPU container, a replay stub in place of the
    vLLM server (``arena.replay_stub``); output ``validation/preflight/<run id>``."""
    os.environ["ARC_PREFLIGHT_STUB"] = spec
    os.environ["ARC_RL1_TRIAL"] = json.loads(spec).get("trial") or "rl-loop"
    if json.loads(spec).get("patch_check"):
        os.environ["ARC_PATCH_CHECK_ONLY"] = "1"
    if json.loads(spec).get("dry_boot"):  # through the engine start (arena.dry_boot)
        os.environ["ARC_DRY_BOOT_ONLY"] = "1"
        if json.loads(spec).get("dry_boot_gpu"):  # the stand-in card (arena.dry_boot GPU_ENV): "b300" for the RL card
            os.environ["ARC_DRY_BOOT_GPU"] = json.loads(spec)["dry_boot_gpu"]
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


def boot_cache_in(study):
    """A trial serves with ARC_BOOT_CACHE_IN (its server_env): the legacy compile-cache archive is not restored."""
    return any((t.get("server_env") or {}).get("ARC_BOOT_CACHE_IN") for t in (study or {}).get("trials") or [])


def compile_restore(key, identity, archives=CACHE_ROOT + "/compile-cache", root="/"):
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


def parallel_ready(launcher_log, server_log):
    """When server 1's copies can start (parallel boot): a restored boot cache (nothing left to
    compile) at once; else once server 1 saved its compiled graphs (backbone and draft head: two "torch.compile took"
    lines), so the copies load them instead of compiling into the same cache roots. Otherwise the loop boots them."""
    if '"event": "boot_cache_restore", "found": true' in launcher_log:  # a carried older key still compiles
        return "cache_restored"
    if server_log.count("torch.compile took") >= 2:
        return "compiled"
    return None


def server_parallel(stop, source, output, serving=Path("/tmp/arena/serving")):
    """Thread: start the redesign's card 1..N-1 servers beside server 1's boot (arena.server_start). Never ends the
    run: whatever it does not start, rl_seq's SecondServer.boot starts itself (start.lock: one spawn per card)."""
    import subprocess

    began = time.time()
    try:
        while True:
            ident = serving / "vllm-server-identity.json"  # server 1 running: its argv and environment readable
            if ident.is_file():
                launcher = (output / "bootstrap.log").read_text(errors="replace") if (output / "bootstrap.log").is_file() else ""
                log = serving / "vllm-openai-server.log"
                why = parallel_ready(launcher, log.read_text(errors="replace") if log.is_file() else "")
                if why or '"phase": "ready"' in ident.read_text(errors="replace"):
                    break
            if stop.wait(2):
                return
        if not why:  # server 1 ready first: the loop is about to boot the copies itself
            return
        env = {**os.environ, "PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run(["/tmp/arena/agent-venv/bin/python", "-m", "arena.server_start",
                            str(source / "configs" / "comparison.json"), str(serving)], env=env, capture_output=True,
                           text=True, timeout=600)
        tail = (r.stdout.strip().splitlines() or [""])[-1] or r.stderr[-400:]
        print(json.dumps({"event": "server_parallel", "why": why, "after_s": round(time.time() - began, 1),
                          "result": tail[:600]}), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"event": "server_parallel_error", "error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)


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
    archive = Path(CACHE_ROOT) / "compile-cache" / f"{key}.tar.gz"
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
        commit_volumes()
        print(json.dumps({"event": "compile_cache_saved", "key": key, "bytes": archive.stat().st_size}), flush=True)
    except Exception as exc:  # the cache never ends the run
        print(f"compile cache save failed: {exc}", flush=True)


MODEL_ROOT = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"


def boot_settings(source):
    """MFU-2 boot switches (bundle configs/boot.json; absent = all off)."""
    path = Path(source) / "configs" / "boot.json"
    return json.loads(path.read_text() or "{}") if path.is_file() else {}


WEIGHTS_DONE = threading.Event()  # prefetch_weights finished, or none configured (prefetch_seq waits)


def read_ranges(path, threads=8, block=16 << 20, cap=None):
    """``path`` read into the page cache by ``threads`` positional readers (one 9P stream is ~0.8 GiB/s); ``cap``
    bytes at most (CPU legs)."""
    from concurrent.futures import ThreadPoolExecutor

    size = Path(path).stat().st_size if cap is None else min(cap, Path(path).stat().st_size)
    fd = os.open(path, os.O_RDONLY)
    try:
        def part(start):
            done = 0
            while done < block and start + done < size:
                got = len(os.pread(fd, min(4 << 20, block - done), start + done))
                if not got:
                    break
                done += got
            return done

        with ThreadPoolExecutor(threads) as pool:
            return sum(pool.map(part, range(0, size, block)))
    finally:
        os.close(fd)


def prefetch_seq(root, identity, threads=8, cap=None):
    """T4r-v2 (26 Oct, resume setup 345 s vs 177 s fresh): the resumed leg's checkpoint (master + optimizer, ~31 GiB on
    Flash; load_checkpoint read it from the volume in 52 s), seq_state and harvest metadata read into the page cache
    beside the server boot, after the weights' own prefetch (one volume). Reads only; the loop opens the same paths."""
    began = time.time()
    WEIGHTS_DONE.wait(600)  # the weights first (both loads need them sooner); never longer than 10 min
    waited = round(time.time() - began, 1)
    state = json.loads((Path(root) / "seq_state.json").read_text())
    ckpt = Path(state["checkpoint"])
    if not ckpt.is_dir() and "state" in ckpt.parts:  # as rl_loop.v7_found maps the leg's own state/sNNNN
        ckpt = Path(root) / Path(*ckpt.parts[ckpt.parts.index("state"):])
    n, files = 0, 0
    for f in sorted(ckpt.glob("*")) if ckpt.is_dir() else []:
        if f.is_file():
            n += read_ranges(f, threads, cap=cap)
            files += 1
    for f in (Path(root) / "harvest").rglob("*.json") if (Path(root) / "harvest").is_dir() else []:
        n += len(f.read_bytes())
        files += 1
    identity["seq_prefetch"] = {"checkpoint": str(ckpt), "files": files, "gib": round(n / 2**30, 2),
                                "waited_s": waited, "seconds": round(time.time() - began - waited, 1)}
    return n


def prefetch_pending(roots, identity, block=16 << 20):
    """Start-up (pfP2 profile, 22 Oct): the loop's pending-pack extraction read its round tar from the volume (~16% of
    the ~3-min setup). The resumed root's packs from its state round on are read into the page cache at container
    start, beside the server boot. Reads only."""
    began, n = time.time(), 0
    try:
        seq = next((Path(r) for r in reversed(roots) if (Path(r) / "seq_state.json").is_file()), None)
        if seq is not None:  # the redesign (rl_seq): its checkpoint, state and harvest metadata
            n = prefetch_seq(seq, identity, cap=(1 << 30) if os.environ.get("ARC_PREFLIGHT_STUB") else None)
            return
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
    WEIGHTS_DONE.set()
    print(json.dumps({"event": "weights_prefetched", **row}), flush=True)
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
        commit_volumes()
        print(json.dumps({"event": "runtime_cache_build", **{k: v for k, v in record.items()
                                                              if k not in ("built_tree", "restored_tree")}}), flush=True)
    except Exception as exc:  # the cache never ends the run
        print(f"runtime cache build failed: {type(exc).__name__}: {exc}", flush=True)


def meminfo(path="/proc/meminfo"):
    """{field: GiB} of /proc/meminfo (MemTotal, MemAvailable, ...); {} where unreadable."""
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                out[parts[0].rstrip(":")] = int(parts[1]) / 2**20
    except OSError:
        pass
    return out


# Host memory per play server, GiB (T3 rl1-loop-seq-t3-v1-attempt2, 1 B300, 26 Oct): PLE host table 51,200,245,760 B
# (ARC_SK_HOST_TABLE), routed-experts CPU buffer 18.88 GB (RoutedExpertsManager), weights offloaded by a level-1 sleep
# 57.22 GiB ("Model loading took"), transient pinned load chunks 29.1 GB peak (ARC_PINNED_LOAD host peak); the container's
# anon memory with the server asleep was 129.9 GiB. A keep-KV sleep (ARC_SLEEP_KEEP_KV) adds the KV pool (141 GiB at
# --kv-cache-memory-bytes 151397597184). Trainer rank: TRAINER_HOST_GIB is a placeholder until a GPU run measures it.
SERVER_HOST_GIB = {"ple": 47.7, "routed": 17.6, "sleep_weights": 57.2, "load_transient": 27.1}
TRAINER_HOST_GIB = 40.0


def host_plan(cards, info, boot=None, keep_kv_gib=0.0):
    """Projected host memory of N play servers asleep (+ one trainer rank per card) against MemTotal; the plan the
    launcher records (identity host_memory, event host_plan). ``boot``: configs/boot.json "host" overrides
    (server_gib, trainer_gib, shared_ple)."""
    boot = boot or {}
    per = dict(SERVER_HOST_GIB)
    server = float(boot.get("server_gib") or (per["ple"] + per["routed"] + per["sleep_weights"] + keep_kv_gib))
    trainer = float(boot.get("trainer_gib") or TRAINER_HOST_GIB)
    shared = bool(boot.get("shared_ple"))  # one PLE table for all servers (not built yet: the N = 8 lever)
    need = cards * server + cards * trainer - (cards - 1) * per["ple"] * shared
    boot_peak = cards * (per["ple"] + per["routed"] + per["load_transient"])  # every server loading at once
    total = info.get("MemTotal")
    row = {"cards": cards, "server_gib": round(server, 1), "trainer_gib": trainer, "need_gib": round(need, 1),
           "boot_peak_gib": round(boot_peak, 1), "total_gib": None if total is None else round(total, 1),
           "available_gib": None if info.get("MemAvailable") is None else round(info["MemAvailable"], 1)}
    if total:
        row["ok"] = need <= 0.9 * total
        row["boot_ok"] = boot_peak <= 0.9 * total
    return row


TRAINER_CACHE, TRAINER_ARCHIVES = Path("/tmp/rl1-compile-cache"), Path(CACHE_ROOT) / "trainer-cache"


def files_under(root):
    return [f for f in Path(root).rglob("*") if f.is_file() and not f.is_symlink()] if Path(root).is_dir() else []


def trainer_cache_restore(key, identity, root=TRAINER_CACHE, archives=TRAINER_ARCHIVES):
    """Stage H (T3 v3: the first accumulate after every trainer start took 126 s at util 17.5%, ~100 s of it first-call
    compile / autotune): the trainer's Inductor and Triton caches (rl_loop.trainer_command under ARC_TRAINER_CACHE)
    from ``<archives>/<key>.tar``. Both caches are content-keyed (graph, source, device); the key (the trainer key and
    card) only picks the archive."""
    import tarfile

    root.mkdir(parents=True, exist_ok=True)
    archive = archives / f"{key}.tar"
    row = {"key": key, "restored": False}
    if not archive.is_file() and archives.is_dir():  # 26 Oct (H4 cold accumulate 209 s): a new trainer key (any trainer
        # source change) missed every time; the caches inside are content-keyed, so the newest archive carries over
        older = sorted((a for a in archives.glob("*.tar") if a.is_file()), key=lambda a: a.stat().st_mtime)
        if older:
            archive, row["carried"] = older[-1], older[-1].stem
    if archive.is_file():
        began = time.time()
        try:
            with tarfile.open(archive) as tar:
                members = [m for m in tar.getmembers() if (m.isfile() or m.isdir()) and not m.name.startswith(("/", ".."))
                           and ".." not in Path(m.name).parts]
                tar.extractall(root, members=members, filter="data")
            row.update(restored=True, files=len(files_under(root)), seconds=round(time.time() - began, 1))
        except Exception as exc:  # a broken archive: a cold trainer compile
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    identity["trainer_cache"] = row
    return row


def first_step_seen(feed):
    try:
        return any('"kind": "step"' in line for line in Path(feed).read_text(errors="replace").splitlines())
    except OSError:
        return False


def trainer_cache_save(stop, key, feed, root=TRAINER_CACHE, archives=TRAINER_ARCHIVES, poll=30, commit=None):
    """Thread: after the run's first trainer step (the first accumulate's compiles and autotunes exist), pack the
    trainer cache to ``<archives>/<key>.tar`` when it holds more files than the archive there (rename into place)."""
    import tarfile

    while not first_step_seen(feed):
        if stop.wait(poll):
            return None
    try:
        files = files_under(root)
        archive = archives / f"{key}.tar"
        if archive.is_file():
            with tarfile.open(archive) as tar:
                if sum(1 for m in tar.getmembers() if m.isfile()) >= len(files):
                    return None
        archives.mkdir(parents=True, exist_ok=True)
        tmp = archive.with_name(f"{archive.name}.{os.getpid()}.tmp")
        with tarfile.open(tmp, "w") as tar:
            for f in files:
                tar.add(f, arcname=str(f.relative_to(root)), recursive=False)
        os.replace(tmp, archive)
        (commit or commit_volumes)()
        row = {"event": "trainer_cache_saved", "key": key, "files": len(files), "bytes": archive.stat().st_size}
        print(json.dumps(row), flush=True)
        return row
    except Exception as exc:  # the cache never ends the run
        print(json.dumps({"event": "trainer_cache_save_error", "error": f"{type(exc).__name__}: {exc}"[:300]}),
              flush=True)
        return None


LOOP_STATES = ("seq_state.json", "seq_pending.json", "seq_play.json", "mg_state.json", "v7_state.json")  # rl_seq first (C-S4 28 Oct)


def self_kill(stop, live, mirror, poll=5.0, exit=os._exit, after_s=600.0):
    """NK attempt 0 (ARC_RL1_PREFLIGHT_SELF_KILL=1 from launch.py nk): at the first
    checkpoint_state past step 0 in the live feed the container dies as a preemption kills it: the feed mirrored and
    the volume committed, then os._exit (no finally: no identity, no pack). A Modal cancel can run the finally
    instead (NKc: a1 found a finished attempt and started fresh)."""
    import shutil

    seen_at = None
    while not stop.wait(poll):
        try:
            lines = (live / "feed.jsonl").read_text().splitlines()
        except OSError:
            continue
        steps, done_after, saved, deep = [], 0, 0, set()
        for line in lines:
            if '"seq_group"' in line and max(steps, default=0) >= 1:  # a group on a harvested start, after the
                try:  # checkpoint: in flight at the kill, a1's resume returns its start first (rl_backlog resumed)
                    row = json.loads(line)
                    if row.get("start") and row.get("start") != row.get("set"):
                        deep.add(row.get("gid"))
                except (ValueError, TypeError, AttributeError):
                    pass
            if '"group_done"' in line and deep:
                try:
                    deep.discard(json.loads(line).get("gid"))
                except (ValueError, TypeError, AttributeError):
                    pass
            if '"seq_phase"' in line:  # harvested starts so far (the judge's harvest_saved): a1's backlog draws them
                try:
                    saved = int(((json.loads(line).get("stats") or {}).get("saved")) or 0)
                except (ValueError, TypeError, AttributeError):
                    pass
            if '"checkpoint_state"' in line:
                try:
                    steps.append(int(json.loads(line).get("step") or 0))
                except (ValueError, TypeError, AttributeError):
                    continue
                done_after = 0
            elif '"group_done"' in line and max(steps, default=0) >= 1:
                done_after += 1
        if max(steps, default=0) >= 1 and seen_at is None:
            seen_at = time.time()
        # v17NKr 30 Oct: a kill right at the checkpoint (2 s into the next play, its groups already pending) left a1
        # no played tries to restore; wait for a group finished after it (play saved mid-phase), at most after_s
        # v17 NKt 30 Oct: a1 restored no start (inconclusive); also wait for a harvested start to exist
        # (a6, a7 NKt2, a10 NKv: 0 restored starts): the kill waits for a group on a harvested
        # start in flight, so a1 must draw it first; after_s is only a ceiling (the judge then fails the leg)
        if seen_at is not None and ((done_after >= 1 and saved > 0 and deep) or time.time() - seen_at >= after_s):
            mirror.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(live / "feed.jsonl", mirror / "feed.jsonl")
            commit_volumes()
            print(json.dumps({"event": "self_kill", "checkpoint_step": max(steps)}), flush=True)
            exit(137)
            return


def lost_state(lost):
    """A dead earlier attempt (no modal-identity.json) that left a loop state to resume (rl_seq: seq_state /
    seq_pending; older loops: mg_state / v7_state / partial)."""
    return lost is not None and not (lost / "modal-identity.json").is_file() and (
        any((lost / name).is_file() for name in LOOP_STATES) or (lost / "partial").is_dir())


def dead_step(folder):
    """The step (rl_seq, v7) or round (mg) of a dead attempt's newest loop state: RestartGuard's checkpoint past step
    0 (a seq_pending without seq_state is step 0's update: no checkpoint)."""
    for name, key in (("seq_state.json", "step"), ("v7_state.json", "step"), ("mg_state.json", "round")):
        try:
            return int(json.loads((Path(folder) / name).read_text()).get(key) or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return 0


def lost_chain(output):
    """The dead attempts of this run folder (``<name>-lost-<epoch>``, oldest first) that left a loop state."""
    output = Path(output)
    found = [p for p in output.parent.glob(f"{output.name}-lost-*") if p.is_dir() and lost_state(p)]
    return sorted(found, key=lambda p: int(p.name.rsplit("-", 1)[1]) if p.name.rsplit("-", 1)[1].isdigit() else 0)


def chain_step(output):
    """RestartGuard's checkpoint step: the dead attempt's own, else the newest of its chain. v17 a10, 30 Oct: a leg
    resumed from a -lost- root and preempted before its first checkpoint holds no state of its own."""
    return dead_step(output) or max((dead_step(p) for p in lost_chain(output)), default=0)


def pending_step1(folder):
    """The dead attempt saved step 0's groups (seq_pending.json at the switch, or seq_play.json mid-play, of step 1
    with groups; no seq_state.json): resumable from them without a checkpoint."""
    if (Path(folder) / "seq_state.json").is_file():
        return False
    for name in ("seq_pending.json", "seq_play.json"):
        try:
            saved = json.loads((Path(folder) / name).read_text())
        except (OSError, ValueError):
            continue
        if int(saved.get("step", -1)) == 1 and saved.get("groups"):
            return True
    return False


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

    # a CPU preflight leg resubmitted by hand is a fresh leg (no training state; its old output goes to -lost-): its
    # new call acks the earlier leg's stop itself; a Modal auto-restart of the leg (same call id) stays blocked
    ack = os.environ.get("ARC_RESTART_ACK") or (PREFLIGHT_LEG_ACK if str(rel).startswith("validation/preflight/") else None)
    return RestartGuard(root, Path(rel).name, policy="resume", commit=commit_volumes, ack=ack)


PREFLIGHT_LEG_ACK = "CPU preflight leg resubmitted: a fresh leg, no training state carried"


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
    guard_start = None
    if guard is not None:
        step0 = chain_step(output) if output.exists() else 0
        has_pending = "pending" in inspect.signature(guard.start).parameters  # an older bundle's guard: no kwarg
        guard_start = guard.start(step0, **({"pending": pending_step1(output)} if has_pending else {}))
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
    if os.environ.get("ARC_PREFLIGHT_STUB"):  # the judge's back-off leg check (stub preempt_after)
        stub = json.loads(os.environ["ARC_PREFLIGHT_STUB"])
        identity["preempt_after"] = int(stub.get("preempt_after") or 0)
        identity["metrics_outage"] = stub.get("metrics_outage")
        identity["die_step"] = int(stub.get("die_step") or 0)
        identity["expect_resume_pending"] = bool(stub.get("expect_resume_pending"))
        identity["restore_check"] = bool(stub.get("restore_check"))
        identity["tiny_trainer"] = bool(stub.get("tiny_trainer"))
        if stub.get("tiny_trainer"):  # the stand-in's commands also run on the real trainer (arena.fake_trainer.shadow)
            os.environ["ARC_PREFLIGHT_TINY_TRAINER"] = "1"
        if stub.get("knobs"):  # the launch's knob put, live from the start as in the run
            Path(KNOBS_LOCAL).write_text(json.dumps(stub["knobs"]))
            identity["knobs"] = stub["knobs"]
    # beside the setup (the card is free until the server's weights load, >= 60 s in); stand-in cards record skipped
    threading.Thread(target=host_check, args=(identity,), daemon=True).start()
    boot = boot_settings(source)
    runtime_dir = None
    if boot.get("runtime_cache"):  # MFU-2: the extracted vLLM runtime from the volume (arena.runtime_cache)
        runtime_dir = CACHE_ROOT + "/runtime-cache" + ("-preflight" if os.environ.get("ARC_PREFLIGHT_STUB") else "")
        os.environ["ARC_RUNTIME_CACHE"] = runtime_dir
    if boot.get("expert_mask_skip_read"):  # MFU-2 (mfu2-maskskip): server.expert_mask's masked experts' packed
        os.environ["ARC_EXPERT_MASK_SKIP_READ"] = "1"  # weights never read (expert_mask_patch; scales still read; exact)
        identity["expert_mask_skip_read"] = True
    if boot.get("expert_mask_compact"):  # MFU-2: masked experts not allocated (expert_mask_patch COMPACT_TARGETS;
        os.environ["ARC_EXPERT_MASK_COMPACT"] = "1"  # +KV on card 1; exactness vs the masked path is GPU-checked)
        identity["expert_mask_compact"] = True
    if boot.get("load_profile"):  # MFU-2: weight-load time split per load (expert_mask_runtime.install_load_profile;
        os.environ["ARC_LOAD_PROFILE"] = "1"  # needs server.expert_mask; exact, rows in the feed's expert_mask event)
        identity["load_profile"] = True
    if boot.get("prefetch_threads"):  # MFU-2: page-cache the weights while the bootstrap installs and extracts
        identity["prefetch"] = {"threads": int(boot["prefetch_threads"])}
        threading.Thread(target=prefetch_weights, daemon=True, args=(
            identity["prefetch"], boot["prefetch_threads"]), kwargs={
            "limit": 2 if os.environ.get("ARC_PREFLIGHT_STUB") else None}).start()
    else:
        WEIGHTS_DONE.set()
    os.environ["ARC_RL1_STARTED_EPOCH"] = str(started)  # stage H: the boot timeline's zero (rl_seq seq_boot)
    if boot.get("prebuilt_venv") and Path(PREBUILT_VENV).is_dir():  # the image's agent venv (bundle_bootstrap)
        os.environ["ARC_PREBUILT_VENV"] = PREBUILT_VENV
    identity["host_memory"] = host_plan(CARDS, meminfo(), boot.get("host"))
    identity["memory"] = memory_limits()  # MemTotal / MemAvailable / cgroup memory.max at container start
    print(json.dumps({"event": "host_plan", **identity["host_memory"], "memory": identity["memory"]}), flush=True)
    # Modal preemption (v6 21 Oct 20:37: "Container terminated due to preemption", restarted with the same input): an
    # earlier attempt that died (no modal-identity.json) with a round state (v7: step state) is resumed, not discarded; its live knobs
    # carry over (applied ones stay applied: knobs_seen is in mg_state)
    # (its kept tries count too: a preemption in the first round after a resume leaves partial/ but no mg_state; the
    # loop resumes the latest root with an mg_state and reuses same-policy tries from every root)
    if lost_state(lost):
        roots = [x for x in (resume or "").split(",") if x]
        if guard_start is not None and guard_start.resume:  # earlier legs of this attempt, before the dead one
            roots += [rel_ for p in lost_chain(output) if p != lost and (rel_ := str(p.relative_to("/runs"))) not in roots]
        resume = ",".join(roots + [str(lost.relative_to("/runs"))])
        identity["preempt_resume"] = str(lost)
    carry_live(lost, output)
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ.update(PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1", MPLBACKEND="Agg", ONLY_RESET_LEVELS="true",
                      PYTHONDONTWRITEBYTECODE="1", ARC_FORCE_CHROOT="1", ARENA_STOP_REQUEST=STOP_LOCAL, ARENA_KNOBS=KNOBS_LOCAL,
                      ARC_VALIDATION_OUTPUT=str(output))
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
    if not (os.environ.get("ARC_PREFLIGHT_STUB") or os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER")):
        try:  # v5: a park check passed on the GPU for this trainer key is not rerun (stand-in trainers never record)
            os.environ["ARC_RL1_TRAINER_KEY"] = identity["trainer_key"] = trainer_key(source)
        except Exception as exc:  # no key: the check runs
            identity["trainer_key_error"] = f"{type(exc).__name__}: {exc}"
    else:  # CPU legs: the same key computation in this image's python, recorded only (stand-ins never use a key)
        try:
            identity["trainer_key_probe"] = trainer_key(source)[:16]
        except Exception as exc:
            identity["trainer_key_error"] = f"{type(exc).__name__}: {exc}"
    cache_key = None
    if boot.get("trainer_cache"):  # stage H: the trainer's compile / autotune caches from the volume (rl_loop)
        cache_key = hashlib.sha256(f"{identity.get('trainer_key') or 'stand-in'}:{GPU}".encode()).hexdigest()[:16]
        os.environ["ARC_TRAINER_CACHE"] = str(TRAINER_CACHE)
        threading.Thread(target=trainer_cache_restore, args=(cache_key, identity), daemon=True).start()
    if resume:
        os.environ["ARC_RL1_RESUME"] = ",".join(unpacked(vabs(d)) for d in resume.split(","))
        threading.Thread(target=prefetch_pending, args=(os.environ["ARC_RL1_RESUME"].split(","), identity),
                         daemon=True).start()
    if os.environ.get("ARC_PREFLIGHT_STUB") and not shutil.which("nvidia-smi"):  # a CPU preflight: a stand-in B300
        fake = Path("/tmp/fake-smi")
        fake.mkdir(exist_ok=True)
        (fake / "nvidia-smi").write_text(FAKE_SMI)
        (fake / "nvidia-smi").chmod(0o755)
        os.environ["PATH"] = f"{fake}:{os.environ['PATH']}"
        os.environ["ARC_FAKE_SMI_DIR"] = str(fake)  # the fake trainer's park writes the card's memory.used here
    if CARDS >= 2 and shutil.which("nvidia-smi"):  # the serving stack checks for exactly one card: it sees GPU 0 (the trainer asks for its own card)
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
                for name in ("feed.jsonl", "status.json", "trainer.log"):  # trainer.log (v17 a5): a failed op's
                    # traceback row reaches the volume each pass, not only at a clean end
                    if (live / name).exists():
                        mirror.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(live / name, mirror / name)
                live_sample(live, output / "live", started)
                began = time.time()
                commit_volumes()
                LAST_COMMIT["seconds"] = round(time.time() - began, 1)
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
        latest = {}  # card index -> its last line from one long-lived nvidia-smi (U2, 27 Oct: no spawn per second)

        def smi_loop():
            try:
                proc = subprocess.Popen([smi, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                         "--format=csv,noheader", "-lms", "1000"], stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True)
                for line in proc.stdout:
                    if line.strip() and line.split(",")[0].strip().isdigit():
                        latest[line.split(",")[0].strip()] = line.strip()
                    if stop.is_set():
                        proc.terminate()
                        return
            except Exception:  # noqa: BLE001 - the per-second spawn below stands in
                return
        if os.environ.get("ARC_SMI_LOOP", "1") == "1":
            threading.Thread(target=smi_loop, daemon=True).start()

        def procmem_loop():  # real-v5 27 Oct: anon grew ~3 GiB/min in play with no process named; per-process RssAnon
            with (util / "procmem.jsonl").open("a") as out:
                while not stop.wait(float(os.environ.get("ARC_PROCMEM_EVERY_S", "30"))):
                    try:
                        out.write(json.dumps(procmem_row()) + "\n")
                        out.flush()
                    except Exception:  # noqa: BLE001 - sampling never ends the run
                        continue
        threading.Thread(target=procmem_loop, daemon=True).start()
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
                    cards = "\n".join(latest[k] for k in sorted(latest, key=int)) if latest else subprocess.run(
                        [smi, "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=10).stdout
                    gpu.write(" ".join(cards.split("\n")).strip() + f" {now}\n")
                    cpu.flush()
                    gpu.flush()
                except Exception:  # sampling never ends the run
                    continue

    key = None
    if boot_cache_in(bundle_study()):  # the server's own boot cache restores after its overlays (arc_infra.boot_cache);
        identity["compile_cache"] = {"skipped": "ARC_BOOT_CACHE_IN"}  # this restore ran first and was cleared (T4, 24 s)
    elif (source / "configs" / "compile-cache.json").is_file():  # v5 bundles opt in
        try:
            key = compile_key(source)
            compile_restore(key, identity)
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
            if CARDS > 1 and not os.environ.get("ARC_DRY_BOOT_ONLY"):  # parallel boot: cards 1..N-1 beside server 1
                threads.append(threading.Thread(target=server_parallel, args=(stop, source, output), daemon=True))
            if runtime_dir:
                threads.append(threading.Thread(target=runtime_cache_save, args=(stop, runtime_dir), daemon=True))
            if rel.startswith("validation/preflight/") and json.loads(os.environ.get("ARC_PREFLIGHT_STUB") or "{}").get(
                    "self_kill"):
                trial = os.environ.get("ARC_RL1_TRIAL") or "rl-loop"
                threads.append(threading.Thread(target=self_kill, args=(
                    stop, Path(f"/tmp/arena/trials/{trial}/rl"), output / f"arena/trials/{trial}/rl"), daemon=True))
            if cache_key:
                trial = os.environ.get("ARC_RL1_TRIAL") or "rl-loop"
                threads.append(threading.Thread(target=trainer_cache_save, args=(
                    stop, cache_key, Path(f"/tmp/arena/trials/{trial}/rl/feed.jsonl")), daemon=True))
            for thread in threads:
                thread.start()
            try:
                process.wait()
            finally:
                stop.set()
                threads[0].join(timeout=30)
                # the volume users too, before the Modal client closes (an in-flight knobs read_file raised
                # ClientClosed at teardown: a traceback the watcher reads as a crash)
                until = time.time() + 60
                for thread in threads[1:]:
                    thread.join(timeout=max(0.0, until - time.time()))
        identity["exit_code"] = process.returncode
        try:  # each play server's effective serving env (rl_seq.server_envs; per-card A/B, mg.redesign.card_env)
            identity["server_env"] = json.loads(Path("/tmp/arena/serving/server-env.json").read_text())
        except (OSError, ValueError):
            pass
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
            try:  # the killed attempt this leg resumed from (NK a0 -> -lost-): ~9K inodes otherwise (infra 28 Oct)
                if lost is not None and lost.is_dir() and not (lost / PACK).exists():
                    identity["packed_lost"] = pack(lost)
            except Exception as exc:  # it stays readable unpacked
                identity["pack_lost_error"] = f"{type(exc).__name__}: {exc}"
        (output / "modal-identity.json").write_text(json.dumps(identity, indent=2))
        commit_volumes()
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
    target = Path("/tmp") / vrel(folder)
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


def trainer_key(bundle):
    """The trainer's key: what the trainer runs, not the loop's driver or its config. The bundle's
    trainer files (rl1/*), the loop's trainer client (``class Trainer`` and ``trainer_command`` in rl_loop.py), the
    trainer's command-line settings and the mg update arguments. A park check passed on the GPU for a key is not rerun."""
    import ast

    bundle = Path(bundle)
    digest = hashlib.sha256()
    for path in sorted((bundle / "rl1").glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    source = (bundle / "src/arena/rl_loop.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("Trainer", "trainer_command"):
            digest.update(ast.get_source_segment(source, node).encode() + b"\0")
        if isinstance(node, ast.Assign) and any(getattr(x, "id", None) == "FAST" for x in node.targets):
            digest.update(ast.get_source_segment(source, node).encode() + b"\0")  # the trainer's MFU stages
    study = json.loads((bundle / "configs/comparison.json").read_text())
    cfg = json.loads(subprocess_settings(bundle, study))
    mg = cfg.get("mg") or {}
    keyed = {k: cfg.get(k) for k in ("model_dir", "trainer_python", "rank", "window", "optimizer", "loop_lora", "tis_cap",
                                      "path_chunk", "trainer_env", "attention", "l2_norm")}
    keyed["mg"] = {k: mg.get(k) for k in ("cap", "batch_tokens", "batch_max")}
    if mg.get("gspo_norm"):  # pilot 1 D; absent keeps earlier bundles' keys
        keyed["mg"]["gspo_norm"] = mg["gspo_norm"]
    digest.update(json.dumps(keyed, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def subprocess_settings(bundle, study):
    """The bundle's own ``rl_loop.settings`` of its study, as JSON (the bundle's code, not the launcher's): the merge
    only where the bundle has ``check`` (T3/T4 26 Oct: the validators of mg.redesign import arcengine, absent from this
    python, so every redesign run lost its trainer key)."""
    import subprocess

    out = subprocess.run([sys.executable, "-c", "import inspect, json, sys; from arena import rl_loop; "
                          "kw = {'check': False} if 'check' in inspect.signature(rl_loop.settings).parameters else {}; "
                          "print(json.dumps(rl_loop.settings(json.load(sys.stdin)['rl_loop'], **kw), default=str))"],
                         input=json.dumps(study), env={**os.environ, "PYTHONPATH": str(Path(bundle) / "src")},
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"rl_loop.settings exit {out.returncode}: " + " | ".join(out.stderr.strip().splitlines()[-3:]))
    return out.stdout


def smoke_check(study):
    """The trainer smoke runs as the run's own first phase (mg.reload_check): load, U1, checkpoint,
    reload, rescore (KL-to-base within 1e-3), the next update; a mismatch stops the run."""
    if not ((study.get("rl_loop") or {}).get("mg") or {}).get("reload_check"):
        raise SystemExit("launch gate: mg.reload_check is off (the in-run trainer smoke is the gate)")
    return "in-run first-phase smoke (mg.reload_check)"


# Settings/state-only restarts need no preflight: same code files as a bundle whose preflight
# passed, and every changed config value is a plain value change at one of these paths (a key appearing, vanishing,
# turning None/bool or changing type switches a code path and counts as a code change). Restart state is not in the
# bundle (ARC_RL1_RESUME_FROM, mg_state via the rollback tool).
SETTINGS_PATHS = (r"/name", r"/status", r"/rl_loop/lr", r"/rl_loop/resume_(from|round|state|adapter)",
                  r"/rl_loop/mg/(active_kv_fill|active_tries|gate_overcommit)", r"/rl_loop/(update|trainer)_timeout")  # crash ceilings (v6d-r2 U9 old pass past 3600 s, 22 Oct)


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
    cap = 8
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
    checks, untested = {}, {}
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
    restore = {"starts": 0, "bad": [], "state_only": 0}
    saved = set()
    for where in [base] + ([prior] if prior else []):
        try:
            saved |= set(names(f"{where}/state") or [])
        except OSError:
            pass
    # RL redesign (arena.rl_seq, mg.redesign; the only loop): sequential phases
    seq = [e for e in feed if e.get("kind") == "seq_start"]
    red = (study["rl_loop"].get("mg") or {}).get("redesign") or {}
    steps = sorted({int(e["step"]): e for e in feed if e.get("kind") == "step"}.values(), key=lambda e: e["step"])
    phases = [e for e in feed if e.get("kind") == "seq_phase"]
    first = int(steps[0]["step"]) if steps else 0
    shrunk = [e for e in feed if e.get("kind") == "preflight_shrink" and e.get("max_steps") is not None]
    want_steps = int(shrunk[-1]["max_steps"]) if shrunk else int(red.get("max_steps") or 0)  # the leg's own cap
    # the blind-guard leg's own alerts ("guard blind: ...") are its expected rows, not errors
    scripted = bool(identity.get("metrics_outage"))
    blinds = [i for i, e in enumerate(feed) if e.get("kind") == "alert" and str(e.get("why", "")).startswith("guard blind")]
    # a blind the guard saw past (seq_guard_sees on that url) is the fail-closed hold working, not an error (real-v5 N1:
    # the CPU stub's /metrics went unread 60 s under 280 tries per server); a blind never seen past stays an error
    seen = {i for i in blinds if any(e.get("kind") == "seq_guard_sees" and e.get("url") == feed[i].get("url")
                                     for e in feed[i + 1:])}
    errors = {e.get("kind") for i, e in enumerate(feed) if e.get("kind") in ("error", "alert")
              and not (e.get("kind") == "alert" and alert_only(e)) and not (scripted and i in blinds) and i not in seen}
    checks["feed_final_no_error"] = bool(seq) and "seq_end" in kinds and not errors
    # F1 (gate-v1 28 Oct: the pending groups' early prepare failed 16/16 as alert-only rows and the leg passed): every
    # ahead prepare lands and every seq_ahead_pending row prepared all its groups
    # ...and the trainer takes them (gate legs 28 Oct: 0 of 192 ahead windows hit, a key mismatch): every one-call
    # update with an ahead file uses >= 90% of its windows from it
    calls = [e for e in feed if e.get("kind") == "seq_accumulate" and e.get("one_call") and e.get("ahead_expected")]
    checks["ahead_used"] = all(isinstance(e.get("ahead_hit_share"), (int, float)) and e["ahead_hit_share"] >= 0.9
                               for e in calls)
    pend = [e for e in feed if e.get("kind") == "seq_ahead_pending"]
    checks["ahead_prepared"] = not any(e.get("kind") == "alert" and "ahead prepare failed" in str(e.get("why"))
                                       for e in feed) and all(int(e.get("prepared") or 0) == int(e.get("groups") or 0)
                                                              for e in pend)
    if identity.get("die_step"):  # leg a: its update failed as scripted after the phase's tries were saved
        pend = [e for e in feed if e.get("kind") == "seq_pending"]
        checks["pending_saved_before_die"] = bool(pend) and int(pend[-1].get("groups") or 0) > 0 and \
            pend[-1]["groups"] == pend[-1].get("full") and int(pend[-1]["step"]) == int(identity["die_step"])
        checks["feed_final_no_error"] = any("die_step" in str(e.get("error") or e.get("why") or "") for e in feed
                                            if e.get("kind") in ("error", "alert"))
    if identity.get("expect_resume_pending"):  # leg b: resumed, then the cut-short update's saved groups, no play
        res = [i for i, e in enumerate(feed) if e.get("kind") == "seq_resume"]
        rep = [i for i, e in enumerate(feed) if e.get("kind") == "seq_replay" and e.get("resumed_pending")]
        after = next((e for e in feed[rep[0]:] if e.get("kind") == "step"), None) if rep else None
        played = rep and any(e.get("kind") == "seq_group" for e in feed[res[0]:rep[0]]) if res else True
        checks["resume_trains_saved_tries"] = bool(res and rep and after) and not played and \
            int(feed[res[0]]["pending_groups"]) == int(feed[rep[0]]["groups"]) > 0 and \
            int(after["step"]) == int(feed[res[0]]["step"]) + 1 and int(after.get("groups") or 0) == feed[rep[0]]["groups"]
    if scripted:  # the guard went blind, held admission, and let go once /metrics answered again
        checks["guard_blind_hold"] = bool(blinds) and any(e.get("kind") == "seq_guard_sees" for e in feed[blinds[0] + 1:])
    # a kill/resume leg (v15 NK 29 Oct): the cancelled attempt's feed copy can lag its last checkpoint (feed to step 1,
    # state s0002), so the steps after the resume are judged from the resumed step on; the earlier ones, if any, as read
    resumed_at = [int(e.get("step") or 0) for e in feed if e.get("kind") == "seq_resume"]
    judged = [e for e in steps if resumed_at and int(e["step"]) > resumed_at[-1]] or steps
    first = resumed_at[-1] + 1 if judged is not steps else first
    checks["steps_ok"] = bool(steps) and [e["step"] for e in judged] == list(range(first, first + len(judged))) and all(
        isinstance(e.get("grad_norm"), (int, float)) and math.isfinite(e["grad_norm"]) for e in steps) and int(
        steps[-1]["step"]) >= (int(identity["die_step"]) - 1 if identity.get("die_step") else want_steps)
    if identity.get("die_step"):  # leg a ends on its scripted update failure: that failure, not a clean exit, is the pass
        died = any("die_step" in str(e.get("error") or "") for e in feed if e.get("kind") == "error")
        checks["exit_clean"] = checks["trial_complete"] = died
    checks["checkpoints_saved"] = bool(steps) and f"s{int(steps[-1]['step']):04d}" in saved
    if identity.get("tiny_trainer"):  # the real trainer really ran: a silent shadow would pass every other check
        tally = json.loads(read(f"{base}/tiny_ops.json") or "{}")
        want = {"accumulate", "step", "checkpoint"} | ({"old_pass"} if any(e.get("minibatches") for e in steps) else set())
        checks["tiny_trainer_ran"] = want <= set(tally.get("ok") or {}) and not tally.get("failed")
    checks["adapter_served"] = bool(phases) and all((e.get("start") or {}).get("load_s") is not None for e in phases)
    skips = sum(1 for e in feed if e.get("kind") == "seq_step_skip")  # an all-ties phase: a switch, no step
    checks["phase_switches"] = len(phases) == len(steps) + skips and all(
        (e.get("stop") or {}).get("switch_s") is not None for e in phases)
    # phase-start back-off (rl_seq.backoff): a phase with a preemption or KV peak > 0.85 before another phase is followed
    # by a cut; the back-off leg (stub preempt_after) must show one, or on a 1-step leg (no next phase: D2 v2 N1) the
    # guard's read of the stub's preemption
    hits = [i for i, e in enumerate(feed) if e.get("kind") == "seq_phase" and ((e.get("guard") or {}).get("preemptions")
                                                                             or ((e.get("guard") or {}).get("kv_peak") or 0) > 0.85)]
    cuts = [i for i, e in enumerate(feed) if e.get("kind") == "seq_knobs" and e.get("backoff")]
    # a phase followed: its play ended (seq_train); a seq_group alone can be the last phase's refill (D2 v2 N1)
    later = [i for i in hits if any(e.get("kind") == "seq_train" for e in feed[i + 1:])]
    # pressure only in the last played phase (no phase after it to cut; real-v6 N9 27 Oct, a 1-step leg too): the
    # guard's read of it is the check
    checks["backoff_on_pressure"] = all(any(c > i for c in cuts) for i in later) and (
        not identity.get("preempt_after") or bool(cuts) or (bool(hits) and not later))
    checks["harvest_saved"] = bool(phases) and int((phases[-1].get("stats") or {}).get("saved") or 0) > 0
    if identity.get("restore_check"):  # NK: a fresh a1 fails here by name (NKc 29 Oct); a checkpoint resume only
        checks["resumed_from_checkpoint"] = any(e.get("kind") == "seq_resume" for e in feed)
    elif resume:  # a preempted attempt also resumes without a checkpoint (N1f 29 Oct, preempted at step 0: the
        # mid-play restore seq_play_resume, no seq_resume row) or by replaying the cut update's saved groups
        checks["resumed_from_checkpoint"] = any(
            e.get("kind") in ("seq_resume", "seq_play_resume") or (e.get("kind") == "seq_replay" and e.get("resumed_pending"))
            for e in feed)
    for name in names(rl):
        if not re.match(r"(g\d{5}|x\d{4}|p\d+g\d+)-", name):
            continue
        bench = json.loads(read(f"{rl}/{name}/level-bench.json") or "{}")
        for row in bench.get("attempts") or []:
            r = row.get("restore")
            if r:
                restore["starts"] += 1
                if int(r.get("scripted") or 0) > 0:
                    restore.setdefault("rows", []).append(r)
                kind = restore_kind(r)
                restore["state_only"] += kind == "state"
                if kind is None:
                    restore["bad"].append([name, row.get("label"), r.get("diff_count"), (r.get("state") or {}).get("detail")])
    checks["restored_starts_exact"] = not restore["bad"]  # a leg restores only the starts it harvested (maybe none)
    if identity.get("restore_check"):  # NK: every restored attempt with a scripted prefix wrote its
        # reply-id seeds, its first live request found them loaded in time, and the digest gate dropped no try
        rows = restore.pop("rows", [])
        live = [r for r in rows if "seed_wait_timeout" in r]
        drops = sum(int(e.get("dropped") or 0) for e in feed if e.get("kind") == "digest_gate")
        restore["seed"] = {"scripted": len(rows), "live": len(live), "seeded": sum(r.get("seeded") == 1 for r in rows),
                           "timeouts": sum(int(r.get("seed_wait_timeout") or 0) for r in live), "gate_drops": drops}
        checks["restore_played"] = bool(live)
        checks["restore_seeded"] = bool(rows) and all(r.get("seeded") == 1 for r in rows)
        checks["seed_wait_no_timeout"] = bool(live) and not restore["seed"]["timeouts"]
        checks["digest_gate_no_drops"] = drops == 0
        # a0 dies with a harvested-start group in flight (self_kill) and the resume reopens that
        # start first (rl_backlog), so a leg that played no restored start is a failure, no longer inconclusive
        # the resumed attempt's route restore reads the tars attempt 0 mirrored, every one
        # indexed (pass-*.idx.json): restore_seconds reported, no tar scanned
        sr = [e for e in feed if e.get("kind") == "served_restore"]
        if sr:
            last = sr[-1]
            restore["served"] = {k: last.get(k) for k in ("seconds", "files", "bytes", "indexed", "scanned", "rids",
                                                          "error")}
            checks["restore_indexed"] = (not last.get("error") and int(last.get("indexed") or 0) > 0
                                         and int(last.get("scanned") or 0) == 0)
    restore.pop("rows", None)
    log = read(f"{base}/arena/serving/vllm-openai-server.log") or ""
    stub = [l.split("STUB ", 1)[1] for l in log.splitlines() if " STUB {" in l]
    stub = json.JSONDecoder().raw_decode(stub[-1])[0] if stub else {}  # KV_STUB {...} follows on the same line
    checks["stub_clean_on_recorded_paths"] = stub.get("errors") == 0 and stub.get("exact", 0) > 0
    na = dict(untested)
    if checks.get("resumed_from_checkpoint"):  # NK2 28 Oct: a resumed attempt's carried tries have contexts the
        # recorded corpus never saw, so no request can hit it exactly; the stub must still answer every one
        checks.pop("stub_clean_on_recorded_paths")
        checks["stub_no_errors"] = stub.get("errors") == 0
        na["stub_clean_on_recorded_paths"] = ("resumed attempt (seq_resume): carried tries' contexts are not in the "
                                              "recorded corpus, exact hits impossible; stub_no_errors checks instead")
    boot = read(f"{base}/bootstrap.log") or ""
    checks["server_stopped"] = "server_stop_timeout" not in boot and bool(boot)
    if red.get("play_only"):  # G-long (27 Oct): generation only; phases close and play on, no switch, no step
        plays = [e for e in feed if e.get("kind") == "seq_play_phase"]
        for key in ("steps_ok", "checkpoints_saved", "adapter_served", "phase_switches"):
            checks.pop(key)
        checks["play_phases"] = bool(plays) and not steps and not any(e.get("kind") == "seq_train" for e in feed)
        checks["harvest_saved"] = bool(plays) and int((plays[-1].get("stats") or {}).get("saved") or 0) > 0
        checks["admit_logged"] = any(e.get("kind") == "seq_admit" and e.get("servers") for e in feed)
        hot = [i for i, e in enumerate(feed) if e.get("kind") == "seq_play_phase" and (
            (e.get("guard") or {}).get("preemptions") or ((e.get("guard") or {}).get("kv_peak") or 0) > 0.85)]
        nxt = [i for i in hot if any(e.get("kind") == "seq_play_phase" for e in feed[i + 1:])]  # a phase followed
        checks["backoff_on_pressure"] = all(any(c > i for c in cuts) for i in nxt) and (
            not identity.get("preempt_after") or bool(cuts) or (bool(hot) and not nxt))
    replay = [e for e in feed if e.get("kind") == "seq_replay"]
    if replay and "seq_group" not in kinds:  # an update leg (mg.redesign.replay_groups): saved groups, nothing played,
        # so no harvest, stub request or phase pressure; the saved groups are the step's
        for key in ("harvest_saved", "stub_clean_on_recorded_paths", "stub_no_errors", "backoff_on_pressure"):
            checks.pop(key, None)  # a resumed replay leg (N11) already swapped the stub check
        na.pop("stub_clean_on_recorded_paths", None)
        trains = [e for e in feed if e.get("kind") == "seq_train"]
        n = int(replay[0].get("groups") or 0)
        checks["replay_trained"] = n > 0 and bool(trains) and int(trains[0].get("groups") or 0) == n and bool(
            steps) and int(steps[0].get("groups") or 0) == n
    probe = [e for e in feed if e.get("kind") == "lr_probe"]
    if probe:  # an lr_probe bundle (P1, rl_seq.lr_probe: mg.redesign.lr_probe): no served step;
        # every arm's row ok and the first arm's in-run smoke (mg.reload_check)
        mg = study["rl_loop"].get("mg") or {}
        rates = [e for e in feed if e.get("kind") == "lr_probe_rate"]
        checks = {"exit_clean": checks["exit_clean"], "trial_complete": checks["trial_complete"],
                  "no_error": not {"error", "alert"} & set(kinds), "server_stopped": checks["server_stopped"],
                  "every_arm_ok": bool(rates) and all(e.get("ok") for e in rates) and not probe[-1].get("skipped"),
                  "steps_probed": bool(rates) and all(s.get("kl_base") is not None for e in rates for s in e.get("steps") or [{}])}
        if mg.get("reload_check"):
            checks["first_arm_smoke"] = bool(rates) and bool((rates[0].get("reload") or {}).get("ok")) and bool(
                (rates[0].get("reload") or {}).get("update_after_reload"))
    scored = [e for e in feed if e.get("kind") == "score_check"]
    if scored:  # a score_check bundle: nothing plays or trains; the path ran (stand-ins hold no tokens to read)
        checks = {"exit_clean": checks["exit_clean"], "trial_complete": checks["trial_complete"],
                  "no_error": "error" not in kinds, "server_stopped": checks["server_stopped"],
                  "probes_built": int(scored[-1].get("probes") or 0) > 0,
                  "seeds_loaded": not any(int(x.get("timeouts") or 0) for x in scored[-1].get("seeded") or []),
                  "both_sides_scored": "verdict" in scored[-1]}
        untested = []
    verdict = "pass" if all(checks.values()) else "fail"
    if verdict == "pass" and untested:
        verdict = "inconclusive"
    return {"verdict": verdict, "checks": checks, "restore": restore, "stub": stub,
            **({"na": na} if na else {}),
            "feed_kinds": {k: kinds.count(k) for k in sorted(set(kinds))}, "elapsed_seconds": identity.get("elapsed_seconds")}


def preflight_verdict(run_id):
    """Pass/fail of a preflight from its output: clean exit, every set played, restored starts exact, stub without
    errors and on recorded paths, server stopped; written into ``docs/validation-rl/preflight-<run id>.json``. The
    container's in-app judgement (``JUDGED``) is used when present (no pack download); an older or unfinished leg is
    judged from the volume."""
    record = ROOT / "docs/validation-rl" / f"preflight-{run_id}{PREFLIGHT_TAG}.json"
    data = json.loads(record.read_text())
    base = f"validation/preflight/{run_id}{PREFLIGHT_TAG}"
    try:  # ARC_RL1_REJUDGE=1: a judge rule changed after the leg ran (re-read from the volume, in-app verdict kept there)
        judged = None if os.environ.get("ARC_RL1_REJUDGE") == "1" else json.loads(
            b"".join(runs.read_file(f"{base}/{JUDGED}")).decode())
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
LAUNCH_ENV = {"ARC_RL1_AUTHORIZATION", "ARC_RL1_BUNDLE", "ARC_RL1_CARDS", "ARC_RL1_GATE_ALONGSIDE",
              "ARC_RL1_OVERRIDES", "ARC_RL1_PATCH_CHECK", "ARC_RL1_DRY_BOOT", "ARC_RL1_DRY_BOOT_GPU", "ARC_RL1_RESUME_FROM",
              "ARC_RL1_REJUDGE", "ARC_RL1_STALL", "ARC_RL1_STUB_SERVING", "ARC_RL1_STALL_DUMP", "ARC_RL1_STALL_KILL", "ARC_RL1_TIMEOUT", "ARC_RL1_TRAINER_KEY",
              "ARC_RL1_VOLUME", "ARC_RL1_INPUT_VOLUMES"}


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


def stub_spec(study):
    """The replay stub's spec from the leg's ARC_RL1_PREFLIGHT_* settings: a CPU preflight's, or a stub-serving
    GPU run's (ARC_RL1_STUB_SERVING=1: the real trainer on the card, no vLLM; D2 27 Oct update legs)."""
    return {"corpus": [vabs(d) for d in os.environ["ARC_RL1_PREFLIGHT_CORPUS"].split(",")],
            "serving": vabs(os.environ["ARC_RL1_PREFLIGHT_SERVING"]),
            "speed": float(os.environ.get("ARC_RL1_PREFLIGHT_SPEED") or 4000), "stall": STALL,
            "shrink": {"1": True, "2": 2}.get(os.environ.get("ARC_RL1_PREFLIGHT_SHRINK") or "", False),  # 2: minimum size (22 Oct)
            # tied group rewards get a fixed ladder so updates run (replays repeat their recordings)
            "spread": True,
            # back-off leg: the stub reports one preemption past this many generated tokens;
            # the judge then requires the next phase's cut (seq_knobs backoff row)
            "preempt_after": int(os.environ.get("ARC_RL1_PREFLIGHT_PREEMPT_AFTER") or 0),
            # blind-guard leg: /metrics answers 503 from START s for SECONDS s ("START,SECONDS");
            # the judge then requires the guard's blind hold and its release
            "metrics_outage": [float(x) for x in os.environ["ARC_RL1_PREFLIGHT_METRICS_OUTAGE"].split(",")]
            if os.environ.get("ARC_RL1_PREFLIGHT_METRICS_OUTAGE") else None,
            "knobs": json.loads(os.environ.get("ARC_RL1_PREFLIGHT_KNOBS") or "null"),  # the launch's put (launch.py)
            # kill-in-update legs: leg a's fake trainer fails its step DIE_STEP; leg b resumes from leg a
            # (ARC_RL1_RESUME_FROM) and must train the saved groups of that update first, with no play
            "die_step": int(os.environ.get("ARC_RL1_PREFLIGHT_DIE_STEP") or 0),
            "expect_resume_pending": os.environ.get("ARC_RL1_PREFLIGHT_EXPECT_RESUME_PENDING") == "1",
            # kill/resume leg: more shrunk steps so the resumed attempt plays restored starts, and
            # the judge's restore checks (reply-id seeds loaded before the first live request, no digest-gate drop)
            "max_steps": int(os.environ.get("ARC_RL1_PREFLIGHT_MAX_STEPS") or 0),
            "restore_check": os.environ.get("ARC_RL1_PREFLIGHT_RESTORE_CHECK") == "1",
            "tiny_trainer": os.environ.get("ARC_RL1_PREFLIGHT_TINY") == "1",
            "self_kill": os.environ.get("ARC_RL1_PREFLIGHT_SELF_KILL") == "1",  # NK attempt 0 (launch.py nk)
            "served_route": (study.get("server") or {}).get("served_route"),  # stand-in route records (c)
            # patch check (24 Oct, v6f's boot death): the real serving prep with the trial's server_env and
            # ARC_SERVING_PATCH_CHECK=1 on the pinned runtime tree, then exit; no loop (bootstrap ARC_PATCH_CHECK_ONLY)
            "patch_check": os.environ.get("ARC_RL1_PATCH_CHECK") == "1",
            "dry_boot": os.environ.get("ARC_RL1_DRY_BOOT") == "1",  # 25 Oct: patch check + env + argv + imports
            "dry_boot_gpu": os.environ.get("ARC_RL1_DRY_BOOT_GPU") or None,
            "trial": os.environ.get("ARC_RL1_PREFLIGHT_TRIAL") or "rl-loop"}  # the bundle's trial to run


@app.local_entrypoint()
def main(run_id: str, dry_run: bool = False, preflight_run: bool = False, verdict: bool = False):
    out = launch(run_id, dry_run, preflight_run, verdict)
    if not (dry_run or preflight_run or verdict):
        record_app(RECORD_ID or run_id)  # a launch: its record names its app
    return out


def launch(run_id, dry_run=False, preflight_run=False, verdict=False):
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
    """An RL run (rl1-loop-*) of the bundle: dry run, its CPU preflight (``preflight_run``), or the run itself."""
    if not run_id.startswith("rl1-loop-"):
        raise ValueError("run ids start with rl1-loop-")
    study = bundle_study()
    mode = study["rl_loop"]["mode"]
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
            "rl_loop": {k: study["rl_loop"].get(k) for k in ("mode", "lr", "rank", "window")}}
    if dry_run:
        print(json.dumps(data, indent=1))
        return
    if preflight_run:
        pf = ROOT / "docs/validation-rl" / f"preflight-{run_id}{PREFLIGHT_TAG}.json"
        if pf.exists():
            raise SystemExit(f"{pf} exists")
        spec = stub_spec(study)
        call = preflight.spawn(run_id, json.dumps(spec), RESUME_FROM, PREFLIGHT_TAG, OVERRIDES)
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
    data["trainer_smoke"] = smoke_check(study)
    record.parent.mkdir(exist_ok=True)
    data["state"] = "submission_intent"
    record.write_text(json.dumps(data, indent=2))
    data["resume"] = RESUME_FROM
    overrides = json.loads(OVERRIDES) if OVERRIDES else {}
    data["overrides"] = overrides or None
    stub = json.dumps(stub_spec(study)) if os.environ.get("ARC_RL1_STUB_SERVING") == "1" else ""
    data["stub_serving"] = json.loads(stub) if stub else None
    call = run.spawn(run_id, RESUME_FROM, STALL, json.dumps(overrides) if overrides else "", stub)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


