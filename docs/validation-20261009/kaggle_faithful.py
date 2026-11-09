"""Kaggle host limits for a Nebius run of a competition bundle (25 Oct, v85l1b hidden-run ERROR test).

Copied into a bundle as ``src/arena/kaggle_faithful.py`` by ``nebius_kaggle_faithful.py``; never part of a Kaggle
build. The bootstrap calls ``apply`` first (root on the VM): the rerun branch (KAGGLE_IS_COMPETITION_RERUN), /dev/shm
resized, the bootstrap moved into a cgroup with Kaggle's memory.max (its children inherit it), and a trace every
``memtrace_seconds`` of the cgroup (used/peak/anon/file/shmem, memory.events) and the vLLM offload files in /dev/shm.
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

CGROUP_ROOT = Path(os.environ.get("ARENA_CGROUP_ROOT", "/sys/fs/cgroup"))
SHM = Path(os.environ.get("ARENA_SHM_DIR", "/dev/shm"))
GIB = 2 ** 30


def _read(path, default=None):
    try:
        return path.read_text().strip()
    except OSError:
        return default


def sample(group, started):
    stat = dict(line.split() for line in (_read(group / "memory.stat", "") or "").splitlines() if line.strip())
    events = dict(line.split() for line in (_read(group / "memory.events", "") or "").splitlines() if line.strip())
    files = [(p.name, p.stat().st_size) for p in SHM.iterdir() if p.name.startswith("vllm_offload")]
    fs = os.statvfs(SHM)
    row = {"s": round(time.time() - started), "used_gib": round(int(_read(group / "memory.current", 0)) / GIB, 2),
           "peak_gib": round(int(_read(group / "memory.peak", 0) or 0) / GIB, 2),
           **{k + "_gib": round(int(stat.get(k, 0)) / GIB, 2) for k in ("anon", "file", "shmem")},
           "events": {k: int(v) for k, v in events.items()},
           "offload_files": len(files), "offload_gib": round(sum(s for _, s in files) / GIB, 2),
           "offload_names": sorted(n for n, _ in files),
           "shm_used_gib": round((fs.f_blocks - fs.f_bfree) * fs.f_frsize / GIB, 2),
           "shm_free_gib": round(fs.f_bavail * fs.f_frsize / GIB, 2)}
    return row


def apply(cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if cfg.get("rerun"):
        os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    if cfg.get("shm_size"):
        subprocess.run(["mount", "-o", f"remount,size={cfg['shm_size']}", str(SHM)], check=True)
    group = CGROUP_ROOT / cfg.get("cgroup", "arc-kaggle")
    if "memory" not in (_read(CGROUP_ROOT / "cgroup.controllers", "") or "").split():
        raise RuntimeError("kaggle_faithful: cgroup v2 with the memory controller is required")
    if "memory" not in (_read(CGROUP_ROOT / "cgroup.subtree_control", "") or "").split():
        (CGROUP_ROOT / "cgroup.subtree_control").write_text("+memory")
    group.mkdir(exist_ok=True)
    (group / "memory.max").write_text(str(int(float(cfg["memory_max_gib"]) * GIB)))
    if (group / "memory.swap.max").exists():
        (group / "memory.swap.max").write_text("0")
    (group / "cgroup.procs").write_text(str(os.getpid()))
    mine = _read(Path("/proc/self/cgroup"), "")
    if CGROUP_ROOT == Path("/sys/fs/cgroup") and not mine.endswith("/" + group.name):
        raise RuntimeError(f"kaggle_faithful: not in {group.name}: {mine}")
    fs = os.statvfs(SHM)
    report = {"kaggle_faithful": {"rerun": os.environ.get("KAGGLE_IS_COMPETITION_RERUN"), "cgroup": str(group),
                                  "memory_max_gib": round(int(_read(group / "memory.max")) / GIB, 1),
                                  "shm_total_gib": round(fs.f_blocks * fs.f_frsize / GIB, 1), "self_cgroup": mine}}
    print(json.dumps(report), flush=True)
    started, trace = time.time(), output / "memtrace.jsonl"
    interval = int(cfg.get("memtrace_seconds", 60))

    def loop():
        while True:
            try:
                row = {"memtrace": sample(group, started)}
            except Exception as error:  # the trace must never stop the run
                row = {"memtrace": {"s": round(time.time() - started), "error": repr(error)}}
            with trace.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True, name="arc-memtrace").start()
    return report
