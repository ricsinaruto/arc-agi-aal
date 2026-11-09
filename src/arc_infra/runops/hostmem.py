"""Host-memory readings: the container's cgroup limit and use (what the OOM killer and Modal's worker act on), the
machine's MemAvailable, this process's RSS, and (``detail``) the biggest processes' RSS, pinned/locked pages, /dev/shm
and /tmp use, the cgroup's shmem and memory pressure (PSI). All sizes GiB; a missing source gives None."""
import os
import shutil
from pathlib import Path

GIB = 1024**3


def _read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _meminfo(text):
    out = {}
    for line in (text or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0]) * 1024  # kB
    return out


def _cgroup():
    """(used, limit) bytes from cgroup v2 or v1; limit None when unlimited."""
    for cur, lim in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                     ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        used = _read(cur)
        if used is None:
            continue
        limit = (_read(lim) or "max").strip()
        limit = None if limit == "max" or not limit.isdigit() or int(limit) >= 1 << 60 else int(limit)
        return int(used.strip()), limit
    return None, None


def _rss(status_text):
    for line in (status_text or "").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def sample(meminfo_text=None, status_text=None, cgroup=None):
    """One reading. ``avail_gib`` is the tighter of MemAvailable and the cgroup headroom. Arguments are for tests."""
    info = _meminfo(meminfo_text if meminfo_text is not None else _read("/proc/meminfo"))
    used, limit = cgroup if cgroup is not None else _cgroup()
    rss = _rss(status_text if status_text is not None else _read("/proc/self/status"))
    total, avail = info.get("MemTotal"), info.get("MemAvailable")
    if limit is not None and used is not None:
        head = limit - used
        avail = head if avail is None else min(avail, head)
        total = limit if total is None else min(total, limit)
    swap = (info["SwapTotal"] - info.get("SwapFree", 0)) if "SwapTotal" in info else None

    def g(x):
        return None if x is None else round(x / GIB, 2)

    return {"total_gib": g(total), "avail_gib": g(avail), "cgroup_used_gib": g(used), "cgroup_limit_gib": g(limit),
            "rss_gib": g(rss), "swap_used_gib": g(swap)}


def tight(s, floor_frac=0.05, floor_gib=8.0):
    """True when a reading shows the host near exhaustion (thrash or OOM territory)."""
    avail, total = s.get("avail_gib"), s.get("total_gib")
    if avail is None:
        return False
    return avail < floor_gib or (bool(total) and avail < floor_frac * total)


def _status_fields(text, keys=("VmRSS", "VmPin", "VmLck", "Name")):
    out = {}
    for line in (text or "").splitlines():
        k, _, v = line.partition(":")
        if k in keys:
            out[k] = v.strip() if k == "Name" else int(v.split()[0]) * 1024
    return out


def processes(proc="/proc"):
    """[(name, pid, rss, pinned+locked bytes)] of every readable process."""
    out = []
    try:
        pids = [d for d in os.listdir(proc) if d.isdigit()]
    except OSError:
        return out
    for pid in pids:
        f = _status_fields(_read(f"{proc}/{pid}/status"))
        if "VmRSS" in f:
            out.append((f.get("Name", "?"), int(pid), f["VmRSS"], f.get("VmPin", 0) + f.get("VmLck", 0)))
    return out


def _du(path, cap=200_000):
    """Bytes under ``path`` (stops after ``cap`` entries: a reading, not an audit)."""
    total, n, stack = 0, 0, [path]
    while stack and n < cap:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    n += 1
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_blocks * 512
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _pressure(text):
    """{"some": avg10, "full": avg10} (% of time stalled on memory, last 10 s)."""
    out = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if parts and parts[0] in ("some", "full"):
            for kv in parts[1:]:
                k, _, v = kv.partition("=")
                if k == "avg10":
                    out[parts[0]] = float(v)
    return out


def _memstat(text):
    out = {}
    for line in (text or "").splitlines():
        k, _, v = line.partition(" ")
        if v.strip().isdigit():
            out[k] = int(v)
    return out


def detail(top=4, proc="/proc", shm="/dev/shm", tmp="/tmp", cgroup_dir="/sys/fs/cgroup"):
    """Metric keys (infra.*) for the exporter: total and top-``top`` process RSS, pinned+locked, /dev/shm, /tmp, the
    cgroup's shmem/anon/file split and memory pressure. Paths are for tests."""
    procs = processes(proc)
    m = {"infra.host_rss_total_gib": sum(p[2] for p in procs) / GIB if procs else None,
         "infra.host_pinned_gib": sum(p[3] for p in procs) / GIB if procs else None}
    for name, pid, rss, _ in sorted(procs, key=lambda p: -p[2])[:top]:
        m[f"infra.host_rss_gib.{name[:24]}-{pid}"] = rss / GIB
    for key, path in (("infra.host_shm_gib", shm), ("infra.host_tmp_gib", tmp)):
        try:
            m[key] = (shutil.disk_usage(path).used if key.endswith("shm_gib") else _du(path)) / GIB
        except OSError:
            m[key] = None
    st = _memstat(_read(f"{cgroup_dir}/memory.stat"))
    for k in ("shmem", "anon", "file"):
        m[f"infra.host_cg_{k}_gib"] = st[k] / GIB if k in st else None
    psi = _pressure(_read(f"{cgroup_dir}/memory.pressure") or _read(f"{proc}/pressure/memory"))
    m["infra.host_psi_some"] = psi.get("some")
    m["infra.host_psi_full"] = psi.get("full")
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}


def thrashing(d, some=40.0, full=10.0):
    """True when memory pressure says the host is already reclaiming hard (PSI avg10 over the bars)."""
    return (d.get("infra.host_psi_some") or 0) >= some or (d.get("infra.host_psi_full") or 0) >= full
