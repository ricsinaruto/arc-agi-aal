"""chk.py RUN [MINUTES=30]: one-line live check of a v9 kf run (streams, gen tok/s, KV, GPU/ext/combined hit, tier use,
gate/waiting, host RAM, preemption mentions in the server tail)."""
import json
import re
import subprocess
import sys
from datetime import datetime

run, mins = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 30
P = f"nebius-arc:arc-runs-bucket/runs/validation/{run}"


def cat(p):
    return subprocess.run(["rclone", "cat", f"{P}/{p}"], capture_output=True, text=True).stdout


tail = cat("arena/live/server-tail.log")
pat = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d)[^\n]*generation throughput: ([\d.]+)[^\n]*Running: (\d+) reqs, "
                 r"Waiting: (\d+)[^\n]*KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"
                 r"(?:, External prefix cache hit rate: ([\d.]+)%)?")
rows = [(datetime.strptime("2026-" + m[0], "%Y-%m-%d %H:%M:%S").timestamp(), *map(float, m[1:6]),
         float(m[6]) if m[6] else None) for m in pat.findall(tail)]
out = []
if rows:
    t1 = rows[-1][0]
    w = [r for r in rows if r[0] > t1 - mins * 60]
    n = len(w)
    g = sum(r[1] for r in w) / n
    kv = sum(r[4] for r in w) / n / 100
    hg = sum(r[5] for r in w) / n / 100
    ext = [r[6] for r in w if r[6] is not None]
    he = sum(ext) / len(ext) / 100 if ext else 0.0
    streak = best = 0.0
    for a, b in zip(w, w[1:]):
        streak = streak + (b[0] - a[0]) if (a[3] > 0 and b[3] > 0) else 0.0
        best = max(best, streak)
    out.append(f"last {pat.findall(tail)[-1][0][6:11]} ({n} lines/{mins:.0f}m): gen {g:.0f} tok/s, running "
               f"{min(r[2] for r in w):.0f}-{max(r[2] for r in w):.0f}, vllm waiting>0 {sum(r[3] > 0 for r in w)}/{n} (longest run {best / 60:.1f} min), "
               f"KV {kv:.3f}, hit GPU {hg:.4f} ext {he:.3f} combined {hg + (1 - hg) * he:.4f}")
cpu = re.findall(r"kv_offload_cpu_cache_usage_perc=([\d.e-]+)", tail)
if cpu:
    out.append(f"tier use {float(cpu[-1]):.3f}")
out.append(f"preempt mentions {len(re.findall(r'preempt', tail, re.I))}")
prog = [json.loads(x) for x in cat("arena/live/progress.jsonl").splitlines() if x.strip()]
if prog:
    d = prog[-1]
    gs = [p["gate"] for p in prog if p.get("gate") and p["elapsed_seconds"] > d["elapsed_seconds"] - mins * 60]
    out.append(f"play {d['elapsed_seconds'] / 3600:.2f} h, active {len(d.get('active', []))}, levels {d.get('levels')}, "
               f"gate reserved mean {sum(x['reserved'] for x in gs) / max(1, len(gs)):.0f} in-flight "
               f"{sum(x['inflight'] for x in gs) / max(1, len(gs)):.1f} waiting>0 {sum(x['waiting'] > 0 for x in gs)}/{len(gs)}")
mt = [json.loads(x)["memtrace"] for x in cat("arena/memtrace.jsonl").splitlines() if x.strip()]
if mt:
    m = mt[-1]
    nr = max(x["anon_gib"] + x["shmem_gib"] for x in mt)
    out.append(f"host used {m['used_gib']} peak {m['peak_gib']} anon {m['anon_gib']} shmem {m['shmem_gib']} "
               f"non-recl peak {nr:.1f} oom {m['events']['oom_kill']} offload files {m['offload_files']}")
print(run + ": " + "; ".join(out))
