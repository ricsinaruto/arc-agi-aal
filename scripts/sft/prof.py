"""Where a training example's GPU time goes (util/MFU audit, 26 Oct): torch.profiler (shapes and FLOPs, no Python
stacks: util-v1's with_stack nearly doubled the example's host time) over one example's forward and backward. The
trace is saved and reduced in a separate process while training goes on (util-v1 reduced in-process for 183 s of
B300 time): the card's busy share (union of kernel intervals), its idle gaps by the profiler phase they fall in
(``record_function`` labels in train.example_loss), by the CUDA runtime/driver calls overlapping them and by the op
the host was in when each ended, the counted FLOPs by op (matmuls apart from with_flops' elementwise counts, plus the
grouped expert GEMMs from their recorded shapes, which with_flops skips), and a BF16 GEMM's measured peak."""
import bisect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MATMUL = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm")


def gemm_peak(device, n=8192, reps=20):
    """Achieved BF16 dense FLOP/s of an n x n x n matmul (the card's practical peak)."""
    import torch

    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(device)
    t = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize(device)
    return 2 * n ** 3 * reps / (time.perf_counter() - t)


def union(intervals):
    """Merged (start, end) intervals, sorted."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def matmul_flops(name, dims):
    """2 x the multiply-adds of a matmul op from its recorded input dims (the chrome trace carries no with_flops
    value); 0 when the dims do not fit."""
    try:
        if name == "aten::mm":
            (m, k), (_, n) = dims[0], dims[1]
            return 2 * m * k * n
        if name == "aten::addmm":
            (m, k), (_, n) = dims[1], dims[2]
            return 2 * m * k * n
        if name == "aten::bmm":
            (b, m, k), (_, _, n) = dims[0], dims[1]
            return 2 * b * m * k * n
        if name == "aten::baddbmm":
            (b, m, k), (_, _, n) = dims[1], dims[2]
            return 2 * b * m * k * n
    except (ValueError, TypeError, IndexError):
        return 0
    return 0


def grouped_flops(dims):
    """2 M K N of aten::_grouped_mm from its recorded input dims (a [M, K], b [E, K, N] or [E, N, K])."""
    if len(dims) < 2 or len(dims[0]) != 2 or len(dims[1]) != 3:
        return 0
    (m, k), (_, b1, b2) = dims[0], dims[1]
    return 2 * m * k * (b2 if b1 == k else b1)


class Spans:
    """Intervals (start, end, name) with point and overlap queries (1 ms buckets)."""

    def __init__(self, rows):
        self.rows, self.b = rows, {}
        for i, (s, e, _) in enumerate(rows):
            for k in range(int(s // 1000), int(e // 1000) + 1):
                self.b.setdefault(k, []).append(i)

    def near(self, a, b):
        seen = set()
        for k in range(int(a // 1000), int(b // 1000) + 1):
            for i in self.b.get(k, ()):
                if i not in seen:
                    seen.add(i)
                    yield self.rows[i]

    def innermost(self, t):
        best = None
        for s, e, name in self.near(t, t):
            if s <= t <= e and (best is None or e - s < best[0]):
                best = (e - s, name)
        return best[1] if best else "-"


def reduce(events, wall_s, min_gap_us=20.0, big_us=1000.0, top=20):
    """The reduction of a chrome trace's ``traceEvents``."""
    kernels, ops, calls, phases, flops, by_kernel = [], [], [], [], {}, {}
    for ev in events:
        cat, ts, dur = ev.get("cat"), ev.get("ts"), ev.get("dur")
        if ts is None or dur is None:
            continue
        ts, end, name = float(ts), float(ts) + float(dur), ev.get("name", "?")
        if cat == "kernel":
            kernels.append((ts, end))
            k = by_kernel.setdefault(name, [0, 0.0])
            k[0], k[1] = k[0] + 1, k[1] + float(dur)
        elif cat == "cpu_op":
            args = ev.get("args") or {}
            f = args.get("flops") or 0
            if name == "aten::_grouped_mm":
                f = grouped_flops(args.get("Input Dims") or [])
            elif name in MATMUL:
                f = matmul_flops(name, args.get("Input Dims") or [])
            if f:
                flops[name] = flops.get(name, 0) + f
            if float(dur) <= 1e6:
                ops.append((ts, end, name))
        elif cat in ("cuda_runtime", "cuda_driver"):
            calls.append((ts, end, name))
        elif cat == "user_annotation":
            phases.append((ts, end, name))
    merged = union(kernels)
    busy = sum(e - s for s, e in merged)
    span = (merged[-1][1] - merged[0][0]) if merged else 0.0
    ops, calls, phases = Spans(ops), Spans(calls), Spans(phases)
    gaps, by_phase, by_call, by_op, biggest = [], {}, {}, {}, []
    for (_, g0), (g1, _) in zip(merged, merged[1:]):
        g = g1 - g0
        if g < min_gap_us:
            continue
        gaps.append(g)
        if g < big_us:
            continue
        phase, op = phases.innermost((g0 + g1) / 2), ops.innermost(g1 - 5.0)
        over = {}
        for s, e, name in calls.near(g0, g1):
            o = min(e, g1) - max(s, g0)
            if o > 0:
                over[name] = over.get(name, 0.0) + o
        for table, key in ((by_phase, phase), (by_op, op)):
            row = table.setdefault(key, [0, 0.0])
            row[0], row[1] = row[0] + 1, row[1] + g
        for name, o in over.items():
            row = by_call.setdefault(name, [0, 0.0])
            row[0], row[1] = row[0] + 1, row[1] + o
        biggest.append((g, g0, phase, op, sorted(over.items(), key=lambda x: -x[1])[:3]))
    t0 = merged[0][0] if merged else 0.0

    def table(d):
        return [{"name": k[:160], "gaps": v[0], "s": round(v[1] / 1e6, 4)} for k, v in sorted(d.items(), key=lambda x: -x[1][1])[:top]]

    mm = sum(v for k, v in flops.items() if k in MATMUL)
    return {
        "wall_s": round(wall_s, 3), "kernel_span_s": round(span / 1e6, 6), "busy_s": round(busy / 1e6, 6),
        "busy_share_wall": round(busy / 1e6 / max(wall_s, 1e-9), 4), "busy_share_span": round(busy / max(span, 1e-9), 4),
        "idle_in_span_s": round((span - busy) / 1e6, 6), "gaps": len(gaps),
        "gap_s_by_size": {f">={lo}us": round(sum(g for g in gaps if g >= lo) / 1e6, 4) for lo in (20, 100, 1000, 10000)},
        "idle_by_phase": table(by_phase), "idle_by_op_at_end": table(by_op), "idle_overlapping_calls": table(by_call),
        "biggest_gaps": [{"ms": round(g / 1e3, 2), "at_s": round((s - t0) / 1e6, 3), "phase": p, "op": o,
                          "calls": [[n, round(x / 1e3, 2)] for n, x in c]} for g, s, p, o, c in sorted(biggest, reverse=True)[:60]],
        "kernels": len(kernels),
        "top_kernels": [{"name": k[:120], "n": v[0], "s": round(v[1] / 1e6, 3)}
                        for k, v in sorted(by_kernel.items(), key=lambda x: -x[1][1])[:top]],
        "flops_by_op": dict(sorted(((k, v) for k, v in flops.items()), key=lambda x: -x[1])[:top]),
        "flops_matmul": mm, "flops_grouped_mm": flops.get("aten::_grouped_mm", 0),
        "flops_hw_matmul": mm + flops.get("aten::_grouped_mm", 0),
    }


def profile(fn, out, device=None, meta=None):
    """``fn()`` under torch.profiler; the trace saved next to ``out`` and reduced into ``out`` by a separate process
    (``meta`` merged in). Returns the in-process part: wall seconds and the GEMM peak."""
    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profile

    cuda = device is not None and torch.device(device).type == "cuda"
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    if cuda:
        torch.cuda.synchronize(device)
    with torch_profile(activities=acts, record_shapes=True, with_flops=True) as prof:
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - t
    trace = f"/tmp/sft-profile-{os.getpid()}.json"
    t = time.perf_counter()
    prof.export_chrome_trace(trace)
    got = {"wall_s": round(wall, 3), "export_s": round(time.perf_counter() - t, 1), **(meta or {})}
    if cuda:
        got["gemm_peak_flops"] = gemm_peak(device)
    Path(out).write_text(json.dumps(got, indent=1))
    proc = subprocess.Popen([sys.executable, __file__, trace, str(out)], start_new_session=True)
    return got, proc


def main(trace, out):
    t = time.perf_counter()
    got = json.loads(Path(out).read_text())
    with open(trace) as f:
        events = json.load(f).get("traceEvents", [])
    got.update(reduce(events, got["wall_s"]), reduce_s=round(time.perf_counter() - t, 1))
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(got, indent=1))
    tmp.replace(out)
    os.remove(trace)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
