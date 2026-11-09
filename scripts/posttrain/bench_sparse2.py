"""Numerics and speed of ``rl1_sparse2.UnionSelectedAttention`` against ``rl1_sparse.TritonSelectedAttention`` and the
fp32 ``rl1_sparse.reference`` (MFU stage E). Picks follow ``qsa_select``'s format (512 blocks of 4 + a 3-key tail,
-1 padding); "real" picks are synthetic, calibrated to served picks (union of R consecutive rows / one row's: 1.42 /
2.00 / 2.79 / 3.87 at R 2/4/8/16; 16-31% in the last 2K tokens): static block salience + an AR(1) per-query field
(rho 0.8) + a recency bonus. "random" picks: any distinct causal keys, not block-aligned (the generic path).

    python bench_sparse2.py --mode test     # numerics (tiny shapes also run under TRITON_INTERPRET=1 on CPU)
    python bench_sparse2.py --mode bench    # + launch sweep and speed on the trainer's shapes (GPU)
Prints one JSON object per line."""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl1_sparse as old  # noqa: E402
import rl1_sparse2 as new  # noqa: E402

RATIO, TOP = 4, 512


def emit(**row):
    print(json.dumps(row), flush=True)


def picks_real(length, prefix, dev, seed=0, top=TOP, rho=0.8, salience=1.0, recency=3.0, tau=3000.0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    nb = (prefix + length) // RATIO
    z = torch.randn(nb, generator=gen, device=dev) * salience
    n = torch.randn(nb, generator=gen, device=dev)
    keep = math.sqrt(1 - rho * rho)
    out = torch.full((length, top * RATIO + RATIO - 1), -1, dtype=torch.long, device=dev)
    blocks = torch.arange(nb, device=dev)
    offs = torch.arange(RATIO, device=dev)
    chunk = 256
    rows = torch.empty((chunk, nb), device=dev)
    for b0 in range(0, length, chunk):
        b1 = min(length, b0 + chunk)
        for i in range(b1 - b0):
            n.mul_(rho).add_(torch.randn(nb, generator=gen, device=dev), alpha=keep)
            rows[i] = n
        counts = prefix + torch.arange(b0, b1, device=dev) + 1
        complete = counts // RATIO
        dist = (counts[:, None] - blocks[None, :] * RATIO).float()
        score = rows[: b1 - b0] + z[None] + recency * torch.exp(-dist / tau)
        score = score.masked_fill(blocks[None, :] >= complete[:, None], float("-inf"))
        k = min(top, nb)
        val, idx = score.topk(k, dim=-1)
        tok = torch.where(torch.isfinite(val)[:, :, None], idx[:, :, None] * RATIO + offs, -1)
        out[b0:b1, : k * RATIO] = tok.reshape(b1 - b0, -1)
        tail = (complete * RATIO)[:, None] + offs[: RATIO - 1]
        out[b0:b1, top * RATIO:] = torch.where(tail < counts[:, None], tail, -1)
    return out


def picks_random(length, prefix, width, dev, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    keys = prefix + length
    counts = prefix + torch.arange(length, device=dev) + 1
    score = torch.rand((length, keys), generator=gen, device=dev)
    score = score.masked_fill(torch.arange(keys, device=dev)[None, :] >= counts[:, None], -1.0)
    val, idx = score.topk(min(width, keys), dim=-1)
    return torch.where(val >= 0, idx, -1)


def union_stats(picked, rows=4096):
    blocks = torch.where(picked >= 0, picked // RATIO, -1)[:rows]
    def uniq(x):
        x = x.sort(dim=-1).values
        first = torch.ones_like(x, dtype=torch.bool)
        first[:, 1:] = x[:, 1:] != x[:, :-1]
        return (first & (x >= 0)).sum(-1).float()
    one = uniq(blocks).mean()
    out = {}
    for r in (2, 4, 8, 16):
        n = blocks.shape[0] // r * r
        out[f"R{r}"] = round(float(uniq(blocks[:n].reshape(n // r, -1)).mean() / one), 2)
    return out


def inputs(length, prefix, dev, seed=0, heads=24, groups=2, dim=256, dtype=torch.bfloat16):
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    keys = prefix + length
    q = torch.randn((1, heads, length, dim), generator=gen, device=dev).to(dtype)
    k = torch.randn((1, groups, keys, dim), generator=gen, device=dev).to(dtype)
    v = torch.randn((1, groups, keys, dim), generator=gen, device=dev).to(dtype)
    go = torch.randn((1, heads, length, dim), generator=gen, device=dev).to(dtype)
    return q, k, v, go


def run(fn, q, k, v, picked, scale, go):
    qs, ks, vs = (x.detach().clone().requires_grad_() for x in (q, k, v))
    o = fn(qs, ks, vs, picked, scale)
    o.backward(go.to(o.dtype))
    return [o.detach(), qs.grad, ks.grad, vs.grad]


def compare(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    cos = float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-300))
    return {"max_abs": float((a - b).abs().max()), "cos": round(cos, 7)}


def numerics(name, q, k, v, picked, go, scale, ref=True):
    U = new.UnionSelectedAttention

    def unaligned(*a):
        U.ALIGNED = False
        try:
            return U.apply(*a, None)
        finally:
            U.ALIGNED = True

    funcs = {"old": lambda *a: old.TritonSelectedAttention.apply(*a, None),
             "new": lambda *a: U.apply(*a, None), "new_unaligned": unaligned}
    row = {"case": name, "L": q.shape[2], "keys": k.shape[2], "S": picked.shape[1]}
    res = {}
    for n, f in funcs.items():
        try:
            res[n] = run(f, q, k, v, picked, scale, go)
        except Exception as exc:
            row[n + "_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    names = ("out", "dq", "dk", "dv")
    if ref:
        r = run(lambda *a: old.reference(*a), q.float(), k.float(), v.float(), picked, scale, go.float())
        for n in res:
            row[n] = {t: compare(x, y) for t, x, y in zip(names, res[n], r)}
    if "new" in res and "old" in res:
        row["new_vs_old"] = {t: compare(x, y) for t, x, y in zip(names, res["new"], res["old"])}
        rerun = run(funcs["new"], q, k, v, picked, scale, go)
        row["new_bitwise_repeat"] = {t: bool(torch.equal(x, y)) for t, x, y in zip(names, res["new"], rerun)}
    emit(**row)
    return row


def test(dev, tiny):
    scale = 1 / 16.0
    cases = ([("tiny-real", 37, 2100, "real", 32), ("tiny-random", 29, 300, "random", 32), ("tiny-short", 5, 0, "real", 32)]
             if tiny else
             [("real-1536@12k", 1536, 10752, "real", 256), ("random-1024@6k", 1024, 5120, "random", 256),
              ("real-start-700", 700, 0, "real", 256), ("real-odd-301@9k", 301, 8890, "real", 256),
              ("real-1@3k", 1, 3001, "real", 256)])
    for name, length, prefix, kind, dim in cases:
        top = TOP if not tiny else 8
        picked = (picks_real(length, prefix, dev, top=top) if kind == "real" else
                  picks_random(length, prefix, top * RATIO + RATIO - 1, dev))
        q, k, v, go = inputs(length, prefix, dev, dim=dim, dtype=torch.float32 if tiny else torch.bfloat16)
        numerics(name, q, k, v, picked, go, scale)


def timed(f, reps):
    f()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        f()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return round(times[len(times) // 2], 3)


def speed(fn, q, k, v, picked, go, reps):
    qs, ks, vs = (x.detach().clone().requires_grad_() for x in (q, k, v))
    with torch.no_grad():
        fwd = timed(lambda: fn(qs, ks, vs, picked), reps)
    o = fn(qs, ks, vs, picked)
    bwd = timed(lambda: torch.autograd.grad(o, (qs, ks, vs), go, retain_graph=True), reps)
    return {"fwd_ms": fwd, "bwd_ms": bwd, "total_ms": round(fwd + bwd, 3)}


def sweep(dev, scale):
    """Launch configs on 8K queries x 64K keys: forward then backward, per tile layout; the best is kept."""
    U = new.UnionSelectedAttention
    q, k, v, go = inputs(8192, 57344, dev)
    picked = picks_real(8192, 57344, dev)
    emit(union=union_stats(picked), shape=[8192, 57344])
    qs, ks, vs = (x.detach().clone().requires_grad_() for x in (q, k, v))

    def fwd_ms():
        with torch.no_grad():
            return timed(lambda: U.apply(qs, ks, vs, picked, scale, None), 3)

    def bwd_ms():
        o = U.apply(qs, ks, vs, picked, scale, None)
        return timed(lambda: torch.autograd.grad(o, (qs, ks, vs), go, retain_graph=True), 3)

    def trial(label, fn):
        try:
            ms = fn()
        except Exception as exc:
            emit(sweep=label, error=f"{type(exc).__name__}: {str(exc)[:120]}")
            return None
        emit(sweep=label, ms=ms)
        return ms

    grid = [{"BN": bn, "num_warps": w, "num_stages": st} for bn in (32, 64, 128) for w in (4, 8) for st in (1, 2, 3)]
    best = None
    for bm, aligned in ((64, True), (64, False), (128, True)):
        U.BM, U.ALIGNED = bm, aligned
        tiles, qt, nslot = new.tiling(8192, 12, bm, aligned)
        ul, _, off = new.union_tiles(picked, k.shape[2], 12, bm, aligned)
        emit(layout={"BM": bm, "aligned": aligned, "tiles": tiles, "union_keys_per_query": round(ul.numel() * 4 / 8192, 1),
                     "build_ms": timed(lambda: new.union_tiles(picked, k.shape[2], 12, bm, aligned), 3)})
        f_best = b_best = None
        for cfg in grid:
            for extra in ({"tma": True, "ws": True}, {}):
                U.FWD = dict(cfg, **extra)
                ms = trial({"fwd": U.FWD, "BM": bm, "aligned": aligned}, fwd_ms)
                if ms is not None and (f_best is None or ms < f_best[0]):
                    f_best = (ms, dict(U.FWD))
        U.FWD = f_best[1]
        for cfg in grid:
            if cfg["BN"] > 64:
                continue
            U.BWD = dict(cfg)
            ms = trial({"bwd": U.BWD, "BM": bm, "aligned": aligned}, bwd_ms)
            if ms is not None and (b_best is None or ms < b_best[0]):
                b_best = (ms, dict(U.BWD))
        if f_best is None or b_best is None:
            continue
        total = f_best[0] + b_best[0]
        emit(layout_best={"BM": bm, "aligned": aligned, "fwd": f_best, "bwd": b_best, "total": round(total, 3)})
        if best is None or total < best[0]:
            best = (total, bm, aligned, f_best[1], b_best[1])
    _, U.BM, U.ALIGNED, U.FWD, U.BWD = best
    emit(chosen={"BM": U.BM, "ALIGNED": U.ALIGNED, "FWD": U.FWD, "BWD": U.BWD})


def bwd_sweep(dev, reps, tiny=False):
    """MFU-1 (24 Oct, B300): backward launch configs incl. TMA gather / warp specialization (mfu-e swept the backward
    without them, on B200). Per config: ms, dq bitwise vs the current BWD (same BN keeps dq's summation order), dk / dv
    max |diff| vs the current BWD beside the current BWD's own repeat (fp32 atomics: its run-to-run floor)."""
    U = new.UnionSelectedAttention
    base = dict(U.BWD)
    scale = 1 / 16.0
    shapes = ([("tiny", 37, 2100, 32)] if tiny else
              [("win8k@64k", 8192, 57344, 256), ("win24k@124k", 24576, 100000, 256), ("win32k@130k", 32768, 98304, 256)])
    grid = [dict(BN=bn, num_warps=w, num_stages=st, **extra) for bn in (16, 32, 64) for w in (4, 8)
            for st in (1, 2, 3) for extra in ({}, {"tma": True}, {"tma": True, "ws": True})]
    if tiny:
        grid = [g for g in grid if g["num_stages"] == 1 and g["num_warps"] == 4]
    ranked = None
    for index, (name, length, prefix, dim) in enumerate(shapes):
        q, k, v, go = inputs(length, prefix, dev, dim=dim, dtype=torch.float32 if tiny else torch.bfloat16)
        picked = picks_real(length, prefix, dev, top=8 if tiny else TOP)
        qs, ks, vs = (x.detach().clone().requires_grad_() for x in (q, k, v))
        hold = {}

        def setup(cfg):  # the backward's config is read at the forward (ctx.bwd): a fresh forward per config
            U.BWD = dict(cfg)
            hold["o"] = U.apply(qs, ks, vs, picked, scale, None)
            assert hold["o"].grad_fn.bwd == dict(cfg), (hold["o"].grad_fn.bwd, cfg)

        def grads():
            return torch.autograd.grad(hold["o"], (qs, ks, vs), go, retain_graph=True)

        setup(base)
        ref, rep = grads(), grads()
        floor = {t: float((a.float() - b.float()).abs().max()) for t, a, b in zip(("dq", "dk", "dv"), ref, rep)}
        emit(bwd_shape=name, base=base, floor=floor, dq_repeat_bitwise=bool(torch.equal(ref[0], rep[0])))
        rows = []
        for cfg in (grid if index == 0 else [dict(base)] + [r[1] for r in ranked[:8]]):
            try:
                setup(cfg)
                got = grads()
                ms = None if tiny else timed(grads, reps)
            except Exception as exc:  # noqa: BLE001 - an unsupported config
                emit(bwd_shape=name, cfg=cfg, error=f"{type(exc).__name__}: {str(exc)[:160]}")
                continue
            diff = {t: float((a.float() - b.float()).abs().max()) for t, a, b in zip(("dq", "dk", "dv"), got, ref)}
            row = {"bwd_shape": name, "cfg": cfg, "ms": ms, "dq_bitwise": bool(torch.equal(got[0], ref[0])), "diff": diff}
            emit(**row)
            rows.append((ms if ms is not None else 0.0, cfg))
        if index == 0:
            ranked = sorted(rows, key=lambda r: r[0])
        del q, k, v, go, picked, qs, ks, vs, hold
        U.BWD = dict(base)
        if not tiny:
            torch.cuda.empty_cache()
    U.BWD = base


def bwd_bound(dev, reps, tiny=False):
    """The union backward's dk / dv share: the same backward with every key below P0 (no dk / dv products or fp32
    atomics: dq only) against the full one, and the forward for scale. Bounds what an atomic-free dk / dv kernel
    (key-major, deterministic) could save."""
    U = new.UnionSelectedAttention
    scale = 1 / 16.0
    orig = U.backward

    def dq_only(ctx, grad):
        ctx.p0 = ctx.saved_tensors[1].shape[2]
        return orig(ctx, grad)

    shapes = [("tiny", 37, 2100, 32)] if tiny else [("win24k@124k", 24576, 100000, 256), ("win32k@130k", 32768, 98304, 256)]
    for name, length, prefix, dim in shapes:
        q, k, v, go = inputs(length, prefix, dev, dim=dim, dtype=torch.float32 if tiny else torch.bfloat16)
        picked = picks_real(length, prefix, dev, top=8 if tiny else TOP)
        qs, ks, vs = (x.detach().clone().requires_grad_() for x in (q, k, v))
        row = {"bound_shape": name}
        try:
            for label, fn in (("full", orig), ("dq_only", dq_only)):
                U.backward = staticmethod(fn)
                o = U.apply(qs, ks, vs, picked, scale, None)
                grads = lambda: torch.autograd.grad(o, (qs, ks, vs), go, retain_graph=True)  # noqa: E731
                got = grads()
                row[label + "_ms"] = None if tiny else timed(grads, reps)
                row[label + "_dk_zero"] = bool((got[1] == 0).all())
            with torch.no_grad():
                row["fwd_ms"] = None if tiny else timed(lambda: U.apply(qs, ks, vs, picked, scale, None), reps)
        finally:
            U.backward = orig
        emit(**row)
        del q, k, v, go, picked, qs, ks, vs
        if not tiny:
            torch.cuda.empty_cache()


def bench(dev, args):
    scale = 1 / 16.0
    U = new.UnionSelectedAttention
    if args.sweep:
        sweep(dev, scale)
    shapes = [("prefix16k@32k", 16384, 16384), ("prefix16k@98k", 16384, 81920), ("win8k@64k", 8192, 57344),
              ("win24k@124k", 24576, 100000), ("win32k@130k", 32768, 98304)]
    for name, length, prefix in shapes:
        q, k, v, go = inputs(length, prefix, dev)
        picked = picks_real(length, prefix, dev)
        row = {"shape": name, "L": length, "keys": length + prefix, "union": union_stats(picked)}
        row["old"] = speed(lambda *a: old.TritonSelectedAttention.apply(*a, scale, None), q, k, v, picked, go, args.reps)
        row["new"] = speed(lambda *a: U.apply(*a, scale, None), q, k, v, picked, go, args.reps)
        row["speedup"] = {x: round(row["old"][x] / row["new"][x], 2) for x in ("fwd_ms", "bwd_ms", "total_ms")}
        emit(**row)
        if length <= 16384:
            numerics(name, q, k, v, picked, go, scale, ref=False)
        del q, k, v, go, picked
        torch.cuda.empty_cache()
    q, k, v, go = inputs(16384, 16384, dev)
    picked = picks_random(16384, 16384, 2051, dev)
    row = {"shape": "random-16k@32k (no shared picks, not block-aligned)", "union": union_stats(picked)}
    row["old"] = speed(lambda *a: old.TritonSelectedAttention.apply(*a, scale, None), q, k, v, picked, go, args.reps)
    row["new"] = speed(lambda *a: U.apply(*a, scale, None), q, k, v, picked, go, args.reps)
    emit(**row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("test", "bench"), default="test")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--bwd-sweep", action="store_true")
    parser.add_argument("--bwd-bound", action="store_true")
    args = parser.parse_args()
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    dev = "cpu" if interp else "cuda"
    emit(torch=torch.__version__, triton=__import__("triton").__version__, device=dev,
         gpu=None if interp else torch.cuda.get_device_name())
    began = time.time()
    test(dev, tiny=interp)
    if args.bwd_bound:
        bwd_bound(dev, args.reps, tiny=interp)
    if args.bwd_sweep:
        bwd_sweep(dev, args.reps, tiny=interp)
    elif args.mode == "bench":
        bench(dev, args)
    emit(done=True, seconds=round(time.time() - began, 1))


if __name__ == "__main__":
    main()
