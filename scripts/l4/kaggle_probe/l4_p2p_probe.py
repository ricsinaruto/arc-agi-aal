"""l4 arm: Kaggle 4x L4 all-reduce probe v4 (under 10 min, private, no model, no submission).
v3 found SM-driven P2P slow (NCCL over P2P 0.3 GB/s, a symm-mem one-shot hung at 16 rows) while copy-engine
peer copies ran ~10 GB/s. v4 asks whether remote writes are fast, by method, at 1 / 16 / 40 / 64 / 1,600 rows x 2,560 bf16:
  ce_ar  copy-engine push all-reduce: cudaMemcpyAsync of the input into each peer's slot, a 4-byte counter
         copy as the flag, a local spin on the peers' flags, then a local fp32 sum of the 4 slots;
  st_ar  the same with remote stores from SM (a Triton kernel), no remote loads;
  raw    push only (no flag, no wait) for both, and a pull kernel (remote loads, as one-shot) for contrast;
NCCL as served (SHM) is the baseline and the reference sum. CUDA graph and eager timings. Spins give up after
~2 s and say so. Hard timer: results written and the process ended at 420 s. The competition is attached only
because Kaggle gives 4x L4 to its kernels; no competition data is read."""
import ctypes, json, os, threading, time

T0 = time.time()
LIMIT, PER_CONFIG = 420, 110
OUT = "/kaggle/working/probe.json" if os.path.isdir("/kaggle/working") else "probe.json"
HIDDEN, ROWS, MAXROWS = 2560, (1, 16, 40, 64, 1600), 1600
CONFIGS = ("ce", "st", "pull")
result = {"limit_s": LIMIT}


def save():
    with open(OUT + ".part", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(OUT + ".part", OUT)


def emit(key, value):
    result[key] = value
    print("L4PROBE " + json.dumps({key: value}), flush=True)
    save()


def timer():
    time.sleep(max(1, LIMIT - (time.time() - T0)))
    result["hard_stop"] = round(time.time() - T0, 1)
    save()
    print("L4PROBE hard stop", flush=True)
    os._exit(0)


def cudart():
    import glob, sys
    for name in ("libcudart.so.12", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    for d in sys.path:
        for p in glob.glob(os.path.join(d, "nvidia", "cuda_runtime", "lib", "libcudart.so*")):
            return ctypes.CDLL(p)
    raise OSError("libcudart not found")


class Ptr:  # a raw (peer) device pointer Triton accepts as a tensor argument
    def __init__(self, ptr, dtype):
        self.ptr, self.dtype = ptr, dtype

    def data_ptr(self):
        return self.ptr


def kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def push3(src, d0, d1, d2, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(src + offs, mask=m)
        tl.store(d0 + offs, x, mask=m)
        tl.store(d1 + offs, x, mask=m)
        tl.store(d2 + offs, x, mask=m)

    @triton.jit
    def pull3(out, own, s0, s1, s2, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        acc = tl.load(own + offs, mask=m).to(tl.float32)
        acc += tl.load(s0 + offs, mask=m, volatile=True).to(tl.float32)
        acc += tl.load(s1 + offs, mask=m, volatile=True).to(tl.float32)
        acc += tl.load(s2 + offs, mask=m, volatile=True).to(tl.float32)
        tl.store(out + offs, acc.to(tl.bfloat16), mask=m)

    @triton.jit
    def flag3(counter, f0, f1, f2):
        c = tl.load(counter)
        tl.store(f0, c)
        tl.store(f1, c)
        tl.store(f2, c)

    @triton.jit
    def spin(flags, idx, c, status):
        v = tl.load(flags + idx, volatile=True)
        i = 0
        while (v < c) & (i < 2000000):
            v = tl.load(flags + idx, volatile=True)
            i += 1
        if v < c:
            tl.store(status, 1)

    @triton.jit
    def wait3(flags, counter, status, q0, q1, q2):
        c = tl.load(counter)
        spin(flags, q0, c, status)
        spin(flags, q1, c, status)
        spin(flags, q2, c, status)

    return push3, pull3, flag3, wait3


def worker(rank, world, config, port, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NCCL_DEBUG="WARN")
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
    gname = dist.group.WORLD.group_name
    rt = cudart()
    rt.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    push3, pull3, flag3, wait3 = kernels()
    peers = [q for q in range(world) if q != rank]
    dev = f"cuda:{rank}"
    slots = symm_mem.empty(world, MAXROWS, HIDDEN, dtype=torch.bfloat16, device=dev)  # slot q: rank q's input
    flags = symm_mem.empty(world * 4, dtype=torch.int64, device=dev)
    hs, hf = symm_mem.rendezvous(slots, gname), symm_mem.rendezvous(flags, gname)
    try:
        sp, fp = list(hs.buffer_ptrs), list(hf.buffer_ptrs)
    except AttributeError:
        sp = [hs.get_buffer(q, [1], torch.bfloat16).data_ptr() for q in range(world)]
        fp = [hf.get_buffer(q, [1], torch.int64).data_ptr() for q in range(world)]
    slots.zero_(); flags.zero_()
    counter = torch.zeros(1, dtype=torch.int64, device=dev)
    status = torch.zeros(1, dtype=torch.int32, device=dev)
    acc = torch.empty(MAXROWS, HIDDEN, dtype=torch.float32, device=dev)
    torch.cuda.synchronize(); dist.barrier()
    slot_bytes = MAXROWS * HIDDEN * 2

    def ce_copy(dst, src, nbytes):
        err = rt.cudaMemcpyAsync(dst, src, nbytes, 4, torch.cuda.current_stream().cuda_stream)
        if err:
            raise RuntimeError(f"cudaMemcpyAsync error {err}")

    def peer_slot(q, me=rank):  # rank me's slot inside rank q's buffer
        return sp[q] + me * slot_bytes

    def mk(rows, src):
        n = rows * HIDDEN
        nbytes = n * 2
        grid = (triton_cdiv(n, 1024),)
        own = slots[rank, :rows]
        dsts = [Ptr(peer_slot(q), torch.bfloat16) for q in peers]
        srcs = [Ptr(sp[q] + q * slot_bytes, torch.bfloat16) for q in peers]  # peer q's own staged input
        fdst = [Ptr(fp[q] + rank * 8, torch.int64) for q in peers]

        def finish():
            wait3[(1,)](flags, counter, status, *peers)
            torch.sum(slots[:, :rows], 0, dtype=torch.float32, out=acc[:rows])
            return acc[:rows]

        def ce_ar():
            own.copy_(src)
            counter.add_(1)
            for q in peers:
                ce_copy(peer_slot(q), src.data_ptr(), nbytes)
            for q in peers:
                ce_copy(fp[q] + rank * 8, counter.data_ptr(), 8)
            return finish()

        def st_ar():
            own.copy_(src)
            counter.add_(1)
            push3[grid](src, *dsts, n, BLOCK=1024)
            flag3[(1,)](counter, *fdst)
            return finish()

        def ce_raw():
            for q in peers:
                ce_copy(peer_slot(q), src.data_ptr(), nbytes)

        def st_raw():
            push3[grid](src, *dsts, n, BLOCK=1024)

        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)

        def pull_raw():
            pull3[grid](out, own, *srcs, n, BLOCK=1024)

        return {"ce": (ce_ar, ce_raw), "st": (st_ar, st_raw), "pull": (None, pull_raw)}[config], out

    def timed(fn, calls=20, reps=8):
        torch.cuda.synchronize()
        eager = []
        for _ in range(20):
            dist.barrier()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); e.synchronize()
            eager.append(s.elapsed_time(e) * 1000)
        eager.sort()
        row = {"eager_us": round(eager[len(eager) // 2], 1)}
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize(); dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(calls):
                    fn()
            dist.barrier()
            g.replay(); torch.cuda.synchronize()
            gt = []
            for _ in range(reps):
                dist.barrier()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); g.replay(); e.record(); e.synchronize()
                gt.append(s.elapsed_time(e) * 1000 / calls)
            gt.sort()
            row["graph_us"] = round(gt[len(gt) // 2], 1)
        except Exception as exc:  # noqa: BLE001
            row["graph_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return row

    try:
        gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
        for rows in ROWS:
            if config == "pull" and rows > 64:
                continue
            src = torch.randn(rows, HIDDEN, device=dev, generator=gen).to(torch.bfloat16)
            ref = src.clone(); dist.all_reduce(ref)
            x = src.clone()
            row = {"nccl": timed(lambda: dist.all_reduce(x))} if config == "ce" else {}
            (ar, raw), out = mk(rows, src)
            if ar is not None:
                status.zero_(); torch.cuda.synchronize(); dist.barrier()
                got = ar(); torch.cuda.synchronize()
                d = (got - ref.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-9)
                row["ar"] = {**timed(ar), "max_rel_vs_nccl": round(d, 6), "spin_timeout": int(status.item())}
            else:  # pull needs every rank's input staged in its own slot first
                slots[rank, :rows].copy_(src); torch.cuda.synchronize(); dist.barrier()
                raw(); torch.cuda.synchronize()
                d = (out.float() - ref.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-9)
                row["pull_check_rel"] = round(d, 6)
            r = timed(raw)
            us = r.get("graph_us") or r["eager_us"]
            r["write_GBps" if config != "pull" else "read_GBps"] = round(3 * rows * HIDDEN * 2 / us / 1e3, 2)
            row["raw"] = r
            if rank == 0:
                queue.put(("row", rows, row))
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            queue.put(("error", None, f"{type(exc).__name__}: {str(exc)[:300]}"))
    finally:
        if rank == 0:
            queue.put(("done", None, None))
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def triton_cdiv(a, b):
    return (a + b - 1) // b


def run(config, port):
    import torch
    import torch.multiprocessing as mp

    world = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, config, port, queue)) for r in range(world)]
    for p in procs:
        p.start()
    found = {}
    end = time.time() + PER_CONFIG
    while time.time() < end:
        try:
            kind, rows, value = queue.get(timeout=max(1, end - time.time()))
        except Exception:  # noqa: BLE001
            found["error"] = found.get("error") or "timeout"
            break
        if kind == "done":
            break
        if kind == "row":
            found[str(rows)] = value
            emit(config, found)
        else:
            found[kind] = value
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    emit(config, found)


def main():
    threading.Thread(target=timer, daemon=True).start()
    import torch
    import triton

    emit("torch", {"version": torch.__version__, "triton": triton.__version__,
                   "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                   "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    for k, config in enumerate(CONFIGS):
        if time.time() - T0 > LIMIT - 60:
            emit(config, "skipped: time")
            continue
        run(config, 29700 + k)
    emit("elapsed_s", round(time.time() - T0, 1))
    os._exit(0)


if __name__ == "__main__":
    main()
