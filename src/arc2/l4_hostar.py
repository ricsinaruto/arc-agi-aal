"""An all-reduce through pinned host memory for four L4s on PCIe, against NCCL (the server's path; 108 us a call,
96 calls a decode step, about 10 ms of a 56 ms step at 32 rows) and against vLLM's own one-shot kernel forced
on over PCIe peer access. ``python -m arc2.l4_hostar --out file.json`` in the serving runtime, cards free.

The host path (``HostAllReduce``): one shared page-locked buffer that every card maps (``cudaHostRegister`` of
a POSIX shared-memory segment; the device address is the host address under UVA). One Triton kernel a call,
in ``PROGRAMS`` programs, each owning a chunk: write the chunk of this rank's input to its slot, raise the
chunk's flag (system-scope release), wait for the three other ranks' flags of that chunk (system-scope acquire),
then sum the four slots in rank order in FP32, so every rank gets the same bits (TP ranks must agree). The call
count lives in device memory (no host argument changes between calls, so the kernel replays inside a CUDA
graph); slots and flags alternate between two parities, which keeps a fast rank from overwriting data a slow
rank still reads.
"""

import argparse
import json
import os
import time

RANKS, HIDDEN, PROGRAMS, BLOCK = 4, 2560, 64, 512
ROWS = (1, 8, 16, 24, 32, 40, 48, 64, 256, 1600, 2048)
# 13 Oct: NCCL picked RING_LL for the 1,600-row prefill chunk (2.6 ms a call, 44% of a prefill step, depthprof-v26d)
VARIANTS = {"default": {}, "simple": {"NCCL_PROTO": "Simple"}, "ll128": {"NCCL_PROTO": "LL128"},
            "ll": {"NCCL_PROTO": "LL"}, "simple_tree": {"NCCL_PROTO": "Simple", "NCCL_ALGO": "Tree"},
            # l4 arm, 27 Oct: Kaggle's 4x L4 have peer access on every pair (PHB), but NCCL 2.27 picks SHM there
            "p2p_sys": {"NCCL_P2P_LEVEL": "SYS"}, "p2p_phb": {"NCCL_P2P_LEVEL": "PHB"}}
MAX_ROWS = 4096


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def hostar(inp, out, slots, flags, counters, N, CHUNK: tl.constexpr, RANKS: tl.constexpr, RANK: tl.constexpr,
               BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        programs = tl.num_programs(0)
        count = tl.load(counters + pid)
        parity = count % 2
        start = pid * CHUNK
        own = slots + (parity * RANKS + RANK) * N
        for off in range(0, CHUNK, BLOCK):
            idx = start + off + tl.arange(0, BLOCK)
            mask = (idx < N) & (off + tl.arange(0, BLOCK) < CHUNK)
            tl.store(own + idx, tl.load(inp + idx, mask=mask), mask=mask)
        tl.debug_barrier()
        base = flags + parity * RANKS * programs
        tl.atomic_xchg(base + RANK * programs + pid, count + 1, sem="release", scope="sys")
        for r in tl.static_range(RANKS):
            if r != RANK:
                seen = tl.atomic_add(base + r * programs + pid, 0, sem="acquire", scope="sys")
                while seen < count + 1:
                    seen = tl.atomic_add(base + r * programs + pid, 0, sem="acquire", scope="sys")
        tl.debug_barrier()
        for off in range(0, CHUNK, BLOCK):
            idx = start + off + tl.arange(0, BLOCK)
            mask = (idx < N) & (off + tl.arange(0, BLOCK) < CHUNK)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for r in tl.static_range(RANKS):  # rank order on every rank: the same bits everywhere
                if r == RANK:
                    acc += tl.load(inp + idx, mask=mask, other=0.0).to(tl.float32)
                else:
                    acc += tl.load(slots + (parity * RANKS + r) * N + idx, mask=mask, other=0.0,
                                   volatile=True).to(tl.float32)
            tl.store(out + idx, acc.to(out.dtype.element_ty), mask=mask)
        tl.store(counters + pid, count + 1)

    return hostar


class Pointer:
    """What Triton needs of a tensor argument: an address and an element type (host memory under UVA)."""

    def __init__(self, address, dtype):
        self.address, self.dtype = int(address), dtype

    def data_ptr(self):
        return self.address


class HostAllReduce:
    """The host path for one rank; ``name`` is the shared segment all ranks open (rank 0 creates it)."""

    def __init__(self, rank, name, device, max_elements=MAX_ROWS * HIDDEN, create=None, barrier=None):
        from multiprocessing import shared_memory

        import torch

        self.rank, self.device, self.max = rank, device, int(max_elements)
        slot_bytes = 2 * RANKS * self.max * 2  # two parities, four ranks, BF16
        flag_bytes = 2 * RANKS * PROGRAMS * 4
        size = slot_bytes + flag_bytes
        create = rank == 0 if create is None else create
        if create:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            self.shm.buf[:size] = bytes(size)
        if barrier:
            barrier()
        if not create:
            self.shm = shared_memory.SharedMemory(name=name)
        import ctypes

        address = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        code = torch.cuda.cudart().cudaHostRegister(address, size, 1 | 2)  # portable | mapped
        if int(code) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {code}")
        self.address, self.size = address, size
        self.slots = Pointer(address, torch.bfloat16)
        self.flags = Pointer(address + slot_bytes, torch.int32)
        self.counters = torch.zeros(PROGRAMS, dtype=torch.int32, device=device)
        self.kernel = kernel()

    def __call__(self, x, out=None):
        n = x.numel()
        if n > self.max or x.dtype.itemsize != 2:
            raise ValueError("too large for the host buffer, or not 16-bit")
        out = torch_empty_like(x) if out is None else out
        chunk = -(-n // PROGRAMS)
        chunk = -(-chunk // BLOCK) * BLOCK
        self.kernel[(PROGRAMS,)](x, out, self.slots, self.flags, self.counters, n, CHUNK=chunk, RANKS=RANKS,
                                 RANK=self.rank, BLOCK=BLOCK)
        return out


def torch_empty_like(x):
    import torch

    return torch.empty_like(x)


def timed(step, device, calls, graph_calls=48):
    """Median eager latency of a call (with a synchronize), and the per-call time of ``graph_calls`` calls
    replayed from one CUDA graph (what a decode step pays: no launch from the host)."""
    import torch

    for _ in range(20):
        step()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(calls):
        begun = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - begun)
    times.sort()
    out = {"median_us": round(times[len(times) // 2] * 1e6, 1), "p90_us": round(times[int(len(times) * 0.9)] * 1e6, 1)}
    try:
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            step()
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(graph_calls):
                    step()
        torch.cuda.synchronize(device)
        reps = []
        for _ in range(20):
            begun = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize(device)
            reps.append(time.perf_counter() - begun)
        reps.sort()
        out["graph_us_a_call"] = round(reps[len(reps) // 2] * 1e6 / graph_calls, 1)
    except Exception as exc:  # noqa: BLE001 - a path that cannot be captured says so
        out["graph_error"] = repr(exc)[:300]
    return out


QUANT_PATHS = ("q8", "f8", "rs8")  # 14 Oct: fewer bytes on the wire (all-reduce is 44% of a prefill-mixed step)
QBLOCK = 128  # elements a scale


def quantized(kind, group, device):
    """A reduced-precision all-reduce on NCCL (torch.distributed), every rank the same bits:
    q8 / f8: int8 / FP8-e4m3 with one FP32 scale per 128 elements, all-to-all of the quantized shards, each rank sums
    its shard in FP32 in rank order, then an all-gather of the quantized sums (1.5 bytes an element on the wire
    against 3 for the BF16 ring, plus scales); rs8: BF16 reduce-scatter, int8 all-gather (2.25 bytes)."""
    import torch
    import torch.distributed as dist

    fp8 = kind == "f8"
    top = 448.0 if fp8 else 127.0

    def quant(x):
        xf = x.float().view(-1, QBLOCK)
        scale = xf.abs().amax(1, keepdim=True).clamp(min=1e-30) / top
        y = xf / scale
        q = y.to(torch.float8_e4m3fn).view(torch.uint8) if fp8 else y.round().clamp(-127, 127).to(torch.int8).view(torch.uint8)
        return q.reshape(-1), scale.reshape(-1)

    def dequant(q, scale):
        v = q.view(torch.float8_e4m3fn).float() if fp8 else q.view(torch.int8).float()
        return (v.view(-1, QBLOCK) * scale.view(-1, 1)).reshape(-1)

    def gather(shard):
        q, sc = quant(shard)
        qs = torch.empty(RANKS * q.numel(), dtype=torch.uint8, device=device)
        ss = torch.empty(RANKS * sc.numel(), dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(qs, q, group=group)
        dist.all_gather_into_tensor(ss, sc, group=group)
        return dequant(qs, ss)

    def reduce(x):
        n = x.numel()
        if n % (RANKS * QBLOCK):
            return None
        if kind == "rs8":
            shard = torch.empty(n // RANKS, dtype=x.dtype, device=device)
            dist.reduce_scatter_tensor(shard, x.reshape(-1), group=group)
            return gather(shard.float()).to(x.dtype).view_as(x)
        q, sc = quant(x)
        qr, sr = torch.empty_like(q), torch.empty_like(sc)
        dist.all_to_all_single(qr, q, group=group)
        dist.all_to_all_single(sr, sc, group=group)
        parts = dequant(qr, sr).view(RANKS, -1)
        acc = parts[0].clone()
        for r in range(1, RANKS):  # rank order on every rank
            acc += parts[r]
        return gather(acc).to(x.dtype).view_as(x)

    return reduce


def worker(rank, port, name, calls, queue, paths, env=None):
    os.environ.update(env or {})  # before NCCL starts
    os.environ.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")  # as every L4 start (610 hosts)
    out = {}
    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=RANKS)
        group = dist.group.WORLD
        reducers = {}
        if "nccl" in paths:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            comm = PyNcclCommunicator(group=group, device=rank)
            reducers["nccl"] = comm.all_reduce
        if "host" in paths:
            try:
                host = HostAllReduce(rank, name, device, barrier=lambda: dist.barrier(group))
                reducers["host"] = host
            except Exception as exc:  # noqa: BLE001
                out["host_error"] = repr(exc)[:400]
                dist.barrier(group)
        if "custom" in paths:  # vLLM's one-shot kernel over PCIe peer access, its topology check bypassed
            try:
                from vllm.distributed.device_communicators import custom_all_reduce as car
                from vllm.platforms import current_platform

                current_platform.is_fully_connected = lambda ids: True
                ca = car.CustomAllreduce(group=group, device=device)
                if ca.disabled:
                    raise RuntimeError("custom all-reduce disabled (peer access or library)")
                ca.fully_connected = True
                reducers["custom"] = lambda x: ca.custom_all_reduce(x) if ca.should_custom_ar(x) else None
                out["custom_max_bytes"] = ca.max_size
            except Exception as exc:  # noqa: BLE001
                out["custom_error"] = repr(exc)[:400]
        if any(p in paths for p in QUANT_PATHS):
            ngroup = dist.new_group(backend="nccl")
            for p in QUANT_PATHS:
                if p in paths:
                    reducers[p] = quantized(p, ngroup, device)
        torch.manual_seed(1)
        for path, reduce in reducers.items():
            rows_out = {}
            for rows in ROWS:
                x = (torch.randn(rows, HIDDEN, device=device) * (rank + 1)).to(torch.bfloat16)
                try:
                    got = reduce(x)
                    if got is None:
                        rows_out[str(rows)] = {"skipped": "size"}
                        continue
                    torch.cuda.synchronize(device)
                    # the same input on every rank but scaled: the sum is x / (rank + 1) * 10
                    base = x.float() / (rank + 1)
                    expect = base * sum(r + 1 for r in range(RANKS))
                    row = {"max_rel_error": round(float((got.float() - expect).abs().max() / expect.abs().max()), 5),
                           "rms_rel_error": round(float((got.float() - expect).pow(2).mean().sqrt() / expect.pow(2).mean().sqrt()), 5)}
                    row.update(timed(lambda x=x: reduce(x), device, calls))
                    rows_out[str(rows)] = row
                except Exception as exc:  # noqa: BLE001
                    rows_out[str(rows)] = {"error": repr(exc)[:300]}
                    break
            out[path] = rows_out
            dist.barrier(group)
        # the bits must agree between ranks (TP replicas): a checksum of one large reduction per path
        sums = {}
        for path, reduce in reducers.items():
            x = (torch.randn(256, HIDDEN, device=device, generator=torch.Generator(device).manual_seed(5 + rank))
                 ).to(torch.bfloat16)
            got = reduce(x)
            if got is not None:
                sums[path] = float(got.float().sum().item()), int(got.view(torch.int16).long().sum().item())
        out["checksums"] = sums
    except BaseException as exc:  # noqa: BLE001
        out["error"] = repr(exc)[:500]
    queue.put((rank, out))
    queue.close()
    queue.join_thread()
    os._exit(0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--calls", type=int, default=400)
    parser.add_argument("--port", type=int, default=29717)
    parser.add_argument("--paths", default="nccl,host,custom")
    parser.add_argument("--wait", type=int, default=600)
    parser.add_argument("--variants", default="", help="NCCL settings (VARIANTS names) for the nccl path, each its own start")
    args = parser.parse_args(argv)
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")

    def measure(paths, port, env=None):
        name, queue = f"arc2hostar{os.getpid()}p{port}", context.Queue()
        jobs = [context.Process(target=worker, args=(rank, port, name, args.calls, queue, paths, env))
                for rank in range(RANKS)]
        for job in jobs:
            job.start()
        rows = {}
        try:
            for _ in jobs:
                rank, out = queue.get(timeout=args.wait)
                rows[str(rank)] = out
        except Exception as exc:  # noqa: BLE001
            rows["missing"] = repr(exc)[:200]
        for job in jobs:
            job.join(timeout=10)
            if job.is_alive():
                job.kill()
        try:
            from multiprocessing import shared_memory

            shared_memory.SharedMemory(name=name).unlink()
        except Exception:  # noqa: BLE001
            pass
        return rows

    paths = tuple(args.paths.split(","))
    import torch

    peers = [[torch.cuda.can_device_access_peer(i, j) if i != j else None for j in range(RANKS)] for i in range(RANKS)]
    print("ARC2_L4_PEERS " + json.dumps(peers), flush=True)
    rows = measure(paths, args.port)
    agree = {path: len({tuple(r.get("checksums", {}).get(path) or ()) for r in rows.values()
                        if isinstance(r, dict)}) == 1 for path in paths}
    record = {"peers": peers, "ranks": RANKS, "hidden": HIDDEN, "programs": PROGRAMS, "rows": ROWS, "by_rank": rows,
              "ranks_agree": agree, "rank0": rows.get("0")}
    for k, variant in enumerate(v for v in args.variants.split(",") if v):
        found = measure(("nccl",), args.port + 1 + k, VARIANTS[variant])
        record.setdefault("variants", {})[variant] = {"env": VARIANTS[variant], "rank0": (found.get("0") or {}).get("nccl"),
                                                      "error": found.get("missing") or (found.get("0") or {}).get("error")}
        with open(args.out, "w") as stream:
            json.dump(record, stream, indent=1)
    with open(args.out, "w") as stream:
        json.dump(record, stream, indent=1)
    summary = {path: {r: (v.get("median_us"), v.get("graph_us_a_call"), v.get("max_rel_error"))
                      for r, v in (rows.get("0", {}).get(path) or {}).items() if isinstance(v, dict)}
               for path in paths}
    summary.update({f"nccl_{v}": {r: (x.get("median_us"), x.get("graph_us_a_call")) for r, x in (row["rank0"] or {}).items()
                                  if isinstance(x, dict)} for v, row in record.get("variants", {}).items()})
    print("ARC2_L4_HOSTAR " + json.dumps({"summary": summary, "agree": agree,
                                          "errors": {k: v for k, v in (rows.get("0") or {}).items()
                                                     if k.endswith("error")}}), flush=True)


if __name__ == "__main__":
    main()
