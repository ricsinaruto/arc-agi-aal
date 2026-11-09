"""Expert-shard weight loading through the GPU (serving_fast.expert_load_fast; installed as vllm/arc_fast_load.py).

The default safetensors path reads each file through one mmap, single-threaded, and hands every expert tensor to its
weight loader as a view of it: each of Flash's ~295k expert tensors costs a page-faulting, synchronous pageable
host-to-device copy, and a cold container reads the 9P volume at ~1 GiB/s. Here ``ARC_FAST_LOAD_READERS`` threads
(default 16) read whole files ahead of the loader in load order, within ``ARC_FAST_LOAD_BUDGET_GIB`` (default 24) of
reads in flight. An expert-only file (every tensor name contains ``.experts.``) lands in a reused pinned buffer, goes
to the GPU in one copy and its tensors are yielded as views of that device buffer (device-to-device, asynchronous
loader copies); any other file lands in its own host buffer and its tensors are yielded as views of it. Same names,
same order (sorted per file, as ``safe_open.keys()``), same bytes.
"""

import json
import os
import queue
import struct
import threading

import torch

DTYPES = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "U16": torch.uint16,
          "I32": torch.int32, "U32": torch.uint32, "I64": torch.int64, "U64": torch.uint64, "F16": torch.float16,
          "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64,
          "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
STATS = {"files_staged": 0, "bytes_staged": 0, "files_host": 0, "bytes_host": 0, "files_plain": 0, "copied_unaligned": 0,
         "readers": 0, "wait_s": 0.0}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return 8 + n, meta


def kind(path):
    """"gpu" for an expert-only file, "host" for any other file whose dtypes are all known, else "plain" (default
    path)."""
    try:
        _, meta = header(path)
    except (OSError, ValueError):
        return "plain"
    if not meta or any(v["dtype"] not in DTYPES for v in meta.values()):
        return "plain"
    return "gpu" if all(".experts." in k for k in meta) else "host"


def _views(buf, start, meta, names):
    for name in names:
        info = meta[name]
        lo, hi = info["data_offsets"]
        raw = buf[start + lo:start + hi]
        dtype = DTYPES[info["dtype"]]
        size = torch.empty((), dtype=dtype).element_size()
        if (start + lo) % size:  # torch views need aligned storage offsets
            raw = raw.clone()
            STATS["copied_unaligned"] += 1
        yield name, raw.view(dtype).reshape(info["shape"])


def _read_into(path, buf, n):
    with open(path, "rb", buffering=0) as h:
        view, got = memoryview(buf.numpy())[:n], 0
        while got < n:
            k = h.readinto(view[got:])
            if not k:
                raise OSError(f"short read of {path}")
            got += k


def iterate(files, plain, skip=lambda name: False, device=None, progress=lambda it: it):
    """(name, tensor) over ``files`` in order. ``plain(path)`` yields a file's tensors the default way; ``skip(name)``
    drops a tensor before it is read (EP); ``progress`` wraps the file loop (vLLM's tqdm bar)."""
    import collections
    import time
    from concurrent.futures import ThreadPoolExecutor

    files = list(files)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    pin = device.type == "cuda"
    kinds = {f: kind(f) for f in files}
    sizes = {f: os.path.getsize(f) for f in files}
    largest = max((sizes[f] for f in files if kinds[f] == "gpu"), default=0)
    readers = max(1, int(os.environ.get("ARC_FAST_LOAD_READERS", "16")))
    budget = float(os.environ.get("ARC_FAST_LOAD_BUDGET_GIB", "24")) * 2**30
    STATS["readers"] = readers
    pool, made, lock = queue.Queue(), [0], threading.Lock()

    def gpu_buffer():  # at most readers + 2 pinned buffers: one per read in flight and one being uploaded
        try:
            return pool.get_nowait()
        except queue.Empty:
            with lock:
                if made[0] < readers + 2:
                    made[0] += 1
                    return torch.empty(largest, dtype=torch.uint8, pin_memory=pin)
            return pool.get()

    def read(f):
        buf = gpu_buffer() if kinds[f] == "gpu" else torch.empty(sizes[f], dtype=torch.uint8)
        _read_into(f, buf, sizes[f])
        return buf

    ahead = [f for f in files if kinds[f] != "plain"]
    # Readers allocate pinned buffers: bind them to this process's device so no context lands on GPU 0.
    executor = ThreadPoolExecutor(readers, thread_name_prefix="arc-fast-load",
                                  initializer=(lambda: torch.cuda.set_device(device)) if pin else None)
    inflight, nxt, used = collections.deque(), [0], [0]

    def fill():
        while nxt[0] < len(ahead) and len(inflight) <= readers and (
                not inflight or used[0] + sizes[ahead[nxt[0]]] <= budget):
            f = ahead[nxt[0]]
            nxt[0] += 1
            used[0] += sizes[f]
            inflight.append((f, executor.submit(read, f)))

    fill()
    gbuf = torch.empty(largest, dtype=torch.uint8, device=device) if pin and largest else None
    try:
        for f in progress(files):
            if kinds[f] == "plain":
                STATS["files_plain"] += 1
                yield from plain(f)
                continue
            got, fut = inflight.popleft()
            assert got == f, (got, f)
            t = time.perf_counter()
            buf = fut.result()
            STATS["wait_s"] += time.perf_counter() - t
            used[0] -= sizes[f]
            n = sizes[f]
            start, meta = header(f)
            if kinds[f] == "gpu":
                if pin:
                    gbuf[:n].copy_(buf[:n])  # synchronous: the host buffer can be refilled after this
                    src = gbuf
                else:  # CPU (tests): views of a copy
                    src = buf[:n].clone()
                pool.put(buf)
                STATS["files_staged"] += 1
                STATS["bytes_staged"] += n
            else:  # its own host buffer, freed with the last view the loaders let go of
                src = buf
                STATS["files_host"] += 1
                STATS["bytes_host"] += n
            fill()
            # Loader copies from a device ``src`` are queued on this stream before the next file overwrites it.
            yield from _views(src, start, meta, [k for k in sorted(meta) if not skip(k)])
            del src, buf
    finally:
        executor.shutdown(wait=not pin, cancel_futures=True)
        del gbuf
        while not pool.empty():
            pool.get_nowait()
        if pin:
            torch.cuda.empty_cache()
            if hasattr(torch._C, "_host_emptyCache"):  # return the cached pinned buffers to the OS
                torch._C._host_emptyCache()
        STATS["wait_s"] = round(STATS["wait_s"], 2)
        print("ARC_FAST_LOAD " + json.dumps(STATS), flush=True)
