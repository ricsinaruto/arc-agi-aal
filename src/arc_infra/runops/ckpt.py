"""Checkpoint I/O with bounded host memory (infra stage 3; c7b died copying 443 GiB to host at its first checkpoint).

``save(tree, path)`` streams every tensor device -> a small reused pinned ring (default 4 x 64 MiB) -> a file, chunk
by chunk; host RAM in use is the ring, never a copy of the state. The call blocks until the bytes are in the file
(local NVMe: seconds per 10 GiB; the tensors must not change meanwhile). With ``stage_dir`` the file is written to
local disk first and copied to ``path`` (a volume) by a background thread through the page cache, which the kernel
can reclaim; ``handle.wait()`` joins that copy.

File: safetensors layout (8-byte little-endian header length, JSON header, raw bytes), readable by
``safetensors.safe_open`` for the tensors. ``__metadata__`` holds the tree (dicts with any key types, lists, tuples,
scalars; tensors by key) and a checksum pair per tensor taken on the device at save time. ``load`` rebuilds the tree
(mmap by default) and also reads torch.save zip files, so readers switch from ``torch.load`` with one line.
``verify`` re-checks the checksums from the file; ``compare`` is the save/reload exactness check against live tensors.
Torch is imported lazily: the module imports without it.
"""
import base64
import json
import mmap
import os
import pickle
import queue
import shutil
import struct
import threading
import time
from pathlib import Path

FORMAT = "runops-ckpt-1"
MOD = 65521  # checksum position weight period
_SUMW = 16  # hex digits per checksum half (fixed width: the header length is known before the data)

_ST = {"float64": "F64", "float32": "F32", "float16": "F16", "bfloat16": "BF16", "int64": "I64", "int32": "I32",
       "int16": "I16", "int8": "I8", "uint8": "U8", "bool": "BOOL", "float8_e4m3fn": "F8_E4M3",
       "float8_e5m2": "F8_E5M2", "uint16": "U16", "uint32": "U32", "uint64": "U64"}


def _torch():
    import torch

    return torch


# --- tree <-> JSON skeleton --------------------------------------------------------------------------------------
def _flatten(tree):
    torch = _torch()
    tensors = []

    def enc(x):
        if isinstance(x, torch.Tensor):
            local = x.to_local() if hasattr(x, "to_local") else x
            key = f"t{len(tensors)}"
            tensors.append((key, local.detach()))
            return {"__t__": key}
        if isinstance(x, torch.Size):
            return {"__size__": list(x)}
        if isinstance(x, dict):
            if all(isinstance(k, str) and not k.startswith("__") for k in x):
                return {k: enc(v) for k, v in x.items()}
            return {"__d__": [[enc(k), enc(v)] for k, v in x.items()]}
        if isinstance(x, tuple):
            return {"__tuple__": [enc(v) for v in x]}
        if isinstance(x, list):
            return [enc(v) for v in x]
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        return {"__pickle__": base64.b64encode(pickle.dumps(x)).decode()}

    return enc(tree), tensors


def _unflatten(skel, tensors):
    torch = None

    def dec(x):
        nonlocal torch
        if isinstance(x, list):
            return [dec(v) for v in x]
        if not isinstance(x, dict):
            return x
        if "__t__" in x:
            return tensors[x["__t__"]]
        if "__size__" in x:
            torch = torch or _torch()
            return torch.Size(x["__size__"])
        if "__d__" in x:
            return {_hashable(dec(k)): dec(v) for k, v in x["__d__"]}
        if "__tuple__" in x:
            return tuple(dec(v) for v in x["__tuple__"])
        if "__pickle__" in x:
            return pickle.loads(base64.b64decode(x["__pickle__"]))
        return {k: dec(v) for k, v in x.items()}

    return dec(skel)


def _hashable(k):
    return tuple(k) if isinstance(k, list) else k


# --- checksums (device side and file side compute the same pair, mod 2^64) ----------------------------------------
class _Sum:
    """Over a tensor's byte stream: s1 = sum of little-endian int32 words (+ tail bytes), s2 = the same weighted by
    (position mod 65521) + 1. Accumulated on the tensor's device; read once per tensor."""

    def __init__(self, device, nbytes):
        torch = _torch()
        self.torch, self.words = torch, nbytes // 4
        self.s1 = torch.zeros((), dtype=torch.int64, device=device)
        self.s2 = torch.zeros((), dtype=torch.int64, device=device)

    def add(self, chunk, start):  # chunk: uint8 view; start: its byte offset in the tensor (a multiple of 4)
        torch = self.torch
        nw = chunk.numel() // 4
        if nw:
            w = chunk[: nw * 4].view(torch.int32).to(torch.int64)
            pos = torch.arange(start // 4, start // 4 + nw, device=chunk.device, dtype=torch.int64) % MOD + 1
            self.s1 += w.sum()
            self.s2 += (w * pos).sum()
        tail = chunk[nw * 4:]
        if tail.numel():
            t = tail.to(torch.int64)
            pos = torch.arange(self.words, self.words + t.numel(), device=chunk.device, dtype=torch.int64) % MOD + 1
            self.s1 += t.sum()
            self.s2 += (t * pos).sum()

    def hex(self):
        return "".join(f"{int(v.item()) & (2**64 - 1):0{_SUMW}x}" for v in (self.s1, self.s2))


def _sums_np(buf):
    import numpy as np

    n = len(buf)
    nw = n // 4
    s1 = s2 = np.int64(0)
    step = 1 << 24
    with np.errstate(over="ignore"):
        for a in range(0, nw, step):
            w = np.frombuffer(buf, dtype="<i4", count=min(step, nw - a), offset=a * 4).astype(np.int64)
            pos = np.arange(a, a + len(w), dtype=np.int64) % MOD + 1
            s1 = s1 + w.sum()
            s2 = s2 + (w * pos).sum()
        tail = np.frombuffer(buf, dtype=np.uint8, count=n - nw * 4, offset=nw * 4).astype(np.int64)
        if len(tail):
            pos = np.arange(nw, nw + len(tail), dtype=np.int64) % MOD + 1
            s1 = s1 + tail.sum()
            s2 = s2 + (tail * pos).sum()
    return "".join(f"{int(v) & (2**64 - 1):0{_SUMW}x}" for v in (s1, s2))


# --- the pinned ring ----------------------------------------------------------------------------------------------
class Ring:
    """``slots`` host buffers of ``slot_mib`` (pinned when CUDA is up), allocated on first use and reused by every
    save in the process. 64 MiB is a power of two: the caching host allocator does not round it up."""

    def __init__(self, slots=4, slot_mib=64):
        self.n, self.slot_bytes = slots, slot_mib << 20
        self.free = None

    def _ensure(self):
        if self.free is None:
            torch = _torch()
            pin = torch.cuda.is_available()
            self.free = queue.Queue()
            for _ in range(self.n):
                self.free.put(torch.empty(self.slot_bytes, dtype=torch.uint8, pin_memory=pin))
        return self.free

    @property
    def host_gib(self):
        return self.n * self.slot_bytes / 2**30


_RING = Ring()


class SaveHandle:
    def __init__(self, path, nbytes, blocking_s):
        self.path, self.nbytes, self.blocking_s = Path(path), nbytes, blocking_s
        self.copy_s = None
        self._thread, self._error = None, None

    def wait(self):
        """Joins the background copy (``stage_dir``); re-raises its error."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise RuntimeError(f"checkpoint copy to {self.path} failed") from self._error
        return self


def _header(entries, skeleton, sums):
    meta = {"format": FORMAT, "tree": json.dumps(skeleton), "sums": json.dumps(sums)}
    raw = json.dumps({"__metadata__": meta, **entries}, separators=(",", ":")).encode()
    return raw + b" " * (-len(raw) % 8)


def save(tree, path, *, stage_dir=None, ring=None, after=None):
    """Write ``tree`` to ``path``; returns a SaveHandle once the bytes are on disk (``stage_dir``: in the local
    file; the copy to ``path`` runs on, then ``after()``, e.g. a volume commit). Non-contiguous tensors are made
    contiguous on their device first (one tensor at a time)."""
    torch = _torch()
    began = time.time()
    ring = ring or _RING
    path = Path(path)
    skeleton, items = _flatten(tree)
    items = sorted(items, key=lambda kv: -kv[1].element_size())  # keeps every tensor aligned to its element size
    entries, off = {}, 0
    for key, t in items:
        n = t.numel() * t.element_size()
        entries[key] = {"dtype": _ST[str(t.dtype).removeprefix("torch.")], "shape": list(t.shape),
                        "data_offsets": [off, off + n]}
        off += n
    placeholder = {key: "0" * (2 * _SUMW) for key, _ in items}
    head_len = len(_header(entries, skeleton, placeholder))
    local = Path(stage_dir) / path.name if stage_dir else path
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".tmp")
    free = ring._ensure()
    work = queue.Queue(maxsize=ring.n)
    errors = []
    cuda = torch.cuda.is_available()
    side = torch.cuda.Stream() if cuda else None
    if cuda:
        side.wait_stream(torch.cuda.current_stream())  # the producers of every tensor have finished

    with open(tmp, "wb") as f:
        f.seek(8 + head_len)

        def writer():
            while True:
                job = work.get()
                if job is None:
                    return
                event, slot, n, _keep = job
                try:
                    if event is not None:
                        event.synchronize()
                    if not errors:
                        f.write(memoryview(slot.numpy())[:n])
                except Exception as e:  # noqa: BLE001 - raised in the caller
                    errors.append(e)
                finally:
                    free.put(slot)

        thread = threading.Thread(target=writer, name="runops-ckpt-writer", daemon=True)
        thread.start()
        sums = {}
        try:
            for key, t in items:
                aligned = (t.storage_offset() * t.element_size()) % 8 == 0
                src = t if t.is_contiguous() and aligned else t.clone(memory_format=torch.contiguous_format)
                n = src.numel() * src.element_size()
                acc = _Sum(src.device, n)
                flat = src.reshape(-1).view(torch.uint8) if n else None
                for start in range(0, n, ring.slot_bytes):
                    if errors:
                        raise errors[0]
                    chunk = flat[start:start + ring.slot_bytes]
                    acc.add(chunk, start)
                    slot = free.get()
                    m = chunk.numel()
                    if src.is_cuda:
                        with torch.cuda.stream(side):
                            slot[:m].copy_(chunk, non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(side)
                    else:
                        slot[:m].copy_(chunk)
                        event = None
                    work.put((event, slot, m, src))
                sums[key] = acc.hex()
        finally:
            work.put(None)
            thread.join()
        if errors:
            raise errors[0]
        head = _header(entries, skeleton, sums)
        assert len(head) == head_len, "checkpoint header length changed"
        f.seek(0)
        f.write(struct.pack("<Q", head_len) + head)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, local)
    handle = SaveHandle(path, off, round(time.time() - began, 2))
    if stage_dir:
        def copy():
            t0 = time.time()
            try:
                part = path.with_name(path.name + ".tmp")
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, part)
                os.replace(part, path)
                local.unlink()
                handle.copy_s = round(time.time() - t0, 2)
                if after:
                    after()
            except Exception as e:  # noqa: BLE001 - re-raised by wait()
                handle._error = e

        handle._thread = threading.Thread(target=copy, name="runops-ckpt-copy", daemon=False)
        handle._thread.start()
    elif after:
        after()
    return handle


# --- reading --------------------------------------------------------------------------------------------------------
def _read_header(f):
    (n,) = struct.unpack("<Q", f.read(8))
    head = json.loads(f.read(n))
    return n, head.pop("__metadata__", {}) or {}, head


def is_runops(path):
    with open(path, "rb") as f:
        return f.read(2) != b"PK"


def load(path, map_location="cpu", mmap_=True):
    """The saved tree. torch.save zip files go to ``torch.load`` (weights_only=False: our own files)."""
    torch = _torch()
    if not is_runops(path):
        return torch.load(path, map_location=map_location, weights_only=False, mmap=mmap_)
    inv = {v: k for k, v in _ST.items()}
    with open(path, "rb") as f:
        n, meta, entries = _read_header(f)
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file ({meta.get('format')})")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) if os.path.getsize(path) > 8 + n else None
    base = 8 + n
    tensors = {}
    for key, e in entries.items():
        dtype = getattr(torch, inv[e["dtype"]])
        b, end = e["data_offsets"]
        numel = 1
        for s in e["shape"]:
            numel *= s
        if numel == 0:
            t = torch.empty(e["shape"], dtype=dtype)
        else:
            t = torch.frombuffer(mm, dtype=dtype, count=numel, offset=base + b).reshape(e["shape"])
            if not mmap_:
                t = t.clone()
        tensors[key] = t if map_location in (None, "cpu") else t.to(map_location)
    return _unflatten(json.loads(meta["tree"]), tensors)


def verify(path):
    """Keys whose bytes in the file no longer match the save-time device checksums ([] = intact)."""
    with open(path, "rb") as f:
        n, meta, entries = _read_header(f)
        sums = json.loads(meta["sums"])
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if os.path.getsize(path) > 8 + n else b""
    bad = []
    for key, e in entries.items():
        b, end = e["data_offsets"]
        if _sums_np(memoryview(mm)[8 + n + b: 8 + n + end]) != sums[key]:
            bad.append(key)
    return bad


def compare(path, tree):
    """Save/reload exactness: [(tree path, reason)] where the file differs from ``tree`` ([] = bit-exact)."""
    torch = _torch()
    saved = load(path)
    diffs = []

    def walk(a, b, at):
        if isinstance(a, torch.Tensor):
            a = (a.to_local() if hasattr(a, "to_local") else a).detach()
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                diffs.append((at, "shape/dtype"))
            elif a.numel() and not torch.equal(a.cpu().contiguous().reshape(-1).view(torch.uint8),
                                               b.contiguous().reshape(-1).view(torch.uint8)):
                diffs.append((at, "bytes"))
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                diffs.append((at, "keys"))
            else:
                for k in a:
                    walk(a[k], b[k], f"{at}/{k}")
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)) or len(a) != len(b):
                diffs.append((at, "length"))
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{at}/{i}")
        elif a != b and not (isinstance(a, float) and a != a and b != b):
            diffs.append((at, f"{a!r} != {b!r}"))

    walk(tree, saved, "")
    return diffs
