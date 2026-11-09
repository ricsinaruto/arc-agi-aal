"""The trainer's park (real-v5 27 Oct): the frozen base parks into one host snapshot in /dev/shm
shared by the launch's ranks, written once by local rank 0 at the first park, instead of a pinned copy per rank kept
for the run (2 x 60 GiB). Every later park only points the base at the snapshot (no copy); unpark copies it back.
Each rank checks its base equals the snapshot once (the DP ranks load the same model)."""

import hashlib
import json
import mmap
import os
import time
from pathlib import Path

ALIGN = 64


def shm_path():
    """One file per launch: the ranks are torchrun's children (one parent pid)."""
    return Path(os.environ.get("RL1_PARK_SHM_DIR") or "/dev/shm") / f"rl1-base-{os.getppid()}"


def layout_of(tensors):
    """[(shape, dtype, nbytes, offset)] and the total, each slice 64-byte aligned."""
    out, at = [], 0
    for t in tensors:
        n = t.numel() * t.element_size()
        out.append((list(t.shape), str(t.dtype).replace("torch.", ""), n, at))
        at += -(-n // ALIGN) * ALIGN
    return out, at


def digest(layout):
    return hashlib.sha256(json.dumps(layout).encode()).hexdigest()[:16]


def _views(mm, tensors, layout):
    import torch

    return [torch.frombuffer(mm, dtype=torch.uint8, count=n, offset=off).view(t.dtype).view(t.shape) if n else
            t.new_empty(t.shape, device="cpu")
            for t, (_, _, n, off) in zip(tensors, layout)]


def _write(path, tensors, layout, total):
    part = path.with_name(path.name + ".part")
    with open(part, "wb+") as f:
        f.truncate(max(total, 1))
        mm = mmap.mmap(f.fileno(), max(total, 1))
    for view, t in zip(_views(mm, tensors, layout), tensors):
        view.copy_(t)
    mm.flush()
    os.replace(part, path)
    return mm


def _map(path, total):
    with open(path, "rb+") as f:
        return mmap.mmap(f.fileno(), max(total, 1))


def snapshot(tensors, rank, names=None, path=None, timeout=900.0, poll=0.2, check=True):
    """(views, info): CPU views of ``tensors`` (same order, shapes, dtypes) in host snapshots. Rank 0 writes the shared
    one (``path``.part, renamed; ``path``.json lists each tensor by name). Another rank maps every tensor that rank 0
    holds under the same name, shape and dtype and with equal values, and writes only the rest to its own file
    (``path``-r<rank>; real-v6 27 Oct: rank 1's base differed from rank 0's, rank-specific tensors among the base's
    storages). ``info["own"]`` names them."""
    import torch

    path = Path(path or shm_path())
    names = list(names) if names else [str(i) for i in range(len(tensors))]
    layout, total = layout_of(tensors)
    meta = path.with_name(path.name + ".json")
    began = time.time()
    if rank == 0:
        mm = _write(path, tensors, layout, total)
        meta.write_text(json.dumps({"digest": digest(layout), "bytes": total,
                                    "entries": [[n, *row] for n, row in zip(names, layout)]}))
        views = _views(mm, tensors, layout)
        return views, {"path": str(path), "gib": round(total / 2**30, 2), "tensors": len(layout), "rank": 0,
                       "shared": len(layout), "own": [], "own_n": 0, "own_gib": 0.0, "seconds": round(time.time() - began, 1),
                       "maps": [mm]}
    while True:
        try:
            got = json.loads(meta.read_text())
            break
        except (OSError, ValueError):
            if time.time() - began > timeout:
                raise TimeoutError(f"park base_shm: no snapshot at {path} after {timeout:.0f} s")
            time.sleep(poll)
    counts = {}
    for e in got["entries"]:
        counts[e[0]] = counts.get(e[0], 0) + 1
    theirs = {e[0]: e for e in got["entries"] if counts[e[0]] == 1}
    shared_mm = _map(path, int(got["bytes"]))
    views, own = [None] * len(tensors), []
    for i, (t, n) in enumerate(zip(tensors, names)):
        e = theirs.get(n)
        if e is not None and list(e[1]) == list(t.shape) and e[2] == layout[i][1] and n.count("?") == 0:
            v = _views(shared_mm, [t], [(e[1], e[2], e[3], e[4])])[0]
            if not check or torch.equal(v.to(t.device), t):
                views[i] = v
                continue
        own.append(i)
    maps, own_gib = [shared_mm], 0.0
    if own:
        mine = [tensors[i] for i in own]
        lay, tot = layout_of(mine)
        mm = _write(path.with_name(f"{path.name}-r{rank}"), mine, lay, tot)
        for i, v in zip(own, _views(mm, mine, lay)):
            views[i] = v
        maps.append(mm)
        own_gib = round(tot / 2**30, 3)
    return views, {"path": str(path), "gib": round(total / 2**30, 2), "tensors": len(layout), "rank": rank,
                   "shared": len(layout) - len(own), "own": [names[i] for i in own][:40], "own_n": len(own),
                   "own_gib": own_gib, "seconds": round(time.time() - began, 1), "maps": maps}


def checksum(t, chunk=1 << 24):
    """Exact (sum, position-weighted sum) of a tensor's bytes as int32 words (int64 wrap-around), chunked so the
    temporaries stay ~0.4 GiB; computed where the tensor lives."""
    import torch

    if t.numel() == 0:
        return (0, 0)
    b = t.detach().reshape(-1).view(torch.uint8)
    n = b.numel() // 4 * 4
    s, ws = int(b[n:].to(torch.int64).sum()) if n < b.numel() else 0, 0
    w = b[:n].view(torch.int32)
    for i in range(0, w.numel(), chunk):
        x = w[i:i + chunk].to(torch.int64)
        idx = torch.arange(i, i + x.numel(), device=x.device, dtype=torch.int64) % 4093 + 1
        s += int(x.sum())
        ws += int((x * idx).sum())
    return (s, ws)


def reuse(snap, tensors, names, sums, rank, path, n):
    """gate leg 4 (28 Oct): after the first op base_tensors() drops tensors that came to share a storage (1743 ->
    1599), so the layout check failed and every first warm park rewrote the whole base (75-115 s). A tensor whose
    unique name, shape and dtype the snapshot holds, and whose bytes still match the checksum taken at the first
    snapshot (``sums``), keeps its view; only the rest is written (``path``-d<n>-r<rank>). Returns (views, info)."""
    began = time.time()
    count = {}
    for name in snap["names"]:
        count[name] = count.get(name, 0) + 1
    old = {name: v for name, v in zip(snap["names"], snap["views"]) if count[name] == 1 and "?" not in name}
    views, new, used = [None] * len(tensors), [], set()
    for i, (t, name) in enumerate(zip(tensors, names)):
        v = old.get(name)
        if (v is not None and name not in used and list(v.shape) == list(t.shape) and v.dtype == t.dtype
                and name in sums and checksum(t) == sums[name]):
            views[i] = v
            used.add(name)
        else:
            new.append(i)
    maps, new_gib = [], 0.0
    if new:
        mine = [tensors[i] for i in new]
        lay, tot = layout_of(mine)
        mm = _write(Path(path).with_name(f"{Path(path).name}-d{n}-r{rank}"), mine, lay, tot)
        for i, v in zip(new, _views(mm, mine, lay)):
            views[i] = v
        maps.append(mm)
        new_gib = round(tot / 2**30, 3)
    return views, {"path": str(path), "reused": len(tensors) - len(new), "tensors": len(tensors), "rank": rank,
                   "new": [names[i] for i in new][:40], "new_n": len(new), "new_gib": new_gib,
                   "dropped": len(set(old) - used), "seconds": round(time.time() - began, 1), "maps": maps,
                   "reused_names": used}


def register(maps, device=None):
    """mg.park_host.register: cudaHostRegister every snapshot mapping, so the unpark copies DMA from pinned pages
    (no extra host memory: the pages are the snapshot's). Any failure unregisters what was done and reports; the
    park then runs on the pageable path. Returns {"ok", "gib", "seconds", "error"?, "bufs"}."""
    import torch

    began, done = time.time(), []
    try:
        if device is not None:
            torch.cuda.set_device(device)
        cudart = torch.cuda.cudart()
        for mm in maps:
            buf = torch.frombuffer(mm, dtype=torch.uint8)
            err = cudart.cudaHostRegister(buf.data_ptr(), buf.numel(), 0)
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister: error {int(err)}")
            done.append(buf)
    except Exception as exc:  # noqa: BLE001 - the pageable path stays
        unregister(done)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300], "seconds": round(time.time() - began, 1),
                "bufs": []}
    return {"ok": True, "gib": round(sum(b.numel() for b in done) / 2**30, 2), "seconds": round(time.time() - began, 1),
            "bufs": done}


def unregister(bufs):
    import torch

    for buf in bufs:
        try:
            torch.cuda.cudart().cudaHostUnregister(buf.data_ptr())
        except Exception:  # noqa: BLE001
            pass
