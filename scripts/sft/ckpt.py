"""SFT trainer checkpoints: <out>/ckpt-<step>/ holds student/ (the trained tensors as safetensors shards; the index
points at the base student's files for the frozen ones), optim.safetensors + optim.pt (the optimizer's tensors, and
the rest of its state dict with those tensors as names), state.json. Written as .ckpt-<step> and renamed when
complete; ``latest`` takes renamed ones only.

``Saver`` (async): ``snapshot`` copies every tensor into one reused host buffer (page-locked on CUDA) on a side
stream and returns at once; a background thread writes the files while training continues. ``before_update`` makes
the next optimizer step wait for the copy. Tensors ``direct`` picks (paged.py's host optimizer states) skip the
buffer: the thread writes optim.safetensors from their own storage first, and ``before_update`` also waits for that
file, since the next step writes those states back. At most one write in flight (``busy``); ``wait`` joins it and re-raises
its error. Without ``async_`` the same files are written inline from the device tensors.

A sharded run (shard.py ``Part``) writes per-rank files instead: student/trained-r<r>-*.safetensors (rank 0 the
replicated tensors, every rank its experts as per-expert keys), optim-r<r>.{safetensors,pt} (states by parameter
name), shard.json (world, expert ranges); ``load_optimizer`` reads either layout at any N."""
import atexit
import json
import os
import shutil
import struct
import threading
import time
from pathlib import Path

import torch

DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64",
          torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
          torch.bool: "BOOL", torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2"}
SHARD = 4 << 30
SMALL = 1 << 16  # optimizer tensors under this many elements stay inline in optim.pt


def write_safetensors(path, items, threads=8, piece=256 << 20):
    """``items``: [(name, tensor)] (any device; a device tensor is copied to the host as it is written). The
    safetensors layout, tensors in the given order, written straight from their memory by ``threads`` positional
    writers in ``piece``-byte ranges (one writer did 0.39 GiB/s to the volume: 337 s for v6's 131 GiB snapshot)."""
    from concurrent.futures import ThreadPoolExecutor

    header, at = {}, 0
    for name, t in items:
        n = t.numel() * t.element_size()
        header[name] = {"dtype": DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + n]}
        at += n
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    start = 8 + len(raw)
    jobs = []
    for name, t in items:
        if not t.numel():
            continue
        lo = header[name]["data_offsets"][0]
        for a in range(0, t.numel() * t.element_size(), piece):
            jobs.append((t, lo, a))
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.truncate(start + at)
        fd = f.fileno()

        def put(job):
            t, lo, a = job
            chunk = t.detach().contiguous().reshape(-1).view(torch.uint8)[a:a + piece]
            view = memoryview((chunk.cpu() if chunk.device.type != "cpu" else chunk).numpy())
            done = 0
            while done < len(view):
                done += os.pwrite(fd, view[done:], start + lo + a + done)

        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(put, jobs))


def _flatten(prefix, t, out):
    """A state tensor as [(name, plain tensor)] plus its rebuild spec (traceable subclasses, e.g. torchao's 8-bit
    states, by their inner tensors)."""
    if type(t) is not torch.Tensor and hasattr(t, "__tensor_flatten__"):
        attrs, ctx = t.__tensor_flatten__()
        cls = type(t)
        inner = {a: _flatten(f"{prefix}.{a}", getattr(t, a), out) for a in attrs}
        return {"cls": f"{cls.__module__}:{cls.__qualname__}", "ctx": ctx, "inner": inner,
                "size": tuple(t.size()), "stride": tuple(t.stride())}
    if t.numel() < SMALL:
        return {"inline": t.detach().cpu().clone()}
    out.append((prefix, t.detach()))
    return {"name": prefix}


def _rebuild(spec, tensors):
    if "inline" in spec:
        return spec["inline"]
    if "name" in spec:
        return tensors[spec["name"]]
    import importlib

    mod, qual = spec["cls"].split(":")
    cls = importlib.import_module(mod)
    for part in qual.split("."):
        cls = getattr(cls, part)
    inner = {a: _rebuild(s, tensors) for a, s in spec["inner"].items()}
    return cls.__tensor_unflatten__(inner, spec["ctx"], spec["size"], spec["stride"])


def optimizer_items(opt):
    """(tensors to write [(name, tensor)], optim.pt skeleton) of ``opt.state_dict()``."""
    sd, items = opt.state_dict(), []
    state = {pid: {k: (_flatten(f"{pid}.{k}", v, items) if torch.is_tensor(v) else {"value": v}) for k, v in s.items()}
             for pid, s in sd["state"].items()}
    groups = [{k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in g.items()} for g in sd["param_groups"]]
    return items, {"state": state, "param_groups": groups}


def load_optimizer(opt, folder, names=None, ranges=None):
    """``opt``'s state from a checkpoint folder. A sharded run's checkpoint (shard.json) is read by parameter name
    (``names``: {id(param): name}; ``ranges``: this process's expert rows per sharded name, default the full range)."""
    from safetensors.torch import load_file

    folder = Path(folder)
    if (folder / "shard.json").is_file():
        import shard

        if names is None:
            raise ValueError(f"{folder} is a sharded checkpoint: load_optimizer needs the parameter names")
        man = json.loads((folder / "shard.json").read_text())
        return shard.load_optimizer(opt, folder, names, ranges if ranges is not None
                                    else {n: (0, man["experts"]) for n in man["sharded"]})
    tensors = load_file(str(folder / "optim.safetensors")) if (folder / "optim.safetensors").is_file() else {}
    skel = torch.load(folder / "optim.pt", map_location="cpu", weights_only=False)
    state = {pid: {k: (v["value"] if "value" in v else _rebuild(v, tensors)) for k, v in s.items()}
             for pid, s in skel["state"].items()}
    opt.load_state_dict({"state": state, "param_groups": skel["param_groups"]})


def trained_items(model, head):
    items = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    items += [(f"mtp.{n}", p) for n, p in head.named_parameters()]
    return sorted(items)


def latest(out):
    done = sorted(Path(out).glob("ckpt-*/state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    return done[-1].parent if done else None


def write(out, student, step, cursor, weights, optim, skeleton, keep, after_optim=None):
    """The checkpoint folder from host (or device) tensors: weights [(name, tensor)] sorted, optim [(name, tensor)].
    optim.safetensors is written first; then ``after_optim()``."""
    out, base = Path(out), Path(student)
    tmp, final = out / f".ckpt-{step}", out / f"ckpt-{step}"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "student").mkdir(parents=True)
    write_safetensors(tmp / "optim.safetensors", optim)
    if after_optim is not None:
        after_optim()
    base_index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = {k: str(base / f) for k, f in base_index["weight_map"].items()}
    shard, size, part = [], 0, 0
    for i, (name, t) in enumerate(weights):
        shard.append((name, t))
        size += t.numel() * t.element_size()
        if size >= SHARD or i == len(weights) - 1:
            part += 1
            file = f"trained-{part:05d}.safetensors"
            write_safetensors(tmp / "student" / file, shard)
            weight_map.update({k: file for k, _ in shard})
            shard, size = [], 0
    for f in ("config.json", "generation_config.json"):
        if (base / f).is_file():
            (tmp / "student" / f).write_bytes((base / f).read_bytes())
    (tmp / "student" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": base_index.get("metadata") or {}, "weight_map": weight_map}))
    torch.save(skeleton, tmp / "optim.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step, "cursor": list(cursor), "student": str(base)}))
    tmp.rename(final)
    prune(out, keep)
    return final


def prune(out, keep):
    """Drop all but the last ``keep`` checkpoints in ``out``; the steps listed in ``out/pins.json`` are kept and not counted."""
    out = Path(out)
    pins = set(json.loads((out / "pins.json").read_text())) if (out / "pins.json").is_file() else set()
    rot = [p for p in sorted(out.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1])) if int(p.name.split("-")[1]) not in pins]
    for old in rot[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)


class Saver:
    def __init__(self, cfg, commit=lambda: None, async_=True, log=lambda **row: None):
        self.cfg, self.commit, self.async_, self.log = cfg, commit, async_, log
        self.buf, self.thread, self.error, self.event, self.registered = None, None, None, None, False
        self.prep = None
        self.part = None  # shard.Part on a sharded run: this rank's items and writer
        self.direct = lambda t: False  # tensors written from their own storage (train: the pager's host states)
        self.direct_done = None
        self.ranks = 1  # processes on this host saving at once (their host copies share its RAM)
        atexit.register(self.finish_at_exit)

    def finish_at_exit(self):
        """A crash in the main thread must not take the write in flight with it (v1c: .ckpt-744, killed at ~5 min)."""
        if self.thread is not None:
            self.thread.join(timeout=900)

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def prepare(self, model, head, opt):
        """Allocate and page-lock the snapshot buffer on a thread, once, ahead of the first save (v6: 66.6 s in the
        main loop at the first checkpoint). Called after a step, when the optimizer states exist."""
        if not self.async_ or self.prep is not None:
            return
        weights, optim, _ = self._collect(model, head, opt)
        total = sum(-(-t.numel() * t.element_size() // 64) * 64 for _, t in weights + optim if not self.direct(t))
        self.prep = threading.Thread(target=self._host, args=(total, (weights + optim)[0][1].device), daemon=True,
                                     name="sft-ckpt-buffer")
        self.prep.start()

    def _collect(self, model, head, opt):
        """(weights, optimizer tensors, optim skeleton) to write."""
        if self.part is not None:
            return self.part.items(model, head, opt)
        return (trained_items(model, head), *optimizer_items(opt))

    def commit_retrying(self, tries=20, delay=30.0):
        """The written checkpoint's commit: a refusal ("too many layers in volume") passes, so it must not end the run."""
        for attempt in range(tries):
            try:
                return self.commit()
            except Exception as error:  # noqa: BLE001 - the volume bridge's RuntimeError carries the server's text
                if attempt == tries - 1 or "too many layers" not in str(error):
                    raise
                self.log(event="ckpt_commit_retry", attempt=attempt, error=str(error)[:200])
                time.sleep(delay)

    def wait(self):
        if self.prep is not None:
            self.prep.join()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError("checkpoint write failed") from error

    def before_update(self):
        """The next optimizer step changes what the snapshot copies: the device waits for the copy first."""
        if self.event is not None:
            torch.cuda.current_stream().wait_event(self.event)
            self.event = None
        if self.direct_done is not None:
            began = time.time()
            self.direct_done.wait()
            self.direct_done = None
            waited = time.time() - began
            if waited > 0.5:
                self.log(event="ckpt_direct_wait", seconds=round(waited, 1))

    def _copy_room(self, nbytes):
        """True when this host holds a copy of the direct states of every rank (with 32 GiB to spare): the writer then
        copies them first and the next step waits only for that copy, not for the volume write (v3 2 Nov: 161 s)."""
        try:
            from arc_infra.runops import hostmem

            avail = hostmem.sample().get("avail_gib")
        except ImportError:
            avail = None
        return avail is not None and avail - self.ranks * nbytes / 2**30 >= 32

    def _host(self, nbytes, device):
        """The reused host buffer (page-locked with cudaHostRegister on CUDA: exact size, no allocator rounding);
        None when the host lacks room for it (the caller writes inline)."""
        if self.buf is not None and self.buf.numel() >= nbytes:
            return self.buf
        try:
            from arc_infra.runops import hostmem

            avail = hostmem.sample().get("avail_gib")
        except ImportError:
            avail = None
        if avail is not None and avail - nbytes / 2**30 < 32:
            self.log(event="ckpt_inline", why=f"host avail {avail} GiB, snapshot {nbytes / 2**30:.1f} GiB")
            return None
        self.release()
        began = time.time()
        self.buf = torch.empty(nbytes, dtype=torch.uint8)
        if device.type == "cuda":
            self.registered = torch.cuda.cudart().cudaHostRegister(self.buf.data_ptr(), nbytes, 0) == 0
        self.log(event="ckpt_buffer", gib=round(nbytes / 2**30, 2), pinned=self.registered,
                 seconds=round(time.time() - began, 1))
        return self.buf

    def release(self):
        if self.buf is not None and self.registered:
            torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        self.buf, self.registered = None, False

    def save(self, model, head, opt, step, cursor):
        """Snapshot and write (async) or write inline; returns the final path."""
        self.wait()
        if self.part is not None:
            self.part.begin(self.cfg.out, step)
        weights, optim, skeleton = self._collect(model, head, opt)
        writer = self.part.write if self.part is not None else write
        final = Path(self.cfg.out) / f"ckpt-{step}"
        args = (self.cfg.out, self.cfg.student, step, cursor)
        device = (weights + optim)[0][1].device
        total = sum(-(-t.numel() * t.element_size() // 64) * 64 for _, t in weights + optim if not self.direct(t))
        direct = sum(t.numel() * t.element_size() for _, t in optim if self.direct(t))
        direct_n = sum(1 for _, t in optim if self.direct(t))
        buf = self._host(total, device) if self.async_ else None
        if buf is None:
            writer(*args, weights, optim, skeleton, self.cfg.keep_ckpts)
            self.commit_retrying()
            return final
        cuda = device.type == "cuda"
        side = torch.cuda.Stream(device) if cuda else None
        if cuda:
            side.wait_stream(torch.cuda.current_stream(device))
        views, at, own = [], 0, set()
        with torch.cuda.stream(side) if cuda else torch.no_grad():
            for name, t in weights + optim:
                if self.direct(t):
                    own.add(len(views))
                    views.append((name, t.detach()))
                    continue
                n = t.numel() * t.element_size()
                v = buf[at:at + n].view(t.dtype).view(t.shape)
                v.copy_(t.detach(), non_blocking=cuda)
                views.append((name, v))
                at += -(-n // 64) * 64
        if cuda:
            self.event = torch.cuda.Event()
            self.event.record(side)
        event = self.event
        done = self.direct_done = threading.Event() if direct else None
        copy = bool(direct) and self._copy_room(direct)

        def run():
            nonlocal views
            try:
                began = time.time()
                if event is not None:
                    event.synchronize()
                if copy:  # the direct states into a host copy first; the next step waits for this only
                    stage, at, out = torch.empty(direct, dtype=torch.uint8), 0, []
                    for i, (name, t) in enumerate(views):
                        if i in own:
                            n = t.numel() * t.element_size()
                            v = stage[at:at + n].view(t.dtype).view(t.shape)
                            v.copy_(t)
                            t, at = v, at + n
                        out.append((name, t))
                    views = out
                    self.log(event="ckpt_direct_copy", step=step, gib=round(direct / 2**30, 2),
                             seconds=round(time.time() - began, 1))
                    done.set()
                writer(*args, views[:len(weights)], views[len(weights):], skeleton, self.cfg.keep_ckpts,
                      after_optim=done.set if done is not None and not copy else None)
                self.commit_retrying()
                self.log(event="ckpt_written", step=step, path=str(final), gib=round(total / 2**30, 2),
                         direct_gib=round(direct / 2**30, 2), direct_n=direct_n, direct_copied=copy,
                         seconds=round(time.time() - began, 1))
            except BaseException as error:  # noqa: BLE001 - re-raised in the main thread by wait()
                self.error = error
            finally:
                if done is not None:
                    done.set()  # a failed write must not hold the step; wait() re-raises its error

        self.thread = threading.Thread(target=run, name=f"ckpt-{step}", daemon=True)
        self.thread.start()
        return final
