"""Offline teacher for the KL loss (train.py ``loss="kl"``): the teacher's top-k next-token log-probs at every reply
position of each build.py example (the same token ids the student trains on), written as <data>/teacher/<split>/<name>.pt
= {"pos": int32 [n] (the predicted token's position), "ids": int32 [n, k + forced], "lp": float32 [n, k + forced]}:
the ``forced`` ids (structural tokens, e.g. <|im_end|>, so the student learns where the teacher ends a turn) and the
top-k of the rest, all distinct (train.py's KL takes any distinct ids plus a tail bucket). One no-grad pass per example
in ``chunk``-token pieces over a cache; the log-softmax over the full vocab in fp32.

Scoring pass (``score``): held-out first, then the train selection (``<data>/priority.json``, in its order; else the
index's ``priority`` order), shard i of n, each written (tmp + rename) and committed as soon as it is done; files already there are
skipped (resumable); no new example starts past the deadline (scoring seconds after the load).

    python teacher.py --model <teacher folder> --data <build.py output> [--k 64] [--forced "<|im_end|>" ...]
        [--shard 0/2] [--deadline-s 3600] [--out <metrics dir>] [--fast]
"""
import argparse
import gc
import io
import os
import json
import time
from pathlib import Path

import torch

import feed
import modelio

FORCED = ("<|im_end|>", "</think>", "<tool_call>")


STAGE_TRIES = 120  # a staging OOM waits for the running pass to end
SIZES = {"chunk": 32768, "rows": 2048}  # halved on an OOM (``score``); leg-v1: fastest, 1 retry in 13 examples
PROFILE = {"at": None, "out": None}
SLOW = {"after": 0, "floor": 0.0}  # --slow-host-*: stop after ``after`` examples if their median host launch rate
# (tokens / topk call seconds, the first 2 skipped) is under ``floor`` tok/s (kl-v9/v10: 72-76% util on slow containers)
LOCAL = {"on": False, "ahead": 4, "dir": "/dev/shm/teacher-io"}  # --local-io: volume reads and writes in child
# processes (prefetch to ``dir``, mmap load; write locally, a mover puts files on the volume). Threads doing them held
# the launching thread's GIL: kl-v12m 11.3K launch tok/s vs graph-v1 17.1K on the same files; CPU repro 0.47-0.59x
PREFETCH = r"""
import json, os, shutil, sys, time
root, manifest, pf, ahead = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
parent = os.getppid()
for i, f in enumerate(json.load(open(manifest))):
    while sum(x.endswith(".pt") for x in os.listdir(pf)) >= ahead:
        if os.getppid() != parent:
            sys.exit(0)
        time.sleep(0.01)
    dst = os.path.join(pf, "%06d.pt" % i)
    shutil.copyfile(os.path.join(root, f), dst + ".part")
    os.replace(dst + ".part", dst)
"""
MOVER = r"""
import os, shutil, sys
for line in sys.stdin:
    src, dst = line.rstrip("\n").split("\t")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    os.remove(src)
"""
SWITCH = {"diag": True, "legacy": False, "sync": False, "inline": False}  # --ab-blocks: per-example drift signals; leg-v5's row
# masking and indexer; staging on the scoring stream (no side stream)  # ``score``: example number ``at`` (1-based) run under prof.profile -> ``out``


def _rows(h, weight, rec, force, width):
    """One row block: (top values, top ids int32, recorded-token log-prob, per-row top-1 hit), forced ids first."""
    lp = torch.log_softmax((h @ weight.t()).float(), -1)
    rec_lp = lp.gather(1, rec[:, None])[:, 0]
    hit = lp.argmax(-1) == rec
    if force.numel():  # the forced ids first, in their order, then the top-k of the rest
        kept = lp[:, force].clone()
        if SWITCH["legacy"]:
            lp[:, force] = -float("inf")
        else:
            lp.index_fill_(1, force, -float("inf"))  # no host value copied (CUDA-graph capture; same result as setitem)
        v, i = lp.topk(width - force.numel(), dim=-1)
        v, i = torch.cat([kept, v], 1), torch.cat([force.expand(len(i), -1), i], 1)
    else:
        v, i = lp.topk(width, dim=-1)
    return v, i.to(torch.int32), rec_lp, hit


GRAPH = {"on": False, "made": {}, "check": None}  # the row block as one CUDA graph (``RowGraph``), per shape; opt-in
# (--row-graph): a capture while the loader thread stages on its side stream can invalidate both (sft-score-kl-v4 shard 0)


class RowGraph:
    """``_rows`` for a fixed ``rows`` captured once as a CUDA graph and replayed per block (the last block padded):
    ~40 launches per block become one (leg-v4: the launching thread needed ~80% of the card's time per example, so
    any other thread's GIL time idled the card). The same kernels on the same shapes as an eager full block."""

    def __init__(self, weight, rows, width, force):
        device = weight.device
        self.weight, self.rows, self.width, self.force = weight, rows, width, force
        self.h = torch.zeros(rows, weight.shape[1], dtype=weight.dtype, device=device)
        self.rec = torch.zeros(rows, dtype=torch.long, device=device)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):
                _rows(self.h, weight, self.rec, force, width)
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        # thread_local: another thread's CUDA call during the capture does not invalidate it (kl-v4 shard 0)
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.out = _rows(self.h, weight, self.rec, force, width)

    def __call__(self, h, rec):
        n = h.shape[0]
        self.h[:n].copy_(h)
        self.rec[:n].copy_(rec)
        if n < self.rows:
            self.h[n:].zero_()
            self.rec[n:].zero_()
        self.graph.replay()
        v, i, rec_lp, hit = self.out
        return v[:n].clone(), i[:n].clone(), rec_lp[:n].clone(), hit[:n].clone()


def row_graph(weight, rows, width, force):
    """The cached ``RowGraph`` for this shape, made and checked bitwise against eager ``_rows`` on a full random
    block the first time; None (eager) if capture fails or the check misses."""
    if not GRAPH["on"] or weight.device.type != "cuda":
        return None
    key = (rows, width, tuple(force.tolist()), weight.data_ptr())
    if key not in GRAPH["made"]:
        try:
            g = RowGraph(weight, rows, width, force)
            gen = torch.Generator(device=weight.device).manual_seed(0)
            h = torch.randn(rows, weight.shape[1], generator=gen, device=weight.device).to(weight.dtype)
            rec = torch.randint(0, weight.shape[0], (rows,), generator=gen, device=weight.device)
            want, got = _rows(h, weight, rec, force, width), g(h, rec)
            ok = all(torch.equal(x, y) for x, y in zip(want, got))
            GRAPH["check"] = {"ok": bool(ok), "rows": rows, "width": width}
            GRAPH["made"][key] = g if ok else None
        except Exception as error:  # noqa: BLE001 - eager stays
            GRAPH["check"] = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
            GRAPH["made"][key] = None
    return GRAPH["made"][key]


@torch.no_grad()
def topk(model, ex, k=64, chunk=8192, rows=2048, forced=(), defer=False):
    """The teacher's top-k (forced ids first) log-probs at every reply position of ``ex``, its log-prob of the
    recorded token and its top-1 rate. Nothing is read back until the example ends: the positions come from the host
    labels, every row chunk's results stay on the card and are copied once."""
    from transformers import DynamicCache

    import attention

    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if "made" in ex:  # staged ahead (``score``'s loader thread)
        pos, made = ex["pos"], feed.ready(ex["made"])
    else:
        pos, made = positions(ex), feed.stage(model, ex, device)
    ex, piece = made["ex"], made["piece"]
    ids = ex["input_ids"]
    total = ids.shape[1]
    out_ids, out_lp, rec_lp = [], [], []
    hits = torch.zeros((), dtype=torch.long, device=device)
    cache, core, weight = DynamicCache(config=model.config.text_config), model.model, model.lm_head.weight
    force = torch.tensor(list(forced), dtype=torch.long, device=device)
    width = min(k + len(forced), weight.shape[0])
    graph = row_graph(weight, rows, width, force) if cuda else None
    model.eval()
    for a in range(0, total, chunk):
        b = min(a + chunk, total)
        attention.reset()  # the causal-mask memo is per model call (26 Oct sw-v1: stale across calls, scores differed)
        with torch.profiler.record_function("t:forward"):
            hidden = core(input_ids=ids[:, a:b], attention_mask=torch.ones_like(ids[:, :b]), past_key_values=cache,
                          use_cache=True, **piece(a, b)).last_hidden_state[0]
        want_h = pos[(pos - 1 >= a) & (pos - 1 < b)] - 1 - a
        phase = torch.profiler.record_function("t:rows")
        phase.__enter__()
        want_d = feed.move(want_h, device, cuda)  # once per chunk, sliced per block on the card
        for s in range(0, len(want_h), rows):
            want = want_d[s:s + rows]
            rec = ids[0, want + a + 1]  # the recorded token each row predicts
            v, i, r, hit = (graph or (lambda h, rc: _rows(h, weight, rc, force, width)))(hidden[want], rec)
            rec_lp.append(r)
            hits += hit.sum()
            out_ids.append(i)
            out_lp.append(v)
        phase.__exit__(None, None, None)
    n = int(pos.numel())
    if defer and cuda:  # copied into pinned memory behind the queued work; ``finish`` waits (``score``'s writer)
        out = {"ids": torch.cat(out_ids) if out_ids else torch.zeros(0, width, dtype=torch.int32, device=device),
               "lp": torch.cat(out_lp) if out_lp else torch.zeros(0, width, device=device),
               "rec_lp": torch.cat(rec_lp) if rec_lp else torch.zeros(0, device=device), "hits": hits}
        got = {"pos": pos.to(torch.int32), "n": n}
        for key, t in out.items():
            got[key] = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            got[key].copy_(t, non_blocking=True)
        got["event"] = torch.cuda.current_stream(device).record_event()
        return got
    got = {"pos": pos.to(torch.int32),
           "ids": torch.cat(out_ids).cpu() if out_ids else torch.zeros(0, width, dtype=torch.int32),
           "lp": torch.cat(out_lp).cpu() if out_lp else torch.zeros(0, width),
           "rec_lp": torch.cat(rec_lp).cpu() if rec_lp else torch.zeros(0)}
    got["top1_rate"] = int(hits) / max(n, 1)
    return got


GC = {"n2": 0, "pause_s": 0.0, "t": None}


def _gc_watch(phase, info):
    """gc callback: count gen-2 collections and their pauses (leg-v5's util fell 95 -> 88.5% over 9 min)."""
    if info.get("generation") != 2:
        return
    if phase == "start":
        GC["t"] = time.perf_counter()
    elif GC["t"] is not None:
        GC["n2"] += 1
        GC["pause_s"] += time.perf_counter() - GC["t"]
        GC["t"] = None


def cpu_stat():
    """The container's cgroup CPU counters (v2 cpu.stat, else v1 cpu.stat / cpuacct): usage and throttling."""
    for path in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat"):
        try:
            with open(path) as f:
                return {k: int(v) for k, v in (line.split() for line in f if len(line.split()) == 2)}
        except (OSError, ValueError):
            continue
    return {}


CPU = {"last": None}


def host_state(device, pending):
    """Per-example drift signals: writer backlog, gen-2 GC, RSS, pinned pool, allocator reserved vs allocated, and
    the cgroup's CPU throttling since the last example (nr_throttled, throttled seconds, usage seconds)."""
    import gc

    got = {"writer_backlog": sum(1 for f in pending if not f.done()), "gc2": GC["n2"],
           "gc2_pause_s": round(GC["pause_s"], 3), "gc_counts": list(gc.get_count())}
    now, last = cpu_stat(), CPU["last"]
    if now and last:
        got["cpu_throttled_n"] = now.get("nr_throttled", 0) - last.get("nr_throttled", 0)
        got["cpu_throttled_s"] = round((now.get("throttled_usec", now.get("throttled_time", 0) // 1000)
                                        - last.get("throttled_usec", last.get("throttled_time", 0) // 1000)) / 1e6, 3)
        got["cpu_usage_s"] = round((now.get("usage_usec", 0) - last.get("usage_usec", 0)) / 1e6, 2)
    CPU["last"] = now or None
    try:
        with open("/proc/self/status") as f:
            got["rss_gib"] = round(next(int(x.split()[1]) for x in f if x.startswith("VmRSS")) / 2 ** 20, 2)
    except (OSError, StopIteration):
        pass
    if device.type == "cuda":
        st = torch.cuda.memory_stats(device)
        res, alloc = st.get("reserved_bytes.all.current", 0), st.get("allocated_bytes.all.current", 0)
        got.update(reserved_gib=round(res / 2 ** 30, 2), allocated_gib=round(alloc / 2 ** 30, 2),
                   inactive_split_gib=round(st.get("inactive_split_bytes.all.current", 0) / 2 ** 30, 2),
                   segments=st.get("segment.all.current", 0))
        try:
            hs = torch.cuda.host_memory_stats()
            got["pinned_gib"] = round(hs.get("allocated_bytes.current", hs.get("reserved_bytes.current", 0)) / 2 ** 30, 2)
        except Exception:  # noqa: BLE001 - not in every torch
            pass
    return got


def finish(got):
    """A deferred ``topk`` result once its copies are done: the same dict as ``topk(defer=False)``."""
    if "event" in got:
        got.pop("event").synchronize()
        got["top1_rate"] = int(got.pop("hits")) / max(got.pop("n"), 1)
    return got


def positions(ex):
    """Predicted token positions of the reply rows (row j - 1 predicts j), from the host labels."""
    return (feed.prepare(ex)["labels"][0, 1:] != -100).nonzero()[:, 0] + 1


SANITY = {"first": 5, "mass_min": 0.7, "rec_lp_min": -2.0, "rows_min": 32}  # pre-registered, back-tested on tA-v1 teacher-R


def sanity(got, n_before):
    """(stats, error or None) of one scored example: any non-finite value stops the card; so do top-k mass mean
    < 0.7 or recorded-token mean logp < -2 on one of its first 5 examples of at least 32 rows (a 3-token reply's mean
    is noise: sft-score-arc2-held-v1)."""
    lp, rec = got["lp"], got["rec_lp"]
    stats = {"topk_mass": round(float(lp.exp().sum(-1).mean()), 4) if lp.numel() else None,
             "rec_lp": round(float(rec.mean()), 4) if rec.numel() else None, "top1": round(got["top1_rate"], 4)}
    if not (torch.isfinite(lp).all() and torch.isfinite(rec).all()):
        return stats, "non-finite teacher log-probs"
    if n_before < SANITY["first"] and rec.numel() >= SANITY["rows_min"] and (stats["topk_mass"] < SANITY["mass_min"]
                                                     or stats["rec_lp"] < SANITY["rec_lp_min"]):
        return stats, f"sanity bar missed on example {n_before + 1}: {stats}"
    return stats, None


def order(root, priority=None):
    """[(split, entry)]: held-out first, then train: exactly ``priority.json``'s (or the ``priority`` list file's)
    example paths in its order (the selection), else every train entry by the index's ``priority`` field (lower
    first; stable)."""
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    train = list(index["train"]["files"])
    pri = Path(priority) if priority else root / "priority.json"
    if pri.is_file():
        by = {e["file"]: e for e in train}
        train = [by[f] for f in json.loads(pri.read_text())]
    else:
        train.sort(key=lambda e: e.get("priority", 0))
    return [("heldout", e) for e in index["heldout"]["files"]] + [("train", e) for e in train]


def merge(dest, root):
    """Teacher files scored into a scratch ``dest`` moved into the data root's pool (<root>/teacher/<split>/<file>);
    a file already in the pool stays and the scratch copy is left. Returns {"moved", "kept"}."""
    src, dst = Path(dest) / "teacher", Path(root) / "teacher"
    moved = kept = 0
    for f in sorted(src.rglob("*.pt")):
        to = dst / f.relative_to(src)
        if to.exists():
            kept += 1
            continue
        to.parent.mkdir(parents=True, exist_ok=True)
        os.replace(f, to)
        moved += 1
    return {"moved": moved, "kept": kept}


def score(model, root, entries, k, forced, deadline, commit=lambda: None, log=print, out=None, every=60.0,
          skip_root=False):
    """Score ``entries`` (from ``order``) into <out or root>/teacher/<file>; returns (examples, context tokens). The
    next example is read and staged (side stream) on a loader thread while this one runs; files are written on a
    writer thread and the volume committed at most every ``every`` seconds (and at the end)."""
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    root, dest = Path(root), Path(out or root)
    device = next(model.parameters()).device
    stream = torch.cuda.Stream(device) if device.type == "cuda" and not SWITCH["sync"] else None
    main = torch.cuda.current_stream(device) if stream is not None else None
    todo = [(split, e) for split, e in entries  # done here, or (``skip_root``: a scratch dest for the pool) there
            if not (dest / "teacher" / e["file"]).is_file()
            and not (skip_root and (root / "teacher" / e["file"]).is_file())]
    ahead, stop = queue.Queue(maxsize=1), threading.Event()
    pre = mover = None
    if LOCAL["on"]:
        import shutil
        import subprocess
        import sys

        work = Path(LOCAL["dir"]) / f"{os.getpid()}-{time.time_ns()}"
        pf, staged_out = work / "in", work / "out"
        pf.mkdir(parents=True)
        staged_out.mkdir()
        (work / "todo.json").write_text(json.dumps([e["file"] for _, e in todo]))
        pre = subprocess.Popen([sys.executable, "-c", PREFETCH, str(root), str(work / "todo.json"), str(pf),
                                str(LOCAL["ahead"])])
        mover = subprocess.Popen([sys.executable, "-c", MOVER], stdin=subprocess.PIPE, text=True)

    def read(i, e):
        """Example ``i`` of ``todo``: from the prefetch dir (mmap; the file is unlinked, the mapping stays) or the
        volume (the read without the GIL; torch.load of the bytes after)."""
        if pre is None:
            with open(root / e["file"], "rb") as f:
                return torch.load(io.BytesIO(f.read()), weights_only=False)
        path = pf / f"{i:06d}.pt"
        while not path.is_file():
            if stop.is_set():
                return None
            if pre.poll() is not None and not path.is_file():
                raise RuntimeError(f"teacher prefetch exited {pre.returncode} before {e['file']}")
            time.sleep(0.005)
        ex = torch.load(path, weights_only=False, mmap=True)
        path.unlink()
        return ex

    def loader():
        try:
            for i, (split, e) in enumerate(todo):
                if stop.is_set():
                    break
                t0 = time.perf_counter()
                raw = read(i, e)
                if raw is None:
                    break
                ex = feed.prepare(raw)
                t1 = time.perf_counter()
                after = None
                if main is not None:  # the staging waits for all work queued so far on the scoring stream: the
                    # last example's staged tensors (side-stream blocks, free for reuse there once dropped) may still
                    # be read by its queued kernels (26 Oct memo-v1/sw-v1: scores differed run to run)
                    after = torch.cuda.Event()
                    after.record(main)
                staged = dict(ex)  # --stage-inline: topk stages it, nothing on the card beside a pass
                for attempt in range(0 if SWITCH["inline"] else STAGE_TRIES):
                    try:
                        staged = {"pos": positions(ex), "made": feed.stage(model, ex, device, stream=stream, after=after)}
                        break
                    except torch.cuda.OutOfMemoryError:
                        if attempt == STAGE_TRIES - 1:
                            raise
                    gc.collect()  # out of the except block, as in the scoring loop
                    torch.cuda.empty_cache()
                    time.sleep(1.0)
                    say({"event": "stage_oom", "file": e["file"], "attempt": attempt})
                staged["load_s"], staged["stage_s"] = round(t1 - t0, 3), round(time.perf_counter() - t1, 3)
                ahead.put((split, e, ex, staged))
        except BaseException as error:  # noqa: BLE001 - raised in the scoring loop
            ahead.put(error)
        ahead.put(None)

    def write(got, dst):
        """Serialized in memory, then written with the GIL released (torch.save to the volume file held it for the
        whole write: the launching thread stalled and the card idled at each example, sft-score-kl-v3)."""
        buf = io.BytesIO()
        torch.save(got, buf)
        if mover is not None:  # local file, then the mover process copies it to the volume (tmp + rename there)
            if mover.poll() is not None:
                raise RuntimeError(f"teacher mover exited {mover.returncode}")
            local = staged_out / f"{time.time_ns()}.pt"
            with open(local, "wb") as f:
                f.write(buf.getbuffer())
            mover.stdin.write(f"{local}\t{dst}\n")
            mover.stdin.flush()
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(buf.getbuffer())
        tmp.rename(dst)

    lock, failed, last = threading.Lock(), [], [time.time()]

    def say(row):
        with lock:
            log(row)

    def save(got, split, e, index, began, info):
        """Writer thread, in order: wait for the copies, sanity, write, log, commit on cadence. After a sanity
        stop nothing more is written."""
        if failed:
            return
        got = finish(got)
        stats, error = sanity(got, index)
        if error:
            say({"event": "sanity_stop", "file": e["file"], **stats})
            failed.append(error)
            return
        got["meta"] = {"k": k, "forced": list(forced), "split": split, **stats}
        write(got, dest / "teacher" / e["file"])
        say({"event": "scored", "file": e["file"], "rows": int(got["pos"].numel()), "tokens": int(e["tokens"]),
             "seconds": round(time.time() - began, 1), **info, **stats})
        if time.time() - last[0] >= every:
            commit()
            last[0] = time.time()

    if GRAPH["on"] and device.type == "cuda":  # captured before the loader thread starts staging
        weight = model.lm_head.weight
        row_graph(weight, SIZES["rows"], min(k + len(forced), weight.shape[0]),
                  torch.tensor(list(forced), dtype=torch.long, device=device))
    threading.Thread(target=loader, daemon=True, name="score-load").start()
    writer = ThreadPoolExecutor(1, thread_name_prefix="score-write")
    pending, done, tokens = [], 0, 0
    retries = lambda: torch.cuda.memory_stats().get("num_alloc_retries", 0) if device.type == "cuda" else 0  # noqa: E731
    rates = []  # host launch tok/s per example (SLOW)
    score.slow = None
    per_example = []  # allocator retries of each scored example (``main``'s sweep skips a size's first)
    try:
        while True:
            waited = time.perf_counter()
            item = ahead.get()
            waited = time.perf_counter() - waited
            if item is None:
                break
            if isinstance(item, BaseException):
                raise RuntimeError("teacher loader failed") from item
            if failed:
                raise RuntimeError(failed[0])
            split, e, ex, staged = item
            if time.time() >= deadline:
                say({"event": "deadline", "examples": done, "tokens": tokens})
                break
            began, r0 = time.time(), retries()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            while True:
                try:
                    run = lambda: topk(model, staged, k, chunk=SIZES["chunk"], rows=SIZES["rows"], forced=forced,  # noqa: E731
                                       defer=True)
                    if PROFILE["at"] == done + 1 and PROFILE["out"]:
                        import prof

                        box = {}
                        info, _ = prof.profile(lambda: box.update(got=run()), PROFILE["out"], device,
                                               {"tokens": int(e["tokens"]), **SIZES})
                        got = box["got"]
                        say({"event": "profiled", "file": e["file"], **info})
                    else:
                        got = run()
                    break
                except torch.cuda.OutOfMemoryError:  # rows first, then the prefill chunk; the example starts over
                    key = "rows" if SIZES["rows"] > 1024 else "chunk"
                    if SIZES[key] <= 1024:
                        raise
                # out of the except block: its traceback holds the failed pass's tensors
                SIZES[key] //= 2
                run = staged = None
                gc.collect()
                torch.cuda.empty_cache()
                say({"event": "oom_halved", "file": e["file"], **SIZES})
                staged = dict(ex) if SWITCH["inline"] else {"pos": positions(ex), "made": feed.stage(model, ex, device)}
            done, tokens = done + 1, tokens + int(e["tokens"])
            per_example.append(retries() - r0)
            info = {"examples": done, "total_tokens": tokens, "alloc_retries": per_example[-1], **SIZES,
                    **(host_state(device, pending) if SWITCH["diag"] else {}),
                    "wait_s": round(waited, 3), "launch_s": round(time.time() - began, 2),  # host: queue wait, topk call
                    "launch_tok_s": round(int(e["tokens"]) / max(time.time() - began, 1e-3)),
                    "load_s": staged.get("load_s"), "stage_s": staged.get("stage_s"),
                    **({"row_graph": GRAPH["check"]} if done == 1 else {}),
                    "reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if device.type == "cuda" else None,
                    "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == "cuda" else None}
            pending = [f for f in pending if not f.done()] + [writer.submit(save, got, split, e, done - 1, began, info)]
            rates.append(info["launch_tok_s"])
            if SLOW["floor"] and done >= SLOW["after"] and len(rates) > 2:
                median = sorted(rates[2:])[(len(rates) - 2) // 2]
                if median < SLOW["floor"]:
                    score.slow = {"launch_tok_s_median": median, "floor": SLOW["floor"], "examples": done}
                    say({"event": "slow_host", **score.slow})
                    break
    finally:
        stop.set()
        try:
            for f in pending:
                f.result()
            writer.shutdown(wait=True)
        finally:
            if pre is not None:
                pre.kill()
                pre.wait()
                mover.stdin.close()
                code = mover.wait()
                shutil.rmtree(work, ignore_errors=True)
                if code:
                    failed.append(f"teacher mover exited {code}")
        commit()
    if failed:
        raise RuntimeError(failed[0])
    score.retries = per_example
    return done if not failed else 0, tokens


def fast_paths():
    """The trainer's GPU fast paths for a scoring model: GDN, hyper-connections, QSA indexer + Triton attention."""
    import attention
    import gdn
    import hyper

    attention.install(mode="triton")
    gdn.install()
    gdn.GDN.update(fast=True)
    hyper.install()
    hyper.HYPER["on"] = True


def main():
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    import vol

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--forced", nargs="*", default=list(FORCED))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--deadline-s", type=float, default=float("inf"), help="scoring seconds after the load; no new example after")
    ap.add_argument("--out", default=None, help="metrics.jsonl / events.jsonl folder (watch.py)")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--dest", default=None, help="write teacher files under <dest>/teacher (a leg), not the data root")
    ap.add_argument("--sizes", default=None, help="a leg: CHUNKxROWS,... each scored for --per-size-s, then the last on")
    ap.add_argument("--per-size-s", type=float, default=75.0)
    ap.add_argument("--profile-at", type=int, default=None, help="profile this example (1-based) -> <out>/profile.json")
    ap.add_argument("--row-graph", action="store_true", help="the row block as one CUDA graph (opt-in)")
    ap.add_argument("--ab-blocks", default="", help="scorer switch test: modes (now, lean, legacy) scored one after "
                    "another on the same first files, each for --ab-block-s into its own folder under --dest")
    ap.add_argument("--ab-block-s", type=float, default=120.0)
    ap.add_argument("--merge", action="store_true", help="at the end move --dest's teacher files into the data root")
    ap.add_argument("--slow-host-after", type=int, default=10)
    ap.add_argument("--slow-host-floor", type=float, default=0.0, help="stop if the median host launch tok/s of the "
                    "first --slow-host-after examples is under this (0: off)")
    ap.add_argument("--files", default="", help="score only these data files (comma list), in this order")
    ap.add_argument("--priority", default="", help="train selection list (JSON paths) instead of <data>/priority.json")
    ap.add_argument("--local-io", action="store_true", help="volume reads/writes in child processes (LOCAL)")
    ap.add_argument("--local-io-ahead", type=int, default=4)
    ap.add_argument("--stage-inline", action="store_true", help="stage each example after the last pass, never beside "
                    "it (image-heavy lists: the vision tower beside a pass ran the card out, sft-score-last-v1)")
    ap.add_argument("--sanity-rec-lp-min", type=float, default=None, help="the sanity bar's recorded-token log-prob "
                    "floor (default -2.0, set on reply rows; every-token labels score observation text too, 27 Oct)")
    a = ap.parse_args()
    if a.sanity_rec_lp_min is not None:
        SANITY["rec_lp_min"] = a.sanity_rec_lp_min
    t0 = time.time()
    commit, _, _ = vol.client()
    out = Path(a.out) if a.out else None
    metrics = None
    if out:
        import train

        out.mkdir(parents=True, exist_ok=True)
        metrics = train.Metrics(out, commit, period=1.0, every=1.0)  # one smi row a second
        metrics.start()

    def log(row):
        row = {"t": round(time.time(), 1), **row}
        print(json.dumps(row), flush=True)
        if out:
            with open(out / "events.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if row.get("event") == "scored":
                metrics.row(**{"score.examples": row["examples"], "score.tokens": row["total_tokens"],
                               "score.tok_s": round(row["tokens"] / max(row["seconds"], 1e-6), 1),
                               "score.topk_mass": row["topk_mass"], "score.rec_lp": row["rec_lp"],
                               "score.top1": row["top1"]})

    tok = AutoTokenizer.from_pretrained(a.model)
    forced = [tok.convert_tokens_to_ids(t) for t in a.forced]
    if any(not isinstance(i, int) or i == tok.unk_token_id for i in forced):
        raise ValueError(f"forced tokens are not single ids: {dict(zip(a.forced, forced))}")
    if a.fast:
        fast_paths()
    modelio.table_in_host_ram()
    with modelio.parallel_reads(a.model, device=a.device_map if a.device_map.startswith("cuda") else None):
        model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": a.device_map},
                                                         experts_implementation="grouped_mm")
    if a.profile_at and out:
        PROFILE.update(at=a.profile_at, out=str(out / "profile.json"))
    GRAPH["on"] = bool(a.row_graph)
    LOCAL.update(on=bool(a.local_io), ahead=a.local_io_ahead)
    SWITCH["inline"] = bool(a.stage_inline)
    import gc

    gc.collect()
    gc.freeze()  # the loaded model's objects out of every later collection
    gc.callbacks.append(_gc_watch)
    i, n = (int(x) for x in a.shard.split("/"))
    entries = order(a.data, a.priority or None)[i::n]
    if a.files:  # a fixed file list (checks), in the given order
        pick = {e["file"]: (split, e) for split, e in entries}
        entries = [pick[f] for f in a.files.split(",") if f in pick]
    log({"event": "start", "load_s": round(time.time() - t0, 1), "shard": a.shard, "entries": len(entries),
         "forced": dict(zip(a.forced, forced)), "k": a.k, **SIZES})
    end = time.time() + a.deadline_s
    done = tokens = 0
    tried = []
    for spec in (a.sizes.split(",") if a.sizes else []):  # the leg's size sweep; then the fastest without retries
        SIZES["chunk"], SIZES["rows"] = (int(x) for x in spec.split("x"))
        torch.cuda.reset_peak_memory_stats()
        r0, t0 = torch.cuda.memory_stats().get("num_alloc_retries", 0), time.time()
        n, t = score(model, a.data, entries, a.k, forced, min(end, time.time() + a.per_size_s), commit, log, out=a.dest)
        row = {**SIZES, "examples": n, "tok_s": round(t / max(time.time() - t0, 1e-9), 1),
               "retries": sum(score.retries[1:]), "retries_first": sum(score.retries[:1]),  # the first absorbs the resize
               "reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}
        log({"event": "sizes", **row})
        tried.append(row)
        done, tokens = done + n, tokens + t
    clean = [r for r in tried if r["retries"] == 0 and r["examples"] >= 3]
    if clean:
        best = max(clean, key=lambda r: r["tok_s"])
        SIZES.update(chunk=best["chunk"], rows=best["rows"])
        log({"event": "sizes_chosen", **SIZES})
    for b, mode in enumerate(a.ab_blocks.split(",") if a.ab_blocks else []):
        import attention

        lean = mode in ("lean", "legacy")
        SWITCH.update(diag=not lean, legacy=mode == "legacy", sync=mode in ("ref", "sync"))
        attention.CAUSAL.update(memo=mode not in ("ref", "nomemo"), dense=mode == "ref")
        attention.INDEX["legacy"] = mode == "legacy"
        if lean and _gc_watch in gc.callbacks:
            gc.callbacks.remove(_gc_watch)
        if not lean and _gc_watch not in gc.callbacks:
            gc.callbacks.append(_gc_watch)
        gc.unfreeze() if mode == "legacy" else (gc.collect(), gc.freeze())
        log({"event": "block", "block": b, "mode": mode})
        n, t = score(model, a.data, entries, a.k, forced, time.time() + a.ab_block_s, commit, log,
                     out=f"{a.dest or a.data}/ab-{b}-{mode}")
        done, tokens = done + n, tokens + t
    SLOW.update(after=a.slow_host_after, floor=a.slow_host_floor)
    if not a.ab_blocks:
        n, t = score(model, a.data, entries, a.k, forced, end, commit, log, out=a.dest, skip_root=a.merge)
        done, tokens = done + n, tokens + t
    SLOW.update(after=0, floor=0.0)
    moved = merge(a.dest, a.data) if a.merge and a.dest else None
    log({"event": "end", "examples": done, "tokens": tokens, "seconds": round(time.time() - t0, 1),
         "slow_host": getattr(score, "slow", None), "merged": moved})
    if metrics:
        metrics.stop()
    commit()


if __name__ == "__main__":
    main()
