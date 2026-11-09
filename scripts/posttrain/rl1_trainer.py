"""RL step 1 trainer service (13 October, docs/rl-plan.md): Flash's own NVFP4 checkpoint with a rank-32 LoRA on every
routed expert and the dense parts, fused fp32 AdamW over the fp32 adapter (the master copy), one B300 shared with the
play server. Runs in the pinned Axolotl image's Python beside ``flash_qlora_job.py`` (loader, request encoder, cached
prefix) and ``rl_job.py`` (window planning, actor log-probabilities), which it imports.

The driver (``arena.rl_loop``) writes ``<dir>/cmd/<n>.json``; the service answers ``<dir>/done/<n>.json`` and keeps
``<dir>/alive.json`` fresh. Operations:

- ``reset``: the adapter back to its initial state (A as initialised, B zero), a new optimizer.
- ``score``: windows of recorded attempts without an update: per sampled token the actor's and the raw
  log-probability; ``label`` keeps them as a KL reference (``base``, or the previous round).
- ``accumulate`` / ``step``: one group's windows forward/backward into summed gradients; the batch's AdamW step
  (both through ``op_update``).
- ``route``: R3 served routing; ``park`` / ``unpark``: the card handed to play;
  ``checkpoint`` / ``load_checkpoint``.
- ``save``: the adapter in BF16 PEFT format (what vLLM loads).
- ``exit``.

The loss of one update: ``-(1/N) sum_t w_t log pi(t)`` over the sampled reply tokens of the update's windows, with
``pi`` the actor (temperature, top-k, top-p set rebuilt from the engine's recorded alternatives), ``w_t`` the
weight times the truncated IS factor (``--tis-cap``) and ``N`` the update's trained tokens (``norm`` "cap": each
try's sum over its level's cap). Logits are kept only at those tokens (``logits_to_keep``).
"""

import argparse
import contextlib
import gzip
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flash_qlora_job as fq  # noqa: E402
import rl1_memory  # noqa: E402
import rl_job  # noqa: E402
import rl1_encode  # noqa: E402
import rl1_dist  # noqa: E402
import rl1_firstcall  # noqa: E402
import rl1_hubk  # noqa: E402
import rl1_table  # noqa: E402

CARD = f"cuda:{fq.CARD_INDEX}"  # the trainer's card (the CPU window-feed test sets "cpu"; RL1_DIST: LOCAL_RANK's)


def vocab_top(rows, k=20, chunk=1024, at_ids=(), device=False):
    """Per position of ``rows`` [n, V] (logits), the full-vocabulary log-distribution at temperature 1 reduced to its
    top ``k`` tokens (``ids``, ``logp``), and the log-probabilities of each id table in ``at_ids`` ([n, k] each),
    computed in row chunks (a 20K x 248K float table is 20 GB). ``device``: leave the results on the logits' device."""
    import torch

    ids, logp, at = [], [], [[] for _ in at_ids]
    with torch.no_grad():
        for i in range(0, rows.shape[0], chunk):
            full = torch.log_softmax(rows[i:i + chunk].float(), dim=-1)
            top = full.topk(k, dim=-1)
            ids.append(top.indices); logp.append(top.values)
            for j, table in enumerate(at_ids):
                at[j].append(full.gather(1, table[i:i + chunk].to(full.device)))
            del full
    cat = lambda xs: (torch.cat(xs) if device else torch.cat(xs).cpu()) if xs else torch.zeros(0)  # noqa: E731
    return {"ids": cat(ids), "logp": cat(logp), "at": [cat(x) for x in at]}


def token_entropy(rows, valid, temperature=1.0, chunk=1024, lazy=False):
    """(sum, count) of the policy's per-token entropy (nats, logits / temperature) over ``valid`` rows of ``rows``
    [n, V]; row chunks, no grad (dashboard collapse signal)."""
    import torch

    with torch.no_grad():  # U2 (27 Oct): sums stay on the device, one host read per window (was 3 syncs per chunk)
        mask = valid.to(rows.device)
        total = torch.zeros((), dtype=torch.float64, device=rows.device)
        for i in range(0, rows.shape[0], chunk):
            keep = mask[i:i + chunk]
            lp = torch.log_softmax(rows[i:i + chunk].float() / float(temperature or 1.0), dim=-1)
            total += (-(lp.exp() * lp).sum(-1) * keep).sum(dtype=torch.float64)
            del lp
        pair = (total, mask.sum().to(torch.float64))
        if lazy:  # the caller reads it later (the chunked head: once per window)
            return pair
        total, count = torch.stack(pair).tolist()
    return total, int(count)




def binned_kl(p_logp, q_logp):
    """Mean over positions of KL(P || Q) on P's top-k tokens plus one bin for the rest of the vocabulary (the same
    k ids on both sides): exact for that coarse-graining, so non-negative and at most the full-vocabulary KL."""
    p, q = p_logp.double().exp(), q_logp.double().exp()
    p_rest, q_rest = (1 - p.sum(1)).clamp(min=1e-12), (1 - q.sum(1)).clamp(min=1e-12)
    terms = (p * (p_logp.double() - q_logp.double())).sum(1) + p_rest * (p_rest.log() - q_rest.log())
    return float(terms.clamp(min=0).mean()) if terms.numel() else float("nan")


WHOLE_SEGMENT_WINDOW = 1 << 20  # a window bound no segment reaches: plan_windows ends windows at drains only


def whole_segment(spec):
    """``whole_segment`` (cmd / trainer_fast flag, RL redesign 4a): the try's windows are its drain segments whole.
    Each window starts at its segment's first model output (the carried-in start context, the restored history and
    any earlier segment are its no-gradient prefix: floor 0, margin 1 = the logit position of the first trained
    token; T1a 26 Oct: margin 0 cut at that token, "a trained token lies outside the gradient window") and ends at
    the segment's last; every model output in it is trained, notes and playbook replies included (captures with gameplay=False); the walk back stops
    only at the restored (scripted) history, not at a zero-credit reply."""
    return {**spec, "whole": True, "whole_segment": True, "window": WHOLE_SEGMENT_WINDOW, "floor": 0, "margin": 1,
            "credited_only": False}


NVML = {}


def nvml_read(card):
    """(used GiB, total GiB, util %) of ``card`` through libnvidia-ml in this process (U2, 27 Oct: the 1-Hz
    nvidia-smi spawn sat beside a 5.6 s launch stall in the clip); None when NVML is not loadable."""
    import ctypes

    if "lib" not in NVML:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            NVML["lib"] = lib if lib.nvmlInit_v2() == 0 else None
        except OSError:
            NVML["lib"] = None
    lib = NVML["lib"]
    if lib is None:
        return None

    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    handle = NVML.get(("handle", card))
    if handle is None:
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(int(card), ctypes.byref(handle)) != 0:
            return None
        NVML[("handle", card)] = handle
    mem, util = Memory(), Util()
    if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) or lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)):
        return None
    return mem.used / 2**30, mem.total / 2**30, float(util.gpu)


def gpu_used_gib():
    import subprocess

    try:
        cards = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        card = (cards[fq.CARD_INDEX] if fq.CARD_INDEX < len(cards) else str(fq.CARD_INDEX)) or "0"
        if os.environ.get("RL1_NVML", "1") == "1" and card.isdigit():
            got = nvml_read(card)
            if got is not None:
                return got
    except Exception:  # noqa: BLE001 - nvidia-smi below
        pass
    try:
        cards = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")  # the trainer's card (2-GPU loop: 1; RL1_DIST: its rank's)
        card = (cards[fq.CARD_INDEX] if fq.CARD_INDEX < len(cards) else str(fq.CARD_INDEX)) or "0"
        out = subprocess.run(["nvidia-smi", "-i", card, "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        used, total, util = (float(v) for v in out.splitlines()[0].split(","))
        return used / 1024, total / 1024, util
    except Exception:  # a sample is optional
        return None, None, None


class Sampler:
    """nvidia-smi every second while a phase runs: peak card memory (every process) and mean utilisation."""

    def __init__(self):
        self.stop, self.peak, self.util, self.thread = threading.Event(), 0.0, [], None

    def __enter__(self):
        def run():
            while not self.stop.wait(1.0):
                used, _, util = gpu_used_gib()
                if used is not None:
                    self.peak = max(self.peak, used)
                    self.util.append(util)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=3)

    def summary(self):
        return {"card_peak_gib": round(self.peak, 2),
                "gpu_util_mean": round(sum(self.util) / len(self.util), 1) if self.util else None}


def set_checkpointing(model, mode):
    """``full`` (the only mode): every decoder layer recomputed in backward."""
    if mode != "full":
        raise ValueError(f"checkpointing {mode!r}: only full")
    layers = [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer") and hasattr(m, "gradient_checkpointing")]
    for layer in layers:
        layer.gradient_checkpointing = True
    return len(layers)


NGRAM = {"on": False, "calls": 0, "slots": [], "next": 0}  # MFU-1: PLE n-gram rows staged through pinned host memory


def pinned_lookup(embedding, device):
    """A module standing in for the host-RAM n-gram ``nn.Embedding``: its one parameter is the same CPU table under
    the same name (``weight``: the model's ``execution_device`` stays the host, parameter names and state dicts are
    unchanged); the row gather goes into a pinned buffer and an asynchronous copy brings it to the card, so the
    model's ``.to(card)`` is a no-op (rl1-prof-m1p: 138 pageable copies, 13 GB, ~1 s plus the host stalls). Same rows,
    same bits. Two buffer slots; a slot is reused only after its copy's event completed."""
    import torch

    class PinnedLookup(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = embedding.weight
            object.__setattr__(self, "original", embedding)  # not a child: no second parameter name

        def forward(self, ids):
            if not NGRAM["on"] or ids.device.type != "cpu" or self.weight.requires_grad:
                return self.original(ids)
            rows, dim = ids.numel(), self.weight.shape[1]
            slots = NGRAM["slots"]
            if len(slots) < 2:
                slots.append({"buf": None, "event": None})
            slot = slots[NGRAM["next"] % len(slots)]
            NGRAM["next"] += 1
            if slot["event"] is not None:
                slot["event"].synchronize()
            if slot["buf"] is None or slot["buf"].numel() < rows * dim or slot["buf"].dtype != self.weight.dtype:
                slot["buf"] = torch.empty(rows * dim, dtype=self.weight.dtype, pin_memory=True)
            host = slot["buf"][: rows * dim].view(rows, dim)
            torch.index_select(self.weight.detach(), 0, ids.reshape(-1), out=host)
            out = host.to(device, non_blocking=True)
            slot["event"] = torch.cuda.Event()
            slot["event"].record()
            NGRAM["calls"] += 1
            return out.view(*ids.shape, dim)

    return PinnedLookup()


def ngram_pinned_install(model):
    """Each PLE layer's host-RAM n-gram embedding behind a pinned_lookup module (once; NGRAM["on"] picks the path
    per call)."""
    import torch

    for module in model.modules():
        if (type(module).__name__ == "Qwen4ExpTextNGramEmbedding"
                and type(module.ngram_embedding).__name__ != "PinnedLookup"
                and module.ngram_embedding.weight.device.type == "cpu"):
            module.ngram_embedding = pinned_lookup(module.ngram_embedding,
                                                   torch.device("cuda", torch.cuda.current_device()))


GROUPED = {"grouped": 0, "loop": 0}  # expert LoRA calls by path (counted in every pass)


def grouped_expert_lora():
    """The fp32 adapter master reaches the pinned Axolotl expert-LoRA functions as fp32, and their grouped-GEMM path
    requires the LoRA factors in the activations' dtype: every call fell back to a per-expert Python loop (two host
    syncs and several small GEMMs per expert, forward and backward; rl1-ttest-v3 profile: 386K syncs, 718K mm per
    window). The factors go in as BF16 casts (the autocast compute dtype; gradients return to the fp32 master through
    the cast), so ``torch._grouped_mm`` runs one launch per projection."""
    import torch
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    for cls in (nvfp4_lora.GroupedUpProjLoRA, nvfp4_lora.GroupedDownProjLoRA):
        if getattr(cls, "_rl1_patched", False):
            continue
        original = cls.apply

        def apply(*args, _original=original, _name=cls.__name__):
            x = args[0]
            key = f"{_name}:{str(x.dtype)[6:]}/{str(args[3].dtype)[6:]}"
            GROUPED[key] = GROUPED.get(key, 0) + 1  # rl1-ttest-v4: 98% of calls still took the loop
            if torch.is_autocast_enabled(x.device.type):
                # the matmuls run in BF16 under autocast either way; the grouped path needs it explicit
                args = (x.to(torch.bfloat16), args[1], args[2], args[3].to(torch.bfloat16), args[4].to(torch.bfloat16),
                        *args[5:])
            x = args[0]
            GROUPED["grouped" if nvfp4_lora._lora_grouped_mm_supported(x, args[3], args[1].shape[0]) else "loop"] += 1
            return _original(*args)

        cls.apply = staticmethod(apply)
        cls._rl1_patched = True


INDEXER = {"vectorized": 0, "fallback": 0}  # QSA indexer calls by path
SELECT = {"fast": False}  # MFU-1 fast_topk: rl1_topk's exact radix select (union kernel only)
# rl1-route: the MoE router's top-k experts recorded in one pass ("capture") and forced
# in a later one ("replay"; probe_pinned): a pass at other weights then runs the recorded discrete choices with its own
# continuous values. Off: the model's own code paths, unchanged numerics.
ROUTE = {"router": "off", "rec": [], "i": 0, "calls": 0}
EARLY = {"hook": None}  # trainer_fast.early_ahead: Trainer.early_ahead, polled at the GPU gate and on idle polls


PROFILE_SEEN = {}


@contextlib.contextmanager
def profiled(op):
    """RL1_PROFILE_OPS="step,accumulate:3": the named op's calls (``op:k``: only its k-th, 1-based) under
    torch.profiler; the CUDA- and CPU-time tables go to <ARC_VALIDATION_OUTPUT>/profiles (D2 27 Oct: the step's
    clip 2.8 s + optimizer 2.2 s at 5-23% util, not attributable from marks). Yields {} or {"path", "seconds"}."""
    PROFILE_SEEN[op] = PROFILE_SEEN.get(op, 0) + 1

    def picked(var):
        want = [w.split(":") for w in (os.environ.get(var) or "").split(",") if w]
        return any(w[0] == op and (len(w) == 1 or int(w[1]) == PROFILE_SEEN[op]) for w in want)
    hit, census = picked("RL1_PROFILE_OPS"), picked("RL1_SYNC_CENSUS")
    out = {}
    if not hit and not census:
        yield out
        return
    import torch
    from torch.profiler import ProfilerActivity, profile

    folder = Path(os.environ.get("ARC_VALIDATION_OUTPUT") or ".") / "profiles"
    folder.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK") or "0"
    began = time.time()
    with contextlib.ExitStack() as stack:
        if census:  # U2 diag (27 Oct): every host sync of the call, counted by call site (torch's sync debug mode)
            out["sync_census"] = sites = {}
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("always")

            def seen(message, category, filename, lineno, file=None, line=None):
                frames = [f for f in traceback.extract_stack()[:-1] if "/torch/" not in f.filename
                          and "/warnings.py" not in f.filename and f.name != "seen"][-3:]
                key = " < ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(frames))
                sites[key] = sites.get(key, 0) + 1
            warnings.showwarning = seen
            torch.cuda.set_sync_debug_mode("warn")
            stack.callback(torch.cuda.set_sync_debug_mode, "default")
        if hit:
            prof = stack.enter_context(profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                               with_stack=os.environ.get("RL1_PROFILE_STACK") == "1"))
        yield out
    if census:
        top = sorted(out["sync_census"].items(), key=lambda kv: -kv[1])
        out["sync_census"] = {"total": sum(v for _, v in top), "sites": top[:60]}
    if hit:  # the tables never fail the op (U1d v1: with_stack's key_averages raised UnicodeDecodeError)
        path = folder / f"{op}-{PROFILE_SEEN[op]}-rank{rank}.txt"
        text, errors = "", []
        try:
            table = prof.key_averages()
            text = (table.table(sort_by="cuda_time_total", row_limit=50) + "\n\n"
                    + table.table(sort_by="cpu_time_total", row_limit=50))
        except Exception as error:  # noqa: BLE001
            errors.append(f"tables: {error!r}"[:300])
        if os.environ.get("RL1_PROFILE_STACK") == "1":
            text += "\n\nSYNC STACKS\n" + sync_stacks(prof, errors)
        if os.environ.get("RL1_PROFILE_TRACE") == "1":  # U3: the raw timeline (CPU ops + kernels), gzipped
            try:
                raw = path.with_suffix(".trace.json")
                prof.export_chrome_trace(str(raw))
                with open(raw, "rb") as f, gzip.open(f"{raw}.gz", "wb") as g:
                    shutil.copyfileobj(f, g)
                raw.unlink()
                out["trace"] = f"{raw}.gz"
            except Exception as error:  # noqa: BLE001
                errors.append(f"trace: {error!r}"[:300])
        path.write_text(text)
        out.update(path=str(path), **({"errors": errors} if errors else {}))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out["seconds"] = round(time.time() - began, 2)


SYNC_OPS = ("aten::item", "aten::_local_scalar_dense", "cudaStreamSynchronize", "cudaDeviceSynchronize", "aten::nonzero",
            "cudaMemcpyAsync", "aten::linalg_vector_norm", "cuMemUnmap")


def sync_stacks(prof, errors):
    """The syncing ops of a with_stack profile by their Python stacks (6 frames), from the raw events: one event's
    undecodable stack is skipped and counted, never raised."""
    rows, bad = {}, 0
    try:
        events = prof.events()
    except Exception as error:  # noqa: BLE001
        errors.append(f"events: {error!r}"[:300])
        return ""
    for e in events:
        try:
            if e.name not in SYNC_OPS:
                continue
            stack = tuple(str(f) for f in (e.stack or [])[:6])
            row = rows.setdefault((e.name, stack), [0, 0.0])
            row[0] += 1
            row[1] += float(e.self_cpu_time_total or 0.0) / 1000
        except Exception:  # noqa: BLE001
            bad += 1
    if bad:
        errors.append(f"stack events skipped: {bad}")
    top = sorted(rows.items(), key=lambda kv: -kv[1][1])[:80]
    return "\n".join(f"{name} calls={n} self_cpu_ms={ms:.1f} :: " + " < ".join(stack)
                     for (name, stack), (n, ms) in top)


def clip_parts(params, max_norm, dump_after=1.0):
    """clip_grad_norm_(foreach) step by step with a sync after each part (U1: 4.5 s of host CPU inside the final
    vector norm): seconds per part, and every thread's Python stack plus a py-spy native dump if the clip passes
    ``dump_after`` s. Same math as torch's."""
    import faulthandler
    import subprocess

    import torch

    folder = Path(os.environ.get("ARC_VALIDATION_OUTPUT") or ".") / "profiles"
    folder.mkdir(parents=True, exist_ok=True)

    def dump():
        with open(folder / "clip-stacks.txt", "w") as f:
            faulthandler.dump_traceback(file=f, all_threads=True)
        spy = Path(sys.executable).parent / "py-spy"
        if spy.exists():
            r = subprocess.run([str(spy), "dump", "--native", "--pid", str(os.getpid())], capture_output=True,
                               text=True, timeout=60)
            (folder / "clip-pyspy.txt").write_text(r.stdout + "\n" + r.stderr)
    timer = threading.Timer(dump_after, dump)
    timer.daemon = True
    parts, t = {}, time.time()

    def mark(name):
        nonlocal t
        torch.cuda.synchronize()
        parts[name] = round(time.time() - t, 4)
        t = time.time()
    grads = [p.grad for p in params if p.grad is not None]
    timer.start()
    try:
        mark("before")
        norms = torch._foreach_norm(grads, 2.0)
        mark("foreach_norm")
        stacked = torch.stack(norms)
        mark("stack")
        total = torch.linalg.vector_norm(stacked, 2.0)
        mark("vector_norm")
        coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        torch._foreach_mul_(grads, coef)
        mark("scale")
        value = float(total)
        mark("read")
    finally:
        timer.cancel()
    parts["devices"] = sorted({str(g.device) for g in grads})
    parts["dtypes"] = sorted({str(g.dtype) for g in grads})
    return value, parts


def warm_step(params, groups=1):
    """U3: the step path's kernels run once at load on small stand-ins of every trainable (device, dtype): foreach
    norm, stack, vector norm, foreach scale, fused AdamW per group, clone and the update norm. U2's first step spent
    5.7 s in its first vector_norm with the card synced (dl2 steps 1/2/3: 5.1/1.1/1.0 s): a first-call cost. The
    stand-ins have their own optimizer; the real parameters and optimizer state are not touched."""
    import torch

    began, parts = time.time(), {}
    kinds = sorted({(str(p.device), p.dtype) for p in params}, key=str)
    for device, dtype in kinds:
        t = time.time()
        stand = [torch.zeros(n, device=device, dtype=dtype, requires_grad=True) for n in (3, 5)]
        for p in stand:
            p.grad = torch.full_like(p, 0.5)
        opt = torch.optim.AdamW([{"params": stand[:1]}, {"params": stand[1:]}] if groups > 1 else stand, lr=1e-6,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=device.startswith("cuda"))
        torch.nn.utils.clip_grad_norm_(stand, 1.0, foreach=True)
        before = [p.detach().clone() for p in stand]
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            float(sum((p.detach() - b).float().pow(2).sum() for p, b in zip(stand, before)))
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        parts[f"{device}/{str(dtype).replace('torch.', '')}"] = round(time.time() - t, 3)
    return {"seconds": round(time.time() - began, 3), "parts": parts}


def stamp():
    """A timing mark: a recorded CUDA event (no sync) on the card, else the wall clock."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return time.time()


def span(a, b):
    """Seconds between two stamps (GPU timeline for events: waits on ``b``)."""
    if isinstance(a, float):
        return b - a
    b.synchronize()
    return a.elapsed_time(b) / 1000.0


def early_poll():
    """The early-ahead hook, never raising (a failure leaves the first call to encode as before)."""
    hook = EARLY.get("hook")
    if hook is not None:
        try:
            hook()
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"event": "early_ahead_error", "error": repr(error)[:500]}), flush=True)
# live progress: window tail tokens through the model and the phase; a thread writes one row per
# PROGRESS_EVERY s to $ARC_VALIDATION_OUTPUT/util/train_progress.jsonl (the efficiency watcher and dashboard read it)
PROGRESS = {"phase": "loading", "op": None, "tail": 0, "windows": 0}
PHASES = {"update": "step", "checkpoint": "checkpoint", "save": "checkpoint",
          "load_checkpoint": "loading", "score": "probe"}


def progress_writer(every=None):
    out = os.environ.get("ARC_VALIDATION_OUTPUT")
    if not out:
        return None
    every = float(every or 60)
    path = Path(out) / "util" / "train_progress.jsonl"

    def run():
        last = (time.time(), 0, 0)
        while True:
            time.sleep(every)
            now, tail, windows = time.time(), PROGRESS["tail"], PROGRESS["windows"]
            row = {"t": round(now, 1), "phase": PROGRESS["phase"], "op": PROGRESS["op"], "tokens": tail - last[1],
                   "windows": windows - last[2], "tok_s": round((tail - last[1]) / max(1e-6, now - last[0]), 1)}
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            except OSError:
                pass
            last = (now, tail, windows)

    thread = threading.Thread(target=run, daemon=True, name="train-progress")
    thread.start()
    return thread
# R3: router "served" forces each row's top-k experts to the play server's (arena.served_route; the
# window's [T, layers, k] table in ROUTE["window"], the rows of the current model call in ROUTE["call"]; -1: the
# trainer's own pick). "agree" / "used" count rows whose own top-k set equals the served one (GPU, no sync).
ROUTE.update(window=None, call=None, agree=None, used=None, unserved_calls=0, layers=None)
# Expert mask (hero run): the pruned (t37) model as the server's expert_probe mask serves it: an
# additive MASK_VALUE on a pruned expert's router logit, so top-k and the renormalized weights see the kept experts
# only (the same routing as a renumbered cut; expert ids, LoRA layout and R3 tables unchanged).
EXPERT_MASK = {"path": None, "layers": 0, "pruned": 0}
MASK_VALUE = -1.0e4  # arc_infra.expert_probe_runtime.MASK_VALUE


def mask_logits(router_logits, mask):
    return router_logits if mask is None else router_logits + mask.to(router_logits.dtype)


def install_expert_mask(model, path, router_class="Qwen4ExpTextTopKRouter", patch=True):
    """Attach each layer's mask (format arc-expert-prune-v1) to its router; every router and every masked layer
    must match. Returns EXPERT_MASK."""
    import re

    import torch

    spec = json.loads(Path(path).read_text())
    if spec.get("format") != "arc-expert-prune-v1":
        raise ValueError(f"{path}: not an arc-expert-prune-v1 mask")
    experts, pruned = int(spec["experts"]), {int(k): [int(e) for e in v] for k, v in spec["pruned"].items()}
    if patch:
        route_patch()
    seen, total = set(), 0
    for name, module in model.named_modules():
        if type(module).__name__ != router_class:
            continue
        found = re.search(r"layers\.(\d+)\.mlp", name)
        layer = int(found.group(1)) if found else None
        if layer not in pruned:
            raise ValueError(f"{name}: no mask row for layer {layer}")
        if int(module.weight.shape[0]) != experts:
            raise ValueError(f"{name}: {tuple(module.weight.shape)} router, the mask has {experts} experts")
        ids = pruned[layer]
        if len(set(ids)) != len(ids) or any(not 0 <= e < experts for e in ids) or experts - len(ids) < int(module.top_k):
            raise ValueError(f"{name}: bad pruned ids")
        mask = torch.zeros(experts, dtype=torch.float32, device=module.weight.device)
        mask[torch.tensor(ids, dtype=torch.long, device=mask.device)] = MASK_VALUE
        module._arc_mask = mask
        seen.add(layer)
        total += len(ids)
    if seen != set(pruned):
        raise ValueError(f"expert mask: routers for {len(seen)} layers, the mask has {len(pruned)}")
    import hashlib

    EXPERT_MASK.update(path=str(path), layers=len(seen), pruned=total,
                       sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return dict(EXPERT_MASK)


def host_compact(model):
    """rl1_experts.compact_experts with the installed mask (EXPERT_MASK); no CUDA call (RL1_PRELOAD_HOST runs it on
    the host model before the gate, so the move carries only the kept experts)."""
    import rl1_experts

    if not EXPERT_MASK.get("path"):
        raise RuntimeError("expert compaction needs RL1_EXPERT_MASK")
    spec = json.loads(Path(EXPERT_MASK["path"]).read_text())
    pruned = {int(k): [int(e) for e in v] for k, v in spec["pruned"].items()}
    rl1_experts.install()
    return rl1_experts.compact_experts(model, pruned)


def rehome(model, device):
    """RL1_PRELOAD_HOST, after the move: the side tensors the host steps made (router masks, compaction ids) to
    ``device``, and the compaction's weight identities checked (the move swaps tensors in place: same objects).
    Returns {"tensors": moved, "compacted_layers": checked}."""
    import torch

    device = torch.device(device)
    moved = checked = 0
    for module in model.modules():
        mask = getattr(module, "_arc_mask", None)
        if torch.is_tensor(mask) and mask.device != device:
            module._arc_mask = mask.to(device)
            moved += 1
    experts = sys.modules.get("rl1_experts")
    if experts is not None and experts.COMPACT["on"]:
        from axolotl.integrations.kernels.libs.sonicmoe import experts as ax_experts

        infos = {id(info): info for info in experts.COMPACT["by"].values()}
        for info in infos.values():
            for key in ("kept", "remap"):
                if info[key].device != device:
                    info[key] = info[key].to(device)
                    moved += 1
        for name, module in model.named_modules():
            if type(module).__name__ != "Qwen4ExpTextExperts":
                continue
            w1, _, w2, _, _, _ = ax_experts._resolve_weights_and_lora(module)
            info = experts.COMPACT["by"].get(id(w1))
            if info is None:
                continue
            if experts.COMPACT["by"].get(id(w2)) is not info or w1.qdata.device != device:
                raise RuntimeError(f"rehome: {name} compaction lost its weights in the move")
            checked += 1
        if checked != experts.COMPACT["layers"]:
            raise RuntimeError(f"rehome: {checked} compacted layers found, {experts.COMPACT['layers']} compacted")
    return {"tensors": moved, "compacted_layers": checked}


def route_patch():
    """Qwen4ExpTextTopKRouter.forward with ROUTE's capture / replay (the original's arithmetic otherwise)."""
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    cls = modeling.Qwen4ExpTextTopKRouter
    if getattr(cls, "_rl1_route", False):
        return

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = mask_logits(F.linear(hidden_states, self.weight), getattr(self, "_arc_mask", None))
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        ROUTE["calls"] += 1
        if ROUTE["router"] == "served":
            router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
            want, col = ROUTE["call"], getattr(self, "_arc_col", None)
            if want is None or col is None:
                ROUTE["unserved_calls"] += 1
            else:
                want = want[:, col].to(device=router_probs.device, dtype=torch.long)
                if tuple(want.shape) != tuple(router_indices.shape):
                    raise RuntimeError(f"served route: {tuple(want.shape)} for a pass of {tuple(router_indices.shape)}")
                use = (want >= 0).all(-1, keepdim=True)
                same = (want.sort(-1).values == router_indices.sort(-1).values).all(-1, keepdim=True) & use
                ROUTE["agree"] = same.sum() if ROUTE["agree"] is None else ROUTE["agree"] + same.sum()
                ROUTE["used"] = use.sum() if ROUTE["used"] is None else ROUTE["used"] + use.sum()
                router_indices = torch.where(use, want, router_indices)
                router_top_value = router_probs.gather(-1, router_indices)
        elif ROUTE["router"] == "replay":
            if ROUTE["i"] >= len(ROUTE["rec"]):
                raise RuntimeError(f"route replay: call {ROUTE['i']} beyond the {len(ROUTE['rec'])} recorded")
            router_indices = ROUTE["rec"][ROUTE["i"]].to(device=router_probs.device, dtype=torch.long)
            ROUTE["i"] += 1
            if tuple(router_indices.shape) != (hidden_states.shape[0], self.top_k):
                raise RuntimeError(f"route replay: recorded {tuple(router_indices.shape)}, "
                                   f"pass {(hidden_states.shape[0], self.top_k)}")
            router_top_value = router_probs.gather(-1, router_indices)
        else:
            router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
            if ROUTE["router"] == "capture":
                ROUTE["rec"].append(router_indices.to(torch.int16))
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    cls.forward = forward
    cls._rl1_route = True


def route_columns(model, layers):
    """Each router's column in the served table: decoder layer N (its module path ``...layers.N.mlp.gate``) at
    ``layers.index(N)``; routers of layers the server did not record keep their own picks."""
    import re

    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    found = 0
    for name, module in model.named_modules():
        if isinstance(module, modeling.Qwen4ExpTextTopKRouter):
            match = re.search(r"layers\.(\d+)\.", name)
            layer = int(match.group(1)) if match else None
            module._arc_col = layers.index(layer) if layer in layers else None
            found += module._arc_col is not None
    ROUTE["layers"] = list(layers)
    return found


SERVED = {}  # served-route directory -> arena.served_route.Store
SERVED_AUDIT = {}  # resolver counts summed over the windows since the last reset


def served_delta(before):
    """The served audit's counts added since ``before`` (a copy of SERVED_AUDIT): one call's R3 coverage."""
    return {k: v - before.get(k, 0) for k, v in SERVED_AUDIT.items()
            if isinstance(v, (int, float)) and v != before.get(k, 0)}


RATIO_CLIP = {"gspo": {"mode": "seq", "low": 3e-4, "high": 4e-4},  # GSPO's eps (Zheng et al. 2025)
              "ppo": {"mode": "tok", "low": 0.2, "high": 0.28}}  # PPO with DAPO's clip-higher


def ratio_clip_spec(v):
    """cmd ``ratio_clip``: a preset name, {"mode": seq|tok, "low", "high"}, or off (None, "off", False)."""
    if v in (None, False, "off"):
        return None
    spec = RATIO_CLIP.get(v) if isinstance(v, str) else v
    if not isinstance(spec, dict) or spec.get("mode") not in ("seq", "tok") \
            or not all(isinstance(spec.get(k), (int, float)) and 0 < spec[k] < 1 for k in ("low", "high")):
        raise ValueError(f"ratio_clip: {v!r} is not gspo | ppo | off | {{mode: seq|tok, low, high}}")
    return {"mode": spec["mode"], "low": float(spec["low"]), "high": float(spec["high"])}


def exact_counts(made):
    """RL1_REPLY_IDS (29 Oct): a window's replies whose prompt is the served one token for token (digest), those
    checked, those left out of the loss: ``[same, checked, excluded]``."""
    rows = made.get("replies") or []
    return [sum(r.get("prompt_same") is True for r in rows), sum(r.get("prompt_same") is not None for r in rows),
            sum(bool(r.get("excluded")) for r in rows)]


def exact_summary(counts):
    """The update's ``prompt_same_rate`` (replies in their served context / replies checked), checked, excluded."""
    same, checked, excluded = counts or (0, 0, 0)
    return {"prompt_same_rate": round(same / checked, 5) if checked else None, "prompt_checked": checked,
            "prompt_excluded": excluded}


def served_call(start, stop):
    """The served rows of positions [start, stop) of the current window (or None)."""
    window = ROUTE["window"]
    if ROUTE["router"] != "served" or window is None:
        return None
    return window[start:stop]


CAUSAL = {"on": False, "key": None, "counts": None, "max": None, "hits": 0}  # MFU-1 stage S: plain_causal memo


def plain_causal(attention_mask):
    """Per query the number of visible keys when the mask is a plain causal one (batch 1, query i sees keys
    0..n_i-1), else None. ``CAUSAL["on"]``: the answer for the same mask tensor (storage, shape, version) is reused
    by the forward's next full-attention layers and the recompute (the check reads an [L, keys] mask: 4.2G elements
    at 32K x 130K, in each of the 12 layers)."""
    if CAUSAL["on"]:
        key = (attention_mask.data_ptr(), tuple(attention_mask.shape), attention_mask.dtype, attention_mask._version)
        if CAUSAL["key"] == key:
            CAUSAL["hits"] += 1
            return CAUSAL["counts"]
        counts = _plain_causal(attention_mask)
        # the largest count on the host with the check's own sync, so qsa_select needs none per layer (exact)
        top = int(counts.max()) if counts is not None and counts.numel() else None
        CAUSAL.update(key=key, counts=counts, max=top)  # cleared at every model call's start (causal_reset): the
        return counts  # mask is alive throughout one forward and its recompute, so its address is its own there
    return _plain_causal(attention_mask)


def causal_reset():
    CAUSAL.update(key=None, counts=None, max=None)


def _plain_causal(attention_mask):
    import torch

    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    if visible.shape[0] != 1:
        return None
    counts = visible[0, 0].sum(-1)
    positions = torch.arange(visible.shape[-1], device=visible.device)
    return counts if torch.equal(visible[0, 0], positions[None, :] < counts[:, None]) else None


TOPK = {"calls": 0}  # MFU-1 fast_topk
# QSA ties (c7j, 24 Oct): topk breaks ties at the budget's edge (often relu-zero scores) by the row's length, so a
# continued cache (other slice ends, other row lengths) picked other blocks than a fresh pass (tiny teacher: max 0.014
# at sub 1000; GPU verify top-1 0.934). RL1_QSA_TIES=index: ties go to the lowest block index, whatever the row length.
TIES = {"mode": os.environ.get("RL1_QSA_TIES") or "topk"}


def tie_by_index(masked, picked, top):
    """The ``top`` largest of each row of ``masked`` with ties at the k-th value going to the lowest indices (the set
    is independent of the row's length and of topk's order). ``picked``: topk's indices (their values give the k-th)."""

    kth = masked.gather(1, picked).min(dim=1, keepdim=True).values
    above, equal = masked > kth, masked == kth
    need = top - above.sum(dim=1, keepdim=True)
    take = above | (equal & (equal.cumsum(dim=1) <= need))
    return take.to(masked.dtype).topk(top, dim=-1).indices


def qsa_select(indexer, hidden_states, position_embeddings, counts, past_key_values, chunk=1024):
    """The QSA indexer's selection for every query at once (batch 1, plain causal): ``[queries, budget + ratio - 1]``
    key positions, -1 padding: the ``block_topk`` best complete blocks (``ratio`` keys each) and the query's
    incomplete tail. Updates the indexer cache (once per call, as the original forward)."""
    import math

    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    self = indexer
    batch, length, _ = hidden_states.shape
    device = hidden_states.device
    with torch.no_grad():
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -length:, :], full_sin[:, -length:, :]
        hidden_shape = (batch, length, -1, self.index_head_dim)
        q, token_k = torch.split(self.index_qk_proj(hidden_states),
                                 [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
                                 dim=-1)
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = modeling.apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
        ratio = self.compress_ratio
        TOPK.setdefault("cfg", [self.index_n_heads, self.index_head_dim, ratio, self.block_topk])
        complete = torch.div(counts, ratio, rounding_mode="floor")  # complete blocks per query
        known = CAUSAL["max"] if CAUSAL["on"] and CAUSAL["counts"] is counts else None
        blocks = known // ratio if known is not None else int(complete.max())  # floor(max) = max(floor): no sync
        top = min(self.block_topk, blocks)
        out = torch.full((length, top * ratio + ratio - 1), -1, dtype=torch.long, device=device)
        offsets = torch.arange(ratio, device=device)
        if blocks > 0:
            pooled = raw_keys[0, : blocks * ratio].reshape(blocks, ratio, self.index_head_dim)
            pooled = self.k_layernorm(pooled.float().mean(dim=1).to(raw_keys.dtype))
            starts = torch.arange(blocks, device=device) * ratio
            block_keys = modeling.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=full_cos[0].index_select(0, starts),
                                                       sin=full_sin[0].index_select(0, starts)).squeeze(1)
            keys_t = block_keys.float().transpose(0, 1)
            ids = torch.arange(blocks, device=device)
            for begin in range(0, length, chunk):
                end = min(length, begin + chunk)
                scores = torch.relu(torch.matmul(q[0, begin:end].float(), keys_t)).sum(dim=1)
                scores = scores / math.sqrt(self.index_head_dim)
                valid = ids[None, :] < complete[begin:end, None]
                if SELECT["fast"]:  # MFU-1 fast_topk: rl1_topk's exact per-row radix select (torch's selection, ascending)
                    import rl1_topk

                    picked = rl1_topk.topk_rows(scores, top, complete[begin:end])
                    TOPK["calls"] += 1
                else:
                    picked = scores.masked_fill(~valid, float("-inf")).topk(top, dim=-1).indices
                if TIES["mode"] == "index":
                    picked = tie_by_index(scores.masked_fill(~valid, float("-inf")), picked, top)
                tokens = picked[:, :, None] * ratio + offsets
                tokens = torch.where(torch.gather(valid, 1, picked)[:, :, None], tokens, -1)
                out[begin:end, : top * ratio] = tokens.reshape(end - begin, top * ratio)
        tail = (complete * ratio)[:, None] + offsets[: ratio - 1]
        out[:, top * ratio:] = torch.where(tail < counts[:, None], tail, -1)
    return out


# attention mode (RL1_ATTENTION): masked (the pinned masked SDPA), selected (``SelectedAttention``, the CPU tests'
# path) or triton (the fused kernel, ``triton_kernel``)
SPARSE = {"mode": os.environ.get("RL1_ATTENTION", "masked"), "chunk": 256, "calls": 0, "fallback": 0}
# MFU-1 stage S (22 Oct): within one grad window pass the full-attention layers' indexer picks and union tiles of the
# forward are kept per layer and reused by the checkpoint recompute (same inputs, so the same picks: exact), instead
# of running the indexer and the union build again. ``window``: a grad pass is running (window_pass).
SEL = {"on": False, "window": False, "store": {}, "hits": 0, "stores": 0}


def vectorized_indexer(chunk=1024):
    """The pinned transformers QSA indexer selects each query's key blocks in a Python loop over the queries (a
    ``nonzero`` host sync, an ``index_select`` of every visible key and a small matmul per query, in each of the 12
    full-attention layers, again under checkpoint recompute). Same selection, all queries at once (``qsa_select``),
    as the boolean mask the attention expects. Used when the mask is a plain causal one; else the original loop.
    No gradient flows through the selection."""
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    cls = modeling.Qwen4ExpTextQSAIndexer
    if getattr(cls, "_rl1_vectorized", False):
        return
    original = cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        counts = plain_causal(attention_mask)
        if counts is None:
            INDEXER["fallback"] += 1
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        INDEXER["vectorized"] += 1
        picked = qsa_select(self, hidden_states, position_embeddings, counts, past_key_values, chunk)
        length, kv_length = picked.shape[0], attention_mask.shape[-1]
        mask = torch.zeros((length, kv_length + 1), dtype=torch.bool, device=picked.device)
        mask.scatter_(1, torch.where(picked >= 0, picked, kv_length), True)
        mask = mask[None, None, :, :kv_length]
        if attention_mask.is_floating_point():
            mask = torch.where(mask, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return mask

    cls.forward = forward
    cls._rl1_vectorized = True
    cls._rl1_original = original
    cls._rl1_vectorized_forward = forward


def selected_attention_function():
    """Exact attention over each query's selected keys only (the indexer keeps at most budget + ratio - 1 = 2,051 of
    up to 131K keys; the masked SDPA computed every key: rl1-prof-v1, 65% of a window's card time). Queries in
    chunks: the selected keys and values gathered (fp32), softmax over them, the log-sum-exp kept; the backward
    gathers again and scatters the key and value gradients back (no gathered tensor is saved)."""
    import torch

    class SelectedAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, picked, scale, chunk):
            with torch.autocast(q.device.type, enabled=False):  # the products run in fp32
                return SelectedAttention.run(ctx, q, k, v, picked, scale, chunk)

        @staticmethod
        def run(ctx, q, k, v, picked, scale, chunk):
            heads, groups = q.shape[1], k.shape[1]
            rep, length = heads // groups, q.shape[2]
            out = torch.empty_like(q)
            lse = torch.empty((groups, length, rep), dtype=torch.float32, device=q.device)
            for begin in range(0, length, chunk):
                end = min(length, begin + chunk)
                p, _, vs, _, _ = SelectedAttention.probs(q, k, v, picked, scale, begin, end, lse, None)
                out[0, :, begin:end] = torch.matmul(p, vs).permute(0, 2, 1, 3).reshape(heads, end - begin, -1).to(q.dtype)
            ctx.save_for_backward(q, k, v, picked, out, lse)
            ctx.scale, ctx.chunk = scale, chunk
            return out

        @staticmethod
        def probs(q, k, v, picked, scale, begin, end, lse, known):
            heads, groups, dim = q.shape[1], k.shape[1], q.shape[-1]
            rep = heads // groups
            ids = picked[begin:end]
            valid = ids >= 0
            safe = ids.clamp(min=0)
            qs = q[0, :, begin:end].reshape(groups, rep, end - begin, dim).permute(0, 2, 1, 3).float()
            ks = k[0][:, safe].float()  # [groups, chunk, selected, dim]
            vs = v[0][:, safe].float()
            scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale  # [groups, chunk, rep, selected]
            scores.masked_fill_(~valid[None, :, None, :], float("-inf"))
            if known is None:
                lse[:, begin:end] = torch.logsumexp(scores, dim=-1)
            p = torch.exp(scores - lse[:, begin:end, :, None])
            return p, ks, vs, qs, safe

        @staticmethod
        def backward(ctx, grad):
            with torch.autocast(grad.device.type, enabled=False):
                return SelectedAttention.grads(ctx, grad)

        @staticmethod
        def grads(ctx, grad):
            q, k, v, picked, out, lse = ctx.saved_tensors
            heads, groups, dim = q.shape[1], k.shape[1], q.shape[-1]
            rep, length = heads // groups, q.shape[2]
            dq = torch.zeros_like(q, dtype=torch.float32)
            dk = torch.zeros_like(k[0], dtype=torch.float32)
            dv = torch.zeros_like(v[0], dtype=torch.float32)
            for begin in range(0, length, ctx.chunk):
                end = min(length, begin + ctx.chunk)
                p, ks, vs, qs, safe = SelectedAttention.probs(q, k, v, picked, ctx.scale, begin, end, lse, True)
                go = grad[0, :, begin:end].reshape(groups, rep, end - begin, dim).permute(0, 2, 1, 3).float()
                o = out[0, :, begin:end].reshape(groups, rep, end - begin, dim).permute(0, 2, 1, 3).float()
                delta = (go * o).sum(-1, keepdim=True)
                dp = torch.matmul(go, vs.transpose(-1, -2))
                ds = p * (dp - delta) * ctx.scale
                dq[0, :, begin:end] = torch.matmul(ds, ks).permute(0, 2, 1, 3).reshape(heads, end - begin, dim)
                flat = safe.reshape(-1)
                dk.index_add_(1, flat, torch.matmul(ds.transpose(-1, -2), qs).reshape(groups, -1, dim))
                dv.index_add_(1, flat, torch.matmul(p.transpose(-1, -2), go).reshape(groups, -1, dim))
            return dq.to(q.dtype), dk[None].to(k.dtype), dv[None].to(v.dtype), None, None, None

    return SelectedAttention


KERNEL = {"union": False}


def op_paths(model, cmd):
    """MFU stages per op (a run adopts them through mg.trainer_fast): union_attention (E), gdn_fast (G: rl1_gdn's
    fused GDN layer), the rest below; a key absent is off."""
    want_gdn = bool(cmd.get("gdn_fast"))
    if want_gdn or "rl1_gdn" in sys.modules:
        import rl1_gdn

        rl1_gdn.install()
        rl1_gdn.GDN["fast"] = want_gdn
        rl1_gdn.GDN["keep"] = bool(cmd.get("gdn_keep"))
    want_p = bool(cmd.get("prefix_nograd"))
    if want_p or "rl1_sparse2" in sys.modules:  # MFU-1 stage P: no dk / dv for the no-grad cached prefix's keys
        import rl1_sparse2

        rl1_sparse2.UnionSelectedAttention.NOGRAD_PREFIX = want_p
    SEL["on"] = bool(cmd.get("sel_cache"))
    CAUSAL["on"] = SEL["on"]
    causal_reset()
    want_hyper = bool(cmd.get("hyper_fast"))
    if want_hyper or "rl1_hyper" in sys.modules:  # MFU-1 stage H: fused hyper-connections (rl1_hyper)
        import rl1_hyper

        rl1_hyper.install()
        rl1_hyper.HYPER["on"] = want_hyper
        rl1_hyper.HYPER["xb"] = bool(cmd.get("hyper_xb"))  # MFU-1 (24 Oct): xn's BF16 cast once, in the norm kernel
    want_cast = bool(cmd.get("cast_once"))
    if want_cast or "rl1_cast" in sys.modules:  # MFU-1 item 2 (24 Oct): one BF16 copy per fp32 Linear input (rl1_cast)
        import rl1_cast

        rl1_cast.install(want_cast, model)
    want_deq = bool(cmd.get("fast_dequant"))
    if want_deq or "rl1_dequant" in sys.modules:  # MFU-1 stage Q: Axolotl's expert dequant, same bits, one kernel
        import rl1_dequant

        rl1_dequant.install()
        rl1_dequant.DEQUANT["on"] = want_deq
    NGRAM["on"] = bool(cmd.get("ngram_pinned"))
    if NGRAM["on"]:
        ngram_pinned_install(model)
    KERNEL["union"] = bool(cmd.get("union_attention"))
    fast = cmd.get("fast_topk")  # MFU-1: union kernel only (ascending picks)
    SELECT["fast"] = bool(fast) and KERNEL["union"]
    if fast:  # MFU-1 (24 Oct): rows past bisect's width by "torch" (default) or "filter"
        import rl1_topk

        rl1_topk.WIDE["method"] = str(cmd.get("fast_topk_wide") or "torch")
    rl1_memory.HEAD["chunked"] = bool(cmd.get("chunked_head"))  # MFU stage M: chunked LM head / log-prob reads
    tail = str(cmd.get("tail_ckpt") or "off")  # checkpoint-boundary offload (T1c-v3: 130K on CP2)
    if tail not in rl1_memory.TAIL_MODES:
        raise ValueError(f"tail_ckpt {tail!r} not in {rl1_memory.TAIL_MODES}")
    rl1_memory.TAIL["mode"] = tail
    rl1_memory.TAIL["min_tokens"] = int(cmd.get("tail_ckpt_min") or 0)
    rl1_memory.ckpt_settings(cmd)  # adaptive_ckpt: per window n layers uncheckpointed within ckpt_budget_gib
    want_x = bool(cmd.get("fused_experts"))
    if want_x or "rl1_experts" in sys.modules:  # MFU-1 stage X: the dequant path's expert elementwise work fused
        import rl1_experts

        rl1_experts.install()
        rl1_experts.EXPERTS["on"] = want_x


def triton_kernel():
    """The fused selected-key kernel: ``rl1_sparse2.UnionSelectedAttention`` (query-union tiles) under
    union_attention, else ``rl1_sparse.TritonSelectedAttention`` (one program per query; same call and results)."""
    if KERNEL["union"]:
        from rl1_sparse2 import UnionSelectedAttention

        return UnionSelectedAttention
    from rl1_sparse import TritonSelectedAttention

    return TritonSelectedAttention


def selected_attention():
    """Patch the full-attention layers: with a plain causal mask (batch 1) the indexer's selection goes to
    ``SelectedAttention`` or the Triton kernel as key positions (no mask is built); else the original forward.
    ``SPARSE["mode"]`` switches it."""
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    cls = modeling.Qwen4ExpTextAttention
    if getattr(cls, "_rl1_selected", False):
        return
    original = cls.forward
    function = selected_attention_function()

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        counts = (plain_causal(attention_mask) if SPARSE["mode"] in ("selected", "triton") and attention_mask is not None
                  else None)
        if counts is None or (self.training and self.attention_dropout):
            SPARSE["fallback"] += 1
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        SPARSE["calls"] += 1
        use_sel = SEL["on"] and SEL["window"]
        kept = SEL["store"].get(self.layer_idx) if use_sel else None
        if kept is not None and kept["length"] == hidden_states.shape[1] and torch.equal(kept["counts"], counts):
            picked = kept["picked"].long()  # the forward's picks (stored int32); the recompute skips the indexer
            SEL["hits"] += 1
        else:
            picked = qsa_select(self.indexer, hidden_states, position_embeddings, counts, past_key_values)
            kept = None
            if use_sel:
                kept = SEL["store"][self.layer_idx] = {"length": hidden_states.shape[1], "counts": counts,
                                                       "picked": picked.to(torch.int32), "tiles": None}
                SEL["stores"] += 1
        position_embeddings = (x[:, -hidden_states.shape[1]:, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = modeling.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if SPARSE["mode"] == "triton":  # the fused gather kernel (triton_kernel)
            kernel = triton_kernel()
            if use_sel and hasattr(kernel, "TILE_CACHE"):  # stage S: the union tiles with the picks
                kernel.TILE_CACHE = SEL["store"].get(self.layer_idx)
            try:
                attn_output = kernel.apply(query_states, key_states, value_states, picked, self.scaling, None)
            finally:
                if hasattr(kernel, "TILE_CACHE"):
                    kernel.TILE_CACHE = None
        else:
            attn_output = function.apply(query_states, key_states, value_states, picked, self.scaling, SPARSE["chunk"])
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    cls.forward = forward
    cls._rl1_selected = True
    cls._rl1_original = original


DRAFT_PREFIX = "draft."  # the MTP draft head's LoRA parameters in ``Trainer.params`` (rl1_draft, stage D)
DRAFT_KEYS = ("kl", "rows", "top1", "agree", "paccept")


def draft_settings():
    """Stage D (rl1_draft): RL1_DRAFT_WEIGHT (mg.draft_weight; 0 = off) the draft's KL weight; RL1_DRAFT_LR_SCALE
    (mg.draft_lr_scale) its LoRA's lr as a multiple of the step's; RL1_DRAFT_FP4 (default 1) its routed experts at
    NVFP4 values, as served."""
    return {"weight": float(os.environ.get("RL1_DRAFT_WEIGHT") or 0.0),
            "lr_scale": float(os.environ.get("RL1_DRAFT_LR_SCALE") or 1.0),
            "fp4": os.environ.get("RL1_DRAFT_FP4", "1") != "0"}


def draft_hook(model):
    """A pre-hook on the trunk's final hyper-connection mixer: its input (the pre-mixer state the MTP draft reads)
    goes to ``box["hc"]`` (detached) while ``box["on"]``."""
    mixers = [m for n, m in model.named_modules() if n.endswith("hyper_connection_mixer") and ".layers." not in n]
    if len(mixers) != 1:
        raise RuntimeError(f"draft: {len(mixers)} final hyper-connection mixers")
    box = {"on": False}

    def hook(module, args, kwargs):
        if box["on"]:
            box["hc"] = (args[0] if args else kwargs["hyper_input"]).detach()

    box["handle"] = mixers[0].register_forward_pre_hook(hook, with_kwargs=True)
    return box


def draft_positions(piece, device):
    """``positions(a, b)``: the trunk's position ids of [a, b) (text positions when the pass has none)."""
    import torch

    def at(a, b):
        pos = piece(a, b).get("position_ids")
        return pos if pos is not None else torch.arange(a, b, device=device).view(1, 1, -1).expand(3, 1, -1)
    return at


def window_key(spec, made):
    """A window's θ0 store key: its name plus the try's rollout file. Names repeat across groups that share a
    start (labels a000-<game>-L<k>-<start>-p0..; a10 U23: 8 of 67 windows lost θ0 to a same-start twin)."""
    rollout = spec.get("rollout")
    return f"{rollout}::{made['name']}" if rollout else made["name"]


class Trainer:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor

        self.args, self.dir = args, Path(args.dir)
        # RL1_DIST=1 (torchrun): this process's rank and groups; None: one GPU. RL1_PRELOAD_HOST: the card is set after
        # the gate (fq.load), not here
        rl1_firstcall.install()  # RL1_FIRST_CALLS: compile / autotune timers before anything compiles
        self.dist = rl1_dist.init(defer_device=fq.preload_wanted())
        records = self.dir / "trainer" if self.dist is None else rl1_dist.recorder_dir(self.dir / "trainer",
                                                                                         self.dist.rank)
        self.encoders = None
        prespawn = int(os.environ.get("RL1_ENCODE_PROCS") or 0)
        if prespawn:  # cold start (b): the encoder processes start (and load their processors) beside the model load
            warm = os.environ.get("RL1_ENCODE_WARM") == "1"
            pool = self.encoder_pool(prespawn, warm=warm)  # (the pool spawns a process per submit)
            if warm:  # each process's first window return path paid now, before the gate (encoder_warm in loaded)
                self.encoder_warm = self.warm_encoders(pool, prespawn)
            else:  # one no-op each, not awaited
                self.encoder_warm = [pool.submit(os.getpid) for _ in range(prespawn)]
        if os.environ.get("RL1_ROUTER"):  # R3 from the first window (rl_seq F1 encodes saved groups at the load)
            if os.environ["RL1_ROUTER"] not in ("off", "served"):
                raise ValueError(f"RL1_ROUTER: {os.environ['RL1_ROUTER']}")
            route_patch()
            ROUTE["router"] = os.environ["RL1_ROUTER"]
        self.early = {"token": None}
        EARLY["hook"] = self.early_ahead  # acts only once the loop writes ahead-early.json (trainer_fast.early_ahead)
        self.prefix_chunk = self.path_chunk = int(getattr(args, "prefix_chunk", 16384) or 16384)
        (self.dir / "cmd").mkdir(parents=True, exist_ok=True)
        (self.dir / "done").mkdir(parents=True, exist_ok=True)
        self.rec = fq.Recorder(records)
        self.rec.save(environment=fq.environment(self.rec))
        # RL1_SHARED_TABLE: the n-gram table built (local rank 0) or awaited (the others) on a thread beside the load
        shared_table = rl1_table.start(args.model)
        fq.set_adapter(args.rank)
        self.processor = AutoProcessor.from_pretrained(args.model)
        self.tokenizer = self.processor.tokenizer
        # transformers pre-reserves the checkpoint's BF16 size (266 GiB) before loading; beside the sleeping
        # server (3.6 GiB) that fails, and the load itself needs about 84 GiB (rl-experts-v1)
        import transformers.modeling_utils as modeling_utils

        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        if not fq.preload_wanted():  # RL1_PRELOAD_HOST: no CUDA call before the gate (fq.load resets after the move)
            fq.reset_peaks()
        began = time.time()
        init_s, clock = {}, [time.time()]  # seconds per post-load step (H attempt 2: 49 s after the load, gated)

        def lap(name):
            now = time.time()
            init_s[name] = round(now - clock[0], 2)
            clock[0] = now

        draft = draft_settings()

        def host_steps(model, preloaded):
            """The post-load steps that need no card: before the gate under RL1_PRELOAD_HOST (the move then carries
            the compacted experts), else after the load as before. Same order either way."""
            clock[0] = time.time()
            if shared_table:  # RL1_SHARED_TABLE: the scaled table, built once per launch (rl1_table), attached
                out = {"table_scale": [], "shared_table": rl1_table.attach(self.rec, model)}
                lap("shared_table")
            else:
                out = {"table_scale": fq.apply_table_scale(self.rec, model, args.model)}
                lap("table_scale")
            if os.environ.get("RL1_EXPERT_MASK"):  # hero run: the t37 mask (the server's expert_probe mask)
                out["expert_mask"] = install_expert_mask(model, os.environ["RL1_EXPERT_MASK"])
                lap("expert_mask")
                if os.environ.get("RL1_EXPERT_COMPACT") == "1":  # MFU-1: the masked experts' base freed
                    out["expert_compact"] = host_compact(model) if preloaded else self.compact_experts(model)
                    lap("expert_compact")
            from axolotl.integrations.kernels.libs.sonicmoe import set_merge_aware_enabled

            set_merge_aware_enabled(False)  # plain LoRA over the dequantised base: what the server computes
            grouped_expert_lora()
            vectorized_indexer()
            selected_attention()  # SPARSE["mode"]: RL1_ATTENTION (masked by default; runs set triton)
            lap("patches")
            out["head"] = rl1_memory.head_capture(model)  # MFU stage M: hidden rows out while HEAD["capture"]
            lap("hooks")
            if os.environ.get("RL1_HUB_PREFETCH") == "1":  # the hub kernels the first forward would fetch (D2)
                out["hub_prefetch"] = rl1_hubk.prefetch(model)
                lap("hub_prefetch")
            if preloaded and draft["weight"] <= 0:  # the LoRA A init, copied on the host (no card-to-host copy later)
                out["initial"] = {n: p.detach().clone() for n, p in model.named_parameters()
                                  if p.requires_grad and "lora_A" in n}
                lap("initial")
            if preloaded:
                print(json.dumps({"event": "rl1_host_steps", **init_s}), flush=True)
            return out

        self.model, _, loaded = load_model(self.rec, args.model, records,
                                           before_gate=lambda model: host_steps(model, True), table_meta=shared_table)
        early = (loaded.get("phases") or {}).pop("before_gate", None)
        clock[0] = time.time()
        if early is not None:  # RL1_PRELOAD_HOST: the side tensors the host steps made follow the model to the card
            loaded["phases"]["host_steps_s"] = dict(init_s)
            init_s.clear()
            loaded["rehomed"] = rehome(self.model, CARD)
            lap("rehome")
        else:
            early = host_steps(self.model, False)
        if self.dist is not None:  # RL1_NCCL_WARM: the NCCL setup on a thread beside the first accumulate
            loaded["nccl_warm"] = self.dist.warm(CARD if self.dist.backend == "nccl" else "cpu")
        for key in ("table_scale", "shared_table", "expert_mask", "expert_compact", "hub_prefetch"):
            if key in early:
                loaded[key] = early[key]
        self.head = early["head"]
        self.model.train()
        self.params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.draft = None
        if draft["weight"] > 0:  # stage D: the MTP draft head's LoRA beside the policy's (rl1_draft)
            import rl1_draft

            text = getattr(self.model.config, "text_config", self.model.config)
            began_draft = time.time()
            self.attach_draft(rl1_draft.Draft.build(text, args.model, rank=fq.ADAPTER["rank"], alpha=fq.ADAPTER["alpha"],
                                                    device=CARD, weight=draft["weight"], fp4=draft["fp4"]),
                              draft["lr_scale"])
            loaded["draft"] = {**draft, "params": sum(p.numel() for _, p in self.draft.params()),
                               "seconds": round(time.time() - began_draft, 1)}
        lap("train_params_draft")
        self.initial = early.get("initial")
        if self.initial is None or set(self.initial) != {n for n, _ in self.params if "lora_A" in n}:
            self.initial = {n: p.detach().to("cpu", copy=True) for n, p in self.params if "lora_A" in n}
        lap("initial")
        self.optimizer = self.make_optimizer(args.lr)
        lap("optimizer")
        if os.environ.get("RL1_STEP_WARM", "1") == "1":  # U3: the step's first-call costs paid at load
            loaded["step_warm"] = warm_step([p for _, p in self.params], len(self.optimizer.param_groups))
            lap("step_warm")
        self.checkpointing = "full"
        set_checkpointing(self.model, self.checkpointing)
        lap("checkpointing")
        self.encoded, _ = fq.example_maker(self.processor, [], tail=args.window, phases=[args.window])
        lap("example_makers")
        self.window_cache = rl1_encode.Cache()  # stage W (window_cache): never filled since the old pass's removal
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>")
        self.references = {}  # probe name -> {"base": [...], "previous": [...]} per-token actor log-probabilities
        self.updates = 0
        self.parked, self.pinned = False, {}
        loaded["prepin"] = self.start_prepin()  # the park's pinned buffers allocated from here on (T4r-v2)
        lap("prepin")
        loaded["base_snap"] = self.start_base_snap()  # the base snapshot from here on (gate leg 5)
        lap("base_snap")
        used, total, _ = gpu_used_gib()
        lap("gpu_used")
        loaded["init_s"] = init_s
        if rl1_firstcall.INSTALLED:
            loaded["first_calls"] = rl1_firstcall.report()
        if (getattr(self, "early", None) or {}).get("token") is not None:  # early_ahead: encodes before the gate
            loaded["early_ahead"] = {k: v for k, v in self.early.items() if k != "began"}
        if isinstance(getattr(self, "encoder_warm", None), dict):  # RL1_ENCODE_WARM: done long before the gate
            loaded["encoder_warm"] = {k: v for k, v in self.encoder_warm.items() if k not in ("thread", "began")}
        loaded.update(experts="dequant", seconds=round(time.time() - began, 1), trainable=sum(p.numel() for _, p in self.params),
                      trainable_dtypes=sorted({str(p.dtype) for _, p in self.params}),
                      gpu_allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 2), card_used_gib=used,
                      card_total_gib=total, adapter_groups=loaded.get("adapter_groups"))
        torch.cuda.empty_cache()
        loaded["load_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        self.loaded = loaded
        self.rec.save(loaded=loaded)

    def make_optimizer(self, lr):
        """Full 32-bit fused AdamW; the draft's LoRA (stage D) last, in its own group."""
        import torch

        params = [p for _, p in self.policy_params()]
        draft = [p for n, p in self.params if n.startswith(DRAFT_PREFIX)]
        if draft:  # a policy-only checkpoint's group comes first
            params = [{"params": params}, {"params": draft, "scale": float(getattr(self, "draft_lr_scale", 1.0)),
                                           "draft": True}]
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
        rl1_table.guard(getattr(self, "model", None), optimizer, "optimizer")  # RL1_SHARED_TABLE: stays frozen
        return optimizer

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("scale", 1.0))

    # ---- stage D: the MTP draft head's LoRA (rl1_draft) ----

    def policy_params(self):
        return [(n, p) for n, p in self.params if not n.startswith(DRAFT_PREFIX)]

    def draft_params(self):
        return [(n, p) for n, p in self.params if n.startswith(DRAFT_PREFIX)]

    def attach_draft(self, draft, lr_scale=1.0):
        """The draft's LoRA after the policy's in ``params`` (the optimizer, checkpoints, parking and the rank
        reductions carry it; the policy's divisor, clip, norms and adapter export leave it out); the trunk-state hook.
        Build ``initial`` and the optimizer after this."""
        self.draft, self.draft_lr_scale = draft, float(lr_scale)
        self.params = self.policy_params() + draft.params()
        self.draft_box = draft_hook(self.model)
        self.draft_live = False
        self.draft_reset()

    def draft_reset(self):
        self.draft_sums = {**{k: 0 for k in DRAFT_KEYS}, "windows": 0, "seconds": 0.0, "skipped": {}}

    def draft_add(self, out):
        if not out:
            return
        if "skipped" in out:
            self.draft_sums["skipped"][out["skipped"]] = self.draft_sums["skipped"].get(out["skipped"], 0) + 1
            return
        for k in DRAFT_KEYS:
            self.draft_sums[k] += out[k]
        self.draft_sums["windows"] += 1
        self.draft_sums["seconds"] += out.get("seconds", 0.0)

    def draft_window(self, made, ids, cut, cache, hc, rows, chunked, piece, kept=None):
        """Single GPU: the window's draft rows on its prefix cache's draft state (rl1_draft.Draft.window)."""
        import torch

        began = time.time()
        if hc is None:
            return {"skipped": "no trunk state"}
        state = getattr(cache, "_rl1_draft", None) if cache is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.draft.window(state, ids, cut, hc, draft_positions(piece, ids.device),
                                    self.model.get_input_embeddings(), self.model.get_output_embeddings().weight,
                                    rows.detach(), self.head._rl1_project if chunked else None,
                                    made["arrays"]["positions"] if kept is None else kept, made["temperature"])
        out["seconds"] = round(time.time() - began, 2)
        return out

    def draft_cp(self, made, ids, cut, pc, cp, hc_pre, hc_tail, full, rows, chunked, kept):
        """RL1_DIST sequence parallel: the window's draft rows over the trunk's shards (rl1_draft.Draft.cp_prefix /
        cp_window), its sums reduced over the ranks (every rank returns the window's)."""
        import torch

        began = time.time()
        total = ids.shape[1]
        positions = full if full is not None else torch.arange(total, device=ids.device).view(1, 1, -1).expand(3, 1, -1)
        embed, lm = self.model.get_input_embeddings(), self.model.get_output_embeddings().weight
        if hc_tail is None or (cut and hc_pre is None):
            raise RuntimeError("draft: no trunk state captured in a CP window")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prefix = last = None
            if cut:
                prefix, last = self.draft.cp_prefix(pc, ids, hc_pre, positions, embed)
            sums = self.draft.cp_window(cp, ids, cut, hc_tail, prefix, last, positions, embed, lm, rows.detach(),
                                        self.head._rl1_project if chunked else None, kept, made["temperature"])
        vals = self.dist.reduce_sums([float(sums[k]) for k in DRAFT_KEYS], hc_tail.device)
        out = dict(zip(DRAFT_KEYS, vals))
        out["rows"] = int(round(out["rows"]))
        out["top1"], out["agree"] = int(round(out["top1"])), int(round(out["agree"]))
        out["seconds"] = round(time.time() - began, 2)
        return out

    def draft_finish(self, cmd):
        """The step's draft sums (over the ranks), its gradients divided by its rows and clipped on their own
        (at 1.0); a non-finite norm drops the draft's step only. The result's draft fields."""
        import torch

        sums = self.draft_sums
        vals = [float(sums[k]) for k in DRAFT_KEYS] + [float(sums["windows"]), float(sums["seconds"])]
        if getattr(self, "dist", None) is not None:
            vals = self.dist.reduce_sums(vals, self.params[0][1].device)
        tot = dict(zip(DRAFT_KEYS + ("windows", "seconds"), vals))
        rows = int(round(tot["rows"]))
        params = [p for _, p in self.draft_params()]
        out = {"draft_rows": rows, "draft_windows": int(round(tot["windows"])), "draft_seconds": round(tot["seconds"], 1),
               "draft_skipped": dict(sums["skipped"]) or None}
        if not rows:
            for p in params:
                p.grad = None
            return out
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.div_(rows)
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], 1.0))
        if not math.isfinite(norm):
            for p in params:
                p.grad = None
        out.update(draft_kl=round(tot["kl"] / rows, 6), draft_top1=round(tot["top1"] / rows, 4),
                   draft_agree=round(tot["agree"] / rows, 4), draft_paccept=round(tot["paccept"] / rows, 4),
                   draft_grad_norm=round(norm, 6), draft_applied=math.isfinite(norm))
        return out

    # ---- windows ----

    def attempt_window(self, spec, seed=0, release=True):
        """One window of a recorded attempt (``rollout``: its rollout.jsonl; ``w``: its weight; the heaviest
        window, the most reply tokens): ``rl_job.attempt_example`` on the compacted rows."""
        import torch

        made, rows = rl1_encode.encode_window(self.encoded, self.think_end, self.args.window, spec, seed, device=CARD)
        made = self.finish_window(spec, made, rows)
        if release:
            torch.cuda.empty_cache()
        return made

    def finish_window(self, spec, made, rows=None):
        """The card-side parts of a window: R3's served routes."""
        if made is None:
            return None
        made["key"] = window_key(spec, made)
        served = spec.get("served_route") and ROUTE["router"] == "served"
        hosted = "served_host_audit" in made  # resolved in an encoder process (rl1_encode.window)
        if served and not hosted:
            rows = rl1_encode.read_rows(spec) if rows is None else rows
        if served and hosted:
            table = made.pop("served_host")
            made["served"], made["served_audit"] = self.served_account(
                None if table is None else table.to(CARD), made.pop("served_host_audit"), made.pop("served_layers"))
        elif served:
            made["served"], made["served_audit"] = self.served_routing(spec, rows, made)
        else:
            for key in ("served_host", "served_host_audit", "served_layers"):
                made.pop(key, None)
        return made

    def encoder_pool(self, procs, warm=False):
        """``procs`` encoder processes (spawned: no CUDA), kept across ops; a new count or a broken pool is replaced.
        Windows come back through shared memory when /dev/shm has room for ~2 GiB per process, else pickled.
        ``warm``: the processes share a barrier for ``warm_encoders``."""
        if self.encoders is not None and (self.encoders[0] != procs or getattr(self.encoders[1], "_broken", False)):
            self.encoders[1].shutdown(wait=False, cancel_futures=True)
            self.encoders = None
        if self.encoders is None:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            context = multiprocessing.get_context("spawn")
            shm = rl1_encode.shm_free_gib() >= 2.0 * procs + 4.0
            barrier = context.Barrier(procs) if warm else None
            pool = ProcessPoolExecutor(procs, mp_context=context, initializer=rl1_encode.init,
                                       initargs=(self.args.model, self.args.window, getattr(self, "encoder_loader", None),
                                                 shm, barrier))
            self.encoders = (procs, pool, shm)
        return self.encoders[1]

    def warm_encoders(self, pool, procs):
        """RL1_ENCODE_WARM: one ``rl1_encode.warm`` per encoder process, collected on a thread (this side's first
        receive too); the row (processes reached, seconds) lands in ``loaded["encoder_warm"]``."""
        import threading

        state = {"procs": procs, "began": time.time()}
        futures = [pool.submit(rl1_encode.warm) for _ in range(procs)]

        def collect():
            pids, met = set(), []
            try:
                for future in futures:
                    made = rl1_encode.unpack(future.result())
                    pids.add(made["pid"])
                    met.append(made["met"])
                    del made
            except Exception as error:  # noqa: BLE001 - a cold first window, as before
                state["error"] = repr(error)[:300]
            state.update(processes=len(pids), met=sum(bool(m) for m in met),
                         seconds=round(time.time() - state["began"], 1))

        state["thread"] = threading.Thread(target=collect, name="encoder-warm", daemon=True)
        state["thread"].start()
        return state

    def host_window(self, spec, made, rows=None):
        """A host-encoded window on the card."""
        if made is None:
            return None
        return self.finish_window(spec, rl1_encode.to_device(made, CARD, self.mrope_how()), rows)

    def collect_host(self, spec, pending):
        """An encoder process's window; a failed or broken process falls back to this process's encode."""
        try:
            made = rl1_encode.unpack(pending.result())
        except Exception as error:
            self.encode_errors = getattr(self, "encode_errors", []) + [repr(error)[:300]]
            return self.attempt_window(spec, 0, False)
        return self.host_window(spec, made, None)

    def mrope_how(self):
        """U2 host index work (RL1_HOST_INDEX, default on): what ``rl1_encode.host_mrope`` needs (the model core's
        class and merge size), so a window's M-RoPE positions are made on the host (encoder process or feed thread);
        None: off (positions by the card rule, every image's grid read from the card)."""
        if os.environ.get("RL1_HOST_INDEX", "1") != "1":
            return None
        if getattr(self, "_mrope_how", None) is None:
            model = getattr(self, "model", None)
            core = getattr(model.get_base_model() if hasattr(model, "get_base_model") else model, "model", None)
            vision = getattr(getattr(core, "config", None), "vision_config", None)
            self._mrope_how = {"cls": f"{type(core).__module__}.{type(core).__qualname__}",
                               "merge": int(vision.spatial_merge_size)} if vision is not None else False
        return self._mrope_how or None

    def host_of(self, made):
        """The window's host tensors for index work (``rl1_encode.to_device``); None under RL1_HOST_INDEX=0."""
        return made.get("host_inputs") if self.mrope_how() is not None else None

    def vision_ahead(self, made):
        """vision_ahead (feed thread, on the feed stream): the window's image features, as ``positions_and_images``
        computes them on the main thread (frozen tower, same call), so the pass starts without the tower's run."""
        if made is None or made["inputs"].get("pixel_values") is None:
            return made
        made["vision_features"] = fq.image_features(self.model, made["inputs"], self.host_of(made))
        return made

    def early_ahead(self):
        """trainer_fast.early_ahead (D2 27 Oct: the update's first window waited 22.8 s on its encode, the trainer
        idle at the gate for 1,620 s): the loop writes ``ahead-early.json`` ({"token", "ahead": the first group's
        specs, the call's args}) when a play phase ends; here (at the GPU gate and on idle polls) its attempts are
        expanded and every window encoded into ``ahead_windows``, which the next accumulate takes by key."""
        path = self.dir / "ahead-early.json"
        st = self.early
        if not path.exists():
            return
        if st.get("state") in (None, "done"):
            try:
                cmd = json.loads(path.read_text())
            except ValueError:  # being written
                return
            if cmd.get("token") == st.get("token"):
                return
            procs = int(cmd.get("encode_procs") or 0)
            st.clear()
            st.update(token=cmd.get("token"), state="done", began=time.time(), windows=0)
            if not procs or not cmd.get("ahead"):
                st["why"] = "no encode_procs or no specs"
                return
            self.ahead_start(cmd)
            st.update(state="expanding", procs=procs)
        if st["state"] != "expanding":
            return
        specs = self.ahead_specs()
        if specs is None:
            return
        pool = self.encoder_pool(st["procs"])
        windows = getattr(self, "ahead_windows", None) or {}
        for spec in specs:
            hosted = {**spec, "served_in_encoder": True} if ROUTE["router"] == "served" and spec.get("served_route") \
                else spec
            hosted = {**hosted, "mrope": self.mrope_how()} if self.mrope_how() else hosted
            key = rl1_encode.spec_key(spec)
            if key not in windows:  # a cumulative ahead file (rl_seq one_call): the encoded ones are not sent again
                windows[key] = pool.submit(rl1_encode.window, hosted, 0)
        self.ahead_windows = windows
        self.ahead_plan = None
        st.update(state="done", windows=len(specs), seconds=round(time.time() - st["began"], 2))
        print(json.dumps({"event": "early_ahead", **{k: v for k, v in st.items() if k != "began"}}), flush=True)

    def ahead_start(self, cmd):
        """``ahead`` (MFU-1, 23 Oct): the attempts of the loop's next call (its next group).
        Their rollouts are parsed in the encoder processes now (no wait) and, once this call's own windows are all
        with the encoders, their first windows are encoded too, so the next call's first window is ready when it
        starts instead of after a ~10 s encode with the card idle. Same specs, same encode: the next call takes the
        encoded window by its key, else encodes as before."""
        ahead, procs = cmd.get("ahead") or [], int(cmd.get("encode_procs") or 0)
        self.ahead_plan = None
        if not ahead or not procs:
            return
        if cmd.get("whole_segment"):  # as expand_cmd (gate-v1/v2 28 Oct: 0 ahead hits; the ahead windows were planned
            ahead = [whole_segment(a) for a in ahead]  # as plain "whole" specs, so no key matched the call's)
        pool = self.encoder_pool(procs)
        memo = getattr(self, "expand_memo", None)
        if memo is None:
            memo = self.expand_memo = {}
        parts = []
        for spec in ahead:
            if not spec.get("whole"):
                parts.append([spec])
                continue
            stat = Path(spec["rollout"]).stat()
            key = json.dumps([spec, int(self.args.window), stat.st_size, stat.st_mtime_ns], sort_keys=True, default=str)
            parts.append((key, memo[key]) if key in memo else (key, pool.submit(rl1_encode.expand_one, spec, int(self.args.window))))
        self.ahead_plan = {"parts": parts, "signed": bool(cmd.get("signed")), "share": bool(cmd.get("share_prefix")),
                           "sent": set()}

    def ahead_specs(self):
        """The next call's windows in its order, once every attempt's expansion is in (else None: not yet)."""
        plan = self.ahead_plan
        if plan is None:
            return None
        out = []
        for part in plan["parts"]:
            if isinstance(part, list):
                out += part
                continue
            key, got = part
            if hasattr(got, "done"):
                if not got.done():
                    return None
                try:
                    got = got.result()
                except Exception:  # noqa: BLE001 - the next call expands it itself
                    return None
                self.expand_memo[key] = got
            out += [dict(x) for x in got]
        out = [a for a in out if a["w"] > 0 or (plan["signed"] and a["w"] < 0)]
        if plan["share"]:
            groups = {}
            for a in out:
                groups.setdefault(a.get("share"), len(groups))
            out = sorted(out, key=lambda a: (groups[a.get("share")], not a.get("first_window", True)))
        return out

    def window_feed(self, specs, cmd):
        """The op's windows in order, encoded ahead of the card: ``(next(), close(), stats)``. Default: one encode
        thread, ``prefetch`` windows queued ahead (MFU 3). Stage W (trainer_fast, off by default): ``encode_procs``
        encoder processes run ``max(prefetch, encode_procs)`` windows ahead on the host and the thread only moves
        ``prefetch`` of them to the card; ``window_cache`` (GiB): windows taken from ``self.window_cache`` instead of
        encoding again; ``vision_ahead`` (with feed_stream): the feed thread runs the vision tower on the feed stream."""
        import collections
        from concurrent.futures import ThreadPoolExecutor

        procs, cap = int(cmd.get("encode_procs") or 0), float(cmd.get("window_cache") or 0)
        use = cap > 0
        encoders = self.encoder_pool(procs) if procs else None
        thread = ThreadPoolExecutor(1)
        # MFU-1 (23 Oct, m1h1): the feed thread's host-to-card copies on their own stream. On the shared default stream
        # a window's copy queued behind the running window's kernels, so the feed never got ahead (1.2-1.4 s a try of
        # main-thread wait). The main stream waits on the window's event; its tensors are recorded on the main stream
        # so the allocator does not hand their blocks back to the feed stream while the pass still reads them.
        side = None
        if cmd.get("feed_stream") and str(CARD).startswith("cuda"):
            import torch

            side = getattr(self, "feed_stream", None) or torch.cuda.Stream(device=CARD)
            self.feed_stream = side
        ahead_vision = bool(cmd.get("vision_ahead")) and side is not None

        def submit(fn, *args):
            if side is None:
                return thread.submit(fn, *args)

            def streamed():
                import torch

                with torch.cuda.stream(side):
                    made = fn(*args)
                    if ahead_vision:
                        made = self.vision_ahead(made)
                    if made is not None:
                        made["_ready"] = torch.cuda.Event()
                        made["_ready"].record(side)
                return made
            return thread.submit(streamed)
        card_ahead = max(1, int(cmd.get("prefetch") or 1))
        host_ahead = max(card_ahead, procs)
        queue, hosted = collections.deque(), {}
        at = {"card": 0, "host": 0}
        stats = {"waits": 0.0, "cache_hits": 0, "encoded": 0, "held": 0, "ahead_hits": 0, "ahead_sent": 0}
        self.encode_errors = []
        carried = {}
        if specs:  # the last call's ahead encodes (a call without windows, e.g. op_step, leaves them for the next)
            carried, self.ahead_windows = getattr(self, "ahead_windows", None) or {}, {}
        keyed = use or bool(carried)
        keys = [rl1_encode.spec_key(spec) for spec in specs] if keyed else [None] * len(specs)

        how = self.mrope_how()

        def hosted_spec(spec):  # R3 on: the encoder process resolves the window's served routes too
            if ROUTE["router"] == "served" and spec.get("served_route"):
                spec = {**spec, "served_in_encoder": True}
            return {**spec, "mrope": how} if how else spec  # U2: the M-RoPE positions made there too

        def cached(i):
            return use and keys[i] in self.window_cache.rows

        def host_submit():  # encoder processes: windows beyond the card queue, on the host
            i = at["host"]
            at["host"] += 1
            if not cached(i):
                hosted[i] = carried.pop(keys[i], None) or encoders.submit(rl1_encode.window, hosted_spec(specs[i]), 0)
                stats["ahead_hits"] += keys[i] is not None and hosted[i] is not None and keys[i] in taken
                taken.discard(keys[i])

        def card_submit():
            i = at["card"]
            at["card"] += 1
            spec, key = specs[i], keys[i]
            if not (procs or use):
                return queue.append(submit(self.attempt_window, spec, 0, False))
            hit = self.window_cache.pop(key) if use else None
            if hit is not None:
                stats["cache_hits"] += 1
                hosted.pop(i, None)
                return queue.append(submit(self.host_window, spec, hit))
            stats["encoded"] += 1
            if encoders is not None:
                pending = hosted.pop(i, None) or encoders.submit(rl1_encode.window, hosted_spec(spec), 0)
                return queue.append(submit(self.collect_host, spec, pending))
            queue.append(submit(self.attempt_window, spec, 0, False))

        taken = set(carried)  # stats: a host submit that took a carried future

        def send_ahead():  # this call's windows all with the encoders: the next call's first windows go too
            plan = getattr(self, "ahead_plan", None)
            if encoders is None or plan is None or len(plan["sent"]) >= host_ahead:
                return
            nxt = self.ahead_specs()
            if nxt is None:
                return
            for spec in nxt[:host_ahead]:
                key = rl1_encode.spec_key(spec)
                if key not in plan["sent"]:
                    plan["sent"].add(key)
                    self.ahead_windows[key] = encoders.submit(rl1_encode.window, hosted_spec(spec), 0)
                    stats["ahead_sent"] += 1

        def fill(number):
            if encoders is not None:
                while at["host"] < min(len(specs), number + host_ahead):
                    host_submit()
                if at["host"] >= len(specs):
                    send_ahead()
            while at["card"] < min(len(specs), number + card_ahead):
                card_submit()

        fill(0)

        def next_window():
            import torch

            since = time.time()
            with torch.profiler.record_function("phase:encode_wait"):
                made = queue.popleft().result()
            ready = made.pop("_ready", None) if made is not None else None
            if ready is not None:  # feed_stream: the copies land before the pass reads them
                main = torch.cuda.current_stream(CARD)
                main.wait_event(ready)
                for tensor in rl1_encode.tensors(made):
                    if tensor.is_cuda:
                        tensor.record_stream(main)
            stats["waits"] += time.time() - since
            fill(at["card"] - len(queue))
            return made

        def close():
            for future in list(queue) + list(hosted.values()) + list(carried.values()):
                future.cancel()
            thread.shutdown()
            if use and cmd.get("save"):  # the update's last step: nothing left to reuse
                self.window_cache.clear()

        return next_window, close, stats

    def served_routing(self, spec, rows, made):
        """R3: the play server's routing for the window (``rl1_encode.served_table``): ``(int16 [T, layers, k] on the
        card or None, audit)``."""
        import torch

        table, audit, layers = rl1_encode.served_table(spec, rows, made, made["inputs"]["input_ids"][0].cpu().numpy(),
                                                       SERVED)
        return self.served_account(None if table is None else torch.from_numpy(table).to(CARD), audit, layers)

    def served_account(self, table, audit, layers):
        """The route columns for the store's layers and the run's served audit, for a table resolved here or in an
        encoder process."""
        if layers and ROUTE["layers"] != layers:
            route_columns(self.model, layers)
        if table is None:
            SERVED_AUDIT["no_record"] = SERVED_AUDIT.get("no_record", 0) + 1
            return None, audit
        for key, value in audit.items():
            SERVED_AUDIT[key] = SERVED_AUDIT.get(key, 0) + value
        SERVED_AUDIT["windows"] = SERVED_AUDIT.get("windows", 0) + 1
        return table, audit

    def window_why(self, spec):
        """Why a window was not made: the end request's counts (trainer vs served prompt, reply, total)."""
        try:
            from rl_rollouts import compact

            rows = [json.loads(line) for line in open(spec["rollout"])]
            rows = [compact(r) if r.get("top") is not None else r for r in rows]
            caps = [c for c in rl_job.rollouts().captures(rows)
                    if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]
            end = spec.get("end")
            end = len(caps) - 1 if end is None else int(end)
            inputs, checks = fq.encode_request(self.processor, caps[end])
            return {"why": {"end": end, "captures": len(caps), "total": int(inputs["input_ids"].shape[1]),
                            **{k: checks[k] for k in ("prompt_tokens", "served_prompt_tokens", "reply_tokens",
                                                      "served_reply_tokens", "images", "image_tokens")}}}
        except Exception as error:  # diagnosis only
            return {"why": {"error": repr(error)[:300]}}

    def prefix_key(self, inputs, cut, host=None):
        """What a window's no-gradient prefix depends on besides its token ids: the pixels of the images that start
        before ``cut`` (count, patch rows, two checksums). U2: ``host`` (the window's host tensors): read there, no
        card sync (every key of a run is made the same way: RL1_HOST_INDEX is one setting per run)."""
        import torch

        if inputs.get("pixel_values") is None:
            return (cut, 0)
        if host is not None and all(k in host for k in ("input_ids", "image_grid_thw", "pixel_values")):
            inputs = host
        ids = inputs["input_ids"][0, :cut]
        token = self.processor.image_token_id
        runs = int(((ids == token) & torch.nn.functional.pad(ids[:-1] != token, (1, 0), value=True)).sum())
        rows = int(inputs["image_grid_thw"][:runs].prod(dim=-1).sum()) if runs else 0
        pixels = inputs["pixel_values"][:rows]
        return (cut, runs, rows, round(float(torch.sum(pixels, dtype=torch.float64)), 3),
                round(float(torch.sum(pixels[::7].abs(), dtype=torch.float64)), 3))

    def prefix_pass(self, ids, cut, piece, start_at=0, cache=None, snap_at=None):
        """The tokens before ``cut`` through the model without gradients, in chunks of the pinned ``prefix_chunk``,
        into a cache. No fallback to another chunk: the chunking changes the log-probs (rl1-gold: 0.054 nats/token
        between 16K and 32K at the same weights), so every pass of a run takes one. ``start_at`` / ``cache``: continue
        a cache that holds [0, start_at) (a chunk boundary: the same chunks as a pass from 0); ``snap_at`` (a chunk
        boundary): also return a copy of the cache as it stood at that position. Returns ``(cache, chunk, copy)``."""
        import copy

        import torch

        chunk, snapshot = int(getattr(self, "prefix_chunk", 16384)), None
        if start_at % chunk or (snap_at is not None and snap_at % chunk):
            raise ValueError(f"prefix pass: {start_at} / {snap_at} not on the {chunk}-token chunk grid")
        starts = range(start_at, cut, chunk)
        draft = getattr(self, "draft", None) is not None and getattr(self, "draft_live", False)
        self.model.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for start in starts:
                    if snap_at == start and cache is not None:
                        snapshot = copy.deepcopy(cache)
                    stop = min((start // chunk + 1) * chunk, cut)
                    ROUTE["call"] = served_call(start, stop)
                    causal_reset()
                    if draft:
                        self.draft_box["on"] = True
                    output = self.model(input_ids=ids[:, start:stop], attention_mask=torch.ones_like(ids[:, :stop]),
                                        past_key_values=cache, use_cache=True, logits_to_keep=1, **piece(start, stop))
                    cache = output.past_key_values
                    del output
                    if draft:  # stage D: the draft's rows of this chunk on the cache's draft state (snapshots copy it)
                        self.draft_box["on"] = False
                        hc = self.draft_box.pop("hc", None)
                        state = getattr(cache, "_rl1_draft", None) if start > 0 else None
                        cache._rl1_draft = None if hc is None else self.draft.prefix_chunk(
                            state, ids, start, stop, hc, draft_positions(piece, ids.device),
                            self.model.get_input_embeddings())
                if snap_at == cut and snapshot is None and cache is not None:
                    snapshot = copy.deepcopy(cache)
        finally:
            ROUTE["call"] = None
            if draft:
                self.draft_box["on"] = False
                self.draft_box.pop("hc", None)
            self.model.train()
        return cache, chunk, snapshot

    def pin_chunk(self, cmd):
        """One prefix chunking for every pass (GSPO's pi_old, the update, probes, scores). A call
        that asks for another chunk is a configuration error, not a fallback."""
        have = int(getattr(self, "path_chunk", 16384))
        want = cmd.get("prefix_chunk")
        if want is not None and int(want) != have:
            raise ValueError(f"prefix chunk pinned at {have}; this call asks for {int(want)}")
        self.prefix_chunk = have

    def prefix_cache(self, inputs, ids, cut, piece, shared, host=None):
        """The no-gradient prefix cache before ``cut``: ``(cache, reused, chunk, prefix positions)``; ``shared``
        keeps one prefix across the windows of a step (reused when ids and images before the cut match)."""
        import copy

        import torch

        key = self.prefix_key(inputs, cut, host) if shared is not None and cut > 0 else None
        self.prefix_start = 0
        if key is not None and shared.get("key") == key and bool(torch.equal(shared["ids"], ids[:, :cut])):
            self.prefix_start = cut
            return shared["cache"], True, None, shared["positions"]
        snaps = shared.get("snaps") if shared is not None else None
        if shared is not None:
            shared.clear()  # the previous prefix's cache goes before this one is built (its snapshots stay)
        cache = chunk = positions = None
        start_at, base, snap_at = 0, None, None
        route = ROUTE["window"] if ROUTE["router"] == "served" else None
        if getattr(self, "snapshots", False) and shared is not None and cut > 0:
            # trainer util: a later window continues the longest snapshot an
            # earlier window of the step left at a chunk boundary of the same context (ids, images and, under R3,
            # served routes before it equal) instead of recomputing that prefix: the same chunks from the same state,
            # so the same numbers. Snapshots are kept (a copy continues), so every try of a start reuses the start's.
            snaps = [] if snaps is None else snaps
            best = None
            for snap in snaps:
                at = snap["at"]
                if at > cut or (best is not None and at <= best["at"]):
                    continue
                if not bool(torch.equal(snap["ids"], ids[:, :at])) or self.prefix_key(inputs, at, host) != snap["pkey"]:
                    continue
                if (snap["route"] is None) != (route is None) or (
                        route is not None and not bool(torch.equal(snap["route"], route[:at].to(snap["route"].device)))):
                    self.snap_route_miss = getattr(self, "snap_route_miss", 0) + 1
                    continue
                best = snap
            if best is not None:
                start_at, base = best["at"], copy.deepcopy(best["cache"])
            grid = int(getattr(self, "prefix_chunk", 16384))
            snap_at = (cut // grid) * grid
            if snap_at <= 0 or snap_at <= start_at:
                snap_at = None
        if snaps is not None:
            shared["snaps"] = snaps
        if cut > 0:
            cache, chunk, copy_at = self.prefix_pass(ids, cut, piece, start_at=start_at, cache=base, snap_at=snap_at)
            positions = cache.position_ids
            self.prefix_start = start_at
            if copy_at is not None:
                snaps.append({"at": snap_at, "ids": ids[:, :snap_at], "pkey": self.prefix_key(inputs, snap_at, host),
                              "route": None if route is None else route[:snap_at].clone(), "cache": copy_at})
                while len(snaps) > int(getattr(self, "snap_keep", 3)):  # ~24 KB of keys per token (12 attention
                    snaps.pop(0)  # layers) + fixed GDN state: a 96K snapshot is ~2.4 GB
        if key is not None:
            shared.update(key=key, ids=ids[:, :cut], cache=cache, positions=positions)
        return cache, False, chunk, positions

    def window_pass(self, made, *, scale=None, vocab=None, shared=None):
        """The cached-prefix pass of one window; with ``scale`` the weighted loss is backpropagated. Returns per
        sampled token the actor and raw log-probabilities, validity and timings. ``shared`` (a dict kept across the
        windows of one optimizer step): a window whose prefix (ids and images before its cut) equals the previous
        one's reads that cache instead of computing it again (the cache is read-only in the window pass)."""
        import torch

        entered = time.time()  # MFU-1 host gaps: time outside the timed phases (pre: positions, checks, the sync)
        inputs, tail, arrays = made["inputs"], made["tail"], made["arrays"]
        PROGRESS["tail"] += int(tail)
        PROGRESS["windows"] += 1
        ids = inputs["input_ids"]
        total = ids.shape[1]
        cut = total - tail
        positions = arrays["positions"]
        keep = positions - cut - 1
        ROUTE["window"] = made.get("served")
        if bool((keep < 0).any()):
            raise RuntimeError("a trained token lies outside the gradient window")
        tokens = ids[0, positions]
        marks = {}  # MFU-1: the pre's vision tower and position-rule seconds
        host = self.host_of(made)
        piece = fq.positions_and_images(self.model, inputs, features=made.pop("vision_features", None), marks=marks,
                                        positions=made.get("mrope") if host else None, host=host)
        t_pre = stamp()  # U2 (27 Oct): CUDA events, no sync per part (4 drains per window before)
        began = time.time()
        pre = began - entered
        cache, reused, chunk, prefix_positions = self.prefix_cache(inputs, ids, cut, piece, shared, host)
        if cache is not None:
            # the model binds the full positions to the cache on every pass (prefix + this tail); a reused prefix
            # must start from its own positions, not the last window's (the indexer's block keys read them)
            cache.position_ids = prefix_positions
        from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

        modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
        t_prefix = stamp()
        began = time.time()
        
        grad = scale is not None
        if not grad:
            self.model.eval()
        elif not self.model.training:  # HF checkpointing runs only in train mode (PTH's resume found +17 GiB, 20 Oct)
            raise RuntimeError("grad pass with the model in eval mode")
        chunked = rl1_memory.HEAD["chunked"]
        SEL.update(window=grad and SEL["on"], store={})
        causal_reset()
        plan = rl1_memory.ckpt_begin(self.model, tail) if grad else None  # adaptive_ckpt: n layers uncheckpointed
        try:
            with (torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16),
                  rl1_memory.tail_scope(self.model, tail) if grad else contextlib.nullcontext()):
                ROUTE["call"] = served_call(cut, total)
                rl1_memory.HEAD["capture"] = chunked
                view = fq.read_only(cache) if cache is not None else None
                draft = grad and getattr(self, "draft", None) is not None and getattr(self, "draft_live", False)
                if draft:
                    self.draft_box["on"] = True
                try:
                    logits = self.model(input_ids=ids[:, cut:], attention_mask=torch.ones_like(ids),
                                        past_key_values=view, use_cache=False, logits_to_keep=keep,
                                        **piece(cut, total)).logits[0]
                finally:
                    rl1_memory.HEAD["capture"] = False
                    if draft:
                        self.draft_box["on"] = False
                if chunked:  # logits holds the kept rows' hidden states; no [N, V] table (stage M)
                    head = rl1_memory.chunked_head(self.head._rl1_project, logits, tokens, arrays["support"],
                                                   made["temperature"], hit=bool(made.get("argmax")), vocab=vocab,
                                                   support_host=made.get("support_host"))
                    actor, valid, raw, hit = head["actor"], head["valid"], head["raw"], head["hit"]
                    entropy, dist = head["entropy"], head["dist"]
                else:
                    actor, valid = rl_job.actor_logprobs(logits, tokens, arrays["support"], made["temperature"])
                    raw = rl_job.raw_logprobs(logits.detach(), tokens)
                    hit = (logits.detach().argmax(-1) == tokens) if made.get("argmax") else None  # next-board greedy hits
                    entropy = token_entropy(logits.detach(), valid, made["temperature"])
                    dist = None if vocab is None else vocab_top(logits.detach(), at_ids=vocab)
                loss = None
                tis = None
                signed = None
                if grad:
                    loss, tis, signed = self.weighted_loss(arrays, actor, valid, scale)
                    # stage D: the MTP draft's loss is its own graph (detached inputs), run after this backward
        except BaseException as error:
            rl1_memory.ckpt_end(plan, error)  # every layer checkpointed again; an OOM raises the coefficient
            raise
        finally:
            if not grad:
                self.model.train()
        t_forward = t_backward = stamp()
        if grad:
            try:
                loss.backward()
                t_backward = stamp()
            except BaseException as error:
                rl1_memory.ckpt_end(plan, error)
                raise
        ckpt = rl1_memory.ckpt_end(plan)  # n, predicted vs measured; the coefficients adapted before the next window
        draft_out = None
        if draft:  # stage D: after the trunk's graph is freed
            draft_out = self.draft_window(made, ids, cut, cache, self.draft_box.pop("hc", None), logits, chunked, piece)
        ROUTE["call"] = ROUTE["window"] = None  # after the backward: checkpoint recompute routes as the forward did
        SEL.update(window=False, store={})
        ended = time.time()
        causal_reset()
        actor_host = actor.detach().float().cpu()  # the window's one drain; the part spans read after it
        prefix, forward, backward = span(t_pre, t_prefix), span(t_prefix, t_forward), span(t_forward, t_backward)
        out = {"actor": actor_host, "raw": raw.float().cpu(), "valid": valid.cpu(),
               "dist": dist,  # vocab_top: already on the host
               "hit": None if hit is None else hit.cpu(),
               "entropy": entropy,
               "signed": signed() if callable(signed) else signed,
               "tis": None if tis is None else {"mean": round(float(tis[valid].mean()), 4),
                                                "capped": round(float((tis[valid] >= self.args.tis_cap).float().mean()), 5)},
               "loss": None if loss is None else float(loss.detach()), "prefix_seconds": round(prefix, 2),
               "prefix_reused": reused, "prefix_chunk": chunk, "cut": cut, "prefix_start": getattr(self, "prefix_start", 0),
               "prefix_carried": False, "carry_moved": 0,  # MFU-1 C prefix_carry (removed): the keys stay
               "forward_seconds": round(forward, 2), "backward_seconds": round(backward, 2), "tokens": total,
               "tail": tail, "targets": int(positions.numel()), **({"draft": draft_out} if draft_out is not None else {}),
               **({"ckpt": ckpt} if ckpt is not None else {})}
        del logits, cache
        out["host"] = {"pre": round(pre, 3), "post": round(time.time() - ended, 3), **marks}
        return out

    def cp_window_pass(self, made, *, scale):
        """RL1_DIST: one window sequence-parallel over every rank (rl1_cp), every rank with the same ``made``. The
        context before the cut is a sharded no-grad prefill (cuts on the 64-token grid), replicated as an
        rl1_cp.Prefix; the tail is sharded likewise. Each rank reads its shard's trained rows; the actor
        log-probabilities are gathered (gradient back to each rank's own rows) so every rank computes the window's
        whole loss as ``window_pass`` does. Returns ``window_pass``'s fields."""
        import torch

        import rl1_cp

        d = self.dist
        entered = time.time()
        inputs, tail, arrays = made["inputs"], made["tail"], made["arrays"]
        PROGRESS["tail"] += int(tail)
        PROGRESS["windows"] += 1
        ids = inputs["input_ids"]
        total = ids.shape[1]
        cut = total - tail
        positions = arrays["positions"]
        if bool((positions - cut - 1 < 0).any()):
            raise RuntimeError("a trained token lies outside the gradient window")
        host = self.host_of(made)
        piece = fq.positions_and_images(self.model, inputs, features=made.pop("vision_features", None),
                                        positions=made.get("mrope") if host else None, host=host)
        full = piece(0, total).get("position_ids")  # [3, 1, total]; None: the model's own (text) positions
        ROUTE["window"] = made.get("served")
        SEL.update(window=False, store={})  # stage S (picks kept for the recompute) is a single-GPU path
        causal_reset()
        began = time.time()
        pre = pc = None
        draft = getattr(self, "draft", None) is not None and getattr(self, "draft_live", False)
        hc_pre = hc_tail = None
        if cut:
            pc = rl1_cp.Ctx(rl1_cp.shard_lengths(cut, d.world), ids=ids[:, :cut],
                            position_ids=None if full is None else full[..., :cut])
            a, b = pc.offset, pc.offset + pc.T
            ROUTE["call"] = served_call(a, b)
            self.model.eval()
            if draft:
                self.draft_box["on"] = True
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pre = rl1_cp.prefill_call(self.model, pc, input_ids=ids[:, a:b], **piece(a, b))
            finally:
                self.model.train()
                if draft:
                    self.draft_box["on"] = False
                    hc_pre = self.draft_box.pop("hc", None)
        prefix = time.time() - began
        cp = rl1_cp.Ctx(rl1_cp.shard_lengths(tail, d.world), ids=ids[:, cut:], prefix=pre,
                        position_ids=None if full is None else full[..., :total])
        g0 = cut + cp.offset
        g1 = g0 + cp.T
        mine = (positions - 1 >= g0) & (positions - 1 < g1)
        keep = positions[mine] - 1 - g0
        tokens = ids[0, positions[mine]]
        support = arrays["support"][mine]
        began = time.time()
        chunked = rl1_memory.HEAD["chunked"]
        with rl1_cp.active(cp):  # the backward's checkpoint recompute runs the CP layers too
            with torch.autocast("cuda", dtype=torch.bfloat16), rl1_memory.tail_scope(self.model, tail):
                ROUTE["call"] = served_call(g0, g1)
                rl1_memory.HEAD["capture"] = chunked
                if draft:
                    self.draft_box["on"] = True
                try:
                    logits = rl1_cp.call(self.model, cp, input_ids=ids[:, g0:g1], logits_to_keep=keep,
                                         **piece(g0, g1)).logits[0]
                finally:
                    rl1_memory.HEAD["capture"] = False
                    if draft:
                        self.draft_box["on"] = False
                        hc_tail = self.draft_box.pop("hc", None)
                if chunked:
                    head = rl1_memory.chunked_head(self.head._rl1_project, logits, tokens, support, made["temperature"],
                                                   hit=bool(made.get("argmax")))
                    actor, valid, raw, hit, entropy = head["actor"], head["valid"], head["raw"], head["hit"], head["entropy"]
                else:
                    actor, valid = rl_job.actor_logprobs(logits, tokens, support, made["temperature"])
                    raw = rl_job.raw_logprobs(logits.detach(), tokens)
                    hit = (logits.detach().argmax(-1) == tokens) if made.get("argmax") else None
                    entropy = token_entropy(logits.detach(), valid, made["temperature"])
                # a rank whose shard has no trained row still backpropagates through its shard (the CP layers'
                # backward collectives): its (empty) actor stays on the graph
                actor = actor + logits.sum() * 0.0
                where = torch.nonzero(mine).flatten()
                order = torch.argsort(rl1_cp.gather_cat(where, cp))
                actor = rl1_cp.gather_cat(actor, cp)[order]
                valid = rl1_cp.gather_cat(valid, cp)[order]
                raw = rl1_cp.gather_cat(raw.float(), cp)[order]
                hit = None if hit is None else rl1_cp.gather_cat(hit, cp)[order]
                ent_sum, ent_count = d.reduce_sums(list(entropy), actor.device)
                entropy = (ent_sum, int(round(ent_count)))
                loss, tis, signed = self.weighted_loss(arrays, actor, valid, scale)
            forward = time.time() - began
            began = time.time()
            loss.backward()
            backward = time.time() - began
        draft_out = None
        if draft:  # stage D: every rank (the draft's attention collectives), after the trunk's graph is freed
            draft_out = self.draft_cp(made, ids, cut, pc, cp, hc_pre, hc_tail, full, logits, chunked,
                                      positions[mine])
        ROUTE["call"] = ROUTE["window"] = None
        causal_reset()
        return {"actor": actor.detach().float().cpu(), "raw": raw.float().cpu(), "valid": valid.cpu(), "dist": None,
                "hit": None if hit is None else hit.cpu(), "entropy": entropy, "signed": signed() if callable(signed) else signed,
                "tis": None if tis is None else {"mean": round(float(tis[valid].mean()), 4),
                                                 "capped": round(float((tis[valid] >= self.args.tis_cap).float().mean()), 5)},
                "loss": float(loss.detach()), "prefix_seconds": round(prefix, 2), "prefix_reused": False,
                "prefix_chunk": None, "cut": cut, "prefix_start": 0, "prefix_carried": False, "carry_moved": 0,
                "forward_seconds": round(forward, 2), "backward_seconds": round(backward, 2), "tokens": total,
                "tail": tail, "targets": int(positions.numel()), "cp": {"world": d.world, "shard": [g0, g1]},
                "host": {"pre": round(began - entered, 3)}, **({"draft": draft_out} if draft_out is not None else {})}

    def dist_loop(self, specs, n_cp, next_window, record, scale_of, shared, result, host_next):
        """RL1_DIST: this rank's windows of the call: the ``n_cp`` sequence-parallel ones first (every rank, the same
        order; rank 0 records them), then its data-parallel share. A failed window is agreed on before the next
        collective (every rank raises the same error)."""
        d = self.dist
        failure = None
        for i, spec in enumerate(specs):
            made = None
            if failure is None:
                try:
                    asked = time.time()
                    made = next_window()
                    host_next[0] += time.time() - asked
                except Exception as error:  # noqa: BLE001 - agreed on below
                    failure = error
            if i < n_cp:
                d.agree(failure, where=f"CP window {spec.get('name')}")
                d.ready()  # RL1_NCCL_WARM: the warm-up collective first, on every rank
                if made is None:  # the same spec encodes the same everywhere: no window fits on every rank
                    if d.rank == 0:
                        result["errors"].append({"name": spec.get("name"), "error": "no window fits",
                                                 **self.window_why(spec)})
                    continue
                out = self.cp_window_pass(made, scale=scale_of(spec, made))
                if d.rank == 0:
                    record(spec, made, out)
                del made, out
                continue
            if failure is not None:
                break
            try:
                if made is None:
                    result["errors"].append({"name": spec.get("name"), "error": "no window fits", **self.window_why(spec)})
                    continue
                out = self.window_pass(made, scale=scale_of(spec, made), shared=shared)
                record(spec, made, out)
                del made, out
            except Exception as error:  # noqa: BLE001 - agreed on below
                failure = error
                break
        d.agree(failure, where="windows")

    def weighted_loss(self, arrays, actor, valid, scale):
        """The policy-gradient loss of one window's sampled tokens: ``(loss, tis, signed)``."""
        import torch

        weights = arrays["weights"] if "weights" in arrays else torch.ones_like(actor)
        weights = weights.float()
        tis = signed = None
        if self.args.tis_cap and "reference" in arrays:  # fit's synthetic windows carry none (rl1-loop-v1)
            # trainer over sampler on the tempered, truncated policy (1 where the engine value is missing):
            # positive weights x min(ratio, cap) (truncated IS, Yao et al. 2025); negative weights x
            # clip(ratio, 0, 1) (TOPR, Le Roux et al. 2025: the push-down term stays bounded)
            ratio = torch.exp(actor.detach() - arrays["reference"]).nan_to_num(1.0, 1.0, 1.0)
            tis = torch.where(weights < 0, ratio.clamp(0.0, 1.0), ratio.clamp(max=self.args.tis_cap))
            down = (weights < 0) & valid
            far = ((ratio < 0.5) | (ratio > 2.0)) & valid  # tokens outside [0.5, 2]
            stats = [down.sum(), ((ratio > 1) & down).sum(), far.sum(), valid.sum()]
            weights = weights * tis
        rc, clip = getattr(self, "ratio_clip", None), []
        if rc and "old" in arrays:  # the trust region against the update's first weights (θ0)
            # r = now / θ0 on the sampled tokens; "seq" (GSPO): the reply's length-normalised log-ratio for each of its
            # tokens; "tok" (PPO). A token whose ratio left the region in its advantage's direction gets no gradient
            # (PPO's clipped surrogate); the REINFORCE x TIS form is kept inside it
            ok = valid & arrays["old_valid"]
            logr = (actor.detach() - arrays["old"]).masked_fill(~ok, 0.0)
            if rc["mode"] == "seq":
                reply = arrays["reply"] if "reply" in arrays else torch.zeros_like(arrays["positions"])
                sums = torch.zeros(logr.shape[0], device=logr.device).index_add_(0, reply, logr)
                count = torch.zeros(logr.shape[0], device=logr.device).index_add_(0, reply, ok.float())
                logr = (sums / count.clamp(min=1.0))[reply]
            out_of = ok & (((weights > 0) & (logr > math.log1p(rc["high"])))
                           | ((weights < 0) & (logr < math.log1p(-rc["low"]))))
            weights = weights.masked_fill(out_of, 0.0)
            clip = [out_of.sum(), ok.sum(), logr.abs().masked_fill(~ok, 0.0).max() if ok.numel() else ok.sum()]
        loss = -(weights[valid] * actor[valid]).sum() * scale
        w = weights[valid].detach()
        head = len(stats) if tis is not None else 0
        stats = (stats if tis is not None else []) + clip + [(w > 0).sum(), w.clamp(min=0).sum(),
                                                             (-w).clamp(min=0).sum(), (w < 0).sum(), valid.sum()]
        stacked = torch.stack([x.to(torch.float64) for x in stats])
        capped = tis is not None

        def settle():  # U2 (27 Oct): read after the backward, one host read (was ~8 syncs before the backward)
            got, out = stacked.tolist(), None
            if capped:
                n_down, n_clip, n_far, n_valid = (int(x) for x in got[:4])
                if n_down:
                    out = {"down_tokens": n_down, "down_clipped": round(n_clip / n_down, 5)}
                out = {**(out or {}), "tis_out": {"tokens": n_far, "of": n_valid}}
            n_up, up, dn, n_neg, n_val = got[-5:]
            out = {**(out or {}), "n_up": int(n_up), "n_down": int(n_neg), "n_valid": int(n_val)}
            if clip:
                n_clip, n_of, top = got[head:head + 3]
                out["ratio_clip"] = {"clipped": int(n_clip), "of": int(n_of), "max_abs_logr": round(top, 6)}
            if n_up > 0 or "tis_out" in out:
                out.update(up=round(up, 4), down=round(dn, 4))
            return out
        return loss, tis, settle

    def window_stats_add(self, mm, out):
        """The update's step-row safety fields, summed over its windows (single card; RL1_DIST ranks keep their own)."""
        w = getattr(self, "wstats", None) or {}
        n = int(mm.get("tokens") or 0)
        if n:
            w["mm_tokens"] = w.get("mm_tokens", 0) + n
            w["k1"] = w.get("k1", 0.0) + float(mm["mean"]) * n
            w["k3"] = w.get("k3", 0.0) + float(mm.get("k3") or 0.0) * n
        sg, tis = out.get("signed") or {}, out.get("tis") or {}
        v = int(sg.get("n_valid") or 0)
        if v:
            for k in ("n_up", "n_down", "n_valid"):
                w[k] = w.get(k, 0) + int(sg.get(k) or 0)
            w["capped"] = w.get("capped", 0.0) + float(tis.get("capped") or 0.0) * v
        rc = sg.get("ratio_clip")
        if rc:
            w["rc_clipped"] = w.get("rc_clipped", 0) + rc["clipped"]
            w["rc_of"] = w.get("rc_of", 0) + rc["of"]
            w["rc_max"] = max(w.get("rc_max", 0.0), rc["max_abs_logr"])
        self.wstats = w

    def window_stats_take(self):
        """{"k3": {approx_kl (k3), approx_kl_tok (k1), clip_frac (TIS-capped share)}, adv_*_share} since the last step."""
        w, self.wstats = getattr(self, "wstats", None) or {}, {}
        if getattr(self, "dist", None) is not None:
            return {}
        out = {}
        if w.get("mm_tokens"):
            n = w["mm_tokens"]
            out["k3"] = {"approx_kl": round(w["k3"] / n, 6), "approx_kl_tok": round(w["k1"] / n, 6), "tokens": n}
        if w.get("n_valid"):
            v = w["n_valid"]
            out.setdefault("k3", {})["clip_frac"] = round(w["capped"] / v, 6)
            out["adv_nonzero_share"] = round((w["n_up"] + w["n_down"]) / v, 4)
            out["adv_up_share"] = round(w["n_up"] / v, 4)
        if w.get("rc_of"):  # the θ0 trust region's share of zeroed tokens over the step's clipped windows
            out["ratio_clip"] = {"frac": round(w["rc_clipped"] / w["rc_of"], 6), "tokens": w["rc_of"],
                                 "max_abs_logr": w["rc_max"]}
        return out

    def mismatch(self, made, out):
        """Trainer against server on the window's sampled tokens (raw log-probabilities, whole vocabulary,
        temperature 1): engine minus trainer."""
        import torch

        engine = torch.tensor(made["engine_raw"], dtype=torch.float32)
        diff = engine - out["raw"]
        keep = torch.isfinite(diff)
        diff = diff[keep]
        if not diff.numel():
            return {"tokens": 0}
        return {"tokens": int(diff.numel()), "mean_abs": round(float(diff.abs().mean()), 5),
                "k3": round(float((torch.exp(-diff) - 1.0 + diff).mean()), 6),  # KL(engine || trainer), k3 on samples
                "mean": round(float(diff.mean()), 5), "beyond_half": round(float((diff.abs() > 0.5).float().mean()), 5),
                "beyond_ln2": round(float((diff.abs() > math.log(2)).float().mean()), 5),
                "engine_nll": round(float(-engine[keep].mean()), 4), "trainer_nll": round(float(-out["raw"][keep].mean()), 4)}

    # ---- operations ----

    def op_reset(self, cmd):
        import torch

        with torch.no_grad():
            for name, param in self.params:
                if "lora_A" in name:
                    param.copy_(self.initial[name].to(param.device))
                elif "lora_B" in name:
                    param.zero_()
        self.optimizer = self.make_optimizer(float(cmd.get("lr", self.args.lr)))
        self.optimizer.zero_grad(set_to_none=True)
        self.references, self.updates = {}, 0
        return {"reset": True, "adapter_norm": self.adapter_norm()}

    def score_windows(self, specs, label=None, dump=False):
        rows = {}
        for spec in specs:
            made = self.attempt_window(spec)
            if made is None:
                rows[spec.get("name")] = {"error": "no window fits"}
                continue
            name = made["name"]
            refs = self.references.setdefault(name, {})
            keys = [key for key in ("base", "previous") if key in refs]
            out = self.window_pass(made, vocab=[refs[key]["ids"] for key in keys])
            row = {"tokens": int(out["valid"].sum()), "actor_mean": round(float(out["actor"][out["valid"]].mean()), 5),
                   "mismatch": self.mismatch(made, out), "seconds": out["prefix_seconds"] + out["forward_seconds"],
                   # the served prompt of every scored reply, token for token (drain_half check, 19 Oct)
                   "prompt_same": [r.get("prompt_same") for r in made.get("replies") or []],
                   "requests": [r.get("request") for r in made.get("replies") or []], "context": made.get("tokens")}
            if getattr(self, "probe_pinned", False):  # diagnostic (23 Oct): KL to base with base's own expert picks
                if label == "base":
                    refs["base_pinned"] = self.pinned_pass(made, "capture", None)
                elif "base_pinned" in refs:
                    ref = refs["base_pinned"]
                    now = self.pinned_pass(made, "replay", ref)
                    keep = ref["valid"] & now["valid"]
                    row["kl_base_pinned"] = round(binned_kl(ref["logp"], now["dist"]["at"][0]), 6)
                    row["logratio_base_pinned"] = round(float((ref["actor"] - now["actor"])[keep].mean()), 6)
            for index, key in enumerate(keys):
                keep = refs[key]["valid"] & out["valid"]
                # kl_*: KL(reference || now) at temperature 1 over the full vocabulary, binned as the reference's
                # top-20 tokens plus the rest, mean over every trained position (non-negative, a lower bound of the
                # full KL). logratio_*: the sampled-token log-ratio of the actor policy (k1; the probe's pick rule)
                row[f"kl_{key}"] = round(binned_kl(refs[key]["logp"], out["dist"]["at"][index]), 6)
                row[f"logratio_{key}"] = round(float((refs[key]["actor"] - out["actor"])[keep].mean()), 6)
            mine = {"actor": out["actor"], "valid": out["valid"], "ids": out["dist"]["ids"], "logp": out["dist"]["logp"]}
            if label == "base":
                refs["base"] = mine
            refs["previous"] = mine
            if dump:  # score_check: each reply's sampled tokens, raw, whole vocabulary, temperature 1
                raw, at, row["replies"] = out["raw"].tolist(), 0, []
                for reply in made.get("replies") or []:
                    if reply.get("excluded"):
                        continue
                    size = reply["stop"] - reply["start"]
                    row["replies"].append({"request": reply["request"], "start": reply["start"],
                                           "raw": [round(x, 5) for x in raw[at:at + size]]})
                    at += size
            rows[name] = row
            del made
        return rows

    def pinned_pass(self, made, mode, ref):
        """One probe window with the MoE routers recording ("capture": the base pass's own top-k picks, kept with its
        top-20 distribution) or forced to a recorded pass ("replay": now's weights on base's picks). Prefix snapshots
        off for both (the recorded router calls must match one for one); ROUTE and snapshots restored after."""
        saved = {k: ROUTE[k] for k in ("router", "rec", "i")}
        snaps, self.snapshots = getattr(self, "snapshots", False), False
        try:
            ROUTE.update(router=mode, rec=[] if mode == "capture" else list(ref["route"]), i=0)
            out = self.window_pass(made, vocab=[] if mode == "capture" else [ref["ids"]])
            if mode == "replay" and ROUTE["i"] != len(ref["route"]):
                raise RuntimeError(f"pinned replay used {ROUTE['i']} of {len(ref['route'])} recorded router calls")
            if mode == "capture":
                return {"route": [r.cpu() if hasattr(r, "cpu") else r for r in ROUTE["rec"]], "ids": out["dist"]["ids"], "logp": out["dist"]["logp"],
                        "actor": out["actor"], "valid": out["valid"]}
            return out
        finally:
            ROUTE.update(saved)
            self.snapshots = snaps

    def op_route(self, cmd):
        """rl1-route: ``router`` off | served (R3) | capture (clears the record) | replay (from the first recorded
        call). Every op restarts the replay pointer."""
        route_patch()
        mode = cmd.get("router")
        if mode is not None:
            if mode not in ("off", "capture", "replay", "served"):
                raise ValueError(f"route router: {mode}")
            if mode == "capture":
                ROUTE["rec"] = []
            ROUTE["router"] = mode
        used = {"router": ROUTE["i"]}
        ROUTE["i"] = 0
        served = {"rows": int(ROUTE["used"]) if ROUTE["used"] is not None else 0,
                  "agree": int(ROUTE["agree"]) if ROUTE["agree"] is not None else 0,
                  "unserved_calls": ROUTE["unserved_calls"], "windows": dict(SERVED_AUDIT)}
        return {"router": ROUTE["router"], "recorded": len(ROUTE["rec"]), "replayed_before": used,
                "calls": ROUTE["calls"], "indexer": dict(INDEXER), "served": served}

    def op_score(self, cmd):
        return {"windows": self.score_windows(cmd["windows"], cmd.get("label"), dump=bool(cmd.get("dump")))}

    def adapter_norm(self):
        import torch

        squares = [p.detach().float().pow(2).sum() for _, p in self.policy_params()]  # one host read (was one per tensor)
        return round(math.sqrt(float(torch.stack(squares).sum())), 5) if squares else 0.0

    def expand_cmd(self, cmd):
        attempts = cmd["attempts"]
        if cmd.get("whole_segment"):  # RL C2 (4a): one window per drain segment, every model output in it trained
            attempts = [whole_segment(a) for a in attempts]
        return self.expand(attempts, procs=int(cmd.get("encode_procs") or 0))

    def expand(self, specs, procs=0):
        """A spec with ``whole`` becomes one spec per planned window end that holds weighted replies (every token of a
        long continuation trained, each window once). MFU-1 (23 Oct): a spec expanded before (same spec, same rollout
        size and mtime) is taken from a memo, and with ``procs`` encoder
        processes the rest are parsed in parallel there (the rollouts' JSON, ~0.3-0.5 s a try, ran on the main
        thread with the card idle at every op's start)."""
        memo = getattr(self, "expand_memo", None)
        if memo is None:
            memo = self.expand_memo = {}
        keys, todo = [], {}
        for i, spec in enumerate(specs):
            key = None
            if spec.get("whole"):
                stat = Path(spec["rollout"]).stat()
                key = json.dumps([spec, int(self.args.window), stat.st_size, stat.st_mtime_ns], sort_keys=True, default=str)
                if key not in memo:
                    todo.setdefault(key, spec)
            keys.append(key)
        if todo:
            if procs and len(todo) > 1:
                pool = self.encoder_pool(procs)
                futures = {key: pool.submit(rl1_encode.expand_one, spec, int(self.args.window)) for key, spec in todo.items()}
                done = {}
                for key, future in futures.items():
                    try:
                        done[key] = future.result()
                    except Exception as error:  # a broken process: this one parses it
                        self.encode_errors = getattr(self, "encode_errors", []) + [repr(error)[:300]]
                        done[key] = rl1_encode.expand_one(todo[key], int(self.args.window))
            else:
                done = {key: rl1_encode.expand_one(spec, int(self.args.window)) for key, spec in todo.items()}
            memo.update(done)
            while len(memo) > 1024:
                memo.pop(next(iter(memo)))
        out = []
        for spec, key in zip(specs, keys):
            out += [spec] if key is None else [dict(x) for x in memo[key]]
        return out

    def op_accumulate(self, cmd):
        """v7 streamed accumulation: one group's windows forward/backward at the current state, gradients summed into
        the parameters' .grad across calls (no divide, clip or step). ``first`` starts a new batch."""
        if cmd.get("first") or getattr(self, "acc", None) is None:
            self.acc_reset()
        return self.op_update({**cmd, "accumulate": True})

    def op_old_pass(self, cmd):
        """the windows of ``attempts`` forward-only at the current weights (the update's θ0, before
        its first step): each window's sampled-token actor log-probabilities kept by name for the clipped later
        steps (``ratio_clip``). No gradient, no accumulated sums. ``old_reset``: drop the previous update's."""
        if getattr(self, "dist", None) is not None:
            raise ValueError("old_pass: single-card only (RL1_DIST would split the windows over ranks)")
        return self.update_once({**cmd, "old_pass": True})

    def attach_old(self, made, tally):
        """The window's θ0 log-probabilities (old_pass / keep_old) into its arrays; a window without them trains
        unclipped and is counted."""
        got = (getattr(self, "old_lp", None) or {}).get(made.get("key") or made["name"])
        arrays = made["arrays"]
        if got is None or got[0].shape[0] != arrays["positions"].shape[0]:
            tally["missing"] += 1
            return
        device = arrays["positions"].device
        arrays["old"], arrays["old_valid"] = got[0].to(device), got[1].to(device)
        tally["attached"] += 1

    def op_step(self, cmd):
        """v7: the batch's step from the accumulated gradients: the same divisor, clip and AdamW step as one
        ``op_update`` over all the accumulated windows."""
        if not getattr(self, "acc", None) or not self.acc["calls"]:
            return {"ok": False, "error": "nothing accumulated"}
        try:
            return self.op_update({**cmd, "attempts": [], "from_acc": True})
        finally:
            self.acc = None

    def acc_reset(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.acc = {"tokens": 0, "losses": 0.0, "finite": True, "tries": {}, "entropy": [0.0, 0], "calls": 0,
                    "windows": 0, "exact": [0, 0, 0]}
        if getattr(self, "draft", None) is not None:
            self.draft_reset()

    def op_update(self, cmd):
        draft = getattr(self, "draft", None) is not None
        if draft and not cmd.get("accumulate") and not cmd.get("from_acc"):
            self.draft_reset()  # a one-call update; v7's accumulate calls sum into the batch's (acc_reset)
        self.draft_live = draft
        before = {k: float(self.draft_sums[k]) for k in DRAFT_KEYS} if draft else None
        try:
            result = self.update_once(cmd)
        finally:
            self.draft_live = False
        if draft and cmd.get("accumulate") and isinstance(result, dict):  # this call's own draft read (T1b-v2:
            d = {k: float(self.draft_sums[k]) - before[k] for k in DRAFT_KEYS}  # a held-out window, then acc_reset)
            if d["rows"] > 0:
                result["draft_call"] = {"rows": int(round(d["rows"])), "kl": round(d["kl"] / d["rows"], 6),
                                        "top1": round(d["top1"] / d["rows"], 4),
                                        "paccept": round(d["paccept"] / d["rows"], 4)}
        return result

    def update_once(self, cmd):
        import torch

        entered = time.time()  # U4 (27 Oct): ~3-4 s of each accumulate sat outside train_seconds; setup_seconds splits it
        self.pin_chunk(cmd)
        result = {"windows": [], "errors": [], "prefix_chunk": self.prefix_chunk}
        served_before = dict(SERVED_AUDIT)  # v17 a7: was taken in op_update, out of this scope (NameError at U20's end)
        old_pass, keep_old = bool(cmd.get("old_pass")), bool(cmd.get("keep_old"))
        if cmd.get("old_reset") or getattr(self, "old_lp", None) is None:
            self.old_lp = {}
        self.ratio_clip = None if old_pass else ratio_clip_spec(cmd.get("ratio_clip"))
        old_tally = {"attached": 0, "missing": 0}
        lr = float(cmd["lr"])
        self.set_lr(lr)
        if cmd.get("policy_frozen"):  # stage D test legs (mg.redesign.freeze_policy): only the draft's group moves
            if getattr(self, "draft", None) is None:
                raise ValueError("policy_frozen without a draft LoRA: the step would train nothing")
            for group in self.optimizer.param_groups:
                if not group.get("draft"):
                    group["lr"] = 0.0  # AdamW, weight decay 0: p - 0 * update is p, bitwise
        expand_began = time.time()
        specs = [a for a in self.expand_cmd(cmd) if a["w"] > 0 or (cmd.get("signed") and a["w"] < 0)]
        result["expand_seconds"] = round(time.time() - expand_began, 2)  # MFU-1: before the timed part, card idle
        self.ahead_start(cmd)
        shared = None
        if cmd.get("share_prefix"):  # the batch's order, grouped by start; each try's window from its start first
            groups = {}
            for a in specs:
                groups.setdefault(a.get("share"), len(groups))
            specs = sorted(specs, key=lambda a: (groups[a.get("share")], not a.get("first_window", True)))
            specs = specs[: int(cmd.get("max_windows", 8))]
            shared = {}
        else:
            specs = sorted(specs, key=lambda a: -abs(a["w"]))[: int(cmd.get("max_windows", 8))]
        per_try = {}
        if cmd.get("norm") == "cap":  # windows per try (``expand`` names them try#i)
            for a in specs:
                key = a.get("name", "").split("#")[0]
                per_try[key] = per_try.get(key, 0) + 1
        dist_plan = None
        if getattr(self, "dist", None) is not None and specs:  # RL1_DIST: this rank's share of the windows
            dist_plan = rl1_dist.plan(specs, self.dist.world, self.dist.rank,
                                      int(cmd.get("fit_tokens") or rl1_dist.FIT_TOKENS), int(self.args.window))
            specs = [specs[i] for i in dist_plan["mine"]]
            result["dist_plan"] = {"cp": len(dist_plan["cp"]), "dp": len(dist_plan["dp"][self.dist.rank]),
                                   "load": dist_plan["load"]}
        fq.reset_peaks()
        rl1_memory.CKPT["step_peak"] = 0.0  # adaptive_ckpt resets the peaks per window: the call's max kept here
        losses, finite, tokens = 0.0, True, 0
        # the next windows are encoded (CPU: template, images) while the GPU runs this one: one encoding thread (no
        # second thread contending for the GIL with the launching one), ``prefetch`` windows queued ahead (MFU 3);
        # stage W: encoder processes, the old pass's host windows (``window_feed``)
        next_window, close, feed = self.window_feed(specs, cmd)
        if self.ratio_clip:
            fed = next_window

            def next_window():
                made = fed()
                if made is not None:
                    self.attach_old(made, old_tally)
                return made
        with Sampler() as sampler:
            began = time.time()
            result["setup_seconds"] = round(began - entered, 2)

            def scale_of(spec, made):
                if old_pass:
                    return None  # window_pass: forward only
                if cmd.get("norm") == "cap":  # Dr. GRPO: each try's summed log-likelihood over a constant (the cap)
                    return 1.0 / float(spec.get("norm_cap") or cmd["norm_cap"])  # mg5b: a per-level constant when the spec has one
                return 1.0  # summed; the gradient is divided by the tokens below

            def record(spec, made, out):
                nonlocal tokens, losses, finite
                now = time.time()  # MFU-1: wall since the previous window's record (or the loop start)
                wall, last_record[0] = now - last_record[0], now
                host = {**(out.get("host") or {}), "next": round(host_next[0], 3), "record_prev": round(host_rec[0], 3)}
                host_next[0] = 0.0
                if old_pass or keep_old:  # θ0's log-probabilities of this window (host, ~5 B a token)
                    self.old_lp[made.get("key") or made["name"]] = (out["actor"], out["valid"].bool())
                if old_pass:
                    tokens += out["targets"]
                    finite &= bool(torch.isfinite(out["raw"]).all())
                    result["windows"].append({"name": made["name"], "targets": out["targets"],
                                              "mismatch": self.mismatch(made, out),
                                              "seconds": round(out["prefix_seconds"] + out["forward_seconds"], 1)})
                    return
                ent = out.get("entropy") or (0.0, 0)
                entropy_sum[0] += ent[0]
                entropy_sum[1] += ent[1]
                tokens += out["targets"]
                losses += out["loss"]
                if getattr(self, "draft", None) is not None:
                    self.draft_add(out.get("draft"))
                finite &= math.isfinite(out["loss"]) and bool(torch.isfinite(out["raw"]).all())
                mm = self.mismatch(made, out)
                self.window_stats_add(mm, out)
                result["windows"].append({"name": made["name"], "tokens": out["tokens"], "tail": out["tail"],
                                          "targets": out["targets"], "trained": int(out["valid"].sum()),
                                          "w": float(made["arrays"]["weights"][0]) if "weights" in made["arrays"] else 1.0,
                                          "mismatch": mm,
                                          "sampled_nll_trainer": round(float(-out["raw"].mean()), 4), "tis": out["tis"],
                                          "signed": out["signed"],
                                          "prefix_seconds": out["prefix_seconds"],
                                          "forward_seconds": out["forward_seconds"], "backward_seconds": out["backward_seconds"],
                                          "prefix_reused": out["prefix_reused"], "cut": out["cut"],
                                          "prefix_start": out.get("prefix_start", 0),
                                          "seconds": round(out["prefix_seconds"] + out["forward_seconds"]
                                                           + out["backward_seconds"], 1),
                                          "wall": round(wall, 2), "host": host, "exact": exact_counts(made),
                                          **({"ckpt": out["ckpt"]} if "ckpt" in out else {})})
                if cmd.get("ref") in ("keep", "collect"):  # test hook: the step's logprobs
                    ref_new[made["name"]] = (out["actor"].float(), out["valid"].bool())
                host_rec[0] = time.time() - now  # this record's own time falls in the next window's wall

            ref_new = {}
            last_record = [time.time()]
            host_next, host_rec = [0.0], [0.0]
            entropy_sum = [0.0, 0]

            if dist_plan is not None:
                self.dist_loop(specs, len(dist_plan["cp"]), next_window, record, scale_of, shared, result,
                               host_next)
            for spec in (specs if dist_plan is None else ()):  # one window in flight (a window's images are GBs)
                asked = time.time()
                made = next_window()
                host_next[0] += time.time() - asked
                if made is None:
                    result["errors"].append({"name": spec.get("name"), "error": "no window fits", **self.window_why(spec)})
                    continue
                out = self.window_pass(made, scale=scale_of(spec, made), shared=shared)
                record(spec, made, out)
                del made, out
            close()
            if shared is not None:
                shared.clear()  # the next step's weights differ: no cache outlives its step
                result["prefix_reused"] = sum(bool(w.get("prefix_reused")) for w in result["windows"])
            result["window_wait_seconds"] = round(feed["waits"], 1)
            if rl1_memory.CKPT["on"]:  # the coefficients after this call's windows (max-tracking), budget, raises
                result["adaptive_ckpt"] = rl1_memory.ckpt_state()
            gaps = sorted(((w["wall"] - w["seconds"], w) for w in result["windows"] if "wall" in w), key=lambda g: -g[0])
            if gaps:  # MFU-1: per-window wall outside its prefix / forward / backward (feed waits included)
                result["window_gap"] = {"sum": round(sum(g for g, _ in gaps), 1), "p50": round(gaps[len(gaps) // 2][0], 2),
                                        "max": round(gaps[0][0], 2),
                                        "top": [{"name": w["name"], "gap": round(g, 2), "tokens": w["tokens"],
                                                 "prefix_reused": w["prefix_reused"]} for g, w in gaps[:5]],
                                        # the gap's named parts (feed wait, pass pre/post, the previous record); rest: other host work
                                        "host": {k: round(sum((w.get("host") or {}).get(k) or 0.0 for _, w in gaps), 1)
                                                 for k in ("next", "pre", "post", "record_prev", "vision", "rope")}}
            if cmd.get("encode_procs") or cmd.get("window_cache"):
                result["encode"] = {"procs": int(cmd.get("encode_procs") or 0), "encoded": feed["encoded"],
                                    "cache_hits": feed["cache_hits"], "errors": self.encode_errors[:5],
                                    "cache": self.window_cache.summary(),
                                    **{k: feed[k] for k in ("held", "ahead_hits", "ahead_sent")}}
            if ROUTE["router"] == "served":  # this call's R3 coverage (windows, no_record, resolver counts)
                result["served"] = served_delta(served_before)
            if self.ratio_clip:
                result["ratio_clip"] = {**self.ratio_clip, **old_tally}
            if old_pass:
                result.update(ok=bool(finite), old_pass=True, old_windows=len(result["windows"]), scored_tokens=tokens,
                              old_kept=len(self.old_lp), train_seconds=round(time.time() - began, 1),
                              **sampler.summary())
                return result
            if cmd.get("accumulate"):  # v7: the sums carry to op_step; the gradients stay in .grad
                acc = self.acc
                acc["tokens"] += tokens
                acc["losses"] += losses
                acc["finite"] &= finite
                acc["calls"] += 1
                acc["windows"] += len(result["windows"])
                for w in result["windows"]:
                    acc.setdefault("exact", [0, 0, 0])
                    acc["exact"] = [a + b for a, b in zip(acc["exact"], w.get("exact") or (0, 0, 0))]
                acc["entropy"][0] += entropy_sum[0]
                acc["entropy"][1] += entropy_sum[1]
                for key, n in per_try.items():
                    acc["tries"][key] = acc["tries"].get(key, 0) + n
                if ref_new:  # ref "collect": the windows' logprobs for the step's ref keep
                    acc.setdefault("ref_actor", {}).update(ref_new)
                result.update(ok=bool(finite), accumulated=True, trained_tokens=tokens, acc_tokens=acc["tokens"],
                              acc_tries=len(acc["tries"]), acc_calls=acc["calls"],
                              entropy=round(entropy_sum[0] / entropy_sum[1], 5) if entropy_sum[1] else None,
                              train_seconds=round(time.time() - began, 1), **sampler.summary())
                first = rl1_firstcall.report()  # RL1_FIRST_CALLS: cumulative compile / autotune seconds
                if first is not None:
                    result["first_calls"] = first
                return result
            # D1 (26 Oct): 22.6 s of the 27 s step sat between here and the optimizer at ~0% util; per-part seconds
            # (the card synchronised at each mark) name it
            parts, mark_at = {}, [began]

            def mark(name):
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                now = time.time()
                parts[name] = round(now - mark_at[0], 3)
                mark_at[0] = now

            mark("windows")
            if cmd.get("from_acc"):  # v7 op_step: the accumulated batch stands in for this call's (empty) windows
                acc = self.acc
                tokens, losses, finite = acc["tokens"], acc["losses"], acc["finite"]
                per_try = dict(acc["tries"]) if cmd.get("norm") == "cap" else {}
                entropy_sum[0], entropy_sum[1] = acc["entropy"]
                ref_new.update(acc.get("ref_actor") or {})  # T1a 26 Oct: else the step compared 0 windows' logprobs
                result.update(accumulated_calls=acc["calls"], accumulated_windows=acc["windows"])
                result.update(exact_summary(acc.get("exact")))
            if getattr(self, "dist", None) is not None:  # RL1_DIST: every rank's gradients and sums, then one step each
                result["dist_collectives_before"] = self.dist.collectives  # 0: this step pays the NCCL setup
                tokens, losses, entropy_sum[0], entropy_sum[1], bad = self.dist.reduce_sums(
                    [tokens, losses, entropy_sum[0], entropy_sum[1], 0 if finite else 1], self.params[0][1].device)
                tokens, entropy_sum[1], finite = int(round(tokens)), int(round(entropy_sum[1])), bad == 0
                mark("reduce_sums")
                result["dist_grad_bytes"] = self.dist.reduce_grads([p for _, p in self.params])
                mark("reduce_grads")
                every = self.dist.all_objects([result["windows"], result["errors"]])
                mark("objects")
                result["windows"] = [w for part in every for w in part[0]]
                result["errors"] = [e for part in every for e in part[1]]
                result["dist_reduced"] = True
                if self.dist.warmed is not None:
                    result["nccl_warm"] = self.dist.warmed
            if not tokens:
                self.optimizer.zero_grad(set_to_none=True)
                result.update(ok=False, error="no trained tokens")
                return result
            divisor = len(per_try) if per_try else tokens
            policy = self.policy_params()
            with torch.no_grad():
                for _, param in policy:
                    if param.grad is not None:
                        param.grad.div_(divisor)
            losses /= divisor
            if getattr(self, "draft", None) is not None:  # stage D: divided by its rows, clipped on its own
                result.update(self.draft_finish(cmd))
            grads = [p.grad for _, p in policy if p.grad is not None]
            mark("divide")
            if cmd.get("ref") == "keep":  # test hook: this step's logprobs and raw gradients kept as self.ref
                self.ref = {"actor": {k: (a.cpu(), v.cpu()) for k, (a, v) in ref_new.items()},
                            "grads": [g.detach().float().cpu() for g in grads]}
            mark("ref")
            if os.environ.get("RL1_CLIP_PARTS") == "1":  # U2 diag: clip's parts timed, stacks dumped if it hangs
                norm, result["clip_parts"] = clip_parts([p for _, p in policy], float(cmd.get("clip", 1.0)))
            else:
                norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in policy], float(cmd.get("clip", 1.0)),
                                                            foreach=True))
            mark("clip")
            finite &= math.isfinite(norm)
            result["entropy"] = round(entropy_sum[0] / entropy_sum[1], 5) if entropy_sum[1] else None
            result.update(loss=round(losses, 6), grad_norm=round(norm, 6),
                          grad_norm_after_clip=round(min(norm, float(cmd.get("clip", 1.0))), 6), lr=lr,
                          trained_tokens=tokens, grads=len(grads))
            if not finite:
                self.optimizer.zero_grad(set_to_none=True)
                result.update(ok=False, stop="nan_or_inf", applied=False)
                return result
            stepped = time.time()
            before = [p.detach().clone() for _, p in self.params]  # on the card: no 21 GiB host copy
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                moved = [((p.detach() - b).float().pow(2).sum(), n.startswith(DRAFT_PREFIX))
                         for (n, p), b in zip(self.params, before)]
                change = math.sqrt(float(sum(v for v, d in moved if not d)))
                if getattr(self, "draft", None) is not None:
                    result["draft_update_norm"] = round(math.sqrt(float(sum(v for v, d in moved if d))), 6)
                    result["draft_norm"] = round(math.sqrt(sum(float(p.detach().float().pow(2).sum())
                                                               for _, p in self.draft_params())), 5)
            del before
            result.update(self.window_stats_take())
            mark("optimizer")
            result["step_parts"] = parts
            result["step_seconds"] = round(time.time() - stepped, 1)
            result["train_seconds"] = round(time.time() - began, 1)
        self.updates += 1
        norm_after = self.adapter_norm()
        result["expert_lora_calls"] = dict(GROUPED)  # cumulative since load: grouped vs loop, per dtype pair
        result["topk"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in TOPK.items()}  # since load
        if "rl1_gdn" in sys.modules:  # GDN layer calls by path since load (fallback reasons)
            gdn = sys.modules["rl1_gdn"].GDN
            result["gdn_calls"] = {"fast": gdn["fast"], "calls": gdn["calls"], "fallback": gdn["fallback"],
                                   "why": dict(gdn["why"])}
        if "rl1_hyper" in sys.modules:  # stage H layer calls by path since load
            hyper = sys.modules["rl1_hyper"].HYPER
            result["hyper_calls"] = {"fast": hyper["calls"], "fallback": hyper["fallback"], "why": dict(hyper["why"]),
                                    "dtypes": dict(hyper.get("dtypes") or {})}
        if "rl1_cast" in sys.modules:  # item 2: BF16 copies made / reused, PEFT round trips skipped since load
            result["cast_once"] = dict(sys.modules["rl1_cast"].CAST)
        if "rl1_dequant" in sys.modules:  # stage Q dequant calls by path since load
            deq = sys.modules["rl1_dequant"].DEQUANT
            result["dequant_calls"] = {"fast": deq["calls"], "fallback": deq["fallback"]}
        if SEL["on"] or SEL["hits"]:  # stage S: recompute reuses of the forward's picks since load
            result["sel_cache"] = {"hits": SEL["hits"], "stores": SEL["stores"], "causal_hits": CAUSAL["hits"]}
        if NGRAM["calls"]:
            result["ngram_pinned_calls"] = NGRAM["calls"]
        if "rl1_experts" in sys.modules:  # stage X MoE calls by path since load
            xs = sys.modules["rl1_experts"].EXPERTS
            result["expert_calls"] = {"fused": xs["calls"], "fallback": xs["fallback"], "why": dict(xs["why"]),
                                      "dtypes": dict(xs.get("dtypes") or {})}
            if sys.modules["rl1_experts"].COMPACT["on"]:
                result["expert_compact"] = sys.modules["rl1_experts"].summary()
        result.update(sampler.summary(), update_norm=round(change, 6), adapter_norm=norm_after, applied=True,
                      torch_peak_gib=round(max(torch.cuda.max_memory_allocated() / 2**30, rl1_memory.CKPT["step_peak"]), 2),
                      update=self.updates)
        if not math.isfinite(norm_after):
            result.update(ok=False, stop="nan_or_inf")
            return result
        if getattr(self, "dist", None) is not None and cmd.get("dist_digest"):  # the adapter identical on every rank
            result["dist_digest"] = self.dist.digest(self.params)
            if not result["dist_digest"]["equal"]:
                result.update(ok=False, stop="dist_desync")
                return result
        torch.cuda.empty_cache()
        if cmd.get("save") and (getattr(self, "dist", None) is None or self.dist.rank == 0):  # RL1_DIST: rank 0's file
            result["adapter"] = self.save_adapter(cmd["save"], background=bool(cmd.get("save_async")))
        result["ok"] = True
        return result

    def save_adapter(self, folder, background=False):
        """The served adapter (BF16) to ``folder``. ``background`` (cmd ``save_async``, mg.async_adapter_save): the
        host copy is taken now, the files are written by a thread into ``folder``.partial and renamed to ``folder``
        once complete; ``op_save_join`` (the loop's, before the servers load it) and any op that changes the adapter
        wait for it. D1 26 Oct: 2.0 s (H4 8.8, H5 11.8) inside the step call."""
        import torch
        from peft.utils import get_peft_model_state_dict

        self.save_join()  # one write at a time; the previous one lands first
        began = time.time()
        folder = Path(folder)
        state = {k: v.detach().to(torch.bfloat16).contiguous().cpu()
                 for k, v in get_peft_model_state_dict(self.model).items()}
        expected = len(self.policy_params())
        if len(state) != expected:  # a partial adapter never reaches the server
            raise ValueError(f"adapter state has {len(state)} tensors, the trainer {expected} LoRA params")
        row = {"path": str(folder), "tensors": len(state),
               "gib": round(sum(t.numel() * 2 for t in state.values()) / 2**30, 2)}
        if not background:
            folder.mkdir(parents=True, exist_ok=True)
            return {**row, **self.write_adapter(folder, state), "seconds": round(time.time() - began, 1)}
        import shutil
        import threading

        partial = folder.with_name(folder.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True, exist_ok=True)
        pending = {"path": str(folder), "began": time.time()}

        def write():
            try:
                pending.update(self.write_adapter(partial, state))
                if folder.exists():
                    shutil.rmtree(folder)
                partial.rename(folder)
            except BaseException as error:  # noqa: BLE001 - raised by save_join
                pending["error"] = repr(error)[:1000]
            pending["write_seconds"] = round(time.time() - pending["began"], 1)

        self.save_pending = (threading.Thread(target=write, name="adapter-save", daemon=True), pending)
        self.save_pending[0].start()
        return {**row, "pending": True, "seconds": round(time.time() - began, 1)}

    def save_join(self):
        """Waits for a background adapter write; its row (None when none was pending). A failed write raises."""
        pending = getattr(self, "save_pending", None)
        if pending is None:
            return None
        thread, row = pending
        waited = time.time()
        thread.join()
        self.save_pending = None
        done = {k: v for k, v in row.items() if k != "began"}
        done["waited"] = round(time.time() - waited, 1)
        if done.get("error"):
            raise RuntimeError(f"background adapter save {done['path']} failed: {done['error']}")
        return done

    def op_save_join(self, cmd):
        """The step's background adapter write landed (the loop calls it before the servers load the adapter)."""
        done = self.save_join()
        return {"pending": done is not None, **(done or {})}

    def write_adapter(self, folder, state):
        """The adapter files of ``state`` (host BF16 tensors) in ``folder``: weights, config, the draft's files."""
        from safetensors.torch import save_file

        save_file(state, str(folder / "adapter_model.safetensors"))
        (folder / "adapter-tensors.json").write_text(json.dumps({"tensors": len(state), "expected": len(state)}))
        self.model.peft_config["default"].save_pretrained(str(folder))
        config = json.loads((folder / "adapter_config.json").read_text())
        config.pop("nvfp4_merge_aware", None)
        config["inference_mode"] = True
        (folder / "adapter_config.json").write_text(json.dumps(config, indent=1))
        draft = None
        if getattr(self, "draft", None) is not None:  # stage D: beside it, its own files
            import rl1_draft

            draft = len(rl1_draft.export(self.draft.head, folder)["modules"])
        return {"draft_modules": draft} if draft else {}

    # ---- play: the adapter master and the optimizer state wait in pinned host memory ----

    def held(self):
        """(key, holder, field) of every adapter parameter and optimizer-state tensor."""
        import torch

        for name, param in self.params:
            yield ("param", name), param, None
            state = self.optimizer.state.get(param) or {}
            for field in [f for f, v in state.items() if torch.is_tensor(v)]:
                yield ("state", name, field), state, field

    def base_tensors(self):
        """The frozen model's CUDA tensors (module parameters, buffers, plain tensor attributes) that alone own a whole
        storage; the adapter (held()), views and shared storages stay where they are."""
        import torch

        skip, seen, names = {id(p) for _, p in self.params}, {}, {}

        def add(value, name, depth=0):
            if not torch.is_tensor(value) or id(value) in skip or depth > 4:
                return
            if hasattr(value, "__tensor_flatten__"):  # a wrapper subclass (e.g. a quantised weight): its inner tensors
                for inner in value.__tensor_flatten__()[0]:
                    add(getattr(value, inner, None), f"{name}.{inner}", depth + 1)
                return
            if value.is_cuda or os.environ.get("RL1_BASE_ANY_DEVICE") == "1":  # CPU gate: the park path on CPU ranks
                seen.setdefault(id(value), value)
                names.setdefault(id(value), name)

        draft = [("draft." + n, m) for n, m in self.draft.named_modules()] if getattr(self, "draft", None) is not None \
            else []
        for prefix, module in [*self.model.named_modules(), *draft]:
            for key, value in [*module._parameters.items(), *module._buffers.items(), *vars(module).items()]:
                add(value, f"{prefix}.{key}" if prefix else key)
        storages, census = {}, {"found": len(seen), "no_pointer": [], "kept": 0, "kept_gib": 0.0}
        for value in seen.values():
            try:  # tscheck-1: a storage without a data pointer raised here; such tensors stay on the card
                storages.setdefault(value.untyped_storage().data_ptr(), []).append(value)
            except RuntimeError:
                if len(census["no_pointer"]) < 8:
                    census["no_pointer"].append([type(value).__name__, str(value.dtype), list(value.shape)])
                continue
        out, self.base_names = [], []
        for v in storages.values():
            ok = (len(v) == 1 and type(v[0]) in (torch.Tensor, torch.nn.Parameter) and v[0].is_contiguous()
                  and v[0].storage_offset() == 0 and v[0].numel() * v[0].element_size() == v[0].untyped_storage().nbytes())
            if ok:
                out.append(v[0])
                self.base_names.append(names.get(id(v[0]), "?"))
            else:
                census["kept"] += len(v)
                census["kept_gib"] += sum(t.numel() * t.element_size() for t in v) / 2**30
        census["kept_gib"] = round(census["kept_gib"], 2)
        self.base_census = census
        return out

    def park_base_shm(self):
        """The park (real-v5 27 Oct .. gate leg 5 28 Oct): the frozen base points at one /dev/shm snapshot shared by the
        launch's ranks (rl1_park), written on a thread from the model load (start_base_snap; else here, by local rank
        0) and checked once by every rank; later parks copy nothing. A changed base layout keeps the snapshot's views
        of the unchanged tensors (rl1_park.reuse), checked bytewise at the next unpark; the snapshot is pinned
        (cudaHostRegister) on a thread after the park. Returns (bytes moved off the card, tensors kept on it)."""
        import torch

        import rl1_park

        early = self.base_snap_join()
        tensors, kept = list(self.base_tensors()), 0
        names = list(getattr(self, "base_names", None) or [])
        layout = rl1_park.layout_of(tensors)[0]
        snap = getattr(self, "base_snap", None)
        if snap is None or snap["layout"] != layout:
            rank = int(os.environ.get("LOCAL_RANK") or 0)
            if snap is not None and getattr(self, "base_sums", None):
                n = self.base_snaps = getattr(self, "base_snaps", 0) + 1
                views, info = rl1_park.reuse(snap, tensors, names, self.base_sums, rank, snap["info"]["path"], n)
                info["maps"] = snap["info"]["maps"] + info["maps"]
                self.base_verify = info.pop("reused_names")  # checked bytewise at the next unpark
            else:
                if snap is not None:
                    self.base_register_join()
                    rl1_park.unregister(self.base_registered())
                views, info = self.base_snapshot(tensors, names, rank)
            snap = self.base_snap = {"layout": layout, "views": views, "info": info, "names": names}
            self.base_shm_info = {k: v for k, v in info.items() if k != "maps"}  # the park result carries it once
            out = os.environ.get("ARC_VALIDATION_OUTPUT")  # every rank's split (rank > 0 results stay on that rank)
            if out:
                try:
                    Path(out, "util").mkdir(parents=True, exist_ok=True)
                    Path(out, "util", f"park-base-shm-r{rank}.json").write_text(json.dumps(self.base_shm_info))
                except OSError:
                    pass
            print(json.dumps({"event": "park_base_shm", **self.base_shm_info}), flush=True)
        if early is not None:  # the first park after start_base_snap: its thread's report rides on the park result
            self.base_shm_info = {**(getattr(self, "base_shm_info", None) or {}), "early": early}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        moved, self.base_parked, self.base_parked_names = 0, [], []
        for value, view, name in zip(tensors, snap["views"], names):
            try:
                value.data = view
            except Exception:  # a tensor subclass that refuses the swap stays on the card
                kept += 1
                continue
            self.base_parked.append(value)
            self.base_parked_names.append(name)
            moved += value.numel() * value.element_size()
        self.base_register_start()  # after the park: the pinning runs while the trainer idles through play
        return moved, kept

    def base_snapshot(self, tensors, names, rank):
        """The first snapshot, with every tensor's checksum taken first (rl1_park.reuse and the unpark check)."""
        import rl1_park

        began = time.time()
        self.base_sums = {nm: rl1_park.checksum(t) for nm, t in zip(names, tensors)}
        sums_s = round(time.time() - began, 2)
        views, info = rl1_park.snapshot(tensors, rank, names=names)
        info["sums_s"] = sums_s
        return views, info

    def start_base_snap(self):
        """The base snapshot written on a thread from the end of the model load, so the first park only points the base
        at it. The park joins it; a failure leaves the park to write."""
        import torch

        if getattr(self, "_base_snap_thread", None) is not None:
            return None
        tensors = list(self.base_tensors())
        names = list(self.base_names)
        rank = int(os.environ.get("LOCAL_RANK") or 0)
        state = {"tensors": len(tensors), "seconds": None, "error": None}

        def work():
            began = time.time()
            try:
                if torch.cuda.is_available():
                    torch.cuda.set_device(CARD)
                import rl1_park

                views, info = self.base_snapshot(tensors, names, rank)
                self.base_snap = {"layout": rl1_park.layout_of(tensors)[0], "views": views, "info": info,
                                  "names": names}
                self.base_shm_info = {k: v for k, v in info.items() if k != "maps"}
            except Exception as exc:  # noqa: BLE001 - the park writes the snapshot itself
                state["error"] = f"{type(exc).__name__}: {exc}"[:300]
                self.base_snap = None
            state["seconds"] = round(time.time() - began, 1)

        thread = threading.Thread(target=work, name="base-snap", daemon=True)
        thread.start()
        self._base_snap_thread = (thread, state)
        return {"started": True, "tensors": len(tensors)}

    def base_snap_join(self):
        pre = getattr(self, "_base_snap_thread", None)
        if pre is None or pre[1].get("joined"):
            return None
        waited = time.time()
        pre[0].join()
        pre[1]["joined"] = True
        return {**pre[1], "wait_s": round(time.time() - waited, 1)}

    def base_register_start(self):
        """Pin the snapshot mappings (cudaHostRegister, logged fallback) on a thread; unpark joins it."""
        import rl1_park

        sent = self.base_reg_ids = getattr(self, "base_reg_ids", set())
        maps = [m for m in self.base_snap["info"]["maps"] if id(m) not in sent]
        if not maps:
            return
        sent |= {id(m) for m in maps}
        state = {}

        def work():
            state.update(rl1_park.register(maps, device=CARD if CARD != "cpu" else None))

        thread = threading.Thread(target=work, name="base-register", daemon=True)
        thread.start()
        self._base_reg = getattr(self, "_base_reg", []) + [(thread, state, maps)]

    def base_register_join(self):
        """Joined register threads' reports (once each; bufs kept for unregister)."""
        out = []
        for thread, state, maps in getattr(self, "_base_reg", []):
            if state.get("reported"):
                continue
            waited = time.time()
            thread.join()
            state["reported"] = True
            out.append({k: v for k, v in state.items() if k not in ("bufs", "reported")} |
                       {"wait_s": round(time.time() - waited, 1)})
        return out

    def base_registered(self):
        bufs = [b for _, state, _ in getattr(self, "_base_reg", []) for b in state.get("bufs", [])]
        self._base_reg, self.base_reg_ids = [], set()
        return bufs

    def prepin_plan(self, on_card=lambda t: t.is_cuda):
        """(key, shape, dtype) of every park buffer: the adapter params, and the optimizer state each will hold, held()
        keys as they exist now or as fused AdamW creates them at the first step (exp_avg / exp_avg_sq like the param,
        a float32 scalar step on the card). A wrong guess costs one allocation: the park checks shape and dtype."""
        import torch

        plan = []
        adam = isinstance(self.optimizer, torch.optim.AdamW)
        for name, param in self.params:
            if not on_card(param):
                continue
            plan.append((("param", name), tuple(param.shape), param.dtype))
            state = self.optimizer.state.get(param) or {}
            if state:
                plan += [(("state", name, f), tuple(v.shape), v.dtype) for f, v in state.items()
                         if torch.is_tensor(v) and on_card(v)]
            elif adam:
                plan += [(("state", name, "step"), (), torch.float32),
                         (("state", name, "exp_avg"), tuple(param.shape), param.dtype),
                         (("state", name, "exp_avg_sq"), tuple(param.shape), param.dtype)]
        return plan

    def start_prepin(self, plan=None):
        """T4r-v2 (26 Oct): the resumed run's first park took 29.6 s for 31 GiB (a warm park 5.5-6.6 s), almost all
        pinned allocation. The buffers are allocated on a thread from the end of the model load (beside the resume's
        checkpoint read and the adapter save); the park joins it (prepin_join) and copies only."""
        import threading

        import torch

        if getattr(self, "_prepin", None) is not None:
            return {"already": True}
        todo = [t for t in (self.prepin_plan() if plan is None else plan) if t[0] not in self.pinned]
        size = lambda shape, dtype: torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        state = {"tensors": len(todo), "bytes": sum(size(s, d) for _, s, d in todo), "done_bytes": 0,
                 "seconds": None, "error": None}
        state["gib"] = round(state["bytes"] / 2**30, 2)

        def work():
            began = time.time()
            try:
                for key, shape, dtype in todo:
                    self.pinned.setdefault(key, torch.empty(shape, dtype=dtype, pin_memory=True))
                    state["done_bytes"] += size(shape, dtype)
            except Exception as exc:  # noqa: BLE001 - the park allocates whatever is missing
                state["error"] = f"{type(exc).__name__}: {exc}"[:300]
            state["seconds"] = round(time.time() - began, 1)

        thread = threading.Thread(target=work, name="prepin", daemon=True)
        thread.start()
        self._prepin = (thread, state)
        return {"started": True, "tensors": state["tensors"], "gib": state["gib"]}

    def prepin_pending_bytes(self):
        """Pinned bytes the prepin thread has still to allocate (a host-memory guard counts them as taken)."""
        pre = getattr(self, "_prepin", None)
        if pre is None or not pre[0].is_alive():
            return 0
        return int(pre[1].get("bytes", 0)) - int(pre[1].get("done_bytes", 0))

    def prepin_join(self):
        pre = getattr(self, "_prepin", None)
        if pre is None:
            return None
        pre[0].join()
        return dict(pre[1])

    def compact_experts(self, model=None):
        """The masked experts' NVFP4 base cut from every masked layer (rl1_experts.compact_experts); routes, LoRA and
        R3 stay on the 512 global ids. Needs the fused expert path on every op (trainer_fast fused_experts)."""
        import torch

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out = host_compact(self.model if model is None else model)
        torch.cuda.synchronize()
        out["allocated_freed_gib"] = round((before - torch.cuda.memory_allocated()) / 2 ** 30, 2)
        out["allocated_gib"] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
        return out

    def op_park(self, cmd=None):
        import torch

        if self.parked:
            return {"parked": True, "already": True}
        waited = time.time()
        prepin = self.prepin_join()
        if prepin is not None:
            prepin["wait_s"] = round(time.time() - waited, 1)
        torch.cuda.synchronize()
        if (cmd or {}).get("base"):
            base, kept = self.park_base_shm()
        else:
            base, kept = 0, 0
        began, moved, swaps = time.time(), 0, []
        # card2.continuous: a park mid-batch keeps the accumulated gradients on pinned host memory,
        # restored by unpark; their sum of squares both ways is the in-run check
        keep = bool((cmd or {}).get("keep_grads")) and bool(getattr(self, "acc", None)) and bool(self.acc["calls"])
        kept_grads, grad_sq = {}, 0.0
        for name, param in self.params:
            if keep and param.grad is not None:
                grad_sq += float(param.grad.detach().float().pow(2).sum())
                buffer = torch.empty(param.grad.shape, dtype=param.grad.dtype, pin_memory=True)
                buffer.copy_(param.grad, non_blocking=True)
                kept_grads[name] = buffer
            param.grad = None
        torch.cuda.synchronize()
        self.parked_grads = kept_grads
        for key, holder, field in self.held():
            value = holder.data if field is None else holder[field]
            if not value.is_cuda:
                continue
            buffer = self.pinned.get(key)
            if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                buffer = self.pinned[key] = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
            buffer.copy_(value, non_blocking=True)
            moved += value.numel() * value.element_size()
            swaps.append((holder, field, buffer))
        torch.cuda.synchronize()
        for holder, field, buffer in swaps:
            if field is None:
                holder.data = buffer
            else:
                holder[field] = buffer
        self.parked = True
        torch.cuda.empty_cache()
        used, _, _ = gpu_used_gib()
        shm_info, self.base_shm_info = getattr(self, "base_shm_info", None), None
        released = fq.release_host()  # the update's freed host chunks back to the OS before play (real-v10 28 Oct)
        return {"parked": True, "gib": round(moved / 2**30, 2), "seconds": round(time.time() - began, 2), "prepin": prepin,
                "release_host": released,
                **({"base_shm": shm_info} if shm_info else {}),
                "grads_kept": {"n": len(kept_grads), "sq": grad_sq} if keep else None,
                "tensors": len(swaps), "card_used_gib": used, "base_gib": round(base / 2**30, 2), "base_kept": kept,
                "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}

    def unpark(self):
        import torch

        began, moved = time.time(), 0
        registered = self.base_register_join()  # the snapshot's pinning lands before any copy from it
        for value in getattr(self, "base_parked", []):
            if not value.is_cuda:
                value.data = value.data.to(CARD, non_blocking=True)
                moved += value.numel() * value.element_size()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        base_s, base_moved = round(time.time() - began, 2), moved
        verify = self.base_verify_check()
        self.base_parked, self.base_parked_names = [], []
        for key, holder, field in self.held():
            value = holder.data if field is None else holder[field]
            if value.is_cuda or not value.is_pinned():
                continue
            moved_value = value.to(CARD, non_blocking=True)
            moved += value.numel() * value.element_size()
            if field is None:
                holder.data = moved_value
            else:
                holder[field] = moved_value
        torch.cuda.synchronize()
        out = {}
        kept = getattr(self, "parked_grads", None) or {}
        if kept:  # card2.continuous: the batch's accumulated gradients back on the card
            grad_sq = 0.0
            for name, param in self.params:
                if name in kept:
                    param.grad = kept[name].to(param.device, non_blocking=True)
            torch.cuda.synchronize()
            for name, param in self.params:
                if name in kept:
                    grad_sq += float(param.grad.detach().float().pow(2).sum())
            out["grads_restored"] = {"n": len(kept), "sq": grad_sq}
        self.parked_grads = {}
        self.parked = False
        out["base_s"], out["base_gib"] = base_s, round(base_moved / 2**30, 2)
        if registered:
            out["shm_register"] = registered
        if verify is not None:
            out["base_verify"] = verify
        return {"gib": round(moved / 2**30, 2), "seconds": round(time.time() - began, 2), **out}

    def base_verify_check(self):
        """After a park on a reused snapshot (rl1_park.reuse): every reused base tensor's checksum, back on the card,
        against the one taken from the loaded weights at the first snapshot. A mismatch aborts (the base would be
        silently wrong)."""
        names = getattr(self, "base_verify", None)
        if not names:
            return None
        import rl1_park

        began, bad, n = time.time(), [], 0
        for value, name in zip(self.base_parked, getattr(self, "base_parked_names", [])):
            if name in names:
                n += 1
                if rl1_park.checksum(value) != self.base_sums.get(name):
                    bad.append(name)
        self.base_verify = None
        if bad or n != len(names):
            raise RuntimeError(f"park base_shm reuse: checksum mismatch on {len(bad)} of {n} tensors "
                               f"(expected {len(names)}): {bad[:8]}")
        return {"tensors": n, "seconds": round(time.time() - began, 2)}

    def op_unpark(self, cmd=None):
        return self.unpark() if self.parked else {"already": True}

    @staticmethod
    def host_free_bytes():
        """Host memory a new allocation can use: the cgroup limit less its anonymous memory (page cache is
        reclaimable), and the machine's MemAvailable; the smaller. None when neither is readable."""
        out = []
        try:
            top = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            rows = dict(l.split() for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines() if l.count(" ") == 1)
            if top.isdigit():
                out.append(int(top) - int(rows.get("anon", 0)))
        except (OSError, ValueError):
            pass
        try:
            info = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
            out.append(int(info["MemAvailable"].split()[0]) * 1024)
        except (OSError, ValueError, KeyError):
            pass
        return min(out) if out else None

    def ckpt_join(self):
        """Waits for a background checkpoint (``op_checkpoint`` with ``background``); its result, None when none was
        pending. A failed write raises here (the next checkpoint, load, wait or exit reports it)."""
        thread = getattr(self, "ckpt_thread", None)
        if thread is None:
            return None
        waited = time.time()
        thread.join()
        self.ckpt_thread = None
        done = dict(self.ckpt_done or {}, waited=round(time.time() - waited, 1))
        if done.get("error"):
            raise RuntimeError(f"background checkpoint {done.get('path')} failed: {done['error']}")
        return done

    def op_checkpoint_wait(self, cmd):
        """The pending background checkpoint's result (the loop reads the state folder after it)."""
        done = self.ckpt_join()
        return {"pending": done is not None, **(done or {})}

    def op_checkpoint(self, cmd):
        """The fp32 adapter master and the optimizer state (resume a later loop from this update), to ``path``.
        ``background`` (MFU-1 E, 23 Oct): host copies taken now, written by a thread into ``path``.partial and renamed
        to ``path`` when complete, while the server plays; the next checkpoint / load / wait / exit joins it."""
        import torch
        from safetensors.torch import save_file

        began = time.time()
        prior = self.ckpt_join()
        folder = Path(cmd["path"])
        refused = None
        if cmd.get("background"):  # the host snapshot (master + optimizer state, ~31 GiB on Flash) must fit with room
            need = sum(p.numel() * 4 for _, p in self.params) + sum(
                v.numel() * v.element_size() for st in self.optimizer.state.values() for v in st.values()
                if isinstance(v, torch.Tensor))
            free = self.host_free_bytes() if cmd.get("free_bytes") is None else int(cmd["free_bytes"])
            if free is not None:  # MFU-2 (RL 23 Oct): pinned park buffers the prepin thread has yet to allocate count
                free -= self.prepin_pending_bytes()  # as taken (the host-load copies are freed before any checkpoint)
            if free is not None and free < 1.25 * need + 16 * 2**30:
                refused = {"need_gib": round(need / 2**30, 1), "free_gib": round(free / 2**30, 1)}
        if cmd.get("background") and refused is None:
            import copy as _copy
            import shutil
            import threading

            copy_groups = lambda g: [{k: copy(v) for k, v in _copy.deepcopy(  # noqa: E731
                {k: v for k, v in x.items() if not isinstance(v, torch.Tensor)}).items()} | {
                k: copy(v) for k, v in x.items() if isinstance(v, torch.Tensor)} for x in g]
            copy = lambda t: t.detach().to("cpu", copy=True) if isinstance(t, torch.Tensor) else t  # noqa: E731
            master = {name: p.detach().to("cpu", dtype=torch.float32, copy=True).contiguous() for name, p in self.params}
            opt = self.optimizer.state_dict()
            opt = {"state": {k: {n: copy(v) for n, v in st.items()} for k, st in opt["state"].items()},
                   "param_groups": copy_groups(opt["param_groups"])}
            meta = {"updates": self.updates, "names": [name for name, _ in self.params]}
            snapshot = round(time.time() - began, 1)

            def write():
                t0 = time.time()
                try:
                    tmp = folder.with_name(folder.name + ".partial")
                    shutil.rmtree(tmp, ignore_errors=True)
                    tmp.mkdir(parents=True)
                    save_file(master, str(tmp / "master.safetensors"))
                    torch.save({"optimizer": opt, **meta}, str(tmp / "optimizer.pt"))
                    if folder.exists():
                        shutil.rmtree(folder)
                    tmp.rename(folder)
                    self.ckpt_done = {"path": str(folder), "write_seconds": round(time.time() - t0, 1),
                                      "gib": round(sum(f.stat().st_size for f in folder.iterdir()) / 2**30, 2)}
                except Exception as error:  # noqa: BLE001 - raised at the join
                    self.ckpt_done = {"path": str(folder), "error": repr(error)[:1000]}

            self.ckpt_done = None
            self.ckpt_thread = threading.Thread(target=write, daemon=True)
            self.ckpt_thread.start()
            return {"path": str(folder), "background": True, "snapshot_seconds": snapshot,
                    **({"prior": prior} if prior else {})}
        folder.mkdir(parents=True, exist_ok=True)
        save_file({name: p.detach().float().contiguous().cpu() for name, p in self.params}, str(folder / "master.safetensors"))
        torch.save({"optimizer": self.optimizer.state_dict(), "updates": self.updates,
                    "names": [name for name, _ in self.params]}, str(folder / "optimizer.pt"))
        return {"path": str(folder), "seconds": round(time.time() - began, 1),
                "gib": round(sum(f.stat().st_size for f in folder.iterdir()) / 2**30, 2),
                **({"prior": prior} if prior else {}), **({"background_refused": refused} if refused else {})}

    def op_load_checkpoint(self, cmd):
        """A state ``op_checkpoint`` wrote (another run's): adapter params and optimizer state, the update count;
        the base references stay (the probes' KL to base keeps its reference)."""
        import torch
        from safetensors.torch import load_file

        began = time.time()
        self.ckpt_join()  # a background checkpoint of this folder lands first
        folder = Path(cmd["path"])
        master = load_file(str(folder / "master.safetensors"))
        state = torch.load(str(folder / "optimizer.pt"), map_location="cpu", weights_only=False)
        names = [name for name, _ in self.params]
        opt = state["optimizer"]
        fresh_draft = (getattr(self, "draft", None) is not None and state["names"] == [n for n, _ in self.policy_params()]
                       and set(master) == set(state["names"]))
        if fresh_draft:  # stage D: a policy-only checkpoint; the draft's LoRA keeps its initial state and no Adam state
            group = self.optimizer.state_dict()["param_groups"][-1]
            opt = {"state": dict(opt["state"]), "param_groups": list(opt["param_groups"]) + [group]}
        elif state["names"] != names or set(master) != set(names):
            raise ValueError(f"checkpoint layout differs: {len(master)} tensors, {len(names)} params")
        with torch.no_grad():
            for name, param in self.params:
                if name in master:
                    param.copy_(master[name].to(device=param.device, dtype=param.dtype))
        groups = [float(g.get("scale", 1.0)) for g in self.optimizer.param_groups]
        saved = [float(g.get("scale", 1.0)) for g in opt["param_groups"]]
        if groups != saved:
            raise ValueError(f"checkpoint optimizer groups {saved} differ from this trainer's {groups}")
        self.optimizer.load_state_dict(opt)
        rl1_table.guard(getattr(self, "model", None), self.optimizer, "load_checkpoint")
        self.updates = int(state["updates"])
        return {"path": str(folder), "updates": self.updates, "adapter_norm": self.adapter_norm(),
                "lr": self.optimizer.param_groups[0]["lr"], "seconds": round(time.time() - began, 1)}

    def op_save(self, cmd):
        return self.save_adapter(cmd["path"])

    def serve(self):
        if getattr(self, "dist", None) is not None:  # RL1_DIST: rank 0 answers, every rank runs its share
            PROGRESS.update(phase="idle", op=None)
            return rl1_dist.serve(self)
        index = 0
        alive = self.dir / "alive.json"
        alive.write_text(json.dumps({"state": "ready", "loaded": self.loaded, "time": time.time()}, default=str))
        PROGRESS.update(phase="idle", op=None)
        while True:
            path = self.dir / "cmd" / f"{index:04d}.json"
            if not path.exists():
                alive.write_text(json.dumps({"state": "idle", "next": index, "time": time.time(), "loaded": self.loaded},
                                            default=str))
                early_poll()
                time.sleep(min(1.0, float(os.environ.get("RL1_DIST_POLL_S") or 1.0)))
                continue
            cmd = json.loads(path.read_text())
            alive.write_text(json.dumps({"state": "busy", "op": cmd["op"], "n": index, "time": time.time()}))
            result = self.run_op(cmd, index)
            tmp = self.dir / "done" / f"{index:04d}.json.tmp"
            tmp.write_text(json.dumps(result, default=str))
            tmp.rename(self.dir / "done" / f"{index:04d}.json")
            index += 1
            if cmd["op"] == "exit":
                return

    def run_op(self, cmd, index):
        """One command: the op's result (an exception is the answer), its seconds and memory; recorded."""
        PROGRESS.update(phase=PHASES.get(cmd["op"], cmd["op"]), op=cmd["op"])
        pre, joined = getattr(self, "_prepin", None), None
        if pre is not None and pre[0].is_alive() and cmd["op"] in GPU_OPS and os.environ.get("RL1_PREPIN_JOIN", "1") == "1":
            # U2 (27 Oct): the prepin thread's cudaHostAlloc calls (U1d: 2,696 in the first accumulate, 9.8 s, beside
            # ~15 s of launch stalls) do not run beside a GPU op; a real run's play phase hides it (replay legs only)
            joined = time.time()
            pre[0].join()
            joined = round(time.time() - joined, 1)
        began = time.time()
        early = getattr(self, "early", None) or {}
        if early.get("state") == "expanding":  # an op came first: the call encodes for itself
            early.update(state="done", cut_by=cmd["op"])
        try:
            saved = self.save_join() if cmd["op"] in SAVE_JOIN_OPS else None  # a background adapter write lands first
            if cmd["op"] == "exit":
                result = {"ok": True, **({"checkpoint": done} if (done := self.ckpt_join()) else {}),
                          **({"adapter_save": saved} if saved else {})}
            else:
                unparked = self.unpark() if self.parked and cmd["op"] in GPU_OPS else None
                self.snapshots = bool(cmd.get("prefix_snapshots"))  # fix a: per op, off unless asked
                self.probe_pinned = bool(cmd.get("probe_pinned"))  # kl_base_pinned diagnostic (trainer_fast knob)
                op_paths(self.model, cmd)
                op_began = time.time()
                with profiled(cmd["op"]) as prof:  # RL1_PROFILE_OPS: a torch profile of named calls
                    result = getattr(self, f"op_{cmd['op']}")(cmd)
                result["pre_seconds"] = round(op_began - began, 2)  # save join, unpark, paths before the op
                result["op_wall_seconds"] = round(time.time() - op_began, 2)
                if prof.get("path") or prof.get("sync_census"):
                    result["profile"] = prof
                result.setdefault("ok", True)
                if unparked is not None:
                    result["unparked"] = unparked
        except Exception as error:  # the failure is the answer
            import torch

            result = {"ok": False, "error": repr(error)[:2000], "traceback": traceback.format_exc()[-6000:]}
            # v17 a3/a5 (30 Oct): the traceback reached only the done/ file in the container's /tmp, lost with it;
            # the trainer log keeps it, with where the call was (windows, tail) and the adaptive-ckpt state
            result["failed_at"] = {"progress": dict(PROGRESS), "ckpt": rl1_memory.ckpt_state(),
                                   "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                                   "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        result["seconds"] = round(time.time() - began, 1)
        if joined is not None:
            result["prepin_join_s"] = joined
        PROGRESS.update(phase="idle", op=None)
        # the play server wakes beside the idle trainer: hand every cached block back (rl1-ttest-v1: the card
        # held ~152 GiB for the trainer between updates against ~105 GiB of state). RL1_KEEP_RESERVE=1: not after an
        # accumulate (the update's next call or its step follows; D2 27 Oct: each call re-mapped ~200 GiB, 1-5 s
        # between calls at 0% util); the step and every other op still hand it back
        import torch

        if not (cmd["op"] in ("accumulate", "old_pass") and result.get("ok") and os.environ.get("RL1_KEEP_RESERVE") == "1"):
            torch.cuda.empty_cache()
        if rl1_memory.TAIL["offloaded"] and hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()  # tail_ckpt offload's pinned blocks back to the host (park, time share)
        used, _, _ = gpu_used_gib()
        result["memory"] = {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2), "card_used_gib": used}
        self.rec.event("op", op=cmd["op"], n=index, **{k: v for k, v in result.items() if k != "traceback" or not result.get("ok")})
        return result


GPU_OPS = ("accumulate", "old_pass", "step", "score", "update", "reset", "checkpoint", "load_checkpoint", "route")  # these bring a parked adapter back first
# ops that change the adapter (the draft's files are read in the background write) or end the process
SAVE_JOIN_OPS = ("accumulate", "step", "update", "reset", "load_checkpoint", "exit")


def load_model(rec, model_dir, out, before_gate=None, table_meta=False):
    """The trainer's model (one load on the card; its first CUDA use waits for the gate, RL1_GPU_GATE)."""
    return fq.load(rec, model_dir, out, sequence_len=131072, before_gate=before_gate, table_meta=table_meta)


def gpu_gate():
    """RL1_GPU_GATE (the loop's mg.trainer_overlap): a second server boots on this card meanwhile, so the load's CPU part
    runs now and the first CUDA initialisation (``torch.cuda._lazy_init``, which every tensor on the card goes through)
    waits until the loop writes the gate file. Returns the hook's state, or None without a gate."""
    path = os.environ.get("RL1_GPU_GATE")
    if not path:
        return None
    import torch

    real, state = torch.cuda._lazy_init, {"began": time.time(), "waited_s": None, "prefetch": True}
    if fq.preload_wanted():  # import-time device queries answered without a context until the gate (fq.preload_shims)
        state["shim"] = fq.preload_shims()

    def gated():
        if not torch.cuda.is_initialized() and not os.path.exists(path):
            if state.pop("prefetch", False):  # CPU-only imports the load needs next
                import traceback  # who touched the card first (RL1_PRELOAD_HOST: nothing before the move should)

                state["blocked_by"] = [f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}"
                                       for f in traceback.extract_stack()[-9:-1]]
                print(json.dumps({"event": "rl1_gpu_gate_blocked", "at_s": round(time.time() - state["began"], 1),
                                  "stack": state["blocked_by"]}), flush=True)
                for name in ("transformers", "peft", "axolotl.cli.config", "axolotl.train"):
                    try:
                        __import__(name)
                    except Exception:  # noqa: BLE001 - imported again (and reported) by the load itself
                        pass
            while not os.path.exists(path):
                early_poll()  # trainer_fast.early_ahead: the first group's encodes while the servers still play
                time.sleep(1.0)
        if state["waited_s"] is None:
            early_poll()  # gate already open (U1: no play, the load ran past the switch): encodes overlap the move
            state["waited_s"] = round(time.time() - state["began"], 1)
            print(json.dumps({"event": "rl1_gpu_gate_open", "waited_s": state["waited_s"]}), flush=True)
        return real()

    torch.cuda._lazy_init = gated
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="command / answer folder (and trainer records)")
    parser.add_argument("--model", required=True, help="the served NVFP4 checkpoint")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--window", type=int, default=24576)
    parser.add_argument("--optimizer", choices=("adamw32",), default="adamw32", help="fused fp32 AdamW (the only one)")
    parser.add_argument("--tis-cap", type=float, default=0.0, help="truncated importance sampling cap (0: off)")
    parser.add_argument("--prefix-chunk", type=int, default=16384, help="the one prefix chunking of every pass")
    args = parser.parse_args()
    gpu_gate()
    progress_writer()
    trainer = Trainer(args)
    trainer.serve()


if __name__ == "__main__":
    main()
