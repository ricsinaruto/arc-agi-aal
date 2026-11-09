"""The trunk/sft trainer on N GPUs in one container (general N, nothing lost from 1 to N).

Routed experts of the trunk (``qat.routed``: Qwen4ExpTextExperts under .language_model.layers.) are sharded on the
expert dim: rank r owns the contiguous slice ``ranges[r]`` of every layer's [E, ...] tensors (``split``: sizes differ
by at most one; the collectives pad to the largest). Everything else (dense weights, routers, the MTP block and its
experts) is replicated, DDP style. FSDP2 is not used: the dense fast paths (packed GDN views, compacted buffers,
hyper.py, union attention, sel_cache) keep pointers into parameter storage, which FSDP2 swaps every pass.

- Expert forward (``install``): local shard -> the QAT grid on the shard (per-expert scales: equal to quantizing after
  the gather; ``qat.refresh`` sees local shards only) -> ``_Gather`` (all-gather) -> the original forward on [E, ...].
  Its backward reduce-scatters the full gradient (``reduce_dtype``, fp32 default) back to the shard, cast to the
  parameter dtype; the shard's .grad accumulates across examples as N=1's does. Checkpoint recompute re-gathers.
- Dense gradients: no sync during the step's examples; at step end one bucketed fp32 all-reduce (``sync_dense``).
- Pass alignment: gathers and reduce-scatters are collectives, so every rank issues the same pass sequence. A
  training round is P no-grad passes then one gradient window on every rank (P = the round's most prefix chunks); a
  rank with fewer prefix chunks runs zero-scale dummy passes first, a rank with no example a dummy window
  (``rounds``, train.aligned). An eval pads its rank's pass count with dummy no-grad passes.
- Loss: every example scaled by 1 / (the step's global label count), so the ranks' gradients sum to N=1's.
- Clip: one global norm (shard norms all-reduced, replicated counted once), clip_grad_norm_'s rule.
- Optimizer: replicated params in group 0, shards in group 1; group 0 steps with an RNG seeded by (seed, step) on
  every rank (stochastic rounding stays identical), group 1 with (seed, step, rank) (``seeded_steps``); replicas are
  compared bitwise every ``replica_check_every`` steps (``check_replicas``).
- Load: the rank reads only its expert slice (per-expert keys, or the rows of a fused tensor) straight from the
  safetensors (``load_experts``); from_pretrained gets an index without the trunk experts (``filtered``) and builds
  shard-sized expert parameters (``building``). The n-gram table is read once by the launcher into shared memory
  (``load_tables``) and attached to every rank's model.
- Checkpoints (``Part``): rank 0 writes the replicated tensors, every rank its experts as per-expert keys (global
  expert ids, so from_pretrained at N=1 merges them) and its optimizer states by parameter name; rank 0 writes the
  index, shard.json (N, ranges) and state.json once every rank's done-marker exists, then renames. ``load_optimizer``
  re-slices any checkpoint (N=1's or another N's) to this rank's range on expert boundaries: 8-bit states keep their
  codes and block scales when an expert's size is a multiple of the block, else they are dequantized and requantized.
- ``launch``: one process per GPU (rank 0 in the caller's process, so the volume bridge and the guard stay there).
"""
import contextlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import weakref
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

NAMES = ("gate_up_proj", "down_proj")
FUSED = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")
PER = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
NGRAM = re.compile(r"^(.*\.ngram_embedding)\.(?:shard_\d+\.)?weight$")
DD = {"ctx": None}  # the run's Ctx; None on one GPU
ST = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8, "I8": torch.int8,
      "I32": torch.int32, "I64": torch.int64}


def split(n, world):
    """Contiguous [lo, hi) per rank, sizes differing by at most one (the first n % world ranks one larger)."""
    base, extra = divmod(n, world)
    out, at = [], 0
    for r in range(world):
        k = base + (r < extra)
        out.append((at, at + k))
        at += k
    return out


def mix(*parts):
    """A 63-bit seed from integers (rank -1: shared by every rank)."""
    h = 1469598103934665603
    for p in parts:
        h = ((h ^ (int(p) & 0xFFFFFFFFFFFF)) * 1099511628211) % (1 << 63)
    return h


class Ctx:
    """This rank's place: world, rank, the expert ranges, and the small collectives the trainer uses."""

    def __init__(self, device, experts, reduce_dtype="fp32"):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device(device)
        self.comm = self.device if dist.get_backend() == "nccl" else torch.device("cpu")
        if experts < self.world:
            raise ValueError(f"{experts} experts per layer on {self.world} ranks")
        self.experts, self.ranges = experts, split(experts, self.world)
        self.lo, self.hi = self.ranges[self.rank]
        self.sizes = [b - a for a, b in self.ranges]
        self.emax = max(self.sizes)
        self.even = len(set(self.sizes)) == 1
        if reduce_dtype == "bf16" and self.world != 2:
            # 2 ranks: one addition, bf16(a + b) = bf16(fp32(a + b)) for BF16 a, b (exact in fp32 unless their exponents
            # differ by > 15, and then both round to the larger); more ranks round after every addition
            raise ValueError(f"reduce_dtype bf16 equals fp32 only on 2 ranks (here {self.world})")
        self.reduce_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[reduce_dtype]
        self.stats = {"gathers": 0, "gather_gib": 0.0, "scatters": 0, "scatter_gib": 0.0, "gather_live_max": 0}
        self.live = 0
        self._spans = {}  # kind -> [(start, end)]: CUDA event pairs (nccl) or perf_counter pairs (gloo)

    def dead(self):
        self.live -= 1

    # --- collective timers ----------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def timed(self, kind):
        """Time a collective on the current stream (CUDA events, read at ``times()``; no sync here)."""
        if self.comm.type == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            yield
            b.record()
        else:
            a = time.perf_counter()
            yield
            b = time.perf_counter()
        self._spans.setdefault(kind, []).append((a, b))

    def times(self):
        """Seconds per kind since the last call ("gather", "scatter", "dense"), then reset."""
        out = {}
        for kind, spans in self._spans.items():
            if spans and isinstance(spans[0][0], float):
                out[kind] = sum(b - a for a, b in spans)
            elif spans:
                spans[-1][1].synchronize()
                out[kind] = sum(a.elapsed_time(b) for a, b in spans) / 1000.0
        self._spans = {}
        return {k: round(v, 4) for k, v in out.items()}

    # --- small collectives (host values) --------------------------------------------------------------------------
    def sums(self, values):
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.comm)
        dist.all_reduce(t)
        return t.tolist()

    def max_int(self, v):
        t = torch.tensor([int(v)], dtype=torch.int64, device=self.comm)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def gather_rows(self, values):
        """[rank][i] of one equal-length float list per rank."""
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.comm)
        out = torch.empty(self.world * len(values), dtype=torch.float64, device=self.comm)
        dist.all_gather_into_tensor(out, t)
        return out.view(self.world, -1).tolist()

    def bcast(self, obj):
        """rank 0's ``obj`` on every rank."""
        box = [obj if self.rank == 0 else None]
        dist.broadcast_object_list(box, src=0, device=self.comm if self.comm.type == "cuda" else None)
        return box[0]

    def barrier(self):
        if self.comm.type == "cuda":
            dist.barrier(device_ids=[self.comm.index])
        else:
            dist.barrier()

    # --- data ---------------------------------------------------------------------------------------------------
    def pick(self, items, cost):
        """This rank's share of ``items`` by LPT on ``cost(item)`` (deterministic from the list): largest first, each to
        the least-loaded rank (ties: lowest rank); a rank's items in that order (rounds pair similar sizes)."""
        return [items[i] for i in lpt([cost(x) for x in items], self.world)[self.rank]]

    def rounds(self, counts):
        """Per round i: the most no-grad prefix passes any rank's i-th example has (0 for a round some rank lacks)."""
        n = self.max_int(len(counts))
        if not n:
            return []
        rows = self.gather_rows(list(counts) + [-1] * (n - len(counts)))
        return [max(0, int(max(r[i] for r in rows))) for i in range(n)]

    # --- parameters ---------------------------------------------------------------------------------------------
    def split_params(self, model, head):
        """(replicated trainable params, expert shard params) in a fixed order (every rank the same)."""
        shards = [getattr(m, n) for m in routed(model) for n in NAMES]
        ids = {id(p) for p in shards}
        rep = [p for p in model.parameters() if p.requires_grad and id(p) not in ids] + list(head.parameters())
        return rep, shards


def lpt(costs, world):
    order = sorted(range(len(costs)), key=lambda i: (-costs[i], i))
    loads, out = [0.0] * world, [[] for _ in range(world)]
    for i in order:
        r = min(range(world), key=lambda k: (loads[k], k))
        out[r].append(i)
        loads[r] += costs[i]
    return out


def routed(model):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    return [mod for name, mod in model.named_modules() if isinstance(mod, m.Qwen4ExpTextExperts)
            and ".language_model.layers." in f".{name}"]


def routed_named(model):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, m.Qwen4ExpTextExperts)
            and ".language_model.layers." in f".{name}"]


def context(device, folder, reduce_dtype="fp32"):
    """The run's Ctx when torch.distributed runs with world > 1 (else None); the experts count from the config."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        DD["ctx"] = None
        return None
    config = json.loads((Path(folder) / "config.json").read_text())
    ctx = Ctx(device, int(config.get("text_config", config)["num_experts"]), reduce_dtype)
    DD["ctx"] = ctx
    return ctx


# --- the warm-up's failure protocol ---------------------------------------------------------------------------------
# A rank that runs out of memory mid-pass would leave the others blocked in their next collective. While WARM is on,
# every expert collective is preceded by a one-int all-reduce (MAX) of the ranks' status, buffers allocated first: a
# failed rank sends its code there once (``warm_guard``), the others raise PeerFailed at that same all-reduce, and all
# ranks leave the pass together. Warm-up only: each check is a host sync.
WARM = {"on": False}
OOM, ERROR = 1, 2


class PeerFailed(RuntimeError):
    def __init__(self, code):
        super().__init__("out of memory on another rank (warm-up)" if code == OOM else "another rank failed (warm-up)")
        self.code = code


def status(c, code):
    t = torch.tensor([int(code)], dtype=torch.int64, device=c.comm)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def check(c):
    if WARM["on"]:
        got = status(c, 0)
        if got:
            WARM["sent"] = True  # this rank's status went out with this all-reduce
            raise PeerFailed(got)


def is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error)


@contextlib.contextmanager
def warm_guard(c):
    """Around a warm-up pass on every rank: an exception on any rank (an OOM, or another error) ends the pass on all
    of them, each rank's raised (the others' as PeerFailed, an OOM's worded as one)."""
    WARM.update(on=True, sent=False)
    try:
        yield
        WARM["sent"] = True
        got = status(c, 0)  # a rank that failed after the last collective
        if got:
            raise PeerFailed(got)
    except BaseException as error:
        if not WARM["sent"]:  # (autograd may re-wrap a PeerFailed raised in a backward: the flag, not the type)
            WARM["sent"] = True
            status(c, OOM if is_oom(error) else ERROR)
        raise
    finally:
        WARM["on"] = False


# --- expert collectives --------------------------------------------------------------------------------------------

def gather(x, c):
    """[E, ...] from every rank's [n_r, ...] (padded to the largest on the wire)."""
    rest = tuple(x.shape[1:])
    src = x.contiguous()
    if not c.even:
        pad = x.new_zeros((c.emax, *rest))
        pad[: x.shape[0]] = src
        src = pad
    out = x.new_empty((c.world * c.emax, *rest))
    check(c)
    with c.timed("gather"):
        dist.all_gather_into_tensor(out, src)
    c.stats["gathers"] += 1
    c.stats["gather_gib"] += out.numel() * out.element_size() / 2 ** 30
    if not c.even:
        out = torch.cat([out[r * c.emax: r * c.emax + c.sizes[r]] for r in range(c.world)])
    c.live += 1  # gathered projections alive (saved for a backward, or in use): their most, per step
    c.stats["gather_live_max"] = max(c.stats["gather_live_max"], c.live)
    weakref.finalize(out, c.dead)
    return out


RS = {"bytes": 1 << 30}  # the reduce-scatter's send buffer (all ranks' rows of an expert chunk) in the reduce dtype


def reduce_scatter(g, c, dtype):
    """This rank's rows of the sum over ranks of [E, ...] gradients, summed in ``c.reduce_dtype``, cast to ``dtype``:
    in expert chunks whose send buffer is <= RS["bytes"] (no whole-layer copy in the reduce dtype). Each element's sum
    is the same as one whole-tensor reduce-scatter's (per element: the same ranks' values, cast once)."""
    rest = tuple(g.shape[1:])
    row = math.prod(rest)
    size = torch.empty((), dtype=c.reduce_dtype).element_size()
    s = max(1, min(c.emax, RS["bytes"] // max(1, c.world * row * size)))
    send = g.new_empty(c.world * s * row, dtype=c.reduce_dtype)
    recv = g.new_empty(s * row, dtype=c.reduce_dtype)
    res = g.new_empty((c.hi - c.lo, *rest), dtype=dtype)
    for j in range(0, c.emax, s):
        n = min(s, c.emax - j)
        buf = send[: c.world * n * row].view(c.world, n, *rest)
        for r, (a, b) in enumerate(c.ranges):
            k = max(0, min(n, b - a - j))
            if k:
                buf[r, :k].copy_(g[a + j: a + j + k])
            if k < n:
                buf[r, k:].zero_()
        out = recv[: n * row].view(n, *rest)
        check(c)
        with c.timed("scatter"):
            dist.reduce_scatter_tensor(out, buf.view(c.world * n, *rest))
        c.stats["scatters"] += 1
        c.stats["scatter_gib"] += buf.numel() * buf.element_size() / 2 ** 30
        m = max(0, min(n, c.hi - c.lo - j))
        if m:
            res[j: j + m].copy_(out[:m])
    return res


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, c):
        ctx.c, ctx.dtype = c, shard.dtype
        with torch.no_grad():
            return gather(shard, c)

    @staticmethod
    def backward(ctx, grad):
        return reduce_scatter(grad, ctx.c, ctx.dtype), None


def install(model, c):
    """Mark the trunk's expert modules as sharded and wrap Qwen4ExpTextExperts.forward once (unmarked modules, e.g.
    the MTP block's, pass through). Works inside or outside qat's wrapper: an own parameter (no outer wrapper ran)
    goes through the QAT grid here, with qat's own wrapper held off for the inner call."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    import qat

    cls = m.Qwen4ExpTextExperts
    if not getattr(cls, "_sft_shard", False):
        inner = cls.forward

        def forward(self, *args, **kwargs):
            c = getattr(self, "_sft_shard_ctx", None)
            if c is None:
                return inner(self, *args, **kwargs)
            kept = {n: self._parameters[n] for n in NAMES}
            own = all(isinstance(kept[n], torch.nn.Parameter) for n in NAMES)
            scales = qat.SCALES.get(id(self)) if own and qat.QAT["on"] else None
            try:
                for n in NAMES:
                    w = qat.ste(kept[n], scales[n]) if scales is not None else kept[n]
                    self._parameters[n] = _Gather.apply(w, c)
                if scales is not None:
                    qat.QAT["calls"] += 1
                    qat.QAT["on"] = False
                return inner(self, *args, **kwargs)
            finally:
                if scales is not None:
                    qat.QAT["on"] = True
                self._parameters.update(kept)

        cls.forward = forward
        cls._sft_shard = True
    for mod in routed(model):
        if mod.gate_up_proj.shape[0] != c.hi - c.lo or mod.num_experts != c.experts:
            raise RuntimeError(f"expert shard {tuple(mod.gate_up_proj.shape)} for range {c.lo}-{c.hi} of {c.experts}")
        mod._sft_shard_ctx = c


# --- dense gradients, clip, optimizer ------------------------------------------------------------------------------

def sync_dense(params, c, bucket=1 << 27):
    """Sum the replicated params' gradients over ranks (fp32 buckets of ``bucket`` elements, back to the param dtype).
    A gradient no rank has stays None (as N=1 would leave it)."""
    if not params:
        return
    dev = params[0].device
    flags = torch.tensor([float(p.grad is not None) for p in params], device=dev)
    dist.all_reduce(flags)
    keep = [p for p, f in zip(params, flags.tolist()) if f > 0]
    i = 0
    while i < len(keep):
        group, n = [], 0
        while i < len(keep) and (not group or n + keep[i].numel() <= bucket):
            group.append(keep[i])
            n += keep[i].numel()
            i += 1
        buf = torch.zeros(n, dtype=torch.float32, device=dev)
        at = 0
        for p in group:
            if p.grad is not None:
                buf[at:at + p.numel()].copy_(p.grad.reshape(-1))
            at += p.numel()
        with c.timed("dense"):
            dist.all_reduce(buf)
        at = 0
        for p in group:
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(buf[at:at + p.numel()].view_as(p))
            at += p.numel()


def clip(rep, shards, max_norm, c):
    """clip_grad_norm_ over the whole model: total = sqrt(sum over ranks of shard norm^2 + replicated norm^2), coef =
    min(1, max_norm / (total + 1e-6)) on every gradient. Returns the total (a 0-d tensor)."""
    def sq(ps):
        gs = [p.grad for p in ps if p.grad is not None]
        if not gs:
            return torch.zeros((), dtype=torch.float32, device=c.comm)
        return torch.stack([torch.linalg.vector_norm(g, 2).float() for g in gs]).pow(2).sum().to(c.comm)

    local = sq(shards)
    dist.all_reduce(local)
    total = (local + sq(rep)).sqrt()
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for p in rep + shards:
        if p.grad is not None:
            p.grad.mul_(coef.to(p.grad.device))
    return total


def seeded_steps(opt, c, seed):
    """Wrap ``opt.step`` (torchao's, or paged.py's): group 0 (replicated) under RNG seed (seed, step), group 1 (shards)
    under (seed, step, rank). Returns the dict whose "step" the trainer sets before each step."""
    inner, box = opt.step, {"step": 0}

    def step(closure=None):
        groups = opt.param_groups
        try:
            for i, g in enumerate(groups):
                torch.manual_seed(mix(seed, box["step"], -1 if i == 0 else c.rank))
                opt.param_groups = [g]
                inner()
        finally:
            opt.param_groups = groups

    opt.step = step
    return box


def replica_sums(params):
    """An int64 per parameter: the sum of its raw bits (bitwise-equal replicas give equal sums)."""
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}
    return torch.stack([p.detach().reshape(-1).view(view[p.element_size()]).sum(dtype=torch.int64) for p in params])


def check_replicas(params, c):
    """Indices of replicated parameters whose bits differ from rank 0's on any rank (the same list on every rank)."""
    s = replica_sums(params).to(c.comm)
    out = torch.empty(c.world * len(params), dtype=torch.int64, device=c.comm)
    dist.all_gather_into_tensor(out, s)
    rows = out.view(c.world, -1)
    return [i for i in range(len(params)) if bool((rows[:, i] != rows[0, i]).any())]


# --- passes --------------------------------------------------------------------------------------------------------

def prefix_passes(cfg, ex):
    """No-grad prefix chunks of a host example (train.loss_plan's cut)."""
    total = int(ex["input_ids"].shape[-1])
    cut = max(0, total - min(int(ex.get("tail") or cfg.window), cfg.window))
    return -(-cut // cfg.prefix_chunk)


DUMMY = {}


def dummy_example(tokens, vocab):
    """A fixed small example (every token a reply token, no prefix) for the alignment passes."""
    key = (tokens, vocab)
    if key not in DUMMY:
        g = torch.Generator().manual_seed(20261029)
        ids = torch.randint(10, max(11, min(vocab, 2000)), (1, tokens), generator=g)
        labels = ids.clone()
        labels[:, 0] = -100
        DUMMY[key] = {"input_ids": ids, "labels": labels, "tail": tokens}
    return DUMMY[key]


# --- load ----------------------------------------------------------------------------------------------------------

def weight_map(folder):
    """{tensor name: absolute file path} of a folder's safetensors index."""
    folder = Path(folder)
    return {k: str(folder / f) for k, f in json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"].items()}


HEADERS = {}
HEADERS_LOCK = threading.Lock()


def header(path):
    """(tensor metadata, data start) of a safetensors file (cached)."""
    with HEADERS_LOCK:
        got = HEADERS.get(path)
    if got is None:
        import struct

        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))
        meta.pop("__metadata__", None)
        got = (meta, 8 + n)
        with HEADERS_LOCK:
            HEADERS[path] = got
    return got


def read_into(path, offset, dst):
    """Bytes [offset, offset + dst bytes) of the file's data section into the contiguous host tensor ``dst``."""
    meta, start = header(path)
    view = memoryview(dst.reshape(-1).view(torch.uint8).numpy())
    got = 0
    with open(path, "rb", buffering=0) as f:
        while got < len(view):
            k = os.preadv(f.fileno(), [view[got:]], start + offset + got)
            if not k:
                raise OSError(f"short read of {path}")
            got += k


def read_rows(path, name, lo, hi, dst):
    """Rows [lo, hi) (dim 0) of tensor ``name`` into ``dst`` (as many elements; its dtype converted when the file's
    differs). ``lo, hi = None, None``: the whole tensor."""
    meta, _ = header(path)
    info = meta[name]
    dtype = ST[info["dtype"]]
    shape = info["shape"]
    per = math.prod(shape[1:]) if len(shape) > 1 else 1
    if lo is None:
        lo, hi, per = 0, 1, math.prod(shape)
    if dst.numel() != (hi - lo) * per:
        raise ValueError(f"{name}: rows {lo}-{hi} of {shape} into {tuple(dst.shape)}")
    size = torch.empty((), dtype=dtype).element_size()
    a = info["data_offsets"][0] + lo * per * size
    if dtype == dst.dtype and dst.is_contiguous():
        read_into(path, a, dst)
    else:
        tmp = torch.empty(dst.shape, dtype=dtype)
        read_into(path, a, tmp)
        dst.copy_(tmp)


def load_experts(model, folder, c, threads=8):
    """This rank's rows of every trunk expert tensor from ``folder``: the fused tensor's rows when the index has it
    (a trainer checkpoint written at N=1 keeps the trained experts fused), else the per-expert keys."""
    from concurrent.futures import ThreadPoolExecutor

    files = weight_map(folder)
    began, nbytes = time.time(), [0]
    lock = threading.Lock()

    def one(item):
        prefix, mod = item
        hosts = {n: torch.empty(tuple(getattr(mod, n).shape), dtype=getattr(mod, n).dtype) for n in NAMES}
        if f"{prefix}.gate_up_proj" in files:
            for n in NAMES:
                read_rows(files[f"{prefix}.{n}"], f"{prefix}.{n}", c.lo, c.hi, hosts[n])
        else:
            half = hosts["gate_up_proj"].shape[1] // 2
            for j, e in enumerate(range(c.lo, c.hi)):
                for proj, dst in (("gate_proj", hosts["gate_up_proj"][j, :half]), ("up_proj", hosts["gate_up_proj"][j, half:]),
                                  ("down_proj", hosts["down_proj"][j])):
                    key = f"{prefix}.{e}.{proj}.weight"
                    if key not in files:
                        raise KeyError(f"{key} not in {folder}'s index")
                    read_rows(files[key], key, None, None, dst)
        with torch.no_grad():
            for n in NAMES:
                getattr(mod, n).copy_(hosts[n])
        with lock:
            nbytes[0] += sum(h.numel() * h.element_size() for h in hosts.values())

    mods = routed_named(model)
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, mods))
    return {"modules": len(mods), "gib": round(nbytes[0] / 2 ** 30, 2), "seconds": round(time.time() - began, 1),
            "range": [c.lo, c.hi]}


@contextlib.contextmanager
def filtered(folder, drop):
    """A temporary folder whose index is ``folder``'s without the names ``drop(name)`` picks (values absolute), with
    its config files: from_pretrained on it loads everything else."""
    folder = Path(folder)
    raw = json.loads((folder / "model.safetensors.index.json").read_text())
    tmp = Path(tempfile.mkdtemp(prefix="sft-shard-load-"))
    try:
        for f in ("config.json", "generation_config.json"):
            if (folder / f).is_file():
                shutil.copyfile(folder / f, tmp / f)
        kept = {k: str(folder / v) for k, v in raw["weight_map"].items() if not drop(k)}
        (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": raw.get("metadata") or {},
                                                                      "weight_map": kept}))
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def trunk_expert_key(name):
    return bool(FUSED.match(name) or PER.match(name))


@contextlib.contextmanager
def building(c, tables=None):
    """While from_pretrained builds the main model: every Qwen4ExpTextExperts gets [hi - lo, ...] parameters (no
    full-size allocation, not even transiently); with ``tables``, every n-gram embedding a one-row stand-in
    (``attach_tables`` puts the shared table in)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    ecls, ncls = m.Qwen4ExpTextExperts, m.Qwen4ExpTextNGramEmbedding
    e_init, n_init = ecls.__init__, ncls.__init__

    def e(self, config, *args, **kwargs):
        e_init(self, config, *args, **kwargs)
        for n in NAMES:
            p = getattr(self, n)
            setattr(self, n, torch.nn.Parameter(torch.empty((c.hi - c.lo, *p.shape[1:]), dtype=p.dtype, device=p.device)))

    def n(self, *args, **kwargs):
        n_init(self, *args, **kwargs)
        emb = self.ngram_embedding
        self._sft_table_shape = tuple(emb.weight.shape)
        self.ngram_embedding = torch.nn.Embedding(1, emb.weight.shape[1], dtype=emb.weight.dtype,
                                                  device=emb.weight.device)

    ecls.__init__ = e
    if tables is not None:
        ncls.__init__ = n
    try:
        yield
    finally:
        ecls.__init__, ncls.__init__ = e_init, n_init


def table_names(folder):
    """{table param name: [checkpoint keys in from_pretrained's concatenation order]}."""
    try:
        from transformers.core_model_loading import dot_natural_key as key
    except ImportError:  # same rule: digits and trailing digits numerically
        def key(s):
            return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]
    groups = {}
    for k in weight_map(folder):
        m = NGRAM.match(k)
        if m:
            groups.setdefault(m.group(1) + ".weight", []).append(k)
    return {t: sorted(ks, key=key) for t, ks in groups.items()}


def load_tables(folder, threads=32, piece=256 << 20):
    """Every n-gram table of ``folder`` in shared host memory (one copy for all rank processes; torch's spawn
    pickling passes the storage, not the bytes): {param name: tensor}."""
    from concurrent.futures import ThreadPoolExecutor

    files = weight_map(folder)
    out, jobs = {}, []
    for target, keys in table_names(folder).items():
        metas = [header(files[k])[0][k] for k in keys]
        rows = sum(m["shape"][0] for m in metas)
        width = metas[0]["shape"][1]
        dtype = ST[metas[0]["dtype"]]
        size = torch.empty((), dtype=dtype).element_size()
        try:
            storage = torch.UntypedStorage._new_shared(rows * width * size)
            t = torch.empty(0, dtype=dtype).set_(storage, 0, (rows, width), (width, 1))
        except Exception:  # noqa: BLE001 - older torch: allocate, then share
            t = torch.empty((rows, width), dtype=dtype).share_memory_()
        at = 0
        for k, m in zip(keys, metas):
            n = m["shape"][0]
            step = max(1, piece // (width * size))
            for a in range(0, n, step):
                jobs.append((files[k], k, a, min(n, a + step), t[at + a: at + min(n, a + step)]))
            at += n
        out[target] = t
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda j: read_rows(j[0], j[1], j[2], j[3], j[4]), jobs))
    return out


def attach_tables(model, tables):
    """Each n-gram embedding's weight becomes its shared table (frozen); shapes checked against the built module."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    done = 0
    for name, mod in model.named_modules():
        if isinstance(mod, m.Qwen4ExpTextNGramEmbedding):
            key = f"{name}.ngram_embedding.weight"
            t = tables[key]
            if tuple(t.shape) != mod._sft_table_shape:
                raise RuntimeError(f"{key}: shared table {tuple(t.shape)}, module {mod._sft_table_shape}")
            mod.ngram_embedding.weight = torch.nn.Parameter(t, requires_grad=False)
            mod.ngram_embedding.num_embeddings = t.shape[0]
            done += 1
    if done != len(tables):
        raise RuntimeError(f"{len(tables)} shared tables, {done} n-gram modules")
    return done


# --- checkpoints ---------------------------------------------------------------------------------------------------

def names_of(model, head):
    """{id(param): checkpoint name} (ckpt.trained_items' names)."""
    got = {id(p): n for n, p in model.named_parameters()}
    got.update({id(p): f"mtp.{n}" for n, p in head.named_parameters()})
    return got


def optim_items(opt, names, keep):
    """(tensors [(name, tensor)], skeleton {"by_name": {param name: {key: spec}}, "param_groups"}) of the states of
    the params whose id is in ``keep`` (ckpt._flatten's specs, tensors named "<param>::<key>...")."""
    import ckpt

    optim, state = [], {}
    for g in opt.param_groups:
        for p in g["params"]:
            st = opt.state.get(p)
            if id(p) not in keep or not st:
                continue
            n = names[id(p)]
            state[n] = {k: (ckpt._flatten(f"{n}::{k}", v, optim) if torch.is_tensor(v) else {"value": v})
                        for k, v in st.items()}
    groups = [{k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in g.items() if k != "params"}
              for g in opt.param_groups]
    return optim, {"by_name": state, "param_groups": groups}


class Part:
    """ckpt.Saver's items and writer for one rank of a sharded run."""

    def __init__(self, c, model, head, rep, shards):
        self.c, self.model, self.head, self.rep, self.shards = c, model, head, rep, shards
        self.names = names_of(model, head)
        self.sharded = [self.names[id(p)] for p in shards]

    def begin(self, out, step):
        """Main thread, every rank: rank 0 makes the empty .ckpt-<step>/student, then all wait for it."""
        if self.c.rank == 0:
            tmp = Path(out) / f".ckpt-{step}"
            shutil.rmtree(tmp, ignore_errors=True)
            (tmp / "student").mkdir(parents=True)
        self.c.barrier()

    def items(self, model, head, opt):
        weights = []
        if self.c.rank == 0:
            weights += sorted((self.names[id(p)], p) for p in self.rep)
        for prefix, mod in routed_named(model):
            gu, dn = mod.gate_up_proj, mod.down_proj
            half = gu.shape[1] // 2
            for j, e in enumerate(range(self.c.lo, self.c.hi)):
                weights += [(f"{prefix}.{e}.gate_proj.weight", gu[j, :half]), (f"{prefix}.{e}.up_proj.weight", gu[j, half:]),
                            (f"{prefix}.{e}.down_proj.weight", dn[j])]
        keep = {id(p) for p in self.shards} | ({id(p) for p in self.rep} if self.c.rank == 0 else set())
        return (weights, *optim_items(opt, self.names, keep))

    def write(self, out, student, step, cursor, weights, optim, skeleton, keep, after_optim=None):
        import ckpt

        r = self.c.rank
        out = Path(out)
        tmp, final = out / f".ckpt-{step}", out / f"ckpt-{step}"
        ckpt.write_safetensors(tmp / f"optim-r{r}.safetensors", optim)
        if after_optim is not None:
            after_optim()
        torch.save(skeleton, tmp / f"optim-r{r}.pt")
        keys, shard, size, part = {}, [], 0, 0
        for i, (name, t) in enumerate(weights):
            shard.append((name, t))
            size += t.numel() * t.element_size()
            if size >= ckpt.SHARD or i == len(weights) - 1:
                part += 1
                file = f"trained-r{r}-{part:05d}.safetensors"
                ckpt.write_safetensors(tmp / "student" / file, shard)
                keys.update({k: file for k, _ in shard})
                shard, size = [], 0
        mark = tmp / f"done-r{r}.json"
        mark.with_suffix(".part").write_text(json.dumps({"keys": keys}))
        mark.with_suffix(".part").rename(mark)
        if r != 0:
            return final
        trained = {}
        deadline = time.time() + float(os.environ.get("SFT_SHARD_CKPT_WAIT_S", 7200))
        for k in range(self.c.world):
            m = tmp / f"done-r{k}.json"
            while not m.is_file():
                if time.time() > deadline:
                    raise TimeoutError(f"checkpoint {step}: rank {k} never finished its part")
                time.sleep(0.5)
            trained.update(json.loads(m.read_text())["keys"])
        base = Path(student)
        raw = json.loads((base / "model.safetensors.index.json").read_text())
        weight_map_ = {k: str(base / f) for k, f in raw["weight_map"].items() if not trunk_expert_key(k)}
        weight_map_.update(trained)
        for f in ("config.json", "generation_config.json"):
            if (base / f).is_file():
                (tmp / "student" / f).write_bytes((base / f).read_bytes())
        (tmp / "student" / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": raw.get("metadata") or {}, "weight_map": weight_map_}))
        (tmp / "shard.json").write_text(json.dumps({"format": "sft-shard-v1", "world": self.c.world,
                                                    "experts": self.c.experts, "ranges": self.c.ranges,
                                                    "sharded": self.sharded}))
        (tmp / "state.json").write_text(json.dumps({"step": step, "cursor": list(cursor), "student": str(base),
                                                    "world": self.c.world}))
        for k in range(self.c.world):
            (tmp / f"done-r{k}.json").unlink()
        tmp.rename(final)
        from ckpt import prune  # pins.json steps stay
        prune(out, keep)
        return final


# --- optimizer state, any N to any N -------------------------------------------------------------------------------

class _Reader:
    """Lazy row reads of the tensors a skeleton names, from one safetensors file."""

    def __init__(self, path):
        self.path = str(path)

    def rows(self, spec, a, b):
        if "inline" in spec:
            return spec["inline"][a:b]
        meta, _ = header(self.path)
        shape = meta[spec["name"]]["shape"]
        dst = torch.empty((b - a, *shape[1:]), dtype=ST[meta[spec["name"]]["dtype"]])
        if b > a:
            read_rows(self.path, spec["name"], a, b, dst)
        return dst

    def whole(self, spec):
        if "inline" in spec:
            return spec["inline"]
        if "name" in spec:
            meta, _ = header(self.path)
            return self.rows(spec, 0, meta[spec["name"]]["shape"][0] if meta[spec["name"]]["shape"] else 1).reshape(
                meta[spec["name"]]["shape"])
        import ckpt

        inner = {k: self.whole(s) for k, s in spec["inner"].items()}
        return ckpt._rebuild({"cls": spec["cls"], "ctx": spec["ctx"], "inner": {k: {"inline": v} for k, v in inner.items()},
                              "size": spec["size"], "stride": spec["stride"]}, {})


def _piece_rows(reader, spec, a, b):
    """Rows [a, b) of a saved state: ("q", codes, scale, qmap, ctx) when 8-bit with block-aligned rows, else ("f",
    fp32 tensor)."""
    if "cls" in spec and spec["cls"].endswith("OptimState8bit"):
        size = spec["size"]
        per = math.prod(size[1:]) if len(size) > 1 else 1
        scale = spec["inner"]["scale"]
        n_scale = scale["inline"].numel() if "inline" in scale else header(reader.path)[0][scale["name"]]["shape"][0]
        bs = math.prod(size) // max(1, n_scale)
        if (a * per) % bs == 0 and (b * per) % bs == 0:
            codes = reader.rows(spec["inner"]["codes"], a, b)
            sc = reader.rows(scale, a * per // bs, b * per // bs) if "inline" not in scale else scale["inline"][a * per // bs: b * per // bs]
            return ("q", codes, sc, reader.whole(spec["inner"]["qmap"]), spec["ctx"], bs)
        return ("f", reader.whole(spec).dequantize().float()[a:b])
    if "cls" in spec:
        raise NotImplementedError(f"re-slicing {spec['cls']} states (8-bit and plain only)")
    return ("f", reader.rows(spec, a, b).float())


def _assemble(parts, p, key, opt):
    """One state tensor for parameter ``p`` from row pieces, in the form torchao would make for ``p`` (8-bit codes and
    scales concatenated when every piece is block-aligned 8-bit, else dequantized and, if 8-bit is due, requantized)."""
    bs = getattr(opt, "block_size", 256)
    want_q = hasattr(opt, "_new_buffer") and p.numel() >= 4096 and p.numel() % bs == 0
    if want_q or any(x[0] == "q" for x in parts):
        from torchao.optim.subclass_8bit import OptimState8bit
    if want_q and all(x[0] == "q" and x[5] == bs for x in parts):
        codes = torch.cat([x[1] for x in parts]).reshape(p.shape)
        signed, dtype = parts[0][4]
        return OptimState8bit(codes, torch.cat([x[2] for x in parts]), parts[0][3].clone(), signed, dtype)
    flat = torch.cat([(OptimState8bit(x[1].reshape(-1), x[2], x[3], *x[4]).dequantize().float().reshape(x[1].shape)
                       if x[0] == "q" else x[1]) for x in parts]).reshape(p.shape)
    if want_q:
        z = OptimState8bit.zeros(p.shape, key == "exp_avg", bs, "cpu", dtype=p.dtype)
        z.copy_(flat)
        return z
    return flat.to(p.dtype)


def sources(folder, n1_names=None, sharded_now=()):
    """{param name: [(lo, hi or None, reader, {key: spec})]} of a checkpoint's optimizer states: a sharded one's
    per-rank files (expert rows lo..hi of the global range), or an N=1 checkpoint's (``n1_names``: its parameter
    order; names in ``sharded_now`` get the full range)."""
    folder = Path(folder)
    out = {}
    if (folder / "shard.json").is_file():
        man = json.loads((folder / "shard.json").read_text())
        sharded = set(man["sharded"])
        for r, (lo, hi) in enumerate(man["ranges"]):
            skel = torch.load(folder / f"optim-r{r}.pt", map_location="cpu", weights_only=False)
            reader = _Reader(folder / f"optim-r{r}.safetensors")
            for n, st in skel["by_name"].items():
                out.setdefault(n, []).append((lo, hi, reader, st) if n in sharded else (None, None, reader, st))
        return out, man["experts"]
    skel = torch.load(folder / "optim.pt", map_location="cpu", weights_only=False)
    reader = _Reader(folder / "optim.safetensors")
    experts = None
    for pid, st in skel["state"].items():
        n = n1_names[int(pid)]
        full = None
        if n in sharded_now:
            v = next(s for s in st.values() if "cls" in s or "name" in s or ("inline" in s and s["inline"].dim()))
            full = v["size"][0] if "size" in v else (header(reader.path)[0][v["name"]]["shape"][0] if "name" in v
                                                      else v["inline"].shape[0])
            experts = full
        out[n] = [(0, full, reader, st) if full is not None else (None, None, reader, st)]
    return out, experts


def load_optimizer(opt, folder, names, ranges, n1_names=None):
    """``opt``'s states from any checkpoint. ``names``: {id(param): name}; ``ranges``: {name: (lo, hi)} of this
    process's expert rows for sharded params (the full range on one GPU). States go to each param's device."""
    src, _ = sources(folder, n1_names, set(ranges))
    for g in opt.param_groups:
        for p in g["params"]:
            n = names[id(p)]
            pieces = src.get(n)
            if not pieces:
                continue
            st = {}
            rng = ranges.get(n)
            for k, spec in pieces[0][3].items():
                if "value" in spec:
                    st[k] = spec["value"]
                    continue
                if rng is None or pieces[0][0] is None or (k == "step") or ("inline" in spec and spec["inline"].dim() == 0):
                    v = pieces[0][2].whole(spec)
                    st[k] = v if k == "step" else v.to(p.device)
                    continue
                lo, hi = rng
                parts = []
                for s_lo, s_hi, reader, sst in sorted(pieces, key=lambda x: x[0]):
                    a, b = max(lo, s_lo), min(hi, s_hi)
                    if a < b:
                        parts.append(_piece_rows(reader, sst[k], a - s_lo, b - s_lo))
                if sum((x[1].shape[0] if x[0] == "q" else x[1].shape[0]) for x in parts) != hi - lo:
                    raise RuntimeError(f"{n}.{k}: checkpoint rows do not cover {lo}-{hi}")
                st[k] = _assemble(parts, p, k, opt).to(p.device)
            opt.state[p] = st


# --- launch --------------------------------------------------------------------------------------------------------

def free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init(rank, world, port, backend, timeout_s=7200):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank))
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world, timeout=timedelta(seconds=timeout_s))


def _orphan_watch(ppid):
    while True:
        time.sleep(5)
        if os.getppid() != ppid:
            os._exit(3)


EXIT = {"destroy_s": 60.0, "join_s": 180.0}  # teardown bounds (tests shorten them)


def _destroy():
    """``destroy_process_group`` with a bound: probe k256-lr-3e6 (29 Oct) hung 12+ min after "done" at 2 NCCL ranks."""
    t = threading.Thread(target=dist.destroy_process_group, daemon=True, name="sft-destroy")
    t.start()
    t.join(EXIT["destroy_s"])
    if t.is_alive():
        print(f"SHARD: destroy_process_group still running after {EXIT['destroy_s']:.0f} s; leaving it", flush=True)


def _rank(rank, world, port, backend, cfg, tables, ppid, make_opt):
    threading.Thread(target=_orphan_watch, args=(ppid,), daemon=True, name="sft-orphan").start()
    init(rank, world, port, backend)
    import train

    kw = {"make_opt": make_opt} if make_opt is not None else {}
    code = 1
    try:
        train.train(cfg, device=f"cuda:{rank}" if backend == "nccl" else "cpu", tables=tables, **kw)
        code = 0
    except BaseException:
        import traceback

        traceback.print_exc()
    finally:
        _destroy()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # no interpreter shutdown: lingering threads / atexit hooks can't keep a finished rank alive


def model_folder(cfg):
    import ckpt

    resume = ckpt.latest(cfg.out) or (Path(cfg.init_ckpt) if cfg.init_ckpt else None)
    return resume / "student" if resume else Path(cfg.student)


def launch(cfg, world, *, backend="nccl", share_table=True, make_opt=None, **kw):
    """``train.train`` on ``world`` processes (rank 0 here, with ``kw``: refresh, commit, guard, knobs); the n-gram
    tables read once into shared memory. A rank that exits with an error stops the whole run."""
    import train

    began = time.time()
    tables = load_tables(model_folder(cfg)) if share_table else None
    if tables is not None:
        print(json.dumps({"event": "shared_tables", "tables": len(tables), "gib": round(sum(
            t.numel() * t.element_size() for t in tables.values()) / 2 ** 30, 2), "seconds": round(time.time() - began, 1)}),
            flush=True)
    port = free_port()
    mp = torch.multiprocessing.get_context("spawn")
    procs = [mp.Process(target=_rank, args=(r, world, port, backend, cfg, tables, os.getpid(), make_opt),
                        name=f"sft-rank{r}") for r in range(1, world)]
    for p in procs:
        p.start()
    stop = threading.Event()

    def watch():
        while not stop.wait(2):
            for p in procs:
                if p.exitcode not in (None, 0):
                    print(f"SHARD: {p.name} exited {p.exitcode}; stopping the run", flush=True)
                    os._exit(1)

    threading.Thread(target=watch, daemon=True, name="sft-rank-watch").start()
    init(0, world, port, backend)
    try:
        if make_opt is not None:
            kw["make_opt"] = make_opt
        train.train(cfg, device="cuda:0" if backend == "nccl" else "cpu", tables=tables, **kw)
    finally:
        stop.set()
    _destroy()
    deadline = time.time() + EXIT["join_s"]
    for p in procs:
        p.join(max(0.0, deadline - time.time()))
    for p in procs:
        if p.is_alive():
            print(f"SHARD: {p.name} still alive {EXIT['join_s']:.0f} s after the run ended; killing it", flush=True)
            p.kill()
            p.join(10)
    bad = [(p.name, p.exitcode) for p in procs if p.exitcode]
    if bad:
        raise RuntimeError(f"ranks failed: {bad}")
