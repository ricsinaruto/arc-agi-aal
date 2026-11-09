"""SFT trainer (trunk/sft): full fine-tune of the pruned Flash student on one GPU.

Plain next-token cross-entropy on recorded reply tokens (prompts masked). Each example is one context (up to the play
window); the tokens before its last ``window`` go through the model without gradients into a cache, and the window is
one forward and backward against that cache (activation checkpointing per decoder layer, the loss in position chunks
so no [T, vocab] tensor is kept). The MTP draft head is trained in the same pass by self-distillation: its inputs
(token embedding, the trunk's pre-mixer state) and the LM head are detached, its target is the trunk's own
next-position distribution (detached), so the trunk's update is the plain SFT one.

Optimizer: 8-bit AdamW (torchao) on BF16 weights with stochastic rounding. Linear warm-up on every start and resume,
then cosine decay to ``final_lr_frac``. Live knobs in ``<out>/knobs.json`` (lr, ckpt tag, eval tag, stop), read every
step. Checkpoints (ckpt.py): every ``ckpt_seconds``, written in the background from a host snapshot.

Speed (all exact unless named): parallel whole-file reads for the load (``fast_load``); a warm-up window of the
largest example's shape before step 1 (compile and autotune land at boot; its peak is the run's) and torchao's
per-shape optimizer compile on scratch tensors (``warm_start``); the next steps' files read ahead and each example
staged one ahead on a side stream with its positions and image features (feed.py: ``prefetch``, ``vision_ahead``);
the full-attention fast path (attention.py, ``fast_attention``; ``sel_cache``); off by default and not bitwise:
the fused GDN layer (gdn.py, ``gdn_fast``) and hyper-connections (hyper.py, ``hyper_fast``). Events in
``events.jsonl`` (one ``boot`` row with each start-up phase's seconds); watch.py rows in ``metrics.jsonl``.

N GPUs (shard.py, ``shard.launch``): routed experts sharded on the expert dim, the rest replicated; the same global
batches, loss scale and clip as one GPU; rank 0 writes events, metrics and the checkpoint's shared files.
"""

import contextlib
import dataclasses
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import attention
import ckpt
import feed
import modelio
import mtp as mtp_mod
import ple
import qat
import shard
from feed import load_example, prepare  # noqa: F401 - the trainer's example API

FROZEN = ("ngram_embedding.", ".visual.", "model.visual")
latest = ckpt.latest


@dataclasses.dataclass
class Config:
    student: str  # base student folder (HF format, mtp.* included)
    data: str  # build.py output: <data>/train, <data>/heldout
    out: str
    heldout_data: str = ""  # the eval set's build.py root (its index's held-out files, teacher files under it); "": data
    exclude_pairs: list = dataclasses.field(default_factory=list)  # [run, game] pairs left out of train and eval (bar)
    exclude_files: str = ""  # JSON list of train example files left out (a continuation leg: the files an earlier leg trained)
    lr: float = 1e-5
    warmup: int = 10
    rewarm: int = 3  # warm-up steps after a resume
    total_steps: int = 200
    final_lr_frac: float = 0.1
    decay: str = "cosine"  # "linear0": linear to 0 over this run's steps (decay-tail probes); "linear_end": linear
    # from lr to lr x final_lr_frac over this run's steps (cosine's frac counts from step 0, so a resume starts late)
    batch_reply_tokens: int = 250_000
    window: int = 65_536
    prefix_chunk: int = 8_192
    loss_chunk: int = 2_048
    mtp_weight: float = 1.0
    grad_clip: float = 1.0
    ckpt_steps: int = 0  # 0: by time only
    ckpt_pin: list = dataclasses.field(default_factory=list)  # steps always saved and never rotated out (out/pins.json)
    ckpt_seconds: int = 3_600
    keep_ckpts: int = 2
    eval_steps: int = 10
    eval_at: list = dataclasses.field(default_factory=list)  # extra slice evals at these steps (probes off the 24-step grid)
    # eval-only screens: after the resume / step-0 slice, the slice again per variant {"name", "kv": None | "kv4" |
    # "kv2" | "kv2kc" (attention.kv_fake), "levels": 16 | 8, "zero": 8 | 4, "group": 128 | 32 (the expert grid),
    # "top_k": routed experts per token (every TopKRouter), "idx_fp8": the QSA indexer's compressed keys through e4m3,
    # "dense_int8": the served W8A16 dense families (per-output-channel amax / 127) on lm_head, GDN in_proj_qkv/z,
    # QSA q/k/v and attention out projections}
    eval_variants: list = dataclasses.field(default_factory=list)
    top_k: int = 0  # >0: routed experts per token of every trunk TopKRouter for the whole run (train and evals); 0: config's
    # eval-only diagnostics: after eval0, each set {"name", "part": "seen" |
    # "unseen", "n"}: n training files spread evenly over that part of the resumed epoch's order (before / after the
    # cursor), read like the slice (event set "train:<name>"); eval_dump: every eval-only read also writes its per-row KL
    # parts, positions and token ids to <out>/diag-<set>.pt
    eval_sets: list = dataclasses.field(default_factory=list)
    eval_dump: bool = False
    # eval-only: {"n": examples, "games": [...] (held-out games to draw from; []
    # = any), "skip": held-out examples skipped first (default eval_examples: none of the slice)}: per routed-expert
    # channel mask-Fisher / attribution over those examples, saved to <out>/channel-fisher.pt (channel_fisher)
    channel_fisher: dict = dataclasses.field(default_factory=dict)
    # step-boundary volume reload: off (28 Oct restart: a reload swapped the mount's view while the loader opened the
    # next example, ENOENT on a file that exists; data is static and knobs come through the parent's read bridge)
    boundary_reload: bool = False
    eval_examples: int = 64  # the eval set's first N held-out examples, every ``eval_steps`` (kept in host RAM)
    eval_full_examples: int = 0  # a second, larger eval set (the first N), read from disk at ``eval_full_steps`` only
    eval_full_steps: list = dataclasses.field(default_factory=list)  # e.g. [0, 116, 231]; [] = no full eval
    eval_full_lead: int = 10  # the full set's extra examples load in the background this many steps ahead (0: at the eval)
    arc2_heldout_data: str = ""  # ARC-2 held-out slice (/runs/sft-data/arc2-heldout-v1): its held-out files, teacher under it
    arc2_heldout_steps: list = dataclasses.field(default_factory=list)  # read from disk at these steps only, set "arc2"
    arc2_heldout_every: int = 0  # also every N steps (0: off); live knob "arc2_heldout_every" in knobs.json
    seed: int = 0
    repeat_examples: int = 0  # >0: train only on epoch 0's first N examples, over and over (memorization read)
    curriculum: bool = False  # train in <data>/priority.json's order (build.py curriculum), the same every epoch, no shuffle
    decay_start: int = 0  # decay="linear_end": the step the decay starts at (0: the run's first)
    init_ckpt: str = ""  # with no checkpoint in ``out``: resume from this one (weights, optimizer, step, cursor)
    eval_resume: bool = False  # slice eval at the resumed step before the first update
    init_fresh_data: bool = False  # init_ckpt on another data pool: its cursor starts at (0, 0), not the checkpoint's
    sr_seed: int = -1  # torch RNG (stochastic rounding) seed; -1: ``seed`` (data order stays on ``seed``)
    fast_attention: bool = True
    async_ckpt: bool = True
    prefetch: int = 1  # steps of files read ahead (0: inline)
    vision_ahead: bool = True
    warm_start: bool = True
    fast_load: bool = True
    sel_cache: bool = False  # ~0.4 GB per full-attention layer at a 48K window
    gdn_fast: bool = False
    gdn_keep: bool = False
    indexer_tf32: bool = False  # QSA indexer scores on TF32 tensor cores, kept only if >= 99.5% of picks match fp32
    qat_refresh_every: int = 1  # fake_quant scales recomputed every N optimizer steps (1: every step)
    qat_drift: bool = False  # at each refresh, log the stale scales' code flips vs the fresh ones (qat.refresh drift)
    keep_layers: int = 0  # the last N trunk decoder layers keep their activations (no checkpoint recompute)
    hyper_fast: bool = False
    metrics_seconds: float = 30.0
    loss: str = "ce"  # "ce": the recorded replies; "kl": KL to the teacher's top-k (teacher.py files in <data>/teacher)
    teacher_k: int = 64  # the teacher files' k (the warm-up's dummy targets)
    # depth mix: [[from_position, weight], ...] ascending; a reply turn's KL rows weigh the
    # weight of its start position (= its served prompt length in a token-exact example); training only, [] = off
    depth_weights: list = dataclasses.field(default_factory=list)
    fake_quant: bool = False  # routed experts forward through the export's int4 grid (qat.py)
    qat_group: int = 128  # fake_quant's input-dim group (32: a channel-cut expert whose width is not a multiple of 128)
    qat_grid: str = "int4"  # fake_quant's grid: "int4" (export.py's) | "int3served" (the served int3 experts, qat.fake3)
    dense_int8_train: bool = False  # the served INT8 dense families forward on their INT8 round trip, straight-through
    mtp_int8_head: bool = False  # with dense_int8_train: the MTP draft's logits on the INT8 lm_head too (v9b: its head has the trunk's shape)
    kv_train: str = ""  # "kv4": QSA K / V forward as the served int4 cache stores them, straight-through (attention.py)
    ple_table: bool = True  # the n-gram rows of each example looked up once on the host (ple.py), not per call
    union_attention: bool = False  # sparse2's query-union selected-key kernel (checked against sparse.py at load)
    qat_fused: bool = True  # fake_quant's grid in one compiled pass (kept only if bit-equal to the eager grid)
    paged_depth: int = 2  # parameters whose states are copied in ahead of the one stepping
    kahan: bool = False  # paged only: BF16 weight + BF16 compensation state instead of stochastic rounding
    paged_states: bool = False  # optimizer states in pinned host memory, paged per parameter in the step (paged.py)
    optim_bits: int = 8  # AdamW state bits (torchao 8 or 4); 4 only for util/MFU tests (states 33 vs 66 GiB)
    mem_budget_gib: float = 256.0  # boot stops when the predicted step peak (warm-up + grads + optimizer) exceeds it
    util_every: float = 1.0  # seconds between nvidia-smi samples (step util / VRAM, metrics rows)
    profile: bool = False  # before step 1: one example of the first batch under torch.profiler -> <out>/profile.json
    stop_step: int = 0  # end after this step (lr probes: the schedule of ``total_steps``, cut short); 0: total_steps
    stop_ckpt: bool = True  # checkpoint at ``stop_step`` (probes: off)
    live_batch: bool = False  # knob "batch_reply_tokens": takes effect past the batches already read ahead (BatchPlan)
    eval_trunc_k: int = 0  # kl loss: held-out reads also report the serving screen's top-k truncated KL (qat.kl_trunc)
    # N GPUs (shard.py; one GPU ignores these): the expert gradients' reduce-scatter dtype ("fp32" | "bf16"), the
    # replicas' bitwise check every N steps (0: off), the alignment passes' length in tokens
    reduce_dtype: str = "fp32"
    replica_check_every: int = 10
    dummy_tokens: int = 128

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)


EVENTS = threading.Lock()  # the checkpoint writer logs from its thread
# Local spool (28 Oct p75 lr probes: the main loop waited 205-240 s on a shared-volume commit while 3 runs and 2
# preflights committed): events.jsonl and metrics.jsonl are appended to a local file; the boundary and metrics threads
# copy the new lines to <out> on the volume (spool_flush) and commit. The training loop never opens a volume file.
SPOOL = {}
WRITER = {"on": True}  # events and metrics rows: rank 0 of a sharded run only
DTYPE = {"model": torch.bfloat16}  # tests: an FP32 model


def spool_open(out):
    import tempfile

    SPOOL[str(out)] = {"local": Path(tempfile.mkdtemp(prefix="sft-spool-")), "lock": threading.Lock(), "at": {}}


def spool_path(out, name):
    s = SPOOL.get(str(out))
    return (s["local"] if s else Path(out)) / name


def spool_flush(out):
    """Append the spooled lines the volume copy lacks (whole lines only); never called from the training loop."""
    s = SPOOL.get(str(out))
    if not s:
        return
    with s["lock"]:
        for src in sorted(s["local"].glob("*.jsonl")):
            at = s["at"].get(src.name, 0)
            with open(src, "rb") as f:
                f.seek(at)
                data = f.read()
            data = data[:data.rfind(b"\n") + 1]
            if data:
                with open(Path(out) / src.name, "ab") as f:
                    f.write(data)
                s["at"][src.name] = at + len(data)


def event(out, **row):
    if not WRITER["on"]:
        return
    row = {"t": round(time.time(), 1), **row}
    print(json.dumps(row), flush=True)
    with EVENTS, open(spool_path(out, "events.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# --- model ---------------------------------------------------------------------------------------------------------

LOAD = {}


def load_model(path, device, fast_attention=True, *, fast_load=False, sel_cache=False, gdn_fast=False, gdn_keep=False,
               hyper_fast=False, union=False, ctx=None, tables=None):
    """The student in BF16 on ``device`` (the n-gram table stays in host RAM, frozen with the vision tower) and its
    MTP head; every other weight trainable. The fast paths are CUDA-only. ``ctx`` (shard.Ctx): only this rank's
    expert rows are read and held; ``tables``: the n-gram tables already in shared host memory (shard.launch)."""
    from transformers import AutoModelForMultimodalLM
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    modelio.table_in_host_ram()
    cuda = str(device).startswith("cuda")
    if fast_attention and cuda:
        attention.install(sel_cache=sel_cache)
        if union:
            ok, errs = attention.union_check(device)
            attention.SPARSE["union"] = ok
            LOAD["union_check"] = {"ok": ok, "rel_err_out_dq_dk_dv": errs}
    if gdn_fast and cuda:
        import gdn

        gdn.install()
        gdn.GDN.update(fast=True, keep=bool(gdn_keep))
    if hyper_fast and cuda:
        import hyper

        hyper.install()
        hyper.HYPER["on"] = True
    source = contextlib.nullcontext(path)
    if ctx is not None:  # from_pretrained without the trunk experts (and the shared tables); shard-sized experts
        source = shard.filtered(path, lambda k: shard.trunk_expert_key(k)
                                or (tables is not None and bool(shard.NGRAM.match(k))))
    began = time.time()
    with source as folder, (shard.building(ctx, tables) if ctx is not None else contextlib.nullcontext()):
        reads = (modelio.parallel_reads(folder, device=device if cuda else None) if fast_load
                 else contextlib.nullcontext({"mode": "mmap"}))
        with reads as how:
            model = AutoModelForMultimodalLM.from_pretrained(str(folder), dtype=DTYPE["model"], device_map={"": device},
                                                             experts_implementation="grouped_mm" if cuda else "eager")
    # read_* from parallel_reads; the rest of from_pretrained (build, cast, host -> GPU) is total - read_wait_s
    LOAD.update(how, from_pretrained_s=round(time.time() - began, 1))
    if ctx is not None:
        LOAD["experts"] = shard.load_experts(model, path, ctx)
        if tables is not None:
            LOAD["shared_tables"] = shard.attach_tables(model, tables)
        shard.install(model, ctx)
    if fast_load:
        t = time.time()
        LOAD.update(compacted=modelio.compact(model), compact_s=round(time.time() - t, 1))
    began = time.time()
    modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True  # the window pass only reads the cache
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for name, p in model.named_parameters():
        p.requires_grad_(not any(k in name for k in FROZEN))
    head = mtp_mod.load(mtp_mod.head_for(model.config.text_config, path), path, device=device, dtype=DTYPE["model"])
    LOAD["mtp_s"] = round(time.time() - began, 1)  # the MTP block: safetensors reads of its own (mmap)
    return model, head


def trainable(model, head):
    return [p for p in model.parameters() if p.requires_grad] + list(head.parameters())


# --- loss ----------------------------------------------------------------------------------------------------------

def _ce(h, weight, target):
    return F.cross_entropy((h @ weight.t()).float(), target, ignore_index=-100, reduction="sum")


TRUNC = {"k": 0, "kl": None, "top1": None}  # held-out reads only (eval_trunc_k): the serving screen's top-k KL
# eval_dump reads: per-row KL parts (qat.kl_rows) with each row's target position ("pos", set by example_loss) and
# each example's token ids
DIAG = {"on": False, "pos": None, "rows": [], "ids": []}


def _distill(h, weight, target, ids, lp, w=None):
    """(CE sum, teacher KL sum) of rows h; the CE detached (reported, not trained); ``w``: per-row KL weights."""
    logits = (h @ weight.t()).float()
    if DIAG["on"] and not torch.is_grad_enabled():
        r = {k: v.cpu() for k, v in qat.kl_rows(logits, ids, lp, target).items()}
        DIAG["rows"].append({**r, "pos": DIAG["pos"], "target": target.cpu(), "ex": len(DIAG["ids"]) - 1})
    if TRUNC["k"] and not torch.is_grad_enabled():
        kt, t1 = qat.kl_trunc(logits, ids, lp, target, TRUNC["k"])
        TRUNC["kl"] = kt if TRUNC["kl"] is None else TRUNC["kl"] + kt
        TRUNC["top1"] = t1 if TRUNC["top1"] is None else TRUNC["top1"] + t1
    return F.cross_entropy(logits.detach(), target, reduction="sum"), qat.kl_topk(logits, ids, lp, w)


def _kl(h_trunk, h_draft, weight):
    p = F.log_softmax((h_trunk @ weight.t()).float(), -1)
    q = F.log_softmax((h_draft @ weight.t()).float(), -1)
    return (p.exp() * (p - q)).sum(), (p.argmax(-1) == q.argmax(-1)).sum()


def loss_plan(cfg, model, ex, device, cuda):
    """What an example's pass needs to know from its labels, worked out on the host (the staging thread) so the pass
    reads nothing back (sft-heal's HEAL_LAZY_STATS): the window cut, the reply rows of each loss chunk and MTP chunk
    (index tensors moved ahead), the teacher rows of each chunk (checked against the window's reply positions), the
    reply-token and draft-row counts, and the PLE rows of the whole sequence (``ple_table``, pinned)."""
    ids, labels = ex["input_ids"], ex["labels"]  # host [1, T]
    total = ids.shape[1]
    cut = max(0, total - min(int(ex.get("tail") or cfg.window), cfg.window))  # build.py's gradient tail
    width = total - cut
    target = labels[0, cut + 1:]  # row i predicts token cut + i + 1
    m = target != -100
    dev = lambda t: feed.move(t, device, cuda)  # noqa: E731
    out = {"cut": cut, "n": int(m.sum()), "ce": [], "at": [], "mtp": [], "rows": 0, "ple": None, "w": [], "wsum": 0.0}
    rw = depth_row_weights(cfg.depth_weights, labels[0], cut) if cfg.depth_weights else None
    if cfg.loss == "kl":  # the teacher's rows of this window's reply tokens, in order
        tpos, want = ex["teacher"]["pos"].long(), m.nonzero()[:, 0] + cut + 1
        at = torch.searchsorted(tpos, want).clamp(max=max(0, len(tpos) - 1))
        if len(want) and (not len(tpos) or not torch.equal(tpos[at], want)):
            raise ValueError("teacher positions do not cover the window's reply tokens")
    used = 0
    for a in range(0, width - 1, cfg.loss_chunk):
        b = min(a + cfg.loss_chunk, width - 1)
        idx = m[a:b].nonzero()[:, 0]
        if len(idx):
            out["ce"].append((a, b, dev(idx)))
            if cfg.loss == "kl":
                out["at"].append(dev(at[used:used + len(idx)]))
                used += len(idx)
            if rw is not None:
                w = rw[a:b][idx]
                out["w"].append(dev(w))
                out["wsum"] += float(w.sum())
    if width > 2:
        reply = target[1:] != -100  # draft row k's token j + 1 = cut + k + 2 is a reply token
        for a in range(0, width - 2, cfg.loss_chunk):
            b = min(a + cfg.loss_chunk, width - 2)
            idx = reply[a:b].nonzero()[:, 0]
            if len(idx):
                out["mtp"].append((a, b, dev(idx)))
                out["rows"] += len(idx)
    if cfg.ple_table:
        out["ple"] = ple.rows(model, ids, pin=cuda)
    return out


def check_depth_weights(weights, loss):
    if not weights:
        return
    starts = [float(a) for a, _ in weights]
    if loss != "kl" or starts[0] != 0 or sorted(set(starts)) != starts or any(float(w) <= 0 for _, w in weights):
        raise ValueError(f"depth_weights {weights}: loss kl, [[0, w0], [p1, w1], ...] strictly ascending, weights > 0")


def depth_row_weights(weights, labels, cut):
    """Per-row loss weights [T - cut - 1] of a window (row i predicts token cut + i + 1): each reply run (consecutive
    labelled tokens) takes the weight of the bucket holding its first token's position, i.e. the turn's prompt length;
    prompt rows 1.0 (unused). ``weights``: [[from_position, weight], ...] ascending from 0."""
    edges = torch.tensor([float(a) for a, _ in weights])
    vals = torch.tensor([float(w) for _, w in weights])
    lab = labels != -100
    pos = torch.arange(len(labels))
    start = lab & ~torch.cat([torch.zeros(1, dtype=torch.bool), lab[:-1]])  # first token of each reply run
    run_start = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
    w = vals[(torch.searchsorted(edges, run_start.float(), right=True) - 1).clamp(min=0)]
    return torch.where(lab, w, torch.ones_like(w))[cut + 1:]


def host(*xs):
    """Floats of card scalars in one read."""
    ts = [x for x in xs if torch.is_tensor(x)]
    if not ts:
        return [float(x) for x in xs]
    got = iter(torch.stack([x.detach().double().reshape(()) for x in ts]).tolist())
    return [next(got) if torch.is_tensor(x) else float(x) for x in xs]


EMPTY = {"n": 0}  # training examples with no reply row in their window, whole run


def example_loss(model, head, box, ex, cfg, *, scale=1.0, train=True):
    """One context (a build.py example, or one feed.stage made with ``loss_plan``): (CE sum, reply tokens, MTP KL sum,
    draft hits, rows, teacher KL sum), the sums as card scalars (no read back), and, when ``train``, the backward of
    ``scale`` x (CE or teacher KL + mtp_weight x MTP KL)."""
    from transformers import DynamicCache

    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    made = feed.ready(ex) if "piece" in ex else feed.stage(
        model, ex, device, vision=False, plan=lambda e: loss_plan(cfg, model, e, device, cuda))
    ex, plan = made["ex"], made["plan"]
    piece = made["piece"] or modelio.positions_and_images(model, ex, host_index=True)
    ids, labels = ex["input_ids"], ex["labels"]
    total, cut = ids.shape[1], plan["cut"]
    core, embed, weight = model.model, model.model.language_model.embed_tokens, model.lm_head.weight
    w_mtp = weight.detach()  # the MTP draft's logits: BF16 lm_head (k256: its arena16k head is not an INT8 shape)
    if qat.DENSE["on"]:  # the served INT8 lm_head for the trunk's logits, straight-through
        weight = qat.int8_ste(weight)
        w_mtp = weight.detach() if cfg.mtp_int8_head else w_mtp
    cache, mcache = DynamicCache(config=model.config.text_config), head.new_cache()
    model.eval()
    try:
        with torch.no_grad(), torch.profiler.record_function("sft:prefix"):
            for a in range(0, cut, cfg.prefix_chunk):
                b = min(a + cfg.prefix_chunk, cut)
                attention.reset()
                ple.ROWS["now"] = ple.piece(plan["ple"], a, b, device)
                core(input_ids=ids[:, a:b], attention_mask=torch.ones_like(ids[:, :b]), past_key_values=cache,
                     use_cache=True, **piece(a, b))
                ple.ROWS["now"] = None
                got = mtp_mod.inputs(ids, a, b, piece)
                if got:
                    ja, jb, cur, pos = got
                    attention.reset()
                    head(embed(cur), box["hc"][:, : jb - ja], pos, mcache)
        model.train(train)
        attention.window(train)
        with torch.set_grad_enabled(train):
            attention.reset()
            ple.ROWS["now"] = ple.piece(plan["ple"], cut, total, device)  # kept for the backward's replays
            with torch.profiler.record_function("sft:window_fwd"):
                hidden = core(input_ids=ids[:, cut:], attention_mask=torch.ones_like(ids),
                              past_key_values=modelio.read_only(cache) if cut else None, use_cache=False,
                              **piece(cut, total)).last_hidden_state[0]
            hc = box["hc"].detach()
            width = total - cut
            target = labels[0, cut + 1:]  # row i predicts token cut + i + 1
            ce = hidden.new_zeros((), dtype=torch.float32)
            tkl = hidden.new_zeros((), dtype=torch.float32)
            if DIAG["on"] and not train:
                DIAG["ids"].append(ids[0].cpu())
            for i, (a, b, idx) in enumerate(plan["ce"]):
                if cfg.loss == "kl":
                    t, sel = ex["teacher"], plan["at"][i]
                    if DIAG["on"] and not train:
                        DIAG["pos"] = (cut + 1 + a + idx).cpu()  # the absolute position of each row's target token
                    args = (hidden[a:b][idx], weight, target[a:b][idx], t["ids"][sel], t["lp"][sel])
                    if train and plan.get("w"):
                        args = args + (plan["w"][i],)
                    c, k = checkpoint(_distill, *args, use_reentrant=False) if train else _distill(*args)
                    ce, tkl = ce + c, tkl + k
                else:  # the reply rows only (prompt rows add 0): the KL path's logits shape
                    args = (hidden[a:b][idx], weight, target[a:b][idx])
                    ce = ce + (checkpoint(_ce, *args, use_reentrant=False) if train else _ce(*args))
            # MTP index j = cut + 1 + k (token j, trunk state j - 1) drafts token j + 1, as the trunk's row k + 1 does
            kl, hits = hidden.new_zeros((), dtype=torch.float32), hidden.new_zeros((), dtype=torch.long)
            if width > 2:
                pos = piece(cut + 1, total)["position_ids"]
                attention.reset()
                draft = head(embed(ids[:, cut + 1:]).detach(), hc[:, : width - 1], pos,
                             modelio.read_only(mcache) if cut else None, checkpointed=train)[0]
                trunk, w = hidden.detach(), w_mtp
                for a, b, idx in plan["mtp"]:
                    args = (trunk[a + 1:b + 1][idx], draft[a:b][idx], w)
                    k, h = checkpoint(_kl, *args, use_reentrant=False) if train else _kl(*args)
                    kl, hits = kl + k, hits + h
            if train:
                main = tkl if cfg.loss == "kl" else ce
                total = (main + cfg.mtp_weight * kl) * scale
                if not (plan["ce"] or plan["mtp"]):  # no reply row in the window (capped labels cleared): a zero
                    EMPTY["n"] += 1  # backward, so every collective of the pass still runs on this rank
                    total = hidden.float().mean() * 0.0
                    if width > 2:
                        total = total + draft.float().mean() * 0.0
                with torch.profiler.record_function("sft:backward"):
                    total.backward()
    finally:
        attention.window(False)
        ple.ROWS["now"] = None
    return ce, plan["n"], kl, hits, plan["rows"], tkl


def host_peak_gib():
    """This process's peak resident host memory (GiB; shared n-gram table pages count once touched)."""
    import resource
    import sys

    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)


def dummy_pass(cfg, model, head, box, train):
    """One alignment pass (shard.py): a small fixed example, no prefix, its loss scaled by 0 (train: a gradient window
    whose gradients are exact zeros; else one no-grad pass)."""
    tokens = min(cfg.dummy_tokens, cfg.window)
    ex = shard.dummy_example(tokens, int(model.config.text_config.vocab_size))
    example_loss(model, head, box, ex, dataclasses.replace(cfg, loss="ce"), scale=0.0, train=train)


def aligned(cfg, model, head, box, exs, stager, *, train, scale=1.0):
    """This rank's examples with every rank issuing the same collective sequence (shard.py): training, round by round
    (each rank's i-th example; P = the round's most prefix chunks): P - own dummy no-grad passes, then the example
    (a dummy window when the rank has none); an eval, the examples then dummy no-grad passes up to the largest
    rank's pass count. Returns example_loss's results of the real examples."""
    c = shard.DD["ctx"]
    counts = [shard.prefix_passes(cfg, e["ex"] if "piece" in e else e) for e in exs]
    out = []
    if train:
        it = stager.run(exs)
        for i, most in enumerate(c.rounds(counts)):
            mine = counts[i] if i < len(counts) else None
            for _ in range(most - (mine or 0)):
                dummy_pass(cfg, model, head, box, False)
            if mine is None:
                dummy_pass(cfg, model, head, box, True)
            else:
                out.append(example_loss(model, head, box, next(it), cfg, scale=scale))
        return out
    passes = sum(k + 1 for k in counts)
    most = c.max_int(passes)  # agreed before any pass (a rank still gathering cannot meet one in an all-reduce)
    for made in stager.run(exs):
        out.append(example_loss(model, head, box, made, cfg, train=False))
    for _ in range(most - passes):
        dummy_pass(cfg, model, head, box, False)
    return out


# --- data, schedule ------------------------------------------------------------------------------------------------

def schedule_start(cfg, out, resume, step):
    """The step the lr schedule counts from: this run's own first step. A resume from the run's own checkpoint (a
    preemption or relaunch: joint c942 2 Nov re-warmed 15 steps at 1060) keeps it, so no warm-up repeats and a decay
    goes on where it was; a new leg (init_ckpt, or no checkpoint) starts it at ``step`` (warm-up / ``rewarm``)."""
    path = Path(out) / "schedule.json"
    own = resume is not None and Path(resume).parent == Path(out)
    if own and path.is_file():
        return int(json.loads(path.read_text())["start"])
    if own:  # a run from before schedule.json: its init checkpoint's step, else 0
        init = Path(cfg.init_ckpt or "/nonexistent") / "state.json"
        return int(json.loads(init.read_text())["step"]) if init.is_file() else 0
    if WRITER["on"]:
        path.write_text(json.dumps({"start": step}))
    return step


def schedule(cfg, step, start):
    """lr at ``step``: warm-up over ``warmup`` steps from 0 (``rewarm`` after a resume at ``start``), cosine to
    ``final_lr_frac`` at ``total_steps``. ``decay="linear0"``: linear from ``lr`` at ``start`` to 0 after the last
    step (``stop_step`` or ``total_steps``). ``decay="linear_end"``: linear from ``lr`` at ``start`` to ``lr`` x
    ``final_lr_frac`` on the last step."""
    ramp = cfg.warmup if start == 0 else cfg.rewarm
    warm = min(1.0, (step - start + 1) / max(1, ramp))
    if cfg.decay == "linear0":
        last = min(cfg.stop_step or cfg.total_steps, cfg.total_steps)
        return cfg.lr * warm * max(0.0, (last - step) / max(1, last - start))
    if cfg.decay == "linear_end":
        last = min(cfg.stop_step or cfg.total_steps, cfg.total_steps) - 1
        begin = max(start, cfg.decay_start)  # held at ``lr`` until then
        return cfg.lr * warm * (1 - (1 - cfg.final_lr_frac) * max(0.0, min(1.0, (step - begin) / max(1, last - begin))))
    frac = min(1.0, step / max(1, cfg.total_steps))
    return cfg.lr * warm * (cfg.final_lr_frac + (1 - cfg.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def order(index, seed, epoch):
    """Shuffled example files; epoch 0 starts with the longest gradient window, so step 1 meets the peak memory.
    A curriculum index (``index["order"]``) keeps its fixed order instead."""
    if index.get("order") is not None:
        return list(index["order"])
    names = [e["file"] for e in index["examples"]]
    random.Random(seed * 1_000_003 + epoch).shuffle(names)
    if epoch == 0:
        big = max(index["examples"], key=lambda e: min(e.get("tail") or e.get("tokens") or 0, e.get("tokens") or 1e18))["file"]
        names.remove(big)
        names.insert(0, big)
    return names


def pool(cfg, index, epoch):
    """The epoch's example order; ``repeat_examples`` N: only epoch 0's first N files, reshuffled every epoch (a
    capacity / memorization read: every step trains on the same few examples)."""
    names = order(index, cfg.seed, epoch)
    if cfg.repeat_examples:
        fixed = set(order(index, cfg.seed, 0)[:cfg.repeat_examples])
        names = [n for n in names if n in fixed]
    return names


class BatchPlan:
    """The reply-token goal of each step's batch. A sharded run's ranks each read ahead on their own thread, so a knob
    never changes a batch any rank may already hold: it names the first step it applies to, the same on every rank."""

    def __init__(self, tokens, step):
        self.changes, self.step = [(0, tokens)], step

    def set(self, tokens, first):
        self.changes.append((first, tokens))

    def take(self):
        """The next batch's goal, in step order (the loader's thread)."""
        self.step += 1
        return next(t for first, t in reversed(self.changes) if first <= self.step)


def next_batch(cfg, index, cursor, goal=None):
    """Examples for one step (whole examples until ``goal``, default ``batch_reply_tokens``, reply tokens) and the new
    cursor."""
    by_file = {e["file"]: e for e in index["examples"]}
    epoch, pos = cursor
    names, picked, tokens = pool(cfg, index, epoch), [], 0
    while tokens < (goal or cfg.batch_reply_tokens):
        if pos >= len(names):
            epoch, pos, names = epoch + 1, 0, pool(cfg, index, epoch + 1)
        picked.append(names[pos])
        tokens += by_file[names[pos]]["reply_tokens"]
        pos += 1
    return picked, tokens, (epoch, pos)


# --- eval ----------------------------------------------------------------------------------------------------------

def evaluate(cfg, model, head, box, examples, stager):
    ce = n = kl = hits = rows = tkl = 0
    TRUNC.update(k=cfg.eval_trunc_k if cfg.loss == "kl" else 0, kl=None, top1=None)
    c = shard.DD["ctx"]
    try:
        if c is None:
            got = (example_loss(model, head, box, made, cfg, train=False) for made in stager.run(examples))
        else:  # this rank's share; every sum below is the ranks' total
            got = aligned(cfg, model, head, box, examples, stager, train=False)
        for c_, k, m, h, r, d in got:
            ce, n, kl, hits, rows, tkl = ce + c_, n + k, kl + m, hits + h, rows + r, tkl + d
    finally:
        TRUNC["k"] = 0
    ce, kl, hits, tkl = host(ce, kl, hits, tkl)
    if c is not None:
        ce, n, kl, hits, rows, tkl = c.sums([ce, n, kl, hits, rows, tkl])
        n, rows = int(n), int(rows)
        if cfg.loss == "kl" and cfg.eval_trunc_k:
            TRUNC["kl"], TRUNC["top1"] = c.sums(host(TRUNC["kl"] if TRUNC["kl"] is not None else 0.0,
                                                     TRUNC["top1"] if TRUNC["top1"] is not None else 0.0))
    a = hits / max(1, rows)
    out = {"heldout_loss": ce / max(1, n), "heldout_kl": tkl / max(1, n), "heldout_tokens": n, "mtp_kl": kl / max(1, rows), "accept1": a,
           "accept_len3": a + a * a + a ** 3}  # mean accepted length over 3 drafts, independence approximation
    if TRUNC["kl"] is not None:
        kt, t1 = host(TRUNC["kl"], TRUNC["top1"])
        out.update({f"heldout_kl_top{cfg.eval_trunc_k}": kt / max(1, n), "heldout_top1": t1 / max(1, n)})
    return out


def channel_fisher(cfg, model, head, box, examples, stager):
    """Per routed expert's intermediate channel (trunk, not MTP): a unit mask m [E, I] scales down_proj's columns; per
    example, the main loss's backward gives dL/dm_c = sum_t g_c h_c (g = dL/dh, h the channel's activation over the
    gradient window). Returns {module name: {"fisher": sum_ex (dL/dm)^2, "attr": sum_ex |dL/dm|, "taylor": |sum_ex
    dL/dm|}} on the host (float64), plus the example count. Only the masks take gradients; MTP off for the pass."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    mods = {name: mod for name, mod in model.named_modules() if isinstance(mod, m.Qwen4ExpTextExperts)
            and ".language_model.layers." in f".{name}"}
    masks = {id(mod): torch.ones(mod.down_proj.shape[0], mod.down_proj.shape[-1], device=mod.down_proj.device,
                                 dtype=torch.float32, requires_grad=True) for mod in mods.values()}
    sums = {name: {k: torch.zeros(mask.shape, dtype=torch.float64) for k in ("fisher", "attr", "signed")}
            for name, mask in ((n, masks[id(mod)]) for n, mod in mods.items())}
    cls = m.Qwen4ExpTextExperts
    inner = cls.forward

    def forward(self, *args, **kwargs):
        mask = masks.get(id(self))
        if mask is None:
            return inner(self, *args, **kwargs)
        kept = self._parameters["down_proj"]
        try:
            self._parameters["down_proj"] = kept * mask.to(kept.dtype).unsqueeze(1)  # [E, H, I] x [E, 1, I]
            return inner(self, *args, **kwargs)
        finally:
            self._parameters["down_proj"] = kept

    params = [p for p in list(model.parameters()) + list(head.parameters())]
    grads = [p.requires_grad for p in params]
    fcfg = dataclasses.replace(cfg, mtp_weight=0.0)
    cls.forward = forward
    try:
        for p in params:
            p.requires_grad_(False)
        n = 0
        for made in stager.run(examples):
            for mask in masks.values():
                mask.grad = None
            example_loss(model, head, box, made, fcfg, scale=1.0, train=True)
            for name, mod in mods.items():
                g = masks[id(mod)].grad
                if g is None:
                    continue
                g = g.double().cpu()
                sums[name]["fisher"] += g * g
                sums[name]["attr"] += g.abs()
                sums[name]["signed"] += g
            n += 1
    finally:
        cls.forward = inner
        for p, r in zip(params, grads):
            p.requires_grad_(r)
    return {name: {"fisher": s["fisher"], "attr": s["attr"], "taylor": s["signed"].abs()} for name, s in sums.items()}, n


def eval_set_files(cfg, index, cursor, s):
    """``s["n"]`` files spread evenly over the part of epoch ``cursor[0]``'s order before ("seen") or from ("unseen")
    position ``cursor[1]``."""
    names = pool(cfg, index, cursor[0])
    part = names[: cursor[1]] if s["part"] == "seen" else names[cursor[1]:]
    if s["part"] not in ("seen", "unseen") or not part:
        raise ValueError(f"eval set {s}: part 'seen' or 'unseen' with files at cursor {cursor}")
    n = min(int(s["n"]), len(part))
    return [part[i * len(part) // n] for i in range(n)]


@contextlib.contextmanager
def diag_read(on, out, name, files):
    """With ``on``: the reads inside collect per-row KL parts (DIAG), saved to <out>/diag-<name>.pt."""
    if not on:
        yield
        return
    DIAG.update(on=True, pos=None, rows=[], ids=[])
    try:
        yield
        rows = DIAG["rows"]
        got = {k: torch.cat([r[k] for r in rows]) for k in rows[0] if k != "ex"} if rows else {}
        got["ex"] = torch.cat([torch.full((len(r["pos"]),), r["ex"]) for r in rows]) if rows else torch.zeros(0)
        torch.save({"files": files, "ids": DIAG["ids"], "rows": got}, Path(out) / f"diag-{name}.pt")
    finally:
        DIAG.update(on=False, pos=None, rows=[], ids=[])


INT8_DENSE = tuple(f + ".weight" for f in qat.INT8_DENSE)  # dense_kernels_runtime INT8_FAMILIES (fused there, per-row here)
int8_rows = qat.int8_rows  # dense_kernels_runtime.quantize_rows round trip (BF16 out)


def dense_int8_params(model):
    """The served INT8 families' trunk weights (MTP excluded): [(name, param)], lm_head included."""
    got = [(n, p) for n, p in model.named_parameters() if not n.startswith("mtp") and n.endswith(INT8_DENSE)]
    got.append(("lm_head.weight", model.lm_head.weight))
    return got


def eval_variant(cfg, model, head, box, examples, stager, v, out, step):
    """One eval-only screen variant: the slice with the QSA K / V through ``attention.kv_fake(v["kv"])`` and / or the
    routed experts on another grid (``levels`` / ``zero`` / ``group``: scales re-searched, the eager grid). Restores
    the run's KV and grid after; event set "slice:<name>"."""
    grid = {k: v[k] for k in ("levels", "zero", "group", "grid") if k in v}
    if grid and not cfg.fake_quant:
        raise ValueError(f"variant {v['name']}: an expert grid needs fake_quant")
    if grid and "grid" not in grid:
        grid["grid"] = "int4"  # levels / zero / group: the int4 family of grids
    if grid.get("grid", "int4") not in qat.GRIDS:
        raise ValueError(f"variant {v['name']}: unknown grid {grid['grid']}")
    saved = {k: qat.QAT[k] for k in ("levels", "zero", "group", "fused", "grid")}
    routers = [m for m in model.modules() if type(m).__name__.endswith("TopKRouter")] if "top_k" in v else []
    if "top_k" in v and not routers:
        raise ValueError(f"variant {v['name']}: no TopKRouter modules for top_k")
    top_ks = [m.top_k for m in routers]
    # dense_int8_train: the forward already reads the INT8 weights (a second round trip is not the identity in BF16)
    dense = dense_int8_params(model) if v.get("dense_int8") and not qat.DENSE["on"] else []
    if dense:
        types = model.config.text_config.layer_types
        want = 3 * sum(t == "linear_attention" for t in types) + 4 * sum(t != "linear_attention" for t in types) + 1
        if len(dense) != want:
            raise ValueError(f"variant {v['name']}: {len(dense)} INT8 dense weights, expected {want}")
    kept = []
    try:
        for m in routers:  # routed experts per token (served num_experts_per_tok), the MTP block's router included
            m.top_k = int(v["top_k"])
        attention.KVQ["mode"] = v.get("kv")
        attention.INDEX["fp8"] = bool(v.get("idx_fp8"))
        with torch.no_grad():  # in place (packed GDN views and compacted buffers keep their pointers), restored after
            for _, p in dense:
                kept.append(p.detach().to("cpu", copy=True))
                p.copy_(int8_rows(p))
        if grid:
            qat.QAT.update(grid, fused=None)  # the run's fused grid is not the variant's: eager
            qat.refresh(model)
        event(out, event="eval", step=step, set=f"slice:{v['name']}", variant=v, examples=len(examples),
              **({"dense_int8_weights": len(dense)} if dense else {}), **evaluate(cfg, model, head, box, examples, stager))
    finally:
        for m, k in zip(routers, top_ks):
            m.top_k = k
        attention.KVQ["mode"] = None
        attention.INDEX["fp8"] = False
        with torch.no_grad():
            for (_, p), w in zip(dense, kept):
                p.copy_(w.to(p.device))
        if grid:
            qat.QAT.update(saved)
            qat.refresh(model)


# --- start-up ------------------------------------------------------------------------------------------------------

def warm_window(cfg, model, head, box, index):
    """Forward and backward of a dummy example as large as the data's largest (total tokens, gradient window), no
    optimizer step: kernel compiles and autotunes happen here, and the peak memory of the run's largest window is
    known before any data flows. Gradients are dropped; the global RNG is untouched."""
    files = index["examples"]
    total = max(int(e.get("tokens") or cfg.window) for e in files)  # build.py indexes carry tokens and tail
    tail = max(min(int(e.get("tail") or cfg.window), cfg.window, int(e.get("tokens") or cfg.window)) for e in files)
    vocab = min(int(model.config.text_config.vocab_size), 200_000)
    g = torch.Generator().manual_seed(cfg.seed)
    ids = torch.randint(10, vocab, (1, total), generator=g)
    labels = torch.full_like(ids, -100)
    labels[:, total - tail:] = ids[:, total - tail:]
    cuda = next(model.parameters()).is_cuda
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ex = {"input_ids": ids, "labels": labels, "tail": tail}
    if cfg.loss == "kl":
        n = tail - 1
        ex["teacher"] = {"pos": torch.arange(total - n, total, dtype=torch.int32),
                         "ids": torch.randint(0, vocab, (n, cfg.teacher_k), generator=g, dtype=torch.int32),
                         "lp": torch.full((n, cfg.teacher_k), -math.log(4 * cfg.teacher_k))}
    params = trainable(model, head)
    for p in params:  # the steady step's peak: every example after a step's first runs with the full gradients
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    example_loss(model, head, box, ex, cfg, scale=1.0)
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
    return {"tokens": total, "window": tail, "peak_gib": None if peak is None else round(peak, 2)}


def init_states(opt):
    """torchao's lazy state init (AdamW8bit.step, torchao 0.18: step 0, exp_avg / exp_avg_sq as ``_new_buffer``
    zeros) done now for every parameter without state, so the warm-up's peak holds them (util-v2: an analytic 66 GiB
    stopped the boot; v8 measured +36 GiB). Same values as the first step's init. Returns the GiB allocated."""
    if not hasattr(opt, "_new_buffer"):
        return 0.0
    cuda = torch.cuda.is_available()
    before = torch.cuda.memory_allocated() if cuda else 0
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state[p]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = opt._new_buffer(p, True)
                state["exp_avg_sq"] = opt._new_buffer(p, False)
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = opt._new_buffer(p, False)
    if hasattr(opt, "page_out"):  # paged.py: made on the card, then moved to the host
        opt.page_out()
    return ((torch.cuda.memory_allocated() if cuda else 0) - before) / 2 ** 30


def warm_optimizer(opt):
    """torchao's AdamW step is torch.compile'd per parameter shape (dynamic=False); compile every shape once on scratch
    tensors so step 1 does not. The real state is untouched and the CUDA RNG (stochastic rounding) restored."""
    try:
        from torchao.optim.adam import single_param_adam
    except ImportError:
        return 0
    if not hasattr(opt, "_new_buffer"):
        return 0
    rng = torch.cuda.get_rng_state()
    seen = set()
    with torch.no_grad(), torch._dynamo.utils.disable_cache_limit():
        for group in opt.param_groups:
            for p in group["params"]:
                dtype = torch.float32 if getattr(opt, "kahan", False) and p.dtype is torch.bfloat16 else p.dtype
                key = (tuple(p.shape), tuple(p.stride()), dtype)
                if key in seen:
                    continue
                seen.add(key)
                q = torch.zeros_like(p, dtype=dtype)
                torch.compile(single_param_adam, fullgraph=True, dynamic=False)(
                    q.detach(), torch.zeros_like(p), torch.tensor(1.0), opt._new_buffer(q, True),
                    opt._new_buffer(q, False), None, group["lr"], group["betas"][0], group["betas"][1],
                    group["weight_decay"], group["eps"], opt.is_adamw,
                    opt.bf16_stochastic_round and q.dtype is torch.bfloat16)
                del q
    torch.cuda.set_rng_state(rng)
    return len(seen)


# --- loop ----------------------------------------------------------------------------------------------------------

class Metrics(threading.Thread):
    """Rows for tools/ops/watch.py (plan ``metrics_remote``: <out>/metrics.jsonl; {"t", "m"}): GPU util (mean of
    nvidia-smi samples every ``every`` s) and VRAM (max), host RAM in use, every ``period`` s; the trainer adds its
    step and eval rows (``row``). Commits at most once a minute."""

    def __init__(self, out, commit, period=30.0, every=5.0, gpus=1):
        super().__init__(daemon=True, name="sft-metrics")
        self.out, self.commit, self.period, self.every, self.gpus = out, commit, period, every, gpus
        self.path = spool_path(out, "metrics.jsonl")
        self.lock, self.halt, self.util, self.mem, self.step_util = threading.Lock(), threading.Event(), [], [], []
        self.step_mem = []
        self.committed = 0.0
        self.per = [([], []) for _ in range(gpus)]  # gpus > 1: each card's samples (util, VRAM) since the last row
        self.step_per = [([], []) for _ in range(gpus)]

    def sample(self):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            got = [tuple(float(x) for x in line.split(",")) for line in r.stdout.splitlines()[: self.gpus]]
            u, m = got[0]
            if self.gpus > 1:  # every card of the container (one rank each)
                for (pu, pm), (su, sm), (gu, gm) in zip(self.per, self.step_per, got):
                    pu.append(gu)
                    pm.append(gm / 1024)
                    su.append(gu)
                    sm.append(gm / 1024)
                u, m = sum(g[0] for g in got) / len(got), max(g[1] for g in got)
            return u, m / 1024
        except Exception:  # noqa: BLE001 - no GPU (CPU tests)
            return None

    def run(self):
        last = time.time()
        while not self.halt.wait(self.every):
            got = self.sample()
            if got is not None:
                self.util.append(got[0])
                self.mem.append(got[1])
                self.step_util.append(got[0])
                self.step_mem.append(got[1])
            if time.time() - last >= self.period:
                last = time.time()
                m = {}
                if self.util and self.gpus == 1:
                    m.update({"infra.gpu_util.0": round(sum(self.util) / len(self.util), 1),
                              "infra.vram_gib.0": round(max(self.mem), 2)})
                for i, (pu, pm) in enumerate(self.per if self.gpus > 1 else []):
                    if pu:
                        m.update({f"infra.gpu_util.{i}": round(sum(pu) / len(pu), 1), f"infra.vram_gib.{i}": round(max(pm), 2)})
                    pu.clear()
                    pm.clear()
                self.util, self.mem = [], []
                try:
                    from arc_infra.runops import hostmem

                    m["infra.host_ram_gib"] = hostmem.sample().get("cgroup_used_gib")
                except ImportError:
                    pass
                self.row(**m)
            if time.time() - self.committed >= 60:  # flush + commit here, never from the training loop's row()
                self.committed = time.time()
                spool_flush(self.out)
                self.commit()

    def take(self):
        """(mean util, max VRAM GiB, samples) since the last take (gpus > 1: over every card; ``per_card`` each)."""
        u, m, self.step_util, self.step_mem = self.step_util, self.step_mem, [], []
        self.per_card = [(round(sum(su) / len(su), 1) if su else None, round(max(sm), 2) if sm else None)
                         for su, sm in self.step_per]
        for su, sm in self.step_per:
            su.clear()
            sm.clear()
        return (round(sum(u) / len(u), 1) if u else None), (round(max(m), 2) if m else None), len(u)

    def row(self, **m):
        if not WRITER["on"]:
            return
        line = json.dumps({"t": round(time.time(), 1), "m": {k: v for k, v in m.items() if v is not None}})
        with self.lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")

    def stop(self):
        self.halt.set()


def ckpt_possible(cfg, step, step_s, since_ckpt):
    """Whether a scheduled checkpoint can fall after ``step`` in this run: the final one, a ``ckpt_steps`` multiple, or
    ``ckpt_seconds`` reached at the measured step time (with 2x slack). False for short probes (stop_step before
    total_steps, stop_ckpt off): their snapshot buffer (cudaHostRegister of ~240 GB) cost 205-240 s of boundary gap at
    steps 2-3 for a checkpoint never written (28 Oct p75 lr probes). A knob-requested checkpoint still saves inline."""
    last = min(cfg.stop_step or cfg.total_steps, cfg.total_steps)
    if last >= cfg.total_steps or cfg.stop_ckpt:
        return True
    if cfg.ckpt_steps and (last // cfg.ckpt_steps) * cfg.ckpt_steps > step:
        return True
    return bool(cfg.ckpt_seconds) and since_ckpt + 2 * (last - step) * step_s >= cfg.ckpt_seconds


def arc2_due(step, steps, every):
    """Whether the ARC-2 held-out read runs at ``step``: a listed step, or a multiple of ``every`` (> 0)."""
    return step in steps or (every > 0 and step % every == 0)


def read_knobs(out, seen, source=None):
    """New knobs from <out>/knobs.json; ``source``: a callable giving its text (on Modal the parent's read of the
    committed file: a volume reload fails while any file on it is open, which is nearly always during training)."""
    try:
        text = source() if source else (Path(out) / "knobs.json").read_text()
        k = json.loads(text) if text else {}
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return {}
    fresh = {}
    for tag in ("ckpt", "eval"):
        if k.get(tag) and k[tag] != seen.get(tag):
            seen[tag] = fresh[tag] = k[tag]
    if "lr" in k:
        fresh["lr"] = float(k["lr"])
    if k.get("batch_reply_tokens") and k["batch_reply_tokens"] != seen.get("batch"):
        seen["batch"] = fresh["batch"] = int(k["batch_reply_tokens"])
    if "arc2_heldout_every" in k and int(k["arc2_heldout_every"]) != seen.get("arc2_every"):
        seen["arc2_every"] = fresh["arc2_every"] = int(k["arc2_heldout_every"])
    if k.get("stop") and k["stop"] != seen.get("stop"):  # a level knob read as a tag: a value present at start is stale
        seen["stop"], fresh["stop"] = k["stop"], True
    return fresh


ALLOC = ("num_alloc_retries", "num_sync_all_streams", "num_device_alloc", "num_device_free", "num_ooms")


def alloc_counts():
    """The caching allocator's counters (a retry frees the cache and syncs; device alloc / free map memory)."""
    if not torch.cuda.is_available():
        return {}
    st = torch.cuda.memory_stats()
    return {k: int(st.get(k, 0)) for k in ALLOC}


def cacheable_states():
    """torchao's state subclasses lack ``_stable_hash_for_caching``, so the per-shape compile of the AdamW step is
    never cached (warm_optimizer: ~3 min at 0% util on every boot, util-v3). The hash covers every attribute the
    compiled step depends on besides the tensor metadata the cache keys on anyway."""
    from torchao.optim import subclass_4bit, subclass_8bit

    def stable(self):
        return (f"{type(self).__name__}:{self.signed}:{self.block_size}:{tuple(self.shape)}:{self.dtype}:"
                f"{self.codes.dtype}:{tuple(self.scale.shape)}:{tuple(self.qmap.shape)}")

    for cls in (subclass_8bit.OptimState8bit, subclass_4bit.OptimState4bit):
        if not hasattr(cls, "_stable_hash_for_caching"):
            cls._stable_hash_for_caching = stable


def adamw8bit(params, lr, bits=8):
    import torchao.optim

    cacheable_states()

    kind = {8: torchao.optim.AdamW8bit, 4: torchao.optim.AdamW4bit}[bits]
    return kind(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, bf16_stochastic_round=True)


def train(cfg, *, device="cuda:0", refresh=lambda: None, commit=lambda: None, guard=None, make_opt=adamw8bit,
          knobs=None, tables=None):
    """The run. Under torch.distributed with world > 1 (shard.launch), one rank of a sharded run: ``tables`` are the
    shared n-gram tables; refresh / commit / guard / knobs matter on rank 0 only."""
    began_at = time.time()
    t0 = float(os.environ.get("ARC_T0") or began_at)  # the container's start (modal_sft)
    boot, mark = {"container": round(began_at - t0, 1)}, [began_at]

    def phase(name):
        now = time.time()
        boot[name] = round(now - mark[0], 1)
        mark[0] = now

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    resume = ckpt.latest(out) or (Path(cfg.init_ckpt) if cfg.init_ckpt else None)
    ctx = shard.context(device, resume / "student" if resume else cfg.student, cfg.reduce_dtype)
    WRITER["on"] = ctx is None or ctx.rank == 0
    if WRITER["on"] and cfg.ckpt_pin:
        pins = out / "pins.json"
        had = set(json.loads(pins.read_text())) if pins.is_file() else set()
        pins.write_text(json.dumps(sorted(had | {int(x) for x in cfg.ckpt_pin})))
    if ctx is not None:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        if cfg.channel_fisher or cfg.eval_dump:
            raise NotImplementedError("channel_fisher and eval_dump run on one GPU")
        if cfg.kahan:
            raise NotImplementedError("kahan states are not re-sliced; one GPU only")
    lock = threading.Lock()
    spool_open(out)
    import atexit

    def at_exit():  # a crash still leaves its last rows on the volume
        spool_flush(out)
        commit()

    atexit.register(at_exit)

    def committed(force=False):
        """Volume commits (the spool copied first); skipped while a checkpoint is being written (the writer commits
        when done)."""
        spool_flush(out)
        if force or not saver.busy:
            with lock:
                commit()

    EMPTY["n"] = 0
    saver = ckpt.Saver(cfg, lambda: committed(force=True), cfg.async_ckpt, log=lambda **row: event(out, **row))
    boundary = {"thread": None, "skipped": 0}

    def volume_work():
        """The step boundary's volume commit, reload and knob read (util-v4: 2-8 s per step at 0% util inline; the
        knob read goes through the same parent bridge as the commit, so it waited behind it in util-v6). Knobs read
        here take effect at the next step boundary."""
        committed()
        got = read_knobs(out, seen, knobs)
        if got:
            boundary["fresh"] = {**boundary.get("fresh", {}), **got}
        if cfg.boundary_reload and not saver.busy:  # a volume reload with the writer's files open fails
            try:  # fails while any volume file is open (26 Oct smoke v4); knobs come through ``knobs`` regardless
                refresh()
            except RuntimeError as error:
                boundary["skipped"] += 1
                if boundary["skipped"] in (1, 10) or boundary["skipped"] % 100 == 0:
                    event(out, event="reload_skipped", step=step, count=boundary["skipped"], error=str(error)[:200])

    def boundary_kick():
        """``volume_work`` on a thread, one at a time (skipped while the last one runs)."""
        t = boundary["thread"]
        if t is None or not t.is_alive():
            boundary["thread"] = threading.Thread(target=volume_work, daemon=True, name="sft-volume")
            boundary["thread"].start()

    torch.manual_seed(cfg.seed if cfg.sr_seed < 0 else cfg.sr_seed)
    if WRITER["on"]:
        for stale in out.glob(".ckpt-*"):  # a write the last run did not finish
            shutil.rmtree(stale, ignore_errors=True)
    model, head = load_model(resume / "student" if resume else cfg.student, device, cfg.fast_attention,
                             fast_load=cfg.fast_load, sel_cache=cfg.sel_cache, gdn_fast=cfg.gdn_fast,
                             gdn_keep=cfg.gdn_keep, hyper_fast=cfg.hyper_fast, union=cfg.union_attention,
                             ctx=ctx, tables=tables)
    if ctx is not None:
        ctx.barrier()  # every rank loaded (and rank 0 cleared stale writes) before any pass
    if cfg.top_k:  # top-k heal: the trunk routes to top_k experts per token from the first read on
        routers = [m for m in model.modules() if type(m).__name__.endswith("TopKRouter")]
        if not routers:
            raise ValueError("top_k: no TopKRouter modules")
        for m in routers:
            m.top_k = int(cfg.top_k)
        LOAD["top_k"] = {"k": int(cfg.top_k), "routers": len(routers)}
    if cfg.qat_grid not in qat.GRIDS or (cfg.qat_grid != "int4" and not cfg.fake_quant):
        raise ValueError(f"qat_grid {cfg.qat_grid!r}: one of {qat.GRIDS}; a non-int4 grid needs fake_quant")
    if cfg.kv_train not in ("", "kv4"):
        raise ValueError(f"kv_train {cfg.kv_train!r}: '' or 'kv4'")
    check_depth_weights(cfg.depth_weights, cfg.loss)
    if cfg.kv_train and not attention.selected_on():
        raise ValueError("kv_train needs the selected-attention forward (fast_attention on CUDA)")
    attention.KVQ["train"] = cfg.kv_train or None
    qat.DENSE["on"] = False
    if cfg.dense_int8_train:
        names = qat.install_dense(model, head)  # the MTP draft's q / k / v / o included, as served
        types = model.config.text_config.layer_types
        want = 3 * sum(t == "linear_attention" for t in types) + 4 * sum(t != "linear_attention" for t in types) \
            + 4 * len(head.layers)
        if len(names) != want:
            raise ValueError(f"dense_int8_train: {len(names)} INT8 dense modules, expected {want}")
        LOAD["dense_int8"] = len(names) + 1  # + lm_head
    if cfg.fake_quant:
        qat.QAT["group"] = int(cfg.qat_group)
        qat.QAT["grid"] = cfg.qat_grid
        qat.install(model)
        if cfg.qat_fused and str(device).startswith("cuda"):
            mod = qat.routed(model)[0]
            LOAD["qat_fused"] = qat.fuse([(getattr(mod, n).detach(), qat.SCALES[id(mod)][n]) for n in qat.NAMES])
            if ctx is not None and ctx.bcast(LOAD["qat_fused"]) != LOAD["qat_fused"]:
                LOAD["qat_fused"], qat.QAT["fused"] = False, None  # one grid path on every rank (eager: bit-equal)
            if ctx is not None and ctx.max_int(not LOAD["qat_fused"]):
                LOAD["qat_fused"], qat.QAT["fused"] = False, None
        if ctx is not None:  # per-expert grids on the shard; the count over every rank's experts
            qat.QAT["zero_experts"] = int(ctx.sums([qat.QAT["zero_experts"]])[0])
        if qat.QAT["zero_experts"]:
            LOAD["qat_zero_experts"] = qat.QAT["zero_experts"]
    if cfg.ple_table:
        ple.install(model)
    trunk_layers = []
    if cfg.keep_layers:  # same math, no recompute for these layers; the warm-up's peak gates the memory
        from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

        trunk_layers = [m for n, m in model.named_modules() if isinstance(m, modeling.Qwen4ExpTextDecoderLayer)
                        and ".language_model.layers." in f".{n}"]

    def keep(n):
        """The last ``n`` trunk layers without the checkpoint recompute; the rest checkpointed."""
        for i, m in enumerate(trunk_layers):
            m.gradient_checkpointing = i < len(trunk_layers) - n
        LOAD["kept_layers"] = n

    keep(cfg.keep_layers)
    phase("load")
    box, _ = mtp_mod.hc_capture(model)
    params = trainable(model, head)
    rep = shards = None
    if ctx is not None:  # group 0 replicated, group 1 this rank's expert shards (shard.seeded_steps)
        rep, shards = ctx.split_params(model, head)
        params = [{"params": rep}, {"params": shards}]
    opt = make_opt(params, cfg.lr, cfg.optim_bits) if cfg.optim_bits != 8 else make_opt(params, cfg.lr)
    pager = None
    if cfg.kahan and not cfg.paged_states:
        raise ValueError("kahan needs paged_states")
    if cfg.paged_states:
        import paged

        if str(device).startswith("cuda"):  # bit-equal to the on-card step on scratch tensors, else no run
            LOAD["paged_check"] = paged.check(lambda ps: make_opt(ps, cfg.lr, cfg.optim_bits) if cfg.optim_bits != 8
                                              else make_opt(ps, cfg.lr), device)
            if not LOAD["paged_check"]["ok"]:
                raise RuntimeError(f"paged optimizer states differ from on-card: {LOAD['paged_check']}")
        pager = paged.Pager(opt, device, depth=cfg.paged_depth, kahan=cfg.kahan).install()
        saver.direct = pager.owns  # the host states go into the checkpoint from their own storage, not the buffer
    names = shard.names_of(model, head)
    seeds = None
    if ctx is not None:
        seeds = shard.seeded_steps(opt, ctx, cfg.seed if cfg.sr_seed < 0 else cfg.sr_seed)
        saver.part = shard.Part(ctx, model, head, rep, shards)
        saver.ranks = ctx.world
    step, cursor = 0, (0, 0)
    if resume:
        if ctx is None:
            ckpt.load_optimizer(opt, resume, names=names)
        else:  # any checkpoint (one GPU's or another N's), re-sliced to this rank's expert rows
            n1 = [names[id(p)] for p in trainable(model, head)]
            shard.load_optimizer(opt, resume, names, {n: (ctx.lo, ctx.hi) for n in saver.part.sharded}, n1_names=n1)
        if pager:
            pager.page_out()
        state = json.loads((resume / "state.json").read_text())
        step, cursor = state["step"], tuple(state["cursor"])
        if cfg.init_fresh_data and resume == Path(cfg.init_ckpt or "/"):
            cursor = (0, 0)
    phase("optimizer")
    start = schedule_start(cfg, out, resume, step)
    if guard is not None:
        guard.start(checkpoint_step=step)
        guard.watch()  # SIGINT/SIGTERM evidence + heartbeat: a preemption is then classified as one (auto-resume)
    root = Path(cfg.data)  # build.py: index.json {"train"|"heldout": {"files": [{"file": "<split>/<name>.pt", ...}]}}
    index = json.loads((root / "index.json").read_text())
    teacher = cfg.loss == "kl"
    hroot = Path(cfg.heldout_data or cfg.data)
    hindex = index if hroot == root else json.loads((hroot / "index.json").read_text())
    out_pairs = {tuple(x) for x in cfg.exclude_pairs}
    out_files = set(json.loads(Path(cfg.exclude_files).read_text())) if cfg.exclude_files else set()
    train_index, held_index = ({"examples": [e for e in ix[s]["files"]  # kl: only examples the teacher scored
                                             if (not teacher or (r / "teacher" / e["file"]).is_file())
                                             and (e.get("run"), e.get("game")) not in out_pairs
                                             and (s == "heldout" or e["file"] not in out_files)]}
                               for ix, r, s in ((index, root, "train"), (hindex, hroot, "heldout")))
    if cfg.curriculum:  # build.py's curriculum order, restricted to the examples kept above
        kept = {e["file"] for e in train_index["examples"]}
        train_index["order"] = [f for f in json.loads((root / "priority.json").read_text()) if f in kept]
        if not train_index["order"]:
            raise ValueError("curriculum: priority.json names none of the training examples")

    def size(e):  # an index entry's compute for the ranks' LPT split
        return float(e.get("tokens") or e.get("reply_tokens") or 1)

    def share(entries):  # a sharded run's rank reads (and evaluates) its LPT share; one GPU all
        return entries if ctx is None else ctx.pick(entries, size)

    n_held = len(held_index["examples"][: cfg.eval_examples])
    n_more = len(held_index["examples"][n_held: cfg.eval_full_examples])

    def load_more(before=lambda: None):  # the full set's examples beyond the slice
        more = []
        for e in share(held_index["examples"][n_held: cfg.eval_full_examples]):
            before()
            more.append(load_example(hroot / e["file"], hroot / "teacher" / e["file"] if teacher else None))
        return more

    ahead = {}  # full-eval step -> (thread, box): its extra examples, loading during the steps before it

    def full_prefetch(step):
        for s in cfg.eval_full_steps:
            if step < s <= step + cfg.eval_full_lead and s not in ahead and cfg.eval_full_examples > n_held:
                box_ = {}

                def load(box_=box_):
                    try:  # each file waits out the paged optimizer step, as the train loader does
                        box_["more"] = load_more(loader.calm.wait if loader is not None else (lambda: None))
                    except Exception as error:  # the eval loads them itself
                        box_["error"] = str(error)[:300]
                thread = threading.Thread(target=load, name=f"full-eval-{s}", daemon=True)
                thread.start()
                ahead[s] = (thread, box_)

    def full_eval(step):  # the full set at the listed steps: the slice in RAM plus the rest loaded for this read only
        if step not in cfg.eval_full_steps or cfg.eval_full_examples <= n_held:
            return None
        thread, box_ = ahead.pop(step, (None, {}))
        if thread is not None:
            thread.join()
        more = box_.get("more")
        if more is None:
            if "error" in box_:
                event(out, event="full_prefetch_failed", step=step, error=box_["error"])
            more = load_more()
        with diag_read(cfg.eval_dump and step in cfg.eval_full_steps, out, f"full-{step}",
                       [e["file"] for e in held_index["examples"][: n_held + n_more]]):
            ev = evaluate(cfg, model, head, box, held + more, stager)
        event(out, event="eval", step=step, set="full", examples=n_held + n_more, **ev)
        return ev

    arc2_every = [cfg.arc2_heldout_every]  # the knob "arc2_heldout_every" changes it live

    def arc2_eval(step):  # every ARC-3 run's ARC-2 held-out read, start, middle and end
        if not cfg.arc2_heldout_data or not arc2_due(step, cfg.arc2_heldout_steps, arc2_every[0]):
            return None
        aroot, began = Path(cfg.arc2_heldout_data), time.time()
        files = json.loads((aroot / "index.json").read_text())["heldout"]["files"]
        exs = [load_example(aroot / e["file"], aroot / "teacher" / e["file"] if teacher else None) for e in share(files)]
        ev = evaluate(cfg, model, head, box, exs, stager)
        event(out, event="eval", step=step, set="arc2", examples=len(files), seconds=round(time.time() - began, 1), **ev)
        return ev

    held = [load_example(hroot / e["file"], hroot / "teacher" / e["file"] if teacher else None)
            for e in share(held_index["examples"][: cfg.eval_examples])]
    by_file = {e["file"]: e for e in train_index["examples"]}
    pick = None if ctx is None else (lambda names: ctx.pick(names, lambda f: size(by_file[f])))
    plan = BatchPlan(cfg.batch_reply_tokens, step)

    def batch_knob(tokens, step):
        """From the first step no rank can hold yet: past the staged batch, the queue and the one the loader is at."""
        if cfg.live_batch and tokens:
            plan.set(tokens, step + cfg.prefetch + 4)
            event(out, event="batch_knob", step=step, tokens=tokens, first_step=step + cfg.prefetch + 4)

    if cfg.prefetch:  # the first steps' files load during the warm-up
        loader = feed.Loader(lambda c: next_batch(cfg, train_index, c, plan.take()), root, cursor, cfg.prefetch, teacher,
                             pick=pick)
        loader.start()
        get_batch = loader.get
    else:
        loader, at = None, [cursor]

        def get_batch():
            names, reply, at[0] = next_batch(cfg, train_index, at[0], plan.take())
            names = pick(names) if pick is not None else names
            exs = [load_example(root / f, root / "teacher" / f if teacher else None) for f in names]
            return names, reply, at[0], exs, sum(int((e["labels"][0, 1:] != -100).sum()) for e in exs)

    phase("data")
    cards = ctx.world if ctx is not None and str(device).startswith("cuda") else 1
    metrics = Metrics(out, committed, cfg.metrics_seconds, every=cfg.util_every, gpus=cards)
    if WRITER["on"]:
        metrics.start()
    n_params = sum(p.numel() for p in trainable(model, head))
    if ctx is not None:  # the model's: replicated once, every rank's shards
        n_params = int(sum(p.numel() for p in rep) + ctx.sums([sum(p.numel() for p in shards)])[0])
    event(out, event="start", step=step, resume=str(resume) if resume else None, config=dataclasses.asdict(cfg),
          params=n_params, examples=len(train_index["examples"]), load=LOAD,
          **({"world": ctx.world, "expert_ranges": ctx.ranges} if ctx is not None else {}))
    warm = {}
    # eval-only (no step to take): no train-window backward / optimizer states (29 Oct: a 256-expert student OOMed
    # there, its gradients + states alone > the card, though the reads need neither)
    eval_only = step >= min(cfg.stop_step or cfg.total_steps, cfg.total_steps)
    if cfg.warm_start and not eval_only:
        state_gib = init_states(opt)  # the steady step's peak: gradients (warm_window) and optimizer states held
        while True:  # kept layers step down one at a time until the warm-up fits (util-v5: 3 kept OOMed)
            try:
                with shard.warm_guard(ctx) if ctx is not None else contextlib.nullcontext():  # a rank's OOM: all raise
                    warm = warm_window(cfg, model, head, box, train_index)
                if ctx is not None and warm["peak_gib"] is not None:  # the ranks step down together (same passes)
                    warm["peak_ranks"] = [r[0] for r in ctx.gather_rows([warm["peak_gib"]])]
                    warm["peak_gib"] = max(warm["peak_ranks"])
                if warm["peak_gib"] is None or warm["peak_gib"] <= cfg.mem_budget_gib or not LOAD.get("kept_layers"):
                    break
                why = f"peak {warm['peak_gib']} GiB"
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:  # the steady step would not fit either
                if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error):
                    raise  # 28 Oct lr probe: a Triton autotune launch OOM is a RuntimeError ("Triton Error [CUDA]")
                if not LOAD.get("kept_layers"):  # N > 1: every rank is here (warm_guard), with the same kept layers
                    event(out, event="memory_stop", state_gib=round(state_gib, 2), error=str(error)[:300])
                    raise
                why = str(error)[:120]
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            keep(LOAD["kept_layers"] - 1)
            event(out, event="keep_layers_down", keep_layers=LOAD["kept_layers"], why=why)
        warm["kept_layers"] = LOAD.get("kept_layers", 0)
        phase("warm_window")
        if warm["peak_gib"] is not None:
            warm["state_gib"] = round(state_gib, 2)
            warm["predicted_gib"] = warm["peak_gib"]
            if warm["predicted_gib"] > cfg.mem_budget_gib:
                event(out, event="memory_stop", **warm, budget=cfg.mem_budget_gib)
                raise RuntimeError(f"predicted step peak {warm['predicted_gib']} GiB > budget {cfg.mem_budget_gib}")
        if str(device).startswith("cuda"):
            warm["optimizer_shapes"] = warm_optimizer(opt)
            phase("warm_optimizer")
    cuda = str(device).startswith("cuda")
    stager = feed.Stager(model, device, cfg.vision_ahead,
                         lambda e: loss_plan(cfg, model, e, torch.device(device), cuda))
    if cfg.indexer_tf32:  # the held-out pass with fp32 picks, TF32 picks counted against them
        LOAD["indexer_tf32"] = attention.index_check(lambda: evaluate(cfg, model, head, box, held, stager))
        if ctx is not None:  # one decision for every rank: rank 0's, on the picks summed over ranks
            ix = attention.INDEX
            match, picked = ctx.sums([float(ix["match"]), float(ix["picked"])])
            share = match / picked if picked else 1.0
            ix["tf32"] = ctx.bcast(share >= 0.995)
            LOAD["indexer_tf32"] = {"match": round(share, 6), "picked": int(picked), "on": ix["tf32"]}
        phase("index_check")
    if step == 0 or cfg.eval_resume:
        with diag_read(cfg.eval_dump, out, "slice", [e["file"] for e in held_index["examples"][: cfg.eval_examples]]):
            event(out, event="eval", step=step, set="slice", examples=n_held,
                  **evaluate(cfg, model, head, box, held, stager),
                  **({"peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)} if cuda else {}))
        full_eval(step)
        arc2_eval(step)
        phase("eval0")
        for v in cfg.eval_variants:  # eval-only screens: the slice again under a served-KV / expert-grid variant
            with diag_read(cfg.eval_dump, out, f"slice-{v['name']}",
                           [e["file"] for e in held_index["examples"][: cfg.eval_examples]]):
                eval_variant(cfg, model, head, box, held, stager, v, out, step)
            phase(f"eval_{v['name']}")
        for s in cfg.eval_sets:  # training-game reads: files already trained on, or not yet reached, at this cursor
            files = eval_set_files(cfg, train_index, cursor, s)
            exs = [load_example(root / f, root / "teacher" / f if teacher else None)
                   for f in (files if pick is None else pick(files))]
            with diag_read(cfg.eval_dump, out, s["name"], files):
                event(out, event="eval", step=step, set=f"train:{s['name']}", examples=len(files), cursor=list(cursor),
                      **evaluate(cfg, model, head, box, exs, stager))
            phase(f"eval_{s['name']}")
        if cfg.channel_fisher:  # expert-channel importance on held-out games' examples past the slice (eval-only)
            cf = cfg.channel_fisher
            files = [e["file"] for e in held_index["examples"][int(cf.get("skip", cfg.eval_examples)):]
                     if not cf.get("games") or e.get("game") in set(cf["games"])][: int(cf["n"])]
            exs = [load_example(hroot / f, hroot / "teacher" / f if teacher else None) for f in files]
            scores, n = channel_fisher(cfg, model, head, box, exs, stager)
            torch.save({"files": files, "examples": n, "scores": scores}, out / "channel-fisher.pt")
            event(out, event="channel_fisher", step=step, examples=n, files=len(files), modules=len(scores),
                  zero_fisher_share=round(sum(int((s["fisher"] == 0).sum()) for s in scores.values())
                                          / max(1, sum(s["fisher"].numel() for s in scores.values())), 4))
            phase("channel_fisher")
    event(out, event="boot", step=step, seconds=boot, total=round(time.time() - t0, 1), warm=warm, load=LOAD)
    seen, last_ckpt, lr_override, pending, first, nxt, next_wait = {}, time.time(), None, None, True, None, 0.0
    # the file as found: an earlier attempt's stop must not stop this one (27 Oct); knobs are rank 0's on every rank
    stale = read_knobs(out, seen, knobs) if WRITER["on"] else {}
    lr_override = stale.get("lr") if ctx is None else ctx.bcast(stale.get("lr"))
    if stale.get("stop"):
        event(out, event="stale_stop_ignored", step=step, knob=seen.get("stop"))
    batch_knob(stale.get("batch") if ctx is None else ctx.bcast(stale.get("batch")), step)  # a restart keeps the file's
    skipped = 0
    if cfg.profile:  # the first batch's first example, forward and backward; its gradients dropped
        import prof

        began = time.time()
        nxt, waited = get_batch(), time.time() - began
        ex = nxt[3][0] if nxt[3] else shard.dummy_example(min(cfg.dummy_tokens, cfg.window), 2000)
        total = int(ex["input_ids"].shape[1])
        meta = dict(file=nxt[0][0] if nxt[0] else None, tokens=total,
                    window=min(int(ex.get("tail") or cfg.window), cfg.window, total),
                    reply=int((ex["labels"][0, 1:] != -100).sum()))
        alloc0 = alloc_counts()
        if ctx is None:
            run_one = lambda: example_loss(model, head, box, ex, cfg, scale=1.0)  # noqa: E731
        else:  # every rank its first example (round 0, aligned); each rank's trace, rank 0's as profile.json
            run_one = lambda: aligned(cfg, model, head, box, nxt[3][:1], stager, train=True)  # noqa: E731
        got, _ = prof.profile(run_one, out / ("profile.json" if WRITER["on"] else f"profile-r{ctx.rank}.json"),
                              device if str(device).startswith("cuda") else None, meta)
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        event(out, event="profile", alloc={k: v - alloc0.get(k, 0) for k, v in alloc_counts().items()}, **got)
        committed()
    metrics.take()
    ended = time.time()
    last = min(cfg.stop_step or cfg.total_steps, cfg.total_steps)
    while step < last:
        fresh = boundary.pop("fresh", {})
        if ctx is not None:  # rank 0's knobs (read on its volume thread) on every rank, at the same step
            fresh = ctx.bcast(fresh)
            torch.manual_seed(shard.mix(cfg.seed if cfg.sr_seed < 0 else cfg.sr_seed, step, ctx.rank))
        if WRITER["on"]:
            boundary_kick()
        lr_override = fresh.get("lr", lr_override)
        pending = fresh.get("ckpt") or pending
        batch_knob(fresh.get("batch"), step)
        if "arc2_every" in fresh:
            arc2_every[0] = fresh["arc2_every"]
            event(out, event="arc2_every_knob", step=step, every=arc2_every[0])
        began = time.time()
        gap_util, _, _ = metrics.take()  # the boundary since the last update: eval, checkpoint, knobs, metrics
        if ctx is not None:
            ctx.times()  # the step's collective timers start here (not the boundary's eval)
        alloc0 = alloc_counts()
        gap_s = began - ended
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        if nxt is None:
            nxt, waited = get_batch(), time.time() - began
        (names, reply, cursor, exs, n_step), nxt = nxt, None
        ce = kl = hits = rows = tkl = 0
        if ctx is None:
            got = (example_loss(model, head, box, made, cfg, scale=1.0 / max(1, n_step)) for made in stager.run(exs))
        else:  # this rank's share; the loss scale is the step's global label count, as one GPU's
            n_step = int(ctx.sums([n_step])[0])
            got = aligned(cfg, model, head, box, exs, stager, train=True, scale=1.0 / max(1, n_step))
        for c, _, m, h, r, d in got:
            ce, kl, hits, rows, tkl = ce + c, kl + m, hits + h, rows + r, tkl + d
        sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)
        sync()  # the update reads the gradients next anyway; the split below is card time
        examples_s = time.time() - began
        split, t = {}, time.time()
        if step + 1 < last:  # the next step's first example stages during the update
            asked = time.time()
            nxt = get_batch()
            next_wait = time.time() - asked
        split["next_batch"], t = round(time.time() - t, 2), time.time()
        if ctx is None:
            gn = float(torch.nn.utils.clip_grad_norm_(trainable(model, head), cfg.grad_clip))
            ce, kl, hits, tkl = host(ce, kl, hits, tkl)
        else:  # replicated gradients summed over ranks, then one global norm
            shard.sync_dense(rep, ctx)
            split["dense_sync"], t = round(time.time() - t, 2), time.time()
            gn = float(shard.clip(rep, shards, cfg.grad_clip, ctx))
            ce, kl, hits, tkl = host(ce, kl, hits, tkl)
            ce, kl, hits, rows, tkl = ctx.sums([ce, kl, hits, rows, tkl])
        split["clip"], t = round(time.time() - t, 2), time.time()
        lr = lr_override if lr_override is not None else schedule(cfg, step, start)
        for g in opt.param_groups:  # torchao keeps lr as a tensor (its compiled step reads it)
            g["lr"].fill_(lr) if torch.is_tensor(g["lr"]) else g.__setitem__("lr", lr)
        saver.before_update()
        if loader is not None:  # the paged step is host-launch-bound: no example loads beside it (util-v6)
            loader.calm.clear()
        if math.isfinite(gn):
            if seeds is not None:
                seeds["step"] = step
            opt.step()
            sync()
            split["opt"], t = round(time.time() - t, 2), time.time()
        if loader is not None:
            loader.calm.set()
        if step + 1 < last and nxt[3]:  # the next step's first example stages during the scale refresh
            stager.prime(nxt[3][0])
        if math.isfinite(gn):
            if pager:
                split["paged"] = dict(pager.stats, saver_busy=saver.busy,  # host contention in the page window
                                      threads=threading.active_count())
            if cfg.fake_quant and (step + 1) % max(1, cfg.qat_refresh_every) == 0:
                drift = qat.refresh(model, drift=cfg.qat_drift)
                sync()
                split["qat_refresh"], t = round(time.time() - t, 2), time.time()
                if ctx is not None:  # each rank measured its own experts: shares over all of them
                    d = drift or {}
                    flips, codes, moved, count, zero = ctx.sums([
                        d.get("code_flip_share", 0) * d.get("codes", 0), d.get("codes", 0),
                        d.get("scale_changed_share", 0) * d.get("scales", 0), d.get("scales", 0),
                        qat.QAT["zero_experts"]])
                    qat.QAT["zero_experts"] = int(zero)
                    if codes:
                        rels = ctx.gather_rows([d.get("scale_rel_max", 0.0)])
                        drift = {"code_flip_share": flips / codes, "scale_changed_share": moved / max(1, count),
                                 "scale_rel_max": max(r[0] for r in rels), "codes": int(codes), "scales": int(count)}
                if drift:
                    split["qat_drift"] = drift
                if qat.QAT["zero_experts"]:  # int3served: all-zero experts (scale_2 1, grid 0; serving's rtn: NaN)
                    split["qat_zero_experts"] = qat.QAT["zero_experts"]
        opt.zero_grad(set_to_none=True)
        if ckpt_possible(cfg, step + 1, time.time() - began, time.time() - last_ckpt):
            saver.prepare(model, head, opt)  # once: the snapshot buffer ahead of the first save, off the loop
        split["prepare"] = round(time.time() - t, 2)
        step += 1
        if ctx is not None and cfg.replica_check_every and step % cfg.replica_check_every == 0:
            bad = shard.check_replicas(rep, ctx)  # the replicated weights must stay bit-identical on every rank
            split["replicas_ok"] = not bad
            if bad:
                event(out, event="replica_mismatch", step=step, params=[names[id(rep[i])] for i in bad[:20]], count=len(bad))
                raise RuntimeError(f"step {step}: {len(bad)} replicated parameters differ between ranks")
        ended = time.time()
        seconds = ended - began
        util, vram, n_util = metrics.take()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if str(device).startswith("cuda") else None
        tokens = sum(e["input_ids"].shape[1] for e in exs)
        lens = [(int(e["input_ids"].shape[1]), min(int(e.get("tail") or cfg.window), cfg.window,
                                                     int(e["input_ids"].shape[1]))) for e in exs]
        window_tokens = sum(w for _, w in lens)
        shape = dict(prefix_tokens=sum(t - w for t, w in lens), mtp_rows=rows,  # mfu: (query, key) pairs per pass
                     prefix_pairs=sum((t - w) ** 2 // 2 for t, w in lens),
                     window_pairs=sum((t * t - (t - w) ** 2) // 2 for t, w in lens))
        n_ex, ranks, empties = len(exs), {}, EMPTY["n"]
        if ctx is not None:  # whole-step sums; per-rank time and memory beside them
            empties = int(ctx.sums([empties])[0])
            keys = ("prefix_tokens", "prefix_pairs", "window_pairs")
            tokens, window_tokens, n_ex, *vals = [int(x) for x in ctx.sums([tokens, window_tokens, n_ex,
                                                                             *(shape[k] for k in keys)])]
            shape.update(zip(keys, vals))
            spent = ctx.times()
            per = ctx.gather_rows([seconds, examples_s, -1 if peak is None else peak, len(exs), spent.get("gather", 0.0),
                                   spent.get("scatter", 0.0), spent.get("dense", 0.0), host_peak_gib()])
            ranks = {"ranks": {"examples_s": [round(r[1], 1) for r in per], "examples": [int(r[3]) for r in per],
                               "peak_gib": [None if r[2] < 0 else round(r[2], 2) for r in per],
                               "gpu_util": [u for u, _ in metrics.per_card] if cards > 1 else None,
                               "vram_gib": [v for _, v in metrics.per_card] if cards > 1 else None,
                               "comm": dict(ctx.stats), "host_peak_gib": [round(r[7], 1) for r in per],
                               # collective seconds on the compute stream: all-gather, reduce-scatter, dense all-reduce
                               "gather_s": [round(r[4], 3) for r in per], "scatter_s": [round(r[5], 3) for r in per],
                               "dense_s": [round(r[6], 3) for r in per]}}
            peak = None if peak is None else max(r[2] for r in per)
        row = dict(loss=ce / max(1, n_step), distill_kl=tkl / max(1, n_step), mtp_kl=kl / max(1, rows), accept1=hits / max(1, rows), grad_norm=gn, lr=lr,
                   reply_tokens=n_step, examples=n_ex, seconds=round(seconds, 1), data_wait=round(waited, 2),
                   tok_s=round(tokens / seconds, 1), util=util, vram_gib=vram, util_n=n_util,
                   examples_s=round(examples_s, 1), update_s=round(seconds - examples_s, 1), gap_s=round(gap_s, 1),
                   gap_util=gap_util, ctx_tokens=tokens, window_tokens=window_tokens, **shape,
                   alloc={k: v - alloc0.get(k, 0) for k, v in alloc_counts().items()}, update_split=split,
                   peak_gib=None if peak is None else round(peak, 2), skipped=not math.isfinite(gn), empty_examples=empties, **ranks)
        event(out, event="step", step=step, **row)
        waited = next_wait
        metrics.row(**{"train.step": step, "train.loss": row["loss"], "train.distill_kl": row["distill_kl"],
                       "train.grad_norm": gn, "train.lr": lr,
                       "train.tok_s": row["tok_s"], "train.step_s": row["seconds"], "train.peak_gib": row["peak_gib"],
                       "train.data_wait_s": row["data_wait"], "train.mtp_kl": row["mtp_kl"],
                       "train.util": util, "train.vram_gib": vram, "train.gap_s": row["gap_s"],
                       "train.accept1": row["accept1"]})
        if first:
            event(out, event="boot", step=step, first_step=round(seconds, 1), total=round(time.time() - t0, 1))
            first = False
        if (cfg.eval_steps and step % cfg.eval_steps == 0) or step in cfg.eval_at or fresh.get("eval") or step == last:
            ev = evaluate(cfg, model, head, box, held, stager)
            event(out, event="eval", step=step, set="slice", examples=n_held, **ev)
            metrics.row(**{"train.step": step, "train.heldout_loss": ev["heldout_loss"], "train.accept_len3": ev["accept_len3"]})
        ev = full_eval(step)
        arc2_eval(step)
        if cfg.eval_full_lead:
            full_prefetch(step)
        if ev is not None:
            metrics.row(**{"train.step": step, "train.heldout_full_loss": ev["heldout_loss"],
                           "train.heldout_full_kl": ev["heldout_kl"]})
        final = bool(fresh.get("stop")) or step == last
        due = ((cfg.ckpt_steps and step % cfg.ckpt_steps == 0) or time.time() - last_ckpt >= cfg.ckpt_seconds
               or pending)
        if final and step == last < cfg.total_steps and not cfg.stop_ckpt:
            final = due = False
        save = bool(final or step in cfg.ckpt_pin or (due and not saver.busy))  # a pinned step waits for a busy writer
        if ctx is not None:  # rank 0's clock and writer decide for every rank
            save = ctx.bcast(save)
        if save:
            path = saver.save(model, head, opt, step, cursor)
            last_ckpt = time.time()
            event(out, event="ckpt", step=step, path=str(path), tag=pending, background=saver.busy)
            pending = None
            if final:
                saver.wait()
        if fresh.get("stop"):
            event(out, event="stopped", step=step)
            break
    saver.wait()
    if boundary["thread"] is not None:
        boundary["thread"].join()
    if loader is not None:
        loader.close()
    stager.close()
    metrics.stop()
    if guard is not None:
        guard.end("ok")
    if ctx is not None:
        ctx.barrier()  # every rank's last write is done before rank 0 reports the run done
    event(out, event="done", step=step)
    committed()
    atexit.unregister(at_exit)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    train(Config.load(a.config), device=a.device)
