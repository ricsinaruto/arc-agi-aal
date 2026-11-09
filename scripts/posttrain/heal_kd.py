"""Healing trainer for prune-then-heal (SFT arm, 16 Oct; design approved by ).

Full-parameter knowledge distillation of a cut Flash student (layers dropped, experts pruned, top-k lowered; later
the hidden width in whole 128-blocks) against the uncut teacher's precomputed top-k log-probabilities on recorded,
token-exact play.

- ``cut_state_dict``: the student's tensors from the teacher's (BF16 source), renumbered, and its text config.
- ``teacher_topk``: the teacher's top-``k`` log-probs and the tail's log-mass at every position of a sequence
  (windows over a no-grad cache, the vocabulary in chunks).
- ``kd_loss``: forward KL from the teacher's top-k + tail bucket to the student's full softmax, chunked over positions
  under activation checkpointing (no [T, vocab] tensor is kept), plus an optional LM term on labelled positions.
- ``train_sequence``: one sequence in windows; each window's prefix is a no-grad cache, the window itself is trained.
- FSDP2 (``fully_shard`` per decoder layer, fp32 sharded parameters, BF16 compute) over ``torchrun`` processes.

Mode ``tiny`` (CPU, gloo; ``torchrun --nproc-per-node 2 heal_kd.py tiny --out DIR``) is the check of this file: a
random tiny Qwen4-Exp teacher, the student cut from it, synthetic sequences; it checks that the windowed loss equals
the whole-sequence loss, that two FSDP ranks give the single-process loss and gradient norm, and that the loss falls.
"""

import argparse
import itertools
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

LAYER = re.compile(r"^(model\.language_model\.layers\.)(\d+)(\..*)$")
EXPERTS = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
ROUTER = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
PER_EXPERT = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.)(\d+)(\..*)$")  # save_pretrained's form


def event(out, **row):
    row = {"t": round(time.time(), 1), **row}
    print(json.dumps(row), flush=True)
    if out is not None:
        with open(Path(out) / "events.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    return row


# ---- idle-GPU cuts (trainer MFU S4, 21 Oct; every flag off by default, numerics unchanged with each on) --------------
# A host read of a GPU value (.item(), .tolist(), float(t), torch.equal, a pageable copy) drains the launch queue: the
# card idles until the host has queued the next kernels, and the other rank waits for it in its next collective.
# HEAL_HOST_TIMERS=1: seconds per host phase of each step ("host" event, both ranks); before each known sync point it
#   synchronizes and books the drain as wait:<site>; also the slot padding share (rank imbalance of lockstep slots).
# HEAL_LAZY_STATS=1: no host read per window or slot: KD sums stay on the card (summed over ranks once per step),
#   positions are counted on the host copies, the M-RoPE image index is a host tensor; one control exchange per slot.
# HEAL_HOST_GROUP=1: the per-slot control exchanges (stop flag, slot positions and call plans) over a gloo group.
# HEAL_MASK_HOST=1: rl1's plain-causal mask check once per mask tensor with one host read for the check and the
#   largest count (QSA uses it; was 2 reads x 12 layers x forward + replay per window); no all-ones attention_mask is
#   built (its all() check was one more read per forward).
# HEAL_PLE_TABLE=1: the n-gram (PLE) rows of a whole sequence looked up once on the host (pinned), sliced per call
#   with an async copy; was a GPU->host id copy, a host gather and a pageable copy at every forward and replay.
# HEAL_PREFETCH=1: the next slot's teacher files read and prepared on the host (loop cut, PLE rows, pinned) by a
#   background thread while this slot trains; the device copies are non-blocking.

import contextlib  # noqa: E402

_NULL = contextlib.nullcontext()


def s4(name):
    return os.environ.get(f"HEAL_{name}") == "1"


class HostTimers:
    """``phase(name)``: a top-level host phase of the main thread, booked exclusive of the leaves inside it;
    ``leaf(name)``: a lookup or sync site inside a phase (any thread: checkpoint replays run on autograd's);
    ``wait(name)``: under the flag, a synchronize right before an existing sync point, booked as ``wait:name`` (the
    card's backlog the host waited for there)."""

    def __init__(self):
        self.on = s4("HOST_TIMERS")
        self.cuda = torch.cuda.is_available()
        self.reset()

    ALLOC = ("num_alloc_retries", "num_sync_all_streams", "num_device_alloc", "num_device_free")

    def reset(self):
        self.s, self.n, self.leaves, self.t0, self.slots = {}, {}, 0.0, time.perf_counter(), []
        self.alloc = self._alloc()
        self.raised = {}  # c7n: GiB each phase / leaf raised the device's peak allocation by (per rank)
        self.alloc0 = torch.cuda.memory_allocated() / 2 ** 30 if self.on and self.cuda else None

    def _peak(self):
        return torch.cuda.max_memory_allocated() if self.cuda else 0

    def _raise(self, name, p0):
        d = (self._peak() - p0) / 2 ** 30
        if d > 0.01:
            self.raised[name] = self.raised.get(name, 0.0) + d

    def _alloc(self):
        """The caching allocator's counters (a cudaMalloc / cudaFree or a retry after freeing the cache synchronizes
        the device: the L40S proxy spent 1.3 s of a 6.8 s step blocked in torch.empty)."""
        if not (self.on and self.cuda):
            return {}
        stats = torch.cuda.memory_stats()
        return {k: stats.get(k, 0) for k in self.ALLOC}

    def add(self, name, seconds):
        self.s[name] = self.s.get(name, 0.0) + seconds
        self.n[name] = self.n.get(name, 0) + 1

    @contextlib.contextmanager
    def _phase(self, name):
        t, leaves, p0 = time.perf_counter(), self.leaves, self._peak()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t - (self.leaves - leaves))
            self._raise(name, p0)

    @contextlib.contextmanager
    def _leaf(self, name):
        t, p0 = time.perf_counter(), self._peak()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self.add(name, dt)
            self.leaves += dt
            self._raise(name, p0)

    def phase(self, name):
        return self._phase(name) if self.on else _NULL

    def leaf(self, name):
        return self._leaf(name) if self.on else _NULL

    def wait(self, name):
        if self.on and self.cuda:
            t = time.perf_counter()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            self.add("wait:" + name, dt)
            self.leaves += dt

    def slot(self, tokens, windows):
        """One lockstep slot: tokens and windows per rank."""
        if self.on:
            self.slots.append((list(tokens), list(windows)))

    def report(self):
        wall = time.perf_counter() - self.t0
        real = sum(sum(t) for t, _ in self.slots)
        padded = sum(max(t) * len(t) for t, _ in self.slots)
        return {"wall_s": round(wall, 3), "other_s": round(wall - sum(self.s.values()), 3),
                "s": {k: round(v, 3) for k, v in sorted(self.s.items(), key=lambda kv: -kv[1])}, "n": dict(self.n),
                "alloc": {k: v - self.alloc.get(k, 0) for k, v in self._alloc().items()},
                "slots": len(self.slots), "pad_share": round(1 - real / padded, 4) if padded else None,
                "slot_tokens": [t for t, _ in self.slots], "slot_windows": [w for _, w in self.slots],
                "peak_raise_gib": {k: round(v, 2) for k, v in sorted(self.raised.items(), key=lambda kv: -kv[1])},
                "alloc_start_gib": None if self.alloc0 is None else round(self.alloc0, 2)}


HOST = HostTimers()
_CTL = {}


def pg_timeout():
    """The process groups' collective timeout (c7q 25 Oct: rank 0 waited for fresh stream chunks, allowed
    HEAL_STREAM_STALL = 1800 s, while ranks 1-2 sat in take()'s gather and the 600 s NCCL watchdog killed the run):
    above the stream stall when streaming, else torch's 600 s."""
    import datetime

    floor = float(os.environ.get("HEAL_PG_TIMEOUT_FLOOR") or 600)  # floor / grace: tests shrink them
    grace = float(os.environ.get("HEAL_PG_TIMEOUT_GRACE") or 300)
    stall = float(os.environ.get("HEAL_STREAM_STALL") or 3600) if os.environ.get("HEAL_STREAM") else 0.0
    return datetime.timedelta(seconds=max(floor, stall + grace if stall else floor))


def ctl_group():
    """HEAL_HOST_GROUP=1: the gloo group of the control exchanges (None: the default group). Created once, by every
    rank in the same order (run_train calls it right after init)."""
    import torch.distributed as dist

    if not s4("HOST_GROUP") or dist.get_backend() == "gloo":
        return None
    if "group" not in _CTL:
        _CTL["group"] = dist.new_group(backend="gloo", timeout=pg_timeout())
    return _CTL["group"]


def ctl_gather(obj):
    import torch.distributed as dist

    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, obj, group=ctl_group())
    return every


def as64(x, device):
    """A KD sum as a float64 card tensor (HEAL_LAZY_STATS keeps them there until the step's one reduction)."""
    return x.double() if isinstance(x, torch.Tensor) else torch.tensor(float(x), dtype=torch.float64, device=device)


# PLE rows (HEAL_PLE_TABLE): {layer_idx: [1, n, D]} for the current runner call, set by dp_sequence; None: the
# model's own lookup
PLE = {"rows": None, "proxies": {}}


def ple_modules(model):
    core = text_model(model).language_model
    return [layer.ple.ple_embedding for layer in core.layers if getattr(layer, "ple", None) is not None]


def install_ple(model):
    """Wrap each n-gram embedding's forward: with rows set for the call, the cache bookkeeping of the original (the
    window's last ids for the next one) and the rows; else the original (timed under HEAL_HOST_TIMERS)."""
    import types

    for mod in ple_modules(model):
        if getattr(mod, "_s4_original", None) is not None:
            continue
        original = mod.forward
        mod._s4_original = original

        def forward(self, input_ids, past_key_values, original=original):
            rows = PLE["rows"]
            got = rows.get(self.layer_idx) if rows is not None else None
            if got is None:
                HOST.wait("ple")
                with HOST.leaf("ple_lookup"):
                    return original(input_ids, past_key_values)
            if got.shape[1] != input_ids.shape[1]:
                raise RuntimeError(f"PLE rows for {got.shape[1]} tokens, call of {input_ids.shape[1]}")
            if past_key_values is not None:  # Qwen4ExpTextNGramEmbedding.forward's cache update, unchanged
                ids = input_ids.long()
                if not past_key_values.has_previous_state(self.layer_idx, state_idx=2) and ids.shape[1] < self.context_len:
                    ids = F.pad(ids, (self.context_len - ids.shape[1], 0), value=self.eos_token_id)
                past_key_values.update_conv_state(ids, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len)
            return got

        mod.forward = types.MethodType(forward, mod)


@torch.no_grad()
def ple_rows(model, ids, pin=False):
    """Every PLE layer's n-gram rows of a whole sequence (``ids`` [T] on the host) by the model's own forward on a
    host copy of the module (its small id buffers on the host, the table shared): a sequence's rows equal its
    windows' (the cache keeps the last ngram_size - 1 ids, the eos rule looks back no further)."""
    import copy
    import weakref

    out = {}
    for mod in ple_modules(model):
        got = PLE["proxies"].get(id(mod))
        proxy = got[1] if got is not None and got[0]() is mod else None
        if proxy is None:
            proxy = copy.copy(mod)
            object.__setattr__(proxy, "_buffers", {k: (None if v is None else v.cpu()) for k, v in mod._buffers.items()})
            PLE["proxies"][id(mod)] = (weakref.ref(mod), proxy)
        rows = type(mod).forward(proxy, ids.reshape(1, -1).long().cpu(), None)
        out[mod.layer_idx] = rows.pin_memory() if pin else rows
    return out


def ple_slice(table, start, stop, device):
    return None if table is None else {i: t[:, start:stop].to(device, non_blocking=True) for i, t in table.items()}


def mask_host_on():
    return s4("MASK_HOST")


def plain_causal_host(original):
    """rl1_trainer.plain_causal once per mask tensor (HEAL_MASK_HOST): the original's check (rows are prefixes) and
    the largest count in one host read; the count rides on the result for rl1_trainer.qsa_select (``_heal_max``).
    The flag off: S2's memo under HEAL_COMPILE, else the original (timed under HEAL_HOST_TIMERS)."""
    import weakref

    memo = {}
    s2 = plain_causal_memo(original)

    def plain_causal(attention_mask):
        if not mask_host_on():
            HOST.wait("mask")
            with HOST.leaf("mask_check"):
                return s2(attention_mask) if compile_mode() else original(attention_mask)
        key = id(attention_mask)
        got = memo.get(key)
        if got is not None and got[0]() is attention_mask and got[1] == attention_mask._version:
            return got[2]
        for k in [k for k, v in memo.items() if v[0]() is None]:
            del memo[k]
        visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        counts = None
        if visible.shape[0] == 1:
            import rl1_trainer as rl1

            counts, same = rl1.causal_counts(visible)  # row chunks (c7m: a whole-mask int64 sum OOM'd)
            HOST.wait("mask")
            with HOST.leaf("mask_check"):
                ok, top = torch.stack([same.long(), counts.max()]).tolist()
            if ok:
                counts._heal_max = int(top)
            else:
                counts = None
        memo[key] = (weakref.ref(attention_mask), attention_mask._version, counts)
        return counts

    plain_causal.memo = memo
    return plain_causal


def install_mask_host():
    """``plain_causal_host`` over rl1_trainer.plain_causal (fast_attention; the flags are read per call). It carries
    ``memo``, so S2's install_compile leaves it in place."""
    import rl1_trainer as rl1

    if not getattr(rl1.plain_causal, "_s4", False):
        rl1.plain_causal = plain_causal_host(rl1.plain_causal)
        rl1.plain_causal._s4 = True


class Prefetch:
    """HEAL_PREFETCH=1: ``prep(path, general)`` (host only: the file read, the loop cut, the PLE rows, pinned) for the
    files the next slot may take, in one background thread; ``get`` waits for a queued file or prepares it here."""

    def __init__(self, prep, keep=4):
        from concurrent.futures import ThreadPoolExecutor

        self.pool, self.prep, self.jobs, self.keep = ThreadPoolExecutor(1), prep, {}, keep
        self.hits = self.misses = 0

    def want(self, key):
        if key[0] is None or key in self.jobs:
            return
        self.jobs[key] = self.pool.submit(self.prep, *key)
        while len(self.jobs) > self.keep:
            self.jobs.pop(next(iter(self.jobs))).cancel()

    def get(self, key):
        job = self.jobs.pop(key, None)
        if job is None:
            self.misses += 1
            return self.prep(*key)
        self.hits += 1
        return job.result()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


# ---- the student -----------------------------------------------------------------------------------------------

def cut_text_config(text, *, keep_layers, experts=None, top_k=None):
    """The student's text config: ``keep_layers`` (teacher layer ids in order), ``experts`` kept per layer (same count
    in every layer), ``top_k``."""
    text = dict(text)
    text["layer_types"] = [text["layer_types"][i] for i in keep_layers]
    text["num_hidden_layers"] = len(keep_layers)
    old_to_new = {old: new for new, old in enumerate(keep_layers)}
    text["ple_layer_ids"] = [old_to_new[i - 1] + 1 for i in text.get("ple_layer_ids") or [] if i - 1 in old_to_new]
    if experts is not None:
        text["num_experts"] = len(next(iter(experts.values())))
    if top_k is not None:
        text["num_experts_per_tok"] = int(top_k)
    return text


def cut_state_dict(state, *, keep_layers, experts=None):
    """Teacher tensors -> student tensors: dropped layers removed, kept ones renumbered, routed experts and router rows
    sliced to ``experts[layer]`` (teacher layer ids, kept expert ids). Everything else is copied as is."""
    old_to_new = {old: new for new, old in enumerate(keep_layers)}
    out = {}
    for name, tensor in state.items():
        m = LAYER.match(name)
        if m:
            old = int(m.group(2))
            if old not in old_to_new:
                continue
            if experts is not None and (EXPERTS.match(name) or ROUTER.match(name)):
                tensor = tensor[torch.tensor(experts[old])]
            name = f"{m.group(1)}{old_to_new[old]}{m.group(3)}"
        out[name] = tensor
    return out


def build_student(teacher, *, keep_layers, experts=None, top_k=None):
    from transformers import AutoConfig, AutoModelForMultimodalLM

    data = teacher.config.to_dict()
    data["text_config"] = cut_text_config(data["text_config"], keep_layers=keep_layers, experts=experts, top_k=top_k)
    config = AutoConfig.for_model(**data)
    student = AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)
    state = cut_state_dict(teacher.state_dict(), keep_layers=keep_layers, experts=experts)
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"student load: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return student


def cut_checkpoint(src, dst, *, drop, prune_file=None, top_k=None, shard_gib=5.0, log=None, width_keep=None):
    """The student checkpoint on disk from the teacher's BF16 shards, one shard at a time: layers in ``drop`` removed
    (the rest renumbered), routed experts and router rows kept per ``prune_file`` (format arc-expert-prune-v1:
    ``pruned`` per layer), ``top_k``. MTP tensors (``mtp.*``) are copied unchanged; the config records their
    expert count. ``width_keep``: hidden channels kept (``width_cut_state_dict``, every tensor, MTP included).
    Tokenizer and processor files are copied."""
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    keep = [i for i in range(text["num_hidden_layers"]) if i not in set(drop)]
    experts = None
    if prune_file:
        spec = json.loads(Path(prune_file).read_text())
        assert spec["format"] == "arc-expert-prune-v1", spec["format"]
        experts = {int(layer): [e for e in range(spec["experts"]) if e not in set(pruned)]
                   for layer, pruned in spec["pruned"].items()}
        if len({len(v) for v in experts.values()}) != 1:
            raise ValueError("unequal expert counts per layer")
    config["text_config"] = cut_text_config(text, keep_layers=keep, experts=experts, top_k=top_k)
    config["text_config"]["mtp_num_experts"] = text["num_experts"]
    config["text_config"]["mtp_num_experts_per_tok"] = text["num_experts_per_tok"]
    config["student_cut"] = {"drop": sorted(drop), "prune_file": str(prune_file) if prune_file else None,
                             "top_k": top_k, "source_layers": keep}
    hidden, count = text["hidden_size"], text["hc_count"]
    if width_keep is not None:
        width_keep = sorted(width_keep)
        config["text_config"]["hidden_size"] = len(width_keep)
        if config.get("vision_config"):
            config["vision_config"]["out_hidden_size"] = len(width_keep)
        config["student_cut"]["width_keep"] = width_keep
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    old_to_new = {old: new for new, old in enumerate(keep)}
    out_map, buffer, size, part, total = {}, {}, 0, 0, 0

    def flush():
        nonlocal buffer, size, part
        if not buffer:
            return
        part += 1
        name = f"model-{part:05d}.safetensors"
        save_file(buffer, str(dst / name), metadata={"format": "pt"})
        for key in buffer:
            out_map[key] = name
        buffer, size = {}, 0

    for shard in sorted(set(index.values())):
        with safe_open(str(src / shard), framework="pt") as f:
            for name in f.keys():
                orig = name
                m = LAYER.match(name)
                if m:
                    old = int(m.group(2))
                    if old not in old_to_new:
                        continue
                    per = PER_EXPERT.match(name)
                    if experts is not None and per:
                        if int(per.group(2)) not in experts[old]:
                            continue
                        name = f"{per.group(1)}{experts[old].index(int(per.group(2)))}{per.group(3)}"
                        m = LAYER.match(name)
                    tensor = f.get_tensor(orig)
                    if experts is not None and (EXPERTS.match(name) or ROUTER.match(name)):
                        tensor = tensor[torch.tensor(experts[old])].contiguous()
                    name = f"{m.group(1)}{old_to_new[old]}{m.group(3)}"
                else:
                    tensor = f.get_tensor(orig)
                if width_keep is not None:
                    tensor = width_cut_state_dict({name: tensor}, hidden=hidden, hc_count=count, keep=width_keep)[name]
                buffer[name] = tensor
                size += tensor.numel() * tensor.element_size()
                total += tensor.numel()
                if size >= shard_gib * 2 ** 30:
                    flush()
        event(log, shard=shard, params_so_far=total)
    flush()
    names = {k: f"model-{int(v[6:11]):05d}-of-{part:05d}.safetensors" for k, v in out_map.items()}
    for i in range(1, part + 1):
        os.replace(dst / f"model-{i:05d}.safetensors", dst / f"model-{i:05d}-of-{part:05d}.safetensors")
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_parameters": total},
                                                                  "weight_map": names}, indent=1))
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for path in src.iterdir():
        if path.is_file() and not path.name.startswith("model") and path.name != "config.json":
            shutil.copy(path, dst / path.name)
    return event(log, cut="done", parameters=total, shards=part, layers=len(keep),
                 experts=None if experts is None else len(next(iter(experts.values()))), top_k=top_k)


# ---- forward pieces -------------------------------------------------------------------------------------------------

def text_model(model):
    return model.model


def hidden_window(model, ids, start, stop, cache, use_cache=None, piece=None):
    """Final hidden states of ids[:, start:stop] given ``cache`` (the prefix ids[:, :start]); ``piece(start, stop)``
    (flash_qlora_job.positions_and_images): the slice's M-RoPE positions and image features."""
    # HEAL_MASK_HOST: no all-ones mask (the same causal mask; create_recurrent_attention_mask's all() read skipped)
    out = text_model(model)(input_ids=ids[:, start:stop],
                            attention_mask=None if mask_host_on() else torch.ones_like(ids[:, :stop]),
                            past_key_values=cache, use_cache=cache is not None if use_cache is None else use_cache,
                            **(piece(start, stop) if piece is not None else {}))
    return out.last_hidden_state, out.past_key_values


@torch.no_grad()
def extend_cache(model, ids, start, stop, cache, chunk):
    """The cache extended over ids[:, start:stop] without gradients (eval mode: checkpointed layers keep the cache)."""
    from transformers import DynamicCache

    was = model.training
    model.eval()
    try:
        if cache is None:
            cache = DynamicCache(config=model.config.text_config)
        for a in range(start, stop, chunk):
            b = min(a + chunk, stop)
            out = text_model(model)(input_ids=ids[:, a:b], attention_mask=torch.ones_like(ids[:, :b]),
                                    past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    finally:
        model.train(was)
    return cache


@torch.no_grad()
def teacher_topk(model, ids, *, k, window, chunk=4096, vocab_chunk=1024):
    """Teacher top-k log-probs [T, k] (fp16), their token ids [T, k] (int32) and the tail log-mass [T] (fp16) at every
    position of ids [1, T] (the distribution of the next token)."""
    from transformers import DynamicCache

    model.eval()
    total = ids.shape[1]
    tops, idxs, tails = [], [], []
    cache = DynamicCache(config=model.config.text_config)  # grows with each window (no-grad eval pass)
    for start in range(0, total, window):
        stop = min(start + window, total)
        hidden, cache = hidden_window(model, ids, start, stop, cache)
        for a in range(0, hidden.shape[1], vocab_chunk):
            logits = model.lm_head(hidden[:, a:a + vocab_chunk]).float()[0]
            logp = torch.log_softmax(logits, dim=-1)
            top, idx = torch.topk(logp, k, dim=-1)
            mass = top.exp().sum(-1).clamp(max=1 - 1e-6)
            tops.append(top.half())
            idxs.append(idx.int())
            tails.append(torch.log1p(-mass).half())
    return torch.cat(tops), torch.cat(idxs), torch.cat(tails)


# The trained divergence (OPD, 18 Oct): forward KL(teacher || student) and reverse KL(student || teacher), both over
# the teacher's top-k tokens plus one tail bucket; the reported "kd" is always the forward one (comparable across runs).
LOSS_MIX = {"forward": 1.0, "reverse": 0.0}


# HEAL_FORCE_IDS (25 Oct, v32 play collapse; off by default): token ids the teacher scores at every row beside its
# top-k ("force_ids" / "force_logp" in a piece), e.g. "248046,248069,248058" = <|im_end|>, </think>, <tool_call>. The
# loader appends them as extra columns (NaN where the id is already in the row's top-k), so a structural token the
# teacher ranks outside its top-k gets its own s log(s/t) term instead of sharing the lumped tail bucket (v32: the
# student's in-think <|im_end|> rose to p ~0.4% while the teacher holds it at ~e^-19, outside its top-128 at 84-89% of
# think rows).
FORCE_IDS = tuple(int(x) for x in re.split(r"[,+\s]+", os.environ.get("HEAL_FORCE_IDS") or "") if x)  # "," or "+"


def bucket_kl(s_top, t_top, t_tail, masked=None):
    """(forward, reverse) KL per position over the teacher's top-k ids + the tail bucket; s_top: student log-probs
    gathered at the teacher's ids, t_top / t_tail: teacher log-probs and tail log-mass. ``masked`` (default: on with
    HEAL_FORCE_IDS): NaN teacher columns (a forced id already in the row's top-k) count for neither side."""
    if masked if masked is not None else bool(FORCE_IDS):
        valid = ~torch.isnan(t_top)
        t_top = torch.where(valid, t_top.float(), torch.zeros((), device=t_top.device))  # no NaN in any branch
        s_p, t_p = s_top.exp() * valid, t_top.exp() * valid
        s_tail = torch.log1p(-s_p.sum(-1).clamp(max=1 - 1e-6))
        t_tail = t_tail.float()
        forward = (t_p * (t_top - s_top)).sum(-1) + t_tail.exp() * (t_tail - s_tail)
        reverse = (s_p * (s_top - t_top)).sum(-1) + s_tail.exp() * (s_tail - t_tail)
        return forward, reverse
    s_mass = s_top.exp().sum(-1).clamp(max=1 - 1e-6)
    s_tail = torch.log1p(-s_mass)
    t_top, t_tail = t_top.float(), t_tail.float()
    forward = (t_top.exp() * (t_top - s_top)).sum(-1) + t_tail.exp() * (t_tail - s_tail)
    reverse = (s_top.exp() * (s_top - t_top)).sum(-1) + s_tail.exp() * (s_tail - t_tail)
    return forward, reverse


# HEAL_THINK_EOS_MASK (25 Oct, EOS-driver step A; off by default): "<eos>+<think_end>" ids, e.g. "248046+248069". On
# every row inside an open think block of a reply (the predicted token is a reply token and the reply has not yet
# emitted </think>), the teacher's <|im_end|> is set to EOS_FLOOR and its other columns and tail renormalised: EOS is
# invalid before </think> (step A: the unpruned teacher puts it at median log p -2.49 at the student's in-think EOS
# rows). Needs the EOS log p per row: a top-k column or a HEAL_FORCE_IDS column; rows without one are left as scored.
THINK_EOS = tuple(int(x) for x in re.split(r"[,+\s]+", os.environ.get("HEAL_THINK_EOS_MASK") or "") if x)
EOS_FLOOR = -30.0
THINK_ACC = {}  # rows masked / rows in think without an EOS column (the loader's counts)


def think_rows(ids, reply_tokens, think_end):
    """Rows r (predicting token r+1) inside an open think block: token r+1 is a reply token and no ``think_end`` sits
    among that reply's tokens before it (the row predicting </think> itself is still open)."""
    rep = reply_tokens.bool()
    is_end = ((ids == think_end) & rep).long()
    excl = torch.cumsum(is_end, 0) - is_end  # think ends strictly before each position
    start = rep & ~torch.cat([torch.zeros(1, dtype=torch.bool), rep[:-1]])
    last = torch.cummax(torch.where(start, excl, torch.full_like(excl, -1)), 0).values  # the span start's count
    open_ = rep & (excl == last)
    rows = torch.zeros_like(rep)
    rows[:-1] = open_[1:]
    return rows


def mask_think_eos(top, idx, tail, rows, eos, t_tok=None, nxt=None):
    """``rows`` [T]: the teacher's ``eos`` to EOS_FLOOR there and the rest renormalised (top columns, tail; ``t_tok``
    as well, for ``nxt`` [T] the realised next ids). Returns (top, tail, t_tok, masked rows, rows without an EOS
    column)."""
    topf = top.float()
    valid = ~torch.isnan(topf)
    col = (idx.long() == eos) & valid
    has = col.any(-1)
    m = rows & has
    p = (torch.nan_to_num(topf, nan=float("-inf")).exp() * col).sum(-1).clamp(max=1 - 1e-6)
    scale = torch.where(m, torch.log1p(-p), torch.zeros_like(p))
    out = torch.where(m[:, None] & valid & ~col, topf - scale[:, None], topf)
    out = torch.where(m[:, None] & col, torch.full_like(out, EOS_FLOOR), out)
    tail2 = torch.where(m, tail.float() - scale, tail.float())
    if t_tok is not None:
        t_tok = torch.where(m & (nxt == eos), torch.full_like(t_tok, EOS_FLOOR), t_tok - scale)
    return out.to(top.dtype), tail2.to(tail.dtype), t_tok, int(m.sum()), int((rows & ~has).sum())


def merge_forced(top, idx, f_ids, f_logp):
    """Teacher rows top [T, k] / idx [T, k] plus the forced ids f_ids [m] with their log p f_logp [T, m]: [T, k + m]
    columns (NaN log p where the id is already in the row's top-k) and the tail log-mass left outside all of them."""
    f_ids = torch.as_tensor(f_ids, dtype=idx.dtype)
    dup = (idx.long()[:, :, None] == f_ids.long()[None, None, :]).any(1)  # [T, m]
    extra = torch.where(dup, torch.full_like(f_logp.float(), float("nan")), f_logp.float())
    top2 = torch.cat([top.float(), extra], 1)
    idx2 = torch.cat([idx, f_ids[None].expand(idx.shape[0], -1)], 1)
    mass = torch.nan_to_num(top2, nan=float("-inf")).double().exp().sum(-1).clamp(max=1 - 1e-6)
    return top2.to(top.dtype), idx2, torch.log1p(-mass).to(top.dtype)


def sampler_logprob(logits, tokens, temperature=1.0, top_k=0, top_p=1.0):
    """log b(x) of ``tokens`` [n] under vLLM-style sampling from ``logits`` [n, V]: temperature, then top-k, then top-p
    (the smallest set reaching ``top_p``), renormalised; a token outside the kept set gets -inf."""
    z = logits.float() / temperature
    if top_k and top_k < z.shape[-1]:
        z = z.masked_fill(z < z.topk(top_k, dim=-1).values[:, -1:], float("-inf"))
    if top_p < 1.0:
        sz, order = z.sort(-1, descending=True)
        probs = sz.softmax(-1)
        sz = sz.masked_fill(probs.cumsum(-1) - probs > top_p, float("-inf"))
        z = torch.full_like(z, float("-inf")).scatter(-1, order, sz)
    return torch.log_softmax(z, -1).gather(-1, tokens.long()[:, None])[:, 0]


def sampled_reverse_terms(s_tok, t_tok, b_tok, clip=2.0):
    """Sampled-token reverse KL (GKD; Thinking Machines on-policy distillation) on realised tokens x ~ b: the estimate
    w (log pS(x) - log pT(x)) of KL(pS || pT) and a surrogate whose gradient is its policy gradient
    w (log pS - log pT) grad log pS; w = min(pS(x) / b(x), clip), the truncated importance weight for rollouts sampled
    by an earlier (tempered, truncated) student."""
    w = torch.exp(s_tok.detach() - b_tok).clamp(max=clip)
    est = w * (s_tok.detach() - t_tok)
    return est, est * s_tok, w


SAMPLED_CLIP = float(os.environ.get("HEAL_SAMPLED_CLIP") or 2.0)
# HEAL_SAMPLED_MODE=topk: on the sampled (student) positions the trained term is the reverse KL over
# the teacher's top-k + tail bucket (bucket_kl), not the sampled-token estimate; w is logged only (arXiv 2603.25562)
SAMPLED_MODE = os.environ.get("HEAL_SAMPLED_MODE") or "token"
BEH_ACC = {}  # behavior pass (HEAL_SAMPLED): per-chunk log b(next token) rows, in position order
SAMP_ACC = {}  # sampled-token reverse KL sums (no-grad reads and the training log)


CLASS_ACC = {}  # held-out class split (HEAL_CLASS_SPLIT): summed forward KL and positions per reply class
REV_ACC = {}  # held-out reverse KL on reply positions (HEAL_HELDOUT_REVERSE; OPD co-primary metric, 21 Oct)
TRAIN_ACC = {}  # HEAL_OPD_LOG (22 Oct): the trained distillation objective per step (value on trained positions, the
# sampled estimate and weight on sampled positions); eager only, sums doubled by checkpoint recompute cancel in the means
READ_ACC = {}  # no-grad reads (22 Oct): the trained mix objective on scored positions (the fixed-batch gate under a
# reverse/forward mix) and the student's mass outside the teacher's top-k on reply positions (bucket-estimator audit)


def _kd_chunk(hidden, weight, t_top, t_idx, t_tail, labels, mask, lm_weight, reply=None, mix=None, classes=None,
              samp=None):
    logits = F.linear(hidden, weight).float()
    logp = torch.log_softmax(logits, dim=-1)
    if BEH_ACC.get("on") and labels is not None:  # behavior pass (no grad): log b of each position's next token
        BEH_ACC.setdefault("rows", []).append(sampler_logprob(logits.detach(), labels.clamp(min=0),
                                                              *BEH_ACC["sampler"]).cpu())
    s_top = logp.gather(-1, t_idx.long())
    forward, reverse = bucket_kl(s_top, t_top, t_tail)
    if classes is not None:  # [C, n] masks (no-grad reads only)
        CLASS_ACC["kl"] = CLASS_ACC.get("kl", 0) + (forward.detach()[None] * classes).sum(-1)
        CLASS_ACC["n"] = CLASS_ACC.get("n", 0) + classes.sum(-1)
    if REV_ACC.get("on") and reply is not None:  # no-grad reads only
        REV_ACC["kl"] = REV_ACC.get("kl", 0) + (reverse.detach() * reply).sum()
        REV_ACC["n"] = REV_ACC.get("n", 0) + reply.sum()
    mix = LOSS_MIX if mix is None else mix
    trained = mix["forward"] * forward + mix["reverse"] * reverse
    value = trained  # the objective's value (reads): the sampled term enters as its estimate, not its surrogate
    if samp is not None:  # (next ids, teacher log p of them, behavior log p of them, mask of the sampled positions)
        nxt, t_tok, b_tok, smask = samp
        sm = smask.float()
        s_tok = logp.gather(-1, nxt.clamp(min=0).long()[:, None])[:, 0]
        t_tok = torch.nan_to_num(t_tok.float(), nan=0.0, neginf=0.0)
        b_tok = torch.nan_to_num(b_tok.float(), nan=0.0, neginf=0.0)
        est, surrogate, w = sampled_reverse_terms(s_tok, t_tok, b_tok, clip=SAMPLED_CLIP)
        if SAMPLED_MODE == "topk":
            trained = trained + mix.get("sampled", 0.0) * reverse * sm
            value = value + mix.get("sampled", 0.0) * reverse.detach() * sm
        else:
            trained = trained + mix.get("sampled", 0.0) * surrogate * sm
            value = value + mix.get("sampled", 0.0) * est.detach() * sm
        if SAMP_ACC.get("on"):
            SAMP_ACC["est"] = SAMP_ACC.get("est", 0) + (est.detach() * sm).sum()
            SAMP_ACC["ratio"] = SAMP_ACC.get("ratio", 0) + ((s_tok.detach() - t_tok) * sm).sum()
            SAMP_ACC["w"] = SAMP_ACC.get("w", 0) + (w * sm).sum()
            SAMP_ACC["n"] = SAMP_ACC.get("n", 0) + sm.sum()
    if TRAIN_ACC.get("on"):
        TRAIN_ACC["value"] = TRAIN_ACC.get("value", 0) + (value.detach() * mask).sum()
        TRAIN_ACC["n"] = TRAIN_ACC.get("n", 0) + mask.sum()
        if samp is not None:
            TRAIN_ACC["est"] = TRAIN_ACC.get("est", 0) + (est.detach() * sm).sum()
            TRAIN_ACC["w"] = TRAIN_ACC.get("w", 0) + (w * sm).sum()
            TRAIN_ACC["sn"] = TRAIN_ACC.get("sn", 0) + sm.sum()
    if READ_ACC.get("on"):  # no-grad reads only
        READ_ACC["mix"] = READ_ACC.get("mix", 0) + (value.detach() * mask).sum()
        READ_ACC["n"] = READ_ACC.get("n", 0) + mask.sum()
        if reply is not None:
            READ_ACC["mix_reply"] = READ_ACC.get("mix_reply", 0) + (value.detach() * reply).sum()
            s_seen = s_top.detach().exp() * ~torch.isnan(t_top) if FORCE_IDS else s_top.detach().exp()
            out_mass = 1.0 - s_seen.sum(-1).clamp(max=1.0)
            READ_ACC["tail"] = READ_ACC.get("tail", 0) + (out_mass * reply).sum()
            out32 = 1.0 - s_top.detach()[:, :32].exp().sum(-1).clamp(max=1.0)  # top_ids are in teacher order
            READ_ACC["tail32"] = READ_ACC.get("tail32", 0) + (out32 * reply).sum()
            READ_ACC["nr"] = READ_ACC.get("nr", 0) + reply.sum()
    kd = (forward.detach() * mask).sum()
    kd_train = (trained * mask).sum()
    kd_reply = (forward.detach() * reply).sum() if reply is not None else torch.zeros((), device=hidden.device)
    lm = torch.zeros((), device=hidden.device)
    if labels is not None:
        valid = labels >= 0
        lm = -(logp.gather(-1, labels.clamp(min=0)[:, None])[:, 0] * valid).sum()
    return kd_train + lm_weight * lm, kd.detach(), lm.detach(), kd_reply


def kd_rows(h, start, stop):
    """HEAL_KD_ROWS (MFU-1): the window rows [start, stop) any loss term or read can see (mask, reply, labels >= 0,
    sampled, class rows), as window-local indices; None when off. Every other row adds exactly 0 to each sum and
    gradient, so the LM head and KD run on these rows only. Keep rows are found once per sequence (one host sync)."""
    if os.environ.get("HEAL_KD_ROWS") != "1" or BEH_ACC.get("on"):
        return None
    keep = getattr(h, "_kd_keep", None)
    if keep is None:
        m = h.mask.reshape(-1) != 0
        if getattr(h, "reply", None) is not None:
            m = m | (h.reply.reshape(-1) != 0)
        if getattr(h, "labels", None) is not None:
            m = m | (h.labels.reshape(-1) >= 0)
        if getattr(h, "samp", None) is not None:
            m = m | (h.samp[3].reshape(-1) != 0)
        if getattr(h, "classes", None) is not None:
            m = m | (h.classes != 0).any(0)
        idx = torch.nonzero(m).reshape(-1)
        h._kd_keep = keep = (idx, idx.cpu().numpy())
    idx, host = keep
    import numpy as np

    lo, hi = (int(x) for x in np.searchsorted(host, [start, stop]))
    if hi <= lo:  # no row: one inert row keeps the call's graph (and its FSDP collectives) as it was
        return torch.zeros(1, dtype=torch.long, device=idx.device)
    return idx[lo:hi] - start


def kd_loss(hidden, weight, t_top, t_idx, t_tail, mask, *, labels=None, lm_weight=0.0, chunk=1024, reply=None, mix=None,
            classes=None, lazy=False, samp=None, rows=None):
    """Summed KD (+ lm_weight x summed NLL on labels >= 0) over positions, chunked under checkpointing. Returns the
    loss, summed KD, summed NLL and summed KD on the ``reply`` positions (reported apart); ``lazy``: the sums as
    float64 card scalars (no host read). ``rows`` (kd_rows): score only these rows of the inputs."""
    from torch.utils.checkpoint import checkpoint

    if rows is not None:
        pick = lambda x: None if x is None else x.index_select(0, rows)  # noqa: E731
        hidden, t_top, t_idx, t_tail, mask = (x.index_select(0, rows) for x in (hidden, t_top, t_idx, t_tail, mask))
        labels, reply = pick(labels), pick(reply)
        classes = None if classes is None else classes.index_select(1, rows)
        samp = None if samp is None else tuple(x.index_select(0, rows) for x in samp)

    fn = _kd_chunk
    if (compile_mode() == "layers" and classes is None and not REV_ACC.get("on") and not READ_ACC.get("on")
            and not BEH_ACC.get("on") and not SAMP_ACC.get("on") and not TRAIN_ACC.get("on")):  # globals: eager
        fn = lambda *args: _run(_kd_chunk, *args)  # noqa: E731
    total, sums = 0.0, None
    for a in range(0, hidden.shape[0], chunk):
        b = min(a + chunk, hidden.shape[0])
        loss, kd, lm, kr = checkpoint(fn, hidden[a:b], weight, t_top[a:b], t_idx[a:b], t_tail[a:b],
                                      None if labels is None else labels[a:b], mask[a:b], lm_weight,
                                      None if reply is None else reply[a:b], mix,
                                      None if classes is None else classes[:, a:b],
                                      None if samp is None else tuple(x[a:b] for x in samp), use_reentrant=False)
        total = total + loss
        part = torch.stack([kd, lm, kr]).double()  # float64 sums as the Python floats were; one host sync per call
        sums = part if sums is None else sums + part
    if lazy and sums is not None:
        return total, sums[0], sums[1], sums[2]
    HOST.wait("kd_sums")
    with HOST.leaf("kd_sums"):
        kd_sum, lm_sum, reply_sum = sums.tolist() if sums is not None else (0.0, 0.0, 0.0)
    return total, kd_sum, lm_sum, reply_sum


class Held:
    """Per-sequence targets passed through FSDP's forward untouched (its input cast would round tensors to BF16)."""

    def __init__(self, **items):
        self.__dict__.update(items)


# ---- hidden-width cut (order; 2560 -> 1920 in whole 128-blocks) ------------------------------------------

_H, _HC = "H", "HC"
WIDTH_AXES = [  # (pattern on the name, {axis: kind}); H: the hidden width, HC: hc_count copies of it
    (r"embed_tokens\.weight$|lm_head\.weight$", {1: _H}),
    (r"(self_attn\.(q|k|v)_proj|self_attn\.indexer\.index_qk_proj|linear_attn\.in_proj_(qkv|z|a|b)|mlp\.gate"
     r"|mlp\.shared_expert\.(gate|up)_proj|mlp\.shared_expert_gate|mlp\.experts\.\d+\.(gate|up)_proj)\.weight$", {1: _H}),
    (r"(self_attn\.o_proj|linear_attn\.out_proj|mlp\.shared_expert\.down_proj|mlp\.experts\.\d+\.down_proj)\.weight$",
     {0: _H}),
    (r"mlp\.experts\.gate_up_proj$", {2: _H}),
    (r"mlp\.experts\.down_proj$", {1: _H}),
    (r"(hyper_connection|hyper_connection_mixer)\.(input_mix_weight_down|block_inject_weight)\.weight$", {1: _HC}),
    (r"(hyper_connection|hyper_connection_mixer)\.input_mix_weight_up\.weight$", {0: _HC}),
    (r"hc_norm\.weight$", {0: _HC}),
    (r"ple\.key_proj\.weight$", {0: _HC}),  # axis 1 is ple_embed_dim
    (r"ple\.value_proj\.weight$", {0: _H}),
    (r"ple\.norm_(key|query|conv)\.weight$|ple\.conv1d\.weight$", {0: _HC}),
    (r"mtp\.fc_(embedding|hidden)\.weight$", {0: _H, 1: _H}),
    (r"mtp\.pre_fc_norm_embedding\.weight$", {0: _H}),
    (r"mtp\.pre_fc_norm_hidden\.weight$", {0: _HC}),
    (r"visual\.merger\.linear_fc2\.(weight|bias)$", {0: _H}),
]
# Gemma-style norms (x / rms x (1 + w)) whose input loses channels: (1 + w) x sqrt(D / D') keeps their output when the
# dropped channels carry nothing; not ple.norm_key (its gate divides by sqrt(hidden_size), which cancels the factor)
WIDTH_COMPENSATE = r"hc_norm\.weight$|ple\.norm_(query|conv)\.weight$|mtp\.pre_fc_norm_(embedding|hidden)\.weight$"


def width_axes(name):
    import re as _re

    for pattern, axes in WIDTH_AXES:
        if _re.search(pattern, name):
            return axes
    return {}


def width_cut_state_dict(state, *, hidden, hc_count, keep, compensate=True):
    """The tensors with hidden channels ``keep`` (sorted ids, shared by the hc copies) only. A tensor with a hidden-
    sized axis that no rule covers raises (except the GDN qkv axis and its conv, whose 4 x 2560 is a coincidence)."""
    import re as _re

    keep = torch.as_tensor(sorted(keep), dtype=torch.long)
    index = {_H: keep, _HC: torch.cat([keep + c * hidden for c in range(hc_count)])}
    factor = math.sqrt(hidden / keep.numel())
    out = {}
    for name, tensor in state.items():
        axes = width_axes(name)
        if not axes and not _re.search(r"linear_attn\.(conv1d|norm|A_log|dt_bias)|visual\.|q_norm|k_norm|layernorm", name) and (
                hidden in tensor.shape or hidden * hc_count in tensor.shape):
            raise ValueError(f"no width rule for {name} {tuple(tensor.shape)}")
        for axis, kind in axes.items():
            tensor = tensor.index_select(axis, index[kind].to(tensor.device))
        if compensate and _re.search(WIDTH_COMPENSATE, name):
            tensor = ((1 + tensor.float()) * factor - 1).to(tensor.dtype)
        out[name] = tensor.contiguous()
    return out


def width_cut_config(config, width):
    import copy

    config = copy.deepcopy(config)
    config.text_config.hidden_size = width
    if getattr(config, "vision_config", None) is not None:
        config.vision_config.out_hidden_size = width
    return config


def width_blocks(scores, *, block=128, keep_blocks):
    """The channels of the ``keep_blocks`` whole blocks with the largest summed score (per-channel activation
    energy, e.g. mean squared residual stream over the hc copies on excl5)."""
    blocks = scores.float().view(-1, block).sum(-1)
    chosen = sorted(blocks.topk(keep_blocks).indices.tolist())
    return [b * block + i for b in chosen for i in range(block)]


# ---- MTP head (port of SGLang qwen4_exp_mtp; transformers' qwen4_exp ignores the mtp.* weights) ----------------------

class MtpHead(torch.nn.Module):
    """Flash's MTP block on transformers modules (SGLang ``Qwen4ExpForCausalLMMTP``, the hc_count > 1 fusion): at
    sequence index j the embedding of token j and the main model's pre-mixer hyper-connection state at j-1 are fused
    (``fc_embedding``, ``fc_hidden`` after Gemma-style RMS norms), go through one full-attention decoder layer (its
    own cache) and a mixer; the main LM head on the result gives the draft distribution of token j+1. Weights:
    ``mtp.*`` of the checkpoint, ``load_mtp``."""

    def __init__(self, text_config, experts=None, top_k=None):
        import copy

        from transformers.models.qwen4_exp import modeling_qwen4_exp as m

        super().__init__()
        cfg = copy.deepcopy(text_config)
        # the main model's own name for its full-attention layers (it differs between transformers versions)
        full = next(t for t in text_config.layer_types if t != "linear_attention")
        cfg.num_hidden_layers, cfg.layer_types, cfg.ple_layer_ids = 1, [full], []
        if experts:
            cfg.num_experts = experts
        if top_k:
            cfg.num_experts_per_tok = top_k
        try:  # BF16 grouped GEMM experts whatever the main model's backend (v3: the NVFP4 teacher's slowed this 4x)
            cfg._experts_implementation = "grouped_mm"
        except Exception:
            pass
        self.config = cfg
        width, count = cfg.hidden_size, cfg.hc_count
        self.fc_embedding = torch.nn.Linear(width, width, bias=False)
        self.fc_hidden = torch.nn.Linear(width, width, bias=False)
        self.pre_fc_norm_embedding = m.Qwen4ExpTextRMSNorm(width, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = m.Qwen4ExpTextRMSNorm(count * width, eps=cfg.rms_norm_eps)
        self.layers = torch.nn.ModuleList([m.Qwen4ExpTextDecoderLayer(cfg, 0)])
        self.hyper_connection_mixer = m.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        self.rotary_emb = m.Qwen4ExpTextRotaryEmbedding(config=cfg)

    def new_cache(self):
        from transformers import DynamicCache

        return DynamicCache(config=self.config)

    def forward(self, embeds, hc_prev, position_ids, cache=None, checkpointed=False):
        """``embeds`` [1, n, H] (tokens j), ``hc_prev`` [1, n, hc x H] (main states at j-1), ``position_ids``
        [3 or 4, 1, n] (those of tokens j); ``cache``: this block's prefix (updated in place unless read-only)."""
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m

        width, count = self.config.hidden_size, self.config.hc_count
        x = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        h = self.pre_fc_norm_hidden(hc_prev)
        h = self.fc_hidden(h.view(*h.shape[:-1], count, width))
        hidden = (x.unsqueeze(-2) + h).view(hc_prev.shape)
        # positions and mask as Qwen4ExpTextModel.forward
        if position_ids.shape[0] == 4:
            text_position_ids, position_ids = position_ids[0], position_ids[1:]
        else:
            text_position_ids = None
        past = cache.get_seq_length() if cache is not None else 0
        if cache is not None:
            if hasattr(cache, "position_ids"):
                position_ids = torch.cat([cache.position_ids, position_ids], dim=-1)
            cache.position_ids = position_ids
        mask = m.create_causal_mask(config=self.config, inputs_embeds=embeds,
                                    attention_mask=None if mask_host_on() else
                                    torch.ones(1, past + embeds.shape[1], device=embeds.device, dtype=torch.long),
                                    past_key_values=cache, position_ids=text_position_ids, allow_is_causal_skip=False)
        position_embeddings = self.rotary_emb(embeds, position_ids)
        layer = self.layers[0]
        call = lambda h: layer(h, position_embeddings=position_embeddings, attention_mask=mask, conv_mask=None,  # noqa: E731
                               past_key_values=cache, ple_input_ids=None)
        if checkpointed and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            hidden = checkpoint(call, hidden, use_reentrant=False)
        else:
            hidden = call(hidden)
        return self.hyper_connection_mixer(hidden)


def load_mtp(head, folder, device="cpu", dtype=torch.bfloat16):
    """The ``mtp.*`` tensors of a checkpoint folder into ``head`` (strict)."""
    from safetensors import safe_open

    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({f for k, f in index.items() if k.startswith("mtp.")})
    state = {}
    for name in files:
        with safe_open(str(folder / name), "pt") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    state[key[4:]] = f.get_tensor(key)
    missing, unexpected = head.load_state_dict(state, strict=False)
    missing = [k for k in missing if "rotary_emb" not in k and "inv_freq" not in k]
    if missing or unexpected:
        raise RuntimeError(f"mtp weights: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return head.to(device=device, dtype=dtype)


def mtp_inputs(ids, start, stop, piece=None):
    """MTP indices j of a main span [start, stop): j in [start + 1, min(stop + 1, T)); their token ids and
    positions."""
    total = ids.shape[1]
    a, b = start + 1, min(stop + 1, total)
    if b <= a:
        return None
    if piece is not None:
        positions = piece(a, b)["position_ids"]
    else:
        positions = torch.arange(a, b, device=ids.device).view(1, 1, -1).expand(3, 1, -1)
    return a, b, ids[:, a:b], positions


def hc_capture(model):
    """A forward pre-hook on the main mixer that keeps its input (the pre-mixer hyper-connection state)."""
    mixer = next(m for n, m in model.named_modules() if n.endswith("language_model.hyper_connection_mixer"))
    box = {}

    def hook(module, args, kwargs):
        box["hc"] = args[0] if args else kwargs["hidden_states"]

    return box, mixer.register_forward_pre_hook(hook, with_kwargs=True)


class Runner(torch.nn.Module):
    """The model behind one forward entry point, so FSDP's root group (embedding, LM head) is gathered for every use:
    ``mode`` "loss" (a window's KD against a read-only prefix cache), "extend" (the prefix cache over a span, no
    gradients)."""

    def __init__(self, model, mtp=None):
        super().__init__()
        self.model = model
        self.mtp = mtp  # MtpHead or None; its KD target at index j is the teacher's row j
        self._box = hc_capture(model)[0] if mtp is not None else None
        self.captured = None  # loss mode with capture: (main Capture, MTP Capture or None), for commit_captured
        if compile_mode():
            install_compile(self)

    def embed(self, ids):
        return text_model(self.model).language_model.embed_tokens(ids)

    def _mtp(self, ids, start, stop, piece, cache, grad):
        got = mtp_inputs(ids, start, stop, piece)
        if got is None:
            return None
        a, b, cur, positions = got
        hc = self._box["hc"][:, : b - a]
        return a, b, self.mtp(self.embed(cur), hc, positions, cache,
                              checkpointed=grad and recompute_policy() != "none")

    @property
    def config(self):
        return self.model.config

    def forward(self, mode, ids, start, stop, cache, held=None, piece=None, mtp_cache=None, capture=False, lazy=False):
        if mode == "extend":
            _, cache = hidden_window(self.model, ids, start, stop, cache, use_cache=True, piece=piece)
            if self.mtp is not None:
                self._mtp(ids, start, stop, piece, mtp_cache, False)
            return cache
        h = held
        main = Capture(cache, self.model.config.text_config) if capture else None
        hidden, _ = hidden_window(self.model, ids, start, stop, main.view if capture else read_only(cache),
                                  use_cache=False, piece=piece)
        cut = lambda x: None if x is None else x[start:stop]  # noqa: E731
        loss, kd, lm, kr = kd_loss(hidden[0], self.model.lm_head.weight, h.top[start:stop], h.idx[start:stop],
                                   h.tail[start:stop], h.mask[start:stop], labels=cut(h.labels), lm_weight=h.lm_weight,
                                   reply=cut(h.reply), lazy=lazy, rows=kd_rows(h, start, stop),
                                   classes=None if getattr(h, "classes", None) is None else h.classes[:, start:stop],
                                   samp=None if getattr(h, "samp", None) is None else tuple(x[start:stop] for x in h.samp))
        self.captured = (main, None) if capture else None
        if self.mtp is None:
            return loss, kd, lm, kr, 0.0, 0.0
        if capture:
            side = Capture(mtp_cache, self.mtp.config)
            self.captured, prefix = (main, side), side.view
        else:
            prefix = read_only(mtp_cache) if mtp_cache is not None and mtp_cache.get_seq_length() > 0 else None
        got = self._mtp(ids, start, stop, piece, prefix, True)
        if got is None:
            return loss, kd, lm, kr, 0.0, 0.0
        a, b, drafted = got
        mtp_loss, mtp_kd, _, _ = kd_loss(drafted[0], self.model.lm_head.weight, h.top[a:b], h.idx[a:b], h.tail[a:b],
                                         h.mask[a:b], mix={"forward": 1.0, "reverse": 0.0}, lazy=lazy,
                                         rows=kd_rows(h, a, b))  # forward KL
        if lazy:
            return loss + h.mtp_weight * mtp_loss, kd, lm, kr, mtp_kd, h.mask[a:b].sum().double()
        HOST.wait("mtp_n")
        with HOST.leaf("mtp_n"):
            mtp_n = float(h.mask[a:b].sum())
        return loss + h.mtp_weight * mtp_loss, kd, lm, kr, mtp_kd, mtp_n


def runner_extend(runner, ids, start, stop, cache, chunk, piece=None, mtp_cache=None):
    from transformers import DynamicCache

    was = runner.training
    runner.eval()
    try:
        with torch.no_grad():
            if cache is None:
                cache = DynamicCache(config=runner.config.text_config)
            for a in range(start, stop, chunk):
                cache = runner("extend", ids, a, min(a + chunk, stop), cache, piece=piece, mtp_cache=mtp_cache)
    finally:
        runner.train(was)
    return cache


def train_sequence(runner, ids, teacher, *, window, mask=None, labels=None, lm_weight=0.0, chunk=4096, scale=1.0,
                   backward=True, piece=None, reply=None, mtp_weight=0.0, classes=None, samp=None):
    """KD over every position of ids [1, T] in windows; each window's prefix is a no-grad cache. Gradients accumulate
    (loss x ``scale``). Returns summed KD, summed NLL, the number of scored positions, summed KD on ``reply``
    positions and their number, summed MTP KD and its positions (with an MTP head on the runner)."""
    total = ids.shape[1]
    mask = torch.ones(total, device=ids.device) if mask is None else mask
    held = Held(top=teacher[0], idx=teacher[1], tail=teacher[2], mask=mask, labels=labels, lm_weight=lm_weight,
                reply=reply, mtp_weight=mtp_weight, classes=classes, samp=samp)
    kd_sum = lm_sum = reply_sum = mtp_sum = mtp_n = 0.0
    cache = None
    head = getattr(runner, "mtp", None)
    mtp_cache = head.new_cache() if head is not None else None
    capture = capture_on()
    for start in range(0, total, window):
        stop = min(start + window, total)
        keep = capture and stop < total
        with torch.set_grad_enabled(backward):
            loss, kd, lm, kr, mk, mn = runner("loss", ids, start, stop, cache, held=held, piece=piece, mtp_cache=mtp_cache,
                                              capture=keep)
        if backward:
            (loss * scale).backward()
        kd_sum, lm_sum, reply_sum, mtp_sum, mtp_n = kd_sum + kd, lm_sum + lm, reply_sum + kr, mtp_sum + mk, mtp_n + mn
        del loss
        if keep:
            cache = commit_captured(runner)
        elif stop < total:
            cache = runner_extend(runner, ids, start, stop, cache, chunk, piece=piece, mtp_cache=mtp_cache)
    return (kd_sum, lm_sum, float(mask.sum()), reply_sum, float(reply.sum()) if reply is not None else 0.0, mtp_sum,
            mtp_n)


def span_windows(total, window):
    """Window spans of a sequence; a last window of one token joins the previous one's end (the MTP pass needs two)."""
    spans = [(s, min(s + window, total)) for s in range(0, total, window)]
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] == 1:
        spans[-2], spans[-1] = (spans[-2][0], spans[-2][1] - 1), (spans[-1][0] - 1, total)
    return spans


def dp_dummy(model, device, k, length=16):
    """A zero-weight sequence for the padding calls (all-zero mask: no gradient, the same FSDP collectives)."""
    import flash_qlora_job as fq

    ids = torch.full((1, length), 11, dtype=torch.long, device=device)
    teacher = (torch.zeros(length, k, dtype=torch.float16, device=device),
               torch.zeros(length, k, dtype=torch.int32, device=device),
               torch.zeros(length, dtype=torch.float16, device=device))
    zeros = torch.zeros(length, device=device)
    out = {"ids": ids, "teacher": teacher, "mask": zeros, "piece": fq.positions_and_images(model, {"input_ids": ids})}
    if s4("PLE_TABLE"):
        out["ple"] = {i: t.to(device) for i, t in ple_rows(model, ids[0].cpu()).items()}
    return out


def prefix_start(live):
    """HEAL_INERT_PREFIX=1 (trainer MFU S1, 23 Oct): where a slot's loss windows start, from the host mask of every
    row that carries a loss or a statistic (``live``: KD mask | reply | sampled rows). The MTP term at row j reads
    the main hidden state at j - 1, so the loss starts one row before the first live row, rounded down to the GDN
    chunk grid (64). Rows before it go through the no-grad prefix extension only. 0 when the flag is off."""
    if os.environ.get("HEAL_INERT_PREFIX") != "1" or live is None:
        return 0
    hit = torch.nonzero(live.reshape(-1).cpu() > 0)
    first = int(hit[0]) if hit.numel() else live.numel()
    return max(first - 1, 0) // 64 * 64


def slot_spans(mine, window):
    """A slot's windows and how many lead inert: with ``mine["prefix"]`` = L > 0 (prefix_start), window 0 is [0, L)
    (a padding loss call, then the prefix extension), the window holding L is cut to start at L, and the later
    windows are span_windows' own: rows before that cut window train exactly as without the flag; the cut window's
    rows lose the backprop into its rows before L (the prefix is a no-grad cache, as at every window start)."""
    total, start = mine["data"]["ids"].shape[1], int(mine.get("prefix", 0))
    start = min(start, total - 2)  # a one-row last window has no MTP rows: that rank alone would skip the MTP unit
    spans = span_windows(total, window)
    if not 0 < start < total:
        return spans, 0
    w = next(i for i, (a, b) in enumerate(spans) if a <= start < b)
    if os.environ.get("HEAL_INERT_PREFIX_EXACT") == "1":  # only whole windows before it: bit-exact, saves less
        return (spans, 0) if w == 0 else ([(0, spans[w][0])] + spans[w:], 1)
    return [(0, start), (start, spans[w][1])] + spans[w + 1:], 1


def slot_plan(mine, window, chunk):
    """This rank's call plan of a lockstep slot (dp_sequence): per window, the extend chunks after it (none with
    HEAL_CAPTURE_CACHE=1, except after an inert prefix window); [] for a padding-only rank."""
    if mine is None:
        return []
    spans, inert = slot_spans(mine, window)
    capture = capture_on()
    return [-(-(b - a) // chunk) if i < inert or (i < len(spans) - 1 and not capture) else 0
            for i, (a, b) in enumerate(spans)]


def slot_losses(mine, window):
    """(first, end) of this rank's real loss windows in a lockstep slot: windows first..end-1 carry a loss; (0, 0) for
    a padding-only rank (HEAL_SKIP_INERT_LOSS)."""
    if mine is None:
        return (0, 0)
    spans, inert = slot_spans(mine, window)
    return (inert, len(spans))


def skip_inert_loss(i, windows, losses):
    """HEAL_SKIP_INERT_LOSS=1 (MFU-1, 24 Oct): window i's loss call (a zero-mask padding call: zero gradients, but a full
    FSDP gather in the forward and the backward) is skipped when no rank has a real loss window there (every slot's
    inert prefix window 0); never the slot's last window (the step's gradient sync rides on its backward)."""
    if os.environ.get("HEAL_SKIP_INERT_LOSS") != "1" or losses is None or i >= windows - 1:
        return False
    return all(not (a <= i < b) for a, b in losses)


def chunk_after(step, chunk):
    """HEAL_CHUNK_AFTER=<step>:<chunk> (PTH bench, 23 Oct): the training slots' prefix-extend chunk from that step on
    (the step being accumulated); ``chunk`` (--chunk) before it."""
    spec = os.environ.get("HEAL_CHUNK_AFTER") or ""
    if not spec:
        return chunk
    at, new = (int(x) for x in spec.split(":"))
    if at < 1 or new < 1:
        raise ValueError(f"HEAL_CHUNK_AFTER={spec}: want <step>=1..:<chunk>=1..")
    return new if step >= at else chunk


def inert_check_pass(runner, mine, dummy, *, window, chunk, mtp_weight, lazy):
    """HEAL_INERT_CHECK=1 (PTH bench, 23 Oct): this slot once more with no grad and S1 off (loss windows from row 0),
    every rank together; the training accumulators (TRAIN_ACC, SAMP_ACC) are left as they were. Returns this rank's
    {kd, kd_reply, loss} (loss: the windows' summed loss calls)."""
    saved = {name: dict(acc) for name, acc in (("train", TRAIN_ACC), ("samp", SAMP_ACC))}
    trace = []
    try:
        row = dp_sequence(runner, None if mine is None else {**mine, "prefix": 0}, dummy, window=window, chunk=chunk,
                          scale=1.0, mtp_weight=mtp_weight, trace=trace, trace_cache=False, lazy=lazy, backward=False)
    finally:
        for name, acc in (("train", TRAIN_ACC), ("samp", SAMP_ACC)):
            acc.clear()
            acc.update(saved[name])
    return {"kd": float(row["kd"]), "kd_reply": float(row["kd_reply"]), "loss": sum(t["loss"] for t in trace)}


def inert_check_event(on, off, device):
    """The inert_check event's fields: the S1-on (training) and S1-off sums over ranks and their relative diffs."""
    import torch.distributed as dist

    keys = ("kd", "kd_reply", "loss")
    box = torch.tensor([on[k] for k in keys] + [off[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(box)
    a, b = box[:3].tolist(), box[3:].tolist()
    return {**{f"{k}": b[i] for i, k in enumerate(keys)}, "s1_on": dict(zip(keys, a)),
            "rel": {k: abs(b[i] - a[i]) / max(abs(a[i]), 1e-30) for i, k in enumerate(keys)}}


def dp_sequence(runner, mine, dummy, *, window, chunk, scale, mtp_weight, trace=None, sync=None, trace_cache=True,
                plans=None, lazy=False, backward=True, losses=None):
    """One lockstep slot of data-parallel training: this rank's sequence (``mine``; None: padding only) in windows
    with no-grad prefix extension, as train_sequence, padded to every rank's call plan: for window i, one loss call
    (+ backward), then the most extend chunks any rank needs after it. Returns this rank's sums. HEAL_CAPTURE_CACHE=1:
    the loss call keeps the next window's prefix (``Capture``), so no rank plans an extend chunk. ``trace`` (a list,
    the capture check): per real window its loss, KD, MTP KD and the prefix cache built after it (CPU copies).
    ``sync`` (GradSync, HEAL_GRAD_SYNC=step): called before each backward with (window, windows). ``plans``: every
    rank's ``slot_plan``, exchanged by the caller (else gathered here). ``lazy`` (HEAL_LAZY_STATS): the KD sums and
    MTP positions stay float64 card scalars, the positions come from ``mine["n_host"]`` / ``mine["nr_host"]``.
    ``mine["ple"]`` / ``dummy["ple"]`` (HEAL_PLE_TABLE): the sequence's host PLE rows / the dummy's card rows.
    ``backward=False`` (HEAL_SHARD_READS): a no-grad read slot; ``mine["labels"]`` / ``mine["classes"]`` when given."""
    import torch.distributed as dist
    from transformers import DynamicCache

    head = getattr(runner, "mtp", None)
    capture = capture_on()
    plan = slot_plan(mine, window, chunk)
    if mine is not None:
        data = mine["data"]
        ids, total = data["ids"], data["ids"].shape[1]
        spans, inert = slot_spans(mine, window)
        held = Held(top=data["teacher"][0], idx=data["teacher"][1], tail=data["teacher"][2], mask=mine["mask"],
                    labels=mine.get("labels"), lm_weight=0.0, reply=data["reply"], mtp_weight=mtp_weight)
        held.samp = mine.get("samp")  # HEAL_SAMPLED: the sampled-token reverse term on this slot's file
        if not backward:  # a read: its own class masks (or none), never the training split
            held.classes = mine.get("classes")
        elif os.environ.get("HEAL_TRAIN_CLASS_SPLIT") == "1":  # per-step training KL by reply class (OPD monitor)
            held.classes = reply_classes(ids[0], data["reply"]) * (mine["mask"].reshape(-1) > 0).to(data["reply"].device)
    else:
        spans, inert = [], 0
    skip_on = os.environ.get("HEAL_SKIP_INERT_LOSS") == "1"
    if plans is None:
        with HOST.phase("plan"):
            HOST.wait("plan")
            every = [None] * dist.get_world_size()
            own = (plan, float(total) if mine is not None else 0.0, slot_losses(mine, window))
            dist.all_gather_object(every, own if (HOST.on or skip_on) else plan, group=ctl_group())
            if HOST.on or skip_on:
                plans = [p for p, _, _ in every]
                losses = [w for _, _, w in every] if skip_on else None
                if HOST.on:
                    HOST.slot([t for _, t, _ in every], [len(p) for p in plans])
            else:
                plans = every
    windows = max(len(x) for x in plans)
    d_held = Held(top=dummy["teacher"][0], idx=dummy["teacher"][1], tail=dummy["teacher"][2], mask=dummy["mask"],
                  labels=None, lm_weight=0.0, reply=dummy["mask"], mtp_weight=0.0)
    d_len = dummy["ids"].shape[1]
    if mine is None:
        n = n_reply = 0.0
    elif lazy and "n_host" in mine:
        n, n_reply = mine["n_host"], mine["nr_host"]
    else:
        HOST.wait("positions")
        n, n_reply = float(mine["mask"].sum()), float(data["reply"].sum())
    out = {"kd": 0.0, "n": n, "kd_reply": 0.0, "n_reply": n_reply, "kd_mtp": 0.0, "n_mtp": 0.0,
           "tokens": float(total) if mine else 0.0}
    cache = None
    mtp_cache = head.new_cache() if (head is not None and mine is not None) else None
    table, d_table = (mine or {}).get("ple"), dummy.get("ple")
    device = dummy["ids"].device
    try:
        for i in range(windows):
            keep = capture and inert <= i < len(spans) - 1
            skip = skip_inert_loss(i, windows, losses)  # no rank trains window i: its padding loss call is skipped
            if skip and trace is not None and i < inert:
                trace.append({"window": i, "loss": 0.0, "kd": 0.0, "kd_mtp": 0.0, "inert": True})
            loss = None
            with HOST.phase("fwd"), torch.set_grad_enabled(backward):
                if not skip:
                    if inert <= i < len(spans):
                        a, b = spans[i]
                        PLE["rows"] = ple_slice(table, a, b, device)
                        loss, kd, _, kr, mk, mn = runner("loss", ids, a, b, cache, held=held, piece=mine["piece"],
                                                         mtp_cache=mtp_cache, capture=keep, lazy=lazy)
                        out["kd"] += kd; out["kd_reply"] += kr; out["kd_mtp"] += mk; out["n_mtp"] += mn
                        if trace is not None:
                            trace.append({"window": i, "loss": float(loss.detach()), "kd": float(kd), "kd_mtp": float(mk)})
                    else:  # padding, or this rank's inert prefix window (its rows carry no loss)
                        PLE["rows"] = d_table
                        loss = runner("loss", dummy["ids"], 0, d_len, None, held=d_held, piece=dummy["piece"],
                                      mtp_cache=head.new_cache() if head is not None else None, lazy=lazy)[0]
                        if trace is not None and i < inert:
                            trace.append({"window": i, "loss": 0.0, "kd": 0.0, "kd_mtp": 0.0, "inert": True})
                    if sync is not None:
                        sync(i, windows)
            if backward and loss is not None:
                with HOST.phase("bwd"):
                    (loss * scale).backward()
            del loss
            if keep:
                with HOST.phase("commit"):
                    cache = commit_captured(runner)
            need = max(x[i] if i < len(x) else 0 for x in plans)
            mine_n = plan[i] if i < len(plan) else 0
            if not need:
                if trace is not None and keep and trace_cache:
                    trace[-1]["cache"] = cache_snapshot(cache, mtp_cache)
                continue
            was = runner.training
            runner.eval()
            try:
                with torch.no_grad(), HOST.phase("extend"):
                    if mine_n and cache is None:
                        cache = DynamicCache(config=runner.config.text_config)
                    a, b = spans[i] if i < len(spans) else (0, 0)
                    for j in range(need):
                        if j < mine_n:
                            s = a + j * chunk
                            PLE["rows"] = ple_slice(table, s, min(s + chunk, b), device)
                            cache = runner("extend", ids, s, min(s + chunk, b), cache, piece=mine["piece"],
                                           mtp_cache=mtp_cache)
                        else:
                            PLE["rows"] = d_table
                            runner("extend", dummy["ids"], 0, d_len, DynamicCache(config=runner.config.text_config),
                                   piece=dummy["piece"], mtp_cache=head.new_cache() if head is not None else None)
            finally:
                runner.train(was)
            if trace is not None and mine_n and trace_cache:
                trace[-1]["cache"] = cache_snapshot(cache, mtp_cache)
    finally:
        PLE["rows"] = None
    return out


def capture_on():
    return os.environ.get("HEAL_CAPTURE_CACHE") == "1"


class Capture:
    """The window pass's own prefix state for the next window (HEAL_CAPTURE_CACHE=1; trainer MFU S1, 21 Oct), in
    place of a second no-grad pass over the window. ``view``: ``read_only(cache)`` whose update calls also keep the
    window's new states, detached, the first time each (layer, kind, state) is updated; the checkpointed layers'
    replays during backward call them again and are ignored. Kept: the window's keys/values and indexer keys, the
    last ``conv_kernel_size`` conv inputs, the final recurrent state. ``commit()`` (after the backward: the replays
    read the prefix) applies them to the prefix cache with the cache's own update calls, as an extension over the
    window in one chunk would. ``cache`` None: an empty prefix."""

    def __init__(self, cache, config):
        import copy

        from transformers import DynamicCache

        self.base = cache if cache is not None else DynamicCache(config=config)
        self.records = {}
        base, records = self.base, self.records
        layers = base.layers
        view = copy.copy(base)

        def keep(key, *tensors):
            if key not in records:
                records[key] = tensors

        def update(key_states, value_states, layer_idx, *args, **kwargs):
            keep((layer_idx, "kv", 0), key_states.detach(), value_states.detach())
            layer = layers[layer_idx]
            if not getattr(layer, "is_initialized", False):
                return key_states, value_states
            return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)

        def update_indexer(states, layer_idx):
            keep((layer_idx, "indexer", 0), states.detach())
            layer = layers[layer_idx]
            if not getattr(layer, "is_indexer_initialized", False):
                return states
            return torch.cat([layer.indexer_keys, states], dim=1)

        def update_conv_state(conv_states, layer_idx, state_idx=0, conv_kernel_size=None, **kwargs):
            width = conv_states.shape[-1] if conv_kernel_size is None else conv_kernel_size
            keep((layer_idx, "conv", state_idx), conv_states.detach()[..., -width:].clone(), conv_kernel_size)
            layer = layers[layer_idx]
            if layer.has_previous_state.get(state_idx, False):
                return torch.cat([layer.conv_states[state_idx], conv_states], dim=-1)
            if conv_states.shape[-1] < width:  # the cache's prefill rule (LinearAttentionLayer.update_conv_state)
                return F.pad(conv_states, (width - conv_states.shape[-1], 0), value=0)
            return conv_states

        def update_recurrent_state(states, layer_idx, state_idx=0, **kwargs):
            keep((layer_idx, "recurrent", state_idx), states.detach())
            return states

        view.update, view.update_indexer = update, update_indexer
        view.update_conv_state, view.update_recurrent_state = update_conv_state, update_recurrent_state
        self.view = view

    def commit(self):
        base = self.base
        for (layer_idx, kind, state_idx), got in self.records.items():
            if kind == "kv":
                base.update(got[0], got[1], layer_idx)
            elif kind == "indexer":
                base.update_indexer(got[0], layer_idx)
            elif kind == "conv":
                base.update_conv_state(got[0], layer_idx, state_idx=state_idx, conv_kernel_size=got[1])
            else:
                base.update_recurrent_state(got[0], layer_idx, state_idx=state_idx)
        if "position_ids" in self.view.__dict__:  # set by the model / MtpHead on the view
            base.position_ids = self.view.position_ids
        self.records.clear()
        return base


def commit_captured(runner):
    """The main prefix cache after the last loss call with ``capture`` (its MTP cache is extended in place)."""
    main, side = runner.captured
    runner.captured = None
    if side is not None:
        side.commit()
    return main.commit()


def cache_snapshot(cache, mtp_cache=None):
    """CPU float copies of every cached state (the capture check)."""
    def one(c):
        if c is None:
            return {}
        out = {}
        for i, layer in enumerate(c.layers):
            for name in ("keys", "values", "indexer_keys"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor) and t.numel():
                    out[f"{i}.{name}"] = t.detach().float().cpu()
            for name in ("conv_states", "recurrent_states"):
                for s, t in (getattr(layer, name, None) or {}).items():
                    if isinstance(t, torch.Tensor):
                        out[f"{i}.{name}.{s}"] = t.detach().float().cpu()
        if hasattr(c, "position_ids"):
            out["position_ids"] = c.position_ids.detach().float().cpu()
        return out

    return {"main": one(cache), "mtp": one(mtp_cache)}


def recompute_policy():
    """HEAL_RECOMPUTE: full (every decoder layer checkpointed; the default), alternate (even layers only), none."""
    policy = os.environ.get("HEAL_RECOMPUTE", "full")
    if policy not in ("full", "alternate", "none"):
        raise ValueError(f"HEAL_RECOMPUTE={policy}")
    return policy


def set_recompute(model, policy=None):
    """Apply the recompute policy to a model whose gradient checkpointing is enabled (decoder layers only)."""
    policy = policy or recompute_policy()
    for i, layer in enumerate(text_model(model).language_model.layers):
        layer.gradient_checkpointing = policy == "full" or (policy == "alternate" and i % 2 == 0)
    return policy


def read_only(cache):
    """A copy of the cache whose update calls are pure (flash_qlora_job.read_only): the window pass reads the prefix
    state and stores nothing, so autograd's saved tensors are never written in place and the prefix cache itself stays
    as it was for the next window's extension."""
    if cache is None:
        return None
    import copy

    cache = copy.copy(cache)
    layers = cache.layers

    def update(key_states, value_states, layer_idx, *args, **kwargs):
        layer = layers[layer_idx]
        return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)

    def update_indexer(states, layer_idx):
        return torch.cat([layers[layer_idx].indexer_keys, states], dim=1)

    def update_conv_state(conv_states, layer_idx, state_idx=0, **kwargs):
        layer = layers[layer_idx]
        if not layer.has_previous_state[state_idx]:
            raise RuntimeError(f"layer {layer_idx} has no prefix state {state_idx}")
        return torch.cat([layer.conv_states[state_idx], conv_states], dim=-1)

    def update_recurrent_state(states, layer_idx, state_idx=0, **kwargs):
        return states

    cache.update, cache.update_indexer = update, update_indexer
    cache.update_conv_state, cache.update_recurrent_state = update_conv_state, update_recurrent_state
    return cache


# ---- regional compile (HEAL_COMPILE; trainer MFU S2, 21 Oct) ---------------------------------------------------------
# The decoder layer's forward split into pure-tensor regions, each compiled whole (fullgraph); in eager between them:
# every cache call (Capture / read_only closures, DynamicCache), the PLE n-gram table (host RAM), the QSA selection and
# its Triton attention, the fla chunk kernel. The regions: hyper-connection in + GDN projections; conv + l2norm + gates;
# gated norm + out-projection + injection + MLP hyper-connection + MoE + injection (attention layers: o-proj onward);
# the final mixers; the KD chunk. HEAL_COMPILE=layers compiles them, =regions runs the same split eagerly (the
# rewrite's own check), unset: the original forward. GDN q/k are not repeated to the value heads (fla's GVA indexes
# them as repeat_interleave does) and are l2-normalised in the region (fla's formula) rather than in the kernel.

_COMPILED, _ORIGINAL = {}, {}
COMPILE_STATS = {"regions": {}}


def compile_mode():
    mode = os.environ.get("HEAL_COMPILE", "")
    if mode not in ("", "layers", "regions"):
        raise ValueError(f"HEAL_COMPILE={mode}")
    return mode


def _run(fn, *args):
    if compile_mode() != "layers":
        return fn(*args)
    compiled = _COMPILED.get(fn)
    if compiled is None:
        import torch._dynamo
        import torch._inductor.config
        import torch.fx.experimental._config

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        # window lengths vary per sequence: symbolic sizes from the first call (automatic dynamic recompiled once per
        # region, then again on every new coincidence of sizes that duck sizing had unified)
        torch.fx.experimental._config.use_duck_shape = False
        # round where eager rounds (BF16 intermediates), so the compiled numerics stay at the eager noise level
        torch._inductor.config.emulate_precision_casts = os.environ.get("HEAL_COMPILE_EMULATE", "1") == "1"
        dynamic = {"1": True, "0": False, "auto": None}[os.environ.get("HEAL_COMPILE_DYNAMIC", "1")]
        compiled = _COMPILED[fn] = torch.compile(fn, fullgraph=True, dynamic=dynamic)
    stats = COMPILE_STATS["regions"].setdefault(fn.__name__, 0)
    COMPILE_STATS["regions"][fn.__name__] = stats + 1
    return compiled(*args)


def _inject(hyper, out, weights):
    return hyper + (out.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


def _mlp_half(layer, h, moe):
    """The MLP half of a decoder layer; ``moe`` False (an experts backend with host syncs or its own kernels): only
    the hyper-connection, the MoE block and the injection follow in ``_finish``."""
    x, hyper, weights = layer.mlp_hyper_connection(h)
    return _inject(hyper, layer.mlp(x), weights) if moe else (x, hyper, weights)


def _finish(layer, got, moe):
    if moe:
        return got
    x, hyper, weights = got
    return _run(_inject, hyper, layer.mlp(x), weights)


def _moe_traceable(layer):
    return getattr(layer.mlp.experts.config, "_experts_implementation", None) in ("grouped_mm", "batched_mm")


def _l2norm(x, eps=1e-6):  # fla.modules.l2norm (fp32, x / sqrt(sum x^2 + eps), output in x's dtype)
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + eps)).to(x.dtype)


def _r_hc(hc, x):
    return _ORIGINAL["hc_forward"](hc, x)


def _r_gdn_in(layer, h, conv_mask):
    x, hyper, weights = layer.attn_hyper_connection(h)
    gdn = layer.linear_attn
    if conv_mask is not None:
        x = (x * conv_mask[:, :, None]).to(x.dtype)
    qkv = gdn.in_proj_qkv(x).transpose(1, 2).contiguous()  # one layout with or without a cached prefix
    return hyper, weights, qkv, gdn.in_proj_z(x), gdn.in_proj_b(x), gdn.in_proj_a(x)


def _r_gdn_mid(gdn, qkv, a, b, repeat):
    seq, length = qkv.shape[-1], a.shape[1]
    # the depthwise causal conv (kernel 4) as shifted products summed in fp32 (cuDNN's accumulation), so it fuses
    # with the activation, the split and the l2norm; dynamic-shape convolution also mis-specialised its strides
    w = gdn.conv1d.weight[:, 0].float()  # [C, K]
    width = w.shape[-1]
    xp = F.pad(qkv.float(), (width - 1, 0))  # the conv's zero padding; outputs kept: the last ``length``
    off = seq - length  # cached prefix columns (0 or width - 1)
    acc = xp[:, :, off:off + length] * w[:, 0, None]
    for j in range(1, width):
        acc = acc + xp[:, :, off + j:off + j + length] * w[:, j, None]
    out = F.silu(acc.to(qkv.dtype)).transpose(1, 2)
    q, k, v = torch.split(out, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
    batch = out.shape[0]
    q = _l2norm(q.reshape(batch, length, -1, gdn.head_k_dim))
    k = _l2norm(k.reshape(batch, length, -1, gdn.head_k_dim))
    v = v.reshape(batch, length, -1, gdn.head_v_dim).contiguous()
    beta = b.sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    if repeat:  # the torch reference kernel (no fla): equal head counts
        q = q.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)
        k = k.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)
    return q, k, v, g, beta


def _r_gdn_out(layer, core, z, hyper, weights, moe):
    gdn = layer.linear_attn
    batch, length = core.shape[:2]
    out = gdn.norm(core.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim)).reshape(batch, length, -1)
    return _mlp_half(layer, _inject(hyper, gdn.out_proj(out), weights), moe)


def _r_attn_in(attn, x, cos, sin):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    shape = x.shape[:-1]
    hidden_shape = (*shape, -1, attn.head_dim)
    q, gate = torch.chunk(attn.q_proj(x).view(*shape, -1, attn.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    q = attn.q_norm(q.reshape(hidden_shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(x).view(hidden_shape)).transpose(1, 2)
    v = attn.v_proj(x).view(hidden_shape).transpose(1, 2)
    q, k = modeling.apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, gate


def _r_attn_out(layer, o, gate, hyper, weights, moe):
    out = o.transpose(1, 2).reshape(*gate.shape[:-1], -1) * torch.sigmoid(gate)
    return _mlp_half(layer, _inject(hyper, layer.self_attn.o_proj(out), weights), moe)


def _r_mix_out(layer, out, hyper, weights, moe):
    return _mlp_half(layer, _inject(hyper, out, weights), moe)


def _gdn_layer(layer, h, conv_mask, cache):
    gdn = layer.linear_attn
    hyper, weights, qkv, z, b, a = _run(_r_gdn_in, layer, h, conv_mask)
    previous = False
    if cache is not None:
        previous = cache.has_previous_state(gdn.layer_idx, state_idx=0)
        qkv = cache.update_conv_state(qkv, gdn.layer_idx, conv_kernel_size=gdn.conv_kernel_size)
    kernel, repeat = _gdr_kernel(h)
    q, k, v, g, beta = _run(_r_gdn_mid, gdn, qkv, a, b, repeat)
    state = cache.layers[gdn.layer_idx].recurrent_states[0] if previous else None
    core, last = kernel(q, k, v, g=g, beta=beta, initial_state=state, output_final_state=cache is not None,
                        use_qk_l2norm_in_kernel=False, cu_seqlens=None)
    if cache is not None:
        cache.update_recurrent_state(last, gdn.layer_idx)
    moe = _moe_traceable(layer)
    return _finish(layer, _run(_r_gdn_out, layer, core, z, hyper, weights, moe), moe)


def _attn_layer(layer, h, position_embeddings, mask, cache):
    attn = layer.self_attn
    moe = _moe_traceable(layer)
    x, hyper, weights = _run(_r_hc, layer.attn_hyper_connection, h)
    rl1 = None
    if getattr(type(attn), "_rl1_selected", False):
        import rl1_trainer as rl1
    if rl1 is None or rl1.SPARSE["mode"] != "triton" or mask is None or (attn.training and attn.attention_dropout):
        out, _ = attn(x, position_embeddings, attention_mask=mask, past_key_values=cache)
        return _finish(layer, _run(_r_mix_out, layer, out, hyper, weights, moe), moe)
    counts = rl1.plain_causal(mask)
    if counts is None:
        out, _ = attn(x, position_embeddings, attention_mask=mask, past_key_values=cache)
        return _finish(layer, _run(_r_mix_out, layer, out, hyper, weights, moe), moe)
    from rl1_sparse import TritonSelectedAttention

    rl1.SPARSE["calls"] += 1
    picked = rl1.qsa_select(attn.indexer, x, position_embeddings, counts, cache)
    length = x.shape[1]
    cos, sin = (t[:, -length:, :].contiguous() for t in position_embeddings)
    q, k, v, gate = _run(_r_attn_in, attn, x, cos, sin)
    if cache is not None:
        k, v = cache.update(k, v, attn.layer_idx)
    o = TritonSelectedAttention.apply(q, k, v, picked, attn.scaling, None)
    return _finish(layer, _run(_r_attn_out, layer, o, gate, hyper, weights, moe), moe)


def _layer_forward(self, hidden_states, position_embeddings=None, attention_mask=None, conv_mask=None,
                   past_key_values=None, ple_input_ids=None, **kwargs):
    if not compile_mode() or kwargs or hidden_states.shape[1] < 2:  # decode steps keep the original path
        return _ORIGINAL["layer_forward"](self, hidden_states, position_embeddings, attention_mask, conv_mask,
                                          past_key_values, ple_input_ids, **kwargs)
    if self.ple is not None:  # host-RAM n-gram table and its cache state: eager (one layer)
        hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask)
    if self.layer_type == "linear_attention":
        return _gdn_layer(self, hidden_states, conv_mask, past_key_values)
    return _attn_layer(self, hidden_states, position_embeddings, attention_mask, past_key_values)


def _gdr_kernel(x):
    """fla's chunk_gated_delta_rule (the function transformers resolves to when fla is installed; q/k at their own head
    count: GVA) or, without fla / on CPU, transformers' torch reference (q/k repeated to the value heads)."""
    key = ("gdr", x.device.type)
    if key not in _ORIGINAL:
        _ORIGINAL[key] = None
        if x.is_cuda:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule

                _ORIGINAL[key] = (chunk_gated_delta_rule, False)
            except ImportError:
                pass
        if _ORIGINAL[key] is None:
            from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

            _ORIGINAL[key] = (modeling.torch_chunk_gated_delta_rule, True)
    return _ORIGINAL[key]


def _mixer_forward(self, hyper_input):
    if not compile_mode():
        return _ORIGINAL["hc_forward"](self, hyper_input)
    return _run(_r_hc, self, hyper_input)


def plain_causal_memo(original):
    """rl1_trainer.plain_causal once per mask tensor (every attention layer and its checkpoint replay get the same
    mask: the full [T, keys] sum, compare and host sync ran 2 x 12 times per window)."""
    import weakref

    memo = {}

    def plain_causal(attention_mask):
        key = id(attention_mask)
        got = memo.get(key)
        if got is not None and got[0]() is attention_mask and got[1] == attention_mask._version:
            return got[2]
        for k in [k for k, v in memo.items() if v[0]() is None]:
            del memo[k]
        counts = original(attention_mask)
        memo[key] = (weakref.ref(attention_mask), attention_mask._version, counts)
        return counts

    plain_causal.memo = memo
    return plain_causal


def install_compile(runner=None):
    """Patch the decoder layer class (once; the mode is read at every call) and ``runner``'s final mixers."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    if "layer_forward" not in _ORIGINAL:
        _ORIGINAL["layer_forward"] = modeling.Qwen4ExpTextDecoderLayer.forward
        _ORIGINAL["hc_forward"] = modeling.Qwen4ExpTextGatedResidual.forward
        modeling.Qwen4ExpTextDecoderLayer.forward = _layer_forward
        try:
            import rl1_trainer as rl1

            if not hasattr(rl1.plain_causal, "memo"):
                rl1.plain_causal = plain_causal_memo(rl1.plain_causal)
        except ImportError:
            pass
    if runner is not None:
        import types

        mixers = [text_model(runner.model).language_model.hyper_connection_mixer]
        if getattr(runner, "mtp", None) is not None:
            mixers.append(runner.mtp.hyper_connection_mixer)
        for mixer in mixers:
            mixer.forward = types.MethodType(_mixer_forward, mixer)


def compile_counters():
    """Dynamo's compiled frames / graph breaks / recompiles so far, and the regions' call counts."""
    import torch._dynamo.utils as du

    stats = {k: dict(v) for k, v in du.counters.items() if k in ("stats", "graph_break", "recompiles")}
    try:  # nested phases overlap: the outermost (_compile.compile_inner) is the frame compile time
        metrics = {k: round(sum(v), 2) for k, v in du.compilation_time_metrics.items()}
        stats["compile_seconds"] = metrics.get("_compile.compile_inner", max(metrics.values(), default=0.0))
        stats["compile_phases"] = dict(sorted(metrics.items(), key=lambda kv: -kv[1])[:6])
    except Exception:
        pass
    stats["frames"] = stats.get("stats", {}).get("unique_graphs", 0)
    stats["calls"] = dict(COMPILE_STATS["regions"])
    return stats


# ---- sharding -------------------------------------------------------------------------------------------------------

def shard(runner, mesh=None, compute_dtype=torch.bfloat16, ignored=None):
    """``ignored``: parameters left out of sharding (frozen: the host-RAM n-gram table, the vision tower)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    flags = comm_flags()
    policy = MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=flags["reduce_dtype"])
    model = runner.model
    extra = {"ignored_params": set(ignored)} if ignored else {}
    layers = list(model.model.language_model.layers)
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy, **extra)
    if not ignored:
        fully_shard(model.model.visual, mesh=mesh, mp_policy=policy)
    if getattr(runner, "mtp", None) is not None:
        fully_shard(runner.mtp.layers[0], mesh=mesh, mp_policy=policy)
    fully_shard(runner, mesh=mesh, mp_policy=policy, **extra)
    n = flags["prefetch"]
    if n:  # explicit all-gather prefetch over the decoder layers (every forward runs all of them, in order)
        for i, layer in enumerate(layers):
            layer.set_modules_to_forward_prefetch(layers[i + 1:i + 1 + n])
            layer.set_modules_to_backward_prefetch(layers[max(0, i - n):i][::-1])
    if flags["rs_buffers"]:
        runner.set_reduce_scatter_max_input_buffers(flags["rs_buffers"])
    return runner  # FSDP averages gradients over ranks: scale each rank's loss by world size / global positions


def comm_flags():
    """FSDP communication knobs (trainer MFU S3, 21 Oct; all off by default). HEAL_GRAD_SYNC=window|step: reduce
    after every window's backward (default) or only after the optimizer step's last one (unsharded gradients
    accumulate in the reduce dtype in between); HEAL_SYNC_EVERY=k (step mode): also after every k-th backward of the
    step; HEAL_REDUCE_DTYPE=fp32|bf16; HEAL_FSDP_PREFETCH=n: explicit all-gather prefetch of the next n decoder
    layers (forward) / previous n (backward); HEAL_RS_BUFFERS=k: FSDP's reduce-scatter input buffers in flight;
    HEAL_EXPERT_ROWS=1: the ESFT expert tensors' gradients bypass FSDP (RowAdam.tap)."""
    sync = os.environ.get("HEAL_GRAD_SYNC", "window")
    reduce = os.environ.get("HEAL_REDUCE_DTYPE", "fp32")
    if sync not in ("window", "step") or reduce not in ("fp32", "bf16"):
        raise ValueError(f"HEAL_GRAD_SYNC={sync} HEAL_REDUCE_DTYPE={reduce}")
    every = int(os.environ.get("HEAL_SYNC_EVERY", "0") or 0)
    if every and sync != "step":
        raise ValueError("HEAL_SYNC_EVERY needs HEAL_GRAD_SYNC=step")
    return {"sync": sync, "every": every, "reduce_dtype": torch.bfloat16 if reduce == "bf16" else torch.float32,
            "prefetch": int(os.environ.get("HEAL_FSDP_PREFETCH", "0") or 0),
            "rs_buffers": int(os.environ.get("HEAL_RS_BUFFERS", "0") or 0),
            "expert_rows": os.environ.get("HEAL_EXPERT_ROWS") == "1"}


class GradSync:
    """HEAL_GRAD_SYNC=step: FSDP reduce-scatters only on the backward that ends the optimizer step (and every
    ``every``-th). ``slot(ends)`` before each lockstep slot (``ends``: a step follows this slot, by run_train's own rule
    on the all-reduced slot positions); dp_sequence calls it before every backward with the window index and the
    slot's window count, both identical on every rank, so the collectives stay in lockstep."""

    def __init__(self, runner, every=0):
        from torch.distributed.fsdp import FSDPModule

        self.modules = [m for m in runner.modules() if isinstance(m, FSDPModule)]
        self.every, self.count, self.ends, self.synced, self.syncs = every, 0, False, True, 0

    def _set(self, on):
        for m in self.modules:
            m.set_requires_gradient_sync(on, recurse=False)

    def slot(self, ends):
        self.ends = ends

    def __call__(self, i, windows):
        self.count += 1
        on = (self.ends and i == windows - 1) or bool(self.every and self.count % self.every == 0)
        self._set(on)
        self.synced = on
        self.syncs += on

    def step_done(self):
        if not self.synced:
            raise RuntimeError("optimizer step after an unsynced backward (HEAL_GRAD_SYNC=step boundary mismatch)")
        self.count, self.ends, self.syncs = 0, False, 0

    def restore(self):
        self._set(True)


COMM_COUNTS = {}


def count_collectives():
    """HEAL_COUNT_COLLECTIVES=1 (the comm check): calls and bytes per collective kind, by wrapping torch.distributed's
    functions (FSDP2 looks them up at call time)."""
    import torch.distributed as dist

    def wrap(name, fn):
        def inner(*a, **kw):
            t = kw.get("input", kw.get("input_tensor", kw.get("tensor")))
            if t is None:
                t = a[1] if name in ("reduce_scatter_single", "all_gather_single", "reduce_scatter_tensor",
                                     "all_gather_into_tensor") and len(a) > 1 else (a[0] if a else None)
            row = COMM_COUNTS.setdefault(name, [0, 0])
            row[0] += 1
            row[1] += t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0
            return fn(*a, **kw)
        inner._counted = True
        return inner

    for name in ("reduce_scatter_single", "reduce_scatter_tensor", "all_gather_single", "all_gather_into_tensor",
                 "all_reduce"):
        fn = getattr(dist, name, None)
        if fn is not None and not getattr(fn, "_counted", False):
            setattr(dist, name, wrap(name, fn))


def full_grads(runner, rows, experts_only=False):
    """Full float gradients by name on CPU (collective); HEAL_EXPERT_ROWS: the expert tensors from the row buffer
    (selected rows, the rest zero)."""
    tapped = rows.tap_grads() if rows.tapped else {}
    out = {}
    for name, q in runner.named_parameters():
        if id(q) in tapped:
            out[name] = tapped[id(q)].cpu()
        elif q.grad is not None and not experts_only:
            out[name] = (q.grad.full_tensor() if hasattr(q.grad, "full_tensor") else q.grad).detach().float().cpu()
    return out


def grad_norm(model):
    total = torch.zeros(())
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
            total = total + g.float().pow(2).sum()
    return float(total.sqrt())


# ---- the tiny check ---------------------------------------------------------------------------------------------------

def tiny_teacher(config_path, seed=0, *, vocab_size=2048, experts=8, per_token=4):
    from transformers import AutoConfig, AutoModelForMultimodalLM

    data = json.loads(Path(config_path).read_text())
    data["text_config"].update(
        hidden_size=64, num_hidden_layers=6, layer_types=["linear_attention"] * 3 + ["full_attention"]
        + ["linear_attention", "full_attention"], num_attention_heads=2, num_key_value_heads=1,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=experts,
        num_experts_per_tok=per_token, hc_lowrank=8, ple_embed_dim=64, ngram_vocab_size_base=5000,
        split_ngram_parts=4, indexer_n_heads=2, indexer_budget=64, vocab_size=vocab_size)
    data["vision_config"].update(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64)
    data.pop("quantization_config", None)
    torch.manual_seed(seed)
    config = AutoConfig.for_model(**data)
    return AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)


def perturb(model, scale, seed):
    """Seeded relative noise on every floating tensor, so the tiny student starts measurably off the teacher."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            if p.is_floating_point() and p.numel() > 1:
                p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * scale * (p.float().std() + 1e-3).to(p.dtype))
    return model


def mtp_tiny(text_config, seed=3):
    head = MtpHead(text_config)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in head.named_parameters():
            if "norm" in name:
                p.zero_()
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return head.float()


def width_check(teacher, log, seed=5, out=None):
    """Width cut exactness: a teacher whose dropped channels carry nothing (their writers and embedding columns
    zeroed) gives the same logits as its width-cut copy (norms compensated)."""
    import copy

    from transformers import AutoModelForMultimodalLM

    text = teacher.config.text_config
    hidden, count = text.hidden_size, text.hc_count
    block = hidden // 4
    keep = width_blocks(torch.arange(hidden, dtype=torch.float32), block=block, keep_blocks=3)
    drop = torch.tensor([c for c in range(hidden) if c not in keep])
    drop_hc = torch.cat([drop + c * hidden for c in range(count)])
    model = copy.deepcopy(teacher).float()
    state = model.state_dict()
    with torch.no_grad():
        for name, tensor in state.items():
            axes = width_axes(name)
            writer = re.search(r"embed_tokens|o_proj|out_proj|down_proj|ple\.(value|key)_proj|linear_fc2", name)
            for axis, kind in axes.items():
                if writer and not re.search(r"lm_head", name):
                    tensor.index_fill_(axis, drop if kind == _H else drop_hc, 0.0)
    model.load_state_dict(state)
    cut_state = width_cut_state_dict(model.state_dict(), hidden=hidden, hc_count=count, keep=keep)
    cut = AutoModelForMultimodalLM.from_config(width_cut_config(model.config, len(keep))).float()
    missing, unexpected = cut.load_state_dict(cut_state, strict=False)
    theirs = dict(model.named_buffers())
    with torch.no_grad():  # the tiny teacher was built in BF16: its rotary inverse frequencies are rounded
        for name, buffer in cut.named_buffers():
            if "inv_freq" in name and name in theirs and theirs[name].shape == buffer.shape:
                buffer.copy_(theirs[name])
    for net in (model, cut):  # the norms' eps is the one inexact term (it meets a smaller mean square); tiny
        for module in net.modules():  # random weights have mean squares near it, so it is set aside here
            if isinstance(getattr(module, "eps", None), float):
                module.eps = 1e-12
            if hasattr(module, "block_topk"):  # the indexer keeps every block: its top-k meets exact ties (relu
                module.block_topk = 4096  # zeros) under random weights, and a tie flips on a 1e-7 change
                module.token_budget = module.block_topk * module.compress_ratio
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, 2000, (1, 200), generator=g)
    model.eval(), cut.eval()
    with torch.no_grad():
        a = model(input_ids=ids).logits.float()
        b = cut(input_ids=ids).logits.float()
    diff = float((a - b).abs().max() / a.abs().max())
    if out is not None:  # the on-disk cutter gives the same model
        model.save_pretrained(str(out / "tiny-silent"), max_shard_size="2MB")
        cut_checkpoint(out / "tiny-silent", out / "tiny-width", drop=[], width_keep=keep, shard_gib=0.002)
        disk = AutoModelForMultimodalLM.from_pretrained(str(out / "tiny-width"), dtype=torch.float32)
        da, db = disk.state_dict(), cut.state_dict()
        unequal = [k for k in db if "inv_freq" not in k and (k not in da or not torch.equal(da[k], db[k]))]
        event(log, check="width_cut_checkpoint_equals_in_memory", unequal=unequal[:5], ok=not unequal)
    event(log, check="width_cut_exact_on_silent_channels", width=f"{hidden}->{len(keep)}", rel_max_diff=diff,
          missing=[k for k in missing if "inv_freq" not in k][:5], unexpected=list(unexpected)[:5], ok=diff < 1e-3)


def run_tiny(args):
    import torch.distributed as dist

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("gloo")
    rank = dist.get_rank() if distributed else 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out if rank == 0 else None
    teacher = tiny_teacher(args.config)
    keep = [0, 1, 2, 3, 5]  # drop layer 4 (a linear-attention layer; PLE layers 0-2 kept)
    experts = {layer: [0, 2, 4, 6] for layer in keep}
    if rank == 0:  # the on-disk cutter gives the in-memory student
        from transformers import AutoModelForMultimodalLM

        teacher.save_pretrained(str(out / "tiny-teacher"), max_shard_size="2MB")
        (out / "prune.json").write_text(json.dumps({"format": "arc-expert-prune-v1", "experts": 8,
                                                    "pruned": {str(i): [1, 3, 5, 7] for i in range(6)}}))
        cut_checkpoint(out / "tiny-teacher", out / "tiny-student", drop=[4], prune_file=out / "prune.json", top_k=2,
                       shard_gib=0.002)
        disk = AutoModelForMultimodalLM.from_pretrained(str(out / "tiny-student"), dtype=torch.bfloat16)
        mem = build_student(teacher, keep_layers=keep, experts=experts, top_k=2)
        da, db = disk.state_dict(), mem.state_dict()
        unequal = [k for k in db if k not in da or da[k].shape != db[k].shape or not torch.equal(da[k], db[k])]
        event(log, check="cut_checkpoint_equals_in_memory", tensors=len(db), unequal=unequal[:5],
              extra=sorted(set(da) - set(db))[:5], ok=not unequal and set(da) == set(db))
    student = perturb(build_student(teacher, keep_layers=keep, experts=experts, top_k=2), 0.5, 7)
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(10, 2000, (1, 256), generator=g) for _ in range(4)]  # equal windows on every rank
    targets = [teacher_topk(teacher, s, k=args.k, window=128, vocab_chunk=64) for s in seqs]
    whole = [teacher_topk(teacher, s, k=args.k, window=10_000, vocab_chunk=64) for s in seqs]
    diff = max(float((a[0].float() - b[0].float()).abs().max()) for a, b in zip(targets, whole))
    event(log, check="teacher_windowed_vs_whole_max_abs_logp", value=diff, ok=diff < 0.05)

    student.float()  # fp32 masters (FSDP casts to BF16 for compute)
    with torch.no_grad():
        win = sum(train_sequence(Runner(student), s, t, window=96, backward=False)[0] for s, t in zip(seqs, targets))
        one = sum(train_sequence(Runner(student), s, t, window=10_000, backward=False)[0] for s, t in zip(seqs, targets))
    event(log, check="student_kd_windowed_vs_whole", windowed=win, whole=one, ok=abs(win - one) / one < 0.01)
    head = mtp_tiny(student.config.text_config)
    with torch.no_grad():
        rw = [train_sequence(Runner(student, head), s, t, window=96, backward=False) for s, t in zip(seqs, targets)]
        ro = [train_sequence(Runner(student, head), s, t, window=10_000, backward=False) for s, t in zip(seqs, targets)]
    mw, mo = sum(r[5] for r in rw), sum(r[5] for r in ro)
    event(log, check="mtp_kd_windowed_vs_whole", windowed=mw, whole=mo, positions=sum(r[6] for r in rw),
          main_unchanged=abs(sum(r[0] for r in rw) - win) < 1e-6 * max(1.0, win), ok=abs(mw - mo) / mo < 0.01)
    if args.real_mtp_keys:  # the head's names against the real checkpoint's mtp.* names (layer index generalised)
        real = {line.split()[0] for line in Path(args.real_mtp_keys).read_text().splitlines()
                if line.strip().startswith("mtp.")}
        mine = {"mtp." + re.sub(r"\.\d+\.", ".N.", k) for k in head.state_dict() if "inv_freq" not in k}
        event(log, check="mtp_names_match_checkpoint", missing=sorted(real - mine)[:8], extra=sorted(mine - real)[:8],
              ok=real == mine)

    width_check(teacher, log, out=out if rank == 0 else None)

    reference = None
    if distributed:  # one step single-process on rank 0 first, for the comparison
        if rank == 0:
            ref = perturb(build_student(teacher, keep_layers=keep, experts=experts, top_k=2), 0.5, 7).float()
            n = sum(float(m.sum()) for m in [torch.ones(s.shape[1]) for s in seqs])
            loss = sum(train_sequence(Runner(ref), s, t, window=96, scale=1 / n)[0] for s, t in zip(seqs, targets)) / n
            reference = (loss, grad_norm(ref))
        from torch.distributed.device_mesh import init_device_mesh

        model = shard(Runner(student), init_device_mesh("cpu", (dist.get_world_size(),)),
                      compute_dtype=torch.float32 if args.fp32 else torch.bfloat16)
    else:
        model = Runner(student)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    n = sum(s.shape[1] for s in seqs)
    losses = []
    for step in range(args.steps):
        mine = [i for i in range(len(seqs)) if not distributed or i % dist.get_world_size() == rank]
        world = dist.get_world_size() if distributed else 1
        kd = sum(train_sequence(model, seqs[i], targets[i], window=96, scale=world / n)[0] for i in mine)
        if distributed:
            box = torch.tensor([kd])
            dist.all_reduce(box)
            kd = float(box[0])
        norm = grad_norm(model)
        if step == 0 and reference is not None:
            event(log, check="fsdp_vs_single_step0", loss=kd / n, loss_single=reference[0], grad_norm=norm,
                  grad_norm_single=reference[1],
                  ok=abs(kd / n - reference[0]) < 1e-3 * max(1.0, reference[0]) and abs(norm - reference[1]) < 0.02 * reference[1])
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(kd / n)
        event(log, step=step, kd=kd / n, grad_norm=norm)
    event(log, check="loss_falls", first=losses[0], last=losses[-1], ok=losses[-1] < 0.7 * losses[0])
    if distributed:
        dist.destroy_process_group()


# ---- real data: teacher files (rl1_trainer op_topk) and the speed / memory read -------------------------------------

def loop_mask(tokens, n=8, window=64, frac=0.9, min_len=256, repeats=2):
    """Positions inside a degenerate repetition: from the first one whose 8-gram already occurred ``repeats`` times
    earlier in the reply and >= 90% of the next 64 8-grams did too, over that window. ``repeats`` 2 = a third pass over
    the same text: a block copied once (c1opd-v1, 19 Oct: a function drafted in the thinking and retyped in the tool
    call was cut as a "loop", with the tool call) is kept, and so is the text after a loop ends."""
    size = len(tokens)
    drop = [False] * size
    if size < min_len:
        return drop
    grams = [tuple(tokens[i:i + n]) for i in range(size - n + 1)]
    seen, first = {}, []
    for g in grams:
        first.append(seen.get(g, 0) >= repeats)
        seen[g] = seen.get(g, 0) + 1
    prefix = [0]
    for f in first:
        prefix.append(prefix[-1] + f)
    for i in range(len(first) - window + 1):
        if first[i] and prefix[i + window] - prefix[i] >= frac * window:
            for k in range(i, min(size, i + window + n - 1)):
                drop[k] = True
    return drop


def loop_onset(tokens, **kw):
    """The first position of :func:`loop_mask` (None: no loop)."""
    return next((i for i, d in enumerate(loop_mask(tokens, **kw)) if d), None)


def truncate_loops(data):
    """Reply tokens inside degenerate repetition (:func:`loop_mask`) stop being reply tokens (OPD: positions inside a
    loop teach looping; the teacher's distribution before and after it stays). Returns the edited dict and (spans,
    looped spans, reply tokens removed)."""
    reply = data["reply"].clone()
    ids = data["ids"].tolist()
    flags = reply.tolist()
    spans, looped, removed, i = 0, 0, 0, 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(flags) and flags[j]:
            j += 1
        spans += 1
        drop = loop_mask(ids[i:j])
        cut = [i + k for k, d in enumerate(drop) if d]
        if cut:
            looped += 1
            reply[cut] = False
            removed += len(cut)
        i = j
    return dict(data, reply=reply), (spans, looped, removed)


def load_teacher(path, device, k=None):
    """One ``op_topk`` file as the tensors ``train_sequence`` takes. Row t is the teacher's distribution of token t+1:
    scored where t+1 exists and is not an image placeholder; ``reply``: the scored rows whose next token is a reply
    token (assistant-position KL, reported apart); ``labels``: the next token on scored rows (the LM term)."""
    return teacher_to(load_teacher_cpu(path, k), device)


def load_teacher_cpu(path, k=None):
    """load_teacher's tensors on the host (``teacher_to`` moves them); ``host``: the ids, the reply mask and the
    position counts kept on the host (HEAL_LAZY_STATS / HEAL_PLE_TABLE / HEAL_PREFETCH)."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    ids = data["ids"].long()
    total = ids.numel()
    scored = torch.zeros(total, dtype=torch.bool)
    scored[:-1] = ~data["image"][1:]
    reply = torch.zeros(total, dtype=torch.bool)
    reply[:-1] = data["reply"][1:]
    reply &= scored
    labels = torch.full((total,), -100, dtype=torch.long)
    labels[:-1] = torch.where(scored[:-1], ids[1:], torch.full_like(ids[1:], -100))
    inputs = {"input_ids": ids[None]}
    for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        if key in data:
            inputs[key] = data[key]
    top, idx, tail = data["top_logp"], data["top_ids"], data["tail"]
    row0 = int(data.get("row0", 0))  # a streamed chunk (OPD stream, 22 Oct): teacher rows row0.. only; the prefix
    if row0:  # before it is context (trained in earlier chunks), padded and masked out here
        pad = lambda t, v: torch.cat([torch.full((row0, *t.shape[1:]), v, dtype=t.dtype), t])  # noqa: E731
        top, idx, tail = pad(top, 0), pad(idx, 0), pad(tail, 0)
        scored[:row0] = False
        reply[:row0] = False
        labels[:row0] = -100
    if top.shape[0] != total:
        raise ValueError(f"{path}: teacher rows {top.shape[0]} vs {total} tokens (row0 {row0})")
    if k is not None and k < top.shape[1]:  # the same file read as a top-k target (the KD-bias comparison)
        top, idx = top[:, :k], idx[:, :k]
        tail = torch.log1p(-top.double().exp().sum(-1).clamp(max=1 - 1e-6)).half()
    if FORCE_IDS and "force_ids" in data:  # HEAL_FORCE_IDS: the forced ids as their own columns (row0 padding: 0)
        if [int(x) for x in data["force_ids"]] != list(FORCE_IDS):
            raise ValueError(f"{path}: forced ids {list(data['force_ids'])} vs HEAL_FORCE_IDS {list(FORCE_IDS)}")
        f_lp = data["force_logp"]
        if row0:
            f_lp = torch.cat([torch.zeros((row0, f_lp.shape[1]), dtype=f_lp.dtype), f_lp])
        top, idx, tail = merge_forced(top, idx, data["force_ids"], f_lp)
    elif FORCE_IDS:  # a piece scored without them: NaN columns (they count for neither side), one width per batch
        top = torch.cat([top, torch.full((top.shape[0], len(FORCE_IDS)), float("nan"), dtype=top.dtype)], 1)
        idx = torch.cat([idx, torch.tensor(FORCE_IDS, dtype=idx.dtype)[None].expand(idx.shape[0], -1)], 1)
    t_tok = torch.full((total,), float("nan"))
    if "tok_logp" in data:  # recorded by op_topk (22 Oct): the teacher's log p of the realised next token
        t_tok[row0:total - 1] = data["tok_logp"][:total - 1 - row0].float()
    elif os.environ.get("HEAL_SAMPLED") and os.environ.get("HEAL_ALLOW_TOPK_TTOK") != "1" and \
            Path(path).name.startswith(sampled_settings()[2]):
        # untruncated sampling draws tail tokens; a top-k lookup drops exactly those (NaN), biasing the term toward
        # the head: a student-sampled file needs the teacher's realised-token log p (op_topk since 22 Oct 02:20)
        raise ValueError(f"{path}: no tok_logp (teacher log p of the realised token); rescore it, or set "
                         "HEAL_ALLOW_TOPK_TTOK=1 to accept the top-k lookup")
    else:  # older files: the top-k lookup (NaN where the realised token is outside the teacher's top-k)
        hit = data["top_ids"][:-1].long() == ids[1:, None]
        t_tok[:-1] = torch.where(hit.any(-1), (data["top_logp"][:-1].float() * hit).sum(-1), float("nan"))
    if THINK_EOS:  # HEAL_THINK_EOS_MASK: no EOS inside an open think block (the reply tokens, their rows)
        rows = think_rows(ids, data["reply"].bool(), THINK_EOS[1]) & scored
        rows[:row0] = False
        nxt = torch.cat([ids[1:], torch.full((1,), -1, dtype=ids.dtype)])
        top, tail, t_tok, n_m, n_u = mask_think_eos(top, idx, tail, rows, THINK_EOS[0], t_tok, nxt)
        THINK_ACC["masked"] = THINK_ACC.get("masked", 0) + n_m
        THINK_ACC["no_eos_column"] = THINK_ACC.get("no_eos_column", 0) + n_u
    b = None
    if "beh_logp" in data:  # the serving engine's log p of each sampled next token (row t: token t+1; NaN elsewhere)
        b = torch.full((total,), float("nan"))
        b[row0:] = data["beh_logp"].float()[:total - row0]
    return {"inputs": inputs, "teacher": (top, idx, tail), "mask": scored.float(), "reply": reply.float(),
            "labels": labels, "w": data.get("w", 1.0), "name": Path(path).stem, "t_tok": t_tok, "b": b,
            "version": data.get("version"), "row0": row0}


def merge_pieces(cpus):
    """HEAL_STREAM_COALESCE (trainer MFU S2, 23 Oct): consecutive streamed pieces of one chain (load_teacher_cpu dicts
    in piece order, each piece's ids a prefix of the next's) as one sequence: the last piece's ids and images, each
    piece's own rows [its row0, the next piece's row0) of the teacher rows, masks, labels, realised-token and
    behavior log p (a piece's last row is the next piece's first); loop counts summed. One forward over the chain
    instead of one per piece over its whole prefix."""
    last = cpus[-1]
    top, idx, tail = (t.clone() for t in last["teacher"])
    rows = {k: last[k].clone() for k in ("mask", "reply", "labels", "t_tok")}
    b = last["b"].clone() if last.get("b") is not None else None
    for c, nxt in zip(cpus[:-1], cpus[1:]):
        lo, hi = c["row0"], nxt["row0"]
        for dst, src in zip((top, idx, tail), c["teacher"]):
            dst[lo:hi] = src[lo:hi]
        for k, dst in rows.items():
            dst[lo:hi] = c[k][lo:hi]
        if b is not None:
            b[lo:hi] = c["b"][lo:hi] if c.get("b") is not None else float("nan")
    out = dict(last, teacher=(top, idx, tail), b=b, row0=cpus[0]["row0"], pieces=len(cpus), **rows)
    if any(c.get("loops") is not None for c in cpus):
        out["loops"] = [sum(float((c.get("loops") or (0.0, 0.0, 0.0))[j]) for c in cpus) for j in range(3)]
    return out


def pin_teacher(cpu):
    """Page-locked copies of a host teacher dict's tensors (non-blocking device copies; HEAL_PREFETCH on CUDA)."""
    pin = lambda t: t.pin_memory() if isinstance(t, torch.Tensor) else t  # noqa: E731
    out = dict(cpu, inputs={k: pin(v) for k, v in cpu["inputs"].items()}, teacher=tuple(pin(t) for t in cpu["teacher"]))
    for key in ("mask", "reply", "labels"):
        out[key] = pin(cpu[key])
    if out.get("ple") is not None:
        out["ple"] = {i: pin(t) for i, t in out["ple"].items()}
    return out


def teacher_to(cpu, device, non_blocking=False):
    """load_teacher's dict on ``device`` from load_teacher_cpu's; ``host``: the host ids, reply mask (after any loop
    cut) and position counts; ``ple`` (HEAL_PLE_TABLE rows) stays on the host."""
    to = lambda t: t.to(device, non_blocking=non_blocking)  # noqa: E731
    inputs = {k: to(v) for k, v in cpu["inputs"].items()}
    reply = cpu["reply"]
    return {"ids": inputs["input_ids"], "inputs": inputs, "teacher": tuple(to(t) for t in cpu["teacher"]),
            "mask": to(cpu["mask"]), "reply": to(reply), "labels": to(cpu["labels"]), "w": cpu["w"], "name": cpu["name"],
            "t_tok": to(cpu["t_tok"]), "b": to(cpu["b"]) if cpu.get("b") is not None else None,
            "version": cpu.get("version"),
            "host": {"ids": cpu["inputs"]["input_ids"][0], "reply": reply, "mask_n": float(cpu["mask"].sum()),
                     "reply_n": float(reply.sum())}, "ple": cpu.get("ple"), "loops": cpu.get("loops")}


def fast_attention(mode="triton"):
    """The rl1 trainer's patches of the full-attention layers (v3: the stock QSA indexer loops over queries in Python,
    360 tok/s): the vectorized indexer and the selected-key Triton attention (rl1_sparse)."""
    if os.environ.get("HEAL_NO_FAST_ATTENTION", "") not in ("", "0"):  # the CPU tiny run (no Triton)
        return
    os.environ.setdefault("RL1_ATTENTION", mode)
    import rl1_trainer

    rl1_trainer.vectorized_indexer()
    rl1_trainer.selected_attention()
    install_mask_host()
    if os.environ.get("HEAL_HYPER_FAST") == "1":  # MFU-1 (24 Oct): rl1_hyper's fused hyper-connections (trainable
        if compile_mode():  # weights: eager's weight-grad ops; bitwise vs the model's code on a BF16 stream)
            raise ValueError("HEAL_HYPER_FAST with HEAL_COMPILE: both patch the decoder layer")
        import rl1_hyper

        rl1_hyper.install()
        rl1_hyper.HYPER["on"] = True
    if os.environ.get("HEAL_FLA_CONV") == "1":
        install_fla_conv()
        FLA_CONV["on"] = True


FLA_CONV = {"on": False, "calls": 0, "fallback": 0}


def install_fla_conv():
    """HEAL_FLA_CONV (MFU-1): the GDN depthwise causal conv (kernel 4) through fla's Triton causal_conv1d instead of
    transformers' F.conv1d fallback (cuDNN conv_depthwise2d, ~10x off the memory roofline at 32K tokens). The
    activation stays outside, as in the fallback (conv rounded to the weight dtype, then SiLU). Dispatch per call on
    FLA_CONV["on"]."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    if "conv_fn" in _ORIGINAL:
        return
    _ORIGINAL["conv_fn"] = orig = modeling.causal_conv1d_fn

    def conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        if not FLA_CONV["on"] or kwargs or not (hidden_states.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
            FLA_CONV["fallback"] += FLA_CONV["on"]
            return orig(hidden_states, weight, bias, activation, **kwargs)
        from fla.modules.convolution import causal_conv1d as fla_conv

        from transformers.activations import ACT2FN

        FLA_CONV["calls"] += 1
        x = hidden_states.to(weight.dtype).transpose(1, 2)  # [B, T, C] (the in_proj layout: no copy)
        y = fla_conv(x.contiguous(), weight, bias, activation=None)
        y = (y[0] if isinstance(y, tuple) else y).transpose(1, 2)
        if activation is not None:
            y = ACT2FN[activation](y)
        return y.to(hidden_states.dtype)

    modeling.causal_conv1d_fn = conv


def prefetch_files(folder, rank, world, threads=8, chunk=64 << 20):
    """HEAL_LOAD_PREFETCH=1 (MFU-2, 23 Oct): the student folder's *.safetensors (symlinks resolved) read once into the
    host page cache, split by size over the ranks, ``threads`` readers each, before from_pretrained. c3: three ranks
    each read the full 130 GiB from the runs volume (386 s at ~1.1 GiB/s together); a page-cache re-read runs at
    16-27 GiB/s (volbench). Returns (files, bytes read by this rank)."""
    from concurrent.futures import ThreadPoolExecutor

    files = sorted({Path(f).resolve() for f in Path(folder).glob("*.safetensors")}, key=lambda f: (-f.stat().st_size, str(f)))
    mine, load = [], [0] * world
    for f in files:  # largest first to the least-loaded rank: the same split on every rank
        r = load.index(min(load))
        load[r] += f.stat().st_size
        if r == rank:
            mine.append(f)

    def read(f):
        n = 0
        with open(f, "rb", buffering=0) as h:
            while b := h.read(chunk):
                n += len(b)
        return n

    with ThreadPoolExecutor(max(1, threads)) as ex:
        return len(files), sum(ex.map(read, mine))


def load_student(path, device, train=True, top_k=None):
    """The cut BF16 student on ``device`` with the n-gram table in host RAM (flash_qlora_job.keep_table_in_host_ram);
    the table and the vision tower frozen, every other weight trainable in fp32 (FSDP computes in BF16)."""
    import flash_qlora_job as fq
    from transformers import AutoModelForMultimodalLM
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    fq.keep_table_in_host_ram()
    fast_attention()
    extra = {}
    if top_k:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(path))
        config.text_config.num_experts_per_tok = top_k
        extra["config"] = config
    model = AutoModelForMultimodalLM.from_pretrained(str(path), dtype=torch.bfloat16, device_map={"": device},
                                                     experts_implementation=os.environ.get("HEAL_EXPERTS", "grouped_mm"),
                                                     **extra)
    modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True  # the window pass only reads the cache
    if s4("PLE_TABLE") or HOST.on:
        install_ple(model)
    frozen = []
    for name, p in model.named_parameters():
        if not train:
            p.requires_grad_(False)
        elif "ngram_embedding." in name or ".visual." in name or name.startswith("model.visual"):
            p.requires_grad_(False)
            frozen.append(p)
        else:
            p.data = p.data.float()
    if not train:
        return model.eval(), frozen
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_recompute(model)
    model.config.use_cache = True
    return model, frozen


def probe_kl(runner, folder, device, *, window, chunk, limit=None, k=None):
    """Mean KD (teacher top-k + tail to student) over the scored positions of the files in ``folder``, no
    gradients: the general-text probe (``op_topk``'s wikitext windows)."""
    kd = n = 0.0
    files = sorted(Path(folder).glob("*.pt"))[:limit]
    for path in files:
        data = load_teacher(path, device, k=k)
        out = train_sequence(runner, data["ids"], data["teacher"], window=window, mask=data["mask"], chunk=chunk,
                             backward=False)
        kd, n = kd + out[0], n + out[2]
    return {"files": len(files), "positions": int(n), "kd": round(kd / max(n, 1.0), 5)}


def sampled_settings():
    """HEAL_SAMPLED=<weight>: the sampled-token reverse term (GKD / on-policy distillation) on student-sampled files
    (names starting HEAL_STUDENT_PREFIX, default "s-"); HEAL_SAMPLER="T,top_k,top_p" (the rollouts' sampling).
    Default T 1, no top-k / top-p: OPD rollouts sample the student's whole distribution, so the
    sampled reverse term sees its tail (T 0.7 / top-k 20 / top-p 0.95 never drew a tail token: P1-x moved mass
    outside the teacher's top-32 uncharged)."""
    weight = float(os.environ.get("HEAL_SAMPLED") or 0.0)
    t, k, p = (os.environ.get("HEAL_SAMPLER") or "1.0,0,1.0").split(",")
    return weight, (float(t), int(k), float(p)), os.environ.get("HEAL_STUDENT_PREFIX", "s-")


BEH_CACHE = {}  # str(path) -> log b(next token) per position (the round-start student's sampling), CPU float32


def behavior_pass(runner, model, paths, device, *, window, chunk, sampler):
    out = _behavior_pass(runner, model, paths, device, window=window, chunk=chunk, sampler=sampler)
    if shard_on() and os.environ.get("HEAL_SHARD_CHECK") == "1" and "behavior" not in SHARD_CHECKED:
        SHARD_CHECKED.add("behavior")  # preflight, once: sharded rows == every-rank rows
        os.environ["HEAL_SHARD_READS"] = "0"
        try:
            ref = _behavior_pass(runner, model, paths, device, window=window, chunk=chunk, sampler=sampler)
        finally:
            os.environ["HEAL_SHARD_READS"] = "1"
        for k, v in ref.items():
            a, b = out[k], v
            if not (torch.isnan(a) == torch.isnan(b)).all() or (a - b).nan_to_num().abs().max() > 1e-3:
                raise RuntimeError(f"shard check: behavior rows differ on {k}")
        print(f"shard check: behavior rows match on {len(ref)} files", flush=True)
    return out


def _behavior_pass(runner, model, paths, device, *, window, chunk, sampler):
    """log b(x_{t+1}) per position of each file under the current weights with the rollouts' sampling transform (the
    behavior policy for the importance weights; run once at the round's start, every rank on the same files)."""
    import flash_qlora_job as fq

    out = {}
    train_on = TRAIN_ACC.pop("on", None)
    if shard_on():  # HEAL_SHARD_READS: every rank scores its share, then the rows are gathered
        import torch.distributed as dist

        def build(path):
            data = load_teacher(path, device)
            BEH_ACC.clear()
            BEH_ACC.update(on=True, sampler=sampler)
            build.last = (path, data)
            return {"data": data, "mask": data["mask"], "labels": data["labels"],
                    "piece": fq.positions_and_images(model, data["inputs"])}

        def collect():
            path, data = build.last
            rows = torch.cat(BEH_ACC.get("rows", []))
            BEH_ACC.clear()
            total = data["ids"].shape[1]
            if rows.numel() != total:
                raise RuntimeError(f"behavior pass: {rows.numel()} rows for {total} positions ({path})")
            b = rows.float()
            b[data["labels"].cpu() < 0] = float("nan")
            out[str(path)] = b
            build.last = None

        build.last = None
        try:
            with torch.no_grad():
                shard_read(runner, model, paths, device, window=window, chunk=chunk, build=build, after=collect)
        finally:
            BEH_ACC.clear()
        every = [None] * dist.get_world_size()
        dist.all_gather_object(every, out, group=ctl_group())
        out = {k: v for d in every for k, v in d.items()}
        missing = [str(p) for p in paths if str(p) not in out]
        if missing:
            raise RuntimeError(f"behavior pass: no rows for {len(missing)} files ({missing[:2]})")
        if train_on:
            TRAIN_ACC["on"] = train_on
        return out
    with torch.no_grad():
        for path in paths:
            data = load_teacher(path, device)
            piece = fq.positions_and_images(model, data["inputs"])
            BEH_ACC.clear()
            BEH_ACC.update(on=True, sampler=sampler)
            try:
                train_sequence(runner, data["ids"], data["teacher"], window=window, mask=data["mask"],
                               labels=data["labels"], chunk=chunk, backward=False, piece=piece)
                rows = torch.cat(BEH_ACC.get("rows", []))
            finally:
                BEH_ACC.clear()
            total = data["ids"].shape[1]
            if rows.numel() != total:
                raise RuntimeError(f"behavior pass: {rows.numel()} rows for {total} positions ({path})")
            b = rows.float()
            b[data["labels"].cpu() < 0] = float("nan")
            out[str(path)] = b
            del data, piece
    if train_on:
        TRAIN_ACC["on"] = train_on
    return out


def slot_samp(data, path, prefix, any_name=False):
    """(next ids, teacher log p, behavior log p, mask) of a student-sampled file's realised reply tokens, or None."""
    b = BEH_CACHE.get(str(path))
    if b is None:
        b = data.get("b")  # streamed chunks carry the engine's log-probs (the weights that sampled them)
    if b is None or not (any_name or Path(path).name.startswith(prefix)):
        return None
    b = b.to(data["labels"].device)
    t = data["t_tok"]
    mask = (data["reply"] > 0) & torch.isfinite(t) & torch.isfinite(b) & (data["labels"] >= 0)
    return data["labels"], t, b, mask


def own_positions(files, positions="reply"):
    """Trained positions per teacher file, in order: the folder's index.json (reply_tokens / tokens per file, the
    teacher pass's record), else the file itself."""
    rows = {}
    for folder in {Path(f).parent for f in files}:
        index = folder / "index.json"
        if index.is_file():
            rows.update({str(folder / r["file"]): r for r in json.loads(index.read_text()).get("rows", []) if "file" in r})
    key, out = "reply_tokens" if positions == "reply" else "tokens", []
    for f in files:
        r = rows.get(str(f))
        if r is not None and key in r:
            out.append(int(r[key]))
        else:
            d = torch.load(f, map_location="cpu", weights_only=True)
            out.append(int(d["reply"].sum()) if positions == "reply" else int(d["ids"].numel()))
    return out


def simulate_steps(own, general_n, world, batch, share, gen_trained=0.0, own_trained=0.0, flush_last=False):
    """Optimizer steps ``run_train`` takes on this queue (``own``: trained positions per queued file, in order), by
    the loop's own rules: slots of ``world`` files, general slots (``general_n`` positions per window, 0: none)
    while general positions trail share / (1 - share) of own ones, a step once the batch is full, the partial last
    batch dropped (``flush_last``: one more step on it). A step overshoots the batch by part of a slot, so
    positions / batch over-counts (c1opd-v1)."""
    steps, acc, i = 0, 0.0, 0
    while True:
        if general_n and gen_trained < share / max(1e-6, 1 - share) * own_trained:
            n = general_n * world
            gen_trained += n
        else:
            slot = own[i:i + world]
            i += world
            if not slot:
                return steps + (1 if flush_last and acc > 0 else 0)
            n = float(sum(slot))
            own_trained += n
        acc += n
        if acc >= batch:
            steps, acc = steps + 1, 0.0


def lr_at(step, args):
    """The schedule over HEAL_SCHEDULE_STEPS (the whole multi-round run; default this run's --total-steps).
    HEAL_WSD_DECAY: the main line's decay share (0 = stable to the end). HEAL_DECAY_FROM/HEAL_DECAY_STEPS: a decay
    branch, linear from the schedule's lr at DECAY_FROM to 0 over DECAY_STEPS (the read at a token point, WSD).
    HEAL_REWARM=N: the first N steps after --step0 ramp up to the schedule (warm-up on every start and resume)."""
    env = os.environ
    total = int(env.get("HEAL_SCHEDULE_STEPS") or args.total_steps)

    def base(k):
        if args.schedule == "cosine":
            return cosine_lr(k, total, args.lr, warmup=args.warmup)
        return wsd_lr(k, total, args.lr, warmup=args.warmup, decay=float(env.get("HEAL_WSD_DECAY") or 0.2))

    lr = base(step)
    d0, dn = int(env.get("HEAL_DECAY_FROM") or 0), int(env.get("HEAL_DECAY_STEPS") or 0)
    if dn and step >= d0:
        lr = base(d0) * max(0.0, (d0 + dn - step) / dn)
    rewarm = int(env.get("HEAL_REWARM") or 0)
    if rewarm and args.step0 <= step < args.step0 + rewarm:
        lr *= (step - args.step0 + 1) / (rewarm + 1)
    return lr


class LiveCkpt:
    """HEAL_LIVE_KNOBS "ckpt" (c7n 25 Oct): a checkpoint at the next step boundary whenever the tag changes from the
    one seen at the previous boundary (the tag present at the first read only arms it). Rank 0 reads, every rank takes
    its value (one gather), so all ranks save together."""

    def __init__(self, path):
        self.path, self.seen, self.first = path, None, True

    def _read(self):
        try:
            v = json.loads(Path(self.path).read_text()).get("ckpt")
            return str(v) if v is not None else None
        except (OSError, ValueError, AttributeError):
            return None

    def due(self, rank):
        import torch.distributed as dist
        mine = self._read() if rank == 0 else None
        tag = ctl_gather(mine)[0] if dist.is_initialized() else mine
        fire = not self.first and tag is not None and tag != self.seen
        self.first, self.seen = False, tag
        return fire


class LiveLr:
    """HEAL_LIVE_KNOBS=<json>: the peak lr as a live knob ({"lr": x}, written by the round's puller from
    <run>/stream/knobs.json). Rank 0 reads the file every step and every rank takes its value (one gather), so the
    ranks never differ. The step's lr is lr_at(step) * x / --lr; a change ramps linearly from the old scale to the new
    over HEAL_LIVE_LR_RAMP steps (default 4: a warm-up on each change). The value present at the first read applies
    at once (HEAL_REWARM covers the start). No file or no "lr": the schedule as launched."""

    def __init__(self, path, base, ramp=4):
        self.path, self.base, self.ramp = path, float(base), max(1, int(ramp))
        self.old = self.target = 1.0
        self.since, self.seen, self.first = 0, None, True

    def _read(self):
        try:
            v = json.loads(Path(self.path).read_text()).get("lr")
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 1e-4 else None
        except (OSError, ValueError, AttributeError):
            return None

    def scale(self, step, rank):
        import torch.distributed as dist
        mine = self._read() if rank == 0 else None
        want = ctl_gather(mine)[0] if dist.is_initialized() else mine
        target = want / self.base if want is not None else 1.0
        if target != self.target:
            self.old = target if self.first else self.current(step)
            self.target, self.since = target, step
        self.first, self.seen = False, want
        return self.current(step)

    def current(self, step):
        if self.old == self.target:
            return self.target
        f = min(1.0, (step - self.since + 1) / self.ramp)
        return self.old + (self.target - self.old) * f


SHARD_K = {}  # the teacher top-k of the read files (the padding dummy's width)
SHARD_CHECKED = set()  # HEAL_SHARD_CHECK: the reads already compared


def shard_on():
    import torch.distributed as dist
    return os.environ.get("HEAL_SHARD_READS") == "1" and dist.is_initialized() and dist.get_world_size() > 1


def acc_snap():
    """The read accumulators' running sums (per-file read values: differences around each file)."""
    g = lambda acc, k: float(acc.get(k, 0.0))  # noqa: E731
    return (g(REV_ACC, "kl"), g(REV_ACC, "n"), g(SAMP_ACC, "est"), g(SAMP_ACC, "n"), g(READ_ACC, "tail32"))


def file_row(kr, nr, before, after):
    d = [a - b for a, b in zip(after, before)]
    return {"kd_reply": kr, "reply_positions": nr, "reverse": d[0], "reverse_n": d[1], "sampled": d[2],
            "sampled_n": d[3], "tail32": d[4]}


def shard_read(runner, model, paths, device, *, window, chunk, build, after=None, record=None):
    """HEAL_SHARD_READS=1 (22 Oct): each rank reads its share of ``paths`` (largest first, dealt round-robin) in
    lockstep slots of dp_sequence without backward, padded with a zero-mask dummy; ``build(path)`` -> the slot dict,
    ``after()`` runs after each of this rank's own slots.
    Returns this rank's (kd, n, kd_reply, n_reply) sums; the caller reduces them and the read accumulators."""
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    order = sorted((str(p) for p in paths), key=lambda f: (-Path(f).stat().st_size, f))
    if not order:
        return 0.0, 0.0, 0.0, 0.0
    if "k" not in SHARD_K:
        SHARD_K["k"] = int(torch.load(order[0], map_location="cpu", weights_only=True, mmap=True)["top_ids"].shape[1])
    dummy = dp_dummy(model, device, k=SHARD_K["k"])
    mine_files = order[rank::world]
    kd = n = kr = nr = 0.0
    for i in range(-(-len(order) // world)):
        f = mine_files[i] if i < len(mine_files) else None
        before = acc_snap() if (record is not None and f is not None) else None
        got = dp_sequence(runner, build(f) if f is not None else None, dummy, window=window, chunk=chunk, scale=0.0,
                          mtp_weight=0.0, backward=False)
        kd, n, kr, nr = kd + float(got["kd"]), n + got["n"], kr + float(got["kd_reply"]), nr + got["n_reply"]
        if before is not None:
            record[Path(f).name] = file_row(float(got["kd_reply"]), float(got["n_reply"]), before, acc_snap())
        if f is not None and after is not None:
            after()
    return kd, n, kr, nr


def reduce_acc(acc, keys, device):
    """Sums the read accumulator ``acc`` over ranks in place (missing keys count 0)."""
    import torch.distributed as dist

    t = torch.tensor([float(acc.get(k, 0.0)) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t)
    for k, v in zip(keys, t.tolist()):
        acc[k] = v


def heldout_kl(runner, model, folder, device, *, window, chunk, limit=None, samp_prefix=None):
    out = _heldout_kl(runner, model, folder, device, window=window, chunk=chunk, limit=limit, samp_prefix=samp_prefix)
    if shard_on() and os.environ.get("HEAL_SHARD_CHECK") == "1" and str(folder) not in SHARD_CHECKED:
        SHARD_CHECKED.add(str(folder))  # preflight, once per folder: sharded read == every-rank read
        os.environ["HEAL_SHARD_READS"] = "0"
        try:
            ref = _heldout_kl(runner, model, folder, device, window=window, chunk=chunk, limit=limit,
                              samp_prefix=samp_prefix)
        finally:
            os.environ["HEAL_SHARD_READS"] = "1"
        bad = {k: (out.get(k), v) for k, v in ref.items() if isinstance(v, (int, float))
               and abs(float(out.get(k, float("nan"))) - v) > 2e-3 * max(abs(v), 1e-2)}
        if ref.get("by_class") != out.get("by_class") and ref.get("by_class"):
            for c, r in ref["by_class"].items():
                o = out["by_class"][c]
                if o["positions"] != r["positions"] or abs(o["kd"] - r["kd"]) > 2e-3 * max(abs(r["kd"]), 1e-2):
                    bad[f"class_{c}"] = (o, r)
        if bad:
            raise RuntimeError(f"shard check: read differs on {folder}: {bad}")
        print(f"shard check: read matches on {folder} ({ref['files']} files)", flush=True)
    return out


def truncate_cpu(cpu, upto):
    """A load_teacher_cpu dict cut to its first ``upto`` tokens (back to an image's start when one straddles the cut;
    whole images only); the last kept row is unscored (its next token is cut)."""
    inputs = cpu["inputs"]
    ids = inputs["input_ids"][0]
    mm = inputs.get("mm_token_type_ids")
    img = (mm[0] > 0) if mm is not None else torch.zeros_like(ids, dtype=torch.bool)
    upto = min(int(upto), int(ids.shape[0]))
    while 0 < upto < ids.shape[0] and img[upto] and img[upto - 1]:
        upto -= 1
    out = dict(cpu)
    new = {"input_ids": inputs["input_ids"][:, :upto]}
    if mm is not None:
        new["mm_token_type_ids"] = mm[:, :upto]
    if inputs.get("pixel_values") is not None:
        edges = torch.diff(torch.nn.functional.pad(img.to(torch.int8), (1, 1)))
        stops = (edges == -1).nonzero().flatten().tolist()
        keep = sum(1 for b in stops if b <= upto)
        if keep:
            grid = inputs["image_grid_thw"]
            new["pixel_values"] = inputs["pixel_values"][:int(grid[:keep].prod(-1).sum())]
            new["image_grid_thw"] = grid[:keep]
    out["inputs"] = new
    out["teacher"] = tuple(t[:upto] for t in cpu["teacher"])
    for key in ("mask", "reply", "labels", "t_tok", "b"):
        if cpu.get(key) is not None:
            out[key] = cpu[key][:upto].clone()
    out["mask"][-1], out["reply"][-1], out["labels"][-1] = 0, 0, -100
    return out


def reply_rows_within(path, tokens):
    """Scored reply rows among a teacher file's first ``tokens`` positions (load_teacher_cpu's rule, after
    truncate_cpu's cut: the last kept row is unscored)."""
    cpu = truncate_cpu(load_teacher_cpu(path), tokens)
    return int(cpu["reply"].sum())


class StepHeldout:
    """HEAL_STEP_HELDOUT=<tokens>: a per-step held-out KD curve. Each rank keeps one fixed HEAL_HELDOUT2
    file (rank r: the r-th by read_files order among those with reply rows inside the cut; c7m 24 Oct: the first three
    dh10 files had none in their first 16K tokens and the curve read 0) cut to ``tokens``, on the card; after every
    optimizer step every rank reads its file without gradient in one lockstep dp_sequence slot (the read accumulators
    and the step's training sums saved and restored: gradient- and log-neutral). Returns per-rank reply KD and its
    mean over ranks."""

    def __init__(self, runner, model, folder, device, *, tokens, window, chunk, rank, world):
        files = [f for f in read_files(folder) if reply_rows_within(f, tokens) > 0]
        if len(files) < world:
            raise ValueError(f"HEAL_STEP_HELDOUT: {len(files)} files in {folder} with reply rows in their first {tokens} "
                             f"tokens for {world} ranks")
        self.runner, self.window, self.chunk = runner, window, chunk
        self.file = files[rank]
        cpu = truncate_cpu(load_teacher_cpu(self.file), tokens)
        self.data = teacher_to(cpu, device)
        import flash_qlora_job as fq
        self.slot = {"data": self.data, "mask": self.data["mask"], "piece": fq.positions_and_images(model, self.data["inputs"])}
        self.dummy = dp_dummy(model, device, k=int(self.data["teacher"][0].shape[1]))
        self.reply_rows = int(self.data["host"]["reply_n"])
        self.tokens = int(self.data["ids"].shape[1])

    def setup_row(self):
        every = ctl_gather([self.file.name, self.tokens, self.reply_rows])
        return {"files": [e[0] for e in every], "tokens": [e[1] for e in every], "reply_rows": [e[2] for e in every]}

    def read(self):
        saved = {name: dict(acc) for name, acc in (("TRAIN", TRAIN_ACC), ("REV", REV_ACC), ("READ", READ_ACC),
                                                   ("SAMP", SAMP_ACC), ("CLASS", CLASS_ACC))}
        for acc in (TRAIN_ACC, REV_ACC, READ_ACC, SAMP_ACC, CLASS_ACC):
            acc.clear()
        try:
            with torch.no_grad():
                got = dp_sequence(self.runner, self.slot, self.dummy, window=self.window, chunk=self.chunk, scale=0.0,
                                  mtp_weight=0.0, backward=False)
            kr, nr = float(got["kd_reply"]), float(got["n_reply"])
        finally:
            reshard_all(self.runner)
            for name, acc in (("TRAIN", TRAIN_ACC), ("REV", REV_ACC), ("READ", READ_ACC), ("SAMP", SAMP_ACC),
                              ("CLASS", CLASS_ACC)):
                acc.clear()
                acc.update(saved[name])
        per = [round(k / max(n, 1), 6) for k, n in ctl_gather([kr, nr])]
        return {"heldout_kd": round(sum(per) / len(per), 6), "heldout_kd_rank": per}


class FreshHoldout:
    """HEAL_FRESH_HOLDOUT=<share>: per step, each rank reads one unread piece of the held-out student
    chains (rank 0 picks the ``world`` freshest: highest version, newest), no gradient, in one lockstep dp_sequence
    slot (accumulators saved and restored as StepHeldout); a rank without a piece pads. Each piece is read once, so
    the curve follows the live play at the cost of piece-to-piece noise. Logs the mean reply KD over the ranks with a
    piece, each rank's value (None: no piece) and the pieces' data ages."""

    def __init__(self, runner, model, stream, device, *, window, chunk, rank, world):
        self.runner, self.model, self.stream, self.device = runner, model, stream, device
        self.window, self.chunk, self.rank, self.world = window, chunk, rank, world
        self.dummy = None

    def read(self, version):
        import flash_qlora_job as fq
        if self.rank == 0:
            self.stream.ready()  # refresh the held-out list
            pick = sorted(self.stream.held, key=lambda f: self.stream.held[f], reverse=True)[:self.world]
            ages = [version - self.stream.held[f][0] for f in pick]
            for f in pick:  # read once: never listed again (taken)
                del self.stream.held[f]
                self.stream.taken.add(f)
        else:
            pick, ages = None, None
        pick, ages = ctl_gather((pick, ages))[0]
        if not pick:
            return {}
        mine, k = None, None
        if self.rank < len(pick):
            cpu = load_teacher_cpu(pick[self.rank])
            data = teacher_to(cpu, self.device)
            row0 = int(cpu.get("row0") or 0)
            mine = {"data": data, "mask": data["mask"], "piece": fq.positions_and_images(self.model, data["inputs"]),
                    "prefix": max(row0 - 1, 0) // 64 * 64 if os.environ.get("HEAL_INERT_PREFIX") == "1" else 0}
            k = int(data["teacher"][0].shape[1])
        k = next(x for x in ctl_gather(k) if x is not None)
        if self.dummy is None:
            self.dummy = dp_dummy(self.model, self.device, k=k)
        accs = (TRAIN_ACC, REV_ACC, READ_ACC, SAMP_ACC, CLASS_ACC)
        saved = [dict(a) for a in accs]
        for a in accs:
            a.clear()
        try:
            with torch.no_grad():
                got = dp_sequence(self.runner, mine, self.dummy, window=self.window, chunk=self.chunk, scale=0.0,
                                  mtp_weight=0.0, backward=False)
            kr, nr = float(got["kd_reply"]), float(got["n_reply"])
        finally:
            reshard_all(self.runner)
            for a, b in zip(accs, saved):
                a.clear()
                a.update(b)
        every = ctl_gather([kr, nr] if mine is not None else None)
        per = [round(x[0] / x[1], 6) if x is not None and x[1] else None for x in every]
        vals = [v for v in per if v is not None]
        return {"fresh_kd": round(sum(vals) / len(vals), 6) if vals else None, "fresh_kd_rank": per, "fresh_age": ages}


def read_files(folder, limit=None):
    """The files a read takes: the first ``limit`` by name; HEAL_READ_SELECT=session (the first 20 dh10
    files by name covered 13 of its 27 sessions) one file per session (name before "--"), the one with the most reply
    tokens (index.json, else the largest), ``limit`` sessions in session order."""
    files = sorted(Path(folder).glob("*.pt"))
    if os.environ.get("HEAL_READ_SELECT") != "session":
        return files[:limit]
    index = Path(folder) / "index.json"
    reply = {}
    if index.is_file():
        reply = {r["file"]: r.get("reply_tokens", 0) for r in json.loads(index.read_text()).get("rows", []) if "file" in r}
    sessions = {}
    for f in files:
        sessions.setdefault(f.name.split("--")[0], []).append(f)
    pick = [max(group, key=lambda f: (reply.get(f.name, -1), f.stat().st_size)) for _, group in sorted(sessions.items())]
    return pick[:limit]


def frozen_behavior(files, runner, model, device, *, window, chunk, prefix, freeze=False):
    """HEAL_HELDOUT2_BEH=<file>: the held-out sampled metric's behaviour log b frozen across rounds (the
    first round computes and saves it; later rounds load it), so round N's end read equals round N+1's start read and
    the start read can be skipped; without it, b comes from each round's starting weights."""
    path = os.environ.get("HEAL_HELDOUT2_BEH") if freeze else None
    if path and Path(path).is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        BEH_CACHE.update({k: v for k, v in saved.items() if k not in BEH_CACHE})
    missing = [f for f in files if str(f) not in BEH_CACHE and f.name.startswith(prefix)]
    if missing:
        BEH_CACHE.update(behavior_pass(runner, model, missing, device, window=window, chunk=chunk,
                                       sampler=sampled_settings()[1]))
        if path:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                keep = torch.load(path, map_location="cpu", weights_only=True) if Path(path).is_file() else {}
                keep.update({str(f): BEH_CACHE[str(f)] for f in files if str(f) in BEH_CACHE})
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(keep, path)


def _heldout_kl(runner, model, folder, device, *, window, chunk, limit=None, samp_prefix=None):
    """Held-out KD (all scored and reply positions) of the current weights on the first ``limit`` teacher files of
    ``folder`` (the score's re86 v4 set), no gradients; images placed as in the score."""
    import flash_qlora_job as fq

    files = read_files(folder, limit)
    kd = n = kr = nr = 0.0
    split = os.environ.get("HEAL_CLASS_SPLIT") == "1"
    # HEAL_SAMPLED_METRIC=1 on HEAL_HELDOUT2 (the student's own held-out rollouts): sampled-token reverse KL, the
    # behavior log b from the first read's weights (the round's start), kept for the later reads
    # samp_prefix (the fixed batch under HEAL_SAMPLED): the same on its student-sampled files, so kd_mix is the
    # trained objective's value
    if os.environ.get("HEAL_SAMPLED_METRIC") == "1" and str(folder) == os.environ.get("HEAL_HELDOUT2"):
        samp_prefix = ""
    samp_metric = samp_prefix is not None
    if samp_metric:
        frozen_behavior(files, runner, model, device, window=window, chunk=chunk, prefix=samp_prefix,
                        freeze=str(folder) == os.environ.get("HEAL_HELDOUT2"))
    train_on = TRAIN_ACC.pop("on", None)  # a read between steps is not training
    SAMP_ACC.clear()
    SAMP_ACC["on"] = samp_metric
    CLASS_ACC.clear()
    REV_ACC.clear()
    REV_ACC["on"] = os.environ.get("HEAL_HELDOUT_REVERSE") == "1"
    READ_ACC.clear()
    READ_ACC["on"] = True
    per_file = {}  # per-file sums (the 10-vs-20-file read noise from split halves)
    if shard_on():  # HEAL_SHARD_READS: every rank reads its share, then the sums are reduced
        def build(path):
            data = load_teacher(path, device)
            return {"data": data, "mask": data["mask"], "piece": fq.positions_and_images(model, data["inputs"]),
                    "classes": reply_classes(data["ids"][0], data["reply"]) if split else None,
                    "samp": slot_samp(data, path, samp_prefix) if samp_metric else None}

        with torch.no_grad():
            kd, n, kr, nr = shard_read(runner, model, files, device, window=window, chunk=chunk, build=build,
                                       record=per_file)
        import torch.distributed as dist
        every = [None] * dist.get_world_size()
        dist.all_gather_object(every, per_file, group=ctl_group())
        per_file = {k: v for d in every for k, v in d.items()}
        sums = {"kd": kd, "n": n, "kr": kr, "nr": nr}
        reduce_acc(sums, ("kd", "n", "kr", "nr"), device)
        kd, n, kr, nr = sums["kd"], sums["n"], sums["kr"], sums["nr"]
        if REV_ACC.get("on"):
            reduce_acc(REV_ACC, ("kl", "n"), device)
        reduce_acc(READ_ACC, ("mix", "n", "tail", "nr", "tail32", "mix_reply"), device)
        if samp_metric:
            reduce_acc(SAMP_ACC, ("est", "ratio", "w", "n"), device)
        if split:
            import torch.distributed as dist
            zero = torch.zeros(len(REPLY_CLASSES), device=device, dtype=torch.float64)
            both = torch.stack([CLASS_ACC.get("kl", zero).to(zero), CLASS_ACC.get("n", zero).to(zero)])
            dist.all_reduce(both)
            CLASS_ACC["kl"], CLASS_ACC["n"] = both[0], both[1]
    else:
        with torch.no_grad():
            for path in files:
                data = load_teacher(path, device)
                piece = fq.positions_and_images(model, data["inputs"])
                cls = reply_classes(data["ids"][0], data["reply"]) if split else None
                before = acc_snap()
                got = train_sequence(runner, data["ids"], data["teacher"], window=window, mask=data["mask"],
                                     chunk=chunk, backward=False, piece=piece, reply=data["reply"], classes=cls,
                                     samp=slot_samp(data, path, samp_prefix) if samp_metric else None)
                kd, n, kr, nr = kd + got[0], n + got[2], kr + got[3], nr + got[4]
                per_file[Path(path).name] = file_row(float(got[3]), float(got[4]), before, acc_snap())
                del data, piece
    out = {"files": len(files), "positions": int(n), "kd": round(kd / max(n, 1), 5), "reply_positions": int(nr),
           "kd_reply": round(kr / max(nr, 1), 5),
           "per_file": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in sorted(per_file.items())}}
    if REV_ACC.get("n") is not None:
        out["kd_reply_reverse"] = round(float(REV_ACC["kl"]) / max(float(REV_ACC["n"]), 1), 5)
    REV_ACC.clear()
    if READ_ACC.get("n") is not None:  # the trained objective (LOSS_MIX) and the out-of-top-k student mass
        out["kd_mix"] = round(float(READ_ACC["mix"]) / max(float(READ_ACC["n"]), 1), 5)
        out["mix"] = {k: LOSS_MIX.get(k, 0.0) for k in ("forward", "reverse", "sampled")}
    if READ_ACC.get("nr") is not None:
        out["student_out_topk_reply"] = round(float(READ_ACC["tail"]) / max(float(READ_ACC["nr"]), 1), 5)
        out["kd_mix_reply"] = round(float(READ_ACC["mix_reply"]) / max(float(READ_ACC["nr"]), 1), 6)
        out["student_out_top32_reply"] = round(float(READ_ACC["tail32"]) / max(float(READ_ACC["nr"]), 1), 5)
    READ_ACC.clear()
    if SAMP_ACC.get("n") is not None:  # IS-weighted estimate, the plain log-ratio, the mean weight, positions
        m = max(float(SAMP_ACC["n"]), 1)
        out.update(sampled_reverse=round(float(SAMP_ACC["est"]) / m, 5), sampled_logratio=round(float(SAMP_ACC["ratio"]) / m, 5),
                   sampled_w=round(float(SAMP_ACC["w"]) / m, 4), sampled_positions=int(SAMP_ACC["n"]))
    SAMP_ACC.clear()
    if split and CLASS_ACC:
        kl, cn = CLASS_ACC["kl"].tolist(), CLASS_ACC["n"].tolist()
        out["by_class"] = {c: {"kd": round(k / max(m, 1), 5), "positions": int(m)} for c, k, m in zip(REPLY_CLASSES, kl, cn)}
    if train_on:
        TRAIN_ACC["on"] = train_on
    return out


def start_reads_cache(path, args, held2, held2_files, mid_files):
    """HEAL_START_READS=<json> (launch-to-first-step): a fresh start's reads of its starting weights
    (general probe, held-out, held-out2, mid read 0) are a function of the student folder and the read settings, so a
    relaunch from the same student reuses them. Returns (cached reads or None, the key they must match). With the
    held-out sampled metric on, b must come frozen from HEAL_HELDOUT2_BEH (else the reads run)."""
    key = {"student": str(Path(args.student).resolve()), "resume": str(getattr(args, "resume", "") or ""),
           "heldout": args.heldout or "",
           "heldout_files": len(read_files(args.heldout, args.heldout_files)) if args.heldout else 0,
           "heldout2": held2 or "", "heldout2_files": held2_files, "general": args.general or "",
           "window": args.window, "chunk": args.chunk, "mid_files": mid_files,
           "class_split": os.environ.get("HEAL_CLASS_SPLIT") == "1",
           "sampled_metric": os.environ.get("HEAL_SAMPLED_METRIC") == "1",
           "mid_general": os.environ.get("HEAL_MID_GENERAL") == "1"}
    if not path or not Path(path).is_file():
        return None, key
    try:
        got = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None, key
    if got.get("key") != key:
        return None, key
    beh = os.environ.get("HEAL_HELDOUT2_BEH")
    if key["sampled_metric"] and held2 and not (beh and Path(beh).is_file()):
        return None, key
    return got.get("reads") or None, key


def preread_from(args, held2):
    """HEAL_PREREAD_FROM=<previous round's events.jsonl> (~16 min a round): the previous round's end reads
    ("heldout", "heldout2") stand in for this round's start reads when they read the same weights and files: this
    round starts from that round's saved student, the read settings match, and the held-out sampled metric's b is
    frozen (HEAL_HELDOUT2_BEH). Anything else: {} (the reads run)."""
    src = os.environ.get("HEAL_PREREAD_FROM")
    if not src or not Path(src).is_file():
        return {}
    if Path(args.student).resolve() != (Path(src).parent / "student").resolve():
        return {}
    if os.environ.get("HEAL_SAMPLED_METRIC") == "1" and held2 and not os.environ.get("HEAL_HELDOUT2_BEH"):
        return {}
    ends = {}
    for line in open(src):
        e = json.loads(line) if line.strip() else {}
        if e.get("event") in ("heldout", "heldout2"):
            ends[e["event"]] = {k: v for k, v in e.items() if k not in ("t", "event")}
    out = {"source": str(src)}
    if "heldout" in ends and ends["heldout"].get("files") == len(read_files(args.heldout, args.heldout_files)):
        out["heldout"] = ends["heldout"]
    h2 = ends.get("heldout2")
    if h2 and held2 and h2.get("folder") == held2 and h2.get("files") == len(read_files(held2, int(
            os.environ.get("HEAL_HELDOUT2_FILES") or 20))):
        out["heldout2"] = {k: v for k, v in h2.items() if k != "folder"}
        out["heldout2"]["folder"] = held2
    return out


REPLY_CLASSES = ("start", "think", "call", "end")


def train_classes(device):
    """This step's training KL by reply class over all ranks (HEAL_TRAIN_CLASS_SPLIT); clears CLASS_ACC."""
    import torch.distributed as dist

    zero = torch.zeros(len(REPLY_CLASSES), device=device, dtype=torch.float64)
    both = torch.stack([CLASS_ACC.get("kl", zero).to(zero), CLASS_ACC.get("n", zero).to(zero)])
    CLASS_ACC.clear()
    if dist.is_initialized():
        dist.all_reduce(both)
    kl, n = both.tolist()
    return {c: {"kd": round(k / max(m, 1), 5), "positions": int(m)} for c, k, m in zip(REPLY_CLASSES, kl, n)}
ETH, ETC = 248069, 248059  # </think>, </tool_call> (heal_audit / classnll)


def reply_classes(ids, reply):
    """[4, T] float masks of each reply span's classes: start (first 8 tokens), think (to </think>), call (after it to
    </tool_call>), end (after that); the same rule as ops/heal_audit.py."""
    ids_l, rep = ids.tolist(), (reply > 0).tolist()
    out = torch.zeros(4, len(ids_l), device=reply.device)
    t, T = 0, len(ids_l)
    while t < T:
        if not rep[t]:
            t += 1
            continue
        a, state = t, 1
        while t < T and rep[t]:
            c = 0 if t - a < 8 else state
            if state == 1 and ids_l[t] == ETH:
                state = 2
            elif state == 2 and ids_l[t] == ETC:
                c, state = 2, 3
            out[c, t] = 1.0
            t += 1
    return out


def run_score(args):
    """The damage map: per student folder (``--students`` under ``--student-root``), the unhealed KD to the teacher
    files (first ``--compare-files``, images included; all scored and reply positions) and to the general-text
    windows, BF16, no gradients, one student at a time on one GPU."""
    import gc

    import flash_qlora_job as fq

    cuda = torch.cuda.is_available()  # CPU: the score preflight (preflights never on GPUs)
    if cuda:
        torch.cuda.set_device(0)
    device = "cuda:0" if cuda else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.teacher).glob("*.pt"))[: args.compare_files]
    if not files:
        raise SystemExit(f"no teacher files in {args.teacher}")
    for name in args.students:  # "folder" or "folder:k" (top-k overridden in the config: same weights)
        began = time.time()
        folder, _, top_k = name.partition(":")
        model, _ = load_student(Path(args.student_root) / folder, device, train=False, top_k=int(top_k) if top_k else None)
        row = {"event": "student_score", "student": name, "load_seconds": round(time.time() - began, 1),
               "params": sum(p.numel() for p in model.parameters()),
               "gpu_gib": (round(torch.cuda.memory_allocated() / 2**30, 2) if cuda else 0.0)}
        runner = Runner(model)
        since = time.time()
        kd = n = kr = nr = 0.0
        with torch.no_grad():
            for path in files:
                data = load_teacher(path, device)
                piece = fq.positions_and_images(model, data["inputs"])
                got = train_sequence(runner, data["ids"], data["teacher"], window=args.window, mask=data["mask"],
                                     chunk=args.chunk, backward=False, piece=piece, reply=data["reply"])
                kd, n, kr, nr = kd + got[0], n + got[2], kr + got[3], nr + got[4]
                del data, piece
            general = probe_kl(runner, args.general, device, window=args.window, chunk=args.chunk) if args.general else None
        row.update(kd=round(kd / max(n, 1), 5), kd_reply=round(kr / max(nr, 1), 5), positions=int(n),
                   reply_positions=int(nr), general=general, score_seconds=round(time.time() - since, 1),
                   peak_gib=(round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0.0))
        event(out, **row)
        del runner, model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def run_speed(args):
    """Speed and memory of full-parameter KD on teacher files (``torchrun --nproc-per-node N``; the first read is
    N = 1 on one B300): per sequence the windowed forward and backward, tokens per second, peak memory, the KD of
    all scored positions and of reply positions. ``--optimizer none`` keeps the gradients (zeroed, still allocated)
    and skips the step: one card cannot also hold fp32 Adam states for the whole student."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    import flash_qlora_job as fq

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out if rank == 0 else None
    began = time.time()
    model, frozen = load_student(args.student, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    event(log, event="student_loaded", seconds=round(time.time() - began, 1), trainable=trainable,
          frozen=sum(p.numel() for p in frozen), gpu_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
          host_gib=fq.host_gib())
    head = None
    if args.mtp_weight:
        text = model.config.text_config
        head = load_mtp(MtpHead(text, experts=getattr(text, "mtp_num_experts", None), top_k=args.mtp_top_k), args.student,
                        device, dtype=torch.float32)
        trainable += sum(p.numel() for p in head.parameters())
    runner = shard(Runner(model, head), init_device_mesh("cuda", (world,)), ignored=frozen)
    runner.train()  # checkpointing on from the first window (run_train, 21 Oct)
    optimizer = None
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW([p for p in runner.parameters() if p.requires_grad], lr=args.lr,
                                      betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    event(log, event="sharded", gpu_gib=round(torch.cuda.memory_allocated() / 2**30, 2))
    if args.general:  # every rank runs it (FSDP gathers); rank 0 logs
        since = time.time()
        event(log, event="general_probe", when="before",
              **probe_kl(runner, args.general, device, window=args.window, chunk=args.chunk))
        event(log, event="general_probe_seconds", seconds=round(time.time() - since, 1))
    for k in args.compare_k:  # the unhealed student's KL to the same teacher files read as top-k targets
        event(log, event="kd_bias", k=k, **probe_kl(runner, args.teacher, device, window=args.window, chunk=args.chunk,
                                                    limit=args.compare_files, k=k))
    files = sorted(Path(args.teacher).glob("*.pt"))[rank::world]
    if not files:
        raise SystemExit(f"no teacher files in {args.teacher}")
    deadline = time.time() + args.seconds  # training time only (the load and the probes come before)
    rows, index = [], 0
    while time.time() < deadline and files:
        path = files[index % len(files)]
        index += 1
        data = load_teacher(path, device)
        piece = fq.positions_and_images(model, data["inputs"])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        since = time.time()
        n = float(data["mask"].sum())
        kd, lm, n, kd_reply, n_reply, kd_mtp, n_mtp = train_sequence(
            runner, data["ids"], data["teacher"], window=args.window, mask=data["mask"], labels=data["labels"],
            lm_weight=args.lm_weight, chunk=args.chunk, scale=1.0 / max(n, 1.0), piece=piece, reply=data["reply"],
            mtp_weight=args.mtp_weight)
        norm = grad_norm(runner) if index <= 2 or optimizer is not None else None
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_([p for p in runner.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)
        else:
            runner.zero_grad(set_to_none=False)
        torch.cuda.synchronize()
        seconds = time.time() - since
        tokens = data["ids"].shape[1]
        row = {"step": index, "file": data["name"], "tokens": tokens, "seconds": round(seconds, 2),
               "tok_per_s": round(tokens / seconds, 1), "kd": round(kd / max(n, 1), 5),
               "kd_reply": round(kd_reply / n_reply, 5) if n_reply else None, "nll": round(lm / max(n, 1), 5),
               "kd_mtp": round(kd_mtp / n_mtp, 5) if n_mtp else None,
               "scored": int(n), "reply": int(n_reply), "grad_norm": norm,
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        rows.append(row)
        event(log, **row)
        del data, piece
    steady = rows[1:] or rows
    tokens, seconds = sum(r["tokens"] for r in steady), sum(r["seconds"] for r in steady)
    summary = {"event": "speed_summary", "world": world, "window": args.window, "optimizer": args.optimizer,
               "sequences": len(rows), "steady_tok_per_s": round(tokens / seconds, 1) if seconds else None,
               "peak_gib": max((r["peak_gib"] for r in rows), default=None),
               "kd_mean": round(sum(r["kd"] * r["scored"] for r in rows) / max(sum(r["scored"] for r in rows), 1), 5),
               "kd_reply_mean": round(sum((r["kd_reply"] or 0) * r["reply"] for r in rows)
                                      / max(sum(r["reply"] for r in rows), 1), 5),
               "trainable": trainable, "seconds": round(time.time() - began, 1)}
    event(log, **summary)
    dist.destroy_process_group()


def run_layers(args):
    """Depth-cut layer choice on one student (prune-heal): each layer skipped at run time (identity),
    KL(reference || cut) on the scored positions of excl5 recorded-play windows (``--windows``: speedmap tf.json;
    reference = the uncut student's top-20 + tail). One-shot single-layer scores, then greedy: drop the best layer,
    re-score the remaining ``--lazy`` best candidates on the cut model, repeat to ``--depth``; objective KL increase
    per unit of layer cost (full attention ``--cost-full`` vs GDN 1). Also the KL of given sets (``--sets``)."""
    import gc

    torch.cuda.set_device(0)
    device = "cuda:0"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_student(Path(args.student_root) / args.student, device, train=False)
    tm = text_model(model)
    layers = tm.language_model.layers if hasattr(tm, "language_model") else tm.layers
    types = list(model.config.text_config.layer_types)
    skip = set()

    def make(i, layer):
        orig = layer.forward

        def fwd(hidden_states, *a, **k):
            return hidden_states if i in skip else orig(hidden_states, *a, **k)
        layer.forward = fwd
    for i, layer in enumerate(layers):
        make(i, layer)
    head = model.get_output_embeddings().weight
    wins = [w for w in json.loads(Path(args.windows).read_text())["windows"] if w["index"] % 4 == 1]
    data = []
    for w in wins:
        ids = torch.tensor(w["ids"][-args.tokens:], device=device)[None]
        start = max(0, ids.shape[1] - 2048)
        asst = torch.zeros(ids.shape[1], dtype=torch.bool)
        off = len(w["ids"]) - ids.shape[1]
        for a, b in w["assistant_spans"]:
            asst[max(0, a - off):max(0, b - off)] = True
        data.append({"ids": ids, "start": start, "asst": asst[start + 1:].to(device)})  # row t scores token t + 1

    def logp(d):
        h = tm(input_ids=d["ids"], use_cache=False).last_hidden_state[0, d["start"]:-1]
        return torch.log_softmax(F.linear(h, head).float(), dim=-1)

    with torch.no_grad():
        for d in data:
            lp = logp(d)
            top, idx = lp.topk(20, dim=-1)
            d["top"], d["idx"] = top, idx
            d["tail"] = torch.log1p(-top.exp().sum(-1).clamp(max=1 - 1e-6))
            del lp

    def score():
        tot = n = ta = na = 0.0
        with torch.no_grad():
            for d in data:
                lp = logp(d)
                s_top = lp.gather(-1, d["idx"])
                s_tail = torch.log1p(-s_top.exp().sum(-1).clamp(max=1 - 1e-6))
                kl = (d["top"].exp() * (d["top"] - s_top)).sum(-1) + d["tail"].exp() * (d["tail"] - s_tail)
                tot, n = tot + float(kl.sum()), n + kl.numel()
                ta, na = ta + float(kl[d["asst"]].sum()), na + int(d["asst"].sum())
                del lp
        return {"all": round(tot / max(n, 1), 5), "asst": round(ta / max(na, 1), 5)}

    cost = {i: (args.cost_full if types[i] != "linear_attention" else 1.0) for i in range(len(layers))}
    began, report = time.time(), {"student": args.student, "windows": len(data), "positions": sum(int(d["asst"].numel()) for d in data),
                                  "types": types, "cost_full": args.cost_full}
    single = {}
    for i in range(len(layers)):
        skip.clear(); skip.add(i)
        single[i] = score()
    skip.clear()
    report["single"] = single
    report["single_seconds"] = round(time.time() - began, 1)
    event(out, event="layers_single", **{"order": sorted(single, key=lambda i: single[i]["asst"]), "single": single})
    sets = {}
    for spec in args.sets:
        name, members = spec.split("=")
        skip.clear(); skip.update(int(x) for x in members.split(","))
        sets[name] = {"layers": sorted(skip), "kl": score()}
        event(out, event="layers_set", name=name, **sets[name])
    for objective in ("kl", "kl_per_cost"):
        skip.clear()
        chosen, base_kl, last = [], 0.0, {i: single[i]["asst"] for i in single}
        trail = []
        while len(chosen) < args.depth:
            rank = sorted((i for i in last if i not in skip),
                          key=lambda i: (last[i] - base_kl) / (cost[i] if objective == "kl_per_cost" else 1.0))
            cands = rank[: args.lazy]
            fresh = {}
            for i in cands:
                skip.add(i)
                fresh[i] = score()
                skip.discard(i)
                last[i] = fresh[i]["asst"]
            best = min(cands, key=lambda i: (fresh[i]["asst"] - base_kl) / (cost[i] if objective == "kl_per_cost" else 1.0))
            skip.add(best)
            chosen.append(best)
            base_kl = fresh[best]["asst"]
            trail.append({"n": len(chosen), "layer": best, "type": types[best], "kl": fresh[best]})
            event(out, event="layers_greedy", objective=objective, **trail[-1], seconds=round(time.time() - began, 1))
        report[f"greedy_{objective}"] = trail
    skip.clear()
    report["sets"] = sets
    report["seconds"] = round(time.time() - began, 1)
    (out / "layers.json").write_text(json.dumps(report, indent=1))
    del model
    gc.collect()



# ---- healing run (PTH OPD, 18 Oct) ----------------------------------------------------------------------------------

EXPERT_PARAM = re.compile(r"language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


def warmup_steps(total, frac):
    """At least 2 warm-up steps: with 1, step 0 trained at the peak, i.e. no warm-up at all (cut1-v2, c1opd-v1)."""
    return max(2, math.ceil(total * frac))


def wsd_lr(step, total, peak, warmup=0.1, decay=0.2):
    """Warm-up, stable, linear decay to 0 over the whole multi-round run (``step`` is global)."""
    w, d = warmup_steps(total, warmup), max(1, int(total * decay)) if decay > 0 else 0
    if step < w:
        return peak * (step + 1) / (w + 1)
    if d and step >= total - d:
        return peak * max(0.0, (total - step) / d)
    return peak


def cosine_lr(step, total, peak, warmup=0.1, floor=0.1):
    """Linear warm-up over ``warmup`` of the steps (>= 2), then cosine decay from ``peak`` to ``floor`` x peak."""
    w = warmup_steps(total, warmup)
    if step < w:
        return peak * (step + 1) / (w + 1)
    frac = min(1.0, (step - w) / max(1, total - w))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac)))


def teacher_files(spec):
    """``.pt`` teacher files of one folder or a comma-separated list of folders, sorted by (folder order, name)."""
    return [f for d in str(spec).split(",") if d for f in sorted(Path(d).glob("*.pt"))]


def route_cache_path(args, files):
    """HEAL_ROUTE_CACHE dir -> <dir>/<sha>.pt keyed on the student's index/config bytes, every rank's routed files
    (name, size), window, chunk and file limit; None when unset."""
    import hashlib

    import torch.distributed as dist

    root = os.environ.get("HEAL_ROUTE_CACHE")
    if not root or not args.student:
        return None
    mine = [(Path(f).name, Path(f).stat().st_size) for f in files[:args.route_files]]
    every = [mine]
    if dist.is_initialized():
        every = [None] * dist.get_world_size()
        dist.all_gather_object(every, mine)
    h = hashlib.sha256(json.dumps([str(Path(args.student).resolve()), every, args.window, args.chunk,
                                   args.route_files]).encode())
    for name in ("model.safetensors.index.json", "config.json"):
        f = Path(args.student) / name
        if f.is_file():  # mtime too: a checkpoint saved again in place keeps the same index bytes
            h.update(f.read_bytes() + str(f.stat().st_mtime_ns).encode())
    return Path(root) / f"{h.hexdigest()[:24]}.pt"


def route_counts(runner, model, files, device, *, window, chunk, limit):
    """Routed-token counts per (layer, expert) of the student on ``files`` (no gradients), summed over ranks."""
    import torch.distributed as dist

    import flash_qlora_job as fq

    text = model.config.text_config
    layers = text_model(model).language_model.layers
    counts = torch.zeros(len(layers), text.num_experts, device=device)

    def hook_for(i):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                ids = next((o for o in output if torch.is_tensor(o) and not o.is_floating_point()), None)
                if ids is None:
                    ids = output[0].float().topk(text.num_experts_per_tok, -1).indices
            else:
                ids = output.float().topk(text.num_experts_per_tok, -1).indices
            counts[i] += torch.bincount(ids.flatten().long(), minlength=text.num_experts).float()
        return hook

    hooks = [layer.mlp.gate.register_forward_hook(hook_for(i)) for i, layer in enumerate(layers)]
    with torch.no_grad():
        for path in files[:limit]:
            data = load_teacher(path, device)
            piece = fq.positions_and_images(model, data["inputs"])
            train_sequence(runner, data["ids"], data["teacher"], window=window, mask=data["mask"], chunk=chunk,
                           backward=False, piece=piece)
    for h in hooks:
        h.remove()
    if dist.is_initialized():
        dist.all_reduce(counts)
    return counts


def wrap_adam_state(adam, state):
    """A resharded optimizer file (heal_reshard.py) holds plain local moments: wrap each on its parameter's mesh as a
    DTensor (in place). Every moment's local shape must equal its parameter's shard (a world change without a reshard
    stops here, not as silent misaligned moments)."""
    params = [p for g in adam.param_groups for p in g["params"]]
    for i, st in state["state"].items():
        p = params[int(i)]
        for k, v in list(st.items()):
            if not isinstance(v, torch.Tensor) or v.dim() == 0:
                continue
            local = v.to_local() if hasattr(v, "to_local") else v
            want = tuple((p.to_local() if hasattr(p, "to_local") else p).shape)
            if tuple(local.shape) != want:
                raise RuntimeError(f"optimizer state {i}.{k}: shard {tuple(local.shape)} vs parameter shard {want} "
                                   "(world size changed: reshard with heal_reshard.py)")
            if hasattr(p, "device_mesh") and not hasattr(v, "to_local"):
                from torch.distributed.tensor import DTensor

                st[k] = DTensor.from_local(local.to(p.to_local().device), p.device_mesh, p.placements, run_check=False, shape=p.shape,
                                           stride=p.stride())


def fake_int4(ws, group=128):
    """The served int4 RTN values (heal_delta.int4_rtn_rows, the export rule) of ``ws`` [E, out, in], dequantized, in
    ``ws``'s dtype."""
    import heal_delta

    qw, _, sc = heal_delta.int4_rtn_rows(ws.float(), group=min(group, ws.shape[-1]))
    e, inp8, out = qw.shape
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qw.device)
    codes = ((qw.unsqueeze(2) >> shifts.view(1, 1, -1, 1)) & 15).reshape(e, inp8 * 8, out).float() - 8
    g = ws.shape[-1] // sc.shape[1]
    return (codes * sc.float().repeat_interleave(g, 1)).transpose(1, 2).to(ws.dtype)


class Int4Rows:
    """HEAL_MID_INT4 (audit 24 Oct: is the served student the trained one?): inside the block the trained expert rows hold
    their served int4 values (gate/up halves and down quantized as heal_delta exports them); the originals (host copy)
    come back on exit, bit for bit. ``names``: id(param) -> name (down_proj is one matrix, gate_up two halves)."""

    def __init__(self, rows, names):
        self.rows, self.names, self.saved = rows, names, []

    @torch.no_grad()
    def __enter__(self):
        for it in self.rows.items:
            if not len(it["rows"]):
                self.saved.append(None)
                continue
            local = it["p"].to_local() if hasattr(it["p"], "to_local") else it["p"].data
            sel = local[it["rows"]]
            self.saved.append(sel.detach().to("cpu", copy=True))
            if self.names.get(id(it["p"]), "").endswith("down_proj"):
                q = fake_int4(sel)
            else:
                half = sel.shape[1] // 2
                q = torch.cat([fake_int4(sel[:, :half]), fake_int4(sel[:, half:])], 1)
            local.index_copy_(0, it["rows"], q.to(local.dtype))
        return self

    @torch.no_grad()
    def __exit__(self, *exc):
        for it, s in zip(self.rows.items, self.saved):
            if s is not None:
                local = it["p"].to_local() if hasattr(it["p"], "to_local") else it["p"].data
                local.index_copy_(0, it["rows"], s.to(local.device, local.dtype))
        self.saved = []
        return False


def prune_checkpoints(out, folder, keep=None):
    """After ``folder`` completes: delete older ``ckpt-<k>`` folders, keeping the newest ``keep`` complete ones
    (HEAL_KEEP_CKPT, default 1; c7h 24 Oct: a rollback point before a bad stretch was gone). Incomplete older folders
    (no progress.json) always go; nothing newer than ``folder`` is touched."""
    import shutil

    keep = int(keep if keep is not None else (os.environ.get("HEAL_KEEP_CKPT") or 1))
    mine = int(folder.name.split("-")[1])
    done = []
    for old in out.glob("ckpt-*"):
        try:
            k = int(old.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if old == folder or k > mine:
            continue
        if (old / "progress.json").is_file():
            done.append((k, old))
        else:
            shutil.rmtree(old, ignore_errors=True)
    for _, old in sorted(done, reverse=True)[max(keep - 1, 0):]:
        shutil.rmtree(old, ignore_errors=True)


class RowAdam:
    """AdamW (no weight decay) on the selected rows (experts, dim 0) of fused expert parameters; the other rows stay
    fixed and keep no state (ESFT: only task-routed experts train). FSDP2 shards dim 0: each rank updates the selected
    rows of its own shard."""

    def __init__(self, items, betas=(0.9, 0.95), eps=1e-8):
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

        self.betas, self.eps, self.t, self.items = betas, eps, 0, []
        for p, selected in items:
            local = p.to_local() if hasattr(p, "to_local") else p.data
            if hasattr(p, "to_local"):
                shape, offset = compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)
                start = offset[0]
            else:
                start = 0
            rows = torch.tensor([e - start for e in selected if start <= e < start + local.shape[0]], dtype=torch.long,
                                device=local.device)
            keep = torch.ones(local.shape[0], dtype=torch.bool, device=local.device)
            keep[rows] = False  # True: frozen rows (their gradients are zeroed before the clip)
            m = torch.zeros((len(rows), *local.shape[1:]), dtype=torch.float32, device=local.device)
            self.items.append({"p": p, "rows": rows, "frozen": keep, "m": m, "v": torch.zeros_like(m), "start": start})

    def mask_grads(self):
        for it in self.items:
            g = it["p"].grad
            if g is not None:
                (g.to_local() if hasattr(g, "to_local") else g)[it["frozen"]] = 0

    @torch.no_grad()
    def step(self, lr):
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for it in self.items:
            if not len(it["rows"]) or (it["p"].grad is None and not self.tapped):
                continue
            local = it["p"].to_local() if hasattr(it["p"], "to_local") else it["p"].data
            if self.tapped:
                g = it["acc"][it["pos"]]
            else:
                g = it["p"].grad
                g = (g.to_local() if hasattr(g, "to_local") else g)[it["rows"]].float()
            it["m"].mul_(b1).add_(g, alpha=1 - b1)
            it["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
            update = (it["m"] / c1) / ((it["v"] / c2).sqrt() + self.eps)
            local.index_copy_(0, it["rows"], (local[it["rows"]].float() - lr * update).to(local.dtype))

    def zero_grad(self):
        if self.tapped:
            self.flat.zero_()
        for it in self.items:
            if it["p"].grad is not None:
                it["p"].grad.zero_()

    def state_dict(self):
        return {"t": self.t, "m": [it["m"].cpu() for it in self.items], "v": [it["v"].cpu() for it in self.items],
                "rows": [it["rows"].cpu() for it in self.items]}

    def load_state_dict(self, state):
        """Restores the moments; a resume on new data keeps the saved expert selection (the moments belong to it).
        Returns the number of parameters whose selection was replaced by the saved one."""
        self.t, adopted = state["t"], 0
        if len(state["rows"]) != len(self.items):
            raise RuntimeError(f"RowAdam state: {len(state['rows'])} expert tensors vs {len(self.items)}")
        for it, m, v, rows in zip(self.items, state["m"], state["v"], state["rows"]):
            rows = rows.to(it["rows"].device)
            if len(rows) and int(rows.max()) >= it["frozen"].shape[0]:  # rows of another world's slice
                raise RuntimeError(f"RowAdam state: row {int(rows.max())} outside this rank's "
                                   f"{it['frozen'].shape[0]} experts (world size changed: reshard)")
            if not torch.equal(rows, it["rows"]):
                adopted += 1
                it["rows"] = rows
                it["frozen"] = torch.ones_like(it["frozen"])
                it["frozen"][rows] = False
                it["m"] = it["v"] = None  # the old moments go before the new ones are allocated
                it["m"], it["v"] = (torch.empty(m.shape, dtype=torch.float32, device=rows.device),
                                    torch.empty(v.shape, dtype=torch.float32, device=rows.device))
            it["m"].copy_(m)
            it["v"].copy_(v)
        return adopted

    # HEAL_EXPERT_ROWS=1 (trainer MFU S3): FSDP would reduce-scatter each expert tensor's whole gradient after every
    # backward, frozen rows included (88% of it at ESFT 12%), and keep a sharded fp32 copy of it. Instead a post-
    # accumulate hook on FSDP's unsharded parameter adds the selected rows (every rank's selection) of each backward's
    # gradient to one flat fp32 buffer and drops the gradient, so FSDP has nothing to reduce for these tensors; the
    # buffer is all-reduced (averaged, as FSDP does) once per step and feeds the clip and the row update.
    tapped = False

    def tap(self, runner):
        """After construction and any resume (the selection is final). Hooks arm on the first forward of each
        expert tensor's FSDP unit (the decoder layer: FSDP creates its unsharded parameters on the first all-gather,
        in the unit's pre-forward hook, which runs before ours). The unit's own call is never compiled: HEAL_COMPILE
        compiles regions inside it with fullgraph, where a hook on the expert module is fatal (s2-read, s2-read2)."""
        import torch.distributed as dist
        from torch.distributed.fsdp import FSDPModule

        world = dist.get_world_size() if dist.is_initialized() else 1
        mine = [(it["rows"] + it["start"]).tolist() for it in self.items]
        every = [None] * world
        if world > 1:
            dist.all_gather_object(every, mine)
        else:
            every = [mine]
        owner, unit = {}, {}
        for m in runner.modules():
            if isinstance(m, FSDPModule):
                for group in m._get_fsdp_state()._fsdp_param_groups:
                    for fp in group.fsdp_params:
                        owner[id(fp.sharded_param)] = fp
                        unit[id(fp.sharded_param)] = m
        module_of = {id(p): unit[id(p)] for p in (it["p"] for it in self.items)}
        sizes = []
        for i, it in enumerate(self.items):
            sel = sorted({e for r in range(world) for e in every[r][i]})
            it["sel"] = torch.tensor(sel, dtype=torch.long, device=it["rows"].device)
            index = {e: j for j, e in enumerate(sel)}
            it["pos"] = torch.tensor([index[r + it["start"]] for r in it["rows"].tolist()], dtype=torch.long,
                                     device=it["rows"].device)
            it["fp"] = owner[id(it["p"])]
            sizes.append(len(sel) * math.prod(it["p"].shape[1:]))
        device = self.items[0]["rows"].device if self.items else "cpu"
        self.flat = torch.zeros(sum(sizes), dtype=torch.float32, device=device)
        offset = 0
        by_module = {}
        for it, n in zip(self.items, sizes):
            it["acc"] = self.flat[offset:offset + n].view(len(it["sel"]), *it["p"].shape[1:])
            offset += n
            by_module.setdefault(id(module_of[id(it["p"])]), (module_of[id(it["p"])], []))[1].append(it)

        def hook(it, param):
            if param.grad is not None:
                it["acc"].add_(param.grad.index_select(0, it["sel"]).float())
                param.grad = None

        def arm(items, module, *_):
            for it in items:
                if "hook" not in it:
                    it["hook"] = it["fp"]._unsharded_param.register_post_accumulate_grad_hook(
                        lambda param, it=it: hook(it, param))
            if all("hook" in x for x in self.items):  # every tensor armed: drop the pre-hooks (no graph breaks after)
                for h in self.arms:
                    h.remove()
                self.arms = []

        self.arms = [module.register_forward_pre_hook(lambda m, a, items=items: arm(items, m))
                     for module, items in by_module.values()]
        self.tapped, self.world = True, world
        return {"tensors": len(self.items), "rows": int(sum(len(it["sel"]) for it in self.items)),
                "buffer_gib": round(self.flat.numel() * 4 / 2**30, 3)}

    def reduce_taps(self):
        import torch.distributed as dist

        for it in self.items:
            if it["p"].grad is not None:
                raise RuntimeError("HEAL_EXPERT_ROWS: an expert tensor's gradient reached FSDP (hook not armed)")
        if self.world > 1:
            dist.all_reduce(self.flat)
            self.flat.div_(self.world)

    def tap_norm2(self):
        return self.flat.pow(2).sum()

    def tap_grads(self):
        """Full expert gradients (selected rows; the others zero), as mask_grads leaves them: the comm check."""
        out = {}
        for it in self.items:
            g = torch.zeros(it["p"].shape, dtype=torch.float32, device=self.flat.device)
            g[it["sel"]] = it["acc"]
            out[id(it["p"])] = g
        return out


def save_student(runner, frozen_root, out, rank, *, step, meta):
    """The trained student as a checkpoint folder: every trainable tensor gathered (FSDP) and written by rank 0 in
    fp32 shards (rounding to BF16 would erase small updates; S16); every other checkpoint tensor (n-gram table,
    vision tower, buffers) mapped to ``frozen_root``'s shard files through links; index + config copied."""
    import shutil

    from safetensors.torch import save_file

    out = Path(out)
    tmp = out.with_name(out.name + ".partial")
    src_index = json.loads((Path(frozen_root) / "model.safetensors.index.json").read_text())["weight_map"]
    if rank == 0:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    names, buffer, size, part = {}, {}, 0, 0

    def flush():
        nonlocal buffer, size, part
        if rank == 0 and buffer:
            part += 1
            file = f"trained-{part:05d}.safetensors"
            save_file(buffer, str(tmp / file))
            names.update({k: file for k in buffer})
        buffer, size = {}, 0

    for key, p in runner.named_parameters():
        if not p.requires_grad:
            continue
        full = p.full_tensor() if hasattr(p, "full_tensor") else p.data  # collective: every rank calls it
        name = key[len("model."):] if key.startswith("model.") else key  # Runner.model.<name>; Runner.mtp.<name>
        if name not in src_index:
            raise KeyError(f"trained tensor not in the checkpoint: {name}")
        if rank == 0:
            buffer[name] = full.detach().float().cpu().contiguous()
            size += full.numel() * 4
        del full
        if size >= 8 << 30:
            flush()
    flush()
    if rank != 0:
        return None
    linked = set()
    for name, file in src_index.items():
        if name not in names:
            names[name] = file
            if file not in linked:
                (tmp / file).symlink_to((Path(frozen_root) / file).resolve())
                linked.add(file)
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": names}, indent=1))
    for path in Path(frozen_root).iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name not in (
                "model.safetensors.index.json", "heal-state.json"):
            shutil.copy(path, tmp / path.name)
    (tmp / "heal-state.json").write_text(json.dumps({"step": step, **meta}, indent=1))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return {"saved": str(out), "trained_shards": part}


CKPT_THREAD = {}  # HEAL_ASYNC_CKPT: the checkpoint writer in flight


CKPT_BUFS = {}  # HEAL_CKPT_PINNED: key -> host buffer reused by every async checkpoint (pinned on CUDA)


def _host_buf(key, like, dtype=None):
    """A reused host buffer for ``like`` (shape, ``dtype`` or its own): allocated on the first checkpoint (pinned
    when CUDA is up), refilled by the next ones; the writer of the last checkpoint is joined before (one in flight)."""
    dtype = dtype or like.dtype
    buf = CKPT_BUFS.get(key)
    if buf is None or buf.shape != like.shape or buf.dtype != dtype:
        buf = torch.empty(like.shape, dtype=dtype, device="cpu", pin_memory=torch.cuda.is_available())
        CKPT_BUFS[key] = buf
    return buf


def host_mem_gib():
    """(available, total) host GiB of this container (/proc/meminfo; Modal's sandbox reports its own limit)."""
    try:
        got = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        return round(got["MemAvailable"] / 2 ** 20, 1), round(got["MemTotal"] / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        return None, None


CMDLINES = {}  # pid -> cmdline (host_breakdown)


def host_breakdown():
    """Where the container's host RAM sits (c7b 24 Oct: ~400 GiB more in use than c7, consumer unknown): RSS GiB by
    process class (from /proc), meminfo Shmem / Cached, the stream folders' sizes, this process's pinned pool."""
    gib = lambda b: round(b / 2 ** 30, 2)  # noqa: E731
    out, rss = {}, {}
    try:
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                if d.name not in CMDLINES:  # read once per pid: cmdline takes the target's mmap lock (c7c: the trace
                    CMDLINES[d.name] = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                cmd = CMDLINES[d.name]  # stopped after 4 rows, before step 1)
                kb = next(int(x.split()[1]) for x in (d / "status").read_text().splitlines() if x.startswith("VmRSS:"))
            except (OSError, StopIteration, ValueError):
                continue
            kind = ("teacher" if "stream_topk" in cmd or "rl1_trainer" in cmd else "trainer" if "heal_kd" in cmd
                    else "encoder" if "multiprocessing" in cmd else "rclone" if "rclone" in cmd
                    else "round" if "modal" in cmd or "python" in cmd else "other")
            rss[kind] = rss.get(kind, 0) + kb * 1024
        out["rss_gib"] = {k: gib(v) for k, v in sorted(rss.items())}
        mi = {x.split(":")[0]: int(x.split()[1]) * 1024 for x in Path("/proc/meminfo").read_text().splitlines()}
        out.update(shmem_gib=gib(mi.get("Shmem", 0)), cached_gib=gib(mi.get("Cached", 0)))
        for name in ("/tmp/stream-chunks", "/tmp/deltas", "/dev/shm"):
            if Path(name).is_dir():
                out[f"du_{Path(name).name}_gib"] = gib(sum(f.stat().st_size for f in Path(name).rglob("*") if f.is_file()))
        if torch.cuda.is_available() and hasattr(torch.cuda, "host_memory_stats"):
            hs = torch.cuda.host_memory_stats()
            out["pinned_gib"] = gib(hs.get("allocated_bytes.current", 0))
    except Exception as exc:  # noqa: BLE001 - diagnostics never break the run
        out["error"] = repr(exc)[:200]
    return out


def _pinned_on():
    return os.environ.get("HEAL_CKPT_PINNED") == "1" and CKPT_PIN.get("ok", True)


CKPT_PIN = {}  # the first pinned checkpoint's decision, agreed over ranks: {"ok": bool}


def _ckpt_need(runner, adam, rows, log, pow2):
    """Collective: every rank's host bytes for one checkpoint copy (pow2: the caching host allocator rounds every
    pinned block up to a power of two), and rank 0's (avail, total) host GiB for all. Returns (need_gib, avail, total)."""
    import torch.distributed as dist

    def pinned(nbytes):
        return (1 << (int(nbytes) - 1).bit_length() if pow2 else int(nbytes)) if nbytes > 0 else 0

    sel = _expert_rows(rows)
    need = 0
    for _, p in runner.named_parameters():
        if p.requires_grad:
            local = p.to_local() if hasattr(p, "to_local") else p.data
            n = local.numel() // max(1, local.shape[0]) * len(sel[id(p)][0]) if id(p) in sel else local.numel()
            need += pinned(n * 4)
    for st in adam.state_dict()["state"].values():
        need += sum(pinned((v.to_local() if hasattr(v, "to_local") else v).numel() * v.element_size())
                    for v in st.values() if isinstance(v, torch.Tensor) and v.dim())
    for it in (getattr(rows, "items", None) if isinstance(getattr(rows, "items", None), list) else []):
        need += pinned(it["m"].numel() * 4) * 2 + pinned(it["rows"].numel() * 8)
    avail, total = host_mem_gib()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.tensor([float(need)], dtype=torch.float64, device=dev)
    b = torch.tensor([float(avail or 0), float(total or 0)], dtype=torch.float64, device=dev)
    if dist.is_initialized():
        dist.all_reduce(a)  # every rank's buffers
        dist.broadcast(b, 0)  # one container: rank 0's reading for all
    return a.item() / 2 ** 30, b[0].item(), b[1].item()


def _pin_decide(runner, adam, log, rows=None):
    """Before the first pinned checkpoint (collective): pin only when the container keeps >= HEAL_CKPT_PIN_HEADROOM
    (default 0.10) of its host RAM free after every rank's buffers (pinned pages are never reclaimed); else the
    plain async copy, for the whole run."""
    need_gib, avail, total = _ckpt_need(runner, adam, rows, log, pow2=True)
    headroom = float(os.environ.get("HEAL_CKPT_PIN_HEADROOM") or 0.10)
    ok = total > 0 and avail - need_gib >= headroom * total
    CKPT_PIN["ok"] = ok
    event(log, event="ckpt_pin", pinned=ok, need_gib=round(need_gib, 1), avail_gib=avail, total_gib=total,
          headroom=headroom)


def _cpu_copy(x, key=()):
    """A host copy of a state tree: DTensors as their local shards (heal_kd wraps them back at load).
    HEAL_CKPT_PINNED=1: into reused (pinned) buffers with non_blocking copies; the caller synchronizes."""
    if isinstance(x, torch.Tensor):
        local = (x.to_local() if hasattr(x, "to_local") else x).detach()
        if _pinned_on() and local.dim():
            return _host_buf(("c",) + key, local).copy_(local, non_blocking=True)
        return local.to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: _cpu_copy(v, key + (k,)) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu_copy(v, key + (i,)) for i, v in enumerate(x))
    return x


def wait_checkpoint():
    """Joins the checkpoint writer in flight (HEAL_ASYNC_CKPT); re-raises its error."""
    t = CKPT_THREAD.pop("thread", None)
    if t is not None:
        t.join()
    err = CKPT_THREAD.pop("error", None)
    if err is not None:
        raise RuntimeError("async checkpoint failed") from err


def checkpoint_async(runner, adam, rows, args, out, rank, *, step, log, progress):
    """HEAL_ASYNC_CKPT=1 (the 390 s save): the state is copied to the host (seconds), then a thread
    writes it while training goes on. Each rank writes its files and a done marker; rank 0 waits for every marker,
    then writes behavior.pt and progress.json (the checkpoint's completion mark) and removes older checkpoints. One
    writer at a time: the next checkpoint and the round's end join it first."""
    import shutil
    import threading

    import torch.distributed as dist

    wait_checkpoint()
    if os.environ.get("HEAL_CKPT_PINNED") == "1" and "ok" not in CKPT_PIN:
        _pin_decide(runner, adam, log if rank == 0 else None, rows=rows)
    if os.environ.get("HEAL_CKPT_GUARD", "1") == "1":  # c7b 24 Oct: the step-1 copy thrashed the host to death
        need_gib, avail, total = _ckpt_need(runner, adam, rows, None, pow2=_pinned_on())
        floor = 1.2 * need_gib + 0.10 * total
        if total > 0 and avail < floor:  # rank 0's reading for all ranks: every rank skips together
            if rank == 0:
                event(log, event="ckpt_skip", step=step, need_gib=round(need_gib, 1),
                      avail_gib=round(avail, 1), floor_gib=round(floor, 1), total_gib=total, alert=True)
            return
    began = time.time()
    world = dist.get_world_size()
    folder = out / f"ckpt-{step}"
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    sd = adam.state_dict()
    # the moments are saved as local shards: their global shapes travel beside them (heal_reshard checks the join)
    shapes = {i: {k: list(v.shape) for k, v in st.items() if isinstance(v, torch.Tensor) and v.dim()}
              for i, st in sd["state"].items()}
    optim = {"adam": _cpu_copy(sd), "rows": _cpu_copy(rows.state_dict()), "adam_global_shapes": shapes}
    params = local_params(runner, pinned=_pinned_on(), rows=rows)
    if _pinned_on() and torch.cuda.is_available():
        torch.cuda.synchronize()  # the non_blocking copies have landed before the writer reads the buffers
    beh = dict(BEH_CACHE) if rank == 0 and BEH_CACHE else None
    # a frozen copy: the loop keeps appending to its consumed list while the thread writes (the tiny resume found
    # every file consumed and nothing left to train)
    progress = json.loads(json.dumps(dict(progress, sharded=True, student=str(args.student), world=world)))
    copied = round(time.time() - began, 1)

    def write():
        try:
            for name, obj in ((f"optim-rank{rank}.pt", optim), (f"params-rank{rank}.pt", params)):
                torch.save(obj, folder / f"{name}.tmp")
                os.replace(folder / f"{name}.tmp", folder / name)
            (folder / f"done-rank{rank}").touch()
            if rank == 0:
                while not all((folder / f"done-rank{r}").exists() for r in range(world)):
                    time.sleep(2)
                if beh:
                    torch.save(beh, folder / "behavior.pt")
                (folder / "progress.json.tmp").write_text(json.dumps(progress))  # the completion mark, atomic
                os.replace(folder / "progress.json.tmp", folder / "progress.json")
                prune_checkpoints(out, folder)
                event(log, event="checkpoint", step=step, path=str(folder), seconds=round(time.time() - began, 1),
                      blocking_seconds=copied, consumed=len(progress["consumed"]), sharded=True, asynchronous=True)
        except Exception as exc:  # noqa: BLE001  re-raised by wait_checkpoint
            CKPT_THREAD["error"] = exc

    CKPT_THREAD["thread"] = threading.Thread(target=write, name=f"ckpt-{step}", daemon=False)
    CKPT_THREAD["thread"].start()


def checkpoint(runner, adam, rows, args, out, rank, *, step, log, progress):
    """A mid-round restart point ``out/ckpt-<step>`` (gate item 4: at most every 30 min): optimizer state per rank,
    the student folder, then ``progress.json`` last (its presence marks the checkpoint complete); older checkpoints
    are removed after it. Restart: --student ckpt/student --resume ckpt --step0 <step> --skip ckpt/progress.json.
    Sharded (default since v25; HEAL_CKPT_FULL=1 for the student folder): each rank writes its own fp32 shards of the
    trainable tensors (``params-rank<r>.pt``) in parallel, no gather; the restart loads the round's starting student
    (progress["student"]) and then these shards (same world size)."""
    import shutil

    import torch.distributed as dist

    if os.environ.get("HEAL_ASYNC_CKPT") == "1" and os.environ.get("HEAL_CKPT_FULL") != "1":
        return checkpoint_async(runner, adam, rows, args, out, rank, step=step, log=log, progress=progress)
    began = time.time()
    folder = out / f"ckpt-{step}"
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    torch.save({"adam": adam.state_dict(), "rows": rows.state_dict()}, folder / f"optim-rank{rank}.pt")
    sharded = os.environ.get("HEAL_CKPT_FULL") != "1"
    if sharded:
        torch.save(local_params(runner, rows=rows), folder / f"params-rank{rank}.pt")
        progress = dict(progress, sharded=True, student=str(args.student), world=dist.get_world_size())
    else:
        save_student(runner, args.frozen_root or args.student, folder / "student", rank, step=step,
                     meta={"checkpoint": True, "lr": args.lr})
    if rank == 0 and BEH_CACHE:
        torch.save(BEH_CACHE, folder / "behavior.pt")
    dist.barrier()
    if rank == 0:
        (folder / "progress.json.tmp").write_text(json.dumps(progress))  # the completion mark, atomic
        os.replace(folder / "progress.json.tmp", folder / "progress.json")
        prune_checkpoints(out, folder)
    dist.barrier()
    event(log, event="checkpoint", step=step, path=str(folder), seconds=round(time.time() - began, 1),
          consumed=len(progress["consumed"]), sharded=sharded)


EXPERTS_KEY = "::experts"  # rows-only shards: <name>::experts = the global expert ids of <name>'s saved matrices


def _expert_rows(rows):
    """id(param) -> (local expert ids, global offset) of RowAdam's selected experts (HEAL_CKPT_ALL_EXPERTS=1: none)."""
    items = getattr(rows, "items", None)
    if not isinstance(items, list) or os.environ.get("HEAL_CKPT_ALL_EXPERTS") == "1":
        return {}
    return {id(it["p"]): (it["rows"], int(it["start"])) for it in items}


def reshard_all(runner):
    """Every FSDP2 group of ``runner`` back to its shards. A forward without backward (the no-grad held-out reads)
    leaves the root group (embed_tokens, lm_head, norms) unsharded: its parameters are then whole plain tensors, which
    a checkpoint saved as rank shards (c7k preflight 24 Oct: embed 248320 x 64 vs the 82774-row shard on resume).
    Local (frees the gathered copies); a no-op on sharded groups and outside FSDP."""
    roots = [runner] if isinstance(runner, torch.nn.Module) else [getattr(runner, "model", None), getattr(runner, "mtp", None)]
    for root in roots:
        for m in (root.modules() if isinstance(root, torch.nn.Module) else ()):
            if hasattr(m, "reshard") and hasattr(m, "unshard"):
                m.reshard()


def local_params(runner, pinned=False, rows=None):
    """This rank's shard of every trainable tensor, fp32 on the host (a sharded checkpoint). ``pinned``: into the
    reused host buffers (non_blocking; the caller synchronizes). ``rows`` (RowAdam): fused expert tensors keep only
    their selected experts (c7b 24 Oct: whole expert tensors were ~2/3 of a 443 GiB host copy; the others never
    train and equal the round's starting student, which a restart loads first), with ``<name>::experts``."""
    sel = _expert_rows(rows)
    out = {}
    reshard_all(runner)  # c7k pf 24 Oct: a no-grad read left the root group unsharded (embed saved whole)
    for name, p in runner.named_parameters():
        if not p.requires_grad:
            continue
        local = (p.to_local() if hasattr(p, "to_local") else p.data).detach()
        if id(p) in sel:
            idx, start = sel[id(p)]
            out[name + EXPERTS_KEY] = (idx.cpu() + start).long()
            local = local.index_select(0, idx) if len(idx) else local[:0]
        if pinned:
            out[name] = _host_buf(("p", name), local, torch.float32).copy_(local, non_blocking=True)
        else:
            out[name] = local.float().cpu().clone()
    return out


def load_local_params(runner, folder, rank):
    """A sharded checkpoint's tensors into this rank's shards (in place); returns how many. Rows-only expert
    tensors (``<name>::experts``) overwrite just those experts (the rest stay the starting student's)."""
    state = torch.load(Path(folder) / f"params-rank{rank}.pt", map_location="cpu", weights_only=True)
    reshard_all(runner)
    n = 0
    with torch.no_grad():
        for name, p in runner.named_parameters():
            if not p.requires_grad:
                continue
            local = p.to_local() if hasattr(p, "to_local") else p.data
            src = state.pop(name)
            ids = state.pop(name + EXPERTS_KEY, None)
            if ids is not None:
                start = 0
                if hasattr(p, "to_local"):
                    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

                    start = compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)[1][0]
                idx = ids.long() - start
                if src.shape[1:] != local.shape[1:] or len(idx) != src.shape[0] or (
                        len(idx) and (int(idx.min()) < 0 or int(idx.max()) >= local.shape[0])):
                    raise ValueError(f"sharded checkpoint {name}: experts {ids.tolist()[:4]}.. outside this shard")
                if len(idx):
                    local.index_copy_(0, idx.to(local.device), src.to(local.device, local.dtype))
            else:
                if src.shape != local.shape:
                    raise ValueError(f"sharded checkpoint {name}: {tuple(src.shape)} vs {tuple(local.shape)} (world size?)")
                local.copy_(src.to(local.dtype))
            n += 1
    if state:
        raise ValueError(f"sharded checkpoint has {len(state)} unknown tensors, e.g. {next(iter(state))}")
    return n


def join_params(shards):
    """Every rank's ``params-rank<r>.pt`` joined: name -> full tensor, or for rows-only expert tensors
    name -> {global expert id: matrix}."""
    import torch

    out = {}
    for key in shards[0]:
        if key.endswith(EXPERTS_KEY):
            continue
        if key + EXPERTS_KEY in shards[0]:
            out[key] = {int(e): s[key][i] for s in shards for i, e in enumerate(s[key + EXPERTS_KEY].tolist())}
        else:
            out[key] = torch.cat([s[key] for s in shards], 0)
    return out


def restore_fp32(model, folder):
    """A healed folder's trained tensors back at fp32 (from_pretrained casts them to BF16): returns how many."""
    from safetensors import safe_open

    params = dict(model.named_parameters())
    n = 0
    for shard in sorted(Path(folder).glob("trained-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if key in params and params[key].requires_grad:
                    params[key].data = f.get_tensor(key).to(params[key].device, torch.float32)
                    n += 1
    return n


PROFILE_KINDS = [  # kernel-name classes for the step profile (HEAL_PROFILE); first match wins
    ("comms", r"nccl"), ("optimizer", r"adam|foreach|multi_tensor|_fused_adam|index_copy|addcmul"),
    ("experts", r"grouped|gemm_grouped|sonic|quack|moe|expert|permute|scatter|gather_kernel|index_select|bincount"),
    ("gdn", r"gated_delta|chunk_|fused_recurrent|causal_conv|delta_rule|l2norm|fla_|kda"),
    ("attention", r"flash|fmha|attn|sdpa|softmax_warp|indexer|^_fwd$|^_bwd$"),  # _fwd/_bwd: rl1_sparse selected attn
    ("lm_head_kd", r"log_softmax|logsumexp|cross_entropy|topk|_kd|vocab"),
    ("gemm", r"gemm|cutlass|cublas|sm\d+_xmma|nvjet|matmul"),
    ("elementwise", r"elementwise|vectorized|reduce|copy|fill|cat|unrolled"),
]


def profile_summary(prof, wall):
    """Device time by class (sum of kernel durations, seconds) from a torch.profiler run, plus busy share."""
    import re

    out, kernels, total, launches, by_count = {k: 0.0 for k, _ in PROFILE_KINDS}, {}, 0.0, {}, {}
    out["other"] = 0.0
    # raw kineto events (C++ objects): prof.events() builds the Python op tree, which hung 25 min at 194 GiB RSS per
    # rank on two real steps with CPU activity on (sA-diag-v1)
    spans, kind, work = [], {}, []  # work: non-NCCL kernels (busy - busy_compute = time only comms ran)
    for e in prof.profiler.kineto_results.events():
        if not str(e.device_type()).endswith("CUDA"):
            continue
        dur = e.duration_ns() / 1e9 if hasattr(e, "duration_ns") else e.duration_us() / 1e6
        if dur <= 0:
            continue
        name0 = e.name()
        start = e.start_ns() / 1e9 if hasattr(e, "start_ns") else e.start_us() / 1e6
        spans.append((start, start + dur))
        if "nccl" not in name0.lower():
            work.append((start, start + dur))
        name, d = name0.lower(), dur
        if name.startswith(("memcpy", "memset")) or "cudamem" in name:
            out.setdefault("memcpy", 0.0)
            out["memcpy"] += d
            continue
        total += d
        kernels[name0[:90]] = kernels.get(name0[:90], 0.0) + d
        launches[name0[:90]] = launches.get(name0[:90], 0) + 1
        if name0 not in kind:
            kind[name0] = next((k for k, rx in PROFILE_KINDS if re.search(rx, name)), "other")
        out[kind[name0]] += d
        by_count[kind[name0]] = by_count.get(kind[name0], 0) + 1
    top = sorted(kernels.items(), key=lambda kv: -kv[1])[:25]
    def union(xs):  # union of kernel intervals over all streams: the time the GPU is doing work
        busy, end = 0.0, None
        for a, b in sorted(xs):
            if end is None or a > end:
                busy, end = busy + (b - a), b
            elif b > end:
                busy, end = busy + (b - end), b
        return busy

    busy = union(spans)
    return {"wall_s": round(wall, 2), "kernel_s": round(total, 2), "busy_s": round(busy, 2),
            "busy_compute_s": round(union(work), 2), "kernels": len(spans),
            "by_class_s": {k: round(v, 2) for k, v in out.items()},
            "top_kernels_s": [[n, round(v, 3)] for n, v in top], "kernels_by_class": by_count,
            "top_kernels_n": sorted(launches.items(), key=lambda kv: -kv[1])[:25]}


def stream_ages(version, versions, stream):
    """The data age of one update (OPD stream): updates between the weights that sampled each trained chunk and the
    weights this update started from (median, max), the buffer wait so far and chunks dropped as too old."""
    ages = sorted(version - v for v in versions)
    return {"age_median": ages[len(ages) // 2] if ages else None, "age_max": ages[-1] if ages else None,
            "stream_chunks": len(ages), "stream_wait_s": round(stream.wait_s, 1), "stream_dropped": stream.dropped}


def stream_pick(stream, world, step, base_share, counts, unit_of, room=None):
    """One slot's streamed picks (collective): base-play pieces while ``counts`` [base, student] is behind
    ``base_share`` (HEAL_STREAM_BASE_SHARE), else the student's. Returns (one path per rank, base?). Plain: counts
    in slots. HEAL_STREAM_COALESCE: counts in pieces; a rank's path is its unit's last piece (``unit_of`` maps it to
    the unit); a base unit is cut to the pieces still due, a student unit to ``room`` / world reply rows (the step's
    remaining positions), so neither overshoots by more than one piece per rank."""
    coalesce = stream.coalesce
    due = base_share / (1 - base_share) * max(1, counts[1]) - counts[0] if base_share > 0 else 0.0
    streamed, base = None, False
    if due > 0:
        streamed = stream.take(world, step, base=True, **({"max_pieces": max(1, math.ceil(due / world))}
                                                          if coalesce else {}))
        base = streamed is not None
    if streamed is None:
        streamed = stream.take(world, step, **({"max_rows": max(1.0, room / world)}
                                               if coalesce and room is not None else {}))
    if coalesce:
        for unit in streamed:
            unit_of[unit[-1]] = unit
        streamed = [unit[-1] for unit in streamed]
    add = sum(len(unit_of[x]) for x in streamed) if coalesce else 1
    counts[0 if base else 1] += add
    return streamed, base


class StreamQueue:
    """HEAL_STREAM=<dir> (OPD stream, 22 Oct): the round's own files arrive while it trains: teacher-scored chunks
    ``<name>.pt`` with a ``<name>.ready`` marker ({"version": the weights that sampled it, ...}). Rank 0 picks the
    freshest unconsumed chunks (highest version, then newest); chunks older than HEAL_STREAM_MAX_AGE updates are
    dropped; every rank gets rank 0's picks (the host group). A wait longer than HEAL_STREAM_STALL seconds is a crash
    (the feed died), not a stopping rule."""

    def __init__(self, folder, rank, consumed=(), max_age=6, poll=2.0, stall=3600.0, coalesce=False):
        self.folder, self.rank, self.max_age, self.poll, self.stall = Path(folder), rank, max_age, poll, stall
        self.taken, self.versions, self.wait_s, self.dropped = set(consumed), {}, 0.0, 0
        self.coalesce, self.meta, self.links = coalesce, {}, {}
        self.plan = []  # HEAL_STREAM_STEPPLAN: the step's remaining slots (rank 0)
        self.window, self.chunk = 32768, 8192  # HEAL_STREAM_ALIGN: the trainer sets its loss window and extend chunk
        self.stats = {"units": 0, "pieces": 0, "breaks": 0, "trimmed": 0}  # breaks / trimmed: rank 0's counts
        self.cut = 0
        self.fresh = float(os.environ.get("HEAL_FRESH_HOLDOUT") or 0)  # held-out student chain share
        self.held = {}  # rank 0: held-out piece path -> (version, mtime), unread
        # HEAL_MAX_UNIT (c7m 24 Oct): pieces whose context exceeds this many tokens are dropped unread (the launcher
        # sets the play's context window; the dry-run sizes the MTP mask and memory at it)
        self.max_unit = int(os.environ.get("HEAL_MAX_UNIT") or 0)
        self.stats["too_long"] = 0

    def ready(self):
        rows = []
        for marker in self.folder.glob("*.ready"):
            path = str(marker.with_suffix(".pt"))
            if path in self.taken:
                continue
            try:
                meta = json.loads(marker.read_text())
            except (OSError, ValueError):
                continue
            base = bool(meta.get("base")) or marker.name.startswith("b-")
            if self.max_unit and int(meta.get("tokens") or 0) > self.max_unit:
                self.taken.add(path)
                self.stats["too_long"] += 1
                continue
            self.meta[path] = meta
            if self.fresh and not base and self.held_out(path):  # HEAL_FRESH_HOLDOUT: never trained (FreshHoldout)
                self.held.setdefault(path, (int(meta.get("version", 0)), marker.stat().st_mtime))
                continue
            rows.append((int(meta.get("version", 0)), marker.stat().st_mtime, path, base))
        return rows

    def held_out(self, path):
        """HEAL_FRESH_HOLDOUT=<share>: whole student chains (a stable hash of the chain name) kept out
        of training; their pieces feed the per-step fresh held-out read."""
        import zlib
        return zlib.crc32(self.chain_of(path)[0].encode()) % 10000 < self.fresh * 10000

    @staticmethod
    def chain_of(path):
        """(chain name, piece number) of a streamed piece ``<chain>-pNNNN.pt``; (stem, None) otherwise."""
        head, sep, piece = Path(path).stem.rpartition("-p")
        return (head, int(piece)) if sep and piece.isdigit() else (Path(path).stem, None)

    def linked(self, a, b):
        """Piece b continues piece a: its row0 is a's last row and a's ids are a prefix of b's (a chain that broke,
        re-rendered or restarted fails this and its pieces stay separate)."""
        key = (a, b)
        if key not in self.links:
            ma, mb = self.meta.get(a, {}), self.meta.get(b, {})
            ta, tb, rb = ma.get("tokens"), mb.get("tokens"), mb.get("rows")
            ok = None not in (ta, tb, rb) and tb - rb == ta - 1
            if ok:
                try:
                    ia = torch.load(a, map_location="cpu", mmap=True, weights_only=True)["ids"]
                    ib = torch.load(b, map_location="cpu", mmap=True, weights_only=True)["ids"]
                    ok = ia.numel() == ta and ib.numel() >= ta and bool(torch.equal(ia, ib[:ta]))
                except Exception:  # noqa: BLE001 - an unreadable piece trains alone
                    ok = False
            self.links[key] = ok
            self.stats["breaks"] += int(not ok)
        return self.links[key]

    def units(self, fresh, n, max_pieces=None, max_rows=None):
        """HEAL_STREAM_COALESCE: ``n`` slot units from the fresh pieces [(version, mtime, path)]: runs of
        consecutive linked pieces of one chain, cut into parts of at most ``max_pieces`` pieces / ``max_rows`` reply
        rows (at least one piece each). Order: the chain's newest piece first, then a chain's parts in piece order;
        the freshest part leads, the n - 1 parts nearest it in tokens among the next 3 (n - 1) fill the slot
        (lockstep ranks wait for the longest). Fewer than ``n`` parts: the longest is split. None: fewer pieces."""
        chains = {}
        for v, mtime, path in fresh:
            head, k = self.chain_of(path)
            chains.setdefault(head if k is not None else path, []).append((k or 0, v, mtime, path))
        parts = []
        for rows in chains.values():
            rows.sort()
            runs, cur = [], [rows[0]]
            for r in rows[1:]:
                if r[0] == cur[-1][0] + 1 and self.linked(cur[-1][3], r[3]):
                    cur.append(r)
                    continue
                runs.append(cur)
                cur = [r]
            runs.append(cur)
            for run in runs:
                newest = max((r[1], r[2]) for r in run)
                part, rows_n = [], 0
                for r in run:
                    more = self.meta.get(r[3], {}).get("reply_rows", 0)
                    if part and ((max_pieces and len(part) >= max_pieces)
                                 or (max_rows is not None and rows_n + more > max_rows)):
                        parts.append((newest, part))
                        part, rows_n = [], 0
                    part.append(r)
                    rows_n += more
                parts.append((newest, part))
        while len(parts) < n and any(len(p) > 1 for _, p in parts):
            big = max(range(len(parts)), key=lambda i: len(parts[i][1]))
            newest, part = parts[big]
            parts[big:big + 1] = [(newest, part[:-1]), (newest, part[-1:])]
        if len(parts) < n:
            return None
        parts.sort(key=lambda x: (-x[0][0], -x[0][1], x[1][0][0]))
        runs = [p for _, p in parts]
        if os.environ.get("HEAL_STREAM_STEPPLAN") == "1" and max_rows is not None:
            return self.step_plan(runs, n, max_rows)
        size = lambda r: self.meta.get(r[-1][3], {}).get("tokens", 0)  # noqa: E731
        width = 3
        if os.environ.get("HEAL_STREAM_BALANCE") == "1":  # lever 1: level the slot's forward-equivalents, wider pool
            size = lambda r: self.fe([x[3] for x in r])  # noqa: E731
            width = int(os.environ.get("HEAL_STREAM_BALANCE_POOL") or 6)
        chain = lambda r: self.chain_of(r[0][3])[0]  # noqa: E731 - a chain's next part waits for a later slot
        pool = sorted(runs[1:1 + width * (n - 1)], key=lambda r: (chain(r) == chain(runs[0]), abs(size(r) - size(runs[0]))))
        picked = [runs[0]] + pool[:n - 1]
        if os.environ.get("HEAL_STREAM_ALIGN") == "1" and n > 1:  # MFU-1: the pool's (n - 1)-subset with the least
            # lockstep idle: per call position, every rank waits for the largest call (FSDP gathers per layer)
            others = [r for r in pool if chain(r) != chain(runs[0])] or pool
            best = None
            for combo in itertools.islice(itertools.combinations(others, n - 1), 5000):
                units = [runs[0], *combo]
                idle = self.lockstep_idle([[x[3] for x in r] for r in units])
                if best is None or idle < best[0]:
                    best = (idle, units)
            if best is not None:
                picked = best[1]
        ready = {x[3] for run in runs for x in run}
        nxt = lambda r: f"{self.chain_of(r[-1][3])[0]}-p{r[-1][0] + 1:04d}.pt"  # noqa: E731
        self.cut = sum(str(Path(r[-1][3]).with_name(nxt(r))) in ready for r in picked)  # units cut before a ready piece
        return [[x[3] for x in r] for r in picked]

    def step_plan(self, runs, n, max_rows):
        """HEAL_STREAM_STEPPLAN (MFU-1): the step's units at once, then grouped by call structure. The parts are taken
        freshest first (as the per-slot pick takes them) until their reply rows fill the step's room (``max_rows`` per
        rank x n) or the pool ends, rounded up to whole slots; sorted by (inert prefix, loss tokens) and dealt n at a
        time, so a slot's ranks run matching extend chunks and loss windows (lockstep waits for the largest call). The
        first slot is returned; the rest wait in ``self.plan`` for the next takes (rank 0 only; every rank gets the
        picks by ctl_gather). The step trains the same units in any grouping: the gradient sum is unchanged up to
        reduction order."""
        rows = lambda r: sum(self.meta.get(x[3], {}).get("reply_rows", 0) for x in r)  # noqa: E731
        take, total = [], 0
        for r in runs:
            if len(take) >= n and len(take) % n == 0 and total >= max_rows * n:
                break
            take.append(r)
            total += rows(r)
        take = take[:len(take) // n * n]

        def shape(r):
            first, last = self.meta.get(r[0][3], {}), self.meta.get(r[-1][3], {})
            row0 = int(first.get("tokens", 0)) - int(first.get("rows", 0))
            return (max(row0 - 1, 0) // 64 * 64, int(last.get("tokens", 0)))

        take.sort(key=shape)
        plan_of = [self.calls([x[3] for x in r]) for r in take]
        slot = list(range(len(take)))  # position -> unit index; slot k = positions k*n .. k*n+n-1
        span = lambda k: self.lockstep_span([plan_of[slot[k * n + j]] for j in range(n)])[0]  # noqa: E731
        spans = [span(k) for k in range(len(take) // n)]
        for _ in range(4):  # pairwise swaps across slots while the summed span drops (the slots' lockstep time)
            better = False
            for a in range(len(slot)):
                for b in range(a + 1, len(slot)):
                    ka, kb = a // n, b // n
                    if ka == kb:
                        continue
                    slot[a], slot[b] = slot[b], slot[a]
                    na, nb = span(ka), span(kb)
                    if na + nb < spans[ka] + spans[kb] - 1e-6:
                        spans[ka], spans[kb], better = na, nb, True
                    else:
                        slot[a], slot[b] = slot[b], slot[a]
            if not better:
                break
        take = [take[i] for i in slot]
        groups = [take[i:i + n] for i in range(0, len(take), n)]
        self.plan = [[[x[3] for x in r] for r in g] for g in groups[1:]]
        self.cut = 0
        return [[x[3] for x in r] for r in groups[0]]

    def planned(self, fresh):
        """The next buffered HEAL_STREAM_STEPPLAN slot when all its pieces are still fresh, else None (plan dropped)."""
        if not getattr(self, "plan", None):
            return None
        have = {path for _, _, path in fresh}
        nxt = self.plan.pop(0)
        if all(path in have for unit in nxt for path in unit):
            return nxt
        self.plan = []
        return None

    def calls(self, unit):
        """A unit's lockstep call costs in forward-equivalents (HEAL_STREAM_ALIGN): per slot window (slot_spans with
        the inert prefix), the loss call (window tokens x 4; 0 for the inert window) and its extend chunks (x 1)."""
        first, last = self.meta.get(unit[0], {}), self.meta.get(unit[-1], {})
        total = int(last.get("tokens", 0))
        row0 = int(first.get("tokens", 0)) - int(first.get("rows", 0))
        prefix = max(row0 - 1, 0) // 64 * 64 if os.environ.get("HEAL_INERT_PREFIX") == "1" else 0
        if total <= 0:
            return []
        spans, inert = slot_spans({"data": {"ids": torch.empty(1, total, device="meta")}, "prefix": prefix},
                                  self.window)
        out = []
        for i, (a, b) in enumerate(spans):
            ext = [min(self.chunk, b - s) for s in range(a, b, self.chunk)] if i < inert else []
            out.append((0.0 if i < inert else 4.0 * (b - a), [float(x) for x in ext]))
        return out

    def lockstep_idle(self, units, dummy=64.0):
        """Idle share of one lockstep slot of ``units`` (one per rank): per call position the largest call sets the
        time; a rank without a call there runs a ``dummy``-cost padding call."""
        span, busy = self.lockstep_span([self.calls(u) for u in units], dummy)
        return 1.0 - busy / max(span * len(units), 1e-9)

    @staticmethod
    def lockstep_span(plans, dummy=64.0):
        """(span, busy) of one lockstep slot from its ranks' call plans (``calls``), in forward-equivalents."""
        span = busy = 0.0
        for i in range(max((len(x) for x in plans), default=0)):
            loss = [x[i][0] if i < len(x) else 0.0 for x in plans]
            if any(loss) or os.environ.get("HEAL_SKIP_INERT_LOSS") != "1":
                c = [v or dummy for v in loss]
                span, busy = span + max(c), busy + sum(c)
            for j in range(max((len(x[i][1]) if i < len(x) else 0) for x in plans)):
                c = [x[i][1][j] if i < len(x) and j < len(x[i][1]) else dummy for x in plans]
                span, busy = span + max(c), busy + sum(c)
        return span, busy

    def fe(self, unit):
        """A unit's forward-equivalents (MFU-1 lever 1): loss-window tokens x 4 (forward, recompute, backward x 2) +
        no-grad extend tokens (S1's inert prefix), from the markers: the last piece's tokens, the first piece's row0."""
        first, last = self.meta.get(unit[0], {}), self.meta.get(unit[-1], {})
        total = int(last.get("tokens", 0))
        row0 = int(first.get("tokens", 0)) - int(first.get("rows", 0))
        prefix = max(row0 - 1, 0) // 64 * 64 if os.environ.get("HEAL_INERT_PREFIX") == "1" else 0
        return 4 * (total - prefix) + prefix

    def take(self, n, version, base=False, max_pieces=None, max_rows=None):
        """``n`` chunk paths for this slot (collective); ``version``: the trainer's current weights version.
        ``base``: base-model play chunks (forward-KD rows; no age limit, no wait: None when fewer than ``n``).
        ``coalesce``: ``n`` units (lists of a chain's consecutive piece paths; see ``units``)."""
        picks = None
        if self.rank == 0:
            t0 = time.time()
            while True:
                fresh = []
                for v, mtime, path, is_base in self.ready():
                    if is_base != base:
                        continue
                    if not base and version - v > self.max_age:
                        self.taken.add(path)
                        self.dropped += 1
                    else:
                        fresh.append((v, mtime, path))
                if len(fresh) >= n:
                    if not self.coalesce:
                        break
                    grouped = None if base else self.planned(fresh)
                    if grouped is None:
                        grouped = self.units(fresh, n, max_pieces, max_rows)
                    if grouped is not None:
                        break
                if base:
                    fresh = None
                    break
                if time.time() - t0 > self.stall:
                    raise SystemExit(f"stream stalled: {len(fresh)} of {n} chunks ready after {self.stall:.0f} s")
                time.sleep(self.poll)
            self.wait_s += time.time() - t0
            if fresh is not None and self.coalesce:
                version_of = {path: v for v, _, path in fresh}
                picks = [[(path, version_of[path], self.meta.get(path, {})) for path in unit] for unit in grouped]
            elif fresh is not None:
                fresh.sort(key=lambda r: (-r[0], -r[1]))
                picks = [(path, v) for v, _, path in fresh[:n]]
        import torch.distributed as dist
        got = ctl_gather(picks)[0] if dist.is_initialized() else picks
        if got is None:
            return None
        if self.coalesce:
            for unit in got:
                for path, v, meta in unit:
                    self.taken.add(path)
                    self.versions[path] = v
                    self.meta[path] = meta  # every rank: the per-rank token log reads it
                self.stats["units"] += 1
                self.stats["pieces"] += len(unit)
            self.stats["trimmed"] += self.cut  # rank 0's count (the others never group)
            return [[path for path, _, _ in unit] for unit in got]
        for path, v in got:
            self.taken.add(path)
            self.versions[path] = v
        return [path for path, _ in got]


def stage(args, world, progress, files=None):
    """The round's data plan (run_train and the CPU dry run share it): the slot queue (files of similar size share a
    lockstep slot of ``world`` files, slots shuffled by the seed), the consumed files a resume skips, the general
    windows (the first HEAL_PROBE_WINDOWS are the drift probe), own positions per file and the steps they fill."""
    files = teacher_files(args.teacher) if files is None else files
    seed = progress.get("seed", args.seed)  # a resume keeps the original order (not Random(step0): 19 Oct)
    rng = __import__("random").Random(seed)
    # a slot's ranks wait for its longest sequence: files of similar size (~length) share a slot, slots shuffled
    by_size = sorted(files, key=lambda f: (os.path.getsize(f), str(f)))
    groups = [by_size[i:i + world] for i in range(0, len(by_size), world)]
    rng.shuffle(groups)
    queue = [f for g in groups for f in g]
    consumed = list(progress.get("consumed", []))  # own files already trained on (identical on every rank)
    skipped = len(queue)
    queue = [f for f in queue if str(f) not in set(consumed)]
    skipped -= len(queue)
    general_all = sorted(Path(args.general).glob("*.pt")) if args.general else []
    probe_n = int(os.environ.get("HEAL_PROBE_WINDOWS") or 16)  # general drift-probe windows (CPU preflight: 4)
    general_n = int(load_teacher(general_all[probe_n], "cpu")["mask"].sum()) if len(general_all) > probe_n else 0
    own = [n * (1.0 - args.loop_share) for n in own_positions(queue, args.positions)]
    fit = simulate_steps(own, general_n, world, args.batch_tokens, args.general_share,
                         progress.get("gen_trained", 0.0), progress.get("own_trained", 0.0), args.flush_last)
    return {"files": files, "queue": queue, "consumed": consumed, "skipped": skipped, "seed": seed,
            "general_all": general_all, "probe_n": probe_n, "general_n": general_n, "own": own, "fit": fit}


def stream_schedule_problem(total_steps, schedule="wsd", env=None):
    """A streaming round's schedule (no staged files: auto counts 0 and gave c3, and every stream round g1e-c3, 1 step;
    23 Oct): an explicit --total-steps; a decay, if any, ends at it (OPD runs warm-up then a fixed lr,
    no decay, ending on play end + stall or a stop). None when fine."""
    env = os.environ if env is None else env
    if total_steps <= 0:
        return "a stream round needs an explicit --total-steps (auto counts staged files and a stream stages none: 1 step)"
    d0, dn = int(env.get("HEAL_DECAY_FROM") or 0), int(env.get("HEAL_DECAY_STEPS") or 0)
    if dn and d0 + dn != total_steps:
        return f"decay branch ends at step {d0 + dn}, the round at {total_steps}: the decay must complete at the last step"
    sched = int(env.get("HEAL_SCHEDULE_STEPS") or total_steps)
    if not dn and schedule == "wsd" and float(env.get("HEAL_WSD_DECAY") or 0.2) > 0 and sched != total_steps:
        return f"HEAL_SCHEDULE_STEPS {sched} != --total-steps {total_steps}: the decay would not complete in the round"
    return None


def schedule_steps(args, progress, fit):
    """The round's total steps: a restart keeps its round's; auto (0) = the staged data's steps less one of slack; an
    explicit count must fit the staged data."""
    if args.total_steps <= 0 and progress.get("total_steps"):  # a restart keeps the total its round settled on
        return int(progress["total_steps"])
    if args.total_steps <= 0:
        if args.step0 or progress:
            raise SystemExit("a resume needs the original round's --total-steps (the schedule must not change)")
        return max(1, fit - 1)  # the decay completes within the staged data, one step of slack
    if args.total_steps - args.step0 > fit and not args.allow_unfinished_schedule:
        raise SystemExit(f"schedule would not complete: {args.total_steps - args.step0} steps from step {args.step0} "
                         f"but the staged data fills {fit} (c1opd-v1 ran 18 of 21)")
    return args.total_steps


def run_plan(args):
    """``heal_kd.py plan <train args> --world N`` (the mandatory CPU dry run over the real files, no
    model): the round's own staging (stage, schedule_steps), every file loaded as the trainer loads it, the resume
    state, the general windows and the read selection. Writes <out>/plan.json; exits 1 listing every problem."""
    world, bad, t0 = args.world, [], time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = json.loads(Path(args.skip).read_text()) if args.skip else {}
    files = teacher_files(args.teacher)
    plan = {"world": world, "teacher": args.teacher, "files": len(files)}
    if not files:
        bad.append(f"no teacher files in {args.teacher}")
    samp_weight, sampler, samp_prefix = sampled_settings()
    plan["sampled"] = {"weight": samp_weight, "sampler": list(sampler), "prefix": samp_prefix}
    rows, k_seen, refused = {}, set(), {}
    for f in files:  # every file as the trainer loads it (tok_logp refusal, top-k width, replies after the loop cut)
        try:
            cpu = load_teacher_cpu(f)
        except Exception as e:  # noqa: BLE001
            refused.setdefault(f"{type(e).__name__}: {str(e).replace(str(f), '<file>')}"[:300], []).append(Path(f).name)
            continue
        k_seen.add(int(cpu["teacher"][0].shape[1]))
        r = {"tokens": int(cpu["inputs"]["input_ids"].numel()), "reply": int(cpu["reply"].sum())}
        if args.truncate_loops and r["reply"]:
            _, (spans, looped, removed) = truncate_loops({"ids": cpu["inputs"]["input_ids"][0], "reply": cpu["reply"]})
            r.update(loop_spans=looped, loop_removed=removed)
        if samp_weight and Path(f).name.startswith(samp_prefix):
            t = cpu["t_tok"][cpu["reply"].bool()]
            r["t_tok_nan"] = round(float(t.isnan().float().mean()), 4) if t.numel() else None
        rows[Path(f).name] = r
    bad += [f"{len(names)} files ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''}): {msg}"
            for msg, names in refused.items()]
    if len(k_seen) > 1:
        bad.append(f"teacher top-k differs across files: {sorted(k_seen)} (the dp dummy takes the first file's)")
    if args.positions == "reply":
        empty = [n for n, r in rows.items() if not r["reply"]]
        if empty:
            plan["no_reply_files"] = empty
            if len(empty) == len(rows):
                bad.append("no file has a reply position")
    reply = sum(r["reply"] for r in rows.values())
    removed = sum(r.get("loop_removed", 0) for r in rows.values())
    plan["reply_positions"], plan["loop_removed"] = reply, removed
    if args.truncate_loops and reply and abs(removed / reply - args.loop_share) > 0.05:
        bad.append(f"--loop-share {args.loop_share} but the loop cut removes {removed / reply:.3f} of reply positions")
    try:
        st = stage(args, world, progress, files)
        plan.update(queue=len(st["queue"]), skipped=st["skipped"], seed=st["seed"], steps_fit=st["fit"],
                    general_windows=len(st["general_all"]), general_window_positions=st["general_n"],
                    own_positions=int(sum(st["own"])))
        if args.general and len(st["general_all"]) <= st["probe_n"]:
            bad.append(f"{len(st['general_all'])} general windows: none left after the {st['probe_n']} probe windows")
        slots = [st["queue"][i:i + world] for i in range(0, len(st["queue"]), world)]
        plan["slots"], plan["short_slot"] = len(slots), (len(slots[-1]) if slots and len(slots[-1]) < world else 0)
        index_own = own_positions(files, args.positions)
        drift = [(Path(f).name, a, rows[Path(f).name]["reply"]) for f, a in zip(files, index_own)
                 if args.positions == "reply" and Path(f).name in rows and a != rows[Path(f).name]["reply"]]
        if drift:  # the step count comes from index.json; the trainer trains what the file holds
            bad.append(f"index.json reply_tokens differ from the files on {len(drift)}: {drift[:3]}")
        total = schedule_steps(args, progress, st["fit"])
        plan["total_steps"] = total
        if args.allow_unfinished_schedule and (why := stream_schedule_problem(args.total_steps, args.schedule)):
            bad.append(f"schedule: {why}")  # a stream round (c4, 23 Oct: the staged-fit rule refused its 500)
    except SystemExit as e:
        bad.append(f"schedule: {e}")
        total = args.total_steps
    except Exception as e:  # noqa: BLE001
        bad.append(f"stage: {type(e).__name__}: {e}"[:300])
        total = args.total_steps
    late = sorted(s for s in args.checkpoint_at if s > total or s <= args.step0)
    if late:
        bad.append(f"--checkpoint-at {late} outside steps {args.step0 + 1}..{total}")
    if args.resume:  # optimizer state written by exactly this many ranks (opd-r1b: 2-rank state resumed on 4 cards)
        have = sorted(int(p.stem.split("rank")[1]) for p in Path(args.resume).glob("optim-rank*.pt"))
        plan["resume_ranks"] = have
        if have != list(range(world)):
            bad.append(f"resume {args.resume}: optimizer ranks {have}, launch world {world} (heal_reshard.py)")
        params = sorted(Path(args.resume).glob("params-rank*.pt"))
        if params and len(params) != world:
            bad.append(f"resume {args.resume}: {len(params)} params-rank files, world {world}")
        if samp_weight and args.step0 and not (Path(args.resume) / "behavior.pt").is_file():
            plan["behavior"] = "recomputed under the resumed weights (no behavior.pt): importance weights shift"
    if progress.get("last_lr") is not None and args.step0:
        want = lr_at(args.step0 - 1, argparse.Namespace(**{**vars(args), "total_steps": total,
                                                          "step0": int(progress.get("lr_step0", 0))}))
        if abs(want - progress["last_lr"]) > 1e-9 * max(abs(want), 1e-12):
            bad.append(f"resume lr({args.step0 - 1}) {want:.4g} here, {progress['last_lr']:.4g} in the checkpoint")
    lrs = [lr_at(s, argparse.Namespace(**{**vars(args), "total_steps": total})) for s in range(args.step0, total)]
    plan["lr"] = {"first": lrs[0] if lrs else None, "max": max(lrs) if lrs else None, "last": lrs[-1] if lrs else None}
    reads = {}
    held2 = os.environ.get("HEAL_HELDOUT2")
    for name, folder, limit in (("heldout", args.heldout, args.heldout_files),
                                ("heldout2", held2, int(os.environ.get("HEAL_HELDOUT2_FILES") or 20)),
                                ("mid", args.heldout, int(os.environ.get("HEAL_MID_READ_FILES") or 5)),
                                ("mid2", held2, int(os.environ.get("HEAL_MID_READ_FILES") or 5))):
        if not folder or (name.startswith("mid") and not os.environ.get("HEAL_MID_READ_EVERY")):
            continue
        picked = read_files(folder, limit)
        no_reply = []
        for f in picked:
            try:
                if not int(load_teacher_cpu(f)["reply"].sum()):
                    no_reply.append(f.name)
            except Exception as e:  # noqa: BLE001
                bad.append(f"read {name} {f.name}: {type(e).__name__}: {e}"[:300])
        sessions = {f.name.split("--")[0] for f in picked}
        reads[name] = {"folder": str(folder), "files": len(picked), "sessions": len(sessions), "no_reply": no_reply,
                       "per_rank": [len(picked[r::world]) for r in range(world)] if os.environ.get(
                           "HEAL_SHARD_READS") == "1" else None}
        if not picked:
            bad.append(f"read {name}: no files in {folder}")
        elif no_reply:
            bad.append(f"read {name}: {len(no_reply)} of {len(picked)} files have no reply position")
    beh = os.environ.get("HEAL_HELDOUT2_BEH")
    if beh and os.environ.get("HEAL_SAMPLED_METRIC") == "1":
        reads["heldout2_beh"] = "exists" if Path(beh).is_file() else "written by this round"
    if os.environ.get("HEAL_PREREAD_FROM"):
        reused = preread_from(args, held2)
        reads["preread"] = sorted(k for k in reused if k != "source") or "none (the start reads run)"
    plan["reads"] = reads
    plan["seconds"] = round(time.time() - t0, 1)
    plan["problems"] = bad
    (out / "plan.json").write_text(json.dumps({"files_detail": rows, **plan}, indent=1))
    print(json.dumps(plan, indent=1))
    print("PLAN OK" if not bad else "PLAN FAILED")
    if bad:
        raise SystemExit(1)


def run_train(args):
    """One healing round (``torchrun --nproc-per-node N``): the student (``--student``: the cut, or the last round's
    saved folder) trained on this round's teacher files (``--teacher``: op_topk on the student's own replies) with
    ``LOSS_MIX`` (``--reverse`` / ``--forward``) on reply positions (``--positions reply``), the MTP head by forward
    KL (``--mtp-weight``); dense / attention / GDN / norm / router weights train in full, routed experts only on
    their top ``--esft`` share per layer by routed tokens (``--routing``: a saved count file, else measured on the
    first files); WSD over ``--total-steps`` from ``--step0``; ``--batch-tokens`` per optimizer step; ``--resume``:
    the last round's optimizer state. Every rank reads every file (FSDP shards memory only: the window and prefix
    passes issue rank-dependent collective counts, so ranks cannot split files; the compute is one card's)."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    import flash_qlora_job as fq

    cuda = torch.cuda.is_available()  # CPU: the train-tiny check (gloo)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo", timeout=pg_timeout())
    rank, world = dist.get_rank(), dist.get_world_size()
    ctl = ctl_group()  # HEAL_HOST_GROUP (a collective: every rank, before anything rank-dependent)
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda:
        torch.cuda.set_device(local)
    device = f"cuda:{local}" if cuda else "cpu"
    gib = (lambda: round(torch.cuda.max_memory_allocated() / 2**30, 2)) if cuda else (lambda: 0.0)
    peak_run = [0.0]  # c7n: the step's own peak (reset after each step event) beside the run peak

    def step_peak():
        if not cuda:
            return 0.0
        cur = torch.cuda.max_memory_allocated() / 2**30
        peak_run[0] = max(peak_run[0], cur)
        torch.cuda.reset_peak_memory_stats()
        return round(cur, 2)

    run_peak = lambda: round(max(peak_run[0], gib()), 2)  # noqa: E731
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out if rank == 0 else None
    LOSS_MIX.update(forward=args.forward, reverse=args.reverse, sampled=sampled_settings()[0])
    began = time.time()
    if os.environ.get("HEAL_LOAD_PREFETCH") == "1":
        n_files, n_bytes = prefetch_files(args.student, rank, world, int(os.environ.get("HEAL_LOAD_PREFETCH_THREADS", "8")))
        dist.barrier(group=ctl)
        event(log, event="load_prefetch", files=n_files, rank0_gib=round(n_bytes / 2**30, 2),
              seconds=round(time.time() - began, 1))
    model, frozen = load_student(args.student, device)
    restored = restore_fp32(model, args.student)
    text = model.config.text_config
    head = load_mtp(MtpHead(text, experts=getattr(text, "mtp_num_experts", None), top_k=args.mtp_top_k), args.student,
                    device, dtype=torch.float32) if args.mtp_weight else None
    runner = shard(Runner(model, head), init_device_mesh("cuda" if cuda else "cpu", (world,)), ignored=frozen)
    # from_pretrained leaves the model in eval mode, which turns gradient checkpointing off; before 21 Oct only the
    # first extend pass (runner.train(was) after it) switched it on, so a round without route_counts trained its first
    # windows in eval mode: smoke v5's next round ran out of memory, its resume took a general step in eval mode
    was_training = model.training
    runner.train()
    if args.resume and (Path(args.resume) / f"params-rank{rank}.pt").is_file():  # a sharded checkpoint (v25)
        event(log, event="params_resumed", tensors=load_local_params(runner, args.resume, rank), path=args.resume)
    files = teacher_files(args.teacher)
    stream_dir = os.environ.get("HEAL_STREAM")
    if stream_dir:  # OPD stream: the files arrive while training; a fixed expert selection (the round's routing)
        if not args.routing:
            raise SystemExit("HEAL_STREAM needs --routing (the ESFT selection cannot be measured on absent files)")
        files, args.allow_unfinished_schedule = [], True
        if why := stream_schedule_problem(args.total_steps, args.schedule):
            raise SystemExit(why)
    if not files and not stream_dir:
        raise SystemExit(f"no teacher files in {args.teacher}")
    if rank == 0 and log is not None:  # MFU-1 24 Oct: c7c's gathers ran RING_LL at ~13 GB/s; is NVLink P2P visible?
        try:
            import subprocess

            topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30).stdout
        except Exception as exc:  # noqa: BLE001 - a diagnostic
            topo = repr(exc)[:200]
        event(log, event="gpu_topo", topo=topo[-4000:], nccl_env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")})
    if rank == 0 and log is not None and os.environ.get("HEAL_HOST_EVERY", "60") != "0":  # c7b 24 Oct: host RAM trace
        import threading

        every = float(os.environ.get("HEAL_HOST_EVERY") or 60)

        def host_trace(every=every):  # bound now: run_train rebinds ``every`` later (c7f: sleep(None) ended the trace)
            while True:
                t0 = time.time()
                try:
                    avail, total = host_mem_gib()
                    event(log, event="host_mem", avail_gib=avail, total_gib=total, **host_breakdown(),
                          took_s=round(time.time() - t0, 2))
                except Exception as exc:  # noqa: BLE001 - a diagnostic: never ends the trace
                    event(log, event="host_mem", error=repr(exc)[:200])
                time.sleep(every)

        threading.Thread(target=host_trace, name="host-trace", daemon=True).start()
    event(log, event="loaded", seconds=round(time.time() - began, 1), files=len(files), world=world, restored_fp32=restored,
          gpu_gib=gib(), was_training=was_training, capture_cache=capture_on(),
          inert_prefix=os.environ.get("HEAL_INERT_PREFIX") == "1", recompute=recompute_policy())
    progress = json.loads(Path(args.skip).read_text()) if args.skip else {}
    st = stage(args, world, progress, files)
    queue, consumed, skipped, seed = st["queue"], st["consumed"], st["skipped"], st["seed"]
    general_all, probe_n, general_n, own, fit = st["general_all"], st["probe_n"], st["general_n"], st["own"], st["fit"]
    event(log, event="staged", positions=int(sum(own)), files=len(queue), skipped=skipped, steps_fit=fit,
          total_steps=args.total_steps, step0=args.step0, loop_share=args.loop_share, general_window=general_n,
          general_share=args.general_share if general_n else 0.0, seed=seed)
    args.total_steps = schedule_steps(args, progress, fit)
    event(log, event="schedule", total_steps=args.total_steps, step0=args.step0, steps_fit=fit)
    if progress.get("last_lr") is not None and args.step0:  # an exact resume continues the same schedule
        # against the writer's own schedule: HEAL_REWARM ramps the first steps after the writer's --step0, not ours
        # (the opd-v31b preflight's round B, 22 Oct: 6.667e-06 here vs the ramped 4.444e-06 in the checkpoint)
        want = lr_at(args.step0 - 1, argparse.Namespace(**{**vars(args), "step0": int(progress.get("lr_step0", 0))}))
        if abs(want - progress["last_lr"]) > 1e-9 * max(abs(want), 1e-12):
            raise SystemExit(f"resume schedule differs: lr({args.step0 - 1}) = {want:.4g} here, "
                             f"{progress['last_lr']:.4g} in the checkpoint (same --lr / --total-steps / --warmup)")
    routing = Path(args.routing) if args.routing else out / "routing.pt"
    cached = route_cache_path(args, files)  # same student + data + window: reuse counts across runs (315 s each)
    if routing.is_file():
        counts = torch.load(routing, map_location=device)
    elif cached is not None and cached.is_file():
        counts = torch.load(cached, map_location=device)
        event(log, event="routing_cached", path=str(cached))
        if rank == 0:
            torch.save(counts.cpu(), routing)
    else:
        counts = route_counts(runner, model, files, device, window=args.window, chunk=args.chunk, limit=args.route_files)
        if rank == 0:
            torch.save(counts.cpu(), routing)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                torch.save(counts.cpu(), cached.with_suffix(".tmp"))
                os.replace(cached.with_suffix(".tmp"), cached)
    top = max(1, int(round(counts.shape[1] * args.esft)))
    selected = {i: counts[i].topk(top).indices.tolist() for i in range(counts.shape[0])}
    expert_params, plain = [], []
    for name, p in runner.named_parameters():
        if not p.requires_grad:
            continue
        m = EXPERT_PARAM.search(name)
        if m and not name.startswith("mtp"):
            expert_params.append((p, selected[int(m.group(1))]))
        else:
            plain.append(p)
    rows = RowAdam(expert_params)
    adam = torch.optim.AdamW(plain, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=cuda)
    if args.resume:
        # host first: a GPU copy of the whole state next to the optimizer's own ran a B300 out of memory (opd1-v3)
        state = torch.load(Path(args.resume) / f"optim-rank{rank}.pt", map_location="cpu", weights_only=False)
        wrap_adam_state(adam, state["adam"])
        adam.load_state_dict(state["adam"])
        adopted = rows.load_state_dict(state["rows"])
        del state
        if cuda:
            torch.cuda.empty_cache()
        if adopted:  # the selection named by the esft event below is replaced by the resumed one for these tensors
            event(log, event="esft_resumed_selection", tensors=adopted)
    flags = comm_flags()
    tap = rows.tap(runner) if flags["expert_rows"] else None
    gsync = GradSync(runner, flags["every"]) if flags["sync"] == "step" else None
    counting = os.environ.get("HEAL_COUNT_COLLECTIVES") == "1"
    if counting:
        count_collectives()
    if any((tap, gsync, flags["prefetch"], flags["rs_buffers"], flags["reduce_dtype"] != torch.float32)):
        event(log, event="comm", sync=flags["sync"], every=flags["every"], reduce=str(flags["reduce_dtype"]),
              prefetch=flags["prefetch"], rs_buffers=flags["rs_buffers"], expert_rows=tap)
    step_dump = os.environ.get("HEAL_DUMP_STEPS")  # the comm check: full gradients (masked) and parameters per step
    if cuda:
        event(log, event="memory", when="optimizer", allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
              reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 2), peak_gib=gib())
    share = float(sum(len(v) for v in selected.values())) / max(1, sum(counts.shape[1] for _ in selected))
    event(log, event="esft", experts_per_layer=top, routed_share=round(float(
        sum(counts[i, selected[i]].sum() for i in selected) / counts.sum()), 4), expert_share=round(share, 4),
        plain_params=sum(p.numel() for p in plain), expert_params=len(expert_params))
    general = sorted(Path(args.general).glob("*.pt")) if args.general else []
    probe_dir = None
    sr_path = os.environ.get("HEAL_START_READS") if not args.step0 else None
    sr_cached, sr_key = start_reads_cache(sr_path, args, os.environ.get("HEAL_HELDOUT2"),
                                          int(os.environ.get("HEAL_HELDOUT2_FILES") or 20),
                                          int(os.environ.get("HEAL_MID_READ_FILES") or 5))
    sr_new = {}
    if general:  # the first probe_n (16) windows are the drift probe (never trained); the rest join the mix
        probe_dir = out / "general-probe"
        if rank == 0:
            probe_dir.mkdir(exist_ok=True)
            for f in general[:probe_n]:
                if not (probe_dir / f.name).exists():
                    (probe_dir / f.name).symlink_to(f)
        dist.barrier()
        general = general[probe_n:]
        if sr_cached and "general_probe" in sr_cached:
            event(log, event="general_probe", when="before", reused_from=sr_path, **sr_cached["general_probe"])
        else:
            sr_new["general_probe"] = probe_kl(runner, str(probe_dir), device, window=args.window, chunk=args.chunk)
            event(log, event="general_probe", when="before", **sr_new["general_probe"])
    held2 = os.environ.get("HEAL_HELDOUT2")  # a second held-out set (OPD: the student's own held-out-game rollouts)
    held2_files = int(os.environ.get("HEAL_HELDOUT2_FILES") or 20)
    prior = preread_from(args, held2)  # HEAL_PREREAD_FROM: the previous round's end reads stand in for these
    if args.heldout and os.environ.get("HEAL_CLASS_SPLIT") == "1" and not args.step0:  # the start's class split too
        if prior.get("heldout"):
            event(log, event="heldout_before", reused_from=prior["source"], **prior["heldout"])
        elif sr_cached and "heldout" in sr_cached:
            event(log, event="heldout_before", reused_from=sr_path, **sr_cached["heldout"])
        else:
            sr_new["heldout"] = heldout_kl(runner, model, args.heldout, device, window=args.window,
                                           chunk=args.chunk, limit=args.heldout_files)
            event(log, event="heldout_before", **sr_new["heldout"])
    if held2 and not args.step0:
        if prior.get("heldout2"):
            if os.environ.get("HEAL_SAMPLED_METRIC") == "1":  # the frozen held-out b, for the later reads
                frozen_behavior(read_files(held2, held2_files), runner, model, device, window=args.window,
                                chunk=args.chunk, prefix="", freeze=True)
            event(log, event="heldout2_before", reused_from=prior["source"], **prior["heldout2"])
        elif sr_cached and "heldout2" in sr_cached:
            if os.environ.get("HEAL_SAMPLED_METRIC") == "1":  # b from the frozen file (start_reads_cache checked it)
                frozen_behavior(read_files(held2, held2_files), runner, model, device, window=args.window,
                                chunk=args.chunk, prefix="", freeze=True)
            event(log, event="heldout2_before", reused_from=sr_path, folder=held2, **sr_cached["heldout2"])
        else:
            sr_new["heldout2"] = heldout_kl(runner, model, held2, device, window=args.window, chunk=args.chunk,
                                            limit=held2_files)
            event(log, event="heldout2_before", folder=held2, **sr_new["heldout2"])
    # HEAL_FIXED_BATCH=1 (P1 sanity): the KD of the first lockstep slot's files (trained at step 1),
    # read before step 1 and after the last step with the held-out reader; it must fall in a sane run
    fixed_dir = None
    if os.environ.get("HEAL_FIXED_BATCH") == "1" and not args.step0 and queue:
        fixed_dir = out / "fixed-batch"
        if rank == 0:
            fixed_dir.mkdir(exist_ok=True)
            for f in queue[:world]:
                if not (fixed_dir / Path(f).name).exists():
                    (fixed_dir / Path(f).name).symlink_to(Path(f).resolve())
        dist.barrier()
        event(log, event="fixed_before", names=[Path(f).name for f in queue[:world]],
              **heldout_kl(runner, model, str(fixed_dir), device, window=args.window, chunk=args.chunk,
                           samp_prefix=sampled_settings()[2] if sampled_settings()[0] else None))
    # HEAL_SAMPLED (22 Oct): the behavior policy's log b of every realised token of the student-sampled files, under
    # the round's starting weights (once; a resume reuses the checkpoint's copy so the importance weights match)
    samp_weight, sampler, samp_prefix = sampled_settings()
    if samp_weight and not stream_dir:  # a stream's chunks carry the engine's own log-probabilities
        saved_beh = Path(args.resume) / "behavior.pt" if args.resume else None
        if args.step0 and saved_beh is not None and saved_beh.is_file():
            BEH_CACHE.update(torch.load(saved_beh, map_location="cpu", weights_only=True))
            event(log, event="behavior_resumed", files=len(BEH_CACHE))
        else:
            want = [f for f in queue if Path(f).name.startswith(samp_prefix)]
            if args.total_steps > 0:  # a probe: only the files its steps can reach (x1.5 slack), in queue order
                need, got, keep = 1.5 * args.total_steps * args.batch_tokens, 0.0, []
                for f, n in zip(queue, own_positions(queue, args.positions)):
                    if got >= need:
                        break
                    got += n
                    if Path(f).name.startswith(samp_prefix):
                        keep.append(f)
                want = keep
            t0 = time.time()
            BEH_CACHE.update(behavior_pass(runner, model, want, device, window=args.window, chunk=args.chunk,
                                           sampler=sampler))
            event(log, event="behavior_pass", files=len(want), seconds=round(time.time() - t0, 1), sampler=list(sampler),
                  weight=samp_weight, clip=SAMPLED_CLIP)
    # data parallel: each rank trains its own sequence per lockstep slot; dp_sequence pads every rank
    # to the same sequence of FSDP calls (dummy windows / extend chunks), the loss scale is world / batch (FSDP
    # averages gradients), the batch and the general share count positions over all ranks
    # HEAL_MID_READ_EVERY=k (OPD round stop rules): every k steps (and before step 1) the first
    # HEAL_MID_READ_FILES (5) files of the held-out set (teacher-H forward) and of HEAL_HELDOUT2 (dh10: forward,
    # bucket reverse, top-32 tail, sampled reverse) -> "mid_read" events the watcher's stop rules read
    mid_every = int(os.environ.get("HEAL_MID_READ_EVERY") or 0)
    mid_files = int(os.environ.get("HEAL_MID_READ_FILES") or 5)

    def mid_read(at):
        got = {"step": at}
        if args.heldout:
            h = heldout_kl(runner, model, args.heldout, device, window=args.window, chunk=args.chunk, limit=mid_files)
            got.update(h_fwd=h["kd_reply"], h_rev=h.get("kd_reply_reverse"), h_tail32=h.get("student_out_top32_reply"))
        if held2:
            h = heldout_kl(runner, model, held2, device, window=args.window, chunk=args.chunk, limit=mid_files)
            got.update(d_fwd=h["kd_reply"], d_rev=h.get("kd_reply_reverse"), d_tail32=h.get("student_out_top32_reply"),
                       d_sampled=h.get("sampled_reverse"), d_w=h.get("sampled_w"))
            if os.environ.get("HEAL_MID_INT4") == "1" and rows is not None:  # the same read on the served int4 rows
                with Int4Rows(rows, {id(p): n for n, p in runner.named_parameters()}):
                    q = heldout_kl(runner, model, held2, device, window=args.window, chunk=args.chunk, limit=mid_files)
                got.update(d_fwd_int4=q["kd_reply"], d_rev_int4=q.get("kd_reply_reverse"),
                           d_sampled_int4=q.get("sampled_reverse"))
        if os.environ.get("HEAL_MID_TIES_AB") == "1" and at == args.step0:  # c7j: the same reads under the old QSA
            import rl1_trainer as rl1  # topk tie order (the anchor reads of D step 0 ran under it)
            mode = rl1.TIES["mode"]
            rl1.TIES["mode"] = "topk"
            try:
                if args.heldout:
                    got["h_fwd_topk"] = heldout_kl(runner, model, args.heldout, device, window=args.window,
                                                   chunk=args.chunk, limit=mid_files)["kd_reply"]
                if held2:
                    got["d_fwd_topk"] = heldout_kl(runner, model, held2, device, window=args.window, chunk=args.chunk,
                                                   limit=mid_files)["kd_reply"]
            finally:
                rl1.TIES["mode"] = mode
        if probe_dir is not None and os.environ.get("HEAL_MID_GENERAL") == "1":  # general-text drift (damage alert)
            got["g_kd"] = probe_kl(runner, str(probe_dir), device, window=args.window, chunk=args.chunk)["kd"]
        event(log, event="mid_read", files=mid_files, **got)
        return got

    if mid_every and not args.step0:
        if sr_cached and "mid_read0" in sr_cached:
            event(log, event="mid_read", files=mid_files, reused_from=sr_path, **sr_cached["mid_read0"])
        else:
            sr_new["mid_read0"] = mid_read(0)
    elif mid_every and args.step0 and os.environ.get("HEAL_MID_READ_AT_RESUME") == "1":  # c7i: the resumed weights
        mid_read(args.step0)
    if sr_path and sr_new and not sr_cached and rank == 0:  # the next fresh start from these weights reuses them
        try:
            Path(sr_path).parent.mkdir(parents=True, exist_ok=True)
            Path(sr_path).write_text(json.dumps({"key": sr_key, "reads": sr_new}, indent=1, default=float))
        except OSError as exc:
            event(log, event="start_reads_save_failed", error=str(exc)[:300])
    step_held = None
    live_lr = LiveLr(os.environ["HEAL_LIVE_KNOBS"], args.lr, int(os.environ.get("HEAL_LIVE_LR_RAMP") or 4)) \
        if os.environ.get("HEAL_LIVE_KNOBS") else None
    live_ckpt = LiveCkpt(os.environ["HEAL_LIVE_KNOBS"]) if os.environ.get("HEAL_LIVE_KNOBS") else None
    if os.environ.get("HEAL_STEP_HELDOUT") and held2:  # per-step held-out KD curve (StepHeldout)
        step_held = StepHeldout(runner, model, held2, device, tokens=int(os.environ["HEAL_STEP_HELDOUT"]),
                                window=args.window, chunk=args.chunk, rank=rank, world=world)
        event(log, event="step_heldout_setup", **step_held.setup_row())
    if os.environ.get("HEAL_OPD_LOG") == "1":
        TRAIN_ACC.clear()
        TRAIN_ACC["on"] = True
    gen_trained, own_trained, gen_index = 0.0, 0.0, 0
    step_files = []  # the current step's own files (picks are identical on every rank)
    step, acc = args.step0, {"kd": 0.0, "kd_reply": 0.0, "mtp": 0.0, "n": 0.0, "nr": 0.0, "nm": 0.0, "tokens": 0.0}
    loop_tot = [0.0, 0.0, 0.0]  # --truncate-loops: reply spans, looped spans, reply positions removed (all ranks)
    deadline = began + args.seconds
    if progress:  # resuming a mid-round checkpoint: its consumed files are skipped, the general counters restored
        gen_trained, own_trained, gen_index = progress["gen_trained"], progress["own_trained"], progress["gen_index"]
        event(log, event="skipped", files=skipped, left=len(queue), step0=args.step0)
    last_ckpt, lr = time.time(), (progress or {}).get("last_lr")

    def ckpt_progress():
        return {"step": step, "consumed": consumed, "gen_trained": gen_trained, "own_trained": own_trained,
                "gen_index": gen_index, "seed": seed, "last_lr": lr, "total_steps": args.total_steps,
                "lr_step0": args.step0}
    dummy = dp_dummy(model, device, k=int(os.environ.get("HEAL_STREAM_K") or 128) if stream_dir else
                     load_teacher(files[0], device)["teacher"][0].shape[1])
    coalesce = s4("STREAM_COALESCE")
    stream = StreamQueue(stream_dir, rank, consumed, max_age=int(os.environ.get("HEAL_STREAM_MAX_AGE") or 6),
                         stall=float(os.environ.get("HEAL_STREAM_STALL") or 3600), coalesce=coalesce) if stream_dir else None
    unit_of = {}  # HEAL_STREAM_COALESCE: a slot path (its unit's last piece) -> the unit's pieces
    fresh = FreshHoldout(runner, model, stream, device, window=args.window, chunk=args.chunk, rank=rank, world=world) \
        if stream is not None and stream.fresh else None
    tok_rank, loss_rank = [0.0] * world, [0.0] * world  # streamed tokens forwarded per rank this step (all / in loss)
    unit_rank = [0] * world  # the longest unit per rank this step (c7n: memory vs prediction at the actual length)
    exporter = None
    if os.environ.get("HEAL_DELTA_DIR"):  # per-step weight deltas for the serving student (heal_delta)
        import heal_delta
        exporter = heal_delta.DeltaExporter(os.environ["HEAL_DELTA_DIR"],
                                            heal_delta.Cut(Path(args.student) / "config.json",
                                                           os.environ.get("HEAL_DELTA_PRUNE")), rank,
                                            remote=os.environ.get("HEAL_DELTA_REMOTE") or None,
                                            log=lambda row: event(log, **row),
                                            base="resume" if args.step0 > 0 else "start",
                                            confirm=heal_delta.live_confirm(os.environ["HEAL_DELTA_CONFIRM_LIVE"],
                                                                            [os.environ["HEAL_DELTA_CONFIRM"]])
                                            if os.environ.get("HEAL_DELTA_CONFIRM_LIVE") and os.environ.get("HEAL_DELTA_CONFIRM")
                                            else heal_delta.play_confirm(os.environ["HEAL_DELTA_CONFIRM"])
                                            if os.environ.get("HEAL_DELTA_CONFIRM") else None)
    if exporter is not None:  # every start or resume (24 Oct): a new generation's full-row version at once, so a play
        # holding a lost step's weights (preemption) switches to these with nothing stale
        event(log, event="delta_export", start=True,
              **exporter.export(runner, rows, args.step0, force=True, first=True))
    step_versions, step_base, base_slots, student_slots = [], 0, 0, 0
    base_share = float(os.environ.get("HEAL_STREAM_BASE_SHARE") or 0.0)
    dump = os.environ.get("HEAL_DUMP_GRADS")
    # HEAL_PROFILE=A-B: torch.profiler over optimizer steps A..B (rank 0 also writes the chrome trace); a "profile"
    # event per rank: device time by kernel class, wall time, the top kernels (trainer MFU work)
    prof_span = tuple(map(int, os.environ["HEAL_PROFILE"].split("-"))) if os.environ.get("HEAL_PROFILE") else None
    prof, prof_t0 = None, None
    CLASS_ACC.clear()  # the held-out before-read's split is not this run's first step
    lazy = s4("LAZY_STATS")
    pending = []  # HEAL_LAZY_STATS: per slot [kd, kd_reply, kd_mtp, n_mtp] on the card, summed over ranks at the step

    def prep(path, is_general):  # a slot file's host part (HEAL_PREFETCH: in the background thread)
        cpu = load_teacher_cpu(path)
        if args.truncate_loops and not is_general:
            cut, loops = truncate_loops({"ids": cpu["inputs"]["input_ids"][0], "reply": cpu["reply"]})
            cpu["reply"], cpu["loops"] = cut["reply"], [float(x) for x in loops]
        if s4("PLE_TABLE"):
            cpu["ple"] = ple_rows(model, cpu["inputs"]["input_ids"][0])
        return pin_teacher(cpu) if cuda else cpu

    pf = Prefetch(prep) if s4("PREFETCH") and not stream_dir else None
    inert_check = os.environ.get("HEAL_INERT_CHECK") == "1"  # once, on the run's first trained (non-general) slot
    if inert_check and args.step0:  # MFU-2 23 Oct: the extra pass changed a later step's gradient (step 1: identical)
        inert_check = False
        event(log, event="inert_check_skipped", why="resumed round: the check runs only on a fresh start's step 1")
    # MFU-1 lever 1: HEAL_STREAM_PREFETCH: the next step's first stream slot is picked (collective, every rank at the
    # same point) before the optimizer step and this rank's pieces load on a host thread during the optimizer and the
    # delta export; fe_rank / fe_idle: forward-equivalents per rank and the lockstep idle share, per step
    ahead, preload = None, {}
    prefetch_on = os.environ.get("HEAL_STREAM_PREFETCH") == "1" and stream is not None
    fe_rank, fe_idle = [0.0] * world, [0.0, 0.0]
    loader = None
    if prefetch_on:
        from concurrent.futures import ThreadPoolExecutor
        loader = ThreadPoolExecutor(1)

    def host_piece(piece_path):
        got = preload.pop(piece_path, None)
        return got.result() if got is not None else load_teacher_cpu(piece_path)

    chunk_after(0, args.chunk)  # a malformed HEAL_CHUNK_AFTER fails at the start
    HOST.reset()
    while True:
        if prof_span and prof is None and step == prof_span[0] - 1 and cuda:
            from torch.profiler import ProfilerActivity, profile as _profile
            torch.cuda.synchronize()
            prof = _profile(activities=[ProfilerActivity.CUDA])  # kernels only: CPU ops made post-processing hang
            prof.start()
            prof_t0 = time.time()
        with HOST.phase("slot_sync"):
            HOST.wait("flag")
            flag = torch.tensor([float(time.time() > deadline or step >= args.total_steps)],
                                device="cpu" if ctl is not None else device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=ctl)
            stop_now = flag.item() > 0
        if stop_now:
            break
        picks = []  # identical on every rank (global counters); one kind per slot (general files are all one length)
        slot_loops = [0.0, 0.0, 0.0]
        use_general = bool(general) and gen_trained < args.general_share / max(1e-6, 1 - args.general_share) * own_trained
        streamed, slot_base = None, False
        if stream is not None and not use_general:  # HEAL_STREAM_BASE_SHARE: base-play slots at that share
            if ahead is not None:  # HEAL_STREAM_PREFETCH: picked at the last step's end, loading since
                streamed, slot_base = ahead
                ahead = None
            else:
                counts = [base_slots, student_slots]
                stream.window, stream.chunk = args.window, chunk_after(step + 1, args.chunk)  # HEAL_STREAM_ALIGN
                streamed, slot_base = stream_pick(stream, world, step, base_share, counts, unit_of,
                                                  room=args.batch_tokens - acc["n"])
                base_slots, student_slots = counts
            if coalesce:  # lever 1's read: forward-equivalents per rank and the slot's lockstep idle (max - own)
                fes = [stream.fe(unit_of.get(x, [x])) for x in streamed]
                for i, f in enumerate(fes):
                    fe_rank[i] += f
                fe_idle[0] += world * max(fes) - sum(fes)
                fe_idle[1] += world * max(fes)
            for i, x in enumerate(streamed):  # per-rank forwarded tokens (markers; S1: the loss from the prefix on)
                first = unit_of.get(x, [x])[0]
                total = stream.meta.get(x, {}).get("tokens", 0)
                row0 = stream.meta.get(first, {}).get("tokens", 0) - stream.meta.get(first, {}).get("rows", 0)
                tok_rank[i] += total
                unit_rank[i] = max(unit_rank[i], int(total))
                loss_rank[i] += total - (max(row0 - 1, 0) // 64 * 64 if os.environ.get("HEAL_INERT_PREFIX") == "1"
                                         else 0)
        for i in range(world):
            if use_general:
                picks.append((general[gen_index % len(general)], True))
                gen_index += 1
            elif streamed is not None:
                picks.append((streamed[i], False))
                for piece_path in unit_of.get(streamed[i], [streamed[i]]):
                    consumed.append(piece_path)
                    if not slot_base:  # the data age counts the student's own chunks
                        step_versions.append(stream.versions[piece_path])
                    step_base += int(slot_base)
            else:
                picks.append((queue.pop(0), False) if queue else (None, False))
                if picks[-1][0] is not None:
                    consumed.append(str(picks[-1][0]))
        slot_general = use_general
        step_files += [str(path) for path, gen in picks if path is not None and not gen]
        if all(path is None for path, _ in picks):
            if not args.flush_last or not acc["n"]:
                break
            mine = None
        else:
            path, use_general = picks[rank]
            mine = None
            if path is not None:
                with HOST.phase("load"):
                    if pf is not None:
                        data = teacher_to(pf.get((path, use_general)), device, non_blocking=cuda)
                        if data["loops"] is not None:
                            slot_loops = data["loops"]
                    elif len(unit_of.get(path, ())) > 1:  # HEAL_STREAM_COALESCE: a chain's pieces as one sequence
                        cpus = []
                        for piece_path in unit_of[path]:
                            cpu = host_piece(piece_path)
                            if args.truncate_loops:  # the loop cut per piece, as the pieces would train alone
                                cut, loops = truncate_loops({"ids": cpu["inputs"]["input_ids"][0], "reply": cpu["reply"]})
                                cpu["reply"], cpu["loops"] = cut["reply"], [float(x) for x in loops]
                            cpus.append(cpu)
                        data = teacher_to(merge_pieces(cpus), device)
                        if data["loops"] is not None:
                            slot_loops = data["loops"]
                        if s4("PLE_TABLE"):
                            data["ple"] = ple_rows(model, data["host"]["ids"], pin=cuda)
                    else:
                        data = teacher_to(host_piece(path), device) if not use_general else load_teacher(path, device)
                        if args.truncate_loops and not use_general:
                            cut, slot_loops = truncate_loops({"ids": data["ids"][0].cpu(), "reply": data["reply"].cpu()})
                            data["reply"] = cut["reply"].to(device)
                            slot_loops = [float(x) for x in slot_loops]
                            data["host"].update(reply=cut["reply"], reply_n=float(cut["reply"].sum()))
                        if s4("PLE_TABLE"):
                            data["ple"] = ple_rows(model, data["host"]["ids"], pin=cuda)
                own_mask = not (use_general or args.positions == "all")
                mask = data["reply"] if own_mask else data["mask"]
                with HOST.phase("pos"):
                    piece = fq.positions_and_images(model, data["inputs"], host_index=lazy)
                mine = {"data": data, "mask": mask, "general": use_general, "piece": piece, "ple": data.get("ple"),
                        "n_host": data["host"]["reply_n" if own_mask else "mask_n"], "nr_host": data["host"]["reply_n"]}
                if samp_weight and not use_general:
                    mine["samp"] = slot_samp(data, path, samp_prefix)
                if own_mask:  # HEAL_INERT_PREFIX: the KD mask is the reply; the sampled rows are a subset of it
                    mine["prefix"] = prefix_start(data["host"]["reply"])
            if pf is not None:  # the next slot's candidates: this rank's next own file, its next general file
                pf.want((queue[rank] if rank < len(queue) else None, False))
                if general:
                    pf.want((general[(gen_index + rank) % len(general)], True))
        if mine is not None or not all(path is None for path, _ in picks):
            slot_chunk = chunk_after(step + 1, args.chunk)  # HEAL_CHUNK_AFTER: the step this slot accumulates
            saved_mix = dict(LOSS_MIX)
            if mine is not None and mine["general"]:  # teacher-forced general text: forward KL only
                LOSS_MIX.update(forward=1.0, reverse=0.0, sampled=0.0)
            every = None
            if lazy:  # one host-side exchange per slot: call plans, positions, tokens, loop counts
                info = {"plan": slot_plan(mine, args.window, slot_chunk), "n": mine["n_host"] if mine else 0.0,
                        "nr": mine["nr_host"] if mine else 0.0,
                        "tok": float(mine["data"]["ids"].shape[1]) if mine else 0.0,
                        "general": bool(mine and mine["general"]), "loops": slot_loops,
                        "loss": slot_losses(mine, args.window)}
                with HOST.phase("slot_sync"):
                    HOST.wait("exchange")
                    every = ctl_gather(info)
                HOST.slot([x["tok"] for x in every], [len(x["plan"]) for x in every])
            if gsync is not None:  # does a step follow this slot? run_train's rule below, on positions known now
                if every is not None:
                    pn = 0.0
                    for x in every:
                        pn += x["n"]
                else:
                    with HOST.phase("slot_sync"):
                        box = torch.tensor([float(mine["mask"].sum()) if mine is not None else 0.0],
                                           dtype=torch.float64, device=device)
                        dist.all_reduce(box)
                        HOST.wait("box")
                        pn = box.item()
                g2, o2 = (gen_trained + pn, own_trained) if slot_general else (gen_trained, own_trained + pn)
                more = bool(general) and g2 < args.general_share / max(1e-6, 1 - args.general_share) * o2
                gsync.slot(acc["n"] + pn >= args.batch_tokens or (not more and not queue and stream is None))
            check_now = inert_check and not slot_general  # global: every rank has the same picks
            on_trace = [] if check_now else None
            try:
                out_row = dp_sequence(runner, mine, dummy, window=args.window, chunk=slot_chunk,
                                      scale=world / args.batch_tokens,
                                      mtp_weight=0.0 if (mine is None or mine["general"]) else args.mtp_weight,
                                      plans=[x["plan"] for x in every] if every is not None else None, lazy=lazy,
                                      losses=[x["loss"] for x in every] if every is not None else None,
                                      trace=on_trace, trace_cache=False,
                                      **({"sync": gsync} if gsync is not None else {}))
                if check_now:  # HEAL_INERT_CHECK: the first trained slot again, no grad, S1 off (training: S1-on only)
                    inert_check = False
                    on = {"kd": float(out_row["kd"]), "kd_reply": float(out_row["kd_reply"]),
                          "loss": sum(t["loss"] for t in on_trace)}
                    off = inert_check_pass(runner, mine, dummy, window=args.window, chunk=slot_chunk,
                                           mtp_weight=0.0 if mine is None else args.mtp_weight, lazy=lazy)
                    event(log, event="inert_check", step=step + 1, chunk=slot_chunk,
                          prefix=int(mine.get("prefix", 0)) if mine else None, **inert_check_event(on, off, device))
            finally:
                LOSS_MIX.update(saved_mix)
            if every is not None:  # the KD sums at the step; the host counts summed over ranks as all_reduce would
                pending.append(torch.stack([as64(out_row[key], device) for key in ("kd", "kd_reply", "kd_mtp", "n_mtp")]))
                kd = kr = km = nm = 0.0
                n = nr = tok = gn = on = lsp = llp = lrm = 0.0
                for x in every:
                    n += x["n"]; nr += x["nr"]; tok += x["tok"]
                    gn += x["n"] if x["general"] else 0.0
                    on += 0.0 if x["general"] else x["n"]
                    lsp += x["loops"][0]; llp += x["loops"][1]; lrm += x["loops"][2]
            else:
                stats = torch.tensor([out_row["kd"], out_row["n"], out_row["kd_reply"], out_row["n_reply"],
                                      out_row["kd_mtp"], out_row["n_mtp"], out_row["tokens"],
                                      out_row["n"] if (mine and mine["general"]) else 0.0,
                                      out_row["n"] if (mine and not mine["general"]) else 0.0, *slot_loops],
                                     dtype=torch.float64, device=device)
                with HOST.phase("slot_sync"):
                    dist.all_reduce(stats)
                    HOST.wait("stats")
                    kd, n, kr, nr, km, nm, tok, gn, on, lsp, llp, lrm = stats.tolist()
            loop_tot = [loop_tot[0] + lsp, loop_tot[1] + llp, loop_tot[2] + lrm]
            gen_trained, own_trained = gen_trained + gn, own_trained + on
            acc["kd"] += kd; acc["n"] += n; acc["kd_reply"] += kr; acc["nr"] += nr; acc["mtp"] += km; acc["nm"] += nm
            acc["tokens"] += tok
            exhausted = all(path is None for path, _ in picks)
            if acc["n"] < args.batch_tokens and not exhausted:
                continue
        late = False
        if prefetch_on and not exhausted and step + 1 < args.total_steps:  # identical on every rank
            # the pick is a collective: the deadline is agreed (MAX over ranks) as at the loop top, never per rank
            flag = torch.tensor([float(time.time() > deadline)], device="cpu" if ctl is not None else device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=ctl)
            late = flag.item() > 0
        if prefetch_on and not exhausted and step + 1 < args.total_steps and not late:
            # the next loop top's own rule (gen/own trained do not change in the optimizer step)
            if not (bool(general) and gen_trained < args.general_share / max(1e-6, 1 - args.general_share) * own_trained):
                with HOST.phase("prefetch"):
                    counts = [base_slots, student_slots]
                    stream.window, stream.chunk = args.window, chunk_after(step + 2, args.chunk)
                    ahead = stream_pick(stream, world, step + 1, base_share, counts, unit_of, room=args.batch_tokens)
                    base_slots, student_slots = counts
                    for piece_path in unit_of.get(ahead[0][rank], [ahead[0][rank]]):
                        preload[piece_path] = loader.submit(load_teacher_cpu, piece_path)
        opt_phase = HOST.phase("opt")
        opt_phase.__enter__()
        if pending:  # HEAL_LAZY_STATS: the step's KD sums over ranks, added slot by slot as the per-slot path does
            block = torch.stack(pending)
            dist.all_reduce(block)
            HOST.wait("stats")
            for kd, kr, km, nm in block.tolist():
                acc["kd"] += kd; acc["kd_reply"] += kr; acc["mtp"] += km; acc["nm"] += nm
            pending = []
        syncs = gsync.syncs if gsync is not None else None
        if gsync is not None:
            gsync.step_done()
        if tap:
            rows.reduce_taps()
        if dump and step == args.step0:  # the data-parallel equality check: gradients of the first step
            grads = {}
            for name, prm in runner.named_parameters():
                if prm.grad is not None:
                    g = prm.grad.full_tensor() if hasattr(prm.grad, "full_tensor") else prm.grad
                    grads[name] = g.detach().float().cpu()
            if tap:
                grads.update(full_grads(runner, rows, experts_only=True))
            if rank == 0:
                torch.save(grads, dump)
        rows.mask_grads()
        # backprop-window truncation bias: HEAL_GRAD_SHARD_DUMP=DIR saves this rank's local gradient shards
        # at the first step; a second process with another --window and HEAL_GRAD_COSINE=DIR (same data, seed, world)
        # reports the cosine of the two gradients (overall and per parameter kind) and stops before the update
        shard_dir = os.environ.get("HEAL_GRAD_SHARD_DUMP") or os.environ.get("HEAL_GRAD_COSINE")
        if shard_dir and step == args.step0:
            local = {n: (q.grad.to_local() if hasattr(q.grad, "to_local") else q.grad)
                     for n, q in runner.named_parameters() if q.grad is not None}
            path = Path(shard_dir) / f"grad-rank{rank}.pt"
            if os.environ.get("HEAL_GRAD_SHARD_DUMP"):
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({n: g.detach().to(torch.bfloat16).cpu() for n, g in local.items()}, path)
                event(log, event="grad_shards_saved", window=args.window, tensors=len(local))
            else:
                ref = torch.load(path, map_location="cpu")
                sums = {}
                for n, g in local.items():
                    if n not in ref:
                        continue
                    kind = "experts" if EXPERT_PARAM.search(n) else ("mtp" if ".mtp." in n or n.startswith("mtp") else "dense")
                    x, y = g.detach().float(), ref[n].to(g.device).float()
                    acc3 = sums.setdefault(kind, torch.zeros(3, dtype=torch.float64, device=device))
                    acc3 += torch.stack([(x * y).sum(), (x * x).sum(), (y * y).sum()]).double()
                kinds = sorted(sums)
                stack = torch.stack([sums[k] for k in kinds]) if kinds else torch.zeros(0, 3, device=device)
                dist.all_reduce(stack)
                tot = stack.sum(0)
                cos = lambda v: float(v[0] / (v[1].sqrt() * v[2].sqrt()).clamp_min(1e-30))
                event(log, event="grad_cosine", window=args.window, cosine=round(cos(tot), 5),
                      norm_ratio=round(float((tot[1] / tot[2].clamp_min(1e-30)).sqrt()), 5),
                      by_kind={k: round(cos(stack[i]), 5) for i, k in enumerate(kinds)})
                dist.destroy_process_group()
                return
        if step_dump:
            grads = full_grads(runner, rows)
            if rank == 0:
                Path(step_dump).mkdir(parents=True, exist_ok=True)
                torch.save(grads, Path(step_dump) / f"grads-step{step + 1}.pt")
        if tap:  # clip_grad_norm_ over the FSDP gradients and the row buffer together
            dense = [q.grad for q in runner.parameters() if q.requires_grad and q.grad is not None]
            dn = torch.nn.utils.get_total_norm(dense, 2.0)
            dn = dn.full_tensor() if hasattr(dn, "full_tensor") else dn
            norm = (dn.float().pow(2) + rows.tap_norm2()).sqrt()
            coef = torch.clamp(1.0 / (norm + 1e-6), max=1.0)
            torch._foreach_mul_([g.to_local() if hasattr(g, "to_local") else g for g in dense], coef)
            rows.flat.mul_(coef)
        else:
            norm = torch.nn.utils.clip_grad_norm_([q for q in runner.parameters() if q.requires_grad], 1.0)
        lr = lr_at(step, args)
        if live_lr is not None:  # HEAL_LIVE_KNOBS: the live peak lr (LiveLr)
            lr_scale = live_lr.scale(step, rank)
            lr *= lr_scale
        # HEAL_FIXED_BATCH descent check: the last step's own batch, read just before and just after
        # its full-lr update, reply positions (kd_mix_reply); the gradients stay untouched by the no-grad reads
        last_dir = None
        if os.environ.get("HEAL_FIXED_BATCH") == "1" and step + 1 == args.total_steps and step_files:
            last_dir = out / "last-batch"
            if rank == 0:
                last_dir.mkdir(exist_ok=True)
                for f in step_files:
                    if not (last_dir / Path(f).name).exists():
                        (last_dir / Path(f).name).symlink_to(Path(f).resolve())
            dist.barrier()
            sp = sampled_settings()[2] if sampled_settings()[0] else None
            event(log, event="last_batch_before", step=step + 1, lr=lr, names=[Path(f).name for f in step_files],
                  **heldout_kl(runner, model, str(last_dir), device, window=args.window, chunk=args.chunk, samp_prefix=sp))
        for group in adam.param_groups:
            group["lr"] = lr
        adam.step()
        rows.step(lr)
        adam.zero_grad(set_to_none=False)
        rows.zero_grad()  # the expert parameters are not in adam's groups
        step_files = []
        if last_dir is not None:
            event(log, event="last_batch_after", step=step + 1,
                  **heldout_kl(runner, model, str(last_dir), device, window=args.window, chunk=args.chunk, samp_prefix=sp))
        step += 1
        HOST.wait("step")
        finite = bool(torch.isfinite(norm))
        opt_phase.__exit__(None, None, None)
        log_phase = HOST.phase("log")
        log_phase.__enter__()
        sig = lambda x: float(f"{x:.7g}")  # significant digits: the exact-resume checks compare these
        by_class = train_classes(device) if os.environ.get("HEAL_TRAIN_CLASS_SPLIT") == "1" else None
        opd = None
        if TRAIN_ACC.get("on"):  # this step's distillation objective over all ranks
            t = torch.tensor([float(TRAIN_ACC.get(k, 0.0)) for k in ("value", "n", "est", "w", "sn")],
                             dtype=torch.float64, device=device)
            dist.all_reduce(t)
            v, n, es, ws, sn = t.tolist()
            opd = {"opd_objective": sig(v / max(n, 1)), "opd_sampled_rev": sig(es / sn) if sn else None,
                   "opd_is_weight": sig(ws / sn) if sn else None, "opd_sampled_positions": int(sn)}
            TRAIN_ACC.clear()
            TRAIN_ACC["on"] = True
        pinned_rank = None
        if cuda and hasattr(torch.cuda, "host_memory_stats"):  # c7b 24 Oct: the pinned host pool, per rank
            def pst():
                hs = torch.cuda.host_memory_stats()
                return [round(hs.get("allocated_bytes.current", 0) / 2 ** 30, 2),
                        round(hs.get("active_bytes.current", 0) / 2 ** 30, 2)]

            before = pst()
            if os.environ.get("HEAL_PIN_EMPTY", "1") == "1" and hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()  # cached (inactive) pinned blocks back to the OS each step
            pinned_rank = ctl_gather(before + pst())  # [pool, active, pool after empty, active after] GiB per rank
        with HOST.leaf("heldout"):
            held_row = step_held.read() if step_held is not None else {}
            if fresh is not None:
                held_row.update(fresh.read(step))
        event(log, event="step", step=step, lr=lr, tokens=int(acc["tokens"]), trained=int(acc["n"]),
              chunk=chunk_after(step, args.chunk),
              kd=sig(acc["kd"] / max(acc["n"], 1)), kd_reply=sig(acc["kd_reply"] / max(acc["nr"], 1)),
              kd_mtp=sig(acc["mtp"] / max(acc["nm"], 1)) if acc["nm"] else None, grad_norm=sig(float(norm)),
              finite=finite, seconds=round(time.time() - began, 1), peak_gib=run_peak(), step_peak_gib=step_peak(), world=world, host_avail_gib=host_mem_gib()[0],
              **({"pinned_rank": pinned_rank} if pinned_rank is not None else {}),
              **({"loop_spans": int(loop_tot[0]), "loop_looped": int(loop_tot[1]), "loop_removed": int(loop_tot[2])}
                 if args.truncate_loops else {}), **({"synced_backwards": syncs} if syncs is not None else {}),
              **({"by_class": by_class} if by_class else {}), **(opd or {}), **held_row,
              **({"think_eos_masked": int(THINK_ACC.get("masked", 0)), "think_eos_no_col": int(THINK_ACC.get("no_eos_column", 0))}
                 if THINK_EOS else {}),  # this rank's loader counts so far (the watcher checks the mask runs)
              **({"live_lr": live_lr.seen, "lr_scale": round(lr_scale, 6)} if live_lr is not None else {}),
              **(stream_ages(step - 1, step_versions, stream) if stream is not None else {}),
              **({"stream_base": step_base} if stream is not None and base_share > 0 else {}),
              **({"fe_rank": [int(x) for x in fe_rank], "fe_idle_share": round(fe_idle[0] / fe_idle[1], 4) if fe_idle[1] else None,
                  "prefetched": bool(ahead is not None)} if stream is not None and coalesce else {}),
              **({"tok_rank": [int(x) for x in tok_rank], "loss_tok_rank": [int(x) for x in loss_rank], "unit_max_rank": list(unit_rank),
                  **({"coalesce": dict(stream.stats)} if coalesce else {})}
                 if stream is not None and (coalesce or os.environ.get("HEAL_INERT_PREFIX") == "1") else {}))
        step_versions, step_base = [], 0
        tok_rank, loss_rank = [0.0] * world, [0.0] * world
        unit_rank = [0] * world
        fe_rank, fe_idle = [0.0] * world, [0.0, 0.0]
        if exporter is not None and step % int(os.environ.get("HEAL_DELTA_EVERY") or 1) == 0:
            # every rank (gathers); rank 0 writes and uploads in the background; HEAL_DELTA_EVERY: sync cadence
            with HOST.leaf("delta"):
                row = exporter.export(runner, rows, step)
            event(log, event="delta_export", **row)
        if mid_every and step % mid_every == 0 and step < args.total_steps:
            with HOST.leaf("mid_read"):
                mid_read(step)
        if counting:  # this step's collectives per rank (equal across ranks), then reset
            mine_c = {k: list(v) for k, v in COMM_COUNTS.items()}
            every_c = [None] * world
            dist.all_gather_object(every_c, {"counts": mine_c, "peak_gib": gib()})
            event(log, event="comm_counts", step=step, counts=every_c[0]["counts"],
                  equal=all(c["counts"] == every_c[0]["counts"] for c in every_c),
                  peak_gib=[c["peak_gib"] for c in every_c])
            COMM_COUNTS.clear()
        if step_dump:
            params = {}
            for name, q in runner.named_parameters():
                if q.requires_grad:
                    params[name] = (q.full_tensor() if hasattr(q, "full_tensor") else q.data).detach().float().cpu()
            if rank == 0:
                torch.save(params, Path(step_dump) / f"params-step{step}.pt")
        if not finite:
            raise SystemExit("non-finite gradient norm")
        if prof is not None and prof_span and step >= prof_span[1]:
            torch.cuda.synchronize()
            prof.stop()
            post0 = time.time()
            summary = profile_summary(prof, post0 - prof_t0)
            summary.update(rank=rank, steps=list(prof_span), tokens=int(acc["tokens"]), post_s=round(time.time() - post0, 1))
            line = json.dumps({"t": round(time.time(), 1), "event": "profile", **summary})
            with open(out / f"profile-rank{rank}.json", "w") as f:
                f.write(line + "\n")
            event(log, **{"event": "profile", **{k: v for k, v in summary.items() if k != "top_kernels_s"}})
            # no chrome trace: rank 0's export of 1.3M kernels outlasted the 600 s NCCL timeout (sA-diag-s1)
            prof, prof_span = None, None
        log_phase.__exit__(None, None, None)
        if HOST.on:  # every rank's host phases of this step (the gather itself is not booked)
            report = HOST.report()
            report.update(rank=rank, prefetch=[pf.hits, pf.misses] if pf is not None else None)
            event(log, event="host", step=step, ranks=ctl_gather(report))
            HOST.reset()
        acc = {k: 0.0 for k in acc}
        if all(path is None for path, _ in picks):
            break
        due = (args.checkpoint_seconds and time.time() - last_ckpt >= args.checkpoint_seconds) or step in args.checkpoint_at
        if live_ckpt is not None and live_ckpt.due(rank):  # HEAL_LIVE_KNOBS "ckpt": a new tag saves at this boundary
            due = True
        if due and (queue or stream is not None):
            with HOST.phase("ckpt"):
                checkpoint(runner, adam, rows, args, out, rank, step=step, log=log, progress=ckpt_progress())
            last_ckpt = time.time()
    if pf is not None:
        pf.close()
    if exporter is not None:
        exporter.wait()
    if not args.no_save and args.checkpoint_seconds and step > args.step0:  # timed checkpoints only (not the smokes')
        # the last step too: a preemption in the held-out read or the final save keeps the round
        checkpoint(runner, adam, rows, args, out, rank, step=step, log=log, progress=ckpt_progress())
    if gsync is not None:
        gsync.restore()
    if probe_dir is not None:
        event(log, event="general_probe", when="after", **probe_kl(runner, str(probe_dir), device, window=args.window,
                                                                   chunk=args.chunk))
    if fixed_dir is not None:
        event(log, event="fixed_after", **heldout_kl(runner, model, str(fixed_dir), device, window=args.window,
                                                     chunk=args.chunk,
                                                     samp_prefix=sampled_settings()[2] if sampled_settings()[0] else None))
    if args.heldout:  # the lr probe's read: held-out reply KL of the trained weights, every rank on the same files
        if held2:
            event(log, event="heldout2", folder=held2, **heldout_kl(runner, model, held2, device, window=args.window,
                                                                   chunk=args.chunk, limit=held2_files))
        event(log, event="heldout", **heldout_kl(runner, model, args.heldout, device, window=args.window,
                                                 chunk=args.chunk, limit=args.heldout_files))
    wait_checkpoint()  # an async checkpoint completes before the round's own save
    saved = None
    if not args.no_save:
        torch.save({"adam": adam.state_dict(), "rows": rows.state_dict()}, out / f"optim-rank{rank}.pt")
        saved = save_student(runner, args.frozen_root or args.student, out / "student", rank, step=step,
                             meta={"teacher": str(args.teacher), "lr": args.lr, "reverse": args.reverse,
                                   "forward": args.forward, "esft": args.esft, "total_steps": args.total_steps,
                                   "last_lr": lr, "seed": seed})
    event(log, event="round_done", step=step, seconds=round(time.time() - began, 1), **(saved or {}))
    dist.destroy_process_group()

def run_preflight_student(args):
    """A tiny cut student on disk in the real student's layout with the real vocabulary and top-10 routing, so a
    launch's own ``train`` arguments run unchanged on real teacher files (heal_preflight.py; PTH 19 Oct)."""
    import shutil

    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(Path(args.config).read_text())["text_config"]["vocab_size"]
    teacher = tiny_teacher(args.config, vocab_size=vocab, experts=32, per_token=10)
    teacher.save_pretrained(str(root / "tiny-teacher"), max_shard_size="64MB")
    (root / "prune.json").write_text(json.dumps({"format": "arc-expert-prune-v1", "experts": 32,
                                                 "pruned": {str(i): list(range(1, 32, 2)) for i in range(6)}}))
    cut_checkpoint(root / "tiny-teacher", root / "tiny-cut", drop=[4], prune_file=root / "prune.json", top_k=10,
                   shard_gib=0.05)
    cut = AutoModelForMultimodalLM.from_pretrained(str(root / "tiny-cut"), dtype=torch.bfloat16)
    (root / "tiny-student").mkdir()
    state = {k: v.contiguous() for k, v in cut.state_dict().items()}
    head = mtp_tiny(teacher.config.text_config)
    state.update({"mtp." + k: v.to(torch.bfloat16).contiguous() for k, v in head.state_dict().items()
                  if "inv_freq" not in k})
    save_file(state, str(root / "tiny-student" / "model-00001.safetensors"))
    (root / "tiny-student" / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model-00001.safetensors" for k in state}}))
    for f in (root / "tiny-cut").iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, root / "tiny-student" / f.name)
    print(json.dumps({"tiny_student": str(root / "tiny-student"), "vocab": vocab, "tensors": len(state)}))


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def compare_snapshots(a, b):
    """Worst differences between two cache_snapshot results: kv (keys, values, indexer keys: max-abs over the
    reference's max-abs), conv (same), recurrent (relative L2); missing/extra entries and shape mismatches listed."""
    worst = {"kv": 0.0, "conv": 0.0, "recurrent": 0.0, "positions": 0.0}
    where, detail = {}, {}
    problems = []
    for part in ("main", "mtp"):
        x, y = a[part], b[part]
        if set(x) != set(y):
            problems.append({part: {"only_capture": sorted(set(x) - set(y))[:5], "only_extend": sorted(set(y) - set(x))[:5]}})
        for key in sorted(set(x) & set(y)):
            if x[key].shape != y[key].shape:
                problems.append({part: key, "shapes": [list(x[key].shape), list(y[key].shape)]})
                continue
            if key == "position_ids":
                kind, value = "positions", float((x[key] - y[key]).abs().max())
            elif ".recurrent_states." in key:
                kind, value = "recurrent", float((x[key] - y[key]).norm() / y[key].norm().clamp_min(1e-12))
            else:
                kind = "conv" if ".conv_states." in key else "kv"
                value = float((x[key] - y[key]).abs().max() / y[key].abs().max().clamp_min(1e-12))
            detail[f"{part}.{key}"] = value
            if value > worst[kind]:
                worst[kind], where[kind] = value, f"{part}.{key}"
    return worst, problems, where, detail


def run_capture_check(args):
    """HEAL_CAPTURE_CACHE bars (trainer MFU S1; ``torchrun --nproc-per-node 2``, gloo on CPU, NCCL on CUDA): the tiny
    student with an MTP head under FSDP, gradient checkpointing on (HEAL_RECOMPUTE), one dp_sequence slot per case.
    Ranks get sequences of 300 and 130 tokens (windows 96, extend chunks 64: 4 and 2 windows, rank 1 pads). Cases:
    off and off2 (the noise floor), on (both traced: per-window loss / KD / MTP KD and the prefix cache after each
    window), on-eq (both ranks on the 130-token sequence: rank 1 unpadded), base (HEAL_BASE_FILE: the pre-S1
    heal_kd.py, flag off). Forward calls per FSDP module are counted per rank (equal across ranks)."""
    import importlib.util
    import sys

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    import flash_qlora_job as fq

    cuda = torch.cuda.is_available()
    if not cuda:
        os.environ["HEAL_NO_FAST_ATTENTION"] = "1"
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda:
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out if rank == 0 else None
    fast_attention()
    modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
    teacher = tiny_teacher(args.config).to(device)  # on the card: the trainer's GDN / attention kernels are GPU-only
    keep = [0, 1, 2, 3, 5]
    experts = {layer: [0, 2, 4, 6] for layer in keep}
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(10, 2000, (1, length), generator=g) for length in (300, 130)]
    targets = [teacher_topk(teacher, s.to(device), k=args.k, window=128, vocab_chunk=64) for s in seqs]
    window, chunk = 96, (args.chunk if args.chunk < 96 else 64)  # --chunk < 96: extend chunks off the GDN chunk grid
    this = sys.modules[__name__]
    base = None
    if os.environ.get("HEAL_BASE_FILE"):
        spec = importlib.util.spec_from_file_location("heal_kd_base", os.environ["HEAL_BASE_FILE"])
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)

    def run(case, flag, plan, module=this, fp32=args.fp32):
        os.environ["HEAL_CAPTURE_CACHE"] = "1" if flag else "0"
        student = perturb(build_student(teacher, keep_layers=keep, experts=experts, top_k=2), 0.5, 7).float()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        set_recompute(student)
        student.config.use_cache = True
        head = mtp_tiny(student.config.text_config)
        student.to(device), head.to(device)
        runner = module.shard(module.Runner(student, head), init_device_mesh(device.type, (world,)),
                              compute_dtype=torch.float32 if fp32 else torch.bfloat16)
        runner.train()
        counts = {}
        for name, m in runner.named_modules():
            if m is runner or isinstance(m, modeling.Qwen4ExpTextDecoderLayer):
                m.register_forward_pre_hook(lambda *a, n=name: counts.__setitem__(n, counts.get(n, 0) + 1))
        i = plan[rank]
        ids = seqs[i].to(device)
        total = ids.shape[1]
        reply = torch.ones(total, device=device)
        reply[: total // 3] = 0
        data = {"ids": ids, "teacher": tuple(t.to(device) for t in targets[i]), "reply": reply}
        mine = {"data": data, "mask": torch.ones(total, device=device), "general": False,
                "piece": fq.positions_and_images(student, {"input_ids": ids})}
        dummy = dp_dummy(student, device, k=args.k)
        trace = [] if module is this else None
        n = sum(seqs[j].shape[1] for j in plan)
        began = time.time()
        row = module.dp_sequence(runner, mine, dummy, window=window, chunk=chunk, scale=world / n, mtp_weight=0.1,
                                 **({"trace": trace} if trace is not None else {}))
        seconds = time.time() - began
        grads = {}
        for name, p in runner.named_parameters():
            if p.grad is not None:
                grads[name] = (p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad).detach().float().cpu()
        every = [None] * world
        dist.all_gather_object(every, counts)
        event(log, case=case, rank0_row={k: row[k] for k in ("kd", "kd_mtp", "kd_reply")}, seconds=round(seconds, 2),
              collectives_equal=all(c == every[0] for c in every), forward_calls=sum(every[0].values()))
        return {"row": row, "trace": trace, "grads": grads, "counts_equal": all(c == every[0] for c in every)}

    def grad_compare(a, b):
        dot = sum(float((a[k] * b[k]).sum()) for k in b)
        na = math.sqrt(sum(float(a[k].pow(2).sum()) for k in a))
        nb = math.sqrt(sum(float(b[k].pow(2).sum()) for k in b))
        worst = max(float((a[k] - b[k]).abs().max()) for k in b)
        return {"cosine": dot / max(na * nb, 1e-30), "norm": na, "norm_ref": nb, "norm_rel": _rel(na, nb),
                "max_abs_diff": worst, "same_names": set(a) == set(b)}

    def window_compare(ta, tb):
        return [{"window": x["window"], "loss": _rel(x["loss"], y["loss"]), "kd": _rel(x["kd"], y["kd"]),
                 "kd_mtp": _rel(x["kd_mtp"], y["kd_mtp"]) if y["kd_mtp"] else 0.0} for x, y in zip(ta, tb)]

    unequal = [0, 1]
    off = run("off", False, unequal)
    off2 = run("off2", False, unequal)
    on = run("on", True, unequal)
    on_eq = run("on_eq", True, [1, 1])
    old = run("base", False, unequal, module=base) if base is not None else None
    # fp32 compute (CPU only: the trainer kernels exceed L40S shared memory in fp32): which side is nearer
    truth = run("truth", False, unequal, fp32=True) if not (args.fp32 or cuda) else None

    mine = {
        "bar1_windows_on_vs_off": window_compare(on["trace"], off["trace"]),
        "bar1_noise_off2_vs_off": window_compare(off2["trace"], off["trace"]),
        "bar3_cache": [dict(zip(("worst", "problems", "where"), compare_snapshots(x["cache"], y["cache"])[:3]),
                            window=x["window"]) for x, y in zip(on["trace"], off["trace"]) if "cache" in x and "cache" in y],
        # per entry, each side's distance to the fp32-compute cache (the worst capture-vs-extend entries first)
        "bar3_vs_fp32": None if truth is None else [
            {"window": x["window"], **{key: {"capture_vs_extend": d[key], "capture_vs_fp32": c[key], "extend_vs_fp32": e[key]}
                                       for key in sorted(d, key=d.get, reverse=True)[:4]},
             "worst_capture_vs_fp32": max(c.values()), "worst_extend_vs_fp32": max(e.values())}
            for x, y, t in zip(on["trace"], off["trace"], truth["trace"]) if "cache" in x and "cache" in t
            for d, c, e in [(compare_snapshots(x["cache"], y["cache"])[3], compare_snapshots(x["cache"], t["cache"])[3],
                             compare_snapshots(y["cache"], t["cache"])[3])]],
        "bar1_vs_fp32": None if truth is None else {"capture": window_compare(on["trace"], truth["trace"]),
                                                    "extend": window_compare(off["trace"], truth["trace"])},
        "bar3_windows_compared": sum(1 for x in on["trace"] if "cache" in x),
        "bar5_on_padded_vs_unpadded": window_compare(on["trace"], on_eq["trace"]) if rank == 1 else None,
        "bar5_counts_equal": [off["counts_equal"], on["counts_equal"], on_eq["counts_equal"]],
    }
    every = [None] * world
    dist.all_gather_object(every, mine)
    if rank == 0:
        result = {"per_rank": every,
                  "bar2_grad_on_vs_off": grad_compare(on["grads"], off["grads"]),
                  "bar2_noise_off2_vs_off": grad_compare(off2["grads"], off["grads"])}
        if old is not None:
            result["bar4_base_vs_off"] = {"rows_identical": all(old["row"][k] == off["row"][k] for k in old["row"]),
                                          "rows": [old["row"], off["row"]], **grad_compare(off["grads"], old["grads"])}
        b1 = max(max(w["loss"], w["kd"], w["kd_mtp"]) for r in every for w in r["bar1_windows_on_vs_off"])
        b3 = [c for r in every for c in r["bar3_cache"]]
        b5 = max(max(w["loss"], w["kd"], w["kd_mtp"]) for w in every[1]["bar5_on_padded_vs_unpadded"])
        g2 = result["bar2_grad_on_vs_off"]
        result["verdict"] = {
            "bar1_max_rel": b1, "bar1_ok": b1 <= 1e-3,
            "bar2_cosine": g2["cosine"], "bar2_norm_rel": g2["norm_rel"],
            "bar2_ok": g2["cosine"] >= 0.9995 and g2["norm_rel"] <= 1e-3,
            "bar3_kv": max(c["worst"]["kv"] for c in b3), "bar3_conv": max(c["worst"]["conv"] for c in b3),
            "bar3_recurrent": max(c["worst"]["recurrent"] for c in b3),
            "bar3_positions": max(c["worst"]["positions"] for c in b3),
            "bar3_ok": bool(b3) and all(not c["problems"] and c["worst"]["kv"] <= 1e-2 and c["worst"]["conv"] <= 1e-2
                                        and c["worst"]["recurrent"] <= 1e-2 and c["worst"]["positions"] == 0 for c in b3),
            "bar4_ok": None if old is None else (result["bar4_base_vs_off"]["rows_identical"]
                                                 and (cuda or result["bar4_base_vs_off"]["max_abs_diff"] == 0)),
            "bar5_max_rel": b5,
            "bar5_ok": b5 <= 1e-3 and all(all(r["bar5_counts_equal"]) for r in every)}
        event(log, check="capture_cache", **result)
    dist.destroy_process_group()


def run_inert_check(args):
    """HEAL_INERT_PREFIX bars (trainer MFU S1b, 23 Oct; ``torchrun --nproc-per-node 2``, gloo on CPU): the capture
    check's tiny student (MTP head, FSDP, recompute on), window 128, extend chunks 64, the loss mix with reverse and
    sampled terms. Scenario "aligned": both ranks live from row 129 (the loss starts on a window boundary, the
    same window calls per rank): S1 with HEAL_CAPTURE_CACHE=0 must equal the flag-off run bit for bit (an inert window adds exact
    zeros). Scenario "stream": rank 0 a g1l-like piece (live from row0 = window - 1, a loop hole in the reply,
    sampled rows a subset), rank 1 live from row 70; S1 against flag off, reported against the capture bars (rel
    <= 1e-3, cosine >= 0.9995, norm rel <= 1e-3; fp32: <= 1e-5): only the cut window's rows differ (their backprop
    into the rows before the prefix start). Per case: sums, grads, forward calls per FSDP module equal
    across ranks, tokens through layer 0 with and without grad."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    import flash_qlora_job as fq

    os.environ["HEAL_NO_FAST_ATTENTION"] = "1"
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out if rank == 0 else None
    modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
    teacher = tiny_teacher(args.config)
    keep = [0, 1, 2, 3, 5]
    experts = {layer: [0, 2, 4, 6] for layer in keep}
    window, chunk = 128, 64
    LOSS_MIX.update(forward=0.5, reverse=0.5, sampled=1.0)

    def sequence(length, first, hole=None, seed=1):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(10, 2000, (1, length), generator=g)
        top = teacher_topk(teacher, ids, k=args.k, window=128, vocab_chunk=64)
        reply = torch.zeros(length)
        reply[first:length - 1] = 1
        if hole:
            reply[hole[0]:hole[1]] = 0  # a loop-cut span
        labels = torch.full((length,), -100, dtype=torch.long)
        labels[:-1] = ids[0, 1:]
        t_tok = -torch.rand(length, generator=g) * 3
        b = t_tok + 0.3 * torch.randn(length, generator=g)
        smask = (reply > 0) & (torch.rand(length, generator=g) < 0.7)
        return ids, top, reply, (labels, t_tok, b, smask)

    scenarios = {"aligned": [(300, 129, None), (280, 129, None)],
                 "stream": [(300, window - 1, (150, 170)), (200, 70, None)],
                 "mixed": [(300, 129, None), (200, 1, None)]}  # rank 1 trains window 0: no skip there

    def run(case, seq, capture, inert, fp32, exact=False, skip=False):
        os.environ["HEAL_SKIP_INERT_LOSS"] = "1" if skip else "0"
        os.environ["HEAL_CAPTURE_CACHE"] = "1" if capture else "0"
        os.environ["HEAL_INERT_PREFIX"] = "1" if inert else "0"
        os.environ["HEAL_INERT_PREFIX_EXACT"] = "1" if exact else "0"
        student = perturb(build_student(teacher, keep_layers=keep, experts=experts, top_k=2), 0.5, 7).float()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        set_recompute(student)
        student.config.use_cache = True
        head = mtp_tiny(student.config.text_config)
        runner = shard(Runner(student, head), init_device_mesh("cpu", (world,)),
                       compute_dtype=torch.float32 if fp32 else torch.bfloat16)
        runner.train()
        counts, tokens = {}, {"grad": 0, "nograd": 0}
        first_layer = None
        for name, m in runner.named_modules():
            if m is runner or isinstance(m, modeling.Qwen4ExpTextDecoderLayer):
                m.register_forward_pre_hook(lambda *a, n=name: counts.__setitem__(n, counts.get(n, 0) + 1))
                if first_layer is None and m is not runner:
                    first_layer = m

        def count(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            tokens["grad" if torch.is_grad_enabled() else "nograd"] += int(x.shape[-2])

        first_layer.register_forward_pre_hook(count, with_kwargs=True)
        ids, top, reply, samp = seq[rank]
        data = {"ids": ids, "teacher": top, "reply": reply}
        mine = {"data": data, "mask": reply, "general": False, "samp": samp,
                "piece": fq.positions_and_images(student, {"input_ids": ids})}
        mine["prefix"] = prefix_start(reply.bool() | samp[3])
        n = sum(float(x[2].sum()) for x in seq)
        dummy, on_trace = dp_dummy(student, device, k=args.k), []
        row = dp_sequence(runner, mine, dummy, window=window, chunk=chunk, scale=world / n, mtp_weight=0.1,
                          trace=on_trace, trace_cache=False)
        grads = {}
        for name, p in runner.named_parameters():
            if p.grad is not None:
                grads[name] = (p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad).detach().float()
        # HEAL_INERT_CHECK's pass (no grad, S1 off) against this training pass's sums
        on = {"kd": float(row["kd"]), "kd_reply": float(row["kd_reply"]), "loss": sum(t["loss"] for t in on_trace)}
        check = inert_check_event(on, inert_check_pass(runner, mine, dummy, window=window, chunk=chunk, mtp_weight=0.1,
                                                       lazy=False), device)
        every = [None] * world
        dist.all_gather_object(every, (counts, tokens, mine["prefix"]))
        sums = {k: float(row[k]) for k in ("kd", "kd_reply", "kd_mtp", "n_mtp")}
        allsums = [None] * world
        dist.all_gather_object(allsums, sums)
        event(log, case=case, prefix=[e[2] for e in every], tokens=[e[1] for e in every],
              forward_calls=[sum(e[0].values()) for e in every])
        return {"sums": allsums, "grads": grads, "counts_equal": all(e[0] == every[0][0] for e in every),
                "calls": sum(every[0][0].values()),
                "tokens": [e[1] for e in every], "inert_check": check}

    def compare(a, b):
        dot = sum(float((a["grads"][k].double() * b["grads"][k].double()).sum()) for k in b["grads"])
        na = math.sqrt(sum(float(a["grads"][k].double().pow(2).sum()) for k in a["grads"]))
        nb = math.sqrt(sum(float(b["grads"][k].double().pow(2).sum()) for k in b["grads"]))
        sums = max(_rel(x[k], y[k]) for x, y in zip(a["sums"], b["sums"]) for k in y)
        return {"bit_equal": set(a["grads"]) == set(b["grads"]) and all(torch.equal(a["grads"][k], b["grads"][k])
                                                                        for k in b["grads"]) and a["sums"] == b["sums"],
                "max_abs_diff": max(float((a["grads"][k] - b["grads"][k]).abs().max()) for k in b["grads"]),
                "cosine": dot / max(na * nb, 1e-30), "norm_rel": _rel(na, nb), "sums_max_rel": sums}

    def pairing(order, capture):
        """MFU-1 lever 1 (HEAL_STREAM_BALANCE re-pairs a step's units across slots): the same four sequences in two
        slots, paired as ``order`` [[slot0 rank0, slot0 rank1], [slot1 ...]]; the step's gradient sums over them."""
        os.environ["HEAL_CAPTURE_CACHE"] = "1" if capture else "0"
        os.environ["HEAL_INERT_PREFIX"] = "1"
        os.environ["HEAL_INERT_PREFIX_EXACT"] = "0"
        student = perturb(build_student(teacher, keep_layers=keep, experts=experts, top_k=2), 0.5, 7).float()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        set_recompute(student)
        student.config.use_cache = True
        head = mtp_tiny(student.config.text_config)
        runner = shard(Runner(student, head), init_device_mesh("cpu", (world,)), compute_dtype=torch.float32)
        runner.train()
        n = sum(float(x[2].sum()) for x in four)
        sums = {"kd": 0.0, "kd_reply": 0.0}
        for slot in order:
            ids, top, reply, samp = four[slot[rank]]
            mine = {"data": {"ids": ids, "teacher": top, "reply": reply}, "mask": reply, "general": False, "samp": samp,
                    "piece": fq.positions_and_images(student, {"input_ids": ids})}
            mine["prefix"] = prefix_start(reply.bool() | samp[3])
            row = dp_sequence(runner, mine, dp_dummy(student, device, k=args.k), window=window, chunk=chunk,
                              scale=world / n, mtp_weight=0.1)
            box = torch.tensor([float(row["kd"]), float(row["kd_reply"])], dtype=torch.float64)
            dist.all_reduce(box)
            sums["kd"] += float(box[0])
            sums["kd_reply"] += float(box[1])
        grads = {}
        for name, p in runner.named_parameters():
            if p.grad is not None:
                grads[name] = (p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad).detach().double()
        return grads, sums

    four = [sequence(300, 129, None, seed=40), sequence(140, 70, None, seed=41), sequence(260, 127, (150, 170), seed=42),
            sequence(200, 64, None, seed=43)]
    pair = {}
    for capture in (False, True):
        a_g, a_s = pairing([[0, 1], [2, 3]], capture)  # balance off: pairs by length / arrival
        b_g, b_s = pairing([[0, 3], [2, 1]], capture)  # balance on: the same four, re-paired (rank order changed too)
        diff = math.sqrt(sum(float((a_g[k] - b_g[k]).pow(2).sum()) for k in a_g))
        norm = math.sqrt(sum(float(a_g[k].pow(2).sum()) for k in a_g))
        pair["capture" if capture else "nocap"] = {
            "grad_rel": diff / max(norm, 1e-30), "same_params": set(a_g) == set(b_g),
            "sums_rel": max(abs(a_s[k] - b_s[k]) / max(abs(a_s[k]), 1e-30) for k in a_s)}
    result = {"pairing": pair}
    for name, specs in scenarios.items():
        seq = [sequence(*spec, seed=10 + i) for i, spec in enumerate(specs)]
        for fp32 in (False, True):
            tag = f"{name}-{'fp32' if fp32 else 'bf16'}"
            off = run(f"{tag}-off", seq, False, False, fp32)
            off2 = run(f"{tag}-off2", seq, False, False, fp32)
            cap = run(f"{tag}-cap", seq, True, False, fp32)
            s1off = run(f"{tag}-s1-nocap", seq, False, True, fp32)
            s1 = run(f"{tag}-s1", seq, True, True, fp32)
            s1x = run(f"{tag}-s1-exact-nocap", seq, False, True, fp32, exact=True)
            sk = run(f"{tag}-s1-skip", seq, True, True, fp32, skip=True)
            skn = run(f"{tag}-s1-skip-nocap", seq, False, True, fp32, skip=True)
            os.environ["HEAL_SKIP_INERT_LOSS"] = "0"
            result[tag] = {"skip_vs_s1": compare(sk, s1), "skip_nocap_vs_s1nocap": compare(skn, s1off),
                           "forward_calls_skip_vs_s1": [sk["calls"], s1["calls"]],"s1exact_nocap_vs_off": compare(s1x, off), "tokens_s1exact": s1x["tokens"],"noise_off2_vs_off": compare(off2, off), "s1nocap_vs_off": compare(s1off, off),
                           "s1_vs_cap": compare(s1, cap), "s1_vs_off": compare(s1, off), "cap_vs_off": compare(cap, off),
                           "counts_equal": all(x["counts_equal"] for x in (off, cap, s1off, s1)),
                           "tokens_off": off["tokens"], "tokens_s1": s1["tokens"],
                           "inert_check": {"s1_nocap": s1off["inert_check"], "s1": s1["inert_check"],
                                           "off": off["inert_check"]}}
    if rank == 0:
        bar = lambda c, fp32: (c["cosine"] >= 0.9995 and c["norm_rel"] <= 1e-3 and c["sums_max_rel"] <= 1e-3  # noqa
                               and (not fp32 or (c["norm_rel"] <= 1e-5 and c["sums_max_rel"] <= 1e-5)))
        verdict = {
            "aligned_bit_equal": all(result[f"aligned-{d}"]["s1nocap_vs_off"]["bit_equal"] for d in ("bf16", "fp32")),
            "exact_mode_bit_equal": all(result[t]["s1exact_nocap_vs_off"]["bit_equal"] for t in result if t != "pairing"),
            # HEAL_SKIP_INERT_LOSS: skipping the all-inert padding loss calls changes no gradient bit
            "skip_bit_equal": all(result[t][k]["bit_equal"] for t in result if t != "pairing"
                                  for k in ("skip_vs_s1", "skip_nocap_vs_s1nocap")),
            "counts_equal": all(result[t]["counts_equal"] for t in result if t != "pairing"),
            # lever 1: the same picks paired differently across slots train the same step (fp32 rel <= 1e-6)
            "pairing_grads_equal": all(v["same_params"] and v["grad_rel"] <= 1e-6 and v["sums_rel"] <= 1e-6
                                       for v in result["pairing"].values()),
            # HEAL_INERT_CHECK: on the aligned case the no-grad S1-off pass equals the S1 training pass (rel 0) without
            # the capture cache; with it, captured vs extended prefixes differ by fp32 rounding (knobs-cpu-v1: 7.9e-7)
            "inert_check_aligned_zero": all(max(result[f"aligned-{d}"]["inert_check"][k]["rel"].values()) == 0.0
                                            for d in ("bf16", "fp32") for k in ("s1_nocap", "off")),
            "inert_check_capture_1e5": all(max(result[t]["inert_check"]["s1"]["rel"].values()) <= 1e-5 for t in result
                                          if t != "pairing")}
        verdict["ok"] = all(verdict.values())
        # informational: the cut window's rows lose backprop into its rows before the prefix (not a numeric bar)
        verdict["stream_within_bars"] = all(bar(result[t][k], t.endswith("fp32")) for t in result if t.startswith("stream")
                                            for k in ("s1nocap_vs_off", "s1_vs_cap", "s1_vs_off"))
        event(log, check="inert_prefix", verdict=verdict, **result)
    dist.destroy_process_group()


def run_recompute_mem(args):
    """S1b read (one CUDA card): activation memory per 1K window tokens per HEAL_RECOMPUTE policy, with and without
    HEAL_CAPTURE_CACHE, on real-width students of ``--depths`` layers (Flash's layer pattern and PLE layer; experts cut
    to ``--mem-experts``, top-10, a small n-gram table and vision tower; BF16 weights, as FSDP computes). Per case the
    peak allocated over one loss pass + backward above the allocation before it (gradients already allocated, the
    prefix cache too), at each of ``--mem-windows`` window lengths, from an empty prefix and from ``--mem-prefix``
    tokens; then train_sequence over three windows timed, extend vs capture, per policy."""
    import gc

    from transformers import AutoConfig, AutoModelForMultimodalLM
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fast_attention()
    modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
    rows = []
    for depth in args.depths:
        data = json.loads(Path(args.config).read_text())
        text = data["text_config"]
        text.update(num_hidden_layers=depth, layer_types=text["layer_types"][:depth], num_experts=args.mem_experts,
                    ngram_vocab_size_base=5000)
        data["vision_config"].update(depth=1)
        data.pop("quantization_config", None)
        config = AutoConfig.for_model(**data)
        torch.manual_seed(0)
        with torch.device(device):
            model = AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)
            head = MtpHead(model.config.text_config, experts=args.mem_experts, top_k=10).to(device, torch.bfloat16)
        model.config._experts_implementation = os.environ.get("HEAL_EXPERTS", "grouped_mm")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        runner = Runner(model, head).train()
        vocab = model.config.text_config.vocab_size
        longest = max(args.mem_windows) * 3 + args.mem_prefix
        ids = torch.randint(10, vocab - 10, (1, longest), device=device)
        top = torch.log_softmax(torch.randn(longest, args.k, device=device), -1).half() - 1.0
        teacher = (top, torch.randint(0, vocab, (longest, args.k), device=device, dtype=torch.int32),
                   torch.full((longest,), -1.0, device=device, dtype=torch.float16))
        held = Held(top=teacher[0], idx=teacher[1], tail=teacher[2], mask=torch.ones(longest, device=device),
                    labels=None, lm_weight=0.0, reply=None, mtp_weight=0.1)
        weights = sum(p.numel() * p.element_size() for p in runner.parameters())
        runner("loss", ids, 0, 256, None, held=held, mtp_cache=head.new_cache())[0].backward()  # allocate .grad

        def peak(t, prefix, capture):
            mc = head.new_cache()  # noqa: F821 - bound before the call (ruff sees the later del / the loop's first pass)
            cache = runner_extend(runner, ids, 0, prefix, None, 4096, mtp_cache=mc) if prefix else None  # noqa: F821 - bound before the call (ruff sees the later del / the loop's first pass)
            gc.collect()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            loss = runner("loss", ids, prefix, prefix + t, cache, held=held, mtp_cache=mc, capture=capture)[0]  # noqa: F821 - bound before the call (ruff sees the later del / the loop's first pass)
            loss.backward()
            del loss
            torch.cuda.synchronize()
            got = torch.cuda.max_memory_allocated() - before
            runner.captured = None  # noqa: F821 - bound before the call (ruff sees the later del / the loop's first pass)
            del cache
            gc.collect()
            torch.cuda.empty_cache()
            return got / 2 ** 30

        for policy in ("full", "alternate", "none"):
            os.environ["HEAL_RECOMPUTE"] = policy
            set_recompute(model, policy)
            for capture in (False, True):
                try:
                    peaks = {t: peak(t, 0, capture) for t in args.mem_windows}
                    with_prefix = peak(args.mem_windows[0], args.mem_prefix, capture)
                except torch.cuda.OutOfMemoryError as error:
                    rows.append(event(out, depth=depth, policy=policy, capture=capture, oom=repr(error)[:200]))
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                ts = sorted(peaks)
                slope = (peaks[ts[-1]] - peaks[ts[0]]) / ((ts[-1] - ts[0]) / 1024)
                prefix_slope = (with_prefix - peaks[ts[0]]) / (args.mem_prefix / 1024)
                rows.append(event(out, depth=depth, policy=policy, capture=capture, peaks_gib={str(k): round(v, 3) for k, v in peaks.items()},
                                  gib_per_1k_window=round(slope, 4), gib_per_1k_prefix=round(prefix_slope, 4),
                                  weights_gib=round(weights / 2 ** 30, 2)))
            w = args.mem_windows[-1]
            for capture in (False, True):  # three windows: two prefix extensions (or captures), chunks of w / 2
                os.environ["HEAL_CAPTURE_CACHE"] = "1" if capture else "0"
                times = []
                try:
                    for _ in range(3):
                        torch.cuda.synchronize()
                        began = time.time()
                        train_sequence(runner, ids[:, : 3 * w], teacher, window=w, chunk=w // 2, scale=1e-4,
                                       mtp_weight=0.1)
                        torch.cuda.synchronize()
                        times.append(time.time() - began)
                except torch.cuda.OutOfMemoryError as error:
                    rows.append(event(out, depth=depth, policy=policy, capture=capture, timed_oom=repr(error)[:200]))
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                finally:
                    os.environ["HEAL_CAPTURE_CACHE"] = "0"
                rows.append(event(out, depth=depth, policy=policy, capture=capture, timed_tokens=3 * w,
                                  seconds=[round(x, 3) for x in times], best=round(min(times[1:]), 3)))
        del runner, model, head
        gc.collect()
        torch.cuda.empty_cache()
    event(out, recompute_mem="done", rows=len(rows))


def tiny_device():
    """train-tiny's teacher targets on the card when there is one (the trainer image's GDN kernels are GPU-only)."""
    return f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"


def run_train_tiny(args):
    """``run_train`` end to end on a tiny model on CPU (``torchrun --nproc-per-node 2``, gloo; PTH 18 Oct, after
    check 1 failed at step 1): phase 1 builds the tiny teacher, the cut student on disk, op_topk-format teacher
    files with reply spans and general files, then trains 2 steps (ESFT selection, reverse + forward KL on reply
    positions, the general mix, the MTP head if the checkpoint has one, clip, AdamW + RowAdam, zero_grad) and saves;
    phase 2 resumes from phase 1's student and optimizer state for 2 more steps and saves again."""
    import torch.distributed as dist

    os.environ.setdefault("HEAL_NO_FAST_ATTENTION", "1")  # =0 with RL1_ATTENTION=selected: rl1's path on CPU (S4)
    os.environ.setdefault("HEAL_EXPERTS", "eager")
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")  # CUDA: the comm check on 2 cards
    rank = dist.get_rank()
    root = Path(args.out)
    tag = args.tag
    if args.tiny_phase == 1 and rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        teacher = tiny_teacher(args.config)
        teacher.save_pretrained(str(root / "tiny-teacher"), max_shard_size="2MB")
        (root / "prune.json").write_text(json.dumps({"format": "arc-expert-prune-v1", "experts": 8,
                                                     "pruned": {str(i): [1, 3, 5, 7] for i in range(6)}}))
        cut_checkpoint(root / "tiny-teacher", root / "tiny-cut", drop=[4], prune_file=root / "prune.json", top_k=2,
                       shard_gib=0.002)
        # the real student's on-disk layout (fused expert tensors under the model's own names, as cut from Flash):
        # save_pretrained here would write per-expert tensors
        import shutil

        from safetensors.torch import save_file
        from transformers import AutoModelForMultimodalLM

        cut = AutoModelForMultimodalLM.from_pretrained(str(root / "tiny-cut"), dtype=torch.bfloat16)
        (root / "tiny-student").mkdir()
        state = {k: v.contiguous() for k, v in cut.state_dict().items()}
        head = mtp_tiny(teacher.config.text_config)  # an uncut MTP block on disk as in the real student (mtp.* names)
        state.update({"mtp." + k: v.to(torch.bfloat16).contiguous() for k, v in head.state_dict().items()
                      if "inv_freq" not in k})
        save_file(state, str(root / "tiny-student" / "model-00001.safetensors"))
        (root / "tiny-student" / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": {k: "model-00001.safetensors" for k in state}}))
        for f in (root / "tiny-cut").iterdir():
            if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
                shutil.copy(f, root / "tiny-student" / f.name)
        teacher = perturb(teacher, 1.0, 11).to(tiny_device())  # targets measurably off the student
        g = torch.Generator().manual_seed(1)
        for folder, count, reply_on in (("teacher", 6, True), ("general", 20, False)):
            (root / folder).mkdir(exist_ok=True)
            lengths = [256, 130, 300, 97, 200, 64] if reply_on else [256] * count  # unequal window plans per rank
            for i in range(count):
                L = lengths[i]
                ids = torch.randint(10, 2000, (1, L), generator=g)
                top, idx, tail = (t.cpu() for t in teacher_topk(teacher, ids.to(tiny_device()), k=args.k, window=128,
                                                                vocab_chunk=64))
                reply = torch.zeros(L, dtype=torch.bool)
                if reply_on:
                    reply[min(120, L // 2):L - 6] = True
                torch.save({"ids": ids[0].int(), "top_ids": idx, "top_logp": top, "tail": tail, "reply": reply,
                            "image": torch.zeros(L, dtype=torch.bool), "segment": f"{folder}-{i}", "end": 0, "w": 1.0},
                           root / folder / f"{folder}-{i:03d}.pt")
    dist.barrier()
    index = json.loads((root / "tiny-student" / "model.safetensors.index.json").read_text())["weight_map"]
    has_mtp = any(k.startswith("mtp.") for k in index)
    args.teacher, args.general, args.frozen_root = str(root / "teacher"), str(root / "general"), str(root / "tiny-student")
    args.window, args.chunk, args.total_steps, args.batch_tokens = 96, 64, 4, 200
    args.mtp_weight = 0.1 if has_mtp else 0.0
    args.mtp_top_k = 2
    args.route_files = 2
    args.seconds = 1e9
    if args.tiny_phase == 3:  # data-parallel equality: one step over all six files, gradients dumped
        world = dist.get_world_size()
        args.student, args.out, args.step0, args.resume = (str(root / "tiny-student"), str(root / f"dp{world}{tag}"), 0,
                                                           None)
        args.general, args.flush_last, args.batch_tokens, args.total_steps = None, True, 10 ** 9, 1
        args.routing = str(root / "round1" / "routing.pt")
        os.environ["HEAL_DUMP_GRADS"] = str(root / f"grads-w{world}{tag}.pt")
        run_train(args)
        return
    if args.tiny_phase == 7:  # OPD stream (22 Oct): chunks arrive while training; per-step weight deltas
        import threading

        stream, deltas = root / f"stream{tag}", root / f"deltas{tag}"
        if rank == 0:
            stream.mkdir(exist_ok=True)

            def feed():  # round-1 teacher files as streamed chunks: teacher rows from the reply's start, engine b
                files = sorted((root / "teacher").glob("*.pt")) * 3
                for i, f in enumerate(files):
                    d = torch.load(f, weights_only=True)
                    on = d["reply"].nonzero()
                    row0 = max(0, int(on[0]) - 1) if len(on) else 0
                    n = d["ids"].numel()
                    if os.environ.get("HEAL_TINY_PIECES") == "1" and n - row0 > 8:  # heal_stream's pieces: a
                        # chain fed in 3 parts, each piece's ids a prefix of the next, row0 = the last piece's last row
                        cuts = [row0 + (n - row0) // 3, row0 + 2 * (n - row0) // 3, n]
                        lo = 0
                        for k, upto in enumerate(cuts):
                            r0 = row0 if k == 0 else lo - 1
                            tok, beh = torch.full((upto - r0,), -2.0), torch.full((upto - r0,), -1.0)
                            tok[-1] = beh[-1] = float("nan")
                            name = f"s-c{i:03d}--c001-p{k:04d}"
                            torch.save({"ids": d["ids"][:upto], "image": d["image"][:upto], "reply": d["reply"][:upto],
                                        "row0": r0, "top_logp": d["top_logp"][r0:upto], "top_ids": d["top_ids"][r0:upto],
                                        "tail": d["tail"][r0:upto], "tok_logp": tok, "beh_logp": beh, "version": 0},
                                       stream / f"{name}.pt")
                            (stream / f"{name}.ready").write_text(json.dumps(
                                {"version": 0, "tokens": upto, "rows": upto - r0,
                                 "reply_rows": int(d["reply"][r0 + 1:upto].sum())}))
                            lo = upto
                        time.sleep(0.5)
                        continue
                    chunk = {"ids": d["ids"], "image": d["image"], "reply": d["reply"], "row0": row0,
                             "top_logp": d["top_logp"][row0:], "top_ids": d["top_ids"][row0:], "tail": d["tail"][row0:],
                             "tok_logp": torch.full((n - row0,), -2.0), "beh_logp": torch.full((n - row0,), -1.0),
                             "version": 0}
                    torch.save(chunk, stream / f"s-c{i:03d}.pt")
                    (stream / f"s-c{i:03d}.ready").write_text(json.dumps({"version": chunk["version"]}))
                    time.sleep(0.5)
            threading.Thread(target=feed, daemon=True).start()
        os.environ.update(HEAL_STREAM=str(stream), HEAL_DELTA_DIR=str(deltas), HEAL_STREAM_K=str(args.k),
                          HEAL_SAMPLED="1.0", HEAL_STREAM_STALL="600", HEAL_OPD_LOG="1")
        args.student, args.out, args.step0, args.resume, args.skip = (str(root / "tiny-student"), str(root / f"st{tag}"), 0,
                                                                      None, None)
        args.routing, args.total_steps, args.checkpoint_seconds = str(root / "round1" / "routing.pt"), 3, 0
        run_train(args)
        return
    if args.tiny_phase == 6:  # the comm check (MFU S3): a round of several steps, most of several slots of 3-7
        # windows (window 48, extend chunks 32), own and general slots, unequal lengths per slot (a padding rank),
        # the flush step at the end; per-step gradients and parameters dumped (HEAL_DUMP_STEPS); the HEAL_* comm
        # flags come from the environment
        world = dist.get_world_size()
        folder = root / "comm-teacher"
        if rank == 0 and not (folder / "done").exists():
            folder.mkdir(exist_ok=True)
            teacher = perturb(tiny_teacher(args.config), 1.0, 11).to(tiny_device())
            g = torch.Generator().manual_seed(2)
            for i, L in enumerate([200, 150, 330, 97, 260, 64, 180, 240, 120, 300, 90, 210]):
                ids = torch.randint(10, 2000, (1, L), generator=g)
                top, idx, tail = (t.cpu() for t in teacher_topk(teacher, ids.to(tiny_device()), k=args.k, window=128,
                                                                vocab_chunk=64))
                reply = torch.zeros(L, dtype=torch.bool)
                reply[min(120, L // 2):L - 6] = True
                torch.save({"ids": ids[0].int(), "top_ids": idx, "top_logp": top, "tail": tail, "reply": reply,
                            "image": torch.zeros(L, dtype=torch.bool), "segment": f"comm-{i}", "end": 0, "w": 1.0},
                           folder / f"comm-{i:03d}.pt")
            (folder / "done").touch()
        dist.barrier()
        args.student, args.out, args.step0, args.resume = (str(root / "tiny-student"), str(root / f"comm{world}{tag}"),
                                                           0, None)
        args.teacher, args.routing = str(folder), str(root / "round1" / "routing.pt")
        args.window, args.chunk, args.batch_tokens = 48, 32, 300
        args.total_steps, args.allow_unfinished_schedule, args.flush_last = 100, True, True
        os.environ.setdefault("HEAL_DUMP_STEPS", str(root / f"steps{world}{tag}"))
        run_train(args)
        return
    if args.tiny_phase in (4, 5):  # exact mid-round resume: A runs 3 steps with a checkpoint after step 2 (4); B, a
        # fresh process, resumes ckpt-2 and redoes step 3 (5); modal_heal_tiny compares their step-3 kd and grad norm
        args.batch_tokens, args.total_steps, args.checkpoint_seconds, args.checkpoint_at = 100, 3, 0, {2}
        args.routing = str(root / "round1" / "routing.pt")
        ck = root / f"ckA{tag}" / "ckpt-2"
        if args.tiny_phase == 4:
            args.student, args.out, args.step0, args.resume, args.skip = (str(root / "tiny-student"),
                                                                          str(root / f"ckA{tag}"), 0, None, None)
        else:
            args.student, args.out, args.step0, args.resume, args.skip = (str(ck / "student") if (ck / "student").is_dir()
                                                                          else str(root / "tiny-student"), str(root / f"ckB{tag}"),
                                                                          2, str(ck), str(ck / "progress.json"))
            args.no_save = os.environ.get("HEAL_TINY_SAVE") != "1"  # the reshard check compares B's saved student
        run_train(args)
        return
    if args.tiny_phase == 1:
        args.student, args.out, args.step0, args.resume = str(root / "tiny-student"), str(root / "round1"), 0, None
    else:
        args.student, args.out, args.step0, args.resume = (str(root / "round1" / "student"), str(root / "round2"), 2,
                                                           str(root / "round1"))
        args.routing = str(root / "round1" / "routing.pt")
    if args.tiny_phase == 1:
        args.total_steps = 2  # phase 1 stops at step 2 of the 4-step schedule ...
        run_train(args)
    else:
        run_train(args)  # ... phase 2 continues to step 4


# ---- one-card proxy of run_train's step (trainer MFU S2) -------------------------------------------------------------

def proxy_config(args):
    """The student's config (``--proxy-config``: a real student's config.json; else flash_config.json) cut to its first
    ``--proxy-layers`` decoder layers (Flash's GDN x3 + attention pattern from layer 0), ``--proxy-experts`` routed
    experts (0: the config's), a small n-gram table (host RAM; the per-token lookup is the same), vision depth 1."""
    data = json.loads(Path(args.proxy_config or args.config).read_text())
    text = data["text_config"]
    n = args.proxy_layers
    text["mtp_num_experts"] = args.proxy_mtp_experts or text.get("mtp_num_experts") or text["num_experts"]
    text.update(num_hidden_layers=n, layer_types=text["layer_types"][:n], ngram_vocab_size_base=args.proxy_ngram)
    text["ple_layer_ids"] = [i for i in text.get("ple_layer_ids") or [] if i - 1 < n]
    if args.proxy_experts:
        text["num_experts"] = args.proxy_experts
    data["vision_config"].update(depth=1)
    for key in ("quantization_config", "student_cut"):
        data.pop(key, None)
    return data


def proxy_weights(model, head, folder, experts, mtp_experts):
    """The real student's tensors for the proxy's layers (experts and router rows sliced to the proxy's count), its
    embeddings, final mixer, LM head and MTP block; the n-gram table and the vision tower stay random."""
    from safetensors import safe_open

    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    own = dict(model.state_dict())
    side = {"mtp." + k: v for k, v in head.state_dict().items()} if head is not None else {}
    loaded, by_file = [], {}
    for name, file in index.items():
        if ".ple_embedding." in name or name.startswith("model.visual"):  # the table and its hash buffers stay small
            continue
        if name in own or name in side:
            by_file.setdefault(file, []).append(name)
    for file, names in sorted(by_file.items()):
        with safe_open(str(folder / file), framework="pt") as f:
            for name in names:
                t = f.get_tensor(name)
                cap = mtp_experts if name.startswith("mtp.") else experts
                if (EXPERTS.match(name) or ROUTER.match(name) or re.search(r"^mtp\.layers\.0\.mlp\.(experts\.|gate\.)", name)) \
                        and t.shape[0] > cap:
                    t = t[:cap]
                dst = own[name] if name in own else side[name]
                with torch.no_grad():
                    dst.copy_(t.to(dst.dtype))
                loaded.append(name)
    return {"loaded": len(loaded), "model_tensors": len(own), "mtp_tensors": len(side)}


def proxy_flops(text, windows, mtp, extends=()):
    """Model FLOPs of one step (6 x active matmul parameters per trained token, the selected attention and the GDN
    chunk recurrence; recompute not counted): ``windows`` [(prefix, length)] of the loss passes; ``extends``
    [(prefix, length)] of no-grad prefix passes (forward only: a third of a loss pass's FLOPs)."""
    H, hc, r, V = text["hidden_size"], text["hc_count"], text["hc_lowrank"], text["vocab_size"]
    nk, dk, nv, dv = (text["linear_num_key_heads"], text["linear_key_head_dim"], text["linear_num_value_heads"],
                      text["linear_value_head_dim"])
    heads, kvh, hd = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]
    ih, idim = text["indexer_n_heads"], text["indexer_head_dim"]
    budget, ratio = text["indexer_budget"], text["indexer_compress_ratio"]
    inter, shared, top = text["moe_intermediate_size"], text["shared_expert_intermediate_size"], text["num_experts_per_tok"]
    hc_block = hc * H * r * 2 + hc * H * hc
    moe = lambda experts, k: k * 3 * H * inter + 3 * H * shared + H + H * experts  # noqa: E731
    gdn = H * (2 * nk * dk + nv * dv) + H * nv * dv + 2 * H * nv + nv * dv * H
    attn = H * heads * hd * 2 + 2 * H * kvh * hd + heads * hd * H + H * (ih * idim + idim)
    layers = text["layer_types"]
    per_token = sum((gdn if t == "linear_attention" else attn) + moe(text["num_experts"], top) + 2 * hc_block
                    for t in layers) + hc * H * r * 2 + H * V
    ple = text.get("ple_embed_dim", 0) * (hc * H + H) * len(text.get("ple_layer_ids") or [])
    per_token += ple
    mtp_token = (2 * H * H + attn + moe(text.get("mtp_num_experts", text["num_experts"]), top) + 2 * hc_block
                 + hc * H * r * 2 + H * V) if mtp else 0
    n_attn = sum(t != "linear_attention" for t in layers)
    n_gdn = len(layers) - n_attn
    total = 0.0
    for (prefix, length), share in [(w, 1.0) for w in windows] + [(w, 1.0 / 3.0) for w in extends]:
        dense = 6.0 * share * (per_token * length + mtp_token * max(length - 1, 0))
        att = 0.0
        for p in (prefix, prefix + length // 2, prefix + length):  # Simpson over the window
            s = min(p + 1, budget + ratio - 1)
            att += (12.0 * heads * hd * s + 2.0 * ih * idim * p / ratio) * (1 if p != prefix + length // 2 else 4)
        att = share * att / 6.0 * length * (n_attn + (1 if mtp else 0))
        rec = share * 3.0 * nv * (4 * dk * dv + 2 * 64 * (dk + dv)) * length * n_gdn
        total += dense + att + rec
    return total


def peak_flops(name, override=0.0):
    if override:
        return override
    for key, value in (("B300", 2.25e15), ("B200", 2.25e15), ("H200", 989e12), ("H100", 989e12), ("L40S", 362e12),
                       ("L4", 121e12), ("A100", 312e12)):
        if key in name:
            return value
    return 0.0


def run_proxy(args):
    """One-card proxy of ``run_train``'s optimizer step (trainer MFU S2; ``torchrun --nproc-per-node 1``): the student
    cut to ``--proxy-layers`` real-width layers (``proxy_config``; ``--proxy-weights``: the real student's tensors, else
    random init), the MTP head, FSDP2 on a one-card mesh (fp32 masters, BF16 compute: the trainer's casts, no
    collectives), checkpointing and the capture cache as the trainer's env sets them; synthetic teacher top-k windows
    over ``--proxy-seqs`` sequence lengths (reply mask on ~60% of positions); per step dp_sequence per sequence, then
    the trainer's mask / clip / AdamW + RowAdam. Per case of ``--proxy-cases`` (HEAL_COMPILE modes; same seed and
    data, a fresh model each): ``--proxy-steps`` steps with per-window losses of step 1 and step losses, the first
    step's gradients against case 0's (cosine, norm), then ``--proxy-timed`` timed steps (step time, tokens/s, MFU on
    active FLOPs) and one profiled step (``profile_summary``: class split, kernels per step). ``--proxy-session``:
    forward+backward over that many random-length sequences after the last case (recompiles over varied windows)."""
    import gc

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig, AutoModelForMultimodalLM
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    import flash_qlora_job as fq

    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fast_attention()
    modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
    data_cfg = proxy_config(args)
    text = data_cfg["text_config"]
    impl = os.environ.get("HEAL_EXPERTS", "grouped_mm")
    # --proxy-pieces "total:prefix:reply,...": OPD stream pieces (rows before prefix: the no-grad inert prefix,
    # HEAL_INERT_PREFIX's slot plan; reply share of the rows after it); else --proxy-seqs (loss from row 0, reply 0.6)
    pieces = [tuple(float(v) for v in x.split(":")) for x in args.proxy_pieces.split(",")] if args.proxy_pieces else \
        [(float(x), 0.0, 0.6) for x in args.proxy_seqs.split(",")]
    lengths = [int(p[0]) for p in pieces]
    vocab, k = text["vocab_size"], args.k
    g = torch.Generator().manual_seed(args.seed + 1)
    seqs = []
    for total, pre, share in pieces:
        length, pre = int(total), int(pre) // 64 * 64
        ids = torch.randint(10, vocab - 10, (1, length), generator=g)
        top = torch.log_softmax(torch.randn(length, k, generator=g) * 2, -1) - 0.5
        idx = torch.randint(0, vocab, (length, k), generator=g, dtype=torch.int32)
        tail = torch.log1p(-top.double().exp().sum(-1).clamp(max=1 - 1e-6)).half()
        reply = (torch.rand(length, generator=g) < share).float()
        reply[:pre + 1] = 0.0
        seqs.append({"ids": ids.to(device), "teacher": (top.half().to(device), idx.to(device), tail.to(device)),
                     "reply": reply.to(device), "ids_cpu": ids[0], "reply_n": float(reply.sum()), "prefix": pre})
    windows, extends = [], []
    for s in seqs:
        spans, inert = slot_spans({"data": s, "prefix": s["prefix"]}, args.window)
        extends += [(0, b - a) for a, b in spans[:inert]]
        windows += [(a, b - a) for a, b in spans[inert:]]
    flops = proxy_flops(text, windows, args.mtp_weight > 0, extends)
    peak = peak_flops(torch.cuda.get_device_name(device), args.proxy_peak)
    positions = sum(float(s["reply"].sum()) for s in seqs)
    event(out, event="proxy_setup", layers=text["layer_types"], experts=text["num_experts"],
          mtp_experts=text["mtp_num_experts"], seqs=lengths, window=args.window, windows=len(windows),
          prefix_tokens=sum(n for _, n in extends), loss_tokens=sum(n for _, n in windows),
          tokens=sum(lengths), model_tflop_per_step=round(flops / 1e12, 2), peak_tflops=peak / 1e12,
          gpu=torch.cuda.get_device_name(device), experts_impl=impl, capture=capture_on(), recompute=recompute_policy(),
          torch=torch.__version__)

    def build():
        config = AutoConfig.for_model(**json.loads(json.dumps(data_cfg)))
        config.text_config._experts_implementation = impl
        torch.manual_seed(args.seed)
        with torch.device(device):
            model = AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)
        model.config.text_config._experts_implementation = impl
        frozen = []
        for name, module in model.named_modules():
            if name.endswith("ngram_embedding") and isinstance(module, torch.nn.Embedding):
                module.to("cpu")  # the trainer keeps the table in host RAM
        if s4("PLE_TABLE") or HOST.on:
            install_ple(model)
        head = None
        if args.mtp_weight:
            with torch.device(device):
                head = MtpHead(model.config.text_config, experts=text["mtp_num_experts"], top_k=args.mtp_top_k)
            gh = torch.Generator(device=device).manual_seed(args.seed + 2)
            with torch.no_grad():
                for name, p in head.named_parameters():
                    if "norm" in name:
                        p.zero_()
                    else:
                        p.copy_(torch.randn(p.shape, generator=gh, device=device) * 0.02)
            head.to(torch.bfloat16)
        loaded = proxy_weights(model, head, args.proxy_weights, text["num_experts"], text["mtp_num_experts"]) \
            if args.proxy_weights else None
        for name, p in model.named_parameters():
            if "ngram_embedding." in name or ".visual." in name or name.startswith("model.visual"):
                p.requires_grad_(False)
                frozen.append(p)
            else:
                p.data = p.data.float()
        if head is not None:
            head.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        set_recompute(model)
        model.config.use_cache = True
        runner = shard(Runner(model, head), init_device_mesh("cuda", (1,)), ignored=frozen)
        runner.train()
        expert_params, plain = [], []
        gsel = torch.Generator().manual_seed(args.seed + 3)
        top = max(1, int(round(text["num_experts"] * args.esft)))
        for name, p in runner.named_parameters():
            if not p.requires_grad:
                continue
            m = EXPERT_PARAM.search(name)
            if m and not name.startswith("mtp"):
                expert_params.append((p, sorted(torch.randperm(text["num_experts"], generator=gsel)[:top].tolist())))
            else:
                plain.append(p)
        rows = RowAdam(expert_params)
        adam = torch.optim.AdamW(plain, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
        return model, runner, rows, adam, loaded

    gj = torch.Generator().manual_seed(args.seed + 11)

    def cut_piece(s, c):
        """The piece ``s`` without its last ``c`` rows (--proxy-jitter: new shapes every timed step)."""
        if not c:
            return s
        n = s["ids"].shape[1] - c
        reply = s["reply"][:n]
        return {**s, "ids": s["ids"][:, :n], "teacher": tuple(t[:n] for t in s["teacher"]), "reply": reply,
                "ids_cpu": s["ids_cpu"][:n], "reply_n": float(reply.sum())}

    def step(model, runner, rows, adam, trace=None, grads=False, jitter=0):
        dummy = dp_dummy(model, device, k=k)
        acc = {"kd": 0.0, "n": 0.0, "kd_mtp": 0.0, "n_mtp": 0.0}
        lazy = s4("LAZY_STATS")
        for s in seqs:
            if jitter:
                s = cut_piece(s, int(torch.randint(0, jitter, (1,), generator=gj)))
            with HOST.phase("pos"):
                mine = {"data": s, "mask": s["reply"], "general": False, "n_host": s["reply_n"], "nr_host": s["reply_n"],
                        "piece": fq.positions_and_images(model, {"input_ids": s["ids"]}, host_index=lazy),
                        "prefix": s["prefix"]}
            if s4("PLE_TABLE"):  # the trainer reads the rows with the file (prefetch thread); here per step, timed
                with HOST.phase("load"):
                    mine["ple"] = ple_rows(model, s["ids_cpu"], pin=True)
            row = dp_sequence(runner, mine, dummy, window=args.window,
                              chunk=int(os.environ.get("HEAL_PROXY_CHUNK") or args.chunk),  # per case (+K=V)
                              scale=1.0 / positions, mtp_weight=args.mtp_weight, trace=trace, trace_cache=False, lazy=lazy)
            for key in acc:
                acc[key] += row[key]
        acc = {key: float(v) for key, v in acc.items()}
        got = None
        if grads:
            got = {n: (p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad).detach().float().cpu()
                   for n, p in runner.named_parameters() if p.grad is not None}
        rows.mask_grads()
        norm = torch.nn.utils.clip_grad_norm_([q for q in runner.parameters() if q.requires_grad], 1.0)
        for group in adam.param_groups:
            group["lr"] = args.lr
        adam.step()
        rows.step(args.lr)
        adam.zero_grad(set_to_none=False)
        rows.zero_grad()
        return {"kd": acc["kd"] / max(acc["n"], 1.0), "kd_mtp": acc["kd_mtp"] / max(acc["n_mtp"], 1.0),
                "grad_norm": float(norm)}, got

    groups = [("mtp", r"^mtp\.|\.mtp\."), ("embed", r"embed_tokens"), ("lm_head", r"lm_head"),
              ("hc", r"hyper_connection"), ("ple", r"\.ple\."), ("gdn", r"linear_attn"), ("attn", r"self_attn"),
              ("experts", r"mlp\.experts"), ("router", r"mlp\.gate\."), ("shared", r"shared_expert")]

    def cosine(a, b):
        sums = {}
        for name in b:
            x, y = a[name].double(), b[name].double()
            kind = next((g for g, rx in groups if re.search(rx, name)), "other")
            acc = sums.setdefault(kind, [0.0, 0.0, 0.0])
            acc[0] += float((x * y).sum()); acc[1] += float((x * x).sum()); acc[2] += float((y * y).sum())
        cos = lambda v: v[0] / max(math.sqrt(v[1] * v[2]), 1e-300)  # noqa: E731
        tot = [sum(v[i] for v in sums.values()) for i in range(3)]
        return {"cosine": cos(tot), "norm_rel": _rel(math.sqrt(tot[1]), math.sqrt(tot[2])), "same_names": set(a) == set(b),
                "by_kind": {k: [round(cos(v), 6), round(math.sqrt(v[2]), 6)] for k, v in sorted(sums.items())}}

    from torch.profiler import ProfilerActivity, profile as _profile

    results, reference = [], None
    install_mask_host()  # the S4 flags are read per call: each case sets its own
    base_env = dict(os.environ)
    for ci, spec in enumerate(args.proxy_cases.split(",")):
        # a case: HEAL_COMPILE mode, then "+K=V" environment of the case (trainer MFU S4), e.g. off+HEAL_MASK_HOST=1
        mode, *pairs = spec.split("+")
        os.environ.clear()
        os.environ.update(base_env, **dict(pair.split("=", 1) for pair in pairs))
        HOST.on = s4("HOST_TIMERS")
        os.environ["HEAL_COMPILE"] = "" if mode == "off" else mode
        import sys

        hyper = None
        if os.environ.get("HEAL_HYPER_FAST") == "1":  # the case's own env (fast_attention ran before it was set)
            import rl1_hyper as hyper

            hyper.install()
            hyper.HYPER.update(on=True, calls=0, fallback=0, why={})
        elif "rl1_hyper" in sys.modules:
            sys.modules["rl1_hyper"].HYPER["on"] = False
        if os.environ.get("HEAL_FLA_CONV") == "1":
            install_fla_conv()
        FLA_CONV.update(on=os.environ.get("HEAL_FLA_CONV") == "1", calls=0, fallback=0)
        before = compile_counters() if mode == "layers" else None
        model, runner, rows, adam, loaded = build()
        torch.cuda.reset_peak_memory_stats(device)
        case = {"case": ci + args.proxy_case_index, "mode": spec, "loaded": loaded, "steps": []}
        for i in range(args.proxy_steps):
            trace = [] if i == 0 else None
            torch.cuda.synchronize(device)
            t0 = time.time()
            row, grads = step(model, runner, rows, adam, trace=trace, grads=(i == 0 and args.proxy_grads))
            torch.cuda.synchronize(device)
            row["seconds"] = round(time.time() - t0, 2)
            if trace is not None:
                case["windows"] = [{k2: w[k2] for k2 in ("window", "loss", "kd", "kd_mtp")} for w in trace]
            if grads is not None:  # case 0 of a series keeps its gradients in --proxy-ref; later cases compare
                ref_file = Path(args.proxy_ref or out) / "proxy-grads-ref.pt"
                if reference is None and ref_file.exists() and ci + args.proxy_case_index > 0:
                    reference = torch.load(ref_file, map_location="cpu", weights_only=True)
                if reference is None:
                    reference = grads
                    if args.proxy_ref:
                        ref_file.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(grads, ref_file)
                else:
                    case["grad_vs_case0"] = cosine(grads, reference)
                del grads
            case["steps"].append(row)
            event(out, case=ci, mode=mode, step=i + 1, **row)
        times = []
        for _ in range(args.proxy_timed):
            torch.cuda.synchronize(device)
            HOST.reset()
            t0 = time.time()
            step(model, runner, rows, adam, jitter=args.proxy_jitter)
            torch.cuda.synchronize(device)
            times.append(time.time() - t0)
            if HOST.on:
                case["host"] = HOST.report()
        if times:
            best = sorted(times)[len(times) // 2]
            case["timed_s"] = [round(t, 3) for t in times]
            case["step_s"] = round(best, 3)
            case["tok_per_s"] = round(sum(lengths) / best, 1)
            case["mfu"] = round(flops / best / peak, 4) if peak else None
        if args.proxy_profile:
            torch.cuda.synchronize(device)
            prof = _profile(activities=[ProfilerActivity.CUDA])
            prof.start()
            t0 = time.time()
            step(model, runner, rows, adam)
            torch.cuda.synchronize(device)
            wall = time.time() - t0
            prof.stop()
            case["profile"] = profile_summary(prof, wall)
            del prof
        if args.proxy_opprof:  # device time by aten op over one step (CPU+CUDA activity; op-level attribution)
            torch.cuda.synchronize(device)
            prof = _profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
            prof.start()
            step(model, runner, rows, adam)
            torch.cuda.synchronize(device)
            prof.stop()
            ka = prof.key_averages(group_by_input_shape=True)
            dev = lambda e: getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)  # noqa: E731
            case["ops"] = [[e.key[:60], str(e.input_shapes)[:120], round(dev(e) / 1e6, 3), e.count]
                           for e in sorted(ka, key=lambda e: -dev(e))[:60]]
            del prof, ka
        if args.proxy_cprofile:  # host time by Python function over one step (the launch-bound share)
            import cProfile
            import pstats

            torch.cuda.synchronize(device)
            prof = cProfile.Profile()
            prof.enable()
            step(model, runner, rows, adam)
            torch.cuda.synchronize(device)
            prof.disable()
            stats = pstats.Stats(prof)
            rows_ = [(f"{Path(f).name}:{line}:{fn}", round(st[2], 3), round(st[3], 3), st[1])
                     for (f, line, fn), st in stats.stats.items()]
            case["cprofile_tottime"] = sorted(rows_, key=lambda r: -r[1])[:30]
            case["cprofile_cumtime"] = sorted(rows_, key=lambda r: -r[2])[:40]
        case["peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        if hyper is not None:
            case["hyper"] = {k: hyper.HYPER[k] for k in ("calls", "fallback", "why", "dtypes")}
        if FLA_CONV["on"]:
            case["fla_conv"] = {k: FLA_CONV[k] for k in ("calls", "fallback")}
        if before is not None:
            case["compile"] = compile_counters()
        event(out, event="proxy_case", **case)
        results.append(case)
        if args.proxy_session and ci == len(args.proxy_cases.split(",")) - 1:
            gs = torch.Generator().manual_seed(args.seed + 9)
            lo, hi = (int(x) for x in args.proxy_session_range.split(","))
            session = []
            dummy = dp_dummy(model, device, k=k)
            for j in range(args.proxy_session):
                length = int(torch.randint(lo, hi + 1, (1,), generator=gs))
                ids = torch.randint(10, vocab - 10, (1, length), generator=gs).to(device)
                top = (torch.log_softmax(torch.randn(length, k, generator=gs), -1) - 0.5)
                tail = torch.log1p(-top.double().exp().sum(-1).clamp(max=1 - 1e-6)).half()
                s = {"ids": ids, "teacher": (top.half().to(device),
                                             torch.randint(0, vocab, (length, k), generator=gs, dtype=torch.int32).to(device),
                                             tail.to(device)), "reply": torch.ones(length, device=device)}
                mine = {"data": s, "mask": s["reply"], "general": False,
                        "piece": fq.positions_and_images(model, {"input_ids": ids})}
                c0 = compile_counters() if mode == "layers" else None
                torch.cuda.synchronize(device)
                t0 = time.time()
                dp_sequence(runner, mine, dummy, window=args.window, chunk=args.chunk, scale=1e-6,
                            mtp_weight=args.mtp_weight)
                torch.cuda.synchronize(device)
                c1 = compile_counters() if mode == "layers" else None
                session.append({"length": length, "last_window": span_windows(length, args.window)[-1][1] - span_windows(length, args.window)[-1][0],
                                "seconds": round(time.time() - t0, 2),
                                "new_frames": (c1["frames"] - c0["frames"]) if c0 else None,
                                "compile_s": round(c1.get("compile_seconds", 0) - c0.get("compile_seconds", 0), 2) if c0 else None})
                runner.zero_grad(set_to_none=False)
            event(out, event="proxy_session", mode=mode, sequences=session,
                  compile=compile_counters() if mode == "layers" else None)
        del model, runner, rows, adam
        gc.collect()
        torch.cuda.empty_cache()
    if len(results) > 1:
        base = results[0]
        verdict = {}
        for case in results[1:]:
            tag = f"{case['case']}:{case['mode']}"
            w = [_rel(x["loss"], y["loss"]) for x, y in zip(case.get("windows", []), base.get("windows", []))]
            verdict[tag] = {"window_loss_max_rel": max(w) if w else None,
                            "step_loss_rel": [_rel(x["kd"], y["kd"]) for x, y in zip(case["steps"], base["steps"])],
                            "grad": case.get("grad_vs_case0"),
                            "speedup": round(base["step_s"] / case["step_s"], 3) if case.get("step_s") and base.get("step_s") else None}
        event(out, check="proxy_compare", **verdict)
    dist.destroy_process_group()


def main():
    try:  # fla chunk_fwd_kernel_o BK/BV 64 on B300 (wrong tiles at 128); no fla: no-op
        import fla_pin
    except ImportError:  # CPU tool images built without it
        if torch.cuda.is_available():
            raise
    else:
        fla_pin.pin()
    if os.environ.get("HEAL_THREADS"):  # CPU preflight ranks: intra-op threads (Modal sets OMP_NUM_THREADS=1)
        torch.set_num_threads(int(os.environ["HEAL_THREADS"]))
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tiny", "speed", "score", "layers", "train", "train-tiny", "preflight-student",
                                     "capture-check", "inert-check", "recompute-mem", "proxy", "plan"])
    ap.add_argument("--students", nargs="*", default=[])
    ap.add_argument("--student-root", default="/assets/sft-heal")
    ap.add_argument("--student")
    ap.add_argument("--teacher", help="speed: folder of op_topk files")
    ap.add_argument("--seconds", type=float, default=900)
    ap.add_argument("--window", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=8192, help="no-grad prefix extension chunk")
    ap.add_argument("--lm-weight", type=float, default=0.1)
    ap.add_argument("--optimizer", choices=["none", "adamw"], default="none")
    ap.add_argument("--compare-k", type=int, nargs="*", default=[], help="speed: KD of the unhealed student per k")
    ap.add_argument("--compare-files", type=int, default=4)
    ap.add_argument("--general", help="speed: folder of general-text teacher files (KL probe before training)")
    ap.add_argument("--mtp-weight", type=float, default=0.0, help="speed: > 0 trains the MTP head (KD on row j)")
    ap.add_argument("--mtp-top-k", type=int, default=10, help="the MTP block keeps the uncut model's top-k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).with_name("flash_config.json")))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--fp32", action="store_true", help="tiny: fp32 compute under FSDP (the exact comparison)")
    ap.add_argument("--windows", help="layers: speedmap tf.json (excl5 recorded play)")
    ap.add_argument("--tokens", type=int, default=4096, help="layers: last tokens of each window")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--lazy", type=int, default=6)
    ap.add_argument("--cost-full", type=float, default=2.0)
    ap.add_argument("--sets", nargs="*", default=[], help="layers: name=i,j,... sets to score")
    ap.add_argument("--reverse", type=float, default=1.0, help="train: reverse-KL weight")
    ap.add_argument("--forward", type=float, default=0.1, help="train: forward-KL weight")
    ap.add_argument("--positions", choices=["reply", "all"], default="reply")
    ap.add_argument("--general-share", type=float, default=0.2, help="train: share of trained positions from general text")
    ap.add_argument("--esft", type=float, default=0.25, help="train: share of routed experts per layer that train")
    ap.add_argument("--routing", help="train: saved routed-token counts (torch [layers, experts])")
    ap.add_argument("--route-files", type=int, default=8)
    ap.add_argument("--total-steps", type=int, default=200, help="train: <= 0 fits the schedule to the staged data")
    ap.add_argument("--allow-unfinished-schedule", action="store_true",
                    help="train: run although the staged data ends before the schedule (the decay would not run)")
    ap.add_argument("--loop-share", type=float, default=0.0, help="train: expected share of own positions the loop cut removes")
    ap.add_argument("--seed", type=int, default=0, help="train: queue order seed (a resume keeps the checkpoint's)")
    ap.add_argument("--checkpoint-at", type=lambda v: {int(x) for x in v.split(",") if x}, default=set(),
                    help="train: also checkpoint after these steps (the smoke's exact-resume check)")
    ap.add_argument("--heldout", help="train: teacher folder for a held-out KL read after training (lr probes)")
    ap.add_argument("--heldout-files", type=int, default=4)
    ap.add_argument("--no-save", action="store_true", help="train: no student or optimizer save at the end (probes)")
    ap.add_argument("--schedule", choices=["wsd", "cosine"], default="wsd", help="train: lr schedule")
    ap.add_argument("--warmup", type=float, default=0.1, help="train: warm-up share of the steps (>= 2 steps)")
    ap.add_argument("--step0", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=500_000, help="train: trained positions per optimizer step")
    ap.add_argument("--resume", help="train: folder with optim-rank<r>.pt")
    ap.add_argument("--checkpoint-seconds", type=float, default=0, help="train: a restart point at most this often")
    ap.add_argument("--skip", help="train: a checkpoint's progress.json (consumed files skipped, counters restored)")
    ap.add_argument("--truncate-loops", action="store_true",
                    help="train: own files' reply positions after each reply's loop onset are not trained (OPD)")
    ap.add_argument("--frozen-root", help="train: checkpoint whose frozen shards (n-gram table, vision) are linked")
    ap.add_argument("--tiny-phase", type=int, default=1)
    ap.add_argument("--tag", default="", help="train-tiny: suffix of the phase 3-6 output folders (the comm check)")
    ap.add_argument("--world", type=int, default=2, help="plan: the launch's ranks (cards)")
    ap.add_argument("--flush-last", action="store_true", help="train: step on the remainder when the files run out")
    ap.add_argument("--depths", type=int, nargs="*", default=[4, 8], help="recompute-mem: layer counts")
    ap.add_argument("--mem-experts", type=int, default=32, help="recompute-mem: routed experts per layer (top-10)")
    ap.add_argument("--mem-windows", type=int, nargs="*", default=[1024, 2048, 4096], help="recompute-mem: windows")
    ap.add_argument("--mem-prefix", type=int, default=4096, help="recompute-mem: prefix tokens of the prefix read")
    ap.add_argument("--real-mtp-keys", help="tiny: a file of the real checkpoint's tensor names (first column)")
    ap.add_argument("--proxy-layers", type=int, default=8, help="proxy: decoder layers (Flash's pattern from layer 0)")
    ap.add_argument("--proxy-experts", type=int, default=0, help="proxy: routed experts per layer (0: the config's)")
    ap.add_argument("--proxy-mtp-experts", type=int, default=0, help="proxy: MTP experts (0: the config's)")
    ap.add_argument("--proxy-config", help="proxy: a real student's config.json (default --config)")
    ap.add_argument("--proxy-weights", help="proxy: a real student folder (its tensors for the kept layers)")
    ap.add_argument("--proxy-ngram", type=int, default=5000, help="proxy: n-gram table base size (host RAM)")
    ap.add_argument("--proxy-seqs", default="65536,40000,20000", help="proxy: sequence lengths of one step")
    ap.add_argument("--proxy-jitter", type=int, default=0, help="proxy: timed steps cut up to this many rows per piece")
    ap.add_argument("--proxy-pieces", default="", help="proxy: OPD pieces total:prefix:reply_share,... (else seqs)")
    ap.add_argument("--proxy-cases", default="off,layers", help="proxy: HEAL_COMPILE per case (off, regions, layers)")
    ap.add_argument("--proxy-steps", type=int, default=6, help="proxy: optimizer steps per case (losses recorded)")
    ap.add_argument("--proxy-timed", type=int, default=3, help="proxy: timed steps per case after those")
    ap.add_argument("--proxy-profile", action="store_true", help="proxy: one profiled step per case")
    ap.add_argument("--proxy-grads", action="store_true", help="proxy: first-step gradients against case 0")
    ap.add_argument("--proxy-cprofile", action="store_true", help="proxy: one step under cProfile (host time)")
    ap.add_argument("--proxy-opprof", action="store_true", help="proxy: one step's device time by aten op")
    ap.add_argument("--proxy-case-index", type=int, default=0, help="proxy: index of the first case (one process each)")
    ap.add_argument("--proxy-ref", help="proxy: folder of case 0's first-step gradients (written by case 0)")
    ap.add_argument("--proxy-session", type=int, default=0, help="proxy: random-length sequences after the last case")
    ap.add_argument("--proxy-session-range", default="2000,70000")
    ap.add_argument("--proxy-peak", type=float, default=0.0, help="proxy: peak FLOP/s (default by GPU name)")
    args = ap.parse_args()
    if args.mode == "tiny":
        run_tiny(args)
    elif args.mode == "score":
        fast_attention()
        run_score(args)
    elif args.mode == "layers":
        fast_attention()
        run_layers(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "train-tiny":
        run_train_tiny(args)
    elif args.mode == "preflight-student":
        run_preflight_student(args)
    elif args.mode == "capture-check":
        run_capture_check(args)
    elif args.mode == "inert-check":
        run_inert_check(args)
    elif args.mode == "recompute-mem":
        run_recompute_mem(args)
    elif args.mode == "proxy":
        run_proxy(args)
    elif args.mode == "plan":
        run_plan(args)
    else:
        run_speed(args)


if __name__ == "__main__":
    main()
