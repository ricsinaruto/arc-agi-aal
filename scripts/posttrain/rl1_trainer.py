"""RL step 1 trainer service (13 October, docs/rl-plan.md): Flash's own NVFP4 checkpoint with a rank-32 LoRA on every
routed expert and the dense parts, 8-bit Adam over the fp32 adapter (the master copy), one B300 shared with the play
server. Runs in the pinned Axolotl image's Python beside ``flash_qlora_job.py`` (loader, request encoder, cached
prefix) and ``rl_job.py`` (window planning, actor log-probabilities), which it imports.

The driver (``arena.rl_loop``) writes ``<dir>/cmd/<n>.json``; the service answers ``<dir>/done/<n>.json`` and keeps
``<dir>/alive.json`` fresh. Operations:

- ``fit``: one optimizer step at ``lr`` on a text window (allocates the 8-bit states; the perturbed adapter for the
  play = train parity check), then a 32K-window step with activation checkpointing off, on every other layer and on
  all layers: seconds and peak memory each (an out-of-memory is a result); the fastest setting that leaves
  ``margin_gib`` free on the card is kept. Saves the perturbed adapter.
- ``score_ids``: raw log-probabilities of plain token sequences, adapter on or off (parity with the server).
- ``reset``: the adapter back to its initial state (A as initialised, B zero), a new optimizer.
- ``score``: windows of recorded attempts without an update: per sampled token the actor's and the raw
  log-probability; ``label`` keeps them as a KL reference (``base``, or the previous round).
- ``update``: one optimizer step from attempts with advantages (positive-only weighting is the driver's), at most
  ``max_windows`` windows, each the attempt's heaviest; every metric of the plan's update row; then the probes.
- ``save``: the adapter in BF16 PEFT format (what vLLM loads).
- ``exit``.

The loss of one update: ``-(1/N) sum_t w_t log pi(t)`` over the sampled reply tokens of the update's windows, with
``pi`` the actor (temperature, top-k, top-p set rebuilt from the engine's recorded alternatives) and ``N`` the
update's trained tokens. Logits are kept only at those tokens (``logits_to_keep``).
"""

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flash_qlora_job as fq  # noqa: E402
import rl_job  # noqa: E402

TOTAL_GIB = None


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


def stream_forward(model, sub, image_token_id, *, adapter=None, device="cuda:0", autocast=True):
    """``heal_stream.StreamTeacher``'s forward for an HF Flash model: [lo, hi) in ``sub``-token slices against the
    chain's cache (checked to hold exactly ``lo`` tokens); ``adapter``: a context factory (the base model under a
    LoRA wrapper)."""
    import contextlib

    import torch

    core = (model.get_base_model() if hasattr(model, "get_base_model") else model).model

    def forward(chain, lo, hi, keep):
        inputs = chain.inputs
        ids = inputs["input_ids"][:, :hi].to(device)
        piece = range_piece(core, inputs, lo, hi, image_token_id, device=device)
        cache = chain.cache if lo > 0 else None
        if lo > 0 and (cache is None or cache.get_seq_length() != lo):
            raise RuntimeError(f"{chain.key}: cache holds {None if cache is None else cache.get_seq_length()} "
                               f"tokens, not {lo}")
        cast = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
        with torch.no_grad(), cast, (adapter() if adapter is not None else contextlib.nullcontext()):
            for a in range(lo, hi, sub):
                b = min(a + sub, hi)
                out = model(input_ids=ids[:, a:b], attention_mask=torch.ones_like(ids[:, :b]), past_key_values=cache,
                            use_cache=True, logits_to_keep=(b - a) if keep else 1, **piece(a, b))
                cache = chain.cache = out.past_key_values
                yield a, b, (out.logits[0] if keep else None)
                del out

    forward.sub = sub  # StreamTeacher.verify's image-safe slicing
    return forward


class _Handoff(Exception):
    """Raised by the split layer's pre-hook on the stage-A thread: that layer's (moved) inputs."""

    def __init__(self, args, kwargs):
        super().__init__("stage handoff")
        self.args, self.kwargs = args, kwargs


class PipeForward:
    """``stream_forward`` as a two-stage pipeline over a spread model (MFU-2, 23 Oct): a worker thread runs the
    model's own forward up to decoder layer ``split`` (the first layer on the second card; a pre-hook raises the layer's
    inputs out), the caller's thread runs layers ``split:``, the mixer and the head. Stage A of slice j+1 overlaps
    stage B of slice j; ``prefetch`` queues stage A of the next piece's first slice behind the current piece. The same
    kernels on the same inputs as ``stream_forward``. A chain's stage-A state runs ahead of its stage-B state until its
    slices are consumed: the caller keeps a prefetched chain's cache (``pending_chain``). The adapter context, if any,
    must be held by the caller for the forward's whole life (the worker runs between pieces)."""

    def __init__(self, model, sub, image_token_id, *, split=None, device="cuda:0", autocast=True, timing=None):
        import queue
        import torch

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.model, self.core, self.sub, self.image_token_id, self.device = model, base.model, sub, image_token_id, device
        layers = self.core.language_model.layers
        if split is None:
            first = next(layers[0].parameters()).device
            split = next((i for i, layer in enumerate(layers) if next(layer.parameters()).device != first), None)
            if split is None:
                raise RuntimeError("PipeForward: every layer on one device")
        # the prologue sizes the next slice's masks from the first attention layer's cache (get_mask_sizes /
        # get_seq_length): that layer must be in stage A, or stage A of slice j+1 reads it before stage B of slice j
        # wrote it (tiny teacher split 3, CPU hunt 23 Oct: a 16-wide mask for 32 keys). Flash: layer 3 < split 24.
        types = list(getattr(self.core.language_model.config, "layer_types", None) or [])
        first_attn = next((i for i, t in enumerate(types) if t != "linear_attention"), 0)
        if first_attn >= split:
            raise RuntimeError(f"PipeForward: split {split} must follow the first attention layer {first_attn}")
        self.split, self.tail = split, list(layers[split:])
        self.mixer, self.head = self.core.language_model.hyper_connection_mixer, base.lm_head
        self.autocast, self.timing = autocast, timing
        self.cuda = torch.device(device).type == "cuda"
        self.local = threading.local()
        self.hook = layers[split].register_forward_pre_hook(self._cut, with_kwargs=True)  # after spread's mover
        self.jobs, self.pending, self.closed, self.fatal = queue.Queue(), None, False, None
        self._torch = torch  # before the worker starts (it reads it at once: a hang, CPU hunt 23 Oct)
        self.worker = threading.Thread(target=self._work, daemon=True, name="stage-a")
        self.worker.start()

    def _cut(self, module, args, kwargs):
        if getattr(self.local, "stage_a", False):
            raise _Handoff(args, kwargs)

    def _cast(self):
        import contextlib
        torch = self._torch
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext()

    # --- stage A (worker thread) ----------------------------------------------------------------------------

    def _work(self):
        try:
            torch = self._torch
            self.local.stage_a = True
            with torch.no_grad(), self._cast():
                while True:
                    job = self.jobs.get()
                    if job is None:
                        return
                    try:
                        self._run(job)
                    except BaseException as exc:  # noqa: BLE001 - delivered to the consumer
                        job.put(("error", exc))
        except BaseException as exc:  # noqa: BLE001 - the consumer raises it instead of waiting forever
            self.fatal = exc

    def _get(self, job):
        """The job's next item; raises if the worker died (never waits on a dead worker)."""
        import queue
        while True:
            try:
                return job.out.get(timeout=1.0)
            except queue.Empty:
                if not self.worker.is_alive():
                    raise RuntimeError("PipeForward: the stage-A worker died") from self.fatal

    def _run(self, job):
        torch = self._torch
        st, chain = job.state, job.state.chain
        if st.piece is None:
            lo, hi = st.lo, st.hi
            st.ids = chain.inputs["input_ids"][:, :hi].to(self.device)
            st.piece = range_piece(self.core, chain.inputs, lo, hi, self.image_token_id, device=self.device)
            st.cache = chain.cache if lo > 0 else None
            if lo > 0 and (st.cache is None or st.cache.get_seq_length() != lo):
                raise RuntimeError(f"{chain.key}: cache holds "
                                   f"{None if st.cache is None else st.cache.get_seq_length()} tokens, not {lo}")
        for a, b in job.slices:
            if job.cancel:
                return
            began = time.perf_counter()
            if self.cuda:  # spread's movers leave the thread on the last card; stream_forward starts on the first
                torch.cuda.set_device(self.device)
            try:
                self.model(input_ids=st.ids[:, a:b], attention_mask=torch.ones_like(st.ids[:, :b]),
                           past_key_values=st.cache, use_cache=True, logits_to_keep=1, **st.piece(a, b))
            except _Handoff as h:
                handoff = h
            else:
                raise RuntimeError(f"PipeForward: layer {self.split} never ran")
            st.cache = chain.cache = handoff.kwargs["past_key_values"]
            if self.timing is not None:
                self.timing["a_seconds"] = self.timing.get("a_seconds", 0.0) + time.perf_counter() - began
            job.put(("slice", (a, b, handoff.args, handoff.kwargs)))

    # --- stage B (caller) -----------------------------------------------------------------------------------

    class _State:
        def __init__(self, chain, lo, hi):
            self.chain, self.lo, self.hi, self.ids, self.piece, self.cache = chain, lo, hi, None, None, None

    class _Job:
        def __init__(self, state, slices):
            import queue
            self.state, self.slices, self.cancel = state, slices, False
            self.out = queue.Queue(maxsize=1)

        def put(self, item):
            import queue
            while not self.cancel:
                try:
                    self.out.put(item, timeout=0.5)
                    return
                except queue.Full:
                    continue

    def _slices(self, lo, hi):
        return [(a, min(a + self.sub, hi)) for a in range(lo, hi, self.sub)]

    def prefetch(self, chain, lo, hi):
        """Queue stage A of [lo, hi)'s first slice; the next ``__call__`` must be this (chain, lo, hi)."""
        if self.pending is not None:
            raise RuntimeError("PipeForward: one prefetch at a time")
        if lo >= hi:
            return
        state = self._State(chain, lo, hi)
        job = self._Job(state, self._slices(lo, hi)[:1])
        self.pending = job
        self.jobs.put(job)

    @property
    def pending_chain(self):
        return None if self.pending is None else self.pending.state.chain

    def __call__(self, chain, lo, hi, keep):
        torch = self._torch
        slices = self._slices(lo, hi)
        jobs = []
        p = self.pending
        if p is not None and p.state.chain is chain:
            if (p.state.lo, p.state.hi) != (lo, hi) or not keep:
                raise RuntimeError(f"{chain.key}: prefetched [{p.state.lo}, {p.state.hi}), asked [{lo}, {hi}) keep={keep}")
            self.pending = None
            jobs.append(p)
            if len(slices) > 1:
                jobs.append(self._Job(p.state, slices[1:]))
                self.jobs.put(jobs[-1])
        elif slices:
            jobs.append(self._Job(self._State(chain, lo, hi), slices))
            self.jobs.put(jobs[-1])
        try:
            with torch.no_grad(), self._cast():
                for job in jobs:
                    for _ in job.slices:
                        kind, item = self._get(job)
                        if kind == "error":
                            raise item
                        a, b, args, kwargs = item
                        began = time.perf_counter()
                        hidden = args[0]
                        for layer in self.tail:
                            hidden = layer(hidden, *args[1:], **kwargs)
                        hidden = self.mixer(hidden)
                        logits = self.head(hidden[:, -(b - a):, :])[0] if keep else None
                        del hidden, args, kwargs, item
                        if self.timing is not None:
                            self.timing["b_seconds"] = self.timing.get("b_seconds", 0.0) + time.perf_counter() - began
                            self.timing["slices"] = self.timing.get("slices", 0) + 1
                        yield a, b, logits
                        del logits
        finally:
            for job in jobs:  # an abandoned generator: the worker skips the rest (the chain's stage A is ahead)
                job.cancel = True
                while not job.out.empty():
                    job.out.get_nowait()

    def close(self):
        """Stop the worker (a pending prefetch is dropped: its chain's cache is ahead, so it is cleared)."""
        if self.closed:
            return
        self.closed = True
        if self.pending is not None:
            self.pending.cancel = True
            self.pending.state.chain.cache = None
            self.pending = None
        self.jobs.put(None)
        self.worker.join(timeout=600)
        self.hook.remove()


def perturbed(inputs, lo, hi, image_id):
    """``inputs`` with the text tokens of [lo, hi) in reverse order (image tokens and the length kept): the same call
    shape, other content after ``lo`` (k000 causality probe, 24 Oct)."""
    import torch

    ids = inputs["input_ids"].clone()
    span = ids[0, lo:hi]
    text = (span != image_id).nonzero().flatten()
    span[text] = span[text].flip(0)
    return {**inputs, "input_ids": ids}


def length_diff(model, inputs, image_id, *, lengths=(64, 2048, 8192, 16384, 24000), rows=(0, 64), layers=(0, 1),
                device="cuda:0", autocast=True, perturb=None, depth=None):
    """c7j v5: rows [r0, r1) scored by no-cache single calls [0, L) of each length L: per module of ``layers`` (forward
    hooks; 3-D [1, L, d] or 2-D [L, d] outputs) and the log-probs, the relative difference to the shortest call. A
    causal model gives the same rows whatever follows them in the call; the first module that does not is the bug.
    ``perturb`` [lo, hi]: one more call [0, hi) with the text tokens of [lo, hi) reversed (key "<hi>p"): rows before
    ``lo`` that change with that content read later tokens (leakage), not only the call's length. ``layers`` "all"
    hooks every layer; ``depth`` 1 keeps a layer's direct children only."""
    import heal_stream
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm = base.model.language_model
    ids0 = inputs["input_ids"][0]
    # each length backed off any image it would split (v5: 8192 cut an image -> get_rope_index size error)
    lengths = sorted({heal_stream.cut_point(ids0, int(x), 0, image_id) for x in lengths if int(x) <= ids0.shape[0]})
    lengths = [x for x in lengths if x >= int(rows[1])]
    r0, r1 = int(rows[0]), int(rows[1])
    rec, order, now = {}, [], {"L": 0}

    def take(name, t):
        if not torch.is_tensor(t) or not t.is_floating_point():
            return
        L = now["L"]
        x = t[0, r0:r1] if t.dim() == 3 and t.shape[0] == 1 and t.shape[1] == L else \
            t[r0:r1] if t.dim() == 2 and t.shape[0] == L else None
        if x is None:
            return
        if name not in order:
            order.append(name)
        rec.setdefault(now.get("key", L), {})[name] = x.detach().float().cpu()

    def post(name):
        return lambda mod, args, out: take(name, out[0] if isinstance(out, (tuple, list)) and out else out)

    handles = [lm.layers[0].register_forward_pre_hook(
        lambda mod, args, kwargs: take("layers0.input", args[0] if args else kwargs.get("hidden_states")), with_kwargs=True)]
    layers = range(len(lm.layers)) if layers == "all" else layers
    for li in layers:
        for name, mod in lm.layers[int(li)].named_modules():
            if depth is not None and name and name.count(".") >= int(depth):
                continue
            handles.append(mod.register_forward_hook(post(f"L{li}.{name}" if name else f"L{li}")))
    forward = stream_forward(model, 10 ** 9, image_id, device=device, autocast=autocast,
                             adapter=getattr(model, "disable_adapter", None))
    logp = {}
    runs = [(L, L, inputs) for L in lengths]
    if perturb is not None:
        lo, hi = heal_stream.cut_point(ids0, int(perturb[0]), 0, image_id), heal_stream.cut_point(ids0, int(perturb[1]), 0, image_id)
        if not r1 <= lo < hi <= ids0.shape[0]:
            raise ValueError(f"perturb [{lo}, {hi}) must follow rows [{r0}, {r1}) inside the sequence")
        if hi not in lengths:
            runs.append((hi, hi, inputs))
        runs.append((f"{hi}p", hi, perturbed(inputs, lo, hi, image_id)))
    try:
        for key, L, inp in runs:
            now["L"] = L
            now["key"] = key
            chain = heal_stream.Chain(f"len{key}", -1, inp)
            for a, b, logits in forward(chain, 0, L, True):
                logp[key] = torch.log_softmax(logits[r0:r1].float(), dim=-1).cpu()
            chain.cache = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    finally:
        for h in handles:
            h.remove()
    ref = lengths[0]
    out = {"lengths": [k for k, _, _ in runs], "rows": [r0, r1], "ref": ref, "modules": {}, "first_over": {}}
    if perturb is not None:  # the content probe's own reference: the same length, unperturbed
        pk, hk = runs[-1][0], runs[-1][1]
        kl = (logp[hk].exp() * (logp[hk] - logp[pk])).sum(-1)
        out["perturb"] = {"span": [lo, hi], "vs_same_length_kl": round(float(kl.mean()), 6),
                          "vs_same_length_kl_max": round(float(kl.max()), 5),
                          "vs_same_length_top1": round(float((logp[pk].argmax(-1) == logp[hk].argmax(-1)).float().mean()), 4)}
    for L in [k for k, _, _ in runs][1:]:
        top = logp[ref].topk(16, dim=-1)
        kl = (logp[ref].exp() * (logp[ref] - logp[L])).sum(-1)  # v7: full-vocab KL(ref || L) per row, nats
        out.setdefault("logp", {})[L] = {
            "top1_agree": round(float((logp[L].argmax(-1) == top.indices[:, 0]).float().mean()), 4),
            "mean_abs": round(float((logp[L].gather(1, top.indices) - top.values).abs().mean()), 5),
            "kl": round(float(kl.mean()), 6), "kl_max": round(float(kl.max()), 5)}
        for name in order:
            a, b = rec.get(L, {}).get(name), rec.get(ref, {}).get(name)
            if a is None or b is None or a.shape != b.shape:
                continue
            rel = round(float((a - b).norm() / max(float(b.norm()), 1e-12)), 6)
            out["modules"].setdefault(name, {})[L] = rel
            if rel > 1e-2 and L not in out["first_over"]:
                out["first_over"][L] = name
    out["order"] = order
    return out


def stream_forward_pipe(model, sub, image_token_id, **kw):
    """``PipeForward`` (``stream_forward``'s pipelined twin)."""
    return PipeForward(model, sub, image_token_id, **kw)


FP8_MAX = 448.0


def expert_params(block):
    """The routed experts' fused (gate_up, down) 3-D parameters of an MoE block, under any wrapper (PEFT's
    ParamWrapper names them experts.base_layer.*, a parametrization *.original); LoRA tensors are skipped."""
    found = {}
    for name, p in block.named_parameters():
        parts = name.split(".")
        if p.ndim != 3 or "lora" in name or "shared_expert" in name or not any(x.startswith("experts") for x in parts):
            continue
        for key in ("gate_up_proj", "down_proj"):
            if key in parts:
                if key in found:
                    raise RuntimeError(f"two {key} tensors: {found[key][0]}, {name}")
                found[key] = (name, p)
    if set(found) != {"gate_up_proj", "down_proj"}:
        raise RuntimeError(f"routed expert weights not found: {sorted(found)}; 3-D: "
                           f"{[n for n, p in block.named_parameters() if p.ndim == 3][:8]}")
    return found["gate_up_proj"][1], found["down_proj"][1]


def rtn_fp8(weight, chunk=64):
    """Per-row (each expert's output channel) RTN to float8_e4m3fn: (codes, fp32 scales [E, rows, 1]);
    dequant = bf16(fp32(code) * scale)."""
    import torch

    codes = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty(weight.shape[:-1] + (1,), dtype=torch.float32, device=weight.device)
    for e in range(0, weight.shape[0], chunk):
        w = weight[e:e + chunk].float()
        s = (w.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
        codes[e:e + chunk] = (w / s).to(torch.float8_e4m3fn)
        scales[e:e + chunk] = s
        del w
    return codes, scales


def dequant_fp8(codes, scales, out, chunk=64):
    for e in range(0, codes.shape[0], chunk):
        out[e:e + chunk] = (codes[e:e + chunk].float() * scales[e:e + chunk]).to(out.dtype)
    return out


_DEQUANT = {}


def _dequant_kernel():
    """Triton: out[r, c] = bf16(fp32(codes[r, c]) * scales[r]) over rows of the flattened [E*rows, cols]."""
    if "k" not in _DEQUANT:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(codes, scales, out, cols, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            s = tl.load(scales + row)
            for c in range(0, cols, BLOCK):
                offs = c + tl.arange(0, BLOCK)
                mask = offs < cols
                base = row.to(tl.int64) * cols
                x = tl.load(codes + base + offs, mask=mask).to(tl.float32)
                tl.store(out + base + offs, (x * s).to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)

        _DEQUANT["k"] = kernel
    return _DEQUANT["k"]


class ExpertsFP8:
    """Routed experts as FP8 e4m3 codes + fp32 row scales (RTN at load), dequantized per MoE block into one shared
    BF16 scratch by a pre-hook (only the routed experts; everything dense stays BF16)."""

    def __init__(self, model, chunk=8):  # small fp32 temporaries: the quantize peak is the load peak + one block's codes
        import torch

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        blocks = [layer.mlp for layer in base.model.language_model.layers]
        self.lock, self.owner, self.store, self.params = threading.Lock(), None, {}, {}
        peak, began = 0, time.time()
        scratch = None
        for block in blocks:
            gu, dn = expert_params(block)
            packed = []
            for p in (gu, dn):
                codes, scales = rtn_fp8(p.data, chunk)
                packed.append((codes, scales))
                if p.is_cuda:
                    peak = max(peak, torch.cuda.memory_allocated(p.device))
            if scratch is None:  # the first block's own BF16 storage becomes the shared scratch (no extra copy)
                scratch = (gu.data, dn.data)
            gu.data, dn.data = scratch[0], scratch[1]  # the other blocks' BF16 copies are freed here
            self.store[id(block)] = packed
            self.params[id(block)] = (gu, dn)
            block.register_forward_pre_hook(self._hook, prepend=True)
        self.scratch = scratch
        if scratch is not None and scratch[0].is_cuda:
            torch.cuda.empty_cache()
        self.blocks = blocks
        self.report = {"blocks": len(blocks), "seconds": round(time.time() - began, 1),
                       "expert_matrices": sum(c.shape[0] for packed in self.store.values() for c, _ in packed),
                       "peak_allocated_gib": round(peak / 2 ** 30, 2)}

    def check(self):
        """The Triton dequant against the torch formula on the first block (bit-equal expected)."""
        import torch

        (gc, gs), _ = self.store[id(self.blocks[0])]
        if not gc.is_cuda:
            return None
        got = dequant(gc, gs, torch.empty(gc.shape, dtype=torch.bfloat16, device=gc.device))
        want = dequant_fp8(gc, gs, torch.empty_like(got))
        return bool(torch.equal(got, want))

    def bench(self, repeats=3):
        """Dequant time of one full forward's worth (every block once), ms."""
        import torch

        if self.scratch is None or not self.scratch[0].is_cuda:
            return None
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(repeats):
            for block in self.blocks:
                self.ready(block)
        torch.cuda.synchronize()
        self.owner = None
        return round((time.perf_counter() - began) * 1000 / repeats, 1)

    def _hook(self, block, args):
        self.ready(block)

    def ready(self, block):
        """The block's experts in the scratch (once per block in a row: batched chains share one dequant)."""
        with self.lock:
            if self.owner is block:
                return
            (gc, gs), (dc, ds) = self.store[id(block)]
            dequant(gc, gs, self.scratch[0])
            dequant(dc, ds, self.scratch[1])
            self.owner = block


def dequant(codes, scales, out):
    """out = bf16(fp32(codes) * scales): Triton on a GPU, torch on CPU (the same rounding)."""
    if codes.is_cuda:
        rows, cols = codes.numel() // codes.shape[-1], codes.shape[-1]
        _dequant_kernel()[(rows,)](codes, scales, out, cols, BLOCK=1024)
        return out
    return dequant_fp8(codes, scales, out)


class _Group:
    """One step of a chain batch: each MoE block's call waits for every chain's rows, runs once on their
    concatenation (routers, routed and shared experts are row-wise) and hands each chain its rows back."""

    def __init__(self, n, lock=None):
        self.xs, self.outs, self.call, self.lock = {}, {}, None, lock
        self.barrier = threading.Barrier(n, action=self._run)

    def _run(self):
        import torch

        block, orig = self.call
        slots = sorted(self.xs)
        xs = [self.xs.pop(i) for i in slots]
        y = orig(torch.cat(xs, dim=-2))
        self.outs = dict(zip(slots, y.split([x.shape[-2] for x in xs], dim=-2)))

    def moe(self, block, orig, slot, x):
        self.xs[slot], self.call = x, (block, orig)
        if self.lock is not None:  # the next chain's dense part runs while this one waits at the rendezvous
            self.lock.release()
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            raise RuntimeError("chain batch: another chain failed") from None
        finally:
            if self.lock is not None:
                self.lock.acquire()
        return self.outs.pop(slot)


class MultiForward:
    """Several chains' pieces per forward on one card (MFU-2, 23 Oct): each chain's slice runs the model's own
    forward in its own thread (prologue, PLE, GDN/attention against its own cache, lm_head, then ``post`` for its
    top-k); the MoE blocks rendezvous, so each layer's experts run once over every chain's rows. ``__call__`` is
    ``stream_forward`` (one chain). The adapter context, if any, is held by the caller."""

    def __init__(self, model, sub, image_token_id, *, chains=4, device="cuda:0", autocast=True):
        from concurrent.futures import ThreadPoolExecutor

        import torch

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.model, self.core, self.sub, self.image_token_id, self.device = model, base.model, sub, image_token_id, device
        self.autocast, self.cuda, self._torch = autocast, torch.device(device).type == "cuda", torch
        self.single = stream_forward(model, sub, image_token_id, device=device, autocast=autocast)
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=chains, thread_name_prefix="chain")
        self.chains = chains
        # coloc-v5 (23 Oct): CUDA illegal memory access after 2.56M tokens at 2 chains. Chains' dense parts (GDN/fla
        # Triton, attention, PLE, lm_head, top-k) ran in parallel threads; Triton's Autotuner keeps per-call state
        # on the instance (nargs, best_config), so concurrent calls can launch with another call's arguments. One
        # chain at a time outside the MoE rendezvous (one CUDA stream either way). RL1_CHAIN_LOCK=0: the old path.
        self.lock = threading.Lock() if os.environ.get("RL1_CHAIN_LOCK", "1") == "1" else None
        self.blocks = [layer.mlp for layer in self.core.language_model.layers]
        for block in self.blocks:
            block.forward = self._wrap(block, block.forward)

    def _wrap(self, block, orig):
        def forward(*args, **kwargs):
            group = getattr(self.local, "group", None)
            if group is None:
                return orig(*args, **kwargs)
            x = args[0] if args else kwargs["hidden_states"]
            return group.moe(block, orig, self.local.slot, x)
        return forward

    def __call__(self, chain, lo, hi, keep):
        return self.single(chain, lo, hi, keep)

    def _one(self, group, slot, st, step, post, index):
        import contextlib
        torch = self._torch
        self.local.group, self.local.slot = group, slot
        if self.lock is not None:
            self.lock.acquire()
        try:
            cast = torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext()
            with torch.no_grad(), cast:
                if self.cuda:
                    torch.cuda.set_device(self.device)
                a, b = st["slices"][step]
                ids = st["ids"]
                out = self.model(input_ids=ids[:, a:b], attention_mask=torch.ones_like(ids[:, :b]),
                                 past_key_values=st["cache"], use_cache=True,
                                 logits_to_keep=(b - a) if st["keep"] else 1, **st["piece"](a, b))
                st["cache"] = st["chain"].cache = out.past_key_values
                logits = out.logits[0] if st["keep"] else None
                del out
                if post is not None and logits is not None:
                    post(index, a, b, logits)
        except BaseException:
            if group is not None:
                group.barrier.abort()
            raise
        finally:
            self.local.group = None
            if self.lock is not None:
                self.lock.release()

    def many(self, requests, post=None):
        """``requests``: [(chain, lo, hi, keep)] of distinct chains. Per step (each chain's next ``sub`` slice, the
        chains that still have one), runs the batch and calls ``post(i, a, b, logits)`` in chain i's thread; yields
        that step's [(i, a, b)]."""
        torch = self._torch
        if len({id(r[0]) for r in requests}) != len(requests):
            raise RuntimeError("chain batch: one piece per chain")
        states = []
        for chain, lo, hi, keep in requests:
            cache = chain.cache if lo > 0 else None
            if lo > 0 and (cache is None or cache.get_seq_length() != lo):
                raise RuntimeError(f"{chain.key}: cache holds {None if cache is None else cache.get_seq_length()} "
                                   f"tokens, not {lo}")
            states.append({"chain": chain, "keep": keep, "cache": cache,
                           "ids": chain.inputs["input_ids"][:, :hi].to(self.device),
                           "piece": range_piece(self.core, chain.inputs, lo, hi, self.image_token_id, device=self.device),
                           "slices": [(a, min(a + self.sub, hi)) for a in range(lo, hi, self.sub)]})
        step = 0
        while True:
            live = [(i, st) for i, st in enumerate(states) if step < len(st["slices"])]
            if not live:
                return
            for start in range(0, len(live), self.chains):  # more requests than threads: in groups
                part = live[start:start + self.chains]
                group = _Group(len(part), self.lock) if len(part) > 1 else None
                futures = [self.pool.submit(self._one, group, slot, st, step, post, i)
                           for slot, (i, st) in enumerate(part)]
                errors = []
                for f in futures:
                    try:
                        f.result()
                    except BaseException as exc:  # noqa: BLE001 - the first real error is raised
                        errors.append(exc)
                if errors:
                    raise next((e for e in errors if "another chain failed" not in str(e)), errors[0])
            yield [(i, *st["slices"][step]) for i, st in live]
            step += 1

    def close(self):
        self.pool.shutdown(wait=True)


def range_piece(core, inputs, lo, hi, image_token_id, device="cuda:0"):
    """``fq.positions_and_images`` for the slice [lo, hi) of a sequence whose earlier tokens are in the cache: M-RoPE
    positions of the first ``hi`` tokens, image features of the images that start in [lo, hi) only (the streaming
    teacher's cut points keep images whole: ``heal_stream.cut_point``). ``inputs`` on the host."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    import heal_stream

    ids = inputs["input_ids"][:, :hi].to(device)
    if inputs.get("pixel_values") is None:
        positions = torch.arange(hi, device=ids.device).view(1, 1, -1).expand(3, 1, -1)
        return lambda a, b: {"position_ids": positions[:, :, a:b]}
    runs = heal_stream.image_runs(inputs["input_ids"][0, :hi], image_token_id)
    grid = inputs["image_grid_thw"]
    whole = [i for i, (a, b) in enumerate(runs) if b <= hi]
    if len(whole) != len(runs):
        raise RuntimeError("an image straddles the slice end")
    mm = inputs["mm_token_type_ids"][:, :hi].to(device)
    positions, _ = core.get_rope_index(ids, mm_token_type_ids=mm, image_grid_thw=grid[:len(runs)].to(device))
    inside = [i for i, (a, b) in enumerate(runs) if a >= lo]
    if any(a < lo < b for a, b in runs):
        raise RuntimeError("an image straddles the slice start")
    before = torch.nn.functional.pad((inputs["input_ids"][0, :hi] == image_token_id).cumsum(0), (1, 0))
    if not inside:
        return lambda a, b: {"position_ids": positions[:, :, a:b]}
    i0, i1 = inside[0], inside[-1] + 1
    rows = grid.prod(-1)
    off, count = int(rows[:i0].sum()), int(rows[i0:i1].sum())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = torch.cat(core.get_image_features(inputs["pixel_values"][off:off + count].to(device),
                                                     grid[i0:i1].to(device), return_dict=True).pooler_output, dim=0)
    base = int(before[lo])
    if features.shape[0] != int(before[hi]) - base:
        raise RuntimeError(f"{features.shape[0]} image features for {int(before[hi]) - base} image tokens")

    def piece(a, b):
        extra = {"position_ids": positions[:, :, a:b]}
        first, last = int(before[a]) - base, int(before[b]) - base
        if last > first:
            extra["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=(features[first:last],))}
        return extra

    return piece


def binned_kl(p_logp, q_logp):
    """Mean over positions of KL(P || Q) on P's top-k tokens plus one bin for the rest of the vocabulary (the same
    k ids on both sides): exact for that coarse-graining, so non-negative and at most the full-vocabulary KL."""
    p, q = p_logp.double().exp(), q_logp.double().exp()
    p_rest, q_rest = (1 - p.sum(1)).clamp(min=1e-12), (1 - q.sum(1)).clamp(min=1e-12)
    terms = (p * (p_logp.double() - q_logp.double())).sum(1) + p_rest * (p_rest.log() - q_rest.log())
    return float(terms.clamp(min=0).mean()) if terms.numel() else float("nan")


def served_tokens(captures, end, window, skip=()):
    """Reply tokens of the replies a window ending at ``end`` holds (``rl_job.heaviest_end``'s count, weight 1)."""
    last = captures[end]
    total, first, index, found = last["prompt_tokens"] + last["usage"]["completion_tokens"], last["prompt_tokens"], end, 0
    while index >= 0:
        row = captures[index]
        stop = row["prompt_tokens"] + row["usage"]["completion_tokens"]
        if index != end and (total - row["prompt_tokens"] + 1 > window or stop > first):
            break
        found += 0 if row["request"] in skip else row["usage"]["completion_tokens"]
        first, index = row["prompt_tokens"], index - 1
    return found


def split_inputs(made, capture, image_token):
    """The split's inputs from one window (``rl_job.attempt_example``) and the capture of its last reply: the prompt
    of that reply as the trainer renders it (``prompt_ids``, which the fresh server pass must reproduce token for
    token), the window's trained positions before it (``positions``; ``keep``: their index in the window's arrays)
    and the engine's sampling-time raw log-probabilities there."""
    last = made["replies"][-1]
    start = int(last["start"])
    ids = made["inputs"]["input_ids"][0].tolist()
    positions = made["arrays"]["positions"].tolist()
    keep = [i for i, p in enumerate(positions) if p < start]
    return {"name": made["name"], "request": last["request"], "prompt_same": last.get("prompt_same"),
            "prompt_ids": ids[:start], "image_tokens": sum(t == image_token for t in ids[:start]), "keep": keep,
            "positions": [positions[i] for i in keep], "engine": [float(made["engine_raw"][i]) for i in keep],
            "payload": capture["payload"]}


def gpu_used_gib():
    import subprocess

    try:
        card = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"  # the trainer's card (2-GPU loop: 1)
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


STREAM_PROMPT_SLACK = 1.0  # streamed requests: the encoder skips its count rule; the stream teacher checks each
# request itself (exact_inputs: the server's ids must match its count exactly; base play: the 8-token rule)
PINNED_LATE = []  # autotuners whose NB key was dropped at their first call


def pin_quack_autotune():
    """BF16 teacher (Stage A smoke v4, 20 Oct): quack's autotuner keys on every tensor shape, so each window's last
    chunk (a new routed-row count M) re-tuned the grouped expert GEMMs, ~35 s a window (73 of 74 configs fail anyway).
    With RL1_PIN_QUACK=1 the first config chosen for a key with M (dim 0 of 2-D tensors) blanked serves every M; CUTLASS
    tiles take any M. Patches every loaded quack Autotuner class once; returns how many."""
    import sys

    from torch import Tensor

    import re

    two_d = re.compile(r"torch\.Size\(\[\d+, (\d+)\]\)")

    def loose_key(key):  # M (dim 0 of every 2-D shape) blanked
        return tuple(two_d.sub(r"torch.Size([M, \1])", k) if isinstance(k, str) else k for k in key)

    patched = 0
    for module in list(sys.modules.values()):
        cls = getattr(module, "Autotuner", None)
        if not isinstance(cls, type) or not hasattr(cls, "check_disk_cache") or getattr(cls, "_m_pinned", False):
            continue
        original = cls.__call__

        def call(self, *args, _original=original, **kwargs):
            if len(getattr(self, "configs", ())) > 1:
                nargs = dict(zip(self.arg_names, args))
                all_args = {k: v for k, v in {**nargs, **kwargs}.items() if k in self.arg_names}
                key = [str(all_args[k]) for k in self.keys if k in all_args]
                for arg in all_args.values():
                    if isinstance(arg, Tensor):
                        key += [str(arg.shape), str([s if s in {0, 1} else 2 for s in arg.stride()]), str(arg.dtype)]
                key = tuple(key)
                pins = self.__dict__.setdefault("_m_pins", {})
                if not self.__dict__.get("_m_seeded"):  # keys tuned before the patch (the first window's)
                    for k0, v0 in list(self.cache.items()):
                        pins.setdefault(loose_key(k0), v0)
                    self._m_seeded = True
                loose = loose_key(key)
                if key not in self.cache and loose in pins:
                    self.cache[key] = pins[loose]
                result = _original(self, *args, **kwargs)
                if key in self.cache:
                    pins.setdefault(loose, self.cache[key])
                return result
            return _original(self, *args, **kwargs)

        cls.__call__, cls._m_pinned = call, True
        patched += 1
    return patched


class QuackAutotuneInJob(RuntimeError):
    """A quack autotune started in a fixed-config job (fix_quack_autotune)."""


def fix_quack_autotune(module=None):
    """Stream teacher (g1k, 23 Oct): no quack autotune in the job. The M-pin above ran before quack's lazily imported
    Autotuner existed, so every new routed-row count M re-tuned (~47 s; its compile pool workers die on the JIT
    module's import, 73 of 74 configs fail). Here quack's Autotuner class is imported and patched before the first
    forward: every call runs the kernel's own untuned default (``config=None``: quack's ``tuned=False``
    path, nvMMH heuristic or ``default_config``), and a benchmark (``prune_configs``) raises. Returns what it patched."""
    if module is None:
        try:
            import quack.autotuner as module
        except Exception as exc:  # noqa: BLE001 - no quack here (CPU): nothing tunes
            return {"patched": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    cls = module.Autotuner
    if getattr(cls, "_fixed", False):
        return {"patched": True, "again": True}
    original = cls.__call__
    untuned = module.AutotuneConfig(config=None)

    def call(self, *args, **kwargs):  # the tuned kernel's own function, untuned (no key: any quack copy's version)
        if len(getattr(self, "configs", ())) > 1:
            self.best_config = untuned
            return self.fn(*args, **kwargs, **untuned.all_kwargs())
        return original(self, *args, **kwargs)

    def refuse(self, *args, **kwargs):
        raise QuackAutotuneInJob(f"quack autotune of {getattr(getattr(self, 'fn', None), '__name__', '?')} in a "
                                 "fixed-config job")

    cls.__call__, cls.prune_configs, cls.check_disk_cache, cls._fixed = call, refuse, refuse, True
    return {"patched": True, "module": module.__name__}


def install_quack_fix():
    """fix_quack_autotune on every quack copy, as each is imported. The BF16 expert path runs sonic-moe's hub build,
    which vendors its own quack (``_sonic_moe_cuda_<hash>...quack.autotuner``; MFU-1 mfu1-qfx-b1, 23 Oct: patching
    pip's quack left it tuning every new M), imported lazily at the first forward. A meta-path finder patches each
    ``*quack.autotuner`` module right after it executes; ``scan()`` patches any already loaded (and any a loader
    bypassed). Returns (patched: {module: info}, scan)."""
    import importlib.abc

    patched = {}

    def scan():
        for name, mod in list(sys.modules.items()):
            if name.endswith("quack.autotuner") and name not in patched and hasattr(mod, "Autotuner") \
                    and hasattr(mod, "AutotuneConfig"):
                patched[name] = fix_quack_autotune(mod)
        return patched

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if not fullname.endswith("quack.autotuner"):
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is None:
                    continue
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    run = spec.loader.exec_module

                    def exec_module(module, _run=run, _name=fullname):
                        _run(module)
                        patched[_name] = fix_quack_autotune(module)

                    spec.loader.exec_module = exec_module
                return spec
            return None

    if not any(type(f).__name__ == "Finder" and getattr(f, "_quack_fix", False) for f in sys.meta_path):
        finder = Finder()
        finder._quack_fix = True
        sys.meta_path.insert(0, finder)
    return scan(), scan


def pin_autotune_keys():
    """Drop the block count ``NB`` from fla's norm-kernel autotune keys (SFT arm, 17 Oct). It changes with the window
    length, so every new length re-tuned l2norm / layer_norm_gated forward and backward (~20 s each time; 303 s of a
    14-minute run in sft-gate-util-v4). One tuned config per kernel serves all lengths."""
    import sys

    try:
        from triton.runtime.autotuner import Autotuner
    except Exception:  # noqa: BLE001
        return {}
    changed = {}
    if not getattr(Autotuner, "_nb_pinned", False):  # also autotuners created later (modules loaded at the first forward)
        original = Autotuner.run

        def run(self, *args, **kwargs):
            if "NB" in list(getattr(self, "keys", []) or []):
                self.keys = [k for k in self.keys if k != "NB"]
                if hasattr(self, "key_idx") and hasattr(self, "arg_names"):
                    self.key_idx = [self.arg_names.index(k) for k in self.keys]
                self.cache = {}
                PINNED_LATE.append(getattr(getattr(self, "fn", None), "__name__", "?"))
            return original(self, *args, **kwargs)

        Autotuner.run, Autotuner._nb_pinned = run, True
    for name, module in list(sys.modules.items()):  # any module: the image vendors fla under other package names
        if module is None:
            continue
        try:
            items = list(vars(module).items())
        except TypeError:
            continue
        for attr, obj in items:
            try:  # only triton objects are walked (others can raise on attribute access, e.g. torch._classes)
                seen = 0
                while obj is not None and seen < 4 and type(obj).__module__.startswith("triton"):
                    if isinstance(obj, Autotuner) and "NB" in list(getattr(obj, "keys", []) or []):
                        obj.keys = [k for k in obj.keys if k != "NB"]
                        if hasattr(obj, "key_idx") and hasattr(obj, "arg_names"):
                            obj.key_idx = [obj.arg_names.index(k) for k in obj.keys]
                        obj.cache = {}
                        changed[f"{name}.{attr}"] = list(obj.keys)
                    obj, seen = getattr(obj, "fn", None), seen + 1
            except Exception:  # noqa: BLE001
                continue
    return changed


def set_checkpointing(model, mode):
    """``full``: every decoder layer recomputed in backward; ``alternate``: every other layer; ``off``: none;
    ``partial:N``: the first N layers not recomputed, the rest recomputed (the peak's slope in N is one layer's saved
    activations plus its saved BF16 expert weights, 13 Oct)."""
    layers = [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer") and hasattr(m, "gradient_checkpointing")]
    kept = int(mode.split(":")[1]) if mode.startswith("partial:") else None
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = (mode == "full" or (mode == "alternate" and index % 2 == 0)
                                        or (kept is not None and index >= kept))
    return len(layers)


GROUPED = {"grouped": 0, "loop": 0}  # expert LoRA calls by path (counted in every pass)
GROUPED_ON = {"on": True}             # the trainer test switches it off to time and compare the loop


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
            if GROUPED_ON["on"] and torch.is_autocast_enabled(x.device.type):
                # the matmuls run in BF16 under autocast either way; the grouped path needs it explicit
                args = (x.to(torch.bfloat16), args[1], args[2], args[3].to(torch.bfloat16), args[4].to(torch.bfloat16),
                        *args[5:])
            x = args[0]
            GROUPED["grouped" if nvfp4_lora._lora_grouped_mm_supported(x, args[3], args[1].shape[0]) else "loop"] += 1
            return _original(*args)

        cls.apply = staticmethod(apply)
        cls._rl1_patched = True


INDEXER = {"vectorized": 0, "fallback": 0}  # QSA indexer calls by path


PLAIN_CAUSAL_ROWS = 2048  # rows per check: temporaries stay rows x keys (c7m: one int64 [55020, 120556] sum = 49.4 GiB OOM)


def causal_counts(attention_mask, rows=None):
    """(counts, ok) of a batch-1 mask as device tensors, no host read: per query the visible-key count, and whether
    every row is a prefix (query i sees keys 0..n_i-1). Checked ``rows`` queries at a time: the bool->int64 sum and the
    comparison never span the whole [queries, keys] mask (c7m 24 Oct: the MTP window of a 120K-token unit OOM'd a B300
    on that sum)."""
    import torch

    rows = int(rows or PLAIN_CAUSAL_ROWS)
    m = attention_mask[0, 0]
    positions = torch.arange(m.shape[-1], device=m.device)
    counts, ok = [], torch.ones((), dtype=torch.bool, device=m.device)
    for a in range(0, m.shape[0], rows):
        part = m[a:a + rows]
        visible = part if part.dtype == torch.bool else part == 0
        c = visible.sum(-1)
        ok &= (visible == (positions[None, :] < c[:, None])).all()
        counts.append(c)
    return (torch.cat(counts) if counts else torch.zeros(0, dtype=torch.long, device=m.device)), ok


def plain_causal(attention_mask, rows=None):
    """Per query the number of visible keys when the mask is a plain causal one (batch 1), else None
    (``causal_counts``, one host read)."""
    if attention_mask.shape[0] != 1:
        return None
    counts, ok = causal_counts(attention_mask, rows)
    return counts if bool(ok) else None


# QSA ties (c7j, 24 Oct): topk breaks ties at the budget's edge (often relu-zero scores) by the row's length, so a
# continued cache (other slice ends, other row lengths) picked other blocks than a fresh pass (tiny teacher: max 0.014
# at sub 1000; GPU verify top-1 0.934). RL1_QSA_TIES=index: ties go to the lowest block index, whatever the row length.
# the heal trainer takes HEAL_* flags only (modal_rl1.env_pairs): HEAL_QSA_TIES is its name for the same switch
TIES = {"mode": os.environ.get("RL1_QSA_TIES") or os.environ.get("HEAL_QSA_TIES") or "topk", "count": False, "rows": 0,
        "tied": 0, "zero": 0}


def tie_by_index(masked, picked, top):
    """The ``top`` largest of each row of ``masked`` with ties at the k-th value going to the lowest indices (the set
    is independent of the row's length and of topk's order). ``picked``: topk's indices (their values give the k-th)."""
    import torch

    kth = masked.gather(1, picked).min(dim=1, keepdim=True).values
    above, equal = masked > kth, masked == kth
    need = top - above.sum(dim=1, keepdim=True)
    take = above | (equal & (equal.cumsum(dim=1) <= need))
    if TIES["count"]:
        TIES["rows"] += masked.shape[0]
        TIES["tied"] += int((equal.sum(dim=1, keepdim=True) > need).sum())
        TIES["zero"] += int(((equal.sum(dim=1, keepdim=True) > need) & (kth == 0)).sum())
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
        complete = torch.div(counts, ratio, rounding_mode="floor")  # complete blocks per query
        host_max = getattr(counts, "_heal_max", None)  # heal_kd HEAL_MASK_HOST: the largest count, read with the check
        blocks = host_max // ratio if host_max is not None else int(complete.max())
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
            # HEAL_INDEXER_BF16 (MFU-1): the same BF16 values on tensor cores with FP32 output (the products are exact
            # in FP32 either way; only the summation order differs) instead of the SIMT FP32 GEMM
            bf16 = (os.environ.get("HEAL_INDEXER_BF16") == "1" and q.is_cuda and q.dtype == torch.bfloat16
                    and block_keys.dtype == torch.bfloat16)
            keys_b = block_keys.transpose(0, 1).contiguous() if bf16 else None
            ids = torch.arange(blocks, device=device)
            for begin in range(0, length, chunk):
                end = min(length, begin + chunk)
                if bf16:
                    qq = q[0, begin:end]
                    raw = torch.mm(qq.reshape(-1, qq.shape[-1]), keys_b, out_dtype=torch.float32)
                    scores = torch.relu(raw.view(end - begin, qq.shape[1], blocks)).sum(dim=1)
                else:
                    scores = torch.relu(torch.matmul(q[0, begin:end].float(), keys_t)).sum(dim=1)
                scores = scores / math.sqrt(self.index_head_dim)
                valid = ids[None, :] < complete[begin:end, None]
                masked = scores.masked_fill(~valid, float("-inf"))
                picked = masked.topk(top, dim=-1).indices
                if TIES["mode"] == "index" or TIES["count"]:
                    tied = tie_by_index(masked, picked, top)
                    picked = tied if TIES["mode"] == "index" else picked
                tokens = picked[:, :, None] * ratio + offsets
                tokens = torch.where(torch.gather(valid, 1, picked)[:, :, None], tokens, -1)
                out[begin:end, : top * ratio] = tokens.reshape(end - begin, top * ratio)
        tail = (complete * ratio)[:, None] + offsets[: ratio - 1]
        out[:, top * ratio:] = torch.where(tail < counts[:, None], tail, -1)
        if SPARSE.get("density"):
            touched, causal = block_density(out, counts)
            SPARSE["blocks"][0] += touched
            SPARSE["blocks"][1] += causal
    return out


# attention mode: masked (the pinned masked SDPA), cudnn (the same call on the cuDNN backend only) or selected
# (``SelectedAttention``); density: per call the share of causal 128x128 blocks the selection touches
SPARSE = {"mode": os.environ.get("RL1_ATTENTION", "masked"), "chunk": 256, "calls": 0, "fallback": 0, "cudnn": 0,
          "cudnn_failed": 0, "density": False, "blocks": [0, 0]}


def block_density(picked, counts, block=128):
    """Touched and causal (query block x key block) pairs of a selection, as ``[touched, causal]``."""
    import torch

    length = picked.shape[0]
    rows = torch.arange(length, device=picked.device) // block
    kv_blocks = int((int(counts.max()) + block - 1) // block)
    q_blocks = int(rows.max()) + 1
    hit = torch.zeros(q_blocks * kv_blocks + 1, dtype=torch.bool, device=picked.device)
    flat = torch.where(picked >= 0, rows[:, None] * kv_blocks + picked // block, q_blocks * kv_blocks)
    hit[flat.reshape(-1)] = True
    last = torch.zeros(q_blocks, dtype=torch.long, device=picked.device).scatter_reduce(
        0, rows, (counts - 1) // block + 1, reduce="amax")
    return [int(hit[:-1].sum()), int(last.sum())]


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


def selected_attention():
    """Patch the full-attention layers: with a plain causal mask (batch 1) the indexer's selection goes to
    ``SelectedAttention`` as key positions (no mask is built); else the original forward (``cudnn``: on the cuDNN
    backend). ``SPARSE["mode"]`` switches it (``op_attention``)."""
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    cls = modeling.Qwen4ExpTextAttention
    if getattr(cls, "_rl1_selected", False):
        return
    original = cls.forward
    function = selected_attention_function()

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        if SPARSE["mode"] == "cudnn":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            try:
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                    out = original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
                SPARSE["cudnn"] += 1
                return out
            except RuntimeError as error:  # no retry: the failed call already extended the caches (rl1-attn-v2)
                SPARSE["cudnn_failed"] += 1
                SPARSE["cudnn_error"] = repr(error)[:500]
                raise
        counts = (plain_causal(attention_mask) if SPARSE["mode"] in ("selected", "triton") and attention_mask is not None
                  else None)
        if counts is None or (self.training and self.attention_dropout):
            SPARSE["fallback"] += 1
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        SPARSE["calls"] += 1
        picked = qsa_select(self.indexer, hidden_states, position_embeddings, counts, past_key_values)
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
        if SPARSE["mode"] == "triton":  # rl1_sparse: the fused gather kernel (fp32 atomics for dk / dv)
            if os.environ.get("HEAL_UNION_ATTENTION") == "1":  # MFU-1: rl1_sparse2's query-union tiles (RL stage E)
                from rl1_sparse2 import UnionSelectedAttention as TritonSelectedAttention
            else:
                from rl1_sparse import TritonSelectedAttention

            attn_output = TritonSelectedAttention.apply(query_states, key_states, value_states, picked, self.scaling, None)
        else:
            attn_output = function.apply(query_states, key_states, value_states, picked, self.scaling, SPARSE["chunk"])
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    cls.forward = forward
    cls._rl1_selected = True
    cls._rl1_original = original


ANNOTATED = {"Qwen4ExpTextAttention": "attn", "Qwen4ExpTextQSAIndexer": "indexer", "Qwen4ExpTextGatedDeltaNet": "gdn",
             "Qwen4ExpTextExperts": "experts", "Qwen4ExpTextTopKRouter": "router", "Qwen4ExpTextMLP": "shared_mlp",
             "Qwen4ExpTextPLELayer": "ple", "Qwen4ExpTextGatedResidual": "hyper", "Qwen4ExpTextNGramEmbedding": "ngram"}
BACKWARD_NODES = (("sdpa", ("ScaledDotProduct", "Sdpa", "Flash", "EfficientAttention", "Cudnn")),
                  ("experts", ("Sonic", "MoE", "Moe", "Expert", "Grouped", "Dequant", "Fp4", "FP4")),
                  ("gdn", ("ChunkGated", "Delta", "Fla", "Conv", "Recurrent", "Gated")))


def annotate_modules(model):
    """Profiler ranges ``rl1:<tag>`` around the forward of the modules in ``ANNOTATED``, the LM head and the SDPA
    call (the checkpoint recompute in the backward runs them again); returns the undo."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    handles, open_ranges = [], {}

    def pre(module, *args):
        rf = torch.profiler.record_function(f"rl1:{module._rl1_tag}")
        rf.__enter__()
        open_ranges.setdefault(id(module), []).append(rf)

    def post(module, *args):
        stack = open_ranges.get(id(module))
        if stack:
            stack.pop().__exit__(None, None, None)

    for name, module in model.named_modules():
        tag = ANNOTATED.get(type(module).__name__) or ("lm_head" if name.endswith("lm_head") else None)
        if tag:
            module._rl1_tag = tag
            handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa(*args, **kwargs):
        with torch.profiler.record_function("rl1:sdpa"):
            return original(*args, **kwargs)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa

    def undo():
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
        for handle in handles:
            handle.remove()

    return undo


def kernel_split(events):
    """Card time (s) of a profile by phase (``phase:*`` ranges; a kernel launched off the main thread during the
    backward is ``backward``) and module (innermost ``rl1:*`` range, else the backward's autograd node, mapped by
    ``BACKWARD_NODES`` or kept as ``node:<name>``), plus the leading kernels per phase."""
    by, kernels, nodes = {}, {}, {}
    for event in events:
        own = getattr(event, "kernels", None) or []
        if not own:
            continue
        phase = module = node = None
        parent = event
        while parent is not None:
            name = parent.name
            if module is None and name.startswith("rl1:"):
                module = name[4:]
            if node is None and name.startswith("autograd::engine::evaluate_function: "):
                node = name.split(": ", 1)[1]
            if name.startswith("phase:"):
                phase = name[6:]
                break
            parent = parent.cpu_parent
        if phase is None:
            phase = "backward" if node is not None else "other"
        if module is None and node is not None:
            module = next((tag for tag, keys in BACKWARD_NODES if any(k in node for k in keys)), f"node:{node[:60]}")
        module = module or "none"
        seconds = sum(k.duration for k in own) / 1e6
        by.setdefault(phase, {})
        by[phase][module] = by[phase].get(module, 0.0) + seconds
        if node is not None and module.startswith("node:"):
            nodes[node] = nodes.get(node, 0.0) + seconds
        for k in own:
            key = (phase, k.name[:100])
            kernels[key] = kernels.get(key, 0.0) + k.duration / 1e6
    out = {"phases": {p: {"total": round(sum(mods.values()), 2),
                          **{m: round(v, 2) for m, v in sorted(mods.items(), key=lambda kv: -kv[1]) if v >= 0.01}}
                      for p, mods in by.items()}}
    out["top_kernels"] = {p: [[name, round(v, 2)] for (q, name), v in sorted(kernels.items(), key=lambda kv: -kv[1])
                              if q == p][:15] for p in by}
    out["unmapped_nodes"] = [[n[:80], round(v, 2)] for n, v in sorted(nodes.items(), key=lambda kv: -kv[1])[:20]]
    return out


class Trainer:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor

        self.args, self.dir = args, Path(args.dir)
        (self.dir / "cmd").mkdir(parents=True, exist_ok=True)
        (self.dir / "done").mkdir(parents=True, exist_ok=True)
        self.rec = fq.Recorder(self.dir / "trainer")
        self.rec.save(environment=fq.environment(self.rec))
        import fla_pin  # fla chunk_fwd_kernel_o BK/BV 64 on B300 (wrong tiles at 128)
        self.rec.event("fla_pinned", config=fla_pin.pin())
        fq.set_adapter(args.rank, False)
        fq.NVFP4.update(on=True, experts=True, backend="sonicmoe")
        fq.CUT_CROSS_ENTROPY["on"] = False  # the loss reads logits at the trained tokens only
        self.processor = AutoProcessor.from_pretrained(args.model)
        self.tokenizer = self.processor.tokenizer
        # transformers pre-reserves the checkpoint's BF16 size (266 GiB) before loading; beside the sleeping
        # server (3.6 GiB) that fails, and the load itself needs about 84 GiB (rl-experts-v1)
        import transformers.modeling_utils as modeling_utils

        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        fq.reset_peaks()
        began = time.time()
        self.model, _, loaded = fq.load(self.rec, args.model, self.dir / "trainer", sequence_len=131072)
        loaded["table_scale"] = fq.apply_table_scale(self.rec, self.model, args.model)
        from axolotl.integrations.kernels.libs.sonicmoe import set_merge_aware_enabled

        set_merge_aware_enabled(False)  # plain LoRA over the dequantised base: what the server computes
        if not args.loop_lora:
            grouped_expert_lora()
        if os.environ.get("RL1_INDEXER", "vectorized") == "vectorized":
            vectorized_indexer()
        selected_attention()  # SPARSE["mode"]: RL1_ATTENTION=masked (default; rl1-attn-v1: the gather path is slower)
        if args.spread > 1:  # BF16 teacher: the decoder layers split over this many cards
            placed = fq.spread(self.model, args.spread)
            loaded["spread"] = {"layers_per_device": [placed.count(i) for i in range(args.spread)],
                                "device_gib": fq.device_gib(peak=False)}
            self.rec.event("spread", **loaded["spread"])
        if os.environ.get("RL1_EXPERTS_FP8") == "1":  # one-card teacher: every op scores with W8A16 routed experts
            self._fp8 = self.experts_fp8()
            loaded["experts_fp8"] = self._fp8.report
        self.model.train()
        self.params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.initial = {n: p.detach().to("cpu", copy=True) for n, p in self.params if "lora_A" in n}
        self.optimizer = self.make_optimizer(args.lr)
        self.checkpointing = "full"
        set_checkpointing(self.model, self.checkpointing)
        loaded["autotune_pinned"] = pin_autotune_keys()
        self.encoded, _ = fq.example_maker(self.processor, [], tail=args.window, phases=[args.window])
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>")
        self.references = {}  # probe name -> {"base": [...], "previous": [...]} per-token actor log-probabilities
        self.updates = 0
        self.parked, self.pinned = False, {}
        used, total, _ = gpu_used_gib()
        loaded.update(seconds=round(time.time() - began, 1), trainable=sum(p.numel() for _, p in self.params),
                      trainable_dtypes=sorted({str(p.dtype) for _, p in self.params}),
                      gpu_allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 2), card_used_gib=used,
                      card_total_gib=total, adapter_groups=loaded.get("adapter_groups"))
        torch.cuda.empty_cache()
        loaded["load_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        self.loaded = loaded
        self.rec.save(loaded=loaded)

    def make_optimizer(self, lr):
        params = [p for _, p in self.params]
        if self.args.optimizer == "adamw32":  # rank 16 with full 32-bit Adam
            import torch

            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # ---- windows ----

    def attempt_window(self, spec, seed=0, release=True):
        """One window of a recorded attempt (``rollout``: its rollout.jsonl; ``w``: its weight; the heaviest
        window, the most reply tokens): ``rl_job.attempt_example`` on the compacted rows."""
        import torch

        from rl_rollouts import compact

        rows = []
        with open(spec["rollout"]) as file:
            for line in file:
                row = json.loads(line)
                if row.get("top") is not None:
                    row = compact(row)
                # a restored attempt's scripted prefix (arena.restore) is context only: weight 0, never trained
                row["w"] = 0.0 if row.get("scripted") else float(spec.get("w", 1.0))
                rows.append(row)
        made = rl_job.attempt_example(lambda capture: self.encoded(capture, pair=True), {"replies": rows},
                                      window=int(spec.get("window") or self.args.window), rng=random.Random(seed), device="cuda:0",
                                      think_end=self.think_end, end=spec.get("end"),
                                      floor=int(spec.get("floor") or 16384), margin=int(spec.get("margin") or 8192),
                                      credited_only=bool(spec.get("credited_only")))
        if made is not None:
            made["name"] = spec.get("name") or Path(spec["rollout"]).parent.name
            made["temperature"] = float(next((r["sampling"]["temperature"] for r in rows if r.get("sampling")), 0.7))
            if spec.get("hint"):
                made["teacher"] = self.teacher_window(made, rows, spec["hint"])
        if release:
            torch.cuda.empty_cache()
        return made

    def teacher_window(self, made, rows, hint):
        """OPD: the window as the hinted teacher sees it: the end reply's request with ``hint`` at the
        end of the system message, encoded as the server would; the window's tokens after the hint must be the
        student's token for token. Returns the window dict for ``window_pass`` (the student's positions shifted by
        the hint's length, the student's sampling sets) or ``{"error": ...}``."""
        import torch

        from rl_rollouts import captures, hint_messages

        request = made["replies"][-1]["request"]
        capture = next((c for c in captures(rows) if c["request"] == request), None)
        if capture is None:
            return {"error": "end capture not found"}
        hinted = {**capture, "payload": {**capture["payload"],
                                         "messages": hint_messages(capture["payload"]["messages"], hint)}}
        inputs, _ = fq.encode_request(self.processor, hinted)
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        student, teacher = made["inputs"]["input_ids"][0], inputs["input_ids"][0]
        shift = teacher.numel() - student.numel()
        positions = made["arrays"]["positions"]
        keep = student.numel() - int(positions.min()) + 1
        if shift <= 0 or not bool(torch.equal(teacher[-keep:], student[-keep:])):
            return {"error": f"teacher tokens do not align (shift {shift})"}
        width = min(student.numel(), teacher.numel())
        differ = (student[:width] != teacher[:width]).nonzero()
        first = int(differ[0]) if differ.numel() else width
        cut = student.numel() - made["tail"]
        cut = cut + shift if cut > first else cut
        arrays = {"total": teacher.numel(), "positions": positions + shift, "support": made["arrays"]["support"],
                  "reference": torch.full_like(made["arrays"]["reference"], float("nan"))}
        return {"inputs": inputs, "tail": teacher.numel() - cut, "arrays": arrays, "temperature": made["temperature"],
                "name": made["name"] + ":teacher", "shift": shift, "hint_at": first}

    def prefix_key(self, inputs, cut):
        """What a window's no-gradient prefix depends on besides its token ids: the pixels of the images that start
        before ``cut`` (count, patch rows, two checksums)."""
        import torch

        if inputs.get("pixel_values") is None:
            return (cut, 0)
        ids = inputs["input_ids"][0, :cut]
        token = self.processor.image_token_id
        runs = int(((ids == token) & torch.nn.functional.pad(ids[:-1] != token, (1, 0), value=True)).sum())
        rows = int(inputs["image_grid_thw"][:runs].prod(dim=-1).sum()) if runs else 0
        pixels = inputs["pixel_values"][:rows]
        return (cut, runs, rows, round(float(torch.sum(pixels, dtype=torch.float64)), 3),
                round(float(torch.sum(pixels[::7].abs(), dtype=torch.float64)), 3))

    def prefix_pass(self, ids, cut, piece):
        """The tokens before ``cut`` through the model without gradients, in chunks, into a cache; a chunk above
        16,384 that runs out of memory falls back to 16,384 for the rest of the run."""
        import torch

        chunk = int(min(getattr(self, "prefix_chunk", 16384), getattr(self, "prefix_cap", 1 << 30)))
        while True:
            cache, failed = None, False
            self.model.eval()
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for start in range(0, cut, chunk):
                        stop = min(start + chunk, cut)
                        output = self.model(input_ids=ids[:, start:stop], attention_mask=torch.ones_like(ids[:, :stop]),
                                            past_key_values=cache, use_cache=True, logits_to_keep=1, **piece(start, stop))
                        cache = output.past_key_values
                        del output
            except torch.cuda.OutOfMemoryError:
                if chunk <= 16384:
                    raise
                failed = True
            finally:
                self.model.train()
            if not failed:
                return cache, chunk
            cache = output = None  # freed outside the handler (its traceback holds the frames)
            torch.cuda.empty_cache()
            chunk = self.prefix_cap = 16384

    def window_pass(self, made, *, scale=None, vocab=None, shared=None):
        """The cached-prefix pass of one window; with ``scale`` the weighted loss is backpropagated. Returns per
        sampled token the actor and raw log-probabilities, validity and timings. ``shared`` (a dict kept across the
        windows of one optimizer step): a window whose prefix (ids and images before its cut) equals the previous
        one's reads that cache instead of computing it again (the cache is read-only in the window pass)."""
        import torch

        inputs, tail, arrays = made["inputs"], made["tail"], made["arrays"]
        ids = inputs["input_ids"]
        total = ids.shape[1]
        cut = total - tail
        positions = arrays["positions"]
        keep = positions - cut - 1
        if bool((keep < 0).any()):
            raise RuntimeError("a trained token lies outside the gradient window")
        tokens = ids[0, positions]
        piece = fq.positions_and_images(self.model, inputs)
        torch.cuda.synchronize()
        began, cache, reused, chunk = time.time(), None, False, None
        phase = torch.profiler.record_function("phase:prefix")
        phase.__enter__()
        key = self.prefix_key(inputs, cut) if shared is not None and cut > 0 else None
        if key is not None and shared.get("key") == key and bool(torch.equal(shared["ids"], ids[:, :cut])):
            cache, reused = shared["cache"], True
        else:
            if shared is not None:
                shared.clear()  # the previous prefix's cache goes before this one is built
            if cut > 0:
                cache, chunk = self.prefix_pass(ids, cut, piece)
            if key is not None:
                shared.update(key=key, ids=ids[:, :cut], cache=cache)
        from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

        modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
        torch.cuda.synchronize()
        phase.__exit__(None, None, None)
        prefix = time.time() - began
        began = time.time()
        grad = scale is not None
        if not grad:
            self.model.eval()
        phase = torch.profiler.record_function("phase:forward")
        phase.__enter__()
        try:
            with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids=ids[:, cut:], attention_mask=torch.ones_like(ids),
                                    past_key_values=fq.read_only(cache) if cache is not None else None,
                                    use_cache=False, logits_to_keep=keep, **piece(cut, total)).logits[0]
                actor, valid = rl_job.actor_logprobs(logits, tokens, arrays["support"], made["temperature"])
                raw = rl_job.raw_logprobs(logits.detach(), tokens)
                dist = None if vocab is None else vocab_top(logits.detach(), at_ids=vocab)
                loss = None
                tis = None
                signed = None
                opd = None
                if grad and "opd_teacher" in arrays:
                    # OPD: per sampled token the reverse-KL advantage log q_teacher - log p_student (both the actor's
                    # tempered policy on the student's sampling set; k1, as GKD / Thinking Machines on-policy
                    # distillation), clipped to +-opd_clip; trainer-over-sampler ratio capped at tis_cap
                    gap = arrays["opd_teacher"] - actor.detach()
                    valid = valid & arrays["opd_valid"] & torch.isfinite(gap)
                    gap = torch.where(valid, gap, torch.zeros_like(gap))
                    weights = gap.clamp(-arrays["opd_clip"], arrays["opd_clip"]).float()
                    ratio = torch.exp(actor.detach() - arrays["reference"]).nan_to_num(1.0, 1.0, 1.0)
                    tis = ratio.clamp(max=self.args.tis_cap or 2.0)
                    loss = -(weights[valid] * tis[valid] * actor[valid]).sum() * scale
                    opd = {"reverse_kl": round(float(-gap[valid].mean()), 5) if bool(valid.any()) else None,
                           "up_share": round(float((gap[valid] > 0).float().mean()), 4) if bool(valid.any()) else None,
                           "clipped": round(float((gap[valid].abs() > arrays["opd_clip"]).float().mean()), 5)
                           if bool(valid.any()) else None, "tokens": int(valid.sum())}
                elif grad:
                    weights = arrays["weights"] if "weights" in arrays else torch.ones_like(actor)
                    weights = weights.float()
                    if self.args.tis_cap and "reference" in arrays:  # fit's synthetic windows carry none (rl1-loop-v1)
                        # trainer over sampler on the tempered, truncated policy (1 where the engine value is missing):
                        # positive weights x min(ratio, cap) (truncated IS, Yao et al. 2025); negative weights x
                        # clip(ratio, 0, 1) (TOPR, Le Roux et al. 2025: the push-down term stays bounded)
                        ratio = torch.exp(actor.detach() - arrays["reference"]).nan_to_num(1.0, 1.0, 1.0)
                        tis = torch.where(weights < 0, ratio.clamp(0.0, 1.0), ratio.clamp(max=self.args.tis_cap))
                        down = (weights < 0) & valid
                        if bool(down.any()):
                            signed = {"down_tokens": int(down.sum()), "down_clipped": round(float((ratio[down] > 1).float().mean()), 5)}
                        weights = weights * tis
                    loss = -(weights[valid] * actor[valid]).sum() * scale
                if grad and opd is None:
                    if signed is not None or bool((weights[valid] > 0).any()):
                        w = weights[valid]
                        signed = {**(signed or {}), "up": round(float(w.clamp(min=0).sum()), 4),
                                  "down": round(float((-w).clamp(min=0).sum()), 4)}
        finally:
            if not grad:
                self.model.train()
        torch.cuda.synchronize()
        phase.__exit__(None, None, None)
        forward = time.time() - began
        backward = 0.0
        if grad:
            began = time.time()
            with torch.profiler.record_function("phase:backward"):
                loss.backward()
                torch.cuda.synchronize()
            backward = time.time() - began
        out = {"actor": actor.detach().float().cpu(), "raw": raw.float().cpu(), "valid": valid.cpu(),
               "dist": dist,  # vocab_top: already on the host
               "signed": signed, "opd": opd,
               "tis": None if tis is None else {"mean": round(float(tis[valid].mean()), 4),
                                                "capped": round(float((tis[valid] >= self.args.tis_cap).float().mean()), 5)},
               "loss": None if loss is None else float(loss.detach()), "prefix_seconds": round(prefix, 2),
               "prefix_reused": reused, "prefix_chunk": chunk, "cut": cut,
               "forward_seconds": round(forward, 2), "backward_seconds": round(backward, 2), "tokens": total,
               "tail": tail, "targets": int(positions.numel())}
        del logits, cache
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
                "mean": round(float(diff.mean()), 5), "beyond_half": round(float((diff.abs() > 0.5).float().mean()), 5),
                "engine_nll": round(float(-engine[keep].mean()), 4), "trainer_nll": round(float(-out["raw"][keep].mean()), 4)}

    # ---- operations ----

    def op_fit(self, cmd):
        import torch

        result = {"modes": {}}
        text = rl_text(self.tokenizer, int(cmd.get("perturb_tokens", 20480)), salt=1)
        began = time.time()
        set_checkpointing(self.model, "full")
        with Sampler() as sampler:
            made = text_made(text, window=int(cmd.get("perturb_window", 16384)))
            loss = self.window_pass(made, scale=1.0 / made["arrays"]["positions"].numel())["loss"]
            for group in self.optimizer.param_groups:
                group["lr"] = float(cmd.get("lr", 1e-3))
            torch.nn.utils.clip_grad_norm_([p for _, p in self.params], 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=False)  # grads stay allocated: an update holds them
        result["perturb"] = {"loss": round(loss, 5), "seconds": round(time.time() - began, 1), **sampler.summary(),
                             "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
        if cmd.get("save"):
            result["perturbed_adapter"] = self.save_adapter(cmd["save"])
        trials = cmd.get("trials") or [[mode, int(cmd.get("window", 32768))]
                                        for mode in cmd.get("modes", ["off", "alternate", "full"])]
        longest = max(w for _, w in trials)
        long_text = rl_text(self.tokenizer, max(int(cmd.get("total", 40960)), longest + 8192), salt=2)
        _, card_total, _ = gpu_used_gib()
        fitted_modes = set()
        for mode, window in trials:
            if cmd.get("first_fit") and mode in fitted_modes:
                continue  # a descending window list per mode: the first that fits is the longest
            set_checkpointing(self.model, mode)
            torch.cuda.empty_cache()
            fq.reset_peaks()
            row = {"mode": mode, "window": window}
            began = time.time()
            try:
                with Sampler() as sampler:
                    made = text_made(long_text, window=window)
                    out = self.window_pass(made, scale=1.0 / made["arrays"]["positions"].numel())
                    row["loss"] = round(out["loss"], 5)
                    row.update({k: out[k] for k in ("prefix_seconds", "forward_seconds", "backward_seconds")})
                    del made, out
                row.update(seconds=round(time.time() - began, 1), **sampler.summary(),
                           torch_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           torch_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
                row["margin_gib"] = round(card_total - max(row["card_peak_gib"], row["torch_reserved_gib"]), 2)
                fitted_modes.add(mode)
            except torch.OutOfMemoryError as error:
                row.update(oom=True, error=str(error)[:300], seconds=round(time.time() - began, 1))
            self.optimizer.zero_grad(set_to_none=False)
            torch.cuda.empty_cache()
            result["modes"][f"{mode}@{window}"] = row
            self.rec.event("fit_mode", **row)
        select = int(cmd.get("select_window", cmd.get("window", 32768)))
        fitting = [(row["seconds"], row["mode"]) for row in result["modes"].values()
                   if row["window"] == select and not row.get("oom")
                   and row.get("margin_gib", 0) >= float(cmd.get("margin_gib", 10))]
        self.checkpointing = min(fitting)[1] if fitting else None
        result["chosen"] = self.checkpointing
        set_checkpointing(self.model, self.checkpointing or "full")
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        result["ok"] = self.checkpointing is not None
        return result

    def op_split_windows(self, cmd):
        """Play = train split (rl1-loop-v2 scored bare token ids, without the prompts' images, on both sides): per
        window the last reply's served request (payload, images included), the trainer's own rendering of its
        prompt, and per earlier reply token of the window (all inside that prompt) the trainer's raw log-probability
        from the image-aware window pass and the engine's sampling-time value."""
        from rl_rollouts import captures

        out = []
        for spec in cmd["windows"]:
            made = self.attempt_window(spec)
            if made is None:
                continue
            last = made["replies"][-1]["request"]
            capture = next(c for c in captures(spec["rollout"]) if c["request"] == last)
            row = split_inputs(made, capture, self.processor.image_token_id)
            raw = self.window_pass(made)["raw"]
            row["trainer"] = [float(raw[i]) for i in row.pop("keep")]
            out.append(row)
        return {"windows": out}

    def op_score_ids(self, cmd):
        import torch

        out = []
        disable = cmd.get("adapter") == "off"
        kept = {}
        if cmd.get("zero_experts"):  # the dense part alone: expert B factors zeroed for this call
            with torch.no_grad():
                for name, param in self.params:
                    if ".experts." in name and "lora_B" in name:
                        kept[name] = param.detach().to("cpu", copy=True)
                        param.zero_()
        self.model.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                context = self.model.disable_adapter() if disable else _null()
                with context:
                    for ids in cmd["sequences"]:
                        tensor = torch.tensor([ids], device="cuda:0")
                        logits = self.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                            use_cache=False, logits_to_keep=torch.arange(len(ids) - 1, device="cuda:0")
                                            ).logits[0]
                        out.append([round(v, 5) for v in rl_job.raw_logprobs(logits, tensor[0, 1:]).tolist()])
                        del logits
        finally:
            self.model.train()
            with torch.no_grad():
                for name, param in self.params:
                    if name in kept:
                        param.copy_(kept[name].to(param.device))
        torch.cuda.empty_cache()
        return {"logprobs": out}

    def op_topk(self, cmd):
        """Healing teacher (SFT arm, 16 Oct; ``heal_kd.py`` reads the files): per segment (``rollout``) each drain
        chain's last request (served prompt with its images, then the reply) through the base model (adapter off) in
        ``chunk``-token pieces against a growing cache; per position the top-``k`` log-probabilities at temperature 1
        and the tail's log-mass, saved with the ids, the reply-token mask and the image inputs. Stops after
        ``max_tokens``. ``mtp_from`` (a checkpoint folder with ``mtp.*``): the port's check, heal_kd.MtpHead (BF16) on
        the teacher's own hyper-connection states; its draft top-1 against the teacher's top-1 per position (greedy
        acceptance) and against the recorded token. ``adapter: "on"``: the adapted model in place of the base (the
        served draft acceptance of an adapter); ``save: false``: no files (the check alone)."""
        import torch

        head = box = handle = embed = None
        if cmd.get("mtp_from"):
            import heal_kd

            head = heal_kd.load_mtp(heal_kd.MtpHead(self.model.config.text_config), cmd["mtp_from"], "cuda:0").eval()
            box, handle = heal_kd.hc_capture(self.model)
            embed = next(m for n, m in self.model.named_modules() if n.endswith("language_model.embed_tokens"))
            lm_head = self.model.get_output_embeddings()
            agree = {"teacher": 0, "token": 0, "rows": 0, "reply_teacher": 0, "reply_rows": 0}
        out_dir = Path(cmd["out"])
        out_dir.mkdir(parents=True, exist_ok=True)
        k, chunk, budget = int(cmd.get("k", 32)), int(cmd.get("chunk", 8192)), int(cmd.get("max_tokens") or 1 << 62)
        # host tensors: the producer thread makes no CUDA call, which would fail while a kernel autotuner captures a
        # CUDA graph on the main thread (smoke v3, 20 Oct: "operation not permitted when stream is capturing")
        encode = lambda capture: self.encoded(capture, pair=True, device=None)  # noqa: E731
        rows, tokens, forward, began = [], 0, 0.0, time.time()
        if os.environ.get("RL1_PIN_QUACK") == "1":  # quack loads with the kernels at the first forward: patch per op
            self.rec.event("quack_pinned", classes=pin_quack_autotune())
        # host work off the GPU's path (v4: 308 of 540 s): a thread encodes the next requests (images, chat
        # template) while the GPU runs, another writes the files
        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor

        saver, stop_producer, ready = ThreadPoolExecutor(max_workers=2), threading.Event(), queue.Queue(maxsize=3)

        def produce():
            try:
                for spec in cmd["segments"]:
                    found = rl_job.rollouts().captures(spec["rollout"])
                    found = [c for c in found if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]
                    for end in (rl_job.plan_windows(found, 1 << 40) if found else [])[::-1]:
                        if stop_producer.is_set():
                            return
                        ready.put((spec, end, rl_job.window_example(encode, found, end, window=1 << 40, floor=0,
                                                                    margin=0)))
            except Exception as error:  # surfaces in the consumer
                ready.put(error)
            finally:
                ready.put(None)

        def drain():
            while True:
                item = ready.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item

        threading.Thread(target=produce, daemon=True).start()
        prefetched = drain()
        self.model.eval()
        try:
            for spec, end, made in prefetched:
                if tokens >= budget:
                    break
                if made is None:
                    rows.append({"segment": spec["name"], "end": end, "skipped": "too long or not the served prompt"})
                    continue
                inputs, _, spans = made
                inputs = {key: value.to("cuda:0") if torch.is_tensor(value) else value for key, value in inputs.items()}
                ids = inputs["input_ids"]
                total = ids.shape[1]
                reply = torch.zeros(total, dtype=torch.bool)
                for _, start, stop in spans:
                    reply[start:stop] = True
                piece = fq.positions_and_images(self.model, inputs)
                torch.cuda.synchronize()
                since = time.time()
                top_ids, top_logp, tok_logp, cache = [], [], [], None
                nxt = torch.cat([ids[0, 1:], ids[0, :1]])[:, None]  # row t's realised next token (last row: unused)
                drafts = torch.full((total,), -1, dtype=torch.long)
                mtp_cache = head.new_cache() if head is not None else None
                adapter = _null() if cmd.get("adapter") == "on" else self.model.disable_adapter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), adapter:
                    for start in range(0, total, chunk):
                        stop = min(start + chunk, total)
                        output = self.model(input_ids=ids[:, start:stop], attention_mask=torch.ones_like(ids[:, :stop]),
                                            past_key_values=cache, use_cache=stop < total, logits_to_keep=stop - start,
                                            **piece(start, stop))
                        cache = output.past_key_values
                        got = heal_kd.mtp_inputs(ids, start, stop, piece) if head is not None else None
                        if got is not None:
                            a, b, cur, positions = got
                            drafted = head(embed(cur), box["hc"][:, : b - a], positions, mtp_cache)
                            drafts[a:b] = lm_head(drafted[0]).argmax(-1).cpu()
                            del drafted
                        top = vocab_top(output.logits[0], k=k, at_ids=(nxt[start:stop],))
                        top_ids.append(top["ids"].int())
                        top_logp.append(top["logp"])
                        tok_logp.append(top["at"][0][:, 0])
                        del output, top
                torch.cuda.synchronize()
                seconds = time.time() - since
                del cache
                top_logp, tok_logp = torch.cat(top_logp), torch.cat(tok_logp)
                tok_logp[-1] = float("nan")
                if head is not None:
                    teacher_top = torch.cat(top_ids)[:, 0].long()
                    rows_ok = drafts[:-1] >= 0
                    rows_ok &= ids[0, 1:].cpu() != self.processor.image_token_id
                    on_reply = rows_ok & reply[1:]
                    same = drafts[:-1] == teacher_top[:-1]
                    agree["rows"] += int(rows_ok.sum())
                    agree["teacher"] += int((same & rows_ok).sum())
                    agree["token"] += int(((drafts[:-1] == ids[0, 1:].cpu()) & rows_ok).sum())
                    agree["reply_rows"] += int(on_reply.sum())
                    agree["reply_teacher"] += int((same & on_reply).sum())
                mass = top_logp.double().exp().sum(-1).clamp(max=1 - 1e-6)
                name = f"{spec['name']}--e{end:03d}"
                saver.submit(torch.save if cmd.get("save", True) else (lambda *a: None), {"ids": ids[0].int().cpu(), "top_ids": torch.cat(top_ids), "top_logp": top_logp.half(),
                            "tok_logp": tok_logp.float(), "tail": torch.log1p(-mass).half(), "reply": reply,
                            "image": (ids[0] == self.processor.image_token_id).cpu(),
                            **{key: inputs[key].cpu() for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
                               if inputs.get(key) is not None},
                            "segment": spec["name"], "end": end, "w": float(spec.get("w", 1.0))}, out_dir / f"{name}.pt")
                tokens, forward = tokens + total, forward + seconds
                rows.append({"segment": spec["name"], "end": end, "file": name + ".pt", "tokens": total,
                             "reply_tokens": int(reply.sum()), "seconds": round(seconds, 2),
                             "topk_mass_mean": round(float(mass.mean()), 4), "peak_gib": fq.device_gib(peak=True)})
                self.rec.event("window", segment=spec["name"], tokens=total, seconds=round(seconds, 2),
                               peak_gib=rows[-1]["peak_gib"])  # the peak per card against the window length
                fq.reset_peaks()
                if os.environ.get("RL1_PIN_QUACK") == "1" and not getattr(self, "_quack_done", False):
                    found = pin_quack_autotune()  # quack's autotuner module appears with the first forward
                    if found:
                        self._quack_done = True
                        self.rec.event("quack_pinned", classes=found, after_window=spec["name"])
                torch.cuda.empty_cache()
            general = cmd.get("general")  # the general-text KL probe: non-ARC windows (wikitext-103 test)
            if general:
                result_general = self.general_windows(general, k)
        finally:
            stop_producer.set()
            while not ready.empty():  # unblock a producer waiting on a full queue
                ready.get_nowait()
            saver.shutdown(wait=True)
            self.model.train()
            if handle is not None:
                handle.remove()
        result = {"sequences": sum("file" in r for r in rows), "tokens": tokens, "forward_seconds": round(forward, 1),
                  "seconds": round(time.time() - began, 1), "tok_per_s": round(tokens / forward, 1) if forward else None,
                  "rows": rows}
        if cmd.get("general"):
            result["general"] = result_general
        if head is not None:
            result["mtp_check"] = {"rows": agree["rows"], "top1_vs_teacher": round(agree["teacher"] / max(agree["rows"], 1), 4),
                                   "top1_vs_token": round(agree["token"] / max(agree["rows"], 1), 4),
                                   "reply_top1_vs_teacher": round(agree["reply_teacher"] / max(agree["reply_rows"], 1), 4)}
            del head
        (out_dir / "index.json").write_text(json.dumps(result, indent=1))
        return result

    def experts_fp8(self):
        """Routed experts to W8A16 in place (RTN; ``ExpertsFP8``), with the load peak, the quantize peak, the Triton
        dequant check and its time per forward."""
        import torch

        load_peak = fq.device_gib(peak=True)  # the BF16 load's peak (before the experts shrink)
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        fp8 = ExpertsFP8(self.model)
        fp8.report.update(load_peak_gib=load_peak, quantize_peak_gib=fq.device_gib(peak=True),
                          dequant_exact=fp8.check(), dequant_ms_per_forward=fp8.bench(),
                          allocated_gib=fq.device_gib(peak=False),
                          total_gib=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
                          if torch.cuda.is_available() else None)
        self.rec.event("experts_fp8", **fp8.report)
        return fp8

    def op_moe_check(self, cmd):
        """Diagnosis (opdS-c1/c2 verify failures, 23 Oct): each listed layer's MoE block on two rows sets run together
        (MultiForward's rendezvous: one call on their concatenation) vs one call each, on the trainer's own model
        (FP8 experts and quack pin as configured). cmd sizes [[n1, n2], ...], layers, seed. Returns max |diff| per
        (layer, sizes), relative to the outputs' max."""
        import torch

        if cmd.get("experts_fp8") and getattr(self, "_fp8", None) is None:
            self._fp8 = self.experts_fp8()
        dev = next(self.model.parameters()).device if not torch.cuda.is_available() else torch.device("cuda:0")
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        layers = base.model.language_model.layers
        width = base.model.language_model.config.hidden_size
        gen = torch.Generator(device="cpu").manual_seed(int(cmd.get("seed", 0)))
        out = []
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        import contextlib  # the stream forward's autocast (MultiForward._one): BF16 activations, FP32 router weights
        cast = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()
        with torch.no_grad(), cast:
            for li in cmd.get("layers") or [0, len(layers) // 2, len(layers) - 1]:
                block = layers[int(li)].mlp
                for n1, n2 in cmd.get("sizes") or [[8192, 8192], [1489, 8192], [4096, 4096], [8192, 0]]:
                    x1 = torch.randn(1, int(n1), width, generator=gen).to(dev, dtype)
                    x2 = torch.randn(1, int(n2), width, generator=gen).to(dev, dtype) if n2 else None
                    y1 = block(x1)
                    y2 = block(x2) if x2 is not None else None
                    y1b = block(x1)  # repeatability of the single call
                    cat = block(torch.cat([x1, x2], dim=-2)) if x2 is not None else block(x1)
                    c1 = cat[:, :int(n1)]
                    scale = float(y1.float().abs().max().clamp_min(1e-12))
                    row = {"layer": int(li), "sizes": [int(n1), int(n2)], "out_max": round(scale, 4),
                           "repeat_rel": round(float((y1b.float() - y1.float()).abs().max()) / scale, 6),
                           "cat_rel_1": round(float((c1.float() - y1.float()).abs().max()) / scale, 6)}
                    if x2 is not None:
                        c2 = cat[:, int(n1):]
                        row["cat_rel_2"] = round(float((c2.float() - y2.float()).abs().max())
                                                 / float(y2.float().abs().max().clamp_min(1e-12)), 6)
                        bad = ((c2.float() - y2.float()).abs().amax(-1) > 0.05 * float(y2.float().abs().max()))[0]
                        row["bad_rows_2"] = int(bad.sum())
                        row["first_bad_row_2"] = int(bad.nonzero()[0, 0]) if bad.any() else None
                    out.append(row)
                    self.rec.event("moe_check", **row)
                    del x1, x2, y1, y2, y1b, cat
        return {"rows": out}

    def op_stream_topk(self, cmd):
        """Streaming teacher (PTH streaming OPD, 22 Oct; ``heal_stream``): tails ``rollouts``/**/rollout.jsonl while
        the student plays and writes scored ``chunk``-token pieces of each session's drain chain to ``out`` (top-``k``
        log-probabilities, the realised token's, the engine's), keeping each chain's cache between pieces; forward
        in ``sub``-token slices. Runs until ``stop_file`` exists (or ``max_tokens`` scored). Caches of idle chains are
        dropped (recomputed on their next piece) while any card holds more than ``cache_frac`` of its memory."""
        import torch

        import heal_stream

        k, chunk, sub = int(cmd.get("k", 128)), int(cmd.get("chunk", 32768)), int(cmd.get("sub", 8192))
        image_id = self.processor.image_token_id
        slack = float(cmd.get("prompt_slack", STREAM_PROMPT_SLACK))
        encode = lambda capture: self.encoded(capture, pair=True, device=None, slack=slack)  # noqa: E731 - host tensors
        if int(cmd.get("encode_procs", os.environ.get("RL1_ENCODE_PROCS", "1"))):  # teacher host fix: no GIL contention
            encode = heal_stream.ProcessEncoder(fq.stream_encode_init, fq.stream_encode, workers=int(cmd.get("encoders", 4)),
                                                args=(self.args.model, slack, self.args.window))
        adapter_on = cmd.get("adapter") == "on"
        quack = None
        if os.environ.get("RL1_PIN_QUACK") == "1":  # no autotune in the job: untuned defaults, a tune raises
            try:
                import quack.autotuner  # noqa: F401 - pip's copy (the hook patches it on import)
            except Exception:  # noqa: BLE001 - none here
                pass
            patched, scan = install_quack_fix()
            quack = {"scan": scan, "seen": set(patched)}
            self.rec.event("quack_fixed", modules=sorted(patched), when="start")

        # pipelined two-card forward (MFU-2, 23 Oct): cmd pipeline or RL1_STREAM_PIPE=1; the adapter context is held
        # for the whole run (its stage-A worker runs between pieces)
        pipe = bool(cmd.get("pipeline", os.environ.get("RL1_STREAM_PIPE") == "1")) and \
            torch.cuda.device_count() > 1 and getattr(self.args, "spread", 1) > 1
        timing = {}
        # one-card teacher (MFU-2, 23 Oct): routed experts W8A16 (cmd experts_fp8 / RL1_EXPERTS_FP8=1; RTN at the first
        # call, dense stays BF16) and pieces of ``chains`` distinct chains per forward (cmd chains / RL1_STREAM_CHAINS)
        fp8 = None
        if cmd.get("experts_fp8", os.environ.get("RL1_EXPERTS_FP8") == "1"):
            if getattr(self, "_fp8", None) is None:
                self._fp8 = self.experts_fp8()
            fp8 = self._fp8.report
        chains = int(cmd.get("chains") or os.environ.get("RL1_STREAM_CHAINS") or 1)
        if chains > 1:  # MultiForward chains 2 fails the GPU stream verify (top-1 0.94-0.96, 23 Oct): disabled
            raise RuntimeError("stream chains > 1 is disabled (MultiForward wrong on GPU, 23 Oct); use chains 1")
        if "fla_pin" in cmd:  # diagnosis: the fla chunk_fwd_kernel_o pin per call
            import fla_pin
            self.rec.event("fla_pinned", config=fla_pin.pin() if cmd["fla_pin"] else fla_pin.unpin(), when="call")
        if pipe:
            forward = stream_forward_pipe(self.model, sub, image_id, timing=timing)
        elif chains > 1:
            if getattr(self, "_multi", None) is None:
                self._multi = MultiForward(self.model, sub, image_id, chains=max(8, chains))
            self._multi.chains = chains  # chains per step (the pool holds the most)
            if "chain_lock" in cmd:  # diagnosis (opdS-c1 verify failure, 23 Oct): the lock per call
                self._multi.lock = threading.Lock() if cmd["chain_lock"] else None
            forward = self._multi
        else:
            forward = stream_forward(self.model, sub, image_id,
                                     adapter=None if adapter_on else self.model.disable_adapter)

        def top_fn(logits, nxt):  # on the device: the stream teacher copies a piece's rows once (no per-slice sync)
            got = vocab_top(logits, k=k, at_ids=(nxt,), device=True)
            return {"ids": got["ids"], "logp": got["logp"], "at": got["at"][0]}

        frac = float(cmd.get("cache_frac", 0.8))
        totals = {d: torch.cuda.get_device_properties(d).total_memory for d in range(torch.cuda.device_count())}
        # bytes (c7m 24 Oct: the 0.8 bound alone left one piece's forward + top-k to OOM at 263 GiB): room for the
        # largest measured piece rise x 1.25, at least RL1_TEACHER_RESERVE_GIB (64: c7m's OOM piece rose >= 50 GiB)
        bytes_budget = heal_stream.ByteBudget(totals, torch.cuda.memory_allocated, torch.cuda.max_memory_allocated,
                                              torch.cuda.reset_peak_memory_stats, frac=frac,
                                              reserve=float(os.environ.get("RL1_TEACHER_RESERVE_GIB") or 64) * 2 ** 30)
        over = bytes_budget.over

        def log(row):
            row = dict(row)
            self.rec.event(row.pop("event"), **row)

        budget_hook = heal_stream.lru_budget(over, first=heal_stream.is_closing  # c7r: closing chains first
                                             if os.environ.get("HEAL_EVICT_CLOSING_FIRST") == "1" else None)
        budget_hook.before, budget_hook.after = bytes_budget.before, bytes_budget.after
        snap_gib = float(os.environ.get("RL1_MEM_SNAPSHOT_GIB") or 0)
        if snap_gib:  # allocation sites, dumped once when the teacher passes snap_gib
            torch.cuda.memory._record_memory_history(max_entries=200000)
            snap = {"done": False}

            def after_snap():
                need = bytes_budget.after()
                if not snap["done"] and torch.cuda.memory_allocated() > snap_gib * 2 ** 30:
                    snap["done"] = True
                    path = Path(cmd["out"]).parent / "teacher-mem-snapshot.pickle"
                    torch.cuda.memory._dump_snapshot(str(path))
                    self.rec.event("mem_snapshot", path=str(path), allocated_gib=round(torch.cuda.memory_allocated() / 2 ** 30, 2))
                return need

            budget_hook.after = after_snap
        teacher = heal_stream.StreamTeacher(
            cmd["rollouts"], cmd["out"], encode, forward, k=k, chunk=chunk, image_token_id=image_id,
            idle=float(cmd.get("idle", 900)), poll=float(cmd.get("poll", 2.0)), encoders=int(cmd.get("encoders", 4)),
            log=log, budget=budget_hook, top_fn=top_fn, stop_file=cmd.get("stop_file"),
            base=[x for x in str(cmd.get("base") or "").split(",") if x], base_share=float(cmd.get("base_share") or 0),
            end_id=self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), batch=chains if not pipe else 1)
        budget = cmd.get("max_tokens")
        check = {"after": int(cmd.get("verify_after", 0)), "result": None,
                 "bar": cmd.get("verify_bar") or ({"kl": float(os.environ["HEAL_VERIFY_KL"])} if os.environ.get("HEAL_VERIFY_KL")
                                                  else {"top_logp_mean_abs": 0.01, "top1_agree": 0.99})}

        prof = {"at": int(cmd.get("profile_after", 3)), "out": cmd.get("profile_out"), "p": None}

        def profile(t):
            """cmd profile_out: a torch profile of the 2 pieces after piece ``profile_after`` (kernel time per card
            over the wall, the top host ops, a Chrome trace)."""
            if not prof["out"] or prof["p"] is False:
                return
            n = t.stats["pieces"]
            if prof["p"] is None and n == prof["at"]:
                prof["p"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA])
                prof["p"].__enter__()
                prof["t0"], prof["n0"] = time.time(), t.stats["tokens"]
            elif prof["p"] and n == prof["at"] + 2:
                torch.cuda.synchronize()
                p, prof["p"] = prof["p"], False
                p.__exit__(None, None, None)
                wall = time.time() - prof["t0"]
                out = Path(prof["out"])
                out.mkdir(parents=True, exist_ok=True)
                p.export_chrome_trace(str(out / "trace.json"))
                spans = {}  # kernel intervals per card (union: the time the card ran anything)
                for e in json.loads((out / "trace.json").read_text()).get("traceEvents", []):
                    if e.get("cat") == "kernel" and "dur" in e:
                        spans.setdefault(str((e.get("args") or {}).get("device")), []).append((e["ts"], e["ts"] + e["dur"]))
                busy = {}
                for card, iv in spans.items():
                    iv.sort()
                    total, end = 0.0, -1.0
                    for a, b in iv:
                        if b > end:
                            total += b - max(a, end)
                            end = b
                    busy[card] = round(total / 1e6, 2)
                rows = sorted(p.key_averages(), key=lambda r: -r.self_cpu_time_total)[:25]
                summary = {"wall_seconds": round(wall, 2), "tokens": t.stats["tokens"] - prof["n0"],
                           "kernel_seconds_by_card": busy,
                           "top_host_ops": [{"name": r.key, "count": r.count,
                                             "self_cpu_s": round(r.self_cpu_time_total / 1e6, 3),
                                             "cpu_total_s": round(r.cpu_time_total / 1e6, 3)} for r in rows]}
                (out / "profile.json").write_text(json.dumps(summary, indent=1))
                self.rec.event("stream_profile", **{k2: v for k2, v in summary.items() if k2 != "top_host_ops"})

        def after_piece(t, chain):
            profile(t)
            if quack is not None:  # each quack copy patched as it loaded (the vendored one: at the first forward)
                now = quack["scan"]()
                if set(now) != quack["seen"]:
                    quack["seen"] = set(now)
                    self.rec.event("quack_fixed", modules=sorted(now), when=f"piece {t.stats['pieces']}")
                if os.environ.get("RL1_QUACK_REQUIRED") == "1" and not any(".quack." in "." + m for m in now):
                    raise RuntimeError(f"no quack autotuner patched after the first piece: {sorted(now)}")
            # first phase: one chain's streamed pieces against a fresh one-pass score, and (c7j, 24 Oct) the first
            # student chain's too (its pieces continue at capture offsets, never slice-aligned); a miss stops the run
            if not check["after"] or t.stats["pieces"] < check["after"] or len(chain.files) < 2:
                return
            student = not t._is_base(chain)
            if check["result"] is not None and (not student or check.get("student") is not None):
                return
            got = t.verify(chain)  # HEAL_VERIFY_SLICING (c7k: pieces = the stored cuts, exact: storage/pipeline faults)
            got["student"] = student
            if got.get("slicing") != "sub":  # cross-cut KL (one fresh call): bf16 shape noise, monitored, never a gate
                cross = t.verify(chain, slicing="sub", worst=8)
                self.rec.event("stream_verify_crosscut", student=student, **{x: y for x, y in cross.items()})
            if check["result"] is None:
                check["result"] = got
            if student:
                check["student"] = got
            bar = check["bar"]
            if "kl" in bar:  # HEAL_VERIFY_KL (c7j v7): k x the measured bf16 call-shape floor, KD-support KL
                got["ok"] = got["kl_mean"] is not None and got["kl_mean"] <= bar["kl"]
            else:
                got["ok"] = got["top_logp_mean_abs"] <= bar["top_logp_mean_abs"] and got["top1_agree"] >= bar["top1_agree"]
            self.rec.event("stream_verify", **got)
            if not got["ok"]:
                if cmd.get("verify_exit", True):
                    raise SystemExit(f"stream teacher verify failed: {got}")
                raise RuntimeError(f"stream teacher verify failed: {got}")  # diagnosis: the process stays up

        if cmd.get("skip_existing"):
            teacher.skip_existing()
        self.model.eval()
        import contextlib
        held = self.model.disable_adapter() if (pipe or chains > 1) and not adapter_on else contextlib.nullcontext()
        began = time.time()
        try:
            with held:
                result = teacher.run(until=(lambda t: t.stats["tokens"] >= int(budget)) if budget else None,
                                     status_every=float(cmd.get("status_every", 60)), after_piece=after_piece)
            result["verify"] = check["result"]
            result["verify_student"] = check.get("student")
            result["wall_seconds"] = round(time.time() - began, 1)
            result["pipeline"] = {"on": pipe, **({"split": forward.split, **timing} if pipe else {})}
            result["chains"], result["experts_fp8"] = chains, fp8
        finally:
            if pipe:
                forward.close()
            self.model.train()
            torch.cuda.empty_cache()
        result["peak_gib"] = fq.device_gib(peak=True)
        Path(cmd["out"], "stream-teacher.json").write_text(json.dumps(result, indent=1))
        return result

    def _stream_verify_diag(self, sv, forward, image_id):
        """``sv``: rollouts (root of <run>/<game path>/rollout.jsonl), chain (key; default the first multi-piece one),
        pieces (stop once it has this many), chunk, k, base (runs scored as base), slicings (``StreamTeacher.verify``),
        worst. The pieces are scored exactly as op_stream_topk scores them, then verified once per slicing."""
        import torch

        import heal_stream

        k = int(sv.get("k", 128))
        TIES.update(mode=os.environ.get("RL1_QSA_TIES") or os.environ.get("HEAL_QSA_TIES") or "topk", count=False)  # as the round

        def top_fn(logits, nxt):
            got = vocab_top(logits, k=k, at_ids=(nxt,), device=True)
            return {"ids": got["ids"], "logp": got["logp"], "at": got["at"][0]}

        slack = float(sv.get("prompt_slack", STREAM_PROMPT_SLACK))
        encode = lambda capture: self.encoded(capture, pair=True, device=None, slack=slack)  # noqa: E731
        t = heal_stream.StreamTeacher(sv["rollouts"], sv.get("out", "/tmp/stream-verify-diag"), encode, forward, k=k,
                                      chunk=int(sv.get("chunk", 16384)), image_token_id=image_id, idle=1e9, encoders=2,
                                      top_fn=top_fn, base=list(sv.get("base") or []),
                                      end_id=self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        want, key = int(sv.get("pieces", 3)), sv.get("chain")

        def target():
            multi = [c for c in t.chains.values() if (c.key == key if key else len(c.files) >= 2)]
            return max(multi, key=lambda c: len(c.files)) if multi else None

        began, closed = time.time(), False
        for _ in range(100000):
            t._scan()
            while not t.events.empty():
                t._handle(t.events.get_nowait())
            c = target()
            if c is not None and len(c.files) >= want:
                break
            chain, upto = t._pick()
            if chain is None:
                if any(tail.backlog for tail in t.tails.values()):
                    continue
                if closed:  # every session read and its remainder scored
                    break
                for name in list(t.chains):  # the recording is complete: close each chain (its remainder is due)
                    t._handle(("idle", name, None, None))
                closed = True
                continue
            t.score(chain, upto)
            t._evict(chain)
        t.saver.shutdown(wait=True)
        chain = target()
        if chain is None or len(chain.files) < 2:
            return {"stream_verify": "no multi-piece chain", "chains": {c.key: len(c.files) for c in t.chains.values()}}
        out = {"chain": chain.key, "pieces": len(chain.files), "score_seconds": round(time.time() - began, 1),
               "piece_tokens": [int(torch.load(Path(t.out) / f"{name}.pt", weights_only=False)["ids"].numel())
                                for name, _ in chain.files], "verify": []}
        for slicing in sv.get("slicings", ["pieces", "safe", "sub", "sub"]):
            got = t.verify(chain, slicing=slicing, worst=int(sv.get("worst", 32)))
            self.rec.event("stream_verify_diag", **{x: y for x, y in got.items() if x != "worst"})
            out["verify"].append(got)
        if sv.get("length_diff"):  # k000 causality probe: rows under calls of several ends (+ later content reversed)
            ld = dict(sv["length_diff"])
            if not torch.cuda.is_available():
                ld.update(device="cpu", autocast=False)
            n = out["piece_tokens"][-1]
            got = length_diff(self.model, heal_stream.prefix_inputs(chain.inputs, n, image_id), image_id, **ld)
            self.rec.event("length_diff", chain=chain.key, first_over=got["first_over"], perturb=got.get("perturb"))
            out["length_diff"] = got
        return {"stream_verify": out}

    def op_stream_check(self, cmd):
        """c7j diagnosis (24 Oct): the stream teacher's cached continuation against a fresh pass on real sequences
        (``files``: op_topk / stream teacher files, their ids, images and reply mask), the forward exactly as
        op_stream_topk builds it (one card, W8A16 experts when RL1_EXPERTS_FP8, adapter off, ``sub`` slices); cuts
        per ``heal_stream.continuation_cuts`` near each of ``near``; ``fla_pin``: [True, False] runs both GDN pins."""
        import torch

        import heal_stream

        sub, image_id = int(cmd.get("sub", 8192)), self.processor.image_token_id
        if os.environ.get("RL1_PIN_QUACK") == "1":
            install_quack_fix()
        if os.environ.get("RL1_EXPERTS_FP8") == "1" and getattr(self, "_fp8", None) is None:
            self._fp8 = self.experts_fp8()
        self.model.eval()  # as op_stream_topk: train mode + gradient checkpointing turns use_cache off (no cache)
        forward = stream_forward(self.model, sub, image_id, adapter=self.model.disable_adapter)
        ratio = int(getattr(self.model.config.get_text_config(), "indexer_compress_ratio", 4) or 4)
        results = []
        if cmd.get("stream_verify"):  # k000 diagnosis (24 Oct): the stream teacher itself on a recorded chain, verified per slicing
            results.append(self._stream_verify_diag(dict(cmd["stream_verify"]), forward, image_id))
        for path in cmd["files"]:
            d = torch.load(path, map_location="cpu", weights_only=True)
            inputs = {"input_ids": d["ids"].long()[None]}
            for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
                if key in d:
                    inputs[key] = d[key]
            if "pixel_values" in d and "mm_token_type_ids" not in inputs:
                inputs["mm_token_type_ids"] = (inputs["input_ids"] == image_id).long()
            reply = d["reply"].bool()
            limit = heal_stream.cut_point(inputs["input_ids"][0], int(cmd.get("limit", 40000)), 0, image_id)
            inputs = heal_stream.prefix_inputs(inputs, limit, image_id)
            if cmd.get("length_diff") is not None:  # v5: rows [r0, r1) under single calls of several lengths, per module
                TIES.update(mode=cmd.get("ties", ["topk"])[0], count=False)
                ld = dict(cmd["length_diff"])
                if not torch.cuda.is_available():
                    ld.update(device="cpu", autocast=False)
                got = length_diff(self.model, inputs, image_id, **ld)
                self.rec.event("length_diff", file=Path(path).name, first_over=got["first_over"], logp=got.get("logp"))
                results.append({"file": Path(path).name, "length_diff": got})
                if not cmd.get("with_cuts"):  # v7: both reads in one load
                    continue
            for near in cmd.get("near", [16384]):
                cuts = heal_stream.continuation_cuts(inputs["input_ids"][0], reply[:limit], image_id, near=int(near), ratio=ratio)
                # v3: explicit offsets past the aligned cut (kept off images), a one-slice reference and other sub sizes
                runs = heal_stream.image_runs(inputs["input_ids"][0], image_id)
                for off in cmd.get("offsets", []):
                    c = cuts.get("aligned", int(near)) + int(off)
                    if 0 < c < limit - 1 and not any(a <= c < b for a, b in runs):
                        cuts[f"off{int(off)}"] = c
                ref = stream_forward(self.model, 10 ** 9, image_id, adapter=self.model.disable_adapter) \
                    if cmd.get("ref_single") else None
                fresh = {f"sub{int(x)}": stream_forward(self.model, int(x), image_id, adapter=self.model.disable_adapter)
                         for x in cmd.get("subs", [])}
                if cmd.get("state_check"):  # v3: the carried state after [0, c), sliced vs one pass, per GDN kernel set
                    TIES.update(mode=cmd.get("ties", ["topk"])[0], count=False)
                    from transformers.models.qwen4_exp import modeling_qwen4_exp as _mk
                    hub = {n: getattr(_mk, n) for n in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")}
                    single = stream_forward(self.model, 10 ** 9, image_id, adapter=self.model.disable_adapter)
                    for kern in cmd.get("kernels", ["hub"]):  # hub | torch_gdn | torch_conv | torch_both
                        for n, f in hub.items():
                            torch_way = kern == "torch_both" or (kern == "torch_gdn" and "delta" in n) or \
                                (kern == "torch_conv" and "conv" in n)
                            setattr(_mk, n, getattr(f, "__wrapped__", f) if torch_way else f)
                        try:
                            st = heal_stream.state_check(forward, single, inputs, cuts if kern == "hub" else
                                                         {k: v for k, v in cuts.items() if k in ("off1", "off64")})
                        except Exception as exc:  # noqa: BLE001 - a diagnosis leg: the logit checks still run
                            st = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                        finally:
                            for n, f in hub.items():
                                setattr(_mk, n, f)
                        self.rec.event("state_check", file=Path(path).name, near=near, kernels=kern, **st)
                        results.append({"file": Path(path).name, "near": near, "kernels": kern, "state_check": st})
                for pin, ties, idx, att in [(p_, t_, i_, a_) for t_ in cmd.get("ties", ["topk"])
                                            for p_ in cmd.get("fla_pin", [True]) for i_ in cmd.get("indexer", ["vectorized"])
                                            for a_ in cmd.get("attention", [SPARSE["mode"]])]:
                    TIES.update(mode=ties, count=True, rows=0, tied=0, zero=0)
                    SPARSE["mode"] = att  # v3: masked (the teacher's default) vs triton (RL's trainer)
                    from transformers.models.qwen4_exp import modeling_qwen4_exp as _m
                    _cls = _m.Qwen4ExpTextQSAIndexer  # v3: the HF loop indexer vs our vectorized one
                    if getattr(_cls, "_rl1_vectorized", False):
                        _cls.forward = _cls._rl1_original if idx == "loop" else _cls._rl1_vectorized_forward
                    try:
                        import fla_pin
                        fla_pin.pin() if pin else fla_pin.unpin()
                    except Exception:  # noqa: BLE001 - CPU: no fla
                        pass
                    t0 = time.time()
                    win = cmd.get("kl_window")  # v7: full-vocab KL vs the reference on rows [a, b) (+ reply-only)
                    got = heal_stream.continuation_check(forward, inputs, cuts, ref_forward=ref, fresh=fresh,
                                                         kl_window=tuple(win) if win else None, reply=reply[:limit])
                    row = {"file": Path(path).name, "near": near, "fla_pin": pin, "ties": ties, "indexer": idx, "attention": att,
                           "ref_single": bool(ref),
                           "tie_rows": {k: TIES[k] for k in ("rows", "tied", "zero")},
                           "images": int((inputs["input_ids"][0] == image_id).sum()),
                           "seconds": round(time.time() - t0, 1), **got}
                    self.rec.event("stream_check", **row)
                    results.append(row)
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        TIES.update(mode=os.environ.get("RL1_QSA_TIES") or os.environ.get("HEAL_QSA_TIES") or "topk", count=False)
        SPARSE["mode"] = os.environ.get("RL1_ATTENTION", "masked")
        return {"results": results}

    def op_text_drift(self, cmd):
        """Untrained-text drift (posttrain-practice 13.14): per window of ``files`` (``.pt`` with ``ids``, the heal
        general-text windows), base (adapter off) and adapter: mean NLL of the next token and mean KL(base || adapter)
        per position, over the first ``count`` files."""
        import torch

        files = sorted(Path(cmd["dir"]).glob("*.pt"))[: int(cmd.get("count", 16))]
        nll_base = nll_adapter = kl = 0.0
        positions = 0
        self.model.eval()
        try:
            for path in files:
                ids = torch.load(path, map_location="cpu")["ids"].long()[None].to("cuda:0")
                logps = []
                for on in (False, True):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), \
                            (_null() if on else self.model.disable_adapter()):
                        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                            logits_to_keep=ids.shape[1]).logits[0, :-1]
                    logps.append(torch.log_softmax(logits.float(), -1))
                    del logits
                target = ids[0, 1:]
                base, adapted = logps
                nll_base -= float(base.gather(-1, target[:, None]).sum())
                nll_adapter -= float(adapted.gather(-1, target[:, None]).sum())
                kl += float((base.exp() * (base - adapted)).sum())
                positions += target.numel()
                del logps, base, adapted
                torch.cuda.empty_cache()
        finally:
            self.model.train()
        n = max(positions, 1)
        return {"files": len(files), "positions": positions, "nll_base": round(nll_base / n, 5),
                "nll_adapter": round(nll_adapter / n, 5), "nll_delta": round((nll_adapter - nll_base) / n, 5),
                "kl_base_adapter": round(kl / n, 6)}

    def op_channel_energy(self, cmd):
        """Width-cut scores (SFT arm, 17 Oct): per decoder layer, the mean square of each hidden channel of the residual
        stream (the layer's hc output folded over its copies), over ``sequences`` (token ids; the excl5 usage set),
        base model (adapter off), in ``chunk``-token pieces against a growing cache. Saved to ``out`` as a [layers,
        hidden] float64 tensor with the token count."""
        import torch

        text = self.model.config.text_config
        width, count = text.hidden_size, text.hc_count
        layers = next(m for n, m in self.model.named_modules() if n.endswith("language_model.layers"))
        sums = torch.zeros(len(layers) + 1, width, dtype=torch.float64, device="cuda:0")
        handles = []
        for i, layer in enumerate(layers):
            def hook(module, args, output, i=i):
                x = (output if torch.is_tensor(output) else output[0]).detach().float()
                sums[i] += x.pow(2).reshape(-1, count, width).sum(dim=(0, 1)).double()
            handles.append(layer.register_forward_hook(hook))
        mixer = next(m for n, m in self.model.named_modules() if n.endswith("language_model.hyper_connection_mixer"))
        def mixed(module, args, output):  # returns None: a forward hook's return value replaces the output (v4)
            sums[-1] += output.detach().float().pow(2).reshape(-1, width).sum(0).double()

        handles.append(mixer.register_forward_hook(mixed))
        chunk, tokens, began = int(cmd.get("chunk", 8192)), 0, time.time()
        self.model.eval()
        try:
            for seq in cmd["sequences"][: cmd.get("limit")]:
                ids = torch.tensor([seq], device="cuda:0")
                cache = None
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), self.model.disable_adapter():
                    for start in range(0, ids.shape[1], chunk):
                        stop = min(start + chunk, ids.shape[1])
                        out = self.model(input_ids=ids[:, start:stop], attention_mask=torch.ones_like(ids[:, :stop]),
                                         past_key_values=cache, use_cache=stop < ids.shape[1], logits_to_keep=1)
                        cache = out.past_key_values
                        del out
                tokens += ids.shape[1]
                del cache
        finally:
            for h in handles:
                h.remove()
            self.model.train()
        torch.save({"mean_square": (sums / max(tokens, 1)).cpu(), "tokens": tokens,
                    "layers": "decoder outputs (hc copies folded), then the final mixer"}, cmd["out"])
        return {"tokens": tokens, "seconds": round(time.time() - began, 1), "out": cmd["out"]}

    def general_windows(self, general, k):
        """``general["windows"]`` windows of ``general["tokens"]`` tokens of wikitext-103 (test split, joined), each
        through the base model alone; files as ``op_topk``'s (no reply positions), in ``general["out"]``."""
        import torch
        from datasets import load_dataset

        out_dir = Path(general["out"])
        out_dir.mkdir(parents=True, exist_ok=True)
        width, count = int(general.get("tokens", 2048)), int(general.get("windows", 256))
        rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"]
        ids_all = self.tokenizer("".join(rows))["input_ids"]
        done, began = 0, time.time()
        for index in range(count):
            piece = ids_all[index * width:(index + 1) * width]
            if len(piece) < width:
                break
            ids = torch.tensor([piece], device="cuda:0")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), self.model.disable_adapter():
                logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                    logits_to_keep=width).logits[0]
            top = vocab_top(logits, k=k)
            del logits
            mass = top["logp"].double().exp().sum(-1).clamp(max=1 - 1e-6)
            none = torch.zeros(width, dtype=torch.bool)
            torch.save({"ids": ids[0].int().cpu(), "top_ids": top["ids"].int(), "top_logp": top["logp"].half(),
                        "tail": torch.log1p(-mass).half(), "reply": none, "image": none, "segment": f"general-{index:03d}",
                        "end": 0, "w": 1.0}, out_dir / f"general-{index:03d}.pt")
            done += 1
        return {"windows": done, "tokens": done * width, "seconds": round(time.time() - began, 1),
                "source": "Salesforce/wikitext wikitext-103-raw-v1 test"}

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

    def score_windows(self, specs, label=None):
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
                   "mismatch": self.mismatch(made, out), "seconds": out["prefix_seconds"] + out["forward_seconds"]}
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
            rows[name] = row
            del made
        return rows

    def op_indexer(self, cmd):
        """Switch the QSA indexer between the pinned per-query loop and the vectorized selection (speed check)."""
        from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

        vectorized_indexer()
        cls = modeling.Qwen4ExpTextQSAIndexer
        cls.forward = cls._rl1_original if cmd["mode"] == "loop" else cls._rl1_vectorized_forward
        return {"mode": cmd["mode"], "calls": dict(INDEXER)}

    def op_attention(self, cmd):
        """Switch the full-attention layers between selected-key attention and the masked SDPA (equality check)."""
        selected_attention()
        SPARSE["mode"] = cmd["mode"]
        SPARSE["chunk"] = int(cmd.get("chunk", SPARSE["chunk"]))
        SPARSE["density"] = bool(cmd.get("density", SPARSE["density"]))
        return {"mode": cmd["mode"], "sparse": dict(SPARSE), "indexer": dict(INDEXER)}

    def op_score(self, cmd):
        return {"windows": self.score_windows(cmd["windows"], cmd.get("label"))}

    def adapter_norm(self):
        import torch

        with torch.no_grad():  # one host sync, not one per tensor
            return round(float(torch.stack([p.detach().float().pow(2).sum() for _, p in self.params]).sum().sqrt()), 5)

    def expand(self, specs):
        """A spec with ``whole`` becomes one spec per planned window end that holds weighted replies (every token of a
        long continuation trained, each window once)."""
        out = []
        for spec in specs:
            if not spec.get("whole"):
                out.append(spec)
                continue
            rows = []
            with open(spec["rollout"]) as file:
                for line in file:
                    row = json.loads(line)
                    row["w"] = 0.0 if row.get("scripted") else float(spec.get("w", 1.0))
                    rows.append(row)
            ends = rl_job.credited_ends({"replies": rows}, int(spec.get("window") or self.args.window))
            name = spec.get("name") or Path(spec["rollout"]).parent.name
            out += [{**spec, "whole": False, "end": end, "name": f"{name}#{i}", "first_window": end == min(ends)}
                    for i, end in enumerate(ends)]
        return out

    def op_update(self, cmd):
        import torch

        if not getattr(self, "pinned", 0):  # fla's kernel modules can load at the first forward: scan until found
            self.pinned = len(pin_autotune_keys())
        result = {"windows": [], "errors": [], "autotune_pinned": self.pinned, "autotune_pinned_late": sorted(set(PINNED_LATE))}
        lr = float(cmd["lr"])
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        specs = [a for a in self.expand(cmd["attempts"]) if a["w"] > 0 or (cmd.get("signed") and a["w"] < 0)]
        opd = cmd.get("opd")  # {"hint", "clip"}: on-policy distillation from the hinted current policy
        if opd:
            specs = [{**a, "hint": opd["hint"]} for a in specs]
        shared, shared_teacher = None, ({} if opd and cmd.get("share_prefix") else None)
        if cmd.get("share_prefix"):  # the batch's order, grouped by start; each try's window from its start first
            groups = {}
            for a in specs:
                groups.setdefault(a.get("share"), len(groups))
            specs = sorted(specs, key=lambda a: (groups[a.get("share")], not a.get("first_window", True)))
            specs = specs[: int(cmd.get("max_windows", 8))]
            shared = {}
            self.prefix_chunk = int(cmd.get("prefix_chunk") or 16384)
        else:
            specs = sorted(specs, key=lambda a: -abs(a["w"]))[: int(cmd.get("max_windows", 8))]
        per_try = {}
        if cmd.get("norm") in ("try", "cap"):  # windows per try (``expand`` names them try#i)
            for a in specs:
                key = a.get("name", "").split("#")[0]
                per_try[key] = per_try.get(key, 0) + 1
        fq.reset_peaks()
        losses, finite, tokens = 0.0, True, 0
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(1)  # the next window is encoded (CPU: template, images) while the GPU runs this one
        upcoming = pool.submit(self.attempt_window, specs[0], 0, False) if specs else None
        with Sampler() as sampler:
            began = time.time()
            waits = 0.0
            for number, spec in enumerate(specs):  # one window in flight: a window's images are gigabytes on the card
                since = time.time()
                made = upcoming.result()
                waits += time.time() - since
                upcoming = (pool.submit(self.attempt_window, specs[number + 1], 0, False)
                            if number + 1 < len(specs) else None)
                if made is None:
                    result["errors"].append({"name": spec.get("name"), "error": "no window fits"})
                    continue
                teacher_row = None
                if opd:
                    teacher = made.pop("teacher", None) or {"error": "no teacher window"}
                    if "error" in teacher:
                        result["errors"].append({"name": spec.get("name"), "error": teacher["error"]})
                        del made, teacher
                        continue
                    seen = self.window_pass(teacher, shared=shared_teacher)
                    made["arrays"].update(opd_teacher=seen["actor"].to("cuda:0"), opd_valid=seen["valid"].to("cuda:0"),
                                          opd_clip=float(opd.get("clip", 5.0)))
                    teacher_row = {"shift": teacher["shift"], "prefix_reused": seen["prefix_reused"],
                                   "seconds": round(seen["prefix_seconds"] + seen["forward_seconds"], 1)}
                    del teacher, seen
                scale = 1.0  # summed; the gradient is divided by the tokens below
                if cmd.get("norm") == "cap":  # Dr. GRPO: each try's summed log-likelihood over a constant (the cap)
                    scale = 1.0 / float(cmd["norm_cap"])
                elif per_try:  # each try counts once: its windows' mean weighted log-likelihood (divided by tries below)
                    scale = 1.0 / (max(1, int(made["arrays"]["positions"].numel())) * per_try[spec.get("name", "").split("#")[0]])
                out = self.window_pass(made, scale=scale, shared=shared)
                tokens += out["targets"]
                losses += out["loss"]
                finite &= math.isfinite(out["loss"]) and bool(torch.isfinite(out["raw"]).all())
                result["windows"].append({"name": made["name"], "tokens": out["tokens"], "tail": out["tail"],
                                          "targets": out["targets"], "trained": int(out["valid"].sum()),
                                          "w": float(made["arrays"]["weights"][0]) if "weights" in made["arrays"] else 1.0,
                                          "mismatch": self.mismatch(made, out),
                                          "sampled_nll_trainer": round(float(-out["raw"].mean()), 4), "tis": out["tis"],
                                          "signed": out["signed"], "opd": out["opd"], "teacher": teacher_row,
                                          "prefix_seconds": out["prefix_seconds"],
                                          "prefix_reused": out["prefix_reused"], "cut": out["cut"],
                                          "seconds": round(out["prefix_seconds"] + out["forward_seconds"]
                                                           + out["backward_seconds"], 1)})
                del made, out
            pool.shutdown()
            if shared_teacher is not None:
                shared_teacher.clear()
            if shared is not None:
                shared.clear()  # the next step's weights differ: no cache outlives its step
                self.prefix_chunk = 16384  # probes and scores keep the chunking their floor was measured with
                result["prefix_reused"] = sum(bool(w.get("prefix_reused")) for w in result["windows"])
            result["window_wait_seconds"] = round(waits, 1)
            if not tokens:
                self.optimizer.zero_grad(set_to_none=True)
                result.update(ok=False, error="no trained tokens")
                return result
            divisor = len(per_try) if per_try else tokens
            with torch.no_grad():
                for _, param in self.params:
                    if param.grad is not None:
                        param.grad.div_(divisor)
            losses /= divisor
            grads = [p.grad for _, p in self.params if p.grad is not None]
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in self.params], float(cmd.get("clip", 1.0))))
            finite &= math.isfinite(norm)
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
                change = float(torch.stack([(p.detach() - b).float().pow(2).sum() for (_, p), b in zip(self.params, before)]).sum().sqrt())
            del before
            result["step_seconds"] = round(time.time() - stepped, 1)
            result["train_seconds"] = round(time.time() - began, 1)
        self.updates += 1
        norm_after = self.adapter_norm()
        result["expert_lora_calls"] = dict(GROUPED)  # cumulative since load: grouped vs loop, per dtype pair
        result.update(sampler.summary(), update_norm=round(change, 6), adapter_norm=norm_after, applied=True,
                      torch_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), update=self.updates)
        if not math.isfinite(norm_after):
            result.update(ok=False, stop="nan_or_inf")
            return result
        torch.cuda.empty_cache()
        if cmd.get("save"):
            result["adapter"] = self.save_adapter(cmd["save"])
        if cmd.get("probes"):
            began = time.time()
            result["probes"] = self.score_windows(cmd["probes"])
            result["probe_seconds"] = round(time.time() - began, 1)
        result["ok"] = True
        return result

    def save_adapter(self, folder, only=None):
        import torch
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        began = time.time()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in get_peft_model_state_dict(self.model).items()
                 if only is None or (only == "dense") != (".experts." in k)}
        save_file(state, str(folder / "adapter_model.safetensors"))
        self.model.peft_config["default"].save_pretrained(str(folder))
        config = json.loads((folder / "adapter_config.json").read_text())
        config.pop("nvfp4_merge_aware", None)
        config["inference_mode"] = True
        (folder / "adapter_config.json").write_text(json.dumps(config, indent=1))
        return {"path": str(folder), "tensors": len(state), "gib": round(sum(t.numel() * 2 for t in state.values()) / 2**30, 2),
                "seconds": round(time.time() - began, 1)}

    def op_profile(self, cmd):
        """One weighted window pass (``spec``: a recorded attempt, else a text window of ``window`` tokens) under the
        torch profiler (``warm``: the same pass once before, unprofiled: autotuning): wall time per phase and the
        card's kernel time by phase (prefix, forward, backward with the checkpoint recompute) and by module (hooks
        on the attention, its SDPA call, the indexer, the GDN, the experts, router, shared MLP, PLE, residual
        mixers; a backward kernel outside them by its autograd node)."""
        import torch
        from torch.profiler import ProfilerActivity, profile

        if cmd.get("mode"):
            set_checkpointing(self.model, cmd["mode"])
        made = (self.attempt_window(cmd["spec"]) if cmd.get("spec")
                else text_made(rl_text(self.tokenizer, int(cmd.get("window", 24576)) + 8192, salt=3),
                               window=int(cmd.get("window", 24576))))
        scale = 1.0 / made["arrays"]["positions"].numel()
        warm = None
        if cmd.get("warm"):
            out = self.window_pass(made, scale=scale)
            warm = {k: out[k] for k in ("prefix_seconds", "forward_seconds", "backward_seconds")}
            self.optimizer.zero_grad(set_to_none=True)
            del out
        undo = annotate_modules(self.model)
        try:
            torch.cuda.synchronize()
            began = time.time()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                out = self.window_pass(made, scale=scale)
                torch.cuda.synchronize()
            wall = time.time() - began
        finally:
            undo()
        self.optimizer.zero_grad(set_to_none=True)
        set_checkpointing(self.model, self.checkpointing or "full")
        began = time.time()
        split = kernel_split(prof.events())
        events = prof.key_averages()

        def device(e):
            return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0

        kernel = sum(device(e) for e in events) / 1e6
        rows = lambda key: [{"name": e.key[:120], "count": e.count, "device_s": round(device(e) / 1e6, 2),
                             "cpu_s": round(e.self_cpu_time_total / 1e6, 2)}
                            for e in sorted(events, key=key, reverse=True)[:25]]
        result = {"name": made.get("name"), "wall_seconds": round(wall, 1), "kernel_seconds": round(kernel, 1),
                  "busy_share": round(kernel / wall, 3) if wall else None, "tokens": out["tokens"], "tail": out["tail"],
                  "cut": out["cut"], "targets": out["targets"], "unprofiled": warm,
                  **{k: out[k] for k in ("prefix_seconds", "forward_seconds", "backward_seconds")},
                  "launches": sum(e.count for e in events if device(e) > 0), "split": split,
                  "analysis_seconds": round(time.time() - began, 1), "expert_lora_calls": dict(GROUPED),
                  "top_device": rows(device), "top_cpu": rows(lambda e: e.self_cpu_time_total)}
        del made, out, prof
        torch.cuda.empty_cache()
        return result

    # ---- play: the adapter master and the optimizer state wait in pinned host memory ----

    def held(self):
        """(key, holder, field) of every adapter parameter and optimizer-state tensor."""
        import torch

        for name, param in self.params:
            yield ("param", name), param, None
            state = self.optimizer.state.get(param) or {}
            for field in [f for f, v in state.items() if torch.is_tensor(v)]:
                yield ("state", name, field), state, field

    def op_park(self, cmd=None):
        import torch

        if self.parked:
            return {"parked": True, "already": True}
        torch.cuda.synchronize()
        began, moved, swaps = time.time(), 0, []
        for name, param in self.params:
            param.grad = None
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
        return {"parked": True, "gib": round(moved / 2**30, 2), "seconds": round(time.time() - began, 2),
                "tensors": len(swaps), "card_used_gib": used}

    def unpark(self):
        import torch

        began, moved = time.time(), 0
        for key, holder, field in self.held():
            value = holder.data if field is None else holder[field]
            if value.is_cuda or not value.is_pinned():
                continue
            moved_value = value.to("cuda:0", non_blocking=True)
            moved += value.numel() * value.element_size()
            if field is None:
                holder.data = moved_value
            else:
                holder[field] = moved_value
        torch.cuda.synchronize()
        self.parked = False
        return {"gib": round(moved / 2**30, 2), "seconds": round(time.time() - began, 2)}

    def op_unpark(self, cmd=None):
        return self.unpark() if self.parked else {"already": True}

    def op_compare_lora(self, cmd):
        """The grouped expert LoRA against Axolotl's per-expert loop on one recorded window: seconds of the weighted
        pass each way, the gradient's cosine and relative difference, and the log-probability difference."""
        made = self.attempt_window(cmd["spec"])
        out, grads = {}, {}
        try:
            self._compare(made, out, grads)
        finally:
            GROUPED_ON["on"] = True  # rl1-ttest-v5: an error left the loop on for the next op
        return self._compared(out, grads)

    def _compare(self, made, out, grads):
        import torch

        for path in ("loop", "grouped"):
            GROUPED_ON["on"] = path == "grouped"
            before = dict(GROUPED)
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            began = time.time()
            result = self.window_pass(made, scale=1.0 / made["arrays"]["positions"].numel())
            torch.cuda.synchronize()
            out[path] = {"seconds": round(time.time() - began, 1),
                         **{k: result[k] for k in ("prefix_seconds", "forward_seconds", "backward_seconds", "loss")},
                         "calls": {k: GROUPED.get(k, 0) - before.get(k, 0) for k in GROUPED}}
            grads[path] = [p.grad.detach().float().cpu() if p.grad is not None else None for _, p in self.params]
            out[path]["actor"] = result["actor"]
        del made

    def _compared(self, out, grads):
        import torch

        self.optimizer.zero_grad(set_to_none=True)
        dot = sum(float((a * b).sum()) for a, b in zip(grads["loop"], grads["grouped"]) if a is not None and b is not None)
        na = math.sqrt(sum(float(a.pow(2).sum()) for a in grads["loop"] if a is not None))
        nb = math.sqrt(sum(float(b.pow(2).sum()) for b in grads["grouped"] if b is not None))
        diff = math.sqrt(sum(float((a - b).pow(2).sum()) for a, b in zip(grads["loop"], grads["grouped"])
                             if a is not None and b is not None))
        actor = (out["loop"].pop("actor") - out["grouped"].pop("actor")).abs()
        del grads
        torch.cuda.empty_cache()
        return {**out, "grad_cosine": round(dot / (na * nb), 6) if na and nb else None,
                "grad_rel_diff": round(diff / na, 6) if na else None, "grad_norm_loop": round(na, 6),
                "grad_norm_grouped": round(nb, 6), "actor_abs_diff_mean": round(float(actor.mean()), 6),
                "actor_abs_diff_max": round(float(actor.max()), 6),
                "speedup": round(out["loop"]["seconds"] / out["grouped"]["seconds"], 3)}

    def op_targets(self, cmd):
        """Per window spec the trained reply tokens of its heaviest window from the served counts (no encoding):
        the loop balances push-up and push-down weight on them."""
        out = {}
        for spec in cmd["windows"]:
            window = int(spec.get("window") or self.args.window)
            with open(spec["rollout"]) as file:
                rows = [json.loads(line) for line in file]
            scripted = {r["n"] for r in rows if r.get("scripted")}
            captures = [c for c in rl_job.rollouts().captures(rows)
                        if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]
            if scripted:  # a restored segment: its live replies only (the scripted prefix is context)
                out[spec["name"]] = max((served_tokens(captures, end, window, skip=scripted)
                                         for end in range(len(captures)) if captures[end]["request"] not in scripted),
                                        default=0)
                continue
            ends = rl_job.plan_windows(captures, window) if captures else []
            out[spec["name"]] = max((served_tokens(captures, end, window) for end in ends), default=0)
        return {"targets": out}

    def op_checkpoint(self, cmd):
        """The fp32 adapter master and the optimizer state (resume a later loop from this update), to ``path``."""
        import torch
        from safetensors.torch import save_file

        began = time.time()
        folder = Path(cmd["path"])
        folder.mkdir(parents=True, exist_ok=True)
        save_file({name: p.detach().float().contiguous().cpu() for name, p in self.params}, str(folder / "master.safetensors"))
        torch.save({"optimizer": self.optimizer.state_dict(), "updates": self.updates,
                    "names": [name for name, _ in self.params]}, str(folder / "optimizer.pt"))
        return {"path": str(folder), "seconds": round(time.time() - began, 1),
                "gib": round(sum(f.stat().st_size for f in folder.iterdir()) / 2**30, 2)}

    def op_load_checkpoint(self, cmd):
        """A state ``op_checkpoint`` wrote (another run's): adapter params and optimizer state, the update count;
        the base references stay (the probes' KL to base keeps its reference)."""
        import torch
        from safetensors.torch import load_file

        began = time.time()
        folder = Path(cmd["path"])
        master = load_file(str(folder / "master.safetensors"))
        state = torch.load(str(folder / "optimizer.pt"), map_location="cpu", weights_only=False)
        names = [name for name, _ in self.params]
        if state["names"] != names or set(master) != set(names):
            raise ValueError(f"checkpoint layout differs: {len(master)} tensors, {len(names)} params")
        with torch.no_grad():
            for name, param in self.params:
                param.copy_(master[name].to(device=param.device, dtype=param.dtype))
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
        return {"path": str(folder), "updates": self.updates, "adapter_norm": self.adapter_norm(),
                "lr": self.optimizer.param_groups[0]["lr"], "seconds": round(time.time() - began, 1)}

    def op_checkpointing(self, cmd):
        """Set the activation checkpointing mode (``full``, ``alternate``, ``partial:N``, ``off``) for later updates."""
        self.checkpointing = str(cmd["mode"])
        set_checkpointing(self.model, self.checkpointing)
        return {"mode": self.checkpointing, "autotune_pinned": len(pin_autotune_keys())}

    def op_load_adapter(self, cmd):
        """A saved PEFT adapter (``save_adapter``, BF16) into the trainable params; a fresh optimizer (no state saved)."""
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        began = time.time()
        state = load_file(str(Path(cmd["path"]) / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(self.model, state)
        missing = [k for k in (getattr(result, "missing_keys", None) or []) if "lora_" in k]
        self.optimizer = self.make_optimizer(float(cmd.get("lr", self.args.lr)))
        self.optimizer.zero_grad(set_to_none=True)
        return {"path": cmd["path"], "tensors": len(state), "missing_lora": len(missing),
                "unexpected": len(getattr(result, "unexpected_keys", None) or []), "adapter_norm": self.adapter_norm(),
                "seconds": round(time.time() - began, 1)}

    def op_save(self, cmd):
        return self.save_adapter(cmd["path"], only=cmd.get("only"))

    def serve(self):
        index = 0
        alive = self.dir / "alive.json"
        alive.write_text(json.dumps({"state": "ready", "loaded": self.loaded, "time": time.time()}, default=str))
        while True:
            path = self.dir / "cmd" / f"{index:04d}.json"
            if not path.exists():
                alive.write_text(json.dumps({"state": "idle", "next": index, "time": time.time(), "loaded": self.loaded},
                                            default=str))
                time.sleep(0.1)
                continue
            cmd = json.loads(path.read_text())
            alive.write_text(json.dumps({"state": "busy", "op": cmd["op"], "n": index, "time": time.time()}))
            began = time.time()
            try:
                if cmd["op"] == "exit":
                    result = {"ok": True}
                else:
                    unparked = self.unpark() if self.parked and cmd["op"] in GPU_OPS else None
                    result = getattr(self, f"op_{cmd['op']}")(cmd)
                    result.setdefault("ok", True)
                    if unparked is not None:
                        result["unparked"] = unparked
            except Exception as error:  # the failure is the answer
                import torch

                result = {"ok": False, "error": repr(error)[:2000], "traceback": traceback.format_exc()[-6000:]}
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            result["seconds"] = round(time.time() - began, 1)
            # the play server wakes beside the idle trainer: hand every cached block back (rl1-ttest-v1: the card
            # held ~152 GiB for the trainer between updates against ~105 GiB of state)
            import torch

            torch.cuda.empty_cache()
            used, _, _ = gpu_used_gib()
            result["memory"] = {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2), "card_used_gib": used,
                                "device_peak_gib": fq.device_gib(peak=True)}
            self.rec.event("op", op=cmd["op"], n=index, **{k: v for k, v in result.items() if k != "traceback"})
            tmp = self.dir / "done" / f"{index:04d}.json.tmp"
            tmp.write_text(json.dumps(result, default=str))
            tmp.rename(self.dir / "done" / f"{index:04d}.json")
            index += 1
            if cmd["op"] == "exit":
                return


GPU_OPS = ("fit", "topk", "stream_topk", "stream_check", "text_drift", "channel_energy", "score_ids", "score", "update", "profile", "reset", "compare_lora", "split_windows", "checkpoint", "load_checkpoint", "load_adapter")  # these bring a parked adapter back first


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def rl_text(tokenizer, tokens, salt=0):
    """Deterministic game-log text of about ``tokens`` tokens, as ids."""
    rng = random.Random(1000 + salt)
    words = ("move", "left", "right", "up", "down", "wall", "door", "key", "blue", "red", "green", "grid", "cell",
             "open", "shut", "turn", "score", "level", "push", "block")
    lines, ids = [], []
    while len(ids) < tokens:
        lines.extend(f"{len(lines):06d} {rng.choice(words)} {rng.choice(words)} "
                     + " ".join(str(rng.randrange(10)) for _ in range(24)) for _ in range(400))
        ids = tokenizer("\n".join(lines))["input_ids"]
    return ids[:tokens]


def text_made(ids, *, window, share=0.5, seed=0):
    """Plain text as a window example: the loss on the last ``share`` of the window's tokens (about the share of
    reply tokens in a played window), each with a sampling set of 20 (the play sampler's top-k)."""
    import torch

    rng = torch.Generator().manual_seed(seed)
    tensor = torch.tensor([ids], device="cuda:0")
    total = tensor.shape[1]
    first = total - int(window * share)
    positions = torch.arange(first, total, device="cuda:0")
    support = torch.randint(0, 150000, (positions.numel(), 20), generator=rng).to("cuda:0")
    support[:, 0] = tensor[0, positions]
    return {"inputs": {"input_ids": tensor}, "tail": window, "temperature": 1.0, "name": "text",
            "engine_raw": [float("nan")] * positions.numel(),
            "arrays": {"positions": positions, "support": support,
                       "weights": torch.ones(positions.numel(), device="cuda:0")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="command / answer folder (and trainer records)")
    parser.add_argument("--model", required=True, help="the served NVFP4 checkpoint")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--window", type=int, default=24576)
    parser.add_argument("--optimizer", choices=("adamw8bit", "adamw32"), default="adamw8bit")
    parser.add_argument("--tis-cap", type=float, default=0.0, help="truncated importance sampling cap (0: off)")
    parser.add_argument("--loop-lora", action="store_true", help="keep Axolotl's per-expert LoRA loop (comparison)")
    parser.add_argument("--spread", type=int, default=1, help="split the decoder layers over this many visible cards")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.serve()


if __name__ == "__main__":
    main()
