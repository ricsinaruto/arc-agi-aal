# SHARED: RL (rl-async) + PTH (sft-heal), one copy since the trunk dedup (24 Oct); PTH runs it with PROFILE "pth".
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
import contextlib
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
import rl1_memory  # noqa: E402
import rl_job  # noqa: E402
import rl1_encode  # noqa: E402

TOTAL_GIB = None
CARD = "cuda:0"  # the trainer's card (the CPU window-feed test sets "cpu")


# One trainer for RL and PTH (trunk dedup, 24 Oct). PROFILE "pth" turns on PTH's unconditional heal/OPD behaviour
# (fla tile pin, autotune pins, one-sync norms, 0.1 s command poll); RL's defaults are unchanged. Set by RL1_PROFILE /
# HEAL_PROFILE or a ``trainer_profile`` file beside this module (launch/pth/build_heal.py writes "pth").
def _profile():
    marker = Path(__file__).with_name("trainer_profile")
    return (os.environ.get("RL1_PROFILE") or os.environ.get("HEAL_PROFILE")
            or (marker.read_text().strip() if marker.is_file() else "") or "rl")


PROFILE = _profile()


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


def token_entropy(rows, valid, temperature=1.0, chunk=1024):
    """(sum, count) of the policy's per-token entropy (nats, logits / temperature) over ``valid`` rows of ``rows``
    [n, V]; row chunks, no grad (dashboard collapse signal)."""
    import torch

    total, count = 0.0, 0
    with torch.no_grad():
        mask = valid.to(rows.device)
        for i in range(0, rows.shape[0], chunk):
            keep = mask[i:i + chunk]
            if not bool(keep.any()):
                continue
            lp = torch.log_softmax(rows[i:i + chunk][keep].float() / float(temperature or 1.0), dim=-1)
            total += float(-(lp.exp() * lp).sum())
            count += int(keep.sum())
            del lp
    return total, count


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


def length_diff(model, inputs, image_id, *, lengths=(64, 2048, 8192, 16384, 24000), rows=(0, 64), layers=(0, 1),
                device="cuda:0", autocast=True):
    """c7j v5: rows [r0, r1) scored by no-cache single calls [0, L) of each length L: per module of ``layers`` (forward
    hooks; 3-D [1, L, d] or 2-D [L, d] outputs) and the log-probs, the relative difference to the shortest call. A
    causal model gives the same rows whatever follows them in the call; the first module that does not is the bug."""
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
        rec.setdefault(L, {})[name] = x.detach().float().cpu()

    def post(name):
        return lambda mod, args, out: take(name, out[0] if isinstance(out, (tuple, list)) and out else out)

    handles = [lm.layers[0].register_forward_pre_hook(
        lambda mod, args, kwargs: take("layers0.input", args[0] if args else kwargs.get("hidden_states")), with_kwargs=True)]
    for li in layers:
        for name, mod in lm.layers[int(li)].named_modules():
            handles.append(mod.register_forward_hook(post(f"L{li}.{name}" if name else f"L{li}")))
    forward = stream_forward(model, 10 ** 9, image_id, device=device, autocast=autocast,
                             adapter=getattr(model, "disable_adapter", None))
    logp = {}
    try:
        for L in lengths:
            now["L"] = L
            chain = heal_stream.Chain(f"len{L}", -1, inputs)
            for a, b, logits in forward(chain, 0, L, True):
                logp[L] = torch.log_softmax(logits[r0:r1].float(), dim=-1).cpu()
            chain.cache = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    finally:
        for h in handles:
            h.remove()
    ref = lengths[0]
    out = {"lengths": lengths, "rows": [r0, r1], "ref": ref, "modules": {}, "first_over": {}}
    for L in lengths[1:]:
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


def carry_reject(carry, held, pure, cut, best, ids, route, pkey):
    """Why prefix_cache does not continue a held carry (None: it does): the first failing check, in order."""
    import torch

    if not held.get("carry_on"):
        return None if carry is None else "off"
    if pure:
        return "pure_window"
    if carry is None:
        return "none_held"
    if carry["at"] > cut:
        return "past_cut"
    if best is not None and carry["at"] <= best["at"]:
        return "snapshot_deeper"
    if carry["ids"].shape[1] > ids.shape[1] or not bool(torch.equal(carry["ids"], ids[:, :carry["at"]])):
        return "ids"
    if pkey(carry["at"]) != carry["pkey"]:
        return "images"
    if (carry["route"] is None) != (route is None):
        return "route_presence"
    if route is not None and not bool(torch.equal(carry["route"], route[:carry["at"]].to(carry["route"].device))):
        return "route"
    return None


GAP_DUMP = []  # MFU-1 sampler gap: per window engine / trainer raw log-probs of the sampled tokens (mismatch)


def windows_engine_gap(windows):
    """Token-weighted trainer-vs-sampler gap over windows' ``mismatch`` rows (engine minus trainer, sampled tokens):
    mean |diff|, mean, and the share beyond log 2 (outside a TIS cap of 2)."""
    rows = [(w["mismatch"], w["mismatch"]["tokens"]) for w in windows if (w.get("mismatch") or {}).get("tokens")]
    n = sum(t for _, t in rows)
    avg = lambda k: round(sum(m.get(k, 0.0) * t for m, t in rows) / n, 5) if n else None
    return {"tokens": n, "mean_abs": avg("mean_abs"), "mean": avg("mean"), "beyond_ln2": avg("beyond_ln2")}


def window_geom(made, out, image_token_id):
    """ref_compare's worst-row read (MFU-1 floor, 24 Oct): a window's trained row positions, cut, prefix start and
    image spans [start, stop) in its ids."""
    import torch

    ids = made["inputs"]["input_ids"][0].cpu()
    spans = []
    if image_token_id is not None:
        m = torch.nn.functional.pad((ids == int(image_token_id)).to(torch.int8), (1, 1))
        edge = (m[1:] - m[:-1]).nonzero().flatten().tolist()
        spans = [(edge[i], edge[i + 1]) for i in range(0, len(edge) - 1, 2)]
    return {"positions": made["arrays"]["positions"].cpu().long(), "cut": int(out["cut"]),
            "prefix_start": int(out.get("prefix_start", 0)), "spans": spans}


def rows_read(per, geom, step, top=40):
    """Worst trained rows against the reference and per-window means, placed against the prefix chunk grid
    (multiples of ``step``; "odd": multiples of ``step`` that are not multiples of 2 * ``step``, the cuts a chunk of
    2 * ``step`` does not make) and the images: image spans in the prefix that a grid cut splits, the distance from a
    row to the cut and to the last image before it."""
    import bisect

    import torch

    rows, windows = [], []
    for name, (d, mask) in per.items():
        g = geom.get(name)
        if g is None or g["positions"].numel() != d.numel():
            continue
        cut, spans = g["cut"], [s for s in g["spans"] if s[0] < g["cut"]]
        split = lambda odd: sum(1 for a, b in spans if any((c // step) % 2 == 1 or not odd
                                                           for c in range(-(-max(a + 1, 1) // step) * step, min(b, cut), step)))
        ends = sorted(b for _, b in g["spans"])
        valid = d[mask]
        windows.append({"name": name, "cut": cut, "n": int(valid.numel()),
                        "mean": round(float(valid.mean()), 5) if valid.numel() else None,
                        "max": round(float(valid.max()), 4) if valid.numel() else None,
                        "prefix_images": len(spans), "split": split(False), "split_odd": split(True),
                        "image_tokens_prefix": sum(min(b, cut) - a for a, b in spans)})
        pos = g["positions"]
        for i in torch.nonzero(mask).flatten().tolist():
            p = int(pos[i])
            k = bisect.bisect_right(ends, p) - 1
            rows.append((float(d[i]), name, p, p - cut, p - ends[k] if k >= 0 else None))
    rows.sort(key=lambda r: -r[0])
    with_split = [w for w in windows if w["split_odd"] and w["mean"] is not None]
    without = [w for w in windows if not w["split_odd"] and w["mean"] is not None]
    avg = lambda ws: round(sum(w["mean"] * w["n"] for w in ws) / max(1, sum(w["n"] for w in ws)), 5) if ws else None
    return {"step": step, "windows": windows, "mean_split_odd": avg(with_split), "mean_no_split_odd": avg(without),
            "worst": [{"d": round(r[0], 4), "window": r[1], "pos": r[2], "after_cut": r[3], "after_image": r[4]}
                      for r in rows[:top]]}


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


COMPILE = {"on": False, "ready": None}  # MFU stage B: compiled regions, switched per op (cmd compile_regions)


def _gated_residual(x, norm_w, down_w, up_w, inj_w, hc: int, hidden: int, eps: float):
    """Qwen4ExpTextGatedResidual.forward's math (grouped RMSNorm, low-rank mix, gates) on plain tensors, so one compiled
    graph serves every layer's instance."""
    import torch
    import torch.nn.functional as F

    xf = x.float().reshape(*x.shape[:-1], hc, hidden)
    xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2)
    xn = (xn * (1.0 + norm_w.float())).type_as(x)
    mix = F.silu(F.linear(xn, down_w) / hc)
    mix = torch.sigmoid(F.linear(mix, up_w)).unflatten(-1, (hc, hidden))
    mixed = (mix * xn.unflatten(-1, (hc, hidden))).mean(dim=-2)
    return mixed, 2 * torch.sigmoid(F.linear(xn, inj_w) / hc)


def _inject(hyper_input, hidden_states, weights):
    return hyper_input + (hidden_states.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


def _rms_norm(x, weight, eps: float):
    import torch

    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).type_as(x)


def compile_regions(model):
    """MFU stage B (21 Oct): the elementwise-bound small regions run as inductor-compiled kernels (dynamic shapes, one
    graph per region for all layers): the hyper-connection GatedResidual (31 s of the 375 s baseline step), the decoder
    layer's injection add (most of MulBackward's 19 s), the ungrouped RMSNorms. Eager stays the path unless
    COMPILE["on"]; not bitwise (fused float order), so its step is held to the numerics bar (ref compare)."""
    import inspect
    import textwrap

    import torch

    if COMPILE["ready"] is not None:
        return COMPILE["ready"]
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    gr, inj, rms = (torch.compile(f, dynamic=True) for f in (_gated_residual, _inject, _rms_norm))
    info = {"gated_residual": 0, "rms_norm": 0, "inject": 0, "skipped": []}
    for module in model.modules():
        kind = type(module).__name__
        if kind == "Qwen4ExpTextGatedResidual":
            linears = [module.input_mix_weight_down, module.input_mix_weight_up, module.block_inject_weight]
            if any(type(m) is not torch.nn.Linear or m.weight.dtype not in (torch.bfloat16, torch.float32) for m in linears):
                info["skipped"].append(kind)
                continue
            eager = module.forward

            def forward(hyper_input, module=module, eager=eager):
                if not COMPILE["on"]:
                    return eager(hyper_input)
                mixed, weights = gr(hyper_input, module.hc_norm.weight, module.input_mix_weight_down.weight,
                                    module.input_mix_weight_up.weight, module.block_inject_weight.weight,
                                    int(module.hc_count), int(module.hidden_size), float(module.hc_norm.eps))
                return mixed, hyper_input, weights

            module.forward = forward
            info["gated_residual"] += 1
        elif kind == "Qwen4ExpTextRMSNorm" and module.group_size is None:
            eager = module.forward

            def forward(x, module=module, eager=eager):
                return rms(x, module.weight, float(module.eps)) if COMPILE["on"] else eager(x)

            module.forward = forward
            info["rms_norm"] += 1
    layer_cls = next((type(m) for m in model.modules() if type(m).__name__ == "Qwen4ExpTextDecoderLayer"), None)
    if layer_cls is not None:  # the injection lines of the layer's own forward, swapped for the compiled add
        import re

        source = textwrap.dedent(inspect.getsource(layer_cls.forward))
        pair = re.compile(r"( *)injection = hidden_states\.unsqueeze\(-2\) \* injection_weights\.unsqueeze\(-1\)\n"
                          r"\s*hidden_states = hyper_input \+ injection\.flatten\(-2\)")
        if len(pair.findall(source)) == 2:
            source = pair.sub(r"\1hidden_states = (_rl1_inject if _rl1_compile['on'] else _rl1_eager_inject)"
                              r"(hyper_input, hidden_states, injection_weights)", source)
            space = dict(inspect.getmodule(layer_cls).__dict__)
            space.update(_rl1_inject=inj, _rl1_eager_inject=_inject, _rl1_compile=COMPILE)
            exec(compile(source, f"<rl1 {layer_cls.__name__}.forward>", "exec"), space)
            layer_cls.forward = space["forward"]
            info["inject"] = sum(1 for m in model.modules() if type(m) is layer_cls)
        else:
            info["skipped"].append("inject: forward source differs")
    COMPILE["ready"] = info
    return info


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
# rl1-chunk (19 Oct): "qsa" the indexer's selection as the model has it; "stable" block scores in float64 with ties
# broken by block index (no shape-dependent rounding or tie order); "dense" every causal key (no selection; diagnostic)
SELECT = {"mode": "qsa"}
# rl1-route: the MoE router's top-k experts and the QSA indexer's picks recorded in one
# pass ("capture") and forced in a later one ("replay"): a pass at other weights then runs the recorded discrete choices
# with its own continuous values (routing replay). Off: the model's own code paths, unchanged numerics.
ROUTE = {"router": "off", "qsa": "off", "rec": [], "qsa_rec": [], "i": 0, "qi": 0, "calls": 0, "qsa_calls": 0}
# live progress: window tail tokens through the model and the phase; a thread writes one row per
# PROGRESS_EVERY s to $ARC_VALIDATION_OUTPUT/util/train_progress.jsonl (the efficiency watcher and dashboard read it)
PROGRESS = {"phase": "loading", "op": None, "tail": 0, "windows": 0}
PHASES = {"old_logprobs": "old_pass", "update": "step", "checkpoint": "checkpoint", "save": "checkpoint",
          "load_checkpoint": "loading", "score": "probe", "score_ids": "probe"}


def progress_writer(every=None):
    out = os.environ.get("ARC_VALIDATION_OUTPUT")
    if not out:
        return None
    every = float(every or os.environ.get("RL1_PROGRESS_EVERY") or 60)
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
            elif ROUTE["router"] == "prefix":  # MFU-1 causal_check: a shorter call's picks on its rows, the rest free
                rec = ROUTE["rec"][ROUTE["i"]].to(device=router_probs.device, dtype=torch.long)
                ROUTE["i"] += 1
                n = min(rec.shape[0], router_indices.shape[0])
                router_indices = torch.cat([rec[:n], router_indices[n:]])
                router_top_value = router_probs.gather(-1, router_indices)
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


def served_call(start, stop):
    """The served rows of positions [start, stop) of the current window (or None)."""
    window = ROUTE["window"]
    if ROUTE["router"] != "served" or window is None:
        return None
    return window[start:stop]


CAUSAL = {"on": False, "key": None, "counts": None, "max": None, "hits": 0, "bad": 0}  # MFU-1 stage S: plain_causal memo


def plain_causal(attention_mask):
    """Per query the number of visible keys when the mask is a plain causal one (batch 1, query i sees keys
    0..n_i-1), else None. ``CAUSAL["on"]``: the answer for the same mask tensor (storage, shape, version) is reused
    by the forward's next full-attention layers and the recompute (the check reads an [L, keys] mask: 4.2G elements
    at 32K x 130K, in each of the 12 layers)."""
    if CAUSAL["on"]:
        key = (attention_mask.data_ptr(), tuple(attention_mask.shape), attention_mask.dtype, attention_mask._version)
        if CAUSAL["key"] == key:
            CAUSAL["hits"] += 1
            if SEL["verify"]:  # diagnostic: the memo against a fresh check (one sync per hit)
                import torch

                fresh = _plain_causal(attention_mask)
                same = (fresh is None) == (CAUSAL["counts"] is None) and (
                    fresh is None or torch.equal(fresh, CAUSAL["counts"]))
                CAUSAL["bad"] += not same
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


TOPK = {"calls": 0, "verified": 0, "rows": 0, "bad_rows": 0, "torch_ms": 0.0, "fast_ms": 0.0}  # MFU-1 fast_topk
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


def topk_verify(scores, valid, top, limit, rl1_topk):
    """fast_topk "verify": torch's picks (returned) and rl1_topk's on the same scores, compared as sets per row;
    both timed (synchronized; a check mode, not a speed path)."""
    import torch

    sync = torch.cuda.synchronize if scores.is_cuda else (lambda: None)
    sync()
    a = time.time()
    picked = scores.masked_fill(~valid, float("-inf")).topk(top, dim=-1).indices
    sync()
    b = time.time()
    mine = rl1_topk.topk_rows(scores, top, limit)
    sync()
    c = time.time()
    bad = (torch.sort(picked, dim=1).values != mine).any(dim=1)
    coarse = scores.to(torch.bfloat16).float().contiguous()  # the same shapes with many ties at the k-th place
    tie = (torch.sort(coarse.masked_fill(~valid, float("-inf")).topk(top, dim=-1).indices, dim=1).values
           != rl1_topk.topk_rows(coarse, top, limit)).any(dim=1)
    TOPK["tie_bad_rows"] = TOPK.get("tie_bad_rows", 0) + int(tie.sum())
    TOPK["verified"] += 1
    TOPK["rows"] += int(bad.numel())
    TOPK["bad_rows"] += int(bad.sum())
    TOPK["torch_ms"] += (b - a) * 1e3
    TOPK["fast_ms"] += (c - b) * 1e3
    return picked


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
        if known is None:  # heal_kd HEAL_MASK_HOST: the largest count, read with the check
            known = getattr(counts, "_heal_max", None)
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
            # HEAL_INDEXER_BF16 (MFU-1): the same BF16 values on tensor cores with FP32 output (the products are exact
            # in FP32 either way; only the summation order differs) instead of the SIMT FP32 GEMM
            bf16 = (os.environ.get("HEAL_INDEXER_BF16") == "1" and q.is_cuda and q.dtype == torch.bfloat16
                    and block_keys.dtype == torch.bfloat16)
            keys_b = block_keys.transpose(0, 1).contiguous() if bf16 else None
            ids = torch.arange(blocks, device=device)
            stable = SELECT["mode"] == "stable"
            if stable:
                keys_t = keys_t.double()
                tie = ids.double() * 1e-12  # exact ties (relu zeros) go to the earlier block, whatever the row width
            for begin in range(0, length, chunk):
                end = min(length, begin + chunk)
                if stable:
                    scores = torch.relu(torch.matmul(q[0, begin:end].double(), keys_t)).sum(dim=1)
                    scores = scores / math.sqrt(self.index_head_dim) - tie
                else:
                    if bf16:
                        qq = q[0, begin:end]
                        raw = torch.mm(qq.reshape(-1, qq.shape[-1]), keys_b, out_dtype=torch.float32)
                        scores = torch.relu(raw.view(end - begin, qq.shape[1], blocks)).sum(dim=1)
                    else:
                        scores = torch.relu(torch.matmul(q[0, begin:end].float(), keys_t)).sum(dim=1)
                    scores = scores / math.sqrt(self.index_head_dim)
                valid = ids[None, :] < complete[begin:end, None]
                ROUTE["qsa_calls"] += 1
                if ROUTE["qsa"] == "replay":  # rl1-route: the recorded block picks of this chunk
                    if ROUTE["qi"] >= len(ROUTE["qsa_rec"]):
                        raise RuntimeError(f"qsa replay: chunk {ROUTE['qi']} beyond the {len(ROUTE['qsa_rec'])} recorded")
                    picked = ROUTE["qsa_rec"][ROUTE["qi"]].to(torch.long)
                    ROUTE["qi"] += 1
                    if tuple(picked.shape) != (end - begin, top):
                        raise RuntimeError(f"qsa replay: recorded {tuple(picked.shape)}, pass {(end - begin, top)}")
                else:
                    picked = None
                    fast = SELECT.get("fast") if not stable else None
                    if fast:  # MFU-1 fast_topk: rl1_topk's exact per-row radix select (torch's selection, ascending)
                        import rl1_topk

                        if fast == "verify":  # the torch picks drive the pass; the radix ones are checked against them
                            picked = topk_verify(scores, valid, top, complete[begin:end], rl1_topk)
                        else:
                            picked = rl1_topk.topk_rows(scores, top, complete[begin:end])
                            TOPK["calls"] += 1
                    if picked is None:
                        # topk_unsorted (MFU-1): the union kernel sorts each tile's union blocks itself, so the picks'
                        # order is unused there; the selected set is the same either way (the sort follows the select)
                        picked = scores.masked_fill(~valid, float("-inf")).topk(top, dim=-1,
                                                                               sorted=not SELECT.get("unsorted")).indices
                    if TIES["mode"] == "index" or TIES["count"]:
                        tied = tie_by_index(scores.masked_fill(~valid, float("-inf")), picked, top)
                        picked = tied if TIES["mode"] == "index" else picked
                    if ROUTE["qsa"] == "capture":
                        ROUTE["qsa_rec"].append(picked.to(torch.int16))
                    elif ROUTE["qsa"] in ("capture_map", "prefix"):  # MFU-1 causal_check, by (layer, chunk start)
                        key = (int(self.layer_idx), begin)
                        rec = ROUTE.setdefault("qsa_map", {}).get(key) if ROUTE["qsa"] == "prefix" else None
                        if ROUTE["qsa"] == "capture_map":
                            ROUTE["qsa_map"][key] = picked.clone()
                        elif rec is not None and rec.shape[1] == picked.shape[1]:
                            n = min(rec.shape[0], picked.shape[0])
                            picked = torch.cat([rec[:n].to(picked.device), picked[n:]])
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
# MFU-1 stage S (22 Oct): within one grad window pass the full-attention layers' indexer picks and union tiles of the
# forward are kept per layer and reused by the checkpoint recompute (same inputs, so the same picks: exact), instead
# of running the indexer and the union build again. ``window``: a grad pass is running (window_pass / batch_pass).
SEL = {"on": False, "window": False, "store": {}, "hits": 0, "stores": 0, "verify": False, "bad": 0}

# MFU-1 C (24 Oct, trainer_fast "prefix_carry", off): a grad window's tail states (attention K/V, indexer keys, GDN
# conv / recurrent state) recorded in its forward and appended to a copy of its prefix cache, so the try's next window
# continues from there instead of re-encoding those tokens. Not bitwise (the tail is one forward, not prefix chunks:
# the chunking floor); only op_update / op_accumulate use it, never for keep_old (k3) windows, a GSPO pi_old or OPD.
CARRY = {"made": 0, "used": 0, "tokens": 0, "reject": {}}  # reject: why a held carry was not taken (m1cr1: 0/13 used)
CARRY_METHODS = ("update", "update_indexer", "update_conv_state", "update_recurrent_state")


def carry_capture(cache, box):
    """Wrap a read_only cache's update calls: while ``box["on"]`` each call's arguments are recorded (detached) for
    ``carry_cache``; the returned values are read_only's (the checkpoint recompute runs with ``box["on"]`` off)."""
    import torch

    def keep(value):
        return value.detach().clone() if torch.is_tensor(value) else value

    for name in CARRY_METHODS:
        pure = getattr(cache, name)

        def wrapped(*args, _pure=pure, _name=name, **kwargs):
            if box.get("on"):
                box["calls"].append((_name, tuple(keep(a) for a in args), {k: keep(v) for k, v in kwargs.items()}))
            return _pure(*args, **kwargs)

        setattr(cache, name, wrapped)
    return cache


def carry_cache(cache, calls):
    """A copy of ``cache`` (the prefix, positions already bound to prefix + tail by the forward) with the recorded
    calls replayed through the cache class's own updates: the state a use_cache pass over the tail would leave."""
    import copy

    out = copy.deepcopy(cache)
    for name in CARRY_METHODS:
        out.__dict__.pop(name, None)
    for name, args, kwargs in calls:
        getattr(type(out), name)(out, *args, **kwargs)
    return out
# batched tails (``Trainer.batch_pass``): ``prefix`` the shared prefix length while a batch runs (None otherwise);
# ``gather`` (rows, positions): the LM head sees only those hidden states (each row's trained positions)
BATCH = {"prefix": None, "gather": None, "calls": 0, "legacy_positions": False, "kernel": "rows"}


class NoMasks(dict):
    """The model's mask mapping with every mask None (its keys differ between transformers builds): the batched
    attention layers compute their own causal counts and the right padding never precedes a real token."""

    def __missing__(self, key):
        return None


def batch_head_hook(model):
    """A forward pre-hook on the LM head: while ``BATCH["gather"]`` is set, its [B, L, H] input is reduced to the
    listed (row, position) pairs, so the logits are [N, V] for the trained tokens of every row and nothing else."""
    head = model.get_output_embeddings()
    if getattr(head, "_rl1_gather", False):
        return

    def pre(module, args):
        if BATCH["gather"] is None:
            return None
        rows, positions = BATCH["gather"]
        return (args[0][rows, positions],) + tuple(args[1:])

    head.register_forward_pre_hook(pre)
    head._rl1_gather = True


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


def batched_forward(self, hidden_states, position_embeddings, past_key_values, function):
    """The full-attention layer for B right-padded tails after one shared prefix (``BATCH["prefix"]`` keys in a
    batch-1 read-only cache, or none): per row the indexer's selection over the prefix and the row's own tail (query
    i sees prefix + i + 1 keys; padding sits after every real token, so causality keeps it out), then selected-key
    attention without copying the prefix per row."""
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    if SPARSE["mode"] not in ("selected", "triton", "triton2"):
        raise RuntimeError("batched tails need selected or triton attention")
    BATCH["calls"] += 1
    prefix = int(BATCH["prefix"])
    batch, length = hidden_states.shape[:2]
    counts = prefix + torch.arange(1, length + 1, device=hidden_states.device)
    cos, sin = position_embeddings  # [B, prefix + L, dim]
    picked = torch.stack([qsa_select(self.indexer, hidden_states[b:b + 1], (cos[b:b + 1], sin[b:b + 1]), counts,
                                     past_key_values if prefix else None) for b in range(batch)])
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = modeling.apply_rotary_pos_emb(query_states, key_states, cos[:, -length:], sin[:, -length:])
    kp = vp = None
    if prefix:
        layer = past_key_values.layers[self.layer_idx]
        kp, vp = layer.keys[0], layer.values[0]  # [groups, prefix, D]
    if SPARSE["mode"] == "triton" and BATCH.get("kernel") == "prefix_tail":  # did not compile on the B300 build (17 Oct)
        from rl1_sparse import TritonPrefixTailAttention

        attn_output = TritonPrefixTailAttention.apply(query_states, kp, vp, key_states, value_states, picked, self.scaling)
    else:  # per row: the prefix joined with the row's tail (a transient copy), then the proven single-row kernel
        if SPARSE["mode"] in ("triton", "triton2"):
            TritonSelectedAttention = triton_kernel()
        rows = []
        for b in range(batch):
            k = key_states[b:b + 1] if kp is None else torch.cat([kp[None], key_states[b:b + 1]], dim=2)
            v = value_states[b:b + 1] if vp is None else torch.cat([vp[None], value_states[b:b + 1]], dim=2)
            rows.append(TritonSelectedAttention.apply(query_states[b:b + 1], k, v, picked[b], self.scaling, None)
                        if SPARSE["mode"] in ("triton", "triton2") else
                        function.apply(query_states[b:b + 1], k, v, picked[b], self.scaling, SPARSE["chunk"]))
        attn_output = torch.cat(rows, dim=0)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return self.o_proj(attn_output), None


KERNEL = {"union": os.environ.get("RL1_TRITON_KERNEL") == "union", "experts": os.environ.get("RL1_EXPERTS", "dequant")}


def paths_state():
    """The live MFU path switches op_paths sets (and the QSA ties): carry_check / causal_check arms report them, and
    an arm meant to change them that leaves them as the first arm had is a failed arm (m1fl1's S arm, 24 Oct)."""
    mod = lambda name, table, key: (getattr(sys.modules[name], table).get(key) if name in sys.modules else None)
    return {"gdn_fast": mod("rl1_gdn", "GDN", "fast"), "gdn_keep": mod("rl1_gdn", "GDN", "keep"), "sel_cache": SEL["on"],
            "hyper": mod("rl1_hyper", "HYPER", "on"), "hyper_xb": mod("rl1_hyper", "HYPER", "xb"),
            "union": KERNEL.get("union"), "select_fast": SELECT.get("fast"), "unsorted": SELECT.get("unsorted"), "wide": mod("rl1_topk", "WIDE", "method"),
            "dequant": mod("rl1_dequant", "DEQUANT", "on"), "experts": mod("rl1_experts", "EXPERTS", "on"),
            "ties": TIES["mode"]}


def op_paths(model, cmd):
    """MFU stages per op (A/B in one job; a run adopts them through mg.trainer_fast): compile_regions (B),
    union_attention (E), experts "native" | "dequant" (D), gdn_fast (G: rl1_gdn's fused GDN layer); a key absent
    keeps the launch default (env)."""
    want_gdn = bool(cmd["gdn_fast"]) if "gdn_fast" in cmd else os.environ.get("RL1_GDN_FAST") == "1"
    if want_gdn or "rl1_gdn" in sys.modules:
        import rl1_gdn

        rl1_gdn.install()
        rl1_gdn.GDN["fast"] = want_gdn
        rl1_gdn.GDN["keep"] = bool(cmd["gdn_keep"]) if "gdn_keep" in cmd else os.environ.get("RL1_GDN_KEEP") == "1"
    want_p = bool(cmd["prefix_nograd"]) if "prefix_nograd" in cmd else os.environ.get("RL1_PREFIX_NOGRAD") == "1"
    if want_p or "rl1_sparse2" in sys.modules:  # MFU-1 stage P: no dk / dv for the no-grad cached prefix's keys
        import rl1_sparse2

        rl1_sparse2.UnionSelectedAttention.NOGRAD_PREFIX = want_p
    SEL["on"] = bool(cmd["sel_cache"]) if "sel_cache" in cmd else os.environ.get("RL1_SEL_CACHE") == "1"
    CAUSAL["on"] = SEL["on"]
    SEL["verify"] = bool(cmd["sel_verify"]) if "sel_verify" in cmd else os.environ.get("RL1_SEL_VERIFY") == "1"
    causal_reset()
    want_hyper = bool(cmd["hyper_fast"]) if "hyper_fast" in cmd else os.environ.get("RL1_HYPER_FAST") == "1"
    if want_hyper or "rl1_hyper" in sys.modules:  # MFU-1 stage H: fused hyper-connections (rl1_hyper)
        import rl1_hyper

        rl1_hyper.install()
        rl1_hyper.HYPER["on"] = want_hyper
        rl1_hyper.HYPER["xb"] = bool(cmd.get("hyper_xb"))  # MFU-1 (24 Oct): xn's BF16 cast once, in the norm kernel
    want_cast = bool(cmd.get("cast_once"))
    if want_cast or "rl1_cast" in sys.modules:  # MFU-1 item 2 (24 Oct): one BF16 copy per fp32 Linear input (rl1_cast)
        import rl1_cast

        rl1_cast.install(want_cast, model)
    want_deq = bool(cmd["fast_dequant"]) if "fast_dequant" in cmd else os.environ.get("RL1_FAST_DEQUANT") == "1"
    if want_deq or "rl1_dequant" in sys.modules:  # MFU-1 stage Q: Axolotl's expert dequant, same bits, one kernel
        import rl1_dequant

        rl1_dequant.install()
        rl1_dequant.DEQUANT["on"] = want_deq
    NGRAM["on"] = bool(cmd["ngram_pinned"]) if "ngram_pinned" in cmd else os.environ.get("RL1_NGRAM_PINNED") == "1"
    if NGRAM["on"]:
        ngram_pinned_install(model)
    COMPILE["on"] = bool(cmd.get("compile_regions"))
    if COMPILE["on"]:
        compile_regions(model)
    KERNEL["union"] = bool(cmd["union_attention"]) if "union_attention" in cmd else os.environ.get("RL1_TRITON_KERNEL") == "union"
    SELECT["unsorted"] = bool(cmd.get("topk_unsorted")) and KERNEL["union"]  # MFU-1: order-free picks, union kernel only
    fast = cmd.get("fast_topk")  # MFU-1: True (union kernel only: ascending picks) or "verify" (check, any kernel)
    SELECT["fast"] = "verify" if fast == "verify" else (bool(fast) and KERNEL["union"])
    if fast:  # MFU-1 (24 Oct): rows past bisect's width by "torch" (default) or "filter"
        import rl1_topk

        rl1_topk.WIDE["method"] = str(cmd.get("fast_topk_wide") or "torch")
    # MFU stage M (memory per tail token): chunked LM head / log-prob reads; checkpoint-boundary offload
    rl1_memory.HEAD["chunked"] = (bool(cmd["chunked_head"]) if "chunked_head" in cmd
                                  else os.environ.get("RL1_CHUNKED_HEAD") == "1")
    tail = str((cmd["tail_ckpt"] if "tail_ckpt" in cmd else os.environ.get("RL1_TAIL_CKPT")) or "off")
    if tail not in rl1_memory.TAIL_MODES:
        raise ValueError(f"tail_ckpt {tail!r} not in {rl1_memory.TAIL_MODES}")
    rl1_memory.TAIL["mode"] = tail
    want = cmd.get("experts") or os.environ.get("RL1_EXPERTS", "dequant")
    if want != KERNEL["experts"]:
        import native_experts

        native_experts.install(want == "native")
        KERNEL["experts"] = want
    want_x = bool(cmd["fused_experts"]) if "fused_experts" in cmd else os.environ.get("RL1_FUSED_EXPERTS") == "1"
    if want_x or "rl1_experts" in sys.modules:  # MFU-1 stage X: the dequant path's expert elementwise work fused
        import rl1_experts

        rl1_experts.install()
        rl1_experts.EXPERTS["on"] = want_x and KERNEL["experts"] != "native"  # stage D replaces the whole forward


def triton_kernel():
    """The fused selected-key kernel: ``rl1_sparse.TritonSelectedAttention`` (one program per query, fp32 atomics per
    (query, key)) or, for mode ``triton2`` or RL1_TRITON_KERNEL=union (survives a loop's attention call with mode
    "triton"), ``rl1_sparse2.UnionSelectedAttention`` (query-union tiles; same call and results)."""
    if SPARSE["mode"] == "triton2" or KERNEL["union"] or os.environ.get("HEAL_UNION_ATTENTION") == "1":  # HEAL_*: PTH's name
        from rl1_sparse2 import UnionSelectedAttention

        return UnionSelectedAttention
    from rl1_sparse import TritonSelectedAttention

    return TritonSelectedAttention


def batched_cache(cache, batch, positions):
    """A read-only view of a batch-1 prefix cache for ``batch`` rows: per layer a shallow copy whose linear-attention
    states (conv, recurrent; small) are widened to the rows, while the full-attention keys, values and indexer keys
    stay batch-1 (``batched_forward`` and ``fq.read_only`` widen them where needed)."""
    import copy

    view = copy.copy(cache)
    view.layers = []
    for layer in cache.layers:
        twin = copy.copy(layer)
        for name in ("conv_states", "recurrent_states"):
            states = getattr(layer, name, None)
            if isinstance(states, dict):
                setattr(twin, name, {k: (v.expand(batch, *v.shape[1:]).contiguous()
                                         if hasattr(v, "shape") and v.dim() and v.shape[0] == 1 else v)
                                     for k, v in states.items()})
        view.layers.append(twin)
    view.position_ids = positions.expand(positions.shape[0], batch, positions.shape[-1])
    return fq.read_only(view)


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
        if BATCH["prefix"] is not None:
            return batched_forward(self, hidden_states, position_embeddings, past_key_values, function)
        counts = (plain_causal(attention_mask) if SPARSE["mode"] in ("selected", "triton", "triton2") and attention_mask is not None
                  else None)
        if counts is None or (self.training and self.attention_dropout):
            SPARSE["fallback"] += 1
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        SPARSE["calls"] += 1
        use_sel = SEL["on"] and SEL["window"] and ROUTE["qsa"] == "off" and SELECT["mode"] == "qsa"
        kept = SEL["store"].get(self.layer_idx) if use_sel else None
        if kept is not None and kept["length"] == hidden_states.shape[1] and torch.equal(kept["counts"], counts):
            picked = kept["picked"].long()  # the forward's picks (stored int32); the recompute skips the indexer
            SEL["hits"] += 1
            if SEL["verify"]:  # diagnostic: the stored picks against the indexer's on this recompute
                SEL["bad"] += not torch.equal(
                    qsa_select(self.indexer, hidden_states, position_embeddings, counts, past_key_values), picked)
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
        if SELECT["mode"] == "dense":  # rl1-chunk diagnostic: every causal key (the selection only fed the caches)
            from torch.nn.attention.bias import causal_lower_right

            rep = query_states.shape[1] // key_states.shape[1]
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states.repeat_interleave(rep, dim=1), value_states.repeat_interleave(rep, dim=1),
                attn_mask=causal_lower_right(query_states.shape[2], key_states.shape[2]), scale=self.scaling)
        elif SPARSE["mode"] in ("triton", "triton2"):  # the fused gather kernel (triton_kernel)
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


GSPO_ALT_BANDS = {"0.004": (0.004, 0.004), "0.004_0.005": (0.004, 0.005)}  # (low, high) bands logged, not applied
GSPO_REPLY_SCALE = 1000.0  # paper weighting: each reply's term x this constant (~ a reply's tokens), not its count


def gspo_loss(actor, old, reference, weights, reply, valid, clip=(3e-4, 4e-4), tis_cap=0.0, norm="tokens"):
    """GSPO (Zheng et al. 2025, Qwen: sequence-level importance ratio and clipping, for MoE RL stability), one window.
    A sequence is one reply. s_i = exp(mean over the reply's sampled tokens of log pi_theta - log pi_old), pi_old the
    trainer at the served state (``old``: logprobs from the update's start; None = the current pass, s_i = 1);
    objective min(s_i A_i, clip(s_i, 1 - clip[0], 1 + clip[1]) A_i), A_i the reply's weight. Each reply's term is
    multiplied by its trained-token count so that at s_i = 1 the gradient equals the token-level loss's (same lr
    scale as the GRPO arm). Train/serve mismatch: a detached sequence-level TIS factor from pi_old over the sampler
    (``reference``), min(., cap) on A > 0 and clip(., 0, 1) on A < 0, as the token-level rule. Returns
    ``(loss, stats)``; the caller multiplies by its normaliser. ``norm`` "reply" (pilot 1 arm D, the paper's
    objective): every reply counts once, its gradient 1/|y_i| x the sum of its token gradients; the constant
    ``GSPO_REPLY_SCALE`` in place of the count keeps the step's scale near the token loss's."""
    import torch

    v = valid.float()
    n = int(reply.max()) + 1 if reply.numel() else 0
    zeros = lambda: torch.zeros(n, dtype=torch.float32, device=actor.device)
    count = zeros().index_add_(0, reply, v)
    live = count > 0
    den = count.clamp(min=1.0)
    old = actor.detach() if old is None else old.to(actor.device, actor.dtype)
    gap = torch.where(valid, actor.float() - old.float(), torch.zeros_like(actor, dtype=torch.float32))
    log_s = zeros().index_add_(0, reply, gap) / den
    s = torch.exp(log_s)
    a = zeros().index_add_(0, reply, weights.float() * v) / den
    tis = torch.ones(n, dtype=torch.float32, device=actor.device)
    out_band = None  # replies / tokens whose pi_old-over-sampler ratio is outside [0.5, 2]
    if tis_cap and reference is not None:
        drift = torch.where(valid, (old.float() - reference.float()).nan_to_num(0.0, 0.0, 0.0), torch.zeros_like(v))
        ratio = torch.exp(zeros().index_add_(0, reply, drift) / den)
        tis = torch.where(a < 0, ratio.clamp(0.0, 1.0), ratio.clamp(max=tis_cap))
        far = drift.detach().abs() > math.log(2.0)
        out_band = {"replies": int(((ratio.detach() < 0.5) | (ratio.detach() > 2.0))[live].sum()),
                    "tokens": int((far & valid).sum())}
    obj = torch.minimum(s * a, s.clamp(1.0 - clip[0], 1.0 + clip[1]) * a)
    length = torch.full_like(count, GSPO_REPLY_SCALE) if norm == "reply" else count
    loss = -(length * tis * obj)[live].sum()
    sd = s.detach()
    clipped = ((sd > 1.0 + clip[1]) & (a > 0)) | ((sd < 1.0 - clip[0]) & (a < 0))
    # the replies narrower bands would clip on this step's ratios (next restart's band)
    alt = {name: int(((((sd > 1.0 + hi) & (a > 0)) | ((sd < 1.0 - lo) & (a < 0))) & live).sum())
           for name, (lo, hi) in GSPO_ALT_BANDS.items()}
    # k3 approx-KL to pi_old (Schulman; SB3/Spinning Up approx_kl = mean((r - 1) - log r), r = pi/pi_old): per reply on
    # GSPO's own ratio s_i (bf16 routing noise averages out over a reply) and per token (diagnostic: ~0.026 floor)
    ls = log_s.detach()
    g = gap.detach()
    stats = {"replies": int(live.sum()), "clipped": int((clipped & live).sum()),
             "clipped_up": int((clipped & live & (a > 0)).sum()), "clipped_down": int((clipped & live & (a < 0)).sum()),
             "up": int((live & (a > 0)).sum()), "down": int((live & (a < 0)).sum()),
             "log_ratio_abs": round(float(log_s.detach()[live].abs().mean()), 6) if bool(live.any()) else None,
             "log_ratio_mean": round(float(log_s.detach()[live].mean()), 6) if bool(live.any()) else None,
             "tis_mean": round(float(tis[live].mean()), 4) if bool(live.any()) else None,
             # replies whose TIS factor hit the cap (A > 0); per-lag staleness cost (mg.lookahead_below, 22 Oct)
             "tis_capped": int(((tis >= tis_cap) & (a > 0) & live).sum()) if tis_cap else 0,
             "k3_seq_sum": float(((torch.exp(ls) - 1.0) - ls)[live].sum()),
             "k3_tok_sum": float(torch.where(valid, (torch.exp(g) - 1.0) - g, torch.zeros_like(g)).sum()),
             "tokens": int(valid.sum()), "clipped_alt": alt, **({"tis_out": out_band} if out_band is not None else {}),
             # band sizing: each live reply's log s toward its advantage's clip side (> ~hi or
             # ~lo clips) and |log s|, for per-step quantiles (log_s_quantiles)
             "log_s_adv": [round(float(x), 6) for x in (torch.sign(a) * ls)[live & (a != 0)].tolist()],
             "log_s_abs": [round(float(x), 6) for x in ls[live].abs().tolist()]}
    return loss, stats


def approx_kl(windows):
    """The step's k3 approx-KL to pi_old from its windows' GSPO stats: (per reply, per token, clipped share)."""
    st = [((w.get("signed") or {}).get("gspo") or {}) for w in windows]
    replies, tokens = sum(int(x.get("replies") or 0) for x in st), sum(int(x.get("tokens") or 0) for x in st)
    seq = sum(float(x.get("k3_seq_sum") or 0.0) for x in st) / replies if replies else None
    tok = sum(float(x.get("k3_tok_sum") or 0.0) for x in st) / tokens if tokens else None
    clip = sum(int(x.get("clipped") or 0) for x in st) / replies if replies else None
    return seq, tok, clip


def log_s_quantiles(windows, qs=(0.5, 0.9, 0.95, 0.99)):
    """Per-step quantiles of GSPO's per-reply log s over the step's windows: ``adv`` = sign(advantage) x log s (the
    side the band clips), ``abs`` = |log s|. None when no window carries them."""
    out = {}
    for key, name in (("log_s_adv", "adv"), ("log_s_abs", "abs")):
        xs = sorted(float(x) for w in windows for x in (((w.get("signed") or {}).get("gspo") or {}).get(key) or []))
        if xs:
            out[name] = {f"p{int(q * 100)}": round(xs[min(len(xs) - 1, int(q * len(xs)))], 6) for q in qs}
            out[name]["max"], out[name]["n"] = round(xs[-1], 6), len(xs)
    return out or None


def engine_gap(actor, reference, valid):
    """[tokens, sum |actor - engine reference|, tokens beyond 0.5 nats] of one window's valid sampled tokens with an
    engine value (the live engine-vs-trainer check, 22 Oct: lag-0 windows at pi_old = the served state)."""
    import torch

    ref, act = reference.detach().double().cpu(), actor.detach().double().cpu()
    m = valid.cpu().bool() & ~torch.isnan(ref)
    d = (act - ref).abs()[m]
    return [int(m.sum()), round(float(d.sum()), 6), int((d > 0.5).sum())]


def reply_gaps(actor, old, reply, valid, weights=None):
    """GSPO's per-reply log s = mean over the reply's valid tokens of ``actor`` - ``old`` (host tensors of one window),
    with each reply's advantage (mean weight) and token count; token-level |gap| sum and max (rl1-gold)."""
    import torch

    if old is None or old.numel() != actor.numel():
        return {"missing": True}
    v = valid.float()
    n = int(reply.max()) + 1 if reply.numel() else 0
    zeros = lambda: torch.zeros(n, dtype=torch.float64)
    count = zeros().index_add_(0, reply, v.double())
    gap = torch.where(valid, actor.double() - old.double(), torch.zeros_like(actor, dtype=torch.float64))
    log_s = zeros().index_add_(0, reply, gap) / count.clamp(min=1.0)
    w = weights.double() if weights is not None else torch.ones_like(gap)
    a = zeros().index_add_(0, reply, w * v.double()) / count.clamp(min=1.0)
    live = count > 0
    return {"log_s": log_s[live].tolist(), "a": a[live].tolist(), "tokens": count[live].tolist(),
            "gap_sum": float(gap.sum()), "gap_abs_sum": float(gap.abs().sum()), "gap_abs_max": float(gap.abs().max()) if gap.numel() else 0.0,
            "valid": int(valid.sum())}


def gap_stats(rows, clip=(3e-4, 4e-4)):
    """Replies pooled over windows (``reply_gaps`` rows): signed and absolute log s, the share outside GSPO's band by
    advantage sign (what a step against this pi_old would clip), token-level mean gap (k1: -KL on tokens sampled
    from pi_old) and |gap|."""
    import math

    missing = sum(1 for r in rows if r and r.get("missing"))
    rows = [r for r in rows if r and not r.get("missing")]
    log_s = [x for r in rows for x in r["log_s"]]
    a = [x for r in rows for x in r["a"]]
    tokens = sum(r["valid"] for r in rows)
    if not log_s:
        return {"replies": 0, "windows": len(rows), "missing": missing}
    hi, lo = math.log1p(clip[1]), math.log1p(-clip[0])
    clipped_up = sum(1 for x, w in zip(log_s, a) if w > 0 and x > hi)
    clipped_down = sum(1 for x, w in zip(log_s, a) if w < 0 and x < lo)
    absolute = sorted(abs(x) for x in log_s)
    pick = lambda q: absolute[min(len(absolute) - 1, int(q * len(absolute)))]
    return {"windows": len(rows), "missing": missing, "replies": len(log_s), "tokens": tokens,
            "up": sum(1 for w in a if w > 0), "down": sum(1 for w in a if w < 0),
            "clipped_up": clipped_up, "clipped_down": clipped_down,
            "clipped_share": round((clipped_up + clipped_down) / len(log_s), 4),
            "log_s_mean": round(sum(log_s) / len(log_s), 6), "log_s_abs_mean": round(sum(absolute) / len(absolute), 6),
            "log_s_abs_p50": round(pick(0.5), 6), "log_s_abs_p90": round(pick(0.9), 6), "log_s_abs_max": round(absolute[-1], 6),
            "outside_3e-4": round(sum(1 for x in absolute if x > 3e-4) / len(absolute), 4),
            "outside_1e-2": round(sum(1 for x in absolute if x > 1e-2) / len(absolute), 4),
            "token_gap_mean": round(sum(r["gap_sum"] for r in rows) / max(1, tokens), 6),
            "token_gap_abs_mean": round(sum(r["gap_abs_sum"] for r in rows) / max(1, tokens), 6),
            "token_gap_abs_max": round(max(r["gap_abs_max"] for r in rows), 4)}


def expert_lr_scale():
    """RL1_EXPERT_LR_SCALE (pilot 1): the routed experts' lr as a multiple of the step's lr; 1 = one group (as before)."""
    return float(os.environ.get("RL1_EXPERT_LR_SCALE") or 1.0)


def is_routed_expert(name):
    return ".mlp.experts." in name


reply_weight = rl1_encode.reply_weight


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
    by, kernels, nodes, chains = {}, {}, {}, {}
    for event in events:
        own = getattr(event, "kernels", None) or []
        if not own:
            continue
        phase = module = node = None
        parent = event
        chain = []  # MFU-1 (24 Oct): the op chain under the innermost rl1 range (which op launched a copy/cast)
        while parent is not None:
            name = parent.name
            if module is None and not name.startswith(("rl1:", "phase:", "autograd::engine")) and len(chain) < 4:
                chain.append(name[:48])
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
        ckey = (phase, module, " < ".join(chain), "copy" if any("copy" in k.name.lower() for k in own) else "")
        n_, s_ = chains.get(ckey, (0, 0.0))
        chains[ckey] = (n_ + len(own), s_ + seconds)
    out = {"phases": {p: {"total": round(sum(mods.values()), 2),
                          **{m: round(v, 2) for m, v in sorted(mods.items(), key=lambda kv: -kv[1]) if v >= 0.01}}
                      for p, mods in by.items()}}
    out["top_kernels"] = {p: [[name, round(v, 2)] for (q, name), v in sorted(kernels.items(), key=lambda kv: -kv[1])
                              if q == p][:15] for p in by}
    out["unmapped_nodes"] = [[n[:80], round(v, 2)] for n, v in sorted(nodes.items(), key=lambda kv: -kv[1])[:20]]
    ranked = sorted(chains.items(), key=lambda kv: -kv[1][1])
    out["op_chains"] = [[*k[:3], n, round(v, 3)] for k, (n, v) in ranked[:60]]  # (phase, module, op chain, kernels, s)
    out["copy_chains"] = [[*k[:3], n, round(v, 3)] for k, (n, v) in ranked if k[3]][:40]
    return out


def digest(tensor):
    """sha1 of a tensor's bytes (host copy): equal digests mean bit-identical values (MFU exactness checks)."""
    import hashlib

    return hashlib.sha1(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def profile_report(prof, wall):
    """A whole update step under the profiler: wall, card kernel time (idle = wall - kernel), split by phase and
    module (``kernel_split``), leading kernels by device time, launches."""
    began = time.time()
    split = kernel_split(prof.events())
    events = prof.key_averages()
    device = lambda e: getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0
    kernel = sum(device(e) for e in events) / 1e6
    return {"wall_seconds": round(wall, 1), "kernel_seconds": round(kernel, 1),
            "busy_share": round(kernel / wall, 3) if wall else None, "split": split,
            "launches": sum(e.count for e in events if device(e) > 0),
            "top_device": [{"name": e.key[:120], "count": e.count, "device_s": round(device(e) / 1e6, 2)}
                           for e in sorted(events, key=device, reverse=True)[:40]],
            "analysis_seconds": round(time.time() - began, 1)}


class Trainer:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor

        self.args, self.dir = args, Path(args.dir)
        self.prefix_chunk = self.path_chunk = int(getattr(args, "prefix_chunk", 16384) or 16384)
        (self.dir / "cmd").mkdir(parents=True, exist_ok=True)
        (self.dir / "done").mkdir(parents=True, exist_ok=True)
        self.rec = fq.Recorder(self.dir / "trainer")
        self.rec.save(environment=fq.environment(self.rec))
        if PROFILE == "pth":
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
        self.model, _, loaded = load_model(self.rec, args.model, self.dir / "trainer")
        loaded["table_scale"] = fq.apply_table_scale(self.rec, self.model, args.model)
        if os.environ.get("RL1_EXPERT_MASK"):  # hero run: the t37 mask (the server's expert_probe mask)
            loaded["expert_mask"] = install_expert_mask(self.model, os.environ["RL1_EXPERT_MASK"])
            if os.environ.get("RL1_EXPERT_COMPACT") == "1":  # MFU-1: the masked experts' base freed
                loaded["expert_compact"] = self.compact_experts()
        from axolotl.integrations.kernels.libs.sonicmoe import set_merge_aware_enabled

        set_merge_aware_enabled(False)  # plain LoRA over the dequantised base: what the server computes
        if not args.loop_lora:
            grouped_expert_lora()
        import split_gemm  # per-expert base GEMM without per-expert slice gradients (arc2, 18 Oct)
        split_gemm.install(os.environ.get("RL1_SPLIT_GEMM", "0") == "1")  # off until rl1-sgm verifies it on our expert LoRA
        if os.environ.get("RL1_EXPERTS", "dequant") == "native":  # MFU stage D: fused routed experts (mfu-d-v2: 2.5x)
            import native_experts

            native_experts.install()
        if os.environ.get("RL1_INDEXER", "vectorized") == "vectorized":
            vectorized_indexer()
        selected_attention()  # SPARSE["mode"]: RL1_ATTENTION=masked (default; rl1-attn-v1: the gather path is slower)
        batch_head_hook(self.model)
        self.head = rl1_memory.head_capture(self.model)  # MFU stage M: hidden rows out while HEAD["capture"]
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
        if PROFILE == "pth":
            loaded["autotune_pinned"] = pin_autotune_keys()
        self.encoded, _ = fq.example_maker(self.processor, [], tail=args.window, phases=[args.window])
        self.encoded_host, _ = fq.example_maker(self.processor, [], tail=args.window, phases=[args.window], device=None)
        self.window_cache, self.encoders = rl1_encode.Cache(), None  # stage W: old-pass windows kept for the update
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>")
        self.references = {}  # probe name -> {"base": [...], "previous": [...]} per-token actor log-probabilities
        self.updates = 0
        self.parked, self.pinned = False, {}
        used, total, _ = gpu_used_gib()
        loaded.update(experts=os.environ.get("RL1_EXPERTS", "dequant"), seconds=round(time.time() - began, 1), trainable=sum(p.numel() for _, p in self.params),
                      trainable_dtypes=sorted({str(p.dtype) for _, p in self.params}),
                      gpu_allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 2), card_used_gib=used,
                      card_total_gib=total, adapter_groups=loaded.get("adapter_groups"))
        torch.cuda.empty_cache()
        loaded["load_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        self.loaded = loaded
        self.rec.save(loaded=loaded)

    def make_optimizer(self, lr):
        params = [p for _, p in self.params]
        scale = expert_lr_scale()
        if scale != 1.0:  # pilot 1: routed experts in their own group at scale x lr (0: frozen)
            experts = [p for n, p in self.params if is_routed_expert(n)]
            dense = [p for n, p in self.params if not is_routed_expert(n)]
            params = [{"params": dense, "scale": 1.0}, {"params": experts, "scale": scale}]
        if self.args.optimizer == "adamw32":  # rank 16 with full 32-bit Adam
            import torch

            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("scale", 1.0))

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
        """The card-side parts of a window: OPD's teacher window, R3's served routes."""
        if made is None:
            return None
        served = spec.get("served_route") and ROUTE["router"] == "served"
        hosted = "served_host_audit" in made  # resolved in an encoder process (rl1_encode.window)
        if spec.get("hint") or (served and not hosted):
            rows = rl1_encode.read_rows(spec) if rows is None else rows
        if spec.get("hint"):
            made["teacher"] = self.teacher_window(made, rows, spec["hint"])
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

    def encoder_pool(self, procs):
        """``procs`` encoder processes (spawned: no CUDA), kept across ops; a new count or a broken pool is replaced.
        Windows come back through shared memory when /dev/shm has room for ~2 GiB per process, else pickled."""
        if self.encoders is not None and (self.encoders[0] != procs or getattr(self.encoders[1], "_broken", False)):
            self.encoders[1].shutdown(wait=False, cancel_futures=True)
            self.encoders = None
        if self.encoders is None:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            shm = rl1_encode.shm_free_gib() >= 2.0 * procs + 4.0
            pool = ProcessPoolExecutor(procs, mp_context=multiprocessing.get_context("spawn"), initializer=rl1_encode.init,
                                       initargs=(self.args.model, self.args.window, getattr(self, "encoder_loader", None), shm))
            self.encoders = (procs, pool, shm)
        return self.encoders[1]

    def host_window(self, spec, made, rows=None, keep=None):
        """A host-encoded window on the card (``keep``: ``(cache, key)``, the host copy kept for a later op: the old
        pass's windows for the update, the k3 subsample's for the post-step k3)."""
        if keep is not None:
            keep[0].put(keep[1], made)
        if made is None:
            return None
        return self.finish_window(spec, rl1_encode.to_device(made, CARD), rows)

    def encode_host(self, spec, keep=None):
        made, rows = rl1_encode.encode_window(self.encoded_host, self.think_end, self.args.window, spec, 0, device="cpu",
                                              cached=True)
        if made is not None and spec.get("mrope"):
            made["mrope"] = rl1_encode.host_mrope(made["inputs"], spec["mrope"])
        return self.host_window(spec, made, rows, keep)

    def collect_host(self, spec, pending, keep=None):
        """An encoder process's window; a failed or broken process falls back to this process's encode."""
        try:
            made = rl1_encode.unpack(pending.result())
        except Exception as error:
            self.encode_errors = getattr(self, "encode_errors", []) + [repr(error)[:300]]
            return self.encode_host(spec, keep) if keep is not None else self.attempt_window(spec, 0, False)
        if keep is not None and made is not None and self.encoders is not None and self.encoders[2]:
            made = rl1_encode.own(made)  # out of shared memory before it is kept
        return self.host_window(spec, made, None, keep)

    def mrope_how(self):
        """host_mrope: what an encoder needs to compute the model's M-RoPE positions (its core class, merge size)."""
        core = (self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model).model
        return {"cls": f"{type(core).__module__}.{type(core).__qualname__}",
                "merge": int(core.config.vision_config.spatial_merge_size)}

    def vision_ahead(self, made):
        """vision_ahead (feed thread, on the feed stream): the window's image features, as ``positions_and_images``
        computes them on the main thread (frozen tower, same call), so the pass starts without the tower's run."""
        if made is None or made["inputs"].get("pixel_values") is None:
            return made
        made["vision_features"] = fq.image_features(self.model, made["inputs"])
        return made

    def ahead_start(self, cmd):
        """``ahead`` (MFU-1, 23 Oct): the attempts of the loop's next call (its next group, or the step's k3 tries).
        Their rollouts are parsed in the encoder processes now (no wait) and, once this call's own windows are all
        with the encoders, their first windows are encoded too, so the next call's first window is ready when it
        starts instead of after a ~10 s encode with the card idle. Same specs, same encode: the next call takes the
        encoded window by its key, else encodes as before."""
        ahead, procs = cmd.get("ahead") or [], int(cmd.get("encode_procs") or 0)
        self.ahead_plan = None
        if not ahead or not procs:
            return
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

    def window_feed(self, specs, cmd, old=False, hold=None, from_hold=False):
        """The op's windows in order, encoded ahead of the card: ``(next(), close(), stats)``. Default: one encode
        thread, ``prefetch`` windows queued ahead (MFU 3). Stage W (trainer_fast, off by default): ``encode_procs``
        encoder processes run ``max(prefetch, encode_procs)`` windows ahead on the host and the thread only moves
        ``prefetch`` of them to the card; ``window_cache`` (GiB): the old pass keeps its host windows, the update
        steps take them instead of encoding again. MFU-1 (23 Oct): ``hold`` (spec -> bool): those windows' host
        copies go to ``self.k3_cache`` (k3_hold) and ``from_hold`` takes windows from it first (the post-step k3
        scores windows the step's accumulate calls encoded); ``host_mrope``: encoders compute the M-RoPE positions;
        ``vision_ahead`` (with feed_stream): the feed thread runs the vision tower on the feed stream."""
        import collections
        from concurrent.futures import ThreadPoolExecutor

        procs, cap = int(cmd.get("encode_procs") or 0), float(cmd.get("window_cache") or 0)
        keep, use = old and cap > 0, not old and cap > 0
        if keep:
            self.window_cache = rl1_encode.Cache(cap)  # a new update: the last one's windows go
        held = getattr(self, "k3_cache", None)
        hold = hold if hold is not None and held is not None else None
        from_hold = bool(from_hold and held is not None)
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
        stats = {"waits": 0.0, "cache_hits": 0, "encoded": 0, "held": 0, "hold_hits": 0, "ahead_hits": 0, "ahead_sent": 0}
        self.encode_errors = []
        carried = {}
        if specs:  # the last call's ahead encodes (a call without windows, e.g. op_step, leaves them for the next)
            carried, self.ahead_windows = getattr(self, "ahead_windows", None) or {}, {}
        keyed = keep or use or hold is not None or from_hold or bool(carried)
        keys = [rl1_encode.spec_key(spec) for spec in specs] if keyed else [None] * len(specs)
        mrope = self.mrope_how() if cmd.get("host_mrope") and (procs or keep or use or hold is not None) else None

        def hosted_spec(spec):  # R3 on: the encoder process resolves the window's served routes too
            if ROUTE["router"] == "served" and spec.get("served_route") and not spec.get("hint"):
                spec = {**spec, "served_in_encoder": True}
            return {**spec, "mrope": mrope} if mrope else spec

        def cached(i):
            return (use and keys[i] in self.window_cache.rows) or (from_hold and keys[i] in held.rows)

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
            held_here = hold is not None and hold(spec)
            if not (procs or keep or use or held_here or from_hold):
                return queue.append(submit(self.attempt_window, spec, 0, False))
            hit = self.window_cache.pop(key) if use else None
            if hit is None and from_hold:
                hit = held.pop(key)
                stats["hold_hits"] += hit is not None
            elif hit is not None:
                stats["cache_hits"] += 1
            if hit is not None:
                hosted.pop(i, None)
                return queue.append(submit(self.host_window, spec, hit))
            stats["encoded"] += 1
            stats["held"] += held_here
            keep_to = (self.window_cache, key) if keep else ((held, key) if held_here else None)
            if encoders is not None:
                pending = hosted.pop(i, None) or encoders.submit(rl1_encode.window, hosted_spec(spec), 0)
                return queue.append(submit(self.collect_host, spec, pending, keep_to))
            if keep_to is not None:
                return queue.append(submit(self.encode_host, hosted_spec(spec), keep_to))
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

    def prefix_pass(self, ids, cut, piece, start_at=0, cache=None, snap_at=None, carried=False):
        """The tokens before ``cut`` through the model without gradients, in chunks of the pinned ``prefix_chunk``,
        into a cache. No fallback to another chunk: the chunking changes the log-probs (rl1-gold: 0.054 nats/token
        between 16K and 32K at the same weights), so every pass of a run takes one. ``start_at`` / ``cache``: continue
        a cache that holds [0, start_at) (a chunk boundary: the same chunks as a pass from 0); ``snap_at`` (a chunk
        boundary): also return a copy of the cache as it stood at that position. Returns ``(cache, chunk, copy)``."""
        import copy

        import torch

        chunk, snapshot = int(getattr(self, "prefix_chunk", 16384)), None
        if (start_at % chunk and not carried) or (snap_at is not None and snap_at % chunk):
            raise ValueError(f"prefix pass: {start_at} / {snap_at} not on the {chunk}-token chunk grid")
        grid = -(-start_at // chunk) * chunk  # a carried cache (off the grid) first runs up to the next boundary
        starts = ([start_at] if start_at < min(grid, cut) else []) + list(range(max(grid, start_at), cut, chunk))
        self.model.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for start in starts:
                    if snap_at == start and cache is not None:
                        snapshot = copy.deepcopy(cache)
                    stop = min((start // chunk + 1) * chunk, cut)
                    ROUTE["call"] = served_call(start, stop)
                    causal_reset()
                    output = self.model(input_ids=ids[:, start:stop], attention_mask=torch.ones_like(ids[:, :stop]),
                                        past_key_values=cache, use_cache=True, logits_to_keep=1, **piece(start, stop))
                    cache = output.past_key_values
                    del output
                if snap_at == cut and snapshot is None and cache is not None:
                    snapshot = copy.deepcopy(cache)
        finally:
            ROUTE["call"] = None
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

    def prefix_cache(self, inputs, ids, cut, piece, shared):
        """The no-gradient prefix cache before ``cut``: ``(cache, reused, chunk, prefix positions)``; ``shared``
        keeps one prefix across the windows of a step (reused when ids and images before the cut match)."""
        import copy

        import torch

        key = self.prefix_key(inputs, cut) if shared is not None and cut > 0 else None
        self.prefix_start, self.prefix_carried = 0, False
        pure = getattr(self, "carry_pure", True)  # a keep_old (k3) window takes no carried state, directly or reused
        if (key is not None and shared.get("key") == key and bool(torch.equal(shared["ids"], ids[:, :cut]))
                and (not pure or not shared.get("impure"))):
            self.prefix_start, self.prefix_carried = cut, bool(shared.get("impure"))
            return shared["cache"], True, None, shared["positions"]
        snaps = shared.get("snaps") if shared is not None else None
        held = {k: shared[k] for k in ("carry", "carry_on") if shared is not None and k in shared}
        if shared is not None:
            shared.clear()  # the previous prefix's cache goes before this one is built (its snapshots stay)
            shared.update(held)
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
                if not bool(torch.equal(snap["ids"], ids[:, :at])) or self.prefix_key(inputs, at) != snap["pkey"]:
                    continue
                if (snap["route"] is None) != (route is None) or (
                        route is not None and not bool(torch.equal(snap["route"], route[:at].to(snap["route"].device)))):
                    self.snap_route_miss = getattr(self, "snap_route_miss", 0) + 1
                    continue
                best = snap
            carry = held.get("carry") if held.get("carry_on") and not pure else None
            why = carry_reject(carry, held, pure, cut, best, ids, route, lambda at: self.prefix_key(inputs, at))
            if why is not None:
                CARRY["reject"][why] = CARRY["reject"].get(why, 0) + 1
            if why is None and carry is not None:
                best, self.prefix_carried = carry, True
                CARRY["used"] += 1
                CARRY["tokens"] += carry["at"] - max([sn["at"] for sn in snaps if sn["at"] <= cut
                                                      and bool(torch.equal(sn["ids"], ids[:, :sn["at"]]))] + [0])
            if best is not None:
                start_at, base = best["at"], copy.deepcopy(best["cache"])
            grid = int(getattr(self, "prefix_chunk", 16384))
            snap_at = (cut // grid) * grid
            if snap_at <= 0 or snap_at <= start_at or self.prefix_carried:  # grid snapshots stay carry-free
                snap_at = None
        if snaps is not None:
            shared["snaps"] = snaps
        if cut > 0:
            cache, chunk, copy_at = self.prefix_pass(ids, cut, piece, start_at=start_at, cache=base, snap_at=snap_at,
                                                     carried=self.prefix_carried)
            positions = cache.position_ids
            self.prefix_start = start_at
            if copy_at is not None:
                snaps.append({"at": snap_at, "ids": ids[:, :snap_at], "pkey": self.prefix_key(inputs, snap_at),
                              "route": None if route is None else route[:snap_at].clone(), "cache": copy_at})
                while len(snaps) > int(getattr(self, "snap_keep", 3)):  # ~24 KB of keys per token (12 attention
                    snaps.pop(0)  # layers) + fixed GDN state: a 96K snapshot is ~2.4 GB
        if key is not None:
            shared.update(key=key, ids=ids[:, :cut], cache=cache, positions=positions, impure=self.prefix_carried)
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
        moved = 0
        carry = (shared.get("carry") if shared is not None and shared.get("carry_on") == "margin"
                 and not made.get("no_carry") else None)
        if (carry is not None and cut < carry["at"] < int(positions.min()) and carry["at"] <= total
                and bool(torch.equal(carry["ids"], ids[:, :carry["at"]]))):
            # MFU-1 C "margin": the window's no-reply context before its first trained token (the margin) starts at
            # the previous window's carried end instead (those tokens' K/V get no gradient here: a small change)
            moved, cut = carry["at"] - cut, carry["at"]
            tail = total - cut
        keep = positions - cut - 1
        ROUTE["window"] = made.get("served")
        if bool((keep < 0).any()):
            raise RuntimeError("a trained token lies outside the gradient window")
        tokens = ids[0, positions]
        marks = {}  # MFU-1: the pre's vision tower and position-rule seconds
        piece = fq.positions_and_images(self.model, inputs, positions=made.get("mrope"),
                                        features=made.pop("vision_features", None), marks=marks)
        torch.cuda.synchronize()
        began, cache, reused, chunk = time.time(), None, False, None
        pre = began - entered
        phase = torch.profiler.record_function("phase:prefix")
        phase.__enter__()
        self.carry_pure = bool(made.get("no_carry")) or not (shared is not None and shared.get("carry_on"))
        cache, reused, chunk, prefix_positions = self.prefix_cache(inputs, ids, cut, piece, shared)
        carried = bool(getattr(self, "prefix_carried", False))
        if cache is not None and not BATCH.get("legacy_positions"):
            # the model binds the full positions to the cache on every pass (prefix + this tail); a reused prefix
            # must start from its own positions, not the last window's (the indexer's block keys read them)
            cache.position_ids = prefix_positions
        from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

        modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
        torch.cuda.synchronize()
        phase.__exit__(None, None, None)
        prefix = time.time() - began
        began = time.time()
        grad = scale is not None
        if not grad:
            self.model.eval()
        elif not self.model.training:  # HF checkpointing runs only in train mode (PTH's resume found +17 GiB, 20 Oct)
            raise RuntimeError("grad pass with the model in eval mode")
        phase = torch.profiler.record_function("phase:forward")
        phase.__enter__()
        chunked = rl1_memory.HEAD["chunked"]
        SEL.update(window=grad and SEL["on"], store={})
        causal_reset()
        try:
            with (torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16),
                  rl1_memory.tail_scope(self.model) if grad else contextlib.nullcontext()):
                ROUTE["call"] = served_call(cut, total)
                rl1_memory.HEAD["capture"] = chunked
                box = ({"on": True, "calls": []} if cache is not None and shared is not None and shared.get("carry_on")
                       else None)
                view = fq.read_only(cache) if cache is not None else None
                if box is not None:
                    carry_capture(view, box)
                try:
                    logits = self.model(input_ids=ids[:, cut:], attention_mask=torch.ones_like(ids),
                                        past_key_values=view, use_cache=False, logits_to_keep=keep,
                                        **piece(cut, total)).logits[0]
                finally:
                    rl1_memory.HEAD["capture"] = False
                    if box is not None:
                        box["on"] = False  # the checkpoint recompute in the backward records nothing
                if chunked:  # logits holds the kept rows' hidden states; no [N, V] table (stage M)
                    head = rl1_memory.chunked_head(self.head._rl1_project, logits, tokens, arrays["support"],
                                                   made["temperature"], hit=bool(made.get("argmax")), vocab=vocab)
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
                    loss, tis, signed = self.weighted_loss(arrays, actor, valid, scale)
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
        if box is not None and box["calls"]:  # MFU-1 C: this window's end as the try's next window's prefix
            route = ROUTE["window"] if ROUTE["router"] == "served" else None
            shared["carry"] = {"at": total, "ids": ids[:, :total].clone(), "pkey": self.prefix_key(inputs, total),
                               "route": None if route is None else route[:total].clone(),
                               "cache": carry_cache(cache, box["calls"])}
            CARRY["made"] += 1
            del box
        ROUTE["call"] = ROUTE["window"] = None  # after the backward: checkpoint recompute routes as the forward did
        SEL.update(window=False, store={})
        ended = time.time()
        causal_reset()
        out = {"actor": actor.detach().float().cpu(), "raw": raw.float().cpu(), "valid": valid.cpu(),
               "dist": dist,  # vocab_top: already on the host
               "hit": None if hit is None else hit.cpu(),
               "entropy": entropy,
               "signed": signed, "opd": opd,
               "tis": None if tis is None else {"mean": round(float(tis[valid].mean()), 4),
                                                "capped": round(float((tis[valid] >= self.args.tis_cap).float().mean()), 5)},
               "loss": None if loss is None else float(loss.detach()), "prefix_seconds": round(prefix, 2),
               "prefix_reused": reused, "prefix_chunk": chunk, "cut": cut, "prefix_start": getattr(self, "prefix_start", 0),
               "prefix_carried": carried, "carry_moved": moved,
               "forward_seconds": round(forward, 2), "backward_seconds": round(backward, 2), "tokens": total,
               "tail": tail, "targets": int(positions.numel())}
        del logits, cache
        out["host"] = {"pre": round(pre, 3), "post": round(time.time() - ended, 3), **marks}
        return out

    def batch_pass(self, mades, scales, shared=None):
        """Windows of one start whose prefixes match, as one batch: the prefix once (or ``shared``), then the B
        tails right-padded through the model together (prefix cache widened to B rows without copying its keys),
        logits only at each row's trained tokens, each row's loss as ``window_pass`` computes it, one backward.
        Returns one ``window_pass``-style result per window."""
        import torch

        first = mades[0]
        PROGRESS["tail"] += sum(int(m["tail"]) for m in mades)
        PROGRESS["windows"] += len(mades)
        ids0 = first["inputs"]["input_ids"]
        cut = ids0.shape[1] - first["tail"]
        for made in mades[1:]:
            ids = made["inputs"]["input_ids"]
            if ids.shape[1] - made["tail"] != cut or not bool(torch.equal(ids[:, :cut], ids0[:, :cut])):
                raise RuntimeError("batched windows must share their prefix")
            if self.prefix_key(made["inputs"], cut) != self.prefix_key(first["inputs"], cut):
                raise RuntimeError("batched windows must share their prefix images")
        torch.cuda.synchronize()
        began = time.time()
        pieces = [fq.positions_and_images(self.model, m["inputs"]) for m in mades]
        ROUTE["window"] = first.get("served")  # the shared prefix's rows
        cache, reused, chunk, prefix_positions = self.prefix_cache(first["inputs"], ids0, cut, pieces[0],
                                                                   shared if shared is not None else None)
        torch.cuda.synchronize()
        prefix = time.time() - began
        began = time.time()
        batch, lengths = len(mades), [m["tail"] for m in mades]
        width = max(lengths)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        tail_ids = torch.full((batch, width), int(pad), dtype=ids0.dtype, device=ids0.device)
        positions, images, rows, cols, spans = [], [], [], [], []
        for b, made in enumerate(mades):
            total = made["inputs"]["input_ids"].shape[1]
            tail_ids[b, : lengths[b]] = made["inputs"]["input_ids"][0, cut:]
            extra = pieces[b](cut, total)
            pos = extra["position_ids"]  # [3, 1, L_b]
            if lengths[b] < width:  # padding positions continue the row's last one
                more = pos[:, :, -1:] + torch.arange(1, width - lengths[b] + 1, device=pos.device).view(1, 1, -1)
                pos = torch.cat([pos, more], dim=-1)
            positions.append(pos)
            if "mm_encoder_outputs" in extra:
                images.append(extra["mm_encoder_outputs"]["image"].pooler_output[0])
            keep = made["arrays"]["positions"] - cut - 1  # logits at keep predict the sampled tokens
            if bool((keep < 0).any()):
                raise RuntimeError("a trained token lies outside the gradient window")
            rows.append(torch.full_like(keep, b))
            cols.append(keep)
            spans.append(int(keep.numel()))
        kwargs = {"position_ids": torch.cat(positions, dim=1)}
        if images:
            from transformers.modeling_outputs import BaseModelOutputWithPooling

            kwargs["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=(torch.cat(images, dim=0),))}
        view = None
        if cache is not None:
            view = batched_cache(cache, batch, prefix_positions)
        from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

        modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
        grad = scales is not None
        if not grad:
            self.model.eval()
        elif not self.model.training:  # HF checkpointing runs only in train mode (PTH's resume found +17 GiB, 20 Oct)
            raise RuntimeError("grad pass with the model in eval mode")
        BATCH.update(prefix=cut if cache is not None else 0, gather=(torch.cat(rows), torch.cat(cols)))
        causal_reset()
        ROUTE["call"] = None
        if ROUTE["router"] == "served" and any(m.get("served") is not None for m in mades):
            table = next(m["served"] for m in mades if m.get("served") is not None)
            call = torch.full((batch, width) + tuple(table.shape[1:]), -1, dtype=table.dtype, device=table.device)
            for b, made in enumerate(mades):
                if made.get("served") is not None:
                    call[b, : lengths[b]] = made["served"][cut: cut + lengths[b]]
            ROUTE["call"] = call.view(batch * width, *table.shape[1:])
        results = []
        chunked = rl1_memory.HEAD["chunked"]
        try:
            with (torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16),
                  rl1_memory.tail_scope(self.model) if grad else contextlib.nullcontext()):
                rl1_memory.HEAD["capture"] = chunked
                try:
                    logits = self.model(input_ids=tail_ids, attention_mask=NoMasks(),
                                        past_key_values=view, use_cache=False, logits_to_keep=0, **kwargs).logits
                finally:
                    rl1_memory.HEAD["capture"] = False
                if logits.dim() == 3:  # [1, N, V] from a wrapper that kept the batch axis ([1, N, H] if chunked)
                    logits = logits[0]
                loss_total = None
                start = 0
                for b, made in enumerate(mades):
                    arrays = made["arrays"]
                    part = logits[start: start + spans[b]]
                    start += spans[b]
                    tokens = made["inputs"]["input_ids"][0, arrays["positions"]]
                    if chunked:
                        head = rl1_memory.chunked_head(self.head._rl1_project, part, tokens, arrays["support"],
                                                       made["temperature"])
                        actor, valid, raw, entropy = head["actor"], head["valid"], head["raw"], head["entropy"]
                    else:
                        actor, valid = rl_job.actor_logprobs(part, tokens, arrays["support"], made["temperature"])
                        raw = rl_job.raw_logprobs(part.detach(), tokens)
                        entropy = token_entropy(part.detach(), valid, made["temperature"])
                    loss = tis = signed = None
                    if grad:
                        loss, tis, signed = self.weighted_loss(arrays, actor, valid, scales[b])
                        loss_total = loss if loss_total is None else loss_total + loss
                    results.append({"actor": actor.detach().float().cpu(), "raw": raw.float().cpu(), "valid": valid.cpu(),
                                    "dist": None, "signed": signed, "opd": None, "entropy": entropy,
                                    "tis": None if tis is None else {
                                        "mean": round(float(tis[valid].mean()), 4),
                                        "capped": round(float((tis[valid] >= self.args.tis_cap).float().mean()), 5)},
                                    "loss": None if loss is None else float(loss.detach()),
                                    "prefix_reused": reused or b > 0, "prefix_chunk": chunk, "cut": cut,
                                    "tokens": cut + lengths[b], "tail": lengths[b], "targets": spans[b]})
            torch.cuda.synchronize()
            forward = time.time() - began
            backward = 0.0
            if grad and loss_total is not None:  # inside: checkpoint recompute runs the batched layers again
                began = time.time()
                loss_total.backward()
                torch.cuda.synchronize()
                backward = time.time() - began
        finally:
            BATCH.update(prefix=None, gather=None)
            ROUTE["call"] = ROUTE["window"] = None
            if not grad:
                self.model.train()
        for row in results:  # the batch's times, shared out by tail length
            share = row["tail"] / max(1, sum(lengths))
            row.update(prefix_seconds=round(prefix * share, 2), forward_seconds=round(forward * share, 2),
                       backward_seconds=round(backward * share, 2), batch=batch)
        del logits, view, cache
        return results

    def weighted_loss(self, arrays, actor, valid, scale):
        """The policy-gradient loss of one window's sampled tokens: ``(loss, tis, signed)``."""
        import torch

        weights = arrays["weights"] if "weights" in arrays else torch.ones_like(actor)
        weights = weights.float()
        tis = signed = None
        policy = getattr(self, "policy", None) or {}
        if policy.get("loss") == "gspo" and "reply" in arrays:
            loss, stats = gspo_loss(actor, arrays.get("old"), arrays.get("reference"), weights, arrays["reply"], valid,
                                    clip=tuple(policy.get("clip") or (3e-4, 4e-4)), tis_cap=self.args.tis_cap,
                                    norm=policy.get("norm") or "tokens")
            w = weights[valid]
            signed = {"up": round(float(w.clamp(min=0).sum()), 4), "down": round(float((-w).clamp(min=0).sum()), 4),
                      "down_tokens": int(((weights < 0) & valid).sum()), "gspo": stats}
            return loss * scale, None, signed
        if self.args.tis_cap and "reference" in arrays:  # fit's synthetic windows carry none (rl1-loop-v1)
            # trainer over sampler on the tempered, truncated policy (1 where the engine value is missing):
            # positive weights x min(ratio, cap) (truncated IS, Yao et al. 2025); negative weights x
            # clip(ratio, 0, 1) (TOPR, Le Roux et al. 2025: the push-down term stays bounded)
            ratio = torch.exp(actor.detach() - arrays["reference"]).nan_to_num(1.0, 1.0, 1.0)
            tis = torch.where(weights < 0, ratio.clamp(0.0, 1.0), ratio.clamp(max=self.args.tis_cap))
            down = (weights < 0) & valid
            if bool(down.any()):
                signed = {"down_tokens": int(down.sum()), "down_clipped": round(float((ratio[down] > 1).float().mean()), 5)}
            far = ((ratio < 0.5) | (ratio > 2.0)) & valid  # tokens outside [0.5, 2]
            signed = {**(signed or {}), "tis_out": {"tokens": int(far.sum()), "of": int(valid.sum())}}
            weights = weights * tis
        loss = -(weights[valid] * actor[valid]).sum() * scale
        if signed is not None or bool((weights[valid] > 0).any()):
            w = weights[valid]
            signed = {**(signed or {}), "up": round(float(w.clamp(min=0).sum()), 4),
                      "down": round(float((-w).clamp(min=0).sum()), 4)}
        return loss, tis, signed

    def mismatch(self, made, out, gap_dump=None):
        """Trainer against server on the window's sampled tokens (raw log-probabilities, whole vocabulary,
        temperature 1): engine minus trainer. ``gap_dump`` (MFU-1 sampler gap, 24 Oct): also keep the window's
        per-token pairs in GAP_DUMP (carry_check saves them)."""
        import torch

        engine = torch.tensor(made["engine_raw"], dtype=torch.float32)
        if gap_dump:
            ids = made["inputs"]["input_ids"][0].cpu()
            pos = made["arrays"]["positions"].cpu().long()
            GAP_DUMP.append({"name": made["name"], "engine": engine.clone(), "trainer": out["raw"].detach().float().cpu(),
                             "positions": pos, "tokens": ids[pos], "temperature": made.get("temperature")})
        diff = engine - out["raw"]
        keep = torch.isfinite(diff)
        diff = diff[keep]
        if not diff.numel():
            return {"tokens": 0}
        return {"tokens": int(diff.numel()), "mean_abs": round(float(diff.abs().mean()), 5),
                "mean": round(float(diff.mean()), 5), "beyond_half": round(float((diff.abs() > 0.5).float().mean()), 5),
                "beyond_ln2": round(float((diff.abs() > math.log(2)).float().mean()), 5),
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
            self.set_lr(float(cmd.get("lr", 1e-3)))
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

    def op_nudge(self, cmd):
        """mg5g (rl1-flr checks): every update that moved the adapter at all read KL ~0.015 to a base scored with the
        adapter's B factors at exactly zero, at any lr. Zero B factors set to ``value`` (1e-12: no measurable output
        change) so the base reference is scored on the same path as every later state; the optimizer is untouched."""
        import torch

        value, count = float(cmd.get("value", 1e-12)), 0
        with torch.no_grad():
            for name, param in self.params:
                if "lora_B" in name and not bool(param.any()):
                    param.fill_(value)
                    count += 1
        return {"nudged": count, "value": value, "adapter_norm": self.adapter_norm()}

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
        over = lambda: any(torch.cuda.memory_allocated(d) > frac * t for d, t in totals.items())  # noqa: E731

        def log(row):
            row = dict(row)
            self.rec.event(row.pop("event"), **row)

        teacher = heal_stream.StreamTeacher(
            cmd["rollouts"], cmd["out"], encode, forward, k=k, chunk=chunk, image_token_id=image_id,
            idle=float(cmd.get("idle", 900)), poll=float(cmd.get("poll", 2.0)), encoders=int(cmd.get("encoders", 4)),
            log=log, budget=heal_stream.lru_budget(over), top_fn=top_fn, stop_file=cmd.get("stop_file"),
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
            got = t.verify(chain)
            got["student"] = student
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

    def score_windows(self, specs, label=None, hold=False):
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
            refs["last"] = mine
            if label == "base":
                refs["base"] = mine
            if label == "base" or not hold:  # ``hold`` (mg5e): "previous" moves only on accept_reference
                refs["previous"] = mine
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

    @staticmethod
    def token_at(made, i):
        """The token id scored at actor index ``i`` of a window (positions index the window's input ids), or None."""
        try:
            pos = int(made["arrays"]["positions"].reshape(-1)[i])
            return int(made["inputs"]["input_ids"][0, pos])
        except (KeyError, IndexError, TypeError, RuntimeError):
            return None

    def old_key(self, made):
        positions = made["arrays"]["positions"]
        return made["name"], int(positions.numel()), int(positions[0]) if positions.numel() else -1

    def op_old_logprobs(self, cmd):
        """GSPO's pi_old: the actor log-probabilities of the update's later steps' windows at the update's start (the
        served state), kept until the next call. As ``op_update`` reads its windows (P2 b: the serial pass took 1,745 s
        for 95 windows at ~45% card util): grouped by start with one shared prefix cache (``share_prefix``), the next
        window encoded on a thread while the card runs this one; the prefix path is the one the steps' pi uses.
        Stage W: ``encode_procs`` / ``window_cache`` as ``window_feed``."""
        began, self.old = time.time(), {}
        keep, compare = cmd.get("keep"), cmd.get("compare")  # rl1-gold: a labelled copy; per-reply gaps to a kept copy
        kept, gaps = ({} if keep else None), []
        against = (getattr(self, "kept", {}) or {}).get(compare) if compare else None
        if compare and against is None:
            raise ValueError(f"no kept old log-probs {compare!r}")
        self.pin_chunk(cmd)
        specs = [a for a in self.expand_cmd(cmd) if a["w"] > 0 or (cmd.get("signed") and a["w"] < 0)]
        shared = None
        if cmd.get("share_prefix"):
            groups = {}
            for a in specs:
                groups.setdefault(a.get("share"), len(groups))
            specs = sorted(specs, key=lambda a: (groups[a.get("share")], not a.get("first_window", True)))
            shared = {}
        reused, engine = 0, []
        next_window, close, feed = self.window_feed(specs, cmd, old=True)
        with Sampler() as sampler:
            try:
                for spec in specs:
                    made = next_window()
                    if made is None:
                        continue
                    out = self.window_pass(made, shared=shared)
                    key = self.old_key(made)
                    self.old[key] = out["actor"]
                    if made["arrays"].get("reference") is not None:
                        engine.append([spec.get("name"), *engine_gap(out["actor"], made["arrays"]["reference"], out["valid"])])
                    if keep or compare:
                        reply = made["arrays"]["reply"].cpu()
                        weights = made["arrays"]["weights"].cpu() if "weights" in made["arrays"] else None
                        if keep:
                            kept[key] = {"actor": out["actor"], "reply": reply, "valid": out["valid"], "weights": weights}
                        if compare:
                            gaps.append(reply_gaps(out["actor"], (against.get(key) or {}).get("actor"), reply, out["valid"], weights))
                    reused += bool(out["prefix_reused"])
                    del made, out
            finally:
                close()
                if shared is not None:
                    shared.clear()
        result = {"windows": len(self.old), "prefix_chunk": self.prefix_chunk, "seconds": round(time.time() - began, 1), "prefix_reused": reused,
                  "window_wait_seconds": round(feed["waits"], 1), "engine_gap": engine, **sampler.summary()}
        if cmd.get("encode_procs") or cmd.get("window_cache"):
            result["encode"] = {"procs": int(cmd.get("encode_procs") or 0), "encoded": feed["encoded"],
                                "shm": bool(self.encoders and self.encoders[2]), "errors": self.encode_errors[:5],
                                "cache": self.window_cache.summary()}
        if keep:
            self.kept = {**(getattr(self, "kept", {}) or {}), keep: kept}
        if compare:
            result["compare"] = {"against": compare, **gap_stats(gaps, tuple(cmd.get("clip") or (3e-4, 4e-4)))}
        return result

    def op_chunk_check(self, cmd):
        """rl1-chunk (GSPO's serial and shared pi_old passes differ by 0.054 nats/token at the same
        weights): each window at every setting {"chunk": prefix chunk, "select": qsa | stable | dense}, no gradient,
        no shared prefix. Per setting: the trainer-vs-sampler gap (``mismatch``, pooled over tokens) and the per-token /
        per-reply gaps to the first setting with the same ``select``."""
        settings = cmd["settings"]
        name = lambda st: f"{st.get('select', 'qsa')}@{int(st['chunk'])}"
        firsts = {}
        for st in settings:
            firsts.setdefault(st.get("select", "qsa"), name(st))
        gaps, seen, seconds = {name(st): [] for st in settings}, {name(st): [] for st in settings}, {name(st): 0.0 for st in settings}
        saved = getattr(self, "prefix_chunk", 16384), SELECT["mode"]
        specs = self.expand_cmd(cmd)[: int(cmd.get("max_windows", 1 << 30))]
        windows = 0
        try:
            for spec in specs:
                made = self.attempt_window(spec, 0, False)
                if made is None:
                    continue
                windows += 1
                reply = made["arrays"]["reply"].cpu() if "reply" in made["arrays"] else None
                weights = made["arrays"]["weights"].cpu() if "weights" in made["arrays"] else None
                refs = {}
                for st in settings:
                    key, select = name(st), st.get("select", "qsa")
                    self.prefix_chunk, SELECT["mode"] = int(st["chunk"]), select
                    began = time.time()
                    out = self.window_pass(made, shared=None)
                    seconds[key] += time.time() - began
                    seen[key].append(self.mismatch(made, out))
                    ref = refs.setdefault(select, out["actor"])
                    if reply is not None:
                        gaps[key].append(reply_gaps(out["actor"], ref, reply, out["valid"], weights))
                    del out
                del made
        finally:
            self.prefix_chunk, SELECT["mode"] = saved
        rows = []
        for st in settings:
            key = name(st)
            mm = [m for m in seen[key] if m.get("tokens")]
            tokens = sum(m["tokens"] for m in mm)
            pool = lambda f: round(sum(m[f] * m["tokens"] for m in mm) / tokens, 5) if tokens else None
            rows.append({"setting": key, "against": firsts[st.get("select", "qsa")], "seconds": round(seconds[key], 1),
                         "sampler_gap": {"tokens": tokens, "mean_abs": pool("mean_abs"), "mean": pool("mean"),
                                         "beyond_half": pool("beyond_half")},
                         "gap": gap_stats(gaps[key], tuple(cmd.get("clip") or (3e-4, 4e-4)))})
        return {"windows": windows, "settings": rows}

    def op_served_check(self, cmd):
        """rl1-route item 3d(ii): does the server apply the routed experts' LoRA? On tries the server played with this
        state: the sampler's recorded raw log-probs e, the trainer's at the full state f and with the routed experts'
        lora_B zeroed x (``zero``: all | gate_up | down). Per tag, over sampled tokens: slope and correlation of e - x on
        d = f - x (~1 when the server applies that part, ~0 when it drops it), rms d, mean |e - f| and |e - x|."""
        import torch

        parts = {"all": lambda n: ".mlp.experts." in n,
                 "gate_up": lambda n: ".mlp.experts.base_layer." in n,
                 "down": lambda n: ".mlp.experts." in n and ".mlp.experts.base_layer." not in n,
                 "dense": lambda n: ".mlp.experts." not in n}  # the control: the part vLLM surely applies
        tags = list(cmd.get("zero") or ("all", "gate_up", "down"))
        saved = {name: param.detach().clone() for name, param in self.params if "lora_B" in name}
        if not any(parts["all"](name) for name in saved):
            raise RuntimeError("no routed-expert lora_B parameters")
        big = float(cmd.get("big", 0.1))
        sums = {t: {k: 0.0 for k in ("n", "d", "dd", "y", "yy", "yd", "ef", "ex", "nb", "db", "ddb", "yb", "yyb", "ydb")}
                for t in tags}
        counts = {t: sum(int(p.numel()) for name, p in saved.items() if parts[t](name)) for t in tags}
        specs = self.expand_cmd(cmd)[: int(cmd.get("max_windows", 1 << 30))]
        windows, began = 0, time.time()

        def zero(tag):
            with torch.no_grad():
                for name, param in self.params:
                    if name in saved:
                        param.copy_(saved[name])
                        if tag is not None and parts[tag](name):
                            param.zero_()

        try:
            for spec in specs:
                made = self.attempt_window(spec, 0, False)
                if made is None:
                    continue
                windows += 1
                engine = torch.tensor(made["engine_raw"], dtype=torch.float64)
                zero(None)
                full = self.window_pass(made, shared=None)["raw"].double().cpu()
                for tag in tags:
                    zero(tag)
                    off = self.window_pass(made, shared=None)["raw"].double().cpu()
                    ok = torch.isfinite(engine) & torch.isfinite(full) & torch.isfinite(off)
                    d, y = (full - off)[ok], (engine - off)[ok]
                    s, b = sums[tag], d.abs() > big
                    for key, v in (("n", float(ok.sum())), ("d", d.sum()), ("dd", (d * d).sum()), ("y", y.sum()),
                                   ("yy", (y * y).sum()), ("yd", (y * d).sum()), ("ef", (engine - full)[ok].abs().sum()),
                                   ("ex", y.abs().sum()), ("nb", float(b.sum())), ("db", d[b].sum()), ("ddb", (d[b] ** 2).sum()),
                                   ("yb", y[b].sum()), ("yyb", (y[b] ** 2).sum()), ("ydb", (y[b] * d[b]).sum())):
                        s[key] += float(v)
                    del off
                del made, full
        finally:
            zero(None)

        def fit(n, sd, sdd, sy, syy, syd):
            if n < 2:
                return {"tokens": int(n)}
            vd, vy, c = sdd / n - (sd / n) ** 2, syy / n - (sy / n) ** 2, syd / n - sd * sy / n ** 2
            slope = c / vd if vd > 0 else None
            resid = max(vy - (c * c / vd if vd > 0 else 0.0), 0.0)
            se = (resid / (n * vd)) ** 0.5 if vd > 0 else None
            return {"tokens": int(n), "slope": None if slope is None else round(slope, 4),
                    "slope_se_iid": None if se is None else round(se, 5),
                    "corr": round(c / (vd * vy) ** 0.5, 4) if vd > 0 and vy > 0 else None,
                    "rms_d": round(sdd / n, 8) ** 0.5, "mean_d": round(sd / n, 6)}

        rows = {}
        for tag in tags:
            s = sums[tag]
            rows[tag] = {"params": counts[tag], **fit(s["n"], s["d"], s["dd"], s["y"], s["yy"], s["yd"]),
                         "abs_e_minus_full": round(s["ef"] / max(1.0, s["n"]), 5),
                         "abs_e_minus_off": round(s["ex"] / max(1.0, s["n"]), 5),
                         f"big_d_over_{big}": fit(s["nb"], s["db"], s["ddb"], s["yb"], s["yyb"], s["ydb"])}
        return {"windows": windows, "seconds": round(time.time() - began, 1), "parts": rows}

    def op_r3_check(self, cmd):
        """R3 check: one update batch at the current state, each window passed with the trainer's own
        routing ("off") and with the play server's ("served"; arena.served_route): per sampled token engine minus
        trainer raw log-probability (mean, mean |.|, rms), the TIS ratio's shares at or above the cap and at or
        below 1/cap, the rows whose own top-k equals the served one, and the cosine between the batch's policy
        gradients (the update's loss and window scales) under the two routings. Nothing is stepped."""
        import torch

        self.pin_chunk(cmd)
        specs = [a for a in self.expand_cmd(cmd) if a["w"] > 0 or (cmd.get("signed") and a["w"] < 0)]
        specs = sorted(specs, key=lambda a: -abs(a["w"]))[: int(cmd.get("max_windows", 8))]
        self.policy = {"loss": cmd.get("policy_loss") or "token", "clip": cmd.get("gspo_clip"),
                       "norm": cmd.get("gspo_norm") or "tokens"}
        cap = float(self.args.tis_cap or 2.0)
        saved = ROUTE["router"]
        stats, began = {}, time.time()
        grads = {}
        try:
            for mode in ("off", "served"):
                ROUTE.update(router="served", agree=None, used=None, unserved_calls=0)  # tables resolve in served mode
                SERVED_AUDIT.clear()
                s = stats[mode] = {"windows": 0, "tokens": 0, "sum": 0.0, "abs": 0.0, "sq": 0.0, "valid": 0,
                                   "tis_hi": 0, "tis_lo": 0, "loss": 0.0}
                self.optimizer.zero_grad(set_to_none=True)
                for spec in specs:
                    ROUTE["router"] = "served"
                    made = self.attempt_window(spec, 0, False)
                    if made is None or made.get("served") is None:
                        s["skipped"] = s.get("skipped", 0) + 1
                        continue
                    ROUTE["router"] = mode
                    out = self.window_pass(made, scale=1.0 / max(1, len(specs)))
                    engine = torch.tensor(made["engine_raw"], dtype=torch.float64)
                    raw = out["raw"].double()
                    ok = torch.isfinite(engine) & torch.isfinite(raw)
                    d = (engine - raw)[ok]
                    ratio = torch.exp(out["actor"].double() - made["arrays"]["reference"].double().cpu()).nan_to_num(1.0, 1.0, 1.0)
                    valid = out["valid"].bool()
                    s["windows"] += 1
                    s["tokens"] += int(ok.sum())
                    s["sum"] += float(d.sum())
                    s["abs"] += float(d.abs().sum())
                    s["sq"] += float((d * d).sum())
                    s["valid"] += int(valid.sum())
                    s["tis_hi"] += int(((ratio >= cap) & valid).sum())
                    s["tis_lo"] += int(((ratio <= 1.0 / cap) & valid).sum())
                    s["loss"] += float(out["loss"] or 0.0)
                    del made, out
                    torch.cuda.empty_cache()
                if mode == "served":
                    s["route"] = {"rows": int(ROUTE["used"] or 0), "agree": int(ROUTE["agree"] or 0),
                                  "unserved_calls": ROUTE["unserved_calls"], "resolve": dict(SERVED_AUDIT)}
                if mode == "off":
                    grads = {name: p.grad.detach().to("cpu", torch.float32) for name, p in self.params if p.grad is not None}
                else:
                    dot = na = nb = 0.0
                    for name, p in self.params:
                        if p.grad is None or name not in grads:
                            continue
                        g, h = p.grad.detach().float(), grads[name].to(p.grad.device)
                        dot += float((g * h).sum())
                        na += float((h * h).sum())
                        nb += float((g * g).sum())
                        del h
                    stats["grad_cos"] = round(dot / math.sqrt(na * nb), 5) if na > 0 and nb > 0 else None
                    stats["grad_norm"] = {"off": round(math.sqrt(na), 6), "served": round(math.sqrt(nb), 6)}
        finally:
            ROUTE.update(router=saved, call=None, window=None)
            self.optimizer.zero_grad(set_to_none=True)
            grads.clear()
            torch.cuda.empty_cache()
        for mode in ("off", "served"):
            s = stats[mode]
            n, v = max(1, s["tokens"]), max(1, s["valid"])
            s.update(mean=round(s.pop("sum") / n, 5), mean_abs=round(s.pop("abs") / n, 5),
                     rms=round(math.sqrt(s.pop("sq") / n), 5), tis_hi_share=round(s.pop("tis_hi") / v, 5),
                     tis_lo_share=round(s.pop("tis_lo") / v, 5))
        route = stats["served"].get("route") or {}
        if route.get("rows"):
            route["agree_share"] = round(route["agree"] / route["rows"], 5)
        stats["seconds"] = round(time.time() - began, 1)
        return stats

    def op_old_use(self, cmd):
        """rl1-gold: a kept copy becomes GSPO's pi_old for the next update."""
        kept = (getattr(self, "kept", {}) or {})[cmd["label"]]
        self.old = {key: row["actor"] for key, row in kept.items()}
        return {"windows": len(self.old), "label": cmd["label"]}

    def op_old_compare(self, cmd):
        """rl1-gold: per-reply log s between two kept copies (``a`` as pi_theta, ``b`` as pi_old)."""
        kept = getattr(self, "kept", {}) or {}
        a, b = kept[cmd["a"]], kept[cmd["b"]]
        rows = [reply_gaps(row["actor"], (b.get(key) or {}).get("actor"), row["reply"], row["valid"], row["weights"])
                for key, row in a.items()]
        return {"compare": {"a": cmd["a"], "b": cmd["b"], **gap_stats(rows, tuple(cmd.get("clip") or (3e-4, 4e-4)))}}

    def op_adapter_scale(self, cmd):
        """rl1-gold (the KL floor): every lora_B at ``factor`` x its value at the first call (the learned delta scaled;
        a smooth policy moves KL ~ factor^2); factor 1 restores it."""
        import torch

        if not hasattr(self, "scaled_from"):
            self.scaled_from = {name: param.detach().clone() for name, param in self.params if "lora_B" in name}
        factor = float(cmd["factor"])
        with torch.no_grad():
            for name, param in self.params:
                if name in self.scaled_from:
                    param.copy_(self.scaled_from[name] * factor)
        if factor == 1.0:
            del self.scaled_from
        return {"factor": factor, "adapter_norm": self.adapter_norm()}

    def op_route(self, cmd):
        """rl1-route: ``router`` / ``qsa`` each off | capture (clears that record) | replay (from the first recorded
        call); ``select`` the QSA scoring mode (qsa | stable | dense). Every op restarts the replay pointers."""
        route_patch()
        for key, rec in (("router", "rec"), ("qsa", "qsa_rec")):
            mode = cmd.get(key)
            if mode is None:
                continue
            if mode not in ("off", "capture", "replay") and not (key == "router" and mode == "served"):
                raise ValueError(f"route {key}: {mode}")
            if mode == "capture":
                ROUTE[rec] = []
            ROUTE[key] = mode
        if cmd.get("select"):
            SELECT["mode"] = cmd["select"]
        used = {"router": ROUTE["i"], "qsa": ROUTE["qi"]}
        ROUTE["i"] = ROUTE["qi"] = 0
        served = {"rows": int(ROUTE["used"]) if ROUTE["used"] is not None else 0,
                  "agree": int(ROUTE["agree"]) if ROUTE["agree"] is not None else 0,
                  "unserved_calls": ROUTE["unserved_calls"], "windows": dict(SERVED_AUDIT)}
        if cmd.get("reset_counts"):
            ROUTE.update(agree=None, used=None, unserved_calls=0)
            SERVED.clear()
            SERVED_AUDIT.clear()
        return {"router": ROUTE["router"], "qsa": ROUTE["qsa"], "select": SELECT["mode"], "recorded": len(ROUTE["rec"]),
                "qsa_recorded": len(ROUTE["qsa_rec"]), "replayed_before": used, "calls": ROUTE["calls"],
                "qsa_calls": ROUTE["qsa_calls"], "indexer": dict(INDEXER), "served": served}

    def op_expert_ref(self, cmd):
        """rl1-route item 3d(i): one MoE layer's routed experts through the trainer's expert-LoRA path against a dense
        float32 reference (dequantised base + scale * B_e A_e in PEFT's ParamWrapper layout: lora_A [R E, in] expert-
        major rows, lora_B [out, R E] expert-minor columns, merge_lora.py's convention), on random inputs routed by the
        layer's own router. Outputs with the adapter x ``boost`` and without; gradients of a fixed projection of the
        output with respect to the four LoRA factors at the trained state (boost 1)."""
        import torch
        import torch.nn.functional as F
        from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import dequantize_expert_weight

        layer, T, boost = int(cmd.get("layer", 24)), int(cmd.get("tokens", 4096)), float(cmd.get("boost", 8.0))
        mlp = next(m for n, m in self.model.named_modules() if n.endswith(f"language_model.layers.{layer}.mlp"))
        outer, mid = mlp.experts, mlp.experts.base_layer
        inner = mid.base_layer
        named = dict(self.params)
        stem = next(n for n in named if n.endswith(f"language_model.layers.{layer}.mlp.experts.lora_A.default.weight"))
        stem = stem[: -len("lora_A.default.weight")]
        P = {"A13": named[stem + "base_layer.lora_A.default.weight"], "B13": named[stem + "base_layer.lora_B.default.weight"],
             "A2": named[stem + "lora_A.default.weight"], "B2": named[stem + "lora_B.default.weight"]}
        scale = lambda m: float((getattr(m, "scaling", None) or {}).get("default", 2.0))
        s13, s2 = scale(mid), scale(outer)
        E = int(inner.num_experts)
        H, I, R = int(P["A13"].shape[1]), int(P["A2"].shape[1]), P["A13"].shape[0] // E  # lora_A [R E, in]
        concat = bool(getattr(inner, "is_concatenated", True))
        dev = P["A13"].device
        gen = torch.Generator(device=dev).manual_seed(int(cmd.get("seed", 0)))
        x = torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32).to(torch.bfloat16)
        proj = torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32)
        saved = {k: p.detach().clone() for k, p in P.items()}
        out = {"layer": layer, "tokens": T, "experts": E, "rank": R, "scales": [s13, s2], "boost": boost, "concat": concat}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # the model's own forward is autocast
                _, w, idx = mlp.gate(x)

            def kernel(grad=False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = outer(x, idx, w)
                if grad:
                    (y.float() * proj).sum().backward()
                return y.detach().float()

            self.optimizer.zero_grad(set_to_none=True)
            y1 = kernel(grad=True)  # trained state: output and gradients
            gk = {k: (p.grad.detach().float().clone() if p.grad is not None else None) for k, p in P.items()}
            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for k in ("B13", "B2"):
                    P[k].mul_(boost)
                yb = kernel()
                for k in ("B13", "B2"):
                    P[k].zero_()
                y0 = kernel()
                for k, v in saved.items():
                    P[k].copy_(v)
            def raw(name):  # the frozen base parameter (PEFT parametrizes it only inside the wrapper's forward)
                pz = getattr(inner, "parametrizations", None)
                return pz[name].original if pz is not None and name in pz else getattr(inner, name)

            W13 = dequantize_expert_weight(raw("gate_up_proj")).float()
            W2 = dequantize_expert_weight(raw("down_proj")).float()
            out["base_shapes"] = [list(W13.shape), list(W2.shape)]
            if W13.shape[1] == H and W13.shape[2] == 2 * I:  # stored [E, in, out]: the reference wants [E, out, in]
                W13 = W13.transpose(1, 2)
            if W2.shape[1] == I and W2.shape[2] == H:
                W2 = W2.transpose(1, 2)
            leaves = {k: v.float().clone().requires_grad_(True) for k, v in saved.items()}
            xf, wf = x.float(), w.float()
            hit = [int(e) for e in torch.unique(idx).tolist()]

            def ref(b, lora=True, cat=concat, leaf=None):
                L = leaf or leaves
                A13 = L["A13"].reshape(E, R, H)
                B13 = L["B13"].reshape(2 * I, R, E)
                A2 = L["A2"].reshape(E, R, I)
                B2 = L["B2"].reshape(H, R, E)
                y = torch.zeros(T, H, device=dev, dtype=torch.float32)
                for e in hit:
                    tok, pos = (idx == e).nonzero(as_tuple=True)
                    w13, w2 = W13[e], W2[e]
                    if lora:
                        w13 = w13 + b * s13 * (B13[:, :, e] @ A13[e])
                        w2 = w2 + b * s2 * (B2[:, :, e] @ A2[e])
                    h = xf[tok] @ w13.T
                    g, u = (h[:, :I], h[:, I:]) if cat else (h[:, 0::2], h[:, 1::2])
                    o = (F.silu(g) * u) @ w2.T
                    y = y.index_add(0, tok, o * wf[tok, pos, None])
                return y

            rel = lambda a, b: float((a - b).norm() / b.norm()) if float(b.norm()) else None
            cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm())) if float(a.norm() * b.norm()) else None
            # stage D bias read (22 Oct): the kernel's error along the reference (a systematic scale: + means the kernel's
            # output is larger) and its mean, beside the relative error
            bias = lambda a, b: {"scale": float(((a - b) * b).sum() / (b * b).sum()) if float(b.norm()) else None,
                                 "mean": float((a - b).mean()), "abs_mean": float((a - b).abs().mean())}
            with torch.no_grad():
                r0 = {"concat": ref(0.0, lora=False, cat=True), "interleaved": ref(0.0, lora=False, cat=False)}
                out["base_rel_err"] = {k: rel(y0, v) for k, v in r0.items()}
                cat = min(r0, key=lambda k: out["base_rel_err"][k]) == "concat"
                out["base_bias"] = bias(y0, r0["concat" if cat else "interleaved"])
                rb = ref(boost, cat=cat)
                d_kernel, d_ref = yb - y0, rb - r0["concat" if cat else "interleaved"]
                base = r0["concat" if cat else "interleaved"]
                out["delta"] = {"cos": cos(d_kernel, d_ref), "rel_err": rel(d_kernel, d_ref),
                                "effect_rel": float(d_ref.norm() / base.norm())}
                r1 = ref(1.0, cat=cat)
                out["trained_rel_err"] = rel(y1, r1)
                out["trained_bias"] = bias(y1, r1)
                out["boost_bias"] = bias(yb, rb)
                del r1
                # wrong-layout controls: the same delta with lora_B read expert-major (a layout bug's signature)
                bad = {**leaves, "B13": leaves["B13"].reshape(2 * I, E, R).permute(0, 2, 1).reshape(2 * I, R * E),
                       "B2": leaves["B2"].reshape(H, E, R).permute(0, 2, 1).reshape(H, R * E)}
                out["delta_wrong_layout_cos"] = cos(d_kernel, ref(boost, cat=cat, leaf=bad) - base)
            yr = ref(1.0, cat=cat)
            (yr * proj).sum().backward()
            out["grads"] = {k: {"cos": cos(gk[k], leaves[k].grad), "rel_err": rel(gk[k], leaves[k].grad),
                                "norm_kernel": float(gk[k].norm()), "norm_ref": float(leaves[k].grad.norm())}
                            for k in P if gk[k] is not None}
            out["experts_hit"] = len(hit)
            del W13, W2, leaves
        finally:
            with torch.no_grad():
                for k, v in saved.items():
                    P[k].copy_(v)
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        return out

    def op_k3_score(self, cmd):
        """mg.k3_measure (23 Oct): the step's dose, forward only. ``attempts`` (the subsample whose pi_old the step
        kept with ``keep_old``) scored at the current state; GSPO's per-reply k3 = mean((s - 1) - log s), s = exp(mean
        token log-ratio over the reply), per-token k3 (diagnostic) and the share GSPO's ``clip`` band would clip. The
        kept pi_old is dropped afterwards. k3_hold: windows come from the step's kept host copies (no encode).
        ``window_gap`` as op_update's (MFU-1: wall outside the passes, by part)."""
        import torch

        began = time.time()
        self.pin_chunk(cmd)
        specs = [a for a in self.expand_cmd(cmd) if a["w"] > 0 or (cmd.get("signed") and a["w"] < 0)]
        expanded = time.time() - began
        self.ahead_start(cmd)
        shared = {} if cmd.get("share_prefix") else None
        if shared is not None:
            groups = {}
            for a in specs:
                groups.setdefault(a.get("share"), len(groups))
            specs = sorted(specs, key=lambda a: (groups[a.get("share")], not a.get("first_window", True)))
        old, rows, tok = dict(getattr(self, "old", None) or {}), [], [0.0, 0]
        tok_vals, top = [], None  # per-token k3 (approx_kl_tok's tail: p99, max, the top token; RL 24 Oct, v6g 0.08 -> 5.7)
        next_window, close, feed = self.window_feed(specs, cmd, old=False, from_hold=True)
        gaps, last = [], time.time()
        try:
            with Sampler() as sampler:
                for _ in specs:
                    asked = time.time()
                    made = next_window()
                    waited = time.time() - asked
                    if made is None:
                        continue
                    out = self.window_pass(made, shared=shared)
                    before = old.get(self.old_key(made))
                    reply = made["arrays"]["reply"].cpu()
                    weights = made["arrays"]["weights"].cpu() if "weights" in made["arrays"] else None
                    rows.append(reply_gaps(out["actor"], before, reply, out["valid"], weights))
                    if before is not None and before.numel() == out["actor"].numel():
                        g = torch.where(out["valid"], out["actor"].double() - before.double(),
                                        torch.zeros_like(out["actor"], dtype=torch.float64))
                        per = ((torch.exp(g) - 1.0) - g)
                        tok[0] += float(per[out["valid"]].sum())
                        tok[1] += int(out["valid"].sum())
                        vals = per[out["valid"]].float()
                        if vals.numel():
                            tok_vals.append(vals.cpu())
                            i = int(torch.argmax(torch.where(out["valid"], per, torch.full_like(per, -1.0))))
                            if top is None or float(per.reshape(-1)[i]) > top["k3"]:
                                top = {"k3": float(per.reshape(-1)[i]), "g": float(g.reshape(-1)[i]),
                                       "logp_old": float(before.reshape(-1)[i]), "window": str(made.get("name")),
                                       "index": i, "token": self.token_at(made, i)}
                    now = time.time()
                    busy = out["prefix_seconds"] + out["forward_seconds"]
                    gaps.append({"gap": now - last - busy, "next": waited, **(out.get("host") or {})})
                    last = now
                    del made, out
        finally:
            close()
            if shared is not None:
                shared.clear()
            self.old = {}
            held, self.k3_cache = getattr(self, "k3_cache", None), None
        log_s = [x for r in rows if not r.get("missing") for x in r["log_s"]]
        clip = tuple(cmd.get("gspo_clip") or (3e-4, 4e-4))
        stats = gap_stats(rows, clip)
        k3 = sum(math.expm1(x) - x for x in log_s) / len(log_s) if log_s else None
        parts = ("next", "pre", "post", "vision", "rope")
        tail = None
        if tok_vals:
            allv = torch.cat(tok_vals)
            tail = {"p99": round(float(torch.quantile(allv[:16_000_000], 0.99)), 6), "max": round(float(allv.max()), 6),
                    "over_1": int((allv > 1.0).sum()), "tokens": int(allv.numel()), "top": top}
        return {"ok": k3 is not None and not stats.get("missing"), "approx_kl": None if k3 is None else round(k3, 8),
                "approx_kl_tok": round(tok[0] / tok[1], 6) if tok[1] else None, "k3_tok_tail": tail,
                "clip_frac": stats.get("clipped_share"), "replies": len(log_s), "windows": len(rows),
                "missing": stats.get("missing"), "kept": len(old), "log_s_abs_p90": stats.get("log_s_abs_p90"),
                "seconds": round(time.time() - began, 1), "window_wait_seconds": round(feed["waits"], 1),
                "expand_seconds": round(expanded, 2),
                "window_gap": {"sum": round(sum(g["gap"] for g in gaps), 1),
                               "host": {k: round(sum(g.get(k) or 0.0 for g in gaps), 1) for k in parts}},
                "encode": {"encoded": feed["encoded"], "hold_hits": feed["hold_hits"], "ahead_hits": feed["ahead_hits"],
                           "ahead_sent": feed["ahead_sent"],
                           "held": None if held is None else held.summary()},
                **sampler.summary()}

    def op_score(self, cmd):
        return {"windows": self.score_windows(cmd["windows"], cmd.get("label"), hold=bool(cmd.get("hold_previous")))}

    def forget_unaccepted(self):
        """After a rollback: a rejected state's probe scores can never be accepted as the reference."""
        for refs in self.references.values():
            refs.pop("last", None)

    def op_accept_reference(self, cmd):
        """mg5e: the probes' last scored state becomes the KL-to-previous reference (the loop serves that adapter)."""
        missing = [name for name, refs in self.references.items() if "last" not in refs]
        if missing:
            raise ValueError(f"no scored state to accept for {missing}")
        for refs in self.references.values():
            refs["previous"] = refs["last"]
        return {"accepted": sorted(self.references)}

    def adapter_norm(self):
        if PROFILE != "pth":
            return round(math.sqrt(sum(float(p.detach().float().pow(2).sum()) for _, p in self.params)), 5)
        import torch

        with torch.no_grad():  # one host sync, not one per tensor
            return round(float(torch.stack([p.detach().float().pow(2).sum() for _, p in self.params]).sum().sqrt()), 5)

    def expand_cmd(self, cmd):
        return self.expand(cmd["attempts"], procs=int(cmd.get("encode_procs") or 0))

    def expand(self, specs, procs=0):
        """A spec with ``whole`` becomes one spec per planned window end that holds weighted replies (every token of a
        long continuation trained, each window once). MFU-1 (23 Oct): a spec expanded before (same spec, same rollout
        size and mtime) is taken from a memo (k3_score re-expands the step's tries), and with ``procs`` encoder
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

    def op_carry_check(self, cmd):
        """MFU-1 C read: the same windows at lr 0, each arm against the first (base: pinned chunk, ``ref`` keep):
        the base at half the chunk (the chunking floor), the base again (repeat), prefix_carry strict and "margin".
        Per arm: train / prefix seconds, prefix tokens computed (cut - prefix_start), carried windows, moved
        tokens and ref_compare (logprob |diff| mean / p99 / max, gradient cosine)."""
        chunk = int(getattr(self, "path_chunk", 16384))
        common = {k: v for k, v in cmd.items() if k not in ("op", "prefix_carry", "ref", "arms")}
        arms = cmd.get("arms") or ["base", "half_chunk", "repeat", "carry", "carry_margin"]
        out, refs, ties, first_state = {"chunk": chunk, "arms": {}}, {}, TIES["mode"], None
        for arm in arms:
            # an arm is a name above or (MFU-1 floor read) a dict: name, ref keep | compare | both (compared, then
            # kept), ref_of / keep_as (reference labels, default "base"), chunk "half", fixes (op_paths keys of this
            # arm only), ties ("index" | "topk": QSA budget-edge ties), prefix_carry
            spec = arm if isinstance(arm, dict) else {
                "name": arm, "ref": "keep" if arm == "base" else "compare", "chunk": "half" if arm == "half_chunk" else None,
                "prefix_carry": {"carry": True, "carry_margin": "margin"}.get(arm)}
            name, mode = spec["name"], spec.get("ref", "compare")
            extra = {**(spec.get("fixes") or {}), "ref": mode, "ref_rows": bool(cmd.get("ref_rows"))}
            if spec.get("prefix_carry") is not None:
                extra["prefix_carry"] = spec["prefix_carry"]
            if mode in ("compare", "both"):
                self.ref = refs.get(spec.get("ref_of", "base"))
            before = {**CARRY, "reject": dict(CARRY["reject"])}
            half = spec.get("chunk") == "half"
            if half:
                self.prefix_chunk = self.path_chunk = chunk // 2
            TIES["mode"] = spec.get("ties") or ties
            arm_cmd = {**{k: v for k, v in common.items() if not (half and k == "prefix_chunk")}, **extra, "lr": 0.0}
            if spec.get("checkpoint"):  # MFU-1 sampler gap: which adapter state played the tries
                self.op_load_checkpoint({"path": spec["checkpoint"]})
            dumped = len(GAP_DUMP)
            try:
                if spec.get("fixes"):  # the dispatcher applies op_paths once per op: an arm's own paths here (m1fl1)
                    op_paths(getattr(self, "model", None), arm_cmd)
                state = paths_state()
                r = self.op_update(arm_cmd)
            finally:
                self.prefix_chunk = self.path_chunk = chunk
                TIES["mode"] = ties
                if spec.get("fixes"):
                    op_paths(getattr(self, "model", None), cmd)
            first_state = first_state if out["arms"] else state
            if (spec.get("fixes") or spec.get("ties")) and state == first_state:
                r = {**r, "ok": False, "error": f"arm {name}: its fixes / ties left the live paths as the first arm's"}
            for row in GAP_DUMP[dumped:]:
                row["arm"] = name
            if mode in ("keep", "both") and r.get("ok"):
                refs[spec.get("keep_as", "base" if mode == "keep" else name)] = self.ref
            ws = r.get("windows") or []
            out["arms"][name] = {
                "ok": r.get("ok"), "error": r.get("error"), "train_seconds": r.get("train_seconds"),
                "prefix_seconds": round(sum(w.get("prefix_seconds") or 0 for w in ws), 2),
                "window_seconds": round(sum(w.get("seconds") or 0 for w in ws), 2),
                "windows": len(ws), "prefix_tokens": sum(w["cut"] - w.get("prefix_start", 0) for w in ws),
                "tail_tokens": sum(w["tail"] for w in ws), "tokens": sum(w["tokens"] for w in ws),
                "carried": sum(bool(w.get("prefix_carried")) for w in ws),
                "moved": sum(int(w.get("carry_moved") or 0) for w in ws),
                "carry": {**{k: CARRY[k] - before[k] for k in CARRY if k != "reject"},
                          "reject": {k: v - before["reject"].get(k, 0) for k, v in CARRY["reject"].items()
                                     if v - before["reject"].get(k, 0)}}, "compare": r.get("ref_compare"),
                "card_peak_gib": r.get("card_peak_gib"), "torch_peak_gib": r.get("torch_peak_gib"), "paths": state,
                "engine_gap": windows_engine_gap(ws)}
        self.ref = None
        if cmd.get("gap_dump") and GAP_DUMP:  # MFU-1 sampler gap: the pairs of every arm, on the run's output volume
            import torch

            path = Path(cmd["gap_dump"]) if isinstance(cmd["gap_dump"], str) else (
                Path(os.environ.get("ARC_VALIDATION_OUTPUT") or "/tmp") / "gap_dump.pt")  # m1gp1 lost its dump in /tmp
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(list(GAP_DUMP), path)
            out["gap_dump"] = {"path": str(path), "windows": len(GAP_DUMP)}
            GAP_DUMP.clear()
        return {"ok": all(a["ok"] for a in out["arms"].values()), **out}

    def op_causal_check(self, cmd):
        """MFU-1 floor hunt (24 Oct): does a row's output depend on tokens after it in the same call? One window's
        ids through the model without gradients as one call of ``length`` tokens and as one call of ``length`` +
        ``extra`` (same start, same tokens before ``length``). Per decoder layer and per direct child of it, max and
        mean |diff| over rows < ``length`` (a causal model: GEMM-shape noise only), and the next-token logprobs of
        the last ``rows`` rows before ``length``. Per entry of ``configs``: op_paths keys and QSA ties."""
        import torch

        windows = self.expand([dict(cmd["attempts"][int(cmd.get("attempt", 0))])])
        made = self.attempt_window(windows[-1], 0, False)
        inputs = made["inputs"]
        ids = inputs["input_ids"]
        total = int(ids.shape[1])
        length = min(int(cmd.get("length", 8704)), total - 1)
        longer = min(length + int(cmd.get("extra", 7680)), total)
        rows = min(int(cmd.get("rows", 512)), length - 1)
        piece = fq.positions_and_images(self.model, inputs)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        layers = [(n, m) for n, m in base.named_modules() if type(m).__name__.endswith("DecoderLayer")]
        watch = [(n, m) for ln, lm in layers for n, m in [(ln, lm)] + [(f"{ln}.{c}", cm) for c, cm in lm.named_children()]]
        core = getattr(base, "model", base)
        norm = getattr(core, "language_model", core)  # the text model: its last hidden state feeds the head
        head = base.get_output_embeddings()
        image = getattr(getattr(base, "config", None), "image_token_id", None)
        special = {t for t in (image, getattr(getattr(base, "config", None), "vision_start_token_id", None),
                               getattr(getattr(base, "config", None), "vision_end_token_id", None)) if t is not None}
        # PTH (24 Oct): the long call with the text tokens of [length, longer) reversed (images and their markers in
        # place): rows < length that change against the long call read later content (leakage), not only its length
        perturbed = ids[:, :longer].clone()
        text = [i for i in range(length, longer) if int(ids[0, i]) not in special]
        perturbed[0, text] = ids[0, text[::-1]] if text else perturbed[0, text]
        out = {"length": length, "longer": longer, "total": total, "reversed": len(text), "configs": {}}
        ties, first_state = TIES["mode"], None
        for conf in cmd.get("configs") or [{"name": "as_is"}]:
            store, diffs, content, hooks = {}, {}, {}, []
            phase = {"n": None}

            def hook(name):
                def run(module, args, output):
                    t = output[0] if isinstance(output, (tuple, list)) else output
                    if not torch.is_tensor(t) or t.dim() < 2:
                        return
                    t = t[0] if t.dim() == 3 else t
                    if t.shape[0] < length:
                        return
                    cut = t[:length].detach()
                    if phase["n"] == "short":
                        store[name] = cut.clone()
                    elif name in store or name + "#long" in store:  # long vs short; reversed vs long (content)
                        ref = store[name] if phase["n"] == "long" else store.pop(name + "#long", None)
                        if ref is None:
                            return
                        d = (cut.float() - ref.float()).abs()
                        scale = float(cut.float().abs().mean()) or 1.0
                        (diffs if phase["n"] == "long" else content)[name] = {
                            "max": round(float(d.max()), 5), "mean": round(float(d.mean()), 6),
                            "rel_mean": round(float(d.mean()) / scale, 6),
                            "worst_row": int(d.amax(dim=-1).argmax()) if d.dim() == 2 else None}
                        if phase["n"] == "long":
                            store.pop(name)
                            store[name + "#long"] = cut.clone()
                return run

            for name, module in watch:
                hooks.append(module.register_forward_hook(hook(name)))
            TIES["mode"] = conf.get("ties") or ties
            if conf.get("fixes"):
                op_paths(self.model, {**cmd, **conf["fixes"]})
            state = paths_state()
            first_state = state if first_state is None else first_state
            replay = str(conf.get("replay") or "")  # "moe" / "moe+qsa": the short call's picks on its rows in the others
            if replay:
                route_patch()
            saved = {k: ROUTE.get(k) for k in ("router", "qsa", "rec", "i", "qsa_map")}
            logps = {}
            self.model.eval()
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for tag, n in (("short", length), ("long", longer), ("reversed", longer)):
                        phase["n"] = tag
                        call_ids = perturbed if tag == "reversed" else ids
                        if replay:
                            ROUTE.update(router="capture" if tag == "short" else "prefix", i=0)
                            if tag == "short":
                                ROUTE.update(rec=[], qsa_map={})
                            if "qsa" in replay:
                                ROUTE["qsa"] = "capture_map" if tag == "short" else "prefix"
                        ROUTE["call"] = served_call(0, n)
                        causal_reset()
                        grab = {}
                        h = norm.register_forward_hook(lambda m, a, o: grab.update(x=(o[0] if isinstance(o, (tuple, list))
                                                                                        else o.last_hidden_state)[0]))
                        self.model(input_ids=call_ids[:, :n], attention_mask=torch.ones_like(ids[:, :n]), use_cache=False,
                                   logits_to_keep=1, **piece(0, n))
                        h.remove()
                        if "x" in grab:
                            x = grab["x"][length - rows - 1:length - 1]
                            lp = torch.log_softmax(torch.nn.functional.linear(x.to(head.weight.dtype), head.weight).float(), -1)
                            logps[tag] = lp.gather(-1, ids[0, length - rows:length, None].to(lp.device))[:, 0]
            finally:
                ROUTE["call"] = None
                if replay:
                    ROUTE.update(saved)
                self.model.train()
                TIES["mode"] = ties
                for h in hooks:
                    h.remove()
                if conf.get("fixes"):
                    op_paths(self.model, cmd)
            bar = float(cmd.get("bar", 1e-3))
            first = lambda table: next((n for n, _ in watch if n in table and table[n]["rel_mean"] > bar), None)
            read = lambda a, b: None if a not in logps or b not in logps else (lambda d: {
                "rows": int(d.numel()), "mean": round(float(d.mean()), 5), "max": round(float(d.max()), 4),
                "worst_row": length - rows + int(d.argmax())})((logps[a] - logps[b]).abs())
            noop = bool(conf.get("fixes") or conf.get("ties")) and state == first_state
            out["configs"][conf.get("name", "as_is")] = {
                "first_over_bar": first(diffs), "first_over_bar_content": first(content), "modules": diffs,
                "content": content, "logprob": read("long", "short"), "logprob_content": read("reversed", "long"),
                "paths": state, **({"error": "its fixes / ties left the live paths as the first config's"} if noop else {})}
        return {"ok": not any("error" in c for c in out["configs"].values()), **out}

    def ref_compare(self, actor, grads, geom=None):
        """This step against the kept reference: per-token logprob |diff| over trained tokens (mean, p99, max), the
        gradient cosine and norm ratio (float64); with ``geom`` (window_geom per window) also the worst rows and
        per-window reads against the chunk grid and the images (``rows_read``)."""
        import torch

        diffs, per = [], {}
        for name, (now, valid) in actor.items():
            ref = self.ref["actor"].get(name)
            if ref is not None and ref[0].shape == now.shape:
                mask = valid & ref[1]
                diffs.append((now[mask] - ref[0][mask]).abs())
                per[name] = ((now - ref[0]).abs(), mask)
        d = torch.cat(diffs) if diffs else torch.zeros(1)
        dot = na = nb = 0.0
        for g, r in zip(grads, self.ref["grads"]):
            a, b = g.detach().double(), r.to(g.device).double()
            dot += float((a * b).sum())
            na += float((a * a).sum())
            nb += float((b * b).sum())
        tries = self.ref_by_try(actor)
        return {"windows": len(diffs), "of": len(actor), "tokens": int(d.numel()), "by_try": tries,
                "logprob_mean_abs": float(d.mean()), "logprob_p99_abs": float(d.quantile(0.99)) if d.numel() < 2**24 else None,
                "logprob_max_abs": float(d.max()), "grad_cos": dot / max(1e-300, (na * nb) ** 0.5),
                "beyond_ln2": float((d > math.log(2)).float().mean()),
                "grad_norm_ratio": (na / nb) ** 0.5 if nb else None, "grads": len(grads), "ref_grads": len(self.ref["grads"]),
                **({"rows": rows_read(per, geom, int(getattr(self, "path_chunk", 16384)))} if geom else {})}

    def ref_by_try(self, actor):
        """Window-size reads (MFU-1): trained-token logprobs per try (windows ``name#i`` in order, valid tokens
        concatenated) against the reference's, where the try trains the same number of tokens both ways."""
        import torch

        def tries(rows):
            out = {}
            order = lambda k: (k.rpartition("#")[0] or k, int(k.rpartition("#")[2]) if k.rpartition("#")[2].isdigit() else 0)
            for name in sorted(rows, key=order):
                now, valid = rows[name]
                out.setdefault(order(name)[0], []).append(now[valid])
            return {k: torch.cat(v) for k, v in out.items()}

        mine, ref = tries(actor), tries(self.ref["actor"])
        same = [k for k in mine if k in ref and mine[k].numel() == ref[k].numel()]
        if not same:
            return {"tries": len(mine), "aligned": 0}
        d = torch.cat([(mine[k] - ref[k]).abs() for k in same])
        return {"tries": len(mine), "aligned": len(same), "tokens": int(d.numel()), "mean_abs": float(d.mean()),
                "max_abs": float(d.max()), "count_mismatch": sorted(k for k in mine if k in ref and k not in same)[:8],
                "logprob_sum_delta": float(sum(float(mine[k].sum() - ref[k].sum()) for k in same))}

    def op_accumulate(self, cmd):
        """v7 streamed accumulation: one group's windows forward/backward at the current state, gradients summed into
        the parameters' .grad across calls (no divide, clip or step). ``first`` starts a new batch (and a new k3 store);
        ``keep_old`` names windows whose actor log-probabilities are kept as pi_old for the post-step k3 measure."""
        if cmd.get("first") or getattr(self, "acc", None) is None:
            self.acc_reset()
        return self.op_update({**cmd, "accumulate": True})

    def op_step(self, cmd):
        """v7: the batch's step from the accumulated gradients: the same divisor, clip and AdamW step as one
        ``op_update`` over all the accumulated windows."""
        if not getattr(self, "acc", None) or not self.acc["calls"]:
            return {"ok": False, "error": "nothing accumulated"}
        try:
            return self.op_update({**cmd, "attempts": [], "from_acc": True})
        finally:
            self.acc = None

    def op_acc_reset(self, cmd=None):
        self.acc_reset()
        return {"ok": True}

    def acc_reset(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.acc = {"tokens": 0, "losses": 0.0, "finite": True, "tries": {}, "entropy": [0.0, 0], "calls": 0,
                    "windows": 0}
        self.old = {}
        self.k3_cache = None  # k3_hold: a new batch's subsample

    def op_update(self, cmd):
        """``oom_fallback_window`` (MFU-1 D, mg.window_guard): a CUDA out-of-memory in the update frees the step's
        graph, gradients and cache and runs the same update again with every whole-try window at that size."""
        import torch

        fallback = int(cmd.get("oom_fallback_window") or 0)
        try:
            return self.update_once(cmd)
        except OOM_ERRORS as exc:
            # v7 accumulation: earlier calls' gradients would be lost; after the optimizer step: it may be applied
            if not fallback or cmd.get("accumulate") or getattr(self, "update_stepped", False):
                raise
            why = repr(exc)[:400]
        self.oom_free()
        attempts = [{**a, "window": fallback} if a.get("whole") and int(a.get("window") or 0) > fallback else a
                    for a in cmd["attempts"]]
        result = self.update_once({**cmd, "attempts": attempts})
        result["window_fallback"] = {"why": "oom", "to": fallback, "error": why,
                                     "from": max((int(a.get("window") or 0) for a in cmd["attempts"]), default=0)}
        return result

    def oom_free(self):
        """After an out-of-memory in an update: the feed's encoders closed, no gradient or cache kept, cache freed."""
        import gc

        import torch

        close, self.update_close = getattr(self, "update_close", None), None
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001 - a half-built feed; the retry builds its own
                pass
        for _, p in self.params:
            p.grad = None
        if getattr(self, "optimizer", None) is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            fq.reset_peaks()

    def update_once(self, cmd):
        import torch

        self.update_stepped = False
        if cmd.get("accumulate") and cmd.get("profile"):
            raise ValueError("accumulate does not profile")
        self.pin_chunk(cmd)
        if cmd.get("keep_old") and not cmd.get("accumulate"):  # one-call step: pi_old of this state only
            self.old = {}
        if PROFILE == "pth" and not getattr(self, "pinned", 0):  # fla's kernel modules can load at the first forward
            self.pinned = len(pin_autotune_keys())
        result = {"windows": [], "errors": [], "prefix_chunk": self.prefix_chunk}
        if PROFILE == "pth":
            result.update(autotune_pinned=self.pinned, autotune_pinned_late=sorted(set(PINNED_LATE)))
        lr = float(cmd["lr"])
        self.set_lr(lr)
        expand_began = time.time()
        specs = [a for a in self.expand_cmd(cmd) if a["w"] > 0 or (cmd.get("signed") and a["w"] < 0)]
        result["expand_seconds"] = round(time.time() - expand_began, 2)  # MFU-1: before the timed part, card idle
        self.ahead_start(cmd)
        self.policy = {"loss": cmd.get("policy_loss") or "token", "clip": cmd.get("gspo_clip"),
                       "norm": cmd.get("gspo_norm") or "tokens"}
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
            if cmd.get("prefix_carry") and self.policy["loss"] != "gspo" and not opd:
                # MFU-1 C: the same windows, each try's in conversation order (window ends), so a window continues
                # the previous one's carried end; requires prefix_snapshots (the candidate search lives there)
                specs = sorted(specs, key=lambda a: (groups[a.get("share")], a.get("name", "").split("#")[0],
                                                     int(a.get("end", 0))))
                shared["carry_on"] = "margin" if cmd["prefix_carry"] == "margin" else True
        else:
            specs = sorted(specs, key=lambda a: -abs(a["w"]))[: int(cmd.get("max_windows", 8))]
        per_try = {}
        if cmd.get("norm") in ("try", "cap"):  # windows per try (``expand`` names them try#i)
            for a in specs:
                key = a.get("name", "").split("#")[0]
                per_try[key] = per_try.get(key, 0) + 1
        fq.reset_peaks()
        losses, finite, tokens = 0.0, True, 0
        # the next windows are encoded (CPU: template, images) while the GPU runs this one: one encoding thread (no
        # second thread contending for the GIL with the launching one), ``prefetch`` windows queued ahead (MFU 3);
        # stage W: encoder processes, the old pass's host windows (``window_feed``)
        hold = None
        if cmd.get("k3_hold") and cmd.get("keep_old"):  # MFU-1: the k3 subsample's host windows kept for op_k3_score
            if getattr(self, "k3_cache", None) is None or not cmd.get("accumulate"):
                self.k3_cache = rl1_encode.Cache(float(cmd["k3_hold"]))
            names = set(cmd["keep_old"])
            hold = lambda spec: spec.get("name", "") in names or spec.get("name", "").split("#")[0] in names  # noqa: E731
        next_window, close, feed = self.window_feed(specs, cmd, hold=hold)
        self.update_close = close  # oom_free closes it after an out-of-memory
        prof = undo_annotate = None
        if cmd.get("profile"):  # MFU work: the whole step (windows, encode waits, optimizer) profiled
            from torch.profiler import ProfilerActivity, profile

            undo_annotate = annotate_modules(self.model)
            torch.cuda.synchronize()
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__()
            prof_began = time.time()
        with Sampler() as sampler:
            began = time.time()
            budget = int(cmd.get("batch_tokens") or 0) if shared is not None and not opd else 0
            if budget and not cmd.get("batch_diagnostic"):  # MFU-1 m1g1 (23 Oct): batch_pass gradients disagree with
                # the per-window pass (grad cos 0.52-0.86, logprob |d| up to 5); training refuses it until fixed
                raise RuntimeError("batch_tokens > 0 refused: batch_pass is numerically wrong (m1g1); batch_diagnostic only")
            pending = []  # batched tails (``batch_tokens``): windows of one start with one prefix, run together

            def scale_of(spec, made):
                if cmd.get("norm") == "cap":  # Dr. GRPO: each try's summed log-likelihood over a constant (the cap)
                    return 1.0 / float(spec.get("norm_cap") or cmd["norm_cap"])  # mg5b: a per-level constant when the spec has one
                if per_try:  # each try counts once: its windows' mean weighted log-likelihood (divided by tries below)
                    return 1.0 / (max(1, int(made["arrays"]["positions"].numel())) * per_try[spec.get("name", "").split("#")[0]])
                return 1.0  # summed; the gradient is divided by the tokens below

            def record(spec, made, out, teacher_row=None):
                nonlocal tokens, losses, finite
                now = time.time()  # MFU-1: wall since the previous window's record (or the loop start)
                wall, last_record[0] = now - last_record[0], now
                host = {**(out.get("host") or {}), "next": round(host_next[0], 3), "record_prev": round(host_rec[0], 3)}
                host_next[0] = 0.0
                ent = out.get("entropy") or (0.0, 0)
                entropy_sum[0] += ent[0]
                entropy_sum[1] += ent[1]
                tokens += out["targets"]
                losses += out["loss"]
                if made["name"] in keep_old or made["name"].split("#")[0] in keep_old:  # k3: pi_old = this state's
                    # actor log-probs, from the step's own forward (no pre-step old pass; v7 and mg.k3_measure)
                    self.old[self.old_key(made)] = out["actor"]
                finite &= math.isfinite(out["loss"]) and bool(torch.isfinite(out["raw"]).all())
                result["windows"].append({"name": made["name"], "tokens": out["tokens"], "tail": out["tail"],
                                          "targets": out["targets"], "trained": int(out["valid"].sum()),
                                          "w": float(made["arrays"]["weights"][0]) if "weights" in made["arrays"] else 1.0,
                                          "mismatch": self.mismatch(made, out, **({"gap_dump": True} if cmd.get("gap_dump") else {})),
                                          "sampled_nll_trainer": round(float(-out["raw"].mean()), 4), "tis": out["tis"],
                                          "signed": out["signed"], "opd": out["opd"], "teacher": teacher_row,
                                          "prefix_seconds": out["prefix_seconds"],
                                          "forward_seconds": out["forward_seconds"], "backward_seconds": out["backward_seconds"],
                                          "prefix_reused": out["prefix_reused"], "cut": out["cut"],
                                          "prefix_start": out.get("prefix_start", 0),
                                          "prefix_carried": out.get("prefix_carried", False),
                                          "carry_moved": out.get("carry_moved", 0),
                                          "batch": out.get("batch", 1),
                                          **({"actor_sha": digest(out["actor"]), "raw_sha": digest(out["raw"])}
                                             if cmd.get("digest") else {}),
                                          **ref_window(made["name"], out, made),
                                          "seconds": round(out["prefix_seconds"] + out["forward_seconds"]
                                                           + out["backward_seconds"], 1),
                                          "wall": round(wall, 2), "host": host})
                host_rec[0] = time.time() - now  # this record's own time falls in the next window's wall

            ref_new, geom_new = {}, {}
            keep_old = set(cmd.get("keep_old") or ())
            last_record = [time.time()]
            host_next, host_rec = [0.0], [0.0]
            entropy_sum = [0.0, 0]

            def ref_window(name, out, made=None):
                """MFU numerics bar: the base step's per-token logprobs kept (ref keep) or compared (ref compare;
                "both": compared, then kept). ``ref_rows``: also the window's geometry (row positions, cut, image
                spans) for ref_compare's worst-row read."""
                if cmd.get("ref") not in ("keep", "compare", "both"):
                    return {}
                ref_new[name] = (out["actor"].float(), out["valid"].bool())
                if cmd.get("ref_rows") and made is not None:
                    geom_new[name] = window_geom(made, out, getattr(getattr(getattr(self, "model", None), "config", None),
                                                                  "image_token_id", None))
                return {}

            def flush():
                if not pending:
                    return
                if len(pending) == 1:
                    spec, made = pending[0]
                    record(spec, made, self.window_pass(made, scale=scale_of(spec, made), shared=shared))
                else:
                    outs = self.batch_pass([m for _, m in pending], [scale_of(sp, m) for sp, m in pending], shared=shared)
                    for (spec, made), out in zip(pending, outs):
                        record(spec, made, out)
                pending.clear()

            def joins(spec, made):
                if not pending:
                    return True
                spec0, made0 = pending[0]
                cut = made["inputs"]["input_ids"].shape[1] - made["tail"]
                cut0 = made0["inputs"]["input_ids"].shape[1] - made0["tail"]
                width = max([m["tail"] for _, m in pending] + [made["tail"]])
                return (spec.get("share") == spec0.get("share") and cut == cut0 and "teacher" not in made
                        and len(pending) < int(cmd.get("batch_max") or 8) and width * (len(pending) + 1) <= budget)

            for spec in specs:  # one window in flight unless batched (a window's images are GBs)
                asked = time.time()
                made = next_window()
                host_next[0] += time.time() - asked
                if made is None:
                    result["errors"].append({"name": spec.get("name"), "error": "no window fits", **self.window_why(spec)})
                    continue
                if self.policy["loss"] == "gspo":  # pi_old from the update's start (op_old_logprobs); absent: s = 1
                    old = (getattr(self, "old", None) or {}).get(self.old_key(made))
                    if old is not None and old.numel() == made["arrays"]["positions"].numel():
                        made["arrays"]["old"] = old.to(made["arrays"]["reference"].device)
                if budget:
                    if not joins(spec, made):
                        flush()
                    pending.append((spec, made))
                    del made
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
                if made["name"] in keep_old or made["name"].split("#")[0] in keep_old:
                    made["no_carry"] = True  # k3's pi_old: the same prefix path as op_k3_score's (carry-free)
                out = self.window_pass(made, scale=scale_of(spec, made), shared=shared)
                record(spec, made, out, teacher_row)
                del made, out
            flush()
            close()
            if shared_teacher is not None:
                shared_teacher.clear()
            if shared is not None:
                shared.clear()  # the next step's weights differ: no cache outlives its step
                result["prefix_reused"] = sum(bool(w.get("prefix_reused")) for w in result["windows"])
                if cmd.get("prefix_carry"):
                    result["prefix_carry"] = {**CARRY, "windows": sum(bool(w.get("prefix_carried"))
                                                                      for w in result["windows"])}
            result["window_wait_seconds"] = round(feed["waits"], 1)
            gaps = sorted(((w["wall"] - w["seconds"], w) for w in result["windows"] if "wall" in w), key=lambda g: -g[0])
            if gaps:  # MFU-1: per-window wall outside its prefix / forward / backward (feed waits included)
                result["window_gap"] = {"sum": round(sum(g for g, _ in gaps), 1), "p50": round(gaps[len(gaps) // 2][0], 2),
                                        "max": round(gaps[0][0], 2),
                                        "top": [{"name": w["name"], "gap": round(g, 2), "tokens": w["tokens"],
                                                 "prefix_reused": w["prefix_reused"], "batch": w["batch"]} for g, w in gaps[:5]],
                                        # the gap's named parts (feed wait, pass pre/post, the previous record); rest: other host work
                                        "host": {k: round(sum((w.get("host") or {}).get(k) or 0.0 for _, w in gaps), 1)
                                                 for k in ("next", "pre", "post", "record_prev", "vision", "rope")}}
            if cmd.get("encode_procs") or cmd.get("window_cache"):
                result["encode"] = {"procs": int(cmd.get("encode_procs") or 0), "encoded": feed["encoded"],
                                    "cache_hits": feed["cache_hits"], "errors": self.encode_errors[:5],
                                    "cache": self.window_cache.summary(),
                                    **{k: feed[k] for k in ("held", "ahead_hits", "ahead_sent")}}
            if cmd.get("accumulate"):  # v7: the sums carry to op_step; the gradients stay in .grad
                acc = self.acc
                acc["tokens"] += tokens
                acc["losses"] += losses
                acc["finite"] &= finite
                acc["calls"] += 1
                acc["windows"] += len(result["windows"])
                acc["entropy"][0] += entropy_sum[0]
                acc["entropy"][1] += entropy_sum[1]
                for key, n in per_try.items():
                    acc["tries"][key] = acc["tries"].get(key, 0) + n
                result.update(ok=bool(finite), accumulated=True, trained_tokens=tokens, acc_tokens=acc["tokens"],
                              acc_tries=len(acc["tries"]), acc_calls=acc["calls"],
                              entropy=round(entropy_sum[0] / entropy_sum[1], 5) if entropy_sum[1] else None,
                              train_seconds=round(time.time() - began, 1), **sampler.summary())
                return result
            if cmd.get("from_acc"):  # v7 op_step: the accumulated batch stands in for this call's (empty) windows
                acc = self.acc
                tokens, losses, finite = acc["tokens"], acc["losses"], acc["finite"]
                per_try = dict(acc["tries"]) if cmd.get("norm") in ("try", "cap") else {}
                entropy_sum[0], entropy_sum[1] = acc["entropy"]
                result.update(accumulated_calls=acc["calls"], accumulated_windows=acc["windows"])
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
            if cmd.get("aux"):  # next-board aux loss (envs arm): its own mean, x weight, added to the policy gradient
                with torch.profiler.record_function("phase:aux"):
                    result["aux"] = self.aux_pass(cmd["aux"])
            for group in self.optimizer.param_groups:  # a frozen group (scale 0) stays out of the clip norm
                if float(group.get("scale", 1.0)) == 0.0:
                    for param in group["params"]:
                        param.grad = None
            grads = [p.grad for _, p in self.params if p.grad is not None]
            if cmd.get("ref") in ("compare", "both"):
                if not getattr(self, "ref", None) and cmd.get("ref_path"):
                    self.ref = torch.load(cmd["ref_path"], map_location="cpu", weights_only=False)
                if getattr(self, "ref", None):
                    here = {k: (a.cpu(), v.cpu()) for k, (a, v) in ref_new.items()}
                    result["ref_compare"] = self.ref_compare(here, grads, geom_new or None)
            if cmd.get("ref") in ("keep", "both"):  # MFU numerics bar: this step is the reference (logprobs, raw gradients)
                self.ref = {"actor": {k: (a.cpu(), v.cpu()) for k, (a, v) in ref_new.items()},
                            "grads": [g.detach().float().cpu() for g in grads]}
                if cmd.get("ref_path"):  # mfu7: arms in fresh processes read the reference from disk
                    torch.save(self.ref, cmd["ref_path"])
            if cmd.get("digest"):  # MFU exactness: per gradient its float64 sum and sum of squares (same bits, same digest)
                sums = torch.stack([torch.stack([g.double().sum(), g.double().square().sum()]) for g in grads]).cpu()
                result["grad_digest"] = digest(sums)
                result["snap_route_miss"] = getattr(self, "snap_route_miss", 0)
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in self.params], float(cmd.get("clip", 1.0))))
            finite &= math.isfinite(norm)
            result["entropy"] = round(entropy_sum[0] / entropy_sum[1], 5) if entropy_sum[1] else None
            result.update(loss=round(losses, 6), grad_norm=round(norm, 6),
                          grad_norm_after_clip=round(min(norm, float(cmd.get("clip", 1.0))), 6), lr=lr,
                          trained_tokens=tokens, grads=len(grads))
            if not finite:
                self.optimizer.zero_grad(set_to_none=True)
                result.update(ok=False, stop="nan_or_inf", applied=False)
                return result
            if self.policy["loss"] == "gspo":
                seq, tok, clip_share = approx_kl(result["windows"])
                result["log_s_q"] = log_s_quantiles(result["windows"])
                result.update(approx_kl=None if seq is None else round(seq, 8),
                              approx_kl_tok=None if tok is None else round(tok, 6),
                              clip_frac=None if clip_share is None else round(clip_share, 4))
                # early stop (Spinning Up PPO: break before the step once approx_kl > 1.5 x target_kl): the KL the
                # update's earlier steps reached, measured on this step's batch; the step is not taken
                if cmd.get("kl_stop") is not None and seq is not None and seq > float(cmd["kl_stop"]):
                    self.optimizer.zero_grad(set_to_none=True)
                    result.update(ok=True, applied=False, kl_stopped=True, kl_stop=float(cmd["kl_stop"]),
                                  train_seconds=round(time.time() - began, 1))
                    if prof is not None:
                        prof.__exit__(None, None, None)
                        undo_annotate()
                    return result
            stepped = time.time()
            before = [p.detach().clone() for _, p in self.params]  # on the card: no 21 GiB host copy
            with torch.profiler.record_function("phase:optimizer"):
                self.update_stepped = True  # from here an out-of-memory is not retried (the step may be applied)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if PROFILE == "pth":  # one host sync
                    change = float(torch.stack([(p.detach() - b).float().pow(2).sum() for (_, p), b in zip(self.params, before)]).sum().sqrt())
                else:
                    change = math.sqrt(float(sum((p.detach() - b).float().pow(2).sum() for (_, p), b in zip(self.params, before))))
            del before
            result["step_seconds"] = round(time.time() - stepped, 1)
            result["train_seconds"] = round(time.time() - began, 1)
            if prof is not None:
                torch.cuda.synchronize()
                wall = time.time() - prof_began
                prof.__exit__(None, None, None)
                undo_annotate()
                result["profile"] = profile_report(prof, wall)
                if cmd.get("profile_trace"):
                    prof.export_chrome_trace(str(cmd["profile_trace"]))
                del prof
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
            result["sel_cache"] = {"hits": SEL["hits"], "stores": SEL["stores"], "causal_hits": CAUSAL["hits"],
                                   "verify": SEL["verify"], "picks_bad": SEL["bad"], "causal_bad": CAUSAL["bad"]}
        if NGRAM["calls"]:
            result["ngram_pinned_calls"] = NGRAM["calls"]
        if "rl1_experts" in sys.modules:  # stage X MoE calls by path since load
            xs = sys.modules["rl1_experts"].EXPERTS
            result["expert_calls"] = {"fused": xs["calls"], "fallback": xs["fallback"], "why": dict(xs["why"]),
                                      "dtypes": dict(xs.get("dtypes") or {})}
            if sys.modules["rl1_experts"].COMPACT["on"]:
                result["expert_compact"] = sys.modules["rl1_experts"].summary()
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
            result["probes"] = self.score_windows(cmd["probes"], hold=bool(cmd.get("hold_previous")))
            result["probe_seconds"] = round(time.time() - began, 1)
        result["ok"] = True
        return result

    def aux_rows(self, path):
        cache = self.__dict__.setdefault("aux_cache", {})
        if path not in cache:
            with open(path) as file:
                cache[path] = [json.loads(line) for line in file if line.strip()]
        return cache[path]

    def aux_pass(self, aux, grad=True):
        """Next-board aux loss (envs arm, nextboard_aux): the next ``windows`` packed conversations of ``aux["file"]``
        (a cursor per file, wrapping), each one whole-sequence window; plain cross-entropy on the assistant targets,
        mean over the step's target tokens x ``weight``, backpropagated into the grads already there (the policy
        loss divided by its own count). ``grad`` False: forward only (reads). Returns teacher-forced stats at the
        weights before the step (exact / body match, token accuracy, NLL)."""
        import torch

        import nextboard_aux as nb

        rows = self.aux_rows(aux["file"])
        cursor = self.__dict__.setdefault("aux_cursor", {})
        k = len(rows) if not grad else min(len(rows), int(aux.get("windows") or 2))
        at = cursor.get(aux["file"], 0)
        picked = [rows[(at + i) % len(rows)] for i in range(k)] if rows else []
        if grad:
            cursor[aux["file"]] = (at + k) % max(1, len(rows))
        encoded = [(r, nb.encode(self.tokenizer, r, int(aux.get("max_tokens") or 16384))) for r in picked]
        encoded = [(r, e) for r, e in encoded if e["spans"]]
        total = sum(e - s for _, enc in encoded for s, e, _ in enc["spans"])
        began, steps, loss, by_game = time.time(), [], 0.0, {}
        weight = float(aux.get("weight") or 0.0)
        for number, (row, enc) in enumerate(encoded):
            made = nb.made(enc, getattr(self, "aux_device", "cuda:0"), name=f"nb{number}")
            out = self.window_pass(made, scale=(weight / total) if grad else None)
            if grad:
                loss += float(out["loss"])
            got = nb.step_stats(enc, out["raw"], out["hit"])
            steps += got
            by_game.setdefault(str(row.get("game")), []).extend(got)
            del made, out
        result = {"windows": len(encoded), "targets": total, "weight": weight, "seconds": round(time.time() - began, 1),
                  **nb.pooled(steps)}
        if grad:
            result["loss"] = round(loss, 6)
        else:
            result["by_game"] = {g: nb.pooled(rs) for g, rs in sorted(by_game.items())}
        return result

    def op_nb_score(self, cmd):
        """Forward-only next-board read of a packed file (every row): pooled and per-game stats."""
        return {"ok": True, **self.aux_pass({"file": cmd["file"], "max_tokens": cmd.get("max_tokens")}, grad=False)}

    def op_batch_check(self, cmd):
        """Batched tails against the per-window path on the same windows (no optimizer step): per mode the
        per-token log-probabilities, the loss and the LoRA gradient against ``fresh`` (each window's prefix computed
        on its own), and the card time. Modes: fresh, fresh again (the noise floor: fp32 atomics), shared prefix as
        before the 17 Oct position fix, shared prefix, batched."""
        import torch

        specs = [s for s in self.expand_cmd(cmd) if s.get("first_window", True)]
        mades = [m for m in (self.attempt_window(s, 0, False) for s in specs) if m is not None]
        scale = 1.0 / float(cmd.get("norm_cap", 24000))
        budget, most = int(cmd.get("batch_tokens", 32768)), int(cmd.get("batch_max", 8))
        report = {"windows": [{"name": m["name"], "cut": m["inputs"]["input_ids"].shape[1] - m["tail"], "tail": m["tail"],
                               "targets": int(m["arrays"]["positions"].numel())} for m in mades], "modes": []}
        reference, actors = None, None
        for mode in cmd.get("modes", ["fresh", "fresh", "legacy", "shared", "batched"]):
            self.optimizer.zero_grad(set_to_none=True)
            fq.reset_peaks()
            torch.cuda.synchronize()
            began = time.time()
            outs, shared = [], ({} if mode != "fresh" else None)
            BATCH["legacy_positions"] = mode == "legacy"
            failed = None
            try:
                if mode == "batched":
                    groups, group = [], []
                    for made in mades:
                        cut = made["inputs"]["input_ids"].shape[1] - made["tail"]
                        width = max([m["tail"] for m in group] + [made["tail"]])
                        if group and (cut != group[0]["inputs"]["input_ids"].shape[1] - group[0]["tail"]
                                      or len(group) >= most or width * (len(group) + 1) > budget):
                            groups.append(group)
                            group = []
                        group.append(made)
                    groups += [group] if group else []
                    for group in groups:
                        outs += (self.batch_pass(group, [scale] * len(group), shared=shared) if len(group) > 1
                                 else [self.window_pass(group[0], scale=scale, shared=shared)])
                else:
                    for made in mades:
                        outs.append(self.window_pass(made, scale=scale, shared=shared))
            except Exception as error:  # one mode's failure leaves the others' numbers
                failed = f"{type(error).__name__}: {error}"[:400]
                report.setdefault("traceback", {})[mode] = traceback.format_exc()[-2000:]
            finally:
                BATCH["legacy_positions"] = False
                if shared:
                    shared.clear()
            if failed:
                report["modes"].append({"mode": mode, "error": failed})
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            torch.cuda.synchronize()
            row = {"mode": mode, "seconds": round(time.time() - began, 2),
                   "loss": round(sum(o["loss"] for o in outs), 6),
                   "batches": max(o.get("batch", 1) for o in outs),
                   "card_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            grads = [p.grad for _, p in self.params]
            if reference is None:
                reference = [None if g is None else g.detach().clone() for g in grads]
                actors = [o["actor"] for o in outs]
            else:
                dot = ref2 = new2 = diff2 = 0.0
                for g, r in zip(grads, reference):
                    if g is None or r is None:
                        continue
                    g, r = g.detach().float(), r.float()
                    dot += float((g * r).sum())
                    ref2 += float((r * r).sum())
                    new2 += float((g * g).sum())
                    diff2 += float(((g - r) ** 2).sum())
                row["grad_cos"] = round(dot / math.sqrt(max(ref2 * new2, 1e-30)), 7)
                row["grad_rel_diff"] = round(math.sqrt(diff2 / max(ref2, 1e-30)), 6)
                diffs = []
                for o, a in zip(outs, actors):
                    keep = o["valid"] & torch.isfinite(a) & torch.isfinite(o["actor"])
                    d = (o["actor"] - a)[keep].abs()
                    diffs.append({"max": round(float(d.max()), 5) if d.numel() else None,
                                  "mean": round(float(d.mean()), 6) if d.numel() else None})
                row["logprob_diff"] = diffs
                row["logprob_max"] = max((d["max"] or 0.0) for d in diffs)
                row["logprob_mean"] = round(sum((d["mean"] or 0.0) for d in diffs) / max(1, len(diffs)), 6)
            row["grad_norm"] = round(math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None)), 6)
            report["modes"].append(row)
            del outs
            torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        del reference
        torch.cuda.empty_cache()
        report["batch_calls"] = BATCH["calls"]
        return report

    def save_adapter(self, folder, only=None):
        import torch
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        began = time.time()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in get_peft_model_state_dict(self.model).items()
                 if only is None or (only == "dense") != (".experts." in k)}
        expected = sum(1 for name, _ in self.params if only is None or (only == "dense") != (".experts." in name))
        if len(state) != expected:  # a partial adapter never reaches the server
            raise ValueError(f"adapter state has {len(state)} tensors, the trainer {expected} LoRA params")
        save_file(state, str(folder / "adapter_model.safetensors"))
        (folder / "adapter-tensors.json").write_text(json.dumps({"tensors": len(state), "expected": expected}))
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

    def base_tensors(self):
        """The frozen model's CUDA tensors (module parameters, buffers, plain tensor attributes) that alone own a whole
        storage; the adapter (held()), views and shared storages stay where they are."""
        import torch

        skip, seen = {id(p) for _, p in self.params}, {}

        def add(value, depth=0):
            if not torch.is_tensor(value) or id(value) in skip or depth > 4:
                return
            if hasattr(value, "__tensor_flatten__"):  # a wrapper subclass (e.g. a quantised weight): its inner tensors
                for name in value.__tensor_flatten__()[0]:
                    add(getattr(value, name, None), depth + 1)
                return
            if value.is_cuda:
                seen.setdefault(id(value), value)

        for module in self.model.modules():
            for value in [*module._parameters.values(), *module._buffers.values(), *vars(module).values()]:
                add(value)
        storages, census = {}, {"found": len(seen), "no_pointer": [], "kept": 0, "kept_gib": 0.0}
        for value in seen.values():
            try:  # tscheck-1: a storage without a data pointer raised here; such tensors stay on the card
                storages.setdefault(value.untyped_storage().data_ptr(), []).append(value)
            except RuntimeError:
                if len(census["no_pointer"]) < 8:
                    census["no_pointer"].append([type(value).__name__, str(value.dtype), list(value.shape)])
                continue
        out = []
        for v in storages.values():
            ok = (len(v) == 1 and type(v[0]) in (torch.Tensor, torch.nn.Parameter) and v[0].is_contiguous()
                  and v[0].storage_offset() == 0 and v[0].numel() * v[0].element_size() == v[0].untyped_storage().nbytes())
            if ok:
                out.append(v[0])
            else:
                census["kept"] += len(v)
                census["kept_gib"] += sum(t.numel() * t.element_size() for t in v) / 2**30
        census["kept_gib"] = round(census["kept_gib"], 2)
        self.base_census = census
        return out

    def park_base(self):
        """Time share (19 Oct): the frozen base to pinned host memory too, so a second server can use the card."""
        import torch

        moved, kept = 0, 0
        self.base_parked = []
        for value in self.base_tensors():
            key = ("base", id(value))
            buffer = self.pinned.get(key)
            if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                buffer = self.pinned[key] = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
            buffer.copy_(value)
            try:
                value.data = buffer
            except Exception:  # a tensor subclass that refuses the swap stays on the card
                kept += 1
                continue
            self.base_parked.append(value)
            moved += value.numel() * value.element_size()
        return moved, kept

    def op_prepin(self, cmd=None):
        """Start-up cut (d), MFU-2 23 Oct: the park's pinned host buffers (adapter, optimizer state so far and, with
        ``base``, the frozen base) allocated on a thread while the card works, so the first park only copies (ovf2: the
        first park took 57.8 s for 84 GiB, mostly pinned allocation). The park joins the thread first."""
        import threading

        import torch

        if getattr(self, "_prepin", None) is not None:
            return {"already": True}
        todo = []
        for key, holder, field in self.held():
            value = holder.data if field is None else holder[field]
            if value.is_cuda and key not in self.pinned:
                todo.append((key, tuple(value.shape), value.dtype))
        if (cmd or {}).get("base", True):
            for value in self.base_tensors():
                if ("base", id(value)) not in self.pinned:
                    todo.append((("base", id(value)), tuple(value.shape), value.dtype))
        size = lambda shape, dtype: torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()  # noqa: E731
        total = sum(size(s, d) for _, s, d in todo)
        state = {"tensors": len(todo), "gib": round(total / 2**30, 2), "bytes": total, "done_bytes": 0, "seconds": None,
                 "error": None}

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
        return {"started": True, **state}

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

    def op_base_export(self, cmd):
        """Start-up cut (c): the probes' base references (``score`` label base) to ``path`` with the loop's ``key``."""
        import torch

        refs = {name: {k: v.detach().cpu() for k, v in r["base"].items()}
                for name, r in self.references.items() if "base" in r and name in set(cmd["names"])}
        if set(refs) != set(cmd["names"]):
            raise ValueError(f"base references missing for {sorted(set(cmd['names']) - set(refs))}")
        path = Path(cmd["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"key": cmd["key"], "refs": refs}, str(tmp))
        os.replace(tmp, path)
        return {"windows": len(refs), "path": str(path), "bytes": path.stat().st_size}

    def op_base_drop(self, cmd):
        """The named probes' references removed (the loop's fresh base score after a refused or failed cache import:
        no imported copy may stay installed for it to compare against)."""
        names = list(cmd["names"])
        dropped = [n for n in names if self.references.pop(n, None) is not None]
        return {"dropped": len(dropped)}

    def op_base_import(self, cmd):
        """The cached base references as base, previous and last of each probe (what ``score`` label base sets)."""
        import torch

        blob = torch.load(str(cmd["path"]), map_location="cpu", weights_only=True)
        if blob.get("key") != cmd["key"]:
            raise ValueError(f"base cache key {blob.get('key')} != {cmd['key']}")
        if set(blob["refs"]) != set(cmd["names"]):
            raise ValueError(f"base cache windows {sorted(blob['refs'])} != {sorted(cmd['names'])}")
        for name, ref in blob["refs"].items():
            # on the host, where score_windows keeps its references (v6j2 01:28: cuda:0 copies broke the check score
            # and the fresh base score after it with a cuda:0/cpu device mismatch)
            mine = dict(ref)
            refs = self.references.setdefault(name, {})
            refs["base"] = refs["previous"] = refs["last"] = mine
        return {"windows": len(blob["refs"]), "names": sorted(blob["refs"])}

    def compact_experts(self):
        """The masked experts' NVFP4 base cut from every masked layer (rl1_experts.compact_experts); routes, LoRA and
        R3 stay on the 512 global ids. Needs the fused expert path on every op (trainer_fast fused_experts)."""
        import torch

        import rl1_experts

        if not EXPERT_MASK.get("path"):
            raise RuntimeError("expert compaction needs RL1_EXPERT_MASK")
        spec = json.loads(Path(EXPERT_MASK["path"]).read_text())
        pruned = {int(k): [int(e) for e in v] for k, v in spec["pruned"].items()}
        rl1_experts.install()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out = rl1_experts.compact_experts(self.model, pruned)
        torch.cuda.synchronize()
        out["allocated_freed_gib"] = round((before - torch.cuda.memory_allocated()) / 2 ** 30, 2)
        out["allocated_gib"] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
        return out

    def op_compact_experts(self, cmd=None):
        """Harness op (MFU-1 read): compaction mid-process (irreversible); the load-time switch is RL1_EXPERT_COMPACT."""
        if self.parked:
            raise RuntimeError("compact_experts: unpark first")
        torch_peak_reset = __import__("torch").cuda.reset_peak_memory_stats
        out = self.compact_experts()
        torch_peak_reset()
        return {"ok": True, **out}

    def op_park(self, cmd=None):
        import torch

        if self.parked:
            return {"parked": True, "already": True}
        waited = time.time()
        prepin = self.prepin_join()
        if prepin is not None:
            prepin["wait_s"] = round(time.time() - waited, 1)
        torch.cuda.synchronize()
        base, kept = self.park_base() if (cmd or {}).get("base") else (0, 0)
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
        return {"parked": True, "gib": round(moved / 2**30, 2), "seconds": round(time.time() - began, 2), "prepin": prepin,
                "grads_kept": {"n": len(kept_grads), "sq": grad_sq} if keep else None,
                "tensors": len(swaps), "card_used_gib": used, "base_gib": round(base / 2**30, 2), "base_kept": kept,
                "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}

    def unpark(self):
        import torch

        began, moved = time.time(), 0
        for value in getattr(self, "base_parked", []):
            if not value.is_cuda:
                value.data = value.data.to("cuda:0", non_blocking=True)
                moved += value.numel() * value.element_size()
        self.base_parked = []
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
        return {"gib": round(moved / 2**30, 2), "seconds": round(time.time() - began, 2), **out}

    def op_park_check(self, cmd):
        """Time share: the probes' actor log-probabilities before and after ``cycles`` full parks (base included) must
        be identical; reports the parked card use the second server would see."""
        import torch

        def actor():
            out = []
            for spec in cmd["windows"]:
                made = self.attempt_window(spec)
                if made is not None:
                    r = self.window_pass(made)
                    out.append(r["actor"][r["valid"]].float().cpu())
                    del made, r
            return out

        before, parks = actor(), []
        for _ in range(int(cmd.get("cycles", 2))):
            parks.append(self.op_park({"base": True}))
            parks[-1]["unpark"] = self.unpark()
            torch.cuda.synchronize()
        after = actor()
        diff = max([float((a - b).abs().max()) for a, b in zip(before, after)] or [float("nan")])
        return {"identical": len(before) == len(after) and diff == 0.0, "max_abs_diff": diff, "parks": parks,
                "tokens": sum(int(a.numel()) for a in before), "census": getattr(self, "base_census", None)}

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

    def op_split_gemm(self, cmd):
        import split_gemm

        return {"on": split_gemm.install(bool(cmd.get("on", True)))}

    def op_experts(self, cmd):
        """Routed-expert path: "native" (native_experts.py) or "dequant" (Axolotl's sonicmoe path)."""
        import native_experts

        on = native_experts.install(cmd.get("path", "native") == "native")
        return {"path": "native" if on else "dequant", "stats": dict(native_experts.STATS)}

    def op_knobs(self, cmd):
        import split_gemm

        return split_gemm.knobs(cmd.get("switch_interval"), cmd.get("prefetch"))

    def op_compare_split(self, cmd):
        """arc2's split expert GEMM against the original per-expert slice loop on one recorded window, with the
        original run twice for its own noise: seconds, loss, gradient cosine / relative difference, log-prob
        difference (orig vs split, orig vs orig2)."""
        import split_gemm
        import torch

        made = self.attempt_window(cmd["spec"])
        out, grads = {}, {}
        try:
            for path in ("orig", "split", "orig2"):
                split_gemm.install(path == "split")
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                began = time.time()
                result = self.window_pass(made, scale=1.0 / made["arrays"]["positions"].numel())
                torch.cuda.synchronize()
                out[path] = {"seconds": round(time.time() - began, 1),
                             **{k: result[k] for k in ("prefix_seconds", "forward_seconds", "backward_seconds", "loss")}}
                grads[path] = [q.grad.detach().float().cpu() if q.grad is not None else None for _, q in self.params]
                out[path]["actor"] = result["actor"]
        finally:
            split_gemm.install(cmd.get("restore", True))
            self.optimizer.zero_grad(set_to_none=True)
        del made

        def versus(a, b):
            pairs = [(x, y) for x, y in zip(grads[a], grads[b]) if x is not None and y is not None]
            na = math.sqrt(sum(float(x.pow(2).sum()) for x, _ in pairs))
            nb = math.sqrt(sum(float(y.pow(2).sum()) for _, y in pairs))
            dot = sum(float((x * y).sum()) for x, y in pairs)
            diff = math.sqrt(sum(float((x - y).pow(2).sum()) for x, y in pairs))
            actor = (out[a]["actor"] - out[b]["actor"]).abs()
            return {"grad_cosine": round(dot / (na * nb), 8) if na and nb else None,
                    "grad_rel_diff": round(diff / na, 8) if na else None, "grad_norm": [round(na, 6), round(nb, 6)],
                    "loss_diff": out[a]["loss"] - out[b]["loss"], "actor_abs_diff_mean": round(float(actor.mean()), 8),
                    "actor_abs_diff_max": round(float(actor.max()), 8), "params_with_grad": len(pairs)}

        result = {"split_vs_orig": versus("orig", "split"), "noise": versus("orig", "orig2"),
                  "speedup": round(out["orig"]["seconds"] / out["split"]["seconds"], 3)}
        for path in out.values():
            path.pop("actor")
        del grads
        torch.cuda.empty_cache()
        return {**out, **result}

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
        if state["names"] != names or set(master) != set(names):
            raise ValueError(f"checkpoint layout differs: {len(master)} tensors, {len(names)} params")
        with torch.no_grad():
            for name, param in self.params:
                param.copy_(master[name].to(device=param.device, dtype=param.dtype))
        groups = [float(g.get("scale", 1.0)) for g in self.optimizer.param_groups]
        saved = [float(g.get("scale", 1.0)) for g in state["optimizer"]["param_groups"]]
        if groups != saved:
            raise ValueError(f"checkpoint optimizer groups {saved} differ from this trainer's {groups}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
        self.forget_unaccepted()
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
        PROGRESS.update(phase="idle", op=None)
        while True:
            path = self.dir / "cmd" / f"{index:04d}.json"
            if not path.exists():
                alive.write_text(json.dumps({"state": "idle", "next": index, "time": time.time(), "loaded": self.loaded},
                                            default=str))
                time.sleep(0.1 if PROFILE == "pth" else 1.0)
                continue
            cmd = json.loads(path.read_text())
            alive.write_text(json.dumps({"state": "busy", "op": cmd["op"], "n": index, "time": time.time()}))
            PROGRESS.update(phase=PHASES.get(cmd["op"], cmd["op"]), op=cmd["op"])
            began = time.time()
            try:
                if cmd["op"] == "exit":
                    result = {"ok": True, **({"checkpoint": done} if (done := self.ckpt_join()) else {})}
                else:
                    unparked = self.unpark() if self.parked and cmd["op"] in GPU_OPS else None
                    self.snapshots = bool(cmd.get("prefix_snapshots"))  # fix a: per op, off unless asked
                    self.probe_pinned = bool(cmd.get("probe_pinned"))  # kl_base_pinned diagnostic (trainer_fast knob)
                    op_paths(self.model, cmd)
                    self.snap_keep = int(cmd.get("snap_keep") or 3)
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
            PROGRESS.update(phase="idle", op=None)
            # the play server wakes beside the idle trainer: hand every cached block back (rl1-ttest-v1: the card
            # held ~152 GiB for the trainer between updates against ~105 GiB of state)
            import torch

            torch.cuda.empty_cache()
            if rl1_memory.TAIL["offloaded"] and hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()  # tail_ckpt offload's pinned blocks back to the host (park, time share)
            used, _, _ = gpu_used_gib()
            result["memory"] = {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2), "card_used_gib": used,
                                **({"device_peak_gib": fq.device_gib(peak=True)} if PROFILE == "pth" else {})}
            self.rec.event("op", op=cmd["op"], n=index, **{k: v for k, v in result.items() if k != "traceback"})
            tmp = self.dir / "done" / f"{index:04d}.json.tmp"
            tmp.write_text(json.dumps(result, default=str))
            tmp.rename(self.dir / "done" / f"{index:04d}.json")
            index += 1
            if cmd["op"] == "exit":
                return



def _oom_errors():
    try:
        import torch

        return (torch.cuda.OutOfMemoryError,)
    except Exception:  # noqa: BLE001 - no torch: nothing to catch
        return ()


OOM_ERRORS = _oom_errors()  # tests add a stand-in class
GPU_OPS = ("fit", "k3_score", "accumulate", "step", "score_ids", "score", "old_logprobs", "batch_check", "update", "profile", "reset", "compare_lora", "compare_split", "split_gemm", "knobs", "split_windows", "checkpoint", "load_checkpoint", "park_check", "adapter_scale", "chunk_check", "route", "expert_ref", "served_check", "r3_check", "topk", "stream_topk", "stream_check", "text_drift", "channel_energy", "load_adapter", "carry_check", "causal_check")  # these bring a parked adapter back first (RL + PTH ops)


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


def layout_tensors(model):
    """(name, owner, slot, tensor) of every parameter, buffer and plain tensor attribute of the model's modules: what
    a cold load places on a device (a quantised weight's scales can be plain attributes)."""
    import torch

    for mod_name, module in model.named_modules():
        prefix = mod_name + "." if mod_name else ""
        for slot, table in (("param", module._parameters), ("buffer", module._buffers)):
            for name, value in table.items():
                if value is not None:
                    yield prefix + name, module, (slot, name), value
        for name, value in vars(module).items():
            if torch.is_tensor(value) and not name.startswith("_"):
                yield prefix + name, module, ("attr", name), value


def tensor_sample(value, n=1024):
    """A device-independent checksum: a strided sample of the values, summed in float64 on the host (inner tensors of
    a wrapper subclass each)."""
    import torch

    if hasattr(value, "__tensor_flatten__"):
        names, _ = value.__tensor_flatten__()
        return [tensor_sample(getattr(value, k), n) for k in names]
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    part = flat[::max(1, flat.numel() // n)][:n].to("cpu")
    if part.dtype == torch.bool:
        part = part.to(torch.uint8)
    return float(part.to(torch.float64).sum())


def layout_digest(model):
    """Start-up cut (b): the model's placement and a sample of its values, per tensor, plus the module classes and
    forward implementations (a kernel swap at load shows here)."""
    import hashlib

    tensors = [[name, slot[0], list(v.shape), str(v.dtype), v.device.type, type(v).__name__, tensor_sample(v)]
               for name, _, slot, v in layout_tensors(model)]
    modules = [[n, type(m).__name__, getattr(getattr(m, "forward", None), "__qualname__", None)]
               for n, m in model.named_modules()]
    return {"tensors": tensors, "modules": hashlib.sha256(json.dumps(modules).encode()).hexdigest()}


def digest_diff(got, want, limit=8):
    """The first differences between two layout digests ([] when equal)."""
    out = []
    if got["modules"] != want["modules"]:
        out.append("module classes / forwards differ")
    a, b = {t[0]: t for t in got["tensors"]}, {t[0]: t for t in want["tensors"]}
    for name in sorted(set(a) | set(b)):
        if a.get(name) != b.get(name):
            out.append(f"{name}: {a.get(name)} != {b.get(name)}"[:300])
            if len(out) >= limit:
                break
    return out


def move_to_layout(model, want):
    """Every tensor to the device the cold load put it on (want: a layout digest)."""
    device = {t[0]: t[4] for t in want["tensors"]}
    moved = 0
    for name, module, (slot, key), value in list(layout_tensors(model)):
        target = device.get(name)
        if target is None or value.device.type == target:
            continue
        dest = "cuda:0" if target == "cuda" else target
        if slot == "param":
            value.data = value.data.to(dest)
        elif slot == "buffer":
            module._buffers[key] = value.to(dest)
        else:
            setattr(module, key, value.to(dest))
        moved += value.numel() * value.element_size()
    return moved


def load_model(rec, model_dir, out):
    """The trainer's model. Start-up cut (b), MFU-2 23 Oct (RL1_HOST_LOAD=1, mg.host_load): the load runs on the host
    (device_map cpu, RL1_HOST_LOAD_THREADS threads, lowered priority) while the play server still holds the card;
    its first CUDA use waits for the gate (RL1_GPU_GATE). Then every tensor moves to the cold load's device and the
    layout digest must equal the cold load's record (RL1_LAYOUT_RECORD); a missing record, a mismatch or any error
    falls back to today's cold load (event host_load with fallback). A cold load with RL1_LAYOUT_RECORD writes the
    record when there is none."""
    import gc

    import torch

    record = os.environ.get("RL1_LAYOUT_RECORD")
    if os.environ.get("RL1_HOST_LOAD") == "1":
        began, info, model = time.time(), {"on": True}, None
        threads = torch.get_num_threads()
        try:
            want = json.loads(Path(record).read_text()) if record and Path(record).is_file() else None
            if want is None:
                raise RuntimeError(f"no cold-load layout record at {record}")
            torch.set_num_threads(int(os.environ.get("RL1_HOST_LOAD_THREADS", "4")))
            try:
                os.setpriority(os.PRIO_PROCESS, 0, 10)
            except OSError:
                pass
            fq.DEVICE_MAP = {"": "cpu"}
            model, tok, loaded = fq.load(rec, model_dir, out, sequence_len=131072)  # waits at the gate (synchronize)
            info["host_s"] = round(time.time() - began, 1)
            torch.set_num_threads(threads)
            try:
                os.setpriority(os.PRIO_PROCESS, 0, 0)
            except OSError:
                pass
            t = time.time()
            info["moved_gib"] = round(move_to_layout(model, want) / 2 ** 30, 2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            info["move_s"] = round(time.time() - t, 1)
            diff = digest_diff(layout_digest(model), want)
            if diff:
                raise RuntimeError(f"layout differs from the cold load: {diff}")
            info["total_s"] = round(time.time() - began, 1)
            loaded["host_load"] = info
            rec.event("host_load", **info)
            return model, tok, loaded
        except Exception as exc:  # noqa: BLE001 - today's cold load instead
            info.update(fallback=True, error=f"{type(exc).__name__}: {exc}"[:1500], seconds=round(time.time() - began, 1))
            rec.event("host_load", **info)
            del model
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            torch.set_num_threads(threads)
            try:
                os.setpriority(os.PRIO_PROCESS, 0, 0)
            except OSError:
                pass
            fq.DEVICE_MAP = {"": 0}
            model, tok, loaded = fq.load(rec, model_dir, out, sequence_len=131072)
            loaded["host_load"] = info
            return model, tok, loaded
    model, tok, loaded = fq.load(rec, model_dir, out, sequence_len=131072)
    if record and not Path(record).is_file():
        try:
            Path(record).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(record + ".tmp")
            tmp.write_text(json.dumps(layout_digest(model)))
            os.replace(tmp, record)
            loaded["layout_record"] = record
        except Exception as exc:  # noqa: BLE001 - the host load stays off without a record
            loaded["layout_record_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return model, tok, loaded


def gpu_gate():
    """RL1_GPU_GATE (the loop's mg.trainer_overlap): a second server boots on this card meanwhile, so the load's CPU part
    runs now and the first CUDA initialisation (``torch.cuda._lazy_init``, which every tensor on the card goes through)
    waits until the loop writes the gate file. Returns the hook's state, or None without a gate."""
    path = os.environ.get("RL1_GPU_GATE")
    if not path:
        return None
    import torch

    real, state = torch.cuda._lazy_init, {"began": time.time(), "waited_s": None, "prefetch": True}

    def gated():
        if not torch.cuda.is_initialized() and not os.path.exists(path):
            if state.pop("prefetch", False):  # CPU-only imports the load needs next
                for name in ("transformers", "peft", "axolotl.cli.config", "axolotl.train"):
                    try:
                        __import__(name)
                    except Exception:  # noqa: BLE001 - imported again (and reported) by the load itself
                        pass
            while not os.path.exists(path):
                time.sleep(1.0)
        if state["waited_s"] is None:
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
    parser.add_argument("--optimizer", choices=("adamw8bit", "adamw32"), default="adamw8bit")
    parser.add_argument("--tis-cap", type=float, default=0.0, help="truncated importance sampling cap (0: off)")
    parser.add_argument("--prefix-chunk", type=int, default=16384, help="the one prefix chunking of every pass")
    parser.add_argument("--loop-lora", action="store_true", help="keep Axolotl's per-expert LoRA loop (comparison)")
    parser.add_argument("--spread", type=int, default=1, help="split the decoder layers over this many visible cards")
    args = parser.parse_args()
    gpu_gate()
    progress_writer()
    trainer = Trainer(args)
    trainer.serve()


if __name__ == "__main__":
    main()
