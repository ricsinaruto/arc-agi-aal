"""MFU stage M (RL arm, 21 Oct): memory per tail token of the trainer's window pass.

- ``chunked_head``: the LM head and every per-token read of its logits (actor and raw log-probabilities, entropy,
  greedy hits, the top-20 vocabulary bins) in position chunks; the [N, V] logits never exist whole. The actor's
  backward recomputes each chunk's logits (Liger fused-linear-cross-entropy style), so nothing of size V is saved.
  Each chunk runs the unchunked path's own functions (``rl_job.actor_logprobs`` etc.) on its rows.

- ``tail_scope`` (restored 26 Oct for 130K on CP2, T1c-v3): saved-tensor hooks that move the decoder layers'
  checkpoint inputs (the hc_count x hidden residual stream, 48 x 20 KiB per token on Flash) to pinned host memory in
  the forward and back in the backward.

Switches: ``HEAD["chunked"]`` (trainer cmd ``chunked_head``), ``TAIL["mode"]`` (cmd ``tail_ckpt``); off by default.
"""
from __future__ import annotations

import contextlib
import os

HEAD = {"chunked": False, "chunk": 1024, "capture": False, "calls": 0}
# cpu: offload CPU tensors too (tests: the copy is a separate host tensor)
TAIL = {"mode": "off", "min_tokens": 0, "min_bytes": 1 << 20, "width": None, "offloaded": 0, "bytes": 0, "kept": 0, "cpu": False}
TAIL_MODES = ("off", "offload")


def head_capture(model):
    """While ``HEAD["capture"]`` the LM head returns its input (the hidden states at the kept rows, after any
    pre-hook such as the batch gather); its own forward stays available as ``head._rl1_project``."""
    head = model.get_output_embeddings()
    if getattr(head, "_rl1_capture", False):
        return head
    project = head.forward

    def forward(x, *args, **kwargs):
        return x if HEAD["capture"] else project(x, *args, **kwargs)

    head.forward = forward
    head._rl1_capture = True
    head._rl1_project = project
    return head


def _actor(rows, tokens, support, temperature, split=None):
    import rl_job

    return rl_job.actor_logprobs(rows, tokens, support, temperature, split=split)


def host_splits(support_host, n, chunk, device):
    """Per chunk of rows: (whole, part) row-index tensors on ``device``, from the host copy of ``support`` (rows
    with no sampling set vs with one); None without a host copy (the masks are then tested on the card)."""
    import torch

    if support_host is None:
        return None
    whole = support_host[:, 0] < 0
    out = []
    for i in range(0, n, chunk):
        w = whole[i:min(n, i + chunk)]
        idx = [torch.nonzero(m).flatten() for m in (w, ~w)]
        if torch.cuda.is_available() and str(device) != "cpu":
            idx = [t.pin_memory().to(device, non_blocking=True) for t in idx]
        out.append(tuple(idx))
    return out


def _chunked_function():
    import torch

    class ChunkedActor(torch.autograd.Function):
        """actor log-probabilities [N] from hidden [N, H]; forward and backward one chunk of rows at a time. Extras
        (no grad) are computed in the forward's loop into ``extras``."""

        @staticmethod
        def forward(ctx, hidden, tokens, support, temperature, project, chunk, extras, splits=None):
            device = hidden.device.type
            ctx.cast = (device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
            ctx.project, ctx.chunk, ctx.temperature, ctx.splits = project, chunk, temperature, splits
            n = hidden.shape[0]
            actor = torch.empty(n, dtype=torch.float32, device=hidden.device)
            valid = torch.empty(n, dtype=torch.bool, device=hidden.device)
            for k, i in enumerate(range(0, n, chunk)):
                j = min(n, i + chunk)
                logits = project(hidden[i:j])
                actor[i:j], valid[i:j] = _actor(logits, tokens[i:j], support[i:j], temperature,
                                                splits[k] if splits else None)
                if extras is not None:
                    extras["each"](logits, i, j, valid[i:j])
                del logits
            ctx.save_for_backward(hidden, tokens, support)
            ctx.mark_non_differentiable(valid)
            return actor, valid

        @staticmethod
        def backward(ctx, grad_actor, grad_valid):
            hidden, tokens, support = ctx.saved_tensors
            device, enabled, dtype = ctx.cast
            n, chunk = hidden.shape[0], ctx.chunk
            grad_hidden = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
            for k, i in enumerate(range(0, n, chunk)):
                j = min(n, i + chunk)
                g = grad_actor[i:j]
                if ctx.splits is None and not bool(g.ne(0).any()):  # host splits: no per-chunk sync (U2)
                    continue
                with torch.enable_grad(), torch.autocast(device, dtype=dtype, enabled=enabled):
                    h = hidden[i:j].detach().requires_grad_(grad_hidden is not None)
                    actor, _ = _actor(ctx.project(h), tokens[i:j], support[i:j], ctx.temperature,
                                      ctx.splits[k] if ctx.splits else None)
                torch.autograd.backward(actor, g)  # h.grad, plus any trainable head weight, accumulate here
                if grad_hidden is not None:
                    grad_hidden[i:j] = h.grad
                del actor, h
            return grad_hidden, None, None, None, None, None, None, None

    return ChunkedActor


_FUNCTION = []


def chunked_head(project, hidden, tokens, support, temperature, *, hit=False, vocab=None, chunk=None,
                 support_host=None):
    """``window_pass``'s reads of the logits without the logits: ``actor`` (grad to ``hidden``), ``valid``, ``raw``,
    ``entropy`` (sum, count over valid rows, as ``token_entropy``), ``hit`` (greedy == token, when asked) and
    ``dist`` (``vocab_top``'s dict, when ``vocab`` is a list of id tables). ``project``: the LM head's forward
    (hidden rows -> logits rows)."""
    import torch

    import rl1_trainer
    import rl_job

    if not _FUNCTION:
        _FUNCTION.append(_chunked_function())
    chunk = int(chunk or HEAD["chunk"])
    HEAD["calls"] += 1
    raws, hits, tops = [], [], []
    entropy = []  # device (sum, count) per chunk, read once at the end (U2: was one host read per chunk)

    def each(logits, i, j, valid):
        rows = logits.detach()
        raws.append(rl_job.raw_logprobs(rows, tokens[i:j]))
        entropy.append(rl1_trainer.token_entropy(rows, valid, temperature, lazy=True))
        if hit:
            hits.append(rows.argmax(-1) == tokens[i:j])
        if vocab is not None:
            tops.append(rl1_trainer.vocab_top(rows, at_ids=[table[i:j] for table in vocab]))

    splits = host_splits(support_host, hidden.shape[0], chunk, hidden.device)
    actor, valid = _FUNCTION[0].apply(hidden, tokens, support, temperature, project, chunk, {"each": each}, splits)
    if entropy:
        total, count = torch.stack([torch.stack(e) for e in entropy]).sum(0).tolist()
        entropy = (total, int(count))
    else:
        entropy = (0.0, 0)
    dist = None
    if vocab is not None:
        cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0)
        dist = {"ids": cat([t["ids"] for t in tops]), "logp": cat([t["logp"] for t in tops]),
                "at": [cat([t["at"][k] for t in tops]) for k in range(len(vocab))]}
    raw = torch.cat(raws) if raws else hidden.new_zeros(0).float()
    return {"actor": actor, "valid": valid, "raw": raw, "entropy": entropy,
            "hit": torch.cat(hits) if hit and hits else (hidden.new_zeros(0, dtype=torch.bool) if hit else None),
            "dist": dist}


# ---- tail_ckpt: checkpoint-boundary inputs off the card


def _offload_hooks(width):
    import torch

    stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def pack(t):
        if (TAIL["mode"] != "offload" or not isinstance(t, torch.Tensor) or (t.device.type == "cpu" and not TAIL["cpu"])
                or t.dim() < 2 or t.shape[-1] != width or t.numel() * t.element_size() < TAIL["min_bytes"]):
            TAIL["kept"] += 1
            return t
        # torch's caching host allocator (pinned blocks are reused across windows)
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=torch.cuda.is_available())
        if stream is not None and t.is_cuda:
            stream.wait_stream(torch.cuda.current_stream(t.device))
            with torch.cuda.stream(stream):
                host.copy_(t, non_blocking=True)
            t.record_stream(stream)  # the card block stays allocated until the copy has run
            event = torch.cuda.Event()
            event.record(stream)
        else:
            host.copy_(t)
            event = None
        TAIL["offloaded"] += 1
        TAIL["bytes"] += t.numel() * t.element_size()
        return ("rl1-offload", t.device, host, event, t.requires_grad)

    def unpack(packed):
        if not (isinstance(packed, tuple) and len(packed) == 5 and packed[0] == "rl1-offload"):
            return packed
        _, device, host, event, _ = packed
        if event is not None:
            torch.cuda.current_stream(device).wait_event(event)
            return host.to(device, non_blocking=True)
        return host.to(device)

    return pack, unpack


@contextlib.contextmanager
def tail_scope(model, tail=None):
    """Around a grad window pass: ``TAIL["mode"]`` "offload" moves the saved tensors seen outside the checkpointed
    layers whose last dim is the residual stream's (hc_count x hidden; the layers' checkpoint inputs and the final
    mixer's input) to pinned host memory. Inside a checkpointed layer the checkpoint's own hooks take precedence, so
    nothing else is touched. "off": no hooks (today's path). ``TAIL["min_tokens"]`` (cmd ``tail_ckpt_min``, real-v4):
    only windows whose gradient tail (whole window, all ranks) reaches it offload (T1c-v4: w96 CP2 215.5 GiB without,
    w128 CP2 255.5 without / 166 with, +26% time)."""
    if TAIL["mode"] == "off" or (tail is not None and int(tail) < int(TAIL["min_tokens"])):
        yield
        return
    if TAIL["mode"] not in TAIL_MODES:
        raise ValueError(f"tail_ckpt {TAIL['mode']!r} not in {TAIL_MODES}")
    import torch

    width = TAIL["width"]
    if width is None:
        config = getattr(model, "config", None)
        text = getattr(config, "text_config", None) or config
        width = int(text.hc_count) * int(text.hidden_size)
    pack, unpack = _offload_hooks(width)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


# adaptive_ckpt (RL redesign, 26 Oct; trainer cmd ``adaptive_ckpt``, default off): per grad window, ``n`` decoder
# layers (evenly spread) run without activation checkpointing when the predicted card peak fits ``budget_gib``.
# Predicted card peak = base (context + allocated) + gap (the reserve above the allocated peak, CKPT["gap"]) +
# (k a + n (k b + c)) x margin, k = tail / 1024 (GiB). D2 (27 Oct) fit over its 24 warm windows: a 2.72 (the fully
# checkpointed window's growth per 1K tail tokens), b 0.441 (one uncheckpointed layer's saved activations per 1K),
# c 3.27 (one uncheckpointed layer's fixed cost, independent of tokens): rms 0.64, max 1.29 GiB; the two-term model
# (no c) missed by up to 11.5 GiB. margin 1.0: a 1.05 factor put ~8.5 GiB under D2's peaks. Max-tracking: a window
# measured above its prediction charges the excess per 1K tokens to a and to b / n, never lowered (D2 replay: 11/11
# warm windows per rank in 241-252 GiB, tests/test_rl1_ckpt_replay.py); an out-of-memory window doubles b (a when n 0).
# Single-card and DP windows only: a CP window keeps full checkpointing. ckpt_calibrate (one-shot): c solved from
# the first non-cold window with n >= 1, up or down. The cold window (the process's first) plans
# ``first_margin_gib`` under the budget (first-call allocations; D2's first windows reached 258/267 GiB).
CKPT_LEVELS = tuple(range(49))  # every n: a coarse ladder (0, 6, 12, 16, 24, ...) left D1 windows 10-50 GiB under the budget
CKPT = {"on": False, "budget_gib": 245.0, "a": 2.72, "b": 0.441, "c": 3.27, "margin": 1.0, "levels": CKPT_LEVELS,
        "force": None, "raised": 0, "ooms": 0, "step_peak": 0.0, "mem": None, "given": {},
        "calibrate": False, "calibrated": None, "windows": 0, "first_margin_gib": 5.0,
        # card terms: the allocator's reserve above the allocated peak, sampled only on windows whose allocated peak
        # passed the reserve held before them (D2: a reserve kept from window 1 minus a smaller later peak read as a
        # 33-77 GiB "gap"); each sample replaces the last (the first replaces the prior). 4.4: D1's call ends.
        "gap": 4.4, "gap_source": "prior"}
KNOBS = {"a": "ckpt_a", "b": "ckpt_b", "c": "ckpt_c", "gap": "ckpt_gap_gib", "first_margin_gib": "ckpt_first_margin_gib"}


def ckpt_settings(cmd):
    """The trainer cmd's knobs (every op): adaptive_ckpt, ckpt_budget_gib; ckpt_a / ckpt_b / ckpt_c / ckpt_gap_gib /
    ckpt_first_margin_gib set a value only when it changes (a repeated value keeps what max-tracking raised)."""
    CKPT["on"] = bool(cmd.get("adaptive_ckpt"))
    CKPT["calibrate"] = bool(cmd.get("ckpt_calibrate"))
    CKPT["budget_gib"] = float(cmd.get("ckpt_budget_gib") or 245.0)
    for key, knob in KNOBS.items():
        given = cmd.get(knob)
        if given is not None and float(given) != CKPT["given"].get(key):
            CKPT[key] = CKPT["given"][key] = float(given)
            if key == "gap":
                CKPT["gap_source"] = "prior"


def ckpt_peak(tail, base_gib, n):
    """Predicted peak (GiB, base_gib's terms) of a window with ``tail`` tokens and ``n`` layers uncheckpointed."""
    k = float(tail) / 1024.0
    return base_gib + (k * CKPT["a"] + n * (k * CKPT["b"] + CKPT["c"])) * CKPT["margin"]


def ckpt_plan(tail, base_gib, layers, budget=None):
    """``(n, predicted_gib)``: the largest level whose predicted peak for ``tail`` tokens over ``base_gib`` fits the
    budget; n 0 (every layer checkpointed) when none does. ``CKPT["force"]`` (tests) sets n."""
    budget = CKPT["budget_gib"] if budget is None else budget
    if CKPT["force"] is not None:
        n = min(int(CKPT["force"]), layers)
    else:
        n = max([m for m in CKPT["levels"] if m <= layers and ckpt_peak(tail, base_gib, m) <= budget], default=0)
    return n, round(ckpt_peak(tail, base_gib, n), 3)


def ckpt_spread(layers, n):
    """The ``n`` of ``layers`` decoder layer indices run without checkpointing, evenly spread."""
    n = max(0, min(int(n), int(layers)))
    return [(2 * i + 1) * layers // (2 * n) for i in range(n)]


def ckpt_layers(model):
    return [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer") and hasattr(m, "gradient_checkpointing")]


def ckpt_apply(layers, off):
    """Checkpointing off on the layers at ``off``, on elsewhere (``off`` empty: full checkpointing)."""
    off = set(off)
    for i, layer in enumerate(layers):
        layer.gradient_checkpointing = i not in off


def ckpt_adapt(tail, n, predicted_gib, measured_gib):
    """Max-tracking: a window measured above its prediction adds the excess per 1K tokens (e) to a and e / n to b
    (n > 0): the true a + n b exceeds the old by at most e, so with both old values at or below the true ones each
    is short by at most its raise (c, the per-layer constant, is calibrated, not raised). Returns e (0.0: unchanged)."""
    if tail <= 0 or measured_gib <= predicted_gib:
        return 0.0
    e = (measured_gib - predicted_gib) / (tail / 1024.0) / CKPT["margin"]
    CKPT["a"] += e
    if n > 0:
        CKPT["b"] += e / n
    CKPT["raised"] += 1
    return round(e, 4)


class CudaMem:
    """The card readings (GiB): ``reset`` -> base (context and other non-allocator memory + allocated, before the
    window's forward); ``peak`` -> (context + the reserve's peak, context + the allocated peak) since the reset."""

    def __init__(self):
        import torch

        self.torch = torch
        self.ctx = 0.0

    def reset(self):
        cuda = self.torch.cuda  # allocator stats are host-side: no sync (U2, 27 Oct: 2 drains per window)
        self.step_seen = cuda.max_memory_allocated() / 2**30  # the step's peak so far (the reset drops it)
        cuda.reset_peak_memory_stats()
        free, total = cuda.mem_get_info()
        self.ctx = (total - free - cuda.memory_reserved()) / 2**30
        self.held = self.ctx + cuda.memory_reserved() / 2**30  # the card's use now (the reserve kept from before)
        return self.ctx + cuda.memory_allocated() / 2**30

    def peak(self):
        cuda = self.torch.cuda
        return self.ctx + cuda.max_memory_reserved() / 2**30, self.ctx + cuda.max_memory_allocated() / 2**30


class CpuMem:
    """No card (the CPU preflight's tiny trainer, CI): the readings of a modelled card, so the planner, its n=0 floor,
    the calibration and the raises run their real code. Base ``RL1_CPU_BASE_GIB`` (default 100); a window's growth
    from the planner's own coefficients x ``RL1_CPU_TRUE`` (default 1.0; > 1: the card grows more than predicted)
    with its tail x ``RL1_CPU_TAIL_X`` (default 256: a tiny model's 160-token tail reads as a 40K one)."""

    def __init__(self):
        self.ctx, self.held, self.planned = 0.0, 0.0, (0, 0)
        self.base = float(os.environ.get("RL1_CPU_BASE_GIB") or 100.0)
        self.true = float(os.environ.get("RL1_CPU_TRUE") or 1.0)
        self.tail_x = float(os.environ.get("RL1_CPU_TAIL_X") or 256.0)

    def plan(self, tail, n):
        self.planned = (tail, n)

    def reset(self):
        self.held = self.base
        return self.base

    def peak(self):
        tail, n = self.planned
        k = tail * self.tail_x / 1024.0
        grown = self.base + self.true * (k * CKPT["a"] + n * (k * CKPT["b"] + CKPT["c"]))
        return grown + float(CKPT["gap"] or 0.0), grown  # (card, allocated)


def card_mem():
    import torch

    return CudaMem() if torch.cuda.is_available() else CpuMem()


def ckpt_begin(model, tail):
    """Before a grad window's forward: plan n, turn checkpointing off on those layers. None when off."""
    if not CKPT["on"]:
        return None
    if CKPT["mem"] is None:
        CKPT["mem"] = card_mem()
    if isinstance(CKPT["mem"], CpuMem):
        tail = int(tail * CKPT["mem"].tail_x)  # planned at the modelled size
    layers = ckpt_layers(model)
    base = CKPT["mem"].reset()
    CKPT["step_peak"] = max(CKPT["step_peak"], getattr(CKPT["mem"], "step_seen", 0.0))
    gap = CKPT["gap"]
    cold = CKPT["windows"] == 0  # the process's first window: first-call allocations on top
    budget = CKPT["budget_gib"] - (CKPT["first_margin_gib"] if cold else 0.0)
    n, predicted = ckpt_plan(tail, base + gap, len(layers), budget)  # card terms: allocated growth + the reserve gap
    if isinstance(CKPT["mem"], CpuMem):
        CKPT["mem"].plan(tail / CKPT["mem"].tail_x, n)
    ckpt_apply(layers, ckpt_spread(len(layers), n))
    return {"layers": layers, "tail": int(tail), "n": n, "base": base, "pred": predicted, "gap": gap,
            "gap_source": CKPT["gap_source"], "cold": cold,
            "held": getattr(CKPT["mem"], "held", 0.0)}


def ckpt_end(plan, failed=None):
    """After the window's backward (or its failure): every layer checkpointed again; the measured peak recorded and
    the coefficients raised if it passed the prediction (before the next window is planned). The window row's
    ``ckpt`` dict: n, pred_gib (card terms: allocated + gap), peak_gib (allocated) and card_gib (the reserve's
    peak), the raise; gap_gib when the reserve grew (card - allocated peak)."""
    if plan is None:
        return None
    ckpt_apply(plan.pop("layers"), ())
    row = {"n": plan["n"], "pred_gib": plan["pred"], "base_gib": round(plan["base"], 3), "gap_used": plan["gap"],
           "gap_source": plan["gap_source"]}  # prior / measured (the first measurement) / tracked (raised since)
    if failed is not None:
        if "out of memory" in str(failed).lower():
            CKPT["ooms"] += 1
            CKPT["b" if plan["n"] > 0 else "a"] *= 2.0
            row["oom"] = True
        return row
    card, measured = CKPT["mem"].peak()  # card: the reserve's peak (fragmentation included), for the record
    CKPT["step_peak"] = max(CKPT["step_peak"], measured - CKPT["mem"].ctx)
    row.update(peak_gib=round(measured, 3), card_gib=round(card, 3))
    if measured > plan["held"] + 1e-3:  # the window's own peak grew the reserve: its excess is the gap (D2 27 Oct:
        # a reserve kept from an earlier, larger window minus this window's peak is not one)
        row["gap_gib"] = gap = round(card - measured, 3)
        CKPT["gap"], CKPT["gap_source"] = gap, ("measured" if CKPT["gap_source"] == "prior" else "tracked")
    cold, CKPT["windows"] = CKPT["windows"] == 0, CKPT["windows"] + 1
    if CKPT["calibrate"] and CKPT["calibrated"] is None and not cold and plan["n"] >= 1 and plan["tail"] > 0:
        row["calibrated"] = CKPT["calibrated"] = ckpt_calibrate(plan["tail"], plan["n"], plan["base"], measured)
    else:
        row["raised"] = ckpt_adapt(plan["tail"], plan["n"], plan["pred"] - plan["gap"], measured)
    return row


def ckpt_calibrate(tail, n, base_gib, measured_gib):
    """ckpt_calibrate (one-shot): c solved from the first non-cold window with n >= 1, up or down (a and b
    kept). Max-tracking only after it."""
    k = tail / 1024.0
    old = CKPT["c"]
    CKPT["c"] = max(1e-3, (measured_gib - base_gib - k * CKPT["a"] - n * k * CKPT["b"]) / n)
    return {"c_old": round(old, 4), "c_new": round(CKPT["c"], 4)}


def ckpt_state():
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in CKPT.items()
            if k not in ("mem", "levels", "given", "force")}
