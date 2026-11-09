"""MFU stage M (RL arm, 21 Oct): memory per tail token of the trainer's window pass.

- ``chunked_head``: the LM head and every per-token read of its logits (actor and raw log-probabilities, entropy,
  greedy hits, the top-20 vocabulary bins) in position chunks; the [N, V] logits never exist whole. The actor's
  backward recomputes each chunk's logits (Liger fused-linear-cross-entropy style), so nothing of size V is saved.
  Each chunk runs the unchunked path's own functions (``rl_job.actor_logprobs`` etc.) on its rows.
- ``boundary_offload``: saved-tensor hooks that move the decoder layers' checkpoint inputs (the hc_count x hidden
  residual stream, 48 x 20 KiB per token on Flash) to pinned host memory in the forward and back in the backward.

Switches: ``HEAD["chunked"]`` (trainer cmd ``chunked_head``), ``TAIL["mode"]`` (cmd ``tail_ckpt``); off by default.
"""
from __future__ import annotations

import contextlib

HEAD = {"chunked": False, "chunk": 1024, "capture": False, "calls": 0}
# cpu: offload CPU tensors too (tests: the copy is a separate host tensor)
TAIL = {"mode": "off", "min_bytes": 1 << 20, "width": None, "offloaded": 0, "bytes": 0, "kept": 0, "cpu": False}
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


def _actor(rows, tokens, support, temperature):
    import rl_job

    return rl_job.actor_logprobs(rows, tokens, support, temperature)


def _chunked_function():
    import torch

    class ChunkedActor(torch.autograd.Function):
        """actor log-probabilities [N] from hidden [N, H]; forward and backward one chunk of rows at a time. Extras
        (no grad) are computed in the forward's loop into ``extras``."""

        @staticmethod
        def forward(ctx, hidden, tokens, support, temperature, project, chunk, extras):
            device = hidden.device.type
            ctx.cast = (device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
            ctx.project, ctx.chunk, ctx.temperature = project, chunk, temperature
            n = hidden.shape[0]
            actor = torch.empty(n, dtype=torch.float32, device=hidden.device)
            valid = torch.empty(n, dtype=torch.bool, device=hidden.device)
            for i in range(0, n, chunk):
                j = min(n, i + chunk)
                logits = project(hidden[i:j])
                actor[i:j], valid[i:j] = _actor(logits, tokens[i:j], support[i:j], temperature)
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
            for i in range(0, n, chunk):
                j = min(n, i + chunk)
                g = grad_actor[i:j]
                if not bool(g.ne(0).any()):
                    continue
                with torch.enable_grad(), torch.autocast(device, dtype=dtype, enabled=enabled):
                    h = hidden[i:j].detach().requires_grad_(grad_hidden is not None)
                    actor, _ = _actor(ctx.project(h), tokens[i:j], support[i:j], ctx.temperature)
                torch.autograd.backward(actor, g)  # h.grad, plus any trainable head weight, accumulate here
                if grad_hidden is not None:
                    grad_hidden[i:j] = h.grad
                del actor, h
            return grad_hidden, None, None, None, None, None, None

    return ChunkedActor


_FUNCTION = []


def chunked_head(project, hidden, tokens, support, temperature, *, hit=False, vocab=None, chunk=None):
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
    entropy = [0.0, 0]

    def each(logits, i, j, valid):
        rows = logits.detach()
        raws.append(rl_job.raw_logprobs(rows, tokens[i:j]))
        s, c = rl1_trainer.token_entropy(rows, valid, temperature)
        entropy[0] += s
        entropy[1] += c
        if hit:
            hits.append(rows.argmax(-1) == tokens[i:j])
        if vocab is not None:
            tops.append(rl1_trainer.vocab_top(rows, at_ids=[table[i:j] for table in vocab]))

    actor, valid = _FUNCTION[0].apply(hidden, tokens, support, temperature, project, chunk, {"each": each})
    dist = None
    if vocab is not None:
        cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0)
        dist = {"ids": cat([t["ids"] for t in tops]), "logp": cat([t["logp"] for t in tops]),
                "at": [cat([t["at"][k] for t in tops]) for k in range(len(vocab))]}
    raw = torch.cat(raws) if raws else hidden.new_zeros(0).float()
    return {"actor": actor, "valid": valid, "raw": raw, "entropy": (entropy[0], entropy[1]),
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
def tail_scope(model):
    """Around a grad window pass: ``TAIL["mode"]`` "offload" moves the saved tensors seen outside the checkpointed
    layers whose last dim is the residual stream's (hc_count x hidden; the layers' checkpoint inputs and the final
    mixer's input) to pinned host memory. Inside a checkpointed layer the checkpoint's own hooks take precedence, so
    nothing else is touched. "off": no hooks (today's path)."""
    if TAIL["mode"] == "off":
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
