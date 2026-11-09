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


def vocab_top(rows, k=20, chunk=1024, at_ids=()):
    """Per position of ``rows`` [n, V] (logits), the full-vocabulary log-distribution at temperature 1 reduced to its
    top ``k`` tokens (``ids``, ``logp``), and the log-probabilities of each id table in ``at_ids`` ([n, k] each),
    computed in row chunks (a 20K x 248K float table is 20 GB)."""
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
    cat = lambda xs: torch.cat(xs).cpu() if xs else torch.zeros(0)
    return {"ids": cat(ids), "logp": cat(logp), "at": [cat(x) for x in at]}


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


def plain_causal(attention_mask):
    """Per query the number of visible keys when the mask is a plain causal one (batch 1, query i sees keys
    0..n_i-1), else None."""
    import torch

    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    if visible.shape[0] != 1:
        return None
    counts = visible[0, 0].sum(-1)
    positions = torch.arange(visible.shape[-1], device=visible.device)
    return counts if torch.equal(visible[0, 0], positions[None, :] < counts[:, None]) else None


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
        blocks = int(complete.max())
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
                picked = scores.masked_fill(~valid, float("-inf")).topk(top, dim=-1).indices
                tokens = picked[:, :, None] * ratio + offsets
                tokens = torch.where(torch.gather(valid, 1, picked)[:, :, None], tokens, -1)
                out[begin:end, : top * ratio] = tokens.reshape(end - begin, top * ratio)
        tail = (complete * ratio)[:, None] + offsets[: ratio - 1]
        out[:, top * ratio:] = torch.where(tail < counts[:, None], tail, -1)
    return out


SPARSE = {"on": os.environ.get("RL1_ATTENTION", "selected") == "selected", "chunk": 256, "calls": 0, "fallback": 0}


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
    ``SelectedAttention`` as key positions (no mask is built); else the original forward. ``SPARSE["on"]`` switches
    it (``op_attention``)."""
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    cls = modeling.Qwen4ExpTextAttention
    if getattr(cls, "_rl1_selected", False):
        return
    original = cls.forward
    function = selected_attention_function()

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        counts = plain_causal(attention_mask) if SPARSE["on"] and attention_mask is not None else None
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
        selected_attention()  # SPARSE["on"]: RL1_ATTENTION=selected (default) or masked
        self.model.train()
        self.params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.initial = {n: p.detach().to("cpu", copy=True) for n, p in self.params if "lora_A" in n}
        self.optimizer = self.make_optimizer(args.lr)
        self.checkpointing = "full"
        set_checkpointing(self.model, self.checkpointing)
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
        SPARSE["on"] = cmd["mode"] == "selected"
        SPARSE["chunk"] = int(cmd.get("chunk", SPARSE["chunk"]))
        return {"mode": cmd["mode"], "sparse": dict(SPARSE), "indexer": dict(INDEXER)}

    def op_score(self, cmd):
        return {"windows": self.score_windows(cmd["windows"], cmd.get("label"))}

    def adapter_norm(self):
        return round(math.sqrt(sum(float(p.detach().float().pow(2).sum()) for _, p in self.params)), 5)

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

        result = {"windows": [], "errors": []}
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
        if cmd.get("norm") == "try":  # windows per try (``expand`` names them try#i)
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
                if per_try:  # each try counts once: its windows' mean weighted log-likelihood (divided by tries below)
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
                change = math.sqrt(float(sum((p.detach() - b).float().pow(2).sum() for (_, p), b in zip(self.params, before))))
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
                time.sleep(1.0)
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
                                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2), "card_used_gib": used}
            self.rec.event("op", op=cmd["op"], n=index, **{k: v for k, v in result.items() if k != "traceback"})
            tmp = self.dir / "done" / f"{index:04d}.json.tmp"
            tmp.write_text(json.dumps(result, default=str))
            tmp.rename(self.dir / "done" / f"{index:04d}.json")
            index += 1
            if cmd["op"] == "exit":
                return


GPU_OPS = ("fit", "score_ids", "score", "update", "profile", "reset", "compare_lora", "split_windows", "checkpoint", "load_checkpoint")  # these bring a parked adapter back first


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
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.serve()


if __name__ == "__main__":
    main()
