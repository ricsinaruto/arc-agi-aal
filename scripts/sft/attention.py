"""The full-attention fast path on GPU (from rl1_trainer): the vectorized QSA indexer (the stock one loops over queries
in Python, ~360 tok/s) and the selected-key attention through sparse.py's Triton kernel. ``SEL["on"]`` (sel_cache):
inside a gradient window pass each layer keeps its forward's picks and the checkpoint recompute reuses them (same
inputs, same picks) instead of running the indexer again; ~0.4 GB per full-attention layer at a 48K window."""
import os


SPARSE = {"mode": os.environ.get("RL1_ATTENTION", "masked"), "chunk": 256, "calls": 0, "fallback": 0, "cudnn": 0,
          "union": False,  # sparse2.UnionSelectedAttention (RL MFU-1 stages E + P, trunk/rl a08289a03) for "triton"
          "cudnn_failed": 0, "density": False, "blocks": [0, 0]}
INDEXER = {"vectorized": 0, "fallback": 0}
INDEX = {"tf32": False, "check": False, "match": 0, "picked": 0}  # indexer scores in TF32 (gated: index_check)
CAUSAL = {"key": None, "counts": None, "max": None, "hits": 0, "piece": 1 << 27,  # plain_causal memo, one model call
          "memo": True, "dense": False}  # checks (teacher --ab-blocks ref): no memo; no shortcut (HF's masked path)
SEL = {"on": False, "window": False, "store": {}, "hits": 0, "stores": 0}
MARKS = {}  # causal_marker stand-ins: data_ptr -> (per-query key counts, keys)
# Served-KV emulation for eval-only KL screens (no grad): the QSA layers' K / V go through the serving cache's
# quantizer before they enter the cache. "kv4": the served int4 page (arc_infra.kv4_runtime: the pinned RHT, rotated
# rows cast to BF16, asymmetric 4-bit per (token, head), zero point in the scale's low mantissa bits); "kv2": the same
# at 2 bits; "kv2kc": 2-bit, K per channel over 128-token groups (KIVI-style), V per token, both rotated.
# ``KVQ["train"]`` (config kv_train): the same in every forward, grad on (straight-through: the gradient reaches K / V
# unchanged); an eval-only ``mode`` takes precedence in no-grad reads.
KVQ = {"mode": None, "group": 128, "calls": 0, "train": None, "train_calls": 0}
_RHT_SIGNS = {}


def _rht_signs(d, device):
    """The pinned ``_get_rht_signs(d, 0, device)`` (kv4_runtime.rht_signs)."""
    import torch

    key = (d, str(device))
    if key not in _RHT_SIGNS:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0x9E3779B9)
        _RHT_SIGNS[key] = (2.0 * torch.bernoulli(torch.full((d,), 0.5), generator=gen) - 1.0).to(device)
    return _RHT_SIGNS[key]


def _hadamard(x):
    import torch

    d, h = x.shape[-1], 1
    while h < d:
        xv = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        x = torch.stack([xv[..., 0, :] + xv[..., 1, :], xv[..., 0, :] - xv[..., 1, :]], dim=-2).reshape(x.shape)
        h <<= 1
    return x


def rht(x, inverse=False):
    """The pinned ``single_rht``: forward H (D x), inverse D (H x); unnormalized (H H = d I)."""
    s = _rht_signs(x.shape[-1], x.device)
    return _hadamard(x) * s if inverse else _hadamard(x * s)


def _round_away(x):
    import torch

    return torch.where(x >= 0, (x + 0.5).trunc(), (x - 0.5).trunc())


def asym_fake(x, bits, dim=-1):
    """kv4_runtime.quantize4_reference + dequantize4_reference generalized to ``bits`` along ``dim``: scale (max -
    min) / (2^bits - 1), zero point and codes rounded half away from zero, dequantized with the stored scale (its 4 low
    mantissa bits hold the zero point)."""
    import torch

    top = float(2 ** bits - 1)
    lo, hi = x.amin(dim=dim, keepdim=True), x.amax(dim=dim, keepdim=True)
    scale = ((hi - lo) / top).clamp_min(1e-6)
    zp = _round_away(-lo / scale).clamp(0, top)
    q = _round_away(x * (1.0 / scale) + zp).clamp(0, top)
    stored = (scale.contiguous().view(torch.int32) & -16).view(torch.float32)
    return (q - zp) * stored


KV_CHUNK = 8192  # kv_fake's tokens per piece (a multiple of every kv2kc group): fp32 temporaries of one piece


def kv_fake(key, value, mode):
    """K, V [B, H, T, D] as the serving cache returns them under ``mode`` (in the input dtype), ``KV_CHUNK`` tokens at
    a time (every value depends only on its own token, or its kv2kc token group)."""
    import torch

    t = key.shape[-2]
    if t <= KV_CHUNK:
        k, v = _kv_fake(key, value, mode)
    else:
        k, v = torch.empty_like(key), torch.empty_like(value)
        for a in range(0, t, KV_CHUNK):
            k[..., a:a + KV_CHUNK, :], v[..., a:a + KV_CHUNK, :] = _kv_fake(key[..., a:a + KV_CHUNK, :],
                                                                            value[..., a:a + KV_CHUNK, :], mode)
    KVQ["calls"] += 1
    return k, v


def _kv_fake(key, value, mode):
    import torch

    bits = 4 if mode == "kv4" else 2
    d = key.shape[-1]
    rk = rht(key.float()).to(torch.bfloat16).float()
    rv = rht(value.float()).to(torch.bfloat16).float()
    if mode == "kv2kc":
        g, t = KVQ["group"], rk.shape[-2]
        pad = (-t) % g
        rkp = torch.nn.functional.pad(rk, (0, 0, 0, pad), mode="replicate") if pad else rk
        qk = asym_fake(rkp.reshape(*rk.shape[:-2], (t + pad) // g, g, d), bits, dim=-2)
        qk = qk.reshape(*rk.shape[:-2], t + pad, d)[..., :t, :]
    else:
        qk = asym_fake(rk, bits)
    qv = asym_fake(rv, bits)
    return (rht(qk, inverse=True) / d).to(key.dtype), (rht(qv, inverse=True) / d).to(value.dtype)


class _KVSTE:
    """Forward: ``kv_fake(key, value, mode)``; backward: identity into key and value."""
    fn = None

    @classmethod
    def get(cls):
        if cls.fn is None:
            import torch

            class Fn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, key, value, mode):
                    return kv_fake(key, value, mode)

                @staticmethod
                def backward(ctx, gk, gv):
                    return gk, gv, None

            cls.fn = Fn
        return cls.fn


def kv_ste(key, value, mode):
    """K, V as the serving cache stores them (``kv_fake``), straight-through."""
    KVQ["train_calls"] += 1
    return _KVSTE.get().apply(key, value, mode)


def selected_on():
    """The selected-key forward is installed and picks the plain causal path (``causal_marker`` is understood)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    return SPARSE["mode"] in ("selected", "triton") and getattr(modeling.Qwen4ExpTextAttention, "_rl1_selected", False)


def causal_marker(rows, keys, device):
    """Stands in for the [1, 1, rows, keys] causal mask of a pass over the last ``rows`` of ``keys`` positions (query
    i sees keys 0 .. keys - rows + i) on the selected-key paths, which need only each query's key count: a 1-element
    tensor registered with its counts (the MTP head's dense mask was 5.4 GiB at 48K x 117K). Only when
    ``selected_on()``."""
    import weakref

    import torch

    mark = torch.zeros(1, 1, 1, 1, dtype=torch.bool, device=device)
    MARKS[mark.data_ptr()] = (torch.arange(keys - rows + 1, keys + 1, device=device), keys)
    weakref.finalize(mark, MARKS.pop, mark.data_ptr(), None)
    return mark


def plain_causal(attention_mask):
    """Per query the number of visible keys when the mask is a plain causal one (batch 1, query i sees keys
    0..n_i-1), else None. The answer for the same mask tensor (storage, shape, version) is reused by the call's
    later full-attention layers and the recompute (the check reads an [L, keys] mask in each layer); ``reset`` at
    each model call."""
    import torch

    if attention_mask.numel() == 1 and attention_mask.data_ptr() in MARKS:  # a marker stands for its counts
        counts, keys = MARKS[attention_mask.data_ptr()]
        CAUSAL.update(key=None, counts=counts, max=keys)
        return counts
    if CAUSAL["dense"]:
        return None
    key = (attention_mask.data_ptr(), tuple(attention_mask.shape), attention_mask.dtype, attention_mask._version)
    if CAUSAL["memo"] and CAUSAL["key"] == key:
        CAUSAL["hits"] += 1
        return CAUSAL["counts"]
    counts = None
    if attention_mask.shape[0] == 1:
        # in row pieces of <= CAUSAL["piece"] mask elements: the [L, keys] temporaries of a whole-mask pass were 43 GiB
        # at a 48K window over a 117K-key MTP cache (26 Oct smoke v5 OOM)
        m = attention_mask[0, 0]
        rows, keys = m.shape
        positions = torch.arange(keys, device=m.device)
        counts = torch.empty(rows, dtype=torch.long, device=m.device)
        step = max(1, CAUSAL["piece"] // max(keys, 1))
        for a in range(0, rows, step):
            v = m[a:a + step]
            v = v if v.dtype == torch.bool else v == 0
            c = v.sum(-1, dtype=torch.int32)
            if not torch.equal(v, positions[None, :] < c[:, None]):
                counts = None
                break
            counts[a:a + step] = c
    CAUSAL.update(key=key if CAUSAL["memo"] else None, counts=counts,
                  max=int(counts.max()) if counts is not None and counts.numel() else None)
    return counts


def reset():
    CAUSAL.update(key=None, counts=None, max=None)


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
        known = CAUSAL["max"] if CAUSAL["counts"] is counts else None
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
            if INDEX.get("fp8"):  # served indexer_fp8: the compressed-key cache stores e4m3 (cast on store, read up-cast)
                block_keys = block_keys.clamp(-448, 448).to(torch.float8_e4m3fn).to(block_keys.dtype)
                INDEX["fp8_calls"] = INDEX.get("fp8_calls", 0) + 1
            keys_t = block_keys.float().transpose(0, 1)
            ids = torch.arange(blocks, device=device)
            for begin in range(0, length, chunk):
                end = min(length, begin + chunk)
                valid = ids[None, :] < complete[begin:end, None]

                def pick(tf32):
                    old = torch.backends.cuda.matmul.allow_tf32
                    torch.backends.cuda.matmul.allow_tf32 = tf32  # the fp32 SIMT GEMM, or TF32 tensor cores
                    try:
                        sc = torch.relu(torch.matmul(q[0, begin:end].float(), keys_t)).sum(dim=1)
                    finally:
                        torch.backends.cuda.matmul.allow_tf32 = old
                    sc = sc / math.sqrt(self.index_head_dim)
                    return sc.masked_fill(~valid, float("-inf")).topk(top, dim=-1).indices

                if INDEX.get("legacy"):  # leg-v5's code (teacher.py --ab-blocks)
                    scores = torch.relu(torch.matmul(q[0, begin:end].float(), keys_t)).sum(dim=1)
                    scores = scores / math.sqrt(self.index_head_dim)
                    picked = scores.masked_fill(~valid, float("-inf")).topk(top, dim=-1).indices
                else:
                    picked = pick(INDEX["tf32"] and not INDEX["check"])
                if INDEX["check"]:  # fp32 picks used; the TF32 picks' share found among them is counted (no sync)
                    alt = pick(True)
                    seen = torch.zeros(end - begin, blocks, dtype=torch.bool, device=device)
                    seen.scatter_(1, picked, True)
                    hit = torch.gather(seen, 1, alt) & torch.gather(valid, 1, alt)
                    INDEX["match"] = INDEX["match"] + hit.sum()
                    INDEX["picked"] = INDEX["picked"] + torch.gather(valid, 1, picked).sum()
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
        use = SEL["on"] and SEL["window"]
        kept = SEL["store"].get(id(self)) if use else None
        if kept is not None and kept["length"] == hidden_states.shape[1] and torch.equal(kept["counts"], counts):
            picked = kept["picked"].long()  # the forward's picks: the recompute skips the indexer
            SEL["hits"] += 1
        else:
            picked = qsa_select(self.indexer, hidden_states, position_embeddings, counts, past_key_values)
            if use:
                SEL["store"][id(self)] = {"length": hidden_states.shape[1], "counts": counts,
                                          "picked": picked.to(torch.int32)}
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
        if KVQ["mode"] and not torch.is_grad_enabled():  # eval-only served-KV screen: quantized as the cache stores it
            key_states, value_states = kv_fake(key_states, value_states, KVQ["mode"])
        elif KVQ["train"]:  # kv_train: every forward (prefix, window, recompute; the MTP draft's QSA layer too, whose
            # served cache is the same int4 page), straight-through
            key_states, value_states = kv_ste(key_states, value_states, KVQ["train"])
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if SPARSE["mode"] == "triton":  # rl1_sparse: the fused gather kernel (fp32 atomics for dk / dv)
            attn_output = kernel().apply(query_states, key_states, value_states, picked, self.scaling, None)
        else:
            attn_output = function.apply(query_states, key_states, value_states, picked, self.scaling, SPARSE["chunk"])
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    cls.forward = forward
    cls._rl1_selected = True
    cls._rl1_original = original


def kernel():
    """The selected-key Triton kernel: the query-union one (``SPARSE["union"]``; no dk / dv for a cached no-grad
    prefix's keys, which are the cache's and get none anyway) or the one-program-per-query one."""
    if SPARSE["union"]:
        from sparse2 import UnionSelectedAttention

        UnionSelectedAttention.NOGRAD_PREFIX = True
        return UnionSelectedAttention
    from sparse import TritonSelectedAttention

    return TritonSelectedAttention


def union_check(device, length=640, keys=4096, width=2051, seed=0):
    """The union kernel against the per-query one on random picks (``length`` queries, the last of ``keys``): out
    and dq equal to BF16 rounding, dk / dv to fp32-atomic order (rel. error <= 1e-2). Returns (ok, errors)."""
    import torch

    from sparse import TritonSelectedAttention
    from sparse2 import UnionSelectedAttention

    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, 24, length, 256, device=device, dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, 2, keys, 256, device=device, dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, keys, 256, device=device, dtype=torch.bfloat16, generator=g)
    rows = []
    for i in range(length):
        n = keys - length + i + 1
        r = torch.randperm(n, device=device, generator=g)[:width].sort().values
        rows.append(torch.nn.functional.pad(r, (0, width - len(r)), value=-1))
    picked = torch.stack(rows)
    grad = torch.randn_like(q)
    got = []
    for fn in (TritonSelectedAttention, UnionSelectedAttention):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        UnionSelectedAttention.NOGRAD_PREFIX = False
        out = fn.apply(qq, kk, vv, picked, 256 ** -0.5, None)
        out.backward(grad)
        got.append([out.float(), qq.grad.float(), kk.grad.float(), vv.grad.float()])
    errs = [float((a - b).norm() / b.norm().clamp_min(1e-12)) for a, b in zip(got[1], got[0])]
    return all(e <= 1e-2 for e in errs), [round(e, 6) for e in errs]


def index_check(run, floor=0.995):
    """``run()`` (a held-out pass) with the fp32 indexer, counting how many of the TF32 indexer's picked blocks the
    fp32 one also picked; TF32 is kept on only if that share is >= ``floor``. Returns {"match", "on"}."""
    import torch

    INDEX.update(check=True, match=0, picked=0)
    try:
        run()
    finally:
        INDEX["check"] = False
    picked = int(INDEX["picked"]) if torch.is_tensor(INDEX["picked"]) else INDEX["picked"]
    match = float(INDEX["match"]) / picked if picked else 1.0
    INDEX["tf32"] = match >= floor
    return {"match": round(match, 6), "picked": picked, "on": INDEX["tf32"]}


def install(mode="triton", sel_cache=False):
    """``mode``: "triton" (GPU) or "selected" (the torch SelectedAttention, CPU tests)."""
    import fla_pin

    fla_pin.pin()  # fla's BK=BV=128 chunk_o config writes wrong tiles on B300 now and then
    SPARSE["mode"] = mode
    SEL["on"] = bool(sel_cache)
    vectorized_indexer()
    selected_attention()


def window(on):
    """Around a gradient window pass: the selection store starts empty and is dropped after."""
    SEL.update(window=bool(on) and SEL["on"], store={})
    reset()
