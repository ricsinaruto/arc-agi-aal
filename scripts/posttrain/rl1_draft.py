"""RL redesign stage D (26 Oct): Flash's MTP draft head trained with LoRA beside the policy LoRA, so speculative
decoding keeps its acceptance while the policy moves ("MTP with LoRA and a 4-bit draft head; draft head trained
with LoRA"; with a frozen draft, mean accepted length fell from ~2.8 to ~1.1 after SFT).

Loss (trunk/sft scripts/sft/mtp.py's self-distillation): per trained token, KL(trunk || draft) between the policy's
own distribution of the token two ahead (detached) and the draft's, on the rollout's tokens, x ``weight``. The draft's
inputs (token embedding, the trunk's pre-mixer hyper-connection state) and the LM head are detached: no gradient
reaches the policy. Logged per window: KL, top-1 agreement with the sampled token, with the trunk's argmax, and the
acceptance of a draft token sampled at the request's temperature, sum_v min(p_T, q_T) (vLLM's drafter samples with
the temperature only; rejection sampling against the trunk).

Alignment (vLLM's MTP proposer): draft row i takes the trunk's pre-mixer state at position i, the embedding of token
i + 1 and the rotary position of i, and drafts token i + 2: its target is the trunk's row i + 1. A trained token p
(actor row p - 1) pairs with draft row p - 2. The draft rows of a span [a, b) of trunk positions are [a - 1, b - 1)
(row -1 dropped): they read ids[:b] and trunk states before b only, so a prefix cache and its chunk-boundary snapshots
carry the draft's prefix state (``prefix_chunk``) and a window continues it (``window``). Sequence parallel
(rl1_cp): the same rows over the trunk's shards, the draft's attention through rl1_cp.attention_forward under its own
Ctx (``cp_prefix``, ``cp_window``); each rank's draft rows pair with its own kept trunk rows, so the KL is a sum and
its metrics are reduced, nothing gathered.

Module: the checkpoint's ``mtp.*`` (BF16; routed experts rounded to NVFP4 values, as the server's online NvFp4 draft
experts), with its own MoE forward (no trainer / Axolotl class patch and no served-route capture or replay reaches it)
and the trainer's selected-attention kernels. LoRA (the policy adapter's rank and alpha) on the BF16 dense projections
the server's draft can merge (``TARGETS``).

Export: ``draft_lora.safetensors`` / ``draft_lora.json`` beside the policy adapter (vLLM reads only
adapter_model.safetensors, so the policy adapter loads unchanged under its id). The pinned vLLM never LoRA-wraps the
draft (v1/worker/gpu/spec_decode/eagle/utils.py; serving_kernels_runtime.ArcNvfp4DraftMoEMethod drops
is_lora_enabled), so the server applies it by merging into the draft's BF16 weights at adapter load (one adapter at
a time: src/arc_infra/draft_lora_runtime.merge, ARC_DRAFT_LORA); ``vllm_module`` maps each module to the draft
model's packed parameter and shard."""
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# checkpoint ``mtp.<module>`` names; the vLLM draft keeps all of them in BF16 (hf_quant_config excludes mtp.*)
TARGETS = ("fc_embedding", "fc_hidden", "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj",
           "layers.0.self_attn.v_proj", "layers.0.self_attn.o_proj", "layers.0.mlp.shared_expert.gate_proj",
           "layers.0.mlp.shared_expert.up_proj", "layers.0.mlp.shared_expert.down_proj")
# pinned vLLM (runtime l24): models/qwen3_8_flash_next/nvidia/mtp.py (sha256 7735cee4..., _remap_mtp_weight_name:
# mtp.X -> model.X) and model_executor/models/qwen3_next.py (28aaeee9..., Qwen3NextModel.hf_to_vllm_mapper stacking)
PACKED = {".q_proj": (".qkv_proj", "q"), ".k_proj": (".qkv_proj", "k"), ".v_proj": (".qkv_proj", "v"),
          ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
          ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1)}
FORMAT = "arc-draft-lora-v1"
E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # rounding boundaries of the E2M1 grid 0 .5 1 1.5 2 3 4 6
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ------------------------------------------------------------------------------------------------ modules
class LoraLinear(torch.nn.Module):
    """``base`` (frozen) + B A x alpha / r; A as PEFT initialises it, B zero; fp32 factors (the master copy)."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.lora_A = torch.nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=dev))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scaling


class _Router(torch.nn.Module):
    def __init__(self, experts, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(experts, hidden))


class _Experts(torch.nn.Module):
    def __init__(self, experts, hidden, inter):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(experts, 2 * inter, hidden))
        self.down_proj = torch.nn.Parameter(torch.zeros(experts, hidden, inter))


class DraftMoe(torch.nn.Module):
    """Qwen4ExpTextSparseMoeBlock's arithmetic (softmax router, top-k, renormalised, shared expert behind a sigmoid
    gate) with the routed experts grouped by expert (one host read of the counts, one GEMM pair per expert)."""

    def __init__(self, cfg, m):
        super().__init__()
        self.gate = _Router(cfg.num_experts, cfg.hidden_size)
        self.experts = _Experts(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        self.shared_expert = m.Qwen4ExpTextMLP(cfg, intermediate_size=cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(cfg.hidden_size, 1, bias=False)
        self.top_k, self.norm_topk = int(cfg.num_experts_per_tok), bool(cfg.norm_topk_prob)
        from transformers.activations import ACT2FN

        self.act = ACT2FN[cfg.hidden_act]

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        shared = self.shared_expert(flat)
        logits = F.linear(flat, self.gate.weight)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        weights, idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = self.routed(flat, idx, weights.to(logits.dtype))
        out = routed + torch.sigmoid(self.shared_expert_gate(flat)) * shared
        return out.reshape(shape)

    def routed(self, x, idx, weights):
        n, k = idx.shape
        gate_up, down = self.experts.gate_up_proj, self.experts.down_proj
        out = torch.zeros((n, x.shape[-1]), dtype=torch.float32, device=x.device)
        if n:
            flat = idx.reshape(-1)
            order = torch.argsort(flat, stable=True)
            tok = torch.div(order, k, rounding_mode="floor")
            xs = x[tok]
            parts, start = [], 0
            for e, c in enumerate(torch.bincount(flat, minlength=gate_up.shape[0]).tolist()):
                if c:
                    g, u = F.linear(xs[start:start + c], gate_up[e]).chunk(2, dim=-1)
                    parts.append(F.linear(self.act(g) * u, down[e]))
                    start += c
            y = torch.cat(parts) * weights.reshape(-1)[order, None]
            out = out.index_add(0, tok, y.float())
        return out.to(x.dtype)


class DraftLayer(torch.nn.Module):
    def __init__(self, cfg, m):
        super().__init__()
        self.attn_hyper_connection = m.Qwen4ExpTextGatedResidual(cfg)
        self.self_attn = m.Qwen4ExpTextAttention(cfg, 0)
        self.mlp_hyper_connection = m.Qwen4ExpTextGatedResidual(cfg)
        self.mlp = DraftMoe(cfg, m)


class DraftHead(torch.nn.Module):
    """Flash's MTP block (vLLM Qwen3_8FlashNextMultiTokenPredictor): token embedding and trunk state fused
    (fc_embedding + shared fc_hidden after Gemma RMS norms), one full-attention decoder layer, the HC mixer; the trunk's
    LM head on the result. Parameter names are the checkpoint's ``mtp.*`` without the prefix."""

    def __init__(self, text_config):
        super().__init__()
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m

        cfg = copy.deepcopy(text_config)
        full = next(t for t in text_config.layer_types if t != "linear_attention")
        cfg.num_hidden_layers, cfg.layer_types, cfg.ple_layer_ids = 1, [full], []
        self.config = cfg
        width, count = cfg.hidden_size, cfg.hc_count
        self.fc_embedding = torch.nn.Linear(width, width, bias=False)
        self.fc_hidden = torch.nn.Linear(width, width, bias=False)
        self.pre_fc_norm_embedding = m.Qwen4ExpTextRMSNorm(width, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = m.Qwen4ExpTextRMSNorm(count * width, eps=cfg.rms_norm_eps)
        self.layers = torch.nn.ModuleList([DraftLayer(cfg, m)])
        self.hyper_connection_mixer = m.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        self.rotary_emb = m.Qwen4ExpTextRotaryEmbedding(config=cfg)

    def fuse(self, embeds, hc_prev):
        width, count = self.config.hidden_size, self.config.hc_count
        x = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        h = self.pre_fc_norm_hidden(hc_prev)
        h = self.fc_hidden(h.view(*h.shape[:-1], count, width))
        return (x.unsqueeze(-2) + h).flatten(-2)

    def block(self, hidden, attend):
        """The decoder layer and the mixer; ``attend(self_attn, x)`` is the attention's output for rows ``x``."""
        layer = self.layers[0]
        x, hyper, inject = layer.attn_hyper_connection(hidden)
        hidden = hyper + (attend(layer.self_attn, x).unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
        x, hyper, inject = layer.mlp_hyper_connection(hidden)
        hidden = hyper + (layer.mlp(x).unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
        return self.hyper_connection_mixer(hidden)

    def rope(self, positions, like):
        """(cos, sin) for draft rows at trunk positions ``positions`` [3 | 4, 1, n]."""
        if positions.shape[0] == 4:
            positions = positions[1:]
        return self.rotary_emb(like, positions)


def add_lora(head, rank, alpha, targets=TARGETS):
    for name in targets:
        parent, _, attr = name.rpartition(".")
        owner = head.get_submodule(parent) if parent else head
        setattr(owner, attr, LoraLinear(getattr(owner, attr), rank, alpha))
    return head


def lora_params(head):
    return [(n, p) for n, p in head.named_parameters() if p.requires_grad]


# ------------------------------------------------------------------------------------------------ weights
def nvfp4_values(w, block=16):
    """``w`` [..., K] as an NVFP4 copy holds it (fp32 global scale amax / (448 x 6), per ``block`` inputs an FP8
    E4M3 scale, E2M1 values rounded to nearest), in ``w``'s dtype."""
    x = w.float()
    x = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    amax = float(x.abs().max())
    if amax == 0.0:
        return w.clone()
    g = amax / (448.0 * 6.0)
    s = (x.abs().amax(-1, keepdim=True) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    s = torch.where(s == 0, torch.ones_like(s), s)
    y = x / (s * g)
    grid = torch.tensor(E2M1, device=x.device)
    q = grid[torch.bucketize(y.abs(), torch.tensor(E2M1_MIDS, device=x.device))] * torch.sign(y)
    return (q * s * g).reshape(w.shape).to(w.dtype)


def mtp_state(folder):
    """The checkpoint's ``mtp.*`` tensors without the prefix, per-expert routed weights fused as the model's
    [E, 2I, H] gate_up / [E, H, I] down."""
    from safetensors import safe_open

    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    state, experts = {}, {}
    for name in sorted({f for k, f in index.items() if k.startswith("mtp.")}):
        with safe_open(str(folder / name), "pt") as f:
            for key in f.keys():
                if not key.startswith("mtp."):
                    continue
                parts = key[4:].split(".")
                if len(parts) == 7 and parts[2:4] == ["mlp", "experts"] and parts[4].isdigit():
                    experts.setdefault(int(parts[4]), {})[parts[5]] = f.get_tensor(key)
                else:
                    state[key[4:]] = f.get_tensor(key)
    if experts:
        ids = sorted(experts)
        if ids != list(range(len(ids))):
            raise RuntimeError(f"mtp experts: ids {ids[:3]}..{ids[-3:]} not 0..{len(ids) - 1}")
        state["layers.0.mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], 0) for e in ids])
        state["layers.0.mlp.experts.down_proj"] = torch.stack([experts[e]["down_proj"] for e in ids])
    return state


def load(head, state, *, fp4=True):
    """``mtp_state`` into ``head`` before LoRA (strict but rotary buffers); with ``fp4`` the routed experts as their
    NVFP4 values (per expert, gate_up and down each)."""
    missing, unexpected = head.load_state_dict(state, strict=False)
    missing = [k for k in missing if "rotary_emb" not in k]
    if missing or unexpected:
        raise RuntimeError(f"mtp weights: missing {missing[:5]}, unexpected {unexpected[:5]}")
    if fp4:
        round_experts(head)
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def round_experts(head):
    """The routed experts as their NVFP4 values, per expert (gate_up and down each)."""
    with torch.no_grad():
        for p in (head.layers[0].mlp.experts.gate_up_proj, head.layers[0].mlp.experts.down_proj):
            for e in range(p.shape[0]):
                p[e].copy_(nvfp4_values(p[e]))


# ------------------------------------------------------------------------------------------------ attention
class _IndexKeys:
    """qsa_select's cache argument: the prefix's indexer keys joined with the new rows' (kept in ``new``)."""

    def __init__(self, prefix):
        self.prefix, self.new = prefix, None

    def update_indexer(self, raw_keys, layer_idx):
        self.new = raw_keys
        return raw_keys if self.prefix is None else torch.cat([self.prefix.to(raw_keys.dtype), raw_keys], dim=1)


def attend_local(mod, x, rope, past, keep=None):
    """The trainer's selected attention for draft rows ``x`` [1, n, H] after ``past`` ({"k", "v", "idx"} of the rows
    before, or None); ``rope`` (cos, sin) over every row up to the last of ``x``. ``keep``: a list that gets the new
    rows' (k, v, indexer keys)."""
    import rl1_cp
    import rl1_trainer as rt
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    n = x.shape[1]
    prior = 0 if past is None else int(past["k"].shape[2])
    heads, groups, dim = mod.config.num_attention_heads, mod.config.num_key_value_heads, mod.head_dim
    q, gate = torch.chunk(mod.q_proj(x).view(1, n, heads, dim * 2), 2, dim=-1)
    gate = gate.reshape(1, n, heads * dim)
    q = mod.q_norm(q.reshape(1, n, heads, dim)).transpose(1, 2)
    k = mod.k_norm(mod.k_proj(x).view(1, n, groups, dim)).transpose(1, 2)
    v = mod.v_proj(x).view(1, n, groups, dim).transpose(1, 2)
    cos, sin = rope
    q, k = modeling.apply_rotary_pos_emb(q, k, cos[:, -n:], sin[:, -n:])
    index = _IndexKeys(None if past is None else past["idx"])
    counts = torch.arange(prior + 1, prior + n + 1, device=x.device)
    picked = rt.qsa_select(mod.indexer, x, (cos, sin), counts, index)
    kernel, dtype = rl1_cp.attention_kernel()
    dtype = dtype or k.dtype
    keys, values = k.to(dtype), v.to(dtype)
    if past is not None:
        keys = torch.cat([past["k"].to(dtype), keys], dim=2)
        values = torch.cat([past["v"].to(dtype), values], dim=2)
    union = hasattr(kernel, "NOGRAD_PREFIX")
    if union:
        kernel.P0 = prior
    try:
        out = kernel.apply(q, keys, values, picked, mod.scaling,
                           rt.SPARSE["chunk"] if kernel is rl1_cp._KERNEL["torch"] else None)
    finally:
        if union:
            kernel.P0 = None
    if keep is not None:
        keep.append((k.detach(), v.detach(), index.new.detach()))
    out = out.to(q.dtype).transpose(1, 2).reshape(1, n, heads * dim) * torch.sigmoid(gate)
    return mod.o_proj(out)


# ------------------------------------------------------------------------------------------------ the loss
def kl_backward(h, trunk, project, weight_lm, tokens, temperature, weight, chunk=512):
    """Draft rows ``h`` [m, H] (graph) against the trunk's rows ``trunk`` (``project`` maps them to logits; None: they
    are logits), ``tokens`` [m] the sampled next+1 tokens: KL(trunk || draft) summed, backpropagated x ``weight``
    (a chunk at a time into a detached copy of ``h``, then once through the draft). Returns the sums."""
    sums = {"kl": 0.0, "rows": int(h.shape[0]), "top1": 0, "agree": 0, "paccept": 0.0}
    leaf = h.detach().requires_grad_(True)
    w = weight_lm.detach()
    for i in range(0, h.shape[0], chunk):
        j = min(h.shape[0], i + chunk)
        with torch.no_grad():
            p = (trunk[i:j] if project is None else project(trunk[i:j])).float()
            logp = torch.log_softmax(p, dim=-1)
        with torch.enable_grad():
            q = F.linear(leaf[i:j], w.to(leaf.dtype) if not torch.is_autocast_enabled(leaf.device.type) else w).float()
            kl = (logp.exp() * (logp - torch.log_softmax(q, dim=-1))).sum()
            (kl * float(weight)).backward()
        with torch.no_grad():
            top = q.argmax(-1)
            sums["kl"] += float(kl)
            sums["top1"] += int((top == tokens[i:j].to(top.device)).sum())
            sums["agree"] += int((top == p.argmax(-1)).sum())
            t = float(temperature or 1.0)
            sums["paccept"] += float(torch.minimum(torch.softmax(p / t, dim=-1), torch.softmax(q / t, dim=-1)).sum())
        del p, logp, q, kl
    h.backward(leaf.grad if leaf.grad is not None else torch.zeros_like(h))
    return sums


def pair_rows(positions, first_row):
    """Kept trunk rows (trained tokens at ``positions``, in order) whose draft row p - 2 is at or after
    ``first_row``: (their indices, the draft rows' offsets from ``first_row``)."""
    rows = positions - 2 - first_row
    ok = torch.nonzero(rows >= 0).flatten()
    return ok, rows[ok]


# ------------------------------------------------------------------------------------------------ the draft
class Draft:
    """The draft head with its LoRA, the knobs and the window / prefix passes the trainer calls."""

    def __init__(self, head, weight=1.0):
        self.head, self.weight = head, float(weight)

    @classmethod
    def build(cls, text_config, folder, *, rank, alpha, device, weight=1.0, fp4=True, dtype=torch.bfloat16):
        head = DraftHead(text_config)
        load(head, mtp_state(folder), fp4=False)
        head = head.to(device=device)
        for p in head.parameters():  # parameters only: the rotary inv_freq buffers stay fp32
            p.data = p.data.to(dtype)
        if fp4:  # on the card (512 experts)
            round_experts(head)
        add_lora(head, rank, alpha)
        head.lora_rank, head.lora_alpha = rank, alpha
        return cls(head, weight)

    def params(self):
        return [(f"draft.{n}", p) for n, p in lora_params(self.head)]

    def modules(self):
        return self.head.modules()

    def _rows(self, embeds, hc_prev, rope, past, keep=None):
        return self.head.block(self.head.fuse(embeds, hc_prev), lambda mod, x: attend_local(mod, x, rope, past, keep))

    # ---- single GPU ----
    def prefix_chunk(self, state, ids, a, b, hc, positions, embed):
        """Trunk prefix chunk [a, b) with its pre-mixer states ``hc`` [1, b - a, C]: the draft rows [a - 1, b - 1)
        (no grad) on ``state`` (the rows before; None when a = 0). Returns the new state, or None when ``state`` does
        not end at ``a`` (that cache's draft prefix is unknown). ``positions(start, stop)``: trunk position ids."""
        if a > 0 and (state is None or state["at"] != a):
            return None
        hc = hc.detach()
        state = dict(state) if a > 0 else {"at": 0, "k": None, "v": None, "idx": None, "hc_last": None}
        first, last = max(a - 1, 0), b - 1
        if last > first:
            with torch.no_grad():
                hc_prev = hc[:, :b - a - 1] if a == 0 else torch.cat([state["hc_last"].to(hc.dtype), hc[:, :b - a - 1]], 1)
                embeds = embed(ids[:, first + 1:b]).detach()
                rope = self.head.rope(positions(0, last), embeds)
                keep = []
                self._rows(embeds, hc_prev, rope, None if state["k"] is None else state, keep)
                k, v, idx = keep[0]
                if state["k"] is not None:
                    k, v = torch.cat([state["k"], k], dim=2), torch.cat([state["v"], v], dim=2)
                    idx = torch.cat([state["idx"].to(idx.dtype), idx], dim=1)
                state.update(k=k, v=v, idx=idx)
        state.update(at=b, hc_last=hc[:, -1:].clone())
        return state

    def window(self, state, ids, cut, hc_tail, positions, embed, lm_weight, trunk, project, kept, temperature):
        """The window [cut, total) after its trunk forward: draft rows [cut - 1, total - 1) on the prefix ``state``,
        KL against the trunk's kept rows ``trunk`` (trained positions ``kept``), backpropagated into the draft LoRA.
        Returns the sums, or {"skipped": why}."""
        total = ids.shape[1]
        if cut > 0 and (state is None or state["at"] != cut):
            return {"skipped": "prefix"}
        first = max(cut - 1, 0)
        if total - 1 <= first:
            return {"skipped": "short"}
        hc_tail = hc_tail.detach()
        hc_prev = hc_tail[:, :total - cut - 1]
        if cut > 0:
            hc_prev = torch.cat([state["hc_last"].to(hc_prev.dtype), hc_prev], 1)
        with torch.enable_grad():
            embeds = embed(ids[:, first + 1:total]).detach()
            rope = self.head.rope(positions(0, total - 1), embeds)
            h = self._rows(embeds, hc_prev, rope, state if cut > 0 and state["k"] is not None else None)[0]
            ok, rows = pair_rows(kept.to(h.device), first)
            if not len(ok):
                return {"skipped": "no rows"}
            ok = ok.to(trunk.device)
            return kl_backward(h[rows], trunk[ok], project, lm_weight, ids[0, kept.to(ids.device)[ok.to(ids.device)]],
                               temperature, self.weight)

    # ---- sequence parallel (rl1_cp) ----
    def cp_prefix(self, pc, ids, hc, positions, embed):
        """Under the trunk's prefill Ctx ``pc`` (shards of [0, cut)) with this rank's shard states ``hc``: the draft's
        prefix rows [0, cut - 1) as an rl1_cp.Prefix (no grad) and the trunk state at cut - 1 (every rank)."""
        import rl1_cp

        cut = pc.total
        with torch.no_grad():
            hc = hc.detach()
            halo = rl1_cp.Halo.apply(hc, 1, pc, None, None)
            last = last_row(hc, pc)
            dc = rl1_cp.Ctx(_draft_lengths(pc.lengths, 0), position_ids=positions[..., :cut - 1], capture=True,
                            group=pc.group)
            embeds, hc_prev = self._cp_inputs(ids, pc.offset, pc.T, hc, halo, embed)
            self._cp_rows(dc, embeds, hc_prev)
        return rl1_cp.Prefix(cut - 1, None, {0: dict(dc.captured.get(0, {}))}), last

    def cp_window(self, cp, ids, cut, hc, prefix, last, positions, embed, lm_weight, trunk, project, kept,
                  temperature):
        """The tail Ctx ``cp`` (shards of [cut, total)): this rank's draft rows [g0 - 1, g1 - 1) on the draft
        ``prefix`` (from ``cp_prefix``; None when cut = 0), KL against this rank's kept trunk rows (positions
        ``kept``), backpropagated (every rank runs the backward: the attention's K/V gradients are collective)."""
        import rl1_cp

        total = ids.shape[1]
        hc = hc.detach()
        with torch.no_grad():
            halo = rl1_cp.Halo.apply(hc, 1, cp, last if cut > 0 else None, None)
        dc = rl1_cp.Ctx(_draft_lengths(cp.lengths, cut), prefix=prefix, position_ids=positions[..., :total - 1],
                        group=cp.group)
        g0 = cut + cp.offset
        with torch.enable_grad():
            embeds, hc_prev = self._cp_inputs(ids, g0, cp.T, hc, halo, embed)
            h = self._cp_rows(dc, embeds, hc_prev)[0]
            first = max(g0 - 1, 0)
            ok, rows = pair_rows(kept.to(h.device), first)
            if len(ok):
                ok_t = ok.to(trunk.device)
                return kl_backward(h[rows], trunk[ok_t], project, lm_weight,
                                   ids[0, kept.to(ids.device)[ok.to(ids.device)]], temperature, self.weight)
            h.backward(torch.zeros_like(h))
            return {"kl": 0.0, "rows": 0, "top1": 0, "agree": 0, "paccept": 0.0}

    def _cp_inputs(self, ids, g0, n, hc, halo, embed):
        """Draft rows of trunk shard [g0, g0 + n): token ids [g0, g0 + n) (from 1 when g0 = 0), trunk states
        [g0 - 1, g0 + n - 1) (the halo, then the shard's own but its last)."""
        if g0 == 0:
            return embed(ids[:, 1:n]).detach(), hc[:, :max(n - 1, 0)]
        if n == 0:
            return embed(ids[:, :0]).detach(), hc[:, :0]
        return embed(ids[:, g0:g0 + n]).detach(), torch.cat([halo.to(hc.dtype), hc[:, :n - 1]], 1)

    def _cp_rows(self, dc, embeds, hc_prev):
        import rl1_cp

        start = dc.P0 + dc.offset
        with rl1_cp.active(dc):
            cos, sin = self.head.rope(_rows_positions(dc), embeds)
            dc.rope = (cos, sin)
            pe = (cos[:, start:start + dc.T], sin[:, start:start + dc.T])
            return self.head.block(self.head.fuse(embeds, hc_prev),
                                   lambda mod, x: rl1_cp.attention_forward(mod, x, pe, dc))


def _draft_lengths(lengths, start):
    """Draft rows per rank of trunk shards ``lengths`` starting at trunk position ``start``: the first shard of a span
    starting at 0 loses its row -1."""
    out = [int(n) for n in lengths]
    if start == 0:
        for i, n in enumerate(out):
            if n:
                out[i] = n - 1
                break
    return out


def _rows_positions(dc):
    pos = dc.position_ids
    if pos is None:
        return torch.arange(dc.P0 + dc.total).view(1, 1, -1).expand(3, 1, -1)
    return pos[..., :dc.P0 + dc.total]


def last_row(x, cp):
    """The trunk state at the last position of ``cp``'s span (every rank)."""
    buf = x[:, -1:] if cp.T else x.new_zeros((x.shape[0], 1, x.shape[-1]))
    parts = cp.gather_fixed(buf.contiguous())
    return parts[max(i for i, n in enumerate(cp.lengths) if n)]


# ------------------------------------------------------------------------------------------------ export / serving
def vllm_module(name):
    """Checkpoint ``mtp.<name>`` (a TARGETS entry) -> (the vLLM draft model's parameter module, shard or None)."""
    path = "model." + name
    for old, (new, shard) in PACKED.items():
        if path.endswith(old):
            return path[: -len(old)] + new, shard
    return path, None


def export(head, folder):
    """``draft_lora.safetensors`` (PEFT keys ``base_model.model.mtp.<module>.lora_A/B.weight``, BF16) and
    ``draft_lora.json`` in the adapter ``folder``. Returns the json."""
    from safetensors.torch import save_file

    folder = Path(folder)
    tensors, modules = {}, {}
    for name, module in head.named_modules():
        if isinstance(module, LoraLinear):
            for part in ("lora_A", "lora_B"):
                tensors[f"base_model.model.mtp.{name}.{part}.weight"] = (
                    getattr(module, part).detach().to(torch.bfloat16).contiguous().cpu())
            param, shard = vllm_module(name)
            modules[name] = {"vllm": param, "shard": shard}
    meta = {"format": FORMAT, "r": int(head.lora_rank), "lora_alpha": float(head.lora_alpha), "modules": modules}
    save_file(tensors, str(folder / "draft_lora.safetensors"))
    (folder / "draft_lora.json").write_text(json.dumps(meta, indent=1))
    return meta

