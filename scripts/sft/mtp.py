"""Flash's MTP draft head on transformers modules (from heal_kd): the head, its weights from a checkpoint folder, its
inputs over a main span, and the hook that keeps the main model's pre-mixer state."""
import json
from pathlib import Path

import torch

import attention


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
        if getattr(text_config, "mtp_moe_intermediate_size", None):  # a channel-cut student keeps the MTP experts' width
            cfg.moe_intermediate_size = text_config.mtp_moe_intermediate_size
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
        if attention.selected_on() and not self.config.attention_dropout:  # counts only, no [n, past + n] mask
            mask = attention.causal_marker(embeds.shape[1], past + embeds.shape[1], embeds.device)
        else:
            mask = m.create_causal_mask(config=self.config, inputs_embeds=embeds,
                                        attention_mask=torch.ones(1, past + embeds.shape[1], device=embeds.device,
                                                                  dtype=torch.long),
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


def head_for(text_config, folder):
    """The MtpHead that a checkpoint's ``mtp.*`` tensors fit. A pruned student keeps Flash's uncut MTP block (512
    experts, top-10: ``mtp_num_experts*`` in its config) beside main layers with fewer experts (26 Oct smoke v3)."""
    from safetensors import safe_open

    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    key = "mtp.layers.0.mlp.gate.weight"
    if key not in index:
        return MtpHead(text_config)
    with safe_open(str(folder / index[key]), "pt") as f:
        experts = f.get_slice(key).get_shape()[0]
    if experts == text_config.num_experts:
        return MtpHead(text_config)
    top_k = getattr(text_config, "mtp_num_experts_per_tok", None) or text_config.num_experts_per_tok
    return MtpHead(text_config, experts=experts, top_k=top_k)


def load(head, folder, device="cpu", dtype=torch.bfloat16):
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


def inputs(ids, start, stop, piece=None):
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
