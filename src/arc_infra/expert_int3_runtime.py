"""ARC 3-bit routed experts at load time (30 September; docs/validation-20260930/expert-3bit.md).

Copied into the pinned runtime as ``vllm/arc_expert_int3.py`` by ``arc_infra.expert_int3_patch``. The routed experts
arrive as ordinary ModelOpt NVFP4 tensors whose E2M1 values all lie on the int3 grid (``arena.expert_3bit_math``:
the integers -4..4 times a per-group scale, each group using -4..3 or -3..4). On vLLM's Humming MoE backend, right
after the checkpoint's NVFP4 tensors are converted to Humming's standard layout, ``repack`` re-encodes them as
Humming ``uint3`` (int3 = code - 4) with the same scales, bit-exact:

- a group that holds +4 is negated (codes and scale), so every group fits -4..3;
- ``group`` > 16 keeps one per-16 scale of each group (they must be equal: the overlay wrote them so).

Anything that is not an exact int3 container raises: a wrong overlay cannot be served silently at the wrong precision.
Settings come from ``ARC_EXPERT_INT3`` (``{"group": 16}``, JSON); unset means the patch does nothing.
"""

import json
import os

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def settings():
    raw = os.environ.get("ARC_EXPERT_INT3")
    if not raw:
        return None
    value = json.loads(raw)
    group = int(value.get("group", 16))
    if group not in (16, 32, 64, 128):
        raise ValueError(f"ARC_EXPERT_INT3 group {group} not in 16/32/64/128")
    return {"group": group}


def unpack_e2m1(packed):
    """[..., K/2] uint8 (first value in the low nibble, sign in bit 3) -> [..., K] float32 values."""
    import torch

    table = torch.tensor(E2M1, device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(*packed.shape[:-1], -1).long()
    return torch.where(codes >= 8, -1.0, 1.0) * table[codes & 7]


def int3_codes(values, scales16, group):
    """(uint3 codes int32 [..., K], signed group scales [..., K/group]) of int3-grid values [..., K] and their
    per-16 scales [..., K/16] (any float dtype)."""
    import torch

    if not torch.equal(values, values.round()) or float(values.abs().max()) > 4:
        raise ValueError("routed experts are not on the int3 grid (non-integer or |v| > 4 E2M1 values)")
    k = values.shape[-1]
    g = values.reshape(*values.shape[:-1], k // group, group)
    has_pos4 = (g == 4).any(-1)
    has_neg4 = (g == -4).any(-1)
    if bool((has_pos4 & has_neg4).any()):
        raise ValueError("an int3 group holds both +4 and -4")
    per = group // 16
    s = scales16.reshape(*scales16.shape[:-1], k // group, per)
    if per > 1 and not bool((s == s[..., :1]).all()):
        raise ValueError(f"per-16 scales differ inside a {group}-group")
    sign = torch.where(has_pos4, -1.0, 1.0)
    codes = (g * sign.unsqueeze(-1) + 4).to(torch.int32).reshape(values.shape)
    scale = (s[..., 0].float() * sign).to(scales16.dtype)
    return codes, scale


def repack(schema, tensors, num_experts):
    """Humming standard-format NVFP4 (schema, tensors) -> the uint3 (schema, tensors); others pass through."""
    import dataclasses

    import torch

    cfg = settings()
    if cfg is None or not num_experts:
        return schema, tensors
    from humming import dtypes, ops

    if schema.b_dtype != dtypes.float4e2m1 or schema.weight_scale_group_size != 16:
        return schema, tensors
    group = cfg["group"]
    weight = tensors["weight"].contiguous().view(torch.uint8)
    scales16 = tensors["weight_scale"]
    out = dict(tensors)
    packed, scales = [], []
    for s in range(0, weight.shape[0], 32):
        values = unpack_e2m1(weight[s:s + 32])
        codes, scale = int3_codes(values, scales16[s:s + 32], group)
        packed.append(ops.pack_weight(codes.contiguous(), 3))
        scales.append(scale)
        del values, codes
    out["weight"] = torch.cat(packed)
    out["weight_scale"] = torch.cat(scales).contiguous()
    new_schema = dataclasses.replace(schema, b_dtype=dtypes.uint3, weight_scale_group_size=group)
    return new_schema, out
