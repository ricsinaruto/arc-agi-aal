"""Merge a LoRA adapter into Ornith-1.5-35B-A3B's BF16 weights and re-quantize exactly the adapted
tensors to the served ModelOpt 0.45 MIXED_PRECISION encoding (``docs/posttrain/lora-test-plan.md``,
section 3, path A).

The formulas reproduce the shipped ``ornith-ai/Ornith-1.5-35B-A3B-NVFP4`` tensors from the BF16 revision
bit for bit (checked on 10 tensors by HTTP range reads, 26 September: q/k/v/o_proj of layer 3,
in_proj_qkv/in_proj_z/out_proj of layer 0, the layer-0 shared expert, then every targeted tensor of
layers 2, 38 and 39 through ``requantize``; the merge job checks all 250 modules):

- FP8 per tensor (attention q/k/v/o, Gated-DeltaNet in_proj_qkv/in_proj_z/out_proj):
  ``scale = amax(|W|) * (1/448)`` in float32 (own tensor; q, k and v are not grouped), and
  ``q = (W_bf16 / scale.to(bf16)).to(float8_e4m3fn)`` (a BF16 division, then the FP8 cast).
  ``input_scale`` (activation calibration) is kept from the served checkpoint.
- W4A16 NVFP4 block 16 (shared expert gate/up/down): ``scale_2 = amax * (1/2688)`` (2688 = 6 * 448) where the amax is
  shared by gate_proj and up_proj of a layer (vLLM fuses them; the shipped up_proj carries gate's
  scale_2) and down_proj's own; block scales ``(block_amax / (6 * scale_2)).to(float8_e4m3fn)``; codes
  are ``W / (block_scale * scale_2)`` rounded to E2M1 with ModelOpt's tie rule (a value on a boundary
  goes to the even code) and a sign bit for negative values (also for -0 codes); two codes per byte,
  the first in the low nibble.

torch is imported inside the functions, so the module (names, regexes, adapter-key parsing) imports
without it.
"""

import re

TARGET_REGEX = (r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z"
                r"|out_proj)|mlp\.shared_expert\.(gate|up|down)_proj)")
FP8_MAX = 448.0
FP4_MAX = 6.0
BLOCK = 16
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
ADAPTER_KEY = re.compile(r"(?:^|\.)(model\.language_model\.layers\.\d+\..+?)\.lora_([AB])(?:\.[^.]+)?\.weight$")


def target_modules(weight_map):
    """Adapted module names (no ``.weight``), sorted by layer then name."""
    pattern = re.compile(TARGET_REGEX + r"\.weight$")
    modules = [name[:-len(".weight")] for name in weight_map if pattern.fullmatch(name)]
    return sorted(modules, key=lambda m: (int(m.split(".")[3]), m))


def quant_formats(quant_config, modules):
    """``{module: "FP8" | "W4A16_NVFP4"}`` from ``hf_quant_config.json``; every module must be listed."""
    layers = quant_config["quantization"]["quantized_layers"]
    out = {}
    for module in modules:
        algo = (layers.get(module) or {}).get("quant_algo")
        if algo not in ("FP8", "W4A16_NVFP4"):
            raise ValueError(f"{module}: served quantization {algo!r} is not FP8 or W4A16_NVFP4")
        if algo == "W4A16_NVFP4" and layers[module].get("group_size") != BLOCK:
            raise ValueError(f"{module}: NVFP4 group size {layers[module].get('group_size')}")
        out[module] = algo
    return out


def scale2_group(module):
    """NVFP4 modules sharing one ``weight_scale_2``: gate_proj and up_proj of a shared expert."""
    if module.endswith((".shared_expert.gate_proj", ".shared_expert.up_proj")):
        return module.rsplit(".", 1)[0] + ".gate_up"
    return module


def served_tensor_names(module, fmt):
    """The served tensors re-quantization replaces (``input_scale`` stays)."""
    names = [f"{module}.weight", f"{module}.weight_scale"]
    return names + [f"{module}.weight_scale_2"] if fmt == "W4A16_NVFP4" else names


def adapter_pairs(state_dict_keys):
    """``{module: {"A": key, "B": key}}`` from PEFT adapter keys; each module must have both."""
    pairs = {}
    for key in state_dict_keys:
        match = ADAPTER_KEY.search(key)
        if not match:
            raise ValueError(f"unexpected adapter key {key}")
        pairs.setdefault(match.group(1), {})[match.group(2)] = key
    incomplete = sorted(m for m, p in pairs.items() if set(p) != {"A", "B"})
    if incomplete:
        raise ValueError(f"adapter modules without both lora_A and lora_B: {incomplete[:5]}")
    return pairs


# --- torch ----------------------------------------------------------------------------------------


def merge(weight, lora_a, lora_b, scaling):
    """``W + scaling * B @ A`` in float32, rounded to BF16."""
    import torch

    delta = lora_b.float() @ lora_a.float()
    return (weight.float() + scaling * delta).to(torch.bfloat16)


def fp8_scale(weight):
    return weight.float().abs().amax() * (1.0 / FP8_MAX)


def fp8_quantize(weight, scale=None):
    """``(q, scale)``: FP8 E4M3 per-tensor codes of a BF16 weight and the float32 0-dim scale."""
    import torch

    weight = weight.to(torch.bfloat16)
    scale = fp8_scale(weight) if scale is None else scale.float().reshape(())
    return (weight / scale.to(torch.bfloat16)).to(torch.float8_e4m3fn), scale


def nvfp4_scale2(weights):
    """The shared ``weight_scale_2`` of the BF16 weights of one group."""
    import torch

    amax = torch.stack([w.float().abs().amax() for w in weights]).amax()
    return amax * (1.0 / (FP4_MAX * FP8_MAX))


def cast_fp4(values):
    """E2M1 codes (uint8, sign in bit 3) of float32 values, ties to the even code (ModelOpt)."""
    import torch

    bounds = torch.tensor(E2M1_BOUNDS, device=values.device, dtype=values.dtype)
    odd = torch.tensor([0, 1, 0, 1, 0, 1, 0], device=values.device, dtype=torch.bool)
    sign = (values < 0).to(torch.uint8)
    magnitude = values.abs()
    order = torch.searchsorted(bounds, magnitude.contiguous(), out_int32=True).to(torch.uint8)
    round_up = ((magnitude.unsqueeze(-1) == bounds) & odd).any(-1).to(torch.uint8)
    return sign * 8 + order + round_up


def pack_fp4(codes):
    codes = codes.reshape(codes.shape[0], -1, 2)
    return (codes[..., 0] | (codes[..., 1] << 4)).to(codes.dtype)


def unpack_fp4(packed):
    """Float32 values of packed E2M1 codes (no scales)."""
    import torch

    values = torch.tensor(E2M1_VALUES, device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    return torch.where(codes >= 8, -1.0, 1.0) * values[codes & 7]


def nvfp4_quantize(weight, scale_2, *, rows=1024):
    """``(packed uint8 [out, in/2], block scales float8 [out, in/16])`` of a BF16 weight."""
    import torch

    out_features, in_features = weight.shape
    if in_features % BLOCK:
        raise ValueError(f"input features {in_features} are not a multiple of {BLOCK}")
    scale_2 = scale_2.float().reshape(())
    packed, scales = [], []
    for start in range(0, out_features, rows):
        blocks = weight[start:start + rows].float().reshape(-1, in_features // BLOCK, BLOCK)
        block_scale = (blocks.abs().amax(-1) / (FP4_MAX * scale_2)).to(torch.float8_e4m3fn)
        divisor = block_scale.float()
        divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor) * scale_2
        codes = cast_fp4(blocks / divisor.unsqueeze(-1)).reshape(blocks.shape[0], in_features)
        packed.append(pack_fp4(codes))
        scales.append(block_scale)
    return torch.cat(packed), torch.cat(scales)


def dequantize_nvfp4(packed, block_scale, scale_2):
    values = unpack_fp4(packed)
    blocks = values.reshape(values.shape[0], -1, BLOCK) * (block_scale.float() * scale_2.float()).unsqueeze(-1)
    return blocks.reshape(values.shape)


def requantize(weights, formats):
    """Served tensors of BF16 ``weights`` (``{module: tensor}``): ``{tensor name: torch tensor}``
    (``weight``, ``weight_scale`` and, for NVFP4, ``weight_scale_2``; scales 0-dim float32)."""
    groups = {}
    for module in weights:
        if formats[module] == "W4A16_NVFP4":
            groups.setdefault(scale2_group(module), []).append(module)
    out = {}
    for module, weight in weights.items():
        if formats[module] == "FP8":
            q, scale = fp8_quantize(weight)
            out[f"{module}.weight"], out[f"{module}.weight_scale"] = q, scale
        else:
            members = groups[scale2_group(module)]
            if len(members) != (2 if scale2_group(module) != module else 1):
                raise ValueError(f"{module}: scale_2 group {members} is incomplete")
            scale_2 = nvfp4_scale2([weights[m] for m in members])
            packed, block_scale = nvfp4_quantize(weight, scale_2)
            out[f"{module}.weight"], out[f"{module}.weight_scale"] = packed, block_scale
            out[f"{module}.weight_scale_2"] = scale_2
    return out


SAFETENSORS_DTYPES = {"torch.bfloat16": "BF16", "torch.float8_e4m3fn": "F8_E4M3", "torch.uint8": "U8",
                      "torch.float32": "F32"}


def tensor_record(tensor):
    """``(safetensors dtype, shape, bytes)`` of a CPU tensor (a 0-dim scale stays 0-dim)."""
    import torch

    tensor = tensor.detach().to("cpu").contiguous()
    dtype = SAFETENSORS_DTYPES[str(tensor.dtype)]
    data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dtype, list(tensor.shape), data
