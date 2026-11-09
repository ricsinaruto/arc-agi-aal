"""NVFP4 arithmetic of the served routed experts, and GPTQ onto that grid (28 September; expert-quant.md).

The served checkpoint (RadixArk Qwen3.8-Flash-Next-NVFP4, ModelOpt 0.46) stores each routed expert projection as
ModelOpt NVFP4: E2M1 codes two per byte (first in the low nibble, sign in bit 3), one FP8 E4M3 block scale per 16
input features, a float32 ``weight_scale_2`` (gate_proj and up_proj of an expert share it: vLLM fuses them into w13
and keeps gate's) and a float32 activation ``input_scale``. The RTN formulas are ModelOpt's (``posttrain.requant``,
verified bit for bit on Ornith's NVFP4 tensors; ``tests/test_expert_quant.py`` keeps the two in step):

- ``scale_2 = amax(|W|) / (6 * 448)`` over the group (gate and up together);
- block scale ``(block_amax / (6 * scale_2)).to(float8_e4m3fn)``;
- codes ``W / (block_scale * scale_2)`` rounded to E2M1, ties to the even code.

``gptq`` quantizes a batch of experts' weights [E, N, K] onto the same grid with Hessian-aware error compensation
(Frantar et al. 2022): columns in order, each group's block scale chosen from the error-updated weights when the
group starts (ModelOpt's amax rule, optionally a small search), no act-order (groups stay contiguous in K).

torch is imported inside the functions, so the module imports without it.
"""

FP8_MAX = 448.0
FP4_MAX = 6.0
BLOCK = 16
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# --- codes ----------------------------------------------------------------------------------------


def cast_fp4(values):
    """E2M1 codes (uint8, sign in bit 3) of float32 values already divided by their scale; ties to the even
    code; magnitudes above 6 saturate to 6."""
    import torch

    bounds = torch.tensor(E2M1_BOUNDS, device=values.device, dtype=values.dtype)
    odd = torch.tensor([0, 1, 0, 1, 0, 1, 0], device=values.device, dtype=torch.bool)
    sign = (values < 0).to(torch.uint8)
    magnitude = values.abs()
    order = torch.searchsorted(bounds, magnitude.contiguous(), out_int32=True).to(torch.uint8)
    round_up = ((magnitude.unsqueeze(-1) == bounds) & odd).any(-1).to(torch.uint8)
    return sign * 8 + order + round_up


def fp4_values(codes):
    """Float32 values of E2M1 codes (any shape)."""
    import torch

    table = torch.tensor(E2M1_VALUES, device=codes.device)
    codes = codes.long()
    return torch.where(codes >= 8, -1.0, 1.0) * table[codes & 7]


def pack_fp4(codes):
    """[..., K] codes -> [..., K/2] bytes, the first code in the low nibble."""
    pairs = codes.reshape(*codes.shape[:-1], -1, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(codes.dtype)


def unpack_fp4(packed):
    """[..., K/2] bytes -> [..., K] float32 values (no scales)."""
    import torch

    codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(*packed.shape[:-1], -1)
    return fp4_values(codes)


# --- scales, RTN, dequantization ------------------------------------------------------------------


def scale2_of(*weights):
    """ModelOpt's shared ``weight_scale_2`` of [E, N, K] (or [N, K]) weights: per expert, float32."""
    import torch

    amax = torch.stack([w.float().abs().amax(dim=(-2, -1)) for w in weights]).amax(0)
    return amax * (1.0 / (FP4_MAX * FP8_MAX))


def block_scales(blocks, scale_2):
    """FP8 block scales of [..., G, 16] float blocks; ``scale_2`` broadcasts against [..., G]."""
    import torch

    return (blocks.abs().amax(-1) / (FP4_MAX * scale_2)).to(torch.float8_e4m3fn)


def quantize_with_scales(weight, scale, scale_2):
    """E2M1 codes [..., K] of float weights [..., K] with FP8 block scales [..., K/16] and ``scale_2``
    (broadcast over the trailing two dims of the scales)."""
    import torch

    blocks = weight.float().reshape(*weight.shape[:-1], -1, BLOCK)
    divisor = scale.float()
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor) * scale_2
    return cast_fp4(blocks / divisor.unsqueeze(-1)).reshape(weight.shape)


def rtn(weight, scale_2):
    """ModelOpt RTN of [E, N, K] weights with per-expert ``scale_2`` [E]: ``(packed [E, N, K/2] uint8,
    block scales [E, N, K/16] float8)``."""
    s2 = scale_2.float().reshape(-1, 1, 1)
    blocks = weight.float().reshape(*weight.shape[:-1], -1, BLOCK)
    scale = block_scales(blocks, s2)
    return pack_fp4(quantize_with_scales(weight, scale, s2)), scale


def rtn_search(weight, scale_2, multipliers=(1.0, 0.95, 0.9, 0.85)):
    """RTN with a per-block scale search: each 16-block takes, among ModelOpt's amax rule applied to the block
    divided by each multiplier (FP8-rounded), the scale with the smallest squared rounding error.
    ``(packed, scales)`` as ``rtn``; ``multipliers=(1.0,)`` is exactly ``rtn``."""
    import torch

    s2 = scale_2.float().reshape(-1, 1, 1)
    blocks = weight.float().reshape(*weight.shape[:-1], -1, BLOCK)
    best_scale, best_err = None, None
    for m in multipliers:
        cand = block_scales(blocks * (1.0 / m), s2)
        divisor = torch.where(cand.float() == 0, torch.ones_like(cand.float()), cand.float()) * s2
        q = fp4_values(cast_fp4(blocks / divisor.unsqueeze(-1))) * divisor.unsqueeze(-1)
        err = (q - blocks).pow(2).sum(-1)
        if best_scale is None:
            best_scale, best_err = cand, err
        else:
            take = err < best_err
            best_scale = torch.where(take, cand.float(), best_scale.float()).to(torch.float8_e4m3fn)
            best_err = torch.minimum(err, best_err)
    return pack_fp4(quantize_with_scales(weight, best_scale, s2)), best_scale


def dequantize(packed, scale, scale_2):
    """Float32 [E, N, K] weights of packed codes [E, N, K/2], FP8 scales [E, N, K/16] and ``scale_2`` [E]."""
    values = unpack_fp4(packed)
    blocks = values.reshape(*values.shape[:-1], -1, BLOCK)
    blocks = blocks * (scale.float() * scale_2.float().reshape(-1, 1, 1)).unsqueeze(-1)
    return blocks.reshape(values.shape)


# --- GPTQ -----------------------------------------------------------------------------------------


def damp_hessian(hessian, count, *, percdamp=0.01, min_count=1):
    """In place: dead inputs get a unit diagonal; each expert's diagonal gets ``percdamp`` x its mean.
    Returns the per-expert damping added. ``count`` [E] is the (weighted) sample count, used only to
    flag experts with no samples (their Hessian becomes the identity: GPTQ then equals RTN)."""
    import torch

    diag = torch.diagonal(hessian, dim1=-2, dim2=-1)
    empty = count < min_count
    if empty.any():
        hessian[empty] = torch.eye(hessian.shape[-1], device=hessian.device, dtype=hessian.dtype)
    dead = diag <= 0
    diag[dead] = 1.0
    damp = percdamp * diag.mean(-1)
    diag += damp.unsqueeze(-1)
    return damp


def inverse_cholesky_upper(hessian):
    """Upper Cholesky factor of H^-1 for a batch [E, K, K] (float32), as GPTQ uses it. Experts whose
    factorization fails get more damping (up to 5 retries)."""
    import torch

    h = hessian
    for attempt in range(6):
        chol, info = torch.linalg.cholesky_ex(h)
        bad = info != 0
        if not bad.any():
            break
        if attempt == 5:
            raise RuntimeError(f"Cholesky failed for {int(bad.sum())} experts")
        diag = torch.diagonal(h, dim1=-2, dim2=-1)
        h = h.clone()
        torch.diagonal(h, dim1=-2, dim2=-1)[bad] += 0.1 * diag[bad].mean(-1, keepdim=True) * (4 ** attempt)
    inv = torch.cholesky_inverse(chol)
    upper, info = torch.linalg.cholesky_ex(inv, upper=True)
    if (info != 0).any():
        raise RuntimeError("Cholesky of the inverse failed")
    return upper


def gptq(weight, upper, scale_2, *, blocksize=128, scale_search=(1.0,)):
    """GPTQ of [E, N, K] float weights with the upper Cholesky factor of H^-1 [E, K, K] onto NVFP4 with
    per-expert ``scale_2`` [E]. Returns ``(packed [E, N, K/2] uint8, block scales [E, N, K/16] float8)``.

    Each 16-column group's block scale is set when the group starts, from the error-updated weights:
    ModelOpt's amax rule times each multiplier in ``scale_search`` (the FP8-rounded candidate with the
    smallest squared error of that group's plain rounding wins; ``(1.0,)`` is ModelOpt's rule alone).
    """
    import torch

    w = weight.float().clone()
    experts, rows, cols = w.shape
    if cols % BLOCK or blocksize % BLOCK:
        raise ValueError("columns and block size must be multiples of 16")
    s2 = scale_2.float().reshape(experts, 1, 1)
    codes = torch.zeros((experts, rows, cols), dtype=torch.uint8, device=w.device)
    scales = torch.zeros((experts, rows, cols // BLOCK), dtype=torch.float8_e4m3fn, device=w.device)
    multipliers = torch.tensor(scale_search, dtype=torch.float32, device=w.device)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        w1 = w[:, :, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        u1 = upper[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % BLOCK == 0:
                group = w1[:, :, i:i + BLOCK]
                g = col // BLOCK
                if len(scale_search) == 1:
                    sf = block_scales(group.unsqueeze(-2) * (1.0 / multipliers[0]), s2).squeeze(-1)
                else:
                    best, best_err = None, None
                    for m in multipliers:
                        cand = block_scales(group.unsqueeze(-2) * (1.0 / m), s2).squeeze(-1)
                        div = torch.where(cand.float() == 0, torch.ones_like(cand.float()), cand.float()) * s2[..., 0]
                        q = fp4_values(cast_fp4(group / div.unsqueeze(-1))) * div.unsqueeze(-1)
                        e = (q - group).pow(2).sum(-1)
                        if best is None:
                            best, best_err = cand, e
                        else:
                            take = e < best_err
                            best = torch.where(take, cand.float(), best.float()).to(torch.float8_e4m3fn)
                            best_err = torch.minimum(e, best_err)
                    sf = best
                scales[:, :, g] = sf
                divisor = torch.where(sf.float() == 0, torch.ones_like(sf.float()), sf.float()) * s2[..., 0]
            column = w1[:, :, i]
            code = cast_fp4(column / divisor)
            codes[:, :, col] = code
            q = fp4_values(code) * divisor
            d = u1[:, i, i].unsqueeze(-1)
            err = (column - q) / d
            w1[:, :, i:] -= err.unsqueeze(-1) * u1[:, i, i:].unsqueeze(1)
            err1[:, :, i] = err
        w[:, :, i2:] -= torch.bmm(err1, upper[:, i1:i2, i2:])
    return pack_fp4(codes), scales


def pool_hessians(hessian, counts, prior_rows):
    """In place: shrink each expert's Hessian towards the layer's pooled input second moment,
    ``H_e += prior_rows * sum(H) / sum(n)`` (every expert sees the same MoE input distribution, only its routing
    differs). An expert with no calibration rows gets ``prior_rows`` rows of the pooled moment, so it is still
    quantized with a Hessian; one with many rows barely moves. Returns the pooled per-row moment [K, K]."""
    total = counts.sum().clamp(min=1.0)
    pooled = hessian.sum(0) / total
    if prior_rows > 0:
        hessian += prior_rows * pooled
    return pooled


def gptq_static(weight, hessian, scale_2, *, act_order=True, blocksize=128, scales=None):
    """GPTQ with static block scales: every 16-column group's FP8 block scale is fixed up front (ModelOpt's amax rule
    on the original weights, or ``scales``), so the columns may be processed in any order. With ``act_order`` each
    expert's columns go in decreasing Hessian diagonal (its most-used input features first). ``hessian`` [E, K, K]
    must be damped already. Returns ``(packed, block scales)`` as ``gptq``."""
    import torch

    w = weight.float()
    experts, rows, cols = w.shape
    s2 = scale_2.float().reshape(experts, 1, 1)
    if scales is None:
        scales = block_scales(w.reshape(experts, rows, -1, BLOCK), s2)
    divisor = torch.where(scales.float() == 0, torch.ones_like(scales.float()), scales.float()) * s2
    divisor = divisor.repeat_interleave(BLOCK, dim=-1)  # [E, N, K]
    if act_order:
        perm = torch.argsort(torch.diagonal(hessian, dim1=-2, dim2=-1), dim=-1, descending=True)
    else:
        perm = torch.arange(cols, device=w.device).expand(experts, cols)
    index = perm.unsqueeze(1).expand(experts, rows, cols)
    wp = w.gather(2, index).clone()
    dp = divisor.gather(2, index)
    hp = hessian.gather(1, perm.unsqueeze(-1).expand(experts, cols, cols)).gather(
        2, perm.unsqueeze(1).expand(experts, cols, cols))
    upper = inverse_cholesky_upper(hp)
    del hp
    codes_p = torch.zeros((experts, rows, cols), dtype=torch.uint8, device=w.device)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        w1 = wp[:, :, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        u1 = upper[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            column = w1[:, :, i]
            div = dp[:, :, i1 + i]
            code = cast_fp4(column / div)
            codes_p[:, :, i1 + i] = code
            err = (column - fp4_values(code) * div) / u1[:, i, i].unsqueeze(-1)
            w1[:, :, i:] -= err.unsqueeze(-1) * u1[:, i, i:].unsqueeze(1)
            err1[:, :, i] = err
        wp[:, :, i2:] -= torch.bmm(err1, upper[:, i1:i2, i2:])
    codes = torch.zeros_like(codes_p).scatter_(2, index, codes_p)
    return pack_fp4(codes), scales


def remap_pruned_ids(expert_ids, kept):
    """Routing captured on a load-time-pruned server uses the kept experts' ranks (``arc_expert_prune``: kept ids
    ascending, renumbered 0..k-1); map them back to the checkpoint's expert ids."""
    import torch

    table = torch.as_tensor(list(kept), dtype=torch.long, device=expert_ids.device)
    return table[expert_ids.long()].to(expert_ids.dtype)


def routed_rows(expert_ids, weights, experts, num_experts):
    """Token rows routed to each wanted expert: ``(token index, routing weight, bounds)`` sorted by the
    position of the expert in ``experts``; rows of slot i are ``[bounds[i], bounds[i + 1])``."""
    import torch

    device = expert_ids.device
    wanted = torch.as_tensor(list(experts), device=device, dtype=torch.long)
    index = torch.full((num_experts,), -1, dtype=torch.long, device=device)
    index[wanted] = torch.arange(wanted.numel(), device=device)
    flat_ids = expert_ids.reshape(-1).long()
    token = torch.arange(expert_ids.shape[0], device=device).repeat_interleave(expert_ids.shape[1])
    flat_w = weights.reshape(-1).float()
    slot = index[flat_ids]
    keep = slot >= 0
    slot, token, flat_w = slot[keep], token[keep], flat_w[keep]
    order = torch.argsort(slot, stable=True)
    slot, token, flat_w = slot[order], token[order], flat_w[order]
    bounds = torch.searchsorted(slot, torch.arange(wanted.numel() + 1, device=device))
    return token, flat_w, bounds.tolist()


def hessians(x, expert_ids, weights, experts, num_experts, *, chunk=65536):
    """Routing-weighted per-expert Hessians sum_t w_te^2 x_t x_t^T [len(experts), K, K] (float32) and row
    counts from token rows ``x`` [T, K], routed expert ids [T, k] and routing weights [T, k]. The weights
    make the objective the expert's share of the MoE output error."""
    import torch

    token, flat_w, bounds = routed_rows(expert_ids, weights, experts, num_experts)
    k = x.shape[-1]
    out = torch.zeros((len(bounds) - 1, k, k), dtype=torch.float32, device=x.device)
    counts = torch.zeros(len(bounds) - 1, dtype=torch.float32, device=x.device)
    for e in range(len(bounds) - 1):
        a, b = bounds[e], bounds[e + 1]
        for s in range(a, b, chunk):
            t = token[s:min(b, s + chunk)]
            rows = x[t].float() * flat_w[s:min(b, s + chunk)].unsqueeze(-1)
            out[e].addmm_(rows.t(), rows)
        counts[e] = float(b - a)
    return out, counts


def relative_error(a, b):
    """||a - b|| / ||b|| over everything (float)."""
    b = b.float()
    return float((a.float() - b).norm() / b.norm().clamp(min=1e-30))


def cosine(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))
