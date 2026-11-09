"""3-bit routed-expert grids that live inside the served NVFP4 container (30 September; expert-3bit.md).

A 3-bit expert format here is a set of eight (or seven) code values times a signed FP8 E4M3 group scale times the
checkpoint-style per-expert float32 ``scale_2``. Every value set used is a subset of E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6
and their negatives), and a negative group scale is folded into the codes' signs, so each 3-bit expert is *also* an
exact ModelOpt NVFP4 tensor with the checkpoint's names, shapes and dtypes:

- quality can be measured on the served Marlin NVFP4 path unchanged (the GPTQ-ARC overlay machinery);
- a fast kernel (Humming W3A16, grouped MoE) can repack the same codes to 3 bits at load, bit-exact.

Groups of 32 or 64 inputs are stored as 2 or 4 equal per-16 block scales in the container.

Formats (``FORMATS``): ``int3`` = the integers -4..3 (Humming ``int3``, symmetric: uint3 - 4); ``fp3`` = 0, ±1, ±2, ±4
(Humming ``float3e2m0``, seven values). The signed scale puts the group's largest magnitude on the long side (-4).

torch is imported inside the functions, so the module imports without it.
"""

FP8_MAX = 448.0
BLOCK = 16
FORMATS = {
    "int3": (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    "fp3": (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0),
}
# Bits per weight in the fast (repacked) layout: 3 code bits + one E4M3 scale per group (+ the per-expert float32).
def bits_per_weight(group):
    return 3.0 + 8.0 / group


def levels_tensor(fmt, device=None):
    import torch

    return torch.tensor(FORMATS[fmt], dtype=torch.float32, device=device)


def nearest(values, levels):
    """Nearest level (float) of float values (any shape) on a sorted level set; ties go to the lower level."""
    import torch

    mids = (levels[1:] + levels[:-1]) * 0.5
    idx = torch.searchsorted(mids, values.contiguous(), right=True)
    return levels[idx]


def scale2_of(weight, fmt):
    """Per-expert float32 global scale of [E, N, K] weights: amax / (|lowest level| * 448), ModelOpt's rule with the
    grid's own maximum."""
    lmax = abs(FORMATS[fmt][0])
    return weight.float().abs().amax(dim=(-2, -1)) * (1.0 / (lmax * FP8_MAX))


def group_scales(weight, scale_2, fmt, group, multipliers=(1.0,)):
    """Signed E4M3 group scales [E, N, K/group] (as float32 values, already E4M3-rounded) of [E, N, K] weights.

    The sign makes the group's largest-magnitude weight map to the grid's most negative level; each group takes the
    candidate (amax rule divided by a multiplier) with the smallest squared rounding error."""
    import torch

    levels = levels_tensor(fmt, weight.device)
    lmax = -float(levels[0])
    e, n, k = weight.shape
    g = weight.float().reshape(e, n, k // group, group)
    s2 = scale_2.float().reshape(e, 1, 1)
    idx = g.abs().argmax(-1, keepdim=True)
    top = g.gather(-1, idx).squeeze(-1)
    sign = torch.where(top > 0, -1.0, 1.0)
    amax = top.abs()
    best, best_err = None, None
    for m in multipliers:
        mag = (amax / (lmax * m) / s2).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float()
        sc = mag * sign
        div = torch.where(mag == 0, torch.ones_like(mag), sc) * s2
        q = nearest(g / div.unsqueeze(-1), levels) * div.unsqueeze(-1)
        err = (q - g).pow(2).sum(-1)
        if best is None:
            best, best_err = sc, err
        else:
            take = err < best_err
            best = torch.where(take, sc, best)
            best_err = torch.minimum(err, best_err)
    return best


def divisor(scales, scale_2, group):
    """Per-weight signed divisor [E, N, K] = group scale x scale_2 (zero scales -> scale_2, their weights are 0)."""
    import torch

    s2 = scale_2.float().reshape(-1, 1, 1)
    d = torch.where(scales == 0, torch.ones_like(scales), scales) * s2
    return d.repeat_interleave(group, dim=-1)


def rtn(weight, scale_2, fmt, group, multipliers=(1.0,)):
    """Dequantized float32 [E, N, K] weights and the signed group scales of a round-to-nearest 3-bit quantization."""
    levels = levels_tensor(fmt, weight.device)
    scales = group_scales(weight, scale_2, fmt, group, multipliers)
    div = divisor(scales, scale_2, group)
    return nearest(weight.float() / div, levels) * div, scales


def gptq_static(weight, hessian, scale_2, fmt, group, scales, *, act_order=True, blocksize=128):
    """GPTQ onto the 3-bit grid with fixed (signed) group scales, optionally in act-order; ``hessian`` [E, K, K] must
    be damped. Returns the dequantized float32 weights [E, N, K] (their codes are ``value / divisor``)."""
    import torch

    from arena.expert_quant_math import inverse_cholesky_upper

    levels = levels_tensor(fmt, weight.device)
    w = weight.float()
    experts, rows, cols = w.shape
    div = divisor(scales, scale_2, group)
    if act_order:
        perm = torch.argsort(torch.diagonal(hessian, dim1=-2, dim2=-1), dim=-1, descending=True)
    else:
        perm = torch.arange(cols, device=w.device).expand(experts, cols)
    index = perm.unsqueeze(1).expand(experts, rows, cols)
    wp = w.gather(2, index).clone()
    dp = div.gather(2, index)
    hp = hessian.gather(1, perm.unsqueeze(-1).expand(experts, cols, cols)).gather(
        2, perm.unsqueeze(1).expand(experts, cols, cols))
    upper = inverse_cholesky_upper(hp)
    del hp
    qp = torch.zeros_like(wp)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        w1 = wp[:, :, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        u1 = upper[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            column = w1[:, :, i]
            d = dp[:, :, i1 + i]
            q = nearest(column / d, levels) * d
            qp[:, :, i1 + i] = q
            err = (column - q) / u1[:, i, i].unsqueeze(-1)
            w1[:, :, i:] -= err.unsqueeze(-1) * u1[:, i, i:].unsqueeze(1)
            err1[:, :, i] = err
        wp[:, :, i2:] -= torch.bmm(err1, upper[:, i1:i2, i2:])
    return torch.zeros_like(qp).scatter_(2, index, qp)


def to_nvfp4(dequant, scales, scale_2, group):
    """The NVFP4 container of a 3-bit expert: ``(packed E2M1 [E, N, K/2] uint8, E4M3 block scales [E, N, K/16])``
    with the checkpoint's ``scale_2`` unchanged. Exact: codes are value / |divisor|, all in E2M1."""
    import torch

    from arena.expert_quant_math import cast_fp4, fp4_values, pack_fp4

    mag = scales.abs()
    div = divisor(mag, scale_2, group)
    codes = cast_fp4(dequant.float() / div)
    back = fp4_values(codes) * div
    if not torch.equal(back, dequant.float()):
        bad = float((back != dequant.float()).float().mean())
        raise ValueError(f"3-bit values are not exact in the NVFP4 container ({bad:.2e} differ)")
    block = mag.repeat_interleave(group // BLOCK, dim=-1).to(torch.float8_e4m3fn)
    return pack_fp4(codes), block
