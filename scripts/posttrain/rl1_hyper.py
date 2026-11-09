"""MFU stage H (MFU-1, 22 Oct): Flash's hyper-connections (hc_count 4 residual streams of hidden 2,560) as fused
autograd functions. rl1-prof-m1a (E+G+chunked head, 372 s step): the two Qwen4ExpTextGatedResidual per layer and the
injection adds ``X + y (x) w`` are ~25% of the card time, nearly all elementwise: the grouped RMSNorm in fp32 over the
10,240-wide stream, sigmoid/mix/mean over [T, 4, H] intermediates, and in the backward MulBackward over broadcast
[T, 4, H] products, Pow/Mean backward and the fp32<->bf16 cast backward.

The stream is fp32 in the trainer from the first PLE layer on (PLE's fp32 gate promotes it; rl1-prof-m1y: 8142 of 8496
layer calls) and BF16 before it; both run here with that dtype's rounding points (autocast: the Linear calls see BF16).
Per GatedResidual (frozen weights: no weight gradients):
- forward: one Triton kernel for the grouped RMSNorm (xn and 1/rms; the sum of squares in torch's reduce order: per
  lane four vector-slot accumulators over 128-wide steps, slots summed in order, then the 32-lane shuffle tree from offset 16 down; libdevice
  rsqrt); the model's own F.linear calls and small elementwise ops (same calls, same bits); one kernel for
  mean_g(sigmoid(u_g) xn_g);
- backward: one kernel for dU (mix and sigmoid backward), the model's three GEMMs and silu/sigmoid backward ops, one
  kernel for dxn = (d_inject + d_mix) + d_down (autograd's accumulation order) and the RMSNorm backward with autograd's
  op order (rsqrt backward -0.5 g r^3, mean backward by the reciprocal, pow backward g (2x)).
Injection ``X' = X + y w``: forward one kernel; backward dy (4-stream sum) and dw (torch's reduce order over H).

Forward kernels run without FMA contraction. Switch: ``HYPER["on"]`` (trainer cmd ``hyper_fast``); anything else (other dtypes, non-plain Linear, widths off the reduce layout) runs the model's code."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

HYPER = {"on": False, "calls": 0, "fallback": 0, "why": {}, "dtypes": {}}
BF16 = torch.bfloat16
SIGMOID_LIB = True  # torch's sigmoid: 1 / (1 + expf(-u))
VEC, WVEC = 4, 8  # torch's reduce vector width (16 B): fp32 / BF16 sums
NOFMA = {"enable_fp_fusion": False}
TUNE = {"rows": 2, "warps": 2}


# ------------------------------------------------------------------------------------------------ Triton kernels
@triton.jit
def _mul(a, b):
    """fp32 a * b rounded on its own: an explicit mul.rn, which ptxas never contracts into an FMA with a following add
    (sm_100+ fuses a plain multiply-add even with enable_fp_fusion=False: rl1_hyper_test dsum on B300)."""
    return tl.inline_asm_elementwise("mul.rn.f32;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _sigmoid(u, LIB: tl.constexpr):
    """fp32 sigmoid: 1 / (1 + expf(-u)) with libdevice's exp and an IEEE division (LIB), or Triton's."""
    if LIB:
        return tl.math.div_rn(1.0, 1.0 + libdevice.exp(-u))
    return tl.sigmoid(u)


@triton.jit
def _tree(acc, V: tl.constexpr):
    """torch's inner-dim reduce of one row: acc [32 lanes, V slots] (each slot summed over the row's steps in order);
    per lane the slots in order, then the warp shuffle tree (offsets 16, 8, 4, 2, 1; rl1_hyper_test dnorm)."""
    cols = tl.arange(0, V)
    v = tl.sum(tl.where(cols[None, :] == 0, acc, 0.0), axis=1)
    for i in tl.static_range(1, V):
        v = v + tl.sum(tl.where(cols[None, :] == i, acc, 0.0), axis=1)
    a, b = tl.split(tl.permute(tl.reshape(v, [2, 16]), (1, 0)))  # lane k + lane k + 16
    a, b = tl.split(tl.permute(tl.reshape(a + b, [2, 8]), (1, 0)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [2, 4]), (1, 0)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [2, 2]), (1, 0)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [2, 1]), (1, 0)))
    return tl.sum(a + b, axis=0)


@triton.jit
def _tree_rows(acc, R: tl.constexpr, V: tl.constexpr):
    """_tree for R independent rows at once: acc [R, 32 lanes, V slots] -> [R] (same per-row order)."""
    cols = tl.arange(0, V)
    v = tl.sum(tl.where(cols[None, None, :] == 0, acc, 0.0), axis=2)
    for i in tl.static_range(1, V):
        v = v + tl.sum(tl.where(cols[None, None, :] == i, acc, 0.0), axis=2)
    a, b = tl.split(tl.permute(tl.reshape(v, [R, 2, 16]), (0, 2, 1)))  # lane k + lane k + 16
    a, b = tl.split(tl.permute(tl.reshape(a + b, [R, 2, 8]), (0, 2, 1)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [R, 2, 4]), (0, 2, 1)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [R, 2, 2]), (0, 2, 1)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [R, 2, 1]), (0, 2, 1)))
    return tl.sum(a + b, axis=1)


@triton.jit
def _norm_fwd_rows(X, NW, XN, RS, XB, M, H, eps, inv_h, G: tl.constexpr, NJ: tl.constexpr, V: tl.constexpr,
                   R: tl.constexpr, CAST: tl.constexpr):
    """Grouped RMSNorm of R consecutive (row, stream) groups of the flat [M = N G, H] stream per program:
    xn = ((x * r) * (1 + w)) in XN's dtype, r = rsqrt(mean(x^2) + eps); CAST: also xn rounded to BF16 (round to
    nearest even, as torch's cast) into XB."""
    i = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    mi = i < M
    off = tl.arange(0, 32)[:, None] * V + tl.arange(0, V)[None, :]
    base = i[:, None, None] * H + off[None, :, :]
    wbase = (i % G)[:, None, None] * H + off[None, :, :]
    m = mi[:, None, None]
    acc = tl.zeros([R, 32, V], dtype=tl.float32)
    for j in range(NJ):  # a partial last step (H not a multiple of 32 V) adds exact zeros on the idle lanes
        mj = m & (off[None, :, :] + j * 32 * V < H)
        x = tl.load(X + base + j * 32 * V, mask=mj, other=0.).to(tl.float32)
        acc += _mul(x, x)
    r = libdevice.rsqrt(_mul(_tree_rows(acc, R, V), inv_h) + eps)
    r3 = r[:, None, None]
    for j in range(NJ):
        mj = m & (off[None, :, :] + j * 32 * V < H)
        x = tl.load(X + base + j * 32 * V, mask=mj, other=0.).to(tl.float32)
        w = tl.load(NW + wbase + j * 32 * V, mask=mj, other=0.).to(tl.float32)
        xn = ((x * r3) * (1.0 + w)).to(XN.dtype.element_ty)
        tl.store(XN + base + j * 32 * V, xn, mask=mj)
        if CAST:
            tl.store(XB + base + j * 32 * V, xn.to(tl.bfloat16), mask=mj)
    tl.store(RS + i, r, mask=mi)


@triton.jit
def _norm_bwd_rows(DMIX, U, GI, GD, X, NW, RS, DX, M, H, inv_h, G: tl.constexpr, NJ: tl.constexpr,
                   V: tl.constexpr, LIB: tl.constexpr, ROUND: tl.constexpr, R: tl.constexpr):
    """Grouped RMSNorm backward of R consecutive (row, stream) groups per program in autograd's op order:
    dx = dn r + (((-0.5 dr) r^3) / H) (2 x), dr = sum_h dn x in torch's reduce order."""
    i = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    mi = i < M
    off = tl.arange(0, 32)[:, None] * V + tl.arange(0, V)[None, :]
    base = i[:, None, None] * H + off[None, :, :]
    wbase = (i % G)[:, None, None] * H + off[None, :, :]
    dbase = (i // G)[:, None, None] * H + off[None, :, :]
    m = mi[:, None, None]
    r = tl.load(RS + i, mask=mi, other=1.0)
    r3 = r[:, None, None]
    acc = tl.zeros([R, 32, V], dtype=tl.float32)
    for j in range(NJ):
        mj = m & (off[None, :, :] + j * 32 * V < H)
        _, dn = _dxn_rows(DMIX, U, GI, GD, NW, base, wbase, dbase, j, mj, G, V, LIB, ROUND)
        acc += _mul(dn, tl.load(X + base + j * 32 * V, mask=mj, other=0.).to(tl.float32))
    dv = (((-0.5 * _tree_rows(acc, R, V)) * ((r * r) * r)) * inv_h)[:, None, None]
    for j in range(NJ):
        mj = m & (off[None, :, :] + j * 32 * V < H)
        _, dn = _dxn_rows(DMIX, U, GI, GD, NW, base, wbase, dbase, j, mj, G, V, LIB, ROUND)
        x = tl.load(X + base + j * 32 * V, mask=mj, other=0.).to(tl.float32)
        tl.store(DX + base + j * 32 * V, (_mul(dn, r3) + _mul(dv, 2.0 * x)).to(DX.dtype.element_ty), mask=mj)


@triton.jit
def _dxn_rows(DMIX, U, GI, GD, NW, base, wbase, dbase, j, m, G: tl.constexpr, V: tl.constexpr, LIB: tl.constexpr,
              ROUND: tl.constexpr):
    """(dxn, dn = dxn * (1 + w)) for an [R, 32, V] chunk, dxn = (d_inject + d_mix) + d_down (summed in BF16 steps
    when ROUND)."""
    k = j * 32 * V
    dp = tl.load(DMIX + dbase + k, mask=m, other=0.).to(tl.float32) * (1.0 / G)
    s = _sigmoid(tl.load(U + base + k, mask=m, other=0.).to(tl.float32), LIB).to(tl.bfloat16).to(tl.float32)
    gm = _mul(dp, s)
    if ROUND:
        gm = gm.to(tl.bfloat16).to(tl.float32)
    d = tl.load(GI + base + k, mask=m, other=0.).to(tl.float32) + gm
    if ROUND:
        d = d.to(tl.bfloat16).to(tl.float32)
    d = d + tl.load(GD + base + k, mask=m, other=0.).to(tl.float32)
    if ROUND:
        d = d.to(tl.bfloat16).to(tl.float32)
    return d, d * (1.0 + tl.load(NW + wbase + k, mask=m, other=0.).to(tl.float32))


@triton.jit
def _mix_fwd(U, XN, MIX, H, G: tl.constexpr, BH: tl.constexpr, LIB: tl.constexpr, ROUND: tl.constexpr):
    """mixed = mean_g(BF16(sigmoid(u_g)) * xn_g): products rounded to BF16 when xn is (ROUND), sum in order, x 1/G."""
    row, j = tl.program_id(0).to(tl.int64), tl.program_id(1)
    h = j * BH + tl.arange(0, BH)
    m = h < H
    acc = tl.zeros([BH], dtype=tl.float32)
    for g in tl.static_range(G):
        off = row * G * H + g * H + h
        s = _sigmoid(tl.load(U + off, mask=m, other=0.).to(tl.float32), LIB).to(tl.bfloat16).to(tl.float32)
        p = _mul(s, tl.load(XN + off, mask=m, other=0.).to(tl.float32))
        if ROUND:
            p = p.to(tl.bfloat16).to(tl.float32)
        acc += p
    tl.store(MIX + row * H + h, (acc * (1.0 / G)).to(MIX.dtype.element_ty), mask=m)


@triton.jit
def _mix_bwd(DMIX, U, XN, DU, H, G: tl.constexpr, BH: tl.constexpr, LIB: tl.constexpr):
    """dS_g = BF16(dmix / G * xn_g) (the mix product's grad for the BF16 sigmoid), dU_g = sigmoid_backward(dS, s) as
    torch computes it for BF16: BF16(BF16(dS BF16(1 - s)) s) (rl1_hyper_test dbwd)."""
    row, j = tl.program_id(0).to(tl.int64), tl.program_id(1)
    h = j * BH + tl.arange(0, BH)
    m = h < H
    dp = tl.load(DMIX + row * H + h, mask=m, other=0.).to(tl.float32) * (1.0 / G)
    for g in tl.static_range(G):
        off = row * G * H + g * H + h
        s = _sigmoid(tl.load(U + off, mask=m, other=0.).to(tl.float32), LIB).to(tl.bfloat16).to(tl.float32)
        ds = (dp * tl.load(XN + off, mask=m, other=0.).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        s1 = (1.0 - s).to(tl.bfloat16).to(tl.float32)  # torch's BF16 sigmoid_backward rounds at each op
        tl.store(DU + off, ((ds * s1).to(tl.bfloat16).to(tl.float32) * s).to(tl.bfloat16), mask=m)


@triton.jit
def _inject_fwd(X, Y, W, OUT, H, G: tl.constexpr, BH: tl.constexpr):
    """OUT = x + BF16(y * w_g) in OUT's dtype (the model's BF16 injection product and residual add)."""
    row, j = tl.program_id(0).to(tl.int64), tl.program_id(1)
    h = j * BH + tl.arange(0, BH)
    m = h < H
    y = tl.load(Y + row * H + h, mask=m, other=0.).to(tl.float32)
    for g in tl.static_range(G):
        off = row * G * H + g * H + h
        w = tl.load(W + row * G + g).to(tl.float32)
        x = tl.load(X + off, mask=m, other=0.).to(tl.float32)
        tl.store(OUT + off, (x + (y * w).to(tl.bfloat16).to(tl.float32)).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _inject_dy(DO, W, DY, H, G: tl.constexpr, BH: tl.constexpr):
    """dy = BF16(sum_g fp32(BF16(BF16(do_g) w_g))) (MulBackward then the sum over the broadcast stream dim)."""
    row, j = tl.program_id(0).to(tl.int64), tl.program_id(1)
    h = j * BH + tl.arange(0, BH)
    m = h < H
    acc = tl.zeros([BH], dtype=tl.float32)
    for g in tl.static_range(G):
        d = tl.load(DO + row * G * H + g * H + h, mask=m, other=0.).to(tl.bfloat16).to(tl.float32)
        acc += (d * tl.load(W + row * G + g).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(DY + row * H + h, acc.to(tl.bfloat16), mask=m)


@triton.jit
def _inject_dw(DO, Y, DW, H, G: tl.constexpr, NJ: tl.constexpr, V: tl.constexpr):
    """dw_g = BF16(sum_h BF16(BF16(do_g) y)) in torch's reduce order; one (row, stream) per program."""
    row, g = tl.program_id(0).to(tl.int64), tl.program_id(1)
    off = tl.arange(0, 32)[:, None] * V + tl.arange(0, V)[None, :]
    acc = tl.zeros([32, V], dtype=tl.float32)
    for j in range(NJ):
        hm = off + j * 32 * V < H
        d = tl.load(DO + row * G * H + g * H + j * 32 * V + off, mask=hm, other=0.).to(tl.bfloat16).to(tl.float32)
        y = tl.load(Y + row * H + j * 32 * V + off, mask=hm, other=0.).to(tl.float32)
        acc += (d * y).to(tl.bfloat16).to(tl.float32)
    tl.store(DW + row * G + g, _tree(acc, V).to(tl.bfloat16))


# --------------------------------------------------------------------------------------------- autograd functions
class GatedResidualFn(torch.autograd.Function):
    """(mixed [N, H] in x's dtype, inject weights [N, G] BF16) from the hyper stream x [N, G*H] (BF16 or fp32 under
    BF16 autocast); the module's BF16 weights frozen (``why_not``: no weight gradients)."""

    @staticmethod
    def forward(ctx, x, norm_w, w_down, w_up, w_inj, G, H, eps):
        n = x.shape[0]
        inv_h = 1.0 / H
        xn = torch.empty_like(x)
        rs = torch.empty((n, G), dtype=torch.float32, device=x.device)
        # xb (MFU-1 24 Oct, HYPER["xb"]): the BF16 copy autocast makes of the fp32 xn for each of the two Linears,
        # written once by the norm kernel instead (same rounding, so the same GEMM inputs)
        cast = bool(HYPER.get("xb")) and x.dtype == torch.float32 and torch.is_autocast_enabled(x.device.type) \
            and torch.get_autocast_dtype(x.device.type) == BF16 and w_down.dtype == BF16 and w_inj.dtype == BF16
        xb = torch.empty(x.shape, dtype=BF16, device=x.device) if cast else xn
        if n:
            R = TUNE["rows"]
            _norm_fwd_rows[(triton.cdiv(n * G, R),)](x, norm_w, xn, rs, xb, n * G, H, eps, inv_h, G=G,
                                                     NJ=triton.cdiv(H, 32 * VEC), V=VEC, R=R, CAST=cast,
                                                     num_warps=TUNE["warps"], **NOFMA)
        if cast:
            HYPER["xb_calls"] = HYPER.get("xb_calls", 0) + 1
        zd = torch.nn.functional.linear(xb, w_down)
        a = torch.nn.functional.silu(zd / G)
        u = torch.nn.functional.linear(a, w_up)
        zi = torch.nn.functional.linear(xb, w_inj)
        inject = 2 * torch.sigmoid(zi / G)
        mixed = torch.empty((n, H), dtype=x.dtype, device=x.device)
        if n:
            _mix_fwd[(n, triton.cdiv(H, 1024))](u, xn, mixed, H, G=G, BH=1024, LIB=SIGMOID_LIB, ROUND=x.dtype == BF16,
                                                 num_warps=4, **NOFMA)
        ctx.save_for_backward(x, norm_w, w_down, w_up, w_inj, xn, rs, zd, u, inject)
        ctx.G, ctx.H = G, H
        return mixed, inject

    @staticmethod
    def backward(ctx, dmix, dinj):
        x, norm_w, w_down, w_up, w_inj, xn, rs, zd, u, inject = ctx.saved_tensors
        G, H = ctx.G, ctx.H
        n = x.shape[0]
        dmix = (dmix.contiguous() if dmix is not None
                else torch.zeros((n, H), dtype=x.dtype, device=x.device))
        du = torch.empty_like(u)
        if n:
            _mix_bwd[(n, triton.cdiv(H, 1024))](dmix, u, xn, du, H, G=G, BH=1024, LIB=SIGMOID_LIB, num_warps=4, **NOFMA)
        aten = torch.ops.aten
        dzd = aten.silu_backward(du @ w_up, zd / G) / G  # the model's silu(zd / G) backward ops
        gd = dzd @ w_down
        if dinj is not None:
            dzi = aten.sigmoid_backward(dinj.to(inject.dtype) * 2, inject / 2) / G  # 2 sigmoid(zi / G)
            gi = dzi @ w_inj
        else:
            gi = torch.zeros_like(gd)
        dx = torch.empty_like(x)
        if n:
            R = TUNE["rows"]
            _norm_bwd_rows[(triton.cdiv(n * G, R),)](dmix, u, gi, gd, x, norm_w, rs, dx, n * G, H, 1.0 / H, G=G,
                                                     NJ=triton.cdiv(H, 32 * VEC), V=VEC, LIB=SIGMOID_LIB, ROUND=x.dtype == BF16,
                                                     R=R, num_warps=TUNE["warps"], **NOFMA)
        return dx, None, None, None, None, None, None, None


class InjectFn(torch.autograd.Function):
    """x' = x + (y [N, H] per stream x w [N, G], BF16) on the hyper stream x [N, G*H] (x's dtype)."""

    @staticmethod
    def forward(ctx, x, y, w, G, H):
        n = x.shape[0]
        out = torch.empty_like(x)
        if n:
            _inject_fwd[(n, triton.cdiv(H, 1024))](x, y, w, out, H, G=G, BH=1024, num_warps=4, **NOFMA)
        ctx.save_for_backward(y, w)
        ctx.G, ctx.H = G, H
        return out

    @staticmethod
    def backward(ctx, dout):
        y, w = ctx.saved_tensors
        G, H = ctx.G, ctx.H
        d = dout.contiguous()
        n = y.shape[0]
        dy, dw = torch.empty_like(y), torch.empty_like(w)
        if n:
            _inject_dy[(n, triton.cdiv(H, 1024))](d, w, dy, H, G=G, BH=1024, num_warps=4, **NOFMA)
            _inject_dw[(n, G)](d, y, dw, H, G=G, NJ=triton.cdiv(H, 32 * WVEC), V=WVEC, num_warps=1, **NOFMA)
        return dout, dy, dw, None, None


# ----------------------------------------------------------------------------------------------- model wiring
def _plain(linear):
    return type(linear) is torch.nn.Linear and linear.weight.dtype == BF16 and linear.bias is None


def why_not(module, x):
    if x.dtype not in (BF16, torch.float32):
        return f"dtype {x.dtype}"
    if x.dtype == torch.float32 and not (torch.is_autocast_enabled(x.device.type)
                                         and torch.get_autocast_dtype(x.device.type) == BF16):
        return "fp32 without BF16 autocast"
    if module.block_inject_weight is None:
        return "no inject"
    linears = (module.input_mix_weight_down, module.input_mix_weight_up, module.block_inject_weight)
    if not all(_plain(m) for m in linears):
        return "not plain BF16 Linear"
    if module.hc_norm.weight.requires_grad or any(m.weight.requires_grad for m in linears):
        return "trainable weights"
    if module.hc_norm.group_size != module.hidden_size or module.hc_count & (module.hc_count - 1):
        return "norm groups"
    H = module.hidden_size
    if H % 8 or H < 32 * WVEC or x.numel() < 16 * module.hc_count * H:
        return "reduce layout"  # torch's reduce config differs for < 16 rows (outputs), short or unaligned rows
    return None


def gated_residual(module, x):
    """``(mixed, inject)`` as Qwen4ExpTextGatedResidual.forward returns them (with x itself as the hyper input)."""
    G, H = int(module.hc_count), int(module.hidden_size)
    shape = x.shape
    flat = x.reshape(-1, G * H).contiguous()
    mixed, inject = GatedResidualFn.apply(flat, module.hc_norm.weight, module.input_mix_weight_down.weight,
                                          module.input_mix_weight_up.weight, module.block_inject_weight.weight, G, H,
                                          float(module.hc_norm.eps))
    return mixed.view(*shape[:-1], H), inject.view(*shape[:-1], G)


def inject(x, y, w):
    """x + (y.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2); fused when y and w are BF16 (the model's case)."""
    if y.dtype != BF16 or w.dtype != BF16:
        return x + (y.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)
    G, H = w.shape[-1], y.shape[-1]
    shape = x.shape
    out = InjectFn.apply(x.reshape(-1, G * H).contiguous(), y.reshape(-1, H).contiguous(),
                         w.reshape(-1, G).contiguous(), G, H)
    return out.view(shape)


def install():
    """Patch Qwen4ExpTextDecoderLayer.forward once; ``HYPER["on"]`` picks the path per call (the model's otherwise)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextDecoderLayer
    if getattr(cls, "_rl1_hyper", False):
        return
    original = cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask=None, conv_mask=None, past_key_values=None,
                ple_input_ids=None, **kwargs):
        if HYPER["on"]:
            why = why_not(self.attn_hyper_connection, hidden_states) or why_not(self.mlp_hyper_connection, hidden_states)
            if why is None:
                HYPER["calls"] += 1
                key = str(hidden_states.dtype)[6:]
                HYPER["dtypes"][key] = HYPER["dtypes"].get(key, 0) + 1
                return fast_forward(self, hidden_states, position_embeddings, attention_mask, conv_mask,
                                    past_key_values, ple_input_ids, **kwargs)
            HYPER["fallback"] += 1
            HYPER["why"][why] = HYPER["why"].get(why, 0) + 1
        return original(self, hidden_states, position_embeddings, attention_mask=attention_mask, conv_mask=conv_mask,
                        past_key_values=past_key_values, ple_input_ids=ple_input_ids, **kwargs)

    cls.forward = forward
    cls._rl1_hyper = True
    cls._rl1_hyper_original = original


def fast_forward(self, hidden_states, position_embeddings, attention_mask, conv_mask, past_key_values, ple_input_ids,
                 **kwargs):
    """Qwen4ExpTextDecoderLayer.forward (transformers 5.18) with the fused hyper-connection calls."""
    if self.ple is not None:
        hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask)
    mixed, injection_weights = gated_residual(self.attn_hyper_connection, hidden_states)
    if self.layer_type == "linear_attention":
        out = self.linear_attn(mixed, cache_params=past_key_values, attention_mask=conv_mask, **kwargs)
    else:
        out, _ = self.self_attn(mixed, position_embeddings, attention_mask=attention_mask,
                                past_key_values=past_key_values, **kwargs)
    hidden_states = inject(hidden_states, out, injection_weights)
    mixed, injection_weights = gated_residual(self.mlp_hyper_connection, hidden_states)
    out = self.mlp(mixed)
    return inject(hidden_states, out, injection_weights)
