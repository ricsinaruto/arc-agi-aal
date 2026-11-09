"""rl1_gdn (trunk/rl, MFU stage G, 21 Oct) for a full fine-tune: the Gated DeltaNet layers as one autograd function,
with the weight gradients the LoRA version leaves out. Config ``gdn_fast`` (``gdn_keep``: the forward's w, h, v_new
kept for the backward instead of recomputed: bitwise, more memory). Needs CUDA, Triton and fla; anything else runs
the model's code.

From rl1_gdn unchanged: the four input projections as one GEMM over their weights packed into one buffer (the
parameters become views of it), the causal conv (+ cached 3 tokens) + SiLU + q/k L2 norm and the sigmoid-gated
RMSNorm as Triton kernels, the gate -exp(A_log) softplus(a + dt_bias) inside fla's chunk kernel; rounding points as
the model's BF16 path. Added: dW of the packed input projections (one GEMM) and of out_proj, the conv and norm
weights' grads (``conv_weight_grad``, ``norm_weight_grad``: torch, fp32 sums) and fla's dA_log / ddt_bias."""

import torch
import triton
import triton.language as tl

import fla_pin

GDN = {"fast": False, "calls": 0, "fallback": 0, "why": {}, "keep": False, "kept": 0}
PROJ_IN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
BF16 = torch.bfloat16
ROWS = 8192  # row chunk of the weight-grad sums


# ------------------------------------------------------------------------------------------------ Triton kernels
@triton.jit
def _conv_fwd(Y, sy, PREV, sp_b, sp_c, KP, W, OUT, R, T, col0, NH: tl.constexpr, HAS_PREV: tl.constexpr,
              NORM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    """One head's D channels of the causal depthwise conv (kernel 4) over BT tokens, SiLU, optional L2 norm."""
    i_t, i_h, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = i_t * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    c = col0 + i_h * D + d
    acc = tl.zeros([BT, D], dtype=tl.float32)
    for j in tl.static_range(4):
        tt = t - 3 + j
        m = (tt >= 0) & (tt < T)
        x = tl.load(Y + (i_b * T + tt).to(tl.int64)[:, None] * sy + c[None, :], mask=m[:, None], other=0.).to(tl.float32)
        if HAS_PREV:
            if i_t == 0:  # only the first token block reaches back into the cached tokens
                mp = (tt < 0) & (tt >= -3)
                x += tl.load(PREV + i_b * sp_b + c[None, :] * sp_c + (KP + tt)[:, None], mask=mp[:, None],
                             other=0.).to(tl.float32)
        w = tl.load(W + c * 4 + j).to(tl.float32)
        acc += x * w[None, :]
    pre = acc.to(tl.bfloat16).to(tl.float32)
    act = (pre * tl.sigmoid(pre)).to(tl.bfloat16).to(tl.float32)
    rows = (i_b * T + t).to(tl.int64)
    mt = t < T
    if NORM:
        r = 1.0 / tl.sqrt(tl.sum(act * act, axis=1) + 1e-6)
        act = act * r[:, None]
        tl.store(R + rows * NH + i_h, r, mask=mt)
    tl.store(OUT + (rows * NH + i_h)[:, None] * D + d[None, :], act.to(tl.bfloat16), mask=mt[:, None])


@triton.jit
def _conv_bwd_pre(Y, sy, PREV, sp_b, sp_c, KP, W, DOUT, YN, R, DPRE, sd, T, col0, NH: tl.constexpr,
                  HAS_PREV: tl.constexpr, NORM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    """Grad at the conv's pre-activation (BF16, as the model's SiLU backward): L2-norm backward, SiLU backward."""
    i_t, i_h, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = i_t * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    c = col0 + i_h * D + d
    acc = tl.zeros([BT, D], dtype=tl.float32)
    for j in tl.static_range(4):
        tt = t - 3 + j
        m = (tt >= 0) & (tt < T)
        x = tl.load(Y + (i_b * T + tt).to(tl.int64)[:, None] * sy + c[None, :], mask=m[:, None], other=0.).to(tl.float32)
        if HAS_PREV:
            if i_t == 0:  # only the first token block reaches back into the cached tokens
                mp = (tt < 0) & (tt >= -3)
                x += tl.load(PREV + i_b * sp_b + c[None, :] * sp_c + (KP + tt)[:, None], mask=mp[:, None],
                             other=0.).to(tl.float32)
        w = tl.load(W + c * 4 + j).to(tl.float32)
        acc += x * w[None, :]
    pre = acc.to(tl.bfloat16).to(tl.float32)
    rows = (i_b * T + t).to(tl.int64)
    mt = t < T
    off = (rows * NH + i_h)[:, None] * D + d[None, :]
    dy = tl.load(DOUT + off, mask=mt[:, None], other=0.).to(tl.float32)
    if NORM:
        yn = tl.load(YN + off, mask=mt[:, None], other=0.).to(tl.float32)
        r = tl.load(R + rows * NH + i_h, mask=mt, other=0.)
        dy = r[:, None] * (dy - yn * tl.sum(yn * dy, axis=1)[:, None])
        dy = dy.to(tl.bfloat16).to(tl.float32)
    s = tl.sigmoid(pre)
    dpre = dy * s * (1 + pre * (1 - s))
    tl.store(DPRE + rows[:, None] * sd + c[None, :], dpre.to(tl.bfloat16), mask=mt[:, None])


@triton.jit
def _conv_bwd_x(DPRE, sd, W, DX, sx, T, C, BT: tl.constexpr, BC: tl.constexpr):
    """Transposed depthwise conv: dx[t, c] = sum_j w[c, j] dpre[t + 3 - j, c] (tokens inside the window)."""
    i_t, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = i_t * BT + tl.arange(0, BT)
    c = i_c * BC + tl.arange(0, BC)
    mc = c < C
    acc = tl.zeros([BT, BC], dtype=tl.float32)
    for j in tl.static_range(4):
        tt = t + 3 - j
        m = (tt < T)[:, None] & mc[None, :]
        g = tl.load(DPRE + (i_b * T + tt).to(tl.int64)[:, None] * sd + c[None, :], mask=m, other=0.).to(tl.float32)
        w = tl.load(W + c * 4 + j, mask=mc, other=0.).to(tl.float32)
        acc += g * w[None, :]
    rows = (i_b * T + t).to(tl.int64)
    tl.store(DX + rows[:, None] * sx + c[None, :], acc.to(tl.bfloat16), mask=(t < T)[:, None] & mc[None, :])


@triton.jit
def _norm_gate_fwd(O, Z, sz, W, OUT, n_rows, eps, NH: tl.constexpr, BR: tl.constexpr, D: tl.constexpr):
    """Qwen4ExpTextRMSNormGated with a sigmoid gate, per (token, head) row of D: BF16 roundings as the model's."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    mr = r < n_rows
    tok, head = r // NH, r % NH
    o = tl.load(O + r.to(tl.int64)[:, None] * D + d[None, :], mask=mr[:, None], other=0.).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + eps)
    n = (o * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + d).to(tl.float32)
    hw = (w[None, :] * n).to(tl.bfloat16).to(tl.float32)
    z = tl.load(Z + tok.to(tl.int64)[:, None] * sz + (head * D)[:, None] + d[None, :], mask=mr[:, None],
                other=0.).to(tl.float32)
    tl.store(OUT + r.to(tl.int64)[:, None] * D + d[None, :], (hw * tl.sigmoid(z)).to(tl.bfloat16), mask=mr[:, None])


@triton.jit
def _norm_gate_bwd(O, Z, sz, W, DOUT, DO, DZ, sdz, n_rows, eps, NH: tl.constexpr, BR: tl.constexpr,
                   D: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    mr = r < n_rows
    tok, head = r // NH, r % NH
    off = r.to(tl.int64)[:, None] * D + d[None, :]
    o = tl.load(O + off, mask=mr[:, None], other=0.).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + eps)
    n = o * rstd[:, None]
    nb = n.to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + d).to(tl.float32)
    hw = (w[None, :] * nb).to(tl.bfloat16).to(tl.float32)
    zo = tok.to(tl.int64)[:, None] * sz + (head * D)[:, None] + d[None, :]
    z = tl.load(Z + zo, mask=mr[:, None], other=0.).to(tl.float32)
    gate = tl.sigmoid(z)
    g = tl.load(DOUT + off, mask=mr[:, None], other=0.).to(tl.float32)
    dhw = (g * gate).to(tl.bfloat16).to(tl.float32)
    dz = g * hw * gate * (1 - gate)
    tl.store(DZ + tok.to(tl.int64)[:, None] * sdz + (head * D)[:, None] + d[None, :], dz.to(tl.bfloat16),
             mask=mr[:, None])
    dn = (dhw * w[None, :]).to(tl.bfloat16).to(tl.float32)
    do = rstd[:, None] * (dn - n * (tl.sum(dn * n, axis=1) / D)[:, None])
    tl.store(DO + off, do.to(tl.bfloat16), mask=mr[:, None])


# ------------------------------------------------------------------------------------------------ launchers
CONV_BT, NORM_BR, BWDX_BT, BWDX_BC = 64, 32, 64, 128


def conv_fwd(y, prev, w, segs, T, B):
    """y [B*T, >=C] (row stride any), prev [B, C, KP] or None, w [C, 1, 4]; segs [(col0, heads, D, norm)]."""
    outs = []
    for col0, nh, D, norm in segs:
        out = torch.empty(B, T, nh, D, dtype=BF16, device=y.device)
        r = torch.empty(B, T, nh, dtype=torch.float32, device=y.device) if norm else out
        p = prev if prev is not None else y
        _conv_fwd[(triton.cdiv(T, CONV_BT), nh, B)](
            y, y.stride(0), p, p.stride(0) if prev is not None else 0, p.stride(1) if prev is not None else 0,
            prev.shape[-1] if prev is not None else 0, w, out, r, T, col0, NH=nh, HAS_PREV=prev is not None,
            NORM=norm, BT=CONV_BT, D=D)
        outs.append((out, r if norm else None))
    return outs


def conv_bwd(y, prev, w, segs, grads, saved, T, B, dx):
    """grads/saved per seg: (dout, normalized out, rstd); writes dx[:, :C] (a column block of the dY buffer) and
    returns the grad at the conv's pre-activation [B*T, C] (for the conv weight's grad)."""
    C = sum(nh * D for _, nh, D, _ in segs)
    dpre = torch.empty(B * T, C, dtype=BF16, device=y.device)
    for (col0, nh, D, norm), g, (yn, r) in zip(segs, grads, saved):
        p = prev if prev is not None else y
        _conv_bwd_pre[(triton.cdiv(T, CONV_BT), nh, B)](
            y, y.stride(0), p, p.stride(0) if prev is not None else 0, p.stride(1) if prev is not None else 0,
            prev.shape[-1] if prev is not None else 0, w, g, yn if norm else g, r if norm else g, dpre, dpre.stride(0),
            T, col0, NH=nh, HAS_PREV=prev is not None, NORM=norm, BT=CONV_BT, D=D)
    _conv_bwd_x[(triton.cdiv(T, BWDX_BT), triton.cdiv(C, BWDX_BC), B)](
        dpre, dpre.stride(0), w, dx, dx.stride(0), T, C, BT=BWDX_BT, BC=BWDX_BC)
    return dpre


def norm_gate_fwd(o, z, w, eps):
    """o [N, NH, D] contiguous, z [N, NH*D] (row stride any) -> [N, NH*D]."""
    N, nh, D = o.shape
    out = torch.empty(N, nh * D, dtype=BF16, device=o.device)
    rows = N * nh
    _norm_gate_fwd[(triton.cdiv(rows, NORM_BR),)](o, z, z.stride(0), w, out, rows, eps, NH=nh, BR=NORM_BR, D=D)
    return out


def norm_gate_bwd(o, z, w, dout, dz, eps):
    N, nh, D = o.shape
    do = torch.empty_like(o)
    rows = N * nh
    _norm_gate_bwd[(triton.cdiv(rows, NORM_BR),)](o, z, z.stride(0), w, dout, do, dz, dz.stride(0), rows, eps,
                                                   NH=nh, BR=NORM_BR, D=D)
    return do


# ------------------------------------------------------------------------------------------------ the function
def _fla():
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd

    fla_pin.pin()
    return chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd


def fwd_keep(q, k, v, g, beta, scale, initial_state, output_final_state, A_log, dt_bias):
    """fla's chunk_gated_delta_rule_fwd (use_gate_in_kernel, chunk 64, no cu_seqlens / CP), the same calls in the same
    order, also returning (w, h, v_new): fla's backward recomputes them with the same kernels on the same inputs
    (recompute_w_u_fwd(k, v, beta, A, g) is also the forward's step 2), so keeping them is bitwise equal."""
    import fla.ops.gated_delta_rule.chunk as C

    fla_pin.pin()
    g_out = C.gdn_gate_chunk_cumsum(g=g, A_log=A_log, chunk_size=64, scale=C.RCP_LN2, dt_bias=dt_bias, cu_seqlens=None,
                                    chunk_indices=None)
    w, u, A = C.chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_out, beta=beta, cu_seqlens=None, chunk_indices=None,
                                                 chunk_size=64)
    h, v_new, final = C.chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g_out, initial_state=initial_state,
                                                     output_final_state=output_final_state, cu_seqlens=None,
                                                     chunk_indices=None, state_v_first=False, chunk_size=64)
    o = C.chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_out, scale=scale, cu_seqlens=None, chunk_indices=None,
                      state_v_first=False, chunk_size=64)
    return g_out, o, A, final, (w, h, v_new)


def bwd_keep(q, k, v, g, beta, A, scale, initial_state, do, g_input, A_log, dt_bias, kept):
    """fla's chunk_gated_delta_rule_bwd (same settings as fwd_keep) without its recompute of w, h, v_new."""
    import fla.ops.gated_delta_rule.chunk as C

    w, h, v_new = kept
    dv = C.chunk_bwd_dv_local(q=q, k=k, g=g, do=do, scale=scale, cu_seqlens=None, chunk_indices=None, chunk_size=64)
    dh, dh0, dv = C.chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, g=g, h0=initial_state, dht=None, do=do, dv=dv,
                                                   scale=scale, cu_seqlens=None, chunk_indices=None,
                                                   state_v_first=False, chunk_size=64)
    dq, dk, dw, dg = C.chunk_bwd_dqkwg(q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh, scale=scale,
                                       cu_seqlens=None, chunk_indices=None, state_v_first=False, chunk_size=64)
    dk2, dv, db, dg2 = C.prepare_wy_repr_bwd(k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv, cu_seqlens=None,
                                             chunk_indices=None)
    dk.add_(dk2)
    dg.add_(dg2)
    dg = C.chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=None, chunk_indices=None)
    dg, dA_log, ddt_bias = C.gdn_gate_bwd(g=g_input, A_log=A_log, dt_bias=dt_bias, dyg=dg)
    return dq, dk, dv, db, dg, dh0, dA_log, ddt_bias


def conv_weight_grad(dpre, y, prev, B, T, C):
    """dW [C, 1, 4] (fp32) of the causal depthwise conv out[t] = sum_j w[j] x[t - 3 + j] (x before the window: the
    cached ``prev`` [B, C, >=3] tokens, else 0): ``dpre`` [B*T, C] the grad at its output, ``y`` [B*T, >=C] its input
    rows."""
    x = y[:, :C].reshape(B, T, C)
    d = dpre.reshape(B, T, C)
    before = prev[..., -3:].transpose(1, 2).to(x.dtype) if prev is not None else x.new_zeros(B, 3, C)
    out = torch.zeros(C, 4, dtype=torch.float32, device=y.device)
    for a in range(0, T, ROWS):
        b = min(T, a + ROWS)
        seg = x[:, max(a - 3, 0):b]
        if a < 3:
            seg = torch.cat([before[:, a:], seg], dim=1)
        for j in range(4):
            out[:, j] += (d[:, a:b].float() * seg[:, j:j + b - a].float()).sum((0, 1))
    return out.view(C, 1, 4)


def norm_weight_grad(o, z, dcore, eps):
    """d weight [D] (fp32) of the gated RMSNorm w * BF16(rmsnorm(o)) * sigmoid(z) (Qwen4ExpTextRMSNormGated, sigmoid
    gate): ``o`` [N, heads, D], ``z`` / ``dcore`` [N, heads * D] (z's row stride any)."""
    N, nh, D = o.shape
    out = torch.zeros(D, dtype=torch.float32, device=o.device)
    for a in range(0, N, ROWS):
        b = min(N, a + ROWS)
        h = o[a:b].float()
        h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(BF16).float()
        g = (dcore[a:b].float() * torch.sigmoid(z[a:b].float())).view(b - a, nh, D).to(BF16).float()
        out += (g * h).sum((0, 1))
    return out


def served_weights(w_in, w_out, int8_rows):
    """The GEMM weights of the forward and of the input grads: with ``int8_rows`` > 0 (dense_int8_train) the first
    ``int8_rows`` rows of the packed buffer (in_proj_qkv, in_proj_z) and out_proj on their served INT8 round trip
    (qat.int8_rows, per row as the separate modules'), in_proj_b / in_proj_a BF16; the weight grads stay
    straight-through (they never read the weights)."""
    if not int8_rows:
        return w_in, w_out
    import qat

    return torch.cat([qat.int8_rows(w_in[:int8_rows]), w_in[int8_rows:]]), qat.int8_rows(w_out)


class GatedDeltaFull(torch.autograd.Function):
    """x [B, T, hidden] -> (out [B, T, hidden], final recurrent state, last <=4 pre-conv qkv tokens [B, C, t]); grads
    for x and every weight. ``w_in`` is the packed buffer the four projection weights are views of (used for the
    GEMMs); their grads go to them."""

    @staticmethod
    def forward(ctx, x, w_in, w_qkv, w_z, w_b, w_a, w_out, conv_w, a_log, dt_bias, norm_w, prev, h0, meta):
        fwd, _ = _fla()
        with torch.autocast("cuda", enabled=False):
            B, T, H = x.shape
            N = B * T
            nk, nv, dk, dv, eps, want_final, _, int8 = meta
            kd, vd = nk * dk, nv * dv
            C = 2 * kd + vd
            x2 = x.reshape(N, H).to(BF16)
            gin, gout = served_weights(w_in, w_out, int8)
            y = x2 @ gin.t()
            segs = ((0, nk, dk, True), (kd, nk, dk, True), (2 * kd, nv, dv, False))
            (q, rq), (k, rk), (v, _) = conv_fwd(y, prev, conv_w, segs, T, B)
            beta = torch.sigmoid(y[:, C + vd:C + vd + nv]).view(B, T, nv)
            ga = y[:, C + vd + nv:C + vd + 2 * nv].contiguous().view(B, T, nv)
            if GDN["keep"] and any(ctx.needs_input_grad):  # (forward runs with grad off: not is_grad_enabled)
                g, o, A, final, kept = fwd_keep(q, k, v, ga, beta, dk ** -0.5, h0, want_final, a_log, dt_bias)
            else:
                g, o, A, final, _, _ = fwd(q=q, k=k, v=v, g=ga, beta=beta, scale=dk ** -0.5, initial_state=h0,
                                           output_final_state=want_final, use_gate_in_kernel=True, A_log=a_log,
                                           dt_bias=dt_bias)
                kept = ()
            o = o.to(BF16)
            core = norm_gate_fwd(o.view(N, nv, dv), y[:, C:C + vd], norm_w, eps)
            out = core @ gout.t()
            del gin, gout
            tail = y.view(B, T, -1)[:, -min(T, 4):, :C].transpose(1, 2).contiguous()
        ctx.save_for_backward(x2, y, q, k, v, rq, rk, beta, g, A, ga, o, core, prev, h0, w_in, w_out, conv_w, a_log,
                              dt_bias, norm_w, *kept)
        ctx.meta, ctx.shape, ctx.kept = meta, (B, T, H), len(kept)
        if final is None:
            final = x.new_zeros(0)
        ctx.mark_non_differentiable(final, tail)
        return out.view(B, T, H).to(x.dtype), final, tail

    @staticmethod
    def backward(ctx, dout, _dfinal, _dtail):
        _, bwd = _fla()
        (x2, y, q, k, v, rq, rk, beta, g, A, ga, o, core, prev, h0, w_in, w_out, conv_w, a_log, dt_bias, norm_w,
         *kept) = ctx.saved_tensors
        nk, nv, dk, dv, eps, _, widths, int8 = ctx.meta
        B, T, H = ctx.shape
        N = B * T
        kd, vd = nk * dk, nv * dv
        C = 2 * kd + vd
        with torch.autocast("cuda", enabled=False):
            d2 = dout.reshape(N, H).to(BF16).contiguous()
            gin, gout = served_weights(w_in, w_out, int8)  # recomputed (bitwise the forward's), not kept
            dcore = d2 @ gout
            d_out = d2.t() @ core
            dY = torch.empty_like(y)
            do = norm_gate_bwd(o.view(N, nv, dv), y[:, C:C + vd], norm_w, dcore, dY[:, C:C + vd], eps)
            d_norm = norm_weight_grad(o.view(N, nv, dv), y[:, C:C + vd], dcore, eps)
            if kept:
                GDN["kept"] += 1
                dq, dk_, dv_, db, dg, _, d_alog, d_dtb = bwd_keep(q, k, v, g, beta, A, dk ** -0.5, h0,
                                                                  do.view(B, T, nv, dv), ga, a_log, dt_bias, tuple(kept))
            else:
                dq, dk_, dv_, db, dg, _, d_alog, d_dtb = bwd(q=q, k=k, v=v, g=g, beta=beta, A=A, scale=dk ** -0.5,
                                                             initial_state=h0, do=do.view(B, T, nv, dv), dht=None,
                                                             use_gate_in_kernel=True, g_input=ga, A_log=a_log,
                                                             dt_bias=dt_bias)
            bf = beta.float()
            dY[:, C + vd:C + vd + nv] = (db.float() * bf * (1 - bf)).view(N, nv).to(BF16)
            dY[:, C + vd + nv:C + vd + 2 * nv] = dg.view(N, nv).to(BF16)
            segs = ((0, nk, dk, True), (kd, nk, dk, True), (2 * kd, nv, dv, False))
            dpre = conv_bwd(y, prev, conv_w, segs, (dq.to(BF16).contiguous(), dk_.to(BF16).contiguous(),
                                                    dv_.to(BF16).contiguous()), ((q, rq), (k, rk), (None, None)), T, B, dY)
            d_conv = conv_weight_grad(dpre, y, prev, B, T, C)
            dx = dY @ gin
            d_in = dY.t() @ x2
        parts, at = [], 0
        for width in widths:
            parts.append(d_in[at:at + width])
            at += width
        return (dx.view(B, T, H), None, *parts, d_out, d_conv.to(conv_w.dtype), d_alog.to(a_log.dtype),
                d_dtb.to(dt_bias.dtype), d_norm.to(norm_w.dtype), None, None, None)


# ------------------------------------------------------------------------------------------------ the module patch
def _packed(mod):
    """The four input-projection weights as one [sum out, hidden] buffer; the parameters become views of it (rebuilt
    if anything re-points them; the optimizer updates them in place)."""
    ws = [getattr(mod, n).weight for n in PROJ_IN]
    have = mod.__dict__.get("_sft_gdn_pack")
    if have is not None and all(w.data_ptr() == p for w, p in zip(ws, have[1])):
        return have[0]
    with torch.no_grad():
        cat = torch.cat([w.detach() for w in ws])
        start = 0
        for w in ws:
            w.data = cat[start:start + w.shape[0]]
            start += w.shape[0]
    mod.__dict__["_sft_gdn_pack"] = (cat, [w.data_ptr() for w in ws])
    return cat


def _plain(lin):
    return type(lin) is torch.nn.Linear and lin.bias is None and lin.weight.dtype == BF16


def why_not(mod, hidden_states, cache_params, kwargs):
    if kwargs.get("cu_seq_lens_q") is not None or kwargs.get("cu_seqlens") is not None:
        return "packed"
    if not hidden_states.is_cuda:
        return "cpu"
    if mod.activation not in ("silu", "swish") or mod.conv_kernel_size != 4 or mod.conv1d.bias is not None:
        return "conv"
    if getattr(mod.norm, "activation", None) != "sigmoid":
        return "norm"
    if not all(_plain(getattr(mod, n)) for n in (*PROJ_IN, "out_proj")):
        return "linear"
    if cache_params is not None:
        layer = cache_params.layers[mod.layer_idx]
        if getattr(layer, "record_past", False):
            return "record_past"
        if hidden_states.shape[1] == 1 and cache_params.has_previous_state(mod.layer_idx, state_idx=0):
            return "decode"
    return None


def int8_rows_of(mod):
    """Packed rows on the served INT8 grid: in_proj_qkv + in_proj_z when dense_int8_train wrapped this layer's
    modules (qat.install_dense) and it is on, else 0."""
    import qat

    wrapped = [getattr(mod, n).__dict__.get("_sft_int8", False) for n in ("in_proj_qkv", "in_proj_z", "out_proj")]
    if not qat.DENSE["on"] or not any(wrapped):
        return 0
    if not all(wrapped):
        raise RuntimeError("dense INT8: a GDN layer with only some of in_proj_qkv / in_proj_z / out_proj wrapped")
    qat.DENSE["calls"] += 1
    return mod.in_proj_qkv.weight.shape[0] + mod.in_proj_z.weight.shape[0]


def fast_forward(mod, hidden_states, cache_params=None, attention_mask=None):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    hidden_states = m.apply_mask_to_padding_states(hidden_states, attention_mask)
    B, T, _ = hidden_states.shape
    prev = h0 = None
    if cache_params is not None and cache_params.has_previous_state(mod.layer_idx, state_idx=0):
        layer = cache_params.layers[mod.layer_idx]
        prev = layer.conv_states[0][..., -3:]
        prev = (prev.expand(B, *prev.shape[1:]) if prev.shape[0] != B else prev).contiguous()
        if prev.data_ptr() == layer.conv_states[0].data_ptr():
            prev = prev.clone()  # the cache's in-place update follows this call
        h0 = layer.recurrent_states[0]
        h0 = (h0.expand(B, *h0.shape[1:]) if h0.shape[0] != B else h0).float().contiguous()
    ws = [getattr(mod, n).weight for n in PROJ_IN]
    meta = (mod.num_k_heads, mod.num_v_heads, mod.head_k_dim, mod.head_v_dim, float(mod.norm.variance_epsilon),
            cache_params is not None, tuple(w.shape[0] for w in ws), int8_rows_of(mod))
    out, final, tail = GatedDeltaFull.apply(hidden_states, _packed(mod), *ws, mod.out_proj.weight, mod.conv1d.weight,
                                            mod.A_log, mod.dt_bias, mod.norm.weight, prev, h0, meta)
    if cache_params is not None:
        cache_params.update_conv_state(tail, mod.layer_idx, conv_kernel_size=mod.conv_kernel_size)
        cache_params.update_recurrent_state(final, mod.layer_idx)
    return out


def install():
    """Patch Qwen4ExpTextGatedDeltaNet.forward once; ``GDN["fast"]`` picks the path per call."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextGatedDeltaNet
    if getattr(cls, "_sft_gdn", False):
        return
    original = cls.forward

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if GDN["fast"]:
            why = why_not(self, hidden_states, cache_params, kwargs)
            if why is None:
                GDN["calls"] += 1
                return fast_forward(self, hidden_states, cache_params, attention_mask)
            GDN["fallback"] += 1
            GDN["why"][why] = GDN["why"].get(why, 0) + 1
        return original(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

    cls.forward = forward
    cls._sft_gdn = True
