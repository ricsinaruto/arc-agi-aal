"""MFU stage G (RL arm, 21 Oct): the Gated DeltaNet layers (Qwen4ExpTextGatedDeltaNet, 36 of Flash's 48) as one
autograd function. The model's path spends ~60% of a window's GDN time outside GEMMs and the FLA chunk kernels:
transposes and fp32 casts, the prefix-state concat, cuDNN-less depthwise conv (fwd 3 ms, bwd 2 ms per 16K window),
q/k repeat_interleave to 48 heads, the unfused gated RMSNorm, peft's separate LoRA adds, and slice-backward
accumulation of the projection grads. Here:

- the four input projections are one GEMM over the concatenated frozen weights (base weights re-pointed as views of
  that buffer: no extra memory) plus the rank-16 LoRA terms as one [N, 64] GEMM and per-projection addmm into the
  output; the backward assembles one dY buffer and runs one dX GEMM;
- causal conv (with the cached 3-token prefix) + SiLU + q/k L2 norm in one Triton kernel per q/k/v, writing the
  FLA layouts directly (q and k stay at 16 heads: FLA's GVA maps value head i to key head i // 3, as the model's
  repeat_interleave does);
- gate -exp(A_log) * softplus(a + dt_bias) inside FLA's chunk kernel (use_gate_in_kernel), beta = sigmoid(b);
- gated RMSNorm (sigmoid gate) fwd/bwd in one Triton kernel each; out_proj GEMM + LoRA addmm.

Rounding points mirror the model's BF16 path (conv out, SiLU out, normalized q/k, the norm's BF16 products).
Switch: ``GDN["fast"]`` (trainer cmd ``gdn_fast``); anything the fast path
does not cover (decode steps, packed sequences, adapters not plain peft LoRA, non-BF16 base) runs the model's code."""

import torch
import triton
import triton.language as tl

GDN = {"fast": False, "calls": 0, "fallback": 0, "why": {}, "keep": True, "kept": 0}  # keep: set by the trainer, always on
PROJ_IN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
BF16 = torch.bfloat16


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
    """grads/saved per seg: (dout, normalized out, rstd); writes dx[:, :C] (a column block of the dY buffer)."""
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
FLA_PIN = {"done": False}


def pin_fla():
    """fla 0.5.2 / Triton 3.6 on B300: chunk_fwd_kernel_o's BK=BV=128 (8-warp) config, which autotune picks, now and
    then writes one wrong 16-row x 128-column tile (|err| 0.02-0.05 against fp32, the right tile 5e-4), so the GDN
    forward is neither repeatable nor exact (MFU-1 gdnrace-b1). Its BK=BV=64 config repeats bit-exactly; pin it."""
    if FLA_PIN["done"]:
        return
    from fla.ops.common import chunk_o

    at = chunk_o.chunk_fwd_kernel_o
    while not hasattr(at, "configs"):
        at = at.fn
    keep = [c for c in at.configs if c.kwargs.get("BK") == 64 and c.kwargs.get("BV") == 64]
    if not keep:
        raise RuntimeError(f"pin_fla: no BK=BV=64 config in {[str(c) for c in at.configs]}")
    at.configs = keep[:1]
    at.cache.clear()
    FLA_PIN["done"] = True


def fwd_keep(q, k, v, g, beta, scale, initial_state, output_final_state, A_log, dt_bias):
    """fla's chunk_gated_delta_rule_fwd (use_gate_in_kernel, chunk 64, no cu_seqlens / CP), the same calls in the same
    order, also returning (w, h, v_new): fla's backward recomputes them with the same kernels on the same inputs
    (recompute_w_u_fwd(k, v, beta, A, g) is also the forward's step 2), so keeping them is bitwise equal."""
    import fla.ops.gated_delta_rule.chunk as C

    pin_fla()
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


class GatedDeltaFast(torch.autograd.Function):
    """x [B, T, hidden] -> (out [B, T, hidden], final recurrent state, last <=4 pre-conv qkv tokens [B, C, t])."""

    @staticmethod
    def forward(ctx, x, w_in, w_out, conv_w, a_log, dt_bias, norm_w, prev, h0, meta, *lora):
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fwd

        pin_fla()
        with torch.autocast("cuda", enabled=False):
            B, T, H = x.shape
            N = B * T
            nk, nv, dk, dv, eps, s, want_final = meta
            kd, vd = nk * dk, nv * dv
            C = 2 * kd + vd
            x2 = x.reshape(N, H).to(BF16)
            a_in = torch.cat([lora[0], lora[2], lora[4], lora[6]]).to(BF16)
            xa = x2 @ a_in.t()
            y = x2 @ w_in.t()
            col, rank = 0, lora[0].shape[0]
            for i in range(4):
                width = lora[2 * i + 1].shape[0]
                y[:, col:col + width].addmm_(xa[:, i * rank:(i + 1) * rank], (lora[2 * i + 1] * s).to(BF16).t())
                col += width
            segs = ((0, nk, dk, True), (kd, nk, dk, True), (2 * kd, nv, dv, False))
            (q, rq), (k, rk), (v, _) = conv_fwd(y, prev, conv_w, segs, T, B)
            beta = torch.sigmoid(y[:, C + vd:C + vd + nv]).view(B, T, nv)
            ga = y[:, C + vd + nv:C + vd + 2 * nv].contiguous().view(B, T, nv)
            # gdn_keep (always on): a pass with a backward keeps (w, h, v_new) for it (forward runs with grad off: not
            # is_grad_enabled, so the inputs' needs_input_grad decides)
            if any(ctx.needs_input_grad):
                g, o, A, final, kept = fwd_keep(q, k, v, ga, beta, dk ** -0.5, h0, want_final, a_log, dt_bias)
            else:
                g, o, A, final, _, _ = fwd(q=q, k=k, v=v, g=ga, beta=beta, scale=dk ** -0.5, initial_state=h0,
                                           output_final_state=want_final, use_gate_in_kernel=True, A_log=a_log,
                                           dt_bias=dt_bias)
                kept = ()
            o = o.to(BF16)
            core = norm_gate_fwd(o.view(N, nv, dv), y[:, C:C + vd], norm_w, eps)
            ao = lora[8].to(BF16)
            xo = core @ ao.t()
            out = core @ w_out.t()
            out.addmm_(xo, (lora[9] * s).to(BF16).t())
            tail = y.view(B, T, -1)[:, -min(T, 4):, :C].transpose(1, 2).contiguous()
        ctx.save_for_backward(x2, xa, y, q, k, v, rq, rk, beta, g, A, ga, o, core, xo, prev, h0, w_in, w_out, conv_w,
                              a_log, dt_bias, norm_w, *lora, *kept)
        ctx.meta, ctx.shape = meta, (B, T, H)
        if final is None:
            final = x.new_zeros(0)
        ctx.mark_non_differentiable(final, tail)
        return out.view(B, T, H).to(x.dtype), final, tail

    @staticmethod
    def backward(ctx, dout, _dfinal, _dtail):
        (x2, xa, y, q, k, v, rq, rk, beta, g, A, ga, o, core, xo, prev, h0, w_in, w_out, conv_w, a_log, dt_bias, norm_w,
         *lora) = ctx.saved_tensors
        lora, kept = lora[:10], tuple(lora[10:])
        nk, nv, dk, dv, eps, s, _ = ctx.meta
        B, T, H = ctx.shape
        N = B * T
        kd, vd = nk * dk, nv * dv
        C = 2 * kd + vd
        with torch.autocast("cuda", enabled=False):
            d2 = dout.reshape(N, H).to(BF16).contiguous()
            bo = (lora[9] * s).to(BF16)
            dcore = d2 @ w_out
            dxo = d2 @ bo
            dcore.addmm_(dxo, lora[8].to(BF16))
            d_bo = (d2.t() @ xo).float() * s
            d_ao = (dxo.t() @ core).float()
            dY = torch.empty_like(y)
            do = norm_gate_bwd(o.view(N, nv, dv), y[:, C:C + vd], norm_w, dcore, dY[:, C:C + vd], eps)
            GDN["kept"] += 1
            dq, dk_, dv_, db, dg, _, _, _ = bwd_keep(q, k, v, g, beta, A, dk ** -0.5, h0, do.view(B, T, nv, dv), ga,
                                                     a_log, dt_bias, kept)
            bf = beta.float()
            dY[:, C + vd:C + vd + nv] = (db.float() * bf * (1 - bf)).view(N, nv).to(BF16)
            dY[:, C + vd + nv:C + vd + 2 * nv] = dg.view(N, nv).to(BF16)
            segs = ((0, nk, dk, True), (kd, nk, dk, True), (2 * kd, nv, dv, False))
            conv_bwd(y, prev, conv_w, segs, (dq.to(BF16).contiguous(), dk_.to(BF16).contiguous(),
                                             dv_.to(BF16).contiguous()), ((q, rq), (k, rk), (None, None)), T, B, dY)
            dx = dY @ w_in
            rank = lora[0].shape[0]
            col, grads, parts = 0, [], []
            for i in range(4):
                width = lora[2 * i + 1].shape[0]
                seg = dY[:, col:col + width]
                parts.append(seg @ (lora[2 * i + 1] * s).to(BF16))
                grads.append((seg.t() @ xa[:, i * rank:(i + 1) * rank]).float() * s)
                col += width
            dxa = torch.cat(parts, dim=1)
            a_in = torch.cat([lora[0], lora[2], lora[4], lora[6]]).to(BF16)
            dx.addmm_(dxa, a_in)
            d_ain = (dxa.t() @ x2).float()
        lora_grads = []
        for i in range(4):
            lora_grads += [d_ain[i * rank:(i + 1) * rank], grads[i]]
        lora_grads += [d_ao, d_bo]
        return (dx.view(B, T, H), None, None, None, None, None, None, None, None, None, *lora_grads)


# ------------------------------------------------------------------------------------------------ the module patch
def _lora(lin):
    """(A, B, scale) of a plain single-adapter peft LoRA Linear over a BF16 nn.Linear without bias, else None."""
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(lin, LoraLinear) or lin.merged or lin.disable_adapters or len(lin.active_adapters) != 1:
        return None
    name = lin.active_adapters[0]
    base = lin.base_layer
    if type(base) is not torch.nn.Linear or base.bias is not None or base.weight.dtype != BF16:
        return None
    if not isinstance(lin.lora_dropout[name], torch.nn.Identity) or lin.use_dora.get(name, False):
        return None
    if getattr(lin, "lora_bias", {}).get(name, False):
        return None
    return lin.lora_A[name].weight, lin.lora_B[name].weight, float(lin.scaling[name])


def _packed(mod):
    """The four frozen input-projection weights as one [sum out, hidden] buffer; the modules' weights become views of
    it (rebuilt if anything re-points them)."""
    ws = [getattr(mod, n).base_layer.weight for n in PROJ_IN]
    have = mod.__dict__.get("_rl1_gdn_pack")
    if have is not None and all(w.data_ptr() == p for w, p in zip(ws, have[1])):
        return have[0]
    with torch.no_grad():
        cat = torch.cat([w.detach() for w in ws])
        start = 0
        for w in ws:
            w.data = cat[start:start + w.shape[0]]
            start += w.shape[0]
    mod.__dict__["_rl1_gdn_pack"] = (cat, [w.data_ptr() for w in ws])
    return cat


def _why_not(mod, hidden_states, cache_params, kwargs):
    if kwargs.get("cu_seq_lens_q") is not None or kwargs.get("cu_seqlens") is not None:
        return "packed"
    if not hidden_states.is_cuda:
        return "cpu"
    if mod.activation not in ("silu", "swish") or mod.conv_kernel_size != 4 or mod.conv1d.bias is not None:
        return "conv"
    if getattr(mod.norm, "activation", None) != "sigmoid":
        return "norm"
    if cache_params is not None:
        layer = cache_params.layers[mod.layer_idx]
        if getattr(layer, "record_past", False):
            return "record_past"
        if hidden_states.shape[1] == 1 and cache_params.has_previous_state(mod.layer_idx, state_idx=0):
            return "decode"
    parts = [_lora(getattr(mod, n)) for n in (*PROJ_IN, "out_proj")]
    if any(p is None for p in parts) or len({p[2] for p in parts}) != 1 or len({p[0].shape[0] for p in parts}) != 1:
        return "adapter"
    return None


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
    parts = [_lora(getattr(mod, n)) for n in (*PROJ_IN, "out_proj")]
    lora = [t for a, b, _ in parts for t in (a, b)]
    meta = (mod.num_k_heads, mod.num_v_heads, mod.head_k_dim, mod.head_v_dim, float(mod.norm.variance_epsilon),
            parts[0][2], cache_params is not None)
    out, final, tail = GatedDeltaFast.apply(hidden_states, _packed(mod), mod.out_proj.base_layer.weight,
                                            mod.conv1d.weight, mod.A_log, mod.dt_bias, mod.norm.weight, prev, h0,
                                            meta, *lora)
    if cache_params is not None:
        cache_params.update_conv_state(tail, mod.layer_idx, conv_kernel_size=mod.conv_kernel_size)
        cache_params.update_recurrent_state(final, mod.layer_idx)
    return out


def install():
    """Patch Qwen4ExpTextGatedDeltaNet.forward once; ``GDN["fast"]`` picks the path per call."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextGatedDeltaNet
    if getattr(cls, "_rl1_gdn", False):
        return
    original = cls.forward

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if GDN["fast"]:
            why = _why_not(self, hidden_states, cache_params, kwargs)
            if why is None:
                GDN["calls"] += 1
                return fast_forward(self, hidden_states, cache_params, attention_mask)
            GDN["fallback"] += 1
            GDN["why"][why] = GDN["why"].get(why, 0) + 1
        return original(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

    cls.forward = forward
    cls._rl1_gdn = True
