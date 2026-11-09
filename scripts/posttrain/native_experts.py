"""Routed experts for the RL trainer, fused (RL1_EXPERTS=native; MFU stage D, 21 Oct).

Replaces Axolotl's sonicmoe NVFP4 grouped path (``nvfp4_lora.grouped_moe_reference_forward``, backend ``dequant``),
same math (W4A16: BF16 activations, base = BF16(q x block scale x expert scale)), fewer passes over memory:

- the base is decoded once per call by one coalesced Triton kernel into a weight buffer that also carries the LoRA
  factors as 16 extra K rows/columns: W1x = [[W1, B1], [A1, 0]] [E, 2I+R, H+R], W2x = [[W2, B2], [A2, 0]]. One
  grouped GEMM (torch._grouped_mm, CUTLASS) then gives base + LoRA: h = [x | s z1] W1x[:2I]^T with z1 = x A1^T, and
  in the backward dx = [dh | s dz1] W1x[:, :H] with dz1 = dh B1. No separate LoRA GEMMs or delta adds.
- Triton tile kernels over each expert's rows do the rest in one pass each: token gather + z = x A^T (writing the
  extended rows), SwiGLU + z2 (forward), SwiGLU backward + router-weight gradient (act . da) + dz1, the router-
  weighted combine and the dx gather back to tokens (fixed order, FP32 sums), the four LoRA weight gradients (FP32).
- The decoded buffer of the last layer stays in a one-slot cache (5.2 GB for Flash), so a layer's backward right
  after its checkpoint recompute skips the decode (RL1_NATIVE_CACHE=0: no cache).

The router-weight gradient uses act . da_u (da_u: the gradient of the unweighted expert output wrt act), which
equals g[token] . y_expert without saving the [rows, H] expert output.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

BM = 64  # rows per tile of the row kernels (the tile schedule depends on it)
STATS = {"calls": 0, "fallback": 0, "decodes": 0}
CACHE = {"key": None, "w1": None, "w2": None}


@triton.jit
def _decode(q, s, ROWS: tl.constexpr, COLS: tl.constexpr):
    """[ROWS, COLS // 2] packed E2M1 (low nibble first) and [ROWS, COLS // 16] E4M3 scales -> FP32 value x scale."""
    q = q.to(tl.int32)
    c = tl.reshape(tl.join(q & 15, q >> 4), (ROWS, COLS))
    mag = c & 7
    # E2M1 magnitude (exponent mag >> 1, mantissa mag & 1) as BF16 bits: normal (mag >= 2) rebias by 126; 0.5 for 1
    bits = (mag << 6) + tl.where(mag >= 2, 0x3F00, tl.where(mag == 1, 0x3EC0, 0))
    bits = bits | ((c & 8) << 12)
    w = bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    sc = s.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return w * tl.reshape(tl.broadcast_to(sc[:, :, None], (ROWS, COLS // 16, 16)), (ROWS, COLS))


@triton.jit
def _dequant(q_ptr, s_ptr, pts_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr, NO: tl.constexpr,
             KO: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    """out[e, :N, :K] = BF16(q s pts) of an [E, N, K] NVFP4 stack into an [E, NO, KO] buffer."""
    pc, pr, e = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    r = (pr * BR + tl.arange(0, BR)).to(tl.int64)
    q = tl.load(q_ptr + e * (N * K // 2) + r[:, None] * (K // 2) + pc * (BC // 2) + tl.arange(0, BC // 2)[None, :])
    s = tl.load(s_ptr + e * (N * K // 16) + r[:, None] * (K // 16) + pc * (BC // 16) + tl.arange(0, BC // 16)[None, :])
    w = _decode(q, s, BR, BC) * tl.load(pts_ptr + e)
    tl.store(out_ptr + e * (NO * KO) + r[:, None] * KO + pc * BC + tl.arange(0, BC)[None, :], w.to(tl.bfloat16))


@triton.jit
def _gather_lora(src_ptr, tok_ptr, fac_ptr, ext_ptr, z_ptr, w_ptr, tile_e_ptr, tile_r_ptr, off_ptr, scaling,
                 H: tl.constexpr, R: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, ZSCALE: tl.constexpr):
    """ext[p] = [src[tok p] | s z_p], z_p = src[tok p] F_e^T (F [E, R, H]); z_ptr gets z (x w[p] with ZSCALE)."""
    pid = tl.program_id(0)
    e = tl.load(tile_e_ptr + pid)
    if e < 0:
        return
    rows = tl.load(tile_r_ptr + pid) + tl.arange(0, BM)
    rmask = rows < tl.load(off_ptr + e + 1)
    rows64 = rows.to(tl.int64)
    src = tl.load(tok_ptr + rows, mask=rmask, other=0).to(tl.int64)
    kk = tl.arange(0, BK)
    ar = tl.arange(0, R)
    fp = fac_ptr + e.to(tl.int64) * (R * H) + ar[:, None] * H + kk[None, :]
    acc = tl.zeros((BM, R), tl.float32)
    for k in range(0, H, BK):
        xt = tl.load(src_ptr + src[:, None] * H + k + kk[None, :], mask=rmask[:, None], other=0.0)
        tl.store(ext_ptr + rows64[:, None] * (H + R) + k + kk[None, :], xt, mask=rmask[:, None])
        acc = tl.dot(xt, tl.trans(tl.load(fp + k)), acc)
    tl.store(ext_ptr + rows64[:, None] * (H + R) + H + ar[None, :], (acc * scaling).to(tl.bfloat16),
             mask=rmask[:, None])
    if ZSCALE:
        acc = acc * tl.load(w_ptr + rows, mask=rmask, other=0.0)[:, None]
    tl.store(z_ptr + rows64[:, None] * R + ar[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _swiglu_fwd(h_ptr, fac_ptr, ext_ptr, z_ptr, tile_e_ptr, tile_r_ptr, off_ptr, scaling,
                I: tl.constexpr, R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """ext[p] = [a_p | s z2_p], a = silu(g) u from h = [g | u], z2 = a A2_e^T (A2 [E, R, I])."""
    pid = tl.program_id(0)
    e = tl.load(tile_e_ptr + pid)
    if e < 0:
        return
    rows = tl.load(tile_r_ptr + pid) + tl.arange(0, BM)
    rmask = rows < tl.load(off_ptr + e + 1)
    rows64 = rows.to(tl.int64)
    nn = tl.arange(0, BN)
    ar = tl.arange(0, R)
    fp = fac_ptr + e.to(tl.int64) * (R * I) + ar[:, None] * I + nn[None, :]
    acc = tl.zeros((BM, R), tl.float32)
    for n in range(0, I, BN):
        hp = h_ptr + rows64[:, None] * (2 * I) + n + nn[None, :]
        g = tl.load(hp, mask=rmask[:, None], other=0.0).to(tl.float32)
        u = tl.load(hp + I, mask=rmask[:, None], other=0.0).to(tl.float32)
        a = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
        tl.store(ext_ptr + rows64[:, None] * (I + R) + n + nn[None, :], a, mask=rmask[:, None])
        acc = tl.dot(a, tl.trans(tl.load(fp + n)), acc)
    tl.store(ext_ptr + rows64[:, None] * (I + R) + I + ar[None, :], (acc * scaling).to(tl.bfloat16),
             mask=rmask[:, None])
    tl.store(z_ptr + rows64[:, None] * R + ar[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _swiglu_bwd(da_ptr, h_ptr, w_ptr, fac_ptr, ext_ptr, z_ptr, dw_ptr, tile_e_ptr, tile_r_ptr, off_ptr, scaling,
                I: tl.constexpr, R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """From da_u (gradient of the unweighted expert output wrt act): dw_p = act_p . da_u_p; da = w da_u; SwiGLU
    backward -> ext[p] = [dh_p | s dz1_p], dz1 = dh B1_e (fac = B1^T [E, R, 2I]); z_ptr gets dz1."""
    pid = tl.program_id(0)
    e = tl.load(tile_e_ptr + pid)
    if e < 0:
        return
    rows = tl.load(tile_r_ptr + pid) + tl.arange(0, BM)
    rmask = rows < tl.load(off_ptr + e + 1)
    rows64 = rows.to(tl.int64)
    nn = tl.arange(0, BN)
    ar = tl.arange(0, R)
    fp = fac_ptr + e.to(tl.int64) * (R * 2 * I) + ar[:, None] * (2 * I) + nn[None, :]
    wr = tl.load(w_ptr + rows, mask=rmask, other=0.0)
    acc = tl.zeros((BM, R), tl.float32)
    dw = tl.zeros((BM,), tl.float32)
    for n in range(0, I, BN):
        hp = h_ptr + rows64[:, None] * (2 * I) + n + nn[None, :]
        g = tl.load(hp, mask=rmask[:, None], other=0.0).to(tl.float32)
        u = tl.load(hp + I, mask=rmask[:, None], other=0.0).to(tl.float32)
        da = tl.load(da_ptr + rows64[:, None] * I + n + nn[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        dw += tl.sum(g * sig * u * da, axis=1)
        da = da * wr[:, None]
        dg = (da * u * sig * (1.0 + g * (1.0 - sig))).to(tl.bfloat16)
        du = (da * g * sig).to(tl.bfloat16)
        ep = ext_ptr + rows64[:, None] * (2 * I + R) + n + nn[None, :]
        tl.store(ep, dg, mask=rmask[:, None])
        tl.store(ep + I, du, mask=rmask[:, None])
        acc = tl.dot(dg, tl.trans(tl.load(fp + n)), acc)
        acc = tl.dot(du, tl.trans(tl.load(fp + I + n)), acc)
    tl.store(ext_ptr + rows64[:, None] * (2 * I + R) + 2 * I + ar[None, :], (acc * scaling).to(tl.bfloat16),
             mask=rmask[:, None])
    tl.store(z_ptr + rows64[:, None] * R + ar[None, :], acc.to(tl.bfloat16), mask=rmask[:, None])
    tl.store(dw_ptr + rows, dw, mask=rmask)


@triton.autotune(configs=[triton.Config({"BN": bn, "BP": bp}, num_warps=w, num_stages=3)
                          for bn in (64, 128) for bp in (64, 128) for w in (4, 8)], key=["NW", "R"])
@triton.jit
def _wgrad(wide_ptr, src_ptr, scale_ptr, narrow_ptr, off_ptr, out_ptr, ld,
           NW: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr,
           GATHER: tl.constexpr, SCALE: tl.constexpr):
    """out[e, n, r] = sum over expert e's rows p of wide[src(p), n] * scale(p) * narrow[p, r] (FP32)."""
    pid_n = tl.program_id(0)
    e = tl.program_id(1)
    start = tl.load(off_ptr + e)
    end = tl.load(off_ptr + e + 1)
    n = pid_n * BN + tl.arange(0, BN)
    ar = tl.arange(0, R)
    acc = tl.zeros((BN, R), tl.float32)
    for p0 in range(start, end, BP):
        p = p0 + tl.arange(0, BP)
        m = p < end
        p64 = p.to(tl.int64)
        if GATHER:
            src = tl.load(src_ptr + p, mask=m, other=0).to(tl.int64)
        else:
            src = p64
        wt = tl.load(wide_ptr + src[:, None] * ld + n[None, :], mask=m[:, None], other=0.0)
        nt = tl.load(narrow_ptr + p64[:, None] * R + ar[None, :], mask=m[:, None], other=0.0)
        if SCALE:
            nt = (nt.to(tl.float32) * tl.load(scale_ptr + p, mask=m, other=0.0)[:, None]).to(tl.bfloat16)
        acc = tl.dot(tl.trans(wt), nt, acc)
    tl.store(out_ptr + e.to(tl.int64) * (NW * R) + n[:, None] * R + ar[None, :], acc)


@triton.jit
def _gather_sum(src_ptr, idx_ptr, w_ptr, out_ptr, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
                BH: tl.constexpr, HAS_W: tl.constexpr):
    """out[t] = sum_k w[t, k] src[idx[t, k]] in FP32, fixed k order (the combine, and dx back to tokens)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    tm = t < T
    hcol = tl.program_id(1) * BH + tl.arange(0, BH)
    t64 = t.to(tl.int64)
    acc = tl.zeros((BT, BH), tl.float32)
    for k in tl.static_range(K):
        p = tl.load(idx_ptr + t64 * K + k, mask=tm, other=0).to(tl.int64)
        v = tl.load(src_ptr + p[:, None] * H + hcol[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        if HAS_W:
            v = v * tl.load(w_ptr + t64 * K + k, mask=tm, other=0.0)[:, None]
        acc += v
    tl.store(out_ptr + t64[:, None] * H + hcol[None, :], acc.to(out_ptr.dtype.element_ty), mask=tm[:, None])


def gather_sum(src, idx, w, T, K, out_dtype):
    H = src.shape[1]
    out = torch.empty(T, H, device=src.device, dtype=out_dtype)
    BT, BH = 16, 256
    _gather_sum[(triton.cdiv(T, BT), triton.cdiv(H, BH))](src, idx, w if w is not None else src, out, T, H=H, K=K,
                                                           BT=BT, BH=BH, HAS_W=w is not None, num_warps=4)
    return out


def wgrad(wide, narrow, offsets, E, *, src=None, scale=None):
    """[E, NW, R] FP32: per expert sum of wide[src(p)]^T (scale(p) narrow[p])."""
    NW, R = wide.shape[1], narrow.shape[1]
    out = torch.empty(E, NW, R, device=wide.device, dtype=torch.float32)
    _wgrad[lambda meta: (NW // meta["BN"], E)](
        wide, src if src is not None else offsets, scale if scale is not None else wide, narrow, offsets, out,
        wide.stride(0), NW=NW, R=R, GATHER=src is not None, SCALE=scale is not None)
    return out


class Weights:
    """Packed components of one frozen NVFP4 stack: qdata u8 [E, N, K/2], E4M3 scale bits [E, N, K/16], FP32 [E]."""

    def __init__(self, w):
        assert not getattr(w, "is_swizzled_scales", False)
        self.q = w.qdata.view(torch.uint8)
        self.s = w.scale.view(torch.uint8)
        E = self.q.shape[0]
        pts = w.per_tensor_scale
        self.pts = (torch.ones(E, device=self.q.device) if pts is None
                    else pts.float().reshape(-1).expand(E).contiguous())
        assert self.q.is_contiguous() and self.s.is_contiguous()
        self.E, self.N, self.K = E, self.q.shape[1], self.q.shape[2] * 2


def decode_into(W, out):
    BR, BC = 32, 128
    _dequant[(W.K // BC, W.N // BR, W.E)](W.q, W.s, W.pts, out, N=W.N, K=W.K, NO=out.shape[1], KO=out.shape[2],
                                          BR=BR, BC=BC, num_warps=4)


def extended(W1, W2, a1, b1, a2, b2):
    """[[W, B], [A, 0]] BF16 buffers for both projections (base from the one-slot cache when it holds this layer)."""
    R = a1.shape[1]
    key = (W1.q.data_ptr(), W2.q.data_ptr())
    use_cache = os.environ.get("RL1_NATIVE_CACHE", "1") != "0"
    if not (use_cache and CACHE["key"] == key):
        shapes = ((W1.E, W1.N + R, W1.K + R), (W2.E, W2.N + R, W2.K + R))
        bufs = [CACHE[k] if use_cache and CACHE[k] is not None and tuple(CACHE[k].shape) == s
                else torch.empty(s, device=W1.q.device, dtype=torch.bfloat16) for k, s in zip(("w1", "w2"), shapes)]
        for W, buf in zip((W1, W2), bufs):
            decode_into(W, buf)
            buf[:, W.N:, W.K:] = 0
        STATS["decodes"] += 1
        CACHE.update(key=key if use_cache else None, w1=bufs[0] if use_cache else None,
                     w2=bufs[1] if use_cache else None)
        w1x, w2x = bufs
    else:
        w1x, w2x = CACHE["w1"], CACHE["w2"]
    for W, buf, a, b in ((W1, w1x, a1, b1), (W2, w2x, a2, b2)):
        buf[:, :W.N, W.K:] = b
        buf[:, W.N:, :W.K] = a
    return w1x, w2x


def schedule(offsets, E, rows_total):
    """Tile -> (expert, first row) for BM-row tiles of every expert's segment, on the GPU (no host sync); the grid is
    the upper bound rows/BM + E and surplus tiles carry expert -1."""
    counts = offsets[1:] - offsets[:-1]
    tiles = (counts + BM - 1) // BM
    cum = torch.cumsum(tiles, 0)
    bound = rows_total // BM + E
    t = torch.arange(bound, device=offsets.device)
    e = torch.searchsorted(cum, t, right=True)
    valid = e < E
    ec = e.clamp(max=E - 1)
    row0 = offsets[ec] + (t - (cum[ec] - tiles[ec])) * BM
    return torch.where(valid, ec, -1).to(torch.int32), row0.to(torch.int32), bound


def factors(A1, B1, A2, B2, E, H, I):
    """Per-expert BF16 LoRA factors from PEFT's layouts (lora_A [R E, in] expert-major rows, lora_B [out, R E]
    expert-minor columns): A [E, R, in], B [E, out, R]."""
    R = A1.shape[0] // E
    a1 = A1.to(torch.bfloat16).reshape(E, R, H)
    b1 = B1.to(torch.bfloat16).reshape(2 * I, R, E).permute(2, 0, 1)
    a2 = A2.to(torch.bfloat16).reshape(E, R, I)
    b2 = B2.to(torch.bfloat16).reshape(H, R, E).permute(2, 0, 1)
    return a1, b1, a2, b2


class NativeMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, top_k_index, top_k_weights, A1, B1, A2, B2, w1, w2, s1, s2, need):
        T, H = hidden.shape
        K = top_k_index.shape[1]
        W1, W2 = Weights(w1), Weights(w2)
        E, I = W1.E, W1.N // 2
        assert W1.K == H and W2.N == H and W2.K == I and W2.E == E
        dev = hidden.device
        a1, b1, a2, b2 = factors(A1, B1, A2, B2, E, H, I)
        R = a1.shape[1]
        x = hidden.to(torch.bfloat16).contiguous()
        flat = top_k_index.reshape(-1)
        sorted_e, order = torch.sort(flat, stable=True)
        tok = (order // K).to(torch.int32)
        rows = flat.numel()
        inv = torch.empty(rows, device=dev, dtype=torch.int32)
        inv[order] = torch.arange(rows, device=dev, dtype=torch.int32)
        wk = top_k_weights.reshape(-1).float().contiguous()
        offsets = torch.searchsorted(sorted_e, torch.arange(E + 1, device=dev, dtype=sorted_e.dtype)).to(torch.int32)
        offs = offsets[1:]
        tile_e, tile_r, bound = schedule(offsets, E, rows)
        w1x, w2x = extended(W1, W2, a1, b1, a2, b2)
        xe = torch.empty(rows, H + R, device=dev, dtype=torch.bfloat16)
        z1 = torch.empty(rows, R, device=dev, dtype=torch.bfloat16)
        a1c = a1.contiguous()
        _gather_lora[(bound,)](x, tok, a1c, xe, z1, wk, tile_e, tile_r, offsets, float(s1), H=H, R=R, BM=BM, BK=128,
                               ZSCALE=False, num_warps=4)
        h = torch._grouped_mm(xe, w1x[:, :2 * I, :].transpose(-2, -1), offs=offs)
        del xe
        ae = torch.empty(rows, I + R, device=dev, dtype=torch.bfloat16)
        z2 = torch.empty(rows, R, device=dev, dtype=torch.bfloat16)
        _swiglu_fwd[(bound,)](h, a2.contiguous(), ae, z2, tile_e, tile_r, offsets, float(s2), I=I, R=R, BM=BM, BN=64,
                              num_warps=4)
        y = torch._grouped_mm(ae, w2x[:, :H, :].transpose(-2, -1), offs=offs)
        out = gather_sum(y, inv, wk, T, K, hidden.dtype)
        STATS["calls"] += 1
        if need:
            ctx.save_for_backward(x, tok, inv, wk[order], offsets, tile_e, tile_r, h, ae, z1, z2, A1, B1, A2, B2)
            ctx.w = (w1, w2)
            ctx.meta = (T, H, I, R, E, K, bound, float(s1), float(s2), hidden.dtype, top_k_weights.dtype)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, tok, inv, wg, offsets, tile_e, tile_r, h, ae, z1, z2, A1, B1, A2, B2 = ctx.saved_tensors
        W1, W2 = (Weights(w) for w in ctx.w)
        T, H, I, R, E, K, bound, s1, s2, hdtype, wdtype = ctx.meta
        rows = tok.numel()
        dev = x.device
        offs = offsets[1:]
        a1, b1, a2, b2 = factors(A1, B1, A2, B2, E, H, I)
        w1x, w2x = extended(W1, W2, a1, b1, a2, b2)
        g = grad.to(torch.bfloat16).contiguous()
        # dy (unweighted, gathered by token) | s dz2_u, dz2_u = g[tok] B2_e; dz2 = w dz2_u for dA2
        dye = torch.empty(rows, H + R, device=dev, dtype=torch.bfloat16)
        dz2 = torch.empty(rows, R, device=dev, dtype=torch.bfloat16)
        _gather_lora[(bound,)](g, tok, b2.transpose(1, 2).contiguous(), dye, dz2, wg, tile_e, tile_r, offsets, s2,
                               H=H, R=R, BM=BM, BK=128, ZSCALE=True, num_warps=4)
        da = torch._grouped_mm(dye, w2x[:, :, :I], offs=offs)
        del dye
        dhe = torch.empty(rows, 2 * I + R, device=dev, dtype=torch.bfloat16)
        dz1 = torch.empty(rows, R, device=dev, dtype=torch.bfloat16)
        dw = torch.empty(rows, device=dev, dtype=torch.float32)
        _swiglu_bwd[(bound,)](da, h, wg, b1.transpose(1, 2).contiguous(), dhe, dz1, dw, tile_e, tile_r, offsets, s1,
                              I=I, R=R, BM=BM, BN=64, num_warps=4)
        del da
        dxr = torch._grouped_mm(dhe, w1x[:, :, :H], offs=offs)
        d_hidden = gather_sum(dxr, inv, None, T, K, hdtype)
        del dxr
        d_weights = dw[inv.long()].reshape(T, K).to(wdtype)
        dA1 = wgrad(x, dz1, offsets, E, src=tok).permute(0, 2, 1).reshape(E * R, H) * s1
        dB1 = wgrad(dhe[:, :2 * I], z1, offsets, E).permute(1, 2, 0).reshape(2 * I, R * E) * s1
        dA2 = wgrad(ae[:, :I], dz2, offsets, E).permute(0, 2, 1).reshape(E * R, I) * s2
        dB2 = wgrad(g, z2, offsets, E, src=tok, scale=wg).permute(1, 2, 0).reshape(H, R * E) * s2
        dA1, dB1, dA2, dB2 = (t.to(f.dtype) for t, f in zip((dA1, dB1, dA2, dB2), (A1, B1, A2, B2)))
        return d_hidden, None, d_weights, dA1, dB1, dA2, dB2, None, None, None, None, None


def supported(w1, w2, lora1, lora2, b1, b2, act, concat, limit, gated):
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import is_nvfp4_param

    return (is_nvfp4_param(w1) and is_nvfp4_param(w2) and lora1 is not None and lora2 is not None and b1 is None
            and b2 is None and act in ("silu", "swiglu") and concat and limit is None and gated
            and not getattr(w1, "is_swizzled_scales", False) and not getattr(w2, "is_swizzled_scales", False)
            and lora1[0].shape[0] // w1.shape[0] == 16)


def moe_forward(hidden_states, top_k_index, top_k_weights, w1, b1, w2, b2, lora1, lora2, num_experts, *, act,
                backend, concat, scaling1, scaling2, limit=None, gated=True):
    """Drop-in for ``nvfp4_lora.grouped_moe_reference_forward``."""
    if not supported(w1, w2, lora1, lora2, b1, b2, act, concat, limit, gated):
        STATS["fallback"] += 1
        return ORIGINAL["fn"](hidden_states, top_k_index, top_k_weights, w1, b1, w2, b2, lora1, lora2, num_experts,
                              act=act, backend=backend, concat=concat, scaling1=scaling1, scaling2=scaling2,
                              limit=limit, gated=gated)
    (A1, B1), (A2, B2) = lora1, lora2
    need = torch.is_grad_enabled() and any(t.requires_grad for t in (hidden_states, top_k_weights, A1, B1, A2, B2))
    return NativeMoE.apply(hidden_states, top_k_index, top_k_weights, A1, B1, A2, B2, w1, w2, scaling1, scaling2,
                           need)


ORIGINAL = {"fn": None}


def install(on=True):
    """Route the sonicmoe NVFP4 experts forward (experts._sonicmoe_nvfp4_forward imports it at call time) here."""
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    if ORIGINAL["fn"] is None:
        ORIGINAL["fn"] = nvfp4_lora.grouped_moe_reference_forward
    nvfp4_lora.grouped_moe_reference_forward = moe_forward if on else ORIGINAL["fn"]
    if not on:
        CACHE.update(key=None, w1=None, w2=None)
    return on
