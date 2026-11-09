# SHARED: RL (rl-async) + PTH (sft-heal), one copy since the trunk dedup (24 Oct); both arms' tests cover it.
"""Selected-key attention in Triton for the QSA layers (rl1 trainer, ``RL1_ATTENTION=triton``).

Each query attends to its indexer selection only (at most budget + ratio - 1 = 2,051 key positions, shared by all
heads of a KV group). One program per (query, KV group) holds the group's heads (12, padded to 16) as the tile rows
and walks the selection in blocks of ``BLOCK_N`` gathered keys: online softmax forward, log-sum-exp kept; backward
recomputes the probabilities, accumulates dq in registers and adds dk / dv with fp32 atomics (as DSA's sparse
attention kernels do). Same shapes as ``SelectedAttention``: q [1, heads, L, D], k / v [1, groups, keys, D],
picked [L, S] (int64, -1 padding)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(Q, K, V, P, O, LSE, scale, L, S, KEYS,
         sqh, sql, skg, skn, spl, soh, sol,
         REP: tl.constexpr, REP_PAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    t = tl.program_id(0)
    g = tl.program_id(1)
    rows = tl.arange(0, REP_PAD)
    dims = tl.arange(0, D)
    heads = g * REP + rows
    hmask = rows < REP
    q = tl.load(Q + heads[:, None] * sqh + t * sql + dims[None, :], mask=hmask[:, None], other=0.0)
    m = tl.full([REP_PAD], float("-inf"), tl.float32)
    den = tl.zeros([REP_PAD], tl.float32)
    acc = tl.zeros([REP_PAD, D], tl.float32)
    cols = tl.arange(0, BLOCK_N)
    for n0 in range(0, S, BLOCK_N):
        idx = tl.load(P + t * spl + n0 + cols, mask=n0 + cols < S, other=-1)
        valid = idx >= 0
        safe = tl.where(valid, idx, 0)
        k = tl.load(K + g * skg + safe[:, None] * skn + dims[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_use[:, None])
        alpha = tl.exp(m - m_use)
        den = den * alpha + tl.sum(p, 1)
        v = tl.load(V + g * skg + safe[:, None] * skn + dims[None, :], mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    den_safe = tl.where(den > 0, den, 1.0)
    out = acc / den_safe[:, None]
    tl.store(O + heads[:, None] * soh + t * sol + dims[None, :], out.to(O.dtype.element_ty), mask=hmask[:, None])
    lse = tl.where(den > 0, tl.where(m == float("-inf"), 0.0, m) + tl.log(den_safe), float("inf"))
    tl.store(LSE + heads * L + t, lse, mask=hmask)


@triton.jit
def _bwd(Q, K, V, P, DO, LSE, DELTA, DQ, DK, DV, scale, L, S,
         sqh, sql, skg, skn, spl, sdh, sdl, sdkg, sdkn,
         REP: tl.constexpr, REP_PAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    t = tl.program_id(0)
    g = tl.program_id(1)
    rows = tl.arange(0, REP_PAD)
    dims = tl.arange(0, D)
    heads = g * REP + rows
    hmask = rows < REP
    q = tl.load(Q + heads[:, None] * sqh + t * sql + dims[None, :], mask=hmask[:, None], other=0.0)
    do = tl.load(DO + heads[:, None] * sdh + t * sdl + dims[None, :], mask=hmask[:, None], other=0.0)
    lse = tl.load(LSE + heads * L + t, mask=hmask, other=float("inf"))
    delta = tl.load(DELTA + heads * L + t, mask=hmask, other=0.0)
    dq = tl.zeros([REP_PAD, D], tl.float32)
    cols = tl.arange(0, BLOCK_N)
    for n0 in range(0, S, BLOCK_N):
        idx = tl.load(P + t * spl + n0 + cols, mask=n0 + cols < S, other=-1)
        valid = idx >= 0
        safe = tl.where(valid, idx, 0)
        k = tl.load(K + g * skg + safe[:, None] * skn + dims[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(V + g * skg + safe[:, None] * skn + dims[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - lse[:, None])
        p = tl.where(valid[None, :] & hmask[:, None], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * scale
        dq += tl.dot(ds.to(k.dtype), k)
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = tl.dot(tl.trans(p.to(do.dtype)), do)
        offs = g * sdkg + safe[:, None] * sdkn + dims[None, :]
        amask = valid[:, None] & (dims[None, :] < D)
        tl.atomic_add(DK + offs, dk, mask=amask, sem="relaxed")
        tl.atomic_add(DV + offs, dv, mask=amask, sem="relaxed")
    tl.store(DQ + heads[:, None] * sqh + t * sql + dims[None, :], dq, mask=hmask[:, None])


def _launch_args(q, k, picked):
    heads, length, dim = q.shape[1], q.shape[2], q.shape[3]
    groups = k.shape[1]
    rep = heads // groups
    return heads, length, dim, groups, rep, max(16, triton.next_power_of_2(rep))


class TritonSelectedAttention(torch.autograd.Function):
    BLOCK_N = 64
    NUM_WARPS = 4

    @staticmethod
    def forward(ctx, q, k, v, picked, scale, chunk=None):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        picked = picked.contiguous()
        heads, length, dim, groups, rep, rep_pad = _launch_args(q, k, picked)
        out = torch.empty_like(q)
        lse = torch.empty((heads, length), dtype=torch.float32, device=q.device)
        _fwd[(length, groups)](q[0], k[0], v[0], picked, out[0], lse, float(scale), length, picked.shape[1], k.shape[2],
                               q.stride(1), q.stride(2), k.stride(1), k.stride(2), picked.stride(0),
                               out.stride(1), out.stride(2),
                               REP=rep, REP_PAD=rep_pad, D=dim, BLOCK_N=TritonSelectedAttention.BLOCK_N,
                               num_warps=TritonSelectedAttention.NUM_WARPS)
        ctx.save_for_backward(q, k, v, picked, out, lse)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, picked, out, lse = ctx.saved_tensors
        grad = grad.contiguous()
        heads, length, dim, groups, rep, rep_pad = _launch_args(q, k, picked)
        delta = (grad[0].float() * out[0].float()).sum(-1).contiguous()  # [heads, L]
        dq = torch.empty((1, heads, length, dim), dtype=torch.float32, device=q.device)
        dk = torch.zeros((1, groups, k.shape[2], dim), dtype=torch.float32, device=q.device)
        dv = torch.zeros_like(dk)
        _bwd[(length, groups)](q[0], k[0], v[0], picked, grad[0], lse, delta, dq[0], dk[0], dv[0], ctx.scale,
                               length, picked.shape[1],
                               q.stride(1), q.stride(2), k.stride(1), k.stride(2), picked.stride(0),
                               grad.stride(1), grad.stride(2), dk.stride(1), dk.stride(2),
                               REP=rep, REP_PAD=rep_pad, D=dim, BLOCK_N=TritonSelectedAttention.BLOCK_N,
                               num_warps=TritonSelectedAttention.NUM_WARPS)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None
