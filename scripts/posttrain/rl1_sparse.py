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


def reference(q, k, v, picked, scale):
    """Dense masked attention over the same selection (fp32), for tests."""
    keys = k.shape[2]
    mask = torch.zeros((picked.shape[0], keys + 1), dtype=torch.bool, device=q.device)
    mask.scatter_(1, torch.where(picked >= 0, picked, keys), True)
    mask = mask[:, :keys]
    rep = q.shape[1] // k.shape[1]
    kk = k.float().repeat_interleave(rep, dim=1)
    vv = v.float().repeat_interleave(rep, dim=1)
    s = torch.matmul(q.float(), kk.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask[None, None], float("-inf"))
    return torch.matmul(torch.softmax(s, -1), vv)


# ---- batched tails over one shared prefix (rl1 trainer batch_pass) ----
# q [B, heads, L, D]; prefix keys / values kp, vp [groups, P, D] (no gradient, shared by the rows); tail keys / values
# kt, vt [B, groups, L, D]; picked [B, L, S]: positions in the row's key space (prefix 0..P-1, then its own tail).


@triton.jit
def _fwd_pt(Q, KP, VP, KT, VT, PK, O, LSE, scale, L, S, PRE, NH,
            sqb, sqh, sql, skpg, skpn, sktb, sktg, sktn, spb, spl, sob, soh, sol,
            REP: tl.constexpr, REP_PAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    t = tl.program_id(0)
    g = tl.program_id(1)
    b = t // L
    i = t % L
    rows = tl.arange(0, REP_PAD)
    dims = tl.arange(0, D)
    heads = g * REP + rows
    hmask = rows < REP
    q = tl.load(Q + b * sqb + heads[:, None] * sqh + i * sql + dims[None, :], mask=hmask[:, None], other=0.0)
    m = tl.full([REP_PAD], float("-inf"), tl.float32)
    den = tl.zeros([REP_PAD], tl.float32)
    acc = tl.zeros([REP_PAD, D], tl.float32)
    cols = tl.arange(0, BLOCK_N)
    for n0 in range(0, S, BLOCK_N):
        idx = tl.load(PK + b * spb + i * spl + n0 + cols, mask=n0 + cols < S, other=-1)
        valid = idx >= 0
        pre = valid & (idx < PRE)
        own = valid & (idx >= PRE)
        ip = tl.where(pre, idx, 0)
        it = tl.where(own, idx - PRE, 0)
        k = (tl.load(KP + g * skpg + ip[:, None] * skpn + dims[None, :], mask=pre[:, None], other=0.0)
             + tl.load(KT + b * sktb + g * sktg + it[:, None] * sktn + dims[None, :], mask=own[:, None], other=0.0))
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_use[:, None])
        alpha = tl.exp(m - m_use)
        den = den * alpha + tl.sum(p, 1)
        v = (tl.load(VP + g * skpg + ip[:, None] * skpn + dims[None, :], mask=pre[:, None], other=0.0)
             + tl.load(VT + b * sktb + g * sktg + it[:, None] * sktn + dims[None, :], mask=own[:, None], other=0.0))
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    den_safe = tl.where(den > 0, den, 1.0)
    out = acc / den_safe[:, None]
    tl.store(O + b * sob + heads[:, None] * soh + i * sol + dims[None, :], out.to(O.dtype.element_ty), mask=hmask[:, None])
    lse = tl.where(den > 0, tl.where(m == float("-inf"), 0.0, m) + tl.log(den_safe), float("inf"))
    tl.store(LSE + (b * NH + heads) * L + i, lse, mask=hmask)


@triton.jit
def _bwd_pt(Q, KP, VP, KT, VT, PK, DO, LSE, DELTA, DQ, DKT, DVT, scale, L, S, PRE, NH,
            sqb, sqh, sql, skpg, skpn, sktb, sktg, sktn, spb, spl, sdb, sdh, sdl, sdkb, sdkg, sdkn,
            REP: tl.constexpr, REP_PAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    t = tl.program_id(0)
    g = tl.program_id(1)
    b = t // L
    i = t % L
    rows = tl.arange(0, REP_PAD)
    dims = tl.arange(0, D)
    heads = g * REP + rows
    hmask = rows < REP
    q = tl.load(Q + b * sqb + heads[:, None] * sqh + i * sql + dims[None, :], mask=hmask[:, None], other=0.0)
    do = tl.load(DO + b * sdb + heads[:, None] * sdh + i * sdl + dims[None, :], mask=hmask[:, None], other=0.0)
    lse = tl.load(LSE + (b * NH + heads) * L + i, mask=hmask, other=float("inf"))
    delta = tl.load(DELTA + (b * NH + heads) * L + i, mask=hmask, other=0.0)
    dq = tl.zeros([REP_PAD, D], tl.float32)
    cols = tl.arange(0, BLOCK_N)
    for n0 in range(0, S, BLOCK_N):
        idx = tl.load(PK + b * spb + i * spl + n0 + cols, mask=n0 + cols < S, other=-1)
        valid = idx >= 0
        pre = valid & (idx < PRE)
        own = valid & (idx >= PRE)
        ip = tl.where(pre, idx, 0)
        it = tl.where(own, idx - PRE, 0)
        k = (tl.load(KP + g * skpg + ip[:, None] * skpn + dims[None, :], mask=pre[:, None], other=0.0)
             + tl.load(KT + b * sktb + g * sktg + it[:, None] * sktn + dims[None, :], mask=own[:, None], other=0.0))
        v = (tl.load(VP + g * skpg + ip[:, None] * skpn + dims[None, :], mask=pre[:, None], other=0.0)
             + tl.load(VT + b * sktb + g * sktg + it[:, None] * sktn + dims[None, :], mask=own[:, None], other=0.0))
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - lse[:, None])
        p = tl.where(valid[None, :] & hmask[:, None], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * scale
        dq += tl.dot(ds.to(k.dtype), k)
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = tl.dot(tl.trans(p.to(do.dtype)), do)
        offs = b * sdkb + g * sdkg + it[:, None] * sdkn + dims[None, :]
        amask = own[:, None] & (dims[None, :] < D)  # the prefix is a no-gradient cache: only tail keys get gradients
        tl.atomic_add(DKT + offs, dk, mask=amask, sem="relaxed")
        tl.atomic_add(DVT + offs, dv, mask=amask, sem="relaxed")
    tl.store(DQ + b * sqb + heads[:, None] * sqh + i * sql + dims[None, :], dq, mask=hmask[:, None])


class TritonPrefixTailAttention(torch.autograd.Function):
    """``TritonSelectedAttention`` for B rows that share a prefix cache: no copy of the prefix per row, and key /
    value gradients only for the rows' own tails."""

    @staticmethod
    def forward(ctx, q, kp, vp, kt, vt, picked, scale):
        q, kt, vt, picked = q.contiguous(), kt.contiguous(), vt.contiguous(), picked.contiguous()
        batch, heads, length, dim = q.shape
        groups = kt.shape[1]
        rep = heads // groups
        rep_pad = max(16, triton.next_power_of_2(rep))
        if kp is None or kp.shape[1] == 0:
            kp, vp, pre = kt[0], vt[0], 0  # any valid pointer; no index falls below PRE = 0
        else:
            kp, vp, pre = kp.contiguous(), vp.contiguous(), kp.shape[1]
        out = torch.empty_like(q)
        lse = torch.empty((batch, heads, length), dtype=torch.float32, device=q.device)
        _fwd_pt[(batch * length, groups)](
            q, kp, vp, kt, vt, picked, out, lse, float(scale), length, picked.shape[2], pre, heads,
            q.stride(0), q.stride(1), q.stride(2), kp.stride(0), kp.stride(1), kt.stride(0), kt.stride(1), kt.stride(2),
            picked.stride(0), picked.stride(1), out.stride(0), out.stride(1), out.stride(2),
            REP=rep, REP_PAD=rep_pad, D=dim, BLOCK_N=TritonSelectedAttention.BLOCK_N,
            num_warps=TritonSelectedAttention.NUM_WARPS)
        ctx.save_for_backward(q, kp, vp, kt, vt, picked, out, lse)
        ctx.scale, ctx.pre = float(scale), pre
        return out

    @staticmethod
    def backward(ctx, grad):
        q, kp, vp, kt, vt, picked, out, lse = ctx.saved_tensors
        grad = grad.contiguous()
        batch, heads, length, dim = q.shape
        groups = kt.shape[1]
        rep = heads // groups
        rep_pad = max(16, triton.next_power_of_2(rep))
        delta = (grad.float() * out.float()).sum(-1).contiguous()  # [B, heads, L]
        dq = torch.empty((batch, heads, length, dim), dtype=torch.float32, device=q.device)
        dkt = torch.zeros((batch, groups, kt.shape[2], dim), dtype=torch.float32, device=q.device)
        dvt = torch.zeros_like(dkt)
        _bwd_pt[(batch * length, groups)](
            q, kp, vp, kt, vt, picked, grad, lse, delta, dq, dkt, dvt, ctx.scale, length, picked.shape[2], ctx.pre, heads,
            q.stride(0), q.stride(1), q.stride(2), kp.stride(0), kp.stride(1), kt.stride(0), kt.stride(1), kt.stride(2),
            picked.stride(0), picked.stride(1), grad.stride(0), grad.stride(1), grad.stride(2),
            dkt.stride(0), dkt.stride(1), dkt.stride(2),
            REP=rep, REP_PAD=rep_pad, D=dim, BLOCK_N=TritonSelectedAttention.BLOCK_N,
            num_warps=TritonSelectedAttention.NUM_WARPS)
        return dq.to(q.dtype), None, None, dkt.to(kt.dtype), dvt.to(vt.dtype), None, None


def reference_pt(q, kp, vp, kt, vt, picked, scale):
    """Per row: the prefix joined with the row's tail, then ``reference`` (fp32), for tests."""
    rows = []
    for b in range(q.shape[0]):
        k = kt[b:b + 1] if kp is None else torch.cat([kp[None].to(kt.dtype), kt[b:b + 1]], dim=2)
        v = vt[b:b + 1] if vp is None else torch.cat([vp[None].to(vt.dtype), vt[b:b + 1]], dim=2)
        rows.append(reference(q[b:b + 1], k, v, picked[b], scale))
    return torch.cat(rows, dim=0)
