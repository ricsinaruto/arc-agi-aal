"""MoE-LoRA for decode-sized batches (MFU-2 stage 2, mfu2-boot c9c9bace8; ARC_MOE_LORA_FAST, installed as
vllm/arc_moe_lora_fast.py by moe_lora_fast_patch; MFU-1 25 Oct: punica wrappers, size cap, capture-safe pointer
tables, ARC_MOE_LORA_BM16).

vLLM applies routed-expert LoRA per MoE layer as moe_lora_align_block_size (pairs sorted by (lora, expert) into
16-row blocks) plus, for w13 and for w2, one fused shrink+expand kernel over a grid of every padded block x slice x
LoRA slot. At decode sizes most of those CTAs are empty and each live one walks K and N in 128-wide steps one after
another, so the layer is latency-bound (~110 us at 64 tokens against a ~15 us weight-read floor).

Here a one-CTA pass lists each (lora, expert)'s pairs (CSR) and the live (lora, expert) keys, shared by w13 and w2,
and persistent kernels walk only live work items:
- ``moe_lora`` (exact): one item = (key, slice, N range); the same tl.dot shrink over 16-row tiles (fp32, cast to
  the weight dtype) and expand (fp32, routed weight, cast, added in the output dtype) as vLLM's one-shot kernel,
  bitwise equal to it.
- ``moe_lora_split``: shrink items (key, slice, K chunk) write fp32 partials, expand items (key, slice, N range)
  sum them in a fixed order; deterministic, differs from vLLM only in the shrink's summation order.

Used only where vLLM itself would take the sorted one-shot path (above its naive-assignment threshold) and up to
``ARC_MOE_LORA_MAX_PAIRS`` (token, expert) pairs; everything else stays on vLLM's path.
"""

import os

import torch

from vllm.triton_utils import tl, triton

MAX_PAIRS = int(os.environ.get("ARC_MOE_LORA_MAX_PAIRS", "10240"))
BLOCK_M = 16
STATS = {"fast": 0}


@triton.jit
def _arc_moe_lists_kernel(topk_ids_ptr, tlm_ptr, enabled_ptr, meta_ptr, P, top_k, E, LE,
                          LEP: tl.constexpr, CH: tl.constexpr):
    """meta = [counts LE | starts LE | cursor LE | live keys LE | n_live 1 | pair lists P] (int32)."""
    offs = tl.arange(0, LEP)
    m = offs < LE
    tl.store(meta_ptr + offs, 0, mask=m)
    tl.store(meta_ptr + 2 * LE + offs, 0, mask=m)
    tl.debug_barrier()
    for c0 in range(0, P, CH):
        idx = c0 + tl.arange(0, CH)
        ok = idx < P
        ids = tl.load(topk_ids_ptr + idx, mask=ok, other=-1)
        lo = tl.load(tlm_ptr + idx // top_k, mask=ok, other=-1)
        en = tl.load(enabled_ptr + tl.maximum(lo, 0), mask=ok, other=0)
        ok = ok & (lo >= 0) & (en != 0) & (ids >= 0) & (ids < E)
        tl.atomic_add(meta_ptr + lo * E + ids, 1, mask=ok, sem="relaxed")
    tl.debug_barrier()
    cnt = tl.load(meta_ptr + offs, mask=m, other=0, cache_modifier=".cg")
    tl.store(meta_ptr + LE + offs, tl.cumsum(cnt, axis=0) - cnt, mask=m)
    live = (cnt > 0).to(tl.int32)
    tl.store(meta_ptr + 3 * LE + tl.cumsum(live, axis=0) - live, offs, mask=m & (cnt > 0))
    tl.store(meta_ptr + 4 * LE, tl.sum(live, axis=0))
    tl.debug_barrier()
    for c0 in range(0, P, CH):
        idx = c0 + tl.arange(0, CH)
        ok = idx < P
        ids = tl.load(topk_ids_ptr + idx, mask=ok, other=-1)
        lo = tl.load(tlm_ptr + idx // top_k, mask=ok, other=-1)
        en = tl.load(enabled_ptr + tl.maximum(lo, 0), mask=ok, other=0)
        ok = ok & (lo >= 0) & (en != 0) & (ids >= 0) & (ids < E)
        key = tl.where(ok, lo * E + ids, 0)
        pos = tl.atomic_add(meta_ptr + 2 * LE + key, 1, mask=ok, sem="relaxed")
        st = tl.load(meta_ptr + LE + key, mask=ok, other=0, cache_modifier=".cg")
        tl.store(meta_ptr + 4 * LE + 1 + st + pos, idx, mask=ok)


@triton.jit
def _arc_moe_lora_kernel(
    x_ptr, out_ptr, A_ptrs, B_ptrs, meta_ptr, topk_w_ptr,
    N, K, E, LE, NSUB, n_split,
    sxm, sxk, sAl, sAe, sAr, sAk, sBl, sBe, sBn, sBr, som, son, slice_off,
    TMF: tl.constexpr, MUL_W: tl.constexpr, BLOCK_M: tl.constexpr, R: tl.constexpr, RANK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
):
    """Persistent; item = (live key, slice x N range). vLLM one-shot arithmetic row for row."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    total = tl.load(meta_ptr + 4 * LE) * NSUB
    offs_r = tl.arange(0, R)
    rmask = offs_r < RANK
    safe_r = tl.where(rmask, offs_r, 0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_j = tl.arange(0, BLOCK_M)
    n_per = tl.cdiv(tl.cdiv(N, n_split), BLOCK_N) * BLOCK_N
    for it in range(pid, total, nprog):
        key = tl.load(meta_ptr + 3 * LE + it // NSUB)
        sub = it % NSUB
        lora = key // E
        e = key % E
        cnt = tl.load(meta_ptr + key)
        start = tl.load(meta_ptr + LE + key)
        slice_id = sub // n_split
        n_lo = (sub % n_split) * n_per
        n_hi = tl.minimum(n_lo + n_per, N)
        cur_A = tl.load(A_ptrs + slice_id).to(tl.pointer_type(out_ptr.dtype.element_ty))
        cur_B = tl.load(B_ptrs + slice_id).to(tl.pointer_type(out_ptr.dtype.element_ty))
        A_base = cur_A + lora * sAl + e * sAe
        B_base = cur_B + lora * sBl + e * sBe
        out_base = out_ptr + slice_id * slice_off
        for g in range(0, cnt, BLOCK_M):
            jm = g + offs_j < cnt
            offs_tok = tl.load(meta_ptr + 4 * LE + 1 + start + g + offs_j, mask=jm, other=0).to(tl.int64)
            x_ptrs = x_ptr + (offs_tok // TMF)[:, None] * sxm + offs_k[None, :] * sxk
            a_ptrs = A_base + offs_k[:, None] * sAk + safe_r[None, :] * sAr
            tmp = tl.zeros((BLOCK_M, R), dtype=tl.float32)
            if EVEN_K:
                for _ in range(0, K, BLOCK_K):
                    xt = tl.load(x_ptrs, mask=jm[:, None], other=0.0)
                    at = tl.load(a_ptrs, mask=rmask[None, :], other=0.0)
                    tmp += tl.dot(xt, at)
                    x_ptrs += BLOCK_K * sxk
                    a_ptrs += BLOCK_K * sAk
            else:
                for kb in range(0, K, BLOCK_K):
                    kmask = offs_k < K - kb
                    xt = tl.load(x_ptrs, mask=jm[:, None] & kmask[None, :], other=0.0)
                    at = tl.load(a_ptrs, mask=kmask[:, None] & rmask[None, :], other=0.0)
                    tmp += tl.dot(xt, at)
                    x_ptrs += BLOCK_K * sxk
                    a_ptrs += BLOCK_K * sAk
            tmp_t = tmp.to(out_ptr.dtype.element_ty)
            if MUL_W:
                w = tl.load(topk_w_ptr + offs_tok, mask=jm, other=0.0).to(tl.float32)
            for n0 in range(n_lo, n_hi, BLOCK_N):
                offs_n = n0 + tl.arange(0, BLOCK_N)
                nmask = offs_n < n_hi
                b = tl.load(B_base + safe_r[:, None] * sBr + offs_n[None, :] * sBn,
                            mask=rmask[:, None] & nmask[None, :], other=0.0)
                acc = tl.dot(tmp_t, b)
                if MUL_W:
                    acc = acc * w[:, None]
                o_ptrs = out_base + offs_tok[:, None] * som + offs_n[None, :] * son
                omask = jm[:, None] & nmask[None, :]
                prev = tl.load(o_ptrs, mask=omask, other=0.0)
                tl.store(o_ptrs, prev + acc.to(out_ptr.dtype.element_ty), mask=omask)


@triton.jit
def _arc_moe_shrink_kernel(
    x_ptr, A_ptrs, meta_ptr, part_ptr, K, E, LE, P, KC, NSUB,
    sxm, sxk, sAl, sAe, sAr, sAk,
    TMF: tl.constexpr, BLOCK_M: tl.constexpr, R: tl.constexpr, RANK: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Persistent; item = (live key, slice x K chunk) -> fp32 partials part[slice, kc, csr position, r]."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    total = tl.load(meta_ptr + 4 * LE) * NSUB
    offs_r = tl.arange(0, R)
    rmask = offs_r < RANK
    safe_r = tl.where(rmask, offs_r, 0)
    offs_j = tl.arange(0, BLOCK_M)
    for it in range(pid, total, nprog):
        key = tl.load(meta_ptr + 3 * LE + it // NSUB)
        sub = it % NSUB
        lora = key // E
        e = key % E
        cnt = tl.load(meta_ptr + key)
        start = tl.load(meta_ptr + LE + key)
        slice_id = sub // KC
        kc = sub % KC
        cur_A = tl.load(A_ptrs + slice_id).to(tl.pointer_type(x_ptr.dtype.element_ty))
        A_base = cur_A + lora * sAl + e * sAe
        offs_k = kc * BLOCK_K + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        at = tl.load(A_base + offs_k[:, None] * sAk + safe_r[None, :] * sAr, mask=kmask[:, None] & rmask[None, :],
                     other=0.0)
        for g in range(0, cnt, BLOCK_M):
            jm = g + offs_j < cnt
            pos = start + g + offs_j
            offs_tok = tl.load(meta_ptr + 4 * LE + 1 + pos, mask=jm, other=0).to(tl.int64)
            xt = tl.load(x_ptr + (offs_tok // TMF)[:, None] * sxm + offs_k[None, :] * sxk,
                         mask=jm[:, None] & kmask[None, :], other=0.0)
            part = tl.dot(xt, at)
            dst = part_ptr + ((slice_id * KC + kc) * P + pos.to(tl.int64))[:, None] * R + offs_r[None, :]
            tl.store(dst, part, mask=jm[:, None])


@triton.jit
def _arc_moe_expand_kernel(
    out_ptr, B_ptrs, meta_ptr, part_ptr, topk_w_ptr, N, E, LE, P, NSUB, n_split,
    sBl, sBe, sBn, sBr, som, son, slice_off,
    KC: tl.constexpr, MUL_W: tl.constexpr, BLOCK_M: tl.constexpr, R: tl.constexpr, RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Persistent; item = (live key, slice x N range): partials summed in K order, cast, expanded, added."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    total = tl.load(meta_ptr + 4 * LE) * NSUB
    offs_r = tl.arange(0, R)
    rmask = offs_r < RANK
    safe_r = tl.where(rmask, offs_r, 0)
    offs_j = tl.arange(0, BLOCK_M)
    n_per = tl.cdiv(tl.cdiv(N, n_split), BLOCK_N) * BLOCK_N
    for it in range(pid, total, nprog):
        key = tl.load(meta_ptr + 3 * LE + it // NSUB)
        sub = it % NSUB
        lora = key // E
        e = key % E
        cnt = tl.load(meta_ptr + key)
        start = tl.load(meta_ptr + LE + key)
        slice_id = sub // n_split
        n_lo = (sub % n_split) * n_per
        n_hi = tl.minimum(n_lo + n_per, N)
        cur_B = tl.load(B_ptrs + slice_id).to(tl.pointer_type(out_ptr.dtype.element_ty))
        B_base = cur_B + lora * sBl + e * sBe
        out_base = out_ptr + slice_id * slice_off
        for g in range(0, cnt, BLOCK_M):
            jm = g + offs_j < cnt
            pos = (start + g + offs_j).to(tl.int64)
            offs_tok = tl.load(meta_ptr + 4 * LE + 1 + pos, mask=jm, other=0).to(tl.int64)
            tmp = tl.zeros((BLOCK_M, R), dtype=tl.float32)
            for kc in tl.static_range(KC):
                tmp += tl.load(part_ptr + ((slice_id * KC + kc) * P + pos)[:, None] * R + offs_r[None, :],
                               mask=jm[:, None], other=0.0)
            tmp_t = tmp.to(out_ptr.dtype.element_ty)
            if MUL_W:
                w = tl.load(topk_w_ptr + offs_tok, mask=jm, other=0.0).to(tl.float32)
            for n0 in range(n_lo, n_hi, BLOCK_N):
                offs_n = n0 + tl.arange(0, BLOCK_N)
                nmask = offs_n < n_hi
                b = tl.load(B_base + safe_r[:, None] * sBr + offs_n[None, :] * sBn,
                            mask=rmask[:, None] & nmask[None, :], other=0.0)
                acc = tl.dot(tmp_t, b)
                if MUL_W:
                    acc = acc * w[:, None]
                o_ptrs = out_base + offs_tok[:, None] * som + offs_n[None, :] * son
                omask = jm[:, None] & nmask[None, :]
                prev = tl.load(o_ptrs, mask=omask, other=0.0)
                tl.store(o_ptrs, prev + acc.to(out_ptr.dtype.element_ty), mask=omask)


_PTRS: dict = {}
_SMS: dict = {}


def _ptrs(ws, device):
    key = tuple(w.data_ptr() for w in ws)
    t = _PTRS.get(key)
    if t is None:  # first sight inside a graph capture: copy from pinned host memory (kept alive with the table)
        host = torch.tensor(key, dtype=torch.uint64)
        if device.type == "cuda":
            host = host.pin_memory()
        t = host.to(device, non_blocking=True)
        _PTRS[key] = t
        _PTRS[("host",) + key] = host
    return t


def _sms(device):
    if device not in _SMS:
        _SMS[device] = (torch.cuda.get_device_properties(device).multi_processor_count
                        if device.type == "cuda" else 4)
    return _SMS[device]


def fast_ok(num_tokens, top_k, num_experts, max_loras, rank, fully_sharded=False, expert_map=None):
    """True where vLLM would run the sorted one-shot kernel and this path covers the shape."""
    pairs = num_tokens * top_k
    return (not fully_sharded and expert_map is None and rank <= 16 and pairs * 8 > num_experts * max_loras
            and pairs <= MAX_PAIRS)


def lists(topk_ids, token_lora_mapping, adapter_enabled, num_loras, num_experts):
    """CSR of (token, expert) pair indices per (lora slot, expert) and the live keys, from topk_ids (tokens, top_k):
    one int32 tensor for w13 and w2."""
    P = topk_ids.numel()
    le = num_loras * num_experts
    meta = torch.empty(4 * le + 1 + P, dtype=torch.int32, device=topk_ids.device)
    _arc_moe_lists_kernel[(1,)](topk_ids.reshape(-1), token_lora_mapping, adapter_enabled, meta, P, topk_ids.shape[1],
                                num_experts, le, LEP=triton.next_power_of_2(le), CH=1024, num_warps=8)
    return meta


# (n_split, BLOCK_N, BLOCK_K, num_warps, num_stages, programs per SM)
DEFAULT = {"w13": (1, 64, 256, 1, 3, 8), "w2": (2, 64, 128, 1, 2, 8)}


def tune(kind):
    env = os.environ.get(f"ARC_MOE_LORA_TUNE_{kind.upper()}")
    return tuple(int(v) for v in env.split(",")) if env else DEFAULT[kind]


def moe_lora(out, x, lora_a, lora_b, meta, topk_weights, top_k, mul_w, kind, cfg=None):
    """Exact path. out: (pairs, slices*N) view of the layer's buffer (updated in place); x: (tokens or pairs, K)."""
    a0, b0 = lora_a[0], lora_b[0]
    L, E, rank, K = a0.shape
    N = b0.shape[2]
    ns = len(lora_a)
    n_split, block_n, block_k, nw, st, per_sm = cfg or tune(kind)
    out2 = out.view(-1, out.shape[-1])
    _arc_moe_lora_kernel[(_sms(x.device) * per_sm,)](
        x, out2, _ptrs(lora_a, x.device), _ptrs(lora_b, x.device), meta, topk_weights.view(-1),
        N, K, E, L * E, ns * n_split, n_split,
        x.stride(0), x.stride(1), a0.stride(0), a0.stride(1), a0.stride(2), a0.stride(3),
        b0.stride(0), b0.stride(1), b0.stride(2), b0.stride(3), out2.stride(0), out2.stride(1), N,
        TMF=1 if mul_w else top_k, MUL_W=mul_w, BLOCK_M=BLOCK_M, R=16, RANK=rank, BLOCK_N=block_n,
        BLOCK_K=block_k, EVEN_K=K % block_k == 0, num_warps=nw, num_stages=st,
    )
    STATS["fast"] += 1


def moe_lora_split(out, x, lora_a, lora_b, meta, topk_weights, top_k, mul_w, cfg):
    """Two launches; cfg = (BLOCK_K, n_split, BLOCK_N, warps_shrink, warps_expand, programs per SM)."""
    a0, b0 = lora_a[0], lora_b[0]
    L, E, rank, K = a0.shape
    N = b0.shape[2]
    ns = len(lora_a)
    bk, n_split, bn, nws, nwx, per_sm = cfg
    kc = triton.cdiv(K, bk)
    P = meta.numel() - 4 * L * E - 1
    part = torch.empty((ns, kc, P, 16), dtype=torch.float32, device=x.device)
    out2 = out.view(-1, out.shape[-1])
    grid = (_sms(x.device) * per_sm,)
    _arc_moe_shrink_kernel[grid](
        x, _ptrs(lora_a, x.device), meta, part, K, E, L * E, P, kc, ns * kc,
        x.stride(0), x.stride(1), a0.stride(0), a0.stride(1), a0.stride(2), a0.stride(3),
        TMF=1 if mul_w else top_k, BLOCK_M=BLOCK_M, R=16, RANK=rank, BLOCK_K=bk, num_warps=nws, num_stages=1)
    _arc_moe_expand_kernel[grid](
        out2, _ptrs(lora_b, x.device), meta, part, topk_weights.view(-1), N, E, L * E, P, ns * n_split, n_split,
        b0.stride(0), b0.stride(1), b0.stride(2), b0.stride(3), out2.stride(0), out2.stride(1), N,
        KC=kc, MUL_W=mul_w, BLOCK_M=BLOCK_M, R=16, RANK=rank, BLOCK_N=bn, num_warps=nwx, num_stages=1)
    STATS["fast"] += 1


# ---- punica_gpu hooks (moe_lora_fast_patch) ----
FAST = os.environ.get("ARC_MOE_LORA_FAST") == "1"
# moel4 (B300, E 320, max_loras 5, cold): stage 2 beats vLLM at M 70 on 1/3/5 live adapters and at M 152 on 3/5, loses
# at M 152 with 1 (130 vs 93 us) and at M 256 (164 vs 156 with BM16). A captured graph fixes the path per size, not
# per live-adapter count, so the cap is by tokens: decode (<= 152 seqs) by default.
FAST_MAX_TOKENS = int(os.environ.get("ARC_MOE_LORA_FAST_MAX_TOKENS", "160"))
BM16 = os.environ.get("ARC_MOE_LORA_BM16") == "1"
BM16_MAX_TOKENS = int(os.environ.get("ARC_MOE_LORA_BM16_MAX_TOKENS", "256"))


class Lists:
    """w13 -> w2 hand-over in the num_tokens_post_padded slot: the pair lists (None: no LoRA token this step)."""
    __slots__ = ("meta",)

    def __init__(self, meta):
        self.meta = meta


def take(num_tokens, top_k, rank, fully_sharded, expert_map, naive, add_inputs):
    """Stage 2 where vLLM would run align + its sorted one-shot kernel, up to FAST_MAX_TOKENS tokens."""
    return (FAST and not fully_sharded and expert_map is None and not naive and add_inputs and rank <= 16
            and 0 < num_tokens <= FAST_MAX_TOKENS and num_tokens * top_k <= MAX_PAIRS)


def bm16(shrink_config, expand_config, num_tokens):
    """ARC_MOE_LORA_BM16: vLLM's own align + one-shot with 16-row blocks (moel4 M 70 / 5 adapters 173 -> 143 us);
    MMA rows are independent, so the result is bitwise the same."""
    if BM16 and num_tokens <= BM16_MAX_TOKENS and int(shrink_config.get("BLOCK_SIZE_M") or 64) > 16:
        shrink_config = {**shrink_config, "BLOCK_SIZE_M": 16}
        expand_config = {**expand_config, "BLOCK_SIZE_M": 16}
    return shrink_config, expand_config


def _meta(punica, n, token_lora_mapping):
    m = punica.token_mapping_meta.meta_args(n, punica.lora_config.specialize_active_lora)
    return (m[0] if token_lora_mapping is None else token_lora_mapping), bool(m[5].item())


def w13(punica, y, x, lora_a, lora_b, topk_ids, topk_weights, top_k_num, top_k, adapter_enabled, token_lora_mapping):
    """add_lora_w13's align + add_lora_fused_moe; returns add_lora_w2's hand-over tuple."""
    tlm, no_lora = _meta(punica, x.size(0), token_lora_mapping)
    flat = topk_ids.reshape(-1)
    if no_lora:
        return None, flat, Lists(None), tlm
    meta = lists(topk_ids, tlm, adapter_enabled, lora_a[0].shape[0], lora_a[0].shape[1])
    moe_lora(y.view(-1, top_k_num, y.shape[-1]), x, lora_a, lora_b, meta, topk_weights, top_k, False, "w13")
    return None, flat, Lists(meta), tlm


def w2(y, x, lora_a, lora_b, topk_weights, hand_over, top_k):
    if hand_over.meta is not None:
        moe_lora(y, x, lora_a, lora_b, hand_over.meta, topk_weights, top_k, True, "w2")
