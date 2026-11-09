"""MFU-1 (23 Oct): the QSA indexer's block pick as one exact radix select per row, in place of
``scores.masked_fill(~valid, -inf).topk(k)`` (13% of the trainer's GPU time in m1p1, 35% of the tail forward).

``topk_rows(scores, k, limit)``: per row r the indices of the k largest of ``scores[r]`` with columns >= ``limit[r]``
as -inf, as an ascending set. It selects exactly what torch.topk selects on CUDA:
- the order is torch's radix key (``TopKTypeConfig<float>::convert``): -0.0 ranks below +0.0, NaN above everything;
- every element above the k-th largest key T is taken, and the elements equal to T are taken in index order (torch's
  gatherTopK writes ties block by block, in index order within a block), so ties at the k-th place go to the lower
  indices.
The order inside a row is not torch's (it returns by value); the union-attention kernel sorts its tiles' blocks
itself, so the trainer uses this only with union_attention.

Per row (one program), "bisect" (rows up to 32,768 wide): the row is read once into registers, T is built bit by bit
(32 compare-and-count rounds), and one pass writes the picks in index order. "radix" (any width; slower on GPU): four
8-bit histogram passes over reloads of the row, then the write pass."""
import torch
import triton
import triton.language as tl

NEG_INF_KEY = 0x007FFFFF  # torch's key of -inf


def torch_key(scores):
    """torch's radix key of fp32 values, as int64 (host reference)."""
    b = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = torch.where(b >= 0x80000000, b ^ 0xFFFFFFFF, b ^ 0x80000000)
    return torch.where(scores == scores, key, torch.full_like(key, 0xFFFFFFFF))


def reference_topk(scores, k, limit=None):
    """The selection rule on the host: key descending, ties by index ascending, first k; columns >= ``limit`` as -inf.
    Returns the picks as ascending index rows (int64 [rows, k])."""
    rows, n = scores.shape
    key = torch_key(scores.float().cpu())
    if limit is not None:
        cols = torch.arange(n)[None, :]
        key = torch.where(cols >= limit.cpu().long()[:, None], torch.full_like(key, NEG_INF_KEY), key)
    order = key * (n + 1) + (n - torch.arange(n))[None, :]  # larger key first, then the lower index
    return torch.sort(torch.topk(order, k, dim=1).indices, dim=1).values


@triton.jit
def _key(x):
    b = x.to(tl.uint32, bitcast=True)
    key = tl.where((b & 0x80000000) != 0, b ^ 0xFFFFFFFF, b ^ 0x80000000)
    return tl.where(x == x, key, 0xFFFFFFFF)


@triton.jit
def _row_keys(s_ptr, row, stride, cols, n, lim):
    inside = cols < n
    x = tl.load(s_ptr + row * stride + cols, mask=inside, other=0.0)
    key = tl.where(cols < lim, _key(x), 0x007FFFFF)  # torch's key of -inf
    return key.to(tl.uint32), inside


@triton.jit
def _topk_rows(s_ptr, lim_ptr, out_ptr, n, stride, K, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lim = tl.load(lim_ptr + row)
    need = K
    prefix = tl.full((), 0, tl.uint32)
    pmask = tl.full((), 0, tl.uint32)
    bins = tl.arange(0, 512)
    for p in tl.static_range(4):  # 8-bit digits, top first
        hist = tl.zeros((512,), dtype=tl.int32)
        for start in range(0, n, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            key, inside = _row_keys(s_ptr, row, stride, cols, n, lim)
            hit = inside & ((key & pmask) == prefix)
            digit = tl.where(hit, ((key >> (24 - 8 * p)) & 0xFF).to(tl.int32), 511)  # misses: bin 511
            hist += tl.histogram(digit, 512)
        h = tl.where(bins < 256, hist, 0)
        ge = tl.sum(h, axis=0) - tl.cumsum(h, axis=0) + h  # matching keys with digit >= bin
        d = tl.max(tl.where((ge >= need) & (bins < 256), bins, -1), axis=0)
        need -= tl.sum(tl.where(bins == d, ge - h, 0), axis=0)  # keys above digit d are all taken
        prefix = prefix | (d.to(tl.uint32) << (24 - 8 * p))
        pmask = pmask | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    wrote = tl.zeros((), dtype=tl.int32)
    ties = tl.zeros((), dtype=tl.int32)
    for start in range(0, n, BLOCK):  # T = prefix: all keys above it, the first ``need`` equal to it
        cols = start + tl.arange(0, BLOCK)
        key, inside = _row_keys(s_ptr, row, stride, cols, n, lim)
        eq = inside & (key == prefix)
        rank = ties + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        take = inside & ((key > prefix) | (eq & (rank < need)))
        slot = wrote + tl.cumsum(take.to(tl.int32), axis=0) - 1
        tl.store(out_ptr + row * K + slot, cols.to(tl.int64), mask=take)
        wrote += tl.sum(take.to(tl.int32), axis=0)
        ties += tl.sum(eq.to(tl.int32), axis=0)


@triton.jit
def _topk_rows_bisect(s_ptr, lim_ptr, out_ptr, n, stride, K, BLOCK: tl.constexpr):
    """The whole row in registers (BLOCK >= n; padding key 0, below every real key, -inf's 0x007FFFFF included);
    the k-th largest key T built bit by bit from the top (keep a bit when >= K keys reach it: 32 compare-and-sum
    rounds, no histogram), then one ordered write pass."""
    row = tl.program_id(0).to(tl.int64)
    lim = tl.load(lim_ptr + row)
    cols = tl.arange(0, BLOCK)
    key, inside = _row_keys(s_ptr, row, stride, cols, n, lim)
    key = tl.where(inside, key, tl.full((), 0, tl.uint32))
    t = tl.full((), 0, tl.uint32)
    for b in tl.static_range(32):
        cand = t | (tl.full((), 1, tl.uint32) << (31 - b))
        if tl.sum((key >= cand).to(tl.int32), axis=0) >= K:
            t = cand
    above = key > t
    need = K - tl.sum(above.to(tl.int32), axis=0)
    eq = key == t
    rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
    take = above | (eq & (rank < need))
    slot = tl.cumsum(take.to(tl.int32), axis=0) - 1
    tl.store(out_ptr + row * K + slot, cols.to(tl.int64), mask=take)


@triton.jit
def _kth_key(key, K):
    """The K-th largest of ``key`` (uint32, padding 0) bit by bit; 0 when fewer than K keys are nonzero."""
    t = tl.full((), 0, tl.uint32)
    for b in tl.static_range(32):
        cand = t | (tl.full((), 1, tl.uint32) << (31 - b))
        if tl.sum((key >= cand).to(tl.int32), axis=0) >= K:
            t = cand
    return t


@triton.jit
def _first_k(out_ptr, row, K, KB: tl.constexpr):
    for s0 in range(0, K, KB):
        c = s0 + tl.arange(0, KB)
        tl.store(out_ptr + row * K + c, c.to(tl.int64), mask=c < K)


@triton.jit
def _select_cand(ck_ptr, cc_ptr, out_ptr, row, cnt, K, CAND: tl.constexpr):
    c = tl.arange(0, CAND)
    ck = tl.load(ck_ptr + row * CAND + c, mask=c < cnt, other=0).to(tl.uint32)
    t = _kth_key(ck, K)
    above = ck > t
    need = K - tl.sum(above.to(tl.int32), axis=0)
    eq = ck == t
    rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
    take = above | (eq & (rank < need))
    slot = tl.cumsum(take.to(tl.int32), axis=0) - 1
    col = tl.load(cc_ptr + row * CAND + c, mask=take, other=0)
    tl.store(out_ptr + row * K + slot, col.to(tl.int64), mask=take)


@triton.jit
def _radix_valid(s_ptr, out_ptr, row, stride, v, K, CHUNK: tl.constexpr):
    """The 8-bit radix select over the valid columns [0, v) (v > K: the -inf columns are never needed)."""
    need = K
    prefix = tl.full((), 0, tl.uint32)
    pmask = tl.full((), 0, tl.uint32)
    bins = tl.arange(0, 512)
    for p in tl.static_range(4):
        hist = tl.zeros((512,), dtype=tl.int32)
        for start in range(0, v, CHUNK):
            cols = start + tl.arange(0, CHUNK)
            key, inside = _row_keys(s_ptr, row, stride, cols, v, v)
            hit = inside & ((key & pmask) == prefix)
            digit = tl.where(hit, ((key >> (24 - 8 * p)) & 0xFF).to(tl.int32), 511)
            hist += tl.histogram(digit, 512)
        h = tl.where(bins < 256, hist, 0)
        ge = tl.sum(h, axis=0) - tl.cumsum(h, axis=0) + h
        d = tl.max(tl.where((ge >= need) & (bins < 256), bins, -1), axis=0)
        need -= tl.sum(tl.where(bins == d, ge - h, 0), axis=0)
        prefix = prefix | (d.to(tl.uint32) << (24 - 8 * p))
        pmask = pmask | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    wrote = tl.zeros((), dtype=tl.int32)
    ties = tl.zeros((), dtype=tl.int32)
    for start in range(0, v, CHUNK):
        cols = start + tl.arange(0, CHUNK)
        key, inside = _row_keys(s_ptr, row, stride, cols, v, v)
        eq = inside & (key == prefix)
        rank = ties + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        take = inside & ((key > prefix) | (eq & (rank < need)))
        slot = wrote + tl.cumsum(take.to(tl.int32), axis=0) - 1
        tl.store(out_ptr + row * K + slot, cols.to(tl.int64), mask=take)
        wrote += tl.sum(take.to(tl.int32), axis=0)
        ties += tl.sum(eq.to(tl.int32), axis=0)


@triton.jit
def _filter_row(s_ptr, out_ptr, ck_ptr, cc_ptr, row, stride, v, K, STEP: tl.constexpr, SBLOCK: tl.constexpr,
                CAND: tl.constexpr, CHUNK: tl.constexpr):
    sc = tl.arange(0, SBLOCK) * STEP
    sk, sin = _row_keys(s_ptr, row, stride, sc, v, v)
    t = _kth_key(tl.where(sin, sk, tl.full((), 0, tl.uint32)), K)
    cnt = tl.zeros((), dtype=tl.int32)
    for start in range(0, v, CHUNK):
        cols = start + tl.arange(0, CHUNK)
        key, inside = _row_keys(s_ptr, row, stride, cols, v, v)
        take = inside & (key >= t)
        slot = cnt + tl.cumsum(take.to(tl.int32), axis=0) - 1
        keep = take & (slot < CAND)
        tl.store(ck_ptr + row * CAND + slot, key, mask=keep)
        tl.store(cc_ptr + row * CAND + slot, cols, mask=keep)
        cnt += tl.sum(take.to(tl.int32), axis=0)
    tl.debug_barrier()  # the compacted rows are read back by other threads of this program
    if cnt <= CAND:
        _select_cand(ck_ptr, cc_ptr, out_ptr, row, cnt, K, CAND)
    else:
        _radix_valid(s_ptr, out_ptr, row, stride, v, K, CHUNK)


@triton.jit
def _topk_rows_filter(s_ptr, lim_ptr, out_ptr, ck_ptr, cc_ptr, n, stride, K, STEP: tl.constexpr, SBLOCK: tl.constexpr,
                      CAND: tl.constexpr, CHUNK: tl.constexpr, KB: tl.constexpr):
    """Wide rows (24 Oct): a lower bound t of the k-th key from a strided sample (the K-th largest of every STEP-th
    valid column: at least K keys of the row reach it), the valid columns with key >= t compacted in index order
    (they hold every pick, ties included), then the exact select on them; more than CAND of them: the 8-bit radix on
    the row. Rows with <= K valid columns take columns 0..K-1 (the valid ones and the lowest-index -inf ties)."""
    row = tl.program_id(0).to(tl.int64)
    v = tl.minimum(tl.load(lim_ptr + row), n).to(tl.int32)
    if v <= K:
        _first_k(out_ptr, row, K, KB)
    else:
        _filter_row(s_ptr, out_ptr, ck_ptr, cc_ptr, row, stride, v, K, STEP, SBLOCK, CAND, CHUNK)


# method "filter" knobs; B300 mfu1-tkf-b3 (1,024 rows, k 512) ms filter/torch: 12K .136/.220, 20K .205/.394,
# 25K .205/.524, 32K .279/.597, 40K .372/.755 (step 8); 0 mismatched rows on real/tied/zero/negative/skewed scores
FILTER = {"step": 4, "cand": 4096, "chunk": 2048, "warps": 8, "wide_step": 8, "wide_cand": 8192}
RESIDENT_MAX = 32768  # widest row "bisect" holds in registers
WIDE = {"method": "torch"}  # rows wider than BISECT_DEFAULT_MAX: "torch" or "filter" (trainer cmd fast_topk_wide)
BISECT_DEFAULT_MAX = 8192  # B300 (mfu1-topk-b3): bisect 0.047 vs torch 0.126 ms at 4K blocks, even at 12K, slower from 20K


def topk_rows(scores, k, limit=None, block=4096, num_warps=None, method=None):
    """Exact top-``k`` column indices per row of fp32 ``scores`` [rows, n] (columns >= ``limit[r]`` as -inf), as
    ascending rows (int64 [rows, k]); torch.topk's selection on CUDA, ties to the lower index; ``k <= n``.
    ``method``: "bisect" (rows up to RESIDENT_MAX wide; the default up to BISECT_DEFAULT_MAX), "radix" (the 8-bit
    histogram passes; slower than torch on B300), "filter" (sample bound + compaction + select, any width) or "torch"
    (masked_fill + topk + sort: the default for wider rows)."""
    rows, n = scores.shape
    if scores.dtype != torch.float32 or scores.stride(1) != 1:
        raise ValueError("topk_rows: fp32 scores with unit column stride")
    if not 0 < k <= n:
        raise ValueError(f"topk_rows: k {k} of {n}")
    if limit is None:
        limit = torch.full((rows,), n, dtype=torch.int64, device=scores.device)
    limit = limit.to(device=scores.device, dtype=torch.int64).clamp(max=n).contiguous()
    width = triton.next_power_of_2(n)
    method = method or ("bisect" if width <= BISECT_DEFAULT_MAX else WIDE["method"])
    if method == "torch":
        masked = scores.masked_fill(torch.arange(n, device=scores.device)[None, :] >= limit[:, None], float("-inf"))
        return torch.sort(masked.topk(k, dim=-1, sorted=False).indices, dim=1).values
    out = torch.empty((rows, k), dtype=torch.int64, device=scores.device)
    if method == "bisect":
        if width > RESIDENT_MAX:
            raise ValueError(f"topk_rows: bisect holds rows up to {RESIDENT_MAX} wide, not {n}")
        warps = num_warps or max(4, min(32, width // 1024))
        _topk_rows_bisect[(rows,)](scores, limit, out, n, scores.stride(0), k, BLOCK=width, num_warps=warps)
        return out
    if method == "filter":
        step, cand = int(FILTER["step"]), int(FILTER["cand"])
        if FILTER.get("wide_step") and n > 32768:  # B300 (mfu1-tkf-b3): step 8 / 8,192 candidates win at 40K blocks
            step, cand = int(FILTER["wide_step"]), int(FILTER["wide_cand"])
        if cand < k:
            raise ValueError(f"topk_rows: filter buffer {cand} below k {k}")
        ck = torch.empty((rows, cand), dtype=torch.int32, device=scores.device)  # keys as int32 bits
        cc = torch.empty((rows, cand), dtype=torch.int32, device=scores.device)
        _topk_rows_filter[(rows,)](scores, limit, out, ck, cc, n, scores.stride(0), k, STEP=step,
                                   SBLOCK=triton.next_power_of_2(triton.cdiv(n, step)), CAND=cand,
                                   CHUNK=min(int(FILTER["chunk"]), width), KB=min(1024, triton.next_power_of_2(k)),
                                   num_warps=num_warps or int(FILTER["warps"]))
        return out
    _topk_rows[(rows,)](scores, limit, out, n, scores.stride(0), k, BLOCK=min(block, width), num_warps=num_warps or 8)
    return out
