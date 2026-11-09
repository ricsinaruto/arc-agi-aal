"""Fused MoE routing for decode batches (kernel profile project 2, 30 September; docs/validation-20260930/kernel-profile.md).

At 40 decode streams every MoE layer spends ~13 us in four launches between the router GEMM and the first Marlin GEMM:
``topkGating`` (softmax, top-10, renormalize; 5.9 us), ``moe_align_block_size`` (4.0 us), ``count_and_sort_expert_tokens``
(1.3 us) and a fill (1.5 us), 0.63 ms per step over 48 layers, plus four graph-node gaps. ``route_align`` does all of it
in one single-CTA Triton launch for M <= 64 tokens (every decode CUDA-graph size of the blocks3 serving):

1. softmax of the router logits in FP32 over the E experts (the pruned 448), top-k by k successive arg-max passes (the
   first index wins a tie), and the renormalized top-k weights (``p_i / sum_topk p``), as vLLM's ``topk_softmax`` with
   ``renormalize=True``;
2. the Marlin block alignment of ``moe_align_block_size``: per-expert counts (``tl.histogram``), counts padded to the
   block size, their exclusive prefix sum as expert offsets, ``num_tokens_post_padded`` = the padded total,
   ``expert_ids`` per block in ascending expert order, and ``sorted_token_ids`` = the flat (token x k) pair indices
   grouped by expert, padding slots holding ``M x top_k`` (the "no token" id Marlin skips).

Differences from the pinned path, by design:
- the order of pairs *within* one expert's rows comes from an atomic cursor (not deterministic). Marlin computes every
  row's product independently and ``moe_sum`` reduces by (token, k), so the MoE output does not depend on it;
- the renormalized weights may differ from ``topk_softmax`` in the last FP32 bit (a different reduction order), which
  can move a BF16 output by one rounding: covered by the drift gate, not bitwise.

The patch that routes the pinned router and ``fused_marlin_moe`` through this kernel is not built yet (section 4 of the
doc: the integration touches ``fused_topk_router`` and ``experts/marlin_moe.py``, hash-guarded like the dense overlay).
"""

import torch
import triton
import triton.language as tl

MAX_TOKENS = 64


@triton.jit
def _route_align_kernel(logits_ptr, weights_ptr, ids_ptr, sorted_ptr, expert_ids_ptr, ntpp_ptr, cursor_ptr,
                        M, stride_m, E, MAX_PADDED, MAX_BLOCKS,
                        TOP_K: tl.constexpr, K_PAD: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr, FILL: tl.constexpr,
                        BLOCKS_PER_EXPERT: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_E)
    kcols = tl.arange(0, K_PAD)
    row_ok = rows < M
    mask = row_ok[:, None] & (cols[None, :] < E)
    x = tl.load(logits_ptr + rows[:, None] * stride_m + cols[None, :], mask=mask, other=-float("inf")).to(tl.float32)
    x = tl.where(row_ok[:, None], x, 0.0)  # padded rows: finite, never stored
    mx = tl.max(x, axis=1)
    e = tl.exp(x - mx[:, None])
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=1)
    p = e / tl.where(row_ok, denom, 1.0)[:, None]  # padded rows: no 0 / 0
    work = tl.where(mask, p, -1.0)
    top_w = tl.zeros((BLOCK_M, K_PAD), dtype=tl.float32)
    top_i = tl.zeros((BLOCK_M, K_PAD), dtype=tl.int32)
    for k in tl.static_range(TOP_K):
        best = tl.max(work, axis=1)
        idx = tl.argmax(work, axis=1).to(tl.int32)  # first maximum
        top_w = tl.where(kcols[None, :] == k, best[:, None], top_w)
        top_i = tl.where(kcols[None, :] == k, idx[:, None], top_i)
        work = tl.where(cols[None, :] == idx[:, None], -1.0, work)
    kmask = row_ok[:, None] & (kcols[None, :] < TOP_K)
    wsum = tl.sum(tl.where(kmask, top_w, 0.0), axis=1)
    top_w = top_w / tl.where(row_ok, wsum, 1.0)[:, None]
    flat = rows[:, None] * TOP_K + kcols[None, :]
    tl.store(weights_ptr + flat, top_w, mask=kmask)
    tl.store(ids_ptr + flat, top_i, mask=kmask)

    # Alignment. Invalid (row, k) slots count in the spare bin E (< BLOCK_E), which is ignored.
    bins = tl.where(kmask, top_i, E)
    counts = tl.histogram(tl.reshape(bins, (BLOCK_M * K_PAD,)), BLOCK_E)
    counts = tl.where(cols < E, counts, 0)
    padded = (counts + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    offsets = tl.cumsum(padded, axis=0) - padded
    total = tl.sum(padded, axis=0)
    tl.store(ntpp_ptr, total)
    # Padding everywhere first, then the pairs over it.
    for start in range(0, MAX_PADDED, 1024):
        span = start + tl.arange(0, 1024)
        tl.store(sorted_ptr + span, tl.full((1024,), FILL, tl.int32), mask=span < MAX_PADDED)
    for start in range(0, MAX_BLOCKS, 1024):
        span = start + tl.arange(0, 1024)
        tl.store(expert_ids_ptr + span, tl.full((1024,), -1, tl.int32), mask=span < MAX_BLOCKS)
    tl.store(cursor_ptr + cols, offsets, mask=cols < E)
    tl.debug_barrier()
    for j in tl.static_range(BLOCKS_PER_EXPERT):
        blk = offsets // BLOCK_SIZE + j
        tl.store(expert_ids_ptr + blk, cols, mask=(cols < E) & (j * BLOCK_SIZE < padded))
    position = tl.atomic_add(cursor_ptr + top_i, tl.full((BLOCK_M, K_PAD), 1, tl.int32), mask=kmask)
    tl.store(sorted_ptr + position, flat.to(tl.int32), mask=kmask)


def buffers(m, top_k, num_experts, block_size, device):
    """Output shapes of ``moe_align_block_size`` for ``m`` tokens (the pinned sizing rule)."""
    numel = m * top_k
    max_padded = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_padded = min(numel * block_size, max_padded)
    max_blocks = triton.cdiv(max_padded, block_size)
    return (torch.empty((m, top_k), dtype=torch.float32, device=device),
            torch.empty((m, top_k), dtype=torch.int32, device=device),
            torch.empty((max_padded,), dtype=torch.int32, device=device),
            torch.empty((max_blocks,), dtype=torch.int32, device=device),
            torch.empty((1,), dtype=torch.int32, device=device),
            torch.empty((num_experts,), dtype=torch.int32, device=device))


def route_align(logits, top_k, block_size, cursor=None):
    """(topk_weights FP32 [M, k], topk_ids int32 [M, k], sorted_token_ids, expert_ids, num_tokens_post_padded) for
    router logits [M, E], M <= 64."""
    m, num_experts = logits.shape
    if not 1 <= m <= MAX_TOKENS:
        raise ValueError(f"route_align serves 1..{MAX_TOKENS} tokens, got {m}")
    weights, ids, sorted_ids, expert_ids, ntpp, scratch = buffers(m, top_k, num_experts, block_size, logits.device)
    cursor = scratch if cursor is None else cursor
    block_m = max(16, triton.next_power_of_2(m))
    block_e = triton.next_power_of_2(num_experts + 1)  # one spare bin for the invalid slots
    _route_align_kernel[(1,)](
        logits, weights, ids, sorted_ids, expert_ids, ntpp, cursor, m, logits.stride(0), num_experts,
        sorted_ids.numel(), expert_ids.numel(), TOP_K=top_k, K_PAD=triton.next_power_of_2(top_k),
        BLOCK_SIZE=block_size, BLOCK_M=block_m, BLOCK_E=block_e, FILL=m * top_k,
        BLOCKS_PER_EXPERT=triton.cdiv(m, block_size), num_warps=8)
    return weights, ids, sorted_ids, expert_ids, ntpp


def reference(logits, top_k, block_size):
    """The pinned semantics in torch: ``topk_softmax`` (renormalize) and ``moe_align_block_size``."""
    m, num_experts = logits.shape
    p = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(p, top_k, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    flat = ids.reshape(-1)
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    offsets = torch.cumsum(padded, 0) - padded
    total = int(padded.sum())
    sorted_ids = torch.full((total,), m * top_k, dtype=torch.int32)
    expert_ids = torch.empty((total // block_size,), dtype=torch.int32)
    for expert in range(num_experts):
        pairs = (flat == expert).nonzero().flatten()
        start = int(offsets[expert])
        sorted_ids[start:start + len(pairs)] = pairs.to(torch.int32)
        expert_ids[start // block_size:(start + int(padded[expert])) // block_size] = expert
    return w, ids.to(torch.int32), sorted_ids, expert_ids, total
