"""Fused dense LoRA for decode-sized batches (serving_fast.lora_fused_dense; installed as vllm/arc_lora_fused.py).

vLLM applies a linear layer's LoRA as three kernels: zero an fp32 buffer, shrink (x @ A^T * scale, split-K atomics) and
expand (buffer cast to the weight dtype, @ B^T, cast, added to the output). At decode batch sizes these are a few
microseconds of work each and Flash runs ~250 such layers per step, so the step pays for launches and tails. Here one
kernel per layer does both products: each program takes one LoRA's token rows (the same sorted-token metadata as the
punica kernels), recomputes the rank-r shrink over K for its rows (cheap at r=16) and writes one output tile. Same
casts as the reference (shrink accumulated in fp32, scaled, cast to the weight dtype; expand accumulated in fp32,
cast, then added to the output in the output dtype); only the shrink's summation order differs.

Batches above ``ARC_LORA_FUSED_MAX_M`` tokens (default 256: prefill chunks, where the per-tile shrink recompute stops
being cheap) and anything the fused kernel does not cover go to vLLM's own shrink/expand, unchanged.
"""

import os

import torch

from vllm.triton_utils import tl, triton

MAX_M = int(os.environ.get("ARC_LORA_FUSED_MAX_M", "256"))
BLOCK_M = 16
STATS = {"fused": 0, "fallback": 0}


@triton.jit
def _arc_lora_fused_kernel(
    x_ptr, y_ptr,
    a0_ptr, a1_ptr, a2_ptr, b0_ptr, b1_ptr, b2_ptr,
    N0, N1, N2, T0, T1, K,
    sorted_ptr, ntok_ptr, start_ptr, ids_ptr,
    scaling,
    sx0, sx1, sy0, sy1,
    sa0, sa1, sa2,
    sb0_0, sb0_1, sb0_2, sb1, sb2,
    R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    lora_idx = tl.program_id(2)
    lora_id = tl.load(ids_ptr + lora_idx)
    if lora_id == -1:
        return
    m_size = tl.load(ntok_ptr + lora_idx)
    m_off = pid_m * BLOCK_M
    if m_off >= m_size:
        return
    if pid_n < T0:
        a_ptr = a0_ptr
        b_ptr = b0_ptr
        n_size = N0
        col0 = 0
        tile = pid_n
        sb0 = sb0_0
    elif pid_n < T1:
        a_ptr = a1_ptr
        b_ptr = b1_ptr
        n_size = N1
        col0 = N0
        tile = pid_n - T0
        sb0 = sb0_1
    else:
        a_ptr = a2_ptr
        b_ptr = b2_ptr
        n_size = N2
        col0 = N0 + N1
        tile = pid_n - T1
        sb0 = sb0_2
    m_len = tl.minimum(BLOCK_M, m_size - m_off)
    offs_m = tl.arange(0, BLOCK_M)
    first = tl.load(start_ptr + lora_idx)
    ram = tl.load(sorted_ptr + first + m_off + offs_m % m_len)

    offs_r = tl.arange(0, R)
    offs_k = tl.arange(0, BLOCK_K)
    x_rows = x_ptr + ram[:, None] * sx0
    a_base = a_ptr + lora_id * sa0 + offs_r[None, :] * sa1
    acc = tl.zeros((BLOCK_M, R), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        if EVEN_K:
            xk = tl.load(x_rows + kk[None, :] * sx1)
            ak = tl.load(a_base + kk[:, None] * sa2)
        else:
            xk = tl.load(x_rows + kk[None, :] * sx1, mask=kk[None, :] < K, other=0.0)
            ak = tl.load(a_base + kk[:, None] * sa2, mask=kk[:, None] < K, other=0.0)
        acc += tl.dot(xk, ak)
    t = (acc * scaling).to(b_ptr.dtype.element_ty)

    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < n_size
    bt = tl.load(b_ptr + lora_id * sb0 + offs_n[None, :] * sb1 + offs_r[:, None] * sb2, mask=n_ok[None, :], other=0.0)
    out = tl.dot(t, bt).to(b_ptr.dtype.element_ty)
    y_ptrs = y_ptr + ram[:, None] * sy0 + (col0 + offs_n)[None, :] * sy1
    mask = (offs_m[:, None] < m_len) & n_ok[None, :]
    out += tl.load(y_ptrs, mask=mask)
    tl.store(y_ptrs, out, mask=mask)


def _squeeze(w):
    if w.ndim == 4:
        assert w.size(1) == 1
        w = w.squeeze(1)
    return w


def config(m, k, n_max):
    """(BLOCK_N, BLOCK_K, num_warps, num_stages): the B300 sweep's best per Flash shape (mfu2-lfcheck2, M 64/152)."""
    if k <= 1024:
        return 256, 256, 4, 3
    if n_max <= 1024:
        return 64, 512, 4, 3
    if n_max <= 6144:
        return 256, 512, 4, 3
    return 256, 256, 4, 3


def fused_ok(y, x, lora_a, lora_b, output_slices):
    if x.size(0) > MAX_M or x.size(0) == 0 or len(output_slices) > 3 or not x.is_contiguous():
        return False
    # where the sweep measured vLLM's kernels faster: one wide slice, or K = 6144 below ~128 tokens
    if (len(output_slices) == 1 and output_slices[0] > 8192) or (x.size(1) > 4096 and x.size(0) < 128):
        return False
    if y.ndim != 2 or y.stride(1) != 1 or x.dtype not in (torch.float16, torch.bfloat16):
        return False
    a = [_squeeze(w) for w in lora_a]
    if any(w.shape != a[0].shape or w.stride() != a[0].stride() or not w.is_contiguous() for w in a):
        return False
    r = a[0].size(1)
    b = [_squeeze(w) for w in lora_b]
    return (r in (16, 32, 64) and all(w.is_contiguous() and w.size(-1) == r for w in b)
            and [w.size(1) for w in b] == list(output_slices) and y.size(1) >= sum(output_slices))


def fused_lora_linear(y, x, lora_a, lora_b, scale, output_slices, token_lora_mapping, sorted_idx, ntok, start, ids,
                      no_lora_flag_cpu, num_active_loras, cfg=None):
    """y[:, slice s] += ((x @ A_s^T) * scale) @ B_s^T for each token's LoRA (y: [M, sum(slices)], x: [M, K])."""
    if no_lora_flag_cpu.item():
        return
    a = [_squeeze(w) for w in lora_a]
    b = [_squeeze(w) for w in lora_b]
    m, k = x.shape
    r = a[0].size(1)
    ns = list(output_slices) + [0] * (3 - len(output_slices))
    a += [a[0]] * (3 - len(a))
    b += [b[0]] * (3 - len(b))
    block_n, block_k, warps, stages = cfg or config(m, k, max(ns))
    t0 = triton.cdiv(ns[0], block_n)
    t1 = t0 + triton.cdiv(ns[1], block_n)
    t2 = t1 + triton.cdiv(ns[2], block_n)
    grid = (t2, triton.cdiv(m, BLOCK_M), int(num_active_loras.item()))
    _arc_lora_fused_kernel[grid](
        x, y, a[0], a[1], a[2], b[0], b[1], b[2],
        ns[0], ns[1], ns[2], t0, t1, k,
        sorted_idx, ntok, start, ids,
        scale,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        a[0].stride(0), a[0].stride(1), a[0].stride(2),
        b[0].stride(0), b[1].stride(0), b[2].stride(0), b[0].stride(1), b[0].stride(2),
        R=r, BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k, EVEN_K=k % block_k == 0,
        num_warps=warps, num_stages=stages,
    )


def _arc_lora_linear(y: torch.Tensor, x: torch.Tensor, lora_a: list[torch.Tensor], lora_b: list[torch.Tensor],
                     scale: float, output_slices: list[int], token_lora_mapping: torch.Tensor,
                     sorted_idx: torch.Tensor, ntok: torch.Tensor, start: torch.Tensor, ids: torch.Tensor,
                     no_lora_flag_cpu: torch.Tensor, num_active_loras: torch.Tensor) -> None:
    meta = (token_lora_mapping, sorted_idx, ntok, start, ids, no_lora_flag_cpu, num_active_loras)
    if fused_ok(y, x, lora_a, lora_b, output_slices):
        STATS["fused"] += 1
        fused_lora_linear(y, x, lora_a, lora_b, scale, output_slices, *meta)
        return
    STATS["fallback"] += 1
    from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
    buffer = torch.empty((len(output_slices), x.size(0), _squeeze(lora_b[0]).size(-1)), dtype=torch.float32,
                         device=x.device)
    lora_shrink(x, lora_a, buffer, *meta, scale)
    lora_expand(buffer, lora_b, y, *meta, offset_start=0, add_inputs=True)


def _arc_lora_linear_fake(y, x, lora_a, lora_b, scale, output_slices, token_lora_mapping, sorted_idx, ntok, start,
                          ids, no_lora_flag_cpu, num_active_loras) -> None:
    return


try:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(op_name="arc_lora_linear", op_func=_arc_lora_linear, mutates_args=["y"],
                              fake_impl=_arc_lora_linear_fake)
    arc_lora_linear = torch.ops.vllm.arc_lora_linear
except (AttributeError, ImportError):
    arc_lora_linear = _arc_lora_linear


def _to(t, device):
    return t.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else t.to(device)


def prepare_cpu(meta, mapping):
    """``LoRAKernelMeta.prepare_tensors`` from the host-side token -> LoRA-slot list (serving_fast.lora_meta_cpu).

    vLLM derives the punica metadata from the device copy of the mapping (``torch.all``, ``torch.sort``,
    ``torch.unique``), which synchronises the host with the device every step and so drains the async-scheduling
    pipeline. The list is on the host already: the same stable sort, unique and cumsum run there and the results go
    to the same device buffers with non-blocking copies. Same values, no synchronisation."""
    import bisect

    meta._reset()
    if not mapping or all(v == -1 for v in mapping):
        meta.no_lora_flag_cpu[0] = True
        return
    t = torch.tensor(mapping, dtype=torch.long)
    n = t.numel()
    dev = meta.token_lora_mapping.device
    meta.token_lora_mapping[:n].copy_(_to(t.to(torch.int32), dev), non_blocking=True)
    order = torch.sort(t, stable=True).indices
    meta.token_indices_sorted_by_lora_ids[:n].copy_(_to(order.to(torch.int32), dev), non_blocking=True)
    ids, counts = torch.unique(t, sorted=True, return_counts=True)
    k = ids.numel()
    meta.active_lora_ids[:k].copy_(_to(ids.to(torch.int32), dev), non_blocking=True)
    meta.num_tokens_per_lora[:k].copy_(_to(counts.to(torch.int32), dev), non_blocking=True)
    active = k
    if meta.captured_lora_counts and active > 0:
        i = bisect.bisect_left(meta.captured_lora_counts, active)
        if i < len(meta.captured_lora_counts):
            active = meta.captured_lora_counts[i]
    meta.num_active_loras_cpu[0] = active
    meta.lora_token_start_loc[1:1 + k].copy_(_to(torch.cumsum(counts, 0).to(torch.int32), dev), non_blocking=True)
