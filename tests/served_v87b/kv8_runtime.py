"""8-bit QSA main KV cache for the pinned Qwen3.8-Flash-Next vLLM runtime (kernel work, 28 September).

Copied into ``vllm/arc_kv8.py`` by ``arc_infra.serving_kernels_patch`` only when ``server.serving_kernels``
sets ``kv_cache_8bit`` ("int8" or "fp8"); the mode is read from ``vllm/arc_serving_kernels.json``. The
server then runs with ``--kv-cache-dtype int8_per_token_head`` (or ``fp8_per_token_head``) and the patched
``nvidia/qsa.py`` calls the entry points below. No arc dependency inside the serving worker; vLLM imports
are deferred so that the kernels load standalone (CPU tests under ``TRITON_INTERPRET=1``, the microbenchmark
``python -m vllm.arc_kv8 bench``).

Layout (the pinned runtime's own per-token-head layout, ``TritonAttentionBackend``): each (token, KV head)
stores ``[K (256 x 1 B) | K scale (fp32) | V (256 x 1 B) | V scale (fp32)]``, i.e. 520 bytes against 1,024
in BF16. The scales live inside the page, so the byte-level block operations of the hybrid cache (new-block
zeroing, prefix-cache block copies, mamba pages aliasing the main-KV tensor) need no change.

- Write: the pinned ``triton_reshape_and_cache_flash_per_token_head_quant`` (scale = amax / 127 or
  amax / 448 per token and head; int8 rounds half away from zero).
- Read: ``_qsa_kv8_splitk_kernel``, the pinned ``_qsa_sparse_paged_gqa_splitk_kernel`` with 8-bit loads.
  int8 and e4m3 values are exact in BF16, so the MMAs take the raw codes and the scales are applied per
  column: ``scores = (q . k_code) * k_scale`` and ``out += (p * v_scale) . v_code``. Only the rounding of
  ``p * v_scale`` to BF16 differs from dequantize-then-multiply.
- The indexer caches (raw-key ring, compressed keys: 64 B per token against 2,048 for the main KV) stay BF16.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

CONFIG_PATH = Path(__file__).with_name("arc_serving_kernels.json")
CACHE_DTYPES = {"int8": "int8_per_token_head", "fp8": "fp8_per_token_head"}
SCALE_BYTES = 4  # one fp32 scale per (token, head) for K and one for V
_STATE = {"config": None, "counts": {}}


def load_config():
    if _STATE["config"] is None:
        _STATE["config"] = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    return _STATE["config"]


def log(kind, payload):
    print(f"ARC_KV8_{kind} " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def configured_cache_dtype():
    """The ``--kv-cache-dtype`` this overlay serves (None: the overlay is installed without kv8)."""
    mode = load_config().get("kv_cache_8bit")
    return None if mode is None else CACHE_DTYPES[mode]


def allowed(cache_dtype):
    """True only for the one 8-bit dtype the settings name; BF16 ("auto"/"bfloat16") is never routed here."""
    return cache_dtype is not None and cache_dtype == configured_cache_dtype()


def is_fp8(cache_dtype):
    return cache_dtype == CACHE_DTYPES["fp8"]


# ============================================================================ backend / impl plumbing
def impl_init_args(args, kwargs):
    """``FlashAttentionImpl.__init__`` refuses quantized caches FlashAttention cannot read; QSA never calls
    FlashAttention on its cache, so it is built with "auto" and the 8-bit dtype is restored afterwards."""
    args = list(args)
    if "kv_cache_dtype" in kwargs:
        dtype = kwargs["kv_cache_dtype"]
        if allowed(dtype):
            kwargs = {**kwargs, "kv_cache_dtype": "auto"}
            return tuple(args), kwargs, dtype
        return tuple(args), kwargs, None
    if len(args) > 6 and allowed(args[6]):
        dtype, args[6] = args[6], "auto"
        return tuple(args), kwargs, dtype
    return tuple(args), kwargs, None


def customize_spec(spec):
    """Per-token-head specs carry the inline scales in their page size (as ``TritonAttentionBackend``)."""
    mode = spec.kv_quant_mode
    if spec.state_content_bytes is not None or not mode.is_per_token_head:
        return spec
    from dataclasses import replace

    from vllm.v1.kv_cache_interface import KVQuantMode

    if mode not in (KVQuantMode.INT8_PER_TOKEN_HEAD, KVQuantMode.FP8_PER_TOKEN_HEAD):
        raise NotImplementedError(f"ARC kv8 serves int8/fp8 per-token-head caches, not {mode}")
    element = torch.tensor([], dtype=spec.dtype).element_size()
    content = (spec.head_size + spec.head_size_v) * element + 2 * SCALE_BYTES
    return replace(spec, state_content_bytes=content)


def kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
    """Logical (blocks, heads, tokens, content) as the pinned FlashAttention/Triton backends; 8-bit content
    is ``2 * (head_size + 4)`` bytes (each half ends in its fp32 scale)."""
    if block_size % 16 != 0:
        raise ValueError("Block size must be a multiple of 16.")
    if str(cache_dtype_str).endswith("per_token_head"):
        if cache_dtype_str not in CACHE_DTYPES.values():
            raise NotImplementedError(f"ARC kv8 serves {tuple(CACHE_DTYPES.values())}, not {cache_dtype_str}")
        return (num_blocks, num_kv_heads, block_size, 2 * (head_size + SCALE_BYTES))
    return (num_blocks, num_kv_heads, block_size, 2 * head_size)


def cache_views(kv_cache, head_size, fp8, holder=None):
    """(K codes, V codes, K scales, V scales) over one packed 8-bit cache tensor, as strided views.

    ``kv_cache``: logical ``(blocks, kv_heads, tokens, 2 * (head_size + 4))`` of a 1-byte dtype (NHD or HND
    physical strides). Codes: ``(blocks, tokens, kv_heads, head_size)`` int8 or float8_e4m3fn; scales:
    ``(blocks, tokens, kv_heads)`` float32. With ``holder`` (the layer's impl) the views are kept on it for its
    current cache tensor only, so a replaced cache is never held alive.
    """
    key = (kv_cache.data_ptr(), tuple(kv_cache.shape), tuple(kv_cache.stride()), kv_cache.storage_offset(),
           head_size, fp8)
    cached = getattr(holder, "_arc_kv8_views", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if kv_cache.ndim != 4 or kv_cache.element_size() != 1:
        raise ValueError("ARC kv8 expects a 4-D one-byte packed cache")
    num_blocks, num_heads, block_size, content = kv_cache.shape
    padded = content // 2
    if content != 2 * padded or padded != head_size + SCALE_BYTES:
        raise ValueError(f"ARC kv8 cache content {content} does not match head size {head_size}")
    tokens_first = kv_cache.transpose(1, 2)
    keys = tokens_first[..., :head_size]
    values = tokens_first[..., padded:padded + head_size]
    code_dtype = torch.float8_e4m3fn if fp8 else torch.int8
    keys, values = keys.view(code_dtype), values.view(code_dtype)
    strides = kv_cache.stride()
    offset = kv_cache.storage_offset()
    if any(s % 4 for s in strides[:3]) or offset % 4 or head_size % 4:
        raise ValueError("ARC kv8 scale slots are not 4-byte aligned")
    base = torch.empty(0, dtype=torch.float32, device=kv_cache.device).set_(kv_cache.untyped_storage())
    size = (num_blocks, block_size, num_heads)
    stride = (strides[0] // 4, strides[2] // 4, strides[1] // 4)
    k_scale = torch.as_strided(base, size=size, stride=stride, storage_offset=(offset + head_size) // 4)
    v_scale = torch.as_strided(base, size=size, stride=stride, storage_offset=(offset + padded + head_size) // 4)
    views = (keys, values, k_scale, v_scale)
    if holder is not None:
        holder._arc_kv8_views = (key, views)
    return views


def _count(name):
    first = name not in _STATE["counts"]
    _STATE["counts"][name] = _STATE["counts"].get(name, 0) + 1
    if first:
        log("FIRST_CALL", {"entry": name, "cache_dtype": configured_cache_dtype()})


def cache_update(impl, key, value, kv_cache, slot_mapping):
    """Patched ``do_kv_cache_update``: quantize this step's K/V rows into the 8-bit cache."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.kv_cache_interface import get_kv_quant_mode

    _count("cache_update")
    keys, values, k_scale, v_scale = cache_views(kv_cache, impl.head_size, is_fp8(impl.kv_cache_dtype), impl)
    rows = slot_mapping.shape[0]
    triton_reshape_and_cache_flash_per_token_head_quant(
        key[:rows], value[:rows], keys, values, k_scale, v_scale, slot_mapping,
        kv_quant_mode=get_kv_quant_mode(impl.kv_cache_dtype))


def attention(impl, query, kv_cache, logical_indices, block_table, token_to_req, out):
    """Patched ``forward_qsa`` body: sparse GQA over the 8-bit cache."""
    _count("attention")
    keys, values, k_scale, v_scale = cache_views(kv_cache, impl.head_size, is_fp8(impl.kv_cache_dtype), impl)
    return sparse_attention(query, keys, values, k_scale, v_scale, logical_indices, block_table, token_to_req, out)


# ============================================================================ kernels
@triton.jit
def _qsa_kv8_splitk_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, indices_ptr, block_table_ptr, token_to_req_ptr,
    partial_output_ptr, partial_lse_ptr, output_ptr,
    stride_q_row, stride_q_head,
    stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head,
    stride_ks_block, stride_ks_token, stride_ks_head,
    stride_vs_block, stride_vs_token, stride_vs_head,
    stride_indices_row, stride_table_req, stride_output_row, stride_output_head,
    num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, INTERPRET: tl.constexpr = False,
) -> None:
    # The pinned _qsa_sparse_paged_gqa_splitk_kernel (ops/qsa.py) with 8-bit K/V codes and per-token scales.
    # INTERPRET (CPU tests only): the Triton interpreter's BF16 dot and int8->BF16 cast are wrong, so it runs
    # the same BF16-rounded operands through FP32 dots.
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1)
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (request >= 0) & (request < num_requests) & (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_codes = tl.load(
            k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        value_codes = tl.load(
            v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        key_scale = tl.load(
            k_scale_ptr + safe_page * stride_ks_block + page_offset * stride_ks_token + kv_head * stride_ks_head,
            mask=valid,
            other=0.0,
        )
        value_scale = tl.load(
            v_scale_ptr + safe_page * stride_vs_block + page_offset * stride_vs_token + kv_head * stride_vs_head,
            mask=valid,
            other=0.0,
        )
        # int8 / e4m3 codes are exact in BF16: the MMA runs on the codes, the scale is per column.
        if INTERPRET:
            scores = tl.dot(query.to(tl.float32), key_codes.to(tl.float32))
        else:
            scores = tl.dot(query, key_codes.to(tl.bfloat16))
        scores *= key_scale[None, :] * softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        weights = (probabilities * value_scale[None, :]).to(tl.bfloat16)
        if INTERPRET:
            accumulator = tl.dot(weights.to(tl.float32), value_codes.to(tl.float32), acc=accumulator * alpha[:, None])
        else:
            accumulator = tl.dot(weights, value_codes.to(tl.bfloat16), acc=accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        tl.store(
            partial_output_ptr + ((split_id * num_rows + row) * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + (split_id * num_rows + row) * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_kv8_merge_kernel(
    partial_output_ptr, partial_lse_ptr, output_ptr, stride_output_row, stride_output_head, num_rows,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, BLOCK_SPLITS: tl.constexpr,
) -> None:
    # The pinned _qsa_merge_splitk_kernel, copied so that this module loads without vLLM.
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head, mask=split_mask,
                  other=-float("inf"))
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    # == the pinned where(split_mask & has_values, ...), split so the Triton interpreter can run it.
    shifted = tl.where(has_values, tl.where(split_mask, lse - lse_max, -float("inf")), -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets, merged)


def _interpreting():
    return os.environ.get("TRITON_INTERPRET") == "1"


def launch_profile(rows, kv_heads, group_size, topk):
    """The pinned tile/split profile (``qsa_sparse_paged_attention``, tuned on GB300), unchanged."""
    block_m = triton.next_power_of_2(group_size)
    base_programs = rows * kv_heads
    small_profile_limit = 8 if block_m <= 8 else 4
    if base_programs <= small_profile_limit:
        block_n, target_splits, warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, warps = 64, 4, 2
    else:
        block_n, target_splits, warps = 64, 1, 2
    num_tiles = triton.cdiv(topk, block_n)
    splits = min(1 << (num_tiles.bit_length() - 1), target_splits)
    return {"block_m": block_m, "block_n": block_n, "splits": splits, "warps": warps, "tiles": num_tiles}


def sparse_attention(q, k_codes, v_codes, k_scale, v_scale, logical_indices, block_table, token_to_req, out=None,
                     profile=None):
    """Sparse GQA over paged 8-bit K/V codes with per-(token, head) fp32 scales.

    Shapes as the pinned ``qsa_sparse_paged_attention``: ``q`` [rows, heads, dim] BF16; codes
    [blocks, page, kv_heads, dim] int8 / float8_e4m3fn; scales [blocks, page, kv_heads] fp32; indices
    [rows, topk] int32 request-relative token ids (-1 padding); ``block_table`` [requests, width] int32.
    """
    if q.ndim != 3 or k_codes.ndim != 4 or v_codes.shape != k_codes.shape:
        raise ValueError("QSA kv8 attention received invalid Q/K/V shapes")
    if k_scale.shape != k_codes.shape[:3] or v_scale.shape != k_scale.shape:
        raise ValueError("QSA kv8 scales must be [blocks, page, kv_heads]")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0] or logical_indices.shape[1] <= 0:
        raise ValueError("QSA kv8 indices must have one positive-width row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA kv8 metadata has invalid shapes")
    if q.shape[2] != k_codes.shape[3] or q.shape[1] % k_codes.shape[2]:
        raise ValueError("QSA kv8 attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16 and k_codes.dtype == v_codes.dtype
    assert k_codes.dtype in (torch.int8, torch.float8_e4m3fn)
    assert k_scale.dtype == v_scale.dtype == torch.float32
    assert logical_indices.dtype == block_table.dtype == token_to_req.dtype == torch.int32
    assert q.stride(2) == k_codes.stride(3) == v_codes.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1 and token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape or out.dtype != q.dtype or out.stride(2) != 1:
        raise ValueError("QSA kv8 output must match its query")
    if not q.shape[0]:
        return out
    group_size = q.shape[1] // k_codes.shape[2]
    p = profile or launch_profile(q.shape[0], k_codes.shape[2], group_size, logical_indices.shape[1])
    if p["splits"] == 1:
        partial_output, partial_lse = out, out
    else:
        partial_output = torch.empty((p["splits"], *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((p["splits"], q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    _qsa_kv8_splitk_kernel[(q.shape[0], k_codes.shape[2], p["splits"])](
        q, k_codes, v_codes, k_scale, v_scale, logical_indices, block_table, token_to_req,
        partial_output, partial_lse, out,
        q.stride(0), q.stride(1),
        k_codes.stride(0), k_codes.stride(1), k_codes.stride(2),
        v_codes.stride(0), v_codes.stride(1), v_codes.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        logical_indices.stride(0), block_table.stride(0), out.stride(0), out.stride(1),
        q.shape[0], k_codes.shape[0], block_table.shape[0],
        TOPK=logical_indices.shape[1], PAGE_SIZE=k_codes.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size, HEAD_DIM=head_dim, NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=p["splits"],
        NUM_TILES=p["tiles"], BLOCK_M=p["block_m"], BLOCK_N=p["block_n"], INTERPRET=_interpreting(),
        num_warps=p["warps"], num_stages=2,
    )
    if p["splits"] == 1:
        return out
    _qsa_kv8_merge_kernel[(q.shape[0], q.shape[1])](
        partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
        HEAD_DIM=head_dim, NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=p["splits"],
        BLOCK_SPLITS=triton.next_power_of_2(p["splits"]), num_warps=2, num_stages=1,
    )
    return out


# ============================================================================ references (tests, bench)
QUANT_MAX = {"int8": 127.0, "fp8": 448.0}


def quantize_reference(x, mode):
    """Torch mirror of the pinned per-token-head writer: x [..., dim] -> (codes, fp32 scale [...])."""
    x = x.float()
    scale = (x.abs().amax(dim=-1) / QUANT_MAX[mode]).clamp_min(1e-6)
    y = x / scale.unsqueeze(-1)
    if mode == "int8":
        y = torch.where(y >= 0, y + 0.5, y - 0.5).clamp(-128.0, 127.0).trunc().to(torch.int8)
    else:
        y = y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return y, scale


def dequantize_reference(codes, scale):
    return codes.float() * scale.unsqueeze(-1)


def attention_reference(q, k, v, logical_indices, block_table, token_to_req):
    """FP32 sparse GQA on dense per-token K/V pages ``[blocks, page, kv_heads, dim]`` (any float dtype)."""
    rows, heads, dim = q.shape
    kv_heads = k.shape[2]
    group = heads // kv_heads
    page = k.shape[1]
    out = torch.zeros((rows, heads, dim), dtype=torch.float32, device=q.device)
    for row in range(rows):
        request = int(token_to_req[row])
        if request < 0 or request >= block_table.shape[0]:
            continue
        tokens = [int(t) for t in logical_indices[row].tolist() if t >= 0 and t // page < block_table.shape[1]]
        tokens = [t for t in tokens if 0 <= int(block_table[request, t // page]) < k.shape[0]]
        if not tokens:
            continue
        index = torch.tensor(tokens, device=q.device)
        pages = block_table[request, index // page].long()
        offsets = index % page
        for kv_head in range(kv_heads):
            keys = k[pages, offsets, kv_head].float()
            values = v[pages, offsets, kv_head].float()
            query = q[row, kv_head * group:(kv_head + 1) * group].float()
            weights = torch.softmax(query @ keys.T * dim ** -0.5, dim=-1)
            out[row, kv_head * group:(kv_head + 1) * group] = weights @ values
    return out


def synthetic_case(rows, *, kv_heads=2, group=12, dim=256, page=64, blocks=64, topk=203, requests=None,
                   device="cpu", seed=0, fill=0.8):
    """Random paged K/V (unit-RMS rows with a few outliers, like normed and RoPE'd keys), block tables and
    top-k indices with -1 padding, for the reference checks and the microbenchmark."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    requests = requests or max(1, rows)
    k = torch.randn((blocks, page, kv_heads, dim), generator=g)
    v = torch.randn((blocks, page, kv_heads, dim), generator=g)
    k[..., :4] *= 4.0  # outlier channels (RoPE'd low-frequency pairs)
    q = torch.randn((rows, kv_heads * group, dim), generator=g)
    width = max(1, blocks // requests)
    table = torch.full((requests, width), -1, dtype=torch.int32)
    order = torch.randperm(blocks, generator=g)
    for r in range(requests):
        table[r] = order[r * width:(r + 1) * width].to(torch.int32)
    length = width * page
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        count = max(1, int(topk * fill))
        chosen = torch.randperm(length, generator=g)[:count].to(torch.int32)
        indices[row, :count] = chosen
    token_to_req = (torch.arange(rows) % requests).to(torch.int32)
    to = dict(device=device)
    return (q.to(torch.bfloat16).to(**to), k.to(torch.bfloat16).to(**to), v.to(torch.bfloat16).to(**to),
            indices.to(**to), table.to(**to), token_to_req.to(**to))


def pack_cache(k, v, mode):
    """Quantize BF16 pages into one packed per-token-head cache in the served layout (NHD physical strides):
    returns the logical ``(blocks, kv_heads, page, 2 * (dim + 4))`` tensor."""
    blocks, page, kv_heads, dim = k.shape
    kq, ks = quantize_reference(k, mode)
    vq, vs = quantize_reference(v, mode)
    dtype = torch.uint8 if mode == "fp8" else torch.int8
    raw = torch.zeros((blocks, page, kv_heads, 2 * (dim + SCALE_BYTES)), dtype=torch.uint8, device=k.device)
    raw[..., :dim] = kq.view(torch.uint8)
    raw[..., dim:dim + 4] = ks.contiguous().view(torch.uint8).view(blocks, page, kv_heads, 4)
    raw[..., dim + 4:2 * dim + 4] = vq.view(torch.uint8)
    raw[..., 2 * dim + 4:] = vs.contiguous().view(torch.uint8).view(blocks, page, kv_heads, 4)
    return raw.view(dtype).transpose(1, 2)


# ============================================================================ microbenchmark (GPU)
def _time(fn, repeats=200):
    """Median µs per call under CUDA-graph replay (decode runs graphed)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fn()
    samples = []
    for _ in range(repeats // 10):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 10)
    samples.sort()
    return samples[len(samples) // 2]


def bench(output=None, topks=(1027, 2051, 4099), rows_list=(1, 4, 6, 24, 36), page=1664, blocks=None):
    """QSA sparse attention per call: the pinned BF16 kernel against int8 / fp8 codes at decode shapes (24 query
    heads, 2 KV heads, 256 dims; MTP-3 verify rows = 4 x streams), plus error against an FP32 reference and
    a writer round trip through the pinned per-token-head quantizer."""
    from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode

    device = "cuda"
    report = {"device": torch.cuda.get_device_name(), "page": page, "rows": []}
    for topk in topks:
        for rows in rows_list:
            requests = max(1, rows // 4) if rows >= 4 else 1
            # Enough distinct pages that the working set is far above L2 (128 MiB on the RTX PRO 6000).
            count = blocks or max(64, requests * 8)
            q, k, v, indices, table, t2r = synthetic_case(rows, page=page, blocks=count, topk=topk,
                                                          requests=requests, device=device, fill=1.0)
            k_bf16 = k.contiguous()
            v_bf16 = v.contiguous()
            reference = attention_reference(q[: min(rows, 4)], k_bf16, v_bf16, indices[: min(rows, 4)], table,
                                            t2r[: min(rows, 4)]) if topk == topks[0] else None
            out = torch.empty_like(q)
            entry = {"topk": topk, "rows": rows, "requests": requests, "pages": count,
                     "profile": launch_profile(rows, 2, 12, topk)}
            entry["bf16_us"] = _time(lambda: qsa_sparse_paged_attention(q, k_bf16, v_bf16, indices, table, t2r, out))
            if reference is not None:
                qsa_sparse_paged_attention(q, k_bf16, v_bf16, indices, table, t2r, out)
                entry["bf16_rel_error"] = _rel(out[: reference.shape[0]].float(), reference)
            for mode in ("int8", "fp8"):
                packed = pack_cache(k, v, mode)
                codes_k, codes_v, sk, sv = cache_views(packed, 256, mode == "fp8")
                entry[f"{mode}_us"] = _time(
                    lambda: sparse_attention(q, codes_k, codes_v, sk, sv, indices, table, t2r, out))
                if reference is not None:
                    sparse_attention(q, codes_k, codes_v, sk, sv, indices, table, t2r, out)
                    entry[f"{mode}_rel_error"] = _rel(out[: reference.shape[0]].float(), reference)
                    # Writer round trip: the pinned quantizer into an empty cache of the same layout.
                    written = torch.zeros_like(packed)
                    wk, wv, wks, wvs = cache_views(written, 256, mode == "fp8")
                    slots = torch.arange(count * page, device=device, dtype=torch.int64)
                    triton_reshape_and_cache_flash_per_token_head_quant(
                        k.reshape(-1, 2, 256), v.reshape(-1, 2, 256), wk, wv, wks, wvs, slots,
                        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD if mode == "int8"
                        else KVQuantMode.FP8_PER_TOKEN_HEAD)
                    torch.cuda.synchronize()
                    entry[f"{mode}_writer_codes_equal"] = float(
                        (written.view(torch.uint8) == packed.view(torch.uint8)).float().mean())
                    entry[f"{mode}_writer_scale_max_rel"] = float(
                        ((wks - cache_views(packed, 256, mode == "fp8")[2]).abs()
                         / cache_views(packed, 256, mode == "fp8")[2].abs().clamp_min(1e-12)).max())
                del packed
            report["rows"].append(entry)
            log("BENCH", entry)
            torch.cuda.empty_cache()
    if output:
        Path(output).write_text(json.dumps(report, indent=2))
    return report


def _rel(x, reference):
    return float((x - reference).norm() / reference.norm().clamp_min(1e-12))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "bench":
        started = time.monotonic()
        result = bench(sys.argv[2] if len(sys.argv) > 2 else None)
        log("BENCH_DONE", {"seconds": round(time.monotonic() - started, 1), "rows": len(result["rows"])})
