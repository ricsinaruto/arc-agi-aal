"""BF16 prefill path for the 8-bit QSA KV cache (kernel work, phase 7; docs/validation-20260927/kernels-phase7.md).

Copied into ``vllm/arc_kv8_prefill.py`` by ``arc_infra.serving_kernels_patch`` only when ``server.serving_kernels``
sets ``kv8_prefill: "bf16"`` (with ``kv_cache_8bit: "int8"``); the patched ``nvidia/qsa.py`` then calls
``attention`` below instead of ``arc_kv8.attention``. Arms without the option get neither this file nor the edit.

Measured (qsa-prefill-bench-v2, RTX PRO 6000, 4,096 query rows of one request, top-k 2,051, 24K / 40K contexts): the
int8 split-K kernel takes 8.8-8.9 ms per layer on a prefill chunk (the served profile), the pinned BF16 kernel 4.2 ms,
the BF16 kernel with the (BLOCK_N 32, 1 split, 4 warps) tile 3.2-3.4 ms, and dequantizing the request's int8 pages to
BF16 0.06-0.10 ms. So for the query rows of a prefill chunk this path dequantizes the chunk's requests' pages into a
scratch BF16 cache and runs the pinned BF16 sparse kernel on them (~3.4 against ~8.9 ms per QSA layer).

- Decode is unchanged: a call with <= ``DECODE_MAX_PROGRAMS`` (rows x KV heads; every CUDA-graph size, <= 32 rows) is
  ``arc_kv8.attention`` itself, byte for byte the served path. The KV cache stays int8 (capacity unchanged).
- Mixed calls (eager: above the 32-token graph sizes) split the rows. Rows of requests with >= ``PREFILL_MIN_ROWS``
  rows in the call take the BF16 path; the others (decode rows) the int8 kernel.
- Numerics: the BF16 path attends over exactly the int8 values (code x scale, one BF16 rounding, as the int8 kernel's
  scores use) with the pinned kernel's BF16 arithmetic. Against FP32 on those values it is more accurate than the
  int8 kernel (0.22% against 0.37% relative error); the two differ by ~0.42%. The first mixed call of each layer is
  checked against the int8 path on the same rows (``ARC_KV8P_CHECK``): above ``CHECK_LIMIT`` the layer keeps the int8
  path for the rest of the run.
"""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl

CONFIG_PATH = Path(__file__).with_name("arc_serving_kernels.json")
# The pinned launch profile's prefill branch starts above 512 programs; graph sizes (<= 32 rows x 2 heads) never get
# here, so decode is always the served kernel.
DECODE_MAX_PROGRAMS = 512
PREFILL_MIN_ROWS = 64
# BF16 kernel tile for prefill rows (BLOCK_N, splits, warps, stages): 3.2-3.4 ms against the pinned (64, 1, 2) 4.2.
BF16_PROFILE = (32, 1, 4, 2)
CHECK_LIMIT = 0.02  # relative Frobenius difference to the int8 path (measured 0.42%)
_STATE = {"config": None, "checked": {}, "counts": {"decode": 0, "mixed": 0, "fallback": 0}, "logged": set()}


def load_config():
    if _STATE["config"] is None:
        _STATE["config"] = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    return _STATE["config"]


def log(kind, payload):
    print(f"ARC_KV8P_{kind} " + json.dumps(payload, sort_keys=True, default=str), flush=True)


@triton.jit
def _dequant_pages_kernel(codes_ptr, scale_ptr, out_ptr, page_ptr, stride_c_block, stride_c_token, stride_c_head,
                          stride_s_block, stride_s_token, stride_s_head, PAGE: tl.constexpr, HEADS: tl.constexpr,
                          HEAD_DIM: tl.constexpr):
    """out[i, token, head] = bf16(code x scale) of cache page ``pages[i]`` (int8 codes are exact in BF16)."""
    i = tl.program_id(0)
    token = tl.program_id(1)
    page = tl.load(page_ptr + i).to(tl.int64)
    dims = tl.arange(0, HEAD_DIM)
    for head in tl.static_range(HEADS):
        code = tl.load(codes_ptr + page * stride_c_block + token * stride_c_token + head * stride_c_head + dims)
        scale = tl.load(scale_ptr + page * stride_s_block + token * stride_s_token + head * stride_s_head)
        tl.store(out_ptr + ((i * PAGE + token) * HEADS + head) * HEAD_DIM + dims,
                 (code.to(tl.float32) * scale).to(tl.bfloat16))


def dequant_pages(codes, scale, pages):
    """codes [blocks, page, heads, dim] int8 (strided view), scale [blocks, page, heads] fp32, pages [n] int ->
    contiguous BF16 [n, page, heads, dim]."""
    blocks, page, heads, dim = codes.shape
    out = torch.empty((pages.numel(), page, heads, dim), dtype=torch.bfloat16, device=codes.device)
    if pages.numel():
        _dequant_pages_kernel[(pages.numel(), page)](
            codes, scale, out, pages, codes.stride(0), codes.stride(1), codes.stride(2), scale.stride(0),
            scale.stride(1), scale.stride(2), PAGE=page, HEADS=heads, HEAD_DIM=dim)
    return out


def dequant_reference(codes, scale):
    return (codes.float() * scale.unsqueeze(-1)).to(torch.bfloat16)


def bf16_attention(pinned, q, k, v, indices, table, t2r, out, profile=BF16_PROFILE):
    """The pinned BF16 split-K kernel with an explicit tile (its wrapper picks (64, 1, 2) for this many rows)."""
    block_n, splits, warps, stages = profile
    group = q.shape[1] // k.shape[2]
    tiles = triton.cdiv(indices.shape[1], block_n)
    if splits == 1:
        part_o, part_l = out, out
    else:
        part_o = torch.empty((splits, *q.shape), dtype=torch.float32, device=q.device)
        part_l = torch.empty((splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    pinned._qsa_sparse_paged_gqa_splitk_kernel[(q.shape[0], k.shape[2], splits)](
        q, k, v, indices, table, t2r, part_o, part_l, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        indices.stride(0), table.stride(0), out.stride(0), out.stride(1), q.shape[0], k.shape[0], table.shape[0],
        TOPK=indices.shape[1], PAGE_SIZE=k.shape[1], PAGE_TABLE_WIDTH=table.shape[1], GROUP_SIZE=group,
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=splits, NUM_TILES=tiles,
        BLOCK_M=triton.next_power_of_2(group), BLOCK_N=block_n, num_warps=warps, num_stages=stages)
    if splits > 1:
        pinned._qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            part_o, part_l, out, out.stride(0), out.stride(1), q.shape[0], HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=splits, BLOCK_SPLITS=triton.next_power_of_2(splits),
            num_warps=2, num_stages=1)
    return out


def plan(token_to_req, logical_indices, block_table, page, num_blocks, min_rows=PREFILL_MIN_ROWS):
    """Host-side split of one call (eager only): the prefill requests (>= ``min_rows`` rows), their rows, the pages
    their selected tokens can reach, and the scratch block table. Returns None when no request qualifies."""
    valid = token_to_req >= 0
    requests = token_to_req.clamp_min(0).long()
    counts = torch.bincount(requests[valid], minlength=block_table.shape[0])
    heavy = torch.nonzero(counts >= min_rows).flatten()
    if heavy.numel() == 0:
        return None
    heavy_rows = torch.nonzero(valid & torch.isin(requests, heavy)).flatten()
    light_rows = torch.nonzero(~(valid & torch.isin(requests, heavy))).flatten()
    # Pages each heavy request needs: logical pages up to its largest selected token (causal: <= its positions).
    remap = torch.full((block_table.shape[0],), -1, dtype=torch.long, device=token_to_req.device)
    remap[heavy] = torch.arange(heavy.numel(), device=token_to_req.device)
    row_req = requests[heavy_rows]
    top = torch.zeros((heavy.numel(),), dtype=torch.long, device=token_to_req.device)
    top.scatter_reduce_(0, remap[row_req], logical_indices[heavy_rows].amax(dim=1).long(), reduce="amax")
    need = (top // page + 1).clamp(max=block_table.shape[1])
    width = int(need.max())
    columns = torch.arange(width, device=token_to_req.device)
    take = columns[None, :] < need[:, None]
    physical = block_table[heavy][:, :width].long()
    # Unallocated entries (outside the cache) are never dequantized; the kernel masks their table slot (-1), as the
    # int8 kernel masks an out-of-range physical page.
    take &= (physical >= 0) & (physical < num_blocks)
    pages = physical[take]
    table = torch.full((heavy.numel(), width), -1, dtype=torch.int32, device=token_to_req.device)
    table[take] = torch.arange(pages.numel(), device=token_to_req.device, dtype=torch.int32)
    return {"heavy_rows": heavy_rows, "light_rows": light_rows, "t2r": remap[row_req].to(torch.int32),
            "pages": pages, "table": table}


def prefill_attention(kv8, pinned, query, keys, values, k_scale, v_scale, logical_indices, block_table, token_to_req,
                      out, split):
    """The split call: BF16 over dequantized pages for the prefill rows, the int8 kernel for the rest."""
    rows = split["heavy_rows"]
    k = dequant_pages(keys, k_scale, split["pages"])
    v = dequant_pages(values, v_scale, split["pages"])
    q_h = query.index_select(0, rows)
    out_h = torch.empty_like(q_h)
    bf16_attention(pinned, q_h, k, v, logical_indices.index_select(0, rows).contiguous(), split["table"],
                   split["t2r"], out_h)
    out.index_copy_(0, rows, out_h)
    light = split["light_rows"]
    if light.numel():
        q_l = query.index_select(0, light)
        out_l = torch.empty_like(q_l)
        kv8.sparse_attention(q_l, keys, values, k_scale, v_scale, logical_indices.index_select(0, light).contiguous(),
                             block_table, token_to_req.index_select(0, light).contiguous(), out_l)
        out.index_copy_(0, light, out_l)
    return out


def attention(impl, query, kv_cache, logical_indices, block_table, token_to_req, out):
    """Patched ``forward_qsa`` body (kv8_prefill): the served int8 path for decode-size calls, the BF16 path for
    the rows of prefill chunks."""
    from vllm import arc_kv8 as kv8

    rows = query.shape[0]
    heads = kv_cache.shape[1]
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    layer = id(impl)
    if rows * heads <= DECODE_MAX_PROGRAMS or capturing or _STATE["checked"].get(layer) is False:
        _STATE["counts"]["decode" if rows * heads <= DECODE_MAX_PROGRAMS else "fallback"] += 1
        return kv8.attention(impl, query, kv_cache, logical_indices, block_table, token_to_req, out)
    from vllm.models.qwen3_8_flash_next.nvidia.ops import qsa as pinned

    keys, values, k_scale, v_scale = kv8.cache_views(kv_cache, impl.head_size, kv8.is_fp8(impl.kv_cache_dtype), impl)
    split = plan(token_to_req, logical_indices, block_table, keys.shape[1], keys.shape[0])
    if split is None:
        _STATE["counts"]["fallback"] += 1
        return kv8.attention(impl, query, kv_cache, logical_indices, block_table, token_to_req, out)
    prefill_attention(kv8, pinned, query, keys, values, k_scale, v_scale, logical_indices, block_table, token_to_req,
                      out, split)
    _STATE["counts"]["mixed"] += 1
    if layer not in _STATE["checked"]:
        # First mixed call of this layer: the int8 path on the same rows, and the dequant against torch.
        reference = torch.empty_like(out)
        kv8.sparse_attention(query, keys, values, k_scale, v_scale, logical_indices, block_table, token_to_req,
                             reference)
        rows_h = split["heavy_rows"]
        diff = float((out.index_select(0, rows_h).float() - reference.index_select(0, rows_h).float()).norm()
                     / reference.index_select(0, rows_h).float().norm().clamp_min(1e-30))
        probe = split["pages"][:2]
        dequant_equal = bool(torch.equal(dequant_pages(keys, k_scale, probe),
                                         dequant_reference(keys.index_select(0, probe), k_scale.index_select(0, probe))))
        ok = diff <= CHECK_LIMIT and dequant_equal and bool(torch.isfinite(out).all())
        _STATE["checked"][layer] = ok
        log("CHECK", {"layer": getattr(impl, "layer_name", None) or layer, "rows": rows,
                      "prefill_rows": int(rows_h.numel()), "pages": int(split["pages"].numel()),
                      "rel_to_int8": diff, "dequant_equal": dequant_equal, "ok": ok, "limit": CHECK_LIMIT})
        if not ok:
            out.copy_(reference)
    return out
