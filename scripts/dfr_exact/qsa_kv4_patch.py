"""NVFP4 KV cache for his QSA backend (SGLang 0.5.19 Pennyroyal). python qsa_kv4_patch.py <site-packages>/sglang
- sparse_attn.py: the decode compaction and the chunk-prefill gather read packed FP4 + per-16 E4M3 block scales +
  the layer's global scale and write BF16 (stock kernels load the pool as-is: KeyError float4_e2m1fn_x2).
- qwen_sparse_attn_backend.py: an NVFP4 pool hands those kernels its raw buffers instead of get_key_buffer's view.
- fp4_kv_cache_quant_method.py: no shared FP8 dequant workspace (only FlashInfer prefill reads it; costs 2 B/token/
  head-dim); global scales calibrated per layer at the first eager write (amax x ARC_KV4_HEADROOM, default 16), as
  the checkpoint carries no k/v scales and 1.0 leaves small V blocks in E4M3 subnormals."""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "srt"
MARK = "arc-kv4"


def edit(rel, pairs):
    p = root / rel; s = p.read_text()
    if MARK in s:
        return print(rel, "already patched")
    for old, new in pairs:
        assert s.count(old) >= 1, f"{rel}: anchor moved: {old[:60]!r}"
        s = s.replace(old, new)
    p.write_text(s); print(rel, "patched")

# HybridLinearKVPool.set_kv_buffer passes k_scale=v_scale=1.0 (floats), which bypass the per-layer GPU scales.
edit("mem_cache/memory_pool.py", [(
    "        if k_scale is None and hasattr(self.quant_method, \"k_scales_gpu\"):",
    "        if (k_scale is None or isinstance(k_scale, float)) and hasattr(  # arc-kv4: hybrid pool passes 1.0\n"
    "            self.quant_method, \"k_scales_gpu\"\n        ):")])

KERNELS = '''

# ---- arc-kv4: NVFP4 (E2M1 + per-16 E4M3 block scale + per-layer FP32 global scale) KV reads ----
@triton.jit
def _arc_e2m1(n):
    mag = n & 7
    e = mag >> 1
    m = (mag & 1).to(tl.float32)
    v = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((n & 8) != 0, -v, v)


@triton.jit
def _arc_fp4_rows(q_ptr, s_ptr, g, rows, head, valid, heads: tl.constexpr, dim: tl.constexpr,
                  HALF: tl.constexpr):
    """Dequantize [BLOCK, dim] for token rows `rows` of one head: returns (even, odd) element values."""
    hd = tl.arange(0, HALF)
    mask = valid[:, None] & (hd[None, :] < dim // 2)
    packed = tl.load(q_ptr + rows[:, None] * (heads * (dim // 2)) + head * (dim // 2) + hd[None, :],
                     mask=mask, other=0).to(tl.int32)
    scale = tl.load(s_ptr + rows[:, None] * (heads * (dim // 16)) + head * (dim // 16) + hd[None, :] // 8,
                    mask=mask, other=0.0).to(tl.float32) * g
    return _arc_e2m1(packed & 15) * scale, _arc_e2m1(packed >> 4) * scale, mask, hd


@triton.jit
def _arc_compact_kv_fp4(kq, vq, ks, vs, gk, gv, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v,
                        topk: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr, req_stride: tl.constexpr,
                        idx_stride: tl.constexpr, BLOCK_TOPK: tl.constexpr, HALF: tl.constexpr):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(req_to_token + req * req_stride + tl.where(valid, positions, 0), mask=valid, other=0).to(tl.int64)
    dst = (pack_start + cols).to(tl.int64)[:, None] * (heads * dim) + head * dim + tl.arange(0, 2 * HALF)[None, :]
    mask = valid[:, None] & (tl.arange(0, 2 * HALF)[None, :] < dim)
    ke, ko, _, _ = _arc_fp4_rows(kq, ks, tl.load(gk), slots, head, valid, heads, dim, HALF)
    tl.store(out_k + dst, tl.interleave(ke, ko).to(out_k.dtype.element_ty), mask=mask)
    ve, vo, _, _ = _arc_fp4_rows(vq, vs, tl.load(gv), slots, head, valid, heads, dim, HALF)
    tl.store(out_v + dst, tl.interleave(ve, vo).to(out_v.dtype.element_ty), mask=mask)


@triton.jit
def _arc_gather_fp4(q, s, g, index, out, n, heads: tl.constexpr, dim: tl.constexpr, BLOCK: tl.constexpr,
                    HALF: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    valid = rows < n
    slots = tl.load(index + rows, mask=valid, other=0).to(tl.int64)
    e, o, _, _ = _arc_fp4_rows(q, s, tl.load(g), slots, head, valid, heads, dim, HALF)
    dst = rows.to(tl.int64)[:, None] * (heads * dim) + head * dim + tl.arange(0, 2 * HALF)[None, :]
    mask = valid[:, None] & (tl.arange(0, 2 * HALF)[None, :] < dim)
    tl.store(out + dst, tl.interleave(e, o).to(out.dtype.element_ty), mask=mask)


class ArcFP4KV:
    """One of K/V of an NVFP4 pool layer, standing in for the plain buffer: shape/dtype as BF16 attention sees it."""

    arc_fp4 = True

    def __init__(self, packed, scales, global_scale, head_dim, dtype=torch.bfloat16):
        self.packed, self.scales, self.global_scale = packed, scales, global_scale
        self.shape = (packed.shape[0], packed.shape[1], head_dim)
        self.dtype, self.device = dtype, packed.device

    def index_select(self, dim, index):
        assert dim == 0
        n, heads, hdim = index.numel(), self.shape[1], self.shape[2]
        out = torch.empty((n, heads, hdim), dtype=self.dtype, device=self.device)
        if n:
            _arc_gather_fp4[(triton.cdiv(n, 32), heads)](
                self.packed, self.scales, self.global_scale, index.contiguous(), out, n, heads, hdim,
                BLOCK=32, HALF=triton.next_power_of_2(hdim // 2), num_warps=4)
        return out
'''

DISPATCH_OLD = '''    _, heads, dim = k.shape
    block_topk = 16
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))]('''
DISPATCH_NEW = '''    _, heads, dim = k.shape
    block_topk = 16
    if getattr(k, "arc_fp4", False):  # arc-kv4
        _arc_compact_kv_fp4[(batch, heads, triton.cdiv(topk, block_topk))](
            k.packed, v.packed, k.scales, v.scales, k.global_scale, v.global_scale, req_to_token, req_indices,
            indices, seq_lens, cu_k, out_k, out_v, topk, heads, dim, req_to_token.stride(0), indices.stride(0),
            BLOCK_TOPK=block_topk, HALF=triton.next_power_of_2(dim // 2), num_warps=4)
        return
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))]('''

edit("layers/attention/qsa/sparse_attn.py", [
    ("\n\ndef qwen_sparse_valid_counts_triton(", KERNELS + "\n\ndef qwen_sparse_valid_counts_triton("),
    (DISPATCH_OLD, DISPATCH_NEW)])

HELPER = '''

def _arc_kv_buffers(pool, layer):  # arc-kv4
    """Plain pools: get_key/value_buffer. NVFP4 pools: raw packed data + block scales + the layer's global scale."""
    full = getattr(pool, "full_kv_pool", pool)
    qm = getattr(full, "quant_method", None)
    if getattr(qm, "name", None) != "nvfp4" or getattr(full, "dq_k_buffer", 0) is not None:
        return pool.get_key_buffer(layer.layer_id), pool.get_value_buffer(layer.layer_id)
    from sglang.srt.layers.attention.qsa.sparse_attn import ArcFP4KV

    kq, vq, ks, vs = pool.get_raw_kv_buffer(layer.layer_id)
    lid = layer.layer_id
    return (ArcFP4KV(kq, ks, qm.k_scales_gpu[lid : lid + 1], layer.head_dim),
            ArcFP4KV(vq, vs, qm.v_scales_gpu[lid : lid + 1], layer.head_dim))
'''
edit("layers/attention/qwen_sparse_attn_backend.py", [
    ("\n\n_TRTLLM_SPARSE_PAGE_SIZE = 64\n", HELPER + "\n\n_TRTLLM_SPARSE_PAGE_SIZE = 64\n"),
    ("        k_buffer = pool.get_key_buffer(layer.layer_id)\n        v_buffer = pool.get_value_buffer(layer.layer_id)\n",
     "        k_buffer, v_buffer = _arc_kv_buffers(pool, layer)  # arc-kv4\n")])

CAL = '''        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        # arc-kv4: per-layer global scales from the first eager write (amax x headroom), before it is quantized.
        if k_scale is not None and not torch.cuda.is_current_stream_capturing():
            cal = self.__dict__.setdefault("_arc_cal", set())
            if k_scale.data_ptr() not in cal and cache_k.numel():
                import os

                head = float(os.environ.get("ARC_KV4_HEADROOM", "16")) / (E2M1_MAX * 448.0)
                k_scale.copy_((cache_k.detach().abs().amax().float() * head).clamp_(min=1e-12).reshape(1))
                v_scale.copy_((cache_v.detach().abs().amax().float() * head).clamp_(min=1e-12).reshape(1))
                cal.add(k_scale.data_ptr())
                print(f"[arc-kv4] layer scales k={k_scale.item():.3e} v={v_scale.item():.3e} "
                      f"from {cache_k.shape[0]} tokens", flush=True)

        cache_k, cache_k_fp4_sf, _ = NVFP4KVQuantizeUtil.quantize(
            cache_k.contiguous(), k_scale
        )'''
edit("layers/quantization/fp4_kv_cache_quant_method.py", [
    ('''        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        cache_k, cache_k_fp4_sf, _ = NVFP4KVQuantizeUtil.quantize(
            cache_k.contiguous(), k_scale
        )''', CAL),
    ('''    NVFP4KVCacheMethod.name: (
        _dq_workspace(_PREFILL, _NVFP4_PREFILL_BACKENDS, _NVFP4_SCALE, _FP8_E4M3),
        _native_fp4(_DECODE, _NVFP4_DECODE_BACKENDS, _NVFP4_SCALE, _TORCH_FP4),
    ),''', '''    NVFP4KVCacheMethod.name: (  # arc-kv4: QSA reads packed FP4 itself; no FP8 dequant workspace
        _native_fp4(_DECODE, _NVFP4_DECODE_BACKENDS, _NVFP4_SCALE, _TORCH_FP4),
    ),''')])
