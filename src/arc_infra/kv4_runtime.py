"""4-bit QSA main KV cache for the pinned Qwen3.8-Flash-Next vLLM runtime (29 September; kv-cache-4bit.md).

Copied into ``vllm/arc_kv4.py`` by ``arc_infra.serving_kernels_patch`` only when ``server.serving_kernels`` sets
``kv_cache_4bit`` ("int4" or "k8v4"). The split-K merge kernel, launch profile and references are the removed kv8
runtime's, source unchanged. The mode is read from ``vllm/arc_serving_kernels.json``; the patched
``nvidia/qsa.py`` calls the entry points below. vLLM imports are deferred, so the kernels load standalone (CPU tests
under ``TRITON_INTERPRET=1``).

Modes and pages (per token and KV head; the scales live inside the page, so the hybrid cache's byte-level block
operations need no change, as for kv8):

- ``int4`` (``--kv-cache-dtype int4_per_token_head``, the pinned runtime's own sub-byte layout):
  ``[K 128 B | K scale | V 128 B | V scale]`` = 264 B against 1,024 in BF16 and 520 in 8 bits. K and V are rotated
  by the pinned random Hadamard transform (``single_rht``: sign flip, then the unnormalized 256-point Hadamard) and
  quantized asymmetrically to 4 bits: scale = (max - min) / 15, zero point round(-min / scale) in [0, 15], codes
  round(x / scale + zp) in [0, 15], two per byte (dim 2j in the low nibble, 2j + 1 in the high one), the zero point
  in the scale's four low mantissa bits (the pinned ``_reshape_cache_int4_kernel`` arithmetic).
- ``k8v4`` (``--kv-cache-dtype int8_per_token_head`` at the vLLM level; the page is this module's):
  ``[K 256 x int8 | K scale | V 128 B | V scale]`` = 392 B. Keys as the kv8 writer (symmetric, amax / 127, no
  rotation); values as ``int4``.

- Write: ``_kv4_write_kernel`` after the rotation (torch; the pinned ``single_rht`` on the GPU).
- Read: ``_qsa_kv4_splitk_kernel``, the kv8 split-K kernel with packed loads. The codes minus the zero point are
  integers in [-15, 15] (or int8), exact in BF16, so the MMAs take them and the scales are applied per column, as
  kv8. With rotated keys the query is rotated too (``rht(q) / 256``: the dot product is then the unrotated one); the
  rotated values give a rotated output, which the inverse transform (``rht(o, inverse) / 256``) undoes in FP32.
- The indexer caches (raw-key ring, compressed keys) stay BF16.
"""

import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

CONFIG_PATH = Path(__file__).with_name("arc_serving_kernels.json")
CONFIG_KEY = "kv_cache_4bit"
CACHE_DTYPES = {"int4": "int4_per_token_head", "k8v4": "int8_per_token_head"}
HEAD_DIM = 256
SCALE_BYTES = 4
# Bytes of each part of a (token, head) page: K codes, K scale, V codes, V scale.
KEY_BYTES = {"int4": HEAD_DIM // 2, "k8v4": HEAD_DIM}
VALUE_BYTES = HEAD_DIM // 2
_STATE = {"config": None, "counts": {}}


def load_config():
    if _STATE["config"] is None:
        _STATE["config"] = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    return _STATE["config"]


def log(kind, payload):
    print(f"ARC_KV4_{kind} " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def configured_mode():
    return load_config().get(CONFIG_KEY)


def configured_cache_dtype():
    """The ``--kv-cache-dtype`` this overlay serves (None: installed without kv_cache_4bit)."""
    mode = configured_mode()
    return None if mode is None else CACHE_DTYPES[mode]


def allowed(cache_dtype):
    """True only for the dtype the settings name; BF16 ("auto"/"bfloat16") is never routed here."""
    return cache_dtype is not None and cache_dtype == configured_cache_dtype()


def content_bytes(mode):
    return KEY_BYTES[mode] + SCALE_BYTES + VALUE_BYTES + SCALE_BYTES


def offsets(mode):
    """Byte offsets of (K codes, K scale, V codes, V scale) inside one (token, head) page."""
    k = KEY_BYTES[mode]
    return 0, k, k + SCALE_BYTES, k + SCALE_BYTES + VALUE_BYTES


# ============================================================================ backend / impl plumbing
def impl_init_args(args, kwargs):
    """As kv8: ``FlashAttentionImpl.__init__`` is built with "auto" and the 4-bit dtype restored afterwards."""
    args = list(args)
    if "kv_cache_dtype" in kwargs:
        dtype = kwargs["kv_cache_dtype"]
        if allowed(dtype):
            _warm_rht()
            return tuple(args), {**kwargs, "kv_cache_dtype": "auto"}, dtype
        return tuple(args), kwargs, None
    if len(args) > 6 and allowed(args[6]):
        dtype, args[6] = args[6], "auto"
        _warm_rht()
        return tuple(args), kwargs, dtype
    return tuple(args), kwargs, None


def _warm_rht():
    """Create the pinned RHT sign vector on the worker's GPU at model construction, never first inside a CUDA-graph
    capture (a host-to-device copy there is illegal); check the device once (``check_device``)."""
    if torch.cuda.is_available():
        from vllm.v1.attention.ops import int4_per_token_head as ops

        device = torch.device("cuda", torch.cuda.current_device())
        if not _STATE.get("device"):
            _STATE["device"] = check_device(torch.cuda.get_device_capability(device), ops._hadacore_available(),
                                            torch.cuda.get_device_name(device))
        ops._get_rht_signs(HEAD_DIM, 0, device)


# Architectures this overlay serves (E1, 26 Oct): the same Triton sources, compiled per device. SM120: RTX PRO 6000
# (the competition card, where the GPU writer was checked equal to the pinned writer); SM100/SM103: B200/B300 (the RL
# play card). The 4-bit pages are bitwise the same only when the rotation is the same: the pinned single_rht takes
# Hadacore (BF16 tensor-core MMAs, the PTX build) on CUDA; without it the pinned dispatch falls back to an FP32 torch
# butterfly for d = 256 (its Triton tier stops at d = 128), whose BF16 ties differ (2.1% of bytes measured on RTX).
SUPPORTED_ARCH = {10: "SM100/SM103 (B200, B300)", 12: "SM120 (RTX PRO 6000)"}


def check_device(capability, hadacore, name=""):
    """Refuse a device the 4-bit pages were not built for, or one whose rotation is not Hadacore's; log the device."""
    major, minor = capability
    row = {"device": name, "capability": f"{major}.{minor}", "arch": SUPPORTED_ARCH.get(major), "hadacore": hadacore,
           "mode": configured_mode()}
    if major not in SUPPORTED_ARCH:
        raise NotImplementedError(f"ARC kv4 serves {sorted(SUPPORTED_ARCH.values())}, not SM{major}{minor} ({name})")
    if not hadacore:
        raise RuntimeError(f"ARC kv4 on SM{major}{minor} ({name}): the pinned Hadacore transform is not available, so "
                           "the rotation would fall back to the FP32 butterfly and the pages would differ from SM120's")
    log("DEVICE", row)
    return row


def customize_spec(spec):
    """The page of the configured mode (inline scales) as the spec's content bytes."""
    mode = spec.kv_quant_mode
    if spec.state_content_bytes is not None or not mode.is_per_token_head:
        return spec
    from dataclasses import replace

    from vllm.v1.kv_cache_interface import KVQuantMode

    kv = configured_mode()
    expected = {"int4": KVQuantMode.INT4_PER_TOKEN_HEAD, "k8v4": KVQuantMode.INT8_PER_TOKEN_HEAD}.get(kv)
    if mode != expected or spec.head_size != HEAD_DIM or spec.head_size_v != HEAD_DIM:
        raise NotImplementedError(f"ARC kv4 ({kv}) cannot serve {mode} with head sizes {spec.head_size}")
    return replace(spec, state_content_bytes=content_bytes(kv))


def kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
    """Logical (blocks, heads, tokens, content) as the pinned FlashAttention/Triton backends."""
    if block_size % 16 != 0:
        raise ValueError("Block size must be a multiple of 16.")
    if str(cache_dtype_str).endswith("per_token_head"):
        if not allowed(cache_dtype_str) or head_size != HEAD_DIM:
            raise NotImplementedError(f"ARC kv4 serves {configured_cache_dtype()}, not {cache_dtype_str}")
        return (num_blocks, num_kv_heads, block_size, content_bytes(configured_mode()))
    return (num_blocks, num_kv_heads, block_size, 2 * head_size)


def cache_views(kv_cache, mode, holder=None):
    """(K codes, V codes, K scales, V scales) over one packed cache tensor, as strided views.

    ``kv_cache``: logical ``(blocks, kv_heads, tokens, content)`` of a one-byte dtype (NHD or HND strides). K codes
    ``(blocks, tokens, kv_heads, 128)`` uint8 (int4) or ``(..., 256)`` int8 (k8v4); V codes ``(..., 128)`` uint8;
    scales ``(blocks, tokens, kv_heads)``: int32 bit patterns for the 4-bit parts (scale with the zero point in the
    low bits), float32 for k8v4 keys.
    """
    key = (kv_cache.data_ptr(), tuple(kv_cache.shape), tuple(kv_cache.stride()), kv_cache.storage_offset(), mode)
    cached = getattr(holder, "_arc_kv4_views", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if kv_cache.ndim != 4 or kv_cache.element_size() != 1:
        raise ValueError("ARC kv4 expects a 4-D one-byte packed cache")
    num_blocks, num_heads, block_size, content = kv_cache.shape
    if content != content_bytes(mode):
        raise ValueError(f"ARC kv4 cache content {content} does not match the {mode} page")
    k_off, ks_off, v_off, vs_off = offsets(mode)
    tokens_first = kv_cache.view(torch.uint8).transpose(1, 2)
    keys = tokens_first[..., k_off:ks_off]
    if mode == "k8v4":
        keys = keys.view(torch.int8)
    values = tokens_first[..., v_off:vs_off]
    strides = kv_cache.stride()
    offset = kv_cache.storage_offset()
    if any(s % 4 for s in strides[:3]) or offset % 4:
        raise ValueError("ARC kv4 scale slots are not 4-byte aligned")
    size = (num_blocks, block_size, num_heads)
    stride = (strides[0] // 4, strides[2] // 4, strides[1] // 4)
    words = torch.empty(0, dtype=torch.int32, device=kv_cache.device).set_(kv_cache.untyped_storage())
    floats = torch.empty(0, dtype=torch.float32, device=kv_cache.device).set_(kv_cache.untyped_storage())
    k_scale = torch.as_strided(floats if mode == "k8v4" else words, size=size, stride=stride,
                               storage_offset=(offset + ks_off) // 4)
    v_scale = torch.as_strided(words, size=size, stride=stride, storage_offset=(offset + vs_off) // 4)
    views = (keys, values, k_scale, v_scale)
    if holder is not None:
        holder._arc_kv4_views = (key, views)
    return views


def _count(name):
    first = name not in _STATE["counts"]
    _STATE["counts"][name] = _STATE["counts"].get(name, 0) + 1
    if first:
        log("FIRST_CALL", {"entry": name, "mode": configured_mode(), "cache_dtype": configured_cache_dtype()})


# ============================================================================ random Hadamard transform
_SIGNS = {}


def rht_signs(d, device):
    """The pinned ``_get_rht_signs(d, 0, device)``: deterministic +-1 from a seeded CPU generator."""
    key = (d, str(device))
    if key not in _SIGNS:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0x9E3779B9)
        _SIGNS[key] = (2.0 * torch.bernoulli(torch.full((d,), 0.5), generator=gen) - 1.0).to(device)
    return _SIGNS[key]


def hadamard_reference(x):
    """Unnormalized Walsh-Hadamard transform along the last dimension (the pinned tier-3 butterfly)."""
    d = x.shape[-1]
    h = 1
    while h < d:
        xv = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        x = torch.stack([xv[..., 0, :] + xv[..., 1, :], xv[..., 0, :] - xv[..., 1, :]], dim=-2).reshape(x.shape)
        h <<= 1
    return x


def rht_reference(x, inverse=False):
    """The pinned ``single_rht``: forward H (D x), inverse D (H x); each has norm sqrt(d)."""
    d1 = rht_signs(x.shape[-1], x.device)
    return hadamard_reference(x) * d1 if inverse else hadamard_reference(x * d1)


def rht(x, inverse=False):
    """The pinned ``single_rht`` on the GPU (Hadacore), the reference elsewhere."""
    if x.is_cuda:
        from vllm.v1.attention.ops.int4_per_token_head import single_rht

        return single_rht(x, inverse)
    return rht_reference(x, inverse)


# ============================================================================ write
@triton.jit
def _quant4(x_even, x_odd):
    """The pinned asymmetric 4-bit quantizer: (codes even, codes odd, scale bits with the zero point)."""
    lo = tl.minimum(tl.min(x_even), tl.min(x_odd))
    hi = tl.maximum(tl.max(x_even), tl.max(x_odd))
    scale = tl.maximum((hi - lo) / 15.0, 1e-6)
    zp_raw = -lo / scale
    zp = tl.clamp(tl.where(zp_raw >= 0, (zp_raw + 0.5).to(tl.int32), (zp_raw - 0.5).to(tl.int32)).to(tl.float32),
                  0.0, 15.0)
    inv = 1.0 / scale
    s_even = x_even * inv + zp
    s_odd = x_odd * inv + zp
    q_even = tl.minimum(tl.maximum(tl.where(s_even >= 0, (s_even + 0.5).to(tl.int32), (s_even - 0.5).to(tl.int32)),
                                   0), 15)
    q_odd = tl.minimum(tl.maximum(tl.where(s_odd >= 0, (s_odd + 0.5).to(tl.int32), (s_odd - 0.5).to(tl.int32)),
                                  0), 15)
    bits = (scale.to(tl.int32, bitcast=True) & -16) | (zp.to(tl.int32) & 0xF)
    return q_even, q_odd, bits


@triton.jit
def _kv4_write_kernel(
    key_ptr, value_ptr, k_codes_ptr, v_codes_ptr, k_scale_ptr, v_scale_ptr, slot_mapping_ptr,
    stride_key_tok, stride_key_head, stride_val_tok, stride_val_head,
    stride_kc_blk, stride_kc_slot, stride_kc_head, stride_vc_blk, stride_vc_slot, stride_vc_head,
    stride_ks_blk, stride_ks_slot, stride_ks_head, stride_vs_blk, stride_vs_slot, stride_vs_head,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, KEY_BITS: tl.constexpr,
) -> None:
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot >= 0:
        blk = slot // BLOCK_SIZE
        pos = slot % BLOCK_SIZE
        half = tl.arange(0, HEAD_DIM // 2)
        key_base = key_ptr + tok * stride_key_tok + head * stride_key_head
        k_codes = k_codes_ptr + blk * stride_kc_blk + pos * stride_kc_slot + head * stride_kc_head
        k_scale = k_scale_ptr + blk * stride_ks_blk + pos * stride_ks_slot + head * stride_ks_head
        if KEY_BITS == 4:
            k_even = tl.load(key_base + 2 * half).to(tl.float32)
            k_odd = tl.load(key_base + 2 * half + 1).to(tl.float32)
            qe, qo, bits = _quant4(k_even, k_odd)
            tl.store(k_scale, bits)
            tl.store(k_codes + half, (qe | (qo << 4)).to(tl.uint8))
        else:
            # The pinned per-token-head int8 writer: amax / 127, round half away from zero, clamp.
            dims = tl.arange(0, HEAD_DIM)
            k = tl.load(key_base + dims).to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(k)) / 127.0, 1e-6)
            tl.store(k_scale, scale)
            kq = k * (1.0 / scale)
            kq = tl.clamp(tl.where(kq >= 0, kq + 0.5, kq - 0.5), -128.0, 127.0)
            tl.store(k_codes + dims, kq.to(tl.int32).to(tl.int8))
        value_base = value_ptr + tok * stride_val_tok + head * stride_val_head
        v_even = tl.load(value_base + 2 * half).to(tl.float32)
        v_odd = tl.load(value_base + 2 * half + 1).to(tl.float32)
        ve, vo, vbits = _quant4(v_even, v_odd)
        tl.store(v_scale_ptr + blk * stride_vs_blk + pos * stride_vs_slot + head * stride_vs_head, vbits)
        tl.store(v_codes_ptr + blk * stride_vc_blk + pos * stride_vc_slot + head * stride_vc_head + half,
                 (ve | (vo << 4)).to(tl.uint8))


def write(key, value, k_codes, v_codes, k_scale, v_scale, slot_mapping, mode):
    """Rotate (int4: K and V; k8v4: V) and quantize ``[tokens, kv_heads, 256]`` rows into the cache views."""
    if key.shape[-1] != HEAD_DIM or value.shape != key.shape:
        raise ValueError("ARC kv4 writes [tokens, kv_heads, 256] K/V rows")
    rows = slot_mapping.shape[0]
    key, value = key[:rows], value[:rows]
    if not rows:
        return
    value = rht(value.float()).to(value.dtype).contiguous()
    key = rht(key.float()).to(key.dtype).contiguous() if mode == "int4" else key
    if key.stride(-1) != 1 or k_codes.stride(-1) != 1 or v_codes.stride(-1) != 1:
        raise ValueError("ARC kv4 needs unit-stride head dims")
    _kv4_write_kernel[(rows, key.shape[1])](
        key, value, k_codes, v_codes, k_scale, v_scale, slot_mapping,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        k_codes.stride(0), k_codes.stride(1), k_codes.stride(2), v_codes.stride(0), v_codes.stride(1),
        v_codes.stride(2), k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), v_scale.stride(0),
        v_scale.stride(1), v_scale.stride(2),
        BLOCK_SIZE=k_codes.shape[1], HEAD_DIM=HEAD_DIM, KEY_BITS=4 if mode == "int4" else 8, num_warps=4,
    )


def cache_update(impl, key, value, kv_cache, slot_mapping):
    """Patched ``do_kv_cache_update``: rotate and quantize this step's K/V rows into the 4-bit page."""
    _count("cache_update")
    mode = configured_mode()
    rows = slot_mapping.shape[0]
    if rows >= SAMPLE_MIN_ROWS and not _capturing(key):
        _sample(impl, key[:rows], value[:rows])
    write(key, value, *cache_views(kv_cache, mode, impl), slot_mapping, mode)


def _capturing(tensor):
    """True inside a CUDA-graph capture: the B300 RL card captures up to 1,024 tokens, so a >= 512-row step can be a
    capture, where the sample's host copies and syncs are illegal (the competition card captures <= 160)."""
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


# Real K/V round trip (logged, never changes the output): the first prefill chunks of each QSA layer (>= 512 rows,
# outside any CUDA-graph capture).
SAMPLE_MIN_ROWS = 512
SAMPLES_PER_LAYER = 2
SAMPLE_MAX_ROWS = 4096


def self_check(device, mode):
    """Once per server, outside any graph capture: this module's GPU writer against the torch reference page, the
    pinned int4 writer's page (int4), and the GPU reader against FP32 dequantize-then-attend and the BF16 cache."""
    q, k, v, indices, table, t2r = synthetic_case(4, page=64, blocks=16, topk=203, requests=2, device=device)
    expected = pack_cache(k.cpu(), v.cpu(), mode).to(device)
    written = torch.zeros_like(expected)
    slots = torch.arange(k.shape[0] * k.shape[1], device=device, dtype=torch.int64)
    rows_k, rows_v = k.reshape(-1, 2, HEAD_DIM), v.reshape(-1, 2, HEAD_DIM)
    write(rows_k, rows_v, *cache_views(written, mode), slots, mode)
    result = {"mode": mode, "writer_bytes_equal_reference": float((written.view(torch.uint8) == expected.view(
        torch.uint8)).float().mean())}
    if mode == "int4":
        from vllm.v1.attention.ops.int4_per_token_head import reshape_and_cache_int4

        pinned = torch.zeros_like(expected)
        pk, pv, pks, pvs = cache_views(pinned, mode)
        reshape_and_cache_int4(rows_k, rows_v, pk, pv, slots, k_scale_cache=pks.view(torch.float32),
                               v_scale_cache=pvs.view(torch.float32))
        result["writer_bytes_equal_pinned"] = float((written == pinned).float().mean())
    out = torch.zeros_like(q)
    attend(q, *cache_views(written, mode), indices, table, t2r, out, mode)
    dk, dv = dequantize_cache(written.cpu(), mode)
    reference = attention_reference(q.cpu(), dk, dv, indices.cpu(), table.cpu(), t2r.cpu())
    exact = attention_reference(q.cpu(), k.cpu(), v.cpu(), indices.cpu(), table.cpu(), t2r.cpu())
    result["reader_rel_error_vs_dequant"] = _relative(reference, out.float().cpu())
    result["reader_rel_error_vs_bf16"] = _relative(exact, out.float().cpu())
    return result


def _sample(impl, key, value):
    if not _STATE.get("self_checked"):
        _STATE["self_checked"] = True
        try:
            log("SELFCHECK", self_check(key.device, configured_mode()))
        except Exception as exc:  # noqa: BLE001 - evidence only
            log("SELFCHECK_ERROR", {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    layers = _STATE.setdefault("layers", {})
    index = layers.setdefault(id(impl), len(layers))
    seen = _STATE.setdefault("samples", {})
    if seen.get(index, 0) >= SAMPLES_PER_LAYER:
        return
    seen[index] = seen.get(index, 0) + 1
    try:
        stats = round_trip_stats(key[:SAMPLE_MAX_ROWS], value[:SAMPLE_MAX_ROWS])
        log("ROUNDTRIP", {"layer_order": index, "sample": seen[index], "rows": int(min(key.shape[0], SAMPLE_MAX_ROWS)),
                          **stats})
    except Exception as exc:  # noqa: BLE001 - evidence only
        log("ROUNDTRIP_ERROR", {"layer_order": index, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})


def _relative(x, y):
    return float((x - y).norm() / x.norm().clamp_min(1e-20))


def round_trip_stats(key, value):
    """Relative round-trip error (Frobenius over the rows) of K and V ``[rows, heads, 256]`` through each part:
    8-bit (kv8 / k8v4 keys), 4-bit after the rotation (int4, k8v4 values) and 4-bit without it; plus the mean
    per-(token, head) peak-to-RMS ratio (outlier channels)."""
    out = {}
    for name, x in (("k", key.float()), ("v", value.float())):
        rotated = rht(x).to(torch.bfloat16).float()
        back = rht(dequantize4_reference(*quantize4_reference(rotated)), inverse=True) / HEAD_DIM
        codes, scale = quantize_reference(x, "int8")
        out[f"{name}_int8"] = _relative(x, dequantize_reference(codes, scale))
        out[f"{name}_int4_rht"] = _relative(x, back)
        out[f"{name}_int4_plain"] = _relative(x, dequantize4_reference(*quantize4_reference(x)))
        rms = x.pow(2).mean(dim=-1).sqrt().clamp_min(1e-20)
        out[f"{name}_peak_to_rms"] = float((x.abs().amax(dim=-1) / rms).mean())
    return out


# ============================================================================ read
@triton.jit
def _qsa_kv4_splitk_kernel(
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KEY_BITS: tl.constexpr, INTERPRET: tl.constexpr = False,
) -> None:
    # The kv8 split-K kernel with packed 4-bit values (and keys for KEY_BITS == 4). Head dimension j of a packed
    # part is 2 * (j // 2) + (j % 2): even dims in the low nibbles, odd dims in the high ones, so the query and the
    # output are split into their even and odd halves. INTERPRET (CPU tests only): FP32 dots of the same
    # BF16-rounded operands (the interpreter's BF16 dot and integer->BF16 casts are wrong).
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HEAD_DIM // 2)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    q_rows = q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head
    if KEY_BITS == 4:
        q_even = tl.load(q_rows + 2 * half_offsets[None, :], mask=head_mask, other=0.0)
        q_odd = tl.load(q_rows + 2 * half_offsets[None, :] + 1, mask=head_mask, other=0.0)
        query = q_even  # unused
    else:
        query = tl.load(q_rows + dim_offsets[None, :], mask=head_mask, other=0.0)
        q_even = query  # unused
        q_odd = query  # unused

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_even = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
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
        key_base = k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token \
            + kv_head * stride_k_head
        key_scale_raw = tl.load(
            k_scale_ptr + safe_page * stride_ks_block + page_offset * stride_ks_token + kv_head * stride_ks_head,
            mask=valid,
            other=0,
        )
        if KEY_BITS == 4:
            packed = tl.load(key_base + half_offsets[:, None], mask=valid[None, :], other=0).to(tl.int32)
            key_zero = key_scale_raw & 0xF
            key_scale = (key_scale_raw & -16).to(tl.float32, bitcast=True)
            k_even = ((packed & 0xF) - key_zero[None, :]).to(tl.float32)
            k_odd = (((packed >> 4) & 0xF) - key_zero[None, :]).to(tl.float32)
            if INTERPRET:
                scores = tl.dot(q_even.to(tl.float32), k_even) + tl.dot(q_odd.to(tl.float32), k_odd)
            else:
                scores = tl.dot(q_even, k_even.to(tl.bfloat16)) + tl.dot(q_odd, k_odd.to(tl.bfloat16))
        else:
            key_codes = tl.load(key_base + dim_offsets[:, None], mask=valid[None, :], other=0)
            key_scale = key_scale_raw
            if INTERPRET:
                scores = tl.dot(query.to(tl.float32), key_codes.to(tl.float32))
            else:
                scores = tl.dot(query, key_codes.to(tl.bfloat16))
        scores *= key_scale[None, :] * softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)

        value_packed = tl.load(
            v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.int32)
        value_scale_raw = tl.load(
            v_scale_ptr + safe_page * stride_vs_block + page_offset * stride_vs_token + kv_head * stride_vs_head,
            mask=valid,
            other=0,
        )
        value_zero = value_scale_raw & 0xF
        value_scale = (value_scale_raw & -16).to(tl.float32, bitcast=True)
        v_even = ((value_packed & 0xF) - value_zero[:, None]).to(tl.float32)
        v_odd = (((value_packed >> 4) & 0xF) - value_zero[:, None]).to(tl.float32)
        weights = (probabilities * value_scale[None, :]).to(tl.bfloat16)
        if INTERPRET:
            acc_even = tl.dot(weights.to(tl.float32), v_even, acc=acc_even * alpha[:, None])
            acc_odd = tl.dot(weights.to(tl.float32), v_odd, acc=acc_odd * alpha[:, None])
        else:
            acc_even = tl.dot(weights, v_even.to(tl.bfloat16), acc=acc_even * alpha[:, None])
            acc_odd = tl.dot(weights, v_odd.to(tl.bfloat16), acc=acc_odd * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    out_even = tl.where(has_values[:, None], acc_even / denominator, 0.0)
    out_odd = tl.where(has_values[:, None], acc_odd / denominator, 0.0)
    even_columns = 2 * half_offsets[None, :]
    if NUM_SPLITS == 1:
        out_rows = output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
        tl.store(out_rows + even_columns, out_even, mask=head_mask)
        tl.store(out_rows + even_columns + 1, out_odd, mask=head_mask)
    else:
        partial_lse = tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        partial_rows = partial_output_ptr + ((split_id * num_rows + row) * NUM_QUERY_HEADS + first_head
                                             + head_offsets[:, None]) * HEAD_DIM
        tl.store(partial_rows + even_columns, out_even, mask=head_mask)
        tl.store(partial_rows + even_columns + 1, out_odd, mask=head_mask)
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


def sparse_attention(q, k_codes, v_codes, k_scale, v_scale, logical_indices, block_table, token_to_req, out,
                     key_bits, profile=None):
    """Sparse GQA over the packed cache views into the FP32 ``out`` (the rotated output when V is rotated).

    ``q`` [rows, heads, 256] BF16 (already rotated and scaled when the keys are); K codes [blocks, page, kv_heads,
    128] uint8 (key_bits 4) or [..., 256] int8 (key_bits 8); V codes [..., 128] uint8; scales [blocks, page,
    kv_heads] (int32 bit patterns for 4-bit parts, float32 for 8-bit keys); indices, table, token_to_req as kv8.
    """
    if q.ndim != 3 or k_codes.ndim != 4 or v_codes.ndim != 4 or q.shape[2] != HEAD_DIM:
        raise ValueError("QSA kv4 attention received invalid Q/K/V shapes")
    if k_codes.shape[3] != (HEAD_DIM // 2 if key_bits == 4 else HEAD_DIM) or v_codes.shape[3] != HEAD_DIM // 2:
        raise ValueError("QSA kv4 codes do not match the page")
    if k_scale.shape != k_codes.shape[:3] or v_scale.shape != k_scale.shape or v_codes.shape[:3] != k_scale.shape:
        raise ValueError("QSA kv4 scales must be [blocks, page, kv_heads]")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0] or logical_indices.shape[1] <= 0:
        raise ValueError("QSA kv4 indices must have one positive-width row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2 or q.shape[1] % k_codes.shape[2]:
        raise ValueError("QSA kv4 metadata has invalid shapes")
    assert q.dtype == torch.bfloat16 and out.dtype == torch.float32 and out.shape == q.shape
    assert v_codes.dtype == torch.uint8 and v_scale.dtype == torch.int32
    assert (k_codes.dtype, k_scale.dtype) == ((torch.uint8, torch.int32) if key_bits == 4 else (torch.int8, torch.float32))
    assert logical_indices.dtype == block_table.dtype == token_to_req.dtype == torch.int32
    assert q.stride(2) == k_codes.stride(3) == v_codes.stride(3) == out.stride(2) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1 and token_to_req.stride(0) == 1
    if not q.shape[0]:
        return out
    group_size = q.shape[1] // k_codes.shape[2]
    p = profile or launch_profile(q.shape[0], k_codes.shape[2], group_size, logical_indices.shape[1])
    if p["splits"] == 1:
        partial_output, partial_lse = out, out
    else:
        partial_output = torch.empty((p["splits"], *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((p["splits"], q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    _qsa_kv4_splitk_kernel[(q.shape[0], k_codes.shape[2], p["splits"])](
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
        GROUP_SIZE=group_size, HEAD_DIM=HEAD_DIM, NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=p["splits"],
        NUM_TILES=p["tiles"], BLOCK_M=p["block_m"], BLOCK_N=p["block_n"], KEY_BITS=key_bits,
        INTERPRET=_interpreting(), num_warps=p["warps"], num_stages=2,
    )
    if p["splits"] > 1:
        _qsa_kv8_merge_kernel[(q.shape[0], q.shape[1])](
            partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
            HEAD_DIM=HEAD_DIM, NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=p["splits"],
            BLOCK_SPLITS=triton.next_power_of_2(p["splits"]), num_warps=2, num_stages=1,
        )
    return out


def attend(query, k_codes, v_codes, k_scale, v_scale, logical_indices, block_table, token_to_req, out, mode,
           profile=None):
    """Rotate the query (int4), attend, inverse-rotate the output into ``out`` (BF16)."""
    q = query
    if mode == "int4":
        q = (rht(query.float()) / HEAD_DIM).to(torch.bfloat16)
    rotated = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    sparse_attention(q, k_codes, v_codes, k_scale, v_scale, logical_indices, block_table, token_to_req, rotated,
                     4 if mode == "int4" else 8, profile)
    out.copy_(rht(rotated, inverse=True) / HEAD_DIM)
    return out


def attention(impl, query, kv_cache, logical_indices, block_table, token_to_req, out):
    """Patched ``forward_qsa`` body: sparse GQA over the 4-bit cache."""
    _count("attention")
    mode = configured_mode()
    if not query.shape[0]:
        return out
    return attend(query, *cache_views(kv_cache, mode, impl), logical_indices, block_table, token_to_req, out, mode)


# ============================================================================ references (tests)
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


def quantize4_reference(x):
    """Torch mirror of the 4-bit writer on (already rotated) rows ``[..., 256]``: (packed uint8 [..., 128], scale
    bits int32 [...])."""
    x = x.float()
    lo, hi = x.amin(dim=-1), x.amax(dim=-1)
    scale = ((hi - lo) / 15.0).clamp_min(1e-6)
    zp_raw = -lo / scale
    zp = torch.where(zp_raw >= 0, (zp_raw + 0.5).trunc(), (zp_raw - 0.5).trunc()).clamp(0, 15)
    s = x * (1.0 / scale).unsqueeze(-1) + zp.unsqueeze(-1)
    q = torch.where(s >= 0, (s + 0.5).trunc(), (s - 0.5).trunc()).clamp(0, 15).to(torch.int32)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)
    bits = (scale.view(torch.int32) & -16) | zp.to(torch.int32)
    return packed, bits


def dequantize4_reference(packed, bits):
    """[..., 128] packed codes and scale bits -> [..., 256] float32 values (in the rotated basis)."""
    codes = packed.to(torch.int32)
    q = torch.stack([codes & 0xF, (codes >> 4) & 0xF], dim=-1).reshape(*packed.shape[:-1], 2 * packed.shape[-1])
    zp = (bits & 0xF).unsqueeze(-1)
    scale = (bits & -16).view(torch.float32).unsqueeze(-1)
    return (q - zp).float() * scale


def round_trip(x, mode_part):
    """``x`` [..., 256] through one part of the cache and back to the original basis: "int8" (kv8 keys),
    "int4" (rotated 4-bit), "int4-norht" (4-bit without the rotation)."""
    x = x.float()
    if mode_part == "int8":
        codes, scale = quantize_reference(x, "int8")
        return dequantize_reference(codes, scale)
    if mode_part == "int4-norht":
        return dequantize4_reference(*quantize4_reference(x))
    rotated = rht_reference(x).to(torch.bfloat16).float()
    return rht_reference(dequantize4_reference(*quantize4_reference(rotated)), inverse=True) / HEAD_DIM


def pack_cache(k, v, mode):
    """Quantize BF16 pages ``[blocks, page, kv_heads, 256]`` into one packed cache in the served layout (NHD
    physical strides), through the torch references: the logical ``(blocks, kv_heads, page, content)`` tensor."""
    blocks, page, kv_heads, dim = k.shape
    raw = torch.zeros((blocks, page, kv_heads, content_bytes(mode)), dtype=torch.uint8, device=k.device)
    k_off, ks_off, v_off, vs_off = offsets(mode)
    if mode == "int4":
        kq, kb = quantize4_reference(rht_reference(k.float()).to(torch.bfloat16))
        raw[..., k_off:ks_off] = kq
        raw[..., ks_off:v_off] = kb.contiguous().view(torch.uint8).view(blocks, page, kv_heads, 4)
    else:
        kq, ks = quantize_reference(k, "int8")
        raw[..., k_off:ks_off] = kq.view(torch.uint8)
        raw[..., ks_off:v_off] = ks.contiguous().view(torch.uint8).view(blocks, page, kv_heads, 4)
    vq, vb = quantize4_reference(rht_reference(v.float()).to(torch.bfloat16))
    raw[..., v_off:vs_off] = vq
    raw[..., vs_off:] = vb.contiguous().view(torch.uint8).view(blocks, page, kv_heads, 4)
    return (raw if mode == "int4" else raw.view(torch.int8)).transpose(1, 2)


def dequantize_cache(kv_cache, mode):
    """(K, V) float32 ``[blocks, page, kv_heads, 256]`` in the original basis from a packed cache."""
    keys, values, k_scale, v_scale = cache_views(kv_cache, mode)
    if mode == "int4":
        k = rht_reference(dequantize4_reference(keys, k_scale), inverse=True) / HEAD_DIM
    else:
        k = dequantize_reference(keys, k_scale)
    v = rht_reference(dequantize4_reference(values, v_scale), inverse=True) / HEAD_DIM
    return k, v


# ============================================================================ cross-device golden (E1, 26 Oct)
GOLDEN_ROWS = 1024


def golden_case(rows=GOLDEN_ROWS, seed=20261026):
    """Fixed BF16 K/V rows ``[rows, 2, 256]`` from a seeded CPU generator (unit RMS, four 4x outlier key channels,
    constant rows for the scale floor, tiny rows): the same inputs on every device."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn((rows, 2, HEAD_DIM), generator=g)
    v = torch.randn((rows, 2, HEAD_DIM), generator=g)
    k[..., :4] *= 4.0
    k[:4], v[:4] = 0.25, -0.5
    k[4:8] *= 1e-4
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def _empty_cache(blocks, page, mode, device):
    """A zeroed packed cache in the served layout: logical ``(blocks, kv_heads, page, content)``, NHD strides."""
    return torch.zeros((blocks, page, 2, content_bytes(mode)), dtype=torch.uint8, device=device).transpose(1, 2)


def golden(device, mode="int4", rows=GOLDEN_ROWS):
    """This device's rotated rows, written pages and attention output on the golden inputs (as CPU tensors)."""
    k, v = (x.to(device) for x in golden_case(rows))
    page = _empty_cache(1, rows, mode, device)
    write(k, v, *cache_views(page, mode), torch.arange(rows, device=device, dtype=torch.int64), mode)
    q, ck, cv, indices, table, t2r = synthetic_case(4, page=64, blocks=16, topk=203, requests=2, seed=7,
                                                         device=device)
    packed = _empty_cache(16, 64, mode, device)
    write(ck.reshape(-1, 2, HEAD_DIM), cv.reshape(-1, 2, HEAD_DIM), *cache_views(packed, mode),
          torch.arange(16 * 64, device=device, dtype=torch.int64), mode)
    out = torch.zeros_like(q)
    attend(q, *cache_views(packed, mode), indices, table, t2r, out, mode)
    dk, dv = dequantize_cache(packed.cpu(), mode)
    reference = attention_reference(q.cpu(), dk, dv, indices.cpu(), table.cpu(), t2r.cpu())
    cuda = torch.device(device).type == "cuda"
    return {"mode": mode, "device": torch.cuda.get_device_name(device) if cuda else "cpu",
            "capability": tuple(torch.cuda.get_device_capability(device)) if cuda else None,
            "rotated_k": rht(k.float()).to(torch.bfloat16).cpu(), "rotated_v": rht(v.float()).to(torch.bfloat16).cpu(),
            "page": page.cpu(), "attn_page": packed.cpu(), "attn_out": out.float().cpu(),
            "attn_rel_error_vs_dequant": _relative(reference, out.float().cpu())}


def compare(ours, theirs):
    """Agreement of two ``golden`` results: fractions of equal bytes, and the attention outputs' relative distance."""
    def equal(a, b):
        return float((a.contiguous().view(torch.uint8) == b.contiguous().view(torch.uint8)).float().mean())

    _, ks_off, v_off, vs_off = offsets(ours["mode"])
    page_a, page_b = ours["page"].transpose(1, 2), theirs["page"].transpose(1, 2)
    result = {"devices": [ours["device"], theirs["device"]],
              "rotated_k_equal": equal(ours["rotated_k"], theirs["rotated_k"]),
              "rotated_v_equal": equal(ours["rotated_v"], theirs["rotated_v"]),
              "page_equal": equal(ours["page"], theirs["page"]),
              "k_scale_equal": equal(page_a[..., ks_off:v_off], page_b[..., ks_off:v_off]),
              "v_scale_equal": equal(page_a[..., vs_off:], page_b[..., vs_off:]),
              "attn_page_equal": equal(ours["attn_page"], theirs["attn_page"]),
              "attn_out_rel_diff": _relative(theirs["attn_out"], ours["attn_out"]),
              "attn_rel_error_vs_dequant": [ours["attn_rel_error_vs_dequant"], theirs["attn_rel_error_vs_dequant"]]}
    result["bitwise"] = result["page_equal"] == 1.0 and result["attn_page_equal"] == 1.0
    return result


def main(argv=None):
    """``python -m vllm.arc_kv4 golden OUT.pt``: this device's golden result (a GPU with the pinned runtime, or the CPU
    references); ``python -m vllm.arc_kv4 compare OURS.pt THEIRS.pt``: the agreement, exit 1 unless bitwise equal."""
    import argparse

    p = argparse.ArgumentParser(prog="arc_kv4")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden")
    g.add_argument("out")
    g.add_argument("--mode", default="int4", choices=sorted(CACHE_DTYPES))
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = sub.add_parser("compare")
    c.add_argument("ours")
    c.add_argument("theirs")
    a = p.parse_args(argv)
    if a.cmd == "compare":
        result = compare(torch.load(a.ours), torch.load(a.theirs))
        log("COMPARE", result)
        return 0 if result["bitwise"] else 1
    _STATE["config"] = {CONFIG_KEY: a.mode}
    if torch.device(a.device).type == "cuda":
        from vllm.v1.attention.ops import int4_per_token_head as ops

        check_device(torch.cuda.get_device_capability(), ops._hadacore_available(), torch.cuda.get_device_name())
    result = golden(a.device, a.mode)
    torch.save(result, a.out)
    log("GOLDEN", {"out": a.out, "device": result["device"], "capability": result["capability"],
                   "attn_rel_error_vs_dequant": result["attn_rel_error_vs_dequant"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
