"""Optional dense-GEMM kernels for the pinned Qwen3.8-Flash-Next runtime (kernel work, phase 2).

Copied into ``vllm/models/qwen3_8_flash_next/nvidia/arc_dense_kernels.py`` by
``arc_infra.dense_kernels_patch``; its settings are ``arc_dense_kernels.json`` next to it. The patched
``enable_qwen38next_low_latency_gemm`` (called by the target model and by the MTP draft model at
construction) hands the module to ``enable`` below. No arc dependency inside the serving worker.

Two independent options (docs/validation-20260927/kernels-phase1.md, options A and C):

- ``skinny``: the pinned CuTe-DSL skinny GEMM for BF16 projections at 1 <= M <= 16 on SM120. The pinned
  plan tables list only TP=4 shapes and the gate accepts only SM103; this adds TP=1 entries for M = 1..16
  with the kernel's own default configs (``ShapeDynamicSkinnyGemm._config``), except HC up (K = 320,
  not a multiple of the kernel's K tile). Same BF16 math as cuBLAS (fp32 accumulation, BF16 output).
- ``int8_weights``: weight-only INT8 for the listed large projection families, quantized at load from
  the BF16 checkpoint weights with one scale per output channel (amax / 127). M <= TRITON_MAX_M runs a
  Triton W8A16 split-K kernel (the phase-1 microbenchmark kernel with an M grid); larger M (prefill)
  dequantizes into a bounded BF16 scratch and calls cuBLAS, so no BF16 copy stays resident.

Every family's first layer is checked against an FP32 reference on its real weights at load and the
result is logged as ``ARC_DENSE_CHECK``; a kernel whose error is far outside the expected range stops
the server instead of serving wrong logits.

Kernel work, phase 5 (docs/validation-20260927/kernels-phase5-stack.md), option ``mid_m``: decode sizes the
two paths above serve poorly. At six MTP-3 streams the verify forward and the first draft step run M = 24 (the
served CUDA-graph sizes are 1, 2, 4, 8, 16, 24, 32), where the skinny plan ends (M <= 16) and every small BF16
projection (HC down/up, router, in_proj_ba, indexer, shared expert) falls back to cuBLAS, and the INT8 kernel
runs its 64-row prefill tile. ``mid_m`` routes the small BF16 shapes through the same Triton split-K kernel as
the INT8 path (BF16 weights, unit scale: fp32 accumulation, one BF16 rounding, as cuBLAS) and gives the INT8
families 32-row tiles, per shape and M bucket (8, 16, 24, 32). ``true`` tunes at load: on each shape's first
layer, every candidate tile is timed against the pinned path (CUDA-graph replay over >= 256 MiB of weight
copies) and checked against FP32 on the real weights; a candidate is kept only when it is >= 3% faster and
its error is <= max(1.25 x, +0.05 points) the pinned path's (``ARC_DENSE_MID`` lines). A fixed plan skips the
timing and keeps only the error check. M <= 4 and M > 32 always take the pinned path.
"""

import gc
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

CONFIG_PATH = Path(__file__).with_name("arc_dense_kernels.json")
# (N, K) of every BF16 projection of the TP=1 model that the skinny GEMM can run (phase-1 §2 table).
SKINNY_SHAPES = (
    (16384, 2560),   # GDN in_proj_qkvz
    (13312, 2560),   # QSA qkv (q + gate, k, v)
    (2560, 6144),    # GDN out_proj, QSA o_proj, MTP o_proj
    (336, 10240),    # HC down + inject
    (320, 10240),    # HC final mixer down
    (1280, 2560),    # shared expert gate_up
    (512, 2560),     # router
    (2560, 640),     # shared expert down
    (96, 2560),      # GDN in_proj_ba
    (640, 2560),     # QSA indexer q/k
    (2560, 2560),    # PLE value, MTP fc_embedding / fc_hidden
    (10240, 2560),   # PLE key
    (248320, 2560),  # lm_head (target and MTP draft)
)
SKINNY_MAX_M = 16
# Weight-only INT8 families: module-name suffixes and the one (N, K) each must have.
LINEAR_FAMILIES = {
    "gdn_in_proj_qkvz": ((".linear_attn.in_proj_qkvz",), (16384, 2560)),
    "qsa_qkv": ((".self_attn.qkv_proj",), (13312, 2560)),
    "attn_out": ((".linear_attn.out_proj", ".self_attn.o_proj"), (2560, 6144)),
}
LM_HEAD_SHAPE = (248320, 2560)
INT8_FAMILIES = ("lm_head", *LINEAR_FAMILIES)
TRITON_MAX_M = 128
# Dequantized BF16 scratch per GEMM call on the large-M path (elements): 64 Mi = 128 MiB.
DEQUANT_CHUNK_ELEMENTS = 64 << 20
# Phase-1 best Triton W8A16 tiles (BLOCK_N, BLOCK_K, num_warps, num_stages) per (N, K), for M <= 16
# and for larger M (kernels-phase1-bench.json, rows triton_w8a16_int8).
TRITON_TILES = {
    (16384, 2560): ((32, 128, 4, 4), (64, 128, 4, 3)),
    (13312, 2560): ((32, 128, 4, 4), (64, 128, 4, 3)),
    (2560, 6144): ((16, 256, 4, 3), (64, 128, 4, 3)),
    (248320, 2560): ((32, 256, 4, 3), (64, 128, 4, 3)),
}
DEFAULT_TILES = ((32, 256, 4, 3), (64, 128, 4, 3))
# Load-time self-check bounds (relative Frobenius error against FP32 on the layer's own weights):
# the kernel against its own dequantized weights (phase 1: BF16 rounding, ~0.2%), and the INT8 layer
# against the BF16 weights (phase 1: 1.0-2.0% on these families).
KERNEL_ERROR_LIMIT = 0.01
INT8_ERROR_LIMIT = 0.05
SKINNY_ERROR_LIMIT = 0.01
# Mid-M (phase 5). Buckets: the served decode CUDA-graph sizes above 4; an M in (b_prev, b] takes bucket b's tile.
MID_M_BUCKETS = (8, 16, 24, 32)
MID_INT8_BUCKETS = (24, 32)  # at M <= 16 the phase-1 INT8 tiles already read at the ceiling
MID_MIN_M = 5
# Small BF16 projections (N, K) that the mid-M path serves (the skinny plan below M = 5 stays in front).
MID_BF16_SHAPES = (
    (336, 10240),   # HC down + inject
    (320, 10240),   # HC final mixer down
    (10240, 320),   # HC up (K = 320: no skinny plan; cuBLAS at every M today)
    (512, 2560),    # router
    (96, 2560),     # GDN in_proj_ba
    (640, 2560),    # QSA indexer q/k
    (1280, 2560),   # shared expert gate_up
    (2560, 640),    # shared expert down
    (2560, 2560),   # MTP fc_embedding / fc_hidden, PLE value
)
# Candidate tiles (BLOCK_N, BLOCK_K, num_warps, num_stages); BLOCK_M is 16 for buckets <= 16, else 32.
MID_CANDIDATES = ((16, 256, 4, 3), (32, 128, 4, 4), (32, 256, 4, 3), (64, 128, 4, 3))
MID_MIN_GAIN = 0.03
MID_ERROR_FACTOR = 1.25
MID_ERROR_SLACK = 5e-4
MID_BENCH_BYTES = 256 << 20
MID_BENCH_REPLAYS = 10

_STATE = {"config": None, "methods": None, "checked": set(), "warm_layers": {}, "quantized": {}, "enabled": 0,
          "mid": {"bf16": {}, "int8": {}}, "mid_done": set(), "mid_warm": {}, "ones": {}}


def load_config():
    if _STATE["config"] is None:
        raw = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
        families = list(raw.get("int8_weights") or [])
        unknown = sorted(set(families) - set(INT8_FAMILIES))
        if unknown:
            raise RuntimeError(f"ARC dense kernels: unknown INT8 families {unknown}")
        mid = raw.get("mid_m") or False
        if isinstance(mid, dict):
            # A fixed plan: {"bf16"|"int8": {"N,K": {"M": [BLOCK_N, BLOCK_K, warps, stages] | null}}}.
            for kind in ("bf16", "int8"):
                for key, buckets in (mid.get(kind) or {}).items():
                    n, k = (int(v) for v in key.split(","))
                    _STATE["mid"][kind][(n, k)] = {int(m): (None if cfg is None else _mid_tile(int(m), cfg))
                                                   for m, cfg in buckets.items()}
        elif mid is not True:
            mid = False
        _STATE["config"] = {"skinny": bool(raw.get("skinny")), "int8_weights": families, "mid_m": mid}
    return _STATE["config"]


def _mid_tile(bucket, cfg):
    """(BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) for a mid-M bucket and a candidate tile."""
    return (16 if bucket <= 16 else 32, *(int(v) for v in cfg))


def mid_bucket(m, buckets=MID_M_BUCKETS):
    return next((b for b in buckets if m <= b), None) if m >= MID_MIN_M else None


def mid_tile(kind, n, k, m):
    """The mid-M tile for an (M, N, K) GEMM, or None for the pinned path."""
    plan = _STATE["mid"][kind].get((n, k))
    bucket = mid_bucket(m) if plan else None
    return None if bucket is None else plan.get(bucket)


def log(kind, payload):
    print(f"ARC_DENSE_{kind} " + json.dumps(payload, sort_keys=True, default=str), flush=True)


# ----------------------------------------------------------------------------- Triton W8A16 kernels
@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, K_PER_SPLIT,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PARTIAL: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_start = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_ptrs = x_ptr + offs_m[:, None] * K + (k_start + offs_k)[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + (k_start + offs_k)[None, :]
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N
    for k in range(0, K_PER_SPLIT, BLOCK_K):
        k_mask = (k_start + k + offs_k)[None, :] < K
        a = tl.load(x_ptrs, mask=m_mask & k_mask, other=0.0)
        b = tl.load(w_ptrs, mask=n_mask & k_mask, other=0, eviction_policy="evict_first")
        acc += tl.dot(a, tl.trans(b.to(tl.bfloat16)))
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * s[None, :]
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if PARTIAL:
        tl.store(out_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=o_mask)
    else:
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16), mask=o_mask)


@triton.jit
def _splitk_reduce_kernel(p_ptr, out_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, out_ptr, NK, K, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NK
    w = tl.load(w_ptr + offs, mask=mask, other=0).to(tl.float32)
    s = tl.load(s_ptr + offs // K, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (w * s).to(tl.bfloat16), mask=mask)


_SMS = {}


def _num_sms(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SMS:
        _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SMS[index]


def tile_config(n, k, m):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for an (M, N, K) W8A16 GEMM."""
    small, large = TRITON_TILES.get((n, k), DEFAULT_TILES)
    bm = 16 if m <= 16 else 64
    return (bm, *(small if m <= 16 else large))


def split_k(tiles, k, bk, num_sms):
    """Phase-1 split-K rule: split K until the grid covers two waves, keeping >= 2 K tiles per split."""
    split = 1
    while split < 16 and tiles * split < 2 * num_sms and k // (split * 2) >= 2 * bk:
        split *= 2
    k_per = triton.cdiv(triton.cdiv(k, split), bk) * bk
    return triton.cdiv(k, k_per), k_per


def w8a16_gemm(x, w, s, tile=None):
    """y = x @ (w * s[:, None]).T with x [M, K] BF16, w [N, K] INT8 (or BF16 with s = 1), s [N] FP32; FP32
    accumulation, one BF16 rounding. ``tile`` (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) overrides the default."""
    m, k = x.shape
    n = w.shape[0]
    bm, bn, bk, warps, stages = tile or tile_config(n, k, m)
    n_tiles, m_tiles = triton.cdiv(n, bn), triton.cdiv(m, bm)
    split, k_per = split_k(n_tiles * m_tiles, k, bk, _num_sms(x.device))
    if split == 1:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _w8a16_kernel[(n_tiles, 1, m_tiles)](x, w, s, out, m, n, k, k_per, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                              PARTIAL=False, num_warps=warps, num_stages=stages)
        return out
    part = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    _w8a16_kernel[(n_tiles, split, m_tiles)](x, w, s, part, m, n, k, k_per, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                              PARTIAL=True, num_warps=warps, num_stages=stages)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _splitk_reduce_kernel[(triton.cdiv(m * n, 1024),)](part, out, m * n, SPLIT=split, BLOCK=1024)
    return out


def dequantize(w, s):
    n, k = w.shape
    out = torch.empty((n, k), dtype=torch.bfloat16, device=w.device)
    _dequant_kernel[(triton.cdiv(n * k, 4096),)](w, s, out, n * k, k, BLOCK=4096)
    return out


def dequant_linear(x, w, s):
    """Large-M path (prefill): dequantize into a bounded BF16 scratch and let cuBLAS run the GEMM."""
    n, k = w.shape
    rows = max(1, min(n, DEQUANT_CHUNK_ELEMENTS // k))
    if rows >= n:
        return F.linear(x, dequantize(w, s))
    return torch.cat([F.linear(x, dequantize(w[i:i + rows], s[i:i + rows])) for i in range(0, n, rows)], dim=-1)


def _arc_w8a16_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if flat.dtype != torch.bfloat16:
        raise RuntimeError(f"ARC W8A16 expects BF16 activations, got {flat.dtype}")
    if not flat.is_contiguous():
        flat = flat.contiguous()
    m = flat.shape[0]
    if m == 0:
        return x.new_empty((*shape[:-1], weight.shape[0]))
    if m <= TRITON_MAX_M:
        y = w8a16_gemm(flat, weight, scale, mid_tile("int8", weight.shape[0], weight.shape[1], m))
    else:
        y = dequant_linear(flat, weight, scale)
    return y.reshape(*shape[:-1], weight.shape[0])


def _arc_w8a16_linear_fake(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _ones(n, device):
    """A unit scale for BF16 weights on the W8A16 kernel (acc * 1.0 is exact); created at load, before capture."""
    key = (int(n), str(device))
    if key not in _STATE["ones"]:
        _STATE["ones"][key] = torch.ones((int(n),), dtype=torch.float32, device=device)
    return _STATE["ones"][key]


def _pinned_bf16(x, weight):
    """The pinned path of a planned BF16 layer: the skinny kernel where the plan lists M, else cuBLAS."""
    from vllm.models.qwen3_8_flash_next.nvidia.low_latency_gemm import _qwen38next_low_latency_gemm

    return _qwen38next_low_latency_gemm(x, weight)


def _arc_bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Small BF16 projection: the tuned mid-M Triton tile for its (N, K, M bucket), else the pinned path."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    m = x.numel() // x.shape[-1] if x.shape[-1] else 0
    tile = mid_tile("bf16", n, k, m) if m else None
    if tile is None or x.dtype != torch.bfloat16:
        return _pinned_bf16(x, weight)
    flat = x.reshape(-1, x.shape[-1])
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return w8a16_gemm(flat, weight, _ones(n, weight.device), tile).reshape(*x.shape[:-1], n)


def _arc_bf16_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _register_op():
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "arc_w8a16_linear"):
        direct_register_custom_op(op_name="arc_w8a16_linear", op_func=_arc_w8a16_linear,
                                  fake_impl=_arc_w8a16_linear_fake)
    if not hasattr(torch.ops.vllm, "arc_bf16_linear"):
        direct_register_custom_op(op_name="arc_bf16_linear", op_func=_arc_bf16_linear,
                                  fake_impl=_arc_bf16_linear_fake)


# ----------------------------------------------------------------------------- quantization and checks
def quantize_rows(weight, chunk_rows=8192):
    """Per-output-channel symmetric INT8: scale = amax(|row|) / 127, q = round(w / scale) in [-127, 127]."""
    n, k = weight.shape
    q = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    s = torch.empty((n,), dtype=torch.float32, device=weight.device)
    for start in range(0, n, chunk_rows):
        w = weight[start:start + chunk_rows].float()
        scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
        q[start:start + chunk_rows] = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
        s[start:start + chunk_rows] = scale
    return q, s


def _rel(a, b):
    denominator = b.float().norm().item()
    return (a.float() - b.float()).norm().item() / denominator if denominator else 0.0


def check_int8(family, prefix, weight, q, s, rows=8192):
    """Errors of the INT8 layer on its real weights (first `rows` output channels), Gaussian activations."""
    key = ("int8", family)
    if key in _STATE["checked"]:
        return
    _STATE["checked"].add(key)
    w, qs, ss = weight[:rows], q[:rows].contiguous(), s[:rows].contiguous()
    generator = torch.Generator(device=weight.device).manual_seed(0)
    result = {"family": family, "prefix": prefix, "shape": list(weight.shape), "rows_checked": int(w.shape[0])}
    dequant_ref = (qs.float() * ss[:, None])
    for m in (1, 4, 16, 64, 192):
        x = torch.randn((m, w.shape[1]), generator=generator, device=weight.device, dtype=torch.float32)
        x = x.to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        ref_q = x.float() @ dequant_ref.t()
        y = _arc_w8a16_linear(x, qs, ss)
        row = {"path": "triton" if m <= TRITON_MAX_M else "dequant+cublas",
               "kernel_vs_own_int8_weights": _rel(y, ref_q), "int8_vs_bf16_weights": _rel(y, ref),
               "bf16_cublas_vs_fp32": _rel(F.linear(x, w), ref)}
        result[f"M{m}"] = row
        if row["kernel_vs_own_int8_weights"] > KERNEL_ERROR_LIMIT or row["int8_vs_bf16_weights"] > INT8_ERROR_LIMIT:
            log("CHECK", {**result, "failed": True})
            raise RuntimeError(f"ARC W8A16 self-check failed for {family}: {row}")
    log("CHECK", result)


def check_skinny(weight, plan, rows=8192):
    """Skinny GEMM against cuBLAS and FP32 on the layer's real weights (first `rows` output channels)."""
    shape = tuple(weight.shape)
    key = ("skinny", shape)
    if key in _STATE["checked"]:
        return
    _STATE["checked"].add(key)
    weight = weight[:rows]
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import shape_dynamic_skinny_gemm

    generator = torch.Generator(device=weight.device).manual_seed(0)
    result = {"shape": list(shape), "rows_checked": int(weight.shape[0])}
    for m in (1, 4, 16):
        if m not in plan:
            continue
        x = torch.randn((m, shape[1]), generator=generator, device=weight.device).to(torch.bfloat16)
        ref = x.float() @ weight.float().t()
        cublas = F.linear(x, weight)
        skinny = shape_dynamic_skinny_gemm(x, weight, plan[m])
        row = {"skinny_vs_fp32": _rel(skinny, ref), "cublas_vs_fp32": _rel(cublas, ref),
               "max_abs_skinny_minus_cublas": (skinny.float() - cublas.float()).abs().max().item(),
               "bitwise_equal_fraction": (skinny == cublas).float().mean().item()}
        result[f"M{m}"] = row
        if row["skinny_vs_fp32"] > SKINNY_ERROR_LIMIT:
            log("CHECK", {**result, "failed": True})
            raise RuntimeError(f"ARC skinny self-check failed for {shape}: {row}")
    log("CHECK", result)


# ----------------------------------------------------------------------------- mid-M tuning at load (phase 5)
def time_graph(fn, calls, replays=MID_BENCH_REPLAYS):
    """Mean microseconds per call of ``fn(i)`` for i in range(calls), captured once in a CUDA graph and replayed
    (graph-node spacing included, as in the model's own graphs)."""
    fn(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(i)
    graph.replay()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) * 1000.0 / (replays * calls)
    del graph
    return elapsed


def tune_mid(kind, weight, scale, prefix):
    """``_tune_mid`` that never stops the server: on any failure the shape keeps its pinned path everywhere.

    Runs from the kernel warmup (after vLLM has sized the KV cache, before graph capture), never at load: the
    stack smoke (kernel-stack-phase5-v1) measured 0.41 GiB more "model loading" memory with tuning at load, which
    cost 20 KV blocks and the sixth 64K window of the MTP-3 arm. Allocated memory before and after is logged."""
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    try:
        _tune_mid(kind, weight, scale, prefix)
    except Exception as exc:  # noqa: BLE001 - the pinned path is always valid
        n, k = int(weight.shape[0]), int(weight.shape[1])
        _STATE["mid"][kind][(n, k)] = {}
        gc.collect()
        torch.cuda.empty_cache()
        log("MID", {"kind": kind, "prefix": prefix, "shape": [n, k], "failed": f"{type(exc).__name__}: {exc}"[:400]})
    if torch.cuda.is_available():
        log("MID_MEMORY", {"kind": kind, "shape": [int(v) for v in weight.shape],
                           "allocated_delta_bytes": torch.cuda.memory_allocated() - before})


def _tune_mid(kind, weight, scale, prefix):
    """Choose the mid-M tile of ``weight``'s shape per bucket (or check a fixed plan's tiles); once per shape.

    ``kind`` "bf16": ``weight`` BF16 [N, K], pinned path = skinny plan or cuBLAS. "int8": INT8 weight and FP32
    per-channel ``scale``, pinned path = the phase-2 tiles. The error of every path is taken against FP32 on the
    real weights (first 8,192 output channels) with Gaussian activations."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if (kind, n, k) in _STATE["mid_done"]:
        return
    _STATE["mid_done"].add((kind, n, k))
    fixed = isinstance(load_config()["mid_m"], dict)
    if fixed and (n, k) not in _STATE["mid"][kind]:
        return
    buckets = MID_M_BUCKETS if kind == "bf16" else MID_INT8_BUCKETS
    device = weight.device
    s = _ones(n, device) if kind == "bf16" else scale
    rows = min(n, 8192)
    ref_w = weight[:rows].float() if kind == "bf16" else weight[:rows].float() * scale[:rows, None]
    copies = 1 if fixed else max(1, min(64, -(-MID_BENCH_BYTES // (weight.numel() * weight.element_size()))))
    weights = [weight] + [weight.clone() for _ in range(copies - 1)]
    generator = torch.Generator(device=device).manual_seed(0)
    result = {"kind": kind, "prefix": prefix, "shape": [n, k], "copies": copies, "fixed_plan": fixed, "buckets": {}}

    def pinned(x, w):
        return _pinned_bf16(x, w) if kind == "bf16" else w8a16_gemm(x, w, scale)

    plan = {}
    for bucket in buckets:
        x = torch.randn((bucket, k), generator=generator, device=device).to(torch.bfloat16)
        ref = x.float() @ ref_w.t()
        entry = {"pinned_err": _rel(pinned(x, weight)[:, :rows], ref)}
        limit = max(MID_ERROR_FACTOR * entry["pinned_err"], entry["pinned_err"] + MID_ERROR_SLACK)
        if fixed:
            planned = _STATE["mid"][kind][(n, k)].get(bucket)
            candidates = [] if planned is None else [tuple(planned[1:])]
        else:
            entry["pinned_us"] = time_graph(lambda i, x=x: pinned(x, weights[i]), copies)
            candidates = list(MID_CANDIDATES)
        best, rows_out = None, []
        for cfg in candidates:
            tile = _mid_tile(bucket, cfg)
            row = {"tile": list(cfg)}
            try:
                row["err"] = _rel(w8a16_gemm(x, weight, s, tile)[:, :rows], ref)
                if not fixed:
                    row["us"] = time_graph(lambda i, x=x, tile=tile: w8a16_gemm(x, weights[i], s, tile), copies)
            except Exception as exc:  # noqa: BLE001 - a tile that does not fit (shared memory) is skipped
                row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                rows_out.append(row)
                continue
            rows_out.append(row)
            if row["err"] <= limit and (fixed or best is None or row["us"] < best[1]):
                best = (cfg, row.get("us"))
        chosen = None
        if best is not None and (fixed or best[1] <= entry["pinned_us"] * (1.0 - MID_MIN_GAIN)):
            chosen = list(best[0])
        entry.update(candidates=rows_out, chosen=chosen)
        plan[bucket] = None if chosen is None else _mid_tile(bucket, chosen)
        result["buckets"][str(bucket)] = entry
    _STATE["mid"][kind][(n, k)] = plan
    weights.clear()  # the copies (the first entry is the layer's own weight)
    gc.collect()
    torch.cuda.empty_cache()
    log("MID", result)


# ----------------------------------------------------------------------------- quant methods
def _methods():
    if _STATE["methods"] is None:
        _STATE["methods"] = _build_methods()
    return _STATE["methods"]


def _build_methods():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from vllm.models.qwen3_8_flash_next.nvidia.low_latency_gemm import (
        QWEN38NEXT_GEMM_PLANS,
        Qwen38NextLowLatencyEmbeddingMethod,
        Qwen38NextLowLatencyLinearMethod,
    )

    class _Int8Mixin:
        family = ""

        def process_weights_after_loading(self, layer):
            weight = layer.weight.data
            if weight.dtype != torch.bfloat16 or weight.dim() != 2:
                raise RuntimeError(f"ARC W8A16 expects a 2-D BF16 weight, got {weight.dtype} {tuple(weight.shape)}")
            prefix = getattr(layer, "prefix", "") or type(layer).__name__
            with torch.no_grad():
                q, s = quantize_rows(weight)
                check_int8(self.family, prefix, weight, q, s)
            layer.weight = torch.nn.Parameter(q, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
            del weight
            stats = _STATE["quantized"].setdefault(self.family, {"layers": 0, "bf16_bytes_freed": 0})
            stats["layers"] += 1
            stats["bf16_bytes_freed"] += q.numel() * 2 - q.numel() - s.numel() * 4
            _STATE["warm_layers"].setdefault(tuple(q.shape), layer)
            log("INT8", {"prefix": prefix, "family": self.family, "shape": list(q.shape), "totals": stats})

        def apply(self, layer, x, bias=None):
            y = torch.ops.vllm.arc_w8a16_linear(x, layer.weight, layer.weight_scale)
            return y if bias is None else y + bias

    class Int8LinearMethod(_Int8Mixin, UnquantizedLinearMethod):
        def __init__(self, family):
            super().__init__()
            self.family = family

    class Int8HeadMethod(_Int8Mixin, UnquantizedEmbeddingMethod):
        def __init__(self, family):
            super().__init__()
            self.family = family

        def embedding(self, layer, input_):
            raise RuntimeError("ARC W8A16 lm_head cannot serve as an input embedding (tied weights)")

    class _SkinnyCheck:
        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)
            plan = QWEN38NEXT_GEMM_PLANS.get(tuple(layer.weight.shape))
            if plan:
                with torch.no_grad():
                    check_skinny(layer.weight.data, plan)

    class SkinnyLinearMethod(_SkinnyCheck, Qwen38NextLowLatencyLinearMethod):
        pass

    class SkinnyHeadMethod(_SkinnyCheck, Qwen38NextLowLatencyEmbeddingMethod):
        pass

    class MidLinearMethod(_SkinnyCheck, UnquantizedLinearMethod):
        """Phase 5: a small BF16 projection on the mid-M Triton tile where tuned, else its pinned path (the skinny
        kernel where planned, cuBLAS otherwise)."""

        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)
            weight = layer.weight.data
            _ones(weight.shape[0], weight.device)
            _STATE["mid_warm"].setdefault(tuple(int(v) for v in weight.shape), layer)  # tuned in the warmup

        def apply(self, layer, x, bias=None):
            y = torch.ops.vllm.arc_bf16_linear(x, layer.weight)
            return y if bias is None else y + bias

    return {"plans": QWEN38NEXT_GEMM_PLANS, "linear": UnquantizedLinearMethod, "head": UnquantizedEmbeddingMethod,
            "int8_linear": Int8LinearMethod, "int8_head": Int8HeadMethod,
            "skinny_linear": SkinnyLinearMethod, "skinny_head": SkinnyHeadMethod, "mid_linear": MidLinearMethod}


def skinny_plan(n, k):
    """TP=1 plan: the skinny kernel's own default config for every M = 1..16."""
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import ShapeDynamicSkinnyGemm

    plan = {}
    for m in range(1, SKINNY_MAX_M + 1):
        config = ShapeDynamicSkinnyGemm._config(m, n, k)
        if k % (config.block_size * config.vector_width) or n % config.outputs_per_block:
            return {}
        plan[m] = config
    return plan


def family_of(child, is_head, shape):
    if is_head:
        return "lm_head" if shape == LM_HEAD_SHAPE else None
    prefix = getattr(child, "prefix", "") or ""
    for name, (suffixes, want) in LINEAR_FAMILIES.items():
        if prefix.endswith(suffixes) and shape == want:
            return name
    return None


# ----------------------------------------------------------------------------- warmup before graph capture
class _Warmup:
    """CuTeDSL-warmup provider (run by vLLM's kernel warmup before CUDA graph capture): compiles every
    Triton specialization the W8A16 op can hit at M = 1..TRITON_MAX_M, on each quantized shape."""

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        units = [CuTeDSLCompileUnit(name="ARC W8A16 Triton GEMM", key=("arc-w8a16", shape),
                                    compile=partial(_warm_shape, shape))
                 for shape in sorted(_STATE["warm_layers"])]
        units += [CuTeDSLCompileUnit(name="ARC mid-M BF16 Triton GEMM", key=("arc-mid-bf16", shape),
                                     compile=partial(_warm_mid_shape, shape))
                  for shape in sorted(_STATE["mid_warm"])]
        return tuple(units)


def _warm_shape(shape):
    layer = _STATE["warm_layers"][shape]
    n, k = shape
    if load_config()["mid_m"]:
        # Phase 5: tune (or check the fixed plan) here, after the KV cache is sized; before the specializations below.
        with torch.no_grad():
            tune_mid("int8", layer.weight.data, layer.weight_scale.data,
                     getattr(layer, "prefix", "") or type(layer).__name__)
    x = torch.zeros((TRITON_MAX_M + 1, k), dtype=torch.bfloat16, device=layer.weight.device)
    for m in range(1, TRITON_MAX_M + 2):
        _arc_w8a16_linear(x[:m], layer.weight, layer.weight_scale)
    torch.cuda.synchronize()


def _warm_mid_shape(shape):
    """Tune ``shape`` (after the KV cache is sized), then compile every Triton specialization its tuned mid-M
    buckets can hit (M = 5..32)."""
    layer = _STATE["mid_warm"][shape]
    with torch.no_grad():
        tune_mid("bf16", layer.weight.data, None, getattr(layer, "prefix", "") or type(layer).__name__)
    x = torch.zeros((MID_M_BUCKETS[-1], shape[1]), dtype=torch.bfloat16, device=layer.weight.device)
    for m in range(MID_MIN_M, MID_M_BUCKETS[-1] + 1):
        if mid_tile("bf16", shape[0], shape[1], m) is not None:
            _arc_bf16_linear(x[:m], layer.weight)
    torch.cuda.synchronize()


_WARMUP = _Warmup()


# ----------------------------------------------------------------------------- entry point
def enable(module, dtype):
    """Called at the top of the pinned ``enable_qwen38next_low_latency_gemm``. Returns True when the
    overlay handled ``module`` (the pinned SM103-only body is then skipped)."""
    config = load_config()
    if not (config["skinny"] or config["int8_weights"]):
        return False
    if dtype != torch.bfloat16:
        raise RuntimeError("ARC dense kernels require a BF16 model dtype")
    capability = torch.cuda.get_device_capability()
    if capability not in ((12, 0), (10, 3)):
        raise RuntimeError(f"ARC dense kernels were measured on SM120 only; device is {capability}")
    methods = _methods()
    plans = methods["plans"]
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import shape_dynamic_skinny_gemm
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    if config["skinny"]:
        if not shape_dynamic_skinny_gemm.is_available():
            raise RuntimeError("ARC skinny GEMM requested but CuTe DSL is unavailable")
        for n, k in SKINNY_SHAPES:
            plan = skinny_plan(n, k)
            if plan:
                plans[(n, k)] = plan
    mid = bool(config["mid_m"])
    if config["int8_weights"] or mid:
        _register_op()
        from vllm.model_executor.warmup.cutedsl_warmup import register_cutedsl_warmup_provider

        register_cutedsl_warmup_provider(_WARMUP)
    warmup, counts = set(), {}
    for child in module.modules():
        is_linear = isinstance(child, LinearBase) and type(child.quant_method) is methods["linear"]
        is_head = isinstance(child, ParallelLMHead) and type(child.quant_method) is methods["head"]
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        shape = (int(weight.shape[0]), int(weight.shape[1]))
        family = family_of(child, is_head, shape)
        if family is not None and family in config["int8_weights"]:
            child.quant_method = methods["int8_head" if is_head else "int8_linear"](family)
            counts[f"int8:{family}"] = counts.get(f"int8:{family}", 0) + 1
            continue
        mid_layer = mid and is_linear and shape in MID_BF16_SHAPES
        if config["skinny"] and shape in plans and shape in SKINNY_SHAPES:
            # Phase 5: a small BF16 shape keeps its skinny plan (the pinned path inside the mid-M op).
            child.quant_method = methods["mid_linear" if mid_layer else "skinny_head" if is_head else "skinny_linear"]()
            warmup.update(plans[shape].values())
            label = f"{'mid' if mid_layer else 'skinny'}:{shape[0]}x{shape[1]}"
            counts[label] = counts.get(label, 0) + 1
        elif mid_layer:
            # HC up (K = 320) has no skinny plan: its pinned path is cuBLAS at every M.
            child.quant_method = methods["mid_linear"]()
            counts[f"mid:{shape[0]}x{shape[1]}"] = counts.get(f"mid:{shape[0]}x{shape[1]}", 0) + 1
    if warmup:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup)
    _STATE["enabled"] += 1
    log("KERNELS", {"module": type(module).__name__, "config": config, "capability": list(capability),
                    "layers": counts, "skinny_warmup_configs": len(warmup), "triton_max_m": TRITON_MAX_M,
                    "skinny_max_m": SKINNY_MAX_M})
    return True
