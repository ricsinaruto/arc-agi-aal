"""Phase-6 extensions of the ARC dense kernels (kernel work, phase 6; docs/validation-20260927/kernels-phase6.md).

Copied into ``vllm/models/qwen3_8_flash_next/nvidia/arc_dense_fused.py`` by ``arc_infra.dense_kernels_patch`` only
when a study's ``server.dense_kernels`` sets ``fused_splitk`` or ``mid_int8_buckets``; the patched
``enable_qwen38next_low_latency_gemm`` then calls ``install()`` before ``arc_dense_kernels.enable``. Arms without
these keys get neither this file nor the extra line (their patched sources stay byte-identical).

- ``fused_splitk``: the W8A16 / mid-M Triton GEMM (``arc_dense_kernels.w8a16_gemm``) runs split-K as two
  launches, the partial GEMM and ``_splitk_reduce_kernel``. At 14 decode streams the hero profile shows 242 reduce
  launches per step (0.40 ms of kernel time, plus a graph-node gap each). Here the last CTA of each output tile to
  finish reduces the partials itself, in the same fixed order (s = 0 .. SPLIT-1, FP32, one BF16 rounding), so the
  output is bitwise identical to the two-launch path; one launch fewer per split-K call. Completion is counted
  with one int32 ticket per output tile (atomic acq_rel at GPU scope after a CTA barrier; partials written and
  read through L2), and the last CTA resets its ticket to 0, so a counter set is always zero between calls and
  across CUDA-graph replays. Ticket sets live in one arena allocated before graph capture; each (N, K, stream)
  gets its own set, so same-shape GEMMs on two streams (vLLM's shared-expert aux stream) never share tickets.
  Every (shape, tile, split) is checked bitwise against the two-launch path on its first use outside capture
  (the kernel warmup compiles every specialization before capture); a mismatch disables fusion for it.
- ``mid_int8_buckets``: the INT8 families' mid-M buckets (phase 5 tunes only 24 and 32; the hero serves 14
  streams at graph size 16 and 21 at 24), so a tuned or fixed plan can carry INT8 tiles at 8 and 16 too.
- ``mid_extra_candidates``: extra mid-M candidate tiles for the load-time tuner (phase 5 has four).
"""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl

CONFIG_PATH = Path(__file__).with_name("arc_dense_kernels.json")
# int32 tile tickets, allocated once before graph capture. A split-K call has fewer than 2 x SMs output tiles (the
# phase-1 split rule splits only while tiles x split < 2 x SMs), so one ticket set per (N, K, stream) needs
# 2 x SMs entries: 376 on the RTX PRO 6000; 65,536 entries hold 174 sets.
ARENA_TICKETS = 1 << 16
_STATE = {"installed": False, "base": None, "original": None, "arena": None, "sets": {}, "next": 0,
          "checked": {}, "stats": {"fused_calls": 0, "fallback_calls": 0}, "logged": set()}


def log(kind, payload):
    print(f"ARC_DENSE_{kind} " + json.dumps(payload, sort_keys=True, default=str), flush=True)


@triton.jit
def _w8a16_fused_kernel(x_ptr, w_ptr, s_ptr, part_ptr, out_ptr, ticket_ptr, M, N, K, K_PER_SPLIT,
                        SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # The partial GEMM: the body of arc_dense_kernels._w8a16_kernel (PARTIAL=True), unchanged.
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
    tile_offs = offs_m[:, None] * N + offs_n[None, :]
    tl.store(part_ptr + pid_k * M * N + tile_offs, acc, mask=o_mask, cache_modifier=".cg")
    # Completion ticket: every thread's partial store is issued before one release/acquire RMW at GPU scope.
    tl.debug_barrier()
    ticket = ticket_ptr + pid_m * tl.num_programs(0) + pid_n
    if tl.atomic_add(ticket, 1, sem="acq_rel", scope="gpu") == SPLIT - 1:
        # The last CTA of this tile: _splitk_reduce_kernel's sum, in its order (s = 0 .. SPLIT-1, FP32 from 0).
        total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for split in tl.static_range(SPLIT):
            total += tl.load(part_ptr + split * M * N + tile_offs, mask=o_mask, other=0.0, cache_modifier=".cg")
        tl.store(out_ptr + tile_offs, total.to(tl.bfloat16), mask=o_mask)
        tl.atomic_xchg(ticket, 0, sem="relaxed", scope="gpu")


def load_config():
    raw = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    return {"fused_splitk": bool(raw.get("fused_splitk")),
            "mid_int8_buckets": [int(b) for b in raw.get("mid_int8_buckets") or []],
            "mid_extra_candidates": [tuple(int(v) for v in c) for c in raw.get("mid_extra_candidates") or []]}


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _stream_key(device):
    return int(torch.cuda.current_stream(device).cuda_stream) if device.type == "cuda" else 0


def tickets(n, k, device, size):
    """The ticket set of (N, K) on the current stream: ``size`` int32 entries of the arena (all zero between calls),
    or None when the arena is missing during capture or full (the caller then runs the two-launch path)."""
    arena = _STATE["arena"]
    if arena is None or arena.device != device:
        if _capturing():
            return None
        _STATE["arena"] = arena = torch.zeros((ARENA_TICKETS,), dtype=torch.int32, device=device)
        _STATE["sets"].clear()
        _STATE["next"] = 0
    per_set = 2 * _STATE["base"]._num_sms(device)
    if size > per_set:
        return None
    key = (int(n), int(k), _stream_key(device))
    offset = _STATE["sets"].get(key)
    if offset is None:
        if _STATE["next"] + per_set > ARENA_TICKETS:
            if "arena_full" not in _STATE["logged"]:
                _STATE["logged"].add("arena_full")
                log("FUSED", {"arena_full": True, "sets": len(_STATE["sets"])})
            return None
        offset = _STATE["next"]
        _STATE["sets"][key] = offset
        _STATE["next"] += per_set
    return arena[offset:offset + size]


def _launch(x, w, s, counters, tile, split, k_per):
    m, k = x.shape
    n = w.shape[0]
    bm, bn, bk, warps, stages = tile
    part = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _w8a16_fused_kernel[(triton.cdiv(n, bn), split, triton.cdiv(m, bm))](
        x, w, s, part, out, counters, m, n, k, k_per, SPLIT=split, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=warps, num_stages=stages)
    return out


def _check(key, x, w, s, tile, split, k_per, counters):
    """Bitwise check of one (shape, tile, split) against the two-launch path, on a random probe of the call's
    shape (the warmup calls with zeros), outside capture; the tickets must be back to zero afterwards."""
    generator = torch.Generator(device=x.device).manual_seed(1234)
    probe = torch.randn(x.shape, generator=generator, device=x.device, dtype=torch.float32).to(x.dtype)
    fused = _launch(probe, w, s, counters, tile, split, k_per)
    reference = _STATE["original"](probe, w, s, tile)
    equal = bool(torch.equal(fused, reference)) and not bool(counters.any())
    _STATE["checked"][key] = equal
    log("FUSED_CHECK", {"shape": [int(w.shape[0]), int(w.shape[1])], "m": int(x.shape[0]), "tile": list(tile),
                        "split": int(split), "weight": str(w.dtype), "bitwise_equal": equal})
    return equal


def fused_w8a16_gemm(x, w, s, tile=None):
    """``arc_dense_kernels.w8a16_gemm`` with the split-K reduction in the GEMM's last CTA per tile (bitwise equal)."""
    base, original = _STATE["base"], _STATE["original"]
    m, k = x.shape
    n = w.shape[0]
    chosen = tuple(tile or base.tile_config(n, k, m))
    bm, bn, bk = chosen[:3]
    n_tiles, m_tiles = triton.cdiv(n, bn), triton.cdiv(m, bm)
    split, k_per = base.split_k(n_tiles * m_tiles, k, bk, base._num_sms(x.device))
    if split == 1:
        return original(x, w, s, tile)
    key = (int(n), int(k), *chosen, int(split), str(w.dtype))
    verdict = _STATE["checked"].get(key)
    counters = None if verdict is False else tickets(n, k, x.device, n_tiles * m_tiles)
    if counters is not None and verdict is None:
        verdict = None if _capturing() else _check(key, x, w, s, chosen, split, k_per, counters)
    if counters is None or not verdict:
        _STATE["stats"]["fallback_calls"] += 1
        return original(x, w, s, tile)
    _STATE["stats"]["fused_calls"] += 1
    return _launch(x, w, s, counters, chosen, split, k_per)


def install():
    """Apply the configured extensions to ``arc_dense_kernels`` (idempotent; called by the patched pinned
    ``enable_qwen38next_low_latency_gemm`` before ``arc_dense_kernels.enable``)."""
    if _STATE["installed"]:
        return
    from vllm.models.qwen3_8_flash_next.nvidia import arc_dense_kernels as base

    config = load_config()
    _STATE["base"] = base
    if config["fused_splitk"]:
        _STATE["original"] = base.w8a16_gemm
        base.w8a16_gemm = fused_w8a16_gemm  # module global: the INT8 op, the mid-M BF16 op and the tuner call it
    if config["mid_int8_buckets"]:
        base.MID_INT8_BUCKETS = tuple(sorted(config["mid_int8_buckets"]))
    if config["mid_extra_candidates"]:
        base.MID_CANDIDATES = tuple(base.MID_CANDIDATES) + tuple(
            c for c in config["mid_extra_candidates"] if c not in base.MID_CANDIDATES)
    _STATE["installed"] = True
    log("PHASE6", {"config": config, "mid_int8_buckets": list(base.MID_INT8_BUCKETS),
                   "mid_candidates": [list(c) for c in base.MID_CANDIDATES]})
