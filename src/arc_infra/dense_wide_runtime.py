"""Wide decode tiles for the ARC dense kernels (kernel profile, 30 September; docs/validation-20260930/kernel-profile.md).

Copied into ``vllm/models/qwen3_8_flash_next/nvidia/arc_dense_wide.py`` by ``arc_infra.dense_kernels_patch`` only when
a study's ``server.dense_kernels`` sets ``mid_wide``; the patched ``enable_qwen38next_low_latency_gemm`` then calls
``install()`` before ``arc_dense_kernels.enable``. Arms without the key get neither this file nor the extra line.

Why: the phase-5 mid-M plan stops at M = 32 (``MID_M_BUCKETS`` 8-32, ``MID_INT8_BUCKETS`` 24/32). The blocks3 campaign
serves 37-56 streams on CUDA graphs of 36-56 rows, where every small BF16 projection (HC down/up, in_proj_ba, indexer
q/k, the shared expert) falls back to cuBLAS (split-K plus a reduce launch per call) and the INT8 families to the
prefill tile. The kernel profile (40 streams, real contexts) puts the main-stream dense GEMMs at 5.8 ms per step for
4.8 GB of weights, 55% of the read ceiling, against 5.1 ms at 24 streams on the tuned tiles. The prune12 router
(448 x 2560 after pruning) is in no plan at any M: 9.5 us per call (cuBLAS + reduce) for 2.3 MB, 0.24 TB/s, 48 calls per
step.

What (same Triton kernel, ``arc_dense_kernels.w8a16_gemm``; FP32 accumulation, one BF16 rounding, error-checked):

- buckets 40 / 48 / 56 for both kinds (M in (32, 40] takes bucket 40, and so on), with 64-row tiles by default or a
  32-row tile over two M tiles (the second read of each weight tile comes from L2), chosen per shape and bucket;
- ``mid_wide_shapes``: extra BF16 shapes served by the mid-M path at every bucket (the pruned router, [448, 2560]);
- tuning in the kernel warmup (before graph capture, after the KV cache is sized), as ``mid_m: true`` does: every
  candidate is timed in a CUDA-graph replay over >= 256 MiB of weight copies against the pinned path (cuBLAS / the
  phase-2 INT8 tile) and checked against FP32 on the real weights; kept when >= 3% faster and within
  max(1.25 x, +0.05 points) of the pinned path's error. With a fixed ``mid_m`` plan, only the buckets and shapes the
  plan lacks are tuned; the plan's own tiles are untouched. ``mid_wide_plan`` fixes the wide tiles instead
  ({"bf16"|"int8": {"N,K": {"40": [BLOCK_M, BLOCK_N, BLOCK_K, warps, stages] | null}}}; only error-checked).
"""

import gc
import json
from pathlib import Path

import torch

CONFIG_PATH = Path(__file__).with_name("arc_dense_kernels.json")
WIDE_BUCKETS = (40, 48, 56)
# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages): 64-row tiles (one M tile up to 56 rows) and 32-row tiles (two M tiles,
# twice the CTAs for narrow N). Shared memory per stage is BLOCK_M x BLOCK_K x 2 + BLOCK_N x BLOCK_K bytes; a candidate
# that does not fit is skipped by the tuner.
WIDE_CANDIDATES = (
    (64, 16, 128, 4, 4), (64, 32, 128, 4, 4), (64, 64, 128, 4, 3), (64, 32, 64, 4, 4), (64, 64, 64, 4, 4),
    (64, 128, 64, 4, 3), (64, 16, 256, 4, 2), (64, 32, 128, 8, 3),
    (32, 16, 256, 4, 3), (32, 32, 128, 4, 4), (32, 32, 256, 4, 3), (32, 64, 128, 4, 3),
)
MIN_GAIN = 0.03
_STATE = {"installed": False, "base": None, "config": None, "fixed": {"bf16": {}, "int8": {}}, "results": []}


def log(kind, payload):
    print(f"ARC_DENSE_{kind} " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def load_config(raw=None):
    raw = (json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}) if raw is None else raw
    plan = raw.get("mid_wide_plan") or {}
    fixed = {"bf16": {}, "int8": {}}
    for kind in fixed:
        for key, buckets in (plan.get(kind) or {}).items():
            n, k = (int(v) for v in key.split(","))
            fixed[kind][(n, k)] = {int(m): (None if cfg is None else tuple(int(v) for v in cfg))
                                   for m, cfg in buckets.items()}
    return {"mid_wide": bool(raw.get("mid_wide")),
            "shapes": [tuple(int(v) for v in s) for s in raw.get("mid_wide_shapes") or []],
            "fixed": fixed,
            # Throughput arm (3 October): MTP-1 verify batches (2 x streams rows) need buckets past 56.
            "buckets": tuple(int(b) for b in raw.get("mid_wide_buckets") or WIDE_BUCKETS)}


def block_m(bucket):
    return 16 if bucket <= 16 else 32 if bucket <= 32 else 64


def wide_mid_tile(bucket, cfg):
    """``arc_dense_kernels._mid_tile`` extended: a 4-entry candidate takes the bucket's row tile (64 above M = 32);
    a 5-entry one carries its own BLOCK_M."""
    cfg = tuple(int(v) for v in cfg)
    return cfg if len(cfg) == 5 else (block_m(bucket), *cfg)


def wide_mid_bucket(m, buckets=None):
    base = _STATE["base"]
    buckets = base.MID_M_BUCKETS if buckets is None else buckets
    return next((b for b in buckets if m <= b), None) if m >= base.MID_MIN_M else None


def _tune_buckets(kind, weight, scale, prefix, buckets):
    """The phase-5 tuner loop for ``buckets`` of one shape (``arc_dense_kernels._tune_mid``'s non-fixed branch, with
    the wide candidates, or the fixed wide tiles when ``mid_wide_plan`` lists the shape)."""
    base = _STATE["base"]
    n, k = int(weight.shape[0]), int(weight.shape[1])
    fixed_plan = _STATE["fixed"][kind].get((n, k))
    device = weight.device
    s = base._ones(n, device) if kind == "bf16" else scale
    rows = min(n, 8192)
    ref_w = weight[:rows].float() if kind == "bf16" else weight[:rows].float() * scale[:rows, None]
    copies = 1 if fixed_plan is not None else max(1, min(64, -(-base.MID_BENCH_BYTES // (
        weight.numel() * weight.element_size()))))
    weights = [weight] + [weight.clone() for _ in range(copies - 1)]
    generator = torch.Generator(device=device).manual_seed(0)
    result = {"kind": kind, "prefix": prefix, "shape": [n, k], "copies": copies, "fixed": fixed_plan is not None,
              "buckets": {}}

    def pinned(x, w):
        return base._pinned_bf16(x, w) if kind == "bf16" else base.w8a16_gemm(x, w, scale)

    chosen_plan = {}
    for bucket in buckets:
        x = torch.randn((bucket, k), generator=generator, device=device).to(torch.bfloat16)
        ref = x.float() @ ref_w.t()
        entry = {"pinned_err": base._rel(pinned(x, weight)[:, :rows], ref)}
        limit = max(base.MID_ERROR_FACTOR * entry["pinned_err"], entry["pinned_err"] + base.MID_ERROR_SLACK)
        if fixed_plan is not None:
            planned = fixed_plan.get(bucket)
            candidates = [] if planned is None else [wide_mid_tile(bucket, planned)]
        else:
            entry["pinned_us"] = base.time_graph(lambda i, x=x: pinned(x, weights[i]), copies)
            candidates = [wide_mid_tile(bucket, c) for c in WIDE_CANDIDATES] if bucket > 32 else [
                wide_mid_tile(bucket, c) for c in base.MID_CANDIDATES]
        best, rows_out = None, []
        for tile in candidates:
            row = {"tile": list(tile)}
            try:
                row["err"] = base._rel(base.w8a16_gemm(x, weight, s, tile)[:, :rows], ref)
                if fixed_plan is None:
                    row["us"] = base.time_graph(lambda i, x=x, tile=tile: base.w8a16_gemm(x, weights[i], s, tile),
                                                copies)
            except Exception as exc:  # noqa: BLE001 - a tile that does not fit (shared memory) is skipped
                row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                rows_out.append(row)
                continue
            rows_out.append(row)
            if row["err"] <= limit and (fixed_plan is not None or best is None or row["us"] < best[1]):
                best = (tile, row.get("us"))
        chosen = None
        if best is not None and (fixed_plan is not None or best[1] <= entry["pinned_us"] * (1.0 - MIN_GAIN)):
            chosen = tuple(best[0])
        entry.update(candidates=rows_out, chosen=None if chosen is None else list(chosen))
        chosen_plan[bucket] = chosen
        result["buckets"][str(bucket)] = entry
    weights.clear()
    gc.collect()
    torch.cuda.empty_cache()
    _STATE["results"].append(result)
    log("WIDE", result)
    return chosen_plan


def wide_tune_mid(original, kind, weight, scale, prefix):
    """``arc_dense_kernels._tune_mid`` plus the wide buckets (and every bucket of a shape the fixed plan lacks)."""
    base = _STATE["base"]
    n, k = int(weight.shape[0]), int(weight.shape[1])
    done = (kind, n, k) in base._STATE["mid_done"]
    original(kind, weight, scale, prefix)
    if done or ("wide", kind, n, k) in base._STATE["mid_done"]:
        return
    base._STATE["mid_done"].add(("wide", kind, n, k))
    plan = base._STATE["mid"][kind].setdefault((n, k), {})
    fixed = isinstance(base.load_config()["mid_m"], dict)
    all_buckets = base.MID_M_BUCKETS if kind == "bf16" else base.MID_INT8_BUCKETS
    if fixed and not any(b in plan for b in all_buckets if b <= 32):
        buckets = [b for b in all_buckets]  # a shape the fixed plan lacks (the pruned router): tune every bucket
    else:
        wide = (_STATE["config"] or {}).get("buckets", WIDE_BUCKETS)
        buckets = [b for b in all_buckets if b in wide and plan.get(b) is None]
    if not buckets:
        return
    try:
        plan.update(_tune_buckets(kind, weight, scale, prefix, buckets))
    except Exception as exc:  # noqa: BLE001 - the wide buckets fall back to the pinned path; the plan's tiles stay
        gc.collect()
        torch.cuda.empty_cache()
        log("WIDE", {"kind": kind, "prefix": prefix, "shape": [n, k], "failed": f"{type(exc).__name__}: {exc}"[:400]})


def install():
    """Apply the wide buckets to ``arc_dense_kernels`` (idempotent; called by the patched pinned
    ``enable_qwen38next_low_latency_gemm`` before ``arc_dense_kernels.enable``)."""
    if _STATE["installed"]:
        return
    from vllm.models.qwen3_8_flash_next.nvidia import arc_dense_kernels as base

    _install(base, load_config())


def _install(base, config):
    _STATE["base"], _STATE["config"] = base, config
    if not config["mid_wide"]:
        _STATE["installed"] = True
        return
    _STATE["fixed"] = config["fixed"]
    wide = config.get("buckets", WIDE_BUCKETS)
    base.MID_M_BUCKETS = tuple(sorted(set(base.MID_M_BUCKETS) | set(wide)))
    base.MID_INT8_BUCKETS = tuple(sorted(set(base.MID_INT8_BUCKETS) | set(wide)))
    base.MID_BF16_SHAPES = tuple(base.MID_BF16_SHAPES) + tuple(s for s in config["shapes"]
                                                               if s not in base.MID_BF16_SHAPES)
    base._mid_tile = wide_mid_tile
    base.mid_bucket = wide_mid_bucket
    original = base._tune_mid

    def tune(kind, weight, scale, prefix):
        return wide_tune_mid(original, kind, weight, scale, prefix)

    base._tune_mid = tune
    _STATE["installed"] = True
    log("WIDE_INSTALL", {"mid_m_buckets": list(base.MID_M_BUCKETS), "mid_int8_buckets": list(base.MID_INT8_BUCKETS),
                         "extra_shapes": [list(s) for s in config["shapes"]],
                         "fixed_shapes": {kind: len(v) for kind, v in config["fixed"].items()}})
