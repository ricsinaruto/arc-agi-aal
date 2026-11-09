"""Kernel work, phase 6: kernel-level measurements next to an idle arena server (docs/validation-20260927/
kernels-phase6.md). Run by the serving bench in the server's runtime environment (``kv8-runtime-env.json``; the
bundle's ``src`` is on its PYTHONPATH) as ``python -m arc_infra.phase6_microbench OUT.json [--model DIR]``.

- ``dense``: the decode-size dense GEMMs at the hero's graph sizes (M buckets 8 / 16 / 24 / 32) on random
  weights of each served shape: the pinned path (skinny / cuBLAS for BF16, the phase-2 INT8 tile), every mid-M
  candidate tile, two-launch against fused split-K (``arc_dense_fused``; bitwise equality checked), microseconds
  under CUDA-graph replay over >= 256 MiB of weight copies (the load-time tuner's method) and FP32-relative error.
- ``qsa``: the int8 QSA sparse attention (``vllm.arc_kv8``) at decode rows 8 / 16 / 24 / 32 (the graph sizes):
  the pinned launch profile against a grid of (BLOCK_N, splits, warps), time and error against the pinned output
  and an FP32 reference. The pinned profile was tuned on GB300; this is the SM120 check.
- ``w4``: weight-only INT4 (group-wise, symmetric and asymmetric) against the served per-channel INT8 on the real
  BF16 dense weights of the checkpoint (the INT8 families), Gaussian activations as ``ARC_DENSE_CHECK``.

The pure-torch parts (``quantize_*``, ``w4_errors``) also run on the CPU (tests, the local W4 check on the MTP
layer's BF16 weights: scripts/kernel_phase6_w4.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# INT8 families (vLLM's fused shapes) and the checkpoint tensors each is made of, on one GDN and one QSA layer.
W4_TENSORS = {
    "gdn_in_proj_qkvz": ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                         "model.language_model.layers.0.linear_attn.in_proj_z.weight"),
    "attn_out(gdn)": ("model.language_model.layers.0.linear_attn.out_proj.weight",),
    "qsa_qkv": ("model.language_model.layers.3.self_attn.q_proj.weight",
                "model.language_model.layers.3.self_attn.k_proj.weight",
                "model.language_model.layers.3.self_attn.v_proj.weight"),
    "attn_out(qsa)": ("model.language_model.layers.3.self_attn.o_proj.weight",),
    "lm_head": ("lm_head.weight",),
}
W4_GROUPS = (32, 64, 128)
MID_BUCKETS = (8, 16, 24, 32)
# (N, K) and kind of every decode-size dense GEMM of the hero (dense_kernels_runtime MID_BF16_SHAPES + INT8 families).
DENSE_SHAPES = (
    ("bf16", 336, 10240), ("bf16", 320, 10240), ("bf16", 10240, 320), ("bf16", 512, 2560), ("bf16", 96, 2560),
    ("bf16", 640, 2560), ("bf16", 1280, 2560), ("bf16", 2560, 640), ("bf16", 2560, 2560),
    ("int8", 16384, 2560), ("int8", 13312, 2560), ("int8", 2560, 6144), ("int8", 248320, 2560),
)
QSA_ROWS = (8, 16, 24, 32)
QSA_TOPKS = (2051, 1027)
QSA_GRID = [(bn, splits, warps) for bn in (16, 32, 64) for splits in (4, 8, 16, 32, 64) for warps in (2, 4)]


def log(kind, payload):
    print(f"ARC_P6_{kind} " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def rel(a, b):
    b = b.float()
    return float((a.float() - b).norm() / b.norm().clamp_min(1e-30))


# ----------------------------------------------------------------------------- weight quantization (CPU or GPU)
def quantize_int8_rows(w):
    """The served INT8: per output channel, scale = amax / 127, round to nearest (dense_kernels_runtime)."""
    w = w.float()
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
    return torch.round(w / scale[:, None]).clamp_(-127, 127) * scale[:, None]


def quantize_int4_groups(w, group, symmetric=True):
    """Weight-only INT4 over groups of ``group`` input channels: symmetric absmax (levels -7..7, as the common
    W4A16 kernels store sign-magnitude-free int4 with 8 as zero) or asymmetric min/max with a zero point (0..15).
    Scales are rounded to BF16 as a W4A16 kernel would store them. Returns the dequantized FP32 weight."""
    n, k = w.shape
    if k % group:
        raise ValueError(f"K = {k} is not a multiple of the group {group}")
    g = w.float().reshape(n, k // group, group)
    if symmetric:
        scale = (g.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-12).to(torch.bfloat16).float()
        q = torch.round(g / scale).clamp_(-7, 7)
        out = q * scale
    else:
        lo, hi = g.amin(dim=2, keepdim=True), g.amax(dim=2, keepdim=True)
        scale = ((hi - lo) / 15.0).clamp(min=1e-12).to(torch.bfloat16).float()
        zero = torch.round(-lo / scale).clamp_(0, 15)
        q = (torch.round(g / scale) + zero).clamp_(0, 15)
        out = (q - zero) * scale
    return out.reshape(n, k)


def w4_errors(weight, rows=8192, m=16, seed=0):
    """Output error (relative Frobenius against FP32 on the BF16 weight, Gaussian BF16 activations) of the served
    per-channel INT8 and of INT4 per group, plus the weight error itself. First ``rows`` output channels."""
    w = weight[:rows].float()
    generator = torch.Generator(device=w.device).manual_seed(seed)
    x = torch.randn((m, w.shape[1]), generator=generator, device=w.device).to(torch.bfloat16).float()
    ref = x @ w.t()
    result = {"shape": list(weight.shape), "rows": int(w.shape[0]), "m": m}

    def row(dequant):
        return {"output_rel": round(rel(x @ dequant.t(), ref), 5), "weight_rel": round(rel(dequant, w), 5)}

    result["int8_channel"] = row(quantize_int8_rows(w))
    for group in W4_GROUPS:
        if w.shape[1] % group == 0:
            result[f"int4_g{group}_sym"] = row(quantize_int4_groups(w, group, True))
            result[f"int4_g{group}_asym"] = row(quantize_int4_groups(w, group, False))
    # Bytes per weight including BF16 scales (and zero points): the bandwidth side of the trade.
    result["bytes_per_weight"] = {"int8_channel": 1.0, **{f"int4_g{g}_sym": 0.5 + 2 / g for g in W4_GROUPS},
                                  **{f"int4_g{g}_asym": 0.5 + 2 / g + 0.5 / g for g in W4_GROUPS}}
    return result


def w4_checkpoint(model_dir):
    """``w4_errors`` on the checkpoint's BF16 dense weights (the INT8 families' tensors, concatenated as vLLM
    fuses them)."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for family, names in W4_TENSORS.items():
        try:
            parts = []
            for name in names:
                with safe_open(str(model_dir / index[name]), framework="pt", device="cpu") as handle:
                    parts.append(handle.get_tensor(name))
            weight = torch.cat(parts, dim=0)
            if weight.dtype != torch.bfloat16:
                raise TypeError(f"{family}: {weight.dtype}, expected BF16")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            out[family] = w4_errors(weight.to(device))
            out[family]["tensors"] = list(names)
            log("W4", {"family": family, **{k: v for k, v in out[family].items() if k != "bytes_per_weight"}})
            del weight, parts
        except Exception as exc:  # noqa: BLE001 - one missing tensor must not lose the rest
            out[family] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return out


# ----------------------------------------------------------------------------- dense GEMMs (GPU)
def dense_modules():
    from vllm.models.qwen3_8_flash_next.nvidia import arc_dense_kernels as base
    from vllm.models.qwen3_8_flash_next.nvidia import low_latency_gemm as pinned

    fused = None
    try:
        from vllm.models.qwen3_8_flash_next.nvidia import arc_dense_fused as fused

        fused.install()
    except ImportError:
        fused = None
    # The pinned BF16 path needs the TP=1 skinny plans that arc_dense_kernels.enable registers at load.
    for n, k in base.SKINNY_SHAPES:
        plan = base.skinny_plan(n, k)
        if plan:
            pinned.QWEN38NEXT_GEMM_PLANS[(n, k)] = plan
    return base, fused


def dense_sweep(budget_seconds=420, *, device="cuda", timer=None, shapes=DENSE_SHAPES, buckets=MID_BUCKETS,
                modules=None, max_copies=32):
    """``timer(fn, calls)`` defaults to the tuner's CUDA-graph replay (``arc_dense_kernels.time_graph``); tests pass a
    CPU stand-in with small shapes."""
    base, fused = modules or dense_modules()
    timer = timer or base.time_graph
    two_launch = fused._STATE["original"] if fused is not None and fused._STATE.get("original") else base.w8a16_gemm
    fused_fn = fused.fused_w8a16_gemm if fused is not None and fused._STATE.get("original") else None
    device = torch.device(device)
    started = time.monotonic()
    rows = []
    candidates = list(dict.fromkeys(tuple(c) for c in base.MID_CANDIDATES))
    for kind, n, k in shapes:
        if time.monotonic() - started > budget_seconds:
            rows.append({"kind": kind, "shape": [n, k], "skipped": "time budget"})
            continue
        generator = torch.Generator(device=device).manual_seed(n * 7 + k)
        if kind == "bf16":
            weight = (torch.randn((n, k), generator=generator, device=device) * 0.02).to(torch.bfloat16)
            scale = torch.ones((n,), dtype=torch.float32, device=device)
            ref_w = weight[:8192].float()
        else:
            weight = torch.randint(-127, 128, (n, k), generator=generator, device=device, dtype=torch.int8)
            scale = torch.rand((n,), generator=generator, device=device) * 2e-4 + 1e-4
            ref_w = weight[:8192].float() * scale[:8192, None]
        nbytes = weight.numel() * weight.element_size()
        copies = max(1, min(max_copies, -(-base.MID_BENCH_BYTES // nbytes)))
        weights = [weight] + [weight.clone() for _ in range(copies - 1)]
        for bucket in buckets:
            x = torch.randn((bucket, k), generator=generator, device=device).to(torch.bfloat16)
            ref = x.float() @ ref_w.t()
            entry = {"kind": kind, "shape": [n, k], "m": bucket, "copies": copies, "tiles": []}

            def pinned(i, x=x):
                return base._pinned_bf16(x, weights[i]) if kind == "bf16" else two_launch(x, weights[i], scale)

            try:
                entry["pinned_us"] = round(timer(pinned, copies), 3)
                entry["pinned_err"] = round(rel(pinned(0)[:, :8192], ref), 6)
            except Exception as exc:  # noqa: BLE001
                entry["pinned_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            for cfg in candidates:
                tile = base._mid_tile(bucket, cfg)
                row = {"tile": list(cfg)}
                try:
                    row["us"] = round(timer(lambda i, t=tile, x=x: two_launch(x, weights[i], scale, t), copies), 3)
                    out = two_launch(x, weight, scale, tile)
                    row["err"] = round(rel(out[:, :8192], ref), 6)
                    if fused_fn is not None:
                        row["fused_us"] = round(timer(lambda i, t=tile, x=x: fused_fn(x, weights[i], scale, t),
                                                      copies), 3)
                        row["fused_bitwise"] = bool(torch.equal(fused_fn(x, weight, scale, tile), out))
                except Exception as exc:  # noqa: BLE001 - a tile that does not fit is recorded and skipped
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                entry["tiles"].append(row)
            rows.append(entry)
            log("DENSE", {k2: v for k2, v in entry.items() if k2 != "tiles"}
                | {"best": min((t for t in entry["tiles"] if "us" in t), key=lambda t: t.get("fused_us", t["us"]),
                               default=None)})
        weights.clear()
        del weight
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"rows": rows, "fused_installed": fused_fn is not None, "candidates": [list(c) for c in candidates],
            "seconds": round(time.monotonic() - started, 1)}


# ----------------------------------------------------------------------------- QSA int8 attention (GPU)
def qsa_sweep(page=1515, budget_seconds=420, *, device="cuda", timer=None, rows_list=QSA_ROWS, topks=QSA_TOPKS,
              grid=QSA_GRID, min_blocks=64):
    """``timer(fn)`` defaults to the kv8 bench's CUDA-graph replay (``vllm.arc_kv8._time``)."""
    from vllm import arc_kv8 as kv8

    timer = timer or kv8._time
    started = time.monotonic()
    rows = []
    for topk in topks:
        for n_rows in rows_list:
            if time.monotonic() - started > budget_seconds:
                rows.append({"topk": topk, "rows": n_rows, "skipped": "time budget"})
                continue
            requests = n_rows
            blocks = max(min_blocks, requests * 6)
            q, k, v, indices, table, t2r = kv8.synthetic_case(n_rows, page=page, blocks=blocks, topk=topk,
                                                              requests=requests, device=device, fill=1.0)
            packed = kv8.pack_cache(k, v, "int8")
            ck, cv, sk, sv = kv8.cache_views(packed, 256, False)
            reference = kv8.attention_reference(q[:4], k, v, indices[:4], table, t2r[:4])
            pinned_profile = kv8.launch_profile(n_rows, 2, 12, topk)
            out = torch.empty_like(q)
            kv8.sparse_attention(q, ck, cv, sk, sv, indices, table, t2r, out)
            pinned_out = out.clone()
            entry = {"topk": topk, "rows": n_rows, "page": page, "pinned_profile": pinned_profile,
                     "pinned_us": round(timer(lambda: kv8.sparse_attention(q, ck, cv, sk, sv, indices, table, t2r,
                                                                           out)), 3),
                     "pinned_err_fp32": round(rel(pinned_out[:4], reference), 6), "grid": []}
            for block_n, splits, warps in grid:
                tiles = -(-topk // block_n)
                if splits > tiles:
                    continue
                profile = {"block_m": pinned_profile["block_m"], "block_n": block_n, "splits": splits,
                           "warps": warps, "tiles": tiles}
                row = {"block_n": block_n, "splits": splits, "warps": warps}
                try:
                    kv8.sparse_attention(q, ck, cv, sk, sv, indices, table, t2r, out, profile=profile)
                    row["rel_to_pinned"] = round(rel(out, pinned_out), 7)
                    row["err_fp32"] = round(rel(out[:4], reference), 6)
                    row["us"] = round(timer(lambda p=profile: kv8.sparse_attention(
                        q, ck, cv, sk, sv, indices, table, t2r, out, profile=p)), 3)
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                entry["grid"].append(row)
            timed = [r for r in entry["grid"] if "us" in r]
            if timed:
                entry["best"] = min(timed, key=lambda r: r["us"])
            rows.append(entry)
            log("QSA", {k2: v2 for k2, v2 in entry.items() if k2 != "grid"})
            packed = k = v = None  # free the pages before the next case (the closures above are done)
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    return {"rows": rows, "seconds": round(time.monotonic() - started, 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="phase-6 kernel microbenchmarks next to an idle server")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--parts", default="dense,qsa,w4")
    parser.add_argument("--budget-seconds", type=float, default=400.0)
    args = parser.parse_args(argv)
    parts = [p for p in args.parts.split(",") if p]
    report = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "parts": parts}
    started = time.monotonic()
    for part in parts:
        try:
            if part == "dense":
                report["dense"] = dense_sweep(args.budget_seconds)
            elif part == "qsa":
                report["qsa"] = qsa_sweep(budget_seconds=args.budget_seconds)
            elif part == "w4":
                if args.model is None:
                    raise ValueError("--model is needed for the checkpoint W4 check")
                report["w4"] = w4_checkpoint(args.model)
            else:
                raise ValueError(f"unknown part {part}")
        except Exception as exc:  # noqa: BLE001 - one failed part must not lose the others
            report[part] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        args.output.write_text(json.dumps(report, indent=1))
    report["seconds"] = round(time.monotonic() - started, 1)
    args.output.write_text(json.dumps(report, indent=1))
    log("DONE", {"seconds": report["seconds"], "parts": parts})
    return 0


if __name__ == "__main__":
    sys.exit(main())
