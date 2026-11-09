"""QSA sparse attention at prefill shapes: the pinned BF16 kernel against the int8 (kv8) kernel over a launch-profile
grid, and the dequantize-to-BF16 cost (kernel work, phase 7; docs/validation-20260927/kernels-phase7.md).

No server: one GPU, synthetic paged K/V. A prefill chunk is ``rows`` query rows of one request (24 query heads, 2 KV
heads, 256 dims) at a context of ``ctx`` tokens in pages of ``page`` tokens; each row attends to ``topk`` selected
tokens. Two selection patterns: ``random`` (independent per row: the fewest reused gathers) and ``shared`` (every row
keeps ``overlap`` of one common set, the rest random: neighbouring query rows of a real chunk select similar keys).

Methods per case (µs per call under CUDA-graph replay, error against FP32 on the dequantized K/V for 8 rows):
- ``bf16_pinned``: the pinned ``qsa_sparse_paged_attention`` on the BF16 cache (what a BF16-KV server runs);
- ``int8_pinned_profile``: ``arc_kv8.sparse_attention`` with its pinned (GB300) profile (what the hero runs);
- ``int8 grid`` and ``bf16 grid``: both kernels over (BLOCK_N, splits, warps, stages);
- ``dequant``: int8 codes + per-token-head scales -> BF16 K/V for the request's pages (option b's extra work), torch
  and a Triton kernel.

Run (Modal): scripts/modal_qsa_bench.py. Locally (CPU, Triton interpreter): the tests call ``run_case`` on tiny shapes.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _dequant_kv8_kernel(codes_ptr, scale_ptr, out_ptr, stride_c_row, stride_s_row, HEAD_DIM: tl.constexpr):
    """One (token, head) row: out = bf16(code x scale), the value the int8 kernel multiplies by (codes are exact)."""
    row = tl.program_id(0)
    dims = tl.arange(0, HEAD_DIM)
    code = tl.load(codes_ptr + row * stride_c_row + dims).to(tl.float32)
    scale = tl.load(scale_ptr + row * stride_s_row)
    tl.store(out_ptr + row * HEAD_DIM + dims, (code * scale).to(tl.bfloat16))


def dequant_triton(codes, scale):
    """codes [blocks, page, heads, dim] int8 (strided view into the packed page), scale [blocks, page, heads] fp32 ->
    contiguous BF16 [blocks, page, heads, dim]."""
    blocks, page, heads, dim = codes.shape
    flat_c = codes.reshape(blocks * page * heads, dim) if codes.is_contiguous() else None
    out = torch.empty((blocks, page, heads, dim), dtype=torch.bfloat16, device=codes.device)
    if flat_c is None:
        # The packed layout: codes are a strided slice of the page row; walk it as (token-head) rows of fixed stride.
        assert codes.stride(3) == 1 and codes.stride(2) * heads == codes.stride(1) and codes.stride(1) * page == \
            codes.stride(0), "unexpected kv8 page layout"
        assert scale.stride(1) == scale.stride(2) * heads and scale.stride(0) == scale.stride(1) * page
    _dequant_kv8_kernel[(blocks * page * heads,)](codes, scale, out, codes.stride(2), scale.stride(2), HEAD_DIM=dim)
    return out


def dequant_torch(codes, scale):
    return (codes.float() * scale.unsqueeze(-1)).to(torch.bfloat16)


def indices_for(rows, topk, length, pattern, overlap, generator, device):
    if pattern == "random":
        keys = torch.rand((rows, length), generator=generator, device="cpu")
        return keys.topk(topk, dim=1).indices.to(torch.int32).to(device)
    common = torch.randperm(length, generator=generator)[:topk]
    keep = int(topk * overlap)
    out = common.repeat(rows, 1)
    extra = torch.randint(0, length, (rows, topk - keep), generator=generator)
    out[:, keep:] = extra
    return out.to(torch.int32).to(device)


def launch_bf16(pinned, q, k, v, indices, table, t2r, out, profile, stages):
    """The pinned BF16 split-K kernel with an explicit profile (its wrapper chooses one from the row count)."""
    block_n, splits, warps = profile
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


def launch_int8(kv8, q, ck, cv, sk, sv, indices, table, t2r, out, profile, stages):
    """``arc_kv8.sparse_attention``'s launch with an explicit profile and pipeline depth."""
    block_n, splits, warps = profile
    group = q.shape[1] // ck.shape[2]
    tiles = triton.cdiv(indices.shape[1], block_n)
    if splits == 1:
        part_o, part_l = out, out
    else:
        part_o = torch.empty((splits, *q.shape), dtype=torch.float32, device=q.device)
        part_l = torch.empty((splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    kv8._qsa_kv8_splitk_kernel[(q.shape[0], ck.shape[2], splits)](
        q, ck, cv, sk, sv, indices, table, t2r, part_o, part_l, out,
        q.stride(0), q.stride(1), ck.stride(0), ck.stride(1), ck.stride(2), cv.stride(0), cv.stride(1), cv.stride(2),
        sk.stride(0), sk.stride(1), sk.stride(2), sv.stride(0), sv.stride(1), sv.stride(2),
        indices.stride(0), table.stride(0), out.stride(0), out.stride(1), q.shape[0], ck.shape[0], table.shape[0],
        TOPK=indices.shape[1], PAGE_SIZE=ck.shape[1], PAGE_TABLE_WIDTH=table.shape[1], GROUP_SIZE=group,
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=splits, NUM_TILES=tiles,
        BLOCK_M=triton.next_power_of_2(group), BLOCK_N=block_n, INTERPRET=kv8._interpreting(),
        num_warps=warps, num_stages=stages)
    if splits > 1:
        kv8._qsa_kv8_merge_kernel[(q.shape[0], q.shape[1])](
            part_o, part_l, out, out.stride(0), out.stride(1), q.shape[0], HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=splits, BLOCK_SPLITS=triton.next_power_of_2(splits),
            num_warps=2, num_stages=1)
    return out


# Prefill has >= 8,192 base programs, so splits above 2 only add merge work; the pinned profile is (64, 1, 2) with
# 2 stages. Each configuration compiles once per case (~2-4 s), so the grid is kept to the informative part.
GRID = [(bn, s, w) for bn in (32, 64, 128) for s in (1, 2) for w in (2, 4, 8)]
BF16_GRID = [(bn, 1, w) for bn in (32, 64, 128) for w in (2, 4, 8)]
STAGES = (2, 3)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def run_case(kv8, pinned, *, rows, ctx, page, topk, pattern, overlap=0.8, device="cuda", timer=None, grid=GRID,
             stages=STAGES, check_rows=8, seed=0, budget_seconds=None, bf16_grid=None):
    """One prefill case: returns a dict of timings and errors (``timer(fn) -> µs``)."""
    started = time.monotonic()
    generator = torch.Generator().manual_seed(seed)
    blocks = -(-ctx // page)
    kv_heads, group, dim = 2, 12, 256
    k = torch.randn((blocks, page, kv_heads, dim), generator=generator)
    v = torch.randn((blocks, page, kv_heads, dim), generator=generator)
    k[..., :4] *= 4.0  # outlier channels, as kv8.synthetic_case
    q = torch.randn((rows, kv_heads * group, dim), generator=generator).to(torch.bfloat16).to(device)
    packed = kv8.pack_cache(k.to(device), v.to(device), "int8")
    ck, cv, sk, sv = kv8.cache_views(packed, dim, False)
    # (b)'s BF16 cache is the dequantized int8 one, so both methods see the same values.
    kb, vb = dequant_torch(ck, sk).contiguous(), dequant_torch(cv, sv).contiguous()
    table = torch.randperm(blocks, generator=generator).to(torch.int32).reshape(1, blocks).to(device)
    t2r = torch.zeros((rows,), dtype=torch.int32, device=device)
    indices = indices_for(rows, topk, ctx, pattern, overlap, generator, device)
    reference = kv8.attention_reference(q[:check_rows], kb, vb, indices[:check_rows], table, t2r[:check_rows])
    out = torch.empty_like(q)
    result = {"rows": rows, "ctx": ctx, "page": page, "topk": topk, "pattern": pattern,
              "overlap": overlap if pattern == "shared" else None,
              "pinned_profile": kv8.launch_profile(rows, kv_heads, group, topk)}
    pinned.qsa_sparse_paged_attention(q, kb, vb, indices, table, t2r, out)
    bf16_out = out.clone()
    result["bf16_pinned"] = {"us": timer(lambda: pinned.qsa_sparse_paged_attention(q, kb, vb, indices, table, t2r,
                                                                                   out)),
                             "err_fp32": rel(bf16_out[:check_rows], reference)}
    kv8.sparse_attention(q, ck, cv, sk, sv, indices, table, t2r, out)
    int8_out = out.clone()
    result["int8_pinned_profile"] = {"us": timer(lambda: kv8.sparse_attention(q, ck, cv, sk, sv, indices, table, t2r,
                                                                               out)),
                                     "err_fp32": rel(int8_out[:check_rows], reference),
                                     "rel_to_bf16": rel(int8_out, bf16_out)}
    for label, fn in (("torch", dequant_torch), ("triton", dequant_triton)):
        try:
            got_k = fn(ck, sk)
            result[f"dequant_{label}"] = {"us": timer(lambda f=fn: (f(ck, sk), f(cv, sv))),
                                          "equal_to_torch": bool(torch.equal(got_k, kb)),
                                          "rel_to_torch": rel(got_k, kb)}
        except Exception as exc:  # noqa: BLE001
            result[f"dequant_{label}"] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    for kind in ("int8", "bf16"):
        rows_out = []
        for profile in (grid if kind == "int8" or bf16_grid is None else bf16_grid):
            for st in stages:
                if budget_seconds is not None and time.monotonic() - started > budget_seconds:
                    rows_out.append({"profile": list(profile), "stages": st, "skipped": "time"})
                    continue
                entry = {"profile": list(profile), "stages": st}
                try:
                    if kind == "int8":
                        def call(p=profile, s=st):
                            return launch_int8(kv8, q, ck, cv, sk, sv, indices, table, t2r, out, p, s)
                    else:
                        def call(p=profile, s=st):
                            return launch_bf16(pinned, q, kb, vb, indices, table, t2r, out, p, s)
                    call()
                    entry["rel_to_pinned_same_kind"] = rel(out, int8_out if kind == "int8" else bf16_out)
                    entry["err_fp32"] = rel(out[:check_rows], reference)
                    entry["us"] = timer(call)
                except Exception as exc:  # noqa: BLE001 - a profile that does not fit is recorded
                    entry["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                rows_out.append(entry)
        timed = [e for e in rows_out if "us" in e]
        result[f"{kind}_grid"] = rows_out
        result[f"{kind}_best"] = min(timed, key=lambda e: e["us"]) if timed else None
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def cuda_timer(repeats=5):
    def timer(fn):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        del graph
        samples.sort()
        return round(samples[len(samples) // 2], 2)

    return timer


CASES = [  # (rows, ctx, pattern): the hero's injected-prefill chunk (4,096 rows) at 24K and 40K contexts
    (4096, 24576, "shared"), (4096, 40960, "shared"), (4096, 24576, "random"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kv8", type=Path, required=True, help="kv8_runtime.py (arc_kv8)")
    parser.add_argument("--pinned-qsa", type=Path, required=True, help="the pinned nvidia/ops/qsa.py")
    parser.add_argument("--page", type=int, default=1515)
    parser.add_argument("--topk", type=int, default=2051)
    parser.add_argument("--budget-seconds", type=float, default=120.0, help="per case")
    args = parser.parse_args(argv)
    kv8 = load_module("arc_kv8_bench", args.kv8)
    pinned = load_module("pinned_qsa_bench", args.pinned_qsa)
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
              "cases": []}
    for rows, ctx, pattern in CASES:
        try:
            case = run_case(kv8, pinned, rows=rows, ctx=ctx, page=args.page, topk=args.topk, pattern=pattern,
                            timer=cuda_timer(), budget_seconds=args.budget_seconds, bf16_grid=BF16_GRID)
        except Exception as exc:  # noqa: BLE001 - one failed case must not lose the others
            case = {"rows": rows, "ctx": ctx, "pattern": pattern, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        report["cases"].append(case)
        brief = {k: case.get(k) for k in ("ctx", "pattern", "bf16_pinned", "int8_pinned_profile", "int8_best",
                                          "bf16_best", "dequant_triton", "seconds", "error")}
        print("QSA_PREFILL " + json.dumps(brief, default=str), flush=True)
        args.out.write_text(json.dumps(report, indent=1))
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
