"""QSA indexer top-k / scoring micro-benchmark in the pinned runtime (throughput arm, 4 October).

Question: why does decode fall from ~34 to ~5 tok/s for every stream once one request's context passes ~150-190K
tokens (arena-competition-preview-dfr-256k-compact-4h)? Suspect: the indexer's block top-k
(``torch.ops._C.persistent_topk`` over ``columns`` = max blocks per request x 784 compressed rows) or its scoring
kernel (``qsa_mqa_paged``) changes path at some row length.

For rows in ROWS and columns in COLUMNS (fp32 logits, ``visible_blocks`` = columns for every row except a mixed case
where one row is long and the rest short), times (CUDA events, median of 20 after warm-up):
- ``persistent_topk`` (the served path on SM120),
- ``top_k_per_row_decode`` (vLLM's other decode top-k, when present),
- a torch fallback: logits masked past ``visible_blocks`` to -inf, ``torch.topk``, indices of -inf entries set to -1,
and checks that the fallback selects the same index sets as ``persistent_topk``.

Output: JSON lines to stdout and ``--out``.
"""
import argparse
import json
import time

import torch

ROWS = (1, 8, 28, 30, 60)
COLUMNS = (8192, 16384, 24576, 32768, 36864, 40960, 45056, 49152, 57344, 65536)
K = 512
WORKSPACE = 1024 * 1024


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def torch_topk(logits, visible, out, k):
    cols = logits.shape[1]
    mask = torch.arange(cols, device=logits.device)[None, :] >= visible[:, None]
    masked = logits.masked_fill(mask, float("-inf"))
    kk = min(k, cols)
    values, idx = torch.topk(masked, kk, dim=1, sorted=False)
    idx = torch.where(torch.isinf(values), torch.full_like(idx, -1), idx).to(torch.int32)
    out[:, :kk].copy_(idx)
    if kk < k:
        out[:, kk:] = -1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import vllm._C  # noqa: F401  (registers torch.ops._C)
    from vllm import _custom_ops as ops

    dev = torch.device("cuda")
    rows_out = []
    gen = torch.Generator(device=dev).manual_seed(0)
    workspace = torch.empty(WORKSPACE, dtype=torch.uint8, device=dev)
    has_per_row = hasattr(ops, "top_k_per_row_decode")
    for rows in ROWS:
        for cols in COLUMNS:
            for mixed in (False, True):
                if mixed and rows == 1:
                    continue
                logits = torch.rand((rows, cols), generator=gen, device=dev, dtype=torch.float32)
                visible = torch.full((rows,), cols, dtype=torch.int32, device=dev)
                if mixed:
                    visible[1:] = min(cols, 12000)  # one long request among short ones
                out_p = torch.full((rows, K), -1, dtype=torch.int32, device=dev)
                out_t = torch.full((rows, K), -1, dtype=torch.int32, device=dev)
                rec = {"rows": rows, "columns": cols, "context_tokens": cols * 4, "mixed": mixed}
                try:
                    rec["persistent_us"] = round(timed(lambda: torch.ops._C.persistent_topk(
                        logits, visible, out_p, workspace, K, cols)), 1)
                except Exception as exc:  # noqa: BLE001
                    rec["persistent_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                rec["torch_us"] = round(timed(lambda: torch_topk(logits, visible, out_t, K)), 1)
                if has_per_row:
                    out_r = torch.full((rows, K), -1, dtype=torch.int32, device=dev)
                    try:
                        rec["per_row_us"] = round(timed(lambda: ops.top_k_per_row_decode(
                            logits, 1, visible, out_r, rows, logits.stride(0), logits.stride(1), K)), 1)
                    except Exception as exc:  # noqa: BLE001
                        rec["per_row_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                if "persistent_us" in rec:
                    same = 0
                    for r in range(rows):
                        a = set(out_p[r].tolist()) - {-1}
                        b = set(out_t[r].tolist()) - {-1}
                        same += a == b
                    rec["same_sets"] = f"{same}/{rows}"
                rows_out.append(rec)
                print(json.dumps(rec), flush=True)
    # The scoring kernel at the same widths (one QSA layer's indexer call, 32 index heads x 128 dims).
    try:
        from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_mqa_paged

        page = 784
        for rows in (28, 60):
            for cols in COLUMNS:
                pages = cols // page + 1
                k_cache = torch.randn((pages * rows + 4, page, 1, 128), device=dev, dtype=torch.bfloat16)
                table = torch.arange(pages * rows, device=dev, dtype=torch.int32).reshape(rows, pages)
                q = torch.randn((rows, 32, 128), device=dev, dtype=torch.bfloat16)
                tok = torch.arange(rows, device=dev, dtype=torch.int32)
                pos = torch.full((rows,), cols * 4 - 1, device=dev, dtype=torch.int64)
                seq = torch.full((rows,), cols * 4, device=dev, dtype=torch.int32)
                us = timed(lambda: qsa_mqa_paged(q, k_cache, table, tok, pos, seq, 4))
                rec = {"kernel": "qsa_mqa_paged", "rows": rows, "columns": cols, "us": round(us, 1)}
                rows_out.append(rec)
                print(json.dumps(rec), flush=True)
    except Exception as exc:  # noqa: BLE001
        rows_out.append({"kernel": "qsa_mqa_paged", "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    with open(args.out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "k": K, "rows": rows_out, "epoch": time.time()}, f, indent=1)


if __name__ == "__main__":
    main()
