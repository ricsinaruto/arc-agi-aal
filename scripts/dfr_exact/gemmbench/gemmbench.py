"""Dense BF16 GEMM roofline bench (throughput item 3, 24 Oct; gpu-script beside the idle server).

For each served dense projection shape of Flash at TP1 (N, K, calls per decode step) and decode M values, times
torch F.linear (cuBLAS, as served) and a Triton skinny GEMM (autotuned over tile/split-K configs) with cold weights
(rotating copies > L2), and reports GB/s vs the HBM roofline, the per-step dense total at the served M, and the
Triton result's error vs an fp32 reference next to cuBLAS's. --dry runs the cuBLAS path on CPU (logic check)."""
import argparse
import json
import time

import torch

# (name, N, K, calls per target decode forward); TP1 shapes of Flash-Next (hidden 2560, 48 layers: 36 GDN + 12 QSA)
SHAPES = [
    ("gdn_in_qkvz", 16384, 2560, 36), ("gdn_in_ba", 96, 2560, 36), ("gdn_out", 2560, 6144, 36),
    ("qsa_qkv_gate", 14336, 2560, 12), ("qsa_out", 2560, 6144, 12), ("qsa_indexer", 640, 2560, 12),
    ("shared_gate_up", 1280, 2560, 48), ("shared_down", 2560, 640, 48), ("router", 512, 2560, 48),
    ("hc", 336, 10240, 48), ("lm_head", 248320, 2560, 1),
    # draft-vocab trim candidate: the 2 MTP draft passes over the excl5 set + ids < 98,304 (98,400 ids) vs full vocab
    ("draft_lm_head_trim", 98400, 2560, 0),
]
BW = 1.79e12  # RTX PRO 6000 HBM bytes/s


def timeit(fn, n, sync):
    for _ in range(3):
        fn(0)
    sync()
    t0 = time.perf_counter()
    for i in range(n):
        fn(i)
    sync()
    return (time.perf_counter() - t0) / n


def triton_linear():
    import triton
    import triton.language as tl

    configs = []
    for bm in (16, 32, 64, 128):
        for bn in (32, 64, 128, 256):
            for bk in (64, 128, 256):
                for sk in (1, 2, 4, 8):
                    for w in (4, 8):
                        if bm * bn > 128 * 128 or bm * bk > 128 * 256 or bn * bk > 256 * 128:
                            continue
                        configs.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "SK": sk}, num_warps=w, num_stages=3))

    def prune(cfgs, nargs, **kw):
        m = nargs["M"]
        keep = [c for c in cfgs if c.kwargs["BM"] >= min(m, 16) and c.kwargs["BM"] <= max(16, triton.next_power_of_2(m))]
        k = nargs["K"]
        return [c for c in keep if k % (c.kwargs["BK"] * c.kwargs["SK"]) == 0] or keep[:1]

    @triton.autotune(configs=configs, key=["M", "N", "K"], prune_configs_by={"early_config_prune": prune},
                     reset_to_zero=["C"])
    @triton.jit
    def kern(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr):
        # C[M, N] (fp32) += A[M, K] @ B[N, K]^T over this program's K slice; split-K partials add atomically.
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        ps = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        kchunk = K // SK
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(ps * kchunk, (ps + 1) * kchunk, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(A + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
            b = tl.load(B + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
            acc += tl.dot(a, tl.trans(b))
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        if SK == 1:
            tl.store(C + rm[:, None] * N + rn[None, :], acc, mask=mask)
        else:
            tl.atomic_add(C + rm[:, None] * N + rn[None, :], acc, mask=mask)

    def run(a, w):
        m, k = a.shape
        n = w.shape[0]
        c = torch.zeros((m, n), device=a.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]), meta["SK"])
        kern[grid](a, w, c, m, n, k)
        return c.to(torch.bfloat16), kern

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="40,80,120,160")
    ap.add_argument("--served-m", type=int, default=120)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cpu" if a.dry else "cuda"
    sync = (lambda: None) if a.dry else torch.cuda.synchronize
    lin = None if a.dry else triton_linear()
    ms = [int(x) for x in a.ms.split(",")]
    rows = []
    for name, n, k, calls in SHAPES:
        nbytes = n * k * 2
        copies = max(2, min(16, int(4 * 2**29 // nbytes) + 1))  # rotate > 2 GB of weights: cold in the 128 MB L2
        if a.dry:
            n, k, copies = min(n, 512), k, 2
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        for m in ms:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            ref = (x.float() @ ws[0].float().t())
            t_cub = timeit(lambda i: torch.nn.functional.linear(x, ws[i % copies]), a.iters, sync)
            err_cub = (torch.nn.functional.linear(x, ws[0]).float() - ref).abs().max().item()
            row = {"name": name, "N": n, "K": k, "M": m, "calls": calls, "bytes": nbytes, "cublas_us": t_cub * 1e6,
                   "cublas_gbs": nbytes / t_cub / 1e9, "roofline_us": nbytes / BW * 1e6, "cublas_err": err_cub}
            if lin is not None:
                try:
                    out, kern = lin(x, ws[0])  # autotune once on copy 0
                    t_tri = timeit(lambda i: lin(x, ws[i % copies]), a.iters, sync)
                    row.update(triton_us=t_tri * 1e6, triton_gbs=nbytes / t_tri / 1e9,
                               triton_err=(out.float() - ref).abs().max().item(),
                               triton_cfg=str(kern.best_config))
                except Exception as error:  # a shape the kernel can't run: keep the cuBLAS row
                    row["triton_error"] = repr(error)[:300]
            rows.append(row)
            print(json.dumps(row), flush=True)
        del ws
        if not a.dry:
            torch.cuda.empty_cache()
    m = a.served_m
    sel = [r for r in rows if r["M"] == m]
    summ = {"served_m": m,
            "cublas_ms_per_step": sum(r["cublas_us"] * r["calls"] for r in sel) / 1e3,
            "roofline_ms_per_step": sum(r["roofline_us"] * r["calls"] for r in sel) / 1e3,
            "best_ms_per_step": sum(min(r["cublas_us"], r.get("triton_us", 1e18)) * r["calls"] for r in sel) / 1e3}
    print(json.dumps(summ), flush=True)
    json.dump({"rows": rows, "summary": summ}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
