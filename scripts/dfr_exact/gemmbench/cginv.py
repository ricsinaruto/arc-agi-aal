"""Exact-CG bucket check (24 Oct; gpu-script beside the idle server).

Padded CUDA graphs over mixed steps shift numerics (EQ-CG: KL +0.02) because padding changes the token count M the
dense GEMMs see. For each served dense shape this finds the M runs over which F.linear (cuBLAS, as served) gives
bitwise-identical rows: run boundaries are where y_M[:M-1] != y_{M-1}. Padding n -> s is exact for that GEMM iff n and
s lie in the same run. Output: per-shape run starts, the common runs across shapes (exact capture sizes = each common
run's top), each M's same-M repeat determinism, and the mean padding waste of that capture set over mixed-step sizes.
--dry runs on CPU with small shapes (logic check)."""
import argparse
import json

import torch

SHAPES = [
    ("gdn_in_qkvz", 16384, 2560), ("gdn_in_ba", 96, 2560), ("gdn_out", 2560, 6144),
    ("qsa_qkv_gate", 14336, 2560), ("qsa_out", 2560, 6144), ("qsa_indexer", 640, 2560),
    ("shared_gate_up", 1280, 2560), ("shared_down", 2560, 640), ("router", 512, 2560), ("hc", 336, 10240),
    ("hc_up", 10240, 320), ("sq2560", 2560, 2560),
]
# --stack v87 (26 Oct): v87b's extra BF16 shapes and its FP8 W8A8 families (cutlass_scaled_mm, per-token activations)
V87_BF16 = [("qsa_qkv", 13312, 2560), ("hc_final", 320, 10240), ("ple_key", 10240, 2560)]
V87_FP8 = [("fp8_gdn_in_qkvz", 16384, 2560), ("fp8_qsa_qkv", 13312, 2560), ("fp8_attn_out", 2560, 6144)]


def fp8_linear():
    from vllm import _custom_ops as ops

    def f(x, wq_s):
        wq, s = wq_s
        xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        return ops.cutlass_scaled_mm(xq, wq.t(), xs, s, torch.bfloat16)
    return f


def fp8_weight(w):
    s = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    return (w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s.float()


def runs_for(w, x, m_max, f):
    """Run starts (1-based M where a new bitwise run begins) and M values whose repeat differs."""
    starts, nondet = [1], []
    prev = None
    for m in range(1, m_max + 1):
        y = f(x[:m], w)
        if not torch.equal(y, f(x[:m], w)):
            nondet.append(m)
        if prev is not None and not torch.equal(y[: m - 1], prev):
            starts.append(m)
        prev = y
    return starts, nondet


def common_runs(all_starts, m_max):
    s = sorted(set().union(*all_starts))
    tops = [b - 1 for b in s[1:]] + [m_max]
    return list(zip(s, tops))


def waste(runs, lo, hi):
    """Mean padded/real token ratio for n uniform in [lo, hi] when each n pads to its common run's top."""
    tot = 0.0
    cnt = 0
    for a, b in runs:
        for n in range(max(a, lo), min(b, hi) + 1):
            tot += b / n
            cnt += 1
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-max", type=int, default=2048)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stack", default="", choices=["", "v87"])
    a = ap.parse_args()
    dev = "cpu" if a.dry else "cuda"
    m_max = 64 if a.dry else a.m_max
    f = torch.nn.functional.linear
    g = torch.Generator(device="cpu").manual_seed(0)
    rows = {}
    shapes = [(s, False) for s in SHAPES]
    if a.stack == "v87":
        shapes += [(s, False) for s in V87_BF16] + ([] if a.dry else [(s, True) for s in V87_FP8])
    for (name, n, k), fp8 in shapes:
        if a.dry:
            n = min(n, 256)
        w = (torch.randn(n, k, generator=g) * 0.02).to(dev, torch.bfloat16)
        x = torch.randn(m_max, k, generator=g).to(dev, torch.bfloat16)
        starts, nondet = runs_for(fp8_weight(w) if fp8 else w, x, m_max, fp8_linear() if fp8 else f)
        rows[name] = {"N": n, "K": k, "run_starts": starts, "nondet_m": nondet[:50], "n_nondet": len(nondet)}
        print(json.dumps({name: {"runs": len(starts), "first_starts": starts[:40], "n_nondet": len(nondet)}}),
              flush=True)
        del w, x
    runs = common_runs([r["run_starts"] for r in rows.values()], m_max)
    summ = {"m_max": m_max, "common_runs": len(runs), "runs": runs,
            "waste_121_512": waste(runs, 121, min(512, m_max)), "waste_121_2048": waste(runs, 121, m_max),
            "any_nondet": any(r["n_nondet"] for r in rows.values())}
    print(json.dumps(summ), flush=True)
    json.dump({"shapes": rows, "summary": summ}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
