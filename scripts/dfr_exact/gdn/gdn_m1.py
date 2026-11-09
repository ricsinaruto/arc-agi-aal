"""GDN deferred-commit M1 (throughput arm, 19 Oct): bit-exactness of the deferred kernel against the served
fused_gdn_decode_post_conv_mtp op over consecutive MTP steps with random acceptance, and the 36-layer step time at
MTP1 / MTP3 (48 streams). Builds bench/gdn_m1.cu with the runtime's nvcc (plain and --use_fast_math)."""

import argparse
import ctypes
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch

N, H, HV, D = 48, 16, 48, 128
LAYERS = 36
RESULT = {"phases": []}


def phase(name, **kw):
    rec = {"phase": name, **kw}
    RESULT["phases"].append(rec)
    print(json.dumps(rec), flush=True)
    Path(ARGS.out).write_text(json.dumps(RESULT, indent=1))


def build(src, fast):
    lib = Path(src).with_name(f"libgdn{'_fast' if fast else ''}.so")
    nvcc = os.environ.get("CUDACXX") or "nvcc"
    cmd = [nvcc, "-O3", "-std=c++17", "-gencode=arch=compute_120,code=sm_120", "--shared", "-Xcompiler", "-fPIC",
           *(["--use_fast_math"] if fast else []), "-o", str(lib), src]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    so = ctypes.CDLL(str(lib))
    f = so.gdn_launch
    P, I, F, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_longlong
    f.argtypes = [I, I, P, P, P, P, P, I, P, I, P, P, P, P, P, I, P, P, P, P, P, P, P, I, I, I, F, F, L, L, L, L, L, P]
    f.restype = I
    return f


class Inputs:
    def __init__(self, T, g):
        L = N * T
        self.T = T
        self.mixed = (torch.randn(L, 2 * H * D + HV * D, generator=g, device="cuda") * 1.0).bfloat16()
        self.a = torch.randn(L, HV, generator=g, device="cuda").bfloat16()
        self.b = torch.randn(L, HV, generator=g, device="cuda").bfloat16()
        self.gate = torch.randn(L, HV, D, generator=g, device="cuda").bfloat16()
        self.cu = (torch.arange(N + 1, device="cuda", dtype=torch.int32) * T).contiguous()


class Params:
    def __init__(self, g):
        self.A_log = torch.log(torch.rand(HV, generator=g, device="cuda") * 15 + 1).float()
        self.dt_bias = (torch.randn(HV, generator=g, device="cuda") * 0.5).float()
        self.norm = (1 + 0.1 * torch.randn(D, generator=g, device="cuda")).float()


def served(ops, x, p, idx, acc, state, out, sigmoid):
    ops.fused_gdn_decode_post_conv_mtp(
        mixed_qkv=x.mixed, a=x.a, b=x.b, A_log=p.A_log, dt_bias=p.dt_bias, state_indices=idx, cu_seqlens=x.cu,
        num_accepted_tokens=acc, state=state, output_gate=x.gate, norm_weight=p.norm, out=out, scale=D ** -0.5,
        norm_eps=1e-6, output_gate_activation="sigmoid" if sigmoid else "silu")


def mine(f, mode, x, p, slots, counts, state, out, sigmoid, bufs=None):
    z = 0
    pk, pv, pdb, ck, cv, cdb = bufs if bufs else (z,) * 6
    ptr = lambda t: t if isinstance(t, int) else t.data_ptr()
    rc = f(mode, int(sigmoid), x.mixed.data_ptr(), x.a.data_ptr(), x.b.data_ptr(), p.A_log.data_ptr(),
           p.dt_bias.data_ptr(), 0, slots.data_ptr(), slots.shape[1] if slots.dim() == 2 else 1, x.cu.data_ptr(),
           counts.data_ptr(), state.data_ptr(), x.gate.data_ptr(), p.norm.data_ptr(), 0, out.data_ptr(),
           ptr(pk), ptr(pv), ptr(pdb), ptr(ck), ptr(cv), ptr(cdb), N, H, HV, D ** -0.5, 1e-6,
           x.mixed.stride(0), x.a.stride(0), x.b.stride(0), x.gate.stride(0), state.stride(0),
           torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"launch error {rc}")


def bufs_pair():
    mk = lambda: (torch.zeros(N, 8, H, D, device="cuda"), torch.zeros(N, 8, HV, D, device="cuda").bfloat16(),
                  torch.zeros(N, 8, HV, 2, device="cuda"))
    return mk(), mk()


def diff(x, y):
    xi, yi = x.view(torch.int16), y.view(torch.int16)
    ne = (xi != yi).sum().item()
    return {"neq": ne, "max_abs": float((x.float() - y.float()).abs().max())}


def accepted(T, g, cumulative):
    """cumulative[k] = P(accepted >= k + 2), decreasing."""
    u = torch.rand(N, generator=g, device="cuda")
    acc = torch.ones(N, dtype=torch.int32, device="cuda")
    for r in cumulative[:T - 1]:
        acc += (u < r).int()
    return acc


def exactness(ops, f, T, sigmoid, seed, steps=6):
    g = torch.Generator(device="cuda").manual_seed(seed)
    p = Params(g)
    idx = (1 + torch.arange(N, device="cuda", dtype=torch.int32)[:, None] * T
           + torch.arange(T, device="cuda", dtype=torch.int32)[None]).contiguous()
    ref = torch.zeros(1 + N * T, HV, D, D, device="cuda").bfloat16()
    full = torch.full((N,), T, dtype=torch.int32, device="cuda")
    out = torch.empty(N * T, HV, D, device="cuda").bfloat16()
    for _ in range(64):  # build realistic states with the served op
        served(ops, Inputs(T, g), p, idx, full, ref, out, sigmoid)
    up = ref.clone()  # upstream rebuild (mode 0) runs on its own copy
    dstate = torch.zeros(1 + N, HV, D, D, device="cuda").bfloat16()
    dstate[1:] = ref[idx[:, T - 1].long()]
    slots = (1 + torch.arange(N, device="cuda", dtype=torch.int32)).reshape(N, 1).contiguous()
    b0, b1 = bufs_pair()
    prev_acc = full
    worst = {"out_served_vs_deferred": 0, "commit_vs_served_slot": 0, "out_served_vs_upstream": 0,
             "state_served_vs_upstream": 0}
    for s in range(steps):
        x = Inputs(T, g)
        src = ref[idx[torch.arange(N, device="cuda"), (prev_acc - 1).long()].long()].clone()
        o_ref = torch.empty_like(out); o_up = torch.empty_like(out); o_def = torch.empty_like(out)
        served(ops, x, p, idx, prev_acc, ref, o_ref, sigmoid)
        mine(f, 0, x, p, idx, prev_acc, up, o_up, sigmoid)
        cur, prev = (b0, b1) if s % 2 == 0 else (b1, b0)
        counts = prev_acc if s > 0 else torch.zeros(N, dtype=torch.int32, device="cuda")
        mine(f, 1, x, p, slots, counts, dstate, o_def, sigmoid, bufs=(*prev, *cur))
        torch.cuda.synchronize()
        d1 = diff(o_ref, o_def)["neq"]; d2 = diff(dstate[1:], src)["neq"]
        d3 = diff(o_ref, o_up)["neq"]; d4 = diff(ref, up)["neq"]
        worst["out_served_vs_deferred"] = max(worst["out_served_vs_deferred"], d1)
        worst["commit_vs_served_slot"] = max(worst["commit_vs_served_slot"], d2)
        worst["out_served_vs_upstream"] = max(worst["out_served_vs_upstream"], d3)
        worst["state_served_vs_upstream"] = max(worst["state_served_vs_upstream"], d4)
        if s == steps - 1:
            worst["last_max_abs_out"] = diff(o_ref, o_def)["max_abs"]
            worst["last_max_abs_commit"] = diff(dstate[1:], src)["max_abs"]
        prev_acc = accepted(T, g, (0.8, 0.55, 0.35))
    worst["state_rms"] = float(ref.float().pow(2).mean().sqrt())
    return worst


def timing(ops, f, T, sigmoid, rates, reps=20):
    g = torch.Generator(device="cuda").manual_seed(7)
    p = Params(g)
    x = Inputs(T, g)
    out = torch.empty(N * T, HV, D, device="cuda").bfloat16()
    acc = accepted(T, g, rates)
    res = {"T": T, "mean_accepted": float(acc.float().mean())}
    idx = (1 + torch.arange(N, device="cuda", dtype=torch.int32)[:, None] * T
           + torch.arange(T, device="cuda", dtype=torch.int32)[None]).contiguous()

    def run(fn, label):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        gr.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            gr.replay()
        e1.record(); torch.cuda.synchronize()
        res[label + "_ms_per_step"] = e0.elapsed_time(e1) / reps

    states = [torch.randn(1 + N * T, HV, D, D, device="cuda").mul_(0.05).bfloat16() for _ in range(LAYERS)]
    run(lambda: [served(ops, x, p, idx, acc, st, out, sigmoid) for st in states], "served")
    run(lambda: [mine(f, 0, x, p, idx, acc, st, out, sigmoid) for st in states], "upstream")
    del states; torch.cuda.empty_cache()
    dst = [torch.randn(1 + N, HV, D, D, device="cuda").mul_(0.05).bfloat16() for _ in range(LAYERS)]
    slots = (1 + torch.arange(N, device="cuda", dtype=torch.int32)).reshape(N, 1).contiguous()
    bufs = [bufs_pair() for _ in range(LAYERS)]
    run(lambda: [mine(f, 1, x, p, slots, acc, st, out, sigmoid, bufs=(*bb[1], *bb[0])) for st, bb in zip(dst, bufs)],
        "deferred")
    zero = torch.zeros(N, dtype=torch.int32, device="cuda")
    run(lambda: [mine(f, 1, x, p, slots, zero, st, out, sigmoid, bufs=(*bb[1], *bb[0])) for st, bb in zip(dst, bufs)],
        "deferred_noreplay")
    del dst, bufs; torch.cuda.empty_cache()
    res["saving_ms"] = res["served_ms_per_step"] - res["deferred_ms_per_step"]
    return res


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="bench/gdn_m1.cu")
    ARGS = ap.parse_args()
    from vllm import _custom_ops as ops
    phase("start", device=torch.cuda.get_device_name(), free_gib=torch.cuda.mem_get_info()[0] / 2**30,
          nvcc=os.environ.get("CUDACXX"), has_op=hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"))
    libs = {}
    for fast in (False, True):
        try:
            libs[fast] = build(ARGS.src, fast)
            phase("build", fast=fast, ok=True)
        except Exception as e:  # noqa: BLE001
            phase("build", fast=fast, ok=False, error=repr(e)[:1500])
    for fast, f in libs.items():
        for T in (2, 4):
            for sigmoid in (True, False):
                for seed in (1, 2):
                    try:
                        phase("exact", fast=fast, T=T, sigmoid=sigmoid, seed=seed,
                              **exactness(ops, f, T, sigmoid, seed))
                    except Exception as e:  # noqa: BLE001
                        phase("exact", fast=fast, T=T, sigmoid=sigmoid, seed=seed, error=traceback.format_exc()[-1500:])
    f = libs.get(False) or libs.get(True)
    for T, rates in ((2, (0.88,)), (4, (0.78, 0.60, 0.47))):
        try:
            phase("timing", **timing(ops, f, T, True, rates))
        except Exception as e:  # noqa: BLE001
            phase("timing", T=T, error=traceback.format_exc()[-1500:])
    phase("done")


if __name__ == "__main__":
    main()
