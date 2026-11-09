"""rl1_hyper against the model's Qwen4ExpTextGatedResidual + injection (CPU under TRITON_INTERPRET=1, or GPU): forward
values (bitwise share, max BF16 ulps), backward error of both paths against a float64 run of the model's code.
Prints one JSON line per check; exit 1 on a failure. usage: rl1_hyper_test.py [N H R]"""
import json
import sys
from types import SimpleNamespace

import rl1_hyper as K
import torch
import triton
import triton.language as tl

BF = torch.bfloat16


def within_ulp(a, b):
    """Share of elements within one BF16 step of the larger magnitude (different fp32 accumulation order)."""
    a, b = a.float(), b.float()
    return float(((a - b).abs() <= torch.maximum(a.abs(), b.abs()) * 2 ** -7 + 1e-30).float().mean())


def rel(a, ref):
    a, ref = a.double(), ref.double()
    return float((a - ref).norm() / ref.norm().clamp_min(1e-300))


def main():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    n, H, R = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (96, 320, 40)
    G = 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    conf = SimpleNamespace(hc_count=G, hidden_size=H, rms_norm_eps=1e-6, hc_lowrank=R)
    gr = m.Qwen4ExpTextGatedResidual(conf).to(dev)
    with torch.no_grad():  # trained-like scales: unit-ish norm weights, mix weights ~ 1/sqrt(fan-in)
        gr.hc_norm.weight.normal_(0, 0.3)
        gr.input_mix_weight_down.weight.normal_(0, (G * H) ** -0.5 * 4)
        gr.input_mix_weight_up.weight.normal_(0, R ** -0.5 * 2)
        gr.block_inject_weight.weight.normal_(0, (G * H) ** -0.5 * 4)
    gr = gr.to(BF).requires_grad_(False)
    proj = torch.randn(H, H, device=dev, dtype=torch.float64) * H ** -0.5  # the block between: y = tanh(mixed) P
    x0 = (torch.randn(n, G * H, device=dev, dtype=torch.float64) * torch.tensor([1.0, 3.0, 0.5, 8.0], device=dev,
                                                                               dtype=torch.float64).repeat_interleave(H))
    probe = torch.randn(n, G * H, device=dev, dtype=torch.float64)
    failures = []

    def model_path(x, module, p):
        mixed, hyper, w = module(x)
        y = torch.tanh(mixed) @ p
        injection = y.unsqueeze(-2) * w.unsqueeze(-1)
        return hyper + injection.flatten(-2), mixed, w, y

    def run(path, dtype):
        x = x0.to(dtype).clone().requires_grad_(True)
        if path == "fused":
            K.HYPER["on"] = True
            mixed, w = K.gated_residual(gr, x)
            y = torch.tanh(mixed) @ proj.to(BF)
            out = K.inject(x, y, w)
        else:
            module = gr if dtype == BF else _double(gr)
            out, mixed, w, y = model_path(x, module, proj.to(dtype))
        (out.double() * probe).sum().backward()
        return {"out": out.detach(), "mixed": mixed.detach(), "w": w.detach(), "dx": x.grad.detach()}

    fused, model, ref = run("fused", BF), run("model", BF), run("model", torch.float64)
    for key in ("mixed", "w", "out"):
        row = {"check": f"forward {key}", "bitwise_share": float((fused[key] == model[key]).float().mean()),
               "within_1ulp_share": within_ulp(fused[key], model[key])}
        row["ok"] = row["within_1ulp_share"] >= 0.999
        print(json.dumps(row), flush=True)
        failures += [] if row["ok"] else [row["check"]]
    rf, rm = rel(fused["dx"], ref["dx"]), rel(model["dx"], ref["dx"])
    row = {"check": "backward dx", "rel_fused_vs_fp64": rf, "rel_model_vs_fp64": rm,
           "cos_fused_model": float(torch.nn.functional.cosine_similarity(fused["dx"].double().flatten(),
                                                                         model["dx"].double().flatten(), dim=0))}
    row["ok"] = rf <= max(1.25 * rm, 2e-3)
    print(json.dumps(row), flush=True)
    failures += [] if row["ok"] else [row["check"]]
    print(json.dumps({"check": "summary", "ok": not failures, "failures": failures, "n": n, "H": H, "R": R,
                      "device": dev}), flush=True)
    sys.exit(1 if failures else 0)


def _double(module):
    import copy

    return copy.deepcopy(module).double()


def bench(n=16384, H=2560, R=320, reps=5):
    """Real shapes on the GPU: GatedResidual + a stand-in block + injection, forward + backward, model vs fused."""
    import time

    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    G = 4
    conf = SimpleNamespace(hc_count=G, hidden_size=H, rms_norm_eps=1e-6, hc_lowrank=R)
    gr = m.Qwen4ExpTextGatedResidual(conf).cuda()
    with torch.no_grad():
        gr.hc_norm.weight.normal_(0, 0.3)
        for lin, fan in ((gr.input_mix_weight_down, G * H), (gr.input_mix_weight_up, R), (gr.block_inject_weight, G * H)):
            lin.weight.normal_(0, fan ** -0.5)
    gr = gr.to(BF).requires_grad_(False)
    x0 = torch.randn(n, G * H, device="cuda", dtype=BF)
    y0 = torch.randn(n, H, device="cuda", dtype=BF)
    rows = {}
    for path in ("model", "fused", "model", "fused"):
        K.HYPER["on"] = path == "fused"
        times = {"fwd": [], "bwd": []}
        for _ in range(reps):
            x = x0.clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=BF):
                if path == "fused":
                    mixed, w = K.gated_residual(gr, x)
                    out = K.inject(x, mixed * y0, w)
                else:
                    mixed, hyper, w = gr(x)
                    out = hyper + ((mixed * y0).unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)
            torch.cuda.synchronize()
            t1 = time.time()
            out.backward(x0)
            torch.cuda.synchronize()
            times["fwd"].append(t1 - t0)
            times["bwd"].append(time.time() - t1)
        rows[path] = {k: round(1000 * sorted(v)[len(v) // 2], 3) for k, v in times.items()}
    print(json.dumps({"check": "bench", "n": n, "ms": rows,
                      "speedup_fwd_bwd": round((rows["model"]["fwd"] + rows["model"]["bwd"])
                                               / (rows["fused"]["fwd"] + rows["fused"]["bwd"]), 2)}), flush=True)


def exact_train(n=2048, H=2560, R=320):
    """exact() with trainable BF16 hyper weights (PTH's heal_kd: fully_shard BF16 compute copies) and the model's
    [1, T, G*H] stream: mixed, w, out, dx and the four weight grads bitwise, fused vs the model's code."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    G = 4
    torch.manual_seed(0)
    conf = SimpleNamespace(hc_count=G, hidden_size=H, rms_norm_eps=1e-6, hc_lowrank=R)
    gr = m.Qwen4ExpTextGatedResidual(conf).cuda()
    with torch.no_grad():
        gr.hc_norm.weight.normal_(0, 0.3)
        for lin, fan in ((gr.input_mix_weight_down, G * H), (gr.input_mix_weight_up, R), (gr.block_inject_weight, G * H)):
            lin.weight.normal_(0, fan ** -0.5 * 2)
    gr = gr.to(BF).requires_grad_(True)
    proj = (torch.randn(H, H, device="cuda") * H ** -0.5).to(BF)
    scales = torch.tensor([1.0, 3.0, 0.5, 8.0], device="cuda").repeat_interleave(H)
    x0 = (torch.randn(n, G * H, device="cuda") * scales).unsqueeze(0)
    probe = torch.randn(1, n, G * H, device="cuda")
    names = ["hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight"]
    failures = []
    for dtype in (torch.float32, BF):
        def run(fused, dtype=dtype):
            K.HYPER["on"] = fused
            gr.zero_grad(set_to_none=True)
            x = x0.to(dtype).clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=BF):
                if fused:
                    mixed, w = K.gated_residual(gr, x)
                    y = torch.nn.functional.linear(mixed, proj)
                    out = K.inject(x, y, w)
                else:
                    mixed, hyper, w = gr(x)
                    y = torch.nn.functional.linear(mixed, proj)
                    out = hyper + (y.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)
            (out.float() * probe).sum().backward()
            params = dict(gr.named_parameters())
            return {"mixed": mixed.detach(), "w": w.detach(), "out": out.detach(), "dx": x.grad.detach(),
                    **{k: params[k].grad.detach().clone() for k in names}}

        a, b = run(True), run(False)
        K.HYPER["xb"] = True
        c = run(True)
        K.HYPER["xb"] = False
        row = {"check": "exact train", "dtype": str(dtype)[6:], "n": n, "stream_out_dtype": str(b["out"].dtype)[6:]}
        for key in a:
            row[key] = float((a[key] == b[key]).float().mean()) if a[key].shape == b[key].shape else "shape"
            row[f"{key}_xb"] = float((c[key] == b[key]).float().mean()) if c[key].shape == b[key].shape else "shape"
        row["ok_xb"] = all(row[f"{k}_xb"] == 1.0 for k in a)
        row["grad_dtypes"] = [str(a[k].dtype)[6:] + "/" + str(b[k].dtype)[6:] for k in names]
        row["ok"] = all(row[k] == 1.0 for k in a) and row["ok_xb"]
        print(json.dumps(row), flush=True)
        failures += [] if row["ok"] else [row["dtype"]]
    return failures


def exact(n=2048, H=2560, R=320):
    """Real shapes on the GPU under BF16 autocast, for a BF16 and an fp32 stream: GatedResidual + a Linear stand-in
    block + injection, fused vs the model's code: bitwise shares of mixed, w, out and dx (all must be 1)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    G = 4
    torch.manual_seed(0)
    conf = SimpleNamespace(hc_count=G, hidden_size=H, rms_norm_eps=1e-6, hc_lowrank=R)
    gr = m.Qwen4ExpTextGatedResidual(conf).cuda()
    with torch.no_grad():
        gr.hc_norm.weight.normal_(0, 0.3)
        for lin, fan in ((gr.input_mix_weight_down, G * H), (gr.input_mix_weight_up, R), (gr.block_inject_weight, G * H)):
            lin.weight.normal_(0, fan ** -0.5 * 2)
    gr = gr.to(BF).requires_grad_(False)
    proj = (torch.randn(H, H, device="cuda") * H ** -0.5).to(BF)
    scales = torch.tensor([1.0, 3.0, 0.5, 8.0], device="cuda").repeat_interleave(H)
    x0 = torch.randn(n, G * H, device="cuda") * scales
    probe = torch.randn(n, G * H, device="cuda")
    failures = []
    for dtype in (torch.float32, BF):
        def run(fused, dtype=dtype):
            K.HYPER["on"] = fused
            x = x0.to(dtype).clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=BF):
                if fused:
                    mixed, w = K.gated_residual(gr, x)
                    y = torch.nn.functional.linear(mixed, proj)
                    out = K.inject(x, y, w)
                else:
                    mixed, hyper, w = gr(x)
                    y = torch.nn.functional.linear(mixed, proj)
                    out = hyper + (y.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)
            (out.float() * probe).sum().backward()
            return {"mixed": mixed.detach(), "w": w.detach(), "out": out.detach(), "dx": x.grad.detach()}

        a, b = run(True), run(False)
        K.HYPER["xb"], calls = True, K.HYPER.get("xb_calls", 0)
        c = run(True)  # MFU-1 24 Oct: xn's BF16 cast written by the norm kernel (fp32 stream only)
        K.HYPER["xb"] = False
        row = {"check": "exact", "dtype": str(dtype)[6:], "n": n, "xb_calls": K.HYPER.get("xb_calls", 0) - calls}
        for key in a:
            row[key] = float((a[key] == b[key]).float().mean())
            row[f"{key}_xb"] = float((c[key] == b[key]).float().mean())
            row[f"{key}_dtype"] = str(a[key].dtype)[6:] + "/" + str(b[key].dtype)[6:]
        row["dx_rel"] = rel(a["dx"], b["dx"])
        row["ok"] = all(row[k] == 1.0 and row[f"{k}_xb"] == 1.0 for k in a)
        print(json.dumps(row), flush=True)
        failures += [] if row["ok"] else [row["dtype"]]
    return failures


def debug_norm(n=2048, H=2560):
    """Which reduce order torch's x.pow(2).mean(-1) over H uses (fp32 rows): emulations vs torch, and the kernel's
    rsqrt vs torch.rsqrt on torch's own mean."""
    G = 4
    torch.manual_seed(0)
    x = (torch.randn(n, G, H, device="cuda") * torch.rand(n, G, 1, device="cuda") * 3)
    p = x.pow(2)
    ref = p.mean(-1)
    tot_ref = p.sum(-1)

    def emulate(vec, lanes, up):
        t = p.view(n, G, H // (lanes * vec), lanes, vec)
        acc = t[:, :, 0].clone()
        for j in range(1, t.shape[2]):
            acc = acc + t[:, :, j]
        v = acc[..., 0].clone()
        for i in range(1, vec):
            v = v + acc[..., i]
        if up:  # shfl_down offsets 1, 2, 4, ...: adjacent pairs
            while v.shape[-1] > 1:
                v = v[..., 0::2] + v[..., 1::2]
        else:  # offsets 16, 8, ...: halves
            while v.shape[-1] > 1:
                h = v.shape[-1] // 2
                v = v[..., :h] + v[..., h:]
        return v[..., 0]

    row = {"check": "debug norm"}
    for vec in (1, 2, 4, 8, 16):
        for lanes in (32, 16, 64, 128):
            if H % (vec * lanes):
                continue
            for up in (True, False):
                e = emulate(vec, lanes, up)
                row[f"v{vec}l{lanes}{'u' if up else 'd'}"] = round(float((e == tot_ref).float().mean()), 4)
    row["mean_is_sum_x_inv"] = float(((tot_ref * (1.0 / H)) == ref).float().mean())
    row["mean_is_sum_div"] = float(((tot_ref / H) == ref).float().mean())
    best = max((k for k in row if k.startswith("v")), key=lambda k: row[k])
    row["best"] = best
    print(json.dumps(row), flush=True)


def debug_bwd(n=2048, H=2560, R=320):
    """GatedResidual backward stage by stage (fp32 stream under autocast): the eager grad of xn vs the fused path's
    composition, and dx from the eager dxn by the fused formula (torch ops) vs eager, and the fused kernel's dx."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    G = 4
    torch.manual_seed(0)
    conf = SimpleNamespace(hc_count=G, hidden_size=H, rms_norm_eps=1e-6, hc_lowrank=R)
    gr = m.Qwen4ExpTextGatedResidual(conf).cuda()
    with torch.no_grad():
        gr.hc_norm.weight.normal_(0, 0.3)
        for lin, fan in ((gr.input_mix_weight_down, G * H), (gr.input_mix_weight_up, R), (gr.block_inject_weight, G * H)):
            lin.weight.normal_(0, fan ** -0.5 * 2)
    gr = gr.to(BF).requires_grad_(False)
    x0 = torch.randn(n, G * H, device="cuda") * torch.tensor([1.0, 3.0, 0.5, 8.0], device="cuda").repeat_interleave(H)
    dmix = torch.randn(n, H, device="cuda")
    dinj = torch.randn(n, G, device="cuda").to(BF)
    eq = lambda a, b: round(float((a == b).float().mean()), 5)  # noqa: E731
    x = x0.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=BF):
        xn = gr.hc_norm(x)
        xn.retain_grad()
        t = gr.input_mix_weight_down(xn) / G
        a = torch.nn.functional.silu(t)
        u = gr.input_mix_weight_up(a)
        S = torch.sigmoid(u)
        mixed = (S.unflatten(-1, (G, H)) * xn.unflatten(-1, (G, H))).mean(dim=-2)
        w = 2 * torch.sigmoid(gr.block_inject_weight(xn) / G)
    torch.autograd.backward((mixed, w), (dmix, dinj))
    row = {"check": "debug bwd"}
    # the fused composition of dxn from (dmix, dinj), in torch
    aten = torch.ops.aten
    with torch.no_grad():
        dp = dmix * (1.0 / G)
        du = aten.sigmoid_backward((dp.unsqueeze(-2) * xn.unflatten(-1, (G, H))).to(BF).flatten(-2), S)
        gd = aten.silu_backward(du @ gr.input_mix_weight_up.weight, t) / G @ gr.input_mix_weight_down.weight
        gi = aten.sigmoid_backward(dinj * 2, w / 2) / G @ gr.block_inject_weight.weight
        gm = (dp.unsqueeze(-2) * S.unflatten(-1, (G, H))).flatten(-2)
        orders = {"(i+m)+d": (gi.float() + gm) + gd.float(), "(d+m)+i": (gd.float() + gm) + gi.float(),
                  "(m+i)+d": (gm + gi.float()) + gd.float(), "(m+d)+i": (gm + gd.float()) + gi.float(),
                  "(i+d)+m": (gi.float() + gd.float()) + gm}
        for k, v in orders.items():
            row[f"dxn {k}"] = eq(v, xn.grad)
        dxn = xn.grad
        xg = x.detach().view(n, G, H)
        r = torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + 1e-6)
        dn = (dxn * (1.0 + gr.hc_norm.weight.float())).view(n, G, H)
        dr = (dn * xg).sum(-1, keepdim=True)
        for name, dv in (("(-0.5g)r3", (-0.5 * dr) * r.pow(3)), ("g r3 -0.5", dr * r.pow(3) * -0.5),
                         ("(-0.5g)(rr)r", (-0.5 * dr) * ((r * r) * r))):
            dx = dn * r + (dv / H) * (2.0 * xg)
            row[f"dx {name}"] = eq(dx.view(n, -1), x.grad)
            row[f"dx {name} b"] = eq((dn * r + (dv / H).expand(n, G, H) * (2.0 * xg)).view(n, -1), x.grad)
        du_k = torch.empty_like(u)
        K._mix_bwd[(n, 3)](dmix.contiguous(), u, xn.contiguous(), du_k, H, G=G, BH=1024, LIB=K.SIGMOID_LIB, num_warps=4,
                           **K.NOFMA)
        row["kernel du"] = eq(du_k, du)
        ds = (dp.unsqueeze(-2) * xn.unflatten(-1, (G, H))).to(BF).flatten(-2).float()
        sf = S.float()
        for name, v in (("(a(1-b))b", (ds * (1 - sf)) * sf), ("a((1-b)b)", ds * ((1 - sf) * sf)),
                        ("(ab)(1-b)", (ds * sf) * (1 - sf)), ("a-ab then b", (ds - ds * sf) * sf),
                        ("(a bf(1-b))b", (ds * (1 - sf).to(BF).float()) * sf),
                        ("bf(a(1-b))b", (ds * (1 - sf)).to(BF).float() * sf),
                        ("bf(a bf(1-b))b", (ds * (1 - sf).to(BF).float()).to(BF).float() * sf),
                        ("a bf((1-b)b)", ds * ((1 - sf) * sf).to(BF).float()),
                        ("a bf(bf(1-b)b)", ds * ((1 - sf).to(BF).float() * sf).to(BF).float())):
            row[f"du {name}"] = eq(v.to(BF), du)
        dx_k = torch.empty_like(x0)
        rs = r.view(n, G).contiguous()
        K._norm_bwd[(n, G)](dmix.contiguous(), u, gi.contiguous(), gd.contiguous(), x0, gr.hc_norm.weight, rs, dx_k, H,
                            1.0 / H, G=G, NJ=H // (32 * K.VEC), V=K.VEC, LIB=K.SIGMOID_LIB, ROUND=False, num_warps=1,
                            **K.NOFMA)
        row["kernel dx (torch dr order?)"] = eq(dx_k, x.grad)
        row["kernel dx rel"] = rel(dx_k, x.grad)
    xf = x0.clone().requires_grad_(True)
    K.HYPER["on"] = True
    with torch.autocast("cuda", dtype=BF):
        mf, wf = K.gated_residual(gr, xf)
    torch.autograd.backward((mf, wf), (dmix, dinj))
    row["fused dx"] = eq(xf.grad, x.grad)
    print(json.dumps(row), flush=True)


def debug_inject_bwd(n=16384, H=2560):
    """InjectFn backward vs eager (fp32 stream): dy and dw bitwise shares for both reduce widths."""
    G = 4
    torch.manual_seed(0)
    x = torch.randn(n, G * H, device="cuda", requires_grad=True)
    y = torch.randn(n, H, device="cuda").to(BF).requires_grad_(True)
    w = (torch.rand(n, G, device="cuda") * 2).to(BF).requires_grad_(True)
    d = torch.randn(n, G * H, device="cuda")
    (x + (y.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)).backward(d)
    row = {"check": "debug inject bwd", "n": n}
    for wv in (4, 8):
        K.WVEC = wv
        y2, w2 = y.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
        K.inject(x.detach(), y2, w2).backward(d)
        row[f"dy wvec{wv}"] = float((y2.grad == y.grad).float().mean())
        row[f"dw wvec{wv}"] = float((w2.grad == w.grad).float().mean())
    print(json.dumps(row), flush=True)


def debug_rows(n=1024, H=2560):
    """_norm_fwd_rows vs _norm_fwd (per (row, stream) program): rs and xn bitwise shares per (rows, warps)."""
    G = 4
    torch.manual_seed(0)
    x = torch.randn(n, G * H, device="cuda") * 3
    nw = (torch.randn(G * H, device="cuda") * 0.3).to(BF)
    xn0, rs0 = torch.empty_like(x), torch.empty(n, G, device="cuda")
    K._norm_fwd[(n, G)](x, nw, xn0, rs0, H, 1e-6, 1.0 / H, G=G, NJ=H // 128, V=4, num_warps=1, **K.NOFMA)
    row = {"check": "debug rows"}
    for R, w in ((1, 1), (1, 4), (2, 1), (4, 4), (8, 4)):
        xn, rs = torch.empty_like(x), torch.empty(n, G, device="cuda")
        K._norm_fwd_rows[(triton.cdiv(n * G, R),)](x, nw, xn, rs, xn, n * G, H, 1e-6, 1.0 / H, G=G, NJ=H // 128, V=4,
                                                   R=R, CAST=False, num_warps=w, **K.NOFMA)
        row[f"R{R}w{w}"] = [float((rs == rs0).float().mean()), float((xn == xn0).float().mean())]
    print(json.dumps(row), flush=True)


@triton.jit
def _sumsq_a(X, OUT, H, NJ: tl.constexpr, V: tl.constexpr, SQ: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    off = tl.arange(0, 32)[:, None] * V + tl.arange(0, V)[None, :]
    acc = tl.zeros([32, V], dtype=tl.float32)
    for j in range(NJ):
        x = tl.load(X + i * H + j * 32 * V + off)
        acc += x * x if SQ else x
    tl.store(OUT + i, K._tree(acc, V))


@triton.jit
def _sumsq_b(X, OUT, H, NJ: tl.constexpr, SQ: tl.constexpr):
    """explicit slot accumulators (strided loads), no where-sums"""
    i = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, 32) * 4
    a0 = tl.zeros([32], dtype=tl.float32)
    a1, a2, a3 = a0, a0, a0
    for j in range(NJ):
        p = X + i * H + j * 128 + lane
        x0, x1, x2, x3 = tl.load(p), tl.load(p + 1), tl.load(p + 2), tl.load(p + 3)
        if SQ:
            x0, x1, x2, x3 = x0 * x0, x1 * x1, x2 * x2, x3 * x3
        a0 += x0
        a1 += x1
        a2 += x2
        a3 += x3
    v = ((a0 + a1) + a2) + a3
    a, b = tl.split(tl.permute(tl.reshape(v, [2, 16]), (1, 0)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [2, 8]), (1, 0)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [2, 4]), (1, 0)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [2, 2]), (1, 0)))
    a, b = tl.split(tl.permute(tl.reshape(a + b, [2, 1]), (1, 0)))
    tl.store(OUT + i, tl.sum(a + b, axis=0))


def debug_sum(n=8192, H=2560):
    torch.manual_seed(0)
    x = torch.randn(n, H, device="cuda") * 3
    row = {"check": "debug sum", "gpu": torch.cuda.get_device_name()}
    for sq in (False, True):
        ref = (x * x if sq else x).sum(-1)
        for name, fn in (("a", lambda o, sq=sq: _sumsq_a[(n,)](x, o, H, NJ=H // 128, V=4, SQ=sq, num_warps=1, **K.NOFMA)),
                         ("a_fma", lambda o, sq=sq: _sumsq_a[(n,)](x, o, H, NJ=H // 128, V=4, SQ=sq, num_warps=1)),
                         ("b", lambda o, sq=sq: _sumsq_b[(n,)](x, o, H, NJ=H // 128, SQ=sq, num_warps=1, **K.NOFMA)),
                         ("b_w4", lambda o, sq=sq: _sumsq_b[(n,)](x, o, H, NJ=H // 128, SQ=sq, num_warps=4, **K.NOFMA))):
            o = torch.empty(n, device="cuda")
            fn(o)
            row[f"{'sq' if sq else 'x'} {name}"] = float((o == ref).float().mean())
    print(json.dumps(row), flush=True)


def debug_inject():
    torch.manual_seed(0)
    n, G, H = 8, 4, 64
    x = torch.randn(n, G * H, device="cuda").to(BF)
    y = torch.randn(n, H, device="cuda").to(BF)
    w = (torch.rand(n, G, device="cuda") * 2).to(BF)
    ref = x + (y.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)
    out = K.inject(x, y, w)
    bad = (out != ref).nonzero()[:6].tolist()
    for r, c in bad:
        g, h = divmod(c, H)
        print(json.dumps({"r": r, "c": c, "x": float(x[r, c]), "y": float(y[r, h]), "w": float(w[r, g]),
                          "prod_bf16": float(y[r, h] * w[r, g]), "ref": float(ref[r, c]), "out": float(out[r, c])}))
    print(json.dumps({"check": "debug inject", "bitwise_share": float((out == ref).float().mean())}))


if __name__ == "__main__":
    while sys.argv[-1].startswith(("rows=", "warps=")):
        k, v = sys.argv.pop().split("=")
        K.TUNE[k] = int(v)
    if sys.argv[-1].startswith("wvec="):
        K.WVEC = int(sys.argv.pop()[5:])
    if sys.argv[-1].startswith("sig="):
        K.SIGMOID_LIB = sys.argv.pop() == "sig=lib"
    if sys.argv[1:2] == ["debug"]:
        debug_inject()
    elif sys.argv[1:2] == ["dbwd"]:
        debug_bwd(*(int(v) for v in sys.argv[2:3]))
    elif sys.argv[1:2] == ["dsum"]:
        debug_sum()
    elif sys.argv[1:2] == ["drows"]:
        debug_rows()
    elif sys.argv[1:2] == ["dinj"]:
        debug_inject_bwd(*(int(v) for v in sys.argv[2:3]))
    elif sys.argv[1:2] == ["dnorm"]:
        debug_norm()
    elif sys.argv[1:2] == ["exacttrain"]:
        sys.exit(1 if exact_train(*(int(v) for v in sys.argv[2:5])) else 0)
    elif sys.argv[1:2] == ["exact"]:
        sys.exit(1 if exact(*(int(v) for v in sys.argv[2:5])) else 0)
    elif sys.argv[1:2] == ["bench"]:
        bench(*(int(v) for v in sys.argv[2:3]))
    else:
        main()
