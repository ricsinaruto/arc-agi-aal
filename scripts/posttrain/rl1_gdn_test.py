"""rl1_gdn's Triton kernels against the model's torch code on small shapes (CPU under TRITON_INTERPRET=1, or GPU):
conv + SiLU + L2 norm (with and without the cached prefix tokens) forward and backward, gated RMSNorm forward and
backward. Prints one JSON line per check and exits 1 on a failure."""
import json
import sys

import torch
import torch.nn.functional as F

import rl1_gdn as G

BF = torch.bfloat16


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cos(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def ulp_ok(a, b):
    """Within one BF16 rounding step of the reference (different fp32 accumulation order)."""
    a, b = a.float(), b.float()
    return bool(((a - b).abs() <= b.abs() * 2 ** -7 + 1e-3).all())


def ref_conv(y_qkv, prev, w, nk, nv, D):
    """The model's path: transpose, cat the conv state, depthwise conv (BF16), SiLU, split, FLA-style L2 norm."""
    B, T, C = y_qkv.shape
    x = y_qkv.transpose(1, 2)
    if prev is not None:
        x = torch.cat([prev, x], dim=-1)
    out = F.conv1d(x.float(), w.float(), None, padding=3, groups=C)[:, :, :x.shape[-1]].to(BF)  # fp32 acc (GPU)
    out = F.silu(out)[:, :, -T:].transpose(1, 2)
    kd = nk * D
    q, k, v = out.split([kd, kd, nv * D], dim=-1)

    def l2(t):
        t = t.reshape(B, T, -1, D).float()
        return (t / torch.sqrt((t * t).sum(-1, keepdim=True) + 1e-6)).to(BF)

    return l2(q), l2(k), v.reshape(B, T, nv, D)


def ref_norm(o, z, w, eps):
    h = o.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = w * h.to(BF)
    return (h * torch.sigmoid(z.float())).to(BF)


def keep_check(dev):
    """MFU-1 GDN["keep"]: fwd_keep / bwd_keep against fla's chunk_gated_delta_rule_fwd / _bwd, bitwise."""
    import os

    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd

    if os.environ.get("TRITON_INTERPRET") == "1":  # fla's fwd_h does not run under the interpreter (int.to)
        print(json.dumps({"check": "gdn_keep", "skipped": "interpreter"}), flush=True)
        return True
    if False:
        for name, mod in list(sys.modules.items()):
            if not name.startswith("fla."):
                continue
            for value in list(vars(mod).values()):
                at = value
                while at is not None and not hasattr(at, "configs"):
                    at = getattr(at, "fn", None)
                if at is not None and len(getattr(at, "configs", [])) > 1:
                    at.configs = at.configs[:1]
    ok = True
    for B, T, H, HV, K, with_h0 in ((1, 150, 2, 4, 64, False), (2, 64, 1, 2, 64, True)):
        g0 = torch.Generator(device=dev).manual_seed(T)
        r = lambda *shape, s=1.0: (torch.randn(*shape, generator=g0, device=dev) * s)  # noqa: E731
        q = torch.nn.functional.normalize(r(B, T, H, K), dim=-1).to(BF)
        k = torch.nn.functional.normalize(r(B, T, H, K), dim=-1).to(BF)
        v = r(B, T, HV, K, s=0.5).to(BF)
        ga = r(B, T, HV).to(BF)
        beta = torch.sigmoid(r(B, T, HV)).to(BF)
        a_log, dt_bias = r(HV, s=0.3), r(HV, s=0.3)
        h0 = r(B, HV, K, K, s=0.1) if with_h0 else None
        do = r(B, T, HV, K, s=0.5).to(BF)
        scale = K ** -0.5
        g, o, A, fin, _, _ = chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=ga, beta=beta, scale=scale, initial_state=h0,
                                                        output_final_state=True, use_gate_in_kernel=True, A_log=a_log,
                                                        dt_bias=dt_bias)
        g2, o2, A2, fin2, kept = G.fwd_keep(q, k, v, ga, beta, scale, h0, True, a_log, dt_bias)
        want = chunk_gated_delta_rule_bwd(q=q, k=k, v=v, g=g, beta=beta, A=A, scale=scale, initial_state=h0, do=do,
                                          dht=None, use_gate_in_kernel=True, g_input=ga, A_log=a_log, dt_bias=dt_bias)
        got = G.bwd_keep(q, k, v, g2, beta, A2, scale, h0, do, ga, a_log, dt_bias, kept)
        fwd_eq = all(torch.equal(a, b) for a, b in ((g, g2), (o, o2), (A, A2), (fin, fin2)))
        names = ("dq", "dk", "dv", "db", "dg", "dh0", "dA_log", "ddt_bias")
        bwd_eq = {n: (a is None and b is None) or bool(torch.equal(a, b)) for n, a, b in zip(names, want, got)}
        row = {"check": "gdn_keep", "B": B, "T": T, "h0": with_h0, "fwd_bitwise": fwd_eq, "bwd_bitwise": bwd_eq}
        row["ok"] = fwd_eq and all(bwd_eq.values())
        print(json.dumps(row), flush=True)
        ok &= row["ok"]
    return ok


def keep_plumbing_check(dev):
    """GDN["keep"] through GatedDeltaFast with fla's calls stubbed (deterministic torch fakes): the kept path runs when
    an input needs a grad (mfu1-gkeep-b3: is_grad_enabled() is False inside Function.forward, so it never did), not
    in a no-grad pass, and gives the same out / grads as the unkept path."""
    import types

    def fake_fwd(q, k, v, g, beta, scale, h0, want_final, A_log, dt_bias):
        g_out = g.float().cumsum(1)
        o = v.float() * torch.sigmoid(g_out)[..., None] + q.float().sum(-1, keepdim=True).repeat_interleave(
            v.shape[2] // q.shape[2], dim=2) * scale
        A = beta.float()[..., None].expand(*beta.shape, 64).contiguous()
        final = torch.zeros(v.shape[0], v.shape[2], q.shape[-1], v.shape[-1], device=v.device) if want_final else None
        return g_out, o, A, final

    def fake_bwd(q, k, v, g, beta, A, do):
        dv = (do.float() * torch.sigmoid(g)[..., None]).to(v.dtype)
        dq = do.float().sum(-1, keepdim=True)[:, :, ::v.shape[2] // q.shape[2]].expand_as(q).to(q.dtype) * 0.1
        return dq, (k * 0.01), dv, (beta * 0.02), (g * 0.03).to(BF), None, None, None

    calls = {"fwd_keep": 0, "bwd_keep": 0}
    orig = (G.fwd_keep, G.bwd_keep, G._fla)

    def fwd_keep(q, k, v, g, beta, scale, h0, want_final, A_log, dt_bias):
        calls["fwd_keep"] += 1
        g_out, o, A, final = fake_fwd(q, k, v, g, beta, scale, h0, want_final, A_log, dt_bias)
        return g_out, o, A, final, (k * 1, g_out * 1, v * 1)

    def bwd_keep(q, k, v, g, beta, A, scale, h0, do, g_input, A_log, dt_bias, kept):
        calls["bwd_keep"] += 1
        assert len(kept) == 3
        return fake_bwd(q, k, v, g, beta, A, do)

    def fla():
        def fwd(q, k, v, g, beta, scale, initial_state, output_final_state, use_gate_in_kernel, A_log, dt_bias):
            g_out, o, A, final = fake_fwd(q, k, v, g, beta, scale, initial_state, output_final_state, A_log, dt_bias)
            return g_out, o, A, final, None, None

        def bwd(q, k, v, g, beta, A, scale, initial_state, do, dht, use_gate_in_kernel, g_input, A_log, dt_bias):
            return fake_bwd(q, k, v, g, beta, A, do)
        return fwd, bwd

    G.fwd_keep, G.bwd_keep, G._fla = fwd_keep, bwd_keep, fla
    try:
        torch.manual_seed(3)
        B, T, H, nk, nv, D, r = 1, 40, 64, 2, 4, 128, 4
        kd, vd = nk * D, nv * D
        C = 2 * kd + vd
        widths = (C, vd, nv, nv)
        x = (torch.randn(B, T, H, device=dev) * 0.5).to(BF)
        w_in = (torch.randn(C + vd + 2 * nv, H, device=dev) * 0.05).to(BF)
        w_out = (torch.randn(H, vd, device=dev) * 0.05).to(BF)
        conv_w = (torch.randn(C, 1, 4, device=dev) * 0.5).to(BF)
        a_log, dt_bias, norm_w = torch.randn(nv, device=dev), torch.randn(nv, device=dev), torch.ones(D, device=dev)
        lora = []
        for w in widths:
            lora += [torch.randn(r, H, device=dev) * 0.1, torch.randn(w, r, device=dev) * 0.1]
        lora += [torch.randn(r, vd, device=dev) * 0.1, torch.randn(H, r, device=dev) * 0.1]
        meta = (nk, nv, D, D, 1e-6, 2.0, False)

        def go(keep):
            G.GDN["keep"] = keep
            xs = x.clone().requires_grad_(True)
            ls = [t.clone().requires_grad_(True) for t in lora]
            out, _, _ = G.GatedDeltaFast.apply(xs, w_in, w_out, conv_w, a_log, dt_bias, norm_w, None, None, meta, *ls)
            out.float().pow(2).sum().backward()
            with torch.no_grad():
                G.GatedDeltaFast.apply(x, w_in, w_out, conv_w, a_log, dt_bias, norm_w, None, None, meta, *lora)
            return out.detach(), xs.grad, [t.grad for t in ls]

        base = go(False)
        n0 = dict(calls)
        kept = go(True)
        same = torch.equal(base[0], kept[0]) and torch.equal(base[1], kept[1]) and all(
            torch.equal(a, b) for a, b in zip(base[2], kept[2]))
        row = {"check": "gdn_keep_plumbing", "calls_off": n0, "calls_on": dict(calls), "same": same}
        row["ok"] = (n0 == {"fwd_keep": 0, "bwd_keep": 0} and calls == {"fwd_keep": 1, "bwd_keep": 1} and same)
    finally:
        G.fwd_keep, G.bwd_keep, G._fla = orig
        G.GDN["keep"] = False
    print(json.dumps(row), flush=True)
    return row["ok"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ok = True
    nk, nv, D = 2, 6, 128
    C = 2 * nk * D + nv * D
    for B, T, with_prev in ((1, 50, False), (2, 70, True), (1, 3, True)):
        extra = 200
        y = torch.randn(B * T, C + extra, device=dev).to(BF)
        prev4 = torch.randn(B, C, 4, device=dev).to(BF) if with_prev else None
        w = (torch.randn(C, 1, 4, device=dev) * 0.5).to(BF)
        segs = ((0, nk, D, True), (nk * D, nk, D, True), (2 * nk * D, nv, D, False))
        prev3 = prev4[..., -3:].contiguous() if with_prev else None
        (q, rq), (k, rk), (v, _) = G.conv_fwd(y, prev3, w, segs, T, B)
        yq = y[:, :C].reshape(B, T, C).clone().requires_grad_(True)
        rq_, rk_, rv_ = ref_conv(yq, prev4, w, nk, nv, D)
        row = {"check": "conv_fwd", "B": B, "T": T, "prev": with_prev,
               "q_maxdiff": float((q.float() - rq_.float()).abs().max()),
               "k_maxdiff": float((k.float() - rk_.float()).abs().max()),
               "v_maxdiff": float((v.float() - rv_.float()).abs().max())}
        row["rel"] = max(rel(q, rq_), rel(k, rk_), rel(v, rv_))
        row["ok"] = (ulp_ok(q, rq_) and ulp_ok(k, rk_) and ulp_ok(v, rv_) and row["rel"] < 4e-3) if dev == "cuda" else (
            row["rel"] < 1.5e-2 and min(cos(q, rq_), cos(k, rk_), cos(v, rv_)) > 0.9999)  # interpreter: truncating casts
        print(json.dumps(row), flush=True)
        ok &= row["ok"]
        gq, gk, gv = (torch.randn_like(t, dtype=torch.float32).to(BF) for t in (q, k, v))
        (rq_.float() * gq.float()).sum().add((rk_.float() * gk.float()).sum()).add(
            (rv_.float() * gv.float()).sum()).backward()
        dY = torch.zeros(B * T, C + extra, device=dev, dtype=BF)
        G.conv_bwd(y, prev3, w, segs, (gq, gk, gv), ((q, rq), (k, rk), (None, None)), T, B, dY)
        got, want = dY[:, :C].reshape(B, T, C), yq.grad
        row = {"check": "conv_bwd", "B": B, "T": T, "prev": with_prev, "rel": rel(got, want), "cos": cos(got, want),
               "untouched_zero": bool((dY[:, C:] == 0).all())}
        row["ok"] = row["rel"] < (4e-3 if dev == "cuda" else 1.5e-2) and row["cos"] > 0.9999 and row["untouched_zero"]
        print(json.dumps(row), flush=True)
        ok &= row["ok"]
    for N in (37, 64):
        nh = 6
        o = torch.randn(N, nh, D, device=dev).to(BF)
        z = torch.randn(N, nh * D + 50, device=dev).to(BF)
        w = (1 + 0.1 * torch.randn(D, device=dev)).to(BF)
        out = G.norm_gate_fwd(o, z[:, :nh * D], w, 1e-6)
        oo = o.clone().requires_grad_(True)
        zz = z[:, :nh * D].reshape(N, nh, D).clone().requires_grad_(True)
        ref = ref_norm(oo, zz, w, 1e-6)
        row = {"check": "norm_fwd", "N": N, "maxdiff": float((out.view(N, nh, D).float() - ref.float()).abs().max())}
        row["rel"] = rel(out.view(N, nh, D), ref)
        row["ok"] = (ulp_ok(out.view(N, nh, D), ref) and row["rel"] < 4e-3) if dev == "cuda" else (
            row["rel"] < 1.5e-2 and cos(out.view(N, nh, D), ref) > 0.9999)
        print(json.dumps(row), flush=True)
        ok &= row["ok"]
        g = torch.randn(N, nh * D, device=dev).to(BF)
        (ref.float() * g.view(N, nh, D).float()).sum().backward()
        dz = torch.zeros(N, nh * D + 50, device=dev, dtype=BF)
        do = G.norm_gate_bwd(o, z[:, :nh * D], w, g, dz[:, :nh * D], 1e-6)
        row = {"check": "norm_bwd", "N": N, "do_rel": rel(do, oo.grad), "do_cos": cos(do, oo.grad),
               "dz_rel": rel(dz[:, :nh * D].view(N, nh, D), zz.grad), "dz_cos": cos(dz[:, :nh * D].view(N, nh, D),
                                                                                       zz.grad)}
        bar = 4e-3 if dev == "cuda" else 1.5e-2
        row["ok"] = row["do_rel"] < bar and row["dz_rel"] < bar and min(row["do_cos"], row["dz_cos"]) > 0.9999
        print(json.dumps(row), flush=True)
        ok &= row["ok"]
    ok &= keep_plumbing_check(dev)
    ok &= keep_check(dev)
    print(json.dumps({"tests": "pass" if ok else "FAIL"}), flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
