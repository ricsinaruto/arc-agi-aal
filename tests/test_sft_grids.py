"""trunk/sft served-format fake-quant ("QAD grids") against the serving code, bit for bit on CPU.

The serving references are the v87b submission's modules, copied verbatim into tests/served_v87b/ (from
dist/arena-flash-competition-v87b-sub/src: arena/expert_3bit_math.py, arena/expert_quant_math.py,
arc_infra/kv4_runtime.py, arc_infra/kv8_runtime.py, arc_infra/dense_kernels_runtime.py). Their torch references import
with a stand-in for Triton where Triton is missing; the Triton kernels themselves (the kv4 cache writer, the INT8
dequant, the fused int3 grid) run under ``TRITON_INTERPRET=1`` where Triton is installed (modal_sft.py --tests).
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

if "triton" not in sys.modules:
    os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
SERVED = ROOT / "tests/served_v87b"
sys.path[:0] = [str(ROOT / "scripts/sft"), str(ROOT / "tests/golden"), str(ROOT / "src"), str(ROOT / "tests")]
import attention  # noqa: E402
import qat  # noqa: E402
import train  # noqa: E402
from test_sft_qat import cfg, setup  # noqa: E402,F401
from test_sft_train import opt  # noqa: E402

BF16 = torch.bfloat16
F = torch.nn.functional
SEARCH = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)  # arena.expert_3bit_overlay.SEARCH


def real_triton():
    try:
        import triton

        return hasattr(triton, "Config") and os.environ.get("TRITON_INTERPRET") == "1"
    except ImportError:
        return False


def _load(name, file, package):
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", SERVED / file)
    mod = importlib.util.module_from_spec(spec)
    pkg = sys.modules.get(package)
    if pkg is None:
        try:
            pkg = importlib.import_module(package)
        except ImportError:
            pkg = types.ModuleType(package)
            pkg.__path__ = []
            sys.modules[package] = pkg
    sys.modules[spec.name] = mod
    setattr(pkg, name, mod)
    spec.loader.exec_module(mod)
    return mod


def served():
    """The serving modules (one load per process): {"tm", "qm", "kv4", "dk"}."""
    if hasattr(served, "mods"):
        return served.mods
    stub = None
    try:
        import triton  # noqa: F401
    except ImportError:  # enough of Triton for the module-level decorators (the torch references never call it)
        tr, tl = types.ModuleType("triton"), types.ModuleType("triton.language")
        tr.jit = lambda f=None, **kw: f if f is not None else (lambda g: g)
        tr.cdiv = lambda a, b: -(-a // b)
        tl.constexpr = object
        tr.language = tl
        stub = {"triton": tr, "triton.language": tl}
        sys.modules.update(stub)
    try:
        qm = _load("expert_quant_math", "expert_quant_math.py", "arena")
        tm = _load("expert_3bit_math", "expert_3bit_math.py", "arena")
        _load("kv8_runtime", "kv8_runtime.py", "arc_infra")
        kv4 = _load("kv4_runtime", "kv4_runtime.py", "arc_infra")
        dk = _load("dense_kernels_runtime", "dense_kernels_runtime.py", "arc_infra")
    finally:
        for k in stub or ():
            sys.modules.pop(k, None)
    served.mods = {"tm": tm, "qm": qm, "kv4": kv4, "dk": dk}
    return served.mods


@pytest.fixture
def int3(monkeypatch):
    monkeypatch.setitem(qat.QAT, "grid", "int3served")
    monkeypatch.setitem(qat.QAT, "fused", None)
    monkeypatch.setitem(qat.QAT, "chunk", 2)


def experts(e, n, k, seed=0):
    """BF16 expert weights [e, n, k]: heavy-tailed rows, an outlier, an all-zero group, a group whose largest weight is
    negative and one whose largest magnitude ties between +a and -a."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(e, n, k, generator=g) * 0.02 * torch.rand(e, n, 1, generator=g).add(0.2)
    w[0, 1, :16] = 0
    w[0, 2, 7] = 0.4
    w[e - 1, 3, 16:32] = -w[e - 1, 3, 16:32].abs()
    w[e - 1, 4, 32], w[e - 1, 4, 33] = 0.05, -0.05
    return w.to(BF16)


def served_rtn(w):
    tm = served()["tm"]
    s2 = tm.scale2_of(w, "int3")
    q, sc = tm.rtn(w, s2, "int3", 16, SEARCH)
    return q, sc, s2


# --- 1. int3served routed experts ---------------------------------------------------------------------------------

def test_int3_grid_equals_served_rtn(int3):
    """scales, scale_2 and dequant bit-equal to expert_3bit_math.rtn(w, scale2_of(w, "int3"), "int3", 16, SEARCH), with
    experts chunked; the straight-through forward is the dequant in BF16, its gradient the identity."""
    w = experts(5, 24, 96)
    q, sc, s2 = served_rtn(w)
    mine = qat.scales_of(w)
    assert torch.equal(mine[0], s2) and torch.equal(mine[1].float(), sc) and mine[1].dtype == BF16
    got = qat.fake(w, mine)
    assert got.dtype == torch.float32 and torch.equal(got, q)
    assert torch.equal(qat.grid(w, mine), q.to(BF16))
    x = w.clone().requires_grad_()
    out = qat.ste(x, mine)
    assert torch.equal(out, q.to(BF16))
    gy = torch.randn(out.shape).to(BF16)
    out.backward(gy)
    assert torch.equal(x.grad, gy)
    assert set(torch.unique(qat.codes3(w, mine)).tolist()) <= set(qat.LEVELS3)


def test_int3_real_shapes_and_nvfp4_container(int3):
    """Real expert shapes (gate_up [E, 2 x 640, 2560], down [E, 2560, 640]): bit-equal to rtn; gate and up share one
    scale_2, computed on their concatenation (convert_shard's w13); to_nvfp4 takes the grid exactly and the served
    NVFP4 dequant gives it back."""
    mods = served()
    tm, qm = mods["tm"], mods["qm"]
    gate, up = experts(2, 640, 2560, seed=1), (experts(2, 640, 2560, seed=2).float() * 3).to(BF16)
    for w in (torch.cat([gate, up], 1), experts(2, 2560, 640, seed=3)):
        q, sc, s2 = served_rtn(w)
        mine = qat.scales_of(w)
        assert torch.equal(mine[0], s2) and torch.equal(mine[1].float(), sc)
        got = qat.fake(w, mine)
        assert torch.equal(got, q)
        packed, block = tm.to_nvfp4(got, mine[1].float(), mine[0], 16)  # raises unless exact
        assert torch.equal(qm.dequantize(packed, block, mine[0]), got)
    s2_13 = qat.scales_of(torch.cat([gate, up], 1))[0]
    assert torch.equal(s2_13, tm.scale2_of(torch.cat([gate, up], 1), "int3"))
    assert not torch.equal(s2_13, tm.scale2_of(gate, "int3"))


def test_int3_all_zero_expert_is_zero(int3):
    """An all-zero expert: rtn's scale_2 is 0 and its dequant NaN; the trainer's grid is 0 (scale_2 1)."""
    w = experts(2, 8, 64)
    w[1] = 0
    qat.QAT["zero_experts"] = 0
    s2, sc = qat.scales_of(w)
    assert qat.QAT["zero_experts"] == 1  # counted; train.py logs it at load and at each refresh
    assert s2[1] == 1 and torch.equal(qat.fake(w, (s2, sc))[1], torch.zeros(8, 64))
    q, _, _ = served_rtn(w[:1])
    assert torch.equal(qat.fake(w, (s2, sc))[:1], q)


@pytest.mark.skipif(not real_triton(), reason="Triton interpreter (modal_sft.py --tests)")
def test_int3_triton_grid_interpreted(int3):
    """The fused int3 grid (Triton, interpreted on CPU) equals the eager grid bit for bit, run in fp32 (the
    interpreter casts fp32 -> BF16 by truncation, the GPU to nearest even; on the GPU qat.fuse keeps the kernel only
    after the BF16 check on the run's own weights)."""
    w = experts(3, 16, 2560 // 8)
    s = qat.scales_of(w)
    assert torch.equal(qat._triton_grid3()(w.float(), s), qat.fake(w, s))


def test_int3_trains_and_refreshes(setup, int3):
    """The tiny student with the int3served grid: QAT forward, finite nonzero expert grads; the same loss with the
    grid's weights written in; refresh(drift) counts no flips on unchanged weights and some after a nudge."""
    import mtp

    tmp, _ = setup
    model, head = train.load_model(tmp / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    ex = train.load_example(tmp / "data/train/e0.pt")
    qat.install(model)
    try:
        c = cfg(tmp, tmp / "x3")
        got = train.example_loss(model, head, box, ex, c)
        mods = qat.routed(model)
        for mod in mods:
            for n in qat.NAMES:
                grad = getattr(mod, n).grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
                assert isinstance(qat.SCALES[id(mod)][n], tuple)
        assert qat.refresh(model, drift=True)["code_flip_share"] == 0
        kept = mods[0].down_proj.detach().clone()
        with torch.no_grad():
            mods[0].down_proj.mul_(1.01)
        moved = qat.refresh(model, drift=True)
        assert moved["code_flip_share"] > 0 and moved["scale_changed_share"] > 0
        with torch.no_grad():
            mods[0].down_proj.copy_(kept)
        qat.refresh(model)
        qat.QAT["on"] = False
        with torch.no_grad():
            for mod in mods:
                for n in qat.NAMES:
                    getattr(mod, n).copy_(qat.fake(getattr(mod, n), qat.SCALES[id(mod)][n]).to(BF16))
        assert train.example_loss(model, head, box, ex, c, train=False)[0] == got[0]
    finally:
        qat.QAT["on"] = False
        qat.SCALES.clear()


# --- 2. served INT8 dense families ----------------------------------------------------------------------------------

def test_int8_rows_equal_quantize_rows():
    """qat.int8_rows == dense_kernels_runtime.quantize_rows -> (codes x scale) in BF16 (its dequant kernel), with a
    zero row, an outlier and row chunks that split differently."""
    dk = served()["dk"]
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(300, 96, generator=g) * torch.rand(300, 1, generator=g)).to(BF16)
    w[5] = 0
    w[7, 3] = 3.0
    q, s = dk.quantize_rows(w, chunk_rows=128)
    want = (q.float() * s[:, None]).to(BF16)
    assert torch.equal(qat.int8_rows(w, chunk_rows=64), want) and torch.equal(train.int8_rows(w), want)
    if real_triton():  # the served large-M dequant kernel itself, interpreted: the interpreter's fp32 -> BF16 cast
        # truncates (the GPU's rounds to nearest even), so its output is the truncated codes x scale
        prod = q.float() * s[:, None]
        assert torch.equal(dk.dequantize(q, s), (prod.view(torch.int32) & -65536).view(torch.float32).to(BF16))


def test_dense_int8_train_forward_and_ste(setup):
    """dense_int8_train on the tiny student: every served family wrapped (lm_head through example_loss), each wrapped
    forward == F.linear on the INT8 round trip; the loss and every gradient equal those of the model with the INT8
    weights written in (straight-through: dL/dW = dL/dW_int8); the MTP draft keeps the BF16 lm_head."""
    import mtp

    tmp, _ = setup
    model, head = train.load_model(tmp / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    ex = train.load_example(tmp / "data/train/e0.pt")
    c = cfg(tmp, tmp / "x8")
    try:
        names = qat.install_dense(model, head)
        types_ = model.config.text_config.layer_types
        trunk = 3 * types_.count("linear_attention") + 4 * (len(types_) - types_.count("linear_attention"))
        mtp_names = sorted(n for n in names if n.startswith("mtp."))
        assert len(names) == trunk + 4
        assert mtp_names == [f"mtp.layers.0.self_attn.{p}_proj" for p in ("k", "o", "q", "v")]
        mods = {**dict(model.named_modules()), **{"mtp." + n: m for n, m in head.named_modules()}}
        assert all(mods[n]._sft_int8 for n in mtp_names) and not head.fc_hidden.__dict__.get("_sft_int8")
        for n in names:
            m = mods[n]
            xin = torch.randn(3, m.in_features).to(BF16)
            assert torch.equal(m(xin), F.linear(xin, qat.int8_rows(m.weight.detach()), m.bias))
        before = qat.DENSE["calls"]
        got = train.example_loss(model, head, box, ex, c)
        assert qat.DENSE["calls"] > before
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        qat.DENSE["on"] = False
        with torch.no_grad():
            for n, p in train.dense_int8_params(model):
                p.copy_(qat.int8_rows(p))
        ref = train.example_loss(model, head, box, ex, c)
        assert got[0] == ref[0]  # CE
        assert got[2] != ref[2]  # MTP KL: BF16 lm_head under dense_int8_train, the written-in INT8 one here
        ref_g = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert grads.keys() == ref_g.keys()
        bad = [n for n in grads if not torch.equal(grads[n], ref_g[n])]
        assert not bad, bad[:5]
    finally:
        qat.DENSE["on"] = False


def _gdn_stubs(monkeypatch, gdn):
    """Torch stand-ins for the GDN kernels (conv, fla, gated norm), deterministic, so that the packed layer's GEMMs and
    their weights are what is compared on CPU."""
    BF = torch.bfloat16

    def conv_fwd(y, prev, w, segs, T, B):
        return [(y[:, c0:c0 + nh * d].reshape(B, T, nh, d).contiguous(), None) for c0, nh, d, _ in segs]

    def conv_bwd(y, prev, w, segs, grads, saved, T, B, dx):
        n = B * T
        dx[:, :sum(nh * d for _, nh, d, _ in segs)] = torch.cat([g.reshape(n, -1) for g in grads], 1)
        return dx[:, :sum(nh * d for _, nh, d, _ in segs)]

    def fwd(q, k, v, g, beta, **kw):
        return g, (v.float() * beta.float().unsqueeze(-1)).to(BF), None, None, None, None

    def bwd(q, k, v, g, beta, A, do, g_input, A_log, dt_bias, **kw):
        dv = (do.float() * beta.float().unsqueeze(-1)).to(BF)
        db = (do.float() * v.float()).sum(-1).to(BF)
        return (torch.zeros_like(q), torch.zeros_like(k), dv, db, torch.zeros_like(g_input), None,
                torch.zeros_like(A_log), torch.zeros_like(dt_bias))

    def norm_fwd(o, z, w, eps):
        return (o.reshape(o.shape[0], -1).float() * torch.sigmoid(z.float())).to(BF)

    def norm_bwd(o, z, w, dcore, dz, eps):
        s = torch.sigmoid(z.float())
        dz.copy_((dcore.float() * o.reshape(o.shape[0], -1).float() * s * (1 - s)).to(BF))
        return (dcore.float() * s).to(BF)

    monkeypatch.setattr(gdn, "conv_fwd", conv_fwd)
    monkeypatch.setattr(gdn, "conv_bwd", conv_bwd)
    monkeypatch.setattr(gdn, "_fla", lambda: (fwd, bwd))
    monkeypatch.setattr(gdn, "norm_gate_fwd", norm_fwd)
    monkeypatch.setattr(gdn, "norm_gate_bwd", norm_bwd)
    monkeypatch.setattr(gdn, "norm_weight_grad", lambda o, z, d, eps: torch.zeros(o.shape[-1]))
    monkeypatch.setattr(gdn, "conv_weight_grad", lambda dpre, y, prev, B, T, C: torch.zeros(C, 1, 4))


def test_dense_int8_inside_packed_gdn(setup, monkeypatch):
    """The packed GDN fast path (gdn.GatedDeltaFull, kernels stood in on CPU) under dense_int8_train: its packed GEMM
    weights are the wrapped modules' INT8 weights row for row (in_proj_b / in_proj_a BF16), and its output, input grad
    and every weight grad equal the same path with the INT8 weights written in (straight-through); off, the path is
    unchanged. The GPU fast-vs-eager comparison with INT8 on is smoke.py's ``gdn_int8``."""
    if "triton" not in sys.modules:
        try:
            import triton  # noqa: F401
        except ImportError:
            from test_sft_speed import stub_triton

            stub_triton()
    import gdn

    _gdn_stubs(monkeypatch, gdn)
    tmp, _ = setup
    model, _ = train.load_model(tmp / "student", "cpu", fast_attention=False)
    mod = next(m for n, m in model.named_modules() if n.endswith("linear_attn"))
    x = (torch.randn(1, 9, model.config.text_config.hidden_size) * 0.5).to(BF16)

    def run():
        mod.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_()
        out = gdn.fast_forward(mod, xi)
        out.float().mul(torch.linspace(-1, 1, out.numel()).view(out.shape)).sum().backward()
        return out.detach(), xi.grad, {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}

    try:
        plain = run()
        names = qat.install_dense(model)
        assert all(getattr(mod, n)._sft_int8 for n in ("in_proj_qkv", "in_proj_z", "out_proj"))
        assert gdn.int8_rows_of(mod) == mod.in_proj_qkv.out_features + mod.in_proj_z.out_features
        w_in = gdn._packed(mod)
        gin, gout = gdn.served_weights(w_in, mod.out_proj.weight, gdn.int8_rows_of(mod))
        eager = torch.cat([qat.int8_rows(mod.in_proj_qkv.weight.detach()), qat.int8_rows(mod.in_proj_z.weight.detach()),
                           mod.in_proj_b.weight.detach(), mod.in_proj_a.weight.detach()])
        assert torch.equal(gin, eager) and torch.equal(gout, qat.int8_rows(mod.out_proj.weight.detach()))
        got = run()
        assert not torch.equal(got[0], plain[0])
        qat.DENSE["on"] = False
        assert gdn.int8_rows_of(mod) == 0
        with torch.no_grad():
            for n in ("in_proj_qkv", "in_proj_z", "out_proj"):
                getattr(mod, n).weight.copy_(qat.int8_rows(getattr(mod, n).weight))
        ref = run()
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
        assert got[2].keys() == ref[2].keys() and all(torch.equal(got[2][n], ref[2][n]) for n in ref[2])
        assert len(names) > 0
    finally:
        qat.DENSE["on"] = False


# --- 3. kv4 served cache --------------------------------------------------------------------------------------------

def kv_rows(t=300, heads=2, seed=0):
    """K / V [1, heads, t, 256] BF16: normed-key-like rows with outlier channels, and plain Gaussian values."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, heads, t, 256, generator=g)
    k[..., :4] *= 4
    k[..., 100] *= 9
    v = torch.randn(1, heads, t, 256, generator=g) * torch.rand(1, heads, t, 1, generator=g)
    return k.to(BF16), v.to(BF16)


def to_pages(x, page=4):
    """[1, H, T, 256] -> the cache's [blocks, page, H, 256]."""
    _, h, t, d = x.shape
    return x[0].transpose(0, 1).reshape(t // page, page, h, d)


def from_pages(x):
    b, p, h, d = x.shape
    return x.reshape(b * p, h, d).transpose(0, 1).unsqueeze(0)


def test_kv_fake_equals_kv4_cache_write_read():
    """attention.kv_fake("kv4") == kv4_runtime's int4 page written (the torch reference writer, pack_cache) and read
    back (dequantize_cache), bit for bit in fp32 and after the BF16 cast the attention takes; the zero point in the
    scale's low mantissa bits changes the values (so the check sees it)."""
    kv4 = served()["kv4"]
    k, v = kv_rows()
    cache = kv4.pack_cache(to_pages(k), to_pages(v), "int4")
    dk_, dv_ = kv4.dequantize_cache(cache, "int4")
    k32, v32 = attention.kv_fake(k.float(), v.float(), "kv4")
    assert torch.equal(k32, from_pages(dk_)) and torch.equal(v32, from_pages(dv_))
    kb, vb = attention.kv_fake(k, v, "kv4")
    assert torch.equal(kb, from_pages(dk_).to(BF16)) and torch.equal(vb, from_pages(dv_).to(BF16))
    rk = attention.rht(k.float()).to(BF16).float()  # the scale before its low mantissa bits hold the zero point
    lo, hi = rk.amin(-1, keepdim=True), rk.amax(-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp_min(1e-6)
    zp = attention._round_away(-lo / scale).clamp(0, 15)
    raw = (attention._round_away(rk * (1.0 / scale) + zp).clamp(0, 15) - zp) * scale
    assert not torch.equal(k32, attention.rht(raw, inverse=True) / 256)


@pytest.mark.skipif(not real_triton(), reason="Triton interpreter (modal_sft.py --tests)")
def test_kv_fake_equals_kv4_triton_writer_interpreted():
    """The served writer kernel itself (_kv4_write_kernel via kv4_runtime.write, interpreted): its page bytes equal the
    torch reference page's and its dequant equals kv_fake (the GPU's single_rht and FMA contraction are not run)."""
    kv4 = served()["kv4"]
    k, v = kv_rows(t=64)
    pk, pv = to_pages(k), to_pages(v)
    expected = kv4.pack_cache(pk, pv, "int4")
    written = torch.zeros_like(expected)
    slots = torch.arange(pk.shape[0] * pk.shape[1], dtype=torch.int64)
    kv4.write(pk.reshape(-1, 2, 256), pv.reshape(-1, 2, 256), *kv4.cache_views(written, "int4"), slots, "int4")
    assert torch.equal(written.view(torch.uint8), expected.view(torch.uint8))
    dk_, dv_ = kv4.dequantize_cache(written, "int4")
    kb, vb = attention.kv_fake(k, v, "kv4")
    assert torch.equal(kb, from_pages(dk_).to(BF16)) and torch.equal(vb, from_pages(dv_).to(BF16))


def test_kv_ste_straight_through():
    """kv_train's K / V: forward exactly kv_fake, gradients unchanged into K and V."""
    k, v = kv_rows(t=20)
    kk, vv = k.clone().requires_grad_(), v.clone().requires_grad_()
    ok, ov = attention.kv_ste(kk, vv, "kv4")
    ek, ev = attention.kv_fake(k, v, "kv4")
    assert torch.equal(ok, ek) and torch.equal(ov, ev)
    gk, gv = torch.randn(k.shape).to(BF16), torch.randn(v.shape).to(BF16)
    (ok.float() * gk.float()).sum().add((ov.float() * gv.float()).sum()).backward()
    assert torch.equal(kk.grad, gk) and torch.equal(vv.grad, gv)


# --- all three in a run ---------------------------------------------------------------------------------------------

def test_run_with_all_served_grids(setup):
    """A KL run with fake_quant on the int3served grid, dense_int8_train and kv_train="kv4" (selected attention):
    steps finite, each grid used in training; a run resumed from its step-2 checkpoint reproduces step 3; the defaults
    leave the flags off."""
    import json
    import shutil

    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    tmp, _ = setup
    out = tmp / "grids-run"
    try:
        with pytest.raises(ValueError, match="kv_train needs"):
            train.train(cfg(tmp, tmp / "nokv", loss="kl", kv_train="kv4"), device="cpu", make_opt=opt)
        attention.install(mode="selected")
        calls = (qat.QAT["calls"], qat.DENSE["calls"], attention.KVQ["train_calls"])
        flags = dict(loss="kl", fake_quant=True, qat_grid="int3served", dense_int8_train=True, kv_train="kv4")
        train.train(cfg(tmp, out, **flags), device="cpu", make_opt=opt)
        rows = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
        steps = {r["step"]: r for r in rows if r.get("event") == "step"}
        assert sorted(steps) == [1, 2, 3] and all(r["distill_kl"] > 0 and r["grad_norm"] > 0 for r in steps.values())
        assert qat.QAT["calls"] > calls[0] and qat.DENSE["calls"] > calls[1] and attention.KVQ["train_calls"] > calls[2]
        start = next(r for r in rows if r.get("event") == "start")
        assert start["load"]["dense_int8"] == 3 * 2 + 4 * 1 + 4 + 1  # trunk, the MTP's q / k / v / o, lm_head
        b = tmp / "grids-resume"
        b.mkdir()
        shutil.copytree(out / "ckpt-2", b / "ckpt-2")
        train.train(cfg(tmp, b, **flags), device="cpu", make_opt=opt)
        sb = [json.loads(x) for x in (b / "events.jsonl").read_text().splitlines()]
        sb = [r for r in sb if r.get("event") == "step"]
        assert [r["step"] for r in sb] == [3]
        assert abs(sb[0]["distill_kl"] - steps[3]["distill_kl"]) < 1e-6 * steps[3]["distill_kl"]
        c = train.Config(student="s", data="d", out="o")
        assert (c.qat_grid, c.dense_int8_train, c.kv_train) == ("int4", False, "")
    finally:
        for cls, flag in ((m.Qwen4ExpTextQSAIndexer, "_rl1_vectorized"), (m.Qwen4ExpTextAttention, "_rl1_selected")):
            if getattr(cls, flag, False):
                cls.forward = cls._rl1_original
                setattr(cls, flag, False)
        attention.SPARSE["mode"], attention.SEL["on"] = "masked", False
        attention.KVQ.update(mode=None, train=None)
        qat.QAT.update(on=False, grid="int4")
        qat.DENSE["on"] = False
        qat.SCALES.clear()
