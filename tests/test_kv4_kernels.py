"""4-bit QSA KV cache kernels on the CPU (docs/validation-20260927/kv-cache-4bit.md).

As tests/test_kv8_kernels.py: torch and Triton, the kernels under the Triton interpreter (``TRITON_INTERPRET=1``, set
before Triton is imported); skipped where Triton is missing (macOS), run in a Linux container (see the doc).
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_runtime():
    spec = importlib.util.spec_from_file_location("arc_kv4_under_test", ROOT / "src/arc_infra/kv4_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kv4 = load_runtime()
kv8 = kv4._kv8
MODES = ("int4", "k8v4")


def rel(x, y):
    return float((x - y).norm() / y.norm())


def kv_like(rows=512, seed=1, outliers=4.0):
    """Unit-RMS rows with four 4x outlier channels (normed, RoPE'd keys), BF16."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, 2, 256), generator=g)
    x[..., :4] *= outliers
    return x.to(torch.bfloat16)


# ----------------------------------------------------------------------------- rotation and quantizer
def test_rht_is_the_pinned_single_rht_and_inverts():
    x = torch.randn((3, 2, 256), dtype=torch.float64)
    signs = kv4.rht_signs(256, "cpu")
    assert set(signs.unique().tolist()) == {-1.0, 1.0} and torch.equal(signs, kv4.rht_signs(256, "cpu"))
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 256:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert torch.allclose(kv4.hadamard_reference(x), x @ h.T)
    forward = kv4.rht_reference(x)
    assert torch.allclose(forward, (x * signs.double()) @ h.T)
    assert torch.allclose(kv4.rht_reference(forward, inverse=True) / 256, x)
    # Norm sqrt(256): the rotated dot product is 256 x the original one.
    q = torch.randn((256,), dtype=torch.float64)
    assert torch.allclose(kv4.rht_reference(q) @ forward[0, 0], 256 * (q @ x[0, 0]))


def test_quantizer_codes_zero_point_and_scale_bits():
    x = kv_like(64).float()
    packed, bits = kv4.quantize4_reference(x)
    assert packed.dtype == torch.uint8 and packed.shape == (64, 2, 128) and bits.shape == (64, 2)
    zp = bits & 0xF
    scale = (bits & -16).view(torch.float32)
    lo, hi = x.amin(-1), x.amax(-1)
    assert torch.allclose(scale, (hi - lo) / 15, rtol=1e-5)
    assert torch.equal(zp.float(), torch.round(-lo / scale).clamp(0, 15))
    back = kv4.dequantize4_reference(packed, bits)
    # Asymmetric 4-bit: every value within half a step (the clamp never bites: zp = round(-min / scale)).
    assert ((back - x).abs() <= scale.unsqueeze(-1) * 0.5 + 1e-5).all()
    assert (packed & 0xF).max() <= 15


@pytest.mark.parametrize("outliers", [1.0, 4.0])
def test_round_trip_errors_with_and_without_rotation(outliers):
    x = kv_like(2048, outliers=outliers)
    errors = {part: rel(kv4.round_trip(x, part), x.float()) for part in ("int8", "int4", "int4-norht")}
    # 4-bit asymmetric per (token, head): ~10-12% on Gaussian rows; the rotation spreads outlier channels.
    assert errors["int8"] < 0.02
    assert errors["int4"] < 0.13
    if outliers > 1:
        assert errors["int4"] < errors["int4-norht"]
    stats = kv4.round_trip_stats(x, x)
    assert abs(stats["k_int4_rht"] - errors["int4"]) < 1e-6 and abs(stats["k_int4_plain"] - errors["int4-norht"]) < 1e-6
    assert abs(stats["k_int8"] - errors["int8"]) < 1e-6 and stats["k_peak_to_rms"] > 2


# ----------------------------------------------------------------------------- page layout and writer
@pytest.mark.parametrize("mode", MODES)
def test_page_layout_and_views(mode, monkeypatch):
    monkeypatch.setitem(kv4._STATE, "config", {"kv_cache_4bit": mode})
    dtype = kv4.CACHE_DTYPES[mode]
    content = {"int4": 264, "k8v4": 392}[mode]
    assert kv4.content_bytes(mode) == content
    assert kv4.kv_cache_shape(4, 16, 2, 256, dtype) == (4, 2, 16, content)
    assert kv4.kv_cache_shape(4, 16, 2, 256, "auto") == (4, 2, 16, 512)
    with pytest.raises(NotImplementedError):
        kv4.kv_cache_shape(4, 16, 2, 256, "fp8_per_token_head")
    q, k, v, *_ = kv8.synthetic_case(2, page=16, blocks=4, topk=19)
    packed = kv4.pack_cache(k, v, mode)
    assert packed.shape == (4, 2, 16, content) and packed.stride() == (16 * 2 * content, content, 2 * content, 1)
    keys, values, k_scale, v_scale = kv4.cache_views(packed, mode)
    assert keys.shape == (4, 16, 2, 128 if mode == "int4" else 256) and values.shape == (4, 16, 2, 128)
    assert keys.dtype == (torch.uint8 if mode == "int4" else torch.int8) and values.dtype == torch.uint8
    assert k_scale.dtype == (torch.int32 if mode == "int4" else torch.float32) and v_scale.dtype == torch.int32
    vq, vb = kv4.quantize4_reference(kv4.rht_reference(v.float()).to(torch.bfloat16))
    assert torch.equal(values, vq) and torch.equal(v_scale, vb)
    # The views alias the page: the V scale sits right after the V codes, inside the page bytes.
    v_scale[1, 3, 1] = 12345
    raw = packed.view(torch.uint8).transpose(1, 2).contiguous()
    assert int(raw[1, 3, 1, content - 4:].view(torch.int32)) == 12345


@pytest.mark.parametrize("mode", MODES)
def test_interpreted_writer_matches_the_reference(mode):
    blocks, page = 4, 16
    q, k, v, *_ = kv8.synthetic_case(1, page=page, blocks=blocks, topk=7)
    expected = kv4.pack_cache(k, v, mode)
    written = torch.zeros_like(expected)
    views = kv4.cache_views(written, mode)
    slots = torch.randperm(blocks * page, generator=torch.Generator().manual_seed(3))
    tokens = slots.numel() - 5
    rows_k = k.reshape(-1, 2, 256)[slots[:tokens]]
    rows_v = v.reshape(-1, 2, 256)[slots[:tokens]]
    mapping = torch.cat([slots[:tokens], torch.full((3,), -1)]).to(torch.int64)
    rows_k = torch.cat([rows_k, rows_k[:3]])
    rows_v = torch.cat([rows_v, rows_v[:3]])
    kv4.write(rows_k, rows_v, *views, mapping, mode)
    got = written.view(torch.uint8).transpose(1, 2).reshape(blocks * page, 2, -1)
    want = expected.view(torch.uint8).transpose(1, 2).reshape(blocks * page, 2, -1)
    filled = slots[:tokens]
    # Codes: the same arithmetic in the interpreter (numpy) and torch; allow rare off-by-one ties.
    assert (got[filled] == want[filled]).float().mean() > 0.999
    k_off, ks_off, v_off, vs_off = kv4.offsets(mode)
    for a, b in ((ks_off, v_off), (vs_off, vs_off + 4)):
        assert torch.equal(got[filled][..., a:b], want[filled][..., a:b])
    # Slot -1 rows and unwritten slots stay zero.
    assert got[slots[tokens:]].abs().sum() == 0


# ----------------------------------------------------------------------------- reader
def run_case(rows, mode, *, page=16, blocks=8, topk=37, profile=None, seed=0):
    q, k, v, indices, table, t2r = kv8.synthetic_case(rows, page=page, blocks=blocks, topk=topk, seed=seed)
    packed = kv4.pack_cache(k, v, mode)
    views = kv4.cache_views(packed, mode)
    out = torch.zeros_like(q)
    kv4.attend(q, *views, indices, table, t2r, out, mode, profile)
    exact = kv8.attention_reference(q, k, v, indices, table, t2r)
    dk, dv = kv4.dequantize_cache(packed, mode)
    dequant = kv8.attention_reference(q, dk, dv, indices, table, t2r)
    return out.float(), exact, dequant


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows,splits", [(1, None), (3, 1)])
def test_interpreted_kernel_matches_dequantize_then_attend(mode, rows, splits):
    profile = None
    if splits is not None:
        profile = {**kv8.launch_profile(rows, 2, 12, 37), "splits": 1}
    out, exact, dequant = run_case(rows, mode, profile=profile)
    assert kv8.launch_profile(1, 2, 12, 37)["splits"] > 1
    # Against dequantize-then-attend in FP32 only BF16 rounding remains (rotated Q, p * v_scale, the output).
    assert rel(out, dequant) < 0.01, rel(out, dequant)
    # Against the BF16 cache: the 4-bit error itself (measured 15% int4, 11% k8v4 on Gaussian rows; int8 1.4%).
    assert rel(out, exact) < (0.18 if mode == "int4" else 0.13), rel(out, exact)


def test_interpreted_kernel_masks_padding_invalid_pages_and_foreign_requests():
    q, k, v, indices, table, t2r = kv8.synthetic_case(2, page=16, blocks=8, topk=21, requests=2, fill=0.5)
    packed = kv4.pack_cache(k, v, "int4")
    keys, values, k_scale, v_scale = kv4.cache_views(packed, "int4")
    nan_bits = torch.tensor([float("nan")]).view(torch.int32)
    other = table[1].long()
    clean_k, clean_v = k_scale.clone(), v_scale.clone()
    k_scale[other] = nan_bits
    v_scale[other] = nan_bits
    indices[0, 5] = 10_000
    table[0, -1] = -1
    t2r_bad = t2r.clone()
    t2r_bad[1] = 7
    out = torch.zeros_like(q)
    rotated = torch.zeros(q.shape, dtype=torch.float32)
    qr = (kv4.rht_reference(q.float()) / 256).to(torch.bfloat16)
    kv4.sparse_attention(qr, keys, values, k_scale, v_scale, indices, table, t2r_bad, rotated, 4)
    assert torch.isfinite(rotated).all() and rotated[1].abs().max() == 0
    out = (kv4.rht_reference(rotated, inverse=True) / 256)
    k_scale.copy_(clean_k)
    v_scale.copy_(clean_v)
    dk, dv = kv4.dequantize_cache(packed, "int4")
    ref = kv8.attention_reference(q[:1], dk, dv, indices[:1], table, t2r[:1])
    assert rel(out[:1], ref) < 0.01


def test_attention_errors_on_real_like_rows_rotation_helps_outliers():
    # One decode row over 512 tokens with 8x outlier key channels: the rotated int4 keys lose less.
    q, k, v, indices, table, t2r = kv8.synthetic_case(1, page=64, blocks=8, topk=509, fill=1.0, seed=5)
    k = k.float()
    k[..., :4] *= 2
    k = k.to(torch.bfloat16)
    exact = kv8.attention_reference(q, k, v, indices, table, t2r)
    errors = {}
    for mode in MODES:
        packed = kv4.pack_cache(k, v, mode)
        out = torch.zeros_like(q)
        kv4.attend(q, *kv4.cache_views(packed, mode), indices, table, t2r, out, mode)
        errors[mode] = rel(out.float(), exact)
    assert errors["k8v4"] < errors["int4"] < 0.2, errors


# ----------------------------------------------------------------------------- plumbing
def test_impl_init_args_allowed_and_spec(monkeypatch):
    import types

    monkeypatch.setitem(kv4._STATE, "config", {"kv_cache_4bit": "int4"})
    args = (24, 256, 0.0625, 2, None, None, "int4_per_token_head", None, "decoder", None)
    new_args, _, dtype = kv4.impl_init_args(args, {})
    assert dtype == "int4_per_token_head" and new_args[6] == "auto"
    assert kv4.impl_init_args(args[:6] + ("int8_per_token_head",) + args[7:], {})[2] is None
    assert kv4.allowed("int4_per_token_head") and not kv4.allowed("int8_per_token_head") and not kv4.allowed("auto")
    monkeypatch.setitem(kv4._STATE, "config", {"kv_cache_4bit": "k8v4"})
    assert kv4.allowed("int8_per_token_head") and not kv4.allowed("int4_per_token_head")
    monkeypatch.setitem(kv4._STATE, "config", {"kv_cache_8bit": "int8"})
    assert not kv4.allowed("int8_per_token_head") and kv4.configured_cache_dtype() is None

    class Mode:
        def __init__(self, name):
            self.name = name
            self.is_per_token_head = name != "NONE"

    modes = types.SimpleNamespace(INT4_PER_TOKEN_HEAD=Mode("INT4"), INT8_PER_TOKEN_HEAD=Mode("INT8"))
    module = types.ModuleType("vllm.v1.kv_cache_interface")
    module.KVQuantMode = modes
    for name in ("vllm", "vllm.v1"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", module)
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Spec:
        kv_quant_mode: object
        head_size: int = 256
        head_size_v: int = 256
        state_content_bytes: int | None = None

    for mode, vllm_mode, content in (("int4", modes.INT4_PER_TOKEN_HEAD, 264), ("k8v4", modes.INT8_PER_TOKEN_HEAD, 392)):
        monkeypatch.setitem(kv4._STATE, "config", {"kv_cache_4bit": mode})
        assert kv4.customize_spec(Spec(vllm_mode)).state_content_bytes == content
        other = modes.INT8_PER_TOKEN_HEAD if mode == "int4" else modes.INT4_PER_TOKEN_HEAD
        with pytest.raises(NotImplementedError):
            kv4.customize_spec(Spec(other))
    assert kv4.customize_spec(Spec(Mode("NONE"))).state_content_bytes is None


def test_prefill_steps_are_sampled_once_per_layer_and_logged(monkeypatch, capsys):
    monkeypatch.setitem(kv4._STATE, "config", {"kv_cache_4bit": "int4"})
    monkeypatch.setitem(kv4._STATE, "layers", {})
    monkeypatch.setitem(kv4._STATE, "samples", {})
    blocks, page = 2, 512
    k = kv_like(blocks * page)
    v = kv_like(blocks * page, seed=2)
    cache = torch.zeros((blocks, 2, page, 264), dtype=torch.uint8)
    impl = types_ns()
    for _ in range(3):
        kv4.cache_update(impl, k, v, cache, torch.arange(blocks * page))
    kv4.cache_update(impl, k[:8], v[:8], cache, torch.arange(8))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("ARC_KV4_ROUNDTRIP ")]
    assert len(lines) == kv4.SAMPLES_PER_LAYER
    assert '"k_int4_rht"' in lines[0] and '"rows": 1024' in lines[0]


def types_ns():
    import types

    return types.SimpleNamespace()


@pytest.mark.parametrize("mode", MODES)
def test_server_self_check_glue(mode, monkeypatch):
    import types

    def pinned_writer(key, value, key_cache, value_cache, slots, *, k_scale_cache, v_scale_cache):
        kv4.write(key, value, key_cache, value_cache, k_scale_cache.view(torch.int32), v_scale_cache.view(torch.int32),
                  slots, "int4")

    module = types.ModuleType("vllm.v1.attention.ops.int4_per_token_head")
    module.reshape_and_cache_int4 = pinned_writer
    for name in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = kv4.self_check("cpu", mode)
    assert result["writer_bytes_equal_reference"] > 0.999
    assert result.get("writer_bytes_equal_pinned", 1.0) == 1.0
    assert result["reader_rel_error_vs_dequant"] < 0.01 and result["reader_rel_error_vs_bf16"] < 0.2


# ----------------------------------------------------------------------------- the pinned int4 writer
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
PINNED_INT4 = "vllm/v1/attention/ops/int4_per_token_head.py"


def pinned_int4():
    """The pinned module's nibble packer, writer kernel, RHT signs and RHT, extracted by name (the module itself
    imports vLLM); its Hadamard is the pinned tier-3 butterfly (the CPU path)."""
    import ast

    import triton
    import triton.language as tl

    tree = ast.parse((PINNED_SITE / PINNED_INT4).read_text())
    wanted = ("pack_int4_nibbles", "_reshape_cache_int4_kernel", "_get_rht_signs", "single_rht")
    namespace = {"triton": triton, "tl": tl, "torch": torch, "_RHT_SIGNS_CACHE": {},
                 "fast_hadamard_transform": kv4.hadamard_reference}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), PINNED_INT4, "exec"), namespace)  # noqa: S102
    assert all(name in namespace for name in wanted)
    return namespace


def test_writer_matches_the_pinned_int4_writer_when_available():
    if not (PINNED_SITE / PINNED_INT4).is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    pinned = pinned_int4()
    x = kv_like(8, seed=4).float()
    assert torch.equal(pinned["_get_rht_signs"](256, 0, torch.device("cpu")), kv4.rht_signs(256, "cpu"))
    assert torch.allclose(pinned["single_rht"](x), kv4.rht_reference(x))
    assert torch.allclose(pinned["single_rht"](x, inverse=True), kv4.rht_reference(x, inverse=True))
    blocks, page = 2, 16
    q, k, v, *_ = kv8.synthetic_case(1, page=page, blocks=blocks, topk=7, seed=6)
    ours = torch.zeros((blocks, 2, page, 264), dtype=torch.uint8)
    theirs = torch.zeros_like(ours)
    slots = torch.arange(blocks * page, dtype=torch.int64)
    rows_k, rows_v = k.reshape(-1, 2, 256), v.reshape(-1, 2, 256)
    kv4.write(rows_k, rows_v, *kv4.cache_views(ours, "int4"), slots, "int4")
    keys, values, k_scale, v_scale = kv4.cache_views(theirs, "int4")
    rk = pinned["single_rht"](rows_k.float()).to(rows_k.dtype)
    rv = pinned["single_rht"](rows_v.float()).to(rows_v.dtype)
    pinned["_reshape_cache_int4_kernel"][(rk.shape[0], 2)](
        rk, rv, keys, values, k_scale.view(torch.float32), v_scale.view(torch.float32), slots,
        rk.stride(0), rk.stride(1), rv.stride(0), rv.stride(1),
        keys.stride(0), keys.stride(1), keys.stride(2), values.stride(0), values.stride(1), values.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), v_scale.stride(0), v_scale.stride(1),
        v_scale.stride(2), block_size=page, head_size=256, head_size_v=256, PACKED_HEAD_PADDED=128)
    page_ours = ours.transpose(1, 2)
    page_theirs = theirs.transpose(1, 2)
    assert (page_ours == page_theirs).float().mean() > 0.999
    assert torch.equal(page_ours[..., 128:132], page_theirs[..., 128:132])
    assert torch.equal(page_ours[..., 260:], page_theirs[..., 260:])
