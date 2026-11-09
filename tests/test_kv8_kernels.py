"""8-bit QSA KV cache kernels on the CPU (docs/validation-20260927/kv-cache-8bit.md).

Needs torch and Triton; the kernel checks run under the Triton interpreter (``TRITON_INTERPRET=1``, set here
before Triton is imported). The project environment has neither on macOS, so these tests skip there; they
were run in a Linux container (torch 2.14 CPU, Triton 3.8): see the doc. The GPU side (the pinned writer, fp8
codes, timing) is ``python -m vllm.arc_kv8 bench`` in the kv8 smoke.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[1]


def load_runtime():
    spec = importlib.util.spec_from_file_location("arc_kv8_under_test", ROOT / "src/arc_infra/kv8_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kv8 = load_runtime()


def rel(x, y):
    return float((x - y).norm() / y.norm())


@pytest.mark.parametrize("mode", ["int8", "fp8"])
def test_quantize_round_trip_accuracy(mode):
    g = torch.Generator().manual_seed(1)
    x = torch.randn((512, 2, 256), generator=g)
    x[..., :4] *= 4
    x = x.to(torch.bfloat16)
    codes, scale = kv8.quantize_reference(x, mode)
    back = kv8.dequantize_reference(codes, scale)
    error = rel(back, x.float())
    # int8: step amax/127 per row, so 4x outlier channels cost ~1.3%; e4m3: 3 mantissa bits (~2.5%).
    assert error < (0.02 if mode == "int8" else 0.04), error
    assert scale.dtype == torch.float32 and scale.shape == x.shape[:-1]
    amax = x.float().abs().amax(-1)
    peak = back.abs().amax(-1)
    assert torch.allclose(peak, amax, rtol=0.01 if mode == "int8" else 0.07)


def test_pack_layout_matches_the_pinned_per_token_head_page():
    q, k, v, *_ = kv8.synthetic_case(2, page=16, blocks=4, topk=19)
    packed = kv8.pack_cache(k, v, "int8")
    # Logical (blocks, kv_heads, page, 2 x (256 + 4)) with NHD strides, as the pinned get_kv_cache_shape.
    assert packed.shape == kv8.kv_cache_shape(4, 16, 2, 256, "int8_per_token_head") == (4, 2, 16, 520)
    assert kv8.kv_cache_shape(4, 16, 2, 256, "auto") == (4, 2, 16, 512)
    assert packed.stride() == (16 * 2 * 520, 520, 2 * 520, 1)
    codes_k, codes_v, sk, sv = kv8.cache_views(packed, 256, False)
    ref_k, ref_ks = kv8.quantize_reference(k, "int8")
    ref_v, ref_vs = kv8.quantize_reference(v, "int8")
    assert torch.equal(codes_k, ref_k) and torch.equal(codes_v, ref_v)
    assert torch.equal(sk, ref_ks) and torch.equal(sv, ref_vs)
    assert codes_k.shape == (4, 16, 2, 256) and sk.shape == (4, 16, 2)
    # The views alias the cache: a write through the scale view lands inside the page bytes.
    sk[1, 3, 1] = 2.5
    raw = packed.transpose(1, 2).contiguous().view(torch.uint8)
    assert torch.equal(raw[1, 3, 1, 256:260].view(torch.float32), torch.tensor([2.5]))
    fp8 = kv8.pack_cache(k, v, "fp8")
    assert fp8.dtype == torch.uint8
    codes_k8, _, sk8, _ = kv8.cache_views(fp8, 256, True)
    assert codes_k8.dtype == torch.float8_e4m3fn
    assert rel(kv8.dequantize_reference(codes_k8, sk8), k.float()) < 0.04


def test_views_are_kept_per_holder_for_the_current_tensor_only():
    _, k, v, *_ = kv8.synthetic_case(1, page=16, blocks=2, topk=3)
    holder = type("Impl", (), {})()
    first = kv8.pack_cache(k, v, "int8")
    views = kv8.cache_views(first, 256, False, holder)
    assert kv8.cache_views(first, 256, False, holder) is views
    second = kv8.pack_cache(k, v, "int8")
    assert kv8.cache_views(second, 256, False, holder) is not views
    assert holder._arc_kv8_views[0][0] == second.data_ptr()
    with pytest.raises(ValueError):
        kv8.cache_views(first, 128, False)


def run_case(rows, mode, *, page, blocks, topk, requests=None, fill=0.8, profile=None, seed=0):
    q, k, v, indices, table, t2r = kv8.synthetic_case(rows, page=page, blocks=blocks, topk=topk, requests=requests,
                                                      fill=fill, seed=seed)
    packed = kv8.pack_cache(k, v, mode)
    codes_k, codes_v, sk, sv = kv8.cache_views(packed, 256, mode == "fp8")
    out = kv8.sparse_attention(q, codes_k, codes_v, sk, sv, indices, table, t2r, profile=profile)
    exact = kv8.attention_reference(q, k, v, indices, table, t2r)
    dequant = kv8.attention_reference(q, kv8.dequantize_reference(codes_k, sk), kv8.dequantize_reference(codes_v, sv),
                                      indices, table, t2r)
    return out.float(), exact, dequant


@pytest.mark.parametrize("rows,splits", [(1, None), (3, 1)])
def test_interpreted_kernel_matches_the_dequantized_reference(rows, splits):
    # rows=1: the pinned decode profile (BLOCK_N 16, split-K 64 -> clipped to the tiles, merge kernel);
    # rows=3 with one split: the direct-store branch.
    profile = None
    if splits is not None:
        profile = {**kv8.launch_profile(rows, 2, 12, 37), "splits": 1}
    out, exact, dequant = run_case(rows, "int8", page=16, blocks=8, topk=37, profile=profile)
    assert kv8.launch_profile(1, 2, 12, 37)["splits"] > 1
    # Against dequantize-then-attend in FP32, only BF16 rounding (Q, p * v_scale, the output) remains.
    assert rel(out, dequant) < 0.01, rel(out, dequant)
    # Against the BF16 cache in FP32, int8 per-token-head adds well under a percent.
    assert rel(out, exact) < 0.015, rel(out, exact)


def test_interpreted_kernel_masks_padding_invalid_pages_and_foreign_requests():
    q, k, v, indices, table, t2r = kv8.synthetic_case(2, page=16, blocks=8, topk=21, requests=2, fill=0.5)
    packed = kv8.pack_cache(k, v, "int8")
    codes_k, codes_v, sk, sv = kv8.cache_views(packed, 256, False)
    # Poison every byte the kernel must not read: unselected tokens of other requests' pages get NaN scales.
    other = table[1].long()
    sk[other] = float("nan")
    sv[other] = float("nan")
    indices[0, 5] = 10_000  # beyond the block table: masked
    table[0, -1] = -1  # an unallocated page in request 0's table
    t2r_bad = t2r.clone()
    t2r_bad[1] = 7  # an out-of-range request: the row produces zeros
    out = kv8.sparse_attention(q, codes_k, codes_v, sk, sv, indices, table, t2r_bad)
    assert torch.isfinite(out.float()).all()
    assert out[1].float().abs().max() == 0
    ref = kv8.attention_reference(q[:1], kv8.dequantize_reference(codes_k, sk.nan_to_num()),
                                  kv8.dequantize_reference(codes_v, sv.nan_to_num()), indices[:1], table, t2r[:1])
    assert rel(out[:1].float(), ref) < 0.01


def test_interpreted_kernel_fp8_codes():
    out, exact, dequant = run_case(1, "fp8", page=16, blocks=4, topk=19)
    assert rel(out, dequant) < 0.01
    assert rel(out, exact) < 0.05


def test_microbench_glue_with_the_gpu_pieces_stubbed(monkeypatch, tmp_path):
    """``bench()`` end to end on the CPU: the pinned BF16 kernel, the pinned writer and CUDA timing are stubbed
    (the dry run caught an int padding value that cannot be cast to e4m3 in Triton, compiled or interpreted)."""
    import types

    def pinned(q, k, v, idx, table, t2r, out):
        out.copy_(kv8.attention_reference(q, k, v, idx, table, t2r).to(out.dtype))

    def writer(key, value, kc, vc, ks, vs, slots, kv_quant_mode):
        mode = "fp8" if kc.dtype == torch.float8_e4m3fn else "int8"
        page = kc.shape[1]
        for x, codes_view, scale_view in ((key, kc, ks), (value, vc, vs)):
            codes, scale = kv8.quantize_reference(x, mode)
            for i, slot in enumerate(slots.tolist()):
                codes_view[slot // page, slot % page] = codes[i]
                scale_view[slot // page, slot % page] = scale[i]

    modules = {"vllm.models.qwen3_8_flash_next.nvidia.ops.qsa": {"qsa_sparse_paged_attention": pinned},
               "vllm.v1.attention.ops.triton_reshape_and_cache_flash": {
                   "triton_reshape_and_cache_flash_per_token_head_quant": writer},
               "vllm.v1.kv_cache_interface": {"KVQuantMode": types.SimpleNamespace(INT8_PER_TOKEN_HEAD=2,
                                                                                   FP8_PER_TOKEN_HEAD=3)}}
    for name in ("vllm", "vllm.models", "vllm.models.qwen3_8_flash_next", "vllm.models.qwen3_8_flash_next.nvidia",
                 "vllm.models.qwen3_8_flash_next.nvidia.ops", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.ops",
                 *modules):
        module = types.ModuleType(name)
        module.__dict__.update(modules.get(name, {}))
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(kv8, "_time", lambda fn, repeats=200: (fn(), 1.0)[1])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    case = kv8.synthetic_case
    monkeypatch.setattr(kv8, "synthetic_case", lambda *a, device=None, **k: case(*a, device="cpu", **k))
    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *a, device=None, **k: arange(*a, **k))
    report = kv8.bench(tmp_path / "bench.json", topks=(67,), rows_list=(1, 4), page=64, blocks=16)
    assert json.loads((tmp_path / "bench.json").read_text()) == report
    for row in report["rows"]:
        assert row["int8_writer_codes_equal"] == row["fp8_writer_codes_equal"] == 1.0
        assert row["int8_rel_error"] < 0.03 and row["fp8_rel_error"] < 0.06
        assert {"bf16_us", "int8_us", "fp8_us"} <= set(row)


def test_impl_init_args_only_touch_the_configured_dtype(monkeypatch):
    monkeypatch.setitem(kv8._STATE, "config", {"kv_cache_8bit": "int8"})
    args = (24, 256, 0.0625, 2, None, None, "int8_per_token_head", None, "decoder", None)
    new_args, kwargs, dtype = kv8.impl_init_args(args, {})
    assert dtype == "int8_per_token_head" and new_args[6] == "auto" and new_args[:6] == args[:6]
    assert kv8.impl_init_args(args[:6] + ("auto",) + args[7:], {})[2] is None
    assert kv8.impl_init_args(args[:6] + ("fp8_per_token_head",) + args[7:], {})[2] is None
    _, kwargs, dtype = kv8.impl_init_args((), {"kv_cache_dtype": "int8_per_token_head"})
    assert kwargs == {"kv_cache_dtype": "auto"} and dtype == "int8_per_token_head"
    assert kv8.allowed("int8_per_token_head") and not kv8.allowed("fp8_per_token_head")
    assert not kv8.allowed("auto") and not kv8.allowed(None)
    monkeypatch.setitem(kv8._STATE, "config", {})
    assert not kv8.allowed("int8_per_token_head") and kv8.configured_cache_dtype() is None
