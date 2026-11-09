"""Kernel work, phase 7: the QSA prefill microbenchmark and the kv8 prefill path (docs/validation-20260927/
kernels-phase7.md). Needs torch and Triton (``TRITON_INTERPRET=1``; macOS has no Triton wheel, so these ran in a Linux
arm64 container with torch 2.14 CPU and Triton 3.8) and the pinned ``nvidia/ops/qsa.py`` (``ARC_PINNED_SITE``)."""

import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[1]
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
PINNED_QSA = PINNED_SITE / "vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
PINNED_QSA_SHA256 = "c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f"


def stub_vllm(monkeypatch):
    """The two vLLM imports of the pinned ops/qsa.py (platform checks and Triton re-exports)."""
    import triton
    import triton.language as tl

    vllm = types.ModuleType("vllm")
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(has_device_capability=lambda *a: False,
                                                       is_device_capability_family=lambda *a: False)
    utils = types.ModuleType("vllm.triton_utils")
    utils.HAS_TRITON, utils.tl, utils.triton = True, tl, triton
    for name, module in (("vllm", vllm), ("vllm.platforms", platforms), ("vllm.triton_utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)


def modules(monkeypatch):
    if not PINNED_QSA.is_file():
        pytest.skip("the pinned ops/qsa.py (ARC_PINNED_SITE)")
    stub_vllm(monkeypatch)
    sys.path.insert(0, str(ROOT / "scripts"))
    import qsa_prefill_bench as bench

    kv8 = bench.load_module("arc_kv8_test_p7", ROOT / "src/arc_infra/kv8_runtime.py")
    pinned = bench.load_module("pinned_qsa_test_p7", PINNED_QSA)
    return bench, kv8, pinned


def test_prefill_bench_case_runs_and_the_int8_grid_matches_the_pinned_profile(monkeypatch):
    import hashlib

    bench, kv8, pinned = modules(monkeypatch)
    assert hashlib.sha256(PINNED_QSA.read_bytes()).hexdigest() == PINNED_QSA_SHA256

    def cpu_wrapper(q, k, v, indices, table, t2r, out):
        """The pinned wrapper refuses CPU tensors; the same kernel with the pinned profile (kv8.launch_profile copies
        the wrapper's choice)."""
        p = kv8.launch_profile(q.shape[0], k.shape[2], q.shape[1] // k.shape[2], indices.shape[1])
        # Split 1 only: the pinned merge kernel does not run under the interpreter (the kv8 copy was rewritten so
        # that it does); the pinned splits are timed and checked on the GPU.
        return bench.launch_bf16(pinned, q, k, v, indices, table, t2r, out, (p["block_n"], 1, p["warps"]), 2)

    monkeypatch.setattr(pinned, "qsa_sparse_paged_attention", cpu_wrapper)
    case = bench.run_case(kv8, pinned, rows=6, ctx=256, page=64, topk=48, pattern="shared", device="cpu",
                          timer=lambda fn: (fn(), 1.0)[1], grid=[(16, 1, 2), (16, 2, 4), (32, 1, 2)], stages=(2,),
                          check_rows=3, bf16_grid=[(16, 1, 2), (32, 1, 4)])
    assert case["int8_pinned_profile"]["err_fp32"] < 0.02  # the interpreter-exact FP32 dots of the kv8 kernel
    # Within one BF16 ulp: the interpreter's FP32 -> BF16 cast truncates (the GPU rounds to nearest, and the job
    # checks equality there).
    assert case["dequant_triton"]["rel_to_torch"] < 8e-3
    int8 = [e for e in case["int8_grid"] if "us" in e]
    # Another tile / split changes the online-softmax rescaling and so where p x v_scale is rounded to BF16: the
    # profiles agree at the BF16-rounding level (0.16-0.2% here), not bitwise.
    assert len(int8) == 3 and all(e["rel_to_pinned_same_kind"] < 5e-3 for e in int8)
    # The pinned BF16 kernel runs under the interpreter (its BF16 dot is not exact there; timed on the GPU only).
    assert "us" in case["bf16_pinned"] and len([e for e in case["bf16_grid"] if "us" in e]) == 2
    assert case["pinned_profile"]["block_n"] == 16  # 6 rows x 2 heads: the pinned decode-size branch


# ----------------------------------------------------------------------------- the kv8 prefill path (runtime)
def mixed_case(kv8, generator, *, page=64, blocks=24, decode=5, prefill_rows=70, topk=48):
    """A mixed call: ``decode`` single-row requests and one prefill request of ``prefill_rows`` rows (its pages
    scattered in the cache), int8 cache, block tables padded with -1 (and one stray out-of-range entry)."""
    kv_heads, group, dim = 2, 12, 256
    k = torch.randn((blocks, page, kv_heads, dim), generator=generator)
    v = torch.randn((blocks, page, kv_heads, dim), generator=generator)
    packed = kv8.pack_cache(k, v, "int8")
    requests = decode + 1
    width = 6
    table = torch.full((requests, width), -1, dtype=torch.int32)
    order = torch.randperm(blocks, generator=generator).to(torch.int32)
    for r in range(requests):
        table[r, :4] = order[r * 4:(r + 1) * 4]
    table[decode, 5] = blocks + 7  # beyond the cache: never dequantized, masked
    rows = decode + prefill_rows
    t2r = torch.cat([torch.arange(decode), torch.full((prefill_rows,), decode)]).to(torch.int32)
    indices = torch.stack([torch.randperm(4 * page, generator=generator)[:topk] for _ in range(rows)]).to(torch.int32)
    indices[0, -3:] = -1  # padding
    q = torch.randn((rows, kv_heads * group, dim), generator=generator).to(torch.bfloat16)
    return q, packed, indices, table, t2r


def test_kv8_prefill_path_matches_the_pinned_bf16_kernel_and_keeps_decode_rows_on_int8(monkeypatch):
    bench, kv8, pinned = modules(monkeypatch)
    spec = __import__("importlib.util").util.spec_from_file_location("arc_kv8_prefill_test",
                                                                     ROOT / "src/arc_infra/kv8_prefill_runtime.py")
    prefill = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(prefill)
    generator = torch.Generator().manual_seed(3)
    q, packed, indices, table, t2r = mixed_case(kv8, generator)
    ck, cv, sk, sv = kv8.cache_views(packed, 256, False)
    split = prefill.plan(t2r, indices, table, ck.shape[1], ck.shape[0], min_rows=64)
    assert split["heavy_rows"].tolist() == list(range(5, q.shape[0])) and split["light_rows"].tolist() == [0, 1, 2, 3, 4]
    assert split["pages"].tolist() == table[5, :4].long().tolist()  # the stray entry is never read
    assert split["table"][0, :4].tolist() == [0, 1, 2, 3] and split["t2r"].unique().tolist() == [0]
    # The dequantized pages are exactly the int8 values in BF16 (the interpreter's cast truncates: within one ulp).
    got = prefill.dequant_pages(ck, sk, split["pages"])
    ref = prefill.dequant_reference(ck.index_select(0, split["pages"]), sk.index_select(0, split["pages"]))
    assert float((got.float() - ref.float()).norm() / ref.float().norm()) < 8e-3
    monkeypatch.setattr(prefill, "dequant_pages", lambda c, s, p: prefill.dequant_reference(c.index_select(0, p),
                                                                                             s.index_select(0, p)))
    out = torch.zeros_like(q)
    prefill.prefill_attention(kv8, pinned, q, ck, cv, sk, sv, indices, table, t2r, out, split)
    # Prefill rows: the pinned BF16 kernel, run the ordinary way on a BF16 cache of the dequantized values (same
    # interpreter arithmetic; only the page remap and the tile differ).
    kb, vb = prefill.dequant_reference(ck, sk).contiguous(), prefill.dequant_reference(cv, sv).contiguous()
    rows = split["heavy_rows"]
    reference = torch.empty((rows.numel(), *q.shape[1:]), dtype=torch.bfloat16)
    safe = table.clamp(0, ck.shape[0] - 1)
    bench.launch_bf16(pinned, q[rows], kb, vb, indices[rows].contiguous(), safe[5:6].contiguous(),
                      torch.zeros((rows.numel(),), dtype=torch.int32), reference, (64, 1, 2), 2)
    assert float((out[rows].float() - reference.float()).norm() / reference.float().norm()) < 5e-3
    # Decode rows: exactly the int8 kernel on those rows.
    light = split["light_rows"]
    int8 = torch.empty((light.numel(), *q.shape[1:]), dtype=torch.bfloat16)
    kv8.sparse_attention(q[light], ck, cv, sk, sv, indices[light].contiguous(), table, t2r[light].contiguous(), int8)
    assert torch.equal(out[light], int8)
    # The interpreter's BF16 dot is wrong in absolute terms (it matches itself above), so the numerics of the split are
    # checked in FP32: the pinned kernel casts p to the values' dtype, so FP32 inputs make it an FP32 kernel. The
    # planned pages, scratch table and remapped requests against the FP32 reference on the dequantized cache.
    k32 = prefill.dequant_reference(ck.index_select(0, split["pages"]), sk.index_select(0, split["pages"])).float()
    v32 = prefill.dequant_reference(cv.index_select(0, split["pages"]), sv.index_select(0, split["pages"])).float()
    out32 = torch.empty((rows.numel(), *q.shape[1:]), dtype=torch.float32)
    prefill.bf16_attention(pinned, q[rows].float(), k32, v32, indices[rows].contiguous(), split["table"], split["t2r"],
                           out32)
    exact = kv8.attention_reference(q[rows], kb, vb, indices[rows], table, t2r[rows])
    assert float((out32 - exact).norm() / exact.norm()) < 1e-5
    # And against the int8 kernel (the interpreter's FP32-dot branch on BF16-rounded operands): the BF16-level gap
    # the load-time check bounds (CHECK_LIMIT).
    full = torch.empty_like(q)
    kv8.sparse_attention(q, ck, cv, sk, sv, indices, table, t2r, full)
    assert float((out32 - full[rows].float()).norm() / full[rows].float().norm()) < prefill.CHECK_LIMIT


def test_kv8_prefill_entry_keeps_decode_calls_on_arc_kv8(monkeypatch):
    spec = __import__("importlib.util").util.spec_from_file_location("arc_kv8_prefill_test2",
                                                                     ROOT / "src/arc_infra/kv8_prefill_runtime.py")
    prefill = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(prefill)
    calls = []
    fake_kv8 = types.SimpleNamespace(attention=lambda *a: calls.append(a) or "served")
    vllm = types.ModuleType("vllm")
    vllm.arc_kv8 = fake_kv8
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    impl = types.SimpleNamespace(head_size=256, kv_cache_dtype="int8_per_token_head")
    cache = torch.zeros((4, 2, 64, 520), dtype=torch.int8)
    for rows in (1, 14, 25, 32, 256):  # every CUDA-graph size and any call up to 512 programs
        q = torch.zeros((rows, 24, 256), dtype=torch.bfloat16)
        assert prefill.attention(impl, q, cache, None, None, None, None) == "served"
    assert len(calls) == 5 and prefill._STATE["counts"]["decode"] == 5
