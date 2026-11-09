"""Wide decode tiles for the dense overlay (kernel profile, 30 September; docs/validation-20260930/kernel-profile.md).

Project venv: the settings (absent keys leave every earlier setting byte-identical), the patch (one install line, after
the phase-6 one and before ``enable``; no helper without a wide key). With torch and Triton (``TRITON_INTERPRET=1``,
Linux container as tests/test_dense_fused.py): bucket and tile selection, the W8A16 kernel on 32- and 64-row tiles at
M = 33..56 against an FP32 reference, and the warmup tuner's merge with a fixed plan (timings mocked)."""

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from arc_infra import dense_kernels_patch as patch

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ("def enable_qwen38next_low_latency_gemm(\n    module,\n    dtype,\n) -> None:\n" + patch.ANCHOR
             + "    return\n")
WIDE = {**patch.PHASE5, "mid_wide": True, "mid_wide_shapes": [list(s) for s in patch.PRUNED_ROUTER_SHAPES]}


# ----------------------------------------------------------------------------- settings and patch
def test_wide_keys_are_validated_and_absent_when_off():
    assert patch.validate(patch.PHASE5) == patch.validate({**patch.PHASE5, "mid_wide": False, "mid_wide_shapes": []})
    assert not patch.wide(patch.validate(patch.PHASE5))
    wide = patch.validate(WIDE)
    assert wide["mid_wide"] is True and wide["mid_wide_shapes"] == [[448, 2560]] and patch.wide(wide)
    assert patch.validate({**WIDE, "mid_wide_shapes": [[448, 2560], [448, 2560]]})["mid_wide_shapes"] == [[448, 2560]]
    plan = {"bf16": {"448,2560": {"40": [32, 16, 256, 4, 3], "56": None}}}
    assert patch.validate({**WIDE, "mid_wide_plan": plan})["mid_wide_plan"] == plan
    for bad in ({**patch.PHASE2, "mid_wide": True},                                   # extends mid_m
                {**patch.PHASE5, "mid_wide_shapes": [[448, 2560]]},                  # extends mid_wide
                {**WIDE, "mid_wide": 1},
                {**WIDE, "mid_wide_shapes": [[448]]},
                {**WIDE, "mid_wide_plan": {"bf16": {"448,2560": {"32": [64, 16, 128, 4, 4]}}}},  # wide buckets only
                {**WIDE, "mid_wide_plan": {"bf16": {"448,2560": {"40": [16, 128, 4, 4]}}}},      # BLOCK_M first
                {**WIDE, "mid_wide_plan": {"bf16": {"448,2560": {"40": [48, 16, 128, 4, 4]}}}},  # 32 or 64 rows
                {**WIDE, "mid_wide_plan": {"fp8": {}}}):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_wide_buckets_key_extends_the_buckets_for_mtp_verify_batches():
    """Throughput arm (3 October): ``mid_wide_buckets`` (absent = 40 / 48 / 56) for MTP-1 verify batches."""
    assert "mid_wide_buckets" not in patch.validate(WIDE)
    mtp = patch.validate({**WIDE, "mid_wide_buckets": [48, 56, 64, 72]})
    assert mtp["mid_wide_buckets"] == [48, 56, 64, 72] and patch.wide(mtp)
    plan = {"int8": {"2560,6144": {"72": [64, 64, 128, 4, 3]}}}
    assert patch.validate({**WIDE, "mid_wide_buckets": [64, 72], "mid_wide_plan": plan})["mid_wide_plan"] == plan
    for bad in ({**WIDE, "mid_wide_buckets": []}, {**WIDE, "mid_wide_buckets": [72, 64]},
                {**WIDE, "mid_wide_buckets": [60]}, {**WIDE, "mid_wide_buckets": [64, 64]},
                {**patch.PHASE5, "mid_wide_buckets": [64]},                           # extends mid_wide
                {**WIDE, "mid_wide_plan": plan},                                      # 72 is not a default bucket
                {**WIDE, "mid_wide_buckets": "64"}):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_runtime_reads_the_wide_buckets(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("dense_wide_cfg", ROOT / "src/arc_infra/dense_wide_runtime.py")
    torch_stub = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.load_config({"mid_wide": True})["buckets"] == (40, 48, 56)
    assert module.load_config({"mid_wide": True, "mid_wide_buckets": [48, 56, 64, 72]})["buckets"] == (48, 56, 64, 72)


def synthetic_site(tmp_path, monkeypatch):
    original = SYNTHETIC.encode()
    monkeypatch.setattr(patch, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    target = tmp_path / "site" / patch.RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    return tmp_path / "site", target, original


def reset(target, original):
    target.write_bytes(original)
    for name in (patch.HELPER, patch.SETTINGS, patch.PHASE6_HELPER, patch.WIDE_HELPER):
        if target.with_name(name).exists():
            target.with_name(name).unlink()


def test_wide_adds_one_install_line_and_its_helper_only_when_set(tmp_path, monkeypatch):
    site, target, original = synthetic_site(tmp_path, monkeypatch)
    identity5 = patch.apply(site, tmp_path / "e5", patch.PHASE5)
    patched5 = target.read_bytes()
    assert patched5 == patch.patched_source(original) and b"arc_dense_wide" not in patched5
    assert "wide_helper_sha256" not in identity5 and not target.with_name(patch.WIDE_HELPER).exists()
    reset(target, original)
    identity = patch.apply(site, tmp_path / "ew", WIDE)
    text = target.read_text()
    assert text.count("_arc_dense_wide_install()") == 1
    assert text.index("_arc_dense_wide_install()") < text.index("_arc_dense_enable(module, dtype)")
    assert text.replace(patch.INSERT_WIDE, "") == patched5.decode()
    helper = target.with_name(patch.WIDE_HELPER)
    assert helper.read_bytes() == (ROOT / "src/arc_infra" / patch.WIDE_SOURCE).read_bytes()
    assert identity["wide_helper_sha256"] == hashlib.sha256(helper.read_bytes()).hexdigest()
    assert json.loads(target.with_name(patch.SETTINGS).read_text())["mid_wide_shapes"] == [[448, 2560]]
    # Re-applying on the patched tree is accepted (restart in place); with phase 6 too, wide comes after it.
    assert patch.apply(site, tmp_path / "ew", WIDE) == identity
    reset(target, original)
    patch.apply(site, tmp_path / "e6w", {**patch.PHASE6_FUSED, "mid_wide": True})
    text = target.read_text()
    assert text.index("_arc_dense_fused_install()") < text.index("_arc_dense_wide_install()") \
        < text.index("_arc_dense_enable(module, dtype)")


# ----------------------------------------------------------------------------- runtime (torch + Triton)
DOT_BF16 = "acc += tl.dot(a, tl.trans(b.to(tl.bfloat16)))"
DOT_FP32 = "acc += tl.dot(a.to(tl.float32), tl.trans(b.to(tl.float32)))"


def load_modules(monkeypatch, config, fp32_dot=None):
    """The runtime modules under the interpreter. ``fp32_dot`` (a directory): load a copy of the base runtime whose
    only change is the dot's operand dtype (BF16 -> FP32), fed FP32 activations: this Triton build's interpreter
    mis-reads BF16 tensors (garbage for the pinned tiles too), so the tiling, masking and split-K logic is tested in
    FP32 and the BF16 numerics stay with the GPU's load-time checks (ARC_DENSE_CHECK / ARC_DENSE_MID / ARC_DENSE_WIDE)."""
    torch = pytest.importorskip("torch")
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton is not installed (Linux container, TRITON_INTERPRET=1)")
    os.environ.setdefault("TRITON_INTERPRET", "1")
    package = "vllm.models.qwen3_8_flash_next.nvidia"
    for name in ("vllm", "vllm.models", "vllm.models.qwen3_8_flash_next", package):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    modules = {}
    sources = {"arc_dense_kernels": ROOT / "src/arc_infra/dense_kernels_runtime.py",
               "arc_dense_wide": ROOT / "src/arc_infra/dense_wide_runtime.py"}
    if fp32_dot is not None:
        text = sources["arc_dense_kernels"].read_text()
        assert text.count(DOT_BF16) == 1
        sources["arc_dense_kernels"] = Path(fp32_dot) / "arc_dense_kernels_fp32_dot.py"
        sources["arc_dense_kernels"].write_text(text.replace(DOT_BF16, DOT_FP32))
    for short, source in sources.items():
        spec = importlib.util.spec_from_file_location(f"{package}.{short}", source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        setattr(sys.modules[package], short, module)
        spec.loader.exec_module(module)
        modules[short] = module
    base, wide = modules["arc_dense_kernels"], modules["arc_dense_wide"]
    monkeypatch.setattr(base, "_num_sms", lambda device: 188)
    path = Path(os.environ.get("PYTEST_TMP", "/tmp")) / "arc_dense_kernels_wide_test.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(base, "CONFIG_PATH", path)
    monkeypatch.setattr(wide, "CONFIG_PATH", path)
    return torch, base, wide


def test_install_extends_buckets_shapes_and_tiles(monkeypatch):
    torch, base, wide = load_modules(monkeypatch, patch.validate(WIDE))
    assert base.mid_bucket(40) is None or base.MID_M_BUCKETS == (8, 16, 24, 32)
    wide.install()
    assert base.MID_M_BUCKETS == (8, 16, 24, 32, 40, 48, 56) and base.MID_INT8_BUCKETS == (24, 32, 40, 48, 56)
    assert (448, 2560) in base.MID_BF16_SHAPES
    assert [base.mid_bucket(m) for m in (4, 5, 32, 33, 40, 41, 56, 57)] == [None, 8, 32, 40, 40, 48, 56, None]
    assert base._mid_tile(40, [32, 128, 4, 4]) == (64, 32, 128, 4, 4)
    assert base._mid_tile(24, [32, 128, 4, 4]) == (32, 32, 128, 4, 4)
    assert base._mid_tile(48, [32, 16, 256, 4, 3]) == (32, 16, 256, 4, 3)
    # The phase-5 fixed plan still parses to its own tiles; its buckets <= 32 are unchanged.
    config = base.load_config()
    assert isinstance(config["mid_m"], dict)
    assert base._STATE["mid"]["int8"][(16384, 2560)] == {24: (32, 32, 128, 4, 4), 32: (32, 32, 256, 4, 3)}
    wide.install()  # idempotent
    assert base.MID_M_BUCKETS.count(40) == 1


@pytest.mark.parametrize("shape,m,tile,int8", [
    ((448, 2560), 40, (64, 16, 128, 4, 4), False),     # the pruned router, one 64-row tile
    ((448, 2560), 33, (32, 16, 256, 4, 3), False),     # two 32-row M tiles
    ((336, 10240), 56, (64, 32, 128, 4, 4), False),    # HC down + inject at 56 streams (split-K)
    ((2560, 6144), 48, (64, 64, 128, 4, 3), True),     # attn out, INT8
    ((96, 2560), 44, (64, 32, 64, 4, 4), False),       # in_proj_ba
])
def test_wide_tiles_match_an_fp32_reference(monkeypatch, tmp_path, shape, m, tile, int8):
    torch, base, wide = load_modules(monkeypatch, patch.validate(WIDE), fp32_dot=tmp_path)
    wide.install()
    n, k = shape
    generator = torch.Generator().manual_seed(11)
    # Activations on a BF16-exact grid and small-integer weights: every product and partial sum is exact in FP32, so
    # the only error left is the output's one BF16 rounding (<= 1 unit in the last place at the output's magnitude).
    x = torch.randint(-8, 9, (m, k), generator=generator).float() / 8
    if int8:
        w = torch.randint(-7, 8, (n, k), generator=generator, dtype=torch.int8)
        s = torch.full((n,), 0.5)
        ref = x.float() @ (w.float() * s[:, None]).t()
    else:
        w = torch.randint(-7, 8, (n, k), generator=generator).to(torch.bfloat16)
        s = torch.ones((n,))
        ref = x.float() @ w.float().t()
    got = base.w8a16_gemm(x, w, s, tile).float()
    assert got.shape == (m, n)
    ulp = torch.maximum(ref.abs(), torch.ones_like(ref)) * 2.0 ** -7  # BF16: 8 significant bits
    assert bool(((got - ref).abs() <= ulp).all())
    # The same GEMM on the phase-2 path (the pinned INT8 / prefill tile) agrees to the same rounding.
    assert bool(((got - base.w8a16_gemm(x, w, s).float()).abs() <= 2 * ulp).all())


def test_warmup_tuner_merges_with_the_fixed_plan(monkeypatch, tmp_path):
    torch, base, wide = load_modules(monkeypatch, patch.validate(WIDE), fp32_dot=tmp_path)
    wide.install()
    base.load_config()
    # Timings: the pinned path costs 10 us; a candidate costs 5 us if it is a 64-row tile with BLOCK_N 32, else 12.
    calls = []

    def fake_time(fn, copies, replays=None):
        calls.append(copies)
        return fake_time.next

    def timed_gemm(original):
        def run(x, w, s, tile=None):
            fake_time.next = 10.0 if tile is None else (5.0 if tile[0] == 64 and tile[1] == 32 else 12.0)
            return original(x, w, s, tile)
        return run

    monkeypatch.setattr(base, "time_graph", lambda fn, copies, replays=None: (fn(0), fake_time(fn, copies))[1])
    monkeypatch.setattr(base, "w8a16_gemm", timed_gemm(base.w8a16_gemm))
    monkeypatch.setattr(base, "_pinned_bf16", lambda x, w: (setattr(fake_time, "next", 10.0),
                                                           (x.float() @ w.float().t()).to(torch.bfloat16))[1])
    monkeypatch.setattr(wide, "WIDE_CANDIDATES", ((64, 32, 128, 4, 4), (64, 16, 128, 4, 4)))
    monkeypatch.setattr(base, "MID_CANDIDATES", ((32, 128, 4, 4),))
    monkeypatch.setattr(base, "MID_BENCH_BYTES", 1)  # one copy per shape on the CPU
    # The merge logic is under test, not the error bound: the fake pinned path here is an exact FP32 product, so the
    # kernel's single BF16 rounding (and the interpreter's BF16 activation reads) would fail the 1.25x bound.
    monkeypatch.setattr(base, "MID_ERROR_FACTOR", 100.0)
    # A shape of the fixed plan (HC down): only the wide buckets are tuned; the plan's own tiles stay.
    w = (torch.randint(-7, 8, (336, 10240)) ).to(torch.bfloat16)
    before = dict(base._STATE["mid"]["bf16"][(336, 10240)])
    base.tune_mid("bf16", w, None, "hc_down")
    plan = base._STATE["mid"]["bf16"][(336, 10240)]
    assert {b: plan[b] for b in (8, 16, 24, 32)} == {b: before[b] for b in (8, 16, 24, 32)}
    assert plan[40] == plan[48] == plan[56] == (64, 32, 128, 4, 4)
    assert base.mid_tile("bf16", 336, 10240, 37) == (64, 32, 128, 4, 4)
    assert base.mid_tile("bf16", 336, 10240, 20) == before[24]
    # The pruned router, absent from the fixed plan: every bucket is tuned (<= 32 with the phase-5 candidates, which
    # are 12 us here, so they stay on the pinned path; the wide buckets take the 5 us tile).
    w = torch.randint(-7, 8, (448, 2560)).to(torch.bfloat16)
    base.tune_mid("bf16", w, None, "router")
    plan = base._STATE["mid"]["bf16"][(448, 2560)]
    assert plan[8] is None and plan[32] is None and plan[40] == (64, 32, 128, 4, 4)
    assert base.mid_tile("bf16", 448, 2560, 16) is None and base.mid_tile("bf16", 448, 2560, 52) == (64, 32, 128, 4, 4)
    result = wide._STATE["results"][-1]
    assert result["shape"] == [448, 2560] and sorted(result["buckets"], key=int) == ["8", "16", "24", "32", "40",
                                                                                     "48", "56"]
    # Tuned once per shape.
    n_results = len(wide._STATE["results"])
    base.tune_mid("bf16", w, None, "router")
    assert len(wide._STATE["results"]) == n_results
