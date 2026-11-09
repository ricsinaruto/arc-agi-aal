"""Kernel work, phase 6: the dense overlay's phase-6 keys (fused split-K reduction, INT8 mid-M buckets, tuner
candidates; docs/validation-20260927/kernels-phase6.md).

Project venv: settings, the patch (earlier arms byte-identical: same settings, patched file and helper, no phase-6
helper), the kernel's partial-GEMM body against the pinned-path kernel's, the install hook. With torch and Triton
(``TRITON_INTERPRET=1``; macOS has no Triton wheel, so these ran in a Linux arm64 container with torch 2.14 CPU and
Triton 3.8, see the doc): the fused kernel against the two-launch path, bitwise, and the ticket bookkeeping."""

import ast
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
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
SYNTHETIC = ("def enable_qwen38next_low_latency_gemm(\n    module,\n    dtype,\n) -> None:\n" + patch.ANCHOR
             + "    return\n")


def synthetic_site(tmp_path, monkeypatch):
    original = SYNTHETIC.encode()
    monkeypatch.setattr(patch, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    target = tmp_path / "site" / patch.RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    return tmp_path / "site", target, original


# ----------------------------------------------------------------------------- settings
def test_phase6_keys_are_validated_and_absent_when_off():
    # The recorded settings serialize exactly as before.
    assert patch.validate(patch.PHASE2) == {"skinny": True, "int8_weights": list(patch.INT8_FAMILIES)}
    assert set(patch.validate(patch.PHASE5)) == {"skinny", "int8_weights", "mid_m"}
    assert patch.validate({**patch.PHASE5, "fused_splitk": False, "mid_int8_buckets": []}) == patch.validate(
        patch.PHASE5)
    fused = patch.validate(patch.PHASE6_FUSED)
    assert fused == {**patch.validate(patch.PHASE5), "fused_splitk": True} and patch.phase6(fused)
    assert not patch.phase6(patch.validate(patch.PHASE5))
    tune = patch.validate(patch.PHASE6_TUNE)
    assert tune["mid_m"] is True and tune["mid_int8_buckets"] == [8, 16, 24, 32]
    assert tune["mid_extra_candidates"] == patch.MID_EXTRA_CANDIDATES_PHASE6
    assert patch.validate({**patch.PHASE5, "mid_int8_buckets": [32, 16]})["mid_int8_buckets"] == [16, 32]
    for bad in ({**patch.PHASE5, "fused_splitk": 1},
                {**patch.PHASE5, "mid_int8_buckets": [12]},
                {**patch.PHASE5, "mid_int8_buckets": [16, 16]},
                {**patch.PHASE2, "mid_int8_buckets": [16]},  # needs mid_m
                {**patch.PHASE5, "mid_extra_candidates": [[16, 128, 4, 4]]},  # a fixed plan is never re-tuned
                {**patch.PHASE2, "mid_m": True, "mid_extra_candidates": [[16, 128, 4]]},
                {"fused_splitk": True}):  # extends the skinny / INT8 paths
        with pytest.raises(ValueError):
            patch.validate(bad)


# ----------------------------------------------------------------------------- patch
def test_earlier_arms_are_byte_identical_and_phase6_adds_one_install_line(tmp_path, monkeypatch):
    site, target, original = synthetic_site(tmp_path, monkeypatch)
    identity5 = patch.apply(site, tmp_path / "e5", patch.PHASE5)
    patched5 = target.read_bytes()
    assert not target.with_name(patch.PHASE6_HELPER).exists() and "phase6_helper_sha256" not in identity5
    assert b"arc_dense_fused" not in patched5
    assert patched5 == patch.patched_source(original)  # the pre-phase-6 function, unchanged
    # Phase 6 on a re-extracted runtime: the install line runs first, then the pinned-path enable.
    target.write_bytes(original)
    for name in (patch.HELPER, patch.SETTINGS):
        target.with_name(name).unlink()
    identity6 = patch.apply(site, tmp_path / "e6", patch.PHASE6_FUSED)
    text = target.read_text()
    assert text.count("_arc_dense_fused_install()") == 1
    assert text.index("_arc_dense_fused_install()") < text.index("_arc_dense_enable(module, dtype)")
    assert text.replace(patch.INSERT_PHASE6, "") == patched5.decode()
    helper = target.with_name(patch.PHASE6_HELPER)
    assert helper.read_bytes() == (ROOT / "src/arc_infra" / patch.PHASE6_SOURCE).read_bytes()
    assert identity6["phase6_helper_sha256"] == hashlib.sha256(helper.read_bytes()).hexdigest()
    assert identity6["helper_sha256"] == identity5["helper_sha256"]
    assert json.loads(target.with_name(patch.SETTINGS).read_text())["fused_splitk"] is True
    # Accepted as is on the same working directory; a tampered phase-6 helper is refused.
    assert patch.apply(site, tmp_path / "e6", patch.PHASE6_FUSED) == identity6
    helper.write_text("# tampered\n")
    with pytest.raises(RuntimeError, match="modified"):
        patch.apply(site, tmp_path / "e6", patch.PHASE6_FUSED)


def test_the_hero_serving_files_are_the_recorded_ones():
    """The hero smoke's recorded dense identity (arena-hero-smoke-v1: patched file, helper, settings): phase 6 changes
    none of them for arms without a phase-6 key."""
    helper = (ROOT / "src/arc_infra/dense_kernels_runtime.py").read_bytes()
    assert hashlib.sha256(helper).hexdigest() == "85e1dbc665096cfd4cf0678b80dc4000d0d11ee0333765ceade3e4b5e9ed82bf"
    assert json.dumps(patch.validate(patch.PHASE5), sort_keys=True) == json.dumps(
        {"int8_weights": list(patch.INT8_FAMILIES), "mid_m": patch.MID_PLAN_PHASE5, "skinny": True}, sort_keys=True)
    pinned = PINNED_SITE / patch.RELATIVE
    if pinned.is_file():
        assert hashlib.sha256(patch.patched_source(pinned.read_bytes())).hexdigest() == \
            "6ee872979604e981b1191d366bddec7de60477a663b6e9960d0e2880ddaa6878"


def test_exact_pinned_low_latency_gemm_when_available():
    pinned = PINNED_SITE / patch.RELATIVE
    if not pinned.is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE); the hash guard is checked if supplied")
    original = pinned.read_bytes()
    assert hashlib.sha256(original).hexdigest() == patch.ORIGINAL_SHA256
    with_phase6 = patch.patched_source(original, with_phase6=True).decode()
    assert with_phase6.replace(patch.INSERT_PHASE6, "") == patch.patched_source(original).decode()
    compile(with_phase6, patch.RELATIVE, "exec")


# ----------------------------------------------------------------------------- kernel structure
def _function(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def test_fused_kernel_runs_the_pinned_path_partial_gemm_unchanged():
    """The partial GEMM (everything up to the store) is statement for statement the two-launch kernel's, so the
    partials, and hence the fixed-order sum, are the same bits."""
    base = _function(ROOT / "src/arc_infra/dense_kernels_runtime.py", "_w8a16_kernel")
    fused = _function(ROOT / "src/arc_infra/dense_fused_runtime.py", "_w8a16_fused_kernel")
    stop = next(i for i, node in enumerate(base.body) if isinstance(node, ast.If))  # "if PARTIAL:"
    body = [ast.dump(node) for node in base.body[:stop]]
    fused_body = [ast.dump(node) for node in fused.body if not isinstance(node, ast.Expr)]  # skip the docstring
    assert fused_body[:len(body)] == body
    reduce = _function(ROOT / "src/arc_infra/dense_kernels_runtime.py", "_splitk_reduce_kernel")
    text = ast.unparse(reduce)
    assert "acc = tl.zeros((BLOCK,), dtype=tl.float32)" in text and "for s in tl.static_range(SPLIT)" in text
    fused_text = ast.unparse(fused)
    assert "for split in tl.static_range(SPLIT)" in fused_text and "total.to(tl.bfloat16)" in fused_text
    assert "tl.atomic_xchg(ticket, 0" in fused_text  # self-resetting tickets


# ----------------------------------------------------------------------------- runtime (torch + Triton)
def load_modules(monkeypatch, config):
    torch = pytest.importorskip("torch")
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton is not installed (Linux container, TRITON_INTERPRET=1)")
    os.environ.setdefault("TRITON_INTERPRET", "1")
    package = "vllm.models.qwen3_8_flash_next.nvidia"
    for name in ("vllm", "vllm.models", "vllm.models.qwen3_8_flash_next", package):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    modules = {}
    for short, source in (("arc_dense_kernels", "dense_kernels_runtime.py"),
                          ("arc_dense_fused", "dense_fused_runtime.py")):
        spec = importlib.util.spec_from_file_location(f"{package}.{short}", ROOT / "src/arc_infra" / source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        setattr(sys.modules[package], short, module)
        spec.loader.exec_module(module)
        modules[short] = module
    base, fused = modules["arc_dense_kernels"], modules["arc_dense_fused"]
    monkeypatch.setattr(base, "_num_sms", lambda device: 188)
    config_path = Path(os.environ.get("PYTEST_TMP", "/tmp")) / "arc_dense_kernels_phase6_test.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(fused, "CONFIG_PATH", config_path)
    return torch, base, fused


def test_install_applies_only_the_configured_extensions(monkeypatch):
    torch, base, fused = load_modules(monkeypatch, {"mid_int8_buckets": [8, 16, 24, 32],
                                                    "mid_extra_candidates": [[16, 128, 4, 4], [16, 256, 4, 3]]})
    original = base.w8a16_gemm
    fused.install()
    assert base.w8a16_gemm is original  # no fused_splitk
    assert base.MID_INT8_BUCKETS == (8, 16, 24, 32)
    assert base.MID_CANDIDATES[-1] == (16, 128, 4, 4) and len(base.MID_CANDIDATES) == 5  # no duplicate
    fused.install()  # idempotent
    assert len(base.MID_CANDIDATES) == 5


@pytest.mark.parametrize("shape,m,tile", [
    ((336, 10240), 16, (16, 32, 128, 4, 4)),   # HC down at 14 streams: split 16
    ((96, 2560), 24, (32, 32, 128, 4, 4)),     # GDN in_proj_ba: split 4
    ((512, 2560), 9, None),                    # router, default INT8 tile path
    ((2560, 6144), 13, (16, 16, 256, 4, 3)),   # attn out
])
def test_fused_split_k_is_bitwise_equal_to_the_two_launch_path(monkeypatch, shape, m, tile):
    torch, base, fused = load_modules(monkeypatch, {"fused_splitk": True})
    two_launch = base.w8a16_gemm
    fused.install()
    assert base.w8a16_gemm is fused.fused_w8a16_gemm
    n, k = shape
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((m, k), generator=generator).to(torch.bfloat16)
    if tile is None:  # INT8 weights with per-channel scales
        w = torch.randint(-127, 128, (n, k), generator=generator, dtype=torch.int8)
        s = torch.rand((n,), generator=generator) * 1e-2
    else:
        w = (torch.randn((n, k), generator=generator) * 0.02).to(torch.bfloat16)
        s = torch.ones((n,))
    chosen = tile or base.tile_config(n, k, m)
    split, _ = base.split_k(-(-n // chosen[1]) * -(-m // chosen[0]), k, chosen[2], 188)
    assert split > 1
    reference = two_launch(x, w, s, tile)
    got = base.w8a16_gemm(x, w, s, tile)
    assert torch.equal(got, reference)
    assert fused._STATE["stats"]["fused_calls"] == 1 and all(fused._STATE["checked"].values())
    assert not fused._STATE["arena"].any()  # every ticket back to zero
    # Not checked against FP32 here: the interpreter's BF16 dot is wrong on this build (as in the kv8 kernel tests),
    # identically for both paths; the GPU checks both against FP32 at load (ARC_DENSE_CHECK / ARC_DENSE_MID).


def test_no_split_calls_and_failed_checks_take_the_two_launch_path(monkeypatch):
    torch, base, fused = load_modules(monkeypatch, {"fused_splitk": True})
    two_launch = base.w8a16_gemm
    calls = []
    monkeypatch.setattr(fused, "_STATE", {**fused._STATE, "checked": {}, "sets": {}, "arena": None, "next": 0,
                                          "stats": {"fused_calls": 0, "fallback_calls": 0}, "installed": False})
    fused.install()

    def counting(x, w, s, tile=None):
        calls.append(tile)
        return two_launch(x, w, s, tile)

    fused._STATE["original"] = counting
    x = torch.randn((4, 2560)).to(torch.bfloat16)
    w = torch.randint(-127, 128, (16384, 2560), dtype=torch.int8)
    s = torch.ones((16384,))
    base.w8a16_gemm(x, w, s)  # qkvz: 512 N tiles, no split
    assert calls == [None] and fused._STATE["stats"]["fused_calls"] == 0
    # A (shape, tile, split) whose check failed stays on the two-launch path.
    x = torch.randn((16, 10240)).to(torch.bfloat16)
    w = (torch.randn((336, 10240)) * 0.02).to(torch.bfloat16)
    tile = (16, 32, 128, 4, 4)
    key = (336, 10240, *tile, 16, str(w.dtype))
    fused._STATE["checked"][key] = False
    base.w8a16_gemm(x, w, torch.ones((336,)), tile)
    assert calls[-1] == tile and fused._STATE["stats"]["fallback_calls"] >= 1


def test_ticket_sets_are_per_shape_and_stream_and_bounded(monkeypatch):
    torch, base, fused = load_modules(monkeypatch, {"fused_splitk": True})
    fused.install()
    device = torch.device("cpu")
    a = fused.tickets(336, 10240, device, 11)
    b = fused.tickets(320, 10240, device, 10)
    again = fused.tickets(336, 10240, device, 11)
    assert a.data_ptr() == again.data_ptr() and a.data_ptr() != b.data_ptr()
    assert fused.tickets(336, 10240, device, 2 * 188 + 1) is None  # more tiles than a split-K call can have
    monkeypatch.setattr(fused, "_stream_key", lambda device: 12345)  # another stream: its own set
    assert fused.tickets(336, 10240, device, 11).data_ptr() != a.data_ptr()
    monkeypatch.setattr(fused, "ARENA_TICKETS", fused._STATE["next"])  # full: None, the caller falls back
    monkeypatch.setattr(fused, "_stream_key", lambda device: 999)
    assert fused.tickets(336, 10240, device, 11) is None


def test_microbench_sweeps_run_under_the_interpreter(monkeypatch):
    """The in-job sweeps' code paths on tiny cases (CPU, Triton interpreter; timing stubbed): the GPU run then only
    has to time them."""
    torch, base, fused = load_modules(monkeypatch, {"fused_splitk": True})
    fused.install()
    import torch.nn.functional as F

    from arc_infra import phase6_microbench as mb

    monkeypatch.setattr(base, "_pinned_bf16", lambda x, w: F.linear(x, w))
    dense = mb.dense_sweep(60, device="cpu", timer=lambda fn, calls: (fn(0), 1.0)[1], buckets=(16,),
                           shapes=(("bf16", 96, 2560), ("int8", 512, 2560)), modules=(base, fused), max_copies=1)
    assert dense["fused_installed"] and len(dense["rows"]) == 2
    for row in dense["rows"]:
        assert "pinned_us" in row and len(row["tiles"]) == len(base.MID_CANDIDATES)
        split_tiles = [t for t in row["tiles"] if "fused_bitwise" in t]
        assert split_tiles and all(t["fused_bitwise"] for t in split_tiles)
    spec = importlib.util.spec_from_file_location("vllm.arc_kv8", ROOT / "src/arc_infra/kv8_runtime.py")
    kv8 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vllm.arc_kv8", kv8)
    sys.modules["vllm"].arc_kv8 = kv8
    spec.loader.exec_module(kv8)
    qsa = mb.qsa_sweep(page=64, budget_seconds=60, device="cpu", timer=lambda fn: (fn(), 1.0)[1], rows_list=(4,),
                       topks=(203,), grid=[(16, 4, 2), (64, 1, 2)], min_blocks=8)
    (row,) = qsa["rows"]
    assert row["pinned_err_fp32"] < 0.05 and len(row["grid"]) == 2
    assert all(g["rel_to_pinned"] < 1e-2 and g["err_fp32"] < 0.05 for g in row["grid"])
