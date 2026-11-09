"""Kernel work, phase 5: the stack overlay (asynchronous PLE short-conv metadata copies), the mid-M dense option, the
stack arm, its smoke build and its Modal test. CPU only: the Triton tiles are timed and checked at load on the GPU
(``ARC_DENSE_MID``); the tuner's selection rule is checked here with torch stand-ins when torch is installed."""

import ast
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arc_infra import dense_kernels_patch as dense
from arc_infra import stack_kernels_patch as stack

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))


def synthetic_site(tmp_path, monkeypatch):
    """The short-conv target as a tiny file holding its import and exactly its anchors."""
    site = tmp_path / "site"
    target = stack.SHORT_CONV
    a, b, c, d = (anchor for anchor, _, _ in target["edits"])  # indented 8, 12, 12 and 16 spaces
    text = (target["requires"] + "\n\ndef build(self, spec_req_idx_cpu, non_spec_req_idx_cpu, decode_req_idx_cpu,\n"
            "          query_start_loc, num_accepted_tokens, num_computed_tokens, req_group):\n"
            "    if True:\n" + a + "        if True:\n" + b + "            x = (\n" + c + "            )\n"
            "            if True:\n" + d)
    data = text.encode()
    compile(data, target["relative"], "exec")
    path = site / target["relative"]
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    monkeypatch.setitem(target, "sha256", hashlib.sha256(data).hexdigest())
    return site, path, data


# ----------------------------------------------------------------------------- stack overlay
def test_stack_settings_are_validated():
    assert stack.validate(None) is None and stack.validate({}) is None
    assert stack.validate({"async_h2d": False}) is None
    assert stack.validate(stack.PHASE5) == {"async_h2d": True}
    assert stack.targets(stack.PHASE5) == [stack.SHORT_CONV]
    assert stack.validate({"async_h2d": "toggle"}) == {"async_h2d": "toggle"}  # smoke only
    for bad in ({"async_h2d": 1}, {"fused": True}, ["async_h2d"], {"async_h2d": "on"}):
        with pytest.raises(ValueError):
            stack.validate(bad)


def test_toggle_patch_routes_the_copies_through_the_flag_helper(tmp_path, monkeypatch):
    site, path, _ = synthetic_site(tmp_path, monkeypatch)
    stack.apply(site, tmp_path / "evidence", {"async_h2d": "toggle"})
    text = path.read_text()
    assert text.count("_arc_stack_h2d(") == 5 + 1 and ".to(query_start_loc.device)" not in text
    namespace = {}
    helper = text[text.index("def _arc_stack_h2d"):text.index("\n\n\n", text.index("def _arc_stack_h2d"))]
    calls = []
    exec(helper, {"async_tensor_h2d": lambda t, device: calls.append(("async", t, device)) or "async"}, namespace)

    class T:
        def to(self, device):
            calls.append(("blocking", device))
            return "blocking"

    flag = Path(stack.TOGGLE_FLAG)
    flag.unlink(missing_ok=True)
    assert namespace["_arc_stack_h2d"](T(), "cuda") == "async"
    flag.touch()
    try:
        assert namespace["_arc_stack_h2d"](T(), "cuda") == "blocking"
    finally:
        flag.unlink(missing_ok=True)


def test_bench_toggle_runs_every_point_in_both_modes_with_the_same_prompts(tmp_path):
    from arena import throughput_bench as tb

    spec = {"points": [[16384, 1], [64512, 6]], "transcript": None, "prompt_tokens": [], "concurrency": [],
            "h2d_toggle": {"order": ["blocking", "async"], "flag": str(tmp_path / "flag")}}
    points = tb.toggled_points(spec)
    assert [(p["h2d"], p["prompt_tokens"], p["concurrency"], p["seed_index"]) for p in points] == [
        ("blocking", 16384, 1, 0), ("blocking", 64512, 6, 1), ("async", 16384, 1, 0), ("async", 64512, 6, 1)]
    assert tb.toggled_points({**spec, "h2d_toggle": None}) == tb.grid_points(spec)
    tb.set_h2d_flag(spec["h2d_toggle"]["flag"], True)
    assert (tmp_path / "flag").exists()
    tb.set_h2d_flag(spec["h2d_toggle"]["flag"], False)
    tb.set_h2d_flag(spec["h2d_toggle"]["flag"], False)
    assert not (tmp_path / "flag").exists()


def test_stack_patch_makes_every_copy_asynchronous_and_is_idempotent(tmp_path, monkeypatch):
    site, path, original = synthetic_site(tmp_path, monkeypatch)
    identity = stack.apply(site, tmp_path / "evidence", stack.PHASE5)
    text = path.read_text()
    assert ".to(query_start_loc.device)" not in text and ".to(num_accepted_tokens.device)" not in text
    assert ".to(num_computed_tokens.device)" not in text
    assert text.count("async_tensor_h2d(") == 5  # the five copies (the import line is checked, not added)
    assert json.loads((tmp_path / "evidence" / stack.MARKER).read_text()) == identity
    assert stack.apply(site, tmp_path / "evidence", stack.PHASE5) == identity  # already patched: accepted
    path.write_bytes(original)
    assert stack.apply(site, tmp_path / "evidence", stack.PHASE5) == identity  # re-extracted: patched again
    path.write_text(path.read_text() + "\n# tampered")
    with pytest.raises(RuntimeError, match="modified"):
        stack.apply(site, tmp_path / "evidence", stack.PHASE5)
    with pytest.raises(ValueError, match="unpatched"):
        stack.apply(site, tmp_path / "evidence", None)
    with pytest.raises(RuntimeError, match="unrecognized"):
        stack.patched_source(stack.SHORT_CONV, b"x = 1\n")


def test_exact_pinned_short_conv_when_available():
    source = PINNED_SITE / stack.SHORT_CONV["relative"]
    if not source.is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE); the hash guard is checked if supplied")
    patched = stack.patched_source(stack.SHORT_CONV, source.read_bytes()).decode()
    tree = ast.parse(patched)
    build = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build"
                 and "spec_req_idx_cpu" in ast.unparse(n))
    body = ast.unparse(build)
    # No blocking host-to-device copy is left on the spec-decode path of the builder.
    assert ".to(query_start_loc.device)" not in body and ".to(num_accepted_tokens.device)" not in body
    assert ".to(num_computed_tokens.device)" not in body
    assert body.count("async_tensor_h2d(") == 6  # the pinned spec-mask copy plus the five patched ones


def test_serving_applies_the_stack_only_when_requested(tmp_path, monkeypatch):
    from arena import serving

    applied = []
    monkeypatch.setattr(serving, "apply", lambda *a: applied.append("layout"))
    monkeypatch.setattr(serving, "apply_stack_kernels",
                        lambda site, working, spec: applied.append(("stack", spec)) or {"settings": spec})
    monkeypatch.setattr(serving, "clear_compile_cache", lambda root: applied.append("clear") or [])

    def recipe():
        return NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ["serve"],
                  RUNTIME_ROOT=tmp_path / "runtime", CACHE_ROOT=tmp_path / "cache", MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X")

    control = recipe()
    assert "stack_kernels" not in serving.configure(control, tmp_path)
    control.verify_and_extract_runtime()
    assert applied == ["layout"]
    applied.clear()
    patched = recipe()
    receipt = serving.configure(patched, tmp_path, stack_kernels=stack.PHASE5)
    patched.verify_and_extract_runtime()
    # Host code outside the compiled graph: no compile-cache clear, no server flag.
    assert applied == ["layout", ("stack", stack.PHASE5)] and receipt["stack_kernels"] == stack.PHASE5
    assert patched.server_command() == ["serve"]
    assert json.loads((tmp_path / "stack-kernels-start.json").read_text())["identity"]["settings"]
    assert serving.lab_overrides({"ARENA_STACK_KERNELS": "off"}) == {"stack_kernels": None}
    assert serving.lab_overrides({"ARENA_STACK_KERNELS": json.dumps(stack.PHASE5)}) == {"stack_kernels": stack.PHASE5}
    with pytest.raises(ValueError):
        serving.lab_overrides({"ARENA_STACK_KERNELS": json.dumps({"x": True})})


# ----------------------------------------------------------------------------- mid-M dense option
PLAN = {"bf16": {"336,10240": {"24": [16, 256, 4, 3], "8": None}}, "int8": {"16384,2560": {"32": [32, 128, 4, 4]}}}


def test_mid_m_settings_extend_phase2_without_changing_it():
    assert dense.validate(dense.PHASE2) == {"skinny": True, "int8_weights": list(dense.INT8_FAMILIES)}  # no mid_m key
    assert dense.validate({**dense.PHASE2, "mid_m": False}) == dense.validate(dense.PHASE2)
    assert dense.validate({**dense.PHASE2, "mid_m": True}) == {**dense.validate(dense.PHASE2), "mid_m": True}
    # The arm serves the plan the smoke measured (a fixed plan: error-checked at start-up, never re-timed).
    phase5 = dense.validate(dense.PHASE5)
    assert phase5["mid_m"] == dense.validate_mid_plan(dense.MID_PLAN_PHASE5) and phase5["mid_m"]["bf16"]["336,10240"]["16"]
    assert all(set(b) <= {"24", "32"} for b in phase5["mid_m"]["int8"].values())
    fixed = dense.validate({**dense.PHASE2, "mid_m": PLAN})["mid_m"]
    assert fixed == {"bf16": {"336,10240": {"8": None, "24": [16, 256, 4, 3]}},
                     "int8": {"16384,2560": {"32": [32, 128, 4, 4]}}}
    assert list(fixed["bf16"]["336,10240"]) == ["8", "24"]  # canonical bucket order
    for bad in ({"bf16": {"336x10240": {}}}, {"bf16": {"336,10240": {"12": None}}}, {"fp8": {}},
                {"bf16": {"336,10240": {"24": [16, 256, 4]}}}, {"bf16": {"336,10240": {"24": [16, 0, 4, 3]}}}, {}):
        with pytest.raises(ValueError):
            dense.validate({**dense.PHASE2, "mid_m": bad})
    with pytest.raises(ValueError, match="mid_m"):
        dense.validate({"mid_m": True})


def runtime_constants():
    tree = ast.parse((ROOT / "src/arc_infra/dense_kernels_runtime.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                if all(isinstance(n, (ast.BinOp, ast.Constant, ast.operator)) for n in ast.walk(node.value)):
                    values[node.targets[0].id] = eval(compile(ast.Expression(node.value), "constant", "eval"))
    return values


def test_mid_m_runtime_constants_match_the_served_graphs():
    values = runtime_constants()
    assert tuple(values["MID_M_BUCKETS"]) == dense.MID_M_BUCKETS
    # The served decode CUDA-graph capture sizes above 4 (vLLM log: [1, 2, 4, 8, 16, 24, 32]).
    assert set(values["MID_M_BUCKETS"]) == {8, 16, 24, 32} and values["MID_MIN_M"] == 5
    assert set(values["MID_INT8_BUCKETS"]) <= set(values["MID_M_BUCKETS"]) and min(values["MID_INT8_BUCKETS"]) > 16
    # Every mid shape is a planned skinny shape except HC up (K = 320), which is cuBLAS at every M today.
    assert set(values["MID_BF16_SHAPES"]) - set(values["SKINNY_SHAPES"]) == {(10240, 320)}
    assert not set(values["MID_BF16_SHAPES"]) & {s for _, s in values["LINEAR_FAMILIES"].values()}
    assert all(len(c) == 4 and c[0] in (16, 32, 64) and c[1] in (64, 128, 256) for c in values["MID_CANDIDATES"])
    assert values["MID_BENCH_BYTES"] >= 128 << 20  # above the 128 MiB L2: no reuse between timed calls
    assert 0 < values["MID_MIN_GAIN"] < 0.1 and 1 < values["MID_ERROR_FACTOR"] <= 1.5


def test_tuning_runs_in_the_kernel_warmup_after_the_kv_cache_is_sized():
    """Tuning at load cost 0.41 GiB of 'model loading' memory and the sixth 64K window (kernel-stack-phase5-v1);
    it now runs only from the warmup units, which vLLM runs after sizing the KV cache and before graph capture."""
    tree = ast.parse((ROOT / "src/arc_infra/dense_kernels_runtime.py").read_text())
    callers = {fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
               for call in ast.walk(fn) if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "tune_mid"}
    assert callers == {"_warm_shape", "_warm_mid_shape"}


def load_runtime(monkeypatch, config):
    torch = pytest.importorskip("torch")
    if "triton" not in sys.modules and importlib.util.find_spec("triton") is None:  # GPU-only; the tuner is torch
        stub = types.ModuleType("triton")
        stub.jit = lambda f=None, **k: f if f is not None else (lambda g: g)
        stub.cdiv = lambda a, b: -(-a // b)
        language = types.ModuleType("triton.language")
        language.constexpr = object
        stub.language = language
        monkeypatch.setitem(sys.modules, "triton", stub)
        monkeypatch.setitem(sys.modules, "triton.language", language)
    spec = importlib.util.spec_from_file_location("arc_dense_kernels_test", ROOT / "src/arc_infra/dense_kernels_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime._STATE["config"] = None
    monkeypatch.setattr(runtime, "CONFIG_PATH", Path("/nonexistent"))
    runtime._STATE["config"] = {"skinny": True, "int8_weights": [], "mid_m": config}
    if isinstance(config, dict):
        for kind in ("bf16", "int8"):
            for key, buckets in config.get(kind, {}).items():
                n, k = (int(v) for v in key.split(","))
                runtime._STATE["mid"][kind][(n, k)] = {int(m): None if c is None else runtime._mid_tile(int(m), c)
                                                       for m, c in buckets.items()}
    return torch, runtime


def fake_kernels(torch, runtime, monkeypatch, speed, error=None, fail=()):
    """CPU stand-ins: exact GEMMs, a timing table (pinned or tile -> us) and optional per-tile error."""
    timed = {"pinned": speed["pinned"]}

    def gemm(x, w, s, tile=None):
        y = (x.float() @ (w.float() * s[:, None]).t())
        if tile is not None and tuple(tile[1:]) in fail:
            raise RuntimeError("out of shared memory")
        if tile is not None and error and tuple(tile[1:]) in error:
            y = y * (1 + error[tuple(tile[1:])])
        timed["last"] = speed.get(tuple(tile[1:]) if tile else "pinned", 99.0)
        return y.to(torch.bfloat16)

    monkeypatch.setattr(runtime, "w8a16_gemm", gemm)
    monkeypatch.setattr(runtime, "_pinned_bf16", lambda x, w: (timed.__setitem__("last", speed["pinned"]),
                                                              (x.float() @ w.float().t()).to(torch.bfloat16))[1])

    def time_graph(fn, calls, replays=1):
        fn(0)
        return timed["last"]

    monkeypatch.setattr(runtime, "time_graph", time_graph)


def test_tuner_keeps_the_pinned_path_unless_a_tile_is_faster_and_as_accurate(monkeypatch):
    torch, runtime = load_runtime(monkeypatch, True)
    candidates = runtime.MID_CANDIDATES
    speed = {"pinned": 10.0, candidates[0]: 9.9, candidates[1]: 6.0, candidates[2]: 5.0, candidates[3]: 8.0}
    # The fastest tile is wrong by 1% (> 1.25x the pinned BF16 rounding): the next fastest is chosen.
    fake_kernels(torch, runtime, monkeypatch, speed, error={candidates[2]: 0.01}, fail={candidates[3]})
    weight = torch.randn(96, 2560).to(torch.bfloat16)
    runtime.tune_mid("bf16", weight, None, "layers.0.linear_attn.in_proj_ba")
    plan = runtime._STATE["mid"]["bf16"][(96, 2560)]
    assert plan == {m: runtime._mid_tile(m, candidates[1]) for m in runtime.MID_M_BUCKETS}
    assert runtime.mid_tile("bf16", 96, 2560, 24) == (32, *candidates[1])
    assert runtime.mid_tile("bf16", 96, 2560, 8) == (16, *candidates[1])
    assert runtime.mid_tile("bf16", 96, 2560, 4) is None and runtime.mid_tile("bf16", 96, 2560, 33) is None
    assert runtime.mid_tile("bf16", 336, 10240, 24) is None  # an untuned shape stays on the pinned path
    # Once per shape (the MTP draft's layers of the same shape reuse the plan).
    runtime._STATE["mid"]["bf16"][(96, 2560)] = {"sentinel": 1}
    runtime.tune_mid("bf16", weight, None, "mtp.layers.0.linear_attn.in_proj_ba")
    assert runtime._STATE["mid"]["bf16"][(96, 2560)] == {"sentinel": 1}
    # A candidate within 3% of the pinned time does not replace it.
    torch, runtime = load_runtime(monkeypatch, True)
    fake_kernels(torch, runtime, monkeypatch, {"pinned": 10.0, **{c: 9.8 for c in candidates}})
    runtime.tune_mid("bf16", weight, None, "x")
    assert set(runtime._STATE["mid"]["bf16"][(96, 2560)].values()) == {None}
    # A tuning failure (e.g. the pinned path cannot be graph-captured) never stops the server: pinned everywhere.
    torch, runtime = load_runtime(monkeypatch, True)
    fake_kernels(torch, runtime, monkeypatch, {"pinned": 10.0})
    monkeypatch.setattr(runtime, "time_graph", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("capture")))
    runtime.tune_mid("bf16", weight, None, "x")
    assert runtime._STATE["mid"]["bf16"][(96, 2560)] == {} and runtime.mid_tile("bf16", 96, 2560, 24) is None


def test_int8_tuning_covers_only_the_mid_buckets_and_a_fixed_plan_is_only_checked(monkeypatch):
    torch, runtime = load_runtime(monkeypatch, True)
    candidates = runtime.MID_CANDIDATES
    fake_kernels(torch, runtime, monkeypatch, {"pinned": 30.0, candidates[1]: 20.0})
    q = torch.randint(-127, 128, (64, 2560), dtype=torch.int8)
    s = torch.rand(64) / 100
    runtime.tune_mid("int8", q, s, "layers.0.linear_attn.in_proj_qkvz")
    plan = runtime._STATE["mid"]["int8"][(64, 2560)]
    assert set(plan) == set(runtime.MID_INT8_BUCKETS) and plan[24] == (32, *candidates[1])
    assert runtime.mid_tile("int8", 64, 2560, 16) is None and runtime.mid_tile("int8", 64, 2560, 20) == plan[24]
    # Fixed plan: no timing; a planned tile that fails the error check falls back to the pinned path.
    fixed = {"bf16": {"96,2560": {"24": list(candidates[0]), "32": list(candidates[1])}}}
    torch, runtime = load_runtime(monkeypatch, fixed)
    fake_kernels(torch, runtime, monkeypatch, {"pinned": 1.0}, error={candidates[1]: 0.02})
    monkeypatch.setattr(runtime, "time_graph", lambda *a, **k: pytest.fail("a fixed plan is not timed"))
    runtime.tune_mid("bf16", torch.randn(96, 2560).to(torch.bfloat16), None, "x")
    assert runtime._STATE["mid"]["bf16"][(96, 2560)] == {8: None, 16: None, 24: (32, *candidates[0]), 32: None}


# ----------------------------------------------------------------------------- arm, smoke build, Modal test
def load_builder():
    spec = importlib.util.spec_from_file_location("phase5_builder", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def test_stack_arm_is_the_phase4_arm_plus_the_stack():
    from arena.variants import COMPOSED, SERVING_CHANGED, VARIANTS, apply_variant

    for name in ("fixed-2x-mtp3-wide-stack", "fixed-2x-depthwise-mtp3-stack"):
        assert name in VARIANTS and name in COMPOSED and name in SERVING_CHANGED
    phase4 = apply_variant(CONTROL, "fixed-2x-mtp3-wide-kernels")
    arm = apply_variant(CONTROL, "fixed-2x-mtp3-wide-stack")
    assert arm["server"] == {**phase4["server"], "dense_kernels": dense.PHASE5, "stack_kernels": stack.PHASE5}
    depth = apply_variant(CONTROL, "fixed-2x-depthwise-mtp3-stack")
    assert depth["depthwise_serving_source"] == "fixed-2x-mtp3-wide-stack" and depth["server"] == arm["server"]
    assert depth["arena_variant"]["reference_difference"]["server.stack_kernels"]["after"] == stack.PHASE5


def test_stack_smoke_build(tmp_path):
    builder = load_builder()
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("step 1\n" * 1000)
    output, study = builder.build("kernel-stack", output=tmp_path / "smoke", paired_30m=True, throughput_bench=True,
                                  arena_variant="fixed-2x-mtp3-wide-stack", kernel_smoke=True,
                                  bench_transcript=str(transcript))
    from arena import serving
    from arena.variants import apply_variant

    assert [t["name"] for t in study["trials"]] == ["stack-mtp3", "stack-mtp0-kv8"]
    assert study["trials"][0]["server_env"] == {"ARENA_TORCH_PROFILER": "1"}
    # The MTP-3 arm: the phase-5 server with the host-copy item switchable at run time on the same points.
    assert study["server"]["stack_kernels"] == {"async_h2d": "toggle"} and study["server"]["dense_kernels"] == dense.PHASE5
    grid = study["throughput_bench"]
    assert grid["h2d_toggle"] == {"order": ["blocking", "async"], "flag": stack.TOGGLE_FLAG}
    assert grid["points"] == [[16384, 1], [16384, 6], [64512, 1], [64512, 6]] and grid["transcript"]["concurrency"] == [1, 6]
    assert grid["transcript"]["prompt_tokens"] == [16384] and "transcript_points" not in grid
    assert grid["profile"]["concurrency"] == [1, 6] and [s["expect"] for s in grid["sanity"]] == ["391", "Paris"]
    assert (output / grid["transcript"]["file"]).read_text() == transcript.read_text()
    # The MTP-0 arm: exactly the fixed-2x-mtp0-wide-kv8-stack server through the lab overrides; no toggle.
    arm = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kv8-stack")["server"]
    lab = serving.lab_overrides(study["trials"][1]["server_env"])
    assert lab == {"mtp_tokens": 0, "torch_profiler": True, "dense_kernels": arm["dense_kernels"],
                   "serving_kernels": arm["serving_kernels"], "stack_kernels": arm["stack_kernels"]}
    mtp0_grid = study["trials"][1]["throughput_bench"]
    assert mtp0_grid["h2d_toggle"] is None and mtp0_grid["transcript"] is None
    assert {c for _, c in mtp0_grid["points"]} == {1, 6, 14} and mtp0_grid["profile"]["concurrency"] == [14]
    with pytest.raises(ValueError, match="transcript"):
        builder.build("bad", output=tmp_path / "bad", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp3-wide-stack", kernel_smoke=True)
    # The phase-4 smoke keeps its single trial.
    _, phase4 = builder.build("kernel-smoke", output=tmp_path / "p4", paired_30m=True, throughput_bench=True,
                              arena_variant="fixed-2x-mtp3-wide-kernels", kernel_smoke=True,
                              bench_transcript=str(transcript))
    assert phase4["trials"] == [{"name": "serving-bench", "kind": "serving-bench"}] and "stack_smoke" not in phase4


def test_modal_stack_test_fits_the_budget_and_the_control_start_check():
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "kernel-stack")
    assert ns["arena_arguments"]("kernel-stack") == []
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 4.0  # the GPU budget for the stack
    builder = load_builder()
    # The MTP-0 trial starts after ~1,550 s (two setups, the toggled MTP-3 bench) and needs wall + 480 s left.
    notebook = timeout - 60
    assert notebook - 1550 >= 120 + 480 and builder.STACK_SMOKE_GRID["points"][-1] == [64512, 6]


# ----------------------------------------------------------------------------- hero smoke (28 September)
def test_hero_arm_and_smoke_build(tmp_path):
    from arena.variants import apply_variant

    hero = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kv8-hero")
    stack_arm = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kv8-stack")
    assert hero["server"] == {**stack_arm["server"],
                              "serving_kernels": {**stack_arm["server"]["serving_kernels"], "moe_backend": "marlin"}}
    assert hero["server"]["dense_kernels"]["mid_m"] == dense.MID_PLAN_PHASE5 and hero["active_limit"] == 14
    builder = load_builder()
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("step 1\n" * 1000)
    reference = tmp_path / "gate.json"
    reference.write_text(json.dumps({"role": "gate", "trial": "kernel-gate", "server_env": {},
                                     "server": {"dense_kernels": dense.PHASE2}, "greedy": {}, "teacher_forced": {}}))
    output, study = builder.build("hero", output=tmp_path / "hero", paired_30m=True, throughput_bench=True,
                                  arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_smoke=True,
                                  bench_transcript=str(transcript), kv8_reference=str(reference))
    assert study["trials"] == [{"name": "hero", "kind": "serving-bench", "server_env": {"ARENA_TORCH_PROFILER": "1"}}]
    grid = study["throughput_bench"]
    assert grid["transcript"]["concurrency"] == [1, 6, 14] and {c for _, c in grid["points"]} == {1, 6, 14}
    assert grid["profile"]["concurrency"] == [1, 14] and (output / grid["quality"]["reference"]).is_file()
    with pytest.raises(ValueError, match="hero"):
        builder.build("bad", output=tmp_path / "bad", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero", bench_transcript=str(transcript),
                      kv8_reference=str(reference))
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (ns["validation_timeout"]("arena", "hero-smoke") + 60) * rate * 1.15 <= 2.5
    assert ns["arena_arguments"]("hero-smoke") == ["--trial", "hero"]
