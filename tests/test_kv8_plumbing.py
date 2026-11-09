"""8-bit QSA KV cache: overlay option, hash-guarded qsa.py patch, BF16 byte-identity when off, the kv8 arm and its
focus fit, the smoke build, its bench checks and the Modal test (docs/validation-20260927/kv-cache-8bit.md).
CPU only; the kernels are in tests/test_kv8_kernels.py."""

import ast
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from test_arena_mtp0_wide import served_layout
from test_arena_throughput_bench import FakeServer
from test_serving_kernels import synthetic_site

from arc_infra import dense_kernels_patch as dense
from arc_infra import serving_kernels_patch as patch
from arena import kv8_checks
from arena import throughput_bench as tb
from arena.cache import BlockBudget
from arena.capacity import focus_fit
from arena.variants import COMPOSED, KV8_FOCUS_SLOTS, SERVING_CHANGED, VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
KV8 = {**patch.PHASE4, "kv_cache_8bit": "int8"}
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))


# ----------------------------------------------------------------------------- option and patch
def test_option_is_validated_and_off_leaves_the_recorded_settings_unchanged():
    assert patch.validate(patch.PHASE4) == patch.PHASE4 and "kv_cache_8bit" not in patch.validate(patch.PHASE4)
    assert patch.validate({**patch.PHASE4, "kv_cache_8bit": None}) == patch.PHASE4
    spec = patch.validate(KV8)
    assert spec == KV8 and list(spec) == [k for k in patch.OPTIONS if k not in patch.TWEAK_OPTIONS + ("kv_cache_4bit", "kv8_prefill", "expert_int3", "indexer_fp8", "pin_overlap", "draft_moe_backend")]
    assert patch.targets(spec) == [patch.MTP, patch.B12X, patch.SAMPLER, patch.STATES, patch.QSA]
    assert patch.server_flags(spec) == ["--mamba-ssm-cache-dtype", "bfloat16", "--kv-cache-dtype", "int8_per_token_head"]
    only = patch.validate({"kv_cache_8bit": "fp8"})
    assert patch.targets(only) == [patch.QSA]
    assert patch.server_flags(only) == ["--kv-cache-dtype", "fp8_per_token_head"]
    for bad in ({"kv_cache_8bit": "int4"}, {"kv_cache_8bit": True}, {"kv_cache_8bit": "fp8_per_token_head"}):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_kv8_patch_installs_its_helper_only_with_the_option(tmp_path, monkeypatch):
    site, originals = synthetic_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "evidence", {"kv_cache_8bit": "int8"})
    assert set(identity["files"]) == {patch.QSA["relative"]}
    assert identity["server_flags"] == ["--kv-cache-dtype", "int8_per_token_head"]
    assert (site / patch.KV8_HELPER).read_bytes() == (ROOT / "src/arc_infra/kv8_runtime.py").read_bytes()
    assert identity["kv8_helper_sha256"]
    text = (site / patch.QSA["relative"]).read_text()
    compile(text, "qsa.py", "exec")
    for needle in ('"int8_per_token_head", "fp8_per_token_head"', "_arc_kv8_customize_spec(spec)",
                   "_arc_kv8_kv_cache_shape(", "_arc_kv8_impl_init_args(args, kwargs)",
                   "_arc_kv8_dtype is None", "def do_kv_cache_update", "_arc_kv8_attention(",
                   "not _arc_kv8_allowed(\n            cache_config.cache_dtype", "not _arc_kv8_allowed(self.kv_cache_dtype)"):
        assert needle in text, needle
    for target in patch.TARGETS:
        if target is not patch.QSA:
            assert (site / target["relative"]).read_bytes() == originals[target["relative"]]
    assert patch.apply(site, tmp_path / "evidence", {"kv_cache_8bit": "int8"}) == identity  # idempotent
    (site / patch.KV8_HELPER).write_text("# tampered\n")
    with pytest.raises(RuntimeError, match="modified"):
        patch.apply(site, tmp_path / "evidence", {"kv_cache_8bit": "int8"})


def test_phase4_overlay_without_kv8_is_byte_identical_to_before(tmp_path, monkeypatch):
    # The recorded phase-4 arm: same files, same helper, same settings file, no kv8 helper, no QSA patch.
    site, originals = synthetic_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "evidence", patch.PHASE4)
    assert not (site / patch.KV8_HELPER).exists() and "kv8_helper_sha256" not in identity
    assert (site / patch.QSA["relative"]).read_bytes() == originals[patch.QSA["relative"]]
    assert json.loads((site / patch.SETTINGS).read_text()) == patch.PHASE4
    assert identity["server_flags"] == ["--mamba-ssm-cache-dtype", "bfloat16"]
    assert set(identity["files"]) == {t["relative"] for t in (patch.MTP, patch.B12X, patch.SAMPLER, patch.STATES)}


def test_exact_pinned_qsa_source_when_available():
    path = PINNED_SITE / patch.QSA["relative"]
    if not path.is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    patched = patch.patched_source(patch.QSA, path.read_bytes()).decode()
    tree = ast.parse(patched)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    backend = {n.name for n in classes["Qwen3_8FlashNextQSAFlashAttentionBackend"].body if isinstance(n, ast.FunctionDef)}
    assert {"customize_spec", "get_kv_cache_shape", "get_name"} <= backend
    impl = {n.name: n for n in classes["Qwen3_8FlashNextQSAFlashAttentionImpl"].body if isinstance(n, ast.FunctionDef)}
    init = ast.unparse(impl["__init__"])
    assert init.index("_arc_kv8_impl_init_args(args, kwargs)") < init.index("super().__init__(*args, **kwargs)")
    forward = ast.unparse(impl["forward_qsa"])
    assert forward.index("output.zero_()") < forward.index("_arc_kv8_attention(") < forward.index(
        "qsa_sparse_paged_attention(")
    layer = ast.unparse(next(n for n in classes["Qwen3_8FlashNextQSAAttention"].body
                             if isinstance(n, ast.FunctionDef) and n.name == "__init__"))
    assert layer.count("_arc_kv8_allowed(") == 2
    # The BF16 guards keep their messages: any other quantized dtype still fails.
    assert patched.count("Qwen3.8-Flash-Next QSA requires a BF16 main KV cache") == 2
    assert "Qwen3.8-Flash-Next QSA requires BF16 cache storage" in patched


def test_runtime_helper_entry_points_match_the_qsa_patch():
    tree = ast.parse((ROOT / "src/arc_infra/kv8_runtime.py").read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for _, replacement, _ in patch.QSA["edits"]:
        for line in replacement.splitlines():
            if "from vllm.arc_kv8 import" in line:
                assert line.split("import ")[1].split(" as ")[0] in names
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "CACHE_DTYPES"}
    assert values["CACHE_DTYPES"] == patch.KV8_CACHE_DTYPES


# ----------------------------------------------------------------------------- serving plumbing
def test_configure_adds_the_flag_and_records_the_runtime_env_only_with_kv8(tmp_path, monkeypatch):
    from arena import serving

    monkeypatch.setattr(serving, "apply", lambda *a: None)
    monkeypatch.setattr(serving, "apply_dense_kernels", lambda site, working, spec: {})
    monkeypatch.setattr(serving, "apply_serving_kernels", lambda site, working, spec: {"settings": spec})
    monkeypatch.setattr(serving, "clear_compile_cache", lambda root: [])
    env = {"PYTHONPATH": "/site", "CUDA_HOME": "/cuda", "HF_TOKEN": "secret", "TRITON_CACHE_DIR": "/t",
           "LD_LIBRARY_PATH": "/lib", "OTHER": "x", "PATH": "/bin"}

    def recipe():
        return NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ["serve"],
                  runtime_environment=lambda **kw: (dict(env), {"ok": True}),
                  RUNTIME_ROOT=tmp_path / "runtime", CACHE_ROOT=tmp_path / "cache", MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X")

    phase4 = recipe()
    serving.configure(phase4, tmp_path / "p4", serving_kernels=patch.PHASE4)
    assert phase4.runtime_environment()[0] == env and not (tmp_path / "p4").exists()
    assert phase4.server_command() == ["serve", "--mamba-ssm-cache-dtype", "bfloat16"]
    (tmp_path / "kv8").mkdir()
    arm = recipe()
    receipt = serving.configure(arm, tmp_path / "kv8", serving_kernels=KV8)
    assert receipt["serving_kernels"] == KV8
    assert arm.server_command() == ["serve", "--mamba-ssm-cache-dtype", "bfloat16", "--kv-cache-dtype",
                                    "int8_per_token_head"]
    assert arm.runtime_environment(deep_preload_validation=True)[0] == env
    record = json.loads((tmp_path / "kv8" / kv8_checks.RUNTIME_ENV).read_text())
    assert record["env"] == {k: env[k] for k in ("PYTHONPATH", "CUDA_HOME", "TRITON_CACHE_DIR", "LD_LIBRARY_PATH",
                                                 "PATH")}
    clash = recipe()
    clash.server_command = lambda: ["serve", "--kv-cache-dtype", "auto"]
    serving.configure(clash, tmp_path / "kv8", serving_kernels=KV8)
    with pytest.raises(ValueError, match="--kv-cache-dtype"):
        clash.server_command()


def test_microbench_skips_without_the_runtime_record(tmp_path):
    assert kv8_checks.microbench(tmp_path, tmp_path)["status"] == "skipped"


# ----------------------------------------------------------------------------- arm and fit
def test_kv8_arm_is_the_phase4_arm_with_int8_kv_and_more_focus_games():
    arm = apply_variant(CONTROL, "fixed-2x-mtp3-wide-kernels-kv8")
    base = apply_variant(CONTROL, "fixed-2x-mtp3-wide-kernels")
    name = "fixed-2x-mtp3-wide-kernels-kv8"
    assert name in VARIANTS and name in COMPOSED and name in SERVING_CHANGED
    assert arm["server"] == {**base["server"], "serving_kernels": KV8, "max_cudagraph_capture_size": 48}
    assert KV8_FOCUS_SLOTS * 4 <= arm["server"]["max_cudagraph_capture_size"]  # MTP-3 verify rows in the graphs
    assert arm["server"]["dense_kernels"] == dense.PHASE2 and arm["server"]["mtp_tokens"] == 3
    assert arm["active_limit"] == arm["opening_switch_level1_games"] == KV8_FOCUS_SLOTS > 6
    a, b = json.loads(json.dumps(arm)), json.loads(json.dumps(base))
    for key in ("server", "arena_variant", "active_limit", "opening_switch_level1_games"):
        a.pop(key), b.pop(key)
    assert a == b  # 64K window, require_focus_windows_fit, compaction, roster: as the phase-4 arm
    assert arm["require_focus_windows_fit"] is True and arm["windows"] == [65536]
    difference = arm["arena_variant"]["reference_difference"]
    assert difference["server.serving_kernels"] == {"before": None, "after": KV8}
    assert difference["active_limit"] == {"before": 2, "after": KV8_FOCUS_SLOTS}
    assert difference["opening_switch_level1_games"] == {"before": 2, "after": KV8_FOCUS_SLOTS}
    assert difference["server.max_cudagraph_capture_size"] == {"before": 32, "after": 48}
    # The phase-4 arm itself is unchanged by the new option.
    assert base["server"]["serving_kernels"] == patch.PHASE4 and base["active_limit"] == 6


def test_mtp0_kv8_arm_is_kernels_v2_with_the_sampler_int8_kv_and_14_slots_keeping_the_six_clear_switch():
    from arena.variants import MTP0_KV8_FOCUS_SLOTS, MTP0_KV8_SERVING

    arm = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kernels-kv8")
    base = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kernels")
    name = "fixed-2x-mtp0-wide-kernels-kv8"
    assert name in VARIANTS and name in COMPOSED and name in SERVING_CHANGED
    assert arm["server"] == {**base["server"], "serving_kernels": MTP0_KV8_SERVING} and arm["server"]["mtp_tokens"] == 0
    patch.validate(MTP0_KV8_SERVING)
    assert arm["active_limit"] == MTP0_KV8_FOCUS_SLOTS == 14 and arm["opening_switch_level1_games"] == 6
    a, b = json.loads(json.dumps(arm)), json.loads(json.dumps(base))
    for key in ("server", "arena_variant", "active_limit"):
        a.pop(key), b.pop(key)
    assert a == b  # 64K window, fit check, fixed-2x, six-clear switch, roster: as kernels-v2
    assert arm["require_focus_windows_fit"] is True and arm["windows"] == [65536]
    difference = arm["arena_variant"]["reference_difference"]
    assert difference["server.serving_kernels"] == {"before": None, "after": MTP0_KV8_SERVING}
    assert difference["active_limit"] == {"before": 2, "after": 14}


def kv8_layout(num_blocks, block):
    """MTP-3 phase-4 geometry with an 8-bit main KV: the same fixed hybrid state (1 + 4 x 5 pages), lookahead 3."""
    layout = served_layout(3)
    rows = []
    for row in layout["costs"]:
        groups = [math.ceil(row["tokens"] / block), 1, 5, 5, 5, 5]
        rows.append({"tokens": row["tokens"], "blocks": sum(groups), "groups": groups})
    return {**layout, "num_blocks": num_blocks, "costs": rows}


def test_sizing_model_reproduces_the_served_bf16_layouts_and_the_kv8_fit():
    spec = importlib.util.spec_from_file_location("kv8_sizing", ROOT / "scripts/kv8_sizing.py")
    sizing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sizing)
    rows = sizing.table()
    p4 = rows["mtp3_phase4"]
    assert (p4["bf16"]["block_tokens"], p4["bf16"]["blocks"], p4["bf16"]["blocks_per_64k"], p4["bf16"]["fit_64k"]) \
        == (832, 691, 101, 6)
    assert p4["kv8"]["block_tokens"] in (1632, 1664) and p4["kv8"]["fit_64k"] == 9
    assert p4["kv8_rule_nopad"]["fit_64k"] == 9
    assert (rows["mtp0_pinned"]["bf16"]["fit_64k"], rows["mtp0_pinned"]["kv8"]["fit_64k"]) == (7, 12)
    assert rows["mtp0_ac_unpinned"]["kv8"]["fit_64k"] == 14
    for row in rows.values():
        assert row["gain_64k"] >= 0.3  # the task's go/no-go threshold
    # The arm's slots against the projected layout: nine fit, ten do not.
    budget = BlockBudget(kv8_layout(p4["kv8"]["blocks"], p4["kv8"]["block_tokens"]), 0.9)
    fit = focus_fit(budget, 65536, KV8_FOCUS_SLOTS, required=True)
    assert fit["full_window_requests"] == KV8_FOCUS_SLOTS and fit["cost_per_full_window_request"] == 62
    with pytest.raises(RuntimeError, match=f"only {KV8_FOCUS_SLOTS} fit"):
        focus_fit(budget, 65536, KV8_FOCUS_SLOTS + 1, required=True)


# ----------------------------------------------------------------------------- smoke: bench checks, build, Modal
def reference_record(prompts):
    greedy = [{"index": p["index"], "prompt_ids": [1, 2, 3], "prompt_tokens": 3, "finish_reason": "length",
               "tokens": [5, 6, 7], "top": [[[5, -0.1], [6, -2.4]], [[6, -0.2], [9, -1.9]], [[7, -0.3], [8, -1.5]]]}
              for p in prompts["greedy"]]
    teacher = [{"index": t["index"], "tokens": [1, 2, 3, 4], "positions": [[[2, -0.2], [3, -1.8]], [[3, -0.4], [2, -1.2]],
                                                                            [[4, -0.1], [5, -2.6]]]}
               for t in prompts["teacher_forced"]]
    return {"role": "gate", "trial": "kernel-gate", "server_env": {}, "server": {"dense_kernels": dense.PHASE2},
            "greedy": {"rows": greedy}, "teacher_forced": {"rows": teacher}}


async def test_bench_runs_the_quality_check_against_the_reference(tmp_path, monkeypatch):
    prompts = {"greedy": [{"index": 0, "system": "s", "user": "u"}, {"index": 1, "system": "s", "user": "v"}],
               "teacher_forced": [{"index": 0, "text": "a b"}, {"index": 1, "text": "c d"}]}
    (tmp_path / "src/configs/kernel-quality").mkdir(parents=True)
    (tmp_path / "src/configs/kernel-quality/prompts.json").write_text(json.dumps(prompts))
    (tmp_path / "src/configs/kernel-quality/gsm8k.jsonl").write_text("")
    (tmp_path / "src/configs/kernel-quality/ref.json").write_text(json.dumps(reference_record(prompts)))
    server = FakeServer()
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path.endswith("/v1/completions"):
            seen.append(body)
            if body.get("prompt_logprobs"):
                entries = [None] + [{f"{t}": {"logprob": -0.3, "rank": 1}, "9": {"logprob": -1.6, "rank": 2}}
                                    for t in body["prompt"][1:]]
                return httpx.Response(200, json={"choices": [{"prompt_logprobs": entries, "text": ""}]})
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "logprobs": {
                "tokens": ["token_id:5", "token_id:6", "token_id:8"],
                "top_logprobs": [{"token_id:5": -0.1, "token_id:6": -2.5}, {"token_id:6": -0.2, "token_id:9": -1.8},
                                 {"token_id:8": -0.4, "token_id:7": -1.1}]}}]})
        return server.handler(request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(*a, **{**k, "transport": transport}))
    monkeypatch.setattr(tb.GpuMonitor, "run", lambda self: asyncio.sleep(3600))
    (tmp_path / "server.log").write_text("worker ARC_CACHE_LAYOUT " + json.dumps(kv8_layout(673, 1632)) + "\n")
    study = {"kv_margin": 0.9, "generation_slots": 25, "completion_cap": 16384, "reasoning_effort": "xhigh",
             "temperature": 0.7, "server": {"mtp_tokens": 3}, "active_limit": KV8_FOCUS_SLOTS, "windows": [65536],
             "throughput_bench": {"prompt_tokens": [], "concurrency": [], "output_tokens": 16, "shared_prefix": None,
                                  "points": [], "kv8_microbench": True,
                                  "quality": {"reference": "configs/kernel-quality/ref.json", "repeat_teacher": True}}}
    report = await tb.run_bench(study, {"name": "serving-bench", "kind": "serving-bench"}, output=tmp_path / "out",
                                server_dir=tmp_path, base_url="http://model/v1", transport=transport,
                                log=lambda *a, **k: None, source=tmp_path / "src")
    assert report["focus_fit"]["full_window_requests"] == KV8_FOCUS_SLOTS
    quality = report["quality"]
    assert not quality["errors"]
    greedy = quality["vs_reference"]["greedy"]
    assert greedy["prompts"] == 2 and greedy["first_divergence"] == [2, 2] and greedy["exact_match"] == 0
    teacher = quality["vs_reference"]["teacher_forced"]
    assert teacher["positions"] == 6 and teacher["top1_agreement"] == 1.0 and teacher["mean_kl"] > 0
    assert quality["self_repeat"]["teacher_forced"]["mean_kl"] == 0
    # The reference's own token ids were sent: no tokenization on this server.
    assert [b["prompt"] for b in seen if "logprobs" in b] == [[1, 2, 3]] * 2
    assert [b["prompt"] for b in seen if b.get("prompt_logprobs")] == [[1, 2, 3, 4]] * 4
    assert report["kv8_microbench"]["status"] == "skipped"  # no runtime record in this test
    rows = json.loads((tmp_path / "out/kv8-quality.json").read_text())
    assert len(rows["teacher_forced_repeat"]["rows"]) == 2


def load_builder():
    spec = importlib.util.spec_from_file_location("kv8_builder", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def test_kv8_smoke_build(tmp_path):
    builder = load_builder()
    prompts = json.loads((ROOT / "data/kernel_quality/prompts.json").read_text())
    reference = tmp_path / "kernel-quality.json"
    reference.write_text(json.dumps({**reference_record(prompts), "gsm8k": {"rows": [1] * 3}, "bench": {}}))
    output, study = builder.build("kv8-smoke", output=tmp_path / "smoke", paired_30m=True, throughput_bench=True,
                                  arena_variant="fixed-2x-mtp3-wide-kernels-kv8", kv8_smoke=True,
                                  kv8_reference=str(reference))
    assert study["trials"] == [{"name": "serving-bench", "kind": "serving-bench"}]
    assert study["server"]["serving_kernels"] == KV8 and study["active_limit"] == KV8_FOCUS_SLOTS
    grid = study["throughput_bench"]
    assert [KV8_FOCUS_SLOTS] == [s for p, s in grid["points"] if p == 64512 and s > 6]
    assert {tuple(p) for p in grid["points"]} >= {(64512, 1), (64512, 6), (16384, 1)}
    assert grid["kv8_microbench"] is True and [s["expect"] for s in grid["sanity"]] == ["391", "Paris"]
    trimmed = json.loads((output / grid["quality"]["reference"]).read_text())
    assert set(trimmed) == {"role", "trial", "server_env", "server", "greedy", "teacher_forced"}
    assert (output / "configs/kernel-quality/prompts.json").read_bytes() == (
        ROOT / "data/kernel_quality/prompts.json").read_bytes()
    assert (output / "src/arc_infra/kv8_runtime.py").is_file()
    assert study["notebook_seconds"] <= 1500 - 60
    with pytest.raises(ValueError, match="8-bit KV smoke"):
        builder.build("bad", output=tmp_path / "bad", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp3-wide-kernels-kv8")
    with pytest.raises(ValueError, match="BF16 reference"):
        builder.build("bad2", output=tmp_path / "bad2", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp3-wide-kernels-kv8", kv8_smoke=True)
    with pytest.raises(ValueError, match="smoke"):
        builder.build("bad3", output=tmp_path / "bad3", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp3-wide-kernels", kv8_smoke=True, kv8_reference=str(reference))


def test_modal_kv8_smoke_budget():
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "kv8-smoke")
    assert ns["arena_arguments"]("kv8-smoke") == ["--trial", "serving-bench"]
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 2.50  # the user's budget for this job
