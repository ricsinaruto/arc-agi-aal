"""Holo4-27B on the hero-v2 arena setup (29 September; docs/validation-20260927/holo4-27b.md)."""

import ast
import hashlib
import importlib.util
import json
import textwrap
from pathlib import Path

import pytest
from test_qwen_serving import import_fork

from arena import candidate_serving as candidate
from arena import hero_v2
from arena.candidate_profiles import HOLO4_27B, PROFILES, QWEN_27_NVFP4, from_study
from arena.variants import apply_variant
from arc_infra import stall_dump_patch

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = Path("./runs/expert-quant/inputs-v2")


def builder():
    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hero():
    return apply_variant(json.loads((ROOT / "configs/arena_30m.json").read_text()), hero_v2.BASE)


def holo_config():
    return json.loads((ROOT / "configs/arena_30m_holo4_27b.json").read_text())


# --- checkpoint identity ---------------------------------------------------------------------------------------


def test_manifest_pins_the_revision_every_file_and_the_qwen38_template():
    manifest = json.loads(HOLO4_27B.manifest.read_text())
    assert (manifest["repo"], manifest["revision"]) == (HOLO4_27B.model, HOLO4_27B.revision)
    assert manifest["dataset"] == "richardcsaky/arc-holo4-27b-nvfp4" and manifest["license"] == "cc-by-nc-4.0"
    assert all(len(spec["sha256"]) == 64 and spec["bytes"] > 0 for spec in manifest["files"].values())
    shards = {n for n in manifest["files"] if n.endswith(".safetensors")}
    assert shards == {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    assert round(sum(manifest["files"][n]["bytes"] for n in shards) / 1e9, 1) == 22.9
    qwen = json.loads(QWEN_27_NVFP4.manifest.read_text())
    assert manifest["files"]["chat_template.jinja"] == qwen["files"]["chat_template.jinja"]  # byte-identical template
    fixture = (ROOT / "tests/fixtures/qwen27-chat-template.jinja").read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == manifest["files"]["chat_template.jinja"]["sha256"]
    assert PROFILES["holo4-27b-nvfp4"] is HOLO4_27B
    assert (HOLO4_27B.mtp_tokens, HOLO4_27B.mtp_tensors, HOLO4_27B.effort) == (0, 0, "xhigh")


def test_verification_reuses_hashes_only_for_unchanged_files(tmp_path):
    config = {"architectures": [HOLO4_27B.architecture],
              "quantization_config": {"quant_method": HOLO4_27B.quant_method},
              "text_config": {"max_position_embeddings": 262144}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}))
    (tmp_path / "a.safetensors").write_bytes(b"weights")
    manifest = {"repo": HOLO4_27B.model, "revision": HOLO4_27B.revision,
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in tmp_path.iterdir()}}
    cache = tmp_path.parent / "cache.json"
    assert candidate.verify_model(tmp_path, manifest, cache=cache)["hashes_reused_from_this_container"] == 0
    assert candidate.verify_model(tmp_path, manifest, cache=cache)["hashes_reused_from_this_container"] == 3
    (tmp_path / "a.safetensors").write_bytes(b"weightz")  # same size, new content and mtime: hashed again
    with pytest.raises(RuntimeError, match="hash mismatch"):
        candidate.verify_model(tmp_path, manifest, cache=cache)


# --- serving command -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("override", [None, "bfloat16"])
def test_pinned_command_serves_compressed_tensors_on_triton_with_fp8_or_the_bf16_reference(
        tmp_path, monkeypatch, override):
    recipe = import_fork(tmp_path, monkeypatch)
    for key, value in {"TAAF_VLLM_MAX_NUM_SEQS": "25", "TAAF_VLLM_ENABLE_PREFIX_CACHING": "1",
                       "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "4096", "TAAF_VLLM_KV_CACHE_DTYPE": "auto"}.items():
        monkeypatch.setenv(key, value)
    candidate.configure_candidate(recipe, HOLO4_27B, override)
    tuning = recipe.resolve_vllm_tuning()
    assert tuning["mtp_speculative_tokens"] == 0
    command = recipe.server_command(tmp_path / "model", tuning=tuning)
    for flag, value in {"--quantization": "compressed-tensors", "--served-model-name": HOLO4_27B.model,
                        "--attention-backend": "TRITON_ATTN", "--kv-cache-dtype": override or "fp8",
                        "--max-model-len": "262144", "--tool-call-parser": "qwen3_coder",
                        "--reasoning-parser": "qwen3"}.items():
        assert command[command.index(flag) + 1] == value
    assert command.count("--kv-cache-dtype") == 1 and "--speculative-config" not in command
    assert candidate.served_kv_dtype(HOLO4_27B, override) == (override or "fp8")


def test_kv_override_needs_a_pinned_kv_flag_and_a_known_dtype():
    assert candidate.candidate_args(HOLO4_27B) == list(HOLO4_27B.extra_args)
    with pytest.raises(ValueError, match="bfloat16 or fp8"):
        candidate.candidate_args(HOLO4_27B, "auto")
    with pytest.raises(ValueError, match="pins no"):
        candidate.candidate_args(QWEN_27_NVFP4, "bfloat16")


def test_trial_environment_reaches_the_candidate_server_options(monkeypatch):
    from arena.serving import lab_overrides

    env = {"ARENA_KV_CACHE_DTYPE": "bfloat16", "ARENA_READY_GUARD": "180,900", "ARENA_STALL_DUMP": "90",
           "ARENA_PROMPT_TOKENS_DETAILS": "1"}
    lab = lab_overrides(env)
    assert set(lab) <= candidate.CANDIDATE_LAB
    from arena.variants import HERO_PREFIX_CACHE

    study = {"cache_experiment": {"prefix_cache": dict(HERO_PREFIX_CACHE)}}
    options = candidate.candidate_configure_options(study, lab)
    assert options == {"prefix_cache": study["cache_experiment"]["prefix_cache"], "ready_guard_seconds": (180, 900),
                       "prompt_tokens_details": True, "stall_dump_seconds": 90}
    assert candidate.candidate_configure_options({}, lab_overrides({"ARENA_PREFIX_CACHE": "off"})) == {}
    assert "dense_kernels" not in candidate.CANDIDATE_LAB  # a Flash overlay env on a candidate server is refused


# --- stall dump without pruning --------------------------------------------------------------------------------


def fake_loader(tmp_path, monkeypatch):
    text = ("def load_model(self, vllm_config, model_config):\n"
            "        with set_default_torch_dtype(model_config.dtype):\n"
            "            model = initialize_model(vllm_config)\n")
    site = tmp_path / "site"
    target = site / stall_dump_patch.BASE_LOADER["relative"]
    target.parent.mkdir(parents=True)
    target.write_text(text)
    monkeypatch.setitem(stall_dump_patch.BASE_LOADER, "sha256", hashlib.sha256(text.encode()).hexdigest())
    return site, target


def test_standalone_stall_dump_arms_the_worker_before_the_model_is_built(tmp_path, monkeypatch):
    site, target = fake_loader(tmp_path, monkeypatch)
    identity = stall_dump_patch.apply(site, tmp_path / "out", 90)
    patched = target.read_text()
    assert patched.index("_arc_install_stall_dump(90, _arc_worker_class(vllm_config)") < patched.index(
        "initialize_model(")
    assert (site / stall_dump_patch.HELPER).read_bytes() == (
        ROOT / "src/arc_infra/expert_prune_runtime.py").read_bytes()
    assert stall_dump_patch.apply(site, tmp_path / "out", 90) == identity  # a restart: accepted, not re-patched
    with pytest.raises(RuntimeError, match="modified"):
        stall_dump_patch.apply(site, tmp_path / "out", 120)
    for bad in (5, 300, "90", True):
        with pytest.raises(ValueError):
            stall_dump_patch.validate(bad)


def test_the_patched_loader_line_arms_the_one_shot_timer(tmp_path, monkeypatch):
    import faulthandler
    import sys
    import types

    from arc_infra import expert_prune_runtime

    calls = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: calls.append("cancel"))
    worker = type("Worker", (), {"execute_model": lambda self, s: s, "sample_tokens": lambda self, g: g})
    monkeypatch.setitem(sys.modules, "arc_fake_worker_mod", types.SimpleNamespace(Worker=worker))
    vllm = types.ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.arc_stall_dump", expert_prune_runtime)
    namespace = {"vllm_config": types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(worker_cls="arc_fake_worker_mod.Worker"))}
    code = textwrap.dedent(stall_dump_patch.edit(90).replace(stall_dump_patch.ANCHOR, ""))
    exec(compile(code, "patch", "exec"), namespace)
    assert worker().execute_model(7) == 7 and calls == [(90,), "cancel"]


def test_standalone_stall_dump_applies_to_the_pinned_loader_when_available(tmp_path):
    import os
    import shutil

    pinned = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
    source = pinned / stall_dump_patch.BASE_LOADER["relative"]
    if not source.is_file():
        pytest.skip("extracted pinned runtime not available (ARC_PINNED_SITE)")
    target = tmp_path / "site" / stall_dump_patch.BASE_LOADER["relative"]
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    stall_dump_patch.apply(tmp_path / "site", tmp_path / "out", 90)
    text = target.read_text()
    assert text.index("_arc_install_stall_dump(90") < text.index("initialize_model(")


def test_configure_refuses_both_stall_dump_overlays():
    from types import SimpleNamespace

    from arena.serving import configure

    with pytest.raises(ValueError, match="mask"):
        configure(SimpleNamespace(), "/tmp", expert_prune={"pruned": {}}, stall_dump_seconds=90)


# --- composition ---------------------------------------------------------------------------------------------


def test_compose_keeps_the_arena_side_of_hero_v2_and_drops_every_flash_component():
    study = hero_v2.compose_candidate(hero(), holo_config(), {})
    flash = hero_v2.compose(hero(), {"serving_tweaks": False, "expert_prune": False, "gptq_arc": False,
                                     "stall_dump": False, "prefix_cache": True})
    for key in ("budget_clock", "token_budget", "live_progress", "compaction_thinking", "reasoning_history",
                "windows", "opening_switch_level1_games", "level_budget_policy", "backlog_policy", "temperature",
                "completion_cap", "reasoning_effort", "games", "kv_margin", "wall_seconds"):
        assert study[key] == flash[key], key
    assert study["cache_experiment"] == {**flash["cache_experiment"], "kv_cache_dtype": "fp8"}
    assert study["cache_experiment"]["prefix_cache"] == {"mamba_align_fix": True, "mamba_keep_state": True,
                                                         "mamba_demote_superseded": True}
    assert set(study["server"]) == set(hero_v2.SERVER_KEYS)
    assert study["server"]["mtp_tokens"] == 0 and study["stall_dump_seconds"] == 90
    assert study["candidate_serving"]["attention_backend"] == "TRITON_ATTN"
    record = study["hero_v2"]
    assert record["on"] == ["compaction_thinking", "live_progress", "prefix_cache", "reasoning_history",
                            "stall_dump", "token_budgets"]
    assert set(record["off"]) == {*hero_v2.CANDIDATE_DROPPED, "harness_fixes"}
    assert all(record["components"][k]["reason"] for k in hero_v2.CANDIDATE_DROPPED)
    assert record["dropped_server_keys"] == ["dense_kernels", "serving_kernels", "stack_kernels"]
    assert study["arena_variant"]["name"] == "holo4-27b-hero-v2"
    assert study["arena_variant"]["reference_variant"] == hero_v2.NAME
    assert from_study(study) is HOLO4_27B


def test_streams_follow_the_served_layout():
    projected = hero_v2.candidate_layout("holo4-27b-nvfp4")
    assert (projected["blocks"], projected["blocks_per_window"], projected["windows_fit"]) == (1396, 49, 25)
    assert projected["active_limit"] == projected["max_num_seqs"] == 25
    measured = hero_v2.candidate_layout("holo4-27b-nvfp4", served_blocks=1500, blocks_per_window=48)
    assert (measured["windows_fit"], measured["max_num_seqs"], measured["active_limit"]) == (28, 28, 28)
    assert hero_v2.candidate_layout("holo4-27b-nvfp4", served_blocks=1500)["windows_fit"] == 27
    small = hero_v2.candidate_layout("holo4-27b-nvfp4", served_blocks=1000)
    assert (small["windows_fit"], small["max_num_seqs"], small["active_limit"]) == (18, 25, 18)
    big = hero_v2.candidate_layout("holo4-27b-nvfp4", served_blocks=2400)
    assert (big["max_num_seqs"], big["active_limit"]) == (32, 32)  # the MTP-0 decode-graph capture size
    study = hero_v2.compose_candidate(hero(), holo_config(), {}, served_blocks=1500, blocks_per_window=48)
    assert study["active_limit"] == study["generation_slots"] == study["server"]["max_num_seqs"] == 28


def test_flash_only_components_cannot_be_switched_on():
    for key, value in (("expert_prune", "prune12"), ("gptq_arc", True), ("serving_tweaks", True)):
        with pytest.raises(hero_v2.ComponentUnavailable):
            hero_v2.compose_candidate(hero(), holo_config(), {key: value})
    off = hero_v2.compose_candidate(hero(), holo_config(), {"prefix_cache": False, "stall_dump": False,
                                                           "expert_prune": False})
    assert "prefix_cache" not in off["cache_experiment"] and "stall_dump_seconds" not in off
    with pytest.raises(ValueError, match="Unknown"):
        hero_v2.compose_candidate(hero(), holo_config(), {"kv8": True})
    wrong = holo_config()
    wrong["model_revision"] = "0" * 40
    with pytest.raises(ValueError, match="identity"):
        hero_v2.compose_candidate(hero(), wrong, {})


def test_compaction_prefill_applies_to_the_holo_template():
    from arena.compaction import PREFILL_TEMPLATE_MODELS

    assert HOLO4_27B.model in PREFILL_TEMPLATE_MODELS


# --- builds ----------------------------------------------------------------------------------------------------

COMMON = dict(paired_30m=True, candidate_profile="holo4-27b-nvfp4", arena_variant=hero_v2.BASE, hero_v2={},
              opening_burst_requests=3, exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], clone_instances=110)


def test_run_bundle_matches_hero_v2_except_the_model(tmp_path):
    output, study = builder().build("holo-run", output=tmp_path / "run", wall_seconds=3600,
                                    hero_v2_served_blocks=1396, hero_v2_window_blocks=49, **COMMON)
    assert [t["name"] for t in study["trials"]] == ["duck-transfer"]
    assert study["trials"][0]["server_env"] == {"ARENA_STALL_DUMP": "90", "ARENA_READY_GUARD": "180,900",
                                                "ARENA_PROMPT_TOKENS_DETAILS": "1"}
    assert study["competition"]["preview_instances"] == 110 and study["competition"]["preview_seconds"] == 3600
    assert "harness_fixes" not in study and study["cache_experiment"]["prefix_cache"]
    assert study["server"]["max_num_seqs"] == 25 and study["windows"] == [65536]
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                  and n.name == "validation_timeout"], type_ignores=[]), "validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "holo-60m")
    # The trial starts while (timeout - 60) - 120 - elapsed >= 3,600 + 300 (bootstrap) and (timeout - 60) - elapsed
    # >= 3,600 + the 600 s cleanup reserve (run.py); run.py's is the tighter. Measured setup: 648 s (the soak).
    deadline = min((timeout - 60) - 120 - 300 - 3600, (timeout - 60) - study["competition"]["cleanup_seconds"] - 3600)
    assert deadline == 1020 and deadline >= 648 + 300
    assert len(study["games"]) == 20 and study["budget_clock"] == "tokens" and study["active_limit"] == 25
    assert (output / "src/arena/models/holo4-27b-nvfp4.json").is_file()
    assert json.loads((output / "configs/comparison.json").read_text()) == study


@pytest.mark.skipif(not (HELD_OUT / "quality/prompts.json").is_file(), reason="held-out drift inputs not present")
def test_smoke_bundle_measures_drift_then_checks_and_soaks_the_arena_server(tmp_path):
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("step 1: ACTION1 moved the block.\n" * 3000)
    output, study = builder().build("holo-smoke", output=tmp_path / "smoke", wall_seconds=600, hero_v2_smoke=True,
                                    hero_v2_drift_inputs=f"{HELD_OUT}:3", hero_v2_transcript=str(transcript),
                                    **COMMON)
    names = [t["name"] for t in study["trials"]]
    assert names == ["kv-bf16-ref", "kv-fp8", "holo-check", "duck-transfer"]
    ref, gate, check, soak = study["trials"]
    assert ref["server_env"]["ARENA_KV_CACHE_DTYPE"] == "bfloat16" and "ARENA_KV_CACHE_DTYPE" not in gate["server_env"]
    assert ref["server_env"]["ARENA_KV_CACHE_MEMORY_BYTES"] == gate["server_env"]["ARENA_KV_CACHE_MEMORY_BYTES"]
    assert gate["reference"] == "kv-bf16-ref" and gate["quality"]["report_only"]
    assert check["server_env"] == soak["server_env"]  # the check and the soak share the arena's server
    assert "ARENA_KV_CACHE_MEMORY_BYTES" not in soak["server_env"]
    prompts = json.loads((output / "configs/kernel-quality/prompts.json").read_text())
    assert len(prompts["teacher_forced"]) == 3 and all(t["ids"] for t in prompts["teacher_forced"])
    assert (output / "bench/transcript.txt").read_text() == transcript.read_text()
    assert study["throughput_bench"]["points"] == [[64512, study["active_limit"]]]
    assert study["throughput_bench"]["prefix_check"]["cut_fractions"] == [0.2, 0.33, 0.66, 0.9]
    assert study["opening_max_wall_seconds"] == 300 and study["opening_cap_absolute"]
    assert study["notebook_seconds"] == 2640
    with pytest.raises(ValueError, match="drift"):
        builder().build("bad", output=tmp_path / "bad", wall_seconds=600, hero_v2_smoke=True, **COMMON)


def test_modal_tests_and_weight_fetch_helpers(tmp_path):
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    names = {"validation_timeout", "arena_arguments"}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[]), "validation", "exec"), ns)
    assert ns["arena_arguments"]("holo-smoke") == [] and ns["arena_arguments"]("holo-60m") == ["--trial",
                                                                                              "duck-transfer"]
    rate = (0.000842 + 8 * 0.0000131 + 192 * 0.00000222) * 1.15
    assert (ns["validation_timeout"]("arena", "holo-smoke") + 60) * rate < 4.4
    assert (ns["validation_timeout"]("arena", "holo-60m") + 60) * rate < 8.5
    fetch = ast.parse((ROOT / "scripts/modal_candidate_weights.py").read_text())
    helpers = {"storage_gib", "bound_usd", "check_file"}
    ns = {"CPU": 8.0, "MEMORY_GIB": 16, "TIMEOUT": 3600, "STORAGE_DAYS": 7, "Path": Path}
    exec(compile(ast.Module(body=[n for n in fetch.body if isinstance(n, ast.FunctionDef) and n.name in helpers],
                            type_ignores=[]), "fetch", "exec"), ns)
    manifest = json.loads(HOLO4_27B.manifest.read_text())
    assert ns["storage_gib"](manifest) == pytest.approx(21.35, abs=0.01)
    bound, storage = ns["bound_usd"]({"rates_per_second": {}}, 21.35)
    assert bound < 1.1 and storage == pytest.approx(21.35 * 0.09 * 7 / 30)
    file = tmp_path / "x.bin"
    file.write_bytes(b"abc")
    good = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert ns["check_file"](file, good)["ok"] and not ns["check_file"](file, {**good, "sha256": "0" * 64})["ok"]
    assert not ns["check_file"](tmp_path / "missing", good)["ok"]


def test_soak_bundle_starts_after_the_measured_setup_within_its_modal_timeout(tmp_path):
    b = builder()
    output, study = b.build("holo-soak", output=tmp_path / "soak", wall_seconds=720, hero_v2_soak=True,
                            hero_v2_served_blocks=1396, hero_v2_window_blocks=49, **COMMON)
    assert [t["name"] for t in study["trials"]] == ["duck-transfer"]
    assert study["trials"][0]["server_env"] == {"ARENA_PROMPT_TOKENS_DETAILS": "1", "ARENA_STALL_DUMP": "90",
                                                "ARENA_READY_GUARD": "180,900"}
    assert study["active_limit"] == study["server"]["max_num_seqs"] == 25
    assert study["cache_experiment"]["prefix_cache"] and study["compaction_thinking"] == "prefilled-empty"
    assert "harness_fixes" not in study and study["hero_v2"]["components"]["harness_fixes"]["on"] is False
    assert study["opening_max_wall_seconds"] == 360 and study["opening_cap_absolute"]
    soak = study["hero_v2"]["soak"]
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                  and n.name in {"validation_timeout", "arena_arguments"}], type_ignores=[]),
                 "validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "holo-soak")
    assert soak["function_timeout"] == timeout and study["notebook_seconds"] == timeout - 60
    # The bootstrap starts the trial while budget = (timeout - 60) - 120 - elapsed >= wall + 300; run.py then needs
    # the preview wall plus the cleanup reserve before its hard end (timeout - 60).
    bootstrap_deadline = (timeout - 60) - 120 - 300 - study["wall_seconds"]
    run_deadline = (timeout - 60) - study["competition"]["cleanup_seconds"] - study["competition"]["preview_seconds"]
    assert soak["trial_start_deadline_seconds"] == bootstrap_deadline <= run_deadline
    assert bootstrap_deadline >= 604 + 100  # the measured first start (smoke v2) plus a margin
    assert ns["arena_arguments"]("holo-soak") == ["--trial", "duck-transfer"]
    assert (timeout + 60) * (0.000842 + 8 * 0.0000131 + 192 * 0.00000222) * 1.15 < 3.2
    with pytest.raises(ValueError, match="600..900"):
        b.build("bad", output=tmp_path / "bad", wall_seconds=300, hero_v2_soak=True, **COMMON)
    with pytest.raises(ValueError, match="harness"):
        hero_v2.compose_candidate({**hero(), "harness_fixes": {"x": 1}}, holo_config(), {})
