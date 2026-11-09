"""Kernel work, phase 7: the kv8 prefill option (serving_kernels.kv8_prefill), its qsa.py edit, the helper installed
only with it, unchanged capacity, the recorded arms byte-identical, and the hero-v2 component (docs/validation-20260927/
kernels-phase7.md). CPU only; the kernels are in tests/test_qsa_prefill.py (Triton interpreter)."""

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest
from test_serving_kernels import synthetic_site

from arc_infra import serving_kernels_patch as patch
from arena import hero_v2
from arena.variants import VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
KV8 = {**patch.PHASE4, "kv_cache_8bit": "int8"}


def test_option_is_validated_opt_in_and_needs_int8():
    assert patch.validate({**KV8, "kv8_prefill": "bf16"}) == {**patch.validate(KV8), "kv8_prefill": "bf16"}
    assert "kv8_prefill" not in patch.validate(KV8)  # off: the recorded settings serialize as before
    for bad in ({**KV8, "kv8_prefill": "fp8"}, {"kv8_prefill": "bf16"}, {"kv_cache_8bit": "fp8", "kv8_prefill": "bf16"},
                {"kv_cache_4bit": "int4", "kv8_prefill": "bf16"}):
        with pytest.raises(ValueError):
            patch.validate(bad)
    # No recorded arm uses it.
    for name in VARIANTS:
        server = apply_variant(CONTROL, name).get("server") or {}
        assert "kv8_prefill" not in (server.get("serving_kernels") or {}), name


def test_capacity_is_unchanged_same_cache_dtype_flags_and_page():
    """The cache stays int8: the server flags and the cache-spec edits are the kv8 arm's exactly; only the attention
    import of the QSA read differs."""
    on = patch.validate({**KV8, "kv8_prefill": "bf16"})
    assert patch.server_flags(on) == patch.server_flags(patch.validate(KV8)) == [
        "--mamba-ssm-cache-dtype", "bfloat16", "--kv-cache-dtype", "int8_per_token_head"]
    edited = patch.qsa_kv8_prefill()["edits"]
    differing = [(a, r) for (a, r, _), (_, r0, _) in zip(edited, patch.QSA["edits"]) if r != r0]
    assert len(differing) == 1 and differing[0][0] == patch.QSA_READ
    assert patch.KV8_PREFILL_IMPORT in differing[0][1] and patch.KV8_ATTENTION_IMPORT not in differing[0][1]
    assert patch.qsa_kv8_prefill()["sha256"] == patch.QSA["sha256"]  # the same pinned file and guard


def test_helper_installed_only_with_the_option_and_the_kv8_arm_byte_identical(tmp_path, monkeypatch):
    site, originals = synthetic_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "e1", {"kv_cache_8bit": "int8"})
    plain = (site / patch.QSA["relative"]).read_bytes()
    assert not (site / patch.KV8_PREFILL_HELPER).exists() and "kv8_prefill_helper_sha256" not in identity
    # A re-extracted runtime with the option.
    for rel, data in originals.items():
        (site / rel).write_bytes(data)
    for helper in (patch.HELPER, patch.KV8_HELPER, patch.SETTINGS):
        (site / helper).unlink()
    identity_on = patch.apply(site, tmp_path / "e2", {"kv_cache_8bit": "int8", "kv8_prefill": "bf16"})
    text = (site / patch.QSA["relative"]).read_text()
    compile(text, "qsa.py", "exec")
    assert text.count("from vllm.arc_kv8_prefill import attention as _arc_kv8_attention") == 1
    assert "from vllm.arc_kv8 import attention" not in text
    assert text.replace(patch.KV8_PREFILL_IMPORT, patch.KV8_ATTENTION_IMPORT).encode() == plain
    helper = (site / patch.KV8_PREFILL_HELPER).read_bytes()
    assert helper == (ROOT / "src/arc_infra/kv8_prefill_runtime.py").read_bytes()
    assert identity_on["kv8_prefill_helper_sha256"] == hashlib.sha256(helper).hexdigest()
    assert identity_on["server_flags"] == identity["server_flags"]
    assert (site / patch.KV8_HELPER).read_bytes() == (ROOT / "src/arc_infra/kv8_runtime.py").read_bytes()
    assert patch.apply(site, tmp_path / "e2", {"kv_cache_8bit": "int8", "kv8_prefill": "bf16"}) == identity_on
    (site / patch.KV8_PREFILL_HELPER).write_text("# tampered\n")
    with pytest.raises(RuntimeError, match="modified"):
        patch.apply(site, tmp_path / "e2", {"kv_cache_8bit": "int8", "kv8_prefill": "bf16"})


def test_exact_pinned_qsa_with_the_option_when_available():
    path = PINNED_SITE / patch.QSA["relative"]
    if not path.is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    patched = patch.patched_source(patch.qsa_kv8_prefill(), path.read_bytes()).decode()
    forward = ast.unparse(next(n for c in ast.parse(patched).body if isinstance(c, ast.ClassDef)
                               for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "forward_qsa"))
    assert "from vllm.arc_kv8_prefill import attention as _arc_kv8_attention" in forward
    assert forward.index("output.zero_()") < forward.index("_arc_kv8_attention(") < forward.index(
        "qsa_sparse_paged_attention(")


def test_runtime_keeps_every_graph_size_on_the_int8_kernel():
    """Static: the decode threshold covers every CUDA-graph size (<= 32 rows x 2 KV heads) and the pinned profile's
    prefill branch starts above it; the load-time check bound is above the measured 0.42% gap."""
    tree = ast.parse((ROOT / "src/arc_infra/kv8_prefill_runtime.py").read_text())
    values = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert values["DECODE_MAX_PROGRAMS"] >= 32 * 2 and values["DECODE_MAX_PROGRAMS"] == 512
    assert 0.0042 < values["CHECK_LIMIT"] <= 0.05 and values["PREFILL_MIN_ROWS"] >= 33
    source = (ROOT / "src/arc_infra/kv8_prefill_runtime.py").read_text()
    assert "is_current_stream_capturing()" in source  # never the split path inside a capture


def test_hero_v2_component_is_off_by_default_and_selectable():
    hero = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kv8-hero")
    assert hero_v2.DEFAULT_COMPONENTS["kv8_prefill"] is False
    base = hero_v2.compose(hero, {"token_budgets": False})
    assert "kv8_prefill" not in base["server"]["serving_kernels"]
    assert base["hero_v2"]["components"]["kv8_prefill"]["on"] is False
    on = hero_v2.compose(hero, {"token_budgets": False, "kv8_prefill": True})
    kernels = on["server"]["serving_kernels"]
    assert kernels == {**base["server"]["serving_kernels"], "kv8_prefill": "bf16"}
    assert patch.validate(kernels) == kernels
    assert on["active_limit"] == base["active_limit"] and on["server"]["max_num_seqs"] == base["server"]["max_num_seqs"]
    with pytest.raises(ValueError):
        hero_v2.normalize({"kv8_prefill": "yes"})


def test_kv8_prefill_check_build_budget_and_timeline(tmp_path):
    import importlib.util

    from arena import throughput_bench as tb
    from arena.serving import lab_overrides

    spec = importlib.util.spec_from_file_location("build_arena_kv8p", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    held_out, long = tmp_path / "held", tmp_path / "long"
    for folder, tf in ((held_out, [0, 1]), (long, [0])):
        (folder / "quality").mkdir(parents=True)
        (folder / "quality/prompts.json").write_text(json.dumps({
            "teacher_forced": [{"index": i, "text": None, "ids": [1, 2, 3, 4], "assistant_spans": [[1, 3]]}
                               for i in tf], "greedy": [{"index": 0, "prompt_ids": [5, 6]}]}))
    output, study = builder.build("kv8p", output=tmp_path / "kv8p", paired_30m=True,
                                  kv8_prefill_check=f"{held_out},{long}")
    assert [t["name"] for t in study["trials"]] == ["kv8p-ref", "kv8p-bf16"]
    reference = study["server"]["serving_kernels"]
    assert "kv8_prefill" not in reference and reference["kv_cache_8bit"] == "int8"
    parsed = {t["name"]: lab_overrides(t["server_env"]) for t in study["trials"]}
    assert parsed["kv8p-ref"].get("serving_kernels") is None
    assert parsed["kv8p-bf16"]["serving_kernels"] == {**reference, "kv8_prefill": "bf16"}
    envs = [{k: v for k, v in t["server_env"].items() if k != "ARENA_SERVING_KERNELS"} for t in study["trials"]]
    assert envs[0] == envs[1]  # the same pinned pool, guard and dump
    prompts = json.loads((output / "configs/kernel-quality/prompts.json").read_text())
    assert prompts["greedy"] == [] and len(prompts["teacher_forced"]) == 3
    for trial in study["trials"]:
        assert trial["quality"] == {"greedy": False, "gsm8k": False, "report_only": True}
        points = tb.grid_points({**tb.DEFAULT_GRID, **trial["bench"]})
        assert [(p["kind"], p["concurrency"]) for p in points] == [("mixed", 23), ("mixed", 30)]
        inject = trial["bench"]["mixed_points"][0]["inject"]
        assert 1100 <= inject["rate"] * inject["tail_tokens"] <= 1250  # hero-v2 phase 2: ~1,170 uncached tok/s
    assert study["server"]["max_num_seqs"] == 32 and study["generation_slots"] == 32  # run 2's sequences
    # Budget and timeline:; each trial starts (wall + 480 s left) after a ~665 s setup and its
    # predecessors, and finishes inside its own timeout (wall + 600 s).
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validation_timeout")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "mv", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "kv8-prefill-check")
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 4.0
    notebook = study["notebook_seconds"]
    assert notebook <= timeout - 60 and study["kernel_timeout"] <= timeout
    # The mixed points fit the pinned pool with slack (v1's overflowed it): streams + injector <= 50% of the gate.
    for m in builder.KV8P_BENCH["mixed_points"]:
        per_stream = -(-(m["prompt_tokens"] + m["output_tokens"] + 256) // builder.KV8P_BLOCK_TOKENS)
        injector = -(-(m["inject"]["prefix_tokens"] + m["inject"]["tail_tokens"] + 256) // builder.KV8P_BLOCK_TOKENS)
        assert m["concurrency"] * per_stream + 2 * injector <= 0.5 * builder.KV8P_POOL_BLOCKS, m
        assert m["concurrency"] + 1 <= study["generation_slots"]
    # The build ran the timeline check with 15% margins (arena.timeline); its plan is recorded.
    plan = study["kv8_prefill_check"]["timeline"]
    assert [p["trial"] for p in plan] == ["kv8p-ref", "kv8p-bf16"] and all(p["end"] <= p["start"] + p["timeout"]
                                                                          for p in plan)
