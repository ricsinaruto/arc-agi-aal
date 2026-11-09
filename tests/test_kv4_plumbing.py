"""4-bit QSA KV cache: overlay option, the kv8 qsa.py edits calling arc_kv4, helpers installed only with the option,
the served block size, the hero-v2 sizing, the drift-test build and its Modal test (docs/validation-20260927/
kv-cache-4bit.md). CPU only; the kernels are in tests/test_kv4_kernels.py."""

import ast
import json
import os
import sys
from pathlib import Path

import pytest
from test_serving_kernels import synthetic_site

from arc_infra import serving_kernels_patch as patch

ROOT = Path(__file__).resolve().parents[1]
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))


# ----------------------------------------------------------------------------- option and patch
def test_option_is_validated_exclusive_and_sets_dtype_and_block_size():
    assert "kv_cache_4bit" not in patch.validate(patch.PHASE4)
    assert patch.validate({**patch.PHASE4, "kv_cache_4bit": None}) == patch.PHASE4
    for mode, dtype, block in (("int4", "int4_per_token_head", "3136"), ("k8v4", "int8_per_token_head", "2112")):
        spec = patch.validate({"kv_cache_4bit": mode})
        assert spec["kv_cache_4bit"] == mode and "kv_cache_8bit" not in spec
        assert patch.targets(spec) == [patch.qsa_kv4()]
        assert patch.server_flags(spec) == ["--kv-cache-dtype", dtype, "--block-size", block]
    for bad in ({"kv_cache_4bit": "int8"}, {"kv_cache_4bit": True}, {"kv_cache_4bit": "int4_per_token_head"},
                {"kv_cache_4bit": "int4", "kv_cache_8bit": "int8"}):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_kv4_edits_are_the_kv8_edits_calling_arc_kv4():
    assert patch.qsa_kv4()["sha256"] == patch.QSA["sha256"] and patch.qsa_kv4()["relative"] == patch.QSA["relative"]
    assert [a for a, _, _ in patch.QSA_KV4_EDITS] == [a for a, _, _ in patch.QSA["edits"]]
    for (_, kv8, _), (_, kv4, _) in zip(patch.QSA["edits"], patch.QSA_KV4_EDITS):
        assert "arc_kv8" not in kv4 and "8-bit" not in kv4
        assert kv4.replace("arc_kv4", "arc_kv8").replace("kv_cache_4bit", "kv_cache_8bit").replace(
            "4-bit", "8-bit").replace('"int4_per_token_head", "int8_per_token_head"',
                                      '"int8_per_token_head", "fp8_per_token_head"') == kv8
    assert '"auto", "bfloat16", "int4_per_token_head", "int8_per_token_head"]' in patch.QSA_KV4_EDITS[0][1]


def test_kv4_patch_installs_both_helpers_only_with_the_option(tmp_path, monkeypatch):
    site, originals = synthetic_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "evidence", {"kv_cache_4bit": "int4"})
    assert set(identity["files"]) == {patch.QSA["relative"]}
    assert identity["server_flags"] == ["--kv-cache-dtype", "int4_per_token_head", "--block-size", "3136"]
    assert (site / patch.KV4_HELPER).read_bytes() == (ROOT / "src/arc_infra/kv4_runtime.py").read_bytes()
    assert (site / patch.KV8_HELPER).read_bytes() == (ROOT / "src/arc_infra/kv8_runtime.py").read_bytes()
    assert identity["kv4_helper_sha256"] and identity["kv8_helper_sha256"]
    text = (site / patch.QSA["relative"]).read_text()
    compile(text, "qsa.py", "exec")
    assert "arc_kv8" not in text
    for needle in ("_arc_kv4_customize_spec(spec)", "_arc_kv4_kv_cache_shape(", "_arc_kv4_impl_init_args(",
                   "_arc_kv4_attention(", "_arc_kv4_cache_update(", "not _arc_kv4_allowed(self.kv_cache_dtype)"):
        assert needle in text, needle
    assert patch.apply(site, tmp_path / "evidence", {"kv_cache_4bit": "int4"}) == identity  # idempotent
    (site / patch.KV4_HELPER).write_text("# tampered\n")
    with pytest.raises(RuntimeError, match="modified"):
        patch.apply(site, tmp_path / "evidence", {"kv_cache_4bit": "int4"})


def test_kv8_overlay_is_unchanged_by_the_kv4_option(tmp_path, monkeypatch):
    site, _ = synthetic_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "evidence", {"kv_cache_8bit": "int8"})
    assert not (site / patch.KV4_HELPER).exists() and "kv4_helper_sha256" not in identity
    assert identity["server_flags"] == ["--kv-cache-dtype", "int8_per_token_head"]
    assert "arc_kv4" not in (site / patch.QSA["relative"]).read_text()


def test_exact_pinned_qsa_source_when_available():
    path = PINNED_SITE / patch.QSA["relative"]
    if not path.is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    patched = patch.patched_source(patch.qsa_kv4(), path.read_bytes()).decode()
    tree = ast.parse(patched)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    impl = {n.name: n for n in classes["Qwen3_8FlashNextQSAFlashAttentionImpl"].body if isinstance(n, ast.FunctionDef)}
    forward = ast.unparse(impl["forward_qsa"])
    assert forward.index("output.zero_()") < forward.index("_arc_kv4_attention(") < forward.index(
        "qsa_sparse_paged_attention(")
    assert "do_kv_cache_update" in impl
    assert patched.count("Qwen3.8-Flash-Next QSA requires a BF16 main KV cache") == 2


def test_runtime_helper_entry_points_match_the_qsa_patch():
    tree = ast.parse((ROOT / "src/arc_infra/kv4_runtime.py").read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    imported = set()
    for _, replacement, _ in patch.QSA_KV4_EDITS:
        for line in replacement.splitlines():
            if "from vllm.arc_kv4 import" in line:
                imported.add(line.split("import ")[1].split(" as ")[0])
    assert imported == {"customize_spec", "kv_cache_shape", "impl_init_args", "cache_update", "attention", "allowed"}
    assert imported <= names
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("CACHE_DTYPES", "CONFIG_KEY")}
    assert values == {"CACHE_DTYPES": patch.KV4_CACHE_DTYPES, "CONFIG_KEY": "kv_cache_4bit"}
    # The helper reuses the kv8 module under its installed name.
    assert "from vllm import arc_kv8 as _kv8" in (ROOT / "src/arc_infra/kv4_runtime.py").read_text()


# ----------------------------------------------------------------------------- sizing
def sizing():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import kv4_sizing
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return kv4_sizing


def test_block_sizes_hold_the_gdn_state_page_and_match_the_sizing():
    kv4_sizing = sizing()
    page = kv4_sizing.kv8.gdn_page(0, 2)
    for mode, block in patch.KV4_BLOCK_TOKENS.items():
        per_token = kv4_sizing.PER_TOKEN[mode]
        assert block % 64 == 0 and block * per_token >= page and (block - 64) * per_token < page
        assert kv4_sizing.SERVED_BLOCK_TOKENS[mode] == block
    # kv8's block rule sees 1,024 B per token: 1,600 (the served hero-v2 geometry).
    assert 16 * -(-page // (16 * 1024)) == kv4_sizing.SERVED_BLOCK_TOKENS["kv8"]


def test_hero_v2_capacity_table():
    rows = sizing().hero_v2_table()  # asserts the served int8 layout (1,335 blocks, 51 per 64K, 23 fit)
    assert [rows[kv]["64k"]["fit"] for kv in ("kv8", "k8v4", "int4")] == [23, 28, 38]
    assert [rows[kv]["128k"]["fit"] for kv in ("kv8", "k8v4", "int4")] == [13, 16, 22]
    assert [rows[kv]["128k"]["per_window"] for kv in ("kv8", "k8v4", "int4")] == [92, 72, 51]
    # The drift test's pinned pool (EXPERT_QUANT_KV_BYTES): Test A served 379 int8 blocks.
    pinned = sizing().hero_v2_table(8053063680)
    assert pinned["kv8"]["64k"]["blocks"] == 379


def test_modal_kv4_drift_budget():
    text = (ROOT / "scripts/modal_validation.py").read_text()
    assert '("arena", "kv4-drift"): 3000' in text
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (3000 + 60) * rate * 1.15 <= 5.0
    builder = load_builder()
    assert builder.KV4_DRIFT_SECONDS <= 3000 - 60


# ----------------------------------------------------------------------------- drift-test build
def load_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_arena_kv4", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# hero-v2 smoke v2's recorded server block (runs/modal-validation/arena-hero-v2-smoke-v2-modal-20260929).
HERO_V2_SERVING = {"mamba_ssm_cache_dtype": None, "moe_backend": "marlin", "draft_experts": None, "hc_fusion": False,
                   "topk_sampler": True, "kv_cache_8bit": "int8", "ple_uva": True, "embed_host": True}


def test_kv4_drift_build(tmp_path):
    from arena.serving import lab_overrides

    builder = load_builder()
    held_out, long = tmp_path / "held", tmp_path / "long"
    for folder, tf, greedy in ((held_out, [0, 1], [0, 1]), (long, [0], [])):
        (folder / "quality").mkdir(parents=True)
        (folder / "quality/prompts.json").write_text(json.dumps({
            "teacher_forced": [{"index": i, "text": None, "ids": [1, 2, 3, 4], "assistant_spans": [[1, 3]]}
                               for i in tf],
            "greedy": [{"index": i, "prompt_ids": [5, 6]} for i in greedy]}))
    output, study = builder.build("kv4", output=tmp_path / "kv4", paired_30m=True, kv4_drift=f"{held_out},{long}")
    assert [t["name"] for t in study["trials"]] == ["kv4-ref", "kv4-int4", "kv4-k8v4"]
    server = study["server"]
    assert server["mtp_tokens"] == 0 and server["serving_kernels"] == HERO_V2_SERVING
    assert server["expert_prune"]["mask"] == "configs/expert-prune/prune12.json"
    assert server["expert_overlay"]["dataset"] == "richardcsaky/arc-expert-gptq-arc-v1"
    assert server["expert_overlay"]["check"] == "size"
    assert server["max_num_seqs"] == 25 and server["stack_kernels"] == {"async_h2d": True}
    envs = {t["name"]: t["server_env"] for t in study["trials"]}
    assert len({json.dumps({k: v for k, v in e.items() if k != "ARENA_SERVING_KERNELS"}) for e in envs.values()}) == 1
    parsed = {name: lab_overrides(env) for name, env in envs.items()}
    assert parsed["kv4-ref"].get("serving_kernels") is None
    for mode in ("int4", "k8v4"):
        kernels = parsed[f"kv4-{mode}"]["serving_kernels"]
        assert kernels == patch.validate({**{k: v for k, v in HERO_V2_SERVING.items() if k != "kv_cache_8bit"},
                                          "kv_cache_4bit": mode})
    prompts = json.loads((output / "configs/kernel-quality/prompts.json").read_text())
    assert [t["index"] for t in prompts["teacher_forced"]] == [0, 1, 1000]
    assert [g["index"] for g in prompts["greedy"]] == [0, 1]
    assert study["kv4_drift"]["positions"] == 9 and study["kernel_timeout"] == builder.KV4_DRIFT_SECONDS
    assert all(t.get("bench") == builder.KV4_DRIFT_BENCH for t in study["trials"])
    assert all(t["quality"] == {"greedy": True, "gsm8k": False, "report_only": True} for t in study["trials"])


def test_analysis_on_fake_servers(tmp_path, monkeypatch, capsys):
    import asyncio
    import importlib.util

    from test_dense_kernels import fake_server

    from arena import kernel_gate as gate

    source = tmp_path / "bundle"
    (source / gate.INPUTS).mkdir(parents=True)
    prompts = {"greedy": [{"index": i, "prompt_ids": [1, 2, 3]} for i in range(2)],
               "teacher_forced": [{"index": i, "text": None, "ids": list(range(1, 40)), "assistant_spans": [[5, 20]]}
                                  for i in range(3)]}
    (source / gate.INPUTS / "prompts.json").write_text(json.dumps(prompts))
    (source / gate.INPUTS / "gsm8k.jsonl").write_text(json.dumps({"index": 0, "question": "q", "answer": "#### 1"}) + "\n")

    async def no_isolation(url):
        return {}

    monkeypatch.setattr(gate, "isolate_trial", no_isolation)
    study = {"server": {"mtp_tokens": 0}, "reasoning_effort": "xhigh"}
    quality = {"greedy": True, "gsm8k": False, "report_only": True}
    run_dir = tmp_path / "trials"

    def run(trial, scale):
        return asyncio.run(gate.run_quality(study, {**trial, "quality": quality}, source=source,
                                            output=run_dir / trial["name"], server_dir=tmp_path,
                                            base_url="http://x/v1", deadline_epoch=1e12, log=lambda *a, **k: None,
                                            transport=fake_server(scale)))

    assert run({"name": "kv4-ref", "role": "reference"}, 0.0)["status"] == "complete"
    run({"name": "kv4-int4", "role": "gate", "reference": "kv4-ref"}, 0.1)
    run({"name": "kv4-k8v4", "role": "gate", "reference": "kv4-ref"}, 0.0)
    (run_dir / "kv4-int4" / "vllm-openai-server.log").write_text(
        'ARC_CACHE_LAYOUT {"num_blocks": 361, "planned_cache_bytes": 1, "groups": [{"type": "FullAttentionSpec", '
        '"block_size": 3136, "page_size_bytes": 1655808}]}\n'
        'ARC_KV4_SELFCHECK {"mode": "int4", "reader_rel_error_vs_dequant": 0.004}\n'
        'ARC_KV4_ROUNDTRIP {"layer_order": 0, "k_int4_rht": 0.1, "k_int4_plain": 0.2, "v_int4_rht": 0.1}\n'
        'ARC_KV4_ROUNDTRIP {"layer_order": 1, "k_int4_rht": 0.3, "k_int4_plain": 0.4, "v_int4_rht": 0.1}\n')
    spec = importlib.util.spec_from_file_location("kv4_drift_analyze", ROOT / "scripts/kv4_drift_analyze.py")
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))
    monkeypatch.setattr(sys, "argv", ["x", str(run_dir), str(tmp_path / "prompts.json")])
    analyze.main()
    summary = json.loads((run_dir / "kv4-drift-summary.json").read_text())
    # The reference repeat is exact on the fake server: zero noise, so only an exact candidate passes.
    assert summary["verdicts"]["kv4-k8v4"]["0K-8K/all"]["passed"]
    assert not summary["verdicts"]["kv4-int4"]["0K-8K/all"]["passed"]
    assert summary["verdicts"]["kv4-int4"]["8K-16K/all"] is None
    evidence = summary["servers"]["kv4-int4"]
    assert evidence["layout"]["num_blocks"] == 361 and evidence["layout"]["groups"][0]["block_size"] == 3136
    assert abs(evidence["roundtrip_mean"]["k_int4_rht"] - 0.2) < 1e-9 and evidence["selfcheck"][0]["mode"] == "int4"
    assert summary["greedy"]["kv4-int4"]["prompts"] == 2
    assert "✗" in capsys.readouterr().out
