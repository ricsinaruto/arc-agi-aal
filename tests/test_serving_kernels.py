"""Kernel work, phase 4: the serving overlay (hash guards, plumbing), the MTP-3 admission arm, its capacity fit,
the kernel smoke build and its Modal test. CPU only: the runtime kernels need a GPU and are measured by
scripts/kernel_bench_phase4.py; their pure-torch parts (the top-k-first mask) are checked here when torch is
installed."""

import ast
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from test_arena_mtp0_wide import served_layout
from test_arena_throughput_bench import FakeServer

from arc_infra import dense_kernels_patch as dense
from arc_infra import serving_kernels_patch as patch
from arena import throughput_bench as tb
from arena.cache import BlockBudget
from arena.capacity import focus_fit
from arena.variants import COMPOSED, SERVING_CHANGED, VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
ALL_ON = {"mamba_ssm_cache_dtype": "bfloat16", "moe_backend": "flashinfer_b12x", "draft_experts": "nvfp4",
          "hc_fusion": True, "topk_sampler": True}
# The extracted pinned runtime (optional): ARC_PINNED_SITE=<runtime>/usr/local/lib/python3.12/dist-packages.
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))


def qsa_synthetic_text():
    """The six nvidia/qsa.py anchors in a compilable class skeleton (8-bit KV option)."""
    return ("class Backend:\n" + patch.QSA_BACKEND
            + "\n\nclass Impl:\n    def __init__(self, *args, **kwargs):\n" + patch.QSA_IMPL_INIT
            + "            pass\n" + patch.QSA_IMPL_GUARD + "        self, query, kv_cache, output, num_tokens):\n"
            + patch.QSA_READ + "        return key_cache\n"
            + "\n\nclass Layer:\n    def __init__(self, cache_config):\n" + patch.QSA_LAYER_GUARD
            + patch.QSA_STORAGE_GUARD + "            pass\n")


def synthetic_site(tmp_path, monkeypatch):
    """Each pinned target as a tiny file holding exactly its anchors, with the hash guards pointed at it."""
    site = tmp_path / "site"
    originals = {}
    for target in patch.TARGETS:
        text = "import torch\n\n\ndef f(self, xn, lora, logits, k, p, draft_vllm_config, draft_quant_config):\n"
        body = "".join(anchor * count for anchor, _, count in target["edits"])
        # The HC import anchor sits at module level in the pinned file; keep the synthetic file compilable.
        if target is patch.HYPERCONNECTION:
            text = "from x import (\n    hc_silu,\n)\n\n\ndef f(self, xn, lora):\n"
            body = patch.HC_CHAIN * 2
        elif target is patch.SAMPLER:
            text, body = "import torch\n\n\n", target["edits"][0][0]
        elif target is patch.MTP:
            text = "def f(vllm_config, draft_quant_config, draft_vllm_config):\n"
        elif target is patch.B12X:
            text = "class E:\n    def _ensure_wrapper(self):\n"
        elif target is patch.STATES:
            text = "def f(self, expanded_idx_mapping, do_top_k, idx_mapping_np):\n"
        elif target is patch.QSA:
            text, body = qsa_synthetic_text(), ""
        data = (text + body + ("" if body.endswith("\n") else "\n")).encode()
        compile(data, target["relative"], "exec")
        path = site / target["relative"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        monkeypatch.setitem(target, "sha256", hashlib.sha256(data).hexdigest())
        originals[target["relative"]] = data
    return site, originals


# ----------------------------------------------------------------------------- settings
def test_settings_are_validated_and_canonical():
    assert patch.validate(None) is None
    assert patch.validate({}) is None
    assert patch.validate({"hc_fusion": False, "draft_experts": None}) is None
    # Options added after the phase-4 arm (8-bit KV) stay out of the canonical setting while off.
    assert patch.validate({"topk_sampler": True}) == {
        **{k: None for k in patch.CHOICES if k not in patch.OPTIONAL}, "hc_fusion": False, "topk_sampler": True}
    assert list(patch.validate(ALL_ON)) == [k for k in patch.OPTIONS if k not in patch.OPTIONAL]
    assert patch.validate(patch.PHASE4) == patch.PHASE4
    # The arm after the microbenchmark: target MoE stays on CUTLASS, b12x only for the NvFp4 draft experts,
    # no HC fusion, the top-k-first sampler.
    assert patch.server_flags(patch.PHASE4) == ["--mamba-ssm-cache-dtype", "bfloat16"]
    assert patch.targets(patch.PHASE4) == [patch.MTP, patch.B12X, patch.SAMPLER, patch.STATES]
    for bad in ({"hc_fusion": 1}, {"moe_backend": "cutlass"}, {"draft_experts": "fp8"}, {"fused": True},
                ["hc_fusion"], {"mamba_ssm_cache_dtype": "float32"}):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_targets_and_flags_follow_the_options():
    spec = patch.validate(ALL_ON)
    assert [t["relative"] for t in patch.targets(spec)] == [t["relative"] for t in patch.TARGETS if t is not patch.QSA]
    assert patch.server_flags(spec) == ["--moe-backend", "flashinfer_b12x", "--mamba-ssm-cache-dtype", "bfloat16"]
    only_state = patch.validate({"mamba_ssm_cache_dtype": "bfloat16"})
    assert patch.targets(only_state) == [] and patch.server_flags(only_state) == ["--mamba-ssm-cache-dtype", "bfloat16"]
    # NvFp4 draft experts run on b12x even when the target keeps the CUTLASS MoE: the shared-wrapper patch too.
    draft = patch.validate({"draft_experts": "nvfp4"})
    assert [t["relative"] for t in patch.targets(draft)] == [patch.MTP["relative"], patch.B12X["relative"]]
    assert patch.server_flags(draft) == []
    assert patch.targets(patch.validate({"hc_fusion": True})) == [patch.HYPERCONNECTION]
    assert patch.targets(patch.validate({"topk_sampler": True})) == [patch.SAMPLER, patch.STATES]


# ----------------------------------------------------------------------------- patch
def test_unknown_source_fails_closed():
    for target in patch.TARGETS:
        with pytest.raises(RuntimeError, match="unrecognized"):
            patch.patched_source(target, b"x = 1\n")


def test_patch_applies_only_the_selected_files_and_is_idempotent(tmp_path, monkeypatch):
    site, originals = synthetic_site(tmp_path, monkeypatch)
    spec = {"hc_fusion": True}
    identity = patch.apply(site, tmp_path / "evidence", spec)
    hc = (site / patch.HYPERCONNECTION["relative"]).read_text()
    assert hc.count("_arc_hc_up_mix(self, xn, lora)") == 2 and "hc_silu(lora" not in hc
    assert "from vllm.arc_serving_kernels import hc_up_mix as _arc_hc_up_mix" in hc
    for target in patch.TARGETS:
        if target is not patch.HYPERCONNECTION:  # untouched files stay byte-identical
            assert (site / target["relative"]).read_bytes() == originals[target["relative"]]
    assert (site / patch.HELPER).read_bytes() == (ROOT / "src/arc_infra/serving_kernels_runtime.py").read_bytes()
    assert json.loads((site / patch.SETTINGS).read_text()) == patch.validate(spec)
    assert set(identity["files"]) == {patch.HYPERCONNECTION["relative"]} and identity["server_flags"] == []
    assert json.loads((tmp_path / "evidence" / patch.MARKER).read_text()) == identity
    assert patch.apply(site, tmp_path / "evidence", spec) == identity  # already patched: accepted as is
    (site / patch.HYPERCONNECTION["relative"]).write_bytes(originals[patch.HYPERCONNECTION["relative"]])
    assert patch.apply(site, tmp_path / "evidence", spec) == identity  # re-extracted: patched again
    with pytest.raises(RuntimeError, match="modified"):  # other settings over a patched file
        patch.apply(site, tmp_path / "evidence", {"hc_fusion": True, "draft_experts": "nvfp4"})
    path = site / patch.HYPERCONNECTION["relative"]
    path.write_text(path.read_text() + "\n# tampered")
    with pytest.raises(RuntimeError, match="modified"):
        patch.apply(site, tmp_path / "evidence", spec)
    with pytest.raises(ValueError, match="unpatched"):
        patch.apply(site, tmp_path / "evidence", None)


def test_full_overlay_edits_every_anchor(tmp_path, monkeypatch):
    site, _ = synthetic_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "evidence", ALL_ON)
    assert identity["server_flags"] == patch.server_flags(patch.validate(ALL_ON))
    text = {t["relative"]: (site / t["relative"]).read_text() for t in patch.TARGETS}
    assert "_arc_draft_quant_hook(draft_vllm_config)\n    return draft_vllm_config" in text[patch.MTP["relative"]]
    b12x = text[patch.B12X["relative"]]
    assert "self._wrapper = _arc_shared_b12x_wrapper(self)\n        return" in b12x
    sampler = text[patch.SAMPLER["relative"]]
    assert sampler.index("return logits") < sampler.index("_arc_topk_first(logits, k, p)")
    assert "top_k = _arc_top_k_hint(top_k, self.top_k.np[idx_mapping_np])" in text[patch.STATES["relative"]]


def test_exact_pinned_sources_when_available():
    missing = [t["relative"] for t in patch.TARGETS if not (PINNED_SITE / t["relative"]).is_file()]
    if missing:
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE); the hash guards are checked if supplied")
    for target in patch.TARGETS:
        patched = patch.patched_source(target, (PINNED_SITE / target["relative"]).read_bytes())
        tree = ast.parse(patched)
        functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        if target is patch.MTP:
            body = ast.unparse(functions["_make_draft_vllm_config"])
            assert body.index("_arc_draft_quant_hook(draft_vllm_config)") < body.index("return draft_vllm_config")
        elif target is patch.B12X:
            body = ast.unparse(functions["_ensure_wrapper"])
            assert body.index("_arc_shared_b12x_wrapper(self)") < body.index("B12xMoEWrapper(")
        elif target is patch.HYPERCONNECTION:
            for name in ("mix", "combine_and_mix"):
                body = ast.unparse(functions[name])
                assert "_arc_hc_up_mix(self, xn, lora)" in body and "hc_gate_mix" not in body
        elif target is patch.SAMPLER:
            body = ast.unparse(functions["apply_top_k_top_p"])
            assert body.index("_arc_topk_first(logits, k, p)") < body.index("current_platform.is_cpu()")
        elif target is patch.STATES:
            assert "_arc_top_k_hint" in ast.unparse(functions["get_top_k_top_p"])
        elif target is patch.QSA:
            body = ast.unparse(functions["forward_qsa"])
            assert body.index("_arc_kv8_attention(") < body.index("qsa_sparse_paged_attention(")
            assert "_arc_kv8_cache_update(" in ast.unparse(functions["do_kv_cache_update"])


def test_runtime_helper_entry_points_match_the_patches():
    tree = ast.parse((ROOT / "src/arc_infra/serving_kernels_runtime.py").read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for target in patch.TARGETS:
        for _, replacement, _ in target["edits"]:
            for line in replacement.splitlines():
                if "from vllm.arc_serving_kernels import" in line:
                    assert line.split("import ")[1].split(" as ")[0] in names
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert values["TOPK_MAX"] >= 20 and values["HC_FUSED_MAX_M"] >= 24  # arena top-k 20; MTP-3 x 6 streams
    assert 0 < values["DRAFT_ERROR_LIMIT"] < 1


# ----------------------------------------------------------------------------- arena serving plumbing
def test_serving_applies_the_overlay_and_flags_only_when_requested(tmp_path, monkeypatch):
    from arena import serving

    applied = []
    monkeypatch.setattr(serving, "apply", lambda *a: applied.append("layout"))
    monkeypatch.setattr(serving, "apply_dense_kernels", lambda site, working, spec: applied.append("dense") or {})
    monkeypatch.setattr(serving, "apply_serving_kernels",
                        lambda site, working, spec: applied.append(("serving", spec)) or {"settings": spec})
    monkeypatch.setattr(serving, "clear_compile_cache", lambda root: applied.append("clear") or [])

    def recipe():
        return NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ["serve", "--flag"],
                  RUNTIME_ROOT=tmp_path / "runtime", CACHE_ROOT=tmp_path / "cache", MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X")

    control = recipe()
    receipt = serving.configure(control, tmp_path)
    control.verify_and_extract_runtime()
    assert applied == ["layout"] and "serving_kernels" not in receipt
    assert control.server_command() == ["serve", "--flag"]
    assert receipt == serving.configure(recipe(), tmp_path, serving_kernels={"hc_fusion": False})
    applied.clear()
    patched = recipe()
    receipt2 = serving.configure(patched, tmp_path, dense_kernels=dense.PHASE2, serving_kernels=ALL_ON)
    patched.verify_and_extract_runtime()
    assert applied == ["layout", "clear", "dense", "clear", ("serving", patch.validate(ALL_ON))]
    assert receipt2["serving_kernels"] == patch.validate(ALL_ON)
    assert patched.server_command() == ["serve", "--flag", "--moe-backend", "flashinfer_b12x",
                                        "--mamba-ssm-cache-dtype", "bfloat16"]
    assert json.loads((tmp_path / "serving-kernels-start.json").read_text())["identity"]["settings"]
    clash = recipe()
    clash.server_command = lambda: ["serve", "--moe-backend", "flashinfer_b12x"]
    serving.configure(clash, tmp_path, serving_kernels=ALL_ON)
    with pytest.raises(ValueError, match="--moe-backend"):
        clash.server_command()
    with pytest.raises(ValueError):
        serving.configure(recipe(), tmp_path, serving_kernels={"moe_backend": "triton"})


def test_lab_override_and_study_key_reach_configure(tmp_path, monkeypatch):
    from arena import serving

    assert serving.lab_overrides({"ARENA_SERVING_KERNELS": "off"}) == {"serving_kernels": None}
    assert serving.lab_overrides({"ARENA_SERVING_KERNELS": json.dumps({"hc_fusion": True})}) == {
        "serving_kernels": patch.validate({"hc_fusion": True})}
    with pytest.raises(ValueError):
        serving.lab_overrides({"ARENA_SERVING_KERNELS": json.dumps({"fused": True})})
    config = {"generation_slots": 25, "kv_margin": .9, "server": {"mtp_tokens": 3, "max_num_batched_tokens": 4096},
              "cache_experiment": {"cpu_cache_gib": 0, "kv_cache_dtype": "auto", "reclaim_skipped_blocks": True}}
    calls, hooks = [], []
    monkeypatch.setattr(serving, "start", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(serving, "configure", lambda recipe, working, **kwargs: hooks.append(kwargs))
    monkeypatch.setattr(serving.BlockBudget, "load", lambda *args: NS(layout={}))
    study = tmp_path / "study.json"

    def run(server, env=None):
        study.write_text(json.dumps({**config, "server": {**config["server"], **server}}))
        monkeypatch.delenv("ARENA_SERVING_KERNELS", raising=False)
        if env is not None:
            monkeypatch.setenv("ARENA_SERVING_KERNELS", env)
        monkeypatch.setattr("sys.argv", ["arena.serving", "--working", str(tmp_path), "--input-root", str(tmp_path),
                                         "--study", str(study)])
        serving.main()
        calls[-1]["recipe_hook"](None, tmp_path)
        return hooks[-1]["serving_kernels"]

    assert run({}) is None
    assert run({"serving_kernels": patch.PHASE4}) == patch.PHASE4
    assert run({"serving_kernels": patch.PHASE4}, env="off") is None
    assert calls[-1]["mtp_tokens"] == 3


# ----------------------------------------------------------------------------- variant and capacity
def test_phase4_variant_is_mtp0_wide_scheduling_on_mtp3_with_the_overlays():
    wide = apply_variant(CONTROL, "fixed-2x-mtp0-wide")
    arm = apply_variant(CONTROL, "fixed-2x-mtp3-wide-kernels")
    assert "fixed-2x-mtp3-wide-kernels" in VARIANTS and "fixed-2x-mtp3-wide-kernels" in COMPOSED
    assert "fixed-2x-mtp3-wide-kernels" in SERVING_CHANGED
    assert arm["server"] == {**wide["server"], "mtp_tokens": patch.PHASE4_MTP_TOKENS, "dense_kernels": dense.PHASE2,
                             "serving_kernels": patch.PHASE4}
    assert arm["server"]["mtp_tokens"] == CONTROL["server"]["mtp_tokens"] == 3
    a, b = json.loads(json.dumps(arm)), json.loads(json.dumps(wide))
    a.pop("server"), b.pop("server"), a.pop("arena_variant"), b.pop("arena_variant")
    assert a == b  # six 64K focus games, the six-clear opening, require_focus_windows_fit: as mtp0-wide
    assert arm["active_limit"] == 6 and arm["windows"] == [65536] and arm["opening_switch_level1_games"] == 6
    assert arm["require_focus_windows_fit"] is True
    difference = arm["arena_variant"]["reference_difference"]
    assert "server.mtp_tokens" not in difference  # MTP-3 is the fixed-2x base's depth
    assert difference["server.serving_kernels"] == {"before": None, "after": patch.PHASE4}
    assert difference["server.dense_kernels"] == {"before": None, "after": dense.PHASE2}
    assert arm["arena_variant"]["checkpoint_and_serving_unchanged"] is False
    for other in ("fixed-2x-mtp0-wide", "fixed-2x-mtp0-wide-kernels", "fixed-2x-deep-explore"):
        assert "serving_kernels" not in apply_variant(CONTROL, other)["server"]
    from arena.trial import duck_config

    cfg = duck_config(arm, ROOT, window=65536, cap=arm["completion_cap"], wall=3600, base_url="http://model/v1")
    assert cfg.speculative_tokens == 3 and cfg.max_model_len == 65536 and cfg.parallel_games == 25
    cfg.__post_init__()  # the agent's own validation accepts lookahead 3 at the 64K window


def phase4_layout(num_blocks):
    """MTP-3 with a bf16 GDN state: 832-token blocks (16 * cdiv(10240*6*2 + 48*128*128*2, 16*2048)), the same
    fixed hybrid state as today's MTP-3 (1 + 4 x 5 blocks) and lookahead 3."""
    layout = served_layout(3)
    page = 10240 * 6 * 2 + 48 * 128 * 128 * 2
    block = 16 * math.ceil(page / (16 * 2048))
    assert block == 832
    rows = []
    for row in layout["costs"]:
        groups = [math.ceil(row["tokens"] / block), 1, 5, 5, 5, 5]
        rows.append({"tokens": row["tokens"], "blocks": sum(groups), "groups": groups})
    return {**layout, "num_blocks": num_blocks, "costs": rows}


def test_six_full_windows_fit_the_projected_phase4_layout_and_the_trial_refuses_less():
    # Phase 3 projection (kernels-phase3-sizing.md §4): 628 gate blocks with a bf16 state, the A+C-freed pool and
    # NvFp4 draft experts; the smoke records the served value.
    budget = BlockBudget(phase4_layout(698), 0.9)
    assert budget.capacity == 628
    fit = focus_fit(budget, 65536, 6, required=True)
    assert fit["lookahead"] == 3 and fit["cost_per_full_window_request"] == 80 + 21
    assert fit["full_window_requests"] == 6
    # Without the NvFp4 draft experts (FP8: 584 gate blocks) five fit, and the arm refuses to start.
    with pytest.raises(RuntimeError, match="only 5 fit"):
        focus_fit(BlockBudget(phase4_layout(649), 0.9), 65536, 6, required=True)


# ----------------------------------------------------------------------------- smoke bench, build, Modal test
async def test_bench_records_sanity_answers_and_the_focus_fit(tmp_path, monkeypatch):
    server = FakeServer()
    greedy = []

    def handler(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path.endswith("/chat/completions") and body.get("temperature") == 0.0 and body["max_tokens"] == 1:
            tokens = 40 + (25 if body.get("tools") else 0)
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}],
                                             "usage": {"prompt_tokens": tokens, "completion_tokens": 1}})
        if request.url.path.endswith("/chat/completions") and body.get("temperature") == 0.0:
            greedy.append(body)
            answer = "391" if "17" in body["messages"][0]["content"] else "Lyon"
            return httpx.Response(200, json={"choices": [{"message": {"content": answer}}],
                                             "usage": {"prompt_tokens": 9, "completion_tokens": 2}})
        return server.handler(request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(*a, **{**k, "transport": transport}))
    monkeypatch.setattr(tb.GpuMonitor, "run", lambda self: asyncio.sleep(3600))
    (tmp_path / "server.log").write_text("worker ARC_CACHE_LAYOUT " + json.dumps(phase4_layout(698)) + "\n")
    study = {"kv_margin": 0.9, "generation_slots": 25, "completion_cap": 16384, "reasoning_effort": "xhigh",
             "temperature": 0.7, "server": {"mtp_tokens": 3}, "active_limit": 6, "windows": [65536],
             "throughput_bench": {"prompt_tokens": [], "concurrency": [], "output_tokens": 16, "shared_prefix": None,
                                  "points": [[300, 1]], "tool_choice_probe": True,
                                  "sanity": [{"prompt": "What is 17 * 23?", "expect": "391"},
                                             {"prompt": "Capital of France?", "expect": "Paris"}]}}
    report = await tb.run_bench(study, {"name": "serving-bench", "kind": "serving-bench"}, output=tmp_path / "out",
                                server_dir=tmp_path, base_url="http://model/v1", transport=transport,
                                log=lambda *a, **k: None)
    assert [s["ok"] for s in report["sanity"]] == [True, False] and report["sanity"][1]["text"] == "Lyon"
    assert all(b["chat_template_kwargs"]["enable_thinking"] is False and b["max_tokens"] == 64 for b in greedy)
    assert report["focus_fit"]["full_window_requests"] == 6 and report["focus_fit"]["required"] is False
    probe = report["tool_choice_probe"]
    assert probe["ok"] and probe["none"]["prompt_tokens"] == probe["auto"]["prompt_tokens"] == 65
    assert probe["no_tools"]["prompt_tokens"] == 40
    assert len(report["points"]) == 1 and report["status"] == "complete"


def load_builder():
    spec = importlib.util.spec_from_file_location("phase4_builder", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


C110 = dict(paired_30m=True, opening_burst_requests=3, exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"],
            wall_seconds=3600, clone_instances=110)


def test_arena_bundle_differs_from_mtp0_wide_only_in_the_server(tmp_path):
    builder = load_builder()
    _, wide = builder.build("c110-mtp0-wide", output=tmp_path / "wide", arena_variant="fixed-2x-mtp0-wide", **C110)
    output, arm = builder.build("c110-mtp3-wide-kernels", output=tmp_path / "arm",
                                arena_variant="fixed-2x-mtp3-wide-kernels", **C110)
    assert json.loads((output / "configs/comparison.json").read_text()) == arm
    assert arm["trials"] == [{"name": "duck-transfer", "kind": "duck-transfer"}]
    assert (output / "src/arc_infra/serving_kernels_runtime.py").is_file()
    a, b = json.loads(json.dumps(arm)), json.loads(json.dumps(wide))
    for key in ("name", "arena_variant"):
        a.pop(key), b.pop(key)
    assert a.pop("server") == {**b.pop("server"), "mtp_tokens": 3, "dense_kernels": dense.PHASE2,
                               "serving_kernels": patch.PHASE4}
    assert a == b  # no KV pin (no server_env), same roster, clones, wall and scheduling


def test_kernel_smoke_build(tmp_path):
    builder = load_builder()
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("step 1\n" * 1000)
    output, study = builder.build("kernel-smoke", output=tmp_path / "smoke", paired_30m=True, throughput_bench=True,
                                  arena_variant="fixed-2x-mtp3-wide-kernels", kernel_smoke=True,
                                  bench_transcript=str(transcript))
    assert study["trials"] == [{"name": "serving-bench", "kind": "serving-bench"}]
    grid = study["throughput_bench"]
    assert grid["points"] == [[16384, 1], [16384, 6], [64512, 1], [64512, 6]]
    assert 64512 + grid["output_tokens"] + 256 <= 65536 + 256  # a full 64K window, as focus_fit prices it
    assert grid["transcript"]["prompt_tokens"] == [16384] and grid["transcript"]["concurrency"] == [1, 6]
    assert (output / grid["transcript"]["file"]).read_text() == transcript.read_text()
    assert [s["expect"] for s in grid["sanity"]] == ["391", "Paris"] and "transcript_points" not in grid
    assert study["server"]["serving_kernels"] == patch.PHASE4 and study["active_limit"] == 6
    assert study["notebook_seconds"] <= 1500 - 60 and study["wall_seconds"] + 480 <= study["notebook_seconds"] - 800
    with pytest.raises(ValueError, match="kernel smoke"):
        builder.build("bad", output=tmp_path / "bad", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp3-wide-kernels")
    with pytest.raises(ValueError, match="kernel smoke"):
        builder.build("bad2", output=tmp_path / "bad2", paired_30m=True, throughput_bench=True,
                      arena_variant="cache-reclaim", kernel_smoke=True)


def test_modal_kernel_smoke_test_and_microbench_budgets():
    source = (ROOT / "scripts/modal_validation.py").read_text()
    tree = ast.parse(source)
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "kernel-smoke")
    assert ns["arena_arguments"]("kernel-smoke") == ["--trial", "serving-bench"]
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 3.0  # the user's budget for the smoke
    bench = ast.parse((ROOT / "scripts/modal_kernel_bench_phase4.py").read_text())
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in bench.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("TIMEOUT", "CPU_CEILING", "MEMORY_CEILING_GIB")}
    rate = 0.000842 + consts["CPU_CEILING"] * 0.0000131 + consts["MEMORY_CEILING_GIB"] * 0.00000222
    assert (consts["TIMEOUT"] + 60) * rate * 1.15 <= 1.0  # the user's budget for the microbenchmark


# ----------------------------------------------------------------------------- top-k-first (pure torch)
def load_helper(monkeypatch):
    torch = pytest.importorskip("torch")
    if importlib.util.find_spec("triton") is None:  # the kernels are GPU-only; the mask is plain torch
        stub = types.ModuleType("triton")
        stub.jit = lambda f=None, **k: f if f is not None else (lambda g: g)
        stub.cdiv = lambda a, b: -(-a // b)
        stub.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
        language = types.ModuleType("triton.language")
        language.constexpr = object
        stub.language = language
        monkeypatch.setitem(sys.modules, "triton", stub)
        monkeypatch.setitem(sys.modules, "triton.language", language)
    spec = importlib.util.spec_from_file_location("arc_serving_kernels_test",
                                                  ROOT / "src/arc_infra/serving_kernels_runtime.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return torch, helper


def pinned_sort(torch, logits, k, p):
    """The pinned apply_top_k_top_p_pytorch."""
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    top_k_mask = logits_sort.gather(1, (logits_sort.size(1) - k.to(torch.long)).unsqueeze(dim=1))
    logits_sort.masked_fill_(logits_sort < top_k_mask, -float("inf"))
    if p is not None:
        probs_sum = torch.cumsum(logits_sort.softmax(dim=-1), dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def test_topk_first_matches_the_pinned_mask(monkeypatch):
    torch, helper = load_helper(monkeypatch)
    gen = torch.Generator().manual_seed(0)
    k = torch.tensor([20, 20, 5, 1, 40, 20])
    p = torch.tensor([0.95, 1.0, 0.8, 0.95, 0.5, 0.99])
    rows = mismatched = 0
    for _ in range(8):
        base = torch.randn(6, 50000, generator=gen) * 2.5
        base.scatter_(1, torch.randint(0, 50000, (6, 40), generator=gen), torch.rand(6, 40, generator=gen) * 4 + 8)
        base = base.to(torch.bfloat16).float() / 0.7  # BF16 logits: ties happen
        ref = pinned_sort(torch, base.clone(), k, p)
        new = helper.topk_first_impl(base.clone(), k, p, 40)
        both = torch.isfinite(ref) & torch.isfinite(new)
        assert torch.equal(ref[both], new[both])
        same = (torch.isfinite(ref) == torch.isfinite(new)).all(1)
        # A differing row may only swap equally likely tied tokens at the top-p cut: same count, same values.
        for r in torch.nonzero(~same).flatten().tolist():
            assert torch.isfinite(ref[r]).sum() == torch.isfinite(new[r]).sum()
            assert torch.equal(ref[r][torch.isfinite(ref[r])].sort().values, new[r][torch.isfinite(new[r])].sort().values)
        rows += 6
        mismatched += int((~same).sum())
    # Which of several tokens tied at the top-p cut stays is up to each torch version's sort/topk ordering; the
    # per-row checks above already require a pure equal-logit swap. On these BF16 logits ~10% of rows swap (200
    # seeds, torch 2.14 CPU: mean 5.0 of 48, p99 9, max 10; seed 0 gives 3), so the old 5% cap failed on 87% of
    # seeds. A quarter of the rows only guards against a systematic mask bug.
    assert mismatched <= rows // 4
    only_k = torch.randn(3, 5000, generator=gen)
    kk = torch.tensor([20, 3, 7])
    assert torch.equal(torch.isfinite(pinned_sort(torch, only_k.clone(), kk, None)),
                       torch.isfinite(helper.topk_first_impl(only_k.clone(), kk, None, 20)))
    # The gate: only with the host-side hint, a small top-k and a modest batch.
    logits = torch.randn(2, 100)
    assert helper.topk_first(logits, torch.tensor([20, 20]), None) is None
    hinted = torch.tensor([20, 20])
    hinted._arc_max_k = 20
    assert helper.topk_first(logits.clone(), hinted, None) is not None
    one = torch.tensor([20])
    one._arc_max_k = 20
    assert helper.topk_first(logits[:1].clone(), one, None) is None  # one row: the pinned sort is faster
    hinted._arc_max_k = 100
    assert helper.topk_first(logits.clone(), hinted, None) is None
    import numpy as np

    marked = helper.top_k_hint(torch.tensor([5, 20]), np.array([5, 20]))
    assert marked._arc_max_k == 20 and helper.top_k_hint(None, np.array([1])) is None
