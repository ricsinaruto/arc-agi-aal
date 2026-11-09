"""Serving tweaks (28 September): the 16K-token MTP draft head, the PLE table read over UVA and the host-resident
input embedding behind ``server.serving_kernels`` (hash guards, flag plumbing, the token subset and its exactness,
the tweaks arms and their builds). CPU only; the pure-torch parts run when torch is installed."""

import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_serving_kernels import C110, load_builder, load_helper, phase4_layout, synthetic_site

from arc_infra import dense_kernels_patch as dense
from arc_infra import serving_kernels_patch as patch
from arena.cache import BlockBudget
from arena.capacity import focus_fit
from arena.variants import COMPOSED, DEPTHWISE_SERVING, SERVING_CHANGED, VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
VOCAB = json.loads((ROOT / "src/arc_infra/serving_draft_vocab.json").read_text())
# The extracted pinned runtime (optional) and the recipe-patched PLE layer (optional: the pinned PLE patch applied
# to the stock file, sha256 a8a0647...).
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
PINNED_PLE = Path(os.environ.get("ARC_PINNED_PLE", "/tmp/arc-pinned-ple/ple_layer.py"))


def tweak_site(tmp_path, monkeypatch):
    """The phase-4 synthetic site plus model.py and the recipe-patched ple_layer.py, each holding its anchors."""
    site, originals = synthetic_site(tmp_path, monkeypatch)
    mtp = site / patch.MTP["relative"]
    data = mtp.read_bytes() + ("class M:\n    def compute_logits(\n        self, hidden_states: torch.Tensor, "
                               "spec_step_idx: int = 0\n" + patch.MTP_DRAFT_LOGITS_EDIT[0]).encode()
    compile(data, "mtp.py", "exec")
    mtp.write_bytes(data)
    monkeypatch.setitem(patch.MTP, "sha256", hashlib.sha256(data).hexdigest())
    originals[patch.MTP["relative"]] = data
    texts = {patch.MODEL["relative"]: "class Model:\n    def __init__(self, config):\n" + patch.MODEL["edits"][0][0],
             patch.PLE["relative"]: ("class Layer:\n    def __init__(self):\n" + patch.PLE["edits"][0][0]
                                     + "            self.ple_embedding = Embedding(\n                x,\n"
                                     + patch.PLE["edits"][1][0] + "            1)\n"
                                     + "        gate = (\n" + patch.PLE["edits"][2][0] + "        )\n")}
    for target in (patch.MODEL, patch.PLE):
        data = texts[target["relative"]].encode()
        compile(data, target["relative"], "exec")
        path = site / target["relative"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        monkeypatch.setitem(target, "sha256", hashlib.sha256(data).hexdigest())
        originals[target["relative"]] = data
    return site, originals


# ----------------------------------------------------------------------------- settings and targets
def test_phase4_settings_stay_byte_identical_and_tweaks_are_opt_in():
    assert patch.validate(patch.PHASE4) == patch.PHASE4 and list(patch.validate(patch.PHASE4)) == list(patch.PHASE4)
    assert patch.validate({"topk_sampler": True}) == {**{k: None for k in ("mamba_ssm_cache_dtype", "moe_backend",
                                                                            "draft_experts")},
                                                      "hc_fusion": False, "topk_sampler": True}
    tweaks = patch.validate(patch.TWEAKS)
    assert tweaks == patch.TWEAKS and list(tweaks) == [k for k in patch.OPTIONS if k not in ("kv_cache_8bit", "kv_cache_4bit", "kv8_prefill", "expert_int3")]
    assert patch.TWEAKS == {**patch.PHASE4, "draft_vocab": "arena16k", "ple_uva": True, "embed_host": True}
    assert patch.validate({"ple_uva": True}) == {**{k: None for k in patch.CHOICES if k not in patch.OPTIONAL},
                                                 "hc_fusion": False, "topk_sampler": False, "ple_uva": True}
    assert patch.validate({"ple_uva": False, "embed_host": False, "draft_vocab": None}) is None
    for bad in ({"draft_vocab": "all"}, {"draft_vocab": 16384}, {"ple_uva": 1}, {"embed_host": "yes"},
                {"language_model_only": True}):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_tweak_targets_and_environment():
    phase4 = patch.targets(patch.PHASE4)
    assert phase4 == [patch.MTP, patch.B12X, patch.SAMPLER, patch.STATES] and patch.ple_target(patch.PHASE4) is None
    tweaks = patch.targets(patch.TWEAKS)
    assert [t["relative"] for t in tweaks] == [patch.MTP["relative"], patch.B12X["relative"], patch.SAMPLER["relative"],
                                               patch.STATES["relative"], patch.MODEL["relative"]]
    assert tweaks[0]["edits"] == [*patch.MTP["edits"], patch.MTP_DRAFT_LOGITS_EDIT]
    assert patch.ple_target(patch.TWEAKS) is patch.PLE
    only_vocab = patch.targets(patch.validate({"draft_vocab": "arena16k"}))
    assert only_vocab == [{**patch.MTP, "edits": [patch.MTP_DRAFT_LOGITS_EDIT]}]
    assert patch.targets(patch.validate({"embed_host": True})) == [patch.MODEL]
    assert patch.targets(patch.validate({"ple_uva": True})) == []  # patched later, after the recipe's PLE patch
    assert patch.server_flags(patch.TWEAKS) == patch.server_flags(patch.PHASE4)
    env = {"VLLM_PLE_CPU_OFFLOAD": "1", "OTHER": "x"}
    assert patch.server_environment(patch.TWEAKS, env) == {"VLLM_PLE_CPU_OFFLOAD": "0", "OTHER": "x"}
    assert env["VLLM_PLE_CPU_OFFLOAD"] == "1"  # not mutated
    assert patch.server_environment(patch.PHASE4, env) == env
    with pytest.raises(RuntimeError, match="VLLM_PLE_CPU_OFFLOAD"):
        patch.server_environment(patch.TWEAKS, {})


def test_draft_vocabulary_file():
    ids = VOCAB["ids"]
    assert VOCAB["name"] == "arena16k" and VOCAB["size"] == len(ids) == 16384 and VOCAB["vocab_rows"] == 248320
    assert ids == sorted(set(ids)) and 0 <= ids[0] and ids[-1] < 248320
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == VOCAB["ids_sha256"]
    assert set(range(248044, 248077)) <= set(ids)  # every special/added token (<|im_end|>, <think>, tool calls)
    assert VOCAB["mandatory"] == {"byte_tokens": 256, "special_tokens": 33}
    assert VOCAB["corpus"]["runs"] == 43 and VOCAB["corpus_coverage"] > 0.999
    assert patch.draft_vocab_bytes(patch.TWEAKS) == (ROOT / "src/arc_infra/serving_draft_vocab.json").read_bytes()
    assert patch.draft_vocab_bytes(patch.PHASE4) is None


def test_draft_vocabulary_rule(tmp_path, monkeypatch):
    """scripts/specialization_vocab.py draft-vocab on a toy corpus: bytes and specials first, then by frequency."""
    import importlib.util

    np = pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location("spec_vocab", ROOT / "scripts/specialization_vocab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = tmp_path / "tokenizer.json"
    alphabet = {}
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    extra = 0
    for b in range(256):
        alphabet[chr(b) if b in printable else chr(256 + (extra := extra + 1) - 1)] = 1000 + b
    tokenizer.write_text(json.dumps({"added_tokens": [{"id": 2000}, {"id": 2001}, {"id": 999999}],
                                     "model": {"vocab": alphabet}}))
    monkeypatch.setattr(module, "TOKENIZER", str(tokenizer))
    monkeypatch.setattr(module, "V", 3000)
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez_compressed(cache / "run.npz", out_tokens=np.array([5, 5, 5, 7, 7, 9, 1001, 2500], np.int32))
    (tmp_path / "analysis.json").write_text(json.dumps({"runs": ["run"]}))
    module.draft_vocab(str(cache), str(tmp_path / "analysis.json"), str(tmp_path / "out.json"), size=261)
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["ids"] == sorted([*range(1000, 1256), 2000, 2001, 5, 7, 9])  # 256 + 2 specials + top 3 (not 2500)
    assert out["mandatory"] == {"byte_tokens": 256, "special_tokens": 2}


# ----------------------------------------------------------------------------- patch
def test_tweaks_patch_every_anchor_copy_the_subset_and_stay_idempotent(tmp_path, monkeypatch):
    site, originals = tweak_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "evidence", patch.TWEAKS)
    mtp = (site / patch.MTP["relative"]).read_text()
    assert "_arc_draft_quant_hook(draft_vllm_config)" in mtp and "return _arc_draft_logits(self, hidden_states)" in mtp
    assert "self.logits_processor(self.lm_head" not in mtp
    model = (site / patch.MODEL["relative"]).read_text()
    assert "self.embed_tokens = _arc_host_embedding(" in model
    assert (site / patch.PLE["relative"]).read_bytes() == originals[patch.PLE["relative"]]  # not yet
    assert (site / patch.DRAFT_VOCAB).read_bytes() == (ROOT / "src/arc_infra/serving_draft_vocab.json").read_bytes()
    assert identity["draft_vocab_sha256"] == hashlib.sha256((site / patch.DRAFT_VOCAB).read_bytes()).hexdigest()
    assert set(identity["files"]) == {t["relative"] for t in patch.targets(patch.TWEAKS)}
    assert json.loads((site / patch.SETTINGS).read_text()) == patch.TWEAKS
    assert patch.apply(site, tmp_path / "evidence", patch.TWEAKS) == identity  # already patched: accepted
    (site / patch.DRAFT_VOCAB).write_text("{}")
    with pytest.raises(RuntimeError, match="modified"):
        patch.apply(site, tmp_path / "evidence", patch.TWEAKS)
    # The PLE overlay: after the helper and the settings, on the recipe-patched file only.
    ple = patch.apply_ple(site, tmp_path / "evidence", patch.TWEAKS)
    text = (site / patch.PLE["relative"]).read_text()
    assert text.index("_arc_ple_uva_device(PleOffloadLayer.get_target_device())") < text.index(
        "self.ple_embedding = Embedding(") < text.index("_arc_ple_uva_place(self.ple_embedding)")
    assert ple["original_sha256"] == patch.PLE["sha256"] and ple["settings"] == patch.TWEAKS
    assert json.loads((tmp_path / "evidence" / patch.PLE_MARKER).read_text()) == ple
    with pytest.raises(RuntimeError, match="unrecognized"):  # never twice, never on the stock file
        patch.apply_ple(site, tmp_path / "evidence", patch.TWEAKS)
    with pytest.raises(ValueError, match="unpatched"):
        patch.apply_ple(site, tmp_path / "evidence", patch.PHASE4)


def test_ple_overlay_needs_this_studys_helper_and_settings(tmp_path, monkeypatch):
    site, _ = tweak_site(tmp_path, monkeypatch)
    patch.apply(site, tmp_path / "evidence", patch.PHASE4)
    with pytest.raises(RuntimeError, match="helper and settings"):
        patch.apply_ple(site, tmp_path / "evidence", patch.TWEAKS)


def test_phase4_arm_patches_exactly_as_before(tmp_path, monkeypatch):
    site, originals = tweak_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "evidence", patch.PHASE4)
    assert set(identity["files"]) == {t["relative"] for t in patch.targets(patch.PHASE4)}
    assert "draft_vocab_sha256" not in identity and not (site / patch.DRAFT_VOCAB).exists()
    assert (site / patch.MODEL["relative"]).read_bytes() == originals[patch.MODEL["relative"]]
    assert "_arc_draft_logits" not in (site / patch.MTP["relative"]).read_text()
    assert (site / patch.SETTINGS).read_bytes() == (json.dumps(patch.PHASE4, sort_keys=True) + "\n").encode()


def test_exact_pinned_sources_when_available():
    missing = [t["relative"] for t in (patch.MTP, patch.MODEL) if not (PINNED_SITE / t["relative"]).is_file()]
    if missing:
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE); the hash guards are checked if supplied")
    for target in patch.targets(patch.TWEAKS):
        patch.patched_source(target, (PINNED_SITE / target["relative"]).read_bytes())
    mtp = patch.patched_source(patch.targets(patch.TWEAKS)[0], (PINNED_SITE / patch.MTP["relative"]).read_bytes())
    functions = {n.name: n for n in ast.walk(ast.parse(mtp)) if isinstance(n, ast.FunctionDef)}
    assert ast.unparse(functions["compute_logits"]).strip().endswith("return _arc_draft_logits(self, hidden_states)")
    model = patch.patched_source(patch.MODEL, (PINNED_SITE / patch.MODEL["relative"]).read_bytes())
    classes = {n.name: n for n in ast.walk(ast.parse(model)) if isinstance(n, ast.ClassDef)}
    init = next(n for n in classes["Qwen3_8FlashNextModel"].body if getattr(n, "name", "") == "__init__")
    assert "self.embed_tokens = _arc_host_embedding(lambda: VocabParallelEmbedding(" in ast.unparse(init)
    if not PINNED_PLE.is_file():
        pytest.skip("Optional recipe-patched ple_layer.py (ARC_PINNED_PLE)")
    ple = patch.patched_source(patch.PLE, PINNED_PLE.read_bytes())
    classes = {n.name: n for n in ast.walk(ast.parse(ple)) if isinstance(n, ast.ClassDef)}
    init = ast.unparse(next(n for n in classes["Qwen3_8FlashNextPLELayer"].body if getattr(n, "name", "") == "__init__"))
    assert (init.index("_arc_ple_uva_device(PleOffloadLayer.get_target_device())")
            < init.index("Qwen3_8FlashNextNGramEmbedding(") < init.index("_arc_ple_uva_place(self.ple_embedding)")
            < init.index("self.key_proj = ReplicatedLinear("))


def test_runtime_helper_has_the_tweak_entry_points():
    tree = ast.parse((ROOT / "src/arc_infra/serving_kernels_runtime.py").read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    replacements = [r for t in patch.TWEAK_TARGETS for _, r, _ in t["edits"]] + [patch.MTP_DRAFT_LOGITS_EDIT[1]]
    for replacement in replacements:
        for line in replacement.splitlines():
            if "from vllm.arc_serving_kernels import" in line:
                assert line.split("import ")[1].split(" as ")[0] in names


# ----------------------------------------------------------------------------- arena serving plumbing
def recipe(tmp_path, calls):
    return NS(verify_and_extract_runtime=lambda: calls.append("extract") or {},
              server_command=lambda: ["serve"], RUNTIME_ROOT=tmp_path / "runtime", CACHE_ROOT=tmp_path / "cache",
              MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X",
              patch_ple_layer=lambda: calls.append("recipe-ple") or {"patched": True},
              runtime_environment=lambda **kw: ({"VLLM_PLE_CPU_OFFLOAD": "1", "A": "b"}, {"validation_mode": "fast"}))


def test_serving_wraps_the_recipe_ple_patch_and_environment_only_for_ple_uva(tmp_path, monkeypatch):
    from arena import serving

    calls = []
    monkeypatch.setattr(serving, "apply", lambda *a: calls.append("layout"))
    monkeypatch.setattr(serving, "apply_dense_kernels", lambda site, working, spec: calls.append("dense") or {})
    monkeypatch.setattr(serving, "clear_compile_cache", lambda root: [])
    monkeypatch.setattr(serving, "apply_serving_kernels", lambda site, working, spec: calls.append("serving") or {})
    monkeypatch.setattr(serving, "apply_serving_kernels_ple",
                        lambda site, working, spec: calls.append(("ple", spec)) or {"settings": spec})
    phase4 = recipe(tmp_path, calls)
    original_ple, original_env = phase4.patch_ple_layer, phase4.runtime_environment
    serving.configure(phase4, tmp_path, dense_kernels=dense.PHASE2, serving_kernels=patch.PHASE4)
    assert phase4.patch_ple_layer is original_ple and phase4.runtime_environment is original_env
    tweaks = recipe(tmp_path, calls)
    receipt = serving.configure(tweaks, tmp_path, dense_kernels=dense.PHASE2, serving_kernels=patch.TWEAKS)
    assert receipt["serving_kernels"] == patch.TWEAKS
    calls.clear()
    tweaks.verify_and_extract_runtime()
    assert tweaks.patch_ple_layer() == {"patched": True}
    assert calls == ["extract", "layout", "dense", "serving", "recipe-ple"]  # the recipe checks its PLE hash next
    env, check = tweaks.runtime_environment(deep_preload_validation=False, tuning=None)
    # Patched only after runtime_environment's own VLLM_PLE_PATCHED_SHA256 check (smoke v2 failed on that check).
    assert calls[-1] == ("ple", patch.TWEAKS)
    assert env == {"VLLM_PLE_CPU_OFFLOAD": "0", "A": "b"}
    assert check == {"validation_mode": "fast", "serving_kernels_environment": {"VLLM_PLE_CPU_OFFLOAD": "0"}}
    assert json.loads((tmp_path / "serving-kernels-ple-start.json").read_text())["identity"]["settings"] == patch.TWEAKS
    assert tweaks.server_command() == ["serve", "--mamba-ssm-cache-dtype", "bfloat16"]
    # The lab override reaches the tweaks too (ARENA_SERVING_KERNELS).
    assert serving.lab_overrides({"ARENA_SERVING_KERNELS": json.dumps({"ple_uva": True})}) == {
        "serving_kernels": patch.validate({"ple_uva": True})}


def test_ple_overlay_passes_the_recipes_own_hash_check_on_the_pinned_files(tmp_path, monkeypatch):
    """The recipe's order (serving_setup.main): extract, patch_ple_layer, runtime_environment (which refuses a
    ple_layer.py whose sha256 is not VLLM_PLE_PATCHED_SHA256), then the server. On the pinned files, with the real
    overlay functions: the recipe check passes and the served file is the overlay's."""
    import re
    import shutil

    from arena import serving

    targets = patch.targets(patch.TWEAKS)
    if not PINNED_PLE.is_file() or any(not (PINNED_SITE / t["relative"]).is_file() for t in targets):
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE) and recipe-patched ple_layer.py (ARC_PINNED_PLE)")
    expected = re.search(r'^VLLM_PLE_PATCHED_SHA256 = "([0-9a-f]{64})"',
                         (ROOT / "src/arc_infra/serving_setup.py").read_text(), re.M).group(1)
    assert expected == patch.PLE["sha256"] == hashlib.sha256(PINNED_PLE.read_bytes()).hexdigest()
    runtime = tmp_path / "runtime"
    site = runtime / "usr/local/lib/python3.12/dist-packages"
    for target in targets:
        (site / target["relative"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PINNED_SITE / target["relative"], site / target["relative"])
    ple = site / patch.PLE["relative"]
    monkeypatch.setattr(serving, "apply", lambda *a: None)
    monkeypatch.setattr(serving, "clear_compile_cache", lambda root: [])

    def recipe_check(**kwargs):
        if hashlib.sha256(ple.read_bytes()).hexdigest() != expected:
            raise RuntimeError("The patched PLE source hash changed before import.")
        return {"VLLM_PLE_CPU_OFFLOAD": "1"}, {}

    recipe = NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ["serve"], RUNTIME_ROOT=runtime,
                CACHE_ROOT=tmp_path / "cache", MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X",
                patch_ple_layer=lambda: shutil.copyfile(PINNED_PLE, ple), runtime_environment=recipe_check)
    serving.configure(recipe, tmp_path / "work", serving_kernels=patch.TWEAKS)
    recipe.verify_and_extract_runtime()
    recipe.patch_ple_layer()
    env, _ = recipe.runtime_environment(deep_preload_validation=False, tuning=None)
    assert env["VLLM_PLE_CPU_OFFLOAD"] == "0"
    assert ple.read_bytes() == patch.patched_source(patch.PLE, PINNED_PLE.read_bytes())


# ----------------------------------------------------------------------------- variants, fit and builds
def test_tweaks_variant_changes_only_the_serving_overlay():
    kernels = apply_variant(CONTROL, "fixed-2x-mtp3-wide-kernels")
    arm = apply_variant(CONTROL, "fixed-2x-mtp3-wide-tweaks")
    for name in ("fixed-2x-mtp3-wide-tweaks", "fixed-2x-depthwise-mtp3-tweaks"):
        assert name in VARIANTS and name in COMPOSED and name in SERVING_CHANGED
    assert arm["server"] == {**kernels["server"], "serving_kernels": patch.TWEAKS}
    a, b = json.loads(json.dumps(arm)), json.loads(json.dumps(kernels))
    for study in (a, b):
        study.pop("server"), study.pop("arena_variant")
    assert a == b
    difference = arm["arena_variant"]["reference_difference"]
    assert difference["server.serving_kernels"] == {"before": None, "after": patch.TWEAKS}
    assert {k: v for k, v in difference.items() if k != "server.serving_kernels"} == {
        k: v for k, v in kernels["arena_variant"]["reference_difference"].items() if k != "server.serving_kernels"}
    assert arm["require_focus_windows_fit"] is True and arm["active_limit"] == 6 and arm["windows"] == [65536]
    # The depthwise arm on the tweaks serving equals the depthwise arm on the phase-4 serving but for the overlay.
    assert DEPTHWISE_SERVING["fixed-2x-depthwise-mtp3-tweaks"] == "fixed-2x-mtp3-wide-tweaks"
    depth, depth4 = (apply_variant(CONTROL, n) for n in ("fixed-2x-depthwise-mtp3-tweaks", "fixed-2x-depthwise-mtp3"))
    assert depth["server"] == {**depth4["server"], "serving_kernels": patch.TWEAKS}
    assert depth["depthwise_serving_source"] == "fixed-2x-mtp3-wide-tweaks"
    a, b = json.loads(json.dumps(depth)), json.loads(json.dumps(depth4))
    for study in (a, b):
        study.pop("server"), study.pop("arena_variant"), study.pop("depthwise_serving_source")
    assert a == b and depth["level_budget_policy"] == "depthwise"


def test_no_ple_arms_drop_only_the_ple_host_read():
    assert patch.TWEAKS_NO_PLE == {k: v for k, v in patch.TWEAKS.items() if k != "ple_uva"}
    assert patch.validate(patch.TWEAKS_NO_PLE) == patch.TWEAKS_NO_PLE and patch.ple_target(patch.TWEAKS_NO_PLE) is None
    for base, nople in (("fixed-2x-mtp3-wide-tweaks", "fixed-2x-mtp3-wide-tweaks-nople"),
                        ("fixed-2x-depthwise-mtp3-tweaks", "fixed-2x-depthwise-mtp3-tweaks-nople")):
        a, b = apply_variant(CONTROL, nople), apply_variant(CONTROL, base)
        assert a["server"] == {**b["server"], "serving_kernels": patch.TWEAKS_NO_PLE}
        for study in (a, b):
            study.pop("server"), study.pop("arena_variant"), study.pop("depthwise_serving_source", None)
        assert a == b


def test_seventh_window_needs_786_blocks():
    """The 64K fit on the phase-4 layout (101 blocks per full window, 0.9 gate): 691 served blocks give six;
    a seventh needs 786 (+95 blocks = +2.0 GiB at 21.8 MiB per block)."""
    assert focus_fit(BlockBudget(phase4_layout(691), 0.9), 65536, 6, required=True)["full_window_requests"] == 6
    assert focus_fit(BlockBudget(phase4_layout(785), 0.9), 65536, 7)["full_window_requests"] == 6
    assert focus_fit(BlockBudget(phase4_layout(786), 0.9), 65536, 7)["full_window_requests"] == 7


def test_tweaks_bundle_and_smoke_builds(tmp_path):
    builder = load_builder()
    _, kernels = builder.build("c110-mtp3-wide-kernels", output=tmp_path / "k", arena_variant="fixed-2x-mtp3-wide-kernels",
                               **C110)
    output, arm = builder.build("c110-mtp3-wide-tweaks", output=tmp_path / "t", arena_variant="fixed-2x-mtp3-wide-tweaks",
                                **C110)
    assert (output / "src/arc_infra/serving_draft_vocab.json").read_bytes() == (
        ROOT / "src/arc_infra/serving_draft_vocab.json").read_bytes()
    a, b = json.loads(json.dumps(arm)), json.loads(json.dumps(kernels))
    for key in ("name", "arena_variant"):
        a.pop(key), b.pop(key)
    assert a.pop("server") == {**b.pop("server"), "serving_kernels": patch.TWEAKS}
    assert a == b
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("step 1\n" * 1000)
    smoke, study = builder.build("tweaks-smoke", output=tmp_path / "smoke", paired_30m=True, throughput_bench=True,
                                 arena_variant="fixed-2x-mtp3-wide-tweaks", kernel_smoke=True,
                                 bench_transcript=str(transcript))
    assert study["server"]["serving_kernels"] == patch.TWEAKS and study["trials"] == [
        {"name": "serving-bench", "kind": "serving-bench"}]
    assert study["throughput_bench"]["points"] == [[16384, 1], [16384, 6], [64512, 1], [64512, 6]]


def test_modal_tweaks_smoke_budget():
    source = (ROOT / "scripts/modal_validation.py").read_text()
    tree = ast.parse(source)
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "tweaks-smoke")
    assert ns["arena_arguments"]("tweaks-smoke") == ["--trial", "serving-bench"]
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 3.0  # the user's budget for the smoke
    assert "tweaks-smoke" in source.split("def reserve", 1)[1].split("return bound", 1)[0]


# ----------------------------------------------------------------------------- the draft head (pure torch)
class FakeProcessor:
    soft_cap = None
    scale = 1.0
    logits_as_input = False
    head_dtype = None

    def __init__(self, width):
        self.org_vocab_size = width

    def __call__(self, lm_head, hidden_states):
        import torch

        return torch.nn.functional.linear(hidden_states, lm_head.weight)[..., : self.org_vocab_size]


def fake_mtp(torch, vocab=512, hidden=32, rows=4):
    weight = torch.randn(vocab, hidden, generator=torch.Generator().manual_seed(0))
    return NS(logits_processor=FakeProcessor(vocab), lm_head=NS(weight=weight, tp_size=1),
              vllm_config=NS(scheduler_config=NS(max_num_seqs=rows)))


def test_draft_head_over_a_subset(monkeypatch):
    torch, helper = load_helper(monkeypatch)
    monkeypatch.setitem(helper._STATE, "draft_head", None)
    monkeypatch.setitem(helper._STATE, "config", {"draft_vocab": "arena16k"})
    model = fake_mtp(torch)
    ids = sorted({3, 17, 18, 100, 101, 250, 511, *range(40, 80)})
    record = {"name": "t", "size": len(ids), "vocab_rows": 512, "ids": ids}
    monkeypatch.setattr(helper, "load_draft_vocab", lambda path=None: record)
    generator = torch.Generator().manual_seed(1)
    outputs = []
    for _ in range(3):  # MTP-3: three draft steps reuse the one persistent row buffer
        h = torch.randn(3, 32, generator=generator)
        out = helper.draft_logits(model, h)
        full = model.logits_processor(model.lm_head, h)
        index = torch.tensor(ids)
        assert out.shape == full.shape and torch.isneginf(out).sum(1).tolist() == [512 - len(ids)] * 3
        assert torch.allclose(out[:, index], full[:, index], atol=1e-5)
        # The greedy draft is the subset's argmax (a full-vocabulary argmax outside the subset is not proposed).
        assert out.argmax(-1).tolist() == index[full[:, index].argmax(-1)].tolist()
        outputs.append(out.argmax(-1).clone())
    head = helper._STATE["draft_head"]
    assert head is not None and head.buffer.shape == (4, 512) and head.scale is None
    wide = head(torch.randn(6, 32, generator=generator))  # more rows than max_num_seqs: a fresh block
    assert wide.shape == (6, 512) and torch.isneginf(wide).sum() == 6 * (512 - len(ids))
    monkeypatch.setitem(helper._STATE, "draft_head", None)
    monkeypatch.setitem(helper._STATE, "config", {})
    h = torch.randn(2, 32, generator=generator)
    assert torch.equal(helper.draft_logits(model, h), model.logits_processor(model.lm_head, h))  # option off
    assert helper._STATE["draft_head"] is None


def test_draft_head_check_fails_closed(monkeypatch):
    torch, helper = load_helper(monkeypatch)
    model = fake_mtp(torch)
    ids = list(range(0, 512, 7))
    head = helper.DraftHead.from_head(model.lm_head, ids, 512, 4, torch.float32)
    helper.check_draft_head(model, head, 32, torch.float32)
    head.weight = head.weight.flip(0)  # a row-mapping mistake
    with pytest.raises(RuntimeError, match="self-check failed"):
        helper.check_draft_head(model, head, 32, torch.float32)
    with pytest.raises(RuntimeError, match="beyond"):
        helper.build_draft_head(model, torch.zeros(1, 32), record={"ids": [0, 600]}, rows=2)
    model.logits_processor.soft_cap = 30.0
    with pytest.raises(RuntimeError, match="plain TP=1 head"):
        helper.build_draft_head(model, torch.zeros(1, 32), record={"ids": [0, 5]}, rows=2)


def test_rejection_sampling_stays_exact_for_any_draft_token(monkeypatch):
    """vLLM's greedy-draft rule (a one-hot draft q): accept the draft d with probability p(d), otherwise sample the
    normalized residual max(p - q, 0). The emitted distribution is p whatever d is, so a draft restricted to a
    token subset changes only how often the draft is accepted (p(d) summed), never the output."""
    torch, _ = load_helper(monkeypatch)
    p = torch.softmax(torch.randn(50, generator=torch.Generator().manual_seed(2)) * 3, 0).double()
    for d in (int(p.argmax()), 7, 49):
        q = torch.zeros_like(p)
        q[d] = 1.0
        residual = (p - q).clamp(min=0)
        emitted = p[d] * q + (1 - p[d]) * residual / residual.sum()
        assert torch.allclose(emitted, p, atol=1e-12)


def test_ple_probe_is_a_padded_graph_batch(monkeypatch):
    torch, helper = load_helper(monkeypatch)
    ids, qsl, context, real = helper.ple_probe(248320, 248044, 2, torch.Generator().manual_seed(1))
    assert ids.shape == (16,) and ids.dtype == torch.int32 and int(ids[3]) == 248044
    assert qsl.tolist() == [0, 5, 6, 13, 13] and real == 13 < ids.shape[0]
    assert context.shape == (4, 2) and context[3].tolist() == [248044, 248044]


# ----------------------------------------------------------------------------- host tables (CPU stand-ins for UVA)
def cpu_host_tables(monkeypatch, helper, torch, tables):
    """host_table on the CPU: a separate 'device view' buffer per table (the aliasing itself needs a GPU)."""
    monkeypatch.setattr(helper, "_worker_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(helper, "_default_device_type", lambda: "cuda")
    monkeypatch.setattr(helper, "_register_uva_op", lambda: None)  # vLLM's op registry is not installed here
    monkeypatch.setattr(helper, "_register_ple_op", lambda: None)

    def host_table(shape, dtype, device, name):
        tables[name] = (torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype))
        return tables[name]

    monkeypatch.setattr(helper, "host_table", host_table)


def test_host_embedding_loads_through_the_cpu_alias(monkeypatch):
    torch, helper = load_helper(monkeypatch)
    monkeypatch.setitem(helper._STATE, "config", {"embed_host": True})
    tables, checks = {}, []
    cpu_host_tables(monkeypatch, helper, torch, tables)
    monkeypatch.setattr(helper, "_gather_check", lambda *a, **k: checks.append(a[0]) or {})

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(8, 4), requires_grad=False)
            self.weight.weight_loader = self.weight_loader
            self.quant_method = NS(process_weights_after_loading=lambda layer: checks.append("pinned"))

        def weight_loader(self, param, loaded):
            param[: loaded.shape[0]].data.copy_(loaded)

    built = []
    layer = helper.host_embedding(lambda: built.append(torch.empty(0).device.type) or Embedding())
    assert built == ["cpu"]  # the 1.18 GiB table is never allocated on the GPU
    cpu, view = tables["embed_tokens"]
    assert layer.weight.data_ptr() == view.data_ptr()
    layer.weight.weight_loader(layer.weight, torch.arange(32.0).view(8, 4))
    assert torch.equal(cpu, torch.arange(32.0).view(8, 4)) and layer.weight.data_ptr() == view.data_ptr()
    layer.quant_method.process_weights_after_loading(layer)
    assert checks == ["pinned", "embed_tokens"]
    assert layer.quant_method.embedding is helper.uva_embedding
    monkeypatch.setattr(helper, "_default_device_type", lambda: "meta")  # the PLE offload process's meta model
    assert helper.host_embedding(lambda: "pinned construction") == "pinned construction"


def test_ple_table_loads_through_the_cpu_alias_and_gathers_bytes(monkeypatch):
    torch, helper = load_helper(monkeypatch)
    monkeypatch.setitem(helper._STATE, "config", {"ple_uva": True})
    tables, checks = {}, []
    cpu_host_tables(monkeypatch, helper, torch, tables)
    monkeypatch.setattr(helper, "_ple_check", lambda ple, cpu, gpu: checks.append("ple"))

    class Ngram(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(64, 4, dtype=torch.float8_e4m3fn), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.quant_method = NS(process_weights_after_loading=lambda layer: checks.append("pinned"),
                                   embedding=None)

    class Ple(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ngram_embedding = Ngram()
            self.register_buffer("layer_multipliers", torch.tensor([3, 5]))

        def load_weights(self, weights):
            for rows, value in weights:
                self.ngram_embedding.weight.data[rows].copy_(value)
            return {"ngram_embedding.weight"}

    ple = Ple()
    helper.ple_uva_place(ple)
    cpu, view = tables["ple"]
    assert ple.ngram_embedding.weight.data_ptr() == view.data_ptr()
    shard = torch.full((4, 4), 2.0).to(torch.float8_e4m3fn)
    assert ple.load_weights([(slice(0, 4), shard)]) == {"ngram_embedding.weight"}
    assert torch.equal(cpu[:4].view(torch.uint8), shard.view(torch.uint8))
    assert ple.ngram_embedding.weight.data_ptr() == view.data_ptr()
    assert ple.ngram_embedding.quant_method.embedding is helper.uva_embedding  # the opaque op, never Inductor
    gathered = helper._arc_uva_embedding(torch.tensor([1, 0, 1]), cpu)
    fake = helper._arc_uva_embedding_fake(torch.tensor([[1, 0], [1, 1]]), cpu)
    assert fake.shape == (2, 2, 4) and fake.dtype == torch.float8_e4m3fn
    assert gathered.dtype == torch.float8_e4m3fn
    assert torch.equal(gathered.view(torch.uint8), cpu[[1, 0, 1]].view(torch.uint8))
    ple.ngram_embedding.quant_method.process_weights_after_loading(ple.ngram_embedding)
    assert checks == ["pinned", "ple"]
    # The whole GPU lookup is one opaque op, keyed to this layer; the op body runs the pinned forward_impl.
    assert ple.forward_impl.func is helper._ple_forward and helper._STATE["ple_layers"][ple.forward_impl.args[0]] is ple
    ple.embedding_dim = 16
    fake = helper._arc_ple_lookup_fake(torch.zeros(5, dtype=torch.int32), None, None, ple.forward_impl.args[0])
    assert fake.shape == (5, 16) and fake.dtype == torch.float8_e4m3fn
    monkeypatch.setitem(helper._STATE, "config", {})
    other = Ple()
    helper.ple_uva_place(other)  # option off: untouched
    assert other.ngram_embedding.quant_method.embedding is None


def pinned_ngram_class(monkeypatch, torch):
    """Qwen3_8FlashNextNGramEmbedding's forward path (forward_impl and its two static helpers) taken verbatim from
    the recipe-patched pinned ple_layer.py, in a stand-in module with its own ``is_offload_process``."""
    import sys
    import types

    if not PINNED_PLE.is_file():
        pytest.skip("Optional recipe-patched ple_layer.py (ARC_PINNED_PLE)")
    text = PINNED_PLE.read_text()
    lines = text.splitlines(keepends=True)
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
               and n.name == "Qwen3_8FlashNextNGramEmbedding")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_shift_precompute", "_shift_apply", "forward_impl")]
    body = "".join("".join(lines[(m.decorator_list[0].lineno if m.decorator_list else m.lineno) - 1:m.end_lineno])
                   for m in methods)
    module = types.ModuleType("arc_test_ple_layer")
    module.torch = torch
    module.is_offload_process = lambda: False
    exec(compile("class NGram:\n" + body, "ple_layer.py", "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.NGram.__module__ = module.__name__
    return module.NGram


def fake_ngram(torch, NGram, table):
    ple = NGram()
    ple.eos_token_id, ple.unigram_vocab_size, ple.ngram_size, ple.heads_per_ngram = 999, 1000, 3, 2
    ple.head_dim, ple.embedding_dim = table.shape[1], 4 * table.shape[1]
    ple.layer_multipliers = torch.tensor([7, 11, 13])
    ple.ngram_heads_vocab_sizes = torch.tensor([101, 103, 107, 109])
    ple.ngram_heads_offsets = torch.tensor([0, 101, 204, 311])
    ple.positions_buffer = torch.arange(64)
    ple.padded_buffer = torch.full((5, 64), 999)

    class Lookup:
        weight = table

        def __call__(self, ids):
            return torch.nn.functional.embedding(ids, table.view(torch.uint8)).view(table.dtype)

    ple.ngram_embedding = Lookup()
    return ple


def test_ple_check_compares_the_gpu_path_with_the_offload_path(monkeypatch):
    """The load-time PLE check on stand-in tensors: the pinned forward path in GPU mode (static-width workspace,
    graph padding) against the offload process's mode (the served path until now) on the same table."""
    torch, helper = load_helper(monkeypatch)
    NGram = pinned_ngram_class(monkeypatch, torch)
    monkeypatch.setattr(helper, "_gather_check", lambda *a, **k: {})
    table = torch.randint(0, 120, (420, 4), generator=torch.Generator().manual_seed(3), dtype=torch.uint8).view(
        torch.float8_e4m3fn)
    ple = fake_ngram(torch, NGram, table.clone())
    helper._ple_check(ple, table, ple.ngram_embedding.weight)  # passes: identical bytes on both paths
    ple.ngram_embedding.weight.view(torch.uint8)[:] += 1  # the 'device' table differs from the host one
    with pytest.raises(RuntimeError, match="PLE UVA self-check failed"):
        helper._ple_check(ple, table, ple.ngram_embedding.weight)
