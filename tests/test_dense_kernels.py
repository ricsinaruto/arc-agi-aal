"""Kernel work, phase 2: the dense-kernel overlay (hash guard, plumbing), the kernel quality gate, the
composed arena variant and the gated build. CPU only: the runtime helper needs torch/triton/vLLM and is
checked structurally here; its numerics are checked at load on the GPU (ARC_DENSE_CHECK)."""

import ast
import asyncio
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from test_serving_core import import_fork

from arc_infra import dense_kernels_patch as patch
from arena import kernel_gate as gate
from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
PINNED = Path("/tmp/arc-pinned-runtime/vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py")
SYNTHETIC = ("def enable_qwen38next_low_latency_gemm(\n    module,\n    dtype,\n) -> None:\n" + patch.ANCHOR
             + "    return\n")


def synthetic_site(tmp_path, monkeypatch):
    original = SYNTHETIC.encode()
    monkeypatch.setattr(patch, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    target = tmp_path / "site" / patch.RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    return tmp_path / "site", target, original


# ----------------------------------------------------------------------------- settings and patch
def test_settings_are_validated_and_canonical():
    assert patch.validate(None) is None
    assert patch.validate({"skinny": False, "int8_weights": []}) is None
    assert patch.validate({"int8_weights": ["attn_out", "lm_head"]}) == {
        "skinny": False, "int8_weights": ["lm_head", "attn_out"]}
    assert patch.validate(patch.PHASE2) == {"skinny": True, "int8_weights": list(patch.INT8_FAMILIES)}
    for bad in ({"skinny": 1}, {"int8_weights": ["router"]}, {"int8_weights": ["lm_head", "lm_head"]},
                {"skinny": True, "fp8": True}, ["skinny"], {"int8_weights": "lm_head"}):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_unknown_source_fails_closed():
    with pytest.raises(RuntimeError, match="unrecognized"):
        patch.patched_source(b"def enable_qwen38next_low_latency_gemm(module, dtype):\n    pass\n")


def test_patch_hooks_the_pinned_enable_once_before_its_sm103_gate(tmp_path, monkeypatch):
    site, target, original = synthetic_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "evidence", patch.PHASE2)
    text = target.read_text()
    assert text.count("arc_dense_kernels import enable") == 1
    assert text.index("_arc_dense_enable(module, dtype)") < text.index("not _is_sm103()")
    helper = target.with_name(patch.HELPER)
    assert helper.read_bytes() == (ROOT / "src/arc_infra/dense_kernels_runtime.py").read_bytes()
    assert json.loads(target.with_name(patch.SETTINGS).read_text()) == patch.validate(patch.PHASE2)
    assert identity["settings"] == patch.validate(patch.PHASE2)
    assert json.loads((tmp_path / "evidence" / patch.MARKER).read_text()) == identity
    # Same overlay again (no re-extraction): accepted as is.
    assert patch.apply(site, tmp_path / "evidence", patch.PHASE2) == identity
    # A restart re-extracts the pinned file: the overlay applies again.
    target.write_bytes(original)
    assert patch.apply(site, tmp_path / "evidence", patch.PHASE2)["patched_sha256"] == identity["patched_sha256"]
    # Other settings over an already patched file, or a tampered file: refused.
    with pytest.raises(RuntimeError, match="modified"):
        patch.apply(site, tmp_path / "evidence", {"skinny": True})
    target.write_text(target.read_text() + "\n# tampered")
    with pytest.raises(RuntimeError, match="modified"):
        patch.apply(site, tmp_path / "evidence", patch.PHASE2)
    with pytest.raises(ValueError, match="unpatched"):
        patch.apply(site, tmp_path / "evidence", None)


def test_exact_pinned_source_when_available():
    if not PINNED.is_file():
        pytest.skip("Optional pinned runtime layer 24 source; the hash guard is checked if supplied")
    patched = patch.patched_source(PINNED.read_bytes())
    compile(patched, str(PINNED), "exec")
    tree = ast.parse(patched)
    enable = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "enable_qwen38next_low_latency_gemm")
    first = ast.unparse(enable.body[1])
    assert "_arc_dense_enable(module, dtype)" in first


def test_runtime_helper_families_match_the_patch_settings():
    tree = ast.parse((ROOT / "src/arc_infra/dense_kernels_runtime.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    linear = values["LINEAR_FAMILIES"]
    assert ("lm_head", *linear) == patch.INT8_FAMILIES
    assert linear["gdn_in_proj_qkvz"][1] == (16384, 2560) and linear["attn_out"][1] == (2560, 6144)
    # HC up (K = 320) cannot run on the skinny kernel's K tile; every other TP=1 projection is planned.
    assert (10240, 320) not in values["SKINNY_SHAPES"] and len(values["SKINNY_SHAPES"]) == 13
    assert values["SKINNY_MAX_M"] == 16 and values["TRITON_MAX_M"] >= 64
    assert set(values["TRITON_TILES"]) == {tuple(v[1]) for v in linear.values()} | {values["LM_HEAD_SHAPE"]}


def test_compile_cache_is_cleared(tmp_path):
    cache = tmp_path / "vllm" / "torch_compile_cache" / "torch_aot_compile" / "abc"
    cache.mkdir(parents=True)
    assert patch.clear_compile_cache(tmp_path) == [str(tmp_path / "vllm" / "torch_compile_cache")]
    assert not cache.exists() and patch.clear_compile_cache(tmp_path) == []


# ----------------------------------------------------------------------------- arena serving plumbing
def test_serving_applies_the_overlay_only_when_requested(tmp_path, monkeypatch):
    from arena import serving

    applied = []
    monkeypatch.setattr(serving, "apply", lambda *a: applied.append("layout"))
    monkeypatch.setattr(serving, "apply_dense_kernels",
                        lambda site, working, spec: applied.append(("dense", spec)) or {"settings": spec})
    monkeypatch.setattr(serving, "clear_compile_cache", lambda root: applied.append(("clear", root)) or [])

    def recipe():
        return NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ["serve", "--flag"],
                  RUNTIME_ROOT=tmp_path / "runtime", CACHE_ROOT=tmp_path / "cache", MAX_CUDAGRAPH_CAPTURE_SIZE_ENV="X")

    control = recipe()
    receipt = serving.configure(control, tmp_path)
    control.verify_and_extract_runtime()
    assert applied == ["layout"] and "dense_kernels" not in receipt
    assert control.server_command() == ["serve", "--flag"]
    assert receipt == serving.configure(recipe(), tmp_path, dense_kernels=None)
    assert receipt == serving.configure(recipe(), tmp_path, dense_kernels={"skinny": False, "int8_weights": []})
    applied.clear()
    patched = recipe()
    receipt2 = serving.configure(patched, tmp_path, dense_kernels=patch.PHASE2)
    patched.verify_and_extract_runtime()
    assert applied == ["layout", ("clear", tmp_path / "cache"), ("dense", patch.validate(patch.PHASE2))]
    assert receipt2 == {**receipt, "dense_kernels": patch.validate(patch.PHASE2)}
    assert patched.server_command() == ["serve", "--flag"]  # the server command itself is unchanged
    assert json.loads((tmp_path / "dense-kernels-start.json").read_text())["identity"]["settings"]
    with pytest.raises(ValueError):
        serving.configure(recipe(), tmp_path, dense_kernels={"int8_weights": ["router"]})


def test_real_fork_recipe_gets_the_overlay_after_extraction(tmp_path, monkeypatch):
    from arena import serving

    recipe = import_fork(tmp_path, monkeypatch)
    site, target, _ = synthetic_site(tmp_path, monkeypatch)
    monkeypatch.setattr(recipe, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(recipe, "CACHE_ROOT", tmp_path / "cache")
    (tmp_path / "runtime/usr/local/lib/python3.12").mkdir(parents=True)
    (tmp_path / "runtime/usr/local/lib/python3.12/dist-packages").symlink_to(site, target_is_directory=True)
    monkeypatch.setattr(recipe, "verify_and_extract_runtime", lambda *a, **k: {"extracted": True})
    monkeypatch.setattr(serving, "apply", lambda *a: None)
    serving.configure(recipe, tmp_path / "work", dense_kernels=patch.PHASE2)
    assert recipe.verify_and_extract_runtime() == {"extracted": True}
    assert target.read_text().count("_arc_dense_enable") == 2  # import alias + call
    assert (tmp_path / "work" / patch.MARKER).is_file()


def test_lab_override_and_study_key_reach_configure(tmp_path, monkeypatch):
    from arena import serving

    assert serving.lab_overrides({"ARENA_DENSE_KERNELS": "off"}) == {"dense_kernels": None}
    assert serving.lab_overrides({"ARENA_DENSE_KERNELS": json.dumps({"skinny": True})}) == {
        "dense_kernels": {"skinny": True, "int8_weights": []}}
    with pytest.raises(ValueError):
        serving.lab_overrides({"ARENA_DENSE_KERNELS": json.dumps({"int8_weights": ["router"]})})
    config = {"generation_slots": 25, "kv_margin": .9, "server": {"mtp_tokens": 0, "max_num_batched_tokens": 4096},
              "cache_experiment": {"cpu_cache_gib": 0, "kv_cache_dtype": "auto", "reclaim_skipped_blocks": True}}
    calls, hooks = [], []
    monkeypatch.setattr(serving, "start", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(serving, "configure", lambda recipe, working, **kwargs: hooks.append(kwargs))
    monkeypatch.setattr(serving.BlockBudget, "load", lambda *args: NS(layout={}))
    study = tmp_path / "study.json"

    def run(server, env=None):
        study.write_text(json.dumps({**config, "server": {**config["server"], **server}}))
        monkeypatch.delenv("ARENA_DENSE_KERNELS", raising=False)
        if env is not None:
            monkeypatch.setenv("ARENA_DENSE_KERNELS", env)
        monkeypatch.setattr("sys.argv", ["arena.serving", "--working", str(tmp_path), "--input-root", str(tmp_path),
                                         "--study", str(study)])
        serving.main()
        calls[-1]["recipe_hook"](None, tmp_path)
        return hooks[-1]["dense_kernels"]

    assert run({}) is None  # the control study: no key, no overlay
    assert run({"dense_kernels": patch.PHASE2}) == patch.PHASE2
    assert run({"dense_kernels": patch.PHASE2}, env="off") is None
    assert run({}, env=json.dumps({"skinny": True})) == {"skinny": True, "int8_weights": []}


# ----------------------------------------------------------------------------- variant and build
def test_kernel_variant_differs_from_mtp0_wide_only_in_the_kernels():
    wide = apply_variant(CONTROL, "fixed-2x-mtp0-wide")
    kernels = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kernels")
    assert kernels["server"]["dense_kernels"] == patch.PHASE2
    a, b = json.loads(json.dumps(kernels)), json.loads(json.dumps(wide))
    a["server"].pop("dense_kernels")
    a.pop("arena_variant"), b.pop("arena_variant")
    assert a == b
    difference = kernels["arena_variant"]["reference_difference"]
    assert difference["server.dense_kernels"] == {"before": None, "after": patch.PHASE2}
    assert kernels["arena_variant"]["checkpoint_and_serving_unchanged"] is False
    # Every other composed arm, and the control, carries no dense kernels.
    assert "dense_kernels" not in wide["server"] and "dense_kernels" not in CONTROL["server"]


def load_builder():
    spec = importlib.util.spec_from_file_location("kernel_builder", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


C110 = dict(paired_30m=True, opening_burst_requests=3, exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"],
            wall_seconds=3600, clone_instances=110)


def test_gated_build_prepends_the_gate_and_leaves_the_arena_study_unchanged(tmp_path):
    builder = load_builder()
    _, wide = builder.build("c110-mtp0-wide", output=tmp_path / "wide", arena_variant="fixed-2x-mtp0-wide", **C110)
    output, study = builder.build("c110-kernels", output=tmp_path / "k", arena_variant="fixed-2x-mtp0-wide-kernels",
                                  kernel_gate=True, **C110)
    assert json.loads((output / "configs/comparison.json").read_text()) == study
    assert [(t["name"], t["kind"]) for t in study["trials"]] == [
        ("kernel-ref", "kernel-quality"), ("kernel-gate", "kernel-quality"), ("duck-transfer", "duck-transfer")]
    ref, check, arena = study["trials"]
    assert ref["server_env"] == {"ARENA_DENSE_KERNELS": "off", "ARENA_TORCH_PROFILER": "1"}
    assert check["server_env"] == arena["server_env"] == {"ARENA_KV_CACHE_MEMORY_BYTES": "15226675200",
                                                           "ARENA_TORCH_PROFILER": "1"}
    assert check["reference"] == "kernel-ref" and arena["requires"] == "kernel-gate"
    assert ref["competition"] is False and check["competition"] is False and "competition" not in arena
    for name in ("prompts.json", "gsm8k.jsonl"):
        assert (output / "configs/kernel-quality" / name).read_bytes() == (ROOT / "data/kernel_quality" / name).read_bytes()
    assert study["kernel_gate"]["thresholds"] == gate.THRESHOLDS
    assert study["notebook_seconds"] == wide["notebook_seconds"] + 2700
    # Apart from the gate, the arena study is the recorded mtp0-wide study plus the kernel key.
    a, b = json.loads(json.dumps(study)), json.loads(json.dumps(wide))
    for key in ("trials", "kernel_gate", "notebook_seconds", "kernel_timeout", "name", "arena_variant"):
        a.pop(key), b.pop(key, None)
    assert a.pop("server") == {**b.pop("server"), "dense_kernels": patch.PHASE2}
    assert a == b
    assert wide["trials"] == [{"name": "duck-transfer", "kind": "duck-transfer"}]
    with pytest.raises(ValueError, match="dense_kernels"):
        builder.build("bad", output=tmp_path / "bad", arena_variant="fixed-2x-mtp0-wide", kernel_gate=True, **C110)


def test_modal_test_for_the_gated_arena():
    source = (ROOT / "scripts/modal_validation.py").read_text()
    tree = ast.parse(source)
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    assert ns["validation_timeout"]("arena", "duck-110-kernels") == 7800
    assert ns["arena_arguments"]("duck-110-kernels") == []
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (7800 + 60) * rate * 1.15 < 13.0  # the user's budget for the whole job


def test_bootstrap_skips_trials_whose_gate_did_not_pass():
    text = (ROOT / "scripts/arena/bootstrap.py").read_text()
    guard = text.index('if trial.get("requires") and results.get(trial["requires"]) != "complete":')
    assert guard < text.index("if server_env(trial) != current_env:")


# ----------------------------------------------------------------------------- gate metrics
def lp(probs):
    return {t: math.log(p) for t, p in probs.items()}


def test_coarse_kl_is_exact_on_the_partition_and_zero_for_identical():
    p = lp({1: .5, 2: .3, 3: .1})
    assert gate.kl_coarse(p, p) == 0.0
    q = lp({1: .4, 2: .4, 3: .1})
    expected = .5 * math.log(.5 / .4) + .3 * math.log(.3 / .4) + .1 * math.log(1) + .1 * math.log(.1 / .1)
    assert gate.kl_coarse(p, q) == pytest.approx(expected)
    # A token missing from one list moves into the rest bucket for both.
    r = lp({1: .5, 2: .3})
    assert gate.kl_coarse(p, r) == pytest.approx(0.0, abs=1e-12)
    assert gate.kl_coarse({5: 0.0}, {6: 0.0}) is None


def test_answer_extraction():
    assert gate.extract_answer("so \\boxed{1,234}") == 1234
    assert gate.extract_answer("The total is \\boxed{\}.") == 18.5
    assert gate.extract_answer("", "thinking ... \\boxed{\\frac{1}{2}} wait \\boxed{7}") == 7
    assert gate.extract_answer("She makes 18 dollars") == 18
    assert gate.extract_answer("no number") is None
    assert gate.gold_answer("... #### 1,200") == 1200


def fake_server(scale):
    """An OpenAI-compatible stand-in: distributions shifted by ``scale`` (0 = the reference)."""
    def dist(position):
        base = {10 + position % 3: .6 - scale, 20: .3 + scale, 30: .1}  # normalized over the listed tokens
        return {f"token_id:{t}": math.log(p) for t, p in base.items()}

    def handler(request):
        body = json.loads(request.content or b"{}")
        path = request.url.path
        if path.endswith("/tokenize"):
            text = body.get("prompt") or json.dumps(body.get("messages"))
            return httpx.Response(200, json={"tokens": list(range(len(text) % 7 + 5)), "count": 5})
        if path.endswith("/chat/completions"):
            wrong = scale > .2 and body["seed"] % 2 == 0
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "stop", "message": {
                    "content": "\\boxed{" + str(-1 if wrong else body["seed"]) + "}", "reasoning_content": "..."}}],
                "usage": {"completion_tokens": 50}})
        if path.endswith("/completions") and "prompt_logprobs" in body:
            ids = body["prompt"]
            entries = [None] + [{str(t.split(":")[1]): {"logprob": v, "rank": 1} for t, v in dist(i).items()}
                                for i in range(1, len(ids))]
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": entries}]})
        if path.endswith("/completions"):
            tops = [dist(i) for i in range(8)]
            tokens = [max(t, key=t.get) for t in tops]
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "logprobs": {
                "tokens": tokens, "top_logprobs": tops}}]})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_reference_then_gate_end_to_end(tmp_path, monkeypatch):
    source = tmp_path / "bundle"
    (source / gate.INPUTS).mkdir(parents=True)
    prompts = {"greedy": [{"index": i, "system": "s", "user": f"u{i}"} for i in range(3)],
               "teacher_forced": [{"index": i, "text": f"passage {i}"} for i in range(4)]}
    (source / gate.INPUTS / "prompts.json").write_text(json.dumps(prompts))
    (source / gate.INPUTS / "gsm8k.jsonl").write_text("".join(
        json.dumps({"index": i, "question": "q", "answer": f"#### {i}"}) + "\n" for i in range(200)))

    async def no_isolation(url):
        return {}

    monkeypatch.setattr(gate, "isolate_trial", no_isolation)
    study = {"server": {"mtp_tokens": 0}, "reasoning_effort": "xhigh"}
    trials = tmp_path / "trials"

    def run(trial, scale):
        return asyncio.run(gate.run_quality(study, trial, source=source, output=trials / trial["name"],
                                            server_dir=tmp_path, base_url="http://x/v1",
                                            deadline_epoch=1e12, log=lambda *a, **k: None,
                                            transport=fake_server(scale)))

    ref = run({"name": "kernel-ref", "role": "reference"}, 0.0)
    assert ref["status"] == "complete" and ref["gsm8k"]["accuracy"] == 1.0
    assert len(ref["teacher_forced"]["rows"][0]["positions"]) == len(ref["teacher_forced"]["rows"][0]["tokens"]) - 1
    same = run({"name": "kernel-gate", "role": "gate", "reference": "kernel-ref"}, 0.0)
    assert same["status"] == "complete" and same["verdict"]["passed"]
    assert same["verdict"]["teacher_forced"]["mean_kl"] == 0.0 and same["verdict"]["greedy"]["exact_match"] == 1.0
    assert same["verdict"]["control_noise"]["teacher_forced"]["top1_agreement"] == 1.0
    assert json.loads((trials / "kernel-gate" / "kernel-gate.json").read_text())["passed"] is True
    # A large distribution shift fails the distribution check; GSM8K is then not run at all.
    bad = run({"name": "kernel-gate-bad", "role": "gate", "reference": "kernel-ref"}, 0.5)
    assert bad["status"] == "gate_failed" and not bad["verdict"]["checks"]["tf_mean_kl"]
    assert "gsm8k" not in bad and bad["verdict"]["checks"]["gsm8k"] is False


def test_gsm8k_drop_fails_the_gate():
    rows = lambda correct: {"rows": [{"index": i, "correct": c} for i, c in enumerate(correct)]}  # noqa: E731
    same_tf = {"rows": [{"index": 0, "tokens": [1, 2], "positions": [[[5, -0.1], [6, -2.5]]]}]}
    reference = {"teacher_forced": same_tf, "gsm8k": rows([True] * 200)}
    ok = gate.verdict(reference, {"teacher_forced": same_tf, "gsm8k": rows([True] * 196 + [False] * 4)})
    assert ok["passed"] and ok["gsm8k"]["delta_points"] == pytest.approx(-2.0)
    bad = gate.verdict(reference, {"teacher_forced": same_tf, "gsm8k": rows([True] * 195 + [False] * 5)})
    assert not bad["passed"] and bad["checks"] == {"tf_mean_kl": True, "tf_top1_agreement": True, "gsm8k": False}
    few = gate.verdict(reference, {"teacher_forced": same_tf, "gsm8k": rows([True] * 100)})
    assert not few["checks"]["gsm8k"]
