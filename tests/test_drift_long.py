"""Long-context drift test (29 September): the kernel-quality options without greedy/GSM8K passes, the study build
(kernels-v2 reference, hero and int8-KV-only gates) and the Modal test's budget. CPU only."""

import ast
import asyncio
import importlib.util
import json
from pathlib import Path

from test_dense_kernels import fake_server

from arena import kernel_gate as gate
from arena.serving import lab_overrides

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_arena_dl", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modal_functions(*names):
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in names:
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    return ns


def test_teacher_forced_only_reference_and_gate(tmp_path, monkeypatch):
    source = tmp_path / "bundle"
    (source / gate.INPUTS).mkdir(parents=True)
    prompts = {"greedy": [], "teacher_forced": [{"index": i, "text": None, "ids": list(range(1, 40))}
                                                for i in range(3)]}
    (source / gate.INPUTS / "prompts.json").write_text(json.dumps(prompts))
    (source / gate.INPUTS / "gsm8k.jsonl").write_text(json.dumps({"index": 0, "question": "q", "answer": "#### 1"}) + "\n")

    async def no_isolation(url):
        return {}

    monkeypatch.setattr(gate, "isolate_trial", no_isolation)
    study = {"server": {"mtp_tokens": 0}, "reasoning_effort": "xhigh"}
    quality = {"greedy": False, "gsm8k": False, "report_only": True}

    def run(trial, scale):
        return asyncio.run(gate.run_quality(study, {**trial, "quality": quality}, source=source,
                                            output=tmp_path / "trials" / trial["name"], server_dir=tmp_path,
                                            base_url="http://x/v1", deadline_epoch=1e12, log=lambda *a, **k: None,
                                            transport=fake_server(scale)))

    ref = run({"name": "dl-ref", "role": "reference"}, 0.0)
    assert ref["status"] == "complete" and not ref["errors"]
    assert {"greedy", "greedy_repeat", "gsm8k", "gsm8k_repeat"}.isdisjoint(ref)
    assert len(ref["teacher_forced_repeat"]["rows"]) == 3
    shifted = run({"name": "dl-hero", "role": "gate", "reference": "dl-ref"}, 0.1)
    assert shifted["status"] == "complete" and "gsm8k" not in shifted and "greedy" not in shifted
    assert shifted["verdict"]["teacher_forced"]["positions"] == 3 * 38 and shifted["verdict"]["teacher_forced"]["mean_kl"] > 0
    assert shifted["verdict"]["control_noise"]["teacher_forced"]["mean_kl"] == 0.0
    spec = importlib.util.spec_from_file_location("drift_long_analyze", ROOT / "scripts/drift_long_analyze.py")
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    table = analyze.compare(analyze.rows_of(ref), analyze.rows_of(shifted), {0: [[5, 20]]})
    assert table["0K-8K/all"]["positions"] == 3 * 38 and table["0K-8K/assistant"]["positions"] == 15
    assert table["0K-8K/all"]["mean_kl"] > 0 and table["8K-16K/all"]["positions"] == 0
    noise = analyze.compare(analyze.rows_of(ref), analyze.rows_of(ref, "teacher_forced_repeat"), {})
    assert noise["0K-8K/all"]["mean_kl"] == 0.0 and noise["0K-8K/all"]["delta_logprob_mean"] == 0.0


def test_drift_study_build_and_budget(tmp_path):
    builder = load_builder()
    inputs = tmp_path / "inputs"
    (inputs / "quality").mkdir(parents=True)
    (inputs / "quality/prompts.json").write_text(json.dumps(
        {"teacher_forced": [{"index": 0, "text": None, "ids": [1, 2, 3], "assistant_spans": []}], "greedy": []}))
    output, study = builder.build("dl", output=tmp_path / "dl", paired_30m=True, drift_long=inputs)
    names = [t["name"] for t in study["trials"]]
    assert names == ["dl-ref", "dl-hero", "dl-kv8"] and study["server"]["mtp_tokens"] == 0
    envs = {t["name"]: t["server_env"] for t in study["trials"]}
    assert len({e["ARENA_KV_CACHE_MEMORY_BYTES"] for e in envs.values()}) == 1
    parsed = {name: lab_overrides(env) for name, env in envs.items()}  # every override is valid for the server
    assert parsed["dl-ref"]["serving_kernels"] is None and parsed["dl-ref"]["stack_kernels"] is None
    assert parsed["dl-hero"]["serving_kernels"]["moe_backend"] == "marlin"
    assert parsed["dl-hero"]["serving_kernels"]["kv_cache_8bit"] == "int8" and parsed["dl-hero"]["stack_kernels"]
    assert "mid_m" in parsed["dl-hero"]["dense_kernels"] and "mid_m" not in parsed["dl-ref"]["dense_kernels"]
    assert parsed["dl-kv8"]["serving_kernels"]["kv_cache_8bit"] == "int8"
    assert parsed["dl-kv8"]["serving_kernels"]["moe_backend"] is None
    assert parsed["dl-kv8"]["dense_kernels"] == parsed["dl-ref"]["dense_kernels"]
    assert all(t["quality"] == {"greedy": False, "gsm8k": False, "report_only": True} for t in study["trials"])
    assert (output / "configs/kernel-quality/prompts.json").is_file()
    ns = modal_functions("validation_timeout", "arena_arguments")
    timeout = ns["validation_timeout"]("arena", "drift-long")
    assert study["notebook_seconds"] <= timeout - 60 and ns["arena_arguments"]("drift-long") == []
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 6.0  # the user's bound
