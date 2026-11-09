"""prefix-unit-int4 (docs/validation-20260927/prefix-unit-int4.md): the warm prompt-logprobs patch, the teacher-forced
Latin square against a fake engine (a correct partial state passes, the deletion control is detected), the verdict
logic, the bundle and its Modal bound. CPU only."""

import ast
import asyncio
import importlib.util
import json
import os
import random
from pathlib import Path

import httpx
import numpy as np
import pytest

from arena import partial_tf as ptf
from arena import prompt_logprobs_patch as plp
from arena import serving
from arena import throughput_bench as tb

ROOT = Path(__file__).resolve().parents[1]
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))


def test_warm_prompt_logprobs_patch_text():
    sampling = ("x = 1\n" + plp.SAMPLING_ANCHOR + "            y = 2\n").encode()
    with pytest.raises(RuntimeError, match="unrecognized"):
        plp.patched(sampling, plp.SAMPLING_SHA256, plp.SAMPLING_ANCHOR, plp.SAMPLING_REPLACEMENT, "s.py")
    # On the real pinned files (optional): the edits apply once, compile and are idempotent.
    if not (PINNED_SITE / plp.SAMPLING_RELATIVE).is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")


def test_warm_prompt_logprobs_patch_on_pinned_files(tmp_path):
    if not (PINNED_SITE / plp.SAMPLING_RELATIVE).is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    site = tmp_path / "site"
    for rel in (plp.SAMPLING_RELATIVE, plp.OUTPUTS_RELATIVE):
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_bytes((PINNED_SITE / rel).read_bytes())
    first = plp.apply(site, tmp_path)
    assert plp.apply(site, tmp_path) == first
    text = (site / plp.SAMPLING_RELATIVE).read_text()
    assert 'extra_args.get("arc_read_prefix_cache")' in text and "torch.zeros(" in (site / plp.OUTPUTS_RELATIVE).read_text()


def test_lab_override_and_limits():
    assert serving.lab_overrides({"ARENA_WARM_PROMPT_LOGPROBS": "1"}) == {"warm_prompt_logprobs": True}
    with pytest.raises(ValueError):
        serving.lab_overrides({"ARENA_WARM_PROMPT_LOGPROBS": "yes"})
    spec, block, unit, b, offsets, end = ptf.layout({})
    assert (block, unit, b, offsets, end) == (3136, 16, 9408, [768, 1536, 2304], 13568)
    with pytest.raises(ValueError):
        ptf.layout({"offsets": [100, 200, 300]})  # not multiples of 64
    arms = ptf.arm_requests(list(range(14000)), {})
    assert arms["P2"][0] == list(range(9408 + 1536 + 8)) and arms["P2"][2] == 9408 + 1536
    assert arms["F"][2] == 9408 and arms["S"][1][9408] == 9408 + 768 and arms["S"][3] == 768
    assert len(ptf.ARMS) == 7


class FakeEngine:
    """Completions with a fine-grained prefix cache; a position's next-token distribution depends on its whole
    prefix (so a deletion changes it) plus per-request noise (the batch nondeterminism)."""

    def __init__(self, block, unit, broken=False):
        self.block, self.unit, self.broken = block, unit, broken
        self.cache, self.rng = set(), random.Random(0)

    def dists(self, ids, noise):
        out, total = [None], 0
        for i in range(1, len(ids)):
            total = (total + ids[i - 1] * 31 + 7) % 9973
            base = {str((total + k * 13) % 1000): -0.4 - 0.9 * k for k in range(6)}
            out.append({t: {"logprob": lp + noise * self.rng.gauss(0, 1)} for t, lp in base.items()})
        return out

    def hit(self, ids):
        best = 0
        for c in range(1, len(ids)):
            if tuple(ids[:c]) in self.cache:
                best = c
        return best

    def register(self, ids):
        for c in range(self.block, len(ids) + 1, self.block):
            self.cache.add(tuple(ids[:c]))
        self.cache.add(tuple(ids[:len(ids) // self.unit * self.unit]))

    async def handler(self, request):
        path = request.url.path
        if path.endswith("/metrics"):
            return httpx.Response(200, text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n")
        if path.endswith("/reset_prefix_cache"):
            self.cache.clear()
            return httpx.Response(200, json={"success": True})
        if path.endswith("/completions"):
            body = json.loads(request.content)
            ids = body["prompt"]
            warm = bool((body.get("vllm_xargs") or {}).get("arc_read_prefix_cache"))
            cached = self.hit(ids) if (warm or not body.get("prompt_logprobs")) else 0
            self.register(ids)
            await asyncio.sleep(0)
            usage = {"prompt_tokens": len(ids), "prompt_tokens_details": {"cached_tokens": cached}}
            if not body.get("prompt_logprobs"):
                return httpx.Response(200, json={"choices": [{"text": "x"}], "usage": usage})
            entries = self.dists(ids, 0.05)
            if self.broken and warm and cached % self.block:
                # a wrong partial state: the positions after the hit see a prefix missing its last 64 tokens
                wrong = self.dists(ids[:cached - 64] + ids[cached:], 0.05)
                entries = entries[:cached + 1] + wrong[cached + 1 - 64:]
            if warm and cached:
                # the served list of a warm request: [None] + the computed positions (cached + 1 ...)
                entries = [None] + entries[cached + 1:]
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": entries}], "usage": usage})
        raise AssertionError(path)


def run_square(tmp_path, broken):
    spec = {"block": 256, "unit": 16, "block_index": 1, "offsets": [64, 128, 192], "tail_tokens": 64,
            "top_logprobs": 5, "lanes": 4, "bootstrap": 400, "near_positions": 32}
    engine = FakeEngine(256, 16, broken=broken)
    transport = httpx.MockTransport(engine.handler)
    rng = random.Random(1)
    sequences = [[rng.randrange(1, 5000) for _ in range(600)] for _ in range(8)]

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            real = httpx.AsyncClient

            def patched(*args, **kwargs):
                kwargs.setdefault("transport", transport)
                return real(*args, **kwargs)

            httpx.AsyncClient = patched
            try:
                return await ptf.partial_tf(client, "http://model/v1", model="m", sequences=sequences, spec=spec,
                                            output=tmp_path, log=lambda *a, **k: None)
            finally:
                httpx.AsyncClient = real

    return asyncio.run(go())


def test_latin_square_passes_a_correct_partial_state(tmp_path):
    result = run_square(tmp_path, broken=False)
    assert not result["errors"] and result["contexts"] == 8
    assert all(h["as_expected"] == h["requests"] == 8 for h in result["hits"].values())
    summary = result["summary"]
    assert summary["verdict"]["negative_control_detected"] and summary["verdict"]["hits_exact"]
    assert summary["R_S"]["lo5"] > 1.25 and summary["R_P"]["hi95"] <= 1.25
    assert summary["verdict"]["result"] == "pass"
    assert len(list((tmp_path / "partial-tf").glob("context-*.npz"))) == 8


def test_latin_square_fails_a_wrong_partial_state(tmp_path):
    result = run_square(tmp_path, broken=True)
    summary = result["summary"]
    assert summary["R_P"]["lo5"] > 1.25 and summary["verdict"]["result"] == "fail"
    assert summary["R_F"]["hi95"] <= 1.25  # the full-block arm is unaffected


def test_bundle_and_modal_bound(tmp_path):
    spec = importlib.util.spec_from_file_location("build_arena_pui4", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    hero = {"hero_v2": {"requested": {}}, "cache_experiment": {"prefix_cache": {"mamba_align_fix": True}},
            "server": {"max_num_seqs": 40, "max_cudagraph_capture_size": 40, "prefix_match_unit": 16,
                       "prompt_tokens_details": True, "serving_kernels": {"kv_cache_4bit": "int4"}},
            "generation_slots": 40, "active_limit": 38}
    transcript = tmp_path / "t.txt"
    transcript.write_text("line\n" * 100)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"teacher_forced": [{"ids": list(range(17500)), "source": f"s{i}", "family": "x"}
                                                      for i in range(24)]}))
    (tmp_path / "out").mkdir()
    study = builder.prefix_unit_int4_study(hero, tmp_path / "out", str(transcript), str(prompts))
    assert study["server"]["max_num_seqs"] == 48 == study["generation_slots"]
    assert study["server"]["max_cudagraph_capture_size"] == 48
    (trial,) = study["trials"]
    assert serving.lab_overrides(trial["server_env"]) == {"warm_prompt_logprobs": True}
    seqs = json.loads((tmp_path / "out" / study["throughput_bench"]["partial_tf"]["file"]).read_text())["sequences"]
    assert len(seqs) == 24 and all(len(s) == 10432 for s in seqs)
    assert tb.grid_points(study["throughput_bench"]) == []
    with pytest.raises(ValueError, match="int4"):
        builder.prefix_unit_int4_study({**hero, "server": {**hero["server"], "prefix_match_unit": None}},
                                       tmp_path / "out", str(transcript), str(prompts))
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "prefix-unit-int4")
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 3.0
    assert ns["arena_arguments"]("prefix-unit-int4") == ["--trial", "prefix-int4"]
    assert study["notebook_seconds"] <= timeout - 60
    from arena.timeline import check

    plan = check(study, setup_seconds=650, restart_seconds=614, run_seconds={"prefix-int4": 590})
    assert plan[0]["timeout"] >= 590 * 1.15


def test_summary_verdict_logic():
    def ctx(scale_p, scale_s):
        rng = np.random.default_rng(0)
        rec = {}
        for off in ("P1", "P2", "P3"):
            c2 = rng.exponential(0.1, 200)
            rec[off] = {"C2": c2, "P": c2 * scale_p, "F": c2, "S": c2 * scale_s, "PF": c2,
                        "top1_P": np.ones(200), "top1_C2": np.ones(200)}
        return rec

    ok = ptf.summarize({i: ctx(1.0, 3.0) for i in range(20)}, {"bootstrap": 200})
    assert ok["verdict"]["result"] == "pass"
    bad = ptf.summarize({i: ctx(2.0, 3.0) for i in range(20)}, {"bootstrap": 200})
    assert bad["verdict"]["result"] == "fail"
    weak = ptf.summarize({i: ctx(1.0, 1.0) for i in range(20)}, {"bootstrap": 200})
    assert weak["verdict"]["result"].startswith("inconclusive")


def test_tf_only_bundle_and_bound(tmp_path):
    spec = importlib.util.spec_from_file_location("build_arena_pui4b", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    hero = {"hero_v2": {"requested": {}}, "cache_experiment": {"prefix_cache": {"mamba_align_fix": True}},
            "server": {"max_num_seqs": 40, "max_cudagraph_capture_size": 40, "prefix_match_unit": 16,
                       "prompt_tokens_details": True, "serving_kernels": {"kv_cache_4bit": "int4"}},
            "generation_slots": 40, "active_limit": 38}
    transcript = tmp_path / "t.txt"
    transcript.write_text("line\n" * 100)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"teacher_forced": [{"ids": list(range(17500)), "source": f"s{i}", "family": "x"}
                                                      for i in range(24)]}))
    (tmp_path / "out").mkdir()
    study = builder.prefix_unit_int4_study(hero, tmp_path / "out", str(transcript), str(prompts), tf_only=True)
    bench = study["throughput_bench"]
    assert bench.get("partial_tf") and "stream_sweep" not in bench and "prefix_replay" not in bench
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "prefix-unit-int4-tf")
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 2.4 and study["notebook_seconds"] <= timeout - 60
    from arena.timeline import check

    # v1 measured: setup 594 s, the square's seven phases ~21 s each
    plan = check(study, setup_seconds=650, restart_seconds=614, run_seconds={"prefix-int4": 250})
    assert plan[0]["timeout"] >= 250 * 1.15
