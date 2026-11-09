"""hero-v2 pre-check (docs/validation-20260927/hero-v2-precheck.md): the stream sweep and the partial-hit check against
a fake engine, the --prefix-match-unit lab override, the unit-aware replay accounting, the bundle and its Modal bound.
CPU only."""

import ast
import asyncio
import importlib.util
import json
import math
import time
from pathlib import Path

import httpx
import pytest
from test_arena_cache import layout

from arena import hero_v2_precheck as pre
from arena import prefix_replay, serving
from arena import throughput_bench as tb

ROOT = Path(__file__).resolve().parents[1]


def test_spread_order_and_window_stats():
    order = pre.spread_order(32)
    assert sorted(order) == list(range(32)) and order[:4] == [0, 16, 8, 24]
    lengths = list(range(32))
    for n in (16, 23, 26, 29):
        subset = [lengths[i] for i in order[:n]]
        assert abs(sum(subset) / n - 15.5) < 2.5  # every prefix keeps the distribution's middle
    with pytest.raises(ValueError):
        pre.spread_order(30)
    first = (100.0, {pre.COUNTERS["steps"]: 1000, pre.COUNTERS["generated"]: 20000, pre.COUNTERS["uncached"]: 5})
    last = (120.0, {pre.COUNTERS["steps"]: 1800, pre.COUNTERS["generated"]: 20000 + 800 * 25,
                    pre.COUNTERS["uncached"]: 5 + 24000})
    stats = pre.window_stats(first, last, 25)
    assert stats["step_ms"] == 25.0 and stats["generated_per_step"] == 25.0 and stats["generated_tok_s"] == 1000.0
    assert stats["per_stream_tok_s"] == 40.0 and stats["uncached_tok_s"] == 1200.0 and stats["uncached_per_step"] == 30
    assert not pre.window_stats(first, first, 25)["valid"]
    fit = pre.linear_fit([16, 23, 26, 29, 32], [15 + 0.5 * n for n in (16, 23, 26, 29, 32)])
    assert fit["intercept_ms"] == 15.0 and fit["ms_per_stream"] == 0.5 and fit["residual_rms_ms"] == 0.0


def test_partial_conditions_on_the_hero_layout():
    conditions = pre.conditions(1600, 16)
    assert conditions == {"W16": (10408, 10400, False), "W16b": (10008, 10000, False),
                          "W1600a": (9615, 9600, False), "W1600b": (8008, 8000, True)}
    # Each primer's last 16-token boundary is the state the condition resumes from.
    for primer, expected, _ in conditions.values():
        assert primer // 16 * 16 == expected
    lengths = pre.probe_lengths(1600, 24)
    assert len(lengths) == 24 and lengths == sorted(lengths, reverse=True)
    assert min(lengths) > 10408 and max(lengths) < 11200  # inside block 6, above every primer
    assert pre.top_dist({"token_id:5": -0.1}) == {5: -0.1}
    assert pre.top_dist({"7": {"logprob": -1.5, "rank": 1}}) == {7: -1.5}


def test_prefix_match_unit_lab_override():
    assert serving.lab_overrides({"ARENA_PREFIX_MATCH_UNIT": "16"}) == {"prefix_match_unit": 16}
    assert serving.lab_overrides({}) == {}
    for bad in ("0", "10", "24", "1600"):
        with pytest.raises(ValueError):
            serving.lab_overrides({"ARENA_PREFIX_MATCH_UNIT": bad})
    # Every allowed unit divides the hero's 1,600-token blocks and the QSA compressor's 4-token group divides it.
    assert all(1600 % u == 0 and u % 4 == 0 for u in serving.PREFIX_MATCH_UNITS)


def test_replay_accounting_with_a_unit():
    block, unit = 1600, 16
    previous = {"prompt_token_ids": list(range(10408)), "token_ids": list(range(10408, 10808))}
    ids = previous["prompt_token_ids"] + previous["token_ids"] + [1] * 500
    record = prefix_replay.classify({"prompt_token_ids": ids, "cached_tokens": 10400}, previous, block, unit)
    assert record["ideal_hit"] == 9600 and record["ideal_hit_unit"] == 10400
    # Without a unit the record is the block accounting alone.
    assert "ideal_hit_unit" not in prefix_replay.classify({"prompt_token_ids": ids}, previous, block)
    # A reply that crosses the block boundary: the block ideal wins.
    long = {"prompt_token_ids": list(range(10408)), "token_ids": list(range(10408, 11300))}
    ids2 = long["prompt_token_ids"] + long["token_ids"] + [1] * 10
    assert prefix_replay.classify({"prompt_token_ids": ids2}, long, block, unit)["ideal_hit_unit"] == 11200
    summary = prefix_replay.summarize_replay([{**record, "prompt_tokens": len(ids), "game": 0, "request": 1}],
                                             block, unit)
    hits = summary["unit_hits"]
    assert hits["extra_over_block_ideal_tokens"] == 800 and hits["at_unit_ideal"] == 1
    assert hits["hits_inside_a_block"] == 1 and hits["hits_off_unit_grid"] == 0


class FakeEngine:
    """A vLLM-like engine: forced chat requests decode until cancelled (engine counters advance with time while
    they run); completions answer probes with a deterministic distribution and a fine-grained prefix cache."""

    def __init__(self, block, unit, step_seconds=0.004):
        self.block, self.unit, self.step_seconds = block, unit, step_seconds
        self.running = 0
        self.counters = {"steps": 0.0, "generated": 0.0, "uncached": 0.0}
        self.last = time.monotonic()
        self.cache = set()
        self.cancelled = 0

    def advance(self):
        now = time.monotonic()
        if self.running:
            steps = (now - self.last) / self.step_seconds
            self.counters["steps"] += steps
            self.counters["generated"] += steps * self.running
        self.last = now

    @staticmethod
    def dist(position):
        return {str((position * 7 + k) % 97): -0.5 - k for k in range(5)}

    def hit(self, ids):
        best = 0
        for c in range(1, len(ids)):
            if tuple(ids[:c]) in self.cache:
                best = c
        return best

    def register(self, ids):
        n = len(ids)
        for c in range(self.block, n + 1, self.block):
            self.cache.add(tuple(ids[:c]))
        self.cache.add(tuple(ids[:n // self.unit * self.unit]))

    async def handler(self, request):
        path = request.url.path
        if path.endswith("/tokenize"):
            body = json.loads(request.content)
            return httpx.Response(200, json={"count": max(1, len(json.dumps(body["messages"])) // 6)})
        if path.endswith("/metrics"):
            self.advance()
            c = self.counters
            return httpx.Response(200, text=(
                f"vllm:iteration_tokens_total_count {c['steps']}\nvllm:iteration_tokens_total_sum {c['generated']}\n"
                f"vllm:generation_tokens_total {c['generated']}\nvllm:num_preemptions_total 0.0\n"
                f'vllm:prompt_tokens_by_source_total{{source="local_compute"}} {c["uncached"]}\n'
                'vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 0.0\n'
                f"vllm:num_requests_running {self.running}\nvllm:num_requests_waiting 0\n"
                "vllm:kv_cache_usage_perc 0.4\n"))
        if path.endswith("/reset_prefix_cache"):
            self.cache.clear()
            return httpx.Response(200, json={"success": True})
        if path.endswith("/chat/completions"):
            body = json.loads(request.content)
            prompt = max(1, len(json.dumps(body["messages"])) // 6)
            if not body.get("ignore_eos"):
                self.advance()
                self.counters["uncached"] += prompt
                return httpx.Response(200, json={"usage": {"prompt_tokens": prompt, "completion_tokens": 1},
                                                 "choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
            self.advance()
            self.running += 1
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            finally:
                self.advance()
                self.running -= 1
            raise AssertionError("a forced sweep request ran to completion")
        if path.endswith("/completions"):
            body = json.loads(request.content)
            ids = body["prompt"]
            if body.get("prompt_logprobs"):
                entries = [None] + [{k: {"logprob": v} for k, v in self.dist(i).items()} for i in range(1, len(ids))]
                self.register(ids)
                return httpx.Response(200, json={"choices": [{"prompt_logprobs": entries}],
                                                 "usage": {"prompt_tokens": len(ids)}})
            cached = self.hit(ids)
            self.register(ids)
            top = [{f"token_id:{k}": v for k, v in self.dist(len(ids)).items()}] if body.get("logprobs") else []
            return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": top}}],
                                             "usage": {"prompt_tokens": len(ids),
                                                       "prompt_tokens_details": {"cached_tokens": cached}}})
        raise AssertionError(path)


async def test_sweep_and_partial_check_against_a_fake_engine(tmp_path, monkeypatch):
    block, unit = 256, 16
    engine = FakeEngine(block, unit)
    transport = httpx.MockTransport(engine.handler)
    real_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)
    monkeypatch.setattr(tb.GpuMonitor, "run", lambda self: asyncio.sleep(3600))
    (tmp_path / "server.log").write_text("worker ARC_CACHE_LAYOUT " + json.dumps(layout()) + "\n")
    sequences = [[(i * 31 + s) % 1000 for i in range(7 * block + 64)] for s in (1, 2)]
    (tmp_path / "src").mkdir()
    (tmp_path / "src/seq.json").write_text(json.dumps({"sequences": sequences}))
    study = {"kv_margin": 0.9, "generation_slots": 32, "completion_cap": 16384, "reasoning_effort": "xhigh",
             "temperature": 0.7, "server": {"mtp_tokens": 0},
             "throughput_bench": {"prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None,
                                  "stream_sweep": {"lengths": [300, 500, 700, 900], "decode_counts": [2, 4],
                                                   "mixed_counts": [3], "output_tokens": 256, "window_seconds": 0.6,
                                                   "settle_seconds": 10, "inject_lead_seconds": 0.2,
                                                   "sample_seconds": 0.1, "calibrate_tokens": 400,
                                                   "inject": {"prefix_tokens": 300, "tail_tokens": 80, "rate": 5}},
                                  "partial_check": {"file": "seq.json", "block": block, "unit": unit, "probes": 6}}}
    report = await tb.run_bench(study, {"name": "precheck", "kind": "serving-bench"}, output=tmp_path / "out",
                                server_dir=tmp_path, base_url="http://model/v1", transport=transport,
                                log=lambda *a, **k: None, source=tmp_path / "src")
    assert report["status"] == "complete", report["task_errors"]
    sweep = report["stream_sweep"]
    assert not sweep["errors"] and sweep["prefill"]["requests"] == 4
    points = [(p["kind"], p["streams"]) for p in sweep["points"]]
    assert points == [("decode", 2), ("decode", 4), ("mixed", 3)]
    for point in sweep["points"]:
        assert point["valid"] and not point["ended_early"] and point["steps"] > 0
        assert math.isclose(point["generated_per_step"], point["streams"], rel_tol=0.05)
        assert math.isclose(point["step_ms"], 4.0, rel_tol=0.25)
    assert sweep["points"][2]["inject"]["sent"] >= 1
    assert engine.cancelled == 2 + 4 + 3 and engine.running == 0  # every forced request was aborted
    assert sweep["decode_fit"]["points"] == 2
    check = report["partial_check"]
    summary = check["summary"]
    assert summary["hits"] == {name: {"as_expected": 12, "probes": 12} for name in ("W16", "W16b", "W1600a", "W1600b")}
    assert summary["verdict"] == {"hits_exact": True, "kl_within": True, "top1_within": True, "pass": True}
    assert summary["pairs"]["C|W16"]["top1_agreement"] == 1.0
    saved = json.loads((tmp_path / "out/bench.json").read_text())
    assert saved["partial_check"]["summary"]["verdict"]["pass"] is True


def load_builder():
    spec = importlib.util.spec_from_file_location("build_arena_precheck", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precheck_study_and_modal_bound(tmp_path):
    builder = load_builder()
    hero = {"hero_v2": {"requested": {}}, "competition": {"research_clones": True},
            "server": {"max_num_seqs": 25}, "generation_slots": 25, "active_limit": 23}
    transcript = tmp_path / "t.txt"
    transcript.write_text("line\n" * 100)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"teacher_forced": [{"ids": list(range(12000)), "source": "a"},
                                                      {"ids": list(range(12000)), "source": "b"}]}))
    (tmp_path / "out").mkdir()
    study = builder.hero_v2_precheck_study(hero, tmp_path / "out", str(transcript), str(prompts))
    assert study["server"]["max_num_seqs"] == 32 and study["generation_slots"] == 32
    (trial,) = study["trials"]
    assert trial["name"] == "precheck" and trial["competition"] is False
    assert serving.lab_overrides(trial["server_env"]) == {"prefix_match_unit": 16, "prompt_tokens_details": True}
    bench = study["throughput_bench"]
    assert len(bench["stream_sweep"]["lengths"]) == 32 and bench["prefix_replay"]["unit"] == 16
    assert len(json.loads((tmp_path / "out" / bench["partial_check"]["file"]).read_text())["sequences"][0]) == 11264
    assert tb.grid_points(bench) == []  # no grid points: the sweep and checks only
    # Mixed passes leave slots for the injector under max_num_seqs 32.
    assert max(bench["stream_sweep"]["mixed_counts"]) <= 30
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "hero-v2-precheck")
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (timeout + 60) * rate * 1.15 <= 2.6
    assert ns["arena_arguments"]("hero-v2-precheck") == ["--trial", "precheck"]
    # The trial starts after ~705 s and needs wall + 480 s of the envelope (max_seconds = timeout - 60).
    assert timeout - 60 - 705 >= study["wall_seconds"] + 480
