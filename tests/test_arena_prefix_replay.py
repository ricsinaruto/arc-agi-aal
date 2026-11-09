"""Prefix-cache v2 smoke checks (29 September): replay hit accounting, warm/cold correctness, the two-arm build."""

import ast
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from arena import prefix_replay as pr

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 16


def render(messages):
    """A fake chat template: append-only concatenation of every message's text."""
    parts = []
    for m in messages:
        parts.append(f"<{m['role']}>")
        for key in ("reasoning", "content"):
            if m.get(key):
                parts.append(str(m[key]))
        for call in m.get("tool_calls") or []:
            parts.append(call["function"]["arguments"])
    return "".join(parts) + "<assistant>"


def ids(text):
    return [ord(c) % 997 for c in text]


class FakeServer:
    """Chat completions with token ids and a perfect block-floored prefix cache (or a lagging one)."""

    def __init__(self, lag=False, drop_reply=False):
        self.cache = []  # token-id sequences seen (prompt + output)
        self.lag, self.drop_reply = lag, drop_reply
        self.resets = 0

    def handler(self, request):
        path = request.url.path
        if path.endswith("/reset_prefix_cache"):
            self.cache.clear()
            self.resets += 1
            return httpx.Response(200, json={"success": True})
        if path.endswith("/detokenize"):
            return httpx.Response(200, json={"prompt": "".join(chr(t) for t in json.loads(request.content)["tokens"])})
        if path.endswith("/metrics"):
            return httpx.Response(200, text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n")
        body = json.loads(request.content)
        messages = body["messages"]
        if self.drop_reply:
            messages = [{**m, "reasoning": None} if m["role"] == "assistant" else m for m in messages]
        prompt = ids(render(messages))
        best = 0
        sources = self.cache[:-1] if self.lag and self.cache else self.cache
        for seq in sources:
            best = max(best, pr.common_prefix(seq, prompt))
        cached = min(best, len(prompt) - 1) // BLOCK * BLOCK
        n = len(self.cache)
        tool = {"id": f"call-{n}", "type": "function", "function": {"name": "python",
                                                                      "arguments": json.dumps({"code": f"x={n}"})}}
        reply_text = f"think {n}" + tool["function"]["arguments"]
        output = ids(reply_text)
        self.cache.append(prompt + output)
        answer = body.get("temperature") == 0.0
        return httpx.Response(200, json={
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(output),
                      "prompt_tokens_details": {"cached_tokens": cached}},
            "prompt_token_ids": prompt,
            "choices": [{"finish_reason": "stop" if answer else "tool_calls", "token_ids": output[:5] if answer else output,
                         "logprobs": {"content": [{"top_logprobs": [{"token": "a", "logprob": -0.1}]}]},
                         "message": {"role": "assistant", "content": "ok" if answer else None,
                                     "reasoning": f"think {n}", "tool_calls": [] if answer else [tool]}}]})


REAL_CLIENT = httpx.AsyncClient
CURRENT = {}


def client_for(server):
    CURRENT["transport"] = httpx.MockTransport(server.handler)
    return REAL_CLIENT(transport=CURRENT["transport"], base_url="http://model")


@pytest.fixture(autouse=True)
def fake_network(monkeypatch):
    async def idle(base_url):
        return None

    def patched(*args, **kwargs):  # isolate_trial opens its own client for the cache reset
        kwargs.setdefault("transport", CURRENT["transport"])
        return REAL_CLIENT(*args, **kwargs)

    monkeypatch.setattr("arena.isolation.wait_idle", idle)
    monkeypatch.setattr("arena.isolation.httpx.AsyncClient", patched)


def test_classify_splits_ideal_round_trip_and_prompt_end():
    previous = {"prompt_token_ids": list(range(40)), "token_ids": list(range(40, 60))}
    exact = pr.classify({"prompt_token_ids": list(range(75)), "cached_tokens": 48}, previous, BLOCK)
    assert exact["ideal_hit"] == 48 and exact["reply_round_trip_exact"] and exact["reply_tokens_reused"] == 20
    assert exact["previous_prompt_end"] == 32
    rewritten = pr.classify({"prompt_token_ids": list(range(50)) + [999] * 25}, previous, BLOCK)
    assert rewritten["ideal_hit"] == 48 and not rewritten["reply_round_trip_exact"]
    assert rewritten["reply_tokens_reused"] == 10
    assert pr.classify({"prompt_token_ids": [1]}, None, BLOCK) == {"prompt_token_ids": [1]}


@pytest.mark.parametrize("lag", [False, True])
async def test_replay_accounts_exact_hits_and_the_lag(lag):
    server = FakeServer(lag=lag)
    async with client_for(server) as client:
        result = await pr.prefix_replay(client, "http://model/v1", model="m", effort="xhigh",
                                        transcript="observation line\n" * 400,
                                        spec={"games": 3, "requests": 4, "concurrency": 1, "system_chars": 200,
                                              "first_chars": 100, "tool_chars": 60}, block=BLOCK,
                                        log=lambda *a, **k: None)
    s = result["summary"]
    assert s["requests"] == 12 and s["errors"] == 0 and s["continuations"] == 9
    assert s["reply_round_trip_exact"] == 9 and s["reply_tokens"] == s["reply_tokens_reused"] > 0
    if lag:
        # Concurrency 1: the lagging cache only knows the request before the previous one.
        assert s["shortfall_tokens"] > 0 and s["hit_vs_ideal"]["ideal"] < 9
    else:
        assert s["shortfall_tokens"] == 0 and s["hit_vs_ideal"] == {"ideal": 9, "previous_prompt_end": 0,
                                                                   "shortfall_other": 0}
    assert all("prompt_token_ids" not in r for r in result["records"])


async def test_replay_measures_a_lossy_reply_round_trip():
    server = FakeServer(drop_reply=True)
    async with client_for(server) as client:
        result = await pr.prefix_replay(client, "http://model/v1", model="m", effort="xhigh",
                                        transcript="obs\n" * 400, spec={"games": 1, "requests": 3, "concurrency": 1,
                                                                        "system_chars": 50, "first_chars": 40,
                                                                        "tool_chars": 30}, block=BLOCK)
    s = result["summary"]
    assert s["continuations"] == 2 and s["reply_round_trip_exact"] == 0 and s["reply_tokens_reused"] < s["reply_tokens"]
    div = next(r["divergence"] for r in result["records"] if r.get("divergence"))
    assert div["offset_in_reply"] >= 0 and div["generated_text"] and div["rendered_text"]


async def test_prefix_check_compares_warm_and_twice_cold():
    server = FakeServer()
    async with client_for(server) as client:
        result = await pr.prefix_check(client, "http://model/v1", model="m", transcript="line of text\n" * 5000,
                                       spec={"prompt_chars": 4000, "cut_fractions": [0.5, 0.9], "output_tokens": 4},
                                       log=lambda *a, **k: None)
    assert result["summary"]["cuts"] == result["summary"]["clean"] == 2
    for cut in result["cuts"]:
        assert cut["warm_cached_tokens"] > 0 and cut["cold_cached_tokens"] == 0
        assert cut["cold_matches_cold"] and cut["warm_cold_first_logprob_distance"] == 0
    # Reset before every cold answer and before re-priming.
    assert server.resets == 1 + 3 * 2


async def test_serving_bench_runs_both_checks_on_the_served_mamba_block(tmp_path, monkeypatch):
    from test_arena_cache import layout

    from arena import throughput_bench as tb

    server = FakeServer()
    transport = httpx.MockTransport(server.handler)
    CURRENT["transport"] = transport
    monkeypatch.setattr(tb.GpuMonitor, "run", lambda self: __import__("asyncio").sleep(3600))
    monkeypatch.setattr("arena.throughput_bench.wait_idle", lambda base_url: __import__("asyncio").sleep(0))
    served = layout()
    served["groups"][1]["type"] = "MambaSpec"
    (tmp_path / "server.log").write_text("worker ARC_CACHE_LAYOUT " + json.dumps(served) + "\n")
    (tmp_path / "t.txt").write_text("observation line\n" * 800)
    seen = {}

    async def replay(client, base_url, **kwargs):
        seen.update(kwargs)
        return {"summary": {"requests": 0}}

    monkeypatch.setattr(pr, "prefix_replay", replay)
    study = {"kv_margin": 0.9, "generation_slots": 25, "completion_cap": 16384, "reasoning_effort": "xhigh",
             "temperature": 0.7, "server": {"mtp_tokens": 0},
             "throughput_bench": {"prompt_tokens": [], "concurrency": [], "shared_prefix": None, "points": [],
                                  "transcript": {"file": str(tmp_path / "t.txt"), "prompt_tokens": [], "concurrency": []},
                                  "prefix_check": {"prompt_chars": 2000, "cut_fractions": [0.5], "output_tokens": 4},
                                  "prefix_replay": {"games": 2}}}
    report = await tb.run_bench(study, {"name": "prefix", "kind": "serving-bench"}, output=tmp_path / "out",
                                server_dir=tmp_path, base_url="http://model/v1", transport=transport,
                                log=lambda *a, **k: None)
    assert not report["task_errors"] and report["points"] == []
    assert report["prefix_check"]["summary"]["clean"] == 1 and report["prefix_replay"] == {"summary": {"requests": 0}}
    assert seen["block"] == 100 and seen["spec"] == {"games": 2}


def test_top_distance():
    assert pr.top_distance([{"a": -0.1, "b": -2.0}], [{"a": -0.3, "c": -1.0}]) == pytest.approx(0.2)
    assert pr.top_distance([{"a": -0.1}], [{"b": -0.1}]) == float("inf")
    assert pr.top_distance([], [{"a": 0}]) is None


def load_builder():
    spec = importlib.util.spec_from_file_location("build_arena_prefix", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefix_cache_smoke_build_and_bound(tmp_path):
    from arena.variants import HERO_PREFIX_CACHE

    builder = load_builder()
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("step 1\n" * 1000)
    output, study = builder.build("prefix", output=tmp_path / "prefix", paired_30m=True, throughput_bench=True,
                                  arena_variant="fixed-2x-mtp0-wide-kv8-hero-prefix", prefix_cache_smoke=True,
                                  bench_transcript=str(transcript))
    pool = str(builder.HERO_KV_POOL_BYTES * builder.PREFIX_SMOKE_POOL_BLOCKS // 860)
    common = {"ARENA_PROMPT_TOKENS_DETAILS": "1", "ARENA_KV_CACHE_MEMORY_BYTES": pool}
    assert study["trials"] == [
        {"name": "pinned", "kind": "serving-bench", "server_env": {**common, "ARENA_PREFIX_CACHE": "off"}},
        {"name": "prefix", "kind": "serving-bench",
         "server_env": {**common, "ARENA_PREFIX_CACHE": json.dumps(HERO_PREFIX_CACHE)}}]
    grid = study["throughput_bench"]
    assert grid["prefix_check"] and grid["prefix_replay"]["games"] == 24
    assert grid["transcript"]["prompt_tokens"] == grid["transcript"]["concurrency"] == [] and grid["points"] == []
    assert (output / grid["transcript"]["file"]).is_file()
    assert study["cache_experiment"]["prefix_cache"] == HERO_PREFIX_CACHE
    with pytest.raises(ValueError, match="prefix-cache smoke"):
        builder.build("bad", output=tmp_path / "bad", paired_30m=True, throughput_bench=True,
                      arena_variant="fixed-2x-mtp0-wide-kv8-hero-prefix", bench_transcript=str(transcript))
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert (ns["validation_timeout"]("arena", "prefix-cache-smoke") + 60) * rate * 1.15 <= 5
    assert ns["arena_arguments"]("prefix-cache-smoke") == []
