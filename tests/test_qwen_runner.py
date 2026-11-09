import asyncio
import json
import os
from types import SimpleNamespace

import httpx
import pytest
from test_qwen_core import Environment, calls

from qwen_baselines.clock import Clock
from qwen_baselines.config import GAMES, Config
from qwen_baselines.evidence import Archive
from qwen_baselines.model import Conversation, RequestLayer
from qwen_baselines.runner import trial


async def test_shared_start_and_fresh_per_game_state(tmp_path):
    started = []

    class Layer:
        def __init__(self, config, clock):
            self.config, self.clock = config, clock

        async def preflight(self):
            return 100

        def payload(self, messages, tools):
            return messages

        def fresh(self, messages):
            return messages

        async def count(self, payload):
            return 100

        async def complete(self, messages, tools, archive):
            started.append((self.clock.start, self.clock.elapsed()))
            await asyncio.sleep(100)

        async def close(self):
            pass

    class Arcade:
        def get_environments(self):
            return [SimpleNamespace(game_id=g) for g in reversed(GAMES)]

        def open_scorecard(self, **kwargs):
            return "card"

        def make(self, *args, **kwargs):
            return Environment()

        def close_scorecard(self, card):
            return SimpleNamespace(
                model_dump=lambda **kwargs: {"environments": [{"id": g, "score": 0} for g in GAMES]}
            )

    report = await trial(
        Config(gameplay_seconds=0.05),
        "A",
        GAMES,
        tmp_path,
        tmp_path / "trial",
        pi_dir=tmp_path,
        layer_factory=Layer,
        arcade_factory=lambda *args: Arcade(),
    )
    assert [g["game"] for g in report["games"]] == list(GAMES)
    assert report["status"] == "complete" and 0.04 < report["gameplay_seconds"] < 1
    assert len(started) == len(GAMES) and len({s[0] for s in started}) == 1
    assert all(elapsed < 0.02 for _, elapsed in started)
    assert all(g["actions"] == 0 for g in report["games"])


async def test_fresh_clock_after_admission_native_allowance_and_mismatch(tmp_path):
    now = [0.0]
    clock = Clock(600, now=lambda: now[0])
    clock.release()
    requests = []

    def endpoint(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 500, "max_model_len": 262144})
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ready"}}]}
        )

    layer = RequestLayer(
        Config(generation_slots=1),
        clock,
        client=httpx.AsyncClient(
            base_url="http://mock/v1/", transport=httpx.MockTransport(endpoint)
        ),
    )
    await layer.slots.acquire()
    task = asyncio.create_task(
        layer.complete([{"role": "user", "content": "observe"}], [], Archive(tmp_path))
    )
    await asyncio.sleep(0)
    now[0] = 12
    layer.slots.release()
    await task
    body = requests[-1][1]
    assert '"elapsed_seconds": 12' in body["messages"][-1]["content"]
    # Wrapped so the template's last query stays at the last real observation.
    assert body["messages"][-1]["content"].startswith("<tool_response>\n")
    assert body["messages"][-1]["content"].endswith("\n</tool_response>")
    assert body["max_tokens"] == 262144 - 500
    assert body["chat_template_kwargs"]["preserve_thinking"] is False
    layer.config = Config(context_window=32768, history_tokens=None, compaction_tokens=None)
    with pytest.raises(RuntimeError, match="No downgrade"):
        await layer.preflight()
    await layer.close()


async def test_oversized_python_output_retrievable_and_reasoning_retained(tmp_path):
    class Layer:
        config = SimpleNamespace(context_window=2000, reserve_tokens=100)

        def payload(self, messages, tools):
            return messages

        def fresh(self, messages):
            return messages

        async def count(self, messages):
            return len(json.dumps(messages))

    archive = Archive(tmp_path)
    call = archive.begin_call("print('x' * 10000)")
    archive.output(call, "x" * 10000)
    archive.end_call(call, status="complete")
    observation = {"role": "user", "content": "authoritative current observation"}
    conv = Conversation("task", observation, archive)
    response = calls("ACTION1")
    response["tool_calls"][0]["function"] = {"name": "python", "arguments": '{"code":"print(1)"}'}
    conv.append(
        [
            response,
            {"role": "tool", "tool_call_id": "0", "content": json.dumps(archive.transcript(call))},
        ]
    )
    messages = await conv.maintain(Layer(), [])
    assert any(m.get("reasoning_content") == "preserved reasoning" for m in messages)
    result = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))
    assert result["retrieval"] == "transcript[0]['output'] (full output)"
    assert 0 < len(result["output"]) < 10000
    assert len(archive.transcript(0)["output"]) == 10000
    assert observation in messages


async def test_comparison_drains_and_stops_after_failed_trial(tmp_path, monkeypatch):
    from qwen_baselines import runner

    events = []

    async def idle(url):
        events.append("drain")

    async def run(config, baseline, *args, **kwargs):
        events.append(baseline)
        return {"status": "error" if baseline == "B" else "complete"}

    monkeypatch.setattr(runner, "wait_idle", idle)
    monkeypatch.setattr(runner, "trial", run)
    with pytest.raises(RuntimeError, match="Trial B failed"):
        await runner.comparison(Config(), "ABC", GAMES, tmp_path, tmp_path, pi_dir=tmp_path)
    assert events == ["drain", "A", "drain", "drain", "B", "drain"]


@pytest.mark.parametrize("cache_bytes", [0, 8 * 1024**3])
@pytest.mark.parametrize("mtp_tokens", [1, 3])
def test_native_profile_keeps_historical_profile_and_fails_closed(tmp_path, monkeypatch, cache_bytes, mtp_tokens):
    from arc_infra.serving import SERVER_CONTEXT
    from qwen_baselines import serving

    recipe = SimpleNamespace(ANALYZER_CONTEXT=SERVER_CONTEXT)
    events = []
    def main():
        events.append(recipe.ANALYZER_CONTEXT)
        (tmp_path / "vllm-server-identity.json").write_text(json.dumps({"argv": ["--max-num-seqs", "8"]}))
    recipe.main = main
    recipe.stop_owned_server = lambda: events.append("stopped")

    async def failed():
        raise RuntimeError("Insufficient KV capacity")

    monkeypatch.setattr(serving, "verify_hardware", lambda: "expected GPU")
    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout='{"type": "ready"}', stderr=""
        ),
    )
    loads = []
    monkeypatch.setattr(serving, "stage_fork_bundle", lambda root, staging: tmp_path / "fork")
    monkeypatch.setattr(
        serving, "load_recipe", lambda *args, **kwargs: (loads.append(kwargs), recipe)[1]
    )
    monkeypatch.setattr(serving, "preflight", failed)
    with pytest.raises(RuntimeError, match="KV capacity"):
        serving.start(tmp_path, tmp_path, kv_cache_memory_bytes=cache_bytes, mtp_tokens=mtp_tokens)
    assert events == [262144, "stopped"] and SERVER_CONTEXT == 32768
    assert loads[0]["bundle_dir"] == tmp_path / "fork" and loads[0]["kv_cache_memory_bytes"] == cache_bytes
    assert loads[0]["prefix_caching"] is True and serving.PROFILE["prefix_caching"] is True
    assert loads[0]["mtp_tokens"] == mtp_tokens
    assert loads[0]["setup_sha256"] == serving.fork_identity()["fork_sha256"]
    failure = json.loads((tmp_path / "native-preflight-failure.json").read_text())
    assert "do not silently lower context" in failure["diagnostic"]
    assert (str(cache_bytes) if cache_bytes else "utilization 0.96") in failure["diagnostic"]
    assert failure["kv_admission_log"] == []
    assert failure["profile"]["kv_cache_memory_bytes"] == cache_bytes
    assert failure["profile"]["speculative_tokens"] == mtp_tokens
    assert serving.PROFILE["kv_cache_memory_bytes"] == 0


def test_native_profile_gpu_memory_utilization_override(tmp_path, monkeypatch):
    # a lower utilization leaves GPU headroom outside the profiled KV pool (vision encoder OOM).
    from qwen_baselines import serving

    seen = {}
    recipe = SimpleNamespace(ANALYZER_CONTEXT=0, stop_owned_server=lambda: None)
    def main():
        seen.update(env=os.environ[serving.GPU_MEMORY_UTILIZATION_ENV])
        (tmp_path / "vllm-server-identity.json").write_text(json.dumps({"argv": ["--max-num-seqs", "8"]}))
    recipe.main = main

    async def failed():
        raise RuntimeError("stop after the server start")

    monkeypatch.setattr(serving, "verify_hardware", lambda: "expected GPU")
    monkeypatch.setattr(serving.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout='{"type": "ready"}', stderr=""))
    monkeypatch.setattr(serving, "stage_fork_bundle", lambda root, staging: tmp_path / "fork")
    monkeypatch.setattr(serving, "load_recipe", lambda *a, **k: recipe)
    monkeypatch.setattr(serving, "preflight", failed)
    monkeypatch.setenv(serving.GPU_MEMORY_UTILIZATION_ENV, "0.96")
    with pytest.raises(RuntimeError, match="stop after"):
        serving.start(tmp_path, tmp_path, gpu_memory_utilization="0.92")
    failure = json.loads((tmp_path / "native-preflight-failure.json").read_text())
    assert seen["env"] == "0.92" and failure["profile"]["gpu_memory_utilization"] == "0.92"
    assert "utilization 0.92" in failure["diagnostic"]
    for bad in ("0.995", "1.0", "0.4", 0.92):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            serving.start(tmp_path, tmp_path, gpu_memory_utilization=bad)
