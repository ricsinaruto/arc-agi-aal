"""Read-only tokenizer recovery never replays generation or exceeds gameplay."""

import asyncio
import json

import httpx
import pytest

from qwen_baselines.clock import Clock
from qwen_baselines.config import Config
from qwen_baselines.evidence import Archive
from qwen_baselines.model import RequestLayer


def layer_for(endpoint, seconds=600):
    clock = Clock(seconds)
    layer = RequestLayer(Config(), clock, client=httpx.AsyncClient(
        base_url="http://mock/v1/", transport=httpx.MockTransport(endpoint)))
    clock.release()
    return layer


async def test_transport_recovery_reuses_exact_native_tokenizer_body():
    bodies = []

    def endpoint(request):
        assert request.url.path == "/tokenize"
        bodies.append(json.loads(request.content))
        if len(bodies) < 3:
            raise httpx.ReadError("connection reset", request=request)
        return httpx.Response(200, json={"count": 417, "max_model_len": 262144})

    layer = layer_for(endpoint)
    payload = layer.payload([
        {"role": "assistant", "reasoning_content": "retained reasoning", "content": ""},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]},
    ], [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}])
    try:
        assert await layer.count(payload) == 417
        assert len(bodies) == 3 and bodies[0] == bodies[1] == bodies[2]
        assert bodies[0]["messages"] == payload["messages"]
        assert bodies[0]["tools"] == payload["tools"]
        assert bodies[0]["chat_template_kwargs"] == payload["chat_template_kwargs"]
        assert layer.tokenizer_retries == 2
    finally:
        await layer.close()


async def test_transport_failure_stops_after_three_attempts():
    requests = []

    def endpoint(request):
        requests.append(request.url.path)
        raise httpx.ReadError("persistent", request=request)

    layer = layer_for(endpoint)
    try:
        with pytest.raises(httpx.ReadError, match="persistent"):
            await layer.preflight()
        assert requests == ["/tokenize"] * 3
        assert layer.tokenizer_retries == 2
    finally:
        await layer.close()


@pytest.mark.parametrize("status,data,match", [
    (400, {"error": "invalid image"}, "400.*invalid image"),
    (200, {"count": 12, "max_model_len": 32768}, "Native context preflight failed"),
])
async def test_validation_failures_are_not_retried(status, data, match):
    requests = []

    def endpoint(request):
        requests.append(request.url.path)
        return httpx.Response(status, json=data)

    layer = layer_for(endpoint)
    try:
        with pytest.raises(RuntimeError, match=match):
            await layer.preflight()
        assert requests == ["/tokenize"] and layer.tokenizer_retries == 0
    finally:
        await layer.close()


async def test_backoff_is_cancelled_at_gameplay_deadline():
    requests = []

    def endpoint(request):
        requests.append(request.url.path)
        raise httpx.ReadError("transient", request=request)

    layer = layer_for(endpoint, seconds=.01)
    try:
        with pytest.raises(TimeoutError):
            await layer.preflight()
        assert requests == ["/tokenize"] and layer.tokenizer_retries == 0
        with pytest.raises(TimeoutError, match="before tokenization"):
            await layer.preflight()
        assert len(requests) == 1
    finally:
        await layer.close()


async def test_external_cancellation_stops_recovery():
    failed = asyncio.Event()
    requests = []

    def endpoint(request):
        requests.append(request.url.path)
        failed.set()
        raise httpx.ReadError("transient", request=request)

    layer = layer_for(endpoint)
    try:
        task = asyncio.create_task(layer.preflight())
        await failed.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert requests == ["/tokenize"] and layer.tokenizer_retries == 0
    finally:
        await layer.close()


async def test_hanging_tokenizer_is_bounded_even_outside_generation():
    entered = asyncio.Event()

    async def endpoint(request):
        entered.set()
        await asyncio.Event().wait()

    layer = layer_for(endpoint, seconds=.01)
    try:
        with pytest.raises(TimeoutError):
            await layer.preflight()
        assert entered.is_set() and layer.tokenizer_retries == 0
    finally:
        await layer.close()


async def test_generation_transport_failure_is_never_replayed(tmp_path):
    requests = []

    def endpoint(request):
        requests.append(request.url.path)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 417, "max_model_len": 262144})
        raise httpx.ReadError("ambiguous generation", request=request)

    layer, archive = layer_for(endpoint), Archive(tmp_path)
    try:
        with pytest.raises(httpx.ReadError, match="ambiguous generation"):
            await layer.complete([{"role": "user", "content": "observe"}], None, archive)
        assert requests == ["/tokenize", "/v1/chat/completions"]
        assert layer.tokenizer_retries == 0 and layer.active == 0
        assert archive.db.execute("select count(*) from events where kind='generation_interrupted'").fetchone()[0] == 1
    finally:
        archive.close()
        await layer.close()
