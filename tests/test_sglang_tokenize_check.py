import asyncio
import json

import httpx
import pytest

from sglang_bench import tokenize_check


@pytest.mark.parametrize("failure", [None, "mismatch", "http", "timeout"])
async def test_parity_records_evidence_and_stops_on_mismatch(tmp_path, failure):
    requests = []

    async def endpoint(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if failure == "timeout":
            await asyncio.sleep(10)
        if request.url.path == "/tokenize":
            assert "max_tokens" not in body
            return httpx.Response(200, json={"count": 123})
        assert body["max_tokens"] == 1
        if failure == "http":
            return httpx.Response(500, json={"error": "preprocessing failed"})
        return httpx.Response(200, json={"usage": {"prompt_tokens": 122 if failure == "mismatch" else 123}})

    output = tmp_path / "parity.json"
    args = ("http://mock/v1", "pennyroyal", output)
    kwargs = {"transport": httpx.MockTransport(endpoint), "timeout": .02 if failure == "timeout" else 10}
    if failure:
        error = {"mismatch": RuntimeError, "http": httpx.HTTPStatusError, "timeout": TimeoutError}[failure]
        with pytest.raises(error):
            await tokenize_check.run(*args, **kwargs)
        record = json.loads(output.read_text())
        assert record["state"] == "failed" and len(record["probes"]) == 1
        assert record["error"]["type"] == error.__name__
    else:
        result = await tokenize_check.run(*args, **kwargs)
        assert result == json.loads(output.read_text())
        assert result["state"] == "passed" and len(result["probes"]) == 3
        assert len(requests) == 6
        for (_, tokenized), (_, generated) in zip(requests[::2], requests[1::2], strict=True):
            assert tokenized == {k: v for k, v in generated.items() if k not in ("max_tokens", "temperature")}
        messages = requests[-1][1]["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
        assert messages[1]["reasoning_content"] and messages[1]["tool_calls"]
