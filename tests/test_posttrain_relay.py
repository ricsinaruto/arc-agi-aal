"""Relay policy transport (posttrain.relay): request files, the vLLM-shaped reply and virtual time."""

import asyncio
import json

from posttrain.relay import RelayTransport, response_json, view_text


async def test_relay_writes_the_request_and_returns_the_answer_as_a_tool_call(tmp_path):
    transport = RelayTransport(tmp_path, poll_seconds=0.01)
    payload = {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "board"}],
               "tools": [{"type": "function", "function": {"name": "python"}}]}

    async def answer():
        while not (tmp_path / "PENDING").exists():
            await asyncio.sleep(0.01)
        assert "board" in (tmp_path / "turn-0001.view.txt").read_text()
        await asyncio.sleep(0.05)
        (tmp_path / "turn-0001.response.json").write_text(json.dumps({"reasoning": "think " * 100, "code": "print(1)"}))

    count = await transport.post("http://relay/v1/tokenize", json=payload)
    assert count.json()["count"] > 0
    before = transport.now()
    response, _ = await asyncio.gather(transport.post("http://relay/v1/chat/completions", json=payload), answer())
    body = response.json()
    message = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "tool_calls" and message["reasoning"].startswith("think")
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"code": "print(1)"}
    assert json.loads((tmp_path / "turn-0001.request.json").read_text()) == payload
    assert not (tmp_path / "PENDING").exists()
    # Virtual time: the wait is removed and replaced by overhead + completion tokens at 110 tok/s.
    advanced = transport.now() - before
    modeled = 1.65 + body["usage"]["completion_tokens"] / 110
    assert abs(advanced - modeled) < 0.05


def test_view_shows_only_the_messages_after_the_last_reply():
    payload = {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "first"},
                            {"role": "assistant", "content": "x"}, {"role": "tool", "content": "result"},
                            {"role": "user", "content": [{"type": "text", "text": "board"}, {"type": "image_url"}]}]}
    text = view_text(payload, 3)
    assert "result" in text and "board" in text and "[image]" in text and "first" not in text
    plain = response_json({"reasoning": "r", "content": "done"}, 1, 10)
    assert plain["choices"][0]["finish_reason"] == "stop" and "tool_calls" not in plain["choices"][0]["message"]
