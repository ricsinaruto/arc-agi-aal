"""``text_count`` (B200, 5 October): exact counts with the images sent as their template text."""

import asyncio
import base64
import json
import struct
import zlib
from types import SimpleNamespace as NS

import httpx
import pytest

from arc_infra import model


def png(width, height):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return "data:image/png;base64," + base64.b64encode(data + b"\0" * 64).decode()


def served_count(payload):
    """A serving stand-in: one token per word, three for the image text, (w/32) * (h/32) + 2 per image."""
    count = 7 + (5 if payload.get("tools") else 0)
    for message in payload["messages"]:
        content = message["content"]
        for part in content if isinstance(content, list) else [{"type": "text", "text": content}]:
            if part["type"] == "image_url":
                width, height = model.png_size(part["image_url"]["url"])
                count += (width // 32) * (height // 32) + 2
            elif part["text"] == model.IMAGE_TEXT:
                count += 3
            else:
                count += len(part["text"].split())
    return count


def client():
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append(sum(isinstance(m["content"], list) and any(p["type"] == "image_url" for p in m["content"])
                         for m in payload["messages"]))
        return httpx.Response(200, json={"count": served_count(payload), "max_model_len": 262144})

    c = model.ModelClient(NS(base_url="http://local/v1", parallel_games=1, model="m", preserve_reasoning=True,
                             reasoning_effort=None, max_model_len=131072))
    c.client = httpx.AsyncClient(base_url="http://local/v1/", transport=httpx.MockTransport(handle))
    return c, calls


def test_text_images_count_equals_the_multimodal_count_and_processes_each_size_once():
    big, small = png(640, 640), png(320, 256)
    messages = [{"role": "system", "content": "rules of the game"},
                {"role": "user", "content": [{"type": "text", "text": "board now"},
                                             {"type": "image_url", "image_url": {"url": big}}]},
                {"role": "assistant", "content": "I move left"},
                {"role": "tool", "content": [{"type": "text", "text": "diff"},
                                             {"type": "image_url", "image_url": {"url": small}},
                                             {"type": "image_url", "image_url": {"url": big}}]}]
    c, calls = client()

    async def run():
        deadline = asyncio.get_running_loop().time() + 60
        exact = await c.count_tokens(messages, deadline, tools=[{"x": 1}])
        first = await c.count_tokens(messages, deadline, tools=[{"x": 1}], text_images=True)
        again = await c.count_tokens(messages, deadline, tools=[{"x": 1}], text_images=True)
        return exact, first, again

    exact, first, again = asyncio.run(run())
    assert exact == first == again
    assert c.image_tokens == {(640, 640): 400, (320, 256): 80}
    # The full count once, then one single-image probe per size; the text counts never carry an image.
    assert calls == [2, 1, 0, 1, 0, 0, 0]


def test_non_png_images_fall_back_to_the_multimodal_count():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/a.png"}}]}]
    assert model.text_image_messages(messages) is None
    assert model.text_image_messages([{"role": "user", "content": "plain"}]) == ([{"role": "user", "content": "plain"}], [])


@pytest.mark.parametrize("body, error", [
    ({"tokens": [7] * 131072, "count": 131072, "max_model_len": 262144}, None),
    ({"count": 1, "max_model_len": 4096}, "Serving context is smaller"),
    ({"count": "1", "max_model_len": 262144}, "valid integer"),
    ({"max_model_len": 262144}, "Field required"),
    (b'{"count":1,"max_model_len":262144,"tokens":[broken]}', "Invalid JSON"),
])
async def test_token_count_ignores_ids_and_validates_metadata(body, error):
    c, _ = client()
    await c.close()
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    c.client = httpx.AsyncClient(base_url="http://local/v1/", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=content)))
    try:
        if error:
            with pytest.raises(ValueError, match=error):
                await c.count_tokens([], 1e12)
        else:
            assert await c.count_tokens([], 1e12) == 131072
    finally:
        await c.close()
