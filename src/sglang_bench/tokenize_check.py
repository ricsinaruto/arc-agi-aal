"""Fail before gameplay if token counts disagree with actual generation usage."""

import asyncio
import base64
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image


def probes(model):
    def image(size):
        output = io.BytesIO()
        Image.new("RGB", (size, size), (40, 120, 200)).save(output, format="PNG")
        return {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}}

    tool = {"type": "function", "function": {"name": "python", "description": "Run Python.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                           "required": ["code"]}}}
    history = [
        {"role": "user", "content": [{"type": "text", "text": "Inspect this synthetic board."}, image(256)]},
        {"role": "assistant", "content": None, "reasoning_content": "Check the observed dimensions.",
         "tool_calls": [{"id": "count_probe", "type": "function", "function": {
             "name": "python", "arguments": '{"code":"print(64)"}'}}]},
        {"role": "tool", "tool_call_id": "count_probe", "content": "64"},
        {"role": "user", "content": [{"type": "text", "text": "Here is the next board."}, image(512)]},
    ]
    for name, messages in [
        ("text", [{"role": "user", "content": "Reply briefly."}]),
        ("image256", history[:1]),
        ("mixed_images_reasoning_tools", history),
    ]:
        yield name, {"model": model, "messages": messages, "tools": [tool],
                     "tool_choice": "auto", "chat_template_kwargs": {
                         "enable_thinking": True, "reasoning_effort": "xhigh"},
                     "temperature": 0, "max_tokens": 1}


async def run(base_url, model, output, *, transport=None, timeout=180):
    started = time.monotonic()
    record = {"state": "running", "probes": []}
    try:
        async with asyncio.timeout(timeout), httpx.AsyncClient(
            timeout=timeout, trust_env=False, transport=transport
        ) as client:
            for name, payload in probes(model):
                began = time.monotonic()
                row = {"name": name}
                record["probes"].append(row)
                token_body = {k: v for k, v in payload.items() if k not in ("temperature", "max_tokens")}
                counted = await client.post(base_url.removesuffix("/v1") + "/tokenize", json=token_body)
                counted.raise_for_status()
                row["count"] = counted.json()["count"]
                row["count_seconds"] = time.monotonic() - began
                generated = await client.post(base_url.rstrip("/") + "/chat/completions", json=payload)
                generated.raise_for_status()
                row["usage"] = generated.json()["usage"]
                row["seconds"] = time.monotonic() - began
                if type(row["count"]) is not int or row["count"] != row["usage"]["prompt_tokens"]:
                    raise RuntimeError(f"SGLang tokenize/generation mismatch: {name}: {row}")
        record["state"] = "passed"
        return record
    except BaseException as exc:
        record.update(state="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        Path(output).write_text(json.dumps(record, indent=2))
