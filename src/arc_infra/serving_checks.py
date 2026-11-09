"""Checks shared by every GPU study: pinned profile, idle server, live tokenizer."""

import asyncio
import time

import httpx

from .metrics import parse_metrics
from .model import ModelClient


async def wait_idle(base_url):
    deadline = time.monotonic() + 45
    async with httpx.AsyncClient(trust_env=False) as client:
        while time.monotonic() < deadline:
            response = await client.get(base_url.removesuffix("/v1") + "/metrics", timeout=5)
            response.raise_for_status()
            metrics = parse_metrics(response.text)
            keys = ("vllm:num_requests_running", "vllm:num_requests_waiting")
            if all(key in metrics and metrics[key] == 0 for key in keys):
                return
            await asyncio.sleep(1)
    raise TimeoutError("Server still has requests from the previous trial")


async def validate_api(cfg):
    client = ModelClient(cfg)
    try:
        count = await client.count_tokens(
            [{"role": "user", "content": "Inspect the puzzle before selecting a tool."}],
            time.monotonic() + 30,
        )
        if count <= 0:
            raise RuntimeError("Serving tokenizer returned an invalid count")
        return {"tokenizer_count": count, "tokenizer_request_schema": "accepted"}
    finally:
        await client.close()
