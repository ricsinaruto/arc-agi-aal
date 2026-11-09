"""Checks shared by every GPU study: pinned profile, idle server, live tokenizer."""

import asyncio
import time

import httpx

from .metrics import parse_metrics
from .model import ModelClient
from .serving import BASE_URL, MODEL, REVISION, SUPPORTED_CONTEXTS


def validate_config(cfg):
    if (cfg.model, cfg.model_revision) != (MODEL, REVISION):
        raise ValueError("Flash-Next experiments require the pinned NVFP4 model")
    if cfg.base_url != BASE_URL or cfg.max_model_len not in SUPPORTED_CONTEXTS or (
        type(cfg.max_num_seqs) is not int or cfg.max_num_seqs not in (8, 12, 16)
    ):
        raise ValueError("Agent configuration differs from the pinned serving profile")
    if cfg.gameplay_seconds not in (600, 900, 1200, 3600) or (
        type(cfg.parallel_games) is not int or cfg.parallel_games not in (4, 8, 12, 16)
        or cfg.parallel_games > cfg.max_num_seqs
    ):
        raise ValueError(
            "Use supported admission/server capacity and a 600-, 900-, 1200- or 3600-second trial"
        )
    if cfg.thinking_budget or cfg.temperature != 0.7:
        raise ValueError("Keep voluntary uncapped gameplay at temperature 0.7")
    if cfg.max_output_tokens not in (0, 4096, 8192):
        raise ValueError("Duck profile output cap is 0 (native), 4096 (v5) or 8192 (v5b corrected agent)")
    if type(cfg.speculative_tokens) is not int or cfg.speculative_tokens not in (1, 3, 4):
        raise ValueError("Flash serving comparisons support MTP depth 1, 3 or 4")


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
