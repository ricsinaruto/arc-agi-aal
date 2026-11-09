"""Fail-closed cache isolation between independent trials on one vLLM server."""

import asyncio
import time

import httpx

from arc_infra.serving_checks import wait_idle


async def isolate_trial(base_url):
    started = time.monotonic()
    await wait_idle(base_url)
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        for attempt in range(10):
            response = await client.post(
                base_url.removesuffix("/v1") + "/reset_prefix_cache",
                params={"reset_external": "true"},
            )
            response.raise_for_status()
            if response.json().get("success") is True:
                return {"idle": True, "prefix_cache_reset": True,
                        "attempts": attempt + 1, "seconds": time.monotonic() - started}
            await asyncio.sleep(1)
    raise RuntimeError("Previous trial cache could not be reset; refusing contaminated comparison")
