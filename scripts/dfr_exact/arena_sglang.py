"""SGLang capacity and telemetry for the arena (v8.2 harness on dfranzen's SGLang server, 2026).

Replaces the bundle's ``arena/sglang.py``, whose ``sglang_bench`` dependency is not packaged: the same ``TokenBudget``
(exact input + output + lookahead, rounded to pages, against the measured ``max_total_num_tokens``), the arena's own
engine monitor (``arc_infra.metrics.parse_metrics`` reads ``sglang:`` counters under their vLLM names), and the
flush-based isolation before a trial.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

from arc_infra.metrics import GpuMonitor, parse_metrics


class TokenBudget:
    units = "tokens"

    def __init__(self, startup, study):
        profile = study["sglang_profile"]
        expected = {"max_total_num_tokens": profile["max_total_tokens"],
                    "max_running_requests": profile["admitted"],
                    "max_mamba_cache_size": profile["mamba_slots"],
                    "context_len": profile["context_length"]}
        mismatches = {k: {"expected": v, "actual": startup.get(k)}
                      for k, v in expected.items() if startup.get(k) != v}
        if mismatches:
            raise RuntimeError(f"SGLang capacity differs from requested profile: {mismatches}")
        if study["generation_slots"] > profile["admitted"]:
            raise ValueError("Arena slots exceed validated SGLang concurrency")
        self.capacity = int(startup["max_total_num_tokens"] * study["kv_margin"])
        self.max_length = startup["context_len"]
        self.page = profile["page_size"]
        self.layout = {"rule": "exact input + output + speculative lookahead, rounded to pages",
                       "startup": startup, "margin": study["kv_margin"], "units": self.units,
                       "num_blocks": startup["max_total_num_tokens"] // self.page}

    def cost(self, tokens):
        if tokens < 0 or tokens + 4 > self.max_length:
            raise ValueError("Request exceeds measured SGLang context capacity")
        return ((tokens + 4 + self.page - 1) // self.page) * self.page

    @classmethod
    def load(cls, server_dir, study):
        return cls(json.loads((Path(server_dir) / "startup.json").read_text()), study)


class ArenaSglangMonitor(GpuMonitor):
    def summary(self):
        result = super().summary()
        result["backend"] = "sglang"
        result["preemption_counter"] = "sglang:num_retracted_requests_total"
        return result


async def isolate(base_url):
    """Wait until the server is idle, then flush its prefix cache (HTTP 200 only once the scheduler has flushed)."""
    started = time.monotonic()
    root = base_url.removesuffix("/v1").rstrip("/")
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        for attempt in range(45):
            metrics = await client.get(root + "/metrics")
            metrics.raise_for_status()
            values = parse_metrics(metrics.text)
            if values.get("vllm:num_requests_running", 0) == 0 and values.get("vllm:num_requests_waiting", 0) == 0:
                response = await client.post(root + "/flush_cache")
                if response.status_code == 200:
                    return {"idle": True, "prefix_cache_reset": True, "backend": "sglang",
                            "attempts": attempt + 1, "seconds": time.monotonic() - started}
                if response.status_code not in (200, 400):
                    response.raise_for_status()
            await asyncio.sleep(1)
    raise RuntimeError("SGLang did not drain and flush; refusing contaminated trial")
