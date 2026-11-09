"""Measured SGLang capacity and telemetry for the shared arena, without vLLM units."""

import asyncio
import json
import time
from pathlib import Path

import httpx

from sglang_bench.metrics import parse_sglang_metrics
from sglang_bench.transfer import FleetMonitor


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
            raise ValueError("Arena slots exceed validated SGLang concurrency/Mamba slots")
        self.capacity = int(startup["max_total_num_tokens"] * study["kv_margin"])
        self.max_length = startup["context_len"]
        self.page = profile["page_size"]
        self.layout = {"rule": "exact input + output + speculative lookahead, rounded to pages",
                       "startup": startup, "margin": study["kv_margin"], "units": self.units}

    def cost(self, tokens):
        if tokens < 0 or tokens + 4 > self.max_length:
            raise ValueError("Request exceeds measured SGLang context capacity")
        return ((tokens + 4 + self.page - 1) // self.page) * self.page

    @classmethod
    def load(cls, server_dir, study):
        return cls(json.loads((Path(server_dir) / "startup.json").read_text()), study)


class ArenaSglangMonitor(FleetMonitor):
    def summary(self):
        result = super().summary()
        result["backend"] = "sglang"
        result["preemptions"] = result.get("retracted_requests", 0)
        result["preemption_counter"] = "sglang:num_retracted_requests_total"
        # Empty/malformed scrapes must not produce a successful arena report.
        required = {"sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:max_total_num_tokens"}
        if not self.samples or any(not required <= sample["values"].keys() for sample in self.samples):
            result["valid"] = False
        return result


async def isolate(base_url):
    """SGLang flush returns HTTP 200 only when the scheduler has actually flushed."""
    started = time.monotonic()
    root = base_url.removesuffix("/v1")
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        for attempt in range(45):
            metrics = await client.get(root + "/metrics")
            metrics.raise_for_status()
            values = parse_sglang_metrics(metrics.text)["values"]
            keys = ("sglang:num_running_reqs", "sglang:num_queue_reqs")
            if all(values.get(key) == 0 for key in keys):
                response = await client.post(root + "/flush_cache")
                if response.status_code == 200 and response.text.startswith("Cache flushed"):
                    return {"idle": True, "prefix_cache_reset": True, "backend": "sglang",
                            "attempts": attempt + 1, "seconds": time.monotonic() - started}
                if response.status_code not in (200, 400):
                    response.raise_for_status()
            await asyncio.sleep(1)
    raise RuntimeError("SGLang did not drain and flush; refusing contaminated trial")
