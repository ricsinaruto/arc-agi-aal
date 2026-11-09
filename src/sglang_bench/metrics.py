"""SGLang ``/metrics`` sampling and summary (``sglang:`` Prometheus names).

The vLLM monitor in ``arc_infra.metrics`` filters ``vllm:`` names; SGLang labels every
series with ``model_name`` (and more), so labels are summed away here and histograms keep
their buckets for quantiles.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

LINE = re.compile(r"^(sglang:[A-Za-z0-9_:]+?)(_bucket|_count|_sum)?(\{[^}]*\})?\s+(\S+)\s*$")
LE = re.compile(r'le="([^"]+)"')

COUNTERS = {
    "accepted_generation_tokens": "sglang:generation_tokens_total",
    "prompt_tokens": "sglang:prompt_tokens_total",
    "cached_prompt_tokens": "sglang:cached_tokens_total",
    "requests": "sglang:num_requests_total",
    "aborted_requests": "sglang:num_aborted_requests_total",
    "retracted_requests": "sglang:num_retracted_requests_total",
    "retracted_input_tokens": "sglang:num_retracted_input_tokens_total",
    "evicted_tokens": "sglang:evicted_tokens_total",
}
PEAKS = {
    "peak_token_usage": "sglang:token_usage",
    "peak_used_tokens": "sglang:num_used_tokens",
    "peak_running_requests": "sglang:num_running_reqs",
    "peak_queued_requests": "sglang:num_queue_reqs",
    "peak_mamba_usage": "sglang:mamba_usage",
    "peak_retracted_requests_gauge": "sglang:num_retracted_reqs",
}
LAST = {
    "spec_accept_length": "sglang:spec_accept_length",
    "spec_accept_rate": "sglang:spec_accept_rate",
    "cache_hit_rate": "sglang:cache_hit_rate",
    "max_total_num_tokens": "sglang:max_total_num_tokens",
    "kv_cache_memory_usage_gb": "sglang:kv_cache_memory_usage_gb",
    "weight_memory_usage_gb": "sglang:weight_memory_usage_gb",
    "graph_memory_usage_gb": "sglang:graph_memory_usage_gb",
    "context_len": "sglang:context_len",
    "startup_time_seconds": "sglang:startup_time_seconds",
}
HISTOGRAMS = {
    "e2e_request_latency_seconds": "sglang:e2e_request_latency_seconds",
    "time_to_first_token_seconds": "sglang:time_to_first_token_seconds",
    "queue_time_seconds": "sglang:queue_time_seconds",
}


def _number(text: str) -> float:
    if text == "+Inf":
        return math.inf
    if text == "-Inf":
        return -math.inf
    return float(text)


def parse_sglang_metrics(text: str) -> dict:
    """``{"values": {name: summed value}, "histograms": {name: {buckets, count, sum}}}``."""
    values: dict[str, float] = {}
    histograms: dict[str, dict] = {}
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        match = LINE.match(raw)
        if not match:
            continue
        name, suffix, labels, value_text = match.groups()
        value = _number(value_text)
        if math.isnan(value):
            continue
        if suffix in ("_bucket", "_count", "_sum"):
            hist = histograms.setdefault(name, {"buckets": {}, "count": 0.0, "sum": 0.0})
            if suffix == "_bucket":
                le = LE.search(labels or "")
                if le is None:
                    continue
                bound = _number(le.group(1))
                hist["buckets"][bound] = hist["buckets"].get(bound, 0.0) + value
            elif suffix == "_count":
                hist["count"] += value
            else:
                hist["sum"] += value
        else:
            values[name] = values.get(name, 0.0) + value
    for hist in histograms.values():
        hist["buckets"] = sorted(hist["buckets"].items())
    return {"values": values, "histograms": histograms}


def histogram_delta(later: dict | None, earlier: dict | None) -> dict | None:
    if later is None:
        return None
    if earlier is None:
        return later
    earlier_buckets = dict(earlier["buckets"])
    return {
        "buckets": [(le, count - earlier_buckets.get(le, 0.0)) for le, count in later["buckets"]],
        "count": later["count"] - earlier["count"],
        "sum": later["sum"] - earlier["sum"],
    }


def histogram_quantile(hist: dict | None, q: float) -> float | None:
    """Prometheus-style linear interpolation inside the bucket holding the quantile."""
    if not hist or hist["count"] <= 0:
        return None
    target = q * hist["count"]
    previous_le, previous_count = 0.0, 0.0
    for le, count in hist["buckets"]:
        if count >= target:
            if math.isinf(le):
                return previous_le
            if count == previous_count:
                return le
            return previous_le + (le - previous_le) * (target - previous_count) / (count - previous_count)
        previous_le, previous_count = le, count
    return previous_le


def histogram_stats(hist: dict | None) -> dict | None:
    if not hist or hist["count"] <= 0:
        return None
    return {
        "count": hist["count"],
        "mean": hist["sum"] / hist["count"],
        "p50": histogram_quantile(hist, 0.5),
        "p95": histogram_quantile(hist, 0.95),
    }


class SglangMonitor:
    """Sample ``/metrics`` every ``interval`` seconds into ``telemetry.jsonl``."""

    def __init__(self, output, base_url, interval=5.0):
        self.output = Path(output)
        self.url = base_url.removesuffix("/v1").rstrip("/") + "/metrics"
        self.interval = interval
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.start_time = None
        self._stop = threading.Event()
        self._thread = None

    def scrape(self, client) -> dict | None:
        try:
            response = client.get(self.url)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - recorded, sampling continues
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return None
        parsed = parse_sglang_metrics(response.text)
        sample = {"elapsed_seconds": time.monotonic() - self.start_time, **parsed}
        self.samples.append(sample)
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "telemetry.jsonl").open("a") as file:
            file.write(json.dumps(sample) + "\n")
        (self.output / "sglang-metrics.txt").write_text(response.text)
        return sample

    def _run(self, client):
        while not self._stop.is_set():
            self.scrape(client)
            self._stop.wait(self.interval)
        self.scrape(client)

    def start(self, client=None):
        import httpx

        self.start_time = time.monotonic()
        client = client or httpx.Client(trust_env=False, timeout=10)
        self._thread = threading.Thread(target=self._run, args=(client,), daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        return summary(self.samples, errors=self.errors)


def summary(samples: list[dict], *, errors=None, cold_seconds=120.0) -> dict:
    if len(samples) < 2:
        return {"valid": False, "samples": len(samples), "errors": list(errors or [])}
    first, last = samples[0], samples[-1]
    seconds = last["elapsed_seconds"] - first["elapsed_seconds"]
    record: dict = {"valid": True, "samples": len(samples), "sampled_seconds": round(seconds, 1),
                    "errors": list(errors or [])}
    for key, name in COUNTERS.items():
        record[key] = last["values"].get(name, 0.0) - first["values"].get(name, 0.0)
    record["accepted_generation_tokens_per_second"] = (
        record["accepted_generation_tokens"] / seconds if seconds > 0 else None
    )
    # Pennyroyal updates this counter in observe_one_finished_request, not on
    # every decode step. Keep the old names for archived-report compatibility,
    # but expose the actual scope so unfinished generation is not implied.
    record["generation_counter_scope"] = "finished_requests"
    record["finished_request_generation_tokens"] = record["accepted_generation_tokens"]
    record["finished_request_generation_tokens_per_second"] = record[
        "accepted_generation_tokens_per_second"
    ]
    record["cached_share"] = (
        record["cached_prompt_tokens"] / record["prompt_tokens"] if record["prompt_tokens"] else None
    )
    for key, name in PEAKS.items():
        record[key] = max(s["values"].get(name, 0.0) for s in samples)
    for key, name in LAST.items():
        record[key] = last["values"].get(name)
    throughput = [s["values"].get("sglang:gen_throughput", 0.0) for s in samples]
    busy = [t for t in throughput if t > 0]
    record["mean_gen_throughput_busy"] = sum(busy) / len(busy) if busy else None
    for key, name in HISTOGRAMS.items():
        record[key] = histogram_stats(histogram_delta(last["histograms"].get(name), first["histograms"].get(name)))
    # Cold window (first ``cold_seconds`` of play) versus the rest, for TTFT.
    boundary = next((s for s in samples if s["elapsed_seconds"] - first["elapsed_seconds"] >= cold_seconds), None)
    ttft = HISTOGRAMS["time_to_first_token_seconds"]
    if boundary is not None and boundary is not last:
        record["ttft_cold_window"] = histogram_stats(
            histogram_delta(boundary["histograms"].get(ttft), first["histograms"].get(ttft)))
        record["ttft_warm_window"] = histogram_stats(
            histogram_delta(last["histograms"].get(ttft), boundary["histograms"].get(ttft)))
    return record
