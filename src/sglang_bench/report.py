"""Summarise a benchmark ``study.json`` as a table."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(record: dict) -> list[dict]:
    rows = []
    for point in record.get("points", []):
        metrics = point.get("metrics") or {}
        startup = point.get("startup") or {}
        e2e = metrics.get("e2e_request_latency_seconds") or {}
        cold = metrics.get("ttft_cold_window") or {}
        warm = metrics.get("ttft_warm_window") or {}
        games = point.get("public_rows") or []
        rows.append({
            "point": point.get("point"),
            "status": point.get("status"),
            "admitted": (point.get("profile") or {}).get("admitted"),
            "fitted_tokens": startup.get("max_total_num_tokens"),
            "mamba": startup.get("max_mamba_cache_size"),
            "boot_seconds": (point.get("timings") or {}).get("boot_seconds"),
            "tokens_per_second": metrics.get("accepted_generation_tokens_per_second"),
            "generated_tokens": metrics.get("accepted_generation_tokens"),
            "cached_share": metrics.get("cached_share"),
            "cache_hit_rate": metrics.get("cache_hit_rate"),
            "p95_e2e": e2e.get("p95"),
            "p95_ttft_cold": cold.get("p95"),
            "p95_ttft_warm": warm.get("p95"),
            "retracted": metrics.get("retracted_requests"),
            "evicted_tokens": metrics.get("evicted_tokens"),
            "spec_accept_length": metrics.get("spec_accept_length"),
            "mean_score": (sum(g.get("final_score") or 0 for g in games) / len(games)) if games else None,
            "levels": sum(g.get("levels_completed") or 0 for g in games),
            "actions": sum(g.get("actions") or 0 for g in games),
            "games": games,
            "error": point.get("error"),
        })
    return rows


def render(record: dict) -> str:
    rows = summarize(record)
    lines = [
        "| Point | Status | Admitted | Fitted pool | Mamba | Boot s | Gen tok/s | Cached share | Hit rate | p95 e2e s | p95 TTFT cold/warm | Retracted | Evicted | Accept len | Mean score | Levels | Actions |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['point']} | {r['status']} | {r['admitted']} | {_fmt(r['fitted_tokens'])} | {_fmt(r['mamba'])} | "
            f"{_fmt(r['boot_seconds'])} | {_fmt(r['tokens_per_second'])} | {_fmt(r['cached_share'], 3)} | "
            f"{_fmt(r['cache_hit_rate'], 3)} | {_fmt(r['p95_e2e'])} | {_fmt(r['p95_ttft_cold'])} / {_fmt(r['p95_ttft_warm'])} | "
            f"{_fmt(r['retracted'], 0)} | {_fmt(r['evicted_tokens'], 0)} | {_fmt(r['spec_accept_length'], 2)} | "
            f"{_fmt(r['mean_score'], 2)} | {r['levels']} | {r['actions']} |"
        )
    for r in rows:
        for g in r["games"]:
            lines.append(f"- {r['point']} {g.get('game_id')}: score {_fmt(g.get('final_score'), 2)}, levels "
                         f"{g.get('levels_completed')}, actions {g.get('actions')}, generated {g.get('generated_tokens')}")
        if r["error"]:
            lines.append(f"- {r['point']} error: {r['error']}")
    return "\n".join(lines)


def main(argv=None):
    path = Path((argv or sys.argv[1:])[0])
    print(render(json.loads(path.read_text())))


if __name__ == "__main__":
    main()
