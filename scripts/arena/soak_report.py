#!/usr/bin/env python3
"""Serving report of a short arena soak (Holo4-27B soak, 29 September; docs/validation-20260927/holo4-27b.md).

    python scripts/arena/soak_report.py <run dir> [--trial duck-transfer] [--bucket 60]

``<run dir>`` is a downloaded Modal validation run (``.../arena``). From the trial's own files:

- the vLLM log's 10-second throughput lines (prompt / generation tok/s, running, waiting, KV use, prefix hit rate),
  bucketed per ``--bucket`` seconds of arena wall time, with the per-stream generation rate (generation / running) and
  the opening / phase-2 split at the recorded switch;
- ``requests.json``: prompt, cached and completion tokens per request (the prefill / decode split, the exact prefix
  hit rate with ARENA_PROMPT_TOKENS_DETAILS) and the context-length distribution by phase;
- preemptions (the engine summary and the metrics text), stall dumps and engine errors in the server log;
- the compaction preflight, levels and score.

Writes ``soak-report.json`` next to the trial and prints the tables.
"""

import argparse
import datetime
import json
import re
import statistics
from pathlib import Path

LINE = re.compile(r"INFO (\d\d-\d\d) (\d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine 000: Avg prompt throughput: ([\d.]+) "
                  r"tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
                  r"GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")
STALL = re.compile(r"ARC_STALL_DUMP|Current thread 0x|Thread 0x[0-9a-f]+ \(most recent call first\)")
ERROR = re.compile(r"\bERROR\b|Traceback|EngineDeadError|CUDA error|out of memory")
BINS = [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32768), (32768, 49152), (49152, 65536), (65536, 10**9)]


def throughput_lines(log, year):
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        stamp = datetime.datetime.strptime(f"{year}-{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=datetime.UTC).timestamp()
        rows.append({"epoch": stamp, "prompt": float(m.group(3)), "gen": float(m.group(4)), "running": int(m.group(5)),
                     "waiting": int(m.group(6)), "kv": float(m.group(7)), "prefix": float(m.group(8))})
    return rows


def percentiles(values, points=(10, 25, 50, 75, 90, 99)):
    if not values:
        return {}
    ordered = sorted(values)
    return {f"p{p}": ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))] for p in points} | {
        "mean": round(statistics.mean(ordered), 1), "max": ordered[-1], "n": len(ordered)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--trial", default="duck-transfer")
    parser.add_argument("--bucket", type=int, default=60)
    args = parser.parse_args()
    run = Path(args.run)
    trial = next(p for p in run.rglob(f"trials/{args.trial}") if p.is_dir())
    admissions = [json.loads(x) for x in (trial / "admission.jsonl").read_text().splitlines() if x.strip()]
    start = min(a["epoch"] - a["wall_seconds"] for a in admissions)
    switch = next((a["wall_seconds"] for a in admissions if a.get("start_phase") == "steady"), None)
    arena = json.loads((trial / "arena.json").read_text())
    wall = float(arena.get("wall_seconds") or 0)
    year = datetime.datetime.fromtimestamp(start, datetime.UTC).year
    rows = [dict(r, t=r["epoch"] - start) for r in throughput_lines(trial / "vllm-openai-server.log", year)]
    rows = [r for r in rows if -30 <= r["t"] <= wall + 300]
    buckets = []
    for lo in range(0, int(max((r["t"] for r in rows), default=0)) + 1, args.bucket):
        part = [r for r in rows if lo <= r["t"] < lo + args.bucket]
        if not part:
            continue
        gen = statistics.mean(r["gen"] for r in part)
        running = statistics.mean(r["running"] for r in part)
        buckets.append({"t": lo, "gen_tok_s": round(gen, 1), "prompt_tok_s": round(statistics.mean(r["prompt"] for r in part), 1),
                        "running": round(running, 1), "per_stream_tok_s": round(gen / running, 1) if running else None,
                        "waiting": round(statistics.mean(r["waiting"] for r in part), 1),
                        "kv_pct": round(max(r["kv"] for r in part), 1), "prefix_hit_pct": part[-1]["prefix"]})

    def span(lo, hi):
        part = [r for r in rows if lo <= r["t"] < hi]
        if not part:
            return None
        gen = statistics.mean(r["gen"] for r in part)
        running = statistics.mean(r["running"] for r in part)
        return {"seconds": [lo, hi], "gen_tok_s": round(gen, 1), "prompt_tok_s": round(statistics.mean(r["prompt"] for r in part), 1),
                "running": round(running, 1), "per_stream_tok_s": round(gen / running, 1) if running else None,
                "kv_pct_max": max(r["kv"] for r in part), "waiting_max": max(r["waiting"] for r in part)}

    cut = switch if switch is not None else wall / 2
    phases = {"opening": span(0, cut), "phase2": span(cut, wall), "phase2_after_60s": span(cut + 60, wall),
              "whole": span(0, wall)}
    requests = json.loads((trial / "requests.json").read_text())
    ok = [r for r in requests if r.get("status") == 200 and r.get("usage")]
    prompt = [r["usage"]["prompt_tokens"] for r in ok]
    cached = [((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0) for r in ok]
    completion = [r["usage"]["completion_tokens"] for r in ok]
    split = {"requests": len(requests), "ok": len(ok), "prompt_tokens": sum(prompt), "cached_prompt_tokens": sum(cached),
             "computed_prompt_tokens": sum(prompt) - sum(cached), "completion_tokens": sum(completion),
             "exact_prefix_hit_rate": round(sum(cached) / sum(prompt), 4) if sum(prompt) else None,
             "cached_tokens_reported": any((r["usage"].get("prompt_tokens_details") or {}) for r in ok)}
    # Requests in start order (started_monotonic): the first and last thirds show how contexts grow over the soak.
    ordered = [r["usage"]["prompt_tokens"] for r in sorted(ok, key=lambda r: r.get("started_monotonic") or 0)]
    third = max(1, len(ordered) // 3)
    contexts = {"all": percentiles(prompt), "first_third": percentiles(ordered[:third]),
                "last_third": percentiles(ordered[-third:]), "bins": {}}
    for lo, hi in BINS:
        n = sum(lo <= p < hi for p in prompt)
        contexts["bins"][f"{lo // 1024}K-{hi // 1024 if hi < 10**9 else 'max'}K"] = n
    server_log = (trial / "vllm-openai-server.log").read_text(errors="replace").splitlines()
    stalls = [x for x in server_log if STALL.search(x) and "stall_dump_seconds" not in x]
    errors = [x[:300] for x in server_log if ERROR.search(x) and "min_frames" not in x and "max_frames" not in x]
    metrics = (trial / "vllm-metrics.txt").read_text(errors="replace") if (trial / "vllm-metrics.txt").is_file() else ""
    preempt = re.findall(r"^vllm:num_preemptions_total\{[^}]*\} ([\d.e+]+)", metrics, re.M)
    preflight = (json.loads((trial / "compaction-preflight.json").read_text())
                 if (trial / "compaction-preflight.json").is_file() else None)
    report = {
        "trial": str(trial), "wall_seconds": wall, "switch_wall_seconds": switch, "buckets": buckets, "phases": phases,
        "tokens": split, "contexts": contexts,
        "preemptions": {"engine": (arena.get("engine") or {}).get("preemptions"),
                        "metrics": [float(x) for x in preempt], "preemption_free": arena.get("preemption_free")},
        "stall_dump_lines": stalls[:20], "engine_errors": errors[:30], "task_errors": arena.get("task_errors"),
        "compaction_preflight": preflight and {k: preflight.get(k) for k in ("ok", "mode", "reason")},
        "levels": arena.get("levels"), "mean_score": arena.get("mean_score"), "actions": arena.get("actions"),
        "completed_replies": arena.get("completed_replies"), "completed_reply_tokens": arena.get("completed_reply_tokens"),
        "engine": arena.get("engine"),
    }
    (trial / "soak-report.json").write_text(json.dumps(report, indent=1))
    print("| t (s) | gen tok/s | per stream | running | waiting | prompt tok/s | KV % (max) | prefix hit % |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in buckets:
        print(f"| {b['t']} | {b['gen_tok_s']} | {b['per_stream_tok_s']} | {b['running']} | {b['waiting']} | "
              f"{b['prompt_tok_s']} | {b['kv_pct']} | {b['prefix_hit_pct']} |")
    print(json.dumps({k: report[k] for k in ("switch_wall_seconds", "phases", "tokens", "contexts", "preemptions",
                                              "compaction_preflight", "levels", "mean_score", "actions",
                                              "completed_replies")}, indent=1))
    print("stall lines", len(stalls), "error lines", len(errors), "task errors", arena.get("task_errors"))
    for line in errors[:10]:
        print("  ", line)


if __name__ == "__main__":
    main()
