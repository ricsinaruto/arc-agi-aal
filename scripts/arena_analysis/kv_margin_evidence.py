"""Evidence for the gate admission margin (0.90 -> 0.96 for the next three-block build; CPU).

For each recorded trial: the served pool (cache-layout.json), the gate capacity (margin x pool), the gate's peak
reservation, the engine's peak KV use (``vllm:kv_cache_usage_perc`` in telemetry.jsonl, x the pool = blocks), the
preemption counter, the gap between reservations and real use at the same moments (progress rows: gate reserved vs
the engine's KV fraction), and any allocation-failure / preemption lines in the vLLM server log. Then what a 0.96 gate
(or another margin) would leave between the gate cap and the pool.

    uv run python scripts/arena_analysis/kv_margin_evidence.py NAME=<trial dir> ... [--margins 0.9 0.96] [--out f]
"""

import argparse
import json
import re
from pathlib import Path

LOG_PATTERNS = re.compile(r"preempt|Failed to allocate|Cannot allocate|out of (?:kv )?blocks|No available (?!shared memory)|"
                          r"KV cache is full|num_free_blocks", re.I)


def evidence(trial):
    trial = Path(trial)
    layout = json.loads((trial / "cache-layout.json").read_text())
    pool = int(layout["num_blocks"])
    out = {"pool_blocks": pool}
    arena = trial / "arena.json"
    if arena.exists():
        report = json.loads(arena.read_text())
        gate = report["gate"]
        out.update(gate_capacity=gate["capacity"], margin=round(gate["capacity"] / pool, 4),
                   peak_reserved=gate["peak_reserved"], preemptions=report["engine"].get("preemptions"),
                   engine_peak_cache_fraction=report["engine"].get("peak_cache_fraction"))
    peak, peak_at = 0.0, None
    tel = trial / "telemetry.jsonl"
    if tel.exists():
        for line in tel.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = (row.get("metrics") or {}).get("vllm:kv_cache_usage_perc")
            if value is not None and value > peak:
                peak, peak_at = value, row.get("elapsed_seconds")
            pre = (row.get("metrics") or {}).get("vllm:num_preemptions_total")
            if pre is not None:
                out["telemetry_preemptions_total"] = pre
    out.update(telemetry_peak_kv_fraction=round(peak, 4), telemetry_peak_kv_blocks=round(peak * pool),
               telemetry_peak_at_seconds=peak_at)
    # Reserved vs used at the same moments (progress rows carry both).
    ratios, used_minus = [], []
    prog = trial / "progress.jsonl"
    if prog.exists():
        for line in prog.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            eng = ((row.get("live") or {}).get("engine") or {})
            frac, res = eng.get("kv_cache_fraction"), (row.get("gate") or {}).get("reserved")
            if frac is not None and res:
                ratios.append(frac * pool / res)
                used_minus.append(frac * pool - res)
    if ratios:
        out.update(progress_rows=len(ratios), used_over_reserved_max=round(max(ratios), 3),
                   used_over_reserved_mean=round(sum(ratios) / len(ratios), 3),
                   used_minus_reserved_max_blocks=round(max(used_minus), 1))
    log = trial / "vllm-openai-server.log"
    if log.exists():
        hits = [line.strip()[:200] for line in log.read_text(errors="replace").splitlines() if LOG_PATTERNS.search(line)]
        out.update(server_log_matches=len(hits), server_log_examples=hits[:5])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="NAME=trial dir")
    parser.add_argument("--margins", type=float, nargs="+", default=[0.90, 0.96])
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        row = evidence(path)
        pool = row["pool_blocks"]
        row["margins"] = {f"{m:.2f}": {"gate_capacity": int(m * pool), "headroom_blocks": pool - int(m * pool),
                                       "headroom_over_peak_use": pool - int(m * pool) - 0}
                          for m in args.margins}
        result[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "server_log_examples"}))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
