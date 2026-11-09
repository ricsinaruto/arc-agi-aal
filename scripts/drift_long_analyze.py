#!/usr/bin/env python3
"""Tables of the long-context drift test (29 September; docs/validation-20260927/hero-behavior.md, Test A).

    PYTHONPATH=src python scripts/drift_long_analyze.py <run dir> <prompts.json>

``<run dir>`` holds the downloaded kernel-quality trials (``dl-ref``, ``dl-hero``, ``dl-kv8``). For every pair against
the reference (its own repeat at concurrency 4 is the noise floor) and every context bin, over all positions and over
assistant positions: mean KL (the kernel gate's coarse partition), top-1 agreement, and the recorded tokens'
log-probability difference (per sequence, 95% t interval over sequences). Writes ``drift-long-summary.json`` next to
the run dir's trials.
"""

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arena.kernel_gate import compare_positions, to_dist  # noqa: E402

BINS = ((0, 8192), (8192, 16384), (16384, 24576), (24576, 32768), (32768, 1 << 20))
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}


def label(lo, hi):
    return f"{lo // 1024}K-{hi // 1024}K" if hi < 1 << 20 else f"{lo // 1024}K+"


def rows_of(result, key="teacher_forced"):
    return {row["index"]: row for row in result.get(key, {}).get("rows", []) if "positions" in row}


def lp_of(entries, token):
    return next((lp for t, lp in entries if int(t) == int(token)), None)


def compare(ref_rows, rows, spans):
    """{(bin, scope): compare_positions + per-sequence recorded-token log-prob deltas}."""
    out = {}
    for lo, hi in BINS:
        for scope in ("all", "assistant"):
            pairs, deltas = [], []
            for index, row in rows.items():
                ref = ref_rows.get(index)
                if ref is None:
                    continue
                inside = [False] * len(row["positions"])
                for a, b in spans.get(index, []):
                    for i in range(max(a - 1, 0), min(b - 1, len(inside))):
                        inside[i] = True
                d, n = 0.0, 0
                for i, (p, q, chosen) in enumerate(zip(ref["positions"], row["positions"], row["tokens"][1:])):
                    context = i + 1  # tokens seen before predicting position i + 1
                    if not lo <= context < hi or (scope == "assistant" and not inside[i]):
                        continue
                    pairs.append((to_dist(p), to_dist(q), chosen))
                    a, b = lp_of(p, chosen), lp_of(q, chosen)
                    if a is not None and b is not None:
                        d, n = d + b - a, n + 1
                if n:
                    deltas.append(d / n)
            entry = compare_positions(pairs)
            if deltas:
                half = (T975.get(len(deltas) - 1, 2.0) * statistics.stdev(deltas) / math.sqrt(len(deltas))
                        if len(deltas) > 1 else float("nan"))
                entry.update(delta_logprob_mean=statistics.mean(deltas), delta_logprob_ci95=half,
                             sequences=len(deltas))
            out[f"{label(lo, hi)}/{scope}"] = entry
    return out


def main():
    run, prompts_path = Path(sys.argv[1]), Path(sys.argv[2])
    prompts = json.loads(prompts_path.read_text())
    spans = {t["index"]: t.get("assistant_spans") or [] for t in prompts["teacher_forced"]}
    trials = {p.parent.name: json.loads(p.read_text()) for p in run.rglob("kernel-quality.json")}
    ref = rows_of(trials["dl-ref"])
    sides = {"dl-ref repeat (noise)": rows_of(trials["dl-ref"], "teacher_forced_repeat")}
    for name in ("dl-hero", "dl-kv8"):
        if name in trials:
            sides[name] = rows_of(trials[name])
    summary = {name: compare(ref, rows, spans) for name, rows in sides.items()}
    for scope in ("all", "assistant"):
        print(f"\n### {scope} positions: mean KL vs dl-ref / top-1 agreement / recorded-token Δ log-prob (95% CI)\n")
        print("| Context | positions | " + " | ".join(sides) + " |")
        print("| --- | ---: |" + " --- |" * len(sides))
        for lo, hi in BINS:
            key = f"{label(lo, hi)}/{scope}"
            first = next(iter(summary.values()))[key]
            cells = []
            for name in sides:
                e = summary[name][key]
                if not e["positions"]:
                    cells.append("–")
                    continue
                ci = (f"{e['delta_logprob_mean']:+.4f} ± {e['delta_logprob_ci95']:.4f}"
                      if "delta_logprob_mean" in e else "–")
                cells.append(f"{e['mean_kl']:.3f} / {100 * e['top1_agreement']:.1f}% / {ci}")
            print(f"| {label(lo, hi)} | {first['positions']:,} | " + " | ".join(cells) + " |")
    (run / "drift-long-summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
