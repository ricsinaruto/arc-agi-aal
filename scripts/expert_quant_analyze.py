#!/usr/bin/env python3
"""Tables for docs/validation-20260927/expert-quant.md from the Modal evidence (28 September).

    uv run python scripts/expert_quant_analyze.py bench <run dir>            # stage 1 (bench.json)
    uv run python scripts/expert_quant_analyze.py gptq <run dir>             # stage 2 (gptq.json)
    uv run python scripts/expert_quant_analyze.py quality <run dir> <prompts.json>   # end-to-end (kernel-quality)

``<run dir>`` is the downloaded ``validation/<run_id>`` folder. Prints markdown and writes a JSON summary next to it.
"""

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def find(run, name):
    hits = sorted(Path(run).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} under {run}")
    return hits[0]


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def bench(run):
    data = json.loads(find(run, "bench.json").read_text())
    rows, summary = [], {"layers": {}, "env": data.get("env"), "errors": data.get("errors")}
    print("| Layer | Weight rounding (dequant vs BF16) | Activation FP4 (served vs dequant) | Served W4A4 vs BF16 | "
          "W4A4, ARC input scale, vs BF16 | Marlin W4A16 vs BF16 | Marlin vs dequant | RTN+scale search vs BF16 | "
          "cos served / Marlin |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for key in sorted(data["layers"], key=int):
        layer = data["layers"][key]
        e = layer.get("errors") or {}

        def g(name, field="vs_bf16"):
            return (e.get(name) or {}).get(field)

        row = {"layer": int(key), "weight": g("dequant_bf16act"), "act": g("served_cutlass_w4a4", "vs_dequant"),
               "served": g("served_cutlass_w4a4"), "arc_scale": g("cutlass_w4a4_arc_input_scale"),
               "marlin": g("marlin_w4a16"), "marlin_vs_deq": g("marlin_w4a16", "vs_dequant"),
               "rtn_search": g("rtn_scale_search_bf16act"), "cos_served": g("served_cutlass_w4a4", "cos_bf16"),
               "cos_marlin": g("marlin_w4a16", "cos_bf16")}
        rows.append(row)
        summary["layers"][key] = {**row, "identity": layer.get("rtn_reproduction"),
                                  "router_identical": layer.get("router_identical"),
                                  "activation_scales": layer.get("activation_scales"), "timing": layer.get("timing"),
                                  "weight_rel_error": layer.get("weight_rel_error"), "rows": layer.get("rows"),
                                  "cutlass_m_consistency": layer.get("cutlass_m_consistency")}
        cos = f"{row['cos_served']:.4f} / {row['cos_marlin']:.4f}" if row["cos_served"] and row["cos_marlin"] else "–"
        print(f"| {key} | {pct(row['weight'])} | {pct(row['act'])} | {pct(row['served'])} | {pct(row['arc_scale'])} | "
              f"{pct(row['marlin'])} | {pct(row['marlin_vs_deq'])} | {pct(row['rtn_search'])} | {cos} |")
    means = {k: statistics.mean(r[k] for r in rows if r[k] is not None) for k in rows[0] if k != "layer"
             and any(r[k] is not None for r in rows)} if rows else {}
    summary["mean"] = means
    print("| mean | " + " | ".join(pct(means.get(k)) for k in ("weight", "act", "served", "arc_scale", "marlin",
                                                               "marlin_vs_deq", "rtn_search")) + " | |")
    print()
    print("| M | " + " | ".join(f"L{k} CUTLASS / Marlin µs" for k in sorted(data["layers"], key=int)) + " |")
    ms = sorted({t["m"] for layer in data["layers"].values() for t in layer.get("timing") or []})
    for m in ms:
        cells = []
        for key in sorted(data["layers"], key=int):
            t = next((t for t in data["layers"][key].get("timing") or [] if t["m"] == m), None)
            cells.append("–" if not t or "cutlass_us" not in t else
                         f"{t['cutlass_us']:.0f} / {t.get('marlin_us', float('nan')):.0f}")
        print(f"| {m} | " + " | ".join(cells) + " |")
    ratio = {m: statistics.mean(t["marlin_over_cutlass"] for layer in data["layers"].values()
                                for t in layer.get("timing") or [] if t["m"] == m and "marlin_over_cutlass" in t)
             for m in ms if any("marlin_over_cutlass" in t for layer in data["layers"].values()
                                for t in layer.get("timing") or [] if t["m"] == m)}
    summary["marlin_over_cutlass"] = ratio
    print("\nMarlin / CUTLASS time by M:", {m: round(r, 3) for m, r in ratio.items()})
    for key in sorted(data["layers"], key=int):
        layer = data["layers"][key]
        print(f"L{key}: identity {layer.get('rtn_reproduction')} router {layer.get('router_identical')} "
              f"scales {layer.get('activation_scales')}")
    return summary


def gptq(run):
    data = json.loads(find(run, "gptq.json").read_text())
    summary = {"layers": {}, "settings": data.get("settings"), "errors": data.get("errors"),
               "manifest": {k: v for k, v in (data.get("manifest") or {}).items() if k != "shards"}}
    keys = ("checkpoint_w4a4_cutlass", "gptq_w4a4_cutlass", "checkpoint_w4a16_marlin", "gptq_w4a16_marlin",
            "checkpoint_w4a16", "gptq_w4a16")
    print("| Layer | GPTQ experts | rows/expert median | served W4A4 | GPTQ-ARC W4A4 | Marlin W4A16 (checkpoint) | "
          "Marlin W4A16 (GPTQ-ARC) | emulated W4A16 checkpoint | emulated W4A16 GPTQ-ARC |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    cols = {k: [] for k in keys}
    for key in sorted(data["layers"], key=int):
        layer = data["layers"][key]
        e = layer.get("errors") or {}
        vals = [(e.get(k) or {}).get("vs_bf16") for k in keys]
        for k, v in zip(keys, vals):
            if v is not None:
                cols[k].append(v)
        summary["layers"][key] = {"experts_gptq": layer.get("experts_gptq"), "rows": layer.get("rows_per_expert"),
                                  **{k: v for k, v in zip(keys, vals)}}
        print(f"| {key} | {layer.get('experts_gptq')} | {(layer.get('rows_per_expert') or {}).get('median', 0):.0f} | "
              + " | ".join(pct(v) for v in vals) + " |")
    summary["mean"] = {k: statistics.mean(v) for k, v in cols.items() if v}
    print("| mean | | | " + " | ".join(pct(summary["mean"].get(k)) for k in keys) + " |")
    fallback = sum(512 - (layer.get("experts_gptq") or 0) for layer in data["layers"].values())
    summary["fallback_experts"] = fallback
    print(f"\nexperts on checkpoint tensors (too few rows): {fallback} of {512 * len(data['layers'])}")
    return summary


def token_logprob(entries, token):
    for t, lp in entries:
        if int(t) == int(token):
            return lp
    return None


def quality(run, prompts_path):
    """Teacher-forced comparison with the served reference: KL / top-1 over all and over assistant positions, and
    the recorded tokens' log-probability at assistant positions (per sequence, paired with the reference; the
    reference's own repeat at another concurrency is the noise floor)."""
    from arena.kernel_gate import compare_positions, to_dist

    prompts = json.loads(Path(prompts_path).read_text())
    spans = {t["index"]: t.get("assistant_spans") or [] for t in prompts["teacher_forced"]}
    trials = {p.parent.name: json.loads(p.read_text()) for p in Path(run).rglob("kernel-quality.json")}
    ref = trials.get("eq-ref")
    if ref is None:
        raise RuntimeError("no eq-ref trial")

    def assistant_positions(index, n):
        mask = [False] * n
        for a, b in spans.get(index, []):
            for i in range(max(a - 1, 0), min(b - 1, n)):
                mask[i] = True
        return mask

    def rows_of(result, key="teacher_forced"):
        return {row["index"]: row for row in result.get(key, {}).get("rows", []) if "positions" in row}

    def per_sequence_logprob(rows):
        out = {}
        for index, row in rows.items():
            mask = assistant_positions(index, len(row["positions"]))
            lps = [token_logprob(pos, chosen) for i, (pos, chosen) in enumerate(zip(row["positions"], row["tokens"][1:]))
                   if mask[i]]
            lps = [lp for lp in lps if lp is not None]
            out[index] = (sum(lps), len(lps))
        return out

    ref_rows = rows_of(ref)
    candidates = {"eq-ref-repeat": rows_of(ref, "teacher_forced_repeat"),
                  **{name: rows_of(result) for name, result in trials.items() if name != "eq-ref"}}
    ref_lp = per_sequence_logprob(ref_rows)
    total = sum(n for _, n in ref_lp.values())
    summary = {"eq-ref": {"assistant_mean_logprob": sum(s for s, _ in ref_lp.values()) / total,
                          "assistant_positions": total, "gsm8k_accuracy": ref.get("gsm8k", {}).get("accuracy"),
                          "gsm8k_scored": ref.get("gsm8k", {}).get("scored")}}
    print("| Server | TF KL vs ref (all) | top-1 agreement (all) | KL (assistant) | top-1 (assistant) | "
          "recorded-token mean log-prob (assistant) | Δ vs ref, 95% CI over sequences | GSM8K-200 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    print(f"| eq-ref (served) | – | – | – | – | {summary['eq-ref']['assistant_mean_logprob']:.4f} | – | "
          f"{pct(summary['eq-ref']['gsm8k_accuracy'])} |")
    for name, rows in candidates.items():
        pairs_all, pairs_asst = [], []
        for index, row in rows.items():
            if index not in ref_rows:
                continue
            mask = assistant_positions(index, len(row["positions"]))
            for i, (a, b, chosen) in enumerate(zip(ref_rows[index]["positions"], row["positions"], row["tokens"][1:])):
                pair = (to_dist(a), to_dist(b), chosen)
                pairs_all.append(pair)
                if mask[i]:
                    pairs_asst.append(pair)
        lp = per_sequence_logprob(rows)
        deltas = [(lp[i][0] - ref_lp[i][0]) / ref_lp[i][1] for i in lp if i in ref_lp and ref_lp[i][1]]
        mean_delta = statistics.mean(deltas)
        half = 2.262 * statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else float("nan")
        n = sum(c for _, c in lp.values())
        entry = {"vs_ref_all": compare_positions(pairs_all), "vs_ref_assistant": compare_positions(pairs_asst),
                 "assistant_mean_logprob": sum(s for s, _ in lp.values()) / n, "assistant_positions": n,
                 "delta_logprob_per_sequence": deltas, "delta_logprob_mean": mean_delta, "delta_logprob_ci95": half}
        if name in trials:
            verdict = trials[name].get("verdict") or {}
            entry["greedy"] = {k: (verdict.get("greedy") or {}).get(k) for k in ("exact_match", "mean_first_divergence",
                                                                                "top1_agreement", "mean_kl")}
            entry["gsm8k"] = verdict.get("gsm8k")
            gsm = trials[name].get("gsm8k", {})
            entry["gsm8k_accuracy"], entry["gsm8k_scored"] = gsm.get("accuracy"), gsm.get("scored")
        summary[name] = entry
        a, s = entry["vs_ref_all"], entry["vs_ref_assistant"]
        print(f"| {name} | {a['mean_kl']:.3f} | {pct(a['top1_agreement'])} | {s['mean_kl']:.3f} | "
              f"{pct(s['top1_agreement'])} | {entry['assistant_mean_logprob']:.4f} | "
              f"{mean_delta:+.4f} ± {half:.4f} | {pct(entry.get('gsm8k_accuracy'))} |")
    return summary


def main():
    mode, run = sys.argv[1], sys.argv[2]
    result = {"bench": bench, "gptq": gptq}.get(mode)
    summary = result(run) if result else quality(run, sys.argv[3])
    out = Path(run) / f"expert-quant-{mode}-summary.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
