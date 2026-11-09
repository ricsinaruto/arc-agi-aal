#!/usr/bin/env python3
"""Tables and the pass bar of the 4-bit KV drift test (29 September; docs/validation-20260927/kv-cache-4bit.md).

    PYTHONPATH=src python scripts/kv4_drift_analyze.py <run dir> <prompts.json>

``<run dir>`` holds the downloaded kernel-quality trials (``kv4-ref``, ``kv4-int4``, ``kv4-k8v4``). Per context bin
(``scripts/drift_long_analyze.py``), all and assistant positions: mean KL against the reference, top-1 agreement and
the recorded tokens' log-probability change (per sequence, 95% t interval), for each candidate and for the
reference's own repeat (the noise floor). Pass bar, stated before the run, per bin: KL and top-1 disagreement at most
1.25 x the repeat's, and the NLL change not clearly positive (the log-probability interval's upper end >= 0). Also the
greedy comparison, the decode bench, each server's cache layout and the in-server ``ARC_KV4_*`` evidence (self-check,
real K/V round trips). Writes ``kv4-drift-summary.json`` into the run dir.
"""

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_long_analyze import BINS, compare, label, rows_of  # noqa: E402

REFERENCE, CANDIDATES = "kv4-ref", ("kv4-int4", "kv4-k8v4")
NOISE = "kv4-ref repeat (noise)"
FACTOR = 1.25


def verdict(entry, noise):
    """The per-bin pass bar (None when the bin has no positions)."""
    if not entry.get("positions") or not noise.get("positions"):
        return None
    kl_ok = entry["mean_kl"] <= FACTOR * noise["mean_kl"]
    top1_ok = (1 - entry["top1_agreement"]) <= FACTOR * (1 - noise["top1_agreement"])
    nll_ok = True
    if "delta_logprob_mean" in entry:
        nll_ok = entry["delta_logprob_mean"] + entry["delta_logprob_ci95"] >= 0
    return {"kl": kl_ok, "top1": top1_ok, "nll": nll_ok, "passed": kl_ok and top1_ok and nll_ok,
            "kl_ratio": entry["mean_kl"] / noise["mean_kl"] if noise["mean_kl"] else None,
            "disagreement_ratio": ((1 - entry["top1_agreement"]) / (1 - noise["top1_agreement"])
                                   if noise["top1_agreement"] < 1 else None)}


LAYOUT = re.compile(r"ARC_CACHE_LAYOUT (\{.*\})")
EVIDENCE = re.compile(r"ARC_KV4_(SELFCHECK|ROUNDTRIP|FIRST_CALL|SELFCHECK_ERROR|ROUNDTRIP_ERROR) (\{.*\})")
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def server_evidence(trial_dir):
    """Cache layout (num_blocks, block sizes) and ARC_KV4 lines from a trial's server log."""
    out = {"layout": None, "kv_cache_tokens": None, "selfcheck": [], "roundtrip": [], "first_calls": [], "errors": []}
    for path in sorted(Path(trial_dir).glob("*.log")):
        text = path.read_text(errors="replace")
        for match in LAYOUT.finditer(text):
            layout = json.loads(match.group(1))
            out["layout"] = {"num_blocks": layout.get("num_blocks"), "planned_cache_bytes": layout.get(
                "planned_cache_bytes"), "groups": [{"type": g["type"], "block_size": g["block_size"],
                                                    "page_size_bytes": g["page_size_bytes"]}
                                                   for g in layout.get("groups", [])]}
        for match in KV_TOKENS.finditer(text):
            out["kv_cache_tokens"] = int(match.group(1).replace(",", ""))
        for kind, payload in EVIDENCE.findall(text):
            record = json.loads(payload)
            key = {"SELFCHECK": "selfcheck", "ROUNDTRIP": "roundtrip", "FIRST_CALL": "first_calls"}.get(kind, "errors")
            out[key].append(record if key != "errors" else {kind: record})
    if out["roundtrip"]:
        keys = [k for k in out["roundtrip"][0] if k[:2] in ("k_", "v_")]
        out["roundtrip_mean"] = {k: statistics.mean(r[k] for r in out["roundtrip"]) for k in keys}
    return out


def bench_summary(trial_dir):
    path = Path(trial_dir) / "bench" / "bench.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    points = []
    for point in report.get("points", []):
        passes = {p["pass"]: p for p in point.get("passes", [])}
        decode = passes.get("decode", {})
        prefill = passes.get("prefill", {})
        points.append({"prompt_tokens": point.get("prompt_tokens"), "concurrency": point.get("concurrency"),
                       "decode_tok_s": decode.get("completion_tokens_per_second"),
                       "decode_tok_s_per_stream": (decode["completion_tokens_per_second"] / point["concurrency"]
                                                   if decode.get("completion_tokens_per_second") else None),
                       "prefill_tok_s": prefill.get("prompt_tokens_per_second"),
                       "preemptions": decode.get("preemptions")})
    return {"status": report.get("status"), "capacity_blocks": report.get("capacity_blocks"),
            "kv_cache_tokens": report.get("kv_cache_tokens"), "points": points}


def main():
    global REFERENCE, CANDIDATES, NOISE
    run, prompts_path = Path(sys.argv[1]), Path(sys.argv[2])
    # Other drift pairs (Holo4-27B's FP8 KV against its BF16-KV reference, 29 September):
    #   ... --reference kv-bf16-ref --candidates kv-fp8 --summary holo4-kv-drift-summary.json
    options = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    if set(options) - {"--reference", "--candidates", "--summary"}:
        raise SystemExit("options: --reference NAME --candidates A,B --summary FILE")
    REFERENCE = options.get("--reference", REFERENCE)
    CANDIDATES = tuple(options["--candidates"].split(",")) if "--candidates" in options else CANDIDATES
    NOISE = f"{REFERENCE} repeat (noise)"
    prompts = json.loads(prompts_path.read_text())
    spans = {t["index"]: t.get("assistant_spans") or [] for t in prompts["teacher_forced"]}
    paths = {p.parent.name: p for p in run.rglob("kernel-quality.json")}
    trials = {name: json.loads(p.read_text()) for name, p in paths.items()}
    ref = rows_of(trials[REFERENCE])
    sides = {NOISE: rows_of(trials[REFERENCE], "teacher_forced_repeat")}
    for name in CANDIDATES:
        if name in trials:
            sides[name] = rows_of(trials[name])
    tables = {name: compare(ref, rows, spans) for name, rows in sides.items()}
    verdicts = {name: {key: verdict(tables[name][key], tables[NOISE][key]) for key in tables[name]}
                for name in sides if name != NOISE}
    summary = {"tables": tables, "verdicts": verdicts, "factor": FACTOR, "greedy": {}, "bench": {}, "servers": {}}
    for name in (REFERENCE, *CANDIDATES):
        if name not in paths:
            continue
        gate = paths[name].parent / "kernel-gate.json"
        if gate.is_file():
            v = json.loads(gate.read_text())
            summary["greedy"][name] = {k: v["greedy"].get(k) for k in ("prompts", "exact_match", "mean_first_divergence",
                                                                       "mean_kl", "top1_agreement")}
            summary["greedy"]["noise"] = {k: v["control_noise"]["greedy"].get(k)
                                          for k in ("prompts", "exact_match", "mean_first_divergence", "mean_kl",
                                                    "top1_agreement")}
            summary.setdefault("overall", {})[name] = {"teacher_forced": v["teacher_forced"],
                                                       "noise": v["control_noise"]["teacher_forced"]}
        summary["bench"][name] = bench_summary(paths[name].parent)
        summary["servers"][name] = server_evidence(paths[name].parent)
    for scope in ("all", "assistant"):
        print(f"\n### {scope} positions: mean KL / top-1 agreement / recorded-token Δ log-prob (95% CI) [pass]\n")
        print("| Context | positions | " + " | ".join(sides) + " |")
        print("| --- | ---: |" + " --- |" * len(sides))
        for lo, hi in BINS:
            key = f"{label(lo, hi)}/{scope}"
            first = tables[NOISE][key]
            cells = []
            for name in sides:
                e = tables[name][key]
                if not e["positions"]:
                    cells.append("–")
                    continue
                ci = (f"{e['delta_logprob_mean']:+.4f} ± {e['delta_logprob_ci95']:.4f}"
                      if "delta_logprob_mean" in e else "–")
                mark = ""
                if name != NOISE and verdicts[name][key] is not None:
                    mark = " ✓" if verdicts[name][key]["passed"] else " ✗"
                cells.append(f"{e['mean_kl']:.3f} / {100 * e['top1_agreement']:.1f}% / {ci}{mark}")
            print(f"| {label(lo, hi)} | {first['positions']:,} | " + " | ".join(cells) + " |")
    print("\n" + json.dumps({k: summary[k] for k in ("greedy", "bench")}, indent=1))
    for name, evidence in summary["servers"].items():
        print(name, json.dumps({k: evidence.get(k) for k in ("layout", "kv_cache_tokens", "selfcheck",
                                                              "roundtrip_mean", "errors")}))
    (run / options.get("--summary", "kv4-drift-summary.json")).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
