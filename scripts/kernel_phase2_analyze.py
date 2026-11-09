#!/usr/bin/env python3
"""Tables for docs/validation-20260927/kernels-phase2.md from a kernel-gate run's trial folders.

Quality: the gate's verdict against the reference, and the control's own noise (reference vs its repeat at
another concurrency), per measurement; teacher-forced KL/agreement per passage. Throughput: the short grid
of both servers (decode-only tok/s per stream) and the profiled decode point (dense-GEMM time per forward).
Load-time kernel self-checks (ARC_DENSE_CHECK) and the served KV pool come from the server logs.

Usage: python scripts/kernel_phase2_analyze.py <run>/arena/trials [--json OUT]
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arena import kernel_gate as gate  # noqa: E402

DENSE = re.compile(r"cutlass_80|gemvx|splitKreduce|_w8a16_kernel|_splitk_reduce_kernel|_dequant_kernel|"
                   r"CuteSkinnyGemm|skinny|cublas|gemm_bf16|Kernel2", re.I)
MOE = re.compile(r"BlockScaled|GroupedGemm|Moe|moe|expert", re.I)


def load(path):
    return json.loads(Path(path).read_text())


def per_passage(reference, candidate):
    rows = []
    other = {r["index"]: r for r in candidate["rows"] if "positions" in r}
    for row in reference["rows"]:
        if "positions" in row and row["index"] in other:
            one = gate.compare_teacher({"rows": [row]}, {"rows": [other[row["index"]]]})
            rows.append({"index": row["index"], **{k: one[k] for k in ("positions", "mean_kl", "top1_agreement")}})
    return rows


def nll_shift(reference, candidate, draws=4000):
    """Mean NLL of the recorded next token per passage (both servers), and the paired shift with a bootstrap
    95% interval over passages: a systematic quality loss raises NLL, numerical noise does not."""
    import random

    def per_passage(tf):
        out = {}
        for row in tf["rows"]:
            if "positions" in row:
                out[row["index"]] = [dict((int(t), lp) for t, lp in pos).get(tok)
                                     for pos, tok in zip(row["positions"], row["tokens"][1:])]
        return out

    a, b = per_passage(reference), per_passage(candidate)
    pairs = []
    for index in a.keys() & b.keys():
        both = [(-x, -y) for x, y in zip(a[index], b[index]) if x is not None and y is not None]
        pairs.append((sum(x for x, _ in both) / len(both), sum(y for _, y in both) / len(both)))
    diffs = [y - x for x, y in pairs]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(draws))
    return {"passages": len(pairs), "reference_nll": sum(x for x, _ in pairs) / len(pairs),
            "candidate_nll": sum(y for _, y in pairs) / len(pairs), "shift": sum(diffs) / len(diffs),
            "shift_ci95": [boots[int(0.025 * draws)], boots[int(0.975 * draws)]]}


def bench_rows(path):
    bench = load(path)
    out = []
    for point in bench["points"]:
        passes = {p["pass"]: p for p in point.get("passes", [])}
        prefill, decode = passes.get("prefill"), passes.get("decode")
        if not decode:
            continue
        # Decode-only estimate as in the serving lab: subtract the pass's computed prompt tokens at the prefill rate.
        seconds = decode["seconds"]
        if prefill and prefill.get("prompt_tokens_per_second"):
            seconds -= decode["computed_prompt_tokens"] / prefill["prompt_tokens_per_second"]
        out.append({"prompt": point["prompt_tokens"], "streams": point["concurrency"],
                    "running": decode.get("peak_running_requests"),
                    "prefill_tok_s": prefill and prefill["prompt_tokens_per_second"],
                    "decode_raw_tok_s": decode["completion_tokens_per_second"],
                    "decode_only_tok_s": round(decode["completion_tokens"] / seconds, 1),
                    "per_stream_tok_s": round(decode["completion_tokens"] / seconds / point["concurrency"], 1)})
    return out, bench


def profile_dense(folder):
    """Dense-GEMM kernel time per decode iteration (one MTP-0 forward) in the profiled 1-stream window."""
    traces = sorted(Path(folder).glob("profile-c1/*.json.gz"))
    if not traces:
        return None
    with gzip.open(traces[0], "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    # One MTP-0 decode iteration runs the GDN recurrent decode kernel once per GDN layer (36 of 48 layers).
    recurrent = sum(1 for e in kernels if e["name"].startswith("fused_recurrent_gated_delta_rule"))
    forwards = [None] * round(recurrent / 36)
    names = {}
    for e in kernels:
        family = "dense" if DENSE.search(e["name"]) and not MOE.search(e["name"]) else "other"
        entry = names.setdefault(e["name"][:140], {"us": 0.0, "count": 0, "family": family})
        entry["us"] += e["dur"]
        entry["count"] += 1
    dense_us = sum(v["us"] for v in names.values() if v["family"] == "dense")
    total_us = sum(v["us"] for v in names.values())
    window = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) if kernels else 0
    steps = len(forwards) or None
    top = sorted(names.items(), key=lambda kv: -kv[1]["us"])[:25]
    return {"forwards": steps, "window_ms": round(window / 1000, 2),
            "ms_per_forward": round(window / 1000 / steps, 3) if steps else None,
            "dense_ms_per_forward": round(dense_us / 1000 / steps, 3) if steps else None,
            "kernel_ms_per_forward": round(total_us / 1000 / steps, 3) if steps else None,
            "top": [{"name": n, "ms_per_forward": round(v["us"] / 1000 / steps, 3) if steps else None,
                     "calls_per_forward": round(v["count"] / steps, 1) if steps else None, "family": v["family"]}
                    for n, v in top]}


def server_facts(log):
    text = Path(log).read_text(errors="replace") if Path(log).is_file() else ""
    facts = {"kv_tokens": [int(m.replace(",", "")) for m in re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)],
             "model_gib": re.findall(r"Model loading took ([\d.]+) GiB", text),
             "checks": [], "kernels": [], "int8_totals": {}}
    for line in text.splitlines():
        for tag, key in (("ARC_DENSE_CHECK ", "checks"), ("ARC_DENSE_KERNELS ", "kernels")):
            if tag in line:
                facts[key].append(json.JSONDecoder().raw_decode(line.split(tag, 1)[1])[0])
        if "ARC_DENSE_INT8 " in line:
            row = json.JSONDecoder().raw_decode(line.split("ARC_DENSE_INT8 ", 1)[1])[0]
            facts["int8_totals"][row["family"]] = row["totals"]
    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    ref = load(args.trials / "kernel-ref/kernel-quality.json")
    cand = load(args.trials / "kernel-gate/kernel-quality.json")
    verdict = gate.verdict(ref, cand)
    out = {"verdict": {k: verdict[k] for k in ("passed", "checks", "thresholds", "distribution_source")},
           "teacher_forced": {"candidate": verdict["teacher_forced"], "control_noise": verdict["control_noise"]["teacher_forced"]},
           "greedy": {"candidate": {k: v for k, v in verdict["greedy"].items() if k != "first_divergence"},
                      "candidate_first_divergence": verdict["greedy"]["first_divergence"],
                      "control_noise": {k: v for k, v in verdict["control_noise"]["greedy"].items()
                                        if k != "first_divergence"},
                      "control_noise_first_divergence": verdict["control_noise"]["greedy"]["first_divergence"]},
           "gsm8k": {"reference": {k: ref["gsm8k"].get(k) for k in ("accuracy", "scored", "completion_tokens", "seconds")},
                     "reference_repeat": {k: ref["gsm8k_repeat"].get(k) for k in ("accuracy", "scored", "completion_tokens",
                                                                                  "seconds")},
                     "control_noise": verdict["control_noise"]["gsm8k"], "candidate": verdict["gsm8k"]},
           "nll": {"candidate": nll_shift(ref["teacher_forced"], cand["teacher_forced"]),
                   "control_noise": nll_shift(ref["teacher_forced"], ref["teacher_forced_repeat"])},
           "per_passage": {"candidate": per_passage(ref["teacher_forced"], cand["teacher_forced"]),
                           "control_noise": per_passage(ref["teacher_forced"], ref["teacher_forced_repeat"])}}
    for name in ("kernel-ref", "kernel-gate"):
        folder = args.trials / name
        rows, bench = bench_rows(folder / "bench/bench.json")
        out[name] = {"bench": rows, "kv_cache_tokens": bench.get("kv_cache_tokens"),
                     "profile": profile_dense(folder / "bench"),
                     "server": server_facts(folder / "vllm-openai-server.log")}
    text = json.dumps(out, indent=1)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
