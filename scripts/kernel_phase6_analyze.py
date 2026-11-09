"""Tables of the phase-6 GPU job (kernel-phase6: the hero control and the phase-6 arm; kernels-phase6.md).

Per arm: decode-only rate per point (as phase 2-5), the mixed passes (streams' tok/s with injected prefill against
the same streams' decode pass: the prefill tax, and the GPU seconds per prefilled token it implies), drift against the
recorded reference and between the arms (teacher-forced logprobs on the same token ids), the decode-step breakdown
at 14 and 21 streams (hero_analyze categories, PLE wait, host prep), the mixed step at 14 streams, and on the p6 arm
the microbenchmarks (fused split-K vs two-launch per shape and bucket, the best tile per bucket against the pinned
path, the QSA launch profiles against the pinned one, the W4 errors on the target's weights).

Run: uv run python scripts/kernel_phase6_analyze.py runs/modal-validation/<run> --json docs/validation-20260927/\
kernels-phase6-result.json
"""

import argparse
import collections
import gzip
import importlib.util
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hero = _module("hero_analyze", "hero_analyze.py")
phase5 = _module("kernel_phase5_analyze", "kernel_phase5_analyze.py")
STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")


def mixed_passes(bench):
    """Streams' tok/s in the mixed pass against their decode pass (same prompts, same server)."""
    rows = []
    for point in bench.get("points", []):
        passes = {p["pass"]: p for p in point.get("passes", [])}
        mixed, decode, prefill = passes.get("mixed"), passes.get("decode"), passes.get("prefill")
        if point.get("kind") != "mixed" or not (mixed and decode):
            continue
        streams = point["concurrency"]
        decode_seconds = decode["seconds"]
        if prefill and prefill.get("prompt_tokens_per_second"):
            decode_seconds -= decode["computed_prompt_tokens"] / prefill["prompt_tokens_per_second"]
        decode_rate = decode["completion_tokens"] / decode_seconds
        mixed_rate = mixed["completion_tokens"] / mixed["seconds"]
        prefilled = mixed["computed_prompt_tokens"]
        # The time the injections took from the streams: seconds the mixed pass lasted beyond decode at the decode
        # rate, per prefilled token.
        lost = mixed["seconds"] - mixed["completion_tokens"] / decode_rate
        rows.append({"prompt": point["prompt_tokens"], "streams": streams, "inject": mixed.get("inject"),
                     "decode_tok_s": round(decode_rate, 1), "mixed_tok_s": round(mixed_rate, 1),
                     "per_stream_decode": round(decode_rate / streams, 1),
                     "per_stream_mixed": round(mixed_rate / streams, 1),
                     "tax": round(1 - mixed_rate / decode_rate, 4),
                     "prefill_tok_s_during": round(prefilled / mixed["seconds"], 1),
                     "injected": mixed.get("injected"),
                     "seconds_per_1k_prefilled": round(1000 * lost / prefilled, 4) if prefilled else None,
                     "effective_prefill_tok_s": round(prefilled / lost) if lost > 0 else None})
    return rows


def mixed_steps(path):
    """Steps with a prefill chunk in a mixed-profile trace: duration, scheduled tokens and GPU time by category."""
    events = json.load(gzip.open(path))["traceEvents"]
    gpu = sorted((e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    if not kernels:
        return []
    main = collections.Counter(e["args"]["stream"] for e in kernels).most_common(1)[0][0]
    ann = sorted((e for e in events if e.get("cat") == "gpu_user_annotation" and e.get("tid") == main
                  and STEP.match(e["name"])), key=lambda e: e["ts"])
    out = []
    for a, b in zip(ann, ann[1:]):
        context, context_tokens, generation, generation_tokens = (int(v) for v in STEP.match(a["name"]).groups())
        if not context:
            continue
        inside = [e for e in gpu if a["ts"] <= e["ts"] < b["ts"]]
        categories = collections.Counter()
        for e in inside:
            categories[hero.category(e["name"]) if e["cat"] == "kernel" else "elementwise / copies / other"] += e["dur"]
        busy = phase5.phase3.union_length([(e["ts"], min(e["ts"] + e["dur"], b["ts"])) for e in inside])
        out.append({"name": a["name"], "context_requests": context, "context_tokens": context_tokens,
                    "decode_requests": generation, "ms": round((b["ts"] - a["ts"]) / 1000, 3),
                    "gpu_busy_ms": round(busy / 1000, 3),
                    "categories_ms": {k: round(v / 1000, 3) for k, v in categories.most_common()}})
    return out


def arm(folder):
    folder = Path(folder)
    bench = json.loads((folder / "bench.json").read_text())
    result = {"status": bench.get("status"), "errors": bench.get("task_errors"),
              "kv_cache_tokens": bench.get("kv_cache_tokens"), "capacity_blocks": bench.get("capacity_blocks"),
              "focus_fit": bench.get("focus_fit"), "points": phase5.arm_points(bench),
              "mixed": mixed_passes(bench), "sanity": bench.get("sanity"),
              "quality": bench.get("quality"), "preemption_free": bench.get("preemption_free")}
    profiles = {}
    for sub in sorted(folder.glob("profile-c*")):
        traces = sorted(sub.glob("*.pt.trace.json.gz"))
        if not traces:
            continue
        record = {}
        try:
            record["breakdown"] = hero.breakdown(traces[0])
        except Exception as exc:  # noqa: BLE001
            record["breakdown_error"] = f"{type(exc).__name__}: {exc}"
        if sub.name.endswith("-mixed"):
            record["mixed_steps"] = mixed_steps(traces[0])
        profiles[sub.name] = record
    result["profiles"] = profiles
    micro = (bench.get("phase6_microbench") or {}).get("report")
    if micro:
        result["microbench"] = microbench_tables(micro)
    return result


def microbench_tables(report):
    tables = {}
    dense = report.get("dense") or {}
    rows = []
    for entry in dense.get("rows", []):
        if "tiles" not in entry:
            continue
        timed = [t for t in entry["tiles"] if "us" in t]
        best = min(timed, key=lambda t: t.get("fused_us", t["us"]), default=None)
        fused = [t for t in timed if "fused_us" in t]
        rows.append({"kind": entry["kind"], "shape": entry["shape"], "m": entry["m"],
                     "pinned_us": entry.get("pinned_us"), "pinned_err": entry.get("pinned_err"),
                     "best_tile": best and best["tile"], "best_us": best and best.get("fused_us", best["us"]),
                     "best_err": best and best.get("err"),
                     "fused_saving_us_mean": round(sum(t["us"] - t["fused_us"] for t in fused) / len(fused), 3)
                     if fused else None,
                     "fused_all_bitwise": all(t.get("fused_bitwise") for t in fused) if fused else None})
    tables["dense"] = rows
    qsa = []
    for entry in (report.get("qsa") or {}).get("rows", []):
        if "grid" not in entry:
            continue
        best = entry.get("best")
        qsa.append({"topk": entry["topk"], "rows": entry["rows"], "pinned_profile": entry["pinned_profile"],
                    "pinned_us": entry["pinned_us"], "best": best,
                    "saving": round(1 - best["us"] / entry["pinned_us"], 4) if best else None})
    tables["qsa"] = qsa
    tables["w4"] = report.get("w4")
    return tables


def drift_between(hero_folder, p6_folder):
    """The two arms' teacher-forced top-k logprobs on the same reference token ids (kv8-quality.json): the gate's
    KL / top-1 comparison, plus the share of positions whose top-k lists are exactly equal (all bitwise-preserving
    items: expected 1.0 when the batches were composed alike)."""
    import sys

    sys.path.insert(0, str(HERE.parent / "src"))
    from arena import kernel_gate as gate

    try:
        a = json.loads((Path(hero_folder) / "kv8-quality.json").read_text())
        b = json.loads((Path(p6_folder) / "kv8-quality.json").read_text())
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    result = {"teacher_forced": gate.compare_teacher(a["teacher_forced"], b["teacher_forced"]),
              "greedy": gate.compare_greedy(a["greedy"], b["greedy"])}
    other = {r["index"]: r for r in b["teacher_forced"].get("rows", []) if "positions" in r}
    same = total = 0
    for row in a["teacher_forced"].get("rows", []):
        if "positions" in row and row["index"] in other:
            for x, y in zip(row["positions"], other[row["index"]]["positions"]):
                total += 1
                same += x == y
    result["exact_topk_share"] = round(same / total, 4) if total else None
    return result


def compare(control, candidate):
    index = {(p["kind"], p["prompt"], p["streams"]): p for p in control["points"] if "decode_only_tok_s" in p}
    rows = []
    for p in candidate["points"]:
        c = index.get((p["kind"], p["prompt"], p["streams"]))
        if c and "decode_only_tok_s" in p:
            rows.append({"kind": p["kind"], "prompt": p["prompt"], "streams": p["streams"],
                         "hero_ms": c["ms_per_iteration"], "p6_ms": p["ms_per_iteration"],
                         "hero_tok_s": c["decode_only_tok_s"], "p6_tok_s": p["decode_only_tok_s"],
                         "gain": round(p["decode_only_tok_s"] / c["decode_only_tok_s"] - 1, 4)})
    mixed = []
    by = {(m["prompt"], m["streams"]): m for m in control["mixed"]}
    for m in candidate["mixed"]:
        c = by.get((m["prompt"], m["streams"]))
        if c:
            mixed.append({"prompt": m["prompt"], "streams": m["streams"], "hero_mixed_tok_s": c["mixed_tok_s"],
                          "p6_mixed_tok_s": m["mixed_tok_s"], "hero_tax": c["tax"], "p6_tax": m["tax"],
                          "gain": round(m["mixed_tok_s"] / c["mixed_tok_s"] - 1, 4)})
    return {"decode": rows, "mixed": mixed,
            "mean_decode_gain": round(sum(r["gain"] for r in rows) / len(rows), 4) if rows else None,
            "geomean_mixed_gain": round(math.exp(sum(math.log(1 + r["gain"]) for r in mixed) / len(mixed)) - 1, 4)
            if mixed else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    trials = next(args.run.rglob("trials"))
    result = {"run": str(args.run), "arms": {}}
    for name in ("hero", "p6"):
        if (trials / name / "bench.json").is_file():
            result["arms"][name] = arm(trials / name)
    if len(result["arms"]) == 2:
        result["compare"] = compare(result["arms"]["hero"], result["arms"]["p6"])
        result["drift_between_arms"] = drift_between(trials / "hero", trials / "p6")
    logs = [p for p in args.run.rglob("*.log")]
    result["log_tags"] = {k: v for k, v in phase5.log_lines("\n".join(p.read_text(errors="replace") for p in logs)).items()
                          if k in ("ARC_DENSE_KERNELS_PATCH", "ARC_SERVING_KERNELS_PATCH")}
    fused_checks = []
    for p in logs:
        for line in p.read_text(errors="replace").splitlines():
            if "ARC_DENSE_FUSED_CHECK " in line:
                fused_checks.append(json.loads(line.split("ARC_DENSE_FUSED_CHECK ", 1)[1]))
    result["fused_checks"] = {"count": len(fused_checks),
                              "all_bitwise_equal": all(c["bitwise_equal"] for c in fused_checks) if fused_checks
                              else None}
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text + "\n")
    print(text[:20000])


if __name__ == "__main__":
    main()
