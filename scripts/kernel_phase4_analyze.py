#!/usr/bin/env python3
"""Tables for docs/validation-20260927/kernels-phase4.md from the microbenchmark and the server smoke.

Microbenchmark (``--bench <run dir>``): b12x against the served CUTLASS MoE per token count and routing, errors
against the FP32 reference, the NvFp4 draft experts, the HC fusion against the pinned chain, the top-k-first
sampler against the pinned sort / Triton paths.
Smoke (``--smoke <run dir>``): the served layout and the six-window fit, MTP acceptance and decode tok/s per
point (decode-only, as in phase 2), preemptions, the greedy sanity answers and the overlay's load-time lines
(ARC_SK_*, ARC_DENSE_CHECK, ARC_SERVING_KERNELS_PATCH).

Usage: python scripts/kernel_phase4_analyze.py [--bench DIR] [--smoke DIR] [--json OUT]
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("phase2_analyze", HERE / "kernel_phase2_analyze.py")
phase2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase2)
TAGS = ("ARC_SK_CHECK ", "ARC_SK_B12X ", "ARC_SK_DRAFT ", "ARC_SK_HC ", "ARC_SERVING_KERNELS_PATCH ",
        "ARC_SK_DRAFT_VOCAB ", "ARC_SK_HOST_TABLE ", "ARC_SK_PLE ", "ARC_SK_EMBED ", "ARC_SERVING_KERNELS_PLE_PATCH ",
        "ARC_DENSE_CHECK ", "ARC_DENSE_KERNELS ")


def load(path):
    return json.loads(Path(path).read_text())


def bench_summary(folder):
    folder = Path(folder)
    out = {}
    for name in ("p1-moe-draft", "p2-hc-sampler"):
        path = folder / f"{name}.json"
        if path.is_file():
            out[name] = load(path)
    moe = out.get("p1-moe-draft", {})
    table = []
    for row in moe.get("moe", []):
        table.append({k: row.get(k) for k in ("m", "routing", "distinct_experts_per_call", "b12x_us", "cutlass_us",
                                              "b12x_over_cutlass", "b12x_error", "cutlass_error")})
    hc = out.get("p2-hc-sampler", {})
    hc_rows = []
    for row in hc.get("hc", []):
        fused = {k: v for k, v in row.items() if k.startswith("fused_")}
        best = min(fused, key=fused.get) if fused else None
        hc_rows.append({"m": row["m"], "pinned_chain_us": row["pinned_chain_us"], "best_fused": best,
                        "best_fused_us": fused.get(best), "fused": fused})
    errors = {}
    for row in hc.get("hc_error", []):
        entry = errors.setdefault(row["config"], {"max_rel": 0.0, "max_abs": 0.0, "min_bitwise": 1.0})
        entry["max_rel"] = max(entry["max_rel"], row["rel_error"])
        entry["max_abs"] = max(entry["max_abs"], row["max_abs"])
        entry["min_bitwise"] = min(entry["min_bitwise"], row["bitwise_equal"])
    return {"env": moe.get("env") or hc.get("env"), "moe": table, "moe_error": moe.get("moe_error"),
            "moe_compile": moe.get("moe_compile"), "draft": moe.get("draft"), "notes": moe.get("notes", []),
            "errors": moe.get("errors", []) + hc.get("errors", []), "hc": hc_rows, "hc_error": errors,
            "sampler": hc.get("sampler"), "part_seconds": {**moe.get("part_seconds", {}), **hc.get("part_seconds", {})}}


def log_lines(text):
    found = {tag.strip(): [] for tag in TAGS}
    for line in text.splitlines():
        for tag in TAGS:
            if tag in line:
                try:
                    found[tag.strip()].append(json.JSONDecoder().raw_decode(line.split(tag, 1)[1])[0])
                except ValueError:
                    found[tag.strip()].append(line.split(tag, 1)[1][:300])
    return found


def smoke_summary(folder):
    folder = Path(folder)
    benches = sorted(folder.rglob("bench.json"))
    if not benches:
        return {"error": f"no bench.json under {folder}"}
    rows, bench = phase2.bench_rows(benches[0])
    points = []
    for point in bench["points"]:
        decode = next((p for p in point.get("passes", []) if p["pass"] == "decode"), None)
        if decode:
            points.append({"kind": point["kind"], "prompt": point["prompt_tokens"], "streams": point["concurrency"],
                           "blocks_per_request": point.get("blocks_per_request"),
                           "fits_concurrently": point.get("fits_concurrently"),
                           "accepted_tokens_per_draft": decode.get("accepted_tokens_per_draft"),
                           "acceptance_rate": (decode["spec_accepted_tokens"] / decode["spec_draft_tokens"]
                                               if decode.get("spec_draft_tokens") else None),
                           "preemptions": decode.get("preemptions"), "peak_running": decode.get("peak_running_requests")})
    logs = "".join(p.read_text(errors="replace") for p in sorted(folder.rglob("vllm-openai-server.log")))
    layouts = [json.JSONDecoder().raw_decode(line.split("ARC_CACHE_LAYOUT ", 1)[1])[0]
               for line in logs.splitlines() if "ARC_CACHE_LAYOUT " in line]
    layout = layouts[-1] if layouts else {}
    kv = [int(m.replace(",", "")) for m in re.findall(r"GPU KV cache size: ([\d,]+) tokens", logs)]
    return {"decode": rows, "points": points, "focus_fit": bench.get("focus_fit"), "sanity": bench.get("sanity"),
            "preemption_free": bench.get("preemption_free"), "engine": bench.get("engine"),
            "status": bench.get("status"), "task_errors": bench.get("task_errors"),
            "layout": {k: layout.get(k) for k in ("num_blocks", "lookahead", "max_model_len", "block_size",
                                                   "planned_cache_bytes")},
            "cost_64k_window": next((r for r in layout.get("costs", []) if r["tokens"] >= 65536 + 256 + 3), None),
            "kv_cache_tokens": kv, "model_gib": re.findall(r"Model loading took ([\d.]+) GiB", logs),
            "available_kv_gib": re.findall(r"Available KV cache memory: ([\d.]+) GiB", logs),
            "startup": re.findall(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s", logs),
            "weights_load_seconds": re.findall(r"Loading weights took ([\d.]+) seconds", logs),
            "ple_offload_worker_lines": len(re.findall(r"PleOffloadWorker", logs)),
            "overlay": log_lines(logs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench")
    ap.add_argument("--smoke")
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {}
    if args.bench:
        out["microbench"] = bench_summary(args.bench)
    if args.smoke:
        out["smoke"] = smoke_summary(args.smoke)
    text = json.dumps(out, indent=1, default=str)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text[:20000])


if __name__ == "__main__":
    main()
