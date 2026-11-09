#!/usr/bin/env python3
"""Tables for docs/validation-20260927/hero-smoke.md (the hero arm's one-start smoke).

From the run folder: the served layout and fit, the tiles' memory lines (ARC_DENSE_MID_MEMORY) and checks, greedy sanity,
drift against the recorded phase-2 A+C gate (and the self-repeat noise), decode points (decode-only tok/s, ms per step)
against the MTP-0 kv8 stack smoke, and a per-iteration breakdown of the torch profiles at 1 and 14 streams.

Breakdown: each decode iteration (one MTP-0 forward + sampling) is cut between main-stream ``execute_context``
annotations. Kernel time is attributed to categories on the main stream (side-stream time is listed apart: it overlaps
the routed experts). GPU idle is split into the PLE wait (the gap before the first kernel after the PLE layer's
``_hc_combine_kernel``, where the graph waits on the CPU PLE semaphore), the host-preparation span before the first
graph kernel, and the rest (launch spacing).

Usage: python scripts/hero_analyze.py RUN_DIR [--mtp0 STACK_RUN_DIR] [--json OUT]
"""

import argparse
import collections
import gzip
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("phase5_analyze", HERE / "kernel_phase5_analyze.py")
phase5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase5)

# (category, regex) in priority order; the first match wins.
CATEGORIES = [
    ("experts (Marlin W4A16)", r"[Mm]arlin|moe_wna16"),
    ("experts (CUTLASS NvFp4)", r"BlockScaled|GemmUniversal.*GroupProblemShape"),
    ("MoE glue (routing, align, quant, sum)", r"topkGating|moe_align|align_block|count_and_sort|moe_sum|expandInputRows|"
                                              r"doActivation|computeStrides|ExpertPrefixSum|cvt_fp16_to_fp4|"
                                              r"finalizeMoeRouting|silu_and_mul|act_and_mul|topk_softmax|moe_"),
    ("dense GEMM (INT8 / tiles / skinny / cuBLAS)", r"_w8a16_kernel|_w8a16_fused_kernel|_splitk_reduce_kernel|CuteSkinnyGemm|skinny|cutlass_80|"
                                                   r"gemvx|splitKreduce|gemm|Gemm|sm80_xmma|_dequant_kernel"),
    ("QSA attention (int8 KV)", r"_qsa_|persistent_topk|reshape_and_cache|_fused_qk_rmsnorm_rope_gate|_expand_qsa_indices|"
                                r"_build_qsa_metadata|_indexer|kv8"),
    ("GDN (conv, recurrence, norm)", r"gdn|gated_delta|causal_conv1d|chunk_|layer_norm_fwd|fused_recurrent|mamba"),
    ("HC (combine, norm, silu, gate-mix)", r"^_hc_|grouped_gemma_rmsnorm"),
    ("sampler", r"SoftMax|softmax|scan_innermost|DeviceScan|RadixSort|sort|_temperature|_gumbel|topk_topp|topk|argmax|"
                r"ArgMax|_min_p|_penalt|_bias_kernel|logits"),
    ("PLE", r"(^|_)ple(_|$)|short_conv|ngram_"),
    ("elementwise / copies / other", r"."),
]
CATEGORIES = [(n, re.compile(rx)) for n, rx in CATEGORIES]
DECODE = re.compile(r"execute_context_0\(0\)_generation_\d+\(\d+\)$")


def category(name):
    return next(cat for cat, rx in CATEGORIES if rx.search(name))


def breakdown(path):
    events = json.load(gzip.open(path))["traceEvents"]
    gpu = sorted((e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    main = collections.Counter(e["args"]["stream"] for e in kernels).most_common(1)[0][0]
    ann = sorted((e for e in events if e.get("cat") == "gpu_user_annotation" and e.get("tid") == main),
                 key=lambda e: e["ts"])
    wins = [(a["ts"], b["ts"]) for a, b in zip(ann, ann[1:]) if DECODE.match(a["name"]) and DECODE.match(b["name"])]
    corr = {e["args"]["correlation"]: e for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    main_ms = collections.Counter()
    side_ms = collections.Counter()
    names = collections.Counter()
    calls = collections.Counter()
    idle = collections.Counter()
    n = len(wins)
    for t0, t1 in wins:
        inside = [e for e in gpu if t0 <= e["ts"] < t1]
        first_graph = next((e for e in inside if (corr.get(e.get("args", {}).get("correlation")) or {}).get("name")
                            == "cudaGraphLaunch"), None)
        prep_end = first_graph["ts"] if first_graph else t0
        for e in inside:
            if e["cat"] != "kernel":
                cat = "elementwise / copies / other"
            else:
                cat = category(e["name"])
            if e.get("args", {}).get("stream") == main:
                main_ms[cat] += e["dur"]
                names[(cat, e["name"][:90])] += e["dur"]
                calls[(cat, e["name"][:90])] += 1
            else:
                side_ms[cat] += e["dur"]
        merged = []
        for e in inside:
            a, b = e["ts"], min(e["ts"] + e["dur"], t1)
            if merged and a <= merged[-1][1]:
                if b > merged[-1][1]:
                    merged[-1][1], merged[-1][3] = b, e
            else:
                merged.append([a, b, e, e])
        idle["host prep before the graph"] += max(0.0, (merged[0][0] if merged else t1) - t0)
        for (a0, b0, f0, l0), (a1, b1, f1, l1) in zip(merged, merged[1:]):
            gap = a1 - b0
            if l0["name"].startswith("_hc_combine_kernel") and gap > 20 and a1 > prep_end:
                idle["PLE wait"] += gap
            elif a1 <= prep_end:
                idle["host prep before the graph"] += gap
            else:
                idle["launch gaps / other idle"] += gap
    period = sum(b - a for a, b in wins) / n / 1000 if n else None
    out = {"iterations": n, "period_ms": period,
           "main_stream_ms": {k: v / n / 1000 for k, v in main_ms.most_common()},
           "side_stream_ms": {k: v / n / 1000 for k, v in side_ms.most_common()},
           "idle_ms": {k: v / n / 1000 for k, v in idle.most_common()},
           "top_kernels": [{"category": c, "name": k, "us_per_iter": v / n, "calls_per_iter": calls[(c, k)] / n}
                           for (c, k), v in names.most_common(30)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mtp0", default="runs/modal-validation/arena-kernel-stack-phase5-v1-modal-20260928")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    trial = Path(args.run) / "arena/trials/hero"
    hero = phase5.arm(trial)
    bench = json.loads((trial / "bench.json").read_text())
    logs = (trial / "vllm-openai-server.log").read_text(errors="replace") if (trial / "vllm-openai-server.log").is_file() else ""
    memory = [json.JSONDecoder().raw_decode(line.split("ARC_DENSE_MID_MEMORY ", 1)[1])[0]
              for line in logs.splitlines() if "ARC_DENSE_MID_MEMORY " in line]
    reference = phase5.arm(Path(args.mtp0) / "arena/trials/stack-mtp0-kv8")
    result = {"run": str(args.run), "points": hero["points"], "sanity": hero["sanity"],
              "preemption_free": hero["preemption_free"], "focus_fit": bench.get("focus_fit"),
              "layout": bench.get("layout"), "capacity_blocks": bench.get("capacity_blocks"),
              "kv_cache_tokens": hero["kv_cache_tokens"], "model_gib": hero["model_gib"],
              "startup_s": hero["startup_s"], "mid_memory": memory, "quality": bench.get("quality"),
              "vs_mtp0_kv8_stack": phase5.compare(hero, reference), "task_errors": hero["task_errors"],
              "moe_backend_lines": [line[:300] for line in logs.splitlines()
                                    if re.search(r"[Mm]arlin|moe.backend|MoE backend", line)][:12]}
    result["profiles"] = {}
    for sub in sorted(trial.glob("profile-c*")):
        traces = sorted(sub.glob("*.pt.trace.json.gz"))
        if traces:
            result["profiles"][sub.name] = breakdown(traces[0])
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str) + "\n")
    for p in result["points"]:
        print(p)
    for row in result["vs_mtp0_kv8_stack"]:
        print(row)
    for name, prof in result["profiles"].items():
        print("==", name, prof["iterations"], "iterations, period", prof["period_ms"])
        for k, v in prof["main_stream_ms"].items():
            print(f"   main {v:7.3f} ms  {k}")
        for k, v in prof["side_stream_ms"].items():
            print(f"   side {v:7.3f} ms  {k}")
        for k, v in prof["idle_ms"].items():
            print(f"   idle {v:7.3f} ms  {k}")


if __name__ == "__main__":
    main()
