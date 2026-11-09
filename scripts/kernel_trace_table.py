#!/usr/bin/env python3
"""Per-iteration dense-GEMM table from the serving-lab decode profiles (kernel work, phase 1).

Reads the torch-profiler traces of the serving lab (`profile-c1`, `profile-c8`), cuts them into
MTP-3 decode iterations (one verify forward + three MTP draft forwards) with vLLM's
`execute_context_*` GPU annotations on the main stream, and labels every dense GEMM launch by
(kernel, grid) with the model projection it belongs to. CUDA-graph kernels carry no shapes, so the
labels come from the grids (tile x grid = output features) and the layer order in the pinned model
source; a cuBLAS split-K reduce is added to the GEMM that precedes it on the same stream.

Usage: python scripts/kernel_trace_table.py [--lab DIR] [--json OUT]
"""

import argparse
import collections
import glob
import gzip
import json
import re

LAB = "runs/modal-validation/arena-serving-lab-v1-modal-20260927/arena/trials/lab-control"

# (N, K) of each dense BF16 projection (Qwen3.8-Flash-Next text config, TP=1).
SHAPES = {
    "hc_down_inject": (336, 10240), "hc_up": (10240, 320), "gdn_in_qkvz": (16384, 2560),
    "gdn_in_ba": (96, 2560), "attn_out": (2560, 6144), "qsa_qkv": (13312, 2560),
    "qsa_index_qk": (640, 2560), "router": (512, 2560), "shared_gate_up": (1280, 2560),
    "shared_down": (2560, 640), "shared_expert_gate": (1, 2560), "ple_key": (10240, 2560),
    "h2560": (2560, 2560), "lm_head": (248320, 2560),
}

# (kernel tag, grid) -> projection, for the 1-stream (verify M=4, drafts M=4/1/1) profile
C1 = {
    ("32x32_64x2b", (8, 2, 40)): "hc_down_inject", ("32x32_64x2f", (8, 2, 40)): "hc_down_inject",
    ("16x16_128x2b", (8, 80, 1)): "hc_up",  # includes the single PLE key_proj call (same kernel/grid)
    ("16x16_64x2b", (8, 128, 2)): "gdn_in_qkvz", ("16x16_128x2b", (8, 1, 1)): "gdn_in_ba",
    ("32x32_64x1f", (8, 10, 24)): "attn_out", ("32x32_128x1f", (8, 10, 16)): "attn_out",
    ("32x32_128x1f", (8, 2, 10)): "router", ("32x32_64x1f", (8, 5, 20)): "shared_gate_up",
    ("16x16_128x2b", (8, 20, 1)): "shared_down", ("gemvx8", (1, 1, 1)): "shared_expert_gate",
    ("32x32_64x1f", (8, 10, 20)): "h2560", ("16x16_64x2b", (8, 104, 2)): "qsa_qkv",
    ("16x16_128x2b", (8, 104, 1)): "qsa_qkv", ("32x32_64x1f", (8, 3, 20)): "qsa_index_qk",
    ("16x16_128x2b", (8, 1940, 1)): "lm_head", ("gemvx7", (15520, 1, 1)): "lm_head",
    ("gemvx7", (160, 1, 1)): "draft_m1_gemv", ("gemvx7", (128, 1, 1)): "router",
    ("gemvx7", (320, 1, 1)): "shared_gate_up", ("gemvx8t", (320, 1, 1)): "draft_m1_gemv",
    ("gemvx8", (320, 1, 1)): "draft_m1_gemv",
}
# 8-stream profile (verify M=32, drafts M=32/8/8)
C8 = {
    ("32x32_128x2f", (8, 2, 32)): "hc_down_inject", ("32x32_64x2b", (8, 2, 40)): "hc_down_inject",
    ("32x32_128x1b", (8, 40, 1)): "hc_up", ("16x16_128x2b", (8, 80, 1)): "hc_up",
    ("64x64_64x6", (8, 32, 1)): "gdn_in_qkvz", ("64x64_32x10", (2, 1, 40)): "gdn_in_ba",
    ("128x64_64x3a2", (8, 3, 16)): "attn_out", ("32x32_64x1f", (8, 10, 24)): "attn_out",
    ("32x32_64x1b", (8, 2, 20)): "router", ("32x32_128x1f", (8, 2, 10)): "router",
    ("128x64_64x3", (8, 2, 14)): "shared_gate_up", ("32x32_64x1f", (8, 5, 20)): "shared_gate_up",
    ("32x32_128x2b", (8, 10, 1)): "shared_down", ("16x16_128x2b", (8, 20, 1)): "shared_down",
    ("gemvx7", (8, 1, 1)): "shared_expert_gate", ("gemvx7", (2, 1, 1)): "shared_expert_gate",
    ("64x64_64x6", (8, 20, 1)): "ple_key", ("128x64_64x4", (8, 3, 14)): "h2560",
    ("32x32_64x1f", (8, 10, 20)): "h2560", ("128x128_32x6", (8, 3, 8)): "h2560",
    ("128x64_32x6", (8, 13, 5)): "qsa_qkv", ("16x16_64x2b", (8, 104, 2)): "qsa_qkv",
    ("32x32_32x1b", (8, 3, 20)): "qsa_index_qk", ("32x32_64x1f", (8, 3, 20)): "qsa_index_qk",
    ("256x64_32x4", (8, 122, 1)): "lm_head", ("32x32_128x1b", (8, 970, 1)): "lm_head",
}


def tag(name):
    if "cutlass_80" in name:
        m = re.search(r"cutlass_80_(wmma_)?tensorop_(bf16_)?s\d+gemm_(relu_)?bf16_(\d+x\d+_\d+x\d+)_tn_align(\d)", name)
        t = m.group(4)
        if m.group(1):
            t += "b" if m.group(2) else "f"
        if m.group(5) == "2":
            t += "a2"
        return t
    if "gemvx::kernel" in name:
        m = re.search(r"gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (\w+), "
                      r"true, true, false, (\d+),", name)
        return "gemvx" + m.group(2) + ("t" if m.group(1) == "true" else "")
    if "GemmUniversal" in name and "BlockScaled" in name:
        return "nvfp4_grouped"
    if "Fused_Moe_Kernel_sm80" in name or "MoeFCGemm" in name:
        return "bf16_moe"
    if "splitKreduce" in name:
        return "splitK"
    return None


def analyse(lab, streams):
    path = glob.glob(f"{lab}/profile-c{streams}/*.pt.trace.json.gz")[0]
    events = json.load(gzip.open(path))["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    ann = sorted((e for e in events if e.get("cat") == "gpu_user_annotation" and e.get("tid") == 33),
                 key=lambda e: e["ts"])
    wins = [(a["ts"], b["ts"]) for a, b in zip(ann, ann[1:])
            if streams == 1 or a["name"] == b["name"] == "execute_context_0(0)_generation_8(32)"]
    mapping = C1 if streams == 1 else C8
    following = {}
    by_stream = collections.defaultdict(list)
    for e in kernels:
        by_stream[e["args"]["stream"]].append(e)
    for lst in by_stream.values():
        for a, b in zip(lst, lst[1:]):
            following[id(a)] = b
    rows = collections.defaultdict(lambda: [0, 0.0])
    kinds = collections.defaultdict(collections.Counter)
    unmatched = collections.Counter()
    for t0, t1 in wins:
        for e in kernels:
            if not t0 <= e["ts"] < t1:
                continue
            t = tag(e["name"])
            if t in (None, "splitK"):
                continue
            dur = e["dur"]
            nxt = following.get(id(e))
            if nxt is not None and tag(nxt["name"]) == "splitK":
                dur += nxt["dur"]
            if t == "nvfp4_grouped":
                label = "nvfp4_experts (grouped GEMM only)"
            elif t == "bf16_moe":
                label = "mtp_bf16_moe"
            else:
                label = mapping.get((t, tuple(e["args"]["grid"])))
                if label is None:
                    unmatched[str((t, tuple(e["args"]["grid"])))] += 1
                    label = "unmatched"
            rows[label][0] += 1
            rows[label][1] += dur
            kinds[label][f"{t}{list(e['args']['grid'])}"] += 1
    n = len(wins)
    out = []
    for label, (count, dur) in rows.items():
        shape = SHAPES.get(label)
        wbytes = shape[0] * shape[1] * 2 if shape else None
        per_call = dur / count
        out.append({"label": label, "calls_per_iter": count / n, "us_per_call": per_call, "us_per_iter": dur / n,
                    "weight_mb": None if wbytes is None else wbytes / 1e6,
                    "tbps": None if wbytes is None else wbytes / (per_call * 1e-6) / 1e12,
                    "gb_per_iter": None if wbytes is None else wbytes * count / n / 1e9,
                    "kernels": {k: v / n for k, v in kinds[label].most_common()}})
    out.sort(key=lambda r: -r["us_per_iter"])
    dense = [r for r in out if r["label"] in SHAPES or r["label"] == "draft_m1_gemv"]
    return {"streams": streams, "iterations": n, "period_us": sum(b - a for a, b in wins) / n, "rows": out,
            "dense_us_per_iter": sum(r["us_per_iter"] for r in dense),
            "dense_gb_per_iter": sum(r["gb_per_iter"] or 0 for r in dense),
            "unmatched": {k: v / n for k, v in unmatched.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=LAB)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    result = {}
    for streams in (1, 8):
        r = analyse(args.lab, streams)
        result[f"c{streams}"] = r
        print(f"\n### {streams} stream(s): {r['iterations']} iterations, period {r['period_us'] / 1000:.1f} ms, "
              f"dense BF16 {r['dense_us_per_iter'] / 1000:.2f} ms/iter for {r['dense_gb_per_iter']:.2f} GB\n")
        print("| Projection | (N, K) | calls/iter | us/call | us/iter | MB/call | TB/s | kernels (tile[grid] x calls/iter) |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in r["rows"]:
            shape = SHAPES.get(row["label"])
            mb = "" if row["weight_mb"] is None else f"{row['weight_mb']:.2f}"
            tb = "" if row["tbps"] is None else f"{row['tbps']:.2f}"
            print(f"| {row['label']} | {shape or ''} | {row['calls_per_iter']:.0f} | {row['us_per_call']:.1f} | "
                  f"{row['us_per_iter']:.0f} | {mb} | {tb} | "
                  + ", ".join(f"{k} x{v:g}" for k, v in row["kernels"].items()) + " |")
        if r["unmatched"]:
            print("unmatched:", r["unmatched"])
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
