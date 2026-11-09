#!/usr/bin/env python3
"""Tables for docs/validation-20260927/kernels-phase5-stack.md from the stack smoke (``kernel-stack`` Modal test).

The run holds two serving-bench trials in one container: ``stack`` (phase-4 serving + the phase-5 stack) and
``control`` (the phase-4 serving, restarted in place). Per arm and point: decode-only tok/s (as in phases 2 and 4),
MTP acceptance and the acceptance-free milliseconds per decode iteration (decode-only seconds / verify steps, a
verify step being one draft per running request); stack/control ratios. From the stack's server log: the mid-M
tuning lines (``ARC_DENSE_MID``: pinned vs chosen tile, microseconds and errors per shape and M bucket), the overlay
identities and the load-time checks. From the torch profiles (16K prompts, 1 and 6 streams, both arms): the
iteration period, GPU idle, the host-preparation span before the first graph kernel, blocking stream syncs per
iteration and the phase-3 kernel census, so each item's share of the gain is read from the timeline.

Usage: python scripts/kernel_phase5_analyze.py RUN_DIR [--json OUT]
"""

import argparse
import collections
import gzip
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase3 = _module("phase3_trace", "kernel_phase3_trace.py")
TAGS = ("ARC_DENSE_MID ", "ARC_DENSE_KERNELS ", "ARC_DENSE_CHECK ", "ARC_STACK_KERNELS_PATCH ", "ARC_SK_CHECK ",
        "ARC_SERVING_KERNELS_PATCH ", "ARC_DENSE_KERNELS_PATCH ")
DECODE = re.compile(r"execute_context_0\(0\)_generation_\d+\(\d+\)$")


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


def arm_points(bench):
    rows = []
    for point in bench["points"]:
        passes = {p["pass"]: p for p in point.get("passes", [])}
        prefill, decode = passes.get("prefill"), passes.get("decode")
        if not decode:
            rows.append({"kind": point.get("kind"), "prompt": point.get("prompt_tokens"),
                         "streams": point.get("concurrency"), "error": point.get("error")})
            continue
        seconds = decode["seconds"]
        if prefill and prefill.get("prompt_tokens_per_second"):
            seconds -= decode["computed_prompt_tokens"] / prefill["prompt_tokens_per_second"]
        streams = point["concurrency"]
        # A decode step is one draft per running request (MTP-3) or one token per request (MTP-0).
        steps = (decode["spec_drafts"] if decode.get("spec_drafts") else decode["completion_tokens"]) / streams
        rows.append({"kind": point["kind"], "prompt": point["prompt_tokens"], "streams": streams, "h2d": point.get("h2d"),
                     "decode_only_tok_s": round(decode["completion_tokens"] / seconds, 1),
                     "per_stream_tok_s": round(decode["completion_tokens"] / seconds / streams, 1),
                     "accepted_per_draft": decode.get("accepted_tokens_per_draft"),
                     "ms_per_iteration": round(seconds / steps * 1000, 2) if steps else None,
                     "prefill_tok_s": prefill and prefill.get("prompt_tokens_per_second"),
                     "preemptions": decode.get("preemptions"), "peak_running": decode.get("peak_running_requests")})
    return rows


def trace_timeline(path):
    """Iteration period, GPU idle, host-prep span before the first graph kernel and blocking syncs per iteration."""
    events = json.load(gzip.open(path))["traceEvents"]
    gpu = sorted((e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    if not kernels:
        return {"error": "no kernels"}
    main = collections.Counter(e["args"]["stream"] for e in kernels).most_common(1)[0][0]
    ann = sorted((e for e in events if e.get("cat") == "gpu_user_annotation" and e.get("tid") == main),
                 key=lambda e: e["ts"])
    wins = [(a["ts"], b["ts"]) for a, b in zip(ann, ann[1:]) if DECODE.match(a["name"]) and DECODE.match(b["name"])]
    corr = {e["args"]["correlation"]: e for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    syncs = [e for e in events if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"
             and e["name"] == "cudaStreamSynchronize" and e["dur"] > 200]
    rows = []
    for t0, t1 in wins:
        inside = [e for e in gpu if t0 <= e["ts"] < t1]
        first = next((e for e in inside if (corr.get(e.get("args", {}).get("correlation")) or {}).get("name")
                      == "cudaGraphLaunch"), None)
        prep_end = first["ts"] if first else t1
        busy = phase3.union_length([(e["ts"], min(e["ts"] + e["dur"], t1)) for e in inside])
        rows.append({"period": t1 - t0, "idle": (t1 - t0) - busy, "prep": prep_end - t0,
                     "prep_busy": sum(e["dur"] for e in inside if e["ts"] < prep_end),
                     "syncs": sum(1 for s in syncs if t0 - 500 <= s["ts"] + s["dur"] <= t1)})
    n = len(rows)
    if not n:
        return {"iterations": 0}
    mean = {k: sum(r[k] for r in rows) / n for k in rows[0]}
    return {"iterations": n, "period_ms": mean["period"] / 1000, "gpu_idle_ms": mean["idle"] / 1000,
            "host_prep_ms": mean["prep"] / 1000, "host_prep_gpu_busy_ms": mean["prep_busy"] / 1000,
            "blocking_syncs_per_iteration": mean["syncs"]}


def profiles(folder):
    out = {}
    for sub in sorted(Path(folder).glob("profile-c*")):
        traces = sorted(sub.glob("*.pt.trace.json.gz"))
        if not traces:
            continue
        streams = int(sub.name.removeprefix("profile-c"))
        record = {"timeline": trace_timeline(traces[0])}
        try:
            census = phase3.analyse(sub, {"mode": "mtp3", "streams": streams, "tokens": 4 * streams})
            record["categories"] = {k: round(v["main_ms"] + v["side_ms"], 3) for k, v in census["categories"].items()}
            record["top_kernels"] = census["kernels"][:25]
        except Exception as exc:  # noqa: BLE001 - keep the timeline when the census cannot run
            record["census_error"] = f"{type(exc).__name__}: {exc}"
        out[streams] = record
    return out


def mid_table(lines):
    rows = []
    for entry in lines:
        if not isinstance(entry, dict):
            continue
        for bucket, b in entry.get("buckets", {}).items():
            chosen = next((c for c in b.get("candidates", []) if c.get("tile") == b.get("chosen")), None)
            rows.append({"kind": entry["kind"], "shape": entry["shape"], "prefix": entry.get("prefix"), "m": int(bucket),
                         "pinned_us": b.get("pinned_us"), "pinned_err": b.get("pinned_err"), "chosen": b.get("chosen"),
                         "chosen_us": chosen and chosen.get("us"), "chosen_err": chosen and chosen.get("err"),
                         "best_candidate_us": min((c["us"] for c in b.get("candidates", []) if c.get("us")),
                                                  default=None),
                         "failed": [c["tile"] for c in b.get("candidates", []) if c.get("error")]})
    return rows


# Calls per MTP-3 decode iteration at six streams (verify and first draft step at M = 24, draft steps 2-3 at M = 8),
# from the pinned model source and the phase-1 per-iteration counts; side-stream shapes are marked (overlapped with
# the routed experts, so their saving is not on the critical path).
CALLS_SIX_STREAMS = {
    ("bf16", (336, 10240)): {24: 98, 8: 4}, ("bf16", (320, 10240)): {24: 2, 8: 2},
    ("bf16", (10240, 320)): {24: 100, 8: 6}, ("bf16", (512, 2560)): {24: 49, 8: 2},
    ("bf16", (96, 2560)): {24: 36}, ("bf16", (640, 2560)): {24: 13, 8: 2},
    ("bf16", (1280, 2560)): {24: 49, 8: 2}, ("bf16", (2560, 640)): {24: 49, 8: 2},
    ("bf16", (2560, 2560)): {24: 1, 8: 2},
    ("int8", (16384, 2560)): {24: 36}, ("int8", (13312, 2560)): {24: 13}, ("int8", (2560, 6144)): {24: 49},
    ("int8", (248320, 2560)): {24: 1},
}
SIDE_STREAM = {(1280, 2560), (2560, 640)}


def _mtp0_calls(bucket):
    """MTP-0: one 48-layer forward per step at the graph size ``bucket`` (6 streams -> 8, 14 -> 16)."""
    return {("bf16", (336, 10240)): {bucket: 96}, ("bf16", (320, 10240)): {bucket: 1},
            ("bf16", (10240, 320)): {bucket: 97}, ("bf16", (512, 2560)): {bucket: 48},
            ("bf16", (96, 2560)): {bucket: 36}, ("bf16", (640, 2560)): {bucket: 12},
            ("bf16", (1280, 2560)): {bucket: 48}, ("bf16", (2560, 640)): {bucket: 48},
            ("bf16", (2560, 2560)): {bucket: 1}}


CALLS_MTP0 = {6: _mtp0_calls(8), 14: _mtp0_calls(16)}


def projected_savings(rows, calls_table=CALLS_SIX_STREAMS):
    """Microbenchmark saving per iteration: sum over shapes of calls x (pinned - chosen) microseconds."""
    main = side = 0.0
    per_shape = {}
    for row in rows:
        calls = calls_table.get((row["kind"], tuple(row["shape"])), {}).get(row["m"], 0)
        if not calls or row["chosen"] is None or row["pinned_us"] is None or row["chosen_us"] is None:
            continue
        saved = calls * (row["pinned_us"] - row["chosen_us"]) / 1000
        key = f"{row['kind']} {row['shape'][0]}x{row['shape'][1]}"
        per_shape[key] = per_shape.get(key, 0.0) + saved
        if tuple(row["shape"]) in SIDE_STREAM:
            side += saved
        else:
            main += saved
    return {"main_stream_ms": round(main, 3), "side_stream_ms": round(side, 3),
            "per_shape_ms": {k: round(v, 3) for k, v in sorted(per_shape.items(), key=lambda kv: -kv[1])}}


def arm(folder):
    folder = Path(folder)
    bench_path = folder / "bench.json"
    if not bench_path.is_file():
        return {"error": f"no bench.json in {folder}"}
    bench = json.loads(bench_path.read_text())
    logs = "".join(p.read_text(errors="replace") for p in sorted(folder.glob("vllm-openai-server.log")))
    lines = log_lines(logs)
    return {"status": bench.get("status"), "task_errors": bench.get("task_errors"), "points": arm_points(bench),
            "sanity": bench.get("sanity"), "preemption_free": bench.get("preemption_free"),
            "engine": bench.get("engine"), "focus_fit": bench.get("focus_fit"),
            "kv_cache_tokens": [int(m.replace(",", "")) for m in re.findall(r"GPU KV cache size: ([\d,]+) tokens", logs)],
            "model_gib": re.findall(r"Model loading took ([\d.]+) GiB", logs),
            "startup_s": re.findall(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s", logs),
            "mid": mid_table(lines["ARC_DENSE_MID"]), "overlay": {k: v for k, v in lines.items() if k != "ARC_DENSE_MID"},
            "profiles": profiles(folder)}


def compare(stack, control):
    rows = []
    index = {(p["kind"], p["prompt"], p["streams"]): p for p in control.get("points", []) if "decode_only_tok_s" in p}
    for p in stack.get("points", []):
        c = index.get((p["kind"], p["prompt"], p["streams"]))
        if c is None or "decode_only_tok_s" not in p:
            continue
        rows.append({"kind": p["kind"], "prompt": p["prompt"], "streams": p["streams"],
                     "control_per_stream": c["per_stream_tok_s"], "stack_per_stream": p["per_stream_tok_s"],
                     "control_aggregate": c["decode_only_tok_s"], "stack_aggregate": p["decode_only_tok_s"],
                     "tok_s_ratio": round(p["decode_only_tok_s"] / c["decode_only_tok_s"], 4),
                     "control_ms_per_iteration": c["ms_per_iteration"], "stack_ms_per_iteration": p["ms_per_iteration"],
                     "iteration_ratio": (round(c["ms_per_iteration"] / p["ms_per_iteration"], 4)
                                         if c["ms_per_iteration"] and p["ms_per_iteration"] else None),
                     "accepted_per_draft": [c["accepted_per_draft"], p["accepted_per_draft"]]})
    return rows


def measured_plan(log_files, min_gain=0.03, error_factor=1.25, error_slack=5e-4):
    """A fixed mid-M plan from the tuner lines of several server starts: per (kind, shape, bucket) the candidate
    with the lowest mean time, kept when its mean is >= 3% below the pinned path's mean and its error passes the
    runtime's bound in every start (the same rule as the load-time tuner, on pooled measurements)."""
    pooled = {}
    for path in log_files:
        for entry in log_lines(Path(path).read_text(errors="replace"))["ARC_DENSE_MID"]:
            if not isinstance(entry, dict) or "buckets" not in entry:
                continue
            key = (entry["kind"], f"{entry['shape'][0]},{entry['shape'][1]}")
            for bucket, b in entry["buckets"].items():
                slot = pooled.setdefault(key, {}).setdefault(bucket, {"pinned": [], "tiles": {}})
                slot["pinned"].append(b["pinned_us"])
                limit = max(error_factor * b["pinned_err"], b["pinned_err"] + error_slack)
                for c in b.get("candidates", []):
                    if c.get("us") is None:
                        continue
                    t = slot["tiles"].setdefault(tuple(c["tile"]), {"us": [], "ok": True})
                    t["us"].append(c["us"])
                    t["ok"] &= c["err"] <= limit
    plan = {"bf16": {}, "int8": {}}
    detail = []
    for (kind, shape), buckets in sorted(pooled.items()):
        for bucket, slot in sorted(buckets.items(), key=lambda kv: int(kv[0])):
            pinned = sum(slot["pinned"]) / len(slot["pinned"])
            runs = len(slot["pinned"])
            tiles = {t: sum(v["us"]) / len(v["us"]) for t, v in slot["tiles"].items() if v["ok"] and len(v["us"]) == runs}
            best = min(tiles, key=tiles.get) if tiles else None
            chosen = list(best) if best is not None and tiles[best] <= pinned * (1 - min_gain) else None
            plan[kind].setdefault(shape, {})[bucket] = chosen
            detail.append({"kind": kind, "shape": shape, "m": int(bucket), "pinned_us": round(pinned, 2),
                           "tile": chosen, "tile_us": round(tiles[best], 2) if chosen else None, "runs": runs})
    return plan, detail


def split_modes(arm_result, mode):
    """The points of one host-copy mode (the MTP-3 arm runs every point blocking, then async)."""
    return {**arm_result, "points": [p for p in arm_result.get("points", []) if p.get("h2d") == mode]}


def phase4_reference(path):
    """The recorded phase-4 smoke (same bundle method, another container) as a cross-container reference."""
    bench = Path(path) / "arena/trials/serving-bench/bench.json"
    return {"points": arm_points(json.loads(bench.read_text()))} if bench.is_file() else {"points": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--phase4", default="runs/modal-validation/arena-kernel-smoke-phase4-v2-modal-20260928")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    trials = Path(args.run) / "arena" / "trials"
    mtp3, mtp0 = arm(trials / "stack-mtp3"), arm(trials / "stack-mtp0-kv8")
    blocking, asynchronous = split_modes(mtp3, "blocking"), split_modes(mtp3, "async")
    phase4 = phase4_reference(args.phase4)
    result = {"run": str(args.run), "stack_mtp3": mtp3, "stack_mtp0_kv8": mtp0,
              # Same server, same prompts: the host-copy item alone.
              "async_h2d_ab": compare(asynchronous, blocking),
              # Cross-container against the recorded phase-4 smoke: blocking = phase 4 + mid-M; async = the stack.
              "mid_m_vs_phase4": compare(blocking, phase4), "stack_vs_phase4": compare(asynchronous, phase4),
              "mid_projected_six_streams_mtp3": projected_savings(mtp3.get("mid", [])),
              "mid_mtp0": mtp0.get("mid", [])}
    plan, detail = measured_plan(sorted(trials.glob("*/vllm-openai-server.log")))
    result["measured_plan"], result["measured_plan_detail"] = plan, detail
    result["mid_projected"] = {
        "mtp3_six_streams": projected_savings([{**d, "shape": [int(v) for v in d["shape"].split(",")],
                                                "chosen": d["tile"], "chosen_us": d["tile_us"]} for d in detail],
                                              CALLS_SIX_STREAMS),
        "mtp0_six_streams": projected_savings([{**d, "shape": [int(v) for v in d["shape"].split(",")],
                                                "chosen": d["tile"], "chosen_us": d["tile_us"]} for d in detail],
                                              CALLS_MTP0[6]),
        "mtp0_fourteen_streams": projected_savings([{**d, "shape": [int(v) for v in d["shape"].split(",")],
                                                     "chosen": d["tile"], "chosen_us": d["tile_us"]} for d in detail],
                                                   CALLS_MTP0[14])}
    path = Path(args.run) / "execution.json"
    if path.is_file():
        result["execution"] = json.loads(path.read_text())
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str) + "\n")
    for key in ("async_h2d_ab", "mid_m_vs_phase4", "stack_vs_phase4"):
        print(key, json.dumps(result[key], indent=1))
    print("mtp0 points", json.dumps(mtp0.get("points"), indent=1))
    for key, value in (("mtp3", mtp3), ("mtp0", mtp0)):
        for streams, prof in value.get("profiles", {}).items():
            print(key, f"c{streams}", json.dumps(prof.get("timeline")), json.dumps(prof.get("categories")))
        for row in value.get("mid", []):
            print(key, json.dumps({k: row[k] for k in ("kind", "shape", "m", "pinned_us", "chosen", "chosen_us",
                                                        "pinned_err", "chosen_err", "failed")}))


if __name__ == "__main__":
    main()
