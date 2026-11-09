"""Tables for docs/validation-20260927/hero-v2-precheck.md from the pre-check's bench.json (29 September).

    uv run python scripts/arena_analysis/hero_v2_precheck_analyze.py <run dir> --out docs/validation-20260927/hero-v2-precheck.json

- served layout: num_blocks at max_num_seqs 32 against the recorded 1,335 at 25, the gate capacity (0.9), the 64K
  windows the full-window rule admits (51 blocks each on the hero layout) and the server's own KV-cache size line;
- stream sweep: every decode / mixed window, the decode-only fit t(n) = a + b n, the joint fit
  t = a + b x (generated per step) + c x (uncached per step) over all windows, and the projection at the arena's
  phase-2 prefill load (k = 1.35 uncached per generated token): tok/s(n) = 1000 n / (a + (b + c k) n);
- partial-hit check: pooled pair statistics and the verdict; replay: served vs block and unit ideals.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

RECORDED_BLOCKS = 1335  # arena-hero-v2-2h-v1 at max_num_seqs 25
WINDOW_BLOCKS = 51      # a full 64K window on the hero layout (arena.hero_v2)
ARENA_K = 1.35          # phase-2 uncached prompt tokens per generated token (hero-v2-analysis/throughput.md)
IN_RUN = (14.99, 0.487, 0.0808)  # the run's own phase-2 fit (a, b, c)


def find(run, name):
    hits = sorted(Path(run).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} under {run}")
    return hits[0]


def fits(points):
    decode = [p for p in points if p["kind"] == "decode" and p.get("valid")]
    every = [p for p in points if p.get("valid")]
    result = {}
    if len(decode) >= 2:
        x = np.array([p["generated_per_step"] for p in decode])
        y = np.array([p["step_ms"] for p in decode])
        b, a = np.polyfit(x, y, 1)
        result["decode_only"] = {"a_ms": round(float(a), 3), "b_ms_per_stream": round(float(b), 4),
                                 "rms_ms": round(float(np.sqrt(np.mean((y - a - b * x) ** 2))), 3),
                                 "points": len(decode)}
    if len(every) >= 4 and any(p["kind"] == "mixed" for p in every):
        X = np.array([[1.0, p["generated_per_step"], p["uncached_per_step"]] for p in every])
        y = np.array([p["step_ms"] for p in every])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = (float(v) for v in coef)
        result["joint"] = {"a_ms": round(a, 3), "b_ms_per_stream": round(b, 4), "c_ms_per_uncached": round(c, 5),
                           "rms_ms": round(float(np.sqrt(np.mean((y - X @ coef) ** 2))), 3), "points": len(every)}
    return result


def project(a, b, c, streams=(23, 25, 28, 30, 32), k=ARENA_K):
    rates = {n: 1000 * n / (a + (b + c * k) * n) for n in streams}
    base = rates[streams[0]]
    return {str(n): {"tok_s": round(r, 1), "vs_23": round(r / base - 1, 4)} for n, r in rates.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    bench = json.loads(find(args.run, "bench.json").read_text())
    layout = bench.get("layout") or {}
    blocks = layout.get("num_blocks")
    capacity = bench.get("capacity_blocks")
    result = {"status": bench.get("status"), "task_errors": bench.get("task_errors"),
              "preemption_free": bench.get("preemption_free"),
              "layout": {"num_blocks": blocks, "recorded_num_blocks_at_25": RECORDED_BLOCKS,
                         "delta_blocks": None if blocks is None else blocks - RECORDED_BLOCKS,
                         "gate_capacity": capacity, "kv_cache_tokens": bench.get("kv_cache_tokens"),
                         "full_64k_windows_at_gate": None if capacity is None else capacity // WINDOW_BLOCKS,
                         "focus_fit": bench.get("focus_fit")},
              "sanity": [(s.get("expect"), s.get("ok")) for s in bench.get("sanity") or []]}
    log = next((p for p in Path(args.run).rglob("vllm-openai-server.log")), None)
    if log is not None:
        text = log.read_text(errors="replace")
        graphs = re.findall(r"cudagraph_capture_sizes'?:? ?(\[[\d, ]+\])", text)
        unit = re.findall(r"prefix_match_unit[=:'\" ]+(\d+)", text)
        result["server_log"] = {"cudagraph_capture_sizes": graphs[-1] if graphs else None,
                                "prefix_match_unit": unit[-1] if unit else None,
                                "max_num_seqs": (re.findall(r"max_num_seqs[=:'\" ]+(\d+)", text) or [None])[-1]}
    sweep = bench.get("stream_sweep") or {}
    points = sweep.get("points") or []
    result["sweep"] = {
        "errors": sweep.get("errors"), "prompt_tokens": sweep.get("prompt_tokens"),
        "prefill": {k: (sweep.get("prefill") or {}).get(k) for k in (
            "seconds", "prompt_tokens", "computed_prompt_tokens", "prompt_tokens_per_second", "preemptions")},
        "points": [{k: p.get(k) for k in (
            "kind", "streams", "mean_prompt_tokens", "step_ms", "generated_per_step", "uncached_per_step",
            "generated_tok_s", "per_stream_tok_s", "uncached_tok_s", "itl_ms", "steps", "seconds", "running_min",
            "kv_cache_usage_max", "ended_early", "preemptions", "inject")} for p in points]}
    fitted = fits(points)
    result["fits"] = fitted
    result["projection"] = {"in_run_fit": project(*IN_RUN)}
    if "joint" in fitted:
        j = fitted["joint"]
        result["projection"]["joint_fit"] = project(j["a_ms"], j["b_ms_per_stream"], j["c_ms_per_uncached"])
    if "decode_only" in fitted:
        d = fitted["decode_only"]
        # decode slope from this sweep, prefill cost per uncached token from the run
        result["projection"]["decode_fit_run_prefill"] = project(d["a_ms"], d["b_ms_per_stream"], IN_RUN[2])
    check = bench.get("partial_check") or {}
    result["partial_check"] = {"summary": check.get("summary"),
                               "cached_tokens": {row["index"]: {k: v.get("cached_tokens") for k, v in
                                                                row["conditions"].items() if "cached_tokens" in v}
                                                 for row in check.get("sequences") or []},
                               "per_sequence_pairs": [row.get("pairs") for row in check.get("sequences") or []]}
    replay = bench.get("prefix_replay") or {}
    result["prefix_replay"] = replay.get("summary")
    result["engine"] = bench.get("engine")
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
