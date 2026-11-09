#!/usr/bin/env python3
"""Tables for docs/validation-20260927/kv-cache-8bit.md from the 8-bit KV smoke.

``--smoke <run dir>``: the served layout and focus fit, decode per point (phase-4 method), the greedy sanity
answers, the kv8 overlay's log lines, the in-run drift against the recorded A+C (BF16 KV) gate trial and the
QSA microbenchmark. ``--control <kernel-ref kernel-quality.json>`` and ``--ac <kernel-gate kernel-quality.json>``
(the recorded phase-2 trials) add the reference noise computed with the same functions: control vs its own
repeat, control vs A+C, and control vs the 8-bit server.

Usage: python scripts/kv8_analyze.py --smoke DIR [--control FILE --ac FILE] [--json OUT]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
_spec = importlib.util.spec_from_file_location("phase4_analyze", HERE / "kernel_phase4_analyze.py")
phase4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase4)
phase4.TAGS = (*phase4.TAGS, "ARC_KV8_FIRST_CALL ")

from arena import kernel_gate as gate  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text())


def drift(reference, candidate):
    keys = ("positions", "mean_kl", "p50_kl", "p99_kl", "top1_agreement", "mean_abs_delta_logprob")
    tf = gate.compare_teacher(reference.get("teacher_forced", {}), candidate.get("teacher_forced", {}))
    greedy = gate.compare_greedy(reference.get("greedy", {}), candidate.get("greedy", {}))
    return {"teacher_forced": {k: tf.get(k) for k in keys},
            "greedy": {k: greedy.get(k) for k in ("prompts", "exact_match", "mean_first_divergence", "positions",
                                                   "mean_kl", "top1_agreement")}}


def microbench_table(report):
    rows = []
    for row in (report or {}).get("rows", []):
        entry = {k: row.get(k) for k in ("topk", "rows", "requests", "bf16_us", "int8_us", "fp8_us",
                                         "bf16_rel_error", "int8_rel_error", "fp8_rel_error",
                                         "int8_writer_codes_equal", "fp8_writer_codes_equal",
                                         "int8_writer_scale_max_rel", "fp8_writer_scale_max_rel")}
        for mode in ("int8", "fp8"):
            if row.get("bf16_us") and row.get(f"{mode}_us"):
                entry[f"{mode}_over_bf16"] = round(row[f"{mode}_us"] / row["bf16_us"], 3)
        rows.append(entry)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--control")
    ap.add_argument("--ac")
    ap.add_argument("--json")
    args = ap.parse_args()
    folder = Path(args.smoke)
    out = {"smoke": phase4.smoke_summary(folder)}
    bench = load(next(folder.rglob("bench.json")))
    out["quality"] = bench.get("quality")
    micro = bench.get("kv8_microbench") or {}
    out["microbench"] = {k: v for k, v in micro.items() if k != "report"}
    out["microbench"]["table"] = microbench_table(micro.get("report"))
    rows = next(folder.rglob("kv8-quality.json"), None)
    if rows and args.control and args.ac:
        kv8 = load(rows)
        control, ac = load(args.control), load(args.ac)
        repeat = {"greedy": control.get("greedy_repeat", {}), "teacher_forced": control.get("teacher_forced_repeat", {})}
        out["noise"] = {"control_vs_control_repeat": drift(control, repeat), "control_vs_ac": drift(control, ac),
                        "control_vs_kv8": drift(control, kv8), "ac_vs_kv8": drift(ac, kv8),
                        "kv8_vs_kv8_repeat": drift(kv8, {"teacher_forced": kv8.get("teacher_forced_repeat", {})})}
    text = json.dumps(out, indent=1, default=str)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text[:30000])


if __name__ == "__main__":
    main()
