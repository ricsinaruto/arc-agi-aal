"""Analyse the speculative-decoding pilots (30 September; docs/validation-20260930/spec-decode.md).

Inputs: per arm (MTP-0 / 1 / 2) the bench report (``arena/trials/spec/bench.json``) and the served layout
(``arena/serving/cache-layout.json``). Output: decode tok/s, step ms and acceptance per stream count; the greedy
agreement against MTP-0; the KV fit from the measured pool; and the frontier projection against the pre-stated bar.

    python scripts/spec_decode_analyze.py --root <dir with mtp0/ mtp1/ mtp2/> --json <out>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_decode_sizing as sz  # noqa: E402

REFERENCE = {"streams": 37, "window": 93184, "engine": 987.0, "useful": 849.0, "tax": 0.14}
PREFILL_SHARE = 0.072  # blocks arm: prefill share of engine time (frontier section 1.2), not sped up by drafting
KEEP_DEMOTE_LOSS = 0.01  # prefix-cache keep/demote disable themselves with speculative state blocks (assumed ~1%)
BAR = 1.15


def load(root, arm):
    base = Path(root) / arm
    bench = json.loads((base / "bench.json").read_text())
    layout = json.loads((base / "cache-layout.json").read_text()) if (base / "cache-layout.json").exists() else None
    return bench, layout


def common_prefix(a, b):
    n = 0
    for x, y in zip(a or [], b or []):
        if x != y:
            break
        n += 1
    return n


def greedy_rows(bench, order):
    runs = ((bench.get("spec_sweep") or {}).get("greedy") or {}).get("runs") or {}
    return {r["id"]: r.get("token_ids") for r in (runs.get(order) or {}).get("rows", [])}


def agreement(a, b):
    ids = sorted(set(a) & set(b))
    prefixes = [common_prefix(a[i], b[i]) for i in ids]
    full = [p == len(a[i] or []) == len(b[i] or []) for p, i in zip(prefixes, ids)]
    return {"pairs": len(ids), "mean_common_prefix": round(sum(prefixes) / len(ids), 1) if ids else None,
            "identical": sum(full), "prefixes": prefixes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    arms = {}
    for arm in ("mtp0", "mtp1", "mtp2"):
        try:
            arms[arm] = load(args.root, arm)
        except FileNotFoundError:
            print(f"{arm}: missing")
    out = {"reference": REFERENCE, "bar": BAR, "arms": {}}
    base_points = {p["streams"]: p for p in arms["mtp0"][0]["spec_sweep"]["points"] if p.get("valid")} if "mtp0" in arms else {}
    g0c = greedy_rows(arms["mtp0"][0], "concurrent") if "mtp0" in arms else {}
    g0s = greedy_rows(arms["mtp0"][0], "sequential") if "mtp0" in arms else {}
    noise = agreement(g0c, g0s) if g0c and g0s else None
    out["greedy_noise_mtp0"] = noise
    for arm, (bench, layout) in arms.items():
        k = int(arm[-1])
        sweep = bench.get("spec_sweep") or {}
        points = {p["streams"]: p for p in sweep.get("points", []) if p.get("valid")}
        row = {"errors": sweep.get("errors", []) + bench.get("task_errors", []),
               "num_blocks": layout.get("num_blocks") if layout else None,
               "lookahead": layout.get("lookahead") if layout else None,
               "block_tokens": ([g["block_size"] for g in layout["groups"] if g["type"] == "FullAttentionSpec"][0]
                                if layout else None),
               "fixed_blocks_1_token": layout["costs"][0]["blocks"] - 1 if layout and layout.get("costs") else None,
               "gpu_memory_after_prefill": sweep.get("gpu_memory_after_prefill"),
               "preemption_free": bench.get("preemption_free"), "points": {}}
        for n, p in sorted(points.items()):
            ref = base_points.get(n)
            row["points"][n] = {key: p.get(key) for key in (
                "step_ms", "generated_per_step", "generated_tok_s", "per_stream_tok_s", "accepted_per_draft",
                "acceptance_rate", "per_position", "mean_prompt_tokens", "preemptions")}
            if ref:
                row["points"][n]["vs_mtp0_same_streams"] = round(p["generated_tok_s"] / ref["generated_tok_s"], 4)
        # KV fit at the measured pool (sizing rule), and the frontier projection at each measured stream count.
        g = sz.geometry(k)
        blocks = row["num_blocks"] or sz.pool_blocks(k, 1.5)
        gate = int(sz.GATE * blocks)
        ref37 = base_points.get(37)
        row["fit"] = {"streams_at_93k": round(sz.streams_at(93184, gate, g), 1),
                      "window_at": {n: int(sz.window_at(n, gate, g)) for n in (24, 31, 32, 37, 40)}}
        proj = {}
        for n, p in sorted(points.items()):
            window = min(REFERENCE["window"], sz.window_at(n, gate, g))
            if not ref37 or window < 40000:
                continue
            decode_ratio = p["generated_tok_s"] / ref37["generated_tok_s"]
            engine_ratio = 1 / ((1 - PREFILL_SHARE) / decode_ratio + PREFILL_SHARE)
            useful_ratio = (engine_ratio * (1 - sz.tax(window)) / (1 - REFERENCE["tax"])
                            * (1 - (KEEP_DEMOTE_LOSS if k else 0)))
            proj[n] = {"window": int(window), "decode_ratio_vs_mtp0_37": round(decode_ratio, 4),
                       "engine_ratio": round(engine_ratio, 4), "tax": round(sz.tax(window), 3),
                       "useful_ratio": round(useful_ratio, 4), "passes_bar": useful_ratio >= BAR}
        row["projection"] = proj
        if k and g0c:
            row["greedy_vs_mtp0"] = {o: agreement(greedy_rows(bench, o), g0c) for o in ("concurrent", "sequential")}
        out["arms"][arm] = row
    text = json.dumps(out, indent=1, default=str)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
