"""tpb2_read.py P75_RUN[,P75_RUN2] D4_RUN[,D4_RUN2] [--local DIR]: p75 vs p75d4 serving read.

Per context L and model: rule streams S, steady decode tok/s and steps/s (steady_window: pure decode at S), acceptance
(accepted tokens per draft), prefill tok/s, KV pool, preemptions, decode-pass prefix hit. Normalised throughput at equal
acceptance: steps/s x S x (1 + a_mean), a_mean = the two models' mean acceptance at L; also the raw tok/s ratio (each model's own acceptance) and the acceptance gap. Ratio d4/p75 with SE over the
decode rounds (delta method). Matched S: d4 steps/s at p75's count vs p75 at its own (per-step gain); 48 streams: the
context cost curve. Pre-registered bar: keep d4 if the geometric mean over L of the normalised ratio is >= 1.10."""
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

BUCKET = "nebius-arc:arc-runs-bucket/runs/validation"


def load(run, local=None):
    if local:
        return json.loads((Path(local) / run / "bench.json").read_text())
    txt = subprocess.run(["rclone", "cat", f"{BUCKET}/{run}/arena/working/trials/bench-tpb2/bench.json"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(txt)


def mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None)


def collect(reports):
    """{L: {"S": .., "by_streams": {n: {"tok_s": [...], "steps_s": [...], "acc": [...], "intervals": [...]}}, ...}}"""
    out = {}
    for rep in reports:
        for p in rep.get("points", []):
            if p.get("kind") != "resident":
                continue
            L = p["prompt_tokens"]
            row = out.setdefault(L, {"counts": p["counts"], "capacity": [], "kv_tokens": [], "prefill_tok_s": [],
                                     "preemptions": 0, "hit": [], "gate_waiting": [], "by_streams": {}})
            row["capacity"].append(p["capacity_blocks"])
            row["kv_tokens"].append(rep.get("kv_cache_tokens"))
            for q in p["passes"]:
                row["preemptions"] += q.get("preemptions") or 0
                row["gate_waiting"].append((q.get("gate") or {}).get("peak_waiting"))
                if q["pass"] == "prefill":
                    row["prefill_tok_s"].append(q.get("prompt_tokens_per_second"))
                    continue
                c, h = q.get("computed_prompt_tokens") or 0, q.get("cached_prompt_tokens") or 0
                row["hit"].append(h / (c + h) if c + h else None)
                s = q.get("steady") or {}
                b = row["by_streams"].setdefault(q["streams"], {"tok_s": [], "steps_s": [], "acc": [], "intervals": []})
                if s.get("intervals"):
                    b["tok_s"].append(s["decode_tokens_per_second"])
                    b["steps_s"].append(s["steps_per_second"])
                    b["acc"].append(s["accepted_tokens_per_draft"])
                b["intervals"].append(s.get("intervals", 0))
    return out


def rule_streams(row):
    return next(c["streams"] for c in row["counts"] if c["asked"] == "rule")


def main(argv):
    local = argv[argv.index("--local") + 1] if "--local" in argv else None
    runs = [a for a in argv if not a.startswith("--") and a != local]
    p75 = collect([load(r, local) for r in runs[0].split(",")])
    d4 = collect([load(r, local) for r in runs[1].split(",")])
    ratios, raws, lines = [], [], []
    for L in sorted(set(p75) & set(d4)):
        a, b = p75[L], d4[L]
        sa, sb = rule_streams(a), rule_streams(b)
        A, B = a["by_streams"].get(sa, {}), b["by_streams"].get(sb, {})
        acc_a, acc_b = mean_se(A.get("acc", []))[0], mean_se(B.get("acc", []))[0]
        if acc_a is None or acc_b is None:
            lines.append({"L": L, "error": "no steady window at rule streams", "intervals_p75": A.get("intervals"),
                          "intervals_d4": B.get("intervals")})
            continue
        abar = (acc_a + acc_b) / 2
        (ka, ka_se), (kb, kb_se) = mean_se(A["steps_s"]), mean_se(B["steps_s"])
        na, nb = ka * sa * (1 + abar), kb * sb * (1 + abar)
        r = nb / na
        rel = math.sqrt(((ka_se or 0) / ka) ** 2 + ((kb_se or 0) / kb) ** 2)
        ratios.append(r)
        raw = mean_se(B["tok_s"])[0] / mean_se(A["tok_s"])[0]  # as served: each model's own acceptance
        raws.append(raw)
        sm = next((c["streams"] for c in b["counts"] if isinstance(c["asked"], dict)), None)  # p75's count on d4
        M = b["by_streams"].get(sm, {})
        km = mean_se(M.get("steps_s", []))[0]
        C48a, C48b = a["by_streams"].get(48, {}), b["by_streams"].get(48, {})
        lines.append({
            "L": L, "streams": {"p75": sa, "d4": sb}, "capacity_blocks": {"p75": a["capacity"][0], "d4": b["capacity"][0]},
            "kv_tokens": {"p75": a["kv_tokens"][0], "d4": b["kv_tokens"][0]},
            "decode_tok_s_raw": {"p75": round(mean_se(A["tok_s"])[0], 1), "d4": round(mean_se(B["tok_s"])[0], 1)},
            "steps_s": {"p75": round(ka, 3), "d4": round(kb, 3)}, "acc": {"p75": round(acc_a, 3), "d4": round(acc_b, 3)},
            "norm_tok_s": {"p75": round(na, 1), "d4": round(nb, 1)}, "ratio_norm": round(r, 4),
            "ratio_raw": round(raw, 4), "acc_gap_d4_minus_p75": round(acc_b - acc_a, 3),
            "ratio_se": round(r * rel, 4),
            "matched_S": {"S": sm, "steps_s_p75": round(ka, 3), "steps_s_d4": None if km is None else round(km, 3),
                          "per_step_gain": None if km is None else round(km / ka, 4),
                          "stream_gain": None if km is None else round((kb * sb) / (km * sm), 4)},
            "steps_s_48": {"p75": mean_se(C48a.get("steps_s", []))[0], "d4": mean_se(C48b.get("steps_s", []))[0]},
            "prefill_tok_s": {"p75": mean_se(a["prefill_tok_s"])[0], "d4": mean_se(b["prefill_tok_s"])[0]},
            "preemptions": {"p75": a["preemptions"], "d4": b["preemptions"]},
            "decode_hit_min": {"p75": min(x for x in a["hit"] if x is not None), "d4": min(x for x in b["hit"] if x is not None)},
            "gate_peak_waiting": {"p75": max(x or 0 for x in a["gate_waiting"]), "d4": max(x or 0 for x in b["gate_waiting"])},
            "steady_intervals": {"p75": A.get("intervals"), "d4": B.get("intervals")}})
    for x in lines:
        print(json.dumps(x))
    if len(ratios) == 3:
        g = math.exp(sum(math.log(x) for x in ratios) / 3)
        graw = math.exp(sum(math.log(x) for x in raws) / 3)
        print(json.dumps({"geomean_ratio_norm": round(g, 4), "bar": 1.10, "keep_d4": g >= 1.10,
                          "geomean_ratio_raw": round(graw, 4), "note": "raw = as served with each model's own draft acceptance"}))


if __name__ == "__main__":
    main(sys.argv[1:])
