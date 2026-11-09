"""read50k.py RUN [RUN ...] [--base RUN]: the 50K throughput play read (the throughput screen for
every healed or partly healed cut). Per run: steady-state generated tok/s = engine generated tokens over the longest
contiguous stretch of progress rows with running >= 90% of the run's peak running (the rule's streams, all busy),
its KV mean, the run's prefix hit and draft acceptance (vLLM counters), and score marks at 25K/50K (score_marks.py,
descriptive). With --base, each run's tok/s ratio to the base."""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import score_marks  # noqa: E402

BUCKET = "nebius-arc:arc-runs-bucket/runs/validation/"


def cat(path):
    r = subprocess.run(["rclone", "cat", path], capture_output=True, text=True, timeout=300)
    return r.stdout if r.returncode == 0 else ""


def steady(rows, frac=0.9):
    pts = [(r["elapsed_seconds"], (r.get("live") or {}).get("engine") or {}) for r in rows if r.get("elapsed_seconds") is not None]
    pts = [(t, e) for t, e in pts if e.get("generated_tokens") is not None]
    peak = max((e.get("running") or 0) for _, e in pts) if pts else 0
    best, cur = [], []
    for t, e in pts:
        if (e.get("running") or 0) >= frac * peak:
            cur.append((t, e))
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    if len(best) < 2:
        return {"peak_running": peak, "error": "no steady stretch"}
    (t0, e0), (t1, e1) = best[0], best[-1]
    kv = [e.get("kv_cache_fraction") or 0 for _, e in best]
    return {"peak_running": peak, "seconds": round(t1 - t0, 1), "rows": len(best),
            "gen_tok_s": round((e1["generated_tokens"] - e0["generated_tokens"]) / (t1 - t0), 1),
            "running_mean": round(sum(e.get("running") or 0 for _, e in best) / len(best), 1),
            "kv_mean": round(sum(kv) / len(kv), 3)}


def counters(text):
    def val(name):
        m = re.search(rf"^{name}{{[^}}]*}}\s+([-+\d.eE]+)", text, re.M)
        return float(m.group(1)) if m else None
    q, h = val("vllm:prefix_cache_queries_total"), val("vllm:prefix_cache_hits_total")
    d, a = val("vllm:spec_decode_num_drafts_total"), val("vllm:spec_decode_num_accepted_tokens_total")
    return {"prefix_hit": round(h / q, 4) if q else None, "accepted_per_draft": round(a / d, 3) if d else None,
            "mean_acceptance_length": round(1 + a / d, 3) if d else None}


def read(run):
    base = f"{BUCKET}{run}/arena/working/trials/duck-transfer/"
    rows = [json.loads(l) for l in cat(base + "progress.jsonl").splitlines() if l.startswith("{")]
    out = {"run": run, **steady(rows), **counters(cat(base + "vllm-metrics.txt"))}
    roster, tries = score_marks.fetch(run)
    marks = score_marks.score(tries, roster, [25000, 50000])
    out["marks"] = {m: {k: v[k] for k in ("coverage", "score", "levels_per_try", "reportable")} for m, v in marks.items()}
    return out


def main(argv):
    base = argv[argv.index("--base") + 1] if "--base" in argv else None
    runs = [a for a in argv if not a.startswith("--") and a != base]
    res = {r: read(r) for r in ([base] if base else []) + runs}
    for r in runs:
        x = res[r]
        if base and x.get("gen_tok_s") and res[base].get("gen_tok_s"):
            x["ratio_to_base"] = round(x["gen_tok_s"] / res[base]["gen_tok_s"], 4)
    for x in res.values():
        print(json.dumps(x))


if __name__ == "__main__":
    main(sys.argv[1:])
