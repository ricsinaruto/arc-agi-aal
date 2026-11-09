"""Summarize the dfranzen-exact Modal run.

    python scripts/dfr_exact/analyze.py RUN_DIR [--json OUT]

RUN_DIR is ``validation/<run_id>`` from the arc-prolong-runs Volume. Gameplay time 0 is the scorecard's open time
(the gateway). Score: the gateway scorecard's per-game score (arc_agi's official rule: min(sum w_i x min(115,
100 (b_i/a_i)^2), 100 x sum of cleared w_i) / sum w_i, w_i = level index), mean over the 110 instances; it is also
recomputed here from the per-level actions as a check. Throughput: SGLang's generation-token counter from the
/metrics scrape (wall clock) and the engine's own per-batch "gen throughput" log lines; concurrency from
num_running_reqs and the decode lines; prefix-cache hit rate from the prefill lines (cached / (cached + new)).
"""

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

LOG_TIME = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
DECODE = re.compile(r"Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+).*?"
                    r"accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
PREFILL = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#running-req: (\d+)")


def official(env):
    scores, actions, base = env.get("level_scores"), env.get("level_actions"), env.get("level_baseline_actions")
    if not scores:
        return 0.0
    total = weights = cleared = 0.0
    for i, s in enumerate(scores, start=1):
        a, b = actions[i - 1], base[i - 1]
        level = min(115.0, 100.0 * (b / a) ** 2) if s > 0 and a > 0 else 0.0
        total += i * level
        weights += i
        cleared += i if s > 0 else 0
    return min(total / weights, 100.0 * cleared / weights)


def nearest(rows, t):
    before = [r for r in rows if r["t"] <= t]
    return before[-1] if before else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--json")
    args = parser.parse_args()
    run = Path(args.run_dir)
    execution = json.loads((run / "execution.json").read_text()) if (run / "execution.json").exists() else {}

    rows = [json.loads(line) for line in (run / "gateway/scorecards.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "score_mean_110" in r]
    open_at = datetime.fromisoformat(rows[0]["open_at"]).timestamp() if rows and rows[0].get("open_at") else None
    t0 = open_at or (rows[0]["epoch"] if rows else None)
    for r in rows:
        r["t"] = r["epoch"] - t0
    final_path = run / "gateway/scorecard-final.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else None

    def point(r):
        if r is None:
            return None
        envs = r["environments"]
        return {"t": round(r["t"]), "score_pct": round(r["score_mean_110"], 3),
                "score_recomputed_pct": round(sum(official(e) for e in envs) / 110, 3),
                "levels": r["levels"], "actions": r["actions"], "games_played": r["games_played"],
                "games_won": r["games_won"]}

    scores = {"t0_epoch": t0, "at_30min": point(nearest(rows, 1800)), "at_60min": point(nearest(rows, 3600)),
              "last_snapshot": point(rows[-1] if rows else None),
              "series_10min": [point(nearest(rows, s)) for s in range(600, 3601, 600)]}
    if final:
        summ = final["_summary"]
        scores["final_card"] = {"t": round(final["_closed_epoch"] - t0), "score_pct": round(summ["score_mean_110"], 3),
                                "levels": summ["levels"], "actions": summ["actions"],
                                "games_played": summ["games_played"], "games_won": summ["games_won"]}

    # Engine counters (wall clock).
    samples = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()] \
        if (run / "metrics.jsonl").exists() else []

    def series(prefix):
        out = []
        for s in samples:
            vals = [v for k, v in s["m"].items() if k.split("{")[0] == prefix]
            if vals:
                out.append((s["epoch"] - t0, sum(vals)))
        return out

    names = sorted({k.split("{")[0] for s in samples[-1:] for k in s["m"]}) if samples else []
    gen = series("sglang:generation_tokens_total")
    prompt = series("sglang:prompt_tokens_total")
    cached = series("sglang:cached_tokens_total")
    running = series("sglang:num_running_reqs")
    queue = series("sglang:num_queue_reqs")

    def rate(ser, a, b):
        pts = [p for p in ser if a <= p[0] <= b]
        if len(pts) < 2:
            return None
        return round((pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]), 1)

    def mean(ser, a, b):
        pts = [v for t, v in ser if a <= t <= b]
        return round(statistics.mean(pts), 2) if pts else None

    windows = [(0, 1800), (1800, 3600), (0, 3600)]
    engine = {f"{a}-{b}": {"generated_tok_s_wall": rate(gen, a, b), "prompt_tok_s_wall": rate(prompt, a, b),
                           "running_mean": mean(running, a, b), "queue_mean": mean(queue, a, b)}
              for a, b in windows}
    engine["per_5min_generated_tok_s"] = [rate(gen, a, a + 300) for a in range(0, 3600, 300)]
    engine["per_5min_running_mean"] = [mean(running, a, a + 300) for a in range(0, 3600, 300)]
    if cached and prompt:
        for a, b in windows:
            p0 = [v for t, v in prompt if a <= t <= b]
            c0 = [v for t, v in cached if a <= t <= b]
            if len(p0) > 1 and len(c0) > 1 and p0[-1] > p0[0]:
                engine[f"{a}-{b}"]["cache_hit_counter"] = round((c0[-1] - c0[0]) / (p0[-1] - p0[0]), 3)
    engine["metric_names"] = names

    # The engine log (decode and prefill lines).
    log = {}
    serve = run / "working/serve.log"
    if not serve.exists():
        serve = run / "serve.log"
    if serve.exists():
        dec, pre = [], []
        for line in serve.read_text(errors="replace").splitlines():
            m = LOG_TIME.match(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            d = DECODE.search(line)
            if d:
                dec.append((t, int(d.group(1)), float(d.group(4)), float(d.group(5)), int(d.group(6))))
                continue
            p = PREFILL.search(line)
            if p:
                pre.append((t, int(p.group(2)), int(p.group(3))))
        # serve.log stamps are the container's wall clock (UTC on Modal).
        for label, (a, b) in (("0-1800", (0, 1800)), ("1800-3600", (1800, 3600)), ("0-3600", (0, 3600))):
            ds = [x for x in dec if a <= x[0] - t0 <= b]
            ps = [x for x in pre if a <= x[0] - t0 <= b]
            new, hit = sum(x[1] for x in ps), sum(x[2] for x in ps)
            log[label] = {
                "decode_lines": len(ds),
                "engine_gen_tok_s_mean": round(statistics.mean(x[3] for x in ds), 1) if ds else None,
                "running_mean": round(statistics.mean(x[1] for x in ds), 2) if ds else None,
                "running_max": max((x[1] for x in ds), default=None),
                "queue_mean": round(statistics.mean(x[4] for x in ds), 2) if ds else None,
                "accept_len_mean": round(statistics.mean(x[2] for x in ds), 3) if ds else None,
                "prefill_new_tokens": new, "prefill_cached_tokens": hit,
                "cache_hit_rate": round(hit / (hit + new), 3) if hit + new else None,
            }
        text = serve.read_text(errors="replace")
        log["retracted_or_preempted_lines"] = len(re.findall(r"(?i)retract|preempt", text))
        log["errors"] = len(re.findall(r"(?i)Traceback|CUDA error|out of memory", text))

    summary = {"run": run.name, "execution": execution, "scores": scores, "engine_metrics": engine,
               "engine_log": log}
    text = json.dumps(summary, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
