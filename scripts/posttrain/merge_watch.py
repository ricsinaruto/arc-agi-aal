"""Alert-only watcher for a Nebius level-bench run (merge read): wakes the agent, never stops the run.
usage: merge_watch.py RUN_ID [MAX_MIN=150] | merge_watch.py --backtest PROGRESS TELEMETRY
Rules (5-min passes; silence and throughput need two signals):
  start   no progress row 50 min after launch AND bootstrap.log silent 10 min        -> ALERT
  error   bootstrap.log or bench log shows a Traceback                                 -> ALERT
  silence progress tokens flat 10 min, not final AND generation_tokens_total flat 5 min -> ALERT
  failed  failed attempts >= 3                                                          -> ALERT
  slow    after 10 min of play, cumulative tok/s < 250 with >= 12 running AND
          vLLM generation rate over the last 5 min < 250 tok/s                          -> ALERT
  final   level_bench_final row -> summary, exit 0"""
import json, subprocess, sys, time

B = "nebius-arc:arc-runs-bucket/runs/validation/"


def decide(prog, tele, now, clock, hist):
    """One pass: rows visible at bench time ``now``; ``clock`` the watcher's own seconds (backtest: = now).
    ``hist``: [(clock, completion_tokens, vLLM generation_tokens_total)]. Returns (kind, message) or None."""
    rows = [r for r in prog if r["elapsed_seconds"] <= now]
    if not rows:
        return None
    last = rows[-1]
    if last["event"] == "level_bench_final":
        return ("final", json.dumps({k: last.get(k) for k in ("launched", "done", "clears", "budget_outs", "failed",
                                                                "completion_tokens", "tokens_per_second")}))
    if last.get("failed", 0) >= 3:
        return ("alert", f"failed attempts {last['failed']}")
    seen = [r for r in tele if r["elapsed_seconds"] <= now]
    gen = (seen[-1]["metrics"].get("vllm:generation_tokens_total") or 0.0) if seen else None
    hist.append((clock, last["completion_tokens"], gen))
    old10 = [h for h in hist if h[0] <= clock - 600]
    old5 = [h for h in hist if h[0] <= clock - 300]
    if old10 and last["completion_tokens"] <= old10[-1][1] and (gen is None or old10[-1][2] is None or gen <= old10[-1][2]):
        return ("alert", f"silence: completion tokens flat at {last['completion_tokens']} and vLLM generation flat for 10 min")
    rate = (gen - old5[-1][2]) / (clock - old5[-1][0]) if old5 and gen is not None and old5[-1][2] is not None else None
    if (last["elapsed_seconds"] >= 600 and last.get("running", 0) >= 12 and last.get("tokens_per_second", 0) < 250
            and rate is not None and rate < 250):
        return ("alert", f"slow: {last['tokens_per_second']} tok/s cumulative, vLLM {rate:.0f} tok/s, running {last['running']}")
    return None


def backtest(prog_path, tele_path):
    prog = [json.loads(l) for l in open(prog_path) if l.strip()]
    tele = [json.loads(l) for l in open(tele_path) if l.strip()]
    end = max(prog[-1]["elapsed_seconds"], tele[-1]["elapsed_seconds"]) + 1200
    hist, t = [], 0.0
    while t <= end:
        d = decide(prog, tele, t, t, hist)
        if d:
            return f"{d[0]} at {t:.0f}s: {d[1]}"
        t += 300
    return "quiet"


def cat(path):
    r = subprocess.run(["rclone", "cat", B + path], capture_output=True, timeout=240)
    return r.stdout.decode(errors="replace") if r.returncode == 0 else None


def main(run, max_min):
    began, boot_len, boot_at, hist = time.time(), -1, time.time(), []
    lb = f"{run}/arena/trials/level-bench/"  # collected at the end; live rows are under arena/working/
    live = f"{run}/arena/working/trials/"
    while True:
        mins = (time.time() - began) / 60
        boot = cat(f"{run}/bootstrap.log") or ""
        if len(boot) != boot_len:
            boot_len, boot_at = len(boot), time.time()
        bench_log = (cat(live + "level-bench-process.log") or "") + (cat(lb + "level-bench.log") or "")
        if "Traceback" in boot or "Traceback" in bench_log:
            tail = (boot if "Traceback" in boot else bench_log).split("Traceback")[-1][:600]
            print(f"ALERT {run}: error: Traceback{tail}", flush=True)
            sys.exit(2)
        # the collected copy once it exists (it holds the final row; the live mirror stops before it), else the live one
        text = cat(lb + "progress.jsonl") or cat(live + "level-bench/progress.jsonl")
        prog = [json.loads(l) for l in (text or "").splitlines() if l.strip()]
        tele = [json.loads(l) for l in (cat(lb + "telemetry.jsonl") or cat(live + "level-bench/telemetry.jsonl") or "").splitlines()
                if l.strip()]
        if not prog:
            if mins > 50 and time.time() - boot_at > 600:
                print(f"ALERT {run}: no play {mins:.0f} min after launch, bootstrap.log silent 10 min", flush=True)
                sys.exit(2)
        else:
            d = decide(prog, tele, float("inf"), time.time() - began, hist)
            last = prog[-1]
            print(f"[{mins:.0f}m] done {last['done']}/{last['planned']} running {last['running']} clears {last['clears']} "
                  f"failed {last['failed']} tokens {last['completion_tokens']} tok/s {last['tokens_per_second']}", flush=True)
            if d and d[0] == "final":
                print(f"FINAL {run}: {d[1]}", flush=True)
                sys.exit(0)
            if d:
                print(f"ALERT {run}: {d[1]}", flush=True)
                sys.exit(2)
        if mins > max_min:
            print("watcher: time limit", flush=True)
            sys.exit(1)
        time.sleep(300)


if __name__ == "__main__":
    if sys.argv[1] == "--backtest":
        print(backtest(sys.argv[2], sys.argv[3]))
    else:
        main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 150)
