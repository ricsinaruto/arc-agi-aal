"""kvprefix_watch.py RUN [--backtest]: the standing rule for RTX plays (28 Oct ~07:00): KV use >80% after
ramp-up and mean prefix hit ~95%. Every 5 min from 20 min of play: over the last 10 min of live_progress rows, alert
(exit 1) when the KV median is below 0.75 ("roughly 0.8" is good) while the play is full (median active games >= 90% of the most seen; a
draining tail is not a miss), or when the mean vLLM prefix-cache hit rate of the window is below 93%. Exits 0 when the
VM is gone. --backtest: replays the run's whole bootstrap.log + vLLM log at every 5-min tick and prints the first alert."""
import json
import re
import statistics as st
import subprocess
import sys
import time

B = "nebius-arc:arc-runs-bucket/runs/validation"
GONE = "./.claude/worktrees/sft-heal/launch/pth/vm_gone.sh"
KV_MIN, HIT_MIN, START, WINDOW = 0.75, 93.0, 1200, 600  # KV "roughly 0.8", prefix 94-95%


def cat(path):
    return subprocess.run(["rclone", "cat", f"{B}/{path}"], capture_output=True, text=True).stdout


def rows_of(boot):
    return [json.loads(l[l.index("{"):]) for l in boot.splitlines() if '"live_progress"' in l and "{" in l]


def hits_of(log):
    # vLLM's periodic stats lines carry their wall time; map them onto play seconds via the order of lines
    return [(m.group(1), float(m.group(2))) for m in
            re.finditer(r"(\d\d:\d\d:\d\d)[^\n]*Prefix cache hit rate: ([\d.]+)%", log)]


def judge(rows, hits, now_s):
    win = [r for r in rows if now_s - WINDOW <= r["elapsed_seconds"] <= now_s and r.get("kv") is not None]
    if now_s < START or len(win) < 3:
        return None
    most = max((r.get("active") or 0) for r in rows if r["elapsed_seconds"] <= now_s)
    full = st.median(r.get("active") or 0 for r in win) >= 0.9 * most
    kv = st.median(r["kv"] for r in win)
    if full and kv < KV_MIN:
        return f"KV median {kv:.3f} < {KV_MIN} over the last 10 min (play full, {st.median(r.get('running') or 0 for r in win):.0f} running)"
    if hits:
        h = st.mean(x for _, x in hits[-max(1, len(hits) // 10):])
        if h < HIT_MIN:
            return f"prefix hit {h:.1f}% < {HIT_MIN}%"
    return None


run = sys.argv[1]
if "--backtest" in sys.argv:
    rows = rows_of(cat(f"{run}/bootstrap.log"))
    log = cat(f"{run}/arena/serving/vllm-openai-server.log") or cat(f"{run}/arena/live/server-tail.log")
    hits = hits_of(log)
    end = rows[-1]["elapsed_seconds"] if rows else 0
    t = START
    while t <= end:
        frac = t / end
        why = judge(rows, hits[:int(len(hits) * frac)], t)
        if why:
            print(f"BACKTEST {run} ALERT at {t:.0f} s: {why}")
            sys.exit(1)
        t += 300
    print(f"BACKTEST {run} quiet to {end:.0f} s")
    sys.exit(0)
while True:
    if subprocess.run(["bash", GONE, run]).returncode == 0:
        print(f"{run} VM gone")
        sys.exit(0)
    rows = rows_of(cat(f"{run}/bootstrap.log"))
    hits = hits_of(cat(f"{run}/arena/live/server-tail.log"))
    if rows:
        why = judge(rows, hits, rows[-1]["elapsed_seconds"])
        if why:
            print(f"{time.strftime('%H:%M', time.gmtime())} ALERT {run}: {why}")
            sys.exit(1)
    time.sleep(300)
