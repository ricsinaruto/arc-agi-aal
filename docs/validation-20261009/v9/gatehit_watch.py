"""gatehit_watch.py RUN [--backtest SERVER_LOG]: v9 steady-state rules, active from 1 h of play:
  gate: arena gate reserved > 0.95 of the gate capacity (kv_margin x served blocks), 3 consecutive polls (2 min apart);
  hit : GPU prefix hit, mean of the vLLM stats lines over the last 30 min, < 0.94 (needs >= 20 lines).
Prints one ALERT line per rule when it starts firing (re-arms after it clears) and exits 0 when the VM's final.json
or 37,850 s pass. Feeds: arena/live/progress.jsonl (gate) and arena/live/server-tail.log (stats; kept by timestamp)."""
import json
import re
import subprocess
import sys
import time
from datetime import datetime

PAT = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d)[^\n]*Running: (\d+) reqs[^\n]*GPU KV cache usage: ([\d.]+)%, "
                 r"Prefix cache hit rate: ([\d.]+)%")


def stats(text):
    out = {}
    for m in PAT.finditer(text):
        t = datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        out[t] = (int(m.group(2)), float(m.group(3)) / 100, float(m.group(4)) / 100)
    return out


def hit_alerts(rows, play_start, window=1800, bar=0.94, after=3600):
    """(t, mean hit) at each line where the 30-min mean first drops under the bar (re-armed when back above)."""
    fired, out = False, []
    ts = sorted(rows)
    for t in ts:
        if t - play_start < after:
            continue
        w = [rows[u][2] for u in ts if t - window < u <= t]
        if len(w) < 20:
            continue
        m = sum(w) / len(w)
        if m < bar and not fired:
            out.append((t, m))
        fired = m < bar or (fired and m < bar + 0.005)  # hysteresis: re-arm only above bar + 0.005
    return out


if sys.argv[2:3] == ["--backtest"]:
    rows = stats(open(sys.argv[3], errors="replace").read())
    t0 = min(t for t, r in rows.items() if r[0] > 0)
    for t, m in hit_alerts(rows, t0):
        print(f"would fire at {(t - t0) / 3600:.2f} h: 30-min GPU hit {m:.3f}")
    print("lines", len(rows))
    sys.exit(0)

run = sys.argv[1]
P = f"nebius-arc:arc-runs-bucket/runs/validation/{run}"


def cat(path):
    return subprocess.run(["rclone", "cat", f"{P}/{path}"], capture_output=True, text=True).stdout


rows, gate_over, gate_fired, hit_fired, cap, t_start = {}, 0, False, False, None, time.time()
while time.time() - t_start < 37850:
    if cat("nebius/final.json").strip():
        print(f"{time.strftime('%H:%M', time.gmtime())} END {run}: final.json")
        break
    if cap is None:
        try:
            lay = json.loads(cat("arena/trials/duck-transfer/cache-layout.json"))
            study = json.loads(cat("arena/trials/study.json") or "{}")
            cap = float(study.get("kv_margin", 0.96)) * int(lay["num_blocks"])
        except Exception:  # noqa: BLE001 - the layout is synced only at the end: GATE_CAP (from the density rule)
            import os
            cap = float(os.environ["GATE_CAP"]) if os.environ.get("GATE_CAP") else None
    rows.update(stats(cat("arena/live/server-tail.log")))
    lines = [ln for ln in cat("arena/live/progress.jsonl").splitlines() if ln.strip()]
    last = json.loads(lines[-1]) if lines else {}
    elapsed = float(last.get("elapsed_seconds") or 0)
    now = time.strftime("%H:%M", time.gmtime())
    if elapsed >= 3600 and cap and last.get("gate"):
        frac = last["gate"]["reserved"] / cap
        gate_over = gate_over + 1 if frac > 0.95 else 0
        if gate_over >= 3 and not gate_fired:
            print(f"{now} ALERT {run} [gate] reserved {last['gate']['reserved']} = {frac:.3f} of gate capacity {cap:.0f} "
                  f"for 3 polls (waiting {last['gate'].get('waiting')}) at {elapsed / 3600:.2f} h", flush=True)
        gate_fired = gate_over >= 3
    if elapsed >= 3600 and rows:
        play_start = time.time() - elapsed
        recent = [r[2] for t, r in rows.items() if t > max(rows) - 1800]
        if len(recent) >= 20:
            m = sum(recent) / len(recent)
            if m < 0.94 and not hit_fired:
                print(f"{now} ALERT {run} [hit] 30-min GPU prefix hit {m:.3f} < 0.94 at {elapsed / 3600:.2f} h", flush=True)
            hit_fired = m < 0.94 or (hit_fired and m < 0.945)  # hysteresis (backtest: B v3 sat at 0.940)
    time.sleep(120)
