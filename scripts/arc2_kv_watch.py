"""arc2_kv_watch.py JSONL [--backtest] [--after S]: arm-side live rules beside watch.py for arc2 RTX plays, read from
watch.py's arena snapshots (scratchpad/watch/<run>.arena.jsonl, one per 5 min). Exits 1 with the alert on:
- any vLLM preemption between two snapshots (engine.preemptions rose), or a KV reading or gate peak above 0.95
  (the optimistic charge can make vLLM preempt);
- rule b (28 Oct ~07:00), from 20 min of play over the last 10 min: KV median below 0.80 while the gate has
  requests waiting (a supply-limited tail is not a miss), or mean prefix hit below 0.93; while requests
  wait, 20M projected past 10 h at the last 30 min's rate (the 43,200 s wall is a crash ceiling only).
--after S: ignore alerts at or before S s of play (restart after an alert was handled). Exits 0 when the trial ends
(no new snapshot for 30 min). --backtest: replays the file, prints the first alert per rule, never waits."""
import json
import statistics as st
import sys
import time
from pathlib import Path

KV_MIN, KV_MAX, HIT_MIN, START, WINDOW = 0.80, 0.95, 0.93, 1200, 600
STOP, ETA_MAX = 20_000_000, 36_000


def snaps(path):
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            s = json.loads(line)["snap"]
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(s, dict) and s.get("elapsed_seconds") is not None and s.get("engine"):
            out.append(s)
    return out


def judge(rows, i):
    """Alerts at snapshot i (rows sorted by play time)."""
    s, prev = rows[i], rows[i - 1] if i else None
    e, g, t = s["engine"], s.get("gate") or {}, s["elapsed_seconds"]
    out = []
    if prev and (e.get("preemptions") or 0) > ((prev.get("engine") or {}).get("preemptions") or 0):
        out.append(("preempt", f"preemptions {prev['engine'].get('preemptions')} -> {e['preemptions']}"))
    peak = max(e.get("kv_cache_fraction") or 0, g.get("peak_kv") or 0)
    if peak > KV_MAX:
        out.append(("kv_high", f"KV {peak:.3f} > {KV_MAX}"))
    win = [r for r in rows[:i + 1] if t - WINDOW <= r["elapsed_seconds"] <= t]
    if t >= START and len(win) >= 2:
        kv = st.median(r["engine"].get("kv_cache_fraction") or 0 for r in win)
        waiting = st.median((r.get("gate") or {}).get("waiting") or 0 for r in win)
        if kv < KV_MIN and waiting > 0:
            out.append(("kv_low", f"KV median {kv:.3f} < {KV_MIN} with {waiting:.0f} waiting at the gate "
                                  f"(inflight {g.get('inflight')}, target {g.get('target')})"))
        # the 43,200 s wall is a crash ceiling: alert while the play is full if 20M is projected past ~10 h
        back = [r for r in rows[:i + 1] if r["elapsed_seconds"] >= t - 1800]
        done = sum((s.get("arm_tokens") or {}).values())
        if waiting > 0 and len(back) >= 2 and back[-1]["elapsed_seconds"] > back[0]["elapsed_seconds"]:
            rate = (done - sum((back[0].get("arm_tokens") or {}).values())) / (
                back[-1]["elapsed_seconds"] - back[0]["elapsed_seconds"])
            eta = t + (STOP - done) / rate if rate > 0 else float("inf")
            if done < STOP and eta > ETA_MAX:
                out.append(("eta", f"20M projected at {eta / 3600:.1f} h ({rate:.0f} tok/s over 30 min, "
                                   f"{done / 1e6:.2f}M done) > {ETA_MAX / 3600:.0f} h"))
        hits = [r["engine"]["prefix_hit_rate"] for r in win if r["engine"].get("prefix_hit_rate") is not None]
        if hits and st.mean(hits) < HIT_MIN:
            out.append(("hit_low", f"prefix hit {st.mean(hits):.3f} < {HIT_MIN}"))
    return out


path = sys.argv[1]
after = float(sys.argv[sys.argv.index("--after") + 1]) if "--after" in sys.argv else -1.0
if "--eta-hours" in sys.argv:  # the launcher's play cap (Nebius: 25,200 s) sits below the default 10 h
    ETA_MAX = 3600 * float(sys.argv[sys.argv.index("--eta-hours") + 1])
if "--backtest" in sys.argv:
    rows, seen = sorted(snaps(path), key=lambda r: r["elapsed_seconds"]), set()
    for i in range(len(rows)):
        for rule, why in judge(rows, i):
            if rule not in seen:
                seen.add(rule)
                print(f"BACKTEST {Path(path).name} {rule} at {rows[i]['elapsed_seconds']:.0f} s: {why}")
    print(f"BACKTEST {Path(path).name} rules fired: {sorted(seen) or 'none'} over {len(rows)} snapshots")
    sys.exit(0)
last_new = time.time()
count = 0
while True:
    rows = sorted(snaps(path), key=lambda r: r["elapsed_seconds"]) if Path(path).exists() else []
    if len(rows) > count:
        count, last_new = len(rows), time.time()
        alerts = [x for x in judge(rows, len(rows) - 1)] if rows[-1]["elapsed_seconds"] > after else []
        if alerts:
            print(f"{time.strftime('%H:%M', time.gmtime())} ALERT {Path(path).name} at "
                  f"{rows[-1]['elapsed_seconds']:.0f} s: " + "; ".join(why for _, why in alerts))
            sys.exit(1)
    elif time.time() - last_new > 1800:
        print(f"{Path(path).name}: no new snapshot for 30 min; ending")
        sys.exit(0)
    time.sleep(60)
