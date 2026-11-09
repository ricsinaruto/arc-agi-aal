"""Admission CPU gate (<= 5 min before each B300 launch). The real SharedSlots and Scheduler on
a fake clock, configured from the exact bundle's mg (active_tries, timeshare.active_tries, admit_stagger_s,
active_kv_fill, kv_project, v7 in_flight / in_flight_mode / ramp_s / max_ready, group size), play recorded tries
from a KV trace (kv_replay's format: per request [prompt, completion, cached]) at a per-stream decode rate. Reports
live tries and server-1 KV (client tokens / pool tokens) per minute; the gate checks both at 10 min.

    python -m arena.admission_sim BUNDLE --trace TRACE --live MIN_LIVE --kv-max MAX_KV [--pool-tokens N]"""

import argparse
import json
import random
import sys
from pathlib import Path


class SimTry:
    """What SharedSlots' KV gate reads of a live attempt."""

    def __init__(self, reqs, group, window, cap):
        self.reqs, self.i, self.prefix_key = reqs, 0, group
        self.floor_tokens = int(reqs[0][0])
        self.next_estimate, self.window, self.cap = int(reqs[0][0]), int(window), int(cap)
        self.budget = sum(int(c) for _, c, _ in reqs)
        self.generated = 0

    def progress(self):
        return self.generated, self.budget


def simulate(study, trace, minutes=30, tok_s=20.0, gap_s=1.0, pool_tokens=5_400_000, server2_at=600.0, seed=0,
             reads=0, read_tries=4, read_trace=None):
    """``reads``: held-out read benches of ``read_tries`` tries each from ``read_trace`` at t0, outranking every group
    (v7 oldest_first), pool class "read" (v7.read_kv_share caps them at read_share of the KV gate)."""
    from arena import level_bench  # absolute: the gate runs this file against a bundle's own arena package
    from arena.async_rl import Scheduler

    mg = study["rl_loop"]["mg"]
    v7 = mg.get("v7") or {}
    clock = [1e9]
    saved = level_bench.time.time
    level_bench.time.time = lambda: clock[0]
    try:
        stagger = float(mg.get("admit_stagger_s") or 0)
        project = mg.get("kv_project")
        p1 = level_bench.SharedSlots(int(mg.get("active_tries") or 80), mg.get("active_kv_fill"), project=project,
                                     stagger_s=stagger)
        p2 = level_bench.SharedSlots(int((mg.get("timeshare") or {}).get("active_tries") or 0), mg.get("active_kv_fill"),
                                     project=project, stagger_s=stagger)
        for p in (p1, p2):
            p.capacity = int(pool_tokens)
        p2.active = False
        size = int(v7.get("group_size") or mg.get("group_size") or 12)
        share = float(v7.get("read_share") or 0) if v7.get("read_kv_share") else 0.0
        for p in (p1, p2):
            if share > 0:
                p.shares["read"] = share
        sched = Scheduler(size, int(v7.get("max_ready") or 8), float(v7.get("ramp_s") or 30), int(v7.get("in_flight") or 110))
        unfinished = (v7.get("in_flight_mode") or "planned") == "unfinished"
        window = int(study.get("server", {}).get("max_model_len") or 131072)
        tries = [t for t in trace["tries"] if t["reqs"]]
        rng = random.Random(seed)
        benches, rows, n = {}, [], [0]

        def new_group():
            n[0] += 1
            key = f"g{n[0]:05d}"
            t = rng.choice(tries)
            benches[key] = {"left": size, "running": [], "done": 0, "group": t["group"] + key,
                            "pick": [rng.choice(tries) for _ in range(size)]}
            for p in (p1, p2):
                p.want(key, rng.randrange(20000, 130000), size)

        for i in range(int(reads)):
            key = f"r{i:03d}"
            rt = [t for t in (read_trace or trace)["tries"] if t["reqs"]]
            benches[key] = {"left": int(read_tries), "running": [], "done": 0, "group": f"read{i}", "read": True,
                            "pick": [rng.choice(rt) for _ in range(int(read_tries))]}
            for p in (p1, p2):
                p.classify(key, "read")
                p.want(key, 10 ** 6, int(read_tries))
        t0 = clock[0]
        for tick in range(int(minutes * 60)):
            now = clock[0] = t0 + tick
            if tick >= server2_at and p2.slots:
                p2.active = True
            planned = sum(size if not unfinished else size - b["done"] for b in benches.values() if not b.get("read"))
            if sched.want(now, p1.queued(), planned, 0):
                new_group()
                sched.made(now)
            for key, b in list(benches.items()):
                for r in list(b["running"]):
                    s, pool = r
                    if now >= s.next_at:
                        s.generated += int(s.reqs[s.i][1])
                        s.i += 1
                        if s.i >= len(s.reqs):
                            b["running"].remove(r)
                            pool.give(s)
                            b["done"] += 1
                            continue
                        s.next_estimate = int(s.reqs[s.i][0])
                        s.next_at = now + int(s.reqs[s.i][1]) / tok_s + gap_s
                while b["left"] > 0:
                    pool = next((p for p in (p1, p2) if p.take(key)), None)
                    if pool is None:
                        break
                    t = b["pick"][len(b["pick"]) - b["left"]]
                    s = SimTry(t["reqs"], b["group"], window, 16384)
                    s.next_at = now + int(s.reqs[0][1]) / tok_s + gap_s
                    pool.enter(s, key)
                    b["left"] -= 1
                    for q in (p1, p2):
                        q.remaining(key, b["left"])
                    b["running"].append((s, pool))
                if b["left"] == 0 and not b["running"]:
                    for q in (p1, p2):
                        q.done(key)
                    benches.pop(key)
            if tick % 60 == 59:
                kv1 = sum(max(s.next_estimate, s.floor_tokens) for s in p1.live) / pool_tokens
                rows.append({"minute": (tick + 1) // 60, "live1": p1.used, "live2": p2.used, "kv1": round(kv1, 3),
                             "groups": sum(1 for b in benches.values() if not b.get("read")),
                             "group_live": sum(1 for p in (p1, p2) for s in p.live if getattr(s, "pool_cls", None) != "read"),
                             "read_live": sum(1 for p in (p1, p2) for s in p.live if getattr(s, "pool_cls", None) == "read"),
                             "projected1": round(p1.load() / pool_tokens, 3)})
        return rows
    finally:
        level_bench.time.time = saved


def check(rows, live_min, kv_max, minute=10):
    row = next(r for r in rows if r["minute"] == minute)
    live = row["live1"] + row["live2"]
    ok = live >= live_min and row["kv1"] <= kv_max
    return ok, f"{'pass' if ok else 'FAIL'}: {minute} min live {live} (>= {live_min}), server-1 KV {row['kv1']} (<= {kv_max})"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--live", type=int, required=True)
    ap.add_argument("--kv-max", type=float, required=True)
    ap.add_argument("--pool-tokens", type=int, default=5_400_000)  # B300 hero pool (rl1-screen-drain-2 log: 5,437,348 tokens)
    ap.add_argument("--tok-s", type=float, default=20.0)
    ap.add_argument("--minutes", type=int, default=30)
    a = ap.parse_args(argv)
    study = json.loads((a.bundle / "configs/comparison.json").read_text())
    rows = simulate(study, json.loads(a.trace.read_text()), minutes=a.minutes, tok_s=a.tok_s, pool_tokens=a.pool_tokens)
    for r in rows:
        if r["minute"] in (1, 2, 5, 10, 15, 20, 30):
            print(json.dumps(r))
    ok, msg = check(rows, a.live, a.kv_max)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
