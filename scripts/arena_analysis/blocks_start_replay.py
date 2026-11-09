"""Lockstep vs staggered block starts, gate margin 0.90 vs 0.96 (hero-v2-blocks next build; CPU).

The first blocks run started its 37 games together; their contexts grew in lockstep and filled the gate
(1,118-1,141 of 1,141 reserved, 31-34 in flight, 3-6 waiting from ~1,300 s). This replays that run's own requests
through the exact gate (``gate_replay.GateSim`` semantics: FIFO, head-of-line, ``max_num_seqs`` slots, reservations
<= capacity) as closed loops:

- a stream is a slot: it plays the requests of the game that first held it (in the recorded order, with the recorded
  gaps from a release to the next arrival and the recorded holds), and when that game stops (retired, parked or the
  wall), the next game admitted into a freed slot continues on it (fillers in their admission order);
- ``--stagger S``: stream i starts at i x S / streams seconds (the next build: 600 s) instead of all at 0;
- ``--margins``: the gate capacity is int(margin x pool);
- holds scale with the decode step model t(n) = 16.3 + 0.32 n ms from the recorded mean running to the replayed one
  (one fixed-point pass); gaps (tools, observations) do not scale.

The recorded run itself (lockstep, 0.90) is the check: the replayed in-flight and waiting counts are compared with
its progress rows.

    uv run python scripts/arena_analysis/blocks_start_replay.py <trial dir> --stagger 0 600 --margins 0.90 0.96
"""

import argparse
import heapq
import importlib.util
import json
import statistics
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate_replay", HERE / "gate_replay.py")
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)
STEP_MS = (16.3, 0.32)


def slot_streams(report, budget):
    """Per slot, the recorded requests (gap, blocks, hold, used) of the games that held it in turn."""
    rows = gr.recorded_attempts(report, budget)
    by_game = {}
    for r in sorted(rows, key=lambda r: r["admitted"]):
        by_game.setdefault(r["game"], []).append(r)
    admits = []
    for g in report["games"]:
        for e in g["admissions"]:
            if e["event"] == "admit":
                admits.append((e["wall_seconds"], g["game_id"], e["reason"]))
                break
    admits.sort()
    first = [g for t, g, reason in admits if reason == "block_start" and t < 1.0]
    later = [g for t, g, reason in admits if not (reason == "block_start" and t < 1.0)]
    streams = [[g] for g in first]
    # Fillers take slots in admission order; assign each to the stream whose game stopped earliest.
    stops = {}
    for g in report["games"]:
        parks = [e["wall_seconds"] for e in g["admissions"] if e["event"] in ("park", "finish")]
        stops[g["game_id"]] = parks[0] if parks else float("inf")
    free = sorted(((stops[s[-1]], i) for i, s in enumerate(streams)))
    for game in later:
        end, i = free.pop(0)
        streams[i].append(game)
        heapq.heappush(free, (stops[game], i))
        free.sort()
    out = []
    for games in streams:
        items = []
        for game in games:
            reqs = by_game.get(game, [])
            prev = None
            for r in reqs:
                gap = 0.0 if prev is None else max(0.0, r["arrival"] - prev)
                items.append((gap, r["blocks"], max(0.0, r["released"] - r["admitted"]), r["used_blocks"]))
                prev = r["released"]
        out.append(items)
    return out, first


def run(streams, capacity, slots, offsets, factor, until):
    heap, seq = [], 0
    iters = [iter(s) for s in streams]
    queue = deque()
    reserved = inflight = 0
    series = [(0.0, 0, 0, 0, 0)]
    waits = []

    def schedule_next(index, after):
        nonlocal seq
        item = next(iters[index], None)
        if item is None:
            return
        gap, blocks, hold, used = item
        t = after + gap
        if t < until:
            seq += 1
            heapq.heappush(heap, (t, seq, 1, (index, [blocks, hold * factor, t])))

    for index, offset in enumerate(offsets):
        schedule_next(index, offset)
    while heap:
        now, _, kind, (index, req) = heapq.heappop(heap)
        if kind == 0:
            reserved -= req[0]
            inflight -= 1
            schedule_next(index, now)
        else:
            queue.append((index, req))
        while queue and inflight < slots and reserved + queue[0][1][0] <= capacity:
            i, head = queue.popleft()
            waits.append(now - head[2])
            reserved += head[0]
            inflight += 1
            seq += 1
            heapq.heappush(heap, (now + head[1], seq, 0, (i, head)))
        point = (now, reserved, inflight, len(queue), 0)
        if series[-1][0] == now:
            series[-1] = point
        else:
            series.append(point)
    return series, waits


def summary(series, waits, lo, hi, streams):
    return {"mean_inflight": round(gr.time_mean(series, lo, hi, 1), 2),
            "mean_waiting": round(gr.time_mean(series, lo, hi, 2), 2),
            "time_share_waiting": round(gr.time_share(series, lo, hi, lambda r, i, w: w > 0), 4),
            "mean_reserved": round(gr.time_mean(series, lo, hi, 0), 1),
            "stream_time_lost": round(sum(waits) / (streams * (hi - lo)), 4),
            "waits": gr.summarize_waits(waits)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_dir")
    parser.add_argument("--stagger", type=float, nargs="+", default=[0.0, 600.0])
    parser.add_argument("--margins", type=float, nargs="+", default=[0.90, 0.96])
    parser.add_argument("--slots", type=int, default=41)
    parser.add_argument("--until", type=float, default=7200.0)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report, budget = gr.load_run(args.trial_dir)
    pool = int(budget.layout["num_blocks"])
    streams, first = slot_streams(report, budget)
    n = len(streams)
    rows = [json.loads(line) for line in (Path(args.trial_dir) / "progress.jsonl").read_text().splitlines() if line]
    recorded = {"mean_inflight": round(statistics.mean(r["gate"]["inflight"] for r in rows), 2),
                "mean_waiting": round(statistics.mean(r["gate"]["waiting"] for r in rows), 2),
                "rows_waiting_share": round(sum(r["gate"]["waiting"] > 0 for r in rows) / len(rows), 4),
                "mean_reserved": round(statistics.mean(r["gate"]["reserved"] for r in rows), 1)}
    base_running = recorded["mean_inflight"]
    result = {"trial_dir": args.trial_dir, "pool": pool, "streams": n, "slots": args.slots, "recorded": recorded,
              "rows": []}
    for margin in args.margins:
        for stagger in args.stagger:
            offsets = [i * stagger / n for i in range(n)]
            factor = 1.0
            for _ in range(2):  # hold slowdown from the recorded mean running to the replayed one
                series, waits = run(streams, int(margin * pool), args.slots, offsets, factor, args.until)
                running = gr.time_mean(series, 300.0, args.until, 1)
                factor = (STEP_MS[0] + STEP_MS[1] * running) / (STEP_MS[0] + STEP_MS[1] * base_running)
            row = {"margin": margin, "capacity": int(margin * pool), "stagger_seconds": stagger,
                   "hold_factor": round(factor, 4),
                   "all": summary(series, waits, 0.0, args.until, n),
                   "first_hour": summary(series, [], 0.0, 3600.0, n),
                   "second_hour": summary(series, [], 3600.0, args.until, n)}
            for lo, hi, key in ((1200.0, 2400.0, "lockstep_window_1200_2400"),):
                row[key] = summary(series, [], lo, hi, n)
            result["rows"].append(row)
            print(json.dumps({"margin": margin, "stagger": stagger, "factor": row["hold_factor"],
                              **{k: row["all"][k] for k in ("mean_inflight", "mean_waiting", "time_share_waiting",
                                                            "stream_time_lost")},
                              "p90": row["all"]["waits"]["p90_seconds"],
                              "w1200_2400": row["lockstep_window_1200_2400"]["mean_inflight"]}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
