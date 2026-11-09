"""Window choice for the three-block run (hero-v2-blocks, 29 September; CPU only).

User: 37 streams, ``max_num_seqs`` 41, the window as large as possible while 37 streams pass the gate-replay bar
(<= 1% of stream time lost to gate waits, p90 wait <= 2 s), with steady residency (every game plays the whole block,
no parking churn), and the prefix cache must also hold the ~37 resident contexts between turns.

The closed-loop exact-gate replay of ``gate_replay.py`` with three changes:

- **steady streams**: every stream plays one recorded game's phase-2 requests in order, its tenures concatenated
  without the parks between them (no ``moves_on``: nobody parks mid-request, no tail gaps); when a game's trace ends
  the stream continues with another game's (a uniform draw), the first entered at a random request;
- **prompt scale to the target window**: every request's prompt part (requested - cap) is scaled by
  (W - cap) / (the recorded window - cap), so the recorded largest request maps onto W (run A 64K x 1.67 for 98,304;
  run B 130,816 x 0.72); the reply's final length scales the same way. The summary floor scales too (conservative for
  A);
- **residency**: besides the gate's reservations it tracks the cache the contexts occupy: every in-flight request's
  final length (``used``) plus every idle or waiting stream's resident context (its last request's final length:
  prompt + reply, the prefix its next request reuses). Occupancy above the served pool (``num_blocks``, not the 0.9
  gate) means some idle contexts are evicted and re-prefilled; the table gives the time share and the mean excess.

Holds scale with the decode step model t(n) = 16.3 + 0.32 n ms (the 48-stream check, prefix-unit-int4.md section 5)
from the recorded run's own focus streams (A 37, B 24) to 37; gaps (tool work) do not scale.

    PYTHONPATH=src uv run python scripts/arena_analysis/blocks_window_replay.py --run A=<trial dir> --run B=<trial dir> \
        --layout <int4 cache-layout.json> --blocks 1268 --windows 98304 94208 90112 86016 81920 --out result.json

``--lockstep`` (30 September; the blocks campaign starts every block's games together, stagger 0): instead of entering
a random request of a steady stream, every stream starts at t = 0 with the first request of a distinct game (drawn
without replacement while enough games exist), so the contexts grow together from the block start into the gate and
the compaction trigger; nothing is excluded as warm-up and only the ``all`` pool is replayed. Trace ``L`` is the first
three-block run (``arena-hero-v2-blocks-v1``: 37 games admitted together at 93,184 tokens, no opening), the recorded
lockstep block start itself; A and B under ``--lockstep`` start their first phase-2 requests together.
"""

import argparse
import copy
import heapq
import importlib.util
import json
import math
import random
import statistics
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate_replay", HERE / "gate_replay.py")
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)

CAP = 16384
STEP_MS = (16.3, 0.32)
REFERENCE = {"A": 37, "B": 24, "L": 37}
RECORDED_WINDOW = {"A": 65536, "B": 130816, "L": 93184}


def slowdown(streams, reference):
    a, b = STEP_MS
    return (a + b * streams) / (a + b * reference)


def steady_games(report, budget, since=None):
    """Per game: its phase-2 requests in order as (gap, blocks, hold, purpose, used), tenures concatenated."""
    tenures = gr.phase2_tenures(report, budget, since=since)
    by_game = defaultdict(list)
    for t in sorted(tenures, key=lambda t: t["start"]):
        items = [(gap if k else min(gap, 5.0), blocks, hold, purpose, used)
                 for k, (gap, blocks, hold, purpose, moves_on, used) in enumerate(t["requests"])]
        by_game[t["game"]].extend(items)
    return [v for v in by_game.values() if v]


def stream(games, rng, factor, first_game=None):
    """``first_game``: the lockstep start (the stream begins with this game's first request)."""
    first = True
    while True:
        items = games[first_game] if first and first_game is not None else rng.choice(games)
        begin = rng.randrange(len(items)) if first and first_game is None else 0
        first = False
        for gap, blocks, hold, purpose, used in items[begin:]:
            yield gr.Request(blocks, hold * factor, gap=gap, tag=purpose, used=used)


def run(streams, capacity, slots, pool, until, start=0.0):
    """GateSim.run with residency: returns admitted requests and the series (t, reserved, inflight, waiting,
    occupancy) where occupancy = in-flight final lengths + idle/waiting streams' resident contexts."""
    heap, seq = [], 0
    iters = [iter(s) for s in streams]
    resident = [0] * len(iters)
    done, queue = [], deque()
    reserved = inflight = used = 0
    series = [(start, 0, 0, 0, 0)]

    def schedule_next(index, after):
        nonlocal seq
        request = next(iters[index])
        request.arrival = after + request.gap
        if request.arrival < until:
            seq += 1
            heapq.heappush(heap, (request.arrival, seq, 1, (index, request)))

    for index in range(len(iters)):
        schedule_next(index, start)
    while heap:
        now, _, kind, (index, request) = heapq.heappop(heap)
        if kind == 0:
            reserved -= request.blocks
            used -= request.used
            inflight -= 1
            resident[index] = request.used
            schedule_next(index, now)
        else:
            queue.append((index, request))
        while queue and inflight < slots and reserved + queue[0][1].blocks <= capacity:
            i, head = queue.popleft()
            head.admitted = now
            head.released = now + head.hold
            reserved += head.blocks
            used += head.used
            inflight += 1
            resident[i] = 0  # the prefix is part of the in-flight request now
            done.append(head)
            seq += 1
            heapq.heappush(heap, (head.released, seq, 0, (i, head)))
        occupancy = used + sum(resident)
        point = (now, reserved, inflight, len(queue), occupancy)
        if series[-1][0] == now:
            series[-1] = point
        else:
            series.append(point)
    return done, series


def lockstep_starts(games, rng, streams):
    """Distinct first games for the streams (with replacement only beyond the number of games)."""
    order = list(range(len(games)))
    rng.shuffle(order)
    return [order[i % len(order)] for i in range(streams)]


def replay(games, capacity, pool, streams, slots, factor, seeds, duration, warmup=300.0, lockstep=False):
    waits, lost, shares, over, excess, occ, occ99, running, reserved = [], [], [], [], [], [], [], [], []
    lo, end = (0.0 if lockstep else warmup), duration
    for seed in range(seeds):
        rng = random.Random(seed)
        firsts = lockstep_starts(games, rng, streams) if lockstep else [None] * streams
        done, series = run([stream(games, rng, factor, first) for first in firsts], capacity, slots, pool, end)
        measured = [q for q in done if q.arrival >= lo]
        w = [q.wait for q in measured]
        waits.extend(w)
        lost.append(sum(w) / (streams * (end - lo)))
        shares.append(gr.time_share(series, lo, end, lambda r, i, q: q > 0))
        over.append(_share_over(series, lo, end, pool))
        excess.append(_mean_excess(series, lo, end, pool))
        occ.append(gr.time_mean(series, lo, end, 3))
        occ99.append(gr.time_quantile(series, lo, end, 3, 0.99))
        running.append(gr.time_mean(series, lo, end, 1))
        reserved.append(gr.time_mean(series, lo, end, 0))
    return {
        "streams": streams, "max_num_seqs": slots, "hold_factor": round(factor, 4), "seeds": seeds,
        "lockstep": lockstep, "measured_from_seconds": lo,
        "stream_time_lost_to_waits": round(statistics.mean(lost), 4),
        "waits": gr.summarize_waits(waits),
        "time_share_waiting": round(statistics.mean(shares), 4),
        "mean_running": round(statistics.mean(running), 2),
        "reserved_blocks_mean": round(statistics.mean(reserved), 1),
        "occupancy_blocks": {"mean": round(statistics.mean(occ), 1), "p99": round(statistics.mean(occ99), 1),
                             "pool": pool},
        "time_share_occupancy_over_pool": round(statistics.mean(over), 4),
        "mean_excess_blocks_when_over": round(statistics.mean(excess), 1),
        "passes": statistics.mean(lost) <= 0.01 and gr.summarize_waits(waits)["p90_seconds"] <= 2.0,
    }


def _share_over(series, lo, end, pool):
    total = 0.0
    for (t0, r, i, q, o), (t1, *_ ) in zip(series, series[1:] + [(end, 0, 0, 0, 0)]):
        a, b = max(t0, lo), min(t1, end)
        if b > a and o > pool:
            total += b - a
    return total / (end - lo)


def _mean_excess(series, lo, end, pool):
    total = weight = 0.0
    for (t0, r, i, q, o), (t1, *_ ) in zip(series, series[1:] + [(end, 0, 0, 0, 0)]):
        a, b = max(t0, lo), min(t1, end)
        if b > a and o > pool:
            total += (b - a) * (o - pool)
            weight += b - a
    return total / weight if weight else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="NAME=trial dir (A or B)")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--blocks", type=int, default=1268)
    parser.add_argument("--margin", type=float, default=0.9)
    parser.add_argument("--output-cap", type=int, default=CAP)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
    parser.add_argument("--streams", type=int, nargs="+", default=[37])
    parser.add_argument("--spare", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--duration", type=float, default=7200.0)
    parser.add_argument("--floors", type=int, nargs="+", default=[0, 8800],
                        help="prompt tokens that do not scale (0: scale the whole prompt; 8,800: the median "
                             "post-compaction prompt of runs A and B)")
    parser.add_argument("--hold-scale", type=float, default=1.0,
                        help="multiplies every hold (sensitivity only, e.g. a faster decode path; default 1.0)")
    parser.add_argument("--lockstep", action="store_true",
                        help="every stream starts a distinct game's first request at t = 0 (no warm-up)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    import sys

    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from arena.cache import BlockBudget

    layout = copy.deepcopy(json.loads(Path(args.layout).read_text()))
    layout["num_blocks"] = args.blocks
    budget = BlockBudget(layout, args.margin)
    result = {"layout": args.layout, "num_blocks": args.blocks, "capacity_blocks": budget.capacity,
              "step_ms": list(STEP_MS), "spare": args.spare, "margin": args.margin, "output_cap": args.output_cap,
              "duration": args.duration, "lockstep": args.lockstep,
              **({"hold_scale": args.hold_scale} if args.hold_scale != 1.0 else {}),
              "runs": dict(spec_.split("=", 1) for spec_ in args.run), "rows": []}
    for spec_ in args.run:
        name, path = spec_.split("=", 1)
        report, _ = gr.load_run(path)
        for window, floor in [(w, f) for w in args.windows for f in args.floors]:
            factor = (window - args.output_cap - floor) / (RECORDED_WINDOW[name] - CAP - floor)
            report["_prompt_scale"] = (factor, CAP, floor, args.output_cap)
            cut = budget  # the scale maps the largest recorded request onto the window
            pools = {"all": steady_games(report, cut)}
            if not args.lockstep:
                pools["late"] = steady_games(report, cut, since=3600.0)
            for pool_name, games in pools.items():
                for n in args.streams:
                    row = replay(games, budget.capacity, args.blocks, n, n + args.spare,
                                 slowdown(n, REFERENCE[name]) * args.hold_scale, args.seeds, args.duration, lockstep=args.lockstep)
                    row = {"trace": name, "window": window, "prompt_scale": round(factor, 4), "floor": floor,
                           "pool": pool_name,
                           "games": len(games), "full_window_blocks": budget.cost(window), **row}
                    result["rows"].append(row)
                    print(json.dumps({k: row[k] for k in ("trace", "window", "floor", "pool", "streams",
                                                          "stream_time_lost_to_waits", "passes",
                                                          "time_share_occupancy_over_pool", "mean_running",
                                                          "reserved_blocks_mean")}
                                     | {"p90": row["waits"]["p90_seconds"], "occ": row["occupancy_blocks"]}),
                          flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
