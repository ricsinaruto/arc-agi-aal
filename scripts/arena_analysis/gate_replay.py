"""Replay of the exact KV admission gate on a recorded arena run, at the recorded and at more focus streams
(29 September, CPU only; docs/validation-20260927/hero-v2-analysis/scheduling.md §4).

The gate (``arena.gate.KVGate`` without a lane, as in hero-v2): at most ``slots`` requests in flight
(``max_num_seqs``) and the sum of reservations at most ``capacity`` blocks (0.9 x the served pool), first come
first served with head-of-line blocking. A request reserves ``BlockBudget.cost(prompt estimate + completion cap)``:
the attention blocks of its full length plus the fixed per-request state blocks (Mamba groups, the indexer ring),
exactly as ``GateSlot.tokens`` / ``BlockBudget.cost`` compute it; the recorded ``requested_tokens`` of every
admission attempt is the gate's input, so the replay uses the same numbers the run used.

Two replays:

- ``validate``: open loop on the recorded phase-2 attempts (arrival = admission minus the gate's own recorded
  wait from ``admission.jsonl``, hold = admitted -> released as recorded; requests in flight at the switch hold
  their blocks, requests queued at the switch arrive at it) at the recorded capacity and slots. Checks the cost
  function against the recorded reservations and the replayed admission times and waits against the recorded ones.
  Its ``after_warmup`` block is the reference for the 23-stream check of the resampling below.
- ``streams``: closed loop at N focus streams (``max_num_seqs`` = N + 2, as 23 / 25 in the run). Every stream plays
  a sequence of phase-2 *tenures* (one game's requests from its admission to its park, with the recorded gaps
  between requests) drawn uniformly with replacement from this run's phase-2 tenures, the first one entered at a
  random request. Uniform draws of concatenated tenures reproduce the recorded mix of context lengths per unit of
  slot time (long tenures hold their slot proportionally longer, as in the run). Holds scale with the per-stream
  decode slowdown at more streams (the simulator's iteration model t(n) = 8.92 + 1.046 n ms,
  ``sim_hero_v2.py``); gaps (tool work, agent overhead) do not. Three tenure pools: ``all`` (every phase-2 tenure),
  ``deep`` (tenures that started at level >= 1) and ``late`` (the tenures clipped to the second hour, when the
  contexts have reached the append-and-compact cycle a 9-hour run stays in; blocks per request 37 against 34).

    PYTHONPATH=src uv run python scripts/arena_analysis/gate_replay.py <trial dir> [--streams 23 26 28 30 32]
        [--seeds 20] [--out result.json]
"""

import argparse
import heapq
import json
import math
import random
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path

ITERATION_MS = (8.92, 1.046)  # sim_hero_v2.py: phase-2 iteration time t(n) = a + b n milliseconds


REFERENCE_STREAMS = 23  # the recorded run's focus streams (hero-v2: 23; --reference for another run)


def slowdown(streams, reference=None):
    """Per-stream hold multiplier at ``streams`` running against ``reference`` (decode iteration model)."""
    reference = REFERENCE_STREAMS if reference is None else reference
    a, b = ITERATION_MS
    return (a + b * streams) / (a + b * reference)


class Request:
    """One gate request. ``moves_on``: seconds after its admission at which its stream sends the next request
    while this one is still in flight (a game parked mid-request whose slot is taken by the next game); ``None``
    = at its release."""
    __slots__ = ("blocks", "hold", "gap", "at", "tag", "moves_on", "used", "arrival", "admitted", "released")

    def __init__(self, blocks, hold, gap=0.0, at=None, tag=None, moves_on=None, used=None):
        self.blocks, self.hold, self.gap, self.at, self.tag = int(blocks), float(hold), float(gap), at, tag
        self.used = int(blocks if used is None else used)  # blocks at its final length (prompt + reply)
        self.moves_on = None if moves_on is None or moves_on >= hold else max(0.0, float(moves_on))
        self.arrival = self.admitted = self.released = None

    @property
    def wait(self):
        return self.admitted - self.arrival


class GateSim:
    """Event-driven FIFO gate (``KVGate._pump_focus`` outside the opening): a request is admitted when every
    request ahead of it is admitted, ``inflight < slots`` and ``reserved + blocks <= capacity``."""

    def __init__(self, capacity, slots):
        self.capacity, self.slots = int(capacity), int(slots)

    def run(self, streams, start=0.0, until=math.inf):
        """``streams``: iterables of ``Request``. A request arrives at ``at`` if given, else ``gap`` seconds after
        the stream's previous release (after ``start`` for the first). Returns every admitted request and the
        step series of (time, reserved, inflight, waiting, used): ``used`` = the blocks every in-flight request
        would hold at its final length (an upper bound on the cache it actually occupies)."""
        heap, seq = [], 0  # (time, order, kind, payload); kind 0 release, 1 arrival, 2 stream moves on
        iters = [iter(s) for s in streams]
        done, queue = [], deque()
        reserved = inflight = used = 0
        series = [(start, 0, 0, 0, 0)]

        def schedule_next(index, after):
            nonlocal seq
            request = next(iters[index], None)
            if request is None:
                return
            request.arrival = request.at if request.at is not None else after + request.gap
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
                if request.moves_on is None:
                    schedule_next(index, now)
            elif kind == 2:
                schedule_next(index, now)
            else:
                queue.append((index, request))
            while queue and inflight < self.slots and reserved + queue[0][1].blocks <= self.capacity:
                i, head = queue.popleft()
                head.admitted = now
                head.released = now + head.hold
                reserved += head.blocks
                used += head.used
                inflight += 1
                done.append(head)
                seq += 1
                heapq.heappush(heap, (head.released, seq, 0, (i, head)))
                if head.moves_on is not None:
                    seq += 1
                    heapq.heappush(heap, (now + head.moves_on, seq, 2, (i, head)))
            if series[-1][0] == now:
                series[-1] = (now, reserved, inflight, len(queue), used)
            else:
                series.append((now, reserved, inflight, len(queue), used))
        return done, series


def time_share(series, start, end, predicate):
    """Share of [start, end] during which ``predicate(reserved, inflight, waiting)`` holds."""
    total = 0.0
    for (t0, r, i, w, *_), (t1, *_ ) in zip(series, series[1:] + [(end, 0, 0, 0, 0)]):
        a, b = max(t0, start), min(t1, end)
        if b > a and predicate(r, i, w):
            total += b - a
    return total / (end - start) if end > start else 0.0


def time_mean(series, start, end, field):
    total = 0.0
    for (t0, *v0), (t1, *_ ) in zip(series, series[1:] + [(end, 0, 0, 0, 0)]):
        a, b = max(t0, start), min(t1, end)
        if b > a:
            total += (b - a) * v0[field]
    return total / (end - start)


def time_quantile(series, start, end, field, q):
    pieces = []
    for (t0, *v0), (t1, *_ ) in zip(series, series[1:] + [(end, 0, 0, 0, 0)]):
        a, b = max(t0, start), min(t1, end)
        if b > a:
            pieces.append((v0[field], b - a))
    pieces.sort()
    target, acc = q * sum(w for _, w in pieces), 0.0
    for value, weight in pieces:
        acc += weight
        if acc >= target:
            return value
    return pieces[-1][0] if pieces else 0


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    return values[min(len(values) - 1, int(q * len(values)))]


# ---------------------------------------------------------------------------------------------- the recorded run

def load_run(trial_dir, margin=0.9):
    trial_dir = Path(trial_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from arena.cache import BlockBudget

    report = json.loads((trial_dir / "arena.json").read_text())
    budget = BlockBudget(json.loads((trial_dir / "cache-layout.json").read_text()), margin)
    # The gate's own admission records (``KVGate._admit``): the FIFO wait of every admission.
    report["_gate_admits"] = defaultdict(list)
    log = trial_dir / "admission.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("event") == "admit" and "waited" in entry:
                report["_gate_admits"][entry["game"]].append((entry["wall_seconds"], entry["waited"]))
    return report, budget


def recorded_attempts(report, budget):
    """Every admitted attempt: game, purpose, requested tokens, replayed and recorded blocks, arrival at the gate
    (admission minus the gate's own recorded wait; without the log, the start of the attempt's eligible wait),
    admission, release (recorded), and the prompt / completion tokens of its reply (None if cut at the wall)."""
    out = []
    gate_admits = report.get("_gate_admits") or {}
    # hero-v2 follow-ups (29 September): ``_prompt_scale`` = (factor, cap) stretches every request's prompt part
    # (requested tokens minus the completion cap) by ``factor``: a 64K run's contexts seen at a longer window, clipped
    # at the layout's max_model_len.
    scale = report.get("_prompt_scale")

    def requested(tokens):
        if scale is None:
            return tokens
        # (factor, cap) or (factor, cap, floor): with a floor (hero-v2-blocks, 29 September) only the prompt above it
        # scales (the post-compaction summary floor, ~8.8K in runs A and B, does not grow with the window).
        factor, cap, *rest = scale
        floor = rest[0] if rest else 0
        prompt = max(0, tokens - cap)
        scaled = prompt if prompt <= floor else floor + (prompt - floor) * factor
        target_cap = rest[1] if len(rest) > 1 else cap
        return min(int(target_cap + scaled), budget.layout["max_model_len"])

    for game in report["games"]:
        admits = gate_admits.get(game["game_id"], [])
        charges = iter(game["charges"])
        for a in game["admission_timing"]["attempts"]:
            if "admitted_wall_seconds" not in a:
                continue
            charge = next(charges) if a["status"] == "closed" else None
            admitted = a["admitted_wall_seconds"]
            waited = [w for t, w in admits if abs(t - admitted) < 0.05]
            arrival = (admitted - waited[0] if len(waited) == 1
                       else max(a["started_wall_seconds"], admitted - a["eligible_wait_seconds"]))
            out.append({
                "game": game["game_id"], "index": game["index"],
                "purpose": charge["purpose"] if charge else "cut",
                "requested_tokens": requested(a["requested_tokens"]), "blocks": budget.cost(requested(a["requested_tokens"])),
                "recorded_blocks": a["tokens"], "arrival": arrival, "admitted": admitted,
                "released": a.get("ended_wall_seconds", admitted), "phase": a.get("admitted_phase"),
                "prompt_tokens": charge["prompt_tokens"] if charge else None,
                "completion_tokens": charge["completion_tokens"] if charge else None,
                "used_blocks": (budget.cost(min(requested(charge["prompt_tokens"] + charge["completion_tokens"]),
                                                budget.layout["max_model_len"])) if charge
                                else budget.cost(requested(a["requested_tokens"]))),
            })
    return out


def validate(report, budget, slots=None):
    """Open-loop replay of the recorded phase 2 (arrivals after the switch) at the recorded gate."""
    switch = report["scheduler"]["switch_wall_seconds"]
    slots = slots or report["gate"]["slots"]
    rows = recorded_attempts(report, budget)
    cost_mismatch = sum(r["blocks"] != r["recorded_blocks"] for r in rows)
    # Requests admitted before the switch and still in flight hold their reservations at the switch; requests
    # queued at the switch (the opening's queue) arrive at the switch, in their recorded order.
    carried = [r for r in rows if r["admitted"] < switch < r["released"]]
    phase2 = sorted((r for r in rows if r["admitted"] >= switch), key=lambda r: r["arrival"])
    streams = [[Request(r["blocks"], r["released"] - switch, at=switch, tag=("carried", r))] for r in carried]
    streams += [[Request(r["blocks"], r["released"] - r["admitted"], at=max(switch, r["arrival"]), tag=("p2", r))]
                for r in phase2]
    done, series = GateSim(budget.capacity, slots).run(streams, start=switch)
    replayed = [q for q in done if q.tag[0] == "p2"]
    admit_err = [abs(q.admitted - q.tag[1]["admitted"]) for q in replayed]
    recorded_wait = [q.tag[1]["admitted"] - max(switch, q.tag[1]["arrival"]) for q in replayed]
    end = report["scheduler"]["events"][-1]["wall_seconds"]
    end = min(end, 7200.0)
    return {
        "attempts": len(rows), "cost_mismatches": cost_mismatch, "phase2_requests": len(replayed),
        "carried_at_switch": len(carried),
        "admission_time_error_seconds": {"mean": round(statistics.mean(admit_err), 3),
                                         "p90": round(quantile(admit_err, 0.9), 3),
                                         "max": round(max(admit_err), 3),
                                         "within_1s": round(sum(e <= 1.0 for e in admit_err) / len(admit_err), 4)},
        "wait_recorded": summarize_waits(recorded_wait),
        "wait_replayed": summarize_waits([q.wait for q in replayed]),
        "time_share_waiting": round(time_share(series, switch, end, lambda r, i, w: w > 0), 4),
        "time_share_waiting_on_blocks": round(time_share(series, switch, end, lambda r, i, w: w > 0 and i < slots), 4),
        "mean_running": round(time_mean(series, switch, end, 1), 2),
        "reserved_blocks_replayed": {"mean": round(time_mean(series, switch, end, 0), 1),
                                     "p50": time_quantile(series, switch, end, 0, 0.5),
                                     "p90": time_quantile(series, switch, end, 0, 0.9),
                                     "max": max(s[1] for s in series)},
        "peak_reserved_recorded": report["gate"]["peak_reserved"],
        # the window the stream replays measure (after a 300 s warm-up), for the 23-stream check of the resampling
        "after_warmup": window_stats(series, switch + 300.0, end, slots,
                                     report["scheduler"].get("active_limit") or slots - 2,
                                     [q.wait for q in replayed if q.arrival >= switch + 300.0]),
    }


def window_stats(series, lo, end, slots, streams, waits):
    return {"time_share_waiting": round(time_share(series, lo, end, lambda r, i, w: w > 0), 4),
            "time_share_waiting_on_blocks": round(time_share(series, lo, end, lambda r, i, w: w > 0 and i < slots), 4),
            "time_share_inflight_above_streams": round(time_share(series, lo, end, lambda r, i, w: i > streams), 4),
            "mean_running": round(time_mean(series, lo, end, 1), 2),
            "reserved_blocks": {"mean": round(time_mean(series, lo, end, 0), 1),
                                "p90": time_quantile(series, lo, end, 0, 0.9)},
            "waits": summarize_waits(waits)}


def summarize_waits(waits):
    waits = list(waits)
    positive = [w for w in waits if w > 0.05]
    return {"requests": len(waits), "waiting_share": round(len(positive) / len(waits), 4) if waits else 0.0,
            "mean_seconds": round(statistics.mean(waits), 3) if waits else 0.0,
            "p90_seconds": round(quantile(waits, 0.9), 3), "p99_seconds": round(quantile(waits, 0.99), 3),
            "max_seconds": round(max(waits), 3) if waits else 0.0,
            "mean_when_waiting_seconds": round(statistics.mean(positive), 3) if positive else 0.0}


def active_intervals(game, end):
    """(start, stop, levels at start) of the game's active spans from its admission records."""
    spans, since, levels = [], None, 0
    for e in game["admissions"]:
        if e["event"] == "admit":
            since, levels = e["wall_seconds"], e["levels"]
        elif e["event"] in ("park", "finish") and since is not None:
            spans.append((since, e["wall_seconds"], levels))
            since = None
    if since is not None:
        spans.append((since, end, levels))
    return spans


def phase2_tenures(report, budget, since=None):
    """This run's phase-2 tenures: per game and active span (clipped at the switch), the requests that arrived in
    it as (gap since the previous release or the span start, blocks, hold, purpose, moves_on), and the trailing gap
    from the last release to the park. A request still in flight at the park (the game sent it before the
    scheduler's next tick parked it) gets ``moves_on`` = park - its admission: the next game starts beside it.
    ``since``: clip the spans at this wall time instead of the switch (``late``: the second hour, whose contexts
    are in the append-and-compact steady state a 9-hour run would stay in)."""
    switch = report["scheduler"]["switch_wall_seconds"] if since is None else since
    end = min(7200.0, float(report.get("wall_seconds", 7200.0)) + 1.0)
    rows = defaultdict(list)
    for r in recorded_attempts(report, budget):
        rows[r["game"]].append(r)
    names = report.get("_names")
    tenures = []
    for game in report["games"]:
        for start, stop, levels in active_intervals(game, end):
            start, stop = max(start, switch), min(stop, end)
            if stop <= start:
                continue
            levels = sum(1 for level in game["time_to_level"] if level["wall_seconds"] <= start)
            reqs = [r for r in rows[game["game_id"]] if start <= r["arrival"] < stop]
            items, prev = [], start
            for r in reqs:
                hold = max(0.0, r["released"] - r["admitted"])
                moves_on = stop - r["admitted"] if (stop < end and r["released"] > stop) else None
                items.append((max(0.0, r["arrival"] - prev), r["blocks"], hold, r["purpose"], moves_on,
                              r["used_blocks"]))
                prev = min(r["released"], stop) if moves_on is not None else r["released"]
            tenures.append({"game": game["game_id"], "kind": names[game["index"] % len(names)] if names else None,
                            "start": start, "stop": stop, "levels": levels, "requests": items,
                            "tail": max(0.0, stop - prev) if stop < end else 0.0})
    return [t for t in tenures if t["requests"]]


def tenure_stream(tenures, rng, factor):
    """An endless stream of requests from uniformly drawn tenures; the first is entered at a random request, and
    a tenure's tail (its last release to its park) is added to the next tenure's first gap."""
    first, carry = True, 0.0
    while True:
        t = rng.choice(tenures)
        items = t["requests"]
        begin = rng.randrange(len(items)) if first else 0
        first = False
        for k, (gap, blocks, hold, purpose, moves_on, used) in enumerate(items[begin:]):
            yield Request(blocks, hold * factor, gap=gap + (carry if k == 0 else 0.0),
                          tag=(t["game"], t["kind"], t["levels"], purpose), moves_on=moves_on, used=used)
        carry = t["tail"]


def replay_streams(tenures, capacity, streams, *, slots=None, seeds=20, start=0.0, duration=6048.0,
                   factor=None, warmup=300.0):
    """Closed-loop replay at ``streams`` focus streams; statistics over [start + warmup, start + duration]."""
    slots = slots or streams + 2
    factor = slowdown(streams) if factor is None else factor
    end = start + duration
    lo = start + warmup
    waits, by_kind, by_blocks, shares, reserved, running, hol = [], defaultdict(list), defaultdict(list), [], [], [], []
    above, used = [], []
    for seed in range(seeds):
        rng = random.Random(seed)
        gens = [tenure_stream(tenures, rng, factor) for _ in range(streams)]
        done, series = GateSim(capacity, slots).run(gens, start=start, until=end)
        measured = [q for q in done if q.arrival >= lo]
        for q in measured:
            waits.append(q.wait)
            by_kind[q.tag[1]].append(q.wait)
            by_blocks[min(51, 10 * (q.blocks // 10))].append(q.wait)
        shares.append(time_share(series, lo, end, lambda r, i, w: w > 0))
        above.append(time_share(series, lo, end, lambda r, i, w: i > streams))
        reserved.append((time_mean(series, lo, end, 0), time_quantile(series, lo, end, 0, 0.9),
                         max(s[1] for s in series if s[0] >= lo)))
        used.append((time_mean(series, lo, end, 3), time_quantile(series, lo, end, 3, 0.99),
                     max(s[4] for s in series if s[0] >= lo)))
        running.append(time_mean(series, lo, end, 1))
        # capacity-bound waits: some request waits and the slot limit is not what holds it
        hol.append(time_share(series, lo, end, lambda r, i, w: w > 0 and i < slots))
    lost = sum(waits) / (streams * (end - lo) * seeds)
    return {
        "streams": streams, "max_num_seqs": slots, "hold_factor": round(factor, 4), "seeds": seeds,
        "time_share_waiting": round(statistics.mean(shares), 4),
        "time_share_waiting_on_blocks": round(statistics.mean(hol), 4),
        "waits": summarize_waits(waits),
        "stream_time_lost_to_waits": round(lost, 4),
        "time_share_inflight_above_streams": round(statistics.mean(above), 4),
        "mean_running": round(statistics.mean(running), 2),
        "reserved_blocks": {"mean": round(statistics.mean(r[0] for r in reserved), 1),
                            "p90": round(statistics.mean(r[1] for r in reserved), 1),
                            "max": max(r[2] for r in reserved)},
        # the cache the in-flight requests would occupy at their final lengths (the engine's pool is
        # capacity / 0.9 blocks; the gate's reservation adds the unused part of each 16,384-token cap)
        "final_length_blocks": {"mean": round(statistics.mean(u[0] for u in used), 1),
                                "p99": round(statistics.mean(u[1] for u in used), 1),
                                "max": max(u[2] for u in used)},
        "wait_by_kind": {k: {"requests": len(v), "mean_seconds": round(statistics.mean(v), 2),
                             "waiting_share": round(sum(w > 0.05 for w in v) / len(v), 3),
                             "share_of_all_wait": round(sum(v) / max(sum(waits), 1e-9), 3)}
                         for k, v in sorted(by_kind.items(), key=lambda kv: -sum(kv[1]))},
        "wait_by_blocks": {f"{k}-{k + 9}" if k < 51 else "51": {
            "requests": len(v), "mean_seconds": round(statistics.mean(v), 2),
            "waiting_share": round(sum(w > 0.05 for w in v) / len(v), 3)}
            for k, v in sorted(by_blocks.items())},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_dir")
    parser.add_argument("--config", help="comparison.json (game names for the per-kind table)")
    parser.add_argument("--streams", type=int, nargs="+", default=[23, 26, 28, 30, 32])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out")
    # hero-v2 pre-check (docs/validation-20260927/hero-v2-precheck.md): the gate capacity of a measured pool
    # (0.9 x its num_blocks) and a measured decode step model in place of the recorded run's and the simulator's.
    parser.add_argument("--capacity", type=int, default=None, help="gate capacity in blocks (default: the run's)")
    parser.add_argument("--iteration", type=float, nargs=2, default=None, metavar=("A_MS", "B_MS"),
                        help="decode step model t(n) = A + B n ms for the hold slowdown (default: the simulator's)")
    # hero-v2 follow-ups (29 September): another served layout (e.g. the int4 pool's cache-layout.json) in place of
    # the run's, the recorded run's own focus streams as the slowdown reference, and prompts stretched to a longer
    # window (``--prompt-scale F --cap 16384``).
    parser.add_argument("--layout", default=None, help="cache-layout.json of another served pool (its costs and 0.9 gate)")
    parser.add_argument("--reference", type=int, default=None, help="the recorded run's focus streams (default 23)")
    parser.add_argument("--prompt-scale", type=float, default=None)
    parser.add_argument("--cap", type=int, default=16384)
    parser.add_argument("--pools", nargs="+", default=["all", "deep", "late"])
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    if args.iteration:
        global ITERATION_MS
        ITERATION_MS = tuple(args.iteration)
    if args.reference:
        global REFERENCE_STREAMS
        REFERENCE_STREAMS = args.reference
    report, budget = load_run(args.trial_dir)
    if args.layout:
        from arena.cache import BlockBudget

        budget = BlockBudget(json.loads(Path(args.layout).read_text()), 0.9)
    if args.prompt_scale:
        report["_prompt_scale"] = (args.prompt_scale, args.cap)
    if args.capacity:
        budget.capacity = args.capacity
    if args.config:
        report["_names"] = [g[:4] for g in json.loads(Path(args.config).read_text())["games"]]
    switch = report["scheduler"]["switch_wall_seconds"]
    result = {"capacity_blocks": budget.capacity, "iteration_ms": list(ITERATION_MS),
              "reference_streams": REFERENCE_STREAMS, "layout": args.layout, "prompt_scale": args.prompt_scale,
              "validation": validate(report, budget) if not (args.layout or args.prompt_scale) else None}
    tenures = phase2_tenures(report, budget)
    deep = [t for t in tenures if t["levels"] >= 1]
    late = phase2_tenures(report, budget, since=3600.0)
    result["tenures"] = {"all": len(tenures), "level1_plus": len(deep), "late": len(late)}
    for name, pool in (("all", tenures), ("deep", deep), ("late", late)):
        if name not in args.pools or not pool:
            continue
        result[name] = [replay_streams(pool, budget.capacity, n, seeds=args.seeds, start=switch,
                                       duration=7200.0 - switch) for n in args.streams]
    # hold-time sensitivity: no slowdown at all, and the simulator's "low" rate (slope x1.5 past 22.5 streams)
    a, b = ITERATION_MS
    low = {n: (a + b * 22.5 + 1.5 * b * (n - 22.5)) / (a + b * 23) for n in args.streams}
    result["hold_sensitivity"] = {} if args.no_sensitivity else {
        label: [replay_streams(tenures, budget.capacity, n, seeds=args.seeds, start=switch, duration=7200.0 - switch,
                               factor=f(n)) for n in args.streams if n >= 28]
        for label, f in (("no_slowdown", lambda n: 1.0), ("slope_x1.5", lambda n: low[n]))}
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
