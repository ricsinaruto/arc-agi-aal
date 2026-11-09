"""CPU estimate of the parallel-samples latency gain from a recorded run (30 September; parallel-samples-plans.md).

    uv run python scripts/arena_analysis/parallel_samples_estimate.py [--run runB] [--k 4] [--min-levels 2] [--json P]

Every recorded gameplay reply of a game with at least ``min_levels`` levels is replayed as k samples. The recorded reply
is the primary. Siblings are drawn from the same game's replies on the same level, which is the closest in-run proxy
for "the same prompt again", in log-length space with a correlation ``rho`` to the primary:
z_sibling = rho * z_primary + sqrt(1 - rho^2) * noise, within the (game, level) mean and spread. A sibling carries a
tool call with the (game, level) share of tool-call replies. The winner is the shortest sample with a tool call, or
the shortest sample when none has one.

Siblings of one prompt are correlated: the situation fixes much of the reasoning length. The recorded runs sample every
prompt once, so rho cannot be measured on CPU. rho = 0 (independent draws from the level's spread) is an upper bound on
the gain; 0.5 and 0.8 show how quickly it shrinks. The run's observe mode (every 8th group lets the losers finish)
measures the real spread.

Costs:
- every stream slows down while siblings decode: the engine step grows by ``slope`` ms per extra running sequence
  (0.32 ms measured at 48 short streams on int4, prefix-unit-int4.md; 0.417 ms at 25K contexts on int8, precheck);
- siblings are limited by the free sequences (max_num_seqs 48 - 24 streams - 1) and by the gate room the run left
  (its reserved blocks against the capacity, the one-full-window headroom and the siblings' own blocks);
- the ping adds one short prefill before every group;
- the aborted siblings' decoded tokens are extra engine work.
"""

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_load import Run  # noqa: E402

SLOPES_MS = (0.32, 0.417)
BLOCK = 3136
CAP = 16384
SEQS, STREAMS = 48, 24
HEADROOM_BLOCKS = 51
PING_SECONDS = 0.5  # one ~1.5K-token tail prefill beside the decode batch plus a round trip (conservative)


def sibling_blocks(prompt):
    total = -(-(-(-(prompt + CAP) // 1024) * 1024) // BLOCK) + 9
    return total - prompt // BLOCK


def gameplay(run):
    rows = []
    for game in run.games.values():
        for reply in game.replies:
            if reply["purpose"] != "gameplay" or not reply["seconds"] or reply["completion"] <= 0:
                continue
            rows.append({"game": game.id, "family": game.family, "level": reply["level"], "wall": reply["wall"],
                         "tokens": reply["completion"], "seconds": float(reply["seconds"]),
                         "prompt": reply["prompt"], "tool": reply["finish"] == "tool_calls"})
    return rows


def decode_rates(rows):
    """Per-game decode rate (tokens / s) from the replies long enough for prefill to be negligible."""
    per = defaultdict(list)
    for r in rows:
        if r["tokens"] >= 800:
            per[r["game"]].append(r["tokens"] / r["seconds"])
    overall = statistics.median(v for vs in per.values() for v in vs)
    return {g: statistics.median(vs) for g, vs in per.items()}, overall


def inflight_eligible(run, min_levels):
    """Mean number of eligible games with a gameplay request in flight over phase 2 (after the first 1,200 s)."""
    step, start, end = 5.0, 1200.0, run.wall
    counts = []
    for game in run.games.values():
        game._clear_walls = [c["wall"] for c in game.clears]
    t = start
    while t < end:
        n = 0
        for game in run.games.values():
            if sum(1 for w in game._clear_walls if w <= t) < min_levels:
                continue
            for iv in game.inference_intervals:
                if iv["purpose"] == "gameplay" and iv["started_wall_seconds"] <= t < iv["ended_wall_seconds"]:
                    n += 1
                    break
        counts.append(n)
        t += step
    return statistics.mean(counts), counts


def gate_room(run):
    """Median and p10 spare gate blocks beside the one-full-window headroom in phase 2 (admission log)."""
    path = run.dir / "admission.jsonl"
    spare = []
    capacity = None
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") == "release" and "reserved_after" in rec and (rec.get("wall_seconds") or 0) > 1200:
            spare.append(rec["reserved_after"] + rec["tokens"])
    capacity = int(0.9 * 1268)
    spare = sorted(capacity - HEADROOM_BLOCKS - s for s in spare)
    return {"p10": spare[len(spare) // 10], "p50": spare[len(spare) // 2], "capacity": capacity}


def simulate(rows, rates, overall_rate, *, siblings, min_levels, rho, slowdown, seed=0, draws=20):
    """``siblings``: the mean number of siblings a request gets (the running limit shared by the eligible requests
    in flight): floor(siblings) each, one more with probability frac(siblings)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["game"], r["level"])].append(r)
    out = {"requests": 0, "primary_seconds": 0.0, "winner_seconds": 0.0, "winner_tokens": 0.0,
           "primary_tokens": 0.0, "aborted_tokens": 0.0, "sibling_wins": 0.0}
    for key, members in groups.items():
        if key[1] < min_levels:
            continue
        logs = [math.log(m["tokens"]) for m in members]
        mu = statistics.mean(logs)
        sigma = statistics.pstdev(logs) if len(logs) > 1 else 0.0
        p_tool = sum(m["tool"] for m in members) / len(members)
        rate = rates.get(key[0], overall_rate)
        for m in members:
            z = (math.log(m["tokens"]) - mu) / sigma if sigma > 0 else 0.0
            prefill = max(0.0, m["seconds"] - m["tokens"] / rate)
            for _ in range(draws):
                samples = [(m["tokens"], m["tool"])]
                extra = int(siblings) + (rng.random() < siblings - int(siblings))
                for _ in range(extra):
                    zs = rho * z + math.sqrt(1 - rho * rho) * rng.gauss(0, 1)
                    samples.append((math.exp(mu + sigma * zs), rng.random() < p_tool))
                tooled = [s for s in samples if s[1]]
                win = min(tooled or samples, key=lambda s: s[0])
                out["requests"] += 1 / draws
                out["primary_tokens"] += m["tokens"] / draws
                out["primary_seconds"] += m["seconds"] / draws
                out["winner_tokens"] += win[0] / draws
                out["winner_seconds"] += (prefill + PING_SECONDS + win[0] / (rate * slowdown)) / draws
                out["aborted_tokens"] += sum(min(s[0], win[0]) for s in samples if s is not win) / draws
                out["sibling_wins"] += (win is not samples[0]) / draws
    return out


def estimate(label="runB", k=4, min_levels=2):
    run = Run(label)
    rows = gameplay(run)
    rates, overall = decode_rates(rows)
    eligible, _ = inflight_eligible(run, min_levels)
    room = gate_room(run)
    median_prompt = statistics.median(r["prompt"] for r in rows if r["level"] >= min_levels)
    per_sibling = sibling_blocks(int(median_prompt))
    siblings = {
        "demand": eligible * (k - 1),
        "sequence_limit": SEQS - STREAMS - 1,
        "gate_limit_p50": room["p50"] // per_sibling,
        "gate_limit_p10": room["p10"] // per_sibling,
    }
    running = min(siblings["demand"], siblings["sequence_limit"], siblings["gate_limit_p50"])
    per_request = min(k - 1, running / eligible) if eligible else k - 1
    base_step = 1000.0 / overall  # ms per decode step at the run's batch
    result = {"run": run.run_id, "k": k, "min_levels": min_levels, "decode_rate_tokens_per_second": round(overall, 1),
              "base_step_ms": round(base_step, 1), "eligible_inflight_mean": round(eligible, 2),
              "median_eligible_prompt": int(median_prompt), "sibling_blocks_at_median_prompt": per_sibling,
              "gate_room": room, "sibling_limits": {k2: round(v, 1) for k2, v in siblings.items()},
              "siblings_running_mean": round(running, 1), "siblings_per_request": round(per_request, 2),
              "cells": []}
    eligible_rows = [r for r in rows if r["level"] >= min_levels]
    result["eligible_share_of_gameplay_tokens"] = round(sum(r["tokens"] for r in eligible_rows)
                                                        / sum(r["tokens"] for r in rows), 3)
    for slope in SLOPES_MS:
        slowdown = base_step / (base_step + running * slope)
        for rho in (0.0, 0.5, 0.8):
            sim = simulate(rows, rates, overall, siblings=per_request, min_levels=min_levels, rho=rho,
                           slowdown=slowdown)
            n = sim["requests"]
            cell = {"slope_ms": slope, "rho": rho, "stream_slowdown": round(slowdown, 3),
                    "requests": round(n),
                    "reply_seconds_before": round(sim["primary_seconds"] / n, 2),
                    "reply_seconds_after": round(sim["winner_seconds"] / n, 2),
                    "reply_latency_change": round(sim["winner_seconds"] / sim["primary_seconds"] - 1, 3),
                    "reply_tokens_before": round(sim["primary_tokens"] / n),
                    "reply_tokens_after": round(sim["winner_tokens"] / n),
                    "sibling_win_share": round(sim["sibling_wins"] / n, 3),
                    "aborted_tokens_per_winner_token": round(sim["aborted_tokens"] / sim["winner_tokens"], 2),
                    "other_games_latency_change": round(1 / slowdown - 1, 3)}
            result["cells"].append(cell)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runB")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-levels", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()
    result = estimate(args.run, args.k, args.min_levels)
    text = json.dumps(result, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
