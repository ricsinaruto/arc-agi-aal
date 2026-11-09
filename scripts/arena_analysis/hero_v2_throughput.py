"""Throughput bottlenecks of one arena run from its recordings (29 September, CPU only; hero-v2 2 h).

    uv run --with tokenizers --with scipy python scripts/arena_analysis/hero_v2_throughput.py \
        --trial <run>/<run>/arena/trials/duck-transfer --out docs/validation-20260927/hero-v2-analysis/throughput.json

Four parts, each from recorded evidence only:

- ``engine``: the 5 s telemetry intervals (cumulative counters). Per interval: engine steps (``iteration_tokens``
  count), generated tokens, computed (uncached) prompt tokens, running, KV use. A least-squares fit
  ``interval_ms = a * steps + b * generated + c * computed`` gives the in-run step cost and the prefill share of
  engine time; decode-only tok/s at the measured batch is compared with the simulator's t(n) = 8.92 + 1.046 n ms.
- ``streams``: every game's request timeline, joined from its ``log.jsonl`` (wall epoch), ``requests.json``
  (monotonic HTTP start and duration; the clock offset is fitted) and ``admission.jsonl`` (gate waits). Each moment
  of a phase-2 focus game is one of: in flight at the server, the tool round (python sandbox and env actions), host
  time after the tool, the exact-count /tokenize, the gate wait, or parked. Integrated over time this gives the mean
  number of focus streams in each state, and per request the gap from the reply to the next HTTP start.
- ``kv``: the engine's KV use (blocks held by running requests) against the gate's reservation and the in-flight
  context lengths.
- ``prefill``: ``prefill_attribution.attribute`` on the run (exact interval matching of the prefix-cache counters),
  plus what a finer cache granularity would recover: the upstream partial-tail entry at the previous *prompt*'s
  last hash boundary (``prefix_match_unit``), and a tail saved at the previous *request's end*.
"""

import argparse
import bisect
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SIM_INTERCEPT, SIM_SLOPE = 8.92, 1.046  # sim_hero_v2.py, ms per iteration at n streams
DECODE_ONLY_SLOPE = 0.43  # hero smoke, decode-only ms per stream (sim-hero-v2.md)


def pct(values, qs=(10, 25, 50, 75, 90, 99)):
    arr = np.asarray([v for v in values if v is not None], float)
    if not len(arr):
        return {}
    out = {f"p{q}": round(float(np.percentile(arr, q)), 3) for q in qs}
    out["mean"] = round(float(arr.mean()), 3)
    out["n"] = int(len(arr))
    return out


# ------------------------------------------------------------------------------------------------ engine

def load_telemetry(trial):
    rows = []
    for line in (trial / "telemetry.jsonl").open():
        r = json.loads(line)
        m = r["metrics"]
        rows.append({
            "t": r["elapsed_seconds"],
            "steps": m["vllm:iteration_tokens_total_count"],
            "gen": m["vllm:generation_tokens_total"],
            "computed": m["vllm:prompt_tokens_by_source_total:local_compute"],
            "cached": m["vllm:prompt_tokens_by_source_total:local_cache_hit"],
            "running": m["vllm:num_requests_running"],
            "waiting": m["vllm:num_requests_waiting"],
            "kv": m["vllm:kv_cache_usage_perc"],
            "itl_sum": m["vllm:inter_token_latency_seconds_sum"],
            "itl_n": m["vllm:inter_token_latency_seconds_count"],
            "prefill_s": m["vllm:request_prefill_time_seconds_sum"],
            "queue_s": m["vllm:request_queue_time_seconds_sum"],
            "done": m["vllm:request_success_total"],
        })
    # The first sample carries the server warm-up (100 requests) and comes from before the gameplay monitor.
    rows = [r for r in rows[1:]]
    rows.sort(key=lambda r: r["t"])
    return rows


def intervals(rows):
    out = []
    for a, b in zip(rows, rows[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0.5:
            continue
        d = {k: b[k] - a[k] for k in ("steps", "gen", "computed", "cached", "itl_sum", "itl_n", "prefill_s",
                                       "queue_s", "done")}
        d.update(t0=a["t"], t1=b["t"], dt=dt, running=(a["running"] + b["running"]) / 2, run_end=b["running"],
                 waiting=b["waiting"], kv=b["kv"])
        out.append(d)
    return out


def engine_fit(ivs, switch, wall=7200.0, fixed_batch_slope=None):
    """Mean step time per interval, ms = a + b * (decode tokens per step) + c * (computed prompt tokens per step),
    weighted by steps. The engine is never idle in these windows (running >= 20), so steps x step time = wall.
    ``fixed_batch_slope`` pins b (the opening runs at a constant batch of ~24.9, so b is not identified there)."""
    phase2 = [i for i in ivs if i["t0"] >= switch + 30 and i["t1"] <= wall and i["steps"] > 0]
    opening = [i for i in ivs if 30 <= i["t0"] and i["t1"] <= switch and i["steps"] > 0]
    res = {}
    for name, sel in (("phase2", phase2), ("opening", opening), ("all", opening + phase2)):
        S = np.array([i["steps"] for i in sel], float)
        G = np.array([i["gen"] for i in sel], float)
        C = np.array([i["computed"] for i in sel], float)
        T = np.array([i["dt"] * 1000 for i in sel], float)
        y = T / S
        w = np.sqrt(S)
        pin = fixed_batch_slope if name != "phase2" else None
        if pin is None:
            A = np.stack([np.ones_like(S), G / S, C / S], 1)
            coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
            a, b, c = (float(x) for x in coef)
        else:
            A = np.stack([np.ones_like(S), C / S], 1)
            coef, *_ = np.linalg.lstsq(A * w[:, None], (y - pin * G / S) * w, rcond=None)
            a, c = (float(x) for x in coef)
            b = pin
        pred = a + b * G / S + c * C / S
        r2 = 1 - (S * (y - pred) ** 2).sum() / (S * (y - np.average(y, weights=S)) ** 2).sum()
        if name == "phase2":
            fixed_batch_slope = b
        tot_t = T.sum()
        n_mean = G.sum() / S.sum()
        res[name] = {
            "intervals": len(sel), "seconds": round(tot_t / 1000, 1),
            "fit_ms": {"per_step": round(a, 3), "per_generated_token": round(b, 4),
                       "per_computed_prompt_token": round(c, 5)},
            "r2": round(float(r2), 3),
            "generated": int(G.sum()), "computed": int(C.sum()), "steps": int(S.sum()),
            "gen_tok_s": round(G.sum() / tot_t * 1000, 1),
            "computed_tok_s": round(C.sum() / tot_t * 1000, 1),
            "computed_per_generated": round(C.sum() / G.sum(), 3),
            "mean_decode_batch": round(float(n_mean), 2),
            "mean_step_ms": round(tot_t / S.sum(), 2),
            "share": {"step_overhead": round(a * S.sum() / tot_t, 3), "decode": round(b * G.sum() / tot_t, 3),
                      "prefill": round(c * C.sum() / tot_t, 3)},
            "decode_only_step_ms_at_mean_batch": round(a + b * n_mean, 2),
            "decode_only_tok_s_at_mean_batch": round(1000 * n_mean / (a + b * n_mean), 1),
            "sim_fit_tok_s_at_mean_batch": round(1000 * n_mean / (SIM_INTERCEPT + SIM_SLOPE * n_mean), 1),
            "mean_itl_ms": round(sum(i["itl_sum"] for i in sel) / max(1, sum(i["itl_n"] for i in sel)) * 1000, 2),
        }
    return res, phase2


def fit_bootstrap(phase2, draws=400, block_s=120.0, seed=0):
    """Moving-block bootstrap (blocks of ``block_s`` seconds) of the phase-2 step fit: 90% intervals of the
    coefficients and of the projected tok/s at n streams (prefill per generated token held at the run's value)."""
    rng = np.random.default_rng(seed)
    t0 = phase2[0]["t0"]
    blocks = defaultdict(list)
    for i in phase2:
        blocks[int((i["t0"] - t0) // block_s)].append(i)
    keys = sorted(blocks)
    k_ratio = sum(i["computed"] for i in phase2) / sum(i["gen"] for i in phase2)
    ns = (23.5, 25, 30, 32, 38, 48)
    coefs, proj = [], []
    for _ in range(draws):
        sel = [i for k in rng.choice(keys, len(keys)) for i in blocks[k]]
        S = np.array([i["steps"] for i in sel], float)
        A = np.stack([np.ones_like(S), np.array([i["gen"] for i in sel]) / S,
                      np.array([i["computed"] for i in sel]) / S], 1)
        y = np.array([i["dt"] * 1000 for i in sel]) / S
        w = np.sqrt(S)
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        coefs.append(c)
        proj.append([1000 * n / (c[0] + (c[1] + c[2] * k_ratio) * n) for n in ns])
    coefs, proj = np.array(coefs), np.array(proj)
    q = lambda x: [round(float(np.percentile(x, 5)), 4), round(float(np.percentile(x, 95)), 4)]  # noqa: E731
    return {"draws": draws, "block_seconds": block_s, "computed_per_generated": round(k_ratio, 3),
            "per_step_ms_90": q(coefs[:, 0]), "per_generated_token_ms_90": q(coefs[:, 1]),
            "per_computed_token_ms_90": q(coefs[:, 2]),
            "projected_tok_s": {str(n): {"median": round(float(np.median(proj[:, j]))),
                                         "90": [round(float(v)) for v in np.percentile(proj[:, j], (5, 95))]}
                                for j, n in enumerate(ns)}}


def itl_cross_check(trial, batch, decode_step_ms):
    """Prefill time from the whole-run inter-token-latency histogram: a token's ITL is the duration of the step
    that produced it, so the steps above 0.1 s are the prefill-bearing ones. Steps = tokens / decode batch; the
    excess over a decode-only step is prefill time. Bucket midpoints make this coarse (about +-15%)."""
    import re
    buckets = []
    for line in (trial / "vllm-metrics.txt").read_text().splitlines():
        m = re.match(r'vllm:inter_token_latency_seconds_bucket\{.*le="([^"]+)".*\}\s+(\S+)', line)
        if m:
            buckets.append((float("inf") if m[1] == "+Inf" else float(m[1]), float(m[2])))
    buckets.sort()
    prev_le, prev_n = 0.0, 0.0
    rows, excess, long_steps = [], 0.0, 0.0
    for le, n in buckets:
        tokens = n - prev_n
        if tokens > 0 and prev_le >= 0.1:
            hi = le if math.isfinite(le) else prev_le * 2
            mid = (prev_le + hi) / 2
            steps = tokens / batch
            long_steps += steps
            excess += steps * (mid - decode_step_ms / 1000)
            rows.append({"itl_s": f"{prev_le:g}-{le:g}", "tokens": int(tokens), "steps": round(steps),
                         "seconds": round(steps * mid, 1)})
        prev_le, prev_n = le, n
    return {"buckets_over_100ms": rows, "long_steps": round(long_steps), "prefill_excess_seconds": round(excess, 1)}


def throughput_by_running(ivs, lo, hi, fit):
    """Phase-2 tok/s by the interval's mean running count, with the fit's decode-only and the sim's t(n)."""
    a, b, c = (fit["fit_ms"][k] for k in ("per_step", "per_generated_token", "per_computed_prompt_token"))
    bins = defaultdict(list)
    for i in ivs:
        if lo <= i["t0"] and i["t1"] <= hi:
            bins[int(round(i["running"]))].append(i)
    out = []
    for n in sorted(bins):
        sel = bins[n]
        T = sum(i["dt"] for i in sel)
        G = sum(i["gen"] for i in sel)
        C = sum(i["computed"] for i in sel)
        S = sum(i["steps"] for i in sel)
        nb = G / S if S else 0
        out.append({"running": n, "intervals": len(sel), "seconds": round(T), "gen_tok_s": round(G / T, 1),
                    "computed_tok_s": round(C / T), "decode_batch": round(nb, 2),
                    "step_ms": round(T * 1000 / S, 2) if S else None,
                    "decode_only_tok_s": round(1000 * nb / (a + b * nb), 1) if nb else None,
                    "sim_fit_tok_s": round(1000 * n / (SIM_INTERCEPT + SIM_SLOPE * n), 1)})
    return out


def prefill_tax_bins(ivs, lo, hi):
    """gen tok/s against the interval's uncached prompt rate (the 'prefill tax' view of the live tail)."""
    edges = (0, 200, 500, 1000, 2000, 4000, 1e9)
    bins = defaultdict(list)
    for i in ivs:
        if lo <= i["t0"] and i["t1"] <= hi:
            rate = i["computed"] / i["dt"]
            bins[bisect.bisect_right(edges, rate) - 1].append(i)
    out = []
    for k in sorted(bins):
        sel = bins[k]
        T = sum(i["dt"] for i in sel)
        out.append({"computed_tok_s": f"{edges[k]:g}-{edges[k + 1]:g}", "intervals": len(sel),
                    "share_of_time": None, "gen_tok_s": round(sum(i["gen"] for i in sel) / T, 1),
                    "running": round(sum(i["running"] * i["dt"] for i in sel) / T, 2)})
    total = sum(o["intervals"] for o in out)
    for o in out:
        o["share_of_time"] = round(o["intervals"] / total, 3)
    return out


# ------------------------------------------------------------------------------------------------ streams

def load_admission(trial):
    rows = [json.loads(line) for line in (trial / "admission.jsonl").open()]
    origin = np.median([r["epoch"] - r["wall_seconds"] for r in rows if (r.get("wall_seconds") or 0) > 10])
    return rows, float(origin)


def load_games(trial, origin):
    """Per game: its log events (wall seconds) and requests (chat start, count done, admitted, end)."""
    games = {}
    for game_dir in sorted((trial / "games").iterdir()):
        log = game_dir / "log.jsonl"
        if not log.is_file():
            continue
        events = []
        for line in log.open():
            d = json.loads(line)
            d["t"] = d["time"] - origin
            d.pop("frames", None)
            if d["kind"] == "duck_python":
                res = d.get("result") or {}
                d["stdout_chars"] = len(str(res.get("stdout") or ""))
                d.pop("result", None)
            if d["kind"] == "model":
                d.pop("message", None)
            events.append(d)
        games[game_dir.name] = events
    return games


def fit_http_offset(games, requests):
    """epoch - monotonic from (model record time) - (HTTP start + HTTP seconds), per matched request."""
    by_game = defaultdict(list)
    for r in requests:
        if r.get("status") == 200:
            by_game[r["instance_id"].split("-", 1)[1]].append(r)
    diffs = []
    for g, events in games.items():
        models = [e for e in events if e["kind"] == "model"]
        for e, r in zip(models, by_game.get(g, [])):
            diffs.append(e["t"] - (r["started_monotonic"] + r["seconds"]))
    diffs = np.array(diffs)
    off = float(np.median(diffs))
    return off, by_game, {"median": round(off, 3), "p1": round(float(np.percentile(diffs, 1) - off), 3),
                          "p99": round(float(np.percentile(diffs, 99) - off), 3)}


def build_requests(games, http_by_game, offset):
    """One record per served request, with its timeline and the tool round that followed it."""
    out = []
    for g, events in games.items():
        https = http_by_game.get(g, [])
        k = 0
        pending_count = pending_admit = None
        cur = None
        for e in events:
            kind = e["kind"]
            if kind == "exact_admission_count":
                pending_count = e["t"]
            elif kind == "model_request":
                pending_admit = e["t"]
            elif kind == "model":
                h = https[k] if k < len(https) else None
                k += 1
                usage = e.get("usage") or {}
                cur = {
                    "game": g, "purpose": e.get("purpose"), "chat_start": e["t"] - e.get("seconds", 0),
                    "count_done": pending_count, "admitted": pending_admit,
                    "http_start": h["started_monotonic"] + offset if h else None,
                    "http_seconds": h["seconds"] if h else None, "end": e["t"],
                    "P": int(usage.get("prompt_tokens") or 0), "G": int(usage.get("completion_tokens") or 0),
                    "tools": 0, "tool_end": None, "actions": 0, "env_seconds": 0.0, "first_action": None,
                    "last_action": None, "stdout_chars": 0,
                }
                out.append(cur)
                pending_count = pending_admit = None
            elif cur is not None and kind == "duck_python":
                cur["tools"] += 1
                cur["tool_end"] = e["t"]
                cur["stdout_chars"] += e.get("stdout_chars", 0)
            elif cur is not None and kind == "action_timing":
                cur["actions"] += 1
                cur["env_seconds"] += e.get("seconds", 0.0)
                if cur["first_action"] is None:
                    cur["first_action"] = e["t"] - e.get("seconds", 0.0)
                cur["last_action"] = e["t"]
    return out


def attach_gate(reqs, adm):
    """Gate wait records (request_admitted) by game in order; aside/gameplay requests both pass the gate."""
    by_game = defaultdict(list)
    for r in adm:
        if r.get("event") == "request_admitted":
            by_game[r["game"]].append(r)
    for g in by_game:
        by_game[g].sort(key=lambda r: r["wall_seconds"])
    per_game = defaultdict(list)
    for r in reqs:
        per_game[r["game"]].append(r)
    for g, rs in per_game.items():
        gates = by_game.get(g, [])
        starts = [x["wall_seconds"] for x in gates]
        for r in rs:
            if r["admitted"] is None:
                continue
            j = bisect.bisect_left(starts, r["admitted"] - 0.5)
            if j < len(gates) and abs(gates[j]["wall_seconds"] - r["admitted"]) < 0.5:
                x = gates[j]
                r["gate_wait"] = x["wait_seconds"]
                r["gate_eligible"] = x["eligible_wait_seconds"]
                r["gate_parked"] = x["parked_wait_seconds"]
                r["gate_unclassified"] = x["unclassified_wait_seconds"]
    return per_game


def park_intervals(adm, wall):
    """[(start, end)] parked intervals per game from park / admit-grant events."""
    events = defaultdict(list)
    for r in adm:
        # Scheduler events carry a reason; a gate "admit" (one request) does not. "grant" extends an active game.
        if r.get("event") in ("park", "admit") and "reason" in r:
            events[r["game"]].append((r["wall_seconds"], 0 if r["event"] == "park" else 1, r["event"]))
    out = {}
    for g, ev in events.items():
        ev.sort()  # a park and its immediate re-admission share a timestamp: park first
        spans, start = [], None
        for t, _, kind in ev:
            if kind == "park" and start is None:
                start = t
            elif kind == "admit" and start is not None:
                spans.append((start, t))
                start = None
        if start is not None:
            spans.append((start, wall))
        out[g] = spans
    return out


STATES = ("in_flight", "tool_actions", "tool_python", "host_after_tool", "tokenize", "gate_wait", "dispatch")


def stream_states(per_game, parks, lo, hi):
    """Integrated seconds of each state over [lo, hi] for the games not parked; plus per-request gap parts."""
    acc = Counter()
    gaps = []
    active_seconds = 0.0

    def add(state, a, b, spans):
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            return
        # Subtract parked time (the game is not a focus stream then).
        t = b - a
        for s, e in spans:
            ov = min(b, e) - max(a, s)
            if ov > 0:
                t -= ov
        acc[state] += max(0.0, t)

    for g, rs in per_game.items():
        spans = parks.get(g, [])
        rs = sorted(rs, key=lambda r: r["end"])
        for n, r in enumerate(rs):
            if r["http_start"] is None:
                continue
            add("in_flight", r["http_start"], r["end"], spans)
            nxt = rs[n + 1] if n + 1 < len(rs) else None
            if nxt is None or nxt["http_start"] is None:
                continue
            # The tool round: python (sandbox) and env actions; ends at the last duck_python record.
            tool_end = r["tool_end"] if r["tool_end"] is not None and r["tool_end"] <= nxt["chat_start"] + 0.05 \
                else r["end"]
            act = 0.0
            if r["actions"]:
                act = min(r["last_action"], tool_end) - max(r["first_action"], r["end"])
                act = max(0.0, act)
            tool = max(0.0, tool_end - r["end"])
            count_done = nxt["count_done"] if nxt["count_done"] is not None else nxt["chat_start"]
            admitted = nxt["admitted"] if nxt["admitted"] is not None else count_done
            parts = {
                "tool_actions": act,
                "tool_python": max(0.0, tool - act),
                "host_after_tool": max(0.0, nxt["chat_start"] - tool_end),
                "tokenize": max(0.0, count_done - nxt["chat_start"]),
                "gate_wait": max(0.0, admitted - count_done),
                "dispatch": max(0.0, nxt["http_start"] - admitted),
            }
            # Integrate each part on the time axis in order.
            t = r["end"]
            for state in ("tool_actions", "tool_python", "host_after_tool", "tokenize", "gate_wait", "dispatch"):
                add(state, t, t + parts[state], spans)
                t += parts[state]
            gap = nxt["http_start"] - r["end"]
            parked = nxt.get("gate_parked", 0.0) or 0.0
            in_window = lo <= r["end"] and nxt["http_start"] <= hi
            gaps.append({"game": g, "t": r["end"], "gap": gap, "parked": parked, "in_window": in_window,
                         "actions": r["actions"], "env_seconds": r["env_seconds"], "tools": r["tools"],
                         "purpose_next": nxt["purpose"], "purpose": r["purpose"], "P_next": nxt["P"],
                         "stdout_chars": r["stdout_chars"], **parts})
    return acc, gaps


def occupancy_timeline(per_game, parks, lo, hi, step=5.0):
    """Number of focus games (not parked) and of those in flight, sampled every ``step`` seconds."""
    ts = np.arange(lo, hi, step)
    active = np.zeros(len(ts))
    flight = np.zeros(len(ts))
    for g, rs in per_game.items():
        spans = parks.get(g, [])
        parked = np.zeros(len(ts), bool)
        for s, e in spans:
            parked |= (ts >= s) & (ts < e)
        started = min((r["chat_start"] for r in rs), default=hi)
        ended = max((r["end"] for r in rs), default=lo)
        alive = (ts >= started) & (ts <= ended) & ~parked
        active += alive
        inf = np.zeros(len(ts), bool)
        for r in rs:
            if r["http_start"] is not None:
                inf |= (ts >= r["http_start"]) & (ts < r["end"])
        flight += inf
    return ts, active, flight


# ------------------------------------------------------------------------------------------------ prefill

def granularity_variants(res, units=(4, 16, 64, 400)):
    """Tokens a finer cache granularity would recover from the block tail (continuation requests only).

    ``prompt_tail``: the pinned runtime's partial-tail entry at the previous prompt's last hash boundary
    (``prefix_match_unit`` = u): a hit at max(floor_B(P' + G'), floor_u(P')) instead of floor_B(P' + G').
    ``end_tail``: a tail registered at the previous request's end (reply included): floor_u(P' + G').
    Only non-evicted requests gain (an evicted prefix is gone at any granularity)."""
    block = res["block"]
    out = {}
    reqs = res["requests"]
    for u in units:
        prompt_gain = end_gain = 0.0
        for r in reqs:
            if r.get("cat") != "continuation":
                continue
            keep = 1.0 - (r["est"].get("d_evicted", 0) / r["evict_amount"] if r.get("evict_amount") else 0.0)
            keep = min(1.0, max(0.0, keep))
            end = min(r["prevEnd"], r["P"])
            base = (end // block) * block
            prompt_hit = max(base, (min(r["prevP"], r["P"]) // u) * u)
            end_hit = (end // u) * u
            prompt_gain += keep * (prompt_hit - base)
            end_gain += keep * (end_hit - base)
        out[str(u)] = {"prompt_tail_tokens": round(prompt_gain), "end_tail_tokens": round(end_gain)}
    return out


BREAKDOWN = (
    ("new_tool_results", "new content: tool results"),
    ("new_user_turns", "new content: user turn, board image, template"),
    ("reply_recomputed", "previous reply / older prefix recomputed (state lag)"),
    ("short_gap_shortfall", "hit 1-3 blocks short on an immediate continuation (gap < 90 s)"),
    ("block_tail", "block granularity: tail above the last 1,600-token boundary"),
    ("readmission_evicted", "re-admission after parking / long wait: evicted prefix (gap >= 90 s)"),
    ("compaction_request", "compaction request: tail + summary prompt"),
    ("post_compaction", "first request after compaction (rewritten prompt)"),
    ("first_request", "first request of each game"),
    ("other", "matched-set residual, unmatched and timed-out requests"),
)


def prefill_breakdown(res, fit, measured_by_phase):
    """Uncached tokens by source and phase, the engine seconds they cost at the phase's fitted per-token cost, and
    the run's tok/s gain if a source were removed (seconds freed become decode seconds at the same step mix)."""
    tok = defaultdict(Counter)
    for r in res["requests"]:
        est, ph = r["est"], r["phase"]
        if r.get("cat") == "continuation":
            tool = min(r["tool_tokens"], r["new"])
            tok[ph]["new_tool_results"] += est.get("a_new", 0) * (tool / r["new"] if r["new"] else 0)
            tok[ph]["new_user_turns"] += est.get("a_new", 0) * (1 - (tool / r["new"] if r["new"] else 0))
        tok[ph]["reply_recomputed"] += est.get("lag_reply", 0) + est.get("lag_old", 0)
        ev = est.get("d_evicted", 0)
        if (r.get("gap") or 0) < 90:
            tok[ph]["short_gap_shortfall"] += ev
        else:
            tok[ph]["readmission_evicted"] += ev
        tok[ph]["block_tail"] += est.get("b_tail", 0) + est.get("b_pop", 0)
        tok[ph]["compaction_request"] += est.get("c_compaction_ideal", 0) + est.get("c_compaction_prefix_mismatch", 0)
        tok[ph]["post_compaction"] += est.get("c_post_compaction", 0)
        tok[ph]["first_request"] += est.get("e_first", 0)
        tok[ph]["other"] += est.get("residual", 0)
    tok["phase2"]["other"] += res["unexplained_remainder"]
    cost = {ph: fit[ph]["fit_ms"]["per_computed_prompt_token"] / 1000 for ph in ("opening", "phase2")}
    engine_seconds = fit["opening"]["seconds"] + fit["phase2"]["seconds"]
    prefill_seconds = sum(measured_by_phase[ph] * cost[ph] for ph in cost)
    rows = []
    for key, label in BREAKDOWN:
        o, p = tok["opening"][key], tok["phase2"][key]
        s = o * cost["opening"] + p * cost["phase2"]
        rows.append({"key": key, "label": label, "opening": round(o), "phase2": round(p), "tokens": round(o + p),
                     "engine_seconds": round(s, 1), "share_of_prefill_time": round(s / prefill_seconds, 3),
                     "run_tok_s_gain_if_removed": round(s / (engine_seconds - s), 4)})
    return {"engine_seconds": engine_seconds, "prefill_seconds": round(prefill_seconds, 1),
            "prefill_share_of_engine_time": round(prefill_seconds / engine_seconds, 3),
            "cost_ms_per_token": {k: round(v * 1000, 4) for k, v in cost.items()}, "rows": rows}


def post_compaction_detail(res):
    rows = [r for r in res["requests"] if r.get("cat") == "post_compaction"]
    comp = [r for r in res["requests"] if r.get("cat") == "compaction"]
    return {"requests": len(rows), "prompt": pct([r["P"] for r in rows]),
            "summary_completion": pct([r["G"] for r in comp]),
            "compaction_prompt": pct([r["P"] for r in comp])}


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--skip-prefill", action="store_true")
    args = ap.parse_args()
    trial = Path(args.trial)
    report = json.loads((trial / "arena.json").read_text())
    switch = float(report["scheduler"]["switch_wall_seconds"])
    wall = float(report.get("wall_seconds") or 7200)
    layout = json.loads((trial / "cache-layout.json").read_text())
    num_blocks = int(layout["num_blocks"])
    out = {"trial": str(trial), "switch_wall_seconds": switch, "wall_seconds": wall, "num_blocks": num_blocks}

    # Engine.
    rows = load_telemetry(trial)
    ivs = intervals(rows)
    fit, phase2_ivs = engine_fit(ivs, switch, wall)
    out["engine"] = fit
    out["engine_bootstrap_phase2"] = fit_bootstrap(phase2_ivs)
    out["itl_cross_check"] = itl_cross_check(trial, fit["all"]["mean_decode_batch"],
                                             fit["phase2"]["decode_only_step_ms_at_mean_batch"])
    out["throughput_by_running"] = throughput_by_running(ivs, switch + 30, wall, fit["phase2"])
    out["throughput_by_running_opening"] = throughput_by_running(ivs, 30, switch, fit["opening"])
    out["gen_vs_uncached_rate_phase2"] = prefill_tax_bins(ivs, switch + 30, wall)
    run2 = [i for i in ivs if switch + 30 <= i["t0"] and i["t1"] <= wall]
    T2 = sum(i["dt"] for i in run2)
    out["phase2_running"] = {
        "mean": round(sum(i["running"] * i["dt"] for i in run2) / T2, 2),
        "hist": dict(sorted(Counter(int(i["run_end"]) for i in run2).items())),
        "waiting_nonzero_share": round(sum(i["dt"] for i in run2 if i["waiting"] > 0) / T2, 3),
        "queue_seconds_per_request": round(sum(i["queue_s"] for i in run2) / max(1, sum(i["done"] for i in run2)), 3),
        "prefill_seconds_per_request": round(sum(i["prefill_s"] for i in run2) / max(1, sum(i["done"] for i in run2)), 3),
    }
    # Time series (60 s bins) for the doc.
    series = []
    for t in range(0, int(wall), 300):
        sel = [i for i in ivs if t <= i["t0"] < t + 300]
        if not sel:
            continue
        T = sum(i["dt"] for i in sel)
        series.append({"t": t, "gen_tok_s": round(sum(i["gen"] for i in sel) / T),
                       "computed_tok_s": round(sum(i["computed"] for i in sel) / T),
                       "cached_tok_s": round(sum(i["cached"] for i in sel) / T),
                       "running": round(sum(i["running"] * i["dt"] for i in sel) / T, 1),
                       "kv": round(float(np.mean([i["kv"] for i in sel])), 3),
                       "step_ms": round(T * 1000 / max(1, sum(i["steps"] for i in sel)), 1)})
    out["series_300s"] = series

    # Streams.
    adm, origin = load_admission(trial)
    games = load_games(trial, origin)
    requests = json.loads((trial / "requests.json").read_text())
    offset, http_by_game, off_stats = fit_http_offset(games, requests)
    out["clock"] = {"admission_origin_epoch": origin, "http_offset": off_stats}
    reqs = build_requests(games, http_by_game, offset)
    per_game = attach_gate(reqs, adm)
    parks = park_intervals(adm, wall)
    lo, hi = switch + 30, wall - 5
    acc, gaps = stream_states(per_game, parks, lo, hi)
    total = sum(acc.values())
    span = hi - lo
    ts, active, flight = occupancy_timeline(per_game, parks, lo, hi)
    out["streams_phase2"] = {
        "window": [lo, hi], "mean_focus_games": round(float(active.mean()), 2),
        "mean_in_flight_from_logs": round(float(flight.mean()), 2),
        "state_mean_streams": {s: round(acc[s] / span, 3) for s in STATES},
        "state_share": {s: round(acc[s] / total, 4) for s in STATES},
    }
    g2 = [x for x in gaps if x["in_window"] and x["parked"] < 1.0]
    out["gaps_phase2"] = {
        "requests": len(g2),
        "gap": pct([x["gap"] for x in g2]),
        "parts_mean": {s: round(float(np.mean([x[s] for x in g2])), 3) for s in STATES[1:]},
        "parts": {s: pct([x[s] for x in g2]) for s in STATES[1:]},
        "gap_hist": dict(Counter(min(int(x["gap"] // 1), 30) for x in g2).most_common()),
        "with_actions": pct([x["gap"] for x in g2 if x["actions"]]),
        "without_actions": pct([x["gap"] for x in g2 if not x["actions"]]),
        "actions_per_request_with_actions": pct([x["actions"] for x in g2 if x["actions"]]),
        "seconds_per_action": round(sum(x["tool_actions"] for x in g2) / max(1, sum(x["actions"] for x in g2)), 4),
        "env_step_seconds_per_action": round(sum(x["env_seconds"] for x in g2) /
                                             max(1, sum(x["actions"] for x in g2)), 4),
        "tokenize_by_prompt": [
            {"P": f"{a // 1000}-{b // 1000}K", **pct([x["tokenize"] for x in g2 if a <= x["P_next"] < b], (50, 90))}
            for a, b in ((0, 16000), (16000, 32000), (32000, 48000), (48000, 70000))
        ],
    }
    # Requests still in flight when their game was parked (they run to completion on a parked game).
    tail = Counter()
    for g, rs in per_game.items():
        for r in rs:
            if r["http_start"] is None or r["end"] < lo:
                continue
            for s, e in parks.get(g, []):
                ov = min(r["end"], e, hi) - max(r["http_start"], s, lo)
                if ov > 0:
                    tail["requests"] += 1
                    tail["seconds"] += ov
                    tail["generated"] += r["G"]
                    tail["started_while_parked"] += r["http_start"] >= s
    out["parked_in_flight_phase2"] = {"requests": tail["requests"], "mean_streams": round(tail["seconds"] / span, 3),
                                      "generated": tail["generated"],
                                      "started_while_parked": tail["started_while_parked"]}
    # Gate waits in phase 2: was the slot limit (inflight = slots) or the block capacity binding?
    slots = int(report["gate"]["slots"])
    waits = [r for r in adm if r.get("event") == "admit" and "reason" not in r and (r.get("waited") or 0) > 0.05
             and r["wall_seconds"] >= lo]
    out["gate_waits_phase2"] = {
        "admits_waiting_over_50ms": len(waits),
        "wait_seconds_total": round(sum(r["waited"] for r in waits), 1),
        "at_slot_limit_share": round(sum(r["waited"] for r in waits if r.get("inflight_after") == slots) /
                                     max(1e-9, sum(r["waited"] for r in waits)), 3),
        "max_reserved_after": max((r["reserved_after"] for r in waits), default=None),
    }
    gp = [x for x in gaps if x["in_window"] and x["parked"] >= 1.0]
    out["readmissions_phase2"] = {"requests": len(gp), "parked_wait": pct([x["parked"] for x in gp])}
    go = [x for x in gaps if x["t"] < switch and x["parked"] < 1.0]
    out["gaps_opening"] = {"requests": len(go), "gap": pct([x["gap"] for x in go]),
                           "parts_mean": {s: round(float(np.mean([x[s] for x in go])), 3) for s in STATES[1:]}}
    # Serialization check: action spacing against the env step's own time.
    spacing = []
    for g, events in games.items():
        obs = [e["t"] for e in events if e["kind"] == "observation" and e.get("action") is not None]
        spacing += [b - a for a, b in zip(obs, obs[1:]) if b - a < 2.0]
    out["action_spacing"] = pct(spacing)
    # In-flight request durations.
    fl = [r for r in reqs if r["http_start"] is not None and lo <= r["http_start"] and r["end"] <= hi]
    out["in_flight_phase2"] = {"seconds": pct([r["end"] - r["http_start"] for r in fl]),
                               "completion": pct([r["G"] for r in fl]),
                               "per_token_ms": round(sum(r["end"] - r["http_start"] for r in fl) * 1000 /
                                                     max(1, sum(r["G"] for r in fl)), 2)}

    # KV use vs reservation.
    res_rows = [r for r in adm if r.get("event") in ("admit", "release") and "reserved_after" in r]
    res_t = np.array([r["wall_seconds"] for r in res_rows])
    res_v = np.array([r["reserved_after"] for r in res_rows])
    order = np.argsort(res_t)
    res_t, res_v = res_t[order], res_v[order]
    kv_rows = []
    for i in ivs:
        if i["t1"] > wall:
            continue
        j = np.searchsorted(res_t, i["t1"]) - 1
        reserved = float(res_v[j]) if j >= 0 else 0.0
        kv_rows.append((i["t1"], i["kv"] * num_blocks, reserved, i["running"]))
    cap = report["window_choice"]["capacity"]

    def kv_summary(sel):
        used = np.array([x[1] for x in sel])
        resv = np.array([x[2] for x in sel])
        return {"used_blocks": pct(used, (50, 90, 99)), "reserved_blocks": pct(resv, (50, 90, 99)),
                "used_over_reserved": round(float(used.mean() / resv.mean()), 3),
                "used_share_of_pool": round(float(used.mean() / num_blocks), 3),
                "reserved_share_of_gate": round(float(resv.mean() / cap), 3)}

    out["kv"] = {"pool_blocks": num_blocks, "gate_capacity_blocks": cap,
                 "opening": kv_summary([x for x in kv_rows if 30 <= x[0] <= switch]),
                 "phase2": kv_summary([x for x in kv_rows if x[0] >= switch + 30]),
                 "peak_used_share": round(max(x[1] for x in kv_rows) / num_blocks, 3)}
    # In-flight context distribution (phase 2), tokens at request start and at end.
    out["kv"]["in_flight_prompt_phase2"] = pct([r["P"] for r in fl])
    out["kv"]["in_flight_end_phase2"] = pct([r["P"] + r["G"] for r in fl])
    blocks_per_req = [math.ceil((r["P"] + r["G"] / 2) / 1600) + 9 for r in fl]
    out["kv"]["mean_blocks_held_per_request_est"] = round(float(np.mean(blocks_per_req)), 1)

    # Prefill attribution.
    if not args.skip_prefill:
        import prefill_attribution as pa
        count = pa.TokenCounter(args.tokenizer or pa.TOKENIZER)
        res = pa.attribute(trial, count)
        pa.link_prev_reasoning(res["requests"])
        summary = pa.summarize(res)
        out["prefill"] = {"summary": summary, "granularity": granularity_variants(res),
                          "post_compaction": post_compaction_detail(res)}
        out["prefill"]["breakdown"] = prefill_breakdown(
            res, fit, {"opening": fit["opening"]["computed"], "phase2": fit["phase2"]["computed"]})
        # Tail by phase and per request.
        cont = [r for r in res["requests"] if r.get("cat") == "continuation"]
        out["prefill"]["tail_per_request"] = pct([r["tail"] for r in cont])
        out["prefill"]["new_per_request"] = pct([r["new"] for r in cont])
        ev = [r for r in res["requests"] if r["est"].get("d_evicted", 0) > 1000]
        out["prefill"]["evicted_requests"] = {
            "n": len(ev), "tokens": pct([r["est"]["d_evicted"] for r in ev]),
            "gap": pct([r["gap"] for r in ev]),
            "phase2_n": sum(1 for r in ev if r["phase"] == "phase2"),
        }
        by_gap = defaultdict(Counter)
        for r in res["requests"]:
            e = r["est"].get("d_evicted", 0)
            if e > 0 and r.get("gap") is not None:
                b = pa.GAP_EDGES[pa.gap_bucket(r["gap"])]
                by_gap[r["phase"]][f">={b:g}s"] += e
        out["prefill"]["evicted_by_gap"] = {ph: {k: round(v) for k, v in sorted(c.items(), key=lambda x: float(
            x[0][2:-1]))} for ph, c in by_gap.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("series_300s",)}, indent=1)[:20000])


if __name__ == "__main__":
    main()
