"""Where the uncached (computed) prompt tokens of an arena run go (28 September, CPU only).

Measurement. The server exports cumulative prefix-cache counters (``telemetry.jsonl``, a sample every ~5 s)
but no per-request cache hits. Every served request's exact prompt length is in the game logs (``model``
records), and a request's prefix-cache lookup is counted once, when it is first scheduled. So for each
telemetry interval the script finds the set of requests sent in (or within ``slack`` seconds of) the interval
whose prompt tokens sum *exactly* to the interval's ``prefix_cache_queries`` delta; the interval's
``prefix_cache_hits`` delta is then the exact hit total of that set, and a request's exact hit when the set has
one member ("single"). The clock offset between the game logs and the telemetry is fitted first.

Decomposition. Each request's uncached tokens U = P - H (P prompt tokens, H cached tokens) are split into
exact parts against an *ideal* append-only cache that keeps every prefix and every block-boundary state:

- ``continuation`` (P follows the previous gameplay request of the same game, prompt P', reply G'):
  ``new`` = P - (P' + G')  (tool results, and at a turn start the user turn and its board image);
  ``tail`` = (P' + G') - floor((P' + G') / B) * B  (the block tail: whole blocks only are reused);
  ``pop`` = B when the layout is speculative (``cache-layout.json`` lookahead > 0, MTP): the scheduler drops the
  last hit block ("eagle pop"), visible in the MTP-3 runs as hits that are never within one block of ideal;
  ``shortfall`` = floor((P' + G') / B) * B - pop - H, split into
    ``evicted``: the game's prefix was gone (H <= the prefix shared by all games);
    ``lag_reply``: the part of the shortfall inside the previous reply [P', P' + G') (reasoning + tool call
      recomputed although the ideal cache would reuse it);
    ``lag_old``: the part below P' (the hit fell back to an older block boundary).
- ``first``: the first request of a game (U = P - S, S the block-floored prefix shared by all games);
- ``compaction``: the tool-less summary request; ``post_compaction``: the next gameplay request, whose prompt was
  rewritten to [system, summary, latest observation].

For singles H is exact. For the other matched requests the set total is exact and the shortfall is shared out
with a model fitted on the same run: a request is evicted with a probability that depends on the idle gap
since the game's previous reply (fitted by bounded least squares on the exact set totals), and otherwise loses
``lag = k x`` where x is the previous request's growth (P' + G' - P'') and k is fitted on the singles.
Unmatched requests (and the 6-odd timed-out requests, which the logs do not record) are the remainder to the
engine's measured total.

Token counts of tool results and user turns use the served tokenizer (``runs/tokenizer-cache``) when the
``tokenizers`` package is importable, else characters / 3.0 (the logs' ``token_count`` records calibrate the
served prompts at 2.9-3.4 characters per token).

    uv run --with tokenizers python scripts/arena_analysis/prefill_attribution.py \
        --trial <run>/<run>/arena/trials/duck-transfer --out attribution.json
"""

import argparse
import bisect
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TOKENIZER = ("./runs/tokenizer-cache/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/"
             "7b719225242aacd3dbd3f9407468c2ee9a9d2594/tokenizer.json")
PREFILL_TOK_S = 13_000  # MTP-0 A+C prefill, kernels-phase2.md (12.8-13.7K tok/s at 4-16K prompts)
BOARD_ROW = re.compile(r"^[\s'\"\[\],0-9:]*[WwgGcBMPRbSYOrNp]{16,}[\s'\",\]]*$")
IMAGE_LABEL = "\n\nGrid image of the board at step {step} (before this turn's actions):"
GAP_EDGES = (0, 5, 30, 60, 90, 120, 180, 1e9)


class TokenCounter:
    """The served model's tokenizer when ``tokenizers`` is importable, else characters / ratio."""

    def __init__(self, path, ratio=3.0):
        self.ratio, self.tok = ratio, None
        try:
            from tokenizers import Tokenizer
            if path and Path(path).is_file():
                self.tok = Tokenizer.from_file(str(path))
        except ImportError:
            pass

    def __call__(self, text):
        if not text:
            return 0
        if self.tok is not None:
            return len(self.tok.encode(text, add_special_tokens=False).ids)
        return round(len(text) / self.ratio)


Counter_ = TokenCounter  # board_encoding_tokens.py


def transcript_user_prompts(path):
    """The [USER PROMPT] text of every analysis turn, in order."""
    if not path.is_file():
        return []
    prompts = []
    for block in path.read_text(errors="replace").split("\n--- analysis_step="):
        i = block.find("[USER PROMPT]\n")
        if i < 0:
            continue
        rest = block[i + len("[USER PROMPT]\n"):]
        m = re.search(r"\n\[[A-Z][A-Z :_]*\]\n", rest)
        prompts.append(rest[: m.start()] if m else rest)
    return prompts


def load_requests(trial, count):
    t0 = json.loads((trial / "execution.json").read_text())["started_epoch"]
    reqs = []
    for game_dir in sorted((trial / "games").iterdir()):
        log = game_dir / "log.jsonl"
        if not log.is_file():
            continue
        prompts = transcript_user_prompts(game_dir / "duck-transcript.txt")
        turn, steps, idx, prev_purpose = 0, 0, 0, None
        pending_start, tools, new_turn, board_rows = None, [], False, 0
        for line in log.open():
            d = json.loads(line)
            kind = d.get("kind")
            if kind == "model_request":
                pending_start = d["time"]
            elif kind == "observation":
                new_turn = new_turn or d.get("action") is not None
                steps = d.get("step", steps)
            elif kind == "duck_python":
                result = d.get("result") or {}
                tools.append(json.dumps(result, indent=2))
                stdout = str(result.get("stdout") or "")
                board_rows += sum(count(ln + "\n") for ln in stdout.split("\n") if BOARD_ROW.match(ln))
            elif kind == "model":
                usage = d.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                purpose = d.get("purpose")
                gameplay = purpose == "gameplay"
                start = pending_start if (pending_start is not None and gameplay) else d["time"] - d.get("seconds", 0)
                user_text = 0
                if gameplay and new_turn:
                    turn += 1
                    if turn < len(prompts):
                        user_text = count(prompts[turn] + IMAGE_LABEL.format(step=steps))
                reqs.append({
                    "game": game_dir.name, "i": idx, "purpose": purpose, "start": start - t0, "end": d["time"] - t0,
                    "P": int(usage.get("prompt_tokens") or 0), "G": int(usage.get("completion_tokens") or 0),
                    "R": int(details.get("reasoning_tokens") or 0), "turn_start": bool(gameplay and new_turn),
                    "tool_tokens": sum(count(t) for t in tools), "board_row_tokens": board_rows,
                    "user_text_tokens": user_text, "after_compaction": prev_purpose == "compaction",
                })
                idx += 1
                if gameplay:
                    pending_start, tools, new_turn, board_rows = None, [], False, 0
                prev_purpose = purpose
    return reqs


def load_telemetry(trial):
    tel = []
    for line in (trial / "telemetry.jsonl").open():
        r = json.loads(line)
        m = r["metrics"]
        tel.append((r["elapsed_seconds"], m["vllm:prefix_cache_queries_total"], m["vllm:prefix_cache_hits_total"]))
    return tel


def fit_offset(reqs, tel):
    q0 = tel[0][1]
    order = sorted(reqs, key=lambda r: r["start"])
    starts = [r["start"] for r in order]
    cum = [0]
    for r in order:
        cum.append(cum[-1] + r["P"])

    def err(off):
        return sum(abs((q - q0) - cum[bisect.bisect_right(starts, t + off)]) for t, q, _ in tel[1:]) / len(tel)

    return min((err(o / 10), o / 10) for o in range(-100, 301))[::-1]


def match_intervals(reqs, tel, offset, slack=2.5, max_set=6):
    """[(request indices, hits)] for every interval whose query delta is an exact sum of nearby prompts."""
    order = sorted(range(len(reqs)), key=lambda j: reqs[j]["start"])
    starts = [reqs[j]["start"] - offset for j in order]
    assigned, sets = set(), []
    for k in range(1, len(tel)):
        (ta, qa, ha), (tb, qb, hb) = tel[k - 1], tel[k]
        dq = qb - qa
        if dq <= 0:
            continue
        lo, hi = bisect.bisect_right(starts, ta - slack), bisect.bisect_right(starts, tb + slack)
        cands = [order[x] for x in range(lo, hi) if order[x] not in assigned]
        found = None
        for n in range(1, min(len(cands), max_set) + 1):
            found = next((s for s in itertools.combinations(cands, n) if sum(reqs[j]["P"] for j in s) == dq), None)
            if found:
                break
        if found:
            assigned.update(found)
            sets.append((list(found), int(hb - ha)))
    return sets


def gap_bucket(gap):
    return max(0, bisect.bisect_right(GAP_EDGES, gap or 0) - 1)


def prepare(reqs, block, pop=0):
    """Per-game links and exact ideal parts; returns request indices by game."""
    by_game = defaultdict(list)
    for j, r in enumerate(reqs):
        by_game[r["game"]].append(j)
    for js in by_game.values():
        gp, resident, epoch = [], Counter(), []
        for n, j in enumerate(js):
            r = reqs[j]
            r["gap"] = r["start"] - reqs[js[n - 1]]["end"] if n else None
            if n == 0:
                r["cat"] = "first"
            elif r["purpose"] == "compaction":
                r["cat"] = "compaction"
            elif r["after_compaction"]:
                r["cat"] = "post_compaction"
            else:
                r["cat"] = "continuation"
            if r["cat"] in ("first", "post_compaction"):
                resident = Counter()
                epoch = []  # gameplay requests since the last prompt rewrite
                if r["cat"] == "post_compaction":
                    resident["summary"] = reqs[js[n - 1]]["G"]
            r["resident"] = dict(resident)
            prev = gp[-1] if gp else None
            if r["cat"] in ("continuation", "compaction") and prev is not None:
                end = prev["P"] + prev["G"]
                r["prevP"], r["prevEnd"] = prev["P"], end
                floor = (min(end, r["P"]) // block) * block
                # With a speculative (MTP) layout the scheduler drops the last hit block ("eagle pop").
                r["pop"] = min(pop, floor)
                r["ideal_hit"] = floor - r["pop"]
                if r["cat"] == "continuation":
                    r["new"] = max(0, r["P"] - end)
                    r["tail"] = min(end, r["P"]) - floor
                    # Growth over the previous request (same prompt epoch; else the previous reply alone).
                    r["x"] = end - epoch[-2]["P"] if len(epoch) > 1 else prev["G"]
            if r["purpose"] == "gameplay":
                gp.append(r)
                epoch.append(r)
                resident["reasoning"] += r["R"]
                resident["tool_call"] += r["G"] - r["R"]
                if r["cat"] == "continuation":
                    resident["tool_results"] += r["tool_tokens"]
                    resident["board_rows"] += r["board_row_tokens"]
                    resident["user_turns"] += r["new"] - r["tool_tokens"]
    return by_game


def attribute(trial, count):
    trial = Path(trial)
    layout = json.loads((trial / "cache-layout.json").read_text())
    block = max(g["block_size"] for g in layout["groups"])
    report = json.loads((trial / "arena.json").read_text())
    engine = report.get("engine") or {}
    switch = (report.get("scheduler") or {}).get("switch_wall_seconds")
    reqs = load_requests(trial, count)
    tel = load_telemetry(trial)
    offset, fit_err = fit_offset(reqs, tel)
    sets = match_intervals(reqs, tel, offset)
    for found, dh in sets:
        for j in found:
            reqs[j]["set_size"] = len(found)
        if len(found) == 1:
            reqs[found[0]]["H"] = dh
    pop = block if int(layout.get("lookahead") or 0) > 0 else 0
    prepare(reqs, block, pop)
    min_first = min(r["P"] for r in reqs if r["cat"] == "first")
    small = [r["H"] for r in reqs if r.get("H") and r["H"] <= min_first]
    shared = Counter(small).most_common(1)[0][0] if small else 0
    for r in reqs:
        r["phase"] = "opening" if switch is not None and r["start"] < switch else "phase2"
        if r["cat"] in ("first", "post_compaction"):
            r["ideal_hit"] = min(shared, r["P"])
        r["ideal_uncached"] = r["P"] - r.get("ideal_hit", 0)
        if r["cat"] == "continuation":
            evict = r["ideal_hit"] - min(shared, r["ideal_hit"])
            # Where a lag of L tokens lands: first inside the previous reply, then below P'.
            r["evict_amount"] = evict

    # Lag model from non-evicted continuation singles: shortfall = k * x.
    xs, ys = [], []
    for r in reqs:
        if r["cat"] == "continuation" and r.get("H") is not None and r["H"] > shared:
            xs.append(r["x"])
            ys.append(max(0, r["ideal_hit"] - r["H"]))
    x, y = np.array(xs, float), np.array(ys, float)
    k = float((x * y).sum() / (x * x).sum()) if len(x) else 0.0
    lag_r2 = float(1 - ((y - k * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(x) > 2 else None

    def lag(r):
        return min(k * r["x"], r["evict_amount"])

    # Compaction requests: measured hit of singles (0 when the summary request's system block differs).
    comp_hits = [r["H"] for r in reqs if r["cat"] == "compaction" and r.get("H") is not None]
    # Compaction requests miss the game's prefix when their hits never exceed the shared prefix (their system
    # block lacks the reasoning-effort line); they then hit only what other compaction requests left (0-4 blocks).
    comp_zero = bool(comp_hits) and all(h <= shared for h in comp_hits)
    comp_hit = float(np.mean(comp_hits)) if comp_zero else None

    # Eviction probability per gap bucket from exact set totals (bounded least squares).
    from scipy.optimize import lsq_linear
    nb = len(GAP_EDGES) - 1
    rows, rhs = [], []
    for found, dh in sets:
        a, const = np.zeros(nb), 0.0
        for j in found:
            r = reqs[j]
            if r["cat"] == "continuation":
                b = gap_bucket(r["gap"])
                const += r["ideal_uncached"] + lag(r)
                a[b] += r["evict_amount"] - lag(r)
            elif r["cat"] == "compaction":
                const += r["P"] - comp_hit if comp_zero else r["ideal_uncached"]
            else:
                const += r["ideal_uncached"]
        measured = sum(reqs[j]["P"] for j in found) - dh
        rows.append(a)
        rhs.append(measured - const)
    A, bvec = np.array(rows), np.array(rhs)
    used = A.any(axis=0)
    p = np.zeros(nb)
    if used.any():
        p[used] = lsq_linear(A[:, used], bvec, bounds=(0, 1)).x
    bucket_n = Counter(gap_bucket(r["gap"]) for r in reqs if r["cat"] == "continuation")

    # Predicted components per request.
    for r in reqs:
        comp = Counter()
        if r["cat"] == "continuation":
            e = p[gap_bucket(r["gap"])]
            comp["a_new"] = r["new"]
            comp["b_tail"] = r["tail"]
            comp["b_pop"] = r["pop"]
            comp["d_evicted"] = e * r["evict_amount"]
            L = (1 - e) * lag(r)
            reply_room = max(0, r["ideal_hit"] - r["prevP"])
            comp["lag_reply"] = min(L, reply_room)
            comp["lag_old"] = L - comp["lag_reply"]
        elif r["cat"] == "compaction":
            comp["c_compaction_ideal"] = r["ideal_uncached"]
            if comp_zero:
                comp["c_compaction_prefix_mismatch"] = r["P"] - comp_hit - r["ideal_uncached"]
        elif r["cat"] == "post_compaction":
            comp["c_post_compaction"] = r["ideal_uncached"]
        else:
            comp["e_first"] = r["ideal_uncached"]
        r["pred"] = comp
    # Exact per-set reconciliation: move each set's residual onto its uncertain parts.
    uncertain = ("d_evicted", "lag_reply", "lag_old", "c_compaction_prefix_mismatch")
    matched = set()
    for found, dh in sets:
        measured = sum(reqs[j]["P"] for j in found) - dh
        pred = sum(sum(reqs[j]["pred"].values()) for j in found)
        soft = sum(sum(reqs[j]["pred"][u] for u in uncertain) for j in found)
        resid = measured - pred
        for j in found:
            matched.add(j)
            r = reqs[j]
            est = Counter(r["pred"])
            if soft > 0:
                scale = max(0.0, 1 + resid / soft)
                for u in uncertain:
                    est[u] *= scale
                if 1 + resid / soft < 0:  # over-prediction beyond the soft parts: rest is negative residual
                    est["residual"] += (resid + soft) / len(found)
            else:
                est["residual"] += resid / len(found)
            r["est"] = est
    for j, r in enumerate(reqs):
        if j not in matched:
            r["est"] = Counter(r["pred"])
    total_est = sum(sum(r["est"].values()) for r in reqs)
    measured_total = int(engine.get("computed_prompt_tokens") or 0)
    return {
        "trial": str(trial), "block": block, "shared_prefix": shared, "switch": switch,
        "offset_s": offset, "fit_error_tokens": round(fit_err), "measured_uncached": measured_total,
        "measured_cached": int(engine.get("cached_prompt_tokens") or 0),
        "generated": int(engine.get("accepted_generation_tokens") or 0),
        "wall_seconds": report.get("wall_seconds"), "lag_k": k, "lag_r2": lag_r2, "lag_n": len(x),
        "evict_p_by_gap": {f"{GAP_EDGES[b]:g}-{GAP_EDGES[b + 1]:g}s": {"p": round(float(p[b]), 3), "n": bucket_n[b]}
                           for b in range(nb)},
        "compaction_hits_zero": comp_zero, "compaction_single_hits": comp_hits,
        "matched_intervals": len(sets), "matched_requests": len(matched),
        "single_requests": sum(1 for f, _ in sets if len(f) == 1),
        "unexplained_remainder": measured_total - total_est, "requests": reqs, "sets": sets,
    }


def summarize(res):
    reqs = res["requests"]
    tot, by_phase = Counter(), defaultdict(Counter)
    for r in reqs:
        for key, v in r["est"].items():
            tot[key] += v
            by_phase[r["phase"]][key] += v
        tot["P"] += r["P"]
        tot["G"] += r["G"]
    tot["unmatched_remainder"] = res["unexplained_remainder"]
    measured = res["measured_uncached"]
    labels = [
        ("a_new", "(a) new content: tool results, user turn + board image"),
        ("b_tail", "(b) block tail of the previous prompt+reply"),
        ("b_pop", "(b) last hit block dropped by the MTP (speculative) layout"),
        ("lag_reply", "(b') hybrid-state lag: previous reply (reasoning + tool call) recomputed"),
        ("lag_old", "(b') hybrid-state lag: older prefix recomputed"),
        ("c_compaction_ideal", "(c) compaction request: tail + summary prompt"),
        ("c_compaction_prefix_mismatch", "(c) compaction request: full re-prefill (system block differs)"),
        ("c_post_compaction", "(c) first request after compaction (rewritten prompt)"),
        ("d_evicted", "(d) evicted game prefix re-prefilled"),
        ("e_first", "(e) first request of each game"),
        ("residual", "matched-set residual (model misfit)"),
        ("unmatched_remainder", "unmatched requests, timed-out requests"),
    ]
    table = []
    for key, label in labels:
        v = tot.get(key, 0)
        table.append({"key": key, "label": label, "tokens": round(v), "share": round(v / measured, 4) if measured else None,
                      "gpu_s": round(v / PREFILL_TOK_S, 1),
                      "opening": round(by_phase["opening"].get(key, 0)), "phase2": round(by_phase["phase2"].get(key, 0))})
    # Composition of the recomputed old prefix (evicted + lag_old): resident parts of each request's prompt.
    comp = Counter()
    for r in reqs:
        old = r["est"].get("d_evicted", 0) + r["est"].get("lag_old", 0)
        res_ = r.get("resident") or {}
        size = sum(res_.values())
        if old > 0 and size > 0:
            for key, v in res_.items():
                comp[key] += old * v / size
    reply_reasoning = sum(r["est"].get("lag_reply", 0) * (reqs_prev_reasoning_share(reqs, r)) for r in reqs)
    new_detail = Counter()
    for r in reqs:
        if r["cat"] == "continuation":
            new_detail["tool_results"] += r["tool_tokens"]
            new_detail["board_rows_in_tool_output"] += r["board_row_tokens"]
            new_detail["user_turns_incl_image_and_template"] += r["new"] - r["tool_tokens"]
            new_detail["turn_starts"] += r["turn_start"]
    return {
        "block": res["block"], "shared_prefix": res["shared_prefix"], "requests": len(reqs),
        "gameplay": sum(r["purpose"] == "gameplay" for r in reqs),
        "compactions": sum(r["cat"] == "compaction" for r in reqs),
        "measured_uncached": measured, "measured_cached": res["measured_cached"], "served_prompt": tot["P"],
        "generated": res["generated"], "prefill_gpu_s": round(measured / PREFILL_TOK_S),
        "table": table, "old_prefix_composition": {k: round(v) for k, v in comp.items()},
        "reply_lag_reasoning_tokens": round(reply_reasoning), "new_content_detail": dict(new_detail),
        "fit": {k: res[k] for k in ("offset_s", "fit_error_tokens", "matched_intervals", "matched_requests",
                                   "single_requests", "lag_k", "lag_r2", "lag_n", "evict_p_by_gap",
                                   "compaction_hits_zero", "compaction_single_hits", "switch")},
    }


def reqs_prev_reasoning_share(reqs, r):
    """Reasoning share of the previous gameplay reply (for the recomputed-reply part)."""
    prev = r.get("_prev")
    return prev if prev is not None else 0.0


def link_prev_reasoning(reqs):
    last = {}
    for r in reqs:
        g = r["game"]
        prev = last.get(g)
        r["_prev"] = (prev["R"] / prev["G"]) if prev and prev["G"] else 0.0
        if r["purpose"] == "gameplay":
            last[g] = r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--chars-per-token", type=float, default=3.0)
    args = parser.parse_args()
    count = TokenCounter(args.tokenizer, args.chars_per_token)
    res = attribute(args.trial, count)
    link_prev_reasoning(res["requests"])
    summary = summarize(res)
    summary["tokenizer"] = "served tokenizer" if count.tok is not None else f"chars/{args.chars_per_token}"
    slim = [{k: (dict(v) if isinstance(v, Counter) else v) for k, v in r.items() if k not in ("pred",)}
            for r in res["requests"]]
    Path(args.out).write_text(json.dumps({"summary": summary, "requests": slim, "sets": res["sets"]}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
