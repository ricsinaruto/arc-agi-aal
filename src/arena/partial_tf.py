"""Teacher-forced correctness of partial-tail prefix hits (``--prefix-match-unit``), 29 September (prefix-unit-int4.md).

Needs the warm prompt-logprobs patch (``arena.prompt_logprobs_patch``, lab env ``ARENA_WARM_PROMPT_LOGPROBS=1``): a
request with ``prompt_logprobs`` and ``vllm_xargs.arc_read_prefix_cache`` reads the prefix cache, so one request
resumes from a cached state and teacher-forces every position after it.

Design (stated before the run; the pass rule is ``verdict``):

- Per recorded arena context (token ids) and KV block ``b`` = ``block_index`` x block, the scored span ends at
  ``E = b + block + tail_tokens`` (the rest of the partial block and the start of the next one).
- Arms, one request each (a warm arm is its primer, then the warm request):
  - ``C``, ``C2``: cold ``prompt_logprobs`` on ids[:E] (a cold request never reads the cache); C2 repeats C in
    another batch composition, so d(C2, C) is the cold-vs-cold noise;
  - ``F``: warm from the full-block state at ``b`` (primer ids[:b + unit/2]);
  - ``P1..P3``: warm from the partial-tail state at ``b + o`` for each offset ``o`` (primer ids[:b + o + unit/2],
    whose last unit boundary is b + o); offsets are multiples of 64 so the recurrent chunk grid is not shifted;
  - ``S``: negative control, cold on ids[:b] + ids[b + o1:E] (the first offset's span deleted: the scale of a state
    that skipped those tokens).
- Batch structure: a Latin square. In phase p every context runs arm (index + p) mod 7, ``lanes`` contexts at a
  time, and the prefix cache is reset between phases. Every arm meets the same kind of batch neighbours, and no arm
  can hit another arm's state.
- Metric: per position, the symmetric KL on {the union of both top-k lists} + {rest}, a token missing from one list
  taking that list's smallest listed probability (``_union_kl``: disjoint lists count as far apart instead of being
  dropped); scored positions are those after each partial state (b + o + 1 .. E - 1).
- Statistic: R_X = sum d(X, C) / sum d(C2, C) over the same positions, pooled over the three offsets; 95% interval
  by a cluster bootstrap over contexts.
- Pass: every partial and full-block warm request hits exactly its state; R_P's 95% upper bound <= 1.25; and the
  negative control's R_S 5% lower bound > 1.25 (else the test has no power: inconclusive). Fail: R_P's 5% lower
  bound > 1.25. Anything else is inconclusive.
"""

import asyncio
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .isolation import isolate_trial

ARMS = ("C", "C2", "F", "P1", "P2", "P3", "S")
DEFAULTS = {"block": 3136, "unit": 16, "block_index": 3, "offsets": [768, 1536, 2304], "tail_tokens": 1024,
            "top_logprobs": 5, "lanes": 8, "bootstrap": 4000, "bar": 1.25, "near_positions": 256}


def layout(spec):
    spec = {**DEFAULTS, **spec}
    block, unit = int(spec["block"]), int(spec["unit"])
    b = int(spec["block_index"]) * block
    offsets = [int(o) for o in spec["offsets"]]
    if len(offsets) != 3 or any(o % 64 or o % unit or not 0 < o < block for o in offsets):
        raise ValueError("partial_tf offsets: three multiples of 64 (and of the unit) inside the block")
    end = b + block + int(spec["tail_tokens"])
    return spec, block, unit, b, offsets, end


def arm_requests(ids, spec):
    """name -> (primer ids or None, scored ids, expected cached tokens or None, position shift, warm flag)."""
    spec, block, unit, b, offsets, end = layout(spec)
    arms = {"C": (None, ids[:end], None, 0, False), "C2": (None, ids[:end], None, 0, False),
            "F": (ids[:b + unit // 2], ids[:end], b, 0, True),
            "S": (None, ids[:b] + ids[b + offsets[0]:end], None, offsets[0], False)}
    for k, o in enumerate(offsets, 1):
        arms[f"P{k}"] = (ids[:b + o + unit // 2], ids[:end], b + o, 0, True)
    return arms


def rows(entries, first, last, shift, top):
    """ids / logprobs arrays for original positions first..last-1 (entry index = position - shift); -1 / -inf pad."""
    n = max(0, last - first)
    out_ids = np.full((n, top + 1), -1, dtype=np.int32)
    out_lps = np.full((n, top + 1), -np.inf, dtype=np.float32)
    for i in range(n):
        entry = entries[first + i - shift] or {}
        ranked = sorted(((int(str(t).split(":")[-1]), float(v["logprob"] if isinstance(v, dict) else v))
                         for t, v in entry.items()), key=lambda r: -r[1])[:top + 1]
        for j, (t, lp) in enumerate(ranked):
            out_ids[i, j], out_lps[i, j] = t, lp
    return out_ids, out_lps


def as_dist(ids, lps, i, top):
    # the top-k of a position: its k highest (the entry also holds the recorded token when it ranks lower)
    return {int(t): float(lp) for t, lp in zip(ids[i][:top], lps[i][:top]) if t >= 0}


def _union_kl(p, q):
    """KL(p || q) on {the union of both top-k lists} + {rest}; a token missing from a list gets that list's smallest
    listed probability (the most it can have), so disjoint lists count as far apart instead of being dropped."""
    fp, fq = min(p.values()), min(q.values())
    tokens = set(p) | set(q)
    ps = {t: math.exp(p.get(t, fp)) for t in tokens}
    qs = {t: math.exp(q.get(t, fq)) for t in tokens}
    total_p, total_q = sum(ps.values()), sum(qs.values())
    if total_p > 1.0:
        ps = {t: v / total_p for t, v in ps.items()}
    if total_q > 1.0:
        qs = {t: v / total_q for t, v in qs.items()}
    rest_p = max(1.0 - sum(ps.values()), 1e-9)
    rest_q = max(1.0 - sum(qs.values()), 1e-9)
    kl = sum(v * math.log(v / qs[t]) for t, v in ps.items()) + rest_p * math.log(rest_p / rest_q)
    return max(kl, 0.0)


def sym_kl(a, b):
    if not a or not b:
        return None
    return 0.5 * (_union_kl(a, b) + _union_kl(b, a))


async def completion(client, base_url, body):
    response = await client.post(base_url + "/completions", json=body)
    if response.status_code != 200:
        raise RuntimeError(f"completions {response.status_code}: {response.text[:300]}")
    return response.json()


async def run_arm(client, base_url, model, ids, name, spec, store):
    primer, scored, expected, shift, warm = arm_requests(ids, spec)[name]
    spec, block, unit, b, offsets, end = layout(spec)
    top = int(spec["top_logprobs"])
    started = time.monotonic()
    if primer is not None:
        await completion(client, base_url, {"model": model, "prompt": primer, "max_tokens": 1, "temperature": 0.0})
    body = {"model": model, "prompt": scored, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": top}
    if warm:
        body["vllm_xargs"] = {"arc_read_prefix_cache": 1}
    reply = await completion(client, base_url, body)
    entries = reply["choices"][0].get("prompt_logprobs") or []
    cached = ((reply.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    # The served output keeps only the computed rows of a warm request (v1: [None] + positions cached + 1 ..), so
    # entry j is position cached + j; a cold request's list is the whole prompt.
    if len(entries) == len(scored):
        offset = 0
    elif warm and cached and len(entries) == len(scored) - cached:
        offset = cached
    else:
        raise RuntimeError(f"{name}: prompt_logprobs length {len(entries)} for {len(scored)} tokens, {cached} cached")
    first = b + 1 + (offsets[0] if name == "S" else 0)
    if warm:
        first = max(first, cached + 1)
    ids_arr, lps_arr = await asyncio.to_thread(rows, entries, first, end, shift + offset, top)
    # positions a warm request did not compute (at or before its hit) carry no logprobs
    valid_from = (cached + 1) if warm else first
    store[name] = {"ids": ids_arr, "lps": lps_arr, "first": first, "valid_from": valid_from, "cached": cached,
                   "expected": expected, "seconds": round(time.monotonic() - started, 2)}


def context_pairs(arms, spec):
    """Per-offset position lists of d(X, C) for X in P, C2, F, S (NaN where a side is missing)."""
    spec, block, unit, b, offsets, end = layout(spec)
    top = int(spec["top_logprobs"])
    out = {}
    base = arms["C"]
    for k, o in enumerate(offsets, 1):
        positions = range(b + o + 1, end)
        rec = {name: [] for name in ("P", "C2", "F", "S", "PF")}
        for pos in positions:
            c = as_dist(base["ids"], base["lps"], pos - base["first"], top)

            def dist(arm):
                a = arms.get(arm)
                if a is None or pos < max(a["first"], a["valid_from"]):
                    return None
                return as_dist(a["ids"], a["lps"], pos - a["first"], top)

            p, c2, f, s = dist(f"P{k}"), dist("C2"), dist("F"), dist("S")
            for key, other in (("P", p), ("C2", c2), ("F", f), ("S", s)):
                value = None if other is None else sym_kl(other, c)
                rec[key].append(np.nan if value is None else value)
            value = None if p is None or f is None else sym_kl(p, f)
            rec["PF"].append(np.nan if value is None else value)
            rec.setdefault("top1_P", []).append(np.nan if p is None else float(max(p, key=p.get) == max(c, key=c.get)))
            rec.setdefault("top1_C2", []).append(np.nan if c2 is None else float(max(c2, key=c2.get) == max(c, key=c.get)))
        out[f"P{k}"] = {key: np.array(v, dtype=np.float64) for key, v in rec.items()}
    return out


def summarize(per_context, spec, seed=0):
    """Pooled ratios with a cluster bootstrap over contexts, per-offset and near/far diagnostics, the verdict."""
    spec = {**DEFAULTS, **spec}
    near = int(spec["near_positions"])
    names = sorted(per_context)
    if not names:
        return {"contexts": 0, "verdict": {"result": "no data"}}

    def sums(ctx, key, offsets=("P1", "P2", "P3"), where=None):
        total = count = 0.0
        for off in offsets:
            rec = per_context[ctx][off]
            both = ~np.isnan(rec[key]) & ~np.isnan(rec["C2"])
            if key == "S":
                both &= ~np.isnan(rec["S"])
            if where is not None:
                idx = np.arange(len(rec[key]))
                both &= where(idx)
            total += float(np.nansum(np.where(both, rec[key], 0.0)))
            count += float(np.nansum(np.where(both, rec["C2"], 0.0)))
        return total, count

    def ratio(key, offsets=("P1", "P2", "P3"), where=None):
        num = np.array([sums(c, key, offsets, where)[0] for c in names])
        den = np.array([sums(c, key, offsets, where)[1] for c in names])
        point = float(num.sum() / den.sum()) if den.sum() > 0 else None
        rng = random.Random(seed)
        boots = []
        for _ in range(int(spec["bootstrap"])):
            pick = [rng.randrange(len(names)) for _ in names]
            d = den[pick].sum()
            if d > 0:
                boots.append(float(num[pick].sum() / d))
        boots.sort()
        lo = boots[int(0.05 * len(boots))] if boots else None
        hi = boots[int(0.95 * len(boots)) - 1] if boots else None
        return {"ratio": point, "lo5": lo, "hi95": hi}

    def pooled(key):
        vals = np.concatenate([per_context[c][o][key] for c in names for o in ("P1", "P2", "P3")])
        vals = vals[~np.isnan(vals)]
        return {"positions": int(vals.size), "mean": float(vals.mean()) if vals.size else None,
                "median": float(np.median(vals)) if vals.size else None,
                "share_over_0.1": float((vals > 0.1).mean()) if vals.size else None}

    result = {"contexts": len(names),
              "sym_kl_vs_C": {k: pooled(k) for k in ("P", "C2", "F", "S")},
              "sym_kl_P_vs_F": pooled("PF"),
              "top1_vs_C": {k: float(np.nanmean(np.concatenate([per_context[c][o][f"top1_{k}"] for c in names
                                                                for o in ("P1", "P2", "P3")])))
                            for k in ("P", "C2")},
              "R_P": ratio("P"), "R_F": ratio("F"), "R_S": ratio("S"),
              "R_P_by_offset": {o: ratio("P", (o,)) for o in ("P1", "P2", "P3")},
              "R_P_near": ratio("P", where=lambda idx: idx < near),
              "R_P_far": ratio("P", where=lambda idx: idx >= near)}
    bar = float(spec["bar"])
    rp, rs = result["R_P"], result["R_S"]
    powered = rs["lo5"] is not None and rs["lo5"] > bar
    if rp["hi95"] is None:
        verdict = "no data"
    elif not powered:
        verdict = "inconclusive (negative control not detected)"
    elif rp["hi95"] <= bar:
        verdict = "pass"
    elif rp["lo5"] > bar:
        verdict = "fail"
    else:
        verdict = "inconclusive"
    result["verdict"] = {"bar": bar, "negative_control_detected": powered, "R_P_hi95": rp["hi95"], "result": verdict}
    return result


async def partial_tf(client, base_url, *, model, sequences, spec, output, log=print, deadline=None):
    """Run the Latin square; save per-context arrays (``partial-tf/*.npz``) and return hits and the summary."""
    spec, block, unit, b, offsets, end = layout(spec)
    contexts = [[int(t) for t in s] for s in sequences if len(s) >= end]
    folder = Path(output) / "partial-tf"
    folder.mkdir(parents=True, exist_ok=True)
    stores = [dict() for _ in contexts]
    errors, phases = [], []
    lanes = asyncio.Semaphore(int(spec["lanes"]))
    for phase in range(len(ARMS)):
        if deadline is not None and deadline() < 90:
            errors.append(f"phase {phase}: skipped, {deadline():.0f} s left")
            break
        started = time.monotonic()
        await isolate_trial(base_url)

        async def one(index):
            name = ARMS[(index + phase) % len(ARMS)]
            async with lanes:
                try:
                    await run_arm(client, base_url, model, contexts[index], name, spec, stores[index])
                except Exception as exc:  # noqa: BLE001 - one bad arm must not lose the square
                    errors.append(f"context {index} {name}: {type(exc).__name__}: {str(exc)[:200]}")

        await asyncio.gather(*(one(i) for i in range(len(contexts))))
        phases.append(round(time.monotonic() - started, 1))
        log(json.dumps({"event": "partial_tf_phase", "phase": phase, "seconds": phases[-1],
                        "errors": len(errors)}), flush=True)
    await isolate_trial(base_url)
    hits = {name: {"as_expected": 0, "requests": 0, "cached": []} for name in ARMS if name not in ("C", "C2", "S")}
    per_context = {}
    for index, store in enumerate(stores):
        for name, arm in store.items():
            if name in hits:
                hits[name]["requests"] += 1
                hits[name]["as_expected"] += arm["cached"] == arm["expected"]
                hits[name]["cached"].append(arm["cached"])
        if not {"C", "C2"} <= set(store):
            continue
        pairs = await asyncio.to_thread(context_pairs, store, spec)
        per_context[index] = pairs
        np.savez_compressed(folder / f"context-{index:02d}.npz",
                            **{f"{off}_{key}": arr for off, rec in pairs.items() for key, arr in rec.items()})
    summary = await asyncio.to_thread(summarize, per_context, spec)
    hits_exact = all(h["requests"] and h["as_expected"] == h["requests"] for h in hits.values())
    summary["verdict"]["hits_exact"] = hits_exact
    if not hits_exact and summary["verdict"]["result"] == "pass":
        summary["verdict"]["result"] = "fail (hits not exact)"
    return {"spec": spec, "block_start": b, "end": end, "contexts": len(contexts), "phase_seconds": phases,
            "errors": errors, "hits": hits, "seconds": {n: [s[n]["seconds"] for s in stores if n in s] for n in ARMS},
            "summary": summary}
