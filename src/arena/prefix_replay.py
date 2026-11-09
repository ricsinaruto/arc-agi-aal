"""Prefix-cache smoke checks for the serving bench (29 September; docs/validation-20260927/prefix-cache-v2.md).

``prefix_check`` (correctness, one request at a time): a long transcript prompt P is prefilled cold, so its Mamba
states are written at its prefill chunk ends; then prompts Q_c that share P's first ~c tokens and continue
differently are answered greedily twice, warm (resuming from P's cached state at the largest cached boundary
<= c) and cold (after a prefix-cache reset). On the pinned split a warm hit lands on a block that holds a
chunk-end state under a later boundary's hash, so warm and cold differ; with the align fix they agree (up to a
warm/warm repeat's noise, which is measured too). The warm hit length is read from ``prompt_tokens_details``.

``prefix_replay`` (hit accounting under pool pressure): G games play an append-only chat through the served chat
template at C concurrent requests (the opening's gap structure: each game waits for the others), with the live
reply (reasoning, content, tool calls) appended as the harness does and a recorded transcript slice as each tool
result. ``return_token_ids`` gives every prompt's and reply's token ids, so each request's ideal hit (the longest
common prefix with the game's previous prompt + reply, floored to the Mamba block), the round-trip identity of the
previous reply (do its generated tokens reappear verbatim?) and the served hit are exact.
"""

import asyncio
import json
import time

from arc_infra.model_templates import template_kwargs

from .isolation import isolate_trial

PYTHON_TOOL = [{"type": "function", "function": {
    "name": "python", "description": "Run Python in the persistent game kernel and return its output.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
SYSTEM = ("You are a coding agent solving a grid-based puzzle game. Each turn: re-read the newest observation, "
          "update your working world model in Python, choose the next action, execute it with the python tool and "
          "expect to re-evaluate on the next turn.\n\nReference notes from earlier games:\n")


def common_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def slice_text(text, start, chars):
    start = start % max(1, len(text) - chars)
    cut = text.rfind("\n", 0, start)
    return text[cut + 1:cut + 1 + chars] if cut >= 0 else text[start:start + chars]


async def chat(client, base_url, body):
    started = time.monotonic()
    response = await client.post(base_url + "/chat/completions", json=body)
    if response.status_code != 200:
        return {"error": f"{response.status_code}: {response.text[:300]}", "seconds": time.monotonic() - started}
    data = response.json()
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    top = [{t["token"]: t["logprob"] for t in (entry.get("top_logprobs") or [])}
           for entry in ((choice.get("logprobs") or {}).get("content") or [])]
    return {"message": choice["message"], "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": details.get("cached_tokens"), "prompt_token_ids": data.get("prompt_token_ids"),
            "token_ids": choice.get("token_ids"), "seconds": time.monotonic() - started, "top_logprobs": top}


def top_distance(a, b):
    """Max |log p| difference over the tokens in both first-position top-k lists (inf if they share none)."""
    if not a or not b:
        return None
    shared = set(a[0]) & set(b[0])
    return max((abs(a[0][t] - b[0][t]) for t in shared), default=float("inf"))


async def prefix_check(client, base_url, *, model, transcript, spec, log=print):
    """Warm vs cold greedy answers on prompts that resume from a long prompt's cached states.

    Cold answers are recomputed from token 0 after a prefix-cache reset, twice (their agreement is the determinism
    floor). A correct warm state agrees with cold up to chunking round-off (long common token runs, tiny first-token
    log-prob differences); a state short of its prefix diverges early.
    """
    kwargs = template_kwargs(model, {"enable_thinking": False})
    chars = int(spec.get("prompt_chars", 48000))
    base = slice_text(transcript, int(spec.get("offset_chars", 0)), chars)
    question = spec.get("question", "\n\nIn one short sentence: what did the last action change on the board?")
    out = int(spec.get("output_tokens", 24))

    def body(text):
        return {"model": model, "temperature": 0.0, "seed": 0, "max_tokens": out, "logprobs": True,
                "top_logprobs": 5, "return_token_ids": True, "chat_template_kwargs": kwargs,
                "messages": [{"role": "user", "content": text + question}]}

    await isolate_trial(base_url)
    primer = await chat(client, base_url, {**body(base), "max_tokens": 1})
    results = {"primer": {k: primer.get(k) for k in ("prompt_tokens", "cached_tokens", "error")}, "cuts": []}
    for fraction in spec.get("cut_fractions", [0.2, 0.33, 0.66, 0.9]):
        text = base[:int(len(base) * fraction)]
        warm = await chat(client, base_url, body(text))
        await isolate_trial(base_url)
        cold = await chat(client, base_url, body(text))
        await isolate_trial(base_url)
        cold2 = await chat(client, base_url, body(text))
        # Re-prime so the next cut resumes from P's states again.
        await isolate_trial(base_url)
        await chat(client, base_url, {**body(base), "max_tokens": 1})
        record = {"fraction": fraction, "prompt_tokens": cold.get("prompt_tokens"),
                  "warm_cached_tokens": warm.get("cached_tokens"), "cold_cached_tokens": cold.get("cached_tokens"),
                  "errors": [r["error"] for r in (warm, cold, cold2) if r.get("error")]}
        if not record["errors"]:
            wt, ct, c2 = warm["token_ids"] or [], cold["token_ids"] or [], cold2["token_ids"] or []
            record.update(warm_matches_cold=wt == ct, cold_matches_cold=ct == c2,
                          warm_cold_common_tokens=common_prefix(wt, ct), cold_cold_common_tokens=common_prefix(ct, c2),
                          warm_cold_first_logprob_distance=top_distance(warm["top_logprobs"], cold["top_logprobs"]),
                          cold_cold_first_logprob_distance=top_distance(cold["top_logprobs"], cold2["top_logprobs"]),
                          warm_text=(warm["message"].get("content") or "")[:200],
                          cold_text=(cold["message"].get("content") or "")[:200])
        log(json.dumps({"event": "prefix_check", **{k: v for k, v in record.items() if "text" not in k}}), flush=True)
        results["cuts"].append(record)
    cuts = [c for c in results["cuts"] if not c["errors"]]
    results["summary"] = {"cuts": len(results["cuts"]), "clean": len(cuts),
                          "warm_equals_cold": sum(c["warm_matches_cold"] for c in cuts),
                          "cold_equals_cold": sum(c["cold_matches_cold"] for c in cuts),
                          "warm_hits": [c["warm_cached_tokens"] for c in cuts]}
    return results


def classify(record, previous, block, unit=None):
    """Exact ideal / served hit and the previous reply's round trip for one replay request.

    ``unit`` (``--prefix-match-unit``, hero-v2 precheck): the server also keeps the previous prompt's last
    ``unit``-token boundary (its partial tail), so ``ideal_hit_unit`` is the better of that and the block ideal."""
    ids = record.get("prompt_token_ids") or []
    if previous is None or not ids:
        return record
    before = (previous.get("prompt_token_ids") or []) + (previous.get("token_ids") or [])
    lcp = common_prefix(before, ids)
    reply_start = len(previous.get("prompt_token_ids") or [])
    record["ideal_hit"] = min(lcp, len(ids) - 1) // block * block
    if unit:
        tail = reply_start // unit * unit
        record["ideal_hit_unit"] = max(record["ideal_hit"], tail if tail <= min(lcp, len(ids) - 1) else 0)
    end = len(before)
    if lcp >= end - 1:
        # request_end_tail (prefix cache v3): the previous request's exact end is cached (``end`` or ``end - 1``:
        # an in-flight step computes the stop token or not), when this prompt repeats the whole previous request.
        record["ideal_hit_tail"] = max(record["ideal_hit"], min(end, lcp, len(ids) - 1))
        record["tail_end"] = end
    record["reply_tokens"] = len(previous.get("token_ids") or [])
    record["reply_tokens_reused"] = max(0, lcp - reply_start)
    record["reply_round_trip_exact"] = lcp >= len(before)
    record["previous_prompt_end"] = min(reply_start, len(ids) - 1) // block * block
    if not record["reply_round_trip_exact"]:
        # Where the next prompt's rendering of the previous reply first differs from the generated tokens.
        record["divergence"] = {"offset_in_reply": lcp - reply_start, "generated": before[max(0, lcp - 6):lcp + 10],
                                "rendered": ids[max(0, lcp - 6):lcp + 10]}
    return record


async def detokenize(client, base_url, model, tokens):
    response = await client.post(base_url.removesuffix("/v1") + "/detokenize", json={"model": model, "tokens": tokens})
    return response.json().get("prompt") if response.status_code == 200 else None


async def prefix_replay(client, base_url, *, model, effort, transcript, spec, block, log=print):
    games, rounds = int(spec.get("games", 24)), int(spec.get("requests", 6))
    concurrency = int(spec.get("concurrency", 8))
    kwargs = template_kwargs(model, {"enable_thinking": True, "reasoning_effort": effort})
    notes = slice_text(transcript, 7, int(spec.get("system_chars", 12000)))
    semaphore = asyncio.Semaphore(concurrency)
    records = []
    await isolate_trial(base_url)
    started = time.monotonic()

    async def game(g):
        messages = [{"role": "system", "content": SYSTEM + notes},
                    {"role": "user", "content": "Observation:\n" + slice_text(transcript, 9973 * (g + 1),
                                                                             int(spec.get("first_chars", 6000)))}]
        previous = None
        for n in range(rounds):
            body = {"model": model, "temperature": float(spec.get("temperature", 0.7)), "top_p": 0.95, "top_k": 20,
                    "seed": 1000 * g + n, "max_tokens": int(spec.get("max_tokens", 1200)), "return_token_ids": True,
                    "tools": PYTHON_TOOL, "tool_choice": "auto", "chat_template_kwargs": kwargs, "messages": messages}
            async with semaphore:
                reply = await chat(client, base_url, body)
            record = {"game": g, "request": n, **{k: reply.get(k) for k in (
                "prompt_tokens", "completion_tokens", "cached_tokens", "finish_reason", "seconds", "error")},
                "end": time.monotonic() - started}
            if reply.get("error"):
                records.append(record)
                return
            record = classify({**record, "prompt_token_ids": reply["prompt_token_ids"],
                               "token_ids": reply["token_ids"]}, previous, block, spec.get("unit"))
            previous = {"prompt_token_ids": reply["prompt_token_ids"], "token_ids": reply["token_ids"]}
            records.append({k: v for k, v in record.items() if k not in ("prompt_token_ids", "token_ids")})
            message = reply["message"]
            assistant = {"role": "assistant", "content": message.get("content")}
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                assistant["reasoning"] = reasoning
            calls = message.get("tool_calls") or []
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            observation = slice_text(transcript, 7919 * (g + 1) + 104729 * (n + 1), int(spec.get("tool_chars", 1200)))
            if calls:
                for call in calls:
                    messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": observation})
            else:
                messages.append({"role": "user", "content": "Observation:\n" + observation})

    await asyncio.gather(*(game(g) for g in range(games)))
    # Text of the first round-trip divergences (what the chat template rendered differently).
    shown = 0
    for record in records:
        div = record.get("divergence")
        if not div or shown >= int(spec.get("divergence_samples", 16)):
            continue
        shown += 1
        try:
            div["generated_text"] = await detokenize(client, base_url, model, div["generated"])
            div["rendered_text"] = await detokenize(client, base_url, model, div["rendered"])
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            div["detokenize_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return {"records": sorted(records, key=lambda r: (r["game"], r["request"])),
            "summary": summarize_replay(records, block, spec.get("unit")),
            "seconds": round(time.monotonic() - started, 1)}


def summarize_unit(served, unit, block):
    """Hits against the partial-tail ideal (``--prefix-match-unit``): extra tokens above the block ideal, hits that
    land inside a block, and the granularity of every served hit."""
    return {
        "unit": unit,
        "ideal_hit_unit_tokens": sum(r["ideal_hit_unit"] for r in served),
        "extra_over_block_ideal_tokens": sum(max(0, r["cached_tokens"] - r["ideal_hit"]) for r in served),
        "ideal_extra_over_block_tokens": sum(r["ideal_hit_unit"] - r["ideal_hit"] for r in served),
        "shortfall_vs_unit_ideal_tokens": sum(max(0, r["ideal_hit_unit"] - r["cached_tokens"]) for r in served),
        "at_unit_ideal": sum(r["cached_tokens"] >= r["ideal_hit_unit"] for r in served),
        "hits_inside_a_block": sum(bool(r["cached_tokens"] % block) for r in served),
        "hits_off_unit_grid": sum(bool(r["cached_tokens"] % unit) for r in served),
        "continuations_with_partial_ideal": sum(r["ideal_hit_unit"] > r["ideal_hit"] for r in served),
    }


def summarize_tail(served):
    """Hits against the request-end ideal (``request_end_tail``): a hit at the previous request's end or its last
    token (the stop token computed or not), extra tokens above the block ideal and the remaining shortfall."""
    tail = [r for r in served if "ideal_hit_tail" in r]
    at_end = [r for r in tail if r["cached_tokens"] in (r["tail_end"] - 1, r["tail_end"])
              and r["cached_tokens"] > r["ideal_hit"]]
    return {
        "continuations_repeating_previous_request": len(tail),
        "hits_at_request_end": len(at_end),
        "hits_at_request_end_including_stop_token": sum(r["cached_tokens"] == r["tail_end"] for r in at_end),
        "ideal_extra_over_block_tokens": sum(r["ideal_hit_tail"] - r["ideal_hit"] for r in tail),
        "extra_over_block_ideal_tokens": sum(max(0, r["cached_tokens"] - r["ideal_hit"]) for r in tail),
        "shortfall_vs_tail_ideal_tokens": sum(max(0, r["ideal_hit_tail"] - 1 - r["cached_tokens"]) for r in tail),
        "hits_above_tail_ideal": sum(r["cached_tokens"] > r["ideal_hit_tail"] for r in tail),
    }


def summarize_replay(records, block, unit=None):
    ok = [r for r in records if not r.get("error")]
    cont = [r for r in ok if "ideal_hit" in r]
    served = [r for r in cont if r.get("cached_tokens") is not None]
    labels = {"ideal": 0, "previous_prompt_end": 0, "shortfall_other": 0}
    for r in served:
        if r["cached_tokens"] >= r["ideal_hit"]:
            labels["ideal"] += 1
        elif r["cached_tokens"] == r["previous_prompt_end"]:
            labels["previous_prompt_end"] += 1
        else:
            labels["shortfall_other"] += 1
    return {
        "requests": len(records), "errors": len(records) - len(ok), "continuations": len(cont),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in ok),
        "cached_tokens": sum(r.get("cached_tokens") or 0 for r in ok),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        "ideal_hit_tokens": sum(r["ideal_hit"] for r in cont),
        "served_hit_tokens_on_continuations": sum(r["cached_tokens"] for r in served),
        "shortfall_tokens": sum(max(0, r["ideal_hit"] - r["cached_tokens"]) for r in served),
        "hit_vs_ideal": labels,
        "reply_round_trip_exact": sum(bool(r.get("reply_round_trip_exact")) for r in cont),
        "reply_tokens": sum(r.get("reply_tokens") or 0 for r in cont),
        "reply_tokens_reused": sum(r.get("reply_tokens_reused") or 0 for r in cont),
        "block": block,
        **({"unit_hits": summarize_unit(served, unit, block)} if unit else {}),
        "tail_hits": summarize_tail(served),
    }


# ----------------------------------------------------------------------------- request-end tail correctness (v3)
def tail_conditions(block):
    """name -> (primer length, next-token bias, reset before every probe). A +100 bias forces the primer's one
    sampled token to be the recorded next token, so its request-end entry (at the primer's length, or one more when an
    in-flight step computes the sampled token) lies on the recorded sequence; -100 forces a token off it, so the
    full-block conditions cannot leave a usable entry. Tail primers end at residues 1, 2 and 3 mod 4 (inside a QSA
    compression group, so the entry's compressor ring is copied too)."""
    b6 = 6 * block
    return {
        "F6": (b6, -100, False),                       # the pinned full-block state at 6 blocks
        "F5": (5 * block, -100, True),                 # the full-block state at 5 blocks (the floor pair)
        "T1": (b6 + block // 2 - (block // 2) % 4 + 1, 100, False),
        "T2": (b6 + block // 4 - (block // 4) % 4 + 2, 100, False),
        "T3": (b6 + 3 * block // 8 - (3 * block // 8) % 4 + 3, 100, False),
    }


async def tail_check(client, base_url, *, model, sequences, spec, log=print):
    """Teacher-forced warm/cold check of request-end tail entries (``request_end_tail``) on recorded token sequences.

    Per sequence: the cold distribution at every probe position (``prompt_logprobs``, which skips the prefix cache),
    a sensitivity control S (the tokens between the 6-block boundary and the T1 entry deleted), then each condition
    from a reset cache: its primer (one request, one forced token), then single-token probes that extend it, longest
    first (a probe's own entry lies beyond every later probe), answered warm with their exact ``cached_tokens``.
    A tail probe must hit exactly at its primer's end (length or length + 1) and agree with the full-block path as
    closely as two full-block paths agree with each other."""
    from .hero_v2_precheck import completion, probe_lengths, top_dist
    from .kernel_gate import compare_positions

    block = int(spec["block"])
    lengths = probe_lengths(block, int(spec.get("probes", 24)))
    end = 7 * block
    top = int(spec.get("top_logprobs", 20))
    conditions = tail_conditions(block)
    results = {"block": block, "probe_lengths": lengths, "conditions": {k: {"primer": v[0], "bias": v[1]}
                                                                        for k, v in conditions.items()},
               "sequences": []}

    def body(ids, max_tokens=1, logprobs=top, bias=None):
        out = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
               "return_tokens_as_token_ids": True}
        if logprobs is not None:
            out["logprobs"] = logprobs
        if bias is not None:
            out["logit_bias"] = bias
        return out

    async def teacher(ids):
        reply = await completion(client, base_url, {"model": model, "prompt": ids, "max_tokens": 1,
                                                    "temperature": 0.0, "prompt_logprobs": top})
        entries = reply["choices"][0].get("prompt_logprobs") or []
        if len(entries) != len(ids):
            raise RuntimeError(f"prompt_logprobs length {len(entries)} != {len(ids)}")
        return entries

    async def warm(ids, length):
        reply = await completion(client, base_url, body(ids[:length]))
        choice = reply["choices"][0]
        details = (reply.get("usage") or {}).get("prompt_tokens_details") or {}
        tops = (choice.get("logprobs") or {}).get("top_logprobs") or []
        return {"length": length, "cached_tokens": details.get("cached_tokens"),
                "dist": top_dist(tops[0] if tops else {})}

    for index, ids in enumerate(sequences):
        ids = [int(t) for t in ids]
        if len(ids) < end + 1:
            raise ValueError(f"tail_check sequence {index}: {len(ids)} tokens < {end + 1}")
        row = {"index": index, "conditions": {}}
        started = time.monotonic()
        await isolate_trial(base_url)
        cold = await teacher(ids[:end])
        row["conditions"]["C"] = {"dists": [top_dist(cold[n]) for n in lengths]}
        cut = conditions["T1"][0] - 6 * block
        await isolate_trial(base_url)
        sens = await teacher(ids[:6 * block] + ids[6 * block + cut:end])
        row["conditions"]["S"] = {"dists": [top_dist(sens[n - cut]) for n in lengths],
                                  "deleted": [6 * block, 6 * block + cut]}
        for name, (primer, bias, reset_each) in conditions.items():
            await isolate_trial(base_url)
            probes = []
            for k, length in enumerate(lengths):
                if reset_each and k:
                    await isolate_trial(base_url)
                if reset_each or not k:
                    await completion(client, base_url, body(ids[:primer], logprobs=None,
                                                            bias={str(ids[primer]): bias}))
                probes.append(await warm(ids, length))
            hits = [p["cached_tokens"] for p in probes]
            expected = [6 * block] if name == "F6" else [5 * block] if name == "F5" else [primer, primer + 1]
            row["conditions"][name] = {"primer": primer, "expected_hits": expected, "cached_tokens": hits,
                                       "hit_as_expected": sum(h in expected for h in hits),
                                       "hit_values": sorted({h for h in hits if h is not None}),
                                       "dists": [p["dist"] for p in probes]}
        chosen = [ids[n] for n in lengths]
        row["pairs"] = {}
        for a, b in (("F6", "F5"), ("F6", "T1"), ("F6", "T2"), ("F6", "T3"), ("T1", "T2"), ("C", "F6"),
                     ("C", "F5"), ("C", "T1"), ("C", "T2"), ("C", "T3"), ("C", "S")):
            row["pairs"][f"{a}|{b}"] = compare_positions(list(zip(row["conditions"][a]["dists"],
                                                                  row["conditions"][b]["dists"], chosen)))
        row["seconds"] = round(time.monotonic() - started, 1)
        log(json.dumps({"event": "tail_check", "index": index, "seconds": row["seconds"],
                        "hits": {k: v.get("hit_as_expected") for k, v in row["conditions"].items()},
                        "mean_kl": {k: v["mean_kl"] for k, v in row["pairs"].items()}}), flush=True)
        results["sequences"].append(row)
    await isolate_trial(base_url)
    results["summary"] = summarize_tail_check(results)
    return results


def summarize_tail_check(results):
    """Pooled pair statistics and the pass rule (stated before the run, the pre-check's partial-hit rule): every tail
    probe hits at its primer's end, every full-block probe at its block, and each tail path's distance to the
    6-block path is within 1.5x the floor pair's (+0.005 nats) with top-1 agreement at most 5 points below it."""
    pooled = {}
    for row in results["sequences"]:
        for pair, stats in row["pairs"].items():
            pooled.setdefault(pair, []).append(stats)

    def pool(items):
        total = sum(s["positions"] for s in items)
        mean = lambda key: (sum((s[key] or 0) * s["positions"] for s in items) / total) if total else None  # noqa: E731
        return {"positions": total, "mean_kl": mean("mean_kl"), "top1_agreement": mean("top1_agreement"),
                "mean_abs_delta_logprob": mean("mean_abs_delta_logprob")}

    pairs = {pair: pool(items) for pair, items in pooled.items()}
    hits = {}
    for row in results["sequences"]:
        for name, cond in row["conditions"].items():
            if "hit_as_expected" in cond:
                entry = hits.setdefault(name, {"as_expected": 0, "probes": 0, "values": set()})
                entry["as_expected"] += cond["hit_as_expected"]
                entry["probes"] += len(cond["cached_tokens"])
                entry["values"] |= set(cond["hit_values"])
    hits = {k: {**v, "values": sorted(v["values"])} for k, v in hits.items()}
    floor = pairs.get("F6|F5", {})
    verdict = None
    if floor.get("mean_kl") is not None:
        verdict = {"hits_exact": all(h["as_expected"] == h["probes"] for h in hits.values())}
        for name in ("T1", "T2", "T3"):
            test = pairs.get(f"F6|{name}", {})
            if test.get("mean_kl") is None:
                verdict[f"{name}_kl_within"] = verdict[f"{name}_top1_within"] = False
                continue
            verdict[f"{name}_kl_within"] = test["mean_kl"] <= 1.5 * floor["mean_kl"] + 0.005
            verdict[f"{name}_top1_within"] = test["top1_agreement"] >= floor["top1_agreement"] - 0.05
        sens, ref = pairs.get("C|S", {}), pairs.get("C|F6", {})
        # Report-only: the control must be clearly worse than a correct path, or the check could not see an error.
        verdict["sensitivity_visible"] = (sens.get("mean_kl") is not None and ref.get("mean_kl") is not None
                                          and sens["mean_kl"] > 2 * ref["mean_kl"] + 0.01)
        verdict["pass"] = all(v for k, v in verdict.items() if k != "sensitivity_visible")
    return {"pairs": pairs, "hits": hits, "verdict": verdict}


async def mtp_tail_check(client, base_url, *, model, sequences, spec, log=print, remaining=None):
    """Teacher-forced warm/cold check of request-end tail entries under MTP-k (throughput arm, 4 October).

    The v3 ``tail_check`` primers finish after a prefill; arena requests finish after a verify step, with an in-flight
    step at the stop token. Per recorded sequence ``ids`` and primer length ``P``:

    - R1 (cache reset): ``ids[:P]`` decoded greedily for up to ``max_tokens``; the stop token is the first generated
      token at index >= ``stop_min`` that does not occur earlier in the reply;
    - D: reset, then R2 = R1 with ``stop_token_ids`` = [that token], so it finishes on a stop token like an arena
      request; its tail entry should sit at ``L = P + len(reply)``; probes ``S[:n]`` for n > L (S = ids[:P] + reply +
      ids[P:]) are answered warm, longest first, and must hit at L;
    - T: reset, a prefill-only primer ``S[:L]`` (max_tokens 1): entry at L (or L + 1 after an in-flight step);
    - F6 / F5: the pinned full-block paths (primers ``S[:6B]`` / ``S[:5B]`` plus one token that is not on the
      sequence, so their own tails never match; F5 resets before every probe): the floor pair;
    - C: the cold distributions (``prompt_logprobs``, no cache) at the probe positions; X: the cold distributions with
      the reply's last token deleted (what a state one token short would look like).
    Pass rule (the v3 rule): D and T hit as expected at every probe, and KL(F6, D) and KL(F6, T) are within
    1.5 x KL(F6, F5) + 0.005 nats with top-1 agreement at most 5 points below the floor pair's."""
    from .hero_v2_precheck import completion, top_dist
    from .kernel_gate import compare_positions

    block = int(spec["block"])
    top = int(spec.get("top_logprobs", 20))
    offsets = [int(o) for o in spec.get("probe_offsets", (1, 3, 7, 16, 31, 64, 100, 150, 220, 300, 420, 600))]
    results = {"block": block, "offsets": offsets, "sequences": []}

    def body(ids, max_tokens=1, logprobs=top, bias=None, **extra):
        out = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
               "return_tokens_as_token_ids": True, **extra}
        if logprobs is not None:
            out["logprobs"] = logprobs
        if bias is not None:
            out["logit_bias"] = bias
        return out

    async def teacher(ids):
        reply = await completion(client, base_url, {"model": model, "prompt": ids, "max_tokens": 1,
                                                    "temperature": 0.0, "prompt_logprobs": top})
        entries = reply["choices"][0].get("prompt_logprobs") or []
        if len(entries) != len(ids):
            raise RuntimeError(f"prompt_logprobs length {len(entries)} != {len(ids)}")
        return entries

    async def warm(ids, length):
        reply = await completion(client, base_url, body(ids[:length]))
        choice = reply["choices"][0]
        details = (reply.get("usage") or {}).get("prompt_tokens_details") or {}
        tops = (choice.get("logprobs") or {}).get("top_logprobs") or []
        return {"length": length, "cached_tokens": details.get("cached_tokens"),
                "dist": top_dist(tops[0] if tops else {})}

    async def probes(seq, lengths, primer=None, reset_each=False):
        await isolate_trial(base_url)
        out = []
        for k, n in enumerate(lengths):
            if reset_each and k:
                await isolate_trial(base_url)
            if primer is not None and (reset_each or not k):
                await primer()
            out.append(await warm(seq, n))
        return out

    for index, ids in enumerate(sequences):
        left = remaining() if remaining is not None else None
        if left is not None and left < float(spec.get("reserve_seconds", 300)):
            results["stopped"] = f"{left:.0f} s left before sequence {index}"
            break
        ids = [int(t) for t in ids]
        P = 6 * block + int(spec.get("primer_offset", 1500)) + (37 * index) % 200
        row = {"index": index, "primer": P, "conditions": {}}
        started = time.monotonic()
        await isolate_trial(base_url)
        r1 = await completion(client, base_url, body(ids[:P], max_tokens=int(spec.get("max_tokens", 96)),
                                                     logprobs=None, return_token_ids=True))
        g1 = r1["choices"][0].get("token_ids") or []
        stop_min = int(spec.get("stop_min", 24))
        j = next((i for i in range(stop_min, len(g1)) if g1[i] not in g1[:i]), None)
        if j is None:
            row["skipped"] = "no unique stop token in the greedy reply"
            results["sequences"].append(row)
            continue
        await isolate_trial(base_url)
        r2 = await completion(client, base_url, body(ids[:P], max_tokens=int(spec.get("max_tokens", 96)),
                                                     logprobs=None, return_token_ids=True,
                                                     stop_token_ids=[int(g1[j])]))
        choice = r2["choices"][0]
        reply = [int(t) for t in (choice.get("token_ids") or [])]
        row["reply_tokens"], row["finish_reason"] = len(reply), choice.get("finish_reason")
        row["reply_matches_r1"] = reply == [int(t) for t in g1[:j + 1]]
        L = P + len(reply)
        seq = ids[:P] + reply + ids[P:]
        if spec.get("continue_greedy") and reply == [int(t) for t in g1[:len(reply)]]:
            # The greedy reply continues past the stop token (R1): an entry at L + 1 holding the dropped bonus token
            # (tail_spec_acceptance) also matches the probes.
            seq = ids[:P] + [int(t) for t in g1] + ids[P:]
        lengths = sorted({L + o for o in offsets if L + o < len(seq)}, reverse=True)
        row["L"], row["probe_lengths"] = L, lengths
        # D: the decode-ended entry (R2 must be the last request before the probes: no reset between).
        d_probes = []
        for n in lengths:
            d_probes.append(await warm(seq, n))
        row["conditions"]["D"] = {"expected_hits": [L, L + 1] if spec.get("continue_greedy") else [L],
                                  "probes": d_probes}

        # No logit_bias under speculative decoding (the server refuses it): the T primer is S[:L] with one free token
        # (a prefill-only finish registers at L; an in-flight one at L + 1, served only if its token is S[L]); the F
        # primers end one token past the boundary with a token that is not S[n], so their own tails never match.
        async def t_primer():
            await completion(client, base_url, body(seq[:L], logprobs=None))

        async def f_primer(n):
            await completion(client, base_url, body(seq[:n] + [seq[n] + 1 if seq[n] + 1 < 150000 else seq[n] - 1],
                                                    logprobs=None))

        row["conditions"]["T"] = {"expected_hits": [L, L + 1], "probes": await probes(seq, lengths, t_primer)}
        row["conditions"]["F6"] = {"expected_hits": [6 * block],
                                   "probes": await probes(seq, lengths, lambda: f_primer(6 * block))}
        row["conditions"]["F5"] = {"expected_hits": [5 * block],
                                   "probes": await probes(seq, lengths, lambda: f_primer(5 * block), True)}
        await isolate_trial(base_url)
        cold = await teacher(seq[:max(lengths) + 1])
        row["conditions"]["C"] = {"dists": [top_dist(cold[n]) for n in lengths]}
        await isolate_trial(base_url)
        short = await teacher(seq[:L - 1] + seq[L:max(lengths) + 1])
        row["conditions"]["X"] = {"dists": [top_dist(short[n - 1]) for n in lengths]}
        for cond in row["conditions"].values():
            if "probes" in cond:
                hits = [p["cached_tokens"] for p in cond["probes"]]
                cond.update(cached_tokens=hits, hit_as_expected=sum(h in cond["expected_hits"] for h in hits),
                            dists=[p["dist"] for p in cond.pop("probes")])
        chosen = [seq[n] for n in lengths]
        row["pairs"] = {}
        for a, b in (("F6", "F5"), ("F6", "D"), ("F6", "T"), ("D", "T"), ("C", "F6"), ("C", "D"), ("C", "T"),
                     ("C", "X"), ("F6", "X")):
            row["pairs"][f"{a}|{b}"] = compare_positions(list(zip(row["conditions"][a]["dists"],
                                                                  row["conditions"][b]["dists"], chosen)))
        row["seconds"] = round(time.monotonic() - started, 1)
        log(json.dumps({"event": "mtp_tail_check", "index": index, "seconds": row["seconds"], "L": L,
                        "finish": row["finish_reason"],
                        "hits": {k: v.get("hit_as_expected") for k, v in row["conditions"].items()},
                        "mean_kl": {k: round(v["mean_kl"] or 0, 4) for k, v in row["pairs"].items()}}), flush=True)
        results["sequences"].append(row)
    await isolate_trial(base_url)
    results["summary"] = summarize_mtp_tail_check(results)
    return results


def summarize_mtp_tail_check(results):
    pooled, hits = {}, {}
    for row in results["sequences"]:
        for pair, stats in (row.get("pairs") or {}).items():
            pooled.setdefault(pair, []).append(stats)
        for name, cond in (row.get("conditions") or {}).items():
            if "hit_as_expected" in cond:
                entry = hits.setdefault(name, {"as_expected": 0, "probes": 0})
                entry["as_expected"] += cond["hit_as_expected"]
                entry["probes"] += len(cond["cached_tokens"])

    def pool(items):
        total = sum(s["positions"] for s in items)
        mean = lambda key: (sum((s[key] or 0) * s["positions"] for s in items) / total) if total else None  # noqa: E731
        return {"positions": total, "mean_kl": mean("mean_kl"), "top1_agreement": mean("top1_agreement"),
                "mean_abs_delta_logprob": mean("mean_abs_delta_logprob")}

    pairs = {pair: pool(items) for pair, items in pooled.items()}
    floor = pairs.get("F6|F5") or {}
    verdict = {}
    for name in ("D", "T"):
        pair = pairs.get(f"F6|{name}") or {}
        if not floor or not pair:
            continue
        verdict[name] = {
            "hits": hits.get(name, {}).get("as_expected") == hits.get(name, {}).get("probes"),
            "kl_ok": pair["mean_kl"] <= 1.5 * floor["mean_kl"] + 0.005,
            "top1_ok": pair["top1_agreement"] >= floor["top1_agreement"] - 0.05}
        verdict[name]["pass"] = all(verdict[name].values())
    finished = [row.get("finish_reason") for row in results["sequences"] if "finish_reason" in row]
    return {"pairs": pairs, "hits": hits, "verdict": verdict, "stop_finishes": finished.count("stop"),
            "sequences": len(results["sequences"]),
            "pass": bool(verdict) and all(v["pass"] for v in verdict.values())}
