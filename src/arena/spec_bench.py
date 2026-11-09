"""Speculative-decoding pilot bench (30 September; docs/validation-20260930/spec-decode.md).

One server (MTP-0, MTP-1 or MTP-2 on the blocks3 campaign serving) and recorded arena contexts
(``scripts/spec_decode_inputs.py``: real Duck requests of the three-block run, rendered with the served template and
tokenized; sent to ``/v1/completions`` as token ids, so every arm decodes exactly the same prompts):

1. ``prefill``: every context once (``max_tokens`` 1) through the arena's KV gate, so the decode points start from
   cached prefixes.
2. ``decode`` points at each stream count (the n contexts closest to the set's length distribution, spread order):
   forced ``output_tokens`` at the arena's sampling (temperature 0.7, top-p 0.95, top-k 20); once every stream runs
   with nothing waiting and no prompt tokens computed, a ``window_seconds`` window of engine counters gives ms per
   engine step, generated tokens per step, aggregate / per-stream tok/s and the speculative counters (drafts, draft
   tokens, accepted tokens, accepted per position). The requests are then aborted (client disconnect).
3. ``greedy``: the first ``prompts`` contexts at temperature 0 for ``output_tokens`` (``return_token_ids``), once all
   concurrently and once one at a time. The token ids are compared offline across arms: MTP-k against MTP-0 must
   agree about as far as MTP-0 agrees with itself under a different batch shape (rejection sampling is exact; only
   batch-shape numerics may differ).
"""

import asyncio
import json
import time

from .hero_v2_precheck import Sampler, gpu_memory, spread_indices, wait_for_idle

COUNTERS = {
    "steps": "vllm:iteration_tokens_total_count",
    "scheduled_tokens": "vllm:iteration_tokens_total_sum",
    "generated": "vllm:generation_tokens_total",
    "uncached": "vllm:prompt_tokens_by_source_total:local_compute",
    "preemptions": "vllm:num_preemptions_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}
POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total:"


def window(first, last, streams):
    seconds = last[0] - first[0]
    d = {k: last[1].get(n, 0.0) - first[1].get(n, 0.0) for k, n in COUNTERS.items()}
    if seconds <= 0 or d["steps"] <= 0:
        return {"valid": False, "seconds": round(seconds, 3), "steps": d["steps"]}
    out = {"valid": True, "streams": streams, "seconds": round(seconds, 3), "steps": int(d["steps"]),
           "step_ms": round(1000 * seconds / d["steps"], 3),
           "generated_per_step": round(d["generated"] / d["steps"], 3),
           "scheduled_tokens_per_step": round(d["scheduled_tokens"] / d["steps"], 2),
           "uncached_per_step": round(d["uncached"] / d["steps"], 2),
           "generated_tok_s": round(d["generated"] / seconds, 2),
           "per_stream_tok_s": round(d["generated"] / seconds / streams, 3),
           "preemptions": d["preemptions"]}
    if d["drafts"]:
        out.update(drafts=int(d["drafts"]), draft_tokens=int(d["draft_tokens"]), accepted=int(d["accepted"]),
                   accepted_per_draft=round(d["accepted"] / d["drafts"], 4),
                   acceptance_rate=round(d["accepted"] / d["draft_tokens"], 4),
                   per_position=[round((last[1].get(POSITION + str(i), 0.0) - first[1].get(POSITION + str(i), 0.0))
                                       / d["drafts"], 4) for i in range(int(round(d["draft_tokens"] / d["drafts"])))])
    return out


async def spec_sweep(bench, client, gate, budget, monitor, spec, *, save=lambda: None):
    path = bench.source / spec["contexts"] if bench.source else spec["contexts"]
    contexts = json.loads(open(path).read())["contexts"]
    # Length order, then the spread order: every first-n subset spans the length range.
    contexts = sorted(contexts, key=lambda c: c["tokens"])
    contexts = [contexts[i] for i in spread_indices(len(contexts))]
    sample = Sampler(client, bench.base_url)
    url = bench.base_url + "/completions"
    out = int(spec["output_tokens"])
    sampling = dict(spec.get("sampling") or {"temperature": 0.7, "top_p": 0.95, "top_k": 20})
    tick = float(spec.get("sample_seconds", 1.0))
    result = {"spec": {k: v for k, v in spec.items()}, "contexts": [
        {"id": c["id"], "game": c["game"], "family": c["family"], "tokens": c["tokens"]} for c in contexts],
        "points": [], "errors": []}

    def body(c, **extra):
        return {"model": bench.model, "prompt": c["prompt_ids"], **extra}

    async def post(name, c, tokens, payload):
        reservation = await gate.acquire(name, c["tokens"] + tokens)
        try:
            response = await client.post(url, json=payload)
            return response
        finally:
            gate.release(reservation)

    started = time.monotonic()
    before = await sample()
    replies = await asyncio.gather(*[post(f"prefill-{i}", c, 1, body(c, max_tokens=1, temperature=0.0))
                                     for i, c in enumerate(contexts)], return_exceptions=True)
    after = await sample()
    result["prefill"] = {"wall_seconds": round(time.monotonic() - started, 2),
                         "ok": sum(getattr(r, "status_code", 0) == 200 for r in replies),
                         "errors": [str(r)[:200] if isinstance(r, Exception) else r.text[:200]
                                    for r in replies if getattr(r, "status_code", 0) != 200][:3],
                         "uncached_tokens": after[1].get(COUNTERS["uncached"], 0) - before[1].get(COUNTERS["uncached"], 0)}
    result["gpu_memory_after_prefill"] = await gpu_memory()
    save()

    async def measured(streams):
        chosen = contexts[:streams]
        record = {"kind": "decode", "streams": streams,
                  "mean_prompt_tokens": round(sum(c["tokens"] for c in chosen) / streams, 1)}
        finished = asyncio.Event()

        async def one(i, c):
            try:
                await post(f"decode-{streams}-{i}", c, out, body(c, max_tokens=out, min_tokens=out, ignore_eos=True,
                                                               **sampling))
            finally:
                finished.set()

        tasks = [asyncio.create_task(one(i, c)) for i, c in enumerate(chosen)]
        try:
            deadline = time.monotonic() + float(spec.get("settle_seconds", 90))
            previous, quiet_run = None, 0
            need = int(spec.get("quiet_samples", 1))  # consecutive samples with every stream running, none computed
            while True:
                now = await sample()
                m = now[1]
                quiet = previous is not None and m.get(COUNTERS["uncached"], 0) == previous[1].get(COUNTERS["uncached"], 0)
                steady = m.get("vllm:num_requests_running", 0) >= streams and not m.get("vllm:num_requests_waiting", 0)
                quiet_run = quiet_run + 1 if (quiet and steady) else 0
                if quiet_run >= need:
                    break
                if finished.is_set() or time.monotonic() > deadline:
                    raise RuntimeError(f"decode {streams}: no steady decode (running {m.get('vllm:num_requests_running')}, "
                                       f"waiting {m.get('vllm:num_requests_waiting')}, gate reserved {gate.reserved})")
                previous = now
                await asyncio.sleep(tick)
            first, samples = now, [now]
            while now[0] - first[0] < float(spec.get("window_seconds", 20)) and not finished.is_set():
                await asyncio.sleep(tick)
                now = await sample()
                if not finished.is_set():
                    samples.append(now)
            record.update(window(first, samples[-1], streams))
            record["running_min"] = min(s[1].get("vllm:num_requests_running", 0) for s in samples)
            record["kv_cache_usage_max"] = max(s[1].get("vllm:kv_cache_usage_perc", 0) for s in samples)
            record["ended_early"] = finished.is_set()
            record["gpu_memory"] = await gpu_memory()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await wait_for_idle(sample)
        return record

    for streams in spec["decode_counts"]:
        if bench.remaining() is not None and bench.remaining() < 150:
            result["errors"].append(f"decode {streams}: skipped, {bench.remaining():.0f} s left")
            continue
        bench.log(json.dumps({"event": "spec_point", "streams": streams}), flush=True)
        try:
            result["points"].append(await measured(int(streams)))
        except Exception as exc:  # noqa: BLE001 - one bad point must not lose the sweep
            result["errors"].append(f"decode {streams}: {type(exc).__name__}: {str(exc)[:300]}")
        save()

    drift = spec.get("drift")
    if drift and (bench.remaining() is None or bench.remaining() > 180):
        try:
            result["drift"] = await drift_check(bench, client, post, url, sorted(contexts, key=lambda c: c["tokens"]),
                                                drift, sampling)
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"drift: {type(exc).__name__}: {str(exc)[:300]}")
        save()

    greedy = spec.get("greedy")
    if greedy:
        n, tokens = int(greedy["prompts"]), int(greedy["output_tokens"])
        chosen = contexts[:n]
        result["greedy"] = {"prompts": [c["id"] for c in chosen], "output_tokens": tokens, "runs": {}}

        async def greedy_one(i, c, label):
            response = await post(f"greedy-{label}-{i}", c, tokens,
                                  body(c, max_tokens=tokens, temperature=0.0, return_token_ids=True))
            data = response.json()
            choice = data["choices"][0]
            return {"id": c["id"], "status": response.status_code, "token_ids": choice.get("token_ids"),
                    "finish_reason": choice.get("finish_reason"), "text_head": (choice.get("text") or "")[:200]}

        for label in greedy.get("orders", ["concurrent", "sequential"]):
            if bench.remaining() is not None and bench.remaining() < 120:
                result["errors"].append(f"greedy {label}: skipped, {bench.remaining():.0f} s left")
                continue
            t0 = time.monotonic()
            try:
                if label == "concurrent":
                    rows = await asyncio.gather(*[greedy_one(i, c, label) for i, c in enumerate(chosen)])
                else:
                    rows = [await greedy_one(i, c, label) for i, c in enumerate(chosen)]
                result["greedy"]["runs"][label] = {"seconds": round(time.monotonic() - t0, 2), "rows": list(rows)}
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"greedy {label}: {type(exc).__name__}: {str(exc)[:300]}")
            save()
    result["gpu_memory_end"] = await gpu_memory()
    return result


async def drift_check(bench, client, post, url, contexts, spec, sampling):
    """Verify-path drift (spec-decode.md, 30 September): sampled continuations at the arena's sampling with their
    decode-path top-k logprobs (on MTP-k the verify forward's raw target logits), then the same tokens teacher-forced
    through the prefill path (``prompt_logprobs``, cold). Per generated position: KL(decode || prefill) on the coarse
    top-k partition, argmax agreement and |delta logprob| of the sampled token (kernel_gate.compare_positions). On
    MTP-0 this is the decode-vs-prefill numerics noise; on MTP-k the verify path's."""
    from .kernel_gate import compare_positions, token_id

    n, out, top = int(spec["prompts"]), int(spec["output_tokens"]), int(spec["top_logprobs"])
    chosen = contexts[:n]

    async def generate(i, c):
        response = await post(f"drift-gen-{i}", c, out, {
            "model": bench.model, "prompt": c["prompt_ids"], "max_tokens": out, "min_tokens": out, "ignore_eos": True,
            "logprobs": top, "return_tokens_as_token_ids": True, **sampling})
        response.raise_for_status()
        lp = response.json()["choices"][0]["logprobs"]
        return {"ids": [token_id(t) for t in lp["tokens"]],
                "decode": [{token_id(k): float(v) for k, v in (d or {}).items()} for d in lp["top_logprobs"]]}

    t0 = time.monotonic()
    generated = await asyncio.gather(*[generate(i, c) for i, c in enumerate(chosen)])
    gen_seconds = time.monotonic() - t0
    semaphore = asyncio.Semaphore(int(spec.get("teacher_concurrency", 4)))

    async def teacher(i, c, g):
        async with semaphore:
            ids = c["prompt_ids"] + g["ids"]
            response = await post(f"drift-tf-{i}", {"tokens": len(ids)}, 1, {
                "model": bench.model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": top})
            response.raise_for_status()
            entries = response.json()["choices"][0].get("prompt_logprobs") or []
        if len(entries) != len(ids):
            raise RuntimeError(f"prompt_logprobs length {len(entries)} != {len(ids)}")
        base = len(c["prompt_ids"])
        return [{token_id(k): float(v["logprob"]) for k, v in (entries[base + t] or {}).items()}
                for t in range(len(g["ids"]))]

    t0 = time.monotonic()
    prefill = await asyncio.gather(*[teacher(i, c, g) for i, (c, g) in enumerate(zip(chosen, generated))])
    tf_seconds = time.monotonic() - t0
    pairs, per_context = [], []
    for c, g, q in zip(chosen, generated, prefill):
        mine = list(zip(g["decode"], q, g["ids"]))
        pairs.extend(mine)
        per_context.append({"id": c["id"], "tokens": c["tokens"], **compare_positions(mine),
                            "chosen_logprob_decode": [round(p.get(t, float("nan")), 5) for p, _, t in mine],
                            "chosen_logprob_prefill": [round(r.get(t, float("nan")), 5) for _, r, t in mine],
                            "generated_ids": g["ids"]})
    return {"spec": spec, "generate_seconds": round(gen_seconds, 2), "teacher_seconds": round(tf_seconds, 2),
            "summary": compare_positions(pairs), "contexts": per_context}
