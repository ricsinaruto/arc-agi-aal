"""hero-v2 pre-check (29 September; docs/validation-20260927/hero-v2-precheck.md).

One server start of hero-v2's exact serving at ``max_num_seqs`` 32, with ``--prefix-match-unit 16`` and
``--enable-prompt-tokens-details`` (neither changes the GPU memory plan or the decode step: the hash unit lives in the
scheduler's prefix-cache index, the details flag in the API server). The serving bench then runs:

``stream_sweep`` (decode throughput against the stream count):
- 32 distinct Duck-shaped prompts whose lengths are the quantiles of hero-v2's recorded phase-2 prompts (median ~25K,
  p90 ~41K), in a spread order so that any first-n subset keeps the distribution; all prefilled once;
- a decode pass per stream count (the first n prompts, forced output, cached prefixes): the engine counters are read
  directly every second, and a window of steady decode (all n running, no prefill) gives ms per engine step,
  generated tokens per step and aggregate / per-stream tok/s; the requests are then aborted (client disconnect);
- a mixed pass per stream count: the same, with ~2.4K-token uncached prefill requests injected at the recorded
  phase-2 rate (~1,100-1,200 uncached tok/s; kernels-phase6's injector), so step time can be fitted on streams and
  uncached tokens per step.

``partial_check`` (``--prefix-match-unit`` correctness, teacher-forced): on recorded arena token sequences, a primer
prompt leaves a cached state at a chosen position, then single-token probes that extend it are answered warm (their
next-token top-20 is compared at the same positions). Conditions: the partial-tail state inside a block (W16 at
6.5 blocks, W16b at 6.25), full-block states (W1600a at 6 blocks, W1600b at 5 blocks, the floor pair: two correct
states reached through different chunk splits), cold ``prompt_logprobs`` (C; skips reading the cache) and a
sensitivity control S (C with the 800 tokens between the block boundary and the partial state deleted). A partial
hit must land exactly on its 16-token position (``cached_tokens``) and agree with the full-block paths as closely as
they agree with each other.
"""

import asyncio
import json
import time

from arc_infra.metrics import parse_metrics

from .isolation import isolate_trial
from .kernel_gate import compare_positions, token_id
from .throughput_bench import request_body

# Counters read for a window (parse_metrics sums the histogram series: *_count / *_sum).
COUNTERS = {
    "steps": "vllm:iteration_tokens_total_count",
    "scheduled_tokens": "vllm:iteration_tokens_total_sum",
    "generated": "vllm:generation_tokens_total",
    "uncached": "vllm:prompt_tokens_by_source_total:local_compute",
    "cached": "vllm:prompt_tokens_by_source_total:local_cache_hit",
    "itl_sum": "vllm:inter_token_latency_seconds_sum",
    "itl_count": "vllm:inter_token_latency_seconds_count",
    "preemptions": "vllm:num_preemptions_total",
}


def spread_order(count):
    """Indices 0..count-1 in bit-reversed order (count a power of two): every prefix is spread over the range."""
    bits = max(1, (count - 1).bit_length())
    if 1 << bits != count:
        raise ValueError("spread_order needs a power of two")
    return [int(format(i, f"0{bits}b")[::-1], 2) for i in range(count)]


def spread_indices(count):
    """``spread_order`` for any count: the bit-reversed order of the next power of two, kept below ``count``."""
    size = 1 << max(1, (count - 1).bit_length())
    return [i for i in spread_order(size) if i < count]


async def gpu_memory():
    """One nvidia-smi read (MiB): used / free / total of GPU 0; None when unavailable (never fails a point)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 10)
        used, free, total = (float(x) for x in out.decode().splitlines()[0].split(","))
        return {"used_mib": used, "free_mib": free, "total_mib": total, "epoch": round(time.time(), 1)}
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def window_stats(first, last, streams):
    """Engine counters between two samples ``(epoch, metrics)`` -> per-step and per-second rates."""
    seconds = last[0] - first[0]
    delta = {k: last[1].get(name, 0.0) - first[1].get(name, 0.0) for k, name in COUNTERS.items()}
    steps = delta["steps"]
    if seconds <= 0 or steps <= 0:
        return {"seconds": round(seconds, 3), "steps": steps, "valid": False}
    return {
        "valid": True, "streams": streams, "seconds": round(seconds, 3), "steps": int(steps),
        "step_ms": round(1000 * seconds / steps, 3),
        "generated_per_step": round(delta["generated"] / steps, 3),
        "uncached_per_step": round(delta["uncached"] / steps, 2),
        "scheduled_tokens_per_step": round(delta["scheduled_tokens"] / steps, 2),
        "generated_tok_s": round(delta["generated"] / seconds, 2),
        "per_stream_tok_s": round(delta["generated"] / seconds / streams, 3),
        "uncached_tok_s": round(delta["uncached"] / seconds, 1),
        "cached_tok_s": round(delta["cached"] / seconds, 1),
        "itl_ms": round(1000 * delta["itl_sum"] / delta["itl_count"], 3) if delta["itl_count"] else None,
        "preemptions": delta["preemptions"],
    }


def linear_fit(xs, ys):
    """Ordinary least squares y = a + b x; returns a, b and the residual RMS."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rms = (sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5
    return {"intercept_ms": round(a, 3), "ms_per_stream": round(b, 4), "residual_rms_ms": round(rms, 3), "points": n}


class Sampler:
    """Direct /metrics reads (the bench monitor samples only every 5 s)."""

    def __init__(self, client, base_url):
        self.client, self.url = client, base_url.removesuffix("/v1") + "/metrics"

    async def __call__(self):
        response = await self.client.get(self.url, timeout=10)
        response.raise_for_status()
        return time.time(), parse_metrics(response.text)


async def stream_sweep(bench, client, gate, budget, monitor, spec, *, save=lambda: None, phases=("decode", "mixed")):
    """Prefill the sweep's prompts, then one steady-state window per decode / mixed point (module docstring)."""
    state = bench.sweep_state if getattr(bench, "sweep_state", None) else None
    result = state["result"] if state else {"spec": spec, "points": [], "errors": []}
    sample = Sampler(client, bench.base_url)
    out = int(spec["output_tokens"])
    tick = float(spec.get("sample_seconds", 1.0))
    if state is None:
        lengths = [int(n) for n in spec["lengths"]]
        order = spread_indices(len(lengths))
        lengths = [lengths[i] for i in order]
        reference = int(spec.get("calibrate_tokens", 24576))
        words_ref = await bench.words_for(client, reference)
        tokens_ref = bench.calibration[reference]["tokens"]
        bodies, counts = [], []
        for i, target in enumerate(lengths):
            words = max(64, int(words_ref * target / tokens_ref))
            body = request_body(4_000_000 + i, words, model=bench.model, output_tokens=1,
                                reasoning_effort=bench.effort, temperature=bench.temperature)
            bodies.append(body)
            counts.append(await bench.count(client, body))
        result.update(targets=lengths, prompt_tokens=counts)
        result["cache_reset"] = await isolate_trial(bench.base_url)
        started = time.monotonic()
        result["prefill"] = await bench.run_pass("sweep-prefill", client, gate, budget, bodies, monitor)
        result["prefill"]["wall_seconds"] = round(time.monotonic() - started, 2)
        result["gpu_memory_after_prefill"] = await gpu_memory()
        state = bench.sweep_state = {"result": result, "bodies": bodies, "counts": counts}
        save()
    bodies, counts = state["bodies"], state["counts"]

    async def measured(kind, streams, inject=None, seed=0, overrides=None, subset="spread", label=None):
        # subset "spread": the first n prompts (spread over the length range); "longest": the n longest prompts.
        chosen = (list(range(streams)) if subset == "spread"
                  else sorted(range(len(counts)), key=lambda i: -counts[i])[:streams])
        record = {"kind": kind, "streams": streams, "label": label or kind, "subset": subset,
                  "overrides": overrides or {},
                  "mean_prompt_tokens": round(sum(counts[i] for i in chosen) / streams, 1),
                  "max_prompt_tokens": max(counts[i] for i in chosen)}
        forced = [{**bodies[i], "max_tokens": out, "min_tokens": out, "ignore_eos": True, **(overrides or {})}
                  for i in chosen]
        sizes = [counts[i] for i in chosen]
        finished, replies = asyncio.Event(), []

        async def one(i, body):
            reservation = await gate.acquire(f"sweep-{kind}-{streams}-{i}", sizes[i] + out)
            try:
                response = await client.post(bench.base_url + "/chat/completions", json=body)
                replies.append(response.status_code)
            finally:
                gate.release(reservation)
                finished.set()

        tasks = [asyncio.create_task(one(i, b)) for i, b in enumerate(forced)]
        injector, injected, inject_tasks, stop = None, [], [], asyncio.Event()
        try:
            # Steady decode: every stream running, nothing waiting, no prompt tokens computed for 2 s.
            deadline = time.monotonic() + float(spec.get("settle_seconds", 90))
            previous = None
            while True:
                now = await sample()
                m = now[1]
                quiet = previous is not None and m.get(COUNTERS["uncached"], 0) == previous[1].get(COUNTERS["uncached"], 0)
                if (m.get("vllm:num_requests_running", 0) >= streams and not m.get("vllm:num_requests_waiting", 0)
                        and quiet):
                    break
                if finished.is_set() or time.monotonic() > deadline:
                    raise RuntimeError(f"{kind} {streams}: no steady decode (running "
                                       f"{m.get('vllm:num_requests_running')}, finished {finished.is_set()})")
                previous = now
                await asyncio.sleep(tick)
            record["settle_epoch"] = now[0]
            if inject is not None:
                make = state["inject_make"]
                interval = 1.0 / inject["rate"]

                async def run_injector():
                    index, next_at = 0, time.monotonic()
                    while not stop.is_set():
                        index += 1
                        inject_tasks.append(asyncio.create_task(
                            bench.inject_one(client, gate, make(seed + index), seed + index, injected)))
                        next_at += interval
                        try:
                            await asyncio.wait_for(stop.wait(), max(0.0, next_at - time.monotonic()))
                        except TimeoutError:
                            pass

                injector = asyncio.create_task(run_injector())
                await asyncio.sleep(float(spec.get("inject_lead_seconds", 4)))
                now = await sample()
            first, samples = now, [now]
            window = float(spec.get("window_seconds", 20))
            while now[0] - first[0] < window:
                await asyncio.sleep(tick)
                if finished.is_set():
                    break
                now = await sample()
                if finished.is_set():
                    break
                samples.append(now)
            record.update(window_stats(first, samples[-1], streams))
            record["running_min"] = min(s[1].get("vllm:num_requests_running", 0) for s in samples)
            record["kv_cache_usage_max"] = max(s[1].get("vllm:kv_cache_usage_perc", 0) for s in samples)
            record["ended_early"] = finished.is_set()
            record["gpu_memory"] = await gpu_memory()
        finally:
            stop.set()
            if injector is not None:
                await asyncio.gather(injector, return_exceptions=True)
            for task in tasks:
                task.cancel()  # client disconnect: vLLM aborts the forced requests
            await asyncio.gather(*tasks, *inject_tasks, return_exceptions=True)
        if inject is not None:
            ok = [r for r in injected if r.get("status") == 200]
            record["inject"] = {**inject, "sent": len(injected), "completed": len(ok),
                                "mean_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 3) if ok else None,
                                "mean_wait_seconds": round(sum(r["wait_seconds"] for r in ok) / len(ok), 3)
                                if ok else None}
        await wait_for_idle(sample)
        return record

    if "decode" in phases:
        for streams in spec["decode_counts"]:
            if bench.remaining() is not None and bench.remaining() < 150:
                result["errors"].append(f"decode {streams}: skipped, {bench.remaining():.0f} s left")
                continue
            bench.log(json.dumps({"event": "sweep_point", "kind": "decode", "streams": streams}), flush=True)
            try:
                result["points"].append(await measured("decode", int(streams)))
            except Exception as exc:  # noqa: BLE001 - one bad point must not lose the sweep
                result["errors"].append(f"decode {streams}: {type(exc).__name__}: {str(exc)[:300]}")
            save()
    for variant in spec.get("variants") or []:
        for streams in variant["streams"]:
            if bench.remaining() is not None and bench.remaining() < 150:
                result["errors"].append(f"{variant['label']} {streams}: skipped, {bench.remaining():.0f} s left")
                continue
            bench.log(json.dumps({"event": "sweep_point", "kind": variant["label"], "streams": streams}), flush=True)
            try:
                result["points"].append(await measured("decode", int(streams), overrides=variant.get("overrides"),
                                                       subset=variant.get("subset", "spread"),
                                                       label=variant["label"]))
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"{variant['label']} {streams}: {type(exc).__name__}: {str(exc)[:300]}")
            save()
    if "mixed" in phases and spec.get("mixed_counts"):
        inject = spec["inject"]
        if "inject_make" not in state:
            state["inject_make"] = await bench.injector_bodies(client, inject, 6_000_000)
            result["inject_warm"] = await bench.run_pass("inject-warm", client, gate, budget,
                                                         [state["inject_make"](0)], monitor)
        for k, streams in enumerate(spec["mixed_counts"]):
            if bench.remaining() is not None and bench.remaining() < 150:
                result["errors"].append(f"mixed {streams}: skipped, {bench.remaining():.0f} s left")
                continue
            bench.log(json.dumps({"event": "sweep_point", "kind": "mixed", "streams": streams}), flush=True)
            try:
                result["points"].append(await measured("mixed", int(streams), inject=inject, seed=1000 * (k + 1)))
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"mixed {streams}: {type(exc).__name__}: {str(exc)[:300]}")
            save()
    decode = [p for p in result["points"] if p["kind"] == "decode" and p.get("valid")
              and p.get("label", "decode") == "decode"]
    result["decode_fit"] = linear_fit([p["generated_per_step"] for p in decode], [p["step_ms"] for p in decode])
    return result


async def wait_for_idle(sample, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, m = await sample()
        if not m.get("vllm:num_requests_running", 0) and not m.get("vllm:num_requests_waiting", 0):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError("sweep: aborted requests still running")


# ----------------------------------------------------------------------------- partial-hit correctness
def probe_lengths(block, count, low_offset=24, high_margin=8):
    """Probe prompt lengths inside the block after ``6 x block``, longest first (each probe then registers its own
    partial tail beyond every later probe, so all of them resume from the condition's primer state)."""
    base, top = 6 * block + block // 2 + low_offset, 7 * block - high_margin
    step = max(1, (top - base) // max(1, count - 1))
    return sorted({min(top, base + i * step) for i in range(count)}, reverse=True)


def conditions(block, unit):
    """name -> (primer length, expected hit, reset before every probe)."""
    b6 = 6 * block
    return {
        "W16": (b6 + block // 2 + unit // 2, b6 + block // 2, False),      # partial tail at 6.5 blocks
        "W16b": (b6 + block // 4 + unit // 2, b6 + block // 4, False),     # partial tail at 6.25 blocks
        "W1600a": (b6 + unit - 1, b6, False),                               # full-block state at 6 blocks
        "W1600b": (5 * block + unit // 2, 5 * block, True),                 # full-block state at 5 blocks
    }


def top_dist(entry):
    """token -> logprob from a top_logprobs dict ({"token_id:N": lp}) or a prompt_logprobs entry ({"N": {logprob}})."""
    return {token_id(t): float(v["logprob"] if isinstance(v, dict) else v) for t, v in (entry or {}).items()}


async def completion(client, base_url, body):
    response = await client.post(base_url + "/completions", json=body)
    if response.status_code != 200:
        raise RuntimeError(f"completions {response.status_code}: {response.text[:300]}")
    return response.json()


async def partial_check(client, base_url, *, model, sequences, spec, log=print):
    block, unit = int(spec.get("block", 1600)), int(spec.get("unit", 16))
    lengths = probe_lengths(block, int(spec.get("probes", 24)))
    end = 7 * block
    top = int(spec.get("top_logprobs", 20))
    results = {"block": block, "unit": unit, "probe_lengths": lengths, "sequences": []}

    def probe_body(ids, max_tokens=1, logprobs=top):
        return {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
                "logprobs": logprobs, "return_tokens_as_token_ids": True}

    async def teacher(ids):
        reply = await completion(client, base_url, {"model": model, "prompt": ids, "max_tokens": 1,
                                                    "temperature": 0.0, "prompt_logprobs": top})
        entries = reply["choices"][0].get("prompt_logprobs") or []
        if len(entries) != len(ids):
            raise RuntimeError(f"prompt_logprobs length {len(entries)} != {len(ids)}")
        return entries

    async def warm(ids, length):
        reply = await completion(client, base_url, probe_body(ids[:length]))
        choice = reply["choices"][0]
        details = (reply.get("usage") or {}).get("prompt_tokens_details") or {}
        tops = (choice.get("logprobs") or {}).get("top_logprobs") or []
        return {"length": length, "cached_tokens": details.get("cached_tokens"), "dist": top_dist(tops[0] if tops else {})}

    for index, ids in enumerate(sequences):
        ids = [int(t) for t in ids]
        if len(ids) < end:
            raise ValueError(f"partial_check sequence {index}: {len(ids)} tokens < {end}")
        row = {"index": index, "conditions": {}}
        started = time.monotonic()
        await isolate_trial(base_url)
        cold = await teacher(ids[:end])
        row["conditions"]["C"] = {"dists": [top_dist(cold[n]) for n in lengths]}
        await isolate_trial(base_url)
        cut = ids[:6 * block] + ids[6 * block + block // 2:end]
        sens = await teacher(cut)
        row["conditions"]["S"] = {"dists": [top_dist(sens[n - block // 2]) for n in lengths],
                                  "deleted": [6 * block, 6 * block + block // 2]}
        for name, (primer, expected, reset_each) in conditions(block, unit).items():
            await isolate_trial(base_url)
            probes = []
            for k, length in enumerate(lengths):
                if reset_each and k:
                    await isolate_trial(base_url)
                if reset_each or not k:
                    await completion(client, base_url, probe_body(ids[:primer], logprobs=None))
                probes.append(await warm(ids, length))
            hits = [p["cached_tokens"] for p in probes]
            row["conditions"][name] = {"primer": primer, "expected_hit": expected, "cached_tokens": hits,
                                       "hit_as_expected": sum(h == expected for h in hits),
                                       "dists": [p["dist"] for p in probes]}
        chosen = [ids[n] for n in lengths]
        row["pairs"] = {}
        for a, b in (("W1600a", "W1600b"), ("W1600a", "W16"), ("W1600a", "W16b"), ("W1600b", "W16"),
                     ("W16", "W16b"), ("C", "W1600a"), ("C", "W1600b"), ("C", "W16"), ("C", "W16b"),
                     ("C", "S")):
            row["pairs"][f"{a}|{b}"] = compare_positions(list(zip(row["conditions"][a]["dists"],
                                                                  row["conditions"][b]["dists"], chosen)))
        row["seconds"] = round(time.monotonic() - started, 1)
        log(json.dumps({"event": "partial_check", "index": index, "seconds": row["seconds"],
                        "hits": {k: v.get("hit_as_expected") for k, v in row["conditions"].items()},
                        "mean_kl": {k: v["mean_kl"] for k, v in row["pairs"].items()}}), flush=True)
        results["sequences"].append(row)
    await isolate_trial(base_url)
    results["summary"] = summarize_partial(results)
    return results


def summarize_partial(results):
    """Pooled pair statistics over the sequences, and the pass rule stated before the run."""
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
                entry = hits.setdefault(name, {"as_expected": 0, "probes": 0})
                entry["as_expected"] += cond["hit_as_expected"]
                entry["probes"] += len(cond["cached_tokens"])
    floor = pairs.get("W1600a|W1600b", {})
    test = pairs.get("W1600a|W16", {})
    verdict = None
    if floor.get("mean_kl") is not None and test.get("mean_kl") is not None:
        # Pass (stated before the run): the partial-tail hit lands on its 16-token position on every probe, and its
        # distance to the full-block path is within 1.5x the floor pair's (+0.005 nats), with top-1 agreement at most
        # 5 points below the floor's.
        verdict = {
            "hits_exact": hits.get("W16", {}).get("as_expected") == hits.get("W16", {}).get("probes"),
            "kl_within": test["mean_kl"] <= 1.5 * floor["mean_kl"] + 0.005,
            "top1_within": test["top1_agreement"] >= floor["top1_agreement"] - 0.05,
        }
        verdict["pass"] = all(verdict.values())
    return {"pairs": pairs, "hits": hits, "verdict": verdict}
