"""Serving-only throughput ceiling: fixed synthetic Duck-shaped requests on the arena server.

Answers "what is the most this GPU/server delivers for the arena's request shapes, and is
gameplay leaving it on the table?" without touching agent behaviour: the same server profile
(context, MTP, prefix caching, chunked prefill, cache reclamation) serves a grid of prompt
lengths x concurrency levels. Every point measures prefill and decode separately:

1. ``prefill`` pass: ``c`` distinct prompts of ~L tokens, one output token each, cold cache.
2. ``decode`` pass: the same prompts (now cached prefixes) with a forced fixed output length
   (``min_tokens = max_tokens``, ``ignore_eos``), so decode tokens/s is exactly comparable.

Admission goes through the arena's own block gate, so an oversized concurrency level is
serialised exactly as gameplay would be; the recorded ``peak_running_requests`` shows the
concurrency the server actually sustained. A ``shared_prefix`` point replays one long prefix
with distinct tails to price cached prefill. No games, actions or scoring are involved.
"""

import asyncio
import base64
import gzip
import io
import json
import re
import shutil
import time
from pathlib import Path

import httpx
from PIL import Image

from arc_infra.metrics import GpuMonitor
from arc_infra.model_templates import template_kwargs
from arc_infra.serving import MODEL
from arc_infra.serving_checks import wait_idle

from .cache import BlockBudget
from .gate import KVGate
from .isolation import isolate_trial

DEFAULT_GRID = {
    "prompt_tokens": [8192, 32768, 65536],
    "concurrency": [1, 2, 4, 8],
    "output_tokens": 512,
    "shared_prefix": {"prompt_tokens": 32768, "concurrency": 8, "tail_tokens": 256},
}
# Torch-profiler trace folder under the server directory (arena.serving's --profiler-config points here).
PROFILE_DIR = "torch-profile"
# Deterministic pseudo-text so a prompt is not a run of one token; prefixes differ per session.
WORDS = ("cell", "row", "col", "grid", "move", "red", "blue", "wall", "goal", "key", "door",
         "up", "down", "left", "right", "shift", "copy", "mirror", "count", "step", "block")


def synthetic_text(seed, words):
    parts = []
    value = seed * 2654435761 % 2**32
    for _ in range(words):
        value = (value * 1103515245 + 12345) % 2**31
        parts.append(WORDS[value % len(WORDS)])
    return " ".join(parts)


def board_image(seed):
    image = io.BytesIO()
    Image.new("RGB", (256, 256), (seed * 9 % 256, 90, 120)).save(image, format="PNG")
    return "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()


def request_body(prefix_seed, words, *, tail_seed=None, tail_words=0, model=MODEL,
                 output_tokens=1, reasoning_effort="xhigh", temperature=0.7, forced=False, image_seed=None):
    """A Duck-shaped request: system line, long user text, one board image, generation cap.

    ``forced`` fixes the output length (``min_tokens``/``ignore_eos``) so decode work is
    identical across points; the prefill pass generates one token. ``image_seed`` (kernel work, phase 6: injected
    prefill requests) gives the board image its own seed, so each request runs the vision encoder as an arena turn
    does; by default the image follows ``prefix_seed``.
    """
    text = f"Session {prefix_seed}. Observation log:\n" + synthetic_text(prefix_seed, words)
    if tail_words:
        text += f"\nLatest step {tail_seed}:\n" + synthetic_text(tail_seed, tail_words)
    text += "\nThink about the mechanics, then describe the next action."
    body = {
        "model": model, "temperature": temperature, "top_p": 0.95, "top_k": 20, "seed": prefix_seed,
        "max_tokens": int(output_tokens),
        "chat_template_kwargs": template_kwargs(model, {"enable_thinking": True, "reasoning_effort": reasoning_effort}),
        "messages": [
            {"role": "system", "content": "You are a coding agent solving a grid puzzle game."},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": board_image(
                    prefix_seed if image_seed is None else image_seed)}},
            ]},
        ],
    }
    if forced:
        body["min_tokens"] = int(output_tokens)
        body["ignore_eos"] = True
    return body


def transcript_tail(text, chars):
    """The last ``chars`` characters of a transcript, starting at a line boundary."""
    if chars >= len(text):
        return text
    cut = text.rfind("\n", 0, len(text) - chars)
    return text[cut + 1:] if cut >= 0 else text[-chars:]


def transcript_body(index, text, *, model=MODEL, output_tokens=1, reasoning_effort="xhigh", temperature=0.7,
                    forced=False):
    """A replay-shaped request: real agent transcript text as the prompt, no image.

    The session header differs per request so concurrent requests share no prefix. Real
    transcripts carry the code and observations the model tends to re-emit, which is what a
    prompt-lookup (n-gram) draft can exploit and synthetic word salad cannot.
    """
    body = {
        "model": model, "temperature": temperature, "top_p": 0.95, "top_k": 20, "seed": index,
        "max_tokens": int(output_tokens),
        "chat_template_kwargs": template_kwargs(model, {"enable_thinking": True, "reasoning_effort": reasoning_effort}),
        "messages": [
            {"role": "system", "content": "You are a coding agent solving a grid puzzle game."},
            {"role": "user", "content": f"Session {index}. Transcript of this game so far:\n" + text
                                        + "\n\nContinue from the latest step: think about the mechanics, then "
                                          "write the next python tool call."},
        ],
    }
    if forced:
        body["min_tokens"] = int(output_tokens)
        body["ignore_eos"] = True
    return body


def grid_points(spec):
    points = [{"kind": "distinct", "prompt_tokens": int(length), "concurrency": int(c)}
              for length in spec.get("prompt_tokens", []) for c in spec.get("concurrency", [])]
    shared = spec.get("shared_prefix")
    if shared:
        points.append({"kind": "shared_prefix", "prompt_tokens": int(shared["prompt_tokens"]),
                       "concurrency": int(shared["concurrency"]), "tail_tokens": int(shared["tail_tokens"])})
    transcript = spec.get("transcript")
    if transcript:
        points.extend({"kind": "transcript", "prompt_tokens": int(length), "concurrency": int(c)}
                      for length in transcript["prompt_tokens"] for c in transcript["concurrency"])
    # Serving lab (27 September): single (prompt, concurrency) points outside the product grid,
    # e.g. 64K at one and two streams beside a 4K/16K x 1-24 grid.
    points.extend({"kind": "distinct", "prompt_tokens": int(length), "concurrency": int(c)}
                  for length, c in spec.get("points") or [])
    # Kernel work, phase 6: a decode point followed by a mixed pass, the same streams decoding while prefill requests
    # arrive at a fixed rate (the arena's phase 2: ~0.34-0.45 requests/s of ~5K uncached tokens on a cached prefix).
    points.extend({"kind": "mixed", "prompt_tokens": int(m["prompt_tokens"]), "concurrency": int(m["concurrency"]),
                   "inject": validate_inject(m["inject"]), "mixed_output_tokens": int(m.get("output_tokens", 2048))}
                  for m in spec.get("mixed_points") or [])
    return points


def validate_inject(inject):
    """An injector: requests of ``tail_tokens`` fresh tokens on a shared cached ``prefix_tokens`` prefix, one new
    board image each, one output token, ``rate`` per second (a fixed interval)."""
    keys = {"prefix_tokens", "tail_tokens", "rate"}
    if not isinstance(inject, dict) or set(inject) != keys:
        raise ValueError(f"inject takes exactly {sorted(keys)}")
    if not (int(inject["prefix_tokens"]) >= 256 and int(inject["tail_tokens"]) >= 64
            and 0 < float(inject["rate"]) <= 5):
        raise ValueError("inject: prefix_tokens >= 256, tail_tokens >= 64, 0 < rate <= 5")
    return {"prefix_tokens": int(inject["prefix_tokens"]), "tail_tokens": int(inject["tail_tokens"]),
            "rate": float(inject["rate"])}


def toggled_points(spec):
    """The grid, or (kernel work, phase 5: ``h2d_toggle``) the grid once per host-copy mode on the same server:
    the stack's asynchronous PLE short-conv copies switched to blocking (the pinned behaviour) by a flag file, then
    back. Both passes use the same prompts (``seed_index``), so the only difference is the per-step stream sync."""
    points = grid_points(spec)
    toggle = spec.get("h2d_toggle")
    if not toggle:
        return points
    return [{**p, "h2d": mode, "seed_index": i} for mode in toggle["order"] for i, p in enumerate(points)]


def set_h2d_flag(path, blocking):
    flag = Path(path)
    if blocking:
        flag.touch()
    else:
        flag.unlink(missing_ok=True)


# Kernel families for the decode profile; the first matching pattern wins (lower-cased names).
KERNEL_FAMILIES = (
    ("moe", ("moe", "expert", "grouped_gemm", "permute", "topk_softmax", "routing")),
    ("attention", ("flash", "attn", "qsa", "indexer", "sparse", "paged", "kv_cache", "reshape_and_cache")),
    ("linear_attention", ("gdn", "fla", "chunk_", "recurrent", "gated_delta", "causal_conv", "mamba", "conv1d")),
    ("gemm", ("gemm", "cutlass", "nvjet", "cublas", "matmul", "sm120", "fp4", "splitk")),
    ("sampling", ("sampl", "top_k", "topk", "top_p", "argmax", "softmax", "multinomial", "penalt", "logit")),
    ("norm_act_elementwise", ("norm", "silu", "gelu", "act_and_mul", "elementwise", "vectorized", "reduce",
                              "rotary", "rope", "quant", "copy", "fill", "cat", "index")),
)


def kernel_family(name):
    lowered = name.lower()
    for family, needles in KERNEL_FAMILIES:
        if any(n in lowered for n in needles):
            return family
    return "other"


def summarize_trace(trace, top=25):
    """GPU time of a torch-profiler chrome trace: busy vs idle over the kernel window, top kernels, families.

    ``busy`` is the union of kernel/memcpy/memset intervals; ``idle`` is the rest of the window between
    the first GPU activity and the last one, i.e. launch gaps, host overhead and synchronisation.
    """
    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
           and e.get("dur") is not None]
    if not gpu:
        return {"kernels": 0}
    spans = sorted((float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in gpu)
    busy, current_start, current_end = 0.0, spans[0][0], spans[0][1]
    for start, end in spans[1:]:
        if start > current_end:
            busy += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    busy += current_end - current_start
    window = spans[-1][1] - spans[0][0]
    by_name, by_family = {}, {}
    for e in gpu:
        name = e.get("name", "?")
        entry = by_name.setdefault(name, {"us": 0.0, "count": 0})
        entry["us"] += float(e["dur"])
        entry["count"] += 1
        family = kernel_family(name) if e["cat"] == "kernel" else e["cat"]
        by_family[family] = by_family.get(family, 0.0) + float(e["dur"])
    total = sum(by_family.values())
    launches = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
                and e.get("name") in ("cudaGraphLaunch", "cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel")]
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1]["us"])[:top]
    return {
        "kernels": len(gpu), "window_ms": round(window / 1000, 3), "busy_ms": round(busy / 1000, 3),
        "busy_fraction": round(busy / window, 4) if window else None,
        "kernel_time_ms": round(total / 1000, 3),
        "families_ms": {k: round(v / 1000, 3) for k, v in sorted(by_family.items(), key=lambda kv: -kv[1])},
        "families_share": {k: round(v / total, 4) for k, v in sorted(by_family.items(), key=lambda kv: -kv[1])} if total else {},
        "launches": {name: sum(1 for e in launches if e["name"] == name) for name in sorted({e["name"] for e in launches})},
        "top_kernels": [{"name": name[:200], "family": kernel_family(name), "ms": round(v["us"] / 1000, 3),
                         "count": v["count"], "share": round(v["us"] / total, 4) if total else None}
                        for name, v in ranked],
    }


def summarize_pass(kind, elapsed, replies, before, after):
    """Throughput of one pass from the replies' usage and the engine counter deltas."""
    prompt = sum(r["usage"]["prompt_tokens"] for r in replies)
    completion = sum(r["usage"]["completion_tokens"] for r in replies)
    delta = {k: (after.get(k, 0.0) - before.get(k, 0.0)) for k in (
        "vllm:generation_tokens_total", "vllm:num_preemptions_total",
        "vllm:prompt_tokens_by_source_total:local_compute", "vllm:prompt_tokens_by_source_total:local_cache_hit",
        "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total")}
    drafts = delta["vllm:spec_decode_num_drafts_total"]
    return {
        "spec_drafts": drafts, "spec_draft_tokens": delta["vllm:spec_decode_num_draft_tokens_total"],
        "spec_accepted_tokens": delta["vllm:spec_decode_num_accepted_tokens_total"],
        "accepted_tokens_per_draft": round(delta["vllm:spec_decode_num_accepted_tokens_total"] / drafts, 3) if drafts else None,
        "pass": kind, "seconds": round(elapsed, 3), "requests": len(replies),
        "prompt_tokens": prompt, "completion_tokens": completion,
        "prompt_tokens_per_second": round(prompt / elapsed, 2) if elapsed else None,
        "completion_tokens_per_second": round(completion / elapsed, 2) if elapsed else None,
        "engine_generation_tokens": delta["vllm:generation_tokens_total"],
        "engine_generation_tokens_per_second": round(delta["vllm:generation_tokens_total"] / elapsed, 2) if elapsed else None,
        "computed_prompt_tokens": delta["vllm:prompt_tokens_by_source_total:local_compute"],
        "cached_prompt_tokens": delta["vllm:prompt_tokens_by_source_total:local_cache_hit"],
        "preemptions": delta["vllm:num_preemptions_total"],
    }


def kv_cache_tokens(server_dir):
    """The server's own "GPU KV cache size: N tokens" line (the last start in the server logs), or None."""
    found = None
    for path in sorted(Path(server_dir).glob("*.log")):
        for match in re.finditer(r"GPU KV cache size: ([\d,]+) tokens", path.read_text(errors="replace")):
            found = int(match.group(1).replace(",", ""))
    return found


def load_trace(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


class Bench:
    def __init__(self, study, output, *, server_dir, base_url, deadline_epoch=None, transport=None, log=print,
                 source=None, trial=None):
        self.study, self.output = study, Path(output)
        self.source = None if source is None else Path(source)
        self._transcript = None
        self.output.mkdir(parents=True, exist_ok=True)
        self.server_dir, self.base_url = server_dir, base_url.rstrip("/")
        self.deadline_epoch = deadline_epoch
        self.transport, self.log = transport, log
        # A trial may carry its own grid (serving lab: one server arm per trial, each with its points).
        self.spec = {**DEFAULT_GRID, **(study.get("throughput_bench") or {}),
                     **((trial or {}).get("throughput_bench") or {})}
        self.model = study.get("served_model", MODEL)
        self.effort = study.get("reasoning_effort", "xhigh")
        self.temperature = float(study.get("temperature", 0.7))
        self.cap = int(study["completion_cap"])
        self.calibration = {}
        self.points = []
        self.errors = []

    def client(self, timeout):
        return httpx.AsyncClient(timeout=timeout, trust_env=False, transport=self.transport)

    def remaining(self):
        return None if self.deadline_epoch is None else self.deadline_epoch - time.time()

    async def count(self, client, body):
        token_body = {k: body[k] for k in ("model", "messages", "chat_template_kwargs")}
        response = await client.post(self.base_url.removesuffix("/v1") + "/tokenize", json=token_body)
        response.raise_for_status()
        return int(response.json()["count"])

    async def words_for(self, client, target_tokens):
        """Words of synthetic text that tokenize to about ``target_tokens`` (two-step fit)."""
        if target_tokens in self.calibration:
            return self.calibration[target_tokens]["words"]
        words = max(64, int(target_tokens * 0.9))
        for _ in range(3):
            counted = await self.count(client, request_body(0, words, model=self.model, reasoning_effort=self.effort))
            if abs(counted - target_tokens) <= max(64, target_tokens // 100):
                break
            words = max(64, int(words * target_tokens / max(counted, 1)))
        counted = await self.count(client, request_body(0, words, model=self.model, reasoning_effort=self.effort))
        self.calibration[target_tokens] = {"words": words, "tokens": counted}
        return words

    def transcript(self):
        if self._transcript is None:
            spec = self.spec["transcript"]
            path = Path(spec["file"])
            if not path.is_absolute():
                if self.source is None:
                    raise ValueError("A relative transcript path needs the bundle source directory")
                path = self.source / path
            self._transcript = path.read_text()
        return self._transcript

    async def chars_for(self, client, target_tokens):
        """Transcript tail length (characters) that tokenizes to about ``target_tokens``."""
        key = f"transcript:{target_tokens}"
        if key in self.calibration:
            return self.calibration[key]["chars"]
        text = self.transcript()
        chars = min(len(text), int(target_tokens * 3.2))
        for _ in range(3):
            counted = await self.count(client, transcript_body(0, transcript_tail(text, chars), model=self.model,
                                                               reasoning_effort=self.effort))
            if abs(counted - target_tokens) <= target_tokens * 0.03 or chars >= len(text):
                break
            chars = min(len(text), max(256, int(chars * target_tokens / max(counted, 1))))
        counted = await self.count(client, transcript_body(0, transcript_tail(text, chars), model=self.model,
                                                           reasoning_effort=self.effort))
        self.calibration[key] = {"chars": chars, "tokens": counted, "transcript_chars": len(text)}
        return chars

    async def run_pass(self, kind, client, gate, budget, bodies, monitor):
        await monitor.sample()
        before = dict(monitor.samples[-1]["metrics"])
        replies, failures = [], []
        started = time.monotonic()

        async def one(index, body):
            counted = await self.count(client, body)
            reservation = await gate.acquire(f"bench-{index}", counted + body["max_tokens"])
            try:
                response = await client.post(self.base_url + "/chat/completions", json=body)
                if response.status_code != 200:
                    failures.append({"index": index, "status": response.status_code, "text": response.text[:500]})
                    return
                reply = response.json()
                replies.append({"index": index, "counted": counted, "usage": reply["usage"],
                                "finish_reason": reply["choices"][0].get("finish_reason")})
            finally:
                gate.release(reservation)

        await asyncio.gather(*(one(i, b) for i, b in enumerate(bodies)))
        elapsed = time.monotonic() - started
        await monitor.sample()
        after = monitor.samples[-1]["metrics"]
        summary = summarize_pass(kind, elapsed, replies, before, after)
        summary["failures"] = failures
        summary["finish_reasons"] = sorted({r["finish_reason"] for r in replies})
        running = [s["metrics"].get("vllm:num_requests_running", 0.0) for s in monitor.samples
                   if s["elapsed_seconds"] >= monitor.samples[-1]["elapsed_seconds"] - elapsed - 5]
        summary["peak_running_requests"] = max(running) if running else None
        return summary

    async def point(self, client, gate, budget, monitor, point):
        out = int(self.spec["output_tokens"])
        if point["kind"] == "shared_prefix":
            words = await self.words_for(client, point["prompt_tokens"])
            tail = max(8, point["tail_tokens"] * words // max(point["prompt_tokens"], 1))
            bodies = [request_body(1_000_000, words, tail_seed=i + 1, tail_words=tail, model=self.model,
                                   output_tokens=1, reasoning_effort=self.effort, temperature=self.temperature)
                      for i in range(point["concurrency"])]
        elif point["kind"] == "transcript":
            chars = await self.chars_for(client, point["prompt_tokens"])
            tail = transcript_tail(self.transcript(), chars)
            bodies = [transcript_body(10_000 * (point.get("seed_index", len(self.points)) + 1) + i, tail, model=self.model,
                                      output_tokens=1, reasoning_effort=self.effort, temperature=self.temperature)
                      for i in range(point["concurrency"])]
            words = None
        else:
            words = await self.words_for(client, point["prompt_tokens"])
            bodies = [request_body(10_000 * (point.get("seed_index", len(self.points)) + 1) + i, words, model=self.model,
                                   output_tokens=1, reasoning_effort=self.effort, temperature=self.temperature)
                      for i in range(point["concurrency"])]
        key = f"transcript:{point['prompt_tokens']}" if point["kind"] == "transcript" else point["prompt_tokens"]
        largest = budget.cost(self.calibration[key]["tokens"] + out + 256)
        record = {**point, "words": words, "calibration": dict(self.calibration[key]), "blocks_per_request": largest,
                  "fits_concurrently": budget.capacity // max(largest, 1), "passes": []}
        isolation = await isolate_trial(self.base_url)
        record["cache_reset"] = isolation
        record["passes"].append(await self.run_pass("prefill", client, gate, budget, bodies, monitor))
        forced = []
        for body in bodies:
            body = dict(body)
            body.update(max_tokens=out, min_tokens=out, ignore_eos=True)
            forced.append(body)
        record["passes"].append(await self.run_pass("decode", client, gate, budget, forced, monitor))
        if point["kind"] == "mixed":
            record["passes"].append(await self.mixed_pass(client, gate, budget, monitor, point, bodies))
        return record

    async def injector_bodies(self, client, inject, seed):
        """A factory of the injector's request bodies: index 0 is the bare shared prefix (the warm-up), index i > 0
        adds tail and board image ``seed + i``."""
        words = await self.words_for(client, inject["prefix_tokens"])
        tail = max(8, inject["tail_tokens"] * words // max(inject["prefix_tokens"], 1))

        def body(i):
            return request_body(seed, words, tail_seed=seed + i if i else None, tail_words=tail if i else 0,
                                image_seed=seed + i, model=self.model, output_tokens=1,
                                reasoning_effort=self.effort, temperature=self.temperature)

        return body

    async def inject_one(self, client, gate, body, index, records):
        started = time.monotonic()
        counted = await self.count(client, body)
        reservation = await gate.acquire(f"inject-{index}", counted + body["max_tokens"])
        try:
            admitted = time.monotonic()
            response = await client.post(self.base_url + "/chat/completions", json=body)
            entry = {"index": index, "counted": counted, "status": response.status_code,
                     "wait_seconds": round(admitted - started, 3), "seconds": round(time.monotonic() - admitted, 3)}
            if response.status_code == 200:
                entry["prompt_tokens"] = response.json()["usage"]["prompt_tokens"]
            records.append(entry)
        finally:
            gate.release(reservation)

    async def mixed_pass(self, client, gate, budget, monitor, point, bodies):
        """The point's streams decode ``mixed_output_tokens`` forced tokens (cached prefixes) while the injector sends
        prefill requests at its rate until the streams finish; the streams' tok/s against the decode pass is the
        prefill tax at this concurrency."""
        inject, out = point["inject"], point["mixed_output_tokens"]
        seed = 2_000_000 + 1_000 * len(self.points)
        make = await self.injector_bodies(client, inject, seed)
        warm = await self.run_pass("inject-warm", client, gate, budget, [make(0)], monitor)
        forced = [{**b, "max_tokens": out, "min_tokens": out, "ignore_eos": True} for b in bodies]
        records, tasks, done = [], [], asyncio.Event()

        async def injector():
            interval, index = 1.0 / inject["rate"], 0
            deadline = time.monotonic() + interval / 2
            while True:
                try:
                    await asyncio.wait_for(done.wait(), max(0.0, deadline - time.monotonic()))
                    return
                except TimeoutError:
                    pass
                index += 1
                tasks.append(asyncio.create_task(self.inject_one(client, gate, make(index), index, records)))
                deadline += interval

        running = asyncio.create_task(injector())
        try:
            summary = await self.run_pass("mixed", client, gate, budget, forced, monitor)
        finally:
            done.set()
            await running
        await asyncio.gather(*tasks, return_exceptions=True)
        ok = [r for r in records if r["status"] == 200]
        summary.update(inject=dict(inject), inject_warm_computed=warm["computed_prompt_tokens"], injected={
            "sent": len(tasks), "completed": len(ok), "failed": len(records) - len(ok),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in ok),
            "mean_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 3) if ok else None,
            "mean_wait_seconds": round(sum(r["wait_seconds"] for r in ok) / len(ok), 3) if ok else None})
        return summary

    async def new_traces(self, folder, before, timeout=180):
        """Trace files written since ``before``, once their sizes are stable."""
        deadline = time.monotonic() + timeout
        sizes = {}
        while time.monotonic() < deadline:
            current = {p: p.stat().st_size for p in folder.rglob("*.json*") if p.is_file() and p not in before} \
                if folder.is_dir() else {}
            if current and current == sizes:
                return sorted(current)
            sizes = current
            await asyncio.sleep(3)
        return sorted(sizes)

    async def profile(self, client, gate, budget, monitor):
        """Torch-profile a short forced decode at each profile concurrency (the server must enable it).

        The prefill pass warms the prefixes before ``/start_profile``, so the profiled window is decode;
        the server's profiler config may skip a few initial steps and cap the iterations.
        """
        spec = self.spec.get("profile")
        if not spec:
            return None
        folder = Path(self.server_dir) / PROFILE_DIR
        root = self.base_url.removesuffix("/v1")
        out = int(spec["output_tokens"])
        results = []
        # Kernel work, phase 6: after the decode profiles, one more at each "inject" concurrency with one prefill
        # request (its prefix cached first) arriving shortly after the profiled decode starts: a mixed step.
        inject_spec = spec.get("inject")
        runs = [(c, False) for c in spec["concurrency"]] + [(c, True) for c in (inject_spec or {}).get("concurrency", ())]
        for concurrency, mixed in runs:
            record = {"prompt_tokens": int(spec["prompt_tokens"]), "concurrency": int(concurrency), "output_tokens": out}
            if mixed:
                record["mixed"] = True
            results.append(record)
            remaining = self.remaining()
            if remaining is not None and remaining < 300:
                record["error"] = f"skipped: {remaining:.0f} s left"
                continue
            self.log(json.dumps({"event": "bench_profile", **record}), flush=True)
            try:
                words = await self.words_for(client, record["prompt_tokens"])
                bodies = [request_body(900_000 + 100 * int(concurrency) + i, words, model=self.model, output_tokens=1,
                                       reasoning_effort=self.effort, temperature=self.temperature)
                          for i in range(int(concurrency))]
                record["cache_reset"] = await isolate_trial(self.base_url)
                record["prefill"] = await self.run_pass("prefill", client, gate, budget, bodies, monitor)
                forced = [{**b, "max_tokens": out, "min_tokens": out, "ignore_eos": True} for b in bodies]
                inject = inject_spec if mixed else None
                if inject:
                    make = await self.injector_bodies(client, validate_inject(
                        {k: inject[k] for k in ("prefix_tokens", "tail_tokens", "rate")}), 3_000_000 + int(concurrency))
                    record["inject_warm"] = await self.run_pass("inject-warm", client, gate, budget, [make(0)],
                                                                monitor)
                before = set(folder.rglob("*")) if folder.is_dir() else set()
                response = await client.post(root + "/start_profile")
                if response.status_code != 200:
                    raise RuntimeError(f"start_profile {response.status_code}: {response.text[:200]}")
                injected = []
                if inject:
                    async def delayed():
                        await asyncio.sleep(float(inject.get("delay_seconds", 0.25)))
                        await self.inject_one(client, gate, make(1), 1, injected)

                    late = asyncio.create_task(delayed())
                record["decode"] = await self.run_pass("profile-decode", client, gate, budget, forced, monitor)
                if inject:
                    await late
                    record["injected"] = injected
                stop = time.monotonic()
                response = await client.post(root + "/stop_profile")
                if response.status_code != 200:
                    raise RuntimeError(f"stop_profile {response.status_code}: {response.text[:200]}")
                record["stop_seconds"] = round(time.monotonic() - stop, 3)
                destination = self.output / (f"profile-c{concurrency}" + ("-mixed" if mixed else ""))
                destination.mkdir(parents=True, exist_ok=True)
                record["traces"] = []
                for trace in await self.new_traces(folder, before):
                    target = destination / trace.name
                    shutil.move(str(trace), target)
                    entry = {"file": target.name, "bytes": target.stat().st_size}
                    try:
                        entry["summary"] = summarize_trace(load_trace(target))
                    except Exception as exc:  # noqa: BLE001 - keep the raw trace when it cannot be parsed
                        entry["summary_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    record["traces"].append(entry)
                for table in folder.glob("profiler_out_*.txt") if folder.is_dir() else []:
                    shutil.copyfile(table, destination / table.name)
            except Exception as exc:  # noqa: BLE001 - a failed profile must not lose the grid
                record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                self.errors.append(f"profile c={concurrency}: {record['error']}")
        return results

    async def sanity(self, client):
        """Greedy answers to fixed short questions (kernel work, phase 4: a served-output sanity check)."""
        results = []
        for item in self.spec.get("sanity") or []:
            body = {"model": self.model, "temperature": 0.0, "max_tokens": int(item.get("max_tokens", 64)), "seed": 0,
                    "chat_template_kwargs": template_kwargs(self.model, {"enable_thinking": False}),
                    "messages": [{"role": "user", "content": item["prompt"]}]}
            response = await client.post(self.base_url + "/chat/completions", json=body)
            record = {"prompt": item["prompt"], "expect": item.get("expect"), "status": response.status_code}
            if response.status_code == 200:
                text = response.json()["choices"][0]["message"].get("content") or ""
                record.update(text=text[:2000], ok=item.get("expect") is None or item["expect"] in text)
            else:
                record.update(error=response.text[:500], ok=False)
            results.append(record)
        return results

    async def tool_choice_probe(self, client):
        """Does the served template render the tool list under ``tool_choice: "none"`` exactly as under "auto"?

        The compaction request (28 September) shares the gameplay prefix by sending the gameplay tools with
        tool_choice "none"; its prompt must tokenize to the same length as a tool-enabled request's, and longer
        than the same messages without tools.
        """
        tools = [{"type": "function", "function": {
            "name": "python", "description": "Run Python in the persistent game kernel and return its output.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
        messages = [{"role": "system", "content": "You are a coding agent solving a grid puzzle game."},
                    {"role": "user", "content": "Summarize what you have learned about this game so far."}]
        result = {}
        for label, extra in (("auto", {"tools": tools, "tool_choice": "auto"}),
                             ("none", {"tools": tools, "tool_choice": "none"}), ("no_tools", {})):
            body = {"model": self.model, "temperature": 0.0, "max_tokens": 1, "messages": messages,
                    "chat_template_kwargs": template_kwargs(self.model, {"enable_thinking": True,
                                                                        "reasoning_effort": self.effort}), **extra}
            response = await client.post(self.base_url + "/chat/completions", json=body)
            result[label] = {"status": response.status_code}
            if response.status_code == 200:
                result[label]["prompt_tokens"] = response.json()["usage"]["prompt_tokens"]
            else:
                result[label]["error"] = response.text[:300]
        tokens = {k: v.get("prompt_tokens") for k, v in result.items()}
        result["ok"] = (None not in tokens.values() and tokens["auto"] == tokens["none"] > tokens["no_tools"])
        return result

    async def run(self):
        budget = BlockBudget.load(self.server_dir, float(self.study["kv_margin"]))
        gate = KVGate(budget.capacity, int(self.study["generation_slots"]), budget=budget)
        monitor = GpuMonitor(self.output, self.base_url)
        report = {"kind": "serving-bench", "status": "error", "spec": self.spec,
                  "server": self.study["server"], "reasoning_effort": self.effort,
                  "layout": {k: budget.layout.get(k) for k in ("num_blocks", "lookahead", "rule")},
                  "capacity_blocks": budget.capacity, "kv_cache_tokens": kv_cache_tokens(self.server_dir),
                  "points": self.points, "task_errors": self.errors}
        if isinstance(self.study.get("active_limit"), int) and self.study.get("windows"):
            # The arena's own admission check on this server's layout (recorded, not enforced, in a bench).
            from .capacity import focus_fit

            report["focus_fit"] = focus_fit(budget, int(self.study["windows"][0]), self.study["active_limit"])
        started = time.monotonic()
        await wait_idle(self.base_url)
        await monitor.sample()
        sampling = asyncio.create_task(monitor.run())
        try:
            async with self.client(600) as client:
                if self.spec.get("sanity"):
                    try:
                        report["sanity"] = await self.sanity(client)
                    except Exception as exc:  # noqa: BLE001 - the grid still runs
                        self.errors.append(f"sanity: {type(exc).__name__}: {str(exc)[:300]}")
                if self.spec.get("tool_choice_probe"):
                    try:
                        report["tool_choice_probe"] = await self.tool_choice_probe(client)
                    except Exception as exc:  # noqa: BLE001 - the grid still runs
                        self.errors.append(f"tool_choice_probe: {type(exc).__name__}: {str(exc)[:300]}")
                if self.spec.get("spec_sweep"):
                    # Speculative-decoding pilot (30 September; arena.spec_bench): recorded arena contexts.
                    from .spec_bench import spec_sweep

                    def save_spec():
                        (self.output / "bench.json").write_text(json.dumps(report, indent=2))

                    try:
                        report["spec_sweep"] = await spec_sweep(self, client, gate, budget, monitor,
                                                                self.spec["spec_sweep"], save=save_spec)
                    except Exception as exc:  # noqa: BLE001 - the report is still written
                        self.errors.append(f"spec_sweep: {type(exc).__name__}: {str(exc)[:300]}")
                    save_spec()
                if self.spec.get("stream_sweep") or self.spec.get("partial_check") or self.spec.get("partial_tf"):
                    # hero-v2 pre-check (29 September): decode / mixed step time against the stream count, then the
                    # --prefix-match-unit partial-hit correctness check (arena.hero_v2_precheck).
                    from . import hero_v2_precheck

                    def save():
                        (self.output / "bench.json").write_text(json.dumps(report, indent=2))

                    if self.spec.get("stream_sweep"):
                        try:
                            report["stream_sweep"] = await hero_v2_precheck.stream_sweep(
                                self, client, gate, budget, monitor, self.spec["stream_sweep"], save=save)
                        except Exception as exc:  # noqa: BLE001 - the checks still run
                            self.errors.append(f"stream_sweep: {type(exc).__name__}: {str(exc)[:300]}")
                        save()
                    remaining = self.remaining()
                    if self.spec.get("partial_check") and remaining is not None and remaining < 200:
                        self.errors.append(f"partial_check skipped: {remaining:.0f} s left")
                    elif self.spec.get("partial_check"):
                        check = self.spec["partial_check"]
                        try:
                            path = self.source / check["file"] if self.source else Path(check["file"])
                            sequences = json.loads(path.read_text())["sequences"]
                            report["partial_check"] = await hero_v2_precheck.partial_check(
                                client, self.base_url, model=self.model, sequences=sequences, spec=check,
                                log=self.log)
                        except Exception as exc:  # noqa: BLE001 - the replay still runs
                            self.errors.append(f"partial_check: {type(exc).__name__}: {str(exc)[:300]}")
                        save()
                    remaining = self.remaining()
                    if self.spec.get("partial_tf") and remaining is not None and remaining < 200:
                        self.errors.append(f"partial_tf skipped: {remaining:.0f} s left")
                    elif self.spec.get("partial_tf"):
                        # prefix-unit-int4 (29 September): the teacher-forced Latin-square test (arena.partial_tf).
                        from .partial_tf import partial_tf

                        check = self.spec["partial_tf"]
                        try:
                            path = self.source / check["file"] if self.source else Path(check["file"])
                            sequences = json.loads(path.read_text())["sequences"]
                            report["partial_tf"] = await partial_tf(
                                client, self.base_url, model=self.model, sequences=sequences, spec=check,
                                output=self.output, log=self.log, deadline=self.remaining)
                        except Exception as exc:  # noqa: BLE001 - the replay still runs
                            self.errors.append(f"partial_tf: {type(exc).__name__}: {str(exc)[:300]}")
                        save()
                if self.spec.get("quality"):
                    # 8-bit KV smoke (kv-cache-8bit.md): drift against a recorded BF16-cache reference first.
                    from .kv8_checks import quality

                    try:
                        await isolate_trial(self.base_url)
                        report["quality"] = await quality(self.study, self.spec["quality"], source=self.source,
                                                          output=self.output, base_url=self.base_url, log=self.log,
                                                          transport=self.transport)
                        await isolate_trial(self.base_url)
                    except Exception as exc:  # noqa: BLE001 - the grid still runs
                        self.errors.append(f"quality: {type(exc).__name__}: {str(exc)[:300]}")
                    (self.output / "bench.json").write_text(json.dumps(report, indent=2))
                for name in ("tail_check", "prefix_check", "prefix_replay"):
                    # Prefix cache v2 smoke (29 September): warm/cold correctness, then the multi-game replay; v3
                    # (request_end_tail) adds the teacher-forced tail check first.
                    if not self.spec.get(name):
                        continue
                    remaining = self.remaining()
                    if remaining is not None and remaining < 150:
                        self.errors.append(f"{name} skipped: {remaining:.0f} s left")
                        continue
                    from . import prefix_replay

                    try:
                        mamba_block = max(g["block_size"] for g in budget.layout["groups"]
                                          if g["type"] == "MambaSpec")
                        if name == "tail_check":
                            check = self.spec[name]
                            path = self.source / check["file"] if self.source else Path(check["file"])
                            report[name] = await prefix_replay.tail_check(
                                client, self.base_url, model=self.model,
                                sequences=json.loads(path.read_text())["sequences"],
                                spec={**check, "block": mamba_block}, log=self.log)
                        elif name == "prefix_check":
                            report[name] = await prefix_replay.prefix_check(
                                client, self.base_url, model=self.model, transcript=self.transcript(),
                                spec=self.spec[name], log=self.log)
                        else:
                            report[name] = await prefix_replay.prefix_replay(
                                client, self.base_url, model=self.model, effort=self.effort,
                                transcript=self.transcript(), spec=self.spec[name], block=mamba_block, log=self.log)
                    except Exception as exc:  # noqa: BLE001 - the other check still runs
                        self.errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
                    (self.output / "bench.json").write_text(json.dumps(report, indent=2))
                for point in toggled_points(self.spec):
                    remaining = self.remaining()
                    if remaining is not None and remaining < 300:
                        self.errors.append(f"stopped before {point}: {remaining:.0f} s left")
                        break
                    if "h2d" in point:
                        set_h2d_flag(self.spec["h2d_toggle"]["flag"], point["h2d"] == "blocking")
                    self.log(json.dumps({"event": "bench_point", **point}), flush=True)
                    try:
                        self.points.append(await self.point(client, gate, budget, monitor, point))
                    except Exception as exc:  # noqa: BLE001 - one bad point must not lose the grid
                        self.errors.append(f"{point}: {type(exc).__name__}: {str(exc)[:300]}")
                        self.points.append({**point, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                    (self.output / "bench.json").write_text(json.dumps(report, indent=2))
                if self.spec.get("h2d_toggle"):
                    set_h2d_flag(self.spec["h2d_toggle"]["flag"], False)  # the profile and later work run async
                if self.spec.get("kv8_microbench"):
                    # The QSA kernel alone, BF16 against 8-bit, in the server's runtime while it idles.
                    from .kv8_checks import microbench_async

                    remaining = self.remaining()
                    if remaining is not None and remaining < 240:
                        self.errors.append(f"kv8 microbench skipped: {remaining:.0f} s left")
                    else:
                        await wait_idle(self.base_url)
                        timeout = 420 if remaining is None else int(min(420, remaining - 90))
                        report["kv8_microbench"] = await microbench_async(self.server_dir, self.output,
                                                                          timeout=timeout)
                    (self.output / "bench.json").write_text(json.dumps(report, indent=2))
                if self.spec.get("profile"):
                    report["profile"] = await self.profile(client, gate, budget, monitor)
                    (self.output / "bench.json").write_text(json.dumps(report, indent=2))
                if self.spec.get("phase6_microbench"):
                    # Kernel work, phase 6: dense tiles / fused split-K, QSA launch profiles and the W4 error, in the
                    # server's runtime while it idles (after the profiles, so they never overlap).
                    from .phase6_checks import microbench_async as phase6_microbench

                    remaining = self.remaining()
                    if remaining is not None and remaining < 300:
                        self.errors.append(f"phase6 microbench skipped: {remaining:.0f} s left")
                    else:
                        await wait_idle(self.base_url)
                        timeout = None if remaining is None else int(remaining - 120)
                        report["phase6_microbench"] = await phase6_microbench(
                            self.server_dir, self.output, self.spec["phase6_microbench"], timeout=timeout)
                    (self.output / "bench.json").write_text(json.dumps(report, indent=2))
            await wait_idle(self.base_url)
            report["status"] = "complete" if not self.errors else "complete_with_errors"
        finally:
            sampling.cancel()
            await asyncio.gather(sampling, return_exceptions=True)
            await monitor.sample()
            report["engine"] = monitor.summary()
            report["preemption_free"] = not report["engine"].get("preemptions", 0)
            report["gate"] = gate.snapshot()
            report["calibration"] = self.calibration
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            (self.output / "bench.json").write_text(json.dumps(report, indent=2))
        return report


async def run_bench(study, trial, *, output, server_dir, base_url, deadline_epoch=None, transport=None, log=print,
                    source=None):
    bench = Bench(study, output, server_dir=server_dir, base_url=base_url,
                  deadline_epoch=deadline_epoch, transport=transport, log=log, source=source, trial=trial)
    return await bench.run()
