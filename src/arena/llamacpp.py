"""llama.cpp (llama-server) arena backend for model screens (27 September).

- The server runs ``-np slots`` with ``-c slots * window`` and no unified KV cache, so every slot
  owns exactly ``window`` tokens: admission is a plain slot gate (``SlotCapacity``), not a vLLM
  KV-pool estimate, and a request that fits the window can never be preempted.
- Token counts come from llama-server's ``/v1/chat/completions/input_tokens`` (the chat template,
  tools and ``chat_template_kwargs`` rendered by the server itself; ``arc_infra.model``).
- ``/metrics`` (``--metrics``) is read through the vLLM names the arena already uses
  (``arc_infra.metrics.LLAMACPP_ALIASES``); llama-server has no preemption counter.
- ``serve_check`` records decode tokens/s at two depths and 1/2/4 streams, a Duck-format python
  tool round trip (with the prefix reuse of the second turn) and a solo-versus-concurrent recall check
  (the GLM PR warns that an earlier multi-sequence path returned NaN or divergent logits silently).
"""

import asyncio
import json
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from .llamacpp_profiles import profile as get_profile

HOST, PORT = "127.0.0.1", 1234
BASE_URL = f"http://{HOST}:{PORT}/v1"
CACHE_RAM_MIB = 16384  # host copies of idle slots: 8 games share 4 slots
READY_TIMEOUT = 2400


def study_profile(study):
    """The profile pinned in the study (the bundle's copy wins over the source registry)."""
    spec = study.get("llamacpp") or {}
    if "profile" not in spec:
        raise ValueError("A llama.cpp study names its profile")
    return {**get_profile(spec["profile"]), **spec.get("overrides", {})}


def server_argv(prof, *, binary, weights_dir, slots, window, host=HOST, port=PORT, cache_ram_mib=CACHE_RAM_MIB,
                kv_unified=False):
    sampling = prof["sampling"]
    return [
        str(binary), "-m", str(Path(weights_dir) / prof["model_file"]), *prof["server_args"],
        *(["--mmproj", str(Path(weights_dir) / prof["mmproj"])] if prof.get("mmproj") else []),
        "-c", str(int(slots) * int(window)), "-np", str(int(slots)),
        "--kv-unified" if kv_unified else "--no-kv-unified",
        "--host", host, "--port", str(port), "-a", prof["served_model"], "--metrics", "--no-webui",
        "--cache-ram", str(cache_ram_mib),
        "--temp", str(sampling["temperature"]), "--top-p", str(sampling["top_p"]),
        "--top-k", str(sampling["top_k"]), "--min-p", str(sampling["min_p"]),
    ]


def check_flags(binary, argv):
    """Every long flag of ``argv`` appears in ``llama-server --help`` (fail before the model loads)."""
    result = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=60)
    text = result.stdout + result.stderr
    if "--port" not in text:
        raise RuntimeError(f"llama-server --help failed (exit {result.returncode}): {text[-2000:]}")
    missing = [a for a in argv if a.startswith("--") and a not in text]
    if missing:
        raise RuntimeError(f"llama-server does not know {missing}")
    return True


def gpu_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        used, total = (float(v) for v in out.splitlines()[0].split(","))
        return {"used_mib": used, "total_mib": total}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# llama-server lines worth a live log line (the rest, per-request slot chatter, stays in the file).
LIVE_SERVER = re.compile(r"error|warn|nan|fail|exception|abort|out of memory|CUDA|model loaded|listening|"
                         r"load_tensors: +(CUDA|CPU)|KV buffer|compute buffer|n_ctx|total_slots", re.IGNORECASE)


def pump(pipe, path, *, live=None, prefix=""):
    """Copy a child's output to ``path`` and, filtered by ``live`` (None: every line), to stdout."""
    with open(path, "ab") as log:
        for line in iter(pipe.readline, b""):
            log.write(line)
            log.flush()
            text = line.decode(errors="replace")
            if live is None or live.search(text):
                sys.stdout.write(prefix + text)
                sys.stdout.flush()


class Server:
    """One llama-server process: start, readiness, identity (``props.json``), stop."""

    def __init__(self, argv, working, *, host=HOST, port=PORT, tag=""):
        self.tag = tag  # file-name suffix of a second server in the same directory
        self.argv, self.working = list(argv), Path(working)
        self.root = f"http://{host}:{port}"
        self.process = None
        self.record = {"argv": self.argv}

    def start(self, timeout=READY_TIMEOUT):
        self.working.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(json.dumps({"event": "llama_server_start", "argv": self.argv}), flush=True)
        self.process = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        threading.Thread(target=pump, args=(self.process.stdout, self.working / f"llama-server{self.tag}.log"),
                         kwargs={"live": LIVE_SERVER, "prefix": "[llama-server] "}, daemon=True).start()
        with httpx.Client(trust_env=False, timeout=10) as client:
            while time.time() - started < timeout:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server exited {self.process.returncode} while loading; "
                                       f"see {self.working / f'llama-server{self.tag}.log'}")
                try:
                    if client.get(self.root + "/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f"llama-server not ready after {timeout} s")
            props = client.get(self.root + "/props").json()
        self.record.update(load_seconds=round(time.time() - started, 1), gpu_memory_after_load=gpu_memory(),
                           props={k: props.get(k) for k in ("total_slots", "model_alias", "model_path", "modalities",
                                                            "build_info", "default_generation_settings")},
                           chat_template_chars=len(props.get("chat_template") or ""))
        (self.working / f"props{self.tag}.json").write_text(json.dumps(props, indent=2, default=str))
        (self.working / f"server{self.tag}.json").write_text(json.dumps(self.record, indent=2, default=str))
        print(json.dumps({"event": "llama_server_ready", "load_seconds": self.record["load_seconds"],
                          "gpu_memory": self.record["gpu_memory_after_load"]}), flush=True)
        return self.record

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return {"stopped": True, "already_exited": True}
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        return {"stopped": True, "returncode": self.process.returncode}


class SlotCapacity:
    """Fixed per-slot windows: ``slots`` requests of up to ``window`` tokens each, nothing else."""

    units = "tokens"

    def __init__(self, props, study):
        self.slots, self.window = int(study["generation_slots"]), int(study["windows"][0])
        total = props.get("total_slots")
        slot_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        if total != self.slots or slot_ctx is None or int(slot_ctx) < self.window:
            raise RuntimeError(f"llama-server has {total} slots of {slot_ctx} tokens; the study needs "
                               f"{self.slots} of {self.window}")
        self.capacity = self.slots * self.window
        self.layout = {"rule": "llama-server -np slots with a fixed per-slot context; slot gate only",
                       "slots": total, "slot_context": int(slot_ctx), "units": self.units}

    def choice(self):
        return {"window": self.window, "capacity": self.capacity, "units": self.units, "rule": self.layout["rule"],
                "slots": self.slots, "slot_context": self.layout["slot_context"]}

    @classmethod
    def load(cls, server_dir, study):
        return cls(json.loads((Path(server_dir) / "props.json").read_text()), study)


# --- serve check -------------------------------------------------------------------------------

WORDS = ("amber basalt cedar delta ember fjord granite harbor island juniper kestrel lagoon meadow nectar "
         "orchard prairie quartz ravine summit tundra upland valley willow xenon yarrow zephyr").split()


def filler(tokens, seed, words_per_token=0.75):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(int(tokens * words_per_token)))


async def tokens_of(client, root, text):
    response = await client.post(root + "/tokenize", json={"content": text, "add_special": False})
    response.raise_for_status()
    return len(response.json()["tokens"])


async def calibrated_filler(client, root, tokens, seed):
    probe = filler(2000, seed)
    ratio = await tokens_of(client, root, probe) / 2000
    return filler(int(tokens / ratio), seed)


def overlap_rate(streams):
    """Aggregate decode tokens/s while every stream is decoding: ``streams`` holds each stream's
    token arrival times (monotonic seconds). None when the streams never all decode at once."""
    if not streams or any(len(times) < 2 for times in streams):
        return None
    start, end = max(times[0] for times in streams), min(times[-1] for times in streams)
    if end - start <= 0.5:
        return None
    tokens = sum(sum(start < t <= end for t in times) for times in streams)
    return tokens / (end - start)


async def stream_completion(client, root, prompt, n_predict):
    """One streamed raw completion: (token arrival times, final event)."""
    times, final = [], {}
    body = {"prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "cache_prompt": True, "stream": True,
            "temperature": 0.0, "samplers": ["top_k"], "top_k": 1}
    async with client.stream("POST", root + "/completion", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("stop"):
                final = event
                break
            times.extend([time.monotonic()] * max(1, len(event.get("tokens") or [0])))
    return times, final


async def decode_point(client, root, *, depth, streams, output_tokens, seed):
    """``streams`` distinct prompts of ``depth`` tokens: prefilled together (1 token each), then decoded
    together from the slots' caches (``output_tokens`` each, EOS ignored). The aggregate is measured
    over the window in which every stream is decoding; ``prompt_n`` of the decode pass shows how much
    of each prompt the slot had to recompute (hybrid KDA state may not roll back)."""
    prompts = [await calibrated_filler(client, root, depth, seed * 100 + i) for i in range(streams)]

    async def prefill(prompt):
        response = await client.post(root + "/completion", json={
            "prompt": prompt, "n_predict": 1, "cache_prompt": True, "temperature": 0.0})
        response.raise_for_status()
        return response.json().get("timings") or {}

    started = time.monotonic()
    prefills = await asyncio.gather(*(prefill(p) for p in prompts))
    prefill_wall = time.monotonic() - started
    started = time.monotonic()
    results = await asyncio.gather(*(stream_completion(client, root, p, output_tokens) for p in prompts))
    wall = time.monotonic() - started
    timings = [final.get("timings") or {} for _, final in results]
    per_stream = [t.get("predicted_per_second") or 0.0 for t in timings]
    aggregate = overlap_rate([times for times, _ in results])
    return {
        "depth": depth, "streams": streams, "output_tokens": output_tokens,
        "prefill_wall_seconds": round(prefill_wall, 2),
        "prefill_tokens_per_second_aggregate": round(sum(t.get("prompt_n") or 0 for t in prefills) / prefill_wall, 1),
        "decode_wall_seconds": round(wall, 2),
        "decode_pass_recomputed_prompt_tokens": [t.get("prompt_n") for t in timings],
        "decode_tokens_per_second_per_stream": [round(v, 2) for v in per_stream],
        "decode_tokens_per_second_aggregate": round(aggregate, 2) if aggregate is not None else None,
        "decode_tokens_per_second_aggregate_wall": round(sum(len(t) for t, _ in results) / wall, 2),
        "predicted": [t.get("predicted_n") for t in timings],
        "nan_or_empty": sum(not (final.get("content") or "").strip() and not times for times, final in results),
    }


def python_tool():
    try:
        from inference.agent.tool_agent import _PYTHON_TOOL_DESCRIPTION as description
    except Exception:  # noqa: BLE001 - the check still runs with a short description
        description = "Run Python code in a sandbox and return its stdout."
    return {"type": "function", "function": {
        "name": "python", "description": description,
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to run."}},
                       "required": ["code"]}}}


def chat_payload(model, messages, *, template_kwargs, sampling, max_tokens, tools=None):
    payload = {"model": model, "messages": messages, "stream": False, "max_tokens": max_tokens,
               "temperature": sampling["temperature"], "top_p": sampling["top_p"],
               "chat_template_kwargs": dict(template_kwargs)}
    if tools:
        payload["tools"], payload["tool_choice"] = tools, "auto"
    return payload


async def tool_round_trip(client, base, model, *, template_kwargs, sampling, system_tokens=6000, max_tokens=8192):
    """Two turns in the Duck format: a python tool call, its result, then an answer that uses it."""
    root = base.removesuffix("/v1")
    background = await calibrated_filler(client, root, system_tokens, 7)
    system = ("You solve puzzles with a python tool. Background notes (ignore them): " + background)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Use the python tool to compute 123456789 * 987654321, then tell me "
                                            "the result. Call the tool; do not compute it yourself."}]
    tools = [python_tool()]
    result = {"system_tokens": system_tokens}
    first = await client.post(base + "/chat/completions", json=chat_payload(
        model, messages, template_kwargs=template_kwargs, sampling=sampling, max_tokens=max_tokens, tools=tools))
    result["first_status"] = first.status_code
    if not first.is_success:
        result["first_error"] = first.text[:2000]
        return {**result, "passed": False}
    data = first.json()
    message = data["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    result.update(first_finish=data["choices"][0].get("finish_reason"), first_usage=data.get("usage"),
                  first_timings=data.get("timings"), reasoning_chars=len(message.get("reasoning_content") or ""),
                  content_head=(message.get("content") or "")[:300], tool_calls=calls)
    try:
        call = calls[0]
        arguments = json.loads(call["function"]["arguments"])
        ok_call = call["function"]["name"] == "python" and isinstance(arguments.get("code"), str)
    except (IndexError, KeyError, TypeError, ValueError):
        ok_call = False
    result["tool_call_ok"] = ok_call
    if not ok_call:
        return {**result, "passed": False}
    messages += [{"role": "assistant", "content": message.get("content") or "",
                  "reasoning_content": message.get("reasoning_content") or "", "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0].get("id") or "call_0", "content": "121932631112635269\n"}]
    second = await client.post(base + "/chat/completions", json=chat_payload(
        model, messages, template_kwargs=template_kwargs, sampling=sampling, max_tokens=max_tokens, tools=tools))
    result["second_status"] = second.status_code
    if not second.is_success:
        result["second_error"] = second.text[:2000]
        return {**result, "passed": False}
    data = second.json()
    text = data["choices"][0]["message"].get("content") or ""
    timings = data.get("timings") or {}
    prompt = (data.get("usage") or {}).get("prompt_tokens") or 0
    result.update(second_finish=data["choices"][0].get("finish_reason"), second_usage=data.get("usage"),
                  second_timings=timings, answer_head=text[:300],
                  answer_uses_result="121932631112635269" in text.replace(",", "").replace(" ", ""),
                  # prompt_n = tokens the server actually processed; the rest came from the slot's cache
                  second_turn_reused_fraction=round(1 - (timings.get("prompt_n") or prompt) / prompt, 3) if prompt else None)
    result["passed"] = bool(ok_call and result["answer_uses_result"])
    return result


RECALL = [("slot-A", 48213), ("slot-B", 90377), ("slot-C", 15562), ("slot-D", 73041)]


async def recall_check(client, base, model, *, template_kwargs, depth=12000, max_tokens=2048):
    """A number hidden mid-context, asked back greedily: each prompt alone, then all four at once."""
    root = base.removesuffix("/v1")

    async def prompt(i, label, code):
        text = await calibrated_filler(client, root, depth, 1000 + i)
        middle = len(text) // 2
        cut = text.rfind(" ", 0, middle)
        return (text[:cut] + f" The access code for {label} is {code}. " + text[cut:]
                + f"\n\nWhat is the access code for {label}? Reply with the number only.")

    prompts = [await prompt(i, label, code) for i, (label, code) in enumerate(RECALL)]

    async def ask(text):
        payload = {"model": model, "messages": [{"role": "user", "content": text}], "stream": False,
                   "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1, "cache_prompt": False,
                   "chat_template_kwargs": dict(template_kwargs)}
        response = await client.post(base + "/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        # The answer, or the reasoning when the reply ran out of tokens while thinking: a model that
        # computes on NaN or cross-sequence garbage names no five-digit code from the middle of 12K tokens.
        return (message.get("content") or "", message.get("reasoning_content") or "",
                {"finish": data["choices"][0].get("finish_reason"),
                 "completion_tokens": (data.get("usage") or {}).get("completion_tokens")})

    solo = [await ask(p) for p in prompts]
    together = await asyncio.gather(*(ask(p) for p in prompts))
    rows = []
    for (label, code), (a, ra, ia), (b, rb, ib) in zip(RECALL, solo, together, strict=True):
        rows.append({"label": label, "code": code, "solo": a[:120], "concurrent": b[:120],
                     "solo_ok": str(code) in a or (not a and str(code) in ra),
                     "concurrent_ok": str(code) in b or (not b and str(code) in rb), "identical": a == b,
                     "answered": [bool(a), bool(b)], "solo_reply": ia, "concurrent_reply": ib,
                     "concurrent_reasoning_tail": rb[-300:]})
    return {"depth": depth, "rows": rows, "passed": all(r["solo_ok"] and r["concurrent_ok"] for r in rows),
            "identical": sum(r["identical"] for r in rows)}


async def vision_check(client, base, model, *, template_kwargs, sampling):
    import base64
    import io
    from PIL import Image
    board = Image.new('RGB', (256, 256), 'red')
    board.paste('blue', (0, 128, 256, 256))
    stream = io.BytesIO()
    board.save(stream, format='PNG')
    data = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Name the two main colors in this image. Answer briefly in English.'},
        {'type': 'image_url', 'image_url': {'url': data}}]}]
    response = await client.post(base + '/chat/completions', json=chat_payload(
        model, messages, template_kwargs=template_kwargs, sampling=sampling, max_tokens=128))
    response.raise_for_status()
    text = response.json()['choices'][0]['message'].get('content') or ''
    return {'passed': 'red' in text.lower() and 'blue' in text.lower(), 'reply': text,
            'width': 256, 'height': 256, 'usage': response.json().get('usage')}


async def serve_check(base, model, *, template_kwargs, sampling, output, quick_template_kwargs=None, tag="",
                      tool_check=True, vision=False,
                      depths=(8192, 32768), streams=(1, 2, 4), output_tokens=256, recall_depth=12000):
    output = Path(output)
    quick = dict(quick_template_kwargs or template_kwargs)
    report = {"model": model, "template_kwargs": template_kwargs, "quick_template_kwargs": quick,
              "sampling": sampling, "started_epoch": time.time()}

    def save():
        (output / f"serve-check{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    async with httpx.AsyncClient(trust_env=False, timeout=1800) as client:
        jobs = [("tool_round_trip", lambda: tool_round_trip(client, base, model, template_kwargs=template_kwargs,
                                                            sampling=sampling)),
                ("recall", lambda: recall_check(client, base, model, template_kwargs=quick, depth=recall_depth))]
        if not tool_check:
            report["tool_round_trip"] = {"passed": None, "skipped": True}
            jobs = jobs[1:]
        if vision:
            jobs.append(('vision', lambda: vision_check(client, base, model, template_kwargs=quick, sampling=sampling)))
        for name, job in jobs:
            started = time.monotonic()
            try:
                report[name] = await job()
            except Exception as exc:  # noqa: BLE001 - every part of the check is recorded
                report[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            report[name]["seconds"] = round(time.monotonic() - started, 1)
            save()
        root = base.removesuffix("/v1")
        report["decode"] = []
        for depth in depths:
            for count in streams:
                try:
                    point = await decode_point(client, root, depth=depth, streams=count,
                                               output_tokens=output_tokens, seed=depth + count)
                except Exception as exc:  # noqa: BLE001
                    point = {"depth": depth, "streams": count, "error": f"{type(exc).__name__}: {exc}"}
                report["decode"].append(point)
                save()
    report["gpu_memory_after_check"] = gpu_memory()
    report["passed"] = bool((report["tool_round_trip"].get("passed") or not tool_check)
                            and report["recall"].get("passed")
                            and (not vision or report['vision'].get('passed')))
    report["seconds"] = round(time.time() - report["started_epoch"], 1)
    save()
    return report


def progress_line(trial_dir, metrics_text, start):
    """One live line: gameplay seconds, levels cleared, completion tokens and tokens/s since ``start``
    (``start``: the llama-server predicted-token counter and epoch when the trial began)."""
    from arc_infra.metrics import parse_metrics

    progress = {}
    path = Path(trial_dir) / "progress.json"
    if path.is_file():
        try:
            progress = json.loads(path.read_text())
        except ValueError:
            progress = {}
    values = parse_metrics(metrics_text) if metrics_text else {}
    tokens = values.get("llamacpp:tokens_predicted_total")
    seconds = time.time() - start["epoch"]
    line = {"event": "probe_progress", "gameplay_seconds": progress.get("elapsed_seconds"),
            "levels": progress.get("levels"), "active": len(progress.get("active") or []),
            "running": values.get("llamacpp:requests_processing"), "waiting": values.get("llamacpp:requests_deferred")}
    if tokens is not None and start.get("tokens") is not None:
        line["completion_tokens"] = int(tokens - start["tokens"])
        line["tokens_per_second"] = round((tokens - start["tokens"]) / max(seconds, 1), 1)
    return line


def decode_table(report):
    """Markdown rows: depth, streams, per-stream and aggregate decode tokens/s."""
    lines = ["| Depth | Streams | Per stream tok/s | Aggregate tok/s | Prefill tok/s (all streams) | Recomputed prompt tokens |",
             "| ---: | ---: | --- | ---: | ---: | --- |"]
    for p in report.get("decode", []):
        if "error" in p:
            lines.append(f"| {p['depth']} | {p['streams']} | error: {p['error'][:60]} | | | |")
            continue
        aggregate = p["decode_tokens_per_second_aggregate"] or p["decode_tokens_per_second_aggregate_wall"]
        lines.append(f"| {p['depth']} | {p['streams']} | {', '.join(f'{v:.1f}' for v in p['decode_tokens_per_second_per_stream'])} "
                     f"| {aggregate:.1f} | {p['prefill_tokens_per_second_aggregate']:.0f} "
                     f"| {', '.join(str(v) for v in p['decode_pass_recomputed_prompt_tokens'])} |")
    return "\n".join(lines)
