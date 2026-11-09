"""Replay stub server for launch preflights: an OpenAI-compatible stand-in for the Flash
vLLM server, so the exact run bundle runs every path on a CPU container.

Replies come from recorded rollouts (``attempts/*/rollout.jsonl``). A request whose messages equal a recorded request
gets that request's recorded reply (round-robin when several recordings share it, e.g. every game's first request), so
fresh games follow recorded paths to deep levels and restored starts continue as recorded. Any other request gets a
recorded reply to the same last message (the same observation), else the next recorded gameplay reply. Replies longer
than the request's ``max_tokens`` are cut there (``finish_reason`` length), as the engine would.

``launch`` follows serving_setup's owned-server contract (isolated session, identity file, readiness wait), so the
bootstrap's stop path runs unchanged; the served folder gets the source run's serving files and startup log lines (cache
layout, KV pool) so the KV gate sizes itself as on the real card. ``serve`` takes the vLLM flags the time-share second
server rewrites (``--port``, ``--kv-cache-memory-bytes``, ``--gpu-memory-utilization``) and ignores the rest.

The spec is JSON in ``ARC_PREFLIGHT_STUB``: {"corpus": [dirs], "serving": source serving dir, "speed": tokens/s per
reply (0 = instant), "kv": true}.

``kv`` (RL 22 Oct, r4 start wave / r3 thrash): KV usage follows the live context of running requests (blocks of the
source run's cache layout, one extra block per request for the recurrent state) instead of running/128; a request whose
blocks do not fit waits (``num_requests_waiting``) and logs one ``ARC_CACHE_ALLOCATION_FAILURE`` line, as the engine's
scheduler does, so a preflight leg can assert the pool is never oversubscribed.
"""

import argparse
import hashlib
import itertools
import json
import pickle
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVED = "Qwen/Qwen3.8-Flash-Next-NVFP4"
HOST, PORT = "127.0.0.1", 1234
MAX_MODEL_LEN = 262144
PLAY_RUNNING = 16  # gameplay has begun: this many requests at once (boot probes, e.g. r3_check's 6, stay below)
HEAD_MARKS = ("ARC_CACHE_LAYOUT", "GPU KV cache size", "Maximum concurrency")


def mark(message):
    return hashlib.sha1(json.dumps(message, sort_keys=True).encode()).digest()


def key_of(marks):
    return hashlib.sha1(b"".join(marks)).hexdigest()


BYTES_PER_TOKEN = 4.38  # request JSON bytes per Flash prompt token (an mg6-7 sc25 game: 445,836 bytes, 101,791 tokens)


def pseudo_ids(messages, tools=None, count=None):
    """Deterministic stand-in prompt token ids (one per ~4.4 request bytes, or ``count`` of them; ids below the Flash
    vocabulary)."""
    raw = json.dumps([messages, tools], sort_keys=True).encode()
    n = count if count is not None else max(1, round(len(raw) / BYTES_PER_TOKEN))
    seed = hashlib.sha1(raw).digest()
    step = max(1, len(raw) // max(1, n))
    return [(int.from_bytes(raw[i * step % len(raw):i * step % len(raw) + 3].ljust(3, b"\0"), "big") + seed[i % 20] * i)
            % 151000 for i in range(n)]


class Corpus:
    """Recorded rows indexed by the full request (messages) and by the last message; rows stay on disk."""

    def __init__(self, dirs):
        self.exact, self.last, self.any = {}, {}, []
        self.files = 0
        paths = [p for d in dirs for p in sorted(Path(d).glob("**/attempts/*/rollout.jsonl"))]
        cache = self.cache_file(paths)
        if cache is not None and cache.is_file():  # the index of the same files (sizes) built by an earlier preflight
            try:
                with open(cache, "rb") as file:
                    self.exact, self.last, self.any, self.files = pickle.load(file)
            except Exception:  # noqa: BLE001 - a bad cache: build afresh
                self.exact, self.last, self.any, self.files = {}, {}, [], 0
        if not self.files:
            for path in paths:
                self.files += 1
                self.add(path)
            if cache is not None:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
                    with open(tmp, "wb") as file:
                        pickle.dump((self.exact, self.last, self.any, self.files), file)
                    os.replace(tmp, cache)
                except OSError:
                    pass
        self.turn = {}
        self.lock = threading.Lock()
        self.cycle = itertools.count()
        self.stats = {"exact": 0, "last": 0, "any": 0, "cut": 0, "errors": 0}

    @staticmethod
    def cache_file(paths):
        """The index cache on the volume (/runs/preflight-corpus), keyed by every file's path and size."""
        root = "/runs/preflight-corpus" if Path("/runs").is_dir() else ""
        if not root or not paths:
            return None
        h = hashlib.sha256()
        for p in paths:
            h.update(f"{p}:{p.stat().st_size}\n".encode())
        return Path(root) / f"{h.hexdigest()[:24]}.pkl"

    def add(self, path):
        full, offset = [], 0
        with open(path, "rb") as file:
            for line in file:
                here, offset = offset, offset + len(line)
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                full = full[:int(row.get("keep") or 0)] + [mark(m) for m in row.get("messages") or []]
                if not full or not row.get("reply"):
                    continue
                ref = (str(path), here)
                self.exact.setdefault(key_of(full), []).append(ref)
                self.last.setdefault(full[-1], []).append(ref)
                if row.get("gameplay") and row["reply"].get("tool_calls"):
                    self.any.append(ref)

    @staticmethod
    def row(ref):
        with open(ref[0], "rb") as file:
            file.seek(ref[1])
            return json.loads(file.readline())

    def pick(self, messages):
        marks = [mark(m) for m in messages]
        for name, table, key in (("exact", self.exact, key_of(marks)), ("last", self.last, marks[-1] if marks else None)):
            refs = table.get(key)
            if refs:
                with self.lock:
                    i = self.turn.get(key, 0)
                    self.turn[key] = i + 1
                    self.stats[name] += 1
                return self.row(refs[i % len(refs)]), name
        if not self.any:
            raise RuntimeError("empty corpus")
        with self.lock:
            self.stats["any"] += 1
            i = next(self.cycle)
        return self.row(self.any[i % len(self.any)]), "any"


class State:
    def __init__(self, corpus, speed, kv_blocks=0, block_size=0):
        self.corpus, self.speed = corpus, float(speed)
        self.running = self.done = self.gen = self.prompt = 0
        self.asleep, self.adapters = False, {}
        self.lock = threading.Lock()
        self.kv_blocks, self.block_size, self.used, self.waiting, self.failures = kv_blocks, block_size, 0, 0, 0
        self.kv_peak = 0.0
        self.preempt_after = 0
        self.served = None  # spec "served_route": the capture directory the stub writes stand-in records to
        self.freed = threading.Condition(self.lock)

    def outage(self):
        """Inside the scripted /metrics outage (spec metrics_outage [start s, seconds] after the first gameplay
        request at PLAY_RUNNING concurrency: v5 N4's stub-start clock put the outage at the phase's end, where no read could follow it)."""
        o, began = getattr(self, "metrics_outage", None), getattr(self, "play_began", None)
        return bool(o) and began is not None and o[0] <= time.time() - began < o[0] + o[1]

    def kv(self):
        return self.used / self.kv_blocks if self.kv_blocks else min(1.0, self.running / 128)

    def admit(self, tokens):
        """Blocks for ``tokens`` of context (kv mode); waits while the pool is full. Returns the blocks held."""
        if not self.kv_blocks:
            return 0
        need = -(-int(tokens) // self.block_size) + 1
        with self.lock:
            if self.used + need > self.kv_blocks:
                self.failures += 1
                print("ARC_CACHE_ALLOCATION_FAILURE " + json.dumps({"stub": True, "epoch": time.time(), "required_blocks": need,
                                                                     "free_blocks": self.kv_blocks - self.used,
                                                                     "running": self.running}), flush=True)
                self.waiting += 1
                while self.used + need > self.kv_blocks and self.used > 0:
                    self.freed.wait(1.0)
                self.waiting -= 1
            self.used += need
            self.kv_peak = max(self.kv_peak, self.kv())
        return need

    def release(self, blocks):
        if blocks:
            with self.lock:
                self.used -= blocks
                self.freed.notify_all()


def served_record(folder, rid, tokens, model):
    """A stand-in served-route record (arc_infra.served_route_runtime's layout: int32 start, n, layers, k; int32
    tokens[n]; int16 ids[n, layers, k]) so a CPU leg exercises the route reader and the stand-in trainer's coverage
    count."""
    import struct

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{rid}-stub0000"
    n = len(tokens)
    data = struct.pack("<4i", 0, n, 1, 1) + struct.pack(f"<{n}i", *tokens) + b"\0\0" * n
    (folder / f"{name}.rt").write_bytes(data)
    (folder / f"{name}.mm.json").write_text(json.dumps({"lora": model}))
    if not (folder / "layers.json").exists():
        (folder / "layers.json").write_text(json.dumps({"layers": 1}))


def kv_layout(text):
    """(num_blocks, block_size) from the source run's ARC_CACHE_LAYOUT line (the full-attention group), else (0, 0)."""
    for line in text.splitlines():
        if "ARC_CACHE_LAYOUT " in line:
            layout = json.loads(line.split("ARC_CACHE_LAYOUT ", 1)[1])
            size = next((g["block_size"] for g in layout["groups"] if g["type"] == "FullAttentionSpec"), None)
            if size:
                return int(layout["num_blocks"]), int(size)
    return 0, 0


def chat_reply(state, payload):
    messages = payload.get("messages") or []
    if not payload.get("tools"):  # a serving probe or summary request: a short text answer
        row, how = {"reply": {"role": "assistant", "content": "ready", "reasoning": None}, "finish_reason": "stop",
                    "tokens": [101, 102]}, "text"
    else:
        row, how = state.corpus.pick(messages)
        if getattr(state, "play_began", None) is None and state.running >= PLAY_RUNNING:
            state.play_began = time.time()
    tokens, values, top = list(row.get("tokens") or []), list(row.get("logprobs") or []), list(row.get("top") or [])
    message, finish = dict(row["reply"]), row.get("finish_reason") or "stop"
    if not tokens:
        tokens = pseudo_ids([message])[:max(1, int((row.get("usage") or {}).get("completion_tokens") or 1))]
    cap = payload.get("max_tokens")
    if cap is not None and len(tokens) > int(cap):
        tokens, values, top = tokens[:int(cap)], values[:int(cap)], top[:int(cap)]
        message = {"role": "assistant", "content": None, "reasoning": (message.get("reasoning") or "")[:4 * int(cap)],
                   "tool_calls": []}
        finish = "length"
        state.corpus.stats["cut"] += 1
    recorded = (row.get("usage") or {}).get("prompt_tokens") if how == "exact" else None
    prompt_ids = pseudo_ids(messages, payload.get("tools"), recorded)  # an exact replay keeps the recorded count
    if len(prompt_ids) + len(tokens) > MAX_MODEL_LEN:
        return 400, {"error": {"message": f"This model's maximum context length is {MAX_MODEL_LEN} tokens; the request "
                                          f"has {len(prompt_ids)} prompt tokens", "type": "BadRequestError"}}
    choice = {"index": 0, "message": message, "finish_reason": finish}
    if payload.get("logprobs"):
        content = []
        for i, t in enumerate(tokens):
            alt = top[i] if i < len(top) else [[t], [values[i] if i < len(values) else -1.0]]
            content.append({"token": f"token_id:{t}", "logprob": values[i] if i < len(values) else -1.0,
                            "top_logprobs": [{"token": f"token_id:{a}", "logprob": b} for a, b in zip(*alt)]})
        choice["logprobs"] = {"content": content}
    if payload.get("return_token_ids"):
        choice["token_ids"] = tokens
    usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens), "total_tokens": len(prompt_ids) + len(tokens),
             "prompt_tokens_details": {"cached_tokens": 0}}
    named = payload.get("_request_id") or payload.get("request_id")  # vLLM: chatcmpl-<X-Request-Id or request_id>
    out = {"id": f"chatcmpl-{named}" if named else f"chatcmpl-stub-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
           "model": payload.get("model") or SERVED, "choices": [choice], "usage": usage, "stub_match": how}
    if payload.get("return_token_ids"):
        out["prompt_token_ids"] = prompt_ids
    if state.served:  # the served-route capture's file for this request (first 64 fed tokens, one layer, top-1)
        served_record(state.served, out["id"], prompt_ids[:64], payload.get("model"))
    held = state.admit(len(prompt_ids) + len(tokens))
    try:
        if state.speed > 0:
            time.sleep(min(30.0, len(tokens) / state.speed))
    finally:
        state.release(held)
    with state.lock:
        state.gen += len(tokens)
        state.prompt += len(prompt_ids)
    return 200, out


class StubServer(ThreadingHTTPServer):
    """socketserver's listen backlog is 5. The benches open up to 256 connections per client in bursts; in the gVisor
    container an overflowing accept queue stalls connects in SYN backoff or leaves a connection the stub never accepted
    (netx2, 19 Oct: backlog 5 -> requests stuck 31-189 s and ConnectErrors; 1024 -> max 1.5 s). Preflight pf2: c/d
    died on the cache probe's ReadError, a idled 35 min at the end of round 1. vLLM's uvicorn listens with 2048."""
    request_queue_size = 1024


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, code, body, kind="application/json"):
            data = (json.dumps(body) if kind == "application/json" else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/v1/models":
                ids = [SERVED] + sorted(state.adapters)
                self.send(200, {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]})
            elif path in ("/health", "/ping"):
                self.send(200, {})
            elif path == "/metrics" and state.outage():  # the blind-guard leg: /metrics unreadable for a while
                self.send(503, {"error": "metrics outage (preflight)"})
            elif path == "/metrics":
                text = (f"vllm:num_requests_running {state.running - state.waiting}\nvllm:num_requests_waiting {state.waiting}\n"
                        f"vllm:generation_tokens_total {state.gen}\nvllm:prompt_tokens_total {state.prompt}\n"
                        f'vllm:prompt_tokens_by_source_total{{source="local_compute"}} {state.prompt}\n'
                        f'vllm:prompt_tokens_by_source_total{{source="local_cache_hit"}} 0\n'
                        f"vllm:num_preemptions_total {int(bool(state.preempt_after) and state.gen > state.preempt_after)}\n"
                        f"vllm:kv_cache_usage_perc {state.kv():.3f}\n")
                drafts = state.gen // 2  # MTP-2 shaped counters (rl_seq's phase mtp row): 2 draft tokens, 0.6 accepted
                text += (f"vllm:spec_decode_num_drafts_total {drafts}\nvllm:spec_decode_num_draft_tokens_total {2 * drafts}\n"
                         f"vllm:spec_decode_num_accepted_tokens_total {int(1.2 * drafts)}\n"
                         f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="0"}} {int(0.7 * drafts)}\n')
                self.send(200, text, "text/plain")
            elif path == "/is_sleeping":
                self.send(200, {"is_sleeping": state.asleep})
            else:
                self.send(404, {"error": path})

        def do_POST(self):
            path = self.path.split("?")[0]
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self.send(400, {"error": "bad json"})
            try:
                if path == "/v1/chat/completions":
                    if state.asleep:
                        return self.send(503, {"error": "server is sleeping"})
                    if payload.get("prompt_logprobs") is not None:  # teacher-forced scoring (kv_score, kv_golden)
                        ids = pseudo_ids(payload.get("messages") or [], payload.get("tools"))
                        rows = [None] + [{str(t): {"logprob": -1.0, "rank": 1}} for t in ids[1:]]
                        return self.send(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                                            "finish_reason": "length"}], "prompt_logprobs": rows,
                                               "usage": {"prompt_tokens": len(ids), "completion_tokens": 1}})
                    if self.headers.get("X-Request-Id"):
                        payload["_request_id"] = self.headers["X-Request-Id"]
                    with state.lock:
                        state.running += 1
                    try:
                        code, body = chat_reply(state, payload)
                    finally:
                        with state.lock:
                            state.running -= 1
                            state.done += 1
                    return self.send(code, body)
                if path == "/tokenize":
                    ids = pseudo_ids(payload.get("prompt") or payload.get("messages"))
                    return self.send(200, {"tokens": ids, "count": len(ids), "max_model_len": MAX_MODEL_LEN})
                if path == "/v1/completions":
                    ids = payload.get("prompt") or []
                    rows = [None] + [{str(t): {"logprob": -1.0, "rank": 1}} for t in ids[1:]] if isinstance(ids, list) else None
                    return self.send(200, {"choices": [{"index": 0, "text": "", "prompt_logprobs": rows, "finish_reason": "length"}],
                                           "usage": {"prompt_tokens": len(ids) if isinstance(ids, list) else 0, "completion_tokens": 1}})
                if path == "/sleep":
                    state.asleep = True
                    return self.send(200, {})
                if path == "/wake_up":
                    state.asleep = False
                    return self.send(200, {})
                if path == "/v1/load_lora_adapter":
                    where = Path(str(payload.get("lora_path") or ""))
                    if not (where / "adapter_config.json").is_file():  # the engine refuses a folder without an adapter
                        return self.send(400, {"error": f"no adapter_config.json under {where}"})
                    state.adapters[payload["lora_name"]] = str(where)
                    return self.send(200, "Success: LoRA adapter loaded", "text/plain")
                if path == "/v1/unload_lora_adapter":
                    if state.adapters.pop(payload.get("lora_name"), None) is None:
                        return self.send(404, {"error": f"adapter {payload.get('lora_name')} not loaded"})
                    return self.send(200, "Success", "text/plain")
                if path in ("/reset_prefix_cache", "/collective_rpc"):
                    return self.send(200, {"success": True})
                return self.send(404, {"error": path})
            except (BrokenPipeError, ConnectionResetError):  # the client hung up (a cut try): no stub error, no reply
                state.corpus.stats["gone"] = state.corpus.stats.get("gone", 0) + 1
                return None
            except Exception as exc:  # noqa: BLE001 - a stub error is the request's 500, never the server's end
                state.corpus.stats["errors"] += 1
                return self.send(500, {"error": f"{type(exc).__name__}: {exc}"[:500]})

    return Handler


SEED_STATS = "stub-seed-stats.txt"  # the loader process's counts, beside the seed files (not a *.json seed)


def seed_loader(directory, stats, poll=0.1, rounds=None):
    """The play server's reply-id seed loader (arena.reply_ids_runtime, on vLLM) stood in: each new ``*.json`` in the
    seed dir is read and marked ``<file>.done``, so a restored attempt's first live request waits as on the card (the
    NK preflight's restore checks). A file that is not a seed file counts as a stub error. Runs in its own process
    (``seed_process``): as a thread of the stub it waited out the GIL behind ~160 busy handler threads (v15 NK
    29 Oct: 22 of 32 restored attempts timed out after 10 s), a stub artefact the card's loader does not share."""
    directory = Path(directory)
    while rounds is None or rounds > 0:
        rounds = None if rounds is None else rounds - 1
        before = dict(stats)
        for path in sorted(directory.glob("*.json")):
            done = path.with_name(path.name + ".done")
            if done.exists():
                continue
            try:
                records = json.loads(path.read_text())["records"]
                stats["seeds"] = stats.get("seeds", 0) + 1
                stats["seed_records"] = stats.get("seed_records", 0) + len(records)
            except FileNotFoundError:  # the attempt took its reply and removed it
                continue
            except (ValueError, KeyError, TypeError):
                stats["errors"] += 1
            done.write_text("")
        if stats != before:
            tmp = directory / (SEED_STATS + ".tmp")
            tmp.write_text(json.dumps(stats))
            tmp.replace(directory / SEED_STATS)
        time.sleep(poll)


def seed_process(directory):
    """The stub's seed loader in a child process (it ends with the stub's process group)."""
    seed_loader(directory, {"errors": 0})


def seed_stats(directory):
    try:
        return json.loads((Path(directory) / SEED_STATS).read_text())
    except (OSError, ValueError):
        return {}


def serve(args, spec):
    corpus = Corpus(spec["corpus"])
    head = Path(args.head) if args.head else None
    if head and head.is_file():
        print(head.read_text(), flush=True)  # the KV gate's cache layout and pool lines for this server's own log
    print(f"STUB_CORPUS files={corpus.files} requests={len(corpus.exact)} gameplay={len(corpus.any)} "
          f"kv_cache_memory_bytes={args.kv_cache_memory_bytes}", flush=True)
    blocks, size = kv_layout(head.read_text()) if (spec.get("kv") and head and head.is_file()) else (0, 0)
    if spec.get("kv") and not blocks:
        raise SystemExit("stub kv: no ARC_CACHE_LAYOUT line in the source serving log")
    state = State(corpus, spec.get("speed", 4000), blocks, size)
    state.preempt_after = int(spec.get("preempt_after") or 0)  # back-off leg: one preemption past this many tokens
    state.started = time.time()
    state.metrics_outage = [float(x) for x in spec["metrics_outage"]] if spec.get("metrics_outage") else None
    state.served = spec.get("served_route")
    server = StubServer((HOST, args.port), handler_for(state))
    server.daemon_threads = True

    seeds = Path(os.environ.get("ARC_REPLY_IDS_SEED_DIR") or "/tmp/arc-reply-ids-seed")  # as reply_ids_runtime makes it
    seeds.mkdir(parents=True, exist_ok=True)

    def merged():  # the loader process's counts (its bad files are stub errors)
        loaded = seed_stats(seeds)
        return {**corpus.stats, **{k: v for k, v in loaded.items() if k != "errors"},
                "errors": corpus.stats["errors"] + int(loaded.get("errors") or 0)}

    def logger():
        last_gen = last_prompt = 0
        while True:
            time.sleep(10)
            gen, prompt = state.gen, state.prompt
            print(f"(APIServer pid={os.getpid()}) INFO [loggers.py] Engine 000: Avg prompt throughput: "
                  f"{(prompt - last_prompt) / 10:.1f} tokens/s, Avg generation throughput: {(gen - last_gen) / 10:.1f} "
                  f"tokens/s, Running: {state.running - state.waiting} reqs, Waiting: {state.waiting} reqs, GPU KV cache usage: "
                  f"{100 * state.kv():.1f}%, Prefix cache hit rate: 0.0% STUB {json.dumps(merged())} "
                  f"KV_STUB {json.dumps({'peak': round(state.kv_peak, 4), 'failures': state.failures})}",
                  flush=True)
            last_gen, last_prompt = gen, prompt
            if state.kv_blocks and os.environ.get("ARC_VALIDATION_OUTPUT"):  # a leg's KV asserts read it from the volume
                try:
                    Path(os.environ["ARC_VALIDATION_OUTPUT"], "stub-kv.json").write_text(json.dumps({
                        "epoch": time.time(), "peak": round(state.kv_peak, 4), "failures": state.failures,
                        "kv": round(state.kv(), 4), "waiting": state.waiting, "blocks": state.kv_blocks}))
                except OSError:
                    pass

    threading.Thread(target=logger, daemon=True).start()
    import multiprocessing

    multiprocessing.Process(target=seed_process, args=(str(seeds),), daemon=True).start()
    print("INFO:     Application startup complete.", flush=True)
    server.serve_forever()


def proc_record(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    return {"pid": pid, "start_ticks": int(fields[19]), "ppid": int(fields[1]), "pgid": int(fields[2]),
            "sid": int(fields[3]), "cmdline_sha256": hashlib.sha256(cmdline).hexdigest()}


def launch(args, spec):
    """Start ``serve`` as an owned server: what ``serve-native`` leaves behind, from a real run's serving folder."""
    working, source = Path(args.working), Path(spec["serving"])
    working.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.name not in ("vllm-server-identity.json", "vllm-openai-server.log"):
            shutil.copyfile(item, working / item.name)
    real = json.loads((source / "vllm-server-identity.json").read_text())
    log_lines = (source / "vllm-openai-server.log").read_text(errors="replace").splitlines()
    head = working / "stub-head.log"
    head.write_text("\n".join(l for l in log_lines if any(m in l for m in HEAD_MARKS)) + "\n")
    argv = [sys.executable, "-m", "arena.replay_stub", "serve", "--working", str(working), "--head", str(head),
            "--port", str(PORT), "--kv-cache-memory-bytes", str(real["vllm_tuning"].get("kv_cache_memory_bytes") or 0)]
    print("VLLM_START_COMMAND", json.dumps(argv), flush=True)
    log = (working / "vllm-openai-server.log").open("w")
    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                               env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
    log.close()
    time.sleep(0.5)
    root = proc_record(process.pid)
    now = time.time()
    identity = {"schema_version": 1, "backend": "vllm",
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "host": HOST, "port": PORT,
                "pid": process.pid, "start_ticks": root["start_ticks"], "pgid": root["pgid"], "sid": root["sid"],
                "argv": argv, "argv_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":"), ensure_ascii=False)
                                                           .encode()).hexdigest(),
                "vllm_tuning": real["vllm_tuning"], "started_epoch": now, "phase": "starting", "snapshot_epoch": now,
                "ready_epoch": None, "workers": [root]}
    (working / "vllm-server-identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    import urllib.request

    deadline = time.time() + 900
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"stub exited before readiness:\n{(working / 'vllm-openai-server.log').read_text()[-3000:]}")
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=5) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(2)
    else:
        raise SystemExit("stub not ready in 900 s")
    ready = time.time()
    identity.update(phase="ready", snapshot_epoch=ready, ready_epoch=ready, workers=[proc_record(process.pid)])
    (working / "vllm-server-identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print("VLLM_SETUP_COMPLETE " + json.dumps({"mode": "replay_stub", "pid": process.pid, "ready_seconds": round(ready - now, 1)}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "serve"))
    parser.add_argument("--working", required=True)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--head", default=None)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", default=None)
    args, _ = parser.parse_known_args()
    spec = json.loads(os.environ["ARC_PREFLIGHT_STUB"])
    (launch if args.command == "launch" else serve)(args, spec)


if __name__ == "__main__":
    main()
