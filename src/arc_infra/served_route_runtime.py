"""Served-routing capture for R3 (rollout routing replay), API-server side.

The pinned vLLM has its own routed-experts return (``--enable-return-routed-experts``): the worker captures every MoE
layer's top-k ids at select_experts (CUDA-graph safe, both model runners), the scheduler keeps them per KV-cache slot
(so a prefix-cache hit carries the routing of the request that computed it), and a finished request's
``CompletionOutput.routed_experts`` holds ``[prompt + output - 1, layers, k]`` (row i = position i; the last sampled
token is never fed). The first build hooked V1's GPUModelRunner, which this server does not use (V2 model runner):
the 20 Oct check recorded nothing.

The overlay (``served_route_patch.py``) copies this file to ``vllm/arc_served_route.py`` with ``DIR`` filled in and
calls ``install(RequestState)`` at the end of ``vllm/v1/engine/output_processor.py``. When a request finishes, its rows
from its first uncached position on go to ``DIR/<engine request id>.rt`` in the record format
``arena.served_route.read`` takes (int32 [start, n, layers, k], int32 tokens[n], int16 ids[n, layers, k]); positions
before the cache hit are in the request that computed them, as before. ``<id>.mm.json`` gets the LoRA, the images
(offset, length, content hash) and the prompt length, ``<id>.json`` closes the request, and the array is dropped from
the output (the OpenAI layer does not use it). One thread writes; the serving path never raises: errors are counted in
``DIR/status.json`` and the capture switches off when free disk drops below ``MIN_FREE``.
"""

import collections
import json
import os
import queue
import shutil
import threading
import time

import numpy as np

DIR = None  # filled in by the overlay
HASH_BASE = 0x9E3779B97F4A7C15  # prefix hash (arena.served_route.prefix_hashes): H[i+1] = H[i] * B + t_i + 1 mod 2^64
MIN_FREE = 24 << 30
QUEUE = 4096  # finished requests waiting for the writer; more are dropped (counted)
FULL_ENV = "ARC_SERVED_ROUTE_FULL"
FULL_FLAG = "full.on"  # in DIR: arena.r3_offload_check sets it around its own requests (play records stay partial)

_WRITER = None
_LOCK = threading.Lock()


def prefix_hashes(tokens):
    """``H`` (uint64 [T + 1]): H[0] = 0, H[i + 1] = H[i] * B + tokens[i] + 1 mod 2^64, vectorised as
    H[i] = B^i * sum_{j < i} (tokens[j] + 1) * B^-(j + 1) (B odd, so invertible mod 2^64; numpy uint64 wraps)."""
    t = np.asarray(tokens, dtype=np.int64).astype(np.uint64) + np.uint64(1)
    with np.errstate(over="ignore"):
        up = np.cumprod(np.full(len(t), HASH_BASE, dtype=np.uint64))
        down = np.cumprod(np.full(len(t), pow(HASH_BASE, -1, 1 << 64), dtype=np.uint64))
        out = np.zeros(len(t) + 1, dtype=np.uint64)
        out[1:] = up * np.cumsum(t * down, dtype=np.uint64)
    return out


def prefix_hash(tokens):
    return int(prefix_hashes(tokens)[-1])


class _Writer:
    def __init__(self, directory):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)
        self.active = True
        self.layers = None
        self.queue = queue.Queue(maxsize=QUEUE)
        self.counts = collections.Counter()
        self.errors = []
        self.status_at = 0.0
        self.started = time.time()
        threading.Thread(target=self._loop, daemon=True, name="arc-served-route").start()

    def error(self, text):
        self.counts["errors"] += 1
        if len(self.errors) < 20:
            self.errors.append(f"{time.strftime('%H:%M:%S')} {text}"[:400])

    def put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.counts["dropped"] += 1

    def _loop(self):
        while True:
            try:
                item = self.queue.get(timeout=2)
            except queue.Empty:
                self.status()
                continue
            try:
                self.write(*item)
            except Exception as exc:  # noqa: BLE001
                self.error(f"write: {type(exc).__name__}: {exc}")
            self.status()

    def write(self, rid, start, seq, ids, meta):
        n, layers, k = ids.shape
        tokens = seq[start:]
        if start:  # identifies the context before the record, so a reader can use it for another request's window
            meta = {**meta, "prefix_hash": f"{prefix_hash(seq[:start]):016x}"}
        if self.layers is None:
            self.layers = list(range(layers))  # vLLM's capture is indexed by the decoder layer id
            with open(os.path.join(self.dir, "layers.json"), "w") as f:
                json.dump({"layers": self.layers, "k": int(k), "source": "vllm return_routed_experts"}, f)
        with open(os.path.join(self.dir, rid + ".mm.json"), "w") as f:
            json.dump(meta, f)
        head = np.array([start, n, layers, k], dtype=np.int32)
        with open(os.path.join(self.dir, rid + ".rt"), "wb") as f:
            f.write(head.tobytes())
            f.write(np.ascontiguousarray(tokens, dtype=np.int32).tobytes())
            f.write(np.ascontiguousarray(ids, dtype=np.int16).tobytes())
        tmp = os.path.join(self.dir, rid + ".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"layers": self.layers, "k": int(k), "rows": int(n), "lo": int(start), "hi": int(start + n),
                       "lora": meta.get("lora")}, f)
        os.replace(tmp, os.path.join(self.dir, rid + ".json"))
        self.counts["closed"] += 1
        self.counts["rows"] += int(n)
        self.counts["bytes"] += 16 + 4 * n + 2 * ids.size

    def status(self, force=False):
        now = time.time()
        if not force and now - self.status_at < 30:
            return
        self.status_at = now
        try:
            free = shutil.disk_usage(self.dir).free
            if free < MIN_FREE and self.active:
                self.active = False
                self.error(f"capture off: {free >> 30} GiB free")
            data = {"active": self.active, "seconds": round(now - self.started, 1), "layers": len(self.layers or []),
                    "queued": self.queue.qsize(), "free_gib": round(free / 2**30, 1), **self.counts,
                    "errors": self.errors}
            tmp = os.path.join(self.dir, "status.json.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, os.path.join(self.dir, "status.json"))
        except Exception:  # noqa: BLE001
            pass


def _writer():
    global _WRITER
    if _WRITER is None and DIR is not None:
        with _LOCK:
            if _WRITER is None:
                _WRITER = _Writer(DIR)
    return _WRITER


def _lora(request):
    if request is None:
        return None  # name, id and path: a reloaded adapter under the same name is another policy
    return "|".join(str(getattr(request, f, "")) for f in ("lora_name", "lora_int_id", "lora_path"))


def dump(state, routed):
    """Queue a finished request's routing (``routed`` [rows, layers, k]) from its first uncached position on."""
    writer = _writer()
    if writer is None or not writer.active:
        return
    writer.counts["finished"] += 1
    if routed.ndim != 3:
        writer.error(f"routed_experts shape {routed.shape}")
        return
    rows = int(routed.shape[0])
    prompt = list(state.prompt_token_ids or [])
    output = list(state.detokenizer.output_token_ids) if state.detokenizer is not None else []
    if rows not in (len(prompt) + len(output) - 1, len(prompt) + len(output)):
        writer.counts["row_mismatch"] += 1  # e.g. a step without routing: positions would shift, so nothing is kept
        if writer.counts["row_mismatch"] <= 5:
            writer.error(f"rows {rows} vs prompt {len(prompt)} + output {len(output)}")
        return
    seq = (prompt + output)[:rows]
    if len(seq) < rows:
        writer.counts["short_tokens"] += 1
    tokens = np.full(rows, -1, dtype=np.int32)
    tokens[:len(seq)] = seq
    start = min(max(int(getattr(state, "num_cached_tokens", 0) or 0), 0), rows)
    if os.environ.get(FULL_ENV) == "1" or os.path.exists(os.path.join(DIR, FULL_FLAG)):
        start = 0  # every row, cached ones too: the R3-through-offload replay check (its flag file only while it runs)
    if start >= rows:
        writer.counts["nothing_new"] += 1
        return
    meta = {"lora": _lora(getattr(state, "lora_request", None)), "mm": getattr(state, "_arc_mm", []),
            "prompt_tokens": len(prompt), "rows": rows, "cached": start}
    writer.put((state.request_id, start, tokens, routed[start:].astype(np.int16), meta))


def install(request_state):
    """Wrap ``RequestState.from_new_request`` (keep the request's images) and ``_new_completion_output`` (dump the
    finished request's routing, drop it from the output)."""
    if getattr(request_state, "_arc_served_route", False) or DIR is None:
        return
    new_request = request_state.from_new_request.__func__
    completion = request_state._new_completion_output

    def from_new_request(cls, *args, **kwargs):
        state = new_request(cls, *args, **kwargs)
        try:
            request = kwargs["request"] if "request" in kwargs else args[1]
            state._arc_mm = [[int(f.mm_position.offset), int(f.mm_position.length), str(f.identifier)]
                             for f in (getattr(request, "mm_features", None) or [])]
        except Exception as exc:  # noqa: BLE001
            writer = _writer()
            if writer is not None:
                writer.error(f"images: {type(exc).__name__}: {exc}")
        return state

    def new_completion_output(self, *args, **kwargs):
        out = completion(self, *args, **kwargs)
        routed = getattr(out, "routed_experts", None)
        if routed is not None:
            try:
                dump(self, routed)
            except Exception as exc:  # noqa: BLE001
                writer = _writer()
                if writer is not None:
                    writer.error(f"dump: {type(exc).__name__}: {exc}")
            out.routed_experts = None
        return out

    request_state.from_new_request = classmethod(from_new_request)
    request_state._new_completion_output = new_completion_output
    request_state._arc_served_route = True
    _writer()
