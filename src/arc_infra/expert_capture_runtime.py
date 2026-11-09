"""MoE-input capture, worker side (28 September; copied to ``vllm/arc_expert_capture.py`` by the overlay).

The patched ``MoERunner._apply_quant_method`` hands every MoE layer's input rows and its served routing (top-k ids and
renormalized weights, exactly what the fused experts receive) to ``on_routed``. Nothing happens until an
``arc_expert_capture`` RPC (``/collective_rpc``) starts a capture:

``{"op": "start", "dir": path, "layers": [ids] | null, "stride": s, "format": "bf16" | "fp8", "max_rows": n | null}``
    every s-th row of each selected layer (a per-layer running row counter, so every layer keeps the same token
    positions) is copied to host memory and appended to ``<dir>/L<layer>.{x,s,ids,w}`` in batches:
    ``x`` the rows (bf16, or E4M3 bytes with a float32 per-row scale in ``s``), ``ids`` int16 [n, k], ``w`` float32
    [n, k]. ``max_rows`` caps the rows kept per layer. The forward pass only copies rows to host memory; a writer
    thread appends them to disk (at most ``max_queued_bytes`` waiting; beyond that rows are dropped and counted),
    and a write error or a free-disk shortfall stops the capture instead of the server.
``{"op": "stop"}``
    flushes, writes ``<dir>/meta.json`` (rows per layer, shapes, dtypes) and switches the capture off.
``{"op": "status"}``

Rows are skipped while a CUDA graph is being captured (no replayed graph ever contains the hook's copies); prefill
runs outside graphs. The copies synchronize the stream: a capture pass is slower than serving, which is fine for a
measurement server.
"""

import json
import os
import queue
import shutil
import threading
import time

import torch

_STATE = None


class _State:
    def __init__(self):
        self.active = False
        self.dir = None
        self.layers = None
        self.stride = 1
        self.format = "bf16"
        self.max_rows = None
        self.flush_rows = 8192
        self.counter = {}
        self.buffers = {}
        self.buffered = {}
        self.accepted = {}
        self.written = {}
        self.shapes = {}
        self.calls = 0
        self.skipped_graph = 0
        self.layers_seen = set()
        self.started = None
        self.bytes = 0
        self.min_free = 8 << 30
        self.max_queued = 32 << 30
        self.queued_bytes = 0
        self.dropped_rows = 0
        self.stopped = None
        self.write_errors = []
        self.write_seconds = 0.0
        self.queue = None
        self.writer = None
        self.lock = threading.Lock()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _install_rpc():
    # collective_rpc resolves a string method on the worker wrapper (MultiprocExecutor: getattr(self.worker, m)).
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    if getattr(WorkerWrapperBase, "arc_expert_capture", None) is not arc_expert_capture:
        WorkerWrapperBase.arc_expert_capture = arc_expert_capture


def _layer_of(runner):
    layer = runner.__dict__.get("_arc_capture_layer")
    if layer is None:
        layer = int(runner.layer_id)
        runner.__dict__["_arc_capture_layer"] = layer
    return layer


def on_routed(runner, hidden_states, topk_weights, topk_ids):
    global _STATE
    if _STATE is None:
        _STATE = _State()
        _install_rpc()
    state = _STATE
    if not state.active:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        state.skipped_graph += 1
        return
    if hidden_states.dim() != 2 or topk_ids.dim() != 2:
        return
    layer = _layer_of(runner)
    state.layers_seen.add(layer)
    if state.layers is not None and layer not in state.layers:
        return
    state.calls += 1
    n = hidden_states.shape[0]
    start = state.counter.get(layer, 0)
    state.counter[layer] = start + n
    first = (-start) % state.stride
    if first >= n:
        return
    rows = slice(first, n, state.stride)
    x = hidden_states[rows]
    ids = topk_ids[rows]
    weights = topk_weights[rows]
    if state.max_rows is not None:
        room = state.max_rows - state.accepted.get(layer, 0)
        if room <= 0:
            return
        x, ids, weights = x[:room], ids[:room], weights[:room]
    if state.format == "fp8":
        scale = x.float().abs().amax(dim=1).clamp(min=1e-12) / 448.0
        q = (x.float() / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
        parts = {"x": q.view(torch.uint8).cpu(), "s": scale.cpu()}
    else:
        parts = {"x": x.contiguous().view(torch.int16).cpu()}
    parts["ids"] = ids.to(torch.int16).cpu()
    parts["w"] = weights.float().cpu()
    state.buffers.setdefault(layer, []).append(parts)
    state.buffered[layer] = state.buffered.get(layer, 0) + x.shape[0]
    state.accepted[layer] = state.accepted.get(layer, 0) + x.shape[0]
    if not state.shapes.get(layer):
        state.shapes[layer] = {"hidden": int(x.shape[1]), "topk": int(ids.shape[1]),
                               "ids_dtype": str(topk_ids.dtype), "x_dtype": str(hidden_states.dtype)}
    if state.buffered[layer] >= state.flush_rows:
        _enqueue(state, layer)


def _enqueue(state, layer):
    """Hand a layer's buffered rows to the writer thread; never blocks the forward pass (rows beyond the queue
    budget are dropped and counted)."""
    chunks = state.buffers.pop(layer, [])
    state.buffered[layer] = 0
    if not chunks:
        return
    size = sum(t.numel() * t.element_size() for c in chunks for t in c.values())
    with state.lock:
        if state.queued_bytes + size > state.max_queued or state.stopped:
            state.dropped_rows += sum(c["ids"].shape[0] for c in chunks)
            return
        state.queued_bytes += size
    state.queue.put((layer, chunks, size))


def _writer(state):
    while True:
        item = state.queue.get()
        try:
            if item is None:
                return
            layer, chunks, size = item
            if state.stopped:
                continue
            t0 = time.time()
            try:
                if shutil.disk_usage(state.dir).free < state.min_free:
                    raise OSError("free disk below the guard")
                for key in chunks[0]:
                    data = torch.cat([c[key] for c in chunks]).numpy().tobytes()
                    with open(os.path.join(state.dir, f"L{layer:02d}.{key}"), "ab") as handle:
                        handle.write(data)
                    state.bytes += len(data)
                state.written[layer] = state.written.get(layer, 0) + sum(c["ids"].shape[0] for c in chunks)
            except OSError as exc:
                # Never fill the container's disk or stall the server: stop recording (the rows written stand).
                state.stopped = f"write: {exc}"
                state.active = False
                state.write_errors.append(str(exc)[:200])
            state.write_seconds += time.time() - t0
            with state.lock:
                state.queued_bytes -= size
        finally:
            state.queue.task_done()


def _summary(state):
    return {"active": state.active, "dir": state.dir, "format": state.format, "stride": state.stride,
            "layers": None if state.layers is None else sorted(state.layers), "calls": state.calls,
            "skipped_graph": state.skipped_graph, "layers_seen": sorted(state.layers_seen),
            "rows": {str(k): v for k, v in sorted(state.written.items())}, "bytes": state.bytes,
            "processed_tokens": {str(k): v for k, v in sorted(state.counter.items())},
            "stopped": state.stopped, "dropped_rows": state.dropped_rows, "queued_bytes": state.queued_bytes,
            "write_seconds": round(state.write_seconds, 1), "write_errors": state.write_errors[:5],
            "seconds": None if state.started is None else round(time.time() - state.started, 1)}


def arc_expert_capture(self, command):
    """RPC: ``{"op": "start" | "stop" | "status", ...}`` (module docstring)."""
    global _STATE
    cmd = json.loads(command) if isinstance(command, str) else dict(command)
    if _STATE is None:
        _STATE = _State()
    state = _STATE
    op = cmd.get("op")
    if op == "start":
        if state.active:
            return {"ok": False, "error": "a capture is already running"}
        if cmd.get("format", "bf16") not in ("bf16", "fp8"):
            return {"ok": False, "error": f"unknown format {cmd.get('format')!r}"}
        _sync()
        os.makedirs(cmd["dir"], exist_ok=True)
        fresh = _State()
        fresh.dir = cmd["dir"]
        fresh.layers = None if cmd.get("layers") is None else {int(x) for x in cmd["layers"]}
        fresh.stride = max(1, int(cmd.get("stride", 1)))
        fresh.format = cmd.get("format", "bf16")
        fresh.max_rows = None if cmd.get("max_rows") is None else int(cmd["max_rows"])
        fresh.flush_rows = int(cmd.get("flush_rows", 8192))
        fresh.min_free = int(cmd.get("min_free_bytes", 8 << 30))
        fresh.max_queued = int(cmd.get("max_queued_bytes", 32 << 30))
        fresh.queue = queue.Queue()
        fresh.writer = threading.Thread(target=_writer, args=(fresh,), daemon=True, name="arc-expert-capture")
        fresh.writer.start()
        fresh.started = time.time()
        fresh.active = True
        _STATE = state = fresh
    elif op == "stop":
        _sync()
        state.active = False
        if state.queue is not None:
            for layer in list(state.buffers):
                _enqueue(state, layer)
            state.queue.put(None)
            state.writer.join(timeout=float(cmd.get("drain_seconds", 1500)))
        if state.dir:
            meta = {**_summary(state), "shapes": {str(k): v for k, v in sorted(state.shapes.items())}}
            with open(os.path.join(state.dir, "meta.json"), "w") as handle:
                json.dump(meta, handle, indent=1)
    elif op != "status":
        return {"ok": False, "error": f"unknown op {op!r}"}
    return {"ok": True, "op": op, **_summary(state)}
