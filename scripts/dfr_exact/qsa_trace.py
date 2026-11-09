"""QSA selection trace (KV-offload locality, throughput arm 18 Oct): installed as vllm/arc_qsa_trace.py and hooked
onto QSAIndexer.forward; needs --enforce-eager (graph replay skips Python). Once ARC_QSA_TRACE_START seconds have
passed since the first call and a decode batch holds >= ARC_QSA_TRACE_MIN_BATCH requests, every step records, for
the first ARC_QSA_TRACE_ROWS requests and the layers in ARC_QSA_TRACE_LAYERS, the selected 4-token groups of the
request's first query row; ARC_QSA_TRACE_STEPS steps per layer, written as compressed chunks to ARC_QSA_TRACE_DIR."""
import os
import time

import numpy as np
import torch

DIR = os.environ.get("ARC_QSA_TRACE_DIR")
START = float(os.environ.get("ARC_QSA_TRACE_START", "900"))
STEPS = int(os.environ.get("ARC_QSA_TRACE_STEPS", "2000"))
ROWS = int(os.environ.get("ARC_QSA_TRACE_ROWS", "8"))
MIN_BATCH = int(os.environ.get("ARC_QSA_TRACE_MIN_BATCH", "16"))
LAYERS = {int(x) for x in os.environ.get("ARC_QSA_TRACE_LAYERS", "3,23,47").split(",") if x}
CHUNK = 200
_state = {"t0": None, "steps": {}, "meta": [], "groups": [], "chunk": 0, "error": False}


def _flush():
    if not _state["meta"]:
        return
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, f"qsa-trace-{os.getpid()}-{_state['chunk']:04d}.npz")
    np.savez_compressed(path, meta=np.asarray(_state["meta"], dtype=np.int64),
                        groups=np.stack(_state["groups"]).astype(np.int32))
    _state["chunk"] += 1
    _state["meta"].clear()
    _state["groups"].clear()


def _record(indexer, result):
    layer = indexer.layer_id
    if layer not in LAYERS or _state["steps"].get(layer, 0) >= STEPS:
        return
    now = time.time()
    if _state["t0"] is None:
        _state["t0"] = now
    if now - _state["t0"] < START:
        return
    md = indexer._metadata()
    if md is None:
        return
    comp = md[1]
    n = int(comp.num_actual_tokens)
    seq_lens = comp.seq_lens.cpu()
    reqs = int(seq_lens.numel())
    qsl = comp.query_start_loc[: reqs + 1].cpu()
    if reqs < MIN_BATCH or n > 8 * reqs or int((qsl[1:] - qsl[:-1]).max()) > 8:  # decode / verify steps only
        return
    rows = qsl[: min(ROWS, reqs)].tolist()
    sel = result[rows].cpu().numpy()
    first_block = comp.block_table[: len(rows), 0].cpu().tolist()
    pos = comp.logical_positions[rows].cpu().tolist()
    step = _state["steps"].get(layer, 0)
    for r in range(len(rows)):
        tokens = sel[r][sel[r] >= 0]
        g = np.unique(tokens // indexer.compress_ratio)
        row = np.full(sel.shape[1] // indexer.compress_ratio + 2, -1, dtype=np.int32)
        row[: min(len(g), len(row))] = g[: len(row)]
        _state["meta"].append([layer, step, r, int(seq_lens[r]), int(pos[r]), int(first_block[r]), int(now * 1000)])
        _state["groups"].append(row)
    _state["steps"][layer] = step + 1
    if len(_state["meta"]) >= CHUNK * ROWS * len(LAYERS):
        _flush()
    if all(_state["steps"].get(x, 0) >= STEPS for x in LAYERS):
        _flush()


def install(cls):
    if not DIR:
        return
    original = cls.forward

    def forward(self, hidden_states, positions, out=None):
        result = original(self, hidden_states, positions, out)
        if not _state["error"] and not torch.cuda.is_current_stream_capturing():
            try:
                _record(self, result)
            except Exception as exc:  # the trace must never take the server down
                _state["error"] = True
                print(f"ARC_QSA_TRACE disabled: {type(exc).__name__}: {exc}", flush=True)
        return result

    cls.forward = forward
    print(f"ARC_QSA_TRACE installed: dir={DIR} start={START} steps={STEPS} rows={ROWS} layers={sorted(LAYERS)}",
          flush=True)
