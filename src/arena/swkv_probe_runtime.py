"""Sliding-window measurement probe, worker side (8 October; copied to ``vllm/arc_swkv_probe.py`` by
``arena.swkv_probe_patch``). Measurement only: launcher test ``tp-swkv-kl``, ``--enforce-eager`` servers.

Question: after a context drain, can the engine drop the dropped half's KV blocks and keep every cached state of the
kept half instead of re-reading it? This probe emulates that inside one ordinary request over the *undrained* token
sequence, with two switches set through ``collective_rpc("arc_swkv_probe", json)`` (both off at start):

- ``cut`` + ``ranges``: QSA selection rows at request-relative position >= ``cut`` cannot select the compressed-key
  groups of the token ranges (whole 4-token groups; scores are sums of ReLUs, so a 0.0 score ranks last). Everything
  else is untouched: the recurrent (GDN) states, the kept tokens' keys and values and the RoPE positions are those of
  the full sequence, which is exactly what a block drop with a position offset would leave. ``count_only`` counts
  the selections in the ranges without masking (how much sparse attention the full context spends there).
- ``shift``: every QSA layer's RoPE positions (attention and indexer) + ``shift`` (a pure translation of the request).

Counters (device tensors, read by ``status``): selection rows at or after the cut, their selected tokens, and how many
of those lie in the ranges (must be 0 when masking).
"""

import inspect
import json

try:
    import torch
except ImportError:  # the source edit is tested without torch
    torch = None

MASK_ANCHOR = "        blocks = blocks_buffer[: row_end - row_start]\n"
MASK_INSERT = "        _arc_swkv_mask(logits, query_positions[row_slice])\n"
CHECK_ANCHOR = "            out[row_slice],\n        )\n"
CHECK_INSERT = "        _arc_swkv_check(out[row_slice], query_positions[row_slice])\n"
MAX_POSITION = 4 * 262144  # the MRoPE cos/sin table (4 x max_position_embeddings)

_STATE = {"cut": None, "ranges": [], "count_only": False, "shift": 0, "installed": None, "select_calls": 0,
          "forward_calls": 0, "counters": None, "rpc": False}


def patched_select_source(source):
    """``qsa_select_paged_tokens`` with the mask after scoring and the check after expansion."""
    for anchor in (MASK_ANCHOR, CHECK_ANCHOR):
        if source.count(anchor) != 1:
            raise RuntimeError("swkv probe: the pinned QSA selection changed (anchor count)")
    source = source.replace(MASK_ANCHOR, MASK_INSERT + MASK_ANCHOR)
    return source.replace(CHECK_ANCHOR, CHECK_ANCHOR + CHECK_INSERT)


def _counters(device):
    if _STATE["counters"] is None:
        _STATE["counters"] = torch.zeros(3, dtype=torch.int64, device=device)  # rows, selected, in ranges
    return _STATE["counters"]


def _mask(logits, positions):
    state = _STATE
    state["select_calls"] += 1
    if state["cut"] is None or state["count_only"]:
        return
    rows = (positions >= state["cut"]).unsqueeze(1)
    for start, end in state["ranges"]:
        g0, g1 = start // 4, min(end // 4, logits.shape[1])
        if g1 > g0:
            logits[:, g0:g1].masked_fill_(rows, 0.0)


def _check(out, positions):
    state = _STATE
    if state["cut"] is None:
        return
    rows = positions >= state["cut"]
    counters = _counters(out.device)
    picked = out[rows]
    counters[0] += rows.sum()
    counters[1] += (picked >= 0).sum()
    for start, end in state["ranges"]:
        counters[2] += ((picked >= start) & (picked < end)).sum()


def _ensure():
    """Wrap the selection and the QSA layer forward once (in the worker, after the model is built)."""
    if _STATE["installed"] is not None:
        return
    from vllm.models.qwen3_8_flash_next.nvidia import qsa as qsa_layer
    from vllm.models.qwen3_8_flash_next.nvidia.ops import qsa as ops

    source = patched_select_source(inspect.getsource(ops.qsa_select_paged_tokens))
    ops._arc_swkv_mask, ops._arc_swkv_check = _mask, _check
    namespace = {}
    exec(compile(source, ops.__file__ + ":arc_swkv", "exec"), ops.__dict__, namespace)  # noqa: S102
    ops.qsa_select_paged_tokens = namespace["qsa_select_paged_tokens"]

    cls = qsa_layer.Qwen3_8FlashNextQSAAttention
    original = cls.forward

    def forward(self, positions, hidden_states):
        _STATE["forward_calls"] += 1
        if _STATE["shift"]:
            positions = positions + _STATE["shift"]
        return original(self, positions=positions, hidden_states=hidden_states)

    cls.forward = forward
    _STATE["installed"] = {"select": True, "forward": True}


def arc_swkv_probe(self, command):
    """RPC: ``{"op": "status" | "clear" | "set" (cut, ranges, count_only, shift)}``."""
    with torch.inference_mode():
        return _rpc(json.loads(command) if isinstance(command, str) else dict(command))


def _rpc(cmd):
    state = _STATE
    op = cmd.get("op")
    try:
        _ensure()
    except Exception as exc:  # noqa: BLE001 - reported to the client, which stops
        return {"ok": False, "error": f"install: {type(exc).__name__}: {str(exc)[:300]}"}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    reply = {"ok": True, "op": op}
    if op in ("set", "clear"):
        counters = state["counters"]
        reply["before"] = None if counters is None else [int(v) for v in counters.tolist()]
        if counters is not None:
            counters.zero_()
        cut = cmd.get("cut") if op == "set" else None
        ranges = [[int(a), int(b)] for a, b in (cmd.get("ranges") or [])] if op == "set" else []
        shift = int(cmd.get("shift") or 0) if op == "set" else 0
        if any(a % 4 or b % 4 or not 0 <= a <= b for a, b in ranges) or (ranges and cut is None):
            return {"ok": False, "error": "ranges are [start, end) token ranges on 4-token groups, with a cut"}
        if not 0 <= shift < MAX_POSITION:
            return {"ok": False, "error": "shift outside the RoPE table"}
        state.update(cut=None if cut is None else int(cut), ranges=ranges, shift=shift,
                     count_only=bool(cmd.get("count_only")) if op == "set" else False)
    elif op != "status":
        return {"ok": False, "error": f"unknown op {op!r}"}
    counters = state["counters"]
    reply.update(cut=state["cut"], ranges=state["ranges"], count_only=state["count_only"], shift=state["shift"],
                 select_calls=state["select_calls"], forward_calls=state["forward_calls"],
                 counters=None if counters is None else [int(v) for v in counters.tolist()])
    return reply


def _install_rpc():
    if _STATE["rpc"]:
        return
    # collective_rpc resolves a string method on the worker wrapper (as arc_expert_probe).
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    WorkerWrapperBase.arc_swkv_probe = arc_swkv_probe
    _STATE["rpc"] = True
    print("ARC_SWKV_PROBE " + json.dumps({"rpc": True}), flush=True)


def install(cache_cls):
    """Called at the end of ``common/qsa_cache.py``: the RPC is registered when the worker binds the QSA caches (the
    worker module is fully imported by then; an import here could be circular)."""
    bind = cache_cls.bind_kv_cache

    def bind_kv_cache(self, kv_cache):
        try:
            _install_rpc()
        except Exception as exc:  # noqa: BLE001 - the client's first RPC then fails loudly
            print("ARC_SWKV_PROBE " + json.dumps({"rpc": False, "error": str(exc)[:200]}), flush=True)
        return bind(self, kv_cache)

    cache_cls.bind_kv_cache = bind_kv_cache
