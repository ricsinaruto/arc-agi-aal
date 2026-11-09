"""Expert-usage probe, worker side (28 September; copied to ``vllm/arc_expert_probe.py`` by the overlay).

The patched ``MoERunner._forward_impl`` hands every MoE layer's router logits to ``on_router_logits``
before routing. The probe

- adds a per-(layer, expert) additive mask (0, or ``MASK_VALUE`` for a pruned expert), so the router's
  top-k and its renormalized weights see only the kept experts;
- records, while switched on, per-(layer, expert) routing counts (top-k selections), gate-weight mass
  (the renormalized top-k weights) and full softmax mass, and the tokens seen per layer.

Graph-safe: every call runs the same kernels on persistent device buffers; the switch and the mask are
device tensors changed in place between passes (``collective_rpc("arc_expert_probe", json)``), so a
CUDA graph captured earlier replays with the current values. No host synchronization in the hot path.
"""

import faulthandler
import json
import sys

import torch

LAYERS = 64          # upper bound on the target model's layer ids (Flash: 48 MoE layers)
EXPERTS = 512        # routed experts per layer (Flash); any other router width is left untouched
TOP_K = 10           # experts per token (Flash); the recorded weights are renormalized over the top-k
MASK_VALUE = -1.0e4  # far below any router logit: a masked expert's softmax weight is exactly 0

_STATE = None


class _State:
    def __init__(self, device):
        f64 = {"dtype": torch.float64, "device": device}
        self.count = torch.zeros(LAYERS, EXPERTS, **f64)
        self.weight = torch.zeros(LAYERS, EXPERTS, **f64)
        self.prob = torch.zeros(LAYERS, EXPERTS, **f64)
        self.tokens = torch.zeros(LAYERS, **f64)
        self.record = torch.zeros((), **f64)
        self.mask = torch.zeros(LAYERS, EXPERTS, dtype=torch.float32, device=device)
        self.layers = set()
        self.calls = 0
        self.mask_label = None
        self.pruned = 0


def _install_rpc():
    # collective_rpc resolves a string method on the worker wrapper (MultiprocExecutor: getattr(self.worker, m)).
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    if getattr(WorkerWrapperBase, "arc_expert_probe", None) is not arc_expert_probe:
        WorkerWrapperBase.arc_expert_probe = arc_expert_probe


def on_router_logits(runner, logits):
    global _STATE
    if logits.dim() != 2 or logits.shape[-1] != EXPERTS:
        return logits
    layer = runner.__dict__.get("_arc_probe_layer")
    if layer is None:
        layer = int(runner.layer_id)
        runner.__dict__["_arc_probe_layer"] = layer
    if not 0 <= layer < LAYERS:
        return logits
    if _STATE is None:
        _STATE = _State(logits.device)
        _install_rpc()
        # Stall watchdog (v1 hung silently in engine init): every thread's Python stack goes to the server
        # log every 240 s until the first probe RPC shows the engine serving.
        faulthandler.dump_traceback_later(240, repeat=True, file=sys.stderr)
    state = _STATE
    state.layers.add(layer)
    state.calls += 1
    masked = logits + state.mask[layer].to(logits.dtype)
    if logits.is_cuda and torch.cuda.is_current_stream_capturing():
        # v1 (28 Sept) stalled right after graph capture with the statistics kernels inside the graphs. v2
        # captures only the mask (an in-place-updated buffer); statistics come from eager steps. Prefill
        # chunks above the largest capture size (32 on this profile) always run eagerly; smaller tails do not
        # count (< 0.1% of the tokens here), nor do decode steps (a max_tokens=1 pass has none).
        return masked
    probs = torch.softmax(masked.float(), dim=-1)
    top_weights, top_ids = torch.topk(probs, TOP_K, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    flat = top_ids.reshape(-1)
    record = state.record
    ones = torch.ones(flat.shape[0], dtype=torch.float64, device=flat.device)
    state.count[layer].index_add_(0, flat, ones * record)
    state.weight[layer].index_add_(0, flat, top_weights.reshape(-1).double() * record)
    state.prob[layer].add_(probs.sum(dim=0).double() * record)
    state.tokens[layer].add_(record * logits.shape[0])
    return masked


def arc_expert_probe(self, command):
    """RPC: ``{"op": "status" | "reset" | "record" (on) | "mask" (pruned, label) | "dump"}``."""
    # The buffers were created inside the forward's torch.inference_mode (inference tensors); in-place updates
    # outside it are refused (v3, 28 Sept), so the RPC runs in inference mode as well.
    with torch.inference_mode():
        return _rpc(json.loads(command) if isinstance(command, str) else dict(command))


def _rpc(cmd):
    state = _STATE
    if state is None:
        return {"ok": False, "error": "probe not initialised (no MoE forward yet)"}
    faulthandler.cancel_dump_traceback_later()
    torch.cuda.synchronize()
    op = cmd.get("op")
    if op == "reset":
        for tensor in (state.count, state.weight, state.prob, state.tokens):
            tensor.zero_()
    elif op == "record":
        state.record.fill_(1.0 if cmd.get("on") else 0.0)
    elif op == "mask":
        state.mask.zero_()
        pruned = 0
        for layer, experts in (cmd.get("pruned") or {}).items():
            experts = [int(e) for e in experts]
            if experts:
                state.mask[int(layer), torch.tensor(experts, device=state.mask.device)] = MASK_VALUE
                pruned += len(experts)
        state.mask_label, state.pruned = cmd.get("label"), pruned
    elif op not in ("status", "dump"):
        return {"ok": False, "error": f"unknown op {op!r}"}
    torch.cuda.synchronize()
    reply = {"ok": True, "op": op, "layers": sorted(state.layers), "calls": state.calls,
             "record": float(state.record.item()), "mask_label": state.mask_label, "pruned": state.pruned,
             "masked_entries": int((state.mask != 0).sum().item()),
             "tokens_per_layer": state.tokens.tolist()}
    if op == "dump":
        layers = sorted(state.layers)
        reply.update(dump_layers=layers,
                     count=state.count[layers].tolist(), weight=state.weight[layers].tolist(),
                     prob=state.prob[layers].tolist(), tokens=state.tokens[layers].tolist())
    return reply
