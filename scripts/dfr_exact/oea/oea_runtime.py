"""Batch-aware expert routing (Opportunistic Expert Activation, arXiv 2511.02237; throughput arm, 16 Oct).

Installed as ``vllm/arc_oea.py``; ``fused_topk_router.py`` calls ``install(FusedTopKRouter)`` at import. With
``ARC_OEA_K0`` = k0 (0 < k0 < top-k) every batch of at most ``ARC_OEA_MAX_M`` rows (decode / verify / draft steps)
routes as: each token keeps its own top-k0 experts; the step's union S of all tokens' top-k0 is what the MoE loads
anyway; each token's remaining top-k - k0 slots go to its highest-probability experts inside S (outside S only when S
has fewer than top-k experts). Weights are the token's softmax probabilities of the chosen experts, renormalized as
the pinned router does. Larger batches (prefill) route as pinned. Static shapes, no host sync: CUDA-graph safe.
Lossy (a token may swap up to top-k - k0 of its experts): needs a quality read.

Union cap (18 Oct; Lynx arXiv 2411.08982, SERE ICLR'26, MoE-Spec): with ``ARC_UCAP`` = N, every such batch keeps the
N experts with the largest summed router probability over its rows; each token takes its top-k inside that set
(weights renormalized), so the MoE reads at most N experts' weights per step. Lossy.
"""

import os
import re
import sys

import torch

K0 = int(os.environ.get("ARC_OEA_K0", "0") or 0)
MAX_M = int(os.environ.get("ARC_OEA_MAX_M", "256") or 256)
UCAP = int(os.environ.get("ARC_UCAP", "0") or 0)
_CALLS = 0
PREFILL_GROUP = int(os.environ.get("ARC_UCAP_PREFILL", "0") or 0)
PLAIN = os.environ.get("ARC_UCAP_PLAIN") == "1"
_FALLS = 0
# Prune toggle (21 Oct, quality screens, eager only): ARC_PRUNE_TOGGLE=<path>; every routing call is the plain path
# (softmax -> top-k -> renormalize; q-b37plain: plain minus pinned +0.03 nat [-0.03, +0.08]) with the router logits of
# the experts listed in the file ({"<layer>": [served expert indices]}) set to -inf. No file: nothing dropped.
PRUNE_TOGGLE = os.environ.get("ARC_PRUNE_TOGGLE") or ""
_PT = {"stamp": None, "sel": {}}


def _layer_of_caller():
    frame = sys._getframe(2)
    for _ in range(8):
        if frame is None:
            break
        name = frame.f_locals.get("layer_name") or frame.f_locals.get("prefix")
        if isinstance(name, str) and name:
            m = re.search(r"layers\.(\d+)\.", name)
            return (None if "mtp" in name else int(m.group(1))) if m else None
        frame = frame.f_back
    return None


def _drop_ids(router):
    try:
        st = os.stat(PRUNE_TOGGLE)
    except FileNotFoundError:
        return None
    stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
    if _PT["stamp"] != stamp:
        import json

        with open(PRUNE_TOGGLE) as fh:
            text = fh.read().strip()
        _PT.update(stamp=stamp, sel={int(k): v for k, v in (json.loads(text) if text else {}).items()})
        print(f"ARC_PRUNE_TOGGLE {sum(len(v) for v in _PT['sel'].values())} experts over {len(_PT['sel'])} layers",
              flush=True)
    layer = getattr(router, "_arc_layer", None)
    return _PT["sel"].get(layer) if layer is not None else None


def install(cls):
    if not (K0 or UCAP or PLAIN or PRUNE_TOGGLE) or getattr(cls, "_arc_oea", False):
        return
    pinned = cls._compute_routing
    if PRUNE_TOGGLE:
        init = cls.__init__

        def __init__(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self._arc_layer = _layer_of_caller()
            print(f"ARC_PRUNE_TOGGLE router layer={self._arc_layer} experts={self.global_num_experts}", flush=True)

        cls.__init__ = __init__

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        m = router_logits.shape[0]
        if PRUNE_TOGGLE:
            p = router_logits.float()
            drop = _drop_ids(self)
            if drop:
                p = p.index_fill(1, torch.tensor(drop, device=p.device, dtype=torch.long), float("-inf"))
            p = torch.softmax(p, dim=-1)
            w, ids = p.topk(self.top_k, dim=-1)
            if self.renormalize:
                w = w / w.sum(dim=-1, keepdim=True)
            return w, ids.to(torch.int32 if indices_type is None else indices_type)
        if PLAIN:
            # Control (ARC_UCAP_PLAIN=1): the pinned routing re-done as plain softmax -> top-k -> renormalize, no cap.
            p = torch.softmax(router_logits.float(), dim=-1)
            w, ids = p.topk(self.top_k, dim=-1)
            if self.renormalize:
                w = w / w.sum(dim=-1, keepdim=True)
            return w, ids.to(torch.int32 if indices_type is None else indices_type)
        if UCAP:
            if PREFILL_GROUP and m > MAX_M and self.scoring_func == "softmax" and UCAP < router_logits.shape[-1]:
                # Quality screen only (ARC_UCAP_PREFILL=G): a prefill chunk is capped in strided groups of ~G rows,
                # standing in for a decode step's union over G rows (prompt_logprobs scoring runs as prefill).
                p = torch.softmax(router_logits.float(), dim=-1)
                groups = max(1, -(-m // PREFILL_GROUP))
                gid = torch.arange(m, device=p.device) % groups
                sums = p.new_zeros(groups, p.shape[-1]).index_add_(0, gid, p)
                keep = p.new_zeros(groups, p.shape[-1]).scatter_(1, sums.topk(UCAP, dim=-1).indices, 1.0)[gid]
                ids = torch.where(keep > 0, p, p - 2.0).topk(self.top_k, dim=-1).indices
                w = p.gather(1, ids)
                if self.renormalize:
                    w = w / w.sum(dim=-1, keepdim=True)
                return w, ids.to(torch.int32 if indices_type is None else indices_type)
            if self.scoring_func != "softmax" or not 0 < m <= MAX_M or UCAP >= router_logits.shape[-1]:
                global _FALLS
                if _FALLS < 4:
                    print(f"ARC_UCAP pinned m={m} scoring={self.scoring_func} experts={router_logits.shape[-1]}",
                          flush=True)
                _FALLS += 1
                return pinned(self, hidden_states, router_logits, indices_type, input_ids=input_ids)
            global _CALLS
            if _CALLS < 4 or torch.cuda.is_current_stream_capturing() and _CALLS < 400:
                print(f"ARC_UCAP call m={m} experts={router_logits.shape[-1]} "
                      f"capturing={torch.cuda.is_current_stream_capturing()}", flush=True)
            _CALLS += 1
            p = torch.softmax(router_logits.float(), dim=-1)
            keep = p.new_zeros(p.shape[-1]).scatter_(0, p.sum(dim=0).topk(UCAP).indices, 1.0)
            ids = torch.where(keep > 0, p, p - 2.0).topk(self.top_k, dim=-1).indices
            w = p.gather(1, ids)
            if self.renormalize:
                w = w / w.sum(dim=-1, keepdim=True)
            return w, ids.to(torch.int32 if indices_type is None else indices_type)
        if self.scoring_func != "softmax" or not 0 < m <= MAX_M or K0 >= self.top_k:
            return pinned(self, hidden_states, router_logits, indices_type, input_ids=input_ids)
        p = torch.softmax(router_logits.float(), dim=-1)
        # scatter_ with a scalar value: no host tensor (index_put of True copies one H2D, refused in capture)
        used = p.new_zeros(p.shape[-1]).scatter_(0, p.topk(K0, dim=-1).indices.reshape(-1), 1.0)
        ids = torch.where(used > 0, p, p - 2.0).topk(self.top_k, dim=-1).indices
        w = p.gather(1, ids)
        if self.renormalize:
            w = w / w.sum(dim=-1, keepdim=True)
        return w, ids.to(torch.int32 if indices_type is None else indices_type)

    cls._compute_routing = _compute_routing
    cls._arc_oea = True
    print(f"ARC_OEA k0={K0} ucap={UCAP} plain={PLAIN} max_m={MAX_M} prune_toggle={PRUNE_TOGGLE} pid={os.getpid()}",
          flush=True)
