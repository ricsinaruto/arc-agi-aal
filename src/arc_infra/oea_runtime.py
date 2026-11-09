"""Batch-aware expert routing (Opportunistic Expert Activation, arXiv 2511.02237; throughput arm, 16 Oct; dfr-exact fab6d74e).

Installed as ``vllm/arc_oea.py``; ``fused_topk_router.py`` calls ``install(FusedTopKRouter)`` at import. With
``ARC_OEA_K0`` = k0 (0 < k0 < top-k) every batch of at most ``ARC_OEA_MAX_M`` rows (decode / verify / draft steps)
routes as: each token keeps its own top-k0 experts; the step's union S of all tokens' top-k0 is what the MoE loads
anyway; each token's remaining top-k - k0 slots go to its highest-probability experts inside S (outside S only when S
has fewer than top-k experts). Weights are the token's softmax probabilities of the chosen experts, renormalized as
the pinned router does. Larger batches (prefill) route as pinned. Static shapes, no host sync: CUDA-graph safe.
Lossy (a token may swap up to top-k - k0 of its experts): needs a quality read.
"""

import os

import torch

K0 = int(os.environ.get("ARC_OEA_K0", "0") or 0)
MAX_M = int(os.environ.get("ARC_OEA_MAX_M", "256") or 256)
_seen = []


def install(cls):
    if not K0 or getattr(cls, "_arc_oea", False):
        return
    if not hasattr(cls, "_compute_routing"):
        raise RuntimeError("ARC_OEA: the pinned FusedTopKRouter has no _compute_routing")
    pinned = cls._compute_routing

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        m = router_logits.shape[0]
        if self.scoring_func != "softmax" or not 0 < m <= MAX_M or K0 >= self.top_k:
            return pinned(self, hidden_states, router_logits, indices_type, input_ids=input_ids)
        if not _seen:  # one line a process at the first routed batch: the hook is on the served path
            _seen.append(m)
            print(f"ARC_OEA first batch m={m} top_k={self.top_k} experts={router_logits.shape[-1]} pid={os.getpid()}",
                  flush=True)
        p = torch.softmax(router_logits.float(), dim=-1)
        top = p.topk(K0, dim=-1).indices.reshape(-1)  # scatter of a device tensor: no host copy under graph capture
        used = torch.zeros(p.shape[-1], dtype=torch.bool, device=p.device).scatter_(0, top, torch.ones_like(top, dtype=torch.bool))
        ids = torch.where(used, p, p - 2.0).topk(self.top_k, dim=-1).indices
        w = p.gather(1, ids)
        if self.renormalize:
            w = w / w.sum(dim=-1, keepdim=True)
        return w, ids.to(torch.int32 if indices_type is None else indices_type)

    cls._compute_routing = _compute_routing
    cls._arc_oea = True
    print(f"ARC_OEA k0={K0} max_m={MAX_M} pid={os.getpid()}", flush=True)
