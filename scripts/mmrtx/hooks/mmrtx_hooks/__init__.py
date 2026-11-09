"""mmrtx reference-run hooks for vLLM's MiniMax-M3 model (loaded in every vLLM process as a general plugin; active only
with MMRTX_HOOKS=1). While a tag is set (``set_tag`` over collective_rpc) they accumulate, per tag, on the GPU:
- router mass: per MoE layer and expert, the summed renormalized top-4 weight (sigmoid scores, selection by score +
  e_score_correction_bias, as the model routes) and the selection count; every SAMPLE-th token's top-8 candidates
  (ids by biased score, unbiased sigmoid scores) for offline pruned-routing replays;
- MSA block churn: on every CHURN_EVERY-th sparse layer, for consecutive prefill positions of one sequence, how many of
  token t+1's top-k blocks (per local index head) were not selected at t, and the distinct blocks per token over the
  local heads, binned by position.
No tag (profiling / warm-up / dummy runs) = no work. ``snapshot(reset)`` returns CPU copies."""
import os

import torch

SAMPLE = int(os.environ.get("MMRTX_SAMPLE", "32"))
CHURN_EVERY = int(os.environ.get("MMRTX_CHURN_EVERY", "4"))
BINS = (8192, 32768, 65536, 131072)  # position bins: <8K, 8-32K, 32-64K, 64-128K, >=128K
STATE = {"tag": None, "acc": {}}


def _acc(tag, device):
    a = STATE["acc"].get(tag)
    if a is None:
        a = STATE["acc"][tag] = {
            "mass": torch.zeros(60, 128, dtype=torch.float64, device=device),
            "count": torch.zeros(60, 128, dtype=torch.int64, device=device),
            "tokens": torch.zeros(60, dtype=torch.int64, device=device),
            "samples": [],  # (layer, ids int16 [n, 8], scores fp16 [n, 8])
            "churn_new": torch.zeros(60, len(BINS) + 1, dtype=torch.float64, device=device),
            "churn_union": torch.zeros(60, len(BINS) + 1, dtype=torch.float64, device=device),
            "churn_n": torch.zeros(60, len(BINS) + 1, dtype=torch.int64, device=device),
            "heads": 0,
        }
    return a


def route_stats(layer, logits, bias, tag):
    """Accumulate one MoE layer's routing for a [T, 128] fp32 logits batch."""
    a = _acc(tag, logits.device)
    s = torch.sigmoid(logits.float())
    biased = s + bias.float()[None, :] if bias is not None else s
    top8 = biased.topk(8, dim=-1).indices
    sel = top8[:, :4]
    w = s.gather(1, sel)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    a["mass"][layer].index_add_(0, sel.reshape(-1), w.reshape(-1).double())
    a["count"][layer].index_add_(0, sel.reshape(-1), torch.ones_like(sel.reshape(-1)))
    a["tokens"][layer] += s.shape[0]
    if SAMPLE > 0:
        idx = torch.arange(0, s.shape[0], SAMPLE, device=s.device)
        a["samples"].append((layer, top8[idx].to(torch.int16), s.gather(1, top8[idx]).half()))


def churn_stats(layer, positions, topk, tag):
    """topk: [T, H, K] block ids of T prefill tokens whose positions are ``positions``."""
    T = topk.shape[0]
    if T < 2:
        return
    a = _acc(tag, topk.device)
    a["heads"] = int(topk.shape[1])
    prev, cur = topk[:-1].long(), topk[1:].long()
    same = positions[1:] == positions[:-1] + 1
    present = (cur.unsqueeze(-1) == prev.unsqueeze(-2)).any(-1)  # [T-1, H, K]
    new = (~present & (cur >= 0)).sum(-1).double().mean(-1)  # new blocks per head, mean over heads
    flat = cur.flatten(1).sort(-1).values
    distinct = ((flat[:, 1:] != flat[:, :-1]) & (flat[:, 1:] >= 0)).sum(-1) + (flat[:, 0] >= 0).long()
    b = torch.bucketize(positions[1:], torch.tensor(BINS, device=positions.device), right=True)
    m = same
    a["churn_new"][layer].index_add_(0, b[m], new[m])
    a["churn_union"][layer].index_add_(0, b[m], distinct[m].double())
    a["churn_n"][layer].index_add_(0, b[m], torch.ones_like(b[m]))


def register():
    if os.environ.get("MMRTX_HOOKS") != "1":
        return
    from vllm.models.minimax_m3.nvidia import model as m

    moe_init = m.MiniMaxM3MoE.__init__

    def moe_init_hook(self, config, layer_id, *args, **kwargs):
        moe_init(self, config, layer_id, *args, **kwargs)
        gate, owner = self.gate, self
        gate_forward = gate.forward

        def forward(x, *a, **k):
            out = gate_forward(x, *a, **k)
            tag = STATE["tag"]
            if tag is not None:
                logits = out[0] if isinstance(out, tuple) else out
                route_stats(layer_id, logits, owner.e_score_correction_bias, tag)
            return out

        gate.forward = forward

    m.MiniMaxM3MoE.__init__ = moe_init_hook

    sa_init = m.MiniMaxM3SparseAttention.__init__

    def sa_init_hook(self, *args, **kwargs):
        sa_init(self, *args, **kwargs)
        self._mmrtx_layer = kwargs.get("layer_id")

    m.MiniMaxM3SparseAttention.__init__ = sa_init_hook
    sa_forward = m.MiniMaxM3SparseAttention.forward

    def sa_forward_hook(self, positions, hidden_states, *a, **k):
        out = sa_forward(self, positions, hidden_states, *a, **k)
        tag, layer = STATE["tag"], getattr(self, "_mmrtx_layer", None)
        if tag is not None and layer is not None and layer % CHURN_EVERY == 0:
            buf = getattr(self, "topk_indices_buffer", None)  # token-major [T, H, k] in both impls
            T = positions.shape[0]
            if buf is not None and 8 < T <= buf.shape[0]:
                churn_stats(layer, positions, buf[:T], tag)
        return out

    m.MiniMaxM3SparseAttention.forward = sa_forward_hook
    STATE["registered"] = True


def set_tag(worker=None, tag=None):
    STATE["tag"] = tag
    return STATE.get("registered", False)


def snapshot(worker=None, reset=True):
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001 - no distributed group (CPU preflight)
        rank = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out = {"rank": rank, "registered": STATE.get("registered", False), "acc": {}}
    for tag, a in STATE["acc"].items():
        d = {k: v.cpu() for k, v in a.items() if torch.is_tensor(v)}
        d["heads"] = a["heads"]
        if a["samples"]:
            d["sample_layer"] = torch.tensor([l for l, i, s in a["samples"] for _ in range(i.shape[0])], dtype=torch.int16)
            d["sample_ids"] = torch.cat([i for _, i, _ in a["samples"]]).cpu()
            d["sample_scores"] = torch.cat([s for _, _, s in a["samples"]]).cpu()
        out["acc"][tag] = d
    if reset:
        STATE["acc"] = {}
    return out
