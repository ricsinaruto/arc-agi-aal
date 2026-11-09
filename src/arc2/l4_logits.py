"""The logits of a tensor-parallel step without the full all-gather (four L4s: 248,320 x 4 bytes a row to every
rank, about 2 ms of a 51-56 ms decode step at 16-32 rows). Each rank takes the top ``k`` of its vocabulary
shard; the (value, index) pairs are all-gathered (``rows x 2k`` floats a rank) and written into a full row of
-inf on every rank, so the sampler sees what it would have seen for every token that can be sampled.

Exact for the play's requests: they sample with top_k 20 (``arc2.tasks.chat_request``), and the global top
``k`` tokens are always among the ranks' top ``k`` each, so with ``k`` >= 20 the top-k / top-p set is the full
row's; penalties only lower logits, so up to ``k - 20`` tokens can fall out of the top 20 and still be there.
Server-wide: a request with no top-k (the recipe's readiness checks) samples from the gathered tokens only;
greedy requests are unchanged. Not applied with a logit soft-cap (tanh would turn -inf into a finite value) or
to a one-rank head. ``ARC2_L4_LOGITS_TOPK`` = k turns it on (``arc2.l4_serving``, option ``logits_topk``).
"""

import json
import os

ENV = "ARC2_L4_LOGITS_TOPK"


def assemble(torch, gathered, rows, ranks, k, per, dtype):
    """The full rows from the all-gathered pairs: ``gathered`` [rows, ranks * 2k] (each rank's k values, then
    its k global indices, as float32), ``per`` the shard width; -inf everywhere else."""
    parts = gathered.view(rows, ranks, 2, k)
    values = parts[:, :, 0, :].reshape(rows, ranks * k)
    index = parts[:, :, 1, :].reshape(rows, ranks * k).to(torch.int64)
    full = torch.full((rows, ranks * per), float("-inf"), dtype=dtype, device=gathered.device)
    full.scatter_(1, index, values.to(dtype))
    return full


def local_pairs(torch, logits, k, rank):
    """This rank's top ``k`` of its shard as [rows, 2k] float32: values, then global indices (exact in float32
    below 2**24; the vocabulary is 248,320)."""
    values, index = logits.topk(k, dim=-1)
    index = index + rank * logits.shape[-1]
    return torch.cat([values.float(), index.float()], dim=-1)


def gather(torch, logits, k, rank, ranks, all_gather):
    rows, per = logits.shape
    pairs = local_pairs(torch, logits, k, rank)
    return assemble(torch, all_gather(pairs), rows, ranks, k, per, logits.dtype)


def install(k):
    import torch
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.model_executor.layers import logits_processor

    cls = logits_processor.LogitsProcessor
    if getattr(cls, "_arc2_topk", None):
        return False
    original = cls._gather_logits

    def gather_logits(self, logits):
        ranks = get_tensor_model_parallel_world_size()
        if (ranks == 1 or logits.dim() != 2 or self.soft_cap is not None or self.scale <= 0
                or k >= logits.shape[-1]):
            return original(self, logits)
        return gather(torch, logits, k, get_tensor_model_parallel_rank(), ranks,
                      lambda pairs: tensor_model_parallel_all_gather(pairs, dim=-1))

    cls._gather_logits = gather_logits
    cls._arc2_topk = k
    return True


def register():
    """vLLM general plugin: in every worker before the model is built (the class is patched, not an instance)."""
    k = int(os.environ.get(ENV) or 0)
    if k <= 0:
        return
    installed = install(k)
    print("ARC2_LOGITS_TOPK " + json.dumps({"k": k, "installed": installed, "pid": os.getpid()}), flush=True)
