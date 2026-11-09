"""Phase-4 serving overlay for the pinned Qwen3.8-Flash-Next vLLM runtime (kernel work, phase 4).

Copied into ``vllm/arc_serving_kernels.py`` by ``arc_infra.serving_kernels_patch``; its settings are
``arc_serving_kernels.json`` next to it. Each patched pinned file calls one entry point here. No arc
dependency inside the serving worker; vLLM imports are deferred so that the kernels can be loaded
standalone by the microbenchmark (scripts/kernel_bench_phase4.py).

Entry points (each is only reached when its pinned file was patched, i.e. its option is on):

- ``draft_quant_hook(draft_vllm_config)`` (mtp.py, ``draft_experts: "nvfp4"``): the MTP draft layer's
  512 BF16 routed experts are quantized to NvFp4 at load (one FP32 global scale per expert, E4M3 scale
  per 16 weights, vLLM's own online quantizer) and served by FlashInfer's fused SM120 b12x kernel, which
  quantizes activations dynamically, so no calibrated activation scale is needed. Draft-only: it moves
  the acceptance rate, never the verified output. A load-time check compares the layer against its BF16
  weights on fixed random tokens (``ARC_SK_CHECK``).
- ``shared_b12x_wrapper(experts)`` (flashinfer_b12x_moe.py): one ``B12xMoEWrapper`` (and its CUDA-graph
  workspace, ~0.6 GB at 4,096 batched tokens) per MoE geometry instead of one per layer (49 x ~0.6 GB).
  Layers run one after another on one stream and the result is copied out at once, so they can share it.
- ``topk_first(logits, k, p)`` (topk_topp_sampler.py + states.py, ``topk_sampler``): top-k first, then
  top-p over the k survivors, instead of sorting the 248K vocabulary. The kept set equals vLLM's
  (``logits >= k-th value``, ties kept) and the top-p cut is taken over the same renormalised top-k
  distribution; it only runs when every row's top-k is known on the host to be <= TOPK_MAX.

Serving tweaks (28 September; docs/validation-20260927/serving-tweaks.md):

- ``draft_logits(model, hidden_states)`` (mtp.py ``compute_logits``, ``draft_vocab``): the draft head over
  the fixed token subset in ``arc_draft_vocab.json``: the shared target head's rows for those ids (INT8
  + scales under the phase-2 A+C kernels) are gathered once, the logits are computed over 16,384 rows
  and written into a persistent full-width row that stays -inf everywhere else. The speculator's greedy
  draft takes its argmax, so a draft token is always in the subset; the target verifies every token, so
  the output is exact and only acceptance can move. Built on the first call (the profile run, before
  graph capture) with a load-time check against the full head (``ARC_SK_CHECK``).
- ``ple_uva_device`` / ``ple_uva_place`` (ple_layer.py, ``ple_uva``): the PLE n-gram table is built in
  page-locked, device-mapped host memory (anonymous mmap + cuMemHostRegister) and the parameter becomes
  its CUDA (UVA) view, so the GPU worker's own PLE path (hash on the GPU, ``F.embedding``) reads the
  2,560 bytes per token over PCIe inside the graph; the CPU offload process, its ZMQ hop and the GPU's
  wait on its semaphore are gone (``VLLM_PLE_CPU_OFFLOAD=0``). Checkpoint shards are copied into the
  host table through its CPU alias. After loading, a check compares the GPU path on the UVA table with
  the offload process's CPU path on the same table (``ARC_SK_CHECK``).
- ``host_embedding(factory)`` (model.py, ``embed_host``): the same for the target's input embedding.
"""

import atexit
import json
import mmap
import os
import sys
import types
from functools import partial
from pathlib import Path

import torch

CONFIG_PATH = Path(__file__).with_name("arc_serving_kernels.json")
DRAFT_VOCAB_PATH = Path(__file__).with_name("arc_draft_vocab.json")
# Draft head check: the subset kernel against the full head's columns on the same rows (the W8A16 split-K
# differs with N, so BF16 rounding may differ; a wrong row mapping is ~100%).
DRAFT_HEAD_ERROR_LIMIT = 0.02
# cuMemHostRegister flags: PORTABLE (every context) | DEVICEMAP (a device pointer for UVA reads).
HOST_REGISTER_FLAGS = 0x01 | 0x02
# Host-table checks: rows gathered through UVA against the CPU alias, and the PLE probe batch.
HOST_CHECK_ROWS = 4096
# Top-k-first sampler: rows' top-k must be <= TOPK_MAX; the window adds TOPK_MARGIN so that ties at the
# k-th value rarely overflow it (an overflowing row keeps vLLM's top-k set and skips the top-p cut).
TOPK_MAX = 64
TOPK_MARGIN = 16
TOPK_MAX_ROWS = 256
# Phase-4 microbenchmark: ~0.29 ms at every batch against the pinned sort's 0.19 ms at one row and 0.51-1.26 ms
# at 2-32 rows (the pinned Triton path, taken from 8 rows, 1.4-2.2 ms); one row keeps the pinned sort.
TOPK_MIN_ROWS = 2
# Draft-expert check: NvFp4 W4A4 against the BF16 layer (weights and dynamic activation quantization
# together) is ~10-15% relative error, ~27% with b12x's baked block scales (phase-4 microbenchmark); a
# layout mistake is ~100%.
DRAFT_ERROR_LIMIT = 0.40
# b12x picks micro (<= 40 routed rows), static (<= this many) or dynamic kernels by routed rows = tokens x 10.
# Micro and static kernels are compiled per token count, dynamic ones once. vLLM pads decode batches to its
# CUDA-graph sizes (<= 32 tokens here), but eager batches of 33-64 tokens would each compile a static kernel
# mid-run at the default 640. b12x serves only the one draft layer: 40 (micro, then dynamic) keeps its static
# workspace at ~26 MB: the KV pool is what the six 64K windows need.
B12X_DRAFT_ONLY_CUTOVER_PAIRS = 40
B12X_CUTOVER_ENV = "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS"

_STATE = {"config": None, "b12x": {}, "counts": {}, "draft_head": None, "host_tables": [],
          "ple_layers": {}}


def load_config():
    if _STATE["config"] is None:
        _STATE["config"] = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    return _STATE["config"]


def log(kind, payload):
    print(f"ARC_SK_{kind} " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def _count(name):
    """Count an entry-point call; logged as ``ARC_SK_COUNT`` at 1, 2, 4, 8, ... calls (so a server killed before a
    clean exit still shows the order of magnitude) and all counts as ``ARC_SK_COUNTS`` at interpreter exit."""
    n = _STATE["counts"][name] = _STATE["counts"].get(name, 0) + 1
    if n & (n - 1) == 0:
        log("COUNT", {"name": name, "count": n, "pid": os.getpid()})


def _log_counts():
    if _STATE["counts"]:
        log("COUNTS", {**_STATE["counts"], "pid": os.getpid()})


atexit.register(_log_counts)


# ============================================================================ 1. NvFp4 draft experts
def draft_quant_hook(draft_vllm_config):
    """Called at the end of the pinned ``_make_draft_vllm_config``: route the draft's MoE layers to an
    NvFp4 online-quantized method served by b12x (the draft quant config is the draft's own instance)."""
    mode = load_config().get("draft_experts")
    if mode is None:
        return
    if mode != "nvfp4":
        raise RuntimeError(f"ARC serving kernels: unknown draft_experts {mode!r}")
    quant_config = draft_vllm_config.quant_config
    if quant_config is None:
        raise RuntimeError("ARC NvFp4 draft experts need the draft's quant config instance")
    original = quant_config.get_quant_method

    def get_quant_method(layer, prefix):
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        if isinstance(layer, RoutedExperts):
            return _draft_method_class()(layer=layer, prefix=prefix)
        return original(layer, prefix)

    quant_config.get_quant_method = get_quant_method
    log("DRAFT", {"draft_experts": mode, "quant_config": type(quant_config).__name__})


_DRAFT_CLASS = []


def draft_moe_backend(major):
    """The draft layer's MoE backend request by compute capability major: b12x on SM120, vLLM's auto order on SM100."""
    if major == 12:
        return "flashinfer_b12x"
    if major == 10:
        return "auto"
    raise RuntimeError(f"ARC NvFp4 draft experts: SM{major}x is neither SM120 (b12x) nor SM100 (TRT-LLM / CUTLASS)")


def draft_backend_ok(requested, chosen):
    """b12x when asked; on auto any real NvFp4 kernel (never the emulation, Marlin's W4A16 or Humming's)."""
    if requested == "flashinfer_b12x":
        return chosen == "FLASHINFER_B12X"
    return chosen in ("FLASHINFER_TRTLLM", "FLASHINFER_CUTLASS", "FLASHINFER_CUTEDSL", "VLLM_CUTLASS")


def _draft_method_class():
    if _DRAFT_CLASS:
        return _DRAFT_CLASS[0]
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.online.moe_base import OnlineMoEMethodBase
    from vllm.model_executor.layers.quantization.online.nvfp4 import Nvfp4OnlineMoEMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    class ArcNvfp4DraftMoEMethod(Nvfp4OnlineMoEMethod):
        """vLLM's online NvFp4 MoE method (BF16 experts loaded, quantized after loading) on the SM120 b12x
        kernel instead of the SM100-only TRT-LLM one; the draft layer's own backend is b12x whatever the
        target's ``--moe-backend`` is."""

        def __init__(self, *, layer, prefix=""):
            OnlineMoEMethodBase.__init__(self, layer.moe_config)
            # SM120 (RTX): b12x, whatever the target's --moe-backend is; SM100/SM103 (B300, RL redesign E2): vLLM's
            # own NvFp4 order (TRT-LLM first), never the emulation
            self.moe.moe_backend = draft_moe_backend(torch.cuda.get_device_capability()[0])  # the draft's own config
            # the draft is not part of a LoRA adapter (vLLM never wraps it), but its config inherits
            # lora_config from the target's; b12x refuses LoRA-enabled configs
            self.moe.is_lora_enabled = False
            self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
                config=self.moe, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic)
            if not draft_backend_ok(self.moe.moe_backend, self.nvfp4_backend.value):
                raise RuntimeError(f"ARC NvFp4 draft experts: backend {self.nvfp4_backend} for {self.moe.moe_backend}")
            self.arc_prefix = prefix

        def process_weights_after_loading(self, layer):
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return
            probe = _draft_probe(layer)
            bf16_bytes = layer.w13_weight.numel() * 2 + layer.w2_weight.numel() * 2
            super().process_weights_after_loading(layer)
            stored = sum(getattr(layer, n).numel() * getattr(layer, n).element_size()
                         for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"))
            _draft_check(self, layer, probe, {"prefix": self.arc_prefix, "bf16_bytes": bf16_bytes,
                                              "nvfp4_bytes": stored, "backend": self.nvfp4_backend.value})
            torch.cuda.empty_cache()

    _DRAFT_CLASS.append(ArcNvfp4DraftMoEMethod)
    return ArcNvfp4DraftMoEMethod


def _draft_probe(layer, tokens=4):
    """Fixed random tokens and routes, with the BF16 layer's FP32 reference output (before quantization)."""
    w13, w2 = layer.w13_weight, layer.w2_weight  # [E, 2I, H] ([gate; up]), [E, H, I]
    experts, hidden, inter = w13.shape[0], w13.shape[2], w2.shape[2]
    top_k = int(layer.top_k)
    generator = torch.Generator(device=w13.device).manual_seed(0)
    x = torch.randn((tokens, hidden), generator=generator, device=w13.device).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, generator=generator, device=w13.device)[:top_k]
                       for _ in range(tokens)]).to(torch.int32)
    weights = torch.softmax(torch.randn((tokens, top_k), generator=generator, device=w13.device), dim=-1)
    ref = torch.zeros((tokens, hidden), dtype=torch.float32, device=w13.device)
    with torch.no_grad():
        for t in range(tokens):
            xt = x[t].float()
            for j in range(top_k):
                e = int(ids[t, j])
                h = w13[e].float() @ xt
                a = torch.nn.functional.silu(h[:inter]) * h[inter:]
                ref[t] += weights[t, j] * (w2[e].float() @ a)
    return {"x": x, "ids": ids, "weights": weights.float(), "ref": ref}


def draft_check_skip(experts):
    """Why the layer self-check cannot run on these fused experts (None: it can). A monolithic kernel (TRT-LLM on
    SM100) has an apply() with a router signature, not the modular one; acceptance in the run's first minutes is the
    check there (draft-only: rejection sampling keeps the output exact)."""
    if not hasattr(experts, "apply"):
        return "no fused_experts.apply on this backend"
    if getattr(experts, "is_monolithic", lambda: False)():
        return f"monolithic {type(experts).__name__}"
    return None


def _draft_check(method, layer, probe, info):
    out = torch.empty_like(probe["x"])
    skip = draft_check_skip(getattr(getattr(method, "moe_kernel", None), "fused_experts", None))
    if skip:
        log("CHECK", {**info, "skipped": skip})
        return
    with torch.no_grad():
        method.moe_kernel.fused_experts.apply(
            out, probe["x"], layer.w13_weight, layer.w2_weight, probe["weights"], probe["ids"],
            activation=layer.activation, global_num_experts=layer.global_num_experts, expert_map=None,
            a1q_scale=None, a2_scale=None, workspace13=None, workspace2=None, expert_tokens_meta=None,
            apply_router_weight_on_input=False)
        torch.cuda.synchronize()
    ref = probe["ref"]
    rel = ((out.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    result = {**info, "nvfp4_b12x_vs_bf16_rel_error": rel, "cosine": cos, "tokens": int(ref.shape[0])}
    if not rel <= DRAFT_ERROR_LIMIT:
        log("CHECK", {**result, "failed": True})
        raise RuntimeError(f"ARC NvFp4 draft-expert self-check failed: {result}")
    log("CHECK", result)


# ============================================================================ 2. shared b12x workspace
def shared_b12x_wrapper(experts):
    """One ``B12xMoEWrapper`` per geometry, shared by every layer of that geometry."""
    key = (experts.global_num_experts, experts.topk, experts.hidden_dim, experts.intermediate_size_per_partition,
           experts.max_num_tokens, experts.num_local_experts, experts._activation_str)
    wrapper = _STATE["b12x"].get(key)
    if wrapper is None:
        # Read (and cached) by FlashInfer on the first b12x call, which is this wrapper's allocation.
        os.environ.setdefault(B12X_CUTOVER_ENV, str(B12X_DRAFT_ONLY_CUTOVER_PAIRS))
        from flashinfer.fused_moe import B12xMoEWrapper

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        wrapper = B12xMoEWrapper(
            num_experts=experts.global_num_experts, top_k=experts.topk, hidden_size=experts.hidden_dim,
            intermediate_size=experts.intermediate_size_per_partition, use_cuda_graph=True,
            max_num_tokens=experts.max_num_tokens, num_local_experts=experts.num_local_experts,
            activation=experts._activation_str)
        torch.cuda.synchronize()
        _STATE["b12x"][key] = wrapper
        log("B12X", {"geometry": list(key), "workspace_bytes": torch.cuda.memory_allocated() - before,
                     "static_cutover_pairs": os.environ.get(B12X_CUTOVER_ENV)})
    _count("b12x_layers")
    return wrapper


# ============================================================================ 4. top-k-first sampler
def topk_first_impl(logits, k, p, max_k):
    """vLLM's top-k then top-p mask, computed on a top-(max_k + margin) window instead of a full sort.

    ``logits`` [B, V] is updated in place (as the pinned PyTorch path does). Rows with top-k == V are not
    allowed here (the caller checks ``max_k``). Top-p follows vLLM's rule on the renormalised top-k
    distribution: a token is dropped when the probability mass strictly above it is >= p (the largest
    token is always kept); rows with p >= 1 are left untouched by the top-p step.
    """
    window = min(logits.shape[1], int(max_k) + TOPK_MARGIN)
    vals, idx = logits.topk(window, dim=-1)
    kk = k.to(torch.long).clamp(1, window) - 1
    kth = vals.gather(1, kk.unsqueeze(1))
    logits.masked_fill_(logits < kth, float("-inf"))
    if p is None:
        return logits
    kept = vals.masked_fill(vals < kth, float("-inf"))
    probs = kept.float().softmax(dim=-1)
    above = probs.cumsum(dim=-1) - probs
    drop = above >= p.to(probs.dtype).unsqueeze(1)
    drop[:, 0] = False
    # A row whose kept set reaches the window's end may have more tied tokens outside it: keep vLLM's top-k
    # set and skip its top-p cut (the only case that can differ from the pinned sort).
    overflow = vals[:, -1:] >= kth
    drop &= ~overflow
    drop &= (p < 1.0).unsqueeze(1)
    logits.scatter_(1, idx, kept.masked_fill(drop, float("-inf")))
    return logits


def topk_first(logits, k, p):
    """Patched ``apply_top_k_top_p``: None falls back to the pinned implementation."""
    # Only reached off vLLM's FlashInfer sampling path (a greedy or seeded request in the batch, or processed
    # logprobs); arena gameplay batches sample through FlashInfer (docs/validation-20260927/hero-behavior.md).
    if k is None or logits.dim() != 2 or not TOPK_MIN_ROWS <= logits.shape[0] <= TOPK_MAX_ROWS:
        _count("topk_pinned")
        return None
    max_k = getattr(k, "_arc_max_k", None)
    if max_k is None or not 1 <= max_k <= TOPK_MAX or max_k >= logits.shape[1]:
        _count("topk_pinned")
        return None
    _count("topk_first")
    return topk_first_impl(logits, k, p, max_k)


def top_k_hint(top_k_gpu, top_k_np):
    """Patched ``SamplingStates.get_top_k_top_p``: attach the batch's largest top-k (known on the host)."""
    if top_k_gpu is not None and len(top_k_np):
        top_k_gpu._arc_max_k = int(top_k_np.max())
    return top_k_gpu


# ============================================================================ 5. draft head over a token subset
def load_draft_vocab(path=None):
    """The draft subset record (``serving_draft_vocab.json``): sorted distinct ids below ``vocab_rows``."""
    record = json.loads(Path(path or DRAFT_VOCAB_PATH).read_text())
    ids = record["ids"]
    if (len(ids) != record["size"] or ids != sorted(set(ids)) or not ids
            or not 0 <= ids[0] <= ids[-1] < record["vocab_rows"]):
        raise RuntimeError("ARC draft vocabulary file is malformed")
    return record


class DraftHead:
    """The draft head over the subset ``ids``: logits for those rows only, written into a persistent full-width
    row buffer whose other columns are -inf and are never written (the greedy draft is always in the subset).

    ``weight`` [K, H] holds the head rows of ``ids``: INT8 with per-row FP32 ``scale`` (the phase-2 A+C head, run
    by the same W8A16 op) or a float weight with ``scale`` None (F.linear, as the unquantized head).
    """

    def __init__(self, weight, scale, ids, width, rows, dtype):
        self.weight, self.scale, self.ids, self.width = weight, scale, ids, int(width)
        self.buffer = torch.full((int(rows), self.width), float("-inf"), dtype=dtype, device=weight.device)

    @classmethod
    def from_head(cls, lm_head, ids, width, rows, dtype):
        weight = lm_head.weight.detach()
        index = torch.as_tensor(ids, dtype=torch.long, device=weight.device)
        scale = getattr(lm_head, "weight_scale", None)
        if weight.dtype == torch.int8:
            if scale is None:
                raise RuntimeError("ARC draft head: an INT8 head without its per-row scales")
            return cls(weight.index_select(0, index).contiguous(), scale.detach().index_select(0, index).contiguous(),
                       index, width, rows, dtype)
        if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
            return cls(weight.index_select(0, index).contiguous(), None, index, width, rows, dtype)
        raise RuntimeError(f"ARC draft head: unsupported head weight dtype {weight.dtype}")

    def sub_logits(self, hidden_states):
        if self.scale is not None:
            return torch.ops.vllm.arc_w8a16_linear(hidden_states, self.weight, self.scale)
        return torch.nn.functional.linear(hidden_states.to(self.weight.dtype), self.weight)

    def __call__(self, hidden_states):
        rows = hidden_states.shape[0]
        sub = self.sub_logits(hidden_states)
        if rows <= self.buffer.shape[0]:
            out = self.buffer[:rows]
        else:  # more rows than max_num_seqs: never expected from the speculator; a fresh -inf row block
            _count("draft_head_wide_batch")
            out = torch.full((rows, self.width), float("-inf"), dtype=self.buffer.dtype, device=self.buffer.device)
        out.index_copy_(1, self.ids, sub.to(out.dtype))
        return out


def draft_logits(model, hidden_states):
    """Patched ``Qwen3_8FlashNextMTP.compute_logits``: the subset head once built, else the pinned head."""
    head = _STATE["draft_head"]
    if head is not None:
        return head(hidden_states)
    if not load_config().get("draft_vocab"):
        return model.logits_processor(model.lm_head, hidden_states)
    if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
        # The profile run's eager draft prefill builds it; a graph captured with the full head would keep it.
        raise RuntimeError("ARC draft head must be built before CUDA graph capture")
    return build_draft_head(model, hidden_states)(hidden_states)


def build_draft_head(model, probe, record=None, rows=None):
    """Gather the subset rows of the draft's (shared, target) head, check them against the full head and compile the
    W8A16 specialisations for every batch size up to max_num_seqs before graph capture."""
    record = record or load_draft_vocab()
    processor, lm_head = model.logits_processor, model.lm_head
    if (processor.soft_cap is not None or processor.scale != 1.0 or getattr(processor, "logits_as_input", False)
            or getattr(processor, "head_dtype", None) not in (None, probe.dtype)
            or getattr(lm_head, "tp_size", 1) != 1):
        raise RuntimeError("ARC draft head expects the plain TP=1 head (no soft cap, scale 1, head dtype = model)")
    width = int(processor.org_vocab_size)
    if record["ids"][-1] >= width:
        raise RuntimeError(f"ARC draft vocabulary reaches id {record['ids'][-1]} beyond the head's {width} rows")
    rows = int(rows or model.vllm_config.scheduler_config.max_num_seqs)
    # Normal (not inference) tensors: the speculator writes the persistent buffer in place under inference mode.
    with torch.inference_mode(False), torch.no_grad():
        head = DraftHead.from_head(lm_head, record["ids"], width, rows, probe.dtype)
        check = check_draft_head(model, head, probe.shape[-1], probe.dtype)
        warm = torch.zeros((rows, probe.shape[-1]), dtype=probe.dtype, device=probe.device)
        for m in range(1, rows + 1):
            head(warm[:m])
        if probe.is_cuda:
            torch.cuda.synchronize()
    _STATE["draft_head"] = head
    log("DRAFT_VOCAB", {"name": record.get("name"), "size": len(record["ids"]), "width": width, "rows": rows,
                        "kind": "int8" if head.scale is not None else str(head.weight.dtype),
                        "weight_bytes": head.weight.numel() * head.weight.element_size(),
                        "buffer_bytes": head.buffer.numel() * head.buffer.element_size(), "check": check})
    return head


def check_draft_head(model, head, hidden, dtype, rows=4):
    """The subset logits against the full head's columns on fixed random rows; the full-width row keeps -inf
    outside the subset and its argmax is the subset's argmax."""
    device = head.weight.device
    generator = torch.Generator(device=device).manual_seed(0)
    h = torch.randn((rows, hidden), generator=generator, device=device).to(dtype)
    full = model.logits_processor(model.lm_head, h)
    reference = full.index_select(1, head.ids).float()
    sub = head.sub_logits(h).float()
    rel = ((sub - reference).norm() / reference.norm().clamp(min=1e-12)).item()
    out = head(h)
    outside = out.shape[1] - head.ids.numel()
    result = {"rel_error_vs_full_head": rel, "rows": rows,
              "neg_inf_outside_subset": bool((torch.isneginf(out).sum(1) == outside).all().item()),
              "argmax_equal_fraction": (out.argmax(-1) == head.ids[reference.argmax(-1)]).float().mean().item(),
              "full_head_argmax_in_subset_fraction": torch.isin(full.argmax(-1), head.ids).float().mean().item()}
    if not (rel <= DRAFT_HEAD_ERROR_LIMIT and result["neg_inf_outside_subset"]
            and result["argmax_equal_fraction"] == 1.0):
        log("CHECK", {"draft_head": result, "failed": True})
        raise RuntimeError(f"ARC draft-head self-check failed: {result}")
    log("CHECK", {"draft_head": result})
    return result


# ============================================================================ 6. host tables read through UVA
class _DeviceBuffer:
    """``__cuda_array_interface__`` of a device-mapped host region (torch keeps this object, and so ``owner``,
    alive for as long as the tensor)."""

    def __init__(self, pointer, nbytes, owner):
        self.owner = owner
        self.__cuda_array_interface__ = {"shape": (int(nbytes),), "typestr": "|u1", "data": (int(pointer), False),
                                         "strides": None, "version": 2}


def _cuda_check(result, what):
    error = result[0] if isinstance(result, tuple) else result
    if int(getattr(error, "value", error)) != 0:
        raise RuntimeError(f"ARC {what} failed: {error}")
    return result


def _registered_region(driver, nbytes, name):
    """Anonymous mmap (page-aligned; the pinned caching host allocator would round 47.7 GiB up to 64 GiB) registered
    PORTABLE | DEVICEMAP; if the registration is refused, the driver's own mapped allocation (cuMemHostAlloc)."""
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if hasattr(mmap, "MADV_HUGEPAGE"):
        try:
            region.madvise(mmap.MADV_HUGEPAGE)
        except OSError:
            pass
    raw = torch.frombuffer(region, dtype=torch.uint8, count=nbytes)
    try:
        _cuda_check(driver.cuMemHostRegister(raw.data_ptr(), nbytes, HOST_REGISTER_FLAGS), f"cuMemHostRegister({name})")
        return raw, region, "cuMemHostRegister"
    except RuntimeError as error:
        log("HOST_TABLE", {"name": name, "register_refused": str(error)})
    del raw
    try:
        region.close()
    except BufferError:
        pass
    import ctypes

    _, pointer = _cuda_check(driver.cuMemHostAlloc(nbytes, HOST_REGISTER_FLAGS), f"cuMemHostAlloc({name})")
    buffer = (ctypes.c_uint8 * nbytes).from_address(int(pointer))
    return torch.frombuffer(buffer, dtype=torch.uint8, count=nbytes), buffer, "cuMemHostAlloc"


def host_table(shape, dtype, device, name):
    """A page-locked, device-mapped host table: (CPU alias, CUDA view of the same bytes)."""
    import time

    from cuda.bindings import driver

    numel = 1
    for size in shape:
        numel *= int(size)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    torch.zeros(1, device=device)  # the device's primary context is current on this thread
    started = time.perf_counter()
    raw, owner, method = _registered_region(driver, nbytes, name)
    pinned = time.perf_counter() - started
    _, pointer = _cuda_check(driver.cuMemHostGetDevicePointer(raw.data_ptr(), 0), f"cuMemHostGetDevicePointer({name})")
    try:
        view = torch.as_tensor(_DeviceBuffer(int(pointer), nbytes, (owner, raw)), device=device)
    except Exception as error:  # vLLM's own UVA view (cudaHostGetDevicePointer + from_blob)
        log("HOST_TABLE", {"name": name, "array_interface_refused": str(error)})
        view = torch.ops._C.get_cuda_view_from_cpu_tensor(raw)
    if view.device.type != "cuda" or view.numel() != nbytes or view.data_ptr() != int(pointer):
        raise RuntimeError(f"ARC host table {name}: the CUDA view is not the mapped region ({view.device})")
    entry = {"name": name, "bytes": nbytes, "method": method, "pin_seconds": pinned,
             "uva_equal_pointer": int(pointer) == raw.data_ptr()}
    _STATE["host_tables"].append({**entry, "owner": owner, "raw": raw, "view": view})
    log("HOST_TABLE", entry)
    return raw.view(dtype).view(shape), view.view(dtype).view(shape)


def host_table_deferred(shape, dtype, device, name):
    """``host_table`` with the page-locking (cuMemHostRegister: ~21 s for the 47.7 GiB PLE table, mostly faulting
    the pages in) in a thread beside the weight load (``pin_overlap``): returns the CPU alias at once and
    ``finish()``, which joins the registration and returns the CUDA view of the same bytes (called once the table is
    needed on the GPU: after its shards are copied in). A refused registration falls back to the driver's mapped
    allocation with the alias's bytes copied in."""
    import threading
    import time

    from cuda.bindings import driver

    numel = 1
    for size in shape:
        numel *= int(size)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    torch.zeros(1, device=device)
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if hasattr(mmap, "MADV_HUGEPAGE"):
        try:
            region.madvise(mmap.MADV_HUGEPAGE)
        except OSError:
            pass
    raw = torch.frombuffer(region, dtype=torch.uint8, count=nbytes)
    state = {"error": None, "seconds": None}

    def register():
        started = time.perf_counter()
        try:
            torch.cuda.set_device(device)
            torch.zeros(1, device=device)  # the primary context is current on this thread
            _cuda_check(driver.cuMemHostRegister(raw.data_ptr(), nbytes, HOST_REGISTER_FLAGS),
                        f"cuMemHostRegister({name})")
        except Exception as error:  # noqa: BLE001 - finish() falls back
            state["error"] = str(error)
        state["seconds"] = time.perf_counter() - started

    began = time.perf_counter()
    thread = threading.Thread(target=register, name=f"arc-pin-{name}", daemon=True)
    thread.start()
    done = {}

    def finish():
        if "view" in done:
            return done["view"]
        waited = time.perf_counter()
        thread.join()
        waited = time.perf_counter() - waited
        source, owner, method = raw, (region, raw), "cuMemHostRegister"
        if state["error"] is not None:
            log("HOST_TABLE", {"name": name, "register_refused": state["error"]})
            import ctypes

            _, pointer = _cuda_check(driver.cuMemHostAlloc(nbytes, HOST_REGISTER_FLAGS), f"cuMemHostAlloc({name})")
            buffer = (ctypes.c_uint8 * nbytes).from_address(int(pointer))
            source = torch.frombuffer(buffer, dtype=torch.uint8, count=nbytes)
            source.copy_(raw)
            owner, method = (buffer, region, raw), "cuMemHostAlloc"
        _, pointer = _cuda_check(driver.cuMemHostGetDevicePointer(source.data_ptr(), 0),
                                 f"cuMemHostGetDevicePointer({name})")
        try:
            view = torch.as_tensor(_DeviceBuffer(int(pointer), nbytes, (owner, source)), device=device)
        except Exception as error:
            log("HOST_TABLE", {"name": name, "array_interface_refused": str(error)})
            view = torch.ops._C.get_cuda_view_from_cpu_tensor(source)
        if view.device.type != "cuda" or view.numel() != nbytes or view.data_ptr() != int(pointer):
            raise RuntimeError(f"ARC host table {name}: the CUDA view is not the mapped region ({view.device})")
        entry = {"name": name, "bytes": nbytes, "method": method, "pin_seconds": state["seconds"],
                 "overlapped": True, "join_wait_seconds": round(waited, 3),
                 "ready_after_seconds": round(time.perf_counter() - began, 1),
                 "uva_equal_pointer": int(pointer) == source.data_ptr()}
        _STATE["host_tables"].append({**entry, "owner": owner, "raw": source, "view": view})
        log("HOST_TABLE", entry)
        done["view"] = view.view(dtype).view(shape)
        done["cpu"] = source.view(dtype).view(shape)
        return done["view"]

    return raw.view(dtype).view(shape), finish, done


def _worker_device():
    return torch.device("cuda", torch.cuda.current_device())


def _default_device_type():
    """The device a model is being built on (vLLM builds under ``with torch.device(...)``)."""
    return torch.empty(0).device.type


def _move_small_tensors(module, keep, device):
    """Every parameter and buffer of ``module`` except ``keep`` to ``device`` (built on the CPU with the table)."""
    for sub in module.modules():
        for param in sub.parameters(recurse=False):
            if param is not keep:
                param.data = param.data.to(device)
        for key, buffer in list(sub._buffers.items()):
            if buffer is not None:
                sub._buffers[key] = buffer.to(device)


def _after_load(layer, check):
    """Run ``check`` right after the layer's quant method has processed the loaded weights."""
    method = layer.quant_method
    original = method.process_weights_after_loading

    def process_weights_after_loading(module):
        original(module)
        check()

    method.process_weights_after_loading = process_weights_after_loading


def _gather(index, table):
    """A host-table lookup; the FP8 PLE table as a byte gather (F.embedding of the uint8 view, viewed back): the
    same bytes as the pinned ``F.embedding(input_, layer.weight)``, without relying on FP8 index kernels."""
    if table.element_size() == 1:
        return torch.nn.functional.embedding(index, table.view(torch.uint8)).view(table.dtype)
    return torch.nn.functional.embedding(index, table)


def _arc_uva_embedding(index: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return _gather(index, table)


def _arc_uva_embedding_fake(index: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return table.new_empty((*index.shape, table.shape[1]))


def _register_uva_op():
    """The host-table lookup as an opaque op: Inductor must never generate, autotune or benchmark a kernel with a
    host table as input (smoke v3: autotuning the backbone tried to allocate a 95.37 GiB copy of the PLE table)."""
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "arc_uva_embedding"):
        direct_register_custom_op(op_name="arc_uva_embedding", op_func=_arc_uva_embedding,
                                  fake_impl=_arc_uva_embedding_fake)


def _arc_ple_lookup(input_ids: torch.Tensor, query_start_loc: torch.Tensor, ngram_context: torch.Tensor,
                    layer: str) -> torch.Tensor:
    ple = _STATE["ple_layers"][layer]
    return type(ple).forward_impl(ple, None, input_ids, query_start_loc, ngram_context)


def _arc_ple_lookup_fake(input_ids: torch.Tensor, query_start_loc: torch.Tensor, ngram_context: torch.Tensor,
                         layer: str) -> torch.Tensor:
    ple = _STATE["ple_layers"][layer]
    return input_ids.new_empty((input_ids.reshape(-1).shape[0], ple.embedding_dim),
                               dtype=ple.ngram_embedding.weight.dtype)


def _register_ple_op():
    """The whole GPU PLE lookup (hashing, packing and the host-table gather) as one opaque op: the pinned eager
    code runs as is inside the graphs, and the compiler never traces or autotunes over the host table."""
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "arc_ple_lookup"):
        direct_register_custom_op(op_name="arc_ple_lookup", op_func=_arc_ple_lookup, fake_impl=_arc_ple_lookup_fake)


def _ple_forward(layer, hidden_states, input_ids, query_start_loc, ngram_context, output_buffer=None):
    """The PLE embedding's ``forward_impl`` (instance attribute) on the worker: the opaque lookup."""
    if output_buffer is not None:
        raise RuntimeError("ARC PLE via UVA has no offload output buffer")
    return torch.ops.vllm.arc_ple_lookup(input_ids, query_start_loc, ngram_context, layer)


def uva_embedding(layer, input_):
    """The embedding method of a host-table layer (instance attribute on its quant method): the opaque lookup,
    byte-wise for the FP8 PLE table (the same bytes as the pinned ``F.embedding``)."""
    return torch.ops.vllm.arc_uva_embedding(input_, layer.weight)


def _gather_check(name, cpu_table, gpu_table, timed_rows, rows=HOST_CHECK_ROWS, min_nonzero=0.0):
    """Rows gathered on the GPU through UVA (as served) against the CPU alias; a table that is still all zeros was
    never loaded (``min_nonzero``: a dense embedding; hashed PLE rows may be zero). Also times gathers of
    ``timed_rows`` rows (decode- and prefill-sized)."""
    generator = torch.Generator().manual_seed(0)
    index = torch.randint(0, cpu_table.shape[0], (rows,), generator=generator)
    device = gpu_table.device
    via_uva = _gather(index.to(device), gpu_table)
    direct = cpu_table.index_select(0, index)
    result = {"table": name, "rows": rows, "gather_equal": torch.equal(via_uva.cpu().view(torch.uint8),
                                                                       direct.view(torch.uint8)),
              "nonzero_byte_fraction": (direct.view(torch.uint8) != 0).float().mean().item()}
    for count in timed_rows:
        label = f"us_{count}_rows"
        sample = torch.randint(0, cpu_table.shape[0], (count,), generator=generator).to(device)
        _gather(sample, gpu_table)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            _gather(sample, gpu_table)
        end.record()
        end.synchronize()
        result[label] = start.elapsed_time(end) * 1000 / 20
    if not result["gather_equal"] or result["nonzero_byte_fraction"] < min_nonzero:
        log("CHECK", {"host_table": result, "failed": True})
        raise RuntimeError(f"ARC host-table self-check failed: {result}")
    return result


def _ple_mirror(ple, cpu_table):
    """The PLE module's CPU twin for the offload process's own code path (same table, CPU buffers)."""
    cls = type(ple)
    names = ("eos_token_id", "ngram_size", "heads_per_ngram", "embedding_dim", "head_dim")
    tensors = ("positions_buffer", "padded_buffer", "layer_multipliers", "ngram_heads_vocab_sizes",
               "ngram_heads_offsets")
    return types.SimpleNamespace(**{n: getattr(ple, n) for n in names},
                                 **{n: getattr(ple, n).detach().cpu().clone() for n in tensors},
                                 ngram_embedding=types.SimpleNamespace(weight=cpu_table),
                                 _shift_precompute=cls._shift_precompute, _shift_apply=cls._shift_apply)


def ple_probe(vocab, eos, context_len, generator):
    """A CUDA-graph-like batch: three requests (5, 1 and 7 tokens, an EOS inside the first), a zero-length padded
    request and three padding tokens after the 13 real ones."""
    ids = torch.randint(0, vocab, (16,), generator=generator, dtype=torch.int64)
    ids[3] = eos
    query_start_loc = torch.tensor([0, 5, 6, 13, 13], dtype=torch.int32)
    context = torch.randint(0, vocab, (4, context_len), generator=generator, dtype=torch.int64)
    context[1, 0] = eos
    context[3] = eos
    return ids.to(torch.int32), query_start_loc, context.to(torch.int32), 13


def _ple_check(ple, cpu_table, gpu_table):
    """The UVA table gathers exactly, and the GPU PLE path on it equals the offload process's CPU path (the served
    path until now) on the real tokens of a padded probe batch, byte for byte."""
    # 24 decode tokens (six MTP-3 streams) and a 4,096-token prefill chunk, 16 rows each.
    result = _gather_check("ple", cpu_table.view(-1, ple.head_dim), gpu_table.view(-1, ple.head_dim),
                           (24 * 16, 4096 * 16))
    generator = torch.Generator().manual_seed(1)
    ids, qsl, context, real = ple_probe(ple.unigram_vocab_size, ple.eos_token_id, ple.ngram_size - 1, generator)
    device = gpu_table.device
    with torch.no_grad():
        on_gpu = ple.forward_impl(None, ids.to(device), qsl.to(device), context.to(device))
        module = sys.modules[type(ple).__module__]
        saved = module.is_offload_process
        module.is_offload_process = lambda: True
        try:
            output = torch.empty((ids.shape[0], ple.embedding_dim), dtype=cpu_table.dtype)
            # As the offload worker calls it: (hidden_states=input_ids, input_ids, query_start_loc, ngram_context).
            reference = type(ple).forward_impl(_ple_mirror(ple, cpu_table), ids, ids, qsl[:4], context[:3],
                                               output_buffer=output)
        finally:
            module.is_offload_process = saved
    result["offload_path_equal"] = torch.equal(on_gpu[:real].cpu().view(torch.uint8),
                                               reference[:real].view(torch.uint8))
    result["probe_tokens"] = real
    if not result["offload_path_equal"]:
        log("CHECK", {"ple_uva": result, "failed": True})
        raise RuntimeError(f"ARC PLE UVA self-check failed: {result}")
    log("CHECK", {"ple_uva": result})


def _check_identity(extra):
    """sha256 (16 hex) of this helper, its settings, the torch/CUDA build, the device and ``extra``."""
    import hashlib

    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(json.dumps(load_config(), sort_keys=True).encode())
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    h.update(json.dumps([torch.__version__, torch.version.cuda, device, extra], default=str).encode())
    return h.hexdigest()[:16]


def cached_check(name, extra, fn, environ=None):
    """ARC_SK_CHECK_CACHE=<dir> (boot cut, MFU-2 24 Oct): a load-time self-check that passed for the same identity
    (helper, settings, torch/CUDA, device, ``extra``) on an earlier boot, whose marker came back with the boot cache,
    is skipped; otherwise it runs (it raises on a failure) and the marker is written. Logs the seconds either way."""
    import time

    root = (os.environ if environ is None else environ).get("ARC_SK_CHECK_CACHE")
    began = time.perf_counter()
    marker = None
    if root:
        marker = Path(root) / f"{name}-{_check_identity(extra)}.ok"
        if marker.is_file():
            log("CHECK_SKIPPED", {name: {"marker": marker.name}})
            return False
    fn()
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"name": name, "extra": extra}, default=str) + "\n")
    log("CHECK_SECONDS", {name: round(time.perf_counter() - began, 3)})
    return True


def _ple_identity(ple, table):
    module = sys.modules.get(type(ple).__module__)
    source = getattr(module, "__file__", None)
    import hashlib

    return {"ple_source": hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
            "table": [list(table.shape), str(table.dtype)]}


def ple_uva_device(default):
    """Patched PLE constructor: build the n-gram embedding on the CPU (then ``ple_uva_place``)."""
    if not load_config().get("ple_uva"):
        return default
    import vllm.envs as envs
    from vllm.model_executor.layers.ple_offload_layer import is_offload_process

    if envs.VLLM_PLE_CPU_OFFLOAD or is_offload_process():
        raise RuntimeError("ARC PLE via UVA needs VLLM_PLE_CPU_OFFLOAD=0 (no PLE offload process)")
    return torch.device("cpu")


def ple_uva_place(ple):
    """The n-gram table into a host table (its parameter becomes the CUDA view), everything else of the PLE
    embedding to the GPU; checkpoint shards are copied through the CPU alias; checks after loading."""
    if not load_config().get("ple_uva"):
        return
    device = _worker_device()
    weight = ple.ngram_embedding.weight
    if weight.device.type != "cpu":
        raise RuntimeError(f"ARC PLE via UVA expects the table built on the CPU, got {weight.device}")
    if load_config().get("pin_overlap"):  # the table's page-locking beside the weight load (MFU-2, 22 Oct)
        cpu_table, finish, tables = host_table_deferred(tuple(weight.shape), weight.dtype, device, "ple")
        _move_small_tensors(ple, weight, device)
        weight.data = cpu_table  # the CUDA view once finish() joined the registration (after the shards)
        load_weights = ple.load_weights

        def load_through_cpu_alias(weights):
            weight.data = cpu_table
            try:
                return load_weights(weights)
            finally:
                weight.data = finish()

        def placed_check():
            weight.data = finish()  # also when no shard reached load_weights
            cached_check("ple_uva", _ple_identity(ple, tables["cpu"]),
                         partial(_ple_check, ple, tables["cpu"], tables["view"]))

        ple.load_weights = load_through_cpu_alias
        check = placed_check
    else:
        cpu_table, gpu_table = host_table(tuple(weight.shape), weight.dtype, device, "ple")
        _move_small_tensors(ple, weight, device)
        weight.data = gpu_table
        load_weights = ple.load_weights

        def load_through_cpu_alias(weights):
            weight.data = cpu_table
            try:
                return load_weights(weights)
            finally:
                weight.data = gpu_table

        ple.load_weights = load_through_cpu_alias
        check = partial(cached_check, "ple_uva", _ple_identity(ple, cpu_table),
                        partial(_ple_check, ple, cpu_table, gpu_table))
    _register_uva_op()
    ple.ngram_embedding.quant_method.embedding = uva_embedding
    _register_ple_op()
    key = f"ple{len(_STATE['ple_layers'])}"
    _STATE["ple_layers"][key] = ple
    ple.forward_impl = partial(_ple_forward, key)
    _after_load(ple.ngram_embedding, check)
    log("PLE", {"table_shape": list(weight.shape), "dtype": str(weight.dtype), "device": str(device)})


def host_embedding(factory):
    """Patched ``Qwen3_8FlashNextModel`` embedding: built on the CPU into a host table read through UVA (a model
    built on another device, e.g. the PLE offload process's meta model, keeps the pinned construction)."""
    if not load_config().get("embed_host") or _default_device_type() != "cuda":
        return factory()
    device = _worker_device()
    with torch.device("cpu"):
        layer = factory()
    weight = layer.weight
    cpu_table, gpu_table = host_table(tuple(weight.shape), weight.dtype, device, "embed_tokens")
    _move_small_tensors(layer, weight, device)
    weight.data = gpu_table
    loader = weight.weight_loader

    def load_through_cpu_alias(param, loaded_weight, *args, **kwargs):
        param.data = cpu_table
        try:
            return loader(param, loaded_weight, *args, **kwargs)
        finally:
            param.data = gpu_table

    weight.weight_loader = load_through_cpu_alias
    _register_uva_op()
    layer.quant_method.embedding = uva_embedding
    _after_load(layer, lambda: log("CHECK", {"embed_host": _gather_check("embed_tokens", cpu_table, gpu_table,
                                                                         (24, 4096), min_nonzero=0.5)}))
    log("EMBED", {"shape": list(weight.shape), "dtype": str(weight.dtype), "device": str(device)})
    return layer

