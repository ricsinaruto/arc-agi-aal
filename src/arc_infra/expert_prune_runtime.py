"""Load-time expert pruning, worker side (copied to ``vllm/arc_expert_prune.py`` by ``expert_prune_patch``).

The model is built with ``num_experts`` = the kept count (``prepare_config``, before ``initialize_model``) and
its weights are streamed through ``filter_weights`` (in ``DefaultModelLoader.load_weights``):

- a pruned expert's tensors (``model.language_model.layers.L.mlp.experts.E.*``) are dropped;
- a kept expert's tensors are renamed to its rank among the layer's kept experts (ascending ids);
- the router (``...layers.L.mlp.gate.weight``, [512, H]) keeps only the kept experts' rows, in the same order.

So the router's softmax, top-10 and renormalization run over the kept experts only: the same routing as the
additive −inf mask measured in docs/validation-20260927/expert-usage.md, with no runtime hook (a hook inside
captured CUDA graphs hangs the pinned decode warm-up). The MTP draft's own experts (``mtp.*``) are untouched, so
the overlay requires MTP-0. The settings (``arc_expert_prune.json``: ``experts``, ``pruned`` per layer) sit next
to this file. Stdlib at import; torch only for the router slice.

A layer-aware mask keeps a different count per layer. ``num_experts`` is then the largest kept count (the config
value the decoder's MoE test reads), and ``install_moe_sizes`` makes each MoE block build with its own layer's
count (the block reads ``config.num_experts`` in ``__init__`` only; each FusedMoE builds its own expert mapping).
"""

import json
import os
import re

SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_expert_prune.json")
NAME = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.(?:experts\.(\d+)\.(.+)|gate\.weight)$")
_PLAN = None


TOP_K_FLOOR = 10  # every layer keeps at least top-k experts


def plan(experts, pruned, min_kept=1):
    """Kept expert ids (ascending) and old -> new id maps per layer. ``kept_per_layer`` is the largest kept count
    (the config's ``num_experts``); ``kept_counts`` has each layer's; ``uniform`` says whether they are all equal.
    The served settings require ``min_kept`` = top-k (10)."""
    kept, remap = {}, {}
    for layer, ids in pruned.items():
        drop = {int(e) for e in ids}
        if len(drop) != len(ids) or not drop <= set(range(experts)):
            raise ValueError(f"layer {layer}: pruned ids must be distinct and in 0..{experts - 1}")
        keep = [e for e in range(experts) if e not in drop]
        if len(keep) < min_kept:
            raise ValueError(f"layer {layer}: keeps {len(keep)} experts, fewer than {min_kept}")
        kept[int(layer)], remap[int(layer)] = keep, {e: i for i, e in enumerate(keep)}
    if not kept:
        raise ValueError("the pruning plan has no layers")
    counts = {layer: len(keep) for layer, keep in kept.items()}
    return {"experts": experts, "kept_per_layer": max(counts.values()), "kept_counts": counts,
            "uniform": len(set(counts.values())) == 1, "kept": kept, "remap": remap}


def settings():
    global _PLAN
    if _PLAN is None:
        with open(SETTINGS) as handle:
            raw = json.load(handle)
        _PLAN = plan(int(raw["experts"]), raw["pruned"], min_kept=TOP_K_FLOOR)
        _PLAN["stall_dump_seconds"] = raw.get("stall_dump_seconds")
    return _PLAN


STEP_METHODS = ("execute_model", "sample_tokens")
DEFAULT_WORKER = "vllm.v1.worker.gpu_worker.Worker"


def worker_class(vllm_config):
    """The worker class this server runs (``parallel_config.worker_cls``, resolved by the platform before load)."""
    import importlib

    name = getattr(getattr(vllm_config, "parallel_config", None), "worker_cls", None)
    if not isinstance(name, str) or name == "auto":
        name = DEFAULT_WORKER
    module, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def _stall_guarded(original, seconds, faulthandler, stream):
    def step(self, *args, **kwargs):
        faulthandler.dump_traceback_later(seconds, repeat=False, file=stream)
        try:
            return original(self, *args, **kwargs)
        finally:
            faulthandler.cancel_dump_traceback_later()

    step.__name__, step.__qualname__, step.__doc__ = original.__name__, original.__qualname__, original.__doc__
    step.__wrapped__, step._arc_stall_dump = original, seconds
    return step


def install_stall_dump(seconds, cls, log=print):
    """Diagnostics (prune12 hang, 29 September): all-thread Python stacks to stderr (the server log) only when a
    single worker step stalls.

    Each ``execute_model`` / ``sample_tokens`` call arms a one-shot ``faulthandler`` timer of ``seconds`` on entry
    and cancels it on return or raise. So the dump fires only if one call has run for ``seconds`` (a hung step,
    whose main thread is blocked in a wait), never while the worker is idle between steps. The periodic timer of
    the first hang test (``repeat=True``) dumped busy threads and crashed the worker (SIGSEGV) in both arms.
    ``seconds`` stays below vLLM's 300 s execute-model RPC timeout, so the dump lands before the engine gives up.
    The C watchdog needs no GIL. Idempotent per class."""
    import faulthandler
    import sys

    seconds = int(seconds)
    wrapped = []
    for name in STEP_METHODS:
        original = getattr(cls, name, None)
        if original is None or getattr(original, "_arc_stall_dump", None) is not None:
            continue
        setattr(cls, name, _stall_guarded(original, seconds, faulthandler, sys.stderr))
        wrapped.append(name)
    log("ARC_EXPERT_PRUNE " + json.dumps({"stall_dump_seconds": seconds, "mode": "per-step one-shot",
                                          "worker": f"{cls.__module__}.{cls.__qualname__}", "wrapped": wrapped,
                                          "pid": os.getpid()}), flush=True)
    return wrapped


MEMORY_WATCH_ENV = "ARENA_MEMORY_WATCH"  # throughput arm (3 October): "1" logs worker GPU-memory growth per step
MEMORY_WATCH_STEP_BYTES = 64 << 20
MEMORY_WATCH_HEARTBEAT_SECONDS = 120.0


def _batch_summary(args, kwargs):
    """What a step scheduled (the SchedulerOutput argument of execute_model), for attributing memory growth."""
    so = args[0] if args else kwargs.get("scheduler_output")
    if so is None:
        return {}
    out = {}
    try:
        counts = getattr(so, "num_scheduled_tokens", None) or {}
        out["tokens"] = int(getattr(so, "total_num_scheduled_tokens", sum(counts.values())))
        out["requests"] = len(counts)
        out["max_request_tokens"] = max(counts.values()) if counts else 0
        new = getattr(so, "scheduled_new_reqs", None) or []
        out["new_requests"] = len(new)
        encoder = getattr(so, "scheduled_encoder_inputs", None) or {}
        out["encoder_inputs"] = sum(len(v) for v in encoder.values())
        spec = getattr(so, "scheduled_spec_decode_tokens", None) or {}
        out["spec_requests"] = len(spec)
        out["max_context"] = max((int(getattr(r, "num_computed_tokens", 0)) for r in new), default=0)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        out["summary_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


def _memory_watched(original, state, torch, log, clock):
    def step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if original.__name__ != "execute_model":
            return result
        state["steps"] += 1
        reserved, peak = torch.cuda.memory_reserved(), torch.cuda.max_memory_allocated()
        now = clock()
        grew = reserved > state["reserved"] + MEMORY_WATCH_STEP_BYTES or peak > state["peak"] + MEMORY_WATCH_STEP_BYTES
        if grew or now - state["beat"] >= MEMORY_WATCH_HEARTBEAT_SECONDS:
            free, total = torch.cuda.mem_get_info()
            log("ARC_MEMORY_WATCH " + json.dumps({
                "event": "growth" if grew else "heartbeat", "step": state["steps"],
                "reserved_mib": reserved >> 20, "allocated_mib": torch.cuda.memory_allocated() >> 20,
                "peak_allocated_mib": peak >> 20, "free_mib": free >> 20, "total_mib": total >> 20,
                "reserved_delta_mib": (reserved - state["reserved"]) >> 20,
                "peak_delta_mib": (peak - state["peak"]) >> 20, **_batch_summary(args, kwargs)}), flush=True)
            state["beat"] = now
        state["reserved"], state["peak"] = max(state["reserved"], reserved), max(state["peak"], peak)
        return result

    step.__name__, step.__qualname__, step.__doc__ = original.__name__, original.__qualname__, original.__doc__
    step.__wrapped__, step._arc_memory_watch = original, True
    return step


def install_memory_watch(cls, log=print, torch_module=None, clock=None):
    """Diagnostics (throughput arm, 3 October): the dfr-combined servers need ~5 GiB beyond the pool and the profiled
    activation peak once play starts (v1 OOMed at 0.96). With ``ARENA_MEMORY_WATCH=1`` every ``execute_model`` that
    raises the worker's reserved or peak-allocated CUDA memory by more than 64 MiB logs one line (and a heartbeat
    every 120 s) with what the step scheduled. Off (the default): nothing is wrapped. Idempotent per class."""
    import time

    torch = torch_module
    if torch is None:
        import torch
    original = getattr(cls, "execute_model", None)
    if original is None or getattr(original, "_arc_memory_watch", False):
        return False
    state = {"steps": 0, "reserved": torch.cuda.memory_reserved(), "peak": torch.cuda.max_memory_allocated(),
             "beat": 0.0}
    setattr(cls, "execute_model", _memory_watched(original, state, torch, log, clock or time.monotonic))
    log("ARC_MEMORY_WATCH " + json.dumps({"event": "installed", "reserved_mib": state["reserved"] >> 20,
                                           "pid": os.getpid()}), flush=True)
    return True


MOE_BLOCK = "vllm.model_executor.models.qwen3_next.Qwen3NextSparseMoeBlock"
MOE_PREFIX = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp$")


def resolve(qualname):
    import importlib

    module, _, attr = qualname.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def install_moe_sizes(counts, cls, log=print):
    """Each MoE block is built with its own layer's kept count: ``config.num_experts`` is set to ``counts[layer]``
    (the layer from the block's prefix, ``...layers.L.mlp``) for the block's ``__init__`` and restored after.
    Idempotent per class; a block whose layer has no count fails the load."""
    original = cls.__init__
    if getattr(original, "_arc_moe_sizes", None) is not None:
        return False
    counts = {int(k): int(v) for k, v in counts.items()}

    def init(self, vllm_config, prefix=""):
        match = MOE_PREFIX.search(prefix)
        config = vllm_config.model_config.hf_text_config
        if (match is None or int(match.group(1)) not in counts) and "mtp" in prefix.split(".") \
                and not hasattr(config, "arc_unpruned_num_experts"):
            # The MTP draft's MoE block (spec-decode pilot, 30 September): built from the draft's own, unpruned
            # model config (prepare_config marks only the target's); its mtp.* weights are never filtered.
            return original(self, vllm_config=vllm_config, prefix=prefix)
        if match is None or int(match.group(1)) not in counts:
            raise RuntimeError(f"expert pruning: no kept count for the MoE block {prefix!r}")
        before = config.num_experts
        config.num_experts = counts[int(match.group(1))]
        try:
            original(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            config.num_experts = before

    init.__name__, init.__qualname__, init.__doc__ = original.__name__, original.__qualname__, original.__doc__
    init.__wrapped__, init._arc_moe_sizes = original, counts
    cls.__init__ = init
    values = sorted(counts.values())
    log("ARC_EXPERT_PRUNE " + json.dumps({"per_layer_experts": True, "min": values[0], "max": values[-1],
                                          "total": sum(values), "block": f"{cls.__module__}.{cls.__qualname__}"}),
        flush=True)
    return True


def prepare_config(vllm_config, the_plan=None, log=print, moe_block=None):
    """Set the text config's ``num_experts`` to the kept count before the model is built (idempotent)."""
    p = the_plan or settings()
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is not None:
        # Spec-decode pilot (30 September): an MTP draft is allowed when it builds from its own model config (the
        # pinned _make_draft_vllm_config: speculative_config.draft_model_config, a separate HF config that keeps the
        # checkpoint's 512 experts; its mtp.* weights never match NAME). A layer-aware mask's per-layer MoE sizes
        # leave the draft's MoE block (prefix ...mtp.layers.N.mlp, unmarked config) at its own count.
        draft = getattr(speculative, "draft_model_config", None)
        draft_text = getattr(draft, "hf_text_config", None)
        if (draft_text is None or draft_text is vllm_config.model_config.hf_text_config
                or getattr(draft_text, "num_experts", None) != p["experts"]):
            raise RuntimeError("expert pruning covers the target's layers only: serve with MTP-0 (or an MTP draft "
                               "with its own, unpruned model config)")
    config = vllm_config.model_config.hf_text_config
    current = getattr(config, "num_experts", None)
    if current == p["kept_per_layer"]:
        return
    if current != p["experts"]:
        raise RuntimeError(f"expert pruning expects {p['experts']} routed experts, the config has {current}")
    layers = getattr(config, "num_hidden_layers", None)
    if layers is not None and sorted(p["kept"]) != list(range(layers)):
        raise RuntimeError(f"the pruning plan covers layers {sorted(p['kept'])[:3]}..., the model has {layers}")
    # The checkpoint's count stays readable: the pinned exact NVFP4-PLE FP8 gate (ple_layer.py, patched by the
    # serving-kernels PLE overlay for ple_uva) checks num_experts == 512 when the worker builds the PLE itself.
    config.arc_unpruned_num_experts = current
    config.num_experts = p["kept_per_layer"]
    if not p.get("uniform", True):
        install_moe_sizes(p["kept_counts"], moe_block or resolve(MOE_BLOCK), log=log)
    if p.get("stall_dump_seconds"):
        install_stall_dump(p["stall_dump_seconds"], worker_class(vllm_config), log=log)
    if os.environ.get(MEMORY_WATCH_ENV) == "1":
        install_memory_watch(worker_class(vllm_config), log=log)
    log("ARC_EXPERT_PRUNE " + json.dumps({"num_experts": p["kept_per_layer"], "from": p["experts"],
                                          "layers": len(p["kept"]), "uniform": p.get("uniform", True),
                                          "kept_total": sum(p.get("kept_counts", {}).values())}), flush=True)


def filter_weights(weights, the_plan=None, log=print):
    """The checkpoint stream with pruned experts dropped, kept ones renumbered and router rows sliced."""
    p = the_plan or settings()
    stats = {"dropped": 0, "renamed": 0, "routers": 0}
    for name, tensor in weights:
        match = NAME.match(name)
        if match is None:
            yield name, tensor
            continue
        layer = int(match.group(1))
        if layer not in p["remap"]:
            raise RuntimeError(f"{name}: no pruning plan for layer {layer}")
        if match.group(2) is None:
            if tensor.shape[0] != p["experts"]:
                raise RuntimeError(f"{name}: {tuple(tensor.shape)}, expected {p['experts']} router rows")
            stats["routers"] += 1
            if hasattr(tensor, "index_select"):
                import torch

                index = torch.tensor(p["kept"][layer], dtype=torch.long, device=tensor.device)
                yield name, tensor.index_select(0, index).contiguous()
            else:  # numpy (CPU tests)
                yield name, tensor[p["kept"][layer]]
            continue
        new = p["remap"][layer].get(int(match.group(2)))
        if new is None:
            stats["dropped"] += 1
            continue
        stats["renamed"] += 1
        yield f"model.language_model.layers.{layer}.mlp.experts.{new}.{match.group(3)}", tensor
    if stats["renamed"] and stats["routers"] != len(p["remap"]):
        raise RuntimeError(f"expert pruning saw {stats['routers']} of {len(p['remap'])} routers")
    log("ARC_EXPERT_PRUNE " + json.dumps(stats), flush=True)
