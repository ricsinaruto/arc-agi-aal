"""Expert mask, worker side (hero run, copied to ``vllm/arc_expert_mask.py`` by ``expert_mask_patch``).

The patched ``MoERunner._forward_impl`` hands every MoE layer's router logits to ``on_router_logits`` before routing
(the expert_probe anchor). A pruned expert's logit gets ``MASK_VALUE``, so the router's top-k and its renormalized
weights see the kept experts only: the routing of a renumbered cut (``expert_prune_runtime``) with the checkpoint's
expert ids, so LoRA adapters and served-route (R3) tables keep their layout. The mask (``arc_expert_mask.json`` next to
this file: ``experts``, ``pruned`` per layer, ``label``) is a device buffer built at the first MoE call (the profile
run, eager) and read by every later call, CUDA graphs included; no statistics, no host sync in the hot path. A fused
router + shared-expert gate width (experts + 1) masks the first ``experts`` columns only.

RPC ``arc_expert_mask`` (``/collective_rpc``, ``{"op": "status"}``): the label, masked entries and layers seen, so a
client can confirm the mask after a start or a wake.
"""

import json
import os

import torch

SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_expert_mask.json")
MASK_VALUE = -1.0e4  # as expert_probe_runtime and the trainer (rl1_trainer.MASK_VALUE)
_STATE = None


class _State:
    def __init__(self, device, spec):
        self.experts = int(spec["experts"])
        layers = max(int(k) for k in spec["pruned"]) + 1
        self.mask = torch.zeros(layers, self.experts + 1, dtype=torch.float32, device=device)
        for layer, ids in spec["pruned"].items():
            if ids:
                self.mask[int(layer), torch.tensor([int(e) for e in ids], device=device)] = MASK_VALUE
        self.label, self.sha256 = spec.get("label"), spec.get("sha256")
        self.pruned = sum(len(v) for v in spec["pruned"].values())
        self.layers = set()


def _install_rpc():
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    if getattr(WorkerWrapperBase, "arc_expert_mask", None) is not arc_expert_mask:
        WorkerWrapperBase.arc_expert_mask = arc_expert_mask


def settings():
    with open(SETTINGS) as f:
        return json.load(f)


def on_router_logits(runner, logits):
    global _STATE
    if _STATE is None:
        if logits.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("expert mask: first MoE call inside a CUDA graph capture")
        _STATE = _State(logits.device, settings())
        _install_rpc()
    state = _STATE
    if runner.__dict__.get("_arc_mask_draft") or "mtp" in str(getattr(runner, "layer_name", "")):
        runner.__dict__["_arc_mask_draft"] = True  # the MTP draft's experts are never masked (nor compacted)
        return logits
    width = logits.shape[-1]
    if logits.dim() != 2 or width not in (state.experts, state.experts + 1):
        raise RuntimeError(f"expert mask: router logits {tuple(logits.shape)}, the mask has {state.experts} experts")
    layer = runner.__dict__.get("_arc_mask_layer")
    if layer is None:
        layer = int(runner.layer_id)
        runner.__dict__["_arc_mask_layer"] = layer
        if not 0 <= layer < state.mask.shape[0]:
            raise RuntimeError(f"expert mask: no row for MoE layer {layer}")
        state.layers.add(layer)
    return logits + state.mask[layer, :width].to(logits.dtype)


PROFILE = {"installed": False, "loads": [], "expert": None}
_ITERATORS = ("safetensors_weights_iterator", "multi_thread_safetensors_weights_iterator", "fastsafetensors_weights_iterator")


def _clock():
    """(wall, thread CPU, thread major faults): cheap enough for every tensor of a load."""
    import resource
    import time

    who = getattr(resource, "RUSAGE_THREAD", resource.RUSAGE_SELF)
    return time.perf_counter(), time.thread_time(), resource.getrusage(who).ru_majflt


def _acc(row, began):
    now = _clock()
    row["n"] += 1
    for i, key in enumerate(("wall", "cpu", "majflt")):
        row[key] += now[i] - began[i]


STACKS_ENV = "ARC_LOAD_STACKS"  # "1": sample the loading thread's stack over each profiled load (H attempt 2, 26 Oct:
# rest_s was 54-57 s of a 77 s weights pass, the loader's own per-tensor work; py-spy cannot attach on Modal: EPERM)


class StackSampler:
    """A daemon thread sampling ``thread_id``'s Python stack every ``interval`` s (sys._current_frames): self and
    inclusive sample counts per function (file:function), the top ``top`` of each in ``summary()``."""

    def __init__(self, thread_id, interval=0.005, top=25):
        import threading

        self.thread_id, self.interval, self.top = thread_id, interval, top
        self.own, self.total, self.samples = {}, {}, 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @staticmethod
    def key(frame):
        code = frame.f_code
        return f"{os.path.basename(code.co_filename)}:{code.co_name}"

    def run(self):
        import sys

        while not self.stop.wait(self.interval):
            frame = sys._current_frames().get(self.thread_id)
            if frame is None:
                continue
            self.samples += 1
            leaf = self.key(frame)
            self.own[leaf] = self.own.get(leaf, 0) + 1
            seen = set()
            while frame is not None:
                k = self.key(frame)
                if k not in seen:
                    seen.add(k)
                    self.total[k] = self.total.get(k, 0) + 1
                frame = frame.f_back

    def start(self):
        self.thread.start()
        return self

    def summary(self):
        self.stop.set()
        self.thread.join(timeout=5)
        n = max(self.samples, 1)

        def top(counts):
            return [[k, round(v / n, 3)] for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:self.top]]

        return {"samples": self.samples, "interval_s": self.interval, "self": top(self.own), "inclusive": top(self.total)}


def _profiled(fn, name):
    """A weights iterator timed per yield (the read side: mmap get_tensor, page faults) and as a whole; the expert
    loader's share is taken from PROFILE["expert"] over the iterator's lifetime. One ARC_LOAD_PROFILE line per load."""
    import functools
    import time

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import threading

        sampler = StackSampler(threading.get_ident()).start() if os.environ.get(STACKS_ENV) == "1" else None
        read = {"n": 0, "wall": 0.0, "cpu": 0.0, "majflt": 0}
        exp0 = dict(PROFILE["expert"]) if PROFILE["expert"] else None
        began, gen = time.perf_counter(), iter(fn(*args, **kwargs))
        try:
            while True:
                t = _clock()
                try:
                    item = next(gen)
                except StopIteration:
                    break
                _acc(read, t)
                yield item
        finally:
            exp = PROFILE["expert"]
            row = {"iterator": name, "seconds": round(time.perf_counter() - began, 2), "tensors": read["n"],
                   "read": {k: round(v, 2) for k, v in read.items() if k != "n"}}
            if exp is not None:
                row["expert_loader"] = {k: round(exp[k] - (exp0 or {}).get(k, 0), 2) for k in exp}
            row["rest_s"] = round(row["seconds"] - read["wall"] - row.get("expert_loader", {}).get("wall", 0.0), 2)
            if sampler is not None:
                row["stacks"] = sampler.summary()
            PROFILE["loads"].append(row)
            print("ARC_LOAD_PROFILE " + json.dumps(row, sort_keys=True), flush=True)

    wrapper._arc_profiled = True
    return wrapper


def install_load_profile(weight_utils=None, loaders=(), fused=None):
    """ARC_LOAD_PROFILE: time the weight iterators (also where a loader module imported them by name) and
    FusedMoE.weight_loader (wall, thread CPU, major faults). Exact: values pass through unchanged. A measurement
    only: an install failure is printed (ARC_LOAD_PROFILE_ERROR) and the load goes on unprofiled."""
    try:
        return _install_load_profile(weight_utils, loaders, fused)
    except Exception as error:  # noqa: BLE001
        print("ARC_LOAD_PROFILE_ERROR " + json.dumps({"error": repr(error)[:300]}), flush=True)
        return False


def _install_load_profile(weight_utils, loaders, fused):
    import functools
    import sys

    if PROFILE["installed"] or not settings().get("load_profile"):
        return False
    if weight_utils is None:
        import vllm.model_executor.model_loader.weight_utils as weight_utils
        loaders = [m for n, m in list(sys.modules.items()) if n.startswith("vllm.model_executor.model_loader.")]
    for name in _ITERATORS:
        orig = getattr(weight_utils, name, None)
        if orig is None or getattr(orig, "_arc_profiled", False):
            continue
        wrapped = _profiled(orig, name)
        setattr(weight_utils, name, wrapped)
        for mod in loaders:
            if getattr(mod, name, None) is orig:
                setattr(mod, name, wrapped)
    if fused is None:  # the pinned runtime (0.1.dev20073): RoutedExperts holds the expert params and their loader
        try:
            from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts as fused
        except ImportError:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE as fused
    orig_loader = fused.weight_loader
    PROFILE["expert"] = {"n": 0, "wall": 0.0, "cpu": 0.0, "majflt": 0}

    @functools.wraps(orig_loader)  # keeps supports_moe_loading (model loaders read it off the bound method)
    def weight_loader(self, *args, **kwargs):
        t = _clock()
        try:
            return orig_loader(self, *args, **kwargs)
        finally:
            _acc(PROFILE["expert"], t)

    fused.weight_loader = weight_loader
    _install_postload_profile(loaders)
    PROFILE["installed"] = True
    return True


def _class_timer(cls, by_class):
    """cls.process_weights_after_loading (its own, not inherited) timed into by_class[cls.__name__] (wall s)."""
    import functools
    import time

    orig = cls.__dict__.get("process_weights_after_loading")
    if orig is None or getattr(orig, "_arc_timed", False):
        return

    @functools.wraps(orig)
    def timed(self, *args, **kwargs):
        began = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            row = by_class.setdefault(cls.__name__, [0, 0.0])
            row[0] += 1
            row[1] += time.perf_counter() - began

    timed._arc_timed = True
    cls.process_weights_after_loading = timed


def _install_postload_profile(loaders):
    """ARC_LOAD_PROFILE: the post-load pass (quant repacks: Marlin NVFP4, FP8 PLE, ...) timed per quant-method class.
    One ARC_POSTLOAD_PROFILE line per call (wall total, top classes). Values pass through unchanged."""
    import functools
    import time

    try:
        import vllm.model_executor.model_loader.utils as utils
    except ImportError:  # tests without vLLM
        return
    orig = utils.process_weights_after_loading
    if getattr(orig, "_arc_timed", False):
        return

    @functools.wraps(orig)
    def timed(model, *args, **kwargs):
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

        by_class, seen, todo = {}, set(), [QuantizeMethodBase]
        while todo:  # every quant-method class loaded by now
            c = todo.pop()
            for sub in c.__subclasses__():
                if sub not in seen:
                    seen.add(sub)
                    todo.append(sub)
                    _class_timer(sub, by_class)
        PROFILE["postload"] = by_class
        began = time.perf_counter()
        try:
            return orig(model, *args, **kwargs)
        finally:
            top = sorted(((k, v) for k, v in by_class.items()), key=lambda kv: -kv[1][1])[:8]
            print("ARC_POSTLOAD_PROFILE " + json.dumps({"seconds": round(time.perf_counter() - began, 2), "classes": {
                k: {"n": v[0], "s": round(v[1], 2)} for k, v in top}}, sort_keys=True), flush=True)

    timed._arc_timed = True
    utils.process_weights_after_loading = timed
    for mod in loaders:
        if getattr(mod, "process_weights_after_loading", None) is orig:
            mod.process_weights_after_loading = timed


# ---- ARC_EXPERT_MASK_COMPACT (MFU-2, 24 Oct): masked experts are not allocated -----------------------------------
# The router, its mask and the served routes keep the checkpoint's 512 expert ids. Each compacted MoE layer holds
# only its kept experts, in ascending id order (local slots 0..kept-1; the manager reports kept as its local count, no
# EP map). ``compact_ids`` turns the routed ids into slots after routing (the routes are captured before it). The
# loader drops a masked expert's tensors (its block-scale max is kept: Marlin's shared scale factor is a max over
# every expert). Kernel heuristics that read the expert count see the checkpoint's count (``marlin_compact``,
# ``moe_shape``, ``moe_experts``), so every kernel choice matches the masked path. LoRA adapters (512 experts) are cut
# to the kept rows (``expert_select``). Uniform masks only (one kept count for every layer).
COMPACT = {"plan": None, "layers": set(), "dropped": 0, "scale_max": 0, "ids_tables": 0}
_LAYER = None


def compact_plan():
    """None without ``compact`` in the settings; else experts, kept count and per-layer kept ids / slot tables."""
    if COMPACT["plan"] is None:
        spec = settings()
        if not spec.get("compact"):
            COMPACT["plan"] = False
        else:
            experts = int(spec["experts"])
            kept, table = {}, {}
            for layer, ids in spec["pruned"].items():
                drop = {int(e) for e in ids}
                keep = [e for e in range(experts) if e not in drop]
                slots = [0] * experts  # a masked id is never routed (its logit is MASK_VALUE); slot 0 is a safe index
                for i, e in enumerate(keep):
                    slots[e] = i
                kept[int(layer)], table[int(layer)] = keep, slots
            counts = {len(v) for v in kept.values()}
            if len(counts) != 1:
                raise RuntimeError(f"expert mask compact: needs one kept count for every layer, got {sorted(counts)}")
            COMPACT["plan"] = {"experts": experts, "kept": counts.pop(), "ids": kept, "slots": table,
                               "masked": {k: set(range(experts)) - set(v) for k, v in kept.items()}}
    return COMPACT["plan"] or None


def layer_of(name):
    """The decoder layer index of a routed-experts module or weight name (``...layers.<l>.mlp.experts...``); None for
    anything else (the MTP draft's experts included)."""
    global _LAYER
    if _LAYER is None:
        import re
        _LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts(?:\.|$)")
    if not isinstance(name, str) or "mtp" in name:
        return None
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


def compact_manager(manager, layer_name):
    """layer.py, right after the layer's ExpertMapManager: a masked layer reports its kept count as local experts."""
    plan = compact_plan()
    layer = layer_of(layer_name)
    if plan is None or layer is None or layer not in plan["ids"]:
        return False
    if manager.global_num_experts != plan["experts"] or manager.use_ep or manager._expert_map is not None:
        raise RuntimeError(f"expert mask compact: {layer_name} has {manager.global_num_experts} experts / EP on")
    manager._local_num_experts = plan["kept"] + manager.num_fused_shared_experts
    manager.arc_layer = layer
    COMPACT["layers"].add(layer)
    return True


def _slot_table(routed, plan, layer, device, dtype):
    key = ("_arc_slots", str(device), dtype)
    table = routed.__dict__.get(key)
    if table is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("expert mask compact: first MoE call inside a CUDA graph capture")
        extra = routed.expert_map_manager.num_fused_shared_experts  # fused shared experts follow the kept slots
        rows = plan["slots"][layer] + [plan["kept"] + i for i in range(extra)]
        table = torch.tensor(rows, dtype=dtype, device=device)
        routed.__dict__[key] = table
        COMPACT["ids_tables"] += 1
    return table


def compact_ids(routed, topk_ids):
    """MoERunner, after routing: checkpoint expert ids -> this layer's slots (unchanged for an uncompacted layer)."""
    layer = getattr(routed.expert_map_manager, "arc_layer", None)
    if layer is None:
        return topk_ids
    table = _slot_table(routed, compact_plan(), layer, topk_ids.device, topk_ids.dtype)
    return table[topk_ids.long()]


def _record_scale_max(routed, kind, loaded_weight):
    """Max of the positive scales (as before: params_dtype, then float); one reduction instead of a masked copy
    (B300 load profile: 33 s over 27.6K masked scales). A NaN max takes the masked path."""
    m = float(loaded_weight.to(routed.params_dtype).max())
    if m != m:
        v = loaded_weight.to(routed.params_dtype).float()
        v = v[v > 0]
        m = float(v.max()) if v.numel() else 0.0
    if m > 0:
        store = routed.__dict__.setdefault("_arc_masked_scale_max", {})
        store[kind] = max(store.get(kind, 0.0), m)
        COMPACT["scale_max"] += 1


def install_compact(fused=None):
    """Wrap the routed experts' weight loader: a kept expert loads into its slot, a masked one is dropped (its
    block scales leave their max for ``marlin_scale_factor``)."""
    import functools

    if compact_plan() is None:
        return False
    if fused is None:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts as fused
    if getattr(fused.weight_loader, "_arc_compact", False):
        return False
    orig = fused.weight_loader

    @functools.wraps(orig)  # keeps supports_moe_loading
    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, *args, **kwargs):
        layer = getattr(self.expert_map_manager, "arc_layer", None)
        if layer is None:
            return orig(self, param, loaded_weight, weight_name, shard_id, expert_id, *args, **kwargs)
        plan = compact_plan()
        expert_id = int(expert_id)
        if expert_id >= plan["experts"]:  # a fused shared expert
            slot = plan["kept"] + expert_id - plan["experts"]
        elif expert_id in plan["masked"][layer]:
            if param is getattr(self, "w13_weight_scale", None):
                _record_scale_max(self, "w13", loaded_weight)
            elif param is getattr(self, "w2_weight_scale", None):
                _record_scale_max(self, "w2", loaded_weight)
            COMPACT["dropped"] += 1
            return_success = kwargs.get("return_success", args[0] if args else False)
            return True if return_success else None
        else:
            slot = plan["slots"][layer][expert_id]
        return orig(self, param, loaded_weight, weight_name, shard_id, slot, *args, **kwargs)

    weight_loader._arc_compact = True
    fused.weight_loader = weight_loader
    if __import__("os").environ.get("ARC_EXPERT_FAST_LOOKUP") == "1":
        install_fast_lookup(fused)
    return True


def mapping_candidates(qual, index, loose):
    """Positions of every mapping entry whose weight name is a substring of ``qual``: names starting "experts." and
    ending "." are looked up by every such substring of ``qual``; the rest (fused names) are tested directly."""
    out = set()
    start = qual.find("experts.")
    while start != -1:
        end = qual.find(".", start + 8)
        while end != -1:
            out.update(index.get(qual[start:end + 1], ()))
            end = qual.find(".", end + 1)
        start = qual.find("experts.", start + 1)
    out.update(pos for pos, name in loose if name in qual)
    return sorted(out)


def install_fast_lookup(routed=None):
    """ARC_EXPERT_FAST_LOOKUP=1 (B300 load profile: RoutedExperts.load_weights scans the whole expert mapping with
    substring tests for each of 295K tensors, 31 s own time): a per-expert (2-D) tensor runs the original loader over
    only its matching mapping entries, in mapping order; fused 3-D tensors keep the full scan. Exact: the skipped
    entries were non-matches the loop passed over without effect."""
    if routed is None:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts as routed
    if getattr(routed.load_weights, "_arc_fast", False):
        return False
    orig_load, orig_map = routed.load_weights, routed.get_expert_mapping

    def get_expert_mapping(self, *args, **kwargs):
        sub = self.__dict__.get("_arc_mapping_override")
        if sub is not None and not args and kwargs == {"include_fused": True}:
            return sub
        return orig_map(self, *args, **kwargs)

    def load_weights(self, weights):
        table = self.__dict__.get("_arc_mapping_table")
        if table is None:
            full = orig_map(self, include_fused=True)
            index, loose = {}, []
            for pos, entry in enumerate(full):
                name = entry[1]
                if name.startswith("experts.") and name.endswith("."):
                    index.setdefault(name, []).append(pos)
                else:
                    loose.append((pos, name))
            table = self.__dict__["_arc_mapping_table"] = (full, index, loose)
        full, index, loose = table
        for expert_name, loaded_weight in weights:
            if loaded_weight.dim() == 3:
                yield from orig_load(self, [(expert_name, loaded_weight)])
                continue
            qual = f"{self.layer_name}.{expert_name}"
            self.__dict__["_arc_mapping_override"] = [full[i] for i in mapping_candidates(qual, index, loose)]
            try:
                done = list(orig_load(self, [(expert_name, loaded_weight)]))
            finally:
                self.__dict__.pop("_arc_mapping_override", None)
            yield from done

    load_weights._arc_fast = True
    routed.get_expert_mapping = get_expert_mapping
    routed.load_weights = load_weights
    return True


def marlin_scale_factor(layer, name, scales, a_dtype, original):
    """Marlin NVFP4 prep: the shared power-of-2 scale factor over every expert, the masked ones included."""
    extra = (layer.__dict__.get("_arc_masked_scale_max") or {}).get("w13" if "w13" in name else "w2")
    if not extra:
        return original(scales, a_dtype)
    tail = torch.tensor([extra], dtype=scales.dtype, device=scales.device)
    return original(torch.cat([scales.reshape(-1), tail]), a_dtype)


def marlin_compact(local_experts, global_experts):
    """fused_marlin_moe: True when this call's weights are a compacted layer (block size picked as for all experts)."""
    plan = compact_plan()
    return plan is not None and local_experts == plan["kept"] and global_experts == plan["experts"]


def moe_shape(shape):
    """Punica MoE-LoRA config lookup: a compacted weight shape reads as the checkpoint's expert count."""
    plan = compact_plan()
    if plan is not None and len(shape) and shape[0] == plan["kept"]:
        return (plan["experts"], *tuple(shape)[1:])
    return shape


def moe_experts(local_experts):
    plan = compact_plan()
    return plan["experts"] if plan is not None and local_experts == plan["kept"] else local_experts


def compacted_module(module_name, module):
    plan = compact_plan()
    layer = layer_of(module_name)
    return (plan is not None and layer is not None and layer in plan["ids"] and not module.use_ep
            and module.local_num_experts == plan["kept"] and module.global_num_experts == plan["experts"])


def expert_select(module_name, module, start, end):
    """LoRA manager: the adapter rows this module keeps (a slice for EP / uncompacted modules)."""
    if not compacted_module(module_name, module):
        return slice(start, end)
    return torch.tensor(compact_plan()["ids"][layer_of(module_name)], dtype=torch.long)


SKIP = {"installed": False, "skipped": 0}
_NAME = None


def masked_weight(name, pruned):
    """True for a masked expert's packed weight (``...layers.<l>.mlp.experts.<e>.{gate,up,down}_proj.weight``);
    scales, input scales and every other tensor (the MTP head's experts too) are read."""
    global _NAME
    if _NAME is None:
        import re
        _NAME = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(?:gate|up|down)_proj\.weight$")
    m = _NAME.match(name)
    return bool(m) and int(m.group(2)) in pruned.get(int(m.group(1)), ())


def install_skip_read(module=None):
    """ARC_EXPERT_MASK_SKIP_READ: wrap the weight iterators' ``should_skip_weight`` (vllm weight_utils; the pinned
    iterator calls it too): a masked expert's packed weight is skipped (counted in SKIP), everything else as before."""
    if SKIP["installed"]:
        return False
    spec = settings()
    if not spec.get("skip_read"):
        return False
    if module is None:
        import vllm.model_executor.model_loader.weight_utils as module
    pruned = {int(k): {int(e) for e in v} for k, v in spec["pruned"].items()}
    orig = module.should_skip_weight

    def should_skip_weight(name, local_expert_ids):
        if orig(name, local_expert_ids):
            return True
        if masked_weight(name, pruned):
            SKIP["skipped"] += 1
            return True
        return False

    should_skip_weight.__wrapped__ = orig
    module.should_skip_weight = should_skip_weight
    SKIP.update(installed=True, expected=3 * sum(len(v) for v in pruned.values()))
    return True


def arc_expert_mask(self, command):
    cmd = json.loads(command) if isinstance(command, str) else dict(command)
    state = _STATE
    if cmd.get("op") != "status":
        return {"ok": False, "error": f"unknown op {cmd.get('op')!r}"}
    if state is None:
        return {"ok": False, "error": "mask not initialised (no MoE forward yet)"}
    out = {"ok": True, "label": state.label, "sha256": state.sha256, "pruned": state.pruned,
           "masked_entries": int((state.mask != 0).sum().item()), "layers": sorted(state.layers)}
    if PROFILE["installed"]:  # ARC_LOAD_PROFILE: one row per weights iterator (main model, draft)
        out["load_profile"] = PROFILE["loads"]
    if SKIP["installed"]:  # ARC_EXPERT_MASK_SKIP_READ: masked packed weights the loads skipped (3 per masked expert)
        out["skipped_reads"], out["skipped_expected"] = SKIP["skipped"], SKIP["expected"]
    if COMPACT["plan"]:  # ARC_EXPERT_MASK_COMPACT
        out["compact"] = {"kept": COMPACT["plan"]["kept"], "layers": len(COMPACT["layers"]),
                          "dropped_loads": COMPACT["dropped"], "scale_max_rows": COMPACT["scale_max"],
                          "slot_tables": COMPACT["ids_tables"]}
    return out
