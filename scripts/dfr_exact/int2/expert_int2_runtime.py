"""ARC INT2 cold half of the routed experts, serving side (8 October; throughput arm LOG section 21).

Copied into the pinned runtime as ``vllm/arc_expert_int2.py`` by ``arc_infra.expert_int2_patch`` (with
``serving_kernels.expert_int2`` on top of ``expert_int3``). An INT2-cold overlay stores some experts of every layer
(``overlay-manifest.json`` ``settings.int2_experts``, checkpoint expert ids) on the int2 grid inside the int3 /
NVFP4 container. One Humming tensor has one weight dtype, so each layer's experts are served as two tensors per
projection, hot experts as uint3 and cold experts as uint2, with two indexed launches per projection:

- ``repack`` (after the checkpoint's NVFP4 conversion; calls the int3 repack) also encodes the cold experts as uint2
  (int2 = code - 2, a group holding +2 negated) and keeps them aside. A listed expert that is not on the int2 grid
  raises: the wrong overlay cannot be served.
- ``split`` (before Humming prepares the sublayer) leaves only the hot experts in the layer's own tensors; ``cold``
  (after it) prepares the cold tensor the same way and registers it on the layer.
- ``lookup`` / ``Split.apply`` (``HummingIndexedExperts.apply``): the routing is aligned once per part with an
  expert map (the other part's experts are invalid and skipped); both parts write the same output rows.

The cold list comes from ``arc_expert_int2.json`` next to this file (written at server start from the overlay's
manifest), mapped to the kept experts' ranks by ``vllm.arc_expert_prune`` when the server prunes.
"""

import json
import os
import re

SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_expert_int2.json")
PRUNE_SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_expert_prune.json")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
BITS = 2
_COLD = None
_SPLITS = {}  # id(the hot w13 layer config) -> Split


def log(obj):
    print("ARC_EXPERT_INT2 " + json.dumps(obj, sort_keys=True), flush=True)


def cold_plan(experts, remap=None):
    """{layer: sorted served expert ranks} of ``{layer: [checkpoint expert ids]}``; ``remap`` is the pruning plan's
    ``{layer: {checkpoint id: rank}}`` (a pruned cold expert is an error: the overlay belongs to another mask)."""
    plan = {}
    for layer, ids in experts.items():
        layer, ids = int(layer), [int(e) for e in ids]
        if len(set(ids)) != len(ids):
            raise ValueError(f"layer {layer}: repeated int2 expert ids")
        if remap is not None:
            missing = [e for e in ids if e not in remap[layer]]
            if missing:
                raise ValueError(f"layer {layer}: int2 experts {missing[:5]} are pruned by the server's mask")
            ids = [remap[layer][e] for e in ids]
        plan[layer] = sorted(ids)
    return plan


def settings():
    global _COLD
    if _COLD is None:
        with open(SETTINGS) as handle:
            raw = json.load(handle)
        remap = None
        if os.path.exists(PRUNE_SETTINGS):
            from vllm import arc_expert_prune

            remap = arc_expert_prune.settings()["remap"]
        _COLD = cold_plan(raw["experts"], remap)
    return _COLD


def cold_slots(layer_name, num_experts, plan=None):
    """The cold experts (served ranks) of the routed-experts layer ``layer_name``; [] for the draft's layer."""
    plan = settings() if plan is None else plan
    match = LAYER.search(layer_name)
    if match is None:
        raise RuntimeError(f"INT2 cold half: no layer index in {layer_name!r}")
    layer = int(match.group(1))
    if layer_name.startswith("mtp.") or ".mtp." in layer_name:
        return []
    if layer not in plan:
        raise RuntimeError(f"INT2 cold half: the overlay lists no int2 experts for layer {layer}")
    cold = plan[layer]
    if not cold or cold[0] < 0 or cold[-1] >= num_experts:
        raise RuntimeError(f"INT2 cold half: layer {layer} cold ranks outside 0..{num_experts - 1}")
    if len(cold) >= num_experts:
        raise RuntimeError(f"INT2 cold half: layer {layer} has no hot expert")
    return cold


def grid_codes(values, scales16, group, bits):
    """(unsigned codes int32 [..., K], signed group scales [..., K/group]) of values [..., K] on the ``bits``-bit grid
    (the integers -h..h-1 per group, h = 2^(bits-1), after negating a group that holds +h) and their per-16 scales."""
    import torch

    half = 1 << (bits - 1)
    if not torch.equal(values, values.round()) or float(values.abs().max()) > half:
        raise ValueError(f"routed experts are not on the int{bits} grid (non-integer or |v| > {half} values)")
    k = values.shape[-1]
    g = values.reshape(*values.shape[:-1], k // group, group)
    pos = (g == half).any(-1)
    if bool((pos & (g == -half).any(-1)).any()):
        raise ValueError(f"an int{bits} group holds both +{half} and -{half}")
    per = group // 16
    s = scales16.reshape(*scales16.shape[:-1], k // group, per)
    if per > 1 and not bool((s == s[..., :1]).all()):
        raise ValueError(f"per-16 scales differ inside a {group}-group")
    sign = torch.where(pos, -1.0, 1.0)
    codes = (g * sign.unsqueeze(-1) + half).to(torch.int32).reshape(values.shape)
    return codes, (s[..., 0].float() * sign).to(scales16.dtype)


def _humming():
    from humming import dtypes, ops

    return dtypes, ops


def _select(tensors, index, num_experts, skip=()):
    """Every per-expert tensor (first dimension = experts) at ``index``; anything else is not a layout this knows."""
    out = {}
    for name, tensor in tensors.items():
        if name in skip:
            continue
        if tensor.dim() < 1 or tensor.shape[0] != num_experts:
            raise RuntimeError(f"INT2 cold half: tensor {name} {tuple(tensor.shape)} is not per expert")
        out[name] = tensor.index_select(0, index).contiguous()
    return out


def repack(schema, tensors, num_experts, layer, sublayer):
    """The int3 repack of every expert (so the pinned conversion sees one consistent tensor), plus the cold experts'
    uint2 tensors kept on the layer for ``split`` / ``cold``."""
    import dataclasses

    import torch

    from vllm.arc_expert_int3 import repack as repack_int3
    from vllm.arc_expert_int3 import settings as int3_settings
    from vllm.arc_expert_int3 import unpack_e2m1

    new_schema, out = repack_int3(schema, tensors, num_experts)
    cfg = int3_settings()
    if cfg is None or not num_experts:
        raise RuntimeError("INT2 cold half needs the int3 repack (ARC_EXPERT_INT3) on a routed-experts layer")
    slots = cold_slots(layer.layer_name, num_experts)
    if not slots:
        return new_schema, out
    if new_schema is schema:
        raise RuntimeError(f"INT2 cold half: {layer.layer_name} {sublayer} is not an int3 NVFP4 container")
    dtypes, ops = _humming()
    weight = tensors["weight"].contiguous().view(torch.uint8)
    index = torch.tensor(slots, dtype=torch.long, device=weight.device)
    packed, scales = [], []
    for start in range(0, len(slots), 32):
        chosen = index[start:start + 32]
        values = unpack_e2m1(weight.index_select(0, chosen))
        codes, scale = grid_codes(values, tensors["weight_scale"].index_select(0, chosen), cfg["group"], BITS)
        packed.append(ops.pack_weight(codes.contiguous(), BITS))
        scales.append(scale)
        del values, codes
    part = _select(tensors, index, num_experts, skip=("weight", "weight_scale"))
    part["weight"] = torch.cat(packed)
    part["weight_scale"] = torch.cat(scales).contiguous()
    state = layer.__dict__.setdefault("_arc_int2", {})
    state[sublayer] = {"cold": slots, "tensors": part,
                       "schema": dataclasses.replace(new_schema, b_dtype=getattr(dtypes, f"uint{BITS}"))}
    return new_schema, out


def split(layer, sublayer, num_experts, extract, replace):
    """Before Humming prepares ``sublayer``: the layer keeps its hot experts only. Returns their count."""
    import torch

    state = getattr(layer, "_arc_int2", {}).get(sublayer)
    if state is None:
        return num_experts
    cold = set(state["cold"])
    hot = [e for e in range(num_experts) if e not in cold]
    tensors = extract(layer, sublayer)
    index = torch.tensor(hot, dtype=torch.long, device=next(iter(tensors.values())).device)
    replace(layer, sublayer, _select(tensors, index, num_experts))
    state.update(hot=hot, experts=num_experts)
    return len(hot)


def cold(layer, sublayer, config, make_config, transform):
    """After Humming prepared the hot tensor of ``sublayer``: the cold one the same way, registered on the layer."""
    import torch

    state = getattr(layer, "_arc_int2", {}).get(sublayer)
    if state is None:
        return
    cold_config = make_config(state["schema"], len(state["cold"]))
    tensors = transform(cold_config, state.pop("tensors"))
    names = {}
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor):
            continue
        attr = f"arc_int2_{sublayer}_{name}"
        setattr(layer, attr, torch.nn.Parameter(tensor, requires_grad=False))
        names[name] = attr
    state.update(config=cold_config, hot_config=config, names=names,
                 bytes=sum(t.numel() * t.element_size() for t in tensors.values() if torch.is_tensor(t)))
    done = layer._arc_int2
    if "w13" in done and "w2" in done and all("config" in done[s] for s in ("w13", "w2")):
        if done["w13"]["cold"] != done["w2"]["cold"] or done["w13"]["experts"] != done["w2"]["experts"]:
            raise RuntimeError("INT2 cold half: the two projections disagree on the cold experts")
        _SPLITS[id(done["w13"]["hot_config"])] = Split(layer)
        log({"layer": layer.layer_name, "experts": done["w13"]["experts"], "cold": len(done["w13"]["cold"]),
             "hot": len(done["w13"]["hot"]), "cold_bytes": done["w13"]["bytes"] + done["w2"]["bytes"]})


def lookup(experts):
    """The split of the layer an experts kernel serves (by its hot w13 config), or None. A kernel whose tensors
    hold fewer experts than its layer routes to, with no split to serve them, must not run."""
    found = experts.__dict__.get("_arc_int2", False)
    if found is False:
        config = experts.humming_configs["w13"]
        found = _SPLITS.get(id(config))
        if found is not None and found.hot_config is not config:
            found = None
        if found is None and config.num_experts != experts.num_experts:
            raise RuntimeError("INT2 cold half: an experts kernel lost its layer's split")
        experts.__dict__["_arc_int2"] = found
    return found


def check(experts):
    """At kernel construction: only the indexed kernel serves a split layer."""
    if lookup(experts) is not None and type(experts).__name__ != "HummingIndexedExperts":
        raise RuntimeError(f"INT2 cold half needs HummingIndexedExperts, got {type(experts).__name__}")


def _block(tuning, valid):
    for low, high, config in tuning:
        if low < valid <= high:
            return config["block_shape"][0]
    return 64


class Split:
    """One layer's two parts. ``apply`` is ``HummingIndexedExperts.apply`` with two launches per projection."""

    def __init__(self, layer):
        import torch

        state = layer._arc_int2
        self.layer, self.hot_config = layer, state["w13"]["hot_config"]
        experts, device = state["w13"]["experts"], getattr(layer, state["w13"]["names"]["weight"]).device
        self.maps = []
        for members in (state["w13"]["hot"], state["w13"]["cold"]):
            table = torch.full((experts,), -1, dtype=torch.int32, device=device)
            table[torch.tensor(members, dtype=torch.long, device=device)] = torch.arange(
                len(members), dtype=torch.int32, device=device)
            self.maps.append(table)
        self.cold = {s: {"config": state[s]["config"],
                         "tensors": {name: getattr(layer, attr) for name, attr in state[s]["names"].items()}}
                     for s in ("w13", "w2")}
        self.tuning = None

    def _tuning(self, experts):
        if self.tuning is None:
            from vllm import envs
            from vllm.utils.humming import get_heuristics_config

            self.tuning = {}
            for sublayer in ("w13", "w2"):
                value = get_heuristics_config(layer_config=self.cold[sublayer]["config"],
                                              use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
                                              use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
                                              gemm_type=experts.humming_gemm_type())
                self.tuning[sublayer] = (value, json.dumps(value))
        return self.tuning

    def _cold_forward(self, experts, sublayer, tuning, forward, **kwargs):
        part = self.cold[sublayer]
        tensors = part["tensors"]
        return forward(part["config"], weight=tensors["weight"], weight_scale=tensors.get("weight_scale"),
                       zero_point=tensors.get("zero_point"), bias=None,
                       weight_scale_2=tensors.get("weight_scale_2"), locks=experts.locks,
                       tuning_config=tuning[sublayer][1], **kwargs)

    def apply(self, experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, expert_map,
              workspace13, workspace2, align=None, forward=None, mul_sum=None):
        if expert_map is not None:
            raise RuntimeError("INT2 cold half serves unmapped experts only (no expert parallelism)")
        if align is None:
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size as align
            from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum as mul_sum
            from vllm.utils.humming import humming_forward as forward
        tuning = self._tuning(experts)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        top_k = topk_ids.size(1)
        buffers = experts.prepare_buffers(workspace13, workspace2, topk_ids.size(0), top_k, activation)
        buffers["output"] = output
        valid = experts.estimate_local_valid_shape_m(topk_ids)
        aligned = []
        for table, own in zip(self.maps, (experts.w13_tuning_config, tuning["w13"][0])):
            sorted_ids, expert_ids, padded = align(topk_ids=topk_ids, block_size=_block(own, valid),
                                                   num_experts=experts.global_num_experts, expert_map=table,
                                                   ignore_invalid_experts=True)
            aligned.append({"sorted_ids": sorted_ids, "expert_ids": expert_ids, "num_tokens_padded": padded,
                            "compute_config": experts.compute_config_str, "valid_shape_m": valid})
        inputs, input_scale = experts.quantize_input("w13", inputs=hidden_states,
                                                     quanted_input=buffers.get("quanted_gate_up_input", None))
        experts.humming_forward("w13", inputs=inputs, weight=w1, input_scale=input_scale,
                                outputs=buffers["gate_up_output"], top_k=top_k,
                                tuning_config=experts.w13_tuning_config_str, **aligned[0])
        self._cold_forward(experts, "w13", tuning, forward, inputs=inputs, input_scale=input_scale,
                           outputs=buffers["gate_up_output"], top_k=top_k, **aligned[1])
        experts.apply_activation(activation=activation, input=buffers["gate_up_output"],
                                 output=buffers["activation_output"])
        inputs, input_scale = experts.quantize_input("w2", inputs=buffers["activation_output"],
                                                     quanted_input=buffers.get("quanted_down_input", None))
        down = buffers["down_output"].view(-1, hidden_states.size(-1))
        experts.humming_forward("w2", inputs=inputs, weight=w2, input_scale=input_scale, outputs=down, top_k=1,
                                tuning_config=experts.w2_tuning_config_str, **aligned[0])
        self._cold_forward(experts, "w2", tuning, forward, inputs=inputs, input_scale=input_scale, outputs=down,
                           top_k=1, **aligned[1])
        mul_sum(inputs=buffers["down_output"].view(*topk_ids.shape, -1), topk_weights=topk_weights,
                topk_ids=topk_ids, expert_map=None, outputs=output)
