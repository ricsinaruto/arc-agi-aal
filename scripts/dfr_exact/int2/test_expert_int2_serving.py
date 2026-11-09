"""INT2 cold half, serving side (arc_infra.expert_int2_runtime / expert_int2_patch). CPU only: Humming and the pinned
kernels are replaced by exact stand-ins, so the partition, the 2-bit codes and scales, the expert maps and the merge of
the two launches are checked against a dense reference; the torch parts are skipped without torch."""
import dataclasses
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arc_infra import expert_int2_patch as patch
from arc_infra import expert_int2_runtime as rt
from arc_infra import expert_int3_runtime as int3
from arc_infra import serving_kernels_patch as sk

PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/nonexistent"))
BASE = {"mamba_ssm_cache_dtype": "bfloat16", "moe_backend": "humming", "draft_experts": "nvfp4", "hc_fusion": False,
        "topk_sampler": True, "expert_int3": 16}


def test_cold_plan_maps_checkpoint_ids_to_served_ranks():
    assert rt.cold_plan({"3": [7, 2]}) == {3: [2, 7]}
    remap = {3: {2: 0, 5: 1, 7: 2}}
    assert rt.cold_plan({"3": [7, 2]}, remap) == {3: [0, 2]}
    with pytest.raises(ValueError, match="pruned"):
        rt.cold_plan({"3": [7, 4]}, remap)
    with pytest.raises(ValueError, match="repeated"):
        rt.cold_plan({"3": [7, 7]})


def test_cold_slots_by_layer_name():
    plan = {4: [1, 3], 5: [0, 1, 2, 3]}
    assert rt.cold_slots("language_model.model.layers.4.mlp.experts", 4, plan) == [1, 3]
    assert rt.cold_slots("mtp.layers.4.mlp.experts", 4, plan) == []  # the draft's own layer is not in the overlay
    with pytest.raises(RuntimeError, match="no hot expert"):
        rt.cold_slots("language_model.model.layers.5.mlp.experts", 4, plan)
    with pytest.raises(RuntimeError, match="lists no int2 experts"):
        rt.cold_slots("language_model.model.layers.6.mlp.experts", 4, plan)
    with pytest.raises(RuntimeError, match="outside"):
        rt.cold_slots("language_model.model.layers.4.mlp.experts", 3, plan)
    with pytest.raises(RuntimeError, match="no layer index"):
        rt.cold_slots("experts", 4, plan)


def test_option_needs_int3_and_leaves_recorded_settings_alone():
    assert "expert_int2" not in sk.validate(BASE)
    assert sk.validate({**BASE, "expert_int2": False}) == sk.validate(BASE)
    on = sk.validate({**BASE, "expert_int2": True})
    assert on["expert_int2"] is True
    with pytest.raises(ValueError, match="needs expert_int3"):
        sk.validate({**BASE, "expert_int3": None, "expert_int2": True})
    files = [t["relative"] for t in sk.targets(on)]
    assert files.count(patch.UTILS_TARGET["relative"]) == 1 and patch.MOE_TARGET["relative"] in files
    assert patch.MOE_TARGET["relative"] not in [t["relative"] for t in sk.targets(sk.validate(BASE))]


def test_settings_come_from_the_overlay_manifest(tmp_path):
    manifest = {"settings": {"grid": "int3", "int2_experts": {"1": [9, 4], "0": [3]}}}
    data = json.loads(patch.settings_bytes(manifest, "abc"))
    assert data == {"experts": {"0": [3], "1": [4, 9]}, "overlay_manifest_sha256": "abc"}
    (tmp_path / "vllm").mkdir()
    identity = patch.write_settings(tmp_path, manifest, "abc")
    assert identity["int2_experts"] == 3 and (tmp_path / patch.SETTINGS).read_bytes() == patch.settings_bytes(
        manifest, "abc")
    with pytest.raises(RuntimeError, match="INT2-cold"):
        patch.settings_bytes({"settings": {"grid": "int3"}}, "abc")


def test_patches_apply_to_the_pinned_sources():
    utils = PINNED_SITE / patch.UTILS_TARGET["relative"]
    if not utils.exists():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    text = sk.patched_source(patch.UTILS_TARGET, utils.read_bytes()).decode()
    assert (text.index("def _convert_sublayer_to_humming(") < text.index("_arc_int2_repack(")
            < text.index("def _prepare_and_transform_sublayer(") < text.index("_arc_int2.split(")
            < text.index("    config = prepare_layer_config(", text.index("_arc_int2.split("))
            < text.index("_arc_int2.cold(")
            < text.index("def _process_single_sublayer("))
    assert "arc_expert_int3 import" not in text  # the int2 helper calls the int3 repack itself
    moe = sk.patched_source(patch.MOE_TARGET, (PINNED_SITE / patch.MOE_TARGET["relative"]).read_bytes()).decode()
    assert (moe.index("class HummingExpertsBase(") < moe.index("_arc_int2_check(self)")
            < moe.index("class HummingIndexedExperts(") < moe.index("_arc_split.apply(")
            < moe.index("class HummingGroupedExperts("))


# ---------------------------------------------------------------------------------------------- torch stand-ins
E2M1_CODE = {0: 0, 1: 2, 2: 4, 3: 5, 4: 6}


def pack_e2m1(torch, values):
    """Integer values in -4..4 [..., K] as E2M1 nibbles [..., K/2] (first value in the low nibble, sign bit 3)."""
    codes = torch.tensor([E2M1_CODE[abs(int(v))] | (8 if v < 0 else 0) for v in values.flatten().tolist()],
                         dtype=torch.uint8).reshape(values.shape)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


@dataclasses.dataclass(frozen=True)
class Schema:
    b_dtype: object
    weight_scale_group_size: int


class Layer:
    """The slice of RoutedExperts the patch touches: named parameters per sublayer."""

    def __init__(self, torch, name, sublayers):
        self.torch, self.layer_name, self.params = torch, name, {}
        for sub, tensors in sublayers.items():
            for key, value in tensors.items():
                self.params[f"{sub}_{key}"] = value

    def __getattr__(self, name):
        params = self.__dict__.get("params", {})
        if name in params:
            return params[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if "params" in self.__dict__ and hasattr(value, "requires_grad"):
            self.params[name] = value.data
        else:
            self.__dict__[name] = value


def extract(layer, sub):
    return {k.removeprefix(sub + "_"): v for k, v in layer.params.items() if k.startswith(sub + "_")}


def replace(layer, sub, tensors):
    for key in [k for k in layer.params if k.startswith(sub + "_")]:
        del layer.params[key]
    for key, value in tensors.items():
        layer.params[f"{sub}_{key}"] = value


def dense(torch, weight, scale, scale2, bits):
    """[n, N, K] float weights of codes [n, N, K] with signed per-16 scales [n, N, K/16] and scale_2 [n]."""
    values = (weight.float() - (1 << (bits - 1))).reshape(*scale.shape, 16)
    return (values * scale.float().unsqueeze(-1)).reshape(weight.shape) * scale2.float().view(-1, 1, 1)


def forward(config, *, inputs, weight, weight_scale, weight_scale_2, outputs, top_k, sorted_ids, expert_ids,
            num_tokens_padded, **_):
    """The indexed GEMM: row ``flat`` of the outputs is its expert's weight times its token's input row."""
    import torch

    weights = dense(torch, weight, weight_scale, weight_scale_2, config.bits)
    assert weights.shape[0] == config.num_experts
    for flat, expert in zip(sorted_ids.tolist()[:num_tokens_padded], expert_ids.tolist()):
        outputs[flat] = weights[expert] @ inputs[flat // top_k if top_k > 1 else flat]


def align(*, topk_ids, block_size, num_experts, expert_map, ignore_invalid_experts):
    assert ignore_invalid_experts and expert_map.shape[0] == num_experts and block_size in (16, 32)
    local = expert_map[topk_ids.flatten().long()]
    rows = (local >= 0).nonzero().flatten()
    return rows, local[rows], rows.numel()


def mul_sum(*, inputs, topk_weights, topk_ids, expert_map, outputs):
    assert expert_map is None
    outputs.copy_((inputs * topk_weights.unsqueeze(-1)).sum(1))


@pytest.fixture
def humming(monkeypatch):
    torch = pytest.importorskip("torch")
    dtypes = NS(float4e2m1="e2m1", uint3="uint3", uint2="uint2")
    ops = NS(pack_weight=lambda codes, bits: codes.clone())
    monkeypatch.setitem(sys.modules, "humming", types.SimpleNamespace(dtypes=dtypes, ops=ops))
    vllm = types.ModuleType("vllm")
    vllm.arc_expert_int3 = int3
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.arc_expert_int3", int3)
    monkeypatch.setenv("ARC_EXPERT_INT3", '{"group": 16}')
    monkeypatch.setattr(rt, "_COLD", {7: [1, 4]})
    monkeypatch.setattr(rt, "_SPLITS", {})
    return torch, dtypes


def build(torch, dtypes, cold=(1, 4), experts=6, hidden=32, inter=16, seed=0):
    """A layer of ``experts`` int3 experts whose ``cold`` ones lie on the int2 grid, as converted NVFP4 tensors."""
    gen = torch.Generator().manual_seed(seed)

    def projection(n, k):
        values = torch.randint(-3, 4, (experts, n, k), generator=gen)
        values[0, 0, 0], values[2, 1, k - 1] = 4, -4  # the int3 extremes, in different groups
        for e in cold:
            values[e] = torch.randint(-1, 2, (n, k), generator=gen)
        values[cold[0], 0, 5], values[cold[1], 1, k - 3] = 2, -2  # the int2 extremes: one group negated, one not
        scale = torch.rand((experts, n, k // 16), generator=gen) + 0.5
        scale2 = torch.rand((experts,), generator=gen) + 0.5
        reference = (values.float().reshape(experts, n, k // 16, 16) * scale.unsqueeze(-1)).reshape(
            experts, n, k) * scale2.view(-1, 1, 1)
        return {"weight": pack_e2m1(torch, values), "weight_scale": scale, "weight_scale_2": scale2}, reference

    w13, ref13 = projection(2 * inter, hidden)
    w2, ref2 = projection(hidden, inter)
    layer = Layer(torch, "language_model.model.layers.7.mlp.experts", {"w13": w13, "w2": w2})
    schema = Schema(dtypes.float4e2m1, 16)
    configs = {}
    for sub in ("w13", "w2"):
        new_schema, tensors = rt.repack(schema, extract(layer, sub), experts, layer, sub)
        assert new_schema.b_dtype == "uint3" and tensors["weight"].shape[0] == experts
        replace(layer, sub, tensors)
        count = rt.split(layer, sub, experts, extract, replace)
        configs[sub] = NS(num_experts=count, bits=3)
        rt.cold(layer, sub, configs[sub], lambda s, n: NS(num_experts=n, bits=int(str(s.b_dtype)[-1]), schema=s),
                lambda config, tensors: tensors)
    return layer, configs, ref13, ref2


def kernel(torch, layer, configs, experts=6, inter=16, hidden=32):
    def hot_forward(sub, *, inputs, weight, input_scale, outputs, **kwargs):
        assert weight is getattr(layer, f"{sub}_weight") and input_scale is None
        kwargs.pop("tuning_config"), kwargs.pop("compute_config"), kwargs.pop("valid_shape_m")
        forward(configs[sub], inputs=inputs, weight=weight, weight_scale=getattr(layer, f"{sub}_weight_scale"),
                weight_scale_2=getattr(layer, f"{sub}_weight_scale_2"), outputs=outputs, **kwargs)

    def buffers(w13, w2, tokens, top_k, activation):
        return {"gate_up_output": torch.full((tokens * top_k, 2 * inter), float("nan")),
                "activation_output": torch.empty((tokens * top_k, inter)),
                "down_output": torch.full((tokens * top_k, hidden), float("nan"))}

    def activation(*, activation, input, output):
        output.copy_(torch.nn.functional.silu(input[:, :inter]) * input[:, inter:])

    return NS(humming_configs=configs, num_experts=experts, global_num_experts=experts, locks=None,
              w13_tuning_config=[(0, 10_000, {"block_shape": [32]})], w13_tuning_config_str="hot13",
              w2_tuning_config_str="hot2", compute_config_str="{}", humming_forward=hot_forward,
              prepare_buffers=buffers, apply_activation=activation,
              quantize_input=lambda sub, inputs, quanted_input: (inputs, None),
              estimate_local_valid_shape_m=lambda ids: ids.numel())


def test_split_layer_equals_the_dense_moe(humming):
    torch, dtypes = humming
    layer, configs, ref13, ref2 = build(torch, dtypes)
    state = layer._arc_int2
    assert state["w13"]["hot"] == [0, 2, 3, 5] and state["w13"]["cold"] == [1, 4]
    assert layer.w13_weight.shape[0] == 4 and layer.arc_int2_w13_weight.shape[0] == 2
    assert int(layer.arc_int2_w13_weight.max()) <= 3 and int(layer.arc_int2_w13_weight.min()) >= 0  # 2-bit codes
    assert int(layer.w13_weight.max()) == 7  # 3-bit codes
    experts = kernel(torch, layer, configs)
    split = rt.lookup(experts)
    assert split is not None and rt.lookup(experts) is split
    split.tuning = {"w13": ([(0, 10_000, {"block_shape": [16]})], "cold13"), "w2": ([], "cold2")}
    gen = torch.Generator().manual_seed(1)
    tokens, top_k = 9, 3
    x = torch.randn((tokens, 32), generator=gen)
    ids = torch.stack([torch.randperm(6, generator=gen)[:top_k] for _ in range(tokens)]).to(torch.int32)
    assert {1, 4} & set(ids.flatten().tolist()) and {0, 2, 3, 5} & set(ids.flatten().tolist())
    wts = torch.rand((tokens, top_k), generator=gen)
    out = torch.empty((tokens, 32))
    split.apply(experts, out, x, layer.w13_weight, layer.w2_weight, wts, ids, "silu", None, None, None,
                align=align, forward=forward, mul_sum=mul_sum)
    expected = torch.zeros((tokens, 32))
    for t in range(tokens):
        for slot in range(top_k):
            e = int(ids[t, slot])
            y = ref13[e] @ x[t]
            expected[t] += wts[t, slot] * (ref2[e] @ (torch.nn.functional.silu(y[:16]) * y[16:]))
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-4)
    with pytest.raises(RuntimeError, match="unmapped"):
        split.apply(experts, out, x, layer.w13_weight, layer.w2_weight, wts, ids, "silu", torch.arange(6), None, None,
                    align=align, forward=forward, mul_sum=mul_sum)


def test_a_listed_expert_off_the_int2_grid_cannot_be_served(humming, monkeypatch):
    torch, dtypes = humming
    monkeypatch.setattr(rt, "_COLD", {7: [1, 3]})  # expert 3 is an ordinary int3 expert
    with pytest.raises(ValueError, match="int2 grid"):
        build(torch, dtypes)


def test_a_kernel_without_its_split_refuses_to_run(humming):
    torch, dtypes = humming
    layer, configs, _, _ = build(torch, dtypes)
    stray = kernel(torch, layer, {"w13": NS(num_experts=4, bits=3), "w2": configs["w2"]})
    with pytest.raises(RuntimeError, match="lost its layer's split"):
        rt.lookup(stray)
    grouped = type("HummingGroupedExperts", (), {})()
    grouped.__dict__.update(kernel(torch, layer, configs).__dict__)
    with pytest.raises(RuntimeError, match="needs HummingIndexedExperts"):
        rt.check(grouped)
    whole = kernel(torch, layer, {"w13": NS(num_experts=6, bits=3), "w2": NS(num_experts=6, bits=3)})
    assert rt.lookup(whole) is None  # a layer that was not split (the draft's) keeps the pinned path


def test_grid_codes_match_the_int3_helper(humming):
    torch, _ = humming
    gen = torch.Generator().manual_seed(2)
    values = torch.randint(-3, 4, (2, 3, 32), generator=gen).float()
    values[0, 0, 0], values[1, 2, 31] = 4, -4
    scales = torch.rand((2, 3, 2), generator=gen) + 0.5
    ours, theirs = rt.grid_codes(values, scales, 16, 3), int3.int3_codes(values, scales, 16)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1])
    with pytest.raises(ValueError, match="both"):
        rt.grid_codes(torch.tensor([[2.0, -2.0] + [0.0] * 14]), torch.ones((1, 1)), 16, 2)
