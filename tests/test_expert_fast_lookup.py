"""ARC_EXPERT_FAST_LOOKUP (MFU-2 24 Oct): the pinned RoutedExperts.load_weights over a filtered mapping makes the same
weight_loader calls, in the same order, as over the full mapping."""

import ast
import os
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
PINNED = Path(os.environ.get("ARC_PINNED_VLLM", "~/arc-shared/pinned-vllm"))
SRC = PINNED / "vllm/model_executor/layers/fused_moe/routed_experts.py"


def pinned_methods():
    tree = ast.parse(SRC.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoutedExperts")
    keep = [f for f in cls.body if getattr(f, "name", "") in ("load_weights", "build_expert_params_mapping")]
    for f in keep:
        f.decorator_list = []
    import typing
    ns = {"torch": torch, "Iterable": typing.Iterable, "logger": types.SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None),
          "FusedMoeWeightScaleSupported": types.SimpleNamespace(BLOCK=types.SimpleNamespace(value="block")),
          "EplbState": types.SimpleNamespace(build_initial_global_physical_to_logical_map=lambda n, r: list(range(n + r)))}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "routed", "exec"), ns)
    return ns["load_weights"], ns["build_expert_params_mapping"]


@pytest.mark.skipif(not SRC.exists(), reason="no local copy of the pinned vLLM")
@pytest.mark.parametrize("lora", ["", "base_layer."])
def test_fast_lookup_same_loader_calls(lora):
    from arc_infra import expert_mask_runtime as r

    load_weights, build = pinned_methods()
    experts = 40

    def make_cls():
        class Routed:
            layer_name = "model.layers.3.mlp"
            is_fused_checkpoint_transposed = False

            def __init__(self, calls):
                self.calls = calls
                for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale", "w13_input_scale",
                             "w2_input_scale", "w13_weight_scale_2", "w2_weight_scale_2"):
                    p = types.SimpleNamespace(quant_method="block" if "scale" in name else None)
                    p.weight_loader = (lambda n: lambda **kw: calls.append(
                        (n, kw["weight_name"], kw["shard_id"], kw["expert_id"], tuple(kw["loaded_weight"].shape))) or True)(name)
                    for prefix in ("experts.", "experts.base_layer."):  # param names as the loader derives them
                        setattr(self, prefix + name, p)

            def get_expert_mapping(self, include_fused=False):
                return build("gate_proj", "down_proj", "up_proj", num_experts=experts, routed_experts_prefix="",
                             lora_base_layer_prefix=lora, include_fused=include_fused)

            def _orient_fused_weight(self, w, transpose):
                return w.transpose(-1, -2) if transpose else w

        Routed.load_weights = load_weights
        return Routed

    names = []
    for e in (0, 1, 5, 12, 15, 39):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                names.append((f"experts.{e}.{proj}.{lora}{suffix}", torch.zeros(4, 2)))
    names.append(("experts.gate_up_proj", torch.zeros(experts, 4, 8)))
    names.append(("experts.down_proj", torch.zeros(experts, 2, 4)))
    names.append(("experts.7.gate_up_proj." + lora + "weight", torch.zeros(8, 2)))

    base_calls, fast_calls = [], []
    base = make_cls()(base_calls)
    base_out = [list(base.load_weights([nw])) for nw in names]
    Fast = make_cls()
    assert r.install_fast_lookup(Fast) and not r.install_fast_lookup(Fast)
    fast = Fast(fast_calls)
    fast_out = [list(fast.load_weights([nw])) for nw in names]
    assert fast_calls == base_calls and fast_out == base_out and len(base_calls) > 100
    assert fast.get_expert_mapping(include_fused=True) == base.get_expert_mapping(include_fused=True)
