"""mmrtx post-import patches for the vLLM minimax-m31-flash image (put this dir on PYTHONPATH; env-gated).

MMRTX_MOE_A16=1: MiniMax's routed experts load as weight-only NVFP4 (W4A16). The model declares per-token NVFP4 QAT
activations (minimax_m3/nvidia/model.py qat_activation_key), which Humming refuses; Marlin already drops them on
sm_120, so this changes nothing there and lets `--moe-backend humming` (+ VLLM_HUMMING_ONLINE_QUANT_CONFIG) serve them.

MMRTX_MOE_REQUANT=<humming weight-schema JSON>: routed experts go through HummingMoEMethod, loading the NVFP4
checkpoint tensors and requantizing them to the given schema (e.g. uint3, group 16, zero point) with bf16 activations;
dense layers keep MiniMax's MXFP8 method (`--quantization humming` rejects MiniMax's mxfp8 config).

MMRTX_MOE_HUMMING=<humming weight-schema JSON>: the checkpoint's routed experts are already stored in that Humming
schema (hum_build.py); they load through HummingMoEMethod as is, no runtime requant.
MMRTX_MOE_HUMMING_LAYERS=3,7,...: only those layers' experts are Humming (hum_build.py variant); the rest keep
MiniMax's NVFP4 method."""
import importlib.abc
import importlib.util
import os
import sys

TARGET = os.environ.get("MMRTX_PATCH_TARGET", "vllm.model_executor.layers.quantization.compressed_tensors."
                        "compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4")
CLASS = os.environ.get("MMRTX_PATCH_CLASS", "CompressedTensorsW4A4Nvfp4MoEMethod")


def _patch(module):
    cls = getattr(module, CLASS)
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        if len(args) >= 3:  # (moe, layer_name, use_a16, ...)
            args = args[:2] + (True,) + args[3:]
        else:
            kwargs["use_a16"] = True
        orig(self, *args, **kwargs)
    cls.__init__ = __init__
    cls._mmrtx_a16 = True


MODELOPT = os.environ.get("MMRTX_MODELOPT_TARGET", "vllm.model_executor.layers.quantization.modelopt")
CT_NVFP4 = {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized", "type": "float", "num_bits": 4,
            "group_size": 16, "strategy": "tensor_group", "symmetric": True}


def _requant_method(layer, force, native=False):
    import vllm.model_executor.layers.quantization.humming as H
    from humming.schema.base import BaseWeightSchema
    from humming.schema.humming import HummingInputSchema
    if native:
        qc = H.HummingLayerQuantizationConfig(weight_schema=BaseWeightSchema.from_config(dict(force)),
                                              force_input_schema=HummingInputSchema())
    else:
        qc = H.HummingLayerQuantizationConfig(weight_schema=BaseWeightSchema.from_config(dict(CT_NVFP4)),
                                              force_weight_schema=BaseWeightSchema.from_config(dict(force)),
                                              force_input_schema=HummingInputSchema())
    m = H.HummingMoEMethod(qc, layer.moe_config)
    m.prepare_weight_loader = _scalar_global_scale(m.prepare_weight_loader)
    return m


def _scalar_global_scale(prepare):
    """Humming's NVFP4 schema keeps one w13 global scale per expert ([E]); RoutedExperts' per-tensor loader writes
    [E, 2] (w1, w3) and fails on it. Write the scalar directly; w1 and w3 must agree (they do in MiniMax's checkpoint)."""
    def wrapped(layer, weight_loader):
        inner = prepare(layer, weight_loader)
        seen = {}

        def loader(param, loaded_weight, weight_name, shard_id, expert_id=None, return_success=False):
            if getattr(param, "param_name", "") == "weight_global_scale" and shard_id in ("w1", "w3") \
                    and param.data.dim() == 1:
                v = float(loaded_weight.reshape(-1)[0])
                if seen.setdefault(expert_id, v) != v:
                    raise ValueError(f"w1/w3 global scales differ for expert {expert_id}: {seen[expert_id]} vs {v}")
                param.data[expert_id] = v
                return True if return_success else None
            return inner(param, loaded_weight, weight_name, shard_id=shard_id, expert_id=expert_id,
                         return_success=return_success)
        return loader
    return wrapped


def _patch_modelopt(module):
    import json
    native = bool(os.environ.get("MMRTX_MOE_HUMMING"))
    force = json.loads(os.environ["MMRTX_MOE_HUMMING"] if native else os.environ["MMRTX_MOE_REQUANT"])
    cls = module.ModelOptMxFp8Config
    orig = cls.get_quant_method

    only = {int(x) for x in os.environ.get("MMRTX_MOE_HUMMING_LAYERS", "").split(",") if x.strip()}

    def chosen(prefix):
        if not only:
            return True
        import re
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
        return bool(m) and int(m.group(1)) in only

    def get_quant_method(self, layer, prefix):
        if getattr(self, "moe_quant_algo", None) == "NVFP4" and type(layer).__name__ == "RoutedExperts" \
                and not self.is_layer_excluded(prefix):  # before orig: skip building the unused NVFP4 method
            if chosen(prefix):
                return _requant_method(layer, force, native)
            return orig(self, layer, prefix)
        m = orig(self, layer, prefix)
        if type(m).__name__ == "CompressedTensorsW4A4Nvfp4MoEMethod" and chosen(prefix):
            return _requant_method(layer, force, native)
        return m
    cls.get_quant_method = get_quant_method
    cls._mmrtx_requant = force


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch):
        self.target, self.patch = target, patch

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        run, patch = spec.loader.exec_module, self.patch

        def exec_module(module):
            run(module)
            patch(module)
        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("MMRTX_MOE_A16") == "1":
    sys.meta_path.insert(0, _Finder(TARGET, _patch))
if os.environ.get("MMRTX_MOE_REQUANT") or os.environ.get("MMRTX_MOE_HUMMING"):
    sys.meta_path.insert(0, _Finder(MODELOPT, _patch_modelopt))
