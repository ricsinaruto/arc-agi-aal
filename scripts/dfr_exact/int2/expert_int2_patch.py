"""Hash-guarded INT2 cold half for vLLM's Humming MoE backend (8 October; ``expert_int2_runtime.py``).

``serving_kernels.expert_int2`` (with ``expert_int3`` and an INT2-cold expert overlay) patches two pinned files:
``humming_utils.py`` (the int3 repack's call site with the layer, and the split of a sublayer into a hot and a cold
Humming tensor around ``prepare_layer_config``) and ``fused_humming_moe.py`` (the indexed kernel's ``apply`` hands a
split layer to the helper; any other Humming kernel refuses one). ``write_settings`` puts the overlay manifest's cold
list next to the helper at server start.
"""

import hashlib
import json
from pathlib import Path

from .expert_int3_patch import TARGET as INT3_TARGET

HELPER = 'vllm/arc_expert_int2.py'
SETTINGS = 'vllm/arc_expert_int2.json'
UTILS_TARGET = {
    'relative': INT3_TARGET['relative'],
    'sha256': INT3_TARGET['sha256'],
    'edits': [(
        '    converted_input_schema, _ = input_schema.convert_humming(\n'
        '        tensors=converted_tensors,\n',
        '    # ARC (INT2 cold half): the int3 repack, and the cold experts as uint2 kept aside on the layer.\n'
        '    from vllm.arc_expert_int2 import repack as _arc_int2_repack\n'
        '    converted_weight_schema, converted_tensors = _arc_int2_repack(\n'
        '        converted_weight_schema, converted_tensors, num_experts, layer, sublayer_name\n'
        '    )\n'
        '    converted_input_schema, _ = input_schema.convert_humming(\n'
        '        tensors=converted_tensors,\n', 1), (
        '    config = prepare_layer_config(\n'
        '        shape_n=shape_n,\n'
        '        shape_k=shape_k,\n'
        '        pad_n_to_multiple=256,\n',
        '    # ARC (INT2 cold half): the layer\'s own tensors keep the hot experts; the cold ones follow below.\n'
        '    import vllm.arc_expert_int2 as _arc_int2\n'
        '    num_experts = _arc_int2.split(\n'
        '        layer, sublayer_name, num_experts, _extract_sublayer_tensors, _replace_layer_parameters\n'
        '    )\n'
        '    config = prepare_layer_config(\n'
        '        shape_n=shape_n,\n'
        '        shape_k=shape_k,\n'
        '        pad_n_to_multiple=256,\n', 1), (
        '    _replace_layer_parameters(layer, sublayer_name, tensors)\n'
        '    return config\n',
        '    _replace_layer_parameters(layer, sublayer_name, tensors)\n'
        '    _arc_int2.cold(\n'
        '        layer, sublayer_name, config,\n'
        '        lambda schema, count: prepare_layer_config(\n'
        '            shape_n=shape_n, shape_k=shape_k, pad_n_to_multiple=256, pad_k_to_multiple=128,\n'
        '            input_schema=input_schema, weight_schema=schema, has_bias=has_bias, num_experts=count,\n'
        '            torch_dtype=param_dtype,\n'
        '        ),\n'
        '        transform_humming_tensors,\n'
        '    )\n'
        '    return config\n', 1)],
}
MOE_TARGET = {
    'relative': 'vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py',
    'sha256': '75849a56604fc5221240cfd0385f8608fbf75a48f19fbe42f138c942b9b804fe',
    'edits': [(
        '        self._permute_scratch: MoEPermuteScratch | None = None\n',
        '        self._permute_scratch: MoEPermuteScratch | None = None\n'
        '        from vllm.arc_expert_int2 import check as _arc_int2_check  # ARC (INT2 cold half)\n'
        '        _arc_int2_check(self)\n', 1), (
        '        Standard apply implementation for Humming indexed experts.\n'
        '\n'
        '        Humming performs activation quantization internally and consumes the\n'
        '        weights supplied by the modular kernel interface.\n'
        '        Results are written to the output buffer supplied by the modular\n'
        '        kernel, which may alias workspace13.\n'
        '        """\n'
        '        assert not apply_router_weight_on_input\n',
        '        Standard apply implementation for Humming indexed experts.\n'
        '\n'
        '        Humming performs activation quantization internally and consumes the\n'
        '        weights supplied by the modular kernel interface.\n'
        '        Results are written to the output buffer supplied by the modular\n'
        '        kernel, which may alias workspace13.\n'
        '        """\n'
        '        assert not apply_router_weight_on_input\n'
        '        # ARC (INT2 cold half): a layer split into hot uint3 and cold uint2 experts runs two launches.\n'
        '        from vllm.arc_expert_int2 import lookup as _arc_int2_lookup\n'
        '        _arc_split = _arc_int2_lookup(self)\n'
        '        if _arc_split is not None:\n'
        '            return _arc_split.apply(\n'
        '                self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, expert_map,\n'
        '                workspace13, workspace2,\n'
        '            )\n', 1)],
}
TARGETS = (UTILS_TARGET, MOE_TARGET)


def settings_bytes(manifest, manifest_sha256):
    """The helper's settings from an INT2-cold overlay's manifest (``settings.int2_experts``, checkpoint ids)."""
    experts = (manifest.get('settings') or {}).get('int2_experts')
    if not isinstance(experts, dict) or not experts or not all(isinstance(v, list) and v for v in experts.values()):
        raise RuntimeError('serving_kernels.expert_int2 needs an INT2-cold expert overlay '
                           '(overlay-manifest.json settings.int2_experts)')
    record = {'experts': {str(int(k)): sorted(int(e) for e in v) for k, v in experts.items()},
              'overlay_manifest_sha256': manifest_sha256}
    return (json.dumps(record, sort_keys=True) + '\n').encode()


def write_settings(site, manifest, manifest_sha256):
    data = settings_bytes(manifest, manifest_sha256)
    (Path(site) / SETTINGS).write_bytes(data)
    identity = {'settings_sha256': hashlib.sha256(data).hexdigest(), 'overlay_manifest_sha256': manifest_sha256,
                'layers': len(manifest['settings']['int2_experts']),
                'int2_experts': sum(len(v) for v in manifest['settings']['int2_experts'].values())}
    print('ARC_EXPERT_INT2_SETTINGS ' + json.dumps(identity, sort_keys=True), flush=True)
    return identity
