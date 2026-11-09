"""Hash-guarded 3-bit routed-expert overlay for vLLM's Humming MoE backend (30 September; expert-3bit.md).

``server.expert_int3 = {"group": 16}`` (with ``serving_kernels.moe_backend = "humming"`` and an int3 expert overlay)
patches one pinned file, ``vllm/model_executor/layers/quantization/utils/humming_utils.py``: right after a MoE
sublayer's checkpoint tensors are converted to Humming's standard NVFP4 layout, ``vllm.arc_expert_int3.repack``
re-encodes them as Humming uint3 (``expert_int3_runtime.py``). Settings reach the worker as ``ARC_EXPERT_INT3``.
Nothing else in vLLM changes; with the variable unset the patched function is the pinned one.
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_expert_int3.py'
MARKER = 'expert-int3-patch.json'
ENV = 'ARC_EXPERT_INT3'
TARGET = {
    'relative': 'vllm/model_executor/layers/quantization/utils/humming_utils.py',
    'sha256': '5d9bb236858b21e695e691bb74adb891c03e1df03cd52eebc29fe154b4cd2552',
    'edits': [(
        '    converted_input_schema, _ = input_schema.convert_humming(\n'
        '        tensors=converted_tensors,\n',
        '    # ARC (3-bit experts): int3-grid NVFP4 routed experts re-encoded as Humming uint3, bit-exact.\n'
        '    from vllm.arc_expert_int3 import repack as _arc_int3_repack\n'
        '    converted_weight_schema, converted_tensors = _arc_int3_repack(\n'
        '        converted_weight_schema, converted_tensors, num_experts\n'
        '    )\n'
        '    converted_input_schema, _ = input_schema.convert_humming(\n'
        '        tensors=converted_tensors,\n', 1)],
}


def validate(spec):
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {'group'} or spec['group'] not in (16, 32, 64, 128):
        raise ValueError('expert_int3 takes exactly {"group": 16|32|64|128}')
    return dict(spec)


def env_value(spec):
    return json.dumps({'group': int(spec['group'])}, sort_keys=True)


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != TARGET['sha256']:
        raise RuntimeError(f"expert-int3 overlay rejected an unrecognized {TARGET['relative']}")
    text = original.decode()
    for anchor, replacement, count in TARGET['edits']:
        if text.count(anchor) != count:
            raise RuntimeError('expert-int3 overlay anchor count changed')
        text = text.replace(anchor, replacement)
    compile(text, TARGET['relative'], 'exec')
    return text.encode()


def apply(site, output):
    """Patch the extracted runtime under ``site``; identity JSON to ``output``. Idempotent for this exact overlay."""
    site = Path(site)
    helper = Path(__file__).with_name('expert_int3_runtime.py').read_bytes()
    path = site / TARGET['relative']
    original = path.read_bytes()
    marker = Path(output) / MARKER
    if hashlib.sha256(original).hexdigest() == TARGET['sha256']:
        patched = patched_source(original)
        (site / HELPER).write_bytes(helper)
        path.write_bytes(patched)
        identity = {'file': TARGET['relative'], 'original_sha256': TARGET['sha256'],
                    'patched_sha256': hashlib.sha256(patched).hexdigest(),
                    'helper_sha256': hashlib.sha256(helper).hexdigest()}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_EXPERT_INT3_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and (site / HELPER).read_bytes() == helper):
            return identity
    raise RuntimeError('expert-int3 overlay found a modified or partially patched pinned file')
