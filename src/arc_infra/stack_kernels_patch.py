"""Hash-guarded optional serving overlay (kernel work, phase 5: the stack); pinned sources stay untouched otherwise.

A study's ``server.stack_kernels`` (absent: the pinned runtime is served unchanged) selects:

- ``async_h2d``: the PLE short-conv attention metadata builder (``v1/attention/backends/short_conv_attn.py``) copies
  its per-step request-index tensors to the GPU with blocking ``.to(device)`` calls. A blocking host-to-device copy
  from pageable memory makes PyTorch synchronize the stream, so with MTP (the spec-decode branch of ``build``) the
  engine's input preparation for step N+1 waits for step N's whole forward, draft steps and sampling to finish, and
  the ~1.2 ms of host preparation that follows runs with the GPU idle (one ~20 ms ``cudaStreamSynchronize`` per
  iteration in the MTP-3 serving-lab traces, followed by 1.3 ms of prep with 0.1 ms of GPU work; MTP-0 never
  takes this branch). The patch makes the five copies asynchronous through the file's own ``async_tensor_h2d``
  (pinned staging, ``non_blocking=True``), which the same builder already uses for the spec mask. Same indices,
  same kernels: the served numerics are bitwise unchanged; only the host stops waiting.

Each option patches only its own pinned file(s), each guarded by the pinned file's sha256 and exact anchors. No
helper module is copied: the edits call functions the pinned file already imports. Composable with
``server.serving_kernels`` and ``server.dense_kernels`` (disjoint files).
"""

import hashlib
import json
from pathlib import Path

MARKER = 'stack-kernels-patch.json'
OPTIONS = ('async_h2d',)
# The stack's serving setting (docs/validation-20260927/kernels-phase5-stack.md).
PHASE5 = {'async_h2d': True}

SHORT_CONV = {
    'relative': 'vllm/v1/attention/backends/short_conv_attn.py',
    'sha256': '985fd9f320eff615f2bc4b4d5576cd7f301a751380709979a251f92101bda14d',
    'edits': [
        ('        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)\n'
         '        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)\n',
         '        # ARC (kernel work, phase 5): asynchronous copies; a blocking H2D copy syncs the stream.\n'
         '        spec_req_idx = async_tensor_h2d(spec_req_idx_cpu, device=query_start_loc.device)\n'
         '        non_spec_req_idx = async_tensor_h2d(non_spec_req_idx_cpu, device=query_start_loc.device)\n', 1),
        ('            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1\n',
         '            req_group[async_tensor_h2d(decode_req_idx_cpu, device=query_start_loc.device)] = 1\n', 1),
        ('            spec_req_idx_cpu.to(num_accepted_tokens.device)\n',
         '            async_tensor_h2d(spec_req_idx_cpu, device=num_accepted_tokens.device)\n', 1),
        ('                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)\n',
         '                non_spec_req_idx = async_tensor_h2d(non_spec_req_idx_cpu, device=num_computed_tokens.device)\n',
         1),
    ],
    # The edits rely on this import of the pinned file (checked, not added).
    'requires': 'from vllm.utils.torch_utils import async_tensor_h2d\n',
}
TARGETS = (SHORT_CONV,)
# Smoke only (``async_h2d: "toggle"``): the same copies go through a helper that falls back to the pinned blocking
# ``.to(device)`` while this flag file exists, so one server start measures both (the bench's ``h2d_toggle``).
TOGGLE_FLAG = '/tmp/arc-stack-blocking-h2d'
TOGGLE_HELPER = ('from vllm.utils.torch_utils import async_tensor_h2d\n\n\n'
                 '# ARC (kernel work, phase 5, smoke only): blocking copies while the flag file exists.\n'
                 'def _arc_stack_h2d(t, device):\n'
                 '    import os\n\n'
                 f'    if os.path.exists({TOGGLE_FLAG!r}):\n'
                 '        return t.to(device)\n'
                 '    return async_tensor_h2d(t, device=device)\n\n\n')


def toggle_target():
    """SHORT_CONV with the copies routed through the flag-checking helper (built from SHORT_CONV's current guard)."""
    edits = [(anchor, replacement.replace('async_tensor_h2d(', '_arc_stack_h2d('), count)
             for anchor, replacement, count in SHORT_CONV['edits']]
    edits.append((SHORT_CONV['requires'], TOGGLE_HELPER, 1))
    return {**SHORT_CONV, 'edits': edits}


def validate(spec):
    """Normalize a ``server.stack_kernels`` value; None/absent/all-off means no overlay."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - set(OPTIONS):
        raise ValueError(f'stack_kernels takes only {OPTIONS}')
    result = {}
    for key in OPTIONS:
        value = spec.get(key, False)
        if type(value) is not bool and value != 'toggle':
            raise ValueError(f'stack_kernels.{key} must be a boolean (or "toggle" for the smoke)')
        result[key] = value
    return result if any(result.values()) else None


def targets(spec):
    """The pinned files an (already validated) setting patches."""
    if spec['async_h2d'] == 'toggle':
        return [toggle_target()]
    return [SHORT_CONV] if spec['async_h2d'] else []


def patched_source(target, original):
    if hashlib.sha256(original).hexdigest() != target['sha256']:
        raise RuntimeError(f"Stack overlay rejected an unrecognized {target['relative']}")
    text = original.decode()
    if target.get('requires') and target['requires'] not in text:
        raise RuntimeError(f"Stack overlay: {target['relative']} lacks {target['requires'].strip()!r}")
    for anchor, replacement, count in target['edits']:
        if text.count(anchor) != count:
            raise RuntimeError(f"Stack overlay anchor count changed in {target['relative']}")
        text = text.replace(anchor, replacement)
    compile(text, target['relative'], 'exec')
    return text.encode()


def apply(site, output, spec):
    """Patch the extracted runtime under ``site`` for ``spec``; the identity goes to ``output``."""
    spec = validate(spec)
    if spec is None:
        raise ValueError('No stack kernels requested; the pinned runtime must stay unpatched')
    site = Path(site)
    marker = Path(output) / MARKER
    chosen = targets(spec)
    originals = {t['relative']: (site / t['relative']).read_bytes() for t in chosen}
    if all(hashlib.sha256(originals[t['relative']]).hexdigest() == t['sha256'] for t in chosen):
        patched = {t['relative']: patched_source(t, originals[t['relative']]) for t in chosen}
        for relative, data in patched.items():
            (site / relative).write_bytes(data)
        identity = {'files': {t['relative']: {'original_sha256': t['sha256'],
                                              'patched_sha256': hashlib.sha256(patched[t['relative']]).hexdigest()}
                              for t in chosen},
                    'settings': spec}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_STACK_KERNELS_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    # A restart in the same working directory re-extracts the pinned runtime first; anything else already patched
    # with this exact overlay is accepted, never re-patched.
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (identity.get('settings') == spec and set(identity['files']) == set(originals)
                and all(hashlib.sha256(data).hexdigest() == identity['files'][rel]['patched_sha256']
                        for rel, data in originals.items())):
            return identity
    raise RuntimeError('Stack overlay found a modified or partially patched pinned file')
