"""Hash-guarded expert-mask overlay (hero run): the pruned (t37) model served as a router-logit mask.

``server.expert_mask = {"mask": <bundle-relative arc-expert-prune-v1 JSON>, "sha256": <its sha256>}`` patches the
fused-MoE runner at the expert_probe anchor (``MoERunner._forward_impl``, after the gate): the router logits pass
through ``vllm.arc_expert_mask.on_router_logits`` (``expert_mask_runtime.py``), which adds MASK_VALUE at each layer's
pruned experts. Same routing as ``expert_prune`` (a renumbered cut) with the checkpoint's expert ids, so LoRA adapters,
served routes and the trainer's mask (rl1_trainer RL1_EXPERT_MASK) line up. Exclusive with expert_probe /
expert_capture (same pinned file) and expert_prune (a cut model).

``ARC_EXPERT_MASK_SKIP_READ=1`` (RL start-up, 23 Oct; set by the launcher's boot switch): the patched runner module also
installs ``install_skip_read`` at import, so the weight iterators never read a masked expert's packed weights
(``...experts.<e>.{gate,up,down}_proj.weight``; every scale is still read, as vLLM's own EP filter does: the Marlin
prep takes a max over all experts' block scales). Masked experts are never routed, so their unread slots are never
computed on. The skipped count is in the worker's status RPC (``skipped_reads``).
"""

import hashlib
import json
import os
from pathlib import Path

from .expert_probe_patch import RUNNER as PROBE_RUNNER
from .expert_prune_patch import load_mask

HELPER = 'vllm/arc_expert_mask.py'
SETTINGS = 'vllm/arc_expert_mask.json'
MARKER = 'expert-mask-patch.json'
RUNNER = {
    'relative': PROBE_RUNNER['relative'],
    'sha256': PROBE_RUNNER['sha256'],
    'edits': [(PROBE_RUNNER['edits'][0][0],
               PROBE_RUNNER['edits'][0][0].replace(
                   '\n        with self._sequence_parallel_context():\n',
                   '\n        # ARC (expert mask): pruned experts never routed (hero run).\n'
                   '        from vllm.arc_expert_mask import on_router_logits as _arc_on_router_logits\n'
                   '        router_logits = _arc_on_router_logits(self, router_logits)\n'
                   '\n'
                   '        with self._sequence_parallel_context():\n'), 1)],
}


def validate(spec):
    """``server.expert_mask``: None, or ``{"mask": <relative path>, "sha256": <hex>}``."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {'mask', 'sha256'}:
        raise ValueError('expert_mask takes exactly mask and sha256')
    digest = spec['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('expert_mask.sha256 must be a sha256 hex digest')
    if not isinstance(spec['mask'], str) or spec['mask'].startswith('/') or '..' in spec['mask']:
        raise ValueError('expert_mask.mask is a bundle-relative path')
    return dict(spec)


SKIP_ENV = 'ARC_EXPERT_MASK_SKIP_READ'
SKIP_TAIL = ('\n\n# ARC (expert mask, ARC_EXPERT_MASK_SKIP_READ): masked experts\' packed weights are never read.\n'
             'from vllm.arc_expert_mask import install_skip_read as _arc_install_skip_read  # noqa: E402\n'
             '\n_arc_install_skip_read()\n')
PROFILE_ENV = 'ARC_LOAD_PROFILE'
PROFILE_TAIL = ('\n\n# ARC (ARC_LOAD_PROFILE): weight-load time split (iterator vs expert loader, wall vs CPU, faults).\n'
                'from vllm.arc_expert_mask import install_load_profile as _arc_install_load_profile  # noqa: E402\n'
                '\n_arc_install_load_profile()\n')


COMPACT_ENV = 'ARC_EXPERT_MASK_COMPACT'
COMPACT_TAIL = ('\n\n# ARC (expert mask, ARC_EXPERT_MASK_COMPACT): kept experts load into compact slots.\n'
                'from vllm.arc_expert_mask import install_compact as _arc_install_compact  # noqa: E402\n'
                '\n_arc_install_compact()\n')
COMPACT_RUNNER_EDIT = (
    '                input_ids=input_ids,\n            )\n\n            fused_out = self.routed_experts.forward_modular(\n',
    '                input_ids=input_ids,\n            )\n'
    '            # ARC (expert mask, compact): checkpoint expert ids -> compact slots (routes were captured above).\n'
    '            from vllm.arc_expert_mask import compact_ids as _arc_compact_ids\n'
    '            topk_ids = _arc_compact_ids(self.routed_experts, topk_ids)\n'
    '\n            fused_out = self.routed_experts.forward_modular(\n', 1)


def _tail(*names):
    return ('\n\n# ARC (expert mask, ARC_EXPERT_MASK_COMPACT)\n'
            + ''.join(f'from vllm.arc_expert_mask import {n} as _arc_{n}  # noqa: E402\n' for n in names))


# Files the compact mode edits besides the runner (sha256 of the pinned 0.1.dev20073 runtime). Every edit keeps the
# masked path's kernel choices: the expert count those choices read is the checkpoint's.
COMPACT_TARGETS = (
    {'relative': 'vllm/model_executor/layers/fused_moe/layer.py',
     'sha256': '688711dd909e107c1712b81038cb1ed055fc4ecc9dce89bb4beef2d30f0541c5',
     'edits': [('        rocm_aiter_enabled=rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul,\n    )\n',
                '        rocm_aiter_enabled=rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul,\n    )\n'
                '    # ARC (expert mask, compact): a masked layer holds its kept experts only.\n'
                '    _arc_compact_manager(expert_map_manager, layer_name)\n', 1)],
     'tail': _tail('compact_manager')},
    {'relative': 'vllm/model_executor/layers/fused_moe/experts/marlin_moe.py',
     'sha256': 'f00460310b033bcb84644677e16af83773464f026c0dad15bcfe25bd28b39745',
     'edits': [('        M = math.ceil(M * E / global_num_experts)\n',
                '        M = M if _arc_marlin_compact(E, global_num_experts) else math.ceil(M * E / global_num_experts)\n', 1),
               ('        if M * topk / E / block_size_m < 0.9:\n',
                '        if M * topk / (global_num_experts if _arc_marlin_compact(E, global_num_experts) else E) / '
                'block_size_m < 0.9:\n', 1)],
     'tail': _tail('marlin_compact')},
    {'relative': 'vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py',
     'sha256': 'a7c61d4f67b5d23c4236645ffc0f6f1fe6f8ef2577e83229c479b311c1cddd8b',
     'edits': [('\n        combined_scale_factor = _nvfp4_compute_scale_factor(scales, param_dtype)\n',
                '\n        combined_scale_factor = _arc_marlin_scale_factor(layer, name, scales, param_dtype,\n'
                '                                                         _nvfp4_compute_scale_factor)\n', 1),
               ('\n            combined_scale_factor = _nvfp4_compute_scale_factor(scales, param_dtype)\n',
                '\n            combined_scale_factor = _arc_marlin_scale_factor(layer, name, scales, param_dtype,\n'
                '                                                             _nvfp4_compute_scale_factor)\n', 1)],
     'tail': _tail('marlin_scale_factor')},
    {'relative': 'vllm/lora/model_manager.py',
     'sha256': 'a784bb0fc0675d363f144ab7115ae70ccef9ef3573a02571a373788b13521a3c',
     'edits': [('expert_start:expert_end]', '_arc_sel]', 9),
               ('        if not module.use_ep:\n            return\n',
                '        if not module.use_ep and not _arc_compacted_module(module_name, module):\n            return\n', 1)],
     'regex': [(r'\n( +)expert_end = expert_start \+ local_num_experts\n',
                r'\n\1expert_end = expert_start + local_num_experts\n'
                r'\1_arc_sel = _arc_expert_select(module_name, module, expert_start, expert_end)\n', 3)],
     'tail': _tail('expert_select', 'compacted_module')},
    {'relative': 'vllm/lora/punica_wrapper/punica_gpu.py',
     'sha256': '7ddedad89b6721751012af6f72bd85d0f8e2f6e9b288fc1909b08a732c360919',
     'edits': [('w1_shape=w1.shape,', 'w1_shape=_arc_moe_shape(w1.shape),', 2),
               ('w2_shape=w2.shape,', 'w2_shape=_arc_moe_shape(w2.shape),', 2),
               ('<= local_num_experts * max_loras', '<= _arc_moe_experts(local_num_experts) * max_loras', 1)],
     'tail': _tail('moe_shape', 'moe_experts')},
)


def compact_source(target, original):
    """One compact-mode target edited (hash-guarded; anchor counts checked; compiles)."""
    import re

    if hashlib.sha256(original).hexdigest() != target['sha256']:
        raise RuntimeError(f"Expert-mask compact overlay rejected an unrecognized {target['relative']}")
    text = original.decode()
    for pattern, replacement, count in target.get('regex', ()):
        text, n = re.subn(pattern, replacement, text)
        if n != count:
            raise RuntimeError(f"Expert-mask compact: {n} matches of a regex anchor in {target['relative']}")
    for anchor, replacement, count in target['edits']:
        if text.count(anchor) != count:
            raise RuntimeError(f"Expert-mask compact anchor count changed in {target['relative']}")
        text = text.replace(anchor, replacement)
    text = text.rstrip('\n') + target['tail']
    compile(text, target['relative'], 'exec')
    return text.encode()


def settings_of(mask, label, sha256=None, skip_read=False, load_profile=False, compact=False):
    row = {'experts': int(mask['experts']), 'pruned': mask['pruned'], 'label': label}
    if sha256:  # the status RPC reports it; the loop checks it against the bundle's (hero run)
        row['sha256'] = sha256
    if skip_read:  # absent when off: the settings (and the patch identity) of a run without it are unchanged
        row['skip_read'] = True
    if load_profile:  # absent when off, as skip_read
        row['load_profile'] = True
    if compact:  # absent when off, as skip_read
        row['compact'] = True
    return (json.dumps(row, sort_keys=True) + '\n').encode()


def patched_source(original, skip_read=False, load_profile=False, compact=False):
    if hashlib.sha256(original).hexdigest() != RUNNER['sha256']:
        raise RuntimeError(f"Expert-mask overlay rejected an unrecognized {RUNNER['relative']}")
    text = original.decode()
    for anchor, replacement, count in RUNNER['edits']:
        if text.count(anchor) != count:
            raise RuntimeError(f"Expert-mask overlay anchor count changed in {RUNNER['relative']}")
        text = text.replace(anchor, replacement)
    if compact:
        anchor, replacement, count = COMPACT_RUNNER_EDIT
        if text.count(anchor) != count:
            raise RuntimeError(f"Expert-mask compact anchor count changed in {RUNNER['relative']}")
        text = text.replace(anchor, replacement)
    if skip_read:
        text = text.rstrip('\n') + SKIP_TAIL
    if load_profile:
        text = text.rstrip('\n') + PROFILE_TAIL
    if compact:
        text = text.rstrip('\n') + COMPACT_TAIL
    compile(text, RUNNER['relative'], 'exec')
    return text.encode()


def apply(site, output, mask_path, sha256):
    """Patch the extracted runtime under ``site`` with the mask file (sha256-checked); identity to ``output``."""
    site = Path(site)
    mask = load_mask(mask_path, sha256)
    helper = Path(__file__).with_name('expert_mask_runtime.py').read_bytes()
    skip_read = os.environ.get(SKIP_ENV) == '1'
    load_profile = os.environ.get(PROFILE_ENV) == '1'
    compact = os.environ.get(COMPACT_ENV) == '1'
    if compact and len({len(v) for v in mask['pruned'].values()}) != 1:
        raise ValueError(f'{COMPACT_ENV}=1 needs one pruned count for every layer')
    settings = settings_of(mask, Path(mask_path).name, sha256, skip_read, load_profile, compact)
    marker = Path(output) / MARKER
    target = site / RUNNER['relative']
    original = target.read_bytes()
    extra = [(t, site / t['relative']) for t in COMPACT_TARGETS] if compact else []
    if hashlib.sha256(original).hexdigest() == RUNNER['sha256']:
        patched = patched_source(original, skip_read, load_profile, compact)
        others = [(path, compact_source(t, path.read_bytes())) for t, path in extra]  # all checked before any write
        (site / HELPER).write_bytes(helper)
        (site / SETTINGS).write_bytes(settings)
        target.write_bytes(patched)
        for path, text in others:
            path.write_bytes(text)
        identity = {'file': RUNNER['relative'], 'original_sha256': RUNNER['sha256'],
                    'patched_sha256': hashlib.sha256(patched).hexdigest(),
                    'helper_sha256': hashlib.sha256(helper).hexdigest(),
                    'settings_sha256': hashlib.sha256(settings).hexdigest(), 'mask_sha256': sha256,
                    'pruned_total': sum(len(v) for v in mask['pruned'].values()),
                    **({'skip_read': True} if skip_read else {}), **({'load_profile': True} if load_profile else {})}
        if compact:
            identity['compact'] = {t['relative']: hashlib.sha256(text).hexdigest()
                                   for (t, _), (_, text) in zip(extra, others)}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_EXPERT_MASK_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    if marker.exists():  # a restart in the same working directory: only this exact overlay
        identity = json.loads(marker.read_text())
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and (site / HELPER).read_bytes() == helper and (site / SETTINGS).read_bytes() == settings
                and all(hashlib.sha256(path.read_bytes()).hexdigest() == (identity.get('compact') or {}).get(t['relative'])
                        for t, path in extra)):
            return identity
    raise RuntimeError('Expert-mask overlay found a modified or partially patched pinned file')
