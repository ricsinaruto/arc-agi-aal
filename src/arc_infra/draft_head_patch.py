"""Hash-guarded draft-head overlay (MTP draft fine-tune, 28 September); pinned sources stay untouched otherwise.

A study's ``server.draft_head`` (absent: the pinned runtime is served unchanged) selects:

- ``dump: true``: prefill-only hidden-state dumps for training (``nvidia/model.py``: the target's
  pre-final-mixer multi-stream rows, the exact MTP input; ``nvidia/mtp.py``: the draft's step-1 sampled
  rows, for checking the training re-implementation). Rows are written only while the control file
  ``<dump dir>/ACTIVE`` (target) / ``ACTIVE_DRAFT`` (draft) exists, so warm-up and probe requests write
  nothing. Needs eager execution (``--enforce-eager``, added by ``arena.serving``): the hook is Python.
- ``overlay: {"path", "sha256"}``: a fine-tuned draft head. ``Qwen3_8FlashNextMTP.load_weights``
  substitutes the overlay's tensors (checkpoint names, BF16, same shapes) for the checkpoint's as they
  stream past; everything after loading (the phase-4 NvFp4 draft-expert quantizer included) sees the
  new weights. ``path`` is an ``arena.weights_overlay`` directory; its manifest's sha256 is pinned.

Each edit is guarded by exact anchors and the file's sha256: ``mtp.py`` may be the pinned file or the
phase-4 serving-kernels version (``serving_kernels_patch`` runs first). The helper
(``draft_head_runtime.py``) is copied to ``vllm/arc_draft_head.py`` with the settings in
``vllm/arc_draft_head.json``.
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_draft_head.py'
SETTINGS = 'vllm/arc_draft_head.json'
MARKER = 'draft-head-patch.json'
DUMP_DIR = '/tmp/arc-draft-dump'
MODEL = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/model.py',
    'sha256': ('d900cd6fcacba18f460e00b3f018fbf36fbe6ecc310692b6adb1451f7f53cc17',),
    'edits': [('        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(\n'
               '            hidden_states, block_output, injection\n'
               '        )\n',
               '        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(\n'
               '            hidden_states, block_output, injection\n'
               '        )\n'
               '        # ARC (draft-head fine-tune): dump the MTP input rows of prefill-only passes.\n'
               '        from vllm.arc_draft_head import dump_target as _arc_dump_target\n'
               '        _arc_dump_target(positions, multi_hidden)\n', 1)],
}
# Pinned mtp.py and the phase-4 serving-kernels version of it (draft_quant_hook inserted).
MTP_SHA256 = ('7735cee47d0d1e4776bebd30d907e4a62160409ce4ef2d65611559f8d58af431',
              '6fc7b441b53b479510de7206478b2b82bfa5751fd69fac874a5329b0575b342c')
MTP_DUMP = ('        return sample_hidden_states, multi_hidden\n',
            '        # ARC (draft-head fine-tune): the draft\'s step-1 rows, to check the training code.\n'
            '        from vllm.arc_draft_head import dump_draft as _arc_dump_draft\n'
            '        _arc_dump_draft(positions, sample_hidden_states, spec_step_idx)\n'
            '        return sample_hidden_states, multi_hidden\n', 1)
MTP_OVERLAY = ('        def remap_weight_names():\n'
               '            for name, weight in weights:\n',
               '        def remap_weight_names():\n'
               '            # ARC (draft-head fine-tune): fine-tuned draft tensors replace the checkpoint\'s.\n'
               '            from vllm.arc_draft_head import overlay_weights as _arc_overlay_weights\n'
               '            for name, weight in _arc_overlay_weights(weights):\n', 1)


def validate(spec):
    """Normalize a ``server.draft_head`` value; None/absent/all-off means no overlay."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - {'dump', 'overlay'}:
        raise ValueError('draft_head takes only dump and overlay')
    dump = spec.get('dump', False)
    if type(dump) is not bool:
        raise ValueError('draft_head.dump must be a boolean')
    overlay = spec.get('overlay')
    if overlay is not None:
        if (not isinstance(overlay, dict) or set(overlay) != {'path', 'sha256'}
                or not all(isinstance(v, str) and v for v in overlay.values())):
            raise ValueError('draft_head.overlay needs path and sha256 (of overlay-manifest.json)')
    if not dump and overlay is None:
        return None
    return {'dump': dump, 'overlay': overlay}


def edits(spec):
    """``{relative: (accepted sha256s, [edits])}`` for an (already validated) setting."""
    chosen = {}
    if spec['dump']:
        chosen[MODEL['relative']] = (MODEL['sha256'], list(MODEL['edits']))
    mtp = []
    if spec['dump']:
        mtp.append(MTP_DUMP)
    if spec['overlay'] is not None:
        mtp.append(MTP_OVERLAY)
    if mtp:
        chosen['vllm/models/qwen3_8_flash_next/nvidia/mtp.py'] = (MTP_SHA256, mtp)
    return chosen


def server_flags(spec):
    return ['--enforce-eager'] if spec['dump'] else []


def patched_source(relative, accepted, file_edits, original):
    if hashlib.sha256(original).hexdigest() not in accepted:
        raise RuntimeError(f'Draft-head overlay rejected an unrecognized {relative}')
    text = original.decode()
    for anchor, replacement, count in file_edits:
        if text.count(anchor) != count:
            raise RuntimeError(f'Draft-head overlay anchor count changed in {relative}')
        text = text.replace(anchor, replacement)
    compile(text, relative, 'exec')
    return text.encode()


def check_overlay(overlay):
    """The overlay directory holds a manifest with the pinned sha256 and its tensor file."""
    path = Path(overlay['path'])
    manifest = (path / 'overlay-manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != overlay['sha256']:
        raise RuntimeError('draft-head overlay manifest differs from the pinned sha256')
    data = json.loads(manifest)
    tensors = path / data['overlay']['file']
    with tensors.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != data['overlay']['sha256']:
        raise RuntimeError('draft-head overlay tensors differ from the manifest')
    if not all(name.startswith('mtp.') for name in data['tensors']):
        raise RuntimeError('a draft-head overlay may replace MTP tensors only')
    return {'path': str(path), 'manifest_sha256': overlay['sha256'], 'tensors': len(data['tensors']),
            'overlay_sha256': digest}


def apply(site, output, spec):
    """Patch the extracted runtime under ``site`` for ``spec``; the identity goes to ``output``."""
    spec = validate(spec)
    if spec is None:
        raise ValueError('No draft-head overlay requested; the pinned runtime must stay unpatched')
    site = Path(site)
    helper = Path(__file__).with_name('draft_head_runtime.py').read_bytes()
    settings = {'dump_dir': DUMP_DIR if spec['dump'] else None, 'overlay': None}
    if spec['overlay'] is not None:
        settings['overlay'] = check_overlay(spec['overlay'])
    settings_bytes = (json.dumps(settings, sort_keys=True) + '\n').encode()
    chosen = edits(spec)
    originals = {rel: (site / rel).read_bytes() for rel in chosen}
    patched = {rel: patched_source(rel, accepted, file_edits, originals[rel])
               for rel, (accepted, file_edits) in chosen.items()}
    (site / HELPER).write_bytes(helper)
    (site / SETTINGS).write_bytes(settings_bytes)
    for rel, data in patched.items():
        (site / rel).write_bytes(data)
    identity = {'files': {rel: {'original_sha256': hashlib.sha256(originals[rel]).hexdigest(),
                                'patched_sha256': hashlib.sha256(patched[rel]).hexdigest()} for rel in chosen},
                'helper_sha256': hashlib.sha256(helper).hexdigest(), 'settings': settings,
                'server_flags': server_flags(spec)}
    marker = Path(output) / MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('ARC_DRAFT_HEAD_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
    return identity
