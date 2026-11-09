"""Hash-guarded load-time expert pruning overlay (28 September; docs/validation-20260927/expert-usage.md).

``server.expert_prune = {"mask": <bundle-relative JSON>, "sha256": <its sha256>}`` patches two pinned files of vLLM's
model loader (nothing in the model or the forward):

- ``base_loader.py``: ``prepare_config(vllm_config)`` right before ``initialize_model`` sets the text config's
  ``num_experts`` to the kept count, so every MoE layer is built with that many experts;
- ``default_loader.py``: the checkpoint stream passes through ``filter_weights`` (pruned experts dropped, kept ones
  renumbered, router rows sliced).

The helper (``expert_prune_runtime.py``) is copied to ``vllm/arc_expert_prune.py`` with the mask as
``vllm/arc_expert_prune.json``. The model's shapes change, so vLLM's compile cache is cleared before a pruned start
(``arena.serving``). MTP-0 only (the draft's experts are not pruned).
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_expert_prune.py'
SETTINGS = 'vllm/arc_expert_prune.json'
MARKER = 'expert-prune-patch.json'
FORMAT = 'arc-expert-prune-v1'
BASE_LOADER = {
    'relative': 'vllm/model_executor/model_loader/base_loader.py',
    'sha256': 'a7e925f232ad3eebbee7ab37d3aba724c24465c3078da29489da0438664c6b08',
    'edits': [('        with set_default_torch_dtype(model_config.dtype):\n',
               '        # ARC (expert pruning): the kept expert count, before the model is built.\n'
               '        from vllm.arc_expert_prune import prepare_config as _arc_prepare_config\n'
               '        _arc_prepare_config(vllm_config)\n'
               '        with set_default_torch_dtype(model_config.dtype):\n', 1)],
}
DEFAULT_LOADER = {
    'relative': 'vllm/model_executor/model_loader/default_loader.py',
    'sha256': '9c9d54b1b650bf5affc924ebb8b8c73711e187ae7486cadd9f737bb1279cc7b8',
    'edits': [('        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))\n',
               '        # ARC (expert pruning): pruned experts dropped, kept ids renumbered, router rows sliced.\n'
               '        from vllm.arc_expert_prune import filter_weights as _arc_filter_weights\n'
               '        loaded_weights = model.load_weights(_arc_filter_weights(self.get_all_weights(model_config, '
               'model)))\n', 1)],
}
TARGETS = (BASE_LOADER, DEFAULT_LOADER)


def load_mask(path, expected_sha256=None):
    """The mask file (``data/expert_prune/prune*.json``): format, 48 layers, distinct ids; either the same count
    pruned per layer (``pruned_per_layer``: int) or a layer-aware mask (``pruned_per_layer: "per-layer"`` with
    ``pruned_counts`` per layer and their ``pruned_total``), keeping at least top-10 experts per layer."""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"{path}: sha256 differs from the pinned {expected_sha256}")
    mask = json.loads(raw)
    if mask.get('format') != FORMAT:
        raise ValueError(f"{path}: not an {FORMAT} mask")
    experts, pruned = int(mask['experts']), mask['pruned']
    if (sorted(int(k) for k in pruned) != list(range(48))
            or any(len(set(ids)) != len(ids) or not all(0 <= int(e) < experts for e in ids)
                   for ids in pruned.values())):
        raise ValueError(f"{path}: expected 48 layers of distinct pruned ids in 0..{experts - 1}")
    per_layer = mask['pruned_per_layer']
    if per_layer == 'per-layer':
        counts = {str(k): int(v) for k, v in (mask.get('pruned_counts') or {}).items()}
        if (counts != {k: len(v) for k, v in pruned.items()} or sum(counts.values()) != mask.get('pruned_total')
                or any(experts - c < 10 for c in counts.values())):
            raise ValueError(f"{path}: pruned_counts and pruned_total must match the per-layer ids")
    elif isinstance(per_layer, bool) or not isinstance(per_layer, int) or {len(v) for v in pruned.values()} != {per_layer}:
        raise ValueError(f"{path}: expected 48 layers with {per_layer} distinct pruned ids each")
    return mask


def kept_summary(mask):
    """``kept_per_layer`` (the largest kept count: the config's ``num_experts``) and ``kept_total`` over layers;
    ``kept_min`` too for a layer-aware mask."""
    kept = [mask['experts'] - len(ids) for ids in mask['pruned'].values()]
    summary = {'kept_per_layer': max(kept), 'kept_total': sum(kept)}
    if len(set(kept)) > 1:
        summary['kept_min'] = min(kept)
    return summary


def validate(spec):
    """``server.expert_prune``: None, or ``{"mask": <relative path>, "sha256": <hex>}``."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {'mask', 'sha256'}:
        raise ValueError('expert_prune takes exactly mask and sha256')
    digest = spec['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('expert_prune.sha256 must be a sha256 hex digest')
    if not isinstance(spec['mask'], str) or spec['mask'].startswith('/') or '..' in spec['mask']:
        raise ValueError('expert_prune.mask is a bundle-relative path')
    return dict(spec)


def patched_source(target, original):
    if hashlib.sha256(original).hexdigest() != target['sha256']:
        raise RuntimeError(f"Expert-prune overlay rejected an unrecognized {target['relative']}")
    text = original.decode()
    for anchor, replacement, count in target['edits']:
        if text.count(anchor) != count:
            raise RuntimeError(f"Expert-prune overlay anchor count changed in {target['relative']}")
        text = text.replace(anchor, replacement)
    compile(text, target['relative'], 'exec')
    return text.encode()


def apply(site, output, mask):
    """Patch the extracted runtime under ``site`` with ``mask`` (a loaded mask dict); identity to ``output``."""
    site = Path(site)
    helper = Path(__file__).with_name('expert_prune_runtime.py').read_bytes()
    settings = {'experts': mask['experts'], 'fraction': mask['fraction'], 'pruned': mask['pruned']}
    if mask.get('stall_dump_seconds'):
        settings['stall_dump_seconds'] = int(mask['stall_dump_seconds'])  # diagnostics only
    settings = (json.dumps(settings, sort_keys=True) + '\n').encode()
    marker = Path(output) / MARKER
    originals = {t['relative']: (site / t['relative']).read_bytes() for t in TARGETS}
    if all(hashlib.sha256(originals[t['relative']]).hexdigest() == t['sha256'] for t in TARGETS):
        patched = {t['relative']: patched_source(t, originals[t['relative']]) for t in TARGETS}
        (site / HELPER).write_bytes(helper)
        (site / SETTINGS).write_bytes(settings)
        for relative, data in patched.items():
            (site / relative).write_bytes(data)
        identity = {'files': {t['relative']: {'original_sha256': t['sha256'],
                                              'patched_sha256': hashlib.sha256(patched[t['relative']]).hexdigest()}
                              for t in TARGETS},
                    'helper_sha256': hashlib.sha256(helper).hexdigest(),
                    'settings_sha256': hashlib.sha256(settings).hexdigest(),
                    'fraction': mask['fraction'], **kept_summary(mask)}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_EXPERT_PRUNE_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    # A restart in the same working directory re-extracts the runtime first; accept only this exact overlay.
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (all(hashlib.sha256(data).hexdigest() == identity['files'][rel]['patched_sha256']
                for rel, data in originals.items())
                and (site / HELPER).read_bytes() == helper and (site / SETTINGS).read_bytes() == settings):
            return identity
    raise RuntimeError('Expert-prune overlay found a modified or partially patched pinned file')
