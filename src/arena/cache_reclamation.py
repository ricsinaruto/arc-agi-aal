"""Opt-in fix for sparse skipped-block reclamation in the pinned vLLM runtime.

Only modifies a disposable extracted runtime. It preserves the caller's processed
token boundary; it never frees a block outside the already-authorized range.
"""
import hashlib
import json
from pathlib import Path

RELATIVE = 'vllm/v1/core/single_type_kv_cache_manager.py'
ORIGINAL_SHA256 = '9ec6cbc0c78f30986f294439f02f57a9586ffbf2939d8f5c4fb968a642625ca2'
ANCHOR = '''        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
'''
REPLACEMENT = ANCHOR.replace('                break\n', '                continue\n')


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError('Sparse cache reclamation rejected unrecognized pinned source')
    text = original.decode()
    if text.count(ANCHOR) != 1:
        raise RuntimeError('Sparse cache reclamation needs exactly one loop anchor')
    result = text.replace(ANCHOR, REPLACEMENT)
    compile(result, RELATIVE, 'exec')
    return result.encode()


def apply(site, output):
    target = Path(site) / RELATIVE
    marker = Path(output) / 'cache-reclamation-patch.json'
    original = target.read_bytes()
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (identity.get('original_sha256') == ORIGINAL_SHA256
                and hashlib.sha256(original).hexdigest() == identity.get('patched_sha256')):
            return identity
        if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
            raise RuntimeError('Sparse cache reclamation runtime identity changed')
        # Re-extracted pinned runtime after a server restart: re-apply the same patch.
    patched = patched_source(original)
    identity = {'original_sha256': ORIGINAL_SHA256,
                'patched_sha256': hashlib.sha256(patched).hexdigest(),
                'change': 'Skip null gaps while freeing only the requested processed-token range',
                'gpu_validated': False}
    target.write_bytes(patched)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return identity
