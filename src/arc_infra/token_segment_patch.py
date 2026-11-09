"""ARC_TOKEN_SEGMENT_CACHE=1: hash-guarded edit of transformers' TokenizersBackend._encode_plus so its Rust
``encode_batch`` call goes through arc_infra.token_segment_cache (prompt pieces between special tokens cached; ids
identical by construction and by test). The API process tokenizes only each turn's new messages (G-long v5, 27 Oct:
51% of the one API process's samples re-encoded whole conversations)."""

import hashlib
import json
from pathlib import Path

ENV = 'ARC_TOKEN_SEGMENT_CACHE'
RELATIVE = 'transformers/tokenization_utils_tokenizers.py'
ORIGINAL_SHA256 = None  # set from the runtime's file on first apply (logged); the anchor check is the guard
OLD = '        encodings = self._tokenizer.encode_batch(\n'
NEW = ('        encodings = __import__("arc_infra.token_segment_cache", fromlist=["encode_batch"]).encode_batch(  # ARC\n'
       '            self._tokenizer,\n')


def patched_source(original):
    text = original.decode()
    if 'arc_infra.token_segment_cache' in text:
        raise RuntimeError('token_segment patch: file already carries the edit')
    if text.count(OLD) != 1:
        raise RuntimeError('token_segment anchor is not unique')
    text = text.replace(OLD, NEW)
    compile(text, RELATIVE, 'exec')
    return text.encode()


def apply(site, output):
    """Never raises: an unrecognized file leaves tokenization as it was (ARC_TOKEN_SEGMENT_PATCH_SKIPPED)."""
    target = Path(site) / RELATIVE
    original = target.read_bytes()
    seen = hashlib.sha256(original).hexdigest()
    marker = Path(output) / 'token-segment-cache.json'  # not *-patch.json: the boot-cache key ignores it (no compiled code)
    if b'arc_infra.token_segment_cache' in original:
        return json.loads(marker.read_text()) if marker.exists() else {'patched_sha256': seen}
    try:
        patched = patched_source(original)
    except Exception as error:  # noqa: BLE001 - a throughput cut, never a reason to fail a boot
        print('ARC_TOKEN_SEGMENT_PATCH_SKIPPED ' + json.dumps({'sha256': seen, 'error': str(error)[:200]}), flush=True)
        return None
    target.write_bytes(patched)
    identity = {'original_sha256': seen, 'patched_sha256': hashlib.sha256(patched).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('ARC_TOKEN_SEGMENT_PATCH ' + json.dumps(identity), flush=True)
    return identity
