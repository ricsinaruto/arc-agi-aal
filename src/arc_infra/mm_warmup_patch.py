"""ARC_MM_WARMUP_SKIP=1: hash-guarded edit of vLLM's renderer warmup so the start-up multi-modal processor warmup
(a dummy max-size image at max_model_len, ~33 s of API-process CPU before the server listens; RTX play boot, MFU-2
24 Oct) is skipped. Outputs are unchanged: the warmup only pre-runs the processor; a first real image request pays
its own processing. The chat-template warmup still runs."""

import hashlib
import json
from pathlib import Path

ENV = 'ARC_MM_WARMUP_SKIP'
RELATIVE = 'vllm/renderers/base.py'
ORIGINAL_SHA256 = 'a358921359b241df2e5c08b547897b874a0eb5c73bc9051293bab800511aeedb'
EDITS = (('            if self.mm_processor:\n',
          '            if self.mm_processor and __import__("os").environ.get("ARC_MM_WARMUP_SKIP") != "1":  # ARC\n'),
         ('            if self._readonly_mm_processor is not None:\n',
          '            if self._readonly_mm_processor is not None and __import__("os").environ.get(\n'
          '                    "ARC_MM_WARMUP_SKIP") != "1":  # ARC\n'))


def patched_source(original):
    """The edited file; any build where both anchors are unique (the pinned file, or it after serving_fast's
    mm_warmup_overlap edit, which keeps both anchors). Raises on anything else; ``apply`` turns that into a skip."""
    text = original.decode()
    if "ARC_MM_WARMUP_SKIP" in text:
        raise RuntimeError('mm_warmup patch: file already carries the edit')
    for old, new in EDITS:
        if text.count(old) != 1:
            raise RuntimeError(f'mm_warmup anchor is not unique: {old.strip()}')
        text = text.replace(old, new)
    compile(text, RELATIVE, 'exec')
    return text.encode()


def apply(site, output):
    """Never raises: an unrecognized file leaves the warmup on (ARC_MM_WARMUP_PATCH_SKIPPED)."""
    target = Path(site) / RELATIVE
    original = target.read_bytes()
    seen = hashlib.sha256(original).hexdigest()
    marker = Path(output) / 'mm-warmup-skip.json'  # not *-patch.json: the boot-cache key ignores it (no compiled code)
    if b"ARC_MM_WARMUP_SKIP" in original:  # applied earlier in this runtime tree
        identity = json.loads(marker.read_text()) if marker.exists() else {'patched_sha256': seen}
        return identity
    try:
        patched = patched_source(original)
    except Exception as error:  # noqa: BLE001 - a boot cut, never a reason to fail a boot
        print('ARC_MM_WARMUP_PATCH_SKIPPED ' + json.dumps({'sha256': seen, 'error': str(error)[:200]}), flush=True)
        return None
    target.write_bytes(patched)
    identity = {'original_sha256': seen, 'pinned': seen == ORIGINAL_SHA256,
                'patched_sha256': hashlib.sha256(patched).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('ARC_MM_WARMUP_PATCH ' + json.dumps(identity), flush=True)
    return identity
