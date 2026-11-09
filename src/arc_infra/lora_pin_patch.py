"""ARC_LORA_NO_PIN=1: vLLM keeps an adapter's host copy pageable instead of pinning it tensor by tensor
(mtpcheck-v6, 26 Oct: add_lora spent ~115 s in lora_model.py's per-tensor pin_memory() on the 5.2 GiB s0017 adapter,
first-time host pinning on a fresh VM; v4's 600 s load timeout is the same path). With one adapter per phase the host
copy goes to its GPU slot once, so pinning buys nothing. Outputs are unchanged (same tensors, same values).
Anchor-based (the expert-mask compact overlay also edits model_manager.py); raises on anything unrecognized."""

import hashlib
import json
from pathlib import Path

ENV = 'ARC_LORA_NO_PIN'
GUARD = ' and __import__("os").environ.get("ARC_LORA_NO_PIN") != "1"  # ARC'
TARGETS = (('vllm/lora/lora_model.py', '        pin_memory = str(device) == "cpu" and PIN_MEMORY\n'),
           ('vllm/lora/model_manager.py', '        pin_memory = str(lora_device) == "cpu" and PIN_MEMORY\n'))


def patched_source(original, anchor, relative):
    text = original.decode()
    if ENV in text:
        raise RuntimeError(f'lora_pin patch: {relative} already carries the edit')
    if text.count(anchor) != 1:
        raise RuntimeError(f'lora_pin anchor is not unique in {relative}')
    text = text.replace(anchor, anchor.rstrip('\n') + GUARD + '\n')
    compile(text, relative, 'exec')
    return text.encode()


def apply(site, output):
    """Both files edited (or both already carrying the edit); raises otherwise, so a boot never runs half-patched."""
    identity = {}
    for relative, anchor in TARGETS:
        target = Path(site) / relative
        original = target.read_bytes()
        seen = hashlib.sha256(original).hexdigest()
        if ENV.encode() in original:
            identity[relative] = {'patched_sha256': seen, 'already': True}
            continue
        patched = patched_source(original, anchor, relative)
        target.write_bytes(patched)
        identity[relative] = {'original_sha256': seen, 'patched_sha256': hashlib.sha256(patched).hexdigest()}
    marker = Path(output) / 'lora-pin-patch.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('ARC_LORA_PIN_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
    return identity
