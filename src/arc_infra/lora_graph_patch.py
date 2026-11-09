"""ARC_LORA_GRAPH_ONE=1: the V2 model runner captures only the LoRA cudagraph case (num_active_loras = max_loras + 1)
instead of [0, max_loras + 1], and a batch with no active adapter replays that LoRA graph (its punica rows are all
-1: _set_active_loras runs on every real batch). Halves capture (B300 hero v6f2: 74 piecewise + 44 full graphs, 83 s)
and graph memory; a base-only batch pays the LoRA ops. vLLM's own cudagraph_specialize_lora=False does not do this
in the V2 runner (it still returns [0, max_loras + 1])."""

import hashlib
import json
from pathlib import Path

ENV = 'ARC_LORA_GRAPH_ONE'
FILES = {
    'vllm/v1/worker/gpu/lora_utils.py': (
        '1e90eddd9ddb4ca55e574bb28e87ae3a6f6750175a62d373102426543f4ccd66',
        (('    if compilation_config.cudagraph_specialize_lora:\n',
          '    if __import__("os").environ.get("ARC_LORA_GRAPH_ONE") == "1":  # ARC lora_graph_patch\n'
          '        return [lora_config.max_loras + 1]\n'
          '    if compilation_config.cudagraph_specialize_lora:\n'),)),
    'vllm/v1/worker/gpu/cudagraph_utils.py': (
        'ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c',
        (('        if num_active_loras <= 0 or not self._lora_dispatch_map:\n',
          '        if (num_active_loras <= 0 and self._lora_dispatch_map and 0 not in self.lora_capture_cases\n'
          '                and __import__("os").environ.get("ARC_LORA_GRAPH_ONE") == "1"):  # ARC lora_graph_patch\n'
          '            return self._max_lora_case\n'
          '        if num_active_loras <= 0 or not self._lora_dispatch_map:\n'),)),
}


def patched_source(relative, original):
    text = original.decode()
    if 'ARC lora_graph_patch' in text:
        raise RuntimeError(f'lora_graph patch: {relative} already carries the edit')
    for old, new in FILES[relative][1]:
        if text.count(old) != 1:
            raise RuntimeError(f'lora_graph anchor is not unique in {relative}: {old.strip()}')
        text = text.replace(old, new)
    compile(text, relative, 'exec')
    return text.encode()


def apply(site, output):
    """Both files or neither. Never raises: an unrecognized build keeps both cases (ARC_LORA_GRAPH_PATCH_SKIPPED).
    Marker lora-graph-patch.json (a *-patch.json: the boot-cache key covers it, the captured graphs differ)."""
    site, marker = Path(site), Path(output) / 'lora-graph-patch.json'
    originals = {rel: (site / rel).read_bytes() for rel in FILES}
    if all(b'ARC lora_graph_patch' in data for data in originals.values()):
        return json.loads(marker.read_text()) if marker.exists() else {'applied_earlier': True}
    try:
        patched = {rel: patched_source(rel, data) for rel, data in originals.items()}
    except Exception as error:  # noqa: BLE001 - a boot cut, never a reason to fail a boot
        print('ARC_LORA_GRAPH_PATCH_SKIPPED ' + json.dumps({'error': str(error)[:200]}), flush=True)
        return None
    identity = {}
    for rel, data in patched.items():
        (site / rel).write_bytes(data)
        seen = hashlib.sha256(originals[rel]).hexdigest()
        identity[rel] = {'original_sha256': seen, 'pinned': seen == FILES[rel][0],
                         'patched_sha256': hashlib.sha256(data).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('ARC_LORA_GRAPH_PATCH ' + json.dumps(identity), flush=True)
    return identity
