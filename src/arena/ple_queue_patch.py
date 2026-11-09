"""PLE offload request queue: wait for the slot instead of failing (RL arm, 15 Oct; rl1-loop-f1 crash).

The pinned ``vllm/v1/ple_offload/connector.py`` hands each forward's batch sizes to its staging thread through a
one-slot queue with ``put_nowait``. Under async scheduling the next forward can be prepared before the staging thread
has taken the previous request; with the host CPU busy (trainer and 64 agent streams on one box) the slot was still
full and the worker raised ``queue.Full`` (rl1-loop-f1, 15 Oct 01:01 UTC), which hung the engine until the 300 s
``sample_tokens`` timeout. The patch blocks for up to ``wait`` seconds instead; the staging thread drains the slot on
its own, so only a real stall still raises. Served values are unchanged.
"""
import hashlib
import json
from pathlib import Path

RELATIVE = 'vllm/v1/ple_offload/connector.py'
SHA256 = 'b911d00cf4a39556d946e86ac34ea70e6230683b22adf5ef14ac299e9ef36206'
ANCHOR = '        self._request_queue.put_nowait(request)\n'
MARKER = 'ple-queue-patch.json'


def patched(original, wait):
    if hashlib.sha256(original).hexdigest() != SHA256:
        raise RuntimeError(f'PLE queue patch rejected an unrecognized pinned {RELATIVE}')
    text = original.decode()
    if text.count(ANCHOR) != 1:
        raise RuntimeError(f'PLE queue patch needs exactly one anchor in {RELATIVE}')
    text = text.replace(ANCHOR, '        # ARC ple_queue_wait: wait for the staging thread to take the previous request.\n'
                                f'        self._request_queue.put(request, timeout={float(wait)!r})\n')
    compile(text, RELATIVE, 'exec')
    return text.encode()


def apply(site, output, wait):
    """Patch the extracted runtime (idempotent; re-applied after a restart re-extracts it)."""
    target, marker = Path(site) / RELATIVE, Path(output) / MARKER
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    current = hashlib.sha256(target.read_bytes()).hexdigest()
    if previous.get('patched_sha256') == current and previous.get('wait') == float(wait):
        return previous
    text = patched(target.read_bytes(), wait)
    target.write_bytes(text)
    identity = {'relative': RELATIVE, 'wait': float(wait), 'original_sha256': current,
                'patched_sha256': hashlib.sha256(text).hexdigest()}
    marker.write_text(json.dumps(identity, indent=2))
    return identity
