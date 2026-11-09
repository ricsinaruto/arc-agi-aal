"""Hash-guarded expert-usage overlay (28 September; docs/validation-20260927/expert-usage.md).

``server.expert_probe = true`` patches one pinned file, the fused-MoE runner: right after the router
logits exist (``MoERunner._forward_impl``, after the gate), they pass through
``vllm.arc_expert_probe.on_router_logits`` (``expert_probe_runtime.py``), which records routing statistics
and applies the pruning mask. The helper is copied next to the pinned sources; the RPC it installs on the
worker (``arc_expert_probe``) is reached through ``/collective_rpc`` (the arena already serves with
``VLLM_SERVER_DEV_MODE=1``). A study without the key serves the pinned runtime unchanged.
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_expert_probe.py'
MARKER = 'expert-probe-patch.json'
RUNNER = {
    'relative': 'vllm/model_executor/layers/fused_moe/runner/moe_runner.py',
    'sha256': 'a19ecf2417c80b93c0fe5594e74528f686665847732196f9b29ef967b38110b0',
    'edits': [('            else:\n'
               '                router_logits, _ = self.gate(hidden_states)\n'
               '\n'
               '        with self._sequence_parallel_context():\n',
               '            else:\n'
               '                router_logits, _ = self.gate(hidden_states)\n'
               '\n'
               '        # ARC (expert usage): routing statistics and the pruning mask.\n'
               '        from vllm.arc_expert_probe import on_router_logits as _arc_on_router_logits\n'
               '        router_logits = _arc_on_router_logits(self, router_logits)\n'
               '\n'
               '        with self._sequence_parallel_context():\n', 1)],
}


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != RUNNER['sha256']:
        raise RuntimeError(f"Expert-probe overlay rejected an unrecognized {RUNNER['relative']}")
    text = original.decode()
    for anchor, replacement, count in RUNNER['edits']:
        if text.count(anchor) != count:
            raise RuntimeError(f"Expert-probe overlay anchor count changed in {RUNNER['relative']}")
        text = text.replace(anchor, replacement)
    compile(text, RUNNER['relative'], 'exec')
    return text.encode()


def apply(site, output):
    """Patch the extracted runtime under ``site``; the identity goes to ``output``."""
    site = Path(site)
    helper = Path(__file__).with_name('expert_probe_runtime.py').read_bytes()
    marker = Path(output) / MARKER
    target = site / RUNNER['relative']
    original = target.read_bytes()
    if hashlib.sha256(original).hexdigest() == RUNNER['sha256']:
        patched = patched_source(original)
        (site / HELPER).write_bytes(helper)
        target.write_bytes(patched)
        identity = {'file': RUNNER['relative'], 'original_sha256': RUNNER['sha256'],
                    'patched_sha256': hashlib.sha256(patched).hexdigest(),
                    'helper_sha256': hashlib.sha256(helper).hexdigest()}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_EXPERT_PROBE_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    # A restart in the same working directory: accept only this exact overlay, never re-patch.
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and (site / HELPER).read_bytes() == helper):
            return identity
    raise RuntimeError('Expert-probe overlay found a modified or partially patched pinned file')
