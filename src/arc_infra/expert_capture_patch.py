"""Hash-guarded MoE-input capture overlay (28 September; docs/validation-20260927/expert-quant.md).

``server.expert_capture = true`` patches one pinned file, the fused-MoE runner: right after the router has selected
the experts (``MoERunner._apply_quant_method``, modular kernels, which the served FlashInfer CUTLASS NvFp4 experts
are), the MoE input rows and the served routing (top-10 ids and renormalized weights) pass through
``vllm.arc_expert_capture.on_routed`` (``expert_capture_runtime.py``). The helper records nothing until switched on
through ``/collective_rpc`` (``arc_expert_capture``; the arena serves with ``VLLM_SERVER_DEV_MODE=1``) and never
during CUDA-graph capture. A study without the key serves the pinned runtime unchanged.
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_expert_capture.py'
MARKER = 'expert-capture-patch.json'
ANCHOR = ('            topk_weights, topk_ids = self.router.select_experts(\n'
          '                hidden_states=hidden_states,\n'
          '                router_logits=router_logits,\n'
          '                topk_indices_dtype=self._quant_method.topk_indices_dtype,\n'
          '                input_ids=input_ids,\n'
          '            )\n')
RUNNER = {
    'relative': 'vllm/model_executor/layers/fused_moe/runner/moe_runner.py',
    'sha256': 'a19ecf2417c80b93c0fe5594e74528f686665847732196f9b29ef967b38110b0',
    'edits': [(ANCHOR,
               ANCHOR
               + '            # ARC (expert quantization): MoE input rows and the served routing, when switched on.\n'
               '            from vllm.arc_expert_capture import on_routed as _arc_on_routed\n'
               '            _arc_on_routed(self, hidden_states, topk_weights, topk_ids)\n', 1)],
}


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != RUNNER['sha256']:
        raise RuntimeError(f"Expert-capture overlay rejected an unrecognized {RUNNER['relative']}")
    text = original.decode()
    for anchor, replacement, count in RUNNER['edits']:
        if text.count(anchor) != count:
            raise RuntimeError(f"Expert-capture overlay anchor count changed in {RUNNER['relative']}")
        text = text.replace(anchor, replacement)
    compile(text, RUNNER['relative'], 'exec')
    return text.encode()


def apply(site, output):
    """Patch the extracted runtime under ``site``; the identity goes to ``output``."""
    site = Path(site)
    helper = Path(__file__).with_name('expert_capture_runtime.py').read_bytes()
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
        print('ARC_EXPERT_CAPTURE_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    # A restart in the same working directory: accept only this exact overlay, never re-patch.
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and (site / HELPER).read_bytes() == helper):
            return identity
    raise RuntimeError('Expert-capture overlay found a modified or partially patched pinned file')
