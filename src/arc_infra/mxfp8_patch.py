"""Hash-guarded optional dispatch backport; historical runtime sources stay untouched."""

import hashlib
import json
from pathlib import Path

ENV = 'ARC_VLLM_ONLINE_MXFP8'
RELATIVE = 'vllm/model_executor/layers/linear.py'
ORIGINAL_SHA256 = 'fc4e96ba5e36895513b801b99e746e2b07959d402e36d56aae7d8ad3f124cfe7'
ANCHOR = '        self.return_bias = return_bias\n        self.disable_tp = disable_tp\n'
INSERT = '''        # ARC optional online MXFP8: preserve all already-quantized methods.
        from vllm.model_executor.layers.arc_online_mxfp8 import select
        self.quant_method = select(self, prefix, self.quant_method, UnquantizedLinearMethod)
'''


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError('Online MXFP8 backport rejected an unrecognized vLLM linear.py')
    text = original.decode()
    if text.count(ANCHOR) != 1:
        raise RuntimeError('Online MXFP8 dispatch anchor is not unique')
    result = text.replace(ANCHOR, INSERT + ANCHOR)
    compile(result, RELATIVE, 'exec')
    return result.encode()


def apply(site, output):
    target = Path(site) / RELATIVE
    original = target.read_bytes()
    helper = Path(__file__).with_name('mxfp8_runtime.py').read_bytes()
    marker = Path(output) / 'online-mxfp8-patch.json'
    helper_target = target.with_name('arc_online_mxfp8.py')
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and hashlib.sha256(helper).hexdigest() == identity['helper_sha256']
                and helper_target.read_bytes() == helper):
            return identity
        raise RuntimeError('Previously patched MXFP8 runtime identity changed')
    patched = patched_source(original)
    helper_target.write_bytes(helper)
    target.write_bytes(patched)
    identity = {'original_sha256': ORIGINAL_SHA256,
                'patched_sha256': hashlib.sha256(patched).hexdigest(),
                'helper_sha256': hashlib.sha256(helper).hexdigest(),
                'scope': 'unquantized attention projections only',
                'kernel': 'FlashInferCutlassMxfp8LinearKernel', 'kv_dtype': 'bfloat16'}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('ARC_ONLINE_MXFP8_PATCH ' + json.dumps(identity), flush=True)
    return identity
