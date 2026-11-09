"""Hash-guarded overlay of the sliding-window measurement probe (8 October; lab env ``ARENA_SWKV_PROBE=1``).

One pinned file gets three lines at its end (``common/qsa_cache.py``, which no other overlay edits): they hand the
raw-key cache class to ``vllm.arc_swkv_probe.install`` (``swkv_probe_runtime.py``), which registers the
``arc_swkv_probe`` RPC when the worker binds its caches. Nothing changes until that RPC switches a mask or a shift on,
and the function wrappers it needs are installed by the first RPC. Measurement servers only (``--enforce-eager``).
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_swkv_probe.py'
MARKER = 'swkv-probe-patch.json'
RELATIVE = 'vllm/models/qwen3_8_flash_next/common/qsa_cache.py'
SHA256 = 'e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8'
APPEND = ('\n\n# ARC swkv probe (measurement only): the RPC that switches the sliding-window emulation.\n'
          'from vllm import arc_swkv_probe as _arc_swkv_probe  # noqa: E402\n'
          '\n'
          '_arc_swkv_probe.install(QSAKeyStateCache)\n')


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != SHA256:
        raise RuntimeError(f'swkv probe overlay rejected an unrecognized {RELATIVE}')
    text = original.decode() + APPEND
    compile(text, RELATIVE, 'exec')
    return text.encode()


def apply(site, output):
    site = Path(site)
    helper = Path(__file__).with_name('swkv_probe_runtime.py').read_bytes()
    marker = Path(output) / MARKER
    target = site / RELATIVE
    original = target.read_bytes()
    if hashlib.sha256(original).hexdigest() == SHA256:
        patched = patched_source(original)
        (site / HELPER).write_bytes(helper)
        target.write_bytes(patched)
        identity = {'file': RELATIVE, 'original_sha256': SHA256,
                    'patched_sha256': hashlib.sha256(patched).hexdigest(),
                    'helper_sha256': hashlib.sha256(helper).hexdigest()}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_SWKV_PROBE_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    if marker.exists():  # a restart in the same working directory: only this exact overlay
        identity = json.loads(marker.read_text())
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and (site / HELPER).read_bytes() == helper):
            return identity
    raise RuntimeError('swkv probe overlay found a modified or partially patched pinned file')
