"""Hash-guarded served-routing overlay for R3.

``server.served_route = DIR`` appends two lines to the pinned ``vllm/v1/engine/output_processor.py`` that install
``vllm.arc_served_route`` on its RequestState; the server also needs ``--enable-return-routed-experts``
(``SERVER_FLAGS``, added by ``arena.serving``). A server without the key serves the pinned runtime unchanged.

Flash's KV groups wrap their layer specs (hybrid model): vLLM's routed-experts capture looked only for a bare
FullAttentionSpec group and refused to start (R3 check v2, 20 Oct). ``CAPTURER`` lets it take a group whose layers
include uncompressed full attention (same block table and slot layout) and logs the group it picked.
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_served_route.py'
MARKER = 'served-route-patch.json'
SERVER_FLAGS = ['--enable-return-routed-experts']
TAIL = ('\n\n# ARC (R3): each finished request\'s served routing, written under the overlay\'s directory.\n'
        'from vllm.arc_served_route import install as _arc_served_route_install  # noqa: E402\n'
        '_arc_served_route_install(RequestState)\n')
TARGET = {
    'relative': 'vllm/v1/engine/output_processor.py',
    'sha256': 'c310a768931ab46ec9ac14ccbb433f07253b65553b7c859d1ca10f453efcb37c',
}


GID_OLD = """    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, FullAttentionSpec):
            return gid
    raise ValueError("Routed-experts capture requires a full-attention KV cache group.")
"""
GID_NEW = """    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, FullAttentionSpec):
            return gid
    # ARC (R3): a hybrid model's group wraps its layer specs; take one with uncompressed full attention.
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        specs = list(getattr(group.kv_cache_spec, "kv_cache_specs", {}).values())
        if any(isinstance(s, FullAttentionSpec) and getattr(s, "compress_ratio", 1) == 1 for s in specs):
            logger.info("ARC routed-experts group %d: %s block_size %d", gid,
                        sorted({type(s).__name__ for s in specs}), group.kv_cache_spec.block_size)
            return gid
    raise ValueError("Routed-experts capture requires a full-attention KV cache group.")
"""
CAPTURER = {
    'relative': 'vllm/model_executor/layers/fused_moe/routed_experts_capturer.py',
    'sha256': '7b15100170d5f3b49d89b1ab29eaf311a3a56551ba26ec18b7cda6bec367316e',
}


def patched_capturer(original):
    if hashlib.sha256(original).hexdigest() != CAPTURER['sha256']:
        raise RuntimeError(f"Served-route overlay rejected an unrecognized {CAPTURER['relative']}")
    text = original.decode()
    if text.count(GID_OLD) != 1:
        raise RuntimeError(f"Served-route overlay: group lookup changed in {CAPTURER['relative']}")
    text = text.replace(GID_OLD, GID_NEW)
    compile(text, CAPTURER['relative'], 'exec')
    return text.encode()


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != TARGET['sha256']:
        raise RuntimeError(f"Served-route overlay rejected an unrecognized {TARGET['relative']}")
    text = original.decode()
    if text.count('\nclass RequestState:') != 1:
        raise RuntimeError(f"Served-route overlay: RequestState not found once in {TARGET['relative']}")
    text = text.rstrip('\n') + TAIL
    compile(text, TARGET['relative'], 'exec')
    return text.encode()


def helper_source(directory):
    text = Path(__file__).with_name('served_route_runtime.py').read_text()
    line = 'DIR = None  # filled in by the overlay\n'
    if text.count(line) != 1:
        raise RuntimeError('served_route_runtime.py lost its DIR line')
    return text.replace(line, f'DIR = {str(directory)!r}\n').encode()


def apply(site, output, directory):
    """Patch the extracted runtime under ``site`` to record routing under ``directory``; identity to ``output``."""
    site, directory = Path(site), Path(directory)
    if not directory.is_absolute():
        raise ValueError('served_route needs an absolute directory')
    helper = helper_source(directory)
    marker = Path(output) / MARKER
    target = site / TARGET['relative']
    capturer = site / CAPTURER['relative']
    original = target.read_bytes()
    original_capturer = capturer.read_bytes()
    if hashlib.sha256(original).hexdigest() == TARGET['sha256']:
        patched = patched_source(original)
        patched_cap = patched_capturer(original_capturer)
        (site / HELPER).write_bytes(helper)
        target.write_bytes(patched)
        capturer.write_bytes(patched_cap)
        identity = {'file': TARGET['relative'], 'original_sha256': TARGET['sha256'],
                    'patched_sha256': hashlib.sha256(patched).hexdigest(),
                    'capturer': CAPTURER['relative'], 'capturer_patched_sha256': hashlib.sha256(patched_cap).hexdigest(),
                    'helper_sha256': hashlib.sha256(helper).hexdigest(), 'dir': str(directory),
                    'server_flags': SERVER_FLAGS}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_SERVED_ROUTE_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    if marker.exists():  # a restart in the same working directory: only this exact overlay
        identity = json.loads(marker.read_text())
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and hashlib.sha256(original_capturer).hexdigest() == identity.get('capturer_patched_sha256')
                and (site / HELPER).read_bytes() == helper):
            return identity
    raise RuntimeError('Served-route overlay found a modified or partially patched pinned file')
