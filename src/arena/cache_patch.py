"""Hash-guarded telemetry-only insertion into a disposable extracted vLLM runtime."""
import hashlib
import json
from pathlib import Path

RELATIVE = 'vllm/v1/core/kv_cache_utils.py'
ORIGINAL_SHA256 = 'f6643bad29b7b3cb044482b6736a042afa91a4f43a11f034317a8e8029d3a8a4'
ANCHOR = '    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)\n'
INSERT = '    from vllm.v1.core.arc_cache_export import emit\n    emit(vllm_config, kv_cache_config)\n'
MANAGER_RELATIVE = 'vllm/v1/core/kv_cache_manager.py'
MANAGER_SHA256 = '2d20c3d98845cfd8d88a2f66b8fc6402ea1fcfbee16879d2364a4a9e45b8fa47'


def patched_manager(original):
    if hashlib.sha256(original).hexdigest() != MANAGER_SHA256:
        raise RuntimeError('Cache diagnostics rejected unrecognized pinned vLLM manager')
    text = original.decode()
    anchors = [
        ('            if required_blocks > self.block_pool.get_num_free_blocks():\n',
         'full_sequence', 'self.block_pool.get_num_free_blocks()', '0', '                '),
        ('        if required_blocks > available_blocks:\n',
         'incremental', 'available_blocks', 'reserved_blocks', '            '),
    ]
    for anchor, stage, available, reserved, indent in anchors:
        if text.count(anchor) != 1:
            raise RuntimeError('Cache diagnostics insertion must have one anchor')
        insertion = (f'{indent}from vllm.v1.core.arc_cache_failure import emit_failure\n'
                     f'{indent}emit_failure(self, request, stage={stage!r},\n'
                     f'{indent}             required=required_blocks, available={available},\n'
                     f'{indent}             reserved={reserved}, watermark=watermark_blocks)\n')
        text = text.replace(anchor, anchor + insertion)
    compile(text, MANAGER_RELATIVE, 'exec')
    return text.encode()


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError('Cache telemetry rejected unrecognized pinned vLLM source')
    text = original.decode()
    if text.count(ANCHOR) != 1:
        raise RuntimeError('Cache telemetry insertion must have one anchor')
    result = text.replace(ANCHOR, INSERT + ANCHOR)
    compile(result, RELATIVE, 'exec')
    return result.encode()


def apply(site, output):
    target = Path(site) / RELATIVE
    helper = Path(__file__).with_name('cache_export_runtime.py').read_bytes()
    helper_target = target.with_name('arc_cache_export.py')
    marker = Path(output) / 'cache-layout-patch.json'
    original = target.read_bytes()
    manager_target = Path(site) / MANAGER_RELATIVE
    failure_helper = Path(__file__).with_name('cache_failure_runtime.py').read_bytes()
    failure_target = target.with_name('arc_cache_failure.py')
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and helper_target.read_bytes() == helper
                and hashlib.sha256(manager_target.read_bytes()).hexdigest() == identity.get('manager_patched_sha256')
                and failure_target.read_bytes() == failure_helper):
            return identity
        if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
            raise RuntimeError('Cache telemetry runtime identity changed')
        # A server restart in the same working directory re-extracts the pinned runtime (serving A/B
        # bench, 26 September): the files are the pinned originals again, so the same patch re-applies.
    patched = patched_source(original)
    manager_patched = patched_manager(manager_target.read_bytes())
    helper_target.write_bytes(helper)
    failure_target.write_bytes(failure_helper)
    target.write_bytes(patched)
    manager_target.write_bytes(manager_patched)
    identity = {'original_sha256': ORIGINAL_SHA256,
                'patched_sha256': hashlib.sha256(patched).hexdigest(),
                'helper_sha256': hashlib.sha256(helper).hexdigest(),
                'manager_original_sha256': MANAGER_SHA256,
                'manager_patched_sha256': hashlib.sha256(manager_patched).hexdigest(),
                'failure_helper_sha256': hashlib.sha256(failure_helper).hexdigest()}
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return identity
