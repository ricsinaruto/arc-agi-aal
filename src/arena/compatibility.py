"""Fail locally for known unsupported Flash-Next cache variants in pinned vLLM.

Evidence and source hashes: docs/validation-20260923/cache-compatibility.json.
Remove these guards only after a validated backend implementation replaces the pin.
"""


def validate_settings(settings):
    if settings['kv_cache_dtype'] not in ('auto', 'bfloat16'):
        raise NotImplementedError('Pinned Flash-Next QSA requires BF16 KV; FP8 needs backend/kernel support')
    # cpu_cache_gib: the pinned OffloadingConnector rejects the QSA ring (CircularBufferSpec); served through
    # arc_infra.arc_offload_runtime (vllm/arc_offload.py), which leaves the non-prefix-cacheable groups out.
