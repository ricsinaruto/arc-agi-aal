"""Fail locally for known unsupported Flash-Next cache variants in pinned vLLM.

Evidence and source hashes: docs/validation-20260923/cache-compatibility.json.
Remove these guards only after a validated backend implementation replaces the pin.
"""


def validate_settings(settings):
    if settings['kv_cache_dtype'] not in ('auto', 'bfloat16'):
        raise NotImplementedError('Pinned Flash-Next QSA requires BF16 KV; FP8 needs backend/kernel support')
    if settings['cpu_cache_gib']:
        raise NotImplementedError('Pinned offloading scheduler rejects QSA CircularBufferSpec; CPU cache needs backend support')
