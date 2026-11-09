"""Runs inside pinned vLLM; export its own conservative per-request block costs."""
import copy
import json


def describe(config, cache):
    maximum = config.model_config.max_model_len
    groups = cache.kv_cache_groups
    rows = []
    probe = copy.copy(config)
    probe.model_config = copy.copy(config.model_config)
    for length in sorted({1, maximum, *range(1024, maximum, 1024)}):
        probe.model_config.max_model_len = length
        costs = []
        for group in groups:
            spec = group.kv_cache_spec
            page = spec.page_size_bytes
            costs.append((spec.max_memory_usage_bytes(probe) + page - 1) // page)
        rows.append({'tokens': length, 'blocks': sum(costs), 'groups': costs})
    tensors = getattr(cache, 'kv_cache_tensors', None)
    allocation = None
    if tensors is not None:
        # Match the pinned model runner: packed views alias ONE backing buffer.
        # Summing every view would substantially overstate available cache bytes.
        packed = [t for t in tensors if getattr(t, 'block_stride', 0) > 0]
        allocation = sum(t.size for t in tensors if getattr(t, 'block_stride', 0) == 0)
        if packed:
            allocation += packed[0].size
    return {
        'schema': 1, 'num_blocks': cache.num_blocks, 'max_model_len': maximum,
        'planned_cache_bytes': allocation,
        'lookahead': config.speculative_config.num_speculative_tokens if config.speculative_config else 0,
        'groups': [{'type': type(g.kv_cache_spec).__name__, 'block_size': g.kv_cache_spec.block_size,
                    'page_size_bytes': g.kv_cache_spec.page_size_bytes, 'layers': g.layer_names}
                   for g in groups],
        'costs': rows,
        'rule': 'sum ceil(spec.max_memory_usage_bytes(length)/spec.page_size_bytes); no shared-prefix discount',
    }


def emit(config, cache):
    print('ARC_CACHE_LAYOUT ' + json.dumps(describe(config, cache), separators=(',', ':')), flush=True)
