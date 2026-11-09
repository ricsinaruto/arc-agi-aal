"""CPU KV offloading for Qwen3.8-Flash-Next's hybrid cache (installed as ``vllm/arc_offload.py``; ARC-AGI-2 arm,
15 October).

The pinned ``OffloadingConnector`` refuses the QSA indexer's ring (``CircularBufferSpec``): one block a request
holding the raw keys of the group still being compressed, never prefix-cached. The GPU prefix cache leaves that
group out of every hit (``kv_cache_coordinator``: non-``prefix_cacheable`` groups are skipped) and a hit claims a
fresh ring block; a hit restored from CPU is the same case. So the connector here sees the cache without its
non-cacheable groups: its config, the per-group block ids the scheduler hands it (block ids, partial-tail handoffs), and the worker's group
list are filtered to the cacheable groups (full attention + indexer keys, the Mamba states). The ring's block is claimed by
the core as on any external hit (``CircularBufferManager.allocate_external_computed_blocks``).
"""

from dataclasses import replace
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector as _base
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import worker as _worker
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.kv_cache_interface import MambaSpec


def _cacheable(kv_cache_config):
    return [i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.kv_cache_spec.prefix_cacheable]


def _no_canonical_mappings(*args, **kwargs):
    # Only the canonical (cross-TP) layout reads them; it is off here, and the 4-bit per-token-head pages have
    # no canonical form.
    return {}


_worker.derive_canonical_mappings = _no_canonical_mappings


def _truncate_computed_blocks(self, blocks, num_computed_tokens):
    # The pinned method lets only Mamba groups end before the endpoint when an external hit supersedes a local
    # sub-block tail; the ring holds no hit blocks either (non-prefix-cacheable), so it is sliced the same way
    # (the engine died on its assert after 45 min of play, v23off3; throughput arm's diagnosis, 16 Oct).
    truncated = []
    for group_blocks, manager, group in zip(blocks.blocks, self.coordinator.single_type_managers,
                                            self.kv_cache_config.kv_cache_groups, strict=True):
        assert num_computed_tokens % manager.block_size == 0
        num_blocks = num_computed_tokens // manager.block_size
        if isinstance(group.kv_cache_spec, MambaSpec) or not group.kv_cache_spec.prefix_cacheable:
            num_blocks = min(num_blocks, len(group_blocks))
        else:
            assert num_blocks <= len(group_blocks)
        truncated.append(list(group_blocks[:num_blocks]))
    return self.create_kv_cache_blocks(tuple(truncated))


KVCacheManager.truncate_computed_blocks = _truncate_computed_blocks


class ArcOffloadingConnector(_base.OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        self._keep = _cacheable(kv_cache_config)
        self._all = len(kv_cache_config.kv_cache_groups)
        groups = [kv_cache_config.kv_cache_groups[i] for i in self._keep]
        filtered = replace(kv_cache_config, kv_cache_groups=groups)
        super().__init__(vllm_config, role, filtered)

    def _ids(self, groups):
        if groups is None:
            return None
        assert len(groups) == self._all, (len(groups), self._all)
        return tuple(groups[i] for i in self._keep)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return super().update_state_after_alloc(request, KVCacheBlocks(self._ids(blocks.blocks)),
                                                num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        new = [SimpleNamespace(req_id=r.req_id, block_ids=self._ids(r.block_ids))
               for r in scheduler_output.scheduled_new_reqs]
        cached = scheduler_output.scheduled_cached_reqs
        view = SimpleNamespace(req_ids=cached.req_ids, resumed_req_ids=cached.resumed_req_ids,
                               new_block_ids=[self._ids(g) for g in cached.new_block_ids])

        class Output:  # the scheduler output with the cacheable groups' block ids only
            def __getattr__(self, name):
                return getattr(scheduler_output, name)

        out = Output()
        out.scheduled_new_reqs, out.scheduled_cached_reqs = new, view
        tails = scheduler_output.partial_tail_offloads  # {req: [(core group index, block, boundary)]}
        if tails:
            index = {g: i for i, g in enumerate(self._keep)}
            tails = {req: [(index[g], block, boundary) for g, block, boundary in entries if g in index]
                     for req, entries in tails.items()}
            tails = {req: entries for req, entries in tails.items() if entries} or None
        out.partial_tail_offloads = tails
        return super().build_connector_meta(out)

    def request_finished_all_groups(self, request, block_ids):
        return super().request_finished_all_groups(request, self._ids(block_ids))
