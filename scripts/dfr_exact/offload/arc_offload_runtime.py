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

import json as _json
import os as _os
from dataclasses import replace
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector as _base
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import worker as _worker
from vllm.v1.core import kv_cache_manager as _kvm
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import MambaSpec as _MambaSpec


# Diagnostics (throughput arm, 16 Oct): per-request lookup and finish records when ARC_OFFLOAD_TRACE=1.
_TRACE = _os.environ.get("ARC_OFFLOAD_TRACE") == "1"


def _cacheable(kv_cache_config):
    return [i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.kv_cache_spec.prefix_cacheable]


def _no_canonical_mappings(*args, **kwargs):
    # Only the canonical (cross-TP) layout reads them; it is off here, and the 4-bit per-token-head pages have
    # no canonical form.
    return {}


_worker.derive_canonical_mappings = _no_canonical_mappings


def _truncate_computed_blocks(self, blocks, num_computed_tokens):
    # Throughput arm (16 Oct): the core's KVCacheManager.truncate_computed_blocks (an external hit superseding a
    # local sub-block tail) exempts only Mamba groups from "every group covers the endpoint"; the QSA ring
    # (non-prefix_cacheable, never part of a local hit) failed the assert and killed the engine (vo16d v2).
    # Non-cacheable groups are sliced like Mamba's; the core claims the ring's fresh block as on any external hit.
    truncated = []
    for group_blocks, manager, group in zip(blocks.blocks, self.coordinator.single_type_managers,
                                            self.kv_cache_config.kv_cache_groups, strict=True):
        spec = group.kv_cache_spec
        assert num_computed_tokens % manager.block_size == 0
        num_blocks = num_computed_tokens // manager.block_size
        if isinstance(spec, _MambaSpec) or not spec.prefix_cacheable:
            num_blocks = min(num_blocks, len(group_blocks))
        else:
            assert num_blocks <= len(group_blocks)
        truncated.append(list(group_blocks[:num_blocks]))
    return self.create_kv_cache_blocks(tuple(truncated))


_kvm.KVCacheManager.truncate_computed_blocks = _truncate_computed_blocks


# Throughput arm (16 Oct, ARC_OFFLOAD_EAGLE_TAIL=1): the draft group's store drops the whole trailing full chunk while
# decoding, and the lookup pops one more chunk (the eagle recompute, as the GPU prefix cache drops its last block):
# a restored turn lost two chunks where the GPU path loses one (vo16e trace: hit = stored - 1, 1,664 tokens per
# restore). Only the draft KV of the last accepted positions is volatile; a full chunk ending at least _MARGIN tokens
# before the offloadable end is final (the GPU prefix cache keeps such blocks too), so it is stored.
_MARGIN = 8  # >= 2 x (3 MTP draft tokens + 1)
# Throughput arm (17 Oct, ARC_OFFLOAD_NO_EAGLE_POP=1): the lookup also pops one chunk off every eagle group's hit, so a
# host restore lost 1-2 chunks (cgtrace: 808 of 1,018 pure restores lost 1, 207 lost 2; ~45% of computed prompt
# tokens). Chunks stored under the margin rule are final, so no group is flagged eagle (no pop); the margin is kept
# for the groups holding MTP layers (_MTP_GROUPS). Draft KV only feeds proposals that the target verifies: exact.
# Without eagle groups the partial-tail store becomes eligible; it stays off unless ARC_OFFLOAD_PARTIAL_TAIL=1.
_NO_POP = _os.environ.get("ARC_OFFLOAD_NO_EAGLE_POP") == "1"
_MTP_GROUPS = set()
if _os.environ.get("ARC_OFFLOAD_EAGLE_TAIL") == "1" or _NO_POP:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as _sched

    def _storable_chunks(self, group_config, group_state, num_offloadable_tokens):
        per = group_config.tokens_per_chunk
        num_chunks = num_offloadable_tokens // per
        mtp = group_config.is_eagle_group or group_config.group_idx in _MTP_GROUPS
        if mtp and num_offloadable_tokens > self.req.num_prompt_tokens:
            num_chunks = max(0, num_offloadable_tokens - _MARGIN) // per
        return min(num_chunks, len(group_state.block_ids) // self.config.blocks_per_chunk)

    _sched.RequestOffloadState.storable_chunks = _storable_chunks

if _NO_POP:
    from vllm.config import SpeculativeConfig as _SpecConfig

    _from_spec = _sched.SchedulerOffloadConfig.from_spec.__func__

    def _from_spec_no_eagle(cls, spec, vllm_config, kv_cache_config):
        use_eagle = _SpecConfig.use_eagle
        _SpecConfig.use_eagle = lambda self: False  # no all-groups eagle fallback for unflagged groups
        try:
            config = _from_spec(cls, spec, vllm_config, kv_cache_config)
        finally:
            _SpecConfig.use_eagle = use_eagle
        if _os.environ.get("ARC_OFFLOAD_PARTIAL_TAIL") != "1" and config.supports_partial_tail:
            config = config._replace(supports_partial_tail=False)  # a NamedTuple
        print("ARC_OFFLOAD_NO_EAGLE_POP " + _json.dumps({"mtp_groups": sorted(_MTP_GROUPS),
              "eagle": [g.is_eagle_group for g in config.kv_group_configs],
              "partial_tail": config.supports_partial_tail}), flush=True)
        return config

    _sched.SchedulerOffloadConfig.from_spec = classmethod(_from_spec_no_eagle)


class ArcOffloadingConnector(_base.OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        self._keep = _cacheable(kv_cache_config)
        self._all = len(kv_cache_config.kv_cache_groups)
        groups = [kv_cache_config.kv_cache_groups[i] for i in self._keep]
        # Throughput arm (16 Oct): no group carries is_eagle_group here, so the offloading scheduler falls back to
        # treating EVERY group as an MTP draft group (SchedulerOffloadConfig.from_spec). For the Mamba groups that
        # doubles the trailing-window lookup (two consecutive stored states), which their sparse align states almost
        # never satisfy: a turn's computed blocks were never restorable (vo16dt trace). Flag only the groups that
        # hold the MTP layers (their draft KV is the volatile part); the Mamba states at committed block
        # boundaries are final and are what the GPU prefix cache reuses as well.
        if not any(g.is_eagle_group for g in groups):
            flagged = [replace(g, is_eagle_group=any("mtp" in name for name in g.layer_names)) for g in groups]
            if not any(g.is_eagle_group for g in flagged):  # names unknown: attention groups only
                flagged = [replace(g, is_eagle_group=not isinstance(g.kv_cache_spec, _MambaSpec)) for g in groups]
            groups = flagged
            if _NO_POP:
                _MTP_GROUPS.update(i for i, g in enumerate(groups) if g.is_eagle_group)
                groups = [replace(g, is_eagle_group=False) for g in groups]
            print("ARC_OFFLOAD_EAGLE " + _json.dumps([[i, g.is_eagle_group, type(g.kv_cache_spec).__name__,
                                                       len(g.layer_names)] for i, g in zip(self._keep, groups)]),
                  flush=True)
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
        if _TRACE:
            sched = self.connector_scheduler
            st = sched._req_status.get(request.request_id) if sched is not None else None
            rec = {"req": request.request_id, "tokens": request.num_tokens, "prompt": request.num_prompt_tokens,
                   "computed": request.num_computed_tokens}
            if st is not None:
                rec["stored_chunks"] = [g.next_stored_chunk_idx for g in st.group_states]
                rec["blocks"] = [len(g.block_ids) for g in st.group_states]
                rec["zero_blocks"] = [sum(1 for b in g.block_ids if b == 0) for g in st.group_states]
                rec["tokens_per_chunk"] = [c.tokens_per_chunk for c in sched.config.kv_group_configs]
                rec["jobs"] = len(st.transfer_jobs)
            print("ARC_OFFLOAD_FINISH " + _json.dumps(rec), flush=True)
        return super().request_finished_all_groups(request, self._ids(block_ids))

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        out = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if _TRACE:
            print("ARC_OFFLOAD_LOOKUP " + _json.dumps({"req": request.request_id, "tokens": request.num_tokens,
                  "local": num_computed_tokens, "ext": out[0], "async": out[1]}), flush=True)
        return out
