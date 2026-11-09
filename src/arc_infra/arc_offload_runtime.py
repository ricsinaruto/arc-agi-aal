"""CPU KV offloading for Qwen3.8-Flash-Next's hybrid cache (installed as ``vllm/arc_offload.py``; ARC-AGI-2 arm,
15 October).

The pinned ``OffloadingConnector`` refuses the QSA indexer's ring (``CircularBufferSpec``): one block a request
holding the raw keys of the group still being compressed, never prefix-cached. The GPU prefix cache leaves that
group out of every hit (``kv_cache_coordinator``: non-``prefix_cacheable`` groups are skipped) and a hit claims a
fresh ring block; a hit restored from CPU is the same case. So the connector here sees the cache without its
non-cacheable groups: its config, the per-group block ids the scheduler hands it (block ids, partial-tail handoffs), and the worker's group
list are filtered to the cacheable groups (full attention + indexer keys, the Mamba states). The ring's block is claimed by
the core as on any external hit (``CircularBufferManager.allocate_external_computed_blocks``).

With routed-experts return on (R3, ``served_route_patch``), the stored chunks carry their routed rows
(``vllm/arc_routed_offload.py``): a host hit without them is refused, a restored one gets them in its new slots.
"""

import inspect as _inspect
import json as _json
from dataclasses import replace
from types import SimpleNamespace

from vllm.arc_routed_offload import RoutedRows, status_path as routed_status_path
from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector as _base
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import worker as _worker
from vllm.v1.core import kv_cache_manager as _kvm
from vllm.v1.core.sched import scheduler as _sched
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import MambaSpec as _MambaSpec


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


# R3 (26 Oct): the scheduler hook. The connector is built inside Scheduler.__init__ before its RoutedExpertsManager,
# so the first schedule() binds it; update_from_output marks the step whose rows store_batch is landing.
_UPDATE_SOURCE = "self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)"


def _routed_of(scheduler):
    routed = getattr(getattr(scheduler, "connector", None), "_routed", None)
    mgr = getattr(scheduler, "routed_experts_mgr", None)
    if routed is not None and not routed.tried and mgr is not None:
        try:
            pinned = _inspect.getsource(_update_from_output.base).count(_UPDATE_SOURCE) == 1
        except (OSError, TypeError):
            pinned = False
        if pinned:
            routed.attach(mgr)
        else:  # host hits refused (recomputed): routing stays exact
            routed.tried = True
            routed.unbound("Scheduler.update_from_output does not call store_batch once")
    return routed


def _schedule(self, *args, **kwargs):
    _routed_of(self)
    return _schedule.base(self, *args, **kwargs)


def _update_from_output(self, scheduler_output, model_runner_output):
    routed = _routed_of(self)
    if routed is None:
        return _update_from_output.base(self, scheduler_output, model_runner_output)
    routed.landing = scheduler_output
    try:
        return _update_from_output.base(self, scheduler_output, model_runner_output)
    finally:
        routed.landing = None


if not getattr(_sched.Scheduler, "_arc_routed_hook", False):
    _schedule.base, _update_from_output.base = _sched.Scheduler.schedule, _sched.Scheduler.update_from_output
    _sched.Scheduler.schedule, _sched.Scheduler.update_from_output = _schedule, _update_from_output
    _sched.Scheduler._arc_routed_hook = True


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
            print("ARC_OFFLOAD_EAGLE " + _json.dumps([[i, g.is_eagle_group, type(g.kv_cache_spec).__name__,
                                                       len(g.layer_names)] for i, g in zip(self._keep, groups)]),
                  flush=True)
        filtered = replace(kv_cache_config, kv_cache_groups=groups)
        super().__init__(vllm_config, role, filtered)
        self._routed = None
        if self.connector_scheduler is not None and getattr(vllm_config.model_config, "enable_return_routed_experts",
                                                            False):
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
            budget = int(extra.get("routed_rows_bytes", int(extra.get("cpu_bytes_to_use", 0)) // 4))
            self._routed = RoutedRows(self.connector_scheduler, self._keep, budget,
                                      getattr(self.connector_scheduler.manager, "_num_blocks", None),
                                      status_file=routed_status_path(extra.get("arc_routed_status_dir")))

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        hit, load_async = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if self._routed is None or not hit:
            return hit, load_async
        hit = self._routed.lookup(request.request_id, self.connector_scheduler._req_status[request.request_id],
                                  num_computed_tokens, hit)
        return hit, bool(hit)

    def reset_cache(self):
        out = super().reset_cache()
        if self._routed is not None:
            self._routed.reset()
        return out

    def _ids(self, groups):
        if groups is None:
            return None
        assert len(groups) == self._all, (len(groups), self._all)
        return tuple(groups[i] for i in self._keep)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        out = super().update_state_after_alloc(request, KVCacheBlocks(self._ids(blocks.blocks)),
                                               num_external_tokens)
        if self._routed is not None:
            self._routed.after_alloc(request.request_id,
                                     self.connector_scheduler._req_status.get(request.request_id), blocks.blocks,
                                     num_external_tokens)
        return out

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
        meta = super().build_connector_meta(out)
        if self._routed is not None:
            self._routed.tick(scheduler_output)
            self._routed.on_store(meta.store_jobs)
        return meta

    def request_finished_all_groups(self, request, block_ids):
        if self._routed is not None:
            self._routed.finished(request.request_id)
        return super().request_finished_all_groups(request, self._ids(block_ids))
