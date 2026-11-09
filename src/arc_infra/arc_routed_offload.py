"""R3 through the host KV tier (RL, 26 Oct): routed-expert rows travel with the KV chunks ``ArcOffloadingConnector``
parks in host RAM. Installed as ``vllm/arc_routed_offload.py`` beside ``vllm/arc_offload.py``; numpy only, so the CPU
tests drive it with a fake connector scheduler and slot buffer.

vLLM keeps routed experts per KV slot in the scheduler (``RoutedExpertsManager.routed_experts_by_slot``) and reads a
request's prompt routing at its prefill end from its blocks' slots. A chunk loaded from host RAM lands in fresh GPU
blocks whose slots hold some earlier request's rows, which is why vLLM refuses routed-experts return with any KV
connector. ``served_route_patch`` lifts that for this connector and calls into ``RoutedRows``:

- save: each routed-group chunk the connector stores in step S is copied (its slots' rows) into an LRU keyed by the
  chunk's offload key (the request's chained block hash, LoRA name in it, and the group) at step S's
  ``update_from_output``, right after ``store_batch``. The store reads KV that steps <= S wrote; under async
  scheduling step S's rows land after schedule S+1, so the copy waits for that landing and never goes by
  ``num_computed_tokens``. A store covers committed tokens only, so rows of rejected MTP drafts never reach it.
- lookup: a host hit with any chunk lacking rows is refused (0 external tokens: recompute); the rows found are held
  for the request until its allocation.
- restore: ``update_state_after_alloc`` writes them into the destination blocks' slots before the prefill, and every
  landing up to step S writes them again (an older in-flight step's ``store_batch`` may hit a block freed under it).
"""

import collections
import json
import os
import time

import numpy as np

STATUS_SECONDS = 300  # the ARC_ROUTED_OFFLOAD log line
STATUS_FILE_SECONDS = 60  # the status file (arena.r3_offload_check reads it)
ALERT = "ARC_ROUTED_OFFLOAD_ALERT"
STATUS_NAME = "r3-offload-{pid}.json"


def status_path(directory):
    return os.path.join(directory, STATUS_NAME.format(pid=os.getpid())) if directory else None


def cdiv(a, b):
    return -(-a // b)


def slots_of(block_ids, block_size):
    return (np.asarray(block_ids, dtype=np.int64)[:, None] * block_size + np.arange(block_size)).ravel()


class RoutedRows:
    """``sched``: the connector's ``OffloadingConnectorScheduler`` (``config``, ``_req_status``, ``_jobs``); ``keep``:
    the core KV group index of each connector group; ``budget_bytes`` and ``num_cpu_blocks`` cap the LRU (entries <=
    the host pool's block count, so a chunk still in host RAM keeps its rows); ``status_file``: see write_status."""

    def __init__(self, sched, keep, budget_bytes, num_cpu_blocks=None, log=print, status_file=None):
        self.sched, self.keep, self.log, self.status_path = sched, list(keep), log, status_file
        self.why, self.written_at = None, 0.0
        self.budget_bytes, self.num_cpu_blocks = int(budget_bytes), num_cpu_blocks
        self.mgr = self.g = None
        self.capacity = 0
        self.rows = collections.OrderedDict()  # offload key -> rows [tokens_per_chunk, layers, k]
        self.seq = 0  # build_connector_meta calls; scheduler_output._arc_routed_seq
        self.pending = []  # (seq, key, gpu block ids): chunks stored in step seq
        self.restores = []  # (seq, slots, rows): written at allocation in schedule seq
        self.held = {}  # request id -> {key: rows} from its lookup
        self.landing = None  # the scheduler output update_from_output is processing (attach)
        self.tried = False
        self.counts = collections.Counter()
        self.logged_at = time.time()

    def attach(self, mgr):
        """Once: bind, then flush after every ``mgr.store_batch`` made inside update_from_output (``landing``)."""
        if self.tried:
            return
        self.tried = True
        if not self.bind(mgr):
            return
        store = mgr.store_batch

        def store_batch(data, slot_mapping):
            store(data, slot_mapping)
            if self.landing is not None:
                self.flush(mgr, self.landing)

        mgr.store_batch = store_batch

    # -------------------------------------------------------------------- setup
    def unbound(self, why):
        """Left unbound: every host hit is refused (recomputed), so routing stays exact but the host tier is lost.
        Loud: an ALERT line in the server log and ``bound: false`` in the status file (an alert row arena side)."""
        self.why = why
        self.log(ALERT + " " + json.dumps({"event": "unbound", "why": why}), flush=True)
        self.write_status(why)
        return False

    def bind(self, mgr):
        """The scheduler's RoutedExpertsManager (at the first schedule). False: left unbound."""
        gid = int(mgr.attn_gid)
        if gid not in self.keep:
            return self.unbound(f"routed group {gid} is not a cacheable group {self.keep}")
        g = self.keep.index(gid)
        group = self.sched.config.kv_group_configs[g]
        if group.tokens_per_block != mgr.block_size:
            return self.unbound(f"group {gid} offloads {group.tokens_per_block}-token blocks, routed slots use "
                                f"{mgr.block_size}")
        chunk_bytes = group.tokens_per_chunk * mgr.routed_experts_by_slot[0].nbytes
        capacity = self.budget_bytes // max(chunk_bytes, 1)
        if self.num_cpu_blocks is not None:
            capacity = min(capacity, int(self.num_cpu_blocks))
        self.mgr, self.g, self.capacity = mgr, g, int(capacity)
        self.log("ARC_ROUTED_OFFLOAD " + json.dumps({
            "event": "bind", "group": gid, "connector_group": g, "tokens_per_chunk": group.tokens_per_chunk,
            "row_bytes": int(mgr.routed_experts_by_slot[0].nbytes), "chunk_bytes": int(chunk_bytes),
            "capacity": self.capacity, "num_cpu_blocks": self.num_cpu_blocks, "budget_bytes": self.budget_bytes}),
            flush=True)
        self.write_status()
        return True

    def _group(self):
        return self.sched.config.kv_group_configs[self.g]

    # -------------------------------------------------------------------- save
    def tick(self, scheduler_output):
        """build_connector_meta: number this schedule; its update_from_output passes the same object back."""
        self.seq += 1
        scheduler_output._arc_routed_seq = self.seq

    def on_store(self, store_jobs):
        """After build_connector_meta: every routed-group chunk in this schedule's store jobs waits for its rows."""
        if self.mgr is None or not store_jobs:
            return
        bpc = self.sched.config.blocks_per_chunk
        for jid, job in store_jobs.items():
            status = self.sched._jobs.get(jid)
            req = self.sched._req_status.get(job.req_id)
            if status is None or req is None:
                continue
            state = req.group_states[self.g]
            keys = state.offload_keys
            wanted = set(status.keys)
            for c in range(len(keys) - 1, -1, -1):  # stores take the newest chunks
                if not wanted:
                    break
                key = keys[c]
                if key not in wanted:
                    continue
                wanted.discard(key)
                blocks = state.block_ids[c * bpc:(c + 1) * bpc]
                if len(blocks) < bpc or 0 in blocks:
                    self.counts["save_skipped"] += 1
                    continue
                self.pending.append((self.seq, key, list(blocks)))

    def put(self, key, rows):
        self.rows[key] = rows
        self.rows.move_to_end(key)
        while len(self.rows) > self.capacity:
            self.rows.popitem(last=False)
            self.counts["evicted"] += 1

    def flush(self, mgr, scheduler_output):
        """update_from_output, right after store_batch: this step's rows are in the slot buffer."""
        seq = getattr(scheduler_output, "_arc_routed_seq", None)
        if self.mgr is None or seq is None:
            return
        buf = mgr.routed_experts_by_slot
        if self.restores:
            for _, slots, rows in self.restores:
                buf[slots] = rows
            self.restores = [r for r in self.restores if r[0] > seq]
        if self.pending:
            later = []
            for item in self.pending:
                if item[0] > seq:
                    later.append(item)
                    continue
                self.put(item[1], buf[slots_of(item[2], mgr.block_size)])  # fancy index: a copy
                self.counts["saved"] += 1
            self.pending = later
        self.status()

    # -------------------------------------------------------------------- load
    def refuse(self, req, num_computed_tokens, why):
        req.update_num_hit_chunks(num_computed_tokens)
        req.partial_tail_boundary = None
        self.counts["refused_" + why] += 1
        self.status()  # an unbound server never flushes: its refusals reach the status file here
        return 0

    def lookup(self, request_id, req, num_computed_tokens, num_hit_tokens):
        """get_num_new_matched_tokens: keep a host hit only if every loaded chunk has its rows."""
        self.held.pop(request_id, None)
        if not num_hit_tokens:
            return num_hit_tokens  # 0, or None (lookup pending)
        if self.mgr is None:
            return self.refuse(req, num_computed_tokens, "unbound")
        if req.partial_tail_boundary is not None:
            return self.refuse(req, num_computed_tokens, "partial_tail")
        group = self._group()
        first = (num_computed_tokens // group.tokens_per_block) // self.sched.config.blocks_per_chunk
        last = cdiv(num_computed_tokens + num_hit_tokens, group.tokens_per_chunk)
        keys = req.group_states[self.g].offload_keys[first:last]
        if len(keys) != last - first:
            return self.refuse(req, num_computed_tokens, "keys")
        got = {}
        for key in keys:
            rows = self.rows.get(key)
            if rows is None:
                return self.refuse(req, num_computed_tokens, "missing")
            self.rows.move_to_end(key)
            got[key] = rows
        self.held[request_id] = got
        self.counts["hits"] += 1
        return num_hit_tokens

    def after_alloc(self, request_id, req, blocks, num_external_tokens):
        """update_state_after_alloc (``blocks``: the core groups' KVCacheBlock lists): write the held rows into the
        routed group's destination blocks, mirroring the connector's own load range."""
        got = self.held.pop(request_id, None)
        if not num_external_tokens or self.mgr is None:
            return
        if got is None or req is None:
            self.counts["restore_unheld"] += 1
            return
        group, bpc = self._group(), self.sched.config.blocks_per_chunk
        tpb = group.tokens_per_block
        cached = req.num_locally_computed_tokens + num_external_tokens
        gpu = list(blocks[self.mgr.attn_gid])
        n_gpu = cdiv(cached, tpb)
        n_local = n_gpu
        for i, block in enumerate(gpu[:n_gpu]):
            if not block.is_null and block.block_hash is None:
                n_local = i
                break
        if n_local == n_gpu:
            return
        first = n_local // bpc
        keys = req.group_states[self.g].offload_keys[first:cdiv(cached, group.tokens_per_chunk)]
        dst = gpu[n_local:n_gpu]
        parts = [got[key] if key in got else self.rows.get(key) for key in keys]
        if any(p is None for p in parts) or any(b.is_null for b in dst):
            self.counts["restore_unheld"] += 1
            return
        rows = np.concatenate(parts)[(n_local - first * bpc) * tpb:(n_gpu - first * bpc) * tpb]
        slots = slots_of([b.block_id for b in dst], tpb)
        if rows.shape[0] != slots.size:
            self.counts["restore_shape"] += 1
            return
        self.mgr.routed_experts_by_slot[slots] = rows
        self.restores.append((self.seq + 1, slots, rows))  # this schedule's tick comes next
        self.counts["restored"] += 1
        self.counts["restored_tokens"] += int(slots.size)

    # -------------------------------------------------------------------- lifecycle
    def finished(self, request_id):
        self.held.pop(request_id, None)

    def reset(self):
        self.rows.clear()
        self.held.clear()
        self.pending = []
        self.counts["resets"] += 1

    def write_status(self, why=None):
        """``status_path`` (the serving working dir, ``r3-offload-<pid>.json``): what the arena side reads
        (arena.r3_offload_check: the bound metric, refusals by reason, an alert row when unbound)."""
        if not self.status_path:
            return
        data = {"t": time.time(), "pid": os.getpid(), "bound": self.mgr is not None, "tried": self.tried,
                "why": why if why is not None else self.why, "entries": len(self.rows), "capacity": self.capacity,
                **self.counts}
        try:
            tmp = f"{self.status_path}.tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.status_path)
        except OSError:
            pass

    def status(self, force=False):
        now = time.time()
        if force or now - self.written_at >= STATUS_FILE_SECONDS:
            self.written_at = now
            self.write_status()
        if not force and now - self.logged_at < STATUS_SECONDS:
            return
        self.logged_at = now
        self.log("ARC_ROUTED_OFFLOAD " + json.dumps({"event": "status", "entries": len(self.rows),
                                                     "capacity": self.capacity, "pending": len(self.pending),
                                                     "restores": len(self.restores), **self.counts}), flush=True)
