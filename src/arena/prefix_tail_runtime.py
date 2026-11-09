"""Runs inside pinned vLLM as ``vllm/v1/core/arc_tail_cache.py``: request-end tail entries (prefix cache v3).

``arena.prefix_cache_patch`` copies this file into the extracted runtime and hooks it into ``KVCacheManager``
(``request_end_tail``). Pinned vLLM caches whole blocks only, so a game's next request resumes at the last
1,600-token (int8 KV; 3,136 at int4) boundary of the previous prompt + reply and recomputes the rest (the "block
tail", ~800 tokens per continuation on hero-v2). The finished request's blocks already hold the exact prefix at its
end: the attention block that holds the partial tail, each Mamba group's running state (the state after exactly
``num_computed_tokens`` tokens) and the QSA compressor ring (the raw keys of the open compression group). They are
freed unhashed today, so they are reused first.

Registration (``KVCacheManager.free`` of a *finished* request, MTP-0 layouts only): with ``L`` the request's
``num_computed_tokens`` (every step scheduled for it, including an in-flight one, has run by the time any later step
reads the blocks: the worker executes steps in order), ``B`` the block and ``k = L // B`` (``L % B != 0``):

- the entry's key is the prefix hash at ``k * B`` (the request's own chained block hash) plus the exact tokens
  ``[k * B, L)`` and their extra keys (images, LoRA);
- its blocks are attention block ``k`` of every attention group, the running state block (index ``k``) of every
  Mamba group and the request's ring block. Each gets a synthetic prefix-cache key, so the pool treats it as a
  cached block: freed to the LRU tail, evicted like any other cached block (eviction drops the key, which
  invalidates the entry), never pinned.

Lookup (``get_computed_blocks``, after the pinned hit ``H``): the longest entry with ``H < L <= num_tokens - 1``
whose prefix hash and tokens match the new request and whose blocks all still carry their keys, with the attention
groups' ``k`` full blocks still cached. The hit is then the pinned *partial* hit shape (attention: ``k`` full blocks +
the entry's block; Mamba: nulls + the entry's state), so the pinned partial-hit copy-on-write gives the request
private copies of the attention block and the states before anything writes them; the ring gets the same treatment
here (``_wrap_ring``: the source ring is touched with the other hit blocks, one extra block is reserved when it is
evictable, and the copy is queued on the ring manager with both endpoints retained until the copy has run).

Capacity: entry blocks are free blocks (``ref_cnt`` 0, in the free queue, counted by ``get_num_free_blocks``), so
they never reduce what running requests can allocate and cannot cause a preemption. A hit holds its sources exactly
like the pinned partial hit (one step, until the copy fence).
"""
import hashlib
import json
from array import array
from collections import Counter

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    make_block_hash_with_group_id,
)

# Where a new entry's blocks enter the free queue: "front" = reused first but hittable until then (like prefix cache
# v2's demoted states), so an entry that is not consumed soon never displaces older cached prefixes; "back" = the LRU
# tail, after the request's other blocks (the entry is only usable while the request's attention blocks are cached,
# so it should never be evicted before them); "swap" = "back", and the request's newest boundary state that prefix
# cache v2 keeps (``mamba_keep_state``: the state at k * B, which the entry's state at L supersedes for the game's next
# request) goes to the front instead, so the entry adds one attention block and the ring to the LRU tail, not 2 + the
# Mamba groups (the recorded-run replay: prefix-cache-v3.md section 3).
PLACEMENT = "swap"
# Entries kept per prefix hash, newest first. Up to 110 games share the first blocks (the system prompt), so their
# first requests share a parent; the v3 smoke's cap of 4 dropped a primer's entry after 4 same-parent probes.
MAX_PER_PARENT = 128
SWEEP_EVERY = 512  # registrations between sweeps of invalid (evicted) entries
LOG_EVERY = 200  # registrations between ARC_TAIL_STATS lines in the server log
TAG = b"arc-tail"


class Entry:
    __slots__ = ("length", "tokens", "extra", "keys")

    def __init__(self, length, tokens, extra, keys):
        self.length, self.tokens, self.extra, self.keys = length, tokens, extra, keys


class TailIndex:
    """Request-end tail entries for one ``KVCacheManager``."""

    @classmethod
    def create(cls, manager):
        if not manager.enable_caching:
            return None
        return cls(manager)

    def __init__(self, manager):
        self.manager = manager
        self.pool = manager.block_pool
        self.hash_block = self.pool.hash_block_size
        self.attention, self.mamba, self.ring = [], [], []
        sizes = set()
        unsupported = []
        managers = manager.coordinator.single_type_managers
        for gid, (group, mgr) in enumerate(zip(manager.kv_cache_config.kv_cache_groups, managers)):
            name = type(mgr).__name__
            if name == "CircularBufferManager":
                self.ring.append(gid)
            elif name == "MambaManager" and mgr.mamba_cache_mode == "align" and mgr.num_speculative_blocks == 0:
                self.mamba.append(gid)
                sizes.add(mgr.block_size)
            elif name == "FullAttentionManager":
                self.attention.append(gid)
                sizes.add(mgr.block_size)
            else:
                unsupported.append(name)
        reasons = []
        if unsupported:
            reasons.append(f"unsupported groups {sorted(set(unsupported))}")
        if len(sizes) != 1 or not self.attention or not self.mamba:
            reasons.append(f"attention/Mamba block sizes {sorted(sizes)}")
        if manager.use_eagle:
            reasons.append("speculative decoding")
        self.block = next(iter(sizes)) if len(sizes) == 1 else None
        if self.block is not None and self.block % self.hash_block:
            reasons.append("hash unit does not divide the block")
        self.disabled = "; ".join(reasons) or None
        self.entries = {}  # prefix hash at k*B -> [Entry], newest first
        self.stats = Counter()
        self.freed = []  # the newest entry's blocks, placed by after_free
        self.superseded = []  # its request's kept boundary states (PLACEMENT "swap")
        self.pending = {}  # ring group id -> {request id: the entry's ring block, between lookup and allocation}
        for gid in self.ring:
            self._wrap_ring(managers[gid])
        print("ARC_TAIL_INDEX " + json.dumps({"attention": self.attention, "mamba": self.mamba, "ring": self.ring,
                                               "block": self.block, "hash_block": self.hash_block,
                                               "disabled": self.disabled}), flush=True)

    # --------------------------------------------------------------------------------------- registration
    def _synthetic(self, parent, length, tokens, extra):
        digest = hashlib.sha256(parent)
        digest.update(length.to_bytes(8, "big"))
        digest.update(array("q", tokens).tobytes())
        digest.update(repr(extra).encode())
        return TAG + digest.digest()

    def register(self, request):
        """Called by ``KVCacheManager.free`` before the request's blocks return to the pool."""
        if self.disabled is not None or not request.is_finished():
            return
        B = self.block
        L = request.num_computed_tokens
        if L % B == 0 or L < B or L > request.num_tokens:
            self.stats["skip_aligned_or_short"] += 1
            return
        k = L // B
        start = k * B
        parent_index = start // self.hash_block - 1
        if parent_index >= len(request.block_hashes):
            self.stats["skip_no_hash"] += 1
            return
        managers = self.manager.coordinator.single_type_managers
        rid = request.request_id
        blocks = {}
        for gid in self.attention:
            table = managers[gid].req_to_blocks.get(rid) or []
            if len(table) <= k or table[k].is_null or table[k - 1].block_hash is None:
                self.stats["skip_attention_layout"] += 1
                return
            blocks[gid] = table[k]
        for gid in self.mamba:
            table = managers[gid].req_to_blocks.get(rid) or []
            # The running state column is cdiv(L, B) - 1 = k and the last one (no speculative blocks).
            if len(table) != cdiv(L, B) or table[k].is_null or table[k].block_hash is not None:
                self.stats["skip_mamba_layout"] += 1
                return
            blocks[gid] = table[k]
        for gid in self.ring:
            table = managers[gid].req_to_blocks.get(rid) or []
            if len(table) != 1 or table[0].is_null or table[0].block_hash is not None:
                self.stats["skip_ring_layout"] += 1
                return
            blocks[gid] = table[0]
        parent = request.block_hashes[parent_index]
        tokens = list(request.all_token_ids[start:L])
        extra, _ = generate_block_hash_extra_keys(request, start, L, 0)
        synthetic = self._synthetic(parent, L, tokens, extra)
        keys = {}
        for gid, block in blocks.items():
            key = make_block_hash_with_group_id(synthetic, gid)
            self.pool._insert_block_hash(key, block, num_tokens=L)
            keys[gid] = (key, block)
        self.freed = [block for _, block in keys.values()]
        self.superseded = []
        if PLACEMENT == "swap":
            for gid in self.mamba:
                kept = getattr(managers[gid], "_arc_kept_state", {}).get(rid)
                if kept is not None and not kept.is_null:
                    self.superseded.append(kept)
        bucket = self.entries.setdefault(parent, [])
        bucket.insert(0, Entry(L, tokens, extra, keys))
        del bucket[MAX_PER_PARENT:]
        self.stats["registered"] += 1
        self.stats["registered_tail_tokens"] += L - start
        if self.stats["registered"] % SWEEP_EVERY == 0:
            self.sweep()
        if self.stats["registered"] % LOG_EVERY == 0:
            self.log()

    def after_free(self):
        """Called by ``KVCacheManager.free`` once the request's blocks are back in the free queue."""
        blocks, self.freed = self.freed, []
        superseded, self.superseded = self.superseded, []
        queue = self.pool.free_block_queue
        front = blocks if PLACEMENT == "front" else superseded if PLACEMENT == "swap" else []
        back = [] if PLACEMENT == "front" else blocks
        # Only blocks now in the free queue move (a block another request still references stays where it is).
        front = [block for block in front if block.ref_cnt == 0 and not block.is_null]
        back = [block for block in back if block.ref_cnt == 0 and not block.is_null]
        for block in front + back:
            queue.remove(block)
        if front:
            queue.prepend_n(front)
            self.stats["demoted_blocks"] += len(front)
        if back:
            queue.append_n(back)

    def valid(self, entry):
        mapping = self.pool.cached_block_hash_to_block
        return all(mapping.contain(key, block.block_id) for key, block in entry.keys.values())

    def sweep(self):
        for parent in list(self.entries):
            kept = [e for e in self.entries[parent] if self.valid(e)]
            self.stats["swept"] += len(self.entries[parent]) - len(kept)
            if kept:
                self.entries[parent] = kept
            else:
                del self.entries[parent]

    def log(self):
        print("ARC_TAIL_STATS " + json.dumps({**self.stats, "entries": sum(map(len, self.entries.values()))},
                                             sort_keys=True), flush=True)

    # --------------------------------------------------------------------------------------- lookup
    def _full_blocks(self, request, k, computed_blocks, hit):
        """Every attention group's ``k`` full blocks by prefix hash (the pinned hit's blocks where they reach)."""
        B, scale = self.block, self.block // self.hash_block
        reuse = hit // B
        full = {}
        for gid in self.attention:
            blocks = list(computed_blocks[gid][:min(k, reuse)])
            for i in range(len(blocks), k):
                cached = self.pool.get_cached_block(request.block_hashes[(i + 1) * scale - 1], [gid])
                if not cached:
                    return None
                blocks.append(cached[0])
            full[gid] = blocks
        return full

    def lookup(self, request, computed_blocks, hit, max_length):
        for gid in self.ring:
            self.pending[gid].pop(request.request_id, None)
        if self.disabled is not None or not self.entries:
            return computed_blocks, hit
        B = self.block
        tokens = request.all_token_ids
        for k in range(max_length // B, max(1, hit // B) - 1, -1):
            index = k * B // self.hash_block - 1
            if index >= len(request.block_hashes):
                continue
            bucket = self.entries.get(request.block_hashes[index])
            if not bucket:
                continue
            start = k * B
            for entry in sorted(bucket, key=lambda e: -e.length):
                L = entry.length
                if L <= hit or L > max_length:
                    continue
                if list(tokens[start:L]) != entry.tokens:
                    continue
                if generate_block_hash_extra_keys(request, start, L, 0)[0] != entry.extra:
                    continue
                if not self.valid(entry):
                    self.stats["lookup_invalid"] += 1
                    continue
                full = self._full_blocks(request, k, computed_blocks, hit)
                if full is None:
                    self.stats["lookup_missing_full_blocks"] += 1
                    continue
                result = [list(blocks) for blocks in computed_blocks]
                for gid in self.attention:
                    result[gid] = full[gid] + [entry.keys[gid][1]]
                for gid in self.mamba:
                    result[gid] = [self.pool.null_block] * k + [entry.keys[gid][1]]
                for gid in self.ring:
                    result[gid] = []
                    self.pending[gid][request.request_id] = entry.keys[gid][1]
                self.stats["hits"] += 1
                self.stats["hit_tokens_above_pinned"] += L - hit
                return tuple(result), L
        return computed_blocks, hit

    # --------------------------------------------------------------------------------------- ring copy
    def _wrap_ring(self, mgr):
        """Instance-level wrappers on the ring manager: touch, reserve and copy the entry's compressor ring."""
        gid = mgr.kv_cache_group_id
        pending, touched = {}, {}
        self.pending[gid] = pending
        pool = self.pool
        num_blocks, add_local = mgr.get_num_blocks_to_allocate, mgr.add_local_computed_blocks
        allocate, pop = mgr.allocate_new_blocks, mgr.pop_blocks_for_free

        def request_of(args, kwargs):
            return kwargs["request_id"] if "request_id" in kwargs else args[0]

        def get_num_blocks_to_allocate(*args, **kwargs):
            count = num_blocks(*args, **kwargs)
            source = pending.get(request_of(args, kwargs))
            # Touching an evictable source takes it off the free queue, like the other groups' hit blocks.
            return count + int(source is not None and source.ref_cnt == 0)

        def add_local_computed_blocks(*args, **kwargs):
            add_local(*args, **kwargs)
            rid = request_of(args, kwargs)
            source = pending.pop(rid, None)
            if source is not None:
                pool.touch([source])
                touched[rid] = source

        def allocate_new_blocks(*args, **kwargs):
            new = allocate(*args, **kwargs)
            source = touched.pop(request_of(args, kwargs), None)
            if source is not None:
                if new:
                    # Both endpoints stay retained until the copy has run (take_kv_cache_block_copies).
                    mgr._pending_cow_copies.append((source, new[0]))
                    new[0].ref_cnt += 1
                    self.stats["ring_copies"] += 1
                else:
                    pool.free_blocks([source])
            return new

        def pop_blocks_for_free(request_id):
            pending.pop(request_id, None)
            source = touched.pop(request_id, None)
            if source is not None:
                pool.free_blocks([source])
            return pop(request_id)

        mgr.get_num_blocks_to_allocate = get_num_blocks_to_allocate
        mgr.add_local_computed_blocks = add_local_computed_blocks
        mgr.allocate_new_blocks = allocate_new_blocks
        mgr.pop_blocks_for_free = pop_blocks_for_free
