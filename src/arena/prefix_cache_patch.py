"""Opt-in hybrid prefix-cache fixes for the pinned vLLM runtime (29 September; prefix-cache-v2.md).

The patches only modify a disposable extracted runtime and are hash-guarded like ``cache_reclamation``.

``mamba_align_fix`` (``vllm/v1/core/sched/scheduler.py``, ``Scheduler._mamba_block_aligned_split``). In
Mamba "align" mode a recurrent state is only reusable at a Mamba block boundary, so prefill chunks must end on
one. The split reads ``self.cache_config.block_size``, but ``EngineCore._initialize_kv_caches`` sets that to
the *smallest* group block (``min(g.kv_cache_spec.block_size)``): Flash-Next's QSA raw-key ring
(``CircularBufferSpec``) has 4-token blocks (8 on MTP-3), so the served value is 4 (``cache_config_info
block_size="4"`` in every c110 run's metrics) while the Mamba block is 1,600 (hero) or 800 (BF16 KV). The split
therefore ends chunks on 4-token boundaries:

- the state at the prompt's last Mamba boundary is never written, so the next request of a game can only
  resume from a decode-crossing state or an older one (the "hybrid-state lag");
- a prefill split into several chunks leaves a chunk-end state in the block that the scheduler then caches
  under the *next* boundary's hash, so a later hit there resumes from a state up to one block short of its
  prefix (tokens between the chunk end and the boundary are skipped by the recurrent layers).

The fix aligns to the Mamba spec's block size (falling back to the scheduler block size, the LCM of the group
block sizes, which equals it on these layouts).

``mamba_keep_state`` (``vllm/v1/core/single_type_kv_cache_manager.py``, ``MambaManager``). In align mode a
request frees each boundary state one step after the next state block exists, so the state the next request of
the game would resume from enters the LRU free queue at the prefill end or the last decode crossing, up to a
whole request earlier than the request's attention blocks, and is evicted first. The patch keeps the newest
cached boundary state per request (MTP-0 only: no speculative state blocks) until a newer one is cached or the
request is freed, and frees it last, so it ages with the attention blocks. A request holds at most one extra
state block per Mamba group, inside the two per group that the admission cost table already charges.

``mamba_demote_superseded`` (needs ``mamba_keep_state``; also ``vllm/v1/core/block_pool.py``). Every state a
request writes but the newest (intermediate prefill chunk ends, earlier decode crossings, the state it resumed
from) is useless to the game's next request, yet pinned vLLM frees it to the MRU end of the free queue, where it
displaces other games' prefixes; one state costs 4 pages, as much as 6,400 tokens of attention KV. With this
option such superseded states go to the front of the free queue (reused first, still hittable until then),
except states some request has resumed from and states within the first ``DEMOTE_MIN_TOKENS`` tokens (the shared
system prompt's state, which every first or evicted request resumes from).

``request_end_tail`` (prefix cache v3, 29 September; prefix-cache-v3.md; optional key, off unless given). A finished
request's exact end (the partial attention block, each Mamba group's running state, the QSA compressor ring) is
registered as a prefix-cache entry, so the game's next request resumes at the previous request's last computed token
instead of its last whole block (``arena.prefix_tail_runtime``, copied into the runtime as
``vllm/v1/core/arc_tail_cache.py`` and hooked into ``KVCacheManager`` at construction, lookup and free). The
entries' blocks are ordinary evictable cached blocks, so they never reduce the pool that running requests allocate
from. MTP-0 layouts only (the index disables itself otherwise and says so in the server log).
"""
import hashlib
import json
from pathlib import Path

SCHEDULER_RELATIVE = 'vllm/v1/core/sched/scheduler.py'
SCHEDULER_SHA256 = 'c710f49e41e974e5b7b8f1cd2f5fb9722523f4da0165252e16351199ebd03124'
ALIGN_ANCHOR = '        block_size = self.cache_config.block_size\n'
ALIGN_REPLACEMENT = '''        # ARC mamba_align_fix: the Mamba state block, not cache_config.block_size (EngineCore sets that to the
        # smallest group block, the QSA raw-key ring's 4 tokens).
        from vllm.v1.kv_cache_interface import MambaSpec as _ArcMambaSpec

        block_size = next((g.kv_cache_spec.block_size for g in self.kv_cache_config.kv_cache_groups
                           if isinstance(g.kv_cache_spec, _ArcMambaSpec)), self.block_size)
'''

MANAGER_RELATIVE = 'vllm/v1/core/single_type_kv_cache_manager.py'
MANAGER_SHA256 = '9ec6cbc0c78f30986f294439f02f57a9586ffbf2939d8f5c4fb968a642625ca2'
# cache_reclamation (applied first when reclaim_skipped_blocks is on, as on the hero) patches the same file.
MANAGER_RECLAIMED_SHA256 = '2633adf8a2404e21ed727def9b5b2aea78e6b046af87e5f6cbe2c38361cf5339'
CLASS_ANCHOR = ('class MambaManager(SingleTypeKVCacheManager):\n'
                '    supports_fine_grained_hash_lookup: ClassVar[bool] = True\n')
CLASS_INSERT = ('    # ARC prefix cache v2: demote superseded recurrent states in the free queue, except the shared prefix.\n'
                '    _arc_demote: ClassVar[bool] = {demote}\n'
                '    _arc_demote_min_tokens: ClassVar[int] = {min_tokens}\n')
# States within the first 8,192 tokens are never demoted: the arena's shared system prompt + tools end at ~4.1K
# tokens (the 3,200-token state on 1,600-token blocks, 4,000 on 800), and every first or evicted request resumes there.
DEMOTE_MIN_TOKENS = 8192
KEEP_INIT_ANCHOR = '            self._producer_partial_tail_reqs: dict[str, int] = {}\n'
KEEP_INIT_INSERT = ('            # ARC prefix cache v2: request id -> its newest cached boundary-state block;\n'
                    '            # block id -> (block hash, prefix hits on that hash).\n'
                    '            self._arc_kept_state: dict[str, KVCacheBlock] = {}\n'
                    '            self._arc_hits: dict[int, tuple] = {}\n')
KEEP_REMOVE_ANCHOR = '''        assert isinstance(self.kv_cache_spec, MambaSpec)

        super().remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )
        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` refers to the block index allocated two steps ago.
'''
KEEP_REMOVE_REPLACEMENT = '''        assert isinstance(self.kv_cache_spec, MambaSpec)

        arc_align = self.mamba_cache_mode == "align" and self.num_speculative_blocks == 0
        arc_demote = arc_align and self._arc_demote
        if arc_demote:
            self.block_pool._arc_evict_first = self._arc_is_superseded_state
        try:
            if arc_align:
                self._arc_keep_latest_state(request_id)
            self._arc_pinned_remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )
        finally:
            if arc_demote:
                self.block_pool._arc_evict_first = None

    def _arc_pinned_remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        super().remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )
        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` refers to the block index allocated two steps ago.
'''
KEEP_HELPER_ANCHOR = '''    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
'''
KEEP_HELPER = '''    def _arc_keep_latest_state(self, request_id: str) -> None:
        """ARC mamba_keep_state: hold the newest cached boundary state below the running block.

        The block leaves the request's table (a null takes its place, as when it is freed) but keeps the
        request's reference until a newer cached state replaces it or the request is freed; then it is freed
        last, so it enters the LRU free queue with the request's attention blocks.
        """
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        # The newest boundary states sit right below the running block (prefill end, decode crossings); a
        # bounded scan keeps this O(1) per request and step.
        for i in range(len(blocks) - 2, max(-1, len(blocks) - 10), -1):
            block = blocks[i]
            if block.is_null or block.block_hash is None:
                continue
            blocks[i] = self._null_block
            previous = self._arc_kept_state.get(request_id)
            self._arc_kept_state[request_id] = block
            if previous is not None:
                self.block_pool.free_blocks([previous])
            return

    def _arc_is_superseded_state(self, block: KVCacheBlock) -> bool:
        """ARC mamba_demote_superseded: a state freed while its request runs is superseded garbage unless some
        request has resumed from it (a game's previous state) or it lies within the first
        ``_arc_demote_min_tokens`` tokens, where the prefix every game shares (the system prompt and tools) ends:
        that state must survive even when the request that wrote it moves on before anyone resumes from it."""
        tokens = block.block_hash_num_tokens
        if tokens is None or tokens <= self._arc_demote_min_tokens:
            return False
        seen = self._arc_hits.get(block.block_id)
        return seen is None or seen[0] != block.block_hash

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        if self._arc_demote and new_computed_blocks:
            hit = new_computed_blocks[-1]
            if not hit.is_null and hit.block_hash is not None:
                seen = self._arc_hits.get(hit.block_id)
                count = seen[1] if seen is not None and seen[0] == hit.block_hash else 0
                self._arc_hits[hit.block_id] = (hit.block_hash, count + 1)
        super().add_local_computed_blocks(
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        )

'''
KEEP_POP_ANCHOR = '''            self._pending_partial_tail_offloads = [
                entry
                for entry in self._pending_partial_tail_offloads
                if entry[0] != request_id
            ]
        return super().pop_blocks_for_free(request_id)
'''
KEEP_POP_REPLACEMENT = '''            self._pending_partial_tail_offloads = [
                entry
                for entry in self._pending_partial_tail_offloads
                if entry[0] != request_id
            ]
            kept = self._arc_kept_state.pop(request_id, None)
            if kept is not None:
                # First in allocation order = freed last by the caller's reversed free: the most recently used.
                return [kept] + super().pop_blocks_for_free(request_id)
        return super().pop_blocks_for_free(request_id)
'''

POOL_RELATIVE = 'vllm/v1/core/block_pool.py'
POOL_SHA256 = 'ddee56dccb2208411b3a035918e917ce8f56a9858471e9ca12b420d5d79bc69c'
DEMOTE_ANCHOR = '''                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
'''
DEMOTE_REPLACEMENT = '''                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                elif (
                    getattr(self, "_arc_evict_first", None) is not None
                    and self._arc_evict_first(block)
                ):
                    # ARC mamba_demote_superseded: a superseded recurrent state, reused first but still
                    # hittable until then.
                    blocks_to_evict_first.append(block)
                else:
'''
KV_MANAGER_RELATIVE = 'vllm/v1/core/kv_cache_manager.py'
KV_MANAGER_SHA256 = '2d20c3d98845cfd8d88a2f66b8fc6402ea1fcfbee16879d2364a4a9e45b8fa47'
# arena.cache_patch (the admission-failure diagnostics, applied first on every arena server) patches the same file.
KV_MANAGER_DIAGNOSTICS_SHA256 = '18cb943ada623f315f128478b8b8ea2ac3135615affd1007cadd0ca0dfd4f382'
TAIL_HELPER_RELATIVE = 'vllm/v1/core/arc_tail_cache.py'
TAIL_INIT_ANCHOR = '        self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}\n'
TAIL_INIT_INSERT = (
    '        # ARC request_end_tail (prefix cache v3): request-end tail entries (vllm/v1/core/arc_tail_cache.py).\n'
    '        from vllm.v1.core.arc_tail_cache import TailIndex as _ArcTailIndex\n'
    '\n'
    '        self._arc_tail = _ArcTailIndex.create(self)\n')
TAIL_LOOKUP_ANCHOR = ('        blocks = self.create_kv_cache_blocks(computed_blocks)\n'
                      '        return blocks, num_new_computed_tokens, shared_prefix_boundary\n')
TAIL_LOOKUP_INSERT = (
    '        if self._arc_tail is not None:\n'
    '            # ARC request_end_tail: extend the hit to a finished request\'s exact end when one matches.\n'
    '            computed_blocks, num_new_computed_tokens = self._arc_tail.lookup(\n'
    '                request, computed_blocks, num_new_computed_tokens, max_cache_hit_length\n'
    '            )\n')
TAIL_FREE_ANCHOR = ('            self.block_pool.free_blocks(pins)\n'
                    '        self.coordinator.free(request.request_id)\n')
TAIL_FREE_REPLACEMENT = ('            self.block_pool.free_blocks(pins)\n'
                         '        if self._arc_tail is not None:\n'
                         '            self._arc_tail.register(request)\n'
                         '        self.coordinator.free(request.request_id)\n'
                         '        if self._arc_tail is not None:\n'
                         '            self._arc_tail.after_free()\n')
TAIL_POP_ANCHOR = '        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n'
TAIL_POP_INSERT = ('        if self._arc_tail is not None:\n'
                   '            # Deferred free: the blocks are still referenced, so after_free only drops the placement.\n'
                   '            self._arc_tail.register(request)\n'
                   '            self._arc_tail.after_free()\n')
MARKER = 'prefix-cache-patch.json'
OPTIONS = ('mamba_align_fix', 'mamba_keep_state', 'mamba_demote_superseded')
# Optional keys (absent = off), so every earlier study's options and identity stay byte-identical.
OPTIONAL = ('request_end_tail',)


def _replace_once(text, anchor, replacement, what):
    if text.count(anchor) != 1:
        raise RuntimeError(f'Prefix-cache patch needs exactly one {what} anchor')
    return text.replace(anchor, replacement)


def patched_scheduler(original):
    if hashlib.sha256(original).hexdigest() != SCHEDULER_SHA256:
        raise RuntimeError('Mamba align fix rejected unrecognized pinned scheduler source')
    text = _replace_once(original.decode(), ALIGN_ANCHOR, ALIGN_REPLACEMENT, 'aligned-split block size')
    compile(text, SCHEDULER_RELATIVE, 'exec')
    return text.encode()


def patched_manager(original, demote=False):
    if hashlib.sha256(original).hexdigest() not in (MANAGER_SHA256, MANAGER_RECLAIMED_SHA256):
        raise RuntimeError('Mamba keep-state patch rejected unrecognized pinned manager source')
    text = original.decode()
    text = _replace_once(text, CLASS_ANCHOR, CLASS_ANCHOR + CLASS_INSERT.format(
        demote=bool(demote), min_tokens=DEMOTE_MIN_TOKENS), 'MambaManager class')
    text = _replace_once(text, KEEP_INIT_ANCHOR, KEEP_INIT_ANCHOR + KEEP_INIT_INSERT, 'align init')
    text = _replace_once(text, KEEP_REMOVE_ANCHOR, KEEP_REMOVE_REPLACEMENT, 'remove_skipped_blocks')
    text = _replace_once(text, KEEP_HELPER_ANCHOR, KEEP_HELPER + KEEP_HELPER_ANCHOR, 'helper')
    text = _replace_once(text, KEEP_POP_ANCHOR, KEEP_POP_REPLACEMENT, 'pop_blocks_for_free')
    compile(text, MANAGER_RELATIVE, 'exec')
    return text.encode()


def patched_pool(original):
    if hashlib.sha256(original).hexdigest() != POOL_SHA256:
        raise RuntimeError('Mamba demote patch rejected unrecognized pinned block pool source')
    text = _replace_once(original.decode(), DEMOTE_ANCHOR, DEMOTE_REPLACEMENT, 'free_blocks')
    compile(text, POOL_RELATIVE, 'exec')
    return text.encode()


def patched_kv_manager(original):
    if hashlib.sha256(original).hexdigest() not in (KV_MANAGER_SHA256, KV_MANAGER_DIAGNOSTICS_SHA256):
        raise RuntimeError('Request-end tail patch rejected unrecognized pinned kv_cache_manager source')
    text = original.decode()
    text = _replace_once(text, TAIL_INIT_ANCHOR, TAIL_INIT_ANCHOR + TAIL_INIT_INSERT, 'KVCacheManager init')
    text = _replace_once(text, TAIL_LOOKUP_ANCHOR, TAIL_LOOKUP_INSERT + TAIL_LOOKUP_ANCHOR, 'get_computed_blocks')
    text = _replace_once(text, TAIL_FREE_ANCHOR, TAIL_FREE_REPLACEMENT, 'free')
    text = _replace_once(text, TAIL_POP_ANCHOR, TAIL_POP_INSERT + TAIL_POP_ANCHOR, 'pop_blocks_for_free')
    compile(text, KV_MANAGER_RELATIVE, 'exec')
    return text.encode()


def validate(options):
    """``None`` or ``{"mamba_align_fix", "mamba_keep_state", "mamba_demote_superseded"}`` booleans, one on, plus the
    optional ``request_end_tail`` boolean (normalized away when false)."""
    if options is None:
        return None
    if (not isinstance(options, dict) or not set(OPTIONS) <= set(options)
            or not set(options) <= set(OPTIONS) | set(OPTIONAL)):
        raise ValueError(f'prefix_cache needs exactly {", ".join(OPTIONS)} (optional: {", ".join(OPTIONAL)})')
    if any(type(v) is not bool for v in options.values()) or not any(options.values()):
        raise ValueError('prefix_cache options are booleans and at least one is on')
    if options['mamba_demote_superseded'] and not options['mamba_keep_state']:
        raise ValueError('mamba_demote_superseded needs mamba_keep_state (the newest state must be held)')
    result = {k: options[k] for k in OPTIONS}
    result.update({k: True for k in OPTIONAL if options.get(k)})
    return result


def _guard(target, allowed, what):
    if hashlib.sha256(target.read_bytes()).hexdigest() not in allowed:
        raise RuntimeError(f'Request-end tail patch rejected unrecognized pinned {what}')


def _apply_one(target, patch, allowed_originals, record):
    current = hashlib.sha256(target.read_bytes()).hexdigest()
    if record is not None and current == record['patched_sha256']:
        return record
    if current not in allowed_originals:
        raise RuntimeError(f'Prefix-cache patch rejected unrecognized pinned {target.name} '
                           '(or its runtime identity changed)')
    # First application, or a server restart re-extracted the pinned file: (re-)apply the same patch.
    patched = patch(target.read_bytes())
    target.write_bytes(patched)
    return {'original_sha256': current, 'patched_sha256': hashlib.sha256(patched).hexdigest()}


def apply(site, output, options):
    options = validate(options)
    if options is None:
        raise ValueError('prefix_cache patch applied without options')
    marker = Path(output) / MARKER
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    if previous and previous.get('options') != options:
        raise RuntimeError('Prefix-cache patch options changed within one working directory')
    identity = {'options': options, 'gpu_validated': False}
    site = Path(site)
    if options['mamba_align_fix']:
        identity['scheduler'] = _apply_one(site / SCHEDULER_RELATIVE, patched_scheduler,
                                           (SCHEDULER_SHA256,), previous.get('scheduler'))
    if options['mamba_keep_state']:
        demote = options['mamba_demote_superseded']
        identity['manager'] = _apply_one(site / MANAGER_RELATIVE, lambda src: patched_manager(src, demote),
                                         (MANAGER_SHA256, MANAGER_RECLAIMED_SHA256), previous.get('manager'))
    if options['mamba_demote_superseded']:
        identity['block_pool'] = _apply_one(site / POOL_RELATIVE, patched_pool, (POOL_SHA256,),
                                            previous.get('block_pool'))
    if options.get('request_end_tail'):
        # The tail index relies on the pinned block pool's and managers' internals: guard the files it does not
        # patch itself (the v2 patches above already guarded the ones they edit).
        if not options['mamba_demote_superseded']:
            _guard(site / POOL_RELATIVE, (POOL_SHA256,), 'block_pool.py')
        if not options['mamba_keep_state']:
            _guard(site / MANAGER_RELATIVE, (MANAGER_SHA256, MANAGER_RECLAIMED_SHA256),
                   'single_type_kv_cache_manager.py')
        helper = Path(__file__).with_name('prefix_tail_runtime.py').read_bytes()
        (site / TAIL_HELPER_RELATIVE).write_bytes(helper)
        identity['tail_helper_sha256'] = hashlib.sha256(helper).hexdigest()
        identity['kv_cache_manager'] = _apply_one(site / KV_MANAGER_RELATIVE, patched_kv_manager,
                                                  (KV_MANAGER_SHA256, KV_MANAGER_DIAGNOSTICS_SHA256),
                                                  previous.get('kv_cache_manager'))
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return identity
