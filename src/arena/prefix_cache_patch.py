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

``spec_decode`` (throughput arm, 3 October; docs/research-throughput/LOG.md; optional key, off unless given). Prefix
caching under MTP-k speculative decoding (``num_speculative_blocks`` k > 0). The pinned runtime loses most hybrid
prefix hits under MTP, which made the 30 September MTP-1 nine-hour arm run at ~0.5x (prefix hit rate 51%):

- ``mamba_keep_state`` / ``mamba_demote_superseded`` switch themselves off under speculation. With this key the kept
  state is the newest cached boundary state below the running block *and its k speculative blocks* (the scan skips
  the last 1 + k table entries), so keep and demote run under MTP-k too;
- the hybrid coordinator applies the EAGLE "last block drop" to every group (no group of this layout is flagged
  ``is_eagle_group``, so all are): a hit at length L is only accepted when the full-attention KV is also cached for
  one more hash unit past L. Every hit point inside a request's last (partial) attention block fails that test, so a
  game's next request falls back to an older full-block boundary: ~2-3K tokens re-prefilled per request on 3,136-token
  blocks. The drop exists because the MTP layer's KV at position L - 1 was written with the token that followed it in
  the request that cached it (a lookahead); it can differ from the new request's token only when the prompts diverge
  exactly at L. With this key the drop is off (``drop_eagle_block = False`` in ``find_longest_cache_hit`` and
  ``find_longest_cache_hit_per_group``): the target model's KV and recurrent states of a hit are exact either way,
  so the served distribution is unchanged; at worst one draft-layer KV entry is stale, which can only lower
  acceptance (rejection sampling keeps the output exact).

CPU replay of the pinned scheduler (``scripts/arena_analysis/prefix_cache_sim.py --spec 1``, 3,136-token blocks,
16-token hash unit, 30 games on 810 blocks): prefix hit rate MTP-0 + v2 0.887; MTP-1 pinned 0.322; + keep only 0.575;
+ no drop only 0.516; + both (this key) 0.878; no leaked blocks or free-queue duplicates.
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
COORDINATOR_RELATIVE = 'vllm/v1/core/kv_cache_coordinator.py'
# Files this module edits, for the CPU patch check (arc_infra.runops.patchcheck); existence only, since the
# manager file is stacked on cache_patch / reclamation at boot. The tail helper is new, so it is not listed.
TARGETS = tuple({'relative': r} for r in (SCHEDULER_RELATIVE, MANAGER_RELATIVE, POOL_RELATIVE, KV_MANAGER_RELATIVE,
                                          COORDINATOR_RELATIVE))
COORDINATOR_SHA256 = '9cd616a0f6bbd27776aa6bb44c7420c95574f9ffb5e254a8f355937642b8451b'
EAGLE_DROP_ANCHOR = '                drop_eagle_block = use_eagle and idx not in eagle_verified\n'
EAGLE_DROP_REPLACEMENT = ('                # ARC spec_decode: no EAGLE last-unit drop (target KV and states of a hit are exact; at\n'
                          '                # most one MTP-layer KV entry is stale, which only moves acceptance).\n'
                          '                drop_eagle_block = False\n')
EAGLE_GROUP_ANCHOR = '                drop_eagle_block=use_eagle,\n'
EAGLE_GROUP_REPLACEMENT = '                drop_eagle_block=False,  # ARC spec_decode: no EAGLE last-unit drop\n'
MARKER = 'prefix-cache-patch.json'
OPTIONS = ('mamba_align_fix', 'mamba_keep_state', 'mamba_demote_superseded')
# Optional keys (absent = off), so every earlier study's options and identity stay byte-identical.
OPTIONAL = ('request_end_tail', 'spec_decode', 'tail_keep_boundary', 'keep_reply_states', 'tail_spec_acceptance',
            'state_cotenant',  # state_cotenant: arena.cache_patch (the layout edit rides this option block)
            'collapse_log',  # diagnostics: ARC_TAIL_COLLAPSE per full-prefix miss (needs request_end_tail)
            'reply_ids',  # API server: a re-sent reply keeps its generated token ids (arena.reply_ids_runtime)
            'tail_keepalive',  # a game idle past 5 s (long python call) has its cached blocks refreshed in the queue
            'admission_tick')  # seconds (a number, not a boolean): waiting requests are admitted together, see TICK_*


def _replace_once(text, anchor, replacement, what):
    if text.count(anchor) != 1:
        raise RuntimeError(f'Prefix-cache patch needs exactly one {what} anchor')
    return text.replace(anchor, replacement)


# tail_spec_acceptance (5 October): the scheduler keeps each request's last verify-step acceptance, so a finish without
# an in-flight step (computed == num_tokens - 1) can register its tail at the state slot that acceptance selects.
ACCEPT_ANCHOR = ('                if not output_is_stale:\n'
                 '                    if request.num_computed_tokens > 0:\n'
                 '                        request.num_computed_tokens -= num_rejected\n')
ACCEPT_REPLACEMENT = ('                if not output_is_stale:\n'
                      '                    request._arc_last_accepted = num_accepted  # ARC tail_spec_acceptance\n'
                      '                    if request.num_computed_tokens > 0:\n'
                      '                        request.num_computed_tokens -= num_rejected\n')


DROP_ANCHOR = ('            if stopped:\n'
               '                del new_token_ids[num_new:]  # Trim new tokens if needed.\n')
DROP_REPLACEMENT = ('            if stopped:\n'
                    '                request._arc_dropped = list(new_token_ids[num_new:])  # ARC tail_spec_acceptance\n'
                    '                del new_token_ids[num_new:]  # Trim new tokens if needed.\n')


# admission_tick (9 October; docs/throughput-arm-status.md): a step that computes prompt tokens next to decoding
# streams takes about 0.25 s against 0.04-0.06 s, whatever the token count (v8.2 long runs: 1.2 such steps per request,
# 18-20% of engine time). With the option, waiting requests are admitted only when ``seconds`` have passed since the
# last step that admitted one, so the starts of a tick share one slow step. Never held: a step that already carries a
# running prefill chunk (slow anyway), the steps after an opening until the requests that were waiting at it are in
# (the token budget can take several), an engine without a decoding stream. No lookup runs for a held request, so a fully cached prompt waits like any
# other. Order is unchanged (the pinned queue), and the wait is bounded by the tick plus the steps its own opening
# needs: a request waits for the clock, never for a later request.
TICK_INIT_ANCHOR = '        self.prefill_capacity_bound = False\n'
TICK_INIT_INSERT = ('        # ARC admission_tick: the next time the waiting queue opens; carry = the last opening left requests\n'
                    '        self._arc_clock = time.monotonic\n'
                    '        self._arc_admit_at = 0.0\n'
                    '        self._arc_admit_carry = False\n'
                    '        self._arc_admit_cohort = 0\n'
                    '        self._arc_admit_stats = {{"openings": 0, "admitted": 0, "max_group": 0, "held_steps": 0,\n'
                    '                                 "rides": 0, "tick_seconds": {seconds!r}}}\n')
TICK_WAIT_ANCHOR = ('        # Next, schedule the WAITING requests.\n'
                    '        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:\n'
                    '            step_skipped_waiting = create_request_queue(self.policy)\n')
TICK_WAIT_REPLACEMENT = (
    '        # Next, schedule the WAITING requests.\n'
    '        # ARC admission_tick: hold them while streams decode, this step carries no prefill chunk, the last opening\n'
    '        # took in everything that waited at it and the tick has not passed since the last admission.\n'
    '        arc_now = self._arc_clock()\n'
    '        arc_hold = bool((self.waiting or self.skipped_waiting) and not prefill_scheduled\n'
    '                        and not self._arc_admit_carry and arc_now < self._arc_admit_at\n'
    '                        and any(not r.is_prefill_chunk for r in self.running))\n'
    '        if arc_hold:\n'
    '            self._arc_admit_stats["held_steps"] += 1\n'
    '        elif not self._arc_admit_carry:\n'
    '            self._arc_admit_cohort = len(self.waiting) + len(self.skipped_waiting)  # waiting at this opening\n'
    '        if not arc_hold and not preempted_reqs and self._pause_state == PauseState.UNPAUSED:\n'
    '            step_skipped_waiting = create_request_queue(self.policy)\n')
TICK_END_ANCHOR = ('            # re-queue requests skipped in this pass ahead of older skipped items.\n'
                   '            if step_skipped_waiting:\n'
                   '                self.skipped_waiting.prepend_requests(step_skipped_waiting)\n')
TICK_END_INSERT = (
    '            # ARC admission_tick: an admission closes the queue for one tick, once the requests that were\n'
    '            # waiting at the opening are in.\n'
    '            arc_new = len(scheduled_new_reqs) + len(scheduled_resumed_reqs)\n'
    '            self._arc_admit_cohort -= arc_new\n'
    '            self._arc_admit_carry = bool(self._arc_admit_cohort > 0 and (self.waiting or self.skipped_waiting))\n'
    '            if arc_new:\n'
    '                self._arc_admit_at = arc_now + {seconds!r}\n'
    '                arc_stats = self._arc_admit_stats\n'
    '                arc_stats["openings"] += 1\n'
    '                arc_stats["admitted"] += arc_new\n'
    '                arc_stats["max_group"] = max(arc_stats["max_group"], arc_new)\n'
    '                arc_stats["rides"] += bool(prefill_scheduled)\n'
    '                if arc_stats["openings"] % 200 == 0:\n'
    '                    print("ARC_ADMIT_TICK_STATS " + str(arc_stats).replace("\'", \'"\'), flush=True)\n')
TICK_MAX_SECONDS = 30.0


def patched_scheduler(original, accept=False, tick=None):
    if hashlib.sha256(original).hexdigest() != SCHEDULER_SHA256:
        raise RuntimeError('Mamba align fix rejected unrecognized pinned scheduler source')
    text = _replace_once(original.decode(), ALIGN_ANCHOR, ALIGN_REPLACEMENT, 'aligned-split block size')
    if accept:
        text = _replace_once(text, ACCEPT_ANCHOR, ACCEPT_REPLACEMENT, 'spec acceptance')
        text = _replace_once(text, DROP_ANCHOR, DROP_REPLACEMENT, 'stop-dropped tokens')
    if tick:
        seconds = float(tick)
        text = _replace_once(text, TICK_INIT_ANCHOR, TICK_INIT_ANCHOR + TICK_INIT_INSERT.format(seconds=seconds),
                             'admission tick state')
        text = _replace_once(text, TICK_WAIT_ANCHOR, TICK_WAIT_REPLACEMENT, 'admission tick hold')
        text = _replace_once(text, TICK_END_ANCHOR, TICK_END_ANCHOR + TICK_END_INSERT.format(seconds=seconds),
                             'admission tick close')
    compile(text, SCHEDULER_RELATIVE, 'exec')
    return text.encode()


# spec_decode (3 October): keep and demote under MTP-k. The kept state lies below the running block and its k
# speculative blocks.
SPEC_ALIGN_OLD = '        arc_align = self.mamba_cache_mode == "align" and self.num_speculative_blocks == 0\n'
SPEC_ALIGN_NEW = ('        # ARC spec_decode: keep/demote also under MTP-k (the scan skips the k speculative blocks).\n'
                  '        arc_align = self.mamba_cache_mode == "align"\n')
SPEC_SCAN_OLD = '        for i in range(len(blocks) - 2, max(-1, len(blocks) - 10), -1):\n'
SPEC_SCAN_NEW = ('        top = len(blocks) - 2 - self.num_speculative_blocks  # ARC spec_decode: below the speculative blocks\n'
                 '        for i in range(top, max(-1, top - 8), -1):\n')


# keep_reply_states (5 October; LOG section 17): a re-sent long reply can differ from the generated tokens inside the
# reply (a full-prefix miss when every state between the prompt and the last decode crossing is gone). The states a
# decode crossing supersedes (the prompt-end state, earlier crossings; up to HOLD_STATES per group) stay held by the
# request and are freed with it, behind its newest state, so the game's next request resumes before the divergence.
HOLD_STATES = 2
HOLD_INIT_OLD = "            self._arc_kept_state: dict[str, KVCacheBlock] = {}\n"
HOLD_INIT_NEW = HOLD_INIT_OLD + "            self._arc_held: dict[str, list[KVCacheBlock]] = {}  # ARC keep_reply_states\n"
HOLD_CALL_OLD = "                self._arc_keep_latest_state(request_id)\n"
HOLD_CALL_NEW = "                self._arc_keep_latest_state(request_id, num_prompt_tokens)\n"
HOLD_DEF_OLD = "    def _arc_keep_latest_state(self, request_id: str) -> None:\n"
HOLD_DEF_NEW = "    def _arc_keep_latest_state(self, request_id: str, num_prompt_tokens: int | None = None) -> None:\n"
HOLD_FREE_OLD = ("            if previous is not None:\n"
                 "                self.block_pool.free_blocks([previous])\n")
HOLD_FREE_NEW = ("            if previous is not None:\n"
                 "                tokens = block.block_hash_num_tokens\n"
                 "                if num_prompt_tokens is not None and tokens is not None and tokens > num_prompt_tokens:\n"
                 "                    # ARC keep_reply_states: a decode crossing; hold the state it supersedes.\n"
                 "                    held = self._arc_held.setdefault(request_id, [])\n"
                 "                    held.append(previous)\n"
                 f"                    if len(held) > {HOLD_STATES}:\n"
                 "                        self.block_pool.free_blocks([held.pop(0)])\n"
                 "                else:\n"
                 "                    self.block_pool.free_blocks([previous])\n")
HOLD_POP_OLD = ("            kept = self._arc_kept_state.pop(request_id, None)\n"
                "            if kept is not None:\n"
                "                # First in allocation order = freed last by the caller's reversed free: the most recently used.\n"
                "                return [kept] + super().pop_blocks_for_free(request_id)\n")
HOLD_POP_NEW = ("            kept = self._arc_kept_state.pop(request_id, None)\n"
                "            held = self._arc_held.pop(request_id, [])[::-1]  # ARC keep_reply_states: newest first\n"
                "            if kept is not None or held:\n"
                "                # First in allocation order = freed last by the caller's reversed free: the most recently used.\n"
                "                return ([kept] if kept is not None else []) + held + super().pop_blocks_for_free(request_id)\n")


def patched_manager(original, demote=False, spec_decode=False, hold=False):
    if hashlib.sha256(original).hexdigest() not in (MANAGER_SHA256, MANAGER_RECLAIMED_SHA256):
        raise RuntimeError('Mamba keep-state patch rejected unrecognized pinned manager source')
    text = original.decode()
    text = _replace_once(text, CLASS_ANCHOR, CLASS_ANCHOR + CLASS_INSERT.format(
        demote=bool(demote), min_tokens=DEMOTE_MIN_TOKENS), 'MambaManager class')
    remove, helper, init, pop = KEEP_REMOVE_REPLACEMENT, KEEP_HELPER, KEEP_INIT_INSERT, KEEP_POP_REPLACEMENT
    if hold:
        remove = _replace_once(remove, HOLD_CALL_OLD, HOLD_CALL_NEW, 'hold call')
        helper = _replace_once(helper, HOLD_DEF_OLD, HOLD_DEF_NEW, 'hold def')
        helper = _replace_once(helper, HOLD_FREE_OLD, HOLD_FREE_NEW, 'hold free')
        init = _replace_once(init, HOLD_INIT_OLD, HOLD_INIT_NEW, 'hold init')
        pop = _replace_once(pop, HOLD_POP_OLD, HOLD_POP_NEW, 'hold pop')
    if spec_decode:
        remove = _replace_once(remove, SPEC_ALIGN_OLD, SPEC_ALIGN_NEW, 'spec keep condition')
        helper = _replace_once(helper, SPEC_SCAN_OLD, SPEC_SCAN_NEW, 'spec keep scan')
    text = _replace_once(text, KEEP_INIT_ANCHOR, KEEP_INIT_ANCHOR + init, 'align init')
    text = _replace_once(text, KEEP_REMOVE_ANCHOR, remove, 'remove_skipped_blocks')
    text = _replace_once(text, KEEP_HELPER_ANCHOR, helper + KEEP_HELPER_ANCHOR, 'helper')
    text = _replace_once(text, KEEP_POP_ANCHOR, pop, 'pop_blocks_for_free')
    compile(text, MANAGER_RELATIVE, 'exec')
    return text.encode()


def patched_coordinator(original):
    if hashlib.sha256(original).hexdigest() != COORDINATOR_SHA256:
        raise RuntimeError('spec_decode patch rejected unrecognized pinned kv_cache_coordinator source')
    text = _replace_once(original.decode(), EAGLE_DROP_ANCHOR, EAGLE_DROP_REPLACEMENT, 'hybrid EAGLE drop')
    text = _replace_once(text, EAGLE_GROUP_ANCHOR, EAGLE_GROUP_REPLACEMENT, 'per-group EAGLE drop')
    compile(text, COORDINATOR_RELATIVE, 'exec')
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


# reply_ids (7 October): 2.3% of replies re-tokenize to another split of the same text, so the re-sent prompt leaves
# the cached sequence inside the reply (9 of 10 full recomputes in tp-streams-dev10-s44-cot-diag-v1). The chat serving
# module substitutes the generated ids back, on exact text identity behind an identical prompt only.
CHAT_SERVING_RELATIVE = 'vllm/entrypoints/openai/chat_completion/serving.py'
CHAT_SERVING_SHA256 = 'a2440b6b76ad87de86bcf6fc7061ad5d02ab67c1c3c89ca7de3eeb29aa12aa2f'
REPLY_IDS_HELPER_RELATIVE = 'vllm/entrypoints/openai/chat_completion/arc_reply_ids.py'
REPLY_IMPORT_ANCHOR = 'from vllm.utils.serial_utils import numpy2base64\n'
REPLY_IMPORT_INSERT = 'from vllm.entrypoints.openai.chat_completion import arc_reply_ids as _arc_reply_ids  # ARC\n'
REPLY_RENDER_ANCHOR = '        conversation, engine_inputs = result\n'
REPLY_RENDER_INSERT = ('        # ARC reply_ids: re-sent replies keep the token ids they were generated with.\n'
                       '        _arc_reply_ids.substitute(\n'
                       '            engine_inputs, tokenizer, self.model_config.max_model_len,\n'
                       '            request.max_completion_tokens if request.max_completion_tokens is not None\n'
                       '            else request.max_tokens)\n')
REPLY_FINISH_ANCHOR = '        choices: list[ChatCompletionResponseChoice] = []\n'
REPLY_FINISH_INSERT = '        _arc_reply_ids.record(final_res, tokenizer)  # ARC reply_ids\n'


def patched_chat_serving(original):
    if hashlib.sha256(original).hexdigest() != CHAT_SERVING_SHA256:
        raise RuntimeError('Reply-id patch rejected unrecognized pinned chat serving source')
    text = original.decode()
    text = _replace_once(text, REPLY_IMPORT_ANCHOR, REPLY_IMPORT_ANCHOR + REPLY_IMPORT_INSERT, 'chat serving import')
    text = _replace_once(text, REPLY_RENDER_ANCHOR, REPLY_RENDER_ANCHOR + REPLY_RENDER_INSERT, 'chat render')
    text = _replace_once(text, REPLY_FINISH_ANCHOR, REPLY_FINISH_INSERT + REPLY_FINISH_ANCHOR, 'chat full generator')
    compile(text, CHAT_SERVING_RELATIVE, 'exec')
    return text.encode()


def validate(options):
    """``None`` or ``{"mamba_align_fix", "mamba_keep_state", "mamba_demote_superseded"}`` booleans, one on, plus the
    optional ``request_end_tail`` boolean (normalized away when false)."""
    if options is None:
        return None
    if (not isinstance(options, dict) or not set(OPTIONS) <= set(options)
            or not set(options) <= set(OPTIONS) | set(OPTIONAL)):
        raise ValueError(f'prefix_cache needs exactly {", ".join(OPTIONS)} (optional: {", ".join(OPTIONAL)})')
    tick = options.get('admission_tick')
    if tick is not None and (type(tick) not in (int, float) or not 0 < tick <= TICK_MAX_SECONDS):
        raise ValueError(f'admission_tick is seconds, a number above 0 and at most {TICK_MAX_SECONDS:g} (absent = off)')
    if tick is not None and not options['mamba_align_fix']:
        raise ValueError('admission_tick rides the scheduler patch: it needs mamba_align_fix')
    if (any(type(v) is not bool for k, v in options.items() if k != 'admission_tick')
            or not any(v for k, v in options.items() if k != 'admission_tick')):
        raise ValueError('prefix_cache options are booleans and at least one is on')
    if options['mamba_demote_superseded'] and not options['mamba_keep_state']:
        raise ValueError('mamba_demote_superseded needs mamba_keep_state (the newest state must be held)')
    if options.get('tail_spec_acceptance') and not (options.get('request_end_tail') and options.get('spec_decode')
                                                    and options['mamba_align_fix']):
        raise ValueError('tail_spec_acceptance needs request_end_tail, spec_decode and mamba_align_fix')
    if options.get('keep_reply_states') and not options['mamba_keep_state']:
        raise ValueError('keep_reply_states extends mamba_keep_state: it needs mamba_keep_state')
    if options.get('collapse_log') and not options.get('request_end_tail'):
        raise ValueError('collapse_log is the tail index diagnostics: it needs request_end_tail')
    if options.get('tail_keepalive') and not options.get('request_end_tail'):
        raise ValueError('tail_keepalive refreshes request-end entries: it needs request_end_tail')
    if options.get('reply_ids') and not options.get('request_end_tail'):
        raise ValueError('reply_ids serves the request-end entry: it needs request_end_tail')
    if options.get('tail_keep_boundary') and not options.get('request_end_tail'):
        raise ValueError('tail_keep_boundary changes request_end_tail placement: it needs request_end_tail')
    result = {k: options[k] for k in OPTIONS}
    result.update({k: True for k in OPTIONAL if options.get(k) and k != 'admission_tick'})
    if tick is not None:
        result['admission_tick'] = float(tick)
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
        accept = bool(options.get('tail_spec_acceptance'))
        tick = options.get('admission_tick')
        identity['scheduler'] = _apply_one(site / SCHEDULER_RELATIVE, lambda src: patched_scheduler(src, accept, tick),
                                           (SCHEDULER_SHA256,), previous.get('scheduler'))
    if options['mamba_keep_state']:
        demote = options['mamba_demote_superseded']
        spec = bool(options.get('spec_decode'))
        identity['manager'] = _apply_one(site / MANAGER_RELATIVE, lambda src: patched_manager(src, demote, spec, bool(options.get('keep_reply_states'))),
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
        if options.get('spec_decode'):
            # MTP-k request-end tails (4 October): the helper registers only in-flight and prefill-only finishes.
            helper = _replace_once(helper.decode(), 'SPEC_TAIL = False\n', 'SPEC_TAIL = True\n',
                                   'tail SPEC_TAIL flag').encode()
        if options.get('tail_spec_acceptance'):
            helper = _replace_once(helper.decode(), 'SPEC_ACCEPT = False\n', 'SPEC_ACCEPT = True\n',
                                   'tail SPEC_ACCEPT flag').encode()
        if options.get('tail_keep_boundary'):
            # Phase 4 (4 October): the request's kept boundary state stays where its free put it (MRU) instead of
            # going to the front, so a game whose tail entry cannot serve still resumes at its last boundary.
            helper = _replace_once(helper.decode(), 'PLACEMENT = "swap"\n', 'PLACEMENT = "back"\n',
                                   'tail PLACEMENT').encode()
        if options.get('tail_keepalive'):
            helper = _replace_once(helper.decode(), 'KEEPALIVE = False\n', 'KEEPALIVE = True\n',
                                   'tail KEEPALIVE flag').encode()
        if options.get('collapse_log'):
            helper = _replace_once(helper.decode(), 'COLLAPSE_LOG = os.environ.get("ARC_TAIL_COLLAPSE_LOG") == "1"\n',
                                   'COLLAPSE_LOG = True\n', 'tail COLLAPSE_LOG flag').encode()
        (site / TAIL_HELPER_RELATIVE).write_bytes(helper)
        identity['tail_helper_sha256'] = hashlib.sha256(helper).hexdigest()
        identity['kv_cache_manager'] = _apply_one(site / KV_MANAGER_RELATIVE, patched_kv_manager,
                                                  (KV_MANAGER_SHA256, KV_MANAGER_DIAGNOSTICS_SHA256),
                                                  previous.get('kv_cache_manager'))
    if options.get('spec_decode'):
        identity['coordinator'] = _apply_one(site / COORDINATOR_RELATIVE, patched_coordinator, (COORDINATOR_SHA256,),
                                             previous.get('coordinator'))
    if options.get('reply_ids'):
        helper = Path(__file__).with_name('reply_ids_runtime.py').read_bytes()
        (site / REPLY_IDS_HELPER_RELATIVE).write_bytes(helper)
        identity['reply_ids_helper_sha256'] = hashlib.sha256(helper).hexdigest()
        identity['chat_serving'] = _apply_one(site / CHAT_SERVING_RELATIVE, patched_chat_serving,
                                              (CHAT_SERVING_SHA256,), previous.get('chat_serving'))
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return identity
