"""Prefix cache v2 (29 September): Mamba chunk alignment and keep-latest-state patches, opt-in wiring."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arena import prefix_cache_patch as patch
from arena.serving import configure, lab_overrides

FIXTURES = Path(__file__).parent / 'fixtures'
SPLIT = FIXTURES / 'vllm-mamba-aligned-split.py.txt'
ALL = {'mamba_align_fix': True, 'mamba_keep_state': True, 'mamba_demote_superseded': True}


def dedent(text):
    return '\n'.join(line[4:] for line in text.splitlines()) + '\n'


def split_method(fixed):
    source = SPLIT.read_text()
    anchor = dedent(patch.ALIGN_ANCHOR)
    assert source.count(anchor) == 1
    if fixed:
        source = source.replace(anchor, dedent(patch.ALIGN_REPLACEMENT))
    namespace = {'Request': object}
    exec(compile(source, str(SPLIT), 'exec'), namespace)
    return namespace['_mamba_block_aligned_split']


def hero_scheduler():
    """The served hero layout: EngineCore's cache_config.block_size is the ring's 4, the Mamba block 1,600."""
    from vllm_stub import CircularBufferSpec, FullAttentionSpec, MambaSpec
    groups = [NS(kv_cache_spec=FullAttentionSpec(1600)), NS(kv_cache_spec=CircularBufferSpec(4))]
    groups += [NS(kv_cache_spec=MambaSpec(1600)) for _ in range(4)]
    return NS(cache_config=NS(block_size=4), kv_cache_config=NS(kv_cache_groups=groups), block_size=1600,
              hash_block_size=1600, use_eagle=False, max_num_scheduled_tokens=4096,
              scheduler_config=NS(long_prefill_token_threshold=0), mamba_partial_cache_hit=False)


@pytest.fixture(autouse=True)
def stub_mamba_spec(monkeypatch, tmp_path):
    # The replacement imports vllm.v1.kv_cache_interface.MambaSpec; provide a stand-in hierarchy.
    import sys
    import types

    class Spec:
        def __init__(self, block_size):
            self.block_size = block_size

    class FullAttentionSpec(Spec):
        pass

    class CircularBufferSpec(Spec):
        pass

    class MambaSpec(Spec):
        pass

    stub = types.ModuleType('vllm_stub')
    stub.FullAttentionSpec, stub.CircularBufferSpec, stub.MambaSpec = FullAttentionSpec, CircularBufferSpec, MambaSpec
    modules = {'vllm_stub': stub, 'vllm': types.ModuleType('vllm'), 'vllm.v1': types.ModuleType('vllm.v1'),
               'vllm.v1.kv_cache_interface': types.ModuleType('vllm.v1.kv_cache_interface')}
    modules['vllm.v1.kv_cache_interface'].MambaSpec = MambaSpec
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def request(computed, prompt):
    return NS(num_computed_tokens=computed, num_prompt_tokens=prompt, num_tokens=prompt, shared_prefix_boundary=0)


def chunks(split, sched, hit, prompt, budget=4096):
    ends, done = [], hit
    while done < prompt:
        n = split(sched, request(done, prompt), min(budget, prompt - done))
        assert n > 0
        done += n
        ends.append(done)
    return ends


def test_pinned_split_ends_chunks_on_the_ring_block_not_the_mamba_block():
    sched = hero_scheduler()
    # A 3,000-token continuation after a hit at 32,000: the pinned split stops only 4-aligned, so no chunk ends on the
    # 33,600 Mamba boundary (the prompt-end state the next request of the game needs).
    assert chunks(split_method(False), sched, 32000, 35003) == [35000, 35003]
    assert chunks(split_method(True), sched, 32000, 35003) == [33600, 35003]
    # An evicted 12K re-prefill in 4,096-token budgets: pinned chunk ends fall mid-block (their states are then cached
    # under the next boundary's hash); fixed ends are Mamba boundaries.
    pinned = chunks(split_method(False), sched, 3200, 15500)
    fixed = chunks(split_method(True), sched, 3200, 15500)
    assert pinned[:-1] == [7296, 11392, 15488] and any(e % 1600 for e in pinned[:-1])
    assert fixed == [6400, 9600, 12800, 14400, 15500] and all(e % 1600 == 0 for e in fixed[:-1])


def test_fixed_split_falls_back_to_the_scheduler_block_without_a_plain_mamba_group():
    sched = hero_scheduler()
    sched.kv_cache_config.kv_cache_groups = sched.kv_cache_config.kv_cache_groups[:2]
    assert chunks(split_method(True), sched, 32000, 35003) == [33600, 35003]


def helper_class(demote=True):
    """The patch's MambaManager helpers in a class over a recording base (zero-argument super() needs a class)."""
    source = ('from collections.abc import Sequence\n\nclass KVCacheBlock:\n    pass\n\nclass Base:\n'
              '    def add_local_computed_blocks(self, *args):\n        self.base_calls.append(args)\n\n'
              'class Manager(Base):\n' f'    _arc_demote = {demote}\n'
              f'    _arc_demote_min_tokens = {patch.DEMOTE_MIN_TOKENS}\n\n' + patch.KEEP_HELPER)
    namespace = {}
    exec(compile(source, 'keep_helper', 'exec'), namespace)
    return namespace['Manager']


def keep_helper():
    return helper_class()._arc_keep_latest_state


def manager(blocks):
    freed = []
    null = NS(is_null=True, block_hash=None, name='null')
    table = [null if b is None else NS(is_null=False, block_hash=b[1], name=b[0]) for b in blocks]
    return NS(req_to_blocks={'r': table}, _null_block=null, _arc_kept_state={},
              block_pool=NS(free_blocks=lambda items: freed.extend(x.name for x in items))), freed


def test_demote_only_states_no_request_has_resumed_from():
    cls = helper_class()
    mgr = cls()
    mgr._arc_hits, mgr.base_calls = {}, []
    garbage = NS(block_id=7, block_hash='h7', block_hash_num_tokens=12800, is_null=False)
    shared = NS(block_id=3, block_hash='h3', block_hash_num_tokens=11200, is_null=False)
    system = NS(block_id=2, block_hash='h2', block_hash_num_tokens=3200, is_null=False)
    null = NS(block_id=0, block_hash=None, block_hash_num_tokens=None, is_null=True)
    assert mgr._arc_is_superseded_state(garbage) and mgr._arc_is_superseded_state(shared)
    # The shared system prompt's state (within the first 8,192 tokens) is never demoted, hit or not.
    assert patch.DEMOTE_MIN_TOKENS == 8192 and not mgr._arc_is_superseded_state(system)
    mgr.add_local_computed_blocks('r', [null, null, shared], 4800, 0)  # a request resumes from `shared`
    assert mgr.base_calls == [('r', [null, null, shared], 4800, 0)]
    assert not mgr._arc_is_superseded_state(shared) and mgr._arc_hits[3] == ('h3', 1)
    # The block was evicted and reused under another hash: its hit count no longer applies.
    shared.block_hash = 'h3b'
    assert mgr._arc_is_superseded_state(shared)
    mgr.add_local_computed_blocks('s', [null], 0, 0)  # a null hit records nothing
    assert set(mgr._arc_hits) == {3}
    off = helper_class(demote=False)()
    off._arc_hits, off.base_calls = {}, []
    off.add_local_computed_blocks('r', [shared], 1600, 0)
    assert off._arc_hits == {} and off.base_calls == [('r', [shared], 1600, 0)]


def pool_free(fixed):
    source = (FIXTURES / 'vllm-block-pool-free.py.txt').read_text()
    anchor = dedent(patch.DEMOTE_ANCHOR)
    assert source.count(anchor) == 1
    if fixed:
        source = source.replace(anchor, dedent(patch.DEMOTE_REPLACEMENT))
    namespace = {'Iterable': list, 'KVCacheBlock': object}
    exec(compile(source, 'free_blocks', 'exec'), namespace)
    return namespace['free_blocks']


@pytest.mark.parametrize('fixed', [False, True])
def test_demoted_states_are_reused_first_but_keep_their_hash(fixed):
    first, last = [], []
    pool = NS(enable_caching=True, free_block_queue=NS(prepend_n=first.extend, append_n=last.extend))
    blocks = [NS(name=n, ref_cnt=1, is_null=False, block_hash=h) for n, h in
              (('garbage', 'hg'), ('prefix', 'hp'), ('unhashed', None))]
    shared = NS(name='shared', ref_cnt=2, is_null=False, block_hash='hs')
    pool._arc_evict_first = lambda b: b.name == 'garbage'
    pool_free(fixed)(pool, blocks + [shared])
    if fixed:
        assert [b.name for b in first] == ['garbage', 'unhashed'] and [b.name for b in last] == ['prefix']
    else:
        assert [b.name for b in first] == ['unhashed'] and [b.name for b in last] == ['garbage', 'prefix']
    assert blocks[0].block_hash == 'hg' and shared.ref_cnt == 1  # still hittable; shared blocks stay held
    # Without a predicate (every free outside the demote window) the patched pool is the pinned pool.
    first.clear(), last.clear()
    for b in blocks:
        b.ref_cnt = 1
    del pool._arc_evict_first
    pool_free(fixed)(pool, blocks)
    assert [b.name for b in first] == ['unhashed'] and [b.name for b in last] == ['garbage', 'prefix']


def test_keep_holds_the_newest_cached_boundary_state_and_frees_the_previous():
    keep = keep_helper()
    # [hit X (cached), null, prefill-end Z (cached), running W (no hash)]
    mgr, freed = manager([('X', 'h1'), None, ('Z', 'h3'), ('W', None)])
    keep(mgr, 'r')
    assert mgr._arc_kept_state['r'].name == 'Z' and freed == []
    assert [b.name for b in mgr.req_to_blocks['r']] == ['X', 'null', 'null', 'W']
    # A decode crossing: W now holds the boundary state (cached) and V runs; Z is released, W kept.
    mgr.req_to_blocks['r'][3].block_hash = 'h4'
    mgr.req_to_blocks['r'].append(NS(is_null=False, block_hash=None, name='V'))
    keep(mgr, 'r')
    assert mgr._arc_kept_state['r'].name == 'W' and freed == ['Z']
    # Nothing cached below the running block: no change; an uncached state is left to the pinned path.
    mgr2, freed2 = manager([None, ('U', None), ('R', None)])
    keep(mgr2, 'r')
    assert mgr2._arc_kept_state == {} and freed2 == []


def test_keep_scan_is_bounded_to_the_blocks_below_the_running_one():
    keep = keep_helper()
    mgr, _ = manager([('X', 'h1')] + [None] * 12 + [('W', None)])
    keep(mgr, 'r')
    assert mgr._arc_kept_state == {}  # the old hit block is left to the pinned remove_skipped_blocks path


def test_remove_replacement_keeps_then_frees_inside_the_demote_window():
    # First in allocation order is freed last by the caller's reversed free: the most recently used entry.
    assert 'return [kept] + super().pop_blocks_for_free(request_id)' in patch.KEEP_POP_REPLACEMENT
    text = patch.KEEP_REMOVE_REPLACEMENT
    # The predicate is set before the keep step (a replaced kept state is superseded) and cleared in finally.
    assert (text.index('self.block_pool._arc_evict_first = self._arc_is_superseded_state')
            < text.index('self._arc_keep_latest_state(request_id)')
            < text.index('self._arc_pinned_remove_skipped_blocks(') < text.index('finally:')
            < text.index('self.block_pool._arc_evict_first = None'))
    assert 'self.num_speculative_blocks == 0' in text
    assert patch.KEEP_REMOVE_ANCHOR.split('\n', 2)[2] in text  # the pinned body follows unchanged


def synthetic_sources():
    scheduler = ('class Scheduler:\n    def _mamba_block_aligned_split(self):\n' + patch.ALIGN_ANCHOR
                 + '        return block_size\n').encode()
    manager_src = ('from collections.abc import Sequence\nfrom typing import ClassVar\n\n'
                   'class KVCacheBlock:\n    pass\n\nclass SingleTypeKVCacheManager:\n'
                   '    def remove_skipped_blocks(self, *a):\n        pass\n\n'
                   '    def pop_blocks_for_free(self, request_id):\n        return []\n\n'
                   'class MambaSpec:\n    pass\n\n' + patch.CLASS_ANCHOR + '\n    def __init__(self):\n'
                   '        if True:\n' + patch.KEEP_INIT_ANCHOR
                   + '\n    def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):\n'
                   + patch.KEEP_REMOVE_ANCHOR + '            pass\n\n' + patch.KEEP_HELPER_ANCHOR + '        return 0\n\n'
                   + '    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:\n'
                   + '        if self.mamba_cache_mode == "align":\n' + patch.KEEP_POP_ANCHOR).encode()
    pool = ('class BlockPool:\n    def free_blocks(self, ordered_blocks):\n'
            '        blocks_to_evict_first, blocks_to_evict_last = [], []\n'
            '        for block in ordered_blocks:\n            if True:\n' + patch.DEMOTE_ANCHOR
            + '                    blocks_to_evict_last.append(block)\n').encode()
    return scheduler, manager_src, pool


def test_apply_is_hash_guarded_idempotent_and_reapplies_after_restart(tmp_path, monkeypatch):
    sources = dict(zip((patch.SCHEDULER_RELATIVE, patch.MANAGER_RELATIVE, patch.POOL_RELATIVE), synthetic_sources()))
    for relative, source in sources.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(source)
    with pytest.raises(RuntimeError, match='unrecognized pinned scheduler.py'):
        patch.apply(tmp_path, tmp_path, ALL)
    assert all((tmp_path / r).read_bytes() == s for r, s in sources.items())
    monkeypatch.setattr(patch, 'SCHEDULER_SHA256', hashlib.sha256(sources[patch.SCHEDULER_RELATIVE]).hexdigest())
    monkeypatch.setattr(patch, 'MANAGER_RECLAIMED_SHA256', hashlib.sha256(sources[patch.MANAGER_RELATIVE]).hexdigest())
    monkeypatch.setattr(patch, 'POOL_SHA256', hashlib.sha256(sources[patch.POOL_RELATIVE]).hexdigest())
    first = patch.apply(tmp_path, tmp_path, ALL)
    assert not first['gpu_validated'] and first['options'] == ALL and set(first) >= {'scheduler', 'manager', 'block_pool'}
    patched = {r: (tmp_path / r).read_bytes() for r in sources}
    assert all(patched[r] != sources[r] for r in sources)
    assert b'_arc_demote: ClassVar[bool] = True' in patched[patch.MANAGER_RELATIVE]
    assert patch.apply(tmp_path, tmp_path, ALL) == first  # idempotent
    # A server restart re-extracts the pinned runtime: the same patches re-apply to the same bytes.
    for relative, source in sources.items():
        (tmp_path / relative).write_bytes(source)
    assert patch.apply(tmp_path, tmp_path, ALL) == first
    assert {r: (tmp_path / r).read_bytes() for r in sources} == patched
    (tmp_path / patch.SCHEDULER_RELATIVE).write_bytes(sources[patch.SCHEDULER_RELATIVE] + b'\n')
    with pytest.raises(RuntimeError, match='identity'):
        patch.apply(tmp_path, tmp_path, ALL)
    with pytest.raises(RuntimeError, match='options changed'):
        patch.apply(tmp_path, tmp_path, {**ALL, 'mamba_demote_superseded': False})


def test_keep_without_demote_bakes_the_flag_off(tmp_path, monkeypatch):
    _, manager_src, _ = synthetic_sources()
    monkeypatch.setattr(patch, 'MANAGER_SHA256', hashlib.sha256(manager_src).hexdigest())
    text = patch.patched_manager(manager_src, demote=False)
    assert b'_arc_demote: ClassVar[bool] = False' in text


def test_pinned_source_identity_is_recorded():
    source = json.loads((FIXTURES / 'vllm-mamba-aligned-split-source.json').read_text())
    assert source['sha256'] == patch.SCHEDULER_SHA256 and source['path'] == patch.SCHEDULER_RELATIVE


@pytest.mark.parametrize('value', [{}, {'mamba_align_fix': True}, {k: False for k in ALL},
                                   {**ALL, 'mamba_align_fix': 1}, {**ALL, 'mamba_keep_state': False}, 'on'])
def test_invalid_options_fail_before_mutation(value, tmp_path):
    with pytest.raises(ValueError):
        configure(NS(), tmp_path, prefix_cache=value)


def recipe(tmp_path):
    return NS(verify_and_extract_runtime=lambda: {'ready': True}, server_command=lambda: ['serve', 'model'],
              RUNTIME_ROOT=tmp_path)


def test_default_arms_are_unchanged(monkeypatch, tmp_path):
    from arena import serving
    calls = []
    monkeypatch.setattr(serving, 'apply', lambda *a: calls.append('telemetry'))
    monkeypatch.setattr(serving, 'apply_reclamation', lambda *a: calls.append('reclaim'))
    monkeypatch.setattr(serving, 'apply_prefix_cache', lambda *a: calls.append('prefix'))
    r = recipe(tmp_path)
    result = configure(r, tmp_path, reclaim_skipped_blocks=True)
    assert r.verify_and_extract_runtime() == {'ready': True}
    assert calls == ['telemetry', 'reclaim'] and r.server_command() == ['serve', 'model']
    assert 'prefix_cache' not in result and 'prompt_tokens_details' not in result


def test_opt_in_applies_after_reclamation_and_adds_the_details_flag(monkeypatch, tmp_path):
    from arena import serving
    calls = []
    monkeypatch.setattr(serving, 'apply', lambda *a: calls.append('telemetry'))
    monkeypatch.setattr(serving, 'apply_reclamation', lambda *a: calls.append('reclaim'))
    monkeypatch.setattr(serving, 'apply_prefix_cache', lambda site, out, opts: calls.append(('prefix', opts)))
    r = recipe(tmp_path)
    result = configure(r, tmp_path, reclaim_skipped_blocks=True, prefix_cache=ALL, prompt_tokens_details=True)
    r.verify_and_extract_runtime()
    assert calls == ['telemetry', 'reclaim', ('prefix', ALL)]
    assert r.server_command() == ['serve', 'model', '--enable-prompt-tokens-details']
    assert result['prefix_cache'] == ALL and result['prompt_tokens_details'] is True


def test_lab_overrides_parse_prefix_cache_and_details():
    assert lab_overrides({'ARENA_PREFIX_CACHE': 'off'}) == {'prefix_cache': None}
    assert lab_overrides({'ARENA_PREFIX_CACHE': json.dumps(ALL), 'ARENA_PROMPT_TOKENS_DETAILS': '1'}) == {
        'prefix_cache': ALL, 'prompt_tokens_details': True}
    with pytest.raises(ValueError):
        lab_overrides({'ARENA_PROMPT_TOKENS_DETAILS': 'yes'})
