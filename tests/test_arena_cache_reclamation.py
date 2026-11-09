import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arena import cache_reclamation as patch
from arena.serving import configure

FIXTURE = Path(__file__).parent / 'fixtures/vllm-skipped-blocks.py.txt'


def reclaim(fixed=True):
    # Exact extracted upstream method, using the same whitespace replacement as
    # the full-source patch. No CUDA objects are needed to exercise this logic.
    source = FIXTURE.read_text()
    anchor = '\n'.join(line[4:] for line in patch.ANCHOR.splitlines()) + '\n'
    replacement = '\n'.join(line[4:] for line in patch.REPLACEMENT.splitlines()) + '\n'
    assert source.count(anchor) == 1
    if fixed:
        source = source.replace(anchor, replacement)
    namespace = {}
    exec(compile(source, str(FIXTURE), 'exec'), namespace)
    return namespace['_remove_blocks_in_range']


def manager(ids):
    null = NS(block_id=0, ref_cnt=100)
    blocks = {i: NS(block_id=i, ref_cnt=1) for i in ids if i}
    released = []

    def free(items):
        for block in items:
            assert block is not null
            assert block.ref_cnt > 0
            block.ref_cnt -= 1
            released.append(block.block_id)

    result = NS(_null_block=null,
                req_to_blocks={'a': [blocks[i] if i else null for i in ids]},
                block_pool=NS(free_blocks=free))
    return result, blocks, released


def test_pinned_loop_leaves_live_blocks_behind_sparse_gap():
    original, _, released = manager([1, 0, 2, 0, 3])
    reclaim(False)(original, 'a', 0, 4)
    assert released == []  # Last skipped slot is empty, but 1 and 2 remain live.
    fixed, blocks, released = manager([1, 0, 2, 0, 3])
    reclaim()(fixed, 'a', 0, 4)
    assert released == [2, 1]
    assert blocks[3].ref_cnt == 1  # Unprocessed tail is never released.


@pytest.mark.parametrize('ids', [[1, 2, 3], [0, 0, 0], [1, 0, 2, 0, 3], []])
def test_reclamation_is_idempotent_and_handles_end_beyond_allocation(ids):
    obj, blocks, released = manager(ids)
    remove = reclaim()
    remove(obj, 'a', 0, 100)
    first = list(released)
    remove(obj, 'a', 0, 100)
    assert released == first == list(reversed([i for i in ids if i]))
    assert all(b.ref_cnt == 0 for b in blocks.values())


def test_partial_range_and_shared_reference_are_preserved():
    obj, blocks, released = manager([1, 2, 0, 3, 4])
    blocks[2].ref_cnt = 2
    obj.req_to_blocks['b'] = [blocks[2]]
    reclaim()(obj, 'a', 1, 4)
    assert released == [3, 2]
    assert blocks[1].ref_cnt == blocks[4].ref_cnt == 1
    assert obj.req_to_blocks['b'][0] is blocks[2] and blocks[2].ref_cnt == 1


def test_missing_request_and_empty_range_do_not_release():
    obj, _, released = manager([1, 2])
    reclaim()(obj, 'missing', 0, 2)
    reclaim()(obj, 'a', 1, 1)
    assert not released


def test_apply_is_hash_guarded_idempotent_and_detects_mutation(tmp_path, monkeypatch):
    target = tmp_path / patch.RELATIVE
    target.parent.mkdir(parents=True)
    # A small synthetic class tests patch plumbing; the separate pinned-method
    # tests above and full-source CPU audit verify the production source.
    source = ('class Manager:\n' + '\n'.join('    ' + x for x in FIXTURE.read_text().splitlines()) + '\n').encode()
    target.write_bytes(source)
    with pytest.raises(RuntimeError, match='unrecognized'):
        patch.apply(tmp_path, tmp_path)
    assert target.read_bytes() == source
    monkeypatch.setattr(patch, 'ORIGINAL_SHA256', hashlib.sha256(source).hexdigest())
    result = patch.apply(tmp_path, tmp_path)
    assert not result['gpu_validated']
    assert patch.apply(tmp_path, tmp_path) == result
    assert target.read_bytes() != source
    patched = target.read_bytes()
    # A server restart re-extracts the pinned runtime: the original source is patched again.
    target.write_bytes(source)
    assert patch.apply(tmp_path, tmp_path) == result and target.read_bytes() == patched
    target.write_bytes(target.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match='identity changed'):
        patch.apply(tmp_path, tmp_path)


@pytest.mark.parametrize('enabled', [False, True])
def test_serving_patch_requires_explicit_opt_in(monkeypatch, tmp_path, enabled):
    from arena import serving
    calls = []
    monkeypatch.setattr(serving, 'apply', lambda *a: calls.append('telemetry'))
    monkeypatch.setattr(serving, 'apply_reclamation', lambda *a: calls.append('reclaim'))
    recipe = NS(verify_and_extract_runtime=lambda: {'ready': True},
                server_command=lambda: ['serve'], RUNTIME_ROOT=tmp_path)
    configure(recipe, tmp_path, reclaim_skipped_blocks=enabled)
    assert recipe.verify_and_extract_runtime() == {'ready': True}
    assert calls == ['telemetry'] + (['reclaim'] if enabled else [])
    assert recipe.server_command() == ['serve']


@pytest.mark.parametrize('value', ['false', 1, None])
def test_invalid_opt_in_fails_before_mutation(value, tmp_path):
    with pytest.raises(ValueError, match='must be boolean'):
        configure(NS(), tmp_path, reclaim_skipped_blocks=value)
