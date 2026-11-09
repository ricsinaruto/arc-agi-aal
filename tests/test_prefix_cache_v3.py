"""Prefix cache v3 (29 September; docs/validation-20260927/prefix-cache-v3.md): the request-end tail index against a
fake block pool, its hash-guarded patch, the arena keep-alive and its gate reservations, the tail-check and replay
accounting against a fake engine, the hero-v2 components and the smoke bundle. CPU only."""

import ast
import asyncio
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest

from arena import gate as gate_module
from arena import keep_alive, prefix_replay, serving
from arena import prefix_cache_patch as patch
from arena import throughput_bench as tb

ROOT = Path(__file__).resolve().parents[1]
V2 = {'mamba_align_fix': True, 'mamba_keep_state': True, 'mamba_demote_superseded': True}


# ----------------------------------------------------------------------------------------------- a fake block pool
class Block:
    def __init__(self, block_id, is_null=False):
        self.block_id, self.is_null, self.ref_cnt, self.block_hash = block_id, is_null, 0, None


class HashMap:
    def __init__(self):
        self.data = {}

    def contain(self, key, block_id):
        return block_id in self.data.get(key, {})

    def insert(self, key, block):
        self.data.setdefault(key, {})[block.block_id] = block

    def get_one_block(self, key):
        blocks = self.data.get(key)
        return next(iter(blocks.values())) if blocks else None


class Queue:
    def __init__(self):
        self.items = []

    def remove(self, block):
        self.items.remove(block)

    def prepend_n(self, blocks):
        self.items[:0] = blocks

    def append_n(self, blocks):
        self.items.extend(blocks)


class Pool:
    def __init__(self, hash_block_size):
        self.hash_block_size = hash_block_size
        self.null_block = Block(0, is_null=True)
        self.cached_block_hash_to_block = HashMap()
        self.free_block_queue = Queue()
        self.keys = {}

    def _insert_block_hash(self, key, block, num_tokens=None):
        if block.block_hash is None:
            block.block_hash = key
        self.keys.setdefault(block.block_id, set()).add(key)
        self.cached_block_hash_to_block.insert(key, block)

    def evict(self, block):
        for key in self.keys.pop(block.block_id, ()):
            self.cached_block_hash_to_block.data.get(key, {}).pop(block.block_id, None)
        block.block_hash = None

    def get_cached_block(self, block_hash, group_ids):
        found = [self.cached_block_hash_to_block.get_one_block(block_hash + g.to_bytes(4, "big")) for g in group_ids]
        return None if None in found else found

    def touch(self, blocks):
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                (self.free_block_queue.append_n if block.block_hash is not None
                 else self.free_block_queue.prepend_n)([block])


class Manager:
    def __init__(self, gid, block_size, pool):
        self.kv_cache_group_id, self.block_size, self.block_pool = gid, block_size, pool
        self.req_to_blocks, self._pending_cow_copies = {}, []
        self.mamba_cache_mode, self.num_speculative_blocks = "align", 0

    def get_num_blocks_to_allocate(self, request_id, *args, **kwargs):
        return 0 if self.req_to_blocks.get(request_id) else 1

    def add_local_computed_blocks(self, request_id, *args, **kwargs):
        return None

    def allocate_new_blocks(self, request_id, *args, **kwargs):
        block = Block(900 + len(self._pending_cow_copies))
        block.ref_cnt = 1
        self.req_to_blocks[request_id] = [block]
        return [block]

    def pop_blocks_for_free(self, request_id):
        return self.req_to_blocks.pop(request_id, [])


FullAttentionManager = type("FullAttentionManager", (Manager,), {})
MambaManager = type("MambaManager", (Manager,), {})
CircularBufferManager = type("CircularBufferManager", (Manager,), {})


@pytest.fixture
def runtime(monkeypatch):
    """arena.prefix_tail_runtime imported against stand-ins for the two pinned vLLM helpers it uses."""
    utils = types.ModuleType("vllm.utils.math_utils")
    utils.cdiv = lambda a, b: -(-a // b)
    kv_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_utils.make_block_hash_with_group_id = lambda h, g: h + g.to_bytes(4, "big")
    kv_utils.generate_block_hash_extra_keys = lambda request, start, end, mm: (getattr(request, "extra", None), 0)
    for name in ("vllm", "vllm.utils", "vllm.v1", "vllm.v1.core"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.utils.math_utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", kv_utils)
    spec = importlib.util.spec_from_file_location("arc_tail_cache", ROOT / "src/arena/prefix_tail_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = 8


def hashes(tokens):
    """Chained per-block prefix hashes (one per whole block of ``tokens``)."""
    out, h = [], b""
    for i in range(len(tokens) // B):
        h = hashlib.sha256(h + bytes(str(tokens[i * B:(i + 1) * B]), "ascii")).digest()
        out.append(h)
    return out


class Request:
    def __init__(self, rid, tokens, computed, finished=True):
        self.request_id, self.all_token_ids = rid, list(tokens)
        self.num_tokens, self.num_computed_tokens = len(tokens), computed
        self.block_hashes, self.finished = hashes(tokens), finished

    def is_finished(self):
        return self.finished


def hero_like(runtime, capsys=None):
    """One attention group (0), the ring (1) and two Mamba groups (2, 3), as on the hero layout."""
    pool = Pool(B)
    managers = [FullAttentionManager(0, B, pool), CircularBufferManager(1, 4, pool),
                MambaManager(2, B, pool), MambaManager(3, B, pool)]
    manager = NS(enable_caching=True, block_pool=pool, use_eagle=False,
                 coordinator=NS(single_type_managers=managers), kv_cache_config=NS(kv_cache_groups=[None] * 4))
    return runtime.TailIndex.create(manager), pool, managers


def finished_request(pool, managers, tokens, computed, rid="r0", kept=None):
    """A finished request's tables: attention blocks with full-block hashes, Mamba running states, the ring."""
    req = Request(rid, tokens, computed)
    k = computed // B
    attention = [Block(10 + i) for i in range(k + 1)]
    for i in range(k):
        pool._insert_block_hash(req.block_hashes[i] + (0).to_bytes(4, "big"), attention[i])
    managers[0].req_to_blocks[rid] = attention
    managers[1].req_to_blocks[rid] = [Block(30)]
    for gid, base in ((2, 40), (3, 50)):
        managers[gid].req_to_blocks[rid] = [pool.null_block] * k + [Block(base + k)]
        if kept is not None:
            managers[gid]._arc_kept_state = {rid: kept[gid]}
    for table in (managers[0].req_to_blocks[rid], managers[1].req_to_blocks[rid], managers[2].req_to_blocks[rid],
                  managers[3].req_to_blocks[rid]):
        for block in table:
            if not block.is_null:
                block.ref_cnt = 1
    return req


def test_register_keys_the_exact_end_and_only_finished_unaligned_requests(runtime):
    tail, pool, managers = hero_like(runtime)
    assert tail.disabled is None and tail.attention == [0] and tail.ring == [1] and tail.mamba == [2, 3]
    tokens = list(range(100, 125))  # 25 tokens: blocks of 8 -> k = 2, tail [16, 21)
    req = finished_request(pool, managers, tokens, 21)
    tail.register(req)
    assert tail.stats["registered"] == 1 and tail.stats["registered_tail_tokens"] == 5
    (entry,) = tail.entries[req.block_hashes[1]]
    assert entry.length == 21 and entry.tokens == tokens[16:21]
    blocks = {gid: block for gid, (key, block) in entry.keys.items()}
    assert blocks == {0: managers[0].req_to_blocks["r0"][2], 1: managers[1].req_to_blocks["r0"][0],
                      2: managers[2].req_to_blocks["r0"][2], 3: managers[3].req_to_blocks["r0"][2]}
    assert all(b.block_hash is not None for b in blocks.values()) and tail.valid(entry)
    # Not finished, block-aligned, or a Mamba table whose last column is not the running state: no entry.
    for request in (Request("r1", tokens, 21, finished=False), finished_request(pool, managers, tokens, 16, "r2")):
        tail.register(request)
    bad = finished_request(pool, managers, tokens, 21, "r3")
    managers[2].req_to_blocks["r3"].append(Block(99))
    tail.register(bad)
    assert tail.stats["registered"] == 1 and tail.stats["skip_aligned_or_short"] == 1
    assert tail.stats["skip_mamba_layout"] == 1


def test_lookup_builds_the_partial_hit_shape_and_rejects_mismatches_and_evictions(runtime):
    tail, pool, managers = hero_like(runtime)
    tokens = list(range(200, 230))
    req = finished_request(pool, managers, tokens[:25], 21)
    tail.register(req)
    for mgr in managers:
        pool.free_blocks(reversed(mgr.pop_blocks_for_free("r0")))
    new = Request("n0", tokens, 0, finished=False)  # continues the finished request past its end
    attention = managers[0]
    # The pinned hit: 2 whole blocks (attention blocks 0-1, the Mamba state at 16 is not modelled here).
    pinned = ([], [], [pool.null_block, Block(77)], [pool.null_block, Block(78)])
    blocks, hit = tail.lookup(new, pinned, 16, len(tokens) - 1)
    assert hit == 21 and tail.stats["hits"] == 1 and tail.stats["hit_tokens_above_pinned"] == 5
    (entry,) = tail.entries[req.block_hashes[1]]
    ids = [b.block_id for b in blocks[0]]
    assert ids == [10, 11, entry.keys[0][1].block_id]
    assert blocks[1] == [] and tail.pending[1]["n0"] is entry.keys[1][1]
    for gid in (2, 3):
        assert blocks[gid][:2] == [pool.null_block, pool.null_block] and blocks[gid][2] is entry.keys[gid][1]
    assert attention.req_to_blocks == {}
    # A different token inside the tail, a hit already at or past the entry, or a prompt too short: no entry.
    other = Request("n1", tokens[:18] + [9] + tokens[19:], 0, finished=False)
    assert tail.lookup(other, pinned, 16, 29)[1] == 16
    assert tail.lookup(new, pinned, 21, 29)[1] == 21 and tail.lookup(new, pinned, 16, 20)[1] == 16
    # Eviction of any entry block drops its key: the entry is no longer served.
    pool.evict(entry.keys[3][1])
    assert tail.lookup(new, pinned, 16, 29)[1] == 16 and tail.stats["lookup_invalid"] == 1
    tail.sweep()
    assert tail.entries == {} and tail.stats["swept"] == 1


def test_ring_copy_is_reserved_touched_queued_and_retained(runtime):
    tail, pool, managers = hero_like(runtime)
    ring = managers[1]
    source = Block(31)
    source.block_hash = b"k"
    pool.free_block_queue.append_n([source])
    tail.pending[1]["n0"] = source
    assert ring.get_num_blocks_to_allocate("n0", 0, [], 0, 0, 0) == 2  # its own block + the evictable source
    ring.add_local_computed_blocks("n0", [], 21, 0)
    assert source.ref_cnt == 1 and source not in pool.free_block_queue.items and "n0" not in tail.pending[1]
    (new,) = ring.allocate_new_blocks("n0", 30, 30)
    assert ring._pending_cow_copies == [(source, new)] and new.ref_cnt == 2 and tail.stats["ring_copies"] == 1
    # A request freed between touch and allocation returns the source; one never touched leaves nothing behind.
    other = Block(32)
    other.block_hash = b"j"
    pool.free_block_queue.append_n([other])
    tail.pending[1]["n1"] = other
    ring.add_local_computed_blocks("n1", [], 21, 0)
    ring.pop_blocks_for_free("n1")
    assert other.ref_cnt == 0 and pool.free_block_queue.items[-1] is other
    tail.pending[1]["n2"] = Block(33)
    ring.pop_blocks_for_free("n2")
    assert "n2" not in tail.pending[1]


def test_swap_placement_sends_the_superseded_kept_state_to_the_front(runtime):
    tail, pool, managers = hero_like(runtime)
    assert runtime.PLACEMENT == "swap"
    kept = {2: Block(61), 3: Block(62)}
    for block in kept.values():
        block.block_hash, block.ref_cnt = b"s", 1
    req = finished_request(pool, managers, list(range(25)), 21, kept=kept)
    tail.register(req)
    for mgr in managers:
        blocks = mgr.pop_blocks_for_free("r0")
        if mgr.kv_cache_group_id in kept:
            blocks = [kept[mgr.kv_cache_group_id]] + blocks
        pool.free_blocks(reversed(blocks))
    tail.after_free()
    queue = pool.free_block_queue.items
    assert queue[:2] == [kept[2], kept[3]] or queue[:2] == [kept[3], kept[2]]
    entry_blocks = [block for _, block in tail.entries[req.block_hashes[1]][0].keys.values()]
    assert all(queue.index(b) > 1 for b in entry_blocks) and tail.stats["demoted_blocks"] == 2
    # The entry is the newest: evicted only after the request's attention blocks it needs.
    assert set(queue[-len(entry_blocks):]) == set(entry_blocks)


def test_index_disables_itself_on_unsupported_layouts(runtime):
    pool = Pool(B)
    managers = [FullAttentionManager(0, B, pool), type("SlidingWindowManager", (Manager,), {})(1, B, pool),
                MambaManager(2, B, pool)]
    manager = NS(enable_caching=True, block_pool=pool, use_eagle=True,
                 coordinator=NS(single_type_managers=managers), kv_cache_config=NS(kv_cache_groups=[None] * 3))
    tail = runtime.TailIndex.create(manager)
    assert "SlidingWindowManager" in tail.disabled and "speculative" in tail.disabled
    tail.register(Request("r", list(range(25)), 21))
    assert tail.stats["registered"] == 0 and tail.lookup(Request("n", list(range(30)), 0), ([], [], []), 16, 29)[1] == 16
    assert runtime.TailIndex.create(NS(enable_caching=False)) is None


# ----------------------------------------------------------------------------------------------- the patch
def kv_manager_source():
    return ('class KVCacheManager:\n    def __init__(self):\n' + patch.TAIL_INIT_ANCHOR
            + '\n    def get_computed_blocks(self, request):\n        max_cache_hit_length = 1\n'
            '        computed_blocks, num_new_computed_tokens, shared_prefix_boundary = (), 0, 0\n'
            + patch.TAIL_LOOKUP_ANCHOR
            + '\n    def free(self, request):\n        pins = None\n        if pins:\n' + patch.TAIL_FREE_ANCHOR
            + '\n    def pop_blocks_for_free(self, request):\n' + patch.TAIL_POP_ANCHOR
            + '        return blocks\n').encode()


def test_tail_patch_is_hash_guarded_and_hooks_init_lookup_and_free(tmp_path, monkeypatch):
    source = kv_manager_source()
    with pytest.raises(RuntimeError, match='unrecognized pinned kv_cache_manager'):
        patch.patched_kv_manager(source)
    monkeypatch.setattr(patch, 'KV_MANAGER_DIAGNOSTICS_SHA256', hashlib.sha256(source).hexdigest())
    text = patch.patched_kv_manager(source).decode()
    assert text.count('self._arc_tail.register(request)') == 2 and 'self._arc_tail.after_free()' in text
    assert 'self._arc_tail = _ArcTailIndex.create(self)' in text and 'self._arc_tail.lookup(' in text
    # apply(): helper copied, the files the index relies on guarded, idempotent, re-applied after a restart.
    target = tmp_path / patch.KV_MANAGER_RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    pool = tmp_path / patch.POOL_RELATIVE
    manager = tmp_path / patch.MANAGER_RELATIVE
    pool.write_bytes(b'pool\n')
    manager.write_bytes(b'manager\n')
    options = {k: False for k in V2} | {'request_end_tail': True}
    with pytest.raises(RuntimeError, match='block_pool.py'):
        patch.apply(tmp_path, tmp_path, options)
    monkeypatch.setattr(patch, 'POOL_SHA256', hashlib.sha256(b'pool\n').hexdigest())
    monkeypatch.setattr(patch, 'MANAGER_SHA256', hashlib.sha256(b'manager\n').hexdigest())
    first = patch.apply(tmp_path, tmp_path, options)
    helper = (ROOT / 'src/arena/prefix_tail_runtime.py').read_bytes()
    assert (tmp_path / patch.TAIL_HELPER_RELATIVE).read_bytes() == helper
    assert first['tail_helper_sha256'] == hashlib.sha256(helper).hexdigest() and 'kv_cache_manager' in first
    assert patch.apply(tmp_path, tmp_path, options) == first
    target.write_bytes(source)
    assert patch.apply(tmp_path, tmp_path, options) == first


def test_options_keep_v2_identities_and_validate_the_optional_key():
    assert patch.validate(dict(V2)) == V2  # v2 studies: no new key
    assert patch.validate({**V2, 'request_end_tail': False}) == V2
    assert patch.validate({**V2, 'request_end_tail': True}) == {**V2, 'request_end_tail': True}
    for bad in ({**V2, 'request_end_tail': 1}, {**V2, 'other': True}, {'request_end_tail': True}):
        with pytest.raises(ValueError):
            patch.validate(bad)
    assert serving.lab_overrides({'ARENA_PREFIX_CACHE': json.dumps({**V2, 'request_end_tail': True})}) == {
        'prefix_cache': {**V2, 'request_end_tail': True}}


# ----------------------------------------------------------------------------------------------- keep-alive
def test_try_acquire_keeps_a_slot_and_a_full_request_free_and_logs_apart():
    events = []
    gate = gate_module.KVGate(100, slots=3, log=events.append)
    assert gate.try_acquire("g", 30, headroom_tokens=60) is not None  # 30 + 60 <= 100, inflight 1 < 3 - 0
    assert gate.try_acquire("h", 30, headroom_tokens=60) is None  # 60 + 60 > 100
    reservation = gate.try_acquire("h", 5, headroom_tokens=60)
    assert reservation is not None and gate.inflight == 2
    assert gate.try_acquire("i", 1) is None  # the last slot stays free
    gate.release_background(reservation)
    assert gate.reserved == 30 and gate.inflight == 1
    assert [e["event"] for e in events] == ["background_admit", "background_admit", "background_release"]
    gate.queue.append((None, "x", 1))
    assert gate.try_acquire("j", 1) is None  # never ahead of a waiting request


def test_keep_alive_settings():
    assert keep_alive.settings(None) is None and keep_alive.settings(False) is None
    assert keep_alive.settings(True) == {k: (float(v) if k != "games" else v) for k, v in keep_alive.DEFAULTS.items()}
    assert keep_alive.settings({"games": 5})["games"] == 5
    for bad in ({"games": 0}, {"every_seconds": 300.0}, {"other": 1}, "on"):
        with pytest.raises(ValueError):
            keep_alive.settings(bad)


class Agent:
    def __init__(self, handler, touched):
        self.cache_ping_request = ({"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 900,
                                    "tools": [1]}, 1000)
        self.cache_touched = touched
        self.client = NS(client=httpx.AsyncClient(base_url="http://model/v1/",
                                                  transport=httpx.MockTransport(handler)))


async def test_keep_alive_pings_the_first_parked_games_in_backlog_order_once_per_period():
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        cached = 10 if body["messages"][0]["content"] == "evicted" else 900
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1000,
                                                   "prompt_tokens_details": {"cached_tokens": cached}}})

    clock = [1000.0]
    games = [NS(game_id=f"g{i}", state="parked" if i != 1 else "active", admissions=[]) for i in range(5)]
    scheduler = NS(games=games, backlog=lambda: [games[i] for i in (4, 3, 1, 2, 0)])
    gate = gate_module.KVGate(10_000_000, slots=10)
    agents = {g.game_id: Agent(handler, touched=clock[0] - 150) for g in games}
    agents["g0"].cache_touched = clock[0] - 300  # past max_age: presumed evicted, never pinged
    agents["g3"].cache_ping_request[0]["messages"][0]["content"] = "evicted"
    spec = keep_alive.settings({"games": 3})
    keeper = keep_alive.KeepAlive(scheduler, gate, agents, spec, window=1000, log=None, now=lambda: clock[0])
    keeper.tick()  # targets: g4, g3, g2 (g1 is active); all touched 150 s ago
    await asyncio.gather(*keeper.inflight.values())
    assert len(sent) == 3 and all(b["max_tokens"] == 1 and b["tools"] == [1] for b in sent)
    assert gate.reserved == 0 and gate.inflight == 0
    assert keeper.evicted == {"g3"} and keeper.stats["found_evicted"] == 1
    assert agents["g4"].cache_touched == clock[0]
    keeper.tick()  # refreshed just now: nothing is due
    assert not keeper.inflight and len(sent) == 3
    clock[0] += 130
    keeper.tick()
    await asyncio.gather(*keeper.inflight.values())
    assert len(sent) == 5 and keeper.stats["pings"] == 5  # g4 and g2 again; g3 stays off until re-admitted
    games[3].admissions.append({"event": "admit"})
    agents["g3"].cache_touched = clock[0] - 125
    keeper.tick()
    await asyncio.gather(*keeper.inflight.values())
    assert len(sent) == 6 and keeper.summary()["settings"]["games"] == 3
    for agent in agents.values():
        await agent.client.client.aclose()


def test_keep_alive_refuses_lanes():
    with pytest.raises(ValueError, match="plain FIFO gate"):
        keep_alive.KeepAlive(NS(), NS(lane=object()), {}, keep_alive.settings(True), window=1)


# ----------------------------------------------------------------------------------------------- tail check / replay
def test_replay_classifies_request_end_hits():
    previous = {"prompt_token_ids": list(range(19)), "token_ids": [50, 51, 52, 53]}  # ends at 23
    record = {"prompt_token_ids": list(range(19)) + [50, 51, 52, 53, 60, 61], "cached_tokens": 23}
    out = prefix_replay.classify(dict(record), previous, 8)
    assert out["ideal_hit"] == 16 and out["ideal_hit_tail"] == 23 and out["tail_end"] == 23
    early = prefix_replay.classify({**record, "cached_tokens": 22}, previous, 8)  # the stop token not computed
    rows = [out, early, prefix_replay.classify({**record, "cached_tokens": 16}, previous, 8)]
    summary = prefix_replay.summarize_tail(rows)
    assert summary["hits_at_request_end"] == 2 and summary["hits_at_request_end_including_stop_token"] == 1
    assert summary["extra_over_block_ideal_tokens"] == 7 + 6 and summary["ideal_extra_over_block_tokens"] == 21
    assert prefix_replay.summarize_replay(rows, 8)["tail_hits"] == summary
    diverged = prefix_replay.classify({**record, "prompt_token_ids": list(range(19)) + [50, 9, 9, 9, 9, 9]},
                                      previous, 8)
    assert "ideal_hit_tail" not in diverged


class TailEngine:
    """Completions with a request-end cache: every prompt's whole blocks, and its end plus the one sampled token (a
    logit bias of +100 forces the given token, -100 moves off it). ``broken`` serves the tail entries from a
    state short of the entry (a different distribution), as a wrong-state bug would."""

    def __init__(self, block, broken=False):
        self.block, self.broken, self.cache = block, broken, set()

    def dist(self, position, shift=0):
        return {str((position * 7 + k + shift) % 97): -0.5 - k for k in range(5)}

    async def handler(self, request):
        path = request.url.path
        if path.endswith("/reset_prefix_cache"):
            self.cache.clear()
            return httpx.Response(200, json={"success": True})
        if path.endswith("/metrics"):
            return httpx.Response(200, text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n")
        body = json.loads(request.content)
        ids = body["prompt"]
        if body.get("prompt_logprobs"):
            entries = [None] + [{k: {"logprob": v} for k, v in self.dist(i).items()} for i in range(1, len(ids))]
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": entries}], "usage": {}})
        cached = max([c for c in range(1, len(ids)) if tuple(ids[:c]) in self.cache], default=0)
        for c in range(self.block, len(ids) + 1, self.block):
            self.cache.add(tuple(ids[:c]))
        bias = body.get("logit_bias") or {}
        token, value = next(iter(bias.items())) if bias else ("7", -100)
        sampled = int(token) if value > 0 else int(token) + 1
        self.cache.add(tuple(ids + [sampled]))
        shift = 1 if self.broken and cached % self.block else 0
        top = [{f"token_id:{k}": v for k, v in self.dist(len(ids), shift).items()}] if body.get("logprobs") else []
        return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": top}}],
                                         "usage": {"prompt_tokens": len(ids),
                                                   "prompt_tokens_details": {"cached_tokens": cached}}})


@pytest.mark.parametrize("broken", [False, True])
async def test_tail_check_verdict_against_a_fake_engine(monkeypatch, broken):
    block = 256
    engine = TailEngine(block, broken)
    transport = httpx.MockTransport(engine.handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": transport}))
    sequences = [[(i * 31 + s) % 1000 for i in range(7 * block + 65)] for s in (1, 2)]
    async with real(transport=transport) as client:
        result = await prefix_replay.tail_check(client, "http://model/v1", model="m", sequences=sequences,
                                                spec={"block": block, "probes": 6}, log=lambda *a, **k: None)
    summary = result["summary"]
    primers = {k: v["primer"] for k, v in result["conditions"].items()}
    assert [primers[k] % 4 for k in ("T1", "T2", "T3")] == [1, 2, 3]
    assert summary["hits"]["F6"]["values"] == [6 * block] and summary["hits"]["F5"]["values"] == [5 * block]
    assert summary["hits"]["T1"]["values"] == [primers["T1"] + 1]  # the forced token is part of the entry
    assert summary["verdict"]["hits_exact"] is True
    assert summary["verdict"]["pass"] is (not broken)


# ----------------------------------------------------------------------------------------------- hero-v2 and bundle
def test_hero_v2_components_are_off_by_default_and_validated():
    from arena import hero_v2

    assert hero_v2.DEFAULT_COMPONENTS["request_end_tail"] is False and hero_v2.DEFAULT_COMPONENTS["keep_alive"] is False
    assert hero_v2.status()["request_end_tail"]["available"] and hero_v2.status()["keep_alive"]["available"]
    with pytest.raises(ValueError, match="needs prefix_cache"):
        hero_v2.normalize({"request_end_tail": True})
    with pytest.raises(ValueError):
        hero_v2.normalize({"keep_alive": {"games": 0}})
    assert hero_v2.normalize({"prefix_cache": True, "request_end_tail": True, "keep_alive": True})["keep_alive"]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_arena_v3", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_smoke_study_timeline_and_modal_bound(tmp_path):
    builder = load_builder()
    kernels = {"kv_cache_4bit": "int4", "topk_first": True}
    hero = {"hero_v2": {"requested": {}}, "server": {"mtp_tokens": 0, "serving_kernels": kernels},
            "cache_experiment": {"prefix_cache": {**V2, "request_end_tail": True}}, "wall_seconds": 7200}
    transcript = tmp_path / "t.txt"
    transcript.write_text("line\n" * 100)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"teacher_forced": [{"ids": list(range(30000)), "source": "a"},
                                                      {"ids": list(range(30000)), "source": "b"}]}))
    (tmp_path / "out").mkdir()
    import arc_infra.serving_kernels_patch as skp

    validated = []
    original = skp.validate
    skp.validate = lambda spec: validated.append(spec) or spec
    try:
        study = builder.prefix_cache_v3_smoke_study(hero, tmp_path / "out", str(transcript), str(prompts))
        with pytest.raises(ValueError, match="request_end_tail"):
            builder.prefix_cache_v3_smoke_study({**hero, "cache_experiment": {"prefix_cache": V2}}, tmp_path / "out",
                                                str(transcript), str(prompts))
    finally:
        skp.validate = original
    assert [t["name"] for t in study["trials"]] == ["tail-int4", "tail-int8"]
    int8 = json.loads(study["trials"][1]["server_env"]["ARENA_SERVING_KERNELS"])
    assert int8 == {"kv_cache_8bit": "int8", "topk_first": True} and validated == [int8]
    assert all(t["server_env"]["ARENA_PROMPT_TOKENS_DETAILS"] == "1" for t in study["trials"])
    bench = study["throughput_bench"]
    assert tb.grid_points(bench) == [] and bench["tail_check"]["probes"] == 16
    assert len(json.loads((tmp_path / "out" / bench["tail_check"]["file"]).read_text())["sequences"][0]) == 7 * 3136 + 65
    plan = study["prefix_cache_v3_smoke"]["timeline"]
    assert [p["trial"] for p in plan] == ["tail-int4", "tail-int8"]
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "prefix-cache-v3-smoke")
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert timeout == study["kernel_timeout"] and (timeout + 60) * rate * 1.15 <= 3.99
    assert ns["arena_arguments"]("prefix-cache-v3-smoke") == []


def test_many_entries_share_one_parent(runtime):
    """Games (and the tail check's probes) whose requests end in the same block share its prefix hash: an older
    entry must stay reachable (the v3 smoke kept 4 per parent, and the primer's entry was pushed out)."""
    tail, pool, managers = hero_like(runtime)
    base = list(range(300, 316))
    for n in range(10):
        req = finished_request(pool, managers, base + [1000 + n] * 9, 21, rid=f"r{n}")
        tail.register(req)
    first = Request("n0", base + [1000] * 9 + [5, 6], 0, finished=False)
    pinned = ([], [], [pool.null_block, Block(77)], [pool.null_block, Block(78)])
    assert runtime.MAX_PER_PARENT >= 110 and tail.lookup(first, pinned, 16, 26)[1] == 21


def test_v3_recheck_is_the_int4_tail_check_alone_within_its_bound(tmp_path):
    builder = load_builder()
    hero = {"hero_v2": {"requested": {}}, "server": {"mtp_tokens": 0, "serving_kernels": {"kv_cache_4bit": "int4"}},
            "cache_experiment": {"prefix_cache": {**V2, "request_end_tail": True}}, "wall_seconds": 7200}
    transcript = tmp_path / "t.txt"
    transcript.write_text("line\n" * 100)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"teacher_forced": [{"ids": list(range(30000))}, {"ids": list(range(30000))}]}))
    (tmp_path / "out").mkdir()
    import arc_infra.serving_kernels_patch as skp

    original, skp.validate = skp.validate, (lambda spec: spec)
    try:
        study = builder.prefix_cache_v3_smoke_study(hero, tmp_path / "out", str(transcript), str(prompts), recheck=True)
    finally:
        skp.validate = original
    assert [t["name"] for t in study["trials"]] == ["tail-int4"]
    assert "prefix_replay" not in study["throughput_bench"] and study["throughput_bench"]["tail_check"]
    assert [p["trial"] for p in study["prefix_cache_v3_smoke"]["timeline"]] == ["tail-int4"]
    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "prefix-cache-v3-recheck")
    rate = 0.000842 + 8 * 0.0000131 + 192 * 0.00000222
    assert timeout == study["kernel_timeout"] and (timeout + 60) * rate * 1.15 <= 2.5
    assert ns["arena_arguments"]("prefix-cache-v3-recheck") == ["--trial", "tail-int4"]


# ------------------------------------------------------------------------- MTP-k request-end tails (SPEC_TAIL)
def spec_like(runtime, monkeypatch, spec=1):
    monkeypatch.setattr(runtime, "SPEC_TAIL", True)
    pool = Pool(B)
    managers = [FullAttentionManager(0, B, pool), CircularBufferManager(1, 8, pool),
                MambaManager(2, B, pool), MambaManager(3, B, pool)]
    for mgr in managers[2:]:
        mgr.num_speculative_blocks = spec
    manager = NS(enable_caching=True, block_pool=pool, use_eagle=True,
                 coordinator=NS(single_type_managers=managers), kv_cache_config=NS(kv_cache_groups=[None] * 4))
    return runtime.TailIndex.create(manager), pool, managers


def spec_request(pool, managers, tokens, computed, prompt, rid="s0", spec=1):
    """A finished MTP-k request: attention to cdiv(computed, B), Mamba running column + k speculative columns."""
    req = Request(rid, tokens, computed)
    req.num_prompt_tokens, req.num_output_tokens = prompt, len(tokens) - prompt
    blocks = -(-computed // B)
    attention = [Block(10 + i) for i in range(blocks)]
    for i in range(min(blocks, len(req.block_hashes))):
        if (i + 1) * B <= computed:
            pool._insert_block_hash(req.block_hashes[i] + (0).to_bytes(4, "big"), attention[i])
    managers[0].req_to_blocks[rid] = attention
    managers[1].req_to_blocks[rid] = [Block(30)]
    for gid, base in ((2, 40), (3, 50)):
        managers[gid].req_to_blocks[rid] = ([pool.null_block] * (blocks - 1)
                                            + [Block(base + i) for i in range(1 + spec)])
    for mgr in managers:
        for block in mgr.req_to_blocks[rid]:
            if not block.is_null:
                block.ref_cnt = 1
    return req


def test_spec_tail_is_off_by_default_and_disables_under_mtp(runtime):
    assert runtime.SPEC_TAIL is False
    pool = Pool(B)
    managers = [FullAttentionManager(0, B, pool), CircularBufferManager(1, 8, pool), MambaManager(2, B, pool)]
    managers[2].num_speculative_blocks = 1
    manager = NS(enable_caching=True, block_pool=pool, use_eagle=True,
                 coordinator=NS(single_type_managers=managers), kv_cache_config=NS(kv_cache_groups=[None] * 3))
    tail = runtime.TailIndex.create(manager)
    assert "speculative" in tail.disabled and tail.spec == 0


def test_spec_tail_registers_in_flight_and_prefill_only_finishes(runtime, monkeypatch):
    tail, pool, managers = spec_like(runtime, monkeypatch)
    assert tail.disabled is None and tail.spec == 1 and tail.mamba == [2, 3]
    tokens = list(range(100, 125))  # 25 tokens, the reply from 10: an in-flight verify step computed 26
    req = spec_request(pool, managers, tokens, 26, 10)
    tail.register(req)
    assert tail.stats["spec_inflight"] == 1 and tail.stats["registered"] == 1
    (entry,) = tail.entries[req.block_hashes[2]]
    assert entry.length == 25 and entry.tokens == tokens[24:25]
    # The running column (index 3), not the speculative one after it.
    assert entry.keys[2][1] is managers[2].req_to_blocks["s0"][3]
    # Prefill-only (one output token, computed = prompt): the entry ends at the prompt.
    req = spec_request(pool, managers, tokens[:22], 21, 21, "s1")
    tail.register(req)
    assert tail.stats["spec_prefill_only"] == 1 and tail.stats["registered"] == 2
    assert tail.entries[req.block_hashes[1]][0].length == 21


def test_spec_tail_skips_unknown_acceptance_and_column_crossings(runtime, monkeypatch):
    tail, pool, managers = spec_like(runtime, monkeypatch)
    tokens = list(range(100, 125))
    tail.register(spec_request(pool, managers, tokens, 25, 10, "a"))  # no in-flight step: acceptance unknown
    assert tail.stats["skip_spec_unknown_acceptance"] == 1
    # computed 25 + 1 = 26 is in column 3 but a 24-token end with an in-flight step crosses into a new column.
    tail.register(spec_request(pool, managers, tokens[:24], 25, 10, "b"))
    assert tail.stats["spec_inflight"] == 1 and tail.stats["skip_spec_column"] == 1
    assert tail.stats["registered"] == 0


def test_spec_decode_flips_the_tail_helper_flag(tmp_path, monkeypatch):
    source = kv_manager_source()
    monkeypatch.setattr(patch, 'KV_MANAGER_DIAGNOSTICS_SHA256', hashlib.sha256(source).hexdigest())
    target = tmp_path / patch.KV_MANAGER_RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    (tmp_path / patch.POOL_RELATIVE).write_bytes(b'pool\n')
    (tmp_path / patch.MANAGER_RELATIVE).write_bytes(b'manager\n')
    monkeypatch.setattr(patch, 'POOL_SHA256', hashlib.sha256(b'pool\n').hexdigest())
    monkeypatch.setattr(patch, 'MANAGER_SHA256', hashlib.sha256(b'manager\n').hexdigest())
    helper = (ROOT / 'src/arena/prefix_tail_runtime.py').read_bytes()
    options = {k: False for k in V2} | {'request_end_tail': True}
    patch.apply(tmp_path, tmp_path, options)
    assert (tmp_path / patch.TAIL_HELPER_RELATIVE).read_bytes() == helper
    written = helper.replace(b'SPEC_TAIL = False\n', b'SPEC_TAIL = True\n')
    assert written != helper
    coordinator = tmp_path / patch.COORDINATOR_RELATIVE
    coordinator.parent.mkdir(parents=True, exist_ok=True)
    coordinator.write_bytes(b'coordinator\n')
    monkeypatch.setattr(patch, 'COORDINATOR_SHA256', hashlib.sha256(b'coordinator\n').hexdigest())
    monkeypatch.setattr(patch, 'patched_coordinator', lambda src: src + b'# patched\n')
    target.write_bytes(source)
    out = tmp_path / 'spec'
    out.mkdir()
    identity = patch.apply(tmp_path, out, options | {'spec_decode': True})
    assert (tmp_path / patch.TAIL_HELPER_RELATIVE).read_bytes() == written
    assert identity['tail_helper_sha256'] == hashlib.sha256(written).hexdigest()


def test_mtp_tail_check_summary_applies_the_v3_rule():
    def stats(kl, top1):
        return {"positions": 10, "mean_kl": kl, "top1_agreement": top1, "mean_abs_delta_logprob": 0.1}

    row = {"conditions": {"D": {"cached_tokens": [5] * 3, "hit_as_expected": 3},
                          "T": {"cached_tokens": [6, 5, 1], "hit_as_expected": 2}},
           "pairs": {"F6|F5": stats(0.06, 0.95), "F6|D": stats(0.09, 0.91), "F6|T": stats(0.2, 0.95)},
           "finish_reason": "stop"}
    summary = prefix_replay.summarize_mtp_tail_check({"sequences": [row]})
    assert summary["verdict"]["D"] == {"hits": True, "kl_ok": True, "top1_ok": True, "pass": True}
    assert summary["verdict"]["T"]["pass"] is False and not summary["pass"] and summary["stop_finishes"] == 1


def test_collapse_report_lists_cached_boundaries_per_group_once(runtime, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "COLLAPSE_LOG", True)
    tail, pool, managers = hero_like(runtime)
    pool.get_num_free_blocks = lambda: 7
    tokens = list(range(300, 380))  # 80 tokens = 10 blocks of 8
    req = Request("c0", tokens, 0, finished=False)
    req.num_prompt_tokens = len(tokens)
    for j in (1, 2, 3):  # attention cached for blocks 1-3 only
        pool._insert_block_hash(req.block_hashes[j - 1] + (0).to_bytes(4, "big"), Block(100 + j))
    for gid in (2, 3):  # the Mamba groups hold the state at boundary 1 and 9 only
        for j in (1, 9):
            pool._insert_block_hash(req.block_hashes[j - 1] + gid.to_bytes(4, "big"), Block(200 + 10 * gid + j))
    tail.lookup(req, ([], [], [], []), 8, 79)
    tail.lookup(req, ([], [], [], []), 8, 79)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("ARC_TAIL_COLLAPSE ")]
    assert len(lines) == 1
    report = json.loads(lines[0].split(" ", 1)[1])
    assert report["groups"] == {"attn0": {"run": 3, "count": 3}, "mamba2": [1, 9], "mamba3": [1, 9]}
    assert report["hit"] == 8 and report["free_blocks"] == 7 and report["boundaries"] == 9
    monkeypatch.setattr(runtime, "COLLAPSE_LOG", False)
    tail.lookup(Request("c1", tokens, 0, finished=False), ([], [], [], []), 8, 79)
    assert "ARC_TAIL_COLLAPSE" not in capsys.readouterr().out


def test_back_placement_keeps_the_kept_boundary_state_as_a_fallback(runtime, monkeypatch):
    """Phase 4: under "swap" a registered finish sends its kept boundary state to the front of the free queue (the next
    allocation evicts it), so a game whose tail entry is then lost resumes from the shared prefix only; under
    "back" (tail_keep_boundary) the kept state stays at the MRU end with the request's other blocks."""
    def run(placement):
        monkeypatch.setattr(runtime, "PLACEMENT", placement)
        tail, pool, managers = hero_like(runtime)
        kept = {2: Block(61), 3: Block(62)}
        for block in kept.values():
            block.block_hash, block.ref_cnt = b"s", 1
        older = Block(70)
        older.block_hash = b"o"
        pool.free_block_queue.append_n([older])  # another game's cached block, older than this request
        req = finished_request(pool, managers, list(range(25)), 21, kept=kept)
        tail.register(req)
        for mgr in managers:
            blocks = mgr.pop_blocks_for_free("r0")
            if mgr.kv_cache_group_id in kept:
                blocks = [kept[mgr.kv_cache_group_id]] + blocks
            pool.free_blocks(reversed(blocks))
        tail.after_free()
        return pool.free_block_queue.items, kept, older

    queue, kept, older = run("swap")
    assert set(queue[:2]) == set(kept.values())  # evicted before the older game's block
    queue, kept, older = run("back")
    assert queue[0] is older and all(queue.index(b) > 0 for b in kept.values())


def test_tail_keep_boundary_needs_the_tail_and_switches_the_helper_placement(tmp_path, monkeypatch):
    base = {k: True for k in V2}
    with pytest.raises(ValueError, match="needs request_end_tail"):
        patch.validate(base | {"tail_keep_boundary": True})
    assert patch.validate(base | {"request_end_tail": True, "tail_keep_boundary": False}) == base | {"request_end_tail": True}
    source = kv_manager_source()
    monkeypatch.setattr(patch, 'KV_MANAGER_DIAGNOSTICS_SHA256', hashlib.sha256(source).hexdigest())
    target = tmp_path / patch.KV_MANAGER_RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    (tmp_path / patch.POOL_RELATIVE).write_bytes(b'pool\n')
    (tmp_path / patch.MANAGER_RELATIVE).write_bytes(b'manager\n')
    monkeypatch.setattr(patch, 'POOL_SHA256', hashlib.sha256(b'pool\n').hexdigest())
    monkeypatch.setattr(patch, 'MANAGER_SHA256', hashlib.sha256(b'manager\n').hexdigest())
    options = {k: False for k in V2} | {'request_end_tail': True, 'tail_keep_boundary': True}
    patch.apply(tmp_path, tmp_path, options)
    written = (tmp_path / patch.TAIL_HELPER_RELATIVE).read_bytes()
    assert b'PLACEMENT = "back"\n' in written and b'PLACEMENT = "swap"\n' not in written
    helper = (ROOT / 'src/arena/prefix_tail_runtime.py').read_bytes()
    assert written == helper.replace(b'PLACEMENT = "swap"\n', b'PLACEMENT = "back"\n')


def test_invalid_report_names_the_lost_blocks(runtime, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "COLLAPSE_LOG", True)
    tail, pool, managers = hero_like(runtime)
    pool.get_num_free_blocks = lambda: 5
    tokens = list(range(200, 230))
    req = finished_request(pool, managers, tokens[:25], 21)
    tail.register(req)
    for mgr in managers:
        pool.free_blocks(reversed(mgr.pop_blocks_for_free("r0")))
    (entry,) = tail.entries[req.block_hashes[1]]
    pool.evict(entry.keys[3][1])
    pinned = ([], [], [pool.null_block, Block(77)], [pool.null_block, Block(78)])
    assert tail.lookup(Request("n0", tokens, 0, finished=False), pinned, 16, 29)[1] == 16
    (line,) = [x for x in capsys.readouterr().out.splitlines() if x.startswith("ARC_TAIL_INVALID ")]
    report = json.loads(line.split(" ", 1)[1])
    assert report["length"] == 21 and set(report["missing"]) == {"3"} and report["free_blocks"] == 5
    assert report["age_seconds"] is not None and report["missing"]["3"]["ref_cnt"] == 0


def test_collapse_report_follows_the_previous_finish_kept_state(runtime, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "COLLAPSE_LOG", True)
    tail, pool, managers = hero_like(runtime)
    pool.get_num_free_blocks = lambda: 9
    tokens = list(range(400, 480))
    probe = Request("x", tokens, 0)
    kept = {}
    for gid in (2, 3):
        block = Block(60 + gid)
        block.ref_cnt, block.block_hash_num_tokens = 1, 16
        pool._insert_block_hash(probe.block_hashes[1] + gid.to_bytes(4, "big"), block)
        kept[gid] = block
    req = finished_request(pool, managers, tokens[:25], 21, kept=kept)
    tail.register(req)
    pool.evict(kept[3])  # one group's kept state is lost before the game's next request
    resent = tokens[:18] + [9999] + tokens[19:]  # and the request-end entry does not match (reported after it is tried)
    new = Request("n", resent, 0, finished=False)
    new.num_prompt_tokens = len(tokens)
    tail.lookup(new, ([], [], [], []), 8, 79)
    (line,) = [x for x in capsys.readouterr().out.splitlines() if x.startswith("ARC_TAIL_COLLAPSE ")]
    finish = json.loads(line.split(" ", 1)[1])["finish"]
    assert finish["boundary"] == 2 and finish["computed"] == 21
    assert finish["groups"]["2"]["cached"] and finish["groups"]["2"]["same_hash"]
    assert not finish["groups"]["3"]["cached"] and not finish["groups"]["3"]["same_hash"]


def test_collapse_report_says_where_the_resent_reply_differs(runtime, monkeypatch, capsys):
    """The reply as generated is kept by its prompt's last boundary; a collapse names the first re-sent token id that
    differs (a re-tokenized reply), with the ids around it."""
    monkeypatch.setattr(runtime, "COLLAPSE_LOG", True)
    tail, pool, managers = hero_like(runtime)
    pool.get_num_free_blocks = lambda: 9
    prompt = list(range(400, 420))  # 20 tokens: last boundary at 16
    generated = [900, 901, 902, 903, 904, 905]
    req = finished_request(pool, managers, prompt + generated, 25)
    req.num_prompt_tokens = len(prompt)
    tail.register(req)
    resent = prompt + [900, 901, 950, 904, 905] + list(range(600, 660))  # two tokens merged into one
    new = Request("n", resent, 0, finished=False)
    new.num_prompt_tokens = len(resent)
    tail.lookup(new, ([], [], [], []), 8, 79)
    (line,) = [x for x in capsys.readouterr().out.splitlines() if x.startswith("ARC_TAIL_COLLAPSE ")]
    resend = json.loads(line.split(" ", 1)[1])["resend"]
    assert resend["prev_prompt"] == 20 and resend["first_diff"] == 2 and resend["prompt_tail_same"]
    assert resend["generated_ids"] == generated and resend["resent_ids"][:4] == [900, 901, 950, 904]
    same = Request("m", prompt + generated + list(range(600, 660)), 0, finished=False)
    same.num_prompt_tokens = len(same.all_token_ids)
    assert tail.lookup(same, ([], [], [], []), 8, 79)[1] == 25
    # Served by the request-end entry: not a collapse (the report follows the entry lookup).
    assert not [x for x in capsys.readouterr().out.splitlines() if x.startswith("ARC_TAIL_COLLAPSE ")]


@pytest.mark.parametrize("accepted, slot", [(0, 3), (1, 4)])
def test_spec_accept_registers_a_finish_without_an_in_flight_step(runtime, monkeypatch, accepted, slot):
    """tail_spec_acceptance: computed == num_tokens - 1; the state after the accepted tokens is slot
    (L - a) // B + a of the last step's state slots (the running column or the speculative one)."""
    tail, pool, managers = spec_like(runtime, monkeypatch)
    tokens = list(range(100, 126))  # 26 tokens; the last sampled one (index 25) was never processed
    req = spec_request(pool, managers, tokens, 25, 10)
    req._arc_last_accepted = accepted
    tail.register(req)
    assert tail.stats["skip_spec_unknown_acceptance"] == 1  # SPEC_ACCEPT off: unchanged behaviour
    monkeypatch.setattr(runtime, "SPEC_ACCEPT", True)
    req = spec_request(pool, managers, tokens, 25, 10, "s1")  # state slots 3 (running) and 4 (speculative)
    req._arc_last_accepted = accepted
    tail.register(req)
    if accepted == 0:
        assert tail.stats["spec_accepted"] == 1 and tail.stats["registered"] == 1
        (entry,) = tail.entries[req.block_hashes[2]]
        assert entry.length == 25 and entry.keys[2][1] is managers[2].req_to_blocks["s1"][slot]
    else:
        # 197d56c: accepted > 0 leaves the conv states outside slot `accepted` (another sequence's window): skipped,
        # as is a stop on an accepted draft (computed == num_tokens).
        assert tail.stats["spec_accepted"] == 0 and tail.stats["registered"] == 0
        stop = spec_request(pool, managers, tokens[:25], 25, 10, "s3")
        stop._arc_last_accepted = 1
        tail.register(stop)
        assert tail.stats["registered"] == 0
    # An acceptance outside 0..k or a missing attribute is skipped.
    bad = spec_request(pool, managers, tokens, 25, 10, "s2")
    bad._arc_last_accepted = 2
    before = tail.stats["skip_spec_unknown_acceptance"]
    tail.register(bad)
    assert tail.stats["skip_spec_unknown_acceptance"] == before + 1


def test_tail_spec_acceptance_patches_the_scheduler_and_the_helper_flag():
    base = {"mamba_align_fix": True, "mamba_keep_state": True, "mamba_demote_superseded": True}
    with pytest.raises(ValueError, match="tail_spec_acceptance needs"):
        patch.validate(base | {"tail_spec_acceptance": True, "request_end_tail": True})
    assert patch.validate(base | {"request_end_tail": True, "spec_decode": True, "tail_spec_acceptance": True})
    text = patch.ACCEPT_REPLACEMENT
    assert "request._arc_last_accepted = num_accepted" in text and text.endswith(patch.ACCEPT_ANCHOR.split("\n", 1)[1])


def test_spec_accept_registers_the_dropped_bonus_after_a_stop_on_the_last_draft(runtime, monkeypatch):
    """computed == num_tokens + k + 1: the stop was the accepted draft, the in-flight step processed the dropped
    bonus b; the entry ends at num_tokens + 1 with b, at the in-flight step's running slot (computed - 1) // B."""
    tail, pool, managers = spec_like(runtime, monkeypatch)
    monkeypatch.setattr(runtime, "SPEC_ACCEPT", True)
    tokens = list(range(100, 125))  # 25 tokens incl. the stop; computed 27 = 25 + 1 + 1
    req = spec_request(pool, managers, tokens, 27, 10, "d0")
    req._arc_last_accepted, req._arc_dropped = 1, [7777]
    tail.register(req)
    assert tail.stats["spec_dropped_bonus"] == 1 and tail.stats["registered"] == 1
    entry = tail.entries[req.block_hashes[2]][0]
    assert entry.length == 26 and entry.tokens == tokens[24:] + [7777]
    assert entry.keys[2][1] is managers[2].req_to_blocks["d0"][(27 - 1) // B]
    # Without the dropped token it stays skipped.
    other = spec_request(pool, managers, tokens, 27, 10, "d1")
    other._arc_last_accepted = 1
    tail.register(other)
    assert tail.stats["skip_spec_unknown_acceptance"] == 1


def test_drop_patch_records_tokens_after_the_stop():
    assert "request._arc_dropped = list(new_token_ids[num_new:])" in patch.DROP_REPLACEMENT
    assert patch.DROP_REPLACEMENT.endswith(patch.DROP_ANCHOR.split("\n", 1)[1])


@pytest.mark.parametrize("n, computed, accepted, dropped", [(26, 25, 0, None), (26, 25, 1, None), (25, 25, 1, None),
                                                            (25, 26, None, None), (25, 27, 1, [7777])])
def test_spec_length_finish_registers_no_entry(runtime, monkeypatch, n, computed, accepted, dropped):
    """v7: a finish by length under MTP-k (accepted, in-flight and dropped-bonus shapes) is skipped and recomputed."""
    tail, pool, managers = spec_like(runtime, monkeypatch)
    monkeypatch.setattr(runtime, "SPEC_ACCEPT", True)
    tokens = list(range(100, 100 + n))
    for rid, status in (("len", NS(name="FINISHED_LENGTH_CAPPED")), ("eos", NS(name="FINISHED_STOPPED"))):
        req = spec_request(pool, managers, tokens, computed, 10, rid)
        req.status, req._arc_last_accepted, req._arc_dropped = status, accepted, dropped
        tail.register(req)
    assert tail.stats["skip_spec_length"] == 1
    # the same shape stopped by EOS registers, except an accepted draft without a dropped bonus (197d56c)
    assert tail.stats["registered"] == (0 if accepted == 1 and dropped is None else 1)
    monkeypatch.setattr(runtime, "SPEC_SKIP_LENGTH", False)
    req = spec_request(pool, managers, tokens, computed, 10, "len2")
    req.status, req._arc_last_accepted, req._arc_dropped = NS(name="FINISHED_LENGTH_CAPPED"), accepted, dropped
    before = tail.stats["skip_spec_length"]
    tail.register(req)
    assert tail.stats["skip_spec_length"] == before


def idle_game(runtime, monkeypatch, placement="back"):
    """A finished request (25 tokens, B = 8: three full attention blocks) freed into a queue that other games' cached
    blocks then follow; the clock is the test's."""
    monkeypatch.setattr(runtime, "KEEPALIVE", True)
    monkeypatch.setattr(runtime, "PLACEMENT", placement)
    tail, pool, managers = hero_like(runtime)
    now = [100.0]
    tail.clock = lambda: now[0]
    kept = {2: Block(61), 3: Block(62)}
    tokens = list(range(400, 480))
    probe = Request("x", tokens, 0)
    for gid, block in kept.items():
        block.ref_cnt = 1
        pool._insert_block_hash(probe.block_hashes[2] + gid.to_bytes(4, "big"), block)  # the state at 3 * B
    req = finished_request(pool, managers, tokens[:29], 29, kept=kept)
    tail.register(req)
    for mgr in managers:
        blocks = mgr.pop_blocks_for_free("r0")
        if mgr.kv_cache_group_id in kept:
            blocks = [kept[mgr.kv_cache_group_id]] + blocks
        pool.free_blocks(reversed(blocks))
    tail.after_free()
    own = list(pool.free_block_queue.items)
    others = [Block(700 + i) for i in range(6)]
    for block in others:
        block.block_hash = b"other"
    pool.free_block_queue.append_n(others)  # churn: newer than the idle game's blocks
    return tail, pool, now, own, others, tokens


def test_keepalive_is_an_optional_tail_option():
    tail = {"mamba_align_fix": True, "mamba_keep_state": True, "mamba_demote_superseded": True, "request_end_tail": True}
    assert "tail_keepalive" not in patch.validate(tail)
    assert patch.validate({**tail, "tail_keepalive": True})["tail_keepalive"] is True
    with pytest.raises(ValueError):
        patch.validate({**tail, "request_end_tail": False, "tail_keepalive": True})
    helper = (ROOT / 'src/arena/prefix_tail_runtime.py').read_text()
    assert helper.count('KEEPALIVE = False\n') == 1


def test_keepalive_moves_an_idle_games_blocks_to_the_newest_end_after_five_seconds(runtime, monkeypatch):
    tail, pool, now, own, others, _ = idle_game(runtime, monkeypatch)
    queue = pool.free_block_queue.items
    cached = [b for b in own if b.block_hash is not None]
    before = list(queue)
    now[0] += 4.9
    tail.keepalive()
    assert queue == before and not tail.stats["keepalive_pings"]  # a normal tool call: nothing moves
    now[0] += 0.2
    tail.keepalive()
    assert tail.stats["keepalive_pings"] == 1 and tail.stats["keepalive_games"] == 1
    assert tail.stats["keepalive_blocks"] == len(cached) == 3 + 2 + 4  # attention, kept states, the entry's blocks
    # Every cached block of the game is now newer than the churn, in the same relative order; nothing is referenced
    # and the free count is unchanged.
    assert queue[-len(cached):] == [b for b in before if b in cached] and len(queue) == len(before)
    assert set(queue[:len(before) - len(cached)]) == set(before) - set(cached)
    assert all(b.ref_cnt == 0 for b in queue) and len(set(map(id, queue))) == len(queue)
    order = list(queue)
    now[0] += 4.0
    tail.keepalive()
    assert queue == order  # the next refresh is due 5 s after the last
    now[0] += 1.5
    pool.free_block_queue.append_n([Block(800)])
    tail.keepalive()
    assert tail.stats["keepalive_pings"] == 2 and tail.stats["keepalive_games"] == 1 and queue[-len(cached):] == order[-len(cached):]


def test_keepalive_skips_blocks_in_use_and_rekeyed_blocks(runtime, monkeypatch):
    tail, pool, now, own, others, _ = idle_game(runtime, monkeypatch)
    queue = pool.free_block_queue.items
    attention = [b for b in own if 10 <= b.block_id < 13]
    shared, lost = attention[0], attention[1]
    pool.touch([shared])  # a clone of the game is running on the shared first block
    now[0] += 6
    tail.keepalive()
    assert shared.ref_cnt == 1 and shared not in queue and tail.stats["keepalive_blocks"] == 3 + 2 + 4 - 1
    # An attention block of the prefix reused by another request: the entry cannot serve, nothing moves any more.
    pool.evict(lost)
    order = list(queue)
    now[0] += 6
    tail.keepalive()
    assert queue == order and tail.stats["keepalive_lost"] == 1 and tail.stats["keepalive_pings"] == 1
    assert not tail.waiting


def test_keepalive_stops_on_return_on_a_lost_entry_and_at_the_deadline(runtime, monkeypatch):
    tail, pool, now, own, others, tokens = idle_game(runtime, monkeypatch)
    now[0] += 6
    tail.keepalive()
    new = Request("n", tokens, 0, finished=False)
    blocks, length = tail.lookup(new, ([], [], [], []), 8, 79)  # the game comes back and resumes from the entry
    assert length == 29 and tail.stats["keepalive_returned"] == 1 and tail.waiting[0].alive is None
    now[0] += 30
    tail.keepalive()
    assert tail.stats["keepalive_pings"] == 1 and not tail.waiting

    tail, pool, now, own, others, _ = idle_game(runtime, monkeypatch)
    entry = tail.waiting[0]
    pool.evict(next(block for _, block in entry.keys.values()))
    now[0] += 6
    tail.keepalive()
    assert tail.stats["keepalive_lost"] == 1 and not tail.stats["keepalive_pings"] and not tail.waiting

    tail, pool, now, own, others, _ = idle_game(runtime, monkeypatch)
    for _ in range(12):
        now[0] += 5.01
        tail.keepalive()
    assert tail.stats["keepalive_pings"] == 8 and tail.stats["keepalive_expired"] == 1 and not tail.waiting


def test_keepalive_leaves_the_demoted_boundary_state_at_the_front_under_swap(runtime, monkeypatch):
    tail, pool, now, own, others, _ = idle_game(runtime, monkeypatch, placement="swap")
    queue = pool.free_block_queue.items
    front = queue[:2]
    assert {b.block_id for b in front} == {61, 62}
    now[0] += 6
    tail.keepalive()
    assert queue[:2] == front and tail.stats["keepalive_blocks"] == 3 + 4


def test_keepalive_off_registers_nothing(runtime, monkeypatch):
    tail, pool, managers = hero_like(runtime)
    tail.register(finished_request(pool, managers, list(range(400, 429)), 29))
    assert not runtime.KEEPALIVE and not tail.waiting
    assert next(iter(tail.entries.values()))[0].alive is None
