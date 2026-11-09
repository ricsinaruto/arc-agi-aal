"""arc_infra.arc_routed_offload (R3 through the host KV tier): a chunk restored from host RAM gets the routed rows the
request that computed it returned. A fake engine: GPU blocks recycled at once (a freed block keeps its old slot rows,
as vLLM's slot buffer does), host chunks keyed by the chained block hash (LoRA in it), every computation routes
differently (nonce), so stale or recomputed rows never pass for the saved ones."""
import collections
import hashlib
import json
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from arc_infra import served_route_patch as P
from arc_infra.arc_routed_offload import RoutedRows, cdiv

L, K = 3, 2


class Mgr:
    def __init__(self, num_blocks, block_size):
        self.attn_gid, self.block_size = 0, block_size
        self.routed_experts_by_slot = np.zeros((num_blocks * block_size, L, K), dtype=np.uint8)

    def store_batch(self, data, slot_mapping):
        self.routed_experts_by_slot[slot_mapping] = data

    def get(self, block_ids, num_tokens):
        slots = (np.asarray(block_ids)[:, None] * self.block_size + np.arange(self.block_size)).ravel()[:num_tokens]
        return self.routed_experts_by_slot[slots]


class ReqState:
    def __init__(self, keys):
        self.group_states = [NS(offload_keys=keys, block_ids=[])]
        self.num_locally_computed_tokens = 0
        self.partial_tail_boundary = None
        self.hit_chunks = None

    def update_num_hit_chunks(self, n):
        self.hit_chunks = n


class Engine:
    def __init__(self, B=4, bpc=1, num_blocks=64, cpu_chunks=1000, routed=True, rows_chunks=None):
        self.B, self.bpc, self.tpc = B, bpc, B * bpc
        self.mgr = Mgr(num_blocks, B)
        self.free = collections.deque(range(1, num_blocks))
        self.cpu, self.cpu_cap = collections.OrderedDict(), cpu_chunks
        self.sched = NS(config=NS(kv_group_configs=[NS(tokens_per_block=B, tokens_per_chunk=B * bpc)],
                                  blocks_per_chunk=bpc), _req_status={}, _jobs={})
        row_bytes = L * K
        budget = (rows_chunks if rows_chunks is not None else 10 ** 6) * self.tpc * row_bytes
        self.routed = RoutedRows(self.sched, [0], budget, cpu_chunks, log=lambda *a, **k: None) if routed else None
        if self.routed:
            self.routed.attach(self.mgr)
        self.nonce, self.jid = 0, 0

    def keys(self, tokens, lora):
        out, parent = [], b""
        for c in range(len(tokens) // self.tpc):
            parent = hashlib.sha256(parent + repr((tokens[c * self.tpc:(c + 1) * self.tpc], lora)).encode()).digest()
            out.append(parent + (0).to_bytes(4, "big"))
        return out

    def admit(self, rid, tokens, lora=None, store=True):
        """schedule(): host lookup, block allocation, update_state_after_alloc."""
        st = ReqState(self.keys(tokens, lora))
        self.sched._req_status[rid] = st
        n = 0
        while n < len(st.group_states[0].offload_keys) and st.group_states[0].offload_keys[n] in self.cpu \
                and (n + 1) * self.tpc <= len(tokens) - 1:
            n += 1
        hit = n * self.tpc
        if self.routed:
            hit = self.routed.lookup(rid, st, 0, hit)
        blocks = [self.free.popleft() for _ in range(cdiv(len(tokens), self.B))]
        st.group_states[0].block_ids = list(blocks)
        if self.routed:
            self.routed.after_alloc(rid, st, [[NS(block_id=b, block_hash=None, is_null=False) for b in blocks]], hit)
        return NS(rid=rid, tokens=list(tokens), lora=lora, blocks=blocks, computed=hit, stored=hit // self.tpc,
                  hit=hit, store=store)

    def schedule(self, reqs, budget=10 ** 9):
        """One step: positions scheduled (num_computed advances at once, as vLLM's does), then
        build_connector_meta's store jobs for the chunks complete after this step."""
        so, store_jobs = NS(work=[]), {}
        for r in reqs:
            end = min(len(r.tokens), r.computed + budget)
            so.work.append((r, r.computed, end))
            r.computed = end
            keys = self.sched._req_status[r.rid].group_states[0].offload_keys
            new = [k for k in keys[r.stored:r.computed // self.tpc] if k not in self.cpu] if r.store else []
            r.stored = r.computed // self.tpc
            for k in new:
                self.cpu[k] = True
                while len(self.cpu) > self.cpu_cap:
                    self.cpu.popitem(last=False)
            if new:
                self.jid += 1
                self.sched._jobs[self.jid] = NS(keys=set(new), req_id=r.rid, is_store=True)
                store_jobs[self.jid] = NS(req_id=r.rid)
        if self.routed:
            self.routed.tick(so)
            self.routed.on_store(store_jobs)
        return so

    def route(self, r, pos):
        seed = hashlib.sha256(repr((r.tokens[:pos + 1], r.lora, self.nonce)).encode()).digest()
        return np.frombuffer(seed[:L * K], dtype=np.uint8).reshape(L, K)

    def land(self, so):
        """update_from_output: this step's routing lands (store_batch) for the positions it computed."""
        self.nonce += 1
        rows, slots = [], []
        for r, start, end in so.work:
            for pos in range(start, end):
                rows.append(self.route(r, pos))
                slots.append(r.blocks[pos // self.B] * self.B + pos % self.B)
        if self.routed:
            self.routed.landing = so
        self.mgr.store_batch(np.array(rows, dtype=np.uint8).reshape(-1, L, K), np.array(slots, dtype=np.int64))
        if self.routed:
            self.routed.landing = None

    def run(self, r, budget=10 ** 9):
        while r.computed < len(r.tokens):
            self.land(self.schedule([r], budget))
        return self.mgr.get(r.blocks, len(r.tokens)).copy()

    def finish(self, r):
        self.free.extendleft(reversed(r.blocks))  # recycled first: a new request lands on these slots
        if self.routed:
            self.routed.finished(r.rid)


def tokens(n, seed=0):
    return [int(x) for x in np.random.default_rng(seed).integers(0, 1000, n)]


def scrub(e, n, seed=99):
    """Another request on the freed blocks: their slots now hold its rows."""
    x = e.admit("scrub", tokens(n, seed), store=False)  # max_offload_tokens 0: host RAM untouched
    e.run(x)
    e.finish(x)


def test_restore_equals_saved_after_offload_evict_restore():
    e = Engine()
    t = tokens(18)
    a = e.admit("a", t)
    rows_a = e.run(a, budget=5)  # chunked prefill: stores land over several steps
    e.finish(a)
    scrub(e, 18)
    b = e.admit("b", t)
    assert b.hit == 16  # 4 chunks from host RAM (the last token is always computed)
    rows_b = e.run(b)
    np.testing.assert_array_equal(rows_b[:16], rows_a[:16])
    assert not np.array_equal(rows_b[16:], rows_a[16:])  # recomputed positions route anew (nonce): a real check
    assert e.routed.counts["restored_tokens"] == 16


def test_rows_lru_evicted_refuses_the_host_hit():
    e = Engine(rows_chunks=4)
    t = tokens(18)
    a = e.admit("a", t)
    e.run(a)
    e.finish(a)
    other = e.admit("o", tokens(18, seed=5))  # 4 more chunks of rows push a's out; host KV keeps them
    e.run(other)
    e.finish(other)
    b = e.admit("b", t)
    assert b.hit == 0 and e.routed.counts["refused_missing"] == 1
    assert e.sched._req_status["b"].hit_chunks == 0
    rows_b = e.run(b)
    np.testing.assert_array_equal(rows_b, np.array([e.route(b, p) for p in range(18)]))  # its own computation


def test_host_evicted_then_recomputed_chunks_restore_the_newest_rows():
    e = Engine(cpu_chunks=4)
    t = tokens(18)
    a = e.admit("a", t)
    e.run(a)
    e.finish(a)
    o = e.admit("o", tokens(18, seed=5))  # evicts a's host chunks
    e.run(o)
    e.finish(o)
    a2 = e.admit("a2", t)
    assert a2.hit == 0
    rows_a2 = e.run(a2)  # stored again, rows saved again
    e.finish(a2)
    scrub(e, 18)
    b = e.admit("b", t)
    assert b.hit == 16
    np.testing.assert_array_equal(e.run(b)[:16], rows_a2[:16])


def test_partial_hit_shared_prefix():
    e = Engine(bpc=2)  # 8-token chunks of two blocks
    t = tokens(26)
    a = e.admit("a", t)
    rows_a = e.run(a, budget=7)
    e.finish(a)
    scrub(e, 26)
    b = e.admit("b", t[:17] + tokens(9, seed=3))  # shares chunks 0-1, diverges inside chunk 2
    assert b.hit == 16
    rows_b = e.run(b)
    np.testing.assert_array_equal(rows_b[:16], rows_a[:16])


def test_missing_chunk_rows_refuse_whole_hit():
    e = Engine()
    t = tokens(18)
    a = e.admit("a", t)
    e.run(a)
    e.finish(a)
    del e.routed.rows[e.keys(t, None)[2]]
    b = e.admit("b", t)
    assert b.hit == 0 and e.routed.counts["refused_missing"] == 1 and "b" not in e.routed.held


def test_lora_distinct_hash():
    e = Engine()
    t = tokens(18)
    a = e.admit("a", t, lora="p1")
    rows_a = e.run(a)
    e.finish(a)
    scrub(e, 18)
    other = e.admit("b", t, lora="p2")
    assert other.hit == 0  # another adapter's chunks are other keys: no KV, no rows
    rows_other = e.run(other)
    e.finish(other)
    assert not np.array_equal(rows_other[:16], rows_a[:16])
    scrub(e, 18, seed=98)
    c = e.admit("c", t, lora="p1")
    assert c.hit == 16
    np.testing.assert_array_equal(e.run(c)[:16], rows_a[:16])


def test_async_race_save_waits_for_the_landing():
    """Schedule S+1 runs before step S's update_from_output (async scheduling): rows for chunks stored in S are only
    in the slot buffer after S lands."""
    e = Engine()
    scrub(e, 18)  # a's blocks will hold stale rows until its own steps land
    t = tokens(18)
    a = e.admit("a", t)
    s1 = e.schedule([a])  # stores chunks 0-3 at schedule time
    assert e.routed.pending and not e.routed.rows
    x = e.admit("x", tokens(10, seed=7))
    s2 = e.schedule([x])  # S+1 scheduled before S landed
    stale = e.mgr.get(a.blocks, 16).copy()
    e.land(s1)
    assert len(e.routed.rows) == 4 and [p[0] for p in e.routed.pending] == [s2._arc_routed_seq] * 2  # x's wait
    e.land(s2)
    rows_a = e.mgr.get(a.blocks, 18).copy()
    assert not np.array_equal(stale, rows_a[:16])  # a save at schedule time would have kept these
    e.finish(a)
    e.finish(x)
    scrub(e, 30, seed=11)
    b = e.admit("b", t)
    np.testing.assert_array_equal(e.run(b)[:16], rows_a[:16])


def test_restore_survives_an_older_inflight_step_on_a_reused_block():
    """A block freed while an older step still writes it (preempted request) is handed to a restore: that step's
    landing comes after the restore's write; the restore is written again on every landing up to its own step."""
    e = Engine()
    t = tokens(18)
    a = e.admit("a", t)
    rows_a = e.run(a)
    e.finish(a)
    x = e.admit("x", tokens(18, seed=4))
    sx = e.schedule([x])  # in flight
    e.finish(x)  # preempted: blocks back to the pool before sx lands
    b = e.admit("b", t)  # restore lands on x's blocks
    assert b.hit == 16 and set(b.blocks) & set(x.blocks)
    sb = e.schedule([b])
    e.land(sx)  # x's rows hit b's restored slots ...
    e.land(sb)  # ... and the restore is written again first
    np.testing.assert_array_equal(e.mgr.get(b.blocks, 18)[:16], rows_a[:16])
    assert not e.routed.restores


def test_unbound_refuses_and_bind_checks_block_size():
    e = Engine(routed=False)
    logged = []
    r = RoutedRows(e.sched, [0], 10 ** 6, 100, log=lambda line, **k: logged.append(line))
    st = ReqState([b"k"])
    assert r.lookup("q", st, 0, 4) == 0 and r.counts["refused_unbound"] == 1
    r.attach(Mgr(8, 16))  # another slot block size: left unbound (hits recomputed), once
    r.attach(e.mgr)
    assert r.mgr is None and r.lookup("q", st, 0, 4) == 0 and "unbound" in logged[-1]
    assert not RoutedRows(e.sched, [1, 2], 10 ** 6, log=lambda *a, **k: None).bind(e.mgr)  # not a cacheable group
    r2 = RoutedRows(e.sched, [0], 10 ** 6, 100, log=lambda *a, **k: None)
    assert r2.bind(e.mgr) and r2.capacity == 100  # host block count caps the budget


def test_reset_drops_rows():
    e = Engine()
    t = tokens(18)
    a = e.admit("a", t)
    e.run(a)
    e.finish(a)
    e.routed.reset()
    assert e.admit("b", t).hit == 0


# ---------------------------------------------------------------------------- the config exemption (served_route_patch)
CONFIG_SRC = '''class Cfg:
    def __init__(self, kv):
        self.kv_transfer_config = kv

    def check(self):
        if True:
''' + P.CONFIG_OLD + '''                    "connectors (PD disaggregation, KV cache offload)."
                )
        return True
'''


def fake_site(tmp_path, monkeypatch):
    site = tmp_path / "site"
    files = {P.TARGET['relative']: "\nclass RequestState:\n    pass\n",
             P.CAPTURER['relative']: "def get_routed_experts_attn_gid(kv_cache_config):\n" + P.GID_OLD,
             P.CONFIG['relative']: CONFIG_SRC}
    for rel in P.OFFLOAD_GUARDS:
        files[rel] = f"# {rel}\n"
    for rel, text in files.items():
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(text)
    sha = {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()}
    monkeypatch.setitem(P.TARGET, 'sha256', sha[P.TARGET['relative']])
    monkeypatch.setitem(P.CAPTURER, 'sha256', sha[P.CAPTURER['relative']])
    monkeypatch.setitem(P.CONFIG, 'sha256', sha[P.CONFIG['relative']])
    monkeypatch.setattr(P, 'OFFLOAD_GUARDS', {rel: sha[rel] for rel in P.OFFLOAD_GUARDS})
    return site


def check(site, connector, module="vllm.arc_offload"):
    ns = {}
    exec(compile((site / P.CONFIG['relative']).read_text(), "cfg", "exec"), ns)
    kv = NS(is_kv_transfer_instance=True, kv_connector=connector, kv_connector_module_path=module)
    return ns["Cfg"](kv).check()


def test_config_exemption_only_for_arc_offload(tmp_path, monkeypatch):
    site = fake_site(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        check(site, "ArcOffloadingConnector")  # pinned file: refused
    ident = P.apply(site, tmp_path / "out", "/tmp/route", offload=True)
    assert ident["offload"] and ident["config_patched_sha256"]
    assert check(site, "ArcOffloadingConnector")
    for other in (("OffloadingConnector", "vllm.arc_offload"), ("ArcOffloadingConnector", "elsewhere")):
        with pytest.raises(ValueError):
            check(site, *other)
    assert P.apply(site, tmp_path / "out", "/tmp/route", offload=True) == ident  # restart, same overlay
    with pytest.raises(RuntimeError):
        P.apply(site, tmp_path / "out", "/tmp/route", offload=False)  # the same tree under another overlay


def test_offload_guard_refuses_unknown_offloading_scheduler(tmp_path, monkeypatch):
    site = fake_site(tmp_path, monkeypatch)
    (site / next(iter(P.OFFLOAD_GUARDS))).write_text("# changed\n")
    with pytest.raises(RuntimeError, match="unrecognized"):
        P.apply(site, tmp_path / "out", "/tmp/route", offload=True)
    P.apply(site, tmp_path / "out2", "/tmp/route")  # without the host tier the guard does not apply
    with pytest.raises(ValueError):
        check(site, "ArcOffloadingConnector")


# ---------------------------------------------------------------------------- the RTX replay check (arena.r3_offload_check)
def fake_server(route_dir, stale=False, n=24, cached=16):
    """Chat requests leave served-route records (``<API id>-<8 chars>``, written a moment after the reply, like the
    runtime's writer thread); B (the third send of a prompt) restores A's rows unless ``stale``. State per server
    (port): servers checked at once share ``route_dir``, and their rows differ, so a record taken from the other
    server fails. ``partial`` counts requests that finished without the full-record flag."""
    import threading

    import httpx

    seen, last_scrub, lock = collections.Counter(), {}, threading.Lock()
    partial = []

    def rows_for(text, salt):
        seed = hashlib.sha256(f"{text}|{salt}".encode()).digest()
        return np.frombuffer((seed * 8)[:n * L * K], dtype=np.uint8).reshape(n, L, K).astype(np.int16)

    def write(rid, ids):
        head = np.array([0, n, L, K], dtype=np.int32).tobytes()
        (route_dir / f"{rid}.mm.json").write_text("{}")
        (route_dir / f"{rid}.rt").write_bytes(head + np.arange(n, dtype=np.int32).tobytes() + ids.tobytes())
        (route_dir / f"{rid}.json").write_text("{}")

    def handler(request):
        if request.url.path == "/reset_prefix_cache":
            return httpx.Response(200, json={"success": True})
        port = request.url.port or 0
        body = json.loads(request.content)
        text = body["messages"][0]["content"]
        key = text.removeprefix("[scrub] ")
        with lock:
            seen[port, key] += 1
            count, nth = sum(seen.values()), seen[port, key]
        salt = f"{port}|{key}"
        if text.startswith("[scrub] "):
            ids, hit = rows_for(salt, "scrub"), 0
            last_scrub[port, key] = ids
        elif nth == 1:
            ids, hit = rows_for(salt, "a"), 0
        else:
            ids, hit = rows_for(salt, "b"), cached
            ids[:hit] = (last_scrub[port, key] if stale else rows_for(salt, "a"))[:hit]
        if not (route_dir / "full.on").exists():
            partial.append((port, key))
        api = f"chatcmpl-{port}-{count:04d}"
        if port % 2:  # the odd server is the slower one: its requests end after the other server's check
            time.sleep(0.004)
        threading.Timer(0.01, write, (f"{api}-{hashlib.sha256(api.encode()).hexdigest()[:8]}", ids)).start()
        return httpx.Response(200, json={"id": api, "usage": {"prompt_tokens": n,
                                                              "prompt_tokens_details": {"cached_tokens": hit}}})

    transport = httpx.MockTransport(handler)
    transport.partial = partial
    return transport


@pytest.mark.parametrize("stale", [False, True])
def test_replay_check_verdict(tmp_path, stale):
    from arena.r3_offload_check import run_check

    route = tmp_path / "route"
    route.mkdir()
    probes = tmp_path / "probes.jsonl"
    probes.write_text("".join(json.dumps({"label": f"p{i}", "body": {"messages": [{"role": "system",
                                                                                  "content": f"game {i}"}]}}) + "\n"
                              for i in range(6)))
    out = run_check({"r3_offload": str(probes), "settle_s": 0}, base_url="http://x/v1", model="m", output=tmp_path,
                    route_dir=str(route), log=lambda *a, **k: None, transport=fake_server(route, stale=stale))
    assert out["hits"] == 6 and out["scrub_diff_median"] > 0.5
    assert out["ok"] is (not stale)
    assert all(r["restored"]["equal"] is (not stale) for r in out["per_probe"])


# ---------------------------------------------------------------------------- the connector glue (vllm/arc_offload.py)
class FakeScheduler:
    """The two Scheduler methods the runtime hooks; update_from_output's store_batch call is the pinned line."""

    def __init__(self, connector, mgr):
        self.connector, self.routed_experts_mgr = connector, mgr

    def schedule(self, request, blocks):
        hit, _ = self.connector.get_num_new_matched_tokens(request, 0)
        self.connector.update_state_after_alloc(request, NS(blocks=blocks), hit)
        so = NS(scheduled_new_reqs=[], scheduled_cached_reqs=NS(req_ids=[], resumed_req_ids=set(), new_block_ids=[]),
                partial_tail_offloads=None)
        so.kv_connector_metadata = self.connector.build_connector_meta(so)
        return so, hit

    def update_from_output(self, scheduler_output, model_runner_output):
        re = model_runner_output
        self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)


def runtime(monkeypatch, sched, status_dir=None):
    """arc_offload_runtime imported against stand-in vllm modules; the base connector serves ``sched``."""
    import dataclasses
    import importlib.util
    import pathlib
    import sys
    import types

    from arc_infra import arc_routed_offload

    class Base:
        def __init__(self, vllm_config, role, kv_cache_config):
            self.connector_scheduler = sched if role == "scheduler" else None
            self.hits, self.store_jobs = {}, {}

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            hit = self.hits.get(request.request_id, 0)
            return hit, bool(hit)

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            return None

        def build_connector_meta(self, out):
            jobs, self.store_jobs = self.store_jobs, {}
            return NS(store_jobs=jobs)

        def request_finished_all_groups(self, request, block_ids):
            return False, None

        def reset_cache(self):
            return True

    mods = {"vllm": {}, "vllm.arc_routed_offload": {"RoutedRows": arc_routed_offload.RoutedRows,
                                                    "status_path": arc_routed_offload.status_path},
            "vllm.distributed": {}, "vllm.distributed.kv_transfer": {}, "vllm.distributed.kv_transfer.kv_connector": {},
            "vllm.distributed.kv_transfer.kv_connector.v1": {},
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector": {"OffloadingConnector": Base},
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading": {},
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker": {},
            "vllm.v1": {}, "vllm.v1.core": {},
            "vllm.v1.core.kv_cache_manager": {"KVCacheManager": type("KVCacheManager", (), {}),
                                              "KVCacheBlocks": lambda blocks: NS(blocks=blocks)},
            "vllm.v1.core.sched": {}, "vllm.v1.core.sched.scheduler": {"Scheduler": FakeScheduler},
            "vllm.v1.kv_cache_interface": {"MambaSpec": type("MambaSpec", (), {})}}
    for name, attrs in mods.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    for name in mods:
        if "." in name:
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, sys.modules[name])
    for attr in ("schedule", "update_from_output"):
        monkeypatch.setattr(FakeScheduler, attr, FakeScheduler.__dict__[attr])
    monkeypatch.delattr(FakeScheduler, "_arc_routed_hook", raising=False)
    spec = importlib.util.spec_from_file_location("vllm.arc_offload", pathlib.Path(__file__).resolve().parents[1]
                                                  / "src/arc_infra/arc_offload_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @dataclasses.dataclass
    class Group:
        kv_cache_spec: object
        layer_names: list
        is_eagle_group: bool = False

    config = dataclasses.make_dataclass("Cfg", [("kv_cache_groups", list)])(
        [Group(NS(prefix_cacheable=True), ["layers.0.attn"])])
    vcfg = NS(model_config=NS(enable_return_routed_experts=True),
              kv_transfer_config=NS(kv_connector_extra_config={"cpu_bytes_to_use": 1 << 20,
                                                              "arc_routed_status_dir": status_dir}))
    return module, module.ArcOffloadingConnector(vcfg, "scheduler", config)


def test_connector_glue_saves_on_landing_and_restores(monkeypatch, tmp_path):
    e = Engine(routed=False)  # the fake engine's pieces; the connector owns the RoutedRows here
    e.sched.manager = NS(_num_blocks=100)
    module, conn = runtime(monkeypatch, e.sched, str(tmp_path))
    s = FakeScheduler(conn, e.mgr)
    t = tokens(18)
    a = e.admit("a", t)
    keys = e.sched._req_status["a"].group_states[0].offload_keys
    conn.store_jobs = {1: NS(req_id="a")}
    e.sched._jobs[1] = NS(keys=set(keys), req_id="a", is_store=True)
    so, hit = s.schedule(NS(request_id="a"), [[NS(block_id=b, block_hash=None, is_null=False) for b in a.blocks]])
    assert hit == 0 and conn._routed.mgr is e.mgr and conn._routed.pending  # bound at the first schedule
    rows_a = np.array([e.route(a, p) for p in range(18)])
    s.update_from_output(so, NS(routing_data=rows_a, slot_mapping=np.array(
        [a.blocks[p // 4] * 4 + p % 4 for p in range(18)])))
    assert len(conn._routed.rows) == 4 and conn._routed.landing is None
    scrub(e, 18)
    b = e.admit("b", t)
    conn.hits["b"] = 16
    _, hit = s.schedule(NS(request_id="b"), [[NS(block_id=x, block_hash=None, is_null=False) for x in b.blocks]])
    assert hit == 16
    np.testing.assert_array_equal(e.mgr.get(b.blocks, 16), rows_a[:16])
    e.admit("c", tokens(18, seed=8))
    conn.hits["c"] = 16
    assert s.schedule(NS(request_id="c"), [[NS(block_id=x, block_hash=None, is_null=False) for x in b.blocks]])[1] == 0
    assert conn.reset_cache() is True and not conn._routed.rows
    status = json.loads(next(tmp_path.glob("r3-offload-*.json")).read_text())
    assert status["bound"] is True and status["tried"] is True  # written at the first schedule's bind


# ---------------------------------------------------------------------------- unbound is loud (RL review 26 Oct)
STUDY_R3 = {"server": {"served_route": "/tmp/arc-served-route"}, "cache_experiment": {"cpu_cache_gib": 32}}


def unbound_server(tmp_path):
    """A server whose rows hook could not bind (16-token routed slots vs 4-token offload blocks): its log and status."""
    from arc_infra.arc_routed_offload import status_path

    e = Engine(routed=False)
    logged = []
    r = RoutedRows(e.sched, [0], 10 ** 6, 100, log=lambda line, **k: logged.append(line),
                   status_file=status_path(str(tmp_path)))
    r.attach(Mgr(8, 16))
    assert r.lookup("q", ReqState([b"k"]), 0, 4) == 0
    r.status(force=True)
    (tmp_path / "vllm-openai-server.log").write_text("\n".join(logged) + "\n")
    return logged


def test_block_size_mismatch_raises_alert_row_and_bound_zero(tmp_path):
    from arena import r3_offload_check as r3

    logged = unbound_server(tmp_path)
    assert any(line.startswith("ARC_ROUTED_OFFLOAD_ALERT ") and "unbound" in line for line in logged)
    feed = []
    out = r3.watch(STUDY_R3, None, [tmp_path], emit=lambda kind, **kw: feed.append((kind, kw)),
                   metrics_path=tmp_path / "train-metrics.jsonl", wait_s=0)
    row = json.loads((tmp_path / "train-metrics.jsonl").read_text().splitlines()[-1])
    assert row["m"]["infra.r3_offload_bound"] == 0 and row["m"]["infra.r3_offload_refused.unbound"] == 1
    assert "unbound" in row["alert"] and "block" in row["alert"]  # watch.py's run-alert rule fires on it
    assert feed[0][0] == "alert" and feed[0][1]["why"] == row["alert"] and out["bound"] == 0  # rl_seq's feed alert


def test_bound_missing_stub_and_off(tmp_path):
    from arc_infra.arc_routed_offload import status_path
    from arena import r3_offload_check as r3

    feed = []

    def emit(kind, **kw):
        feed.append(kind)

    m = tmp_path / "m.jsonl"
    (tmp_path / "ok").mkdir()
    e = Engine()  # bound: a status file with bound true, no alert
    e.routed.status_path = status_path(str(tmp_path / "ok"))
    e.routed.write_status()
    assert r3.watch(STUDY_R3, None, [tmp_path / "ok"], emit=emit, metrics_path=m, wait_s=0)["bound"] == 1
    assert not feed and "alert" not in json.loads(m.read_text())
    out = r3.watch(STUDY_R3, None, [tmp_path / "none"], emit=emit, metrics_path=m, wait_s=0)
    assert feed == ["alert"] and "no status" in out["alert"]  # the hook never ran: as loud as unbound
    off = ({"server": {"served_route": "/x"}}, {"server_env": {"ARENA_CPU_CACHE_GIB": "0"}})
    assert r3.watch(*off, [tmp_path], emit=emit, metrics_path=m, wait_s=0) is None  # host tier off: nothing to check
    (tmp_path / "stub").mkdir()
    (tmp_path / "stub/vllm-openai-server.log").write_text("STUB_CORPUS files=1\n")
    assert r3.watch(STUDY_R3, None, [tmp_path / "stub"], emit=emit, metrics_path=m, wait_s=0)["alert"] is None


def probes_file(tmp_path, n=6):
    p = tmp_path / "probes.jsonl"
    p.write_text("".join(json.dumps({"label": f"p{i}", "body": {"messages": [{"role": "system",
                                                                              "content": f"game {i}"}]}}) + "\n"
                         for i in range(n)))
    return p


@pytest.mark.parametrize("case", ["ok", "stale", "miss", "unbound"])
def test_boot_check_row_mismatch_host_miss_and_unbound_fail(tmp_path, case):
    """rl_seq's r3_check stage (H5): restored rows equal A's on every hit, >= min_hits hits, bound == 1; its records
    and the full-record flag are gone after."""
    from arena import r3_offload_check as r3

    route, srv = tmp_path / "route", tmp_path / "srv"
    route.mkdir()
    srv.mkdir()
    if case == "unbound":
        unbound_server(srv)
    else:
        (srv / "r3-offload-7.json").write_text(json.dumps({"bound": 1, "saved": 4}))
    transport = fake_server(route, stale=case == "stale", cached=0 if case == "miss" else 16)
    servers = [NS(root="http://127.0.0.1:1234", name="server1"), NS(root="http://127.0.0.1:1235", name="server2")]
    row = r3.boot_check(servers, {"probes": str(probes_file(tmp_path)), "min_hits": 6, "settle_s": 0},
                        tmp_path / "out", model="policy", route_dir=str(route), status_dirs=[srv],
                        transport=transport)
    assert row["ok"] is (case == "ok") and set(row["servers"]) == {"server1", "server2"}
    assert row["bound"] == (0 if case == "unbound" else 1) and not row["stub"]
    s1 = row["servers"]["server1"]
    assert s1["ok"] is (case in ("ok", "unbound"))
    assert s1["hits"] == (0 if case == "miss" else 6)
    if case != "ok":
        assert row["why"].startswith("r3_check: ")
    assert not list(route.glob("*.rt")) and not (route / "full.on").exists()
    assert json.loads((tmp_path / "out" / "verdict.json").read_text())["ok"] is row["ok"]


def test_boot_check_two_servers_at_once_share_the_route_dir_safely(tmp_path):
    """Both servers checked at once in one route dir: each request's record is its own (by API id; the servers'
    rows differ, so a borrowed record fails), the full-record flag stays up until the slower server ends (a per-check
    flag drops it under the slower server: partial records), each check removes only its own records; probes_max."""
    import concurrent.futures

    from arena import r3_offload_check as r3

    route, srv = tmp_path / "route", tmp_path / "srv"
    route.mkdir()
    srv.mkdir()
    (srv / "r3-offload-7.json").write_text(json.dumps({"bound": 1}))
    (route / "play-0001-abcdefgh.rt").write_bytes(b"")  # a record that is not the check's stays
    probes = probes_file(tmp_path, n=8)
    fakes = [fake_server(route), fake_server(route)]
    servers = [NS(root="http://127.0.0.1:1234", name="server1"), NS(root="http://127.0.0.1:1235", name="server2")]
    began = time.time()
    row = r3.boot_check(servers, {"probes": str(probes), "min_hits": 6, "probes_max": 6, "settle_s": 0.2},
                        tmp_path / "out", model="policy", route_dir=str(route), status_dirs=[srv], transport=fakes)
    took = time.time() - began
    assert row["ok"] and not row["errors"] and row["bound"] == 1
    assert all(v["probes"] == 6 and v["hits"] == 6 and v["ok"] for v in row["servers"].values())
    assert not fakes[0].partial and not fakes[1].partial
    assert took < 2 * 0.2 + 3.0  # the settles overlap (one server's whole check is ~1 s here)
    time.sleep(0.05)  # the last records land
    assert [p.name for p in route.glob("*.rt")] == ["play-0001-abcdefgh.rt"] and not (route / "full.on").exists()

    # the collision a per-check flag would cause: the faster check drops the flag under the slower one
    fakes = [fake_server(route), fake_server(route)]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda i: r3.run_check(
            {"r3_offload": str(probes), "settle_s": 0.2 * i}, base_url=servers[i].root + "/v1", model="m",
            output=tmp_path / f"own{i}", route_dir=str(route), log=lambda *a, **k: None, transport=fakes[i],
            cleanup=True), [0, 1]))
    assert fakes[1].partial  # what boot_check's single flag prevents
