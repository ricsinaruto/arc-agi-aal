"""R3 served-routing capture: overlay on the pinned output processor, the API-side dumper on fake
request states, reader and resolver."""

import hashlib
import importlib
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import pytest

from arc_infra import served_route_patch as patch
from arena import served_route as sr

PINNED = Path("./.claude/worktrees/throughput-arm/keep/vllm-full")  # same build (g8e685d198)


CAPTURER_BODY = "def get_routed_experts_attn_gid(kv_cache_config):\n" + patch.GID_OLD


def synthetic_site(tmp_path, monkeypatch, body="import os\n\n\nclass RequestState:\n    pass\n"):
    site = tmp_path / "site"
    target = site / patch.TARGET["relative"]
    target.parent.mkdir(parents=True)
    target.write_text(body)
    monkeypatch.setitem(patch.TARGET, "sha256", hashlib.sha256(body.encode()).hexdigest())
    capturer = site / patch.CAPTURER["relative"]
    capturer.parent.mkdir(parents=True)
    capturer.write_text(CAPTURER_BODY)
    monkeypatch.setitem(patch.CAPTURER, "sha256", hashlib.sha256(CAPTURER_BODY.encode()).hexdigest())
    return site, target


def test_capturer_takes_a_wrapped_full_attention_group():
    class FullAttentionSpec:
        compress_ratio = 1

    class Compressed(FullAttentionSpec):
        compress_ratio = 4

    class Circular:
        pass

    def group(*specs, wrapped=True):
        spec = types.SimpleNamespace(kv_cache_specs={str(i): s for i, s in enumerate(specs)}, block_size=64) \
            if wrapped else specs[0]
        return types.SimpleNamespace(kv_cache_spec=spec)

    ns = {"FullAttentionSpec": FullAttentionSpec, "logger": types.SimpleNamespace(info=lambda *a: None)}
    exec("def get_routed_experts_attn_gid(kv_cache_config):\n" + patch.GID_NEW, ns)
    pick = ns["get_routed_experts_attn_gid"]
    config = types.SimpleNamespace(kv_cache_groups=[group(Circular()), group(Compressed()), group(Compressed(), FullAttentionSpec())])
    assert pick(config) == 2
    assert pick(types.SimpleNamespace(kv_cache_groups=[group(Circular()), group(FullAttentionSpec(), wrapped=False)])) == 1
    with pytest.raises(ValueError):
        pick(types.SimpleNamespace(kv_cache_groups=[group(Circular()), group(Compressed())]))


def test_overlay_patches_once_fills_dir_and_accepts_restart(tmp_path, monkeypatch):
    site, target = synthetic_site(tmp_path, monkeypatch)
    identity = patch.apply(site, tmp_path / "out", tmp_path / "routes")
    assert target.read_text().count("_arc_served_route_install(RequestState)") == 1
    assert "ARC (R3): a hybrid model's group" in (site / patch.CAPTURER["relative"]).read_text()
    assert f"DIR = {str(tmp_path / 'routes')!r}" in (site / patch.HELPER).read_text()
    assert identity["server_flags"] == ["--enable-return-routed-experts"]
    assert patch.apply(site, tmp_path / "out", tmp_path / "routes") == identity  # restart: same overlay accepted
    with pytest.raises(RuntimeError):
        patch.apply(site, tmp_path / "out", tmp_path / "other")  # another directory: a different helper
    with pytest.raises(ValueError):
        patch.apply(site, tmp_path / "out2", "relative/dir")


def test_overlay_rejects_unknown_file(tmp_path, monkeypatch):
    site, target = synthetic_site(tmp_path, monkeypatch)
    target.write_text(target.read_text() + "# drift\n")
    with pytest.raises(RuntimeError):
        patch.apply(site, tmp_path / "out", tmp_path / "routes")


@pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
def test_overlay_on_the_pinned_output_processor(tmp_path):
    """The real pinned file: hash accepted, the tail compiles, RequestState has the two methods the dumper wraps."""
    original = (PINNED / patch.TARGET["relative"]).read_bytes()
    assert hashlib.sha256(original).hexdigest() == patch.TARGET["sha256"]
    text = patch.patched_source(original).decode()
    assert text.endswith("_arc_served_route_install(RequestState)\n")
    assert "    def from_new_request(\n" in text and "    def _new_completion_output(\n" in text
    assert "routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)" in text
    capturer = (PINNED / patch.CAPTURER["relative"]).read_bytes()
    assert hashlib.sha256(capturer).hexdigest() == patch.CAPTURER["sha256"]
    assert "ARC (R3)" in patch.patched_capturer(capturer).decode()
    args = (PINNED / "vllm/engine/arg_utils.py").read_text()
    assert '"--enable-return-routed-experts"' in args


def test_serving_configure_rejects_bad_served_route():
    from arena import serving

    with pytest.raises(ValueError):
        serving.configure(types.SimpleNamespace(), "/tmp/x", served_route="relative")


# ----------------------------------------------------------------------------- API-side dumper on fake states
class FakeLora:
    lora_name, lora_int_id, lora_path = "policy", 3, "/runs/a/r02"


class FakeRequestState:
    """The two RequestState methods the overlay wraps, with vLLM's signatures."""

    def __init__(self, request_id, prompt, cached, lora=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt
        self.num_cached_tokens = cached
        self.lora_request = lora
        self.detokenizer = types.SimpleNamespace(output_token_ids=[])

    @classmethod
    def from_new_request(cls, tokenizer, request, prompt, parent_req, request_index, queue, log_stats,
                         stream_interval):
        return cls(request.request_id, request.prompt_token_ids, 0, request.lora_request)

    def _new_completion_output(self, new_token_ids, finish_reason, stop_reason):
        routed = getattr(self, "routed", None) if finish_reason is not None else None
        return types.SimpleNamespace(token_ids=new_token_ids, routed_experts=routed)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import arc_infra.served_route_runtime as module

    module = importlib.reload(module)
    monkeypatch.setattr(module, "DIR", str(tmp_path / "routes"))
    cls = type("RS", (FakeRequestState,), {})
    module.install(cls)
    yield module, cls
    module._WRITER = None


def wait_closed(module, n, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if module._WRITER.counts["closed"] >= n:
            return
        time.sleep(0.02)
    raise AssertionError(dict(module._WRITER.counts))


def test_dump_writes_uncached_rows_meta_and_drops_the_array(runtime, tmp_path):
    module, cls = runtime
    image = types.SimpleNamespace(mm_position=types.SimpleNamespace(offset=2, length=3), identifier="h1")
    request = types.SimpleNamespace(request_id="chatcmpl-abc-1234abcd", prompt_token_ids=list(range(10, 20)),
                                    lora_request=FakeLora(), mm_features=[image])
    state = cls.from_new_request(tokenizer=None, request=request, prompt=None, parent_req=None, request_index=0,
                                 queue=None, log_stats=False, stream_interval=1)
    assert state._arc_mm == [[2, 3, "h1"]]
    state.num_cached_tokens = 4
    state.detokenizer.output_token_ids = [90, 91, 92]
    rows = 10 + 3 - 1  # the last sampled token is never fed
    state.routed = np.arange(rows * 2 * 2, dtype=np.uint8).reshape(rows, 2, 2)
    assert state._new_completion_output([92], None, None).routed_experts is None  # not finished
    out = state._new_completion_output([92], "stop", None)
    assert out.routed_experts is None  # dropped from the output
    wait_closed(module, 1)
    routes = tmp_path / "routes"
    rec = sr.read(routes / "chatcmpl-abc-1234abcd.rt")
    assert (rec["lo"], rec["hi"]) == (4, rows)
    assert list(rec["tokens"][4:]) == list(range(14, 20)) + [90, 91]
    assert np.array_equal(rec["ids"][4:], state.routed[4:].astype(np.int16))
    assert not rec["covered"][:4].any()
    meta = json.loads((routes / "chatcmpl-abc-1234abcd.mm.json").read_text())
    assert meta == {"lora": "policy|3|/runs/a/r02", "mm": [[2, 3, "h1"]], "prompt_tokens": 10, "rows": rows,
                    "cached": 4, "prefix_hash": f"{int(sr.prefix_hashes(list(range(10, 14)))[-1]):016x}"}
    assert json.loads((routes / "layers.json").read_text())["layers"] == [0, 1]
    store = sr.Store(routes, scratch=True)
    assert store.engine_id("chatcmpl-abc") == "chatcmpl-abc-1234abcd"


def test_dump_fully_cached_writes_nothing_and_counts(runtime, tmp_path):
    module, cls = runtime
    state = cls("chatcmpl-x-00000000", [1, 2, 3], 3)
    state.routed = np.zeros((3, 2, 2), dtype=np.uint8)
    state._new_completion_output([], "stop", None)
    assert module._WRITER.counts["nothing_new"] == 1
    assert not list((tmp_path / "routes").glob("*.rt"))


def test_dump_errors_are_counted_not_raised(runtime, tmp_path):
    module, cls = runtime
    state = cls("chatcmpl-y-00000000", [1, 2, 3], 0)
    state.routed = np.zeros((3, 2), dtype=np.uint8)  # wrong rank
    assert state._new_completion_output([], "stop", None).routed_experts is None
    assert module._WRITER.counts["errors"] == 1
    module._WRITER.status(force=True)
    status = sr.status(tmp_path / "routes")
    assert status["rt"] == 0 and status["server"]["errors"]


def test_capture_switches_off_on_low_disk(runtime, tmp_path, monkeypatch):
    module, cls = runtime
    # a fake free figure, not a huge MIN_FREE: Modal's Linux overlay reports more than 1 << 62 bytes free
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: types.SimpleNamespace(total=1 << 40, used=0, free=1 << 30))
    module._WRITER.status(force=True)
    state = cls("chatcmpl-z-00000000", [1, 2, 3], 0)
    state.routed = np.zeros((3, 2, 2), dtype=np.uint8)
    state._new_completion_output([], "stop", None)
    assert module._WRITER.counts["finished"] == 0 and not module._WRITER.active


# ----------------------------------------------------------------------------- resolver
def rec(lo, tokens, layer_value, total=None, meta=None):
    tokens = np.asarray(tokens, dtype=np.int32)
    hi = lo + len(tokens)
    out = {"lo": lo, "hi": hi, "tokens": np.full(hi, -1, np.int32), "ids": np.full((hi, 2, 2), -1, np.int16),
           "covered": np.zeros(hi, bool), "meta": meta}
    out["tokens"][lo:] = tokens
    out["ids"][lo:] = layer_value
    out["covered"][lo:] = True
    return out


def test_resolve_takes_own_then_chain_then_scratch():
    target = np.arange(12)
    meta = {"lora": "p", "mm": [[1, 2, "x"]]}
    own = rec(8, target[8:11], 3, meta=meta)  # positions 8-10 (11: last sampled token, never fed)
    chain = [rec(4, list(target[4:7]) + [99], 2)]  # computed 4-7, token 7 differs (re-tokenized reply)
    scratch = [rec(0, target[:6], 1, meta={"lora": "p", "mm": [[1, 2, "x"]]})]
    ids, audit = sr.resolve(target, own, chain, scratch)
    assert ids[:, 0, 0].tolist() == [1, 1, 1, 1, 2, 2, 2, -1, 3, 3, 3, -1]
    assert audit == {"tokens": 12, "own": 3, "own_mismatch": 0, "chain": 3, "scratch": 4, "donor": 0, "unresolved": 2}


def test_resolve_scratch_stops_at_a_different_image():
    target = np.arange(10)
    own = rec(6, target[6:9], 3, meta={"lora": "p", "mm": [[1, 2, "x"], [4, 2, "y"]]})
    other = rec(0, target[:8], 1, meta={"lora": "p", "mm": [[1, 2, "x"], [4, 2, "z"]]})
    wrong_lora = rec(0, target[:8], 5, meta={"lora": "q", "mm": [[1, 2, "x"], [4, 2, "y"]]})
    ids, audit = sr.resolve(target, own, [], [wrong_lora, other])
    assert ids[:, 0, 0].tolist() == [1, 1, 1, 1, -1, -1, 3, 3, 3, -1]
    assert audit["scratch"] == 4


def test_resolve_counts_own_mismatch():
    target = np.arange(6)
    own = rec(0, [0, 1, 7, 3, 4], 3, meta={"lora": "p", "mm": []})
    ids, audit = sr.resolve(target, own)
    assert audit["own_mismatch"] == 1 and audit["own"] == 4 and ids[2, 0, 0] == -1


def write_rt(path, start, tokens, ids):
    tokens = np.asarray(tokens, dtype=np.int32)
    ids = np.asarray(ids, dtype=np.int16)
    with open(path, "ab") as f:
        f.write(np.array([start, len(tokens), ids.shape[1], ids.shape[2]], dtype=np.int32).tobytes())
        f.write(tokens.tobytes())
        f.write(ids.tobytes())


def test_store_indexes_scratch_requests_and_ranks_by_images(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "KEY", 4)
    d = tmp_path / "r"
    d.mkdir()
    (d / "layers.json").write_text(json.dumps({"layers": [0, 1], "k": 2}))
    tok = np.arange(10)
    for rid, mm, value in (("chatcmpl-x", [[5, 2, "a"]], 1), ("chatcmpl-y", [[5, 2, "b"]], 2)):
        write_rt(d / f"{rid}-00000000.rt", 0, tok[:8], np.full((8, 2, 2), value))
        (d / f"{rid}-00000000.mm.json").write_text(json.dumps({"lora": "p", "mm": mm}))
    write_rt(d / "chatcmpl-e-11111111.rt", 8, tok[8:10], np.full((2, 2, 2), 3))
    (d / "chatcmpl-e-11111111.mm.json").write_text(json.dumps({"lora": "p", "mm": [[5, 2, "b"]]}))
    store = sr.Store(d, scratch=True)
    own = store.record("chatcmpl-e")
    picks = store.scratch_for(own, tok)
    assert [p["ids"][0, 0, 0] for p in picks] == [2, 1]  # y shares the image: ranked first
    table, audit = sr.resolve(tok, own, [], picks)
    assert table[:, 0, 0].tolist() == [2] * 8 + [3, 3] and audit["unresolved"] == 0


def test_prefix_hash_matches_between_writer_and_reader():
    import arc_infra.served_route_runtime as rt

    t = np.random.default_rng(0).integers(0, 150000, size=3000)
    assert np.array_equal(rt.prefix_hashes(t), sr.prefix_hashes(t))
    assert rt.prefix_hash(t[:1234]) == int(sr.prefix_hashes(t)[1234])


def test_shared_restored_prefix_resolves_through_a_hash_donor(runtime, tmp_path):
    """Try 1 computed the restored prefix after a system-prompt cache hit (record starts at 40, not 0); try 2's request
    hit the cache for that whole prefix. Without the hash donor its positions 40..400 are unresolved."""
    module, cls = runtime
    rng = np.random.default_rng(1)
    system, restored = list(rng.integers(0, 1000, 40)), list(rng.integers(0, 1000, 360))

    def finish(rid, prompt, cached, output):
        state = cls(rid, prompt, cached)
        state.detokenizer.output_token_ids = output
        rows = len(prompt) + len(output) - 1
        state.routed = (np.arange(rows)[:, None, None] % 200 + np.zeros((1, 2, 2), dtype=np.int64)).astype(np.uint8)
        state._new_completion_output([], "stop", None)

    finish("chatcmpl-first-00000000", system + [7] * 40, 0, [1, 2])  # cold: from scratch (scratch index)
    finish("chatcmpl-tryone-00000000", system + restored, 40, [5, 6, 7])  # system prompt cached
    target = system + restored + [5, 9, 9, 9]
    finish("chatcmpl-trytwo-00000000", target, 400, [3, 3])  # whole restored prefix cached (from try one)
    wait_closed(module, 3)
    store = sr.Store(tmp_path / "routes", scratch=True)
    own = store.record("chatcmpl-trytwo")
    full = target + [3]
    table, audit = sr.resolve(full, own, (), store.scratch_for(own, full))
    assert audit["unresolved"] >= 360  # the old reader
    table, audit = sr.resolve(full, own, (), store.scratch_for(own, full), store.donors(own, full))
    assert (audit["unresolved"], audit["donor"], audit["scratch"]) == (0, 360, 40)
    assert np.array_equal(table[:, 0, 0], np.arange(len(full)) % 200)


def test_hash_donor_is_ranked_by_token_match_past_the_shared_context(runtime, tmp_path):
    """Several tries' first requests share the system-prompt hit (same start and prefix hash) but restored other starts;
    the right donor is listed last. Ranking by match past the start finds it (q2r-v2: the first two were taken)."""
    module, cls = runtime
    rng = np.random.default_rng(2)
    system = list(rng.integers(0, 1000, 40))

    def finish(rid, prompt, cached, output):
        state = cls(rid, prompt, cached)
        state.detokenizer.output_token_ids = output
        rows = len(prompt) + len(output) - 1
        state.routed = (np.arange(rows)[:, None, None] % 200 + np.zeros((1, 2, 2), dtype=np.int64)).astype(np.uint8)
        state._new_completion_output([], "stop", None)

    finish("chatcmpl-first-00000000", system + [7] * 40, 0, [1, 2])
    for i in range(4):  # decoys: other restored starts after the same system prompt
        finish(f"chatcmpl-decoy{i}-00000000", system + list(rng.integers(0, 1000, 360)), 40, [5])
    restored = list(rng.integers(0, 1000, 360))
    finish("chatcmpl-right-00000000", system + restored, 40, [5, 6, 7])
    target = system + restored + [5, 9, 9, 9]
    finish("chatcmpl-two-00000000", target, 400, [3, 3])
    wait_closed(module, 7)
    store = sr.Store(tmp_path / "routes", scratch=True)
    own = store.record("chatcmpl-two")
    full = target + [3]
    donors = store.donors(own, full)
    assert donors and donors[0]["tokens"][40] == restored[0]
    table, audit = sr.resolve(full, own, (), store.scratch_for(own, full), donors)
    assert (audit["unresolved"], audit["donor"]) == (0, 360)


def test_dump_skips_a_row_count_that_would_shift_positions(runtime, tmp_path):
    module, cls = runtime
    state = cls("chatcmpl-r-00000000", [1, 2, 3, 4], 0)
    state.detokenizer.output_token_ids = [5, 6]
    state.routed = np.zeros((2, 2, 2), dtype=np.uint8)  # 2 rows for 6 tokens: a step without routing
    state._new_completion_output([], "stop", None)
    assert module._WRITER.counts["row_mismatch"] == 1
    assert not list((tmp_path / "routes").glob("*.rt"))


def write_one_rt(path, start, ids):
    n, layers, k = ids.shape
    head = np.array([start, n, layers, k], dtype=np.int32).tobytes()
    path.write_bytes(head + np.arange(n, dtype=np.int32).tobytes() + ids.astype(np.int16).tobytes())


def test_mask_hits_flags_routes_outside_the_mask(tmp_path):
    """rl1-prof-m1v1: unmasked served routes replayed into the masked trainer; mask_hits catches them on CPU."""
    (tmp_path / "layers.json").write_text(json.dumps({"layers": [0, 3], "k": 2}))
    mask = {"experts": 8, "pruned": {"0": [5, 6], "3": [1]}}
    inside = np.array([[[0, 1], [2, 3]], [[4, 7], [0, -1]]])  # layer 3 avoids 1, layer 0 avoids 5/6
    write_one_rt(tmp_path / "a.rt", 0, inside)
    got = sr.mask_hits(tmp_path, mask)
    assert got == {"files": 1, "routed": 7, "hits": 0, "by_layer": {}}
    outside = inside.copy()
    outside[1, 0, 0] = 6  # layer 0, pruned
    outside[0, 1, 1] = 1  # layer 3, pruned
    write_one_rt(tmp_path / "b.rt", 0, outside)
    got = sr.mask_hits(tmp_path, mask)
    assert got["hits"] == 2 and got["by_layer"] == {0: 1, 3: 1}
    assert sr.mask_hits(tmp_path, mask, max_files=1)["hits"] == 0  # sorted: a.rt only
