"""Reply-id seeds (29 Oct): a restored attempt's harness writes its recorded replies for the play server's reply-id
cache, so a server that never generated them re-sends them with the ids the source server used (v13: 4 of 9 starts
harvested in v12 failed the restore digest)."""

import json

from arena import reply_ids_runtime as R
from arena import rollout_record

END = 99


class Tok:
    vocab = {1: "<s>", 2: "sys", 3: "user", 4: "assistant", 10: " Circular", 11: " C", 12: "ircular", 13: " ok",
             END: "<|im_end|>"}
    all_special_ids = [END]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]

    def encode(self, text, add_special_tokens=False):  # canonical: longest piece first
        pieces = sorted(self.vocab.items(), key=lambda kv: -len(kv[1]))
        out = []
        while text:
            i, piece = next((i, p) for i, p in pieces if text.startswith(p))
            out.append(i)
            text = text[len(piece):]
        return out


class Final:
    def __init__(self, prompt, generated):
        self.prompt_token_ids = prompt
        self.outputs = [type("O", (), {"token_ids": generated})()]


def served(cache, ids):
    engine_input = {"type": "token", "prompt_token_ids": list(ids)}
    cache.substitute(engine_input, Tok(), 10_000)
    return engine_input["prompt_token_ids"]


def test_a_seeded_server_resends_the_source_servers_ids(tmp_path):
    p1, gen1 = [1, 2, 3, 4], [11, 12, 13, END]  # sampled " C" "ircular"; canonical is " Circular"
    canon2 = p1 + [10, 13, END] + [3, 4]
    source = R.ReplyIds()
    source.record(Final(p1, gen1), Tok())
    want = served(source, canon2)
    assert want == p1 + gen1 + [3, 4]

    fresh = R.ReplyIds()  # the restoring container: no record, the canonical prompt
    assert served(fresh, canon2) == canon2

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    rows = [{"usage": {"prompt_tokens": len(p1)}, "prompt_digest": rollout_record.ids_digest(p1), "tokens": gen1},
            {"usage": {"prompt_tokens": 2}, "prompt_digest": rollout_record.ids_digest([1, 2]), "tokens": [13, END]}]
    path = rollout_record.seed_reply_ids(rows, seeds)
    assert path is not None and path.suffix == ".json"
    seeded = R.ReplyIds()
    assert seeded.load_seeds(Tok(), seeds) == 1 and seeded.stats["seed_canonical"] == 1  # " ok" needs no record
    assert served(seeded, canon2) == want
    assert seeded.load_seeds(Tok(), seeds) == 0  # read once


def test_no_seed_file_without_the_servers_directory(tmp_path):
    rows = [{"usage": {"prompt_tokens": 4}, "prompt_digest": "ab" * 16, "tokens": [11, END]}]
    assert rollout_record.seed_reply_ids(rows, tmp_path / "absent") is None
    assert R.ReplyIds().load_seeds(Tok(), tmp_path / "absent") == 0


def test_a_seed_with_another_text_is_never_substituted(tmp_path):
    p1, gen1 = [1, 2, 3, 4], [11, 12, 13, END]
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "a.json").write_text(json.dumps({"records": [[4, rollout_record.ids_digest(p1), gen1]]}))
    cache = R.ReplyIds()
    cache.load_seeds(Tok(), seeds)
    other = p1 + [13, 13, END] + [3, 4]  # the re-sent reply's text differs
    assert served(cache, other) == other


def test_a_groups_identical_seed_files_are_judged_once(tmp_path):
    p1, gen1 = [1, 2, 3, 4], [11, 12, 13, END]
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    body = json.dumps({"records": [[4, rollout_record.ids_digest(p1), gen1], [2, rollout_record.ids_digest([1, 2]), [13, END]]]})
    for i in range(8):  # one per try of a group: the same start
        (seeds / f"t{i}.json").write_text(body)
    cache = R.ReplyIds()
    assert cache.load_seeds(Tok(), seeds) == 1
    assert cache.stats["seed_file_repeat"] == 7 and cache.stats["seed_canonical"] == 1


def test_a_first_request_before_slow_seeds_waits_for_them_alone(tmp_path):
    """The loader thread judges a seed file slowly; a chat request meanwhile is not held up, and the restored
    attempt's first live request awaits its own file's mark, then renders with the source server's ids."""
    import asyncio
    import time

    class Slow(Tok):
        def decode(self, ids, skip_special_tokens=False):
            time.sleep(0.4)
            return super().decode(ids, skip_special_tokens)

    p1, gen1 = [1, 2, 3, 4], [11, 12, 13, END]
    canon2 = p1 + [10, 13, END] + [3, 4]
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    cache = R.ReplyIds()
    cache.poll_seeds(Slow(), seeds)  # the loader thread watches the directory from now on
    rows = [{"usage": {"prompt_tokens": 4}, "prompt_digest": rollout_record.ids_digest(p1), "tokens": gen1}]
    path = rollout_record.seed_reply_ids(rows, seeds)
    began = time.time()
    assert served(cache, canon2) == canon2  # another attempt's request: no wait on the seed work
    assert time.time() - began < 0.3

    async def first_live_request():
        waited = await rollout_record.wait_seeds(path, timeout=5)
        return waited, served(cache, canon2)

    waited, prompt = asyncio.run(first_live_request())
    assert waited is not None and waited >= 0.3
    assert prompt == p1 + gen1 + [3, 4]
    assert cache.stats["seeded"] == 1


def test_a_seed_wait_times_out_and_the_request_goes_on(tmp_path):
    import asyncio

    path = tmp_path / "never.json"
    path.write_text("{}")
    assert asyncio.run(rollout_record.wait_seeds(path, timeout=0.2, poll=0.05)) is None


def test_restore_writes_seeds_at_its_first_scripted_request(tmp_path, monkeypatch):
    """rollout_record: the seed file exists from the scripted replay's start and goes after the first live reply."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    monkeypatch.setattr(rollout_record, "SEED_DIR", str(seeds))
    rows = [{"usage": {"prompt_tokens": 4}, "prompt_digest": "ab" * 16, "tokens": [11, END]}]
    path = rollout_record.seed_reply_ids(rows)
    assert path is not None and path.parent == seeds and path.exists()


def test_stub_seed_loader_marks_each_file_once(tmp_path):
    """The replay stub's stand-in loader (NK preflight): each seed file read and marked done once; a bad file is a stub
    error, still marked so its attempt does not wait."""
    from arena.replay_stub import seed_loader

    rows = [{"usage": {"prompt_tokens": 4}, "prompt_digest": "ab" * 16, "tokens": [11, END]}]
    path = rollout_record.seed_reply_ids(rows, tmp_path)
    (tmp_path / "bad.json").write_text("{")
    stats = {"errors": 0}
    seed_loader(tmp_path, stats, poll=0, rounds=2)
    assert stats == {"errors": 1, "seeds": 1, "seed_records": 1}
    assert (tmp_path / (path.name + ".done")).exists() and (tmp_path / "bad.json.done").exists()
    from arena.replay_stub import seed_stats
    assert seed_stats(tmp_path) == stats  # what the stub's log line merges from the loader process


def test_restore_result_records_seeding(tmp_path, monkeypatch):
    """post(): the restore result carries seeded=1 from the first scripted request (level_bench copies it and the live
    request's seed_wait_timeout into the attempt's restore row; the NK leg judges both end to end)."""
    import asyncio

    import httpx

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    monkeypatch.setattr(rollout_record, "SEED_DIR", str(seeds))
    monkeypatch.setattr(rollout_record, "SEED_WAIT_S", 0.3)
    h = rollout_record.RolloutHTTP.__new__(rollout_record.RolloutHTTP)
    h.restore = {"request": 1, "rows": [{"usage": {"prompt_tokens": 4}, "prompt_digest": "ab" * 16, "tokens": [11, END]},
                                        {"prompt_digest": "cd" * 16}]}
    h.restore_at, h.restore_result, h.seeded, h.stats = 0, None, None, {}
    h.unclock = lambda p: p
    h.restore_post = lambda url, payload: setattr(h, "restore_at", 1) or "scripted"
    asyncio.run(h.post("http://x/v1/chat/completions", json={"messages": []}))
    assert h.restore_result["seeded"] == 1 and h.seeded.exists()


def test_stub_seed_process_marks_files_from_its_own_process(tmp_path):
    """The stub starts the loader as a child process (no GIL share with its handler threads)."""
    import multiprocessing
    import time

    from arena.replay_stub import seed_process, seed_stats

    proc = multiprocessing.Process(target=seed_process, args=(str(tmp_path),), daemon=True)
    proc.start()
    try:
        path = rollout_record.seed_reply_ids([{"usage": {"prompt_tokens": 4}, "prompt_digest": "ab" * 16,
                                               "tokens": [11, END]}], tmp_path)
        done = tmp_path / (path.name + ".done")
        deadline = time.time() + 10
        while not done.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert done.exists()
        while seed_stats(tmp_path).get("seeds") != 1 and time.time() < deadline:
            time.sleep(0.05)
        assert seed_stats(tmp_path) == {"errors": 0, "seeds": 1, "seed_records": 1}
    finally:
        proc.terminate()
