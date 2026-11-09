"""OPD stream pieces (22 Oct): streamed chunks in the trainer's loader, the stream queue, the per-step weight delta.
Needs torch (Modal CPU)."""
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import heal_delta  # noqa: E402
import heal_kd  # noqa: E402


def chunk_file(path, n, row0, reply_from, version=3):
    ids = torch.arange(10, 10 + n)
    reply = torch.zeros(n, dtype=torch.bool)
    reply[reply_from:] = True
    rows = n - row0
    beh = torch.full((rows,), -0.5)
    torch.save({"ids": ids.int(), "image": torch.zeros(n, dtype=torch.bool), "reply": reply, "row0": row0,
                "top_logp": torch.full((rows, 4), -1.5).half(), "top_ids": torch.zeros(rows, 4).int(),
                "tail": torch.full((rows,), -2.0).half(), "tok_logp": torch.full((rows,), -1.0),
                "beh_logp": beh, "version": version}, path)


def test_streamed_chunk_masks_the_prefix(tmp_path):
    chunk_file(tmp_path / "s-g1--c002.pt", 100, row0=59, reply_from=40)
    cpu = heal_kd.load_teacher_cpu(tmp_path / "s-g1--c002.pt")
    reply = cpu["reply"].bool()
    assert not reply[:59].any() and reply[59:99].all() and not reply[99]  # rows 59..98 predict tokens 60..99
    assert torch.isnan(cpu["t_tok"][:59]).all() and (cpu["t_tok"][59:99] == -1.0).all()
    assert torch.isnan(cpu["b"][:59]).all() and (cpu["b"][59:99] == -0.5).all()
    assert (cpu["labels"][:59] == -100).all() and cpu["version"] == 3
    assert cpu["teacher"][0].shape[0] == 100
    data = heal_kd.teacher_to(cpu, "cpu")
    got = heal_kd.slot_samp(data, tmp_path / "s-g1--c002.pt", "s-")
    assert got is not None and int(got[3].sum()) == 40  # the engine b without any behaviour pass


def test_stream_queue_freshest_first_and_drops_stale(tmp_path):
    for name, v in (("a", 1), ("b", 5), ("c", 4), ("d", 9)):
        (tmp_path / f"{name}.pt").write_bytes(b"x")
        (tmp_path / f"{name}.ready").write_text(json.dumps({"version": v}))
    q = heal_kd.StreamQueue(tmp_path, 0, max_age=6)
    assert q.take(2, 10) == [str(tmp_path / "d.pt"), str(tmp_path / "b.pt")]
    assert q.dropped == 1  # a: 9 updates old
    assert q.take(1, 10) == [str(tmp_path / "c.pt")]
    q2 = heal_kd.StreamQueue(tmp_path, 0, consumed=[str(tmp_path / "d.pt")], max_age=6, stall=0.0, poll=0.0)
    with pytest.raises(SystemExit, match="stalled"):
        q2.take(4, 10)
    ages = heal_kd.stream_ages(10, [9, 5, 4], q)
    assert ages["age_median"] == 5 and ages["age_max"] == 6


def test_stream_queue_base_slots(tmp_path):
    """Base-play chunks ("b-", marker base): taken only when asked, no age limit, None (no wait) when too few."""
    for name, v, base in (("s-a", 9, False), ("s-b", 8, False), ("b-a", -1, True), ("b-b", -1, True), ("b-c", -1, True)):
        (tmp_path / f"{name}.pt").write_bytes(b"x")
        (tmp_path / f"{name}.ready").write_text(json.dumps({"version": v, "base": base}))
    q = heal_kd.StreamQueue(tmp_path, 0, max_age=6, stall=0.0, poll=0.0)
    got = q.take(2, 10, base=True)
    assert got is not None and all(Path(x).name.startswith("b-") for x in got) and q.dropped == 0
    assert q.take(2, 10, base=True) is None  # one base chunk left: no wait
    assert sorted(Path(x).name for x in q.take(2, 10)) == ["s-a.pt", "s-b.pt"]  # base never in a student slot


def session(n, seed=0, k=4):
    """One scored session's full rows (a fresh pass): ids, reply, teacher rows, realised-token and behavior log p."""
    g = torch.Generator().manual_seed(seed)
    reply = torch.zeros(n, dtype=torch.bool)
    for a in range(5, n, 23):  # alternating prompt / reply turns
        reply[a:a + 15] = True
    ids = torch.randint(10, 900, (n,), generator=g)
    ids[60:100] = torch.tensor([7, 8, 9, 11] * 10)  # a degenerate repeat inside the replies (the loop cut)
    return {"ids": ids, "reply": reply, "top_logp": -torch.rand(n, k, generator=g) * 4,
            "top_ids": torch.randint(10, 900, (n, k), generator=g).int(), "tail": -torch.rand(n, generator=g) - 2,
            "tok": -torch.rand(n, generator=g) * 3, "beh": -torch.rand(n, generator=g) * 3}


def write_pieces(folder, name, full, cuts, version=3, base=False, epoch_chain="c001"):
    """heal_stream's pieces of ``full`` fed up to each cut: piece p holds ids[:cut_p], rows [row0_p, cut_p) with row0_p
    = cut_{p-1} - 1 (the previous piece's last row, now with its next token); its last row has no next token."""
    paths, lo = [], 0
    for p, upto in enumerate(cuts):
        row0 = lo - 1 if lo else 0
        tok, beh = full["tok"][row0:upto].clone(), full["beh"][row0:upto].clone()
        tok[-1], beh[-1] = float("nan"), float("nan")
        stem = f"{'b' if base else 's'}-{name}--{epoch_chain}-p{p:04d}"
        torch.save({"ids": full["ids"][:upto].int(), "image": torch.zeros(upto, dtype=torch.bool),
                    "reply": full["reply"][:upto].clone(), "row0": row0, "top_logp": full["top_logp"][row0:upto].half(),
                    "top_ids": full["top_ids"][row0:upto], "tail": full["tail"][row0:upto].half(), "tok_logp": tok,
                    "beh_logp": beh, "version": version}, folder / f"{stem}.pt")
        (folder / f"{stem}.ready").write_text(json.dumps({"version": version, "base": base, "tokens": upto,
                                                          "rows": upto - row0,
                                                          "reply_rows": int(full["reply"][row0 + 1:upto].sum())}))
        paths.append(str(folder / f"{stem}.pt"))
        lo = upto
    return paths


def piece_cpu(path):
    cpu = heal_kd.load_teacher_cpu(path)
    cut, loops = heal_kd.truncate_loops({"ids": cpu["inputs"]["input_ids"][0], "reply": cpu["reply"]})
    cpu["reply"], cpu["loops"] = cut["reply"], [float(x) for x in loops]
    return cpu


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(torch.nan_to_num(a.float(), nan=-7.0),
                                                   torch.nan_to_num(b.float(), nan=-7.0)))


def test_coalesced_pieces_equal_the_pieces_rows(tmp_path):
    """S2: a chain's merged pieces carry exactly the rows each piece trains alone (teacher rows, KD mask, per-piece
    loop-cut reply, labels, realised-token and behavior log p: the samp / IS rows), and the fresh pass's rows."""
    full = session(700)
    full["reply"][350:650] = True  # a 300-token reply looping from 400 on (loop_mask needs >= 256 reply tokens)
    full["ids"][400:650] = torch.tensor([7, 8, 9, 11] * 63)[:250]
    paths = write_pieces(tmp_path, "g", full, [150, 330, 700])
    cpus = [piece_cpu(p) for p in paths]
    merged = heal_kd.merge_pieces(cpus)
    assert merged["row0"] == 0 and merged["pieces"] == 3 and merged["inputs"]["input_ids"].shape[1] == 700
    for c, hi in zip(cpus, [149, 329, 700]):
        lo = c["row0"]
        for i in range(3):
            assert same(merged["teacher"][i][lo:hi], c["teacher"][i][lo:hi])
        for key in ("mask", "reply", "labels", "t_tok", "b"):
            assert same(merged[key][lo:hi], c[key][lo:hi]), key
    assert merged["loops"] == [sum(c["loops"][j] for c in cpus) for j in range(3)] and merged["loops"][2] > 0
    whole = tmp_path / "whole"
    whole.mkdir()
    one = heal_kd.load_teacher_cpu(write_pieces(whole, "g", full, [700])[0])
    assert same(merged["teacher"][0], one["teacher"][0]) and same(merged["mask"], one["mask"])
    assert same(merged["t_tok"], one["t_tok"]) and same(merged["b"], one["b"]) and same(merged["labels"], one["labels"])
    data = heal_kd.teacher_to(merged, "cpu")
    samp = heal_kd.slot_samp(data, paths[-1], "s-")
    alone = sum(int(heal_kd.slot_samp(heal_kd.teacher_to(c, "cpu"), p, "s-")[3].sum()) for c, p in zip(cpus, paths))
    assert samp is not None and int(samp[3].sum()) == alone > 0


def test_stream_queue_coalesces_chains(tmp_path):
    """S2 picks: a chain's linked pieces form one unit; a broken link (ids not a prefix) stays separate and is
    counted; ages and versions stay per piece; a stale piece is dropped before merging."""
    a = write_pieces(tmp_path, "ga", session(200, 1), [64, 128, 200], version=9)
    b = write_pieces(tmp_path, "gb", session(150, 2), [80, 150], version=9)
    broken = session(150, 3)
    c = write_pieces(tmp_path, "gc", broken, [70, 150], version=9)
    torch.save(dict(torch.load(c[1]), ids=torch.cat([broken["ids"][:69] + 1, broken["ids"][69:150]]).int()), c[1])
    old = write_pieces(tmp_path, "gd", session(100, 4), [50, 100], version=1)
    for paths, t in ((a, 300), (b, 200), (c, 100), (old, 50)):  # marker times: a newest
        for path in paths:
            os.utime(Path(path).with_suffix(".ready"), (t, t))
    q = heal_kd.StreamQueue(tmp_path, 0, max_age=6, stall=0.0, poll=0.0, coalesce=True)
    got = q.take(2, 10)
    assert q.dropped == 2 and all(p not in sum(got, []) for p in old)
    assert sorted(map(len, got)) == [2, 3] and a in got and b in got  # nearest sizes paired: a (200) with b (150)
    assert all(q.versions[p] == 9 for p in a + b)
    rest = q.take(2, 10)
    assert sorted(rest) == sorted([[c[0]], [c[1]]]) and q.stats["breaks"] == 1
    assert q.stats["units"] == 4 and q.stats["pieces"] == 7
    ages = heal_kd.stream_ages(10, [q.versions[p] for p in sum(got + rest, [])], q)
    assert ages["stream_chunks"] == 7 and ages["age_max"] == 1


def test_stream_queue_coalesce_off_is_unchanged(tmp_path):
    write_pieces(tmp_path, "ga", session(200, 1), [64, 128, 200], version=9)
    q = heal_kd.StreamQueue(tmp_path, 0, max_age=6, stall=0.0, poll=0.0)
    got = q.take(2, 10)
    assert all(isinstance(p, str) for p in got) and len(got) == 2


def test_coalesce_bounds_the_batch_and_the_base_share(tmp_path):
    """S2 with HEAL_STREAM_BASE_SHARE: picks counted in pieces; a student unit is cut to the step's remaining rows
    per rank and a base unit to the pieces still due, so the batch and the base share overshoot by at most one
    piece per rank; the leftover pieces come later as their own unit."""
    for i in range(6):
        write_pieces(tmp_path, f"s{i}", session(400, 10 + i), [100, 200, 300, 400], version=9)
        write_pieces(tmp_path, f"b{i}", session(400, 20 + i), [100, 200, 300, 400], version=-1, base=True)
    q = heal_kd.StreamQueue(tmp_path, 0, max_age=6, stall=0.0, poll=0.0, coalesce=True)
    rows = {p: m["reply_rows"] for p, m in ((str(m.with_suffix(".pt")), json.loads(m.read_text()))
                                            for m in tmp_path.glob("*.ready"))}
    world, share, batch = 2, 0.5, 260
    counts, unit_of, got_rows, steps = [0, 0], {}, 0.0, []
    base_pieces = student_pieces = 0
    for _ in range(8):
        streamed, base = heal_kd.stream_pick(q, world, 10, share, counts, unit_of, room=batch - got_rows)
        pieces = sum(len(unit_of[x]) for x in streamed)
        if base:
            base_pieces += pieces
        else:
            student_pieces += pieces
            got_rows += sum(rows[p] for x in streamed for p in unit_of[x])
            max_piece = max(rows[p] for x in streamed for p in unit_of[x])
            if got_rows >= batch:
                assert got_rows - batch <= world * max_piece  # the step's overshoot: at most a piece per rank
                steps.append(got_rows)
                got_rows = 0.0
        assert base_pieces - share / (1 - share) * max(1, student_pieces) <= world  # never past the share + a piece/rank
    assert counts == [base_pieces, student_pieces] and steps
    assert q.stats["trimmed"] > 0 and any(len(u) > 1 for u in unit_of.values())


def gptq_dequant(qweight, qzeros, scales, group=128):
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    codes = ((qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).reshape(-1, qweight.shape[1])
    zeros = ((qzeros.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF).reshape(qzeros.shape[0], -1) + 1
    g = codes.shape[0] // group
    w = (codes.reshape(g, group, -1).float() - zeros.reshape(g, 1, -1).float()) * scales.float().reshape(g, 1, -1)
    return w.reshape(codes.shape).t()


def test_int4_round_trip():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    qw, qz, sc = heal_delta.int4_rtn(w, search=20)
    assert qw.shape == (32, 64) and qz.shape == (2, 8) and sc.shape == (2, 64)
    back = gptq_dequant(qw, qz, sc)
    rel = float((back - w).norm() / w.norm())
    assert rel < 0.12, rel
    worse = gptq_dequant(*heal_delta.int4_rtn(w, search=0))
    assert float((worse - w).norm()) >= float((back - w).norm()) - 1e-7  # the scale search never hurts


def test_entries_served_names(tmp_path):
    prune = {"format": "arc-expert-prune-v1", "experts": 8, "pruned": {str(l): [0, 2, 5, 7] for l in range(4)}}
    (tmp_path / "heal_prune-t50-excl5-v1.json").write_text(json.dumps(prune))
    cut = heal_delta.Cut({"student_cut": {"source_layers": [0, 1, 3], "prune_file": str(tmp_path / "heal_prune-t50-excl5-v1.json")}})
    fused = torch.randn(4, 2 * 128, 256) * 0.02  # [kept experts, 2I, H]
    got = dict(heal_delta.entries("model.language_model.layers.2.mlp.experts.gate_up_proj", fused, cut, rows=[1, 3],
                                  search=0))
    # student layer 2 = served layer 3; kept experts [1, 3, 4, 6]: rows 1, 3 -> original 3, 6
    assert sorted({k.split(".mlp.experts.")[1].split(".")[0] for k in got}) == ["3", "6"]
    assert "model.language_model.layers.3.mlp.experts.3.gate_proj.qweight" in got
    router = dict(heal_delta.entries("model.language_model.layers.0.mlp.gate.weight", torch.ones(4, 16), cut))
    r = router["model.language_model.layers.0.mlp.gate.weight"]
    assert r.shape == (8, 16) and r[[1, 3, 4, 6]].eq(1).all() and r[[0, 2, 5, 7]].eq(0).all()
    dense = dict(heal_delta.entries("model.language_model.layers.1.self_attn.o_proj.weight", torch.ones(4, 4), cut))
    assert list(dense) == ["model.language_model.layers.1.self_attn.o_proj.weight"] and dense[list(dense)[0]].dtype == torch.bfloat16
    assert heal_delta.entries("mtp.layers.0.mlp.experts.gate_up_proj", fused, cut) == []
    assert heal_delta.entries("model.language_model.layers.1.ple.key_proj.weight", torch.ones(2, 2), cut) == []


def test_exporter_writes_manifest_last(tmp_path):
    cut = heal_delta.Cut({"student_cut": {"source_layers": [0]}})
    runner = torch.nn.Module()
    runner.model = torch.nn.Module()
    runner.model.w = torch.nn.Parameter(torch.ones(3, 3))
    # Runner names "model.<checkpoint name>": fake one dense tensor under a served name
    runner = type("R", (), {"named_parameters": lambda self: [("model.model.language_model.norm.weight",
                                                                torch.nn.Parameter(torch.ones(4)))]})()
    ex = heal_delta.DeltaExporter(tmp_path / "deltas", cut, 0, files=2)
    info = ex.export(runner, None, 5)
    ex.wait()
    assert info["tensors"] == 1
    man = json.loads((tmp_path / "deltas/v5/manifest.json").read_text())
    assert man["version"] == 5 and man["files"] and not list((tmp_path / "deltas").glob(".v5.partial"))
    os.environ.pop("X", None)


def test_exporter_skips_to_the_newest_while_an_upload_runs(tmp_path):
    """c6 (23 Oct): a 228 s upload held the next step's export. While the previous version is still being written or
    uploaded, a step exports nothing; the next free step exports its own (newest) weights; force waits."""
    import threading

    cut = heal_delta.Cut({"student_cut": {"source_layers": [0]}})
    w = torch.nn.Parameter(torch.ones(4))
    runner = type("R", (), {"named_parameters": lambda self: [("model.model.language_model.norm.weight", w)]})()
    ex = heal_delta.DeltaExporter(tmp_path / "deltas", cut, 0, files=2)
    gate = threading.Event()
    real = ex._write

    def slow_write(step, host, began, meta=None):  # the upload of v1 takes a while
        if step == 1:
            gate.wait(10)
        return real(step, host, began, meta)

    ex._write = slow_write
    assert ex.export(runner, None, 1).get("tensors") == 1
    with torch.no_grad():
        w.fill_(2.0)
    assert ex.export(runner, None, 2) == {"step": 2, "skipped": True}  # v1 still uploading: no gather, no v2
    with torch.no_grad():
        w.fill_(3.0)
    gate.set()
    ex.writer.join()
    assert ex.export(runner, None, 3).get("tensors") == 1  # free again: the newest weights as v3
    ex.wait()
    assert not (tmp_path / "deltas/v2").exists()
    from safetensors.torch import load_file
    got = load_file(str(next((tmp_path / "deltas/v3").glob("delta-*.safetensors"))))
    assert float(next(iter(got.values())).float().mean()) == 3.0
    gate.clear()
    ex._write = slow_write
    ex.export(runner, None, 1)  # busy again
    threading.Timer(0.2, gate.set).start()
    assert ex.export(runner, None, 4, force=True).get("tensors") == 1  # force waits for the writer
    ex.wait()



def test_skipped_step_rows_reach_the_next_version(tmp_path):
    """Expert rows trained only on a skipped step are in the next exported version (fused tensors export only the
    rows trained so far)."""
    import threading

    cut = heal_delta.Cut({"student_cut": {"source_layers": [0]}})
    w = torch.nn.Parameter(torch.randn(4, 8, 8))
    runner = type("R", (), {"named_parameters": lambda self: [("model.model.language_model.layers.0.mlp.experts.down_proj", w)]})()
    ex = heal_delta.DeltaExporter(tmp_path / "deltas", cut, 0, files=2)
    gate, real = threading.Event(), ex._write

    def slow_write(step, host, began, meta=None):
        if step == 1:
            gate.wait(10)
        return real(step, host, began, meta)

    ex._write = slow_write
    rows = lambda r: type("Rows", (), {"items": [{"p": w, "rows": torch.tensor(r), "start": 0}]})()  # noqa: E731
    ex.export(runner, rows([0]), 1)
    assert ex.export(runner, rows([3]), 2) == {"step": 2, "skipped": True}  # row 3 trains only here
    gate.set()
    ex.writer.join()
    ex.export(runner, rows([1]), 3)
    ex.wait()
    from safetensors.torch import load_file
    got = {}
    for f in (tmp_path / "deltas/v3").glob("delta-*.safetensors"):
        got.update(load_file(str(f)))
    experts = {int(k.split(".mlp.experts.")[1].split(".")[0]) for k in got}
    assert experts == {0, 1, 3}

# --- streaming teacher (heal_stream) ------------------------------------------------------------------------

import heal_stream  # noqa: E402

V, IMG = 50, 49


def fake_encode(capture):
    """Messages hold space-separated ids; the prompt is their concatenation, the reply follows."""
    ids = [int(t) for m in capture["payload"]["messages"] for t in m["content"].split()]
    start = len(ids)
    ids += [int(t) for t in capture["reply"].split()]
    tensor = torch.tensor([ids])
    inputs = {"input_ids": tensor, "mm_token_type_ids": (tensor == IMG).int()}
    runs = heal_stream.image_runs(tensor[0], IMG)
    if runs:  # one 1x2x(n/... ) grid per image: 4 pixel rows per image token (merge 2)
        inputs["image_grid_thw"] = torch.tensor([[1, 2, 2 * (b - a)] for a, b in runs])
        rows = int(inputs["image_grid_thw"].prod(-1).sum())
        inputs["pixel_values"] = torch.arange(rows, dtype=torch.float32)[:, None]
    return inputs, start


def fake_forward(emb, sub=7, calls=None):
    def forward(chain, lo, hi, keep):
        if lo == 0:
            chain.cache = (torch.zeros(V), 0)
        state, length = chain.cache
        assert length == lo, (length, lo)
        ids = chain.ids
        for a in range(lo, hi, sub):
            b = min(a + sub, hi)
            rows = []
            for t in range(a, b):
                state = state * 0.9 + emb[int(ids[t])]
                rows.append(state.clone())
            chain.cache = (state, b)
            if calls is not None:
                calls.append((a, b, keep))
            yield a, b, (torch.stack(rows) if keep else None)
    return forward


def write_rows(path, rows):
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def game_rows(rng, n_requests, drain_at=None, version0=0):
    """Rollout rows of one session: each request = the previous request + its reply + a new user message (a
    drain restarts the messages)."""
    rows, messages, v, last_reply = [], [], version0, None
    for n in range(n_requests):
        drain = drain_at is not None and n == drain_at
        new_user = {"role": "user", "content": " ".join(str(int(t)) for t in torch.randint(1, 40, (int(rng.integers(5, 30)),)))}
        if drain or last_reply is None:
            keep, add = 0, [{"role": "system", "content": "1 2 3"}, new_user]
            messages = add
        else:  # the delta: the previous request's messages kept, then its reply and the new message
            keep, add = len(messages), [{"role": "assistant", "content": last_reply}, new_user]
            messages = messages + add
        if n % 3 == 1:  # an image in some prompts
            add[-1]["content"] += " 7 " + " ".join([str(IMG)] * 3) + " 8"
        reply_ids = [int(t) for t in torch.randint(1, 40, (int(rng.integers(3, 25)),))]
        reply = " ".join(map(str, reply_ids))
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        rows.append({"n": n, "keep": keep, "messages": add, "reply": reply, "gameplay": True, "template": {},
                     "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(reply_ids)},
                     "tokens": reply_ids, "logprobs": [-0.01 * (i + 1) for i in range(len(reply_ids))],
                     "version": v})
        last_reply = reply
        if n % 4 == 3:
            v += 1
    return rows


def pump(t, rounds=200):
    for _ in range(rounds):
        t._scan()
        while not t.events.empty():
            t._handle(t.events.get_nowait())
        chain, upto = t._due()
        if chain is None:
            return
        t.score(chain, upto)
        if t.budget is not None:
            t.stats["evictions"] += t.budget([c for c in t.chains.values() if c is not chain])


def run_stream(tmp_path, evict, chunk=20, base=(), encode=None):
    import numpy as np

    torch.manual_seed(0)
    emb = torch.randn(V, V)
    root, out = tmp_path / "roll", tmp_path / "out"
    for g in ("g1/a0", "g2/a0"):
        (root / g).mkdir(parents=True)
    rows1 = game_rows(np.random.default_rng(1), 12, drain_at=7)
    rows2 = game_rows(np.random.default_rng(2), 9)
    t = heal_stream.StreamTeacher(root, out, encode or fake_encode, fake_forward(emb), k=8, chunk=chunk,
                                  image_token_id=IMG, idle=1e9, encoders=2,
                                  budget=heal_stream.lru_budget(lambda: True) if evict else None,
                                  base=base)
    write_rows(root / "g1/a0/rollout.jsonl", rows1[:5])
    write_rows(root / "g2/a0/rollout.jsonl", rows2[:4])
    pump(t)
    write_rows(root / "g1/a0/rollout.jsonl", rows1[5:])
    write_rows(root / "g2/a0/rollout.jsonl", rows2[4:])
    pump(t)
    for key in list(t.chains):  # the games end: the open chains close
        t._handle(("idle", key, None, None))
    pump(t)
    t.saver.shutdown(wait=True)
    assert t.error is None
    return t, emb, out


def reference_rows(emb, ids):
    state, rows = torch.zeros(V), []
    for tok in ids.tolist():
        state = state * 0.9 + emb[tok]
        rows.append(state.clone())
    full = torch.log_softmax(torch.stack(rows), -1)
    return full


@pytest.mark.parametrize("evict", [False, True])
def test_stream_teacher_pieces_match_one_pass(tmp_path, evict):
    t, emb, out = run_stream(tmp_path, evict)
    assert t.stats["breaks"] == 1 and t.stats["unaligned"] == 0 and t.stats["unversioned"] == 0
    if evict:
        assert t.stats["recompute_tokens"] > 0 and t.stats["evictions"] > 0
    pieces = {}
    for f in sorted(out.glob("s-*.pt")):
        d = torch.load(f, weights_only=False)
        assert (out / f"{f.stem}.ready").is_file()
        d["path"] = f
        pieces.setdefault((d["segment"], d["chain"]), []).append(d)
    assert len(pieces) == 3  # g1: 2 chains (a drain), g2: 1
    for (seg, ch), ds in pieces.items():
        ds.sort(key=lambda d: d["piece"])
        ids = ds[-1]["ids"].long()
        full = reference_rows(emb, ids)
        covered = torch.zeros(ids.numel(), dtype=torch.bool)
        for d in ds:
            n, row0 = d["ids"].numel(), d["row0"]
            assert torch.equal(d["ids"].long(), ids[:n])  # each piece's ids prefix the chain's
            assert d["top_logp"].shape[0] == n - row0
            ref = full[row0:n]
            assert torch.allclose(d["top_logp"].float(), ref.topk(8, -1).values, atol=1e-2)
            tok_ref = ref[:-1].gather(1, ids[row0 + 1:n, None])[:, 0]
            assert torch.allclose(d["tok_logp"][:-1], tok_ref, atol=1e-5) and torch.isnan(d["tok_logp"][-1])
            # rows row0..n-2 are trained here; row n-1 waits for the next piece
            assert not covered[row0:n - 1].any()
            covered[row0:n - 1] = True
            # no image is cut: the piece's pixel rows are those of its whole images
            runs = heal_stream.image_runs(d["ids"].long(), IMG)
            assert all(b <= n for _, b in runs)
            if runs:
                assert d["image_grid_thw"].shape[0] == len(runs)
                assert d["pixel_values"].shape[0] == int(d["image_grid_thw"].prod(-1).sum())
            # the engine log p sits on the row before each reply token
            b = d["beh_logp"]
            on = d["reply"][row0 + 1:n]
            assert torch.isfinite(b[:-1][on]).all() and torch.isnan(b[:-1][~on]).all()
            cpu = heal_kd.load_teacher_cpu(d["path"])
            assert cpu["version"] == d["version"]
            r = cpu["reply"].bool()
            assert not r[:row0].any() and torch.isfinite(cpu["b"][r]).all()
        assert covered[:ids.numel() - 1].all()  # every row of the chain trained exactly once


def test_process_encoder_pieces_equal_thread_encode(tmp_path):
    """Teacher host fix: the encode in spawned workers writes the same pieces as the in-process threads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import stream_encode_toy

    enc = heal_stream.ProcessEncoder(stream_encode_toy.init, stream_encode_toy.encode, args=("toy",), workers=2)
    try:
        t_p, _, out_p = run_stream(tmp_path / "p", False, encode=enc)
    finally:
        enc.shutdown()
    t_t, _, out_t = run_stream(tmp_path / "t", False)
    names = sorted(f.name for f in out_t.glob("*.pt"))
    assert names and names == sorted(f.name for f in out_p.glob("*.pt"))
    for name in names:
        a, b = torch.load(out_t / name, weights_only=False), torch.load(out_p / name, weights_only=False)
        for key in ("ids", "top_ids", "top_logp", "tok_logp", "reply"):
            assert torch.equal(a[key], b[key]) or (key == "tok_logp" and torch.allclose(a[key], b[key], equal_nan=True))
    assert t_p.stats["captures"] == t_t.stats["captures"] and t_p.stats["encode_seconds"] > 0


def test_process_encoder_reports_errors_as_values():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import stream_encode_toy

    enc = heal_stream.ProcessEncoder(stream_encode_toy.init, stream_encode_toy.encode, args=("toy",), workers=1)
    try:
        made, seconds = enc.submit({"payload": {"messages": [{"content": "x"}]}, "reply": "1"}).result(timeout=120)
    finally:
        enc.shutdown()
    assert isinstance(made, RuntimeError) and "ValueError" in str(made) and seconds >= 0


def test_stream_versions_and_reply_marks(tmp_path):
    t, emb, out = run_stream(tmp_path, False, chunk=1000)  # one piece per chain
    for f in out.glob("s-*.pt"):
        d = torch.load(f, weights_only=False)
        marker = json.loads((out / f"{f.stem}.ready").read_text())
        assert marker["version"] == d["version"] and d["version_min"] <= d["version"]
        assert int(d["reply"].sum()) == marker["reply_rows"] + int(d["reply"][0])


def test_session_tail_refuses_rewrite(tmp_path):
    p = tmp_path / "rollout.jsonl"
    import numpy as np
    rows = game_rows(np.random.default_rng(0), 4)
    write_rows(p, rows[:2])
    tail = heal_stream.SessionTail(p)
    assert len(tail.read()) == 2
    full = p.read_text() + json.dumps(rows[2]) + "\n" + json.dumps(rows[3])[:10]  # a replaced, longer copy; a partial line
    p.write_text(full)
    got = tail.read()
    assert [c["request"] for c in got] == [2] and got[0]["rollout"]["version"] == rows[2]["version"]
    p.write_text("x" * (len(full) + 5))
    with pytest.raises(heal_stream.Rewritten, match="rewritten"):
        tail.read()


def test_session_tail_waits_out_a_shrunk_copy(tmp_path, monkeypatch):
    """opdS-g1h: a shorter stale copy of a session must not kill the teacher; it is read on once it grows back."""
    import numpy as np
    p = tmp_path / "rollout.jsonl"
    rows = game_rows(np.random.default_rng(0), 4)
    write_rows(p, rows[:3])
    tail = heal_stream.SessionTail(p)
    assert len(tail.read()) == 3
    whole = p.read_text()
    p.write_text(whole[: len(whole) // 3])  # the stale copy
    assert tail.read() == []
    p.write_text(whole + json.dumps(rows[3]) + "\n")  # grown back past the offset, same bytes
    assert [c["request"] for c in tail.read()] == [3]
    p.write_text(whole[:10])
    assert tail.read() == []
    monkeypatch.setattr(heal_stream, "SHRINK_WAIT", 0.0)
    with pytest.raises(heal_stream.Rewritten, match="shrank"):
        tail.read()


def test_stream_teacher_survives_rewrite_and_shrink(tmp_path):
    """A rewritten session (a new game in the same file) closes its chain and starts a new one; a shrunk copy waits;
    the teacher keeps scoring (no error), and the new session's pieces are written."""
    import numpy as np
    torch.manual_seed(0)
    emb = torch.randn(V, V)
    root, out = tmp_path / "roll", tmp_path / "out"
    (root / "g1/a0").mkdir(parents=True)
    rows = game_rows(np.random.default_rng(1), 10)
    t = heal_stream.StreamTeacher(root, out, fake_encode, fake_forward(emb), k=8, chunk=20, image_token_id=IMG,
                                  idle=1e9, encoders=2)
    f = root / "g1/a0/rollout.jsonl"
    write_rows(f, rows[:6])
    pump(t)
    whole = f.read_text()
    f.write_text(whole[:200])  # shrunk copy: waits
    pump(t)
    assert t.error is None and t.stats.get("rewrites", 0) == 0
    other = game_rows(np.random.default_rng(7), 5)
    f.write_text("".join(json.dumps(r) + "\n" for r in other))  # a new session in the same file
    heal_stream.SHRINK_WAIT, keep = 0.0, heal_stream.SHRINK_WAIT
    try:
        pump(t)
    finally:
        heal_stream.SHRINK_WAIT = keep
    for key in list(t.chains):
        t._handle(("idle", key, None, None))
    pump(t)
    t.saver.shutdown(wait=True)
    assert t.error is None and t.stats["rewrites"] == 1 and t.stats["closed"] >= 2
    assert len(list(out.glob("*.pt"))) >= 2


def test_cut_point_keeps_images_whole():
    ids = torch.tensor([1, 2, IMG, IMG, IMG, 3, 4])
    assert heal_stream.cut_point(ids, 3, 0, IMG) == 2
    assert heal_stream.cut_point(ids, 4, 2, IMG) == 5  # nothing left before the image: past it
    assert heal_stream.cut_point(ids, 5, 0, IMG) == 5


def test_stream_verify_matches_one_pass(tmp_path):
    t, emb, out = run_stream(tmp_path, False, chunk=20)
    t.saver = __import__("concurrent.futures").futures.ThreadPoolExecutor(1)
    chain = max(t.chains.values(), key=lambda c: len(c.files))
    assert len(chain.files) >= 2
    got = t.verify(chain)
    assert got["top_logp_mean_abs"] < 2e-3 and got["top1_agree"] == 1.0 and got["tok_logp_max_abs"] < 1e-5


def test_stream_skip_existing_scores_only_new_tokens(tmp_path):
    import numpy as np

    torch.manual_seed(0)
    emb = torch.randn(V, V)
    root, out = tmp_path / "roll", tmp_path / "out"
    (root / "g1/a0").mkdir(parents=True)
    rows = game_rows(np.random.default_rng(3), 10)
    write_rows(root / "g1/a0/rollout.jsonl", rows[:6])
    t = heal_stream.StreamTeacher(root, out, fake_encode, fake_forward(emb), k=8, chunk=20, image_token_id=IMG, idle=1e9)
    t.skip_existing()
    chain = t.chains["g1/a0"]
    seen = chain.fed
    assert seen == chain.ids.shape[0] and chain.cache is None and t.stats["skipped_tokens"] == seen
    write_rows(root / "g1/a0/rollout.jsonl", rows[6:])
    pump(t)
    t._handle(("idle", "g1/a0", None, None))
    pump(t)
    t.saver.shutdown(wait=True)
    pieces = sorted((torch.load(f, weights_only=False) for f in out.glob("s-*.pt")), key=lambda d: d["piece"])
    assert pieces and pieces[0]["row0"] == seen  # nothing before the restart point is trained again
    full = reference_rows(emb, pieces[-1]["ids"].long())
    assert torch.allclose(pieces[0]["top_logp"].float(), full[seen:pieces[0]["ids"].numel()].topk(8, -1).values, atol=1e-2)


def test_stream_bars(tmp_path):
    ev, st = tmp_path / "events.jsonl", tmp_path / "status.json"
    bars = {"step_seconds": 230, "teacher_tok_per_s": 2960, "teacher_after": 1000}
    rows = [{"event": "step", "step": 1, "seconds": 400, "stream_wait_s": 300},
            {"event": "delta_export", "step": 1, "gather_seconds": 5},
            {"event": "step", "step": 2, "seconds": 650, "stream_wait_s": 330},
            {"event": "delta_export", "step": 2, "gather_seconds": 5}]
    ev.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert heal_stream.stream_bars(ev, st, bars)[0] is None  # two steps, no teacher yet
    with ev.open("a") as f:
        f.write(json.dumps({"event": "step", "step": 3, "seconds": 860, "stream_wait_s": 330}) + "\n")
    st.write_text(json.dumps({"tokens": 5000, "tok_per_s": 4000.0}))
    verdict, read = heal_stream.stream_bars(ev, st, bars)
    assert verdict == "pass" and read["step_compute"] == 225.0  # 250 - 30 wait + 5 gather
    st.write_text(json.dumps({"tokens": 5000, "tok_per_s": 2000.0}))
    assert heal_stream.stream_bars(ev, st, bars)[0].startswith("fail: teacher")


def test_stream_base_pieces(tmp_path):
    """Pieces of a base play run (a root subfolder in ``base``) are "b-" files with marker base=True."""
    t, emb, out = run_stream(tmp_path, False, chunk=1000, base=("g2",))
    names = sorted(f.name for f in out.glob("*.pt"))
    assert any(n.startswith("b-g2") for n in names) and any(n.startswith("s-g1") for n in names)
    for f in out.glob("b-*.ready"):
        assert json.loads(f.read_text())["base"] is True
    for f in out.glob("s-*.ready"):
        assert json.loads(f.read_text())["base"] is False


def test_stream_fill_copies_matching_sessions(tmp_path):
    for name in ("b-k000", "b-k001", "k000"):
        (tmp_path / "src" / name).mkdir(parents=True)
        (tmp_path / "src" / name / "rollout.jsonl").write_text(name + "\n")
    root = heal_stream.stream_fill(str(tmp_path / "src" / "b-*"), tmp_path / "roll")
    got = sorted(p.parent.name for p in (tmp_path / "roll" / root).glob("*/rollout.jsonl"))
    assert root == "fill" and got == ["b-k000", "b-k001"]
    with pytest.raises(SystemExit):
        heal_stream.stream_fill(str(tmp_path / "src" / "none-*"), tmp_path / "roll")


def test_restarted_teacher_never_reuses_piece_names(tmp_path):
    import numpy as np

    torch.manual_seed(0)
    emb = torch.randn(V, V)
    root, out = tmp_path / "roll", tmp_path / "out"
    (root / "g1/a0").mkdir(parents=True)
    rows = game_rows(np.random.default_rng(3), 10)
    write_rows(root / "g1/a0/rollout.jsonl", rows[:6])
    t = heal_stream.StreamTeacher(root, out, fake_encode, fake_forward(emb), k=8, chunk=20, image_token_id=IMG, idle=1e9)
    pump(t)
    t._handle(("idle", "g1/a0", None, None))
    pump(t)
    t.saver.shutdown(wait=True)
    first = {f.name: f.stat().st_mtime_ns for f in out.glob("s-*.ready")}
    assert first
    t2 = heal_stream.StreamTeacher(root, out, fake_encode, fake_forward(emb), k=8, chunk=20, image_token_id=IMG, idle=1e9)
    t2.skip_existing()
    write_rows(root / "g1/a0/rollout.jsonl", rows[6:])
    pump(t2)
    t2._handle(("idle", "g1/a0", None, None))
    pump(t2)
    t2.saver.shutdown(wait=True)
    new = {f.name for f in out.glob("s-*.ready")} - set(first)
    assert new  # the restart's pieces get names of their own
    assert all(f.stat().st_mtime_ns == first[f.name] for f in out.glob("s-*.ready") if f.name in first)  # none overwritten



def test_teacher_scores_student_before_base_backlog(tmp_path):
    import numpy as np

    torch.manual_seed(0)
    emb = torch.randn(V, V)
    root, out = tmp_path / "roll", tmp_path / "out"
    rows = game_rows(np.random.default_rng(3), 10)
    for key in ("a-base/s0", "z-play/s0"):  # the base session sorts (and is seen) first
        (root / key).mkdir(parents=True)
        write_rows(root / key / "rollout.jsonl", rows)
    t = heal_stream.StreamTeacher(root, out, fake_encode, fake_forward(emb), k=8, chunk=20, image_token_id=IMG,
                                  idle=1e9, base=("a-base",))
    t._scan()
    while not t.events.empty():
        t._handle(t.events.get_nowait())
    t._handle(("idle", "a-base/s0", None, None))  # a finished base session: a closed chain
    order = []
    for _ in range(400):
        chain, upto = t._due()
        if chain is None:
            break
        order.append(chain.key.split("/")[0])
        t.score(chain, upto)
    t.saver.shutdown(wait=True)
    assert order and order[0] == "z-play" and "a-base" in order  # student first; base still scored with room
    assert order.index("a-base") > max(i for i, k in enumerate(order) if k == "z-play") or \
        all(k == "z-play" for k in order[:order.index("a-base")])


def test_live_teacher_scores_a_student_piece_first_behind_a_base_backlog(tmp_path):
    """g1k (23 Oct): 2,532 base captures in sessions that sort before the student's were encoded and queued first;
    0 student pieces in 20 min. The threaded run() path, as the round runs it."""
    import time

    import numpy as np

    torch.manual_seed(0)
    emb = torch.randn(V, V)
    root, out = tmp_path / "roll", tmp_path / "out"
    rows = game_rows(np.random.default_rng(5), 25)
    base_rows = [{k: v for k, v in r.items() if k not in ("tokens", "logprobs")} for r in rows]  # base play: no ids
    for i in range(60):  # 1,500 base captures, sorted first
        (root / f"a-base/s{i:03d}").mkdir(parents=True)
        write_rows(root / f"a-base/s{i:03d}/rollout.jsonl", base_rows)
    (root / "z-play/s0").mkdir(parents=True)
    write_rows(root / "z-play/s0/rollout.jsonl", rows[:6])

    def slow_encode(capture):  # host encode cost (g1k: ~1.2 s a capture; here 5 ms)
        time.sleep(0.005)
        return fake_encode(capture)

    t = heal_stream.StreamTeacher(root, out, slow_encode, fake_forward(emb), k=8, chunk=20, image_token_id=IMG,
                                  idle=1e9, poll=0.05, encoders=2, base=("a-base",))
    first, began = {}, time.time()
    try:
        t.run(until=lambda t: bool(first) or time.time() - began > 30,
              after_piece=lambda t, chain: first.setdefault("key", chain.key) and first.setdefault("s", time.time() - began))
    except RuntimeError:  # the producer's cancelled encodes at the stop
        pass
    assert first.get("key", "").startswith("z-play/"), first
    assert first["s"] < 1.0, first  # 1,500 x 5 ms / 2 encoders = 3.75 s if the backlog comes first


def test_base_backlog_is_encoded_a_slice_per_scan_in_order(tmp_path):
    import numpy as np

    root, out = tmp_path / "roll", tmp_path / "out"
    rows = [{k: v for k, v in r.items() if k not in ("tokens", "logprobs")} for r in game_rows(np.random.default_rng(6), 40)]
    (root / "a-base/s0").mkdir(parents=True)
    write_rows(root / "a-base/s0/rollout.jsonl", rows)
    t = heal_stream.StreamTeacher(root, out, fake_encode, None, k=8, chunk=20, image_token_id=IMG, idle=0.0,
                                  base=("a-base",), base_per_scan=16)
    got = []
    for _ in range(3):
        t._scan()
        batch = []
        while not t.events.empty():
            batch.append(t.events.get_nowait())
        got.append(batch)
    assert [len([e for e in b if e[0] == "capture"]) for b in got] == [16, 16, 8]
    assert [e[2]["request"] for b in got for e in b if e[0] == "capture"] == list(range(40))
    assert not any(e[0] == "idle" for b in got[:2] for e in b) and any(e[0] == "idle" for e in got[2])  # idle once drained


def test_teacher_rate_leaves_out_the_first_piece(tmp_path):
    t = heal_stream.StreamTeacher(tmp_path / "roll", tmp_path / "out", fake_encode, None, k=8, chunk=20, image_token_id=IMG)
    t.stats.update(pieces=3, tokens=3000, forward_seconds=102.0, warm_seconds=100.0, warm_tokens=1000)
    assert t.status()["tok_per_s"] == 1000.0  # 2,000 tokens in the 2 s after the warm first piece
    t.stats.update(pieces=1, tokens=1000, forward_seconds=100.0)
    assert t.status()["tok_per_s"] == 10.0  # one piece: the plain mean


def test_student_session_with_noncanonical_splits_round_trips_to_server_ids(tmp_path):
    """A pruned student samples non-canonical splits (canonical "y" sampled as "41 42"); its server keeps the sampled
    ids in later prompts. The teacher's chain must hold exactly the server's ids: canonical new turns, sampled replies,
    each reply closed by the end token; served counts match at every request; no breaks, nothing dropped."""
    END = 48

    def encode(capture):  # canonical: every message's ids, an assistant turn closed by END; the reply + END
        ids = []
        for m in capture["payload"]["messages"]:
            ids += [int(t) for t in m["content"].split()] + ([END] if m["role"] == "assistant" else [])
        start = len(ids)
        ids += [int(t) for t in capture["reply"].split()] + [END]
        tensor = torch.tensor([ids])
        return {"input_ids": tensor, "mm_token_type_ids": (tensor == IMG).int()}, start

    torch.manual_seed(0)
    rng = __import__("numpy").random.default_rng(5)
    rows, server, messages, prev_reply, prev_sampled, prev_capped, prev_start = [], [], [], None, None, False, 0
    for n in range(8):
        user = [int(t) for t in torch.randint(1, 40, (int(rng.integers(4, 12)),))]
        new = [{"role": "user", "content": " ".join(map(str, user))}]
        if prev_reply is None:
            new = [{"role": "system", "content": "1 2 3"}] + new
            server = [1, 2, 3] + user
        else:
            new = [{"role": "assistant", "content": prev_reply}] + new
            if prev_capped:
                server = server[:prev_start] + [int(t) for t in prev_reply.split()] + [END] + user
            else:
                server = server + prev_sampled + [END] + user  # the server appends the sampled ids, then the new turn
        keep = len(messages)
        messages = messages + new
        canon = [int(t) for t in torch.randint(1, 40, (int(rng.integers(4, 10)),))]
        sampled = canon[:1] + [41, 42] + canon[2:] if n % 2 == 0 else list(canon)  # a split in every other reply
        capped = n == 4  # cut at the token cap: the server re-renders it, closed (canonical) in the next prompt
        rows.append({"n": n, "keep": keep, "messages": new, "reply": " ".join(map(str, canon)), "gameplay": True,
                     "template": {}, "usage": {"prompt_tokens": len(server), "completion_tokens": len(sampled)},
                     "tokens": sampled, "logprobs": [-0.1] * len(sampled), "version": 0})
        prev_reply, prev_sampled, prev_capped, prev_start = rows[-1]["reply"], sampled, capped, len(server)
        server = server + []  # (the prompt; the reply is appended at the next request)
    whole = server + prev_sampled + [END]
    root, out = tmp_path / "roll", tmp_path / "out"
    (root / "p/s0").mkdir(parents=True)
    write_rows(root / "p/s0/rollout.jsonl", rows)
    emb = torch.randn(V, V)
    t = heal_stream.StreamTeacher(root, out, encode, fake_forward(emb), k=8, chunk=1000, image_token_id=IMG, idle=1e9,
                                  end_id=END)
    t._scan()
    while not t.events.empty():
        t._handle(t.events.get_nowait())
    chain = t.chains["p/s0"]
    assert chain.ids.tolist() == whole  # the server's ids exactly, sampled splits included
    # the capped reply (n=4) ends its chain (the server rewrote it); the next chain starts from the exact prompt
    assert t.stats["breaks"] == 1 and t.stats.get("inexact", 0) == 0 and t.stats["unaligned"] == 0
    assert int(chain.reply.sum()) == sum(len(r["tokens"]) + 1 for r in rows[5:])  # its sampled replies (+ END)
    sampled_pos = chain.reply & (chain.ids != END)
    assert not bool(torch.isnan(chain.beh[sampled_pos]).any())  # engine log p on every sampled id (END has none)


def test_exact_inputs_refuses_a_rewritten_history():
    END = 48
    first = {"input_ids": torch.tensor([[1, 2, 3, 9, 9, END]])}
    ex, start = heal_stream.exact_inputs(first, 3, [9, 41, 42], END)
    assert ex["input_ids"][0].tolist() == [1, 2, 3, 9, 41, 42, END] and start == 3
    second = {"input_ids": torch.tensor([[1, 2, 5, 9, 9, END, 7, 4, 4, END]])}  # history changed (a drain)
    assert heal_stream.exact_inputs(second, 7, [4, 4], END, (first["input_ids"][0], ex)) is None


def test_fixed_quack_never_tunes_and_serves_the_untuned_default():
    """quack 0.6.1's Autotuner.__call__ shape: a key not in ``cache`` is benchmarked (prune_configs first)."""
    import types

    import rl1_trainer

    class AutotuneConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def all_kwargs(self):
            return self.kwargs

    class Autotuner:
        def __init__(self, fn, key, configs):
            self.fn, self.keys, self.configs, self.cache = fn, key, configs, {}
            self.arg_names = ["A", "B", "out", "dynamic_scheduler", "config"]
            self.cache_results = True

        def prune_configs(self, kwargs):
            return self.configs

        def check_disk_cache(self, key, configs, bench):
            bench()

        def __call__(self, *args, **kwargs):
            if len(self.configs) > 1:
                all_args = {**dict(zip(self.arg_names, args)), **kwargs}
                key = [str(all_args[k]) for k in self.keys if k in all_args]
                for arg in all_args.values():
                    if isinstance(arg, torch.Tensor):
                        key += [str(arg.shape), str([s if s in {0, 1} else 2 for s in arg.stride()]), str(arg.dtype)]
                key = tuple(key)
                if key not in self.cache:
                    self.check_disk_cache(key, self.prune_configs(kwargs), lambda: self.cache.setdefault(key, self.configs[1]))
                config = self.cache[key]
            else:
                config = self.configs[0]
            return self.fn(*args, **kwargs, **config.all_kwargs())

    mod = types.SimpleNamespace(Autotuner=Autotuner, AutotuneConfig=AutotuneConfig, __name__="fakequack.autotuner")
    used = []
    gemm = Autotuner(lambda A, B, out, dynamic_scheduler=False, config="unset": used.append(config),
                     ["dynamic_scheduler"], [AutotuneConfig(config="c0"), AutotuneConfig(config="c1")])
    assert rl1_trainer.fix_quack_autotune(mod)["patched"]
    assert rl1_trainer.fix_quack_autotune(mod).get("again")  # once per class
    for m in (81920, 78050, 9430):  # every routed-row count: no tune, the untuned default
        gemm(torch.zeros(m, 4), torch.zeros(2, 4, 3), torch.zeros(m, 3), dynamic_scheduler=False)
    assert used == [None, None, None] and not gemm.cache
    with pytest.raises(rl1_trainer.QuackAutotuneInJob):
        gemm.prune_configs({})
    with pytest.raises(rl1_trainer.QuackAutotuneInJob):
        gemm.check_disk_cache((), [], lambda: None)
    assert rl1_trainer.fix_quack_autotune()["patched"] in (True, False)  # the real import: patched or absent, no error


FAKE_AUTOTUNER = """
class AutotuneConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def all_kwargs(self):
        return self.kwargs


class Autotuner:
    def __init__(self, fn, key, configs):
        self.fn, self.keys, self.configs, self.cache, self.arg_names = fn, key, configs, {}, ["A", "config"]

    def prune_configs(self, kwargs):
        return self.configs

    def check_disk_cache(self, key, configs, bench):
        bench()

    def __call__(self, *args, **kwargs):
        key = (str(args[0].shape),)
        if key not in self.cache:
            self.prune_configs(kwargs)
            self.cache[key] = self.configs[1]
        return self.fn(*args, **kwargs, **self.cache[key].all_kwargs())
"""


def test_quack_fix_reaches_a_vendored_copy_imported_later(tmp_path):
    """sonic-moe's hub build vendors quack and imports it at the first forward (MFU-1 mfu1-qfx-b1, 23 Oct)."""
    import importlib
    import importlib.util

    import rl1_trainer

    hub = tmp_path / "_sonic_moe_cuda_test1"
    (hub / "quack").mkdir(parents=True)
    (hub / "__init__.py").write_text("")
    (hub / "quack" / "__init__.py").write_text("")
    (hub / "quack" / "autotuner.py").write_text(FAKE_AUTOTUNER)
    spec = importlib.util.spec_from_file_location("_sonic_moe_cuda_test1", hub / "__init__.py",
                                                  submodule_search_locations=[str(hub)])  # as hub loaders do
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    spec.loader.exec_module(pkg)
    patched, scan = rl1_trainer.install_quack_fix()
    assert "_sonic_moe_cuda_test1.quack.autotuner" not in patched
    mod = importlib.import_module("_sonic_moe_cuda_test1.quack.autotuner")  # the first forward
    assert mod.Autotuner._fixed and "_sonic_moe_cuda_test1.quack.autotuner" in scan()
    used = []
    tuner = mod.Autotuner(lambda A, config="unset": used.append(config), ["x"],
                          [mod.AutotuneConfig(config="c0"), mod.AutotuneConfig(config="c1")])
    for m in (81920, 78050):
        tuner(torch.zeros(m, 2))
    assert used == [None, None]
    with pytest.raises(rl1_trainer.QuackAutotuneInJob):
        tuner.prune_configs({})
    rl1_trainer.install_quack_fix()  # again: one finder
    assert sum(1 for f in sys.meta_path if getattr(f, "_quack_fix", False)) == 1


def test_quack_fix_scan_patches_a_copy_already_loaded(tmp_path):
    import importlib

    import rl1_trainer

    hub = tmp_path / "hubtwo"
    (hub / "_vend2" / "quack").mkdir(parents=True)
    (hub / "_vend2" / "__init__.py").write_text("")
    (hub / "_vend2" / "quack" / "__init__.py").write_text("")
    (hub / "_vend2" / "quack" / "autotuner.py").write_text(FAKE_AUTOTUNER)
    sys.path.insert(0, str(hub))
    try:
        mod = importlib.import_module("_vend2.quack.autotuner")
        mod.Autotuner._fixed = False
        patched, _ = rl1_trainer.install_quack_fix()
        assert "_vend2.quack.autotuner" in patched and mod.Autotuner.__dict__.get("_fixed")
    finally:
        sys.path.remove(str(hub))


def test_teacher_keeps_base_pieces_at_the_trainer_share(tmp_path):
    """g1l (23 Oct): student-first alone gave 212 student pieces and 0 base behind the play's backlog (trainer base 0)."""
    import numpy as np

    torch.manual_seed(0)
    emb = torch.randn(V, V)
    root, out = tmp_path / "roll", tmp_path / "out"
    rows = game_rows(np.random.default_rng(7), 30)
    base_rows = [{k: v for k, v in r.items() if k not in ("tokens", "logprobs")} for r in rows]
    for key, rr in (("a-base/s0", base_rows), ("a-base/s1", base_rows), ("z-play/s0", rows), ("z-play/s1", rows)):
        (root / key).mkdir(parents=True)
        write_rows(root / key / "rollout.jsonl", rr)
    t = heal_stream.StreamTeacher(root, out, fake_encode, fake_forward(emb), k=8, chunk=20, image_token_id=IMG,
                                  idle=1e9, base=("a-base",), base_share=0.5, base_per_scan=1000)
    t._scan()
    while not t.events.empty():
        t._handle(t.events.get_nowait())
    order = []
    for _ in range(40):
        chain, upto = t._due()
        if chain is None:
            break
        order.append("b" if chain.key.startswith("a-base") else "s")
        t.score(chain, upto)
    t.saver.shutdown(wait=True)
    assert order[0] == "s" and len(order) >= 20  # the student's piece first
    for i in range(2, len(order) + 1):  # never more than one piece away from 1:1 while both are due
        assert abs(order[:i].count("b") - order[:i].count("s")) <= 1, order[:i]


def test_reply_re_rendered_from_a_normalized_client_message_keeps_the_session_exact(tmp_path):
    """g1l k009 #279 (23 Oct): the history's assistant message differs from the recorded reply (reasoning "\\n\\n" came
    back "\\n"); the server re-renders the client's message, so neither extend nor rewrite match and v51 dropped the
    rest of the session (990 requests). The chain must restart from the exact prompt + the canonical rest."""
    END = 48

    def encode(capture):
        ids = []
        for m in capture["payload"]["messages"]:
            ids += [int(t) for t in m["content"].split()] + ([END] if m["role"] == "assistant" else [])
        start = len(ids)
        ids += [int(t) for t in capture["reply"].split()] + [END]
        tensor = torch.tensor([ids])
        return {"input_ids": tensor, "mm_token_type_ids": (tensor == IMG).int()}, start

    rng = __import__("numpy").random.default_rng(9)
    rows, server, messages, prev = [], [], [], None
    for n in range(9):
        user = [int(t) for t in torch.randint(1, 40, (int(rng.integers(4, 12)),))]
        new = [{"role": "user", "content": " ".join(map(str, user))}]
        if prev is None:
            new = [{"role": "system", "content": "1 2 3"}] + new
            server = [1, 2, 3] + user
        else:
            reply, sampled, normalized, pstart = prev
            history = reply.split()[:-1] if normalized else reply.split()  # the client's stored text differs
            new = [{"role": "assistant", "content": " ".join(history)}] + new
            if normalized:  # re-rendered from the client's message
                server = server[:pstart] + [int(t) for t in history] + [END] + user
            else:
                server = server + sampled + [END] + user
        keep = len(messages)
        messages = messages + new
        canon = [int(t) for t in torch.randint(1, 40, (int(rng.integers(4, 10)),))]
        sampled = canon[:1] + [41, 42] + canon[2:] if n % 2 == 0 else list(canon)
        rows.append({"n": n, "keep": keep, "messages": new, "reply": " ".join(map(str, canon)), "gameplay": True,
                     "template": {}, "usage": {"prompt_tokens": len(server), "completion_tokens": len(sampled)},
                     "tokens": sampled, "logprobs": [-0.1] * len(sampled), "version": 0})
        prev = (rows[-1]["reply"], sampled, n == 4, len(server))
    root, out = tmp_path / "roll", tmp_path / "out"
    (root / "p/s0").mkdir(parents=True)
    write_rows(root / "p/s0/rollout.jsonl", rows)
    t = heal_stream.StreamTeacher(root, out, encode, fake_forward(torch.randn(V, V)), k=8, chunk=1000,
                                  image_token_id=IMG, idle=1e9, end_id=END)
    t._scan()
    while not t.events.empty():
        t._handle(t.events.get_nowait())
    assert t.stats.get("inexact", 0) == 0 and t.stats.get("rerenders") == 1 and t.stats["breaks"] == 1
    assert t.chains["p/s0"].ids.tolist() == server + rows[-1]["tokens"] + [END]


def test_balance_pairs_units_by_forward_equivalents(tmp_path, monkeypatch):
    """MFU-1 lever 1 (HEAL_STREAM_BALANCE): a slot's units pair by forward-equivalents (S1: loss tokens x 4 + the
    inert prefix), not by sequence length; off: by length as before."""
    monkeypatch.setenv("HEAL_INERT_PREFIX", "1")
    q = heal_kd.StreamQueue(tmp_path, 0, max_age=6, stall=0.0, poll=0.0, coalesce=True)
    spec = {"a": (1000, 100, 50), "b": (1000, 1000, 40), "c": (400, 100, 30), "d": (600, 300, 20)}  # tokens, rows, t
    fresh = []
    for name, (tokens, rows, t) in spec.items():
        path = str(tmp_path / f"{name}.pt")
        q.meta[path] = {"tokens": tokens, "rows": rows, "reply_rows": rows}
        fresh.append((9, t, path))
    fe = {n: q.fe([str(tmp_path / f"{n}.pt")]) for n in spec}
    assert fe == {"a": 4 * 104 + 896, "b": 4000, "c": 4 * 144 + 256, "d": 4 * 344 + 256}
    name = lambda unit: Path(unit[0]).stem  # noqa: E731
    monkeypatch.delenv("HEAL_STREAM_BALANCE", raising=False)
    assert [name(u) for u in q.units(fresh, 2)] == ["a", "b"]  # equal length
    monkeypatch.setenv("HEAL_STREAM_BALANCE", "1")
    assert [name(u) for u in q.units(fresh, 2)] == ["a", "d"]  # nearest forward-equivalents (1312 vs 1632)
    monkeypatch.setenv("HEAL_INERT_PREFIX", "0")
    assert q.fe([str(tmp_path / "a.pt")]) == 4000


def test_base_slice_grows_while_the_share_lags(tmp_path):
    """c4/c6 (23 Oct): 16 base captures a scan behind a live play gave 0 base pieces for 10-30 min; while the teacher is
    behind the trainer's base share, a scan takes BASE_BEHIND_MULT x the slice (students still first)."""
    import numpy as np

    root, out = tmp_path / "roll", tmp_path / "out"
    rows = [{k: v for k, v in r.items() if k not in ("tokens", "logprobs")} for r in game_rows(np.random.default_rng(6), 40)]
    (root / "a-base/s0").mkdir(parents=True)
    write_rows(root / "a-base/s0/rollout.jsonl", rows * 4)
    t = heal_stream.StreamTeacher(root, out, fake_encode, None, k=8, chunk=20, image_token_id=IMG, idle=1e9,
                                  base=("a-base",), base_per_scan=4, base_share=0.2)
    t.stats.update(pieces=10, base_pieces=0)  # behind: 0 of 10
    t._scan()
    n = sum(1 for e in list(t.events.queue) if e[0] == "capture")
    assert n == 4 * heal_stream.BASE_BEHIND_MULT
    while not t.events.empty():
        t.events.get_nowait()
    t.stats.update(pieces=10, base_pieces=3)  # at the share: the plain slice
    t._scan()
    assert sum(1 for e in list(t.events.queue) if e[0] == "capture") == 4


def test_generation_manifests_and_full_rows_on_resume(tmp_path):
    """24 Oct: every start export is a full-row version of a new generation; a resumed trainer's versions all carry
    every expert row; shard names carry the generation (a re-used v<step> folder never pairs with an old shard)."""
    cut = heal_delta.Cut({"student_cut": {"source_layers": [0]}})
    w = torch.nn.Parameter(torch.randn(4, 8, 8))
    runner = type("R", (), {"named_parameters": lambda self: [("model.model.language_model.layers.0.mlp.experts.down_proj", w)]})()
    rows = lambda r: type("Rows", (), {"items": [{"p": w, "rows": torch.tensor(r), "start": 0}]})()  # noqa: E731

    def experts(folder):
        from safetensors.torch import load_file
        got = {}
        for f in folder.glob("delta-*.safetensors"):
            got.update(load_file(str(f)))
        return {int(k.split(".mlp.experts.")[1].split(".")[0]) for k in got}

    a = heal_delta.DeltaExporter(tmp_path / "a", cut, 0, files=2, generation=5)
    a.export(runner, rows([]), 0, force=True, first=True)
    a.wait()
    a.export(runner, rows([2]), 1)
    a.wait()
    m0, m1 = (json.loads((tmp_path / f"a/v{v}/manifest.json").read_text()) for v in (0, 1))
    assert (m0["generation"], m0["base"], m0["full"], m0["first"]) == (5, "start", True, True)
    assert (m1["full"], m1["first"]) == (False, False) and all(f.startswith("delta-g5-") for f in m1["files"])
    assert experts(tmp_path / "a/v0") == {0, 1, 2, 3} and experts(tmp_path / "a/v1") == {2}
    b = heal_delta.DeltaExporter(tmp_path / "b", cut, 0, files=2, base="resume")
    b.export(runner, rows([1]), 7)
    b.wait()
    mb = json.loads((tmp_path / "b/v7/manifest.json").read_text())
    assert mb["full"] and mb["base"] == "resume" and mb["generation"] > 5 and experts(tmp_path / "b/v7") == {0, 1, 2, 3}


def test_full_rows_until_the_play_confirms_the_generation(tmp_path):
    """A resumed trainer exports every expert row only until the play has applied a version of its generation
    (confirm, polled after each write), then the rows trained since its start."""
    cut = heal_delta.Cut({"student_cut": {"source_layers": [0]}})
    w = torch.nn.Parameter(torch.randn(4, 8, 8))
    runner = type("R", (), {"named_parameters": lambda self: [("model.model.language_model.layers.0.mlp.experts.down_proj", w)]})()
    rows = lambda r: type("Rows", (), {"items": [{"p": w, "rows": torch.tensor(r), "start": 0}]})()  # noqa: E731
    seen = {"applied": False}
    ex = heal_delta.DeltaExporter(tmp_path, cut, 0, files=2, base="resume", confirm=lambda g: seen["applied"])
    full = []
    for step, r, applied in ((6, [], False), (7, [1], False), (8, [2], True), (9, [3], True)):
        seen["applied"] = applied
        ex.export(runner, rows(r), step, force=step == 6, first=step == 6)
        ex.wait()
        full.append(json.loads((tmp_path / f"v{step}/manifest.json").read_text())["full"])
    # v8's writer saw the play's confirmation: v9 carries only the rows trained since the start (1, 2, 3)
    assert full == [True, True, True, False]
    from safetensors.torch import load_file
    got = {}
    for f in (tmp_path / "v9").glob("delta-*.safetensors"):
        got.update(load_file(str(f)))
    assert {int(k.split(".mlp.experts.")[1].split(".")[0]) for k in got} == {1, 2, 3}


def test_student_backlog_is_encoded_a_bounded_slice_per_scan(tmp_path):
    """c7b (24 Oct): recorded student plays read in one scan held every capture's encode in host RAM. A scan now
    encodes at most ``student_per_scan`` student captures, one per session in turn, in order within a session; every
    chain's rows still equal one pass over it."""
    import numpy as np

    def run(cap):
        torch.manual_seed(0)
        emb = torch.randn(V, V)
        root, out = tmp_path / f"roll{cap}", tmp_path / f"out{cap}"
        for g in ("g1/a0", "g2/a0"):
            (root / g).mkdir(parents=True)
        write_rows(root / "g1/a0/rollout.jsonl", game_rows(np.random.default_rng(1), 12, drain_at=7))
        write_rows(root / "g2/a0/rollout.jsonl", game_rows(np.random.default_rng(2), 9))
        t = heal_stream.StreamTeacher(root, out, fake_encode, fake_forward(emb), k=8, chunk=20, image_token_id=IMG,
                                      idle=1e9, encoders=2, student_per_scan=cap)
        t._scan()
        first = [(e[1], e[2]["request"]) for e in list(t.events.queue) if e[0] == "capture"]
        for _ in range(400):
            t._scan()
            while not t.events.empty():
                t._handle(t.events.get_nowait())
            chain, upto = t._due()
            if chain is not None:
                t.score(chain, upto)
            elif not any(tail.backlog for tail in t.tails.values()):
                break
        for key in list(t.chains):
            t._handle(("idle", key, None, None))
        pump(t)
        t.saver.shutdown(wait=True)
        assert t.error is None and t.stats["captures"] == 21
        pieces = {p.name: torch.load(p, weights_only=False) for p in sorted(out.glob("s-*.pt"))}
        return first, pieces, emb

    first, capped, emb = run(3)
    assert [k for k, _ in first] == ["g1/a0", "g2/a0", "g1/a0"]
    assert [n for k, n in first if k == "g1/a0"] == sorted(n for k, n in first if k == "g1/a0")
    chains = {}
    for d in capped.values():
        chains.setdefault((d["segment"], d["chain"]), []).append(d)
    assert len(chains) == 3  # g1: 2 chains (a drain), g2: 1
    for ds in chains.values():  # each chain's rows equal one pass over it, each row trained exactly once
        ds.sort(key=lambda d: d["piece"])
        ids = ds[-1]["ids"].long()
        full = reference_rows(emb, ids)
        covered = torch.zeros(ids.numel(), dtype=torch.bool)
        for d in ds:
            n, row0 = d["ids"].numel(), d["row0"]
            assert torch.equal(d["ids"].long(), ids[:n])
            tok_ref = full[row0:n][:-1].gather(1, ids[row0 + 1:n, None])[:, 0]
            assert torch.allclose(d["tok_logp"][:-1], tok_ref, atol=1e-5)
            assert not covered[row0:n - 1].any()
            covered[row0:n - 1] = True
        assert covered[:ids.numel() - 1].all()
