"""Pipelined stream teacher (MFU-2, 23 Oct): rl1_trainer.PipeForward against stream_forward on the tiny Qwen4-Exp
teacher, CPU fp32. Needs torch + transformers 5.18 (Modal CPU); OMP_NUM_THREADS must equal torch's thread count."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import heal_kd  # noqa: E402
import heal_stream  # noqa: E402
import numpy as np  # noqa: E402
import rl1_trainer  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "flash_config.json"


@pytest.fixture(scope="module")
def model():
    return heal_kd.tiny_teacher(str(CONFIG), vocab_size=248320).float().eval()  # real image ids


def harness(model):
    vc = model.config.vision_config
    image_id = model.config.image_token_id
    patch = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
    grid, per_image = (1, 4, 4), 16 // vc.spatial_merge_size ** 2

    def encode(capture):
        ids, pix, grids, start = [], [], [], None
        text = [m["content"] for m in capture["payload"]["messages"]]
        for i, part in enumerate(text + [capture["reply"]]):
            if i == len(text):
                start = len(ids)
            for tok in part.split():
                if tok.startswith("IMG"):
                    g = torch.Generator().manual_seed(int(tok[3:]))
                    pix.append(torch.randn(16, patch, generator=g))
                    grids.append(grid)
                    ids += [image_id] * per_image
                else:
                    ids.append(int(tok))
        t = torch.tensor([ids])
        inputs = {"input_ids": t, "mm_token_type_ids": (t == image_id).long()}
        if pix:
            inputs["pixel_values"] = torch.cat(pix)
            inputs["image_grid_thw"] = torch.tensor(grids)
        return inputs, start

    def rows_of(rng, n_requests, drain_at, salt):
        rows, messages, last = [], [], None
        for n in range(n_requests):
            user = " ".join(str(int(x)) for x in rng.integers(10, 1500, int(rng.integers(5, 40))))
            if n % 2:
                user += f" 5 IMG{salt * 100 + n} 6"
            if n == drain_at or last is None:
                keep, add = 0, [{"role": "system", "content": "1 2 3"}, {"role": "user", "content": user}]
                messages = add
            else:
                keep, add = len(messages), [{"role": "assistant", "content": last}, {"role": "user", "content": user}]
                messages = messages + add
            reply_ids = [int(x) for x in rng.integers(10, 1500, int(rng.integers(3, 30)))]
            last = " ".join(map(str, reply_ids))
            rows.append({"n": n, "keep": keep, "messages": add, "reply": last, "gameplay": True, "template": {},
                         "usage": {"prompt_tokens": 1, "completion_tokens": len(reply_ids)}, "tokens": reply_ids,
                         "logprobs": [-0.1] * len(reply_ids), "version": n // 3})
        return rows

    return encode, rows_of, image_id


def stream(model, tmp, fwd, *, sessions, chunk=48, budget=None, halves=2):
    """Score the sessions' rows with ``fwd`` the way ``StreamTeacher.run`` picks pieces (``_pick``/``_evict``);
    returns the teacher, every chain seen and the piece files."""
    encode, rows_of, image_id = harness(model)
    root = Path(tmp) / "roll"
    files = {}
    for g, (seed, n, drain) in sessions.items():
        (root / g).mkdir(parents=True)
        files[g] = rows_of(np.random.default_rng(seed), n, drain, seed)
    t = heal_stream.StreamTeacher(root, Path(tmp) / "out", encode, fwd, k=16, chunk=chunk, image_token_id=image_id,
                                  idle=1e9, encoders=2, budget=budget)
    seen = {}

    def pump():
        for _ in range(2000):
            t._scan()
            while not t.events.empty():
                t._handle(t.events.get_nowait())
                seen.update({id(c): c for c in t.chains.values()})
            chain, upto = t._pick()
            if chain is None:
                return
            t.score(chain, upto)
            t._evict(chain)

    for half in range(halves):
        for g, rows in files.items():
            cut = len(rows) * half // halves, len(rows) * (half + 1) // halves
            with open(root / g / "rollout.jsonl", "a") as f:
                for row in rows[cut[0]:cut[1]]:
                    f.write(json.dumps(row) + "\n")
        pump()
    for key in list(t.chains):
        t._handle(("idle", key, None, None))
    pump()
    t.saver.shutdown(wait=True)
    pieces = {p.stem: torch.load(p, weights_only=False) for p in sorted((Path(tmp) / "out").glob("*.pt"))}
    return t, seen, pieces


def same(a, b):
    if torch.is_tensor(a):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.is_floating_point():
            return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
        return torch.equal(a, b)
    return a == b


SESSIONS = {"g1/a0": (1, 10, 6), "g2/a0": (2, 8, None), "g3/a0": (3, 12, None)}


def test_pipe_pieces_bitwise_equal_serial(model, tmp_path):
    """Every piece (top ids, top logp, realised-token logp, tail; images, a drain, 3 sessions, pieces of 3 slices
    and a prefetched next piece) bit-equal to the serial forward's."""
    _, _, image_id = harness(model)
    serial = rl1_trainer.stream_forward(model, 16, image_id, device="cpu", autocast=False)
    t0, _, want = stream(model, tmp_path / "s", serial, sessions=SESSIONS)
    timing = {}
    pipe = rl1_trainer.stream_forward_pipe(model, 16, image_id, split=4, device="cpu", autocast=False, timing=timing)
    try:
        t1, seen, got = stream(model, tmp_path / "p", pipe, sessions=SESSIONS)
        checks = [t1.verify(c) for c in seen.values() if len(c.files) >= 2]
    finally:
        pipe.close()
    assert sorted(want) == sorted(got) and len(got) >= 9
    bad = [(n, k) for n in want for k in want[n] if not same(want[n][k], got[n][k])]
    assert not bad, bad[:10]
    assert t1.stats["breaks"] == t0.stats["breaks"] == 1
    assert timing["slices"] > len(got) and timing["a_seconds"] > 0
    assert sum(int((p["ids"] == image_id).sum()) for p in got.values()) > 0
    assert checks and all(c["top1_agree"] == 1.0 and c["top_logp_max_abs"] < 1e-3 for c in checks)


def test_pipe_eviction_never_drops_the_prefetched_chain(model, tmp_path):
    """Evict every idle cache after each piece: the prefetched chain is spared (its stage-A state is ahead), the
    others recompute; the pieces still match a fresh pass."""
    _, _, image_id = harness(model)
    pipe = rl1_trainer.stream_forward_pipe(model, 16, image_id, split=4, device="cpu", autocast=False)
    hits = []

    def evict_all(chains):
        hits.extend(c for c in chains if c is pipe.pending_chain)
        dropped = sum(c.cache is not None for c in chains)
        for c in chains:
            c.cache = None
        return dropped

    try:
        t, seen, got = stream(model, tmp_path, pipe, sessions=SESSIONS, budget=evict_all)
        checks = [t.verify(c) for c in seen.values() if len(c.files) >= 2]
    finally:
        pipe.close()
    assert not hits
    assert t.stats["recompute_tokens"] > 0 and t.stats["evictions"] > 0
    assert checks and all(c["top1_agree"] == 1.0 and c["top_logp_max_abs"] < 1e-3 for c in checks)


@pytest.mark.skipif(__import__("os").environ.get("RUN_TIMING_TESTS") != "1", reason="wall-clock overlap needs dedicated cores (the Modal CPU test box has a fractional share)")
def test_pipe_overlaps_stages(model, tmp_path):
    """Scheduler: each decoder layer sleeps (a stand-in for kernels on its card; the GIL is free). Serial: every
    slice pays both halves; pipelined with the cross-piece prefetch: about one half per slice plus one fill."""
    _, _, image_id = harness(model)
    layers = model.model.language_model.layers
    d = 0.02
    originals = [layer.forward for layer in layers]

    def slow(f):
        def run(*a, **k):
            time.sleep(d)
            return f(*a, **k)
        return run

    for layer in layers:
        layer.forward = slow(layer.forward)
    try:
        sessions = {"g1/a0": (5, 14, None)}
        serial = rl1_trainer.stream_forward(model, 16, image_id, device="cpu", autocast=False)
        began = time.perf_counter()
        t0, _, _ = stream(model, tmp_path / "s", serial, sessions=sessions, halves=1)
        walls = [time.perf_counter() - began]
        pipe = rl1_trainer.stream_forward_pipe(model, 16, image_id, split=4, device="cpu", autocast=False)
        try:
            began = time.perf_counter()
            t1, _, _ = stream(model, tmp_path / "p", pipe, sessions=sessions, halves=1)
            walls.append(time.perf_counter() - began)
        finally:
            pipe.close()
    finally:
        for layer, f in zip(layers, originals):
            layer.forward = f
    ratio = walls[1] / walls[0]
    print("overlap", {"walls": walls, "ratio": ratio, "pieces": t0.stats["pieces"], "tokens": t0.stats["tokens"]})
    assert t1.stats["tokens"] == t0.stats["tokens"] and t0.stats["pieces"] >= 3
    assert ratio < 0.8, ratio  # split 4 of 6 layers: stage A is 4/6 of a slice, so ~0.67 plus one fill


@pytest.mark.parametrize("rep", range(int(__import__("os").environ.get("PIPE_REPEAT", "1"))))
def test_pipe_repeat(model, tmp_path, rep):
    """The bitwise test again (PIPE_REPEAT=n: a hang hunt)."""
    test_pipe_pieces_bitwise_equal_serial(model, tmp_path)


def test_pipe_abandoned_generator_and_errors(model):
    """A generator dropped mid-piece frees the worker; a stage-A error reaches the caller; close() joins."""
    _, _, image_id = harness(model)
    pipe = rl1_trainer.stream_forward_pipe(model, 16, image_id, split=4, device="cpu", autocast=False)
    try:
        ids = torch.randint(10, 1500, (1, 80))
        chain = heal_stream.Chain("x", 0, {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)})
        gen = pipe(chain, 0, 80, True)
        a, b, logits = next(gen)
        assert (a, b) == (0, 16) and logits.shape[0] == 16
        gen.close()
        bad = heal_stream.Chain("y", 1, {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)})
        with pytest.raises(RuntimeError, match="cache holds"):
            list(pipe(bad, 16, 32, True))  # no cache for lo > 0
        fresh = heal_stream.Chain("z", 2, {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)})
        assert len(list(pipe(fresh, 0, 48, True))) == 3  # the worker is still alive
    finally:
        pipe.close()
    assert not pipe.worker.is_alive() and not any(t.name == "stage-a" for t in threading.enumerate())


def test_pipe_refuses_a_split_before_the_first_attention_layer(model):
    """The next slice's masks are sized from the first attention layer's cache: a split at or before it races."""
    _, _, image_id = harness(model)
    with pytest.raises(RuntimeError, match="first attention layer"):
        rl1_trainer.stream_forward_pipe(model, 16, image_id, split=3, device="cpu", autocast=False)
