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


@pytest.mark.parametrize("chunk,sub,newest,image_cut", [(48, 16, "0", False), (40, 7, "1", True), (64, 9, "1", True)])
@pytest.mark.parametrize("indexer", ["hf", "vectorized"])  # vectorized: the teacher's, with RL1_QSA_TIES=index
def test_cached_pieces_equal_a_fresh_pass(model, tmp_path, monkeypatch, chunk, sub, newest, image_cut, indexer):
    """opdS-c7j (24 Oct: GPU verify top-1 0.934 on a chain cut at an image start, 16172): every multi-piece chain's
    pieces (the cache continued across cuts, some at an image start, slices unaligned) equal one fresh pass."""
    monkeypatch.setenv("HEAL_STREAM_NEWEST", newest)
    if indexer == "vectorized":  # the teacher process's own indexer (rl1_trainer load)
        from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
        cls = modeling.Qwen4ExpTextQSAIndexer
        rl1_trainer.vectorized_indexer()
        monkeypatch.setattr(cls, "forward", cls._rl1_vectorized_forward)
        monkeypatch.setitem(rl1_trainer.TIES, "mode", "index")
        before = dict(rl1_trainer.INDEXER)
    image_id = model.config.image_token_id
    fwd = rl1_trainer.stream_forward(model, sub, image_id, device="cpu", autocast=False)
    t, seen, pieces = stream(model, tmp_path, fwd, sessions=SESSIONS, chunk=chunk)
    multi = [c for c in seen.values() if len(c.files) > 1]
    at_image = sum(int(p["ids"][p["row0"] + 1]) == image_id for p in pieces.values() if p["row0"] > 0)
    assert multi and (at_image > 0) == image_cut, (len(multi), at_image)
    for c in multi:
        r = t.verify(c)
        # top_logp is stored FP16: a ~1e-6 FP32 call-shape difference can flip one FP16 rounding (2^-7 at |logp| 8-16,
        # c058 CI); the FP32 realised-token rows carry the real bar
        assert r["tok_logp_max_abs"] < 1e-4 and r["top_logp_max_abs"] <= 2 ** -6 and r["top1_agree"] == 1.0, r
        assert r["kl_mean"] is not None and -1e-4 <= r["kl_mean"] < 1e-3, r  # KD-support KL: only the FP16 storage floor
    if indexer == "vectorized":
        assert rl1_trainer.INDEXER["vectorized"] > before["vectorized"]


def test_continuation_check_on_cpu(model):
    """The c7j GPU diagnosis' own measure (heal_stream.continuation_check) on the tiny teacher: every cut kind is found
    on a session with images and replies, and on CPU fp32 each equals the fresh pass."""
    encode, rows_of, image_id = harness(model)
    rows = rows_of(np.random.default_rng(5), 14, None, 5)
    msgs = []
    for r in rows:
        msgs = msgs + r["messages"] if r["keep"] else list(r["messages"])
    reply_text = " ".join(r["reply"] for r in rows)
    inputs, start = encode({"payload": {"messages": msgs}, "reply": reply_text})
    ids = inputs["input_ids"][0]
    reply = torch.zeros(ids.shape[0], dtype=torch.bool)
    reply[start:] = True
    reply[start + 3:start + 6] = False  # two reply runs: a reply end inside
    near = ids.shape[0] // 2
    cuts = heal_stream.continuation_cuts(ids, reply, image_id, near=near, ratio=4)
    assert set(cuts) >= {"unaligned", "image_start", "reply_end"}, cuts
    fwd = rl1_trainer.stream_forward(model, 7, image_id, device="cpu", autocast=False)
    got = heal_stream.continuation_check(fwd, inputs, cuts, rows=5)
    for name in ("fresh_repeat", *cuts):
        assert got[name]["top1_agree"] == 1.0 and got[name]["max_abs"] < 1e-3, (name, got[name])


@pytest.mark.parametrize("ties,exact", [("topk", False), ("index", True)])
def test_qsa_ties_decide_cached_vs_fresh(model, monkeypatch, ties, exact):
    """c7j root cause on the tiny teacher: the vectorized indexer's topk broke budget-edge ties by the row's length, so
    a continued cache (short calls) differed from one long fresh call (sub 1000); RL1_QSA_TIES=index makes every cut
    equal the fresh pass."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    cls = modeling.Qwen4ExpTextQSAIndexer
    rl1_trainer.vectorized_indexer()
    monkeypatch.setattr(cls, "forward", cls._rl1_vectorized_forward)
    monkeypatch.setitem(rl1_trainer.TIES, "mode", ties)
    encode, rows_of, image_id = harness(model)
    rows = rows_of(np.random.default_rng(1), 10, None, 1)
    msgs = []
    for r in rows:
        msgs = msgs + r["messages"] if r["keep"] else list(r["messages"])
    inputs, _ = encode({"payload": {"messages": msgs}, "reply": " ".join(r["reply"] for r in rows)})
    fwd = rl1_trainer.stream_forward(model, 1000, image_id, device="cpu", autocast=False)
    got = heal_stream.continuation_check(fwd, inputs, {"c100": 100, "c101": 101, "c151": 151}, rows=50)
    worst = max(got[k]["max_abs"] for k in ("c100", "c101", "c151"))
    assert (worst < 1e-4) == exact, got


def _check_setup(model, tmp_path, monkeypatch):
    """The stream_check op's CPU stand-in: the tiny teacher in train mode with gradient checkpointing (as the trainer
    loads it), a teacher file with images and a masked reply span, and a fake trainer ``self``."""
    import contextlib
    import copy
    import functools
    import types

    m = copy.deepcopy(model).train()
    m.gradient_checkpointing_enable()
    m.disable_adapter = contextlib.nullcontext
    monkeypatch.setattr(rl1_trainer, "stream_forward",
                        functools.partial(rl1_trainer.stream_forward, device="cpu", autocast=False))
    encode, rows_of, image_id = harness(model)
    rows = rows_of(np.random.default_rng(5), 14, None, 5)
    msgs = []
    for r in rows:
        msgs = msgs + r["messages"] if r["keep"] else list(r["messages"])
    inputs, start = encode({"payload": {"messages": msgs}, "reply": " ".join(r["reply"] for r in rows)})
    n = inputs["input_ids"].shape[1]
    reply = torch.zeros(n, dtype=torch.bool)
    reply[start:] = True
    reply[start + 3:start + 6] = False
    path = tmp_path / "f.pt"
    torch.save({"ids": inputs["input_ids"][0].int(), "reply": reply, "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"]}, path)
    fake = types.SimpleNamespace(model=m, processor=types.SimpleNamespace(image_token_id=image_id),
                                 rec=types.SimpleNamespace(event=lambda name, **kw: None), _fp8=None,
                                 experts_fp8=lambda: "stub")
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    cls = modeling.Qwen4ExpTextQSAIndexer
    rl1_trainer.vectorized_indexer()
    monkeypatch.setattr(cls, "forward", cls._rl1_vectorized_forward)
    return fake, path, n


def test_stream_check_op_on_cpu(model, tmp_path, monkeypatch):
    """rl1 op stream_check (the c7j GPU diagnosis) end to end on the tiny teacher left in train mode with gradient
    checkpointing (as the trainer loads it: use_cache off unless the op sets eval): every cut equals the fresh pass."""
    fake, path, n = _check_setup(model, tmp_path, monkeypatch)
    got = rl1_trainer.Trainer.op_stream_check(fake, {"files": [str(path)], "near": [n // 2], "limit": 10 ** 6, "sub": 7,
                                                     "ties": ["index"], "fla_pin": [True], "ref_single": True,
                                                     "subs": [5, 64], "offsets": [1, 4], "indexer": ["vectorized", "loop"],
                                                     "state_check": True, "kernels": ["hub", "torch_both"]})
    assert [r["kernels"] for r in got["results"] if "state_check" in r] == ["hub", "torch_both"]
    st = [r["state_check"] for r in got["results"] if "state_check" in r][0]
    assert st and all(k["max_rel"] < 1e-4 for v in st.values() for k in v["kinds"].values()), st
    assert any("recurrent_states" in v["kinds"] and "keys" in v["kinds"] for v in st.values()), st
    got["results"] = [r for r in got["results"] if "indexer" in r]
    assert [r["indexer"] for r in got["results"]] == ["vectorized", "loop"]
    for row in got["results"]:
        assert row["fresh_sub5"]["max_abs"] < 1e-3 and row["fresh_sub64"]["max_abs"] < 1e-3 and "off1" in row, row
    row = got["results"][0]
    cuts = [k for k, v in row.items() if isinstance(v, dict) and "top1_agree" in v]
    assert {"fresh_repeat", "unaligned", "image_start", "reply_end"} <= set(cuts), cuts
    assert all(row[k]["top1_agree"] == 1.0 and row[k]["max_abs"] < 1e-3 for k in cuts), row
    assert row["tie_rows"]["rows"] > 0 and rl1_trainer.TIES["count"] is False


V4_EXTRA = ('{"ref_single": true, "subs": [4096, 2048], "offsets": [1, 4, 16, 37, 63, 64, 128], "state_check": true, '
            '"kernels": ["hub", "torch_gdn", "torch_conv"]}')
V5_EXTRA = '{"length_diff": {"lengths": [24, 64, 120], "rows": [0, 24], "layers": [0, 1]}}'
V7_EXTRA = ('{"length_diff": {"lengths": [64, 2048, 8192, 24000], "rows": [0, 64], "layers": [0]}, "with_cuts": true, '
            '"ref_single": true, "subs": [8192, 2048], "offsets": [1, 37, 64], "kl_window": [16448, 16960]}')


V8_EXTRA = ('{"sub": 7, "stream_verify": {"bucket_run": "b2-x", "bucket_path": "duck-transfer/games/k000", "base": ["b2-x"], '
            '"chunk": 40, "pieces": 3, "slicings": ["pieces", "safe", "sub", "sub"], "worst": 8}}')


V9_EXTRA = ('{"sub": 7, "stream_verify": {"bucket_run": "b2-x", "bucket_path": "duck-transfer/games/k000", "base": ["b2-x"], '
            '"chunk": 40, "pieces": 3, "slicings": [], "length_diff": {"lengths": [24, 64, 96], "rows": [0, 24], '
            '"layers": "all", "depth": 1, "perturb": [64, 96]}}}')


# EOS-driver step A (25 Oct): the stream teacher over staged rollouts on the volume, pieces kept (out), forced ids on,
# no verify slicings
V10_EXTRA = ('{"sub": 7, "stream_verify": {"rollouts": "ROLL", "out": "OUT", "chunk": 40, "pieces": 100000, '
             '"slicings": []}}')


@pytest.mark.parametrize("extra", [V4_EXTRA, V5_EXTRA, V7_EXTRA, V8_EXTRA, V9_EXTRA, V10_EXTRA],
                         ids=["v4", "v5", "v7", "v8", "v9", "v10"])
def test_stream_check_entry_on_cpu(model, tmp_path, monkeypatch, extra):
    """Rule 4's CPU gate for the stream_check diagnosis (two GPU launches died in the wrapper, 24 Oct): the Modal
    function's own body (modal_rl1.stream_check, decorator off) with the launch's ARC_SC_* settings (env, as
    launch_stream_check.sh sets them; ``near`` scaled to the tiny file), its trainer faked by the op run in-process on
    the tiny teacher, through to stream-check.json."""
    import ast
    import os
    import types

    fake, path, n = _check_setup(model, tmp_path, monkeypatch)
    e = os.environ
    calls = []

    class Trainer:  # rl_loop.Trainer's protocol: a JSON command in, the op's dict (or ok False + error) out
        def __init__(self, folder, command, env, log):
            for k in ("RL1_EXPERTS_FP8", "RL1_QSA_TIES", "RL1_STREAM_CHAINS", "HEAL_FORCE_IDS"):
                if k in env:
                    monkeypatch.setenv(k, env[k])

        async def ready(self, timeout):
            return {"seconds": 0.0}

        async def call(self, op, timeout=3600, **payload):
            cmd = json.loads(json.dumps({"op": op, **payload}))
            calls.append(cmd)
            if op == "exit":
                return {"ok": True}
            try:
                return {"ok": True, **getattr(rl1_trainer.Trainer, f"op_{op}")(fake, cmd)}
            except Exception as error:  # noqa: BLE001 - as the trainer: the failure is the answer
                return {"ok": False, "error": repr(error)}

        def close(self):
            pass

    rl_loop = types.SimpleNamespace(settings=lambda x: {"trainer_timeout": 60}, Trainer=Trainer,
                                    trainer_command=lambda cfg, source, work: (["python"], {}))
    monkeypatch.setitem(sys.modules, "arena", types.SimpleNamespace(rl_loop=rl_loop))
    src = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "modal_rl1.py"
    keep = [f for f in ast.parse(src.read_text()).body
            if isinstance(f, ast.FunctionDef) and f.name in ("stream_check", "stream_teacher_env")]
    for f in keep:
        f.decorator_list = []
    ns = {"Path": Path, "sys": sys, "time": time, "json": json, "os": os, "CHAINS_OFF": "chains",
          "SC_ROOT": str(tmp_path / "runs"), "SC_WORK": str(tmp_path / "work"), "bundle_study": lambda: {"rl_loop": {}},
          "teacher_cache_key": lambda python: ("k", "v"), "teacher_cache_seed": lambda k, v: ({"quack": tmp_path}, False),
          "teacher_cache_env": lambda dirs: {}, "runs": types.SimpleNamespace(commit=lambda: None)}
    if "stream_verify" in extra:  # v8 (k000 diagnosis): a recorded session pulled from the bucket (rclone faked)
        import shutil
        import subprocess

        encode, rows_of, image_id = harness(model)
        fake.encoded = lambda capture, **kw: encode(capture)
        fake._stream_verify_diag = types.MethodType(rl1_trainer.Trainer._stream_verify_diag, fake)
        fake.processor.tokenizer = types.SimpleNamespace(convert_tokens_to_ids=lambda tok: None)  # no exact ids (tiny rows: no prompt_tokens)
        src = tmp_path / "bucket" / "duck-transfer" / "games" / "k000"
        if "ROLL" in extra:  # v10: rollouts staged on the volume (no bucket pull), pieces kept in OUT
            src = tmp_path / "roll" / "b2-x" / "k000"
            extra = extra.replace("ROLL", str(tmp_path / "roll")).replace("OUT", str(tmp_path / "pieces"))
        src.mkdir(parents=True)
        with open(src / "rollout.jsonl", "w") as f:
            for row in rows_of(np.random.default_rng(3), 16, None, 3):
                f.write(json.dumps(row) + "\n")

        def rclone(args, **kw):
            assert args[:2] == ["rclone", "copy"] and args[2].endswith("/b2-x/stream/trials/duck-transfer/games/k000"), args
            shutil.copytree(src, args[3], dirs_exist_ok=True)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", rclone)
        monkeypatch.setenv("ARC_SC_TENV", "RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=1,ARC_TEACHER_CACHE=1,RL1_QSA_TIES=index"
                           + (",HEAL_FORCE_IDS=3+5" if "pieces" in extra and '"slicings": []' in extra and "length_diff" not in extra else ""))
        ns.update(nebius_rclone_env=lambda: None, NEBIUS_BUCKET="bucket")
    exec(compile(ast.Module(keep, []), "modal_rl1_stream_check", "exec"), ns)
    kw = {"teacher_env": e["ARC_SC_TENV"]} if e.get("ARC_SC_TENV") else {}
    ids = torch.load(path, weights_only=True)["ids"]
    runs = heal_stream.image_runs(ids.long(), model.config.image_token_id)
    inside = next(a for a, b in runs if 30 < a < n - 8) + 2  # v5 died on a length that split an image: the gate's v5 case splits one too
    if "length_diff" in json.loads(extra):  # the launch's keys, the tiny file's lengths / rows / window
        x = json.loads(extra)
        x["length_diff"].update(lengths=[24, inside, n], rows=[0, 24])
        if "kl_window" in x:
            x["kl_window"], x["subs"] = [n // 2, n], [5, 64]
        extra = json.dumps(x)
    report = ns["stream_check"]("gate", str(path), "tiny", near=str(n // 2), limit=int(e.get("ARC_SC_LIMIT", "24000")),
                                fla_pin=e.get("ARC_SC_PIN", "1"), ties=e.get("ARC_SC_TIES", "topk"), extra=extra, **kw)
    assert report["status"] == "complete", {k: report.get(k) for k in ("error", "traceback")}
    assert json.loads((tmp_path / "runs" / "gate" / "stream-check.json").read_text())["status"] == "complete"
    assert set(json.loads(extra)) <= set(calls[0]) and calls[-1]["op"] == "exit"
    rows = report["results"]
    assert rows and any("top1_agree" in str(r) for r in rows), rows
    if json.loads(extra).get("state_check"):
        assert all("error" not in r["state_check"] for r in rows if "state_check" in r), rows
    for r in rows:  # v8: the stream teacher's own pieces of the pulled chain, verified per slicing (CPU fp32: exact)
        if "stream_verify" in r:
            sv = r["stream_verify"]
            assert isinstance(sv, dict), r
            if "length_diff" in sv:  # v9: CPU fp32 is causal: no module moves with the call's end or later content
                ld = sv["length_diff"]
                assert ld["first_over"] == {} and ld["perturb"]["vs_same_length_kl"] < 1e-6, ld
                assert any(k.endswith("p") for k in map(str, ld["lengths"])) and len(ld["order"]) > 3, ld
                continue
            if not json.loads(extra)["stream_verify"]["slicings"]:  # v10: every piece kept with the forced ids' log p
                files = sorted((tmp_path / "pieces").glob("s-*.pt"))
                assert len(files) >= 3 and sv.get("pieces", 0) >= 2, sv
                for f in files:
                    d = torch.load(f, weights_only=False)
                    assert d["force_ids"].tolist() == [3, 5] and d["force_logp"].shape == (d["ids"].numel() - d["row0"], 2)
                    assert torch.isfinite(d["force_logp"].float()).all()
                # option 1 (HEAL_THINK_EOS_MASK) on these pieces through the trainer's loader: 3 = "EOS", 5 = "</think>"
                monkeypatch.setattr(heal_kd, "FORCE_IDS", (3, 5))
                monkeypatch.setattr(heal_kd, "THINK_EOS", (3, 5))
                monkeypatch.setattr(heal_kd, "THINK_ACC", {})
                for f in files:
                    top, idx, tail = heal_kd.load_teacher_cpu(f)["teacher"]
                    assert top.shape[1] == idx.shape[1] and not torch.isnan(tail.float()).any()
                assert heal_kd.THINK_ACC["no_eos_column"] == 0 and heal_kd.THINK_ACC["masked"] > 0, heal_kd.THINK_ACC
                continue
            assert sv.get("pieces", 0) >= 3 and [v["slicing"] for v in sv["verify"]] == ["pieces", "safe", "sub", "sub"], sv
            for v in sv["verify"]:
                assert v["tok_logp_max_abs"] < 1e-4 and v["top1_agree"] == 1.0 and len(v["worst"]) == 8, v
                assert v["rows_image"] > 0 and v["rows_text"] > 0, v
    for r in rows:  # v5 on the tiny CPU teacher: rows [0, 24) do not depend on what follows them in the call
        if "length_diff" in r:
            ld = r["length_diff"]
            assert ld["lengths"][0] == 24 and len(ld["lengths"]) == 3 and ld["first_over"] == {}, ld
            assert inside not in ld["lengths"] and not any(a < L < b for a, b in runs for L in ld["lengths"]), ld
            last = json.loads(extra)["length_diff"]["layers"][-1]
            assert "layers0.input" in ld["order"] and any(k.startswith(f"L{last}.mlp") for k in ld["modules"]), ld["order"]
            assert max(v for m in ld["modules"].values() for v in m.values()) < 1e-4, ld["modules"]
            assert all(x["top1_agree"] == 1.0 and x["kl"] < 1e-6 for x in ld["logp"].values()), ld["logp"]
    if "kl_window" in extra:  # v7: KL vs the one-slice reference on the window, every fresh shape and cut
        cut_rows = [r for r in rows if "indexer" in r]
        assert cut_rows, rows
        kls = {k: v for r in cut_rows for k, v in r.items() if isinstance(v, dict) and "kl" in v}
        assert {"fresh_sub5", "fresh_sub64", "off1"} <= set(kls) and all(v["kl"] < 1e-6 and v["kl_rows"] > 0
                                                                         for v in kls.values()), kls


def test_heal_trainer_ties_flag():
    """The heal trainer's env takes HEAL_* flags only: HEAL_QSA_TIES=index sets the same tie mode as RL1_QSA_TIES."""
    import os
    import subprocess

    src = str(Path(rl1_trainer.__file__).parent)
    env = {k: v for k, v in os.environ.items() if k != "RL1_QSA_TIES"}
    out = subprocess.run([sys.executable, "-c", "import rl1_trainer; print(rl1_trainer.TIES['mode'])"], cwd=src,
                         env={**env, "HEAL_QSA_TIES": "index", "PYTHONPATH": src}, capture_output=True, text=True)
    assert out.stdout.strip().splitlines()[-1] == "index", out.stderr[-800:]
