"""One-card stream teacher (MFU-2, 23 Oct): chain batching (rl1_trainer.MultiForward, StreamTeacher batch) and
W8A16 routed experts (rl1_trainer.ExpertsFP8) on the tiny Qwen4-Exp teacher, CPU fp32. Needs torch +
transformers 5.18 (Modal CPU)."""
import copy
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import heal_kd  # noqa: E402
import heal_stream  # noqa: E402
import numpy as np  # noqa: E402
import rl1_trainer  # noqa: E402
from test_stream_pipe import SESSIONS, harness, same  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "flash_config.json"


def tiny(seed=0):
    return heal_kd.tiny_teacher(str(CONFIG), seed=seed, vocab_size=248320).float().eval()


@pytest.fixture(scope="module")
def model():
    return tiny()


def stream(model, tmp, fwd, *, sessions=SESSIONS, chunk=48, budget=None, halves=2, batch=1):
    """StreamTeacher's own picking (``_due_many``/``score_many`` when batched, else ``_pick``/``score``) over the
    sessions' rows written in ``halves``; returns the teacher, the chains seen, the pieces."""
    encode, rows_of, image_id = harness(model)
    root = Path(tmp) / "roll"
    files = {}
    for g, (seed, n, drain) in sessions.items():
        (root / g).mkdir(parents=True)
        files[g] = rows_of(np.random.default_rng(seed), n, drain, seed)
    t = heal_stream.StreamTeacher(root, Path(tmp) / "out", encode, fwd, k=16, chunk=chunk, image_token_id=image_id,
                                  idle=1e9, encoders=2, budget=budget, batch=batch)
    seen, sizes = {}, []

    def pump():
        for _ in range(2000):
            t._scan()
            while not t.events.empty():
                t._handle(t.events.get_nowait())
                seen.update({id(c): c for c in t.chains.values()})
            if t.batch > 1:
                picks = t._due_many(t.batch)
                if not picks:
                    return
                sizes.append(len(picks))
                t.score_many(picks)
                used = {c for c, _ in picks}
                if t.budget is not None:
                    t.stats["evictions"] += t.budget([c for c in t.chains.values() if c not in used])
            else:
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
    t.batch_sizes = sizes
    return t, seen, pieces


def tie_only(want, got, bad):
    """Fields that differ beyond 1-ulp batching noise: tok_logp within 1e-5; top_ids only where the fp16 top_logp rows
    are equal (tied or boundary ids reordered, a near-uniform tiny teacher's ties); nothing else may differ."""
    real = []
    for n, k in bad:
        if k == "tok_logp":
            if float((want[n][k] - got[n][k]).nan_to_num().abs().max()) > 1e-5:
                real.append((n, k))
        elif k == "top_logp":  # 1 fp16 ulp (batched MoE GEMM rounding on some CPUs)
            u, v = want[n][k].float(), got[n][k].float()
            if bool(((u - v).abs() > torch.finfo(torch.float16).eps * torch.maximum(u.abs(), v.abs()) + 1e-7).any()):
                real.append((n, k))
        elif k == "top_ids":
            rows = (want[n][k] != got[n][k]).any(-1)
            if not torch.equal(want[n]["top_logp"][rows], got[n]["top_logp"][rows]):
                real.append((n, k))
        else:
            real.append((n, k))
    return real


def diff(want, got):
    """Fields that differ (bitwise), and the largest top-logp gap."""
    bad, gap = [], 0.0
    for n in want:
        for k in want[n]:
            if not same(want[n][k], got[n][k]):
                bad.append((n, k))
                if k == "tok_logp" and len(bad) <= 3:
                    u, v = want[n][k], got[n][k]
                    idx = ((u != v) & ~(torch.isnan(u) & torch.isnan(v))).nonzero().flatten()[:5].tolist()
                    print("tok_logp differs", n, u.shape, v.shape, idx, [(float(u[i]), float(v[i])) for i in idx])
                if k == "top_logp":
                    gap = max(gap, float((want[n][k].float() - got[n][k].float()).abs().max()))
    return bad, gap


def test_batched_pieces_equal_one_chain_at_a_time(model, tmp_path):
    """3 chains per forward (images, a drain, uneven piece lengths): every piece equals the one-chain forward's."""
    _, _, image_id = harness(model)
    serial = rl1_trainer.stream_forward(model, 16, image_id, device="cpu", autocast=False)
    _, _, want = stream(model, tmp_path / "s", serial)
    multi = rl1_trainer.MultiForward(model, 16, image_id, chains=3, device="cpu", autocast=False)
    try:
        t, seen, got = stream(model, tmp_path / "m", multi, batch=3)
        checks = [t.verify(c) for c in seen.values() if len(c.files) >= 2]
    finally:
        multi.close()
    assert sorted(want) == sorted(got) and len(got) >= 9
    assert max(t.batch_sizes) == 3 and t.stats["batches"] == len(t.batch_sizes)
    bad, gap = diff(want, got)
    print("batched vs one-chain", {"pieces": len(got), "differ": len(bad), "max_top_logp_gap": gap,
                                   "batches": t.batch_sizes})
    tok = max((float((want[n]["tok_logp"] - got[n]["tok_logp"]).nan_to_num().abs().max()) for n in want), default=0.0)
    print("tok_logp max abs", tok)
    assert not tie_only(want, got, bad), (tie_only(want, got, bad)[:8], gap)  # MoE rows in one GEMM: ulp noise only
    assert checks and all(c["top1_agree"] == 1.0 and c["top_logp_max_abs"] < 1e-3 for c in checks)


def test_batched_rebuilds_evicted_caches(model, tmp_path):
    """Every idle cache evicted after each batch: the next batch rebuilds them together (keep=False slices in the
    rendezvous) and the pieces still match a fresh pass."""
    _, _, image_id = harness(model)
    multi = rl1_trainer.MultiForward(model, 16, image_id, chains=3, device="cpu", autocast=False)

    def evict_all(chains):
        dropped = sum(c.cache is not None for c in chains)
        for c in chains:
            c.cache = None
        return dropped

    try:
        t, seen, _ = stream(model, tmp_path, multi, budget=evict_all, batch=3)
        checks = [t.verify(c) for c in seen.values() if len(c.files) >= 2]
    finally:
        multi.close()
    assert t.stats["recompute_tokens"] > 0 and t.stats["evictions"] > 0
    assert checks and all(c["top1_agree"] == 1.0 and c["top_logp_max_abs"] < 1e-3 for c in checks)


def test_batch_error_reaches_the_caller(model):
    """A chain whose cache is missing fails the request up front; a failing chain inside a step breaks the
    rendezvous for the others (no hang) and its error is raised."""
    _, _, image_id = harness(model)
    multi = rl1_trainer.MultiForward(model, 16, image_id, chains=2, device="cpu", autocast=False)
    try:
        ids = torch.randint(10, 1500, (1, 40))
        mk = lambda key, n: heal_stream.Chain(key, n, {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)})  # noqa: E731
        with pytest.raises(RuntimeError, match="cache holds"):
            list(multi.many([(mk("a", 0), 16, 32, True), (mk("b", 1), 0, 16, True)]))

        def post(i, a, b, logits):
            if i == 1:
                raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            list(multi.many([(mk("c", 2), 0, 40, True), (mk("d", 3), 0, 40, True)], post=post))
        got = list(multi.many([(mk("e", 4), 0, 40, True), (mk("f", 5), 0, 24, True)]))
        assert got[-1] == [(0, 32, 40)] and len(got) == 3  # still usable; the shorter chain drops out
    finally:
        multi.close()


def test_chain_lock_serializes_dense_parts_and_keeps_the_moe_batch(model, monkeypatch):
    """coloc-v5 IMA: with RL1_CHAIN_LOCK (default) no two chains are ever inside a non-MoE module at once (Triton's
    Autotuner keeps per-call state on the instance), the logits equal the unlocked path, and a failing chain still
    frees the lock (no hang)."""
    import threading
    import time as _time

    _, _, image_id = harness(model)
    ids = torch.randint(10, 1500, (1, 40))
    mk = lambda key, n: heal_stream.Chain(key, n, {"input_ids": ids + n, "mm_token_type_ids": torch.zeros_like(ids)})  # noqa: E731
    core = model.model.language_model
    inside, peak = [0], [0]
    guard = threading.Lock()

    def track(module):
        orig = module.forward

        def fwd(*a, **k):
            with guard:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            _time.sleep(0.002)  # widen any overlap window
            try:
                return orig(*a, **k)
            finally:
                with guard:
                    inside[0] -= 1
        module.forward = fwd
        return orig

    def run(lock_on):
        monkeypatch.setenv("RL1_CHAIN_LOCK", "1" if lock_on else "0")
        m = copy.deepcopy(model)
        c = m.model.language_model
        saved = [track(layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn) for layer in c.layers]
        multi = rl1_trainer.MultiForward(m, 16, image_id, chains=2, device="cpu", autocast=False)
        outs = {}
        try:
            inside[0] = peak[0] = 0
            for _ in multi.many([(mk("a", 0), 0, 40, True), (mk("b", 1), 0, 40, True)],
                                post=lambda i, a, b, lg: outs.setdefault(i, []).append(lg.clone())):
                pass
            return outs, peak[0], multi
        finally:
            multi.close()
            del saved

    locked, peak_on, multi_on = run(True)
    assert multi_on.lock is not None and peak_on == 1
    unlocked, peak_off, multi_off = run(False)
    assert multi_off.lock is None
    for i in (0, 1):
        assert all(torch.equal(x, y) for x, y in zip(locked[i], unlocked[i]))
    assert core is model.model.language_model  # the shared fixture untouched

    monkeypatch.setenv("RL1_CHAIN_LOCK", "1")
    multi = rl1_trainer.MultiForward(model, 16, image_id, chains=2, device="cpu", autocast=False)
    try:
        def post(i, a, b, logits):
            if i == 1:
                raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            list(multi.many([(mk("c", 2), 0, 40, True), (mk("d", 3), 0, 40, True)], post=post))
        assert not multi.lock.locked()
        got = list(multi.many([(mk("e", 4), 0, 40, True), (mk("f", 5), 0, 24, True)]))
        assert got[-1] == [(0, 32, 40)] and not multi.lock.locked()
    finally:
        multi.close()


def test_rtn_fp8_round_trip():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(6, 8, 32, generator=g) * torch.rand(6, 8, 1, generator=g)).to(torch.bfloat16)
    w[0, 0] = 0  # an all-zero row
    codes, scales = rl1_trainer.rtn_fp8(w, chunk=4)
    back = rl1_trainer.dequant_fp8(codes, scales, torch.empty_like(w), chunk=4)
    assert codes.dtype == torch.float8_e4m3fn and scales.shape == (6, 8, 1)
    assert torch.equal(back[0, 0], torch.zeros(32, dtype=torch.bfloat16))
    rel = ((back.float() - w.float()).abs() / w.float().abs().amax(-1, keepdim=True).clamp(min=1e-12)).max()
    assert float(rel) <= 2 ** -4 + 1e-3  # half an e4m3 step at the row max
    assert torch.equal(codes.float().abs().amax(-1)[1:], torch.full((5, 8), 448.0))  # each row reaches the max


def test_fp8_experts_hook_path_equals_offline_copy(tmp_path):
    """The hook + shared scratch path gives exactly the logits of a model whose routed experts were replaced
    offline by dequant(rtn(w)); only the routed experts change; the BF16 copies are released."""
    torch.manual_seed(0)
    ref = tiny(seed=1)
    offline = copy.deepcopy(ref)
    live = copy.deepcopy(ref)
    blocks = [layer.mlp for layer in offline.model.language_model.layers]
    for block in blocks:
        for p in rl1_trainer.expert_params(block):
            codes, scales = rl1_trainer.rtn_fp8(p.data)
            p.data = rl1_trainer.dequant_fp8(codes, scales, torch.empty_like(p.data))
    fp8 = rl1_trainer.ExpertsFP8(live)
    ptrs = {p.data_ptr() for b in fp8.blocks for p in rl1_trainer.expert_params(b)}
    assert len(ptrs) == 2  # every layer's experts point at the one scratch pair
    dense = [(n, p) for n, p in ref.named_parameters() if ".mlp.experts." not in n]
    live_named = dict(live.named_parameters())
    assert all(torch.equal(p, live_named[n]) for n, p in dense)  # dense weights untouched
    ids = torch.randint(10, 1500, (1, 64))
    with torch.no_grad():
        a = offline(input_ids=ids).logits
        b = live(input_ids=ids).logits
        c = ref(input_ids=ids).logits
    assert torch.equal(a, b)
    kl = torch.nn.functional.kl_div(b.log_softmax(-1), c.log_softmax(-1), log_target=True, reduction="none").sum(-1)
    print("fp8 vs bf16 tiny", {"kl_mean": float(kl.mean()), "kl_max": float(kl.max())})
    assert float((a - c).abs().max()) > 0  # the quantization is real


def test_fp8_experts_batched_equal_unbatched(tmp_path):
    """FP8 experts + 3 chains per forward: one dequant per block per step, pieces equal the one-chain FP8 path."""
    m = tiny(seed=2)
    rl1_trainer.ExpertsFP8(m)
    _, _, image_id = harness(m)
    serial = rl1_trainer.stream_forward(m, 16, image_id, device="cpu", autocast=False)
    _, _, want = stream(m, tmp_path / "s", serial)
    multi = rl1_trainer.MultiForward(m, 16, image_id, chains=3, device="cpu", autocast=False)
    try:
        _, _, got = stream(m, tmp_path / "m", multi, batch=3)
    finally:
        multi.close()
    bad, gap = diff(want, got)
    tok = max((float((want[n]["tok_logp"] - got[n]["tok_logp"]).nan_to_num().abs().max()) for n in want), default=0.0)
    assert sorted(want) == sorted(got) and not tie_only(want, got, bad), (tie_only(want, got, bad)[:8], tok)


def test_fp8_finds_every_expert_under_the_trainer_peft_wrapping():
    """The teacher process wraps the model with PEFT exactly as fq.config's NVFP4 path asks (rank 16 on the dense
    targets + target_parameters on the routed experts: ParamWrapper renames them experts.base_layer.*; the gate's
    first launch found none). Every routed-expert tensor is found and quantized (layers x experts x 2), and the
    adapter-off forward equals the offline-quantized unwrapped copy."""
    peft = pytest.importorskip("peft")
    import flash_qlora_job as fq

    fq.set_adapter(16, False)
    fq.NVFP4.update(on=True, experts=True, backend="sonicmoe")
    cfg = fq.config("/x", "/tmp/fq", sequence_len=1024)
    base = tiny(seed=3)
    offline = copy.deepcopy(base)
    lc = peft.LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                         target_modules=cfg["lora_target_modules"], target_parameters=cfg["lora_target_parameters"])
    wrapped = peft.get_peft_model(base, lc)
    three_d = [n for n, p in wrapped.named_parameters() if p.ndim == 3 and "lora" not in n and ".mlp." in n]
    print("expert tensors under PEFT", three_d[:4])
    fp8 = rl1_trainer.ExpertsFP8(wrapped)
    text = wrapped.get_base_model().config.text_config
    count = sum(codes.shape[0] for packed in fp8.store.values() for codes, _ in packed)
    assert len(fp8.blocks) == text.num_hidden_layers
    assert count == text.num_hidden_layers * text.num_experts * 2
    for block in [layer.mlp for layer in offline.model.language_model.layers]:
        for p in rl1_trainer.expert_params(block):
            codes, scales = rl1_trainer.rtn_fp8(p.data)
            p.data = rl1_trainer.dequant_fp8(codes, scales, torch.empty_like(p.data))
    ids = torch.randint(10, 1500, (1, 48))
    with torch.no_grad(), wrapped.disable_adapter():
        got = wrapped(input_ids=ids).logits
    with torch.no_grad():
        want = offline(input_ids=ids).logits
    assert torch.equal(got, want)


def test_trainer_experts_fp8_method_reports_and_hooks():
    """Trainer.experts_fp8 (what RL1_EXPERTS_FP8=1 runs at load) on the PEFT-wrapped tiny teacher: the event carries
    the expert count and the dequant fields; the hook path then matches the offline copy."""
    peft = pytest.importorskip("peft")
    import types

    import flash_qlora_job as fq

    fq.set_adapter(16, False)
    fq.NVFP4.update(on=True, experts=True, backend="sonicmoe")
    cfg = fq.config("/x", "/tmp/fq", sequence_len=1024)
    base = tiny(seed=4)
    offline = copy.deepcopy(base)
    lc = peft.LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                         target_modules=cfg["lora_target_modules"], target_parameters=cfg["lora_target_parameters"])
    wrapped = peft.get_peft_model(base, lc)
    seen = []
    fake = types.SimpleNamespace(model=wrapped, rec=types.SimpleNamespace(event=lambda name, **kw: seen.append((name, kw))))
    fp8 = rl1_trainer.Trainer.experts_fp8(fake)
    text = wrapped.get_base_model().config.text_config
    assert seen and seen[0][0] == "experts_fp8"
    report = seen[0][1]
    assert report["expert_matrices"] == text.num_hidden_layers * text.num_experts * 2
    assert {"load_peak_gib", "quantize_peak_gib", "dequant_exact", "dequant_ms_per_forward"} <= set(report)
    for block in [layer.mlp for layer in offline.model.language_model.layers]:
        for p in rl1_trainer.expert_params(block):
            codes, scales = rl1_trainer.rtn_fp8(p.data)
            p.data = rl1_trainer.dequant_fp8(codes, scales, torch.empty_like(p.data))
    ids = torch.randint(10, 1500, (1, 40))
    with torch.no_grad(), wrapped.disable_adapter():
        got = wrapped(input_ids=ids).logits
    with torch.no_grad():
        want = offline(input_ids=ids).logits
    assert torch.equal(got, want) and fp8.report == report


def test_triton_dequant_kernel_matches_torch_under_the_interpreter():
    """The GPU dequant kernel (Triton) run by Triton's CPU interpreter (TRITON_INTERPRET=1 set before import) on
    Flash-shaped rows (2560 and 640 columns: a partial last block): the same codes, scales and products as torch."""
    import os
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("needs TRITON_INTERPRET=1")
    pytest.importorskip("triton")
    g = torch.Generator().manual_seed(1)
    for shape in ((3, 40, 2560), (3, 50, 640)):
        w = (torch.randn(shape, generator=g) * 0.02).to(torch.bfloat16)
        codes, scales = rl1_trainer.rtn_fp8(w)
        want = rl1_trainer.dequant_fp8(codes, scales, torch.empty_like(w))
        got = torch.empty_like(w)
        rows, cols = codes.numel() // shape[-1], shape[-1]
        rl1_trainer._dequant_kernel()[(rows,)](codes, scales, got, cols, BLOCK=1024)
        # the interpreter casts fp32 -> bf16 by truncation (Triton 3.8, rtne ignored; checked on a probe kernel); the GPU
        # rounds to nearest even (the gate's dequant_exact). Every load, index and product is still checked here.
        product = codes.float() * scales
        trunc = (product.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
        assert torch.equal(got, trunc), (shape, (got.float() - trunc.float()).abs().max())
        assert (got.float() - want.float()).abs().max() <= want.float().abs().max() * 2 ** -7


def test_status_reports_wall_rate_and_forward_share(model, tmp_path):
    """After the first forward the status carries the wall tok/s (supply waits included) and the forward share."""
    import time as _t

    _, _, image_id = harness(model)
    t, _, got = stream(model, tmp_path, rl1_trainer.stream_forward(model, 16, image_id, device="cpu", autocast=False))
    assert got and "warm_seconds" in t.stats
    t._warm_end = _t.time() - 10  # as if the first forward ended 10 s ago
    s = t.status()
    want = (t.stats["tokens"] - t.stats["warm_tokens"]) / 10
    assert abs(s["wall_tok_per_s"] - want) <= want * 0.05 + 0.1
    assert 0 <= s["forward_share"] <= 1


def test_topk_file_kl_readout():
    """The gate's readout: identical files 0 KL and full top-1; a shifted copy positive; non-reply rows ignored."""
    g = torch.Generator().manual_seed(0)
    n, k, V = 300, 16, 500
    logits = torch.randn(n, V, generator=g) * 3
    def file(lg):
        full = lg.log_softmax(-1)
        top = full.topk(k, -1)
        mass = top.values.double().exp().sum(-1).clamp(max=1 - 1e-6)
        return {"ids": torch.arange(n + 1), "reply": torch.arange(n + 1) % 3 != 0, "top_ids": top.indices.int(),
                "top_logp": top.values.half(), "tail": torch.log1p(-mass).half()}
    a = file(logits)
    same = heal_stream.topk_file_kl(a, file(logits))
    assert abs(same["kl"]) < 1e-4 and same["top1"] == 1.0 and same["rows"] == int((torch.arange(1, n + 1) % 3 != 0).sum())
    moved = heal_stream.topk_file_kl(a, file(logits + 0.3 * torch.randn(n, V, generator=g)))
    assert moved["kl"] > 1e-3 and 0 < moved["top1"] < 1


def test_moe_check_op_on_cpu_finds_no_batching_difference(model):
    """op_moe_check (opdS-c1/c2 diagnosis): on CPU the MoE block on two concatenated row sets equals one call each."""
    import types as _types
    events = []
    fake = _types.SimpleNamespace(model=model, _fp8=object(), rec=_types.SimpleNamespace(event=lambda name, **k: events.append(k)))
    got = rl1_trainer.Trainer.op_moe_check(fake, {"sizes": [[16, 24], [7, 0]], "layers": [0, 1]})
    rows = got["rows"]
    assert len(rows) == 4 and len(events) == 4
    for r in rows:
        assert r["repeat_rel"] == 0.0 and r["cat_rel_1"] < 1e-5
        if r["sizes"][1]:
            assert r["cat_rel_2"] < 1e-5 and r["bad_rows_2"] == 0
