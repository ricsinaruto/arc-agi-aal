"""trunk/sft quantization-aware distillation on CPU: the fake-quant weights are bit-equal to export -> dequant; the
straight-through gradients reach the BF16 experts finite and nonzero; the trainer's teacher KL equals a direct full-KL
reference on a tiny model (k = vocab) and the top-k formula's own reference; a KL + fake-quant run trains and resumes."""
import json
import shutil
import sys
import threading
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/sft"), str(ROOT / "tests/golden"), str(ROOT / "tests")]
import common  # noqa: E402
import export  # noqa: E402
import mtp  # noqa: E402
import qat  # noqa: E402
import teacher  # noqa: E402
import train  # noqa: E402
from test_sft_train import data, opt  # noqa: E402

F = torch.nn.functional


def decode(qweight, scales, zero=8):
    inp, out = qweight.shape[0] * 8, qweight.shape[1]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    codes = ((qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).reshape(inp, out)
    return ((codes - zero).float() * scales.float().repeat_interleave(128, dim=0)).t()


def test_fake_quant_bit_equal_export():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(5, 24, 384, generator=g) * 0.02).to(torch.bfloat16)
    w[1, :, :128] = 0  # an all-zero group
    w[2, 3, 7] = 0.5  # an outlier
    qat.QAT["chunk"] = 2
    scales = qat.scales_of(w)
    got = qat.fake(w, scales)
    for e in range(w.shape[0]):
        qw, _, sc = export.int4_rtn(w[e], search=20)
        assert torch.equal(sc, scales[e]) and torch.equal(got[e], decode(qw, sc)), e


def qat_model(seed=0):
    """A tiny model whose expert widths take g128 (hidden and expert intermediate 128)."""
    from transformers import AutoConfig, AutoModelForMultimodalLM

    d = json.loads(common.CONFIG.read_text())
    d["text_config"].update(
        hidden_size=128, num_hidden_layers=3, layer_types=["linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=2, num_key_value_heads=1, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, moe_intermediate_size=128, shared_expert_intermediate_size=16,
        num_experts=4, num_experts_per_tok=2, hc_lowrank=8, ple_embed_dim=64, ngram_vocab_size_base=5000,
        split_ngram_parts=4, indexer_n_heads=2, indexer_budget=64, vocab_size=2048, ple_layer_ids=[1])
    d["vision_config"].update(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=128)
    d.pop("quantization_config", None)
    torch.manual_seed(seed)
    return AutoModelForMultimodalLM.from_config(AutoConfig.for_model(**d)).to(torch.bfloat16)


def on_disk(model, path):
    from safetensors.torch import save_file

    path.mkdir(parents=True)
    model.config.save_pretrained(str(path))
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    head = common.mtp_tiny(model.config.text_config)
    state.update({"mtp." + k: v.to(torch.bfloat16).contiguous() for k, v in head.state_dict().items() if "inv_freq" not in k})
    save_file(state, str(path / "model-00001.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {k: "model-00001.safetensors" for k in state}}))
    return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("qat")
    on_disk(qat_model(0), tmp / "student")
    t = qat_model(1)
    index = {}
    data(tmp / "data", "train", 4, 1, index)
    data(tmp / "data", "heldout", 2, 2, index)
    t.eval()
    for split in ("train", "heldout"):
        for e in index[split]["files"]:
            got = teacher.topk(t, train.load_example(tmp / "data" / e["file"]), k=2048)
            (tmp / "data/teacher" / e["file"]).parent.mkdir(parents=True, exist_ok=True)
            torch.save(got, tmp / "data/teacher" / e["file"])
    return tmp, t


def cfg(tmp, out, **kw):
    base = dict(student=str(tmp / "student"), data=str(tmp / "data"), out=str(out), lr=1e-3, warmup=1, rewarm=1,
                total_steps=3, batch_reply_tokens=150, window=128, prefix_chunk=32, loss_chunk=48, ckpt_steps=2,
                eval_steps=3, eval_examples=2, fast_attention=False, teacher_k=2048)
    base.update(kw)
    return train.Config(**base)


def test_ste_gradients_and_forward(setup):
    tmp, _ = setup
    model, head = train.load_model(tmp / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    ex = train.load_example(tmp / "data/train/e0.pt")
    qat.install(model)
    try:
        c = cfg(tmp, tmp / "x")
        got = train.example_loss(model, head, box, ex, c)
        experts = qat.routed(model)
        assert len(experts) == 3 and qat.QAT["calls"] > 0
        for mod in experts:
            for n in qat.NAMES:
                grad = getattr(mod, n).grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        qat.QAT["on"] = False  # the same forward with the grid's weights written in: the same loss
        with torch.no_grad():
            for mod in experts:
                for n in qat.NAMES:
                    getattr(mod, n).copy_(qat.fake(getattr(mod, n), qat.SCALES[id(mod)][n]).to(torch.bfloat16))
        assert train.example_loss(model, head, box, ex, c, train=False)[0] == got[0]
    finally:
        qat.QAT["on"] = False
        qat.SCALES.clear()


def test_kl_rows_sum_to_kl_topk():
    g = torch.Generator().manual_seed(1)
    s, t = torch.randn(9, 50, generator=g), torch.log_softmax(torch.randn(9, 50, generator=g), -1)
    v, i = t.topk(7, dim=-1)
    target = i[:, 0].clone()
    target[:3] = torch.tensor([int(next(x for x in range(50) if x not in i[r].tolist())) for r in range(3)])
    r = qat.kl_rows(s, i, v, target)
    assert torch.allclose(r["head"].sum() + r["tail"].sum(), qat.kl_topk(s, i, v), rtol=1e-5, atol=1e-6)
    assert r["in_topk"].tolist() == [False] * 3 + [True] * 6 and r["p_tail"].min() > 0


def test_eval_sets_and_dump(setup):
    """Diagnostics read: training files before / after the cursor, per-row KL parts dumped; the dumped
    head + tail rows sum to the event's KL and each row's position holds its target token."""
    tmp, _ = setup
    out = tmp / "evalsets"
    c = cfg(tmp, out, loss="kl", total_steps=0, eval_dump=True, eval_sets=[{"name": "unseen", "part": "unseen", "n": 2}])
    toy = {"examples": [{"file": f"f{j}"} for j in range(9)]}
    with pytest.raises(ValueError):  # nothing is seen at cursor (0, 0)
        train.eval_set_files(c, toy, (0, 0), {"part": "seen", "n": 2})
    got = train.eval_set_files(c, toy, (0, 4), {"part": "unseen", "n": 2})
    assert len(got) == 2 and not set(got) & set(train.pool(c, toy, 0)[:4])
    assert set(train.eval_set_files(c, toy, (0, 4), {"part": "seen", "n": 9})) == set(train.pool(c, toy, 0)[:4])
    train.train(c, device="cpu", make_opt=opt)
    ev = {r["set"]: r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines())
          if r.get("event") == "eval"}
    assert {"slice", "train:unseen"} <= set(ev)
    for name, key in (("slice", "slice"), ("unseen", "train:unseen")):
        d = torch.load(out / f"diag-{name}.pt")
        rows = d["rows"]
        assert len(rows["pos"]) == ev[key]["heldout_tokens"] == len(rows["ex"]) > 0
        kl = float(rows["head"].sum() + rows["tail"].sum()) / len(rows["pos"])
        assert abs(kl - ev[key]["heldout_kl"]) < 1e-4 * max(1.0, kl)
        for j in range(len(rows["pos"])):
            assert int(d["ids"][int(rows["ex"][j])][int(rows["pos"][j])]) == int(rows["target"][j])


def test_kl_topk_reference():
    g = torch.Generator().manual_seed(0)
    s, t = torch.randn(9, 50, generator=g), torch.randn(9, 50, generator=g)
    lt = torch.log_softmax(t, -1)
    full = F.kl_div(torch.log_softmax(s, -1), lt, log_target=True, reduction="sum")
    ids = torch.arange(50).expand(9, 50)
    assert torch.allclose(qat.kl_topk(s, ids, lt), full, rtol=1e-5, atol=1e-6)
    v, i = lt.topk(7, dim=-1)  # top-k + one tail bucket, written out row by row
    ls = torch.log_softmax(s, -1)
    ref = 0.0
    for r in range(9):
        p, q = v[r].exp(), ls[r, i[r]].exp()
        ref += float((p * (p.log() - q.log())).sum() + (1 - p.sum()) * ((1 - p.sum()).log() - (1 - q.sum()).log()))
    assert abs(float(qat.kl_topk(s, i, v)) - ref) < 1e-5 * max(1.0, abs(ref))


def test_trainer_kl_equals_full_reference(setup):
    """The trainer's windowed teacher KL (whole sequence in the window, k = vocab) against KL computed from the
    teacher's and student's full-sequence logits."""
    tmp, t = setup
    model, head = train.load_model(tmp / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    ex = train.load_example(tmp / "data/train/e1.pt", tmp / "data/teacher/train/e1.pt")
    ex["tail"] = ex["input_ids"].shape[1]  # no prefix cache: the same forward as the reference's
    c = cfg(tmp, tmp / "x", loss="kl", window=256)
    ce, n, _, _, _, tkl = train.example_loss(model, head, box, ex, c, train=False)
    ids, labels = ex["input_ids"], ex["labels"]
    rows = (labels[0, 1:] != -100).nonzero()[:, 0]
    with torch.no_grad():
        ls = torch.log_softmax(model(input_ids=ids).logits[0, rows].float(), -1)
        lt = torch.log_softmax(t(input_ids=ids).logits[0, rows].float(), -1)
    ref = float(F.kl_div(ls, lt, log_target=True, reduction="sum"))
    assert n == len(rows) and abs(tkl - ref) <= 1e-3 * abs(ref) + 1e-4, (tkl, ref)


def test_kl_fake_quant_run_and_resume(setup):
    tmp, _ = setup
    a = tmp / "runa"
    try:
        train.train(cfg(tmp, a, loss="kl", fake_quant=True), device="cpu", make_opt=opt)
        steps = {r["step"]: r for r in map(json.loads, (a / "events.jsonl").read_text().splitlines()) if r.get("event") == "step"}
        assert sorted(steps) == [1, 2, 3] and all(r["distill_kl"] > 0 and r["grad_norm"] > 0 for r in steps.values())
        b = tmp / "runb"
        b.mkdir()
        shutil.copytree(a / "ckpt-2", b / "ckpt-2")
        train.train(cfg(tmp, b, loss="kl", fake_quant=True), device="cpu", make_opt=opt)
        sb = [r for r in map(json.loads, (b / "events.jsonl").read_text().splitlines()) if r.get("event") == "step"]
        assert [r["step"] for r in sb] == [3] and abs(sb[0]["distill_kl"] - steps[3]["distill_kl"]) < 1e-6 * steps[3]["distill_kl"]
    finally:
        qat.QAT["on"] = False
        qat.SCALES.clear()



def test_kl_trunc_matches_screen_rule():
    """qat.kl_trunc equals the serving screen's per-row rule (throughput fp8kl.compare): the reference's top-k, the
    arm's log-prob where the token is in its top-k or is the actual token, else its k-th log-prob."""
    import math

    g = torch.Generator().manual_seed(0)
    n, v, k, kt = 6, 50, 5, 12
    logits = torch.randn(n, v, generator=g) * 2
    t_full = torch.log_softmax(torch.randn(n, v, generator=g) * 2, -1)
    lp, ids = t_full.topk(kt, -1)
    target = torch.randint(0, v, (n,), generator=g)
    got, top1 = qat.kl_trunc(logits, ids, lp, target, k)
    q = torch.log_softmax(logits.float(), -1)
    want, agree = 0.0, 0
    for i in range(n):
        r = {int(t): float(x) for x, t in zip(*t_full[i].topk(k))}
        sv, si = q[i].topk(k)
        arm = {int(t): float(x) for x, t in zip(sv, si)}
        arm[int(target[i])] = float(q[i, target[i]])
        floor = float(sv[-1])
        want += sum(math.exp(lr) * (lr - arm.get(t, floor)) for t, lr in r.items())
        agree += max(r, key=r.get) == int(si[0])
    assert abs(float(got) - want) < 1e-4 and int(top1) == agree


def test_kl_probe_stop_step_skips_unscored(setup, tmp_path):
    """lr-probe path: ``stop_step`` ends the run early on ``total_steps``' schedule (warm-up 0), evals at 0 and the
    stop step, no checkpoint with ``stop_ckpt`` off; kl trains only on examples with a teacher file."""
    tmp, _ = setup
    root = tmp_path / "data"
    shutil.copytree(tmp / "data", root)
    index = json.loads((root / "index.json").read_text())
    gone = index["train"]["files"][1]["file"]
    (root / "teacher" / gone).unlink()
    index["train"]["files"][2].update(run="r", game="g")  # a pair failing the bar: left out
    (root / "index.json").write_text(json.dumps(index))
    done = tmp_path / "done.json"  # a continuation leg: files an earlier leg trained, left out
    done.write_text(json.dumps([index["train"]["files"][3]["file"]]))
    held = tmp_path / "held"  # the eval set from its own root
    shutil.copytree(tmp / "data", held)
    out = tmp_path / "p"
    try:
        c = cfg(tmp, out, data=str(root), loss="kl", fake_quant=True, warmup=0, total_steps=50, stop_step=2,
                stop_ckpt=False, ckpt_steps=0, eval_steps=4, exclude_pairs=[["r", "g"]], heldout_data=str(held),
                eval_trunc_k=5, exclude_files=str(done))
        train.train(c, device="cpu", make_opt=opt)
        rows = [json.loads(r) for r in (out / "events.jsonl").read_text().splitlines()]
        steps = [r for r in rows if r.get("event") == "step"]
        assert [r["step"] for r in steps] == [1, 2]
        evs = [r for r in rows if r.get("event") == "eval"]
        assert all(0 <= r["heldout_kl_top5"] and 0 <= r["heldout_top1"] <= 1 for r in evs) and train.TRUNC["k"] == 0
        assert [r["step"] for r in rows if r.get("event") == "eval"] == [0, 2]
        assert next(r for r in rows if r.get("event") == "start")["examples"] == len(index["train"]["files"]) - 3
        assert abs(steps[1]["lr"] - train.schedule(c, 1, 0)) < 1e-12 and steps[1]["lr"] > 0.99 * c.lr
        assert not list(out.glob("ckpt-*")) and rows[-1]["event"] == "done"
    finally:
        qat.QAT["on"] = False
        qat.SCALES.clear()


def test_teacher_forced_ids_and_resumable_score(setup, tmp_path, monkeypatch):
    """Forced ids come first with their own log-probs, then the top-k of the rest, all distinct. ``score`` writes
    each example once (a second pass skips it) and starts nothing past its deadline."""
    tmp, t = setup
    ex = train.load_example(tmp / "data/train/e1.pt")
    forced = [7, 11, 13]
    got = teacher.topk(t, ex, k=5, forced=forced)
    ref = teacher.topk(t, ex, k=2048)
    full = torch.full((ref["ids"].shape[0], 2048), float("nan"))
    full.scatter_(1, ref["ids"].long(), ref["lp"])
    assert got["ids"].shape[1] == 8 and torch.equal(got["pos"], ref["pos"])
    assert (got["ids"][:, :3] == torch.tensor(forced, dtype=torch.int32)).all()
    assert all(len(set(r.tolist())) == 8 for r in got["ids"])
    assert torch.allclose(got["lp"], full.gather(1, got["ids"].long()), atol=1e-5)
    rest = full.clone()
    rest[:, forced] = -float("inf")
    assert torch.allclose(got["lp"][:, 3:].sort(-1).values, rest.topk(5, -1).values.sort(-1).values, atol=1e-5)
    root = tmp_path / "data"
    shutil.copytree(tmp / "data", root, ignore=shutil.ignore_patterns("teacher"))
    entries = teacher.order(root)
    assert [s for s, _ in entries][:2] == ["heldout", "heldout"] and len(entries) == 6
    assert teacher.score(t, root, entries, 4, forced, deadline=0) == (0, 0)
    monkeypatch.setitem(teacher.SANITY, "rows_min", 0)
    with pytest.raises(RuntimeError, match="sanity bar missed on example 1"):  # the random tiny teacher misses the bars
        teacher.score(t, root, entries, 4, forced, deadline=float("inf"))
    assert not (root / "teacher").exists() or not list((root / "teacher").rglob("*.pt"))
    monkeypatch.setitem(teacher.SANITY, "first", 0)
    n, tokens = teacher.score(t, root, entries[:3], 4, forced, deadline=float("inf"))
    assert n == 3 and tokens == sum(int(e["tokens"]) for _, e in entries[:3])
    first = (root / "teacher" / entries[0][1]["file"]).stat().st_mtime_ns
    n, _ = teacher.score(t, root, entries, 4, forced, deadline=float("inf"))
    assert n == 3 and (root / "teacher" / entries[0][1]["file"]).stat().st_mtime_ns == first
    saved = torch.load(root / "teacher" / entries[0][1]["file"], weights_only=False)
    assert {k: saved["meta"][k] for k in ("k", "forced", "split")} == {"k": 4, "forced": forced, "split": "heldout"}
    assert {"topk_mass", "rec_lp", "top1"} <= set(saved["meta"]) and not list(root.rglob("*.tmp"))
    (root / "priority.json").write_text(json.dumps(["train/e2.pt", "train/e0.pt"]))  # the selection, in its order
    assert [e["file"] for _, e in teacher.order(root)] == ["heldout/e0.pt", "heldout/e1.pt", "train/e2.pt", "train/e0.pt"]
    assert teacher.score.retries == [0, 0, 0]  # the last pass scored 3 (CPU: no allocator)


def test_teacher_profile_one_example(setup, tmp_path, monkeypatch):
    """PROFILE: the example numbered ``at`` runs under prof.profile (forward / rows phases in the trace) and writes
    the same file as an unprofiled pass; per-example retries are kept for the size sweep."""
    import time

    tmp, t = setup
    monkeypatch.setitem(teacher.SANITY, "first", 0)
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        shutil.copytree(tmp / "data", root, ignore=shutil.ignore_patterns("teacher"))
    entries = teacher.order(a)[:2]
    monkeypatch.setitem(teacher.PROFILE, "at", 2)
    monkeypatch.setitem(teacher.PROFILE, "out", str(tmp_path / "profile.json"))
    rows = []
    assert teacher.score(t, a, entries, 4, [7], deadline=float("inf"), log=rows.append)[0] == 2
    assert [r["event"] for r in rows].count("profiled") == 1 and len(teacher.score.retries) == 2
    monkeypatch.setitem(teacher.PROFILE, "at", None)
    teacher.score(t, b, entries, 4, [7], deadline=float("inf"), log=lambda r: None)
    for _, e in entries:
        x = torch.load(a / "teacher" / e["file"], weights_only=False)
        y = torch.load(b / "teacher" / e["file"], weights_only=False)
        assert torch.equal(x["ids"], y["ids"]) and torch.equal(x["lp"], y["lp"])
    for _ in range(600):
        got = json.loads((tmp_path / "profile.json").read_text())
        if "idle_by_phase" in got:
            break
        time.sleep(0.1)
    assert got["wall_s"] > 0 and got["flops_matmul"] > 0 and got["chunk"] == teacher.SIZES["chunk"]


def test_mtp_head_for_uncut_mtp(tmp_path):
    """A pruned student keeps Flash's uncut MTP block: the head takes the checkpoint's expert count and the
    config's MTP top-k (26 Oct smoke v3: 512 vs 128)."""
    from safetensors.torch import save_file

    text = qat_model(0).config.text_config
    text.mtp_num_experts_per_tok = 3
    e = text.num_experts
    save_file({"mtp.layers.0.mlp.gate.weight": torch.zeros(2 * e, text.hidden_size)}, str(tmp_path / "m.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.layers.0.mlp.gate.weight": "m.safetensors"}}))
    head = mtp.head_for(text, tmp_path)
    assert head.config.num_experts == 2 * e and head.config.num_experts_per_tok == 3
    save_file({"mtp.layers.0.mlp.gate.weight": torch.zeros(e, text.hidden_size)}, str(tmp_path / "m.safetensors"))
    head = mtp.head_for(text, tmp_path)
    assert head.config.num_experts == e and head.config.num_experts_per_tok == text.num_experts_per_tok


def image_model(seed=0):
    """qat_model with the image / vision token ids inside its 2048-token vocab."""
    from transformers import AutoConfig, AutoModelForMultimodalLM

    d = json.loads(common.CONFIG.read_text())
    d["text_config"].update(
        hidden_size=128, num_hidden_layers=3, layer_types=["linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=2, num_key_value_heads=1, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, moe_intermediate_size=128, shared_expert_intermediate_size=16,
        num_experts=4, num_experts_per_tok=2, hc_lowrank=8, ple_embed_dim=64, ngram_vocab_size_base=5000,
        split_ngram_parts=4, indexer_n_heads=2, indexer_budget=64, vocab_size=2048, ple_layer_ids=[1])
    d["vision_config"].update(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=128)
    ids = {"image_token_id": 2000, "vision_start_token_id": 1998, "vision_end_token_id": 1999, "video_token_id": 2001}
    for part in (d, d["text_config"]):
        part.update({k: v for k, v in ids.items() if k in part or part is d})
    d.pop("quantization_config", None)
    torch.manual_seed(seed)
    return AutoModelForMultimodalLM.from_config(AutoConfig.for_model(**d)).to(torch.bfloat16).eval()


def test_teacher_topk_equals_model_logits():
    """teacher.topk's chunked cache pass (7-token pieces: boundaries inside both images and inside the reply) gives
    log_softmax(model(**ex).logits) at the reply rows of one whole-sequence pass: the hidden-state @ lm_head path is
    the model's own logits path; the forced-id branch keeps the same values."""
    model = image_model(4)
    vc = model.config.vision_config
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(10, 1990, (200,), generator=g)
    mm = torch.zeros(200, dtype=torch.long)
    for start in (32, 60):  # image tokens 33..36 and 61..64 straddle the piece boundaries 35 and 63
        ids[start], ids[start + 5] = 1998, 1999
        ids[start + 1:start + 5] = 2000
        mm[start + 1:start + 5] = 1
    labels = torch.full_like(ids, -100)
    labels[120:200] = ids[120:200]  # the reply straddles 126, 133, ...
    patch = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])  # 16 patches -> 4 tokens each (merge 2)
    ex = {"input_ids": ids.to(torch.int32), "labels": labels.to(torch.int32), "mm_token_type_ids": mm.to(torch.int32),
          "pixel_values": torch.randn(32, patch, generator=g).to(torch.bfloat16), "image_grid_thw": grid}
    got = teacher.topk(model, ex, k=2048, chunk=7)
    rows = got["pos"].long() - 1
    with torch.no_grad():
        ref = torch.log_softmax(model(input_ids=ids[None], labels=None, pixel_values=ex["pixel_values"],
                                      image_grid_thw=grid, mm_token_type_ids=mm[None]).logits[0, rows].float(), -1)
    full = torch.empty_like(ref).scatter_(1, got["ids"].long(), got["lp"])
    assert got["pos"].tolist() == list(range(120, 200))
    assert (full - ref).abs().max() < 0.05 and (full.argmax(-1) == ref.argmax(-1)).float().mean() > 0.97
    assert torch.allclose(got["rec_lp"], ref.gather(1, ids[got["pos"].long()][:, None])[:, 0], atol=0.05)
    forced = [7, 11, 13]
    f = teacher.topk(model, ex, k=5, forced=forced, chunk=7)
    assert torch.allclose(f["lp"][:, :3], ref[:, forced], atol=0.05)
    rest = ref.clone()
    rest[:, forced] = -float("inf")
    assert torch.allclose(f["lp"][:, 3:], rest.topk(5, -1).values, atol=0.05)


def test_fuse_kept_only_if_bit_equal(monkeypatch):
    """qat.fuse keeps a fused grid only when it equals the eager grid on every sample (an all-zero group included:
    fp16 scale 0, 0/0); a grid that differs is refused and eager stays."""
    monkeypatch.setitem(qat.QAT, "fused", None)
    w = torch.randn(2, 64, 256).to(torch.bfloat16)
    w[0, :, :128] = 0
    s16 = qat.scales_of(w)
    got = qat.fuse([(w, s16)])
    assert got in ("compile", False)
    if got:
        assert torch.equal(qat.QAT["fused"](w, s16), qat.grid(w, s16))
    monkeypatch.setitem(qat.QAT, "fused", None)
    monkeypatch.setattr(qat, "_grid_one", lambda w, s: qat.fake(w, s).to(w.dtype) + 1)
    assert qat.fuse([(w, s16)]) is False and qat.QAT["fused"] is None


def test_refresh_drift_counts_stale_code_flips(setup):
    """refresh(drift=True) gives the same scales as refresh(); unchanged weights: no flips; a moved weight: the flips
    counted are exactly the codes that differ between the stale and fresh grids."""
    tmp, _ = setup
    model, _ = train.load_model(tmp / "student", "cpu", fast_attention=False)
    try:
        qat.refresh(model)
        same = qat.refresh(model, drift=True)
        assert same["code_flip_share"] == 0 and same["scale_changed_share"] == 0 and same["codes"] > 0
        mod = qat.routed(model)[0]
        stale = {n: qat.SCALES[id(mod)][n].clone() for n in qat.NAMES}
        with torch.no_grad():
            mod.gate_up_proj.mul_(1.05)
        got = qat.refresh(model, drift=True)
        fresh = {n: qat.scales_of(getattr(mod, n).detach()) for n in qat.NAMES}
        for n in qat.NAMES:
            assert torch.equal(qat.SCALES[id(mod)][n], fresh[n])
        w = mod.gate_up_proj.detach().float().transpose(-1, -2)
        g = w.reshape(*w.shape[:-2], w.shape[-2] // qat.QAT["group"], qat.QAT["group"], w.shape[-1])

        def codes(s16):
            return (g / s16.float().unsqueeze(-2)).round().clamp(-8, 7).nan_to_num(0.0)

        want = (codes(stale["gate_up_proj"]) != codes(fresh["gate_up_proj"])).sum().item()
        assert want > 0 and got["code_flip_share"] * got["codes"] == pytest.approx(want, abs=0.5)
        assert got["scale_changed_share"] > 0 and got["scale_rel_max"] > 0
    finally:
        qat.SCALES.clear()


def test_teacher_switches_same_files(setup, tmp_path, monkeypatch):
    """--ab-blocks switches (drift signals off, leg-v5's row masking) write the same teacher files."""
    tmp, t = setup
    monkeypatch.setitem(teacher.SANITY, "first", 0)
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        shutil.copytree(tmp / "data", root, ignore=shutil.ignore_patterns("teacher"))
    entries = teacher.order(a)[:2]
    rows = []
    teacher.score(t, a, entries, 4, [7], deadline=float("inf"), log=rows.append)
    monkeypatch.setitem(teacher.SWITCH, "diag", False)
    monkeypatch.setitem(teacher.SWITCH, "legacy", True)
    lean = []
    teacher.score(t, b, entries, 4, [7], deadline=float("inf"), log=lean.append)
    for _, e in entries:
        x = torch.load(a / "teacher" / e["file"], weights_only=False)
        y = torch.load(b / "teacher" / e["file"], weights_only=False)
        assert torch.equal(x["ids"], y["ids"]) and torch.equal(x["lp"], y["lp"])
    scored = [r for r in lean if r.get("event") == "scored"]
    assert scored and all("rss_gib" not in r for r in scored)
    assert any("rss_gib" in r for r in rows if r.get("event") == "scored")


def test_teacher_resets_causal_memo_each_call(setup, monkeypatch):
    """topk resets attention's causal-mask memo before every model call (sw-v1: a stale memo across calls)."""
    import attention

    tmp, t = setup
    ex = train.load_example(tmp / "data/train/e1.pt")
    calls = []
    monkeypatch.setattr(attention, "reset", lambda: calls.append(1))
    chunk = 16
    teacher.topk(t, ex, k=4, chunk=chunk)
    assert len(calls) == -(-ex["input_ids"].shape[1] // chunk) > 1


def test_plain_causal_memo_stale_without_reset_and_switches():
    """The memo answers by (storage, shape, dtype, version): a mask changed behind the version counter returns the
    old counts until ``reset``; ``memo`` off recomputes every call; ``dense`` gives None (HF's masked path)."""
    import attention

    n = 6
    m = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
    attention.reset()
    try:
        first = attention.plain_causal(m).clone()
        m.data[0, 0, n - 1, 0] = False  # not plain any more; .data writes do not bump m's version
        assert torch.equal(attention.plain_causal(m), first)  # stale: why topk resets per call
        attention.reset()
        assert attention.plain_causal(m) is None
        m.data[0, 0, n - 1, 0] = True
        attention.CAUSAL["memo"] = False
        attention.plain_causal(m)
        m.data[0, 0, n - 1, 0] = False
        assert attention.plain_causal(m) is None  # no memo: recomputed
        attention.CAUSAL.update(memo=True, dense=True)
        m.data[0, 0, n - 1, 0] = True
        assert attention.plain_causal(m) is None
    finally:
        attention.CAUSAL.update(memo=True, dense=False)
        attention.reset()


def test_teacher_scratch_dest_skips_pool_and_merges(setup, tmp_path, monkeypatch):
    """A scratch dest for the pool: files already in the data root's pool are skipped (skip_root), and ``merge``
    moves the new ones into the pool unchanged, keeping pool files that exist."""
    tmp, t = setup
    monkeypatch.setitem(teacher.SANITY, "first", 0)
    root, scratch = tmp_path / "data", tmp_path / "scratch"
    shutil.copytree(tmp / "data", root, ignore=shutil.ignore_patterns("teacher"))
    entries = teacher.order(root)
    teacher.score(t, root, entries[:2], 4, [7], deadline=float("inf"), log=lambda r: None)
    rows = []
    teacher.score(t, root, entries, 4, [7], deadline=float("inf"), log=rows.append, out=scratch, skip_root=True)
    scored = [r["file"] for r in rows if r.get("event") == "scored"]
    assert scored == [e["file"] for _, e in entries[2:]]
    made = {f: (scratch / "teacher" / f).read_bytes() for f in scored}
    assert teacher.merge(scratch, root) == {"moved": len(scored), "kept": 0}
    assert all((root / "teacher" / f).read_bytes() == b for f, b in made.items())
    assert not list((scratch / "teacher").rglob("*.pt"))


def test_teacher_local_io_same_files(setup, tmp_path, monkeypatch):
    """LOCAL (prefetch + mmap load, local write + mover processes) writes the same teacher files as the threads, into
    a scratch dest that skips the pool and merges; the work dir is removed and a dead prefetch raises."""
    tmp, t = setup
    monkeypatch.setitem(teacher.SANITY, "first", 0)
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        shutil.copytree(tmp / "data", root, ignore=shutil.ignore_patterns("teacher"))
    entries = teacher.order(a)
    teacher.score(t, a, entries, 4, [7], deadline=float("inf"), log=lambda r: None)
    teacher.score(t, b, entries[:1], 4, [7], deadline=float("inf"), log=lambda r: None)
    monkeypatch.setitem(teacher.LOCAL, "on", True)
    monkeypatch.setitem(teacher.LOCAL, "ahead", 2)
    monkeypatch.setitem(teacher.LOCAL, "dir", str(tmp_path / "io"))
    rows = []
    teacher.score(t, b, entries, 4, [7], deadline=float("inf"), log=rows.append, out=tmp_path / "s", skip_root=True)
    scored = [r["file"] for r in rows if r.get("event") == "scored"]
    assert scored == [e["file"] for _, e in entries[1:]]
    assert teacher.merge(tmp_path / "s", b) == {"moved": len(scored), "kept": 0}
    for _, e in entries:
        x = torch.load(a / "teacher" / e["file"], weights_only=False)
        y = torch.load(b / "teacher" / e["file"], weights_only=False)
        assert torch.equal(x["ids"], y["ids"]) and torch.equal(x["lp"], y["lp"])
    assert not list((tmp_path / "io").iterdir())
    monkeypatch.setattr(teacher, "PREFETCH", "import sys; sys.exit(3)")
    with pytest.raises(RuntimeError, match="teacher loader failed"):
        teacher.score(t, b, entries, 4, [7], deadline=float("inf"), log=lambda r: None, out=tmp_path / "s2")


def test_teacher_order_priority_list(setup, tmp_path):
    """``order(root, priority=path)``: held-out first, then exactly that list's train files in its order, ignoring
    <data>/priority.json."""
    tmp, _ = setup
    root = tmp / "data"
    full = teacher.order(root)
    train = [e["file"] for sp, e in full if sp == "train"]
    pick = tmp_path / "p.json"
    pick.write_text(json.dumps(train[::-1][:2]))
    got = teacher.order(root, pick)
    assert [sp for sp, _ in got] == [sp for sp, _ in full if sp == "heldout"] + ["train", "train"]
    assert [e["file"] for sp, e in got if sp == "train"] == train[::-1][:2]


def test_teacher_slow_host_exit(setup, tmp_path, monkeypatch):
    """SLOW: after ``after`` examples a median launch rate under the floor stops scoring cleanly, logged with the
    measured rate; the scored files are written."""
    tmp, t = setup
    monkeypatch.setitem(teacher.SANITY, "first", 0)
    root = tmp_path / "data"
    shutil.copytree(tmp / "data", root, ignore=shutil.ignore_patterns("teacher"))
    monkeypatch.setitem(teacher.SLOW, "after", 4)
    monkeypatch.setitem(teacher.SLOW, "floor", 1e12)
    rows = []
    n, _ = teacher.score(t, root, teacher.order(root), 4, [7], deadline=float("inf"), log=rows.append)
    slow = [r for r in rows if r.get("event") == "slow_host"]
    assert n == 4 and len(slow) == 1 and slow[0]["launch_tok_s_median"] < 1e12 and slow[0]["examples"] == 4
    assert teacher.score.slow["examples"] == 4
    assert len(list((root / "teacher").rglob("*.pt"))) == 4


def _served_kv4_round_trip(x):
    """The served int4 page round trip, verbatim arithmetic of arc_infra/kv4_runtime.py (v87b bundle): rht_signs,
    hadamard_reference, quantize4_reference, dequantize4_reference; rows rotated, cast to BF16, quantized, dequantized,
    rotated back / 256."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0x9E3779B9)
    d1 = 2.0 * torch.bernoulli(torch.full((x.shape[-1],), 0.5), generator=gen) - 1.0

    def had(y):
        d, h = y.shape[-1], 1
        while h < d:
            yv = y.reshape(*y.shape[:-1], d // (2 * h), 2, h)
            y = torch.stack([yv[..., 0, :] + yv[..., 1, :], yv[..., 0, :] - yv[..., 1, :]], dim=-2).reshape(y.shape)
            h <<= 1
        return y

    rotated = had(x.float() * d1).to(torch.bfloat16).float()
    lo, hi = rotated.amin(dim=-1), rotated.amax(dim=-1)
    scale = ((hi - lo) / 15.0).clamp_min(1e-6)
    zp_raw = -lo / scale
    zp = torch.where(zp_raw >= 0, (zp_raw + 0.5).trunc(), (zp_raw - 0.5).trunc()).clamp(0, 15)
    s = rotated * (1.0 / scale).unsqueeze(-1) + zp.unsqueeze(-1)
    q = torch.where(s >= 0, (s + 0.5).trunc(), (s - 0.5).trunc()).clamp(0, 15).to(torch.int32)
    bits = (scale.view(torch.int32) & -16) | zp.to(torch.int32)
    deq = (q - (bits & 0xF).unsqueeze(-1)).float() * (bits & -16).view(torch.float32).unsqueeze(-1)
    return (had(deq) * d1 / x.shape[-1]).to(x.dtype)


def test_kv_fake_kv4_is_the_served_page():
    """attention.kv_fake("kv4") returns exactly the served int4 cache's K / V; kv2 (2 bits) errs more; kv2kc (K per
    channel over 128-token groups) keeps shapes on a ragged length."""
    import attention

    torch.manual_seed(0)
    k = (torch.randn(1, 2, 300, 256) * torch.linspace(0.1, 4, 256)).to(torch.bfloat16)
    v = torch.randn(1, 2, 300, 256).to(torch.bfloat16)
    k4, v4 = attention.kv_fake(k, v, "kv4")
    assert torch.equal(k4, _served_kv4_round_trip(k)) and torch.equal(v4, _served_kv4_round_trip(v))
    k2, v2 = attention.kv_fake(k, v, "kv2")
    err = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
    assert err(k4, k) < 0.15 and err(k2, k) > 2 * err(k4, k) and err(v2, v) > 2 * err(v4, v)
    kc, vc = attention.kv_fake(k, v, "kv2kc")
    assert kc.shape == k.shape and kc.dtype == k.dtype and torch.equal(vc, v2) and not torch.equal(kc, k2)


def test_eval_variant_top_k(setup):
    """Eval-only screen: a routed top-k variant changes the slice and every router's top_k is restored after."""
    tmp, _ = setup
    out = tmp / "variant-topk"
    seen = {}
    real = train.eval_variant

    def spy(cfg_, model, *a, **k):
        seen["model"] = model
        return real(cfg_, model, *a, **k)

    train.eval_variant = spy
    try:
        train.train(cfg(tmp, out, loss="kl", total_steps=0, eval_variants=[{"name": "k1", "top_k": 1}]), device="cpu",
                    make_opt=opt)
    finally:
        train.eval_variant = real
    ev = {r["set"]: r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines()) if r.get("event") == "eval"}
    assert ev["slice:k1"]["heldout_kl"] != ev["slice"]["heldout_kl"]
    routers = [m for m in seen["model"].modules() if type(m).__name__.endswith("TopKRouter")]
    assert routers and all(m.top_k > 1 for m in routers)


def test_run_top_k_routes_every_read_and_step(setup):
    """Top-k heal: ``top_k`` sets every trunk router for the run; its step-0 slice equals the top-k eval variant's on
    the unchanged model, and training steps run at that top_k."""
    tmp, _ = setup
    seen = {}
    real = train.load_model

    def spy(*a, **k):
        seen["model"], head = real(*a, **k)
        return seen["model"], head

    a, b = tmp / "topk-variant", tmp / "topk-run"
    train.train(cfg(tmp, a, loss="kl", total_steps=0, eval_variants=[{"name": "k1", "top_k": 1}]), device="cpu",
                make_opt=opt)
    train.load_model = spy
    try:
        train.train(cfg(tmp, b, loss="kl", total_steps=2, eval_steps=0, top_k=1), device="cpu", make_opt=opt)
    finally:
        train.load_model = real
    ev = lambda out: [r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines())  # noqa: E731
                      if r.get("event") == "eval"]
    variant = next(r for r in ev(a) if r["set"] == "slice:k1")
    run0 = next(r for r in ev(b) if r["set"] == "slice" and r["step"] == 0)
    assert abs(run0["heldout_kl"] - variant["heldout_kl"]) < 1e-6
    steps = [r for r in map(json.loads, (b / "events.jsonl").read_text().splitlines()) if r.get("event") == "step"]
    assert len(steps) == 2 and all(r["loss"] == r["loss"] for r in steps)
    routers = [m for m in seen["model"].modules() if type(m).__name__.endswith("TopKRouter")]
    assert routers and all(m.top_k == 1 for m in routers)


def test_channel_fisher_scores_every_trunk_expert_channel(setup):
    """Eval-only channel_fisher: one [E, I] score set per trunk expert module from the held-out examples past the
    skip (non-negative, not all zero); weights and requires_grad come back unchanged."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    tmp, _ = setup
    out = tmp / "fisher"
    seen = {}
    real = train.load_model

    def spy(*a, **k):
        seen["model"], seen["head"] = real(*a, **k)
        seen["before"] = {n: p.detach().clone() for n, p in seen["model"].named_parameters()}
        seen["grad"] = {n: p.requires_grad for n, p in seen["model"].named_parameters()}
        return seen["model"], seen["head"]

    train.load_model = spy
    try:
        train.train(cfg(tmp, out, loss="kl", total_steps=0, eval_examples=1, channel_fisher={"n": 1}), device="cpu",
                    make_opt=opt)
    finally:
        train.load_model = real
    got = torch.load(out / "channel-fisher.pt", weights_only=False)
    model = seen["model"]
    mods = {n: mod for n, mod in model.named_modules() if isinstance(mod, m.Qwen4ExpTextExperts)
            and ".language_model.layers." in f".{n}"}
    assert got["examples"] == 1 and got["files"] and set(got["scores"]) == set(mods)
    for n, s in got["scores"].items():
        e, i = mods[n].down_proj.shape[0], mods[n].down_proj.shape[-1]
        assert s["fisher"].shape == (e, i) and (s["fisher"] >= 0).all() and torch.allclose(s["taylor"], s["attr"])
    assert any(s["fisher"].sum() > 0 for s in got["scores"].values())
    assert all(torch.equal(p, seen["before"][n]) for n, p in model.named_parameters())
    assert all(p.requires_grad == seen["grad"][n] for n, p in model.named_parameters())


def test_eval_variants_served_dense_int8_and_indexer_fp8(setup):
    """The 4xL4 stack's variant: INT8 dense rows on every served family (count checked) and e4m3 indexer keys; the
    weights come back bit-exact and the flags off after the read."""
    import attention
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    tmp, _ = setup
    out = tmp / "served-variant"
    seen = {}
    real = train.load_model

    def spy(*a, **k):
        seen["model"], head = real(*a, **k)
        seen["before"] = {n: p.detach().clone() for n, p in train.dense_int8_params(seen["model"])}
        return seen["model"], head

    variants = [{"name": "int8", "dense_int8": True}, {"name": "served", "kv": "kv4", "idx_fp8": True,
                                                      "dense_int8": True}]
    train.load_model = spy
    try:
        attention.install(mode="selected")
        attention.INDEX["fp8_calls"] = 0
        train.train(cfg(tmp, out, loss="kl", total_steps=0, eval_variants=variants), device="cpu", make_opt=opt)
        ev = {r["set"]: r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines())
              if r.get("event") == "eval"}
        types = seen["model"].config.text_config.layer_types
        want = 3 * types.count("linear_attention") + 4 * (len(types) - types.count("linear_attention")) + 1
        assert ev["slice:int8"]["dense_int8_weights"] == want and ev["slice:served"]["dense_int8_weights"] == want
        assert ev["slice:int8"]["heldout_kl"] != ev["slice"]["heldout_kl"]
        assert attention.INDEX["fp8_calls"] > 0 and not attention.INDEX["fp8"] and attention.KVQ["mode"] is None
        after = dict(train.dense_int8_params(seen["model"]))
        assert all(torch.equal(after[n], w) for n, w in seen["before"].items())
        w = torch.randn(8, 32, dtype=torch.bfloat16)
        s = w.float().abs().amax(1, keepdim=True) / 127  # served: int8 codes x fp32 scale, dequantized to BF16 (prefill)
        assert torch.equal(train.int8_rows(w), (torch.round(w.float() / s).clamp(-127, 127) * s).to(torch.bfloat16))
        assert (train.int8_rows(w).float() - w.float()).abs().le(s / 2 + w.float().abs() / 128).all()
    finally:
        train.load_model = real
        for cls, flag in ((m.Qwen4ExpTextQSAIndexer, "_rl1_vectorized"), (m.Qwen4ExpTextAttention, "_rl1_selected")):
            if getattr(cls, flag, False):
                cls.forward = cls._rl1_original
                setattr(cls, flag, False)
        attention.SPARSE["mode"], attention.SEL["on"] = "masked", False
        attention.KVQ["mode"] = None
        attention.INDEX["fp8"] = False


def test_eval_variants_kv_and_expert_grid(setup):
    """Eval-only screen: after the step-0 slice, one slice per variant (served KV through the QSA layers, the int3
    expert grid); the run's KV mode and int4 scales are restored after."""
    import attention
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    tmp, _ = setup
    out = tmp / "variants"
    variants = [{"name": "kv4", "kv": "kv4"}, {"name": "kv2", "kv": "kv2"}, {"name": "int3", "levels": 8, "zero": 4}]
    try:
        attention.install(mode="selected")
        train.train(cfg(tmp, out, loss="kl", fake_quant=True, total_steps=0, eval_variants=variants), device="cpu",
                    make_opt=opt)
        ev = {r["set"]: r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines())
              if r.get("event") == "eval"}
        assert {"slice", "slice:kv4", "slice:kv2", "slice:int3"} <= set(ev) and attention.KVQ["calls"] > 0
        base = ev["slice"]["heldout_kl"]
        assert all(ev[f"slice:{n}"]["heldout_kl"] != base for n in ("kv2", "int3"))
        assert attention.KVQ["mode"] is None and qat.QAT["levels"] == 16 and qat.QAT["zero"] == 8
    finally:
        for cls, flag in ((m.Qwen4ExpTextQSAIndexer, "_rl1_vectorized"), (m.Qwen4ExpTextAttention, "_rl1_selected")):
            if getattr(cls, flag, False):
                cls.forward = cls._rl1_original
                setattr(cls, flag, False)
        attention.SPARSE["mode"], attention.SEL["on"] = "masked", False
        attention.KVQ["mode"] = None
        qat.QAT["on"] = False
        qat.SCALES.clear()


def test_depth_row_weights_by_turn_start():
    """Each reply run takes the weight of its first token's position bucket (the turn's prompt length); rows past the
    window cut only; prompt rows 1.0."""
    labels = torch.full((40,), -100)
    labels[5:9] = 1  # turn starting at 5: bucket [0, 8) -> 1.0, even past 8
    labels[12:14] = 1  # starts at 12: [8, 16) -> 2.0
    labels[20:30] = 1  # starts at 20: [16, ...) -> 3.0
    w = train.depth_row_weights([[0, 1.0], [8, 2.0], [16, 3.0]], labels, cut=3)
    assert w.shape == (40 - 3 - 1,)
    full = torch.ones(40)
    full[5:9], full[12:14], full[20:30] = 1.0, 2.0, 3.0
    assert torch.equal(w, full[4:])


def test_kl_topk_weights():
    torch.manual_seed(0)
    s = torch.randn(6, 50)
    lp, i = torch.log_softmax(torch.randn(6, 50), -1).topk(8, -1)
    base = qat.kl_topk(s, i, lp)
    assert torch.allclose(qat.kl_topk(s, i, lp, torch.ones(6)), base, rtol=1e-6)
    w = torch.tensor([1.0, 1.0, 1.24, 1.24, 1.3, 1.3])
    rows = torch.stack([qat.kl_topk(s[j:j + 1], i[j:j + 1], lp[j:j + 1]) for j in range(6)])
    assert torch.allclose(qat.kl_topk(s, i, lp, w), (rows * w).sum(), rtol=1e-6)


def test_trainer_depth_weights(setup):
    """depth_weights scale the trained KL per turn and leave evals unweighted: all 1.0 = off; all 2.0 = 2x."""
    tmp, _ = setup
    model, head = train.load_model(tmp / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    ex = train.load_example(tmp / "data/train/e1.pt", tmp / "data/teacher/train/e1.pt")
    got = {}
    for name, dw in (("off", []), ("one", [[0, 1.0]]), ("two", [[0, 2.0]])):
        c = cfg(tmp, tmp / "x", loss="kl", window=256, depth_weights=dw)
        got[name] = float(train.example_loss(model, head, box, ex, c, train=True)[5])
        got[name + "_eval"] = float(train.example_loss(model, head, box, ex, c, train=False)[5])
        model.zero_grad(set_to_none=True)
    assert abs(got["one"] - got["off"]) <= 1e-5 * abs(got["off"]) and abs(got["two"] - 2 * got["off"]) <= 1e-5 * abs(got["off"])
    assert got["two_eval"] == got["off_eval"]
    train.check_depth_weights([[0, 1.0], [65536, 1.24], [98304, 1.3]], "kl")
    for bad, loss in (([[100, 1.0]], "kl"), ([[0, 1.0], [0, 2.0]], "kl"), ([[0, 0.0]], "kl"), ([[0, 1.0]], "ce")):
        with pytest.raises(ValueError):
            train.check_depth_weights(bad, loss)


def test_trainer_mtp_int8_head(setup):
    """mtp_int8_head moves only the MTP draft's logits onto the INT8 lm_head: the trunk KL does not change."""
    tmp, _ = setup
    model, head = train.load_model(tmp / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    ex = train.load_example(tmp / "data/train/e1.pt", tmp / "data/teacher/train/e1.pt")
    got = {}
    qat.DENSE["on"] = True
    try:
        for flag in (False, True):
            c = cfg(tmp, tmp / "x", loss="kl", window=256, mtp_int8_head=flag)
            got[flag] = train.example_loss(model, head, box, ex, c, train=False)
    finally:
        qat.DENSE["on"] = False
    assert float(got[False][5]) == float(got[True][5]) and float(got[False][2]) != float(got[True][2])


def scoring_root(setup, tmp_path, monkeypatch):
    tmp, t = setup
    root = tmp_path / "data"
    shutil.copytree(tmp / "data", root, ignore=shutil.ignore_patterns("teacher"))
    monkeypatch.setitem(teacher.SANITY, "first", 0)
    monkeypatch.setitem(teacher.SIZES, "rows", 2048)
    monkeypatch.setitem(teacher.SIZES, "chunk", 8192)
    return t, root, teacher.order(root)[:2]


def test_score_oom_retry_frees_the_failed_pass(setup, tmp_path, monkeypatch):
    """A pass that runs out of card memory: the re-stage runs after the failed pass's tensors are gone, at 1,024 rows
    (sft-score-long-v1 shard 0: inside the except block its traceback held them, so the re-stage ran out too)."""
    t, root, entries = scoring_root(setup, tmp_path, monkeypatch)
    real_topk, real_stage, box = teacher.topk, teacher.feed.stage, {}

    def topk(*a, **k):
        if "ref" not in box:
            held = torch.zeros(8)
            box["ref"] = weakref.ref(held)
            raise torch.cuda.OutOfMemoryError("test")
        return real_topk(*a, **k)

    def stage(*a, **k):
        if "ref" in box and "gone" not in box and threading.current_thread() is threading.main_thread():
            box["gone"], box["handling"] = box["ref"]() is None, sys.exc_info()[0] is not None
        return real_stage(*a, **k)

    monkeypatch.setattr(teacher, "topk", topk)
    monkeypatch.setattr(teacher.feed, "stage", stage)
    n, _ = teacher.score(t, root, entries, 4, [7, 11, 13], deadline=float("inf"))
    assert n == 2 and box["gone"] and not box["handling"] and teacher.SIZES["rows"] == 1024


def test_score_staging_oom_waits_for_the_pass(setup, tmp_path, monkeypatch):
    """The loader thread's staging runs out of card memory beside a pass: it retries, the job goes on."""
    t, root, entries = scoring_root(setup, tmp_path, monkeypatch)
    real_stage, box = teacher.feed.stage, {"n": 0}

    def stage(*a, **k):
        if threading.current_thread() is not threading.main_thread() and box["n"] < 2:
            box["n"] += 1
            raise torch.cuda.OutOfMemoryError("test")
        return real_stage(*a, **k)

    monkeypatch.setattr(teacher.feed, "stage", stage)
    n, _ = teacher.score(t, root, entries, 4, [7, 11, 13], deadline=float("inf"))
    assert n == 2 and box["n"] == 2


def test_score_stage_inline(setup, tmp_path, monkeypatch):
    """--stage-inline: nothing staged on the loader thread, every file as with stage-ahead, and an OOM pass retries."""
    t, root, entries = scoring_root(setup, tmp_path, monkeypatch)
    teacher.score(t, root, entries, 4, [7, 11, 13], deadline=float("inf"))
    want = {e["file"]: torch.load(root / "teacher" / e["file"], weights_only=False) for _, e in entries}
    shutil.rmtree(root / "teacher")
    monkeypatch.setitem(teacher.SWITCH, "inline", True)
    real_topk, real_stage, box = teacher.topk, teacher.feed.stage, {"threads": set(), "oom": 0}

    def stage(*a, **k):
        box["threads"].add(threading.current_thread() is threading.main_thread())
        return real_stage(*a, **k)

    def topk(*a, **k):
        if not box["oom"]:
            box["oom"] = 1
            raise torch.cuda.OutOfMemoryError("test")
        return real_topk(*a, **k)

    monkeypatch.setattr(teacher.feed, "stage", stage)
    monkeypatch.setattr(teacher, "topk", topk)
    n, _ = teacher.score(t, root, entries, 4, [7, 11, 13], deadline=float("inf"))
    assert n == 2 and box["threads"] == {True} and teacher.SIZES["rows"] == 1024
    for _, e in entries:
        got = torch.load(root / "teacher" / e["file"], weights_only=False)
        tensors = [k for k, v in want[e["file"]].items() if torch.is_tensor(v)]
        assert tensors and all(torch.equal(got[k], want[e["file"]][k]) for k in tensors)
