"""scripts/sft/train.py on CPU with a tiny student: a 3-step run with a checkpoint at step 2, and a resume from it
that reproduces step 3; the held-out eval and MTP self-distillation run."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/sft"), str(ROOT / "tests/golden")]
import common  # noqa: E402
import mtp  # noqa: E402
import train  # noqa: E402


def data(root, split, n, seed, index):
    """build.py's layout: <root>/<split>/<name>.pt (1-D int32 rows), one index.json for both splits."""
    g = torch.Generator().manual_seed(seed)
    (root / split).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        ids = torch.randint(10, 2000, (200,), generator=g)
        labels = torch.full_like(ids, -100)
        labels[120:200] = ids[120:200]
        torch.save({"input_ids": ids.to(torch.int32), "labels": labels.to(torch.int32), "tail": 128}, root / split / f"e{i}.pt")
        rows.append({"file": f"{split}/e{i}.pt", "reply_tokens": 80, "tokens": 200, "tail": 128})
    index[split] = {"files": rows}
    (root / "index.json").write_text(json.dumps(index))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sft")
    common.tiny_student_on_disk(tmp / "student")
    index = {}
    data(tmp / "data", "train", 4, 1, index)
    data(tmp / "data", "heldout", 2, 2, index)
    return tmp


def cfg(tmp, out, **kw):
    base = dict(student=str(tmp / "student"), data=str(tmp / "data"), out=str(out), lr=1e-3, warmup=1, rewarm=1,
                total_steps=3, batch_reply_tokens=150, window=128, prefix_chunk=32, loss_chunk=48,
                ckpt_steps=2, eval_steps=3, eval_examples=2, fast_attention=False)
    return train.Config(**{**base, **kw})


def opt(params, lr):  # torchao's 8-bit AdamW where installed (the trainer image); the CI image has no torchao
    try:
        return train.adamw8bit(params, lr)
    except ImportError:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)


def steps(out):
    return {r["step"]: r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines()) if r.get("event") == "step"}


def test_run_and_resume(setup):
    tmp = setup
    a = tmp / "a"
    train.train(cfg(tmp, a), device="cpu", make_opt=opt)
    sa = steps(a)
    assert sorted(sa) == [1, 2, 3] and all(r["loss"] > 0 and r["mtp_kl"] > 0 for r in sa.values())
    assert sa[3]["loss"] < sa[1]["loss"]
    assert (a / "ckpt-2/state.json").is_file() and (a / "ckpt-3/state.json").is_file()
    b = tmp / "b"  # resume from a's step-2 checkpoint
    b.mkdir()
    import shutil

    shutil.copytree(a / "ckpt-2", b / "ckpt-2")
    train.train(cfg(tmp, b), device="cpu", make_opt=opt)
    sb = steps(b)
    assert sorted(sb) == [3]
    assert abs(sb[3]["loss"] - sa[3]["loss"]) < 1e-6 * max(1.0, sa[3]["loss"])
    assert abs(sb[3]["grad_norm"] - sa[3]["grad_norm"]) < 1e-4 * sa[3]["grad_norm"]
    evals = [r for r in map(json.loads, (a / "events.jsonl").read_text().splitlines()) if r.get("event") == "eval"]
    assert [e["step"] for e in evals] == [0, 3] and 0 <= evals[-1]["accept1"] <= 1


def test_eval_only(setup, tmp_path):
    """total_steps 0, no warm-up: the step-0 held-out read of a student (any checkpoint's student folder), no step, no
    checkpoint (diagnostic reads: fake quant on/off on one student)."""
    tmp = setup
    out = tmp_path / "ev"
    train.train(cfg(tmp, out, total_steps=0, warm_start=False, ckpt_steps=0), device="cpu", make_opt=opt)
    ev = [r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines())]
    assert [(e["step"], e["set"]) for e in ev if e.get("event") == "eval"] == [(0, "slice")]
    assert not steps(out) and not list(out.glob("ckpt-*")) and ev[-1]["event"] == "done"
    # warm_start on: an eval-only run still skips the train-window warm-up (29 Oct OOM on a 256-expert student)
    out2 = tmp_path / "ev-warm"
    seen = []
    real = train.warm_window
    train.warm_window = lambda *a, **k: seen.append(1) or real(*a, **k)
    try:
        train.train(cfg(tmp, out2, total_steps=0, warm_start=True, ckpt_steps=0), device="cpu", make_opt=opt)
    finally:
        train.warm_window = real
    ev2 = [r for r in map(json.loads, (out2 / "events.jsonl").read_text().splitlines())]
    assert not seen and [e["set"] for e in ev2 if e.get("event") == "eval"] == ["slice"]


def test_arc2_heldout_eval_at_its_steps(setup, tmp_path):
    """The ARC-2 slice reads at arc2_heldout_steps only, as set "arc2", with its own KL."""
    out = tmp_path / "a2"
    train.train(cfg(setup, out, total_steps=2, warm_start=False, ckpt_steps=0, arc2_heldout_data=str(setup / "data"),
                    arc2_heldout_steps=[0, 2]), device="cpu", make_opt=opt)
    ev = [r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines()) if r.get("event") == "eval"]
    got = [e for e in ev if e["set"] == "arc2"]
    assert [e["step"] for e in got] == [0, 2] and all(e["heldout_kl"] == e["heldout_kl"] and e["seconds"] >= 0 for e in got)
    assert any(e["set"] == "slice" for e in ev)


def test_repeat_examples_trains_on_a_fixed_few():
    """repeat_examples N: every batch comes from epoch 0's first N files; 0 keeps the full shuffled epochs."""
    index = {"examples": [{"file": f"e{i}", "reply_tokens": 100, "tokens": 1000 + i} for i in range(20)]}
    cfg_ = train.Config(student="s", data="d", out="o", batch_reply_tokens=150, repeat_examples=3)
    fixed = set(train.order(index, cfg_.seed, 0)[:3])
    cursor, seen = (0, 0), set()
    for _ in range(12):
        picked, tokens, cursor = train.next_batch(cfg_, index, cursor)
        seen.update(picked)
        assert tokens >= 150
    assert seen == fixed and cursor[0] >= 4
    full = train.Config(student="s", data="d", out="o", batch_reply_tokens=150)
    cursor, seen = (0, 0), set()
    for _ in range(10):
        picked, _, cursor = train.next_batch(full, index, cursor)
        seen.update(picked)
    assert len(seen) == 20


def test_repeat_examples_after_resume_draws_only_seen():
    """N/R read: resumed at epoch-0 cursor P with repeat_examples=P, every batch comes from the P examples already
    trained on; without it the next batch is new data."""
    index = {"examples": [{"file": f"e{i}", "reply_tokens": 100, "tokens": 1000 + i} for i in range(30)]}
    seen = set(train.order(index, 0, 0)[:11])
    rep = train.Config(student="s", data="d", out="o", batch_reply_tokens=150, repeat_examples=11)
    cursor, got = (0, 11), set()
    for _ in range(8):
        picked, _, cursor = train.next_batch(rep, index, cursor)
        got.update(picked)
    assert got <= seen and len(got) > 5
    new = train.next_batch(train.Config(student="s", data="d", out="o", batch_reply_tokens=150), index, (0, 11))[0]
    assert not set(new) & seen


def test_init_ckpt_resumes_from_another_run(setup, tmp_path):
    """init_ckpt: a fresh out resumes another run's checkpoint (step, cursor, weights, optimizer); eval_resume reads
    the slice at the resumed step; a checkpoint already in out wins."""
    a, b = tmp_path / "a", tmp_path / "b"
    train.train(cfg(setup, a, stop_step=2), device="cpu", make_opt=opt)
    train.train(cfg(setup, b, init_ckpt=str(a / "ckpt-2"), eval_resume=True, ckpt_steps=0), device="cpu", make_opt=opt)
    ev = [json.loads(l) for l in (b / "events.jsonl").read_text().splitlines()]
    start = next(r for r in ev if r.get("event") == "start")
    assert start["step"] == 2 and start["resume"] == str(a / "ckpt-2")
    assert [r["step"] for r in ev if r.get("event") == "step"] == [3]
    assert [r["step"] for r in ev if r.get("event") == "eval" and r["set"] == "slice"] == [2, 3]
    a3 = tmp_path / "a3"  # same run continued in place: step 3's loss equals the init_ckpt resume's
    train.train(cfg(setup, a3, stop_step=2), device="cpu", make_opt=opt)
    train.train(cfg(setup, a3), device="cpu", make_opt=opt)
    s3 = [json.loads(l) for l in (a3 / "events.jsonl").read_text().splitlines() if '"step"' in l]
    ref = [r for r in s3 if r.get("event") == "step" and r["step"] == 3][0]
    got = [r for r in ev if r.get("event") == "step"][0]
    assert abs(ref["loss"] - got["loss"]) < 1e-6
    c = tmp_path / "c"  # init_fresh_data: another pool's cursor starts at its first batch
    train.train(cfg(setup, c, init_ckpt=str(a / "ckpt-2"), init_fresh_data=True, ckpt_steps=0), device="cpu", make_opt=opt)
    first = [json.loads(l) for l in (c / "events.jsonl").read_text().splitlines() if '"event": "step"' in l][0]
    fresh = [json.loads(l) for l in (a / "events.jsonl").read_text().splitlines() if '"event": "step"' in l][0]
    assert first["step"] == 3 and first["examples"] == fresh["examples"] and first["reply_tokens"] == fresh["reply_tokens"]


def test_two_eval_sets(setup, tmp_path):
    """The slice (first eval_examples) on its cadence and at the end; the full set (first eval_full_examples, the
    slice plus the rest read from disk) at eval_full_steps only; the slice equals a full read cut to the slice."""
    tmp = setup
    out = tmp_path / "e"
    train.train(cfg(tmp, out, eval_examples=1, eval_full_examples=2, eval_full_steps=[0, 2], eval_steps=4, ckpt_steps=0),
                device="cpu", make_opt=opt)
    evals = [r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines()) if r.get("event") == "eval"]
    assert [(e["step"], e["set"], e["examples"]) for e in evals] == [(0, "slice", 1), (0, "full", 2), (2, "full", 2),
                                                                     (3, "slice", 1)]
    assert evals[1]["heldout_tokens"] > evals[0]["heldout_tokens"]


def test_full_eval_prefetch_matches_loading_at_the_eval(setup, tmp_path, monkeypatch):
    """eval_full_lead > 0: the full set's extra examples load in a background thread before the eval step and the
    read equals loading them at the eval; a failed prefetch falls back to the load at the eval."""
    reads, threads = {}, {}
    real = train.load_example

    def counted(path, *a, **k):
        import threading
        threads.setdefault(threading.current_thread().name, 0)
        threads[threading.current_thread().name] += 1
        return real(path, *a, **k)
    monkeypatch.setattr(train, "load_example", counted)
    for lead in (0, 2):
        out = tmp_path / f"p{lead}"
        train.train(cfg(setup, out, eval_examples=1, eval_full_examples=2, eval_full_steps=[3], eval_steps=0,
                        eval_full_lead=lead, ckpt_steps=0), device="cpu", make_opt=opt)
        reads[lead] = [r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines())
                       if r.get("event") == "eval" and r["set"] == "full"]
    assert [e["step"] for e in reads[2]] == [3] and abs(reads[0][0]["heldout_kl"] - reads[2][0]["heldout_kl"]) < 1e-6
    assert threads.get("full-eval-3") == 1

    out = tmp_path / "pf"
    calls = []

    def flaky(path, *a, **k):
        import threading
        if threading.current_thread().name.startswith("full-eval"):
            raise OSError("volume gone")
        calls.append(path)
        return real(path, *a, **k)
    monkeypatch.setattr(train, "load_example", flaky)
    train.train(cfg(setup, out, eval_examples=1, eval_full_examples=2, eval_full_steps=[3], eval_steps=0,
                    eval_full_lead=2, ckpt_steps=0), device="cpu", make_opt=opt)
    rows = list(map(json.loads, (out / "events.jsonl").read_text().splitlines()))
    assert [r["step"] for r in rows if r.get("event") == "full_prefetch_failed"] == [3]
    assert [r["examples"] for r in rows if r.get("event") == "eval" and r["set"] == "full"] == [2]


def test_mtp_trunk_unchanged_by_mtp_loss(setup):
    """The MTP term moves only the head: trunk gradients equal with mtp_weight 0 and 1."""
    tmp = setup
    model, head = train.load_model(tmp / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    ex = train.load_example(tmp / "data/train/e0.pt")
    grads = []
    for w in (0.0, 1.0):
        model.zero_grad(); head.zero_grad()
        train.example_loss(model, head, box, ex, cfg(tmp, tmp / "x", mtp_weight=w))
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and all(torch.equal(grads[0][k], grads[1][k]) for k in grads[0])


def test_schedule_linear0_and_rewarm_constant():
    """lr probes from a checkpoint: linear0 runs lr -> 0 over [start, stop_step); rewarm 5 + final_lr_frac 1 is a
    5-step ramp then constant."""
    c = train.Config(student="s", data="d", out="o", lr=3e-6, rewarm=0, total_steps=231, stop_step=140, final_lr_frac=1.0,
                     decay="linear0")
    lrs = [train.schedule(c, s, 116) for s in range(116, 140)]
    assert abs(lrs[0] - 3e-6) < 1e-18 and abs(lrs[-1] - 3e-6 / 24) < 1e-18
    assert all(a > b for a, b in zip(lrs, lrs[1:]))
    a = train.Config(student="s", data="d", out="o", lr=1e-5, rewarm=5, total_steps=231, stop_step=140, final_lr_frac=1.0)
    got = [train.schedule(a, s, 116) for s in range(116, 140)]
    assert [round(x / 1e-5, 6) for x in got[:6]] == [0.2, 0.4, 0.6, 0.8, 1.0, 1.0] and len(set(got[4:])) == 1


def test_schedule_linear_end_holds_until_decay_start():
    cfg = train.Config(student="s", data="d", out="o", lr=3e-6, warmup=0, rewarm=0, final_lr_frac=0.1, total_steps=1100,
                       stop_step=1100, decay="linear_end", decay_start=1000)
    assert train.schedule(cfg, 731, 731) == 3e-6 and train.schedule(cfg, 1000, 731) == 3e-6
    assert train.schedule(cfg, 1099, 731) == pytest.approx(3e-7)
    assert 3e-7 < train.schedule(cfg, 1050, 731) < 3e-6


def test_schedule_linear_end_lands_on_the_final_frac():
    """A resume at 541 with total 561: 3e-6 on the first step, 3e-7 on the last, falling; cosine on the same config
    would start near its final lr because its frac counts from step 0."""
    c = train.Config(student="s", data="d", out="o", lr=3e-6, rewarm=0, total_steps=561, final_lr_frac=0.1,
                     decay="linear_end")
    lrs = [train.schedule(c, s, 541) for s in range(541, 561)]
    assert abs(lrs[0] - 3e-6) < 1e-15 and abs(lrs[-1] - 3e-7) < 1e-15
    assert all(a > b for a, b in zip(lrs, lrs[1:]))
    cosine = train.Config(student="s", data="d", out="o", lr=3e-6, rewarm=0, total_steps=561, final_lr_frac=0.1)
    assert train.schedule(cosine, 541, 541) < 4e-7


def test_mtp_keeps_its_expert_width_on_a_channel_cut_student():
    """A channel-cut student (moe_intermediate_size cut) keeps its uncut MTP block's experts at
    mtp_moe_intermediate_size."""
    import copy
    text = copy.deepcopy(common.tiny_student().config.text_config)
    full = text.moe_intermediate_size
    text.mtp_moe_intermediate_size, text.moe_intermediate_size = full, full // 2
    head = mtp.MtpHead(text, experts=8, top_k=4)
    assert head.config.moe_intermediate_size == full
    assert head.layers[0].mlp.experts.down_proj.shape[-1] == full


def test_prune_keeps_pinned_steps(tmp_path):
    import ckpt
    for s in (1, 2, 3, 4, 5):
        (tmp_path / f"ckpt-{s}").mkdir()
    (tmp_path / "pins.json").write_text(json.dumps([2]))
    ckpt.prune(tmp_path, 2)
    assert sorted(p.name for p in tmp_path.glob("ckpt-*")) == ["ckpt-2", "ckpt-4", "ckpt-5"]


def test_schedule_constant_after_rewarm():
    cfg = train.Config(student="s", data="d", out="o", lr=3e-6, warmup=15, rewarm=15, total_steps=1594, decay="linear_end", decay_start=1594, final_lr_frac=0.1)
    lrs = [train.schedule(cfg, s, 984) for s in (984, 990, 998, 999, 1300, 1593, 1594)]
    assert lrs[0] == pytest.approx(3e-6 / 15) and lrs[2] == pytest.approx(3e-6) and all(x == pytest.approx(3e-6) for x in lrs[3:])


def test_schedule_start_keeps_the_runs_first_step_on_its_own_resume(tmp_path):
    """joint c942 2 Nov: a preemption resume from the run's own ckpt-1059 re-warmed 15 steps (1e-6 at 1064). A same-run
    resume keeps the run's first step (no warm-up, decay continues); a new leg from init_ckpt starts at its step."""
    init = tmp_path / "prev" / "ckpt-942"
    init.mkdir(parents=True)
    (init / "state.json").write_text(json.dumps({"step": 942}))
    out = tmp_path / "run"
    out.mkdir()
    cfg = train.Config(student="s", data="d", out=str(out), init_ckpt=str(init), lr=3e-6, warmup=0, rewarm=15,
                       total_steps=1524, decay="linear_end", decay_start=1400, final_lr_frac=0.1)
    assert train.schedule_start(cfg, out, init, 942) == 942  # new leg: rewarm from its step, recorded
    assert json.loads((out / "schedule.json").read_text()) == {"start": 942}
    own = out / "ckpt-1059"
    own.mkdir()
    s = train.schedule_start(cfg, out, own, 1059)
    assert s == 942 and train.schedule(cfg, 1060, s) == pytest.approx(3e-6)  # no re-warm after the preemption
    assert train.schedule(cfg, 1460, s) == pytest.approx(train.schedule(cfg, 1460, 942))  # decay continues
    (out / "schedule.json").unlink()  # a run from before the file: its init checkpoint's step
    assert train.schedule_start(cfg, out, own, 1059) == 942


def test_arc2_heldout_every_due_and_knob(tmp_path):
    assert train.arc2_due(942, [942, 985], 0) and not train.arc2_due(950, [942, 985], 0)
    assert train.arc2_due(960, [942], 12) and not train.arc2_due(961, [942], 12)
    seen = {}
    (tmp_path / "knobs.json").write_text(json.dumps({"arc2_heldout_every": 12}))
    assert train.read_knobs(tmp_path, seen) == {"arc2_every": 12}
    assert train.read_knobs(tmp_path, seen) == {}  # unchanged: not fresh again
    (tmp_path / "knobs.json").write_text(json.dumps({"arc2_heldout_every": 0}))
    assert train.read_knobs(tmp_path, seen) == {"arc2_every": 0}


def _header_student(folder, experts=8, layers=2):
    """A safetensors folder of headers only (memest reads no weights)."""
    import struct
    folder.mkdir(parents=True, exist_ok=True)
    meta = {f"model.language_model.layers.{i}.mlp.experts.{p}": {"dtype": "BF16", "shape": [experts, 64, 32],
                                                               "data_offsets": [0, 0]}
            for i in range(layers) for p in ("gate_up_proj", "down_proj")}
    meta["model.language_model.embed_tokens.weight"] = {"dtype": "BF16", "shape": [1000, 32], "data_offsets": [0, 0]}
    raw = json.dumps(meta).encode()
    (folder / "m.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "m.safetensors" for k in meta}}))
    (folder / "config.json").write_text(json.dumps({"text_config": {"num_experts": experts}}))
    return folder


def test_memest_counts_the_warm_up_example_and_steps_kept_layers_down(tmp_path):
    import memest
    s = str(_header_student(tmp_path / "s"))
    ref = {"student": s, "gpus": 1, "keep_layers": 0, "peak_gib": 200.0, "tokens": 100_000, "window": 40_000}
    p = lambda k, tokens=100_000, window=40_000: memest.predict(s, 1, k, ref=ref, tokens=tokens, window=window)["predicted_peak_gib"]
    assert p(0) == 200.0  # target == reference
    assert abs(p(3) - p(2) - memest.KEEP_GIB_KTOK * 40) < 0.11  # a kept layer holds its window's activations
    assert abs(p(2, window=60_000) - p(2) - 3 * memest.KEEP_GIB_KTOK * 20) < 0.11  # kept + recompute layer scale with the window
    assert p(0, tokens=150_000) > p(0)  # every layer's checkpointed input scales with the longest example
    assert p(0, tokens=30_000, window=40_000) < p(0, tokens=40_000, window=40_000)  # window clipped to the example
    budget = (p(2) + p(1)) / 2
    got = memest.fit(s, 1, 4, budget, ref=ref, tokens=100_000, window=40_000)
    assert got["keep_layers"] == 1 and got["configured_keep_layers"] == 4 and got["predicted_peak_gib"] <= budget
    assert memest.fit(s, 1, 2, 0.0, ref=ref)["keep_layers"] == 0  # nothing fits: stops at 0, as the trainer
    p50d7 = "/assets/sft-heal/student-p50d7-c647"  # a measured student predicts from its own warm-up
    assert memest.reference(p50d7, 4)["peak_gib"] == 222.97 and memest.reference(p50d7 + "/", 2)["gpus"] == 4
    assert memest.reference("/assets/sft-heal/student-p75", 2)["gpus"] == 2
    assert memest.reference("/assets/sft-heal/student-p75", 1) is memest.REF and memest.reference(s, 2) is memest.REF
