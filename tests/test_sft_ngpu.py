"""The N-GPU trainer (shard.py) on CPU over gloo: the expert all-gather / reduce-scatter, dense all-reduce, global-norm
clip and pass alignment against single-process references (3 ranks, uneven expert split, an empty rank); N=1 vs N=2
vs N=3 on the same global batches (KL + fake quant, examples of different lengths, windows and prefix chunk counts):
FP32 within 1e-5, BF16 within BF16 tolerance; a 2-example step on 3 ranks; checkpoints saved at N=2 resumed at N=1
and N=3, and a 1-GPU checkpoint resumed at N=2; 8-bit optimizer states re-sliced between layouts; the launcher with
the shared n-gram table."""
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/sft"), str(ROOT / "tests/golden"), str(ROOT / "tests"), str(ROOT / "src")]
import ckpt  # noqa: E402
import qat  # noqa: E402
import shard  # noqa: E402
import teacher  # noqa: E402
import train  # noqa: E402
from arc_infra.runops.multirank import run_ranks  # noqa: E402
from test_sft_qat import on_disk, qat_model  # noqa: E402

pytestmark = pytest.mark.multirank
TIMEOUT = 600.0

# (tokens, tail): windows 32-128 of a 128 window, 0-7 prefix chunks of 32
TRAIN = [(260, 128), (200, 64), (140, 128), (96, 96), (230, 32), (180, 128), (120, 64), (210, 100)]
HELD = [(150, 128), (100, 64), (220, 96)]


def data(root, split, shapes, seed, index, reply=60):
    g = torch.Generator().manual_seed(seed)
    (root / split).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (n, tail) in enumerate(shapes):
        ids = torch.randint(10, 2000, (n,), generator=g)
        labels = torch.full_like(ids, -100)
        r = min(tail, reply)
        labels[n - r:] = ids[n - r:]
        torch.save({"input_ids": ids.to(torch.int32), "labels": labels.to(torch.int32), "tail": tail},
                   root / split / f"e{i}.pt")
        rows.append({"file": f"{split}/e{i}.pt", "reply_tokens": r, "tokens": n, "tail": tail})
    index[split] = {"files": rows}
    (root / "index.json").write_text(json.dumps(index))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """A g128-shaped tiny student (4 experts: uneven on 3 ranks), asymmetric data scored by a tiny teacher (k = vocab),
    and a uniform data set whose steps take 2 examples."""
    tmp = tmp_path_factory.mktemp("ngpu")
    on_disk(qat_model(0), tmp / "student")
    t = qat_model(1).eval()
    for name, tr, he in (("data", TRAIN, HELD), ("uniform", [(160, 64)] * 6, [(160, 64)] * 2)):
        index = {}
        data(tmp / name, "train", tr, 1, index)
        data(tmp / name, "heldout", he, 2, index)
        for split in ("train", "heldout"):
            for e in index[split]["files"]:
                got = teacher.topk(t, train.load_example(tmp / name / e["file"]), k=2048)
                (tmp / name / "teacher" / e["file"]).parent.mkdir(parents=True, exist_ok=True)
                torch.save(got, tmp / name / "teacher" / e["file"])
    return tmp


def cfg(tmp, out, **kw):
    base = dict(student=str(tmp / "student"), data=str(tmp / "data"), out=str(out), lr=1e-3, warmup=1, rewarm=1,
                total_steps=3, batch_reply_tokens=150, window=128, prefix_chunk=32, loss_chunk=48, ckpt_steps=3,
                eval_steps=3, eval_examples=3, fast_attention=False, teacher_k=2048, loss="kl", fake_quant=True,
                replica_check_every=1, dummy_tokens=32)
    base.update(kw)
    return base


def sgd(params, lr):
    return torch.optim.SGD(params, lr=lr)


def adamw(params, lr):
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)


def ao8(params, lr):  # torchao's 8-bit AdamW (stochastic rounding) where installed
    try:
        return train.adamw8bit(params, lr)
    except ImportError:
        return adamw(params, lr)


OPTS = {"sgd": sgd, "adamw": adamw, "ao8": ao8}


def rank_train(rank, world, inp):
    import faulthandler

    faulthandler.dump_traceback_later(240, exit=False)  # a hang shows each rank's stack in the log
    torch.set_num_threads(2)
    if inp.get("fp32"):
        train.DTYPE["model"] = torch.float32
    if inp.get("selected"):
        import attention

        attention.install(mode="selected")
    if inp.get("oom") and inp["oom"]["rank"] == rank:
        inject_oom(**inp["oom"])
    texts = list(inp.get("knobs") or [])  # rank 0 reads them in turn, the last one for good

    def knobs():
        return (texts.pop(0) if len(texts) > 1 else texts[0]) if texts else ""

    try:
        train.train(train.Config(**inp["cfg"]), device="cpu", make_opt=OPTS[inp["opt"]], knobs=knobs)
    except Exception as error:  # noqa: BLE001
        if not inp.get("catch"):
            raise
        return f"{type(error).__name__}: {error}"
    return rank


def inject_oom(rank, kind, at):
    """This rank's ``at``-th expert ``kind`` ("gather" | "scatter") of the first warm-up raises a CUDA OOM before its
    collective (as an allocation between collectives would)."""
    real, seen = getattr(shard, kind if kind == "gather" else "reduce_scatter"), [0]

    def hook(*a, **k):
        if shard.WARM["on"] and seen[0] >= 0:
            seen[0] += 1
            if seen[0] == at:
                seen[0] = -1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB (test)")
        return real(*a, **k)

    setattr(shard, kind if kind == "gather" else "reduce_scatter", hook)


def run(world, c, opt, fp32=False, selected=False, **extra):
    """``train`` at ``world`` ranks (1: in this process); ``selected``: the selected-attention forward (kv_train)."""
    if world == 1:
        import attention
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m

        saved, threads = dict(train.DTYPE), torch.get_num_threads()
        torch.set_num_threads(2)  # as the ranks: CPU matmul blocking (and its fp32 rounding) follows the thread count
        try:
            if fp32:
                train.DTYPE["model"] = torch.float32
            if selected:
                attention.install(mode="selected")
            knob = extra.get("knobs")
            train.train(train.Config(**c), device="cpu", make_opt=OPTS[opt], **({"knobs": lambda: knob[0]} if knob else {}))
        finally:
            train.DTYPE.update(saved)
            torch.set_num_threads(threads)
            qat.QAT.update(on=False, grid="int4", fused=None)
            qat.SCALES.clear()
            qat.DENSE["on"] = False
            if selected:  # as test_sft_grids: the selected / vectorized forwards off again
                for cls, flag in ((m.Qwen4ExpTextQSAIndexer, "_rl1_vectorized"), (m.Qwen4ExpTextAttention, "_rl1_selected")):
                    if getattr(cls, flag, False):
                        cls.forward = cls._rl1_original
                        setattr(cls, flag, False)
                attention.SPARSE["mode"], attention.SEL["on"] = "masked", False
                attention.KVQ.update(mode=None, train=None)
        return
    return run_ranks(rank_train, world=world, inputs=[{"cfg": c, "opt": opt, "fp32": fp32, "selected": selected, **extra}]
                     * world, timeout=TIMEOUT)


def rows(out, kind):
    return [r for r in map(json.loads, (Path(out) / "events.jsonl").read_text().splitlines()) if r.get("event") == kind]


def weights(folder):
    """The trained tensors of a checkpoint's student, experts fused (one GPU writes them fused, N GPUs per expert)."""
    from safetensors import safe_open

    folder = Path(folder) / "student"
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    got, per = {}, {}
    for k, f in index.items():
        if Path(f).is_absolute():
            continue  # the base student's frozen tensors
        with safe_open(str(folder / f), "pt") as h:
            t = h.get_tensor(k)
        m = shard.PER.match(k)
        if m:
            per.setdefault(m.group(1), {})[(int(m.group(2)), m.group(3))] = t
        else:
            got[k] = t
    for prefix, d in per.items():
        n = 1 + max(e for e, _ in d)
        got[f"{prefix}.gate_up_proj"] = torch.stack([torch.cat([d[(e, "gate_proj")], d[(e, "up_proj")]]) for e in range(n)])
        got[f"{prefix}.down_proj"] = torch.stack([d[(e, "down_proj")] for e in range(n)])
    return got


def close(a, b, rel):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


# --- collectives (no model) ----------------------------------------------------------------------------------------

def rank_collectives(rank, world, inp):
    import torch.distributed as dist

    g = torch.Generator().manual_seed(0)
    experts = 5  # 2, 2, 1 rows on 3 ranks: the padded path
    w = torch.randn(experts, 3, 4, generator=g)
    ups = [torch.randn(experts, 3, 4, generator=g) for _ in range(world)]
    empty = inp["empty"]
    c = shard.Ctx("cpu", experts)
    out = {"range": (c.lo, c.hi)}
    # gather forward = the full tensor; backward = this rank's rows of the sum over ranks' upstream gradients
    for dtype in (torch.float32, torch.bfloat16):
        p = torch.nn.Parameter(w[c.lo:c.hi].to(dtype).clone())
        full = shard._Gather.apply(p, c)
        assert torch.equal(full, w.to(dtype))
        up = torch.zeros_like(ups[rank]) if empty[rank] else ups[rank]
        (full.float() * up).sum().backward()
        # each rank's gradient arrives in the param dtype, is summed in fp32 and cast back
        want = sum(ups[r].to(dtype).float() for r in range(world) if not empty[r])[c.lo:c.hi].to(dtype)
        assert p.grad.dtype == dtype and torch.allclose(p.grad.float(), want.float(), atol=1e-6, rtol=1e-6)
    # dense sync: summed over ranks; a grad only rank 0 has; a grad no rank has stays None
    a, b, z = (torch.nn.Parameter(torch.zeros(7)) for _ in range(3))
    a.grad = torch.full((7,), float(rank + 1))
    if rank == 0:
        b.grad = torch.ones(7)
    shard.sync_dense([a, b, z], c, bucket=5)
    assert torch.equal(a.grad, torch.full((7,), float(sum(range(1, world + 1)))))
    assert torch.equal(b.grad, torch.ones(7)) and z.grad is None
    # global norm: shards' squares summed over ranks, replicated counted once; clip_grad_norm_'s coefficient
    full_g = torch.randn(experts, 6, generator=g)
    rep_g = torch.randn(9, generator=g)
    s = torch.nn.Parameter(torch.zeros(c.hi - c.lo, 6))
    s.grad = full_g[c.lo:c.hi].clone()
    r = torch.nn.Parameter(torch.zeros(9))
    r.grad = rep_g.clone()
    total = float(shard.clip([r], [s], 1.0, c))
    ref_s, ref_r = torch.nn.Parameter(torch.zeros(experts, 6)), torch.nn.Parameter(torch.zeros(9))
    ref_s.grad, ref_r.grad = full_g.clone(), rep_g.clone()
    ref = float(torch.nn.utils.clip_grad_norm_([ref_r, ref_s], 1.0))
    assert close(total, ref, 1e-6) and torch.allclose(s.grad, ref_s.grad[c.lo:c.hi], atol=1e-7)
    assert torch.allclose(r.grad, ref_r.grad, atol=1e-7)
    # pass alignment: per round the largest prefix count over ranks; a missing example counts 0
    out["rounds"] = c.rounds(inp["counts"])
    # replicas: equal -> nothing; one rank's copy one ulp off -> that index on every rank
    ps = [torch.ones(4, dtype=torch.bfloat16), torch.ones(3)]
    out["same"] = shard.check_replicas(ps, c)
    if rank == 1:
        ps[1] = ps[1].clone()
        ps[1][0] = torch.nextafter(ps[1][0], torch.tensor(2.0))
    out["differ"] = shard.check_replicas(ps, c)
    dist.barrier()
    return out


def test_collectives_match_single_process():
    counts = [[3, 0, 5], [1, 7], []]
    got = run_ranks(rank_collectives, world=3, inputs=[{"empty": [False, False, True], "counts": k} for k in counts],
                    timeout=60)
    assert [g["range"] for g in got] == [(0, 2), (2, 4), (4, 5)]
    assert all(g["rounds"] == [3, 7, 5] for g in got)
    assert all(g["same"] == [] and g["differ"] == [1] for g in got)


def test_collectives_four_ranks_one_empty():
    counts = [[3, 0, 5], [1, 7], [], [2]]
    got = run_ranks(rank_collectives, world=4, inputs=[{"empty": [False, False, True, False], "counts": k} for k in counts],
                    timeout=60)
    assert [g["range"] for g in got] == [(0, 2), (2, 3), (3, 4), (4, 5)]
    assert all(g["rounds"] == [3, 7, 5] for g in got)


def test_split_and_lpt():
    assert shard.split(256, 2) == [(0, 128), (128, 256)] and shard.split(4, 3) == [(0, 2), (2, 3), (3, 4)]
    # largest first to the least-loaded rank: 5->r0, 4->r1, 3->r1 (4<5), 3->r0 (5<7), 1->r1 (7<8)
    assert shard.lpt([5, 1, 4, 3, 3], 2) == [[0, 4], [2, 3, 1]]
    assert shard.lpt([2, 2], 3) == [[0], [1], []]  # ties go to the lower rank; a rank may get nothing


# --- the trainer ---------------------------------------------------------------------------------------------------

def compare(a, b, *, rel, wtol, evals=True):
    """Two runs' step rows (loss, KL, grad norm), held-out reads and final weights."""
    sa, sb = rows(a, "step"), rows(b, "step")
    assert [r["step"] for r in sa] == [r["step"] for r in sb] and sa
    for x, y in zip(sa, sb):
        assert x["reply_tokens"] == y["reply_tokens"] and x["examples"] == y["examples"]
        assert x["ctx_tokens"] == y["ctx_tokens"] and x["window_tokens"] == y["window_tokens"]
        for k in ("loss", "distill_kl", "grad_norm", "mtp_kl"):
            assert close(x[k], y[k], rel), (x["step"], k, x[k], y[k])
    if evals:
        ea, eb = rows(a, "eval"), rows(b, "eval")
        assert [(e["step"], e["set"], e["examples"], e["heldout_tokens"]) for e in ea] == \
            [(e["step"], e["set"], e["examples"], e["heldout_tokens"]) for e in eb]
        for x, y in zip(ea, eb):
            assert close(x["heldout_kl"], y["heldout_kl"], rel) and close(x["mtp_kl"], y["mtp_kl"], rel)
    last = max(r["step"] for r in sa)
    wa, wb = weights(Path(a) / f"ckpt-{last}"), weights(Path(b) / f"ckpt-{last}")
    assert wa.keys() == wb.keys()
    worst = max(float((wa[k].float() - wb[k].float()).abs().max()) for k in wa)
    dev = max(abs(x[k] - y[k]) / max(abs(x[k]), 1e-12) for x, y in zip(sa, sb) for k in ("loss", "distill_kl", "grad_norm"))
    print(f"COMPARE {Path(a).name} vs {Path(b).name}: worst weight diff {worst:.3g}, worst step rel diff {dev:.3g}")
    assert worst <= wtol, worst
    return worst


def test_fp32_n1_n2_n3_equal(setup, tmp_path):
    """Same global batches at 1, 2 and 3 ranks (KL + fake quant, plain SGD): losses, grad norms, held-out reads and
    the weights after 3 steps within 1e-5."""
    outs = {}
    for n in (1, 2, 3):
        outs[n] = tmp_path / f"n{n}"
        run(n, cfg(setup, outs[n]), "sgd", fp32=True)
    for n in (2, 3):
        compare(outs[1], outs[n], rel=1e-5, wtol=1e-5)
    s3 = rows(outs[3], "step")
    assert all(r["ranks"]["examples"] and sum(r["ranks"]["examples"]) == r["examples"] for r in s3)
    start = rows(outs[3], "start")[0]
    assert start["world"] == 3 and start["expert_ranges"] == [[0, 2], [2, 3], [3, 4]]
    for n in (2, 3):  # no kept layer: one layer's gathered projections at a time
        assert rows(outs[n], "step")[-1]["ranks"]["comm"]["gather_live_max"] == 2
    assert start["params"] == rows(outs[1], "start")[0]["params"]


def test_fp32_n1_n4_equal(setup, tmp_path):
    """N=4, the d16 run's shape: same global batches as one GPU, every rank one expert."""
    one, four = tmp_path / "one", tmp_path / "four"
    run(1, cfg(setup, one), "sgd", fp32=True)
    run(4, cfg(setup, four), "sgd", fp32=True)
    compare(one, four, rel=1e-5, wtol=1e-5)
    start = rows(four, "start")[0]
    assert start["world"] == 4 and start["expert_ranges"] == [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert all(sum(r["ranks"]["examples"]) == r["examples"] for r in rows(four, "step"))


def test_two_empty_ranks_n4(setup, tmp_path):
    """4 ranks, 2 examples a step: two ranks run only alignment passes; equal to one GPU."""
    one, four = tmp_path / "one", tmp_path / "four"
    c = dict(data=str(setup / "uniform"), batch_reply_tokens=120, total_steps=2, ckpt_steps=2, eval_steps=2,
             eval_examples=2)
    run(1, cfg(setup, one, **c), "sgd", fp32=True)
    run(4, cfg(setup, four, **c), "sgd", fp32=True)
    assert all(r["examples"] == 2 and sorted(r["ranks"]["examples"]) == [0, 0, 1, 1] for r in rows(four, "step"))
    compare(one, four, rel=1e-5, wtol=1e-5)


def test_checkpoint_resume_n4(setup, tmp_path):
    """Saved at N=4, resumed at N=4: the next step equals the uninterrupted run's (FP32, AdamW)."""
    base = dict(ckpt_steps=2)
    ref, out = tmp_path / "ref", tmp_path / "resumed"
    run(4, cfg(setup, ref, **base), "adamw", fp32=True)
    assert json.loads((ref / "ckpt-2/shard.json").read_text())["world"] == 4
    out.mkdir()
    shutil.copytree(ref / "ckpt-2", out / "ckpt-2")
    run(4, cfg(setup, out, **base), "adamw", fp32=True)
    assert [r["step"] for r in rows(out, "step")] == [3]
    sa, sb = rows(ref, "step")[-1], rows(out, "step")[0]
    assert close(sa["distill_kl"], sb["distill_kl"], 1e-5) and close(sa["grad_norm"], sb["grad_norm"], 1e-5)
    wa, wb = weights(ref / "ckpt-3"), weights(out / "ckpt-3")
    assert max(float((wa[k] - wb[k]).abs().max()) for k in wa) <= 1e-5


def test_live_batch_at_start_n2_equals_n1(setup, tmp_path):
    """The knob found at start (a restart): same step on every rank, same batches as one GPU."""
    outs, c = {}, dict(live_batch=True, total_steps=8, ckpt_steps=8, eval_steps=8)
    for n in (1, 2):
        outs[n] = tmp_path / f"n{n}"
        run(n, cfg(setup, outs[n], **c), "sgd", fp32=True, knobs=[json.dumps({"batch_reply_tokens": 400})])
    compare(outs[1], outs[2], rel=1e-5, wtol=1e-5)
    got = rows(outs[2], "step")
    assert rows(outs[2], "batch_knob")[0]["first_step"] == 5
    assert max(r["examples"] for r in got[:4]) < min(r["examples"] for r in got[4:])


@pytest.mark.parametrize("world", [2, 3])
def test_live_batch_mid_run_ranks_stay_aligned(setup, tmp_path, world):
    """A knob landing mid-run on rank 0's thread: every rank changes at the same step (a mismatch hangs the collectives
    or fails the batch sums)."""
    out = tmp_path / "out"
    run(world, cfg(setup, out, live_batch=True, total_steps=10, ckpt_steps=10, eval_steps=10), "sgd", fp32=True,
        knobs=["", "", "", json.dumps({"batch_reply_tokens": 400})])
    knob = rows(out, "batch_knob")[0]
    got = {r["step"]: r for r in rows(out, "step")}
    assert knob["first_step"] == knob["step"] + 5
    assert all(sum(r["ranks"]["examples"]) == r["examples"] for r in got.values())
    assert got[knob["first_step"]]["reply_tokens"] >= 400
    assert all(r["reply_tokens"] < 400 for s, r in got.items() if s < knob["first_step"])


def test_bf16_n1_n2_n3_close(setup, tmp_path):
    """BF16 with torchao's 8-bit AdamW (stochastic rounding, per-rank streams on the shards): BF16 tolerance."""
    outs = {}
    for n in (1, 2, 3):
        outs[n] = tmp_path / f"n{n}"
        run(n, cfg(setup, outs[n], lr=1e-4), "ao8")
    for n in (2, 3):
        compare(outs[1], outs[n], rel=2e-2, wtol=0.02)


def test_empty_rank_two_example_steps(setup, tmp_path):
    """3 ranks, every step 2 examples (one rank runs only alignment passes): equal to one GPU."""
    one, three = tmp_path / "one", tmp_path / "three"
    c = dict(data=str(setup / "uniform"), batch_reply_tokens=120, total_steps=2, ckpt_steps=2, eval_steps=2,
             eval_examples=2)
    run(1, cfg(setup, one, **c), "sgd", fp32=True)
    run(3, cfg(setup, three, **c), "sgd", fp32=True)
    assert all(r["examples"] == 2 and sorted(r["ranks"]["examples"]) == [0, 1, 1] for r in rows(three, "step"))
    compare(one, three, rel=1e-5, wtol=1e-5)


def test_empty_window_example_ranks_stay_aligned(setup, tmp_path):
    """A training example with no reply row in its window (index still lists its replies) on one rank of 3: a counted
    zero backward, every collective of the pass still met, equal to one GPU."""
    import shutil

    data_ = tmp_path / "data"
    shutil.copytree(setup / "data", data_)
    f = data_ / "train/e0.pt"
    ex = torch.load(f)
    torch.save({**ex, "labels": torch.full_like(ex["labels"], -100)}, f)
    (data_ / "teacher/train/e0.pt").write_bytes((data_ / "teacher/train/e1.pt").read_bytes())
    one, three = tmp_path / "one", tmp_path / "three"
    run(1, cfg(setup, one, data=str(data_)), "sgd", fp32=True)
    run(3, cfg(setup, three, data=str(data_)), "sgd", fp32=True)
    assert rows(one, "step")[-1]["empty_examples"] == rows(three, "step")[-1]["empty_examples"] > 0
    compare(one, three, rel=1e-5, wtol=1e-5)


def test_checkpoint_resize(setup, tmp_path):
    """A checkpoint saved at N=2 resumes at N=1 and at N=3, and a 1-GPU checkpoint at N=2: the next step's loss,
    grad norm and weights equal the uninterrupted run's (FP32, AdamW: the states re-sliced)."""
    base = dict(ckpt_steps=2)
    ref = tmp_path / "ref2"
    run(2, cfg(setup, ref, **base), "adamw", fp32=True)
    man = json.loads((ref / "ckpt-2/shard.json").read_text())
    assert man["world"] == 2 and man["ranges"] == [[0, 2], [2, 4]]
    assert not list(ref.glob("ckpt-2/done-r*")) and (ref / "ckpt-2/optim-r1.pt").is_file()
    for n in (1, 3):
        out = tmp_path / f"resume{n}"
        out.mkdir()
        shutil.copytree(ref / "ckpt-2", out / "ckpt-2")
        run(n, cfg(setup, out, **base), "adamw", fp32=True)
        assert [r["step"] for r in rows(out, "step")] == [3]
        sa, sb = rows(ref, "step")[-1], rows(out, "step")[0]
        assert close(sa["distill_kl"], sb["distill_kl"], 1e-5) and close(sa["grad_norm"], sb["grad_norm"], 1e-5)
        wa, wb = weights(ref / "ckpt-3"), weights(out / "ckpt-3")
        assert max(float((wa[k] - wb[k]).abs().max()) for k in wa) <= 1e-5
    one = tmp_path / "one"  # a one-GPU run's checkpoint, resumed at N=2 by init_ckpt
    run(1, cfg(setup, one, **base), "adamw", fp32=True)
    two = tmp_path / "two"
    run(2, cfg(setup, two, init_ckpt=str(one / "ckpt-2"), **base), "adamw", fp32=True)
    sa, sb = rows(one, "step")[-1], rows(two, "step")[0]
    assert sb["step"] == 3 and close(sa["distill_kl"], sb["distill_kl"], 1e-5) and close(sa["grad_norm"], sb["grad_norm"], 1e-5)
    wa, wb = weights(one / "ckpt-3"), weights(two / "ckpt-3")
    assert max(float((wa[k] - wb[k]).abs().max()) for k in wa) <= 1e-5


def test_optimizer_states_reslice(tmp_path):
    """torchao 8-bit states: N=1 -> a rank's rows keeps codes and block scales bit-equal (an expert = whole blocks);
    a sharded save (2 ranks) -> 1 and 3 ranks bit-equal; an expert size that splits a block is requantized (close)."""
    pytest.importorskip("torchao")

    def made(shape, steps=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        q = torch.nn.Parameter(torch.randn(300, 64, generator=g).to(torch.bfloat16))
        p = torch.nn.Parameter(torch.randn(*shape, generator=g).to(torch.bfloat16))
        opt = train.adamw8bit([q, p], 1e-3)
        for _ in range(steps):
            q.grad = torch.randn(q.shape, generator=g).to(torch.bfloat16)
            p.grad = torch.randn(p.shape, generator=g).to(torch.bfloat16)
            opt.step()
        return q, p, opt

    def save_n1(opt, folder):
        folder.mkdir(parents=True)
        items, skel = ckpt.optimizer_items(opt)
        ckpt.write_safetensors(folder / "optim.safetensors", items)
        torch.save(skel, folder / "optim.pt")

    def load(folder, q, p, lo, hi, n1=("q", "p")):
        q2 = torch.nn.Parameter(q.detach().clone())
        p2 = torch.nn.Parameter(p.detach()[lo:hi].clone())
        opt2 = train.adamw8bit([{"params": [q2]}, {"params": [p2]}], 1e-3)
        shard.load_optimizer(opt2, folder, {id(q2): "q", id(p2): "p"}, {"p": (lo, hi)}, n1_names=list(n1))
        return q2, p2, opt2

    q, p, opt = made((6, 16, 256))  # 4096 per expert: 16 blocks; every shard >= 4096 stays 8-bit (torchao's rule)
    a = tmp_path / "n1"
    save_n1(opt, a)
    full = opt.state[p]
    for lo, hi in ((0, 6), (2, 5), (5, 6)):
        _, p2, opt2 = load(a, q, p, lo, hi)
        for k in ("exp_avg", "exp_avg_sq"):
            got = opt2.state[p2][k]
            assert type(got).__name__ == "OptimState8bit", (lo, hi, k, type(got))
            assert torch.equal(got.codes, full[k].codes[lo:hi]), (lo, hi, k)
            assert torch.equal(got.scale, full[k].scale[lo * 16:hi * 16])
        assert torch.equal(opt2.state[p2]["step"], full["step"])
    # a sharded save from two ranks (rows 0-3, 3-6) back to 1 and to 3 ranks
    b = tmp_path / "n2"
    b.mkdir()
    for r, (lo, hi) in enumerate(((0, 3), (3, 6))):
        q2, p2, opt2 = load(a, q, p, lo, hi)
        keep = {id(p2)} | ({id(q2)} if r == 0 else set())
        items, skel = shard.optim_items(opt2, {id(q2): "q", id(p2): "p"}, keep)
        ckpt.write_safetensors(b / f"optim-r{r}.safetensors", items)
        torch.save(skel, b / f"optim-r{r}.pt")
    (b / "shard.json").write_text(json.dumps({"world": 2, "experts": 6, "ranges": [[0, 3], [3, 6]], "sharded": ["p"]}))
    for lo, hi in ((0, 6), (0, 2), (2, 4), (4, 6), (1, 5)):
        q2, p2, opt2 = load(b, q, p, lo, hi)
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(opt2.state[p2][k].codes, full[k].codes[lo:hi])
            assert torch.equal(opt2.state[p2][k].scale, full[k].scale[lo * 16:hi * 16])
            assert torch.equal(opt2.state[q2][k].codes, opt.state[q][k].codes)
    # a shard under 4096 elements gets torchao's plain state: the 8-bit rows dequantized, in the param dtype
    q, p, opt = made((6, 16, 32), seed=2)  # 512 per expert
    d = tmp_path / "small"
    save_n1(opt, d)
    _, p2, opt2 = load(d, q, p, 2, 5)  # 1536 < 4096
    for k in ("exp_avg", "exp_avg_sq"):
        got = opt2.state[p2][k]
        assert type(got) is torch.Tensor and torch.equal(got, opt.state[p][k].dequantize()[2:5].to(got.dtype))
    # 24 x 24 x 16 = 384 per expert (a block and a half): rows 1-13 start mid-block, so they are dequantized, sliced
    # and requantized (8-bit: 12 x 384 = 18 blocks)
    q, p, opt = made((24, 24, 16), seed=1)
    c = tmp_path / "odd"
    save_n1(opt, c)
    _, p2, opt2 = load(c, q, p, 1, 13)
    for k in ("exp_avg", "exp_avg_sq"):
        want = opt.state[p][k].dequantize().float()[1:13]
        got = opt2.state[p2][k]
        assert type(got).__name__ == "OptimState8bit"
        got = got.dequantize().float()
        assert torch.allclose(got, want, rtol=0.05, atol=2e-2 * float(want.abs().max()))


def sgd_linger(params, lr):  # sgd, plus a non-daemon thread that never ends on every spawned rank
    import threading

    if dist_rank() > 0:
        threading.Thread(target=threading.Event().wait, name="linger").start()
    return sgd(params, lr)


def dist_rank():
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def test_launch_exits_with_lingering_thread(setup, tmp_path, monkeypatch):
    """A finished rank exits even if a non-daemon thread lingers (probe k256-lr-3e6 sat 12+ min after "done" with both
    ranks alive); launch returns and every rank exits 0."""
    import torch.distributed as dist

    monkeypatch.setitem(shard.EXIT, "join_s", 90.0)
    began = time.time()
    try:
        shard.launch(train.Config(**cfg(setup, tmp_path / "o", total_steps=2, ckpt_steps=2)), 2,
                     backend="gloo", make_opt=sgd_linger)
    finally:
        qat.QAT["on"] = False
        qat.SCALES.clear()
        shard.DD["ctx"] = None
        if dist.is_initialized():
            dist.destroy_process_group()
    assert time.time() - began < 85, "rank 1 was killed by the join bound, not exited by itself"
    assert rows(tmp_path / "o", "done")


def test_launch_shared_table(setup, tmp_path):
    """shard.launch (gloo, 2 processes): the n-gram table read once into shared memory and attached on every rank;
    the run equals the per-rank-load run."""
    import torch.distributed as dist

    a, b = tmp_path / "launch", tmp_path / "ranks"
    try:  # rank 0 in this process, rank 1 spawned
        shard.launch(train.Config(**cfg(setup, a, total_steps=2, ckpt_steps=2)), 2, backend="gloo", make_opt=sgd)
    finally:
        qat.QAT["on"] = False
        qat.SCALES.clear()
        shard.DD["ctx"] = None
        if dist.is_initialized():
            dist.destroy_process_group()
    run(2, cfg(setup, b, total_steps=2, ckpt_steps=2), "sgd")
    assert rows(a, "start")[0]["load"]["shared_tables"] == 1
    compare(a, b, rel=1e-3, wtol=0.02)  # BF16; the two runs differ only in CPU thread counts


# --- served grids (int3served experts, INT8 dense, kv4) at N > 1 ----------------------------------------------------

@pytest.mark.parametrize("grid", ["int4", "int3served"])
def test_shard_grid_equals_gathered_grid(monkeypatch, grid):
    """The QAT grid on a rank's expert rows equals those rows of the grid on the whole tensor (every scale is per
    expert: int3served's scale_2 from that expert's own weights), so quantize-then-gather = gather-then-quantize; the
    all-zero expert count over the shards is the whole tensor's."""
    monkeypatch.setitem(qat.QAT, "grid", grid)
    monkeypatch.setitem(qat.QAT, "fused", None)
    monkeypatch.setitem(qat.QAT, "chunk", 2)
    monkeypatch.setitem(qat.QAT, "zero_experts", 0)
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(7, 32, 128, generator=g) * 0.02 * torch.rand(7, 32, 1, generator=g).add(0.2)).to(torch.bfloat16)
    w[5] = 0  # an all-zero expert (int3served: scale_2 1, grid 0)
    w[2, 3, 40] = 0.5  # an outlier sets expert 2's scale_2 alone
    full = qat.grid(w, qat.scales_of(w))
    zero_full = qat.QAT["zero_experts"]
    for world in (2, 3, 4):
        qat.QAT["zero_experts"] = 0
        parts = [qat.grid(w[a:b], qat.scales_of(w[a:b])) for a, b in shard.split(7, world)]
        assert torch.equal(torch.cat(parts), full), (grid, world)
        assert qat.QAT["zero_experts"] == zero_full
    assert zero_full == (1 if grid == "int3served" else 0)


SERVED = dict(fake_quant=True, qat_grid="int3served", dense_int8_train=True, kv_train="kv4")


def test_served_grids_n1_n2_n3(setup, tmp_path):
    """qat_grid int3served + dense_int8_train + kv_train kv4 together (selected attention): FP32 SGD at 1, 2 and 3
    ranks on the asymmetric batches within 1e-5, and an empty rank (2-example steps on 3 ranks); BF16 8-bit AdamW at
    1 and 2 ranks within BF16 tolerance. Each grid in use (the start event's INT8 count)."""
    outs = {}
    for n in (1, 2, 3):
        outs[n] = tmp_path / f"n{n}"
        run(n, cfg(setup, outs[n], **SERVED), "sgd", fp32=True, selected=True)
    for n in (2, 3):
        compare(outs[1], outs[n], rel=1e-5, wtol=1e-5)
    for n in (1, 2, 3):
        start = rows(outs[n], "start")[0]
        assert start["load"]["dense_int8"] == 3 * 2 + 4 * 1 + 4 + 1  # trunk, the MTP's q / k / v / o, lm_head
        assert start["config"]["qat_grid"] == "int3served" and start["config"]["kv_train"] == "kv4"
    u = dict(data=str(setup / "uniform"), batch_reply_tokens=120, total_steps=2, ckpt_steps=2, eval_steps=2,
             eval_examples=2, **SERVED)
    one, three = tmp_path / "u1", tmp_path / "u3"
    run(1, cfg(setup, one, **u), "sgd", fp32=True, selected=True)
    run(3, cfg(setup, three, **u), "sgd", fp32=True, selected=True)
    assert all(sorted(r["ranks"]["examples"]) == [0, 1, 1] for r in rows(three, "step"))
    compare(one, three, rel=1e-5, wtol=1e-5)
    b1, b2 = tmp_path / "b1", tmp_path / "b2"
    run(1, cfg(setup, b1, lr=1e-4, **SERVED), "ao8", selected=True)
    run(2, cfg(setup, b2, lr=1e-4, **SERVED), "ao8", selected=True)
    compare(b1, b2, rel=2e-2, wtol=0.02)


# --- warm-up OOM at N > 1 -------------------------------------------------------------------------------------------

def test_warm_oom_steps_down_together(setup, tmp_path):
    """A CUDA OOM on one rank inside the warm-up (before a forward all-gather, or before a backward reduce-scatter):
    every rank leaves the pass at the same collective, all step kept layers down by one, and the run equals one GPU
    at that kept count; with no kept layers left, every rank stops (memory_stop), none hangs."""
    ref = tmp_path / "ref"
    run(1, cfg(setup, ref, keep_layers=1), "sgd", fp32=True)
    for world, oom in ((3, {"rank": 1, "kind": "gather", "at": 3}), (2, {"rank": 0, "kind": "scatter", "at": 2})):
        out = tmp_path / f"oom{world}"
        run(world, cfg(setup, out, keep_layers=2), "sgd", fp32=True, oom=oom)
        downs = rows(out, "keep_layers_down")
        assert [d["keep_layers"] for d in downs] == [1], downs
        assert "out of memory" in downs[0]["why"]
        assert rows(out, "boot")[0]["warm"]["kept_layers"] == 1
        compare(ref, out, rel=1e-5, wtol=1e-5)
        if world == 2:  # its warm-up reached the backward with 2 kept layers: both layers' projections held
            assert rows(out, "step")[-1]["ranks"]["comm"]["gather_live_max"] == 4
    stop = tmp_path / "stop"
    got = run(2, cfg(setup, stop, keep_layers=0), "sgd", fp32=True, catch=True,
              oom={"rank": 1, "kind": "gather", "at": 2})
    assert all("out of memory" in g for g in got), got
    assert rows(stop, "memory_stop")


# --- memory: chunked temporaries, the chunked reduce-scatter, BF16 reduce at N=2, paged states -----------------------

def test_memory_paths_exact(monkeypatch):
    """The chunked forms equal the whole-tensor ones bit for bit: int3served scale_2 (per expert), the INT8 row round
    trip (per row), kv_fake (per token / kv2kc group)."""
    import attention

    g = torch.Generator().manual_seed(5)
    w = (torch.randn(11, 16, 64, generator=g) * torch.rand(11, 1, 1, generator=g).add(0.01)).to(torch.bfloat16)
    w[4] = 0
    monkeypatch.setitem(qat.QAT, "chunk", 3)
    monkeypatch.setitem(qat.QAT, "zero_experts", 0)
    whole = w.float().abs().amax(dim=(-2, -1)) * (1.0 / (abs(qat.LEVELS3[0]) * qat.FP8_MAX))
    whole = torch.where(whole == 0, torch.ones_like(whole), whole)
    assert torch.equal(qat.scale2_of3(w), whole)
    d = (torch.randn(1000, 96, generator=g) * torch.rand(1000, 1, generator=g)).to(torch.bfloat16)
    assert torch.equal(qat.int8_rows(d, chunk_rows=7), qat.int8_rows(d, chunk_rows=1 << 30))
    k = torch.randn(1, 2, 1000, 64, generator=g).to(torch.bfloat16)
    v = torch.randn(1, 2, 1000, 64, generator=g).to(torch.bfloat16)
    for mode in ("kv4", "kv2", "kv2kc"):
        monkeypatch.setattr(attention, "KV_CHUNK", 1 << 20)
        a = attention.kv_fake(k, v, mode)
        monkeypatch.setattr(attention, "KV_CHUNK", 256)  # pieces of 256 tokens, a last one of 232
        b = attention.kv_fake(k, v, mode)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), mode


def rank_scatter(rank, world, inp):
    """Every rank's gradients are known to all (seeded): the chunked reduce-scatter equals the whole one and the fp32
    sum cast once; at 2 ranks the BF16 reduce equals the fp32 one (values spread over 2^-40 .. 2^40)."""
    experts = 5
    grads = []
    for r in range(world):
        g = torch.Generator().manual_seed(100 + r)
        x = torch.randn(experts, 6, 8, generator=g) * torch.pow(2.0, torch.randint(-40, 41, (experts, 6, 1), generator=g))
        grads.append(x.to(torch.bfloat16))
    want_all = sum(x.float() for x in grads).to(torch.bfloat16)
    c = shard.Ctx("cpu", experts)
    got = {}
    for name, nbytes in (("whole", 1 << 30), ("rows", 1)):
        shard.RS["bytes"] = nbytes
        got[name] = shard.reduce_scatter(grads[rank], c, torch.bfloat16)
    assert torch.equal(got["whole"], got["rows"])
    if world == 2:  # one addition: any order gives the fp32 sum (3 ranks: gloo's ring order, not Python's)
        assert torch.equal(got["whole"], want_all[c.lo:c.hi])
    else:
        assert torch.allclose(got["whole"].float(), want_all[c.lo:c.hi].float(), rtol=1e-2, atol=0)
    assert c.stats["scatters"] == 1 + c.emax
    if world == 2:
        b = shard.Ctx("cpu", experts, "bf16")
        for nbytes in (1 << 30, 1):
            shard.RS["bytes"] = nbytes
            assert torch.equal(shard.reduce_scatter(grads[rank], b, torch.bfloat16), got["whole"])
    else:
        with pytest.raises(ValueError, match="only on 2 ranks"):
            shard.Ctx("cpu", experts, "bf16")
    return True


@pytest.mark.parametrize("world", [2, 3, 4])
def test_reduce_scatter_chunked_and_bf16(world):
    assert all(run_ranks(rank_scatter, world=world, inputs=[{}] * world, timeout=60))


def test_paged_states_n2(setup, tmp_path):
    """paged_states (the main run's optimizer layout) at 2 ranks, BF16 8-bit AdamW, 1 kept layer: within BF16 tolerance
    of one GPU; a resume from the sharded step-2 checkpoint repeats step 3 exactly; the gathered projections never
    exceed one layer's (the kept layer is the last)."""
    pytest.importorskip("torchao")  # paged.Pager steps with torchao's single_param_adam; the CI image has no torchao
    c = dict(lr=1e-4, paged_states=True, keep_layers=1, ckpt_steps=2)
    one, two = tmp_path / "p1", tmp_path / "p2"
    run(1, cfg(setup, one, **c), "ao8")
    run(2, cfg(setup, two, **c), "ao8")
    compare(one, two, rel=2e-2, wtol=0.02)
    assert rows(two, "start")[0]["load"].get("paged_check", {}).get("ok", True)
    assert rows(two, "step")[-1]["ranks"]["comm"]["gather_live_max"] == 2
    again = tmp_path / "again"
    again.mkdir()
    shutil.copytree(two / "ckpt-2", again / "ckpt-2")
    run(2, cfg(setup, again, **c), "ao8")
    sa, sb = rows(two, "step")[-1], rows(again, "step")[0]
    assert sb["step"] == 3 and all(sa[k] == sb[k] for k in ("loss", "distill_kl", "grad_norm"))
    wa, wb = weights(two / "ckpt-3"), weights(again / "ckpt-3")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
