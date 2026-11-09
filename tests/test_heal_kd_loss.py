"""heal_kd.bucket_kl: forward and reverse KL over the teacher's top-k + tail bucket (OPD loss, 18 Oct)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)
import heal_kd  # noqa: E402


def test_bucket_kl_matches_exact_kl_when_k_covers_the_vocabulary():
    g = torch.Generator().manual_seed(0)
    s = torch.log_softmax(torch.randn(5, 12, generator=g), -1)
    t = torch.log_softmax(torch.randn(5, 12, generator=g), -1)
    top, idx = t.topk(11, -1)  # all but one token; the last one is the tail bucket
    tail = torch.log1p(-top.exp().sum(-1))
    forward, reverse = heal_kd.bucket_kl(s.gather(-1, idx), top, tail)
    exact_f = (t.exp() * (t - s)).sum(-1)
    exact_r = (s.exp() * (s - t)).sum(-1)
    assert torch.allclose(forward, exact_f, atol=1e-4) and torch.allclose(reverse, exact_r, atol=1e-4)


def test_bucket_kl_is_zero_for_equal_distributions_and_reverse_differs_from_forward():
    t = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.0, -1.0, -3.0]]), -1)
    top, idx = t.topk(3, -1)
    tail = torch.log1p(-top.exp().sum(-1))
    f, r = heal_kd.bucket_kl(t.gather(-1, idx), top, tail)
    assert abs(float(f)) < 1e-5 and abs(float(r)) < 1e-5
    s = torch.log_softmax(torch.tensor([[0.0, 2.0, 1.0, -1.0, -3.0]]), -1)
    f, r = heal_kd.bucket_kl(s.gather(-1, idx), top, tail)
    assert float(f) > 0 and float(r) > 0 and abs(float(f) - float(r)) > 1e-3


def test_kd_loss_reply_sum_mix_and_gradients():
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(6, 8, generator=g, requires_grad=True)
    weight = torch.randn(12, 8, generator=g)
    t = torch.log_softmax(torch.randn(6, 12, generator=g), -1)
    top, idx = t.topk(4, -1)
    tail = torch.log1p(-top.exp().sum(-1))
    mask = torch.ones(6)
    reply = torch.tensor([0.0, 1, 1, 0, 1, 0])
    s = torch.log_softmax(hidden.detach() @ weight.T, -1)
    f, r = heal_kd.bucket_kl(s.gather(-1, idx), top, tail)
    loss, kd, lm, kr = heal_kd.kd_loss(hidden, weight, top, idx.int(), tail, mask, reply=reply, chunk=4,
                                       mix={"forward": 0.1, "reverse": 1.0})
    assert abs(kd - float(f.sum())) < 1e-4 and abs(kr - float((f * reply).sum())) < 1e-4 and lm == 0.0
    assert abs(float(loss) - float((0.1 * f + r).sum())) < 1e-4
    loss.backward()
    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0


def test_cosine_lr_warms_up_then_decays_to_the_floor():
    lrs = [heal_kd.cosine_lr(s, 40, 2e-5, warmup=0.05, floor=0.1) for s in range(41)]
    assert abs(lrs[0] - 2e-5 / 3) < 1e-12 and abs(lrs[1] - 4e-5 / 3) < 1e-12 and abs(lrs[2] - 2e-5) < 1e-12
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:]))  # monotone after warm-up
    assert abs(lrs[40] - 2e-6) < 1e-12 and abs(lrs[21] - 1.1e-5) < 1e-6  # floor; mid-point (0.1 + 0.9 / 2)


def test_every_schedule_warms_up_and_wsd_decays_within_total():
    # a 1-step warm-up trained step 0 at the peak (cut1-v2, c1opd-v1): never again, for any total or share
    for total in (3, 6, 18, 21, 26, 200):
        for fn in (heal_kd.wsd_lr, heal_kd.cosine_lr):
            lrs = [fn(s, total, 1e-5, warmup=0.05) for s in range(total)]
            assert lrs[0] < 0.5 * 1e-5 and lrs[1] < 1e-5 and max(lrs) <= 1e-5 + 1e-15
    wsd = [heal_kd.wsd_lr(s, 21, 1e-5, warmup=0.1) for s in range(21)]
    assert wsd[20] <= 0.25 * 1e-5 + 1e-15 and wsd[18] < 1e-5 and wsd[17] == 1e-5  # the decay runs inside 21 steps


def test_own_positions_read_the_index_else_the_files(tmp_path):
    import json

    folder = tmp_path / "teacher"
    folder.mkdir()
    for i, n in enumerate((300, 500)):
        torch.save({"ids": torch.zeros(1000, dtype=torch.int32), "reply": torch.arange(1000) < n}, folder / f"s{i}.pt")
    files = sorted(folder.glob("*.pt"))
    assert heal_kd.own_positions(files) == [300, 500] and heal_kd.own_positions(files, "all") == [1000, 1000]
    (folder / "index.json").write_text(json.dumps({"rows": [{"file": "s1.pt", "reply_tokens": 5000, "tokens": 9000}]}))
    assert heal_kd.own_positions(files) == [300, 5000]  # s0 has no row: counted from its file


def test_simulated_steps_follow_the_loop_rules():
    # slot overshoot: 20 files of 300 in slots of 2 (600) over a 1000 batch fill 5 steps, not 6000 / 1000 = 6
    assert heal_kd.simulate_steps([300] * 20, 0, 2, 1000, 0.2) == 5
    assert heal_kd.simulate_steps([300] * 19, 0, 2, 1000, 0.2) == 4  # the partial last batch is dropped ...
    assert heal_kd.simulate_steps([300] * 19, 0, 2, 1000, 0.2, flush_last=True) == 5  # ... unless flushed
    # general slots (2 x 255) join while general trails 0.25 x own: the tiny train-tiny data, seed 0 order
    assert heal_kd.simulate_steps([26, 43, 130, 174, 59, 94], 255, 2, 100, 0.2) == 3
    # a resume from that run's ckpt-2: counters restored, the two files left fill the one step left
    assert heal_kd.simulate_steps([59, 94], 255, 2, 100, 0.2, gen_trained=510, own_trained=373) == 1
    assert heal_kd.simulate_steps([], 255, 2, 100, 0.2) == 0


def test_teacher_files_lists_every_folder_in_order(tmp_path):
    for d, names in (("a", ["2.pt", "1.pt", "x.txt"]), ("b", ["0.pt"])):
        (tmp_path / d).mkdir()
        for n in names:
            (tmp_path / d / n).write_text("")
    files = heal_kd.teacher_files(f"{tmp_path / 'a'},{tmp_path / 'b'}")
    assert [f"{f.parent.name}/{f.name}" for f in files] == ["a/1.pt", "a/2.pt", "b/0.pt"]
    assert heal_kd.teacher_files(str(tmp_path / "b")) == [tmp_path / "b" / "0.pt"]


def test_row_adam_resume_on_new_data_keeps_the_saved_selection():
    p = torch.nn.Parameter(torch.zeros(6, 3))
    saved = heal_kd.RowAdam([(p, [0, 2, 4])])
    p.grad = torch.ones(6, 3)
    saved.step(1e-2)
    state = saved.state_dict()
    fresh = heal_kd.RowAdam([(p, [1, 2, 3])])  # another routing count picks other experts
    assert fresh.load_state_dict(state) == 1
    assert fresh.items[0]["rows"].tolist() == [0, 2, 4] and fresh.t == 1
    assert fresh.items[0]["frozen"].tolist() == [False, True, False, True, False, True]
    assert torch.equal(fresh.items[0]["m"], saved.items[0]["m"])
    assert heal_kd.RowAdam([(p, [0, 2, 4])]).load_state_dict(state) == 0


def test_truncate_loops_drops_only_repeated_text():
    import random
    rnd = random.Random(0)
    word = lambda k: [rnd.randrange(1000, 9000) for _ in range(k)]
    head, unit, tail = word(300), word(23), word(120)
    looped = head + unit * 40 + tail  # a loop, then fresh text (the tool call after it)
    onset = heal_kd.loop_onset(looped)
    assert onset is not None and 300 + 2 * 23 - 8 <= onset <= 300 + 3 * 23  # the third pass, not the second
    block, other = word(150), word(200)
    copied = head + block + other + block  # a drafted block retyped once: no loop
    assert heal_kd.loop_onset(copied) is None and heal_kd.loop_onset(word(900)) is None
    ids = [5] * 10 + looped + [6] * 10 + copied
    reply = [False] * 10 + [True] * len(looped) + [False] * 10 + [True] * len(copied)
    out, (spans, n_looped, removed) = heal_kd.truncate_loops({"ids": torch.tensor(ids), "reply": torch.tensor(reply)})
    kept = out["reply"].tolist()
    assert (spans, n_looped) == (2, 1) and 0 < removed < 40 * 23
    assert all(kept[10:10 + onset]) and not kept[10 + onset] and all(kept[10 + len(looped) - 100:10 + len(looped)])
    assert all(kept[-len(copied):])  # the copied block stays trained




def test_sampled_topk_mode_trains_bucket_reverse_on_sampled_rows_only(monkeypatch):
    g = torch.Generator().manual_seed(2)
    weight = torch.randn(12, 8, generator=g)
    h0 = torch.randn(6, 8, generator=g)
    t = torch.log_softmax(torch.randn(6, 12, generator=g), -1)
    top, idx = t.topk(4, -1)
    tail = torch.log1p(-top.exp().sum(-1))
    mask = torch.ones(6)
    nxt = torch.randint(0, 12, (6,), generator=g)
    t_tok, b_tok = t.gather(-1, nxt[:, None])[:, 0], torch.full((6,), -1.0)
    smask = torch.tensor([1.0, 1, 0, 1, 0, 0])
    mix = {"forward": 0.25, "reverse": 0.0, "sampled": 1.0}

    def run(mode):
        monkeypatch.setattr(heal_kd, "SAMPLED_MODE", mode)
        hidden = h0.clone().requires_grad_(True)
        loss, *_ = heal_kd.kd_loss(hidden, weight, top, idx.int(), tail, mask, chunk=4, mix=mix,
                                   samp=(nxt, t_tok, b_tok, smask))
        loss.backward()
        return loss.detach(), hidden.grad

    hidden = h0.clone().requires_grad_(True)
    s = torch.log_softmax(hidden @ weight.T, -1)
    f, r = heal_kd.bucket_kl(s.gather(-1, idx), top, tail)
    ref = (0.25 * f + r * smask).sum()
    ref.backward()
    loss, grad = run("topk")
    assert abs(float(loss) - float(ref)) < 1e-4 and torch.allclose(grad, hidden.grad, atol=1e-5)
    loss_tok, grad_tok = run("token")  # the default path is unchanged and differs from topk
    assert not torch.allclose(grad_tok, grad, atol=1e-4)
    # unsampled rows get the forward term only in topk mode
    g_only = h0.clone().requires_grad_(True)
    (0.25 * heal_kd.bucket_kl(torch.log_softmax(g_only @ weight.T, -1).gather(-1, idx), top, tail)[0]).sum().backward()
    rows = smask == 0
    assert torch.allclose(grad[rows], g_only.grad[rows], atol=1e-5)


def test_kd_rows_scores_only_live_rows_with_equal_sums_and_gradients(monkeypatch):
    g = torch.Generator().manual_seed(3)
    n, v = 40, 24
    hidden = torch.randn(n, 8, generator=g, dtype=torch.float64)
    weight = torch.randn(v, 8, generator=g, dtype=torch.float64)
    t = torch.log_softmax(torch.randn(n, v, generator=g, dtype=torch.float64), -1)
    top, idx = t.topk(6, -1)
    tail = torch.log1p(-top.exp().sum(-1))
    mask = (torch.rand(n, generator=g) < 0.3).double()
    reply = mask.clone()
    reply[5] = 1.0  # a reply-only row (read, not trained)
    nxt = torch.randint(0, v, (n,), generator=g)
    smask = torch.zeros(n, dtype=torch.bool)
    smask[7] = True  # a sampled-only row
    samp = (nxt, torch.randn(n, generator=g, dtype=torch.float64), torch.randn(n, generator=g, dtype=torch.float64),
            smask)
    monkeypatch.setattr(heal_kd, "LOSS_MIX", {"forward": 0.3, "reverse": 1.0, "sampled": 0.5})
    h = heal_kd.Held(top=top, idx=idx.int(), tail=tail, mask=mask, labels=None, reply=reply, samp=samp)

    def run(rows_on, start, stop):
        monkeypatch.setenv("HEAL_KD_ROWS", "1" if rows_on else "0")
        x = hidden[start:stop].clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        rows = heal_kd.kd_rows(h, start, stop)
        assert (rows is None) != rows_on
        out = heal_kd.kd_loss(x, w, top[start:stop], idx[start:stop].int(), tail[start:stop], mask[start:stop],
                              reply=reply[start:stop], samp=tuple(s[start:stop] for s in samp), chunk=4, rows=rows)
        out[0].backward()
        return [float(out[0])] + list(out[1:]), x.grad, w.grad, rows

    for start, stop in ((0, 40), (10, 25), (30, 32)):
        base, gx, gw, _ = run(False, start, stop)
        got, hx, hw, rows = run(True, start, stop)
        live = ((mask != 0) | (reply != 0) | smask)[start:stop]
        assert rows.tolist() == (torch.nonzero(live).reshape(-1).tolist() or [0])
        assert all(abs(a - b) <= 1e-12 * max(1.0, abs(b)) for a, b in zip(got, base))
        assert torch.allclose(hx, gx, rtol=0, atol=1e-12) and torch.allclose(hw, gw, rtol=0, atol=1e-12)
        assert float(gx[~live].abs().sum()) == 0.0


def test_kd_rows_empty_window_keeps_one_inert_row(monkeypatch):
    """A rank whose window has no live row still runs the LM head (one zero-mask row): the same graph, no rank-local
    skip of the root unit's use."""
    monkeypatch.setenv("HEAL_KD_ROWS", "1")
    n, v = 12, 10
    h = heal_kd.Held(top=torch.zeros(n, 3), idx=torch.zeros(n, 3, dtype=torch.int32), tail=torch.zeros(n),
                     mask=torch.zeros(n), labels=None, reply=torch.zeros(n))
    rows = heal_kd.kd_rows(h, 4, 9)
    assert rows.tolist() == [0]
    x = torch.randn(5, 4, requires_grad=True)
    w = torch.randn(v, 4, requires_grad=True)
    loss, kd, _, _ = heal_kd.kd_loss(x, w, h.top[4:9], h.idx[4:9], h.tail[4:9], h.mask[4:9], reply=h.reply[4:9],
                                     rows=rows)
    loss.backward()
    assert kd == 0.0 and w.grad is not None and float(w.grad.abs().sum()) == 0.0
