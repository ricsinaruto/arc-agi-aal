"""heal_kd.bucket_kl: forward and reverse KL over the teacher's top-k + tail bucket (OPD loss, 18 Oct)."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
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


def think_row(tail_logit=-8.8, eos_teacher=-19.0, p_eos=0.004, V=2000, K=128):
    """A synthetic think position (v32, 25 Oct): the teacher's top-128 hold all but ~1.6e-4 of its mass, <|im_end|>
    (the last id) sits at log p -19 outside the top-128; the student equals the teacher except p(EOS) = 0.4%."""
    g = torch.Generator().manual_seed(0)
    eos = V - 1
    tz = torch.full((V,), -30.0)
    tz[:K] = torch.randn(K, generator=g) * 2
    tz[K:eos] = tail_logit + 0.1 * torch.randn(eos - K, generator=g)
    tz[eos] = eos_teacher + torch.logsumexp(tz[:eos], 0)
    t = torch.log_softmax(tz, -1)
    sz = tz.clone()
    sz[eos] = math.log(p_eos / (1 - p_eos)) + torch.logsumexp(tz[:eos], 0)
    top, idx = t.topk(K)
    return t, sz, top, idx, eos


def trained(z, t, top, idx, eos, forced):
    """The c7n-c7q objective on a sampled row: 0.5 forward + 1.0 reverse over the teacher columns + tail bucket."""
    s = torch.log_softmax(z, -1)
    if forced:
        top2, idx2, tail2 = heal_kd.merge_forced(top[None], idx[None], [eos], t[eos].view(1, 1))
        f, r = heal_kd.bucket_kl(s[idx2[0]][None], top2, tail2, masked=True)
    else:
        f, r = heal_kd.bucket_kl(s[idx][None], top[None], torch.log1p(-top.exp().sum()).view(1), masked=False)
    return 0.5 * f + r, f, r


def test_forced_eos_term_charges_in_think_eos_against_the_bucket_only_path():
    t, sz, top, idx, eos = think_row()
    assert eos not in idx.tolist() and float(1 - top.exp().sum()) < 3e-4
    out = {}
    for forced in (False, True):
        z = sz.clone().requires_grad_(True)
        loss, f, r = trained(z, t, top, idx, eos, forced)
        loss.sum().backward()
        out[forced] = (float(loss), float(f), float(r), float(z.grad[eos]))
    p = 0.004
    own = p * (math.log(p) - float(t[eos]))  # the EOS term s log(s / t) = 0.054
    # bucket only (as trained in c7n-c7q): reverse 0.0096, forward 0.0035, dL/dz_eos 0.015
    # forced: reverse 0.050 (the EOS term less the head's mass shift), dL/dz_eos 0.056
    assert abs(out[True][2] - own) < 0.1 * own
    assert out[True][2] > 4 * out[False][2]
    assert out[True][3] > 3 * out[False][3] > 0
    assert abs(out[True][1] - out[False][1]) < 1e-3  # forward is near-blind either way (~p)


def test_forced_column_already_in_the_top_k_changes_nothing():
    t, sz, top, idx, _ = think_row()
    inside = int(idx[3])
    s = torch.log_softmax(sz, -1)
    top2, idx2, tail2 = heal_kd.merge_forced(top[None], idx[None], [inside], t[inside].view(1, 1))
    assert torch.isnan(top2[0, -1])
    f2, r2 = heal_kd.bucket_kl(s[idx2[0]][None], top2, tail2, masked=True)
    tail = torch.log1p(-top.double().exp().sum()).float().view(1)  # as merge_forced (float64: 1 - 0.99984)
    f1, r1 = heal_kd.bucket_kl(s[idx][None], top[None], tail, masked=False)
    assert torch.allclose(f1, f2, rtol=1e-4) and torch.allclose(r1, r2, rtol=1e-4)
    z = sz.clone().requires_grad_(True)
    s = torch.log_softmax(z, -1)
    f, r = heal_kd.bucket_kl(s[idx2[0]][None], top2, tail2, masked=True)
    (0.5 * f + r).sum().backward()
    assert torch.isfinite(z.grad).all()  # the NaN column leaks no NaN into the gradient


def test_masked_bucket_equals_plain_without_nan():
    g = torch.Generator().manual_seed(1)
    s = torch.log_softmax(torch.randn(6, 50, generator=g), -1)
    t = torch.log_softmax(torch.randn(6, 50, generator=g), -1)
    top, idx = t.topk(10, -1)
    tail = torch.log1p(-top.exp().sum(-1))
    a = heal_kd.bucket_kl(s.gather(-1, idx), top, tail, masked=False)
    b = heal_kd.bucket_kl(s.gather(-1, idx), top, tail, masked=True)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))


def test_think_rows_cover_the_open_think_block_only():
    END, IM, TC = 9, 8, 7
    #        prompt      | reply 1: think 3 tokens, </think>, "\n\n", <tool_call>, x, <|im_end|> | prompt | reply 2: in-think EOS
    ids = torch.tensor([1, 2, 3, 4, 5, 6, END, 2, TC, 3, IM, 1, 2, 4, 5, IM])
    reply = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], dtype=torch.bool)
    rows = heal_kd.think_rows(ids, reply, END)
    # rows predicting tokens 3, 4, 5 (think) and 6 (</think>) are open; rows predicting 7.. are not; reply 2: rows
    # predicting 13, 14, 15 (its in-think <|im_end|> included)
    assert rows.nonzero().flatten().tolist() == [2, 3, 4, 5, 12, 13, 14]


def test_think_eos_mask_turns_the_eos_gradient_around_on_a_think_row():
    """Step A's case: the teacher endorses EOS mid-think (log p -2.49), the student is at p 0.4%. Bucket-only and
    forced-column paths both pull the student's EOS up (toward ~8%); the mask sets the teacher's EOS to EOS_FLOOR,
    renormalises the rest, and the same objective pushes EOS down."""
    t, sz, top, idx, eos = think_row(eos_teacher=-2.49)
    assert eos in idx.tolist()  # at -2.49 it is in the top-128
    tail = torch.log1p(-top.double().exp().sum()).float().view(1)
    rows = torch.tensor([True])
    top_m, tail_m, _, n_m, n_u = heal_kd.mask_think_eos(top[None], idx[None], tail, rows, eos)
    assert n_m == 1 and n_u == 0
    col = idx == eos
    assert float(top_m[0][col]) == heal_kd.EOS_FLOOR
    total = top_m[0][~col].double().exp().sum() + tail_m.double().exp().sum()
    assert abs(float(total) - 1.0) < 1e-4
    grads = {}
    for name, (tt, tl) in {"plain": (top[None], tail), "masked": (top_m, tail_m)}.items():
        z = sz.clone().requires_grad_(True)
        s = torch.log_softmax(z, -1)
        f, r = heal_kd.bucket_kl(s[idx][None], tt, tl, masked=False)
        (0.5 * f + r).sum().backward()
        grads[name] = float(z.grad[eos])
    assert grads["plain"] < 0 < grads["masked"]  # descent: plain raises z_eos, masked lowers it
    assert grads["masked"] > 0.05  # ~p (log p/1e-13 + 1): strong


def test_think_eos_mask_uses_the_forced_column_and_leaves_other_rows():
    t, sz, top, idx, eos = think_row()  # EOS at -19: outside the top-128, in its forced column
    assert eos not in idx.tolist()
    top2, idx2, tail2 = heal_kd.merge_forced(top[None].expand(2, -1), idx[None].expand(2, -1), [eos],
                                             t[eos].view(1, 1).expand(2, 1))
    rows = torch.tensor([True, False])
    top_m, tail_m, t_tok, n_m, _ = heal_kd.mask_think_eos(top2, idx2, tail2, rows, eos,
                                                          torch.tensor([float(t[eos]), -1.0]), torch.tensor([eos, 3]))
    assert n_m == 1 and float(top_m[0, -1]) == heal_kd.EOS_FLOOR and float(t_tok[0]) == heal_kd.EOS_FLOOR
    assert torch.equal(top_m[1], top2[1]) and float(tail_m[1]) == float(tail2[1]) and float(t_tok[1]) == -1.0
    assert abs(float(top_m[0, :-1].double().exp().sum() + tail_m[0].double().exp()) - 1.0) < 1e-4
