"""v7 streamed accumulation (rl1_trainer op_accumulate / op_step): groups accumulated one call at a time, then one
step, give the same gradient, loss and update as one op_update over every window; the kept actor log-probs give the
post-step k3 measure the same pi_old as a pre-step old pass."""
import math

import pytest

torch = pytest.importorskip("torch")
from test_rl1_window_feed import bits, make_trainer, tries  # noqa: E402,F401

COMMON = {"share_prefix": True, "prefix_chunk": 16384, "signed": True, "norm": "cap", "norm_cap": 1000}


@pytest.mark.parametrize("norm,split", [("cap", 2), (None, 2), ("cap", 1), ("cap", 3)])
def test_streamed_groups_equal_one_update(monkeypatch, tries, norm, split):  # noqa: F811
    """split 2: one call per start; 1 and 3: one start's windows over two calls (the per-call prefix cache and
    snapshots are rebuilt in the second call, at the same state)."""
    common = {**COMMON, "norm": norm}
    ref = make_trainer(monkeypatch, [])
    ref.args.tis_cap = 2.0
    one = ref.op_update({"attempts": tries, "lr": 1.0, "max_windows": 100, "digest": True, **common})
    t = make_trainer(monkeypatch, [])
    t.args.tis_cap = 2.0
    groups = [tries[:split], tries[split:]]  # the fixture's starts: s0 = tries 0-1, s1 = tries 2-3
    acc = [t.op_accumulate({"attempts": g, "lr": 1.0, "max_windows": 100, "first": i == 0, **common})
           for i, g in enumerate(groups)]
    assert all(a["ok"] and a["accumulated"] for a in acc) and t.updates == 0
    assert acc[-1]["acc_calls"] == 2 and acc[-1]["acc_tokens"] == one["trained_tokens"]
    step = t.op_step({"lr": 1.0, "max_windows": 100, "digest": True, **common})
    assert step["ok"] and step["applied"] and t.updates == 1 and t.acc is None
    assert step["grad_digest"] == one["grad_digest"]
    assert step["loss"] == pytest.approx(one["loss"], rel=1e-12, abs=1e-15)
    assert step["trained_tokens"] == one["trained_tokens"] and step["accumulated_calls"] == 2
    assert bits(t.params[0][1].reshape(1)) == bits(ref.params[0][1].reshape(1))


def test_step_without_accumulate_refuses(monkeypatch, tries):  # noqa: F811
    t = make_trainer(monkeypatch, [])
    t.acc = None
    assert t.op_step({"lr": 1.0, **COMMON})["ok"] is False and t.updates == 0


def test_first_resets_a_half_accumulated_batch(monkeypatch, tries):  # noqa: F811
    ref = make_trainer(monkeypatch, [])
    ref.op_accumulate({"attempts": tries[2:], "lr": 1.0, "max_windows": 100, "first": True, **COMMON})
    clean = ref.op_step({"lr": 1.0, "digest": True, **COMMON})
    t = make_trainer(monkeypatch, [])
    t.op_accumulate({"attempts": tries[:2], "lr": 1.0, "max_windows": 100, "first": True, **COMMON})  # abandoned
    t.op_accumulate({"attempts": tries[2:], "lr": 1.0, "max_windows": 100, "first": True, **COMMON})
    assert t.op_step({"lr": 1.0, "digest": True, **COMMON})["grad_digest"] == clean["grad_digest"]


def test_kept_actor_gives_the_old_pass_pi_old(monkeypatch, tries):  # noqa: F811
    sub = [tries[0], tries[3]]
    ref = make_trainer(monkeypatch, [])
    ref.op_old_logprobs({"attempts": sub, **COMMON})
    names = list(ref.old)
    t = make_trainer(monkeypatch, [])
    t.args.tis_cap = 2.0
    keep = sorted({k[0] for k in names})
    for i, g in enumerate([tries[:2], tries[2:]]):
        t.op_accumulate({"attempts": g, "lr": 1.0, "max_windows": 100, "first": i == 0, "keep_old": keep, **COMMON})
    assert bits(t.old) == bits(ref.old)
    t.op_step({"lr": 1.0, **COMMON})
    measure = t.op_update({"attempts": sub, "lr": 0.0, "max_windows": 100, **COMMON, "policy_loss": "gspo",
                           "gspo_clip": [0.015, 0.019], "kl_stop": -1.0})
    assert measure["kl_stopped"] and measure["approx_kl"] > 0 and t.updates == 1


def test_one_call_keep_old_and_forward_k3(monkeypatch, tries):  # noqa: F811
    """mg.k3_measure: one op_update keeps the subsample's pi_old by try name (no old pass); op_k3_score after the
    step gives the k3 the lr-0 update path measures, forward only, and drops the kept pi_old."""
    sub = [tries[0], tries[3]]
    ref = make_trainer(monkeypatch, [])
    ref.op_old_logprobs({"attempts": sub, **COMMON})
    t = make_trainer(monkeypatch, [])
    t.args.tis_cap = 2.0
    t.old = {("stale", 1, 0): torch.zeros(1)}
    t.op_update({"attempts": tries, "lr": 1.0, "max_windows": 100, "keep_old": [s["name"] for s in sub], **COMMON})
    assert t.updates == 1 and bits(t.old) == bits(ref.old)
    kept = dict(t.old)
    k3 = t.op_k3_score({"attempts": sub, **COMMON, "gspo_clip": [0.015, 0.019]})
    assert k3["ok"] and k3["kept"] == len(kept) and k3["windows"] == len(kept) and t.old == {}
    t.old = kept
    lr0 = t.op_update({"attempts": sub, "lr": 0.0, "max_windows": 100, **COMMON, "policy_loss": "gspo",
                       "gspo_clip": [0.015, 0.019], "kl_stop": -1.0})
    assert lr0["kl_stopped"] and t.updates == 1
    assert k3["approx_kl"] == pytest.approx(lr0["approx_kl"], rel=1e-4) and k3["approx_kl"] > 0
    assert k3["approx_kl_tok"] == pytest.approx(lr0["approx_kl_tok"], rel=1e-4)
    assert k3["clip_frac"] == pytest.approx(lr0["clip_frac"], abs=1e-9)
    tail = k3["k3_tok_tail"]  # approx_kl_tok's tail: p99 <= max, the top token is the max's
    assert tail["tokens"] > 0 and 0 <= tail["p99"] <= tail["max"] and tail["top"]["k3"] == pytest.approx(tail["max"], rel=1e-3)
    assert tail["top"]["k3"] == pytest.approx(math.expm1(tail["top"]["g"]) - tail["top"]["g"], rel=1e-3, abs=1e-6)
    assert tail["top"]["token"] is None or isinstance(tail["top"]["token"], int)


def test_tie_by_index_takes_the_lowest_tied_blocks():
    import rl1_trainer

    masked = torch.tensor([[0.0, 3.0, 0.0, 0.0, 2.0, float("-inf")], [1.0, 1.0, 1.0, 5.0, 1.0, 1.0]])
    picked = torch.tensor([[1, 4, 3], [3, 5, 4]])  # a valid top-3 whose tie picks are not the lowest
    got = torch.sort(rl1_trainer.tie_by_index(masked, picked, 3), dim=1).values
    assert got.tolist() == [[0, 1, 4], [0, 1, 3]]
