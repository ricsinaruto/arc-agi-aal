"""steps after an update's first are clipped against the update's own first weights (θ0): GSPO's
sequence region ("seq") or PPO's token region ("tok"). The loss math on hand-made tensors, then the tiny Flash-shaped
trainer of test_rl1_dist through old_pass / keep_old / clipped accumulate / step."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

import test_rl1_dist as base  # noqa: E402


def bare(clip, tis_cap=0.0):
    import rl1_trainer as rt

    t = rt.Trainer.__new__(rt.Trainer)
    t.args = SimpleNamespace(tis_cap=tis_cap)
    t.ratio_clip = rt.ratio_clip_spec(clip)
    return t


def grad_of(t, arrays, actor0, valid=None):
    actor = actor0.clone().requires_grad_(True)
    valid = torch.ones(actor.shape[0], dtype=torch.bool) if valid is None else valid
    loss, _, signed = t.weighted_loss(arrays, actor, valid, 1.0)
    loss.backward()
    return actor.grad, signed() if callable(signed) else signed


def test_spec_presets_and_refusals():
    import rl1_trainer as rt

    assert rt.ratio_clip_spec("gspo") == {"mode": "seq", "low": 3e-4, "high": 4e-4}
    assert rt.ratio_clip_spec("ppo") == {"mode": "tok", "low": 0.2, "high": 0.28}
    assert rt.ratio_clip_spec("off") is None and rt.ratio_clip_spec(None) is None
    for bad in ("grpo", {"mode": "seq", "low": 0.0, "high": 0.1}, {"mode": "x", "low": 0.1, "high": 0.1}):
        with pytest.raises(ValueError, match="ratio_clip"):
            rt.ratio_clip_spec(bad)


def test_token_clip_zeroes_exactly_the_tokens_out_of_region_in_their_advantage_direction():
    old = torch.full((6,), -1.0)
    logr = torch.tensor([0.0, 0.5, -0.5, 0.5, -0.5, 0.1])
    weights = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    arrays = {"weights": weights, "old": old, "old_valid": torch.ones(6, dtype=torch.bool),
              "positions": torch.arange(6)}
    g, signed = grad_of(bare("ppo"), arrays, old + logr)
    # up past 1.28 with A>0 (token 1) and down past 0.8 with A<0 (token 4): zero; the rest -w exactly
    assert g.tolist() == [-1.0, 0.0, -1.0, 1.0, 0.0, -1.0]
    assert signed["ratio_clip"]["clipped"] == 2 and signed["ratio_clip"]["of"] == 6
    assert signed["ratio_clip"]["max_abs_logr"] == pytest.approx(0.5)
    g_off, signed_off = grad_of(bare(None), arrays, old + logr)  # off: the plain REINFORCE gradient, no tally
    assert g_off.tolist() == (-weights).tolist() and "ratio_clip" not in signed_off


def test_sequence_clip_uses_each_replys_mean_log_ratio():
    old = torch.full((8,), -2.0)
    # reply 0: mean +1e-3 (> log 1.0004) with A>0 -> all 3 zero; reply 1: mean 0 -> kept;
    # reply 2: mean -1e-3 (< log 0.9997) with A<0 -> zero; reply 3: mean -1e-3 with A>0 -> kept (wrong direction)
    logr = torch.tensor([0.001, 0.0005, 0.0015, 0.0002, -0.0002, -0.001, -0.001, -0.001])
    reply = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    arrays = {"weights": weights, "old": old, "old_valid": torch.ones(8, dtype=torch.bool), "reply": reply,
              "positions": torch.arange(8)}
    g, signed = grad_of(bare("gspo"), arrays, old + logr)
    assert g.tolist() == [0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0]
    assert signed["ratio_clip"]["clipped"] == 5
    # a token without θ0 (old_valid False) is left out of its reply's mean and never clipped
    ov = torch.ones(8, dtype=torch.bool)
    ov[0] = False
    g2, _ = grad_of(bare("gspo"), {**arrays, "old_valid": ov}, old + logr)
    assert g2[0].item() == -1.0 and g2[1].item() == 0.0  # reply 0's mean without token 0: +1e-3, still out


def test_clip_reads_theta0_not_the_engine_reference():
    """TIS stays against the engine (``reference``); the region is against θ0 only: a window 0.09 nats off the
    engine but at θ0 is not clipped (the engine reference clipped every A<0 reply at step 1, 30 Oct)."""
    actor = torch.full((5,), -1.0)
    arrays = {"weights": torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0]), "reference": actor + 0.09, "old": actor.clone(),
              "old_valid": torch.ones(5, dtype=torch.bool), "reply": torch.tensor([0, 0, 1, 1, 2]),
              "positions": torch.arange(5)}
    g, signed = grad_of(bare("gspo", tis_cap=2.0), arrays, actor)
    g_off, _ = grad_of(bare(None, tis_cap=2.0), arrays, actor)
    assert signed["ratio_clip"]["clipped"] == 0 and torch.equal(g, g_off)
    assert g_off[0].item() == pytest.approx(-math.exp(-0.09), rel=1e-5)  # TIS weight still applied


def test_no_theta0_trains_unclipped():
    actor = torch.tensor([-1.0, -0.5])
    arrays = {"weights": torch.tensor([1.0, -1.0]), "positions": torch.arange(2)}
    g, signed = grad_of(bare("ppo"), arrays, actor)
    assert g.tolist() == [-1.0, 1.0] and "ratio_clip" not in signed


@base.isolate
def test_tiny_trainer_old_pass_keep_old_and_clipped_steps(tmp_path):
    """θ0 two ways (forward-only old_pass; keep_old on the grad pass) agree; a clipped accumulate at θ0 clips
    nothing; after a step the region bites and the step row reports it; a window without θ0 is counted."""
    t = base.trainer(tmp_path / "t")
    specs = base.specs()
    got = t.op_old_pass({"attempts": specs, "lr": base.LR, "old_reset": True})
    assert got["ok"] and got["old_pass"] and got["old_windows"] == 3 and got["old_kept"] == 3, got
    assert all(p.grad is None for _, p in t.params)  # forward only: nothing accumulated
    passed = {k: (a.clone(), v.clone()) for k, (a, v) in t.old_lp.items()}
    acc = t.op_accumulate({"attempts": specs, "lr": base.LR, "first": True, "keep_old": True, "old_reset": True})
    assert acc["ok"], acc
    for name, (a, v) in passed.items():  # eval no-grad pass == the train-mode grad pass at the same weights
        assert torch.equal(v, t.old_lp[name][1])
        assert torch.allclose(a[v], t.old_lp[name][0][v], atol=1e-5), name
    # a clipped accumulate still at θ0: r = 1, nothing clipped, every window attached
    t.acc_reset()
    acc = t.op_accumulate({"attempts": specs, "lr": base.LR, "first": True, "ratio_clip": "gspo"})
    assert acc["ok"] and acc["ratio_clip"]["attached"] == 3 and acc["ratio_clip"]["missing"] == 0, acc
    step = t.op_step({"lr": 0.05})
    assert step["ok"] and step["ratio_clip"]["frac"] == 0.0 and step["ratio_clip"]["max_abs_logr"] < 1e-4, step
    # after the step (θ1 != θ0): clipped against θ0, the region bites
    acc = t.op_accumulate({"attempts": specs, "lr": base.LR, "first": True, "ratio_clip": "gspo"})
    assert acc["ok"], acc
    step = t.op_step({"lr": base.LR})
    assert step["ok"] and step["ratio_clip"]["frac"] > 0 and step["ratio_clip"]["max_abs_logr"] > 4e-4, step
    # a window θ0 never saw trains unclipped and is counted
    t.old_lp.pop("a")
    acc = t.op_accumulate({"attempts": specs, "lr": base.LR, "first": True, "ratio_clip": "ppo"})
    assert acc["ok"] and acc["ratio_clip"]["missing"] == 1 and acc["ratio_clip"]["attached"] == 2, acc
    assert t.op_step({"lr": base.LR})["ok"]
    # old_reset on the next update's first call drops the previous θ0
    t.op_old_pass({"attempts": specs[:1], "lr": base.LR, "old_reset": True})
    assert set(t.old_lp) == {specs[0]["name"]}


def test_theta0_key_separates_same_named_windows_of_different_tries():
    """a10 U23: two groups from one start carry the same try labels; keyed by name alone, the second group's θ0
    overwrote the first's and 8 of 67 windows trained unclipped."""
    import rl1_trainer as rt

    made = {"name": "a000-s5i5-game-L1-s5i5-L1-p0"}
    a = rt.window_key({"rollout": "/runs/x/p26g542-s5i5-L1/attempts/a000/rollout.jsonl"}, made)
    b = rt.window_key({"rollout": "/runs/x/p26g557-s5i5-L1/attempts/a000/rollout.jsonl"}, made)
    assert a != b and made["name"] in a
    assert rt.window_key({}, made) == made["name"]  # specs without a rollout (tests, probes): the name
