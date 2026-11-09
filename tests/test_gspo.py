"""GSPO loss option: sequence-level ratio and clipping per reply, token-level gradient at s = 1."""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import rl1_trainer  # noqa: E402

REPLY = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
VALID = torch.tensor([True, True, False, True, True, True, True, True, True])
W = torch.tensor([0.5, 0.5, 0.5, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0])


def leaf():
    return torch.linspace(-2.0, -0.5, len(REPLY), dtype=torch.float32).requires_grad_(True)


def test_gradient_equals_token_level_at_ratio_one():
    a, b = leaf(), leaf()
    loss, stats = rl1_trainer.gspo_loss(a, None, None, W, REPLY, VALID)
    loss.backward()
    token = -(W[VALID] * b[VALID]).sum()
    token.backward()
    assert torch.allclose(a.grad, b.grad) and stats["clipped"] == 0 and stats["replies"] == 3


def test_sequence_clipping_zeroes_the_clipped_replies_only():
    a = leaf()
    shift = torch.tensor([0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
    old = a.detach() - shift  # reply 0: s = e^0.01 > 1 + 4e-4 with A > 0; reply 1: s < 1 - 3e-4 with A < 0
    loss, stats = rl1_trainer.gspo_loss(a, old, None, W, REPLY, VALID)
    loss.backward()
    assert stats["clipped"] == 2
    assert torch.all(a.grad[:5] == 0)
    # reply 2 (A = 2 > 0, s = e^-0.01 below the band: unclipped side): d/dx of -(4 * 2 * s) with s = exp(mean) -> -2 s
    assert torch.allclose(a.grad[5:], torch.full((4,), -2.0 * math.exp(-0.01)), atol=1e-6)


def test_sequence_tis_caps_up_and_bounds_down():
    a = leaf()
    reference = a.detach() - math.log(3.0)  # pi_old over the sampler: 3 on every reply
    loss, _ = rl1_trainer.gspo_loss(a, None, reference, W, REPLY, VALID, tis_cap=2.0)
    loss.backward()
    expect = torch.zeros_like(W)
    expect[VALID] = -W[VALID] * torch.where(W[VALID] < 0, torch.tensor(1.0), torch.tensor(2.0))
    assert torch.allclose(a.grad, expect)


def test_gspo_stats_split_clips_by_advantage_sign():
    a = leaf()
    shift = torch.tensor([0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
    _, stats = rl1_trainer.gspo_loss(a, a.detach() - shift, None, W, REPLY, VALID)
    assert (stats["clipped_up"], stats["clipped_down"], stats["up"], stats["down"]) == (1, 1, 2, 1)
    assert abs(stats["log_ratio_mean"] - (0.01 - 0.01 - 0.01) / 3) < 1e-6


def test_reply_gaps_and_stats_match_gspo_clipping():
    a = leaf().detach()
    shift = torch.tensor([0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
    row = rl1_trainer.reply_gaps(a, a - shift, REPLY, VALID, W)
    assert [round(x, 6) for x in row["log_s"]] == [0.01, -0.01, -0.01] and row["a"] == [0.5, -1.0, 2.0]
    stats = rl1_trainer.gap_stats([row, rl1_trainer.reply_gaps(a, a[:3], REPLY, VALID, W)])
    _, loss_stats = rl1_trainer.gspo_loss(a.clone().requires_grad_(True), a - shift, None, W, REPLY, VALID)
    assert stats["missing"] == 1 and stats["replies"] == 3
    assert stats["clipped_up"] + stats["clipped_down"] == loss_stats["clipped"]
    assert abs(stats["token_gap_mean"] - float(shift[VALID].mean())) < 1e-9
    assert stats["outside_3e-4"] == 1.0 and stats["outside_1e-2"] == 0.0


def test_chunk_check_settings_refs_and_restore():
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.prefix_chunk = 16384
    t.expand = lambda specs, **kw: specs
    t.attempt_window = lambda spec, seed=0, release=True: {
        "name": spec["name"], "arrays": {"reply": torch.tensor([0, 0, 1, 1]), "weights": torch.tensor([1.0, 1.0, -1.0, -1.0])}}

    def window_pass(made, shared=None):  # log-probs drift with the chunk under qsa, not under dense
        shift = 0.0 if rl1_trainer.SELECT["mode"] == "dense" else 1e-3 * (t.prefix_chunk // 2048)
        return {"actor": torch.full((4,), -1.0) + shift, "valid": torch.ones(4, dtype=torch.bool)}

    t.window_pass = window_pass
    t.mismatch = lambda made, out: {"tokens": 4, "mean_abs": float(out["actor"][0].abs()), "mean": 0.0, "beyond_half": 0.0}
    settings = [{"chunk": 32768, "select": "qsa"}, {"chunk": 2048, "select": "qsa"},
                {"chunk": 32768, "select": "dense"}, {"chunk": 2048, "select": "dense"}]
    out = t.op_chunk_check({"attempts": [{"name": "w1"}, {"name": "w2"}], "settings": settings})
    rows = {r["setting"]: r for r in out["settings"]}
    assert out["windows"] == 2 and rows["qsa@2048"]["against"] == "qsa@32768" and rows["dense@2048"]["against"] == "dense@32768"
    assert rows["qsa@32768"]["gap"]["log_s_abs_max"] == 0.0 and rows["dense@2048"]["gap"]["log_s_abs_max"] == 0.0
    assert abs(rows["qsa@2048"]["gap"]["log_s_abs_mean"] - 0.015) < 1e-9 and rows["qsa@2048"]["sampler_gap"]["tokens"] == 8
    assert t.prefix_chunk == 16384 and rl1_trainer.SELECT["mode"] == "qsa"


def test_one_prefix_chunk_for_every_pass():
    """20 Oct: pi_old, the update and the probes take the trainer's pinned chunking; another chunk is an error and the
    prefix pass never falls back to a different one."""
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.path_chunk = t.prefix_chunk = 16384
    t.pin_chunk({"share_prefix": True})
    t.pin_chunk({"prefix_chunk": 16384})
    assert t.prefix_chunk == 16384
    with pytest.raises(ValueError, match="pinned at 16384"):
        t.pin_chunk({"prefix_chunk": 32768})
    starts = []

    class Model:
        def eval(self):
            pass

        def train(self):
            pass

        def __call__(self, input_ids, attention_mask, past_key_values, use_cache, logits_to_keep):
            starts.append((input_ids.shape[1], attention_mask.shape[1]))
            return type("Out", (), {"past_key_values": (past_key_values or 0) + 1})()

    t.model = Model()
    ids = torch.zeros((1, 40000), dtype=torch.long)
    cache, chunk, snap = t.prefix_pass(ids, 40000, lambda start, stop: {})
    assert chunk == 16384 and cache == 3 and snap is None and starts == [(16384, 16384), (16384, 32768), (7232, 40000)]


def test_trainer_attributes_do_not_shadow_methods():
    """20 Oct (rl1-loop-p2b-c7g6): ``self.pinned = {}`` (parked host buffers, set in __init__) shadowed the new
    ``pinned(cmd)`` method, so every real update raised "'dict' object is not callable"; tests that build the trainer
    with __new__ never saw it. No instance attribute may share a name with a method."""
    import ast

    tree = ast.parse(Path(rl1_trainer.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    attrs = set()
    for node in ast.walk(cls):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
        for t in targets:
            for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self":
                    attrs.add(e.attr)
    assert not (methods & attrs), sorted(methods & attrs)


def test_fake_trainer_refuses_another_chunk():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from arena import fake_trainer

    fake = fake_trainer.Fake(16384)
    assert fake.run("old_logprobs", {"attempts": [], "prefix_chunk": 16384})["windows"] == 0
    with pytest.raises(ValueError, match="pinned at 16384"):
        fake.run("update", {"attempts": [], "prefix_chunk": 32768})


def test_paper_weighting_counts_each_reply_once():
    a = leaf()
    loss, _ = rl1_trainer.gspo_loss(a, None, None, W, REPLY, VALID, norm="reply")
    loss.backward()
    count = torch.zeros(3).index_add_(0, REPLY, VALID.float())
    expect = torch.where(VALID, -W * rl1_trainer.GSPO_REPLY_SCALE / count[REPLY], torch.zeros_like(W))
    assert torch.allclose(a.grad, expect)  # 1/|y_i| x the reply's token gradients, a constant scale


def test_paper_band_clips_outside_plus_minus_0015():
    a = leaf()
    shift = torch.tensor([0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    _, stats = rl1_trainer.gspo_loss(a, a.detach() - shift, None, W, REPLY, VALID, clip=(0.015, 0.015), norm="reply")
    assert stats["clipped"] == 1 and stats["clipped_up"] == 1  # reply 0 (A > 0, s = e^0.02); s = e^0.01 inside


def _optim_trainer(monkeypatch, scale):
    monkeypatch.setenv("RL1_EXPERT_LR_SCALE", str(scale))
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    names = ["model.layers.0.mlp.experts.base_layer.lora_B", "model.layers.0.mlp.shared_expert.up.lora_B",
             "model.layers.0.self_attn.q_proj.lora_B"]
    t.params = [(n, torch.nn.Parameter(torch.zeros(4))) for n in names]

    class Args:
        optimizer = "adamw32"

    t.args = Args()
    return t


ADAMW = torch.optim.AdamW


def _adam(params, lr, **kw):  # the trainer's AdamW without fused (CPU)
    return ADAMW(params, lr=lr, betas=kw["betas"], eps=kw["eps"], weight_decay=kw["weight_decay"])


def test_expert_lr_scale_groups_and_freeze(monkeypatch):
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    t = _optim_trainer(monkeypatch, 0.1)
    t.optimizer = t.make_optimizer(1e-4)
    assert [g["scale"] for g in t.optimizer.param_groups] == [1.0, 0.1]
    assert t.optimizer.param_groups[1]["params"][0] is t.params[0][1]  # only the routed experts
    t.set_lr(2e-4)
    assert [g["lr"] for g in t.optimizer.param_groups] == pytest.approx([2e-4, 2e-5])
    one = _optim_trainer(monkeypatch, 1.0)
    one.optimizer = one.make_optimizer(1e-4)
    assert len(one.optimizer.param_groups) == 1  # as before: old checkpoints load
    frozen = _optim_trainer(monkeypatch, 0.0)
    frozen.optimizer = frozen.make_optimizer(1e-4)
    for _, p in frozen.params:
        p.grad = torch.ones(4)
    frozen.set_lr(1e-3)
    frozen.optimizer.step()
    assert torch.all(frozen.params[0][1] == 0) and torch.all(frozen.params[1][1] != 0)


def test_checkpoint_groups_must_match(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    st = pytest.importorskip("safetensors.torch")
    a = _optim_trainer(monkeypatch, 0.1)
    a.optimizer, a.updates = a.make_optimizer(1e-4), 0
    a.op_checkpoint({"path": str(tmp_path / "c")})
    b = _optim_trainer(monkeypatch, 1.0)
    b.optimizer, b.updates = b.make_optimizer(1e-4), 0
    b.adapter_norm = lambda: 0.0
    b.forget_unaccepted = lambda: None
    with pytest.raises(ValueError, match="optimizer groups"):
        b.op_load_checkpoint({"path": str(tmp_path / "c")})
    c = _optim_trainer(monkeypatch, 0.1)
    c.optimizer, c.updates = c.make_optimizer(1e-4), 0
    c.adapter_norm = lambda: 0.0
    c.forget_unaccepted = lambda: None
    out = c.op_load_checkpoint({"path": str(tmp_path / "c")})
    assert [g["scale"] for g in c.optimizer.param_groups] == [1.0, 0.1] and out["updates"] == 0


def test_k3_approx_kl_per_reply_and_token():
    """Standard dosing: k3 = (r - 1) - log r on GSPO's per-reply ratio and per token; 0 at pi = pi_old."""
    a = leaf()
    _, same = rl1_trainer.gspo_loss(a, None, None, W, REPLY, VALID)
    assert same["k3_seq_sum"] == 0.0 and same["k3_tok_sum"] == 0.0 and same["tokens"] == 8
    shift = torch.tensor([0.01, 0.01, 0.01, -0.02, -0.02, 0.0, 0.0, 0.0, 0.0])
    _, st = rl1_trainer.gspo_loss(a, a.detach() - shift, None, W, REPLY, VALID)
    k3 = lambda x: math.exp(x) - 1.0 - x
    assert math.isclose(st["k3_seq_sum"], k3(0.01) + k3(-0.02) + 0.0, rel_tol=1e-4)
    assert math.isclose(st["k3_tok_sum"], 2 * k3(0.01) + 2 * k3(-0.02), rel_tol=1e-4)  # the masked token excluded
    seq, tok, clip = rl1_trainer.approx_kl([{"signed": {"gspo": st}}, {"signed": {"gspo": same}}])
    assert math.isclose(seq, st["k3_seq_sum"] / 6) and math.isclose(tok, st["k3_tok_sum"] / 16) and clip == st["clipped"] / 6
    assert rl1_trainer.approx_kl([{}]) == (None, None, None)


def test_log_s_quantiles_toward_the_clip_side():
    """Band sizing: reply 0 (a > 0) log s +0.01, reply 1 (a < 0) log s -0.02 -> +0.02 toward its clip side, reply 2 0."""
    a = leaf()
    shift = torch.tensor([0.01, 0.01, 0.01, -0.02, -0.02, 0.0, 0.0, 0.0, 0.0])
    _, st = rl1_trainer.gspo_loss(a, a.detach() - shift, None, W, REPLY, VALID)
    assert sorted(round(x, 5) for x in st["log_s_adv"]) == [0.0, 0.01, 0.02]
    assert sorted(round(x, 5) for x in st["log_s_abs"]) == [0.0, 0.01, 0.02]
    q = rl1_trainer.log_s_quantiles([{"signed": {"gspo": st}}, {}])
    assert round(q["adv"]["p50"], 5) == 0.01 and round(q["adv"]["p90"], 5) == 0.02 and q["adv"]["n"] == 3
    assert rl1_trainer.log_s_quantiles([{}]) is None


def test_narrower_bands_are_counted_not_applied():
    a = leaf()
    shift = torch.tensor([0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
    _, stats = rl1_trainer.gspo_loss(a, a.detach() - shift, None, W, REPLY, VALID, clip=(0.015, 0.015))
    assert stats["clipped"] == 0 and stats["clipped_alt"] == {"0.004": 2, "0.004_0.005": 2}
    small = torch.tensor([0.0045] * 3 + [-0.0045] * 2 + [0.0] * 4)  # inside 0.005 up, outside 0.004 down
    _, stats = rl1_trainer.gspo_loss(a, a.detach() - small, None, W, REPLY, VALID, clip=(0.015, 0.015))
    assert stats["clipped_alt"] == {"0.004": 2, "0.004_0.005": 1}


def test_tis_out_counts_replies_and_tokens_outside_half_to_two():
    a = leaf()
    drift = torch.tensor([1.0, 1.0, 5.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])  # reply 0: mean 1.0 > log 2; reply 1: 0.1
    _, stats = rl1_trainer.gspo_loss(a, None, a.detach() - drift, W, REPLY, VALID, tis_cap=2.0)
    assert stats["tis_out"] == {"replies": 1, "tokens": 2}  # the invalid token (5.0) is not counted


def test_token_tis_counts_tokens_outside_half_to_two():
    from types import SimpleNamespace
    me = SimpleNamespace(args=SimpleNamespace(tis_cap=2.0), policy={"loss": "token"})
    a = leaf()
    drift = torch.tensor([1.0, 0.0, 5.0, -1.0, 0.1, 0.0, 0.0, 0.0, 0.0])  # trainer - sampler; token 2 invalid
    arrays = {"weights": W, "reference": a.detach() - drift}
    _, tis, signed = rl1_trainer.Trainer.weighted_loss(me, arrays, a, VALID, 1.0)
    assert signed["tis_out"] == {"tokens": 2, "of": 8} and float(tis[0]) == 2.0


def test_fake_trainer_k3_score_needs_the_steps_kept_pi_old(tmp_path):
    """The CPU preflight's stand-in follows rl1_trainer's k3 contract: k3_score scores tries the last update kept
    (keep_old), once."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from arena import fake_trainer

    roll = tmp_path / "t0" / "rollout.jsonl"
    roll.parent.mkdir()
    roll.write_text('{"usage": {"completion_tokens": 7}}\n')
    fake = fake_trainer.Fake()
    spec = {"rollout": str(roll), "w": 1.0}
    fake.run("update", {"op": "update", "attempts": [spec], "lr": 1e-4, "keep_old": ["t0"]})
    got = fake.run("k3_score", {"op": "k3_score", "attempts": [spec]})
    assert got["ok"] and got["approx_kl"] == 1e-5 and got["missing"] == 0
    again = fake.run("k3_score", {"op": "k3_score", "attempts": [spec]})
    assert not again["ok"] and again["missing"] == 1


def test_background_checkpoint_matches_sync_and_is_a_snapshot(monkeypatch, tmp_path):
    """MFU-1 E (async checkpoint): the background write lands the same files as the synchronous one, from host copies
    taken at the call (later steps do not leak in); the folder appears only complete; load and wait join it."""
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    st = pytest.importorskip("safetensors.torch")
    a = _optim_trainer(monkeypatch, 0.1)
    a.optimizer, a.updates = a.make_optimizer(1e-4), 3
    for _, p in a.params:
        p.grad = torch.ones(4)
    a.optimizer.step()
    a.op_checkpoint({"path": str(tmp_path / "sync")})
    out = a.op_checkpoint({"path": str(tmp_path / "bg"), "background": True, "free_bytes": 1 << 40})
    assert out["background"] and "snapshot_seconds" in out
    with torch.no_grad():  # the next update moves the weights and moments while the write runs
        for _, p in a.params:
            p.add_(1.0)
        a.optimizer.step()
    done = a.op_checkpoint_wait({})
    assert done["pending"] and done["path"] == str(tmp_path / "bg") and not (tmp_path / "bg.partial").exists()
    assert a.op_checkpoint_wait({}) == {"pending": False}
    m1, m2 = st.load_file(str(tmp_path / "sync" / "master.safetensors")), st.load_file(str(tmp_path / "bg" / "master.safetensors"))
    assert set(m1) == set(m2) and all(torch.equal(m1[k], m2[k]) for k in m1)
    o1 = torch.load(str(tmp_path / "sync" / "optimizer.pt"), weights_only=False)
    o2 = torch.load(str(tmp_path / "bg" / "optimizer.pt"), weights_only=False)
    assert o1["updates"] == o2["updates"] == 3 and o1["names"] == o2["names"]
    assert o1["optimizer"]["param_groups"] == o2["optimizer"]["param_groups"]
    for k, s1 in o1["optimizer"]["state"].items():
        assert all(torch.equal(torch.as_tensor(s1[n]), torch.as_tensor(o2["optimizer"]["state"][k][n])) for n in s1)
    b = _optim_trainer(monkeypatch, 0.1)
    b.optimizer, b.updates = b.make_optimizer(1e-4), 0
    b.adapter_norm, b.forget_unaccepted = (lambda: 0.0), (lambda: None)
    assert b.op_load_checkpoint({"path": str(tmp_path / "bg")})["updates"] == 3


def test_background_checkpoint_failure_surfaces_at_the_join(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    pytest.importorskip("safetensors.torch")
    a = _optim_trainer(monkeypatch, 0.1)
    a.optimizer, a.updates = a.make_optimizer(1e-4), 0
    (tmp_path / "file").write_text("x")  # a file where the folder's parent should be
    a.op_checkpoint({"path": str(tmp_path / "file" / "c"), "background": True, "free_bytes": 1 << 40})
    with pytest.raises(RuntimeError, match="background checkpoint"):
        a.op_checkpoint_wait({})
    assert a.op_checkpoint_wait({}) == {"pending": False}


def test_background_checkpoint_refused_without_host_room(monkeypatch, tmp_path):
    """MFU-1 E: a host snapshot that would not fit (free < 1.25 x state + 16 GiB) is written synchronously instead."""
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    pytest.importorskip("safetensors.torch")
    a = _optim_trainer(monkeypatch, 0.1)
    a.optimizer, a.updates = a.make_optimizer(1e-4), 0
    out = a.op_checkpoint({"path": str(tmp_path / "c"), "background": True, "free_bytes": 8 * 2**30})
    assert "background" not in out and out["background_refused"]["free_gib"] == 8.0
    assert (tmp_path / "c" / "master.safetensors").exists() and getattr(a, "ckpt_thread", None) is None
    assert a.op_checkpoint_wait({}) == {"pending": False}


class _FakeOOM(RuntimeError):
    pass


def _oom_trainer(monkeypatch, *, raise_after_step=False):
    monkeypatch.setattr(rl1_trainer, "OOM_ERRORS", (_FakeOOM,))
    t = _optim_trainer(monkeypatch, 0.1)
    t.model = torch.nn.Linear(2, 2)
    t.calls, t.closed = [], []

    def update_once(cmd):
        t.update_stepped = False
        t.calls.append([a.get("window") for a in cmd["attempts"]])
        t.update_close = lambda: t.closed.append(True)
        if len(t.calls) > 1:
            assert all(p.grad is None for _, p in t.params)  # the retry starts from freed gradients
        for _, p in t.params:
            p.grad = torch.ones(4)  # the pass's partial gradients
        if any(a.get("whole") and a.get("window", 0) > 32768 for a in cmd["attempts"]):
            t.update_stepped = raise_after_step
            raise _FakeOOM("CUDA out of memory (stand-in)")
        return {"ok": True, "windows": [{"name": a["name"]} for a in cmd["attempts"]]}

    t.update_once = update_once
    return t


def test_update_oom_retries_once_at_the_fallback_window(monkeypatch):
    """MFU-1 D: an out-of-memory at 48K frees the step and redoes the same update at the fallback window."""
    t = _oom_trainer(monkeypatch)
    attempts = [{"name": "a", "whole": True, "window": 49152, "w": 1.0},
                {"name": "b", "whole": True, "window": 49152, "w": -1.0}, {"name": "c", "window": 8192, "w": 1.0}]
    out = t.op_update({"attempts": attempts, "lr": 1e-5, "oom_fallback_window": 32768})
    assert t.calls == [[49152, 49152, 8192], [32768, 32768, 8192]] and t.closed == [True]
    assert out["ok"] and out["window_fallback"]["why"] == "oom" and out["window_fallback"]["to"] == 32768
    assert out["window_fallback"]["from"] == 49152 and "out of memory" in out["window_fallback"]["error"]
    assert [a["window"] for a in attempts] == [49152, 49152, 8192]  # the caller's specs are not modified


def test_update_oom_is_raised_without_fallback_after_the_step_or_when_accumulating(monkeypatch):
    attempts = [{"name": "a", "whole": True, "window": 49152, "w": 1.0}]
    for cmd, after in (({}, False), ({"oom_fallback_window": 32768, "accumulate": True}, False),
                       ({"oom_fallback_window": 32768}, True)):
        t = _oom_trainer(monkeypatch, raise_after_step=after)
        with pytest.raises(_FakeOOM):
            t.op_update({"attempts": attempts, "lr": 1e-5, **cmd})
        assert len(t.calls) == 1


def test_background_checkpoint_guard_counts_pending_prepin_bytes(monkeypatch, tmp_path):
    """MFU-2 + MFU-1 E: pinned park buffers still being allocated (mg.park_prepin) count against the host snapshot's
    room; with them the same free memory refuses the background write."""
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    a = _optim_trainer(monkeypatch, 0.1)
    a.optimizer, a.updates = a.make_optimizer(1e-4), 3
    need = sum(p.numel() * 4 for _, p in a.params)
    free = int(1.25 * need) + 16 * 2**30 + (1 << 20)
    out = a.op_checkpoint({"path": str(tmp_path / "a"), "background": True, "free_bytes": free})
    assert out.get("background")
    a.op_checkpoint_wait({})
    monkeypatch.setattr(type(a), "prepin_pending_bytes", lambda self: 2 << 20)
    out = a.op_checkpoint({"path": str(tmp_path / "b"), "background": True, "free_bytes": free})
    assert not out.get("background")
