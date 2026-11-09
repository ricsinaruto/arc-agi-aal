"""Trainer units on CPU: the pinned prefix chunk, token TIS, background checkpoints."""
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
    with pytest.raises(ValueError, match="pinned at 16384"):
        fake.run("update", {"attempts": [], "prefix_chunk": 32768})


def _optim_trainer(monkeypatch):
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


def test_token_tis_counts_tokens_outside_half_to_two():
    from types import SimpleNamespace
    me = SimpleNamespace(args=SimpleNamespace(tis_cap=2.0))
    a = leaf()
    drift = torch.tensor([1.0, 0.0, 5.0, -1.0, 0.1, 0.0, 0.0, 0.0, 0.0])  # trainer - sampler; token 2 invalid
    arrays = {"weights": W, "reference": a.detach() - drift}
    _, tis, signed = rl1_trainer.Trainer.weighted_loss(me, arrays, a, VALID, 1.0)
    signed = signed()  # read after the backward (U2: one host read per window)
    assert signed["tis_out"] == {"tokens": 2, "of": 8} and float(tis[0]) == 2.0


def test_background_checkpoint_matches_sync_and_is_a_snapshot(monkeypatch, tmp_path):
    """MFU-1 E (async checkpoint): the background write lands the same files as the synchronous one, from host copies
    taken at the call (later steps do not leak in); the folder appears only complete; load and wait join it."""
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    st = pytest.importorskip("safetensors.torch")
    a = _optim_trainer(monkeypatch)
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
    b = _optim_trainer(monkeypatch)
    b.optimizer, b.updates = b.make_optimizer(1e-4), 0
    b.adapter_norm = lambda: 0.0
    assert b.op_load_checkpoint({"path": str(tmp_path / "bg")})["updates"] == 3


def test_background_checkpoint_failure_surfaces_at_the_join(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    pytest.importorskip("safetensors.torch")
    a = _optim_trainer(monkeypatch)
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
    a = _optim_trainer(monkeypatch)
    a.optimizer, a.updates = a.make_optimizer(1e-4), 0
    out = a.op_checkpoint({"path": str(tmp_path / "c"), "background": True, "free_bytes": 8 * 2**30})
    assert "background" not in out and out["background_refused"]["free_gib"] == 8.0
    assert (tmp_path / "c" / "master.safetensors").exists() and getattr(a, "ckpt_thread", None) is None
    assert a.op_checkpoint_wait({}) == {"pending": False}


def test_background_checkpoint_guard_counts_pending_prepin_bytes(monkeypatch, tmp_path):
    """MFU-2 + MFU-1 E: pinned park buffers still being allocated (start_prepin) count against the host snapshot's
    room; with them the same free memory refuses the background write."""
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, fused=False, **kw: _adam(params, lr, **kw))
    a = _optim_trainer(monkeypatch)
    a.optimizer, a.updates = a.make_optimizer(1e-4), 3
    need = sum(p.numel() * 4 for _, p in a.params)
    free = int(1.25 * need) + 16 * 2**30 + (1 << 20)
    out = a.op_checkpoint({"path": str(tmp_path / "a"), "background": True, "free_bytes": free})
    assert out.get("background")
    a.op_checkpoint_wait({})
    monkeypatch.setattr(type(a), "prepin_pending_bytes", lambda self: 2 << 20)
    out = a.op_checkpoint({"path": str(tmp_path / "b"), "background": True, "free_bytes": free})
    assert not out.get("background")


def test_step_row_safety_fields_from_the_windows():
    """approx_kl (k3), TIS-capped share and the share of trained tokens with a non-zero advantage were
    absent from the rl_seq step rows; the trainer sums them over the update's windows and the step result carries them."""
    from types import SimpleNamespace
    me = SimpleNamespace(args=SimpleNamespace(tis_cap=2.0))
    a = leaf()
    arrays = {"weights": W, "reference": a.detach() - torch.zeros(9)}
    _, _, signed = rl1_trainer.Trainer.weighted_loss(me, arrays, a, VALID, 1.0)
    sg = signed()
    assert sg["n_valid"] == int(VALID.sum()) and sg["n_up"] + sg["n_down"] == int((W[VALID] != 0).sum())
    t = SimpleNamespace(dist=None)
    for name in ("window_stats_add", "window_stats_take"):
        setattr(t, name, getattr(rl1_trainer.Trainer, name).__get__(t))
    t.window_stats_add({"tokens": 10, "mean": 0.02, "k3": 0.001}, {"signed": {"n_up": 3, "n_down": 2, "n_valid": 10},
                                                                   "tis": {"capped": 0.1}})
    t.window_stats_add({"tokens": 30, "mean": 0.0, "k3": 0.003}, {"signed": {"n_up": 10, "n_down": 0, "n_valid": 30},
                                                                   "tis": {"capped": 0.0}})
    out = t.window_stats_take()
    assert out["k3"] == {"approx_kl": 0.0025, "approx_kl_tok": 0.005, "tokens": 40, "clip_frac": 0.025}
    assert out["adv_nonzero_share"] == 0.375 and out["adv_up_share"] == 0.325 and t.window_stats_take() == {}
    mm = rl1_trainer.Trainer.mismatch(None, {"engine_raw": [-1.0, -2.0]}, {"raw": torch.tensor([-1.0, -1.5])})
    assert mm["k3"] == round(float(torch.exp(torch.tensor(0.5)) - 1 - 0.5) / 2, 6)
