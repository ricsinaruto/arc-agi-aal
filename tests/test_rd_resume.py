"""Resume setup cuts (T4r-v2 26 Oct: 345 s vs 177 s fresh): checkpoint prefetch, park pre-pinning."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/posttrain"))


def test_prefetch_seq_reads_the_checkpoint_the_state_names(tmp_path):
    m = pytest.importorskip("modal_rl1")
    root = tmp_path / "leg1"
    (root / "state/s0001").mkdir(parents=True)
    (root / "state/s0001/master.safetensors").write_bytes(b"m" * 5000)
    (root / "state/s0001/optimizer.pt").write_bytes(b"o" * 70000)
    (root / "harvest/a").mkdir(parents=True)
    (root / "harvest/a/meta.json").write_text("{}")
    # the earlier container's path, as rl_loop.v7_found maps it: the leg's own state/sNNNN
    (root / "seq_state.json").write_text(json.dumps({"checkpoint": "/runs/validation/leg1/state/s0001", "step": 1}))
    m.WEIGHTS_DONE.set()
    identity = {}
    m.prefetch_pending([str(tmp_path / "none"), str(root)], identity)
    row = identity["seq_prefetch"]
    assert row["checkpoint"] == str(root / "state/s0001") and row["files"] == 3
    assert round(row["gib"] * 2**30) == 75002 or row["gib"] == round(75002 / 2**30, 2)
    assert m.read_ranges(root / "state/s0001/optimizer.pt", threads=3, block=4096) == 70000


def test_prepin_plan_predicts_fused_adam_state_and_the_park_uses_it():
    torch = pytest.importorskip("torch")
    rl1_trainer = pytest.importorskip("rl1_trainer")
    a, b = torch.nn.Parameter(torch.zeros(4, 3)), torch.nn.Parameter(torch.zeros(5, dtype=torch.bfloat16))
    fake = SimpleNamespace(params=[("a", a), ("b", b)], optimizer=torch.optim.AdamW([a, b], lr=1e-3), pinned={})
    plan = rl1_trainer.Trainer.prepin_plan(fake, on_card=lambda t: True)
    keys = {k: (s, d) for k, s, d in plan}
    assert keys[("param", "a")] == ((4, 3), torch.float32) and keys[("state", "b", "exp_avg_sq")] == ((5,), torch.bfloat16)
    assert keys[("state", "a", "step")] == ((), torch.float32) and len(plan) == 8
    # after a real step the plan reads the state as it is
    a.grad, b.grad = torch.ones_like(a), torch.ones_like(b)
    fake.optimizer.step()
    real = {k: (s, d) for k, s, d in rl1_trainer.Trainer.prepin_plan(fake, on_card=lambda t: True)}
    assert real[("state", "a", "exp_avg")] == ((4, 3), torch.float32) and ("state", "a", "step") in real
    assert rl1_trainer.Trainer.prepin_plan(fake) == []  # nothing on a card here


def test_start_prepin_threads_and_the_join_reports(monkeypatch):
    torch = pytest.importorskip("torch")
    rl1_trainer = pytest.importorskip("rl1_trainer")
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **k: real_empty(*a, **k))  # no card for pinning
    fake = SimpleNamespace(pinned={("param", "a"): "kept"})
    plan = [(("param", "a"), (2,), torch.float32), (("state", "a", "exp_avg"), (1024,), torch.float32)]
    out = rl1_trainer.Trainer.start_prepin(fake, plan=plan)
    assert out["started"] and out["tensors"] == 1
    done = rl1_trainer.Trainer.prepin_join(fake)
    assert done["error"] is None and done["done_bytes"] == done["bytes"] == 4096
    assert fake.pinned[("param", "a")] == "kept" and tuple(fake.pinned[("state", "a", "exp_avg")].shape) == (1024,)
    assert rl1_trainer.Trainer.prepin_pending_bytes(fake) == 0
    assert rl1_trainer.Trainer.start_prepin(fake, plan=plan) == {"already": True}


def test_load_digest_reaches_the_trainer_env(tmp_path, monkeypatch):
    from arena.rl_loop import trainer_command

    cfg = {"trainer_script": "rl1/rl1_trainer.py", "trainer_python": "py", "model_dir": "/m", "rank": 16, "lr": 1e-5,
           "window": 1024, "optimizer": "adamw32", "loop_lora": False, "tis_cap": 0, "path_chunk": 16384, "mg": {}}
    monkeypatch.delenv("ARC_PREFLIGHT_FAKE_TRAINER", raising=False)
    assert "RL1_LOAD_DIGEST" not in trainer_command(cfg, tmp_path, tmp_path / "w")[1]
    cfg["mg"] = {"load_digest": True}
    assert trainer_command(cfg, tmp_path, tmp_path / "w")[1]["RL1_LOAD_DIGEST"] == "1"


def test_a_leg_preempted_before_its_first_checkpoint_keeps_the_chain(tmp_path):
    m = pytest.importorskip("modal_rl1")
    import json

    run = tmp_path / "run"
    run.mkdir()
    (run / "seq_play.json").write_text(json.dumps({"step": 28, "version": 27}))  # the resumed leg: no state of its own
    assert m.dead_step(run) == 0 and m.chain_step(run) == 0
    old = tmp_path / "run-lost-100"
    old.mkdir()
    (old / "seq_state.json").write_text(json.dumps({"step": 27}))
    (tmp_path / "run-lost-50").mkdir()  # no loop state: not in the chain
    (tmp_path / "run2-lost-200").mkdir()
    (tmp_path / "run2-lost-200" / "seq_state.json").write_text(json.dumps({"step": 99}))  # another run's
    assert m.lost_chain(run) == [old] and m.chain_step(run) == 27
    (run / "seq_state.json").write_text(json.dumps({"step": 28}))
    assert m.chain_step(run) == 28


def test_a_dead_seq_attempt_is_resumed_and_its_step_read(tmp_path):
    m = pytest.importorskip("modal_rl1")
    import json

    assert not m.lost_state(None) and not m.lost_state(tmp_path) and m.dead_step(tmp_path) == 0
    (tmp_path / "seq_pending.json").write_text(json.dumps({"step": 1, "groups": []}))
    assert m.lost_state(tmp_path) and m.dead_step(tmp_path) == 0  # step 0's update: no checkpoint past 0
    (tmp_path / "seq_state.json").write_text(json.dumps({"step": 3}))
    assert m.lost_state(tmp_path) and m.dead_step(tmp_path) == 3
    (tmp_path / "modal-identity.json").write_text("{}")  # a finished attempt is not a lost one
    assert not m.lost_state(tmp_path)
