"""the trainer branches a run takes, on the real rl1_trainer (rl1_tiny: the tiny Flash-shaped model,
CPU) through rl_seq.train_step and the serve loop's run_op, with the run's own trainer arguments (v17's, from its run
config; adaptive ckpt on a modelled card): served routing on/off x minibatches 1/2 x epochs 1/2 (clipped against
θ0), the async adapter save, a background checkpoint -> a fresh trainer's load_checkpoint -> the pending update."""
import asyncio
import json
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
from arena import rl_seq, served_route  # noqa: E402
from test_rl_seq import FakeLoop  # noqa: E402

sys.modules.setdefault("served_route", served_route)  # the bundle copies it beside rl1_encode (rl1/)

RUN = ROOT / "launch/rl/runs/rl1-loop-seq-real-v17.json"
GROUPS = {"g1": [{"name": "g1-t0#1", "w": 1.0}, {"name": "g1-t1#1", "w": -0.6}],
          "g2": [{"name": "g2-t0#1", "w": 0.8}],
          "g3": [{"name": "g3-t0#1", "w": -0.4}, {"name": "g3-t1#1", "w": 1.2}],
          "g4": [{"name": "g4-t0#1", "w": 0.5}]}


def run_args():
    """The trainer arguments the v17 run sends (rl_loop.Loop.v7_trainer_args on its config), prefix chunk at the tiny
    model's 64."""
    from arena import rl_loop

    cfg = json.loads(RUN.read_text())["study"]["rl_loop"]
    args = rl_loop.Loop.v7_trainer_args(SimpleNamespace(cfg=cfg, ckpt_budget_gib=None))
    assert args["adaptive_ckpt"] and args["signed"] and args["norm"] == "cap"  # the run's branches, still on
    import rl1_tiny

    return rl1_tiny.cpu_args(args)


def grp(gid):
    return type("G", (), {"gid": gid, "unit": "x"})()


def make_loop(tmp, tiny, mb):
    loop = FakeLoop(tmp)
    loop.cfg["mg"]["redesign"] = {**(loop.cfg["mg"].get("redesign") or {}), "one_call": True, "minibatches": mb}
    loop.trainer = SimpleNamespace(call=tiny.acall)
    args = run_args()
    loop.v7_trainer_args = lambda: args
    loop.v7_prepare = lambda g: ({"gid": g.gid, "set": "x", "tries": 2, "trained": len(GROUPS[g.gid])}, GROUPS[g.gid])
    return loop


def update(loop, save, **kw):
    return asyncio.run(rl_seq.train_step(loop, [(grp(k), [{"label": "a000"}]) for k in GROUPS], 1e-3, save,
                                         background=True, **kw))


def assert_finite(row):
    for k in ("grad_norm", "loss", "entropy", "update_norm"):
        assert row.get(k) is not None and math.isfinite(row[k]), (k, row)


@pytest.mark.parametrize("served", [False, True])
@pytest.mark.parametrize("mb,epochs", [(1, 1), (2, 1), (2, 2)])
@base.isolate
def test_update_branches_through_the_real_trainer(tmp_path, served, mb, epochs, monkeypatch):
    import rl1_memory
    import rl1_tiny

    monkeypatch.setenv("RL1_CPU_BASE_GIB", "200")  # v17's budget 240 with ~40K-token tails: the planner picks n per window
    saved = (dict(rl1_memory.CKPT), )
    try:
        tiny = rl1_tiny.Tiny(tmp_path / "a", served=served)
        loop = make_loop(tmp_path, tiny, mb)
        row = update(loop, tmp_path / "v0001", epochs=epochs)
        assert row is not None and row["trained"] == 6, row
        assert_finite(row)
        if mb * epochs > 1:
            assert len(row["minibatches"]) == mb * epochs and row["ratio_clip"] == "gspo"
            assert all(m["ratio_clip_frac"] is not None for m in row["minibatches"][1:]), row["minibatches"]
            olds = loop.feed.of("seq_old_pass")
            assert len(olds) == (1 if mb > 1 else 0) and all(o["windows"] for o in olds)
        accs = [r for r in loop.feed.of("seq_accumulate") if r.get("one_call")]
        if served:
            assert all(r["served_share"] == 1.0 for r in accs), accs
        assert all(r["encode_errors"] == 0 for r in accs)
        log = [json.loads(x) for x in (tmp_path / "a" / "trainer.log").read_text().splitlines()]
        ckpt_rows = [w["ckpt"] for e in log if e.get("op") == "accumulate" for w in e.get("windows") or [] if "ckpt" in w]
        assert ckpt_rows and all("pred_gib" in c and "card_gib" in c for c in ckpt_rows)  # adaptive ckpt ran its code
        # the async adapter save lands; a background checkpoint; a fresh trainer resumes and trains the pending update
        assert tiny.call("save_join")["ok"] and (tmp_path / "v0001" / "adapter_model.safetensors").is_file()
        assert tiny.call("checkpoint", path=str(tmp_path / "s0001"), background=True, free_bytes=2**50)["ok"]
        assert tiny.call("checkpoint_wait")["ok"] and (tmp_path / "s0001" / "master.safetensors").is_file()
        fresh = rl1_tiny.Tiny(tmp_path / "b", served=served)
        got = fresh.call("load_checkpoint", path=str(tmp_path / "s0001"))
        assert got["ok"] and got["updates"] == tiny.t.updates, got
        for (_, p), (_, q) in zip(tiny.t.params, fresh.t.params):
            assert torch.equal(p, q)
        loop2 = make_loop(tmp_path / "r", fresh, mb)
        row2 = update(loop2, tmp_path / "v0002", epochs=epochs)
        assert row2 is not None and row2["trained"] == 6
        assert_finite(row2)
        assert fresh.call("exit")["ok"] and tiny.call("exit")["ok"]
    finally:
        rl1_memory.CKPT.clear(), rl1_memory.CKPT.update(saved[0])


@base.isolate
def test_ckpt_floor_window_runs_and_is_recorded(tmp_path, monkeypatch):
    """A window that fits the budget only at n=0 (v17 a5's U20 class) runs at the floor on the modelled card; the
    window row records n 0 and a prediction above the budget, the call still trains."""
    import rl1_memory
    import rl1_tiny

    monkeypatch.setenv("RL1_CPU_BASE_GIB", "236")
    monkeypatch.setenv("RL1_CPU_TAIL_X", "512")
    saved = dict(rl1_memory.CKPT)
    try:
        tiny = rl1_tiny.Tiny(tmp_path / "a")
        loop = make_loop(tmp_path, tiny, 1)
        row = update(loop, tmp_path / "v0001")
        assert row is not None and row["trained"] == 6
        log = [json.loads(x) for x in (tmp_path / "a" / "trainer.log").read_text().splitlines()]
        rows = [w["ckpt"] for e in log if e.get("op") == "accumulate" for w in e.get("windows") or [] if "ckpt" in w]
        assert any(c["n"] == 0 and c["pred_gib"] > 240 for c in rows), rows
    finally:
        rl1_memory.CKPT.clear(), rl1_memory.CKPT.update(saved)


def shadow_env(monkeypatch):
    monkeypatch.setenv("ARC_PREFLIGHT_TINY_TRAINER", "1")
    monkeypatch.setenv("ARC_TINY_PYTHON", sys.executable)  # the bundle's: /workspace/axolotl-venv/bin/python


def attempts():
    return [{"name": "g1-t0#1", "w": 1.0}, {"name": "g1-t1#1", "w": -0.6}, {"name": "g2-t0#1", "w": 0.8}]


def test_stand_in_shadow_runs_the_real_update_in_a_child_process(tmp_path, monkeypatch):
    """The stand-in's python has no torch: its tiny shadow is a child on the trainer's python. One 2-step update, its
    checkpoint and a reload go through rl1_trainer there, with the run's own arguments; the child's stdout carries only
    answers though the trainer's imports print."""
    from arena import fake_trainer

    shadow_env(monkeypatch)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    args = run_args()
    fake = fake_trainer.Fake(folder=tmp_path / "t")
    specs = attempts()
    first, rest = specs[:2], specs[2:]
    got = fake.shadow({"op": "accumulate", "attempts": first, "first": True, "keep_old": True, "old_reset": True,
                       "lr": 1e-3, **args})
    assert got["ok"] and got["trained_tokens"] > 0, got
    got = fake.shadow({"op": "old_pass", "attempts": rest, "lr": 1e-3, **args})
    assert got["ok"] and got["old_kept"] == 3, got  # chunk 1's kept by keep_old + chunk 2's
    assert fake.shadow({"op": "step", "lr": 1e-3, **args})["ok"]
    got = fake.shadow({"op": "accumulate", "attempts": rest, "first": True, "ratio_clip": "gspo", "lr": 1e-3, **args})
    assert got["ok"] and got["ratio_clip"]["attached"] == 1 and got["ratio_clip"]["missing"] == 0, got
    got = fake.shadow({"op": "step", "lr": 1e-3, "save": str(tmp_path / "v0001"), **args})
    assert got["ok"] and got["ratio_clip"]["frac"] is not None, got
    path = str(tmp_path / "s0001")
    assert fake.shadow({"op": "checkpoint", "path": path, "background": True, "free_bytes": 2**50})["ok"]
    assert fake.shadow({"op": "checkpoint_wait"})["ok"]
    assert (tmp_path / "t" / "tiny" / "paths" / tmp_path.name / "s0001" / "master.safetensors").is_file()
    assert fake.shadow({"op": "load_checkpoint", "path": path})["updates"] == 2
    assert fake.shadow({"op": "exit"})["ok"] and fake.tiny.proc.wait(30) == 0
    assert fake.shadow({"op": "park"}) is None  # an op it does not take
    tally = json.loads((tmp_path / "tiny_ops.json").read_text())  # what the leg's verdict reads (tiny_trainer_ran)
    assert tally["ok"]["accumulate"] == 2 and tally["ok"]["old_pass"] == 1 and tally["ok"]["step"] == 2
    assert tally["ok"]["checkpoint"] == 1 and not tally["failed"]


def test_shadow_failures_fail_the_command_and_never_hang(tmp_path, monkeypatch):
    from arena import fake_trainer

    shadow_env(monkeypatch)
    fake = fake_trainer.Fake(folder=tmp_path / "t")
    # a resume's load: the checkpoint belongs to an earlier container's child
    got = fake.shadow({"op": "load_checkpoint", "path": str(tmp_path / "gone" / "s0001")})
    assert got["ok"] and got["skipped"]
    assert fake.shadow({"op": "accumulate", "attempts": attempts()[:1], "first": True, "lr": 1e-3,
                        "ratio_clip": "nonsense", **run_args()})["ok"] is False  # the trainer's own refusal
    fake.tiny.proc.kill()
    fake.tiny.proc.wait(30)
    got = fake.shadow({"op": "step", "lr": 1e-3})
    assert got["ok"] is False and "tiny trainer" in got["error"]
    monkeypatch.setattr(fake_trainer, "TINY_TIMEOUT_S", 1)
    silent = tmp_path / "silent"
    silent.write_text("#!/bin/sh\nexec sleep 60\n")  # a child that never answers
    silent.chmod(0o755)
    monkeypatch.setenv("ARC_TINY_PYTHON", str(silent))
    got = fake.shadow({"op": "step", "lr": 1e-3})
    assert got["ok"] is False and "silent" in got["error"] and fake.tiny.proc is None


def test_old_pass_keeps_same_named_windows_of_different_tries_apart(tmp_path, monkeypatch):
    """Two groups from one start share try labels (v17 a10 U23): each rollout keeps its own θ0."""
    from arena import fake_trainer

    shadow_env(monkeypatch)
    fake = fake_trainer.Fake(folder=tmp_path / "t")
    same = [{"rollout": str(tmp_path / g / "a000-x-L1-p0" / "rollout.jsonl"), "w": 1.0} for g in ("p1g1", "p1g2")]
    got = fake.shadow({"op": "old_pass", "attempts": same, "old_reset": True, "lr": 1e-3, **run_args()})
    assert got["ok"] and got["old_windows"] == 2 and got["old_kept"] == 2, got
    fake.shadow({"op": "exit"})
