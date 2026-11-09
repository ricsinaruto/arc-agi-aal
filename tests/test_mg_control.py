"""mg5e controller paths: trainer KL references, target-KL early stopping, rollback / serve, rule (f), resume state."""
import asyncio
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

from arena import rl_loop

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import rl1_trainer  # noqa: E402


class Feed:
    def __init__(self):
        self.events = []

    def emit(self, kind, **data):
        self.events.append((kind, data))


class FakeTrainer:
    """Answers ops from a script: ``probes`` is a list of per-update-step kl_previous / kl_base values."""

    def __init__(self, steps=()):
        self.calls, self.steps = [], list(steps)

    async def call(self, op, timeout=3600, **payload):
        self.calls.append((op, payload))
        if op == "update":
            out = {"ok": True, "windows": [], "loss": 0.0, "grad_norm": 0.01, "update_norm": 0.5, "trained_tokens": 10}
            if payload.get("probes"):
                kl_prev, kl_base = self.steps.pop(0)
                out["probes"] = {p["name"]: {"kl_previous": kl_prev, "kl_base": kl_base} for p in payload["probes"]}
            elif self.steps and payload.get("probes") is None:
                pass
            return out
        return {"ok": True}

    async def must(self, op, timeout=3600, **payload):
        return await self.call(op, timeout=timeout, **payload)

    def ops(self):
        return [op for op, _ in self.calls]


class FakeServer:
    def __init__(self):
        self.loads = []

    async def load(self, name, path):
        self.loads.append((name, str(path)))
        return 0.1


def make_loop(tmp_path, mg=None, lr=5e-5, steps=()):
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = rl_loop.settings({"mode": "loop"})
    loop.cfg.update(lr=lr, l2_kl_stop=0.15, update_timeout=60, clip=1.0)
    loop.cfg["mg"] = {"kl_control": "target", "kl_target": 0.01, "kl_floor": 0.018, "probe_every": 2, "lr_max": 6e-5,
                      "rollback": True, "checkpoint": True, "kl_trips_stop": 2, "step_rollbacks_stop": 3, **(mg or {})}
    loop.feed, loop.trainer, loop.server = Feed(), FakeTrainer(steps), FakeServer()
    loop.adapters = tmp_path / "adapters"
    loop.serving_adapter = None
    loop.keep_checkpoint = "r00"
    loop.mg_probes = [{"name": "p1", "rollout": "x"}, {"name": "p2", "rollout": "y"}]
    return loop


# --- trainer references -------------------------------------------------------------------------------------------

def toy_trainer(monkeypatch):
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.references, t.theta = {}, 0.0
    t.attempt_window = lambda spec: {"name": spec["name"]}
    t.mismatch = lambda made, out: {}

    def window_pass(made, vocab=None):
        n = len(vocab or [])
        return {"valid": np.ones(3, bool), "actor": np.full(3, -t.theta), "prefix_seconds": 0.0, "forward_seconds": 0.0,
                "dist": {"ids": None, "logp": t.theta, "at": [t.theta] * n}}

    t.window_pass = window_pass
    monkeypatch.setattr(rl1_trainer, "binned_kl", lambda p, q: (p - q) ** 2)
    return t


def test_trainer_hold_keeps_previous_until_accept(monkeypatch):
    t, specs = toy_trainer(monkeypatch), [{"name": "p"}]
    t.score_windows(specs, label="base")
    t.theta = 1.0
    assert t.score_windows(specs, hold=True)["p"]["kl_previous"] == 1.0
    t.theta = 2.0
    row = t.score_windows(specs, hold=True)["p"]
    assert row["kl_previous"] == 4.0 and row["kl_base"] == 4.0  # still against the served (base) state
    t.op_accept_reference({})
    t.theta = 2.5
    assert t.score_windows(specs, hold=True)["p"]["kl_previous"] == 0.25


def test_trainer_rollback_cannot_accept_rejected_state(monkeypatch):
    t, specs = toy_trainer(monkeypatch), [{"name": "p"}]
    t.score_windows(specs, label="base")
    t.theta = 3.0
    t.score_windows(specs, hold=True)  # a rejected update's probe
    t.forget_unaccepted()  # what load_checkpoint does
    with pytest.raises(ValueError):
        t.op_accept_reference({})
    t.theta = 0.1  # the next update from the restored state compares with the served state, not the rejected one
    assert abs(t.score_windows(specs, hold=True)["p"]["kl_previous"] - 0.01) < 1e-12


def test_trainer_legacy_previous_moves_every_score(monkeypatch):
    t, specs = toy_trainer(monkeypatch), [{"name": "p"}]
    t.score_windows(specs, label="base")
    t.theta = 1.0
    t.score_windows(specs)
    t.theta = 1.5
    assert t.score_windows(specs)["p"]["kl_previous"] == 0.25


# --- early stopping inside an update ------------------------------------------------------------------------------

def early(loop):
    return loop.mg_early()


def run_update(loop, n_batches):
    batches = [[{"name": f"t{i}", "w": 1.0}] for i in range(n_batches)]
    return asyncio.run(loop.dosed_update(2, None, None, None, loop.mg_probes, batches=batches, signed=False,
                                         early=early(loop), cmd_extra={"hold_previous": True}))


def test_early_full_update_probes_every_second_step_and_saves(tmp_path):
    loop = make_loop(tmp_path, steps=[(0.020, 0.02), (0.022, 0.02), (0.023, 0.02), (0.024, 0.024)])
    merged, path = run_update(loop, 8)
    probed = [bool(p.get("probes")) for op, p in loop.trainer.calls if op == "update"]
    assert probed == [False, True, False, True, False, True, False, True]
    assert merged["early"]["action"] == "full" and merged["early"]["steps"] == 8
    assert loop.trainer.ops()[-1] == "save" and all(p.get("save") is None for op, p in loop.trainer.calls if op == "update")
    assert all(p.get("hold_previous") for op, p in loop.trainer.calls if op == "update")


def test_early_stop_keeps_steps_and_saves(tmp_path):
    loop = make_loop(tmp_path, steps=[(0.025, 0.02), (0.036, 0.03)])  # excess 0.018 > 0.015 at step 4
    merged, _ = run_update(loop, 8)
    assert merged["early"]["action"] == "stop" and merged["early"]["steps"] == 4
    assert loop.trainer.ops().count("update") == 4 and loop.trainer.ops()[-1] == "save"


def test_early_rollback_saves_nothing(tmp_path):
    loop = make_loop(tmp_path, steps=[(0.055, 0.05)])  # excess 0.037 > 0.03 at step 2
    merged, _ = run_update(loop, 8)
    assert merged["early"]["action"] == "rollback" and "save" not in loop.trainer.ops()


def test_early_needs_kl_previous(tmp_path):
    loop = make_loop(tmp_path, steps=[(None, 0.02)])
    with pytest.raises(RuntimeError):
        run_update(loop, 2)


# --- lr control and the serve / rollback decision -----------------------------------------------------------------

def test_lr_control_paths(tmp_path):
    loop = make_loop(tmp_path, lr=4e-5)
    row = {}
    assert loop.mg_kl_target_control(row, {"early": {"action": "rollback", "excess": 0.04}}) == "rollback"
    assert loop.cfg["lr"] == 2e-5
    loop.mg_kl_target_control({}, {"early": {"action": "stop", "excess": 0.02, "steps": 4}})
    assert math.isclose(loop.cfg["lr"], 2e-5 / 1.5)
    loop.cfg["lr"] = 4e-5
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.003}})
    assert math.isclose(loop.cfg["lr"], 6e-5)  # x1.5
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.003}})
    assert math.isclose(loop.cfg["lr"], 6e-5)  # capped at lr_max
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.009}})
    assert math.isclose(loop.cfg["lr"], 6e-5)  # inside the band: unchanged
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.016}})  # last step over 1.5 x target
    assert math.isclose(loop.cfg["lr"], 4e-5)



def test_lr_dose_rule(tmp_path):
    """mg.lr_dose: a full update's next lr aims at the target under KL ~ lr^2, within x[1/3, 3] and lr_max."""
    loop = make_loop(tmp_path, mg={"lr_dose": True, "kl_target": 0.08, "lr_max": 2e-4}, lr=6e-5)
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.012}})
    assert math.isclose(loop.cfg["lr"], 6e-5 * math.sqrt(0.08 / 0.012))
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.001}})  # x3 at most, then lr_max
    assert math.isclose(loop.cfg["lr"], 2e-4)
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.079}})  # within 5%: unchanged
    assert math.isclose(loop.cfg["lr"], 2e-4)
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 1.0}})  # /3 at most
    assert math.isclose(loop.cfg["lr"], 2e-4 / 3)
    assert loop.mg_kl_target_control({}, {"early": {"action": "rollback", "excess": 0.5}}) == "rollback"
    assert math.isclose(loop.cfg["lr"], 1e-4 / 3)  # a rollback still halves


def test_resume_dose_reads_the_source_feed(tmp_path):
    rl = tmp_path / "src" / "arena" / "trials" / "rl-loop" / "rl"
    rl.mkdir(parents=True)
    rows = [{"kind": "round", "round": 1, "kl_step": {"action": "full", "excess": 0.012, "lr": 6e-5}},
            {"kind": "round", "round": 2, "kl_step": {"action": "full", "excess": 0.027, "lr": 9e-5}},
            {"kind": "round", "round": 3, "kl_step": {"action": "rollback", "excess": 0.4, "lr": 1.35e-4}},
            {"kind": "update_step", "round": 3}]
    (rl / "feed.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    loop = make_loop(tmp_path, mg={"lr_dose": True, "kl_target": 0.08, "lr_max": 2e-4}, lr=1.35e-4)
    last = loop.mg_source_kl(tmp_path / "src")
    assert last == {"round": 2, "lr": 9e-5, "excess": 0.027, "previous": {"round": 1, "lr": 6e-5, "excess": 0.012}}
    loop.mg_dose_lr({}, 0.08, last["excess"], last["lr"], "resume")
    assert math.isclose(loop.cfg["lr"], 9e-5 * math.sqrt(0.08 / 0.027))
    assert loop.mg_source_kl(tmp_path / "missing") is None
    assert loop.mg_source_kl(tmp_path / "src", upto=1)["round"] == 1  # a rollback restart from r01 ignores U2
    # a measured excess at or below the floor (preflight pf4: -0.011) is the clamp's limit, x3 within lr_max
    loop.mg_dose_lr({}, 0.08, -0.011, 6e-5, "resume")
    assert math.isclose(loop.cfg["lr"], 1.8e-4)
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.0}})
    assert math.isclose(loop.cfg["lr"], 2e-4)
    loop.mg_dose_lr({}, 0.08, None, 6e-5, "resume")  # unknown: unchanged
    assert math.isclose(loop.cfg["lr"], 2e-4)
    # the resume path itself: the feed row (pf4 c7d: a duplicate lr keyword crashed it)
    loop.cfg["lr"] = 1.35e-4
    row = loop.mg_resume_dose({"resume": str(tmp_path / "src"), "kl_target": 0.08})
    assert row == {"source_round": 2, "source_lr": 9e-5, "source_excess": 0.027, "lr": loop.cfg["lr"]}
    assert math.isclose(loop.cfg["lr"], 9e-5 * math.sqrt(0.08 / 0.027))
    assert loop.feed.events[-1][0] == "resume_dose"
    rows.append({"kind": "round", "round": 4, "kl_step": {"action": "full", "excess": -0.011, "lr": 6e-5}})
    (rl / "feed.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert loop.mg_resume_dose({"resume": str(tmp_path / "src"), "kl_target": 0.08})["lr"] == 1.8e-4
    assert loop.mg_resume_dose({"resume": str(tmp_path / "missing"), "kl_target": 0.08}) is None


def update_row(action, excess, kl_base=0.02, ok=True):
    return {"ok": ok, "early": {"action": action, "excess": excess, "steps": 8},
            "probes": {"p1": {"kl_base": kl_base, "kl_previous": 0.018 + excess}}}


def test_serve_accepts_reference_and_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = make_loop(tmp_path)
    stop, served = asyncio.run(loop.mg_after_update(3, {}, update_row("full", 0.008), "/a/r03"))
    assert (stop, served) == (None, True)
    assert loop.trainer.ops() == ["accept_reference"] and loop.server.loads == [("policy", "/a/r03")]
    assert loop.mg_accepted == 1 and loop.keep_checkpoint == "r03" and loop.serving_adapter == "/a/r03"


def test_step_rollbacks_do_not_count_as_guard_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = make_loop(tmp_path)
    for n in range(3):
        row = {}
        stop, served = asyncio.run(loop.mg_after_update(2 + n, row, update_row("rollback", 0.04), "/a/x"))
        assert not served and getattr(loop, "kl_trips", 0) == 0
        assert row["kl_guard"]["rollback"] is not None
    assert stop == "rule_c_step_rollbacks" and loop.mg_step_rollbacks == 3
    assert "accept_reference" not in loop.trainer.ops() and loop.server.loads == []
    assert [p["path"] for op, p in loop.trainer.calls if op == "load_checkpoint"] == [str(tmp_path / "state" / "r00")] * 3


def test_guard_trip_counts_and_second_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = make_loop(tmp_path)
    stop, served = asyncio.run(loop.mg_after_update(2, {}, update_row("full", 0.008, kl_base=0.2), "/a/r02"))
    assert (stop, served) == (None, False) and loop.kl_trips == 1 and loop.cfg["lr"] == 2.5e-5
    stop, served = asyncio.run(loop.mg_after_update(3, {}, update_row("full", 0.008, kl_base=0.16), "/a/r03"))
    assert stop == "rule_c_kl_guard_twice" and loop.kl_trips == 2


def test_legacy_rollback_rescores_the_reference(tmp_path, monkeypatch):
    """mg5d's bug: after a rollback the KL-to-previous reference stayed on the rejected state."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = make_loop(tmp_path, mg={"kl_control": None, "kl_step": 0.01})
    row = {}
    stop, served = asyncio.run(loop.mg_after_update(2, row, {"ok": True, "probes": {
        "p1": {"kl_base": 0.05, "kl_previous": 0.06}}}, "/a/r02"))
    assert not served and loop.trainer.ops() == ["load_checkpoint", "score"]


def test_rollback_without_checkpoint_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = make_loop(tmp_path)
    loop.keep_checkpoint = None
    with pytest.raises(RuntimeError):
        asyncio.run(loop.mg_after_update(2, {}, update_row("rollback", 0.04), "/a/r02"))


def test_failed_update_stops(tmp_path):
    loop = make_loop(tmp_path)
    stop, served = asyncio.run(loop.mg_after_update(2, {}, {"ok": False, "stop": "nan_or_inf"}, "/a/r02"))
    assert (stop, served) == ("nan_or_inf", False)


# --- rule (f) and the held-out reference --------------------------------------------------------------------------

def held(tokens):
    return [{"outcome": "cleared", "level_rows": [{"tokens": t, "actions": 10, "score": 100}], "completion_tokens": t}
            for t in tokens]


def test_shortening_uses_every_base_round(tmp_path):
    loop = make_loop(tmp_path)
    loop.mg_base_rounds = [1, 2, 3]
    loop.mg_fixed = {1: {"a": held([100] * 4)}, 2: {"a": held([200] * 4)}, 3: {"a": held([300] * 4)},
                     5: {"a": held([100] * 4)}}
    assert loop.mg_shortening(5)["a"]["base_median"] == 200.0 and loop.mg_shortening(5)["a"]["ratio"] == 2.0


def rule_f(loop, ratios, played_by=7):
    row = {"shortening": {f"l{i}": {"ratio": r} for i, r in enumerate(ratios)}}
    return loop.mg_rule_f(row, played_by), row


def test_rule_f_counts_accepted_updates_not_rounds(tmp_path):
    loop = make_loop(tmp_path, mg={"rule_f_updates": 6, "rule_f_ratio": 1.5, "rule_f_pooled": 1.15})
    loop.mg_accepted = 5  # round 7 after one rollback: 5 accepted updates, played_by - 1 = 6
    assert rule_f(loop, [1.0] * 6)[0] is None
    loop.mg_accepted = 6
    stop, row = rule_f(loop, [1.0, 1.1, 0.9, 1.2, 1.05, 1.0])
    assert stop == "rule_f_no_shortening" and row["rule_f"]["updates"] == 6
    assert rule_f(loop, [1.0] * 6)[0] is None  # evaluated once


def test_rule_f_pooled_and_single_level(tmp_path):
    loop = make_loop(tmp_path, mg={"rule_f_updates": 6, "rule_f_ratio": 1.5, "rule_f_pooled": 1.15})
    loop.mg_accepted = 6
    assert rule_f(loop, [1.2, 1.2, 1.1, 1.2, 1.15, 1.1])[0] is None  # geo mean ~1.16
    loop.mg_rule_f_done = False
    assert rule_f(loop, [1.6, 0.9, 1.0, 1.0, 1.0, 0.9])[0] is None  # one level at 1.6x


# --- replay ---------------------------------------------------------------------------------------------------------

def test_plumbing_timeout_while_replaying():
    from arena.level_bench import LevelBenchAgent

    class O:
        bench_restore = {"request": 3}
        bench_restore_state = None
        bench_script = None
        bench_script_end = None

    assert LevelBenchAgent.replaying(O())
    O.bench_restore_state = {"ok": True}
    assert not LevelBenchAgent.replaying(O())


def test_stratified_steps_hold_each_level_once(tmp_path):
    loop = make_loop(tmp_path, mg={"stratify": True, "max_groups": 48, "balance": False, "checkpoint": False})
    loop.cfg.update(l2_starts_per_step=6, lvl_window=32768, path_chunk=16384, l2_norm="cap")
    loop.cfg["mg"]["cap"] = 32000
    counts = {"A": 8, "B": 6, "C": 7, "D": 7, "E": 8, "F": 8}
    loop.mg_screen = {"sets": {n: {"metrics": {"median_to_clear": 1000}} for n in counts}}
    loop.cfg["mg_sets"] = {n: {"cap": 4000} for n in counts}
    kept = [(tmp_path, {"label": f"{n}{g}-t{t}"}, 0.5 if t % 2 else -0.5, f"{n}-g{g}", n)
            for n, k in counts.items() for g in range(k) for t in range(6)]
    seen = {}

    async def fake_dosed(number, *a, batches=None, **k):
        seen["batches"] = batches
        return {"ok": True, "early": {"action": "full"}}, tmp_path / "r02"

    loop.dosed_update = fake_dosed
    asyncio.run(loop.mg_update(2, kept, loop.mg_probes))
    steps = seen["batches"]
    assert len(steps) == 8 and sum(len(s) for s in steps) == 6 * sum(counts.values())
    for step in steps:
        per_level = {}
        for spec in step:
            per_level.setdefault(spec["set"], set()).add(spec["share"])
        assert all(len(v) == 1 for v in per_level.values()) and len(per_level) >= 5


def test_early_stop_on_forecast(tmp_path):
    loop = make_loop(tmp_path, steps=[(0.028, 0.02)])  # excess 0.010 at step 2, forecast 0.040 at step 4 > 0.03
    merged, _ = run_update(loop, 8)
    assert merged["early"]["action"] == "stop" and merged["early"]["steps"] == 2 and merged["early"]["predicted"] == 0.04


# --- mg5f seed ------------------------------------------------------------------------------------------------------

SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 16384}
STUDY = {"temperature": 0.7, "reasoning_effort": "xhigh", "completion_cap": 16384,
         "level_bench": {"window": 130816, "rollout": {"top_logprobs": 20, "token_ids": True}}}


def screen_set(root, name, starts, tries, cap, head=None, sampling=None):
    """A screen bench folder: ``tries`` {start index: [(clear tokens or None, spent tokens)]}."""
    folder = root / f"screen-{name}"
    rows = []
    for i, outs in tries.items():
        att, req = starts[i]
        tag = rl_loop.Loop.mg_tag({"level": 2}, att, req)
        for p, (clear, spent) in enumerate(outs):
            label = f"a{i:02d}{p}-p{p}"
            (folder / "attempts" / label).mkdir(parents=True)
            samp = (sampling or {}).get((i, p), SAMPLING)
            (folder / "attempts" / label / "rollout.jsonl").write_text(json.dumps({"sampling": samp}) + "\n")
            rows.append({"label": label, "point": f"g-game-{tag}", "pass": p, "restore": {"ok": True}, "token_budget": cap,
                         "outcome": "cleared" if clear else "budget", "level": 2, "completion_tokens": spent,
                         "level_rows": [{"tokens": clear, "actions": 5}] if clear else []})
    h = {"temperature": 0.7, "reasoning_effort": "xhigh", "completion_cap": 16384, "window": 130816,
         "settings": {"sampling": None, "rollout": {"top_logprobs": 20, "token_ids": True}}, **(head or {})}
    (folder / "level-bench.json").write_text(json.dumps({**h, "attempts": rows}))
    return {"folder": str(folder), "cap": cap, "starts": [list(s) for s in starts]}


def seed_loop(tmp_path, sets, screen_sets):
    loop = make_loop(tmp_path, mg={"cap": 32000, "group_size": 2, "select": "all"})
    loop.study = STUDY
    loop.cfg.update(l2_per_side=12, l2_reward="tokens")
    loop.cfg["mg_sets"] = {s["name"]: s for s in sets}
    loop.mg_screen = {"sets": screen_sets}
    return loop


def test_seed_keeps_matching_sets_and_excludes_held_out(tmp_path):
    from arena import rl_single  # noqa: F401  (Loop's mixin)

    st = [("/tmp/arena/x/a", 3), ("/tmp/arena/x/b", 4), ("/tmp/arena/x/c", 5)]
    sets = [{"name": "s2", "game": "sb26", "level": 2, "role": "train", "cap": 32000},
            {"name": "t4", "game": "sb26", "level": 2, "role": "test", "cap": 16000}]
    scr = {"s2": screen_set(tmp_path, "s2", st, {0: [(10000, 10000), (20000, 20000)], 1: [(9000, 9000), (None, 32000)],
                                                 2: [(5000, 5000), (6000, 6000)]}, 32000),
           "t4": screen_set(tmp_path, "t4", st, {0: [(1000, 1000), (2000, 2000)]}, 16000)}
    loop = seed_loop(tmp_path, sets, scr)
    pool = {"s2": [tuple(s) for s in st[:2]]}
    kept, info = loop.mg_seed(pool, {"s2": st[2]}, ["t4"])
    labels = {k[1]["label"] for k in kept}
    assert labels == {"a000-p0", "a011-p1", "a010-p0", "a001-p1"}  # start c (held out) and the test set never
    assert all(k[-1] == "s2" for k in kept) and "t4" not in info
    assert info["s2"]["screen_groups"] == 2 and info["s2"]["exact_groups"] == 2


def test_seed_drops_inexact_groups_and_mismatched_benches(tmp_path):
    st = [("/tmp/arena/x/a", 3), ("/tmp/arena/x/b", 4), ("/tmp/arena/x/c", 5)]
    sets = [{"name": "s3", "game": "sb26", "level": 2, "role": "train", "cap": 8000},
            {"name": "s9", "game": "sb26", "level": 2, "role": "train", "cap": 16000}]
    # s3 played at 16K: a clear within 8K or a try ended within 8K is exact; a budget fail at 16K is not
    scr = {"s3": screen_set(tmp_path, "s3", st, {0: [(4000, 4000), (None, 16000)], 1: [(7000, 7000), (3000, 3000)],
                                                 2: [(5000, 5000), (None, 6000)]}, 16000, sampling={(2, 1): {**SAMPLING, "top_k": 0}}),
           "s9": screen_set(tmp_path, "s9", st, {0: [(4000, 4000), (5000, 5000)]}, 16000, head={"reasoning_effort": "high"})}
    loop = seed_loop(tmp_path, sets, scr)
    kept, info = loop.mg_seed({"s3": [tuple(s) for s in st], "s9": [tuple(s) for s in st]}, {}, [])
    assert {k[1]["label"] for k in kept} == {"a010-p0", "a011-p1"}  # start a: 16K fail; start c: a wrong sampler
    assert info["s3"]["screen_groups"] == 3
    assert info["s3"]["exact_groups"] == 1 and info["s3"]["bad_sampling"] == 1
    assert "reasoning_effort" in info["s9"]["dropped"]


def test_seed_same_budget_fail_overshoot_is_exact(tmp_path):
    st = [("/tmp/arena/x/a", 3), ("/tmp/arena/x/b", 4)]
    sets = [{"name": "s2", "game": "sb26", "level": 2, "role": "train", "cap": 32000}]
    scr = {"s2": screen_set(tmp_path, "s2", st, {0: [(25000, 25000), (None, 32030)], 1: [(None, 37000), (None, 34000)]}, 32000)}
    loop = seed_loop(tmp_path, sets, scr)
    kept, info = loop.mg_seed({"s2": [tuple(s) for s in st]}, {}, [])
    assert {k[1]["label"] for k in kept} == {"a000-p0", "a001-p1"} and info["s2"]["exact_groups"] == 2


def test_seed_from_an_earlier_loops_base_round(tmp_path):
    mapped = [("/runs/v/s/a", 3), ("/runs/v/s/b", 4), ("/runs/v/s/c", 5)]
    raw = [("/tmp/arena/a", 3), ("/tmp/arena/b", 4), ("/tmp/arena/c", 5)]
    sets = [{"name": "s2", "game": "sb26", "level": 2, "role": "train", "cap": 24000}]
    root = tmp_path / "old"
    root.mkdir()
    # the loop's bench rows carry the mapped starts' tags
    entry = screen_set(root, "s2", mapped, {0: [(9000, 9000), (None, 24500)], 1: [(5000, 5000), (7000, 7000)],
                                            2: [(4000, 4000), (6000, 6000)]}, 24000)
    (root / "screen-s2").rename(root / "r02-s2")
    scr = {"s2": {"folder": "/nowhere", "cap": 24000, "starts": [list(x) for x in mapped],
                  "raw_starts": [list(x) for x in raw]}}
    loop = seed_loop(tmp_path, sets, scr)
    loop.cfg["mg"]["seed_from"] = {"root": str(root), "round": 2}
    (root / "feed.jsonl").write_text(json.dumps({"kind": "round", "round": 2, "played_by": 0}) + "\n")
    kept, info = loop.mg_seed({"s2": [tuple(x) for x in mapped[:2]]}, {"s2": mapped[2]}, [])
    assert {k[1]["label"] for k in kept} == {"a000-p0", "a001-p1", "a010-p0", "a011-p1"}  # start c held out
    assert all(str(k[0]).endswith("r02-s2") for k in kept) and info["s2"]["folder"] == "r02-s2"
    (root / "feed.jsonl").write_text(json.dumps({"kind": "round", "round": 2, "played_by": 2}) + "\n")
    kept, info = loop.mg_seed({"s2": [tuple(x) for x in mapped[:2]]}, {}, [])
    assert not kept and "not played by the base" in info["s2"]["dropped"]


def test_base_tpr_uses_base_played_rounds_only(tmp_path):
    loop = make_loop(tmp_path, mg={"length_guard": 1.15})
    loop.cfg["mg_sets"] = {"s": {"role": "train"}}
    loop.mg_screen_base = lambda numbers: []
    rows = lambda tok: [{"completion_tokens": tok, "admitted_requests": 10, "outcome": "budget"}]
    row1, row2 = {}, {}
    loop.mg_rules(1, {"s": (None, rows(1000), None)}, row1, 0)
    loop.mg_rules(2, {"s": (None, rows(5000), None)}, row2, 1)  # U1 seeded: round 2 is a policy round
    assert row2["tokens_per_request"]["base"] == 100.0 and row2["tokens_per_request"]["over"] is True


def test_early_success_needs_two_rounds_updates_and_clear_rate(tmp_path):
    loop = make_loop(tmp_path, mg={"success_ratio": 1.3, "success_rounds": 2, "success_updates": 3, "success_clear_drop": 0.05})
    row = lambda r, clears="15/16": {"shortening": {f"l{i}": {"ratio": r, "clears": clears, "base_clears": "30/32"} for i in range(3)}}
    loop.mg_accepted = 2
    assert loop.mg_success(row(1.5), 3) is None and loop.mg_success_streak == 0  # too few updates
    loop.mg_accepted = 3
    assert loop.mg_success(row(1.5), 4) is None and loop.mg_success_streak == 1
    assert loop.mg_success(row(1.5, "12/16"), 5) is None and loop.mg_success_streak == 0  # clear rate fell
    assert loop.mg_success(row(1.4), 6) is None
    r = row(1.35)
    assert loop.mg_success(r, 7) == "success_shortening" and r["success"]["streak"] == 2
    assert loop.mg_success(row(1.2), 8) is None and loop.mg_success_streak == 0


# --- paired re-eval -------------------------------------------------------------------------------------------------

def eval_loop(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, mg={"cap": 32000, "active_tries": 8, "pick_seed": 0,
                                   "sets": [{"name": "a", "game": "g", "level": 2, "role": "train", "cap": 8000},
                                            {"name": "f", "game": "g", "level": 1, "role": "train", "cap": 16000, "fresh": True},
                                            {"name": "t", "game": "g", "level": 3, "role": "test", "cap": 16000}]})
    loop.cfg["mg_sets"] = {s["name"]: s for s in loop.cfg["mg"]["sets"]}
    s1, s2 = tmp_path / "s1.json", tmp_path / "s2.json"
    s1.write_text(json.dumps({"recorded": {"g": "/tmp/arena/rec-g"}, "spec": {"pick_seed": 1, "sets": [
        {"name": "a", "starts": 1, "sources": ["record:g"]}]}, "sets": {"a": {"starts": [["/tmp/arena/rec-g/attempts/x0", 3]]}}}))
    s2.write_text(json.dumps({"recorded": {}, "spec": {"pick_seed": 2, "sets": [
        {"name": "a", "starts": 1, "sources": ["/runs/src"]}, {"name": "t", "starts": 1, "sources": ["/runs/src"]}]},
        "sets": {"a": {"starts": [["/runs/src/attempts/y0", 4]]}, "t": {"starts": [["/runs/src/attempts/t0", 9]]}}}))
    loop.cfg["mg"].update(screen=str(s1), screen_root="/runs/one", screens=[{"screen": str(s2), "root": "/runs/two"}])
    calls = []

    def fake_starts(spec, srcs):
        calls.append((spec["name"], srcs, loop.cfg["mg"]["pick_seed"], spec["starts"]))
        if spec["name"] == "t":
            return [("/runs/src/attempts/t0", 9), ("/runs/src/attempts/t1", 2)], 2
        return [("/runs/src/attempts/y0", 4), ("/runs/one/rec-g/attempts/x0", 3), ("/runs/src/attempts/z1", 5),
                ("/runs/src/attempts/z2", 6), ("/runs/src/attempts/z3", 7)], 5

    loop.mg_starts = fake_starts
    return loop, calls


def test_fresh_starts_skip_every_screen_start(tmp_path, monkeypatch):
    loop, calls = eval_loop(tmp_path, monkeypatch)
    out, info = loop.mg_fresh_starts(2)
    assert out["a"] == [("/runs/src/attempts/z1", 5), ("/runs/src/attempts/z2", 6)]  # both screens' starts out
    assert out["t"] == [("/runs/src/attempts/t1", 2)]
    assert "f" not in out  # fresh level-1 sets have no restored starts
    assert ("a", ["/runs/src"], 2, 10 ** 9) in calls  # the later screen's spec and its pick seed
    assert info["a"]["screen_prefix_match"] and info["a"]["screen_used"] == 2 and info["a"]["fresh"] == 3
    assert loop.cfg["mg"]["pick_seed"] == 0  # restored


def test_eval_plays_both_arms_on_the_same_starts(tmp_path, monkeypatch):
    loop, _ = eval_loop(tmp_path, monkeypatch)
    loop.cfg["mg"]["eval"] = {"adapter": "/runs/r04", "base_adapter": "/runs/zero", "starts": 2, "tries": 3,
                              "fresh_tries": 5, "seed": 9300}
    loop.output = tmp_path / "out"
    (loop.output / "rl").mkdir(parents=True)
    played = []

    async def fake_play(name, spec, starts, tries, *, model, seed, concurrency=None):
        folder = tmp_path / name
        folder.mkdir()
        rows = [{"label": f"a{i}", "point": "p", "pass": i, "outcome": "cleared" if i % 2 else "budget",
                 "completion_tokens": 100 * (i + 1), "level_rows": [{"tokens": 100 * (i + 1)}] if i % 2 else []}
                for i in range(tries)]
        (folder / "level-bench.json").write_text(json.dumps({"attempts": rows}))
        played.append((name, model, list(starts), tries, seed))
        return folder, rows, {"attempts": tries}

    loop.mg_play = fake_play
    from arena import level_bench
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    assert asyncio.run(loop.mg_eval()) == "eval"
    assert loop.server.loads == [("ev-base", "/runs/zero"), ("ev-policy", "/runs/r04")]
    by = {(n.split("-", 2)[2], m): (st, k, sd) for n, m, st, k, sd in played}
    for name in ("a", "f", "t"):
        assert by[(name, "ev-base")] == by[(name, "ev-policy")]
    assert by[("f", "ev-base")] == ([], 5, 9300) and by[("a", "ev-base")][1] == 3
    out = json.loads((loop.output / "rl" / "eval.json").read_text())
    assert out["sets"]["a"]["policy"]["outcomes"] == {"budget": 2, "cleared": 1}
    assert out["sets"]["a"]["base"]["clear_tokens"] == [None, 200, None]


def test_joint_read_refuses_arms_other_than_the_launch_resolved(tmp_path, monkeypatch):
    """23 Oct: a read served the bundle's adapter because the requested one never reached the container."""
    loop, _ = eval_loop(tmp_path, monkeypatch)
    loop.cfg["mg"].update(held_starts=1, fixed_tries=4, seed=8000, active_tries=9,
                          eval={"adapters": {"r12": "/runs/x/r12"}, "fixed_tries": 8})
    monkeypatch.setenv("ARC_RL1_EXPECT_EVAL", json.dumps({"r02": "/runs/p/r02"}))
    with pytest.raises(RuntimeError, match="differ from the launch"):
        asyncio.run(loop.mg_eval())
    assert loop.server.loads == []  # nothing served, nothing played


def test_joint_read_plays_every_adapter_and_the_base_on_the_loop_reads(tmp_path, monkeypatch):
    loop, _ = eval_loop(tmp_path, monkeypatch)
    loop.server.model = "base-model"
    loop.cfg["mg"].update(held_starts=1, fixed_tries=4, seed=8000, active_tries=9,
                          eval={"adapters": {"A": "/runs/a/r03", "C": "/runs/c/r03"}, "fixed_tries": 8})
    loop.mg_screen = {"sets": {"a": {"starts": [["/runs/s/attempts/x0", 3], ["/runs/s/attempts/x1", 4]]}}}
    loop.output = tmp_path / "out"
    (loop.output / "rl").mkdir(parents=True)
    played = []

    async def fake_play(name, spec, starts, tries, *, model, seed, concurrency=None):
        folder = tmp_path / name
        folder.mkdir()
        rows = [{"label": f"a{i}", "point": "p", "pass": i, "outcome": "cleared", "completion_tokens": 1000,
                 "level_rows": [{"tokens": 1000, "score": 50.0 if model == "ev-A" else 25.0}]} for i in range(tries)]
        (folder / "level-bench.json").write_text(json.dumps({"attempts": rows}))
        played.append((name, model, list(starts), tries, seed, concurrency))
        return folder, rows, {"attempts": tries}

    loop.mg_play = fake_play
    from arena import level_bench
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    monkeypatch.delenv("ARC_VALIDATION_OUTPUT", raising=False)
    assert asyncio.run(loop.mg_eval()) == "eval"
    assert loop.server.loads == [("ev-A", "/runs/a/r03"), ("ev-C", "/runs/c/r03")]  # the base takes no LoRA slot
    by = {n: (m, st, k, sd, c) for n, m, st, k, sd, c in played}
    assert set(by) == {f"jr-{a}-{s}" for a in ("A", "C", "base") for s in ("a", "f")}  # train sets only, no test set
    assert by["jr-base-a"] == ("base-model", [("/runs/s/attempts/x0", 3)], 8, 8500, 3)  # held_starts, seed+500, 9 // 3
    assert by["jr-A-f"] == ("ev-A", [], 8, 8500, 3)
    out = json.loads((loop.output / "rl" / "eval.json").read_text())
    assert out["pooled"]["A"]["rhae_per_10k"] == 5.0 and out["pooled"]["base"]["rhae_per_10k"] == 2.5
    assert out["sets"]["a"]["A"]["rhae"] == [0.5] * 8
    assert any(k == "eval" and d.get("label") == "joint_read" for k, d in loop.feed.events)


# --- mg6: reward, per-turn credit, behaviour guard -------------------------------------------------------------------

def mg6_loop(tmp_path, **mg):
    loop = make_loop(tmp_path, mg={"cap": 32000, "group_size": 3, "reward_shape": "rhae_free", "rhae_hi": 4.0,
                                   "action_backstop": 2.0, **mg})
    loop.cfg["l2_per_side"] = 12
    loop.mg_point = lambda game: {"baselines": [20, 40]}
    loop.mg_screen = {"sets": {"x-L2": {"metrics": {"median_to_clear": 10000, "median_actions": 30}}}}
    return loop


def try_row(label, tokens, actions, cleared=True, point="xgam-game-L2-a-k1"):
    return {"label": label, "point": point, "restore": True, "outcome": "cleared" if cleared else "budget",
            "completion_tokens": tokens, "level_rows": [{"level": 2, "tokens": tokens, "actions": actions}] if cleared else []}


def test_mg6_reward_uncapped_rhae_and_backstop(tmp_path):
    loop = mg6_loop(tmp_path)
    assert loop.mg_reward(try_row("a", 10000, 40), "x-L2") == pytest.approx(1.0)  # (40/40)^2 x 10K/10K
    assert loop.mg_reward(try_row("a", 5000, 20), "x-L2") == pytest.approx(4.0 * 2)  # (40/20)^2 = 4 (= rhae_hi) x 2
    assert loop.mg_reward(try_row("a", 5000, 10), "x-L2") == pytest.approx(4.0 * 2)  # capped at rhae_hi
    assert loop.mg_reward(try_row("a", 10000, 61), "x-L2") == 0.0  # > 2 x base median actions (30)
    assert loop.mg_reward(try_row("a", 10000, 40, cleared=False), "x-L2") == 0.0


def test_mg6_select_drops_zero_variance_and_keeps_signed(tmp_path):
    loop = mg6_loop(tmp_path)
    same = [try_row(f"a{i}-p{i}", 10000, 40, point="xgam-game-L2-a-k1") for i in range(3)]
    fails = [try_row(f"b{i}-p{i}", 10000, 40, cleared=False, point="xgam-game-L2-b-k2") for i in range(3)]
    mixed = [try_row("c0-p0", 5000, 40, point="xgam-game-L2-c-k3"), try_row("c1-p1", 7000, 40, point="xgam-game-L2-c-k3"),
             try_row("c2-p2", 10000, 40, cleared=False, point="xgam-game-L2-c-k3")]
    out, info = loop.mg_select(tmp_path, same + fails + mixed, "x-L2")
    assert info["groups"] == 3 and {r["label"] for _, r, _, _ in out} == {"c0-p0", "c1-p1", "c2-p2"}
    w = {r["label"]: a for _, r, a, _ in out}
    assert w["c0-p0"] > w["c1-p1"] > 0 > w["c2-p2"]


def test_preflight_spread_trains_tied_groups_only_in_a_preflight(tmp_path, monkeypatch):
    loop = mg6_loop(tmp_path)
    same = [try_row(f"a{i}-p{i}", 10000, 40, point="xgam-game-L2-a-k1") for i in range(3)]
    fails = [try_row(f"b{i}-p{i}", 10000, 40, cleared=False, point="xgam-game-L2-b-k2") for i in range(3)]
    monkeypatch.delenv("ARC_PREFLIGHT_STUB", raising=False)
    out, info = loop.mg_select(tmp_path, same + fails, "x-L2")
    assert out == [] and "preflight_spread" not in info
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", json.dumps({"corpus": [], "spread": False}))
    assert loop.mg_select(tmp_path, same + fails, "x-L2")[0] == []
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", json.dumps({"corpus": [], "spread": True}))
    out, info = loop.mg_select(tmp_path, same + fails, "x-L2")
    assert info["preflight_spread"] == 2 and len(out) == 6
    w = {r["label"]: a for _, r, a, _ in out}
    assert w["a2-p2"] > 0 > w["a0-p0"] and w["b2-p2"] > 0 > w["b0-p0"]
    mixed = [try_row("c0-p0", 5000, 40, point="xgam-game-L2-c-k3"), try_row("c1-p1", 7000, 40, point="xgam-game-L2-c-k3")]
    assert "preflight_spread" not in loop.mg_select(tmp_path, mixed, "x-L2")[1]  # untied groups keep their rewards


def write_try(folder, label, replies, boards):
    """replies: [(step, tokens)]; boards: {step: hex row}; the try ends at the last board's step."""
    att = folder / "attempts" / label
    att.mkdir(parents=True)
    with open(att / "rollout.jsonl", "w") as f:
        for n, (step, tokens) in enumerate(replies):
            f.write(json.dumps({"n": n, "step": step, "usage": {"completion_tokens": tokens}}, separators=(",", ":")) + "\n")
    with open(att / "log.jsonl", "w") as f:
        for step, board in boards.items():
            f.write(json.dumps({"kind": "observation", "step": step, "levels": 1, "frames": [[board]]},
                               separators=(",", ":")) + "\n")


def test_mg6_turn_credit_skips_root_and_uses_reward_shape(tmp_path):
    loop = mg6_loop(tmp_path, turn_credit={"anchor": 1.0, "token": 0.0, "clip": 5.0})
    # all three start on board R (root, skipped); A and B both reach board S at step 2, then A clears in 2 actions and
    # 1,000 tokens, B in 2 actions and 3,000 tokens; C fails
    write_try(tmp_path, "A", [(0, 500), (2, 1000)], {0: "0f0f", 2: "0a0a", 4: "0c0c"})
    write_try(tmp_path, "B", [(0, 500), (2, 3000)], {0: "0f0f", 2: "0a0a", 4: "0c0c"})
    write_try(tmp_path, "C", [(0, 500), (2, 800)], {0: "0f0f", 2: "0a0a", 9: "0d0d"})
    grp = [try_row("A", 1500, 4), try_row("B", 3500, 4), try_row("C", 1300, 9, cleared=False)]
    out = loop.mg_turn_credit(tmp_path, grp, [2.0, 1.0, 0.0], "x-L2")
    assert 0 not in out[0]["extra"] and 0 not in out[2]["extra"]  # root anchor skipped
    t_ref, ret = 2000.0, [2000 / 1000, 2000 / 3000, 0.0]  # (b_rem/a_rem)^2 = 1 for both clears
    loo = [r - (sum(ret) - r) / 2 for r in ret]
    assert [o["extra"][1] for o in out] == pytest.approx(loo)
    assert out[1]["tokens"] == {0: 500, 1: 3000}


def test_mg6_turn_credit_short_reply_with_more_actions_gets_less(tmp_path):
    loop = mg6_loop(tmp_path, turn_credit={"anchor": 1.0, "token": 0.5, "clip": 5.0})
    # at board S: A clears with 4 actions left and 2,000 tokens; B thinks less (1,000 tokens) but needs 12 actions
    write_try(tmp_path, "A", [(0, 300), (2, 2000)], {0: "0f0f", 2: "0a0a", 6: "0c0c"})
    write_try(tmp_path, "B", [(0, 300), (2, 1000)], {0: "0f0f", 2: "0a0a", 14: "0c0c"})
    grp = [try_row("A", 2300, 6), try_row("B", 1300, 14)]
    out = loop.mg_turn_credit(tmp_path, grp, [1.0, 1.0], "x-L2")
    assert out[0]["extra"][1] > out[1]["extra"][1]  # fewer tokens but 3x the actions: less credit at the anchor
    # B's 14 actions are within the base median (30): the token term applies to both; A's actions win anyway


def test_mg6_token_term_gated_on_base_actions(tmp_path):
    loop = mg6_loop(tmp_path, turn_credit={"anchor": 0.0, "token": 0.5, "clip": 5.0})
    write_try(tmp_path, "A", [(0, 1000)], {0: "0f0f", 10: "0c0c"})
    write_try(tmp_path, "B", [(0, 400)], {0: "0f0f", 40: "0c0c"})
    grp = [try_row("A", 1000, 10), try_row("B", 400, 40)]  # B: 40 actions > base median 30: no token credit
    out = loop.mg_turn_credit(tmp_path, grp, [1.0, 0.5], "x-L2")
    assert out[0]["extra"] == {0: 0.0} and out[1]["extra"] == {}


def test_mg6_reward_token_factor_capped(tmp_path):
    loop = mg6_loop(tmp_path)
    assert loop.mg_reward(try_row("a", 500, 40), "x-L2") == pytest.approx(6.0)  # 10K/500 = 20 -> token_hi 6


def test_mg6_reply_balance_scales_negative_replies(tmp_path):
    loop = mg6_loop(tmp_path, turn_credit={"anchor": 1.0})
    turn = {("f", "A"): {"tokens": {0: 100, 1: 100}, "extra": {0: 0.2, 1: -0.2}},
            ("f", "B"): {"tokens": {0: 400}, "extra": {0: -0.1}}}
    specs = [{"w": 1.0, "turn_key": ["f", "A"], "set": "x"}, {"w": -0.5, "turn_key": ["f", "B"], "set": "x"}]
    bal = loop.mg_reply_balance(specs, turn)
    up = 1.2 * 100 + 0.8 * 100
    down = 0.6 * 400
    assert bal["down_scale"] == pytest.approx(up / down, abs=1e-4)
    assert specs[1]["reply_w"]["0"] == pytest.approx(-0.6 * up / down) and specs[1]["w"] == pytest.approx(-0.5 * up / down)
    assert specs[0]["reply_w"] == {"0": pytest.approx(1.2), "1": pytest.approx(0.8)}
    assert rl1_trainer.reply_weight(specs[0], {"n": 1}) == pytest.approx(0.8)
    assert rl1_trainer.reply_weight(specs[0], {"n": 0, "scripted": True}) == 0.0


def test_behaviour_guard_actions_rule_off_with_none(tmp_path):
    loop = mg6_loop(tmp_path, behaviour_guard={"clear_drop": 0.08, "actions_rise": None})
    loop.mg_base_rounds = [1]
    loop.mg_fixed = {1: {"x": [try_row(f"b{i}", 1000, 20) for i in range(10)]},
                     2: {"x": [try_row(f"n{i}", 900, 40) for i in range(10)]}}
    out = loop.mg_behaviour(2)
    assert out["actions_rise"] == pytest.approx(1.0) and out["trip"] is False


def test_mg6_behaviour_guard_trips_and_stops(tmp_path):
    loop = mg6_loop(tmp_path, behaviour_guard={"clear_drop": 0.08, "actions_rise": 0.15})
    base = [try_row(f"b{i}", 1000, 20) for i in range(10)]
    loop.mg_base_rounds = [1]
    loop.mg_fixed = {1: {"x": base}, 2: {"x": [try_row(f"n{i}", 900, 21) for i in range(10)]},
                     3: {"x": [try_row(f"n{i}", 900, 25) for i in range(10)]},
                     4: {"x": [try_row(f"n{i}", 900, 25) for i in range(10)]},
                     5: {"x": [try_row(f"n{i}", 900, 20, cleared=i > 2) for i in range(10)]}}
    assert loop.mg_behaviour(2)["trip"] is False
    first = loop.mg_behaviour(3)
    assert first["trip"] and "stop" not in first and first["actions_rise"] == pytest.approx(0.25)
    assert loop.mg_behaviour(4)["stop"] == "behaviour_guard"  # second trip in a row
    loop.mg_behaviour_trips = 0
    assert loop.mg_behaviour(5)["stop"] == "behaviour_guard"  # clear rate -0.3 > 2 x 0.08 at once


def test_behaviour_guard_backtest_mra_v2_r4_r7(tmp_path):
    """actions_rise 0.20 and rhae_floor 0.783 (0.9 x the R4-R6 mean 0.87) with trips_stop 1 fire at
    mra-v2's R7 only (actions +0.076 / +0.028 / +0.059 / +0.469, mean RHAE 0.867 / 0.902 / 0.869 / 0.633)."""
    loop = mg6_loop(tmp_path, behaviour_guard={"clear_drop": 0.15, "actions_rise": 0.20, "rhae_floor": 0.783, "trips_stop": 1})

    def rows(actions, score):
        out = [try_row(f"r{i}", 9000, actions) for i in range(16)]
        for r in out:
            r["level_rows"][0]["score"] = score * 100
        return out

    loop.mg_base_rounds = [1]
    loop.mg_fixed = {1: {"x": rows(30, 0.8)}, 4: {"x": rows(32.28, 0.867)}, 5: {"x": rows(30.84, 0.902)},
                     6: {"x": rows(31.77, 0.869)}, 7: {"x": rows(44.07, 0.633)}}
    got = {n: loop.mg_behaviour(n) for n in (4, 5, 6, 7)}
    assert [got[n]["trip"] for n in (4, 5, 6, 7)] == [False, False, False, True]
    assert got[7]["stop"] == "behaviour_guard" and got[7]["mean_rhae"] == pytest.approx(0.633)
    loop.mg_behaviour_trips = 0
    loop.mg_fixed[8] = {"x": rows(31.0, 0.70)}  # RHAE floor alone
    assert loop.mg_behaviour(8)["trip"] is True


def test_screen_sets_wait_only_for_their_own_recordings(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, mg={"cap": 1000, "seed": 1, "record": [{"game": "slow", "games": 1, "budget": 10},
                                                                      {"game": "fast", "games": 1, "budget": 10}],
                                   "sets": [{"name": "a", "game": "slow", "level": 2, "role": "train", "starts": 1, "tries": 1,
                                             "sources": ["record:slow"]},
                                            {"name": "b", "game": "fast", "level": 2, "role": "train", "starts": 1, "tries": 1,
                                             "sources": ["record:fast"]},
                                            {"name": "c", "game": "x", "level": 1, "role": "train", "fresh": True, "tries": 1}]})
    loop.server.model = "base"
    loop.output = tmp_path
    (tmp_path / "rl").mkdir()
    events = []

    def made(name):
        (tmp_path / "rl" / name).mkdir(parents=True, exist_ok=True)
        (tmp_path / "rl" / name / "level-bench.json").write_text('{"attempts": []}')
        return tmp_path / "rl" / name

    async def bench(name, games, n, budgets, **kw):
        await asyncio.sleep(0.2 if "slow" in name else 0.01)
        events.append(("done", name))
        return made(name), [], {"seconds": 0}

    async def mg_play(name, spec, starts, tries, **kw):
        events.append(("set", name))
        return made(name), [try_row(f"a{i:03d}-x-p{i}", 900, 20) for i in range(max(1, len(starts)) * tries)], {"seconds": 0}

    loop.bench, loop.mg_play = bench, mg_play
    loop.mg_starts = lambda spec, sources: ([("s", 1)], 1)
    asyncio.run(loop.mg_screen())
    order = [e[1] for e in events]
    assert order.index("screen-c") < order.index("rec-fast") < order.index("screen-b") < order.index("rec-slow") < order.index("screen-a")

    # finished parts persist to the volume; a relaunch reuses them when the spec matches and replays the rest
    first = tmp_path / "v1"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(first))
    events.clear()
    asyncio.run(loop.mg_screen())
    assert sorted(x.name for x in (first / "partial").iterdir()) == ["rec-fast", "rec-slow", "screen-a", "screen-b", "screen-c"]
    sets1 = json.loads((tmp_path / "rl" / "screen.json").read_text())["sets"]
    loop.cfg["mg"]["reuse_from"] = [str(first)]
    loop.cfg["mg"]["record"][0]["budget"] = 20  # slow's recording changed: it and set a replay, the rest is reused
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v2"))
    events.clear()
    asyncio.run(loop.mg_screen())
    assert sorted(e[1] for e in events) == ["rec-slow", "screen-a"]
    assert json.loads((tmp_path / "rl" / "screen.json").read_text())["sets"] == sets1
    # a set cut short (fewer than 3/4 of its planned tries usable) is replayed, not reused
    marker = tmp_path / "v2" / "partial" / "screen-c" / "screen-set.json"
    saved = json.loads(marker.read_text())
    saved["out"]["tries"] = 0
    marker.write_text(json.dumps(saved))
    loop.cfg["mg"]["reuse_from"] = [str(tmp_path / "v2")]
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v3"))
    events.clear()
    asyncio.run(loop.mg_screen())
    assert sorted(e[1] for e in events) == ["screen-c"]


def test_mg_pick_per_set_starts_and_no_repeat(tmp_path):
    loop = mg6_loop(tmp_path, train_starts=3, train_tries=6, fill_tries=True, no_repeat=True)
    loop.feed.emit = lambda *a, **k: None
    pool = {"big": [(f"a{i}", i) for i in range(30)], "small": [(f"b{i}", i) for i in range(9)]}
    big, small = {"role": "train"}, {"role": "train", "train_starts": 1, "train_tries": 18}
    seen = {"big": [], "small": []}
    for number in range(1, 9):
        for name, s in (("big", big), ("small", small)):
            picked = loop.mg_pick(number, name, s, pool)
            if picked is not None:
                st, tries = picked
                assert tries == 18 * (name == "small") + 6 * (name == "big") or len(st) < (3 if name == "big" else 1)
                seen[name] += st
    assert len(seen["big"]) == 24 and len(set(seen["big"])) == 24  # 3 new starts x 8 rounds, none twice
    assert len(seen["small"]) == 8 and len(set(seen["small"])) == 8  # 1 x 18 tries per round
    loop.cfg["mg"]["no_repeat"] = False  # the old behaviour wraps
    assert loop.mg_pick(5, "small", {"role": "train", "train_starts": 3}, {"small": pool["small"][:4]})[0] == [("b0", 0), ("b1", 1), ("b2", 2)]
    loop.cfg["mg"]["no_repeat"] = True
    short = {"small": pool["small"][:6]}
    assert [loop.mg_pick(n, "small", small, short) is None for n in (6, 7)] == [False, True]  # dropped once spent


def test_mg7_floor_reward_makes_the_action_trade_a_loss(tmp_path):
    loop = mg6_loop(tmp_path, reward_shape="floor", action_floor=1.15, action_pow=3.0, action_hi=4.0, token_hi=3.0)
    r = lambda t, a: loop.mg_reward(try_row("a", t, a), "x-L2")  # base median 10K tokens, 30 actions
    assert r(10000, 30) == pytest.approx(1.0)
    assert r(10000, 34) == pytest.approx((30 / 34) ** 3)
    assert r(10000, 35) == 0.0  # above 1.15 x 30
    assert r(5000, 30) == pytest.approx(2.0) and r(2000, 30) == pytest.approx(3.0)  # token term capped at 3
    assert r(10000, 20) == pytest.approx(1.5 ** 3) and r(10000, 15) == pytest.approx(4.0)  # action term capped at 4
    old = mg6_loop(tmp_path)
    o = lambda t, a: old.mg_reward(try_row("a", t, a), "x-L2")
    # 10% fewer tokens for 10% more actions: a loss under both, a larger one here (0.84 vs mg6's 0.92)
    assert r(9000, 33) / r(10000, 30) < o(9000, 33) / o(10000, 30) < 1
    assert r(10000, 30) < r(9000, 30) and r(10000, 30) < r(10000, 28)


def test_mg7_reference_actions_pool_round1_base(tmp_path):
    loop = mg6_loop(tmp_path, reward_shape="floor", ref_round1=True)
    loop.mg_screen = {"sets": {"x-L2": {"metrics": {"median_to_clear": 10000, "median_actions": 20, "clears": 16}}}}
    assert loop.mg_level_ref("x-L2", "xgam", 2)["actions"] == 20
    loop.mg_fixed = {1: {"x-L2": [try_row(f"b{i}", 10000, 28) for i in range(8)]}}
    assert loop.mg_level_ref("x-L2", "xgam", 2)["actions"] == pytest.approx((16 * 20 + 8 * 28) / 24)


def test_mg7b_score_aware_cap(tmp_path):
    # mg6_loop: base median 10K tokens, 30 actions; human baseline 40 (level 2) -> cap max(31.5, 40 / sqrt(1.15) = 37.3)
    loop = mg6_loop(tmp_path, reward_shape="aware", action_floor=1.05, rhae_cap=1.15, token_hi=3.0)
    r = lambda t, a: loop.mg_reward(try_row("a", t, a), "x-L2")
    cap = 40 / 1.15 ** 0.5
    assert loop.mg_action_cap(loop.mg_level_ref("x-L2", "xgam", 2)) == pytest.approx(cap)
    assert r(10000, 37) == pytest.approx(1.0) and r(10000, 38) == 0.0  # free actions up to where RHAE starts to fall
    assert r(5000, 20) == pytest.approx(2.0) and r(2000, 30) == pytest.approx(3.0)  # tokens alone, capped at 3
    loop.mg_point = lambda game: {"baselines": [20, 25]}  # baseline under base actions: cap = 1.05 x 30
    loop.mg_refs = {}
    assert r(10000, 31) == pytest.approx(1.0) and r(9000, 33) == 0.0  # the 10%-for-10% trade earns nothing here


def test_mg7b_tight_action_term_and_guard_levels(tmp_path):
    loop = mg6_loop(tmp_path, reward_shape="aware", action_floor=1.05, rhae_cap=1.15, token_hi=3.0, tight_action_pow=2.0)
    r = lambda t, a: loop.mg_reward(try_row("a", t, a), "x-L2")
    assert r(10000, 34) == pytest.approx(1.0)  # loose level (baseline 40): no action term below the cap
    loop.mg_point = lambda game: {"baselines": [20, 25]}  # tight: cap 31.5 = 1.05 x 30
    loop.mg_refs = {}
    assert r(10000, 30) == pytest.approx(1.0) and r(10000, 31) == pytest.approx((30 / 31) ** 2)
    assert r(10000, 15) == pytest.approx(2.0)  # tight action term capped at 2
    loop.cfg["mg"]["behaviour_guard"] = {"clear_drop": 0.1, "actions_rise": 0.1, "levels": ["x-L2"]}
    loop.mg_base_rounds = [1]
    loop.mg_fixed = {1: {"x-L2": [try_row(f"a{i}", 9000, 20) for i in range(4)], "y-L1": [try_row(f"b{i}", 9000, 20) for i in range(4)]},
                     2: {"x-L2": [try_row(f"a{i}", 9000, 21) for i in range(4)], "y-L1": [try_row(f"b{i}", 9000, 40) for i in range(4)]}}
    out = loop.mg_behaviour(2)
    assert out["actions_rise"] == pytest.approx(0.05) and not out["trip"]  # y-L1's doubling is outside the guard's levels


def test_sc25_logeq_reward(tmp_path):
    # base median 10K tokens, 30 actions; baseline 40 -> R_b = min(1.15, (40/30)^2) = 1.15
    loop = mg6_loop(tmp_path, reward_shape="logeq", rhae_cap=1.15, log_clip=8.0, log_fail=16.0)
    r = lambda t, a, c=True: loop.mg_reward(try_row("a", t, a, cleared=c), "x-L2")
    assert r(10000, 30) == pytest.approx(0.0) and r(5000, 30) == pytest.approx(math.log(2))
    assert r(10000, 50) == pytest.approx(math.log(0.64 / 1.15))  # RHAE (40/50)^2 against the capped base
    assert r(100, 30) == pytest.approx(math.log(8)) and r(10000, 30, False) == pytest.approx(-math.log(16))
    assert r(10 ** 7, 30) == pytest.approx(-math.log(8))  # a very long clear stays above a fail
    loop.mg_point = lambda game: {"baselines": [20, 25]}
    loop.mg_refs = {}
    assert r(10000, 25) == pytest.approx(math.log(1 / (25 / 30) ** 2))


def test_sc25_logeq_select_and_anchor(tmp_path):
    loop = mg6_loop(tmp_path, reward_shape="logeq", turn_credit={"anchor": 1.0, "token": 0.0, "clip": 5.0})
    fails = [try_row(f"b{i}-p{i}", 10000, 40, cleared=False, point="xgam-game-L2-b-k2") for i in range(3)]
    slow = [try_row("c0-p0", 20000, 30, point="xgam-game-L2-c-k3"), try_row("c1-p1", 10000, 30, point="xgam-game-L2-c-k3")]
    out, info = loop.mg_select(tmp_path, fails + slow, "x-L2")
    w = {r["label"]: a for _, r, a, _ in out}
    assert info["groups"] == 2 and set(w) == {"c0-p0", "c1-p1"} and w["c1-p1"] > 0 > w["c0-p0"]  # all-fail group dropped
    write_try(tmp_path, "A", [(0, 500), (2, 1000)], {0: "0f0f", 2: "0a0a", 4: "0c0c"})
    write_try(tmp_path, "B", [(0, 500), (2, 3000)], {0: "0f0f", 2: "0a0a", 4: "0c0c"})
    write_try(tmp_path, "C", [(0, 500), (2, 800)], {0: "0f0f", 2: "0a0a", 9: "0d0d"})
    grp = [try_row("A", 1500, 4), try_row("B", 35000, 4), try_row("C", 1300, 9, cleared=False)]
    rewards = [loop.mg_reward(r, "x-L2") for r in grp]  # B's clear is negative: still a clear at the anchor
    assert rewards[1] < 0
    out = loop.mg_turn_credit(tmp_path, grp, rewards, "x-L2")
    ret = [math.log(2000 / 1000), math.log(2000 / 3000), -math.log(16)]
    assert [o["extra"][1] for o in out] == pytest.approx([x - (sum(ret) - x) / 2 for x in ret])


# --- time-shared trainer card (19 Oct) ----------------------------------------------------------------------------

def test_second_server_argv_moves_port_and_memory_share():
    argv = ["python", "-m", "vllm", "--port", "1234", "--gpu-memory-utilization", "0.96", "--max-model-len", "65536"]
    out = rl_loop.second_server_argv(argv, 1235, gpu_util=0.88)
    assert out[out.index("--port") + 1] == "1235" and out[out.index("--gpu-memory-utilization") + 1] == "0.88"
    assert argv[argv.index("--port") + 1] == "1234"  # the source argv untouched
    kv = rl_loop.second_server_argv(["vllm", "--kv-cache-memory-bytes", "1000"], 1235, kv_scale=0.9)
    assert kv[kv.index("--kv-cache-memory-bytes") + 1] == "900" and kv[-2:] == ["--port", "1235"]


def test_second_server_command_targets_the_trainer_card(tmp_path, monkeypatch):
    s = rl_loop.SecondServer(tmp_path, "m", card="1", gpu_util=0.88)
    monkeypatch.setattr(s, "source", lambda: (["vllm", "--port", "1234", "--gpu-memory-utilization", "0.96"],
                                              {"PATH": "/tmp/gpu0-bin:/usr/bin", "ARC_REAL_NVIDIA_SMI": str(tmp_path / "smi")}))
    monkeypatch.setattr(rl_loop.Path, "mkdir", lambda self, **kw: None)
    written = {}
    monkeypatch.setattr(rl_loop.Path, "write_text", lambda self, text: written.setdefault(str(self), text))
    monkeypatch.setattr(rl_loop.Path, "chmod", lambda self, mode: None)
    argv, env = s.command()
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["PATH"].split(":")[0] == "/tmp/gpu1-bin" and "/tmp/gpu0-bin" not in env["PATH"]
    assert "-i 1" in written["/tmp/gpu1-bin/nvidia-smi"]
    assert argv[argv.index("--port") + 1] == "1235" and s.root == "http://127.0.0.1:1235"
    assert s.folder == tmp_path / "server2"


def test_second_server_extra_env_only_on_server_2(tmp_path, monkeypatch):
    """timeshare.env (determinism A/B design, 24 Oct): server 2's env on top of server 1's copied env."""
    s = rl_loop.SecondServer(tmp_path, "m", card="1", env={"ARC_DET_FIX": 1})
    monkeypatch.setattr(s, "source", lambda: (["vllm", "--port", "1234"], {"PATH": "/usr/bin", "ARC_DET_FIX": "0"}))
    argv, env = s.command()
    assert env["ARC_DET_FIX"] == "1" and env["CUDA_VISIBLE_DEVICES"] == "1"


class FakeServer2:
    def __init__(self):
        self.card, self.memory, self.calls, self.root = "1", {"awake_gib": 240.0, "asleep_gib": 4.0}, [], "http://x:1235"

    async def wake(self):
        self.calls.append("wake")
        return 1.0

    async def serve(self, name, path):
        self.calls.append(("serve", name, str(path)))
        return 0.5

    async def sleep(self):
        self.calls.append("sleep")
        return 0.2


def test_time_share_on_parks_wakes_and_opens_the_pool_and_off_sleeps(tmp_path, monkeypatch):
    from arena import level_bench

    loop = make_loop(tmp_path, mg={"timeshare": {"min_queued": 4, "park_base": True}, "active_tries": 8})
    loop.server2 = FakeServer2()
    pool, pool2 = level_bench.SharedSlots(8), level_bench.SharedSlots(8)
    pool2.active = False
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", pool)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", pool2)

    class Task:
        def __init__(self, done):
            self._done = done

        def done(self):
            return self._done

    used = [20.0]
    monkeypatch.setattr(rl_loop, "card_memory_of", lambda card: (used[0], 268.6))
    # too little queued play: nothing parks
    pool.want("k", 0, 2)
    info = asyncio.run(loop.ts_on(3, [Task(False)], "/a/r02"))
    assert "park" not in info and not pool2.active and loop.trainer.calls == []
    # enough queued, room on the card: park with the base, wake, serve the first server's adapter, open the pool
    pool.want("k", 0, 30)
    info = asyncio.run(loop.ts_on(3, [Task(False)], "/a/r02"))
    assert loop.trainer.calls[0] == ("park", {"base": True}) and pool2.active
    assert loop.server2.calls == ["wake", ("serve", "policy", "/a/r02")] and info["need_gib"] == 237.0
    pool2.served, pool2.peak = 5, 3
    off = asyncio.run(loop.ts_off())
    assert not pool2.active and off["served"] == 5 and off["peak"] == 3 and loop.server2.calls[-1] == "sleep"
    # no room beside the parked trainer (its residue too large): skip, the pool stays shut
    used[0] = 40.0
    loop.server2.calls.clear()
    info = asyncio.run(loop.ts_on(4, [Task(False)], "/a/r03"))
    assert info["free_gib"] < info["need_gib"] and not pool2.active and loop.server2.calls == []
    assert [k for k, _ in loop.feed.events].count("timeshare_skip") == 2
    # play already over: skip
    assert "park" not in asyncio.run(loop.ts_on(5, [Task(True)], "/a/r04"))


def test_park_check_failure_turns_the_time_share_off(tmp_path, monkeypatch):
    from arena import level_bench

    loop = make_loop(tmp_path, mg={"timeshare": {}})
    stopped = []
    loop.server2 = type("S", (), {"stop": lambda self: stopped.append(1)})()
    pool2 = level_bench.SharedSlots(4)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", pool2)

    async def bad(op, timeout=0, **kw):
        return {"ok": True, "identical": False, "max_abs_diff": 0.01}
    loop.trainer.call = bad
    asyncio.run(loop.ts_park_check([{"name": "p"}]))
    assert loop.ts_dead and not pool2.active and stopped == [1]

    loop2 = make_loop(tmp_path, mg={"timeshare": {}})

    async def good(op, timeout=0, **kw):
        return {"ok": True, "identical": True, "max_abs_diff": 0.0}
    loop2.trainer.call = good
    asyncio.run(loop2.ts_park_check([{"name": "p"}]))
    assert not getattr(loop2, "ts_dead", False)



# --- drain_half starts and windows (19 Oct) -----------------------------------------------------------------------

def test_restorable_starts_pass_token_drains_and_stop_at_forced_drains_and_interrupts(tmp_path):
    from arena.restore import unsafe_from

    rows = [{"kind": "observation", "step": 1}, {"kind": "context_drain", "step": 40, "trigger": "tokens"},
            {"kind": "context_drain", "step": 70, "trigger": "turns"}, {"kind": "context_drain", "step": 90, "trigger": "forced"},
            {"kind": "duck_python_interrupt", "step": 120}]
    (tmp_path / "log.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    assert unsafe_from(tmp_path) == (90, [40, 70])
    (tmp_path / "log.jsonl").write_text(json.dumps({"kind": "context_drain", "step": 5, "trigger": "tokens"},
                                                   separators=(",", ":")) + "\n")
    assert unsafe_from(tmp_path) == (None, [5])


def test_summary_requests_are_not_gameplay_rows():
    from arena.rollout_record import request_purpose

    tools = [{"type": "function", "function": {"name": "python"}}]
    assert request_purpose({"tools": tools}) == "gameplay"
    assert request_purpose({"tools": tools, "tool_choice": "none"}) == "summary"
    assert request_purpose({}) == "other"


def test_training_windows_never_cross_a_drain():
    import rl_job

    def cap(prompt, out):
        return {"prompt_tokens": prompt, "usage": {"completion_tokens": out}}

    # append-only to request 3, a drain before request 4 (the prompt shrinks), append-only again
    caps = [cap(20000, 1000), cap(22000, 1000), cap(24000, 1000), cap(26000, 1000),
            cap(12000, 1000), cap(14000, 1000), cap(16000, 1000)]
    ends = rl_job.plan_windows(caps, 32768)
    assert ends == [6, 3]  # one window per append-only stretch, none spanning the drain
    # a narrow window splits a stretch further but still never crosses the drain
    ends = rl_job.plan_windows(caps, 5000)
    spans = []
    for end in ends:
        first = end
        while first - 1 >= 0 and caps[first - 1]["prompt_tokens"] + 1000 <= caps[first]["prompt_tokens"] \
                and caps[end]["prompt_tokens"] + 1000 - caps[first - 1]["prompt_tokens"] + 1 <= 5000:
            first -= 1
        spans.append((first, end))
    assert all(not (a <= 3 < b) for a, b in spans)



def test_recordings_split_into_two_stable_halves():
    from arena.rl_single import start_half

    labels = [f"a{i:03d}-sc25-game-p{i}" for i in range(200)]
    halves = [start_half(x) for x in labels]
    assert halves == [start_half(x) for x in labels] and 80 < sum(halves) < 120


def test_screen_set_without_starts_or_failing_is_missing_not_fatal(tmp_path, monkeypatch):
    # preflight smoke2 (19 Oct): a level no recorded game reached raised "No start points selected" and ended the screen
    loop = make_loop(tmp_path, mg={"cap": 1000, "seed": 1, "record": [{"game": "g", "games": 1, "budget": 10}],
                                   "sets": [{"name": "deep", "game": "g", "level": 5, "role": "test", "starts": 2, "tries": 1,
                                             "sources": ["record:g"]},
                                            {"name": "bad", "game": "g", "level": 2, "role": "test", "starts": 1, "tries": 1,
                                             "sources": ["record:g"]},
                                            {"name": "ok", "game": "x", "level": 1, "role": "train", "fresh": True, "tries": 1}]})
    loop.server.model, loop.output = "base", tmp_path
    (tmp_path / "rl").mkdir()

    def made(name):
        (tmp_path / "rl" / name).mkdir(parents=True, exist_ok=True)
        (tmp_path / "rl" / name / "level-bench.json").write_text('{"attempts": []}')
        return tmp_path / "rl" / name

    async def bench(name, games, n, budgets, **kw):
        return made(name), [], {"seconds": 0}

    async def mg_play(name, spec, starts, tries, **kw):
        if spec["name"] == "bad":
            raise RuntimeError("boom")
        return made(name), [try_row(f"a{i:03d}-x-p{i}", 900, 20) for i in range(max(1, len(starts)) * tries)], {"seconds": 0}

    loop.bench, loop.mg_play = bench, mg_play
    loop.mg_starts = lambda spec, sources: (([], 0) if spec["level"] == 5 else ([("s", 1)], 1))
    asyncio.run(loop.mg_screen())
    screen = json.loads((tmp_path / "rl" / "screen.json").read_text())
    assert list(screen["sets"]) == ["ok"] and sorted(screen["missing"]) == ["bad", "deep"]


# --- P1 trainer-only lr probe -----------------------------------------------------------------------------------------

def lr_probe_loop(tmp_path, monkeypatch, rates=(6e-5, 1.2e-4, 3e-5), min_seconds=0):
    import time as _time

    loop = make_loop(tmp_path, mg={"lr_probe": {"rates": list(rates), "min_seconds": min_seconds}, "max_groups": 9})
    loop.cfg["l2_starts_per_step"] = 3
    loop.deadline = _time.time() + 10_000
    loop.mg_seed = lambda pool, fixed, tests: ([("f", {"label": "a"}, 1.0, "k", "A")], {"A": {"trained": 1}})
    calls = []

    async def fake_update(number, kept, probes):
        calls.append((number, loop.cfg["lr"], loop.cfg["mg"]["max_groups"], loop.cfg["l2_starts_per_step"]))
        steps = [{"step": i + 1, "kl_base": {"p1": 0.02 * (i + 1)}, "logratio_base": {"p1": -0.01}, "loss": 0.0}
                 for i in range(3)]
        return {"ok": True, "steps": steps, "probes": {"p1": {"kl_base": 0.06}}}, tmp_path / "r01"

    loop.mg_update = fake_update
    started = []

    async def start():
        started.append(1)

    async def must(op, timeout=3600, **payload):
        loop.trainer.calls.append((op, payload))
        if op == "score":
            label = str(payload.get("label") or "")
            return {"windows": {"p1": {"kl_base": loop.reload_kl.get(label, 0.06) if label.startswith("reload") else 0.0}}}
        return {"ok": True}

    loop.trainer.must = must
    loop.reload_kl = {}
    return loop, calls, start, started


def test_lr_probe_resets_and_scores_base_per_rate(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_RL1_RESUME", raising=False)
    loop, calls, start, started = lr_probe_loop(tmp_path, monkeypatch)
    out = asyncio.run(loop.mg_lr_probe({}, {}, [], loop.mg_probes, start))
    assert started == [1] and [c[1] for c in calls] == [6e-5, 1.2e-4, 3e-5]
    assert all(c[0] == 1 and c[2] == 9 and c[3] == 3 for c in calls)  # the same groups and step size for every rate
    ops = [op for op, _ in loop.trainer.calls]
    assert ops == ["reset", "score", "reset", "score", "reset", "score"]
    assert [p.get("lr") for op, p in loop.trainer.calls if op == "reset"] == [6e-5, 1.2e-4, 3e-5]
    assert out["rates"]["6e-05"]["steps"][2]["kl_base"] == 0.06 and not out["skipped"]
    assert [k for k, _ in loop.feed.events].count("lr_probe_rate") == 3


def test_lr_probe_skips_done_rates_and_short_time(tmp_path, monkeypatch):
    prior = tmp_path / "prior"
    (prior / "arena/trials/rl-loop/rl").mkdir(parents=True)
    (prior / "arena/trials/rl-loop/rl/feed.jsonl").write_text(
        json.dumps({"kind": "lr_probe_rate", "lr": 6e-5, "ok": True}) + "\n"
        + json.dumps({"kind": "lr_probe_rate", "lr": 1.2e-4, "ok": False}) + "\n")
    monkeypatch.setenv("ARC_RL1_RESUME", str(prior))
    loop, calls, start, _ = lr_probe_loop(tmp_path, monkeypatch, min_seconds=5000)
    loop.deadline -= 6000  # 4,000 s left: under min_seconds
    out = asyncio.run(loop.mg_lr_probe({}, {}, [], loop.mg_probes, start))
    assert calls == [] and [s["why"] for s in out["skipped"]][0] == "done" and "s left" in out["skipped"][1]["why"]
    loop, calls, start, _ = lr_probe_loop(tmp_path, monkeypatch)
    asyncio.run(loop.mg_lr_probe({}, {}, [], loop.mg_probes, start))
    assert [c[1] for c in calls] == [1.2e-4, 3e-5]  # a failed rate is played again


def test_lr_probe_reload_check_path(tmp_path, monkeypatch):
    """mg.reload_check: the first rate is the run's own first-phase smoke (save, park, reload, rescore, one more step)."""
    monkeypatch.delenv("ARC_RL1_RESUME", raising=False)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    loop, calls, start, _ = lr_probe_loop(tmp_path, monkeypatch, rates=(6e-5,))
    loop.cfg["mg"]["reload_check"] = True
    saved = []

    async def save_checkpoint():
        saved.append(loop.pending_checkpoint)
        loop.pending_checkpoint = None

    loop.save_checkpoint = save_checkpoint
    loop.cfg["checkpoint_every"] = 0
    out = asyncio.run(loop.mg_lr_probe({}, {}, [], loop.mg_probes, start))
    assert [c[:4] for c in calls] == [(1, 6e-5, 9, 3), (2, 6e-5, 9, 3)]  # the rate's update, then one more after the reload
    assert saved and saved[0][0] == 1
    ops = [op for op, _ in loop.trainer.calls]
    assert ops == ["reset", "score", "score", "score", "park", "load_checkpoint", "score"]
    row = out["rates"]["6e-05"]
    assert row["ok"] and row["reload"]["ok"] and row["reload"]["max_diff"] == 0.0


@pytest.mark.parametrize("pre2,reload,ok", [(0.0605, 0.0612, True), (0.0601, 0.0620, False), (0.06, 0.0611, False)])
def test_lr_probe_reload_bar_is_the_measured_repeat_floor(tmp_path, monkeypatch, pre2, reload, ok):
    """hprobe-v2: a 1.13e-3 reload diff failed the fixed 1e-3 bar; the bar is 3 x the repeat floor + 2e-4 (<= 5e-3)."""
    from arena.rl_loop import StopLoop

    monkeypatch.delenv("ARC_RL1_RESUME", raising=False)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    loop, calls, start, _ = lr_probe_loop(tmp_path, monkeypatch, rates=(6e-5,))
    loop.cfg["mg"]["reload_check"] = True

    async def save_checkpoint():
        loop.pending_checkpoint = None

    loop.save_checkpoint = save_checkpoint
    loop.cfg["checkpoint_every"] = 0
    loop.reload_kl = {"reload-pre1": 0.06, "reload-pre2": pre2, "reload": reload}
    if ok:
        r = asyncio.run(loop.mg_lr_probe({}, {}, [], loop.mg_probes, start))["rates"]["6e-05"]["reload"]
    else:  # a failed reload check stops the run
        with pytest.raises(StopLoop) as stop:
            asyncio.run(loop.mg_lr_probe({}, {}, [], loop.mg_probes, start))
        assert stop.value.reason == "reload_check_failed"
        r = stop.value.detail
    assert r["ok"] is ok and abs(r["bar"] - min(5e-3, 3 * abs(pre2 - 0.06) + 2e-4)) < 1e-9


def test_mg_update_under_lr_probe_probes_every_step_without_control(tmp_path):
    loop = make_loop(tmp_path, mg={"lr_probe": {"rates": [1e-5]}, "max_groups": 4, "balance": False})
    loop.cfg.update(l2_starts_per_step=2, lvl_window=32768, path_chunk=16384, l2_norm="cap")
    loop.cfg["mg"]["cap"] = 32000
    loop.mg_screen = {"sets": {"A": {"metrics": {"median_to_clear": 1000}}}}
    loop.cfg["mg_sets"] = {"A": {"cap": 4000}}
    kept = [(tmp_path, {"label": f"g{g}-t{t}"}, 0.5 if t % 2 else -0.5, f"A-g{g}", "A") for g in range(4) for t in range(2)]
    seen = {}

    async def fake_dosed(number, *a, **k):
        seen.update(k)
        return {"ok": True}, tmp_path / "r01"

    loop.dosed_update = fake_dosed
    loop.checkpoint = lambda *a, **k: seen.setdefault("checkpointed", True)
    asyncio.run(loop.mg_update(1, kept, loop.mg_probes))
    assert seen["early"] is None and seen["probe_last"] is False and "checkpointed" not in seen
    assert len(seen["batches"]) == 2


def test_gspo_scores_pi_old_once_for_the_later_steps_and_passes_the_loss(tmp_path):
    loop = make_loop(tmp_path, mg={"policy_loss": "gspo", "gspo_clip": [3e-4, 4e-4], "kl_control": None})
    loop.cfg.update(steps_per_round=1, update_timeout=60)
    batches = [[{"name": "a", "w": 1.0}], [{"name": "b", "w": -1.0}], [{"name": "c", "w": 0.5}]]
    loop.trainer = FakeTrainer(steps=[(0.0, 0.01)] * 3)
    merged, _ = asyncio.run(loop.dosed_update(1, None, None, None, loop.mg_probes, batches=batches, signed=True))
    ops = loop.trainer.calls
    assert [op for op, _ in ops] == ["old_logprobs", "update", "update", "update"]
    assert [s["name"] for s in ops[0][1]["attempts"]] == ["b", "c"] and ops[0][1]["signed"] is True
    assert all(p["policy_loss"] == "gspo" and p["gspo_clip"] == [3e-4, 4e-4] for op, p in ops[1:])
    plain = make_loop(tmp_path, mg={"kl_control": None}, steps=[(0.0, 0.01)] * 3)
    plain.cfg.update(steps_per_round=1)
    asyncio.run(plain.dosed_update(1, None, None, None, plain.mg_probes, batches=batches, signed=True))
    assert [op for op, _ in plain.trainer.calls] == ["update"] * 3 and "policy_loss" not in plain.trainer.calls[0][1]


def test_gspo_old_pass_takes_the_update_window_path(tmp_path):
    """Stage W (22 Oct): pi_old's pass gets the update's window settings (prefix sharing, trainer_fast: prefetch,
    snapshots, encoder processes, window cache), not its update-only keys."""
    loop = make_loop(tmp_path, mg={"policy_loss": "gspo", "kl_control": None})
    loop.cfg.update(steps_per_round=1, update_timeout=60)
    loop.trainer = FakeTrainer(steps=[(0.0, 0.01)] * 2)
    fast = {"prefix_snapshots": True, "prefetch": 2, "encode_procs": 8, "window_cache": 64}
    extra = {"share_prefix": True, "prefix_chunk": 16384, "norm": "cap", "norm_cap": 4000, "batch_tokens": 0, "batch_max": 8,
             "hold_previous": False, "aux": {"file": "x"}, **fast}
    asyncio.run(loop.dosed_update(1, None, None, None, loop.mg_probes, batches=[[{"name": "a", "w": 1.0}], [{"name": "b", "w": 1.0}]],
                                  signed=True, cmd_extra=extra))
    (op, old), update = loop.trainer.calls[0], loop.trainer.calls[1][1]
    assert op == "old_logprobs" and {k: old[k] for k in fast} == fast and old["share_prefix"] and old["prefix_chunk"] == 16384
    assert not set(old) & set(rl_loop.OLD_PASS_SKIP) and {k: update[k] for k in extra} == extra


def test_fake_trainer_protocol(tmp_path):
    import subprocess

    roll = tmp_path / "a" / "rollout.jsonl"
    roll.parent.mkdir()
    roll.write_text(json.dumps({"usage": {"completion_tokens": 7}}) + "\n" + json.dumps({"scripted": True, "usage": {"completion_tokens": 5}}) + "\n")
    command, env = None, None
    import os
    os.environ["ARC_PREFLIGHT_FAKE_TRAINER"] = "1"
    try:
        cfg = rl_loop.settings({"mode": "loop"})
        command, env = rl_loop.trainer_command(cfg, Path(__file__).resolve().parents[1], tmp_path)
    finally:
        del os.environ["ARC_PREFLIGHT_FAKE_TRAINER"]
    trainer = rl_loop.Trainer(tmp_path / "trainer", command, env, tmp_path / "t.log")

    async def go():
        await trainer.ready(30)
        upd = await trainer.must("update", attempts=[{"name": "a", "rollout": str(roll), "w": 1.0}], probes=[{"name": "p"}],
                                 save=str(tmp_path / "ad"), signed=True)
        ck = await trainer.must("checkpoint", path=str(tmp_path / "st"))
        back = await trainer.must("load_checkpoint", path=str(tmp_path / "st"))
        bad = await trainer.call("load_checkpoint", path=str(tmp_path / "none"))
        await trainer.call("exit")
        return upd, ck, back, bad

    upd, ck, back, bad = asyncio.run(go())
    trainer.close()
    assert upd["trained_tokens"] == 7 and upd["probes"]["p"]["kl_base"] > 0 and (tmp_path / "ad" / "adapter_config.json").exists()
    assert back["updates"] == 1 and bad["ok"] is False


def test_fresh_starts_take_the_loop_sets_half(tmp_path, monkeypatch):
    loop, calls = eval_loop(tmp_path, monkeypatch)
    loop.cfg["mg_sets"]["a"]["start_half"] = 0
    loop.mg_fresh_starts(2)
    assert all(c[0] != "a" or True for c in calls)
    seen = []
    loop.mg_starts = lambda spec, srcs: (seen.append(spec.get("start_half")), ([], 0))[1]
    loop.mg_fresh_starts(2)
    assert seen and seen[0] == 0  # the screen's half (held-out reads) replaced by the loop set's training half


def _live_attempt(folder, reach_step=None, steps=6, partial=True):
    """A recording mid-play: an observation and a gameplay request per step; level 1 done at ``reach_step``."""
    folder.mkdir(parents=True, exist_ok=True)
    dump = lambda r: json.dumps(r, separators=(",", ":")) + "\n"  # noqa: E731
    log = [dump({"kind": "observation", "step": s, "levels": int(reach_step is not None and s >= reach_step),
                 "frames": [["00", "00"]], "action": None if s == 0 else {"action": "ACTION1"}}) for s in range(steps)]
    if reach_step is not None:
        log.insert(reach_step + 1, dump({"kind": "level_bench_level", "level": 1, "steps": reach_step}))
    rows = [dump({"n": s, "step": s, "gameplay": True, "usage": {"prompt_tokens": 100}}) for s in range(steps)]
    (folder / "log.jsonl").write_text("".join(log))
    (folder / "rollout.jsonl").write_text("".join(rows) + ('{"n": 99, "st' if partial else ""))


def test_screen_stream_starts_begin_sets_before_the_recordings_end(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, mg={"cap": 1000, "seed": 1, "stream_starts": True, "stream_poll": 0.01,
                                   "record": [{"game": "g", "games": 3, "budget": 10}],
                                   "sets": [{"name": "a", "game": "g", "level": 2, "role": "train", "starts": 2, "tries": 1,
                                             "sources": ["record:g"]}]})
    loop.server.model = "base"
    loop.output = tmp_path
    (tmp_path / "rl").mkdir()
    events, seen = [], {}

    async def bench(name, games, n, budgets, **kw):
        live = tmp_path / "rl" / name / "attempts"
        _live_attempt(live / "c", None)
        await asyncio.sleep(0.05)
        _live_attempt(live / "a", 3)
        await asyncio.sleep(0.05)
        _live_attempt(live / "b", 2)
        await asyncio.sleep(0.4)
        events.append(("done", name))
        (tmp_path / "rl" / name / "level-bench.json").write_text('{"attempts": []}')
        return tmp_path / "rl" / name, [], {"seconds": 0}

    async def mg_play(name, spec, starts, tries, **kw):
        events.append(("set", name))
        seen[name] = starts
        (tmp_path / "rl" / name).mkdir(parents=True, exist_ok=True)
        return tmp_path / "rl" / name, [try_row(f"a{i:03d}-x-p{i}", 900, 20) for i in range(len(starts))], {"seconds": 0}

    loop.bench, loop.mg_play = bench, mg_play
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v1"))
    asyncio.run(loop.mg_screen())
    assert [e[1] for e in events] == ["screen-a", "rec-g"]
    starts = seen["screen-a"]
    assert [(Path(a).name, r) for a, r in starts] == [("g-a", 3), ("g-b", 2)]
    assert not (Path(starts[0][0]) / "rollout.jsonl").read_text().endswith('"st')  # cut at the last complete line
    assert (tmp_path / "v1" / "partial" / "snap-a" / "starts.json").exists()
    assert "screen-a" in __import__("arena.level_bench", fromlist=["x"]).TRY_REUSE_SKIP

    # a restart takes the persisted choice back, even with the recording replayed
    import shutil as sh
    sh.rmtree(tmp_path / "rl")
    (tmp_path / "rl").mkdir()
    loop.cfg["mg"]["reuse_from"] = [str(tmp_path / "v1")]
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "v2"))
    events.clear()
    asyncio.run(loop.mg_screen())
    assert seen["screen-a"] == starts and (Path(starts[0][0]) / "rollout.jsonl").exists()


def test_screen_stream_falls_back_to_the_usual_pick_when_recordings_end_first(tmp_path):
    loop = make_loop(tmp_path, mg={"cap": 1000, "seed": 1, "stream_starts": True, "stream_poll": 0.01,
                                   "record": [{"game": "g", "games": 1, "budget": 10}],
                                   "sets": [{"name": "a", "game": "g", "level": 2, "role": "train", "starts": 2, "tries": 1,
                                             "sources": ["record:g"]}]})
    loop.server.model = "base"
    loop.output = tmp_path
    (tmp_path / "rl").mkdir()

    async def bench(name, games, n, budgets, **kw):
        _live_attempt(tmp_path / "rl" / name / "attempts" / "a", 3)
        await asyncio.sleep(0.1)
        (tmp_path / "rl" / name / "level-bench.json").write_text('{"attempts": []}')
        return tmp_path / "rl" / name, [], {"seconds": 0}

    picked = []
    loop.bench = bench
    loop.mg_starts = lambda spec, sources: (picked.append(sources) or [("s", 1)], 1)

    async def mg_play(name, spec, starts, tries, **kw):
        (tmp_path / "rl" / name).mkdir(parents=True, exist_ok=True)
        return tmp_path / "rl" / name, [try_row("a000-x-p0", 900, 20)], {"seconds": 0}

    loop.mg_play = mg_play
    asyncio.run(loop.mg_screen())
    assert picked == [[str(tmp_path / "rl" / "rec-g")]]


def test_gate_overcommit_reaches_the_bench_study(tmp_path, monkeypatch):
    from arena import level_bench

    loop = make_loop(tmp_path, mg={"cap": 1000, "gate_overcommit": 1.6, "transport_backoff": [1, 2, 4]})
    loop.output = tmp_path
    loop.study, loop.trial, loop.deadline = {"level_bench": {}}, {}, 10 ** 10
    loop.source = loop.games_dir = loop.server_dir = loop.analysis_python = loop.base_url = None
    loop.gpu = type("G", (), {"mark": lambda self, x: None})()
    seen = {}

    async def run(study, trial, **kw):
        seen["study"] = study
        raise RuntimeError("stop")

    monkeypatch.setattr(level_bench, "run_level_bench", run)
    with pytest.raises(RuntimeError):
        asyncio.run(loop.bench("x", ["g"], 1, {"g": 10}))
    assert seen["study"]["gate_overcommit"] == 1.6 and "gate_overcommit" not in loop.study
    assert seen["study"]["transport_backoff"] == [1, 2, 4]


def test_mg_keys_cover_every_key_the_code_reads_and_unknown_keys_refuse():
    import glob
    import re

    read = set()
    for f in glob.glob(str(Path(rl_loop.__file__).parent / "*.py")):
        read |= set(re.findall(r'(?:\bmg|\["mg"\])(?:\.get\(|\[)\s*"([a-z0-9_]+)"', Path(f).read_text()))
    assert read <= rl_loop.MG_KEYS, sorted(read - rl_loop.MG_KEYS)
    rl_loop.settings({"mg": {"cap": 1, "sets": [{"name": "a", "level": 1}], "screens": [{"screen": "x", "map": {}}]}})
    for bad in ({"cap": 1, "gate_overcomit": 1.5}, {"sets": [{"name": "a", "start_halves": 1}]},
                {"record": [{"game": "g", "budgets": 1}]}):
        with pytest.raises(ValueError, match="mg keys"):
            rl_loop.settings({"mg": bad})


def test_adapter_load_refuses_partial_or_unlisted_adapters(tmp_path):
    from arena.fake_trainer import adapter

    folder = Path(adapter(tmp_path / "a"))
    assert rl_loop.adapter_tensors(folder) == 4
    (folder / "adapter-tensors.json").write_text(json.dumps({"tensors": 4, "expected": 6}))
    with pytest.raises(ValueError, match="manifest"):
        rl_loop.adapter_tensors(folder)
    (folder / "adapter-tensors.json").unlink()
    with pytest.raises(ValueError, match="no readable"):
        rl_loop.adapter_tensors(folder)


def test_every_trainer_op_that_runs_the_model_unparks_first():
    import re

    src = (Path(__file__).parents[1] / "scripts/posttrain/rl1_trainer.py").read_text()
    gpu = eval(re.search(r"GPU_OPS = (\([^)]*\))", src).group(1))
    ops = re.findall(r"\n    def op_([a-z_0-9]+)\(self[^\n]*\n(.*?)(?=\n    def |\nclass |\n[A-Z_]+ = )", src, re.S)
    runs = [n for n, body in ops if any(k in body for k in ("window_pass", "self.model(", ".backward(", "attempt_window"))]
    assert runs and set(runs) <= set(gpu), sorted(set(runs) - set(gpu))


def test_kl_damage_only_steers_lr_and_rolls_back_past_the_damage_line(tmp_path):
    loop = make_loop(tmp_path, mg={"kl_target": 0.08, "kl_damage": 0.3, "lr_max": 2e-4}, lr=6e-5)
    early = loop.mg_early()
    assert early["stop_at"] == float("inf") and early["rollback_at"] == 0.3
    loop.cfg["l2_kl_stop"] = None
    over = {"ok": True, "early": {"action": "full", "steps": 4, "excess": 0.2}, "probes": {}}
    stop, served = asyncio.run(loop.mg_after_update(2, {}, over, "a2", load=False))
    assert (stop, served) == (None, True) and abs(loop.cfg["lr"] - 4e-5) < 1e-12  # over target: lr / 1.5, kept
    bad = {"ok": True, "early": {"action": "rollback", "steps": 2, "excess": 0.35}, "probes": {}}
    loop.mg_served = [("r00", 0), ("r02", 2)]
    loop.keep_checkpoint = "r02"
    import os
    os.environ["ARC_VALIDATION_OUTPUT"] = str(tmp_path)
    try:
        stop, served = asyncio.run(loop.mg_after_update(3, {}, bad, "a3", load=False))
    finally:
        del os.environ["ARC_VALIDATION_OUTPUT"]
    assert served is False and stop is None and loop.mg_step_rollbacks == 1 and abs(loop.cfg["lr"] - 2e-5) < 1e-12


def test_play_breakage_against_the_base_round():
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = {"mg": {"breakage": {"reply_ratio": 2.0, "ratio": 2.0, "abs": 0.05}}}
    row = lambda toks, asked, live, capped, game, noact: {"completion_tokens": toks, "admitted_requests": asked,  # noqa: E731
                                                         "gameplay_requests": game, "noaction_requests": noact,
                                                         "rollout": {"live": live, "capped": capped}}
    base = loop.mg_play_stats([row(2000, 10, 10, 0, 10, 0), row(4000, 10, 10, 1, 10, 1)])
    assert base == {"tries": 2, "filler_k": None, "reply_tokens": 300.0, "cap_share": 0.05, "noaction_share": 0.05}
    same = loop.mg_play_stats([row(3000, 10, 10, 1, 10, 1)])
    assert loop.mg_broken(same, base) is None
    longer = loop.mg_play_stats([row(6100, 10, 10, 0, 10, 0)])
    assert "reply tokens" in loop.mg_broken(longer, base)[0]
    capped = loop.mg_play_stats([row(3000, 10, 10, 2, 10, 0)])  # 0.2 > 2 x 0.05 and > 0.05 + 0.05
    assert loop.mg_broken(capped, base) == ["cap_share 0.2 vs base 0.05"]
    small = {"tries": 1, "reply_tokens": 300.0, "cap_share": 0.09, "noaction_share": 0.0}  # 2x but under +0.05
    assert loop.mg_broken(small, {**base, "cap_share": 0.04}) is None


def test_breakage_rollback_restores_the_earlier_served_state(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, mg={"kl_target": 0.08, "breakage": {}})
    loop.work = tmp_path
    events = []

    async def sleep():
        events.append("sleep")

    async def unload(name):
        events.append(("unload", name))

    async def card_play():
        events.append("card")
        return {}

    loop.server.sleep, loop.server.unload, loop.mg_card_play = sleep, unload, card_play
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop.mg_served, loop.keep_checkpoint, loop.serving_adapter = [("r00", 0), ("r02", 2), ("r03", 3)][-2:], "r03", "a3"
    row = {"breakage": ["reply tokens 900 vs base 300"]}
    assert asyncio.run(loop.mg_break_rollback(4, row, loop.mg_probes)) == 2
    ops = loop.trainer.ops()
    assert ops == ["load_checkpoint", "score", "accept_reference", "save"] and loop.trainer.calls[0][1]["path"].endswith("state/r02")
    assert loop.serving_adapter.endswith("back-r04") and loop.server.loads[-1] == ("policy", loop.serving_adapter)
    assert loop.keep_checkpoint == "r02" and loop.mg_served == [("r02", 2)] and loop.mg_step_rollbacks == 1
    # one state left: back to it is impossible, the caller stops
    assert asyncio.run(loop.mg_break_rollback(5, row, loop.mg_probes)) is None
    # back to the base (r00): nothing saved, the policy adapter unloaded
    loop.mg_served, loop.serving_adapter, loop.trainer = [("r00", 0), ("r02", 2)], "a2", FakeTrainer()
    assert asyncio.run(loop.mg_break_rollback(3, row, loop.mg_probes)) == 0
    assert "save" not in loop.trainer.ops() and loop.serving_adapter is None and events[-1] == ("unload", "policy")


def test_bench_and_probe_clients_retire_idle_connections_before_the_server_keepalive():
    import re

    from arc_infra import KEEPALIVE_EXPIRY

    assert KEEPALIVE_EXPIRY < 5.0  # vLLM / uvicorn close idle connections at 5 s
    for name in ("level_bench.py", "cache_probe.py"):
        src = (Path(rl_loop.__file__).parent / name).read_text()
        made = re.findall(r"httpx\.AsyncClient\((.*?)\) as client|httpx\.AsyncClient\((.*?)\)\n", src, re.S)
        assert made and src.count("httpx.Limits(") == src.count("keepalive_expiry=KEEPALIVE_EXPIRY"), name


def test_stall_watch_dumps_await_chains_of_a_silent_loop(tmp_path):
    import time as _t

    feed = rl_loop.Feed(tmp_path / "rl")
    feed.emit("start")

    async def inner_wait(event):
        await event.wait()

    async def stuck_bench(event):
        await inner_wait(event)

    async def main():
        event = asyncio.Event()
        task = asyncio.create_task(stuck_bench(event))
        rl_loop.StallWatch(feed, asyncio.get_running_loop(), 0.3, False, tmp_path / "out")
        for _ in range(60):
            await asyncio.sleep(0.05)
            if '"stall_dump"' in (tmp_path / "rl" / "feed.jsonl").read_text():  # emitted once the dump is written
                break
        event.set()
        await task

    asyncio.run(main())
    dumps = list((tmp_path / "out").glob("stall-dump-*.txt"))
    assert dumps
    text = dumps[0].read_text()
    assert "stuck_bench" in text and "inner_wait" in text and "awaits" in text and "== processes" in text
    assert json.loads((tmp_path / "rl" / "feed.jsonl").read_text().splitlines()[-1])["kind"] == "stall_dump"


def test_replay_stub_listens_with_a_deep_backlog():
    from arena import replay_stub

    assert replay_stub.StubServer.request_queue_size >= 1024
    assert "StubServer((HOST" in (Path(replay_stub.__file__)).read_text()


def _carry_source(tmp_path, rows):
    folder = tmp_path / "src" / "arena" / "trials" / "rl-loop" / "rl" / "r04-sc25-L2"
    folder.mkdir(parents=True)
    (folder / "level-bench.json").write_text(json.dumps({"attempts": rows}))
    return tmp_path / "src", folder


def _carry_row(i, outcome="cleared", tokens=400):
    return {"label": f"a{i:03d}-sc25-game-L2-x-k1-p{i}", "game": "sc25-635fd71a", "point": "sc25-game-L2-x-k1",
            "outcome": outcome, "restore": {"ok": True}, "admitted_requests": 10, "completion_tokens": 10 * tokens,
            "gameplay_requests": 10, "noaction_requests": 1, "rollout": {"live": 10, "capped": 0}}


def test_carry_trains_the_sources_final_round_tries(tmp_path):
    loop = make_loop(tmp_path, mg={"carry": True, "breakage": {"reply_ratio": 2.0, "ratio": 2.0, "abs": 0.05}})
    rows = [_carry_row(0), _carry_row(1, "budget"), _carry_row(2), _carry_row(3), _carry_row(4, "replay_mismatch")]
    src, folder = _carry_source(tmp_path, rows)
    loop.cfg["mg"]["resume"] = str(src)
    loop.mg_base_play = {"tries": 3, "reply_tokens": 400.0, "cap_share": 0.0, "noaction_share": 0.1}
    sets = {"sc25-L2": {"name": "sc25-L2", "game": "sc25", "role": "train"},
            "cn04-L1": {"name": "cn04-L1", "game": "cn04", "role": "test"}}
    seen = {}

    def select(f, r, n):
        seen["select"] = (Path(f), [x["label"] for x in r], n)
        return [(Path(f), r[0], 1.0, "k"), (Path(f), r[1], -1.0, "k")], {"groups": 1, "mixed": 1, "trained": 2}

    async def update(number, new, probes):
        seen["update"] = (number, len(new), new[0][-1])
        return {"ok": True}, "/adapters/u04"

    async def after(number, row, update_row, serve, load=True):
        seen["after"] = (number, serve, load)
        return None, True

    loop.mg_select, loop.mg_update, loop.mg_after_update = select, update, after
    stop, served, row = asyncio.run(loop.mg_carry(4, sets, loop.mg_probes))
    assert stop is None and served and row["round"] == 4 and row["played_by"] == 3
    assert seen["select"][0] == folder and len(seen["select"][1]) == 4 and seen["select"][2] == "sc25-L2"  # infra drop out
    assert seen["update"] == (4, 2, "sc25-L2") and seen["after"] == (4, "/adapters/u04", False)
    assert loop.feed.events[-1][0] == "carry" and not row.get("breakage")
    # a broken final round never trains
    loop.mg_base_play = {"tries": 3, "reply_tokens": 100.0, "cap_share": 0.0, "noaction_share": 0.1}
    seen.clear()
    stop, served, row = asyncio.run(loop.mg_carry(4, sets, loop.mg_probes))
    assert stop == "carry_breakage" and not served and "update" not in seen


def test_carry_refuses_without_the_sources_benches(tmp_path):
    loop = make_loop(tmp_path, mg={"carry": True})
    loop.cfg["mg"]["resume"] = str(tmp_path / "nowhere")
    with pytest.raises(RuntimeError, match="carry: no round-4"):
        asyncio.run(loop.mg_carry(4, {"sc25-L2": {"name": "sc25-L2", "game": "sc25", "role": "train"}}, []))


def test_carry_key_is_known_and_runs_only_after_a_read_only_round():
    rl_loop.check_mg({"carry": True})
    src = (Path(rl_loop.__file__).parent / "rl_single.py").read_text()
    assert 'mg.get("carry") and int(state.get("played_by", state["round"])) == int(state["round"]) - 1' in src


def test_c7f_breakage_absolute_cap_rise_and_clear_drop():
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = {"mg": {"breakage": {"reply_ratio": 2.0, "ratio": 2.0, "abs": 0.05, "cap_abs": 0.10, "clear_drop": 0.25}}}
    base = {"tries": 96, "reply_tokens": 437.4, "cap_share": 0.0, "noaction_share": 0.41}
    c_r3 = {"tries": 96, "reply_tokens": 968.0, "cap_share": 0.1733, "noaction_share": 0.48}  # pilot c, R3
    assert loop.mg_broken(c_r3, base, 0.625) == ["reply tokens 968.0 vs base 437.4", "cap_share 0.1733 vs base 0.0",
                                                 "held-out clear rate down 0.625"]
    assert loop.mg_broken({**base, "cap_share": 0.09}, base) is None  # +9 points: under the absolute line
    assert loop.mg_broken({**base, "cap_share": 0.11}, base) == ["cap_share 0.11 vs base 0.0"]
    assert loop.mg_broken(base, base, 0.25) is None and loop.mg_broken(base, base, 0.26) == ["held-out clear rate down 0.26"]


def test_c7f_broken_round_rolls_back_before_the_guard_stops():
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = {"mg": {"breakage": {"rollback_first": True}}}
    broken = {"behaviour": {"stop": "behaviour_guard"}, "breakage": ["held-out clear rate down 0.625"]}
    assert loop.mg_guard_stop(broken) is None
    assert loop.mg_guard_stop({"behaviour": {"stop": "behaviour_guard"}}) == "behaviour_guard"  # slow decline: the guard
    loop.cfg["mg"]["breakage"] = {}
    assert loop.mg_guard_stop(broken) == "behaviour_guard"  # pilots' bundles: unchanged


def test_c7f_early_warning():
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = {"mg": {"warn": {"ratio": 0.7, "reply_ratio": 1.5}}}
    loop.mg_base_play = {"reply_tokens": 437.4}
    c_r2 = {"shortening": {"sc25-L1": {"ratio": 1.301}, "sc25-L2": {"ratio": 0.929}, "sc25-L3": {"ratio": 0.616}},
            "play_stats": {"reply_tokens": 454.0}}
    assert loop.mg_warning(c_r2) == ["sc25-L3 token ratio 0.616"]
    a_r2 = {"shortening": {"sc25-L1": {"ratio": 0.854}, "sc25-L3": {"ratio": 0.87}, "sc25-L2": {"ratio": None}},
            "play_stats": {"reply_tokens": 429.0}}
    assert loop.mg_warning(a_r2) is None
    assert loop.mg_warning({"play_stats": {"reply_tokens": 700.0}}) == ["reply tokens 700.0 vs base 437.4"]


def test_carry_warning_halves_the_carried_update(tmp_path):
    loop = make_loop(tmp_path, mg={"carry": True, "breakage": {"reply_ratio": 2.0, "ratio": 2.0, "abs": 0.05},
                                   "warn": {"ratio": 0.7, "reply_ratio": 1.5}})
    rows = [_carry_row(0), _carry_row(1, "budget"), _carry_row(2), _carry_row(3)]
    src, folder = _carry_source(tmp_path, rows)
    loop.cfg["mg"]["resume"] = str(src)
    loop.mg_base_play = {"tries": 3, "reply_tokens": 400.0, "cap_share": 0.0, "noaction_share": 0.1}
    loop.mg_shortening = lambda number: {"sc25-L2": {"ratio": 0.6}}
    loop.mg_fixed = {4: {"sc25-L2": []}}
    sets = {"sc25-L2": {"name": "sc25-L2", "game": "sc25", "role": "train"}}
    seen = {}
    loop.mg_select = lambda f, r, n: ([(Path(f), r[0], 1.0, "k"), (Path(f), r[1], -1.0, "k")], {"groups": 1})

    async def update(number, new, probes):
        seen["lr"] = loop.cfg["lr"]
        return {"ok": True}, "/adapters/u04"

    async def after(number, row, update_row, serve, load=True):
        return None, True

    loop.mg_update, loop.mg_after_update = update, after
    stop, served, row = asyncio.run(loop.mg_carry(4, sets, loop.mg_probes))
    assert row["warning"] == ["sc25-L2 token ratio 0.6"] and seen["lr"] == pytest.approx(2.5e-5)


def test_reward_rhae_floor_scores_a_ruined_clear_near_a_fail():
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    lo, fail = math.log(8), -math.log(16)
    loop.cfg = {"mg": {}}
    spam = math.log(min(1.15, (36 / 283) ** 2) / min(1.15, (36 / 41.5) ** 2))  # pilot a, sc25 L1 at R4
    assert loop.mg_log_terms(spam, math.log(25370.5 / 15829), lo, fail) == pytest.approx(-lo)  # old: above a fail
    loop.cfg["mg"]["log_rhae_floor"] = True
    assert loop.mg_log_terms(spam, math.log(25370.5 / 15829), lo, fail) == pytest.approx(fail + math.log(25370.5 / 15829))
    assert loop.mg_log_terms(spam, -0.5, lo, fail) == pytest.approx(fail)  # slow too: a fail, never below
    assert loop.mg_log_terms(0.2, 0.3, lo, fail) == pytest.approx(0.5)  # ordinary clears unchanged
    assert loop.mg_log_terms(1.5, 3.0, lo, fail) == pytest.approx(lo)  # the top stays at +log clip


def test_actions_guard_warns_then_breaks():
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = {"mg": {"warn": {"ratio": 0.7, "reply_ratio": 1.5, "actions": 0.3},
                       "breakage": {"reply_ratio": 2.0, "ratio": 2.0, "abs": 0.05, "actions_rise": 1.0}}}
    loop.mg_base_play = {"reply_tokens": 470.6}
    a_r2 = {"behaviour": {"actions_rise": 0.289}, "play_stats": {"reply_tokens": 411.1}}
    a_r3 = {"behaviour": {"actions_rise": 0.48}, "play_stats": {"reply_tokens": 272.2}}
    assert loop.mg_warning(a_r2) is None and loop.mg_warning(a_r3) == ["actions per clear up 0.48"]
    base = {"reply_tokens": 470.6, "cap_share": 0.0, "noaction_share": 0.4}
    now = {"reply_tokens": 243.6, "cap_share": 0.0, "noaction_share": 0.4}
    assert loop.mg_broken(now, base, -0.042, 0.48) is None
    assert loop.mg_broken(now, base, -0.042, 2.38) == ["held-out actions per clear up 2.38"]  # pilot a, R4


def test_warm_start_rounds_are_policy_rounds():
    """mg.init_state (next-board warm start): round 1 is a policy round against mg.base_ref's base rows; a breakage there
    stops the run (no rollback past the init state); a later rollback to r00 serves the init state, not the base."""
    from pathlib import Path

    from arena.rl_single import MultiMixin

    src = Path(MultiMixin.__module__.replace(".", "/") + ".py")
    src = (Path(__file__).resolve().parents[1] / "src" / src).read_text()
    obj = MultiMixin.__new__(MultiMixin)
    assert not obj.mg_policy_round(0) and obj.mg_policy_round(1)
    obj.mg_init = True
    assert obj.mg_policy_round(0)
    assert 'rule = "init_breakage"' in src
    assert 'if back > 0 or getattr(self, "mg_init", False):' in src
    assert 'if read_by == 0 and not self.mg_init and fixed_rows:' in src
    assert "extra = (start_seeded() if seed else start()) if number == 1 and not started else None" in src


def test_final_pair_key_ignores_the_attempt_path_prefix():
    from arena.rl_single import final_pair_key

    def row(source, request, label, point):
        return {"restore": {"source": source, "request": request}, "label": label, "point": point}

    screen = row("/tmp/arena/trials/rl-loop/rl/rec-sc25/attempts/a015-sc25-game-p15", 23, "a003-sc25-game-L4-aaaaaa-k023-p1", "x-aaaaaa")
    policy = row("/runs/validation/rl1-screen-sc25ref-all/arena/trials/rl-loop/rl/rec-sc25/attempts/a015-sc25-game-p15", 23,
                 "a007-sc25-game-L4-bbbbbb-k023-p1", "x-bbbbbb")
    other_seed = {**policy, "label": "a008-sc25-game-L4-bbbbbb-k023-p2"}
    assert final_pair_key(screen) == final_pair_key(policy) != final_pair_key(other_seed)
    fresh = {"label": "a000-sc25-game-L1-sc25-L1-p3", "point": "sc25-game-L1-sc25-L1"}
    assert final_pair_key(fresh) == ("sc25-game-L1-sc25-L1", "3")


# --- pilot 1 (20 Oct): lr schedule, dose on the peak, filler guard, reply filter -------------------------------------

def test_lr_schedule_warmup_decay_and_resume_ramp(tmp_path):
    loop = make_loop(tmp_path, mg={"lr_schedule": {"warmup": 4, "total": 12, "end": 0.5, "start": 0.25}}, lr=1e-4)
    mults = []
    for update in range(3):
        for index in range(4):
            mults.append(loop.step_lr(index)[1])
            loop.proc_steps = getattr(loop, "proc_steps", 0) + 1  # as dosed_update counts sent steps
        loop.mg_steps_done = getattr(loop, "mg_steps_done", 0) + 4
    assert mults == pytest.approx([0.25, 0.5, 0.75, 1.0, 0.9375, 0.875, 0.8125, 0.75, 0.6875, 0.625, 0.5625, 0.5])
    assert loop.step_lr(0)[0] == pytest.approx(0.5e-4)  # past the end: stays at end
    resumed = make_loop(tmp_path, mg={"lr_schedule": {"warmup": 4, "total": 12, "end": 0.5}}, lr=1e-4)
    resumed.mg_steps_done = 4  # a restart after U1: warm-up again on top of the decay
    got = []
    for i in range(4):
        got.append(resumed.step_lr(i)[1])
        resumed.proc_steps = i + 1
    assert got == pytest.approx([0.25 * 0.9375, 0.5 * 0.875, 0.75 * 0.8125, 0.75])
    plain = make_loop(tmp_path, lr=3e-5)
    assert plain.step_lr(2) == (3e-5, 1.0)


def test_dosed_update_sends_scheduled_lr_and_reports_the_mean(tmp_path):
    loop = make_loop(tmp_path, mg={"lr_schedule": {"warmup": 4, "total": 12, "end": 0.5}}, lr=1e-4,
                     steps=[(0.02, 0.02)] * 4)
    loop.cfg["steps_per_round"] = 4
    batches = [[{"name": f"t{i}", "w": 1.0}] for i in range(4)]
    merged, _ = asyncio.run(loop.dosed_update(1, None, None, None, loop.mg_probes, batches=batches, signed=True,
                                              early=loop.mg_early()))
    sent = [p["lr"] for op, p in loop.trainer.calls if op == "update"]
    assert sent == pytest.approx([2.5e-5, 5e-5, 7.5e-5, 1e-4])
    assert merged["lr_effective"] == pytest.approx(6.25e-5) and [s["lr_mult"] for s in merged["steps"]] == [0.25, 0.5, 0.75, 1.0]


def test_dose_rule_sets_the_peak_from_the_effective_lr(tmp_path):
    loop = make_loop(tmp_path, mg={"lr_dose": True, "kl_damage": 0.1, "lr_max": 1e-3}, lr=1e-4)
    loop.cfg["l2_kl_stop"] = None
    # an update at mean multiplier 0.625 landed exactly on target: at multiplier 1 the peak must be 0.625 x
    row = {"ok": True, "early": {"action": "full", "steps": 4, "excess": 0.01}, "probes": {}, "lr_effective": 6.25e-5,
           "steps": [{}] * 4}
    stop, served = asyncio.run(loop.mg_after_update(2, {}, row, "a2", load=False))
    assert served and loop.cfg["lr"] == pytest.approx(6.25e-5) and loop.mg_steps_done == 4
    loop.cfg["lr"] = 1e-4  # a quarter of the target at the same effective lr: x2 on it
    r = {}
    loop.mg_kl_target_control(r, {**row, "early": {**row["early"], "excess": 0.0025}})
    assert loop.cfg["lr"] == pytest.approx(1.25e-4) and r["kl_step"]["lr_effective"] == 6.25e-5


def test_filler_guard_trips_pilot_c_r2_only():
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.cfg = {"mg": {"breakage": {"reply_ratio": 2.0, "ratio": 2.0, "abs": 0.05, "cap_abs": 0.05, "filler_ratio": 1.5}}}
    row = lambda fill, think: {"completion_tokens": 100, "admitted_requests": 1, "gameplay_requests": 1,  # noqa: E731
                               "noaction_requests": 0, "rollout": {"live": 1, "capped": 0, "filler": fill, "think_tokens": think}}
    base = loop.mg_play_stats([row(777, 100000)])
    assert base["filler_k"] == 7.77  # pilot c R1 train
    assert loop.mg_broken(loop.mg_play_stats([row(1232, 100000)]), base) == ["filler 12.32 per 1K thinking tokens vs base 7.77"]
    for healthy in (881, 685, 606, 266, 44):  # d R2, b R2, a R2, a R3, a R4 (per 1K x 100)
        assert loop.mg_broken(loop.mg_play_stats([row(healthy, 100000)]), base) is None
    assert loop.mg_broken({**base, "cap_share": 0.06}, base) == ["cap_share 0.06 vs base 0.0"]
    assert loop.mg_broken(loop.mg_play_stats([row(0, 0)]), base) is None  # no counts: no filler reading


def test_filler_counts_hmm_before_think_end(monkeypatch):
    from arena import filler, rollout_record

    toks = [1, 85152, 2, 84485, 77264, filler.THINK_END, 85152]
    assert filler.reply_filler(toks) == (3, 5)
    monkeypatch.delenv("ARC_FILLER", raising=False)
    assert rollout_record.filler_count(toks) is None
    monkeypatch.setenv("ARC_FILLER", "1")
    assert rollout_record.filler_count(toks) == (3, 5)


def test_filler_classes_from_a_byte_level_vocab(tmp_path):
    from arena import filler

    dec = {b: c for c, b in filler.byte_decoder().items()}
    enc = lambda text: "".join(dec[b] for b in text.encode())  # noqa: E731
    vocab = {enc("Hmm"): 5, enc(" hmm"): 6, enc("hm,"): 7, enc("ham"): 8, enc(" Hmmm"): 9, enc("hmmmmm"): 10, enc("h"): 11}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": [{"id": 99, "content": "</think>"}]}))
    assert filler.classes(path) == (frozenset({5, 6, 7, 9, 10}), 99)
    assert filler.check(path)["same"] is False


def write_replies(folder, label, replies):
    """replies: [(finish, tokens list)]."""
    att = folder / "attempts" / label
    att.mkdir(parents=True)
    with open(att / "rollout.jsonl", "w") as f:
        for n, (finish, toks) in enumerate(replies):
            f.write(json.dumps({"n": n, "step": n, "finish_reason": finish, "tokens": toks,
                                "usage": {"completion_tokens": len(toks)}}) + "\n")
    (att / "log.jsonl").write_text("")


def test_reply_filter_zeroes_positive_capped_and_loop_replies_with_audit(tmp_path):
    from arena import filler

    loop = mg6_loop(tmp_path, reward_shape="logeq", turn_credit={"anchor": 1.0, "token": 0.0, "clip": 5.0}, balance=True,
                    reply_filter={"capped": True, "hmm_min": 50, "hmm_density": 0.1})
    hmm = next(iter(filler.HMM_IDS))
    looped = [hmm] * 60 + [1] * 40 + [filler.THINK_END, 2]  # 60 hmm in 100 thinking tokens
    write_replies(tmp_path, "A", [("tool_calls", [1] * 30), ("length", [3] * 50), ("tool_calls", looped)])
    write_replies(tmp_path, "B", [("tool_calls", [1] * 20), ("length", [3] * 40)])
    grp = [try_row("A", 1500, 4), try_row("B", 35000, 4, cleared=False)]
    rewards = [loop.mg_reward(r, "x-L2") for r in grp]
    info = loop.mg_turn_credit(tmp_path, grp, rewards, "x-L2")
    assert {n: f["kind"] for n, f in info[0]["filtered"].items()} == {1: "capped", 2: "loop"}
    assert info[0]["filtered"][2]["think"] == 100 and set(info[1]["filtered"]) == {1}
    turn = {(str(tmp_path), "A"): info[0], (str(tmp_path), "B"): info[1]}
    specs = [{"w": 1.0, "turn_key": (str(tmp_path), "A"), "set": "x-L2"}, {"w": -1.0, "turn_key": (str(tmp_path), "B"), "set": "x-L2"}]
    out = loop.mg_reply_balance(specs, turn)
    assert specs[0]["reply_w"]["1"] == 0.0 and specs[0]["reply_w"]["2"] == 0.0 and specs[0]["reply_w"]["0"] > 0
    assert specs[1]["reply_w"]["1"] < 0  # a negative reply keeps its push-down
    audit = out["reply_filter"]
    assert audit["replies"] == 2 and audit["kinds"] == {"capped": 1, "loop": 1} and audit["positive_tokens"] == 30 + 50 + 102
    assert audit["tokens"] == 152 and audit["think_tokens"] == 50 + 100 and audit["answer_tokens"] == 2
    assert loop.feed.events[-1][0] == "reply_filter"


def test_reply_flag_healthy_reply_passes():
    from arena import filler

    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    hmm = next(iter(filler.HMM_IDS))
    rf = {"capped": True, "hmm_min": 50, "hmm_density": 0.1}
    assert loop.mg_reply_flag({"finish_reason": "tool_calls", "tokens": [hmm] * 89 + [1] * 4900}, rf) is None  # 0.018
    assert loop.mg_reply_flag({"finish_reason": "tool_calls", "tokens": [hmm] * 49 + [1] * 10}, rf) is None  # under 50


def test_train_sampling_reaches_training_benches_only():
    from arena import rl_loop

    rl_loop.check_mg({"train_sampling": {"temperature": 1.0}})
    for bad in ({"train_sampling": {"temperature": "1"}}, {"train_sampling": {"seed": 1}}, {"train_sampling": {}}):
        with pytest.raises(ValueError):
            rl_loop.check_mg(bad)
    src = (Path(rl_loop.__file__).with_name("rl_single.py")).read_text()
    assert src.count('sampling=mg.get("train_sampling")') == 2  # the training plays and branch groups; fixed reads keep the study's
    assert src.count("sampling=train_sampling") == 1  # the curriculum's training plays


# --- ramp rule --------------------------------------------------------------------------------------

def test_ramp_rule_rises_when_quiet_falls_back_and_holds_on_trip(tmp_path):
    loop = make_loop(tmp_path, mg={"kl_ramp": {"start": 0.01, "factor": 1.5, "cap": 0.05, "hold": 2}, "kl_target": 0.01})
    quiet = lambda c="50/72": {"train": {"a": {"clears": c}}}
    seen = []
    for row, early in [(quiet(), {"action": "full", "excess": 0.01}),  # U1 at 0.01 -> 0.015
                       (quiet(), {"action": "full", "excess": 0.015}),  # -> 0.0225
                       ({**quiet(), "breakage": {"why": "filler"}}, {"action": "full"}),  # trip: back to safe 0.015
                       (quiet(), {"action": "full"}), (quiet(), {"action": "full"}),  # held two quiet rounds
                       (quiet(), {"action": "full"})]:  # then rises again
        loop.mg_ramp(row, early)
        seen.append((row["kl_ramp"]["target_used"], row["kl_ramp"]["trip"], round(loop.cfg["mg"]["kl_target"], 5)))
    assert [s[2] for s in seen] == [0.015, 0.0225, 0.015, 0.015, 0.015, 0.0225]
    assert seen[2][1] == "breakage"


def test_dose_predicts_k_growth_and_ramp_holds_after_overshoot(tmp_path):
    """21 Oct (mra-v2: k = excess / lr^2 doubled each update): kl_k_predict aims with the predicted k; an update
    above overshoot x its target holds the ramp's target."""
    loop = make_loop(tmp_path, mg={"lr_dose": True, "kl_target": 0.02, "lr_max": 1e-3, "kl_k_predict": 2.5}, lr=2.76e-5)
    loop.mg_last_dose = (2.52e-5, 0.01254)  # U4: k 1.97e7
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.0315}, "lr_effective": 2.76e-5})  # U5: k 4.1e7
    growth = (0.0315 / 2.76e-5 ** 2) / (0.01254 / 2.52e-5 ** 2)
    assert math.isclose(loop.cfg["lr"], 2.76e-5 * math.sqrt(0.02 / (0.0315 * growth)))
    assert loop.mg_last_dose == (2.76e-5, 0.0315)
    loop2 = make_loop(tmp_path, mg={"lr_dose": True, "kl_target": 0.02, "lr_max": 1e-3}, lr=2.76e-5)  # off: last k only
    loop2.mg_last_dose = (2.52e-5, 0.01254)
    loop2.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.0315}, "lr_effective": 2.76e-5})
    assert math.isclose(loop2.cfg["lr"], 2.76e-5 * math.sqrt(0.02 / 0.0315))
    # k falling: growth clamps at 1 (no boost)
    loop.cfg["lr"] = 2e-5
    loop.mg_last_dose = (2e-5, 0.04)
    loop.mg_kl_target_control({}, {"early": {"action": "full", "excess": 0.01}, "lr_effective": 2e-5})
    assert math.isclose(loop.cfg["lr"], 2e-5 * math.sqrt(0.02 / 0.01))
    ramp = make_loop(tmp_path, mg={"kl_ramp": {"start": 0.015, "factor": 1.5, "cap": 0.05, "overshoot": 1.5}, "kl_target": 0.015})
    row = {"train": {"a": {"clears": "60/72"}}}
    ramp.mg_ramp(row, {"action": "full", "excess": 0.0315})  # 2.1x over: hold at 0.015
    assert row["kl_ramp"]["trip"] is None and ramp.cfg["mg"]["kl_target"] == 0.015
    row = {"train": {"a": {"clears": "60/72"}}}
    ramp.mg_ramp(row, {"action": "full", "excess": 0.02})  # within 1.5x: rises
    assert math.isclose(ramp.cfg["mg"]["kl_target"], 0.0225)


def test_mg_loop_reads_the_stop_request():
    src = (Path(rl_loop.__file__).with_name("rl_single.py")).read_text()
    assert src.count('stop = "manual"') >= 1 and 'os.environ.get("ARENA_STOP_REQUEST")' in src


def test_ramp_rule_caps_and_trips_on_kl_damage_and_clear_drop(tmp_path):
    loop = make_loop(tmp_path, mg={"kl_ramp": {"start": 0.04, "cap": 0.05}, "kl_target": 0.04})
    row = {"train": {"a": {"clears": "60/72"}}}
    loop.mg_ramp(row, {"action": "full"})
    assert loop.cfg["mg"]["kl_target"] == 0.05
    row = {"train": {"a": {"clears": "60/72"}}}
    loop.mg_ramp(row, {"action": "rollback"})
    assert row["kl_ramp"]["trip"] == "kl_damage" and loop.cfg["mg"]["kl_target"] == 0.04
    t = rl_loop.Loop.ramp_trip
    assert t({"train": {"a": {"clears": "40/72"}}}, {}, (60, 72)).startswith("clear_drop")  # 0.83 -> 0.56
    assert t({"train": {"a": {"clears": "56/72"}}}, {}, (60, 72)) is None  # within noise


def test_reload_check_passes_and_stops_on_mismatch(tmp_path, monkeypatch):
    """Two scores before the reload give the repeat floor; the reload must stay within 3x it + 2e-4 (ceiling 5e-3)."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    loop = make_loop(tmp_path, mg={"reload_check": True})
    calls = []

    def trainer(saved, reloaded):
        state = {"loaded": False}

        async def must(op, timeout=3600, **payload):
            calls.append((op, payload))
            if op == "load_checkpoint":
                state["loaded"] = True
                return {"updates": 1}
            v = reloaded if state["loaded"] else saved
            return {"windows": {"p1": {"kl_base": v}, "p2": {"kl_base": 0.03}}}
        return must

    merged = {"probes": {"p1": {"kl_base": 0.02}, "p2": {"kl_base": 0.03}}}
    loop.trainer.must = trainer(0.02, 0.0201)
    asyncio.run(loop.mg_reload_check(1, merged, ["w"]))
    assert [op for op, _ in calls] == ["score", "score", "load_checkpoint", "score"]
    assert calls[2][1]["path"].endswith("state/r01")
    assert calls[3][1] == {"windows": ["w"], "label": "reload", "hold_previous": True}
    row = dict(loop.feed.events)["reload_check"]
    assert row["ok"] and row["compared"] == 2 and row["repeat_floor"] == 0.0
    loop.trainer.must = trainer(0.02, 0.025)  # exact repeats: a 5e-3 reload shift is a real mismatch
    with pytest.raises(RuntimeError, match="reload check failed"):
        asyncio.run(loop.mg_reload_check(1, merged, ["w"]))
    assert ("alert" in [k for k, _ in loop.feed.events])


def test_turn_credit_ends_at_the_clear_not_at_later_observations(tmp_path):
    loop = mg6_loop(tmp_path, turn_credit={"anchor": 1.0, "token": 0.0, "clip": 5.0})
    write_try(tmp_path, "A", [(0, 500), (2, 1000)], {0: "0f0f", 2: "0a0a", 4: "0c0c"})
    write_try(tmp_path, "B", [(0, 500), (2, 1000)], {0: "0f0f", 2: "0a0a", 4: "0c0c", 9: "0e0e"})
    grp = [try_row("A", 1500, 4), try_row("B", 1500, 4)]
    before = loop.mg_turn_credit(tmp_path, grp, [1.0, 1.0], "x-L2")
    assert before[0]["extra"][1] > 0 > before[1]["extra"][1]  # without a clear mark B's try ends at step 9
    with open(tmp_path / "attempts/B/log.jsonl", "a") as f:  # a harvested try: the clear at 4, the next level after
        f.write(json.dumps({"kind": "level_bench_clear", "step": 4}, separators=(",", ":")) + "\n")
    after = loop.mg_turn_credit(tmp_path, grp, [1.0, 1.0], "x-L2")
    assert after[0]["extra"][1] == pytest.approx(0.0) and after[1]["extra"][1] == pytest.approx(0.0)


def test_curriculum_plans_the_round_and_learns_from_it(tmp_path):
    import asyncio

    loop = mg6_loop(tmp_path, reward_shape="logeq", train_starts=2, train_tries=4,
                    curriculum={"groups": 4, "group_size": 4, "probe_share": 0.0, "seed": 1})
    sets = {"x-L1": {"name": "x-L1", "game": "xgam", "level": 1, "role": "train", "fresh": True, "cap": 9000},
            "x-L2": {"name": "x-L2", "game": "xgam", "level": 2, "role": "train", "cap": 8000},
            "t": {"name": "t", "game": "ygam", "level": 1, "role": "test", "fresh": True}}
    loop.cfg["mg_sets"] = sets
    pool = {"x-L2": [(f"/rec/a{i}", 10 + i) for i in range(6)]}
    screen = {"sets": {"x-L1": {"metrics": {"tries": 8, "clears": 4, "median_to_clear": 9000}},
                       "x-L2": {"metrics": {"tries": 8, "clears": 8, "median_to_clear": 10000, "median_actions": 30}}}}
    loop.mg_screen = screen
    loop.mg_curr = loop.mg_curriculum_init(sets, pool, screen)
    assert set(loop.mg_unit_set.values()) == {"x-L1", "x-L2"}
    played = []

    async def fake_play(name, spec, st, tries, **kw):
        played.append((name, len(st), tries, kw.get("harvest")))
        return name

    loop.mg_play = fake_play
    loop.mg_round_starts = {}
    jobs, names = loop.mg_curriculum_jobs(1, "policy", sets, pool)
    async def run_all():
        return await asyncio.gather(*jobs)

    asyncio.run(run_all())
    total = sum((n_st or 1) * t if n_st else t for _, n_st, t, _ in played)
    assert sorted(names) == sorted({p[0][4:] for p in played}) and total == 16  # 4 groups x 4 tries
    rows = {"x-L2": (tmp_path, [try_row(f"a{i}-p{i}", 6000 + 1000 * i, 30, point=f"xgam-game-L2-s{i % 2}-k1")
                                for i in range(4)], {}),
            "x-L1": (tmp_path, [try_row(f"b{i}-p{i}", 9000, 30, cleared=i < 2, point="xgam-game-L1-x") for i in range(4)],
                     {})}
    out = loop.mg_curriculum_observe(rows, sets)
    assert out["x-L2"]["spread"] > 0 and loop.mg_curr.units["xgam@L2"].seen == 2
    assert loop.mg_curr.units["xgam@L1"].tries == 4


def test_curriculum_multi_game_sources_refs_and_spent_pools(tmp_path):
    """envs 22 Oct: units carry their set's source/family (the mix), env-table refs stand in for a screen,
    a restored unit with no start left is not planned, and the round still fills its groups elsewhere."""
    import asyncio

    loop = mg6_loop(tmp_path, reward_shape="logeq", log_rhae_floor=True, rhae_cap=1.15,
                    curriculum={"groups": 4, "group_size": 3, "probe_share": 0.0, "seed": 2, "family_cap": 1.0,
                                "shares": {"pub": 0.5, "gen": 0.5}})
    loop.mg_point = lambda game: {"baselines": [20, 40]}
    loop.mg_screen = {"sets": {}}
    ref = {"tokens": 12000, "actions": 30}
    sets = {"a-L2": {"name": "a-L2", "game": "agam", "level": 2, "role": "train", "cap": 8000, "source": "pub",
                     "family": "fa", "ref": ref, "prior": {"tries": 6, "clears": 3}},
            "b-L2": {"name": "b-L2", "game": "bgam", "level": 2, "role": "train", "cap": 8000, "source": "gen",
                     "family": "fb", "ref": ref, "prior": {"tries": 6, "clears": 3}},
            "c-L1": {"name": "c-L1", "game": "cgam", "level": 1, "role": "train", "fresh": True, "cap": 9000,
                     "source": "gen", "family": "fc", "ref": ref, "prior": {"tries": 6, "clears": 3}}}
    loop.cfg["mg_sets"] = sets
    pool = {"a-L2": [("/rec/a0", 11)]}
    loop.mg_curr = loop.mg_curriculum_init(sets, pool, None)
    assert {u.source for u in loop.mg_curr.units.values()} == {"pub", "gen"}
    assert loop.mg_curr.units["agam@L2"].family == "fa" and loop.mg_curr.units["agam@L2"].pa == 2.0
    assert loop.mg_curr.units["agam@L2"].tok0 is None  # an unmarked env ref: the prior's median (none here)
    marked = {k: {**v, "ref": {**ref, "measured": True, "n": 3}, "prior": {**v["prior"], "median_to_clear": 64000}}
              for k, v in sets.items()}
    assert loop.mg_curriculum_init(marked, pool, None).units["agam@L2"].tok0 == 12000  # measured ref, not the 64K prior
    # the reward reads the set's env-table references (no screen): a faster clear scores above a slower one
    fast = try_row("f-p0", 6000, 30, point="agam-game-L2-a-k1")
    slow = try_row("s-p1", 24000, 30, point="agam-game-L2-a-k1")
    assert loop.mg_reward(fast, "a-L2") == pytest.approx(math.log(2.0))
    assert loop.mg_reward(slow, "a-L2") == pytest.approx(math.log(0.5))
    played = []

    async def fake_play(name, spec, st, tries, **kw):
        played.append((name, len(st), tries))
        return name

    loop.mg_play = fake_play
    loop.mg_round_starts = {}
    jobs, names = loop.mg_curriculum_jobs(1, "policy", sets, pool)
    asyncio.run(_gather_all(jobs))
    groups = {n[4:]: (st or t // 3) for n, st, t in played}
    assert "b-L2" not in groups and groups.get("a-L2", 0) <= 1  # b has no start; a has one
    assert sum(groups.values()) == 4  # the round is still full (c-L1 takes the rest)
    row = [e for k, e in loop.feed.events if k == "mg_curriculum"][-1]
    assert row["groups"] == 4 and set(row["by_source"]) <= {"pub", "gen"}
    # a set with no reference at all is refused at init
    bad = {**sets, "d-L1": {"name": "d-L1", "game": "dgam", "level": 1, "role": "train", "fresh": True, "cap": 9000}}
    loop.cfg["mg_sets"] = bad
    with pytest.raises(ValueError, match="lack reward references"):
        loop.mg_curriculum_init(bad, pool, None)


async def _gather_all(jobs):
    return await asyncio.gather(*jobs)



def test_in_place_reconfigure_applies_new_values_once_and_rolls_back_from_a_snapshot(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, mg={"max_rounds": 10, "updates": 9})
    loop.work = tmp_path
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    knobs = tmp_path / "knobs.json"
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    knobs.write_text(json.dumps({"lr": 2e-5, "kl_target": 0.015, "max_rounds": 14, "active_tries": 100}))
    assert asyncio.run(loop.mg_reconfigure(6, 5, loop.mg_probes)) is None
    assert loop.cfg["lr"] == 2e-5 and loop.cfg["mg"]["kl_target"] == 0.015 and loop.cfg["mg"]["updates"] == 13
    loop.cfg["lr"] = 3e-5  # the dose controller moved it: the unchanged knob does not undo that
    assert asyncio.run(loop.mg_reconfigure(7, 6, loop.mg_probes)) is None and loop.cfg["lr"] == 3e-5
    # rollback: refused without the state's snapshot, applied with it (counters, lr and target from the snapshot)
    (tmp_path / "state" / "r04").mkdir(parents=True)
    (tmp_path / "state" / "r04" / "state.json").write_text("{}")
    knobs.write_text(json.dumps({"lr": 2e-5, "rollback": {"to": "r04", "id": 1}}))
    assert asyncio.run(loop.mg_reconfigure(8, 7, loop.mg_probes)) is None
    assert [d for k, d in loop.feed.events if k == "alert"][-1]["why"] == "rollback refused"
    loop.keep_checkpoint, loop.mg_served = "r04", [("r04", 4)]
    loop.mg_accepted, loop.mg_steps_done = 3, 12
    loop.mg_write_state(4, 4)  # what the run writes when r04 becomes the served state
    assert json.loads((tmp_path / "state" / "r04" / "mg_state.json").read_text())["played_by"] == 4
    loop.mg_accepted, loop.mg_steps_done, loop.cfg["lr"], loop.keep_checkpoint = 6, 30, 9e-6, "r07"
    loop.mg_served = [("r04", 4), ("r06", 6), ("r07", 7)]
    knobs.write_text(json.dumps({"lr": 2e-5, "rollback": {"to": "r04", "id": 2}}))
    loop.trainer = FakeTrainer()
    assert asyncio.run(loop.mg_reconfigure(9, 7, loop.mg_probes)) == 4
    assert loop.trainer.ops() == ["load_checkpoint", "score", "accept_reference", "save"]
    assert loop.trainer.calls[0][1]["path"].endswith("state/r04") and loop.keep_checkpoint == "r04"
    assert loop.mg_served == [("r04", 4)] and (loop.mg_accepted, loop.mg_steps_done) == (3, 12)
    assert loop.server.loads[-1] == ("policy", loop.serving_adapter) and loop.mg_reload_checked is False
    # the snapshot's lr was 3e-5; the same knob file again changes nothing
    assert loop.cfg["lr"] == 3e-5 and asyncio.run(loop.mg_reconfigure(10, 4, loop.mg_probes)) is None
    # after_round: waits (not consumed) until the boundary after that round
    knobs.write_text(json.dumps({"lr": 2e-5, "rollback": {"to": "r04", "id": 9, "after_round": 12}}))
    assert asyncio.run(loop.mg_reconfigure(11, 4, loop.mg_probes)) is None and loop.feed.events[-1][0] != "rollback"
    assert "after_round" not in str(loop.mg_knobs_seen.get("rollback"))
    # a later state than the served one is refused
    knobs.write_text(json.dumps({"rollback": {"to": "r04", "id": 3}}))
    assert asyncio.run(loop.mg_reconfigure(11, 4, loop.mg_probes)) is None


def test_in_place_rollback_finds_a_state_in_an_earlier_chain_leg(tmp_path, monkeypatch):
    loop = make_loop(tmp_path / "leg2", mg={"max_rounds": 10, "updates": 9})
    loop.work = tmp_path / "leg2"
    leg1, leg2 = tmp_path / "leg1", tmp_path / "leg2"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(leg2))
    monkeypatch.setenv("ARC_RL1_RESUME", str(leg1))
    knobs = tmp_path / "knobs.json"
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    (leg1 / "state" / "r04").mkdir(parents=True)
    (leg1 / "state" / "r04" / "state.json").write_text("{}")
    (leg1 / "state" / "r04" / "mg_state.json").write_text(json.dumps({"checkpoint": str(leg1 / "state" / "r04"),
                                                                       "played_by": 4, "accepted": 3, "steps_done": 12}))
    loop.mg_served = [("r04", 4), ("r07", 7)]
    loop.trainer = FakeTrainer()
    knobs.write_text(json.dumps({"rollback": {"to": "r04", "id": 1}}))
    assert asyncio.run(loop.mg_reconfigure(8, 7, loop.mg_probes)) == 4
    assert loop.trainer.calls[0][1]["path"] == str(leg1 / "state" / "r04")
    assert [d for k, d in loop.feed.events if k == "rollback"][-1]["folder"] == str(leg1 / "state" / "r04")
    # this run's own copy, so its mg_state (and a restart after a handoff) points inside this run
    assert ("checkpoint", str(leg2 / "state" / "r04")) in [(op, kw.get("path")) for op, kw in loop.trainer.calls]


def test_boot_phases_and_park_record(tmp_path, monkeypatch):
    from arena.rl_single import boot_phases, park_record

    text = ("Loading weights took 177.00 seconds\nModel loading took 81.8 GiB memory and 330.5 seconds\n"
            "Dynamo bytecode transform time: 5.23 s\nGraph capturing finished in 28 secs, took 0.41 GiB\n"
            "init engine (profile, create kv cache, warmup model) took 153.00 s (compilation: 19.50 s)\n")
    out = boot_phases(text)
    assert (out["weights_s"], out["compile_s"], out["graphs_s"], out["engine_init_s"]) == (330.5, 19.5, 28.0, 153.0)
    monkeypatch.delenv("ARC_RL1_TRAINER_KEY", raising=False)
    assert park_record() is None
    monkeypatch.setenv("ARC_RL1_TRAINER_KEY", "ab" * 32)
    monkeypatch.setenv("ARC_PARK_RECORDS", str(tmp_path / "park"))
    assert park_record() == tmp_path / "park" / ("ab" * 16 + ".json")


def test_state_check_matches_the_recorded_probes_and_stops_on_a_mismatch(tmp_path):
    loop = make_loop(tmp_path)
    feed = tmp_path / "arena" / "trials" / "rl-loop" / "rl"
    feed.mkdir(parents=True)
    feed.joinpath("feed.jsonl").write_text(json.dumps({"kind": "round", "round": 5, "accepted_updates": 4, "update": {
        "probes": {"p1": {"kl_base": 0.0428}, "p2": {"kl_base": 0.031}}}}) + "\n")
    rec = loop.mg_recorded_probes(tmp_path, 5)
    assert rec == {"p1": 0.0428, "p2": 0.031} and loop.mg_recorded_probes(tmp_path, 4) is None
    ok = loop.mg_state_check("resume", rec, {"windows": {"p1": {"kl_base": 0.0429}, "p2": {"kl_base": 0.0309}}})
    assert ok["ok"] and ok["probes"] == 2
    assert loop.mg_state_check("resume", rec, {"windows": {}, "stand_in": True})["skipped"]
    with pytest.raises(RuntimeError):
        loop.mg_state_check("resume", rec, {"windows": {"p1": {"kl_base": 0.02}, "p2": {"kl_base": 0.031}}})


def test_logeq_select_logs_the_ratio_reward_shadow(tmp_path):
    loop = mg6_loop(tmp_path, reward_shape="logeq", rhae_cap=1.15, log_clip=8.0, log_fail=16.0)

    def row(label, tokens, actions, cleared=True):
        r = try_row(label, tokens, actions, cleared)
        if cleared:
            r["level_rows"][0]["score"] = 100.0 * min(1.15, (40 / actions) ** 2)
        return r

    rows = [row("a-p0", 8000, 40), row("b-p1", 12000, 40), row("c-p2", 30000, 40, cleared=False)]
    _, info = loop.mg_select(tmp_path, rows, "x-L2")
    sh = info["shadow"]
    assert sh["groups"] == 1 and sh["pooled_10k"] == pytest.approx(1e4 * 2.0 / 50000, abs=1e-4)
    assert sh["spearman"] == pytest.approx(1.0) and sh["flip_share"] == 0.0  # same order: a > b > fail
    # the next call prices tokens at this call's pooled rate
    _, info2 = loop.mg_select(tmp_path, rows, "x-L2")
    assert info2["shadow"]["rho_10k"] == sh["pooled_10k"]


def test_round_pack_writes_a_readable_archive(tmp_path, monkeypatch):
    import tarfile

    loop = make_loop(tmp_path)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    f = tmp_path / "play" / "r03-sc25-L1" / "attempts" / "a1"
    f.mkdir(parents=True)
    (f / "rollout.jsonl").write_text('{"n": 0}\n')
    loop.mg_pack_round(3, [f.parents[1], f.parents[1], tmp_path / "missing"])
    row = [d for k, d in loop.feed.events if k == "round_pack"][-1]
    assert row["round"] == 3 and row["folders"] == 2
    with tarfile.open(tmp_path / "out" / "rounds" / "r03.tar") as tar:
        assert tar.extractfile("r03-sc25-L1/attempts/a1/rollout.jsonl").read() == b'{"n": 0}\n'


def test_policy_uid_follows_the_served_adapter_and_a_restored_state():
    from arena.rl_single import state_uid

    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.serving_adapter = None
    assert loop.policy_uid == "base"
    loop.serving_adapter = "/a/r05"
    first = loop.policy_uid
    loop.serving_adapter = "/a/r05"  # served again (e.g. a retrained r05): a new policy, never a reuse
    assert loop.policy_uid != first and len(first) == 12
    state = {"round": 5, "checkpoint": "/runs/x/state/r05", "played_by": 5}
    assert state_uid(state) == state_uid(json.loads(json.dumps(state)))  # a pre-uid source: same uid on each resume
    assert state_uid({**state, "policy_uid": "abc"}) == "abc"


def test_try_keep_reuses_only_the_same_policys_tries_from_earlier_roots(tmp_path, monkeypatch):
    from arena import level_bench as lb

    out, died = tmp_path / "run", tmp_path / "run-lost-1"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(out))
    monkeypatch.setenv("ARC_RL1_RESUME", f"{tmp_path / 'src'},{died}")
    monkeypatch.setattr(lb, "TRY_KEEP", None)
    monkeypatch.setattr(lb, "TRY_REUSE", [])
    loop = make_loop(tmp_path, mg={"try_keep": True})
    loop.serving_adapter = "/w/resume-r05"
    loop.policy_uid = "u5"
    assert loop.mg_try_keep(5) == "p05-u5"
    assert lb.TRY_KEEP == out / "partial" / "p05-u5"
    assert lb.TRY_REUSE == [tmp_path / "src" / "partial" / "p05-u5", died / "partial" / "p05-u5"]

    def keep(root, key, label):
        d = root / "partial" / key / "r06-L1"
        (d / "attempts" / label).mkdir(parents=True)
        (d / "attempts" / label / "rollout.jsonl").write_text("{}\n")
        with (d / "tries.jsonl").open("a") as f:
            f.write(json.dumps({"label": label, "outcome": "cleared", "plan": "x"}) + "\n")

    keep(died, "p05-u5", "a000")  # played by this policy before the preemption
    keep(died, "p05-other", "a001")  # same played_by, other weights (a retrained r05)
    assert set(lb.kept_tries("r06-L1", lb.TRY_REUSE)) == {"a000"}
    loop.cfg["mg"]["try_keep"] = False
    lb.TRY_KEEP = None
    assert loop.mg_try_keep(5) is None and lb.TRY_KEEP is None


def test_resume_and_rollback_snapshots_carry_the_policy_uid(tmp_path, monkeypatch):
    loop = make_loop(tmp_path / "leg2", mg={"max_rounds": 10, "updates": 9})
    loop.work = tmp_path / "leg2"
    leg1, leg2 = tmp_path / "leg1", tmp_path / "leg2"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(leg2))
    monkeypatch.setenv("ARC_RL1_RESUME", str(leg1))
    knobs = tmp_path / "knobs.json"
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    (leg1 / "state" / "r04").mkdir(parents=True)
    (leg1 / "state" / "r04" / "mg_state.json").write_text(json.dumps({"checkpoint": str(leg1 / "state" / "r04"),
                                                                       "played_by": 4, "policy_uid": "u4"}))
    (leg1 / "state" / "r04" / "state.json").write_text("{}")
    loop.mg_served = [("r04", 4), ("r07", 7)]
    loop.trainer = FakeTrainer()
    loop.serving_adapter, loop.keep_checkpoint = "/w/r07", "r07"
    knobs.write_text(json.dumps({"rollback": {"to": "r04", "id": 1}}))
    assert asyncio.run(loop.mg_reconfigure(8, 7, loop.mg_probes)) == 4
    assert loop.policy_uid == "u4"  # the restored weights' uid: its kept tries are reusable again
    loop.cfg["mg"]["checkpoint"] = True
    leg2.mkdir(exist_ok=True)
    loop.mg_write_state(7, 4)
    assert json.loads((leg2 / "mg_state.json").read_text())["policy_uid"] == "u4"


def test_trainer_fast_knob_gate_keeps_a_floor_level_change_and_reverts_a_fault(tmp_path, monkeypatch):
    knobs = tmp_path / "knobs.json"
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    loop = make_loop(tmp_path, mg={"trainer_fast": {"prefetch": 2}})
    kl = {"value": 0.027}

    class Scoring(FakeTrainer):
        async def call(self, op, timeout=3600, **payload):
            if op == "score":
                self.calls.append((op, payload))
                return {"windows": {p["name"]: {"kl_previous": kl["value"]} for p in payload["windows"]}}
            return await super().call(op, timeout, **payload)

    loop.trainer = Scoring()
    eg = {"prefetch": 2, "union_attention": True, "gdn_fast": True}
    knobs.write_text(json.dumps({"trainer_fast": eg}))
    loop.mg_knobs(10)
    row = asyncio.run(loop.mg_trainer_fast_gate(10, loop.mg_probes))
    assert row["ok"] and loop.trainer.always == eg and loop.cfg["mg"]["trainer_fast"] == eg
    score = [kw for op, kw in loop.trainer.calls if op == "score"][-1]
    assert score["hold_previous"] and score["label"] == "knob"  # the reference does not move
    assert asyncio.run(loop.mg_trainer_fast_gate(11, loop.mg_probes)) is None  # no change: no rescore
    bad = {**eg, "experts": "native"}
    knobs.write_text(json.dumps({"trainer_fast": bad}))
    kl["value"] = 0.08
    loop.mg_knobs(12)
    row = asyncio.run(loop.mg_trainer_fast_gate(12, loop.mg_probes))
    assert not row["ok"] and row["reverted"] and loop.trainer.always == eg and loop.cfg["mg"]["trainer_fast"] == eg
    loop.mg_knobs(13)  # the file still asks for it: not re-applied
    assert loop.cfg["mg"]["trainer_fast"] == eg and asyncio.run(loop.mg_trainer_fast_gate(13, loop.mg_probes)) is None
    assert [d["ok"] for k, d in loop.feed.events if k == "knob_gate"] == [True, False]


# --- standard dosing: k3 early stop on pi_old, constant lr, probe KL diagnostic only ---------------

class KlStopTrainer(FakeTrainer):
    """Update steps report approx_kl from a script; above kl_stop the step is not taken (trainer contract)."""

    def __init__(self, kls):
        super().__init__()
        self.kls = list(kls)

    async def call(self, op, timeout=3600, **payload):
        if op == "k3_score":  # mg.k3_measure: forward-only k3 vs the kept pi_old
            self.calls.append((op, payload))
            kl = self.kls.pop(0)
            return {"ok": True, "approx_kl": kl, "approx_kl_tok": 0.027, "clip_frac": 0.01, "windows": 2, "replies": 4}
        if op != "update":
            return await super().call(op, timeout, **payload)
        self.calls.append((op, payload))
        kl = self.kls.pop(0)
        if payload.get("kl_stop") is not None and kl > payload["kl_stop"]:
            return {"ok": True, "windows": [], "approx_kl": kl, "clip_frac": 0.2, "kl_stopped": True, "applied": False,
                    "kl_stop": payload["kl_stop"]}
        return {"ok": True, "windows": [], "loss": 0.0, "grad_norm": 0.01, "update_norm": 0.5, "trained_tokens": 10,
                "approx_kl": kl, "approx_kl_tok": 0.027, "clip_frac": 0.01}


def std_loop(tmp_path, kls, auto=True):
    loop = make_loop(tmp_path, mg={"policy_loss": "gspo", "gspo_clip": [0.015, 0.015], "dosing": "standard",
                                   "target_kl": 2.5e-5, **({"kl_stop_auto": True} if auto else {})}, lr=2e-5)
    loop.trainer = KlStopTrainer(kls)
    return loop


def test_standard_dosing_stops_before_the_step_over_1_5_x_target_and_saves(tmp_path):
    loop = std_loop(tmp_path, [0.0, 1e-5, 3e-5, 4e-5, 1e-6, 1e-6])
    batches = [[{"name": f"t{i}", "w": 1.0}] for i in range(6)]
    merged, _ = asyncio.run(loop.dosed_update(3, None, None, None, loop.mg_probes, batches=batches, signed=True,
                                              cmd_extra={"hold_previous": True}))
    ups = [p for op, p in loop.trainer.calls if op == "update"]
    assert "kl_stop" not in ups[0] and all(math.isclose(p["kl_stop"], 3.75e-5) for p in ups[1:])
    assert len(ups) == 4 and merged["early"] == {"action": "kl_stop", "steps": 3, "of": 6, "approx_kl": 4e-5, "limit": 1.5 * 2.5e-5}
    assert len(merged["steps"]) == 3 and loop.trainer.ops()[-2:] == ["save", "score"]
    rows = [d for k, d in loop.feed.events if k == "update_step"]
    assert rows[-1]["kl_stopped"] and rows[-1]["applied"] is False and rows[0]["approx_kl_tok"] == 0.027


def test_standard_dosing_alert_only_takes_every_step(tmp_path):
    """k3 over target / stop is an alert; every step is taken and no kl_stop reaches the trainer."""
    loop = std_loop(tmp_path, [0.0, 1e-5, 3e-5, 4e-5, 1e-6, 1e-6], auto=False)
    batches = [[{"name": f"t{i}", "w": 1.0}] for i in range(6)]
    merged, _ = asyncio.run(loop.dosed_update(3, None, None, None, loop.mg_probes, batches=batches, signed=True,
                                              cmd_extra={"hold_previous": True}))
    ups = [p for op, p in loop.trainer.calls if op == "update"]
    assert len(ups) == 6 and all("kl_stop" not in p for p in ups)
    assert "kl_stopped" not in merged and "early" not in merged and len(merged["steps"]) == 6
    alerts = [d for k, d in loop.feed.events if k == "alert"]
    assert [(a["step"], a["why"].split()[2]) for a in alerts] == [(3, "target"), (4, "stop")]
    assert all(math.isclose(a["k3_stop"], 3.75e-5) for a in alerts)


def test_one_step_update_measures_k3_after_the_step(tmp_path):
    """one optimizer step per update; mg.k3_measure: the step keeps the subsample's pi_old from its own
    forward (keep_old, no old pass) and k3_score measures it after the step, forward only; alerts over target / stop."""
    loop = std_loop(tmp_path, [0.0, 5e-5], auto=False)
    loop.cfg["mg"]["k3_measure"] = 2
    batch = [{"name": f"t{i}", "w": 1.0} for i in range(5)]
    merged, _ = asyncio.run(loop.dosed_update(3, None, None, None, loop.mg_probes, batches=[batch], signed=True,
                                              cmd_extra={"share_prefix": True, "norm": "cap"}))
    ops = [(op, p) for op, p in loop.trainer.calls]
    ups = [p for op, p in ops if op == "update"]
    k3 = [p for op, p in ops if op == "k3_score"]
    assert "old_logprobs" not in [op for op, _ in ops] and [op for op, _ in ops].index("k3_score") > [op for op, _ in ops].index("update")
    assert len(ups) == 1 and ups[0]["save"] and "kl_stop" not in ups[0] and ups[0]["keep_old"] == ["t0", "t4"]
    assert len(k3) == 1 and [a["name"] for a in k3[0]["attempts"]] == ["t0", "t4"] and k3[0]["gspo_clip"] == [0.015, 0.015]
    assert k3[0]["share_prefix"] and "norm" not in k3[0] and len(merged["steps"]) == 1
    assert merged["k3_measure"]["approx_kl"] == 5e-5 and merged["k3_measure"]["ok"] and "kl_stopped" not in merged
    alerts = [d for k, d in loop.feed.events if k == "alert"]
    assert len(alerts) == 1 and "over stop" in alerts[0]["why"]


def test_one_step_token_loss_measures_k3_with_gspo_and_alerts_on_tis_out(tmp_path):
    """default: token-level TIS loss (no policy_loss), one step; the k3 measure is GSPO's per-reply k3;
    a token share outside [0.5, 2] over 5% alerts."""
    loop = make_loop(tmp_path, mg={"dosing": "standard", "target_kl": 2.5e-4, "k3_measure": 2}, lr=1e-5)

    class T(KlStopTrainer):
        async def call(self, op, timeout=3600, **payload):
            out = await super().call(op, timeout, **payload)
            if op == "update":
                out["windows"] = [{"signed": {"tis_out": {"tokens": 6, "of": 100}, "up": 1.0, "down": 0.0}}]
            return out

    loop.trainer = T([0.0, 1e-4])
    batch = [{"name": f"t{i}", "w": 1.0} for i in range(3)]
    merged, _ = asyncio.run(loop.dosed_update(3, None, None, None, loop.mg_probes, batches=[batch], signed=True))
    ops = [(op, p) for op, p in loop.trainer.calls]
    ups = [p for op, p in ops if op == "update"]
    assert [op for op, _ in ops] == ["update", "k3_score"]
    assert "policy_loss" not in ups[0] and ups[0]["keep_old"] == ["t0", "t2"]
    assert merged["k3_measure"]["approx_kl"] == 1e-4 and merged["k3_measure"]["ok"]
    rows = [d for k, d in loop.feed.events if k == "update_step"]
    assert rows[0]["tis_out"] == {"tokens": 6, "of": 100}
    alerts = [d for k, d in loop.feed.events if k == "alert"]
    assert len(alerts) == 1 and alerts[0]["why"].startswith("TIS") and alerts[0]["share"] == 0.06


def test_standard_dosing_full_update_and_no_lr_steering(tmp_path):
    loop = std_loop(tmp_path, [0.0, 1e-5, 2e-5])
    batches = [[{"name": f"t{i}", "w": 1.0}] for i in range(3)]
    merged, _ = asyncio.run(loop.dosed_update(3, None, None, None, loop.mg_probes, batches=batches, signed=True))
    assert "kl_stopped" not in merged and len(merged["steps"]) == 3
    row = {}
    assert loop.mg_kl_target_control(row, {"early": {"action": "stop", "excess": 0.5}, "probes": {
        "p1": {"kl_previous": 0.03}, "p2": {"kl_previous": 0.02}}}) is None
    assert loop.cfg["lr"] == 2e-5 and row["kl_step"]["diagnostic"] and math.isclose(row["kl_step"]["excess"], 0.007)


def test_tail_fill_round_benches_get_history_and_held_out_last(tmp_path):
    loop = make_loop(tmp_path, mg={"cap": 1000, "tail_fill": True})
    loop.output = tmp_path
    prev = tmp_path / "rl" / "r03-a"
    prev.mkdir(parents=True)
    rows = [{"point": p, "outcome": "cleared", "completion_tokens": t, "label": f"a{i:03d}-{p}-p0"}
            for i, (p, t) in enumerate([("s1", 100), ("s1", 300), ("s1", 900), ("s2", 50), ("s2", 70)])]
    rows.append({"point": "s3", "outcome": "error", "completion_tokens": 99999, "label": "a009-s3-p0"})  # not usable
    (prev / "level-bench.json").write_text(json.dumps({"attempts": rows}))
    tail = loop.mg_tail("r04-a", {"role": "train"})
    assert tail == {"expected": {"s1": 300, "s2": 60}, "last": False}
    assert loop.mg_tail("r01-a", {"role": "train"}) == {"expected": {}, "last": False}  # no history: plan order
    assert loop.mg_tail("r04-t", {"role": "test"})["last"] and loop.mg_tail("r04-fixed-a", {"role": "train"})["last"]

    seen = {}

    async def bench(name, games, n, budgets, **kw):
        seen[name] = kw
        return tmp_path, [], {"seconds": 0}

    loop.bench = bench
    loop.mg_points = lambda spec, starts, name, harvest=False: (tmp_path / "p.json", [{"id": "s1"}])
    spec = {"name": "a", "game": "g", "role": "train"}
    asyncio.run(loop.mg_play("r04-a", spec, [], 2, model="policy", seed=None))
    asyncio.run(loop.mg_play("final-x-a", spec, [], 2, model="policy", seed=None))  # not a round bench
    assert seen["r04-a"]["tail"]["expected"] == {"s1": 300, "s2": 60} and "tail" not in seen["final-x-a"]
    loop.cfg["mg"]["tail_fill"] = False  # the key off: the bench call carries no tail (today's call)
    asyncio.run(loop.mg_play("r04-a", spec, [], 2, model="policy", seed=None))
    assert "tail" not in seen["r04-a"]


def test_tail_fill_reaches_the_bench_study_only_when_set(tmp_path, monkeypatch):
    from arena import level_bench

    loop = make_loop(tmp_path, mg={"cap": 1000})
    loop.output = tmp_path
    loop.study, loop.trial, loop.deadline = {"level_bench": {}}, {}, 10 ** 10
    loop.source = loop.games_dir = loop.server_dir = loop.analysis_python = loop.base_url = None
    loop.gpu = type("G", (), {"mark": lambda self, x: None})()
    seen = []

    async def run(study, trial, **kw):
        seen.append(study)
        raise RuntimeError("stop")

    monkeypatch.setattr(level_bench, "run_level_bench", run)
    for tail in (None, {"expected": {"s": 1}, "last": True}):
        with pytest.raises(RuntimeError):
            asyncio.run(loop.bench("x", ["g"], 1, {"g": 10}, tail=tail))
    assert "tail_fill" not in seen[0] and seen[1]["tail_fill"] == {"expected": {"s": 1}, "last": True}
    rl_loop.settings({"mg": {"tail_fill": True}})  # a known key


def test_tail_fill_history_comes_from_memory_before_files(tmp_path):
    """Round packs move rNN folders away as the next round starts: mg_tail reads the in-memory medians first."""
    loop = make_loop(tmp_path, mg={"tail_fill": True})
    loop.output = tmp_path
    loop.mg_tail_hist = {"-sc25-L1": {"p0": 9000, "p1": 300}}
    got = loop.mg_tail("r10-sc25-L1", {"role": "train"})
    assert got == {"expected": {"p0": 9000, "p1": 300}, "last": False}
    assert loop.mg_tail("r10-fixed-sc25-L2", {"role": "test"}) == {"expected": {}, "last": True}


def _pending_source(tmp_path, their_mg, rounds, pack_round=8, rows=None):
    """A source run dir: feed (start settings + round rows) and rounds/rNN.tar with one train bench."""
    import tarfile

    src = tmp_path / "src"
    rl = src / "arena" / "trials" / "rl-loop" / "rl"
    rl.mkdir(parents=True)
    events = [{"kind": "start", "settings": {"mg": their_mg}}] + [{"kind": "round", "round": r, "played_by": pb}
                                                                  for r, pb in rounds]
    (rl / "feed.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    bench = tmp_path / "pack" / f"r{pack_round:02d}-sc25-L2"
    bench.mkdir(parents=True)
    (bench / "level-bench.json").write_text(json.dumps({"attempts": rows or [_carry_row(i) for i in range(4)]}))
    (src / "rounds").mkdir()
    with tarfile.open(src / "rounds" / f"r{pack_round:02d}.tar", "w:gz") as tar:
        tar.add(bench, arcname=bench.name)
    return src


def _pending_loop(tmp_path, mg, src):
    loop = make_loop(tmp_path, mg={"resume_train_pending": True, **mg})
    loop.cfg["mg"]["resume"] = str(src)
    loop.work = tmp_path / "work"
    seen = {}

    def select(f, r, n):
        seen.setdefault("select", []).append((Path(f).name, len(r), n))
        return [(Path(f), r[0], 1.0, "k"), (Path(f), r[1], -1.0, "k")], {"groups": 1, "mixed": 1, "trained": 2}

    loop.mg_select = select
    return loop, seen


PENDING_SETS = {"sc25-L2": {"name": "sc25-L2", "game": "sc25", "role": "train", "level": 2}}
PENDING_MG = {"group_size": 10, "fresh_tries": 50, "reward_shape": "logeq", "cap": 64000,
              "sets": [{"name": "sc25-L2", "game": "sc25", "level": 2, "role": "train"}]}


def test_pending_round_trains_a_round_played_by_the_resumed_state(tmp_path):
    src = _pending_source(tmp_path, dict(PENDING_MG), [(7, 5), (8, 7)])
    loop, seen = _pending_loop(tmp_path, dict(PENDING_MG), src)
    got = loop.mg_pending_round({"round": 7, "played_by": 7}, PENDING_SETS)
    assert got["round"] == 8 and got["played_by"] == 7 and len(got["data"]) == 2 and got["data"][0][-1] == "sc25-L2"
    assert seen["select"] == [("r08-sc25-L2", 4, "sc25-L2")]
    ev = [e for k, e in loop.feed.events if k == "pending_round"][-1]
    assert ev["used"] and ev["layout"] == "strict" and ev["layout_differs"] == {}


def test_pending_round_after_a_stop_at_the_next_round_top(tmp_path):
    # state r08 (trained on R7); R8 was played by r07 and never trained: lag 1, as the uninterrupted run
    src = _pending_source(tmp_path, dict(PENDING_MG), [(7, 5), (8, 7)])
    loop, _ = _pending_loop(tmp_path, dict(PENDING_MG), src)
    got = loop.mg_pending_round({"round": 8, "played_by": 8}, PENDING_SETS)
    assert got["round"] == 8 and got["played_by"] == 7


def test_pending_round_strict_refuses_a_changed_layout_and_lenient_takes_it(tmp_path):
    theirs = {**PENDING_MG, "group_size": 6, "fresh_tries": 48}
    src = _pending_source(tmp_path, theirs, [(8, 7)])
    loop, seen = _pending_loop(tmp_path, dict(PENDING_MG), src)
    assert loop.mg_pending_round({"round": 7, "played_by": 7}, PENDING_SETS) is None
    ev = [e for k, e in loop.feed.events if k == "pending_round"][-1]
    assert not ev["used"] and ev["why"] == "settings differ" and set(ev["differs"]) == {"group_size", "fresh_tries"}
    loop.cfg["mg"]["resume_train_pending"] = {"layout": "lenient"}
    got = loop.mg_pending_round({"round": 7, "played_by": 7}, PENDING_SETS)
    assert got is not None and got["info"]["layout_differs"]["group_size"] == [6, 10]


def test_pending_round_ignores_harvest_since_but_not_other_harvest_keys(tmp_path):
    theirs = {**PENDING_MG, "harvest": {"levels": [2, 3], "train_starts": 5}}
    src = _pending_source(tmp_path, json.loads(json.dumps(theirs)), [(7, 5), (8, 7)])
    loop, _ = _pending_loop(tmp_path, {**PENDING_MG, "harvest": {"levels": [2, 3], "train_starts": 5, "since": 9}}, src)
    assert loop.mg_pending_round({"round": 7, "played_by": 7}, PENDING_SETS)["round"] == 8
    loop.cfg["mg"]["harvest"] = {"levels": [2, 3], "train_starts": 6, "since": 9}
    assert loop.mg_pending_round({"round": 7, "played_by": 7}, PENDING_SETS) is None


def test_pending_round_compares_a_shrunk_preflight_source_after_its_shrink(tmp_path):
    """pfP2 (22 Oct): a preflight leg's start row holds its settings before the shrink; the resume compares after it."""
    import tarfile

    src = _pending_source(tmp_path, json.loads(json.dumps(PENDING_MG)), [(7, 5), (8, 7)])
    feed = src / "arena" / "trials" / "rl-loop" / "rl" / "feed.jsonl"
    feed.write_text(feed.read_text() + json.dumps({"kind": "preflight_shrink", "fresh_tries": 8}) + "\n")
    loop, _ = _pending_loop(tmp_path, rl_loop.Loop.shrink_mg(json.loads(json.dumps(PENDING_MG))), src)
    assert loop.mg_pending_round({"round": 7, "played_by": 7}, PENDING_SETS)["round"] == 8
    assert tarfile  # the pack path is the same as the unshrunk case


def test_pending_round_never_crosses_a_reward_change_or_a_missing_pack(tmp_path):
    theirs = {**PENDING_MG, "reward_shape": "floor"}
    src = _pending_source(tmp_path, theirs, [(8, 7)])
    loop, _ = _pending_loop(tmp_path, {**PENDING_MG, "resume_train_pending": {"layout": "lenient"}}, src)
    assert loop.mg_pending_round({"round": 7, "played_by": 7}, PENDING_SETS) is None
    # a round played by a rolled-back state (r09 is not the served r07) is not pending
    src2 = tmp_path / "two"
    src2.mkdir()
    s2 = _pending_source(src2, dict(PENDING_MG), [(10, 9)], pack_round=10)
    loop2, _ = _pending_loop(src2, dict(PENDING_MG), s2)
    assert loop2.mg_pending_round({"round": 7, "played_by": 7}, PENDING_SETS) is None
    ev = [e for k, e in loop2.feed.events if k == "pending_round"][-1]
    assert ev["why"] == "no pending round with a pack"


def test_pending_round_key_is_known():
    assert "resume_train_pending" in rl_loop.MG_KEYS


# --- v6x: PPO epochs, full GiGPO -------------------------------------------------------------------

def test_epochs_repeat_the_minibatches_in_a_new_order_and_old_pass_covers_every_window(tmp_path):
    loop = make_loop(tmp_path, mg={"minibatches": 3, "stratify": True, "max_groups": 6, "balance": False, "epochs": 4,
                                   "dosing": "standard"})
    loop.cfg.update(l2_starts_per_step=2, lvl_window=32768, path_chunk=16384, l2_norm="cap")
    loop.cfg["mg"]["cap"] = 32000
    loop.cfg["mg_sets"] = {"A": {"cap": 4000}}
    kept = [(tmp_path, {"label": f"g{g}-t{t}"}, 0.5 if t % 2 else -0.5, f"A-g{g}", "A") for g in range(6) for t in range(2)]
    seen = {}

    async def fake_dosed(number, *a, **k):
        seen.update(k)
        return {"ok": True}, tmp_path / "r01"

    loop.dosed_update = fake_dosed
    loop.checkpoint = lambda *a, **k: None
    asyncio.run(loop.mg_update(5, kept, loop.mg_probes))
    b = seen["batches"]
    assert len(b) == 12
    def names(batch):
        return sorted(s["name"] for s in batch)

    first = [names(x) for x in b[:3]]
    for e in range(1, 4):  # every epoch: the same three minibatches, in some order
        assert sorted(names(x) for x in b[3 * e:3 * e + 3]) == sorted(first)
    assert any([names(x) for x in b[3 * e:3 * e + 3]] != first for e in range(1, 4))
    # dosed_update: pi_old for every window once (step 1's windows recur in later epochs), not s = 1 for them
    g = make_loop(tmp_path, mg={"policy_loss": "gspo", "gspo_clip": [0.004, 0.005], "kl_control": None})
    g.cfg.update(steps_per_round=1, update_timeout=60)
    g.trainer = FakeTrainer(steps=[(0.0, 0.01)] * 4)
    a1, a2 = {"name": "a", "w": 1.0}, {"name": "b", "w": -1.0}
    asyncio.run(g.dosed_update(1, None, None, None, g.mg_probes, batches=[[a1], [a2], [a2], [a1]], signed=True))
    (op, old), *ups = g.trainer.calls
    assert op == "old_logprobs" and [s["name"] for s in old["attempts"]] == ["b", "a"] and len(ups) == 4


def gigpo_loop(tmp_path, **tc):
    return make_loop(tmp_path, mg={"reward_shape": "logeq", "log_fail": 16.0, "cap": 4000,
                                   "turn_credit": {"mode": "gigpo", "gamma": 0.95, "omega": 1.0, **tc}})


def test_gigpo_step_advantage_is_the_normalised_discounted_return_per_anchor(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_arena_branch import recorded_try

    loop = gigpo_loop(tmp_path)
    f = tmp_path / "r07-L1"
    grp = [{**recorded_try(f, "a", replies=5, cleared=True), "point": "p"},   # boards 0..4, clears at the 5th reply
           {**recorded_try(f, "b", replies=9, cleared=True), "point": "p"},   # boards 0..8 (the same boards)
           {**recorded_try(f, "c", replies=9, cleared=False), "point": "p"}]  # fails
    fail = -math.log(16.0)
    out = loop.mg_gigpo_steps(grp, [1.0, 1.0, fail], [
        ([(k, 100, (0, str(k)), k) for k in range(5)], True, 5, True),
        ([(k, 100, (0, str(k)), k) for k in range(9)], True, 9, True),
        ([(k, 100, (0, str(k)), k) for k in range(9)], False, 8, False)], [{}, {}, {}], fail)
    r = 1.0 - fail
    for k in range(5):  # three tries: G_a = .95^(4-k) r > G_b = .95^(8-k) r > G_c = 0
        g = [0.95 ** (4 - k) * r, 0.95 ** (8 - k) * r, 0.0]
        mu, sd = statistics.mean(g), statistics.pstdev(g)
        for i, name in enumerate("abc"):
            assert math.isclose(out[i]["extra"][k], (g[i] - mu) / sd, rel_tol=1e-9)
    for k in range(5, 9):  # b and c only: +1 / -1
        assert math.isclose(out[1]["extra"][k], 1.0) and math.isclose(out[2]["extra"][k], -1.0)
    summary = loop.mg_gigpo_summary()
    assert summary["steps"] == 23 and summary["share_group2"] == 1.0 and summary["share_cross_try"] == 1.0
    assert summary["group_size_median"] == 3 and loop.mg_gigpo_summary() is None  # reset after the read
    # a revisit in the same try joins its anchor group; a board no other step saw gets nothing
    out = loop.mg_gigpo_steps(grp[:1], [1.0], [([(0, 10, "x", 0), (1, 10, "y", 1), (2, 10, "x", 2), (3, 10, "z", 3)],
                                                True, 3, True)], [{}], fail)
    assert out[0]["extra"][0] < 0 < out[0]["extra"][2] and 1 not in out[0]["extra"] and 3 not in out[0]["extra"]
    assert loop.mg_gigpo_summary()["share_cross_try"] == 0.0


def test_gigpo_step_scale_episode_std_puts_the_step_term_on_the_episode_scale(tmp_path):
    """turn_credit.step_scale "episode_std" (GiGPO normalizes both terms per group): omega x A_step x the group's reward
    std, so a low-spread group is not step-dominated."""
    loop = gigpo_loop(tmp_path)
    loop.cfg["mg"]["turn_credit"]["step_scale"] = "episode_std"
    grp = [{"point": "p"}, {"point": "p"}]
    tries = [([(0, 10, "x", 0), (1, 10, "y", 1)], True, 1, True), ([(0, 10, "x", 0), (1, 10, "y", 1)], True, 1, True)]
    out = loop.mg_gigpo_steps(grp, [1.0, 1.2], [(r, w, e, g) for r, w, e, g in tries], [{}, {}], -math.log(16.0))
    sd = statistics.pstdev([1.0, 1.2])
    assert math.isclose(out[1]["extra"][0], sd * 1.0, rel_tol=1e-9) and math.isclose(out[0]["extra"][0], -sd, rel_tol=1e-9)
    out = loop.mg_gigpo_steps(grp, [1.0, 1.0], tries, [{}, {}], -math.log(16.0))  # no episode spread: no step term
    assert all(v == 0.0 for o in out for v in o["extra"].values())


def test_gigpo_through_turn_credit_and_reply_balance_adds_omega_a_step_unscaled(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_arena_branch import recorded_try

    loop = gigpo_loop(tmp_path)
    loop.mg_level_ref = lambda name, game, level: {}
    f = tmp_path / "r07-L1"
    grp = [{**recorded_try(f, "a", replies=5, cleared=True), "point": "p"},
           {**recorded_try(f, "b", replies=9, cleared=True), "point": "p"},
           {**recorded_try(f, "c", replies=9, cleared=False), "point": "p"}]
    out = loop.mg_turn_credit(f, grp, [1.0, 1.0, -math.log(16.0)], None)
    assert set(out[0]["extra"]) == set(range(5)) and out[0]["extra"][0] > 0 > out[2]["extra"][0]  # start board in
    loop.cfg["mg"]["balance"] = True
    specs = [{"w": 0.5, "turn_key": ("f", "a"), "set": "L1"}, {"w": -0.5, "turn_key": ("f", "c"), "set": "L1"}]
    turn = {("f", "a"): {"tokens": {0: 100}, "extra": {0: 2.0}}, ("f", "c"): {"tokens": {0: 100}, "extra": {0: -2.0}}}
    loop.mg_gigpo_stats = {"steps": 2, "in_group": 2, "cross_try": 2, "abs_sum": 4.0, "sizes": [2]}
    bal = loop.mg_reply_balance(specs, turn)
    assert specs[0]["reply_w"]["0"] == 2.5 and specs[1]["reply_w"]["0"] == -2.5 and bal["extra_scale"] == 1.0
    assert bal["gigpo"]["share_group2"] == 1.0 and bal["gigpo"]["mean_abs_step"] == 2.0


def test_unseen_reads_list_caps_starts_and_frozen_start_sets(tmp_path):
    import pytest

    from arena.rl_single import MultiMixin

    class Loop(MultiMixin):
        def __init__(self, u):
            self.output = Path("/nonexistent")
            self.cfg = {"mg": {"unseen_read": u, "sets": [{"name": "L2", "role": "test"}, {"name": "L4", "role": "test"}]},
                        "mg_sets": {"L2": {"name": "L2", "role": "test", "tries": 0, "round_tries": 0},
                                    "L4": {"name": "L4", "role": "test", "tries": 0, "round_tries": 0}}}

    reads = [{"set": "L2", "tries": 4, "every": 3, "offset": 1, "starts": 4, "base_first": True},
             {"set": "L4", "tries": 2, "every": 3, "offset": 1, "base_first": True}]
    loop = Loop(reads)
    assert [loop.mg_test_tries(n, "L2") for n in range(9, 14)] == [0, 4, 0, 0, 4]
    assert [loop.mg_test_tries(n, "L4") for n in range(9, 14)] == [0, 2, 0, 0, 2]
    assert loop.mg_unseen_read(name="L2")["starts"] == 4 and loop.mg_unseen_read(name="L4")["starts"] is None
    assert {"L2-base", "L4-base"} <= set(loop.cfg["mg_sets"])
    with pytest.raises(ValueError):
        Loop([reads[0], reads[0]]).mg_unseen_reads()
    with pytest.raises(ValueError):
        Loop([{**reads[0], "starts": 0}]).mg_unseen_reads()
    screen = {"sets": {"L4": {"starts_only": True, "starts": [["/r/a030", 53], ["/r/a032", 50]], "folder": "/none"}}}
    assert loop.mg_ok_starts("L4", screen) == [("/r/a030", 53), ("/r/a032", 50)]


def test_held_metrics_standard_error_of_the_ratio():
    from arena.rl_single import held_metrics

    def row(t, score):
        return {"outcome": "cleared" if score else "budget", "completion_tokens": t,
                "level_rows": [{"tokens": t, "actions": 5, "score": score}] if score else []}

    rows = [row(1000, 100.0), row(2000, 50.0), row(4000, None), row(1000, 115.0)]
    m = held_metrics(rows)
    ratio = (1.0 + 0.5 + 1.15) / 8000
    dev = [1.0 - ratio * 1000, 0.5 - ratio * 2000, -ratio * 4000, 1.15 - ratio * 1000]
    assert math.isclose(m["rhae_10k_se"], round(math.sqrt(4 / 3 * sum(d * d for d in dev)) / 8000 * 1e4, 4))
    assert held_metrics(rows[:1])["rhae_10k_se"] is None


def test_update_step_rows_carry_pass_and_k3_lines(tmp_path):
    loop = std_loop(tmp_path, [0.0, 1e-5, 2e-5, 2e-5])
    loop.cfg["mg"]["epochs"] = 2
    batches = [[{"name": f"t{i % 2}", "w": 1.0}] for i in range(4)]
    asyncio.run(loop.dosed_update(3, None, None, None, loop.mg_probes, batches=batches, signed=True))
    rows = [d for k, d in loop.feed.events if k == "update_step"]
    assert [r["pass"] for r in rows] == [1, 1, 2, 2]
    assert all(math.isclose(r["k3_target"], 2.5e-5) and math.isclose(r["k3_stop"], 3.75e-5) for r in rows)


def test_read_norm_takes_the_base_round_then_reads_per_game(tmp_path):
    loop = mg6_loop(tmp_path, read_norm={"n_boot": 50})
    loop.mg_init = False
    sets = {f"g{i}-L1": {"name": f"g{i}-L1", "game": f"g{i}am", "level": 1, "role": "test", "fresh": True,
                         "source": "pub" if i < 2 else "gen"} for i in range(3)}

    def rows(i, tokens, n=6):
        return [{"label": f"t{i}{k}", "point": f"g{i}am-game-L1-x", "outcome": "cleared", "completion_tokens": tokens,
                 "level_rows": [{"level": 1, "tokens": tokens, "actions": 20}]} for k in range(n)]

    base_rows = {n: (tmp_path, rows(i, 20000), {}) for i, n in enumerate(sets)}
    row = {"fixed": {}}
    loop.mg_read_norm(1, 0, base_rows, sets, row)
    assert row["fixed"]["norm"]["base_round"] and set(loop.mg_read_base) == set(sets)
    assert row["fixed_sets"]["g0-L1"][:2] == [6, 6]
    half = {n: (tmp_path, rows(i, 10000), {}) for i, n in enumerate(sets)}
    row = {"fixed": {}}
    loop.mg_read_norm(3, 2, half, sets, row)
    norm = row["fixed"]["norm"]
    assert norm["all"]["games"] == 3 and norm["all"]["ratio"] > 1.5 and norm["source:gen"]["games"] == 1


def test_screen_base_skips_sets_no_screen_played(tmp_path):
    loop = mg6_loop(tmp_path)
    loop.cfg["mg_sets"] = {"q301-L2": {"name": "q301-L2", "game": "q301", "level": 2, "role": "train"}}
    loop.mg_round_starts = {3: {"q301-L2": [("/runs/harvest/x", 5)]}}
    assert loop.mg_screen_base([3]) == []
    from arena import rl_loop
    rl_loop.check_mg({"fixed_every": 2, "fixed_base_tries": 16,
                      "sets": [{"name": "tr87-L1", "game": "tr87", "level": 1, "role": "test", "fresh": True, "read": True,
                                "tries": 0}]})


def test_live_per_set_play_stats_and_round_filter(tmp_path):
    """play_progress rows carry per-set running aggregates of the current round's benches."""
    loop = make_loop(tmp_path, mg={})
    clear = {"outcome": "cleared", "completion_tokens": 900, "level_rows": [{"tokens": 800, "actions": 40, "score": 64.0}]}
    miss = {"outcome": "budget", "completion_tokens": 1500, "level_rows": [{"actions": 70}]}
    loop.mg_live_sets = {"r13-fixed-sc25-L1": {**loop.mg_live_zero(), "planned": 32}}
    for bench, row in (("r13-fixed-sc25-L1", clear), ("r13-fixed-sc25-L1", miss), ("r14-sc25-L1", clear),
                       ("games-00", clear)):
        loop.mg_progress_row(bench, row)
    got = loop.mg_live_round(13)
    assert got == {"fixed-sc25-L1": {"tries": 2, "clears": 1, "tok_sum": 2400, "act_sum": 40, "rhae_sum": 0.64,
                                     "planned": 32}}
    assert loop.mg_live_round(14)["sc25-L1"]["tries"] == 1 and "games-00" not in loop.mg_live_sets


def test_heldout_first_puts_every_held_out_read_first(tmp_path):
    loop = make_loop(tmp_path, mg={"tail_fill": True})
    loop.output = tmp_path
    loop.mg_tail_hist = {}
    assert loop.mg_tail("r13-fixed-sc25-L1", {"role": "train"})["last"] is True
    loop.cfg["mg"]["heldout_first"] = True
    fixed = loop.mg_tail("r13-fixed-sc25-L1", {"role": "train"})
    unseen = loop.mg_tail("r13-sc25-L4", {"role": "test"})
    train = loop.mg_tail("r13-sc25-L1", {"role": "train"})
    assert fixed["first"] and not fixed["last"] and unseen["first"] and not unseen["last"]
    assert not train["first"] and not train["last"]


class CkptTrainer(FakeTrainer):
    """Writes checkpoint folders: a background one stays .partial until checkpoint_wait (or the next checkpoint)."""

    def __init__(self, refuse=False):
        super().__init__()
        self.partial, self.refuse = None, refuse

    def land(self):
        if self.partial is None:
            return None
        part, self.partial = self.partial, None
        part.rename(part.with_name(part.name[: -len(".partial")]))
        return {"path": str(part.with_name(part.name[: -len(".partial")])), "write_seconds": 1.0}

    async def call(self, op, timeout=3600, **payload):
        self.calls.append((op, payload))
        if op == "checkpoint":
            prior = self.land()
            folder = Path(payload["path"])
            if payload.get("background") and not self.refuse:
                self.partial = folder.with_name(folder.name + ".partial")
                self.partial.mkdir(parents=True)
                return {"ok": True, "background": True, **({"prior": prior} if prior else {})}
            folder.mkdir(parents=True)
            return {"ok": True, **({"background_refused": {}} if payload.get("background") else {})}
        if op == "checkpoint_wait":
            done = self.land()
            return {"ok": True, "pending": done is not None, **(done or {})}
        return {"ok": True}


def test_async_checkpoint_keeps_the_last_complete_state(tmp_path, monkeypatch):
    """MFU-1 E: with mg.async_checkpoint the previous complete state stays until the new one lands; mg_state.json
    written while the state is still being saved lands inside it; the synchronous path is unchanged."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = make_loop(tmp_path, mg={"async_checkpoint": True})
    loop.trainer = CkptTrainer()
    state = tmp_path / "state"

    async def go():
        loop.keep_checkpoint = None
        loop.checkpoint(1, force=True)
        await loop.save_checkpoint(background=True)
        assert sorted(p.name for p in state.iterdir()) == ["r01.partial"]
        loop.keep_checkpoint = "r01"
        (tmp_path / "mg_state.json").write_text('{"round": 1}')
        loop.ckpt_deferred = ("r01", '{"round": 1}') if loop.ckpt_pending == "r01" else None
        loop.checkpoint(2, force=True)
        await loop.save_checkpoint(background=True)  # r01 lands first (its mg_state with it), stays until r02 lands
        assert (state / "r01" / "mg_state.json").read_text() == '{"round": 1}'
        assert sorted(p.name for p in state.iterdir()) == ["r01", "r02.partial"]
        await loop.checkpoint_landed()
        assert sorted(p.name for p in state.iterdir()) == ["r01", "r02"]
        loop.keep_checkpoint = "r02"
        loop.checkpoint(3, force=True)
        await loop.save_checkpoint()  # synchronous: as before, every other folder but the kept ones goes
        assert sorted(p.name for p in state.iterdir()) == ["r02", "r03"]
        assert [e for e, _ in loop.feed.events].count("checkpoint_done") == 2

    asyncio.run(go())


def test_async_checkpoint_crash_before_landing_resumes_the_last_complete_state(tmp_path, monkeypatch):
    """RL review of E: while rNN is being written the root mg_state.json still names the last complete state (a crash
    there resumes it and replays the round); when the folder lands the watcher writes both files (no trainer op)."""
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = make_loop(tmp_path, mg={"async_checkpoint": True, "checkpoint": True})
    loop.trainer = CkptTrainer()
    state = tmp_path / "state"

    async def go():
        loop.checkpoint(1, force=True)
        await loop.save_checkpoint()  # r01 complete, synchronous
        loop.keep_checkpoint = "r01"
        loop.mg_write_state(1, 1)
        assert json.loads((tmp_path / "mg_state.json").read_text())["round"] == 1
        loop.checkpoint(2, force=True)
        await loop.save_checkpoint(background=True)
        loop.keep_checkpoint = "r02"
        loop.mg_write_state(2, 2)  # the state is still r02.partial: the root file is not moved to it
        root = json.loads((tmp_path / "mg_state.json").read_text())
        assert root["round"] == 1 and Path(root["checkpoint"]).name == "r01"
        assert (state / "r01").is_dir()  # a crash here resumes r01 (complete) and replays round 2
        assert sorted(p.name for p in state.iterdir()) == ["r01", "r02.partial"]
        loop.trainer.land()  # the write completes (rename); the watcher follows without a trainer call
        await asyncio.wait_for(loop.ckpt_watch_folder("r02", state / "r02", every=0.01), 5)
        root = json.loads((tmp_path / "mg_state.json").read_text())
        assert root["round"] == 2 and Path(root["checkpoint"]).name == "r02"
        assert json.loads((state / "r02" / "mg_state.json").read_text())["round"] == 2
        assert not (tmp_path / "mg_state.json.tmp").exists()
        loop.mg_write_state(2, 2)  # landed: written directly again
        await loop.checkpoint_landed()
        assert [e for e, _ in loop.feed.events].count("checkpoint_state") == 1

    asyncio.run(go())


def test_async_checkpoint_refused_by_the_trainer_runs_as_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = make_loop(tmp_path, mg={"async_checkpoint": True})
    loop.trainer = CkptTrainer(refuse=True)

    async def go():
        loop.keep_checkpoint = None
        loop.checkpoint(1, force=True)
        await loop.save_checkpoint(background=True)
        assert loop.ckpt_pending is None and loop.ckpt_landed == "r01"
        assert sorted(p.name for p in (tmp_path / "state").iterdir()) == ["r01"]

    asyncio.run(go())


def test_window_guard_falls_back_after_a_peak_over_the_prediction(tmp_path):
    """MFU-1 D: lvl_window 48K with mg.window_guard; a step peak under the limit keeps it, one over sets 32K."""
    loop = make_loop(tmp_path, mg={"window_guard": {"peak_gib": 261.5, "fallback": 32768}})
    loop.cfg["lvl_window"] = 49152
    assert not loop.window_guard(2, {"step": 1, "torch_peak_gib": 240.0}) and loop.cfg["lvl_window"] == 49152
    assert not loop.window_guard(2, {"step": 1}) and loop.cfg["lvl_window"] == 49152
    assert loop.window_guard(3, {"step": 1, "torch_peak_gib": 262.0}) and loop.cfg["lvl_window"] == 32768
    assert not loop.window_guard(4, {"step": 1, "torch_peak_gib": 300.0})  # already at the fallback
    kinds = [e for e, _ in loop.feed.events]
    assert kinds.count("window_fallback") == 1 and "alert" in kinds
    off = make_loop(tmp_path / "b", mg={})
    off.cfg["lvl_window"] = 49152
    assert not off.window_guard(2, {"torch_peak_gib": 999.0}) and off.cfg["lvl_window"] == 49152


def test_window_guard_oom_retry_falls_back_with_an_alert_only(tmp_path):
    """MFU-1 D: the trainer's out-of-memory retry (window_fallback why oom) moves lvl_window to the fallback; the alert
    is marked alert only (the loop judge passes it)."""
    loop = make_loop(tmp_path, mg={"window_guard": {"peak_gib": 261.5, "fallback": 32768}})
    loop.cfg["lvl_window"] = 49152
    assert loop.window_guard(2, {"step": 1, "why": "oom", "to": 32768, "error": "OOM"})
    assert loop.cfg["lvl_window"] == 32768
    events = dict((e, row) for e, row in loop.feed.events)
    assert events["window_fallback"]["why"] == "oom" and "alert only" in events["alert"]["why"]


class OomRetryTrainer(KlStopTrainer):
    """The trainer's contract after an out-of-memory retry: the update result carries window_fallback."""

    async def call(self, op, timeout=3600, **payload):
        out = await super().call(op, timeout, **payload)
        fb = payload.get("oom_fallback_window")
        if op == "update" and fb and any(a.get("window", 0) > fb for a in payload["attempts"]):
            out["window_fallback"] = {"why": "oom", "from": 49152, "to": fb, "error": "OOM"}
        return out


def test_oom_retry_moves_the_k3_measure_windows_to_the_fallback(tmp_path):
    """RL review of 347ebf432: after an OOM retry the step kept pi_old at 32K windows; k3_score must ask for the same."""
    loop = std_loop(tmp_path, [0.0, 5e-5], auto=False)
    loop.trainer = OomRetryTrainer([0.0, 5e-5])
    loop.cfg["mg"].update(k3_measure=2, window_guard={"peak_gib": 261.5, "fallback": 32768})
    loop.cfg["lvl_window"] = 49152
    batch = [{"name": f"t{i}", "w": 1.0, "whole": True, "window": 49152} for i in range(5)]
    merged, _ = asyncio.run(loop.dosed_update(3, None, None, None, loop.mg_probes, batches=[batch], signed=True,
                                              cmd_extra={"share_prefix": True, "norm": "cap"}))
    ups = [p for op, p in loop.trainer.calls if op == "update"]
    k3 = [p for op, p in loop.trainer.calls if op == "k3_score"]
    assert ups[0]["oom_fallback_window"] == 32768 and loop.cfg["lvl_window"] == 32768
    assert [a["window"] for a in k3[0]["attempts"]] == [32768, 32768] and [a["name"] for a in k3[0]["attempts"]] == ["t0", "t4"]
    assert [a["window"] for a in batch] == [49152] * 5  # the update's specs are not modified
    assert [d["why"] for k, d in loop.feed.events if k == "window_fallback"] == ["oom"]
    assert merged["k3_measure"]["ok"]


def test_probe_pinned_scores_kl_to_base_on_base_routes(monkeypatch):
    """kl_base_pinned (23 Oct): the base pass records its router picks (capture, snapshots off); a later probe replays
    them at the new weights and scores KL against that capture; ROUTE and snapshots come back as they were."""
    t, specs = toy_trainer(monkeypatch), [{"name": "p"}]
    seen = []
    plain = t.window_pass

    def window_pass(made, vocab=None):
        r = rl1_trainer.ROUTE
        seen.append((r["router"], t.snapshots))
        if r["router"] == "capture":
            r["rec"].extend([np.zeros((3, 2), np.int16)] * 4)
        if r["router"] == "replay":
            r["i"] += 4
            return {**plain(made, vocab), "actor": np.full(3, -t.theta / 2), "dist": {"ids": None, "logp": t.theta / 2, "at": [t.theta / 2]}}
        out = plain(made, vocab)
        if vocab == []:
            out["dist"]["ids"] = "ids"
        return out

    t.window_pass = window_pass
    t.probe_pinned, t.snapshots = True, True
    rl1_trainer.ROUTE.update(router="served", rec=[], i=0)
    t.score_windows(specs, label="base")
    assert seen == [("served", True), ("capture", False)] and len(t.references["p"]["base_pinned"]["route"]) == 4
    t.theta = 2.0
    row = t.score_windows(specs)["p"]
    assert seen[-1] == ("replay", False) and row["kl_base"] == 4.0 and row["kl_base_pinned"] == 1.0
    assert rl1_trainer.ROUTE["router"] == "served" and t.snapshots
    t.probe_pinned = False
    assert "kl_base_pinned" not in t.score_windows(specs)["p"]
    rl1_trainer.ROUTE.update(router="off")


def test_probe_pinned_replay_must_use_every_recorded_call(monkeypatch):
    t, specs = toy_trainer(monkeypatch), [{"name": "p"}]
    plain = t.window_pass

    def window_pass(made, vocab=None):
        r = rl1_trainer.ROUTE
        if r["router"] == "capture":
            r["rec"].extend([np.zeros((3, 2), np.int16)] * 4)
        if r["router"] == "replay":
            r["i"] += 3  # a prefix reused from somewhere: one router call short
        return plain(made, vocab)

    t.window_pass, t.probe_pinned, t.snapshots = window_pass, True, False
    t.score_windows(specs, label="base")
    with pytest.raises(RuntimeError, match="3 of 4"):
        t.score_windows(specs)
    assert rl1_trainer.ROUTE["router"] == "off"


def test_batched_tails_refused():
    """MFU-1 m1g1 (23 Oct): batch_pass gradients disagree with per-window; a bundle asking for it refuses."""
    rl_loop.check_mg({"batch_tokens": 0, "batch_max": 8})
    with pytest.raises(ValueError, match="batch_tokens > 0"):
        rl_loop.check_mg({"batch_tokens": 32768})


def test_ref_fallback_v0_then_group_for_units_the_screen_lacks(tmp_path):
    """mg.ref_fallback: a set without screen metrics takes the lineage's version-0 clears (>= 2), else ("v0_group")
    its group's own clears, before the env table's ref (v6g: 64,000-token placeholders clipped every clear at +log 8)."""
    loop = mg6_loop(tmp_path, reward_shape="logeq", rhae_cap=1.15, log_clip=8.0, log_fail=16.0)
    loop.cfg["mg_sets"] = {"y-L2": {"ref": {"tokens": 64000, "actions": 30, "baseline": 40}}}
    r = lambda t, a: loop.mg_reward(try_row("a", t, a, point="ygam-game-L2-a-k1"), "y-L2")
    assert r(3000, 30) == pytest.approx(math.log(8))  # off: the placeholder clips the clear
    loop.cfg["mg"]["ref_fallback"] = "v0"
    v0 = [{**try_row(f"v{i}", t, 30, point="ygam-game-L2-a-k1"), "policy_version": pv}
          for i, (t, pv) in enumerate([(4000, 0), (2000, 0), (1000, 1)])]
    loop.mg_v0_add("y-L2", v0 + v0[:1])  # a resumed group's kept try again: counted once
    assert loop.mg_ref_fallback("y-L2")["median_to_clear"] == 3000
    assert r(3000, 30) == pytest.approx(0.0) and r(1500, 30) == pytest.approx(math.log(2))
    assert r(3000, 30) == pytest.approx(0.0) and loop.mg_reward(try_row("a", 10000, 30), "x-L2") == pytest.approx(0.0)
    loop.mg_v0_refs, loop.mg_refs = {}, {}
    loop.cfg["mg"]["ref_fallback"] = "v0_group"
    grp = [try_row(f"g{i}-p{i}", t, 30, point="ygam-game-L2-g-k1") for i, t in enumerate([2000, 4000, 8000])]
    out, info = loop.mg_select(tmp_path, grp, "y-L2")
    assert info["groups"] == 1 and not info.get("zero_spread")  # off, all three clip at +log 8 (zero spread)
    assert loop.mg_group_ref is None
    loop.cfg["mg_sets"]["y-L2"]["ref"].update(tokens=6000, measured=True)  # a measured table ref outranks the fallback
    loop.mg_refs = {}
    assert loop.mg_reward(try_row("a", 3000, 30, point="ygam-game-L2-a-k1"), "y-L2") == pytest.approx(math.log(2))
    loop.cfg["mg_sets"]["y-L2"]["ref"].update(tokens=12000, measured=False, derived=True)  # a derived ref too
    loop.mg_refs = {}
    assert loop.mg_reward(try_row("a", 3000, 30, point="ygam-game-L2-a-k1"), "y-L2") == pytest.approx(math.log(4))
    with pytest.raises(ValueError):
        loop.cfg["mg"]["ref_fallback"] = "screen"
        loop.mg_ref_fallback("y-L2")


def test_derived_ref_with_null_actions_takes_the_fallback_actions_per_field(tmp_path):
    """mg-refs-v3 (envs 25 Oct): a derived ref keeps its tokens; actions null -> the v0 fallback's actions only
    (ka59-L4: derived 35.4 vs real 220.5 actions saturated the action term)."""
    loop = mg6_loop(tmp_path, reward_shape="logeq", rhae_cap=1.15, log_clip=8.0, log_fail=16.0)
    loop.cfg["mg"]["ref_fallback"] = "v0"
    loop.cfg["mg_sets"] = {"y-L2": {"ref": {"tokens": 12000, "actions": None, "derived": True, "measured": False}}}
    v0 = [{**try_row(f"v{i}", t, a, point="ygam-game-L2-a-k1"), "policy_version": 0}
          for i, (t, a) in enumerate([(4000, 200), (2000, 240)])]
    loop.mg_v0_add("y-L2", v0)
    ref = loop.mg_level_ref("y-L2", "ygam", 2)
    assert ref["tokens"] == 12000 and ref["actions"] == loop.mg_ref_fallback("y-L2")["median_actions"]
    loop.cfg["mg_sets"]["y-L2"]["ref"]["actions"], loop.mg_refs = 35.4, {}  # a derived ref with actions wins whole
    assert loop.mg_level_ref("y-L2", "ygam", 2)["actions"] == 35.4


def test_check_mg_refuses_placeholder_reward_refs():
    mg = {"sets": [{"name": "a-L1", "ref": {"tokens": 64000}}, {"name": "b-L1", "ref": {"tokens": 64000, "derived": True}},
                   {"name": "c-L1", "role": "test", "ref": {"tokens": 64000}}]}
    with pytest.raises(ValueError, match="placeholder reward refs.*a-L1"):
        rl_loop.check_mg(mg)
    mg["sets"][0]["ref"]["measured"] = True
    rl_loop.check_mg(mg)


def test_reward_ref_group_centres_the_clip_on_the_group_clear_median(tmp_path):
    """mg.reward_ref "group": the reward's T_b / R_b are the group's own clear medians, whatever the ref."""
    loop = mg6_loop(tmp_path, reward_shape="logeq", rhae_cap=1.15, log_clip=8.0, log_fail=16.0, reward_ref="group",
                    group_size=4)
    loop.cfg["mg_sets"] = {"y-L2": {"ref": {"tokens": 64000, "actions": 30, "baseline": 40, "measured": True, "n": 3}}}
    grp = [try_row(f"g{i}-p{i}", t, 30, point="ygam-game-L2-g-k1") for i, t in enumerate([2000, 4000, 8000])]
    grp.append(try_row("g3-p3", 9000, 30, cleared=False, point="ygam-game-L2-g-k1"))
    out, info = loop.mg_select(tmp_path, grp, "y-L2")
    w = {r["label"]: a for _, r, a, _ in out}
    assert w["g0-p0"] > w["g1-p1"] > w.get("g2-p2", 0.0) > w["g3-p3"]  # a zero advantage is not trained
    loop.mg_group_ref = {"name": "y-L2", "clears": [(2000, 30), (4000, 30), (8000, 30)]}
    assert loop.mg_reward(grp[0], "y-L2") == pytest.approx(math.log(2)) and loop.mg_reward(grp[1], "y-L2") == pytest.approx(0.0)
    assert loop.mg_reward(grp[3], "y-L2") == pytest.approx(-math.log(16))
    loop.mg_group_ref = None


def _s2_with_fake_post(tmp_path, monkeypatch, keep_lora, keep_kv):
    s = rl_loop.SecondServer(tmp_path, "m", card="1", keep_lora=keep_lora)
    s.keep_kv, s.loaded = keep_kv, None
    posts = []

    async def post(path, body=None):
        posts.append((path, (body or {}).get("lora_name")))
        return {}
    monkeypatch.setattr(s, "post", post)
    monkeypatch.setattr(s, "mask_check", lambda *a: asyncio.sleep(0))
    monkeypatch.setattr(rl_loop, "adapter_tensors", lambda path: 0)
    monkeypatch.setattr(rl_loop, "card_memory_of", lambda card: (4.0, 0))
    return s, posts


def test_second_server_keep_lora_skips_reloads_after_keep_kv_sleep(tmp_path, monkeypatch):
    """MFU-2 (25 Oct): keep_lora + ARC_SLEEP_KEEP_KV: a block after sleep/wake reloads nothing already loaded from the
    same folder; serve() leaves other versions to the block's prune; a new folder under a kept name reloads."""
    s, posts = _s2_with_fake_post(tmp_path, monkeypatch, keep_lora=True, keep_kv=True)

    async def block(served, extra):
        await s.wake()
        await s.serve(f"policy-r{served:02d}", tmp_path / f"v{served}")
        for v in extra:
            await s.load(f"policy-r{v:02d}", tmp_path / f"v{v}")
        gone = await s.prune({f"policy-r{v:02d}" for v in [served, *extra]})
        await s.sleep()
        return gone

    asyncio.run(block(3, [2, 1]))
    assert [p for p in posts if p[0] == "/v1/load_lora_adapter"] == [
        ("/v1/load_lora_adapter", "policy-r03"), ("/v1/load_lora_adapter", "policy-r02"),
        ("/v1/load_lora_adapter", "policy-r01")]
    posts.clear()
    assert asyncio.run(block(3, [2, 1])) == [] and not [p for p in posts if "lora" in p[0]]  # unchanged set: no reload
    assert s.skipped_loads == 2  # the extras (serve's own skip is the adapter match)
    posts.clear()
    gone = asyncio.run(block(4, [3, 2]))  # a new version: one load, the oldest pruned
    assert [p for p in posts if p[0] == "/v1/load_lora_adapter"] == [("/v1/load_lora_adapter", "policy-r04")]
    assert gone == ["policy-r01"] and ("/v1/unload_lora_adapter", "policy-r01") in posts
    posts.clear()
    asyncio.run(s.load("policy-r04", tmp_path / "v4b"))  # same name, other folder: unload + load
    assert posts == [("/v1/unload_lora_adapter", "policy-r04"), ("/v1/load_lora_adapter", "policy-r04")]


@pytest.mark.parametrize("keep_lora,keep_kv", [(False, True), (True, False)])
def test_second_server_reloads_without_both_keep_flags(tmp_path, monkeypatch, keep_lora, keep_kv):
    s, posts = _s2_with_fake_post(tmp_path, monkeypatch, keep_lora=keep_lora, keep_kv=keep_kv)

    async def block():
        await s.wake()
        await s.serve("policy-r03", tmp_path / "v3")
        await s.load("policy-r02", tmp_path / "v2")
        await s.sleep()
    asyncio.run(block())
    posts.clear()
    asyncio.run(block())
    assert [p for p in posts if p[0] == "/v1/load_lora_adapter"] == [
        ("/v1/load_lora_adapter", "policy-r03"), ("/v1/load_lora_adapter", "policy-r02")]
    assert not s.keeps_lora() and s.skipped_loads == 0
