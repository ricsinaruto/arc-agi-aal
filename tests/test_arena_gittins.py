"""Bayesian Gittins-style slot allocation with a broad opening (fixed-2x-gittins-broad): posterior and
index arithmetic, scheduler behaviour, the scheduler audit and the variant."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_arena_hazard_index import build, charged, report

from arena.gittins import DEFAULTS, GittinsModel, settings
from arena.scheduler import Scheduler
from arena.scheduler_audit import audit_scheduler
from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())


def record(game_id, levels, level_tokens, spent, total=8):
    game = SimpleNamespace(game_id=game_id, levels=levels, total_levels=total, level_tokens=list(level_tokens))
    game.tokens_on_level = lambda: spent
    return game


def test_posterior_learns_from_every_level_and_the_index_prefers_fast_deep_games():
    cfg = settings()
    model = GittinsModel([], cfg)
    fast = record("fast", 3, [6000, 9000, 12000], 2000)
    fresh = record("fresh", 0, [], 6000)
    stuck = record("stuck", 0, [], 40000)
    mean = {g.game_id: float(model.posterior(g) @ model.grid) for g in (fast, fresh, stuck)}
    assert mean["fast"] > mean["fresh"] > mean["stuck"]
    # A clear does not reset the game's evidence: the game that just cleared its third level (0 tokens on
    # the fourth) outranks a fresh level-0 game and one stuck at 40K tokens.
    just_cleared = record("just", 3, [6000, 9000, 12000], 0)
    index = {g.game_id: model.index(g, 3000)[0] for g in (just_cleared, fresh, stuck)}
    assert index["just"] > index["fresh"] > index["stuck"] > 0
    score, detail = model.index(just_cleared, 3000)
    assert detail["plan"]["levels"] >= 1 and detail["plan"]["cap_tokens"] in cfg["caps"]
    # Common random numbers: the same record gives the same index.
    assert model.index(just_cleared, 3000)[0] == score
    # Nothing left to play, or no time left: zero.
    assert model.index(record("done", 8, [5000] * 8, 0), 3000)[0] == 0
    assert model.index(fresh, 0)[0] == 0
    # Past the largest cap on the current level no plan continues it.
    assert model.index(record("over", 1, [5000], 170000), 3000) == (0.0, {
        "posterior_mean": pytest.approx(model.index(record("over", 1, [5000], 170000), 3000)[1]["posterior_mean"]),
        "spent_tokens": 170000, "plan": None})


def test_settings_are_strict():
    assert settings()["caps"] == [float(c) for c in DEFAULTS["caps"]]
    with pytest.raises(ValueError, match="Unknown"):
        settings({"horizon": 1})
    with pytest.raises(ValueError, match="model"):
        settings({"model": "hazard"})
    with pytest.raises(ValueError):
        settings({"caps": []})
    with pytest.raises(ValueError):
        settings({"k": 0})


def test_broad_opening_waits_for_every_game_then_keeps_the_best_indexes():
    now, wall, games, sched = build("index", index_settings={"model": "gittins"}, opening_min_tokens=15000)
    g0, g1, g2, g3 = games
    for g in games:
        charged(now, g, 20, 6000)
    for g in (g0, g1, g2):
        g.report_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "opening"  # three clears, but every game has only 6K tokens
    charged(now, g2, 10, 3000)
    g2.report_level(2, wall.elapsed())  # g2 clears its second level in 3K tokens
    for g in games:
        charged(now, g, 20, 9000 if g is not g3 else 5000)
    sched.tick()
    assert sched.phase == "opening"  # g3 has 11K
    charged(now, g3, 20, 4000)
    sched.tick()
    assert sched.phase == "steady" and sched.switch_reason == "clears"
    # Kept by index, not by first-clear time: g2 (two fast clears) and g0 (roster order over g1);
    # g3 (15K tokens, no clear) waits.
    assert {g.game_id for g in games if g.state == "active"} == {"g0", "g2"}
    assert sched.summary()["opening_min_tokens"] == 15000
    audit = audit_scheduler(report(games, sched, wall))
    assert audit["valid"], audit["issues"]


def test_broad_opening_cap_and_validation():
    now, wall, games, sched = build("index", index_settings={"model": "gittins"}, opening_min_tokens=15000,
                                    opening_max_wall_seconds=100)
    for g in games:
        charged(now, g, 10, 2000)
    games[0].report_level(1, wall.elapsed())
    games[1].report_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "opening"
    now.advance(60.5)  # ticks run every second: the switch lands within 1.1 s of the cap
    sched.tick()
    assert sched.phase == "steady" and sched.switch_reason == "opening_cap"
    audit = audit_scheduler(report(games, sched, wall))
    assert audit["valid"], audit["issues"]
    with pytest.raises(ValueError, match="opening_min_tokens"):
        Scheduler(games, wall, opening_min_tokens=0)


def test_gittins_policy_goes_deep_and_swaps_out_a_stalled_game():
    now, wall, games, sched = build("index", index_settings={"model": "gittins"})
    g0, g1, g2, g3 = games
    for g, tokens in ((g0, 6000), (g1, 7000), (g2, 5000), (g3, 5000)):
        charged(now, g, 20, tokens)
    g0.report_level(1, wall.elapsed())
    g1.report_level(1, wall.elapsed())
    sched.tick()
    for step in range(12):
        for g in (g0, g1):
            if g.state == "active":
                charged(now, g, 60, 12000 if g is g0 else 5000)
        if g1.state == "active" and step % 2 == 1 and g1.levels < 6:
            g1.report_level(g1.levels + 1, wall.elapsed())
        sched.tick()
    assert g1.state == "active" and g1.levels >= 5
    assert g0.state == "parked" and g0.admissions[-1]["reason"] == "index" and sched.index_swaps >= 1
    history = sched.summary()["index_history"]
    assert history and history[-1]["model"]["model"] == "gittins"
    audit = audit_scheduler(report(games, sched, wall))
    assert audit["valid"], audit["issues"]


def test_variant_changes_only_the_scheduler_and_builds(tmp_path):
    study = apply_variant(CONTROL, "fixed-2x-gittins-broad")
    control = apply_variant(CONTROL, "reclaim-fixed-2x")
    meta = study.pop("arena_variant")
    control.pop("arena_variant")
    assert {k for k in study.keys() | control.keys() if study.get(k) != control.get(k)} == {
        "level_budget_policy", "index_scheduler", "opening_min_tokens", "opening_max_wall_seconds"}
    assert study["index_scheduler"]["model"] == "gittins" and study["opening_min_tokens"] == 15000
    assert meta["checkpoint_and_serving_unchanged"] and meta["reference_variant"] == "reclaim-fixed-2x"
    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    _, built = builder.build("gittins-test", paired_30m=True, output=tmp_path / "g",
                             arena_variant="fixed-2x-gittins-broad", opening_burst_requests=3,
                             exclude_games=["ft09", "lp85"], wall_seconds=3600)
    assert built["level_budget_policy"] == "index" and "giving up on a level" in built["level_budget_rule"]
    assert "15000 completion tokens" in built["level_budget_rule"]
    assert json.loads((tmp_path / "g/configs/comparison.json").read_text())["opening_min_tokens"] == 15000
