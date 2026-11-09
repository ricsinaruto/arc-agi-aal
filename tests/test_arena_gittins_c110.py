"""fixed-2x-gittins-c110: the within-level hazard shape and in-run prior scale in arena.gittins, the variant,
its build on the 110-clone roster, and the absolute opening cap in arena.competition."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_arena_hazard_index import build, charged, report

from arena import competition
from arena.gittins import (
    BROAD,
    C110,
    C110_OPENING_MAX_WALL_SECONDS,
    DEFAULTS,
    H0_DEEP,
    H0_DEEP_WITHIN,
    H0_FIRST,
    H0_FIRST_WITHIN,
    GittinsModel,
    settings,
)
from arena.scheduler_audit import audit_scheduler
from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
EXCLUDED = ["ft09", "lp85", "ar25", "vc33", "tu93"]


def record(game_id, levels, level_tokens, spent, total=8):
    game = SimpleNamespace(game_id=game_id, levels=levels, total_levels=total, level_tokens=list(level_tokens))
    game.tokens_on_level = lambda: spent
    return game


def builder():
    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shape_option_keeps_the_recorded_pooled_model_reachable():
    # A settings dict without the new keys (every recorded study) is the pooled model, bit for bit.
    assert settings(dict(BROAD)) == settings() and settings()["shape"] == "pooled"
    assert settings()["prior_scale"] == "fixed"
    pooled = GittinsModel([], settings(dict(BROAD)))
    assert np.array_equal(pooled.h0["first"], np.array(H0_FIRST)) and np.array_equal(pooled.h0["deep"], np.array(H0_DEEP))
    within = GittinsModel([], settings({"shape": "within"}))
    assert np.array_equal(within.h0["first"], np.array(H0_FIRST_WITHIN))
    assert np.array_equal(within.h0["deep"], np.array(H0_DEEP_WITHIN))
    # Within a level the fitted clear rate rises with tokens through 40K (the pooled curve falls there).
    assert H0_FIRST_WITHIN[3] > H0_FIRST_WITHIN[2] > H0_FIRST_WITHIN[1] > H0_FIRST_WITHIN[0]
    assert H0_DEEP_WITHIN[3] > H0_DEEP_WITHIN[1] and H0_FIRST[4] < H0_FIRST[3]
    game = record("g", 1, [9000], 12000)
    assert within.index(game, 3000)[0] != pooled.index(game, 3000)[0]
    assert within.summary()["shape"] == "within" and pooled.summary()["shape"] == "pooled"
    with pytest.raises(ValueError, match="shape"):
        settings({"shape": "falling"})
    with pytest.raises(ValueError, match="prior_scale"):
        settings({"prior_scale": "per-game"})


def test_run_prior_scale_learns_the_roster_difficulty_from_this_run():
    cfg = settings({**C110})
    hard = [record(f"h{i}", 0, [], 30000, 7) for i in range(40)] + [record("a", 1, [9000], 1000)]
    easy = [record(f"e{i}", 2, [6000, 8000], 2000, 7) for i in range(40)]
    fixed = GittinsModel(hard, settings({**C110, "prior_scale": "fixed"}))
    low, high = GittinsModel(hard, cfg), GittinsModel(easy, cfg)
    assert fixed.scale == 1.0 and low.scale < 0.5 < 1.5 < high.scale
    assert GittinsModel([], cfg).scale == 1.0  # no evidence: the fitted prior
    fresh = record("f", 0, [], 0, 7)
    # A roster where long attempts fail lowers a fresh game's value; an easy one raises it.
    assert low.index(fresh, 3000)[0] < fixed.index(fresh, 3000)[0] < high.index(fresh, 3000)[0]


def test_c110_scheduler_commits_through_the_hysteresis_and_audits():
    now, wall, games, sched = build("index", count=6, index_settings=dict(C110),
                                    opening_max_wall_seconds=C110_OPENING_MAX_WALL_SECONDS)
    g0, g1, *rest = games
    for g in games:
        charged(now, g, 20, 3000)
    g0.report_level(1, wall.elapsed())
    g1.report_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "steady" and sched.switch_reason == "clears"
    assert {g.game_id for g in games if g.state == "active"} == {"g0", "g1"}
    # g1 stalls 30K tokens on level 2: a 0.15 hysteresis would hand its slot to a fresh game; 4x keeps it.
    for _ in range(5):
        charged(now, g1, 30, 6000)
        charged(now, g0, 30, 6000)
        sched.tick()
    assert g1.state == "active" and sched.index_swaps == 0
    history = sched.summary()["index_history"]
    assert history[-1]["model"]["shape"] == "within"
    audit = audit_scheduler(report(games, sched, wall))
    assert audit["valid"], audit["issues"]


def test_c110_variant_changes_only_the_scheduler_and_broad_stays_frozen():
    study = apply_variant(CONTROL, "fixed-2x-gittins-c110")
    control = apply_variant(CONTROL, "reclaim-fixed-2x")
    meta = study.pop("arena_variant")
    control.pop("arena_variant")
    assert {k for k in study.keys() | control.keys() if study.get(k) != control.get(k)} == {
        "level_budget_policy", "index_scheduler", "opening_max_wall_seconds", "opening_cap_absolute"}
    assert study["index_scheduler"] == C110 and study["opening_cap_absolute"] is True
    assert study["opening_max_wall_seconds"] == C110_OPENING_MAX_WALL_SECONDS and "opening_min_tokens" not in study
    assert settings(study["index_scheduler"])["shape"] == "within"
    assert meta["reference_variant"] == "reclaim-fixed-2x" and meta["checkpoint_and_serving_unchanged"]
    # fixed-2x-gittins-broad resolves exactly as recorded: no new keys, pooled shape, fixed prior.
    broad = apply_variant(CONTROL, "fixed-2x-gittins-broad")
    assert broad["index_scheduler"] == BROAD and "shape" not in broad["index_scheduler"]
    assert "opening_cap_absolute" not in broad and set(DEFAULTS) - set(BROAD) == {"shape", "prior_scale"}


def test_c110_variant_builds_on_the_110_clone_roster(tmp_path):
    build_arena = builder()
    _, built = build_arena.build("c110-test", paired_30m=True, output=tmp_path / "c", arena_variant="fixed-2x-gittins-c110",
                                 opening_burst_requests=3, exclude_games=EXCLUDED, wall_seconds=3600, clone_instances=110)
    assert len(built["games"]) == 20 and built["competition"]["preview_instances"] == 110
    assert built["index_scheduler"]["shape"] == "within" and built["opening_cap_absolute"] is True
    assert "within-level" in built["level_budget_rule"] and "not rescaled" in built["level_budget_rule"]
    saved = json.loads((tmp_path / "c/configs/comparison.json").read_text())
    assert saved["opening_max_wall_seconds"] == C110_OPENING_MAX_WALL_SECONDS and saved["opening_cap_absolute"]


def test_opening_cap_scales_with_the_clone_roster_unless_absolute():
    base = {"games": ["a"] * 25, "competition": {}}
    scaled = {**base, "competition": {}, "opening_max_wall_seconds": 360}
    resolved = json.loads(json.dumps(scaled))
    competition.resolve_opening_cap(resolved, scaled, 110)
    assert resolved["opening_max_wall_seconds"] == 1584.0
    assert resolved["competition"]["opening_cap_scaled"]["per_25_games_seconds"] == 360
    absolute = {**base, "games": ["a"] * 20, "competition": {}, "opening_max_wall_seconds": 1200,
                "opening_cap_absolute": True}
    resolved = json.loads(json.dumps(absolute))
    competition.resolve_opening_cap(resolved, absolute, 110)
    assert resolved["opening_max_wall_seconds"] == 1200.0
    assert resolved["competition"]["opening_cap_scaled"] == {"absolute": True, "roster": 110, "seconds": 1200.0}
    plain = {**base, "competition": {}}
    resolved = json.loads(json.dumps(plain))
    competition.resolve_opening_cap(resolved, plain, 110)
    assert "opening_max_wall_seconds" not in resolved and "opening_cap_scaled" not in resolved["competition"]


def test_open_arcade_keeps_the_c110_cap_on_the_local_clone_gateway(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    from taaf import competition_arcade

    class Server:
        base_url = "http://local"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return self

        def stop(self):
            pass

    monkeypatch.setattr(competition_arcade, "CompetitionArcadeServer", Server)
    clones = [SimpleNamespace(game_id=f"k{i:03d}") for i in range(110)]
    monkeypatch.setattr(competition, "gateway_arcade", lambda *a: SimpleNamespace(get_environments=lambda: clones))
    study = apply_variant(CONTROL, "fixed-2x-gittins-c110")
    study["games"] = study["games"][:20]
    study["competition"] = {"preview_instances": 110, "preview_seconds": 3600, "submission_seconds": 32400,
                            "cleanup_seconds": 600, "max_actions_per_game": 100000, "research_clones": True}
    with competition.open_arcade(study, tmp_path / "games", tmp_path / "out") as (resolved, _):
        assert len(resolved["games"]) == 110
        assert resolved["opening_max_wall_seconds"] == C110_OPENING_MAX_WALL_SECONDS  # not x 5.5
    study.pop("opening_cap_absolute")
    with competition.open_arcade(study, tmp_path / "games", tmp_path / "out2") as (resolved, _):
        assert resolved["opening_max_wall_seconds"] == C110_OPENING_MAX_WALL_SECONDS * 110 / 20


def test_wide_combination_fills_six_slots_after_two_clears_and_keeps_both_parents():
    study = apply_variant(CONTROL, "fixed-2x-gittins-c110-wide")
    wide = apply_variant(CONTROL, "fixed-2x-mtp0-wide")
    index = apply_variant(CONTROL, "fixed-2x-gittins-c110")
    assert study["server"] == wide["server"] and study["server"]["mtp_tokens"] == 0
    assert study["active_limit"] == 6 and study["windows"] == [65536] and study["require_focus_windows_fit"]
    assert study["opening_switch_level1_games"] == 2 and study["opening_cap_absolute"]
    assert study["index_scheduler"] == {**index["index_scheduler"], "stream_tokens_per_second": 75.0}
    assert not study["arena_variant"]["checkpoint_and_serving_unchanged"]
    now, wall, games, sched = build("index", count=10, active_limit=6, index_settings=study["index_scheduler"],
                                    opening_max_wall_seconds=C110_OPENING_MAX_WALL_SECONDS)
    for g in games:
        charged(now, g, 20, 3000)
    games[0].report_level(1, wall.elapsed())
    games[1].report_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "steady" and sched.switch_reason == "clears"
    now.advance(1)
    sched.tick()  # the free focus slots fill on the next tick (index_fill)
    active = {g.game_id for g in games if g.state == "active"}
    assert len(active) == 6 and {"g0", "g1"} <= active
    audit = audit_scheduler(report(games, sched, wall))
    assert audit["valid"], audit["issues"]
