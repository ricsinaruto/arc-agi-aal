"""Hazard-index slot allocation (fixed-2x-index) and the clipped T1 unit (fixed-2x-token-budget):
tables, index arithmetic, scheduler behaviour and the scheduler audit."""

from types import SimpleNamespace

import pytest
from test_arena_core import Now

from arena.clock import ActiveClock, GlobalClock
from arena.hazard_index import RunModel, settings
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler


def record(levels, level_tokens, spent, total=8):
    game = SimpleNamespace(levels=levels, total_levels=total, level_tokens=list(level_tokens))
    game.tokens_on_level = lambda: spent
    return game


def test_online_model_is_estimated_from_the_run_and_scale_free():
    import math

    cfg = settings()
    games = [record(1, [4000], 2000, total=8), record(2, [6000, 10000], 30000), record(0, [], 5000),
             record(0, [], 12000)]
    for i, g in enumerate(games):
        g.game_id = f"g{i}"
    model = RunModel(games, cfg)
    # Scale: ln 2 x all tokens on levels / clears (unfinished attempts are exposure too).
    assert model.scale == pytest.approx(math.log(2) * 69000 / 3)
    assert model.events == {"first": 2, "deep": 1}

    def scaled(k):
        out = [record(g.levels, [k * t for t in g.level_tokens], k * g.tokens_on_level()) for g in games]
        for i, g in enumerate(out):
            g.game_id = f"g{i}"
        return out
    # Every token count x3: the same rates, probabilities and decisions (the rule carries to game
    # sets whose levels cost other amounts of tokens).
    tripled_games = scaled(3)
    tripled = RunModel(tripled_games, cfg)
    assert tripled.scale == pytest.approx(3 * model.scale) and tripled.level == pytest.approx(model.level)
    for g, h in zip(games, tripled_games):
        assert tripled.index(h, 10 ** 6)[0] == pytest.approx(model.index(g, 10 ** 6)[0] / 3)
    # Per-game evidence: the game that cleared two levels has a higher rate than the level-0 game
    # that spent 12K tokens without a clear, and more tokens without a clear lower a game's rate.
    rate = {g.game_id: model.game_rate(g)[0] for g in games}
    assert rate["g1"] > rate["g3"] and rate["g2"] > rate["g3"]
    stuck = record(0, [], 60000)
    stuck.game_id = "g3"
    assert RunModel(games[:3] + [stuck], cfg).game_rate(stuck)[0] < rate["g3"]
    # The recorded shape: far past the scale little is left; the flat shape keeps it constant.
    r = model.level["deep"]
    assert model.clear_probability(r, 4 * model.scale, model.scale) < 0.3 * model.clear_probability(r, model.scale, model.scale)
    flat = RunModel(games, settings({"shape": "flat"}))
    r = flat.level["deep"]
    assert flat.clear_probability(r, 4 * flat.scale, flat.scale) == pytest.approx(flat.clear_probability(r, flat.scale, flat.scale))
    # Level 3 of an 8-level game (weights 1..8 of 36) plus later levels discounted by continuation q;
    # only one level fits when the time left is shorter than a level.
    q = 0.6
    assert model.level_value(2, 8, 10 ** 9, q) == pytest.approx(sum(k / 36 * q ** (k - 3) for k in range(3, 9)))
    assert model.level_value(2, 8, 1000, q) == pytest.approx(3 / 36) and model.level_value(8, 8, 10 ** 9, q) == 0
    with pytest.raises(ValueError):
        settings({"horizon": 1})
    with pytest.raises(ValueError):
        settings({"shape": "public-tables"})


def charged(now, game, seconds, tokens):
    """One request of ``seconds`` producing ``tokens`` completion tokens."""
    with game.clock.inference():
        now.advance(seconds)
    game.clock.charge(1000, tokens)


def build(policy, count=4, **kwargs):
    now = Now()
    wall = GlobalClock(3600, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="busy"), total_levels=7)
             for i in range(count)]
    sched = Scheduler(games, wall, budget_policy=policy, backlog_policy="readmissions", **kwargs)
    for g in games:
        g.observer = sched
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    return now, wall, games, sched


def report(games, sched, wall):
    return {"wall_seconds": wall.elapsed(), "config": {"gameplay_seconds": 3600},
            "games": [game_row(g, {"game_id": g.game_id}) for g in games], "scheduler": sched.summary()}


def test_index_policy_goes_deep_then_opens_the_search_when_a_level_stalls():
    now, wall, games, sched = build("index", index_settings={})
    g0, g1, g2, g3 = games
    for g, tokens in ((g0, 4000), (g1, 6000), (g2, 5000), (g3, 3000)):
        charged(now, g, 20, tokens)
    g0.report_level(1, wall.elapsed())
    g1.report_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "steady" and {g.state for g in (g0, g1)} == {"active"}
    assert g0.admissions[-1]["reason"] == "opening_switch" and g0.clock.budget is None
    assert g0.level_tokens == [4000] and g0.tokens_on_level() == 0
    # g1 keeps clearing fast (promising); g0 spends token after token on level 2 without a clear.
    for step in range(12):
        for g in (g0, g1):
            if g.state == "active":
                charged(now, g, 60, 8000 if g is g0 else 4000)
        if g1.state == "active" and step % 2 == 1 and g1.levels < 6:
            g1.report_level(g1.levels + 1, wall.elapsed())
        sched.tick()
    assert g1.state == "active" and g1.levels >= 5
    assert g0.state == "parked" and g0.admissions[-1]["reason"] == "index" and sched.index_swaps >= 1
    entered = next(g for g in (g2, g3) if g.state == "active")
    assert entered.admissions[-1]["reason"] == "index" and entered.clock.budget is None
    assert sched.summary()["index_history"] and sched.summary()["index_settings"]["hysteresis"] == 0.25
    assert sched.summary()["index_history"][-1]["model"]["scale_tokens"] > 0
    audit = audit_scheduler(report(games, sched, wall))
    assert audit["valid"], audit["issues"]
    assert audit["checks"]["index_admissions"] >= 3


def test_index_quantum_blocks_early_swaps_and_settings_need_the_policy():
    now, wall, games, sched = build("index", index_settings={"quantum_scale": 1000})
    g0, g1, g2, g3 = games
    for g in games:
        charged(now, g, 10, 3000)
    g0.report_level(1, wall.elapsed())
    g1.report_level(1, wall.elapsed())
    sched.tick()
    charged(now, g0, 600, 150000)
    sched.tick()
    assert g0.state == "active" and sched.index_swaps == 0
    with pytest.raises(ValueError, match="index budget policy"):
        Scheduler(games, wall, budget_policy="fixed-2x", index_settings={})
    with pytest.raises(ValueError, match="readmissions"):
        Scheduler(games, wall, budget_policy="index", backlog_policy="spent")


def test_clipped_t1_budgets_and_the_audit_follow_the_median_of_the_cleared_games():
    now, wall, games, sched = build("fixed-2x", t1_clip=[0.5, 2.0])
    g0, g1, g2, g3 = games
    charged(now, g0, 20, 1000)   # g0 clears at 20 busy s
    g0.report_level(1, wall.elapsed())
    charged(now, g1, 200, 1000)  # g1 clears at 200 busy s; median T1 = 110
    g1.report_level(1, wall.elapsed())
    sched.tick()
    assert sched.phase == "steady"
    assert sched.budget_unit(g0) == pytest.approx(55) and sched.budget_unit(g1) == pytest.approx(200)
    assert g0.clock.budget == pytest.approx(2 * 2 * 55 - 0)
    data = report(games, sched, wall)
    audit = audit_scheduler(data)
    assert audit["valid"], audit["issues"]
    data["scheduler"]["t1_clip"] = None  # without the clip the recorded grants no longer match
    assert not audit_scheduler(data)["valid"]
