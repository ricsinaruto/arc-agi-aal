"""hero-v2-rank: the continuous token/clear ranking (``arena.rank``, ``Scheduler(budget_policy="rank")``): the score
matches the simulator's, no opening, fills and swaps on the tick with hysteresis and tenure, refills between ticks, no
level budgets, the audit, and the variant."""

import json
import math
import sys
from pathlib import Path

import pytest
from test_arena_core import Now
from test_arena_hazard_index import charged

from arena.clock import ActiveClock, GlobalClock
from arena.rank import DEFAULTS, horizon, score, settings
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler
from arena.variants import RANK_SERVING, VARIANTS, _rank_variant, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
sys.path.insert(0, str(ROOT / "scripts/arena_analysis"))


def build(count=6, **rank):
    now = Now()
    wall = GlobalClock(3600, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="busy"), total_levels=7)
             for i in range(count)]
    for g in games:
        g.clock.pause()
    sched = Scheduler(games, wall, budget_policy="rank", backlog_policy="readmissions", rank={"slots": 2, **rank})
    for g in games:
        g.observer = sched
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    return now, wall, games, sched


def reasons(game):
    return [(e["event"], e["reason"]) for e in game.admissions]


def report(games, sched, wall):
    return {"wall_seconds": wall.elapsed(), "config": {"gameplay_seconds": 3600},
            "games": [game_row(g, {"game_id": g.game_id}) for g in games], "scheduler": sched.summary()}


def test_defaults_are_the_simulator_tuned_constants_and_settings_validate():
    import sim_hero_v2 as hv

    sel = {"horizon_rel": 1.0, "c0": 0.25, "t0": 5e3, "gamma": 1.5, "hysteresis": 0.05, "tenure": 10e3,
           "clears": "all", "opening": "none", "tick": 60.0}
    assert hv.tag({**sel, "horizon": 40e3}) == "rank hr1 c0.25 t5K g1.5 h0.05 q10K all none"
    assert (DEFAULTS["horizon_rel"], DEFAULTS["prior_clears"], DEFAULTS["prior_tokens"], DEFAULTS["depth_power"],
            DEFAULTS["hysteresis"], DEFAULTS["tenure_tokens"], DEFAULTS["tick_seconds"], DEFAULTS["clears"],
            DEFAULTS["horizon_tokens"]) == (1.0, 0.25, 5000, 1.5, 0.05, 10000, 60.0, "all", 40000)
    assert settings() == DEFAULTS and settings({"horizon_rel": None})["horizon_rel"] is None
    for bad in ({"slots": 0}, {"window": 1.5}, {"prior_tokens": 0}, {"hysteresis": -1}, {"clears": "first"},
                {"tick_seconds": 0.5}, {"horizon_rel": 0}, {"surprise": 1}, {"tenure_tokens": 1.5}):
        with pytest.raises(ValueError):
            settings(bad)


def test_score_matches_the_simulator_rank_score():
    import schedule_sim as sim

    now, wall, games, sched = build()
    g = games[0]
    charged(now, g, 5, 7000)
    g.report_level(1, wall.elapsed())
    charged(now, g, 5, 3000)
    g.report_level(3, wall.elapsed())  # two levels at once: a zero-token level
    charged(now, g, 5, 4500)
    assert g.level_tokens == [7000, 3000, 0] and g.tokens_on_level() == 4500
    twin = sim.Game("x", 0, 7, [1e9] * 7, levels=3, tok=4500.0, level_tokens=[7000.0, 3000.0, 0.0], spent=14500.0)
    for clears in ("all", "last"):
        for h in (2000.0, 7000.0, 40000.0):
            cfg = settings({"clears": clears})
            ours = score(g, h, cfg)[0]
            theirs = sim.rank_score(twin, h, cfg["prior_clears"], cfg["prior_tokens"], cfg["depth_power"], clears)
            assert math.isclose(ours, theirs, rel_tol=1e-12)
    # H = the median tokens per cleared level (zero-token levels excluded) x horizon_rel, floored; 40K before a clear.
    assert horizon(games, settings()) == 5000.0
    assert horizon(games, settings({"horizon_rel": 0.1})) == 2000.0
    assert horizon(games[1:], settings()) == 40000.0


def test_recent_clears_rank_first_then_deeper_levels_and_unplayed_games_score_the_prior():
    now, wall, games, sched = build()
    cfg, h = settings(), 10000.0
    fresh = score(games[5], h, cfg)[0]
    assert math.isclose(fresh, 1 / 28 * 0.25 / 5000)  # (0 + 1)^1.5 / (7 * 8 / 2) x prior rate
    a, b = games[0], games[1]
    for g in (a, b):
        charged(now, g, 5, 8000)
        g.report_level(1, wall.elapsed())
    charged(now, a, 5, 1000)
    charged(now, b, 5, 30000)
    assert score(a, h, cfg)[0] > score(b, h, cfg)[0]  # a cleared 1K tokens ago, b 30K ago
    deep = games[2]
    for _ in range(2):
        charged(now, deep, 5, 8000)
        deep.report_level(deep.levels + 1, wall.elapsed())
    charged(now, deep, 5, 1000)
    assert score(deep, h, cfg)[0] > score(a, h, cfg)[0]  # same recency, one level deeper
    stuck = games[3]
    charged(now, stuck, 5, 40000)
    assert score(stuck, h, cfg)[0] < fresh  # 40K tokens without a clear rank below an unplayed game


def test_no_opening_the_top_games_take_the_slots_at_the_release_without_budgets():
    now, wall, games, sched = build()
    assert sched.phase == "steady" and sched.switch_wall_seconds == 0.0 and sched.active_limit == 2
    assert [g.state for g in games] == ["active"] * 2 + ["backlog"] * 4
    assert all(reasons(g) == [("admit", "rank_start")] for g in games[:2])
    assert all(g.clock.budget is None for g in games[:2]) and all(g.clock.parked for g in games[2:])
    with pytest.raises(ValueError, match="rank policy"):
        Scheduler(games, wall, budget_policy="rank", backlog_policy="spent")
    with pytest.raises(ValueError, match="rank settings"):
        Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions", rank={})


def test_swaps_wait_for_the_tick_and_the_tenure_and_need_the_hysteresis_margin():
    now, wall, games, sched = build()
    g0, g1, g2 = games[:3]
    charged(now, g0, 5, 12000)  # past the tenure, no clear: below an unplayed game
    charged(now, g1, 5, 8000)
    g1.report_level(1, wall.elapsed())
    sched.tick()
    assert g0.state == "active"  # the tick period has not passed
    now.advance(60)
    sched.tick()
    assert g0.state == "parked" and reasons(g0)[-1] == ("park", "rank")
    assert g2.state == "active" and reasons(g2) == [("admit", "rank")] and g2.clock.budget is None
    assert g1.state == "active" and g1.clock.budget is None  # a clear grants no budget
    assert sched.rank_counts["swaps"] == 1 and sched.rank_swaps[0]["out"]["game"] == "g0"
    # g2 is under its tenure, g1 has a fresh clear: nothing moves at the next tick.
    now.advance(60)
    sched.tick()
    assert [g.game_id for g in sched.focus()] == ["g1", "g2"] and sched.rank_counts["ticks"] == 3
    # A waiting game within the hysteresis margin does not displace a focus game.
    now2, wall2, games2, sched2 = build(hysteresis=10.0)
    charged(now2, games2[0], 5, 12000)
    now2.advance(60)
    sched2.tick()
    assert games2[0].state == "active" and sched2.rank_counts["swaps"] == 0


def test_a_finished_games_slot_refills_between_ticks_from_the_latest_ranking():
    now, wall, games, sched = build()
    games[0].note_finished("won")
    sched.tick()
    assert games[0].state == "finished" and games[2].state == "active"
    assert reasons(games[2]) == [("admit", "rank_fill")] and sched.rank_counts["refills"] == 1


def test_audit_accepts_the_policy_and_rejects_budgets_late_starts_and_other_policies():
    now, wall, games, sched = build()
    charged(now, games[0], 5, 12000)
    charged(now, games[1], 5, 8000)
    games[1].report_level(1, wall.elapsed())
    now.advance(60)
    sched.tick()
    charged(now, games[2], 5, 11000)
    now.advance(60)
    sched.tick()
    sched.wall_end()
    rows = report(games, sched, wall)
    audit = audit_scheduler(rows)
    assert audit["valid"], audit["issues"]
    assert audit["checks"]["rank_admissions"] >= 4
    summary = sched.summary()["rank"]
    assert summary["settings"]["slots"] == 2 and summary["swaps"] >= 1 and summary["games_played"] == 3
    assert summary["history"] and summary["history"][0]["top"]
    bad = report(games, sched, wall)
    next(e for e in bad["games"][2]["admissions"] if e["reason"] == "rank")["budget"] = 100.0
    assert not audit_scheduler(bad)["valid"]
    other = report(games, sched, wall)
    other["scheduler"]["budget_policy"] = "fixed-2x"
    assert not audit_scheduler(other)["valid"]
    late = report(games, sched, wall)
    late["scheduler"]["switch_wall_seconds"] = 30.0
    assert not audit_scheduler(late)["valid"]


def test_variant_is_the_pruned_hero_serving_with_the_rank_scheduler():
    assert "hero-v2-rank" in VARIANTS and RANK_SERVING["hero-v2-rank"] == "fixed-2x-mtp0-wide-kv8-hero-prune25"
    result = apply_variant(CONTROL, "hero-v2-rank")
    source = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kv8-hero-prune25")
    assert result["server"] == source["server"] and result["server"]["expert_prune"]
    assert result["level_budget_policy"] == "rank" and result["rank"] == {**DEFAULTS, "slots": 25, "window": 65536}
    assert result["active_limit"] == 25 and result["windows"] == [65536] and result["require_focus_windows_fit"]
    same = {k: v for k, v in result.items() if k not in ("level_budget_policy", "rank", "arena_variant",
                                                         "rank_serving_source")}
    assert same == {k: v for k, v in source.items() if k not in ("arena_variant", "level_budget_policy")}
    meta = result["arena_variant"]
    assert meta["name"] == "hero-v2-rank" and not meta["checkpoint_and_serving_unchanged"]
    assert set(meta["reference_difference"]) == {*source["arena_variant"]["reference_difference"],
                                                 "level_budget_policy", "rank"}
    for bad in ("fixed-2x-depthwise", "fixed-2x-deep-explore", "fixed-2x-gittins-c110"):
        with pytest.raises(ValueError, match="rank arm"):
            _rank_variant(CONTROL, bad)
