"""Token budgets: the level budgets count completion tokens instead of busy seconds, so
a budget buys the same model work at any number of running streams. The clock, the fixed-2x / incremental / depthwise
grants and floors in tokens, parked games, compaction and aside requests, the audit on a recorded token trace, the
LLM-rank evidence, the builds (default tokens, ``--budget-clock time`` reproduces the old clock) and the exclusions."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_arena_core import Now

from arena import llm_rank
from arena.clock import ActiveClock, GlobalClock
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler
from arena.timing_analysis import inference_timing

ROOT = Path(__file__).resolve().parents[1]
FLOOR = Scheduler.TOKEN_FLOOR


def builder():
    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reply(now, game, tokens, seconds=10.0, purpose="gameplay", prompt=4000):
    """One completed request: ``seconds`` of latency, ``tokens`` completion tokens charged as ``purpose``."""
    with game.clock.inference(purpose):
        now.advance(seconds)
    game.clock.charge(prompt, tokens, purpose)


def games_on(clock_kind, count=6, now=None, **clock_kw):
    now = now or Now()
    wall = GlobalClock(3600, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock=clock_kind, **clock_kw), total_levels=7)
             for i in range(count)]
    for g in games:
        g.clock.pause()
    return now, wall, games


def start(games, wall, sched):
    for g in games:
        g.observer = sched
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()


def test_token_clock_counts_gameplay_completion_tokens_not_seconds():
    now, wall, (game,) = games_on("tokens", count=1)
    wall.release()
    game.clock.release()
    game.clock.resume()
    clock = game.clock
    assert clock.budget_unit == "completion_tokens" and clock.token_purposes == ("gameplay",)
    reply(now, game, 700, seconds=40)  # a slow reply at 21 streams ...
    reply(now, game, 700, seconds=11)  # ... and a fast one at 6 buy the same budget
    with clock.tool():
        now.advance(30)  # tool execution is not charged
    now.advance(500)  # neither is waiting or thinking between requests
    reply(now, game, 1600, seconds=60, purpose="compaction")  # a compaction summary: harness overhead
    clock.charge(3000, 300, "notes")  # a notes aside
    assert clock.budget_seconds() == 1400.0 and clock.budget_tokens() == 1400
    assert clock.completion_tokens() == 3300  # every purpose (the hazard index and level_tokens use this)
    assert clock.inference_seconds() == 111 and clock.tool_seconds() == 30
    clock.grant(2000)
    reply(now, game, 450)
    assert clock.budget_spent() == 450 and clock.budget_remaining() == 1550 and not clock.budget_exhausted()
    reply(now, game, 1550)
    assert clock.budget_exhausted()
    feedback = clock.feedback()
    assert feedback["level_budget_clock"] == "tokens" and feedback["level_budget_tokens"] == 2000
    assert feedback["level_budget_remaining_tokens"] == 0 and "level_budget_seconds" not in feedback
    # Counting compaction too is one setting away (recorded in study.json when used).
    both = ActiveClock(wall, now=now, budget_clock="tokens", token_purposes=["gameplay", "compaction"])
    both.charge(1000, 100, "gameplay")
    both.charge(1000, 50, "compaction")
    both.charge(1000, 25, "memory")
    assert both.budget_seconds() == 150
    for bad in (dict(budget_clock="busy", token_purposes=["gameplay"]), dict(budget_clock="tokens", token_purposes=[])):
        with pytest.raises(ValueError, match="purposes"):
            ActiveClock(wall, now=now, **bad)
    # The time clocks are unchanged: a charge only records, the budget stays seconds.
    busy = ActiveClock(wall, now=now, budget_clock="busy")
    busy.charge(1000, 5000)
    assert busy.budget_seconds() == 0 and busy.token_purposes is None and busy.budget_unit == "seconds"
    assert "level_budget_seconds" in busy.feedback() and "level_budget_tokens" not in busy.feedback()


def test_late_reply_charge_preserves_cutoff_and_audits_observation_order():
    now, wall, games = games_on("tokens", count=1)
    wall.seconds = 20
    scheduler = Scheduler(games, wall)
    start(games, wall, scheduler)
    game = games[0]
    reply(now, game, 1000, seconds=10)
    now.advance(10.01)
    scheduler.wall_end()
    now.advance(.1)
    game.clock.charge(4000, 45)
    assert game.clock.charges[-1]["wall_seconds"] == 20
    assert game.clock.charges[-1]["observed_wall_seconds"] == 20.11
    assert game.clock.budget_tokens() == game.clock.completion_tokens() == 1045
    report = dict(wall_seconds=20.05, config={"gameplay_seconds": 20},
                  games=[game_row(game, {}) | {"game_id": game.game_id}], scheduler=scheduler.summary())
    result = audit_scheduler(report)
    assert result["valid"], result["issues"]

    # Old records lack the evidence needed to distinguish a late charge.
    legacy = copy.deepcopy(report)
    for charge in legacy["games"][0]["charges"]:
        charge.pop("observed_wall_seconds")
    assert any("event budget clock" in issue for issue in audit_scheduler(legacy)["issues"])
    # An actual pre-finish charge still must appear in that finish clock.
    report["games"][0]["charges"][-1]["observed_wall_seconds"] = 20.001
    assert any("event budget clock" in issue for issue in audit_scheduler(report)["issues"])
    for invalid in (float("nan"), float("inf"), -1, -.001, 19):
        report["games"][0]["charges"][-1]["observed_wall_seconds"] = invalid
        assert any("Invalid charge observation time" in issue for issue in audit_scheduler(report)["issues"])


def test_parked_games_accrue_nothing_on_the_token_clock():
    now, wall, (game,) = games_on("tokens", count=1)
    wall.release()
    game.clock.release()
    game.clock.resume()
    reply(now, game, 1000)
    game.clock.grant(3000)
    game.clock.pause()
    now.advance(900)  # parked: wall time passes, no request is admitted
    assert game.clock.budget_seconds() == 1000 and game.clock.budget_remaining() == 3000
    assert game.clock.parked_seconds() == 900
    # A reply already in flight at the park is charged when it lands (as the busy clock charged its seconds).
    reply(now, game, 200)
    assert game.clock.budget_spent() == 200


def test_fixed_2x_on_the_token_clock_grants_2_l_t1_tokens_and_floors_the_level0_grant_at_1920():
    now, wall, games = games_on("tokens")
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions")
    assert sched.token_clock and sched.regrant_floor == FLOOR == 1920
    start(games, wall, sched)
    g0, g1, g2, g3 = games[:4]
    # The opening: slow replies (24 streams) cost many seconds but only their tokens count.
    reply(now, g0, 1200, seconds=60)
    g0.report_level(1, wall.elapsed())  # T1 = 1200 tokens (60 busy seconds)
    reply(now, g1, 800, seconds=40)
    reply(now, g1, 1000, seconds=45)
    reply(now, g2, 5000, seconds=220)
    reply(now, g3, 900, seconds=40)  # g3 barely played: its level-0 grant is floored
    reply(now, g0, 400, seconds=20)  # g0 spends 400 tokens on level 2 before the switch
    g1.report_level(1, wall.elapsed())  # T1 = 1800: the second clear switches
    sched.tick()
    assert sched.phase == "steady" and g0.t1 == 1200 and g1.t1 == 1800
    assert g0.time_to_level[0]["budget_seconds"] == 1200
    # The switch keeps the two fastest first clears in tokens with 2*2*T1 minus the level's spend.
    assert g0.clock.budget == 2 * 2 * 1200 - 400 and g1.clock.budget == 2 * 2 * 1800
    assert g2.level0_grant == 5000 and g3.level0_grant == FLOOR and games[4].level0_grant == FLOOR
    # A clear grants the full 2*L*T1 tokens.
    reply(now, g0, 2000)
    g0.report_level(2, wall.elapsed())
    assert g0.clock.budget == 2 * 3 * 1200 and g0.clock.budget_remaining() == 7200
    # Busy seconds no longer exhaust a budget: an hour of slow replies with few tokens keeps the slot.
    for _ in range(20):
        reply(now, g0, 100, seconds=120)
    sched.tick()
    assert g0.state == "active" and g0.clock.budget_spent() == 2000
    reply(now, g0, 5200)
    sched.tick()
    assert g0.state == "parked" and g2.state == "active" and g2.clock.budget == 5000
    assert sched.grant_for(g0) == 2 * 3 * 1200  # the full level budget again at its re-admission
    assert sched.grant_for(g3) == FLOOR
    summary = sched.summary()
    assert summary["budget_clock"] == "tokens" and summary["token_budget"] == {
        "unit": "completion_tokens", "floor_tokens": 1920, "reference_tokens_per_busy_second": 64.0,
        "purposes": ["gameplay"]}


def test_compaction_tokens_do_not_count_towards_t1_or_the_level_budget():
    now, wall, games = games_on("tokens", count=3)
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions")
    start(games, wall, sched)
    g0, g1 = games[:2]
    reply(now, g0, 1000)
    reply(now, g0, 1500, purpose="compaction")
    reply(now, g0, 500)
    g0.report_level(1, wall.elapsed())
    reply(now, g1, 3000)
    g1.report_level(1, wall.elapsed())
    sched.tick()
    assert g0.t1 == 1500 and g0.level_tokens == [3000]  # T1 in gameplay tokens; level_tokens count every purpose
    reply(now, g0, 4000, purpose="compaction")
    assert g0.spent_on_level() == 0 and not g0.clock.budget_exhausted()


def test_token_floor_replaces_every_30_second_constant_in_the_budget_path():
    # Incremental rule: regrant max(1.5*spent, spent + floor) and the renewable T1 cap's floor.
    now, wall, games = games_on("tokens", count=3)
    sched = Scheduler(games, wall, max_grant_t1_multiple=2)
    start(games, wall, sched)
    reply(now, games[2], 1000)
    assert sched.grant_for(games[2]) == max(1.5 * 1000, 1000 + FLOOR)
    games[0].t1 = 100.0
    assert sched.cap_grant(10_000, games[0]) == FLOOR  # max(2 * 100, floor)
    # A custom floor from the study, and the progress order's prior in the same unit (three floors).
    progress = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="progress", token_floor=1000)
    assert progress.regrant_floor == 1000 and progress.progress_prior == 3000
    assert progress.summary()["progress_order"]["prior_tokens"] == 3000
    assert Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="progress").progress_prior == 3 * FLOOR
    # Knobs defined in seconds are refused; so are mixed clocks and a token floor without the token clock.
    for knobs in (dict(max_grant_seconds=180), dict(budget_policy="fixed-2x", t1_cap_seconds=120)):
        with pytest.raises(ValueError, match="seconds"):
            Scheduler(games, wall, **knobs)
    busy_now, busy_wall, busy = games_on("busy", count=2)
    with pytest.raises(ValueError, match="token budget clock"):
        Scheduler([games[0], busy[0]], wall)
    with pytest.raises(ValueError, match="token_floor"):
        Scheduler(busy, busy_wall, token_floor=1920)
    # The time clocks keep the 30 s floor and their records.
    timed = Scheduler(busy, busy_wall, budget_policy="fixed-2x", backlog_policy="progress")
    assert timed.regrant_floor == 30.0 and timed.progress_prior == 90.0 and "token_budget" not in timed.summary()
    assert timed.summary()["progress_order"]["prior_seconds"] == 90.0


def test_depthwise_regrant_and_summary_in_tokens():
    now, wall, games = games_on("tokens", count=8)
    sched = Scheduler(games, wall, budget_policy="depthwise", backlog_policy="readmissions",
                      depthwise={"regrant": "spent"})
    start(games, wall, sched)
    g0 = games[0]
    reply(now, g0, 3000, seconds=100)
    g0.report_level(1, wall.elapsed())
    assert g0.slot == "depth" and g0.clock.budget == 2 * 2 * 3000
    reply(now, g0, 13_000)
    assert sched.dw_grant(g0) == max(1.5 * 13_000, 13_000 + FLOOR)
    summary = sched.depthwise_summary()
    assert summary["regrant_floor_tokens"] == FLOOR and "regrant_floor_seconds" not in summary


def token_report(now, wall, games, sched):
    return {"wall_seconds": wall.elapsed(), "config": {"gameplay_seconds": 3600},
            "games": [game_row(g, {"game_id": g.game_id}) for g in games], "scheduler": sched.summary()}


def test_the_audit_and_the_timing_analysis_reconcile_a_token_clock_trace():
    now, wall, games = games_on("tokens", count=4)
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions", active_limit=1,
                      switch_games=1)
    start(games, wall, sched)
    g0, g1, g2, g3 = games
    reply(now, g1, 800)
    reply(now, g2, 3000)
    reply(now, g0, 1000)
    reply(now, g0, 900, purpose="compaction")
    g0.report_level(1, wall.elapsed())  # the first clear switches (T1 = 1000); g1 and g3 floored, g2 3000
    sched.tick()
    for _ in range(4):
        reply(now, g0, 1000)
        sched.tick()
    assert g0.state == "parked" and g1.state == "active" and g1.clock.budget == FLOOR
    reply(now, g1, 2000)
    sched.tick()
    assert g1.state == "parked" and g2.state == "active"
    now.advance(10)
    sched.wall_end()
    report = token_report(now, wall, games, sched)
    assert report["games"][0]["budget_unit"] == "completion_tokens"
    out = audit_scheduler(report)
    assert out["valid"], out["issues"]
    assert out["checks"]["grants"] >= 3 and out["checks"]["budget_parks"] == 2 and out["clock_semantics"] == "tokens"
    assert inference_timing(report)["available"]
    # A grant off the rule, a floor changed after the run, and a missing charge are caught.
    bad = copy.deepcopy(report)
    next(e for e in bad["games"][1]["admissions"] if e["reason"] == "regrant")["budget"] = 30.0
    assert any("regrant grant" in i for i in audit_scheduler(bad)["issues"])
    bad = copy.deepcopy(report)
    bad["scheduler"]["token_budget"]["floor_tokens"] = 30
    assert not audit_scheduler(bad)["valid"]
    bad = copy.deepcopy(report)
    bad["games"][0]["charges"] = bad["games"][0]["charges"][1:]
    assert not audit_scheduler(bad)["valid"]
    bad = copy.deepcopy(report)
    del bad["scheduler"]["token_budget"]
    assert "token_budget settings" in audit_scheduler(bad)["issues"][0]


def test_llm_rank_evidence_speaks_tokens_on_the_token_clock():
    now, wall, games = games_on("tokens", count=3)
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="llm")
    start(games, wall, sched)
    reply(now, games[0], 1234)
    messages, rows = llm_rank.messages(sched, 100, 3500)
    assert rows[0]["tokens_total"] == 1234 and "busy_s_total" not in rows[0]
    assert "completion tokens its first level took" in messages[0]["content"]
    assert messages[1]["content"].splitlines()[2].startswith("id | state | levels | t1_tokens | tokens_total")
    busy_now, busy_wall, busy = games_on("busy", count=3)
    timed = Scheduler(busy, busy_wall, budget_policy="fixed-2x", backlog_policy="llm")
    start(busy, busy_wall, timed)
    messages, rows = llm_rank.messages(timed, 100, 3500)
    assert messages[0]["content"] == llm_rank.SYSTEM and "busy_s_total" in rows[0]


def test_new_builds_take_the_token_clock_and_the_time_clock_reproduces_old_bundles(tmp_path):
    b = builder()
    cases = {  # variant -> (time clock it was run on, text in the token rule)
        "reclaim-fixed-2x": ("busy", "(floor 1,920 tokens)"),
        "fixed-2x-depthwise-mtp3": ("busy", "the full level allowance again"),
        "fixed-2x-mtp0-wide-kv8-hero-21x42k": ("busy", "(floor 1,920 tokens)"),
        "fixed-2x-index-lane": ("busy", "T1 and the opening-switch keep order in completion tokens"),
        "fixed-2x-progress": ("busy", "5,760 tokens"),
    }
    for variant, (clock, text) in cases.items():
        _, tokens = b.build(variant, output=tmp_path / variant, paired_30m=True, arena_variant=variant)
        _, timed = b.build(variant, output=tmp_path / (variant + "-t"), paired_30m=True, arena_variant=variant,
                           budget_clock="time")
        assert tokens["budget_clock"] == "tokens" and timed["budget_clock"] == clock, variant
        assert text in tokens["level_budget_rule"], tokens["level_budget_rule"]
        assert tokens["token_budget"]["rule"] == tokens["level_budget_rule"] and "token_budget" not in timed
        assert "seconds" not in tokens["level_budget_rule"].split(";")[0] or variant == "fixed-2x-depthwise-mtp3"
        same = {"budget_clock", "token_budget", "level_budget_rule", "backlog_priority", "budget_clock_note"}
        assert {k: v for k, v in tokens.items() if k not in same} == {k: v for k, v in timed.items() if k not in same}
        # The bundled study (configs/comparison.json, the run's study.json) records the clock and the rule.
        bundled = json.loads((tmp_path / variant / "configs" / "comparison.json").read_text())
        assert bundled["budget_clock"] == "tokens" and bundled["token_budget"] == tokens["token_budget"]
        assert json.loads((tmp_path / (variant + "-t") / "configs" / "comparison.json").read_text()) == timed
    # Kept on their time clock, with the reason recorded: the normalized clock (its own factor) or a seconds knob.
    for variant, kwargs, clock, reason in (
            ("fixed-2x-pool", {}, "normalized", "normalized clock"),
            ("fixed-2x-t1cap-120", {}, "busy", "t1_cap_seconds"),
            ("reclaim-grant-180s", {}, "inference", "max_level_grant_seconds")):
        _, study = b.build(variant, output=tmp_path / (variant + "-k"), paired_30m=True, arena_variant=variant,
                           **kwargs)
        assert study["budget_clock"] == clock and "token_budget" not in study, variant
        assert reason in study["budget_clock_note"], study["budget_clock_note"]
    with pytest.raises(ValueError, match="budget_clock"):
        b.build("x", output=tmp_path / "x", paired_30m=True, arena_variant="reclaim-fixed-2x", budget_clock="busy")


def test_trial_refuses_the_token_clock_without_the_duck_agents_and_without_its_settings():
    source = (ROOT / "src/arena/trial.py").read_text()
    assert "The token budget clock is wired for the Duck agents only" in source
    assert "The token budget clock and token_budget go together" in source
    # Old study.json files carry budget_clock busy / inference and no token_budget: the clock is built as before.
    wall = GlobalClock(10)
    clock = ActiveClock(wall, budget_clock="busy", normalization=None, token_purposes=None)
    assert clock.token_purposes is None and clock.budget_clock == "busy"


def test_the_runtime_status_names_the_budget_unit():
    import time
    from types import SimpleNamespace

    from arena.duck import ArenaDuckMixin

    def status(kind):
        now, wall, (game,) = games_on(kind, count=1)
        wall.release()
        game.clock.release()
        game.clock.resume()
        game.clock.grant(2000 if kind == "tokens" else 200)
        reply(now, game, 500, seconds=20)
        agent = SimpleNamespace(game=game, steps=3, obs=None, last_observation_at=time.monotonic(), analysis_turns=0,
                                model_seconds=20.0, turns=1, config=SimpleNamespace(action_target=40))
        return json.loads(ArenaDuckMixin.runtime_message(agent, record=False)["content"].split("\n", 1)[1])

    tokens, busy = status("tokens"), status("busy")
    assert tokens["level_budget_clock"] == "tokens" and tokens["level_budget_remaining_tokens"] == 1500
    assert "level_budget_remaining_seconds" not in tokens
    assert busy["level_budget_remaining_seconds"] == 180.0 and "level_budget_remaining_tokens" not in busy
    assert list(busy) == ["elapsed_seconds", "remaining_seconds", "game_inference_seconds", "level_budget_clock",
                          "level_budget_remaining_seconds", "actions_executed", "levels_completed",
                          "seconds_since_last_observation", "analysis_replies_since_action",
                          "mean_model_request_seconds", "target_actions_per_900_seconds"]


def test_rank_decisions_do_not_depend_on_the_budget_clock():
    """The rank policy has no opening and no level budgets and ranks on every request's tokens: the same trace on the
    busy and the token clock admits, swaps and parks identically; only the recorded budget fields differ."""

    def run(kind):
        now, wall, games = games_on(kind, count=6)
        sched = Scheduler(games, wall, budget_policy="rank", backlog_policy="readmissions", rank={"slots": 2})
        start(games, wall, sched)
        for step in range(12):
            for g in sched.active():
                reply(now, g, 3000 + 700 * g.index, seconds=5 + g.index)
                if step == 3 and g.index == 0:
                    g.report_level(g.levels + 1, wall.elapsed())
            now.advance(61)
            sched.tick()
        report = token_report(now, wall, games, sched)
        audit = audit_scheduler(report)
        assert audit["valid"], audit["issues"]
        return [(e["game"], e["event"], e["reason"], e["levels"], e["budget"]) for e in sched.events], sched

    busy, _ = run("busy")
    tokens, sched = run("tokens")
    assert busy == tokens and any(e[1] == "park" for e in tokens)
    assert sched.summary()["token_budget"]["floor_tokens"] == FLOOR


def test_hero_v2_builds_on_the_token_clock_with_the_shared_token_budget_block(tmp_path):
    from arena import hero_v2
    from arena.scheduler import token_budget_settings

    b = builder()
    args = dict(paired_30m=True, arena_variant=hero_v2.BASE, opening_burst_requests=3,
                exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], clone_instances=110, wall_seconds=7200)
    # token_budgets on (the default component; the phase-6 kernels are not merged in this tree).
    _, study = b.build("hv2", output=tmp_path / "on", hero_v2={"kernels_phase6": False}, **args)
    assert study["budget_clock"] == "tokens" and study["hero_v2"]["components"]["token_budgets"]["on"]
    assert study["token_budget"] == {**token_budget_settings(), "rule": study["level_budget_rule"]}
    assert "(floor 1,920 tokens)" in study["level_budget_rule"]
    assert study["arena_variant"]["reference_difference"]["token_budget"]["after"] == token_budget_settings()
    assert study["token_budget"]["purposes"] == ["gameplay"] and study["token_budget"]["floor_tokens"] == FLOOR
    # token_budgets asked for with --budget-clock time: refused, not silently reverted.
    with pytest.raises(ValueError, match="asks for token budgets"):
        b.build("hv2t", output=tmp_path / "t", hero_v2={"kernels_phase6": False}, budget_clock="time", **args)
    # token_budgets off: the hero's busy clock, with the reason recorded.
    _, off = b.build("hv2off", output=tmp_path / "off", hero_v2={"kernels_phase6": False, "token_budgets": False},
                     **args)
    assert off["budget_clock"] == "busy" and "token_budget" not in off
    assert "token_budgets component off" in off["budget_clock_note"]


def test_hero_v2_rank_builds_on_the_token_clock_and_rebuilds_on_busy(tmp_path):
    b = builder()
    _, tokens = b.build("rank", output=tmp_path / "r", paired_30m=True, arena_variant="hero-v2-rank")
    _, timed = b.build("rank-t", output=tmp_path / "rt", paired_30m=True, arena_variant="hero-v2-rank",
                       budget_clock="time")
    assert tokens["level_budget_policy"] == "rank" and tokens["budget_clock"] == "tokens"
    assert "does not enter the ranking" in tokens["level_budget_rule"] and timed["budget_clock"] == "busy"
    same = {"budget_clock", "token_budget", "level_budget_rule", "name"}
    assert {k: v for k, v in tokens.items() if k not in same} == {k: v for k, v in timed.items() if k not in same}
