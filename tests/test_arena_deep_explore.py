"""fixed-2x-deep-explore: deep slot for the first clear, exploration rotation on a token quantum, the promotion
queue, stall-budget demotion only with a candidate, per-game windows (compaction threshold), the mixed-window
capacity check, the audit, the variant and the build."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_arena_core import Now
from test_arena_hazard_index import charged
from test_arena_mtp0_wide import served_layout
from test_arena_trial import STUDY, FakeArcade, FakeWorker, OfflineMonitor

from arena import trial as trial_module
from arena.cache import BlockBudget
from arena.capacity import deep_explore_fit
from arena.clock import ActiveClock, GlobalClock
from arena.compaction import bind_compaction
from arena.deep_explore import DEFAULTS, settings
from arena.duck import ArenaDuckMixin
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler
from arena.trial import Trial
from arena.variants import VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
DEEP, EXPLORE = DEFAULTS["deep_window"], DEFAULTS["explore_window"]

@pytest.fixture(autouse=True)
def _fast_scheduler_ticks(monkeypatch):
    """Real-wall trials: a 0.1 s tick keeps the audit's 1.1 s switch bar clear of CI CPU jitter (25 Oct flakes: 1.157 s
    and 1.185 s on a 1 s tick)."""
    from arena import trial
    monkeypatch.setattr(trial, "TICK_SECONDS", 0.1)



def build(count=12, **deep_explore):
    now = Now()
    wall = GlobalClock(3600, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="busy"), total_levels=7)
             for i in range(count)]
    for g in games:
        g.clock.pause()  # as the trial does before the release
    sched = Scheduler(games, wall, budget_policy="deep-explore", backlog_policy="readmissions",
                      deep_explore=deep_explore or None)
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


def reasons(game):
    return [(e["event"], e["reason"]) for e in game.admissions]


def test_no_opening_eight_exploration_slots_from_the_release_at_32k():
    now, wall, games, sched = build()
    assert sched.phase == "steady" and sched.switch_wall_seconds == 0.0 and sched.active_limit == 9
    assert [g.game_id for g in sched.explore_games()] == [f"g{i}" for i in range(8)] and not sched.deep_games()
    assert all(reasons(g) == [("admit", "explore_start")] for g in games[:8])
    assert all(g.state == "backlog" and g.clock.parked and not g.admissions for g in games[8:])
    assert all(g.window == EXPLORE for g in games)
    with pytest.raises(ValueError, match="deep-explore"):
        Scheduler(games, wall, budget_policy="deep-explore", backlog_policy="spent")
    with pytest.raises(ValueError, match="deep_explore settings"):
        Scheduler(games, wall, budget_policy="commit", backlog_policy="readmissions", deep_explore={})


def test_the_first_clear_takes_the_deep_slot_at_once_and_its_window_grows():
    now, wall, games, sched = build()
    g3 = games[3]
    windows = []
    g3.window_listener = windows.append
    charged(now, g3, 10, 5000)
    g3.report_level(1, wall.elapsed())
    assert g3.state == "active" and g3.slot == "deep" and g3.window == DEEP and windows == [DEEP]
    assert reasons(g3)[-2:] == [("progress", "explore_clear"), ("grant", "deep_promote")]
    assert sched.deep_tenures[0]["source"] == "first_clear" and sched.deep_decisions[-1]["decision"] == "deep_promote"
    sched.tick()  # the freed exploration slot goes to the next never-played game
    assert len(sched.explore_games()) == 8 and games[8].slot == "explore" and reasons(games[8]) == [
        ("admit", "explore_rotate")]
    assert sched.deep_clears == {"deep": 0, "explore": 1, "none": 0}


def test_exploration_rotates_on_the_token_quantum_never_played_then_fewest_tokens_then_exhausted():
    now, wall, games, sched = build(count=5, explore_slots=2)
    g0, g1, g2, g3, g4 = games
    charged(now, g0, 10, 17_999)
    sched.tick()
    assert g0.slot == "explore"  # under the quantum
    charged(now, g0, 1, 1)
    sched.tick()
    assert g0.state == "parked" and g0.slot is None and reasons(g0)[-1] == ("park", "explore_rotate")
    assert g2.slot == "explore"  # never played first, roster order
    for g in (g1, g2):
        charged(now, g, 10, 70_000 if g is g1 else 18_000)
    sched.tick()
    assert (g1.state, g2.state) == ("parked", "parked") and {g3.slot, g4.slot} == {"explore"}
    # Waiting: g0 (18K), g2 (18K), g1 (70K on level 1 without a clear: last). Ties by roster order.
    assert [g.game_id for g in sched.explore_waiting()] == ["g0", "g2", "g1"]
    g2.clock.charge(0, 1000)  # g2 now has more tokens than g0 but stays ahead of the exhausted g1
    assert [g.game_id for g in sched.explore_waiting()] == ["g0", "g2", "g1"]
    g0.clock.charge(0, 2000)
    assert [g.game_id for g in sched.explore_waiting()] == ["g2", "g0", "g1"]
    # With nobody waiting a quantum is renewed instead of idling the slot.
    for g in (g0, g1, g2):
        g.note_finished("won")
    sched.tick()
    charged(now, g3, 10, 18_000)
    sched.tick()
    assert g3.state == "active" and reasons(g3)[-1] == ("grant", "explore_renew")
    assert sched.deep_decisions[-1]["decision"] == "explore_renew"


def test_an_exploration_clear_keeps_its_slot_to_the_quantum_end_then_queues_and_is_not_rotated():
    now, wall, games, sched = build()
    g1, g3 = games[1], games[3]
    charged(now, g3, 10, 5000)
    g3.report_level(1, wall.elapsed())  # g3 holds the deep slot
    charged(now, g1, 10, 5000)
    g1.report_level(1, wall.elapsed())
    sched.tick()
    assert g1.state == "active" and g1.slot == "explore" and not g1.deep_queued
    assert sched.deep_decisions[-1]["decision"] == "explore_clear"
    charged(now, g1, 10, 13_000)
    sched.tick()
    assert g1.state == "parked" and g1.deep_queued and reasons(g1)[-1] == ("park", "explore_queue")
    assert sched.promotion_queue() == [g1] and g1 not in sched.explore_waiting()
    assert len(sched.explore_games()) == 8 and g3.slot == "deep"  # g3 is within its budget: no demotion


def test_promotion_queue_most_recent_clear_first_then_fewer_tokens():
    now, wall, games, sched = build()
    a, b, c = games[8:11]
    for g, clear, tokens in ((a, 100.0, [9000]), (b, 200.0, [4000]), (c, 200.0, [3000, 1000])):
        g.last_clear_wall, g.level_tokens, g.deep_queued = clear, tokens, True
        sched.deep_queue.append(g)
    assert [g.game_id for g in sched.promotion_queue()] == ["g9", "g10", "g8"]  # c ties b at 200 s: 4,000 each, roster
    c.level_tokens = [3000, 500]
    assert [g.game_id for g in sched.promotion_queue()] == ["g10", "g9", "g8"]
    b.level_tokens = [3400]
    assert [g.game_id for g in sched.promotion_queue()] == ["g9", "g10", "g8"]
    c.note_finished("won")  # a finished game leaves the queue
    assert [g.game_id for g in sched.promotion_queue()] == ["g9", "g8"] and not c.deep_queued


def test_deep_budget_is_the_larger_of_the_floor_and_the_multiple_of_the_median_cleared_level():
    now, wall, games, sched = build()
    g0 = games[0]
    for level_tokens, budget in (([4000], 25_000), ([20_000], 50_000), ([10_000, 30_000], 50_000),
                                 ([0, 12_000, 14_000], 32_500), ([], 25_000)):
        g0.level_tokens = level_tokens
        assert sched.deep_budget(g0) == budget
    custom = Scheduler(games, wall, budget_policy="deep-explore", backlog_policy="readmissions",
                       deep_explore={"deep_min_tokens": 10_000, "deep_multiple": 3})
    g0.level_tokens = [5000]
    assert custom.deep_budget(g0) == 15_000


def test_demotion_past_the_budget_only_with_a_candidate_and_the_windows_follow_the_slots():
    now, wall, games, sched = build()
    g1, g3 = games[1], games[3]
    seen = {g1.game_id: [], g3.game_id: []}
    g1.window_listener, g3.window_listener = seen["g1"].append, seen["g3"].append
    charged(now, g3, 10, 5000)
    g3.report_level(1, wall.elapsed())
    assert sched.deep_budget(g3) == 25_000
    charged(now, g3, 60, 25_001)
    sched.tick()
    assert g3.slot == "deep" and g3.state == "active"  # past its budget, but nobody to promote: keeps playing
    charged(now, g1, 10, 5000)
    g1.report_level(1, wall.elapsed())
    charged(now, g1, 10, 13_000)
    sched.tick()  # g1's quantum ends -> queued -> g3 demoted and g1 promoted in the same tick
    assert g3.state == "parked" and g3.slot is None and g3.window == EXPLORE and reasons(g3)[-1] == ("park", "deep_demote")
    assert g1.state == "active" and g1.slot == "deep" and g1.window == DEEP and reasons(g1)[-1] == ("admit", "deep_promote")
    assert seen == {"g3": [DEEP, EXPLORE], "g1": [DEEP]}
    demote = next(d for d in sched.deep_decisions if d["decision"] == "deep_demote")
    assert demote["deep_budget"] == 25_000 and demote["candidate"] == "g1" and demote["tokens_on_level"] == 25_001
    assert sched.deep_tenures[0]["end_reason"] == "deep_demote" and sched.deep_tenures[1]["replaced"] == "g3"
    # The demoted game rejoins the rotation (not the queue); its tokens put it behind never-played games.
    assert g3 in sched.explore_waiting() and not g3.deep_queued
    assert sched.explore_waiting()[0].readmissions == 0
    summary = sched.summary()["deep_explore"]
    assert summary["decision_counts"]["deep_demote"] == 1 and summary["deep"] == ["g1"]
    assert summary["first_clear_wall_seconds"] == g3.time_to_level[0]["wall_seconds"]


def test_a_late_clear_after_the_slot_ended_queues_or_takes_a_free_deep_slot():
    now, wall, games, sched = build(count=4, explore_slots=2)
    g0, g1 = games[:2]
    charged(now, g0, 10, 18_000)
    sched.tick()
    assert g0.state == "parked" and g0.slot is None
    g0.report_level(1, wall.elapsed())  # its last request's actions cleared the level after the rotation
    assert g0.slot == "deep" and g0.state == "active" and reasons(g0)[-1] == ("admit", "deep_promote")
    charged(now, g1, 10, 18_000)
    sched.tick()
    g1.report_level(1, wall.elapsed())
    assert g1.state == "parked" and g1.deep_queued and sched.promotion_queue() == [g1]


def test_slot_usage_attributes_each_request_to_the_slot_it_was_admitted_in():
    now, wall, games, sched = build(count=4, explore_slots=2)
    g0 = games[0]
    g0.note_slot_request()
    charged(now, g0, 10, 1000)
    g0.report_level(1, wall.elapsed())  # -> deep
    g0.note_slot_request()
    charged(now, g0, 5, 2000)
    usage = sched.summary()["deep_explore"]["slot_usage"]
    assert usage["explore"]["completion_tokens"] == 1000 and usage["explore"]["inference_seconds"] == 10
    assert usage["deep"]["completion_tokens"] == 2000 and usage["deep"]["tokens_per_second_per_stream"] == 400.0
    assert g0.slot_usage == {"explore": {"completion_tokens": 1000, "inference_seconds": 10.0, "requests": 1}}


def test_audit_accepts_the_policy_including_games_that_never_reached_a_slot():
    now, wall, games, sched = build()
    g1, g3 = games[1], games[3]
    charged(now, g3, 10, 5000)
    g3.report_level(1, wall.elapsed())
    for g in games[:8]:
        if g is not g3:
            charged(now, g, 5, 18_000 if g is not g1 else 5000)
    g1.report_level(1, wall.elapsed())
    sched.tick()
    charged(now, g1, 10, 13_000)
    charged(now, g3, 60, 30_000)
    sched.tick()
    assert g1.slot == "deep" and g3.state == "parked"
    now.advance(5)
    sched.wall_end()
    audit = audit_scheduler(report(games, sched, wall))
    assert audit["valid"], audit["issues"]
    assert audit["checks"]["deep_explore_admissions"] >= 10
    # The same records under another policy are rejected, and a policy switch after the release is flagged.
    bad = report(games, sched, wall)
    bad["scheduler"]["budget_policy"] = "commit"
    assert not audit_scheduler(bad)["valid"]
    late = report(games, sched, wall)
    late["scheduler"]["switch_wall_seconds"] = 30.0
    assert not audit_scheduler(late)["valid"]


# ----------------------------------------------------------------------------- agent window
class WindowAgent(ArenaDuckMixin):
    def __init__(self, game):
        self.game, self.window, self.cap, self.steps = game, DEEP, 16384, 0
        self.workspace = SimpleNamespace(append=lambda *a, **k: None)
        self._arena_analyzer = None
        self.sizes = {}

    def carried_reserve(self):
        return 0

    def estimate_prompt_tokens(self, messages, tools):
        return sum(self.sizes.get(m["content"], 10) for m in messages)

    def summarize(self, messages):
        return "summary"


def test_the_compaction_threshold_follows_the_games_slot_window():
    now, wall, games, sched = build(count=4, explore_slots=2)
    game = games[0]
    agent = WindowAgent(game)
    analyzer = SimpleNamespace(_request_safety_margin_tokens=512)
    bind_compaction(analyzer, agent)
    analyzer._context_budget_tokens = agent.context_budget_tokens(analyzer)
    agent._arena_analyzer = analyzer
    game.window_listener = agent.apply_context_window
    assert analyzer._context_budget_tokens == EXPLORE - 16384 - 512
    agent.sizes = {"old": 40_000}
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "old"},
                {"role": "assistant", "content": "a"}, {"role": "user", "content": "new"}]
    charged(now, game, 10, 1000)
    game.report_level(1, wall.elapsed())  # deep: 128K
    assert game.window == DEEP and analyzer._context_budget_tokens == DEEP - 16384 - 512
    assert analyzer._trim_messages_for_context(messages) is messages  # 40K fits the deep window
    game.set_window(EXPLORE)  # demoted: the next request compacts to the exploration window
    trimmed = analyzer._trim_messages_for_context(messages)
    assert trimmed[0] == messages[0] and "summary" in trimmed[1]["content"] and trimmed[-1] == messages[-1]
    assert agent.context_window() == EXPLORE
    game.window = None  # no slot window (every other policy): the agent's own window
    assert agent.context_window() == DEEP


# ----------------------------------------------------------------------------- capacity
def test_capacity_one_128k_and_eight_32k_windows_fit_the_mtp0_layout_and_more_is_refused():
    mtp0 = BlockBudget(served_layout(0), 0.9)
    fit = deep_explore_fit(mtp0, settings())
    assert fit["deep"]["cost_per_full_window_request"] == 175 and fit["explore"]["cost_per_full_window_request"] == 52
    assert fit["total_units"] == 175 + 8 * 52 == 591 and fit["capacity"] == 675 and fit["fits"]
    assert fit["deep"]["request_tokens"] == DEEP + 256 and fit["lookahead"] == 0
    with pytest.raises(RuntimeError, match="need 766 of 675"):
        deep_explore_fit(mtp0, settings({"deep_slots": 2}))
    assert deep_explore_fit(mtp0, settings({"explore_slots": 10}), required=False)["total_units"] == 695
    with pytest.raises(RuntimeError):
        deep_explore_fit(mtp0, settings({"explore_slots": 10}))
    mtp3 = BlockBudget(served_layout(3), 0.9)
    with pytest.raises(RuntimeError, match="lookahead 3"):
        deep_explore_fit(mtp3, settings())
    for bad in ({"deep_slots": 0}, {"explore_window": DEEP * 2}, {"deep_multiple": "2"}, {"surprise": 1}):
        with pytest.raises(ValueError):
            settings(bad)


def trial_study(tmp_path, monkeypatch, mtp, **deep_explore):
    games = STUDY["games"][:12]
    cfg = settings(deep_explore)
    study = {**STUDY, "games": games, "wall_seconds": 0.6, "cache_admission": "hybrid_blocks",
             "windows": [cfg["deep_window"]], "active_limit": cfg["deep_slots"] + cfg["explore_slots"],
             "level_budget_policy": "deep-explore", "backlog_policy": "readmissions", "deep_explore": cfg}
    budget = BlockBudget(served_layout(mtp), 0.9)
    monkeypatch.setattr(trial_module.BlockBudget, "load", lambda *args: budget)
    monkeypatch.setattr(trial_module, "make_arcade", lambda *a: FakeArcade(games))
    monkeypatch.setattr(trial_module, "GpuMonitor", OfflineMonitor)
    return study, budget


async def test_trial_checks_the_mixed_windows_and_runs_the_policy(tmp_path, monkeypatch):
    study, budget = trial_study(tmp_path, monkeypatch, 0)

    async def agents(self, records, gate, window, wall, card, arc):
        assert window == DEEP and gate.capacity == 675
        assert all(r.window == EXPLORE for r in records)
        workers = [FakeWorker(record, clears_at=0.1 if record.index == 2 else None) for record in records]
        self.config_dict = {}

        async def close():
            pass

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t",
                  server_dir=tmp_path, base_url="http://localhost/v1", log=lambda *a, **k: None)
    result = await trial.run()
    fit = result["window_choice"]["deep_explore_fit"]
    assert fit["total_units"] == 591 and fit["required"] and "focus_fit" not in result["window_choice"]
    deep = result["scheduler"]["deep_explore"]
    assert deep["deep"] == [study["games"][2]] and deep["tenures"][0]["source"] == "first_clear"
    assert result["scheduler"]["active_limit"] == 9 and deep["games_played"] == 9
    assert result["scheduler_audit"]["valid"], result["scheduler_audit"]["issues"]


async def test_trial_refuses_windows_that_do_not_fit_before_gameplay(tmp_path, monkeypatch):
    study, _ = trial_study(tmp_path, monkeypatch, 0, deep_slots=2)

    async def agents(*args):
        raise AssertionError("no gameplay before the capacity check")

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t",
                  server_dir=tmp_path, base_url="http://localhost/v1", log=lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="need 766 of 675"):
        await trial.run()


# ----------------------------------------------------------------------------- variant and build
def test_variant_is_the_kernels_serving_with_the_deep_explore_scheduler():
    from arc_infra.dense_kernels_patch import PHASE2

    assert "fixed-2x-deep-explore" in VARIANTS
    result = apply_variant(CONTROL, "fixed-2x-deep-explore")
    kernels = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kernels")
    assert result["server"] == kernels["server"] == {**CONTROL["server"], "mtp_tokens": 0, "dense_kernels": PHASE2}
    assert result["level_budget_policy"] == "deep-explore" and result["deep_explore"] == {**DEFAULTS, "explore_slots": 5}
    assert result["active_limit"] == 6 and result["windows"] == [DEEP] and not result.get("require_focus_windows_fit")
    assert result["cache_admission"] == "hybrid_blocks" and result["cache_experiment"]["compaction_changed"] is True
    assert result["backlog_policy"] == "readmissions" and result["budget_clock"] == "busy"
    meta = result["arena_variant"]
    assert meta["reference_variant"] == "reclaim-fixed-2x" and not meta["checkpoint_and_serving_unchanged"]
    assert set(meta["reference_difference"]) == {"server.mtp_tokens", "server.dense_kernels", "level_budget_policy",
                                                 "deep_explore", "active_limit", "cache_experiment.compaction_changed"}
    with pytest.raises(ValueError, match="deep-explore arm"):
        apply_variant({**CONTROL, "server": {**CONTROL["server"], "mtp_tokens": 1}}, "fixed-2x-deep-explore")


def load_builder():
    spec = importlib.util.spec_from_file_location("deep_builder", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def test_c110_build_satisfies_the_duck_110_modal_checks(tmp_path):
    builder = load_builder()
    output, study = builder.build("c110-deep-explore", output=tmp_path / "d", arena_variant="fixed-2x-deep-explore",
                                  paired_30m=True, opening_burst_requests=3,
                                  exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], wall_seconds=3600,
                                  clone_instances=110)
    assert json.loads((output / "configs/comparison.json").read_text()) == study
    assert study["level_budget_policy"] == "deep-explore" and study["deep_explore"] == {**DEFAULTS, "explore_slots": 5}
    assert "exploration slots" in study["level_budget_rule"] and "arena.deep_explore" in study["backlog_priority"]
    assert (output / "src/arena/deep_explore.py").is_file()
    # scripts/modal_validation.py, TEST == 'duck-110'
    clones = study["competition"]
    assert clones["research_clones"] and clones["preview_instances"] == 110 and clones["preview_seconds"] == 3600
    assert len(set(study["games"])) + len(study.get("games_excluded", [])) == 25 and study["wall_seconds"] == 3600
    assert study["trials"][0] == {"name": "duck-transfer", "kind": "duck-transfer"}
