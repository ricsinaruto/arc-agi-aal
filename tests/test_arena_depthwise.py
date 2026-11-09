"""fixed-2x-depthwise: six slots from the release, a level-0 rotation at 32K on a token quantum, a level-1 clear keeps
its slot at 64K under fixed-2x budgets, parks only while someone waits, the re-admission grant, the refill order
(parked clears, fastest T1, then the rotation), the audit, the capacity check, the variant and the c110 build."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_arena_core import Now
from test_arena_hazard_index import charged
from test_arena_mtp0_wide import served_layout
from test_arena_trial import STUDY, FakeArcade, FakeWorker, OfflineMonitor

from arena import trial as trial_module
from arena.cache import BlockBudget
from arena.clock import ActiveClock, GlobalClock
from arena.depthwise import DEFAULTS, settings
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler
from arena.timing_analysis import admission_timing
from arena.trial import Trial
from arena.variants import DEPTHWISE_SERVING, VARIANTS, _depthwise_variant, apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
DEPTH, LEVEL0 = DEFAULTS["deep_window"], DEFAULTS["level0_window"]
QUANTUM = DEFAULTS["level0_quantum_tokens"]


def build(count=10, **depthwise):
    now = Now()
    wall = GlobalClock(3600, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="busy"), total_levels=7)
             for i in range(count)]
    for g in games:
        g.clock.pause()  # as the trial does before the release
    sched = Scheduler(games, wall, budget_policy="depthwise", backlog_policy="readmissions",
                      depthwise=depthwise or None)
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


def slots(sched):
    return {g.game_id: g.slot for g in sched.active()}


def clear(now, wall, game, seconds=10, tokens=5000):
    """One request, then a level clear: T1 = the game's busy seconds so far."""
    charged(now, game, seconds, tokens)
    game.report_level(game.levels + 1, wall.elapsed())


def test_no_opening_six_level0_slots_from_the_release_at_32k():
    now, wall, games, sched = build()
    assert sched.phase == "steady" and sched.switch_wall_seconds == 0.0 and sched.active_limit == 6
    assert slots(sched) == {f"g{i}": "level0" for i in range(6)}
    assert all(reasons(g) == [("admit", "level0_start")] for g in games[:6])
    assert all(g.state == "backlog" and g.clock.parked and not g.admissions for g in games[6:])
    assert all(g.window == LEVEL0 for g in games) and all(g.clock.budget is None for g in games[:6])
    with pytest.raises(ValueError, match="depthwise policy"):
        Scheduler(games, wall, budget_policy="depthwise", backlog_policy="spent")
    with pytest.raises(ValueError, match="depthwise settings"):
        Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions", depthwise={})
    for bad in ({"slots": 0}, {"level0_window": DEPTH * 2}, {"regrant": "half"}, {"surprise": 1},
                {"level0_quantum_tokens": 1.5}):
        with pytest.raises(ValueError):
            settings(bad)


def test_level0_turns_rotate_on_the_token_quantum_never_played_then_fewest_tokens():
    now, wall, games, sched = build(count=5, slots=2)
    g0, g1, g2, g3, g4 = games
    charged(now, g0, 10, QUANTUM - 1)
    sched.tick()
    assert g0.slot == "level0"  # under the quantum
    charged(now, g0, 1, 1)
    sched.tick()
    assert g0.state == "parked" and g0.slot is None and reasons(g0)[-1] == ("park", "level0_rotate")
    assert g2.slot == "level0" and reasons(g2) == [("admit", "level0_rotate")]  # never played first, roster order
    charged(now, g1, 10, 30_000)
    charged(now, g2, 10, QUANTUM)
    sched.tick()
    assert (g1.state, g2.state) == ("parked", "parked") and {g3.slot, g4.slot} == {"level0"}
    # Waiting now: every level-0 game has played once; fewest completion tokens first (g0 = g2 at 18K: roster).
    assert [g.game_id for g in sched.dw_waiting()] == ["g0", "g2", "g1"]
    g0.clock.charge(0, 1)
    assert [g.game_id for g in sched.dw_waiting()] == ["g2", "g0", "g1"]
    decision = next(d for d in sched.dw_decisions if d["decision"] == "level0_rotate")
    assert decision["game"] == "g0" and decision["turn_tokens"] == QUANTUM and decision["next"] == "g2"
    # With nobody waiting the turn is renewed instead of idling the slot.
    for g in (g0, g1, g2):
        g.note_finished("won")
    sched.tick()
    charged(now, g3, 10, QUANTUM)
    sched.tick()
    assert g3.state == "active" and reasons(g3)[-1] == ("grant", "level0_renew")
    assert sched.dw_decisions[-1]["decision"] == "level0_renew"


def test_a_level1_clear_keeps_the_slot_switches_to_64k_and_starts_the_fixed_2x_budget_from_t1():
    now, wall, games, sched = build()
    g3 = games[3]
    windows = []
    g3.window_listener = windows.append
    clear(now, wall, g3, seconds=10)
    assert g3.state == "active" and g3.slot == "depth" and g3.window == DEPTH and windows == [DEPTH]
    assert g3.t1 == 10 and g3.clock.budget == 2 * 2 * 10  # level 1 -> 2: 2 x (1 + 1) x T1
    assert reasons(g3)[-1] == ("grant", "level0_clear") and g3.admissions[-1]["budget"] == 40
    assert sched.dw_tenures[0]["source"] == "level0_clear" and sched.dw_tenures[0]["budget"] == 40
    assert sched.dw_clears == {"depth": 0, "level0": 1, "none": 0}
    # The quantum no longer applies to it.
    charged(now, g3, 5, 3 * QUANTUM)
    sched.tick()
    assert g3.slot == "depth" and g3.state == "active"
    # A further clear in the depth slot: the full allowance for the new level.
    g3.report_level(2, wall.elapsed())
    assert reasons(g3)[-1] == ("grant", "depth_level") and g3.clock.budget == 2 * 3 * 10
    assert sched.dw_clears["depth"] == 1


def test_an_exhausted_depth_game_parks_only_while_someone_waits_and_never_retakes_its_own_slot():
    now, wall, games, sched = build(count=6)  # every game holds a slot: nobody waits
    g0 = games[0]
    clear(now, wall, g0, seconds=10)
    charged(now, g0, 40, 1000)  # budget 40 s spent
    assert g0.clock.budget_exhausted()
    sched.tick()
    assert g0.state == "active" and g0.slot == "depth"
    assert [d["decision"] for d in sched.dw_decisions].count("depth_keep") == 1
    sched.tick()  # logged once per grant
    assert [d["decision"] for d in sched.dw_decisions].count("depth_keep") == 1

    now, wall, games, sched = build(count=7)  # g6 waits in the level-0 rotation
    g0, g6 = games[0], games[6]
    clear(now, wall, g0, seconds=10)
    charged(now, g0, 39, 1000)
    sched.tick()
    assert g0.slot == "depth"  # 39 of 40 s
    charged(now, g0, 1, 1000)
    sched.tick()
    assert g0.state == "parked" and g0.slot is None and reasons(g0)[-1] == ("park", "depth_park")
    assert g6.slot == "level0" and reasons(g6) == [("admit", "level0_rotate")]  # not g0, though it outranks g6
    park = next(d for d in sched.dw_decisions if d["decision"] == "depth_park")
    assert park["budget"] == 40 and park["spent_since_grant"] == 40 and park["next"] == "g6"
    assert sched.dw_tenures[0]["end_reason"] == "depth_park"
    assert g0.window == DEPTH  # a parked depth game keeps its window (no compaction on return)


def test_readmission_grant_is_the_rest_of_the_allowance_else_one_and_a_half_times_spent():
    now, wall, games, sched = build(count=7)
    g0, g1 = games[:2]
    clear(now, wall, g0, seconds=10)
    charged(now, g0, 40, 1000)
    sched.tick()
    assert g0.state == "parked"
    assert g0.spent_on_level() == 40 and sched.dw_grant(g0) == max(1.5 * 40, 40 + 30) == 70
    charged(now, g1, 10, QUANTUM)  # g1's turn ends; g0 (a depth game) takes the slot before the rotation
    sched.tick()
    assert g0.slot == "depth" and reasons(g0)[-1] == ("admit", "depth_readmit")
    assert g0.clock.budget == 70 and g1.state == "parked"
    readmit = next(d for d in sched.dw_decisions if d["decision"] == "depth_readmit")
    assert readmit["budget"] == 70 and readmit["spent_on_level"] == 40 and readmit["t1_seconds"] == 10
    assert [t["source"] for t in sched.dw_tenures] == ["level0_clear", "depth_readmit"]
    # A short level: the floor (spent + 30) is larger than 1.5 x spent below 60 s.
    g0.level_started_budget = g0.clock.budget_seconds() - 50
    assert sched.dw_grant(g0) == 80
    # ``regrant="full"`` re-grants the full fixed-2x allowance (the fixed-2x code's re-admission).
    full = build(count=7, regrant="full")[3]
    g0.level_started_budget = g0.clock.budget_seconds() - 400
    assert full.dw_grant(g0) == 2 * 2 * 10


def test_a_freed_slot_goes_to_a_parked_depth_game_before_the_level0_rotation():
    now, wall, games, sched = build(count=9)
    fast, slow = games[0], games[1]
    clear(now, wall, slow, seconds=30)  # T1 30
    clear(now, wall, fast, seconds=10)  # T1 10
    charged(now, slow, 120, 1000)  # 2 x 2 x 30 s spent
    sched.tick()  # g6..g8 wait: slow parks and the never-played g6 takes its slot
    assert slow.state == "parked" and games[6].slot == "level0"
    charged(now, fast, 40, 1000)
    sched.tick()  # fast parks; its slot goes to the parked depth game, not to the waiting level-0 games g7, g8
    assert fast.state == "parked" and slow.slot == "depth" and reasons(slow)[-1] == ("admit", "depth_readmit")
    assert slow.clock.budget == max(1.5 * 120, 120 + 30) and games[7].state == "backlog"
    charged(now, games[2], 5, QUANTUM)
    sched.tick()  # a level-0 turn ends: the parked depth game again comes first
    assert fast.slot == "depth" and reasons(fast)[-1] == ("admit", "depth_readmit")
    charged(now, games[3], 5, QUANTUM)
    sched.tick()
    assert games[7].slot == "level0"  # no depth game waits: the next never-played level-0 game


def test_fixed_2x_backlog_puts_a_parked_depth_game_behind_every_game_admitted_fewer_times():
    now, wall, games, sched = build(count=9, backlog="fixed-2x")
    fast, slow = games[0], games[1]
    clear(now, wall, slow, seconds=30)  # T1 30
    clear(now, wall, fast, seconds=10)  # T1 10
    charged(now, slow, 120, 1000)  # 2 x 2 x 30 s spent
    sched.tick()  # slow parks; the never-played g6 takes its slot
    assert slow.state == "parked" and games[6].slot == "level0"
    charged(now, fast, 40, 1000)
    sched.tick()  # fast parks: the never-played g7 comes before the parked depth game (depth-first readmits slow)
    assert fast.state == "parked" and games[7].slot == "level0" and slow.state == "parked"
    assert [g.game_id for g in sched.dw_waiting()] == ["g8", "g0", "g1"]  # one admission and one level each: roster
    charged(now, games[2], 5, QUANTUM)
    sched.tick()
    assert games[8].slot == "level0" and slow.state == "parked"
    charged(now, games[3], 5, QUANTUM)
    sched.tick()  # nobody waits with fewer admissions: the parked depth games return, roster order first
    assert fast.slot == "depth" and reasons(fast)[-1] == ("admit", "depth_readmit") and slow.state == "parked"
    # A clear that landed while parked still goes first.
    games[2].pending_parked_progress = True
    assert sched.dw_waiting()[0] is games[2]
    with pytest.raises(ValueError, match="backlog"):
        settings({"backlog": "gittins"})


def test_refill_order_parked_clears_then_fastest_t1_then_never_played_then_fewest_tokens():
    now, wall, games, sched = build(count=10)
    g0, g1, g2 = games[:3]
    charged(now, g0, 30, QUANTUM)
    charged(now, g1, 10, QUANTUM + 500)
    sched.tick()  # both turns end: g6 and g7 play
    g0.report_level(1, wall.elapsed())  # late clears (their last requests' actions): T1 30 and 10
    g1.report_level(1, wall.elapsed())
    assert g0.t1 == 30 and g1.t1 == 10 and g0.pending_parked_progress and g1.pending_parked_progress
    assert [g.game_id for g in sched.dw_waiting()] == ["g1", "g0", "g8", "g9"]
    # Without a pending parked clear, g1 still leads the depth games (smallest T1) but follows the parked clear g0.
    g1.pending_parked_progress = False
    assert [g.game_id for g in sched.dw_waiting()] == ["g0", "g1", "g8", "g9"]
    g1.pending_parked_progress = True
    charged(now, g2, 5, QUANTUM)
    sched.tick()
    assert g1.slot == "depth" and g1.window == DEPTH and reasons(g1)[-1] == ("admit", "depth_parked_clear")
    # A new level: the fresh allowance 2 x 2 x T1 less the seconds spent on it since the clear (none).
    assert g1.clock.budget == 2 * 2 * 10 and not g1.pending_parked_progress
    parked = next(d for d in sched.dw_decisions if d["decision"] == "depth_parked_clear")
    assert parked["budget"] == 40 and parked["spent_on_level"] == 0 and parked["t1_seconds"] == 10
    # Level-0 games that have played go by fewest completion tokens after the never-played ones.
    for game in games[8:]:
        game.note_finished("won")
    assert [g.game_id for g in sched.dw_waiting()] == ["g0", "g2"]
    assert sched._dw_key(g2)[0] == 3 and sched._dw_key(games[3])[0] == 3


def test_a_level0_clear_after_its_turn_ended_is_a_parked_clear_and_returns_as_a_depth_game():
    now, wall, games, sched = build(count=8)
    g0, g1 = games[:2]
    charged(now, g0, 10, QUANTUM)
    sched.tick()
    assert g0.state == "parked" and g0.slot is None
    g0.report_level(1, wall.elapsed())  # its last request's actions cleared the level after the rotation
    assert g0.state == "parked" and g0.pending_parked_progress and sched.dw_clears["none"] == 1
    assert sched.dw_decisions[-1]["decision"] == "parked_clear" and sched.dw_decisions[-1]["was_level0"]
    charged(now, g1, 10, QUANTUM)
    sched.tick()
    assert g0.slot == "depth" and g0.window == DEPTH and reasons(g0)[-1] == ("admit", "depth_parked_clear")
    assert g0.clock.budget == 2 * 2 * 10 - g0.spent_on_level() == 40


def test_finished_games_free_their_slots_and_slot_usage_is_split_depth_and_level0():
    now, wall, games, sched = build(count=8)
    g0, g1 = games[:2]
    g0.note_slot_request()
    charged(now, g0, 10, 1000)
    g0.report_level(1, wall.elapsed())  # -> depth
    g0.note_slot_request()
    charged(now, g0, 5, 2000)
    usage = sched.summary()["depthwise"]["slot_usage"]
    assert usage["level0"]["completion_tokens"] == 1000 and usage["level0"]["tokens_per_second_per_stream"] == 100.0
    assert usage["depth"]["completion_tokens"] == 2000 and usage["depth"]["tokens_per_second_per_stream"] == 400.0
    g1.note_finished("won")
    sched.tick()
    assert g1.state == "finished" and g1.slot is None and games[6].slot == "level0"
    assert any(d["decision"] == "level0_finished" for d in sched.dw_decisions)


def scenario():
    """Clears in turns, a park with the rotation waiting, a keep, a readmission, a parked clear."""
    now, wall, games, sched = build(count=9)
    g0, g1, g2 = games[:3]
    for g in games[:6]:
        g.note_slot_request()
    clear(now, wall, g1, seconds=12)
    clear(now, wall, g0, seconds=10)
    charged(now, g2, 5, QUANTUM)
    sched.tick()  # g2 rotates out; g6 plays
    charged(now, g0, 40, 4000)
    sched.tick()  # g0 exhausted, g2/g7/g8 wait: parks; g7 plays
    charged(now, g1, 20, 3000)
    g1.report_level(2, wall.elapsed())
    charged(now, games[3], 5, QUANTUM)
    sched.tick()  # games[3] rotates out; g0 (a parked depth game) returns before the rotation
    g0.note_slot_request()
    charged(now, g0, 10, 1000)
    g0.report_level(2, wall.elapsed())
    now.advance(5)
    sched.wall_end()
    return now, wall, games, sched


def test_audit_accepts_the_policy_and_rejects_tampered_grants_parks_and_policies():
    now, wall, games, sched = scenario()
    audit = audit_scheduler(report(games, sched, wall))
    assert audit["valid"], audit["issues"]
    assert audit["checks"]["depthwise_admissions"] >= 10 and audit["checks"]["budget_parks"] == 1
    assert audit["checks"]["grants"] >= 5
    summary = sched.summary()["depthwise"]
    assert summary["decision_counts"]["depth_park"] == 1 and summary["decision_counts"]["depth_readmit"] == 1
    assert summary["t1"]["g0"]["t1_seconds"] == 10 and summary["t1"]["g1"]["t1_seconds"] == 12
    assert list(summary["t1"]) == ["g0", "g1"] and summary["first_clear_wall_seconds"] == 12
    assert summary["parks"] == 1 and summary["readmissions"] == 1
    assert summary["per_game"]["g0"] == {"level0_clear": 1, "depth_park": 1, "depth_readmit": 1}
    assert [t["source"] for t in summary["tenures"]] == ["level0_clear", "level0_clear", "depth_readmit"]
    assert summary["tenures"][2]["levels"] == 1 and summary["tenures"][2].get("open")
    assert set(summary["token_share"]) == {"level0", "depth"} and summary["games_played"] == 8
    # g8 never reached a slot (its only event is the wall-end finish); the audit above accepted it.
    assert reasons(games[8]) == [("finish", "wall_end")]
    # Tampering is caught.
    bad = report(games, sched, wall)
    bad["scheduler"]["budget_policy"] = "fixed-2x"
    assert not audit_scheduler(bad)["valid"]
    late = report(games, sched, wall)
    late["scheduler"]["switch_wall_seconds"] = 30.0
    assert not audit_scheduler(late)["valid"]
    for reason in ("level0_clear", "depth_readmit"):
        wrong = report(games, sched, wall)
        row = next(g for g in wrong["games"] if g["game_id"] == "g0")
        event = next(e for e in row["admissions"] if e["reason"] == reason)
        event["budget"] += 5
        assert not audit_scheduler(wrong)["valid"], reason
    early = report(games, sched, wall)
    row = next(g for g in early["games"] if g["game_id"] == "g0")
    next(e for e in row["admissions"] if e["reason"] == "level0_clear")["budget"] = 1000
    assert any("depth_park before exhausting" in issue or "grant" in issue
               for issue in audit_scheduler(early)["issues"])
    full = report(games, sched, wall)
    full["scheduler"]["depthwise"]["settings"]["regrant"] = "full"  # recorded rule != the grants made
    assert not audit_scheduler(full)["valid"]


def test_timing_analysis_skips_games_that_never_reached_a_slot():
    now, wall, games, sched = scenario()
    rows = report(games, sched, wall)
    for row in rows["games"]:
        row["admission_timing"] = {"attempts": []}
    result = admission_timing(rows)  # played games lack attempts here (issues); g8 never played and is skipped
    assert result["issues"] and not any(issue.startswith("g8:") for issue in result["issues"])
    rows["scheduler"]["budget_policy"] = "fixed-2x"
    assert any(issue.startswith("g8:") for issue in admission_timing(rows)["issues"])


# ----------------------------------------------------------------------------- capacity and trial
def trial_study(tmp_path, monkeypatch, mtp, **depthwise):
    games = STUDY["games"][:9]
    cfg = settings(depthwise)
    study = {**STUDY, "games": games, "wall_seconds": 0.6, "cache_admission": "hybrid_blocks",
             "windows": [cfg["deep_window"]], "active_limit": cfg["slots"], "require_focus_windows_fit": True,
             "level_budget_policy": "depthwise", "backlog_policy": "readmissions", "depthwise": cfg}
    budget = BlockBudget(served_layout(mtp), 0.9)
    monkeypatch.setattr(trial_module.BlockBudget, "load", lambda *args: budget)
    monkeypatch.setattr(trial_module, "make_arcade", lambda *a: FakeArcade(games))
    monkeypatch.setattr(trial_module, "GpuMonitor", OfflineMonitor)
    return study, budget


async def test_trial_checks_six_full_64k_windows_and_runs_the_policy(tmp_path, monkeypatch):
    study, budget = trial_study(tmp_path, monkeypatch, 0)

    async def agents(self, records, gate, window, wall, card, arc):
        assert window == DEPTH and gate.capacity == 675
        assert all(r.window == LEVEL0 for r in records)
        workers = [FakeWorker(record, clears_at=0.1 if record.index == 2 else None) for record in records]
        self.config_dict = {}

        async def close():
            pass

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t",
                  server_dir=tmp_path, base_url="http://localhost/v1", log=lambda *a, **k: None)
    result = await trial.run()
    fit = result["window_choice"]["focus_fit"]
    assert fit["cost_per_full_window_request"] == 93 and fit["full_window_requests"] == 7 and fit["required"]
    assert result["window_choice"]["depthwise_level0"]["cost_per_full_window_request"] == 52
    depthwise = result["scheduler"]["depthwise"]
    assert depthwise["depth"] == [study["games"][2]] and depthwise["tenures"][0]["source"] == "level0_clear"
    assert result["scheduler"]["active_limit"] == 6 and result["scheduler"]["budget_policy"] == "depthwise"
    assert result["scheduler_audit"]["valid"], result["scheduler_audit"]["issues"]
    log = [json.loads(line) for line in (tmp_path / "t/admission.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "depthwise" and e["decision"] == "level0_clear" for e in log)


async def test_trial_refuses_a_serving_where_six_64k_windows_do_not_fit(tmp_path, monkeypatch):
    study, _ = trial_study(tmp_path, monkeypatch, 3)  # MTP-3: 199 usable blocks, 3 full 64K windows

    async def agents(*args):
        raise AssertionError("no gameplay before the capacity check")

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t",
                  server_dir=tmp_path, base_url="http://localhost/v1", log=lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="6 focus windows of 65536 tokens"):
        await trial.run()
    study, _ = trial_study(tmp_path, monkeypatch, 0)
    trial = Trial({**study, "require_focus_windows_fit": False}, study["trials"][0], source=".", games_dir="games",
                  output=tmp_path / "u", server_dir=tmp_path, base_url="http://localhost/v1",
                  log=lambda *a, **k: None)
    with pytest.raises(ValueError, match="require_focus_windows_fit"):
        await trial.run()


# ----------------------------------------------------------------------------- variant and build
def test_variant_is_the_kernels_serving_with_the_depthwise_scheduler():
    from arc_infra.dense_kernels_patch import PHASE2

    assert "fixed-2x-depthwise" in VARIANTS and DEPTHWISE_SERVING["fixed-2x-depthwise"] == "fixed-2x-mtp0-wide-kernels"
    result = apply_variant(CONTROL, "fixed-2x-depthwise")
    kernels = apply_variant(CONTROL, "fixed-2x-mtp0-wide-kernels")
    assert result["server"] == kernels["server"] == {**CONTROL["server"], "mtp_tokens": 0, "dense_kernels": PHASE2}
    assert result["level_budget_policy"] == "depthwise" and result["depthwise"] == {**DEFAULTS, "regrant": "full"}
    assert result["active_limit"] == 6 and result["windows"] == [DEPTH] and result["require_focus_windows_fit"] is True
    assert result["cache_admission"] == "hybrid_blocks" and result["cache_experiment"]["compaction_changed"] is True
    assert result["backlog_policy"] == "readmissions" and result["budget_clock"] == "busy"
    assert result["depthwise_serving_source"] == "fixed-2x-mtp0-wide-kernels"
    # Everything but the scheduler is the kernels arm's study.
    same = {k: v for k, v in result.items() if k not in ("level_budget_policy", "depthwise", "arena_variant",
                                                          "depthwise_serving_source")}
    assert same == {k: v for k, v in kernels.items() if k not in ("arena_variant", "level_budget_policy")}
    meta = result["arena_variant"]
    assert meta["reference_variant"] == "reclaim-fixed-2x" and not meta["checkpoint_and_serving_unchanged"]
    assert set(meta["reference_difference"]) == {
        *kernels["arena_variant"]["reference_difference"], "level_budget_policy", "depthwise"}
    # The serving source is swappable: any composed variant serving six 64K fixed-2x slots (MTP-0 without the
    # kernels here); a source that does not serve them is refused.
    plain, changed = _depthwise_variant(CONTROL, "fixed-2x-mtp0-wide")
    assert plain["server"] == {**CONTROL["server"], "mtp_tokens": 0} and "server.dense_kernels" not in changed
    for source in ("fixed-2x-mtp4", "fixed-2x-deep-explore", "fixed-2x-gittins-c110-wide"):
        with pytest.raises(ValueError, match="depthwise arm"):
            _depthwise_variant(CONTROL, source)


def test_fixed_order_variant_differs_from_depthwise_mtp3_only_in_the_refill_order():
    base = apply_variant(CONTROL, "fixed-2x-depthwise-mtp3")
    fixed = apply_variant(CONTROL, "fixed-2x-depthwise-mtp3-fixed-order")
    assert fixed["depthwise"] == {**base["depthwise"], "backlog": "fixed-2x"} and base["depthwise"]["backlog"] == "depth-first"
    def strip(result):
        return {k: v for k, v in result.items() if k not in ("depthwise", "arena_variant")}
    assert strip(fixed) == strip(base)


def load_builder():
    spec = importlib.util.spec_from_file_location("depthwise_builder", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def test_c110_build_satisfies_the_duck_110_modal_checks(tmp_path):
    builder = load_builder()
    output, study = builder.build("c110-depthwise", output=tmp_path / "d", arena_variant="fixed-2x-depthwise",
                                  paired_30m=True, opening_burst_requests=3,
                                  exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], wall_seconds=3600,
                                  clone_instances=110)
    assert json.loads((output / "configs/comparison.json").read_text()) == study
    assert study["level_budget_policy"] == "depthwise" and study["depthwise"] == {**DEFAULTS, "regrant": "full"}
    assert "level-0 games rotate" in study["level_budget_rule"] and "2*(k+1)*T1" in study["level_budget_rule"]
    assert "the full level allowance again" in study["level_budget_rule"]
    assert "arena.depthwise" in study["backlog_priority"]
    assert (output / "src/arena/depthwise.py").is_file()
    assert study["require_focus_windows_fit"] is True and study["windows"] == [DEPTH] and study["active_limit"] == 6
    # scripts/modal_validation.py, TEST == 'duck-110'
    clones = study["competition"]
    assert clones["research_clones"] and clones["preview_instances"] == 110 and clones["preview_seconds"] == 3600
    assert len(set(study["games"])) + len(study.get("games_excluded", [])) == 25 and study["wall_seconds"] == 3600
    assert study["trials"][0] == {"name": "duck-transfer", "kind": "duck-transfer"}
