"""Priority scheduling (``arena.priority``, ``Scheduler(budget_policy="priority")``), dfranzen's priority gate ported at
our stream count: priority_value parity with their original function, first windows in roster order, handovers only at
context evictions, the refreshed queue, the tail fade, finished games freeing slots, no retirement, total-level
clipping, the audit, the hero-v2 component and the real trial loop; other policies unchanged."""

import asyncio
import itertools
import json
from pathlib import Path

import pytest
from test_arena_core import Now

from arena.clock import ActiveClock, GlobalClock
from arena.priority import (
    DEFAULTS,
    UNSTARTED_BASE,
    PrioritySnapshot,
    parse_total_levels,
    priority_value,
    score,
    settings,
    tail_fraction,
)
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler

ROOT = Path(__file__).resolve().parents[1]


def build(count=5, slots=2, wall_seconds=300, total_levels=8, **cfg):
    now = Now()
    wall = GlobalClock(wall_seconds, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="tokens"), total_levels=total_levels)
             for i in range(count)]
    for g in games:
        g.clock.pause()
    log = []
    sched = Scheduler(games, wall, log=log.append, active_limit=slots, budget_policy="priority",
                      backlog_policy="readmissions", priority=cfg or None)
    for g in games:
        g.observer = sched
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    return now, wall, games, sched, log


def at(now, sched, t):
    now.value = float(t)
    sched.tick()


def charge(now, game, tokens, seconds=1.0):
    with game.clock.inference():
        now.advance(seconds)
    game.clock.charge(1000, tokens)


def act(game, count):
    for i in range(count):
        game.note_action(("frame", game.game_id, game.total_actions, i))


def events(game):
    return [(e["event"], e["reason"]) for e in game.admissions]


def active(games):
    return [g.game_id for g in games if g.state == "active"]


def report(games, sched, wall, target=300):
    return {"wall_seconds": wall.elapsed(), "config": {"gameplay_seconds": target},
            "games": [game_row(g, {"game_id": g.game_id}) for g in games], "scheduler": sched.summary()}


# --- the ported scoring --------------------------------------------------------------------------------------------

def test_priority_value_matches_the_original_function_on_a_grid():
    data = json.loads((ROOT / "tests/fixtures/priority_value_parity.json").read_text())
    grid = list(itertools.product(*data["axes"].values()))
    assert len(grid) == 480
    for mode in data["modes"]:
        ours = [priority_value(PrioritySnapshot(level, actions, tokens, total_levels=total), **mode["kwargs"])
                for level, total, actions, tokens in grid]
        assert ours == mode["values"], mode["kwargs"]
    # One worked value: level 1 of 8, fresh: A = 1 x 55 / 36, B = 8, C = 1 -> 952.
    assert priority_value(PrioritySnapshot(1, 0, 0.0, total_levels=8), normalize_score=True,
                          tail_lookup="remaining") == 952
    with pytest.raises(ValueError):
        priority_value(PrioritySnapshot(1, 0, 0.0), tail_lookup=True)  # the 80K table is not ported


def test_total_levels_are_clipped_to_6_through_10():
    assert [parse_total_levels(v) for v in (3, 6, 7.0, "8", 10, 12, 40)] == [6, 6, 7, 8, 10, 10, 10]
    assert [parse_total_levels(v) for v in (None, True, 0, -2, 2.5, "x", float("inf"))] == [None] * 7
    cfg = settings()
    for total, clipped in ((3, 6), (7, 7), (15, 10), (None, 10)):
        game = GameRecord("g", 0, ActiveClock(GlobalClock(100), budget_clock="tokens"), total_levels=total)
        value, details = score(game, cfg, 100, 100)
        assert details["total_levels"] == clipped
        assert value == priority_value(PrioritySnapshot(1, 0, 0.0, total_levels=clipped), normalize_score=True,
                                       tail_lookup="remaining", tail_fraction=5.0)


def test_settings_default_to_their_submission_and_validate():
    cfg = settings()
    assert cfg == {**DEFAULTS} and cfg["tail_fade_fraction"] == 0.2 and cfg["tail_lookup"] == "remaining"
    assert (cfg["human_actions"], cfg["action_scale"], cfg["token_scale"], cfg["action_weight"]) == (25, 115, 62000,
                                                                                                    .25)
    for bad in ({"tail_fade_fraction": 0}, {"tail_fade_fraction": 1.5}, {"tail_lookup": "80k"}, {"pace": True},
                {"refresh_queue": 1}, {"action_weight": 2}, {"token_scale": 0}):
        with pytest.raises(ValueError):
            settings(bad)


def test_tail_fade_scales_the_tail_over_the_last_fifth_of_the_wall():
    cfg = settings()
    assert tail_fraction(cfg, 300, 120) == 2.0  # before the window: clipped to 1 by priority_value
    assert tail_fraction(cfg, 300, 30) == 0.5 and tail_fraction(cfg, 300, 0) == 0.0
    assert tail_fraction(settings({"tail_fade": False}), 300, 0) is None
    now, wall, games, sched, log = build(count=2, slots=1)
    g = games[0]
    early = score(g, sched.priority_cfg, 300, 300)[0]
    half = score(g, sched.priority_cfg, 300, 30)[0]
    end = score(g, sched.priority_cfg, 300, 0)[0]
    assert (early, half, end) == (952, 552, 152)  # A = 152.8, B = 8 -> 4 -> 0
    at(now, sched, 239)
    assert sched.priority_fade_wall is None
    at(now, sched, 240.5)
    at(now, sched, 250)
    fades = [d for d in sched.priority_decisions if d["decision"] == "tail_fade"]
    assert len(fades) == 1 and fades[0]["wall_seconds"] == 240.5 and fades[0]["window_seconds"] == 60.0
    # A gateway deadline shortens the real wall: the window follows it.
    wall.seconds = 100
    assert score(g, sched.priority_cfg, wall.seconds, wall.remaining())[1]["tail_fraction"] == 0.0


# --- the state machine ---------------------------------------------------------------------------------------------

def test_release_admits_the_first_slots_in_roster_order_without_budgets():
    now, wall, games, sched, log = build()
    assert active(games) == ["g0", "g1"] and sched.phase == "steady" and sched.switch_reason == "priority_start"
    assert all(events(g) == [("admit", "first_window")] for g in games[:2])
    assert all(g.admissions[0]["budget"] is None and g.clock.budget is None for g in games[:2])
    assert all(g.state == "backlog" for g in games[2:])
    assert sched.priority_value(games[4])[0] == UNSTARTED_BASE - 4


def test_every_unstarted_game_gets_a_first_window_before_a_requeued_game_returns():
    now, wall, games, sched, log = build()
    order = []
    for t, g in ((10, "g0"), (20, "g1"), (30, "g2")):
        at(now, sched, t)
        games[int(g[1])].note_compaction()
        order.append(active(games))
    assert order == [["g1", "g2"], ["g2", "g3"], ["g3", "g4"]]
    assert events(games[0]) == [("admit", "first_window"), ("park", "handover")]
    assert [e["freed"] for e in games[2].admissions if e["event"] == "admit"] == ["handover"]
    assert events(games[4]) == [("admit", "first_window")]
    # Every game has started: the next eviction is decided by score (g0 is a fresh level 1; g3 has spent its level).
    act(games[3], 300)
    charge(now, games[3], 90_000)
    games[3].note_compaction()
    assert games[3].state == "parked" and games[0].state == "active"
    assert events(games[0])[-1] == ("admit", "handover") and games[0].admissions[-1]["freed"] == "handover"
    assert sched.priority_counts["first_window"] == 5 and sched.priority_counts["handover"] == 1


def test_handover_happens_only_at_an_eviction_and_may_keep_the_slot():
    now, wall, games, sched, log = build(count=3, slots=1)
    g0 = games[0]
    for t in range(1, 60):
        charge(now, g0, 2000)
        act(g0, 3)
        at(now, sched, t)
    g0.report_level(1, wall.elapsed())
    at(now, sched, 61)
    assert active(games) == ["g0"] and events(g0) == [("admit", "first_window")]  # no preemption mid-prefix
    g0.note_context_drain(tokens_before=120_000, tokens_after=60_000)  # the drain-half hook
    assert active(games) == ["g1"] and g0.context_drains == 1 and g0.context_evictions == 1
    handover = [d for d in sched.priority_decisions if d["decision"] == "handover"][0]
    assert handover["eviction"] == "drain" and handover["tokens_before"] == 120_000 and handover["to"] == "g1"
    games[1].note_compaction()
    assert active(games) == ["g2"]
    # Everyone started: g0 is on level 2 with nothing spent on it yet (1,105) against fresh level-1 games (952 each).
    games[2].note_compaction()
    assert active(games) == ["g0"]
    act(games[0], 1)
    games[0].note_compaction()  # g0 (level 2, 1 action) still beats g1 and g2: it keeps its slot
    assert active(games) == ["g0"] and events(games[0])[-1] == ("grant", "handover")
    assert games[0].admissions[-1]["kept"] is True and sched.priority_counts["handover_kept"] == 1
    # An eviction of a parked game changes nothing.
    games[1].note_compaction()
    assert games[1].state == "parked" and sched.priority_counts["eviction_not_active"] == 1


def test_the_refreshed_queue_rescores_waiting_games_at_the_handover():
    def play(**cfg):
        now, wall, games, sched, log = build(count=3, slots=1, **cfg)
        g0, g1, g2 = games
        g0.report_level(7, wall.elapsed())  # the last of 8 levels: no tail, 50 actions spent -> 131 at any time
        act(g0, 50)
        at(now, sched, 10)
        g0.note_compaction()
        g1.report_level(2, wall.elapsed())  # level 3: a long tail, 60 actions -> 803 now, 37 at the end
        act(g1, 60)
        at(now, sched, 20)
        g1.note_compaction()
        act(g2, 400)  # g2 itself: level 1 with 400 actions, worth almost nothing at the end
        at(now, sched, 300)
        assert games[2].state == "active"
        assert score(g1, sched.priority_cfg, 300, 300)[0] == 803 and score(g1, sched.priority_cfg, 300, 0)[0] == 37
        g2.note_compaction()
        return active(games)

    assert play() == ["g0"]  # refreshed at the handover: the tail has faded, g0's last level is worth more
    assert play(refresh_queue=False) == ["g1"]  # queued scores: g1's early 803 still wins


def test_a_finished_game_frees_its_slot_and_parked_games_keep_their_state():
    now, wall, games, sched, log = build(count=4, slots=2)
    games[0].note_compaction()  # g0 -> g2
    games[0].report_level(1, wall.elapsed())  # a parked game's late reply: kept, no grant
    assert games[0].levels == 1 and events(games[0])[-1] == ("park", "handover")
    games[1].finished_reason = "won"
    at(now, sched, 5)
    assert games[1].state == "finished" and events(games[1])[-1] == ("finish", "finished")
    assert games[1].admissions[-1]["finished_reason"] == "won"
    assert events(games[3]) == [("admit", "first_window")] and games[3].admissions[0]["freed"] == "finished"
    games[2].finished_reason = "won"
    at(now, sched, 6)
    assert events(games[0])[-1] == ("admit", "finished") and games[0].levels == 1
    assert active(games) == ["g0", "g3"] and sched.priority_counts["finished"] == 1


def test_no_level_budgets_and_no_retirement():
    now, wall, games, sched, log = build(count=3, slots=1)
    g0 = games[0]
    for level in (1, 2, 3):
        charge(now, g0, 500_000)
        act(g0, 500)
        g0.report_level(level, wall.elapsed())
        at(now, sched, wall.elapsed() + 1)
    assert g0.state == "active" and not g0.retired and g0.clock.budget is None
    assert {e["event"] for e in g0.admissions} == {"admit"} and len(sched.events) == 1
    sched.wall_end()
    assert events(g0)[-1] == ("finish", "wall_end") and events(games[1]) == [("finish", "wall_end")]


def test_audit_accepts_a_priority_run_and_refuses_a_handover_without_an_eviction():
    now, wall, games, sched, log = build(count=4, slots=2)
    charge(now, games[0], 3000)
    games[0].note_compaction()
    at(now, sched, 50)
    charge(now, games[1], 2000)
    games[1].note_compaction()
    games[2].finished_reason = "won"
    at(now, sched, 100)
    charge(now, games[3], 1000)
    games[3].note_compaction()
    at(now, sched, 300)
    sched.wall_end()
    rep = report(games, sched, wall)
    result = audit_scheduler(rep)
    assert result["valid"], result["issues"]
    # g0 -> g2 and g1 -> g3 (first windows), g2's finished slot -> g1 (fewer tokens than g0), g3 keeps its slot.
    assert events(games[1])[-2] == ("admit", "finished")
    assert events(games[3])[-2] == ("grant", "handover")
    assert result["checks"]["priority_admissions"] == 5 and result["checks"]["priority_handovers"] == 3
    rep["scheduler"]["priority"]["games"]["g0"]["context_evictions"] = 0
    assert any("context evictions" in issue for issue in audit_scheduler(rep)["issues"])
    bad = report(games, sched, wall)
    bad["games"][0]["admissions"][0]["reason"] = "handover"  # a first admission must be a first window
    assert not audit_scheduler(bad)["valid"]


def test_other_policies_ignore_the_eviction_hooks():
    now = Now()
    wall = GlobalClock(300, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="tokens")) for i in range(3)]
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions")
    for g in games:
        g.observer = sched
    wall.release()
    sched.start_all()
    before = json.dumps(sched.summary(), sort_keys=True)
    games[0].note_compaction()
    games[1].note_context_drain(tokens_before=10, tokens_after=5)
    assert json.dumps(sched.summary(), sort_keys=True) == before and "priority" not in sched.summary()
    GameRecord("x", 0, ActiveClock(wall)).note_context_drain()  # no observer: a no-op
    with pytest.raises(ValueError):
        Scheduler(games, wall, budget_policy="fixed-2x", priority={})
    with pytest.raises(ValueError):
        Scheduler(games, wall, budget_policy="priority", backlog_policy="readmissions", backlog_key="progress-fruitless")


# --- hero-v2 component and the trial loop --------------------------------------------------------------------------

def test_hero_v2_component_switches_the_comp_build_to_the_priority_gate(monkeypatch):
    from test_hero_v2 import hero, pretend_merged

    from arena.campaign import components
    from arena.hero_v2 import compose, normalize

    pretend_merged(monkeypatch)
    base = components("comp-prune25")
    assert "priority" not in normalize(base)
    control = compose(hero(), base, served_blocks=1755)
    assert "priority" not in control and control["level_budget_policy"] == "fixed-2x"
    assert "priority" not in control["hero_v2"]["components"] and "priority" not in control["hero_v2"]["requested"]
    with pytest.raises(ValueError):
        compose(hero(), {**base, "budget_policy": "priority"}, served_blocks=1755)  # backlog_fix stays fixed-2x's
    with pytest.raises(ValueError):
        normalize({**base, "priority": {"tail_fade_fraction": 0.3}})  # settings need the policy
    cfg = compose(hero(), {**base, "backlog_fix": False, "budget_policy": "priority",
                           "priority": {"tail_fade_fraction": 0.25}}, served_blocks=1755)
    assert cfg["level_budget_policy"] == "priority" and cfg["priority"]["tail_fade_fraction"] == 0.25
    assert cfg["priority"]["tail_lookup"] == "remaining" and cfg["active_limit"] == 43
    assert cfg["hero_v2"]["components"]["budget_policy"]["slots"] == 43
    assert cfg["arena_variant"]["changed"]["level_budget_policy"]["after"] == "priority"


async def test_the_trial_loop_runs_priority_handovers(tmp_path, monkeypatch):
    import test_arena_early_stop as fake

    from arena import trial as trial_module
    from arena.scheduler import token_budget_settings
    from arena.trial import Trial

    games = fake.STUDY["games"][:3]
    study = {**fake.STUDY, "games": games, "wall_seconds": 2.0, "active_limit": 1, "level_budget_policy": "priority",
             "backlog_policy": "readmissions", "budget_clock": "tokens", "token_budget": token_budget_settings(),
             "priority": {}}
    study.pop("cost_normalization", None)
    arcade = fake.ScoreArcade(games)
    arcade.score = 0.0
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda server_dir: 291_900)
    monkeypatch.setattr(trial_module, "GpuMonitor", fake.OfflineMonitor)

    class Compacting(fake.Worker):
        async def run(self):
            self.game.ready.set()
            while True:
                await self.game.clock.wait_active()
                await asyncio.sleep(0.2)
                self.game.compactions += 1
                self.game.note_compaction()

    async def agents(self, records, gate, window, wall_clock, card, arc):
        workers = [Compacting(r) for r in records]
        self.config_dict = {"reasoning_effort": "xhigh", "temperature": 0.7, "gameplay_seconds": 2.0}

        async def close():
            return None

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t", server_dir=tmp_path,
                  base_url="http://127.0.0.1:1234/v1", log=lambda *a, **k: None, arcade=arcade)
    report_ = await trial.run()
    log = [json.loads(line) for line in (tmp_path / "t/admission.jsonl").read_text().splitlines()]
    reasons = [(e["game"], e["event"], e["reason"]) for e in log if e.get("event") in ("admit", "park")]
    assert reasons[:6] == [(games[0], "admit", "first_window"), (games[0], "park", "handover"),
                           (games[1], "admit", "first_window"), (games[1], "park", "handover"),
                           (games[2], "admit", "first_window"), (games[2], "park", "handover")]
    summary = report_["scheduler"]["priority"]
    assert summary["games_started"] == 3 and summary["counts"]["compaction"] >= 3
    assert report_["scheduler"]["budget_policy"] == "priority"
