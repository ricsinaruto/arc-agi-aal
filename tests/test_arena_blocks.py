"""hero-v2-blocks: the three-block scheduler (``arena.blocks``, ``Scheduler(budget_policy="blocks")``) with injected
time and tokens: block start, level-0 retirement and the filler that takes the slot, a filler's tenure, the boundary
(originals park, fillers stay), finished games, the old-game refill order, the wall end, the audit and the c110
block-1 composition."""

import json
from pathlib import Path

import pytest
from test_arena_core import Now

from arena.blocks import block_sizes, check_block1, composition, settings
from arena.clock import ActiveClock, GlobalClock
from arena.report import game_row
from arena.scheduler import GameRecord, Scheduler
from arena.scheduler_audit import audit_scheduler

ROOT = Path(__file__).resolve().parents[1]
# 9 games in 3 blocks of 3 (3 streams), a 300 s wall (100 s blocks), a 1,000-token level-0 budget.
SMALL = {"wall_seconds": 300, "blocks": 3, "slots": 3, "roster_games": 9, "level0_budget_tokens": 1000,
         "stagger_seconds": 0}


def build(count=9, **overrides):
    now = Now()
    wall = GlobalClock(300, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="tokens"), total_levels=7)
             for i in range(count)]
    for g in games:
        g.clock.pause()
    log = []
    sched = Scheduler(games, wall, log=log.append, budget_policy="blocks", backlog_policy="readmissions",
                      blocks={**SMALL, **overrides})
    for g in games:
        g.observer = sched
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    return now, wall, games, sched, log


def charge(now, game, tokens, seconds=1.0):
    with game.clock.inference():
        now.advance(seconds)
    game.clock.charge(1000, tokens)


def at(now, sched, t):
    now.value = float(t)
    sched.tick()


def events(game):
    return [(e["event"], e["reason"]) for e in game.admissions]


def active(games):
    return [g.game_id for g in games if g.state == "active"]


def report(games, sched, wall):
    return {"wall_seconds": wall.elapsed(), "config": {"gameplay_seconds": 300},
            "games": [game_row(g, {"game_id": g.game_id}) for g in games], "scheduler": sched.summary()}


def test_settings_split_110_into_37_37_36_and_validate():
    cfg = settings()
    assert cfg["block_sizes"] == [37, 37, 36] and cfg["slots"] == 37 and cfg["block_seconds"] == 10200
    assert cfg["wall_seconds"] == 30600 and cfg["level0_budget_tokens"] == 90000 and cfg["tenure_unit"] == "wall_seconds"
    assert block_sizes(10, 3) == [4, 3, 3]
    for bad in ({"slots": 36}, {"blocks": 0}, {"tenure_unit": "turns"}, {"tenure_unit": "tokens"},
                {"tenure_tokens": 5}, {"surprise": 1}, {"level0_budget_tokens": 0}):
        with pytest.raises(ValueError):
            settings(bad)
    assert settings({"tenure_unit": "tokens", "tenure_tokens": 50_000})["tenure_tokens"] == 50_000
    with pytest.raises(ValueError):
        settings(games=109)  # the roster must have the configured size


def test_c110_roster_order_block1_is_20_sources_plus_17_second_clones():
    study = json.loads((ROOT / "configs/arena_30m.json").read_text())
    excluded = ("ft09", "lp85", "ar25", "vc33", "tu93")
    games = [g for g in study["games"] if not g.startswith(excluded)]
    assert len(games) == 20
    sources = [games[i % 20] for i in range(110)]  # the local gateway's cyclic clones in roster order
    first = check_block1(sources, settings())
    assert first["distinct_sources"] == 20 and len(first["second_clones"]) == 17 and first["games"] == 37
    assert sorted({g[:4] for g in games} - {g[:4] for g in first["second_clones"]}) == ["sb26", "su15", "tr87"]
    blocks = composition(sources, settings())
    assert [b["games"] for b in blocks] == [37, 37, 36]
    with pytest.raises(ValueError):
        check_block1(sorted(sources), settings())  # sorted by source: block 1 would hold 2 clones of few games


def test_block_start_admits_block_one_together_without_budgets():
    now, wall, games, sched, log = build()
    assert active(games) == ["g0", "g1", "g2"] and sched.phase == "steady" and sched.switch_wall_seconds == 0
    assert all(events(g) == [("admit", "block_start")] for g in games[:3])
    assert all(g.admissions[0]["budget"] is None and g.role == "original" for g in games[:3])
    assert all(g.state == "backlog" for g in games[3:])
    assert [g.home_block for g in games] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert games[0].admissions[0]["block"] == 0 and games[0].admissions[0]["home_block"] == 0
    assert sched.blocks_progress()["block"] == 1 and sched.blocks_progress()["unrun"] == 6


def test_level0_retirement_at_the_budget_fills_the_next_roster_game():
    now, wall, games, sched, log = build()
    charge(now, games[1], 999)
    at(now, sched, 10)
    assert games[1].state == "active"  # 999 < 1,000
    charge(now, games[1], 1)
    at(now, sched, 12)
    assert games[1].retired and games[1].state == "parked" and events(games[1])[-1] == ("park", "level0_retired")
    assert games[3].state == "active" and events(games[3]) == [("admit", "filler")] and games[3].role == "filler"
    assert active(games) == ["g0", "g2", "g3"]
    # A game with a level is never retired, whatever its tokens.
    games[0].report_level(1, wall.elapsed())
    charge(now, games[0], 5000)
    at(now, sched, 20)
    assert games[0].state == "active" and not games[0].retired
    # Retired for good: never refilled, even when nothing else is left.
    assert games[1] not in sched.blocks_old()


def test_filler_tenure_ends_at_first_admission_plus_one_block():
    now, wall, games, sched, log = build()
    at(now, sched, 30)
    charge(now, games[0], 1000)  # retire g0 at t=31 -> g3 fills
    at(now, sched, 31)
    assert games[3].first_admit_wall == 31 and games[3].role == "filler"
    at(now, sched, 100)  # boundary: g1, g2 park; the filler stays
    assert games[3].state == "active"
    at(now, sched, 130.9)
    assert games[3].state == "active"
    at(now, sched, 131)
    assert events(games[3])[-1] == ("park", "tenure_end") and games[3].state == "parked"
    # Its slot went to the next never-run game in roster order (block 3's g6: block 2's g4, g5 started at 100).
    assert games[6].state == "active" and events(games[6]) == [("admit", "filler")]
    assert sched.blocks_counts["tenure_end"] == 1


def test_boundary_parks_originals_but_not_fillers_and_starts_the_next_block():
    now, wall, games, sched, log = build()
    charge(now, games[0], 1000)
    at(now, sched, 40)  # g0 retires, g3 is pulled forward
    at(now, sched, 99.5)
    assert active(games) == ["g1", "g2", "g3"]
    at(now, sched, 100.2)
    assert [events(g)[-1] for g in games[1:3]] == [("park", "block_end")] * 2
    assert games[3].state == "active" and events(games[3]) == [("admit", "filler")]
    assert [events(g) for g in games[4:6]] == [[("admit", "block_start")]] * 2
    assert all(g.role == "original" and g.home_block == 1 for g in games[4:6])
    assert active(games) == ["g3", "g4", "g5"] and sched.blocks_index == 1
    boundary = [d for d in sched.blocks_decisions if d["decision"] == "boundary"]
    assert boundary == [{"decision": "boundary", "ended_block": 0, "block": 1, "wall_seconds": 100.2,
                         "parked": ["g1", "g2"], "unrun_in_block": ["g4", "g5"]}]
    # The parked originals are old games that can come back; the retired one is not.
    assert [g.game_id for g in sched.blocks_old()] == ["g1", "g2"]


def test_finished_game_frees_its_slot():
    now, wall, games, sched, log = build()
    games[2].note_finished("won")
    at(now, sched, 5)
    assert games[2].state == "finished" and events(games[2])[-1] == ("finish", "finished")
    assert games[2].admissions[-1]["finished_reason"] == "won"
    assert games[3].state == "active" and events(games[3]) == [("admit", "filler")]


def test_old_refill_order_most_levels_then_fewest_tokens_once_no_unrun_games_remain():
    now, wall, games, sched, log = build()
    # Block 1: g0 clears 2 levels on 400 tokens, g1 one level on 300, g2 one level on 200.
    games[0].report_level(2, wall.elapsed())
    charge(now, games[0], 400)
    games[1].report_level(1, wall.elapsed())
    charge(now, games[1], 300)
    games[2].report_level(1, wall.elapsed())
    charge(now, games[2], 200)
    at(now, sched, 100)  # boundary 1: g0-g2 park; g3-g5 start
    at(now, sched, 200)  # boundary 2: g3-g5 park; g6-g8 start
    assert active(games) == ["g6", "g7", "g8"] and not sched.blocks_unrun()
    charge(now, games[6], 1000)  # g6 retires: nothing unrun, so the best old game refills
    at(now, sched, 201)
    assert events(games[0])[-1] == ("admit", "old_refill") and games[0].role == "old"
    charge(now, games[7], 1000)
    at(now, sched, 202)
    # g1 and g2 tie on one level: fewer tokens first (g2 200 < g1 300).
    assert events(games[2])[-1] == ("admit", "old_refill")
    assert sched.blocks_old()[0] is games[1]
    # An old refill plays to the wall end: no tenure end, no boundary park.
    at(now, sched, 299)
    assert games[0].state == "active" and games[2].state == "active"


def test_old_game_waits_until_the_unrun_games_are_used_up():
    now, wall, games, sched, log = build()
    at(now, sched, 100)
    charge(now, games[3], 1000)
    at(now, sched, 101)  # g3 retires: block 3's g6 comes before the parked block-1 games
    assert events(games[6]) == [("admit", "filler")]
    assert all(g.state == "parked" for g in games[:3])


def test_wall_end_finishes_every_game():
    now, wall, games, sched, log = build()
    at(now, sched, 150)
    now.value = 300
    sched.wall_end()
    assert all(g.admissions[-1]["event"] == "finish" and g.admissions[-1]["reason"] == "wall_end" for g in games)


def test_late_level_lifts_a_retirement():
    now, wall, games, sched, log = build()
    charge(now, games[0], 1000)
    at(now, sched, 10)
    assert games[0].retired
    games[0].report_level(1, wall.elapsed())  # the retiring reply's actions cleared level 1
    assert not games[0].retired and events(games[0])[-1] == ("progress", "retired_level")
    assert games[0] in sched.blocks_old()


def test_parked_game_over_budget_retires_in_place():
    now, wall, games, sched, log = build()
    at(now, sched, 100)  # g0 parks at the boundary with 0 tokens
    charge(now, games[0], 1200)  # its in-flight reply lands after the park
    at(now, sched, 101)
    assert games[0].retired and events(games[0])[-1] == ("progress", "level0_retired")
    assert active(games) == ["g3", "g4", "g5"]


def test_token_tenure_unit():
    now, wall, games, sched, log = build(tenure_unit="tokens", tenure_tokens=500, level0_budget_tokens=10_000)
    charge(now, games[0], 10_000)
    at(now, sched, 5)  # g0 retires; g3 fills
    charge(now, games[3], 499)
    at(now, sched, 6)
    assert games[3].state == "active"
    charge(now, games[3], 1)
    at(now, sched, 7)
    assert events(games[3])[-1] == ("park", "tenure_end")


def test_admission_log_reasons_and_progress_counts():
    now, wall, games, sched, log = build()
    charge(now, games[0], 1000)
    at(now, sched, 5)
    at(now, sched, 100)
    reasons = {(e["event"], e["reason"]) for e in log if e.get("event") in ("admit", "park", "finish")}
    assert reasons == {("admit", "block_start"), ("park", "level0_retired"), ("admit", "filler"),
                       ("park", "block_end")}
    assert all("block" in e and "role" in e for e in log if e.get("event") in ("admit", "park"))
    progress = sched.blocks_progress()
    assert progress["block"] == 2 and progress["filler"] == 1 and progress["level0_retired"] == 1
    assert progress["started"] == 6 and progress["retired"] == 1 and progress["active"] == 3
    assert progress["idle_slot_seconds"] == 0


def test_audit_accepts_a_blocks_run():
    now, wall, games, sched, log = build()
    charge(now, games[0], 1000)
    at(now, sched, 5)
    games[1].report_level(1, wall.elapsed())
    at(now, sched, 100)
    at(now, sched, 150)
    now.value = 180
    sched.wall_end()
    result = audit_scheduler(report(games, sched, wall))
    assert result["valid"], result["issues"]
    assert result["checks"]["blocks_admissions"] == 7


def test_audit_rejects_a_retirement_before_the_budget():
    now, wall, games, sched, log = build()
    charge(now, games[0], 1000)
    at(now, sched, 5)
    now.value = 50
    sched.wall_end()
    rep = report(games, sched, wall)
    rep["scheduler"]["blocks"]["settings"]["level0_budget_tokens"] = 5000
    result = audit_scheduler(rep)
    assert not result["valid"] and any("level0_retired" in i for i in result["issues"])


# --- the hero-v2-blocks build, its study and the Modal guard ------------------------------------------------------------

def load(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXES9 = ["exact_trigger", "summary_guard", "inventory", "cap_followup", "stuck_notice", "sandbox",
          "timeout_interrupt", "prompt_text", "summary_headroom"]
BLOCKS_RUN = {"token_budgets": True, "kernels_phase6": False, "prefix_cache": True, "kv_cache": "int4",
              "focus_oversubscription": True, "prompt_tokens_details": True, "prefix_match_unit": 16,
              "budget_policy": "blocks", "blocks": {"stagger_seconds": 0}, "focus_slots": 37, "spare_sequences": 4,
              "density_blocks": 1260, "graph_capture": 41,
              "window": 93184, "harness_fixes": FIXES9}


def build_blocks(tmp_path, monkeypatch, **overrides):
    from arena import hero_v2

    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)
    b = load("build_arena_blocks", ROOT / "scripts/build_arena.py")
    args = dict(paired_30m=True, arena_variant=hero_v2.BASE, opening_burst_requests=3, wall_seconds=30600,
                exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], clone_instances=110,
                hero_v2_served_blocks=1335, early_stop={"at_seconds": 7200, "always": True})
    _, study = b.build("hero-v2-blocks", output=tmp_path / "bundle", hero_v2={**BLOCKS_RUN, **overrides}, **args)
    return study


def guard(source_text):
    start = source_text.index("            if TEST == 'hero-v2-blocks':")
    end = source_text.index("            if TEST in ('hero-v2-128k-int4', 'hero-v2-streams'):")
    return "\n".join(line[12:] for line in source_text[start:end].splitlines())


def test_the_blocks_build_records_the_plan_the_stop_and_passes_its_guard(tmp_path, monkeypatch):
    study = build_blocks(tmp_path, monkeypatch)
    assert study["level_budget_policy"] == "blocks" and study["wall_seconds"] == 30600
    assert study["competition"]["preview_seconds"] == 30600 and study["active_limit"] == 37
    assert study["early_stop"] == {"at_seconds": 7200, "score_above": None, "retry_seconds": 300, "always": True}
    assert study["server"]["max_num_seqs"] == study["generation_slots"] == 41
    assert study["server"]["max_cudagraph_capture_size"] == 41 and study["server"]["prefix_match_unit"] == 16
    assert study["harness_fixes"]["fixes"] == FIXES9 and "backlog_key" not in study
    # density_blocks 1,260 (the replay's pool less a margin): a served pool of >= 1,260 blocks keeps all 37 streams.
    assert study["focus_oversubscription"]["gate_blocks_per_stream"] * 37 <= int(0.9 * 1260)
    plan = study["blocks_plan"]
    assert plan["block_sizes"] == [37, 37, 36] and plan["block1"]["distinct_sources"] == 20
    assert len(plan["block1"]["second_clones"]) == 17 and plan["run_stop_seconds"] == 7200
    timeline = study["timeline"]["plan"][0]
    assert timeline["timeout"] == 7800 and timeline["end"] + 300 <= 9540  # budgeted by the 7,200 s stop
    modal = load("modal_blocks", ROOT / "scripts/modal_validation.py")
    assert modal.validation_timeout("arena", "hero-v2-blocks") == 9600
    assert modal.arena_arguments("hero-v2-blocks") == ["--trial", "duck-transfer"]
    code = guard((ROOT / "scripts/modal_validation.py").read_text())
    exec(code, {"json": json, "TEST": "hero-v2-blocks", "source": tmp_path / "bundle"})  # passes
    path = tmp_path / "bundle/configs/comparison.json"
    for key, bad in (("wall_seconds", 7200), ("active_limit", 36), ("windows", [65536])):
        broken = {**study, key: bad}
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="hero-v2-blocks"):
            exec(code, {"json": json, "TEST": "hero-v2-blocks", "source": tmp_path / "bundle"})
    broken = json.loads(json.dumps(study))
    broken["harness_fixes"]["fixes"] = FIXES9 + ["inventory_imports"]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="hero-v2-blocks"):
        exec(code, {"json": json, "TEST": "hero-v2-blocks", "source": tmp_path / "bundle"})
    broken = json.loads(json.dumps(study))
    broken["games"] = broken["games"][:19] + broken["games"][:1]  # a repeated source: block 1 would break
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="hero-v2-blocks"):
        exec(code, {"json": json, "TEST": "hero-v2-blocks", "source": tmp_path / "bundle"})


def test_competition_rehearsal_budget_and_bootstrap_use_the_run_stop():
    from arena.competition import gameplay_seconds
    from arena.timeline import check

    study = {"competition": {"cleanup_seconds": 600, "preview_seconds": 30600},
             "early_stop": {"at_seconds": 7200, "always": True}}
    assert gameplay_seconds(study, hard_end=10_000.0, now=1_000.0) == 30600  # 8,400 s left >= 7,200: plays the wall
    with pytest.raises(TimeoutError):
        gameplay_seconds({**study, "early_stop": None}, hard_end=10_000.0, now=1_000.0)
    plan = check({"notebook_seconds": 9540, "wall_seconds": 30600, "early_stop": study["early_stop"],
                  "trials": [{"name": "duck-transfer", "kind": "duck-transfer"}]},
                 setup_seconds=1058, restart_seconds=0, run_seconds={"duck-transfer": 7500}, margin=1.0,
                 teardown_seconds=300)
    assert plan[0]["timeout"] == 7800
    text = (ROOT / "scripts/arena/bootstrap.py").read_text()
    assert "trial_wall = min(trial_wall, float(stop['at_seconds']))" in text


def test_unconditional_stop_ends_at_its_time_whatever_the_score():
    from arena.early_stop import EarlyStop
    from arena.early_stop import settings as stop_settings

    cfg = stop_settings({"at_seconds": 7200, "always": True}, 30600)
    assert cfg["always"] and cfg["score_above"] is None
    with pytest.raises(ValueError):
        stop_settings({"at_seconds": 7200, "always": True, "score_above": 1.0})
    assert "always" not in stop_settings({"at_seconds": 3600, "score_above": 1.68})  # the score rule is unchanged
    reads = []
    rule = EarlyStop(cfg, read_score=lambda t: reads.append(t) or {"score": 0.0}, paths=[])
    assert rule.check(7199.9) is None and not reads
    assert rule.check(7200.2) == "stop_at" and rule.decision == "stop" and rule.score == {"score": 0.0}
    failing = EarlyStop(cfg, read_score=lambda t: 1 / 0, paths=[])
    assert failing.check(7201) == "stop_at" and "ZeroDivisionError" in failing.error


# --- the real trial loop with fake agents: blocks, the unconditional stop, progress rows --------------------------------

async def test_the_trial_loop_runs_blocks_and_stops_unconditionally(tmp_path, monkeypatch):
    import test_arena_early_stop as fake

    from arena import trial as trial_module
    from arena.scheduler import token_budget_settings
    from arena.trial import Trial

    games = fake.STUDY["games"][:3]
    study = {**fake.STUDY, "games": games, "wall_seconds": 3.0, "active_limit": 1, "level_budget_policy": "blocks",
             "backlog_policy": "readmissions", "budget_clock": "tokens", "token_budget": token_budget_settings(),
             "blocks": {"wall_seconds": 3.0, "blocks": 3, "slots": 1, "roster_games": 3, "level0_budget_tokens": 1000,
                        "stagger_seconds": 0},
             "early_stop": {"at_seconds": 2, "always": True, "score_above": None, "retry_seconds": 300}}
    study.pop("cost_normalization", None)
    arcade = fake.ScoreArcade(games)
    arcade.score = 0.0
    monkeypatch.setattr(trial_module, "kv_pool_tokens", lambda server_dir: 291_900)
    monkeypatch.setattr(trial_module, "GpuMonitor", fake.OfflineMonitor)

    async def agents(self, records, gate, window, wall_clock, card, arc):
        workers = [fake.Worker(r) for r in records]
        self.config_dict = {"reasoning_effort": "xhigh", "temperature": 0.7, "gameplay_seconds": 3.0}

        async def close():
            return None

        return workers, (lambda: [w.result() for w in workers]), close

    monkeypatch.setattr(Trial, "duck_agents", agents)
    trial = Trial(study, study["trials"][0], source=".", games_dir="games", output=tmp_path / "t", server_dir=tmp_path,
                  base_url="http://127.0.0.1:1234/v1", log=lambda *a, **k: None, arcade=arcade)
    report = await trial.run()
    assert report["end_reason"] == "stop_at" and report["early_stop"]["decision"] == "stop"
    assert report["wall"]["configured_seconds"] == 3.0 and 2.0 <= report["wall"]["ended_seconds"] < 2.6
    blocks = report["scheduler"]["blocks"]
    assert blocks["settings"]["block_seconds"] == 1.0 and blocks["counts"]["block_end"] == 2
    assert [g["role"] for g in blocks["games"].values()] == ["original"] * 3
    log = [json.loads(line) for line in (tmp_path / "t/admission.jsonl").read_text().splitlines()]
    reasons = [(e["game"], e["event"], e["reason"]) for e in log if e.get("event") in ("admit", "park", "finish")]
    assert reasons[:5] == [(games[0], "admit", "block_start"), (games[0], "park", "block_end"),
                           (games[1], "admit", "block_start"), (games[1], "park", "block_end"),
                           (games[2], "admit", "block_start")]
    assert {r[2] for r in reasons[5:]} == {"wall_end"}
    progress = json.loads((tmp_path / "t/progress.json").read_text())
    assert progress["blocks"]["block"] == 3 and progress["blocks"]["started"] == 3


# --- staggered block starts (next build) -------------------------------------------------------------------------------

def test_staggered_start_opens_slots_one_by_one_and_originals_end_staggered():
    # 3 slots over a 30 s stagger: one slot per 10 s.
    now, wall, games, sched, log = build(stagger_seconds=30)
    assert active(games) == ["g0"] and sched.blocks_open_slots() == 1
    at(now, sched, 9.9)
    assert active(games) == ["g0"] and sched.blocks_idle["free"] == 0  # a slot not yet started is not idle
    at(now, sched, 10)
    assert active(games) == ["g0", "g1"]
    at(now, sched, 20)
    assert active(games) == ["g0", "g1", "g2"] and [g.first_admit_wall for g in games[:3]] == [0, 10, 20]
    # Boundary at 100: g0 (one block since its admission) parks; g1 and g2 play on until 110 and 120.
    at(now, sched, 100)
    assert events(games[0])[-1] == ("park", "block_end") and games[1].state == games[2].state == "active"
    assert events(games[3]) == [("admit", "block_start")] and games[3].first_admit_wall == 100
    at(now, sched, 110)
    assert events(games[1])[-1] == ("park", "block_end") and events(games[4]) == [("admit", "block_start")]
    at(now, sched, 120)
    assert events(games[2])[-1] == ("park", "block_end") and games[5].first_admit_wall == 120
    # Block 2's originals inherited the spacing: they end at 200, 210, 220 and block 3 starts the same way.
    at(now, sched, 200)
    at(now, sched, 210)
    at(now, sched, 220)
    assert [g.first_admit_wall for g in games[6:]] == [200, 210, 220]
    assert sched.blocks_idle["slot_seconds"] == 0


def test_a_retirement_inside_the_stagger_takes_the_next_roster_game_early():
    now, wall, games, sched, log = build(stagger_seconds=30)
    charge(now, games[0], 1000)
    at(now, sched, 5)
    # g0 retired inside the stagger; one slot is open, so g1 starts now instead of at 10 s.
    assert games[0].retired and active(games) == ["g1"] and games[1].first_admit_wall == 5
    at(now, sched, 10)
    assert active(games) == ["g1", "g2"]


def test_staggered_filler_keeps_its_own_tenure():
    now, wall, games, sched, log = build(stagger_seconds=30)
    at(now, sched, 20)
    charge(now, games[2], 1000)
    at(now, sched, 25)  # g2 retires: g3 fills from block 2
    assert events(games[3]) == [("admit", "filler")] and games[3].first_admit_wall == 25
    at(now, sched, 110)  # g1's block ends at 110; g3 plays on
    assert games[3].state == "active"
    at(now, sched, 125)
    assert events(games[3])[-1] == ("park", "tenure_end")


def test_audit_accepts_a_staggered_run():
    now, wall, games, sched, log = build(stagger_seconds=30)
    for t in (10, 20, 100, 110, 120, 150):
        at(now, sched, t)
    now.value = 180
    sched.wall_end()
    result = audit_scheduler(report(games, sched, wall))
    assert result["valid"], result["issues"]


def test_stagger_setting_validates():
    assert settings()["stagger_seconds"] == 600
    with pytest.raises(ValueError):
        settings({"stagger_seconds": 10200})
    with pytest.raises(ValueError):
        settings({"stagger_seconds": -1})


def test_the_next_build_has_the_096_gate_and_the_600_s_stagger_and_passes_its_guard(tmp_path, monkeypatch):
    study = build_blocks(tmp_path, monkeypatch, blocks={"stagger_seconds": 600}, gate_margin=0.96)
    assert study["kv_margin"] == 0.96 and study["blocks"]["stagger_seconds"] == 600
    layout = study["hero_v2"]["layout"]
    assert layout["gate_capacity"] == int(0.96 * 1270)
    # The density keeps 37 streams on a pool of >= 1,260 blocks at the 0.96 gate.
    assert study["focus_oversubscription"]["gate_blocks_per_stream"] * 37 <= int(0.96 * 1260)
    assert int(int(0.96 * 1260) // study["focus_oversubscription"]["gate_blocks_per_stream"]) >= 37
    assert "stagger" in study["level_budget_rule"]
    code = guard((ROOT / "scripts/modal_validation.py").read_text())
    exec(code, {"json": json, "TEST": "hero-v2-blocks", "source": tmp_path / "bundle"})  # passes
    from arena import hero_v2

    with pytest.raises(ValueError):
        hero_v2.normalize({"gate_margin": 0.99})
    assert "kv_margin" not in hero_v2.compose(hero_v2_base(), {"token_budgets": False})["arena_variant"]["changed"]


def hero_v2_base():
    from arena import hero_v2
    from arena.variants import apply_variant

    return apply_variant(json.loads((ROOT / "configs/arena_30m.json").read_text()), hero_v2.BASE)
