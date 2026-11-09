"""Competition build of the blocks3-plans-prune25-40 campaign arm ("build and push a competition
submission with the same 30-minute public preview ... 3-blocks 25% pruning, 40 x 124k"; correction: "it should be
40/40/40, where the last block continues playing some earlier games").

The blocks wall on the gateway (the submission's real play time, set at runtime), the preview as block 1 of that wall,
and a full-size dry run of the 110-game, 40-stream schedule: every block runs 40 streams, block 3 = its 30 never-run
games + the 10 best old games, and no slot stays idle while an eligible game waits."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_arena_core import Now

from arena import hero_v2
from arena.blocks import settings as blocks_settings
from arena.campaign import components
from arena.clock import ActiveClock, GlobalClock
from arena.competition import gameplay_seconds, resolve_blocks
from arena.scheduler import GameRecord, Scheduler

ROOT = Path(__file__).resolve().parents[1]
ARM = "blocks3-plans-prune25-40"
NOMINAL = 32400 - 1160 - 600


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_competition(tmp_path, monkeypatch, **hero_overrides):
    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)
    b = load("build_arena_competition_blocks", ROOT / "scripts/build_arena.py")
    hero = components(ARM)
    hero["blocks"] = {**hero["blocks"], "wall_seconds": NOMINAL}
    hero.update(hero_overrides)
    _, study = b.build("arena-flash-competition-blocks3-p25-40-test", output=tmp_path / "bundle", hero_v2=hero,
                       paired_30m=True, arena_variant=hero_v2.BASE, opening_burst_requests=3,
                       exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], hero_v2_served_blocks=1755,
                       competition=True, preview_seconds=1800, preview_instances=110, sanity_trial=False)
    return b, study


def test_the_competition_build_keeps_the_arm_and_changes_only_the_competition_fields(tmp_path, monkeypatch):
    _, study = build_competition(tmp_path, monkeypatch)
    requested = study["hero_v2"]["requested"]
    expected = components(ARM)
    for key, value in expected.items():
        if key == "blocks":
            assert requested[key] == {**value, "wall_seconds": NOMINAL}
        else:
            assert requested[key] == value, key
    assert study["level_budget_policy"] == "blocks" and study["action_plans"] is True
    assert study["windows"] == [123904] and study["active_limit"] == 40 and study["server"]["max_num_seqs"] == 44
    assert study["server"]["max_cudagraph_capture_size"] == 48 and study["completion_cap"] == 16384
    assert study["server"]["expert_prune"]["mask"] == "configs/expert-prune/prune25-layer.json"
    assert (tmp_path / "bundle" / study["server"]["expert_prune"]["mask"]).is_file()
    assert study["server"]["expert_overlay"]["check"] == "size"
    assert study["model_datasets"] == ["richardcsaky/arc-expert-gptq-arc-v1"]
    # The same 30-minute 110-instance public preview as run B's submission; the hidden rerun plays the envelope.
    competition = study["competition"]
    assert competition["preview_seconds"] == 1800 and competition["preview_instances"] == 110
    assert competition["submission_seconds"] == 32400 and competition["cleanup_seconds"] == 600
    assert "research_clones" not in competition and study["wall_seconds"] == 1800 and "early_stop" not in study
    assert competition["blocks_wall"]["nominal_seconds"] == NOMINAL
    assert [t["name"] for t in study["trials"]] == ["duck-transfer"]
    assert all(t["server_env"]["ARENA_READY_GUARD"] == "180,3000" for t in study["trials"])
    plan = study["blocks_plan"]
    assert plan["min_pool_blocks"] == 1630 and plan["block_sizes"] == [40, 40, 30]
    assert plan["streams_per_block"] == [40, 40, 40] and plan["spare_slots_for_old_refills"] == [0, 0, 10]
    assert plan["run_stop_seconds"] == 1800 and plan["block1"]["distinct_sources"] == 20
    assert "40 / 40 / 40" in study["level_budget_rule"]


def test_the_competition_build_refuses_a_wall_outside_the_envelope(tmp_path, monkeypatch):
    for wall in (1800, 31800):
        hero = components(ARM)
        with pytest.raises(ValueError, match="competition blocks"):
            build_competition(tmp_path / str(wall), monkeypatch, blocks={**hero["blocks"], "wall_seconds": wall})


def resolved_study(tmp_path, monkeypatch, games=None):
    _, study = build_competition(tmp_path, monkeypatch)
    resolved = copy.deepcopy(study)
    resolved["games"] = games or [f"c{i:03d}" for i in range(110)]
    return resolved


def test_the_preview_plays_block_one_of_the_projected_competition_wall(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    resolved = resolved_study(tmp_path, monkeypatch)
    start = 1_000_000.0
    hard_end = start + resolved["notebook_seconds"]  # the preview kernel's own 4,500 s
    now = start + 1160.0
    resolved["wall_seconds"] = gameplay_seconds(resolved, hard_end, now=now)
    assert resolved["wall_seconds"] == 1800
    resolve_blocks(resolved, start, now=now)
    assert resolved["wall_seconds"] == pytest.approx(32400 - 1160 - 600)
    assert resolved["blocks"]["wall_seconds"] == resolved["wall_seconds"]  # the trial's check
    assert resolved["early_stop"] == {"at_seconds": 1800, "score_above": None, "retry_seconds": 300, "always": True}
    cfg = blocks_settings(resolved["blocks"], games=110)
    assert cfg["block_seconds"] == pytest.approx(30640 / 3) and cfg["block_sizes"] == [40, 40, 30]
    assert resolved["blocks_plan"]["run_stop_seconds"] == 1800 and resolved["blocks_plan"]["min_pool_blocks"] == 1630
    record = resolved["competition"]["blocks_wall"]
    assert record["setup_seconds"] == 1160.0 and record["preview_stop_seconds"] == 1800
    # The preview of 1,800 s stays inside block 1 (10,213 s).
    assert 1800 < cfg["block_seconds"]


def test_the_hidden_rerun_uses_its_measured_play_time_as_the_blocks_wall(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    resolved = resolved_study(tmp_path, monkeypatch, games=[f"hidden{i}" for i in range(110)])
    start = 2_000_000.0
    hard_end = start + resolved["competition"]["submission_seconds"]
    now = start + 1203.5
    resolved["wall_seconds"] = gameplay_seconds(resolved, hard_end, now=now)
    resolve_blocks(resolved, start, now=now)
    wall = 32400 - 1203.5 - 600
    assert resolved["wall_seconds"] == pytest.approx(wall) and resolved["blocks"]["wall_seconds"] == resolved["wall_seconds"]
    assert "early_stop" not in resolved  # the hidden run plays its whole wall
    cfg = blocks_settings(resolved["blocks"], games=110)
    assert cfg["blocks"] == 3 and cfg["block_seconds"] == pytest.approx(wall / 3)
    assert resolved["competition"]["blocks_wall"]["preview_stop_seconds"] is None


@pytest.mark.parametrize("count,blocks,sizes", [(110, 3, [40, 40, 30]), (120, 3, [40, 40, 40]), (81, 3, [40, 40, 1]),
                                                (125, 4, [40, 40, 40, 5]), (80, 2, [40, 40]), (41, 2, [40, 1])])
def test_a_hidden_roster_of_another_size_keeps_40_streams(tmp_path, monkeypatch, count, blocks, sizes):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    resolved = resolved_study(tmp_path, monkeypatch, games=[f"h{i}" for i in range(count)])
    resolved["wall_seconds"] = 30000.0
    resolve_blocks(resolved, 0.0, now=1200.0)
    cfg = blocks_settings(resolved["blocks"], games=count)
    assert cfg["blocks"] == blocks and cfg["block_sizes"] == sizes and cfg["slots"] == 40
    assert cfg["block_seconds"] == pytest.approx(30000.0 / blocks)


def test_fewer_hidden_games_than_streams_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    resolved = resolved_study(tmp_path, monkeypatch, games=[f"h{i}" for i in range(30)])
    resolved["wall_seconds"] = 30000.0
    with pytest.raises(ValueError, match="largest block"):
        resolve_blocks(resolved, 0.0, now=1200.0)


def test_research_clone_campaign_studies_keep_the_build_blocks_wall(monkeypatch):
    # The nine-hour Modal campaign (110 public research clones, submission = preview = the 32,400 s study wall) runs
    # through the competition path of arena.run; the gateway projection would refuse it after setup.
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    study = {"wall_seconds": 32400, "games": [f"k{i:03d}" for i in range(110)],
             "blocks": {"wall_seconds": 32400, "blocks": 2, "slots": 76, "partition": "streams"},
             "blocks_plan": {"block_sizes": [76, 34]},
             "competition": {"preview_instances": 110, "preview_seconds": 32400, "submission_seconds": 32400,
                             "cleanup_seconds": 600, "research_clones": True}}
    resolved = copy.deepcopy(study)
    assert resolve_blocks(resolved, 0.0, now=1500.0) == study
    gateway = copy.deepcopy(study)
    del gateway["competition"]["research_clones"]
    with pytest.raises(TimeoutError):
        resolve_blocks(gateway, 0.0, now=1500.0)


def test_the_run_resolves_the_blocks_wall_on_the_gateway():
    text = (ROOT / "src/arena/run.py").read_text()
    first = text.index("resolved['wall_seconds'] = gameplay_seconds(resolved, hard_end)")
    assert text.index("resolve_blocks(resolved, args.notebook_start_epoch)") > first


# --- a full-size dry run of the 110-game, 40-stream, three-block schedule ------------------------------------------------

WALL = 30640.0
BLOCK = WALL / 3


def schedule():
    now = Now()
    wall = GlobalClock(WALL, now=now)
    games = [GameRecord(f"c{i:03d}", i, ActiveClock(wall, now=now, budget_clock="tokens"), total_levels=8)
             for i in range(110)]
    for g in games:
        g.clock.pause()
    log = []
    cfg = {**components(ARM)["blocks"], "wall_seconds": WALL, "roster_games": 110}
    sched = Scheduler(games, wall, log=log.append, budget_policy="blocks", backlog_policy="readmissions", blocks=cfg)
    for g in games:
        g.observer = sched
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    return now, wall, games, sched


def charge(now, game, tokens):
    with game.clock.inference():
        now.advance(0.001)
    game.clock.charge(1000, tokens)


def at(now, sched, t):
    now.value = float(t)
    sched.tick()


def active(games):
    return [g for g in games if g.state == "active"]


def last(game):
    return (game.admissions[-1]["event"], game.admissions[-1]["reason"])


def test_every_block_runs_40_streams_and_block_three_adds_the_10_best_old_games():
    now, wall, games, sched = schedule()
    assert [g.index for g in active(games)] == list(range(40))
    # Block 1: levels and tokens that fix the old-game order (most levels, then fewest gameplay tokens).
    for g in games[:40]:
        charge(now, g, 20000 + g.index)
    for i, levels in ((5, 3), (12, 2), (3, 2), (30, 1), (7, 1), (21, 1)):
        games[i].report_level(levels, wall.elapsed())
    at(now, sched, BLOCK - 1)
    assert len(active(games)) == 40
    at(now, sched, BLOCK + 0.5)  # boundary 1: block 1 parks, block 2's 40 start together
    assert [g.index for g in active(games)] == list(range(40, 80))
    for g in games[40:80]:
        charge(now, g, 30000)
    for i, levels in ((45, 4), (60, 1), (61, 1)):
        games[i].report_level(levels, wall.elapsed())
    at(now, sched, 2 * BLOCK + 0.5)  # boundary 2: 30 never-run games + 10 old refills = 40 streams
    now_active = active(games)
    assert len(now_active) == 40
    originals = [g.index for g in now_active if g.role == "original"]
    old = [g for g in now_active if g.role == "old"]
    assert originals == list(range(80, 110)) and len(old) == 10
    assert all(last(g) == ("admit", "old_refill") for g in old)
    # The refill order: most levels, then fewest gameplay completion tokens, then roster order.
    expected = sorted(games[:80], key=lambda g: (-g.levels, g.clock.budget_tokens(), g.index))[:10]
    assert sorted(g.index for g in old) == sorted(g.index for g in expected) == [0, 3, 5, 7, 12, 21, 30, 45, 60, 61]
    # Admitted best first: the old_refill notes in the scheduler's order.
    refills = [d["game"] for d in sched.blocks_decisions if d.get("decision") == "old_refill"]
    assert refills == [f"c{i:03d}" for i in (45, 5, 3, 12, 7, 21, 30, 60, 61, 0)]
    boundary = [d for d in sched.blocks_decisions if d["decision"] == "boundary"][-1]
    assert boundary["block"] == 2 and len(boundary["unrun_in_block"]) == 30
    progress = sched.blocks_progress()
    assert progress["block"] == 3 and progress["active"] == 40 and progress["active_by_role"]["old"] == 10
    # A retirement or a finish in block 3 frees a slot that the next best old game takes at once.
    target = games[80]
    nxt = sched.blocks_old()[0]
    assert nxt is games[1]  # level 0, the fewest tokens among the parked (20,001) after g0
    charge(now, target, (target.levels + 1) * components(ARM)["blocks"]["level_token_allowance"])
    at(now, sched, 2 * BLOCK + 10)
    assert target.retired and len(active(games)) == 40
    assert nxt.state == "active" and nxt.role == "old" and last(nxt) == ("admit", "old_refill")
    nxt = sched.blocks_old()[0]
    games[81].note_finished("won")
    at(now, sched, 2 * BLOCK + 20)
    assert games[81].state == "finished" and len(active(games)) == 40
    assert nxt.state == "active" and nxt.role == "old"
    assert sum(g.role == "old" for g in active(games)) == 12
    # No slot was idle while an eligible game waited, over the whole run to the wall end.
    at(now, sched, WALL - 1)
    assert len(active(games)) == 40
    assert sched.blocks_idle["slot_seconds_with_eligible"] == 0.0


# --- the idling fix: retired games continue when nothing else can take a free slot -------------

def small(count=6):
    now = Now()
    wall = GlobalClock(300, now=now)
    games = [GameRecord(f"g{i}", i, ActiveClock(wall, now=now, budget_clock="tokens"), total_levels=7)
             for i in range(count)]
    for g in games:
        g.clock.pause()
    cfg = {"wall_seconds": 300, "blocks": 2, "slots": 3, "roster_games": count, "level0_budget_tokens": 1000,
           "stagger_seconds": 0, "level_token_allowance": 1000}
    sched = Scheduler(games, wall, log=[].append, budget_policy="blocks", backlog_policy="readmissions", blocks=cfg)
    for g in games:
        g.observer = sched
    wall.release()
    for g in games:
        g.clock.release()
    sched.start_all()
    return now, wall, games, sched


def retire_all_old(now, wall, games, sched):
    games[1].report_level(2, wall.elapsed())
    games[2].report_level(1, wall.elapsed())
    at(now, sched, 150.5)  # boundary: g0-g2 park, g3-g5 start; every game has started
    for g in games[:3]:  # the parked old games retire in place (over their allowances)
        charge(now, g, 1000 * (g.levels + 1))
    at(now, sched, 151)
    assert all(g.retired for g in games[:3]) and not sched.blocks_old() and not sched.blocks_unrun()


def test_a_free_slot_with_nothing_else_eligible_continues_the_retired_game_with_most_levels():
    now, wall, games, sched = small()
    retire_all_old(now, wall, games, sched)
    charge(now, games[3], 1000)  # g3 retires: only retired games remain -> the most levels (g1) continues
    at(now, sched, 152)
    assert games[3].retired and games[1].state == "active" and last(games[1]) == ("admit", "retired_continue")
    assert not games[1].retired and games[1].continuations == 1 and len(active(games)) == 3
    # A fresh allowance for its current level: not retired at once, retired again after another 1,000 x (2 + 1).
    charge(now, games[1], 2999)
    at(now, sched, 154)
    assert games[1].state == "active"
    charge(now, games[1], 1)
    at(now, sched, 155)
    # g1 re-retires; the next retired game in rotation (fewest continuations, then most levels: g2) takes the slot.
    assert games[1].retired and last(games[2]) == ("admit", "retired_continue") and len(active(games)) == 3
    assert sched.blocks_counts["retired_continue"] == 2
    # A finished game never continues; its slot goes to the next retired game.
    games[4].note_finished("won")
    at(now, sched, 156)
    assert games[4].state == "finished" and len(active(games)) == 3
    at(now, sched, 299)
    assert sched.blocks_idle["slot_seconds_with_eligible"] == 0.0 and len(active(games)) == 3


def test_a_new_clear_after_a_continuation_restores_the_normal_allowance():
    now, wall, games, sched = small()
    retire_all_old(now, wall, games, sched)
    charge(now, games[3], 1000)
    at(now, sched, 152)
    assert games[1].allowance_base == 3000
    games[1].report_level(3, wall.elapsed())
    assert games[1].allowance_base == 0.0 and games[1].spent_on_level() == 0


def test_the_audit_accepts_continuations():
    from arena.report import game_row
    from arena.scheduler_audit import audit_scheduler

    now, wall, games, sched = small()
    retire_all_old(now, wall, games, sched)
    charge(now, games[3], 1000)
    at(now, sched, 152)
    at(now, sched, 300)
    audit_scheduler({"wall_seconds": wall.elapsed(), "config": {"gameplay_seconds": 300},
                     "games": [game_row(g, {"game_id": g.game_id}) for g in games], "scheduler": sched.summary()})


# --- competition int3 builds: the pinned recipe's setup ceilings raised (30 September) ------------------------------

def test_setup_timeouts_env_is_parsed_and_bounded():
    from arena.serving import lab_overrides as lab_override

    assert lab_override({"ARENA_SETUP_TIMEOUTS": "2700,3600"})["setup_timeouts"] == (2700, 3600)
    assert "setup_timeouts" not in lab_override({})
    for bad in ("1200,1800", "2700,1700", "2700,8000"):
        with pytest.raises(ValueError, match="ARENA_SETUP_TIMEOUTS"):
            lab_override({"ARENA_SETUP_TIMEOUTS": bad})


def test_configure_raises_the_recipe_globals_before_main():
    text = (ROOT / "src/arena/serving.py").read_text()
    assert "recipe.SERVER_READY_TIMEOUT, recipe.SETUP_TOTAL_TIMEOUT = (int(v) for v in setup_timeouts)" in text
    assert "setup_timeouts=lab.get('setup_timeouts')" in text
    pinned = (ROOT / "src/arc_infra/serving_setup.py").read_text()
    # main() and wait_for_server read the module globals at call time.
    assert "setup_deadline = setup_started + SETUP_TOTAL_TIMEOUT" in pinned
    assert "deadline = min(time.monotonic() + SERVER_READY_TIMEOUT, setup_deadline)" in pinned


def test_only_competition_int3_builds_carry_the_raised_setup_timeouts(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)
    b = load("build_arena_competition_int3", ROOT / "scripts/build_arena.py")
    monkeypatch.syspath_prepend(str(ROOT / "scripts/arena"))
    sys.modules.pop("build_campaign", None)
    from build_campaign import served_blocks

    arm = "blocks2-plans-prune50-int3"
    hero = components(arm)
    hero["blocks"] = {**hero["blocks"], "wall_seconds": 30500}
    _, study = b.build("int3-competition-test", output=tmp_path / "int3", hero_v2=hero, paired_30m=True,
                       arena_variant=hero_v2.BASE, opening_burst_requests=3,
                       exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], hero_v2_served_blocks=served_blocks(arm),
                       competition=True, preview_seconds=600, preview_instances=110, sanity_trial=False)
    env = study["trials"][0]["server_env"]
    assert env["ARENA_SETUP_TIMEOUTS"] == "2700,3600" and env["ARENA_READY_GUARD"] == "180,3000"
    assert study["notebook_seconds"] == 4500
    _, marlin = build_competition(tmp_path / "marlin", monkeypatch)
    assert "ARENA_SETUP_TIMEOUTS" not in marlin["trials"][0]["server_env"]
