"""hero-v2 follow-up runs (29 September; docs/validation-20260927/hero-v2-next-runs.md): the int4 / 128K layout, the
explicit and oversubscribed focus targets, the progress-fruitless backlog order, the server flags and both builds."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_arena_core import Now

from arena import hero_v2
from arena.capacity import competition_focus_fit
from arena.clock import ActiveClock, GlobalClock
from arena.scheduler import GameRecord, Scheduler
from arena.serving import configure
from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
SERVED = 1335  # hero-v2 smoke v2: the measured int8 pool


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hero():
    return apply_variant(CONTROL, hero_v2.BASE)


def layout(**components):
    return hero_v2.served_layout(hero_v2.normalize({"token_budgets": False, **components}), SERVED)


# --- layout -------------------------------------------------------------------------------------------------------------

def test_int4_128k_layout_is_the_int8_pool_recut_and_the_gate_rows():
    row = layout(kv_cache="int4", window=131072)
    # The kv4 sizing's int4 pool on hero-v2's served bytes (kv-cache-4bit.md section 4).
    sizing = load("kv4_sizing_next", ROOT / "scripts/kv4_sizing.py").hero_v2_table()["int4"]["128k"]
    assert (row["blocks"], row["block_tokens"], row["gate_capacity"]) == (sizing["blocks"], 3136, sizing["gate"]) \
        == (1270, 3136, 1143)
    # The gate charges the cost table's 1,024-token row at or above 131,328 tokens: 132,096 -> 43 + 9 = 52 blocks
    # (the sizing's 51 rounds to the token), so 21 full 128K windows fit 1,143 gate blocks; a 22nd needs 1,272 blocks.
    assert sizing["per_window"] == 51 and row["blocks_per_window"] == 52 and row["windows_fit"] == 21
    assert hero_v2.cost_blocks("int8", 65536) == 51 and hero_v2.cost_blocks("int8", 131072) == 92
    assert int(0.9 * 1272) // 52 == 22 and int(0.9 * 1271) // 52 == 21
    assert row["max_num_seqs"] == 25 and row["active_limit"] == 21 and row["graph_capture_size"] is None
    # A target one above the projection runs what the served pool fits (focus_fit_fallback); two above refuses.
    target = layout(kv_cache="int4", window=131072, focus_slots=22)
    assert target["active_limit"] == 22 and target["max_num_seqs"] == 25 and not target["oversubscribed"]
    with pytest.raises(ValueError, match="focus_oversubscription"):
        layout(kv_cache="int4", window=131072, focus_slots=23)
    # The user's run-1 window (29 September): 130,816 + 256 = 131,072 tokens, row 131,072 -> 42 + 9 = 51 blocks: 22 fit.
    run1 = layout(kv_cache="int4", window=130816, focus_slots=22)
    assert run1["blocks_per_window"] == 51 and run1["windows_fit"] == 22 and not run1["oversubscribed"]
    assert run1["active_limit"] == 22 and run1["max_num_seqs"] == 25
    # hero-v2 itself is unchanged: 23 x 64K on int8, 25 sequences.
    base = layout()
    assert (base["blocks"], base["blocks_per_window"], base["active_limit"], base["max_num_seqs"]) == (1335, 51, 23, 25)


def test_oversubscribed_focus_keeps_the_exact_gate_and_raises_sequences_and_graphs():
    row = layout(focus_slots=30, focus_oversubscription=True)
    assert row["oversubscribed"] and row["windows_fit"] == 23 and row["active_limit"] == 30
    assert row["max_num_seqs"] == 32 and row["graph_capture_size"] is None     # 32 is the pinned capture size
    big = layout(focus_slots=36, focus_oversubscription=True)
    assert big["max_num_seqs"] == 38 and big["graph_capture_size"] == 40
    for bad in ({"focus_slots": 13}, {"focus_slots": 93}, {"focus_oversubscription": True},
                {"kv_cache": "k8v4"}, {"window": 98000}, {"window": 262144}, {"budget_policy": "index"},
                {"prefix_match_unit": 24}, {"prefix_match_unit": 1600}):
        with pytest.raises(ValueError):
            hero_v2.normalize(bad)
    assert hero_v2.normalize({"prefix_match_unit": 16})["prefix_match_unit"] == 16
    assert hero_v2.normalize({"kv_cache": "int4", "prefix_match_unit": 64})["prefix_match_unit"] == 64


# --- composition ----------------------------------------------------------------------------------------------------------

def test_run1_composition_int4_128k_and_the_hero_v2_harness():
    study = hero_v2.compose(hero(), {"token_budgets": False, "prefix_cache": True, "kv_cache": "int4",
                                     "window": 131072, "focus_slots": 22, "prompt_tokens_details": True},
                            served_blocks=SERVED)
    kernels = study["server"]["serving_kernels"]
    assert kernels["kv_cache_4bit"] == "int4" and "kv_cache_8bit" not in kernels and study["server"]["mtp_tokens"] == 0
    assert study["windows"] == [131072] and study["active_limit"] == 22 and study["generation_slots"] == 25
    assert study["require_focus_windows_fit"] and study["focus_fit_fallback"] == {"min_slots": 14}
    assert study["server"]["prompt_tokens_details"] is True and "prefix_match_unit" not in study["server"]
    assert study["level_budget_policy"] == "fixed-2x" and "backlog_key" not in study and "level0_order" not in study
    assert study["cache_experiment"]["prefix_cache"]
    variant = study["arena_variant"]
    assert variant["reference_difference"]["windows"] == {"before": [65536], "after": [131072]}
    assert "windows" not in variant["changed"]  # the paired control's own 128K window
    record = study["hero_v2"]["components"]
    assert record["kv_cache"]["on"] and record["kv_cache"]["block_tokens"] == 3136
    assert not record["backlog_fix"]["on"] and record["prefix_match_unit"]["on"] is False


def test_run2_composition_streams_oversubscribed_and_the_backlog_fix():
    study = hero_v2.compose(hero(), {"token_budgets": False, "prefix_cache": True, "focus_slots": 30,
                                     "focus_oversubscription": True, "prompt_tokens_details": True,
                                     "backlog_fix": True}, served_blocks=SERVED)
    assert study["active_limit"] == 30 and study["server"]["max_num_seqs"] == study["generation_slots"] == 32
    assert study["require_focus_windows_fit"] is False and "focus_fit_fallback" not in study
    over = study["focus_oversubscription"]
    assert over["full_windows_fit_projected"] == 23 and over["gate_blocks_per_stream"] == pytest.approx(1201 / 30, 1e-3)
    assert study["backlog_key"] == "progress-fruitless" and study["level0_order"] == "tokens"
    assert study["server"]["serving_kernels"]["kv_cache_8bit"] == "int8" and study["windows"] == [65536]
    assert "max_cudagraph_capture_size" not in study["server"]


def test_rank_policy_stays_selectable_on_hero_v2():
    study = hero_v2.compose(hero(), {"token_budgets": False, "budget_policy": "rank", "focus_slots": 28,
                                     "focus_oversubscription": True}, served_blocks=SERVED)
    assert study["level_budget_policy"] == "rank" and study["rank"]["slots"] == 28 == study["active_limit"]
    assert study["rank"]["tick_seconds"] == 60.0 and study["rank"]["tenure_tokens"] == 10_000
    with pytest.raises(ValueError, match="backlog_fix"):
        hero_v2.compose(hero(), {"token_budgets": False, "budget_policy": "rank", "backlog_fix": True})


# --- the startup fit ----------------------------------------------------------------------------------------------------

def budget(capacity, cost=51):
    return NS(capacity=capacity, cost=lambda tokens: cost, layout={"max_model_len": 262144, "lookahead": 0})


def test_oversubscription_records_the_fit_and_keeps_the_validated_density():
    over = {"gate_blocks_per_stream": 1201 / 30, "min_full_windows": 14, "min_slots": 14}
    study = {"active_limit": 30, "focus_oversubscription": over}
    result = competition_focus_fit(budget(1201), 65536, study)
    assert study["active_limit"] == 30 and result["oversubscribed"] and result["oversubscription"] == 7
    assert result["full_window_requests"] == 23 and not result["required"]
    smaller = {"active_limit": 30, "focus_oversubscription": over, "rank": {"slots": 30}}
    result = competition_focus_fit(budget(1150), 65536, smaller)   # a pool that came up smaller at 32 sequences
    assert smaller["active_limit"] == 28 == result["active_limit"] and smaller["rank"]["slots"] == 28
    with pytest.raises(RuntimeError, match="at least 14"):
        competition_focus_fit(budget(13 * 51 + 5), 65536, {"active_limit": 30, "focus_oversubscription": over})
    with pytest.raises(ValueError, match="replaces"):
        competition_focus_fit(budget(1201), 65536, {"active_limit": 30, "focus_oversubscription": over,
                                                    "focus_fit_fallback": {"min_slots": 14}})
    # The run-1 fallback: the target 22 runs 21 when the served int4 pool fits 21 full 128K windows.
    run1 = {"active_limit": 22, "focus_fit_fallback": {"min_slots": 14}, "require_focus_windows_fit": True}
    assert competition_focus_fit(budget(1143, 52), 131072, run1)["active_limit"] == 21 == run1["active_limit"]


# --- the progress-fruitless backlog order ------------------------------------------------------------------------------

def token_games(count, now):
    wall = GlobalClock(7200, now=now)
    games = [GameRecord(f"k{i:03d}", i, ActiveClock(wall, now=now, budget_clock="tokens"), total_levels=7)
             for i in range(count)]
    for g in games:
        g.clock.pause()
    return wall, games


def scheduler(games, wall, **kw):
    return Scheduler(games, wall, active_limit=1, switch_games=1, budget_policy="fixed-2x",
                     backlog_policy="readmissions", token_floor=1920, **kw)


def test_the_order_is_progress_then_fruitless_on_level_then_levels_then_level0_tokens():
    now = Now()
    wall, games = token_games(6, now)
    s = scheduler(games, wall, backlog_key="progress-fruitless", level0_order="tokens")
    for g in games:
        g.state = "parked"
    progress, deep, stuck, failed, rich, poor = games
    progress.levels, progress.admit_levels = 1, 0                      # cleared since its last admission
    deep.levels, deep.admit_levels, deep.readmissions = 2, 2, 4          # parked twice, no fruitless tenure on L2
    stuck.levels, stuck.admit_levels, stuck.fruitless_on_level = 3, 3, 2
    failed.readmissions, failed.fruitless_on_level = 1, 1               # a failed level-0 audition
    wall.release()
    for g in (rich, poor):
        g.clock.release()
    rich.clock.charge(4000, 9000)
    poor.clock.charge(4000, 5000)
    assert [g.game_id for g in s.backlog()] == ["k000", "k001", "k004", "k005", "k003", "k002"]
    # The readmissions order (hero-v2): the stuck L3 game first, the deep game parked four times last.
    old = scheduler(games, wall)
    assert [g.game_id for g in old.backlog()] == ["k002", "k000", "k004", "k005", "k003", "k001"]
    with pytest.raises(ValueError, match="progress-fruitless"):
        Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="spent", backlog_key="progress-fruitless")
    with pytest.raises(ValueError, match="level0_order"):
        scheduler(games, wall, level0_order="alphabet")


def run_steady(s, games, wall, now):
    for g in games:
        g.observer = s
    wall.release()
    for g in games:
        g.clock.release()
    s.start_all()


def test_fruitless_parks_count_per_level_and_reset_at_a_clear_and_bounces_are_logged():
    now = Now()
    wall, games = token_games(3, now)
    s = scheduler(games, wall, backlog_key="progress-fruitless", level0_order="tokens")
    run_steady(s, games, wall, now)
    a, b, c = games
    a.clock.charge(4000, 3000)
    a.report_level(1, 10.0)            # the switch: a is kept at level 1, b and c park (not fruitless: the opening)
    s.tick()
    assert s.phase == "steady" and a.state == "active" and a.admit_levels == 1
    assert b.fruitless_on_level == c.fruitless_on_level == 0
    b.clock.charge(4000, 800)          # b has more level-0 tokens: it auditions first
    a.clock.charge(4000, 20000)        # a exhausts its L1 budget without a clear: a fruitless tenure on L1
    s.tick()
    assert a.state == "parked" and a.fruitless_on_level == 1 and b.state == "active"
    assert s.backlog()[0] is c         # c (0 fruitless) passes a (1 fruitless on L1)
    b.clock.charge(4000, 50000)        # b's audition fails; c takes the slot; a waits behind c, ahead of b
    s.tick()
    assert b.fruitless_on_level == 1 and c.state == "active" and [g.game_id for g in s.backlog()] == ["k000", "k001"]
    a.report_level(2, 30.0)            # a clear while parked (the in-flight reply): progress, count reset
    assert a.fruitless_on_level == 0 and a.pending_parked_progress
    c.clock.charge(4000, 50000)
    s.tick()
    assert a.state == "active" and a.admissions[-1]["reason"] == "parked_progress"
    # A bounce: the parked game tops the backlog again in the same tick; park + re-admission stay, logged as such.
    lone_now = Now()
    wall2, (x, y) = token_games(2, lone_now)
    s2 = scheduler([x, y], wall2, backlog_key="progress-fruitless")
    run_steady(s2, [x, y], wall2, lone_now)
    x.clock.charge(4000, 3000)
    x.report_level(1, 5.0)
    y.clock.charge(4000, 5000)         # y clears later in tokens: x is the kept game
    y.report_level(1, 6.0)
    s2.tick()
    # y cleared L1 in the opening and was not kept: that is progress since its last admission (it would go first);
    # make it a stuck game instead, so x's exhausted budget bounces x straight back.
    assert s2.priority(y)[0] is False
    y.admit_levels, y.fruitless_on_level = 1, 3
    x.clock.charge(4000, 20000)
    s2.tick()
    reasons = [e["reason"] for e in x.admissions]
    assert reasons[-2:] == ["level_budget", "regrant_same_tick"] and x.readmissions == 2 and x.fruitless_on_level == 1
    assert s2.summary()["same_tick_regrants"] == 1 and s2.summary()["backlog_key"] == "progress-fruitless"


def test_the_audit_accepts_same_tick_regrants():
    source = (ROOT / "src/arena/scheduler_audit.py").read_text()
    assert "'regrant_same_tick'" in source


# --- server flags ---------------------------------------------------------------------------------------------------------

def test_prompt_tokens_details_and_prefix_match_unit_reach_the_server(monkeypatch, tmp_path):
    from arena import serving

    monkeypatch.setattr(serving, "apply", lambda *a: None)
    r = NS(verify_and_extract_runtime=lambda: {}, server_command=lambda: ["serve", "model"], RUNTIME_ROOT=tmp_path)
    result = configure(r, tmp_path, prompt_tokens_details=True, prefix_match_unit=16)
    assert r.server_command() == ["serve", "model", "--enable-prompt-tokens-details", "--prefix-match-unit", "16"]
    assert result["prefix_match_unit"] == 16 and result["prompt_tokens_details"] is True
    with pytest.raises(ValueError):
        configure(NS(), tmp_path, prefix_match_unit=10)
    source = (ROOT / "src/arena/serving.py").read_text()
    assert "study['server'].get('prompt_tokens_details', False)" in source
    assert "prefix_match_unit=lab.get('prefix_match_unit', study['server'].get('prefix_match_unit'))" in source


# --- builds and the study check ----------------------------------------------------------------------------------------

def build_args(**extra):
    return dict(paired_30m=True, arena_variant=hero_v2.BASE, opening_burst_requests=3, wall_seconds=7200,
                exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], clone_instances=110,
                hero_v2_served_blocks=SERVED, early_stop={"at_seconds": 3600, "score_above": 1.68}, **extra)


RUN1 = {"token_budgets": False, "kernels_phase6": False, "prefix_cache": True, "kv_cache": "int4", "window": 130816,
        "focus_slots": 22, "prompt_tokens_details": True}
RUN2 = {"token_budgets": False, "kernels_phase6": False, "prefix_cache": True, "focus_slots": 30,
        "focus_oversubscription": True, "prompt_tokens_details": True, "backlog_fix": True}


@pytest.mark.parametrize("test,components", [("hero-v2-128k-int4", RUN1), ("hero-v2-streams", RUN2)])
def test_builds_carry_the_early_stop_and_pass_their_modal_checks(tmp_path, test, components, monkeypatch):
    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)
    b = load(f"build_arena_{test}", ROOT / "scripts/build_arena.py")
    _, study = b.build(test, output=tmp_path / "bundle", hero_v2=dict(components, token_budgets=True), **build_args())
    assert study["early_stop"] == {"at_seconds": 3600, "score_above": 1.68, "retry_seconds": 300}
    assert study["budget_clock"] == "tokens" and study["level_budget_policy"] == "fixed-2x"
    check = load("study_check", ROOT / "scripts/arena/study_check.py")
    summary = check.summarize(study)
    assert summary["early_stop"]["score_above"] == 1.68 and summary["window"] == study["windows"][0]
    text = check.render(summary)
    assert "early_stop" in text and "kv_cache" in text
    modal = load(f"modal_{test}", ROOT / "scripts/modal_validation.py")
    assert modal.validation_timeout("arena", test) == 9600
    source = (ROOT / "scripts/modal_validation.py").read_text()
    assert f"'{test}'" in source
    with pytest.raises(ValueError, match="before the wall"):
        b.build(test, output=tmp_path / "bad", hero_v2=dict(components, token_budgets=True),
                **{**build_args(), "early_stop": {"at_seconds": 7200, "score_above": 1.68}})


def test_the_backlog_replay_reproduces_the_recorded_order():
    sys.path.insert(0, str(ROOT / "scripts/arena_analysis"))
    signals = load("backlog_signals_test", ROOT / "scripts/arena_analysis/backlog_signals.py")
    assert set(signals.RULES) >= {"current", "progress-fruitless+tokens"}
    state = {"pending": False, "levels": 0, "admit_levels": 0, "fruitless_on_level": 0, "readmissions": 0,
             "level0_tokens": 7000, "index": 5}
    deep = {**state, "levels": 2, "admit_levels": 2, "readmissions": 3, "index": 1}
    rule = signals.RULES["progress-fruitless+tokens"]
    assert rule(deep) < rule(state) and signals.RULES["current"](state) < signals.RULES["current"](deep)


# --- runs A and B (two parallel 2 h runs, no early stop) --------------------------------------------

RUN_A = {"token_budgets": False, "kernels_phase6": False, "prefix_cache": True, "kv_cache": "int4",
         "focus_slots": 38, "focus_oversubscription": True, "prompt_tokens_details": True, "backlog_fix": True,
         "harness_fixes": True}
RUN_B = {**RUN_A, "window": 130816, "focus_slots": 24}


def test_runs_a_and_b_oversubscribe_the_int4_fit_within_the_captured_graphs():
    a = hero_v2.compose(hero(), RUN_A, served_blocks=SERVED)
    assert a["hero_v2"]["layout"]["windows_fit"] == 36 and a["active_limit"] == 38     # 31 blocks per 64K window
    assert a["server"]["max_num_seqs"] == a["generation_slots"] == 40
    assert a["server"]["max_cudagraph_capture_size"] == 40
    assert a["focus_oversubscription"]["focus_slots"] > a["focus_oversubscription"]["full_windows_fit_projected"]
    b = hero_v2.compose(hero(), RUN_B, served_blocks=SERVED)
    assert b["windows"] == [130816] and b["active_limit"] == 24 and b["server"]["max_num_seqs"] == 26
    assert b["focus_oversubscription"]["full_windows_fit_projected"] == 22 and "max_cudagraph_capture_size" not in b["server"]
    assert a["harness_fixes"]["fixes"] and b["backlog_key"] == "progress-fruitless"
    source = (ROOT / "src/arc_infra/serving.py").read_text()
    assert "25 <= generation_slots <= 96" in source


@pytest.mark.parametrize("test,components,window", [("hero-v2-int4-streams", RUN_A, 65536),
                                                    ("hero-v2-128k-streams", RUN_B, 130816)])
def test_runs_a_and_b_build_without_early_stop_with_a_timeline_and_pass_their_guard(tmp_path, monkeypatch, test,
                                                                                     components, window):
    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)
    b = load(f"build_arena_{test}_ab", ROOT / "scripts/build_arena.py")
    args = {k: v for k, v in build_args().items() if k != "early_stop"}
    _, study = b.build(test, output=tmp_path / "bundle", hero_v2=dict(components, token_budgets=True), **args)
    assert "early_stop" not in study and study["windows"] == [window]
    plan = study["timeline"]["plan"][0]
    assert plan["trial"] == "duck-transfer" and plan["end"] + 300 <= 9540 and plan["timeout"] >= 7500
    modal = load(f"modal_{test}_ab", ROOT / "scripts/modal_validation.py")
    assert modal.validation_timeout("arena", test) == 9600
    assert modal.arena_arguments(test) == ["--trial", "duck-transfer"]
    source = (ROOT / "scripts/modal_validation.py").read_text()
    start = source.index("            if TEST in ('hero-v2-int4-streams', 'hero-v2-128k-streams'):")
    end = source.index("            if TEST in ('hero-v2-128k-int4', 'hero-v2-streams'):")
    guard = "\n".join(line[12:] for line in source[start:end].splitlines())
    exec(guard, {"json": json, "TEST": test, "source": tmp_path / "bundle"})       # passes
    other = "hero-v2-128k-streams" if test == "hero-v2-int4-streams" else "hero-v2-int4-streams"
    with pytest.raises(ValueError):
        exec(guard, {"json": json, "TEST": other, "source": tmp_path / "bundle"})
    assert "CPU_CEILING = 16 if TEST in (\"hero-v2-int4-streams\", \"hero-v2-128k-streams\"" in source
