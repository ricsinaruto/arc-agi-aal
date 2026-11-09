import copy

import pytest

from arena.campaign import ARMS, FIXES, PLANS_OFF_TWO_BLOCK, components
from arena.hero_v2 import compose
from arena.scheduler import Scheduler
from test_arena_blocks import build, charge, at, report
from test_hero_v2 import hero
from arena.scheduler_audit import audit_scheduler


def served_blocks(arm):
    import runpy
    from pathlib import Path

    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/arena/build_campaign.py"))[
        "served_blocks"](arm)


def test_per_level_allowance_is_independent_and_excludes_summaries():
    now, wall, games, sched, _ = build(level_token_allowance=1000)
    game = games[0]
    charge(now, game, 700)
    game.report_level(1, wall.elapsed())
    game.clock.charge(1000, 500, purpose="compaction")
    charge(now, game, 1999)
    sched.tick()
    assert not game.retired
    # Neither reset reports nor wall/admission waiting refresh or consume allowance.
    game.report_level(0, wall.elapsed())
    assert game.levels == 1
    charge(now, game, 1)
    sched.tick()
    assert game.retired and game.state == "parked"
    game.report_level(2, wall.elapsed())  # late already-issued action may clear
    assert not game.retired
    assert game.spent_on_level() == 0
    assert audit_scheduler(report(games, sched, wall))["valid"]


def test_level_allowance_survives_block_parks_and_old_refill():
    now, wall, games, sched, _ = build(level_token_allowance=1000)
    game = games[0]
    charge(now, game, 600)
    at(now, sched, 101)
    assert game.state == "parked"
    charge(now, game, 400)  # pending completion after park
    sched.tick()
    assert game.retired
    assert game not in sched.blocks_old()


PRUNE50_ARM = "blocks2-plans-prune50-int3"
PRUNE37_INT3_ARM = "blocks2-plans-prune37-int3"


@pytest.mark.parametrize("arm", ARMS)
def test_campaign_composition(arm):
    cfg = compose(hero(), components(arm), served_blocks=served_blocks(arm))
    assert cfg["budget_clock"] == "tokens"
    if arm == PRUNE37_INT3_ARM:
        assert cfg["active_limit"] == 60 and cfg["server"]["max_num_seqs"] == 64
        assert cfg["server"]["max_cudagraph_capture_size"] == 64
    elif arm == PRUNE50_ARM:
        assert cfg["active_limit"] == 76 and cfg["server"]["max_num_seqs"] == 80
        assert cfg["server"]["max_cudagraph_capture_size"] == 80
    elif arm.startswith("blocks2"):
        assert cfg["active_limit"] == 55
        assert cfg["server"]["max_num_seqs"] == 59
        assert cfg["server"]["max_cudagraph_capture_size"] == 64
    if arm in PLANS_OFF_TWO_BLOCK:
        assert cfg["completion_cap"] == 12288
        assert not cfg.get("action_plans")
    elif arm.startswith("blocks"):
        # every other blocks arm is based on the hero blocks3-plans run (plans on, 16,384 cap).
        assert cfg["completion_cap"] == 16384 and cfg["action_plans"]
    if cfg.get("blocks"):
        assert cfg["blocks"]["level_token_allowance"] == 90000
        assert cfg["blocks"]["stagger_seconds"] == 0


def test_b31_retirement_and_remaining_grants():
    now, wall, games, _, _ = build()
    for g in games:
        g.state = "backlog"
    sched = Scheduler(games, wall, budget_policy="fixed-2x", backlog_policy="readmissions",
                      active_limit=2, level_token_allowance=1000)
    for g in games:
        g.observer = sched
    sched.start_all()
    g = games[0]
    charge(now, g, 700)
    assert sched.grant_for(g) == 300
    g.report_level(1, wall.elapsed())
    assert sched.grant_for(g) == 2000
    charge(now, g, 2000)
    sched.tick()
    assert g.retired and g not in sched.backlog()
    g.report_level(2, wall.elapsed())
    assert not g.retired and sched.grant_for(g) == 3000


def test_two_block_balanced_roster():
    from arena.blocks import settings, check_block1
    cfg = settings({"blocks": 2, "slots": 55})
    sources = [f"g{i % 20}" for i in range(110)]
    assert check_block1(sources, cfg)["max_clones_of_one_source"] == 3
    sources[:55] = ["g0"] * 55
    with pytest.raises(ValueError):
        check_block1(sources, cfg)


def test_pruned_campaign_keeps_55_streams_on_measured_pool():
    c = components("blocks2-prune25")
    cfg = compose(hero(), c, served_blocks=1647)
    density = cfg["focus_oversubscription"]["gate_blocks_per_stream"]
    assert int(int(1647 * .96) // density) >= 55
    assert c["window"] == 68608
    assert c["density_blocks"] == 1630


def test_four_block_boundaries_and_independent_allowances():
    from arena.blocks import settings, block_at, check_block1

    cfg = settings(components('blocks4-plans')['blocks'])
    assert cfg['block_sizes'] == [28, 28, 27, 27]
    assert cfg['block_seconds'] == 8100
    assert [block_at(t, cfg) for t in (0, 8099, 8100, 16200, 24300, 32399)] == [0, 0, 1, 2, 3, 3]
    assert check_block1([f'g{i % 20}' for i in range(110)], cfg)['distinct_sources'] == 20
    now, wall, games, sched, _ = build(count=110, wall_seconds=300, blocks=4, slots=28,
                                      roster_games=110, level_token_allowance=90000)
    assert sum(g.state == 'active' for g in games) == 28
    at(now, sched, 75)
    assert all(g.state == 'active' for g in games[28:56])
    at(now, sched, 150)
    assert all(g.state == 'active' for g in games[56:83])
    # The 27-game third block leaves one slot: the first fourth-block game starts early.
    assert games[83].role == 'filler' and games[83].first_admit_wall == 150
    at(now, sched, 225)
    assert games[83].state == 'parked'  # its full tenure is already consumed
    assert all(g.state == 'active' for g in games[84:110])
    assert audit_scheduler(report(games, sched, wall))['valid']


def test_runb64_retains_first_clear_grants_and_readmission_policy():
    cfg = compose(hero(), components('runb64-plans'), served_blocks=1335)
    assert cfg['level_budget_policy'] == 'fixed-2x'
    assert cfg.get('level_token_allowance') is None and cfg.get('blocks') is None
    assert cfg['backlog_key'] == 'progress-fruitless'
    assert cfg['level0_order'] == 'tokens'
    assert cfg['action_plans'] and cfg['active_limit'] == 46
    assert cfg['server']['max_num_seqs'] == 50
    assert cfg['server']['max_cudagraph_capture_size'] == 56
    now, wall, games, _, _ = build()
    sched = Scheduler(games, wall, budget_policy='fixed-2x', backlog_policy='readmissions',
                      backlog_key='progress-fruitless', level0_order='tokens', active_limit=46)
    for g in games:
        g.state = 'backlog'
        g.observer = sched
    sched.start_all()
    charge(now, games[0], 12345)
    games[0].report_level(1, wall.elapsed())
    assert sched.grant_for(games[0]) == 2 * 2 * 12345


def test_source_manifest_excludes_generated_build_receipt(tmp_path):
    import runpy
    from pathlib import Path
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/build_arena.py"))
    (tmp_path / "source.py").write_text("x = 1")
    (tmp_path / "build-manifest.json").write_text('{}')
    before = module['manifest'](tmp_path)
    (tmp_path / "build-manifest.json").write_text('{"new_receipt": true}')
    assert module['manifest'](tmp_path) == before
    assert list(before) == ['source.py']


def test_runb93_changes_only_capacity_from_runb64():
    old, new = components('runb64-plans'), components('runb93-plans')
    assert {k for k in new if old.get(k) != new[k]} == {'window', 'focus_slots', 'graph_capture'}
    cfg = compose(hero(), new, served_blocks=1260)
    assert cfg['windows'] == [93184] and cfg['active_limit'] == 40
    assert cfg['server']['max_num_seqs'] == 44
    assert cfg['server']['max_cudagraph_capture_size'] == 48
    assert cfg['level_budget_policy'] == 'fixed-2x'
    assert cfg.get('level_token_allowance') is None and cfg.get('blocks') is None
    assert cfg['backlog_key'] == 'progress-fruitless' and cfg['action_plans']


def test_prune25_40_changes_only_pruning_streams_window_and_partition():
    old, new = components('blocks3-plans'), components('blocks3-plans-prune25-40')
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {
        'expert_prune', 'window', 'density_blocks', 'focus_slots', 'blocks'}
    assert {k for k in set(old['blocks']) | set(new['blocks'])
            if old['blocks'].get(k) != new['blocks'].get(k)} == {'slots', 'partition'}
    assert new['window'] == 123904 and new['window'] % 1024 == 0 and new['window'] <= 130816
    assert new['expert_prune'] == components('blocks2-prune25')['expert_prune'] == 'prune25-layer'
    assert new['density_blocks'] == components('blocks2-prune25')['density_blocks'] == 1630
    assert new['action_plans'] and new['gate_margin'] == .96 and new['blocks']['stagger_seconds'] == 0
    assert new['blocks']['level_token_allowance'] == 90000 and new['blocks']['wall_seconds'] == 32400
    cfg = compose(hero(), new, served_blocks=1755)
    assert cfg['windows'] == [123904] and cfg['active_limit'] == 40
    assert cfg['server']['max_num_seqs'] == 44 and cfg['server']['max_cudagraph_capture_size'] == 48
    assert cfg['server']['expert_prune']['mask'].endswith('prune25-layer.json')
    assert cfg['blocks']['partition'] == 'streams' and cfg['blocks']['slots'] == 40
    assert cfg['action_plans'] and cfg['kv_margin'] == .96
    # Unchanged arms record no partition key (their studies stay byte-identical).
    assert 'partition' not in compose(hero(), old, served_blocks=1335)['blocks']


@pytest.mark.parametrize("blocks,streams", [(1669, 40), (1647, 40), (1631, 40), (1630, 40), (1629, 39), (1600, 39)])
def test_prune25_40_density_keeps_40_streams_on_the_validated_floor_only(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit

    cfg = compose(hero(), components('blocks3-plans-prune25-40'), served_blocks=1755)
    over = cfg['focus_oversubscription']
    assert over['gate_blocks_per_stream'] == 39.1  # floor(.96 x 1,630) = 1,564 gate blocks / 40
    fit = {'active_limit': 40, 'focus_oversubscription': over}
    served = NS(capacity=int(.96 * blocks), cost=lambda tokens: 49, layout={'max_model_len': 262144, 'lookahead': 0})
    competition_focus_fit(served, 123904, fit)
    # Fewer than 40 streams makes the blocks trial refuse before gameplay (a block is the stream count).
    assert fit['active_limit'] == streams


def test_prune25_40_stream_sized_blocks_in_roster_order():
    from arena.blocks import settings, check_block1, home_blocks

    cfg = settings(components('blocks3-plans-prune25-40')['blocks'])
    assert cfg['block_sizes'] == [40, 40, 30] and cfg['block_seconds'] == 10800
    assert home_blocks(110, cfg) == [0] * 40 + [1] * 40 + [2] * 30
    first = check_block1([f'g{i % 20}' for i in range(110)], cfg)
    assert first['distinct_sources'] == 20 and len(first['second_clones']) == 20
    assert first['sources'] == [f'g{i % 20}' for i in range(40)]
    with pytest.raises(ValueError):
        settings({**components('blocks3-plans-prune25-40')['blocks'], 'slots': 36})
    now, wall, games, sched, _ = build(count=110, wall_seconds=300, blocks=3, slots=40, roster_games=110,
                                      partition='streams', level_token_allowance=90000)
    assert [g.game_id for g in games if g.state == 'active'] == [f'g{i}' for i in range(40)]
    at(now, sched, 100)
    assert [g.game_id for g in games if g.state == 'active'] == [f'g{i}' for i in range(40, 80)]
    at(now, sched, 200)
    # The 30-game last block leaves ten slots: they go to parked old games (old_refill).
    assert all(g.state == 'active' for g in games[80:])
    assert sum(g.state == 'active' for g in games) == 40
    assert sum(g.state == 'active' for g in games[:80]) == 10
    assert audit_scheduler(report(games, sched, wall))['valid']


def test_prune37_changes_only_pruning_window_and_floor_from_prune25():
    old, new = components('blocks2-prune25'), components('blocks2-prune37')
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {'expert_prune', 'window', 'density_blocks'}
    assert new['expert_prune'] == 'prune37-layer' and new['density_blocks'] == 2008
    assert new['window'] == 106496 and new['window'] % 1024 == 0 and new['window'] <= 131072
    assert new['completion_cap'] == 12288 and new['gate_margin'] == .96 and not new.get('action_plans')
    assert new['blocks'] == old['blocks'] and new['focus_slots'] == 55
    cfg = compose(hero(), new, served_blocks=2132)
    assert cfg['windows'] == [106496] and cfg['active_limit'] == 55
    assert cfg['server']['max_num_seqs'] == 59 and cfg['server']['max_cudagraph_capture_size'] == 64
    assert cfg['server']['expert_prune']['mask'] == 'configs/expert-prune/prune37-layer.json'
    note = cfg['hero_v2']['components']['expert_prune']
    assert note['level'] == 'prune37-layer' and note['fraction'] == .375


@pytest.mark.parametrize("blocks,streams", [(2050, 55), (2028, 55), (2009, 55), (2008, 55), (2007, 54), (1900, 52)])
def test_prune37_density_keeps_55_streams_on_the_floor_only(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit

    cfg = compose(hero(), components('blocks2-prune37'), served_blocks=2132)
    over = cfg['focus_oversubscription']
    assert over['gate_blocks_per_stream'] == 35.036  # floor(.96 x 2,008) = 1,927 gate blocks / 55
    fit = {'active_limit': 55, 'focus_oversubscription': over}
    served = NS(capacity=int(.96 * blocks), cost=lambda tokens: 43, layout={'max_model_len': 262144, 'lookahead': 0})
    competition_focus_fit(served, 106496, fit)
    # Fewer than 55 streams makes the blocks trial refuse before gameplay (a block is the stream count).
    assert fit['active_limit'] == streams


def test_prune37_build_projects_the_estimated_pool():
    import runpy
    from pathlib import Path
    from arena.hero_v2 import block_bytes

    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/arena/build_campaign.py"))
    assert module['served_blocks']('blocks2-prune37') == 2132
    assert module['served_blocks']('blocks2-prune25') == module['served_blocks']('blocks3-plans-prune25-40') == 1755
    assert module['served_blocks']('blocks2') == 1335
    estimate = 1647 + round(3072 * 2764800 / block_bytes('int4'))
    assert 2132 * block_bytes('int8') // block_bytes('int4') == 2028 == estimate


def test_blocks3_samples2_changes_only_streams_sampler_and_partition():
    old, new = components('blocks3-plans'), components('blocks3-samples2')
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {
        'focus_slots', 'density_blocks', 'parallel_samples', 'blocks'}
    assert {k for k in set(old['blocks']) | set(new['blocks'])
            if old['blocks'].get(k) != new['blocks'].get(k)} == {'slots', 'partition'}
    assert new['window'] == 93184 and new['gate_margin'] == .96 and new['prefix_match_unit'] == 16
    assert 'summary_headroom' in new['harness_fixes'] and new['action_plans'] and new['blocks']['stagger_seconds'] == 0
    cfg = compose(hero(), new, served_blocks=1335)
    # 38 regular streams on 42 regular sequences; the server runs 48 (graphs to 48), six sibling-only.
    assert cfg['active_limit'] == 38 and cfg['generation_slots'] == 42
    assert cfg['server']['max_num_seqs'] == 48 and cfg['server']['max_cudagraph_capture_size'] == 48
    spec = cfg['parallel_samples']
    assert spec['k'] == 2 and spec['min_levels'] == 2 and spec['max_num_seqs'] == 48
    assert spec['prefix_warmup'] is False and spec['late_admission_seconds'] == 0 and spec['observe_every'] == 0
    record = cfg['hero_v2']['components']['parallel_samples']
    assert record['generation_slots'] == 42 and record['background_slots'] == 6
    assert cfg['level_budget_policy'] == 'blocks' and cfg['blocks']['partition'] == 'streams'
    assert cfg['blocks']['slots'] == 38 and cfg['blocks']['level_token_allowance'] == 90000
    assert cfg['kv_margin'] == .96 and cfg['server']['prefix_match_unit'] == 16 and cfg['action_plans']


def test_blocks3_samples2_stream_sized_blocks():
    from arena.blocks import settings, check_block1

    cfg = settings(components('blocks3-samples2')['blocks'])
    assert cfg['block_sizes'] == [38, 38, 34] and cfg['block_seconds'] == 10800
    first = check_block1([f'g{i % 20}' for i in range(110)], cfg)
    assert first['distinct_sources'] == 20 and len(first['second_clones']) == 18


@pytest.mark.parametrize("blocks,streams", [(1270, 38), (1269, 38), (1268, 38), (1267, 37), (1260, 37)])
def test_blocks3_samples2_density_keeps_38_streams_on_the_validated_floor_only(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit

    cfg = compose(hero(), components('blocks3-samples2'), served_blocks=1335)
    over = cfg['focus_oversubscription']
    assert over['gate_blocks_per_stream'] == 32.026  # floor(.96 x 1,268) = 1,217 gate blocks / 38
    fit = {'active_limit': 38, 'focus_oversubscription': over}
    served = NS(capacity=int(.96 * blocks), cost=lambda tokens: 39, layout={'max_model_len': 262144, 'lookahead': 0})
    competition_focus_fit(served, 93184, fit)
    # Fewer than 38 streams makes the blocks trial refuse before gameplay (a block is the stream count).
    assert fit['active_limit'] == streams


def test_samples_need_a_plain_gate_policy_and_the_prefix_cache():
    with pytest.raises(ValueError, match="plain gate"):
        compose(hero(), {**components('b31'), 'budget_policy': 'rank', 'backlog_fix': False,
                         'parallel_samples': {'k': 2}}, served_blocks=1335)
    with pytest.raises(ValueError, match="prefix_cache"):
        compose(hero(), {**components('blocks3-samples2'), 'prefix_cache': False}, served_blocks=1335)


# The submitted competition study's components (dist/arena-flash-competition-hero-v2-runb-v1, hero_v2.requested).
COMPETITION = {"token_budgets": True, "serving_tweaks": True, "kernels_phase6": False, "kv8_prefill": False,
               "expert_prune": "prune12", "gptq_arc": True, "topk_sampler": True, "stall_dump": 90,
               "prefix_cache": True, "request_end_tail": False, "keep_alive": False, "live_progress": True,
               "compaction_thinking": "prefilled-empty", "reasoning_history": "verbatim", "kv_cache": "int4",
               "window": 130816, "budget_policy": "fixed-2x", "focus_slots": 31, "focus_oversubscription": True,
               "prompt_tokens_details": True, "prefix_match_unit": None, "backlog_fix": True, "harness_fixes": True,
               "spare_sequences": 4, "density_blocks": 1268}


def test_comp_prune25_is_the_competition_plus_pruning_cache_changes_and_streams():
    from arena.hero_v2 import normalize
    old, new = normalize(dict(COMPETITION)), normalize(components("comp-prune25"))
    changed = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
    assert changed == {"expert_prune": ("prune12", "prune25-layer"), "focus_slots": (31, 43),
                       "density_blocks": (1268, 1630), "gate_margin": (0.9, 0.96), "prefix_match_unit": (None, 16),
                       "harness_fixes": (True, FIXES + ["summary_headroom"])}
    cfg = compose(hero(), components("comp-prune25"), served_blocks=1755)
    assert cfg["windows"] == [130816] and cfg["active_limit"] == 43 and cfg["kv_margin"] == 0.96
    assert cfg["server"]["max_num_seqs"] == cfg["generation_slots"] == 47
    assert cfg["server"]["max_cudagraph_capture_size"] == 48
    assert cfg["server"]["expert_prune"]["mask"].endswith("prune25-layer.json")
    assert cfg["level_budget_policy"] == "fixed-2x" and cfg["budget_clock"] == "tokens"
    assert cfg["backlog_key"] == "progress-fruitless" and cfg["level0_order"] == "tokens"
    assert cfg["opening_switch_level1_games"] == 6 and cfg["completion_cap"] == 16384
    assert not cfg.get("action_plans") and not cfg.get("blocks") and cfg.get("level_token_allowance") is None
    assert not cfg.get("parallel_samples")


@pytest.mark.parametrize("blocks,streams", [(1669, 43), (1647, 43), (1630, 43), (1629, 42), (1560, 41),
                                            (1500, 39)])
def test_comp_prune25_startup_density_reduces_streams_below_the_validated_floor(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit
    cfg = compose(hero(), components("comp-prune25"), served_blocks=1755)
    over = cfg["focus_oversubscription"]
    assert over["gate_blocks_per_stream"] == 36.372  # floor(.96 x 1,630) = 1,564 gate blocks / 43, rounded down
    study = {"active_limit": 43, "focus_oversubscription": over}
    budget = NS(capacity=int(.96 * blocks), cost=lambda tokens: 51,
                layout={"max_model_len": 262144, "lookahead": 0})
    competition_focus_fit(budget, 130816, study)
    assert study["active_limit"] == streams


MTP_ARM = "blocks3-plans-prune25-40-mtp1"


def test_mtp1_arm_changes_only_the_draft_window_floor_and_graphs_from_prune25_40():
    old, new = components("blocks3-plans-prune25-40"), components(MTP_ARM)
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {
        "mtp_tokens", "window", "density_blocks", "graph_capture"}
    assert new["mtp_tokens"] == 1 and new["density_blocks"] == 1420 and new["graph_capture"] == 88
    assert new["window"] == 84992 and new["window"] % 1024 == 0
    assert new["blocks"] == old["blocks"] and new["focus_slots"] == 40
    base = compose(hero(), old, served_blocks=served_blocks("blocks3-plans-prune25-40"))
    cfg = compose(hero(), new, served_blocks=served_blocks(MTP_ARM))
    server = cfg["server"]
    assert server["mtp_tokens"] == 1 and base["server"]["mtp_tokens"] == 0
    assert server["max_num_seqs"] == cfg["generation_slots"] == 44 and cfg["active_limit"] == 40
    assert server["max_cudagraph_capture_size"] == 88
    assert server["cudagraph_capture_sizes"] == [2, 4, 8, 16, 32, 48, 60, 64, 72, 80, 88]
    kernels = {k: v for k, v in server["serving_kernels"].items() if base["server"]["serving_kernels"].get(k) != v}
    assert kernels == {"mamba_ssm_cache_dtype": "bfloat16", "draft_experts": "nvfp4", "draft_vocab": "arena16k"}
    # Only the MTP keys differ in the server block.
    assert {k for k in set(server) | set(base["server"]) if server.get(k) != base["server"].get(k)} == {
        "mtp_tokens", "cudagraph_capture_sizes", "max_cudagraph_capture_size", "serving_kernels"}
    assert cfg["cache_experiment"] == base["cache_experiment"] and cfg["windows"] == [84992]
    layout = cfg["hero_v2"]["layout"]
    assert layout["blocks"] == 1438 and layout["fixed_blocks"] == 13
    assert layout["blocks_per_window"] == 28 + 13  # ceil(86,016 / 3,136) + 13 (the pilot's cost table: 41)
    assert cfg["hero_v2"]["components"]["mtp_tokens"]["on"]
    # Earlier arms record no MTP component (their studies stay byte-identical).
    assert "mtp_tokens" not in base["hero_v2"]["requested"] and "cudagraph_capture_sizes" not in base["server"]


def test_mtp1_full_window_cost_matches_the_measured_pilot_layout():
    import json
    from pathlib import Path
    from arena.cache import BlockBudget
    from arena.hero_v2 import cost_blocks

    path = Path(__file__).resolve().parents[1] / "docs/validation-20260930/prune25-40-mtp1-capacity.json"
    layout = json.loads(path.read_text())["layout_table"]
    assert layout["num_blocks"] == 1438 and layout["lookahead"] == 1
    budget = BlockBudget(layout, .96)
    for window in (65536, 84992, 86016, 93184, 123904):
        assert budget.cost(window + 256) == cost_blocks("int4", window, 1)
    assert budget.cost(1) == 14  # 1 attention block + 13 fixed


@pytest.mark.parametrize("blocks,streams", [(1438, 40), (1421, 40), (1420, 40), (1419, 39), (1400, 39)])
def test_mtp1_density_keeps_40_streams_on_the_1420_floor_only(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit

    cfg = compose(hero(), components(MTP_ARM), served_blocks=served_blocks(MTP_ARM))
    over = cfg["focus_oversubscription"]
    assert over["gate_blocks_per_stream"] == 34.075  # floor(.96 x 1,420) = 1,363 gate blocks / 40
    fit = {"active_limit": 40, "focus_oversubscription": over}
    served = NS(capacity=int(.96 * blocks), cost=lambda tokens: 41, layout={"max_model_len": 262144, "lookahead": 1})
    competition_focus_fit(served, 84992, fit)
    # Fewer than 40 streams makes the blocks trial refuse before gameplay (a block is the stream count).
    assert fit["active_limit"] == streams


def _campaign_study(arm):
    cfg = compose(hero(), components(arm), served_blocks=served_blocks(arm))
    cfg.update(campaign={"arm": arm}, wall_seconds=32400, early_stop=None,
               competition={"preview_instances": 110, "research_clones": True},
               trials=[{"kind": "duck-transfer"}])
    return cfg


def test_campaign_validate_allows_mtp1_for_the_mtp_arm_only():
    from arena.campaign import validate

    assert validate(_campaign_study(MTP_ARM))
    assert validate(_campaign_study("blocks3-plans-prune25-40"))
    other = _campaign_study("blocks3-plans-prune25-40")
    other["server"]["mtp_tokens"] = 1
    with pytest.raises(ValueError, match="MTP0"):
        validate(other)
    mtp0 = _campaign_study(MTP_ARM)
    mtp0["server"]["mtp_tokens"] = 0
    with pytest.raises(ValueError, match="MTP0"):
        validate(mtp0)
    for key, value in (("mamba_ssm_cache_dtype", None), ("draft_vocab", None)):
        broken = _campaign_study(MTP_ARM)
        broken["server"]["serving_kernels"][key] = value
        with pytest.raises(ValueError, match="draft stack"):
            validate(broken)
    graphs = _campaign_study(MTP_ARM)
    graphs["server"]["cudagraph_capture_sizes"] = [1, 2, 4, 8, 16, 24, 32, 40, 44]
    with pytest.raises(ValueError, match="draft stack"):
        validate(graphs)


def test_mtp_component_refusals():
    with pytest.raises(ValueError, match="graph_capture"):
        compose(hero(), {**components(MTP_ARM), "graph_capture": 48}, served_blocks=1638)
    with pytest.raises(ValueError, match="graph_capture"):
        compose(hero(), {**components(MTP_ARM), "graph_capture": 96}, served_blocks=1638)
    with pytest.raises(ValueError, match="mtp_tokens"):
        compose(hero(), {**components(MTP_ARM), "mtp_tokens": 2}, served_blocks=1638)
    with pytest.raises(ValueError, match="int4"):
        compose(hero(), {**components(MTP_ARM), "kv_cache": "int8", "prefix_match_unit": 16}, served_blocks=1638)


INT3_ARM = "blocks3-plans-prune25-40-mtp1-int3"


def test_int3_arm_changes_only_the_experts_window_and_floor_from_the_mtp1_arm():
    from arena.variants import EXPERT_OVERLAY_INT3

    old, new = components(MTP_ARM), components(INT3_ARM)
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {"expert_int3", "window", "density_blocks"}
    assert new["expert_int3"] == 16 and new["density_blocks"] == 1860
    assert new["window"] == 131072 and new["window"] % 1024 == 0
    assert new["focus_slots"] == 40 and new["blocks"] == old["blocks"] and new["mtp_tokens"] == 1
    base = compose(hero(), old, served_blocks=served_blocks(MTP_ARM))
    cfg = compose(hero(), new, served_blocks=served_blocks(INT3_ARM))
    server, before = cfg["server"], base["server"]
    # The server block differs only in the routed experts: the overlay and the MoE backend + repack.
    assert {k for k in set(server) | set(before) if server.get(k) != before.get(k)} == {
        "expert_overlay", "serving_kernels"}
    kernels = {k: v for k, v in server["serving_kernels"].items() if before["serving_kernels"].get(k) != v}
    assert kernels == {"moe_backend": "humming", "expert_int3": 16}
    assert before["serving_kernels"]["moe_backend"] == "marlin"
    assert before["expert_overlay"]["dataset"] == "richardcsaky/arc-expert-gptq-arc-v1"
    assert server["expert_overlay"] == EXPERT_OVERLAY_INT3[16]
    assert server["expert_overlay"]["dataset"] == "richardcsaky/arc-expert-int3-g16-v1"
    assert server["expert_overlay"]["check"] == "size"
    from arena.expert_overlay import validate as validate_overlay

    assert validate_overlay(server["expert_overlay"]) == server["expert_overlay"]  # a pinned sha256, not a placeholder
    # The MTP draft keeps its NvFp4 b12x experts, the same pruning mask, graphs and 44 sequences.
    assert server["serving_kernels"]["draft_experts"] == "nvfp4"
    assert server["expert_prune"] == before["expert_prune"]
    assert server["cudagraph_capture_sizes"] == before["cudagraph_capture_sizes"]
    assert server["max_num_seqs"] == 44 and cfg["active_limit"] == 40 and cfg["windows"] == [131072]
    assert cfg["cache_experiment"] == base["cache_experiment"]
    layout = cfg["hero_v2"]["layout"]
    assert layout["blocks"] == 1907 and layout["fixed_blocks"] == 13 and layout["density_blocks"] == 1860
    assert cfg["hero_v2"]["components"]["expert_int3"]["on"]
    # Earlier arms record no int3 component (their studies stay byte-identical).
    assert "expert_int3" not in base["hero_v2"]["requested"] and "expert_int3" not in before["serving_kernels"]


def test_int3_estimated_pool_is_the_mtp1_pool_plus_the_saved_expert_bytes():
    from arena.hero_v2 import block_bytes

    served_experts = 48 * 512 * 3 // 4  # prune25-layer
    per_expert = (2 * 640 * 2560 + 2560 * 640)  # gate+up and down weights
    saved = served_experts * per_expert * (4.5 - 3.5) / 8  # Marlin NVFP4 4.5 -> Humming uint3 g16 3.5 bits
    assert saved == 11324620800
    assert round(saved / block_bytes("int4", 1)) == 469 and round(saved / block_bytes("int8")) == 534
    assert served_blocks(INT3_ARM) == served_blocks(MTP_ARM) + 534
    assert served_blocks(INT3_ARM) * block_bytes("int8") // block_bytes("int4", 1) == 1438 + 469


@pytest.mark.parametrize("blocks,streams", [(1907, 40), (1861, 40), (1860, 40), (1859, 39), (1800, 38)])
def test_int3_density_keeps_40_streams_on_the_1860_floor_only(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit
    from arena.hero_v2 import cost_blocks

    cfg = compose(hero(), components(INT3_ARM), served_blocks=served_blocks(INT3_ARM))
    over = cfg["focus_oversubscription"]
    assert over["gate_blocks_per_stream"] == 44.625  # floor(.96 x 1,860) = 1,785 gate blocks / 40
    fit = {"active_limit": 40, "focus_oversubscription": over}
    cost = cost_blocks("int4", 131072, 1)
    served = NS(capacity=int(.96 * blocks), cost=lambda tokens: cost, layout={"max_model_len": 262144, "lookahead": 1})
    competition_focus_fit(served, 131072, fit)
    assert fit["active_limit"] == streams


def test_int3_window_is_the_largest_passing_replay_window():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "docs/validation-20260930/prune25-40-mtp1-int3-capacity.json"
    data = json.loads(path.read_text())
    assert data["pool_floor_blocks"] == 1860 and data["streams"] == 40 and data["margin"] == 0.96
    windows = {int(w): v for w, v in data["windows"].items()}
    assert windows[131072]["passes"] and not windows[132096]["passes"]
    assert all(v["passes"] for w, v in windows.items() if w <= 131072)
    assert data["chosen_window"] == components(INT3_ARM)["window"] == 131072


def test_campaign_validate_allows_int3_for_the_int3_arm_only():
    from arena.campaign import validate

    assert validate(_campaign_study(INT3_ARM))
    assert validate(_campaign_study(MTP_ARM))
    for key, value in (("moe_backend", "marlin"), ("expert_int3", None)):
        broken = _campaign_study(INT3_ARM)
        broken["server"]["serving_kernels"][key] = value
        with pytest.raises(ValueError, match="int3"):
            validate(broken)
    overlay = _campaign_study(INT3_ARM)
    overlay["server"]["expert_overlay"] = {**overlay["server"]["expert_overlay"], "check": "sha256"}
    with pytest.raises(ValueError, match="int3"):
        validate(overlay)
    other = _campaign_study(MTP_ARM)
    other["server"]["serving_kernels"].update(moe_backend="humming", expert_int3=16)
    with pytest.raises(ValueError, match="int3 arm only"):
        validate(other)


def test_int3_component_refusals():
    with pytest.raises(ValueError, match="expert_int3"):
        compose(hero(), {**components(INT3_ARM), "expert_int3": 32}, served_blocks=2172)
    with pytest.raises(ValueError, match="gptq_arc"):
        compose(hero(), {**components(INT3_ARM), "gptq_arc": False}, served_blocks=2172)


def test_int3_serving_kernels_patch_humming_utils_and_environment():
    from arc_infra import serving_kernels_patch as k
    from arc_infra.expert_int3_patch import TARGET

    cfg = compose(hero(), components(INT3_ARM), served_blocks=served_blocks(INT3_ARM))
    spec = k.validate(cfg["server"]["serving_kernels"])
    assert spec == cfg["server"]["serving_kernels"]
    assert TARGET["relative"] in [t["relative"] for t in k.targets(spec)]
    assert k.server_flags(spec, 1)[:2] == ["--moe-backend", "humming"]
    assert k.server_environment(spec, {"VLLM_PLE_CPU_OFFLOAD": "1"})["ARC_EXPERT_INT3"] == '{"group": 16}'
    with pytest.raises(ValueError, match="humming"):
        k.validate({**spec, "moe_backend": "marlin"})


def test_mtp1_records_the_paired_controls_depth_before():
    cfg = compose(hero(), components(MTP_ARM), served_blocks=served_blocks(MTP_ARM))
    assert cfg["arena_variant"]["changed"]["server.mtp_tokens"] == {"before": 3, "after": 1}
    assert cfg["arena_variant"]["reference_difference"]["server.mtp_tokens"] == {"before": 0, "after": 1}


INT3_MTP0_ARM = "blocks3-plans-prune25-40-int3"


def test_int3_mtp0_arm_changes_only_the_experts_window_and_floor_from_prune25_40():
    from arena.variants import EXPERT_OVERLAY_INT3

    old, new = components("blocks3-plans-prune25-40"), components(INT3_MTP0_ARM)
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {"expert_int3", "window", "density_blocks"}
    assert new["expert_int3"] == 16 and new["density_blocks"] == 2100
    assert new["window"] == 172032 and new["window"] % 1024 == 0 and new["window"] <= 261120
    assert "mtp_tokens" not in new and new["focus_slots"] == 40 and new["blocks"] == old["blocks"]
    base = compose(hero(), old, served_blocks=served_blocks("blocks3-plans-prune25-40"))
    cfg = compose(hero(), new, served_blocks=served_blocks(INT3_MTP0_ARM))
    server, before = cfg["server"], base["server"]
    assert {k for k in set(server) | set(before) if server.get(k) != before.get(k)} == {
        "expert_overlay", "serving_kernels"}
    kernels = {k: v for k, v in server["serving_kernels"].items() if before["serving_kernels"].get(k) != v}
    assert kernels == {"moe_backend": "humming", "expert_int3": 16}
    assert server["expert_overlay"] == EXPERT_OVERLAY_INT3[16] and server["mtp_tokens"] == 0
    assert server["max_num_seqs"] == 44 and server["max_cudagraph_capture_size"] == 48 and cfg["active_limit"] == 40
    assert cfg["windows"] == [172032] and cfg["cache_experiment"] == base["cache_experiment"]
    layout = cfg["hero_v2"]["layout"]
    assert layout["blocks"] == 2177 and layout["density_blocks"] == 2100 and "fixed_blocks" not in layout


def test_int3_mtp0_estimated_pool_and_floor():
    from arena.hero_v2 import block_bytes

    saved = 18432 * 4915200 // 8
    assert block_bytes("int4") == 22278144 and round(1647 + saved / block_bytes("int4")) == 2155
    assert served_blocks(INT3_MTP0_ARM) == 1755 + 534


@pytest.mark.parametrize("blocks,streams", [(2177, 40), (2101, 40), (2100, 40), (2099, 39), (2050, 39)])
def test_int3_mtp0_density_keeps_40_streams_on_the_2100_floor_only(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit
    from arena.hero_v2 import cost_blocks

    cfg = compose(hero(), components(INT3_MTP0_ARM), served_blocks=served_blocks(INT3_MTP0_ARM))
    over = cfg["focus_oversubscription"]
    assert over["gate_blocks_per_stream"] == 50.4  # floor(.96 x 2,100) = 2,016 gate blocks / 40
    fit = {"active_limit": 40, "focus_oversubscription": over}
    cost = cost_blocks("int4", 172032)
    served = NS(capacity=int(.96 * blocks), cost=lambda tokens: cost, layout={"max_model_len": 262144, "lookahead": 0})
    competition_focus_fit(served, 172032, fit)
    assert fit["active_limit"] == streams


def test_int3_mtp0_window_is_the_largest_passing_replay_window():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "docs/validation-20260930/prune25-40-int3-capacity.json"
    data = json.loads(path.read_text())
    assert data["pool_floor_blocks"] == 2100 and data["streams"] == 40 and data["margin"] == 0.96
    windows = {int(w): v for w, v in data["windows"].items()}
    assert windows[172032]["passes"] and not windows[173056]["passes"]
    assert all(v["passes"] for w, v in windows.items() if w <= 172032)
    assert data["chosen_window"] == components(INT3_MTP0_ARM)["window"]


def test_campaign_validate_accepts_the_int3_mtp0_arm_mtp0_only():
    from arena.campaign import validate

    assert validate(_campaign_study(INT3_MTP0_ARM))
    mtp = _campaign_study(INT3_MTP0_ARM)
    mtp["server"]["mtp_tokens"] = 1
    with pytest.raises(ValueError, match="MTP0"):
        validate(mtp)
    broken = _campaign_study(INT3_MTP0_ARM)
    broken["server"]["serving_kernels"]["moe_backend"] = "marlin"
    with pytest.raises(ValueError, match="int3"):
        validate(broken)
    other = _campaign_study("blocks3-plans-prune25-40")
    other["server"]["serving_kernels"].update(moe_backend="humming", expert_int3=16)
    with pytest.raises(ValueError, match="int3 arm only"):
        validate(other)


def test_prune50_int3_changes_only_pruning_int3_window_streams_and_floor_from_blocks2_plans_prune37():
    from arena.blocks import settings
    from arena.variants import EXPERT_OVERLAY_INT3

    old, new = components("blocks2-plans-prune37"), components(PRUNE50_ARM)
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {
        "expert_prune", "expert_int3", "window", "density_blocks", "focus_slots", "graph_capture", "blocks"}
    assert {k for k in set(old["blocks"]) | set(new["blocks"])
            if old["blocks"].get(k) != new["blocks"].get(k)} == {"slots", "partition"}
    assert new["expert_prune"] == "prune50-layer" and new["expert_int3"] == 16 and new["density_blocks"] == 2658
    assert new["window"] == 98304 and new["focus_slots"] == new["blocks"]["slots"] == 76 and new["graph_capture"] == 80
    assert new["action_plans"] and "completion_cap" not in new and new["gate_margin"] == .96
    assert new["prefix_match_unit"] == 16 and "summary_headroom" in new["harness_fixes"] and "mtp_tokens" not in new
    assert new["blocks"]["blocks"] == 2 and new["blocks"]["stagger_seconds"] == 0
    assert new["blocks"]["level_token_allowance"] == 90000 and new["kv_cache"] == "int4"
    base = compose(hero(), old, served_blocks=served_blocks("blocks2-plans-prune37"))
    cfg = compose(hero(), new, served_blocks=served_blocks(PRUNE50_ARM))
    server, before = cfg["server"], base["server"]
    assert {k for k in set(server) | set(before) if server.get(k) != before.get(k)} == {
        "expert_overlay", "serving_kernels", "expert_prune", "max_num_seqs", "max_cudagraph_capture_size"}
    kernels = {k: v for k, v in server["serving_kernels"].items() if before["serving_kernels"].get(k) != v}
    assert kernels == {"moe_backend": "humming", "expert_int3": 16}
    assert server["expert_overlay"] == EXPERT_OVERLAY_INT3[16] and server["mtp_tokens"] == 0
    assert server["expert_prune"]["mask"] == "configs/expert-prune/prune50-layer.json"
    assert server["max_num_seqs"] == cfg["generation_slots"] == 80 and server["max_cudagraph_capture_size"] == 80
    assert cfg["windows"] == [98304] and cfg["active_limit"] == 76 and cfg["completion_cap"] == 16384
    assert cfg["cache_experiment"] == base["cache_experiment"] and cfg["kv_margin"] == .96
    note = cfg["hero_v2"]["components"]["expert_prune"]
    assert note["level"] == "prune50-layer" and note["fraction"] == .5
    assert cfg["hero_v2"]["layout"]["density_blocks"] == 2658 and cfg["hero_v2"]["layout"]["blocks"] == 2770
    blocks = settings(cfg["blocks"])
    assert blocks["block_sizes"] == [76, 34] and blocks["block_seconds"] == 16200


def test_prune50_int3_estimated_pool_and_floor():
    from arena.hero_v2 import block_bytes

    pruned = 6144 * 2764800 / block_bytes("int4")
    int3 = 12288 * 4915200 // 8 / block_bytes("int4")
    assert round(pruned, 1) == 762.5 and round(int3, 1) == 338.9 and round(1647 + pruned + int3) == 2748
    assert components(PRUNE50_ARM)["density_blocks"] == 2748 - 90
    assert served_blocks(PRUNE50_ARM) == 1755 + round(6144 * 2764800 / block_bytes("int8")) + round(
        12288 * 614400 / block_bytes("int8"))


@pytest.mark.parametrize("blocks,streams", [(2770, 76), (2748, 76), (2659, 76), (2658, 76), (2657, 75), (2600, 74)])
def test_prune50_int3_density_keeps_76_streams_on_the_2658_floor_only(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit
    from arena.hero_v2 import cost_blocks

    cfg = compose(hero(), components(PRUNE50_ARM), served_blocks=served_blocks(PRUNE50_ARM))
    over = cfg["focus_oversubscription"]
    assert over["gate_blocks_per_stream"] == 33.565  # floor(.96 x 2,658) = 2,551 gate blocks / 76
    fit = {"active_limit": 76, "focus_oversubscription": over}
    cost = cost_blocks("int4", 98304)
    assert cost == 41
    served = NS(capacity=int(.96 * blocks), cost=lambda tokens: cost, layout={"max_model_len": 262144, "lookahead": 0})
    competition_focus_fit(served, 98304, fit)
    assert fit["active_limit"] == streams


def test_prune50_int3_streams_are_the_most_passing_replay_count():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "docs/validation-20260930/prune50-int3-capacity.json"
    data = json.loads(path.read_text())
    assert data["pool_floor_blocks"] == 2658 and data["window"] == 98304 and data["margin"] == 0.96
    assert data["output_cap"] == 16384 and data["max_num_seqs"] == 80
    counts = {int(n): v for n, v in data["streams"].items()}
    assert counts[76]["passes"] and not counts[78]["passes"] and counts[76]["replays"] == 18
    assert all(v["passes"] for n, v in counts.items() if n <= 76)
    assert data["chosen_streams"] == components(PRUNE50_ARM)["focus_slots"]


def test_campaign_validate_accepts_the_prune50_int3_arm():
    from arena.campaign import validate

    assert validate(_campaign_study(PRUNE50_ARM))
    broken = _campaign_study(PRUNE50_ARM)
    broken["server"]["serving_kernels"]["moe_backend"] = "marlin"
    with pytest.raises(ValueError, match="int3"):
        validate(broken)
    mtp = _campaign_study(PRUNE50_ARM)
    mtp["server"]["mtp_tokens"] = 1
    with pytest.raises(ValueError, match="MTP0"):
        validate(mtp)
    other = _campaign_study(PRUNE50_ARM)
    other["hero_v2"]["requested"]["expert_prune"] = "prune37-layer"
    with pytest.raises(ValueError, match="expert_prune"):
        validate(other)


def test_prune37_int3_is_the_prune50_int3_arm_with_prune37_and_the_kaggle_builds_streams_and_floor():
    from arena.blocks import settings

    old, new = components(PRUNE50_ARM), components(PRUNE37_INT3_ARM)
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {
        "expert_prune", "density_blocks", "focus_slots", "graph_capture", "blocks"}
    assert new["expert_prune"] == "prune37-layer" and new["expert_int3"] == 16 and new["window"] == 98304
    assert new["density_blocks"] == 2340 and new["focus_slots"] == new["blocks"]["slots"] == 60
    assert new["graph_capture"] == 64 and new["blocks"]["partition"] == "streams"
    cfg = compose(hero(), new, served_blocks=served_blocks(PRUNE37_INT3_ARM))
    # The Kaggle build arena-flash-competition-blocks2-p37-int3-v1: 60 streams, 64 sequences, graphs to 64, 2,340 floor,
    # int8 projection 2,577 -> 2,451 int4 blocks.
    assert cfg["active_limit"] == 60 and cfg["server"]["max_num_seqs"] == cfg["generation_slots"] == 64
    assert cfg["server"]["max_cudagraph_capture_size"] == 64 and cfg["windows"] == [98304]
    assert cfg["server"]["serving_kernels"]["moe_backend"] == "humming" and cfg["server"]["mtp_tokens"] == 0
    assert cfg["server"]["expert_prune"]["mask"] == "configs/expert-prune/prune37-layer.json"
    layout = cfg["hero_v2"]["layout"]
    assert layout["int8_blocks"] == 2577 and layout["blocks"] == 2451 and layout["density_blocks"] == 2340
    assert cfg["focus_oversubscription"]["gate_blocks_per_stream"] == 37.433  # floor(.96 x 2,340) = 2,246 / 60
    assert settings(cfg["blocks"])["block_sizes"] == [60, 50]
    assert cfg["completion_cap"] == 16384 and cfg["action_plans"] and cfg["kv_margin"] == .96


@pytest.mark.parametrize("blocks,streams", [(2451, 60), (2432, 60), (2341, 60), (2340, 60), (2339, 59), (2300, 58)])
def test_prune37_int3_density_keeps_60_streams_on_the_2340_floor_only(blocks, streams):
    from types import SimpleNamespace as NS
    from arena.capacity import competition_focus_fit

    cfg = compose(hero(), components(PRUNE37_INT3_ARM), served_blocks=served_blocks(PRUNE37_INT3_ARM))
    fit = {"active_limit": 60, "focus_oversubscription": cfg["focus_oversubscription"]}
    served = NS(capacity=int(.96 * blocks), cost=lambda tokens: 41, layout={"max_model_len": 262144, "lookahead": 0})
    competition_focus_fit(served, 98304, fit)
    assert fit["active_limit"] == streams


def test_prune37_int3_replay_passes_at_60_streams():
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).resolve().parents[1] / "docs/validation-20260930/prune37-int3-capacity.json")
                      .read_text())
    assert data["pool_floor_blocks"] == 2340 and data["window"] == 98304 and data["chosen_streams"] == 60
    counts = {int(n): v for n, v in data["streams"].items()}
    assert counts[60]["passes"] and counts[66]["passes"] and not counts[68]["passes"]
    assert round(2008 + 15360 * 614400 / 22278144) == data["pool_estimate_blocks"] == 2432


def test_campaign_validate_accepts_the_prune37_int3_arm():
    from arena.campaign import validate

    assert validate(_campaign_study(PRUNE37_INT3_ARM))
    broken = _campaign_study(PRUNE37_INT3_ARM)
    broken["server"]["serving_kernels"]["moe_backend"] = "marlin"
    with pytest.raises(ValueError, match="int3"):
        validate(broken)
