"""Competition build of run B at 32 focus streams ("please create a competition submission with run
B, use 32 streams as discussed before, include a 30-minute public preview on our 110-game arena roster")."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arena import hero_v2
from arena.capacity import competition_focus_fit
from arena.variants import apply_variant

ROOT = Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "configs/arena_30m.json").read_text())
SERVED = 1335  # hero-v2 smoke v2: the measured int8 pool (int4: 1,270 blocks)
RUN_B = {"kernels_phase6": False, "prefix_cache": True, "kv_cache": "int4", "focus_oversubscription": True,
         "prompt_tokens_details": True, "backlog_fix": True, "harness_fixes": True, "window": 130816,
         "focus_slots": 24}
COMPETITION_B = {**RUN_B, "focus_slots": 32, "spare_sequences": 4, "density_blocks": 1268}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hero():
    return apply_variant(CONTROL, hero_v2.BASE)


def budget(blocks, cost=51):
    """A served int4 pool of ``blocks`` (the gate is 0.9 x it); a full 130,816-token window costs 51 blocks."""
    return NS(capacity=int(0.9 * blocks), cost=lambda tokens: cost, layout={"max_model_len": 262144, "lookahead": 0})


def test_run_b_at_32_streams_has_four_spare_sequences_and_graphs_to_40():
    study = hero_v2.compose(hero(), {**COMPETITION_B, "token_budgets": False}, served_blocks=SERVED)
    assert study["active_limit"] == 32 and study["server"]["max_num_seqs"] == study["generation_slots"] == 36
    # vLLM captures decode graphs in steps of 8 above 8: max_num_seqs 36 runs in the 40 graph.
    assert study["server"]["max_cudagraph_capture_size"] == 40
    assert study["hero_v2"]["layout"]["windows_fit"] == 22 and study["focus_oversubscription"]["focus_slots"] == 32
    # Run B itself (defaults) is unchanged: two spares, the projected pool's density.
    run_b = hero_v2.compose(hero(), {**RUN_B, "token_budgets": False}, served_blocks=SERVED)
    assert run_b["server"]["max_num_seqs"] == 26 and run_b["focus_oversubscription"]["gate_blocks_per_stream"] == 47.625


@pytest.mark.parametrize("blocks,streams", [(1270, 32), (1269, 32), (1268, 32), (1267, 31), (1240, 31), (1209, 30),
                                            (1150, 29)])
def test_the_startup_density_runs_32_on_the_validated_pool_and_fewer_below(blocks, streams):
    study = hero_v2.compose(hero(), {**COMPETITION_B, "token_budgets": False}, served_blocks=SERVED)
    over = study["focus_oversubscription"]
    assert over["gate_blocks_per_stream"] == 35.656          # floor(0.9 x 1,268) = 1,141 gate blocks / 32, rounded down
    fit = {"active_limit": 32, "focus_oversubscription": over}
    result = competition_focus_fit(budget(blocks), 130816, fit)
    assert fit["active_limit"] == streams and result["full_window_requests"] == int(0.9 * blocks) // 51
    # Without density_blocks the projected 1,270-block pool would be the reference: one block less -> 31 streams.
    projected = hero_v2.compose(hero(), {**COMPETITION_B, "density_blocks": None, "token_budgets": False},
                                served_blocks=SERVED)["focus_oversubscription"]
    loose = {"active_limit": 32, "focus_oversubscription": projected}
    competition_focus_fit(budget(1269), 130816, loose)
    assert loose["active_limit"] == 31


def test_spare_and_density_are_validated():
    with pytest.raises(ValueError, match="spare_sequences"):
        hero_v2.normalize({**COMPETITION_B, "spare_sequences": 1})
    with pytest.raises(ValueError, match="explicit focus_slots"):
        hero_v2.normalize({"spare_sequences": 4})
    with pytest.raises(ValueError, match="density_blocks"):
        hero_v2.normalize({**COMPETITION_B, "focus_oversubscription": False, "focus_slots": 22})


def build_args(**extra):
    return dict(paired_30m=True, arena_variant=hero_v2.BASE, opening_burst_requests=3,
                exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], hero_v2_served_blocks=SERVED,
                competition=True, preview_seconds=1800, preview_instances=110, sanity_trial=False, **extra)


def test_the_competition_build_composes_hero_v2_on_the_gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)
    b = load("build_arena_competition_b", ROOT / "scripts/build_arena.py")
    _, study = b.build("arena-flash-competition-hero-v2-runb-test", output=tmp_path / "bundle",
                       hero_v2=dict(COMPETITION_B), **build_args())
    competition = study["competition"]
    assert competition["submission_seconds"] == 32400 and competition["cleanup_seconds"] == 600
    assert competition["preview_seconds"] == 1800 and competition["preview_instances"] == 110
    assert "research_clones" not in competition and study["wall_seconds"] == 1800
    assert [t["name"] for t in study["trials"]] == ["duck-transfer"] and study["kernel_timeout"] == 32400
    assert study["budget_clock"] == "tokens" and study["windows"] == [130816] and study["active_limit"] == 32
    assert study["server"]["expert_overlay"]["check"] == "size"
    # Kaggle loads the weights and the overlay in ~15 min: the 900 s Modal ready guard killed the first preview.
    assert all(t["server_env"]["ARENA_READY_GUARD"] == "180,3000" for t in study["trials"])
    assert study["model_datasets"] == [study["server"]["expert_overlay"]["dataset"]]
    # The oversubscribed focus has no fit fallback (the exact gate and the density rule replace it).
    assert "focus_fit_fallback" not in study and study["require_focus_windows_fit"] is False
    assert len(study["games"]) == 20
    # The research clone roster and the competition gateway are exclusive; a competition build is the submission only.
    with pytest.raises(ValueError, match="research roster|clone roster or the competition gateway"):
        b.build("x", output=tmp_path / "b2", hero_v2=dict(COMPETITION_B), **build_args(clone_instances=110))
    with pytest.raises(ValueError, match="submission only"):
        b.build("x", output=tmp_path / "b3", hero_v2=dict(COMPETITION_B),
                **build_args(early_stop={"at_seconds": 900, "score_above": 1.0}))
