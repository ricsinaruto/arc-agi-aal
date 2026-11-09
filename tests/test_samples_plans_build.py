"""The run-B + parallel_samples + action_plans build: earlier builds stay
byte-for-byte, the new bundle differs from run B only by the two features and the sequences they need, and it passes
its Modal guard (which refuses run B's own bundle)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from arena import hero_v2

ROOT = Path(__file__).resolve().parents[1]
SERVED = 1335
REASONS = {"kernels_phase6": "phase-6 GPU test: no speed gain (fused split-K saves nothing; UVA PLE already in "
                             "serving_tweaks)",
           "topk_sampler": "kept: the sampler behavior check found the top-k-first path never runs on gameplay "
                           "batches",
           "prefix_match_unit": "off: the pre-check warm/cold check was inconclusive (narrowly fails its bar, CIs "
                                "include 0); user rule: only if it passes"}
RUN_B = {"kernels_phase6": False, "prefix_cache": True, "kv_cache": "int4", "focus_oversubscription": True,
         "prompt_tokens_details": True, "backlog_fix": True, "harness_fixes": True, "window": 130816, "focus_slots": 24}
COMPETITION_B = {**RUN_B, "focus_slots": 32, "spare_sequences": 4, "density_blocks": 1268}
SAMPLES_PLANS = {**RUN_B, "parallel_samples": True, "action_plans": True, "density_blocks": 1266}
COMMON = dict(paired_30m=True, arena_variant=hero_v2.BASE, opening_burst_requests=3,
              exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], hero_v2_served_blocks=SERVED)
# configs/comparison.json of run B's command (build_ab.sh) and of the competition run-B build (test_competition_run_b),
# built on main without the two features (e2201d5, 30 September: the competition ready guard); a later change to
# either build on main must update these on purpose.
PINNED = {"run-b": "1527f674de44dda8d150265782c9a5a4beb0b432921f6919743b9b77f97245ad",
          "competition-run-b": "1ec8afbff0a81d16a342ce584c37a390bf1d80d4033db72750bac90f4dadc176"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(tmp_path, tag, name, components, **extra):
    b = load(f"build_arena_{tag}", ROOT / "scripts/build_arena.py")
    b.build(name, output=tmp_path / tag, hero_v2=components, **COMMON, **extra)
    return tmp_path / tag


@pytest.fixture
def token_clock(monkeypatch):
    monkeypatch.setattr(hero_v2, "_token_clock", lambda: True)


def test_run_b_and_the_competition_build_are_byte_for_byte_unchanged(tmp_path, token_clock):
    run_b = build(tmp_path, "b", "arena-hero-v2-128k-streams-v1", {**RUN_B, "reasons": REASONS},
                  wall_seconds=7200, clone_instances=110)
    competition = build(tmp_path, "c", "arena-flash-competition-hero-v2-runb-v1", dict(COMPETITION_B),
                        competition=True, preview_seconds=1800, preview_instances=110, sanity_trial=False)
    for tag, folder in (("run-b", run_b), ("competition-run-b", competition)):
        data = (folder / "configs/comparison.json").read_bytes()
        assert hashlib.sha256(data).hexdigest() == PINNED[tag], tag
        study = json.loads(data)
        assert "parallel_samples" not in study and "action_plans" not in study
        assert not {"parallel_samples", "action_plans"} & set(study["hero_v2"]["requested"])


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            out += diff(a.get(key, "<absent>"), b.get(key, "<absent>"), f"{path}.{key}")
        return out
    return [] if a == b else [path]


def test_the_feature_build_differs_from_run_b_only_by_the_features_and_passes_its_guard(tmp_path, token_clock):
    run_b = build(tmp_path, "b", "arena-hero-v2-128k-streams-v1", {**RUN_B, "reasons": REASONS},
                  wall_seconds=7200, clone_instances=110)
    feature = build(tmp_path, "f", "arena-hero-v2-samples-plans-v1", {**SAMPLES_PLANS, "reasons": REASONS},
                    wall_seconds=7200, clone_instances=110)
    before = json.loads((run_b / "configs/comparison.json").read_text())
    after = json.loads((feature / "configs/comparison.json").read_text())
    # Only the features, the sequences / graphs they need and the startup density on the 48-sequence pool; the
    # records (arena_variant, hero_v2) and the bundle name list the same.
    changed = {p for p in diff(before, after) if not p.startswith((".arena_variant.", ".hero_v2.", ".name"))}
    assert changed == {".parallel_samples", ".action_plans", ".server.max_num_seqs",
                       ".server.max_cudagraph_capture_size", ".focus_oversubscription.gate_blocks_per_stream"}
    assert after["server"]["max_num_seqs"] == 48 and after["generation_slots"] == before["generation_slots"] == 26
    assert after["server"]["max_cudagraph_capture_size"] == 48 and after["active_limit"] == 24
    assert after["focus_oversubscription"]["gate_blocks_per_stream"] == 47.458  # 24 streams on >= 1,266 blocks
    assert after["timeline"] == before["timeline"] and after["windows"] == [130816]
    modal = load("modal_samples_plans", ROOT / "scripts/modal_validation.py")
    test = "hero-v2-samples-plans"
    assert modal.validation_timeout("arena", test) == 9600 and modal.arena_arguments(test) == ["--trial", "duck-transfer"]
    source = (ROOT / "scripts/modal_validation.py").read_text()
    start = source.index("            if TEST == 'hero-v2-samples-plans':")
    end = source.index("            if TEST in ('hero-v2-128k-int4', 'hero-v2-streams'):")
    guard = "\n".join(line[12:] for line in source[start:end].splitlines())
    exec(guard, {"json": json, "TEST": test, "source": feature})  # passes
    with pytest.raises(ValueError, match="run B exactly"):
        exec(guard, {"json": json, "TEST": test, "source": run_b})
    assert 'if TEST in ("hero-v2-samples-plans", "campaign-9h"):\n    CPU_CEILING = 16' in source
