"""Test the paid fallback's budget guard without importing Modal or launching jobs."""

import ast
import json
from pathlib import Path

import pytest


def load_reserve(root):
    source = Path(__file__).resolve().parents[1] / "scripts/modal_validation.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reserve")
    ns = {"ROOT": root, "json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    return ns["reserve"]


def test_budget_rejects_duplicate_and_over_cap(tmp_path):
    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs/spend.json"
    path.write_text(json.dumps({"infrastructure_reserve_usd": 2, "runs": [], "cap_usd": 5}))
    reserve = load_reserve(tmp_path)
    bound = reserve("arena", "arena", 1680)
    assert bound == pytest.approx(2.74745304)
    before = path.read_text()
    with pytest.raises(ValueError, match="already reserved"):
        reserve("arena", "arena", 1)
    assert path.read_text() == before
    with pytest.raises(RuntimeError, match="Spend cap"):
        reserve("sglang", "sglang", 3600)
    assert path.read_text() == before


def test_modal_only_submission_reserves_before_spawn_without_kaggle(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "scripts/modal_validation.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    fn.decorator_list = []
    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs/spend.json"
    path.write_text(json.dumps({"infrastructure_reserve_usd": 2, "runs": [], "cap_usd": 5}))
    calls = []

    def spawn(run_id):
        ledger = json.loads(path.read_text())
        assert ledger["runs"][-1]["run_id"] == run_id
        calls.append(run_id)
        return SimpleNamespace(object_id="fc-test")

    def reject_kaggle(*args, **kwargs):
        raise AssertionError("Modal-only launch must not query Kaggle")

    monkeypatch.setattr(subprocess, "check_output", reject_kaggle)
    ns = {"ROOT": tmp_path, "json": json, "reserve": load_reserve(tmp_path),
          "KIND": "arena", "TEST": "smoke", "TIMEOUT": 1680, "BUNDLE": tmp_path,
          "benchmark": SimpleNamespace(spawn=spawn)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    ns["main"]("test")
    assert calls == ["test"]
    record = json.loads((tmp_path / "docs/validation-20260922/modal-test.json").read_text())
    assert record["state"] == "submitted" and record["function_call_id"] == "fc-test"
    with pytest.raises(RuntimeError, match="Spend cap"):
        ns["main"]("over-budget")
    assert calls == ["test"]
    assert not (tmp_path / "docs/validation-20260922/modal-over-budget.json").exists()


def test_paired_budget_records_full_two_trial_timeout(tmp_path):
    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs/spend.json"
    path.write_text(json.dumps({"infrastructure_reserve_usd": 2, "runs": [], "cap_usd": 20}))
    bound = load_reserve(tmp_path)("paired", "arena", 8100, test="paired-30m")
    row = json.loads(path.read_text())["runs"][0]
    assert bound == pytest.approx((8100 + 60) * (0.000842 + 8 * 0.0000131 + 192 * 0.00000222) * 1.15)
    assert row["max_seconds"] == 8100
    assert "Kaggle-first" in row["authorization"]


def test_full_arena_selects_one_trial_without_smoke_overrides():
    from arena.trial import trial_settings

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts/modal_validation.py").read_text())
    names = {"validation_timeout", "arena_arguments"}
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), "validation", "exec"), ns)
    args = ns["arena_arguments"]("full-duck-transfer")
    assert args == ["--trial", "duck-transfer"]
    study = json.loads((root / "configs/arena.json").read_text())
    games, wall = trial_settings(study, smoke="--smoke" in args)
    assert len(set(games)) == 25 and wall == 900
    assert ns["validation_timeout"]("arena", "full-duck-transfer") == 2100
    assert ns["arena_arguments"]("smoke") == ["--smoke"]
    assert ns["arena_arguments"]("paired-30m") == []
    assert ns["validation_timeout"]("arena", "paired-30m") == 8100
    with pytest.raises(ValueError):
        ns["validation_timeout"]("sglang", "full-duck-transfer")


def test_duck_30m_fallback_selects_duck_only_with_its_own_budget(tmp_path):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts/modal_validation.py").read_text())
    names = {"validation_timeout", "arena_arguments"}
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), "validation", "exec"), ns)
    assert ns["arena_arguments"]("duck-30m") == ["--trial", "duck-transfer"]
    assert ns["validation_timeout"]("arena", "duck-30m") == 4500
    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs/spend.json"
    path.write_text(json.dumps({"infrastructure_reserve_usd": 2, "runs": [], "cap_usd": 20}))
    bound = load_reserve(tmp_path)("duck30", "arena", 4500, test="duck-30m")
    row = json.loads(path.read_text())["runs"][0]
    assert bound == pytest.approx((4500 + 60) * (0.000842 + 8 * 0.0000131 + 192 * 0.00000222) * 1.15)
    assert "10 minutes" in row["authorization"]


def test_bench_fallback_selects_the_serving_benchmark_trial():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts/modal_validation.py").read_text())
    names = {"validation_timeout", "arena_arguments"}
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), "validation", "exec"), ns)
    assert ns["arena_arguments"]("bench") == ["--trial", "serving-bench"]
    assert ns["validation_timeout"]("arena", "bench") == 3600


def test_dev10_preview_rejects_heldout_and_wrong_roster():
    source = Path(__file__).resolve().parents[1] / "scripts/modal_validation.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_dev10_preview")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    study = {"games": [f"{g}-version" for g in "cd82 cn04 ft09 lp85 ls20 re86 sb26 sk48 vc33 wa30".split()],
             "competition": {"preview_instances": 100, "preview_seconds": 3600}, "wall_seconds": 3600,
             "hero_v2": {"requested": {}}, "trials": [{"name": "duck-transfer", "kind": "duck-transfer"}]}
    validate = ns["validate_dev10_preview"]
    validate(study)
    study["games"][0] = "ar25-version"
    with pytest.raises(ValueError, match="DEV10"):
        validate(study)
    study["games"][0] = "cd82-version"
    study["competition"]["preview_instances"] = 110
    with pytest.raises(ValueError, match="DEV10"):
        validate(study)


def test_heldout15_preview_rejects_dev_and_wrong_roster():
    source = Path(__file__).resolve().parents[1] / "scripts/modal_validation.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_heldout15_preview")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    study = {"games": [f"{g}-version" for g in "ar25 bp35 dc22 g50t ka59 lf52 m0r0 r11l s5i5 sc25 sp80 su15 tn36 tr87 tu93".split()],
             "competition": {"preview_instances": 150, "preview_seconds": 3600}, "wall_seconds": 3600,
             "hero_v2": {"requested": {}}, "trials": [{"name": "duck-transfer", "kind": "duck-transfer"}]}
    validate = ns["validate_heldout15_preview"]
    validate(study)
    study["games"][0] = "cd82-version"
    with pytest.raises(ValueError, match="HELDOUT15"):
        validate(study)
    study["games"][0] = "ar25-version"
    study["competition"]["preview_instances"] = 110  # build_v71.py's cap (v7.1-harness capture runs, 7 October)
    validate(study)
    study["competition"]["preview_instances"] = 100
    with pytest.raises(ValueError, match="HELDOUT15"):
        validate(study)


def test_quality_fixed_validates_bundle_before_reserving(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts/modal_validation.py").read_text())
    names = {"validate_quality_fixed", "validation_timeout", "arena_arguments"}
    ns = {"json": json}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                           type_ignores=[]), "validation", "exec"), ns)
    (tmp_path / "configs").mkdir()
    (tmp_path / "src/arena").mkdir(parents=True)
    validator = tmp_path / "src/arena/quality_eval.py"
    validator.write_text("def validate_study(study):\n    return study.get('quality_eval')\n")
    study = {"quality_eval": {"version": 1}, "wall_seconds": 3600, "notebook_seconds": 7500,
             "trials": [{"name": "duck-transfer", "kind": "duck-transfer"}]}
    config = tmp_path / "configs/comparison.json"
    config.write_text(json.dumps(study))
    assert ns["validate_quality_fixed"](tmp_path) == {"version": 1}
    assert ns["arena_arguments"]("quality-fixed") == ["--trial", "duck-transfer"]
    assert ns["validation_timeout"]("arena", "quality-fixed") == 7800
    study["wall_seconds"] = 3601
    config.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="one hour"):
        ns["validate_quality_fixed"](tmp_path)
    study["wall_seconds"] = 3600
    study.pop("quality_eval")
    config.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="explicit quality_eval"):
        ns["validate_quality_fixed"](tmp_path)
    validator.write_text("def validate_study(study):\n    raise ValueError('invalid fixed roster')\n")
    with pytest.raises(ValueError, match="invalid fixed roster"):
        ns["validate_quality_fixed"](tmp_path)
    # Version 2 (replicated roster) has its own test and limits; neither version launches as the other.
    replicated = "quality-fixed-replicated"
    validator.write_text("def validate_study(study):\n    return study.get('quality_eval')\n")
    study.update(quality_eval={"version": 2}, wall_seconds=9000, notebook_seconds=12900,
                 competition={"preview_seconds": 9000})
    config.write_text(json.dumps(study))
    assert ns["validate_quality_fixed"](tmp_path, replicated) == {"version": 2}
    assert ns["arena_arguments"](replicated) == ["--trial", "duck-transfer"]
    assert ns["validation_timeout"]("arena", replicated) == 13200
    # A bundle with a shorter notebook reserves only its own time; the module computes TIMEOUT before BUNDLE exists.
    config.write_text(json.dumps({**study, "notebook_seconds": 4200}))
    monkeypatch.setenv("ARC_VALIDATION_BUNDLE", str(tmp_path))
    assert ns["validation_timeout"]("arena", replicated) == 4500
    monkeypatch.delenv("ARC_VALIDATION_BUNDLE")
    config.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="version 2"):
        ns["validate_quality_fixed"](tmp_path)
    for change in ({"wall_seconds": 9001}, {"notebook_seconds": 12901}, {"competition": {"preview_seconds": 3600}},
                   {"quality_eval": {"version": 1}}):
        config.write_text(json.dumps({**study, **change}))
        with pytest.raises(ValueError):
            ns["validate_quality_fixed"](tmp_path, replicated)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [(n.lineno, n.func.id) for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert min(line for line, name in calls if name == "validate_quality_fixed") < min(
        line for line, name in calls if name == "reserve")


def test_throughput_arm_cap_is_atomic_and_separate(tmp_path):
    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs/spend.json"
    path.write_text(json.dumps({"infrastructure_reserve_usd": 0, "cap_usd": 5000,
                               "runs": [{"run_id": "tp-existing", "charged_bound_usd": 1999}]}))
    reserve = load_reserve(tmp_path)
    before = path.read_text()
    with pytest.raises(RuntimeError, match="Throughput arm spend cap"):
        reserve("tp-over", "arena", 1680)
    assert path.read_text() == before
    reserve("separate-harness", "arena", 1680)
