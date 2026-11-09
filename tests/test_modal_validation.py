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
