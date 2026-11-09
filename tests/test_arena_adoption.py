"""Noise floor and adoption rule: run metrics, strata and the verdict."""

import json

import pytest

from arena.adoption import compare, main, pooled_sd, run_metrics


def report(tmp_path, name, score, *, levels, ft09_t1=None, lp85_t1=None, switch=360.0, score_900=None):
    folder = tmp_path / name / "arena" / "trials" / "duck-transfer"
    folder.mkdir(parents=True)
    games = []
    for gid in ("ft09-x", "lp85-y", "vc33-z"):
        t1 = {"ft09-x": ft09_t1, "lp85-y": lp85_t1}.get(gid)
        games.append({"game_id": gid, "levels": 1 if t1 is not None else 0, "time_to_level": [{"level": 1, "wall_seconds": t1}] if t1 is not None else []})
    (folder / "arena.json").write_text(json.dumps({"mean_score_equal_weight_25_games": score, "levels": levels, "games": games,
                                                   "scheduler": {"switch_wall_seconds": switch}, "status": "complete"}))
    if score_900 is not None:
        (tmp_path / name / "analysis.json").write_text(json.dumps([{"score_curve": {"checkpoints": [{"wall_seconds": 900.0, "score": score_900}]}}]))
    return tmp_path / name


def test_run_metrics_counts_fast_games_by_the_switch(tmp_path):
    folder = report(tmp_path, "r1", 7.0, levels=13, ft09_t1=250, lp85_t1=380, switch=360, score_900=3.5)
    m = run_metrics(folder)
    assert m["fast_games_at_switch"] == 2 and m["games_with_clears"] == 2 and m["clears"] == 13 and m["score_900"] == 3.5
    assert m["level1_times"] == [250, 380] and m["fast_games_present"] == 2
    late = report(tmp_path, "r2", 3.0, levels=9, ft09_t1=200, lp85_t1=900)
    assert run_metrics(late)["fast_games_at_switch"] == 1 and run_metrics(late)["score_900"] is None
    with pytest.raises(FileNotFoundError):
        run_metrics(tmp_path / "missing")


def test_compare_is_stratified_and_needs_a_margin(tmp_path):
    base = [run_metrics(report(tmp_path, f"b{i}", s, levels=10, ft09_t1=t, lp85_t1=u))
            for i, (s, t, u) in enumerate([(7.0, 250, 300), (6.5, 260, 310), (3.5, 250, None), (4.0, None, 300)])]
    cand = [run_metrics(report(tmp_path, f"c{i}", s, levels=12, ft09_t1=t, lp85_t1=u))
            for i, (s, t, u) in enumerate([(7.4, 250, 300), (3.9, 250, None)])]
    result = compare(base, cand)
    assert result["strata"]["2"]["diff"] == pytest.approx(7.4 - 6.75) and result["strata"]["1"]["diff"] == pytest.approx(3.9 - 3.75)
    assert result["stratified"] == pytest.approx(0.4) and result["unstratified"] == pytest.approx(5.65 - 5.25)
    assert result["pooled_sd"] == pytest.approx(pooled_sd([[7.0, 6.5], [3.5, 4.0]]))
    assert result["verdict"].startswith("undecided: |0.40| within") and "runs per side" in result["verdict"]
    big = [dict(r, score=r["score"] + 3.0) for r in cand]
    assert compare(base, big)["verdict"] == "adopt" and compare(big, base)["verdict"] == "reject"
    assert compare(base, cand, metric="score_900")["verdict"] == "undecided: not enough runs"
    one = compare(base[:1], cand[:1], floor_sd=1.4)
    assert one["pooled_sd"] == 1.4 and one["margin"] == pytest.approx(2 * 1.4 * (2 ** 0.5))


def test_cli_prints_runs_and_verdict(tmp_path, capsys):
    b = report(tmp_path, "b", 4.0, levels=10, ft09_t1=250)
    c = report(tmp_path, "c", 4.5, levels=11, ft09_t1=250)
    result = main(["--base", str(b), "--candidate", str(c)])
    out = capsys.readouterr().out
    assert "base        4.00%  10 clears" in out and "verdict: undecided" in out and result["margin"] == pytest.approx(2 * 1.4 * (2 ** 0.5))
    main(["--base", str(b), "--candidate", str(c), "--json"])
    assert json.loads(capsys.readouterr().out)["n_base"] == 1
