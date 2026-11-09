"""Level bench: start points, replay, the attempt agent's stop rules, the summary and the pairing script."""

import asyncio
import importlib.util
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from conftest import FakeEnv, observation, tool_reply

from arc_infra.perception import board_hash
from arena import level_bench
from arena.level_points import check_replay, decode_action, encode_action, level_score, point_id

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/arena_analysis/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefix_actions_round_trip_and_explorer_steps_are_not_replayable():
    assert encode_action({"action": "ACTION3"}) == "ACTION3"
    assert encode_action({"action": "RESET"}) == "RESET"
    assert encode_action({"action": "ACTION6", "x": 12, "y": 40}) == "ACTION6@12,40"
    assert encode_action({"action": "EXPLORER", "count": 30}) is None
    assert encode_action({"action": "ACTION6"}) is None
    assert decode_action("ACTION6@12,40") == ("ACTION6", 12, 40)
    assert decode_action("RESET") == ("RESET", None, None)
    with pytest.raises(ValueError):
        decode_action("ACTION9")
    assert point_id("ar25-0c556536", 0) == "ar25-L1"


def test_level_score_is_the_competition_formula_capped_at_115():
    assert level_score(32, 32, True) == 100.0
    assert level_score(32, 64, True) == 25.0
    assert level_score(32, 16, True) == 115.0
    assert level_score(32, 10, False) == 0.0
    assert level_score(None, 10, True) == 0.0


def test_replay_reproduces_the_recorded_level_and_board():
    board = np.zeros((64, 64), dtype=np.uint8)
    after = board.copy()
    after[3, 3] = 5
    env = FakeEnv([observation(board), observation(after, levels=1)])
    ok, detail, raw = check_replay(env, {"prefix": ["ACTION1"], "level_index": 1, "board_hash": board_hash(after)})
    assert ok and detail == "ok" and raw.levels_completed == 1 and env.calls == [("ACTION1", {})]
    env = FakeEnv([observation(board), observation(after, levels=1)])
    ok, detail, _ = check_replay(env, {"prefix": ["ACTION6@3,4"], "level_index": 1, "board_hash": board_hash(board)})
    assert not ok and "board differs" in detail and env.calls == [("ACTION6", {"x": 3, "y": 4})]


def test_start_point_file_is_valid_and_split_tagged():
    from posttrain.split import DEV, EXCLUDED, HOLDOUT

    data = json.loads((ROOT / "data/level_bench/start_points.json").read_text())
    assert data["schema"] == 1 and data["points"]
    ids = [p["id"] for p in data["points"]]
    assert len(ids) == len(set(ids))
    for point in data["points"]:
        assert point["game"][:4] not in EXCLUDED
        assert point["split"] == ("dev" if point["game"][:4] in DEV else "holdout")
        assert point["game"][:4] in DEV + HOLDOUT
        assert len(point["prefix"]) == point["prefix_actions"]
        assert (point["level_index"] == 0) == (point["prefix"] == [])
        assert 0 <= point["level_index"] < point["total_levels"]
        for action in point["prefix"]:
            decode_action(action)
    assert {p["level"] for p in data["points"] if p["level_index"] == 0} == {1}


def test_settings_validation_and_seeded_plan():
    cfg = level_bench.settings({"concurrency": 12})
    assert cfg["concurrency"] == 12 and cfg["split"] == "dev" and cfg["token_budget"] == 40000
    with pytest.raises(ValueError):
        level_bench.settings({"unknown": 1})
    with pytest.raises(ValueError):
        level_bench.settings({"progress_seconds": 120})
    points = [{"id": f"g{i}-L1"} for i in range(5)]
    first = level_bench.plan(points, 2, 0)
    assert first == level_bench.plan(points, 2, 0) and len(first) == 10
    assert [p for p, _ in first] == [0] * 5 + [1] * 5
    assert sorted(pt["id"] for _, pt in first[:5]) == sorted(pt["id"] for pt in points)


def attempt(point, outcome, tokens, clear=None, score=0.0, level=1, requests=4, noaction=1):
    return {"point": point, "game": point[:4], "level": level, "split": "dev", "outcome": outcome,
            "cleared": outcome == "cleared", "completion_tokens": tokens, "tokens_to_clear": clear,
            "actions": 10, "level_score": score, "gameplay_requests": requests, "noaction_requests": noaction}


def test_summary_excludes_censored_attempts_and_reports_poisson_intervals():
    rows = [attempt("ar25-L1", "cleared", 12000, 11000, 80.0), attempt("ar25-L2", "budget", 40500, level=2),
            attempt("tu93-L1", "cleared", 8000, 8000, 115.0), attempt("tu93-L2", "censored", 9000, level=2),
            attempt("vc33-L1", "error", 100)]
    s = level_bench.summarize(rows, wall_seconds=100.0, budget=40000)
    assert s["attempts_completed"] == 3 and s["attempts_censored"] == 1 and s["attempts_failed"] == 1
    assert s["clears"] == 2 and s["completion_tokens"] == 60500 and s["clear_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert s["clears_per_10k_tokens"] == pytest.approx(2e4 / 60500, abs=1e-4)
    low, high = s["clears_per_10k_tokens_95"]
    assert low < s["clears_per_10k_tokens"] < high
    assert s["median_tokens_to_clear"] == 9500 and s["mean_level_score"] == pytest.approx(65.0)
    assert s["tokens_per_second"] == pytest.approx(69600 / 100.0)
    assert s["censored_completion_tokens"] == 9000 and s["by_depth"]["L2+"]["attempts"] == 1
    assert s["noaction_share"] == pytest.approx(0.25)
    assert level_bench.poisson_interval(0)[0] == 0.0


def test_request_action_counts_from_a_workspace_log(tmp_path):
    log = tmp_path / "log.jsonl"
    lines = [{"kind": "observation", "time": 0, "step": 0, "action": None},
             {"kind": "model", "purpose": "gameplay"},
             {"kind": "model", "purpose": "gameplay"},
             {"kind": "observation", "time": 1, "step": 1, "action": {"action": "ACTION1"}},
             {"kind": "model", "purpose": "compaction"},
             {"kind": "model", "purpose": "gameplay"}]
    log.write_text("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines))
    assert level_bench.request_actions(log) == (3, 2)


class BenchArcade:
    """Hands out scripted environments: every attempt clears unless ``stuck`` names its game."""

    def __init__(self, stuck=()):
        self.stuck, self.made = set(stuck), []

    def open_scorecard(self, tags):
        return "card"

    def close_scorecard(self, card):
        return None

    def make(self, game_id, **kwargs):
        board = np.zeros((64, 64), dtype=np.uint8)
        after = board.copy()
        after[5, 5] = 3
        levels = 0 if game_id in self.stuck else 1
        self.made.append(game_id)
        return FakeEnv([observation(board), *(observation(after, levels=levels) for _ in range(40))])


class QuietMonitor:
    def __init__(self, output, base_url):
        self.start = None

    async def sample(self):
        return None

    async def run(self):
        await asyncio.Event().wait()

    def summary(self):
        return {"valid": True, "preemptions": 0}


async def test_bench_runs_attempts_to_a_clear_or_the_token_budget(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    points = [{"id": point_id(g, 0), "game": g, "split": "dev", "level_index": 0, "level": 1, "total_levels": 5,
               "baseline_actions": 2, "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board)}
              for g in ("aa11-00000000", "bb22-00000000")]
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    calls = []

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
        calls.append(json.loads(request.content))
        reply = tool_reply("action([{'action':'UP'}])", call_id=f"c{len(calls)}",
                           usage={"completion_tokens": 3000, "prompt_tokens": 4000})
        return httpx.Response(200, json=reply.json())

    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                             "token_budget": 5000, "concurrency": 2, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test"}}
    lines = []
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / "out", server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 60,
                                   log=lambda line, **kw: lines.append(json.loads(line)),
                                   arcade=BenchArcade(stuck={"bb22-00000000"}),
                                   transport=httpx.MockTransport(handler))
    report = await bench.run()
    rows = {r["point"]: r for r in report["attempts"]}
    assert rows["aa11-L1"]["outcome"] == "cleared" and rows["aa11-L1"]["tokens_to_clear"] == 3000
    assert rows["aa11-L1"]["actions"] == 1 and rows["aa11-L1"]["level_score"] == 115.0
    # The stuck game acts once per reply; the reply that crosses 5,000 tokens still executes, then nothing more.
    assert rows["bb22-L1"]["outcome"] == "budget" and rows["bb22-L1"]["completion_tokens"] == 6000
    assert rows["bb22-L1"]["gameplay_requests"] == 2 and rows["bb22-L1"]["total_actions"] == 2
    assert report["summary"]["clears"] == 1 and report["summary"]["attempts_completed"] == 2
    assert report["status"] == "complete" and report["gate"]["reserved"] == 0
    assert all(c["max_tokens"] == 4096 for c in calls)
    events = [line["event"] for line in lines]
    assert events[0] == "level_bench_start" and "level_bench_progress" in events and events[-1] == "level_bench_result"
    saved = json.loads((tmp_path / "out/level-bench.json").read_text())
    assert saved["summary"] == report["summary"]


async def test_mismatched_replay_is_recorded_not_played(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    points = [{"id": "aa11-L2", "game": "aa11-00000000", "split": "dev", "level_index": 1, "level": 2,
               "total_levels": 5, "baseline_actions": 2, "prefix": ["ACTION1"], "prefix_actions": 1,
               "board_hash": "not-this-board"}]
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))

    def handler(request):
        raise AssertionError("A mismatched start point must not reach the model")

    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0}, "wall_seconds": 30,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                             "concurrency": 1, "passes": 1, "min_wall_seconds": 5, "launch_cutoff_seconds": 0}}
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / "out", server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 60,
                                   log=lambda *a, **k: None, arcade=BenchArcade(),
                                   transport=httpx.MockTransport(handler))
    report = await bench.run()
    row = report["attempts"][0]
    assert row["outcome"] == "replay_mismatch" and "board differs" in row["error"]
    assert report["status"] == "complete_with_errors" and report["summary"]["attempts_completed"] == 0


def compare_report(outcomes):
    rows = []
    for index, (outcome, tokens, clear) in enumerate(outcomes):
        rows.append({"point": f"p{index}-L1", "pass": 0, "split": "dev", "outcome": outcome,
                     "cleared": outcome == "cleared", "completion_tokens": tokens, "tokens_to_clear": clear,
                     "level_score": 100.0 if outcome == "cleared" else 0.0})
    return {"attempts": rows, "settings": {"token_budget": 40000}}


def test_compare_pairs_attempts_mcnemar_and_token_ratios():
    compare = load_script("level_bench_compare")
    a = compare_report([("cleared", 10000, 10000), ("cleared", 20000, 20000), ("budget", 40000, None),
                        ("budget", 40000, None), ("censored", 5000, None)])
    b = compare_report([("cleared", 5000, 5000), ("budget", 40000, None), ("cleared", 30000, 30000),
                        ("cleared", 8000, 8000), ("cleared", 4000, 4000)])
    result = compare.compare(a, b)
    assert result["pairs"] == 4 and result["unpaired"] == {"a": 0, "b": 1}
    assert result["mcnemar"] == {"both": 1, "a_only": 1, "b_only": 2, "neither": 0}
    assert result["mcnemar_p_two_sided"] == pytest.approx(1.0)
    assert result["token_ratio_b_over_a_geomean"] == pytest.approx(0.5)
    assert result["clear_rate_diff_b_minus_a"] == pytest.approx(0.25)
    ratio, (low, high) = result["rate_ratio_b_over_a"], result["rate_ratio_95"]
    assert low < ratio < high
    # At a 25K budget B's 30K clear no longer counts and its tokens are capped.
    tight = compare.compare(a, b, budget=25000)
    assert tight["mcnemar"] == {"both": 1, "a_only": 1, "b_only": 1, "neither": 1}
    assert tight["clears_per_10k_tokens"]["b"] == pytest.approx(2 * 1e4 / (5000 + 25000 + 25000 + 8000))


def test_level_bench_build_uses_the_fixed_2x_agent_mtp_off_and_the_committed_points(tmp_path):
    spec = importlib.util.spec_from_file_location("build_arena", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build("level-bench-test", output=tmp_path / "bundle", paired_30m=True,
                                  arena_variant="reclaim-fixed-2x", level_bench={"wall_seconds": 2400, "mtp_tokens": 0})
    assert study["trials"] == [{"name": "level-bench", "kind": "level-bench"}]
    assert study["server"]["mtp_tokens"] == 0 and study["reasoning_effort"] == "xhigh"
    assert study["completion_cap"] == 16384 and study["cache_experiment"]["reclaim_skipped_blocks"] is True
    cfg = level_bench.settings(study["level_bench"])
    assert cfg["split"] == "dev" and cfg["serving"].startswith("mtp0") and cfg["window"] == 49152
    points = json.loads((output / cfg["start_points"]).read_text())
    assert points == json.loads((ROOT / "data/level_bench/start_points.json").read_text())
    assert study["level_bench_points"]["selected"] == sum(p["split"] == "dev" for p in points["points"])
    assert study["wall_seconds"] == 2400 and study["notebook_seconds"] == 3600
    mtp3 = builder.build("level-bench-mtp3", output=tmp_path / "b3", paired_30m=True,
                         arena_variant="reclaim-fixed-2x", level_bench={"mtp_tokens": 3, "split": "holdout"})[1]
    assert mtp3["server"]["mtp_tokens"] == 3 and mtp3["level_bench"]["split"] == "holdout"
    with pytest.raises(ValueError):
        builder.build("bad", output=tmp_path / "b4", paired_30m=True, arena_variant="reclaim-fixed-2x",
                      level_bench={"mtp_tokens": 1})
    with pytest.raises(ValueError):
        builder.build("bad", output=tmp_path / "b5", level_bench={})


def test_modal_level_bench_test_kind():
    import ast

    source = (ROOT / "scripts/modal_validation.py").read_text()
    tree = ast.parse(source)
    ns = {"Path": Path}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("validation_timeout", "arena_arguments"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "modal_validation", "exec"), ns)
    assert ns["validation_timeout"]("arena", "level-bench") == 3600
    assert ns["arena_arguments"]("level-bench") == ["--trial", "level-bench"]


def test_bootstrap_echoes_only_bench_lines(tmp_path, capsys, monkeypatch):
    import sys

    # In the bundle the Qwen bootstrap sits next to the arena one as ``bootstrap``.
    helpers = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts/qwen/bootstrap.py")
    qwen = importlib.util.module_from_spec(helpers)
    helpers.loader.exec_module(qwen)
    monkeypatch.setitem(sys.modules, "bootstrap", qwen)
    spec = importlib.util.spec_from_file_location("bootstrap_arena", ROOT / "scripts/arena/bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = ("print('noise'); print('{\"event\": \"level_bench_progress\", \"done\": 1}'); "
               "print('{\"event\": \"other\"}')")
    code = module.run_echoing([sys.executable, "-c", program], None, tmp_path / "p.log", timeout=30,
                              prefix='{"event": "level_bench')
    out = capsys.readouterr().out
    assert code == 0 and out.strip() == '{"event": "level_bench_progress", "done": 1}'
    assert "noise" in (tmp_path / "p.log").read_text()


def test_start_point_builder_parses_first_observation_of_each_level(tmp_path):
    builder = load_script("level_bench_start_points")
    frame = ["0" * 64] * 64
    other = ["1" + "0" * 63] + ["0" * 64] * 63
    records = [
        {"kind": "observation", "step": 0, "action": None, "frames": [frame], "levels": 0, "total_levels": 3,
         "state": "NOT_FINISHED"},
        {"kind": "model"},
        {"kind": "observation", "step": 1, "action": {"action": "ACTION6", "x": 2, "y": 3}, "frames": [frame],
         "levels": 0, "total_levels": 3, "state": "NOT_FINISHED"},
        {"kind": "observation", "step": 2, "action": {"action": "ACTION1"}, "frames": [other], "levels": 1,
         "total_levels": 3, "state": "NOT_FINISHED"},
        {"kind": "observation", "step": 3, "action": {"action": "EXPLORER", "count": 5}, "frames": [frame],
         "levels": 2, "total_levels": 3, "state": "NOT_FINISHED"},
    ]
    log = tmp_path / "log.jsonl"
    log.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
    found = builder.candidates(log)
    assert len(found) == 1
    k, prefix, digest, total, step = found[0]
    assert (k, prefix, total, step) == (1, ["ACTION6@2,3", "ACTION1"], 3, 2)
    from arc_infra.perception import decode_grid

    assert digest == board_hash(decode_grid(other))


def test_cli_runs_the_level_bench_trial_without_the_arena_wall_check(tmp_path, monkeypatch):
    from arena import run as run_module

    study = json.loads((ROOT / "configs/arena.json").read_text())
    study.update(trials=[{"name": "level-bench", "kind": "level-bench"}], wall_seconds=2400, level_bench={})
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study))
    seen = {}

    async def fake_bench(study, trial, **kwargs):
        seen.update(kwargs)
        return {"status": "complete", "summary": {"clears": 3}, "attempts": [{"x": 1}], "config": {}}

    async def ok(*args, **kwargs):
        return {"tokenizer_count": 5}

    monkeypatch.setattr(level_bench, "run_level_bench", fake_bench)
    monkeypatch.setattr(run_module, "validate_api", ok)
    monkeypatch.setattr(run_module, "wait_idle", ok)
    monkeypatch.setattr(run_module, "isolate_trial", ok)
    monkeypatch.setattr(run_module, "hardware", lambda: {"gpu": "fake"})
    monkeypatch.setattr(run_module.signal, "setitimer", lambda *a: None)
    start = time.time()
    # 1,500 s left: less than the study wall plus 480 s, which the bench sizes itself.
    run_module.main(["--source", ".", "--study", str(path), "--output", str(tmp_path / "out"),
                     "--server-dir", str(tmp_path), "--games-dir", "games", "--trial", "level-bench",
                     "--notebook-start-epoch", str(start), "--notebook-seconds", "1500"])
    record = json.loads((tmp_path / "out/study.json").read_text())
    trial = record["trials"][0]
    assert trial["status"] == "complete" and trial["report"] == {"status": "complete", "summary": {"clears": 3}}
    assert seen["deadline_epoch"] == pytest.approx(start + 1200, abs=1) and seen["games_dir"] == "games"
