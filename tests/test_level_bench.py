"""Level bench: start points, replay, the attempt agent's stop rules, the summary and the pairing script."""

import asyncio
import importlib.util
import json
import sys
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
    assert [p for p, _, _ in first] == [0] * 5 + [1] * 5
    assert sorted(pt["id"] for _, pt, _ in first[:5]) == sorted(pt["id"] for pt in points)


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
    partial = [json.loads(line) for line in open(tmp_path / "out/level-bench.partial.jsonl")]
    assert sorted(r["point"] for r in partial) == sorted(rows) and all(r["outcome"] for r in partial)


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
    assert ns["arena_arguments"]("hx4-level") == ["--trial", "level-bench"]
    ns["hx4_study"] = lambda: {"notebook_seconds": 2280}
    assert ns["validation_timeout"]("arena", "hx4-level") == 2580
    assert ns["arena_arguments"]("hx4-quality") == ["--trial", "duck-transfer"]
    ns["hx4_study"] = lambda: {"notebook_seconds": 10300}
    assert ns["validation_timeout"]("arena", "hx4-quality") == 10600


def test_bootstrap_echoes_only_bench_lines(tmp_path, capsys, monkeypatch):
    import sys

    # In the bundle the Qwen bootstrap sits next to the arena one as ``bootstrap``.
    helpers = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts/arena/bundle_bootstrap.py")
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


V71 = {
    "harness_fixes": {"fixes": ["exact_trigger", "summary_guard", "inventory", "cap_followup", "stuck_notice", "sandbox",
                                "timeout_interrupt", "prompt_text", "summary_headroom", "text_count"],
                      "summary_room_tokens": 4096, "inventory_max_tokens": 1536, "inventory_receipts": 12,
                      "stuck_turns": 16, "stuck_notice_turns": 8, "stuck_no_change_actions": 5,
                      "interrupt_grace_seconds": 3.0},
    "drain_half": {"drain_tokens": 59392, "turn_cap": 150, "turn_drain": 30, "coalesce": True,
                   "tool_output_tokens": 3072, "middle_truncation": True, "drop_malformed": True, "note": True},
    "hx_harness": {"toolbox": True, "reflect": True, "economy": True, "memo": True, "reflect_style": 2, "memo_once": 1},
    "dfr_harness": {"vision": True, "guards": True, "actions": True, "level_prompt": True, "frame_diff": True,
                    "animation": True},
    "dfr_loop": {"prompt_fix": True, "yield_tokens": 0, "no_world_model": False, "action_feedback": False,
                 "inventory_animation": True},
    "loop_rule": {"action": "fresh_context", "streak": 20, "mean_completion": 300, "similarity": 0.9},
    "budget_clock": "tokens", "reasoning_history": "verbatim", "compaction_thinking": "prefilled-empty",
}


class PrefixBenchArcade(BenchArcade):
    """One prefix action reaches level 2; the attempt's first action clears it."""

    def make(self, game_id, **kwargs):
        board = np.zeros((64, 64), dtype=np.uint8)
        second = board.copy()
        second[5, 5] = 3
        third = second.copy()
        third[9, 9] = 4
        return FakeEnv([observation(board), observation(second, levels=1),
                        *(observation(third, levels=2) for _ in range(10))])


def test_plan_rotates_arms_over_points_and_passes():
    points = [{"id": f"p{i}"} for i in range(3)]
    arms = [{"name": "a"}, {"name": "b"}]
    work = level_bench.plan(points, 2, 0, arms, first_pass=1)
    assert len(work) == 12 and {w[0] for w in work} == {1, 2}
    assert {(w[0], w[1]["id"], w[2]["name"]) for w in work} == {(p, f"p{i}", a) for p in (1, 2) for i in range(3)
                                                                 for a in "ab"}
    first = work[:3]  # the first third of a pass already holds both arms
    assert {w[2]["name"] for w in first} == {"a", "b"}
    assert level_bench.plan(points, 1, 0)[0][2]["name"] == "base"
    mixed = points + [{"id": "d0", "deep_tokens": 150000}]
    split = level_bench.plan(mixed, 1, 0, [{"name": "a", "points": "start"}, {"name": "b", "points": "deep"}, {"name": "c"}])
    assert sorted((w[1]["id"], w[2]["name"]) for w in split) == sorted(
        [(f"p{i}", a) for i in range(3) for a in "ac"] + [("d0", "b"), ("d0", "c")])
    games = [{"id": "x", "game": "aa11-1"}, {"id": "y", "game": "bb22-1"}]
    only = level_bench.plan(games, 1, 0, [{"name": "a"}, {"name": "b", "games": ["bb22"]}])
    assert sorted((w[1]["id"], w[2]["name"]) for w in only) == [("x", "a"), ("y", "a"), ("y", "b")]


async def test_arms_run_the_v71_harness_with_warm_notes_and_replayed_history(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    second = board.copy()
    second[5, 5] = 3
    point = {"id": "aa11-L2", "game": "aa11-00000000", "split": "dev", "level_index": 1, "level": 2, "total_levels": 5,
             "baseline_actions": 2, "prefix": ["ACTION1"], "prefix_actions": 1, "board_hash": board_hash(second),
             "warm": {"cross_level_notes": "Prior-level hypotheses: the red key opens doors.",
                      "notes": {"mech": "UP moves the cursor"}}}
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": [point]}))
    calls = {}

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 131072})
        payload = json.loads(request.content)
        text = json.dumps(payload["messages"])
        arm = next(a for a in ("cold", "notes", "hist") if f"ARM-{a}" in text) if "ARM-" in text else None
        seen = calls.setdefault(id(request.extensions.get("x", request)), [])
        seen.append(payload)
        n = sum(1 for m in payload["messages"] if m["role"] == "assistant")
        code = ("print('HIST', len(history), current_frame.step, current_frame.level, len(transitions));"
                "print('NOTES', sorted(note('zz', '1')))") if n == 0 else "action([{'action':'UP'}])"
        return httpx.Response(200, json=tool_reply(code, call_id=f"c{n}",
                                                   usage={"completion_tokens": 300, "prompt_tokens": 4000}).json())

    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 1, "max_model_len": 131072}, "wall_seconds": 60, **V71,
             "level_bench": {"start_points": str(path), "capacity_tokens": 2000000, "window": 130816,
                             "token_budget": 5000, "concurrency": 4, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test",
                             "arms": [{"name": "cold"}, {"name": "notes", "warm": True},
                                      {"name": "hist", "history": True},
                                      {"name": "both", "warm": True, "history": True,
                                       "study": {"hx_harness": {**V71["hx_harness"], "reflect": False}}}]}}
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / "out", server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 120,
                                   log=lambda *a, **k: None, arcade=PrefixBenchArcade(),
                                   transport=httpx.MockTransport(handler))
    report = await bench.run()
    rows = {r["arm"]: r for r in report["attempts"]}
    assert set(rows) == {"cold", "notes", "hist", "both"}, report["task_errors"]
    assert all(r["outcome"] == "cleared" and r["actions"] == 1 for r in rows.values()), rows
    assert report["summary"]["by_arm"]["both"]["clears"] == 1
    logs = {arm: (tmp_path / "out/attempts" / rows[arm]["label"] / "log.jsonl").read_text() for arm in rows}
    transcripts = {arm: (tmp_path / "out/attempts" / rows[arm]["label"] / "duck-transcript.txt").read_text()
                   for arm in rows}
    assert "HIST 1 0 2 0" in logs["cold"] and "HIST 1 0 2 0" in logs["notes"]
    assert "HIST 2 1 2 1" in logs["hist"] and "HIST 2 1 2 1" in logs["both"]   # seed frame + the replayed action
    assert "'mech', 'zz'" in logs["notes"] and "'mech', 'zz'" in logs["both"] and "'mech'" not in logs["cold"]
    for arm in ("notes", "both"):
        assert "Cross-level notes: Prior-level hypotheses: the red key opens doors." in transcripts[arm]
        assert "Your notes from the earlier levels: mech: UP moves the cursor" in transcripts[arm]
    assert "red key" not in transcripts["cold"] and "red key" not in transcripts["hist"]
    assert bench.attempts[0]["agent"].hx["reflect"] in (True, False)
    by = {a["arm"]: a["agent"] for a in bench.attempts}
    assert by["both"].hx["reflect"] is False and by["cold"].hx["reflect"] is True and by["cold"].drain is not None


class DeepBenchArcade(BenchArcade):
    """ACTION1 reaches level 2, ACTION4 moves inside it; a reset returns to its first board; the next action clears."""

    def __init__(self):
        super().__init__()
        self.envs = []

    def make(self, game_id, **kwargs):
        board = np.zeros((64, 64), dtype=np.uint8)
        start = board.copy()
        start[5, 5] = 3
        inside = start.copy()
        inside[20, 20] = 6
        won = start.copy()
        won[9, 9] = 4

        class Env(FakeEnv):
            def reset(self):
                self.reset_count += 1
                self.observation_space = raw(observation(start, levels=1))
                return self.observation_space

        from conftest import raw

        env = Env([observation(board), observation(start, levels=1), observation(inside, levels=1),
                   *(observation(won, levels=2) for _ in range(10))])
        self.envs.append(env)
        return env


async def test_fresh_attempt_restarts_with_the_earlier_beliefs_as_suspects(tmp_path, monkeypatch):
    import arc_infra.metrics

    from arena import fresh_attempt

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    start = np.zeros((64, 64), dtype=np.uint8)
    start[5, 5] = 3
    inside = start.copy()
    inside[20, 20] = 6
    point = {"id": "aa11-L2-deep", "game": "aa11-00000000", "split": "dev", "level_index": 1, "level": 2,
             "total_levels": 5, "baseline_actions": 2, "prefix": ["ACTION1", "ACTION4"], "prefix_actions": 2,
             "level_start_actions": 1, "board_hash": board_hash(inside),
             "warm": {"cross_level_notes": "", "notes": {"mech": "UP moves the cursor"},
                      "knowledge": {"goal_model": "fill the blue slot", "world_model": "the red block is immovable"}}}
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": [point]}))
    requests = []

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 131072})
        payload = json.loads(request.content)
        requests.append(payload)
        text = json.dumps(payload["messages"])
        n = sum(1 for m in payload["messages"] if m["role"] == "assistant")
        act = "[Fresh attempt]" in text or n >= 5
        code = "action([{'action':'UP'}])" if act else "print('LOOK', current_frame.step, len(history))"
        return httpx.Response(200, json=tool_reply(code, call_id=f"c{len(requests)}", content="World model: wall.",
                                                   usage={"completion_tokens": 300, "prompt_tokens": 4000}).json())

    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 1, "max_model_len": 131072}, "wall_seconds": 60, **V71,
             "level_bench": {"start_points": str(path), "capacity_tokens": 2000000, "window": 130816,
                             "token_budget": 5000, "concurrency": 1, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test",
                             "arms": [{"name": "cont", "warm": True, "history": True},
                                      {"name": "refute", "warm": True, "history": True,
                                       "study": {"fresh_attempt": {"at_start": True}}},
                                      {"name": "amnesia", "history": True, "reset": True},
                                      {"name": "periodic", "warm": True, "history": True,
                                       "study": {"fresh_attempt": {"tokens": 1000}}}]}}
    arcade = DeepBenchArcade()
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / "out", server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 120,
                                   log=lambda *a, **k: None, arcade=arcade, transport=httpx.MockTransport(handler))
    report = await bench.run()
    rows = {r["arm"]: r for r in report["attempts"]}
    assert all(r["outcome"] == "cleared" for r in rows.values()), (rows, report["task_errors"])
    order = [a["arm"] for a in sorted(bench.attempts, key=lambda a: a["index"])]
    resets = dict(zip(order, (env.reset_count for env in arcade.envs)))
    assert resets == {"cont": 0, "refute": 1, "amnesia": 1, "periodic": 1}
    assert {arm: rows[arm]["actions"] for arm in rows} == {"cont": 1, "refute": 2, "amnesia": 2, "periodic": 2}
    logs = {arm: [json.loads(line) for line in (tmp_path / "out/attempts" / rows[arm]["label"] / "log.jsonl").open()]
            for arm in rows}
    first = {}
    for payload in requests:
        text = json.dumps(payload["messages"])
        key = ("refute" if "[Fresh attempt]" in text and len(payload["messages"]) == 3 and "A long earlier" in text
               else "cont" if "Goal model: fill the blue slot" in text else None)
        if key and key not in first:
            first[key] = payload["messages"]
    note = first["refute"][1]["content"]
    assert note.startswith("[Fresh attempt] A long earlier attempt went into level 2")
    assert "reset to its starting board" in note and "Treat every line as a suspect" in note
    assert "- Goal: fill the blue slot" in note and "- World: the red block is immovable" in note
    assert "- Note mech: UP moves the cursor" in note and "Live python definitions" in note or "python" in note
    assert "Goal model: fill the blue slot" not in json.dumps(first["refute"][2])  # no carried model beside the note
    assert "Goal model: fill the blue slot" in json.dumps(first["cont"])
    fresh = [r for r in logs["periodic"] if r["kind"] == "fresh_attempt"]
    assert len(fresh) == 1 and fresh[0]["reset"] and fresh[0]["tokens"] >= 1000 and fresh[0]["number"] == 1
    assert "world_model" in fresh[0]["belief_fields"]
    assert [r["kind"] for r in logs["periodic"]].count("fresh_attempt_context") == 1
    assert not [r for r in logs["cont"] if r["kind"] == "fresh_attempt"]
    assert fresh_attempt.cutoff(fresh_attempt.settings({"tokens": 1000, "growth": 2, "max": 2}), 1) == 2000
    assert fresh_attempt.cutoff(fresh_attempt.settings({"tokens": 1000, "max": 2}), 2) is None
    with pytest.raises(ValueError):
        fresh_attempt.settings({"beliefs": "keep"})


async def test_lead_points_let_the_agent_play_the_levels_before_the_target(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    point = {"id": "aa11-L2-lead1", "game": "aa11-00000000", "split": "dev", "level_index": 0, "level": 2,
             "lead_levels": 1, "total_levels": 5, "baseline_actions": 2, "prefix": [], "prefix_actions": 0,
             "board_hash": board_hash(np.zeros((64, 64), dtype=np.uint8)), "warm": {}}
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": [point]}))
    acting = {"on": True}

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 131072})
        code = "action([{'action':'UP'}])" if acting["on"] else "print('LOOK')"
        return httpx.Response(200, json=tool_reply(code, call_id="c", content="World model: wall.",
                                                   usage={"completion_tokens": 300, "prompt_tokens": 4000}).json())

    def run(name, lead_budget):
        study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
                 "server": {"mtp_tokens": 1, "max_model_len": 131072}, "wall_seconds": 60, **V71,
                 "level_bench": {"start_points": str(path), "capacity_tokens": 2000000, "window": 130816,
                                 "token_budget": 5000, "concurrency": 1, "passes": 1, "min_wall_seconds": 5,
                                 "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test",
                                 "lead_token_budget": lead_budget,
                                 "arms": [{"name": "lead", "points": "lead"}, {"name": "start", "points": "start"}]}}
        return level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                      games_dir=None, output=tmp_path / name, server_dir=tmp_path,
                                      base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 120,
                                      log=lambda *a, **k: None, arcade=DeepBenchArcade(),
                                      transport=httpx.MockTransport(handler)).run()

    report = await run("a", 2000)
    (row,) = report["attempts"]  # the start-only arm takes no lead point
    assert (row["arm"], row["outcome"], row["level"]) == ("lead", "cleared", 2), (row, report["task_errors"])
    assert (row["lead_tokens"], row["lead_actions"], row["actions"]) == (300, 1, 2)
    assert row["tokens_to_clear"] == row["completion_tokens"] == 600
    acting["on"] = False
    report = await run("b", 500)
    (row,) = report["attempts"]
    assert row["outcome"] == "lead_out" and row["completion_tokens"] == 0 and row["lead_tokens"] >= 500
    assert report["summary"]["attempts_completed"] == 0


def test_enter_arms_and_play_through_points():
    from arena import level_bench as lb

    base = {"start_points": "x.json"}
    arms = [{"name": "keep", "points": "game"}, {"name": "clear", "points": "game", "enter": "clear"},
            {"name": "restart", "points": "game", "enter": "restart"}, {"name": "both", "points": "game", "enter": "both"}]
    assert lb.settings({**base, "arms": arms})["arms"] == arms
    for bad in ({"name": "a", "enter": "clear"}, {"name": "a", "points": "start", "enter": "clear"},
                {"name": "a", "points": "game", "enter": "wipe"}):
        with pytest.raises(ValueError):
            lb.settings({**base, "arms": [bad]})
    points = [{"id": f"g{i}-game", "game": f"g{i}aa-x", "through": True} for i in range(4)]
    assert {lb.point_kind(p) for p in points} == {"game"}
    work = lb.plan(points, 3, 0, arms)
    assert len(work) == 48
    counts = {}
    for _, point, arm in work:
        counts[(point["id"], arm["name"])] = counts.get((point["id"], arm["name"]), 0) + 1
    assert set(counts.values()) == {3} and len(counts) == 16
    # the first 16 launches (one pass) hold every game under every arm once
    assert len({(p["id"], a["name"]) for _, p, a in work[:16]}) == 16
    assert "note() entries" in lb.ENTER_RESTART_NOTE and "note() entries" in lb.ENTER_BOTH_NOTE
    assert "python session is unchanged" in lb.ENTER_CLEAR_NOTE


async def test_trim_bed_records_a_script_and_replays_it_before_each_arms_trim(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    point = {"id": "aa11-game", "game": "aa11-00000000", "split": "dev", "level_index": 0, "level": 1,
             "through": True, "total_levels": 5, "baselines": [2, 2, 2, 2, 2], "baseline_actions": None,
             "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board)}
    live, seen = {"on": False}, []

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": len(request.content) // 4, "max_model_len": 262144})
        payload = json.loads(request.content)
        n = sum(1 for m in payload["messages"] if m["role"] == "assistant")
        usage = {"completion_tokens": 300, "prompt_tokens": len(request.content) // 4}
        if live["on"]:
            seen.append(payload)
            if payload.get("tool_choice") == "none":
                message = {"role": "assistant", "content": "## Rules (verified)\nUP paints (5, 5)."}
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}],
                                                 "usage": usage})
            book = "playbook('Working model: UP paints (5, 5) - checked')\n" if "Your playbook:" in json.dumps(
                payload["messages"][0]) else ""
            return httpx.Response(200, json=tool_reply(book + "print('LIVE', x + 1)", call_id=f"v{n}",
                                                       usage=usage).json())
        code = "x = 41\nprint('DEF')" if n == 0 else "action([{'action':'UP'}])" if n % 2 else "print('LOOK')"
        return httpx.Response(200, json=tool_reply(code, call_id=f"c{n}", content="World model: " + "wall " * 3000,
                                                   usage=usage).json())

    def run(name, points, arms, budget):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"schema": 1, "points": points}))
        study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
                 "server": {"mtp_tokens": 1, "max_model_len": 262144}, "wall_seconds": 60, **V71,
                 "level_bench": {"start_points": str(path), "capacity_tokens": 4000000, "window": 196608,
                                 "token_budget": budget, "concurrency": 4, "passes": 1, "min_wall_seconds": 5,
                                 "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test", "arms": arms}}
        bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                       games_dir=None, output=tmp_path / name, server_dir=tmp_path,
                                       base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 120,
                                       log=lambda *a, **k: None, arcade=BenchArcade(stuck={"aa11-00000000"}),
                                       transport=httpx.MockTransport(handler))
        bench_attempts[name] = bench.attempts
        return bench.run()

    bench_attempts = {}
    report = await run("rec", [point], [{"name": "rec", "study": {"trim_record": {"stop_tokens": 40000}}}], 10 ** 6)
    (row,) = report["attempts"]
    assert row["outcome"] == "recorded", (row, report["task_errors"])
    folder = tmp_path / "rec/attempts" / row["label"]
    replies = [json.loads(line) for line in (folder / "script.jsonl").read_text().splitlines()]
    stop = json.loads((folder / "script_stop.json").read_text())
    assert stop["stop"]["requests"] == len(replies) >= 4 and stop["stop"]["estimate"] >= 40000
    assert stop["stop"]["steps"] == len(stop["actions"]) >= 2 and set(stop["actions"]) == {"ACTION1"}
    assert report["summary"]["attempts_completed"] == 0  # a recording is not a result

    live["on"] = True
    scripted = {**point, "script": {"replies": replies, **stop}}
    trim = {"budget_tokens": 38000}
    arms = [{"name": "drain", "study": {"trim": trim}},
            {"name": "long", "study": {"trim": {**trim, "summary": "long", "summary_target_tokens": 1500},
                                       "compaction_keep_fraction": 0.5, "compaction_summary_cap": 2048,
                                       "compaction_full_prefix": True, "compaction_thinking": "gameplay"}},
            {"name": "play", "study": {"trim": {**trim, "summary": "playbook"}, "compaction_keep_fraction": 0.1,
                                       "compaction_summary_cap": 2048,
                                       "hx_harness": {**V71["hx_harness"], "memo_style": 1}}},
            {"name": "none", "study": {}}]
    report = await run("bed", [scripted], arms, 800)
    rows = {r["arm"]: r for r in report["attempts"]}
    assert set(rows) == {"drain", "long", "play", "none"}, report["task_errors"]
    for arm, row in rows.items():
        assert row["script"]["ok"] and row["script"]["requests"] == len(replies), (arm, row["script"])
        assert row["script"]["steps"] == stop["stop"]["steps"] and row["outcome"] == "budget", row
        assert 800 <= row["completion_tokens"] < 1500 and row["actions"] == 0, row  # counted from the first live request
        log = (tmp_path / "bed/attempts" / row["label"] / "log.jsonl").read_text()
        assert "LIVE 42" in log, arm  # the interpreter was rebuilt by the replayed tool calls
    logs = {arm: (tmp_path / "bed/attempts" / rows[arm]["label"] / "log.jsonl").read_text() for arm in rows}
    assert '"fraction_compaction"' in logs["long"] and '"fraction_compaction"' not in logs["drain"]
    assert '"kind":"context_drain"' in logs["drain"] and '"kind":"context_drain"' not in logs["none"]
    assert rows["long"]["script"]["completion_tokens"] == rows["drain"]["script"]["completion_tokens"] == 300 * len(replies)
    assert rows["long"]["compactions"] == 1  # its summary's tokens are part of the 800
    # Arm F: the trim sends no request; it leaves the playbook (none yet: the recording has no notes either).
    assert '"kind":"playbook_handoff"' in logs["play"] and '"kind":"compaction"' not in logs["play"]
    play = [p for p in seen if "Your playbook:" in json.dumps(p["messages"][0])]
    assert play and all(p.get("tool_choice") != "none" for p in play)
    assert "is dropped and your playbook, as you last wrote it, takes its place; the newest 10% stays" in play[0]["messages"][0]["content"]
    first_play = json.dumps(min(play, key=lambda p: len(p["messages"]))["messages"][1])
    assert "[Context cut]" in first_play and "You have not written a playbook yet" in first_play and "Harness inventory" in first_play
    agent = next(a["agent"] for a in bench_attempts["bed"] if a["arm"] == "play")
    assert agent.sandbox.hx_state["playbook"] == "Working model: UP paints (5, 5) - checked"
    assert rows["play"]["compactions"] == 1
    summaries = [p for p in seen if p.get("tool_choice") == "none"]
    assert summaries and all("## Experiment log" in p["messages"][-1]["content"] and "about 1500 tokens"
                             in p["messages"][-1]["content"] for p in summaries)
    assert all("stay in the context verbatim" in p["messages"][-1]["content"] for p in summaries)
    # No live gameplay request before the script's end: the first one already carries every recorded reply.
    first = min((p for p in seen if p.get("tool_choice") != "none"), key=lambda p: len(p["messages"]))
    assert sum(1 for m in first["messages"] if m["role"] == "assistant") >= 1
    with pytest.raises(ValueError):
        level_bench.trim_settings({"summary": "short"})


def test_fraction_split_keeps_the_newest_observation_when_an_earlier_turn_has_no_cut():
    from arena.fraction_compaction import split_fraction

    def measure(messages, tools):
        return sum(len(m.get("content") or "") for m in messages)

    turn = []
    for k in range(30):
        turn += [{"role": "assistant", "content": "x" * 50, "tool_calls": [{"id": f"c{k}"}]},
                 {"role": "tool", "tool_call_id": f"c{k}", "content": "y" * 50}]
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "handoff"},
                {"role": "user", "content": "obs 1"}, *turn, {"role": "user", "content": "obs 2"}]
    source, tail, record = split_fraction(messages, [], measure, 0.1)
    assert tail[0]["content"] == "obs 1" and record["retained_tokens"] > 3000  # the whole turn: no cut inside it
    source, tail, record = split_fraction(messages, [], measure, 0.1, anchor=len(messages) - 1)
    assert tail == [messages[-1]] and source == messages[:-1] and record["retained_tokens"] == 5
    # the turn in play: its observation and the newest complete exchanges
    source, tail, record = split_fraction(messages[:-1], [], measure, 0.1, anchor=2)
    assert tail[0]["content"] == "obs 1" and tail[1]["role"] == "assistant" and 300 <= record["retained_tokens"] < 500


def test_playbook_trim_cuts_inside_a_long_turn_and_survives_one_left_behind():
    from types import SimpleNamespace
    from arena.compaction import CompactionMixin

    def measure(messages, tools):
        return sum(len(m.get("content") or "") for m in messages)

    def turn(n):
        out = []
        for k in range(n):
            out += [{"role": "assistant", "content": "x" * 50, "tool_calls": [{"id": f"c{k}"}]},
                    {"role": "tool", "tool_call_id": f"c{k}", "content": "y" * 50}]
        return out

    log = []
    owner = SimpleNamespace(estimate_prompt_tokens=measure, fraction_measure=measure, compaction_keep_fraction=0.1,
                            compaction_summary_cap=100, trim={"summary": "playbook"}, steps=3, first_trigger_done=False,
                            playbook_handoff=lambda: "playbook", is_observation=lambda m: m["content"].startswith("obs"),
                            workspace=SimpleNamespace(append=lambda kind, **k: log.append((kind, k))))
    agent = CompactionMixin()
    agent._owner, agent._context_budget_tokens = owner, 4000
    system, first = {"role": "system", "content": "s"}, {"role": "user", "content": "obs 1"}
    # the turn in play fills the context: the old rule kept all of it (it "fits"), the trim now cuts inside it
    handoff = {"role": "user", "content": "h" * 500}
    kept = agent._trim_messages_for_context([system, handoff, first, *turn(36)], tools=[])
    assert [m["content"] for m in kept[:3]] == ["s", "playbook", "obs 1"] and measure(kept, []) < 1000
    assert log[-1][1]["boundary"] == "turn_exchange"
    # an over-long earlier turn with only the new observation after it: that observation alone stays
    late = [system, handoff, first, *turn(40), {"role": "user", "content": "obs 2"}]
    kept = agent._trim_messages_for_context(late, tools=[])
    assert [m["content"] for m in kept] == ["s", "playbook", "obs 2"]


async def test_action_cap_ends_a_stuck_attempt(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    points = [{"id": point_id("bb22-00000000", 0), "game": "bb22-00000000", "split": "dev", "level_index": 0,
               "level": 1, "total_levels": 5, "baseline_actions": 1, "prefix": [], "prefix_actions": 0,
               "board_hash": board_hash(board)}]
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    calls = []

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
        calls.append(1)
        reply = tool_reply("action([{'action':'UP'}])", call_id=f"c{len(calls)}",
                           usage={"completion_tokens": 100, "prompt_tokens": 4000})
        return httpx.Response(200, json=reply.json())

    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                             "token_budget": 50000, "concurrency": 1, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1,
                             "action_cap": {"multiple": 2, "floor": 3}}}
    assert level_bench.action_cap(level_bench.settings(study["level_bench"]), {"baseline_actions": 10}) == 20
    by_level = level_bench.settings({"token_budget": 9, "token_budget_by_level": {"2": 70}})
    assert [level_bench.token_budget(by_level, {"level": n}) for n in (1, 2)] == [9, 70]
    by_point = level_bench.settings({"token_budget": 9, "token_budget_by_level": {"2": 70},
                                     "token_budget_by_point": {"aa11-L2": 300}})
    assert [level_bench.token_budget(by_point, {"id": i, "level": 2}) for i in ("aa11-L2", "bb22-L2")] == [300, 70]
    with pytest.raises(ValueError, match="token_budget_by_level"):
        level_bench.settings({"token_budget_by_level": {"two": 70}})
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / "out", server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 60,
                                   log=lambda *a, **k: None, arcade=BenchArcade(stuck={"bb22-00000000"}),
                                   transport=httpx.MockTransport(handler))
    row = (await bench.run())["attempts"][0]
    assert (row["outcome"], row["total_actions"], row["action_cap"], row["cleared"]) == ("action_cap", 3, 3, False)
    with pytest.raises(ValueError, match="action_cap"):
        level_bench.settings({"action_cap": {"multiple": 0, "floor": 3}})


def test_rl0_level_test_kind_and_probe_settings(tmp_path):
    from arena import train_probe

    cfg = train_probe.settings({"variants": [{"storage": "bf16", "lengths": [8, 16], "share": 0.5}]})
    command = train_probe.job_command("/m", cfg["variants"][0], cfg, tmp_path / "o.json", 5.0)
    assert train_probe.runtime_python(tmp_path) == sys.executable
    venv = tmp_path / "python"
    venv.write_text("")
    (tmp_path / "vllm-server-identity.json").write_text(json.dumps({"argv": [str(venv), "-m", "vllm.entrypoints.cli.main"]}))
    assert train_probe.runtime_python(tmp_path) == str(venv)
    assert train_probe.job_command("/m", cfg["variants"][0], cfg, tmp_path / "o.json", 5.0, str(venv))[0] == str(venv)
    assert command[2:4] == ["-m", "arena.train_probe_job"] and command[command.index("--lengths") + 1:][:2] == ["8", "16"]
    with pytest.raises(ValueError):
        train_probe.settings({"variants": [{"storage": "int8", "lengths": [8]}]})
    import ast

    ns = {"Path": Path, "hx4_study": lambda: {"notebook_seconds": 6300}}
    for node in ast.parse((ROOT / "scripts/modal_validation.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("validation_timeout", "arena_arguments"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "modal_validation", "exec"), ns)
    assert ns["validation_timeout"]("arena", "rl0-level") == 6600
    assert ns["arena_arguments"]("rl0-level") == ["--trial", "level-bench", "--trial", "train-probe"]


async def test_arm_thinking_levels_and_passes(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    points = [{"id": point_id("aa11-00000000", i), "game": "aa11-00000000", "split": "dev", "level_index": i,
               "level": i + 1, "total_levels": 5, "baseline_actions": 2, "prefix": [], "prefix_actions": 0,
               "board_hash": board_hash(board)} for i in (0,)] + [
              {"id": "aa11-L2", "game": "aa11-00000000", "split": "dev", "level_index": 0, "level": 2,
               "total_levels": 5, "baseline_actions": 2, "prefix": [], "prefix_actions": 0,
               "board_hash": board_hash(board)}]
    arms = [{"name": "think"}, {"name": "nothink", "thinking": False, "levels": [1], "passes": 1}]
    work = level_bench.plan(points, 3, 0, arms)
    assert sorted((p, pt["level"], a["name"]) for p, pt, a in work if a["name"] == "nothink") == [(0, 1, "nothink")]
    assert sum(a["name"] == "think" for _, _, a in work) == 6
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    seen = []

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
        seen.append(json.loads(request.content)["chat_template_kwargs"]["enable_thinking"])
        reply = tool_reply("action([{'action':'UP'}])", call_id=f"c{len(seen)}",
                           usage={"completion_tokens": 100, "prompt_tokens": 4000})
        return httpx.Response(200, json=reply.json())

    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                             "token_budget": 5000, "concurrency": 2, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "arms": arms}}
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / "out", server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 60,
                                   log=lambda *a, **k: None, arcade=BenchArcade(),
                                   transport=httpx.MockTransport(handler))
    rows = (await bench.run())["attempts"]
    assert sorted((r["point"], r["arm"]) for r in rows) == [("aa11-L1", "nothink"), ("aa11-L1", "think"),
                                                            ("aa11-L2", "think")]
    assert sorted(seen) == [False, True, True]
    with pytest.raises(ValueError, match="thinking"):
        level_bench.settings({"arms": [{"name": "x", "thinking": "off"}]})


async def test_rollout_record_stores_requests_tokens_and_logprobs(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    points = [{"id": point_id("aa11-00000000", 0), "game": "aa11-00000000", "split": "dev", "level_index": 0,
               "level": 1, "total_levels": 5, "baseline_actions": 2, "prefix": [], "prefix_actions": 0,
               "board_hash": board_hash(board)}]
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    sent = []

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
        sent.append(json.loads(request.content))
        reply = tool_reply("action([{'action':'UP'}])", call_id="live",
                           usage={"completion_tokens": 3, "prompt_tokens": 4000}).json()
        reply["prompt_token_ids"] = list(range(40))
        reply["choices"][0]["token_ids"] = [11, 12, 13]
        reply["choices"][0]["logprobs"] = {"content": [
            {"token": f"token_id:{t}", "logprob": v, "bytes": [],
             "top_logprobs": [{"token": f"token_id:{t}", "logprob": v, "bytes": []}]} for t, v in ((11, -0.1), (12, -0.7), (13, 0.0))]}
        return httpx.Response(200, json=reply)

    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                             "token_budget": 5000, "concurrency": 1, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "rollout": {"top_logprobs": 5}}}
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / "out", server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 60,
                                   log=lambda *a, **k: None, arcade=BenchArcade(),
                                   transport=httpx.MockTransport(handler))
    report = await bench.run()
    (row,) = report["attempts"]
    assert row["outcome"] == "cleared" and row["clear_replay"]["ok"] is True, row
    assert (sent[0]["logprobs"], sent[0]["top_logprobs"], sent[0]["return_token_ids"]) == (True, 5, True)
    assert row["rollout"]["replies"] == 1 and row["rollout"]["tokens"] == 3 and row["rollout"]["errors"] == 0
    assert report["rollout"]["clears_replayed"] == 1
    folder = tmp_path / "out/attempts" / row["label"]
    (stored,) = [json.loads(line) for line in (folder / "rollout.jsonl").read_text().splitlines()]
    assert stored["tokens"] == [11, 12, 13] and stored["logprobs"] == [-0.1, -0.7, 0.0] and stored["prompt_digest"]
    assert stored["messages"][0]["role"] == "system" and stored["tools"] and stored["template"]["enable_thinking"] is True
    assert stored["sampling"] == {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 4096}
    assert "prompt_token_ids" not in (folder / "log.jsonl").read_text()


def test_sampling_setting_is_validated():
    card = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
    assert level_bench.settings({"sampling": card})["sampling"] == card
    assert level_bench.settings({})["sampling"] is None
    for bad in ({}, {"seed": 1}, {"top_k": True}, {"temperature": "1"}):
        with pytest.raises(ValueError):
            level_bench.settings({"sampling": bad})


def test_level_token_cap_ends_a_play_through_attempt_stuck_on_one_level():
    from types import SimpleNamespace

    from duck_transfer.agent import GameplayRequestExpired

    assert level_bench.settings({"level_token_cap": 60000})["level_token_cap"] == 60000
    with pytest.raises(ValueError):
        level_bench.settings({"level_token_cap": 0})
    agent = level_bench.LevelBenchAgent.__new__(level_bench.LevelBenchAgent)
    tokens, events = {"n": 0}, []
    agent.game = SimpleNamespace(clock=SimpleNamespace(completion_tokens=lambda: tokens["n"]))
    agent.workspace = SimpleNamespace(append=lambda kind, **kw: events.append((kind, kw)))
    agent.bench_setup = None
    agent.bench_outcome = agent.bench_enter = agent.bench_script = agent.bench_action_cap = None
    agent.bench_lead, agent.bench_budget, agent.steps, agent.bench_offset = 0, 10 ** 6, 0, 0
    agent.bench_through, agent.bench_level_cap = True, 1000
    agent.bench_levels = [{"level": 1, "completion_tokens": 700, "steps": 5}]
    tokens["n"] = 1600
    agent.bench_gate()  # 900 tokens on level 2
    tokens["n"] = 1700
    with pytest.raises(GameplayRequestExpired):
        agent.bench_gate()
    assert agent.bench_outcome == "level_cap" and events[-1][1]["level"] == 2


def test_shared_slots_priority_and_accounting():
    pool = level_bench.SharedSlots(2)
    a, b = object(), object()
    pool.want(a, 8000, 3)
    pool.want(b, 32000, 1)
    assert not pool.take(a)  # the larger budget goes first
    assert pool.take(b) and not pool.take(b)  # b has no more work
    assert pool.take(a) and not pool.take(a)  # the pool is full
    pool.give()
    assert pool.take(a) and pool.peak == 2
    pool.done(b)
    pool.done(a)
    assert pool.pending == {}


def test_shared_slots_kv_fill_limits_live_context():
    class S:
        def __init__(self, est, cap=4000):
            self.next_estimate, self.cap = est, cap

    pool = level_bench.SharedSlots(10, kv_fill=0.8, new_tokens=20000)
    pool.capacity = 100000
    key = object()
    pool.want(key, 0, 10)
    assert pool.take(key)  # nothing live yet
    s1 = S(0)
    pool.enter(s1)  # before its first request: the default allowance
    assert pool.load() == 20000
    s1.next_estimate = 36000  # live 40000
    assert pool.take(key)  # 40000 + 40000 <= 80000
    s2 = S(36000)
    pool.enter(s2)
    assert not pool.take(key) and pool.kv_waits == 1  # 80000 + 40000 > 80000
    pool.give(s2)
    assert pool.take(key) and pool.used == 2 and pool.peak_load == 80000
    blocks = level_bench.SharedSlots(10, kv_fill=0.8, new_tokens=20000)  # a block budget: 1 unit per 1000 tokens
    blocks.capacity, blocks.budget = 100, type("B", (), {"cost": staticmethod(lambda t: -(-int(t) // 1000))})()
    blocks.want(key, 0, 10)
    for est in (36000, 36000):
        assert blocks.take(key)
        blocks.enter(S(est))
    assert blocks.load() == 80 and not blocks.take(key)
    plain = level_bench.SharedSlots(10)
    plain.want(key, 0, 3)
    plain.enter(S(10 ** 7))
    assert plain.take(key)  # no kv_fill: the count alone


class ProjSlot:
    def __init__(self, est=0, floor=0, generated=0, budget=30000, window=131072, cap=16384):
        self.next_estimate, self.floor_tokens, self.cap, self.window, self._stack = est, floor, cap, window, []
        self.generated, self.budget = generated, budget
        self.progress = lambda: (self.generated, self.budget)


def test_kv_project_counts_end_of_try_context():
    """mg.kv_project (ovf2 R1): a live try counts at its end-of-try context (restored prefix or latest prompt + budget
    left x growth, capped at ctx_frac x (window - cap)) x scale, not at its next request."""
    pool = level_bench.SharedSlots(10, kv_fill=1.0, project={"growth": 2.0, "ctx_frac": 0.6, "scale": 0.5})
    pool.capacity = 200000
    s = ProjSlot(floor=40000, generated=0, budget=10000)
    assert pool.project_tokens(s) == (40000 + 20000) * 0.5  # restored prefix + 10000 x 2, under the cap
    s.next_estimate, s.generated = 50000, 8000
    assert pool.project_tokens(s) == (50000 + 4000) * 0.5  # latest prompt + budget left
    s.generated = 12000
    assert pool.project_tokens(s) == 50000 * 0.5  # past its budget: the context now
    big = ProjSlot(est=100000, generated=0, budget=30000)
    assert pool.project_tokens(big) == 0.6 * (131072 - 16384) * 0.5  # at the window: the saw-tooth cap
    assert level_bench.SharedSlots(2, kv_fill=1.0, project=True).project == level_bench.KV_PROJECT


def test_kv_project_gate_admits_on_projection():
    key = object()
    pool = level_bench.SharedSlots(10, kv_fill=1.0, project={"growth": 1.0, "ctx_frac": 1.0, "scale": 1.0})
    pool.capacity = 100000
    pool.want(key, 0, 10)
    assert pool.take(key)
    a = ProjSlot(est=0, floor=10000, budget=30000)  # projects to 40000, though its next request is ~26000
    pool.enter(a)
    assert a.project_start == 40000 and pool.load() == 40000
    assert pool.take(key)  # 40000 + the median start projection 40000 <= 100000
    pool.enter(ProjSlot(est=0, floor=10000, budget=30000))
    assert not pool.take(key) and pool.kv_waits == 1  # 80000 + 40000 > 100000: the old rule (prompt + cap) admitted it
    old = level_bench.SharedSlots(10, kv_fill=1.0)
    old.capacity = 100000
    old.want(key, 0, 10)
    for _ in range(2):
        assert old.take(key)
        old.enter(ProjSlot(est=10000))
    assert old.take(key)  # 2 x 26384 + 26384 fits: the next-request projection
    a.generated = 30000  # a's budget is spent: it now counts at its context
    a.next_estimate = 20000
    assert pool.load() == 20000 + 40000 and pool.take(key)
    assert 0.99 < pool.project_peak <= 1.0


def test_kv_project_scale_sample_and_block_capacity():
    pool = level_bench.SharedSlots(4, kv_fill=1.0, project=True)
    pool.capacity = 1000000
    a, b = ProjSlot(est=300000), ProjSlot(est=100000)
    pool.enter(a); pool.enter(b)
    assert pool.scale_sample(0.5) is None  # under half the live tries in flight... (none)
    a._stack = [1]
    assert pool.scale_sample(0.24) == 0.24 * 1000000 / 300000 and pool.scale_samples == [0.8]
    blocks = level_bench.SharedSlots(4, kv_fill=1.0, project=True)
    blocks.capacity = 900
    blocks.budget = type("B", (), {"layout": {"num_blocks": 1000, "costs": [{"tokens": 4096, "blocks": 2}]}})()
    assert blocks.capacity_tokens() == 2048000  # the whole pool in tokens, not the budget's margin


async def test_benches_share_slots_and_finish(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    inflight, peak = [0], [0]

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
        reply = tool_reply("action([{'action':'UP'}])", call_id="c", usage={"completion_tokens": 3000, "prompt_tokens": 4000})
        return httpx.Response(200, json=reply.json())

    async def one(name, games, budget):
        points = [{"id": point_id(g, 0), "game": g, "split": "dev", "level_index": 0, "level": 1, "total_levels": 5,
                   "baseline_actions": 2, "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board)} for g in games]
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"schema": 1, "points": points}))
        study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
                 "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
                 "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                                 "token_budget": budget, "concurrency": 4, "passes": 2, "min_wall_seconds": 5,
                                 "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test",
                                 "token_budget_by_point": {f"{games[0].split('-')[0]}-game": budget}}}
        bench = level_bench.LevelBench(study, {"name": name, "kind": "level-bench"}, source=ROOT, games_dir=None,
                                       output=tmp_path / name, server_dir=tmp_path, base_url="http://127.0.0.1:9/v1",
                                       deadline_epoch=time.time() + 60, log=lambda line, **kw: None,
                                       arcade=BenchArcade(stuck=set()), transport=httpx.MockTransport(handler))
        return await bench.run()

    monkeypatch.setattr(level_bench, "SHARED_SLOTS", level_bench.SharedSlots(1))
    first, second = await asyncio.gather(one("b1", ["aa11-00000000"], 5000), one("b2", ["bb22-00000000"], 8000))
    pool = level_bench.SHARED_SLOTS
    assert len(first["attempts"]) == 2 and len(second["attempts"]) == 2
    assert all(r["outcome"] == "cleared" for r in first["attempts"] + second["attempts"])
    assert pool.peak == 1 and pool.used == 0 and pool.pending == {}

    # kv_fill: one gate for both benches; an 8000-token live context per attempt (4000 prompt + 4096 cap) in a 200000 x 0.1 budget
    # admits two attempts at once although the count allows four
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", level_bench.SharedSlots(4, kv_fill=0.1, new_tokens=8000))
    first, second = await asyncio.gather(one("b3", ["aa11-00000000"], 5000), one("b4", ["bb22-00000000"], 8000))
    pool = level_bench.SHARED_SLOTS
    assert all(r["outcome"] == "cleared" for r in first["attempts"] + second["attempts"])
    assert pool.gate is not None and pool.capacity == 200000 and pool.peak == 2 and pool.kv_waits > 0
    assert pool.used == 0 and not pool.live and pool.gate.reserved == 0

    # kv_project: the same gate on projected end-of-try contexts; each live attempt reads its budget progress
    seen = []
    project = level_bench.SharedSlots(4, kv_fill=0.1, new_tokens=8000, project={"growth": 1.0, "ctx_frac": 1.0, "scale": 1.0})
    tokens = project.project_tokens

    def spy(slot):
        seen.append(slot.progress())
        return tokens(slot)

    project.project_tokens = spy
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", project)
    first, second = await asyncio.gather(one("b5", ["aa11-00000000"], 5000), one("b6", ["bb22-00000000"], 8000))
    assert all(r["outcome"] == "cleared" for r in first["attempts"] + second["attempts"])
    assert project.used == 0 and not project.live and project.peak >= 1 and project.project_peak > 0
    assert seen and {b for _, b in seen} <= {5000, 8000} and all(g >= 0 for g, _ in seen)



async def test_time_share_second_pool_takes_attempts_only_while_active(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    hits = {1: 0, 2: 0}

    def handler_for(server):
        def handler(request):
            if request.url.path.endswith("/tokenize"):
                return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
            hits[server] += 1
            reply = tool_reply("action([{'action':'UP'}])", call_id="c", usage={"completion_tokens": 3000, "prompt_tokens": 4000})
            return httpx.Response(200, json=reply.json())
        return handler

    async def one(name, games, budget, passes=2):
        points = [{"id": point_id(g, 0), "game": g, "split": "dev", "level_index": 0, "level": 1, "total_levels": 5,
                   "baseline_actions": 2, "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board)} for g in games]
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"schema": 1, "points": points}))
        study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
                 "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
                 "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                                 "token_budget": budget, "concurrency": 4, "passes": passes, "min_wall_seconds": 5,
                                 "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test",
                                 "token_budget_by_point": {f"{games[0].split('-')[0]}-game": budget}}}
        bench = level_bench.LevelBench(study, {"name": name, "kind": "level-bench"}, source=ROOT, games_dir=None,
                                       output=tmp_path / name, server_dir=tmp_path, base_url="http://127.0.0.1:9/v1",
                                       deadline_epoch=time.time() + 60, log=lambda line, **kw: None,
                                       arcade=BenchArcade(stuck=set()), transport=httpx.MockTransport(handler_for(1)))
        bench.transport2 = httpx.MockTransport(handler_for(2))
        return await bench.run()

    monkeypatch.setattr(level_bench, "SERVER2_URL", "http://127.0.0.1:8/v1")
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", level_bench.SharedSlots(1))
    pool2 = level_bench.SharedSlots(1)
    pool2.active = False
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", pool2)
    first, second = await asyncio.gather(one("t1", ["aa11-00000000"], 5000), one("t2", ["bb22-00000000"], 8000))
    assert hits[2] == 0 and pool2.served == 0  # inactive: everything on server 1
    assert all(r["outcome"] == "cleared" for r in first["attempts"] + second["attempts"])

    hits.update({1: 0, 2: 0})
    pool2 = level_bench.SharedSlots(1)
    pool2.capacity = 150000  # its own server's pool, set by the loop from that server's cache layout
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", pool2)
    pool1 = level_bench.SharedSlots(1)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", pool1)
    first, second = await asyncio.gather(one("t3", ["aa11-00000000"], 5000, 3), one("t4", ["bb22-00000000"], 8000, 3))
    assert all(r["outcome"] == "cleared" for r in first["attempts"] + second["attempts"])
    assert len(first["attempts"]) == 3 and len(second["attempts"]) == 3
    assert hits[1] > 0 and hits[2] > 0 and pool2.served > 0 and pool1.peak == 1 and pool2.peak == 1
    assert {r["server_index"] for r in first["attempts"] + second["attempts"]} == {1, 2}
    assert pool2.gate.capacity == 150000
    for p_ in (pool1, pool2):
        assert p_.used == 0 and p_.pending == {} and not p_.live


async def test_a_bench_started_before_the_second_server_joins_its_pool(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    hits = {1: 0, 2: 0}

    def handler_for(server, delay):
        async def handler(request):
            if request.url.path.endswith("/tokenize"):
                return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
            hits[server] += 1
            await asyncio.sleep(delay)
            reply = tool_reply("action([{'action':'UP'}])", call_id="c", usage={"completion_tokens": 3000, "prompt_tokens": 4000})
            return httpx.Response(200, json=reply.json())
        return handler

    points = [{"id": point_id("aa11-00000000", 0), "game": "aa11-00000000", "split": "dev", "level_index": 0, "level": 1,
               "total_levels": 5, "baseline_actions": 2, "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board)}]
    path = tmp_path / "late.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384, "token_budget": 5000,
                             "concurrency": 4, "passes": 6, "min_wall_seconds": 5, "launch_cutoff_seconds": 0,
                             "progress_seconds": 1, "serving": "test", "token_budget_by_point": {"aa11-game": 5000}}}
    bench = level_bench.LevelBench(study, {"name": "late", "kind": "level-bench"}, source=ROOT, games_dir=None,
                                   output=tmp_path / "late", server_dir=tmp_path, base_url="http://127.0.0.1:9/v1",
                                   deadline_epoch=time.time() + 60, log=lambda line, **kw: None,
                                   arcade=BenchArcade(stuck=set()), transport=httpx.MockTransport(handler_for(1, 0.2)))
    bench.transport2 = httpx.MockTransport(handler_for(2, 0.0))
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", level_bench.SharedSlots(1))
    monkeypatch.setattr(level_bench, "SHARED_SLOTS2", None)
    monkeypatch.setattr(level_bench, "SERVER2_URL", None)

    async def boot_later():
        await asyncio.sleep(0.1)
        level_bench.SHARED_SLOTS2, level_bench.SERVER2_URL = level_bench.SharedSlots(2), "http://127.0.0.1:8/v1"

    result, _ = await asyncio.gather(bench.run(), boot_later())
    assert len(result["attempts"]) == 6 and hits[2] > 0
    assert {r["server_index"] for r in result["attempts"]} == {1, 2}

async def test_a_bench_on_a_version_server_2_lacks_stays_on_server_1(tmp_path, monkeypatch):
    """v7 Mode B: server 2 holds only SERVER2_VERSIONS; a bench pinned to another version never places a try there."""
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    hits = {1: 0, 2: 0}

    def handler_for(server, delay):
        async def handler(request):
            if request.url.path.endswith("/tokenize"):
                return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
            hits[server] += 1
            await asyncio.sleep(delay)
            reply = tool_reply("action([{'action':'UP'}])", call_id="c", usage={"completion_tokens": 3000, "prompt_tokens": 4000})
            return httpx.Response(200, json=reply.json())
        return handler

    points = [{"id": point_id("aa11-00000000", 0), "game": "aa11-00000000", "split": "dev", "level_index": 0, "level": 1,
               "total_levels": 5, "baseline_actions": 2, "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board)}]
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    monkeypatch.setattr(level_bench, "SERVER2_URL", "http://127.0.0.1:8/v1")
    monkeypatch.setattr(level_bench, "SERVER2_VERSIONS", {5})
    seen = {}
    for version in (3, 5):
        hits.update({1: 0, 2: 0})
        study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
                 "policy_version": version, "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
                 "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384, "token_budget": 5000,
                                 "concurrency": 4, "passes": 6, "min_wall_seconds": 5, "launch_cutoff_seconds": 0,
                                 "progress_seconds": 1, "serving": "test", "token_budget_by_point": {"aa11-game": 5000}}}
        bench = level_bench.LevelBench(study, {"name": f"v{version}", "kind": "level-bench"}, source=ROOT, games_dir=None,
                                       output=tmp_path / f"v{version}", server_dir=tmp_path, base_url="http://127.0.0.1:9/v1",
                                       deadline_epoch=time.time() + 60, log=lambda line, **kw: None,
                                       arcade=BenchArcade(stuck=set()), transport=httpx.MockTransport(handler_for(1, 0.2)))
        bench.transport2 = httpx.MockTransport(handler_for(2, 0.0))
        monkeypatch.setattr(level_bench, "SHARED_SLOTS", level_bench.SharedSlots(1))
        monkeypatch.setattr(level_bench, "SHARED_SLOTS2", level_bench.SharedSlots(2))
        result, = await asyncio.gather(bench.run())
        result = result[0] if isinstance(result, tuple) else result
        seen[version] = ({r["server_index"] for r in result["attempts"]}, hits[2], len(result["attempts"]))
    assert seen[3] == ({1}, 0, 6)
    assert seen[5][0] == {1, 2} and seen[5][1] > 0


def _admission_sim(fill, slots=128, attempts=400, cap=16384, gap=3.0, setup=5.0):
    """Shared-slot admission on the real B300 Flash cache layout (132 GiB, 6,990 blocks) and real recorded tries
    (prompt, completion, seconds per request: rl1-screen-mg2 rec-re86, ev5g base sb26-L2 / re86-L2). Returns the
    peak and mean block need of the live tries' resident contexts against the pool."""
    import heapq
    import itertools

    from arena.cache import BlockBudget

    fixtures = Path(__file__).parent / "fixtures"
    layout = json.loads((fixtures / "b300_flash_cache_layout.json").read_text())
    budget = BlockBudget(layout, 0.96)
    traces = [t for v in json.loads((fixtures / "mg_live_traces.json").read_text()).values() for t in v]
    pool = level_bench.SharedSlots(slots, kv_fill=fill)
    pool.capacity, pool.budget = budget.capacity, budget
    key, events, seq, live, samples, state = object(), [], itertools.count(), {}, [], {"started": 0, "now": 0.0}
    pool.want(key, 0, attempts)

    class Slot:
        def __init__(self):
            self.next_estimate, self.cap = 0, cap

    def admit():
        while state["started"] < attempts and pool.take(key):
            s = Slot()
            pool.enter(s)
            live[s] = 0
            heapq.heappush(events, (state["now"] + setup, next(seq), "req", s, traces[state["started"] % len(traces)], 0))
            state["started"] += 1

    admit()
    while events:
        now, _, kind, s, trace, i = heapq.heappop(events)
        state["now"] = now
        if kind == "req":
            s.next_estimate, live[s] = trace[i][0], trace[i][0] + trace[i][1]
            heapq.heappush(events, (now + trace[i][2], next(seq), "done", s, trace, i))
        elif i + 1 < len(trace):
            heapq.heappush(events, (now + gap, next(seq), "req", s, trace, i + 1))
        else:
            pool.give(s)
            live.pop(s)
            admit()
        samples.append((now, len(live), sum(budget.cost(t) if t else 10 for t in live.values())))
    end = samples[-1][0]
    mid = [x for x in samples if 0.2 * end < x[0] < 0.7 * end]
    return {"peak": max(n for *_, n in samples) / layout["num_blocks"],
            "mean": sum(n for *_, n in mid) / len(mid) / layout["num_blocks"],
            "live": sum(x[1] for x in mid) / len(mid), "budget": budget, "pool": pool}


def test_shared_slots_admission_on_real_cache_layout_and_traces():
    # uniform live contexts: the admitted count is the cache fit (6,710 usable blocks; 32K = 50, 64K = 90 blocks)
    from arena.cache import BlockBudget

    layout = json.loads((Path(__file__).parent / "fixtures" / "b300_flash_cache_layout.json").read_text())
    budget = BlockBudget(layout, 0.96)
    for context, expect in ((32768, 128), (65536, 6710 // budget.cost(65536))):
        pool = level_bench.SharedSlots(128, kv_fill=1.0)
        pool.capacity, pool.budget = budget.capacity, budget
        key = object()
        pool.want(key, 0, 1000)
        n = 0
        while pool.take(key):
            slot = type("S", (), {})()
            slot.next_estimate, slot.cap = context - 16384, 16384
            pool.enter(slot)
            n += 1
        assert n == expect, (context, n)
    assert 6710 // budget.cost(65536) in range(70, 78)
    # real tries: the loop's fill (0.9) keeps every live context resident (peak under the pool, mean >= 60%); 1.0 peaks
    # ~4% over; without the KV rule 128 tries overflow the pool by ~20% (screen mg6-4's thrash)
    loop, screen, unlimited = _admission_sim(0.9), _admission_sim(1.0), _admission_sim(None)
    assert loop["peak"] <= 1.0 and loop["mean"] >= 0.6 and loop["live"] >= 80, loop
    assert screen["peak"] <= 1.06 and screen["mean"] >= 0.65, screen
    assert unlimited["peak"] >= 1.15, unlimited


async def test_finished_tries_are_kept_and_a_relaunch_skips_them(tmp_path, monkeypatch):
    # gate item 4: a restarted screen keeps every finished try
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
        calls.append(1)
        return httpx.Response(200, json=tool_reply("action([{'action':'UP'}])", call_id=f"c{len(calls)}",
                                                   usage={"completion_tokens": 3000, "prompt_tokens": 4000}).json())

    def bench(out, budget=5000):
        study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
                 "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
                 "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                                 "token_budget": budget, "concurrency": 2, "passes": 1, "min_wall_seconds": 5,
                                 "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test", "rollout": {}}}
        return level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT, games_dir=None,
                                      output=tmp_path / out / "rec-x", server_dir=tmp_path,
                                      base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 60, log=lambda *a, **k: None,
                                      arcade=BenchArcade(stuck={"bb22-00000000"}), transport=httpx.MockTransport(handler))

    monkeypatch.setattr(level_bench, "TRY_KEEP", tmp_path / "keep")
    first = await bench("one").run()
    played = len(calls)
    assert len((tmp_path / "keep/rec-x/tries.jsonl").read_text().splitlines()) == 2
    monkeypatch.setattr(level_bench, "TRY_KEEP", None)
    monkeypatch.setattr(level_bench, "TRY_REUSE", [tmp_path / "keep"])
    pool = level_bench.SharedSlots(4)  # v6h: reused tries leave the shared pool's pending count on every bench
    seen = []
    remaining = pool.remaining
    pool.remaining = lambda key, count: (seen.append(count), remaining(key, count))
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", pool)
    again = await bench("two").run()
    assert seen and min(seen) == 0  # both tries reused: nothing left pending for lower benches to wait on
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    assert len(calls) == played  # nothing replayed
    assert [r["label"] for r in again["attempts"]] == [r["label"] for r in first["attempts"]]
    assert again["summary"]["clears"] == 1 and (tmp_path / "two/rec-x/attempts" / first["attempts"][0]["label"]).is_dir()
    await bench("three", budget=7000).run()  # another plan (budget): played anew
    assert len(calls) > played
    monkeypatch.setattr(level_bench, "TRY_REUSE_SKIP", {"rec-x"})
    before = len(calls)
    await bench("four").run()  # its recording was replayed in this run: no reuse
    assert len(calls) > before


async def test_restore_sends_recorded_text_when_only_the_wall_clock_differs(tmp_path):
    from types import SimpleNamespace

    from arena import rollout_record as rr

    S = {"role": "system", "content": "sys"}
    U = lambda s: {"role": "user", "content": f"Stuck check: 3 turns ({s}s of active play), 2 consecutive actions"}
    A = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}
    T = {"role": "tool", "content": "ok", "tool_call_id": "c1"}
    sent = []

    class Inner:
        async def post(self, url, **kw):
            sent.append(kw["json"])
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}},
                                  request=httpx.Request("POST", str(url)))

    log = tmp_path / "a" / "log.jsonl"
    log.parent.mkdir()
    agent = SimpleNamespace(workspace=SimpleNamespace(log=str(log), append=lambda *a, **k: None), steps=0)
    http = rr.RolloutHTTP(Inner(), agent, {})
    http.record = lambda payload, data: None
    rows = [{"n": 0, "keep": 0, "messages": [S, U(10)], "tools_now": None, "usage": {}, "reply": A, "finish_reason": "stop"},
            {"n": 1, "keep": 2, "messages": [A, T], "tools_now": None, "usage": {}, "reply": A, "finish_reason": "stop"}]
    http.restore = {"request": 1, "rows": rows}
    url = "http://x/v1/chat/completions"
    await http.post(url, json={"messages": [S, U(12)]})
    await http.post(url, json={"messages": [S, U(12), A, T]})
    assert http.restore_result["diffs"] == [] and sent[-1]["messages"][1] == U(10)
    await http.post(url, json={"messages": [S, U(12), A, T, A, U(30)]})  # a later notice keeps its own seconds
    assert sent[-1]["messages"][1] == U(10) and sent[-1]["messages"][-1] == U(30)
    other = rr.RolloutHTTP(Inner(), agent, {})  # any other difference is still a diff
    other.restore = {"request": 1, "rows": [{**rows[0], "messages": [S, {"role": "user", "content": "else (10s of active play)"}]}, rows[1]]}
    await other.post(url, json={"messages": [S, U(12)]})
    assert other.restore_result["diffs"]


def test_tail_order_longest_expected_first_and_plan_order_without_history():
    work = [(0, {"id": p}, None) for p in ("a", "b", "c", "a", "d")]
    assert level_bench.tail_order(work, None) == [0, 1, 2, 3, 4]
    assert level_bench.tail_order(work, {}) == [0, 1, 2, 3, 4]
    assert level_bench.tail_order(work, {"x": 9}) == [0, 1, 2, 3, 4]  # no start of this plan has history
    # a 1000, b 9000, c 5000; d unknown counts as the known median (5000): ties keep the plan order
    assert level_bench.tail_order(work, {"a": 1000, "b": 9000, "c": 5000}) == [1, 2, 4, 0, 3]


def test_shared_slots_hold_until_every_bench_registers():
    pool = level_bench.SharedSlots(4)
    a, b = object(), object()
    pool.hold(2)
    pool.want(a, 0, 2)
    assert not pool.take(a)  # b has not registered yet
    pool.want(b, -10 ** 9, 2)  # a held-out bench
    assert pool.take(a) and pool.take(a) and pool.take(b)
    late = level_bench.SharedSlots(4)
    late.hold(3, seconds=0)  # a bench that never registers does not stall the round past the hold
    late.want(a, 0, 1)
    assert late.take(a)
    plain = level_bench.SharedSlots(4)  # no hold: today's behaviour
    plain.want(a, 0, 1)
    assert plain.take(a)


class _OrderSlots(level_bench.SharedSlots):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = []

    def enter(self, slot):
        self.started.append(slot.game.game_id)
        super().enter(slot)


async def _tail_bench(tmp_path, name, games, budget, *, passes=2, tail=None):
    board = np.zeros((64, 64), dtype=np.uint8)

    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
        reply = tool_reply("action([{'action':'UP'}])", call_id="c", usage={"completion_tokens": 3000, "prompt_tokens": 4000})
        return httpx.Response(200, json=reply.json())

    points = [{"id": point_id(g, 0), "game": g, "split": "dev", "level_index": 0, "level": 1, "total_levels": 5,
               "baseline_actions": 2, "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board)} for g in games]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                             "token_budget": budget, "concurrency": 8, "passes": passes, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test",
                             "token_budget_by_point": {f"{games[0].split('-')[0]}-game": budget}}}
    if tail is not None:
        study["tail_fill"] = tail
    bench = level_bench.LevelBench(study, {"name": name, "kind": "level-bench"}, source=ROOT, games_dir=None,
                                   output=tmp_path / name, server_dir=tmp_path, base_url="http://127.0.0.1:9/v1",
                                   deadline_epoch=time.time() + 60, log=lambda line, **kw: None,
                                   arcade=BenchArcade(stuck=set()), transport=httpx.MockTransport(handler))
    return await bench.run()


async def test_tail_fill_off_keeps_the_plan_launch_order(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    games = ["aa11-00000000", "bb22-00000000", "cc33-00000000"]
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", _OrderSlots(1))
    off = await _tail_bench(tmp_path, "off", games, 5000)
    started_off = list(level_bench.SHARED_SLOTS.started)
    assert started_off == [r["label"] for r in sorted(off["attempts"], key=lambda r: r["index"])]
    # the key on without history: the same labels in the same order
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", _OrderSlots(1))
    await _tail_bench(tmp_path, "on", games, 5000, tail={"expected": {}, "last": False})
    assert level_bench.SHARED_SLOTS.started == started_off


async def test_tail_fill_longest_first_and_held_out_last_round_completes(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    games = ["aa11-00000000", "bb22-00000000", "cc33-00000000"]
    expected = {point_id("cc33-00000000", 0): 9000, point_id("aa11-00000000", 0): 1000}  # bb22 unknown -> 5000
    pool = _OrderSlots(2)
    pool.hold(2)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", pool)
    # the held-out bench registers first and has the larger budget (today: it would take the slots first)
    held, train = await asyncio.gather(
        _tail_bench(tmp_path, "held", ["dd44-00000000"], 8000, passes=3, tail={"expected": {}, "last": True}),
        _tail_bench(tmp_path, "train", games, 5000, tail={"expected": expected, "last": False}))
    rows = held["attempts"] + train["attempts"]
    assert len(held["attempts"]) == 3 and len(train["attempts"]) == 6
    assert all(r["outcome"] == "cleared" for r in rows)
    first_held = min(i for i, label in enumerate(pool.started) if "dd44" in label)
    assert first_held == 6  # every training try started before the first held-out try
    order = [label.split("-")[1] for label in pool.started[:6]]
    assert order == ["cc33", "cc33", "bb22", "bb22", "aa11", "aa11"]
    assert pool.used == 0 and pool.pending == {} and not pool.live


def test_per_level_caps_of_a_play_through_point():
    """envs explorer stuck stop (22 Oct): a list cap applies per level, its last entry to deeper levels."""
    agent = object.__new__(level_bench.LevelBenchAgent)
    agent.bench_level_cap, agent.bench_levels = [1000, 2000], []
    assert agent.level_cap_now() == 1000
    agent.bench_levels = [{"completion_tokens": 10}]
    assert agent.level_cap_now() == 2000
    agent.bench_levels = [{}, {}, {}]
    assert agent.level_cap_now() == 2000
    agent.bench_level_cap = 300
    assert agent.level_cap_now() == 300


async def test_heldout_first_takes_slots_before_training_and_hook_sees_every_try(tmp_path, monkeypatch):
    """mg.heldout_first: the fixed read's tries start before the training bench's, though its budget is
    smaller; PROGRESS_HOOK gets each finished try with its bench name."""
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    seen = []
    monkeypatch.setattr(level_bench, "PROGRESS_HOOK", lambda bench, row: seen.append((bench, row["outcome"])))
    pool = _OrderSlots(2)
    pool.hold(2)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", pool)
    held, train = await asyncio.gather(
        _tail_bench(tmp_path, "train", ["aa11-00000000", "bb22-00000000"], 8000, tail={"expected": {}, "last": False}),
        _tail_bench(tmp_path, "fixed", ["dd44-00000000"], 5000, passes=3, tail={"expected": {}, "last": False, "first": True}))
    assert [label.split("-")[1] for label in pool.started[:3]] == ["dd44"] * 3
    assert sorted(b for b, _ in seen) == ["fixed"] * 3 + ["train"] * 4


def test_kv_gate_counts_reserved_explorer_kv():
    """Explorers play outside the pool; their reserved KV counts in the kv_fill gate (ovf3r3 KV replay)."""
    key = object()
    pool = level_bench.SharedSlots(10, kv_fill=1.0, project={"growth": 1.0, "ctx_frac": 1.0, "scale": 1.0})
    pool.capacity = 100000
    pool.want(key, 0, 10)
    assert pool.take(key)
    pool.enter(ProjSlot(est=0, floor=10000, budget=30000))  # 40000
    pool.reserved = 30000
    assert pool.load() == 70000 and not pool.take(key)  # 70000 + 40000 > 100000
    pool.reserved = 0.0
    assert pool.take(key)


def test_kv_project_group_prefix_counts_a_starts_prefix_once_and_growth_in_full():
    """KV replay (23 Oct): tries restored from one start share its prefix blocks in vLLM, never their growth."""
    pool = level_bench.SharedSlots(10, kv_fill=1.0, project={"growth": 1.0, "ctx_frac": 1.0, "scale": 0.5,
                                                             "group_prefix": True})
    pool.capacity = 1_000_000
    a, b, c = (ProjSlot(est=0, floor=40000, budget=10000) for _ in range(3))
    a.prefix_key = b.prefix_key = "start-1"
    c.prefix_key = None  # an unrestored try: its own prefix
    for s in (a, b, c):
        pool.enter(s)
    assert a.project_start == 50000  # unscaled end of try
    assert pool.load() == 40000 + 40000 + 3 * 10000  # start-1's prefix once, c's own, every try's growth
    plain = level_bench.SharedSlots(10, kv_fill=1.0, project={"growth": 1.0, "ctx_frac": 1.0, "scale": 0.5})
    plain.capacity = 1_000_000
    for s in (a, b, c):
        plain.enter(s)
    assert plain.load() == 3 * 50000 * 0.5  # the old whole-context scale


def test_admit_stagger_spaces_a_benchs_takes_not_other_benches(monkeypatch):
    """mg.admit_stagger_s: one bench's tries start at least stagger_s apart (no lockstep growth)."""
    clock = [1000.0]
    monkeypatch.setattr(level_bench.time, "time", lambda: clock[0])
    pool = level_bench.SharedSlots(10, stagger_s=150)
    a, b = object(), object()
    pool.want(a, 0, 5)
    pool.want(b, 0, 5)
    assert pool.take(a) and not pool.take(a)  # a's second try waits
    assert pool.take(b)  # another bench is not held back
    clock[0] += 149
    assert not pool.take(a)
    clock[0] += 1
    assert pool.take(a)
    assert level_bench.SharedSlots(10).take.__self__.stagger_s == 0.0


def test_admit_stagger_is_one_clock_across_both_servers(monkeypatch):
    """hero-v3 card-2 read (23 Oct): server 2's pool had no stagger, so every try server 1's stagger refused went to
    card 2 at once (card 1 ran 6 tries, card 2 30). Both pools share the bench's admission clock."""
    clock = [1000.0]
    monkeypatch.setattr(level_bench.time, "time", lambda: clock[0])
    p1, p2 = level_bench.SharedSlots(10, stagger_s=150), level_bench.SharedSlots(10, stagger_s=150)
    p2.last_take = p1.last_take
    a = object()
    p1.want(a, 0, 5)
    p2.want(a, 0, 5)
    assert p1.take(a)
    assert not p1.take(a) and not p2.take(a)  # the bench's next try waits on both cards
    clock[0] += 150
    assert p2.take(a) and not p1.take(a)  # a take on card 2 also starts the bench's clock


def test_engine_scale_calibrates_from_engine_kv_over_the_projection():
    """kv_project.auto_scale (v6h: projected 0.75 vs engine KV 0.23): engine_scale moves to the q quantile of the
    measured ratio (engine KV / unscaled projection) over the window, clamped to [floor, ceil]; a set knob stops it."""
    class T:
        def __init__(self, i):
            self.next_estimate = self.floor_tokens = 20000
            self.window, self.cap, self.prefix_key = 131072, 16384, i

        def progress(self):
            return 0, 30000

    auto = {"window_s": 600, "min_s": 120, "q": 0.8, "floor": 0.5, "ceil": 1.0}
    pool = level_bench.SharedSlots(80, 0.75, project={"group_prefix": True, "auto_scale": auto})
    pool.capacity = 1_000_000
    for i in range(10):
        pool.enter(T(i))
    kv = lambda ratio: ratio * pool.load() / pool.project.get("engine_scale", 1.0) / 1e6
    assert pool.calibrate(kv(0.3), now=0.0) is None  # before min_s
    assert pool.calibrate(kv(0.3), now=130.0) == 0.5  # ratio 0.3: clamped at the floor
    for t in range(140, 800, 10):
        got = pool.calibrate(kv(0.9), now=float(t))
    assert pool.project["engine_scale"] == 0.9  # the window rolled past the young-try samples
    pool.project["auto_off"] = True  # a kv_engine_scale knob
    assert pool.calibrate(kv(0.2), now=900.0) is None and pool.project["engine_scale"] == 0.9


async def test_load_adapters_and_arm_served_models(tmp_path, monkeypatch):
    """Standalone held-out read (RL, 25 Oct): adapters load before any try; each arm's study.served_model is the model
    its requests name (base vs the trained adapter, one bench)."""
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    board = np.zeros((64, 64), dtype=np.uint8)
    second = board.copy()
    second[5, 5] = 3
    point = {"id": "aa11-L2", "game": "aa11-00000000", "split": "dev", "level_index": 1, "level": 2, "total_levels": 5,
             "baseline_actions": 2, "prefix": ["ACTION1"], "prefix_actions": 1, "board_hash": board_hash(second)}
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": [point]}))
    order, models = [], {}

    def handler(request):
        if request.url.path.endswith("/load_lora_adapter"):
            assert request.url.path == "/v1/load_lora_adapter", request.url.path  # preflight pf1: /v1/v1/ was a 404
            order.append(("load", json.loads(request.content)))
            return httpx.Response(200, text="Success")
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 131072})
        payload = json.loads(request.content)
        order.append(("chat", payload["model"]))
        n = sum(1 for m in payload["messages"] if m["role"] == "assistant")
        code = "print('hi')" if n == 0 else "action([{'action':'UP'}])"
        return httpx.Response(200, json=tool_reply(code, call_id=f"c{n}",
                                                   usage={"completion_tokens": 300, "prompt_tokens": 4000}).json())

    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 1, "max_model_len": 131072}, "wall_seconds": 60, **V71,
             "served_model": "base-model", "load_adapters": [{"name": "s0017", "path": "/data/s0017/adapter"}],
             "level_bench": {"start_points": str(path), "capacity_tokens": 2000000, "window": 130816,
                             "token_budget": 5000, "concurrency": 4, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "serving": "test",
                             "arms": [{"name": "base"}, {"name": "s0017", "study": {"served_model": "s0017"}}]}}
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / "out", server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 120,
                                   log=lambda *a, **k: None, arcade=PrefixBenchArcade(),
                                   transport=httpx.MockTransport(handler))
    report = await bench.run()
    assert order[0] == ("load", {"lora_name": "s0017", "lora_path": "/data/s0017/adapter", "is_3d_lora_weight": True})
    assert {r["arm"] for r in report["attempts"]} == {"base", "s0017"}, report["task_errors"]
    chats = {m for k, m in order if k == "chat"}
    assert chats == {"base-model", "s0017"}
