"""Offline harness evolution: the proposer's tools and session, the gate and the round loop."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from arena.evolve import Evolution, Proposer, better, metrics


def workspace_with_traces(tmp_path):
    traces = tmp_path / "eval" / "duck-transfer" / "games" / "ar25-0c556536"
    traces.mkdir(parents=True)
    (traces / "duck-transcript.txt").write_text("turn 1\nStuck check: 12 turns\nturn 3\n")
    (traces / "log.jsonl").write_text('{"kind": "stuck_notice"}\n')
    harness = tmp_path / "candidate"
    (harness / "skills").mkdir(parents=True)
    ws = tmp_path / "proposer"
    ws.mkdir()
    (ws / "traces").symlink_to(tmp_path / "eval" / "duck-transfer" / "games", target_is_directory=True)
    (ws / "harness").symlink_to(harness, target_is_directory=True)
    return ws, harness


def test_proposer_tools_read_traces_and_write_only_harness_files(tmp_path):
    ws, harness = workspace_with_traces(tmp_path)
    events = []
    proposer = Proposer(workspace=ws, harness_dir=harness, base_url="http://x/v1", model="m",
                        settings={"tool_turns": 5, "seconds": 60, "cap_tokens": 100, "reasoning_effort": "low",
                                  "result_chars": 8000, "context_chars": 100000, "temperature": 0.7},
                        python=sys.executable, log=events.append)
    assert proposer.call("list_dir", {"path": "traces"}) == "ar25-0c556536/"
    assert proposer.call("read", {"path": "traces/ar25-0c556536/duck-transcript.txt", "start": 2, "lines": 1}) == (
        "traces/ar25-0c556536/duck-transcript.txt: lines 2-2 of 3\n     2| Stuck check: 12 turns")
    assert proposer.call("grep", {"pattern": "stuck", "path": "traces"}).splitlines() == [
        "traces/ar25-0c556536/duck-transcript.txt:2: Stuck check: 12 turns", 'traces/ar25-0c556536/log.jsonl:1: {"kind": "stuck_notice"}']
    assert proposer.call("write", {"path": "harness/notes.txt", "content": "x"}).startswith("Refused")
    assert proposer.call("write", {"path": "../agent.py", "content": "x"}).startswith("Refused")
    assert proposer.call("read", {"path": "../../etc/passwd"}).startswith("PermissionError")
    assert proposer.call("write", {"path": "harness/prompt.md", "content": "Probe once."}) == "Wrote prompt.md (11 chars)"
    reply = proposer.call("write", {"path": "harness/skills/diff.py", "content": '"""Board diff."""\ndef d(a, b):\n    return a != b\n'})
    assert reply.startswith("Wrote skills/diff.py") and "skills: ['diff']" in reply
    assert proposer.call("write", {"path": "harness/skills/cls.py", "content": "class A:\n    pass\n"}).count("class statements") == 1
    assert proposer.call("write", {"path": "harness/skills/cls.py", "content": ""}) == "Deleted skills/cls.py"
    assert proposer.call("write", {"path": "harness/skills/bad.py", "content": "def (:\n"}).count("invalid syntax") == 1
    assert "problems: [\"skills/bad.py: invalid syntax (line 1); skipped\"]" in proposer.call("check", {})
    assert proposer.call("write", {"path": "harness/skills/bad.py", "content": ""}) == "Deleted skills/bad.py"
    assert proposer.call("nope", {}) == "Unknown tool nope" and proposer.call("read", {"lines": 2}).startswith("Bad arguments")
    assert proposer.call("finish", {"summary": "done"}) == "Session ended" and proposer.summary == "done"
    assert [w["path"] for w in proposer.writes] == ["prompt.md", "skills/diff.py", "skills/cls.py", "skills/cls.py", "skills/bad.py", "skills/bad.py"]
    assert (harness / "prompt.md").read_text() == "Probe once." and not (harness / "skills" / "bad.py").exists()


def test_proposer_session_runs_tool_calls_until_finish_and_trims_old_results(tmp_path, monkeypatch):
    ws, harness = workspace_with_traces(tmp_path)
    script = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "traces/ar25-0c556536/duck-transcript.txt"})}},
            {"id": "b", "type": "function", "function": {"name": "write", "arguments": json.dumps({"path": "harness/playbook.md", "content": "- Read the receipt after every action."})}}]},
        {"role": "assistant", "content": "thinking about it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "finish", "arguments": json.dumps({"summary": "Added a receipt rule."})}}]},
    ]
    seen = []

    def fake_request(self, client, messages):
        seen.append([m["role"] for m in messages])
        return script.pop(0), {"prompt_tokens": 10, "completion_tokens": 5}, 0.01

    monkeypatch.setattr(Proposer, "request", fake_request)
    proposer = Proposer(workspace=ws, harness_dir=harness, base_url="http://x/v1", model="m",
                        settings={"tool_turns": 40, "seconds": 60, "cap_tokens": 100, "reasoning_effort": "low",
                                  "result_chars": 50, "context_chars": 600, "temperature": 0.7},
                        python=sys.executable, log=lambda e: None)
    outcome = proposer.run("Round 1 brief")
    assert outcome["ended"] == "finish" and outcome["turns"] == 3 and outcome["summary"] == "Added a receipt rule."
    assert outcome["usage"] == {"prompt_tokens": 30, "completion_tokens": 15} and [w["path"] for w in outcome["writes"]] == ["playbook.md"]
    assert seen[0] == ["system", "user"] and seen[1] == ["system", "user", "assistant", "tool", "tool"]
    assert seen[2] == ["system", "user", "assistant", "tool", "tool", "assistant", "user"]  # the nudge to call finish
    lines = [json.loads(line) for line in (ws / "transcript.jsonl").read_text().splitlines()]
    assert [entry.get("tool") for entry in lines if "tool" in entry] == ["read", "write", "finish"]
    assert lines[1]["result"].endswith("(truncated)")  # result_chars cap


def evolution_bundle(tmp_path):
    bundle = tmp_path / "bundle" / "configs"
    bundle.mkdir(parents=True)
    games = [f"g{i:02d}-aaaa" for i in range(6)]
    study = {"name": "evo-test", "games": games, "wall_seconds": 1800, "served_model": "m",
             "trials": [{"name": "duck-transfer", "kind": "duck-transfer"}],
             "evolution": {"dev_games": games[:4], "holdout_games": games[4:], "rounds": 3, "round_wall_seconds": 10,
                           "eval_grace_seconds": 1, "smoke": {"games": 1, "wall_seconds": 1},
                           "proposer": {"seconds": 1}, "holdout": True}}
    (bundle / "comparison.json").write_text(json.dumps(study))
    return SimpleNamespace(bundle=str(tmp_path / "bundle"), working=str(tmp_path / "working"), output=str(tmp_path / "out"),
                           games_dir=None, base_url="http://x/v1", python=sys.executable, pi_dir=None, node="node",
                           started_epoch=__import__("time").time(), max_seconds=1_000_000.0)


def test_evolution_keeps_better_candidates_reverts_worse_and_tests_holdout(tmp_path, monkeypatch):
    args = evolution_bundle(tmp_path)
    Path(args.output).mkdir()
    scores = {"round-00": 1.0, "round-01/eval": 2.0, "round-02/eval": 1.5, "round-03/eval": 2.0, "holdout-evolved": 0.7, "holdout-baseline": 0.4}
    evaluated = []

    def fake_evaluate(self, name, harness_dir, games, wall):
        evaluated.append((name, Path(harness_dir).name, len(games), wall))
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "duck-transfer" / "games").mkdir(parents=True, exist_ok=True)
        (folder / "duck-transfer" / "arena.json").write_text(json.dumps({"games": [], "levels": 0, "actions": 0, "status": "complete", "mean_score_equal_weight_25_games": 0}))
        if name.endswith("smoke"):
            return {"score": 0, "clears": 0, "games_with_clears": 0, "actions": 1, "level1_times": [], "errors": 0, "status": "complete", "folder": str(folder)}
        return {"score": scores[name], "clears": 3 if name == "round-03/eval" else 2, "games_with_clears": 1, "actions": 5,
                "level1_times": [100], "errors": 0, "status": "complete", "folder": str(folder)}

    def fake_propose(self, index, incumbent_eval):
        folder = self.root / f"round-{index:02d}"
        candidate = folder / "candidate"
        candidate.mkdir(parents=True)
        (candidate / "skills").mkdir()
        for src in self.incumbent.iterdir():
            if src.is_file():
                (candidate / src.name).write_text(src.read_text())
        (candidate / "prompt.md").write_text(f"round {index}")
        return candidate, {"writes": [{"path": "prompt.md", "chars": 7}], "summary": f"edit {index}", "turns": 2, "seconds": 1, "usage": {}, "request_failures": 0, "ended": "finish"}

    monkeypatch.setattr(Evolution, "evaluate", fake_evaluate)
    monkeypatch.setattr(Evolution, "propose", fake_propose)
    summary = Evolution(args).run()
    assert [r["decision"] for r in summary["rounds"]] == ["kept", "reverted", "kept"]  # round 3 ties on score, wins on clears
    assert summary["kept_rounds"] == [1, 3] and summary["final"]["score"] == 2.0 and summary["baseline"]["score"] == 1.0
    assert summary["holdout"] == {"evolved": {**summary["holdout"]["evolved"]}, "baseline": summary["holdout"]["baseline"]}
    assert summary["holdout"]["evolved"]["score"] == 0.7 and summary["holdout"]["baseline"]["score"] == 0.4
    names = [e[0] for e in evaluated]
    assert names == ["round-00", "round-01/smoke", "round-01/eval", "round-02/smoke", "round-02/eval", "round-03/smoke", "round-03/eval", "holdout-evolved", "holdout-baseline"]
    assert evaluated[1][2] == 1 and evaluated[1][3] == 1 and evaluated[2][2] == 4 and evaluated[-1][2] == 2  # smoke games, dev roster, holdout roster
    assert evaluated[-1][1] == "baseline-harness" and evaluated[-2][1] == "harness"
    assert (Path(args.output) / "evolved-harness" / "prompt.md").read_text() == "round 3"
    written = json.loads((Path(args.output) / "evolution.json").read_text())
    assert written["kept_rounds"] == [1, 3] and written["harness"]["prompt_chars"] == 7


def test_gate_and_metrics_from_a_report():
    assert better({"score": 2.0, "clears": 1}, {"score": 1.9, "clears": 5}) and not better({"score": 1.0, "clears": 9}, {"score": 1.0, "clears": 9})
    assert better({"score": 1.0, "clears": 10}, {"score": 1.0, "clears": 9})
    report = {"mean_score_equal_weight_25_games": 3.5, "levels": 4, "actions": 40, "status": "complete",
              "games": [{"levels": 2, "time_to_level": [{"level": 1, "wall_seconds": 120.4}, {"level": 2, "wall_seconds": 300}], "error": None},
                        {"levels": 0, "time_to_level": [], "error": "RuntimeError: x"},
                        {"levels": 2, "time_to_level": [{"level": 1, "wall_seconds": 80}], "error": None}]}
    assert metrics(report) == {"score": 3.5, "clears": 4, "games_with_clears": 2, "actions": 40, "level1_times": [80, 120], "errors": 1, "status": "complete"}


def test_builder_evolution_study_splits_the_roster(tmp_path):
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location("build_arena", Path(__file__).resolve().parents[1] / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output, study = builder.build("evolve-test", paired_30m=True, output=tmp_path / "bundle", arena_variant="reclaim-fixed-2x",
                                  opening_burst_requests=3, exclude_games=["ft09", "lp85"], evolve=True)
    evo = study["evolution"]
    assert len(evo["dev_games"]) == 14 and len(evo["holdout_games"]) == 9
    assert set(evo["dev_games"]).isdisjoint(evo["holdout_games"]) and set(evo["dev_games"] + evo["holdout_games"]) == set(study["games"])
    assert evo["rounds"] == 5 and evo["proposer"]["reasoning_effort"] == "xhigh" and study["notebook_seconds"] == 28800
    assert json.loads((output / "configs/comparison.json").read_text())["evolution"] == evo
    with pytest.raises(ValueError, match="reclaim-fixed-2x"):
        builder.build("evolve-bad", paired_30m=True, output=tmp_path / "bad", exclude_games=["ft09", "lp85"], evolve=True)


def test_evaluate_uses_a_written_report_even_when_the_trial_exits_nonzero(tmp_path, monkeypatch):
    import subprocess

    args = evolution_bundle(tmp_path)
    Path(args.output).mkdir()
    evo = Evolution(args)
    statuses = iter(["complete_with_errors", "no_such_status", None])

    def fake_run(command, **kwargs):
        folder = Path(command[command.index("--output") + 1])
        status = next(statuses)
        if status is not None:
            (folder / "duck-transfer").mkdir(parents=True, exist_ok=True)
            report = {"games": [{"levels": 1, "time_to_level": [{"level": 1, "wall_seconds": 50}], "error": "x"}],
                      "levels": 1, "actions": 9, "status": status, "mean_score_equal_weight_25_games": 1.5}
            (folder / "duck-transfer" / "arena.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    kept = evo.evaluate("a", evo.incumbent, ["g00-aaaa"], 10)
    assert kept["score"] == 1.5 and kept["exit"] == 1 and kept["errors"] == 1 and kept["status"] == "complete_with_errors"
    assert evo.evaluate("b", evo.incumbent, ["g00-aaaa"], 10) is None  # unknown status
    assert evo.evaluate("c", evo.incumbent, ["g00-aaaa"], 10) is None  # no report
    evo.save_evidence()
    assert (Path(args.output) / "evolve" / "a" / "study.json").is_file() and (Path(args.output) / "evolve" / "events.jsonl").is_file()
