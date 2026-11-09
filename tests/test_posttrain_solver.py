"""Solver policy (data source (a)): a generated game played to WIN through the real harness, and
the shared SFT record shape (posttrain.sft_records)."""

import importlib.util
import json
from pathlib import Path

import pytest

from posttrain import sft_records
from posttrain.games import admission, store
from posttrain.solver_policy import SolverPolicy, load_plan

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(f"test_{name}", ROOT / "scripts" / "posttrain" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_game(root, prefix="q006"):
    """Assemble a hand-made catalog game with its solution, as build_games.py does."""
    module_path = next(p for p in store.catalog_modules() if p.name.startswith(prefix + "_"))
    prefix, source, meta = store.assemble(module_path)
    cls = store.load_class(prefix, source)
    paths = [admission.solve_level(cls, i) for i in range(len(cls.LEVELS))]
    solution = [[[a, list(c) if c else None] for a, c in path] for path in paths]
    return store.write_game(root, prefix, source, meta, [len(p) for p in paths], {"solution": solution}), paths


def request(text, *, tools=True, extra=()):
    user = {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "x"}}]}
    payload = {"messages": [{"role": "system", "content": "rules"}, user, *extra]}
    if tools:
        payload["tools"] = [{"type": "function", "function": {"name": "python"}}]
    return payload


def test_policy_batches_levels_and_answers_summaries(tmp_path):
    game_id, paths = build_game(tmp_path)
    plan = load_plan(tmp_path, game_id)
    assert [len(p) for p in plan] == [len(p) for p in paths]
    click = plan[0][0]
    assert set(click) == {"action", "row", "col"} and click["action"] == "MOUSE"
    policy = SolverPolicy(plan, chunk=4)
    first = policy.answer(request("Current state: step 1, level 1."))
    assert first["code"].startswith("r = action([{'action': 'MOUSE'") and first["content"].startswith("Plan: ")
    assert first["reasoning"].startswith("Level 1, step 1:")
    # Balanced batches: a 9-action level with chunk 4 splits 3/3/3.
    assert [policy.batch_end(9, o) for o in (0, 3, 6)] == [3, 6, 9]
    assert policy.batch_end(5, 0) == 3 and policy.batch_end(5, 3) == 5
    summary = policy.answer(request("Current state: step 3, level 1.", tools=False))
    assert "code" not in summary and summary["content"].startswith("Current level and state: level 1, step 3")
    with pytest.raises(RuntimeError, match="expects 1"):
        policy.answer(request("Current state: step 2, level 2."))
    failed = [{"role": "assistant", "content": None}, {"role": "tool", "content": "error"}]
    with pytest.raises(RuntimeError, match="did not execute"):
        policy.answer(request("Current state: step 1, level 1.", extra=failed))
    assert SolverPolicy(plan, reasoning=False).answer(request("Current state: step 1, level 1."))["reasoning"] == ""


@pytest.mark.skipif(not (ROOT / ".analysis-venv/bin/python").exists(), reason="needs the harness sandbox venv")
async def test_solver_plays_a_game_to_win_through_the_harness_and_builds_records(tmp_path):
    games = tmp_path / "games"
    game_id, paths = build_game(games)
    play = script("solver_play")
    run = tmp_path / "runs" / game_id
    row = await play.play_one(game_id, games, run, chunk=4)
    baselines = [len(p) for p in paths]
    assert row["state"] == "WIN" and row["check"], row
    assert row["actions"] == baselines and all(s == 100.0 for s in row["scores"])
    assert row["turns"] == row["answered"] == sum(-(-n // 4) for n in baselines)

    # q006 is a held-out generated mechanic: never training data, only the next-action eval split.
    train, train_stats = script("build_solver_sft").build(tmp_path / "runs", games)
    assert train == [] and train_stats["games"] == 0
    records, stats = script("build_solver_sft").build(tmp_path / "runs", games, split="eval")
    assert len(records) == stats["records"] == row["turns"] and stats["split"] == "eval"
    assert all(r["split"] == "eval" for r in records)
    assert stats["per_game"][game_id]["actions"] == baselines
    for turn, record in enumerate(records, start=1):
        sft_records.check_record(record)
        assert record["source"] == "solver" and record["game_id"] == game_id and record["turn"] == turn
        assert record["purpose"] == "gameplay" and record["prompt_tokens_estimate"] > 1000
        call = record["target"]["tool_calls"][0]
        assert call["function"]["name"] == "python" and "action([" in json.loads(call["function"]["arguments"])["code"]
        # The exact request context: system prompt first, the newest observation last.
        assert record["messages"][0]["role"] == "system" and record["messages"][-1]["role"] == "user"
        assert all("reasoning" not in m for m in record["messages"])
    assert [r["level"] for r in records] == [lvl + 1 for lvl, n in enumerate(baselines) for _ in range(-(-n // 4))]
    # The history carries the previous replies (the arena context is append-only).
    later = records[-1]["messages"]
    assert sum(m["role"] == "assistant" for m in later) == len(records) - 1
    assert later[2]["reasoning_content"].startswith("Level 1, step 1:")


def test_holdout_games_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="sb26"):
        sft_records.solver_records(tmp_path, "sb26-7fbdac44")
    record = sft_records.make_record(
        source="solver", game_id="tr87-cd924810", level=1, turn=1,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tools=[],
        target={"content": "x"})
    with pytest.raises(ValueError, match="tr87"):
        sft_records.check_record(record)
    with pytest.raises(ValueError, match="ls20"):
        script("build_solver_sft").build(_fake_runs(tmp_path, "ls20-9607627b"), tmp_path)


def _fake_runs(tmp_path, game_id):
    runs = tmp_path / "runs"
    (runs / game_id / "relay").mkdir(parents=True)
    return runs


def test_segment_adapter_gives_the_shared_shape():
    tool_call = {"id": "c1", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "1"})}}
    segment = {
        "id": "run:ar25:L1", "run": "run", "game_id": "ar25-0c556536", "level": 1,
        "tools": [{"type": "function", "function": {"name": "python"}}],
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "board 1"},
            {"role": "assistant", "content": None, "reasoning_content": "think", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": "summary"},
            {"role": "user", "content": "board 2"},
            {"role": "assistant", "content": "done", "reasoning_content": "more"},
        ],
        "compaction_indices": [4],
    }
    records = sft_records.segment_records(segment)
    assert [r["turn"] for r in records] == [1, 2] and all(r["source"] == "segment" for r in records)
    assert records[0]["messages"] == segment["messages"][:2]
    assert records[0]["target"] == {"reasoning": "think", "content": "", "tool_calls": [tool_call]}
    # After the compaction the request saw the system prompt and the messages from the summary on.
    assert [m["content"] for m in records[1]["messages"]] == ["rules", "summary", "board 2"]
    assert set(sft_records.RECORD_KEYS) <= set(records[1])
