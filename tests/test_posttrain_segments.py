"""Level-clear segments from recorded Duck-transfer logs (posttrain.segments)."""

import json
from pathlib import Path

import pytest

from posttrain import segments

ROOT = Path(__file__).resolve().parents[1]
REAL_GAME = (ROOT / "runs/modal-validation/arena-duck-60m-r23-base-v3-modal-20260925/arena/trials"
             / "duck-transfer/games/ar25-0c556536")


def rows(fill="0"):
    return [fill * 64 for _ in range(64)]


def obs(step, action=None, levels=0, state="NOT_FINISHED", fill="0", frames=None):
    return {"kind": "observation", "step": step, "action": action, "frames": frames or [rows(fill)],
            "levels": levels, "total_levels": 2, "state": state,
            "available": ["ACTION1", "ACTION6"], "full_reset": False}


def model(step, *, code=None, content=None, reasoning="", call_id="c", purpose="gameplay", tokens=10):
    message = {"role": "assistant", "content": content, "reasoning": reasoning, "tool_calls": []}
    if code is not None:
        message["tool_calls"] = [{"id": call_id, "type": "function",
                                  "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
    return {"kind": "model", "step": step, "purpose": purpose, "message": message,
            "finish_reason": "tool_calls" if code else "stop",
            "usage": {"prompt_tokens": 100 + step, "completion_tokens": tokens,
                      "completion_tokens_details": {"reasoning_tokens": tokens // 2}},
            "reasoning_effort": "xhigh"}


def python(step, code, stdout):
    return {"kind": "duck_python", "step": step, "code": code,
            "result": {"tool": "python", "returncode": 0, "stdout": stdout}}


def synthetic_log():
    """Level 1: two requests (a look, then UP + MOUSE clears it). Level 2: a reply without a tool
    call, a compaction, UP ends in GAME_OVER, the runner resets, UP clears the last level."""
    return [
        obs(0),
        {"kind": "stuck_notice", "step": 0, "text": "Stuck check: synthetic."},
        model(0, code="print(1)", reasoning="Look first.", call_id="a"),
        python(0, "print(1)", "1\n"),
        model(0, code="action(['UP', {'action': 'MOUSE', 'row': 4, 'col': 3}])",
              content="World model: a box.\nPlan: go up.", reasoning="Act now.", call_id="b"),
        obs(1, {"action": "ACTION1"}, fill="1"),
        obs(2, {"action": "ACTION6", "x": 3, "y": 4}, levels=1, fill="2"),
        python(2, "action(...)", "done\n"),
        model(2, content="Thinking out loud.", reasoning="No tool."),
        model(2, purpose="compaction", content="Summary: level 2 needs UP.", tokens=7),
        {"kind": "compaction", "step": 2, "mechanical_fallback": False},
        model(2, code="action(['UP'])", reasoning="Try UP.", call_id="d"),
        obs(3, {"action": "ACTION1"}, levels=1, state="GAME_OVER", fill="3"),
        python(3, "action(['UP'])", "over\n"),
        obs(4, {"action": "RESET"}, levels=1, fill="2"),
        {"kind": "auto_reset", "step": 4, "game_over_step": 3, "actions": ["UP"]},
        model(4, code="action(['UP'])", reasoning="Again.", call_id="e"),
        obs(5, {"action": "ACTION1"}, levels=2, state="WIN", fill="5"),
        python(5, "action(['UP'])", "won\n"),
        {"kind": "finished", "status": "won"},
    ]


def write_game(tmp_path, records, name="ar25-0c556536"):
    game = tmp_path / "runs" / "demo-run" / "arena" / "trials" / "duck-transfer" / "games" / name
    game.mkdir(parents=True)
    with open(game / "log.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.write("<html>not json</html>\n")
    return game


INFO = {"run": "demo-run", "roster": {}, "max_model_len": 131072, "request_cap": None,
        "keep_observations": 1, "variant": None, "unreconstructed": []}


def test_turns_close_at_the_executing_python_call_or_the_request_cap():
    turns, observations, extras = segments.game_turns(synthetic_log())
    assert [(t["start_step"], t["levels"], len(t["requests"])) for t in turns] == [(0, 0, 2), (2, 1, 2), (4, 1, 1)]
    assert turns[0]["requests"][1]["actions"] == [1, 2]
    assert turns[0]["notice"] == "Stuck check: synthetic."
    assert turns[1]["requests"][1]["summary"]["text"] == "Summary: level 2 needs UP."
    assert extras["external"] == [{"step": 4, "action": "RESET", "auto_reset": True}]
    capped, _, _ = segments.game_turns(synthetic_log(), max_requests=1)
    assert [len(t["requests"]) for t in capped] == [1, 1, 1, 1, 1]
    assert segments.level_clears(observations) == [
        {"level": 1, "start_step": 0, "clear_step": 2, "jump": 1},
        {"level": 2, "start_step": 2, "clear_step": 5, "jump": 1},
    ]


def test_segment_messages_follow_the_harness(tmp_path):
    game = write_game(tmp_path, synthetic_log())
    result = segments.reconstruct_game(game, INFO, images=True)
    first, second = result["segments"]
    assert first["actions"] == [{"action": "ACTION1"}, {"action": "ACTION6", "x": 3, "y": 4}]
    assert first["user_source"] == "simulated" and first["turns"] == 1 and first["requests"] == 2
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    user = first["messages"][1]["content"]
    assert user[0]["text"].startswith("No previous sequence has been executed yet.\nCurrent state: step 1, level 1.")
    assert "Stuck check: synthetic." in user[0]["text"]
    assert user[0]["text"].endswith("Grid image of the board at step 0 (before this turn's actions):")
    assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")
    act = first["messages"][4]
    assert act["reasoning_content"] == "Act now." and act["content"] == "World model: a box.\nPlan: go up."
    assert act["tool_calls"][0]["id"] == "b"
    assert first["messages"][5] == {"role": "tool", "tool_call_id": "b",
                                    "content": json.dumps({"tool": "python", "returncode": 0,
                                                           "stdout": "done\n"}, indent=2)}
    assert first["completion_tokens"] == 20 and first["first_prompt_tokens"] == 100

    assert first["target_start"] == 1 and not first["starts_after_compaction"]
    assert [m["prompt_tokens"] for m in first["request_meta"]] == [100, 100]
    assert [m["actions"] for m in first["request_meta"]] == [0, 2]

    # Level 2 keeps level 1 as context (the arena context is append-only until a compaction): the
    # follow-up after the tool-less reply, the compaction summary inserted before the latest user
    # message (the follow-up), the GAME_OVER note of the automatic reset.
    assert second["actions"] == [{"action": "ACTION1"}, {"action": "RESET"}, {"action": "ACTION1"}]
    assert second["auto_resets"] == 1 and second["exclusion"] == []
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool",
                     "user", "assistant", "user", "user", "assistant", "tool", "user", "assistant", "tool"]
    assert second["messages"][1:6] == first["messages"][1:6]
    assert second["target_start"] == 6 and not second["starts_after_compaction"]
    assert second["compaction_indices"] == [8]
    assert second["messages"][8]["content"].startswith("[Context compaction]")
    assert second["messages"][8]["content"].endswith("Summary: level 2 needs UP.")
    assert second["messages"][9]["content"].startswith("You have not acted yet. Investigate first.")
    assert "tool_calls" not in second["messages"][7]
    later = second["messages"][12]["content"][0]["text"]
    assert later.startswith("The code executed 1 action in the previous sequence.\nExecuted actions: UP.")
    assert "GAME_OVER at step 3 after actions [UP]; the level was reset automatically" in later
    assert second["compaction_completion_tokens"] == 7
    assert [(m["prompt_tokens"], m["finish_reason"]) for m in second["request_meta"]] == [
        (102, "stop"), (102, "tool_calls"), (104, "tool_calls")]

    # Records: the first level-2 request saw level 1; after the compaction only the summary on.
    from posttrain import sft_records

    records = sft_records.segment_records(second)
    assert [r["turn"] for r in records] == [1, 2, 3]
    assert records[0]["messages"] == second["messages"][:7]
    assert records[0]["served_prompt_tokens"] == 102 and records[0]["finish_reason"] == "stop"
    assert records[1]["messages"] == [second["messages"][0], *second["messages"][8:10]]
    assert records[2]["messages"] == [second["messages"][0], *second["messages"][8:13]]
    # The old truncated context (the level's own turns only) stays available for comparisons.
    old = sft_records.segment_records(second, context="segment")
    assert old[0]["messages"] == [second["messages"][0], second["messages"][6]]
    assert old[1]["messages"] == records[1]["messages"]


def without_served_sizes(records):
    for record in records:
        if record["kind"] == "model":
            record["usage"].pop("prompt_tokens")
    return records


def test_a_later_compaction_does_not_change_an_earlier_segment(tmp_path):
    """A keep-3 compaction on level 2 reaches back into level 1's messages; level 1's segment is cut
    from a snapshot taken at its last turn, so it never sees that summary."""
    game = write_game(tmp_path, without_served_sizes(synthetic_log()))
    first, second = segments.reconstruct_game(game, {**INFO, "keep_observations": 3}, images=False)["segments"]
    assert all(m["role"] != "user" or not str(m["content"]).startswith("[Context compaction]")
               for m in first["messages"])
    # The summary went before the third-latest user message: level 1's first user turn. The
    # request before the compaction saw level 1 without it; the ones after, the summary on.
    assert second["compaction_indices"] == [1] and not second["starts_after_compaction"]
    assert second["messages"][1]["content"].startswith("[Context compaction]")
    assert second["target_start"] == 7
    assert "compaction_keeps_3_observations" in second["caveats"]
    contexts = [m["context"] for m in second["request_meta"]]
    assert contexts == [[[0, 1], [2, 8]], [[0, 10]], [[0, 13]]]


def test_the_compaction_tail_follows_the_served_size(tmp_path):
    """The harness keeps only the latest observation when the kept tail does not fit: the logged
    prompt size decides which tail the rebuilt context uses."""
    game = write_game(tmp_path, synthetic_log())
    _, second = segments.reconstruct_game(game, {**INFO, "keep_observations": 3}, images=False)["segments"]
    # 102 served tokens: far below the keep-3 tail, so the latest observation (the follow-up) only.
    assert "compaction_tail_latest" in second["caveats"]
    assert [m["compaction_tail"] for m in second["request_meta"]] == [None, "latest", None]
    after = second["request_meta"][1]["context"]
    records = __import__("posttrain.sft_records", fromlist=["x"]).segment_records(second)
    assert [m["role"] for m in records[1]["messages"]] == ["system", "user", "user"]
    assert records[1]["messages"][1]["content"].startswith("[Context compaction]")
    assert after[0] == [0, 1]


def test_transcript_prompts_are_used_when_the_turns_align(tmp_path):
    game = write_game(tmp_path, synthetic_log())
    plain = segments.reconstruct_game(game, INFO, images=False)
    first_turn = plain["segments"][0]["messages"][1]["content"][0]["text"].split("\n\nGrid image")[0]
    transcript = (
        "\n--- analysis_step=0 | action=1 | 10:00:00 | tool-agent ---\n[SYSTEM PROMPT]\nSYS\n\n"
        f"[USER PROMPT]\n{first_turn}\n\n[MODEL RESPONSE META]\ntool_call_count: 1\n\n[THINKING]\nx\n\n"
        "[MODEL RESPONSE META]\ntool_call_count: 1\n\n[ANALYZER STATUS]\nok\n\n"
        "\n--- analysis_step=2 | action=3 | 10:00:01 | tool-agent ---\n[SYSTEM PROMPT]\nSYS\n\n"
        "[USER PROMPT]\nTURN TWO\n\n[MODEL RESPONSE META]\ntool_call_count: 0\n\n"
        "[USER PROMPT]\nFOLLOW\n\n[MODEL RESPONSE META]\ntool_call_count: 1\n\n"
        "\n--- analysis_step=4 | action=5 | 10:00:02 | tool-agent ---\n[SYSTEM PROMPT]\nSYS\n\n"
        "[USER PROMPT]\nTURN THREE\n\n[MODEL RESPONSE META]\ntool_call_count: 1\n\n"
    )
    parsed = segments.parse_transcript(transcript)
    assert [t["metas"] for t in parsed] == [[1, 1], [0, 1], [1]]
    assert parsed[1]["users"] == ["TURN TWO", "FOLLOW"]
    (game / "duck-transcript.txt").write_text(transcript)
    result = segments.reconstruct_game(game, INFO, images=False)
    assert result["aligned"]
    first, second = result["segments"]
    assert first["messages"][0]["content"] == "SYS"
    assert first["user_source"] == "transcript_verified"
    assert second["user_source"] == "transcript"  # the stand-in texts differ from the simulation
    start = second["target_start"]
    assert second["messages"][start]["content"][0]["text"].startswith("TURN TWO\n\nGrid image")
    assert second["messages"][start + 3]["content"] == "FOLLOW"


def fake_replay(recorded, *, animation_at=None, frame_at=None, level_at=None):
    def replay(game_id, actions, environments_dir):
        for index, want in enumerate(recorded[:len(actions) + 1]):
            frames = [list(f) for f in want["frames"]]
            if index == animation_at:
                frames = [rows("f"), *frames]
            if index == frame_at:
                frames = [rows("e")]
            levels = want["levels"] + (1 if index == level_at else 0)
            yield frames, levels, want["state"]
    return replay


@pytest.mark.parametrize("kwargs, expected", [
    ({}, {1: (True, "reproduced"), 2: (True, "reproduced")}),
    ({"animation_at": 1}, {1: (True, "reproduced_final_frames"), 2: (True, "reproduced_final_frames")}),
    ({"frame_at": 3}, {1: (True, "reproduced"), 2: (False, "frame_mismatch_at_step_3")}),
    ({"level_at": 1}, {1: (False, "level_mismatch_at_step_1"), 2: (False, "level_mismatch_at_step_1")}),
])
def test_verification_compares_every_recorded_step(tmp_path, kwargs, expected):
    (tmp_path / "ar25" / "0c556536").mkdir(parents=True)
    _, observations, _ = segments.game_turns(synthetic_log())
    clears = segments.level_clears(observations)
    replay = fake_replay(observations, **kwargs)
    assert segments.verify_game("ar25-0c556536", observations, clears, tmp_path, replay) == expected
    assert segments.verify_game("ar25-ffffffff", observations, clears, tmp_path, replay)[1] == (
        False, "game_files_missing")


def test_process_game_scores_against_the_baseline(tmp_path):
    game = write_game(tmp_path, synthetic_log())
    env = tmp_path / "environment_files" / "ar25" / "0c556536"
    env.mkdir(parents=True)
    (env / "metadata.json").write_text(json.dumps({"baseline_actions": [4, 3]}))
    _, observations, _ = segments.game_turns(synthetic_log())
    result = segments.process_game(game, INFO, environments_dir=tmp_path / "environment_files",
                                   images=False, replay_fn=fake_replay(observations))
    first, second = result["segments"]
    assert (first["baseline_actions"], first["efficiency"], first["level_score"]) == (4, 2.0, 115.0)
    assert (second["baseline_actions"], second["efficiency"], second["level_score"]) == (3, 1.0, 100.0)
    assert first["verified"] and second["verified"]


def test_deduplicate_and_finalize_keep_dev_games_only():
    base = {"game": "ar25", "game_id": "ar25-0c556536", "level": 1, "verified": True,
            "verify_reason": "reproduced", "exclusion": [], "action_count": 2, "level_score": 115.0,
            "completion_tokens": 5, "reasoning_tokens": 2, "caveats": [], "compaction_indices": [],
            "starts_after_compaction": False, "user_source": "simulated", "baseline_actions": 4,
            "actions": [{"action": "ACTION1"}], "messages": [], "tools": None}
    records = [{**base, "id": "a", "run": "r1"}, {**base, "id": "b", "run": "r2"},
               {**base, "id": "c", "run": "r3", "actions": [{"action": "ACTION2"}]},
               {**base, "id": "d", "run": "r4", "verified": False, "verify_reason": "frame_mismatch_at_step_3"}]
    kept, rejected, stats = segments.finalize([{"segments": records, "prompt_check": {}}], {}, [])
    assert [(r["id"], r["sequence_count"], r["duplicate_ids"]) for r in kept] == [("a", 2, ["b"]), ("c", 1, [])]
    assert [r["id"] for r in rejected] == ["d"] and "messages" not in rejected[0]
    row = stats["per_game_level"][0]
    assert (row["segments"], row["verified"], row["unique_verified"]) == (4, 3, 2)
    assert row["failed"] == {"frame_mismatch_at_step_3": 1}
    with pytest.raises(ValueError, match="non-dev"):
        segments.finalize([{"segments": [{**base, "id": "x", "run": "r", "game_id": "ls20-9607627b"}],
                            "prompt_check": {}}], {}, [])
    # Held-out generated mechanics: rejected from training, or split off as the evaluation set.
    held = {**base, "id": "q", "run": "r", "game": "q009", "game_id": "q009-6696a439"}
    with pytest.raises(ValueError, match="q009"):
        segments.finalize([{"segments": [held], "prompt_check": {}}], {}, [])
    kept, _, stats, evaluation = segments.finalize(
        [{"segments": [{**base, "id": "a", "run": "r1"}, held], "prompt_check": {}}], {}, [], with_eval=True)
    assert [r["id"] for r in kept] == ["a"] and [r["id"] for r in evaluation] == ["q"]
    assert stats["eval_split"]["games"] == ["q009-6696a439"]


def test_request_cap_follows_the_bundle(tmp_path):
    arena = tmp_path / "dist" / "arena-duck-60m-r23-base-v9" / "src" / "arena"
    arena.mkdir(parents=True)
    (arena / "snapshot.py").write_text("from inference.agent.action_names import to_model_action\n")
    assert segments.request_cap("modal-validation/arena-duck-60m-r23-base-v9-modal-20260925", tmp_path) == (
        12, "bundle")
    (arena / "snapshot.py").write_text("def f():\n    from inference.agent import x\n")
    assert segments.request_cap("kaggle-arena-duck-60m-r23-base-v9", tmp_path) == (None, "bundle")
    assert segments.request_cap("kaggle-arena-unknown", tmp_path) == (None, "no_bundle")


@pytest.mark.skipif(not (REAL_GAME / "log.jsonl").is_file() or not (ROOT / "environment_files/ar25").is_dir(),
                    reason="recorded run or local game files not present")
def test_real_run_smoke():
    info = segments.trial_info(REAL_GAME.parents[1])
    assert info["model"] == segments.TEACHER_MODEL and info["effort"] == "xhigh"
    result = segments.process_game(REAL_GAME, info, environments_dir=ROOT / "environment_files", images=False)
    check = result["prompt_check"]
    assert result["aligned"] and check["turns"] > 0
    assert check["user_exact"] == check["turns"] and check["system_exact"] == check["turns"]
    assert result["segments"] and all(r["verified"] for r in result["segments"])
    for record in result["segments"]:
        assert record["messages"][0]["role"] == "system"
        assert record["action_count"] == record["clear_step"] - record["start_step"]
        tool_ids = [c["id"] for m in record["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])]
        answered = [m["tool_call_id"] for m in record["messages"] if m["role"] == "tool"]
        assert tool_ids == answered


def test_generated_games_replay_from_the_generated_store(tmp_path):
    from posttrain import segments

    public = tmp_path / "environment_files"
    assert segments.game_environments_dir("ar25-0c556536", public) == public
    assert segments.game_environments_dir("q003-fa4c376b", public) == tmp_path / "data/posttrain/games/environment_files"


def test_episode_games_replay_from_their_batch_folder(tmp_path):
    from posttrain import segments

    folder = tmp_path / "data/posttrain/episodes/batch-1/environment_files/q200/abcd1234"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text("{}")
    public = tmp_path / "environment_files"
    assert segments.game_environments_dir("q200-abcd1234", public) == tmp_path / "data/posttrain/episodes/batch-1/environment_files"
    assert segments.game_environments_dir("q003-fa4c376b", public) == tmp_path / "data/posttrain/games/environment_files"
