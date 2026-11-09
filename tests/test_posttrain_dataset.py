"""The pilot SFT mixture (posttrain.dataset): filters, per-request contexts, packing, budgets, splits."""

import json

import pytest

from posttrain import dataset, sft_records

TOOLS = [{"type": "function", "function": {"name": "python", "description": "Run code.",
                                           "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def call(call_id, code):
    return {"id": call_id, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}


def board(text):
    return {"role": "user", "content": [{"type": "text", "text": text}, IMAGE]}


def segment(seg_id="run:ar25-0c556536:L2", game_id="ar25-0c556536", level=2, actions=("ACTION1",), tokens=10,
            **extra):
    """Level 2 after level 1 (context only). The first level-2 request sees level 1; a compaction
    before the second request inserts the summary before the level-2 board (keep 1); the third
    request was cut by the completion cap."""
    messages = [
        {"role": "system", "content": "rules"},
        board("board L1"),
        {"role": "assistant", "content": None, "reasoning_content": "look at level one", "tool_calls": [call("c1", "print(1)")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok 1"},
        {"role": "user", "content": "[Context compaction] summary of level one"},
        board("board L2"),
        {"role": "assistant", "content": "Plan: up.", "reasoning_content": "plan level two",
         "tool_calls": [call("c2", "action(['UP'])")]},
        {"role": "tool", "tool_call_id": "c2", "content": "ok 2"},
        {"role": "assistant", "content": None, "reasoning_content": "after the summary", "tool_calls": [call("c3", "print(3)")]},
        {"role": "tool", "tool_call_id": "c3", "content": "ok 3"},
        {"role": "assistant", "content": "cut off", "reasoning_content": "a very long thought"},
    ]
    meta = [
        {"prompt_tokens": 100, "finish_reason": "tool_calls", "python_error": None, "actions": 1,
         "context": [[0, 4], [5, 6]]},
        {"prompt_tokens": 90, "finish_reason": "tool_calls", "python_error": "IndexError", "actions": 0,
         "context": [[0, 1], [4, 8]]},
        {"prompt_tokens": 95, "finish_reason": "length", "python_error": None, "actions": 0,
         "context": [[0, 1], [4, 10]]},
    ]
    base = {"id": seg_id, "run": "run", "game": game_id.split("-")[0], "game_id": game_id, "level": level,
            "actions": [{"action": a} for a in actions], "action_count": len(actions), "completion_tokens": tokens,
            "caveats": [], "context_injections": [], "compaction_indices": [4], "target_start": 5,
            "request_meta": meta, "messages": messages, "tools": TOOLS}
    base.update(extra)
    return base


@pytest.fixture(scope="module")
def template():
    return dataset.ChatTemplate()


def test_records_see_the_exact_request_of_their_time():
    records = sft_records.segment_records(segment())
    assert [r["served_prompt_tokens"] for r in records] == [100, 90, 95]
    texts = [[m["content"] if not isinstance(m["content"], list) else m["content"][0]["text"] for m in r["messages"]]
             for r in records]
    # Before the compaction: level 1 in full, no summary; after it: the summary, not level 1.
    assert texts[0] == ["rules", "board L1", None, "ok 1", "board L2"]
    assert texts[1] == ["rules", "[Context compaction] summary of level one", "board L2", "Plan: up.", "ok 2"]
    assert len(texts[2]) == 7
    # The old behaviour (the level's own messages only) stays available for comparisons.
    old = sft_records.segment_records(segment(), context="segment")
    assert [m.get("content") for m in old[0]["messages"]][0] == "rules" and len(old[0]["messages"]) == 2
    assert old[1]["messages"] == records[1]["messages"]


def test_template_keeps_earlier_reasoning_and_packs_at_compactions(template):
    counter = dataset.TokenCounter()
    packed, turns, checks = dataset.segment_items(segment(), "dev", template, counter)
    # Two sequences: the pre-compaction request alone, then the two requests after the summary.
    assert [p["targets"] for p in packed] == [1, 1] and len(turns) == 2 and len(checks) == 3
    first = turns[0]
    assert "<think>\nlook at level one\n</think>" in first["text"]  # history reasoning is rendered
    assert first["text"].count("<|image_pad|>") == len(first["images"]) == 2
    start, end = first["loss_spans"][0]
    assert first["text"][:start].endswith("<|im_start|>assistant\n<think>\n")
    assert first["text"][start:end].startswith("plan level two\n</think>\n\nPlan: up.")
    assert first["text"][start:end].endswith("</tool_call><|im_end|>")
    assert '<parameter=code>\naction([\'UP\'])\n</parameter>' in first["text"][start:end]
    # The truncated reply gets no loss but stays in the packed sequence as context.
    second = packed[1]
    assert "a very long thought" in second["text"]
    assert all("a very long thought" not in second["text"][a:b] for a, b in second["loss_spans"])
    assert "summary of level one" in second["text"] and "board L1" not in second["text"]
    assert [c["truncated"] for c in checks] == [False, False, True]
    assert [c["python_error"] for c in checks] == [False, True, False]
    # The pre-compaction request lost level 1 in the old context; the rebuilt one keeps it.
    assert checks[0]["segment_only_prompt_tokens"] < checks[0]["prompt_tokens"]
    assert checks[1]["segment_only_prompt_tokens"] == checks[1]["prompt_tokens"]


def test_packed_chunks_follow_prefix_extension():
    contexts = [(3, [0, 1, 2]), (5, [0, 1, 2, 3, 4]), (8, [0, 6, 7]), (10, [0, 6, 7, 8, 9]), (12, [0, 6, 11])]
    assert dataset.packed_chunks(contexts) == [[0, 1], [2, 3], [4]]


def test_filters_drop_injected_context_near_duplicates_and_cap_levels():
    long = ["ACTION1", "ACTION2"] * 10
    segments = [
        segment("a", actions=long, tokens=50),
        segment("b", actions=[*long[:-1], "ACTION3"], tokens=40),  # near twin of a (19/20 equal)
        segment("c", actions=["ACTION4"] * 25, tokens=5),
        segment("d", actions=["ACTION5"] * 30, tokens=5),
        segment("e", actions=["ACTION1"], caveats=["carried_context_omitted:cross_game_notes"]),
        segment("f", actions=["ACTION2"], context_injections=["stuck_procedure"]),
        segment("g", level=3, actions=["ACTION6"]),
    ]
    kept, report = dataset.filter_segments(segments, cap_per_level=2, threshold=0.9)
    # Level 2: b (20 actions, fewer tokens) beats its twin a; then c (25); d is over the cap.
    assert [s["id"] for s in kept] == ["b", "c", "g"]
    assert report["injected_context"] == {"cross_game_notes": 1, "stuck_procedure": 1}
    assert (report["injected_context_segments"], report["near_duplicates"], report["over_level_cap"]) == (2, 1, 1)
    assert dataset.filter_segments(list(reversed(segments)), cap_per_level=2)[0] == kept  # order-independent


def test_budgets_and_level_rounds():
    budgets = dataset.solve_budgets(100, {"dev": 1000, "aux": 1000}, {"dev": 0.7, "aux": 0.1})
    total = 100 + budgets["dev"] + budgets["aux"]
    assert abs(budgets["dev"] / total - 0.7) < 0.01 and abs(budgets["aux"] / total - 0.1) < 0.01
    assert dataset.solve_budgets(100, {"dev": 50, "aux": 1000}, {"dev": 0.7, "aux": 0.1})["dev"] == 50
    with pytest.raises(ValueError):
        dataset.solve_budgets(1, {"a": 1}, {"a": 1.0})
    groups = {"L1": [10, 10, 10], "L2": [30, 5], "L3": [5]}
    # Round 1 takes every level's best (10, 30, 5); round 2 skips nothing that fits.
    assert dataset.select_rounds(groups, 60, lambda x: x) == [10, 30, 5, 10, 5]
    assert dataset.select_rounds(groups, 20, lambda x: x) == [10, 5, 5]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def aux_task(task_id, game, answer="```text\nNO CHANGE\n```", family="predict"):
    return {"id": task_id, "family": family, "game": game, "level": 1, "prompt": f"Predict the board. {task_id}",
            "answer": answer, "verifier": {}, "meta": {}}


def test_build_writes_both_formats_with_budgets_and_the_eval_split(tmp_path, template):
    dev = write_jsonl(tmp_path / "dev.jsonl", [segment(f"d{i}", level=i + 1, actions=[f"A{i}"]) for i in range(4)])
    teacher = write_jsonl(tmp_path / "teacher.jsonl", [segment("t1", game_id="q001-c8ba2de3", level=1)])
    aux = write_jsonl(tmp_path / "aux.jsonl", [aux_task(f"x{i}", g) for i in range(10)
                                              for g in ("q001-c8ba2de3", "ar25-0c556536")]
                      + [aux_task("tr", "q002-4ec7e433", answer=None, family="transition")])
    held = write_jsonl(tmp_path / "held.jsonl", [segment("h1", game_id="q009-6696a439", level=1)])
    sources = dataset.Sources(dev=(dev,), teacher=(teacher,), aux=(aux,), solver=(), eval_teacher=(held,),
                              eval_solver=(), eval_aux=(write_jsonl(tmp_path / "ea.jsonl", [aux_task("e", "q113-2f0e9b1a")]),))
    out = tmp_path / "out"
    stats = dataset.build(out, sources=sources, mixture=dataset.Mixture(teacher_epochs=2), template=template,
                          log=lambda *a: None)
    mix = stats["mixture"]
    assert mix["teacher"]["target_tokens"] == 2 * mix["teacher"]["target_tokens_unique"] > 0
    assert mix["dev"]["share"] <= 0.70 + 1e-9 and mix["aux"]["share"] <= 0.10 + 1e-9
    assert mix["dev"]["segments"] >= 1 and stats["filters"]["dev"]["kept"] == 4
    assert stats["prompt_check"]["dev"]["truncated_targets"] == 4
    rows = {name: [json.loads(line) for line in open(out / f"{name}.jsonl")]
            for name in ("train_packed", "train_turns", "eval_packed", "eval_turns")}
    teacher_rows = [r for r in rows["train_packed"] if r["source"] == "teacher"]
    assert sorted(r["repeat"] for r in teacher_rows) == [0, 0, 1, 1]
    aux_rows = [r for r in rows["train_turns"] if r["source"] == "aux"]
    assert aux_rows and all(r["no_reasoning"] and r["format"] == "single" for r in aux_rows)
    assert {r["id"] for r in aux_rows} == {r["id"] for r in rows["train_packed"] if r["source"] == "aux"}
    for row in rows["train_turns"]:
        a, b = row["loss_spans"][0]
        assert row["text"][b - len("<|im_end|>"):b] == "<|im_end|>" and 0 < a < b
    assert {r["game_id"] for r in rows["eval_turns"]} == {"q009-6696a439", "q113-2f0e9b1a"}
    assert all(r["split"] == "eval" for r in rows["eval_packed"] + rows["eval_turns"])
    assert stats["eval"]["teacher_segments"] == 1 and stats["eval"]["aux_tasks"] == 1
    assert json.loads((out / "stats.json").read_text())["mixture"]["total_target_tokens"] == mix["total_target_tokens"]
    assert not list(out.glob(".candidates-*"))


def test_build_rejects_held_out_games_in_training_sources(tmp_path, template):
    dev = write_jsonl(tmp_path / "dev.jsonl", [segment("x", game_id="q015-544a385b", level=1)])
    sources = dataset.Sources(dev=(dev,), teacher=(), aux=(), solver=(), eval_teacher=(), eval_solver=(), eval_aux=())
    with pytest.raises(ValueError, match="q015"):
        dataset.build(tmp_path / "out", sources=sources, template=template, log=lambda *a: None)
    held = write_jsonl(tmp_path / "held.jsonl", [segment("y", game_id="q001-c8ba2de3", level=1)])
    sources = dataset.Sources(dev=(), teacher=(), aux=(), solver=(), eval_teacher=(held,), eval_solver=(), eval_aux=())
    with pytest.raises(ValueError, match="GENERATED_HOLDOUT"):
        dataset.build(tmp_path / "out", sources=sources, template=template, log=lambda *a: None)


def test_token_counter_counts_images_as_pad_tokens():
    counter = dataset.TokenCounter()
    text = "<|im_start|>user\nhello<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
    plain = text.replace("<|image_pad|>", "")
    assert counter.count(text) - counter.count(plain) >= dataset.IMAGE_TOKENS - 1
