"""Envs multi-game: compaction summaries as trained policy replies, and restores across a summary compaction."""

import importlib.util
import json
from pathlib import Path

from arena import restore
from posttrain import rl_rollouts

ROOT = Path(__file__).resolve().parents[1]


def rl_job():
    spec = importlib.util.spec_from_file_location("rl_job", ROOT / "scripts/posttrain/rl_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(n, keep, messages, purpose="gameplay", prompt=0, completion=10, step=0):
    return {"n": n, "step": step, "gameplay": purpose == "gameplay", "purpose": purpose, "keep": keep,
            "messages": messages, "template": {}, "reply": {"role": "assistant", "content": f"r{n}"},
            "finish_reason": "stop", "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
            **({"tools": [{"type": "function"}]} if n == 0 else {})}


def episode():
    s, o1, a1, o2 = ({"role": "system", "content": "s"}, {"role": "user", "content": "o1"},
                     {"role": "assistant", "content": "r0"}, {"role": "user", "content": "o2"})
    ask = {"role": "user", "content": "summarise"}
    return [
        row(0, 0, [s, o1], prompt=100),
        row(1, 2, [a1, o2], prompt=200),
        # full-prefix summary: the whole context plus the instruction (extends the last prompt)
        row(2, 4, [{"role": "assistant", "content": "r1"}, ask], purpose="summary", prompt=320, completion=50),
        # after the compaction: system, summary, the kept tail
        row(3, 1, [{"role": "user", "content": "[summary] r2"}, o2], prompt=150),
    ]


def test_summary_rows_off_by_default(monkeypatch):
    monkeypatch.delenv("ARC_RL1_TRAIN_SUMMARY", raising=False)
    assert [c["request"] for c in rl_rollouts.captures(episode())] == [0, 1, 3]


def test_summary_rows_captured_and_windowed(monkeypatch):
    monkeypatch.setenv("ARC_RL1_TRAIN_SUMMARY", "1")
    caps = rl_rollouts.captures(episode())
    assert [c["request"] for c in caps] == [0, 1, 2, 3]
    summary = caps[2]
    assert summary["payload"]["messages"][-1]["content"] == "summarise"
    assert summary["payload"]["tools"] == [{"type": "function"}]  # carried from the first row, as served
    ends = rl_job().plan_windows(caps, 4096)
    # the post-compaction request starts a new window; the summary closes the pre-compaction one with the gameplay
    # replies before it (its prompt extends theirs)
    assert ends == [3, 2]


def test_credited_ends_cover_summary_window(monkeypatch):
    monkeypatch.setenv("ARC_RL1_TRAIN_SUMMARY", "1")
    rows = [dict(r, w=0.5) for r in episode()]
    assert sorted(rl_job().credited_ends({"replies": rows}, 4096)) == [2, 3]
    monkeypatch.delenv("ARC_RL1_TRAIN_SUMMARY")
    # without the switch the summary is dropped: its window ends at the last gameplay reply before it
    assert sorted(rl_job().credited_ends({"replies": rows}, 4096)) == [1, 2]


def write_log(folder, kinds):
    with open(folder / "log.jsonl", "w") as file:
        for step, kind in kinds:
            file.write(json.dumps({"kind": kind, "step": step}, separators=(",", ":")) + "\n")


def test_unsafe_from_allows_summary_compactions_when_switched(tmp_path, monkeypatch):
    write_log(tmp_path, [(5, "fraction_compaction"), (9, "duck_python_interrupt")])
    monkeypatch.delenv("ARC_RL1_TRAIN_SUMMARY", raising=False)
    assert restore.unsafe_from(tmp_path)[0] == 5
    monkeypatch.setenv("ARC_RL1_TRAIN_SUMMARY", "1")
    assert restore.unsafe_from(tmp_path)[0] == 9
    assert restore.unsafe_from(tmp_path, allow=())[0] == 5
