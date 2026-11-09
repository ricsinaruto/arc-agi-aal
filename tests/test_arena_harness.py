"""The evolvable Duck harness directory (prompt addition, playbook, pre-loaded skills)."""

import json

import pytest
from conftest import tool_reply
from test_arena_duck import CAP, WINDOW, arena_agent

from arena.duck import ArenaTransferDuckAgent
from arena.harness import PLAYBOOK_LABEL, SKILLS_HEADER, DuckHarness


def make_harness(tmp_path):
    root = tmp_path / "harness"
    (root / "skills").mkdir(parents=True)
    (root / "prompt.md").write_text("Probe one action at a time before committing to a plan.\n")
    (root / "playbook.md").write_text("- Arrow games: the controllable object is the one that moves.\n")
    (root / "skills" / "boarddiff.py").write_text('"""Cells that differ between two frames.\n\nmore"""\n\ndef diff(a, b):\n    return [(r, c) for r, row in enumerate(a) for c, v in enumerate(row) if b[r][c] != v]\n')
    (root / "skills" / "broken.py").write_text("def nope(:\n    pass\n")
    (root / "CHANGELOG.md").write_text("round 1: added boarddiff\n")
    return root


def test_surfaces_caps_and_validation(tmp_path):
    root = make_harness(tmp_path)
    harness = DuckHarness(root)
    text = harness.prompt_text()
    assert text.startswith("Probe one action") and SKILLS_HEADER in text and "- boarddiff: Cells that differ between two frames. [diff(a, b)]" in text
    assert "broken" not in text
    assert harness.playbook_message() == {"role": "user", "content": PLAYBOOK_LABEL + "- Arrow games: the controllable object is the one that moves."}
    assert harness.validate() == ["skills/broken.py: invalid syntax (line 1); skipped"]
    summary = harness.summary()
    assert summary["skills"] == ["boarddiff"] and summary["problems"] == ["skills/broken.py: invalid syntax (line 1); skipped"]
    namespace = {}
    exec(harness.skills_preamble(), namespace)  # noqa: S102 - the preamble is our own generated code
    assert namespace["skills_names"] == ["boarddiff"] and namespace["diff"]([[0, 1]], [[0, 2]]) == [(0, 1)] and "def diff" in namespace["skills_source"]["boarddiff"]
    (root / "skills" / "cls.py").write_text("class A:\n    pass\n")
    (root / "skills" / "imp.py").write_text("import numpy as np\n")
    (root / "skills" / "ex.py").write_text("def f():\n    return exec('1')\n")
    assert sorted(DuckHarness(root).validate()) == [
        "skills/broken.py: invalid syntax (line 1); skipped",
        "skills/cls.py: class statements are not available in the sandbox (line 1); skipped",
        "skills/ex.py: exec() is not available in the sandbox (line 2); skipped",
        "skills/imp.py: import of 'numpy' is not allowed in the sandbox (line 1); skipped"]
    for name in ("cls", "imp", "ex"):
        (root / "skills" / f"{name}.py").unlink()
    (root / "prompt.md").write_text("x" * 6000)
    assert sorted(DuckHarness(root, prompt_cap_tokens=100).validate()) == ["prompt.md: 6000 chars over the 350-char cap; truncated", "skills/broken.py: invalid syntax (line 1); skipped"]
    with pytest.raises(ValueError, match="Unknown harness caps"):
        DuckHarness(root, notes_cap=1)
    empty = DuckHarness(tmp_path / "missing")
    assert empty.prompt_text() == "" and empty.playbook_message() is None and empty.validate() == [] and empty.skills() == []


async def test_agent_loads_prompt_playbook_and_skills_and_reloads_after_a_restart(tmp_path):
    root = make_harness(tmp_path)

    class Transport:
        payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(kwargs["json"])
            code = "print(skills_names, diff([[0]], [[1]])); action([{'action':'UP'}])" if len(self.payloads) in (1, 3) else "print(skills_names)"
            return tool_reply(code, call_id=f"c{len(self.payloads)}")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, levels=1)
    agent.attach(game, window=WINDOW, cap=CAP, harness=root)
    agent.config.max_actions = 1
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 1
    first = transport.payloads[0]["messages"]
    assert first[0]["role"] == "system" and first[0]["content"].rstrip().endswith("- boarddiff: Cells that differ between two frames. [diff(a, b)]")
    assert "Probe one action at a time" in first[0]["content"]
    assert first[1] == {"role": "user", "content": PLAYBOOK_LABEL + "- Arrow games: the controllable object is the one that moves."}
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    preloads = [e for e in events if e["kind"] == "harness_preload"]
    assert len(preloads) == 1 and preloads[0]["skills"] == ["boarddiff"]
    tool = next(e for e in events if e["kind"] == "duck_python")
    assert "['boarddiff'] [(0, 0)]" in json.dumps(tool["result"])
    stats = result["duck_harness"]
    assert stats["preloads"] == 1 and stats["skills"] == ["boarddiff"] and stats["problems"] == ["skills/broken.py: invalid syntax (line 1); skipped"]
    assert stats["prompt_chars"] > 50 and stats["playbook_chars"] > 40
    plain, _, _, _ = arena_agent(tmp_path / "plain", Transport(), ArenaTransferDuckAgent, levels=1)
    plain.attach(game, window=WINDOW, cap=CAP)
    assert plain.harness is None and "duck_harness" not in plain.result()


def test_exploration_harness_coverage_lists_untried_objects_and_actions():
    from pathlib import Path
    from types import SimpleNamespace as NS

    from arena.harness import DuckHarness
    from inference.utils.segmentation import segment_layer

    root = Path(__file__).resolve().parents[1] / "src/arena/harnesses/exploration"
    harness = DuckHarness(root)
    assert harness.validate() == [] and "coverage" in harness.prompt_text()

    def frame(grid, level=0):
        view = NS(level=level, _grid=grid)
        view.segmentation = segment_layer(grid, [str(i) for i in range(16)])
        return view

    before = [[0] * 10 for _ in range(10)]
    for r in (2, 3):
        for c in (2, 3):
            before[r][c] = 3
    before[7][7] = 5
    after = [row[:] for row in before]
    after[7][7], after[7][8] = 0, 5
    history = [NS(action="", frame=frame(before)), NS(action="MOUSE(row=2, col=2)", frame=frame(before)),
               NS(action="LEFT", frame=frame(after)), NS(action="LEFT", frame=frame(after))]
    env = {"current_frame": frame(after), "history": history, "valid_actions": ["LEFT", "RIGHT", "MOUSE"]}
    exec((root / "skills/coverage.py").read_text(), env)
    text = env["coverage"]()
    assert "untried key actions: RIGHT" in text and "LEFT 2/1" in text
    assert "#2 5 1px rows 7-7 cols 8-8 click MOUSE(row=7, col=8)" in text and "#1 3 4px rows 2-3 cols 2-3: 1/0" in text


def test_play_harness_runs_a_sequence_until_the_first_surprise():
    from pathlib import Path

    from arena.harness import DuckHarness

    root = Path(__file__).resolve().parents[1] / "src/arena/harnesses/play"
    harness = DuckHarness(root)
    assert harness.validate() == [] and "play(" in harness.prompt_text()
    calls = []

    def fake_action(items):
        calls.append(items[0])
        step = len(calls)
        return {"executed": True, "executed_count": 1, "action_display": items[0]["action"], "board_changed": step != 3,
                "level_completed": False, "game_over": False, "done": False, "state": "NOT_FINISHED", "level": 1,
                "transition": {"changed_cells": 4 if step != 3 else 0, "change_bbox": [0, 0, 1, 1]}}

    env = {"action": fake_action}
    exec((root / "skills/play.py").read_text(), env)
    text = env["play"](["UP", {"action": "LEFT"}, ("MOUSE", 3, 4), "DOWN"])
    assert "executed 3 of 4" in text and "step 2 (MOUSE) changed nothing" in text and len(calls) == 3
    assert calls[2] == {"action": "MOUSE", "row": 3, "col": 4}
    calls.clear()
    assert "sequence complete" in env["play"](["UP", "UP"], stop_on_no_change=False)
    calls.clear()
    env["action"] = lambda items: {"executed": False, "executed_count": 0, "stop_reason": "no_change_repeat"}
    assert "step 0 not executed: no_change_repeat" in env["play"](["UP"])
