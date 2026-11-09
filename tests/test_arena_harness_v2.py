"""Harness v2 (failure-signature study edits E1-E5): effect-aware receipts on recorded dev-game frames,
the nudge, the legend, the new-level notice, the compaction handoff and the variant.

The recorded frames (``fixtures/harness_v2_dev_frames.json.gz``) are short windows of DEV games only
(tn36 and bp35 timer/counter bars, tu93, wa30 and m0r0 keyboard movement); no holdout game is read.
"""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from conftest import raw, tool_reply
from test_arena_duck import CAP, WINDOW, arena_agent

from arc_infra.perception import Observation, decode_grid
from arena.compaction import DUCK_SUMMARY_PROMPT
from arena.duck import ArenaTransferDuckAgent
from arena.harness import PLAYBOOK_LABEL, sandbox_violation
from arena.harness_v2 import (
    DEFAULTS,
    EDITS,
    ROOT,
    EffectStuckTracker,
    EffectTracker,
    HarnessV2,
    HarnessV2Dir,
    handoff_prompt,
    new_elements,
    new_level_notice,
    rigid_shift,
    segment,
    settings,
)
from arena.variants import apply_variant
from posttrain.split import DEV, HOLDOUT

REPO = Path(__file__).resolve().parents[1]
CONTROL = json.loads((REPO / "configs/arena_30m.json").read_text())
FIXTURE = Path(__file__).parent / "fixtures/harness_v2_dev_frames.json.gz"
CHARS = "WwgGcBMPRbSYOrNp"
BAND = 3


def in_band(r, c):
    return min(r, c) < BAND or max(r, c) >= 64 - BAND


def replay(game):
    """``[(step, effect, before, after)]`` of one recorded window through a fresh tracker, with the
    same learning rule as ``HarnessV2.observe_step`` (no learning on RESET, level changes, terminal frames)."""
    data = json.loads(gzip.decompress(FIXTURE.read_bytes()))[game]
    tracker = EffectTracker(settings({}))
    before, levels = decode_grid(data["start"]["frames"][-1]), data["start"]["levels"]
    rows = []
    for step in data["steps"]:
        action = step["action"]
        name = action["action"]
        frames = [decode_grid(f) for f in step["frames"]]
        changed = step["levels"] != levels
        effect = tracker.observe(before, frames, identity=(name, action.get("x"), action.get("y")), name=name,
                                 click=(action.get("y"), action.get("x")) if name == "ACTION6" else None,
                                 learn=not (name == "RESET" or changed or step["state"] != "NOT_FINISHED"))
        rows.append((step, effect, before, frames[-1], tracker.masked_key(before), tracker.masked_key(frames[-1])))
        if changed:
            tracker.reset_level()
        before, levels = frames[-1], step["levels"]
    return rows, tracker


def test_fixture_holds_dev_games_only():
    games = set(json.loads(gzip.decompress(FIXTURE.read_bytes())))
    assert games <= set(DEV) and not games & set(HOLDOUT)


@pytest.mark.parametrize("game", ["tn36", "bp35", "tu93", "wa30", "m0r0"])
def test_a_gameplay_change_is_never_masked(game):
    rows, _ = replay(game)
    for step, effect, before, after, _, _ in rows:
        diff = before != after
        interior = [(r, c) for r, c in np.argwhere(diff) if not in_band(r, c)]
        if interior:  # anything that changed off the border band is always gameplay
            assert effect["gameplay_changed"] and effect["gameplay_cells"] >= len(interior), (game, step["line"])
        assert effect["gameplay_cells"] + effect["clock_cells"] == int(diff.sum())
        assert effect["clock_only"] == (bool(diff.any()) and not effect["gameplay_changed"])


def test_tn36_timer_bar_is_learned_and_its_ticks_are_not_gameplay():
    rows, tracker = replay("tn36")
    (clock,) = tracker.clocks()
    assert (clock.axis, clock.index, clock.src, clock.dst) == ("row", 1, 9, 3)
    learned = next(i for i, row in enumerate(rows) if row[1]["events"])
    assert learned == 2  # third click: three ticks under three click targets, alone on the board
    for step, effect, before, after, key_before, key_after in rows[learned:]:
        interior = bool((before != after)[BAND:64 - BAND, BAND:64 - BAND].any())
        if step["action"]["action"] == "RESET":
            continue
        assert effect["gameplay_changed"] == interior
        if not interior:
            assert effect["clock_only"] and effect["clock_cells"] == 1 and key_before == key_after
    # the run-program ball click: seven frames, the final board is the start plus the timer tick
    animated = rows[12][1]
    assert animated["animated_effect"] and not animated["gameplay_changed"] and animated["effect"]
    assert animated["animation"]["frames"] == 7 and animated["animation"]["final_equals_start"]
    assert animated["animation"]["peak_cells"] > 0


def test_bp35_step_counter_scroll_and_undo():
    rows, tracker = replay("bp35")
    (clock,) = tracker.clocks()
    assert (clock.axis, clock.index, clock.src, clock.dst) == ("row", 63, 0, 15)
    click = rows[4][1]  # a click that only advanced the counter
    assert click["clock_only"] and not click["effect"]
    right, undo = rows[7], rows[8]
    assert right[1]["view_shift"]["share"] >= 0.9 and undo[1]["view_shift"]["share"] >= 0.9
    assert (right[1]["view_shift"]["dr"], right[1]["view_shift"]["dc"]) == (
        -undo[1]["view_shift"]["dr"], -undo[1]["view_shift"]["dc"])
    assert undo[0]["action"]["action"] == "ACTION7" and undo[5] == right[4]  # ACTION7 restored the pre-scroll board


def test_keyboard_games_learn_their_bars_but_every_move_stays_gameplay():
    for game, lines in (("tu93", {("row", 63)}), ("m0r0", {("row", 0), ("row", 63)}), ("wa30", {("row", 63)})):
        rows, tracker = replay(game)
        seen = {(e[0], e[1]) for _, effect, *_ in rows for kind, t in effect["events"] if kind == "clock"
                for e in [(t.axis, t.index)]}
        assert seen == lines, game
        moves = [effect for step, effect, before, after, *_ in rows
                 if step["action"]["action"] != "RESET" and (before != after)[BAND:64 - BAND, BAND:64 - BAND].any()]
        assert moves and all(e["gameplay_changed"] for e in moves)


def _board():
    return np.zeros((64, 64), dtype=np.uint8)


def test_synthetic_movers_paint_and_single_actions_never_become_clocks():
    cfg = settings({})
    # a 1-px player walking along the top band row, left and right: changes go both ways
    tracker, board = EffectTracker(cfg), _board()
    x = 10
    for i in range(12):
        after = board.copy()
        step = 1 if i % 6 < 3 else -1
        after[1, x], after[1, x + step] = 0, 7
        board[1, x] = 7
        tracker.observe(board, [after], identity=("ACTION4" if step > 0 else "ACTION3", None, None), name="X")
        board, x = after, x + step
    assert not tracker.clocks()
    # clicks that paint the clicked band cell: the change is the click's own effect
    tracker, board = EffectTracker(cfg), _board()
    for c in range(10, 20):
        after = board.copy()
        after[0, c] = 4
        tracker.observe(board, [after], identity=("ACTION6", c, 0), name="ACTION6", click=(0, c))
        board = after
    assert not tracker.clocks()
    # a bar that ticks under one action only (a charge meter) is not action-independent
    tracker, board = EffectTracker(cfg), _board()
    board[63, :] = 9
    for c in range(63, 55, -1):
        after = board.copy()
        after[63, c] = 3
        tracker.observe(board, [after], identity=("ACTION5", None, None), name="ACTION5")
        board = after
    assert not tracker.clocks()
    # the same bar inside the board (off the band) is never learned
    tracker, board = EffectTracker(cfg), _board()
    board[30, :] = 9
    for i, c in enumerate(range(63, 55, -1)):
        after = board.copy()
        after[30, c] = 3
        tracker.observe(board, [after], identity=(f"ACTION{1 + i % 4}", None, None), name="X")
        board = after
    assert not tracker.clocks()


def test_a_reverse_change_revokes_a_learned_clock():
    tracker, board = EffectTracker(settings({})), _board()
    board[0, :] = 9
    for i, c in enumerate(range(63, 58, -1)):
        after = board.copy()
        after[0, c] = 3
        effect = tracker.observe(board, [after], identity=(f"ACTION{1 + i % 2}", None, None), name="X")
        board = after
    assert tracker.clocks() and effect["clock_only"]
    after = board.copy()
    after[0, 60] = 9  # the "timer" refilled without a RESET: it was not a clock
    effect = tracker.observe(board, [after], identity=("ACTION1", None, None), name="X")
    assert not tracker.clocks() and effect["gameplay_changed"] and effect["events"][0][0] == "revoked"
    after2 = after.copy()
    after2[0, 58] = 3
    assert tracker.observe(after, [after2], identity=("ACTION2", None, None), name="X")["gameplay_changed"]


def test_rigid_shift_detects_a_pan_only():
    board = np.random.default_rng(0).integers(0, 4, (64, 64)).astype(np.uint8)
    panned = np.roll(board, 5, axis=1)
    assert rigid_shift(board, panned)["dc"] == 5
    assert rigid_shift(board, np.random.default_rng(1).integers(0, 4, (64, 64)).astype(np.uint8)) is None


# --- E2, E4, E5 host-side state -------------------------------------------------------------------


def test_nudge_fires_once_per_streak_after_three_requests_and_records_the_follow_up():
    state = HarnessV2(DEFAULTS, skills=["check_model"])
    for _ in range(2):
        state.note_request()
        assert state.nudge() is None
    state.note_request()
    note = state.nudge()
    assert note.startswith("NOTE: 3 requests since your last action") and "check_model(predict, observe)" in note
    state.note_request()
    assert state.nudge() is None  # once per streak
    state.end_batch({"executed_count": 2})
    assert state.stats["nudges"] == 1 and state.stats["nudges_followed"] == 1 and state.stats["actions_after_nudge"] == 2
    plain = HarnessV2(DEFAULTS)
    for _ in range(3):
        plain.note_request()
    assert "check_model" not in plain.nudge()
    late = HarnessV2(DEFAULTS)
    for _ in range(3):
        late.note_request()
    late.nudge()
    for _ in range(3):
        late.note_request()  # the window of two requests closes without an action
    late.end_batch({"executed_count": 1})
    assert late.stats["nudges_followed"] == 0
    assert HarnessV2({**DEFAULTS, "nudge": False}).nudge() is None


def test_compaction_window_counts_requests_and_tokens_before_the_next_action():
    state = HarnessV2(DEFAULTS)
    state.note_compaction("3. Action/effect table:\nUP moves the player VERIFIED step 4\nRefuted on this level: none")
    for tokens in (900, 700, 300):
        state.note_reply({"completion_tokens": tokens})
    state.end_batch({"executed_count": 1})
    stats = state.stats
    assert stats["compactions"] == stats["compactions_closed"] == 1
    assert stats["compaction_noaction_requests"] == 2 and stats["compaction_noaction_tokens"] == 1600
    assert stats["handoff_verified_lines"] == 1 and stats["handoff_refuted_sections"] == 1


def test_handoff_prompt_replaces_items_three_and_five_and_adds_conventions():
    prompt = handoff_prompt(DUCK_SUMMARY_PROMPT)
    assert "3. Action/effect table" in prompt and "VERIFIED or HYPOTHESIS" in prompt
    assert "5. Refuted on this level" in prompt and "9. Conventions" in prompt
    assert "3. Action model" not in prompt and "5. Actions already tried" not in prompt
    assert prompt.endswith("Be precise and compact; prefer facts over narrative.")
    assert "3. Action model" in DUCK_SUMMARY_PROMPT  # the control's prompt is untouched


def test_new_level_notice_lists_new_colours_and_shapes():
    first = _board()
    first[10:13, 10:13] = 5
    second = first.copy()
    second[40:42, 20:24] = 12  # new colour
    second[50:52, 50:51] = 5  # a known colour, new shape
    earlier = [segment(first, CHARS, 400)]
    colours, objects, total = new_elements(segment(second, CHARS, 400), earlier)
    assert colours == [CHARS[12]] and total == 2 and objects[0]["color"] == CHARS[12]
    text = new_level_notice(2, colours, objects, total)
    assert text.startswith("New on level 2") and f"colours never seen before: {CHARS[12]}" in text
    assert "[40, 20, 41, 23]" in text
    assert new_level_notice(3, [], [], 0).startswith("New on level 3: nothing")


def test_effect_stuck_notice_rewrites_the_evidence_sentence():
    state = HarnessV2(DEFAULTS)
    cfg = NS(stuck_no_change_actions=2, stuck_turns=99, stuck_notice_turns=1)
    stuck = EffectStuckTracker(cfg, state)
    stuck.note_action(False)
    stuck.note_action(False)
    stuck.note_turn()
    state.board, state.levels = _board(), 0
    state.animated[(0, state.tracker.masked_key(state.board))] = ["MOUSE(row=5, col=5)"]
    text = stuck.notice(1, ["UP"])
    assert "2 consecutive actions changed only the clock/counter or nothing" in text
    assert "Actions with no gameplay effect on this board: UP." in text
    assert "only mid-animation (the board returned to its start): MOUSE(row=5, col=5); these are not no-ops" in text
    assert "Re-derive the goal" in text


# --- the harness directory ------------------------------------------------------------------------


def test_harness_directory_composes_the_enabled_edits_only():
    full = HarnessV2Dir(ROOT, DEFAULTS)
    assert full.validate() == []
    prompt = full.prompt_text()
    assert "ACTION7 is usually UNDO" in prompt and "`gameplay_changed`" in prompt
    assert [name for name, _, _ in full.skills()] == ["check_model", "effects", "new_elements"]
    playbook = full.playbook_message()["content"]
    assert playbook.startswith(PLAYBOOK_LABEL) and "its goal is refuted" in playbook and "VERIFIED lines" in playbook
    assert full.summary()["harness_v2"] == list(EDITS)
    for path in (ROOT / "skills").glob("*.py"):
        assert sandbox_violation(__import__("ast").parse(path.read_text())) is None
    legend_off = HarnessV2Dir(ROOT, {**DEFAULTS, "legend": False, "nudge": False, "handoff": False})
    assert "ACTION7" not in legend_off.prompt_text() and "check_model" not in legend_off.prompt_text()
    assert "VERIFIED lines" not in legend_off.playbook_message()["content"]
    none = HarnessV2Dir(ROOT, {edit: False for edit in EDITS})
    assert none.prompt_text() == "" and none.playbook_message() is None and none.skills() == []
    with pytest.raises(ValueError, match="Unknown harness_v2 settings"):
        settings({"nudge_every": 3})


def _frame(grid, level=1, step=0):
    from inference.utils.segmentation import segment_layer

    view = NS(level=level, step=step, ascii="", _grid=grid)
    view.segmentation = segment_layer(grid, CHARS)
    return view


def test_skills_read_receipts_check_models_and_list_new_elements():
    root = ROOT / "skills"
    a, b = _frame([[0, 0], [0, 0]], step=1), _frame([[0, 1], [0, 0]], step=2)
    c = _frame([[0, 1], [0, 0]], step=3)
    transitions = [
        NS(action="UP", before_frame=a, after_frame=b, result={"gameplay_changed": True, "gameplay_cells": 1,
                                                                "gameplay_bbox": [0, 1, 1, 2], "clock_only": False}),
        NS(action="MOUSE(row=0, col=0)", before_frame=b, after_frame=c,
           result={"gameplay_changed": False, "clock_only": True,
                   "animation": {"frames": 7, "peak_cells": 12, "peak_bbox": [0, 0, 2, 2], "final_equals_start": True}}),
    ]
    env = {"current_frame": c, "transitions": transitions}
    exec((root / "effects.py").read_text(), env)  # noqa: S102 - our own skill source
    text = env["effects"](8)
    assert "step 2 UP: 1 gameplay cells in bbox [0, 1, 1, 2]" in text
    assert "step 3 MOUSE(row=0, col=0): EFFECT ONLY MID-ANIMATION: up to 12 cells" in text
    exec((root / "check_model.py").read_text(), env)  # noqa: S102
    observe = lambda frame: frame._grid[0][1]  # noqa: E731
    assert env["check_model"](lambda s, action: 1, observe) == "all 2 recorded transitions on this level reproduced"
    assert env["check_model"](lambda s, action: s, observe).startswith("MISMATCH at step 2 (UP)")
    assert env["check_model"](lambda s, action: 1 / 0, observe).startswith("ERROR at step 2 (UP): ZeroDivisionError")
    level1 = [[5 if 1 <= r <= 2 and 1 <= col <= 2 else 0 for col in range(50)] for r in range(50)]
    level2 = [row[:] for row in level1]
    level2[4][4] = 12  # the background (over 2,048 cells) reshapes around it and is not listed
    history = [NS(action="", frame=_frame(level1, 1, 0)), NS(action="UP", frame=_frame(level1, 1, 1)),
               NS(action="UP", frame=_frame(level2, 2, 2))]
    env.update(history=history, current_frame=history[-1].frame)
    exec((root / "new_elements.py").read_text(), env)  # noqa: S102
    assert env["new_elements"]().startswith(f"1 new objects on level 2: {CHARS[12]} 1px bbox [4, 4, 4, 4]")


# --- end to end through the arena agent ----------------------------------------------------------


class TimerEnv:
    """A timer bar in the top band ticks on every action; clicking the 3x3 object moves it right
    by four cells; two hits clear the level, and level 2 adds an object of a new colour."""

    def __init__(self):
        self.level, self.calls = 0, []
        self.new_board()
        self.observation_space = self.raw()

    def new_board(self):
        board = _board()
        board[1, 2:62] = 9
        board[30:33, 30:33] = 5
        if self.level:
            board[40:42, 10:14] = 12
        self.board, self.hits = board, 0

    def raw(self):
        return raw(Observation([self.board.copy()], self.level, 3, "NOT_FINISHED", []))

    def step(self, action, data):
        self.calls.append((action.name, data))
        left = np.nonzero(self.board[1] == 9)[0]
        if len(left):
            self.board[1, left.max()] = 3
        if action.name == "ACTION6" and self.board[data["y"], data["x"]] == 5:
            cells = np.argwhere(self.board == 5)
            self.board[self.board == 5] = 0
            for r, c in cells:
                self.board[r, c + 4] = 5
            self.hits += 1
            if self.hits == 2:
                self.level += 1
                self.new_board()
        self.observation_space = self.raw()
        return self.observation_space

    def reset(self):
        self.new_board()
        self.observation_space = self.raw()
        return self.observation_space


CODES = [
    "print(skills_names); print(effects(3))",
    "print(check_model(lambda s, a: s, lambda f: f.level))",
    "x = 1",
    "action([{'action': 'MOUSE', 'row': 10, 'col': 10}])",
    "action([{'action': 'MOUSE', 'row': 12, 'col': 12}])",
    "action([{'action': 'MOUSE', 'row': 14, 'col': 14}])",
    "r = action([{'action': 'MOUSE', 'row': 14, 'col': 14}]); print(r.get('stop_reason'), r.get('error'))",
    "action([{'action': 'MOUSE', 'row': 31, 'col': 31}])",
    "action([{'action': 'MOUSE', 'row': 31, 'col': 35}])",
    "print(new_elements()); action([{'action': 'MOUSE', 'row': 10, 'col': 10}])",
]


async def test_agent_runs_all_five_edits_end_to_end(tmp_path):
    class Transport:
        payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(kwargs["json"])
            index = len(self.payloads) - 1
            if index >= len(CODES):
                raise RuntimeError(f"unexpected request {index + 1}")
            return tool_reply(CODES[index], call_id=f"c{len(self.payloads)}")

    transport = Transport()
    agent, _, game, _ = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, levels=1)
    env = TimerEnv()
    agent.arc = NS(make=lambda *a, **kw: env)
    agent.attach(game, window=WINDOW, cap=CAP, harness=ROOT, harness_v2=DEFAULTS)
    agent.config.max_actions = 6
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 6 and result["levels"] == 1
    first = transport.payloads[0]["messages"]
    assert "ACTION7 is usually UNDO" in first[0]["content"] and "- effects:" in first[0]["content"]
    assert first[1]["content"].startswith(PLAYBOOK_LABEL) and "VERIFIED lines" in first[1]["content"]
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    tools = [e for e in events if e["kind"] == "duck_python"]
    assert "['check_model', 'effects', 'new_elements']" in tools[0]["result"]["stdout"]
    assert "no actions on this level yet" in tools[0]["result"]["stdout"]
    assert "all 0 recorded transitions" in tools[1]["result"]["stdout"]
    assert tools[2]["result"]["note"].startswith("NOTE: 3 requests since your last action")
    assert "note" not in tools[1]["result"] and "note" not in tools[3]["result"]
    # E1: the third empty click taught the timer bar; its receipt is clock-only, the raw flag stays
    third = tools[5]["result"]["result"]
    assert third["board_changed"] is True and third["gameplay_changed"] is False and third["clock_only"] is True
    assert third["transition"]["changed_cells"] == 1
    assert "no_change_repeat" in tools[6]["result"]["stdout"] and "no gameplay effect" in tools[6]["result"]["stdout"]
    move = tools[7]["result"]["result"]
    assert move["gameplay_changed"] is True and move["gameplay_cells"] == 18
    # E4: the level-up turn carries the notice; the skill agrees
    level_up = transport.payloads[9]["messages"]
    user = next(m for m in reversed(level_up) if m["role"] == "user")
    text = user["content"] if isinstance(user["content"], str) else user["content"][0]["text"]
    assert f"New on level 2 (not on any earlier level's board): colours never seen before: {CHARS[12]}" in text
    assert f"1 new objects on level 2: {CHARS[12]} 8px bbox [40, 10, 41, 13]" in tools[9]["result"]["stdout"]
    stats = result["harness_v2"]
    assert stats["edits"] == list(EDITS)
    assert stats["nudges"] == stats["nudges_followed"] == 1 and stats["actions_after_nudge"] == 1
    assert stats["clock_tracks"] == 1 and stats["clock_only"] == 1 and stats["refusals"] == 1
    assert stats["masked_key_repeats"] == 1 and stats["raw_key_repeats"] == 0  # the full-board rule would not refuse it
    assert stats["new_level_notices"] == 1 and stats["new_colours_listed"] == 1
    assert stats["check_model_calls"] == 1 and stats["effects_calls"] == 1 and stats["new_elements_calls"] == 1
    assert stats["clocks"] == []  # the mask is per level: level 2 has seen one action so far
    clocks = [e for e in events if e["kind"] == "harness_v2_clock"]
    assert [(e["event"], e["axis"], e["index"], e["src"], e["dst"]) for e in clocks] == [("clock", "row", 1, 9, 3)]
    assert [e["kind"] for e in events if e["kind"] in ("harness_v2_nudge", "new_level_notice")] == [
        "harness_v2_nudge", "new_level_notice"]


async def test_summary_request_uses_the_handoff_prompt_only_with_the_flag(tmp_path):
    class Transport:
        async def post(self, url, **kwargs):
            raise AssertionError("no request expected")

    agent, _, game, _ = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    prompts = []

    async def aside(messages, prompt, **kwargs):
        prompts.append(prompt)
        return "World model: x\n3. UP VERIFIED step 2", {"usage": {}}, 0.1, 100

    agent.aside_request = aside
    agent.obs = NS(levels=0, total_levels=3)
    await agent.summary_request([])
    agent.attach(game, window=WINDOW, cap=CAP, harness_v2=DEFAULTS)
    await agent.summary_request([])
    agent.attach(game, window=WINDOW, cap=CAP, harness_v2={**DEFAULTS, "handoff": False})
    await agent.summary_request([])
    assert prompts == [DUCK_SUMMARY_PROMPT, handoff_prompt(DUCK_SUMMARY_PROMPT), DUCK_SUMMARY_PROMPT]


def test_nudge_rewrites_the_public_tool_dispatch_result(tmp_path):
    from inference.agent.tool_agent import _ToolDispatchResult

    class Transport:
        async def post(self, url, **kwargs):
            raise AssertionError("no request expected")

    agent, _, game, _ = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    agent.attach(game, window=WINDOW, cap=CAP, harness_v2=DEFAULTS)
    agent.harness_v2.requests_since_action = 3
    out = agent.run_python_tool(lambda s, a: _ToolDispatchResult(json.dumps({"tool": "python", "stdout": "1"}, indent=2)),
                                None, {"code": "print(1)"})
    assert json.loads(out.content)["note"].startswith("NOTE: 3 requests") and out.step_executed is False
    acted = agent.run_python_tool(lambda s, a: _ToolDispatchResult("{}", step_executed=True), None, {"code": "x"})
    assert acted.content == "{}"
    agent.attach(game, window=WINDOW, cap=CAP)
    assert agent.harness_v2 is None and "harness_v2" not in agent.result()


def test_all_edits_off_keeps_the_control_surfaces(tmp_path):
    from duck_transfer.feedback import StuckTracker

    class Transport:
        async def post(self, url, **kwargs):
            raise AssertionError("no request expected")

    agent, _, game, _ = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent)
    off = {**DEFAULTS, **{edit: False for edit in EDITS}}
    agent.attach(game, window=WINDOW, cap=CAP, harness_v2=off)
    assert type(agent.stuck) is StuckTracker and agent.harness.prompt_text() == ""
    assert agent.harness_playbook is None and agent.harness.skills() == []
    assert agent.harness_v2.nudge() is None
    agent.attach(game, window=WINDOW, cap=CAP, harness_v2=DEFAULTS)
    assert isinstance(agent.stuck, EffectStuckTracker)
    with pytest.raises(ValueError, match="replace the stuck tracker"):
        agent.attach(game, window=WINDOW, cap=CAP, harness_v2=DEFAULTS, procedure={"collapse_window": 30})


def test_variant_adds_the_harness_directory_and_the_flags_only():
    base = apply_variant(CONTROL, "reclaim-fixed-2x")
    candidate = apply_variant(CONTROL, "fixed-2x-harness-v2")
    meta = candidate.pop("arena_variant")
    base.pop("arena_variant")
    assert {k for k in base.keys() | candidate.keys() if base.get(k) != candidate.get(k)} == {"duck_harness", "harness_v2"}
    assert candidate["harness_v2"] == DEFAULTS and all(candidate["harness_v2"][edit] for edit in EDITS)
    assert (REPO / candidate["duck_harness"] / "skills/effects.py").is_file()
    assert meta["reference_variant"] == "reclaim-fixed-2x" and meta["checkpoint_and_serving_unchanged"]
