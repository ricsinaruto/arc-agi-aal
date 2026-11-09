"""hx5 model check (arena.sim_check): the child block on a toy grid game, and the host texts."""
import textwrap

import pytest

from arena import dfr_harness as dfr
from arena import sim_check as sc

SIZE = 16          # board; the interior the checks compare is rows/cols 4..11
WALLS = {(r, 7) for r in range(4, 9)}  # the way round is row 9
GOAL = (7, 10)
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


class Frame:
    def __init__(self, player, step, level=1, timer=0):
        grid = [[0] * SIZE for _ in range(SIZE)]
        for r, c in WALLS:
            grid[r][c] = 5
        grid[GOAL[0]][GOAL[1]] = 3
        grid[player[0]][player[1]] = 9
        for c in range(min(timer, SIZE)):  # a step bar in the outer ring
            grid[15][c] = 2
        self._grid, self.step, self.level, self.player = grid, step, level, player
        self.ascii = "\n".join("".join(str(v) for v in row) for row in grid)
        self.shape = (SIZE, SIZE)


class Transition:
    def __init__(self, action, before, after):
        self.action, self.before_frame, self.after_frame, self.frame, self.result = action, before, after, after, {}


def true_step(player, action):
    if isinstance(action, dict):  # a click does nothing in this game
        return player
    dr, dc = MOVES[action]
    target = (player[0] + dr, player[1] + dc)
    if target in WALLS or not (4 <= target[0] < 12 and 4 <= target[1] < 12):
        return player
    return target


class World:
    """The toy game behind ``action``: executes whole lists, stops at a no-op like the host's batch guard."""

    def __init__(self, player=(6, 5)):
        self.frame = Frame(player, 0)
        self.transitions, self.calls, self.level = [], [], 1
        self.runtime = {"current_frame": self.frame, "transitions": self.transitions,
                        "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"], "action": self.act}

    def act(self, actions):
        items = [actions] if isinstance(actions, (str, dict)) else list(actions)
        self.calls.append(list(items))
        result = {"executed": False}
        for item in items:
            before = self.frame
            player = true_step(before.player, item)
            won = player == GOAL
            self.level += won
            self.frame = Frame((6, 5) if won else player, before.step + 1, self.level, timer=before.step + 1)
            label = f"MOUSE(row={item['row']}, col={item['col']})" if isinstance(item, dict) else item
            self.transitions.append(Transition(label, before, self.frame))
            self.runtime["current_frame"] = self.frame
            changed = won or player != before.player
            result = {"executed": True, "gameplay_changed": changed, "level_completed": won}
            if won or not changed:
                break
        return result


def child(world, mode=True, table=True):
    """The child block run on ``world``: its names, with model mode switched as the host would."""
    source = (sc.CHILD_BLOCK.replace("__SC_CONSTANTS__", repr((sc.CAP, sc.FIT, sc.MIN_LEDGER, 0.2, 4, 140)))
              .replace("__SC_CAP_DETAIL__", repr(sc.CAP_DETAIL))
              .replace("__SC_MISMATCH_DETAIL__", repr(sc.MISMATCH_DETAIL)))
    code = ("def install(runtime_globals):\n    def _refresh_state(state_payload):\n        pass\n"
            + textwrap.indent(textwrap.dedent(source), "    ") + "\n    return locals()\n")
    names = {}
    exec(compile(code, "sim_check_child", "exec"), names)
    scope = names["install"](world.runtime)
    scope["_refresh_state"]({"simcheck": {"mode": mode, "table": table, "level": 1}})
    return scope


def encode(frame):
    return frame.player


def play(world, scope, moves):
    for move in moves:
        scope["action"]([move])


def test_flag_is_opt_in_and_listed():
    for flag in ("simcheck", "evidence"):
        assert flag in dfr.FLAGS and flag in dfr.OPT_IN and dfr.settings(True)[flag] is False
    assert dfr.settings({"evidence": True})["simcheck"] is False


def test_ledger_and_evidence_table():
    world = World()
    scope = child(world, mode=False)
    assert "MODEL" not in world.runtime
    play(world, scope, ["RIGHT", "RIGHT", "LEFT", "RIGHT", "RIGHT", "UP", "DOWN"])  # (6,6) blocked (6,5) (6,6) blocked
    scope["action"]([{"action": "MOUSE", "row": 5, "col": 5}])
    rows = scope["_sc_ledger"]()
    assert [r["changed"] for r in rows] == [True, False, True, True, False, True, True, False]
    assert [r["repeat"] for r in rows] == [False, False, False, True, True, False, False, False]  # step bar ignored
    assert not any(r["conflict"] for r in rows)
    report = scope["_sc_take"]()
    assert "fit" not in report and report["mode"] is None and report["table"] == 1
    assert report["ledger"] == {
        "transitions": 8, "wins": 0, "single": [], "cells": 1, "single_cells": 1, "untried": [], "conflicts": [],
        "actions": {"DOWN": [1, 1, 0, 1], "LEFT": [1, 1, 0, 1], "MOUSE": [1, 0, 1, 1], "RIGHT": [4, 2, 2, 2],
                    "UP": [1, 1, 0, 1]}}
    line = sc.evidence_line(report, 1)
    assert line == ("Evidence (the harness's record of level 1, 8 actions; actions that changed the board / tries): "
                    "DOWN 1/1, LEFT 1/1, MOUSE 0/1 (1 cells clicked; 1 did nothing on the one board they were tried "
                    "on), RIGHT 2/4, UP 1/1.")
    assert sc.check_line(report, 1) is None


def test_a_no_op_seen_on_one_board_is_flagged_and_untried_actions_are_listed():
    world = World()
    world.runtime["valid_actions"] = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "RESET"]
    scope = child(world, mode=False)
    play(world, scope, ["LEFT", "LEFT", "LEFT"])  # (6,4), then the edge twice: one board
    line = sc.evidence_line(scope["_sc_take"](), 1)
    assert "LEFT 1/3." in line and "ONE board" not in line
    world = World(player=(6, 4))
    world.runtime["valid_actions"] = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "RESET"]
    scope = child(world, mode=False)
    play(world, scope, ["LEFT", "LEFT"])
    line = sc.evidence_line(scope["_sc_take"](), 1)
    assert "LEFT 0/2 (tried on ONE board only" in line
    assert "Never tried on this level: UP, DOWN, RIGHT, SPACE." in line


def test_a_right_model_fits_a_wrong_one_shows_the_first_mismatch_and_a_constant_encode_misses_changes():
    world = World()
    scope = child(world)
    play(world, scope, ["RIGHT", "RIGHT", "LEFT", "RIGHT", "RIGHT", "UP"])
    model = world.runtime["MODEL"]
    report = model.register(encode, true_step)
    assert report["reproduced"] == 6 and report["wrong"] == 0 and report["lists_unlimited"] is True
    no_walls = lambda state, action: (state[0] + MOVES[action][0], state[1] + MOVES[action][1])  # noqa: E731
    report = model.register(encode, no_walls)
    assert report["reproduced"] == 4 and report["wrong"] == 2 and report["lists_unlimited"] is False
    assert report["first_mismatches"][0] == {"step": 2, "action": "RIGHT", "where": "[1]", "predicted": "7",
                                             "actual": "6"}
    assert "wrong: step 2 RIGHT: state[1]: predicted 7, actual 6" in sc.check_line(scope["_sc_take"](), 1)
    report = model.register(lambda frame: 0, lambda state, action: state)
    assert report["reproduced"] == 2 and report["missed_changes"] == 4 and report["lists_unlimited"] is False
    line = sc.check_line(scope["_sc_take"](), 1)
    assert "reproduces 2 of 6 transitions of level 1 (4 missed changes)" in line and "cut after 3 actions" in line
    assert "missed: step 1 RIGHT changed the board but not your state" in line


def test_the_mismatch_shows_where_two_states_part():
    diff = child(World())["_sc_diff"]
    assert diff((1, (2, 3)), (1, (2, 4))) == ("[1][1]", "3", "4")
    assert diff(frozenset({(1, 1), (2, 2)}), frozenset({(1, 1), (3, 3)})) == (
        " (items only one side has)", "[(2, 2)]", "[(3, 3)]")
    assert diff({"p": (1, 2), "keys": 0}, {"p": (1, 2), "keys": 1}) == ("['keys']", "0", "1")
    assert diff(("a" * 50 + "b" + "a" * 20,), ("a" * 50 + "c" + "a" * 20,)) == ("[0][50:51]", "'b'", "'c'")
    assert diff((1, 2), [1, 2]) == ("", "(1, 2)", "[1, 2]")
    report = sc.check_line({"mode": 1, "fit": None, "ledger": {}}, 3)
    assert report == "Model check: no model registered for level 3 (action lists are cut after 3 actions)."


def test_without_a_fitting_model_a_list_is_cut_and_nothing_is_refused():
    world = World()
    scope = child(world)
    result = scope["action"](["DOWN", "DOWN", "DOWN", "DOWN", "DOWN"])
    assert world.calls == [["DOWN", "DOWN", "DOWN"]] and result["stop_reason"] == "model_required"
    assert "first 3 were executed" in result["stop_detail"] and result["requested_count"] == 5
    assert scope["action"](["UP", "UP"])["executed"] is True and world.calls[-1] == ["UP", "UP"]
    events = scope["_sc_take"]()["events"]
    assert [e["kind"] for e in events] == ["cap"]


def test_outside_model_mode_nothing_changes():
    world = World()
    scope = child(world, mode=False, table=False)
    scope["action"](["DOWN"] * 5)
    assert world.calls == [["DOWN"] * 5] and "MODEL" not in world.runtime
    assert scope["_sc_take"]() == {"mode": None, "table": None, "events": [], "registered": 0, "solves": 0,
                                    "frontiers": 0, "checked_lists": 0}


def test_solve_runs_under_the_per_step_check_and_a_wrong_model_stops_at_the_step_that_refutes_it():
    world = World()
    scope = child(world)
    play(world, scope, ["DOWN", "UP", "LEFT", "RIGHT", "UP"])
    model = world.runtime["MODEL"]
    no_walls = lambda state, action: (state[0] + MOVES[action][0], state[1] + MOVES[action][1])  # noqa: E731
    assert model.register(encode, no_walls)["lists_unlimited"] is True  # no wall met yet: the wrong model fits
    plan = model.solve(lambda state: state == GOAL, actions=["UP", "DOWN", "LEFT", "RIGHT"])
    assert plan["reason"] == "found" and len(plan["path"]) == 7
    before = len(world.transitions)
    result = scope["action"](plan["path"])
    mismatch = result["model_mismatch"]
    assert result["stop_reason"] == "model_mismatch" and mismatch["predicted"] != mismatch["actual"]
    assert len(world.transitions) - before == result["executed_count"] < len(plan["path"])
    report = model.check()
    assert report["wrong"] == 1 and report["first_mismatches"][0]["step"] == world.transitions[-1].after_frame.step
    # the corrected model plans around the wall and the checked list runs to the win
    assert model.register(encode, true_step)["wrong"] == 0
    plan = model.solve(lambda state: state == GOAL, actions=["UP", "DOWN", "LEFT", "RIGHT"])
    result = scope["action"](plan["path"])
    assert result["level_completed"] is True and result["executed_count"] == len(plan["path"])
    kinds = [e["kind"] for e in scope["_sc_take"]()["events"]]
    assert kinds.count("mismatch") == 1 and kinds.count("solve") == 2 and kinds[-1] == "checked"


def test_frontier_leads_to_a_state_that_never_occurred():
    world = World()
    scope = child(world)
    play(world, scope, ["DOWN", "UP", "LEFT", "RIGHT", "UP"])
    model = world.runtime["MODEL"]
    model.register(encode, true_step)
    known = {encode(t.before_frame) for t in world.transitions} | {encode(t.after_frame) for t in world.transitions}
    path = model.frontier(actions=["UP", "DOWN", "LEFT", "RIGHT"])["path"]
    state = encode(world.frame)
    for move in path:
        state = true_step(state, move)
    assert path and state not in known and len(path) == 1


def test_a_slow_model_is_cut_at_the_time_limit_and_finishes_over_later_calls():
    import time

    world = World()
    scope = child(world)
    play(world, scope, ["DOWN", "UP"] * 6)

    def slow(state, action):
        time.sleep(0.05)
        return true_step(state, action)

    report = world.runtime["MODEL"].register(encode, slow)
    assert 0 < report["not_checked_yet"] < 12 and report["reproduced"] + report["not_checked_yet"] == 12
    for _ in range(4):
        report = world.runtime["MODEL"].check()
    assert report["not_checked_yet"] == 0 and report["reproduced"] == 12 and report["lists_unlimited"] is True


def test_a_broken_model_never_breaks_the_call():
    world = World()
    scope = child(world)
    play(world, scope, ["DOWN", "UP", "DOWN", "UP", "DOWN", "UP"])

    def boom(state, action):
        raise KeyError("nope")

    report = world.runtime["MODEL"].register(encode, boom)
    assert report["wrong"] == 6 and "KeyError" in report["first_mismatches"][0]["predicted"]
    taken = scope["_sc_take"]()
    assert taken["fit"]["wrong"] == 6 and "error" not in taken
    with pytest.raises(TypeError):
        world.runtime["MODEL"].register(encode, 3)
    acting = lambda state, action: scope["action"]([action]) and state  # noqa: E731
    count = len(world.transitions)
    report = world.runtime["MODEL"].register(encode, acting)
    assert "must not act" in report["first_mismatches"][0]["predicted"] and len(world.transitions) == count
    assert "step failed: RuntimeError" in world.runtime["MODEL"].solve(lambda s: False, actions=["UP"])["reason"]
    assert scope["action"](["UP"])["executed"] is True and len(world.transitions) == count + 1


def test_notice_and_bootstrap_install():
    text = sc.notice(2, 31234)
    assert "31,234 generated tokens have gone into level 2" in text and "cut after 3 actions" in text and "80%" in text
    assert "claim" not in text
    bootstrap = ("X\n" + sc.RUNTIME_NAMES + "}\n    request = initial\n    while True:\n        a = {"
                 + sc.RESULT_FIELD + "}\n        b = {" + sc.RESULT_FIELD + "}\n")
    installed = sc.with_simcheck(bootstrap)
    assert installed.count('"simcheck": _sc_take(),') == 2 and '"MODEL"' in installed and "_SC = {" in installed
    with pytest.raises(ValueError):
        sc.with_simcheck(installed)


def test_the_stuck_signal_switches_the_level_and_counts_the_tokens_without_a_fitting_model():
    from types import SimpleNamespace

    from arena.duck import ArenaDuckMixin as Duck

    events = []
    agent = SimpleNamespace(
        dfr={"simcheck": True, "evidence": True}, obs=SimpleNamespace(levels=1, state="NOT_FINISHED"), steps=7,
        gameplay_completion_tokens=50000, sc_stats=sc.new_stats(), sc_mark=None, sc_mode=None, sc_row=None,
        sc_notice=None, sc_marker=None, sc_seen_report=None, drain_stats={"drains": 0}, context_compactions=0,
        sandbox=SimpleNamespace(sc_state=None), workspace=SimpleNamespace(append=lambda kind, **k: events.append(kind)))
    for name in ("sc_on", "sc_active", "sc_check", "sc_turn_lines", "sc_note_reply"):
        setattr(agent, name, getattr(Duck, name).__get__(agent))
    agent.sc_check()                      # the level's first turn: the mark
    agent.sc_note_reply(29999)
    agent.gameplay_completion_tokens += 29999
    agent.sc_check()
    assert agent.sc_turn_lines() == [] and agent.sc_stats["mode_tokens"] == 0 and not events  # nothing before the signal
    agent.gameplay_completion_tokens += 1
    agent.sc_check()
    lines = agent.sc_turn_lines()
    assert events == ["simcheck_mode"] and lines[0].startswith("[Model check, by the harness] 30,000 generated")
    assert lines[1] == "Model check: no model registered for level 2 (action lists are cut after 3 actions)."
    agent.sc_note_reply(4000)
    fit = {"exact": 9, "scored": 10, "wrong": 1, "missed": 0, "pending": 0, "ok": True, "first": []}
    agent.sandbox.sc_state = {"mode": 2, "table": 2, "fit": fit, "registered": 1, "events": [{"kind": "cap"}],
                              "ledger": {"transitions": 10, "actions": {"UP": [10, 9, 1, 9]}}}
    lines = agent.sc_turn_lines()
    assert lines[0].startswith("Evidence (the harness's record of level 2, 10 actions") and "9 of 10" in lines[1]
    agent.sc_note_reply(5000)
    agent.drain_stats["drains"] = 1       # the contract left the context: the short form goes out once
    assert agent.sc_turn_lines()[0].startswith("[Model check, by the harness] Level 2 is in model mode")
    assert not agent.sc_turn_lines()[0].startswith("[Model check")
    agent.obs.levels = 2
    agent.sc_check()
    stats = agent.sc_stats
    assert stats["levels"] == [{"level": 2, "at_tokens": 30000, "at_step": 7, "mode_tokens": 9000,
                                "unfit_tokens": 4000, "fit": True, "cleared": True}]
    assert (stats["mode_tokens"], stats["unfit_tokens"], stats["caps"], stats["registered"]) == (9000, 4000, 1, 1)
    assert agent.sc_turn_lines() == [] and agent.sc_mode is None
