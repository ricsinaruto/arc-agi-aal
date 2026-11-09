"""hero-v2 ``action_plans`` (arena.action_plans, duck_transfer.sandbox.with_plans): run_plan's stop conditions and
receipts in the persistent sandbox (both request loops), the plan summaries on the host, the interrupt path, the prompt
paragraph only when on, and the per-game stats."""

import asyncio
import json

import numpy as np
import pytest
from conftest import observation, tool_reply
from test_arena_duck import arena_agent

from arena import action_plans as ap
from arena.duck import ArenaTransferDuckAgent


def bootstraps():
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import stateful_bootstrap

    original = load().run_sandboxed_python.__globals__
    public = preserve_transition_results(original["_SANDBOX_BOOTSTRAP"])
    return original, stateful_bootstrap(public), stateful_bootstrap(public, fixes=("modules",))


def make_sandbox(fixes=True):
    from duck_transfer.sandbox import StatefulSandbox, with_plans

    original, plain, fixed = bootstraps()
    events = []
    sandbox = StatefulSandbox(with_plans(plain), color_chars=original["ARC_COLOR_CHARS"],
                              on_event=lambda kind, **data: events.append((kind, data)))
    if fixes:
        sandbox.configure_fixes(with_plans(fixed), interrupt=True, grace=3)
    sandbox.events = events
    return sandbox


class Game:
    """A 4x4 board: UP paints the next cell (4 cells change in the receipt), LEFT does nothing, RIGHT completes the
    level, DOWN is GAME_OVER, NOPE is refused (not executed), SLOW paints after ``slow`` seconds."""

    def __init__(self, slow=0.0):
        self.grid = [[0] * 4 for _ in range(4)]
        self.step, self.level, self.history, self.slow = 0, 1, [], slow
        self.history.append({"action": "", "frame": self.frame()})

    def frame(self):
        return {"ascii": "", "step": self.step, "level": self.level, "shape": [4, 4],
                "grid": [row[:] for row in self.grid]}

    def state(self):
        return {"current_frame": self.frame(), "history": list(self.history), "valid_actions": ["UP", "LEFT"],
                "last_action_result": {}}

    def __call__(self, actions):
        import time

        (item,) = actions
        name = item["action"]
        if name == "NOPE":
            result = {"executed": False, "board_changed": False, "stop_reason": "no_change_repeat"}
            return {"action_result": result, "state": self.state()}
        if name == "SLOW":
            time.sleep(self.slow)
            name = "UP"
        self.step += 1
        changed = name in ("UP", "RIGHT", "DOWN")
        if name == "UP":
            cell = self.step - 1
            self.grid[cell // 4][cell % 4] = 3
        transition = {"changed_cells": 4 if changed else 0, "change_bbox": [0, 0, 1, 1] if changed else None,
                      "objects": {"appeared": 1, "disappeared": 0, "moved": 0}}
        result = {"executed": True, "action_display": name, "board_changed": changed,
                  "level_completed": name == "RIGHT", "game_over": name == "DOWN", "done": False,
                  "state": "GAME_OVER" if name == "DOWN" else "NOT_FINISHED", "level": self.level}
        if name == "RIGHT":
            self.level += 1
        self.history.append({"action": name, "frame": self.frame(), "result": {**result, "transition": transition}})
        return {"action_result": result, "state": self.state()}


def run(sandbox, code, game=None, timeout=5):
    game = game or Game()
    out = sandbox.run(code=code, timeout_seconds=timeout, initial_state=game.state(), action_handler=game)
    return out, game


@pytest.fixture(params=[True, False], ids=["fixes-loop", "plain-loop"])
def sandbox(request):
    box = make_sandbox(fixes=request.param)
    yield box
    box.close()


def test_the_bootstrap_is_unchanged_without_the_feature():
    from duck_transfer.sandbox import with_plans

    _, plain, fixed = bootstraps()
    assert "run_plan" not in plain and "run_plan" not in fixed and "_take_plans" not in fixed
    for text in (plain, fixed):
        planned = with_plans(text)
        assert planned.count("def run_plan(") == 1 and planned.count('"plans": _take_plans(),') == 2
        with pytest.raises(ValueError):
            with_plans(planned)  # installed once
    assert '"action", "result", "run_plan"}' in with_plans(fixed)  # a runtime name for the inventory


def test_a_plan_runs_to_completion_with_one_receipt_per_action(sandbox):
    out, game = run(sandbox, "import json\nr = run_plan(['UP', 'UP', {'action': 'UP'}])\nprint(json.dumps(r))")
    assert not out["error"], out
    result = json.loads(out["stdout"])
    assert result["executed"] == 3 and result["requested"] == 3 and result["stop_reason"] == "completed"
    assert result["stopped_at"] is None and [r["i"] for r in result["receipts"]] == [0, 1, 2]
    first = result["receipts"][0]
    assert first["action"] == "UP" and first["board_changed"] and first["changed_cells"] == 4
    assert first["change_bbox"] == [0, 0, 1, 1] and first["objects"]["appeared"] == 1 and first["state"] == "NOT_FINISHED"
    assert len(out["action_results"]) == 3 and game.step == 3  # three separate actions, three host receipts
    assert sandbox.plans == [{"requested": 3, "executed": 3, "attempted": 3, "stop_reason": "completed",
                              "expect": False}]
    assert [kind for kind, _ in sandbox.events].count("duck_python_plans") == 1


@pytest.mark.parametrize("plan,stop_on,reason,at,executed", [
    ("['UP', 'LEFT', 'UP']", None, "no_change", 1, 2),
    ("['UP', 'LEFT', 'UP']", "('level', 'game_over')", "completed", None, 3),
    ("['UP', 'RIGHT', 'UP']", None, "level", 1, 2),
    ("['DOWN', 'UP']", None, "game_over", 0, 1),
    ("['UP', 'NOPE', 'UP']", None, "not_executed:no_change_repeat", 1, 1),
    ("['UP', 'NOPE', 'UP']", "()", "not_executed:no_change_repeat", 1, 1),  # a refusal always stops
])
def test_stop_conditions(sandbox, plan, stop_on, reason, at, executed):
    arg = "" if stop_on is None else f", stop_on={stop_on}"
    out, game = run(sandbox, f"import json\nprint(json.dumps(run_plan({plan}{arg})))")
    assert not out["error"], out
    result = json.loads(out["stdout"])
    assert result["stop_reason"] == reason and result["stopped_at"] == at and result["executed"] == executed
    assert sandbox.plans[-1]["stop_reason"] == reason


def test_expectations_stop_at_the_first_unexpected_receipt(sandbox):
    code = """import json
print(json.dumps(run_plan(['UP', 'LEFT', 'UP'], stop_on=('unexpected',),
                          expect=lambda i, action, receipt: receipt['changed_cells'] == 4)))
print(json.dumps(run_plan(['UP', 'UP'], expect=[{'board_changed': True}, {'level_completed': True}])))
print(json.dumps(run_plan(['UP', 'UP'], expect=[None, lambda i, a, r: 1 / 0])))
print(json.dumps(run_plan(['UP'], expect=[{'board_changed': True}])))
"""
    out, _ = run(sandbox, code)
    assert not out["error"], out
    first, second, third, fourth = (json.loads(line) for line in out["stdout"].splitlines())
    assert first["stop_reason"] == "unexpected" and first["stopped_at"] == 1 and first["executed"] == 2
    assert second["stop_reason"] == "unexpected" and second["stopped_at"] == 1
    assert third["stop_reason"] == "unexpected" and "ZeroDivisionError" in third["receipts"][1]["expect_error"]
    assert fourth["stop_reason"] == "completed"
    assert [p["expect"] for p in sandbox.plans] == [True] * 4


def test_bad_arguments_raise_in_the_model_code(sandbox):
    out, _ = run(sandbox, "run_plan(['UP'], stop_on=('sometimes',))")
    assert "Unknown run_plan stop conditions" in out["error"]
    out, _ = run(sandbox, "run_plan([])")
    assert "at least one action" in out["error"]
    assert sandbox.plans == []  # nothing ran
    out, _ = run(sandbox, "print(run_plan('UP')['executed'])")  # a single action string is a one-step plan
    assert out["stdout"] == "1\n" and sandbox.plans[-1]["requested"] == 1


def test_a_timeout_inside_a_plan_still_reports_it_and_keeps_python_state():
    sandbox = make_sandbox(fixes=True)
    try:
        run(sandbox, "kept = 41")
        out, game = run(sandbox, "run_plan(['UP', 'SLOW', 'UP', 'UP'])", game=Game(slow=1.6), timeout=1)
        assert out.get("interrupted") and "interrupted" in out["error"]
        assert game.step == 2  # the slow action finished, the interrupt landed after it
        # The receipt of the interrupted call never reached the plan (the host's action_results have it).
        plan = sandbox.plans[-1]
        assert plan["attempted"] == 2 and plan["executed"] == 1 and plan["stop_reason"] == "error"
        out, _ = run(sandbox, "print(kept + 1)")
        assert out["stdout"] == "42\n"
    finally:
        sandbox.close()


def test_the_inventory_does_not_list_run_plan():
    sandbox = make_sandbox(fixes=True)
    try:
        run(sandbox, "def helper(x):\n    return x\n")
        inventory = sandbox.inventory()
        names = [f[0] for f in inventory["functions"]] + inventory["modules"] + [v[0] for v in inventory["values"]]
        assert "helper" in names and "run_plan" not in names and "_take_plans" not in names
    finally:
        sandbox.close()


# --- the agent: prompt paragraph only when on, stats --------------------------------------------------------------

async def test_the_prompt_paragraph_is_added_only_when_on(tmp_path):
    agent, *_ = arena_agent(tmp_path / "on", None, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    agent.apply_action_plans(True)
    on = agent.analyzer(asyncio.get_running_loop())._system_prompt
    control, *_ = arena_agent(tmp_path / "off", None, ArenaTransferDuckAgent)
    control.apply_harness_fixes(True)
    off = control.analyzer(asyncio.get_running_loop())._system_prompt
    assert "run_plan" not in off and on == off.rstrip() + "\n\n" + ap.PROMPT
    assert "run_plan" not in control.sandbox.bootstrap and "def run_plan(" in agent.sandbox.bootstrap
    assert control.action_plans is False and "action_plans" not in control.result()
    for a in (agent, control):
        a.executor.shutdown()
    with pytest.raises(ValueError):
        ap.settings("yes")


async def test_a_model_plan_runs_through_the_duck_step_and_is_counted(tmp_path):
    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(kwargs["json"])
            if len(self.payloads) == 1:
                return tool_reply("r = run_plan(['UP', 'UP', 'UP'])\nprint(r['stop_reason'], r['executed'])",
                                  call_id="plan", usage={"completion_tokens": 600, "prompt_tokens": 1000})
            return tool_reply("action(['UP'])", call_id="one", usage={"completion_tokens": 10, "prompt_tokens": 1000})

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    agent.apply_action_plans(True)
    agent.config.max_actions = 4
    boards = []
    for i in range(4):
        board = np.zeros((64, 64), dtype=np.uint8)
        board[2, 2 + i] = 4
        boards.append(observation(board))
    env.observations = iter(boards)
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 4
    tool = next(m for m in transport.payloads[1]["messages"] if m["role"] == "tool")
    assert "completed 3" in tool["content"]
    stats = result["action_plans"]
    assert stats["plan_calls"] == 1 and stats["plan_actions_executed"] == 3 and stats["actions_per_plan"] == 3
    assert stats["stop_reasons"] == {"completed": 1}
    assert stats["actions"] == 4 and stats["gameplay_replies"] == 2 and stats["actions_per_reply"] == 2.0
    assert stats["completion_tokens_per_action"] == round(610 / 4, 1)
    total = ap.aggregate([stats, None])
    assert total["plan_calls"] == 1 and total["games_with_plans"] == 1
