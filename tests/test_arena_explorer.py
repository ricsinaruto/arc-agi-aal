"""Model-free graph explorer (fixed-2x-explorer): search logic, report text, the arena hand-off."""

import asyncio
from types import SimpleNamespace

import numpy as np
from test_arena_duck import CAP, WINDOW, arena_agent

from arena.duck import ArenaTransferDuckAgent
from arena.explorer import (
    ExplorerService,
    LevelExplorer,
    compress,
    explore_level,
    region_text,
    report_message,
)


class CorridorEnv:
    """A cursor on row 10 starts at column 0; RIGHT/LEFT move it, UP does nothing, a click on the
    column-5 marker toggles it. Reaching column 3 clears the level; column 0 after a LEFT is a
    GAME_OVER. Only level resets exist."""

    def __init__(self, total=2):
        self.total, self.levels = total, 0
        self.calls = []
        self.resets = 0
        self._start()
        self.observation_space = self.raw()

    def _start(self):
        self.x, self.toggle, self.over = 0, False, False

    def board(self):
        board = np.zeros((64, 64), dtype=np.uint8)
        board[10, self.x] = 3
        board[20, 5] = 12 if self.toggle else 9
        board[40, 40 + self.levels] = 7
        return board

    def raw(self):
        state = "GAME_OVER" if self.over else ("WIN" if self.levels >= self.total else "NOT_FINISHED")
        return SimpleNamespace(frame=[self.board()], levels_completed=self.levels, win_levels=self.total,
                               state=SimpleNamespace(name=state), available_actions=[1, 3, 4, 6], full_reset=False)

    def step(self, action, data):
        self.calls.append((action.name, dict(data)))
        if action.name == "ACTION4":
            self.x += 1
        elif action.name == "ACTION3":
            if self.x == 0:
                self.over = True
            self.x = max(0, self.x - 1)
        elif action.name == "ACTION6" and (data.get("y"), data.get("x")) == (20, 5):
            self.toggle = not self.toggle
        if self.x >= 3:
            self.levels += 1
            self._start()
        self.observation_space = self.raw()
        return self.observation_space

    def reset(self):
        self.resets += 1
        self._start()
        self.observation_space = self.raw()
        return self.observation_space


def run(env, **kwargs):
    raw = env.observation_space
    settings = {"max_actions": 500, "max_seconds": 30, "should_stop": lambda: False, **kwargs}
    return explore_level(env, np.array(raw.frame[-1]), raw.levels_completed,
                         ["ACTION1", "ACTION3", "ACTION4", "ACTION6"], **settings)


def test_explorer_clears_the_level_and_recovers_the_shortest_winning_path():
    env = CorridorEnv()
    result = run(env)
    assert result["cleared"] and result["stop_reason"] == "cleared" and env.levels == 1
    assert [a[0] for a in result["path"]] == ["ACTION4"] * 3
    assert result["game_overs"] >= 1  # LEFT at column 0 was tried and the level reset
    kinds = {kind: (changed, taken) for kind, changed, taken in result["effects"]}
    assert kinds["ACTION1"][0] == 0 and kinds["ACTION4"][0] >= 1 and kinds["click b"][0] >= 1
    assert (result["start_frame"] != result["prewin_frame"]).any()


def test_uncleared_budget_leaves_the_level_at_its_start():
    env = CorridorEnv()
    result = run(env, max_actions=2)
    assert not result["cleared"] and result["stop_reason"] == "budget"
    assert env.x == 0 and env.levels == 0 and result["resets"] >= 1
    stopped = run(CorridorEnv(), should_stop=lambda: True)
    assert stopped["stop_reason"] == "admitted" and stopped["actions"] == 0 and stopped["raw"] is None


def test_non_deterministic_edge_triggers_a_replan():
    explorer = LevelExplorer(np.zeros((64, 64), np.uint8), 0, ["ACTION1"])
    a = ("ACTION1", None, None)
    one, two = np.zeros((64, 64), np.uint8), np.zeros((64, 64), np.uint8)
    one[0, 0], two[0, 1] = 1, 1
    explorer.observe(a, one, 0)
    explorer.state = explorer.start
    explorer.plan = [a]
    explorer.observe(a, two, 0)
    assert explorer.plan == []


def test_report_text_carries_the_demonstration_and_the_changed_region():
    env = CorridorEnv()
    result = run(env)
    names = {"ACTION4": "RIGHT", "ACTION3": "LEFT", "ACTION1": "UP"}
    message = report_message(result, 1, lambda a: names.get(a[0], f"MOUSE(row={a[2]}, col={a[1]})"))
    text = message["content"]
    assert message["role"] == "user" and "CLEARED level 1" in text and "RIGHT x3" in text
    assert "level 2" in text and "just before the winning action" in text
    missed = report_message(run(CorridorEnv(), max_actions=2), 1, lambda a: a[0])
    assert "did not clear level 1" in missed["content"] and "reset to its start" in missed["content"]
    assert "UP changed the board 0 of" in text and "click b changed the board" in text
    assert compress(["UP", "UP", "LEFT", "UP"]) == "UP x2, LEFT, UP"
    timer = {**result, "effects": [("ACTION1", 5, 5), ("click b", 2, 2)]}
    assert "every action changed the board" in report_message(timer, 1, lambda a: a[0])["content"]
    before = np.zeros((64, 64), np.uint8)
    after = before.copy()
    after[3, 4] = 9
    assert region_text(before, before) is None and "rows 3-3, cols 4-4 (1 cells differ)" in region_text(before, after)


async def test_waiting_agent_hands_its_environment_to_the_explorer_and_carries_the_report(tmp_path):
    seen = []

    class Transport:
        async def post(self, url, **kwargs):
            seen.append(kwargs["json"])
            agent.config.max_actions = agent.steps + 1  # one model action after the hand-back, then stop
            from conftest import tool_reply

            return tool_reply("action([{'action':'UP'}])", call_id="act",
                              usage={"completion_tokens": 50, "prompt_tokens": 7000})

    agent, _, game, gate = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent, levels=0)
    settings = {"max_level": 0, "max_actions": 500, "max_seconds": 30}
    agent.attach(game, window=WINDOW, cap=CAP, explorer=settings)
    env = CorridorEnv()
    agent.arc = SimpleNamespace(make=lambda *a, **kw: env)
    agent.config.max_actions = 1000
    game.state = "parked"
    game.clock.pause()
    task = asyncio.ensure_future(agent.run())
    for _ in range(200):
        if agent.explorer_waiting:
            break
        await asyncio.sleep(0.02)
    assert agent.explorer_waiting and not seen
    wall = game.clock.global_clock
    service = ExplorerService(SimpleNamespace(phase="steady"), [agent], wall, settings)
    assert service.candidate() is agent
    record = await service.explore(agent)
    assert record["cleared"] and record["path"] == ["RIGHT", "RIGHT", "RIGHT"] and game.levels == 1
    assert agent.obs.levels == 1 and agent.steps == record["actions"] and service.candidate() is None
    assert agent.duck_history[-1].frame.level == 2 and agent.explorer_message is not None
    game.state = "active"
    game.clock.resume()
    result = await asyncio.wait_for(task, 20)
    assert result["error"] is None and len(seen) == 1
    contents = [m.get("content") for m in seen[0]["messages"]]
    assert any(isinstance(c, str) and c.startswith("[Automatic explorer, level 1]") for c in contents)
    assert result["explorer"][0]["cleared"] and env.calls[-1][0] == "ACTION1"
    assert service.summary()["cleared"] == 1


async def run_at_start(tmp_path, report):
    seen = []

    class Transport:
        async def post(self, url, **kwargs):
            seen.append(kwargs["json"])
            agent.config.max_actions = agent.steps + 1
            from conftest import tool_reply

            return tool_reply("action([{'action':'UP'}])", call_id="act",
                              usage={"completion_tokens": 50, "prompt_tokens": 7000})

    agent, _, game, gate = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent, levels=0)
    agent.attach(game, window=WINDOW, cap=CAP, explorer={"at_start": True, "report": report, "max_level": -1,
                                                         "max_actions": 500, "max_seconds": 30})
    env = CorridorEnv()
    agent.arc = SimpleNamespace(make=lambda *a, **kw: env)
    agent.config.max_actions = 1000
    result = await asyncio.wait_for(agent.run(), 20)
    contents = [m.get("content") for m in seen[0]["messages"]]
    carried = any(isinstance(c, str) and c.startswith("[Automatic explorer, level 1]") for c in contents)
    return result, game, carried


async def test_explorer_at_start_clears_level_one_with_or_without_the_demonstration(tmp_path):
    result, game, carried = await run_at_start(tmp_path / "report", True)
    assert result["error"] is None and game.levels == 1 and carried and result["explorer"][0]["cleared"]
    result, game, carried = await run_at_start(tmp_path / "silent", False)
    assert result["error"] is None and game.levels == 1 and not carried


def test_winning_path_is_found_from_the_level_start_reached_by_a_reset():
    def board(value):
        grid = np.zeros((64, 64), np.uint8)
        grid[0, 0] = value
        return grid

    # The explorer starts on a mid-level board (the model already played), resets to the level start,
    # then wins in two steps from there; explorer-v1 searched from the mid-level board only.
    explorer = LevelExplorer(board(5), 0, ["ACTION1"])
    explorer.observe(("RESET", None, None), board(1), 0)
    up = ("ACTION1", None, None)
    explorer.observe(up, board(2), 0)
    assert explorer.observe(up, board(3), 1)
    assert explorer.winning_path(up) == [up, up]


async def test_explorer_actions_do_not_use_up_the_harness_action_cap(tmp_path):
    # explorer-v1 and the demo probes: the explorer's actions counted against duck_controls'
    # max_actions (1,200), so every explored game ended before the model's first request.
    seen = []

    class Transport:
        async def post(self, url, **kwargs):
            seen.append(kwargs["json"])
            agent.config.max_actions = agent.steps + 1
            from conftest import tool_reply

            return tool_reply("action([{'action':'UP'}])", call_id="act",
                              usage={"completion_tokens": 50, "prompt_tokens": 7000})

    agent, _, game, gate = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent, levels=0)
    agent.attach(game, window=WINDOW, cap=CAP, explorer={"at_start": True, "report": True, "max_level": -1,
                                                         "max_actions": 500, "max_seconds": 30})
    env = CorridorEnv()
    agent.arc = SimpleNamespace(make=lambda *a, **kw: env)
    shared = agent.config
    shared.max_actions = 2  # smaller than the explorer's own action count
    result = await asyncio.wait_for(agent.run(), 20)
    assert result["explorer"][0]["cleared"] and result["explorer"][0]["actions"] > 2
    assert len(seen) == 1 and game.levels == 1  # the model still got its turn
    assert shared.max_actions == 2 and agent.config is not shared  # the shared Config is untouched
