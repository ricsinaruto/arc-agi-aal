"""Per-level snapshot probe (fixed-2x-level-probe): the game-start probe again at every level's start."""

import json

import numpy as np
from conftest import observation, tool_reply
from test_arena_snapshot import boards, snapshot_agent

from arc_infra.perception import Observation
from arena.duck import LEVEL_PROBE_PENDING
from arena.snapshot import HEADER, LEVEL_HEADER, click_targets
from duck_transfer.feedback import TransitionCache

LEVEL_PROBE = {"probe": True, "objects": 20, "every_level": True, "clicks": 1}


def level_boards():
    board = np.zeros((64, 64), dtype=np.uint8)
    board[30:34, 30:34] = 3  # one 4x4 object of colour 3 ("G")
    shifted = board.copy()
    shifted[30:34, 30] = 0
    shifted[30:34, 34] = 3  # one column right
    clicked = shifted.copy()
    clicked[31, 31] = 8
    other = board.copy()
    other[0, 0] = 5  # RESET does not give back the observed level start
    return board, shifted, clicked, other


def text_of(message):
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


def events(agent):
    return [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]


def test_click_targets_pick_each_objects_own_cell_nearest_its_centre():
    board = np.zeros((64, 64), dtype=np.uint8)
    board[20, 20:30] = board[29, 20:30] = 3
    board[20:30, 20] = board[20:30, 29] = 3  # a 10x10 ring: its centre is background
    board[10:13, 10:13] = 7
    board[40, 5] = 12
    objects = TransitionCache(component_limit=400).objects(board)
    targets = click_targets(board, objects, 2)
    assert [cell for cell, _ in targets] == [(20, 24), (11, 11)]  # ring first (36 px), on the ring itself
    assert [node["color"] for _, node in targets] == [3, 7]
    assert [cell for cell, _ in click_targets(board, objects, 5)] == [(20, 24), (11, 11), (40, 5)]
    assert click_targets(board, objects, 0) == [] and click_targets(board, {"skipped": 900, "nodes": []}, 3) == []


async def test_probe_runs_at_game_start_and_again_at_each_new_level_until_the_game_ends(tmp_path):
    start, moved = boards()
    level2, shifted, clicked, other = level_boards()

    class Transport:
        payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(kwargs["json"])
            return tool_reply("action([{'action':'RIGHT'}])", call_id=f"act{len(self.payloads)}")

    transport = Transport()
    agent, env, game = snapshot_agent(tmp_path, transport, [
        observation(start),
        # game start: UP (no change), RIGHT (moves), one click on P (no change), RESET (restores)
        observation(start), observation(moved), observation(moved), observation(start),
        observation(level2, levels=1),  # the model's RIGHT clears level 1
        # level 2: UP (no change), RIGHT, one click on G, RESET (a different board)
        observation(level2, levels=1), observation(shifted, levels=1), observation(clicked, levels=1),
        observation(other, levels=1),
        observation(other, levels=2, state="WIN"),  # the model's RIGHT wins
    ], snapshot=LEVEL_PROBE, max_actions=40)
    result = await agent.run()
    assert result["error"] is None and result["status"] == "won"
    assert env.calls == [("ACTION1", {}), ("ACTION4", {}), ("ACTION6", {"x": 11, "y": 11}), ("ACTION4", {}),
                         ("ACTION1", {}), ("ACTION4", {}), ("ACTION6", {"x": 31, "y": 31}), ("ACTION4", {})]
    assert env.reset_count == 2 and len(transport.payloads) == 2

    first, second = transport.payloads
    carried = text_of(first["messages"][1])
    assert carried.startswith(HEADER) and "Level 1 of 5" in carried
    assert "MOUSE(row=11, col=11) on P px=9 at rows 10-12, cols 10-12: no change" in carried
    assert "RESET: start board restored." in carried and "Probe cost: 4 of this level's actions." in carried
    assert f"A {LEVEL_HEADER} like this one is taken at the start of every later level" in carried
    assert LEVEL_HEADER not in text_of(first["messages"][-1])  # the game-start block is carried only

    block = text_of(second["messages"][1])  # the carried message is now the level-2 block
    assert block.startswith(f"{LEVEL_HEADER} Deterministic probe taken at the start of level 2")
    assert "Level 2 of 5" in block and "UP: no change" in block and HEADER not in block
    assert "MOUSE(row=31, col=31) on G px=16 at rows 30-33, cols 30-33: 1 cells changed" in block
    assert "RESET: the board after RESET differs from the start board" in block
    assert "Probe cost: 4 of this level's actions." in block
    assert block in text_of(second["messages"][-1])  # and it reaches the first user turn on the level
    assert sum(text_of(m).count(LEVEL_HEADER + " Deterministic") for m in second["messages"]) == 2

    snapshots = result["level_snapshots"]
    assert [s["level"] for s in snapshots] == [1, 2] and result["game_snapshot"] == snapshots[0]
    assert [(s["probes"], s["clicks"], s["reset"], s["restored"], s["stopped"], s["actions"]) for s in snapshots] == [
        (2, 1, True, True, None, 4), (2, 1, True, False, None, 4)]
    assert agent.level_probed == {0, 1} and not agent.level_probe_due()  # nothing after the win
    assert [i["purpose"] for i in game.clock.tool_intervals].count("probe") == 2
    logged = events(agent)
    (level_event,) = [e for e in logged if e["kind"] == "level_snapshot"]
    assert level_event["step"] == 9 and level_event["level"] == 2 and level_event["actions"] == 4
    assert [e["kind"] for e in logged].count("level_snapshot_notice") == 1
    assert agent.transition_results[8]["action_data"] == {"row": 31, "col": 31}
    assert agent.transition_results[8]["probe"] is True and agent.transition_results[9]["reset"] is True


async def test_a_level_probe_that_clears_its_level_hands_over_to_the_next_levels_probe(tmp_path):
    start, _ = boards()
    level2, _, _, _ = level_boards()
    level3 = np.zeros((64, 64), dtype=np.uint8)
    level3[5:8, 50:60] = 9
    seen = []

    class Transport:
        payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(kwargs["json"])
            seen.append((agent.steps, agent.level_start_step))
            return tool_reply("action([{'action':'MOUSE','row':1,'col':1}])", call_id=f"act{len(self.payloads)}")

    transport = Transport()
    agent, env, game = snapshot_agent(tmp_path, transport, [
        observation(start), observation(start), observation(start),  # game start: nothing changes, no RESET
        observation(level2, levels=1),  # the model's click clears level 1
        observation(level3, levels=2),  # level 2: its UP probe clears the level
        observation(level3, levels=2), observation(level3, levels=2),  # level 3: nothing changes
        observation(level3, levels=3, state="WIN"),
    ], snapshot={**LEVEL_PROBE, "clicks": 0}, max_actions=40)
    result = await agent.run()
    assert result["error"] is None and result["status"] == "won" and env.reset_count == 0
    assert [name for name, _ in env.calls] == ["ACTION1", "ACTION4", "ACTION6", "ACTION1", "ACTION1", "ACTION4",
                                               "ACTION6"]
    snapshots = result["level_snapshots"]
    assert [(s["level"], s["probes"], s["stopped"], s["reset"], s["actions"]) for s in snapshots] == [
        (1, 2, None, False, 2), (2, 1, "level_cleared", False, 1), (3, 2, None, False, 2)]
    assert seen == [(2, 2), (6, 6)]  # a RESET right after a probe is refused as a no-op
    notice = text_of(transport.payloads[1]["messages"][-1])
    assert "start of level 2" in notice and "-> the level was cleared by this probe" in notice
    assert "start of level 3" in notice  # both blocks, in order, in the first turn after the clear
    assert text_of(transport.payloads[1]["messages"][1]).startswith(
        f"{LEVEL_HEADER} Deterministic probe taken at the start of level 3")
    assert game.levels == 3 and agent.level_probed == {0, 1, 2}


async def test_actions_wait_for_a_pending_level_probe_and_no_probe_after_the_end(tmp_path):
    start, _ = boards()
    level2, _, _, _ = level_boards()

    class Transport:
        async def post(self, url, **kwargs):
            raise AssertionError("no model request in this test")

    agent, env, game = snapshot_agent(tmp_path, Transport(), [observation(start), observation(level2)],
                                      snapshot=LEVEL_PROBE, max_actions=40)
    agent.env = env
    agent.level_probed = {0}  # the game-start probe ran; level 2 (obs.levels 1) has not been probed
    agent.obs = Observation([level2], 1, 5, "NOT_FINISHED", ["ACTION1", "ACTION4", "ACTION6"])
    assert agent.level_probe_due()
    result = await agent.step_duck({"action": "RIGHT"})
    assert result == {"executed": False, "stop_reason": "level_probe_pending", "error": LEVEL_PROBE_PENDING}
    assert env.calls == [] and agent.steps == 0
    assert any(e.get("reason") == "level_probe_pending" for e in events(agent))
    agent.obs = Observation([level2], 1, 5, "GAME_OVER", ["ACTION1"])
    assert not agent.level_probe_due()
    agent.obs = Observation([level2], 5, 5, "WIN", ["ACTION1"])
    assert not agent.level_probe_due()
    agent.obs = Observation([level2], 1, 5, "NOT_FINISHED", ["ACTION1"])
    agent.stopping.set()  # the wall is over
    assert not agent.level_probe_due()
    await agent.before_turn()
    assert env.calls == [] and agent.level_snapshots == []
    # The fixed-2x-snapshot control (every_level off) never probes after the start and never defers.
    control, _, _ = snapshot_agent(tmp_path / "control", Transport(), [observation(start)])
    control.obs = Observation([level2], 1, 5, "NOT_FINISHED", ["ACTION1"])
    assert control.snapshot["every_level"] is False and not control.level_probe_due()
    assert "level_snapshots" not in control.result()
