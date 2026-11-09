"""Game-start snapshot (fixed-2x-snapshot): the deterministic probe before the first turn."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeEnv, observation, tool_reply
from test_arena_duck import CAP, WINDOW, arena_agent

from arena.duck import ArenaTransferDuckAgent
from arena.snapshot import HEADER, probe_line, settings


def boards():
    start = np.zeros((64, 64), dtype=np.uint8)
    start[10:13, 10:13] = 7  # one 3x3 object of colour 7 (letter "P")
    start[40, 5] = 12
    moved = start.copy()
    moved[10:13, 10:13] = 0
    moved[9:12, 10:13] = 7  # the object one row up
    return start, moved


def test_settings_and_probe_lines():
    off = {"every_level": False, "clicks": 0}  # fixed-2x-snapshot: game start only, no clicks
    assert settings({}) == {"probe": True, "objects": 20, **off}
    assert settings({"probe": False, "objects": 5}) == {"probe": False, "objects": 5, **off}
    assert settings({"every_level": True, "clicks": 3}) == {"probe": True, "objects": 20, "every_level": True,
                                                            "clicks": 3}
    with pytest.raises(ValueError, match="subset"):
        settings({"levels": 3})
    with pytest.raises(ValueError, match="objects"):
        settings({"objects": -1})
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="clicks"):
            settings({"clicks": bad})
    assert probe_line("UP", {"changed_cells": 0, "change_bbox": None}) == "UP: no change"
    line = probe_line("DOWN", {
        "changed_cells": 6, "change_bbox": [9, 10, 13, 13],
        "objects": {"moved": [{"color": 7, "pixels": 9, "bbox": [9, 10, 12, 13], "from": [10, 10, 13, 13]}],
                    "appeared": [], "disappeared": [], "counts": {"moved": 2, "appeared": 0, "disappeared": 0},
                    "background_reshaped": False},
    })
    assert line == ("DOWN: 6 cells changed within rows 9-12, cols 10-12; moved P px=9 at rows 9-11, cols 10-12 "
                    "by (-1 rows, +0 cols); 1 more object changes")


def snapshot_agent(tmp_path, transport, observations, *, snapshot={"probe": True, "objects": 20}, max_actions=4):
    agent, _, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, levels=1)
    env = FakeEnv(observations)
    agent.arc = SimpleNamespace(make=lambda *a, **kw: env)
    agent.attach(game, window=WINDOW, cap=CAP, snapshot=snapshot)
    agent.config.max_actions = max_actions
    return agent, env, game


async def test_probe_precedes_the_first_turn_and_is_carried_as_the_first_user_message(tmp_path):
    start, moved = boards()
    tried = []

    class Transport:
        payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(kwargs["json"])
            tried.append(list(agent.tried_no_effect_here()))
            return tool_reply("action([{'action':'RIGHT'}])", call_id="act")  # UP would be refused as a no-op repeat

    transport = Transport()
    # start -> UP changes nothing -> RIGHT moves the object -> RESET restores -> the model's RIGHT clears the level
    agent, env, game = snapshot_agent(tmp_path, transport, [
        observation(start), observation(start), observation(moved), observation(start), observation(moved, levels=1),
    ])
    result = await agent.run()
    assert result["error"] is None
    assert [name for name, _ in env.calls] == ["ACTION1", "ACTION4", "ACTION4"] and env.reset_count == 1
    assert result["actions"] == 4 and game.levels == 1
    text = transport.payloads[0]["messages"][1]["content"]
    assert transport.payloads[0]["messages"][1]["role"] == "user" and text.startswith(HEADER)
    assert "Level 1 of 5; board 64x64." in text
    assert "Colours on the start board (letters as in the board views): W x4086 (100%), P x9 (0%), O x1 (0%)." in text
    assert "objects: 3 (by colour: 0x1 7x1 12x1)" in text and "c=7 px=9 bbox=[10, 10, 13, 13]" in text
    assert "RIGHT: 6 cells changed within rows 9-12, cols 10-12; moved P px=9 at rows 9-11, cols 10-12 by (-1 rows, +0 cols)" in text
    assert "UP: no change" in text and "RESET: start board restored." in text
    assert "Probe cost: 3 of this level's actions." in text and text.rstrip().endswith("p=purple")
    assert tried == [["UP"]]  # the start-board no-op survives the RESET (RIGHT was probed from the moved board)
    stats = result["game_snapshot"]
    assert {k: stats[k] for k in ("probes", "changed", "reset", "restored", "stopped")} == {
        "probes": 2, "changed": 1, "reset": True, "restored": True, "stopped": None}
    assert stats["tokens"] > 100 and stats["chars"] == len(text)
    assert game.clock.tool_seconds() > 0 and game.clock.tool_intervals[0]["purpose"] == "probe"
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    snapshot = next(e for e in events if e["kind"] == "game_snapshot")
    assert snapshot["step"] == 3 and snapshot["probes"] == 2
    assert [entry.action for entry in agent.duck_history] == ["", "UP", "RIGHT", "RESET", "RIGHT"]
    assert agent.transition_results[1]["probe"] is True and agent.transition_results[1]["board_changed"] is False
    assert agent.transition_results[2]["board_changed"] is True and agent.transition_results[3]["reset"] is True
    # The analyzer's own history never holds the block: the second request carries it again after the system prompt.
    assert all(p["messages"][1]["content"] == text for p in transport.payloads)


async def test_a_probe_that_clears_the_level_stops_the_sequence_without_reset(tmp_path):
    start, moved = boards()

    class Transport:
        payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(kwargs["json"])
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    transport = Transport()
    agent, env, game = snapshot_agent(tmp_path, transport, [
        observation(start), observation(moved, levels=1), observation(moved, levels=1, state="WIN"),
    ])
    result = await agent.run()
    assert result["error"] is None and result["status"] == "won"
    assert [name for name, _ in env.calls] == ["ACTION1", "ACTION1"] and env.reset_count == 0
    text = transport.payloads[0]["messages"][1]["content"]
    assert "-> the level was cleared by this probe" in text and "RESET" not in text.split("Probe cost")[0].split("sequence")[1]
    assert result["game_snapshot"]["stopped"] == "level_cleared" and result["game_snapshot"]["reset"] is False
    assert game.levels == 1 and game.time_to_level[0]["level"] == 1


async def test_probe_off_keeps_the_static_snapshot_and_the_control_has_none(tmp_path):
    start, moved = boards()

    class Transport:
        payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(kwargs["json"])
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    transport = Transport()
    agent, env, game = snapshot_agent(tmp_path, transport, [observation(start), observation(moved, levels=1)],
                                      snapshot={"probe": False}, max_actions=1)
    result = await agent.run()
    assert result["error"] is None and env.calls == [("ACTION1", {})] and env.reset_count == 0
    text = transport.payloads[0]["messages"][1]["content"]
    assert text.startswith(HEADER) and "One-step effect" not in text and "Probe cost" not in text
    assert result["game_snapshot"]["probes"] == 0
    plain, _, _ = snapshot_agent(tmp_path / "plain", Transport(), [observation(start), observation(moved, levels=1)],
                                 snapshot=None, max_actions=1)
    assert plain.snapshot is None and "game_snapshot" not in plain.result()
