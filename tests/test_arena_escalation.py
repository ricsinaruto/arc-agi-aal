"""Executable-model escalation of the stuck notice (fixed-2x-escalate)."""

import pytest
from test_arena_notes import CAP, WINDOW, arena_agent

from arena.duck import ArenaTransferDuckAgent
from arena.escalation import ESCALATION_NOTICE, EscalatingStuckTracker


class Cfg:
    stuck_no_change_actions = 3
    stuck_turns = 4
    stuck_notice_turns = 2


def test_notice_escalates_from_the_second_notice_on_a_level():
    with pytest.raises(ValueError, match="after_notices"):
        EscalatingStuckTracker(Cfg(), after_notices=0)
    tracker = EscalatingStuckTracker(Cfg(), after_notices=2, now=0.0)
    for _ in range(3):
        tracker.note_action(False)
    assert tracker.stuck() and tracker.notice(1, ["UP"], now=10.0).startswith("Stuck check: ")
    tracker.note_turn()
    assert tracker.notice(1, ["UP"], now=20.0) is None  # the public cadence: every two turns
    tracker.note_turn()
    text = tracker.notice(1, ["UP", "DOWN"], now=30.0)
    assert text.startswith("Escalation (2 stuck checks on level 1, 2 turns, 3 actions, 30s; tried on this exact board with no effect: UP, DOWN)")
    assert "retrodicts all of them" in text and "`transition(state, act)`" in text and "`action([...])`" in text
    assert tracker.summary() == {"after_notices": 2, "notices": 2, "escalations": 1}
    tracker.reset_level(now=40.0)  # a clear resets the notice count: the next level starts with the plain notice
    for _ in range(3):
        tracker.note_action(False)
    assert tracker.notice(2, [], now=41.0).startswith("Stuck check: ")
    assert tracker.summary() == {"after_notices": 2, "notices": 3, "escalations": 1}
    assert ESCALATION_NOTICE.count("{") == ESCALATION_NOTICE.count("}") == 6


async def test_adapter_swaps_in_the_escalating_tracker_and_reports_it(tmp_path):
    class Transport:
        async def post(self, url, **kwargs):
            from test_arena_notes import tool_reply
            return tool_reply("action([{'action':'UP'}])", call_id="act")

    agent, env, game, gate = arena_agent(tmp_path, Transport(), ArenaTransferDuckAgent, levels=1)
    agent.attach(game, window=WINDOW, cap=CAP, escalation={"after_notices": 2})
    assert isinstance(agent.stuck, EscalatingStuckTracker) and agent.stuck.after_notices == 2
    assert agent.stuck.no_change_limit == agent.config.stuck_no_change_actions
    agent.config.max_actions = 1
    result = await agent.run()
    assert result["error"] is None and result["model_escalation"] == {"after_notices": 2, "notices": 0, "escalations": 0}
    plain, _, _, _ = arena_agent(tmp_path / "plain", Transport(), ArenaTransferDuckAgent, levels=1)
    plain.attach(game, window=WINDOW, cap=CAP)
    assert not isinstance(plain.stuck, EscalatingStuckTracker) and "model_escalation" not in plain.result()
