"""Cross-game notes (fixed-2x-notes): notebook, de-duplication, injection cap, the level-clear
and park triggers, and the unchanged fixed-2x control payload."""

import asyncio
import copy
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ROOT, observation, tool_reply
from test_arena_core import Now, make_games
from test_arena_duck import CAP, WINDOW, arena_agent, text_reply
from test_arena_fixed_2x import CONTROL

from arena.duck import ArenaTransferDuckAgent
from arena.gate import KVGate
from arena.notes import NOTES_LABEL, NOTES_PROMPT, CrossGameNotes, parse_notes
from arena.scheduler import Scheduler
from arena.variants import apply_variant


def test_parse_dedup_and_file_round_trip(tmp_path):
    reply = ("Here are the notes:\n- HUD counters often sit in a border row.\n"
             "* hud counters often sit in a border row!\n2) Print only bounding boxes, never whole boards.\n"
             "```\nNONE\n\n- " + "x" * 400)
    lines = parse_notes(reply, 10)
    assert lines[:3] == ["HUD counters often sit in a border row.",
                         "hud counters often sit in a border row!",
                         "Print only bounding boxes, never whole boards."]
    assert len(lines) == 4 and len(lines[3]) == 300
    assert parse_notes(reply, 2) == lines[:2] and parse_notes("NONE", 10) == []
    path = tmp_path / "cross-game-notes.md"
    notes = CrossGameNotes(path, lines_per_request=10, request_cap_tokens=1024, inject_cap_tokens=2000)
    assert notes.add(lines, source="a@1") == 3 and notes.duplicates == 1  # rough: same letters and digits
    assert notes.add(["HUD counters often sit in a border row"], source="b@2") == 0 and notes.duplicates == 2
    text = path.read_text()
    assert text.startswith("# Cross-game notes") and text.count("\n- ") == 3
    # A later run in the same folder carries the notebook.
    again = CrossGameNotes.from_settings(path, {"lines_per_request": 5, "unknown": 1})
    assert again.lines == notes.lines and again.carried == 3 and again.lines_per_request == 5
    assert again.summary()["lines"] == 3 and again.summary()["inject_cap_tokens"] == 2000


def test_injection_keeps_the_newest_lines_within_the_token_cap(tmp_path):
    notes = CrossGameNotes(tmp_path / "n.md", inject_cap_tokens=200)
    assert notes.render() == (None, 0, 0)
    notes.add([f"note number {i} " + " ".join(["abc"] * 20) for i in range(40)], source="g@0")
    message, kept, tokens = notes.render(3.0)
    assert message["role"] == "user" and message["content"].startswith(NOTES_LABEL)
    assert tokens <= 200 and 0 < kept < 40
    assert message["content"].endswith("note number 39 " + " ".join(["abc"] * 20))
    assert "note number 0 " not in message["content"] and message["content"].count("\n- ") == kept
    assert notes.render(1.0)[0] is None  # a single line over the cap is not injected


def test_scheduler_park_runs_the_game_callback():
    now = Now()
    wall, games = make_games(now, count=2)
    sched = Scheduler(games, wall)
    wall.release()
    for game in games:
        game.clock.release()
    sched.start_all()
    seen = []
    games[0].on_park = seen.append
    sched.park(games[0], "level_budget")
    sched.park(games[1], "opening_switch")  # no callback: nothing to run
    assert seen == ["level_budget"] and games[0].state == games[1].state == "parked"


def test_notes_variant_is_one_change_on_the_fixed_2x_base():
    study = apply_variant(CONTROL, "fixed-2x-notes")
    assert study["cross_game_notes"] == {"lines_per_request": 10, "request_cap_tokens": 1024,
                                         "inject_cap_tokens": 2000}
    assert study["arena_variant"]["reference_difference"] == {
        "cross_game_notes": {"before": None, "after": study["cross_game_notes"]}}
    assert "memory_block" not in study and study["budget_clock"] == "busy"


async def test_control_payload_is_unchanged_and_notes_follow_the_system_prompt(tmp_path):
    payloads = {}

    def transport_for(name):
        class Transport:
            async def post(self, url, **kwargs):
                payloads.setdefault(name, []).append(copy.deepcopy(kwargs["json"]))
                return tool_reply("action([{'action':'UP'}])", call_id="act")

        return Transport()

    control, env, game, gate = arena_agent(tmp_path / "control", transport_for("control"), ArenaTransferDuckAgent)
    assert control.notes is None and control.memory is None and control.carried_reserve() == 0
    messages = [{"role": "system", "content": "s"}]
    assert await control.carried_messages(messages) is messages and control.inject(messages) is messages
    result = await control.run()
    assert "cross_game_notes" not in result and "memory_block" not in result
    # The variant with an empty notebook sends the same bytes as the control.
    empty = CrossGameNotes(tmp_path / "empty.md")
    agent, env, game, gate = arena_agent(tmp_path / "empty", transport_for("empty"), ArenaTransferDuckAgent)
    agent.attach(game, window=WINDOW, cap=CAP, notes=empty)
    game.admissions.append({"event": "admit"})
    result = await agent.run()
    assert json.dumps(payloads["empty"]) == json.dumps(payloads["control"])
    assert result["cross_game_notes"]["injections"] == 0 and not (tmp_path / "empty.md").exists()
    assert agent.carried_reserve() == 2128 and agent._analyzer._context_budget_tokens == control._analyzer._context_budget_tokens - 2128
    # With notes in the notebook and an admission recorded, the labelled notes message is the
    # first user message after the system prompt and everything else is byte-identical.
    filled = CrossGameNotes(tmp_path / "filled.md")
    filled.add(["Print bounding boxes, not boards."], source="x@1")
    agent, env, game, gate = arena_agent(tmp_path / "filled", transport_for("filled"), ArenaTransferDuckAgent)
    agent.attach(game, window=WINDOW, cap=CAP, notes=filled)
    game.admissions.append({"event": "admit"})
    result = await agent.run()
    sent, ctrl = payloads["filled"][0], payloads["control"][0]
    assert [m["role"] for m in sent["messages"]][:3] == ["system", "user", "user"]
    assert sent["messages"][0] == ctrl["messages"][0] and sent["messages"][2:] == ctrl["messages"][1:]
    assert sent["messages"][1] == {"role": "user", "content": NOTES_LABEL + "- Print bounding boxes, not boards."}
    assert {k: v for k, v in sent.items() if k != "messages"} == {k: v for k, v in ctrl.items() if k != "messages"}
    stats = result["cross_game_notes"]
    assert stats["injections"] == 1 and stats["injected_lines"] == 1 and stats["injected_tokens"] > 0
    assert stats["requests"] == 0 and stats["notebook_lines"] == 1


async def test_level_clear_asks_for_notes_and_the_next_admission_injects_them(tmp_path):
    notes = CrossGameNotes(tmp_path / "cross-game-notes.md")

    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return tool_reply("action([{'action':'UP'}])", call_id="act")
            if len(self.payloads) == 2:
                assert "tools" not in payload and payload["chat_template_kwargs"] == {"enable_thinking": False}
                assert payload["max_tokens"] == 1024
                prompt = payload["messages"][-1]["content"]
                assert prompt.startswith(NOTES_PROMPT[:40]) and prompt.endswith("(empty)")
                assert [m["role"] for m in payload["messages"]][:2] == ["system", "user"]
                game.admissions.append({"event": "admit"})  # re-admitted before the next request
                return text_reply("- Levels often reuse the previous layout with one new rule.\n"
                                  "- Print bounding boxes, not boards.\nNONE",
                                  usage={"completion_tokens": 40, "prompt_tokens": 3000})
            return tool_reply("action([{'action':'UP'}])", call_id="act2")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, levels=1)
    agent.attach(game, window=WINDOW, cap=CAP, notes=notes)
    agent.config.max_actions = 2
    after = np.zeros((64, 64), dtype=np.uint8)
    after[5, 5] = 3
    env.observations = itertools.chain(env.observations, [observation(after, levels=1)])
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2 and len(transport.payloads) == 3
    assert notes.lines == ["Levels often reuse the previous layout with one new rule.",
                           "Print bounding boxes, not boards."]
    assert notes.sources == ["synthetic@1"] * 2
    second, third = transport.payloads[1]["messages"], transport.payloads[2]["messages"]
    assert [m["role"] for m in second][:2] == ["system", "user"]  # nothing injected before an admission
    assert [m["role"] for m in third][:3] == ["system", "user", "user"]
    assert third[1]["content"] == NOTES_LABEL + "- " + notes.lines[0] + "\n- " + notes.lines[1]
    stats = result["cross_game_notes"]
    assert stats["requests"] == 1 and stats["failures"] == 0 and stats["seconds"] > 0
    assert stats["lines_offered"] == 2 and stats["lines_added"] == 2 and stats["notebook_lines"] == 2
    assert stats["injections"] == 1 and stats["injected_lines"] == 2 and stats["injected_tokens"] > 0
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert [e["purpose"] for e in events if e["kind"] == "model"] == ["gameplay", "notes", "gameplay"]
    assert [i["purpose"] for i in game.clock.inference_intervals] == ["gameplay", "notes", "gameplay"]
    assert game.clock.inference_seconds() > 0 and result["model_calls"] == 3
    kinds = [(e["kind"], e.get("reason")) for e in events if e["kind"].startswith("notes")]
    assert kinds == [("notes_request", "level_clear"), ("notes_injected", None)]
    assert (tmp_path / "cross-game-notes.md").read_text().count("\n- ") == 2


async def test_park_asks_for_notes_while_parked_and_the_resumed_request_carries_them(tmp_path):
    notes = CrossGameNotes(tmp_path / "cross-game-notes.md")
    seen = []

    def resume():
        game.admissions.append({"event": "admit"})
        game.clock.resume()

    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            self.payloads.append(payload)
            seen.append(game.clock.parked)
            if len(self.payloads) == 1:
                # The scheduler parks the game while its reply is in flight.
                game.clock.pause()
                agent.note_park("level_budget")
                return tool_reply("action([{'action':'UP'}])", call_id="act")
            if len(self.payloads) == 2:
                assert "tools" not in payload and payload["max_tokens"] == 1024
                asyncio.get_running_loop().call_later(0.3, resume)
                return text_reply("- Re-ground after every level clear.")
            return tool_reply("action([{'action':'UP'}])", call_id="act2")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, levels=0)
    agent.attach(game, window=WINDOW, cap=CAP, notes=notes)
    agent.config.max_actions = 2
    after = np.zeros((64, 64), dtype=np.uint8)
    after[5, 5] = 3
    env.observations = itertools.chain(env.observations, [observation(after, levels=0)])
    result = await asyncio.wait_for(agent.run(), 10)
    assert result["error"] is None and result["actions"] == 2 and len(transport.payloads) == 3
    assert seen == [False, True, False]  # the notes request went out while parked
    assert gate.snapshot()["reserved"] == 0 and gate.snapshot()["admitted"] == 3
    assert agent.client.slots.allow_parked is False
    third = transport.payloads[2]["messages"]
    assert third[1] == {"role": "user", "content": NOTES_LABEL + "- Re-ground after every level clear."}
    stats = result["cross_game_notes"]
    assert stats["requests"] == 1 and stats["injections"] == 1 and stats["lines_added"] == 1
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    request = next(e for e in events if e["kind"] == "notes_request")
    assert request["reason"] == "park:level_budget" and request["added"] == 1
    injected = next(e for e in events if e["kind"] == "notes_injected")
    assert injected["parked"] is True and injected["lines"] == 1
    attempts = game.admission_attempts
    assert [a["status"] for a in attempts] == ["closed"] * 3
    assert attempts[1]["parked_wait_seconds"] == 0 and attempts[2]["parked_wait_seconds"] > 0
    assert [i["purpose"] for i in game.clock.inference_intervals] == ["gameplay", "notes", "gameplay"]


async def test_failed_notes_request_never_ends_the_game(tmp_path):
    notes = CrossGameNotes(tmp_path / "cross-game-notes.md")

    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            self.payloads.append(copy.deepcopy(kwargs["json"]))
            if len(self.payloads) == 2:
                import httpx
                raise httpx.ConnectError("serving hiccup")
            return tool_reply("action([{'action':'UP'}])", call_id=f"act{len(self.payloads)}")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, levels=1)
    agent.attach(game, window=WINDOW, cap=CAP, notes=notes)
    agent.config.max_actions = 2
    after = np.zeros((64, 64), dtype=np.uint8)
    after[5, 5] = 3
    env.observations = itertools.chain(env.observations, [observation(after, levels=1)])
    result = await agent.run()
    assert result["error"] is None and result["actions"] == 2 and len(transport.payloads) == 3
    stats = result["cross_game_notes"]
    assert stats["requests"] == 0 and stats["failures"] == 1 and notes.lines == []
    assert agent.notes_due is None and gate.snapshot()["reserved"] == 0
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    assert next(e for e in events if e["kind"] == "notes_request_failed")["error"].startswith("ConnectError")


async def test_trial_shares_one_notebook_between_its_duck_agents(tmp_path):
    from arena.trial import Trial

    study = apply_variant(CONTROL, "fixed-2x-notes")
    trial = Trial(study, study["trials"][0], source=ROOT, games_dir=None, output=tmp_path / "out",
                  server_dir=tmp_path, base_url="http://model/v1")
    now = Now()
    wall, games = make_games(now, count=2)
    agents, rows, close = await trial.duck_agents(games, KVGate(1000, slots=2), 131072, wall, "card", SimpleNamespace())
    try:
        assert trial.notes.path == tmp_path / "out" / "cross-game-notes.md"
        assert all(agent.notes is trial.notes and agent.memory is None for agent in agents)
        assert [game.on_park for game in games] == [agent.note_park for agent in agents]
        assert trial.notes.summary()["lines_per_request"] == 10 and trial.notes.inject_cap_tokens == 2000
        assert trial.notes.request_cap_tokens == 1024
    finally:
        for agent in agents:
            agent.executor.shutdown(wait=False)
        await close()
        trial.admission_log.close()
    assert pytest.approx(agents[0].carried_reserve()) == 2128


async def test_live_refresh_injects_at_a_level_clear_and_at_a_compaction(tmp_path):
    from test_arena_pool import CONTROL

    from arena.variants import apply_variant
    live = apply_variant(CONTROL, "fixed-2x-notes-live")
    assert live["cross_game_notes"] == {"lines_per_request": 10, "request_cap_tokens": 1024,
                                        "inject_cap_tokens": 2000, "refresh": "live"}
    assert apply_variant(CONTROL, "fixed-2x-notes")["cross_game_notes"].get("refresh") is None
    with pytest.raises(ValueError, match="refresh"):
        CrossGameNotes(tmp_path / "x.md", refresh="sometimes")
    notes = CrossGameNotes.from_settings(tmp_path / "cross-game-notes.md", live["cross_game_notes"])
    assert notes.refresh == "live" and notes.summary()["refresh"] == "live"
    notes.add(["A HUD strip along one border is a step counter."], source="other@1")

    class Transport:
        def __init__(self):
            self.payloads = []

        async def post(self, url, **kwargs):
            payload = copy.deepcopy(kwargs["json"])
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return tool_reply("action([{'action':'UP'}])", call_id="act")
            if len(self.payloads) == 2:  # the notes request at the clear
                return text_reply("- Clicking an inert cell reports board_changed False.",
                                  usage={"completion_tokens": 20, "prompt_tokens": 3000})
            return tool_reply("action([{'action':'UP'}])", call_id="act2")

    transport = Transport()
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent, levels=1)
    agent.attach(game, window=WINDOW, cap=CAP, notes=notes)
    agent.config.max_actions = 2
    after = np.zeros((64, 64), dtype=np.uint8)
    after[5, 5] = 3
    env.observations = itertools.chain(env.observations, [observation(after, levels=1)])
    result = await agent.run()
    assert result["error"] is None and len(transport.payloads) == 3
    first, third = transport.payloads[0]["messages"], transport.payloads[2]["messages"]
    # No admission event in this fixture, so nothing is carried before the clear; the clear
    # refreshes the snapshot in place (no re-admission) with both the other game's line and
    # the line the clear itself added.
    assert [m["role"] for m in first] == ["system", "user"]
    assert [m["role"] for m in third][:3] == ["system", "user", "user"]
    assert third[1]["content"] == (NOTES_LABEL + "- A HUD strip along one border is a step counter."
                                   "\n- Clicking an inert cell reports board_changed False.")
    events = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()]
    injected = [e for e in events if e["kind"] == "notes_injected"]
    assert [e["trigger"] for e in injected] == ["level_clear"]
    assert result["cross_game_notes"]["injections"] == 1
    # A compaction refreshes too, but only when the notebook changed since the last snapshot.
    agent.context_compactions += 1
    agent.refresh_notes_snapshot()
    assert agent.notes_stats["injections"] == 1
    notes.add(["Level transitions swap the whole layout."], source="other@2")
    agent.context_compactions += 1
    agent.refresh_notes_snapshot()
    assert agent.notes_stats["injections"] == 2
    assert agent.notes_snapshot["content"].endswith("- Level transitions swap the whole layout.")
    last = [json.loads(line) for line in agent.workspace.log.read_text().splitlines()][-1]
    assert last["kind"] == "notes_injected" and last["trigger"] == "compaction"
