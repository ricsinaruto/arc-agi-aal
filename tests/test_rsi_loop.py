"""rsi arm (arena.rsi_loop): the record and the test queue on a toy game, the second reader's form, the host hooks."""
import asyncio
from types import SimpleNamespace as NS

import pytest

from arena import rsi_loop as rsi

SIZE = 24                     # the interior the record compares is rows/cols 4..19
WALL = {(r, 12) for r in range(4, 12)}
KEYS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
CHARS = "0123456789abcdef"
VALID = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]


class Game:
    """A player (9) that walks, a wall (5), a 2x2 button (3) that a click toggles to 4, a 2x2 lamp (7) that ignores
    clicks; SPACE does nothing; ``limit`` actions per attempt; a bar in row 5 that runs down one cell per action when
    ``bar``. Reaching column 19 clears the level."""

    def __init__(self, limit=None, bar=False):
        self.history, self.results = [], {}
        self.level, self.step, self.limit, self.bar = 1, 0, limit, bar
        self.start()
        self.history.append(NS(action="", frame=self.frame()))

    def start(self):
        self.player, self.button, self.used = (15, 6), 3, 0

    def frame(self):
        grid = [[0] * SIZE for _ in range(SIZE)]
        for r, c in WALL:
            grid[r][c] = 5
        for r in (17, 18):
            for c in (6, 7):
                grid[r][c] = self.button
            for c in (15, 16):
                grid[r][c] = 7
        if self.bar:
            for c in range(4, max(4, 19 - self.used)):
                grid[5][c] = 2
        grid[self.player[0]][self.player[1]] = 9
        return NS(grid=tuple(tuple(row) for row in grid), step=self.step, level=self.level)

    def act(self, action):
        self.step += 1
        over = False
        if action == "RESET":
            self.start()
        else:
            self.used += 1
            if action in KEYS:
                r, c = self.player[0] + KEYS[action][0], self.player[1] + KEYS[action][1]
                if (r, c) not in WALL and 4 <= r < 17 and 4 <= c < 20:
                    self.player = (r, c)
                if c == 19:
                    self.level += 1
                    self.start()
            elif action.startswith("MOUSE"):
                row, col = (int(v.split("=")[1]) for v in action[6:-1].split(","))
                if row in (17, 18) and col in (6, 7):
                    self.button = 7 - self.button
            over = self.limit is not None and self.used >= self.limit
        self.results[self.step] = {"game_over": over}
        self.history.append(NS(action=action, frame=self.frame()))
        return over

    def play(self, *actions):
        for action in actions:
            if self.act(action):
                self.act("RESET")


def record(game):
    rec = rsi.Record(CHARS)
    rec.update(game.history, game.results)
    return rec


def click(row, col):
    return f"MOUSE(row={row}, col={col})"


def test_settings_are_opt_in_and_checked():
    assert rsi.settings(None) is None and rsi.settings({}) is None and rsi.settings({"record": False}) is None
    assert rsi.settings(True) == {**rsi.DEFAULTS, "record": True, "tests": True}
    with pytest.raises(ValueError):
        rsi.settings({"verify": True})                 # the reader feeds the queue
    with pytest.raises(ValueError):
        rsi.settings({"records": True})


def test_gain_falls_with_evidence_and_the_floor_drops_known_units():
    assert rsi.gain(0, 0) == pytest.approx(0.1931, abs=1e-3)
    assert rsi.gain(0, 1) == pytest.approx(0.1365, abs=1e-3) and rsi.gain(1, 0) == pytest.approx(rsi.gain(0, 1))
    values = [rsi.gain(0, n) for n in range(8)]
    assert values == sorted(values, reverse=True)
    assert rsi.gain(0, 3) > rsi.MIN_GAIN > rsi.gain(0, 4)
    assert rsi.gain(0.5, 0.5) < rsi.gain(0, 0)            # a half-weight prior already tells something


def test_counts_are_per_distinct_board_and_untried_actions_lead_the_queue():
    game = Game()
    game.play("UP", "UP", "LEFT", "LEFT", "LEFT")         # the last LEFT runs into the edge (col 4 is the limit)
    rec = record(game)
    table = rec.stats()
    assert len(table[("K", "UP")][0]) == 2 and len(table[("K", "LEFT")][0]) == 2 and len(table[("K", "LEFT")][1]) == 1
    queue = rec.refresh(VALID)
    assert [t["kind"] for t in queue[:3]] == ["untried"] * 3
    assert {t["unit"][1] for t in queue[:3]} == {"DOWN", "RIGHT", "SPACE"}
    assert sum(t["kind"] == "unclicked" for t in queue) == 1
    assert "LEFT 2/3" in rec.action_line() and "`LEFT` at step 5" in rec.few_line()
    game.play("SPACE")
    rec.update(game.history, game.results)
    assert rec.results[-1]["text"] == "first `SPACE`: did nothing" and rec.results[-1]["carry"]
    assert ("untried", ("K", "SPACE")) in rec.closes and all(t["unit"] != ("K", "SPACE") for t in rec.refresh(VALID))


def test_a_single_no_op_is_listed_for_a_retry_and_the_retry_closes_it():
    game = Game()
    game.play("SPACE", "UP", "UP", "LEFT", "RIGHT", "DOWN")
    rec = record(game)
    queue = rec.refresh(VALID)
    retry = next(t for t in queue if t["kind"] == "retry")
    assert retry["unit"] == ("K", "SPACE") and "the only board it was tried on" in retry["text"] and retry["cost"] == 1
    game.play("SPACE")
    rec.update(game.history, game.results)
    assert rec.results[-1]["text"] == "`SPACE` retried after its no-op at step 1: did nothing"
    assert not any(t["kind"] == "retry" for t in rec.refresh(VALID))  # two no-op boards: no longer a single observation
    # a working action that then fails once and is dropped
    game = Game()
    game.play("LEFT", "LEFT", "LEFT", "UP", "UP", "UP", "DOWN", "DOWN")
    rec = record(game)
    assert not any(t["kind"] == "retry" for t in rec.refresh(VALID))   # five actions without it are routine play
    game.play(*["UP", "DOWN"] * 8)
    rec.update(game.history, game.results)
    retry = next(t for t in rec.refresh(VALID) if t["kind"] == "retry")
    assert "changed 2 board(s), then did nothing at step 3 and has not been tried since (21 actions ago)" in retry["text"]
    game.play("RIGHT", "RIGHT", "RIGHT", "LEFT")
    rec.update(game.history, game.results)
    assert rec.results[-1]["text"] == "`LEFT` retried after its no-op at step 3: changed the board"
    assert "`LEFT` at step 3" in rec.few_line()


def test_click_units_are_colours_and_dead_colours_discount_the_rest():
    game = Game()
    rec = record(game)
    first = [t for t in rec.refresh(VALID) if t["kind"] == "unclicked"]
    assert len(first) == 1 and first[0]["gain"] == pytest.approx(rsi.gain(0, 0))  # no click has reacted yet: one at a time
    game.play(click(17, 15))                               # the lamp: nothing
    rec.update(game.history, game.results)
    second = [t for t in rec.refresh(VALID) if t["kind"] == "unclicked"]
    assert len(second) == 1 and second[0]["gain"] == pytest.approx(rsi.gain(0, 1))
    game.play(click(8, 12))                                # the wall: nothing; two dead colours end the click tests
    rec.update(game.history, game.results)
    assert not [t for t in rec.refresh(VALID) if t["kind"] == "unclicked"]
    game.play(click(17, 6))                                # the button reacts: the colours left are listed again
    rec.update(game.history, game.results)
    left = [t for t in rec.refresh(VALID) if t["kind"] == "unclicked"]
    assert {t["unit"] for t in left} == {("C", 4), ("C", 9)} and left[0]["gain"] == pytest.approx(rsi.gain(0, 2))
    assert rec.results[-1]["text"] == "first click on a '5' object: did nothing"   # an unlisted test leaves no result
    assert "MOUSE 3 clicks on 3 colours of object (1 reacted)" in rec.action_line()
    game.play(click(10, 10))                               # the background is never a test
    rec.update(game.history, game.results)
    assert ("C", "bg") in rec.stats() and not any(t["unit"] == ("C", "bg") for t in rec.refresh(VALID))


def test_a_death_ends_the_attempt_and_the_record_survives_it():
    game = Game(limit=4)
    game.play("UP", "UP", "SPACE", "LEFT")                 # the fourth action is the last of the attempt
    rec = record(game)
    assert rec.event == "death" and rec.lives == [4] and rec.life == 0
    text, queue = rec.block({"record": True, "tests": True}, VALID, lead="death")
    assert text.startswith("The level restarted. Before planning the next attempt")
    assert "The level restarted 1 time(s), after 4 actions; this attempt has used 0." in text
    assert "Open tests" in text and len(queue) == rsi.TOP_TESTS
    game.play("UP", "UP", "UP", "UP")
    rec.update(game.history, game.results)
    assert rec.lives == [4, 4] and "a limit of 4 actions per attempt" in rec.lives_line()
    assert len(rec.stats()[("K", "UP")][0]) == 4            # boards of both attempts


def test_uptake_is_counted_by_context():
    game = Game(limit=6)
    game.play("UP", "UP")
    rec = record(game)
    rec.note_push(rec.refresh(VALID)[:3], "mid")
    game.play("SPACE")                                      # one of the pushed tests, at once
    rec.update(game.history, game.results)
    rec.note_push(rec.refresh(VALID)[:3], "mid")            # the same tests again: one opportunity each
    game.play("UP", "UP", "UP")                             # the attempt ends
    rec.update(game.history, game.results)
    rec.note_push(rec.refresh(VALID)[:3], "entry")
    game.play("DOWN")
    rec.update(game.history, game.results)
    taken = rec.uptake()
    assert taken["mid"]["taken"] == 1 and taken["mid"]["pushed"] >= 3
    assert taken["entry"] == {"taken": 1, "pushed": 3}


def test_a_ticking_bar_does_not_make_a_refused_move_a_change():
    game = Game(bar=True)
    game.play("UP", "UP", "LEFT", "LEFT", "LEFT", "DOWN", "DOWN", "RIGHT", "RIGHT", "SPACE", "SPACE")
    rec = record(game)
    assert rec.clock_lines() == ({5}, set())
    table = rec.stats()
    assert not table[("K", "SPACE")][0] and len(table[("K", "SPACE")][1]) == 2
    assert "row 5 ticks with every action" in rec.side_line(block=True)
    assert rec.change_text(rec.rows[-1]) == "no change (only the counter ticked)"
    plain = Game()                                          # a player that walks one row is not a counter
    plain.play(*["LEFT", "RIGHT"] * 6)
    assert record(plain).clock_lines() == (set(), set())


def test_a_cleared_level_becomes_a_prior_and_surprises_are_carried():
    game = Game()
    game.play("SPACE", "UP", "UP", "LEFT", "RIGHT", "DOWN")
    rec = record(game)
    rec.refresh(VALID)
    game.play("SPACE", *["RIGHT"] * 13)                     # SPACE again (still nothing), then out through column 19
    rec.update(game.history, game.results)
    assert rec.level == 2 and rec.event == "level" and not rec.rows
    assert rec.earlier[("K", "SPACE")] == [0, 2] and rec.earlier[("K", "RIGHT")][0] >= 13
    queue = rec.refresh(VALID)
    untried = {t["unit"][1]: t for t in queue if t["kind"] == "untried"}
    assert "RIGHT" not in untried                           # known from level 1
    assert untried["SPACE"]["gain"] == pytest.approx(rsi.gain(0, 1))
    assert "earlier levels: it changed 0 of 2 boards" in untried["SPACE"]["text"]
    text, _ = rec.block({"record": False, "tests": True}, VALID, lead="level")
    assert text.startswith("New level.") and "level 1 step 7: `SPACE` retried after its no-op at step 1" in text


def test_the_reader_form_is_parsed_and_its_test_closes_on_the_exact_actions():
    answer = """RULE: the lamp cannot be switched | WHY SHAKY: one click at step 3
TEST: click the button, then the lamp. | ACTIONS: MOUSE(row=17, col=6), mouse(17, 15) | SETTLES: a change means the button arms it
TEST: walk into the wall from below | ACTIONS: UP, FLY | SETTLES: x
GOAL: light both squares | CHECK: click each once
some prose that is dropped"""
    parsed = rsi.parse_reader(answer, VALID)
    assert parsed["rules"] == ["the lamp cannot be switched (one click at step 3)"]
    assert parsed["tests"][0]["actions"] == ["MOUSE(ROW=17,COL=6)", "MOUSE(ROW=17,COL=15)"]
    assert parsed["tests"][1]["actions"] == [] and parsed["goals"] == ["light both squares; check: click each once"]
    assert rsi.parse_reader("RULES SUPPORTED.")["supported"] and not rsi.parse_reader("")["tests"]
    game = Game()
    game.play("UP", click(17, 15))
    rec = record(game)
    rec.take_reader(parsed, game.step)
    queue = rec.refresh(VALID)
    assert [t["source"] for t in queue[:2]] == ["reader", "reader"] and queue[0]["cost"] == 2
    text, _ = rec.block({"record": True, "tests": True}, VALID)
    assert "1. [second reader] click the button, then the lamp. Actions: MOUSE(ROW=17,COL=6), MOUSE(ROW=17,COL=15)." in text
    assert "Shaky rule: the lamp cannot be switched (one click at step 3)." in text
    game.play(click(17, 6), click(17, 15))
    rec.update(game.history, game.results)
    assert rec.results[-1]["text"].endswith("executed, per action: changed, nothing") and len(rec.reader) == 1
    rec.take_reader({"rules": [], "tests": [], "goals": ["g"]}, game.step)  # the next answer replaces the open ones
    assert rec.reader == [] and rec.reader_notes == ["Untested goal: g."]


def test_the_reader_text_holds_claims_record_and_transitions_but_no_reasoning():
    game = Game(limit=20)
    game.play("UP", "UP", "LEFT", "SPACE", click(17, 15), "RIGHT", "RIGHT")
    rec = record(game)
    knowledge = {"action_model": "SPACE does nothing. The lamp is  decoration.", "goal_model": "reach the right edge",
                 "current_plan": "", "reasoning": "private"}
    tails = [rsi.reply_tail({"reasoning": "x" * 900 + " so SPACE is  useless.", "content": "World model: w"}), ""]
    text = rsi.reader_text(rec, knowledge, VALID, tokens=31000, recent=tails)
    assert len(tails[0]) == rsi.CLAIM_TAIL and 'reply ended: "...' in text and "so SPACE is useless. World model: w" in text
    assert "- Goal: reach the right edge" in text and "- Actions: SPACE does nothing. The lamp is decoration." in text
    assert "private" not in text and "step 4 SPACE: no change" in text and "'7' x1, 4 cells" in text
    assert "31,000 generated tokens" in text and "Open tests" not in text
    assert "RULE:" in rsi.reader_system() and "TEST:" in rsi.reader_system()


def agent_for(game, flags):
    from arena.duck import ArenaDuckMixin as Duck

    events, asides = [], []
    agent = NS(rsi=rsi.settings(flags), dfr=None, obs=NS(levels=0, state="NOT_FINISHED"), steps=0,
               gameplay_completion_tokens=0, duck_history=game.history, transition_results=game.results,
               drain_stats={"drains": 0}, context_compactions=0, rsi_record=rsi.Record(CHARS),
               rsi_stats=rsi.new_stats(), rsi_mark=None, rsi_mode=None, rsi_carry=False, rsi_row=None,
               rsi_shown=None, rsi_reader_at=None, rsi_tails=["and so the lamp is decoration"],
               valid_actions=lambda: VALID,
               config=NS(model="text-only"), fresh_beliefs=lambda: ({"action_model": "SPACE is dead"}, {}),
               workspace=NS(append=lambda kind, **k: events.append((kind, k))))

    async def aside(messages, prompt, **kwargs):
        asides.append((messages, prompt, kwargs))
        return agent.answer, {"usage": {"completion_tokens": 300, "prompt_tokens": 2000},
                              "choices": [{"finish_reason": "stop"}]}, 1.0, 2000

    agent.aside_request, agent.answer = aside, "TEST: press SPACE away from the wall | ACTIONS: SPACE | SETTLES: x"
    for name in ("rsi_check", "rsi_reader", "rsi_turn_lines", "rsi_report"):
        setattr(agent, name, getattr(Duck, name).__get__(agent))
    return agent, events, asides


def test_nothing_before_the_signal_then_the_block_only_when_it_changes(monkeypatch):
    import arena.duck as duck

    monkeypatch.setattr(duck, "text_only", lambda model: True)
    game = Game(limit=5)
    agent, events, asides = agent_for(game, {"record": True, "tests": True})
    asyncio.run(agent.rsi_check())                          # the level's first turn: the mark
    game.play("UP", "UP")
    agent.gameplay_completion_tokens = 59999         # a game's first level: the signal is at 60K
    asyncio.run(agent.rsi_check())
    assert agent.rsi_turn_lines() == (False, []) and not events
    agent.gameplay_completion_tokens = 60000
    asyncio.run(agent.rsi_check())
    leading, lines = agent.rsi_turn_lines()
    assert not leading and lines[0].startswith("Level record (the harness's own count over level 1")
    assert agent.rsi_turn_lines() == (False, [])            # unchanged and still in the context: not repeated
    agent.drain_stats["drains"] = 1
    assert agent.rsi_turn_lines()[1] == lines               # it left the context
    game.play("DOWN", "UP", "UP")                           # a pushed test, then the attempt ends
    asyncio.run(agent.rsi_check())
    leading, lines = agent.rsi_turn_lines()
    assert leading and lines[0].startswith("The level restarted.") and "Open tests" in lines[0]
    assert "Already tested" not in lines[0]                 # a key that works at its first try is not listed
    report = agent.rsi_report()
    assert report["entry_blocks"] == {"death": 1, "level": 0} and report["blocks"] == 3 and report["restarts"] == 1
    assert report["uptake"]["mid"]["taken"] == 1 and not asides
    assert [kind for kind, _ in events] == ["rsi_signal", "rsi_block", "rsi_block", "rsi_block"]


def test_the_level_after_a_stuck_one_opens_with_what_carries(monkeypatch):
    import arena.duck as duck

    monkeypatch.setattr(duck, "text_only", lambda model: True)
    game = Game()
    agent, events, _ = agent_for(game, {"record": True, "tests": True})
    asyncio.run(agent.rsi_check())
    game.play("SPACE", "UP", "UP", "LEFT", "RIGHT", "DOWN")
    agent.gameplay_completion_tokens = 70000
    asyncio.run(agent.rsi_check())
    agent.rsi_turn_lines()
    game.play("SPACE", *["RIGHT"] * 13)
    agent.obs.levels = 1
    asyncio.run(agent.rsi_check())
    assert agent.rsi_stats["levels"][0]["cleared"] and agent.rsi_mode is None
    leading, lines = agent.rsi_turn_lines()
    assert leading and lines[0].startswith("New level. Carried by the harness") and "Boards changed" not in lines[0]
    assert agent.rsi_turn_lines() == (False, [])
    # a level cleared before the signal carries nothing
    game = Game()
    agent, events, _ = agent_for(game, {"record": True, "tests": True})
    asyncio.run(agent.rsi_check())
    game.play(*["RIGHT"] * 13)
    agent.obs.levels = 1
    asyncio.run(agent.rsi_check())
    assert agent.rsi_turn_lines() == (False, []) and not events


def test_the_second_reader_is_called_at_the_signal_and_every_30k_and_its_test_leads_the_queue(monkeypatch):
    import arena.duck as duck

    monkeypatch.setattr(duck, "text_only", lambda model: True)
    game = Game()
    agent, events, asides = agent_for(game, {"record": True, "tests": True, "verify": True, "thinking": True})
    agent.obs.levels, agent.gameplay_completion_tokens = 1, 100000   # a second level: the signal is at 30K
    for frame in (entry.frame for entry in game.history):
        frame.level = 2
    game.level = 2
    asyncio.run(agent.rsi_check())
    game.play("SPACE", "UP", "UP")
    agent.gameplay_completion_tokens = 131000
    asyncio.run(agent.rsi_check())
    assert len(asides) == 1 and agent.rsi_stats["levels"][0]["at_tokens"] == 31000
    messages, prompt, kwargs = asides[0]
    assert messages[0]["role"] == "system" and "- Actions: SPACE is dead" in prompt
    assert 'latest reply ended: "...and so the lamp is decoration"' in prompt
    assert kwargs == {"cap": 1536, "purpose": "rsi_reader", "thinking": True, "reasoning_effort": "low"}
    _, lines = agent.rsi_turn_lines()
    assert "1. [second reader] press SPACE away from the wall. Actions: SPACE." in lines[0]
    agent.gameplay_completion_tokens = 160000
    asyncio.run(agent.rsi_check())
    assert len(asides) == 1                                 # 29K since the call
    agent.answer = "I think the player should explore more."
    agent.gameplay_completion_tokens = 161000
    asyncio.run(agent.rsi_check())
    stats = agent.rsi_stats
    assert len(asides) == 2 and stats["reader_empty"] == 1 and stats["reader_tokens"] == 600
    assert len(agent.rsi_record.reader) == 1                # an empty answer leaves the open test in place
    game.play("SPACE")
    agent.rsi_turn_lines()
    assert agent.rsi_record.results[-1]["text"].endswith("executed, per action: nothing")
    assert agent.rsi_report()["uptake"]["mid"]["taken"] >= 1


def test_a_sentinel_arm_runs_in_the_first_passes_only():
    from arena.level_bench import plan

    points = [{"id": f"p{i}", "game": f"g{i}00", "lead_levels": 1} for i in range(3)]
    arms = [{"name": "rsi", "points": "lead"}, {"name": "rsiv", "points": "lead"},
            {"name": "base", "points": "lead", "passes": 1}]
    work = plan(points, 6, 1, arms, 6)
    count = {name: sum(1 for _, _, arm in work if arm["name"] == name) for name in ("rsi", "rsiv", "base")}
    assert count == {"rsi": 18, "rsiv": 18, "base": 3}
    assert {(index, point["id"]) for index, point, arm in work if arm["name"] == "base"} == {(6, "p0"), (6, "p1"), (6, "p2")}
    for name in ("rsi", "rsiv"):  # every point once per pass and arm
        assert {(i, p["id"]) for i, p, a in work if a["name"] == name} == {(i, f"p{k}") for i in range(6, 12) for k in range(3)}
