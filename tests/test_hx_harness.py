"""Harness arm ``hx_harness`` (arena.hx_harness): schema and hero-v2 recording, the sandbox toolbox (registered tools
and observers, both request loops, on top of dfr_harness), the user-turn listing, the STEP BACK reflector's triggers,
and the component off: unchanged payloads, prompts and sandbox."""

import asyncio
import json

import numpy as np
import pytest
from conftest import observation, raw
from test_arena_duck import arena_agent
from test_dfr_harness import Game, Transport, board, bootstraps, run, texts, user_turns

from arena import dfr_harness as dfr
from arena import hx_harness as hx
from arena.duck import ArenaTransferDuckAgent

ALL = hx.settings(True)


def test_settings_schema():
    assert hx.settings(None) is None and hx.settings(False) is None and hx.settings({}) is None
    assert {f: ALL[f] for f in hx.FLAGS} == {f: True for f in hx.FLAGS}
    assert {k: ALL[k] for k in hx.DEFAULTS} == hx.DEFAULTS
    one = hx.settings({"reflect": True, "reflect_requests": 5})
    assert one["reflect"] and not one["toolbox"] and one["reflect_requests"] == 5
    assert hx.settings(one) == one  # idempotent (hero_v2 normalizes then composes)
    with pytest.raises(ValueError, match="Unknown hx_harness keys"):
        hx.settings({"reflect": True, "mirror": True})
    with pytest.raises(ValueError, match="boolean"):
        hx.settings({"reflect": 1})
    with pytest.raises(ValueError, match="positive integer"):
        hx.settings({"reflect": True, "reflect_every": 0})


def test_hero_v2_records_the_component_only_when_on():
    from test_competition_run_b import RUN_B, SERVED, hero

    from arena import hero_v2

    off = hero_v2.compose(hero(), {**RUN_B}, served_blocks=SERVED)
    explicit = hero_v2.compose(hero(), {**RUN_B, "hx_harness": False}, served_blocks=SERVED)
    assert json.dumps(off, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert "hx_harness" not in off and "hx_harness" not in off["hero_v2"]["components"]
    on = hero_v2.compose(hero(), {**RUN_B, "hx_harness": {"toolbox": True}}, served_blocks=SERVED)
    assert on["hx_harness"]["toolbox"] and not on["hx_harness"]["reflect"]
    assert on["hero_v2"]["components"]["hx_harness"]["on"]
    strip = {k: v for k, v in on.items() if k not in ("hx_harness", "hero_v2", "arena_variant")}
    base = {k: v for k, v in off.items() if k not in ("hero_v2", "arena_variant")}
    assert json.dumps(strip, sort_keys=True) == json.dumps(base, sort_keys=True)
    assert hero_v2.status()["hx_harness"]["available"]


# --- the sandbox toolbox ---------------------------------------------------------------------------------------------

def make_sandbox(fixes, with_dfr=True):
    from duck_transfer.sandbox import StatefulSandbox

    original, plain, fixed = bootstraps()
    wrap = (lambda text: hx.with_toolbox(dfr.with_dfr(text, dfr.settings(True)), ALL)) if with_dfr else (
        lambda text: hx.with_toolbox(text, ALL))
    sandbox = StatefulSandbox(wrap(plain), color_chars=original["ARC_COLOR_CHARS"])
    if fixes:
        sandbox.configure_fixes(wrap(fixed), interrupt=True, grace=3)
    return sandbox


@pytest.fixture(params=[(True, True), (False, True), (True, False)], ids=["fixes-dfr", "plain-dfr", "fixes-only"])
def box(request):
    sandbox = make_sandbox(*request.param)
    yield sandbox
    sandbox.close()


def test_registered_tools_persist_and_are_reported(box):
    game = Game(level=1)
    code = ("def where(frame):\n    '''Locate the thing.'''\n    return frame.step\n"
            "print(register_tool(where), register_tool(lambda: 1, note='anon', name='one'))")
    out = run(box, code, game)
    assert not out["error"], out
    assert out["stdout"] == "where one\n"
    state = box.hx_state
    assert state["tools"] == {"where": {"sig": "(frame)", "note": "Locate the thing."},
                              "one": {"sig": "()", "note": "anon"}}
    out = run(box, "print(where(current_frame), one())", game)  # persists into the next call
    assert out["stdout"] == "0 1\n"
    out = run(box, "print(unregister('one'))", game)
    assert out["stdout"] == "True\n" and list(box.hx_state["tools"]) == ["where"]
    lines = hx.toolbox_lines(box.hx_state, ALL)
    assert lines == "Your registered tools (persist across turns; call them from python): where(frame) - Locate the thing."


def test_observers_run_after_every_executed_action(box):
    game = Game(level=1)
    code = ("def steps(before, after):\n    return f'{before.step}->{after.step}'\n"
            "def broken(before, after):\n    return 1 / 0\n"
            "register_observer(steps)\nregister_observer(broken)\n"
            "r = action(['UP'])\nprint(r['observers'])")
    out = run(box, code, game)
    assert not out["error"], out
    assert "'steps': '0->1'" in out["stdout"] and "observer error: ZeroDivisionError" in out["stdout"]
    state = box.hx_state
    assert state["out"]["steps"] == "0->1" and state["out_step"] == 1 and state["errors"] == 1
    assert set(state["observers"]) == {"steps", "broken"}
    text = hx.toolbox_lines(state, ALL)
    assert text.startswith("Your observers after the last action (step 1): steps: 0->1 | broken: observer error")
    # Observers never break a guarded or refused call; a non-executed result carries none.
    out = run(box, "unregister('broken')\nr = action(['UP'])\nprint(sorted(r['observers']))", game)
    assert out["stdout"] == "['steps']\n" and box.hx_state["out"] == {"steps": "1->2"}


def test_the_dfr_latches_still_apply_under_the_observer_wrapper():
    sandbox = make_sandbox(True, True)
    try:
        game = Game(level=2)
        code = ("register_observer(lambda b, a: 'seen')\naction(['LEFT'])\ntry:\n    action(['UP'])\n"
                "except StaleStateActionError:\n    print('stale')")
        out = run(sandbox, code, game)
        assert not out["error"] and out["stdout"] == "stale\n" and game.step == 1
        out = run(sandbox, "action(['DOWN'])\naction(['UP'])", game)
        assert "TerminalStateActionError" in out["error"] and game.step == 2
    finally:
        sandbox.close()


def test_the_toolbox_patch_is_pinned_and_installed_once():
    _, plain, fixed = bootstraps()
    for text in (plain, fixed):
        patched = hx.with_toolbox(text, ALL)
        assert patched.count("_HX = {") == 1 and patched.count('"hx": _hx_take(),') == 2
        with pytest.raises(ValueError, match="already installed"):
            hx.with_toolbox(patched, ALL)
    assert hx.HX_RUNTIME_NAMES in hx.with_toolbox(fixed, ALL)


# --- the reflector ---------------------------------------------------------------------------------------------------

class Entry:
    def __init__(self, action, grid, step):
        self.action = action
        self.frame = type("F", (), {"grid": grid, "step": step})()


def grid(value=0, at=(10, 10)):
    rows = [[0] * 16 for _ in range(16)]
    rows[at[0]][at[1]] = value
    return tuple(tuple(r) for r in rows)


def test_the_reflector_fires_on_requests_streaks_revisits_and_deaths():
    block = hx.settings({"reflect": True, "reflect_requests": 4, "reflect_every": 4, "reflect_no_change": 3,
                         "reflect_revisits": 3, "reflect_deaths": 2, "reflect_min_gap": 1})
    r = hx.Reflector(block)
    history = [Entry("", grid(), 0)]
    r.update(history, {}, 0, 0, 0)
    assert r.notice(1) is None
    # Three interior no-ops (the board never changes): the streak fires.
    for step in (1, 2, 3):
        history.append(Entry("UP", grid(), step))
    r.update(history, {}, 0, 1, 100)
    text = r.notice(1)
    assert text.startswith("STEP BACK (harness check: your last 3 actions changed nothing inside the board)")
    assert "UP 3x (0 changed the board)" in text and "register_tool(fn)" in text and "register_observer(fn)" in text
    assert "How can you modify your own code or your interfaces to the environment to improve?" in text
    # The same board seen 4 times: revisits (the streak already noticed).
    r.update(history, {}, 0, 2, 200)
    assert r.last_reason == "no_change" and r.notice(1).startswith("STEP BACK (harness check: this exact board")
    # Requests: 4 on the level, but the clock restarts at the last notice of any kind (request 2): due at 6.
    history.append(Entry("LEFT", grid(1), 4))
    r.update(history, {}, 0, 4, 400)
    assert r.notice(1) is None
    r.update(history, {}, 0, 6, 400)
    assert "6 model requests on this level" in r.notice(1)
    # Two deaths.
    history.append(Entry("DOWN", grid(2), 5))
    history.append(Entry("RESET", grid(), 6))
    history.append(Entry("DOWN", grid(2), 7))
    r.update(history, {5: {"game_over": True}, 7: {"game_over": True}}, 0, 7, 500)
    text = r.notice(1)
    assert text.startswith("STEP BACK (harness check: 2 deaths on this level)") and "DOWN 2x (2 changed the board, 2 deaths)" in text
    # A level clear resets every counter.
    history.append(Entry("RIGHT", grid(3), 8))
    r.update(history, {}, 1, 8, 600)
    assert r.actions == 0 and r.requests == 0 and r.notice(2) is None
    assert r.summary() == {"notices": 4, "no_change": 1, "revisits": 1, "requests": 1, "deaths": 1}


def test_the_minimum_gap_holds_back_a_second_notice():
    r = hx.Reflector(hx.settings({"reflect": True, "reflect_no_change": 1, "reflect_min_gap": 3}))
    history = [Entry("", grid(), 0), Entry("UP", grid(), 1)]
    r.update(history, {}, 0, 1, 0)
    assert r.notice(1) is not None
    history.append(Entry("UP", grid(), 2))
    r.update(history, {}, 0, 2, 0)
    assert r.notice(1) is None  # one request later
    r.update(history, {}, 0, 4, 0)
    assert r.notice(1) is not None


# --- the agent -------------------------------------------------------------------------------------------------------

async def run_agent(tmp_path, boards, codes, block, *, dfr_block=True):
    transport = Transport(*codes)
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    agent.apply_dfr_harness(dfr_block)
    agent.apply_hx_harness(block)
    obs = [observation(b) for b in boards]
    env.observations = iter(obs[1:])
    env.observation_space = raw(obs[0])
    agent.config.max_actions = len(boards) - 1
    result = await agent.run()
    return agent, transport, result


async def test_component_off_keeps_the_payloads_prompts_and_sandbox(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    codes = ["action(['UP'])", "action(['UP'])"]
    control, ct, cr = await run_agent(tmp_path / "control", boards, codes, None)
    off, ot, orr = await run_agent(tmp_path / "off", boards, codes, False)
    assert cr["error"] is None and orr["error"] is None
    assert json.dumps(ct.payloads, sort_keys=True) == json.dumps(ot.payloads, sort_keys=True)
    assert control.sandbox.bootstrap == off.sandbox.bootstrap and off.hx is None and "hx_harness" not in orr


async def test_the_toolbox_turn_listing_and_system_text(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20)), board((10, 10), (20, 20), (30, 30))]
    codes = ["def pos(b, a):\n    return 'step %d' % a.step\nregister_observer(pos, note='where')\n"
             "register_tool(lambda: 0, name='zero', note='nothing')\naction(['UP'])",
             "action(['UP'])", "action(['UP'])"]
    agent, transport, result = await run_agent(tmp_path, boards, codes, {"toolbox": True})
    assert result["error"] is None and result["actions"] == 3
    system = transport.payloads[0]["messages"][0]["content"]
    system = system if isinstance(system, str) else json.dumps(system)
    assert "register_observer(fn, note=None)" in system and "Tool session rules" in system
    second = texts(user_turns(transport.payloads[1])[-1])
    assert "Your registered tools (persist across turns; call them from python): zero() - nothing" in second
    assert "Your observers after the last action (step 1): pos: step 1" in second
    # Append-only history: every request extends the previous one.
    for before, after in zip(transport.payloads, transport.payloads[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]
    stats = result["hx_harness"]
    assert stats["max_tools"] == 1 and stats["max_observers"] == 1 and stats["toolbox_turns"] >= 2


async def test_the_reflect_block_reaches_the_user_turn(tmp_path):
    same = board()
    boards = [same, same, same, same]
    agent, transport, result = await run_agent(
        tmp_path, boards, ["action(['UP'])"] * 3,
        {"reflect": True, "reflect_no_change": 2, "reflect_requests": 50, "reflect_min_gap": 1})
    assert result["error"] is None
    turns = [texts(user_turns(p)[-1]) for p in transport.payloads]
    assert any("STEP BACK (harness check: your last 2 actions changed nothing inside the board)" in t for t in turns)
    assert result["hx_harness"]["reflect_notices"] >= 1 and result["hx_harness"]["reflect"]["no_change"] >= 1


async def test_attach_wires_the_study_block(tmp_path):
    agent, *_ = arena_agent(tmp_path, None, ArenaTransferDuckAgent)
    assert agent.hx is None
    agent.apply_hx_harness({"toolbox": True})
    assert agent.hx["toolbox"] and "_HX = {" in agent.sandbox.bootstrap
    text = agent.analyzer(asyncio.get_running_loop())._system_prompt
    assert text.count("Self-built tools (you can extend your own interface):") == 1
    agent.executor.shutdown()
    assert np is not None


# --- probe and perceive ----------------------------------------------------------------------------------------------

def test_probe_runs_every_action_past_an_interior_no_op():
    sandbox = make_sandbox(True, True)
    try:
        game = Game(level=2)  # the dfr stale latch is armed from level 2
        out = run(sandbox, "rows = probe(['LEFT', 'UP', 'DOWN', 'UP'])\nprint([(r['action'], r['gameplay_changed'], "
                           "r.get('game_over', False)) for r in rows])", game)
        assert not out["error"], out
        assert out["stdout"] == "[('LEFT', False, False), ('UP', True, False), ('DOWN', True, True)]\n"
        assert game.step == 3  # stopped after the game over
    finally:
        sandbox.close()


def big(*cells, value=9, edge=None):
    rows = [[0] * 64 for _ in range(64)]
    for r, c in cells:
        rows[r][c] = value
    if edge is not None:
        rows[0][edge] = 3
    return tuple(tuple(r) for r in rows)


def perceiver():
    import numpy as np

    from arc_infra.transition import TransitionCache, summarize

    cache = TransitionCache()
    p = hx.Perceiver(hx.settings({"perceive": True, "digest_actions": 4}),
                     lambda g: cache.objects(np.asarray(g, dtype=np.uint8)))

    def receipt(before, after, **extra):
        b, a = np.asarray(before, dtype=np.uint8), np.asarray(after, dtype=np.uint8)
        return {"transition": summarize(b, a, before_objects=cache.objects(b), after_objects=cache.objects(a)),
                **extra}
    return p, receipt


def test_the_digest_describes_effects_ledger_untried_actions_and_resets():
    p, receipt = perceiver()
    start = big((10, 10), (10, 11))
    history = [Entry("", start, 0)]
    assert p.turn(history, {}, 0, ["UP", "DOWN", "MOUSE"]) == [
        "Scene (harness segmentation, 4-connected same-colour components): large regions: W 4094 cells; objects "
        "(count, colour, cells, height x width, top-left): 1x b 2px 1x2 at (10,10)."]
    moved = big((10, 12), (10, 13))
    edge = big((10, 12), (10, 13), edge=5)
    history += [Entry("UP", moved, 1), Entry("DOWN", edge, 2), Entry("MOUSE(row=10, col=12)", edge, 3)]
    results = {1: receipt(start, moved), 2: receipt(moved, edge), 3: receipt(edge, edge)}
    lines = p.turn(history, results, 0, ["UP", "DOWN", "MOUSE", "SPACE"])
    assert lines[0].startswith("Harness digest - effects since your previous turn: UP: 4 cells changed in rows 10-10, "
                               "cols 10-13; moved: b 2px (10,10)->(10,12)")
    assert "DOWN: only the outer edge ring changed (1 cells; HUD/timer?)" in lines[0]
    assert "MOUSE(row=10, col=12): no change" in lines[0]
    assert lines[1] == ("Ledger for this level: DOWN 1x (0 changed the interior); MOUSE 1x (0 changed the interior) "
                        "[clicked on: b 1x/0 changed]; UP 1x (1 changed the interior). Not yet tried on this level: "
                        "SPACE.")
    # Back to the level's start board: noted once.
    history.append(Entry("LEFT", start, 4))
    lines = p.turn(history, {4: receipt(edge, start)}, 0, ["UP"])
    assert any(line.startswith("Note: the board is identical to this level's starting board again (after 4 actions")
               for line in lines)
    assert not any(line.startswith("Note:") for line in p.turn(history, {}, 0, ["UP"]))


def test_a_new_level_lists_the_new_objects_and_resets_the_ledger():
    p, receipt = perceiver()
    start = big((10, 10))
    history = [Entry("", start, 0)]
    p.turn(history, {}, 0, ["UP"])
    level2 = tuple(tuple(r) for r in [[0] * 64 for _ in range(30)] + [[0] * 30 + [4, 4] + [0] * 32] +
                   [[0] * 64 for _ in range(33)])
    history.append(Entry("UP", level2, 1))
    lines = p.turn(history, {1: receipt(start, level2, level_completed=True)}, 1, ["UP"])
    assert "-> LEVEL CLEARED" in lines[0]
    assert lines[2].startswith("Scene (harness segmentation")
    assert lines[1].startswith("New level: objects whose colour and shape were not on the previous level's start: "
                               "c 2px at (30,30); 1 earlier object kinds are gone.")
    assert p.ledger == {} and p.level == 1


async def test_the_digest_reaches_the_user_turn_with_every_flag_on(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20)), board((10, 10), (20, 20), (30, 30))]
    codes = ["print(probe(['UP']))", "action(['UP'])", "action(['UP'])"]
    every = {**{flag: True for flag in hx.FLAGS}, "autoexplore": False}  # autoexplore would use up the boards
    agent, transport, result = await run_agent(tmp_path, boards, codes, every)
    assert result["error"] is None and result["actions"] == 3
    system = json.dumps(transport.payloads[0]["messages"][0]["content"])
    assert "probe(actions)" in system and "harness digest" in system and "register_tool" in system
    second = texts(user_turns(transport.payloads[1])[-1])
    assert "Harness digest - effects since your previous turn: UP: 1 cells changed" in second
    assert "Ledger for this level: UP 1x (1 changed the interior)." in second
    for before, after in zip(transport.payloads, transport.payloads[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]
    assert result["hx_harness"]["perceive"]["digests"] >= 2


async def test_economy_rewrites_the_action_line_and_counts_level_actions(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20)), board((10, 10), (20, 20), (30, 30))]
    agent, transport, result = await run_agent(tmp_path, boards, ["action(['UP', 'UP'])", "action(['UP'])"],
                                               {"economy": True})
    assert result["error"] is None
    system = json.dumps(transport.payloads[0]["messages"][0]["content"])
    assert "Your thinking time is the scarce resource" in system and "as few in-game actions as possible" not in system
    turns = [texts(user_turns(p)[-1]) for p in transport.payloads]
    assert "Action economy: 0 actions so far on level 1" in turns[0]
    assert "Action economy: 2 actions so far on level 1" in turns[1]


def test_the_scene_summary_groups_objects_and_finds_border_strips():
    p, _ = perceiver()
    rows = [[0] * 64 for _ in range(64)]
    for c in range(10, 50):
        rows[63][c] = 3  # a bar flush with the bottom border
    for r0, c0 in ((5, 5), (5, 20), (30, 30)):
        for r in range(r0, r0 + 3):
            for c in range(c0, c0 + 3):
                rows[r][c] = 9
    rows[40][40] = 8
    text = hx.scene_summary(rows, p.objects(tuple(tuple(r) for r in rows)), 14)
    assert text.startswith("Scene (harness segmentation, 4-connected same-colour components): large regions: W ")
    assert "1x R 1px 1x1 at (40,40); 3x b 9px 3x3 at (5,5), (5,20), (30,30)" in text
    assert text.endswith("thin strips at the border (likely HUD/timer bars): G 40px rows 63-63, cols 10-49.")


def test_the_kit_helpers_read_the_board():
    sandbox = make_sandbox(True, True)
    try:
        grids = [[[0] * 8 for _ in range(8)], [[0] * 8 for _ in range(8)]]
        for r in (2, 3):
            for c in (4, 5, 6):
                grids[0][r][c] = 9
                grids[1][r][c] = 9
        grids[1][3][6] = 2
        class AsciiGame(Game):
            def frame(self):
                frame = super().frame()
                frame["ascii"] = "\n".join("".join(hx.COLOR_CHARS[v] for v in row) for row in self.grid)
                return frame

        game = AsciiGame(level=1, grids=grids)
        code = ("print(crop(1, 4, 3, 7))\nprint(objects(color='b'))\nprint(at(2, 5))\n"
                "action(['UP'])\nprint(changes())")
        out = run(sandbox, code, game)
        assert not out["error"], out
        lines = out["stdout"].split("\n")
        assert lines[0] == "    0000" and lines[1] == "    3456" and lines[2] == "  1 WWWW" and lines[3] == "  2 Wbbb"
        assert "'bbox': [2, 4, 3, 6], 'height': 2, 'width': 3, 'center': (2.5, 5.0)" in lines[5]
        assert lines[6] == "{'color': 'b', 'pixels': 6, 'bbox': [2, 4, 3, 6], 'center': (2.5, 5.0)}"
        assert lines[7] == "{'count': 1, 'bbox': [3, 6, 3, 6], 'cells': [(3, 6, 'b', 'g')]}"
    finally:
        sandbox.close()


def test_a_research_clone_wall_fits_a_slow_start_and_a_submission_never_does():
    from arena.competition import gameplay_seconds

    study = {"competition": {"cleanup_seconds": 60, "preview_seconds": 900, "research_clones": True,
                             "fit_wall_min_seconds": 600}}
    assert gameplay_seconds(study, 2000, now=0) == 900
    assert gameplay_seconds(study, 1500, now=0) == 900  # 1,440 s available
    assert gameplay_seconds(study, 900, now=0) == 840
    with pytest.raises(TimeoutError):
        gameplay_seconds(study, 600, now=0)
    plain = {"competition": {"cleanup_seconds": 60, "preview_seconds": 900}}
    with pytest.raises(TimeoutError):
        gameplay_seconds(plain, 900, now=0)


def test_style_two_fires_on_stalls_not_on_request_counts():
    r = hx.Reflector(hx.settings({"reflect": True, "reflect_style": 2, "reflect_stall": 3, "reflect_min_gap": 1,
                                  "reflect_requests": 2}))
    history = [Entry("", grid(), 0), Entry("UP", grid(1), 1)]
    r.update(history, {}, 0, 2, 100)  # an interior change at request 2
    assert r.notice(1) is None        # style 2 has no request-count trigger
    r.update(history, {}, 0, 4, 200)
    assert r.notice(1) is None        # 2 requests since the change
    r.update(history, {}, 0, 5, 300)
    text = r.notice(1)
    assert text.startswith("STEP BACK (harness check: 3 model requests since an action last changed the board interior)")
    assert "modify your own code or your interface to the environment" in text and "register_tool(fn)" in text
    r.update(history, {}, 0, 7, 300)
    assert r.notice(1) is None        # the clock restarts at the notice
    r.update(history, {}, 0, 8, 300)
    assert r.notice(1) is not None and r.summary()["stall"] == 2


async def test_economy_style_two_also_replaces_the_tool_call_lines(tmp_path):
    agent, *_ = arena_agent(tmp_path, None, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    agent.apply_dfr_harness(True)
    agent.apply_hx_harness({"economy": True, "economy_style": 2})
    text = agent.analyzer(asyncio.get_running_loop())._system_prompt
    agent.executor.shutdown()
    assert hx.CALLS_NEW in text and hx.CALLS_OLD not in text and hx.ECONOMY_NEW in text


async def test_economy_spend_reports_the_cost_since_the_previous_turn(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20)), board((10, 10), (20, 20), (30, 30))]
    agent, transport, result = await run_agent(tmp_path, boards, ["print(1)", "action(['UP'])", "action(['UP'])",
                                                                  "action(['UP'])"],
                                               {"economy": True, "economy_spend": 1})
    assert result["error"] is None
    turns = [texts(user_turns(p)[-1]) for p in transport.payloads]
    assert "Since your previous turn: 0 model requests, 0 generated tokens, 0 s of play time." in turns[0]
    assert any("Since your previous turn: 2 model requests, 100 generated tokens" in t for t in turns)
    assert hx.economy_line(3, 2, (1, 1500, 61.4)).endswith(
        "Since your previous turn: 1 model request, 1,500 generated tokens, 61 s of play time.")


async def test_the_opener_appears_once_per_level(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    agent, transport, result = await run_agent(tmp_path, boards, ["action(['UP'])", "action(['UP'])"],
                                               {"opener": True, "probe": True})
    assert result["error"] is None
    turns = [texts(user_turns(p)[-1]) for p in transport.payloads]
    assert turns[0].count("Level opener (first turn on level 1)") == 1 and "`probe([...])`" in turns[0]
    assert all("Level opener" not in t for t in turns[1:])
    line = hx.opener_line(2, ["UP", "DOWN", "MOUSE", "RESET"], False)
    assert "tests each of UP, DOWN once and MOUSE on the two or three most distinctive objects with a loop" in line


async def test_economy_style_three_adds_the_lean_reasoning_line(tmp_path):
    agent, *_ = arena_agent(tmp_path, None, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    agent.apply_dfr_harness(True)
    agent.apply_hx_harness({"economy": True, "economy_style": 3})
    text = agent.analyzer(asyncio.get_running_loop())._system_prompt
    agent.executor.shutdown()
    assert hx.ECONOMY_NEW + hx.LEAN_REASONING in text and hx.CALLS_OLD in text


async def test_the_guide_goes_before_the_visual_guidance(tmp_path):
    agent, *_ = arena_agent(tmp_path, None, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    agent.apply_dfr_harness(True)
    agent.apply_hx_harness({"guide": True})
    text = agent.analyzer(asyncio.get_running_loop())._system_prompt
    agent.executor.shutdown()
    assert text.count("Field guide (generic patterns of these games") == 1
    assert text.index("Field guide") < text.index("Visual-game guidance:")


def test_the_coordinate_margin_keeps_the_board_pixels():
    import base64
    import io

    from PIL import Image

    grid = [[0] * 64 for _ in range(64)]
    grid[10][20] = 9
    url = dfr.board_image_url(grid)
    out = hx.with_coordinates(url)
    image = Image.open(io.BytesIO(base64.b64decode(out.partition(",")[2]))).convert("RGB")
    source = Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2]))).convert("RGB")
    assert image.size == (672, 672) and source.size == (640, 640)
    assert image.crop((32, 32, 672, 672)).tobytes() == source.tobytes()
    assert image.getpixel((0, 40)) == hx.GRID_MARGIN_RGB
    assert hx.with_coordinates("data:image/jpeg;base64,xx") == "data:image/jpeg;base64,xx"


async def test_grid_labels_the_observation_images_and_keeps_history_append_only(tmp_path):
    from duck_transfer.trimming import png_dimensions

    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    agent, transport, result = await run_agent(tmp_path, boards, ["action(['UP'])", "action(['UP'])"],
                                               {"grid": True})
    assert result["error"] is None
    second = user_turns(transport.payloads[1])[-1]
    urls = [p["image_url"]["url"] for p in second["content"] if p.get("type") == "image_url"]
    assert [png_dimensions(u) for u in urls] == [(672, 672), (672, 672)]  # board + diff
    system = json.dumps(transport.payloads[0]["messages"][0]["content"])
    assert "dark brown margin on its top and left" in system
    for before, after in zip(transport.payloads, transport.payloads[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]


def test_the_lattice_helper_infers_the_stride_from_moves():
    sandbox = make_sandbox(True, True)
    try:
        def grid_with(r, c):
            g = [[0] * 24 for _ in range(24)]
            for rr in range(2, 22, 6):  # walls every 6 px from row 2
                for cc in range(24):
                    g[rr][cc] = 5
            for rr in range(r, r + 4):
                for cc in range(c, c + 4):
                    g[rr][cc] = 9
            return g

        class Moving(Game):
            def frame(self):
                frame = super().frame()
                frame["ascii"] = "\n".join("".join(hx.COLOR_CHARS[v] for v in row) for row in self.grid)
                return frame

            def __call__(self, actions):
                reply = super().__call__(actions)
                before = self.history[-2]["frame"]["grid"]
                moved = {"objects": {"moved": [{"bbox": [3, 3 + 6 * self.step, 7, 7 + 6 * self.step],
                                                "from": [3, 3 + 6 * (self.step - 1), 7, 7 + 6 * (self.step - 1)],
                                                "color": 9, "pixels": 16}]}}
                reply["action_result"]["transition"] = moved
                self.history[-1]["result"]["transition"] = moved
                assert before is not None
                return reply

        game = Moving(level=1, grids=[grid_with(3, 3), grid_with(3, 9), grid_with(3, 15)])
        out = run(sandbox, "action(['UP'])\naction(['UP'])\nr = lattice()\nprint(r['pitch'], r['origin'], r['rows'], "
                           "r['cols'])\nprint(r['map'][0])\nprint(lattice(pitch=6, origin=(3, 3))['pitch'])", game)
        assert not out["error"], out
        lines = out["stdout"].split("\n")
        assert lines[0] == "(6, 6) (3, 3) 3 3" and lines[2] == "(6, 6)"
        out = run(sandbox, "print(lattice(frame=history[0].frame) if False else 'ok')", game)
        assert out["stdout"] == "ok\n"
    finally:
        sandbox.close()


def test_stride_and_hint_on_the_host():
    moved = lambda d: {"transition": {"objects": {"moved": [{"bbox": [10, 10 + d, 14, 14 + d], "from": [10, 10, 14, 14]}]}}}  # noqa: E731
    assert hx.stride_of([moved(6)]) is None             # one step only
    assert hx.stride_of([moved(6), moved(12)]) == (6, (10, 22))
    assert hx.stride_of([moved(1), moved(2)]) is None   # pitch 1: no lattice
    assert "lattice of 6 x 6-pixel cells (one cell starts at row 4, col 4)" in hx.lattice_hint(6, (10, 22))


def test_coverage_names_no_effect_actions_tried_on_few_boards():
    c = hx.Coverage()
    a, b = grid(0), grid(1, at=(5, 5))
    history = [Entry("", a, 0)]
    c.update(history, 0)
    history += [Entry("UP", a, 1), Entry("RIGHT", b, 2)]
    c.update(history, 0)
    text = c.line(b)
    assert text.startswith("Coverage: UP did nothing on the 1 board where it was tried; the board has changed since")
    assert c.line(b) is None                            # not repeated verbatim
    history.append(Entry("UP", b, 3))
    c.update(history, 0)
    assert c.line(b) is None                            # now tried here too
    c.update(history + [Entry("LEFT", a, 4)], 1)        # a new level resets
    assert c.tried == {}


async def test_coverage_and_lattice_flags_run_in_the_agent(tmp_path):
    same = board()
    boards = [same, same, board((10, 10)), board((10, 10)), board((10, 10), (20, 20))]
    agent, transport, result = await run_agent(tmp_path, boards, ["action(['UP'])", "action(['LEFT'])",
                                                                  "print(lattice())", "action(['UP'])",
                                                                  "action(['UP'])"],
                                               {"coverage": True, "lattice": True})
    assert result["error"] is None
    system = json.dumps(transport.payloads[0]["messages"][0]["content"])
    assert "lattice(pitch=None, origin=None" in system
    assert agent.hx_coverage.tried and "UP" in agent.hx_coverage.tried  # the tracker follows the real history
    for before, after in zip(transport.payloads, transport.payloads[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]


def test_detect_lattice_finds_exact_block_boards_only():
    import numpy as np

    logical = np.random.default_rng(0).integers(0, 4, size=(11, 11))
    board_ = np.zeros((64, 64), dtype=np.int64)
    board_[4:59, 4:59] = np.kron(logical, np.ones((5, 5), dtype=np.int64))
    found = hx.detect_lattice(board_)
    assert found is not None and found[0] == 5 and found[1] == (4, 4) and found[2] == 1.0
    assert "lattice of 5 x 5-pixel cells starting at row 4, col 4" in hx.lattice_scene_hint(found)
    free = np.zeros((64, 64), dtype=np.int64)
    for r, c, h, w in ((3, 7, 9, 4), (20, 31, 5, 11), (40, 2, 7, 7), (50, 45, 3, 13)):
        free[r:r + h, c:c + w] = 3
    assert hx.detect_lattice(free) is None


def test_merge_notes_report_contact_and_separation():
    import numpy as np

    from arc_infra.transition import TransitionCache

    cache = TransitionCache()
    apart = np.zeros((16, 16), dtype=np.uint8)
    apart[5:7, 2:4] = 9
    apart[5:7, 8:10] = 9
    touch = np.zeros((16, 16), dtype=np.uint8)
    touch[5:7, 4:6] = 9
    touch[5:7, 6:8] = 9
    nodes = lambda g: cache.objects(g)["nodes"]  # noqa: E731
    (note,) = hx.merge_notes(nodes(apart), nodes(touch), "RIGHT")
    assert note.startswith("after RIGHT, 2 separate b objects now touch and show as one component (8 cells)")
    (back,) = hx.merge_notes(nodes(touch), nodes(apart), "LEFT")
    assert back.startswith("after LEFT, a b component split into 2 parts (8 cells)")
    assert hx.merge_notes(nodes(apart), nodes(apart), "UP") == []


async def test_merges_reach_the_user_turn(tmp_path):
    apart, touch = board((10, 10), (10, 14)), board((10, 11), (10, 12))
    agent, transport, result = await run_agent(tmp_path, [apart, touch, touch], ["action(['UP'])", "action(['UP'])"],
                                               {"merges": True})
    assert result["error"] is None
    turns = [texts(user_turns(p)[-1]) for p in transport.payloads]
    assert any("Harness: after UP, 2 separate" in t and "now touch and show as one component" in t for t in turns)


# --- phase 2: autopilot / explore / world model / notes ---------------------------------------------------------------

class AsciiGame(Game):
    """Game with ascii frames; UP adds a cell (interior change), LEFT nothing, DOWN game over, RIGHT clears."""

    def frame(self):
        frame = super().frame()
        frame["ascii"] = "\n".join("".join(hx.COLOR_CHARS[v] for v in row) for row in self.grid)
        return frame

    def __call__(self, actions):
        (item,) = actions
        name = item["action"] if isinstance(item, dict) else item
        if name == "UP":
            self.grid = [row[:] for row in self.grid]
            k = sum(map(sum, self.grid)) // 9 if self.grid else 0
            self.grid[2 + k // 4][2 + k % 4] = 9
        return super().__call__([{"action": name}])


@pytest.fixture
def full_box():
    from duck_transfer.sandbox import StatefulSandbox

    original, plain, fixed = bootstraps()
    block = hx.settings(True)
    sandbox = StatefulSandbox(hx.with_toolbox(dfr.with_dfr(plain, dfr.settings(True)), block),
                              color_chars=original["ARC_COLOR_CHARS"])
    sandbox.configure_fixes(hx.with_toolbox(dfr.with_dfr(fixed, dfr.settings(True)), block), interrupt=True, grace=3)
    yield sandbox
    sandbox.close()


def test_autopilot_stops_on_policy_noops_and_terminal(full_box):
    game = AsciiGame(level=2)
    out = run(full_box, "seq = iter(['UP', 'UP', 'UP'])\nr = autopilot(lambda f: next(seq, None))\n"
                        "print(r['stop'], r['actions'])", game)
    assert not out["error"], out
    assert out["stdout"] == "policy_done 3\n" and game.step == 3
    out = run(full_box, "r = autopilot(lambda f: 'LEFT', max_noops=2)\nprint(r['stop'], r['actions'])", game)
    assert out["stdout"] == "no_change 2\n"  # the dfr stale latch does not abort the loop
    out = run(full_box, "r = autopilot(lambda f: 'RIGHT')\nprint(r['stop'], r['actions'])", game)
    assert out["stdout"] == "level_completed 1\n"
    out = run(full_box, "r = autopilot(lambda f: 'UP', expect=lambda b, a, x: False)\nprint(r['stop'])", game)
    assert out["stdout"] == "surprise\n"
    assert full_box.hx_state["loops"] == 4 and full_box.hx_state["loop_actions"] == 7


def test_explore_tries_untried_actions_per_state(full_box):
    game = AsciiGame(level=1)
    out = run(full_box, "r = explore(max_actions=6)\nprint(r['stop'], r['actions'], r['effects'])", game)
    assert not out["error"], out
    stop, n, effects = out["stdout"].split(" ", 2)
    assert int(n) >= 2 and "UP" in effects and "LEFT" in effects


def test_check_sim_register_sim_and_plan(full_box):
    game = AsciiGame(level=1)
    code = ("action(['UP'])\naction(['LEFT'])\nprint(check_sim(lambda r, a: r))\n"
            "print(register_sim(lambda r, a: r)['exact'])\n"
            "def sim(rows, a):\n    n = int(rows[0]) if rows[0].isdigit() else 0\n"
            "    return [str(n + (1 if a == 'UP' else 2))]\n"
            "print(plan(sim, lambda r: r[0] == '5', actions=['UP', 'DOWN'], start=['0']))")
    out = run(full_box, code, game)
    assert not out["error"], out
    lines = out["stdout"].split("\n")
    first = eval(lines[0])  # noqa: S307 - test-only literal
    assert first["checked"] == 2 and first["exact"] == 1 and first["first_mismatch"]["action"] == "UP"
    assert lines[1] == "1"
    found = eval(lines[2])  # noqa: S307
    assert found["reason"] == "found" and len(found["path"]) == 3
    assert full_box.hx_state["sim"]["exact"] == 1 and "first wrong: step 1 UP" in hx.sim_line(full_box.hx_state["sim"])


def test_notes_persist_and_render(full_box):
    game = AsciiGame(level=1)
    run(full_box, "note('goal', 'reach   the red cell')\nnote('tmp', 'x')\nnote('tmp')", game)
    assert full_box.hx_state["notes"] == {"goal": "reach the red cell"}
    assert hx.notes_line(full_box.hx_state["notes"]).endswith("goal: reach the red cell")
    run(full_box, "print(note('controls', 'UP adds a cell'))", game)
    assert list(full_box.hx_state["notes"]) == ["goal", "controls"]


async def test_phase_two_flags_in_the_agent(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20)), board((10, 10), (20, 20), (30, 30))]
    codes = ["note('goal', 'find it')\nregister_sim(lambda r, a: r)\nprint(autopilot(lambda f: 'UP', max_actions=1))",
             "action(['UP'])", "action(['UP'])"]
    agent, transport, result = await run_agent(tmp_path, boards, codes,
                                               {"autopilot": True, "worldmodel": True, "memo": True})
    assert result["error"] is None
    system = json.dumps(transport.payloads[0]["messages"][0]["content"])
    assert "autopilot(policy" in system and "check_sim(sim)" in system and "note(key, text)" in system
    second = texts(user_turns(transport.payloads[1])[-1])
    assert "Your notes (persist across turns" in second and "goal: find it" in second
    assert "Your registered sim predicts" in second
    stats = result["hx_harness"]
    assert stats["loops"] == 1 and stats["loop_actions"] == 1 and stats["notes_turns"] >= 1
    for before, after in zip(transport.payloads, transport.payloads[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]


def test_the_opener_points_at_explore_with_autopilot():
    assert "`explore(max_actions=12)`" in hx.opener_line(1, ["UP"], False, True)
    assert "explore" not in hx.opener_line(1, ["UP"], False)


def test_explore_ignores_a_timer_in_the_outer_ring():
    from duck_transfer.sandbox import StatefulSandbox

    class Timer(AsciiGame):
        def __call__(self, actions):
            self.grid = [row[:] for row in self.grid]
            self.grid[0][self.step % 8] = 3  # the timer ticks on every action
            return super().__call__(actions)

    original, plain, fixed = bootstraps()
    block = hx.settings(True)
    box = StatefulSandbox(hx.with_toolbox(dfr.with_dfr(plain, dfr.settings(True)), block),
                          color_chars=original["ARC_COLOR_CHARS"])
    try:
        game = Timer(level=1)
        out = run(box, "r = explore(max_actions=6)\nprint(r['known_states'], sorted(r['effects']))", game)
        assert not out["error"], out
        known = int(out["stdout"].split(" ")[0])
        assert known <= 4  # LEFT keeps the interior state; only UP creates states
    finally:
        box.close()


async def test_autoexplore_plays_probes_before_the_first_turn(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20)), board((10, 10), (20, 20), (30, 30)),
              board((10, 10), (20, 20), (30, 30), (40, 40))]
    agent, transport, result = await run_agent(tmp_path, boards, ["action(['UP'])", "action(['UP'])"],
                                               {"autoexplore": True, "autoexplore_actions": 2})
    assert result["error"] is None
    first = texts(user_turns(transport.payloads[0])[-1])
    assert "Harness exploration of level 1 before your first turn on it" in first
    assert first.count(": 1 cells changed") == 2  # two probes, each changed one cell
    assert result["hx_harness"]["autoexplore_actions"] == 2 and result["hx_harness"]["autoexplore_levels"] == 1
    assert all("Harness exploration" not in texts(user_turns(p)[-1]) for p in transport.payloads[1:])
    for before, after in zip(transport.payloads, transport.payloads[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]


def test_explore_candidates_keys_then_distinct_interior_objects():
    nodes = [{"color": 9, "hash": "a", "pixels": 4, "bbox": [10, 10, 12, 12]},
             {"color": 9, "hash": "a", "pixels": 4, "bbox": [20, 20, 22, 22]},
             {"color": 3, "hash": "b", "pixels": 9, "bbox": [0, 5, 2, 40]},   # HUD strip in the ring
             {"color": 2, "hash": "c", "pixels": 6, "bbox": [30, 30, 32, 33]},
             {"color": 0, "hash": "d", "pixels": 3000, "bbox": [0, 0, 64, 64]}]
    out = hx.explore_candidates(["UP", "RESET", "MOUSE", "UNDO"], nodes, 5)
    assert out == [{"action": "UP"}, {"action": "MOUSE", "row": 10, "col": 10},
                   {"action": "MOUSE", "row": 30, "col": 31}]


async def test_stuckexplore_plays_untried_actions_when_the_level_stalls(tmp_path):
    same = board()
    boards = [same] * 8
    agent, transport, result = await run_agent(
        tmp_path, boards, ["print(1)", "print(2)", "action(['UP'])", "print(3)", "print(4)", "print(5)"],
        {"stuckexplore": True, "reflect_stall": 2, "reflect_min_gap": 1, "stuck_explore_actions": 2})
    assert result["error"] is None
    turns = [texts(user_turns(p)[-1]) for p in transport.payloads]
    notes = [t for t in turns if "Harness, level 1 stalled" in t]
    assert notes and "not yet tried on this exact board" in notes[0]
    assert result["hx_harness"]["stuck_probes"] >= 1
    assert hx.label_of({"action": "MOUSE", "row": 3, "col": 4}) == "MOUSE(row=3, col=4)"


async def test_autoexplore_probes_do_not_trigger_the_stuck_notice(tmp_path):
    same = board()
    boards = [same] * 8  # every probe is a no-op
    agent, transport, result = await run_agent(
        tmp_path, boards, ["action(['UP'])", "print(1)"],
        {"autoexplore": True, "autoexplore_actions": 6, "reflect": True, "reflect_style": 2, "reflect_no_change": 3,
         "reflect_stall": 50, "reflect_min_gap": 1})
    assert result["error"] is None
    first = texts(user_turns(transport.payloads[0])[-1])
    assert "Harness exploration of level 1" in first and "STEP BACK" not in first


def test_memo_once_shows_notes_only_on_change_drain_or_level():
    from arena import hx_harness as hx

    assert hx.settings({"memo": True})["memo_once"] == 0
    assert hx.settings({"memo": True, "memo_once": 1})["memo_once"] == 1
    line, shown = hx.memo_turn("Your notes: a", (0, 0, 0), None)
    assert line == "Your notes: a"
    assert hx.memo_turn("Your notes: a", (0, 0, 0), shown) == (None, shown)       # unchanged: already in history
    assert hx.memo_turn("Your notes: b", (0, 0, 0), shown)[0] == "Your notes: b"  # text changed
    assert hx.memo_turn("Your notes: a", (1, 0, 0), shown)[0] == "Your notes: a"  # after a drain
    assert hx.memo_turn("Your notes: a", (0, 0, 1), shown)[0] == "Your notes: a"  # new level
    assert hx.memo_turn(None, (0, 0, 0), shown) == (None, shown)
