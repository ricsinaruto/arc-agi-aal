"""hero-v2 ``dfr_harness`` (arena.dfr_harness, arena.dfr_sandbox_lib): the ported dfranzen harness features.

Schema and hero-v2 recording; exact image token accounting; the diff image's pixels; the interior change rule (the
outer 4-cell ring); the batch no-op stop; the sandbox's stale-state and terminal latches (both request loops);
UNDO naming; frame_diff on synthetic frames with a rotation and a recolour; animation transient detection and the
lazy views; the vision, game-over, level and animation user turns; and the component off: unchanged payloads,
prompts and sandbox."""

import asyncio
import base64
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import observation, raw, tool_reply
from PIL import Image
from test_arena_duck import arena_agent

from arena import dfr_harness as dfr
from arena.dfr_sandbox_lib import compute_frame_diff
from arena.duck import ArenaTransferDuckAgent
from duck_transfer.trimming import image_tokens_for, png_dimensions, smart_resize, vision_tokens

ALL = dfr.settings(True)


def flags(*on):
    return dfr.settings({name: True for name in on})


def decode(url):
    return Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2]))).convert("RGB")


def board(*cells, value=9):
    grid = np.zeros((64, 64), dtype=np.uint8)
    for r, c in cells:
        grid[r, c] = value
    return grid


# --- schema and hero-v2 ------------------------------------------------------------------------------------------

def test_settings_schema():
    assert dfr.settings(None) is None and dfr.settings(False) is None and dfr.settings({}) is None
    assert ALL == {flag: flag not in dfr.OPT_IN for flag in dfr.FLAGS}  # sweep is opt-in
    assert dfr.settings({"guards": True}) == {**{f: False for f in dfr.FLAGS}, "guards": True}
    assert dfr.settings({"guards": False}) is None
    with pytest.raises(ValueError, match="Unknown dfr_harness flags"):
        dfr.settings({"guards": True, "undo": True})
    with pytest.raises(ValueError, match="booleans"):
        dfr.settings({"guards": 1})
    with pytest.raises(ValueError):
        dfr.settings("on")


def test_hero_v2_records_the_component_only_when_on():
    from test_competition_run_b import RUN_B, SERVED, hero

    from arena import hero_v2

    off = hero_v2.compose(hero(), {**RUN_B}, served_blocks=SERVED)
    explicit = hero_v2.compose(hero(), {**RUN_B, "dfr_harness": False}, served_blocks=SERVED)
    assert json.dumps(off, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert "dfr_harness" not in off and "dfr_harness" not in off["hero_v2"]["components"]
    on = hero_v2.compose(hero(), {**RUN_B, "dfr_harness": {"vision": True, "guards": True}}, served_blocks=SERVED)
    assert on["dfr_harness"] == {**{f: False for f in dfr.FLAGS}, "vision": True, "guards": True}
    record = on["hero_v2"]["components"]["dfr_harness"]
    assert record["on"] and record["upscale"] == 10 and record["border"] == 4 and record["guards_from_level"] == 2
    # Nothing else moves: the study differs from run B only by the component and its record.
    strip = {k: v for k, v in on.items() if k not in ("dfr_harness", "hero_v2", "arena_variant")}
    base = {k: v for k, v in off.items() if k not in ("hero_v2", "arena_variant")}
    assert json.dumps(strip, sort_keys=True) == json.dumps(base, sort_keys=True)
    with pytest.raises(ValueError, match="Unknown dfr_harness flags"):
        hero_v2.normalize({"dfr_harness": {"vision": True, "pictures": True}})
    assert hero_v2.status()["dfr_harness"]["available"]


# --- image token accounting ----------------------------------------------------------------------------------------

def test_vision_tokens_follow_the_served_processor():
    assert vision_tokens(256, 256) == 66        # MULTIMODAL_UPSCALE 4 (every earlier build)
    assert vision_tokens(640, 640) == 402       # dfr_harness vision (upscale 10)
    assert vision_tokens(1024, 1024) == 1026
    # smart_resize: rounded to multiples of 32 and scaled up to the 65,536-pixel minimum (the old ceil rule: 18).
    assert smart_resize(128, 128) == (256, 256) and vision_tokens(128, 128) == 66
    assert smart_resize(650, 650) == (640, 640)
    assert smart_resize(8192, 8192) == (4096, 4096)  # the 16,777,216-pixel maximum
    url = dfr.board_image_url(board((5, 5)))
    assert png_dimensions(url) == (640, 640) and image_tokens_for(dfr.image_part(url), 400) == 402
    diff = dfr.diff_image_url(board(), board((5, 5)))
    assert png_dimensions(diff) == (640, 640) and image_tokens_for(dfr.image_part(diff), 400) == 402


def test_the_trimmer_estimate_counts_every_larger_image():
    from duck_transfer.trimming import ObservationPreservingTrim

    url = dfr.board_image_url(board((5, 5)))
    message = {"role": "user", "content": [dfr.image_part(url),
                                           dfr.image_part(dfr.diff_image_url(board(), board((5, 5))))]}
    assert ObservationPreservingTrim()._message_tokens(message) - 804 < 10  # two images + the role field


async def test_count_prompt_adds_the_geometric_tokens_of_both_images(tmp_path):
    agent, *_ = arena_agent(tmp_path, None, ArenaTransferDuckAgent)
    url = dfr.board_image_url(board((5, 5)))
    message = {"role": "user", "content": [{"type": "text", "text": "x"}, dfr.image_part(url),
                                           dfr.image_part(dfr.diff_image_url(board(), board((5, 5))))]}
    count, images, _, image_tokens = await agent.count_prompt([message], None)
    assert images == 2 and image_tokens == 804  # (the fake /tokenize count is irrelevant here)
    agent.executor.shutdown()


# --- diff image pixels ---------------------------------------------------------------------------------------------

def test_diff_image_pixels():
    from inference.agent.vision_context import ARC_COLOR_MAP

    before = board((10, 20), value=8)
    after = before.copy()
    after[10, 20] = 14   # recolour
    after[30, 40] = 9    # appears
    image = decode(dfr.diff_image_url(before, after))
    assert image.size == (640, 640)
    assert image.getpixel((20 * 10 + 5, 10 * 10 + 5)) == ARC_COLOR_MAP[14]
    assert image.getpixel((40 * 10, 30 * 10)) == ARC_COLOR_MAP[9]
    assert image.getpixel((40 * 10 + 9, 30 * 10 + 9)) == ARC_COLOR_MAP[9]   # nearest-neighbour block
    assert image.getpixel((0, 0)) == dfr.DIFF_UNCHANGED_RGB
    assert image.getpixel((20 * 10 + 5, 11 * 10 + 5)) == dfr.DIFF_UNCHANGED_RGB
    assert dfr.diff_image_url(before, before) is None
    assert dfr.diff_image_url(before, np.zeros((32, 32), dtype=np.uint8)) is None
    board_image = decode(dfr.board_image_url(after))
    assert board_image.getpixel((40 * 10, 30 * 10)) == ARC_COLOR_MAP[9]
    assert board_image.getpixel((0, 0)) == ARC_COLOR_MAP[0]


def test_extra_images_are_stubbed_in_history_and_the_board_image_is_kept():
    board_part = dfr.image_part(dfr.board_image_url(board()))
    diff_part = dfr.image_part(dfr.diff_image_url(board(), board((1, 1))))
    message = {"role": "user", "content": [{"type": "text", "text": "Grid image of the board at step 1 (before this "
                                            "turn's actions):"}, board_part,
                                           {"type": "text", "text": "Diff image (step 0 -> 1): ..."}, diff_part,
                                           {"type": "text", "text": "Death frame image (the board ...)"}, board_part]}
    stubbed = dfr.stub_extra_images(message)
    kinds = [p["type"] for p in stubbed["content"]]
    assert kinds == ["text", "image_url", "text", "text", "text", "text"]
    assert stubbed["content"][1] is board_part and stubbed["content"][3]["text"] == dfr.HISTORY_IMAGE_PLACEHOLDER
    assert message["content"][3] is diff_part  # never mutated
    plain = {"role": "user", "content": [{"type": "text", "text": "x"}, board_part]}
    assert dfr.stub_extra_images(plain) is plain


# --- guards: the interior rule and the batch stop ------------------------------------------------------------------

@pytest.mark.parametrize("cell,changed", [((0, 0), False), ((3, 63), False), ((63, 3), False), ((60, 10), False),
                                          ((10, 60), False), ((4, 4), True), ((59, 59), True), ((32, 3), False),
                                          ((32, 4), True)])
def test_gameplay_changed_ignores_the_outer_four_cell_ring(cell, changed):
    assert dfr.gameplay_changed(board(), board(cell)) is changed


def test_gameplay_changed_fails_open():
    assert dfr.gameplay_changed(board(), np.zeros((32, 32), dtype=np.uint8)) is True
    small = np.zeros((8, 8), dtype=np.uint8)
    edge = small.copy()
    edge[0, 0] = 1
    assert dfr.gameplay_changed(small, edge) is True  # the ring swallows the grid: plain inequality
    assert dfr.gameplay_changed(board(), board((0, 0)), border=0) is True


def test_batch_stop_only_with_actions_left_after_an_interior_no_op():
    moved = {"executed": True, "gameplay_changed": True}
    still = {"executed": True, "gameplay_changed": False}
    requested = ["UP", "LEFT", "UP", "UP"]
    assert dfr.batch_stop(requested, 0, [("UP", moved)]) is None
    stop = dfr.batch_stop(requested, 1, [("UP", moved), ("LEFT", still)])
    assert stop["stop_reason"] == "no_op_action" and stop["no_op_action"] == "LEFT" and stop["no_op_action_index"] == 2
    assert stop["skipped_actions"] == ["UP", "UP"]
    assert stop["action_trace"] == ["1: UP -> effect", f"2: LEFT -> {dfr.NOOP_VERDICT}",
                                    "3: UP -> SKIPPED (never executed)", "4: UP -> SKIPPED (never executed)"]
    assert stop["stop_detail"].startswith("Action 2 of 4 ('LEFT') EXECUTED but produced no change inside the board")
    assert "only the first 1 executed action(s) changed the board" in stop["stop_detail"]
    assert dfr.batch_stop(["LEFT"], 0, [("LEFT", still)]) is None           # a single action is never stopped
    assert dfr.batch_stop(["UP", "LEFT"], 1, [("UP", moved), ("LEFT", still)]) is None  # nothing left to skip
    refused = dfr.batch_stop(requested, 1, [("UP", None), ("LEFT", still)])
    assert refused["action_trace"][0] == "1: UP -> REFUSED (no-change repeat; never executed)"
    notes = dfr.guard_notes([stop], [{"kind": "stale_state", "action": "UP", "skipped": ["UP", "UP"]}])
    assert notes[0].startswith("Note: a batched sequence was stopped by the no-op guard: action 'LEFT' (position 2")
    assert notes[1].startswith("Per-action trace: 1: UP -> effect;")
    assert notes[2].startswith("Note: the sequence was stopped before 'UP' - an earlier action in that snippet")


# --- the persistent sandbox: latches, frame_diff, animation --------------------------------------------------------

def bootstraps():
    from duck_transfer.history import preserve_transition_results
    from duck_transfer.loader import load
    from duck_transfer.sandbox import stateful_bootstrap

    original = load().run_sandboxed_python.__globals__
    public = preserve_transition_results(original["_SANDBOX_BOOTSTRAP"])
    return original, stateful_bootstrap(public), stateful_bootstrap(public, fixes=("modules",))


def make_sandbox(block, fixes):
    from duck_transfer.sandbox import StatefulSandbox

    original, plain, fixed = bootstraps()
    sandbox = StatefulSandbox(dfr.with_dfr(plain, block), color_chars=original["ARC_COLOR_CHARS"])
    if fixes:
        sandbox.configure_fixes(dfr.with_dfr(fixed, block), interrupt=True, grace=3)
    return sandbox


class Game:
    """UP changes the interior, LEFT changes nothing (executed), DOWN is GAME_OVER, RIGHT completes the level."""

    def __init__(self, level=2, grids=None):
        self.step, self.level = 0, level
        self.grids = list(grids or [])
        self.grid = self.grids.pop(0) if self.grids else [[0] * 8 for _ in range(8)]
        self.history = [{"action": "", "frame": self.frame()}]

    def frame(self):
        return {"ascii": "", "step": self.step, "level": self.level, "shape": [len(self.grid), len(self.grid[0])],
                "grid": [row[:] for row in self.grid]}

    def state(self):
        return {"current_frame": self.frame(), "history": list(self.history), "valid_actions": ["UP", "LEFT"],
                "last_action_result": {}}

    def __call__(self, actions):
        (item,) = actions
        name = item["action"]
        self.step += 1
        if self.grids:
            self.grid = self.grids.pop(0)
        result = {"executed": True, "action_display": name, "executed_actions": [name],
                  "board_changed": name != "LEFT", "gameplay_changed": name != "LEFT",
                  "level_completed": name == "RIGHT", "game_over": name == "DOWN", "done": False,
                  "run_complete": False, "state": "GAME_OVER" if name == "DOWN" else "NOT_FINISHED",
                  "level": self.level}
        self.history.append({"action": name, "frame": self.frame(), "result": result})
        return {"action_result": result, "state": self.state()}


def run(sandbox, code, game):
    return sandbox.run(code=code, timeout_seconds=5, initial_state=game.state(), action_handler=game)


@pytest.fixture(params=[True, False], ids=["fixes-loop", "plain-loop"])
def guarded(request):
    box = make_sandbox(flags("guards"), request.param)
    yield box
    box.close()


def test_the_stale_latch_refuses_the_next_action_from_level_two(guarded):
    game = Game(level=2)
    code = ("action(['LEFT'])\ntry:\n    action(['UP'])\nexcept StaleStateActionError as exc:\n"
            "    print('stale:', str(exc)[:40])\nprint(len(history))")
    out = run(guarded, code, game)
    assert not out["error"], out
    assert out["stdout"] == "stale: UP was not executed. The action immediat\n2\n" and game.step == 1
    assert guarded.dfr_guards[-1]["kind"] == "stale_state" and guarded.dfr_guards[-1]["previous"] == "LEFT"
    # Uncaught, it ends the snippet as an ordinary error; a new snippet starts clear.
    out = run(guarded, "action(['LEFT'])\naction(['UP'])\nprint('after')", game)
    assert "StaleStateActionError" in out["error"] and game.step == 2 and "after" not in out["stdout"]
    out = run(guarded, "action(['UP'])\naction(['LEFT'])\nprint('ok')", game)
    assert not out["error"] and out["stdout"] == "ok\n" and game.step == 4
    # A change clears it; the latch arms only on an executed interior no-op.
    out = run(guarded, "action(['UP'])\naction(['UP'])", game)
    assert not out["error"] and game.step == 6
    assert guarded.restarts == 0


def test_the_stale_latch_does_not_refuse_on_level_one(guarded):
    game = Game(level=1)
    out = run(guarded, "action(['LEFT'])\naction(['UP'])\nprint('both')", game)
    assert not out["error"] and out["stdout"] == "both\n" and game.step == 2


@pytest.mark.parametrize("terminal", ["DOWN", "RIGHT"])
def test_the_terminal_latch_cannot_be_swallowed(guarded, terminal):
    game = Game(level=1)
    code = f"action(['{terminal}'])\ntry:\n    action(['UP'])\nexcept Exception:\n    print('swallowed')\nprint('x')"
    out = run(guarded, code, game)
    assert "TerminalStateActionError" in out["error"] and "swallowed" not in out["stdout"]
    reported = "game_over" if terminal == "DOWN" else "level_completed"
    assert f"the previous action reported {reported}" in out["error"] and game.step == 1
    assert guarded.restarts == 0 and guarded.dfr_guards[-1] == {"kind": "terminal", "after": reported, "action": "UP"}
    out = run(guarded, "action(['UP'])\nprint('fresh')", game)  # the next snippet starts clear
    assert not out["error"] and out["stdout"] == "fresh\n" and game.step == 2


def test_the_inventory_skips_the_new_runtime_names():
    sandbox = make_sandbox(ALL, True)
    try:
        run(sandbox, "def helper(x):\n    return x\n", Game())
        inventory = sandbox.inventory()
        names = ([f[0] for f in inventory["functions"]] + inventory["modules"] + inventory["classes"]
                 + [v[0] for v in inventory["values"]])
        assert "helper" in names
        for name in ("frame_diff", "last_animation_frames", "last_animation_timeline", "StaleStateActionError",
                     "TerminalStateActionError"):
            assert name not in names
    finally:
        sandbox.close()


def shape_grid(cells_by_colour, size=48):
    grid = [[0] * size for _ in range(size)]
    for colour, cells in cells_by_colour.items():
        for r, c in cells:
            grid[r][c] = colour
    return grid


L_SHAPE = [(10, 10), (11, 10), (12, 10), (12, 11)]
L_ROTATED = [(10, 10), (10, 11), (10, 12), (11, 10)]  # the same L, 90 degrees clockwise, in place
SQUARE = [(30, 30), (30, 31), (31, 30), (31, 31)]


@pytest.mark.parametrize("fixes", [True, False], ids=["fixes-loop", "plain-loop"])
def test_frame_diff_reports_a_rotation_and_a_recolour(fixes):
    before = shape_grid({9: L_SHAPE, 8: SQUARE})
    after = shape_grid({9: L_ROTATED, 14: SQUARE})
    sandbox = make_sandbox(flags("frame_diff"), fixes)
    try:
        game = Game(level=1, grids=[before, after])
        code = ("import json\naction(['UP'])\nd = frame_diff()\nnodes = current_frame.segmentation['nodes']\n"
                "l = [n for n in nodes if n['color'] == 'b'][0]\n"
                "print(json.dumps({'d': d, 'l': [l['rotation'], l['rotational_symmetry'], l['pose_hash'] != l['hash']],"
                " 'same': frame_diff(history[-1].frame, current_frame)['changed_cell_count']}))")
        out = run(sandbox, code, game)
        assert not out["error"], out
        data = json.loads(out["stdout"])
        diff = data["d"]
        assert diff["changed_cell_count"] == 8
        assert [(e["color"], e["rotated_by"], e["at"]) for e in diff["rotated"]] == [("b", 90, [10, 10])]
        assert [(e["before_color"], e["after_color"], e["at"]) for e in diff["changed_color"]] == [("R", "N", [30, 30])]
        assert not diff["moved"] and not [e for e in diff["appeared"] + diff["disappeared"] if e["color"] != "W"]
        assert data["l"][1] == 1 and data["l"][0] in (0, 90, 180, 270) and data["same"] == 0
    finally:
        sandbox.close()


def test_frame_diff_on_the_host_matches_moves():
    from arena.dfr_sandbox_lib import _object_signature

    def nodes(cells, colour):
        canonical, rotation, symmetry, pose, shape = _object_signature(cells, colour)
        return [{"color": colour, "hash": canonical, "rotation": rotation, "rotational_symmetry": symmetry,
                 "pose_hash": pose, "shape_hash": shape, "pixels": len(cells),
                 "boundary": [[r, c] for r, c in cells]}]

    moved = [(r + 5, c + 7) for r, c in L_SHAPE]
    before, after = shape_grid({9: L_SHAPE}), shape_grid({9: moved})
    diff = compute_frame_diff(before, after, nodes(L_SHAPE, "b"), nodes(moved, "b"))
    assert diff["moved"] == [{"color": "b", "pixels": 4, "hash": nodes(L_SHAPE, "b")[0]["hash"], "from": [10, 10],
                              "to": [15, 17]}]
    # The canonical hash ignores rotation; the pose hash does not; the rotation is the same for any colour.
    a, b = _object_signature(L_SHAPE, "b"), _object_signature(L_ROTATED, "b")
    assert a[0] == b[0] and a[3] != b[3] and (b[1] - a[1]) % 360 == 90
    assert _object_signature(L_SHAPE, "R")[1] == a[1]
    assert _object_signature(SQUARE, "R")[1:3] == (None, 4)


@pytest.mark.parametrize("fixes", [True, False], ids=["fixes-loop", "plain-loop"])
def test_the_animation_views_fetch_from_the_host_lazily(fixes):
    sandbox = make_sandbox(flags("animation"), fixes)
    requests = []
    record = {"action_display": "SPACE", "step": 1, "level": 1,
              "chain": [((0, 0), (0, 0)), ((0, 5), (0, 0)), ((0, 0), (0, 0))]}

    def handler(request):
        requests.append(request["kind"])
        if request["kind"] == "frames":
            return dfr.animation_frames(record, lambda grid, step, level: {
                "ascii": "", "step": step, "level": level, "shape": [2, 2], "grid": [list(r) for r in grid]})
        return dfr.animation_timeline(record)

    sandbox.animation_handler = handler
    try:
        game = Game(level=1)
        out = run(sandbox, "print(len(last_animation_frames), last_animation_frames[0].shape, "
                           "last_animation_timeline['frames'], len(last_animation_frames))", game)
        assert not out["error"], out
        assert out["stdout"] == "2 (2, 2) 2 2\n" and requests == ["frames", "timeline"]  # cached within a snippet
        out = run(sandbox, "action(['UP'])\nprint(last_animation_timeline['steps'][0]['changes'])", game)
        assert out["stdout"] == "['W>B @ (0,1)']\n" and requests == ["frames", "timeline", "timeline"]
    finally:
        sandbox.close()


def test_the_bootstrap_patch_is_pinned_and_installed_once():
    _, plain, fixed = bootstraps()
    from duck_transfer.sandbox import with_plans

    for text in (plain, fixed, with_plans(plain), with_plans(fixed)):
        patched = dfr.with_dfr(text, ALL)
        compile(patched, "<bootstrap>", "exec")
        assert patched.count('"dfr_guards": _dfr_take(),') == 2 and patched.count("_dfr_begin()\n") == 1
        assert '"rotation": signature[1],' in patched
        with pytest.raises(ValueError):
            dfr.with_dfr(patched, ALL)
    assert '"rotation": signature[1],' not in dfr.with_dfr(plain, flags("guards"))
    planned = with_plans(dfr.with_dfr(fixed, ALL))  # either order with the action plans
    compile(planned, "<bootstrap>", "exec")


# --- animation helpers ---------------------------------------------------------------------------------------------

def test_transient_detection_and_the_opener():
    a, b, c = ((0, 0, 0),), ((0, 7, 0),), ((0, 0, 3),)
    assert dfr.collapse_frames(a, [b, b, a]) == [a, b, a]
    assert dfr.transient_cells([a, b, a]) == [(0, 1)]
    assert dfr.transient_cells([a, b, c]) == [(0, 1)]  # (0,1) shown then reverted; (0,2) changed once
    assert dfr.summarize_animation([a, c]) is None      # one frame: no key at all
    assert dfr.summarize_animation(dfr.collapse_frames(a, [b, c])) == {"frames": 2, "transient_pixels": 1,
                                                                        "transient_bbox": [0, 1, 0, 1]}
    motion = dfr.summarize_animation(dfr.collapse_frames(a, [((0, 7, 0),), ((0, 7, 3),)]))
    assert motion == {"frames": 2, "transient_pixels": 0}
    assert dfr.describe_animation(motion, action_display="UP", gameplay_changed=True) == ""
    summary = dfr.summarize_animation([a, b, a])
    line = dfr.describe_animation(summary, action_display="SPACE", gameplay_changed=False)
    assert line.startswith("The last executed action (`SPACE`) animated over 2 frames and left the board area "
                           "unchanged, but it was not a no-op: 1 cell changed and changed back at (0,1).")
    death = dfr.describe_animation(summary, action_display="SPACE", gameplay_changed=True, game_over=True)
    assert death.startswith("That fatal action animated over 2 frames") and "`last_animation_timeline`" in death
    timeline = dfr.animation_timeline({"action_display": "SPACE", "chain": [a, b, a]})
    assert timeline == {"action": "SPACE", "frames": 2, "steps": [
        {"step": 1, "changed": 1, "bbox": "rows 0-0, cols 1-1", "changes": ["W>P @ (0,1)"]},
        {"step": 2, "changed": 1, "bbox": "rows 0-0, cols 1-1", "changes": ["P>W @ (0,1)"]}]}
    assert "error" in dfr.animation_timeline(None) and dfr.animation_frames(None, None) == []


# --- the agent: prompts and payloads -------------------------------------------------------------------------------

class Transport:
    def __init__(self, *codes):
        self.codes, self.payloads = list(codes), []

    async def post(self, url, **kwargs):
        self.payloads.append(copy.deepcopy(kwargs["json"]))  # the analyzer keeps extending its message list
        code = self.codes.pop(0) if self.codes else "action(['UP'])"
        return tool_reply(code, call_id=f"c{len(self.payloads)}",
                          usage={"completion_tokens": 50, "prompt_tokens": 7000})


def user_turns(payload):
    return [m for m in payload["messages"] if m["role"] == "user" and isinstance(m.get("content"), list)]


def images(message):
    return [p["image_url"]["url"] for p in message["content"] if p.get("type") == "image_url"]


def texts(message):
    return "\n".join(p["text"] for p in message["content"] if p.get("type") == "text")


async def test_game_over_turn_with_the_audit_fixes_has_one_death_text(tmp_path):
    from arena import harness_fixes as hf

    start, death, restart = board(), board((10, 10), (0, 5)), board()
    obs = [observation(start), observation(death, state="GAME_OVER"), observation(restart)]
    transport = Transport("action(['UP'])", "action(['UP'])")
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_harness_fixes([*hf.FIXES, "solvable_line", "death_turn_v2", "findings_keep"])
    agent.apply_dfr_harness(flags("vision", "animation"))
    env.observations = iter(obs[1:] + [observation(board((7, 7))), observation(board((7, 7), (8, 8)))])
    env.observation_space = raw(obs[0])
    agent.config.max_actions = 4
    result = await agent.run()
    assert result["error"] is None
    assert hf.SOLVABLE_LINE in transport.payloads[0]["messages"][0]["content"]
    text = texts(user_turns(transport.payloads[1])[-1])
    assert text.count("GAME_OVER at step 1 after UP: the attempt on level 1 ended") == 1
    assert "The attempt lasted 1 actions." in text and "step-budget bar" in text
    for old in ("BAR RULE", "GAME OVER occurred", "Do NOT re-submit", "The game over occurred immediately", "avoid that sequence",
                "Recent findings: GAME_OVER"):
        assert old not in text, old
    assert hf.DEATH_IMAGE_LABEL in text and dfr.FATAL_DIFF_LABEL in text and len(images(user_turns(transport.payloads[1])[-1])) == 3
    assert "GAME_OVER" not in texts(user_turns(transport.payloads[2])[-1])  # one turn only
    assert result["dfr_harness"]["gameover_turns"] == 1 and result["harness_fixes"]["death_texts"] == 1
    assert agent.death_log and agent.death_log[0]["actions"] == 1


async def run_agent(tmp_path, boards, codes, block=None, *, max_actions=None, fixes=True, levels=None, states=None,
                    reset_board=None):
    transport = Transport(*codes)
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    if fixes:
        agent.apply_harness_fixes(True)
    agent.apply_dfr_harness(block)
    obs = [observation(b, levels=(levels or {}).get(i, 0), state=(states or {}).get(i, "NOT_FINISHED"))
           if isinstance(b, np.ndarray) else b for i, b in enumerate(boards)]
    if reset_board is not None:
        obs.append(observation(reset_board))
    env.observations = iter(obs[1:])
    env.observation_space = raw(obs[0])
    agent.config.max_actions = max_actions or len(boards) - 1
    result = await agent.run()
    return agent, transport, env, result


async def test_component_off_keeps_the_payloads_prompts_and_sandbox(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    codes = ["action(['UP'])", "action(['UP'])"]
    control, ct, _, cr = await run_agent(tmp_path / "control", boards, codes, None)
    off, ot, _, orr = await run_agent(tmp_path / "off", boards, codes, False)
    assert cr["error"] is None and orr["error"] is None
    assert json.dumps(ct.payloads, sort_keys=True) == json.dumps(ot.payloads, sort_keys=True)
    assert control.sandbox.bootstrap == off.sandbox.bootstrap and off.dfr is None
    assert "dfr_harness" not in orr and off.sandbox.animation_handler is None
    # Prompt-only flags leave the sandbox untouched.
    agent, *_ = arena_agent(tmp_path / "prompt", None, ArenaTransferDuckAgent)
    before = agent.sandbox.bootstrap
    agent.apply_dfr_harness({"actions": True, "level_prompt": True, "vision": True})
    assert agent.sandbox.bootstrap == before
    agent.executor.shutdown()


async def test_each_flag_adds_its_system_text(tmp_path):
    def prompt(block):
        agent, *_ = arena_agent(tmp_path / str(len(seen)), None, ArenaTransferDuckAgent)
        agent.apply_harness_fixes(True)
        agent.apply_dfr_harness(block)
        seen.append(agent)
        text = agent.analyzer(asyncio.get_running_loop())._system_prompt
        agent.executor.shutdown()
        return text

    seen = []
    base = prompt(None)
    assert prompt(False) == base
    checks = {"vision": "Use it to locate change at a glance.", "guards": "`gameplay_changed` ignores those",
              "actions": "- When available, `UNDO` reverses a previous action", "level_prompt":
              "Levels usually build on mechanics learned in earlier levels", "frame_diff":
              "- `frame_diff(before=None, after=None)` compares two frames.", "animation":
              "- `last_animation_timeline` is a diff timeline"}
    for flag, needle in checks.items():
        text = prompt(flags(flag))
        assert needle in text and needle not in base, flag
        for other, other_needle in checks.items():
            if other != flag and not (flag == "animation" and other == "guards"):
                assert other_needle not in text, (flag, other)
    full = prompt(True)
    assert dfr.MOUSE_LINE not in full and dfr.MOUSE_LINE in base
    assert "\n\n\nAction meanings (use only actions currently available):\n" in full
    assert "- `WIN` means the whole game is solved.\n" in full and "Strategies may transfer loosely" not in full
    assert "ignores position AND rotation" in full


async def test_vision_sends_the_larger_board_and_a_diff_image_kept_in_the_append_only_history(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10)), board((10, 10), (20, 20))]
    agent, transport, _, result = await run_agent(tmp_path, boards, ["action(['UP'])"] * 3, flags("vision"))
    assert result["error"] is None and result["actions"] == 3
    first, second, third = (user_turns(p)[-1] for p in transport.payloads[:3])
    assert [png_dimensions(u) for u in images(first)] == [(640, 640)]  # no previous turn: no diff
    assert len(images(second)) == 2 and "Diff image (step 0 -> 1)" in texts(second)
    diff = decode(images(second)[1])
    assert diff.getpixel((10 * 10 + 5, 10 * 10 + 5)) == (30, 147, 255) and diff.getpixel((5, 5)) == dfr.DIFF_UNCHANGED_RGB
    assert len(images(third)) == 1  # the board did not change: no diff image
    # between drains the history is append-only, so the next request repeats the earlier turn byte
    # for byte (board and diff image) and the prefix cache can reuse it; nothing is stubbed without a drain.
    earlier = [m for m in user_turns(transport.payloads[2]) if "Diff image (step 0 -> 1)" in texts(m)]
    assert len(earlier) == 1 and earlier[0] == second and dfr.HISTORY_IMAGE_PLACEHOLDER not in texts(earlier[0])
    for before, after in zip(transport.payloads, transport.payloads[1:]):
        assert after["messages"][:len(before["messages"])] == before["messages"]
    stats = result["dfr_harness"]
    assert stats["diff_images"] == 1 and stats["history_images_stubbed"] == 0


async def test_game_over_turn_carries_the_bar_rule_and_the_death_images(tmp_path):
    start, death, restart = board(), board((10, 10), (0, 5)), board()
    obs = [observation(start), observation(death, state="GAME_OVER"), observation(restart)]
    transport = Transport("action(['UP'])", "print(history[-1].action, history[-2].action)\naction(['UP'])")
    agent, env, game, gate = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_harness_fixes(True)
    agent.apply_dfr_harness(flags("vision", "animation"))
    env.observations = iter(obs[1:] + [observation(board((7, 7))), observation(board((7, 7), (8, 8)))])
    env.observation_space = raw(obs[0])
    agent.config.max_actions = 4
    result = await agent.run()
    assert result["error"] is None
    turn = user_turns(transport.payloads[1])[-1]
    text = texts(turn)
    assert "BAR RULE: the bar shrinking with each action is NORMAL" in text
    assert "The game over occurred immediately after action 'UP'." in text and "Do NOT re-submit" in text
    assert text.index("GAME_OVER at step 1") < text.index("GAME OVER occurred") < text.index("Do NOT re-submit")
    assert "Diff image" not in text and dfr.DEATH_IMAGE_LABEL in text and dfr.FATAL_DIFF_LABEL in text
    urls = images(turn)
    assert len(urls) == 3 and all(png_dimensions(u) == (640, 640) for u in urls)
    from inference.agent.vision_context import ARC_COLOR_MAP

    assert decode(urls[1]).getpixel((10 * 10, 10 * 10)) == ARC_COLOR_MAP[9]     # the death frame
    assert decode(urls[2]).getpixel((5 * 10, 0)) == ARC_COLOR_MAP[9]            # the fatal step's diff
    tool = [m for m in transport.payloads[2]["messages"] if m["role"] == "tool"][-1]
    assert "RESET UP" in tool["content"]
    later = user_turns(transport.payloads[2])[-1]
    assert "BAR RULE" not in texts(later)  # once per reset
    assert result["dfr_harness"]["gameover_turns"] == 1 and result["dfr_harness"]["gameover_images"] == 2


async def test_the_batch_stop_and_its_next_turn_note(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (0, 0)), board((10, 10), (0, 0), (20, 20))]
    codes = ["r = action(['UP', 'UP', 'UP'])\nprint(r['stop_reason'], r['executed_count'], r['gameplay_changed'])",
             "action(['UP'])"]
    agent, transport, env, result = await run_agent(tmp_path, boards, codes, flags("guards"))
    assert result["error"] is None and len(env.calls) == 3
    tool = [m for m in transport.payloads[1]["messages"] if m["role"] == "tool"][-1]
    content = json.loads(tool["content"])
    assert content["stdout"] == "no_op_action 2 False\n"
    text = texts(user_turns(transport.payloads[1])[-1])
    assert "Note: a batched sequence was stopped by the no-op guard: action 'UP' (position 2 in that batch)" in text
    assert "Per-action trace: 1: UP -> effect; 2: UP -> NO-OP (executed, no change inside the board area); 3: UP -> " \
           "SKIPPED (never executed)." in text
    receipts = [agent.transition_results[s] for s in (1, 2)]
    assert [r["gameplay_changed"] for r in receipts] == [True, False]
    assert result["dfr_harness"]["noop_stops"] == 1


async def test_undo_is_advertised_and_executes_action7(tmp_path):
    seven = SimpleNamespace(**{**vars(raw(observation())), "available_actions": [1, 4, 6, 7]})
    transport = Transport("print(valid_actions)\nr = action(['UNDO'])\nprint(r['executed_actions'])")
    agent, env, *_ = arena_agent(tmp_path, transport, ArenaTransferDuckAgent)
    agent.apply_dfr_harness(flags("actions"))
    after = SimpleNamespace(**{**vars(raw(observation(board((10, 10))))), "available_actions": [1, 4, 6, 7]})
    env.observation_space = seven
    env.observations = iter([observation(board((10, 10)))])
    env.step = lambda action, data: env.calls.append((action.name, data)) or after
    agent.config.max_actions = 1
    result = await agent.run()
    assert result["error"] is None and env.calls[0][0] == "ACTION7"
    text = texts(user_turns(transport.payloads[0])[-1])
    assert "UNDO" in text and "ACTION7" not in text
    assert agent.valid_actions() == ["UP", "RIGHT", "MOUSE", "UNDO"] and agent.engine_action("ACTION7") == "ACTION7"
    assert agent.model_action("ACTION7") == "UNDO"
    control, *_ = arena_agent(tmp_path / "control", None, ArenaTransferDuckAgent)
    control.obs = observation()
    control.obs.available = ["ACTION1", "ACTION7"]
    assert control.valid_actions() == ["UP", "ACTION7"] and control.engine_action("UNDO") is None
    control.executor.shutdown()


async def test_the_level_prompt_replaces_the_public_line(tmp_path):
    boards = [board(), board((10, 10)), board((10, 10), (20, 20))]
    agent, transport, _, result = await run_agent(tmp_path, boards, ["action(['UP'])"] * 2, flags("level_prompt"),
                                                  levels={1: 1, 2: 1}, max_actions=2)
    assert result["error"] is None
    text = texts(user_turns(transport.payloads[1])[-1])
    assert dfr.LEVEL_START_USER_PROMPT in text and dfr.PUBLIC_LEVEL_LINE not in text
    assert (dfr.CROSS_LEVEL_CARRIED in text) or (dfr.CROSS_LEVEL_EMPTY in text)
    assert result["dfr_harness"]["level_prompts"] == 1


async def test_an_animation_that_reverts_gets_its_receipt_and_opener(tmp_path):
    start = board()
    mid = board((10, 10))
    animated = observation(start)
    animated.frames = [mid, start.copy()]  # cell (10,10) shown and reverted; the board ends unchanged
    boards = [observation(start), animated, observation(board((20, 20)))]
    codes = ["r = action(['UP'])\nprint(r['animation'], r['gameplay_changed'])",
             "print(last_animation_timeline['frames'], len(last_animation_frames))\naction(['UP'])"]
    agent, transport, env, result = await run_agent(tmp_path, boards, codes, flags("animation"))
    assert result["error"] is None
    tool = json.loads([m for m in transport.payloads[1]["messages"] if m["role"] == "tool"][-1]["content"])
    assert tool["stdout"] == "{'frames': 2, 'transient_pixels': 1, 'transient_bbox': [10, 10, 10, 10]} False\n"
    text = texts(user_turns(transport.payloads[1])[-1])
    assert ("The last executed action (`UP`) animated over 2 frames and left the board area unchanged, but it was "
            "not a no-op: 1 cell changed and changed back at (10,10).") in text
    tool = json.loads([m for m in transport.payloads[2]["messages"] if m["role"] == "tool"][-1]["content"])
    assert tool["stdout"] == "2 2\n"
    stats = result["dfr_harness"]
    assert stats["animated_actions"] == 1 and stats["transient_actions"] == 1 and stats["animation_reads"] == 2


async def test_animation_history_keeps_every_animated_action(tmp_path):
    start = board()
    first = observation(board((5, 5)))
    first.frames = [board((10, 10)), board((5, 5))]            # step 1: two frames
    second = observation(board((5, 5), (6, 6)))
    second.frames = [board((5, 5), (30, 30)), board((5, 5), (31, 31)), board((5, 5), (6, 6))]  # step 2: three frames
    plain = observation(board((7, 7)))                         # step 3: a single frame
    boards = [observation(start), first, second, plain, observation(board((8, 8))), observation(board((9, 9)))]
    codes = ["action(['UP'])", "action(['UP'])", "action(['UP'])",
             "print(animated_steps(), len(animation_frames(1)), len(animation_frames(history[-2])), "
             "len(animation_frames(transitions[-1])), animation_timeline(2)['frames'], len(last_animation_frames))\n"
             "print(animation_frames(1)[0].ascii == history[1].frame.ascii, animation_frames(1)[0].shape)\n"
             "action(['UP'])", "action(['UP'])"]
    agent, transport, env, result = await run_agent(tmp_path, boards, codes, flags("animation", "animation_history"))
    assert result["error"] is None
    assert dfr.ANIMATION_HISTORY_ADDENDUM in transport.payloads[0]["messages"][0]["content"]
    tool = json.loads([m for m in transport.payloads[4]["messages"] if m["role"] == "tool"][-1]["content"])
    # steps 1 and 2 animated; history[-2] is step 2's entry, transitions[-1] the single-frame step 3
    assert tool["stdout"].splitlines()[0] == "[1, 2] 2 3 0 3 0", tool
    assert tool["stdout"].splitlines()[1] == "False (64, 64)", tool
    assert result["dfr_harness"]["animation_history_reads"] == 7
    store = agent.animation_store
    assert store.steps() == [1, 2] and 0 < store.bytes < 2000 and store.dropped == 0
    small = dfr.AnimationStore(limit=1)
    for step in (1, 2, 3):
        small.put({"chain": [start.tolist(), board((1, 1)).tolist(), start.tolist()], "step": step, "action_display": "UP"})
    assert small.steps() == [3] and small.dropped == 2 and small.get(1) is None and len(small.get(3)["chain"]) == 3
    with pytest.raises(ValueError, match="needs animation"):
        dfr.settings({"animation_history": True})
    assert "animation_frames(x)" not in (await run_agent(tmp_path / "off", [observation(start), observation(board((1, 1)))],
                                                         ["action(['UP'])"], flags("animation")))[1].payloads[0]["messages"][0]["content"]


async def test_validation_rejects_incompatible_agents(tmp_path):
    from arena.duck import ArenaPublicDuckAgent

    public, *_ = arena_agent(tmp_path / "public", None, ArenaPublicDuckAgent)
    with pytest.raises(ValueError, match="corrected Duck agent"):
        public.apply_dfr_harness(True)
    public.executor.shutdown()


async def test_attach_wires_the_study_block(tmp_path):
    """arena.trial passes ``study["dfr_harness"]`` to ``attach`` (as for harness_fixes and action_plans)."""
    import inspect

    from arena import trial

    assert 'dfr_harness=self.study.get("dfr_harness") or None' in inspect.getsource(trial)
    agent, env, game, gate = arena_agent(tmp_path, None, ArenaTransferDuckAgent)
    agent.attach(game, window=agent.window, cap=agent.cap, dfr_harness={"guards": True})
    assert agent.dfr == flags("guards") and "_DFR_FLAGS = " in agent.sandbox.bootstrap
    agent.executor.shutdown()


def test_a_drain_stubs_the_surviving_turns_extra_images(tmp_path):
    """extra images stay in the append-only history until an eviction; the drain (which rewrites
    the prefix anyway) turns the kept turns' diff images into the placeholder and leaves board images alone."""
    agent, _, game, _ = arena_agent(tmp_path, None, ArenaTransferDuckAgent, window=65536)
    agent.apply_dfr_harness(flags("vision"))
    agent.apply_context_drain({"drain_tokens": 1000})
    game.note_context_drain = lambda **kw: None
    board_part, diff_part = dfr.image_part("data:image/png;base64,AAAA"), dfr.image_part("data:image/png;base64,BBBB")

    def turn(i):
        label = dfr.DIFF_LABEL.format(before=i, after=i + 1)
        user = {"role": "user", "content": [{"type": "text", "text": f"obs {i} " + "x" * 4000}, board_part,
                                            {"type": "text", "text": label}, diff_part]}
        return [user, {"role": "assistant", "content": "r" * 4000}]

    history = [m for i in range(12) for m in turn(i)] + turn(12)[:1]

    class Analyzer:
        @staticmethod
        def _estimate_request_input_tokens(messages, tools=None):
            return sum(len(json.dumps(m)) // 4 for m in messages)

    def measure(messages, tools):
        return Analyzer._estimate_request_input_tokens(messages)

    messages = [{"role": "system", "content": "s"}, *history]
    budget = measure(messages, None) - 3000
    out = agent.drain_context(Analyzer(), messages, None, measure, budget)
    kept_users = [m for m in out if m.get("role") == "user" and isinstance(m.get("content"), list)]
    assert 2 <= len(kept_users) < 13 and measure(out, None) <= budget
    assert out[-1] == history[-1]  # the current observation keeps its diff image
    for m in kept_users[:-1]:
        kinds = [p.get("type") for p in m["content"]]
        assert kinds.count("image_url") == 1 and m["content"][1] == board_part  # board kept, diff stubbed
        assert any(p.get("text") == dfr.HISTORY_IMAGE_PLACEHOLDER for p in m["content"])
    assert agent.dfr_stats["history_images_stubbed"] == len(kept_users) - 1
    # Without a drain the same history is returned untouched.
    assert agent.drain_context(Analyzer(), messages, None, measure, measure(messages, None) + 1) is messages


class ClickGame(Game):
    """Clicks change nothing (executed no-ops) and are displayed with their cell; UP changes the interior."""

    def __call__(self, actions):
        (item,) = actions
        result = super().__call__([{"action": "LEFT" if item["action"] == "MOUSE" else item["action"]}])
        if item["action"] == "MOUSE":
            shown = "MOUSE(row=%s, col=%s)" % (item["row"], item["col"])
            result["action_result"].update(action_display=shown, executed_actions=[shown])
        return result


def test_sweep_lets_a_click_on_another_cell_follow_a_no_op_click_and_nothing_else():
    click = "action([{'action': 'MOUSE', 'row': %d, 'col': %d}])\n"
    box = make_sandbox(flags("guards", "sweep"), True)
    try:
        game = ClickGame(level=2)
        out = run(box, click % (1, 1) + click % (2, 2) + click % (3, 3) + "print('swept')", game)
        assert not out["error"] and out["stdout"] == "swept\n" and game.step == 3, out
        assert [g["kind"] for g in box.dfr_guards] == ["sweep_exempt", "sweep_exempt"]
        out = run(box, click % (1, 1) + click % (1, 1), game)  # the same cell again stays refused
        assert "StaleStateActionError" in out["error"] and game.step == 4
        out = run(box, click % (1, 1) + "action(['UP'])", game)  # a key after a no-op click stays refused
        assert "StaleStateActionError" in out["error"] and game.step == 5
        out = run(box, "action(['LEFT'])\n" + click % (1, 1), game)  # a click after a no-op key stays refused
        assert "StaleStateActionError" in out["error"] and game.step == 6
    finally:
        box.close()
    box = make_sandbox(flags("guards"), True)
    try:
        game = ClickGame(level=2)
        out = run(box, click % (1, 1) + click % (2, 2), game)
        assert "StaleStateActionError" in out["error"] and game.step == 1
    finally:
        box.close()
    assert dfr.settings(True)["sweep"] is False and dfr.settings({"guards": True, "sweep": True})["sweep"] is True
    labels = ["MOUSE(row=1, col=1)", "MOUSE(row=2, col=2)", "UP"]
    noop = ("MOUSE(row=1, col=1)", {"gameplay_changed": False})
    assert dfr.batch_stop(labels, 0, [noop]) is not None and dfr.batch_stop(labels, 0, [noop], sweep=True) is None
    assert dfr.batch_stop(labels, 1, [noop, ("MOUSE(row=2, col=2)", {"gameplay_changed": False})], sweep=True)


# --- hx4: transient effects (analysis 8a) and the RESET offer (8c) ---------------------------------------------------

# b2-v71-24h-v1 clone k012 (cd82, DEV family), actions 1-67: the last click plays a 15-frame animation and ends on the
# board it started from; v7.1 reported it as a NO-OP.
TRANSIENT_GAME = "cd82-fb555c5d"
TRANSIENT_PREFIX = (
    "ACTION1 ACTION5 ACTION2 ACTION3 ACTION4 ACTION3 ACTION5 ACTION6@30,40 ACTION6@36,4 ACTION6@43,7 "
    "ACTION6@7,10 ACTION3 ACTION3 ACTION4 ACTION4 ACTION4 ACTION6@41,29 ACTION1 ACTION2 ACTION1 ACTION1 "
    "ACTION2 ACTION2 ACTION2 ACTION3 ACTION4 ACTION3 ACTION6@43,4 ACTION5 ACTION3 ACTION3 ACTION3 ACTION1 "
    "ACTION1 ACTION1 ACTION4 ACTION6@36,4 ACTION5 ACTION6@47,4 ACTION4 ACTION5 ACTION6@41,4 ACTION2 "
    "ACTION2 ACTION5 ACTION6@41,4 ACTION4 ACTION3 ACTION3 ACTION1 ACTION1 ACTION4 ACTION5 ACTION6@35,4 "
    "ACTION4 ACTION2 ACTION2 ACTION3 ACTION5 ACTION6@47,4 ACTION4 ACTION5 ACTION3 ACTION4 ACTION5 ACTION2 "
    "ACTION6@31,21 "
).split()
_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]  # a worktree sits three levels below the checkout
ENVIRONMENTS = next((p for p in (_ROOT / "environment_files", _ROOT.parents[2] / "environment_files") if p.is_dir()),
                    None)


@pytest.mark.skipif(ENVIRONMENTS is None, reason="public environments unavailable")
def test_a_recorded_animation_that_ends_on_its_starting_board_is_a_transient_effect(tmp_path):
    from arcengine import GameAction

    from arc_infra.arcade import make_arcade
    from arena.level_points import replay

    arc = make_arcade("offline", ENVIRONMENTS, tmp_path)
    env = arc.make(TRANSIENT_GAME, scorecard_id=arc.open_scorecard(tags=["test"]), save_recording=False)
    _, before = replay(env, TRANSIENT_PREFIX[:-1])
    name, _, where = TRANSIENT_PREFIX[-1].partition("@")
    x, y = (int(v) for v in where.split(","))
    raw = env.step(GameAction[name], {"x": x, "y": y})
    frames = [np.array(f, dtype=np.uint8) for f in raw.frame]
    (start,) = dfr.normalize_frames([before])
    summary = dfr.summarize_animation(dfr.collapse_frames(start, dfr.normalize_frames(frames)))
    assert len(frames) == 15 and summary["transient_pixels"] == 98
    assert dfr.gameplay_changed(before, frames[-1]) is False
    plain = {"gameplay_changed": False, "animation": summary}
    assert dfr.verdict(plain) == dfr.NOOP_VERDICT  # v7.1
    assert dfr.verdict({**plain, "transient_effect": 98}).startswith("TRANSIENT EFFECT (animated: 98 cell(s)")
    note = dfr.transient_note([{"step": 67, "label": "MOUSE(row=21, col=31)", "animation": summary}])
    assert "step 67 MOUSE(row=21, col=31): 98 cells" in note and "NOT no-ops" in note and dfr.transient_note([]) == ""


def test_transient_and_reset_offer_are_opt_in_and_reword_the_refusal():
    assert dfr.settings(True)["transient"] is False and dfr.settings(True)["reset_offer"] is False
    box = make_sandbox(flags("guards", "transient"), True)
    try:
        class Bump(Game):
            def __call__(self, actions):
                result = super().__call__(actions)
                if actions[0]["action"] == "LEFT":
                    result["action_result"]["transient_effect"] = 5
                return result

        out = run(box, "action(['LEFT'])\naction(['UP'])", Bump(level=2))
        assert "played an animation and left the board area exactly as it was" in out["error"]
        assert "was a no-op" not in out["error"]
    finally:
        box.close()
    _, plain, _ = bootstraps()
    assert "_DFR_STALE_TRANSIENT" in dfr.with_dfr(plain, flags("guards"))


def test_transient_note_in_the_user_turn_has_its_own_switch():
    from types import SimpleNamespace

    from arena.duck import ArenaDuckMixin

    assert dfr.settings(True)["transient_note"] is False
    with pytest.raises(ValueError):
        dfr.settings({"transient_note": True})
    item = {"step": 7, "label": "MOUSE(row=2, col=3)", "animation": {"frames": 3, "transient_pixels": 5}}

    def turn(**on):
        owner = SimpleNamespace(
            _analyzer=None, dfr=dfr.settings({"guards": True, "transient": True, **on}), auto_reset=None,
            dfr_last_action=None, dfr_opener_step=None, dfr_transients=[dict(item)], dfr_stats={}, dfr_noop_stops=[],
            dfr_guard_stops=lambda: [], dfr_gameover_images=None)
        return ArenaDuckMixin.dfr_rewrite(owner, "Executed actions (1): MOUSE\nBoard:", None), owner

    text, owner = turn()
    assert "NOT no-ops" not in text and owner.dfr_transients == [] and "transient_notes" not in owner.dfr_stats
    text, owner = turn(transient_note=True)
    assert "step 7 MOUSE(row=2, col=3): 5 cells" in text and "replay one alone" in text
    assert owner.dfr_stats["transient_notes"] == 1
    # The per-action verdict does not depend on the note.
    assert dfr.verdict({"executed": True, "gameplay_changed": False, "transient_effect": 5}).startswith("TRANSIENT EFFECT")


def test_border_revisit_tells_a_moving_edge_object_from_a_timer_bar():
    assert dfr.settings(True)["border_moves"] is False
    board = np.zeros((16, 16), dtype=np.uint8)
    seen = None
    for cell in range(5):  # a bar shrinking along the bottom row: each cell changes once
        after = board.copy()
        after[15, cell] = 3
        revisit, seen = dfr.border_revisit(board, after, seen)
        assert revisit is False and dfr.gameplay_changed(board, after) is False
        board = after
    board = np.zeros((16, 16), dtype=np.uint8)
    board[1, 2] = 7  # an object on the edge ring moving right: it leaves the cell it entered
    seen = None
    moved = board.copy()
    moved[1, 2], moved[1, 3] = 0, 7
    revisit, seen = dfr.border_revisit(board, moved, seen)
    assert revisit is False  # the first move looks like a bar step
    again = moved.copy()
    again[1, 3], again[1, 4] = 0, 7
    revisit, seen = dfr.border_revisit(moved, again, seen)
    assert revisit is True and dfr.gameplay_changed(moved, again) is False
    inner = again.copy()
    inner[8, 8] = 5  # interior changes never count as edge revisits
    assert dfr.border_revisit(again, inner, None)[0] is False
