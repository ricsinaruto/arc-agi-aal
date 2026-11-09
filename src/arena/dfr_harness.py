"""hero-v2 ``dfr_harness``: harness features ported from dfranzen's ARC-AGI-3 Milestone 2 notebook.

Ported from dfranzen/arc-agi-3-milestone-2-solution (Apache-2.0), a patch over the Tufa Labs Duck harness (the same
public ToolAgent as ``src/inference``); the prompt texts, the diff image, the interior ``gameplay_changed`` rule, the
batch no-op stop, the snippet latches, the frame diff, the rotation-aware segmentation and the animation helpers are
theirs, adapted to the corrected Duck agent (``duck_transfer``) under the arena. One opt-in study key ``dfr_harness``
(``settings``): None on every earlier build, so their payloads and prompts stay byte for byte; true switches every
sub-flag on, a dict of booleans selects some (``FLAGS``):

- ``vision``: the current board image at ``UPSCALE`` (their ``MULTIMODAL_UPSCALE=10``: 640 x 640 px, 400 + 2 vision
  tokens instead of 64 + 2) with a diff image after it (the cells changed since the previous analyzer turn in their
  new colour, every other cell an off-palette navy; one more 402-token image when the board changed) and one system
  sentence; on the first turn after a GAME_OVER, their diagnosis text (the BAR RULE) and two images instead of the
  diff: the death frame and the fatal step's diff. As theirs, the history keeps every image between evictions (append-
  only: stubbing the previous turn's extra images on every request broke the hybrid prefix cache, 3 October); at a
  drain the surviving turns' extra images (diff, death frame, fatal diff) become a one-line placeholder (theirs keep
  them until the turn is trimmed). The board image itself keeps the arena's existing policy.
- ``guards``: ``gameplay_changed`` in every receipt (a change inside the board, ignoring the outer ``BORDER`` cells
  where timer bars sit) and their system line explaining it; a batch of two or more actions stops after an executed
  action with no interior change (their stop text and per-action trace); in the persistent sandbox, from level
  ``GUARDS_FROM_LEVEL`` on, the next ``action()`` of a snippet after such a no-op raises the catchable
  ``StaleStateActionError`` without executing, and after GAME_OVER, a completed level or the win any further
  ``action()`` of the snippet raises ``TerminalStateActionError`` (a BaseException, so ``except Exception`` cannot
  swallow it).
- ``actions``: ACTION7 is advertised as ``UNDO`` (accepted as UNDO or ACTION7; executes ACTION7 as before) and the
  system prompt carries their short per-action explanations in place of the MOUSE coordinate line.
- ``level_prompt``: their level-transfer system guidance and new-level user prompt (build a new plan for this layout,
  inspect unfamiliar elements, reassess the goal), with one sentence pointing at the carried cross-level notes.
- ``frame_diff``: ``frame_diff(before=None, after=None)`` in the sandbox (``arena.dfr_sandbox_lib``: Hungarian
  matching into moved / rotated / appeared / disappeared / changed_color / resized), rotation-canonical segmentation
  hashes with ``rotation``, ``rotational_symmetry``, ``pose_hash`` and ``shape_hash``, and their prompt paragraph.
- ``animation``: an executed action that returned several frames carries ``animation`` (frames, transient pixels and
  their bbox: cells that changed and changed back) in its receipt; ``last_animation_frames`` (frame views with
  segmentation) and ``last_animation_timeline`` (their diff timeline) in the sandbox, fetched from the host on use;
  their opener line when the last action animated, and their system paragraph.
"""

import base64
import io
import re
from collections import defaultdict

import numpy as np

FLAGS = ("vision", "guards", "actions", "level_prompt", "frame_diff", "animation", "sweep", "transient", "reset_offer",
         "border_moves", "simcheck", "evidence", "transient_note", "animation_history", "simcarry")
# sweep (hx4, 7 October; never on with ``True``): after a no-op CLICK, a click on a DIFFERENT cell is executed (a click
# sweep is a set of independent probes). Everything else after a no-op stays refused: the same click again, any key
# after a no-op, any click after a no-op key. v7.1 DEV traces: 97 of 597 stale-state refusals were such clicks.
# transient (hx4; v7.1 24 h analysis 8a): an action whose animation ends on the starting board is reported as a
# transient effect, not as a no-op, for EVERY action of a python call (the animation line covered only the last one).
# transient_note (9 October, user; needs transient): the next user turn also lists those actions and asks the model to
# replay one alone. Its own switch since then: ``transient`` alone is the per-action verdict and the refusal wording.
# reset_offer (hx4; analysis 8c): RESET is listed among the valid actions and explained in one system line (it was
# accepted but never offered: 0 manual resets in 68,471 actions of the v7.1 24 h run).
# border_moves (hx4; analysis 8b): a change confined to the outer ring is a no-op only while it looks like a timer bar,
# i.e. it touches cells that have not changed before in this level attempt. A change that touches an edge cell again
# (an object moving along the edge) counts as a gameplay change (v7.1 counted every edge-only move as a no-op).
# simcheck (hx5, 8 October; arena.sim_check): after a stuck signal the harness checks the agent's own model of the
# level against the recorded transitions and runs action lists under that check. evidence: after the same signal, one
# line per turn with the level's recorded counts per action (tries / changed / no-op, single-board no-ops flagged).
# animation_history (v8.2, 9 October, user; needs animation): the frames of EVERY animated action stay readable,
# not only the last one's: ``animation_frames(x)`` / ``animation_timeline(x)`` (x: a step number, a history entry or a
# transition) and ``animated_steps()``. The host keeps each chain zlib-compressed, ANIMATION_STORE_BYTES per game,
# oldest dropped first.
# simcarry (10 October, the bet; arena.sim_check): at each clear the model writes encode / step / win for the level
# just won, the harness scores it on that level's full record (actions held until it fits or an allowance is spent),
# carries it, checks it on every action of the following levels and offers its own search of it. Not with simcheck.
OPT_IN = ("sweep", "transient", "reset_offer", "border_moves", "simcheck", "evidence", "transient_note",
          "animation_history", "simcarry")
ANIMATION_STORE_BYTES = 24 * 1024 * 1024
UPSCALE = 10            # their MULTIMODAL_UPSCALE
BORDER = 4              # their ARC3_NOOP_GUARD_BORDER: the outer ring ignored by gameplay_changed
GUARDS_FROM_LEVEL = 2   # their ARC3_GUARDS_FROM_LEVEL: the stale-state latch starts on level 2
UNDO = ("ACTION7", "UNDO")
# Their off-palette neutral for unchanged cells in diff images (ARC indices 0-5 are a grey ramp).
DIFF_UNCHANGED_RGB = (16, 24, 48)
HISTORY_IMAGE_PLACEHOLDER = "[grid image omitted from history]"
ANIMATION_MAX_CELLS_PER_STEP = 24
ANIMATION_MAX_TOTAL_CELLS = 80


def settings(value):
    """The study block for ``value`` (None/False: off; True: every flag; a dict of booleans over ``FLAGS``, missing
    flags off), or None when no flag is on."""
    if value is None or value is False:
        return None
    if value is True:
        return {flag: flag not in OPT_IN for flag in FLAGS}
    if not isinstance(value, dict):
        raise ValueError("dfr_harness is true/false or a dict of booleans over " + ", ".join(FLAGS))
    unknown = sorted(set(value) - set(FLAGS))
    if unknown:
        raise ValueError(f"Unknown dfr_harness flags {unknown}; known: {list(FLAGS)}")
    bad = sorted(k for k, v in value.items() if type(v) is not bool)
    if bad:
        raise ValueError(f"dfr_harness flags are booleans; got {bad}")
    block = {flag: bool(value.get(flag, False)) for flag in FLAGS}
    if block["transient_note"] and not block["transient"]:
        raise ValueError("dfr_harness transient_note needs transient")
    if block["animation_history"] and not block["animation"]:
        raise ValueError("dfr_harness animation_history needs animation")
    if block["simcarry"] and block["simcheck"]:
        raise ValueError("dfr_harness simcarry replaces simcheck (one model mode at a time)")
    return block if any(block.values()) else None


def describe(block):
    """The hero-v2 record detail of an on block."""
    return {"flags": dict(block), "upscale": UPSCALE if block["vision"] else None, "border": BORDER,
            "guards_from_level": GUARDS_FROM_LEVEL, "source": "dfranzen/arc-agi-3-milestone-2-solution (Apache-2.0)",
            "module": "arena.dfr_harness"}


# --- prompt texts (theirs, renamed to our runtime variables) --------------------------------------------------------

LEVEL_TRANSFER_SYSTEM_GUIDANCE = (
    "- Levels usually build on mechanics learned in earlier levels, especially the most "
    "recent one. Carry forward supported knowledge as a starting hypothesis, while "
    "re-checking anything contradicted by new evidence. New levels often introduce "
    "additional mechanics, sometimes through unfamiliar board elements. These additions "
    "are often important for solving the level. The goal may remain the same but require "
    "new mechanics to reach it, or the goal itself may change.\n"
)
LEVEL_TRANSFER_REPLACEMENTS = {
    "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n":
        LEVEL_TRANSFER_SYSTEM_GUIDANCE,
    "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board "
    "before repeating a plan.\n": "",
    "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already "
    "be the next level.\n": "",
    "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase "
    "while play continues.\n": "- `WIN` means the whole game is solved.\n",
    "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately "
    "and re-ground on the next turn.\n":
        "- If an action result reports `game_over`, `run_complete`, or `done`, stop acting immediately and re-ground "
        "on the next turn.\n",
}
LEVEL_START_USER_PROMPT = (
    "You have completed the previous level. `current_frame` now contains the starting "
    "board of the next level; any accompanying current-grid image shows this new board. "
    "Build a new plan for this layout rather than continuing the previous level's action "
    "sequence.\n\n"
    "Start from the mechanics you established on the previous level; do not rediscover "
    "them without reason. Inspect the new board for unfamiliar elements, changed "
    "arrangements, or interactions your previous understanding does not explain. New "
    "elements often introduce mechanics needed to solve this level, so prioritize small, "
    "informative tests when their behavior is unclear.\n\n"
    "Reassess the goal: does the previous objective still apply, now requiring the new "
    "mechanics, or does the evidence suggest a different objective? Combine retained "
    "knowledge with new findings to plan for this board."
)
PUBLIC_LEVEL_LINE = "You have progressed to a new level!"
# Ours (integration with duck_transfer's cross-level notes): one sentence on the carried notes.
CROSS_LEVEL_CARRIED = (
    " The cross-level notes in the carried world model below are what you recorded about earlier levels: treat "
    "them as starting hypotheses for this board and update `Cross-level notes:` with what still holds."
)
CROSS_LEVEL_EMPTY = (
    " Once you have checked which mechanics still hold on this board, record them in `Cross-level notes:`."
)

ACTION_INFO_ADDENDUM = (
    "\n\nAction meanings (use only actions currently available):\n"
    "- `UP`, `DOWN`, `LEFT`, and `RIGHT` are directional controls; what they affect "
    "depends on the game.\n"
    "- When available, `SPACE` performs a game-specific action, such as interacting, "
    "selecting, rotating, attaching/detaching, or executing. Test its effect rather "
    "than assuming what it does.\n"
    "- When available, `MOUSE` clicks a board location. Pass integer `row` and `col` "
    "fields from 0 to 63. Coordinates are zero-based from the top-left: `row` "
    "increases downward and `col` increases rightward.\n"
)
UNDO_INFO_ADDENDUM = (
    "- When available, `UNDO` reverses a previous action, usually the last turn. "
    "Check what it restores. It cannot recover a failed attempt after game over.\n"
)
MOUSE_LINE = "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"

GAMEPLAY_CHANGED_ADDENDUM = (
    "- `board_changed` is true when ANY cell differs, including cells within {border} of the "
    "edge where a timer or remaining-steps bar usually sits; `gameplay_changed` ignores those "
    "edge cells and is true only when a cell further in differs. An action that only shrinks "
    "a timer bar in those edge cells sets `board_changed` but not `gameplay_changed`.\n"
).format(border=f"{BORDER} cells")

ANIMATION_ADDENDUM = (
    "- One action can return a short animation. `current_frame` is its final frame.\n"
    "- When the final executed action of the latest `action(...)` call animated, `last_action_result['animation']` "
    "reports `frames`, `transient_pixels` and `transient_bbox`. Transient cells "
    "changed and then changed BACK during the animation, so they appear in no frame "
    "you can otherwise reach - not in `current_frame`, not in `previous_frame`, not "
    "in `history`.\n"
    "- `last_animation_frames` is the frames of the last executed action that "
    "animated, as views with "
    "`.ascii`, `.shape` and `.segmentation`, the same as `current_frame`. Index 0 is "
    "the first frame of the animation and the last entry is the board it settled on. "
    "After a death these are the frames of the FATAL action, not of the automatic "
    "reset that followed it. Crop "
    "their `.ascii` yourself; printing a whole 64x64 frame will not fit the tool "
    "budget. Empty when the action did not animate, and reading them costs no "
    "in-game action.\n"
)
ANIMATION_ADDENDUM_TIMELINE = (
    "- `last_animation_timeline` is a diff timeline of those frames: it "
    "shows which cells changed at each step. It is a dict with `action`, `frames` and "
    "`steps`, one entry per step that changed anything. Each entry has `step`, "
    "`changed`, `bbox`, and "
    "either `changes` (the cells, as `old>new @ (row,col) ...`) or `transitions` (a "
    "count per colour change, when there were too many cells to list).\n"
)
ANIMATION_HISTORY_ADDENDUM = (
    "- The animation of every earlier action is kept as well: `animation_frames(x)` and `animation_timeline(x)` "
    "return what `last_animation_frames` and `last_animation_timeline` held right after that action; `x` is a step "
    "number, a `history` entry or a transition (`animation_frames(transitions[-3])`). `animated_steps()` lists the "
    "steps whose action animated. An action that returned a single frame has no animation (an empty list); its "
    "boards are in `history`. Reading them costs no in-game action.\n"
)

RESET_OFFER_ADDENDUM = (
    "- `RESET` restarts the current level from its first board and costs one action. Use it when the level is in a "
    "state you cannot recover from (pieces stuck, wrong irreversible move); do not spend the remaining step budget "
    "to force a restart. It does nothing as the first action of a level.\n"
)
TRANSIENT_VERDICT = "TRANSIENT EFFECT (animated: {cells} cell(s) changed and changed back; final board unchanged)"


def transient_note(items):
    """One user-turn note for the executed actions of the last sequence that animated and ended on the board they
    started from (``items``: ``{"step", "label", "animation"}``), or "" when there is none."""
    if not items:
        return ""
    shown = "; ".join(f"step {i['step']} {i['label']}: {int(i['animation'].get('transient_pixels') or 0)} cells"
                      f"{_bbox_phrase(i['animation'].get('transient_bbox'))}" for i in items[:6])
    more = f" (+{len(items) - 6} more)" if len(items) > 6 else ""
    return (f"These actions left the board area unchanged but were NOT no-ops: each played an animation in which cells "
            f"changed and changed back: {shown}{more}. Do not conclude that they do nothing; replay one alone and "
            "read `last_animation_timeline`.")


DIFF_IMAGE_SYSTEM = (
    " Alongside the current grid image you receive a diff image: cells that "
    "changed since your previous analyzer turn are shown in their new true "
    "color; unchanged cells are dimmed to a dark navy that is not a game "
    "color. Use it to locate change at a glance."
)

SEGMENTATION_NODE_LINE = (
    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered "
    "top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that "
    "ignores its position -- equal hashes mean the same object regardless of where it is, so use it to track an "
    "object across frames or to spot multiple identical objects in one frame), `pixels` (cell count), `boundary` "
    "(clockwise outer-perimeter corner points as `[row, col]`), and `children` (ids of objects fully enclosed by "
    "this one).\n"
)
SEGMENTATION_NODE_LINE_ROTATION = (
    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered "
    "top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that "
    "ignores position AND rotation -- equal hashes indicate matching color and shape up to rotation; they do not "
    "establish object identity), `rotation` (degrees clockwise from the object's canonical orientation: "
    "0/90/180/270, reduced to 0/90 for 2-fold symmetric shapes, `None` for fully symmetric ones), "
    "`rotational_symmetry` (1, 2, or 4), `pose_hash` (the rotation-SENSITIVE variant, if you need to match exact "
    "orientation), `shape_hash` (color-independent pose shape signature), `pixels` (cell count), `boundary` "
    "(clockwise outer-perimeter corner points as `[row, col]` cell coordinates), and `children` (ids of objects "
    "fully enclosed by this one).\n"
)
FRAME_DIFF_HINT_ADDENDUM = (
    "- `frame_diff(before=None, after=None)` compares two frames. Omitted arguments default to `previous_frame` and "
    "`current_frame`, respectively.\n"
    "- It returns a dictionary containing `changed_cell_count` (an integer) and lists named `moved`, `rotated`, "
    "`appeared`, `disappeared`, `changed_color`, and `resized`. Unchanged objects are omitted.\n"
    "- Object-level changes are inferred by matching same-color, 4-connected components between frames: cells "
    "connect through shared edges, not diagonally. A game object may therefore be split into several components, "
    "or touching same-color objects may form one component.\n"
    "- `moved` describes components matched at different positions. Matching does not establish persistent object "
    "identity: with multiple similar components, the reported correspondence may be ambiguous or incorrect. "
    "Movement combined with rotation includes `rotated_by` in degrees clockwise; `rotated` describes rotation in "
    "place.\n"
    "- `changed_color` requires identical occupied cells with a different color. Entries contain `at`, `pixels`, "
    "`before_color`, `after_color`, `before_hash`, and `after_hash`.\n"
    "- `resized` describes same-color components with overlapping footprints whose size changed. This can include "
    "backgrounds and budget bars. `at_before` and `at_after` can differ.\n"
    "- `appeared` and `disappeared` entries include `bbox` in `[r0, c0, r1, c1]` order. Object hashes use the same "
    "rotation-invariant shape-and-color representation as segmentation; they are not unique object identities, and "
    "multiple objects can share a hash.\n"
    "- Examples: `frame_diff()` compares the latest action's before/after frames; "
    "`frame_diff(history[-3].frame, history[-2].frame)` compares two explicitly selected historical frames, when "
    "available.\n"
)

# Anchors in the corrected Duck system prompt (each must occur exactly once).
MULTIMODAL_ANCHOR = "\n\n\nMultimodal context:\n"
VISUAL_ANCHOR = "\n\n\nVisual-game guidance:\n"
PYTHON_ANCHOR = "\n\n\nPython tool guidance:\n"
SESSION_ANCHOR = "\n\n\nTool session rules:\n"


def _insert_before(text, anchor, addition, what):
    if text.count(anchor) != 1:
        raise ValueError(f"dfr_harness: pinned system prompt anchor for {what} changed")
    return text.replace(anchor, addition + anchor, 1)


def _replace_once(text, old, new, what):
    if text.count(old) != 1:
        raise ValueError(f"dfr_harness: pinned system prompt text for {what} changed")
    return text.replace(old, new, 1)


def system_prompt(text, block):
    """The corrected Duck system prompt with the on flags' additions (in their prompt's order)."""
    if block["frame_diff"]:
        text = _replace_once(text, SEGMENTATION_NODE_LINE, SEGMENTATION_NODE_LINE_ROTATION, "frame_diff")
    runtime = ""
    if block["guards"] or block["animation"]:
        runtime += GAMEPLAY_CHANGED_ADDENDUM
    if block["animation"]:
        runtime += ANIMATION_ADDENDUM + ANIMATION_ADDENDUM_TIMELINE
    if block.get("animation_history"):
        runtime += ANIMATION_HISTORY_ADDENDUM
    if block.get("reset_offer"):
        runtime += RESET_OFFER_ADDENDUM
    if runtime:
        # The runtime-variable section ends "...are refreshed.\n" before the blank lines of the next section.
        text = _insert_before(text, MULTIMODAL_ANCHOR, "\n" + runtime.rstrip("\n"), "runtime variables")
    if block["vision"]:
        text = _insert_before(text, VISUAL_ANCHOR, DIFF_IMAGE_SYSTEM, "vision")
    if block["actions"]:
        text = _replace_once(text, MOUSE_LINE, "", "actions")
        text = _insert_before(text, PYTHON_ANCHOR, "\n" + ACTION_INFO_ADDENDUM + UNDO_INFO_ADDENDUM.rstrip("\n"),
                              "actions")
    if block["frame_diff"]:
        text = _insert_before(text, SESSION_ANCHOR, "\n" + FRAME_DIFF_HINT_ADDENDUM.rstrip("\n"), "frame_diff")
    if block["level_prompt"]:
        for old, new in LEVEL_TRANSFER_REPLACEMENTS.items():
            text = _replace_once(text, old, new, "level_prompt")
    return text


# --- GAME_OVER diagnosis (vision) ------------------------------------------------------------------------------------

GAME_OVER_DIAGNOSIS = (
    "GAME OVER occurred during the previous sequence. This means the "
    "attempt FAILED (it does NOT mean the run is finished). The game has "
    "already been automatically RESET: the current level restarted from "
    "its initial state and previously completed levels are kept. A game "
    "over is caused either by the specific action taken (e.g., moving "
    "onto a hazard cell) or by a step/time budget running out, which is "
    "usually indicated by a bar at the grid border that shrinks with each "
    "action. Before acting again, diagnose the cause by inspecting the "
    "board right before the death. `history[-1].action` is 'RESET', "
    "`history[-2].action` is the fatal action, `history[-2].frame` is the "
    "death frame (the board after the fatal action), and "
    "`history[-3].frame` is the board immediately before "
    "the fatal action — check what cell the fatal move targeted there, and "
    "inspect the budget bar in the death frame. BAR RULE: the bar "
    "shrinking with each action is NORMAL — a slightly shorter bar is "
    "NOT evidence of the cause of death. The budget caused the death "
    "ONLY if the bar is fully (or almost fully) depleted in the death "
    "frame (`history[-2].frame`). If it IS depleted: you must reach the "
    "goal in fewer actions, or look for a way to restore the bar (e.g., "
    "collecting an item that refills it). If the bar clearly had budget "
    "remaining: the fatal action itself caused the death (hazard or "
    "forbidden move) — do NOT retry the same path; a DIFFERENT approach "
    "must be tried. Record the diagnosed cause in your world model."
)
FATAL_ACTION_LINE = (
    "The game over occurred immediately after action {action!r}. "
    "Apply the BAR RULE above to decide: bar fully depleted in the "
    "death frame means budget death (shorter route or bar restore "
    "needed); bar with budget remaining means this action itself was "
    "fatal and a DIFFERENT approach is required."
)
NO_REPLAY_LINE = (
    "Do NOT re-submit this same series of moves: the game is "
    "deterministic, so replaying the sequence that just killed you "
    "will kill you again. Change the plan BEFORE the fatal point - a "
    "different route, action, or timing. If you believe the same "
    "moves 'should' work, that belief is exactly what this death "
    "falsified; record the correction in your world model instead "
    "of retesting it."
)
DEATH_IMAGE_LABEL = (
    "Death frame image (the board at the moment the attempt ended, before the automatic reset): inspect the budget "
    "bar here for the BAR RULE."
)
FATAL_DIFF_LABEL = (
    "Fatal step diff image (last alive frame -> death frame): cells changed by the killing action in their new true "
    "color; unchanged cells dimmed to dark navy (an off-palette color)."
)
DIFF_LABEL = (
    "Diff image (step {before} -> {after}): cells that changed since your previous analyzer turn in their new true "
    "color; unchanged cells dimmed to dark navy (an off-palette color)."
)
EXTRA_IMAGE_LABELS = ("Diff image (step ", "Death frame image (", "Fatal step diff image (")


# --- guards: interior change, batch stop, opener notes ---------------------------------------------------------------

def gameplay_changed(before, after, border=BORDER):
    """True when any cell outside the outer ``border`` ring differs (their ``_interior_gameplay_changed``); fails open
    (plain inequality) when the border is 0 or swallows the grid, and True when the shapes differ."""
    a, b = np.asarray(before), np.asarray(after)
    if border <= 0:
        return bool(a.shape != b.shape or (a != b).any())
    if a.shape != b.shape or a.ndim != 2:
        return True
    rows, cols = a.shape
    if rows <= 2 * border or cols <= 2 * border:
        return bool((a != b).any())
    return bool((a[border:rows - border, border:cols - border] != b[border:rows - border, border:cols - border]).any())


def border_revisit(before, after, seen, border=BORDER):
    """``(revisit, seen)`` for an action without an interior change: ``revisit`` is True when an edge-ring cell that
    already changed earlier in this level attempt (``seen``, a boolean grid or None) changed again; ``seen`` is
    returned with this action's edge changes added. A shrinking timer bar changes each cell once; a moving object
    leaves the cells it entered."""
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.ndim != 2:
        return False, None
    diff = a != b
    rows, cols = a.shape
    if border > 0 and rows > 2 * border and cols > 2 * border:
        diff = diff.copy()
        diff[border:rows - border, border:cols - border] = False
    if seen is None or seen.shape != diff.shape:
        seen = np.zeros(diff.shape, dtype=bool)
    return bool((diff & seen).any()), seen | diff


NOOP_VERDICT = "NO-OP (executed, no change inside the board area)"


def verdict(payload):
    """Their one-phrase account of an executed action (``_action_verdict``, new wording)."""
    if payload.get("game_over"):
        return "GAME OVER (attempt ended immediately after this action)"
    if payload.get("run_complete") or payload.get("done"):
        return "RUN COMPLETE"
    if payload.get("level_completed"):
        return "LEVEL COMPLETED"
    changed = payload.get("gameplay_changed")
    if changed is False and payload.get("transient_effect"):
        return TRANSIENT_VERDICT.format(cells=payload["transient_effect"])
    if changed is False:
        return NOOP_VERDICT
    return "effect" if changed is True else "executed"


def click_sweep(previous, following):
    """True when ``following`` is a click on another cell than the click ``previous`` (labels as displayed)."""
    return str(previous).startswith("MOUSE(") and str(following).startswith("MOUSE(") and previous != following


def batch_stop(requested, position, verdicts, sweep=False):
    """The batch no-op stop after the action at ``position`` (0-based in ``requested``, the requested labels), or
    None. ``verdicts`` is one ``(label, payload or None)`` per action tried so far (None: refused, never executed).
    Fires only when a batch of two or more has actions left after an executed action without an interior change."""
    label, payload = verdicts[-1]
    total = len(requested)
    if total < 2 or payload is None or payload.get("gameplay_changed") is not False or position >= total - 1:
        return None
    if sweep and click_sweep(requested[position], requested[position + 1]):
        return None
    executed = sum(1 for _, item in verdicts if item is not None)
    index = position + 1
    skipped = list(requested[index:])
    trace = []
    for number, (name, item) in enumerate(verdicts, 1):
        trace.append(f"{number}: {name} -> " + (verdict(item) if item is not None
                                                else "REFUSED (no-change repeat; never executed)"))
    trace.extend(f"{number}: {name} -> SKIPPED (never executed)" for number, name in enumerate(skipped, index + 1))
    detail = (
        f"Action {index} of {total} ({label!r}) EXECUTED but produced no change inside the board area - only the "
        f"outer {BORDER} cells, where a timer or remaining-steps bar usually sits, could have moved. "
        f"Actions {index + 1}-{total} ({skipped!r}) were SKIPPED and did not execute. Indices are 1-based positions "
        f"in your submitted list. Net effect: only the first {max(0, executed - 1)} executed action(s) changed the "
        "board; the final executed action moved nothing. Do NOT update your position estimate by counting submitted "
        "moves - re-locate the player from current_frame before planning. The per-action verdicts are listed in "
        "action_trace. This stop is about one action from one board state, not about batch size - batching is "
        "still the right approach, and the stop is what keeps a wrong movement model from costing the whole sequence."
    )
    return {"stop_reason": "no_op_action", "stop_detail": detail, "no_op_action": label, "no_op_action_index": index,
            "skipped_actions": skipped, "action_trace": trace}


def noop_stop_note(stop):
    """Their next-turn note for one batch no-op stop."""
    skipped = stop.get("skipped_actions") or []
    skipped_text = f" The remaining {len(skipped)} action(s) {skipped!r} were skipped." if skipped else ""
    index = int(stop.get("no_op_action_index") or 1)
    lines = [
        f"Note: a batched sequence was stopped by the no-op guard: action {stop.get('no_op_action')!r} (position "
        f"{index} in that batch) executed but produced no change inside the board area.{skipped_text} That action IS "
        f"included in the executed list above, so it is the LAST of them - only the first {max(0, index - 1)} "
        "executed action(s) moved anything. Do not count submitted or executed actions to update your position; "
        "re-locate from current_frame."
    ]
    if stop.get("action_trace"):
        lines.append("Per-action trace: " + "; ".join(str(t) for t in stop["action_trace"]) + ".")
    return lines


def stale_stop_note(stop):
    """Their next-turn note for a stale-state refusal (``kind`` stale_state)."""
    skipped = stop.get("skipped") or []
    skipped_text = f" The remaining {len(skipped)} action(s) ({skipped!r}) were never executed." if skipped else ""
    return (f"Note: the sequence was stopped before {stop.get('action')!r} - an earlier action in that snippet produced "
            "no change inside the board area, so the positions and paths it had computed may have been wrong."
            f"{skipped_text} Re-ground on the current board before planning again.")


def guard_notes(noop_stops, guard_stops):
    lines = []
    for stop in noop_stops[:3]:
        lines.extend(noop_stop_note(stop))
    if len(noop_stops) > 3:
        lines.append(f"({len(noop_stops) - 3} further no-op guard stop(s) occurred this sequence.)")
    stale = [s for s in guard_stops if s.get("kind") == "stale_state"]
    lines.extend(stale_stop_note(stop) for stop in stale[:3])
    if len(stale) > 3:
        lines.append(f"({len(stale) - 3} further guard stop(s) occurred this sequence.)")
    return lines


# --- images (vision) -------------------------------------------------------------------------------------------------

def _png_url(image, upscale):
    from PIL import Image

    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


def board_image_url(grid, upscale=UPSCALE):
    """The public board rendering (``frame_to_png_data_url``) at ``upscale``."""
    from inference.agent.runtime_state import Frame
    from inference.agent.vision_context import frame_to_png_data_url

    return frame_to_png_data_url(Frame(grid=tuple(tuple(int(c) for c in row) for row in grid), step=0, level=1),
                                 upscale=upscale)


def diff_image_url(before, after, upscale=UPSCALE):
    """Their ``diff_image_part`` as a data URL: cells that differ in their new palette colour, every other cell
    ``DIFF_UNCHANGED_RGB``; None when the shapes differ or nothing changed."""
    from PIL import Image

    from inference.agent.vision_context import ARC_COLOR_MAP

    a, b = np.asarray(before, dtype=np.int64), np.asarray(after, dtype=np.int64)
    if a.shape != b.shape or a.ndim != 2 or a.size == 0 or not (a != b).any():
        return None
    rows, cols = b.shape
    image = Image.new("RGB", (cols, rows), DIFF_UNCHANGED_RGB)
    pixels = image.load()
    for r, c in zip(*np.nonzero(a != b), strict=True):
        pixels[int(c), int(r)] = ARC_COLOR_MAP.get(int(b[r, c]), ARC_COLOR_MAP[0])
    return _png_url(image, upscale)


def stub_extra_images(message):
    """``message`` with every extra image (one that follows an ``EXTRA_IMAGE_LABELS`` text part) replaced by the
    placeholder text; the same object when there is none (the board image is untouched)."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts, changed, previous = [], False, None
    for part in content:
        if (isinstance(part, dict) and part.get("type") == "image_url" and isinstance(previous, dict)
                and previous.get("type") == "text" and str(previous.get("text", "")).startswith(EXTRA_IMAGE_LABELS)):
            parts.append({"type": "text", "text": HISTORY_IMAGE_PLACEHOLDER})
            changed = True
        else:
            parts.append(part)
        previous = part
    return {**message, "content": parts} if changed else message


# --- animation (theirs: inference/utils/animation.py) ----------------------------------------------------------------

def normalize_frames(frames):
    out = []
    for frame in frames or ():
        array = np.asarray(frame)
        if array.dtype.kind in "iu":  # v7.1: tolist already gives python ints
            out.append(tuple(map(tuple, array.tolist())))
        else:
            out.append(tuple(tuple(int(cell) for cell in row) for row in array.tolist()))
    return out


def collapse_frames(before, frames):
    """The pre-action board, then each distinct state the action passed through (runs of identical grids collapse;
    non-adjacent repeats stay). ``before`` is prepended so a cell X -> Y -> X registers two changes."""
    collapsed = [] if before is None else [before]
    for grid in frames:
        if collapsed and collapsed[-1] == grid:
            continue
        collapsed.append(grid)
    return collapsed


def transient_cells(chain):
    """Cells whose value changes two or more times across the chain (shown, then no longer there)."""
    if len(chain) > 1:
        try:
            stack = np.asarray(chain)
        except ValueError:
            stack = None
        if stack is not None and stack.ndim == 3 and stack.dtype.kind in "iu":
            # v7.1: one array pass (same cells as the loop below for same-shape grids)
            counts = (stack[1:] != stack[:-1]).sum(axis=0)
            return [(int(r), int(c)) for r, c in np.argwhere(counts >= 2)]
    changes = defaultdict(int)
    for before, after in zip(chain, chain[1:]):
        for row in range(max(len(before), len(after))):
            before_row = before[row] if row < len(before) else ()
            after_row = after[row] if row < len(after) else ()
            for col in range(max(len(before_row), len(after_row))):
                old = before_row[col] if col < len(before_row) else None
                new = after_row[col] if col < len(after_row) else None
                if old != new:
                    changes[(row, col)] += 1
    return sorted(cell for cell, count in changes.items() if count >= 2)


def summarize_animation(chain):
    """``{"frames", "transient_pixels", "transient_bbox"?}`` for an action that animated, None otherwise."""
    if len(chain) <= 2:
        return None
    transient = transient_cells(chain)
    summary = {"frames": len(chain) - 1, "transient_pixels": len(transient)}
    if transient:
        rows = [cell[0] for cell in transient]
        cols = [cell[1] for cell in transient]
        summary["transient_bbox"] = [min(rows), min(cols), max(rows), max(cols)]
    return summary


def _bbox_phrase(bbox):
    if not bbox:
        return ""
    top, left, bottom, right = bbox
    if top == bottom and left == right:
        return f" at ({top},{left})"
    if top == bottom:
        return f" in row {top}, cols {left}-{right}"
    if left == right:
        return f" in col {left}, rows {top}-{bottom}"
    return f" around rows {top}-{bottom}, cols {left}-{right}"


def describe_animation(summary, *, action_display, gameplay_changed, game_over=False):
    """Their opener line, or "" unless cells changed and changed back."""
    if not summary:
        return ""
    transient = int(summary.get("transient_pixels") or 0)
    if transient <= 0:
        return ""
    frames = summary.get("frames")
    cells = "1 cell changed and changed back" if transient == 1 else f"{transient} cells changed and changed back"
    where = _bbox_phrase(summary.get("transient_bbox"))
    name = f" (`{action_display}`)" if action_display else ""
    look = "`last_animation_timeline`"
    if game_over:
        return (f"That fatal action animated over {frames} frames, and inside it {cells}{where}. Those cells are in "
                f"neither `history[-3].frame` nor `history[-2].frame` - read {look} to see what happened between them.")
    if gameplay_changed is False:
        return (f"The last executed action{name} animated over {frames} frames and left the board area unchanged, but "
                f"it was not a no-op: {cells}{where}. Whatever this action did is visible only there - read {look} "
                "before concluding it failed.")
    return (f"The last executed action{name} animated over {frames} frames. {cells[0].upper()}{cells[1:]}{where}. "
            f"{look} shows them.")


def _color_char(value):
    from inference.utils.grid_utils import ARC_COLOR_CHARS

    return "?" if value is None else ARC_COLOR_CHARS[max(0, min(15, int(value)))]


def _diff_cells(before, after):
    changes = []
    for row in range(max(len(before), len(after))):
        before_row = before[row] if row < len(before) else ()
        after_row = after[row] if row < len(after) else ()
        for col in range(max(len(before_row), len(after_row))):
            old = before_row[col] if col < len(before_row) else None
            new = after_row[col] if col < len(after_row) else None
            if old != new:
                changes.append((row, col, old, new))
    return changes


def animation_timeline(record):
    """Their ``build_animation_view``: which cells changed at each step of the last animated action."""
    chain = list((record or {}).get("chain") or ())
    if len(chain) <= 2:
        return {"error": "No animation is available: the last executed action returned a single frame."}
    rendered, remaining = [], ANIMATION_MAX_TOTAL_CELLS
    for position in range(1, len(chain)):
        cells = _diff_cells(chain[position - 1], chain[position])
        if not cells:
            continue
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        entry = {"step": position, "changed": len(cells),
                 "bbox": f"rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)}"}
        budget = min(ANIMATION_MAX_CELLS_PER_STEP, max(0, remaining))
        if len(cells) > ANIMATION_MAX_CELLS_PER_STEP or budget == 0:
            census = {}
            for _, _, old, new in cells:
                key = f"{_color_char(old)}>{_color_char(new)}"
                census[key] = census.get(key, 0) + 1
            entry["transitions"] = census
        else:
            grouped = {}
            for row, col, old, new in cells[:budget]:
                grouped.setdefault(f"{_color_char(old)}>{_color_char(new)}", []).append(f"({row},{col})")
            lines = [f"{key} @ {' '.join(points)}" for key, points in grouped.items()]
            if len(cells) > budget:
                lines.append(f"... {len(cells) - budget} further changed cells omitted")
            entry["changes"] = lines
            remaining -= min(len(cells), budget)
        rendered.append(entry)
    return {"action": record.get("action_display"), "frames": len(chain) - 1, "steps": rendered}


class AnimationStore:
    """``animation_history``: every animated action's frame chain of one game, zlib-compressed, at most
    ``limit`` bytes (oldest steps dropped first)."""

    def __init__(self, limit=ANIMATION_STORE_BYTES):
        self.limit, self.bytes, self.items, self.dropped, self.stored = int(limit), 0, {}, 0, 0

    def put(self, record):
        import zlib

        chain = np.asarray(record["chain"], dtype=np.uint8)
        blob = zlib.compress(chain.tobytes(), 1)
        step = int(record["step"])
        previous = self.items.pop(step, None)
        if previous is not None:
            self.bytes -= len(previous["blob"])
        self.items[step] = {"shape": chain.shape, "blob": blob, "action_display": record.get("action_display"),
                            "level": record.get("level")}
        self.bytes += len(blob)
        self.stored += 1
        while self.bytes > self.limit and len(self.items) > 1:
            oldest = next(iter(self.items))
            self.bytes -= len(self.items.pop(oldest)["blob"])
            self.dropped += 1

    def steps(self):
        return sorted(self.items)

    def get(self, step):
        """The record of ``step`` (the shape ``animation_frames`` / ``animation_timeline`` take), or None."""
        import zlib

        item = self.items.get(int(step))
        if item is None:
            return None
        chain = np.frombuffer(zlib.decompress(item["blob"]), dtype=np.uint8).reshape(item["shape"])
        return {"action_display": item["action_display"], "step": int(step), "level": item["level"],
                "chain": [tuple(map(tuple, grid.tolist())) for grid in chain]}


def animation_frames(record, frame_payload):
    """Every distinct state the action passed through after the pre-action board, as frame-view payloads
    (``frame_payload(grid, step, level)``, the public ``_ascii_frame_view_payload`` shape with the grid)."""
    chain = list((record or {}).get("chain") or ())
    if len(chain) <= 2:
        return []
    return [frame_payload(grid, record.get("step") or 0, record.get("level") or 0) for grid in chain[1:]]


# --- the persistent sandbox (frame_diff, guards, animation) ----------------------------------------------------------

# Their stale-state and terminal messages (new wording), with our runtime variable names.
STALE_DETAIL = (
    "{action} was not executed. The action immediately before it, {previous}, was a no-op - it produced no change "
    "inside the board area - so positions and paths computed since then may be wrong, which is why execution has "
    "been halted here. Re-read current_frame and last_action_result, work out why nothing moved, then act on what "
    "you find. (StaleStateActionError can be caught if a snippet has good reason to keep acting through no-ops.)"
)
STALE_DETAIL_TRANSIENT = (
    "{action} was not executed. The action immediately before it, {previous}, played an animation and left the board "
    "area exactly as it was (cells changed and changed back; it was not a no-op), so positions and paths computed "
    "for the actions after it may be wrong, which is why execution has been halted here. Read "
    "`last_animation_timeline` to see what {previous} did, then act on what you find. (StaleStateActionError can be "
    "caught if a snippet has good reason to keep acting.)"
)
TERMINAL_DETAIL = {
    "game_over": ("the previous action reported game_over. The level is reset to its start before your next turn, so "
                  "the board you planned against is gone and this action was not executed. The next turn begins from "
                  "the new state."),
    "level_completed": ("the previous action reported level_completed. A new level has been loaded, so the board you "
                        "planned against is gone and this action was not executed. The next turn begins from the new "
                        "state."),
    "run_complete": "the previous action reported run_complete. The run is finished and this action was not executed.",
}

# Child code inside ``main()`` before the request loop. ``__DFR_FLAGS__`` and the two message tables are filled in by
# ``with_dfr``.
CHILD_BLOCK = r'''
    _DFR_FLAGS = __DFR_FLAGS__
    _DFR_STALE_DETAIL = __DFR_STALE__
    _DFR_STALE_TRANSIENT = __DFR_STALE_T__
    _DFR_TERMINAL_DETAIL = __DFR_TERMINAL__
    _DFR = {"stale": None, "terminal": None, "events": [], "views": []}

    class StaleStateActionError(Exception):
        """Raised INSTEAD of executing an action after an earlier action of this snippet changed nothing inside the
        board area (catchable: a warning about the snippet's beliefs, not a verdict on the action)."""

    class TerminalStateActionError(BaseException):
        """Raised INSTEAD of executing an action after this snippet's attempt or level ended."""

    def _dfr_display(item):
        if isinstance(item, dict):
            name = str(item.get("action", "")).strip().upper()
            if name == "MOUSE":
                return "MOUSE(row=%s, col=%s)" % (item.get("row"), item.get("col"))
            return name
        return str(item).strip().upper()

    def _dfr_items(actions):
        if isinstance(actions, (str, dict)):
            return [actions]
        try:
            return list(actions)
        except TypeError:
            return []

    _dfr_inner_action = runtime_globals["action"]

    def action(actions):
        if _DFR_FLAGS["guards"]:
            items = _dfr_items(actions)
            if _DFR["terminal"]:
                _DFR["events"].append({"kind": "terminal", "after": _DFR["terminal"],
                                       "action": _dfr_display(items[0]) if items else None})
                raise TerminalStateActionError(_DFR_TERMINAL_DETAIL[_DFR["terminal"]])
            frame = runtime_globals.get("current_frame")
            level = int(getattr(frame, "level", 0) or 0)
            if (_DFR["stale"] is not None and items and _DFR_FLAGS.get("sweep")
                    and _DFR["stale"].startswith("MOUSE(") and _dfr_display(items[0]).startswith("MOUSE(")
                    and _dfr_display(items[0]) != _DFR["stale"]):
                _DFR["events"].append({"kind": "sweep_exempt", "action": _dfr_display(items[0]),
                                       "previous": _DFR["stale"], "level": level})
                _DFR["stale"] = None
            if _DFR["stale"] is not None and items and level >= __DFR_FROM_LEVEL__:
                shown = _dfr_display(items[0])
                _DFR["events"].append({"kind": "stale_state", "action": shown, "previous": _DFR["stale"],
                                       "level": level, "skipped": [_dfr_display(i) for i in items]})
                detail = _DFR_STALE_TRANSIENT if _DFR.get("stale_transient") else _DFR_STALE_DETAIL
                raise StaleStateActionError(detail.format(action=shown, previous=_DFR["stale"]))
        for view in _DFR["views"]:
            view.invalidate()
        result = _dfr_inner_action(actions)
        if _DFR_FLAGS["guards"] and isinstance(result, dict):
            for flag in ("run_complete", "done", "level_completed", "game_over"):
                if result.get(flag) and result.get("executed"):
                    _DFR["terminal"] = "run_complete" if flag == "done" else flag
                    break
            changed = result.get("gameplay_changed")
            if changed is True:
                _DFR["stale"] = None
            elif changed is False and result.get("executed") and _DFR["stale"] is None:
                executed = result.get("executed_actions")
                _DFR["stale"] = str(executed[-1] if isinstance(executed, list) and executed
                                    else result.get("action_display") or "?")
                _DFR["stale_transient"] = bool(result.get("transient_effect"))
        return result

    runtime_globals["action"] = action
    runtime_globals["StaleStateActionError"] = StaleStateActionError
    runtime_globals["TerminalStateActionError"] = TerminalStateActionError

    def frame_diff(before=None, after=None):
        """Object-level diff between two frames (default: previous_frame -> current_frame); only what changed."""
        if before is None:
            before = runtime_globals.get("previous_frame")
        if after is None:
            after = runtime_globals.get("current_frame")
        if before is None or after is None:
            return {"error": "need two frames (no previous_frame available?)"}
        if tuple(before.shape) != tuple(after.shape):
            return {"changed_cell_count": -1, "note": "frame shapes differ; full redraw",
                    "before_shape": before.shape, "after_shape": after.shape}
        return compute_frame_diff(before._grid, after._grid, before.segmentation.get("nodes", []),
                                  after.segmentation.get("nodes", []))

    def _dfr_fetch(kind, step=None):
        interrupt = globals().get("_INTERRUPT")
        armed = bool(interrupt and interrupt["armed"])
        if interrupt:
            interrupt["armed"] = False
        try:
            _send({"type": "animation", "request": {"kind": kind} if step is None else {"kind": kind, "step": step}})
            reply = _recv()
        finally:
            if interrupt:
                interrupt["armed"] = armed
        if reply.get("type") == "animation_error":
            raise RuntimeError(str(reply.get("error", "animation failed")))
        if reply.get("type") != "animation_result":
            raise RuntimeError("Invalid animation response from sandbox host.")
        payload = reply.get("animation")
        if step is not None and isinstance(payload, dict) and payload.get("unavailable"):
            raise LookupError(str(payload["unavailable"]))
        if kind == "steps":
            return list(payload or [])
        if kind != "frames":
            return payload or {}
        return [_frame_from_payload(item) for item in (payload or [])]

    def _dfr_step_of(x):
        if isinstance(x, int) and not isinstance(x, bool):
            return x
        frame = getattr(x, "after_frame", None) or getattr(x, "frame", None)
        step = getattr(frame, "step", None)
        if step is None:
            raise TypeError("pass a step number, a history entry or a transition")
        return int(step)

    def animation_frames(x):
        """The frames of the animated action at step/entry/transition x (as last_animation_frames right after it);
        [] when that action returned a single frame."""
        return _dfr_fetch("frames", _dfr_step_of(x))

    def animation_timeline(x):
        """The diff timeline of the action at step/entry/transition x (as last_animation_timeline right after it)."""
        return _dfr_fetch("timeline", _dfr_step_of(x))

    def animated_steps():
        """The steps whose action animated and whose frames are stored."""
        return _dfr_fetch("steps", -1)

    class _LazyAnimation:
        def __init__(self, kind, empty):
            self._kind, self._empty, self._value = kind, empty, None

        def _get(self):
            if self._value is None:
                self._value = _dfr_fetch(self._kind) or self._empty
            return self._value

        def invalidate(self):
            self._value = None

        def __len__(self):
            return len(self._get())

        def __iter__(self):
            return iter(self._get())

        def __getitem__(self, key):
            return self._get()[key]

        def __bool__(self):
            return bool(self._get())

        def __contains__(self, key):
            return key in self._get()

        def keys(self):
            return self._get().keys()

        def items(self):
            return self._get().items()

        def get(self, key, default=None):
            return self._get().get(key, default)

        def __repr__(self):
            return repr(self._get())

    _dfr_frames_view = _LazyAnimation("frames", [])
    _dfr_timeline_view = _LazyAnimation("timeline", {})
    if _DFR_FLAGS["animation"]:
        _DFR["views"].extend((_dfr_frames_view, _dfr_timeline_view))

    _dfr_refresh_inner = _refresh_state

    def _refresh_state(state_payload):
        _dfr_refresh_inner(state_payload)
        if _DFR_FLAGS["frame_diff"]:
            runtime_globals["frame_diff"] = frame_diff
        if _DFR_FLAGS["animation"]:
            runtime_globals["last_animation_frames"] = _dfr_frames_view
            runtime_globals["last_animation_timeline"] = _dfr_timeline_view
        if _DFR_FLAGS.get("animation_history"):
            runtime_globals.update(animation_frames=animation_frames, animation_timeline=animation_timeline,
                                   animated_steps=animated_steps)

    def _dfr_begin():
        _DFR["stale"] = None
        _DFR["terminal"] = None
        for view in _DFR["views"]:
            view.invalidate()

    def _dfr_take():
        taken = list(_DFR["events"])
        del _DFR["events"][:]
        return taken

'''
ANCHOR = "    request = initial\n    while True:\n"
RESULT_FIELD = '"action_results": _json_safe(action_results),'
BEGIN_RE = re.compile(r"^( *)del action_results\[:\]\n", re.M)
CATCHES = {"        except Exception as exc:\n            _send(":
               "        except (Exception, TerminalStateActionError) as exc:\n            _send(",
           "            except (Exception, ToolTimeout) as exc:\n":
               "            except (Exception, ToolTimeout, TerminalStateActionError) as exc:\n"}
SEGMENTATION_OLD = (
    '        boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))\n'
    "        nodes.append(\n"
    "            {\n"
    '                "id": cid,\n'
    '                "color": color,\n'
    '                "hash": _object_hash(comp["cells"], color),\n'
)
SEGMENTATION_NEW = (
    '        boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))\n'
    '        signature = _object_signature(comp["cells"], color)\n'
    "        nodes.append(\n"
    "            {\n"
    '                "id": cid,\n'
    '                "color": color,\n'
    '                "hash": signature[0],\n'
    '                "rotation": signature[1],\n'
    '                "rotational_symmetry": signature[2],\n'
    '                "pose_hash": signature[3],\n'
    '                "shape_hash": signature[4],\n'
)
LIB_ANCHOR = "\nHOST_STDOUT = sys.stdout\n"
RUNTIME_NAMES = '"last_action_result", "action", "result"'
DFR_RUNTIME_NAMES = ('"frame_diff", "last_animation_frames", "last_animation_timeline", "StaleStateActionError", '
                     '"TerminalStateActionError", "animation_frames", "animation_timeline", "animated_steps"')


def _pinned(text, old, new, what):
    if text.count(old) != 1:
        raise ValueError(f"dfr_harness: pinned Duck sandbox {what} changed")
    return text.replace(old, new, 1)


def with_dfr(bootstrap, block):
    """``bootstrap`` (either loop of ``duck_transfer.sandbox.stateful_bootstrap``, with or without action plans)
    with the on flags' child code: the frame-diff library and rotation-aware segmentation, the snippet latches and
    the lazy animation views, plus the ``dfr_guards`` field of the call's final and error messages."""
    import inspect
    import json

    from . import dfr_sandbox_lib

    if "_DFR_FLAGS = " in bootstrap:
        raise ValueError("dfr_harness: the sandbox code is already installed")
    flags = {key: bool(block[key]) for key in ("guards", "frame_diff", "animation")}
    if block.get("sweep"):
        flags["sweep"] = True
    if block.get("animation_history"):
        flags["animation_history"] = True
    child = (CHILD_BLOCK.replace("__DFR_FLAGS__", repr(flags))
             .replace("__DFR_STALE__", repr(STALE_DETAIL))
             .replace("__DFR_STALE_T__", repr(STALE_DETAIL_TRANSIENT))
             .replace("__DFR_TERMINAL__", json.dumps(TERMINAL_DETAIL, ensure_ascii=False))
             .replace("__DFR_FROM_LEVEL__", str(int(GUARDS_FROM_LEVEL))))
    text = _pinned(bootstrap, ANCHOR, child.lstrip("\n") + ANCHOR, "loop")
    if text.count(RESULT_FIELD) != 2:
        raise ValueError("dfr_harness: pinned Duck sandbox result messages changed")
    text = text.replace(RESULT_FIELD, RESULT_FIELD + ' "dfr_guards": _dfr_take(),')
    begins = BEGIN_RE.findall(text)
    if len(begins) != 1:
        raise ValueError("dfr_harness: pinned Duck sandbox request start changed")
    text = BEGIN_RE.sub(lambda m: m.group(0) + m.group(1) + "_dfr_begin()\n", text, count=1)
    catches = [old for old in CATCHES if text.count(old) == 1]
    if len(catches) != 1:
        raise ValueError("dfr_harness: pinned Duck sandbox error handler changed")
    text = text.replace(catches[0], CATCHES[catches[0]], 1)
    lib = inspect.getsource(dfr_sandbox_lib)
    text = _pinned(text, LIB_ANCHOR, "\n" + lib + LIB_ANCHOR, "module head")
    if flags["frame_diff"]:
        text = _pinned(text, SEGMENTATION_OLD, SEGMENTATION_NEW, "segmentation")
    if text.count(RUNTIME_NAMES) == 1:  # the fixes loop's inventory skips runtime names
        text = text.replace(RUNTIME_NAMES, RUNTIME_NAMES + ", " + DFR_RUNTIME_NAMES, 1)
    return text
