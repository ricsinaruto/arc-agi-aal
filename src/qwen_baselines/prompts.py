"""Interface documentation plus the shared play guidance adopted on 16 September 2026."""

import base64
import json

from arc_infra.perception import PALETTE

from .evidence import grid_text, png

TASK = """You are playing an unfamiliar grid game.
Infer its rules and goal from observations and action outcomes.
Complete its levels using as few game actions as practical.
{interface}

You have to finish the game within the remaining time limit. A game usually consists of 5-10 levels, and on average requires 500-1000 actions to complete all levels based on average human gameplay performance. Be efficient.

Your solve target is {seconds:g} seconds of actual gameplay. Waiting for inference,
tools and context maintenance consumes this time; setup does not.
Coordinates are zero-based: x=column, y=row, grid[y,x]; ACTION6 needs x,y in 0..63.
Other advertised actions (including ACTION7) take no coordinates. RESET resets the
current level using the competition interface. GAME_OVER is reset by the runner,
and the reset receipt is reported. Level transitions, resets and terminal states
stop further actions in the current response. A new decision may resume play.
An increase in levels_completed is a clear; WIN means the entire game is complete.
board_changed reports a difference, not proof of progress.
Frames are ordered: intermediate animation first, settled board last. frame_count
includes the settled frame. A count does not specify animation duration.
{legend_line}

Games may lack an avatar: the relevant state may be an object, region, cursor, selector
or whole-board configuration. A repeated strip of small blocks or a bar flush against a
border that changes on every action while the interior does not is a HUD element such as
a step counter or timer, not something to click through; HUD-only changes are not
evidence that a move worked. After every action check whether a gameplay object changed
or only HUD elements did. After a level clear or any abrupt scene change, re-read the new
board before reusing a plan; layouts and mechanics can change between levels. Probe with
single discriminating actions while the mechanics are uncertain; once they are
understood, stop probing and search (enumerate or BFS over candidate sequences) for the
shortest reliable sequence, then execute it.
{cap_rule}{evidence_rule}Re-issuing an action (same coordinates) that already produced no board
change from the current board is rejected as an error: choose a different action.

End {turn_unit} with a block that starts with the line MEMORY: and keep it below 500
words: what you have learned about this game, which mechanics are verified, the goal as
currently understood, and the plan. It is the only text carried to your next turn, so
rewrite it fully each time.
"""

MEMORY_MARK = "MEMORY:"

ONE_ACTION = (
    "act takes an ordered list of actions. Only one action is executed per response; "
    "further list items and act calls receive a skipped result. Observe the outcome before "
    "acting again.\n"
)

PYTHON_API = """
Only python(code) is available. Functions and variables persist across calls,
levels and resets. Available modules: numpy as np, math, collections, itertools,
functools, heapq, bisect, statistics, random, re, json, operator, fractions, decimal.
observe() returns authoritative metadata, immutable grid and ordered frame arrays.
step(action, x=None, y=None) returns the authoritative post-action observation.
Bindings current_frame and grid refresh at call entry; retained references remain
immutable snapshots. Helpers should accept a board argument or call observe().
segment(board) returns the board's 4-connected same-colour components as nodes with
id, color, hash (colour+shape signature, translation-invariant), pixels, bbox
[row0, col0, row1, col1], clockwise boundary corner points [row, col] and enclosed
children, plus an adjacency_list of touching node pairs. Use it as the primary structural
view (exact tile and object positions for clicks, hashes to track pieces across frames);
read only small crops of grid for detail and never print a whole board.
changes() returns the latest transition summary (changed cells and bbox, objects that
appeared, disappeared or moved, board_changed, no_change_repeat); previous() returns
the observation before the latest one. Each observation's metadata carries the same
transition summary under 'transition'.
history contains all observations including the initial state; transitions contains
each executed action with before/after observations and outcome. transcript contains
submitted code, full output, errors and receipt references. These read-only sequences
support len, indexing, slicing and iteration, loading older evidence on demand.
Observation messages show only the settled frame image; intermediate animation frames
are in observe()['frames'] and history[i]['frames'] (settled frame last).
Ordinary exceptions preserve earlier assignments and actions. Each call has at most
30 seconds, clipped by remaining gameplay time. Hard timeout restarts the namespace,
explicitly reports its loss, and preserves archive access. Code is never replayed.
An acting Python call may inspect and branch on step results in the same program;
remaining tool calls in that assistant response are skipped after it completes.
"""

PI_API = """# ARC API
The initial settled image is supplied once. Later images and grids are on-demand
immutable references. Use read on an image to view it. Current metadata identifies
all frame paths and evidence/index.jsonl lists the complete observation chronology.
Files persist across calls and levels; each bash process starts fresh. Quick
experiments can run directly in bash; writing a reusable file is optional.
Only this game's workspace, read-only evidence and public Python runtime are visible.
There is no external network. A bash call has at most 30 seconds, clipped by gameplay.
Actions and their receipts survive shell failure; never assume a failed program did
nothing. Further actions stop at a level/reset/terminal boundary until a new decision.

```bash
python - <<'PY'
import numpy as np
from arc_api import observe, step
from segmentation import segment
before = np.load(observe()['settled']['grid_npy'], allow_pickle=False)
print([(n['id'], n['color'], n['bbox']) for n in segment(before)['nodes'] if n['pixels'] < 200])
out = step('ACTION4')  # Choose an advertised action.
after = np.load(out['settled']['grid_npy'], allow_pickle=False)
print({'observation_id': out['observation_id'],
       'changed_cells': np.argwhere(before != after).tolist()})
PY
```
step(action, x=None, y=None) synchronously publishes the result and returns metadata,
including executed action, outcome, levels, status, frame count and ordered paths.
observe() returns current metadata without executing an action; observe(grid=True) adds
the grid and frames as lists. transition() returns the latest transition summary (changed
cells and bbox, objects that appeared, disappeared or moved, board_changed,
no_change_repeat); every step result carries it under 'transition'. Evidence references
remain snapshots. ACTION6 uses x=column,y=row in 0..63; other actions have no arguments.
segment(board) (module segmentation, importable in every bash Python) returns the
board's 4-connected same-colour components: nodes with id, color, hash (colour+shape
signature, translation-invariant), pixels, bbox [row0, col0, row1, col1], clockwise
boundary corner points [row, col] and enclosed children, plus an adjacency_list of
touching pairs. Use it as the primary structural view (exact tile and object positions
for clicks, hashes to track pieces across frames); read only small crops of a grid for
detail and never print or cat a whole board into the conversation.
"""


def legend_line(*, text_grid, image_pixels):
    if text_grid:
        legend = ", ".join(f"{i:X}=#{c}" for i, c in enumerate(PALETTE))
        return f"Grid symbols 0-9,A-F represent color IDs 0-15. Color legend (hex RGB): {legend}"
    legend = ", ".join(f"{i}=#{c}" for i, c in enumerate(PALETTE))
    return (
        f"Each image shows the 64x64 board at {image_pixels // 64} pixels per cell; cell values "
        f"are color IDs 0-15. Color legend (ID=hex RGB): {legend}"
    )


OBJECT_VIEW_RULE = (
    "Every observation carries an object view (the board's 4-connected same-colour "
    "components, largest first, with colour, size, bbox and a shape hash) and, after an "
    "action, a transition summary (changed cells and their bounding box, objects that "
    "appeared, disappeared or moved, board_changed, level_completed). Read these before "
    "deciding; use small crops of the grid only for detail. "
)
TRANSITION_RULE = (
    "After an action, every observation carries a transition summary (changed cells and "
    "their bounding box, objects that appeared, disappeared or moved, board_changed, "
    "level_completed). Read it before deciding; use segment() for structure and small crops "
    "of the grid for detail. "
)


def evidence_rule(object_view):
    return OBJECT_VIEW_RULE if object_view else TRANSITION_RULE


def cap_rule(actions_per_turn):
    if not actions_per_turn:
        return ""
    return (
        f"At most {actions_per_turn} game actions are executed per response; further ones are "
        "rejected until your next decision. "
    )


def action_rule(actions_per_turn):
    if actions_per_turn == 1:
        return ONE_ACTION
    text = (
        f"act takes an ordered list of 1-{actions_per_turn or 20} actions executed in order; "
        "execution stops at a level, reset or terminal boundary and the remaining items "
        "receive a skipped result. Observe the outcome before acting again.\n"
    )
    if actions_per_turn is not None:
        text += (
            f"At most {actions_per_turn} actions are executed per response across all act "
            "calls; further ones are skipped.\n"
        )
    return text


def system(
    baseline,
    seconds=600,
    *,
    text_grid=False,
    image_pixels=512,
    actions_per_turn=20,
    object_view=False,
):
    interfaces = {
        "A": "Use the act tool to interact with the game.",
        "B": "Use the python tool and step function to interact with the game.",
        "C": "Use the coding tools and the arc_api Python client described in ARC_API.md to interact with the game.",
    }
    text = TASK.format(
        interface=interfaces[baseline],
        seconds=seconds,
        cap_rule=cap_rule(actions_per_turn),
        evidence_rule=evidence_rule(object_view),
        legend_line=legend_line(text_grid=text_grid, image_pixels=image_pixels),
        turn_unit="every response"
        if baseline in ("A", "B")
        else "every turn (your final message before you stop calling tools)",
    )
    if baseline == "A":
        text += "\n" + action_rule(actions_per_turn)
    if baseline == "B":
        text += PYTHON_API
    return text


def memory_block(text):
    """The MEMORY: block of an assistant message, or None when absent."""
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    if not isinstance(text, str):
        return None
    position = -1
    for line_start in range(len(text)):
        if text.startswith(MEMORY_MARK, line_start) and (
            line_start == 0 or text[line_start - 1] == "\n"
        ):
            position = line_start
    if position < 0:
        return None
    body = text[position + len(MEMORY_MARK) :].strip()
    return body or None


def observation(
    archive,
    index=-1,
    *,
    text_grid=False,
    pixels=512,
    settled_only=False,
    memory=None,
    objects=None,
):
    meta = archive.metadata(index)
    content = []
    if memory:
        content.append({"type": "text", "text": "Memory from your previous turn:\n" + memory})
    content.append({"type": "text", "text": "Observation: " + json.dumps(meta)})
    if objects:
        content.append({"type": "text", "text": "Objects (settled frame):\n" + objects})
    if text_grid:
        content.append(
            {"type": "text", "text": "Settled grid:\n" + grid_text(archive.frame(index))}
        )
    # A: every frame, intermediate animation included (no other evidence access).
    # B: the settled frame only; observe()['frames'] and history hold the rest.
    indices = [meta["settled_frame_index"]] if settled_only else range(meta["frame_count"])
    for i in indices:
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"{meta['observation_id']} frame {i}: {meta['frames'][i]['role']}"
                    + (
                        f" ({meta['intermediate_frame_count']} intermediate frames omitted)"
                        if settled_only and meta["intermediate_frame_count"]
                        else ""
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(png(archive.frame(index, i), pixels)).decode()
                    },
                },
            ]
        )
    return {"role": "user", "content": content}


def tools(baseline, actions_per_turn=20):
    if baseline == "A":
        name = "act"
        description = (
            "Execute an ordered list of advertised game actions and return their receipts."
        )
        properties = {
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": actions_per_turn or 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["RESET", *[f"ACTION{i}" for i in range(1, 8)]],
                        },
                        "x": {"type": "integer", "minimum": 0, "maximum": 63},
                        "y": {"type": "integer", "minimum": 0, "maximum": 63},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            }
        }
        required = ["actions"]
    else:
        name, description = "python", "Execute Python in this game's persistent session."
        properties, required = {"code": {"type": "string"}}, ["code"]
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
    ]
