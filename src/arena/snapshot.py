"""Game-start snapshot (fixed-2x-snapshot): a deterministic probe before the first model call.

Meta-Harness's one clean harness win on a weak model was an environment bootstrap: a fixed
program inspects the environment before the agent's first turn and hands the result over as
context. Here the probe reads the start board (palette, objects) and takes each simple
available action once, in sequence, recording the transition receipt the agent would otherwise
spend a model turn per action to learn, then restores the start board with one RESET. The block
is carried as the first user message for the whole game (never inside the analyzer's
append-only history, so compaction leaves it in place). Costs actions, not model time.

fixed-2x-level-probe (27 September, user: "start every level for every game with this
deterministic trying of actions"): ``every_level`` repeats the probe at the start of every later
level, plus ``clicks`` clicks on the largest objects; the latest level's block replaces the carried
one and is also shown once in the first user turn on the new level (``arena.duck``).
"""

import time
from collections import Counter

import numpy as np

from arc_infra.objects import object_view
from arc_infra.perception import board_hash
from arc_infra.plans import Action
from duck_transfer.feedback import transition_summary
from inference.utils.grid_utils import ARC_COLOR_CHARS, ARC_COLOR_LEGEND

PROBE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
# ``every_level`` (fixed-2x-level-probe): the probe runs again at the start of every later level;
# ``clicks``: one ACTION6 click at the centre of each of the N largest non-background objects.
# The defaults keep fixed-2x-snapshot's game-start-only, click-free probe.
DEFAULTS = {"probe": True, "objects": 20, "every_level": False, "clicks": 0}
SETTINGS = frozenset(DEFAULTS)
HEADER = "[Game snapshot]"
LEVEL_HEADER = "[Level snapshot]"


def settings(raw):
    """Validated study settings (``{"probe": bool, "objects": int, "every_level": bool, "clicks": int}``)."""
    if not isinstance(raw, dict) or set(raw) - SETTINGS:
        raise ValueError(f"game_snapshot settings must be a subset of {sorted(SETTINGS)}")
    merged = {**DEFAULTS, **raw}
    if not isinstance(merged["objects"], int) or merged["objects"] < 0:
        raise ValueError("game_snapshot.objects must be a non-negative integer")
    if isinstance(merged["clicks"], bool) or not isinstance(merged["clicks"], int) or merged["clicks"] < 0:
        raise ValueError("game_snapshot.clicks must be a non-negative integer")
    return {"probe": bool(merged["probe"]), "objects": merged["objects"],
            "every_level": bool(merged["every_level"]), "clicks": merged["clicks"]}


def click_targets(board, objects, count):
    """``[((row, col), node), ...]`` for the ``count`` largest objects not of the board's most
    common colour (the background): the object's own cell nearest its bounding-box centre, so a
    ring or an L-shape is clicked on itself, not on what it encloses. Largest first; a cell is
    clicked once."""
    if count <= 0 or not objects or objects.get("skipped"):
        return []
    board = np.asarray(board)
    background = Counter(int(v) for v in board.ravel()).most_common(1)[0][0]
    nodes = sorted((node for node in objects["nodes"] if int(node["color"]) != background),
                   key=lambda node: (-node["pixels"], node["id"]))
    targets, seen = [], set()
    for node in nodes:
        r0, c0, r1, c1 = node["bbox"]
        rows, cols = np.nonzero(board[r0:r1, c0:c1] == int(node["color"]))
        if not len(rows):
            continue
        centre_r, centre_c = (r0 + r1 - 1) / 2, (c0 + c1 - 1) / 2
        k = int(np.argmin(np.abs(rows + r0 - centre_r) + np.abs(cols + c0 - centre_c)))
        cell = (int(rows[k]) + r0, int(cols[k]) + c0)
        if cell in seen:
            continue
        seen.add(cell)
        targets.append((cell, node))
        if len(targets) == count:
            break
    return targets


def palette_line(board):
    counts = Counter(int(v) for row in board for v in row)
    total = sum(counts.values())
    parts = [
        f"{ARC_COLOR_CHARS[colour]} x{count} ({100 * count / total:.0f}%)"
        for colour, count in counts.most_common()
    ]
    return f"Colours on the start board (letters as in the board views): {', '.join(parts)}."


def _box(bbox):
    return f"rows {bbox[0]}-{bbox[2] - 1}, cols {bbox[1]}-{bbox[3] - 1}"


def _entry(entry):
    letter = ARC_COLOR_CHARS[max(0, min(15, int(entry["color"])))]
    return f"{letter} px={entry['pixels']} at {_box(entry['bbox'])}"


def probe_line(label, transition):
    """One line per probed action from its transition receipt."""
    if not transition.get("changed_cells"):
        return f"{label}: no change"
    parts = [f"{transition['changed_cells']} cells changed within {_box(transition['change_bbox'])}"]
    objects = transition.get("objects")
    if objects:
        for entry in objects["moved"]:
            dr = entry["bbox"][0] - entry["from"][0]
            dc = entry["bbox"][1] - entry["from"][1]
            parts.append(f"moved {_entry(entry)} by ({dr:+d} rows, {dc:+d} cols)")
        for entry in objects["appeared"]:
            parts.append(f"appeared {_entry(entry)}")
        for entry in objects["disappeared"]:
            parts.append(f"disappeared {_entry(entry)}")
        counts = objects["counts"]
        hidden = sum(counts.values()) - sum(len(objects[k]) for k in ("moved", "appeared", "disappeared"))
        if hidden > 0:
            parts.append(f"{hidden} more object changes")
        if objects.get("background_reshaped"):
            parts.append("the background reshaped")
    return f"{label}: " + "; ".join(parts)


async def game_snapshot(agent, config, *, level_start=False):
    """Probe the current level's start board through ``agent`` and return ``(text, stats)``.

    Probes every simple action the environment advertises, each from the board the previous
    probe left, then (``clicks`` > 0 and ACTION6 advertised) one click at the centre of each of
    the ``clicks`` largest non-background objects of the start board; a probe that clears the
    level or ends the game stops the sequence. A changed board is restored with RESET (the
    level's clock and action count keep the probes: they are actions the competition counts).

    ``level_start`` (fixed-2x-level-probe): the probe of a later level, taken right after the
    clear, headed ``[Level snapshot]``; the game-start text is unchanged when ``clicks`` is 0.
    """
    from inference.agent.action_names import (
        to_model_action,  # lazy: importing inference.agent
    )
    # at module level loaded the public tool_agent with its default 12 requests per turn
    # before duck_transfer.loader could set the analyzer environment (25-26 Sept runs)

    started = time.monotonic()
    obs = agent.obs
    start_board = obs.board.copy()
    start_hash = board_hash(start_board)
    objects = agent.transitions.objects(start_board)
    intro = (f"{LEVEL_HEADER} Deterministic probe taken at the start of level {obs.levels + 1}, before your first "
             "turn on it, at no model cost; it replaces the earlier snapshot." if level_start
             else HEADER + " Deterministic probe taken before your first turn, at no model cost.")
    lines = [
        intro,
        f"Level {obs.levels + 1} of {obs.total_levels}; board 64x64.",
        palette_line(start_board),
        "Objects on the start board (largest first): " + object_view(objects, limit=config["objects"]).replace(
            "click a cell inside with ACTION6 x=col,y=row", "click a cell inside with MOUSE(row, col)"),
    ]
    stats = {"level": obs.levels + 1, "probes": 0, "clicks": 0, "changed": 0, "reset": False, "restored": None,
             "stopped": None}
    probed = []

    def running():
        return (stats["stopped"] is None and agent.obs.state == "NOT_FINISHED" and agent.obs.levels == obs.levels
                and agent.remaining() > 0)

    async def take(action, label, note=""):
        before = agent.obs.board
        before_hash = board_hash(before)
        await agent.step(action)
        transition = transition_summary(agent.transitions, before, agent.obs.board)
        changed = before_hash != board_hash(agent.obs.board)
        probed.append((before_hash, action, changed))
        stats["changed"] += changed
        result = {
            "executed": True, "action_num": agent.steps, "probe": True,
            "level": max(1, min(agent.obs.total_levels, agent.obs.levels + 1)),
            "state": agent.obs.state, "board_changed": changed,
            "level_completed": agent.obs.levels > obs.levels,
            "game_over": agent.obs.state == "GAME_OVER", "done": agent.obs.state == "WIN",
            "action_name": action.name, "action_display": label, "transition": transition,
        }
        if action.name == "ACTION6":
            result["action_data"] = {"row": action.y, "col": action.x}
        agent.write_duck_state(label, result)
        line = probe_line(label + note, transition)
        if agent.obs.levels > obs.levels:
            line += " -> the level was cleared by this probe"
            stats["stopped"] = "level_cleared"
        elif agent.obs.state != "NOT_FINISHED":
            line += f" -> {agent.obs.state}"
            stats["stopped"] = agent.obs.state
        lines.append(line)

    if config["probe"]:
        candidates = [name for name in PROBE_ACTIONS if name in obs.available]
        if candidates:
            lines.append("One-step effect of each simple action, applied in sequence from the start board:")
        for name in candidates:
            if not running():
                break
            await take(Action(name), to_model_action(name))
            stats["probes"] += 1
        targets = (click_targets(start_board, objects, config["clicks"])
                   if config["clicks"] and "ACTION6" in obs.available else [])
        if targets and running():
            lines.append(f"One click on each of the {len(targets)} largest objects of the start board (the object's "
                         "cell nearest its centre), applied in sequence after the actions above:")
            for (row, col), node in targets:
                if not running():
                    break
                await take(Action("ACTION6", col, row), f"MOUSE(row={row}, col={col})", f" on {_entry(node)}")
                stats["clicks"] += 1
        if (candidates or targets) and stats["stopped"] is None and agent.remaining() > 0:
            if board_hash(agent.obs.board) != start_hash:
                await agent.step(Action("RESET"))
                stats["reset"] = True
                stats["restored"] = board_hash(agent.obs.board) == start_hash
                agent.write_duck_state("RESET")
                agent.level_start_step = agent.steps  # RESET right after the probe reset is refused
                lines.append("RESET: start board restored." if stats["restored"]
                             else "RESET: the board after RESET differs from the start board (the level is not deterministic from RESET).")
                if stats["restored"]:
                    # The session forgot its no-ops at RESET; the probes from the start board still count.
                    for before_hash, action, changed in probed:
                        if before_hash == start_hash and not changed:
                            agent.no_ops[(agent.obs.levels, start_hash, action.identity())] += 1
            else:
                lines.append("The probes left the start board unchanged; no RESET needed.")
                if config["every_level"]:
                    agent.level_start_step = agent.steps  # a RESET here would only repeat the start board
        if config["every_level"] and stats["stopped"] == "level_cleared":
            agent.level_start_step = agent.steps  # the next level starts here, as after a model clear
        lines.append(f"Probe cost: {stats['probes'] + stats['clicks'] + stats['reset']} of this level's actions.")
        agent.stuck.reset_level()
    if config["every_level"] and not level_start:
        lines.append(f"A {LEVEL_HEADER} like this one is taken at the start of every later level and replaces this block.")
    lines.append(ARC_COLOR_LEGEND)
    stats["actions"] = stats["probes"] + stats["clicks"] + stats["reset"]
    stats["seconds"] = round(time.monotonic() - started, 3)
    return "\n".join(lines), stats
