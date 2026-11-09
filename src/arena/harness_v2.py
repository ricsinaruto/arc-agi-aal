"""Harness v2 for Duck-transfer: the five edits of the 26 September failure-signature study.

docs/validation-20260926/failure-signatures.md §4 proposes one change per failure cluster; each is
behind its own study flag (``harness_v2`` settings) so that later ablations can switch single
edits off. With every flag off the agent behaves exactly as the fixed-2x control.

- ``effects`` (E1): the receipt separates gameplay effects from clock/counter cells, mid-action
  animations and camera shifts; the stuck tracker, the no-change refusal and
  ``tried_no_effect_here`` use the gameplay change instead of the raw board hash.
- ``nudge`` (E2): after three gameplay requests without an action, the python tool result carries
  one note asking for the cheapest distinguishing action (or ``check_model``), once per streak.
- ``legend`` (E3): an action legend in the system-prompt addition (ACTION7 is usually undo, RESET
  instead of draining a timer, one no-op does not make an action dead, selection toggles, scrolls).
- ``new_level`` (E4): at each new level one notice lists the objects and colours no earlier level
  showed, plus a playbook rule that a carried-over plan that runs out without a clear is refuted.
- ``handoff`` (E5): the compaction prompt asks for a verified action/effect table, a refuted list
  and the conventions (button sides, grid pitch, RESET), plus a playbook line to act on them.

The model-facing text lives in ``src/arena/harnesses/v2`` (``HarnessV2Dir`` composes its prompt
addition, playbook and skills from the enabled edits); this module holds the host-side logic and
the co-metrics reported per game under ``harness_v2``.

Clock detection (E1) is deliberately conservative: a *clock track* is a (row or column, colour
``src`` -> ``dst``) whose every tick changes at most ``clock_tick_cells`` contiguous cells of that
line from ``src`` to ``dst``. It becomes a clock only once it has ticked on at least
``clock_min_ticks`` of the level's actions and on at least ``clock_share`` of them since its first
tick, under at least ``clock_min_actions`` distinct actions, at least once as the *only* change on
the board, never in reverse (``dst`` -> ``src`` outside a RESET), always next to an earlier tick
(a sweep), and never at a clicked cell. Only changes ``src`` -> ``dst`` inside the bar's span are
masked; anything else in that line still counts as gameplay, and a later reverse change or jump
revokes the track for the rest of the level. A moving player (changes in both directions), a
click-placed paint or an action-specific effect therefore never becomes a clock.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from duck_transfer.feedback import StuckTracker

from .harness import DuckHarness

EDITS = ("effects", "nudge", "legend", "new_level", "handoff")
DEFAULTS = {
    "effects": True, "nudge": True, "legend": True, "new_level": True, "handoff": True,
    # E1 clock tracks
    "clock_min_ticks": 3, "clock_share": 0.25, "clock_tick_cells": 3, "clock_min_actions": 2,
    "click_radius": 2, "clock_band": 3, "shift_share": 0.9, "shift_min_changed": 0.15,
    # E2 nudge: first note after this many gameplay requests without an action; 0 = once per streak
    "nudge_after_requests": 3, "nudge_repeat_requests": 0, "nudge_window_requests": 2,
    # E4 notice
    "new_level_limit": 10, "new_level_component_limit": 400,
}
ROOT = Path(__file__).resolve().parent / "harnesses" / "v2"
# Skills and text parts per edit (``HarnessV2Dir``).
SKILLS = {"effects": "effects", "nudge": "check_model", "new_level": "new_elements"}
PROMPT_PARTS = {"effects": "effects.md", "legend": "legend.md"}
PLAYBOOK_PARTS = {"new_level": "new_level.md", "handoff": "handoff.md"}
PLAYBOOK_CAP_TOKENS = 600
SENTINEL = 255
BACKGROUND_PIXELS = 2048


def settings(value):
    """The full settings dict (defaults filled in); unknown keys are an error."""
    value = dict(value or {})
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown harness_v2 settings: {sorted(unknown)}")
    return {**DEFAULTS, **value}


def enabled(config):
    return [edit for edit in EDITS if config.get(edit)]


# --- E1: effect-aware change detection ------------------------------------------------------


def _bbox(mask):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    return [int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1]


def rigid_shift(before, after, *, max_offset=32, share_min=0.9, gain_min=0.15):
    """``{"dr", "dc", "share"}`` of the axis-aligned shift that best maps ``before`` onto
    ``after`` (share of the overlap that matches), when it explains at least ``share_min`` of
    the overlap and ``gain_min`` more than no shift; else None (a scroll or pan)."""
    a, b = np.asarray(before), np.asarray(after)
    rows, cols = a.shape
    base = float((a == b).mean())
    best = None
    for d in range(-max_offset, max_offset + 1):
        if d == 0:
            continue
        for dr, dc in ((d, 0), (0, d)):
            src = a[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)]
            dst = b[max(0, dr):rows - max(0, -dr), max(0, dc):cols - max(0, -dc)]
            share = float((src == dst).mean())
            if best is None or share > best[2]:
                best = (dr, dc, share)
    if best is None or best[2] < share_min or best[2] - base < gain_min:
        return None
    return {"dr": best[0], "dc": best[1], "share": round(best[2], 3)}


class ClockTrack:
    """One candidate timer/counter bar: a line, an axis and a colour change ``src -> dst``."""

    def __init__(self, axis, index, src, dst, first):
        self.axis, self.index, self.src, self.dst = axis, index, int(src), int(dst)
        self.first = first  # learning-transition number of the first tick
        self.ticks = self.solo = self.reverse = self.jumps = self.local = 0
        self.actions = set()
        self.positions = set()
        self.clock = False
        self.revoked = None

    @property
    def key(self):
        return (self.axis, self.index, self.src, self.dst)

    def adjacent(self, positions, before, tick_cells):
        """True when the new tick continues the sweep: every cell between it and the nearest
        earlier tick is a depleted or separator cell (a tick missed while learning was off, a tick
        wider than ``clock_tick_cells`` or a segmented bar), and the gap is short."""
        line = self.line(before)
        low, high = min(positions), max(positions)
        nearest = min(self.positions, key=lambda q: min(abs(q - low), abs(q - high)))
        if low <= nearest <= high:
            return True
        gap = range(nearest + 1, low) if nearest < low else range(high + 1, nearest)
        # A segmented bar has separators between its cells; an untouched ``src`` cell in the gap
        # means the change is not the bar's next tick.
        return len(gap) <= 2 * tick_cells and all(int(line[p]) != self.src for p in gap)

    def line(self, board):
        return board[self.index, :] if self.axis == "row" else board[:, self.index]

    def span(self, board):
        """Positions of the bar in ``board``: the ticked range plus the contiguous run of
        not-yet-ticked ``src`` cells on either side (the rest of the bar)."""
        if not self.positions:
            return []
        line = self.line(board)
        low, high = min(self.positions), max(self.positions)
        cells = [p for p in range(low, high + 1) if int(line[p]) in (self.src, self.dst)]
        p = low - 1
        while p >= 0 and int(line[p]) == self.src:
            cells.append(p)
            p -= 1
        p = high + 1
        while p < len(line) and int(line[p]) == self.src:
            cells.append(p)
            p += 1
        return cells

    def cells(self, board):
        return [(p, self.index) if self.axis == "col" else (self.index, p) for p in self.span(board)]

    def describe(self):
        return {"axis": self.axis, "index": self.index, "src": self.src, "dst": self.dst, "ticks": self.ticks,
                "solo": self.solo, "actions": len(self.actions),
                "positions": [min(self.positions), max(self.positions)] if self.positions else None}


class EffectTracker:
    """Per-level clock tracks learned online from the level's own transitions."""

    def __init__(self, config):
        self.config = config
        self.reset_level()

    def reset_level(self):
        self.tracks = {}
        self.learned = 0  # learning transitions on this level

    def clocks(self):
        return [t for t in self.tracks.values() if t.clock]

    def clock_mask(self, *boards):
        mask = np.zeros((64, 64), dtype=bool)
        for track in self.clocks():
            for board in boards:
                for r, c in track.cells(board):
                    mask[r, c] = True
        return mask

    def masked_key(self, board):
        """Board identity without the clock bars (the raw ``board_hash`` when none is known)."""
        board = np.asarray(board, dtype=np.uint8)
        clocks = self.clocks()
        if not clocks:
            from arc_infra.perception import board_hash

            return board_hash(board)
        copy = board.copy()
        for track in clocks:
            for r, c in track.cells(board):
                copy[r, c] = SENTINEL
        return "m" + hashlib.blake2s(copy.tobytes(), digest_size=12).hexdigest()

    # -- learning ------------------------------------------------------------------------------

    def learn(self, before, after, identity, name, click):
        """Update the tracks with one ordinary transition; returns tracks that became (or
        stopped being) clocks: ``[("clock"|"revoked", track)]``."""
        cfg = self.config
        diff = before != after
        n = self.learned
        self.learned += 1
        changed = np.argwhere(diff)
        if not len(changed):
            return self._promote(n)
        lines = defaultdict(list)
        for r, c in changed:
            src, dst = int(before[r, c]), int(after[r, c])
            lines[("row", int(r))].append((int(c), src, dst))
            lines[("col", int(c))].append((int(r), src, dst))
        events = []
        # A reverse change on a known track's line breaks it (outside RESET, which never gets here).
        for track in list(self.tracks.values()):
            cells = lines.get((track.axis, track.index))
            if cells and any(s == track.dst and d == track.src for _, s, d in cells):
                track.reverse += 1
                if track.clock:
                    track.clock, track.revoked = False, "reverse"
                    events.append(("revoked", track))
        candidates = {}
        band = cfg["clock_band"]
        for (axis, index), cells in lines.items():
            if band is not None and band <= index < 64 - band:
                continue  # bars are learned in the outer band only (the dev games' timers and counters)
            pairs = {(s, d) for _, s, d in cells}
            positions = sorted(p for p, _, _ in cells)
            if (len(cells) <= cfg["clock_tick_cells"] and len(pairs) == 1
                    and positions[-1] - positions[0] == len(positions) - 1):
                src, dst = pairs.pop()
                candidates[(axis, index, src, dst)] = positions
        covered = set()
        for (axis, index, _, _), positions in candidates.items():
            covered |= {(index, p) if axis == "row" else (p, index) for p in positions}
        # "No-op-looking": the ticks are the whole change (at most two bars ticking at once).
        solo = len(covered) == len(changed) <= 2 * cfg["clock_tick_cells"]
        for key, positions in candidates.items():
            axis, index, src, dst = key
            cells = [(index, p) if axis == "row" else (p, index) for p in positions]
            track = self.tracks.get(key)
            if track is None:
                track = self.tracks[key] = ClockTrack(axis, index, src, dst, n)
            if click is not None and any(max(abs(r - click[0]), abs(c - click[1])) <= cfg["click_radius"]
                                         for r, c in cells):
                track.local += 1  # a change at the clicked cell is the click's effect, never a clock
                if track.clock:
                    track.clock, track.revoked = False, "click"
                    events.append(("revoked", track))
                continue
            if track.positions and not track.adjacent(positions, before, cfg["clock_tick_cells"]):
                track.jumps += 1
                if track.clock:
                    track.clock, track.revoked = False, "jump"
                    events.append(("revoked", track))
            track.ticks += 1
            track.solo += solo
            track.actions.add(identity)
            track.positions.update(positions)
        return events + self._promote(n)

    def _promote(self, n):
        cfg = self.config
        events = []
        for track in self.tracks.values():
            if track.clock or track.revoked:
                continue
            seen = n - track.first + 1
            if (track.ticks >= cfg["clock_min_ticks"] and track.ticks >= cfg["clock_share"] * seen
                    and track.reverse == 0 and track.jumps == 0 and track.local == 0 and track.solo >= 1
                    and len(track.actions) >= cfg["clock_min_actions"]):
                track.clock = True
                events.append(("clock", track))
        return events

    # -- one executed action -------------------------------------------------------------------

    def observe(self, before, frames, *, identity, name, click=None, learn=True):
        """Effect of one action from ``before`` to ``frames[-1]`` (``frames`` are every frame the
        action returned). ``learn`` is False for RESET, level changes and terminal frames."""
        cfg = self.config
        before = np.asarray(before, dtype=np.uint8)
        frames = [np.asarray(f, dtype=np.uint8) for f in frames]
        after = frames[-1]
        events = self.learn(before, after, identity, name, click) if learn else []
        raw = before != after
        clock = np.zeros_like(raw)
        for track in self.clocks():
            line_before, line_after = track.line(before), track.line(after)
            for p in track.span(before):
                if int(line_before[p]) == track.src and int(line_after[p]) == track.dst:
                    clock[(track.index, p) if track.axis == "row" else (p, track.index)] = True
        clock &= raw
        gameplay = raw & ~clock
        cells = int(gameplay.sum())
        effect = {
            "raw_changed": bool(raw.any()),
            "gameplay_changed": cells > 0,
            "gameplay_cells": cells,
            "gameplay_bbox": _bbox(gameplay),
            "clock_cells": int(clock.sum()),
            "clock_only": bool(raw.any()) and cells == 0,
        }
        if len(frames) > 1:
            static = self.clock_mask(before, after)
            peak, peak_mask, differs = 0, None, False
            for frame in frames[:-1]:
                moved = (frame != before) & ~static
                count = int(moved.sum())
                if count > peak:
                    peak, peak_mask = count, moved
                differs = differs or bool(((frame != after) & ~static).any())
            if differs:
                effect["animation"] = {"frames": len(frames), "peak_cells": peak, "peak_bbox": _bbox(peak_mask)
                                       if peak_mask is not None else None, "final_equals_start": cells == 0}
        effect["animated_effect"] = cells == 0 and bool(effect.get("animation", {}).get("peak_cells"))
        if cells > cfg["shift_min_changed"] * raw.size:
            shift = rigid_shift(before, after, share_min=cfg["shift_share"])
            if shift:
                effect["view_shift"] = shift
        effect["effect"] = effect["gameplay_changed"] or effect["animated_effect"]
        effect["events"] = events
        return effect


def receipt_fields(effect):
    """The receipt keys E1 adds next to the raw ``board_changed`` and ``transition``."""
    fields = {"gameplay_changed": effect["gameplay_changed"], "clock_only": effect["clock_only"]}
    if effect["gameplay_changed"]:
        fields["gameplay_cells"] = effect["gameplay_cells"]
        fields["gameplay_bbox"] = effect["gameplay_bbox"]
    for key in ("animation", "view_shift"):
        if key in effect:
            fields[key] = effect[key]
    return fields


# --- E4: new elements at a level start ------------------------------------------------------


def segment(board, chars, limit):
    """Duck's segmentation with the sandbox colour letters (so hashes match
    ``current_frame.segmentation``), plus bboxes; None for noise-like boards."""
    from arc_infra.objects import component_count
    from inference.utils.segmentation import segment_layer

    layer = [[int(v) for v in row] for row in np.asarray(board)]
    if component_count(layer) > limit:
        return None
    nodes = segment_layer(layer, chars)["nodes"]
    for node in nodes:
        rows = [r for r, _ in node["boundary"]]
        cols = [c for _, c in node["boundary"]]
        node["bbox"] = [min(rows), min(cols), max(rows), max(cols)]
    return nodes


def new_elements(current, earlier, limit=10):
    """``(new colours, new objects)`` of a level start against the nodes of earlier frames."""
    seen = {(n["color"], n["hash"]) for nodes in earlier for n in nodes}
    colours = {n["color"] for nodes in earlier for n in nodes}
    fresh = [n for n in current if (n["color"], n["hash"]) not in seen and n["pixels"] <= BACKGROUND_PIXELS]
    fresh.sort(key=lambda n: (n["color"] in colours, -n["pixels"], n["bbox"]))
    new_colours = sorted({n["color"] for n in current} - colours)
    return new_colours, fresh[:limit], len(fresh)


def new_level_notice(level, new_colours, objects, total):
    if not objects and not new_colours:
        return (f"New on level {level}: nothing; every object's colour and shape appeared on an earlier level. "
                "Compare the layout with the previous level before replaying its plan.")
    listed = "; ".join(f"{n['color']} {n['pixels']}px bbox {n['bbox']}" for n in objects)
    more = f" (+{total - len(objects)} more)" if total > len(objects) else ""
    colours = f"colours never seen before: {', '.join(new_colours)}; " if new_colours else ""
    return (f"New on level {level} (not on any earlier level's board): {colours}objects (colour, pixels, bbox "
            f"[row0, col0, row1, col1]): {listed}{more}. Probe each new element once (click it, or move next to it "
            "and try SPACE/ACTION5) before replaying the previous level's plan.")


# --- E2: nudge --------------------------------------------------------------------------------

NUDGE = ("NOTE: {n} requests since your last action. Your current model is untested until an action checks it. "
         "In your next call do one of two things: (a) take the single cheapest action whose `transition` would tell "
         "your two leading hypotheses apart, writing the change you expect first; or (b) if you built a simulator "
         "or solver, run `print(check_model(predict, observe))` and fix the first mismatch before trusting any plan "
         "it returns. Do not start another solver on an unchecked model.")
NUDGE_NO_SKILL = ("NOTE: {n} requests since your last action. Your current model is untested until an action checks "
                  "it. In your next call take the single cheapest action whose `transition` would tell your two "
                  "leading hypotheses apart, writing the change you expect first. Do not start another solver on an "
                  "unchecked model.")


# --- E5: compaction handoff -------------------------------------------------------------------


def handoff_prompt(base):
    """``DUCK_SUMMARY_PROMPT`` with items 3 and 5 replaced and a conventions item added."""
    replacements = {
        "3. Action model: what every action does (verified vs. hypothesised), including no-ops.\n":
            "3. Action/effect table, one line per action or click target: what it did, the step that showed it, "
            "and VERIFIED or HYPOTHESIS. Keep VERIFIED lines word for word from the previous handoff unless "
            "evidence contradicted them.\n",
        "5. Actions already tried on this exact board without effect.\n":
            "5. Refuted on this level: hypotheses, goal guesses and exact action sequences that failed or led to "
            "GAME_OVER, each with its step. These must not be re-tested.\n",
        "Be precise and compact; prefer facts over narrative.":
            "9. Conventions: grid origin and pitch, coordinate order, which button or object does what, and that "
            "`action(['RESET'])` restarts the level.\nBe precise and compact; prefer facts over narrative.",
    }
    for old, new in replacements.items():
        if base.count(old) != 1:
            raise ValueError("Pinned Duck summary prompt changed")
        base = base.replace(old, new, 1)
    return base


# --- harness directory --------------------------------------------------------------------------


class HarnessV2Dir(DuckHarness):
    """``src/arena/harnesses/v2`` with only the enabled edits' prompt parts, playbook parts and
    skills; the directory has no ``prompt.md``/``playbook.md`` of its own."""

    def __init__(self, root, config, **caps):
        super().__init__(root, **{"playbook_cap_tokens": PLAYBOOK_CAP_TOKENS, **caps})
        self.config = settings(config)

    def _parts(self, parts):
        texts = []
        for edit, name in parts.items():
            path = self.root / "parts" / name
            if self.config.get(edit) and path.is_file():
                texts.append(path.read_text(errors="replace").strip())
        return "\n\n".join(texts)

    def _text(self, name, cap_tokens):
        text = {"prompt.md": self._parts(PROMPT_PARTS), "playbook.md": self._parts(PLAYBOOK_PARTS)}.get(name)
        if text is None:
            return super()._text(name, cap_tokens)
        limit = int(cap_tokens * 3.5)
        if len(text) > limit:
            self.problems.append(f"{name}: {len(text)} chars over the {limit}-char cap; truncated")
            text = text[:limit]
        return text

    def skill_files(self):
        wanted = {SKILLS[edit] for edit in SKILLS if self.config.get(edit)}
        return [path for path in super().skill_files() if path.stem in wanted]

    def summary(self):
        return {**super().summary(), "harness_v2": enabled(self.config)}


# --- per-game state and co-metrics ------------------------------------------------------------

STUCK_OLD = ("consecutive actions produced no board change. Tried on this exact board with no effect: ",
             ". Re-derive the goal")


class EffectStuckTracker(StuckTracker):
    """The public stuck tracker (same counters and cadence) whose count is fed the gameplay
    effect, with the notice's evidence sentence rewritten for it (E1 item 5)."""

    def __init__(self, cfg, state, now=None):
        self.state = state
        super().__init__(cfg, now)

    def notice(self, level, tried, now=None):
        text = super().notice(level, tried, now)
        if text is None:
            return None
        if any(text.count(part) != 1 for part in STUCK_OLD):
            raise ValueError("Pinned stuck notice changed")
        animated = self.state.animated_here()
        animated_text = ", ".join(animated[:12]) + (" ..." if len(animated) > 12 else "") if animated else "none"
        text = text.replace(STUCK_OLD[0], "consecutive actions changed only the clock/counter or nothing. Actions with "
                            "no gameplay effect on this board: ", 1)
        return text.replace(STUCK_OLD[1], ". Actions whose effect happened only mid-animation (the board returned to "
                            f"its start): {animated_text}; these are not no-ops. Re-derive the goal", 1)


class HarnessV2:
    """Host-side state of one game: the effect tracker, the no-effect memory, the nudge streak,
    the pending new-level notice and every co-metric (``stats``)."""

    def __init__(self, config, *, skills=()):
        self.config = settings(config)
        self.on = {edit: bool(self.config[edit]) for edit in EDITS}
        self.skills = set(skills)
        self.tracker = EffectTracker(self.config)
        self.no_effect = {}  # (levels, masked key, identity) -> label: no gameplay effect there
        self.animated = {}  # (levels, masked key) -> labels whose effect was only mid-animation
        self.levels = 0
        self.board = None
        self.last_effect = None
        self.noop_run = 0
        self.boards = []  # masked keys of the level's recent boards (ACTION7 undo check)
        self.batch_gameplay = False
        # E2
        self.requests_since_action = 0
        self.nudged_at = None
        self.nudge_open = None  # requests_since_action at the last nudge, until an action or the window closes
        # E4
        self.level_starts = []  # segmentation nodes of earlier level boards (starts and pre-clear ends)
        self.pending_notice = None
        self.new_targets = None  # bboxes of the current level's listed new objects (co-metric)
        self.level_actions = 0
        # E5: [requests, completion tokens, tokens of the latest request] since the last compaction
        self.compaction_open = None
        self.stats = {
            "edits": enabled(self.config),
            # E1: raw (full-board) against gameplay (clock-masked) change, per executed non-RESET action
            "actions": 0, "raw_changed": 0, "raw_no_change": 0, "gameplay_changed": 0, "clock_only": 0,
            "masked_no_change": 0, "animations": 0, "animated_effects": 0, "view_shifts": 0,
            "clock_tracks": 0, "clock_revoked": 0,
            # refusals executed, and requested actions matching the no-effect memory under each key
            "refusals": 0, "masked_key_repeats": 0, "raw_key_repeats": 0,
            # E2
            "nudges": 0, "nudges_followed": 0, "actions_after_nudge": 0, "max_requests_without_action": 0,
            "check_model_calls": 0, "effects_calls": 0, "new_elements_calls": 0,
            # E3 (recorded whether or not the legend is on, for the ablation comparison)
            "action7": 0, "action7_undo": 0, "model_resets": 0, "game_overs": 0, "drain_game_overs": 0,
            # E4
            "new_level_notices": 0, "new_objects_listed": 0, "new_colours_listed": 0, "first_clicks": 0,
            "first_clicks_on_new": 0,
            # E5: requests without an action (and their completion tokens) between a compaction and the
            # next action, the doc's post_compaction_noaction_* measure
            "compactions": 0, "compactions_closed": 0, "compaction_noaction_requests": 0,
            "compaction_noaction_tokens": 0, "handoff_verified_lines": 0, "handoff_refuted_sections": 0,
        }

    # -- E1 -----------------------------------------------------------------------------------

    def observe_step(self, action, previous, obs):
        """Every environment step (``ArenaDuckMixin.step``): learn and classify."""
        level_changed = obs.levels != previous.levels
        reset = action.name == "RESET"
        click = (action.y, action.x) if action.name == "ACTION6" else None
        learn = not (reset or level_changed or obs.state != "NOT_FINISHED")
        effect = self.tracker.observe(previous.board, obs.frames, identity=action.identity(), name=action.name,
                                      click=click, learn=learn)
        self.last_effect = effect
        if reset or level_changed:
            self.no_effect.clear()
            self.animated.clear()
            self.boards = []
        if level_changed:
            self.tracker.reset_level()
        for kind, _ in effect["events"]:
            self.stats["clock_tracks" if kind == "clock" else "clock_revoked"] += 1
        self.levels, self.board = obs.levels, obs.board
        return effect

    def repeat(self, levels, board, identity, raw):
        """Refusal key under the gameplay change: ``identity`` already had no gameplay effect here."""
        masked = (levels, self.tracker.masked_key(board), identity) in self.no_effect
        self.stats["masked_key_repeats"] += masked
        self.stats["raw_key_repeats"] += bool(raw)
        return masked

    def tried_here(self, levels, board):
        digest = self.tracker.masked_key(board)
        return [label for (lv, key, _), label in self.no_effect.items() if lv == levels and key == digest]

    def animated_here(self):
        if self.board is None:
            return []
        return list(self.animated.get((self.levels, self.tracker.masked_key(self.board)), []))

    def note_action(self, action, label, payload, before, before_levels, obs):
        """Book-keeping after one executed ``step_duck`` action; returns the change the stuck
        tracker counts (the gameplay effect when ``effects`` is on, else the raw change)."""
        effect = self.last_effect or {}
        stats = self.stats
        ordinary = action.name != "RESET"
        if not ordinary:
            stats["model_resets"] += 1
        else:
            stats["actions"] += 1
            stats["raw_changed" if effect.get("raw_changed") else "raw_no_change"] += 1
            stats["gameplay_changed"] += bool(effect.get("gameplay_changed"))
            stats["clock_only"] += bool(effect.get("clock_only"))
            stats["masked_no_change"] += not effect.get("effect")
            stats["animations"] += "animation" in effect
            stats["animated_effects"] += bool(effect.get("animated_effect"))
            stats["view_shifts"] += "view_shift" in effect
        after_key = self.tracker.masked_key(obs.board)
        if action.name == "ACTION7":
            stats["action7"] += 1
            stats["action7_undo"] += len(self.boards) >= 2 and after_key == self.boards[-2]
        if self.on["effects"] and ordinary and not payload["level_completed"]:
            payload.update(receipt_fields(effect))
            before_key = self.tracker.masked_key(before)
            if effect.get("animated_effect"):
                seen = self.animated.setdefault((before_levels, before_key), [])
                if label not in seen:
                    seen.append(label)
            elif not effect.get("effect"):
                self.no_effect.setdefault((before_levels, before_key, action.identity()), label)
        self.batch_gameplay |= bool(effect.get("gameplay_changed")) if ordinary else False
        effective = bool(effect.get("effect")) if self.on["effects"] else payload["board_changed"]
        self.noop_run = 0 if (not ordinary or effect.get("effect")) else self.noop_run + 1
        if payload["game_over"]:
            stats["game_overs"] += 1
            stats["drain_game_overs"] += self.noop_run >= 5
        if payload["level_completed"] or not ordinary:
            self.boards = [after_key] if not payload["level_completed"] else []
        else:
            if not self.boards:
                self.boards = [self.tracker.masked_key(before)]
            self.boards = [*self.boards[-3:], after_key]
        # E4 co-metric: clicks among the level's first five actions that hit a listed new object
        self.level_actions += 1
        if self.new_targets is not None and self.level_actions <= 5 and action.name == "ACTION6":
            stats["first_clicks"] += 1
            stats["first_clicks_on_new"] += any(r0 - 1 <= action.y <= r1 + 1 and c0 - 1 <= action.x <= c1 + 1
                                                for r0, c0, r1, c1 in self.new_targets)
        if payload["level_completed"]:
            self.level_actions = 0
        return effective

    def begin_batch(self):
        self.batch_gameplay = False

    def end_batch(self, result):
        """One ``step_duck`` call's result: batch-level ``gameplay_changed``, refusals, and the
        nudge/compaction windows closed by an executed action."""
        self.stats["refusals"] += len(result.get("skipped_repeats") or [])
        executed = int(result.get("executed_count") or 0)
        if self.on["effects"] and executed and "gameplay_changed" in result:
            result["gameplay_changed"] = self.batch_gameplay
        if executed:
            self.note_executed(executed)
        return result

    # -- E2 -----------------------------------------------------------------------------------

    def note_request(self):
        self.requests_since_action += 1
        self.stats["max_requests_without_action"] = max(self.stats["max_requests_without_action"],
                                                        self.requests_since_action)
        if (self.nudge_open is not None
                and self.requests_since_action - self.nudge_open > self.config["nudge_window_requests"]):
            self.nudge_open = None

    def note_executed(self, count):
        """``count`` actions executed by one ``step_duck`` call."""
        if self.nudge_open is not None:
            self.stats["nudges_followed"] += 1
            self.stats["actions_after_nudge"] += count
            self.nudge_open = None
        if self.compaction_open is not None:
            requests, tokens, last = self.compaction_open
            self.stats["compactions_closed"] += 1
            self.stats["compaction_noaction_requests"] += max(0, requests - 1)
            self.stats["compaction_noaction_tokens"] += tokens - last
            self.compaction_open = None
        self.requests_since_action = 0
        self.nudged_at = None

    def note_python(self, code):
        code = code or ""
        for name in ("check_model", "effects", "new_elements"):
            self.stats[f"{name}_calls"] += f"{name}(" in code

    def nudge(self):
        """The note for a python result without an action, or None (once per streak unless
        ``nudge_repeat_requests`` is set)."""
        if not self.on["nudge"]:
            return None
        n = self.requests_since_action
        after, repeat = self.config["nudge_after_requests"], self.config["nudge_repeat_requests"]
        due = n >= after if self.nudged_at is None else (repeat > 0 and n - self.nudged_at >= repeat)
        if not due:
            return None
        self.nudged_at = self.nudge_open = n
        self.stats["nudges"] += 1
        return (NUDGE if "check_model" in self.skills else NUDGE_NO_SKILL).format(n=n)

    # -- E4 -----------------------------------------------------------------------------------

    def level_start(self, board, chars, *, level, previous_end=None):
        """Record a level's first board (and the previous level's last board); from the second
        level on, queue the notice. Runs in a worker thread (segmentation)."""
        limit = self.config["new_level_component_limit"]
        earlier = [nodes for nodes in self.level_starts if nodes is not None]
        if previous_end is not None:
            end = segment(previous_end, chars, limit)
            if end is not None:
                earlier.append(end)
        nodes = segment(board, chars, limit)
        self.level_starts.append(nodes)
        self.new_targets = None
        if not self.on["new_level"] or nodes is None or not earlier or level <= 1:
            return None
        colours, objects, total = new_elements(nodes, earlier, self.config["new_level_limit"])
        self.pending_notice = new_level_notice(level, colours, objects, total)
        self.new_targets = [n["bbox"] for n in objects]
        self.stats["new_level_notices"] += 1
        self.stats["new_objects_listed"] += len(objects)
        self.stats["new_colours_listed"] += len(colours)
        return {"level": level, "new_colours": colours, "objects": len(objects), "total": total}

    def take_notice(self):
        notice, self.pending_notice = self.pending_notice, None
        return notice

    # -- E5 -----------------------------------------------------------------------------------

    def note_compaction(self, text):
        self.stats["compactions"] += 1
        if self.compaction_open is None:
            self.compaction_open = (0, 0, 0)
        text = text or ""
        self.stats["handoff_verified_lines"] += sum("VERIFIED" in line for line in text.splitlines())
        self.stats["handoff_refuted_sections"] += "refuted" in text.lower()

    def note_reply(self, usage):
        if self.compaction_open is not None:
            requests, tokens, _ = self.compaction_open
            last = int((usage or {}).get("completion_tokens") or 0)
            self.compaction_open = (requests + 1, tokens + last, last)

    def summary(self):
        stats = dict(self.stats)
        acts = max(1, stats["actions"])
        stats["clock_only_share"] = round(stats["clock_only"] / acts, 3)
        stats["clocks"] = [t.describe() for t in self.tracker.clocks()]
        return json.loads(json.dumps(stats))
