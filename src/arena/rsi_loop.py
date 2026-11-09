"""RSI arm (8 October): what to test next, a test memory that outlives deaths, drains and level changes, and a
second reader with a fresh context. Opt-in study key ``rsi_loop``; silent before the stuck signal: ``SIGNAL_TOKENS``
generated gameplay tokens on a level without a clear, ``FIRST_LEVEL_TOKENS`` on a game's first level (hx5's 30K signal
fired on level 1 in 27 of 36 attempts, where the mechanics are still being learned, and level 1 got dearer).

Why (docs/harness-arm.md 5.2, the hx4 reader reports): 16 of 33 stuck levels kept a mechanic read off one refusal;
tests were announced and never run after a death ("test X first next life", three times), or repeated after a reset;
an input was left untried for 35 turns. The model-written world model is wiped at every death and level change.

Three flags, all computed on the host from the game's own history (``duck_history``), so nothing depends on the
python session or the conversation:
- ``record``: per action, boards changed / boards tried on this level; the few places where a working action did
  nothing; restarts and actions per attempt; changes away from the main change.
- ``tests``: one queue of open tests ranked by expected information per action. The unit is an action name or a kind
  of clicked object; the belief is a Beta posterior on "changes the board" over distinct boards, with the earlier
  levels' counts as a discounted prior; the gain of one more try is the mutual information between its outcome and
  that rate. A test closes when the history shows it executed and its result stays listed. After a death or a level
  change the block leads the user turn.
- ``verify``: one tool-less request with a fresh context at the signal and every ``VERIFY_EVERY`` tokens after it. It
  reads the actor's carried claims (not its reasoning), the record and selected transitions, and names rules that
  rest on one observation or are contradicted, one discriminating action list and untested goal alternatives.
Generic: no per-game information; constants below.
"""
import math
import re

FLAGS = ("record", "tests", "verify")
DEFAULTS = {"record": False, "tests": False, "verify": False,
            "thinking": False,      # verify: reasoning on (at ``effort``) or off
            "effort": "low",
            "cap": 1536}            # verify: completion cap (reasoning included)
SIGNAL_TOKENS = 30000        # generated gameplay tokens on a level without a clear (hx5's signal)
FIRST_LEVEL_TOKENS = 60000   # ... on the first level of a game
VERIFY_EVERY = 30000    # generated gameplay tokens on the level between two reader calls
BORDER = 4              # the outer ring the comparison ignores (the dfr guards' ring)
PRIOR_WEIGHT = 0.5      # an earlier level's board counts as half a board of this level
MIN_GAIN = 0.08         # nats: below this a unit is known well enough (three no-op boards, or six changing ones)
RETRY_GAP = 5           # actions since the only try of a unit (a no-op) before a retry is listed
DROPPED_GAP = 20        # ... and since the no-op of a unit that had worked before (a shorter gap is routine play)
TOP_TESTS = 3
RESULTS_SHOWN = 4
BACKGROUND = 400        # cells: a larger component is background, not an object to click
DEAD_CLICKS = 2         # colours clicked without any reaction in the game before click tests stop
CELLS_KEPT = 96         # a change of more cells is kept as a count only
FEW = 3                 # "did nothing only in a few places": at most this many no-op boards, fewer than the changes
UPTAKE_ACTIONS = 10     # a pushed test counts as taken up when it closes within this many actions
READER_TRANSITIONS = 48
READER_RULES, READER_TESTS, READER_GOALS = 3, 2, 2
MOUSE_RE = re.compile(r"^MOUSE\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)$", re.I)


def signal_tokens(levels_completed):
    return FIRST_LEVEL_TOKENS if not levels_completed else SIGNAL_TOKENS


def settings(value):
    """Validated ``study["rsi_loop"]`` (None: off)."""
    if not value:
        return None
    if value is True:
        value = {"record": True, "tests": True}
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown rsi_loop settings: {sorted(unknown)}")
    block = {**DEFAULTS, **value}
    for flag in FLAGS + ("thinking",):
        if not isinstance(block[flag], bool):
            raise ValueError(f"rsi_loop.{flag} must be a boolean")
    if block["effort"] not in ("low", "medium", "xhigh"):
        raise ValueError("rsi_loop.effort must be low, medium or xhigh")
    if type(block["cap"]) is not int or block["cap"] < 256:
        raise ValueError("rsi_loop.cap must be an integer of at least 256")
    if block["verify"] and not block["tests"]:
        raise ValueError("rsi_loop.verify feeds the test queue (tests must be on)")
    return block if any(block[flag] for flag in FLAGS) else None


# --- information per action ---------------------------------------------------------------------------------------

def _digamma(x):
    value = 0.0
    while x < 6.0:
        value -= 1.0 / x
        x += 1.0
    inverse = 1.0 / (x * x)
    return value + math.log(x) - 0.5 / x - inverse * (1.0 / 12 - inverse * (1.0 / 120 - inverse / 252))


def gain(changed, noop):
    """Nats one more try tells about a unit's rate of changing the board, under Beta(1 + changed, 1 + noop)."""
    a, b = 1.0 + changed, 1.0 + noop
    p, total = a / (a + b), _digamma(a + b + 1.0)
    return (-(p * math.log(p) + (1 - p) * math.log(1 - p))
            - p * (total - _digamma(a + 1.0)) - (1 - p) * (total - _digamma(b + 1.0)))


# --- boards -------------------------------------------------------------------------------------------------------

def _board(grid):
    return hash(tuple(row[BORDER:len(row) - BORDER] for row in grid[BORDER:len(grid) - BORDER]))


def _changed_cells(before, after):
    """Interior cells that differ, as (row, col); None when the shapes differ."""
    if len(before) != len(after):
        return None
    cells = []
    for r in range(BORDER, len(after) - BORDER):
        a, b = before[r], after[r]
        if a is b or a == b:
            continue
        if len(a) != len(b):
            return None
        cells.extend((r, c) for c in range(BORDER, len(b) - BORDER) if a[c] != b[c])
    return cells


def _component(grid, row, col, limit=BACKGROUND):
    """The 4-connected component of equal value at (row, col): (value, cells); cells is None past ``limit``."""
    height, width = len(grid), len(grid[0]) if grid else 0
    if not (0 <= row < height and 0 <= col < width):
        return None, None
    value, seen, stack = grid[row][col], {(row, col)}, [(row, col)]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in seen and grid[nr][nc] == value:
                seen.add((nr, nc))
                if len(seen) > limit:
                    return value, None
                stack.append((nr, nc))
    return value, seen


def objects(grid):
    """The board's non-background components inside the outer ring, by colour: {value: [count, cells of the largest,
    (row, col) inside the largest]}."""
    height, width = len(grid), len(grid[0]) if grid else 0
    done, kinds = set(), {}
    for r in range(BORDER, height - BORDER):
        for c in range(BORDER, width - BORDER):
            if (r, c) in done:
                continue
            value, cells = _component(grid, r, c, limit=height * width)
            done |= cells
            if len(cells) > BACKGROUND:
                continue
            entry = kinds.setdefault(int(value), [0, 0, None])
            entry[0] += 1
            if len(cells) > entry[1]:
                rows = sorted(cell for cell in cells if BORDER <= cell[0] < height - BORDER
                              and BORDER <= cell[1] < width - BORDER)  # a cell to click: never in the outer ring
                entry[1], entry[2] = len(cells), rows[len(rows) // 2]
    return kinds


def _clusters(cells):
    """Changed cells grouped when within two cells of each other: a list of cell lists, largest first."""
    left, groups = set(cells), []
    while left:
        start = left.pop()
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr in (-2, -1, 0, 1, 2):
                for dc in (-2, -1, 0, 1, 2):
                    near = (r + dr, c + dc)
                    if near in left:
                        left.discard(near)
                        group.append(near)
                        stack.append(near)
        groups.append(group)
    return sorted(groups, key=len, reverse=True)


def _box(cells):
    rows, cols = [r for r, _ in cells], [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


# --- the record ---------------------------------------------------------------------------------------------------

class Record:
    """Per-game ledger statistics and the test queue, fed from ``duck_history`` (entries with ``action`` and
    ``frame.grid`` / ``.step`` / ``.level``) and the receipts by step (``game_over``)."""

    def __init__(self, color_chars=""):
        self.chars = str(color_chars or "")
        self.seen = 0
        self.level = None
        self.earlier = {}        # unit -> [boards changed, boards without a change] over the levels before
        self.results = []        # closed tests, oldest first: {"level", "step", "text"}
        self.event = None        # "death" | "level": the next block leads the user turn
        self.pushes = []         # (test id, step, context) per shown test and turn kind
        self.closes = {}         # test id -> step it closed at
        self.grid = None
        self._start(None)

    def _start(self, level):
        self.level = level
        self.rows = []           # dicts: step, label, unit, board, cells (list | None), n, won, died, life
        self.lives = []          # actions of each ended attempt on the level
        self.life = 0
        self.tests = {}          # id -> {"id", "unit", "kind", "text", "gain", "cost", "board", "open", "step"}
        self.reader = []         # the second reader's open tests: {"id", "text", "actions", "settles", "step"}
        self.reader_notes = []
        self.reader_round = 0
        self._clock = None

    # -- naming --

    def color(self, value):
        return self.chars[value] if 0 <= value < len(self.chars) else str(value)

    def kind_text(self, unit):
        return "the background" if unit[1] == "bg" else f"a '{self.color(unit[1])}' object"

    def unit_text(self, unit):
        return f"`{unit[1]}`" if unit[0] == "K" else "click on " + self.kind_text(unit)

    # -- feeding --

    def update(self, history, results):
        """Consume the new history entries; closes the tests they execute."""
        for index in range(self.seen, len(history)):
            entry = history[index]
            frame = entry.frame
            self.grid = frame.grid
            if self.level is None:
                self._start(int(frame.level))
            action = str(entry.action or "").strip()
            before = history[index - 1].frame if index else None
            if not action or before is None:
                continue
            if action.upper() == "RESET":
                if self.life:
                    self.lives.append(self.life)
                    self.event = "death"
                self.life = 0
                if int(frame.level) != self.level:  # an engine that restarts the whole game
                    self._start(int(frame.level))
                continue
            won = int(frame.level) > int(before.level)
            cells = _changed_cells(before.grid, frame.grid)
            click = MOUSE_RE.match(action)
            if click:
                row, col = int(click.group(1)), int(click.group(2))
                value, body = _component(before.grid, row, col)
                unit = ("C", "bg") if body is None else ("C", int(value))
            else:
                unit = ("K", action.upper())
            count = len(cells) if cells is not None else CELLS_KEPT + 1
            kept = cells if cells is not None and count <= CELLS_KEPT else None
            item = {"step": int(frame.step), "label": action, "unit": unit, "board": _board(before.grid),
                    "cells": kept, "n": count, "boxes": [_box(g) for g in _clusters(kept)[:4]] if kept else [],
                    "won": won, "died": bool((results.get(int(frame.step)) or {}).get("game_over")),
                    "life": len(self.lives), "index": len(self.rows)}
            self.rows.append(item)
            self.life += 1
            self._close(item)
            if item["died"]:  # the attempt ends with the action; the RESET that follows starts the next one
                self.lives.append(self.life)
                self.life, self.event = 0, "death"
            if won:
                self._roll(int(frame.level))
            elif int(frame.level) != self.level:
                self._start(int(frame.level))
        self.seen = len(history)

    def _roll(self, level):
        for unit, (changed, noop, *_) in self.stats().items():
            entry = self.earlier.setdefault(unit, [0, 0])
            entry[0] += len(changed)
            entry[1] += len(noop)
        self._start(level)
        self.event = "level"

    # -- statistics --

    def clock_lines(self):
        """Rows and columns that tick: at least 8 actions, at least 80% of them changed 1-4 cells on the line, and
        at least 80% of the line's changed cells changed once per attempt (a bar that runs down, not an object that
        moves back and forth). ``({rows}, {cols})``."""
        if self._clock is not None and len(self.rows) < self._clock[0]:
            return self._clock[1]
        rows = [r for r in self.rows if r["cells"]]
        self._clock = (int(len(self.rows) * 1.1) + 8, (set(), set()))  # recomputed as the level grows by a tenth
        if len(rows) < 8:
            return self._clock[1]
        found = []
        for axis in (0, 1):
            hits, cells = {}, {}
            for item in rows:
                lines = {}
                for cell in item["cells"]:
                    lines[cell[axis]] = lines.get(cell[axis], 0) + 1
                    key = (item["life"], cell)
                    cells[key] = cells.get(key, 0) + 1
                for line, count in lines.items():
                    if count <= 4:
                        hits[line] = hits.get(line, 0) + 1
            ticking = set()
            for line, count in hits.items():
                if count < 0.8 * len(rows):
                    continue
                on_line = [n for (_, cell), n in cells.items() if cell[axis] == line]
                if sum(1 for n in on_line if n == 1) >= 0.8 * len(on_line):
                    ticking.add(line)
            found.append(ticking)
        self._clock = (self._clock[0], (found[0], found[1]))
        return self._clock[1]

    def effective(self, item, clock=None):
        """The action changed something other than a ticking line."""
        if item["won"] or item["cells"] is None:
            return True
        rows, cols = clock if clock is not None else self.clock_lines()
        return any(r not in rows and c not in cols for r, c in item["cells"])

    def stats(self):
        """unit -> [boards it changed, boards it did nothing on, tries, last row, last no-op row]."""
        clock, table = self.clock_lines(), {}
        for item in self.rows:
            entry = table.setdefault(item["unit"], [set(), set(), 0, None, None])
            changed = self.effective(item, clock)
            entry[0 if changed else 1].add(item["board"])
            entry[2] += 1
            entry[3] = item
            if not changed:
                entry[4] = item
        for entry in table.values():
            entry[1] -= entry[0]  # a board the unit both changed and did not change counts as changed
        return table

    def belief(self, unit, table):
        entry, prior = table.get(unit), self.earlier.get(unit, (0, 0))
        changed = (len(entry[0]) if entry else 0) + PRIOR_WEIGHT * prior[0]
        noop = (len(entry[1]) if entry else 0) + PRIOR_WEIGHT * prior[1]
        return changed, noop

    # -- the queue --

    def candidates(self, valid_actions, grid):
        """The harness's tests for the current board, best first: (id, unit, kind, gain per action, cost, text)."""
        table, found = self.stats(), []
        board = _board(grid) if grid is not None else None
        names = [str(a).upper() for a in (valid_actions or [])]
        for name in names:
            if name in ("RESET", "UNDO", "MOUSE"):
                continue
            unit = ("K", name)
            if unit not in table:
                value = gain(*self.belief(unit, table))
                prior = self.earlier.get(unit)
                text = f"`{name}` has not been tried on this level"
                if prior and sum(prior):
                    text += f" (earlier levels: it changed {prior[0]} of {sum(prior)} boards)"
                found.append((("untried", unit), unit, "untried", value, 1, text))
        if "MOUSE" in names and grid is not None:
            clicked = {unit for unit in table if unit[0] == "C"}
            # colours whose clicks never did anything count against the colours not clicked yet; reactions do not
            dead = (sum(1 for u in clicked if not table[u][0])
                    + PRIOR_WEIGHT * sum(1 for u, e in self.earlier.items() if u[0] == "C" and not e[0]))
            # until a click has reacted in this game: one click test at a time, none after DEAD_CLICKS dead colours
            alive = any(table[u][0] for u in clicked) or any(e[0] for u, e in self.earlier.items() if u[0] == "C")
            listed = 0
            for value, (count, size, cell) in sorted(objects(grid).items(), key=lambda kv: (kv[1][0], -kv[1][1])):
                unit = ("C", int(value))
                if unit in clicked or (not alive and (listed or dead >= DEAD_CLICKS)):
                    continue
                listed += 1
                prior = self.earlier.get(unit) or (0, 0)
                text = (f"no click yet on {self.kind_text(unit)} on this level ("
                        + (f"{count} on the board; one" if count > 1 else f"{size} cells,")
                        + f" at row {cell[0]}, col {cell[1]})")
                value = gain(PRIOR_WEIGHT * prior[0], dead + PRIOR_WEIGHT * prior[1])
                found.append((("unclicked", unit), unit, "unclicked", value, 1, text))
        for unit, (changed, noop, tries, last, last_noop) in table.items():
            if last_noop is None or last is not last_noop or unit[0] == "C" and unit[1] == "bg":
                continue
            since = len(self.rows) - 1 - last_noop["index"]
            if since < (DROPPED_GAP if changed else RETRY_GAP):
                continue
            # only what happened since the unit last worked counts: one no-op board after any number of changes
            cost = 2 if board == last_noop["board"] else 1
            if changed:
                text = (f"{self.unit_text(unit)} changed {len(changed)} board(s), then did nothing at step "
                        f"{last_noop['step']} and has not been tried since ({since} actions ago): one no-op is not "
                        "a rule, try it once on the present board")
                value = gain(0, 1)
            elif len(noop) == 1:
                text = (f"{self.unit_text(unit)} did nothing at step {last_noop['step']}, the only board it was "
                        "tried on: one no-op is not a rule, try it once on a different board")
                value = gain(*self.belief(unit, table))
            else:
                continue
            found.append((("retry", unit, last_noop["step"]), unit, "retry", value, cost, text, last_noop["board"]))
        found = [f for f in found if f[3] >= MIN_GAIN]
        return sorted(found, key=lambda f: -f[3] / f[4])

    def refresh(self, valid_actions, grid=None):
        """Open or drop the harness's tests for the current board; returns the open queue, best first."""
        grid = self.grid if grid is None else grid
        step = self.rows[-1]["step"] if self.rows else 0
        current = {}
        for test_id, unit, kind, value, cost, text, *board in self.candidates(valid_actions, grid):
            old = self.tests.get(test_id)
            current[test_id] = {"id": test_id, "unit": unit, "kind": kind, "gain": value, "cost": cost, "text": text,
                                "step": old["step"] if old else step, "source": "harness",
                                "board": board[0] if board else None}
        self.tests = current
        queue = [dict(t, source="reader", cost=max(1, len(t["actions"])), gain=None) for t in self.reader]
        return queue + sorted(current.values(), key=lambda t: -t["gain"] / t["cost"])

    def _close(self, item):
        clock = self.clock_lines()
        changed = self.effective(item, clock)
        outcome = "cleared the level" if item["won"] else "changed the board" if changed else "did nothing"
        for test_id, test in list(self.tests.items()):
            if test["unit"] != item["unit"]:
                continue
            if test["kind"] == "retry":
                if test["board"] == item["board"]:
                    continue
                text = f"{self.unit_text(test['unit'])} retried after its no-op at step {test_id[2]}: {outcome}"
            else:
                text = f"first {self.unit_text(test['unit'])}: {outcome}"
            # a key that works at its first try is no news: counted as closed, not listed
            self._closed(test_id, item["step"], text, carry=test["kind"] == "retry" or not changed,
                         listed=not (test["kind"] == "untried" and changed))
            del self.tests[test_id]
        for test in list(self.reader):
            wanted = test["actions"]
            if not wanted or len(self.rows) < len(wanted) or [
                    r["label"].upper().replace(" ", "") for r in self.rows[-len(wanted):]] != wanted:
                continue
            if self.rows[-len(wanted)]["step"] <= test["step"]:
                continue  # the list must be executed after the test was proposed
            flags = ["cleared the level" if r["won"] else "changed" if self.effective(r, clock) else "nothing"
                     for r in self.rows[-len(wanted):]]
            self._closed(test["id"], item["step"], f"second reader's test \"{test['text']}\": executed, per action: "
                         + ", ".join(flags), carry=True)
            self.reader.remove(test)

    def _closed(self, test_id, step, text, carry=False, listed=True):
        """``carry``: worth showing on a later level (a surprise: a no-op, a retry, a reader's test)."""
        if listed:
            self.results.append({"level": self.level, "step": step, "text": text, "carry": carry})
        self.closes[test_id] = step

    def note_push(self, queue, context):
        """Remember which tests a user turn showed (``context``: "entry" after a death or level change, "mid")."""
        step = self.rows[-1]["step"] if self.rows else 0
        for test in queue:
            self.pushes.append((test["id"], step, context, test["source"], self.level))

    def uptake(self):
        """Pushed tests closed within ``UPTAKE_ACTIONS`` actions of the push, by context. One opportunity per test and
        entry event; one per test for the mid-life pushes (its first)."""
        out = {"entry": [0, 0], "mid": [0, 0]}
        seen_mid = set()
        for test_id, step, context, _source, level in self.pushes:
            if context == "mid":
                if (level, test_id) in seen_mid:
                    continue
                seen_mid.add((level, test_id))
            closed = self.closes.get(test_id)
            out[context][1] += 1
            out[context][0] += closed is not None and 0 < closed - step <= UPTAKE_ACTIONS
        return {key: {"taken": value[0], "pushed": value[1]} for key, value in out.items()}

    # -- lines --

    def action_line(self, table=None):
        table = self.stats() if table is None else table
        parts, kinds = [], [0, 0, 0]
        for unit in sorted(u for u in table if u[0] == "K"):
            changed, noop, tries = table[unit][:3]
            parts.append(f"{unit[1]} {len(changed)}/{len(changed) + len(noop)}")
        for unit, (changed, noop, tries, *_) in table.items():
            if unit[0] == "C":
                kinds[0] += 1
                kinds[1] += bool(changed)
                kinds[2] += tries
        if kinds[0]:
            parts.append(f"MOUSE {kinds[2]} clicks on {kinds[0]} colours of object ({kinds[1]} reacted)")
        return "boards changed / boards tried on this level: " + ", ".join(parts) if parts else None

    def few_line(self, table=None):
        """Working units with a few no-op boards: where a rule may have been read off one refusal."""
        table = self.stats() if table is None else table
        clock, parts = self.clock_lines(), []
        few = {unit: [] for unit, (changed, noop, *_) in table.items()
               if noop and len(noop) <= FEW and len(noop) < len(changed)}
        for item in self.rows if few else ():
            if item["unit"] in few and item["board"] in table[item["unit"]][1] and not self.effective(item, clock):
                few[item["unit"]].append(item["step"])
        for unit, steps in sorted(few.items(), key=lambda kv: str(kv[0])):
            parts.append(f"{self.unit_text(unit)} at step{'s' if len(steps) > 1 else ''} "
                         + ", ".join(str(s) for s in steps[:4]))
        return ("did nothing only here (it works elsewhere; what differs on these boards is not established): "
                + "; ".join(parts)) if parts else None

    def lives_line(self):
        if not self.lives:
            return None
        shown = ", ".join(str(n) for n in self.lives[-4:])
        line = f"the level restarted {len(self.lives)} time(s), after {shown} actions"
        if len(self.lives) >= 2 and len(set(self.lives[-2:])) == 1:
            line += f": a limit of {self.lives[-1]} actions per attempt"
        return line + f"; this attempt has used {self.life}"

    def side_line(self, places_too=False, block=False):
        """Changes away from the main change: a ticking line; for the second reader also "on most actions" (something
        acts or counts by itself: true of every step counter, so not in the player's block) and, with ``places_too``,
        places that changed now and then."""
        rows = [r for r in self.rows if r["cells"] and not r["won"]]
        if len(rows) < 8:
            return None
        tick_rows, tick_cols = self.clock_lines()
        with_side, places = 0, {}
        for item in rows:
            if len(item["boxes"]) < 2:
                continue
            with_side += 1
            for top, bottom, left, right in item["boxes"][1:]:
                key = ((top + bottom) // 16, (left + right) // 16)
                entry = places.setdefault(key, [0, top, bottom, left, right, []])
                entry[0] += 1
                entry[1:5] = [min(entry[1], top), max(entry[2], bottom), min(entry[3], left), max(entry[4], right)]
                entry[5].append(item["step"])
        parts = []
        if tick_rows or tick_cols:
            where = [f"row {r}" for r in sorted(tick_rows)[:2]] + [f"col {c}" for c in sorted(tick_cols)[:2]]
            parts.append(f"{', '.join(where)} ticks with every action (a counter; an action that changes nothing else "
                         "is counted as doing nothing)")
        if block:
            pass
        elif with_side >= 0.6 * len(rows):
            parts.append(f"on {with_side} of {len(rows)} actions something also changed away from the main change: "
                         "something moves or counts by itself")
        elif places_too:
            rare = sorted((e for e in places.values() if 2 <= e[0] <= 0.5 * len(rows)), key=lambda e: -e[0])[:2]
            for count, top, bottom, left, right, steps in rare:
                parts.append(f"rows {top}-{bottom}, cols {left}-{right} changed on {count} actions away from the main "
                             f"change (steps {', '.join(str(s) for s in steps[-4:])})")
        return "; ".join(parts) if parts else None

    def block(self, flags, valid_actions, *, lead=None, evidence_shown=False, grid=None):
        """The user-turn block and the queue it shows: ``(text | None, shown tests)``."""
        lines, table = [], self.stats()
        if flags["record"]:
            facts = [None if evidence_shown else self.action_line(table), self.few_line(table), self.lives_line(),
                     self.side_line(block=True)]
            lines += ["- " + fact[0].upper() + fact[1:] + "." for fact in facts if fact]
        shown = []
        if flags["tests"]:
            shown = self.refresh(valid_actions, grid)[:TOP_TESTS]
            if shown:
                lines.append("- Open tests, most information per action first:")
                for number, test in enumerate(shown, 1):
                    lines.append(f"  {number}. " + self.test_text(test))
            done = [r for r in self.results if r["level"] == self.level][-RESULTS_SHOWN:]
            if not done and lead == "level":
                done = [r for r in self.results if r["carry"]][-RESULTS_SHOWN // 2:]
            if done:
                lines.append("- Already tested (do not repeat): " + "; ".join(
                    (f"level {r['level']} " if r["level"] != self.level else "") + f"step {r['step']}: {r['text']}"
                    for r in done) + ".")
            if self.reader_notes:
                lines.append("- Second reader (it saw your stated rules and this record, not your reasoning): "
                             + " ".join(self.reader_notes))
        if not lines:
            return None, []
        head = HEAD[lead] if lead in HEAD else HEAD[None]
        return head.format(level=self.level) + "\n" + "\n".join(lines), shown

    def test_text(self, test):
        if test["source"] == "reader":
            text = "[second reader] " + test["text"]
            if test["actions"]:
                text += f" Actions: {', '.join(test['actions'])}."
            if test.get("settles"):
                text += f" It settles: {test['settles']}"
            return text
        return f"{test['text']} ({test['cost']} action{'s' if test['cost'] > 1 else ''})."

    # -- the second reader --

    def transition_lines(self, limit=READER_TRANSITIONS):
        """One line per selected transition: every no-op of a unit that also works, each unit's first try, the
        last ones; oldest first."""
        clock, table = self.clock_lines(), self.stats()
        keep, first = set(), set()
        for index, item in enumerate(self.rows):
            entry = table[item["unit"]]
            if item["unit"] not in first:
                first.add(item["unit"])
                keep.add(index)
            if entry[0] and not self.effective(item, clock):
                keep.add(index)
            if item["died"] or item["won"]:
                keep.add(index)
        keep = sorted(keep)[-limit // 2:]
        keep = sorted(set(keep) | set(range(max(0, len(self.rows) - (limit - len(keep))), len(self.rows))))
        lines = []
        for index in keep:
            item = self.rows[index]
            lines.append(f"step {item['step']} {item['label']}: " + self.change_text(item, clock)
                         + (" -> the level restarted" if item["died"] else ""))
        return lines

    def change_text(self, item, clock=None):
        if item["won"]:
            return "cleared the level"
        if not self.effective(item, clock):
            return "no change" if not item["n"] else "no change (only the counter ticked)"
        if item["cells"] is None:
            return f"{item['n'] if item['n'] <= CELLS_KEPT else 'more than ' + str(CELLS_KEPT)} cells changed"
        boxes = item["boxes"]
        top, bottom, left, right = boxes[0]
        text = f"{item['n']} cells changed, mainly rows {top}-{bottom}, cols {left}-{right}"
        if len(boxes) > 1:
            top, bottom, left, right = boxes[1]
            text += f"; also rows {top}-{bottom}, cols {left}-{right}"
            if len(boxes) > 2:
                text += f" and {len(boxes) - 2} more place(s)"
        return text

    def take_reader(self, parsed, step):
        """Install the second reader's answer: its tests replace its earlier open ones."""
        self.reader_round += 1
        self.reader = []
        for number, test in enumerate(parsed["tests"][:READER_TESTS]):
            self.reader.append({"id": ("reader", self.level, self.reader_round, number), "text": test["text"],
                                "actions": test["actions"], "settles": test.get("settles"), "step": step})
        notes = [f"Shaky rule: {r}." for r in parsed["rules"][:READER_RULES]]
        notes += [f"Untested goal: {g}." for g in parsed["goals"][:READER_GOALS]]
        self.reader_notes = notes


HEAD = {
    None: "Level record (the harness's own count over level {level}; it outlives context loss and restarts):",
    "death": "The level restarted. Before planning the next attempt, the harness's record of level {level} (it "
             "outlives restarts and context loss; your earlier plan may not have):",
    "level": "New level. Carried by the harness from the earlier levels:",
}

READER_SYSTEM = (
    "You are a second reader for an agent that plays an unknown grid game through actions. You have a fresh context: "
    "you see the rules the player currently states, the harness's own record of what every action did on this level, "
    "selected transitions and the board. You do not see the player's reasoning and you must not trust its rules: "
    "check each against the record.\n"
    "A rule is shaky when the record holds one observation for it, when it turns one refusal or one no-op into "
    "\"never\" / \"impossible\" / \"immovable\" / \"frozen\", when a transition contradicts it, or when the test that "
    "produced it could not have told it from a nearby rule (a different position, colour, order or object).\n"
    "Answer in exactly this form and nothing else, at most {rules} RULE lines, {tests} TEST lines and {goals} GOAL "
    "lines:\n"
    "RULE: <the player's rule, short> | WHY SHAKY: <the steps it rests on or that contradict it>\n"
    "TEST: <one experiment whose outcome differs under the rule and under its negation> | ACTIONS: <the exact "
    "actions, comma separated, e.g. UP, UP, MOUSE(row=12, col=30); at most 6> | SETTLES: <what each outcome means>\n"
    "GOAL: <a goal the record does not rule out and the player has not tested> | CHECK: <the cheapest check>\n"
    "Prefer the rule the player's plan depends on, or the one that makes it call the level impossible. If every "
    "rule is well supported, write only: RULES SUPPORTED."
)
CLAIM_FIELDS = (("goal_model", "Goal"), ("action_model", "Actions"), ("world_model", "World"),
                ("recent_findings", "Recent findings"), ("open_questions", "Open questions"),
                ("current_plan", "Plan"), ("cross_level_notes", "From earlier levels"))
CLAIM_CHARS = 1400
CLAIM_REPLIES, CLAIM_TAIL = 2, 800   # the player's last replies: how many, and the characters kept from each end


def reply_tail(message):
    """The end of a gameplay reply (its reasoning and its text): where the player states what it concluded."""
    text = " ".join(str(message.get(key) or "") for key in ("reasoning_content", "reasoning", "content")
                    if isinstance(message.get(key), str))
    text = " ".join(text.split())
    return text[-CLAIM_TAIL:]


def reader_text(record, knowledge, valid_actions, *, tokens, grid=None, recent=()):
    """The second reader's user text (the board image is appended by the caller). ``recent``: the ends of the
    player's last replies (the carried world model is often empty: the rules live in the reasoning)."""
    flags = {"record": True, "tests": False}
    block, _ = record.block(flags, valid_actions, grid=grid)
    claims = []
    for key, label in CLAIM_FIELDS:
        text = " ".join(str((knowledge or {}).get(key) or "").split())
        if text:
            claims.append(f"- {label}: " + (text if len(text) <= CLAIM_CHARS else text[:CLAIM_CHARS] + " [cut]"))
    done = [f"step {r['step']}: {r['text']}" for r in record.results if r["level"] == record.level][-RESULTS_SHOWN:]
    kinds = sorted(objects(grid if grid is not None else record.grid).items(), key=lambda kv: -kv[1][1])[:14]
    parts = [
        f"Level {record.level}; {int(tokens):,} generated tokens on it without a clear; {len(record.rows)} actions.",
        "Valid actions: " + ", ".join(str(a) for a in (valid_actions or [])) + ".",
        "THE PLAYER'S STATED RULES (unverified):\n" + ("\n".join(claims) or "- (none stated)")
        + "".join(f"\n- The player's {'latest' if i == len(recent) - 1 else 'earlier'} reply ended: \"...{tail}\""
                  for i, tail in enumerate(recent) if tail),
        "THE HARNESS'S RECORD:\n" + (block or "(no actions yet)"),
        "Tests already run: " + ("; ".join(done) if done else "none") + ".",
        "SELECTED TRANSITIONS (rows and columns count from 0 at the top left):\n" + "\n".join(record.transition_lines()),
        "OBJECTS ON THE BOARD NOW (colour letter, how many, cells of the largest, a position inside it): " + "; ".join(
            f"'{record.color(value)}' x{count}, {size} cells, row {cell[0]}, col {cell[1]}"
            for value, (count, size, cell) in kinds) + ".",
    ]
    return "\n\n".join(parts)


def reader_system():
    return READER_SYSTEM.format(rules=READER_RULES, tests=READER_TESTS, goals=READER_GOALS)


ACTION_TOKEN = re.compile(r"MOUSE\(\s*(?:row\s*=\s*)?(-?\d+)\s*,\s*(?:col\s*=\s*)?(-?\d+)\s*\)|[A-Z][A-Z0-9_]*", re.I)


def parse_reader(text, valid_actions=None):
    """``{"rules": [str], "tests": [{"text", "actions", "settles"}], "goals": [str], "supported": bool}``; lines that
    do not follow the form are dropped; an action list with an unknown action is dropped (the test stays)."""
    valid = {str(a).upper() for a in (valid_actions or [])}
    out = {"rules": [], "tests": [], "goals": [], "supported": False}
    for raw in str(text or "").splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        head, _, body = line.partition(":")
        head, fields = head.strip().upper(), [f.strip() for f in body.split("|")]
        if line.upper().startswith("RULES SUPPORTED"):
            out["supported"] = True
        elif head == "RULE" and fields[0]:
            why = next((f.partition(":")[2].strip() for f in fields[1:] if f.upper().startswith("WHY")), "")
            out["rules"].append(fields[0].rstrip(".") + (f" ({why.rstrip('.')})" if why else ""))
        elif head == "GOAL" and fields[0]:
            check = next((f.partition(":")[2].strip() for f in fields[1:] if f.upper().startswith("CHECK")), "")
            out["goals"].append(fields[0].rstrip(".") + (f"; check: {check.rstrip('.')}" if check else ""))
        elif head == "TEST" and fields[0]:
            actions, settles = [], None
            for field in fields[1:]:
                name, _, value = field.partition(":")
                if name.strip().upper() == "ACTIONS":
                    for match in ACTION_TOKEN.finditer(value):
                        if match.group(1) is not None:
                            actions.append(f"MOUSE(ROW={int(match.group(1))},COL={int(match.group(2))})")  # as _close compares
                        else:
                            actions.append(match.group(0).upper())
                elif name.strip().upper() == "SETTLES":
                    settles = value.strip()
            names = {a.split("(")[0] for a in actions}
            if len(actions) > 6 or (valid and not names <= valid) or names & {"RESET"}:
                actions = []
            out["tests"].append({"text": fields[0].rstrip(".") + ".", "actions": actions, "settles": settles})
    return out


def new_stats():
    return {"signal_levels": 0, "blocks": 0, "block_chars": 0, "entry_blocks": {"death": 0, "level": 0},
            "reader_calls": 0, "reader_empty": 0, "reader_failed": 0, "reader_tokens": 0, "reader_prompt_tokens": 0,
            "reader_tests": 0, "reader_rules": 0, "reader_goals": 0, "reader_supported": 0, "levels": []}
