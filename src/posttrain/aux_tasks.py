"""AZR-style single-turn auxiliary tasks from game code, verified by the engine (paper notes 2.7 (d)).

Three families, each Duck-transfer sub-skill as one prompt and one answer:

- ``predict``: a board, the recent actions on the level and 1-5 further actions; answer the board
  after the last action, either in full (64 lines of 64 letters) or as a patch against the given board.
  Verified by exact comparison (either answer form is accepted).
- ``reach``: a board and a target board (as a patch) reachable within k <= 6 actions; answer a JSON
  action list of at most k actions. Verified by replaying the level from its start in the engine and
  executing the answer: any sequence that ends on the target board without leaving the level counts.
- ``transition``: 3-6 before/after pairs of one action on one level; answer a Python function
  ``step(board, action)`` over the letter grid. Verified by running it in a subprocess sandbox (stdlib
  subset and numpy, time and memory limits) on held-out pairs of the same action and level. The engine
  gives no program, so these tasks carry no reference answer (``answer`` is None; the self-check runs a
  lookup-table oracle through the same sandbox): they are for RL or rejection sampling, not plain SFT.

Boards are rendered exactly like the Duck-transfer harness (``inference.utils.grid_utils``), and actions
use the harness's model-facing names (``UP``... ``SPACE``, ``ACTION7``, ``MOUSE(row=R, col=C)``).

States come from any ``ARCBaseGame`` subclass: level starts via ``set_level`` (checked on 9 DEV games
against recorded play: level-start boards are identical) and novelty-biased random walks within the
level, never crossing a level change or a GAME_OVER. Games are loaded from any environments directory in
the public layout (``<dir>/<game>/<version>/<game>.py`` + ``metadata.json``), so the same generator runs on
generated games (``data/posttrain/games/environment_files``).
"""

import collections
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from inference.utils.grid_utils import ARC_COLOR_CHARS, ARC_COLOR_LEGEND, format_grid_ascii

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ENV_DIRS = (REPO / "environment_files", REPO / "data" / "posttrain" / "games" / "environment_files")
FAMILIES = ("predict", "reach", "transition")
SIDE = 64
PALETTE = set(ARC_COLOR_CHARS)
_BOARD_LINE = re.compile(rf"^[{ARC_COLOR_CHARS}]{{{SIDE}}}$")
_LETTERS = re.compile(rf"^[{ARC_COLOR_CHARS}]+$")


def _load_action_names():
    """The harness's engine -> model action names, loaded from its file (importing the
    ``inference.agent`` package would import the public tool agent with its default environment)."""
    import inference

    path = Path(inference.__file__).parent / "agent" / "action_names.py"
    spec = importlib.util.spec_from_file_location("_duck_action_names", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.ENGINE_TO_MODEL_ACTION)


ENGINE_TO_MODEL = _load_action_names()
MODEL_TO_ENGINE = {v: k for k, v in ENGINE_TO_MODEL.items()}
KEY_IDS = (1, 2, 3, 4, 5, 7)

PREAMBLE = (
    "You are studying a turn-based ARC-AGI-3 grid game. Boards are 64 x 64 color grids rendered with "
    "ARC color symbols: one line of 64 letters per row, row 0 at the top and column 0 at the left.\n"
    f"Color legend: {ARC_COLOR_LEGEND}.\n"
    "Actions: UP, DOWN, LEFT, RIGHT, SPACE and ACTION7 are keys; MOUSE(row=R, col=C) clicks the cell in "
    "row R, column C. One action can play a short animation; only the final board is shown.\n"
    "A patch describes a board by its differences from another board: one or more rectangles, each a "
    "header line `@ row=R col=C` (the rectangle's top-left cell) followed by the rectangle's rows of "
    "letters; cells outside every rectangle keep their value."
)


# --------------------------------------------------------------------------------------------------
# Boards, actions and patches


def board_text(board):
    """The harness's ``current_frame.ascii`` for a 64x64 colour-index board."""
    return format_grid_ascii(np.asarray(board).tolist())


def parse_board(text):
    rows = str(text).split("\n")
    if len(rows) != SIDE or not all(_BOARD_LINE.match(r) for r in rows):
        raise ValueError("not a 64x64 letter board")
    lookup = {ch: i for i, ch in enumerate(ARC_COLOR_CHARS)}
    return np.array([[lookup[ch] for ch in row] for row in rows], dtype=np.uint8)


def board_key(board):
    return hashlib.blake2b(np.ascontiguousarray(board, dtype=np.uint8).tobytes(), digest_size=12).hexdigest()


def action_label(action):
    """``("ACTION1", None, None)`` -> ``UP``; ``("ACTION6", x, y)`` -> ``MOUSE(row=y, col=x)`` (the
    harness's receipt form)."""
    name, x, y = action
    label = ENGINE_TO_MODEL.get(name, name)
    return f"{label}(row={y}, col={x})" if name == "ACTION6" else label


def action_json(action):
    """The harness's ``action([...])`` item: ``"UP"`` or ``{"action": "MOUSE", "row": r, "col": c}``."""
    name, x, y = action
    label = ENGINE_TO_MODEL.get(name, name)
    return {"action": label, "row": int(y), "col": int(x)} if name == "ACTION6" else label


def action_dict(action):
    """The ``action`` argument of a transition task's ``step``: always a dict."""
    item = action_json(action)
    return item if isinstance(item, dict) else {"action": item}


def parse_action(item):
    """One action from JSON (string or ``{"action", "row", "col"}``) or a text label; None if invalid."""
    if isinstance(item, dict):
        label = str(item.get("action", "")).strip().upper()
        name = MODEL_TO_ENGINE.get(label, label if re.fullmatch(r"ACTION[1-7]", label) else None)
        if name == "ACTION6":
            row, col = item.get("row"), item.get("col")
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in (row, col)):
                return None
            return (name, col, row) if 0 <= row < SIDE and 0 <= col < SIDE else None
        return (name, None, None) if name and name != "RESET" else None
    if isinstance(item, str):
        text = item.strip().upper()
        match = re.fullmatch(r"(?:MOUSE|ACTION6)\s*\(\s*ROW\s*=\s*(\d+)\s*,\s*COL\s*=\s*(\d+)\s*\)", text)
        if match:
            return parse_action({"action": "MOUSE", "row": int(match[1]), "col": int(match[2])})
        return parse_action({"action": text}) if text not in ("MOUSE", "ACTION6") else None
    return None


_ACTION_TOKEN = re.compile(r"MOUSE\s*\(\s*row\s*=\s*\d+\s*,\s*col\s*=\s*\d+\s*\)|\b(?:UP|DOWN|LEFT|RIGHT|SPACE|"
                           r"ACTION[1-57])\b", re.IGNORECASE)


def parse_action_list(text):
    """A JSON list (the harness's ``action([...])`` argument) or comma/newline-separated labels."""
    text = str(text).strip()
    try:
        items = json.loads(text)
    except (ValueError, TypeError):
        items = None
    if isinstance(items, list):
        actions = [parse_action(item) for item in items]
        return None if any(a is None for a in actions) else actions
    if items is not None:
        return None
    leftover = _ACTION_TOKEN.sub("", text)
    if re.search(r"[A-Za-z0-9]", leftover):
        return None
    actions = [parse_action(tok) for tok in _ACTION_TOKEN.findall(text)]
    return None if any(a is None for a in actions) else actions


def patch_rects(before, after, gap=1):
    """Rectangles ``(row, col, block)`` covering every changed cell: changed cells within ``gap`` of
    each other are grouped, their bounding boxes merged until disjoint; blocks hold the after values."""
    from scipy import ndimage

    mask = np.asarray(before) != np.asarray(after)
    if not mask.any():
        return []
    grown = ndimage.binary_dilation(mask, iterations=gap) if gap else mask
    labels, _ = ndimage.label(grown, structure=np.ones((3, 3)))
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = np.nonzero((labels[sl] == i) & mask[sl])
        if len(ys):
            boxes.append([sl[0].start + ys.min(), sl[1].start + xs.min(), sl[0].start + ys.max() + 1,
                          sl[1].start + xs.max() + 1])
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    after = np.asarray(after)
    return [(int(r0), int(c0), after[r0:r1, c0:c1]) for r0, c0, r1, c1 in sorted(boxes)]


def patch_text(before, after):
    rects = patch_rects(before, after)
    if not rects:
        return "NO CHANGE"
    parts = []
    for row, col, block in rects:
        parts.append(f"@ row={row} col={col}")
        parts.extend("".join(ARC_COLOR_CHARS[int(v)] for v in line) for line in block)
    return "\n".join(parts)


_HEADER = re.compile(r"^@\s*(?:row\s*=\s*)?(\d+)\s*[,\s]\s*(?:col\s*=\s*)?(\d+)\s*$", re.IGNORECASE)


def parse_patch(text):
    """``[(row, col, [letter rows])]`` or None; ``NO CHANGE`` is the empty patch."""
    text = str(text).strip()
    if text.upper().rstrip(".") == "NO CHANGE":
        return []
    rects = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        header = _HEADER.match(line)
        if header:
            rects.append((int(header[1]), int(header[2]), []))
        elif _LETTERS.match(line) and rects:
            rects[-1][2].append(line)
        else:
            return None
    return rects if rects and all(rows for _, _, rows in rects) else None


def apply_patch(board, rects):
    out = np.array(board, dtype=np.uint8, copy=True)
    lookup = {ch: i for i, ch in enumerate(ARC_COLOR_CHARS)}
    for row, col, rows in rects:
        for i, line in enumerate(rows):
            if row + i >= SIDE or col + len(line) > SIDE:
                return None
            out[row + i, col:col + len(line)] = [lookup[ch] for ch in line]
    return out


def extract_answer(output, languages=None):
    """The last fenced block (optionally of the given languages), else the whole output."""
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", str(output), re.DOTALL)
    if languages:
        preferred = [body for lang, body in blocks if lang.lower() in languages]
        blocks = [("", b) for b in preferred] or blocks
    return blocks[-1][1].strip("\n") if blocks else str(output).strip()


def read_board_answer(text, start):
    """A full board (the last 64 board lines) or a patch against ``start``; None if neither parses."""
    lines = [ln.strip() for ln in str(text).split("\n")]
    board_lines = [ln for ln in lines if _BOARD_LINE.match(ln)]
    if len(board_lines) >= SIDE:
        return parse_board("\n".join(board_lines[-SIDE:]))
    rects = parse_patch(text)
    return None if rects is None else apply_patch(start, rects)


# --------------------------------------------------------------------------------------------------
# Games


def discover(env_dirs=DEFAULT_ENV_DIRS):
    """``{game_id: version directory}`` for every ``<dir>/<game>/<version>/metadata.json``."""
    found = {}
    for root in env_dirs:
        for meta in sorted(Path(root).glob("*/*/metadata.json")):
            try:
                game_id = json.loads(meta.read_text())["game_id"]
            except (ValueError, KeyError, OSError):
                continue
            found.setdefault(game_id, meta.parent)
    return found


def load_game_class(game_dir):
    """The ``ARCBaseGame`` subclass of a version directory, loaded like ``arc_agi``'s local wrapper
    (class name from metadata or the id's first four characters, capitalised)."""
    from arcengine import ARCBaseGame

    game_dir = Path(game_dir)
    meta = json.loads((game_dir / "metadata.json").read_text())
    game_id = meta["game_id"]
    class_name = meta.get("class_name") or (game_id[0].upper() + game_id[1:4])
    candidates = [game_dir / f"{class_name.lower()}.py", game_dir / f"{class_name}.py"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"no game source for {game_id} in {game_dir}")
    namespace = {"__name__": f"aux_tasks_game.{game_id}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)  # noqa: S102 - game code
    cls = namespace.get(class_name)
    if not (isinstance(cls, type) and issubclass(cls, ARCBaseGame)):
        raise TypeError(f"{path}: class {class_name} is not an ARCBaseGame")
    return game_id, cls


def game_factory(cls):
    kwargs = {"seed": 0} if "seed" in inspect.signature(cls).parameters else {}
    return lambda: cls(**kwargs)


class GameLibrary:
    """Game factories by id, from environments directories and/or explicit ``{game_id: factory}``."""

    def __init__(self, env_dirs=DEFAULT_ENV_DIRS, factories=None):
        self.env_dirs = tuple(env_dirs)
        self.factories = dict(factories or {})
        self._dirs = None

    def dirs(self):
        if self._dirs is None:
            self._dirs = discover(self.env_dirs)
        return self._dirs

    def resolve(self, game):
        """A full game id from an id or its prefix (the prefix must be unambiguous)."""
        if game in self.factories or game in self.dirs():
            return game
        matches = [g for g in (*self.factories, *self.dirs()) if g.split("-")[0] == game]
        if len(set(matches)) != 1:
            raise KeyError(f"game {game!r}: {'ambiguous ' + str(sorted(set(matches))) if matches else 'not found'}")
        return matches[0]

    def factory(self, game):
        game_id = self.resolve(game)
        if game_id not in self.factories:
            _, cls = load_game_class(self.dirs()[game_id])
            self.factories[game_id] = game_factory(cls)
        return self.factories[game_id]


_DEFAULT_LIBRARY = None


def default_library():
    global _DEFAULT_LIBRARY
    if _DEFAULT_LIBRARY is None:
        _DEFAULT_LIBRARY = GameLibrary()
    return _DEFAULT_LIBRARY


class Sim:
    """One game instance positioned inside a level; ``step`` returns False when the level ends
    (cleared, GAME_OVER, WIN) or the engine fails, and the instance must then be discarded."""

    def __init__(self, game):
        self.game = game
        self.board = np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.uint8)

    @classmethod
    def start(cls, factory, level_index=0):
        from arcengine import ActionInput, GameAction

        game = factory()
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if level_index:
            game.set_level(level_index)
        return cls(game)

    @classmethod
    def replay(cls, factory, level_index, prefix):
        sim = cls.start(factory, level_index)
        for action in prefix:
            if not sim.step(action):
                raise RuntimeError("recorded prefix leaves the level")
        return sim

    def clone(self):
        new = object.__new__(Sim)
        new.game = copy.deepcopy(self.game)
        new.board = self.board
        return new

    @property
    def level_index(self):
        return self.game.level_index

    @property
    def n_levels(self):
        return len(self.game._levels)

    @property
    def available(self):
        return [int(a) for a in self.game._available_actions]

    def step(self, action):
        from arcengine import ActionInput, GameAction

        name, x, y = action
        level = self.game.level_index
        data = {"x": int(x), "y": int(y)} if name == "ACTION6" else {}
        try:
            raw = self.game.perform_action(ActionInput(id=GameAction[name], data=data), raw=True)
        except Exception:  # noqa: BLE001 - a crashing action ends the level for sampling purposes
            return False
        if raw.frame:
            frame = np.asarray(raw.frame[-1], dtype=np.uint8)
            if frame.shape != (SIDE, SIDE):
                return False
            self.board = frame
        state = getattr(raw.state, "name", str(raw.state))
        return state == "NOT_FINISHED" and self.game.level_index == level


def components(board, limit=48):
    """``(size, [cells])`` per 4-connected single-colour component, the most common colour skipped,
    smallest first (the explorer's click targets)."""
    from scipy import ndimage

    colors, counts = np.unique(board, return_counts=True)
    background = colors[int(np.argmax(counts))]
    found = []
    for color in colors:
        if color == background:
            continue
        labels, n = ndimage.label(board == color)
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            ys, xs = np.nonzero(labels[sl] == i)
            found.append((len(ys), list(zip((ys + sl[0].start).tolist(), (xs + sl[1].start).tolist()))))
    found.sort(key=lambda item: (item[0], item[1][0]))
    return found[:limit]


def substantive(before, after, band=2):
    """True when a change reaches inside the outer ``band`` rows and columns: changes confined to the
    border are usually a timer or counter bar (the harness prompt calls these HUD state)."""
    mask = np.asarray(before) != np.asarray(after)
    return bool(mask[band:SIDE - band, band:SIDE - band].any())


def key_actions(available):
    return [(f"ACTION{i}", None, None) for i in available if i in KEY_IDS]


def candidate_actions(board, available, rng, *, click_limit=48, random_click=0.05):
    """Keys plus one random cell per click target (and rarely a random cell anywhere)."""
    actions = key_actions(available)
    if 6 in available:
        for _, cells in components(board, click_limit):
            y, x = cells[rng.randrange(len(cells))]
            actions.append(("ACTION6", x, y))
        if rng.random() < random_click:
            actions.append(("ACTION6", rng.randrange(SIDE), rng.randrange(SIDE)))
    return actions


def action_kind(action, board):
    name, x, y = action
    return name if name != "ACTION6" else f"click {ARC_COLOR_CHARS[int(board[y, x])]}"


# --------------------------------------------------------------------------------------------------
# State sampling


@dataclass
class State:
    """A position inside a level: the actions from the level start that reach it (replayable), the
    last few ``(action, before, after)`` steps and the board."""

    level_index: int
    prefix: list
    history: list
    board: np.ndarray
    snapshot: object = field(default=None, repr=False, compare=False)  # a Sim, made on first use


def clone_is_faithful(factory, n_levels, rng, *, sequences=3, length=16):
    """Whether a deep copy of the game behaves like the original. It does not for every game (tn36:
    copies of levels 2-5 diverge after a few clicks), so each game is checked on every level: random
    sequences played on a fresh game, on a copy of a fresh game and from a copy taken mid-sequence must
    end on the same board."""
    for level in range(n_levels):
        for _ in range(sequences):
            sim, actions, mid = Sim.start(factory, level), [], None
            copy_of_start, final = sim.clone(), sim.board
            for i in range(length):
                acts = candidate_actions(sim.board, sim.available, rng)
                if not acts:
                    break
                action = acts[rng.randrange(len(acts))]
                if i == length // 2:
                    mid = (sim.clone(), i)
                if not sim.step(action):
                    if mid is not None and mid[1] == i:
                        mid = None
                    break
                actions.append(action)
                final = sim.board
            for other, start in ((copy_of_start, 0), mid or (None, 0)):
                if other is None:
                    continue
                try:
                    replayed = StatePool._replay(other, actions[start:], copy_first=False)
                except RuntimeError:
                    return False
                if not np.array_equal(replayed.board, final):
                    return False
    return True


def step_seconds(factory, rng, n=20):
    """Median seconds per engine step on random level-1 actions."""
    sim, times = Sim.start(factory, 0), []
    for _ in range(n):
        acts = candidate_actions(sim.board, sim.available, rng)
        if not acts:
            break
        started = time.perf_counter()
        alive = sim.step(acts[rng.randrange(len(acts))])
        times.append(time.perf_counter() - started)
        if not alive:
            sim = Sim.start(factory, 0)
    return statistics.median(times) if times else 0.0


class StatePool:
    """Diverse states per level: the level start plus states along random walks within the level.

    Actions are applied in place (a deep copy of some games costs 30-300 ms once their state grows), so a
    walk's prefix also holds its no-change actions, and an action that ends the level ends the walk.
    Action choice is weighted by how often each action kind (a key, or a click on a colour) has changed
    the board inside its border on this level, so walks and task sequences spend few actions on inert
    clicks. States are recorded when their board is new. Games whose deep copies are not faithful are
    positioned by replaying from a fresh instance only."""

    def __init__(self, game_id, factory, rng, *, per_level=32, walks=8, walk_max=40, history=3, click_limit=48):
        self.game_id, self.factory, self.rng = game_id, factory, rng
        self.per_level, self.walks, self.walk_max = per_level, walks, walk_max
        self.history, self.click_limit = history, click_limit
        self.n_levels = Sim.start(factory, 0).n_levels
        self.clone_ok = clone_is_faithful(factory, self.n_levels, random.Random(f"clone:{game_id}"))
        # slow engines (lf52 steps take about 12 ms, bp35 8 ms) walk half as far
        self.slow = step_seconds(factory, random.Random(f"speed:{game_id}")) > 0.004
        if self.slow:
            self.walks, self.walk_max = max(1, walks // 2), max(1, walk_max // 2)
        self._starts, self._states = {}, {}
        self.tried = collections.defaultdict(collections.Counter)  # level -> kind -> times taken
        self.changed = collections.defaultdict(collections.Counter)  # level -> kind -> substantive changes

    def distrust_copies(self):
        """Switch to replay-only positioning and resample every level (the self-check found a state
        that a fresh replay does not reproduce)."""
        self.clone_ok = False
        self._starts.clear()
        self._states.clear()

    def start(self, level_index):
        """A fresh game at the level start (a copy of a cached one when copies are faithful)."""
        if not self.clone_ok:
            return Sim.start(self.factory, level_index)
        if level_index not in self._starts:
            self._starts[level_index] = Sim.start(self.factory, level_index)
        return self._starts[level_index].clone()

    def sim(self, state, extra=()):
        """A fresh game positioned at ``state``, then ``extra`` actions applied (RuntimeError if the
        replay leaves the level). With faithful copies the state's game is replayed once and copied
        afterwards."""
        if not self.clone_ok:
            return self._replay(self.start(state.level_index), [*state.prefix, *extra], copy_first=False)
        if state.snapshot is None:
            sim = self._replay(self.start(state.level_index), state.prefix, copy_first=False)
            state.snapshot = sim.clone()
        else:
            sim = state.snapshot.clone()
        return self._replay(sim, extra, copy_first=False) if extra else sim

    @staticmethod
    def _replay(sim, actions, copy_first=True):
        sim = sim.clone() if copy_first else sim
        for action in actions:
            if not sim.step(action):
                raise RuntimeError("replayed actions leave the level (non-deterministic game?)")
        return sim

    def available(self, level_index):
        if level_index not in self._starts:
            self._starts[level_index] = Sim.start(self.factory, level_index)
        return self._starts[level_index].available

    def pick(self, sim, level_index):
        """A random action, weighted by the kind's observed substantive-change rate on this level."""
        actions = candidate_actions(sim.board, sim.available, self.rng, click_limit=self.click_limit)
        if not actions:
            return None
        tried, changed = self.tried[level_index], self.changed[level_index]
        weights = []
        for action in actions:
            kind = action_kind(action, sim.board)
            rate = (changed[kind] + 1) / (tried[kind] + 2)
            weights.append(0.05 + rate)
        return self.rng.choices(actions, weights)[0]

    def act(self, sim, action, level_index):
        """Apply ``action`` in place and record its effect; returns ``(alive, before)``."""
        before = sim.board
        kind = action_kind(action, before)
        alive = sim.step(action)
        self.tried[level_index][kind] += 1
        if alive and substantive(before, sim.board):
            self.changed[level_index][kind] += 1
        return alive, before

    def states(self, level_index):
        if level_index not in self._states:
            self._states[level_index] = self._build(level_index)
        return self._states[level_index]

    def _build(self, level_index):
        rng = self.rng
        start = self.start(level_index)
        seen = {board_key(start.board)}
        found = [State(level_index, [], [], start.board)]
        for _ in range(self.walks):
            sim, prefix, hist = self.start(level_index), [], []
            target, changes, budget = rng.randint(1, self.walk_max), 0, 0
            while changes < target and budget < 3 * target + 10:
                budget += 1
                action = self.pick(sim, level_index)
                if action is None:
                    break
                alive, before = self.act(sim, action, level_index)
                if not alive:
                    break  # the level ended: no rollback (a deep copy of some games costs 0.1-0.3 s)
                prefix = [*prefix, action]
                hist = [*hist, (action, before, sim.board)][-self.history:]
                if substantive(before, sim.board):
                    changes += 1
                key = board_key(sim.board)
                if key not in seen:
                    seen.add(key)
                    found.append(State(level_index, prefix, hist, sim.board))
        if len(found) > self.per_level:
            found = [found[0], *rng.sample(found[1:], self.per_level - 1)]
        return found

    def random_state(self):
        return next(iter(self.random_states(1)), None)

    def random_states(self, n, level_index=None):
        """``n`` distinct states of one level (a random level with enough states unless given)."""
        levels = [level_index] if level_index is not None else list(range(self.n_levels))
        self.rng.shuffle(levels)
        for level in levels:
            states = self.states(level)
            if len(states) >= n:
                return self.rng.sample(states, n)
        return []


# --------------------------------------------------------------------------------------------------
# Prompts


def _level_lines(level_index, available):
    labels = [ENGINE_TO_MODEL.get(f"ACTION{i}", f"ACTION{i}") for i in available if i]
    return f"Level {level_index + 1}. Valid actions on this level: {', '.join(labels)}."


def _history_lines(state, max_patch_chars=1600):
    if not state.prefix:
        return "The board below is the level's starting board."
    shown = state.history
    lines = ["Actions taken on this level so far (oldest first), each with the patch it produced:"]
    if len(state.prefix) > len(shown):
        lines[0] = (f"{len(state.prefix)} actions have been taken on this level; the last {len(shown)} "
                    "(oldest first), each with the patch it produced:")
    for i, (action, before, after) in enumerate(shown, start=1):
        patch = patch_text(before, after)
        if len(patch) > max_patch_chars:
            rows, cols = np.nonzero(before != after)
            patch = (f"(large change: {len(rows)} cells in rows {rows.min()}-{rows.max()}, cols "
                     f"{cols.min()}-{cols.max()})")
        lines.append(f"{i}. {action_label(action)}\n{patch}")
    return "\n".join(lines)


def _task_id(game_id, family, prompt):
    return f"{game_id}-{family}-{hashlib.sha1(prompt.encode()).hexdigest()[:12]}"


def _fence(body, lang="text"):
    return f"```{lang}\n{body}\n```"


# --------------------------------------------------------------------------------------------------
# Families


def make_predict(pool, rng, *, patch_fraction=0.5, keep_unchanged=0.2, max_actions=5):
    state = pool.random_state()
    if state is None:
        return None
    level = state.level_index
    sim, seq = pool.sim(state), []
    for _ in range(rng.randint(1, max_actions)):
        action = pool.pick(sim, level)
        if action is None or not pool.act(sim, action, level)[0]:
            return None
        seq.append(action)
    changed = int((sim.board != state.board).sum())
    inner = substantive(state.board, sim.board)
    if not inner and rng.random() > keep_unchanged:
        return None  # no change, or only the border (timer) changed: kept as a small calibration share
    start_text, end_text = board_text(state.board), board_text(sim.board)
    patch = patch_text(state.board, sim.board)
    fmt = "patch" if rng.random() < patch_fraction and len(patch) < len(end_text) else "board"
    labels = ", ".join(action_label(a) for a in seq)
    ask = ("Answer with the complete resulting board (64 lines of 64 letters) in a final ```text block."
           if fmt == "board" else
           "Answer with a patch against the current board that covers every cell that changes (or the single "
           "line NO CHANGE if nothing changes) in a final ```text block.")
    prompt = "\n\n".join([
        PREAMBLE, _level_lines(level, pool.available(level)), _history_lines(state),
        f"Current board:\n{start_text}",
        f"The following {'action is' if len(seq) == 1 else f'{len(seq)} actions are'} taken next, in order: "
        f"{labels}.\nPredict the board after the last action. {ask}",
    ])
    return {
        "id": _task_id(pool.game_id, "predict", prompt), "family": "predict", "game": pool.game_id,
        "level": level + 1, "prompt": prompt, "answer": _fence(end_text if fmt == "board" else patch),
        "verifier": {"kind": "predict", "format": fmt, "start": start_text, "expected": end_text},
        "meta": {"format": fmt, "n_actions": len(seq), "changed_cells": changed, "substantive": inner,
                 "walk": len(state.prefix), "actions": [action_json(a) for a in seq],
                 "replay": {"level_index": level, "prefix": [action_json(a) for a in state.prefix]}},
    }


def _reaches(pool, state, path, target):
    sim = pool.sim(state)
    return all(sim.step(a) for a in path) and np.array_equal(sim.board, target)



def make_reach(pool, rng, *, max_k=6):
    """A walk of 1-``max_k`` actions (each changing the board inside its border, never back to a board
    already on the walk) gives the target; actions whose removal still reaches it are dropped, and
    the answer may use at most as many actions as the shortened walk."""
    state = pool.random_state()
    if state is None:
        return None
    level = state.level_index
    length = rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6][: 2 * max_k - 1])
    sim, walk, seen = pool.sim(state), [], {board_key(state.board)}
    for _ in range(3 * length):
        if len(walk) == length:
            break
        action = pool.pick(sim, level)
        if action is None:
            return None
        alive, before = pool.act(sim, action, level)
        if alive and substantive(before, sim.board) and board_key(sim.board) not in seen:
            walk.append(action)
            seen.add(board_key(sim.board))
        else:  # roll back by replay (cheaper than a deep copy per trial)
            sim = pool.sim(state, walk)
    target = sim.board
    if not walk or not substantive(state.board, target):
        return None
    path = list(walk)
    i = 0
    while len(path) > 1 and i < len(path):
        shorter = path[:i] + path[i + 1:]
        if _reaches(pool, state, shorter, target):
            path = shorter
        else:
            i += 1
    k = len(path)
    patch = patch_text(state.board, target)
    answer = json.dumps([action_json(a) for a in path])
    prompt = "\n\n".join([
        PREAMBLE, _level_lines(level, pool.available(level)), _history_lines(state),
        f"Current board:\n{board_text(state.board)}",
        f"Target board, as a patch against the current board:\n{patch}",
        f"Find a sequence of at most {k} action{'s' if k > 1 else ''} that turns the current board into exactly "
        "the target board (the level must not end or restart on the way). Answer with a JSON list in a final "
        '```json block, for example ["UP", {"action": "MOUSE", "row": 12, "col": 30}].',
    ])
    return {
        "id": _task_id(pool.game_id, "reach", prompt), "family": "reach", "game": pool.game_id,
        "level": level + 1, "prompt": prompt, "answer": _fence(answer, "json"),
        "verifier": {"kind": "reach", "game_id": pool.game_id, "level_index": level,
                     "prefix": [action_json(a) for a in state.prefix], "target": board_text(target),
                     "max_actions": k},
        "meta": {"k": k, "walk_k": len(walk), "changed_cells": int((target != state.board).sum()),
                 "walk": len(state.prefix)},
    }


ALLOWED_IMPORTS = ("bisect", "collections", "copy", "fractions", "functools", "heapq", "itertools", "json", "math",
                   "operator", "random", "re", "statistics", "string", "numpy")

STEP_SPEC = (
    "Write a Python function `step(board, action)` that returns the board after the action. `board` is a "
    "list of 64 strings of 64 letters (row 0 first; `board[r][c]` is row r, column c). `action` is a dict: "
    '{"action": "UP"} for a key, {"action": "MOUSE", "row": R, "col": C} for a click. Return the new '
    "board in the same form (a list of 64 strings of 64 letters). The function must be deterministic, must not "
    "print, and may import only: " + ", ".join(ALLOWED_IMPORTS) + ". It will be tested on other boards of this "
    "level. Answer with the code (helpers allowed) in a final ```python block."
)


def make_transition(pool, rng, *, n_held=4, min_changing=2):
    level = rng.randrange(pool.n_levels)
    states = list(pool.states(level))
    n_show = rng.randint(3, 6)
    if len(states) < n_show + n_held:
        return None
    avail = pool.available(level)
    kinds = [a[0] for a in key_actions(avail)] + (["ACTION6"] if 6 in avail else [])
    if not kinds:
        return None
    kind = rng.choice(kinds)
    rng.shuffle(states)
    changed, tried = pool.changed[level], pool.tried[level]
    pairs, afters = [], {}
    for state in states:
        if kind == "ACTION6":
            clicks = [a for a in candidate_actions(state.board, avail, rng, click_limit=pool.click_limit,
                                                   random_click=0) if a[0] == "ACTION6"]
            # clicks on colours that changed the board before first, random order otherwise
            clicks.sort(key=lambda a: -((changed[action_kind(a, state.board)] + 1)
                                        / (tried[action_kind(a, state.board)] + 2) + rng.random() * 0.2))
            chosen = None
            for action in clicks[:3]:
                trial = pool.sim(state)
                if pool.act(trial, action, level)[0]:
                    chosen = chosen or (trial, action)
                    if substantive(state.board, trial.board):
                        chosen = (trial, action)
                        break
            if chosen is None:
                continue
            trial, action = chosen
        else:
            action, trial = (kind, None, None), pool.sim(state)
            if not pool.act(trial, action, level)[0]:
                continue
        key = (board_key(state.board), action)
        if key in afters:
            if afters[key] != board_key(trial.board):
                return None  # the board does not determine the outcome (hidden state)
            continue
        afters[key] = board_key(trial.board)
        pairs.append((state.board, action, trial.board, state.prefix))
        if len(pairs) >= n_show + n_held + 2:
            break
    changing = [p for p in pairs if substantive(p[0], p[2])]
    still = [p for p in pairs if not substantive(p[0], p[2])]
    if len(pairs) < n_show + n_held or len(changing) < 2 * min_changing:
        return None
    if len({patch_text(b, a) for b, _, a, _p in changing}) == 1:
        return None  # one constant patch explains every pair: trivial
    rng.shuffle(changing)
    held, shown = changing[:min_changing], changing[min_changing:2 * min_changing]
    rest = changing[2 * min_changing:] + still
    rng.shuffle(rest)
    fill = n_show - len(shown)
    shown, rest = shown + rest[:fill], rest[fill:]
    held += rest[: n_held - len(held)]
    rng.shuffle(shown)
    if len(shown) != n_show or len(held) != n_held:
        return None
    label = ENGINE_TO_MODEL.get(kind, kind)
    blocks = []
    for i, (before, action, after, _p) in enumerate(shown, start=1):
        blocks.append(f"Example {i}: action {action_label(action)}\nBefore:\n{board_text(before)}\n"
                      f"After, as a patch against the board before:\n{patch_text(before, after)}")
    prompt = "\n\n".join([
        PREAMBLE, f"Level {level + 1}. Below are {len(shown)} observed transitions of the action {label} on this "
        "level, each from a different board.", *blocks, STEP_SPEC,
    ])
    cases = [{"board": board_text(b), "action": action_dict(a), "expected": board_text(af)} for b, a, af, _p in held]
    return {
        "id": _task_id(pool.game_id, "transition", prompt), "family": "transition", "game": pool.game_id,
        "level": level + 1, "prompt": prompt, "answer": None,
        "verifier": {"kind": "transition", "action": label, "cases": cases, "timeout": 10.0},
        "meta": {"n_shown": len(shown), "n_held": len(held),
                 "shown_substantive": sum(1 for b, _, a, _p in shown if substantive(b, a)),
                 "held_substantive": sum(1 for b, _, a, _p in held if substantive(b, a)),
                 "held_unchanged": sum(1 for c in cases if c["board"] == c["expected"]),
                 "replay": {"level_index": level,
                            "shown": [[[action_json(x) for x in p], action_json(a)] for _, a, _, p in shown],
                            "held": [[[action_json(x) for x in p], action_json(a)] for _, a, _, p in held]}},
    }


MAKERS = {"predict": make_predict, "reach": make_reach, "transition": make_transition}


# --------------------------------------------------------------------------------------------------
# Sandbox and verification

_SANDBOX = r'''
import builtins, json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (__CPU__, __CPU__ + 1))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (__MEM__, __MEM__))
    except (ValueError, OSError):
        pass
except ImportError:
    pass
import numpy
ALLOWED = set(__ALLOWED__)
_real_import = builtins.__import__
def _guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if globals is None or globals.get("__name__") == "__candidate__":
        if level or name.split(".")[0] not in ALLOWED:
            raise ImportError("import of %r is not allowed" % name)
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _guarded
payload = json.loads(sys.stdin.read())
safe = dict(vars(builtins))
for name in ("open", "input", "exit", "quit", "breakpoint", "help", "compile", "exec", "eval", "globals",
             "locals", "vars", "memoryview"):
    safe.pop(name, None)
ns = {"__name__": "__candidate__", "__builtins__": safe}
def emit(obj):
    sys.stdout.write("\n@@AUX_RESULT@@" + json.dumps(obj) + "\n")
    sys.stdout.flush()
try:
    exec(payload["code"], ns)
    step = ns["step"]
except BaseException as exc:
    emit({"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
    sys.exit(0)
out = []
for case in payload["cases"]:
    try:
        result = step(case["board"].split("\n"), dict(case["action"]))
        if isinstance(result, str):
            rows = result.split("\n")
        else:
            rows = ["".join(str(ch) for ch in row) if not isinstance(row, str) else row for row in list(result)]
        out.append("\n".join(rows))
    except BaseException as exc:
        out.append({"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
emit({"results": out})
'''


def run_step_program(code, cases, *, timeout=10.0, memory_bytes=2 << 30):
    """Run ``step`` from ``code`` on ``cases`` (``{"board", "action"}``) in a separate interpreter
    (isolated mode, empty working directory, CPU/memory limits, import allowlist). Returns
    ``{"results": [board text or {"error"}]}``, ``{"error"}`` or ``{"timeout": True}``."""
    script = (_SANDBOX.replace("__CPU__", str(int(timeout) + 1)).replace("__MEM__", str(int(memory_bytes)))
              .replace("__ALLOWED__", repr(list(ALLOWED_IMPORTS))))
    payload = json.dumps({"code": code, "cases": [{"board": c["board"], "action": c["action"]} for c in cases]})
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(prefix="aux-sandbox-") as cwd:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", script], input=payload, capture_output=True,
                                  text=True, timeout=timeout, cwd=cwd, env=env)
        except subprocess.TimeoutExpired:
            return {"timeout": True}
    marker = proc.stdout.rfind("@@AUX_RESULT@@")
    if marker < 0:
        return {"error": f"no result (exit {proc.returncode}): {proc.stderr[-300:]}"}
    try:
        return json.loads(proc.stdout[marker + len("@@AUX_RESULT@@"):].strip())
    except ValueError:
        return {"error": "unreadable result"}


def _verify_predict(task, output):
    ver = task["verifier"]
    start = parse_board(ver["start"])
    got = read_board_answer(extract_answer(output, ("text", "")), start)
    return got is not None and board_text(got) == ver["expected"]


def _verify_reach(task, output, library):
    ver = task["verifier"]
    actions = parse_action_list(extract_answer(output, ("json",)))
    if not actions or len(actions) > int(ver["max_actions"]):
        return False
    factory = (library or default_library()).factory(ver["game_id"])
    sim = Sim.replay(factory, int(ver["level_index"]), [parse_action(a) for a in ver["prefix"]])
    names = {f"ACTION{i}" for i in sim.available}
    for action in actions:
        if action[0] not in names or not sim.step(action):
            return False
    return board_text(sim.board) == ver["target"]


def _verify_transition(task, output, timeout=None):
    ver = task["verifier"]
    code = extract_answer(output, ("python", "py"))
    cases = ver["cases"]
    result = run_step_program(code, cases, timeout=float(timeout or ver.get("timeout", 10.0)))
    got = result.get("results")
    return bool(got) and len(got) == len(cases) and all(g == c["expected"] for g, c in zip(got, cases))


def verify(task, model_output, *, library=None, timeout=None):
    """True when ``model_output`` (the full model reply; the last fenced block is read) solves ``task``."""
    if model_output is None:
        return False
    family = task["family"]
    try:
        if family == "predict":
            return _verify_predict(task, model_output)
        if family == "reach":
            return _verify_reach(task, model_output, library)
    except (ValueError, KeyError, TypeError, RuntimeError):
        return False
    if family == "transition":
        return _verify_transition(task, model_output, timeout)
    raise ValueError(f"unknown family {family!r}")


def oracle_program(task):
    """A lookup-table ``step`` over the held-out cases: it proves the sandbox and comparison accept a
    correct function. Not a training target."""
    table = {json.dumps([c["board"], c["action"]], sort_keys=True): c["expected"] for c in task["verifier"]["cases"]}
    return ("import json\nTABLE = " + repr(table) + "\n\ndef step(board, action):\n"
            "    key = json.dumps(['\\n'.join(board), action], sort_keys=True)\n"
            "    return TABLE[key].split('\\n')\n")


IDENTITY_PROGRAM = "def step(board, action):\n    return list(board)\n"


def replay_check(task, library=None):
    """Every board a task states is reproduced by replaying its level from a fresh game instance
    (the recorded ``meta.replay`` prefixes): guards against games whose deep copies diverge."""
    info = task.get("meta", {}).get("replay")
    if not info:
        return task["family"] == "reach"  # reach verification always replays
    factory = (library or default_library()).factory(task["game"])
    ver = task["verifier"]

    def run(prefix, actions):
        sim = Sim.replay(factory, int(info["level_index"]), [parse_action(a) for a in prefix])
        before = sim.board
        alive = all(sim.step(parse_action(a)) for a in actions)
        return alive, before, sim.board

    try:
        if task["family"] == "predict":
            alive, before, after = run(info["prefix"], task["meta"]["actions"])
            return alive and board_text(before) == ver["start"] and board_text(after) == ver["expected"]
        for (prefix, action), case in zip(info["held"], ver["cases"], strict=True):
            alive, before, after = run(prefix, [action])
            if not (alive and board_text(before) == case["board"] and board_text(after) == case["expected"]):
                return False
        for prefix, action in info["shown"]:
            alive, before, after = run(prefix, [action])
            if not alive or (f"Before:\n{board_text(before)}\nAfter, as a patch against the board before:\n"
                             f"{patch_text(before, after)}") not in task["prompt"]:
                return False
    except (RuntimeError, ValueError, KeyError):
        return False
    return True


def self_check(task, *, library=None):
    """The task's boards replay from a fresh game, the reference (or, for ``transition``, the oracle)
    passes and a trivial wrong answer fails."""
    ver = task["verifier"]
    if not replay_check(task, library):
        return False
    if task["family"] == "transition":
        wrong = "```python\n" + IDENTITY_PROGRAM + "```"
        return verify(task, _fence(oracle_program(task), "python"), library=library) and not verify(
            task, wrong, library=library)
    if not verify(task, task["answer"], library=library):
        return False
    if task["family"] == "predict":
        if ver["start"] == ver["expected"]:
            return True
        return not verify(task, _fence(ver["start"]), library=library)
    return not verify(task, _fence("[]", "json"), library=library)


# --------------------------------------------------------------------------------------------------
# Generation


def generate(game_id, factory, n_per_family, *, seed=0, families=FAMILIES, library=None, check=True,
             max_attempts_factor=8, pool_kwargs=None, maker_kwargs=None):
    """Up to ``n_per_family`` self-checked tasks per family for one game (no duplicate prompts).
    Returns ``(tasks, stats)``."""
    rng = random.Random(f"{seed}:{game_id}")
    pool = StatePool(game_id, factory, rng, **(pool_kwargs or {}))
    library = library or GameLibrary(factories={game_id: factory})
    tasks, stats = [], {}
    seen = set()
    replay_failures = 0
    for family in families:
        started = time.monotonic()
        made = attempts = rejected = failed_check = duplicates = 0
        while made < n_per_family and attempts < n_per_family * max_attempts_factor:
            attempts += 1
            task = MAKERS[family](pool, rng, **((maker_kwargs or {}).get(family, {})))
            if task is None:
                rejected += 1
                continue
            if task["prompt"] in seen:
                duplicates += 1
                continue
            if check and not self_check(task, library=library):
                failed_check += 1
                if pool.clone_ok and not replay_check(task, library):
                    replay_failures += 1
                    if replay_failures >= 2:  # deep copies of this game diverge: replay only
                        pool.distrust_copies()
                continue
            seen.add(task["prompt"])
            tasks.append(task)
            made += 1
        stats[family] = {"made": made, "attempts": attempts, "rejected": rejected, "duplicates": duplicates,
                         "failed_self_check": failed_check, "seconds": round(time.monotonic() - started, 2)}
    stats["pool"] = {"clone_faithful": pool.clone_ok, "replay_failures_before_switch": replay_failures,
                     "slow_engine": pool.slow, "levels": pool.n_levels,
                     "states": {lvl + 1: len(states) for lvl, states in sorted(pool._states.items())}}
    return tasks, stats
