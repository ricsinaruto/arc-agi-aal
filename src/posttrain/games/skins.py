"""Skins: the same rules and levels shown through a different observation (paper-notes 2.7, "rules /
observation / skin layers"; relabelling alone costs ARC solvers a large share of their score).

A skin of catalog game ``qNNN`` is the game ``q1NN``. Its file inlines the base game's rules unchanged
and adds ``SkinLayer``, which changes only what the player sees and how input maps back:

- the board is mirrored left-right (optionally) and then turned by 0-3 clockwise quarter turns;
- every colour index is replaced through a permutation of the 16 colours;
- direction keys (ACTION1-4 = up, down, left, right, when the base game uses them as directions) are
  remapped so that each key moves things the way it looks on screen, and the skin's available actions
  are the displayed images of the base game's;
- a click on a displayed cell is mapped back to the base cell.

Because every displayed state and action corresponds one-to-one to a base state and action, a skin
keeps the base game's solvability and optimal action counts exactly. Each skin level spec is
``{"map": <the base map turned like the board>, "base": <the base level spec>}``.
"""

import ast
import random
from pathlib import Path

import numpy as np
from arcengine import GameAction

HERE = Path(__file__).resolve().parent
STANDARD_MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
DIRECTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}
COLOUR_NAMES = ("white", "light gray", "gray", "dark gray", "charcoal", "black", "magenta", "pink", "red",
                "blue", "sky blue", "yellow", "orange", "dark red", "green", "purple")


class SkinLayer:
    """Mixed in before the base game class (``class Q101(SkinLayer, Game)``); see the module docstring."""

    SKIN_FLIP = False
    SKIN_TURNS = 0
    SKIN_PALETTE = list(range(16))
    SKIN_KEYS = {}

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.camera.background = self.SKIN_PALETTE[self.BACKGROUND]
        self.camera.letter_box = self.SKIN_PALETTE[self.BACKGROUND]

    def skin_cell(self, x, y):
        """Base cell -> displayed cell."""
        w, h = self.skin_base_size
        if self.SKIN_FLIP:
            x = w - 1 - x
        for _ in range(self.SKIN_TURNS):
            x, y, w, h = h - 1 - y, x, h, w
        return x, y

    def load(self, spec):
        rows = spec["base"]["map"]
        self.skin_base_size = (len(rows[0]), len(rows))
        super().load(spec["base"])
        self.skin_cells = {self.skin_cell(x, y): (x, y) for y in range(len(rows)) for x in range(len(rows[0]))}

    def render_grid(self):
        out = np.asarray(self.SKIN_PALETTE)[super().render_grid()]
        if self.SKIN_FLIP:
            out = out[:, ::-1]
        return np.rot90(out, -self.SKIN_TURNS).copy()

    def solver_clicks(self):
        cells = super().solver_clicks() if hasattr(super(), "solver_clicks") else None
        return None if cells is None else [self.skin_cell(*cell) for cell in cells]

    def apply(self, action, cell):
        if action == 6:
            cell = self.skin_cells.get(tuple(cell))
            if cell is None:
                return None
        else:
            action = self.SKIN_KEYS.get(action, action)
        return super().apply(action, cell)

    def step(self):
        # GridGame.step, except that a click is checked against the displayed board, not the base one.
        action = self.action.id
        if action == GameAction.RESET:
            self.complete_action()
            return
        cell = None
        if action == GameAction.ACTION6:
            data = self.action.data or {}
            cell = self.camera.display_to_grid(int(data.get("x", -1)), int(data.get("y", -1)))
            if cell is None or tuple(cell) not in self.skin_cells:
                self.complete_action()
                return
        result = self.apply(int(action.value), cell)
        self.redraw()
        if result == "win":
            self.next_level()
        elif result == "lose":
            self.lose()
        self.complete_action()


# -- building skins -----------------------------------------------------------------------------

def skin_prefix(base_prefix):
    number = int(base_prefix[1:])
    if not 1 <= number <= 99:
        raise ValueError(f"only catalog games q001-q099 get skins, got {base_prefix}")
    return f"q{100 + number:03d}"


def turn_vector(dx, dy, flip, turns):
    if flip:
        dx = -dx
    for _ in range(turns):
        dx, dy = -dy, dx
    return dx, dy


def make_skin(prefix, moves=None, actions=(1, 2, 3, 4)):
    """Seeded skin parameters for ``prefix``: a non-identity orientation, a colour permutation that
    moves every colour, and the key remapping when the base game uses ACTION1-4 as directions."""
    rng = random.Random(f"skin-{prefix}")
    flip, turns = rng.choice([(f, t) for f in (False, True) for t in range(4) if (f, t) != (False, 0)])
    palette = list(range(16))
    while any(i == c for i, c in enumerate(palette)):
        rng.shuffle(palette)
    keys = {}
    if moves == STANDARD_MOVES and any(a in STANDARD_MOVES for a in actions):
        for base, (dx, dy) in STANDARD_MOVES.items():
            shown = turn_vector(dx, dy, flip, turns)
            displayed = next(a for a, v in STANDARD_MOVES.items() if v == shown)
            keys[displayed] = base
    shown_actions = sorted({next((d for d, b in keys.items() if b == a), a) for a in actions})
    return {"flip": flip, "turns": turns, "palette": palette, "keys": keys, "actions": shown_actions}


def turn_map(rows, flip, turns):
    grid = np.array([list(row) for row in rows])
    if flip:
        grid = grid[:, ::-1]
    grid = np.rot90(grid, -turns)
    return ["".join(row) for row in grid]


def skin_levels(levels, skin):
    return [{"map": turn_map(spec["map"], skin["flip"], skin["turns"]), "base": spec} for spec in levels]


def describe(skin, base_prefix, base_name):
    """The paragraph appended to the base designer note: how this skin shows the base rules."""
    orientation = []
    if skin["flip"]:
        orientation.append("mirrored left-right")
    if skin["turns"]:
        orientation.append(f"turned {90 * skin['turns']} degrees clockwise")
    text = (f"Skin of {base_prefix} ({base_name}): the same rules and levels as above, shown differently. "
            f"The board is {' and then '.join(orientation)}, so every direction, side and axis named in "
            "the rules above refers to the unturned board.")
    if skin["keys"]:
        shown = {base: DIRECTION_NAMES[d] for d, base in skin["keys"].items()}
        text += (" The direction keys follow the screen: the base game's "
                 + ", ".join(f"{DIRECTION_NAMES[b]} appears as {shown[b]}" for b in (1, 2, 3, 4))
                 + ", and the key for the screen direction moves things that way (for example UP moves "
                   "the base game's " + DIRECTION_NAMES[skin["keys"][1]] + ").")
    text += " Clicks act on the cell clicked on screen. Every colour is replaced: " + ", ".join(
        f"{COLOUR_NAMES[c]} -> {COLOUR_NAMES[skin['palette'][c]]}" for c in range(16)) + "."
    return text


def _class_source(name):
    tree = ast.parse((HERE / "skins.py").read_text())
    source = (HERE / "skins.py").read_text()
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    segment = ast.get_source_segment(source, node)
    body = node.body[0]
    if isinstance(body, ast.Expr) and isinstance(body.value, ast.Constant):  # drop the class docstring
        doc = ast.get_source_segment(source, body)
        segment = segment.replace("    " + doc + "\n", "", 1)
    return segment


def assemble_skin(module_path, prefix, base_levels, skin, base_meta):
    """(prefix, game source, notes metadata) for a skin of the catalog module at ``module_path``."""
    from posttrain.games import store

    base_prefix, base_source, _ = store.assemble(module_path)
    # The base assembly ends with "class QNNN(Game):\n    pass\n"; keep only gridbase + rules.
    rules = base_source[:base_source.rindex("\n\nclass ")]
    rules = rules.replace(f"# Generated ARC-like game {base_prefix} ", f"# Generated ARC-like game {prefix} ", 1)
    class_name = prefix[0].upper() + prefix[1:]
    levels = skin_levels(base_levels, skin)
    source = (f"{rules}\n\n\n{_class_source('SkinLayer')}\n\n\nclass {class_name}(SkinLayer, Game):\n"
              f"    ACTIONS = {skin['actions']!r}\n    SKIN_FLIP = {skin['flip']!r}\n"
              f"    SKIN_TURNS = {skin['turns']!r}\n    SKIN_PALETTE = {skin['palette']!r}\n"
              f"    SKIN_KEYS = {skin['keys']!r}\n    LEVELS = {levels!r}\n")
    meta = {"name": f"{base_meta['name']}-skin", "tags": list(base_meta["tags"]) + ["skin"],
            "note": base_meta["note"] + "\n\n" + describe(skin, base_prefix, base_meta["name"])}
    return prefix, source, meta
