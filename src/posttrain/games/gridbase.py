"""Base class for generated ARC-like games (inlined into every generated game file).

A generated game keeps its state in plain Python (a grid of colour indices plus whatever the rules
need), rebuilt from the level's text map on every level start or reset, and renders the grid as one
sprite. The camera is sized to the grid, so the engine scales it into the 64x64 frame exactly like
the public games. Subclasses define:

- ``LEVELS``: list of dicts, each with ``"map"`` (list of equal-length strings) and optional extras;
- ``LEGEND``: map character -> colour index (0-15);
- ``ACTIONS``: the available action ids (1-5 keys, 6 click, 7 undo-like key);
- ``load(spec)``: build ``self.grid`` (numpy int array, rows x cols) and the rule state;
- ``apply(action, cell)``: change the state for one action (``cell`` is ``(x, y)`` for ACTION6, else
  None) and return ``"win"``, ``"lose"`` or None;
- ``state_key()``: a hashable summary of the rule state (the solver and admission tests use it).

This file must stay self-contained (arcengine and numpy only): the store copies its source into
every game file, so a game loads through the unchanged ``arc_agi`` local loader on Kaggle.
"""

import copy

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, Level, Sprite


class GridGame(ARCBaseGame):
    GAME_ID = "q000-00000000"
    LEVELS = []
    LEGEND = {}
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 0

    def __init__(self, seed=0):
        levels = []
        for index, spec in enumerate(self.LEVELS):
            rows = spec["map"]
            levels.append(Level(sprites=[], grid_size=(len(rows[0]), len(rows)), data={"spec": spec},
                                name=f"level-{index + 1}"))
        first = self.LEVELS[0]["map"]
        camera = Camera(width=len(first[0]), height=len(first), background=self.BACKGROUND, letter_box=self.BACKGROUND)
        super().__init__(game_id=self.GAME_ID, levels=levels, camera=camera, available_actions=list(self.ACTIONS),
                         seed=seed)

    # -- helpers for subclasses -------------------------------------------------------------
    def parse(self, spec):
        """Grid of colours from the map, and the cell positions of every map character."""
        rows = spec["map"]
        grid = np.full((len(rows), len(rows[0])), self.BACKGROUND, dtype=int)
        where = {}
        for y, row in enumerate(rows):
            if len(row) != len(rows[0]):
                raise ValueError("map rows must have equal length")
            for x, ch in enumerate(row):
                grid[y, x] = self.LEGEND.get(ch, self.BACKGROUND)
                where.setdefault(ch, []).append((x, y))
        return grid, where

    def inside(self, x, y):
        return 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]

    # -- engine hooks -----------------------------------------------------------------------
    def on_set_level(self, level):
        spec = level.get_data("spec")
        rows = spec["map"]
        self.camera.resize(len(rows[0]), len(rows))
        self.load(copy.deepcopy(spec))
        self.redraw()

    def redraw(self):
        level = self.current_level
        level.remove_all_sprites()
        level.add_sprite(Sprite(pixels=self.render_grid().tolist(), name="board", x=0, y=0, layer=0,
                                blocking=BlockingMode.NOT_BLOCKED, collidable=False))

    def render_grid(self):
        return self.grid

    def step(self):
        action = self.action.id
        if action == GameAction.RESET:  # the engine steps RESET too; the level was already rebuilt
            self.complete_action()
            return
        cell = None
        if action == GameAction.ACTION6:
            data = self.action.data or {}
            cell = self.camera.display_to_grid(int(data.get("x", -1)), int(data.get("y", -1)))
            if cell is None or not self.inside(*cell):
                self.complete_action()
                return
        result = self.apply(int(action.value), cell)
        self.redraw()
        if result == "win":
            self.next_level()
        elif result == "lose":
            self.lose()
        self.complete_action()

    # -- solver interface -------------------------------------------------------------------
    def snapshot(self):
        return copy.deepcopy({k: v for k, v in self.__dict__.items() if k.startswith("s_") or k == "grid"})

    def restore(self, snap):
        for key, value in copy.deepcopy(snap).items():
            setattr(self, key, value)

    def state_key(self):
        return (self.grid.tobytes(),) + tuple(repr(getattr(self, k)) for k in sorted(self.__dict__) if k.startswith("s_"))

    def load(self, spec):
        raise NotImplementedError

    def apply(self, action, cell):
        raise NotImplementedError
