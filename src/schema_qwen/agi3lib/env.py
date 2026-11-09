"""Environments for the Schema CLI: the game over our bridge (``rpc``) and the toy game.

Both expose the same two calls, ``reset() -> Frame`` and ``act(action, x=None, y=None)
-> Frame``, and are resumable across CLI invocations via ``state_dict()`` /
``from_state()``, because every ``arc3.py`` command is a separate process. ``Frame`` and
``ToyEnv`` are verbatim from open-schema's ``agi3lib/env.py``; the live-API, replay and
toolkit environments are replaced by ``RpcEnv``, which talks to the parent process's game
through the sandbox socket (``arc_api``), so no replay is needed: the engine keeps state.

Action numbering follows the ARC-AGI-3 wire format: 0 RESET, 1..5 simple actions, 6 click
(x, y in 0..63), 7 undo.
"""

RESET = 0


class Frame:
    """What the environment returns after RESET or an action."""

    def __init__(self, grid, state, level, win_levels, legal, frames=None):
        self.grid = grid  # last grid of the frame list
        self.frames = frames if frames is not None else [grid]
        self.state = state  # NOT_STARTED | NOT_FINISHED | WIN | GAME_OVER
        self.level = level  # levels completed so far
        self.win_levels = win_levels
        self.legal = legal  # available action ids

# --- Toy game -----------------------------------------------------------------
# A deterministic 64x64 three-level game for exercising the whole loop offline:
# move with 1..4, pick up a key to open the door, avoid the void, reach the
# goal. Small enough that a correct world model fits in a page, rich enough to
# produce level_up / dead / win flags and click no-ops.

FLOOR, WALL, DOOR, GOAL, KEY, AVATAR, VOID = 0, 4, 6, 7, 8, 9, 10
TOY_WIN_LEVELS = 3

# (avatar r, c), (goal r, c), walls, door cell or None, key cell or None, voids
TOY_LEVELS = [
    ((5, 5), (20, 20), [], None, None, []),
    ((5, 5), (20, 40), [(r, 30) for r in range(0, 64) if r != 12],
     (12, 30), (18, 10), []),
    ((5, 5), (40, 40), [], None, None, [(30, c) for c in range(0, 60)]),
]


class ToyEnv:
    def __init__(self):
        self.level = 0
        self.pos = TOY_LEVELS[0][0]
        self.have_key = False
        self.state = "NOT_STARTED"
        self.win_levels = TOY_WIN_LEVELS

    def _spec(self):
        return TOY_LEVELS[min(self.level, TOY_WIN_LEVELS - 1)]

    def _grid(self):
        _, goal, walls, door, key, voids = self._spec()
        g = [[FLOOR] * 64 for _ in range(64)]
        for c in range(64):
            g[0][c] = g[63][c] = WALL
        for r in range(64):
            g[r][0] = g[r][63] = WALL
        for r, c in walls:
            g[r][c] = WALL
        for r, c in voids:
            g[r][c] = VOID
        if door and not self.have_key:
            g[door[0]][door[1]] = DOOR
        if key and not self.have_key:
            g[key[0]][key[1]] = KEY
        g[goal[0]][goal[1]] = GOAL
        r, c = self.pos
        g[r][c] = AVATAR
        return g

    def _frame(self):
        return Frame(self._grid(), self.state, self.level, self.win_levels,
                     [1, 2, 3, 4, 6] if self.state == "NOT_FINISHED" else [0])

    def reset(self):
        # Matches the real games' observed semantics: RESET restarts the
        # CURRENT level (recovery after GAME_OVER, or a voluntary retry);
        # only a fresh game or a finished one restarts from level 0.
        if self.state in ("NOT_STARTED", "WIN"):
            self.level = 0
        self.have_key = False
        self.pos = self._spec()[0]
        self.state = "NOT_FINISHED"
        return self._frame()

    def act(self, action, x=None, y=None):
        if action == RESET:
            return self.reset()
        if self.state != "NOT_FINISHED":
            raise ValueError(f"game is {self.state}; only RESET is legal")
        _, goal, walls, door, key, voids = self._spec()
        dr = {1: -1, 2: 1}.get(action, 0)
        dc = {3: -1, 4: 1}.get(action, 0)
        if dr or dc:
            r, c = self.pos[0] + dr, self.pos[1] + dc
            cell_is_wall = ((r, c) in set(walls) or r in (0, 63) or c in (0, 63)
                            or ((r, c) == door and not self.have_key))
            if not cell_is_wall:
                self.pos = (r, c)
                if (r, c) == key and not self.have_key:
                    self.have_key = True
                if (r, c) in set(voids):
                    self.state = "GAME_OVER"
                elif (r, c) == goal:
                    self.level += 1
                    if self.level >= self.win_levels:
                        self.state = "WIN"
                    else:
                        self.have_key = False
                        self.pos = self._spec()[0]
        # action 6 (click) and 5/7 are deliberate no-ops in the toy game
        return self._frame()

    def state_dict(self):
        return {"level": self.level, "pos": list(self.pos),
                "have_key": self.have_key, "state": self.state}

    def from_state(self, st):
        self.level = st["level"]
        self.pos = (st["pos"][0], st["pos"][1])
        self.have_key = st["have_key"]
        self.state = st["state"]


class RpcEnv:
    """The parent process's game, reached through the ``arc_api`` socket client.

    The observation carries the frame list (animation frames), the settled grid, the level
    counters and the legal actions; ``frames`` keeps every animation frame as upstream's
    ToolkitEnv did.
    """

    def __init__(self):
        import arc_api  # staged next to this package in the sandbox api directory

        self.api = arc_api

    @staticmethod
    def _frame(obs):
        frames = obs.get("frames") or [obs["grid"]]
        return Frame(
            grid=[[int(v) for v in row] for row in obs["grid"]],
            state=str(obs["status"]),
            level=int(obs["levels_completed"]),
            win_levels=int(obs["win_levels"]),
            legal=[0 if name == "RESET" else int(name.removeprefix("ACTION"))
                   for name in obs["legal_actions"]],
            frames=[[[int(v) for v in row] for row in frame] for frame in frames],
        )

    def current(self):
        return self._frame(self.api.observe(grid=True))

    def reset(self):
        return self._frame(self.api.step("RESET", grid=True))

    def act(self, action, x=None, y=None):
        if action == RESET:
            return self.reset()
        name = f"ACTION{int(action)}"
        return self._frame(self.api.step(name, x, y, grid=True))

    def state_dict(self):
        return {}

    def from_state(self, st):
        return


def make_env(run_cfg, env_state=None, history=None):
    """Build the environment named by run.json, restored to saved state."""
    kind = run_cfg.get("env", "toy")
    if kind == "toy":
        env = ToyEnv()
    elif kind == "rpc":
        env = RpcEnv()
    else:
        raise ValueError(f"unknown env kind {kind!r}")
    if env_state:
        env.from_state(env_state)
    return env
