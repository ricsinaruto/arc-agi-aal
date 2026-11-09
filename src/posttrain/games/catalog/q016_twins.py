"""q016 twins: steer two avatars at once, one of them mirrored, onto the two green cells.

Hidden rules: every arrow moves the blue avatar one cell in the arrow's direction and the orange avatar
one cell in the mirrored direction. Levels 1-2: the orange avatar mirrors left-right only (LEFT moves
it right, RIGHT moves it left, UP and DOWN move it the same way as blue). From level 3 it mirrors both
axes (it always moves opposite to blue). Each avatar that would walk into a dark-gray wall, off the
board, or into the other avatar's current cell stays where it is while the other still moves. From
level 4 charcoal pits appear: an avatar that steps into a pit falls in and the game is over (RESET
restarts the level). The level clears when the two avatars stand on the two green cells (either
avatar on either cell).
"""

NAME = "twins"
TAGS = ["keyboard", "symmetry", "multi-avatar", "hazard", "rule-change"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "@": 5, "&": 5, "X": 14, "x": 4}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_blue = where["@"][0]
        self.s_orange = where["&"][0]
        self.goals = frozenset(where["X"])
        self.pits = frozenset(where.get("x", []))
        self.mirror = (-1, -1) if spec["mirror"] == "xy" else (-1, 1)

    def state_key(self):
        return (self.s_blue, self.s_orange)

    def render_grid(self):
        out = self.grid.copy()
        out[self.s_blue[1], self.s_blue[0]] = 9
        out[self.s_orange[1], self.s_orange[0]] = 12
        return out

    def step_to(self, pos, delta, other):
        p = (pos[0] + delta[0], pos[1] + delta[1])
        if not self.inside(*p) or self.grid[p[1], p[0]] == 3 or p == other:
            return pos
        return p

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        blue = self.step_to(self.s_blue, (dx, dy), self.s_orange)
        orange = self.step_to(self.s_orange, (dx * self.mirror[0], dy * self.mirror[1]), self.s_blue)
        if blue == orange:  # both stepped into the same free cell: neither moves
            blue, orange = self.s_blue, self.s_orange
        self.s_blue, self.s_orange = blue, orange
        if blue in self.pits or orange in self.pits:
            return "lose"
        return "win" if {blue, orange} == set(self.goals) else None


TIER_ACTIONS = [(6, 14), (7, 18), (7, 18), (8, 22), (9, 26)]
# (interior width, interior height, wall density, pits, mirror axes)
TIER_SHAPES = [((6, 5), 0.2, 0, "x"), ((7, 6), 0.25, 0, "x"), ((7, 6), 0.25, 0, "xy"), ((7, 6), 0.2, 3, "x"),
               ((8, 6), 0.2, 4, "xy")]


def generate_level(rng, tier):
    """Random walls, pits, avatars and goals; the admission search keeps only solvable, non-trivial
    levels (walls are what let the two avatars move out of step)."""
    (w, h), density, pits, mirror = TIER_SHAPES[tier]
    cells = [(x, y) for y in range(h) for x in range(w)]
    rng.shuffle(cells)
    blue, orange, goal_a, goal_b = cells[:4]
    rest = cells[4:]
    wall = {c for c in rest if rng.random() < density}
    pit = set([c for c in rest if c not in wall][:pits])
    rows = []
    for y in range(h):
        row = ""
        for x in range(w):
            p = (x, y)
            row += ("@" if p == blue else "&" if p == orange else "X" if p in (goal_a, goal_b) else "#" if p in wall
                    else "x" if p in pit else ".")
        rows.append("#" + row + "#")
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)], "mirror": mirror}
