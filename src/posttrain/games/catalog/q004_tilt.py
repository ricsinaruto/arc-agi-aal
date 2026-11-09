"""q004 tilt: tilt the board so the red block comes to rest on the green cell.

Hidden rules: every arrow slides all red and blue blocks as far as they go in that direction
(blocks stop at walls and at other blocks). Only resting on green counts, passing over it does not.
From level 3, charcoal pits appear: a blue block that slides into a pit fills it (the pit becomes
floor); the red block falling into a pit is a game over (RESET restarts the level).
"""

NAME = "tilt"
TAGS = ["keyboard", "sliding", "hazard"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "r": 5, "b": 5, "G": 5, "x": 5}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_red = where["r"][0]
        self.s_blue = frozenset(where.get("b", []))
        self.s_pits = frozenset(where.get("x", []))
        self.goal = where["G"][0]

    def render_grid(self):
        out = self.grid.copy()
        out[self.goal[1], self.goal[0]] = 14
        for x, y in self.s_pits:
            out[y, x] = 4
        for x, y in self.s_blue:
            out[y, x] = 9
        out[self.s_red[1], self.s_red[0]] = 8
        return out

    def state_key(self):
        return (self.s_red, self.s_blue, self.s_pits)

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        blocks = [("r", self.s_red)] + [("b", p) for p in self.s_blue]
        blocks.sort(key=lambda item: -(item[1][0] * dx + item[1][1] * dy))  # leading blocks first
        occupied = {p for _, p in blocks}
        pits = set(self.s_pits)
        red, blue, lost = self.s_red, set(), False
        for kind, (x, y) in blocks:
            occupied.discard((x, y))
            while True:
                nx, ny = x + dx, y + dy
                if not self.inside(nx, ny) or self.grid[ny, nx] == 3 or (nx, ny) in occupied:
                    break
                x, y = nx, ny
                if (x, y) in pits:
                    break
            if (x, y) in pits:
                if kind == "r":
                    lost = True
                    red = (x, y)
                else:
                    pits.discard((x, y))
                continue
            occupied.add((x, y))
            if kind == "r":
                red = (x, y)
            else:
                blue.add((x, y))
        self.s_red, self.s_blue, self.s_pits = red, frozenset(blue), frozenset(pits)
        if lost:
            return "lose"
        return "win" if self.s_red == self.goal else None


TIER_ACTIONS = [(4, 8), (5, 10), (6, 12), (7, 14), (8, 16)]
TIER_SHAPES = [((6, 5), 3, 0, 0), ((7, 5), 4, 1, 0), ((7, 6), 5, 1, 1), ((8, 6), 6, 2, 2), ((8, 7), 7, 2, 3)]


def generate_level(rng, tier):
    """Random walls, blue blocks and pits; the admission search keeps only solvable, non-trivial boards."""
    (w, h), walls, blues, pits = TIER_SHAPES[tier]
    cells = [(x, y) for y in range(h) for x in range(w)]
    rng.shuffle(cells)
    wall = set(cells[:walls])
    rest = cells[walls:]
    red, goal = rest[0], rest[1]
    blue = set(rest[2:2 + blues])
    pit = set(rest[2 + blues:2 + blues + pits])
    rows = []
    for y in range(h):
        row = ""
        for x in range(w):
            p = (x, y)
            row += ("#" if p in wall else "r" if p == red else "G" if p == goal else "b" if p in blue
                    else "x" if p in pit else ".")
        rows.append("#" + row + "#")
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)]}
