"""q002 lights: click cells until every board cell is yellow.

Hidden rules: a click on a board cell flips it and its neighbours between gray and yellow. Levels 1-2
flip the four orthogonal neighbours (plus pattern); from level 3 the flip pattern is the four
diagonal neighbours instead (the interaction changes between levels, as in the public games).
Clicks outside the board do nothing.
"""

NAME = "lights"
TAGS = ["click", "toggle", "rule-change"]

PATTERNS = {"plus": ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
            "cross": ((0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1))}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [6]
    BACKGROUND = 5
    LEGEND = {"#": 3, "o": 2, "x": 11}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pattern = spec["pattern"]
        self.board = set(where.get("o", [])) | set(where.get("x", []))

    def apply(self, action, cell):
        if cell not in self.board:
            return None
        for dx, dy in PATTERNS[self.s_pattern]:
            p = (cell[0] + dx, cell[1] + dy)
            if p in self.board:
                self.grid[p[1], p[0]] = 11 if self.grid[p[1], p[0]] == 2 else 2
        return "win" if all(self.grid[y, x] == 11 for x, y in self.board) else None


TIER_ACTIONS = [(4, 6), (5, 8), (6, 10), (7, 12), (8, 14)]
TIER_SHAPES = [((3, 3), "plus"), ((4, 3), "plus"), ((4, 4), "cross"), ((5, 3), "cross"), ((4, 4), "plus")]


def generate_level(rng, tier):
    """Start from the solved board and apply random clicks (solvable by construction)."""
    (w, h), pattern = TIER_SHAPES[tier]
    on = [[True] * w for _ in range(h)]
    for _ in range(rng.randint(3 + tier, 6 + 2 * tier)):
        cx, cy = rng.randrange(w), rng.randrange(h)
        for dx, dy in PATTERNS[pattern]:
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                on[y][x] = not on[y][x]
    rows = ["#" * (w + 2)] + ["#" + "".join("x" if v else "o" for v in row) + "#" for row in on] + ["#" * (w + 2)]
    return {"map": rows, "pattern": pattern}
