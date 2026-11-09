"""q005 paint: walk over every gray floor cell exactly once.

Hidden rules: the blue avatar paints each cell it leaves sky blue; painted cells cannot be entered
again, so a wrong route strands the avatar (RESET restarts the level). The level clears when no gray
floor cell is left.
"""

NAME = "paint"
TAGS = ["keyboard", "path-cover"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 2, "@": 2}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.grid[self.s_pos[1], self.s_pos[0]] = 10

    def render_grid(self):
        out = self.grid.copy()
        out[self.s_pos[1], self.s_pos[0]] = 9
        return out

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        x, y = self.s_pos[0] + dx, self.s_pos[1] + dy
        if not self.inside(x, y) or self.grid[y, x] != 2:
            return None
        self.grid[y, x] = 10
        self.s_pos = (x, y)
        return "win" if not (self.grid == 2).any() else None


TIER_ACTIONS = [(8, 12), (11, 15), (14, 18), (16, 21), (19, 24)]
TIER_SHAPES = [((5, 4), 12), ((6, 4), 15), ((6, 5), 18), ((7, 5), 21), ((7, 6), 24)]


def generate_level(rng, tier):
    """The floor is a random self-avoiding walk, so a covering route from the start exists."""
    (w, h), length = TIER_SHAPES[tier]
    for _ in range(200):
        start = (rng.randrange(w), rng.randrange(h))
        walk = [start]
        seen = {start}
        while len(walk) < length:
            x, y = walk[-1]
            options = [(x + dx, y + dy) for dx, dy in MOVES.values()
                       if 0 <= x + dx < w and 0 <= y + dy < h and (x + dx, y + dy) not in seen]
            if not options:
                break
            nxt = rng.choice(options)
            walk.append(nxt)
            seen.add(nxt)
        if len(walk) == length:
            break
    rows = []
    for y in range(h):
        rows.append("#" + "".join("@" if (x, y) == walk[0] else "." if (x, y) in seen else "#" for x in range(w)) + "#")
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)]}
