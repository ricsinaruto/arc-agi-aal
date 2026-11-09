"""q015 guards: reach the green exit without being caught by guards that move after every action.

Hidden rules: arrows move the blue avatar one cell (a move into a dark-gray wall leaves it in place);
ACTION5 waits a turn. After every action each red guard takes one step along its straight patrol:
some guards patrol horizontally (starting rightwards), others vertically (starting downwards); a guard
that would walk into a wall or off the board turns back and steps the other way on the same turn if
it can (otherwise it stays). Guards and the hunter pass through each other and over the exit. From
level 4 an orange hunter also takes one step towards the avatar after every action, along the axis
with the larger distance first (horizontal on ties) and along the other axis if that step is blocked
by a wall; it stays put if both are blocked. The avatar is caught, and the game is over (RESET
restarts the level), if after a turn a guard or the hunter stands on its cell, or it and a guard or
the hunter swapped cells. Reaching the green exit clears the level at once, before anyone else moves.
"""

NAME = "guards"
TAGS = ["keyboard", "interact", "enemies", "timing"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4, 5]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "@": 5, "X": 14, "h": 5, "v": 5, "c": 5}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.exit = where["X"][0]
        self.s_guards = tuple(sorted([(p, (1, 0)) for p in where.get("h", [])] + [(p, (0, 1)) for p in where.get("v", [])]))
        self.s_hunter = where["c"][0] if "c" in where else None

    def state_key(self):
        return (self.s_pos, self.s_guards, self.s_hunter)

    def render_grid(self):
        out = self.grid.copy()
        out[self.s_pos[1], self.s_pos[0]] = 9
        for (x, y), _ in self.s_guards:
            out[y, x] = 8
        if self.s_hunter is not None:
            out[self.s_hunter[1], self.s_hunter[0]] = 12
        return out

    def open(self, p):
        return self.inside(*p) and self.grid[p[1], p[0]] != 3

    def apply(self, action, cell):
        old = self.s_pos
        if action != 5:
            dx, dy = MOVES[action]
            p = (old[0] + dx, old[1] + dy)
            if self.open(p):
                self.s_pos = p
        if self.s_pos == self.exit:
            return "win"
        pos = self.s_pos
        caught = False
        guards = []
        for (x, y), (dx, dy) in self.s_guards:
            if not self.open((x + dx, y + dy)):
                dx, dy = -dx, -dy
            new = (x + dx, y + dy) if self.open((x + dx, y + dy)) else (x, y)
            caught |= new == pos or (new == old and (x, y) == pos)
            guards.append((new, (dx, dy)))
        self.s_guards = tuple(sorted(guards))
        if self.s_hunter is not None:
            hx, hy = self.s_hunter
            ddx, ddy = pos[0] - hx, pos[1] - hy
            sx = (ddx > 0) - (ddx < 0)
            sy = (ddy > 0) - (ddy < 0)
            steps = [(sx, 0), (0, sy)] if abs(ddx) >= abs(ddy) else [(0, sy), (sx, 0)]
            new = (hx, hy)
            for step in steps:
                if step != (0, 0) and self.open((hx + step[0], hy + step[1])):
                    new = (hx + step[0], hy + step[1])
                    break
            caught |= new == pos or (new == old and (hx, hy) == pos)
            self.s_hunter = new
        return "lose" if caught else None


TIER_ACTIONS = [(6, 14), (8, 18), (9, 22), (8, 22), (10, 26)]
# (interior width, interior height, wall density, horizontal guards, vertical guards, hunter)
TIER_SHAPES = [((7, 5), 0.15, 1, 1, False), ((8, 6), 0.15, 2, 1, False), ((9, 6), 0.2, 2, 2, False),
               ((8, 6), 0.2, 1, 0, True), ((9, 7), 0.2, 1, 2, True)]


def generate_level(rng, tier):
    """Random sparse walls; the avatar starts in the left column and the exit is in the right column;
    guards patrol the middle columns and the hunter starts on the far side. The admission search keeps
    only solvable, non-trivial levels."""
    (w, h), density, horizontal, vertical, hunter = TIER_SHAPES[tier]
    cells = {(x, y): "#" if rng.random() < density and 1 <= x < w - 1 else "." for y in range(h) for x in range(w)}
    start = (0, rng.randrange(h))
    goal = (w - 1, rng.randrange(h))
    cells[start], cells[goal] = "@", "X"
    middle = [p for p, ch in cells.items() if ch == "." and 2 <= p[0] < w - 1]
    rng.shuffle(middle)
    for ch in "h" * horizontal + "v" * vertical:
        cells[middle.pop()] = ch
    if hunter:
        far = [p for p in middle if p[0] >= w - 3]
        cells[rng.choice(far or middle)] = "c"
    rows = ["#" + "".join(cells[(x, y)] for x in range(w)) + "#" for y in range(h)]
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)]}
