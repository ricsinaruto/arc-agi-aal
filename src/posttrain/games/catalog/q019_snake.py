"""q019 snake: steer a growing snake to eat every yellow food, then leave through the exit.

Hidden rules: arrows move the snake's blue head one cell and the sky-blue body follows (each segment
takes the place of the one ahead of it). The head cannot move into a dark-gray wall or into its own
body (the tail cell is allowed, since the tail moves away), and such a move does nothing. Moving onto
yellow food eats it: the snake grows by one segment (its tail stays put that turn). The exit is
charcoal and blocks while any food is left; it turns green when the last food is eaten, and moving the
head onto the green exit clears the level. From level 4 the border has gaps: moving out through a gap
brings the head in through the matching gap on the opposite side (the board wraps around there).
"""

NAME = "snake"
TAGS = ["keyboard", "snake", "growth", "wrap-around", "rule-change"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "f": 5, "X": 5}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_body = tuple(tuple(p) for p in spec["snake"])
        self.s_food = frozenset(where.get("f", []))
        self.exit = where["X"][0]

    def state_key(self):
        return (self.s_body, self.s_food)

    def render_grid(self):
        out = self.grid.copy()
        out[self.exit[1], self.exit[0]] = 4 if self.s_food else 14
        for x, y in self.s_food:
            out[y, x] = 11
        for x, y in self.s_body[1:]:
            out[y, x] = 10
        out[self.s_body[0][1], self.s_body[0][0]] = 9
        return out

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        rows, cols = self.grid.shape
        x, y = (self.s_body[0][0] + dx) % cols, (self.s_body[0][1] + dy) % rows
        if self.grid[y, x] == 3 or (x, y) in self.s_body[:-1]:
            return None
        if (x, y) == self.exit:
            if self.s_food:
                return None
            self.s_body = ((x, y),) + self.s_body[:-1]
            return "win"
        if (x, y) in self.s_food:
            self.s_food = self.s_food - {(x, y)}
            self.s_body = ((x, y),) + self.s_body
        else:
            self.s_body = ((x, y),) + self.s_body[:-1]
        return None


TIER_ACTIONS = [(6, 14), (8, 18), (10, 22), (8, 22), (10, 26)]
# (interior width, interior height, snake length, food, wall density, border gap pairs)
TIER_SHAPES = [((5, 4), 3, 1, 0.1, 0), ((6, 5), 3, 2, 0.12, 0), ((6, 5), 4, 3, 0.15, 0), ((6, 5), 3, 2, 0.15, 2),
               ((7, 5), 4, 3, 0.15, 3)]


def generate_level(rng, tier):
    """A random snake (a self-avoiding chain), food, exit, sparse walls and, from level 4, matching
    gaps in opposite borders; the admission search keeps only solvable, non-trivial levels."""
    (w, h), length, food, density, gaps = TIER_SHAPES[tier]
    cells = [(x, y) for y in range(1, h + 1) for x in range(1, w + 1)]
    while True:
        snake = [rng.choice(cells)]
        while len(snake) < length:
            x, y = snake[-1]
            options = [(x + dx, y + dy) for dx, dy in MOVES.values() if (x + dx, y + dy) in cells
                       and (x + dx, y + dy) not in snake]
            if not options:
                break
            snake.append(rng.choice(options))
        if len(snake) == length:
            break
    rest = [c for c in cells if c not in snake]
    rng.shuffle(rest)
    goal, foods = rest[0], set(rest[1:1 + food])
    wall = {c for c in rest[1 + food:] if rng.random() < density}
    grid = {(x, y): "#" for x in range(w + 2) for y in range(h + 2)}
    for c in cells:
        grid[c] = "#" if c in wall else "X" if c == goal else "f" if c in foods else "."
    for _ in range(gaps):
        if rng.random() < 0.5:
            y = rng.randint(1, h)
            grid[(0, y)] = grid[(w + 1, y)] = "."
        else:
            x = rng.randint(1, w)
            grid[(x, 0)] = grid[(x, h + 1)] = "."
    rows = ["".join(grid[(x, y)] for x in range(w + 2)) for y in range(h + 2)]
    return {"map": rows, "snake": [list(p) for p in snake]}
