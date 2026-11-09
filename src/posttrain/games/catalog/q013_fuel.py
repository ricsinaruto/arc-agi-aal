"""q013 fuel: reach the green exit before the fuel bar runs out, refuelling on sky-blue pads.

Hidden rules: arrows move the blue avatar one cell; dark-gray walls block (a blocked move costs
nothing). The bar above the maze shows the fuel: one yellow cell per unit left, charcoal for spent
units, and its length is the tank size. Every move costs one unit; from level 4 entering a magenta mud
cell costs three. Reaching the green exit clears the level even on the last unit. Stepping on a
sky-blue pad refills the tank to full and the pad disappears. If a move needs more fuel than is left,
or the tank is empty after a move that did not end on the exit or a pad, the game is over (RESET
restarts the level). Levels 1-2 can be done on one tank; from level 3 the tank is too small for the
trip and pads must be visited on the way.
"""

NAME = "fuel"
TAGS = ["keyboard", "navigation", "counter", "resource"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "=": 5, "@": 5, "X": 14, "f": 5, "m": 6}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.exit = where["X"][0]
        self.mud = frozenset(where.get("m", []))
        self.cap = int(spec["fuel"])
        self.s_fuel = self.cap
        self.s_pads = frozenset(where.get("f", []))

    def state_key(self):
        return (self.s_pos, self.s_fuel, self.s_pads)

    def render_grid(self):
        out = self.grid.copy()
        for x in range(self.cap):
            out[0, x] = 11 if x < self.s_fuel else 4
        for x, y in self.s_pads:
            out[y, x] = 10
        out[self.s_pos[1], self.s_pos[0]] = 9
        return out

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        x, y = self.s_pos[0] + dx, self.s_pos[1] + dy
        if not self.inside(x, y) or self.grid[y, x] == 3 or y == 0:
            return None
        self.s_pos = (x, y)
        self.s_fuel -= 3 if (x, y) in self.mud else 1
        if self.s_fuel < 0:
            return "lose"
        if self.s_pos == self.exit:
            return "win"
        if self.s_pos in self.s_pads:
            self.s_pads = self.s_pads - {self.s_pos}
            self.s_fuel = self.cap
        return "lose" if self.s_fuel == 0 else None


TIER_ACTIONS = [(6, 14), (8, 18), (9, 22), (10, 24), (12, 28)]
# (interior width, interior height, wall density, pads, mud cells, tank size relative to the direct route)
TIER_SHAPES = [((6, 5), 0.25, 1, 0, "direct"), ((8, 6), 0.3, 1, 0, "direct"), ((8, 6), 0.25, 2, 0, "short"),
               ((9, 6), 0.25, 2, 3, "short"), ((10, 7), 0.3, 3, 4, "short")]


def distances(open_cells, start):
    dist = {start: 0}
    queue = [start]
    for x, y in queue:
        for dx, dy in MOVES.values():
            p = (x + dx, y + dy)
            if p in open_cells and p not in dist:
                dist[p] = dist[(x, y)] + 1
                queue.append(p)
    return dist


def generate_level(rng, tier):
    """A random maze with the start and exit far apart. The tank is the direct route plus one or two
    units on levels 1-2, and 55-85% of it from level 3 (pads needed); the admission search keeps only
    solvable, non-trivial levels."""
    (w, h), density, pads, muds, tank = TIER_SHAPES[tier]
    cells = [(x, y) for y in range(h) for x in range(w)]
    while True:
        wall = {c for c in cells if rng.random() < density}
        open_cells = set(cells) - wall
        start = rng.choice(sorted(open_cells))
        dist = distances(open_cells, start)
        if len(dist) > w * h // 2:
            break
    far = sorted(dist, key=lambda p: -dist[p])[:4]
    goal = rng.choice(far)
    route = dist[goal]
    rest = sorted(set(dist) - {start, goal})
    rng.shuffle(rest)
    pad = set(rest[:pads])
    mud = set(rest[pads:pads + muds])
    fuel = route + rng.randint(1, 2) if tank == "direct" else max(3, int(route * rng.uniform(0.55, 0.85)))
    fuel = min(fuel, w + 2)
    rows = ["=" * (w + 2), "#" * (w + 2)]
    for y in range(h):
        row = ""
        for x in range(w):
            p = (x, y)
            row += ("#" if p in wall else "@" if p == start else "X" if p == goal else "f" if p in pad
                    else "m" if p in mud else ".")
        rows.append("#" + row + "#")
    rows.append("#" * (w + 2))
    return {"map": rows, "fuel": fuel}
