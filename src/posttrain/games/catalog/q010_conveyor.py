"""q010 conveyor: reach the green exit across walls whose gaps hold conveyor belts.

Hidden rules: arrows move the blue avatar one cell; dark-gray walls block. A belt cell's colour gives
its direction: sky blue carries up, magenta down, orange left, yellow right. When the avatar steps
onto a belt cell it is carried along, one cell at a time in the belt's direction (and on through any
further belt cells), until it reaches a non-belt cell or the next cell is a wall, where it stops (a
ride that would come back to a belt cell it already crossed stops there instead). A belt
pointing back the way the avatar came throws it straight back, so such a gap cannot be crossed. The
level clears when the avatar stands on the green exit. Levels 1-2: clicks do nothing. From level 3 a
click on a belt cell reverses that whole belt (the straight run of same-coloured belt cells it belongs
to): up and down swap, left and right swap, and the colours change with them.
"""

NAME = "conveyor"
TAGS = ["keyboard", "click", "conveyor", "one-way", "rule-change"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
BELTS = {"^": (0, -1), "v": (0, 1), "<": (-1, 0), ">": (1, 0)}
BELT_COLOURS = {"^": 10, "v": 6, "<": 12, ">": 11}
REVERSE = {"^": "v", "v": "^", "<": ">", ">": "<"}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4, 6]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "@": 5, "X": 14, "^": 5, "v": 5, "<": 5, ">": 5}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.exit = where["X"][0]
        self.reversible = bool(spec.get("reversible"))
        start = {p: ch for ch in BELTS for p in where.get(ch, [])}
        self.tiles = sorted(start)
        self.index = {p: i for i, p in enumerate(self.tiles)}
        self.s_dirs = tuple(start[p] for p in self.tiles)
        belt = {p: p for p in self.tiles}

        def root(p):
            while belt[p] != p:
                p = belt[p]
            return p

        for p, ch in start.items():
            q = (p[0] + BELTS[ch][0], p[1] + BELTS[ch][1])
            if start.get(q) == ch:
                belt[root(q)] = root(p)
        self.belt = {p: [q for q in self.tiles if root(q) == root(p)] for p in self.tiles}

    def state_key(self):
        return (self.s_pos, self.s_dirs)

    def solver_clicks(self):
        if not self.reversible:
            return []
        return sorted({min(cells) for cells in self.belt.values()})

    def render_grid(self):
        out = self.grid.copy()
        for (x, y), ch in zip(self.tiles, self.s_dirs):
            out[y, x] = BELT_COLOURS[ch]
        out[self.s_pos[1], self.s_pos[0]] = 9
        return out

    def open(self, x, y):
        return self.inside(x, y) and self.grid[y, x] != 3

    def apply(self, action, cell):
        if action == 6:
            if not self.reversible or cell not in self.index:
                return None
            dirs = list(self.s_dirs)
            for p in self.belt[cell]:
                dirs[self.index[p]] = REVERSE[dirs[self.index[p]]]
            self.s_dirs = tuple(dirs)
            return None
        dx, dy = MOVES[action]
        x, y = self.s_pos[0] + dx, self.s_pos[1] + dy
        if not self.open(x, y):
            return None
        seen = set()
        while (x, y) in self.index and (x, y) not in seen:
            seen.add((x, y))
            bx, by = BELTS[self.s_dirs[self.index[(x, y)]]]
            if not self.open(x + bx, y + by):
                break
            x, y = x + bx, y + by
        self.s_pos = (x, y)
        return "win" if self.s_pos == self.exit else None


TIER_ACTIONS = [(6, 14), (8, 20), (6, 16), (8, 22), (9, 26)]
# (interior width, interior height, wall columns, gaps per wall, extra belts, reversible)
TIER_SHAPES = [((7, 5), 1, 2, 1, False), ((10, 5), 2, 2, 2, False), ((7, 5), 1, 2, 1, True),
               ((10, 5), 2, 2, 2, True), ((11, 6), 2, 3, 3, True)]


def generate_level(rng, tier):
    """Walls with belt-filled gaps separate the start from the exit. On levels without clicks some gap
    of every wall carries the avatar across; with clicks every gap of one wall throws it back until
    reversed. Extra belts elsewhere are decoys or shortcuts; the admission search checks the level."""
    (w, h), cuts, gaps, extra, reversible = TIER_SHAPES[tier]
    cols = sorted(rng.sample(range(2, w - 2), cuts))
    while any(b - a < 3 for a, b in zip(cols, cols[1:])):
        cols = sorted(rng.sample(range(2, w - 2), cuts))
    cells = {(x, y): "." for x in range(w) for y in range(h)}
    for x in cols:
        for y in range(h):
            cells[(x, y)] = "#"
    blocked_wall = rng.randrange(cuts) if reversible else None
    for i, x in enumerate(cols):
        rows = rng.sample(range(h), gaps)
        forward = set() if i == blocked_wall else {rng.choice(rows)}
        for y in rows:
            ch = ">" if y in forward or (i != blocked_wall and rng.random() < 0.3) else "<"
            for dx in range(-rng.randint(0, 1), rng.randint(1, 2)):
                if 0 <= x + dx < w and cells[(x + dx, y)] == ".":
                    cells[(x + dx, y)] = ch
            cells[(x, y)] = ch
    for _ in range(extra):
        free = [p for p, ch in cells.items() if ch == "." and p[0] not in cols]
        x, y = rng.choice(free)
        ch = rng.choice("^v")
        for dy in range(rng.randint(2, 3)):
            if y + dy < h and cells[(x, y + dy)] == ".":
                cells[(x, y + dy)] = ch
    left = [p for p, ch in cells.items() if ch == "." and p[0] < cols[0]]
    right = [p for p, ch in cells.items() if ch == "." and p[0] > cols[-1]]
    start, goal = rng.choice(left), rng.choice(right)
    cells[start], cells[goal] = "@", "X"
    rows = ["#" + "".join(cells[(x, y)] for x in range(w)) + "#" for y in range(h)]
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)], "reversible": reversible}
