"""q014 pipes: rotate pipe tiles until water flows from the red source to every green sink.

Hidden rules: the board is a grid of 3x3-cell tiles. A pipe tile shows a centre cell and one arm per
opening (up, right, down, left). Clicking any cell of a rotatable tile (light gray when dry, blue when
wet) turns it a quarter turn clockwise; clicks on blank tiles or the border do nothing. The red-centred source and green-centred sinks never turn; from level 4 gray tiles are
locked and ignore clicks too. Water spreads from the source through every pair of neighbouring tiles
whose facing openings meet; wet pipes are drawn blue (locked ones sky blue), dry pipes light gray
(locked ones gray). A sink's arm turns blue when water reaches it. The level clears when every sink
is wet. Levels 1-2 have one sink, level 3 two sinks, level 4 one sink with locked tiles, level 5 two
sinks with locked tiles.
"""

NAME = "pipes"
TAGS = ["click", "rotation", "connectivity"]

DIRS = ((1, 0, -1), (2, 1, 0), (4, 0, 1), (8, -1, 0))  # (bit, dx, dy): up, right, down, left
OPPOSITE = {1: 4, 2: 8, 4: 1, 8: 2}
ARMS = {1: (1, 0), 2: (2, 1), 4: (1, 2), 8: (0, 1)}


def rotate(mask):
    return ((mask << 1) | (mask >> 3)) & 15


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [6]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, _ = self.parse(spec)
        self.tw, self.th = spec["size"]
        self.kinds = spec["kinds"]  # per tile, row-major: r rotatable, l locked, s source, k sink, e empty
        self.s_open = tuple(spec["open"])

    def state_key(self):
        return self.s_open

    def tile_of(self, cell):
        x, y = cell[0] - 1, cell[1] - 1
        if 0 <= x < 3 * self.tw and 0 <= y < 3 * self.th:
            return (y // 3) * self.tw + x // 3
        return None

    def solver_clicks(self):
        return [(1 + 3 * (i % self.tw) + 1, 1 + 3 * (i // self.tw) + 1) for i, kind in enumerate(self.kinds)
                if kind == "r" and self.s_open[i] not in (0, 15)]

    def wet(self):
        todo = [i for i, kind in enumerate(self.kinds) if kind == "s"]
        wet = set(todo)
        while todo:
            i = todo.pop()
            tx, ty = i % self.tw, i // self.tw
            for bit, dx, dy in DIRS:
                nx, ny = tx + dx, ty + dy
                j = ny * self.tw + nx
                if self.s_open[i] & bit and 0 <= nx < self.tw and 0 <= ny < self.th and j not in wet \
                        and self.s_open[j] & OPPOSITE[bit]:
                    wet.add(j)
                    todo.append(j)
        return wet

    def render_grid(self):
        out = self.grid.copy()
        wet = self.wet()
        for i, kind in enumerate(self.kinds):
            mask = self.s_open[i]
            if not mask:
                continue
            ox, oy = 1 + 3 * (i % self.tw), 1 + 3 * (i // self.tw)
            locked = kind == "l"
            colour = (10 if locked else 9) if i in wet else (2 if locked else 1)
            out[oy + 1, ox + 1] = {"s": 8, "k": 14}.get(kind, colour)
            for bit, (ax, ay) in ARMS.items():
                if mask & bit:
                    out[oy + ay, ox + ax] = colour
        return out

    def apply(self, action, cell):
        i = self.tile_of(cell)
        if i is None or self.kinds[i] != "r":
            return None
        opened = list(self.s_open)
        opened[i] = rotate(opened[i])
        self.s_open = tuple(opened)
        wet = self.wet()
        return "win" if all(i in wet for i, kind in enumerate(self.kinds) if kind == "k") else None


MAX_STATES = 100_000
TIER_ACTIONS = [(4, 10), (5, 12), (6, 14), (6, 16), (7, 18)]
# (tiles wide, tiles high, sinks, locked tiles, decoy pipes, tiles on the source-sink path)
TIER_SHAPES = [((3, 2), 1, 0, 0, 5), ((3, 3), 1, 0, 1, 6), ((3, 3), 2, 0, 1, 5), ((4, 3), 1, 2, 2, 7),
               ((4, 3), 2, 2, 1, 6)]


def walk(rng, start, min_len, blocked, tw, th, max_len):
    """A random self-avoiding tile path of ``min_len`` to ``max_len`` tiles from ``start`` (DFS)."""
    stack = [[start]]
    while stack:
        path = stack.pop()
        if len(path) >= min_len:
            return path
        if len(path) >= max_len:
            continue
        x, y = path[-1]
        steps = [(x + dx, y + dy) for _, dx, dy in DIRS]
        rng.shuffle(steps)
        for p in steps:
            if 0 <= p[0] < tw and 0 <= p[1] < th and p not in path and p not in blocked:
                stack.append(path + [p])
    return None


def generate_level(rng, tier):
    """A random source-to-sink path (plus a branch to the second sink) sets the pipe shapes; other
    tiles are empty or decoy pipes; locked tiles keep their solved turn, the rest are turned randomly.
    Solvable by construction."""
    (tw, th), sinks, locked, decoys, min_len = TIER_SHAPES[tier]
    tiles = [(x, y) for y in range(th) for x in range(tw)]
    while True:
        source = rng.choice(tiles)
        path = walk(rng, source, min_len, set(), tw, th, tw * th)
        if path is None:
            continue
        edges = list(zip(path, path[1:]))
        ends = [path[-1]]
        if sinks == 2:
            branch_from = rng.choice(path[1:-1])
            branch = walk(rng, branch_from, 3, set(path) - {branch_from}, tw, th, 4)
            if branch is None:
                continue
            edges += list(zip(branch, branch[1:]))
            ends.append(branch[-1])
        break
    mask = {p: 0 for p in tiles}
    for a, b in edges:
        for bit, dx, dy in DIRS:
            if (a[0] + dx, a[1] + dy) == b:
                mask[a] |= bit
                mask[b] |= OPPOSITE[bit]
    kinds = {p: ("s" if p == source else "k" if p in ends else "r" if mask[p] else "e") for p in tiles}
    empty = [p for p in tiles if kinds[p] == "e"]
    rng.shuffle(empty)
    for p in empty[:decoys]:
        kinds[p], mask[p] = "r", rng.choice((3, 5, 7))
    network = [p for p in tiles if kinds[p] == "r"]
    for p in rng.sample(network, min(locked, len(network))):
        kinds[p] = "l"
    for p in tiles:
        if kinds[p] == "r":
            turned = [m for m in (rotate(mask[p]), rotate(rotate(mask[p])), rotate(rotate(rotate(mask[p]))))
                      if m != mask[p]]
            mask[p] = rng.choice(turned)
    w, h = 3 * tw + 2, 3 * th + 2
    rows = ["#" * w] + ["#" + "." * (w - 2) + "#" for _ in range(h - 2)] + ["#" * w]
    return {"map": rows, "size": [tw, th], "kinds": [kinds[p] for p in tiles], "open": [mask[p] for p in tiles]}
