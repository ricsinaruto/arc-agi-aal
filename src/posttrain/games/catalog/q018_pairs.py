"""q018 pairs: clear the board by removing pairs of same-coloured tiles that can be linked.

Hidden rules: the board holds 2x2-cell coloured tiles on a grid with an empty margin around it.
Clicking a tile selects it (a white corner cell marks it); clicking it again deselects it. With a tile
selected, clicking another tile of the same colour removes both if they can be linked by a line of at
most three straight segments (at most two turns) running through empty tile positions only, where
the empty margin around the board counts as empty. Otherwise the second tile just becomes the
selected one. Clicks on empty positions do nothing. From level 3 gravity applies: after a removal the
tiles above an emptied position in each column fall down until the column is packed at the bottom.
The level clears when the board is empty. Removing pairs in the wrong order can leave tiles that can
no longer be linked (RESET restarts the level).
"""

NAME = "pairs"
TAGS = ["click", "matching", "gravity", "rule-change"]

COLOURS = (8, 9, 11, 14, 6, 12, 15, 10)
STEPS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [6]
    BACKGROUND = 5
    LEGEND = {".": 5}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, _ = self.parse(spec)
        self.tw, self.th = spec["size"]
        self.gravity = bool(spec.get("gravity"))
        self.s_tiles = tuple(c if c >= 0 else None for c in spec["tiles"])
        self.s_sel = None

    def state_key(self):
        return (self.s_tiles, self.s_sel)

    def slot(self, cell):
        sx, sy = cell[0] // 2 - 1, cell[1] // 2 - 1
        return sy * self.tw + sx if 0 <= sx < self.tw and 0 <= sy < self.th else None

    def solver_clicks(self):
        return [(2 * (i % self.tw + 1), 2 * (i // self.tw + 1)) for i in range(self.tw * self.th)]

    def render_grid(self):
        out = self.grid.copy()
        for i, colour in enumerate(self.s_tiles):
            if colour is not None:
                x, y = 2 * (i % self.tw + 1), 2 * (i // self.tw + 1)
                out[y:y + 2, x:x + 2] = colour
                if i == self.s_sel:
                    out[y, x] = 0
        return out

    def linked(self, tiles, a, b):
        tw, th = self.tw, self.th
        start, goal = (a % tw + 1, a // tw + 1), (b % tw + 1, b // tw + 1)

        def empty(p):
            x, y = p
            if not (0 <= x < tw + 2 and 0 <= y < th + 2):
                return False
            return not (1 <= x <= tw and 1 <= y <= th) or tiles[(y - 1) * tw + x - 1] is None

        best = {}
        queue = [(start, d, 0) for d in range(4)]
        while queue:
            (x, y), d, turns = queue.pop(0)
            q = (x + STEPS[d][0], y + STEPS[d][1])
            if q == goal:
                return True
            if not empty(q):
                continue
            for nd in range(4):
                nt = turns + (nd != d)
                if nt <= 2 and best.get((q, nd), 3) > nt:
                    best[(q, nd)] = nt
                    queue.append((q, nd, nt))
        return False

    def apply(self, action, cell):
        i = self.slot(cell)
        if i is None or self.s_tiles[i] is None:
            return None
        sel = self.s_sel
        if sel is None or sel == i:
            self.s_sel = None if sel == i else i
            return None
        tiles = self.s_tiles
        if tiles[sel] != tiles[i] or not self.linked(tiles, sel, i):
            self.s_sel = i
            return None
        tiles = list(tiles)
        tiles[sel] = tiles[i] = None
        if self.gravity:
            for x in range(self.tw):
                column = [tiles[y * self.tw + x] for y in range(self.th) if tiles[y * self.tw + x] is not None]
                column = [None] * (self.th - len(column)) + column
                for y in range(self.th):
                    tiles[y * self.tw + x] = column[y]
        self.s_tiles, self.s_sel = tuple(tiles), None
        return "win" if all(t is None for t in tiles) else None


MAX_STATES = 150_000
TIER_ACTIONS = [(6, 8), (8, 14), (8, 14), (12, 18), (14, 22)]
# (tiles wide, tiles high, pairs, colours, gravity)
TIER_SHAPES = [((3, 2), 3, 3, False), ((4, 3), 5, 4, False), ((4, 3), 6, 4, True), ((4, 4), 7, 5, False),
               ((5, 4), 8, 6, True)]


def generate_level(rng, tier):
    """Random pairs on random positions (colours may repeat across pairs); the admission search keeps
    only solvable levels, and a random policy almost never finds a full clearing order."""
    (tw, th), pairs, colours, gravity = TIER_SHAPES[tier]
    palette = rng.sample(COLOURS, colours)
    tiles = [palette[i % colours] for i in range(pairs)] * 2 + [-1] * (tw * th - 2 * pairs)
    rng.shuffle(tiles)
    if gravity:  # start packed at the bottom, as gravity would leave it
        for x in range(tw):
            column = [tiles[y * tw + x] for y in range(th) if tiles[y * tw + x] >= 0]
            column = [-1] * (th - len(column)) + column
            for y in range(th):
                tiles[y * tw + x] = column[y]
    w, h = 2 * (tw + 2), 2 * (th + 2)
    return {"map": ["." * w for _ in range(h)], "size": [tw, th], "tiles": tiles, "gravity": gravity}
