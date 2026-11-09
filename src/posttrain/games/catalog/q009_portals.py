"""q009 portals: reach the green exit through a maze split into chambers linked by portal pairs.

Hidden rules: arrows move the blue avatar one cell; dark-gray walls block. Portals come in pairs of
one colour (magenta, purple, orange, pink): stepping onto one end moves the avatar at once to the
other end of the same colour (arriving there does not teleport again; step off and back on to use it
again). Levels 1-2: portals can be used any number of times. From level 3 each pair works once:
after a teleport both ends of that pair turn into plain floor, so a portal taken in the wrong order
can strand the avatar (RESET restarts the level). The level clears when the avatar stands on the
green exit.
"""

NAME = "portals"
TAGS = ["keyboard", "navigation", "teleport", "rule-change"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PAIRS = {"a": 6, "b": 15, "c": 12, "d": 7}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "@": 5, "X": 14, **PAIRS, **{k.upper(): v for k, v in PAIRS.items()}}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.exit = where["X"][0]
        self.once = bool(spec.get("once"))
        self.partner = {}
        for ch in PAIRS:
            if ch in where:
                a, b = where[ch][0], where[ch.upper()][0]
                self.partner[a], self.partner[b] = b, a

    def render_grid(self):
        out = self.grid.copy()
        out[self.s_pos[1], self.s_pos[0]] = 9
        return out

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        x, y = self.s_pos[0] + dx, self.s_pos[1] + dy
        if not self.inside(x, y) or self.grid[y, x] == 3:
            return None
        self.s_pos = (x, y)
        other = self.partner.get((x, y))
        if other is not None and self.grid[y, x] != 5:
            self.s_pos = other
            if self.once:
                self.grid[y, x] = 5
                self.grid[other[1], other[0]] = 5
        return "win" if self.s_pos == self.exit else None


TIER_ACTIONS = [(6, 14), (8, 18), (8, 20), (10, 24), (11, 28)]
# (interior width, interior height, chambers, extra portal pairs, loose walls, portals work once)
TIER_SHAPES = [((8, 4), 2, 1, 2, False), ((10, 5), 3, 1, 4, False), ((9, 5), 3, 1, 3, True),
               ((11, 5), 3, 2, 4, True), ((12, 6), 4, 1, 6, True)]


def generate_level(rng, tier):
    """Full-height walls cut the board into chambers; one portal pair joins each chamber to a random
    other chamber and extra pairs are decoys. The admission search keeps only solvable, non-trivial
    levels (the exit chamber is reachable only through portals)."""
    (w, h), chambers, extra, loose, once = TIER_SHAPES[tier]
    cuts = sorted(rng.sample(range(2, w - 2), chambers - 1))
    while any(b - a < 2 for a, b in zip(cuts, cuts[1:])):
        cuts = sorted(rng.sample(range(2, w - 2), chambers - 1))
    bounds = [-1] + cuts + [w]
    rooms = [[(x, y) for x in range(bounds[i] + 1, bounds[i + 1]) for y in range(h)] for i in range(chambers)]
    wall = {(x, y) for x in cuts for y in range(h)}
    used = set()

    def pick(room):
        cell = rng.choice([c for c in room if c not in used])
        used.add(cell)
        return cell

    start, goal = pick(rooms[0]), pick(rooms[-1])
    pairs = []
    order = list(range(chambers))
    for i in range(chambers - 1):
        pairs.append((pick(rooms[order[i]]), pick(rooms[rng.choice(order[i + 1:])])))
    for _ in range(extra):
        pairs.append((pick(rng.choice(rooms)), pick(rng.choice(rooms))))
    rng.shuffle(pairs)
    for _ in range(loose):
        cell = rng.choice([c for room in rooms for c in room if c not in used])
        wall.add(cell)
    letters = {}
    for ch, (a, b) in zip(PAIRS, pairs):
        letters[a], letters[b] = ch, ch.upper()
    rows = []
    for y in range(h):
        row = ""
        for x in range(w):
            p = (x, y)
            row += "@" if p == start else "X" if p == goal else letters.get(p) or ("#" if p in wall else ".")
        rows.append("#" + row + "#")
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)], "once": once}
