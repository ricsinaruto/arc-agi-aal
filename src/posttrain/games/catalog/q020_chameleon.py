"""q020 chameleon: take on colours from pads to pass through barriers of the same colour.

Hidden rules: arrows move the white avatar one cell; dark-gray walls block. The avatar carries one
colour at a time, shown by the colour of the board's border (dark gray at the start: no colour).
Stepping onto a coloured pad (a single red, yellow, magenta or purple cell) gives the avatar that
colour. A barrier cell of a colour (drawn exactly like a pad of that colour) can be entered only while
the avatar carries that colour; otherwise the move does nothing. Pads and barriers look alike: only
trying tells which is which. Levels 1-3: pads stay and can be used again. From level 4 a pad is used
up when stepped on (it disappears), so the order of visits matters (RESET restarts the level). The
level clears when the avatar stands on the green exit.
"""

NAME = "chameleon"
TAGS = ["keyboard", "navigation", "colour-state", "keys-doors", "rule-change"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
COLOURS = {"r": 8, "y": 11, "m": 6, "p": 15}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "@": 5, "X": 14, **{k.upper(): v for k, v in COLOURS.items()}, **{k: 5 for k in COLOURS}}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.exit = where["X"][0]
        self.once = bool(spec.get("once"))
        self.barriers = {p: COLOURS[ch] for ch in COLOURS for p in where.get(ch.upper(), [])}
        self.s_pads = frozenset((p, COLOURS[ch]) for ch in COLOURS for p in where.get(ch, []))
        self.s_colour = 3

    def state_key(self):
        return (self.s_pos, self.s_colour, self.s_pads)

    def render_grid(self):
        out = self.grid.copy()
        out[0, :], out[-1, :], out[:, 0], out[:, -1] = self.s_colour, self.s_colour, self.s_colour, self.s_colour
        for (x, y), colour in self.s_pads:
            out[y, x] = colour
        out[self.s_pos[1], self.s_pos[0]] = 0
        return out

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        p = (self.s_pos[0] + dx, self.s_pos[1] + dy)
        if not self.inside(*p) or self.grid[p[1], p[0]] == 3:
            return None
        if p in self.barriers and self.barriers[p] != self.s_colour:
            return None
        self.s_pos = p
        for pad in self.s_pads:
            if pad[0] == p:
                self.s_colour = pad[1]
                if self.once:
                    self.s_pads = self.s_pads - {pad}
                break
        return "win" if p == self.exit else None


TIER_ACTIONS = [(6, 16), (8, 20), (10, 24), (10, 26), (12, 30)]
# (interior width, interior height, walls, gaps per wall, colours, pads per colour, pads used up)
TIER_SHAPES = [((7, 5), 1, 2, "ry", 1, False), ((9, 5), 2, 2, "ry", 1, False), ((10, 6), 2, 2, "rym", 1, False),
               ((10, 5), 2, 2, "ry", 1, True), ((11, 6), 3, 2, "rymp", 1, True)]


def generate_level(rng, tier):
    """Full-height walls whose gaps are barrier cells split the board into chambers; pads of each
    colour are scattered over the chambers. The admission search keeps only solvable, non-trivial
    levels."""
    (w, h), walls, gaps, colours, per_colour, once = TIER_SHAPES[tier]
    cols = sorted(rng.sample(range(2, w - 2), walls))
    while any(b - a < 2 for a, b in zip(cols, cols[1:])):
        cols = sorted(rng.sample(range(2, w - 2), walls))
    cells = {(x, y): "#" if x in cols else "." for x in range(w) for y in range(h)}
    for x in cols:
        for y in rng.sample(range(h), gaps):
            cells[(x, y)] = rng.choice(colours).upper()
    free = [p for p, ch in cells.items() if ch == "."]
    rng.shuffle(free)
    start = min(free[:6], key=lambda p: p[0])
    free.remove(start)
    goal = max(free[:6], key=lambda p: p[0])
    free.remove(goal)
    cells[start], cells[goal] = "@", "X"
    for colour in colours:
        for _ in range(per_colour):
            cells[free.pop()] = colour
    rows = ["#" + "".join(cells[(x, y)] for x in range(w)) + "#" for y in range(h)]
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)], "once": once}
