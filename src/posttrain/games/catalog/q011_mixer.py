"""q011 mixer: paint the right-hand canvas to match the left-hand picture by mixing colours.

Hidden rules: the picture on the left of the dark-gray divider is fixed; the canvas on the right starts
white. Below them is a row of swatches: red, blue, yellow and, from level 4, a white eraser. Clicking a
swatch selects it (a light-gray marker appears under the selected swatch). Clicking a canvas cell adds
the selected swatch's colour to it: white + a primary gives that primary; a primary + itself stays;
red + blue gives purple, red + yellow gives orange, blue + yellow gives green; adding any colour to a
purple, orange or green cell, or to charcoal, gives charcoal (mud). The eraser turns a canvas cell back
to white. Clicking a canvas cell with no swatch selected, or clicking the picture, does nothing. The
level clears when every canvas cell equals the picture cell in the same position. Levels 1-2 use only
primaries and white; level 3 needs mixed colours; from level 4 the canvas starts partly painted
(including mud) and the eraser appears.
"""

NAME = "mixer"
TAGS = ["click", "colour-mixing", "pattern-copy"]

COLOURS = {"w": 0, "r": 8, "b": 9, "y": 11, "p": 15, "o": 12, "g": 14, "m": 4}
SWATCHES = {"1": 8, "2": 9, "3": 11, "0": 0}
MIX = {frozenset((8, 9)): 15, frozenset((8, 11)): 12, frozenset((9, 11)): 14}


def mix(colour, paint):
    if paint == 0:
        return 0
    if colour == 0 or colour == paint:
        return paint
    if colour in (8, 9, 11):
        return MIX[frozenset((colour, paint))]
    return 4


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [6]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "_": 5, **COLOURS, **{k.upper(): v for k, v in COLOURS.items()}, **SWATCHES}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.canvas = sorted(((x, y) for ch in COLOURS for x, y in where.get(ch, [])), key=lambda p: (p[1], p[0]))
        target = sorted(((x, y) for ch in COLOURS for x, y in where.get(ch.upper(), [])),
                        key=lambda p: (p[1], p[0]))
        self.target = tuple(int(self.grid[y, x]) for x, y in target)
        self.swatches = sorted(p for ch in SWATCHES for p in where.get(ch, []))
        self.s_canvas = tuple(int(self.grid[y, x]) for x, y in self.canvas)
        self.s_sel = None

    def state_key(self):
        return (self.s_sel, self.s_canvas)

    def solver_clicks(self):
        return self.swatches + self.canvas

    def render_grid(self):
        out = self.grid.copy()
        for (x, y), colour in zip(self.canvas, self.s_canvas):
            out[y, x] = colour
        if self.s_sel is not None:
            out[self.s_sel[1] + 1, self.s_sel[0]] = 1
        return out

    def apply(self, action, cell):
        if cell in self.swatches:
            self.s_sel = cell
            return None
        if cell not in self.canvas or self.s_sel is None:
            return None
        paint = int(self.grid[self.s_sel[1], self.s_sel[0]])
        canvas = list(self.s_canvas)
        i = self.canvas.index(cell)
        canvas[i] = mix(canvas[i], paint)
        self.s_canvas = tuple(canvas)
        return "win" if self.s_canvas == self.target else None


MAX_STATES = 200_000
TIER_ACTIONS = [(4, 6), (5, 9), (6, 12), (7, 14), (8, 16)]
# (canvas width, canvas height, target colours, prefilled cells, eraser)
TIER_SHAPES = [(2, 1, "rby", 0, False), (2, 2, "wrby", 0, False), (2, 2, "rbypog", 0, False),
               (2, 2, "wrbypog", 2, True), (2, 2, "rbypog", 3, True)]


def generate_level(rng, tier):
    """A random picture over the tier's colours and a canvas that is white or partly pre-painted; every
    picture is reachable (a primary is one click, a mixed colour two, mud is cleared by the eraser), and
    the admission search gives the optimal count."""
    cw, ch, colours, prefilled, eraser = TIER_SHAPES[tier]
    target = [[rng.choice(colours) for _ in range(cw)] for _ in range(ch)]
    start = [["w"] * cw for _ in range(ch)]
    for i in rng.sample(range(cw * ch), prefilled):
        start[i // cw][i % cw] = rng.choice("rbypogm")
    swatches = "1230" if eraser else "123"
    width = 2 * cw + 1
    rows = ["#" * (width + 2)]
    for y in range(ch):
        rows.append("#" + "".join(c.upper() for c in target[y]) + "#" + "".join(start[y]) + "#")
    rows.append("#" * (width + 2))
    rows.append("#" + swatches.ljust(width, ".") + "#")
    rows.append("#" + ("_" * len(swatches)).ljust(width, ".") + "#")
    rows.append("#" * (width + 2))
    return {"map": rows}
