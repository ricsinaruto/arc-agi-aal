"""q017 flow: draw a path between each pair of same-coloured dots by clicking cells.

Hidden rules: each colour (red, blue, magenta, yellow) has exactly two endpoint dots. Clicking an
endpoint selects its colour and starts that colour's path afresh from the clicked dot (any earlier
path of that colour is erased); the board's border turns the selected colour. Clicking an empty cell
next to (up, down, left or right of) the end of the selected colour's path extends the path onto it;
path cells are drawn in a darker or paler shade (red: dark red, blue: sky blue, magenta: pink, yellow:
orange). Clicking the other endpoint of the same colour next to the path's end completes that path,
and a completed path takes no more cells. Paths cannot cross each other or pass through a dot, and all
other clicks do nothing. Levels 1-3 clear when every pair is joined. From level 4 the level clears
only when every pair is joined and every board cell is covered by a path or a dot.
"""

NAME = "flow"
TAGS = ["click", "path-drawing", "pairs", "rule-change"]

DOTS = {"r": 8, "b": 9, "m": 6, "y": 11}
SHADES = {"r": 13, "b": 10, "m": 7, "y": 12}
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [6]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, **DOTS}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.colours = [c for c in DOTS if c in where]
        self.ends = {c: tuple(where[c]) for c in self.colours}
        self.dot = {p: c for c in self.colours for p in where[c]}
        self.board = [(x, y) for y in range(self.grid.shape[0]) for x in range(self.grid.shape[1])
                      if self.grid[y, x] != 3]
        self.fill = bool(spec.get("fill"))
        self.s_sel = None
        self.s_paths = tuple(() for _ in self.colours)

    def state_key(self):
        return (self.s_sel, self.s_paths)

    def solver_clicks(self):
        return self.board

    def complete(self, i):
        path = self.s_paths[i]
        return len(path) > 1 and path[-1] in self.ends[self.colours[i]] and path[-1] != path[0]

    def render_grid(self):
        out = self.grid.copy()
        if self.s_sel is not None:
            ring = DOTS[self.colours[self.s_sel]]
            out[0, :], out[-1, :], out[:, 0], out[:, -1] = ring, ring, ring, ring
        for c, path in zip(self.colours, self.s_paths):
            for x, y in path:
                if (x, y) not in self.dot:
                    out[y, x] = SHADES[c]
        return out

    def apply(self, action, cell):
        paths = list(self.s_paths)
        if cell in self.dot:
            colour = self.colours.index(self.dot[cell])
            if self.s_sel == colour and paths[colour] and not self.complete(colour) \
                    and cell != paths[colour][0] and self.adjacent(cell, paths[colour][-1]):
                paths[colour] = paths[colour] + (cell,)
            else:
                self.s_sel = colour
                paths[colour] = (cell,)
        elif self.s_sel is not None and cell in self.board and paths[self.s_sel] and not self.complete(self.s_sel):
            used = {p for path in paths for p in path}
            if cell in used or not self.adjacent(cell, paths[self.s_sel][-1]):
                return None
            paths[self.s_sel] = paths[self.s_sel] + (cell,)
        else:
            return None
        self.s_paths = tuple(paths)
        if not all(self.complete(i) for i in range(len(self.colours))):
            return None
        if self.fill:
            covered = {p for path in paths for p in path}
            if any(p not in covered for p in self.board):
                return None
        return "win"

    @staticmethod
    def adjacent(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


MAX_STATES = 150_000
TIER_ACTIONS = [(6, 12), (7, 16), (9, 20), (12, 18), (14, 22)]
# (interior width, interior height, pairs, every cell must be covered)
TIER_SHAPES = [((4, 3), 2, False), ((4, 4), 2, False), ((5, 3), 3, False), ((4, 3), 2, True), ((4, 4), 3, True)]


def hamiltonian(rng, w, h):
    """A random path through every cell of a w x h board (Warnsdorff's rule with random ties)."""
    while True:
        cur = (rng.randrange(w), rng.randrange(h))
        path, seen = [cur], {cur}

        def free(p):
            return [(p[0] + dx, p[1] + dy) for dx, dy in MOVES.values()
                    if 0 <= p[0] + dx < w and 0 <= p[1] + dy < h and (p[0] + dx, p[1] + dy) not in seen]

        while len(path) < w * h:
            options = free(cur)
            if not options:
                break
            rng.shuffle(options)
            cur = min(options, key=lambda p: len(free(p)))
            path.append(cur)
            seen.add(cur)
        if len(path) == w * h:
            return path


def generate_level(rng, tier):
    """Cut a random path through every cell into ``pairs`` segments of at least three cells whose ends
    are not neighbours; each segment's ends are one colour's dots, so joining every pair (and covering
    the board) is possible."""
    (w, h), pairs, fill = TIER_SHAPES[tier]
    path = hamiltonian(rng, w, h)
    n = len(path)
    while True:
        cuts = sorted(rng.sample(range(3, n - 2), pairs - 1))
        bounds = [0] + cuts + [n]
        if all(b - a >= 3 and abs(path[a][0] - path[b - 1][0]) + abs(path[a][1] - path[b - 1][1]) >= 2
               for a, b in zip(bounds, bounds[1:])):
            break
    cells = {}
    for colour, a, b in zip(DOTS, bounds, bounds[1:]):
        cells[path[a]] = cells[path[b - 1]] = colour
    rows = ["#" + "".join(cells.get((x, y), ".") for x in range(w)) + "#" for y in range(h)]
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)], "fill": fill}
