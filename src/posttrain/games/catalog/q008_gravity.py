"""q008 gravity: a side view with gravity; reach the green exit, pushing orange crates to build steps.

Hidden rules: LEFT/RIGHT walk the blue avatar one cell. Walking into an orange crate pushes it one cell
if the cell beyond it is empty (not a wall or crate). Walking into a wall or an unpushable crate climbs
it instead when it is exactly one cell high (the cell above the avatar and the cell above the obstacle
are both empty); anything higher blocks. After every action gravity pulls the avatar and every crate
straight down (towards the gray border row) until each rests on a wall, a crate or the avatar, so
walking off a ledge drops the avatar and a pushed crate falls into the gap below it. The level clears
when the avatar comes to rest on the green exit. Levels 1-2: ACTION5 does nothing. From level 3,
ACTION5 flips gravity: "down" becomes up (the gray border row moves to the top), and the avatar and
crates fall towards the ceiling until the next flip; walking and climbing then work upside down.
"""

NAME = "gravity"
TAGS = ["keyboard", "gravity", "pushing", "interact", "rule-change"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [3, 4, 5]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "@": 5, "o": 5, "X": 5}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.s_crates = frozenset(where.get("o", []))
        self.s_down = 1
        self.exit = where["X"][0]
        self.flip = bool(spec.get("flip"))
        self.settle()

    def state_key(self):
        return (self.s_pos, self.s_crates, self.s_down)

    def render_grid(self):
        out = self.grid.copy()
        out[-1 if self.s_down == 1 else 0, :] = 2
        out[self.exit[1], self.exit[0]] = 14
        for x, y in self.s_crates:
            out[y, x] = 12
        out[self.s_pos[1], self.s_pos[0]] = 9
        return out

    def empty(self, p, crates, pos):
        return self.inside(*p) and self.grid[p[1], p[0]] != 3 and p not in crates and p != pos

    def settle(self):
        crates, pos, dy = set(self.s_crates), self.s_pos, self.s_down
        moved = True
        while moved:
            moved = False
            for p in sorted(list(crates) + [pos], key=lambda q: -q[1] * dy):
                below = (p[0], p[1] + dy)
                if self.empty(below, crates, pos):
                    if p == pos:
                        pos = below
                    else:
                        crates.discard(p)
                        crates.add(below)
                    moved = True
        self.s_pos, self.s_crates = pos, frozenset(crates)

    def apply(self, action, cell):
        if action == 5:
            if not self.flip:
                return None
            self.s_down = -self.s_down
        else:
            dx = MOVES[action][0]
            x, y = self.s_pos
            ahead, up = (x + dx, y), -self.s_down
            crates = self.s_crates
            if self.empty(ahead, crates, None):
                self.s_pos = ahead
            elif ahead in crates and self.empty((x + 2 * dx, y), crates, None):
                self.s_crates = (crates - {ahead}) | {(x + 2 * dx, y)}
                self.s_pos = ahead
            elif self.empty((x, y + up), crates, None) and self.empty((x + dx, y + up), crates, None):
                self.s_pos = (x + dx, y + up)
            else:
                return None
        self.settle()
        return "win" if self.s_pos == self.exit else None


TIER_ACTIONS = [(7, 14), (9, 18), (8, 18), (10, 22), (12, 26)]
# (interior width, interior height, rises, crates, gravity flip)
TIER_SHAPES = [(11, 5, 2, 2, False), (12, 6, 2, 3, False), (9, 7, 1, 1, True), (10, 7, 2, 2, True),
               (12, 7, 2, 3, True)]


def generate_level(rng, tier):
    """A height map walked left to right with ``rises`` two-cell rises that only a pushed crate makes
    climbable (crates start on the floor left of each rise); from level 3 ceiling columns hang down and
    the exit may hang under the ceiling. The admission search keeps only solvable, non-trivial levels."""
    w, h, rises, n_crates, flip = TIER_SHAPES[tier]
    for _ in range(1000):
        floor = [rng.randint(0, 1)]
        rise_at = sorted(rng.sample(range(3, w - 2), rises))
        for x in range(1, w):
            delta = 2 if x in rise_at else rng.choice((-1, 0, 0, 0, 1, -2) if x > 1 else (0, 0, -1))
            floor.append(floor[-1] + delta)
        if min(floor) >= 0 and max(floor) <= h - 3:
            break
    else:
        floor = [0] * w
    top = [min(rng.choice((0, 0, 1, 2)) if flip else 0, h - 2 - floor[x]) for x in range(w)]
    solid = {(x, y) for x in range(w) for y in range(h) if y >= h - floor[x] or y < top[x]}
    rest_down = [(x, h - 1 - floor[x]) for x in range(w)]
    rest_up = [(x, top[x]) for x in range(w)]
    start = rest_down[0]
    goal = rng.choice(rest_down[rise_at[-1] + 1:] + (rest_up[rise_at[-1]:] if flip else []))
    crate_cols = set()
    for r in rise_at:
        options = [x for x in range(1, r - 1) if x not in crate_cols]
        if options:
            crate_cols.add(rng.choice(options))
    extra = [x for x in range(1, w) if x not in crate_cols and x != goal[0]]
    while len(crate_cols) < n_crates and extra:
        crate_cols.add(extra.pop(rng.randrange(len(extra))))
    crates = {rest_down[x] for x in crate_cols} - {goal}
    rows = []
    for y in range(h):
        row = ""
        for x in range(w):
            p = (x, y)
            row += "#" if p in solid else "@" if p == start else "X" if p == goal else "o" if p in crates else "."
        rows.append("#" + row + "#")
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)], "flip": flip}
