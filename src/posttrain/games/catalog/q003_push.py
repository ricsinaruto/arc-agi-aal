"""q003 push: push every orange box onto a green cell.

Hidden rules: arrows move the blue avatar; walking into a box pushes it one cell if the cell behind
it is free (not wall, not box); boxes cannot be pulled. The level clears when every green cell holds
a box. A box on a green cell turns dark red.
"""

NAME = "push"
TAGS = ["keyboard", "sokoban"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "@": 5, "o": 5, "*": 5}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.s_boxes = frozenset(where.get("o", []))
        self.targets = frozenset(where.get("*", []))

    def render_grid(self):
        out = self.grid.copy()
        for x, y in self.targets:
            out[y, x] = 14
        for x, y in self.s_boxes:
            out[y, x] = 13 if (x, y) in self.targets else 12
        out[self.s_pos[1], self.s_pos[0]] = 9
        return out

    def state_key(self):
        return (self.s_pos, self.s_boxes)

    def free(self, x, y):
        return self.inside(x, y) and self.grid[y, x] != 3 and (x, y) not in self.s_boxes

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        x, y = self.s_pos[0] + dx, self.s_pos[1] + dy
        if not self.inside(x, y) or self.grid[y, x] == 3:
            return None
        if (x, y) in self.s_boxes:
            if not self.free(x + dx, y + dy):
                return None
            self.s_boxes = (self.s_boxes - {(x, y)}) | {(x + dx, y + dy)}
        self.s_pos = (x, y)
        return "win" if self.targets <= self.s_boxes else None


TIER_ACTIONS = [(5, 10), (8, 16), (12, 24), (16, 32), (20, 44)]
TIER_SHAPES = [((6, 4), 1, 1), ((6, 5), 2, 2), ((7, 5), 2, 3), ((7, 6), 3, 4), ((8, 6), 3, 5)]


def generate_level(rng, tier):
    """Reverse play: boxes start on targets and the avatar pulls them away (solvable by construction)."""
    (w, h), boxes, walls = TIER_SHAPES[tier]
    wall = {(rng.randrange(w), rng.randrange(h)) for _ in range(walls)}
    free = [(x, y) for y in range(h) for x in range(w) if (x, y) not in wall]
    rng.shuffle(free)
    targets = free[:boxes]
    box = set(targets)
    pos = free[boxes]
    for _ in range(40 + 30 * tier):
        dx, dy = rng.choice(list(MOVES.values()))
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in wall or (nx, ny) in box:
            continue
        behind = (pos[0] - dx, pos[1] - dy)
        if behind in box and rng.random() < 0.7:
            box.remove(behind)
            box.add(pos)
        pos = (nx, ny)
    rows = []
    for y in range(h):
        row = ""
        for x in range(w):
            p = (x, y)
            row += "#" if p in wall else "@" if p == pos else "o" if p in box else "*" if p in targets else "."
        rows.append("#" + row + "#")
    return {"map": ["#" * (w + 2)] + rows + ["#" * (w + 2)]}
