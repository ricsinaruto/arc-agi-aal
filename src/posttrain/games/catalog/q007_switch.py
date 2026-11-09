"""q007 switch: reach the green exit through gates that a pink switch flips.

Hidden rules: arrows move the blue avatar; ACTION5 while standing on a pink switch flips every gate:
red (closed, blocking) becomes dark red (open, walkable) and dark red becomes red. The avatar can
not flip a gate shut on itself (the flip is refused while it stands in a gate that would close).
"""

NAME = "switch"
TAGS = ["keyboard", "interact", "gates"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4, 5]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "@": 5, "X": 14, "R": 8, "r": 13, "s": 7}
    LEVELS = [
        {"map": ["########",
                 "#@.s#..#",
                 "#...R.X#",
                 "#...#..#",
                 "########"]},
        {"map": ["#########",
                 "#@..R...#",
                 "#...#.s.#",
                 "##r###R##",
                 "#s.....X#",
                 "#########"]},
        {"map": ["##########",
                 "#@.#s.R..#",
                 "#..r..#..#",
                 "#..#..#r##",
                 "#s.#..R..#",
                 "#..#..#.X#",
                 "##########"]},
        {"map": ["###########",
                 "#@..#..R..#",
                 "#.s.r..#.s#",
                 "#...#..#..#",
                 "##R###r##R#",
                 "#.....s...#",
                 "#.X.#.....#",
                 "###########"]},
    ]

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.exit = where["X"][0]
        self.switches = frozenset(where.get("s", []))

    def render_grid(self):
        out = self.grid.copy()
        out[self.s_pos[1], self.s_pos[0]] = 9
        return out

    def apply(self, action, cell):
        if action == 5:
            if self.s_pos not in self.switches:
                return None
            here = self.grid[self.s_pos[1], self.s_pos[0]]
            if here == 13:
                return None
            flipped = self.grid.copy()
            flipped[self.grid == 8] = 13
            flipped[self.grid == 13] = 8
            self.grid = flipped
            return None
        dx, dy = MOVES[action]
        x, y = self.s_pos[0] + dx, self.s_pos[1] + dy
        if not self.inside(x, y) or self.grid[y, x] in (3, 8):
            return None
        self.s_pos = (x, y)
        return "win" if self.s_pos == self.exit else None
