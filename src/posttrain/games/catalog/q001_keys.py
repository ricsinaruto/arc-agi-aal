"""q001 keys: walk to the green exit; a coloured cell you step on opens every wall cell of the same colour.

Hidden rules the player must discover: arrows move the blue avatar one cell; dark-gray walls block;
stepping on a coloured cell that is not blocking (a key) removes it and every blocking cell of that
colour (doors); keys and doors look identical until you try to walk into them.
"""

NAME = "keys"
TAGS = ["keyboard", "navigation", "keys-doors"]

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [1, 2, 3, 4]
    BACKGROUND = 5
    LEGEND = {".": 5, "#": 3, "@": 9, "X": 14, "a": 8, "A": 8, "b": 11, "B": 11, "c": 6, "C": 6}
    LEVELS = [
        {"map": ["#########",
                 "#@..#..X#",
                 "#...#...#",
                 "#.a.A...#",
                 "#...#...#",
                 "#########"]},
        {"map": ["###########",
                 "#@.#...#..#",
                 "#..B.a.A.X#",
                 "#..#...#..#",
                 "#b.#####..#",
                 "###########"]},
        {"map": ["###########",
                 "#@....#..X#",
                 "#.###.#.###",
                 "#.#c#.A...#",
                 "#.#.#######",
                 "#...C..b.a#",
                 "#..########",
                 "#.........#",
                 "###########"]},
        {"map": ["#############",
                 "#@..#...#...#",
                 "#...A.b.B...#",
                 "#c..#...#...#",
                 "##C######...#",
                 "#.......#####",
                 "#.a.....#..X#",
                 "#.......B...#",
                 "#############"]},
    ]

    def load(self, spec):
        self.grid, where = self.parse(spec)
        self.s_pos = where["@"][0]
        self.s_exit = where["X"][0]
        self.s_keys = {p: ch for ch in "abc" for p in where.get(ch, [])}
        self.s_doors = {p: ch.lower() for ch in "ABC" for p in where.get(ch, [])}

    def apply(self, action, cell):
        dx, dy = MOVES[action]
        x, y = self.s_pos[0] + dx, self.s_pos[1] + dy
        if not self.inside(x, y) or (x, y) in self.s_doors or self.grid[y, x] == 3:
            return None
        self.grid[self.s_pos[1], self.s_pos[0]] = 5
        self.s_pos = (x, y)
        key = self.s_keys.pop((x, y), None)
        if key is not None:
            for door in [p for p, k in self.s_doors.items() if k == key]:
                del self.s_doors[door]
                self.grid[door[1], door[0]] = 5
        self.grid[y, x] = 9
        return "win" if (x, y) == self.s_exit else None
