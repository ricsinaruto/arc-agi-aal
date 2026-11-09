"""q006 mirror: make the editable half the mirror image of the fixed half.

Hidden rules: clicking an editable cell (right of / below the dark-gray divider) cycles its colour
light gray -> red -> blue -> light gray; the fixed half never changes. Levels 1-3 mirror across a
vertical divider (left-right), level 4 across a horizontal divider (top-bottom). The level clears
when the editable half equals the mirror image of the fixed half.
"""

NAME = "mirror"
TAGS = ["click", "symmetry", "rule-change"]

CYCLE = {1: 8, 8: 9, 9: 1}


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [6]
    BACKGROUND = 5
    LEGEND = {"#": 3, "|": 3, "w": 1, "r": 8, "b": 9, "e": 1}
    LEVELS = [
        {"map": ["#######",
                 "#rw|ee#",
                 "#bb|ee#",
                 "#######"], "axis": "v"},
        {"map": ["#########",
                 "#rwb|eee#",
                 "#wrw|eee#",
                 "#########"], "axis": "v"},
        {"map": ["#########",
                 "#brw|eee#",
                 "#wwr|eee#",
                 "#rbb|eee#",
                 "#########"], "axis": "v"},
        {"map": ["#####",
                 "#rbw#",
                 "#wrr#",
                 "#---#",
                 "#eee#",
                 "#eee#",
                 "#####"], "axis": "h"},
    ]

    def load(self, spec):
        spec = dict(spec)
        spec["map"] = [row.replace("-", "|") for row in spec["map"]]
        self.grid, where = self.parse(spec)
        self.editable = set(where.get("e", []))
        divider = where["|"]
        self.s_axis = spec["axis"]
        self.pairs = {}
        for x, y in self.editable:
            if self.s_axis == "v":
                d = divider[0][0]
                self.pairs[(x, y)] = (2 * d - x, y)
            else:
                d = divider[0][1]
                self.pairs[(x, y)] = (x, 2 * d - y)

    def apply(self, action, cell):
        if cell not in self.editable:
            return None
        x, y = cell
        self.grid[y, x] = CYCLE[int(self.grid[y, x])]
        done = all(self.grid[y, x] == self.grid[my, mx] for (x, y), (mx, my) in self.pairs.items())
        return "win" if done else None
