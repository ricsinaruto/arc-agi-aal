"""q012 loopover: rotate rows and columns of the right-hand board until it matches the left-hand picture.

Hidden rules: the picture left of the dark-gray divider is fixed. The board on the right holds the same
coloured cells in a scrambled order, with light-gray handles beside its rows and columns. Levels 1-3:
the handles sit left of each row and above each column; clicking a row handle shifts that row one cell
to the right, cyclically (the rightmost cell wraps round to the left end), and clicking a column
handle shifts that column one cell down, cyclically (the bottom cell wraps to the top). From level 4
the handles sit right of each row and below each column and shift the other way: a row handle shifts
its row one cell left, a column handle shifts its column one cell up (both cyclic). Clicks elsewhere
do nothing. The level clears when the board equals the picture.
"""

NAME = "loopover"
TAGS = ["click", "permutation", "pattern-copy", "rule-change"]

COLOURS = {"r": 8, "b": 9, "y": 11, "g": 14}
SHIFTS = {"h": 1, "H": -1, "v": 1, "V": -1}  # handle -> shift along its row (h, H) or column (v, V)


class Game(GridGame):  # noqa: F821 - GridGame is inlined by the store
    ACTIONS = [6]
    BACKGROUND = 5
    LEGEND = {"#": 3, ".": 5, "h": 1, "v": 1, "H": 1, "V": 1, **COLOURS, **{k.upper(): v for k, v in COLOURS.items()}}
    LEVELS = []  # generated: see generate_level

    def load(self, spec):
        self.grid, where = self.parse(spec)
        board = sorted(((x, y) for ch in COLOURS for x, y in where.get(ch, [])), key=lambda p: (p[1], p[0]))
        target = sorted(((x, y) for ch in COLOURS for x, y in where.get(ch.upper(), [])),
                        key=lambda p: (p[1], p[0]))
        self.x0, self.y0 = board[0]
        self.w = len({x for x, _ in board})
        self.h = len({y for _, y in board})
        self.target = tuple(int(self.grid[y, x]) for x, y in target)
        self.s_board = tuple(int(self.grid[y, x]) for x, y in board)
        self.rows = {p: (p[1] - self.y0, SHIFTS[ch]) for ch in "hH" for p in where.get(ch, [])}
        self.cols = {p: (p[0] - self.x0, SHIFTS[ch]) for ch in "vV" for p in where.get(ch, [])}

    def state_key(self):
        return self.s_board

    def solver_clicks(self):
        return sorted(self.rows) + sorted(self.cols)

    def render_grid(self):
        out = self.grid.copy()
        for i, colour in enumerate(self.s_board):
            out[self.y0 + i // self.w, self.x0 + i % self.w] = colour
        return out

    def apply(self, action, cell):
        board = list(self.s_board)
        w, h = self.w, self.h
        if cell in self.rows:
            r, k = self.rows[cell]
            board = shift_row(board, w, r, k)
        elif cell in self.cols:
            c, k = self.cols[cell]
            board = shift_col(board, w, h, c, k)
        else:
            return None
        self.s_board = tuple(board)
        return "win" if self.s_board == self.target else None


def shift_row(board, w, r, k):
    board = list(board)
    row = board[r * w:(r + 1) * w]
    board[r * w:(r + 1) * w] = row[-k:] + row[:-k]
    return board


def shift_col(board, w, h, c, k):
    board = list(board)
    col = [board[r * w + c] for r in range(h)]
    col = col[-k:] + col[:-k]
    for r in range(h):
        board[r * w + c] = col[r]
    return board


TIER_ACTIONS = [(4, 10), (5, 12), (6, 16), (5, 12), (7, 18)]
# (board width, board height, colours, handles left/top ("lt") or right/bottom ("rb"), scramble clicks)
TIER_SHAPES = [(3, 3, "rb", "lt", 4), (3, 3, "rby", "lt", 6), (4, 3, "rbyg", "lt", 8), (3, 3, "rby", "rb", 6),
               (4, 3, "rbyg", "rb", 9)]


def generate_level(rng, tier):
    """A random picture; the board is the picture scrambled by random handle clicks. Every shift is
    undone by repeating it, so the level is solvable by construction."""
    w, h, colours, side, scramble = TIER_SHAPES[tier]
    cells = [colours[i % len(colours)] for i in range(w * h)]
    rng.shuffle(cells)
    k = 1 if side == "lt" else -1
    board = list(cells)
    for _ in range(scramble):
        if rng.random() < 0.5:
            board = shift_col(board, w, h, rng.randrange(w), k)
        else:
            board = shift_row(board, w, rng.randrange(h), k)
    width = w + 1 + w + 1  # picture, divider, board and one handle column
    if side == "lt":
        rows = ["#" * (width + 2), "#" + "." * w + "#." + "v" * w + "#"]
        for r in range(h):
            rows.append("#" + "".join(c.upper() for c in cells[r * w:(r + 1) * w]) + "#h"
                        + "".join(board[r * w:(r + 1) * w]) + "#")
        rows.append("#" * (width + 2))
    else:
        rows = ["#" * (width + 2)]
        for r in range(h):
            rows.append("#" + "".join(c.upper() for c in cells[r * w:(r + 1) * w]) + "#"
                        + "".join(board[r * w:(r + 1) * w]) + "H#")
        rows.append("#" + "." * w + "#" + "V" * w + ".#")
        rows.append("#" * (width + 2))
    return {"map": rows}
