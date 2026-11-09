"""Stage Duck's connected-component helper into the model sandboxes.

The implementation is the public Duck module ``src/inference/utils/segmentation.py``.
It is never copied into a second repository location: this module reads that source at
run time and writes it, followed by a small wrapper, into each sandbox (B's persistent
worker, C's and P's ``api`` directory, P's workspace). Duck itself splices the same
source into its Python-tool bootstrap.
"""

from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "inference" / "utils" / "segmentation.py"

HEADER = '''"""Board segmentation helper: segment(board).

segment(board) returns {"nodes": [...], "adjacency_list": [[i, j], ...]} where board is a
64x64 grid of integer colour IDs (a numpy array or nested lists) or the board as text
(a list of equal-length strings, or one string with newlines; the character is the colour).
Each node is one 4-connected same-colour component (background regions included) with:
  id        index in reading order (top-most, then left-most cell); ids change between frames
  color     the cell value (integer ID or character) of the component
  hash      colour+shape signature, translation-invariant only: equal hashes mean the same
            shape and colour anywhere on the board, not the same game entity
  pixels    number of cells
  bbox      [row0, col0, row1, col1] with row1/col1 exclusive
  boundary  clockwise outer-perimeter corner points as [row, col] (holes are not traced)
  children  ids of components directly enclosed by this one
adjacency_list pairs share a 4-connected edge (parent/child pairs included).
Use bbox/boundary for exact click targets (ACTION6 x=col, y=row) instead of guessing a
stride; compare hashes to track pieces across frames; read only small crops of the grid.
This is the public Duck harness helper (Kaggle notebook wuliao0/duck-qwen3-8-anim-base,
ARC3-Inference by Tufa Labs), unmodified, plus the segment() wrapper below.
"""

'''

WRAPPER = """

def _rows(board):
    if isinstance(board, str):
        return [line for line in board.splitlines() if line.strip()]
    return list(board)


def segment(board):
    rows = _rows(board)
    if not rows:
        raise ValueError("segment() needs a non-empty board")
    if isinstance(rows[0], str):
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("board lines must have equal length")
        symbols = sorted({ch for row in rows for ch in row})
        if len(symbols) > 16:
            raise ValueError("more than 16 distinct symbols; pass the board rows only")
        index = {ch: i for i, ch in enumerate(symbols)}
        layer = [[index[ch] for ch in row] for row in rows]
        palette = symbols + [symbols[-1]] * (16 - len(symbols))
    else:
        layer = [[int(v) for v in row] for row in rows]
        if any(not 0 <= v <= 15 for row in layer for v in row):
            raise ValueError("colour IDs must be 0..15")
        palette = list(range(16))
    result = segment_layer(layer, palette)
    for node in result["nodes"]:
        rows_ = [r for r, _ in node["boundary"]]
        cols_ = [c for _, c in node["boundary"]]
        node["bbox"] = [min(rows_), min(cols_), max(rows_) + 1, max(cols_) + 1]
    return result
"""


def source():
    return HEADER + SOURCE.read_text() + WRAPPER


def write(path):
    """Write the staged helper module to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source())
    return path
