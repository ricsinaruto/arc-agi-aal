"""The ARC-AGI-2 task of this session: load() and submit(answers)."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK, ANSWER = HERE / "task.json", HERE / "answer.json"


def load():
    """{"train": [{"input": grid, "output": grid}, ...], "test": [{"input": grid}, ...]}"""
    return json.loads(TASK.read_text())


def _grid(grid, where):
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if not (isinstance(grid, list) and 1 <= len(grid) <= 30 and all(isinstance(r, list) for r in grid)
            and 1 <= len(grid[0]) <= 30 and all(len(r) == len(grid[0]) for r in grid)):
        raise ValueError(f"{where}: a grid is a rectangular list of 1-30 rows of 1-30 cells")
    out = [[int(c) for c in row] for row in grid]
    if any(not 0 <= c <= 9 or c != v for row, src in zip(out, grid) for c, v in zip(row, src)):
        raise ValueError(f"{where}: cells are integers 0-9")
    return out


def _is_grid(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    return isinstance(x, list) and bool(x) and isinstance(x[0], list) and bool(x[0]) and not isinstance(x[0][0], list)


def submit(answers):
    """answers[i]: one grid or a list of one or two candidate grids for test input i. Replaces any earlier answer."""
    n = len(load()["test"])
    if hasattr(answers, "tolist"):
        answers = answers.tolist()
    if not isinstance(answers, (list, tuple)) or len(answers) != n:
        raise ValueError(f"answers must have one entry per test input ({n})")
    rows = []
    for i, cands in enumerate(answers):
        cands = [cands] if _is_grid(cands) else list(cands)
        if not 1 <= len(cands) <= 2:
            raise ValueError(f"test input {i}: give one or two candidate grids")
        rows.append([_grid(c, f"test input {i}, candidate {k + 1}") for k, c in enumerate(cands)])
    part = ANSWER.with_name(ANSWER.name + ".part")
    part.write_text(json.dumps({"answers": rows}))
    os.replace(part, ANSWER)
    return f"submitted: {n} test input(s), {sum(len(r) for r in rows)} candidate grid(s)"
