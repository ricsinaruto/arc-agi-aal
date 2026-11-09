"""Grid rendering and diffing, in the trace release's wire format.

Grids are lists of rows of ints 0..15. The text form matches the released
trajectories exactly: a ``shape=HxW (values 0-15 as hex)`` header followed by
one hex character per cell.
"""

HEX = "0123456789abcdef"


def render(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    lines = [f"shape={rows}x{cols} (values 0-15 as hex)"]
    for row in grid:
        lines.append("".join(HEX[v] for v in row))
    return "\n".join(lines)


def diff_cells(a, b):
    """Cells where grids disagree, as (row, col, a_val, b_val).

    Compares the overlapping region; a shape difference is the caller's to
    report (see diff_summary) — a malformed model prediction must not crash
    the comparison.
    """
    out = []
    for r, (ra, rb) in enumerate(zip(a, b, strict=False)):
        if ra == rb:
            continue
        for c, (va, vb) in enumerate(zip(ra, rb, strict=False)):
            if va != vb:
                out.append((r, c, va, vb))
    return out


def diff_summary(a, b, max_shown=12):
    """One-line-per-cell summary of a grid mismatch, expected vs got."""
    shapes = ((len(a), len(a[0]) if a else 0), (len(b), len(b[0]) if b else 0))
    if shapes[0] != shapes[1]:
        return (f"shape mismatch: expected {shapes[0][0]}x{shapes[0][1]}, "
                f"got {shapes[1][0]}x{shapes[1][1]}")
    cells = diff_cells(a, b)
    if not cells:
        return "grids identical"
    lines = [f"{len(cells)} cell(s) differ (row,col: expected!=got):"]
    for r, c, va, vb in cells[:max_shown]:
        lines.append(f"  ({r},{c}): {HEX[va]}!={HEX[vb]}")
    if len(cells) > max_shown:
        lines.append(f"  ... and {len(cells) - max_shown} more")
    return "\n".join(lines)


def grid_key(grid):
    """Hashable identity for a grid."""
    return bytes(v for row in grid for v in row)
