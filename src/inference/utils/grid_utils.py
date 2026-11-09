"""Grid formatting helpers."""
from __future__ import annotations

from typing import Sequence

ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
ARC_COLOR_LEGEND = (
    "W=white, w=light gray, g=gray, G=dark gray, c=charcoal, B=black, "
    "M=magenta, P=pink, R=red, b=blue, S=sky blue, Y=yellow, O=orange, "
    "r=dark red, N=light green, p=purple"
)


_ASCII_TABLE = bytes.maketrans(bytes(range(16)), ARC_COLOR_CHARS.encode())


def format_grid_ascii(grid: Sequence[Sequence[int]]) -> str:
    if not grid:
        return "(empty grid)"
    fast = _format_fast(grid)  # arena v7.1: one table lookup instead of a python loop per cell (same text)
    if fast is not None:
        return fast
    lines = []
    for row in grid:
        chars = []
        for v in row:
            chars.append(ARC_COLOR_CHARS[max(0, min(15, int(v)))])
        lines.append("".join(chars))
    return "\n".join(lines)


def _format_fast(grid):
    try:
        import numpy as np

        array = np.asarray(grid)
        if array.ndim != 2 or not array.size or array.dtype.kind not in "iu":
            return None
        data = np.clip(array, 0, 15).astype(np.uint8).tobytes().translate(_ASCII_TABLE).decode()
    except (TypeError, ValueError, ImportError):
        return None
    width = array.shape[1]
    return "\n".join(data[i:i + width] for i in range(0, len(data), width))
