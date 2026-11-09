"""Grid helpers preloaded in the python tool and in programs (setting ``helpers``). Pure numpy/scipy; every function
takes and returns grids as lists of lists or numpy arrays (returns numpy int arrays). Loaded by ``arc2.worker`` by
path (no package imports)."""

import numpy as np
from scipy import ndimage

# HELP (the prompt's list): arc2.tasks.HELPERS


_EIGHT = {
    "identity": lambda a: a, "rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3), "flipud": np.flipud, "fliplr": np.fliplr, "transpose": lambda a: a.T,
    "antitranspose": lambda a: np.rot90(a, 2).T,
}


def _a(g):
    return np.array(g, dtype=int)


def background(g):
    values, counts = np.unique(_a(g), return_counts=True)
    return int(values[np.argmax(counts)])


def colours(g):
    values, counts = np.unique(_a(g), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def bbox(g, bg=None):
    a = _a(g)
    bg = background(a) if bg is None else bg
    rows, cols = np.nonzero(a != bg)
    if not len(rows):
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def crop(g, box):
    r0, c0, r1, c1 = box
    return _a(g)[r0:r1 + 1, c0:c1 + 1].copy()


def objects(g, bg=None, diag=False, multicolour=False):
    a = _a(g)
    bg = background(a) if bg is None else bg
    structure = np.ones((3, 3), int) if diag else None
    found = []
    masks = [a != bg] if multicolour else [a == c for c in np.unique(a) if c != bg]
    for mask in masks:
        labels, n = ndimage.label(mask, structure=structure)
        for k in range(1, n + 1):
            rows, cols = np.nonzero(labels == k)
            r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
            part = labels[r0:r1 + 1, c0:c1 + 1] == k
            sub = np.where(part, a[r0:r1 + 1, c0:c1 + 1], bg)
            vals, counts = np.unique(a[rows, cols], return_counts=True)
            found.append({"colour": int(vals[np.argmax(counts)]), "cells": list(zip(rows.tolist(), cols.tolist())),
                          "size": int(len(rows)), "box": (r0, c0, r1, c1), "grid": sub, "mask": part})
    found.sort(key=lambda o: (o["box"][0], o["box"][1]))
    return found


def panels(g):
    a = _a(g)
    h, w = a.shape
    full_rows = [r for r in range(h) if len(set(a[r].tolist())) == 1]
    full_cols = [c for c in range(w) if len(set(a[:, c].tolist())) == 1]
    seps = {int(a[r, 0]) for r in full_rows} | {int(a[0, c]) for c in full_cols}
    for colour in seps:  # separators: lines of one colour that cut the grid; a panel's own uniform rows are not
        rows = [r for r in full_rows if a[r, 0] == colour]
        cols = [c for c in full_cols if a[0, c] == colour]
        if not rows and not cols:
            continue

        def spans(cuts, n):
            out, start = [], 0
            for cut in cuts + [n]:
                if cut > start:
                    out.append((start, cut))
                start = cut + 1
            return out

        rs, cs = spans(rows, h), spans(cols, w)
        if len(rs) * len(cs) > 1:
            return [(r0, c0, a[r0:r1, c0:c1].copy()) for r0, r1 in rs for c0, c1 in cs]
    return [(0, 0, a.copy())]


def split(g, rows, cols):
    a = _a(g)
    h, w = a.shape
    return [[a[i * h // rows:(i + 1) * h // rows, j * w // cols:(j + 1) * w // cols].copy() for j in range(cols)]
            for i in range(rows)]


def transform8(g, name):
    return np.ascontiguousarray(_EIGHT[name](_a(g)))


def symmetries(g):
    a = _a(g)
    return [name for name, f in _EIGHT.items() if f(a).shape == a.shape and np.array_equal(f(a), a)]


def period(g, unknown=None):
    a = _a(g)
    h, w = a.shape
    known = np.ones_like(a, bool) if unknown is None else a != unknown

    def fits(p, axis):
        n = h if axis == 0 else w
        for start in range(p):
            idx = list(range(start, n, p))
            block = a[idx] if axis == 0 else a[:, idx]
            kn = known[idx] if axis == 0 else known[:, idx]
            for line in range(block.shape[1 - axis]):
                vals = (block[:, line] if axis == 0 else block[line])[(kn[:, line] if axis == 0 else kn[line])]
                if len(set(vals.tolist())) > 1:
                    return False
        return True

    return (next(p for p in range(1, h + 1) if fits(p, 0)), next(p for p in range(1, w + 1) if fits(p, 1)))


def complete(g, unknown):
    a = _a(g)
    out = a.copy()
    hole = a == unknown
    pr, pc = period(a, unknown)
    if (pr, pc) != a.shape:
        for r, c in zip(*np.nonzero(hole)):
            same = a[r % pr::pr, c % pc::pc]
            vals = same[same != unknown]
            if len(vals):
                out[r, c] = np.bincount(vals).argmax()
    for name in ("fliplr", "flipud", "rot180", "transpose", "antitranspose", "rot90", "rot270"):
        if not (out == unknown).any():
            break
        f = _EIGHT[name]
        m = f(out)
        if m.shape != out.shape:
            continue
        ok = (out != unknown) & (m != unknown)
        if ok.any() and np.array_equal(out[ok], m[ok]):
            fill = (out == unknown) & (m != unknown)
            out[fill] = m[fill]
    return out


def enclosed(g, bg=None):
    a = _a(g)
    bg = background(a) if bg is None else bg
    labels, n = ndimage.label(a == bg)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    return (labels > 0) & ~np.isin(labels, list(border))


def paste(g, sub, r, c, transparent=None):
    a, s = _a(g).copy(), _a(sub)
    h, w = a.shape
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if 0 <= r + i < h and 0 <= c + j < w and (transparent is None or s[i, j] != transparent):
                a[r + i, c + j] = s[i, j]
    return a


def recolour(g, mapping):
    a = _a(g)
    out = a.copy()
    for old, new in mapping.items():
        out[a == old] = new
    return out


def line(g, r0, c0, r1, c1, colour):
    a = _a(g).copy()
    n = max(abs(r1 - r0), abs(c1 - c0))
    dr, dc = (r1 > r0) - (r1 < r0), (c1 > c0) - (c1 < c0)
    if r0 != r1 and c0 != c1 and abs(r1 - r0) != abs(c1 - c0):
        raise ValueError("line: straight or 45-degree only")
    for k in range(n + 1):
        r, c = r0 + k * dr, c0 + k * dc
        if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            a[r, c] = colour
    return a


def upscale(g, k):
    return np.kron(_a(g), np.ones((k, k), int))


def downscale(g, k):
    a = _a(g)
    h, w = a.shape[0] // k, a.shape[1] // k
    return np.array([[np.bincount(a[i * k:(i + 1) * k, j * k:(j + 1) * k].ravel()).argmax() for j in range(w)]
                     for i in range(h)], dtype=int)


def diff(a, b):
    a, b = _a(a), _a(b)
    if a.shape != b.shape:
        raise ValueError(f"diff: shapes {a.shape} and {b.shape}")
    return [(int(r), int(c), int(a[r, c]), int(b[r, c])) for r, c in zip(*np.nonzero(a != b))]


NAMES = ["background", "colours", "bbox", "crop", "objects", "panels", "split", "transform8", "symmetries", "period",
         "complete", "enclosed", "paste", "recolour", "line", "upscale", "downscale", "diff"]
