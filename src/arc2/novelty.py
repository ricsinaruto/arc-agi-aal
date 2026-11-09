"""rep1 / rep2 (30 Oct): what each test input shows outside the training inputs' range (colours, size, object count and
sizes, border contact). Misses are mostly right rules with a few wrong cells where the test input differs (base12c:
23 of 50 pass-1 misses within 20 cells); the report points the repair at those situations."""
import collections
def comps(g):
    h, w = len(g), len(g[0]); bg = collections.Counter(c for r in g for c in r).most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]; out = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or g[i][j] == bg: continue
            col = g[i][j]; st = [(i, j)]; seen[i][j] = True; cells = []
            while st:
                a, b = st.pop(); cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and not seen[x][y] and g[x][y] == col:
                        seen[x][y] = True; st.append((x, y))
            out.append((col, cells))
    return bg, out
def feats(g):
    bg, cs = comps(g); h, w = len(g), len(g[0])
    sizes = [len(c) for _, c in cs]
    border = sum(any(a in (0, h - 1) or b in (0, w - 1) for a, b in c) for _, c in cs)
    return dict(colors=set(c for r in g for c in r), h=h, w=w, n=len(cs), smax=max(sizes, default=0), smin=min(sizes, default=0),
                border=border, ncol=len(set(c for r in g for c in r)))
def novelty(task, k=0):
    tr = [feats(p['input']) for p in task['train']]; te = feats(task['test'][k]['input'])
    notes = []
    new = te['colors'] - set().union(*(f['colors'] for f in tr))
    if new: notes.append(f"colours {sorted(new)} appear in the test input but in no training input")
    for key, name in (('h', 'height'), ('w', 'width'), ('n', 'number of objects'), ('smax', 'largest object size'),
                      ('smin', 'smallest object size'), ('border', 'objects touching the border'), ('ncol', 'number of colours')):
        lo, hi = min(f[key] for f in tr), max(f[key] for f in tr)
        if te[key] < lo or te[key] > hi:
            notes.append(f"{name} is {te[key]} in the test input, {lo}-{hi} in the training inputs")
    return notes


def report(task):
    """One line per test input with its notes; a fallback line when nothing lies outside the training range."""
    lines = []
    for k in range(len(task["test"])):
        notes = novelty(task, k)
        head = f"test input {k + 1}: " if len(task["test"]) > 1 else ""
        lines.append(head + ("; ".join(notes) if notes else "no feature outside the training range (check object "
                                                              "boundaries, overlaps and cells near the edges)"))
    return "\n".join(f"- {line}" for line in lines)
