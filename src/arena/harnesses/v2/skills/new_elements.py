"""new_elements(): objects on this level's first frame whose colour and shape no earlier level showed (at its start or just before its clear)."""
# ruff: noqa: F821  (current_frame and history are globals of the Duck interpreter)


def new_elements(limit=12):
    """List (color, pixels, bbox) of objects new to this level; probe each once before replaying an old plan."""
    level = current_frame.level
    starts, ends, last = {}, {}, None
    for entry in history:
        frame = entry.frame
        if frame.level not in starts:
            starts[frame.level] = frame
            if last is not None:
                ends[last.level] = last
        last = frame
    seen = set()
    for lv, frame in list(starts.items()) + list(ends.items()):
        if lv < level:
            seen |= {(n["color"], n["hash"]) for n in frame.segmentation["nodes"]}
    if not seen:
        return "no earlier level in history"
    first = starts.get(level, current_frame)
    new = []
    for n in first.segmentation["nodes"]:
        if (n["color"], n["hash"]) in seen or n["pixels"] > 2048:
            continue
        rows = [p[0] for p in n["boundary"]]
        cols = [p[1] for p in n["boundary"]]
        new.append((n["color"], n["pixels"], [min(rows), min(cols), max(rows), max(cols)]))
    new.sort(key=lambda x: -x[1])
    return f"{len(new)} new objects on level {level}: " + "; ".join(f"{c} {p}px bbox {b}" for c, p, b in new[:limit])
