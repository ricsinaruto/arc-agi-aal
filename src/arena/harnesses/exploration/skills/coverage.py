"""coverage(): objects never clicked and key actions not yet tried on this level, with change counts for the ones tried."""
# ruff: noqa: F821  (current_frame, history and valid_actions are globals of the Duck interpreter)
import re


def coverage(top=15):
    """Exploration map of this level: board objects (id, colour, pixels, bbox, click point) with the clicks that hit them and how many changed the board, key actions tried with their change counts, untried objects and actions first. Use it to pick an untested (object, action) pair when a level stalls."""
    frame = current_frame
    if frame is None:
        return "coverage: no frame yet"
    level = frame.level
    boxes = []
    for node in frame.segmentation["nodes"]:
        rows = [p[0] for p in node["boundary"]] or [0]
        cols = [p[1] for p in node["boundary"]] or [0]
        boxes.append((node["id"], node["color"], node["pixels"], min(rows), min(cols), max(rows), max(cols)))
    if boxes:
        background = max(boxes, key=lambda b: b[2])
        boxes = [b for b in boxes if b is not background]
    keys = {}
    hits = {}
    prev = None
    for entry in history:
        after = entry.frame
        if after is None:
            continue
        if prev is None or after.level != level:
            prev = after
            continue
        act = str(entry.action or "")
        changed = int(after._grid != prev._grid)
        prev = after
        if not act:
            continue
        if act.startswith("MOUSE"):
            nums = [int(x) for x in re.findall(r"-?\d+", act)]
            if len(nums) < 2:
                continue
            row, col = nums[0], nums[1]
            inside = [b for b in boxes if b[3] <= row <= b[5] and b[4] <= col <= b[6]]
            key = min(inside, key=lambda b: b[2])[0] if inside else "background"
            count, moved = hits.get(key, (0, 0))
            hits[key] = (count + 1, moved + changed)
        else:
            count, moved = keys.get(act, (0, 0))
            keys[act] = (count + 1, moved + changed)
    lines = [f"level {level}: {len(boxes)} objects; valid actions {list(valid_actions)}"]
    untried_actions = [a for a in valid_actions if a not in keys and not str(a).upper().startswith(("MOUSE", "ACTION6", "CLICK", "RESET"))]
    lines.append("untried key actions: " + (", ".join(untried_actions) or "none"))
    lines.append("key actions tried (times, board changes): " + (", ".join(f"{a} {n}/{m}" for a, (n, m) in sorted(keys.items())) or "none"))
    fresh = [b for b in boxes if b[0] not in hits]
    fresh.sort(key=lambda b: (b[2] > 400, -b[2]))
    lines.append(f"objects never clicked ({len(fresh)}):")
    for b in fresh[:top]:
        lines.append(f"  #{b[0]} {b[1]} {b[2]}px rows {b[3]}-{b[5]} cols {b[4]}-{b[6]} click MOUSE(row={(b[3] + b[5]) // 2}, col={(b[4] + b[6]) // 2})")
    clicked = [(b, hits[b[0]]) for b in boxes if b[0] in hits]
    if clicked or "background" in hits:
        lines.append("objects clicked (times, board changes):")
        for b, (n, m) in clicked[:top]:
            lines.append(f"  #{b[0]} {b[1]} {b[2]}px rows {b[3]}-{b[5]} cols {b[4]}-{b[6]}: {n}/{m}")
        if "background" in hits:
            n, m = hits["background"]
            lines.append(f"  background/empty cells: {n}/{m}")
    return "\n".join(lines)
