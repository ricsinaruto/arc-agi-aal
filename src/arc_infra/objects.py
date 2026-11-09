"""Host-side object view of a board: Duck's segmentation rendered as compact text.

Duck pushes ``current_frame.segmentation`` into every python call and hides the raw
grid; B and C had the same helper as an import and used it in 0–12 of their 30–56 tool
calls per game (v3). ``segment_host`` runs the public segmentation on the host and
``object_view`` renders it in well under a thousand tokens so every observation can
carry it.
"""

from collections import Counter, deque

from inference.utils.segmentation import segment_layer

# Boards with more components than this (checkerboards, noise) take seconds to segment
# because containment is a flood fill per component; the view is skipped instead.
COMPONENT_LIMIT = 400


def component_count(grid):
    """Number of 4-connected same-colour components (a few milliseconds at 64x64)."""
    rows = [list(map(int, row)) for row in grid]
    height, width = len(rows), len(rows[0]) if rows else 0
    seen = [[False] * width for _ in range(height)]
    count = 0
    for r in range(height):
        for c in range(width):
            if seen[r][c]:
                continue
            count += 1
            colour = rows[r][c]
            queue = deque([(r, c)])
            seen[r][c] = True
            while queue:
                y, x = queue.popleft()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not seen[ny][nx]:
                        if rows[ny][nx] == colour:
                            seen[ny][nx] = True
                            queue.append((ny, nx))
    return count


def segment_host(grid, *, limit=COMPONENT_LIMIT):
    """``{"nodes": [...], "adjacency_list": [...]}`` with bbox, or ``{"skipped": n}``."""
    layer = [[int(v) for v in row] for row in grid]
    count = component_count(layer)
    if count > limit:
        return {"skipped": count, "nodes": [], "adjacency_list": []}
    result = segment_layer(layer, list(range(16)))
    for node in result["nodes"]:
        rows = [r for r, _ in node["boundary"]]
        cols = [c for _, c in node["boundary"]]
        node["bbox"] = [min(rows), min(cols), max(rows) + 1, max(cols) + 1]
    return result


def object_view(objects, *, limit=40):
    """Render the largest ``limit`` objects as one line each, largest first."""
    if objects is None:
        return "objects: unavailable"
    if objects.get("skipped"):
        return (
            f"objects: {objects['skipped']} components, too many to list; the board is "
            "noise-like or a fine pattern, read small crops instead"
        )
    nodes = objects["nodes"]
    by_colour = Counter(node["color"] for node in nodes)
    header = f"objects: {len(nodes)} (by colour: " + " ".join(
        f"{colour}x{count}" for colour, count in sorted(by_colour.items())
    ) + ")"
    ordered = sorted(nodes, key=lambda node: (-node["pixels"], node["id"]))
    lines = [header]
    for node in ordered[:limit]:
        line = (
            f"#{node['id']} c={node['color']} px={node['pixels']} "
            f"bbox={node['bbox']} hash={str(node['hash'])[:8]}"
        )
        if node.get("children"):
            kids = node["children"]
            line += f" kids={kids[:8]}" + (f"+{len(kids) - 8}" if len(kids) > 8 else "")
        lines.append(line)
    if len(ordered) > limit:
        lines.append(f"... {len(ordered) - limit} smaller objects omitted")
    lines.append(
        "bbox=[row0,col0,row1,col1] (exclusive); click a cell inside with ACTION6 x=col,y=row; "
        "equal hashes are the same shape and colour"
    )
    return "\n".join(lines)
