"""Object-level frame diff and rotation-aware object signatures for the Duck sandbox (hero-v2 ``dfr_harness``).

Ported from dfranzen/arc-agi-3-milestone-2-solution (Apache-2.0): ``inference/utils/frame_diff.py`` (verbatim logic)
and the rotation-canonical signature of ``inference/utils/segmentation.py``. This module must stay dependency-free
(standard library only, no ``from __future__`` import): its source is spliced into the persistent sandbox bootstrap
(``arena.dfr_harness.with_dfr``), which runs under ``python -I -S``. The host imports it normally for the tests.
"""

import hashlib


def _normalize_cells(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _rotate_cells_90(norm_cells):
    """Rotate a normalized cell set 90 degrees clockwise; returns normalized."""
    max_r = max(r for r, _ in norm_cells)
    return sorted((c, max_r - r) for r, c in norm_cells)


def _cells_hash(norm_cells, color):
    payload = repr((color, norm_cells)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def _object_signature(cells, color):
    """Rotation-canonical identity of an object: ``(canonical_hash, rotation, symmetry, pose_hash, shape_hash)``.

    ``canonical_hash`` is identical for all rotated copies of the same shape and colour; ``rotation`` is the degrees
    clockwise from the canonical form to the observed object (None for 4-fold symmetric shapes, reduced mod 180 for
    2-fold ones); ``symmetry`` is the rotational symmetry order (1, 2 or 4); ``pose_hash`` is the classic
    rotation-sensitive hash of the observed pose; ``shape_hash`` the colour-free pose hash. The reference orientation
    is picked from the colour-free hash, so the same shape reports the same rotation whatever its colour."""
    norm = _normalize_cells(cells)
    variants = [norm]
    for _ in range(3):
        variants.append(_rotate_cells_90(variants[-1]))
    hashes = [_cells_hash(v, color) for v in variants]
    shape_hashes = [_cells_hash(v, "__shape__") for v in variants]
    k = shape_hashes.index(min(shape_hashes))
    canonical = hashes[k]
    distinct = len(set(shape_hashes))
    symmetry = {1: 4, 2: 2, 4: 1}.get(distinct, 1)
    rotation = ((4 - k) % 4) * 90  # observed = canonical rotated this much clockwise
    if symmetry == 4:
        rotation = None
    elif symmetry == 2:
        rotation = rotation % 180
    shape_hash = _cells_hash(norm, "__shape__")
    return canonical, rotation, symmetry, hashes[0], shape_hash


def _node_topleft(node):
    points = node.get("boundary") or []
    if not points:
        return (0, 0)
    return (min(p[0] for p in points), min(p[1] for p in points))


def _hungarian(cost):
    # Minimum-cost assignment (Jonker-Volgonant style with potentials).
    # cost: n x m matrix with n <= m. Returns list row -> column.
    n = len(cost)
    m = len(cost[0]) if n else 0
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = row assigned to column j (1-based; 0 = none)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def _node_bbox(node):
    points = node.get("boundary") or []
    if not points:
        return (0, 0, 0, 0)
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return (min(rows), min(cols), max(rows), max(cols))


def _node_center2(node):
    # bbox center in half-cell units (ints): (r0+r1, c0+c1)
    points = node.get("boundary") or []
    if not points:
        return (0, 0)
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return (min(rows) + max(rows), min(cols) + max(cols))


def _node_entry(node):
    return {"color": node.get("color"), "pixels": node.get("pixels"), "hash": node.get("hash"),
            "shape_hash": node.get("shape_hash"), "at": list(_node_topleft(node)), "bbox": list(_node_bbox(node))}


def compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b, max_group_match=None):
    # Object-level diff between two same-shaped grids with precomputed segmentation nodes. Returns only what
    # changed; unchanged objects are omitted. When max_group_match is set, identical-hash groups larger than it
    # skip pairwise matching and report plain appeared/disappeared entries.
    changed = set()
    for r, (row_a, row_b) in enumerate(zip(grid_a, grid_b)):
        if row_a == row_b:
            continue
        for c, (va, vb) in enumerate(zip(row_a, row_b)):
            if va != vb:
                changed.add((r, c))
    if not changed:
        return {"changed_cell_count": 0, "moved": [], "rotated": [], "appeared": [], "disappeared": [],
                "changed_color": [], "resized": []}

    def touched_nodes(nodes):
        touched = []
        for node in nodes:
            points = node.get("boundary") or []
            if not points:
                continue
            r0 = min(p[0] for p in points)
            r1 = max(p[0] for p in points)
            c0 = min(p[1] for p in points)
            c1 = max(p[1] for p in points)
            # dilate by 1 so objects that shrank/grew still register as touched by the cells they vacated/claimed
            if any(r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 for (r, c) in changed):
                touched.append(node)
        return touched

    before_nodes = touched_nodes(nodes_a)
    after_nodes = touched_nodes(nodes_b)
    moved, appeared, disappeared, rotated = [], [], [], []

    def classify_match(node_a, node_b):
        rot_a = node_a.get("rotation")
        rot_b = node_b.get("rotation")
        symmetry = node_a.get("rotational_symmetry") or 1
        rotated_by = None
        if rot_a is not None and rot_b is not None:
            modulus = 180 if symmetry == 2 else 360
            delta = (rot_b - rot_a) % modulus
            if delta:
                rotated_by = delta
        pa, pb = _node_topleft(node_a), _node_topleft(node_b)
        base_entry = {"color": node_a.get("color"), "pixels": node_a.get("pixels"), "hash": node_a.get("hash")}
        if rotated_by is not None:
            ca, cb = _node_center2(node_a), _node_center2(node_b)
            in_place = abs(ca[0] - cb[0]) <= 2 and abs(ca[1] - cb[1]) <= 2
            base_entry["rotated_by"] = rotated_by
            if in_place:
                base_entry["at"] = list(pb)
                rotated.append(base_entry)
            else:
                base_entry["from"] = list(pa)
                base_entry["to"] = list(pb)
                moved.append(base_entry)
        elif pa != pb:
            base_entry["from"] = list(pa)
            base_entry["to"] = list(pb)
            moved.append(base_entry)
        # same hash, same position, same rotation: unchanged -> omit

    groups_before = {}
    for node in before_nodes:
        groups_before.setdefault(node.get("hash"), []).append(node)
    groups_after = {}
    for node in after_nodes:
        groups_after.setdefault(node.get("hash"), []).append(node)

    for key, group_a in groups_before.items():
        group_b = groups_after.pop(key, [])
        if max_group_match is not None and max(len(group_a), len(group_b)) > max_group_match:
            disappeared.extend(_node_entry(node) for node in group_a)
            appeared.extend(_node_entry(node) for node in group_b)
            continue
        if not group_b:
            disappeared.extend(_node_entry(node) for node in group_a)
            continue
        # optimal assignment over bbox-center distances; rows = smaller side
        transposed = len(group_a) > len(group_b)
        rows, cols = (group_b, group_a) if transposed else (group_a, group_b)
        cost = []
        for row_node in rows:
            cr = _node_center2(row_node)
            cost.append([abs(cr[0] - _node_center2(col_node)[0]) + abs(cr[1] - _node_center2(col_node)[1])
                         for col_node in cols])
        assignment = _hungarian(cost)
        matched_cols = set()
        for row_index, col_index in enumerate(assignment):
            if col_index < 0:
                continue
            matched_cols.add(col_index)
            if transposed:
                classify_match(cols[col_index], rows[row_index])
            else:
                classify_match(rows[row_index], cols[col_index])
        unmatched = [node for index, node in enumerate(cols) if index not in matched_cols]
        if transposed:
            disappeared.extend(_node_entry(node) for node in unmatched)
        else:
            appeared.extend(_node_entry(node) for node in unmatched)
    for group_b in groups_after.values():
        appeared.extend(_node_entry(node) for node in group_b)

    # Phase A - changed_color: EXACT pairs only. Same position, byte-identical shape (colour-independent pose
    # signature), different colour.
    changed_color = []
    used_gone, used_new = set(), set()
    for gi, gone in enumerate(disappeared):
        for ni, new in enumerate(appeared):
            if ni in used_new:
                continue
            if (gone["at"] == new["at"] and gone.get("shape_hash") and gone.get("shape_hash") == new.get("shape_hash")
                    and gone["color"] != new["color"]):
                changed_color.append({"at": gone["at"], "pixels": gone["pixels"], "before_color": gone["color"],
                                      "after_color": new["color"], "before_hash": gone.get("hash"),
                                      "after_hash": new.get("hash")})
                used_gone.add(gi)
                used_new.add(ni)
                break
    disappeared = [entry for index, entry in enumerate(disappeared) if index not in used_gone]
    appeared = [entry for index, entry in enumerate(appeared) if index not in used_new]

    # Phase B - resized: same-colour pairs whose bboxes substantially overlap (overlap area / smaller bbox area
    # >= 0.5); greedy one-to-one by descending overlap then size similarity.
    def bbox_area(b):
        return max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)

    def overlap_ratio(b1, b2):
        inter_r = min(b1[2], b2[2]) - max(b1[0], b2[0]) + 1
        inter_c = min(b1[3], b2[3]) - max(b1[1], b2[1]) + 1
        if inter_r <= 0 or inter_c <= 0:
            return 0.0
        smaller = min(bbox_area(b1), bbox_area(b2))
        return (inter_r * inter_c) / smaller if smaller else 0.0

    candidates = []
    for gi, gone in enumerate(disappeared):
        for ni, new in enumerate(appeared):
            if gone["color"] != new["color"]:
                continue
            bg = gone.get("bbox") or (0, 0, -1, -1)
            bn = new.get("bbox") or (0, 0, -1, -1)
            ratio = overlap_ratio(bg, bn)
            if ratio >= 0.5:
                area_g, area_n = bbox_area(bg), bbox_area(bn)
                similarity = (min(area_g, area_n) / max(area_g, area_n)) if max(area_g, area_n) else 0.0
                candidates.append((ratio, similarity, gi, ni))
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    resized = []
    used_gone, used_new = set(), set()
    for _ratio, _similarity, gi, ni in candidates:
        if gi in used_gone or ni in used_new:
            continue
        used_gone.add(gi)
        used_new.add(ni)
        gone, new = disappeared[gi], appeared[ni]
        resized.append({"at_before": gone["at"], "at_after": new["at"], "color": gone["color"],
                        "before_pixels": gone["pixels"], "after_pixels": new["pixels"],
                        "before_hash": gone.get("hash"), "after_hash": new.get("hash")})
    disappeared = [entry for index, entry in enumerate(disappeared) if index not in used_gone]
    appeared = [entry for index, entry in enumerate(appeared) if index not in used_new]
    return {"changed_cell_count": len(changed), "moved": moved, "rotated": rotated, "appeared": appeared,
            "disappeared": disappeared, "changed_color": changed_color, "resized": resized}
