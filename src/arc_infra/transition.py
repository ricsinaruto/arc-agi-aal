"""Compact before/after summary of one action, attached to every observation.

Duck preloads ``previous_frame``/``transitions`` and tells the model whether the last
action changed the board; B and C only got a ``board_changed`` flag and had to diff
frames themselves. ``summarize`` gives the changed-cell count and bounding box plus which
segmented objects appeared, disappeared or moved (matched by Duck's shape hash).
"""

from collections import OrderedDict, defaultdict

import numpy as np

from .objects import segment_host
from .perception import board_hash

LISTED = 4
# Background regions (more than half the board) reshape whenever a piece moves through
# them; they are reported as a flag, not as objects that appeared or disappeared.
BACKGROUND_PIXELS = 2048


def _origin(node):
    return tuple(node["bbox"][:2])


def _entry(node, **extra):
    return {
        "hash": str(node["hash"])[:8],
        "color": node["color"],
        "pixels": node["pixels"],
        "bbox": list(node["bbox"]),
        **extra,
    }


def _object_changes(before, after):
    """Objects that appeared, disappeared or moved between two segmentations."""
    groups = defaultdict(lambda: ([], []))
    for node in before.get("nodes", []):
        groups[str(node["hash"])][0].append(node)
    for node in after.get("nodes", []):
        groups[str(node["hash"])][1].append(node)
    appeared, disappeared, moved = [], [], []
    background = False
    for old, new in groups.values():
        old_left = {_origin(n): n for n in old}
        new_left = {_origin(n): n for n in new}
        for origin in list(old_left):
            if origin in new_left:  # same shape at the same place: unchanged
                del old_left[origin]
                del new_left[origin]
        old_rest, new_rest = list(old_left.values()), list(new_left.values())
        while old_rest and new_rest:  # greedy nearest pairing = a move
            source = old_rest.pop(0)
            target = min(
                new_rest,
                key=lambda n: abs(n["bbox"][0] - source["bbox"][0])
                + abs(n["bbox"][1] - source["bbox"][1]),
            )
            new_rest.remove(target)
            moved.append(_entry(target, **{"from": list(source["bbox"])}))
        for source, target in ((old_rest, disappeared), (new_rest, appeared)):
            for node in source:
                if node["pixels"] > BACKGROUND_PIXELS:
                    background = True
                else:
                    target.append(_entry(node))
    key = lambda e: (-e["pixels"], e["bbox"])  # noqa: E731
    return sorted(appeared, key=key), sorted(disappeared, key=key), sorted(moved, key=key), background


def summarize(previous, current, *, before_objects=None, after_objects=None, cells=0):
    """Summary dict; ``previous`` may be None for the initial observation."""
    if previous is None:
        return {"changed_cells": None, "change_bbox": None, "note": "initial observation"}
    previous, current = np.asarray(previous), np.asarray(current)
    diff = previous != current
    changed = int(diff.sum())
    summary = {"changed_cells": changed, "change_bbox": None}
    if changed:
        rows, cols = np.nonzero(diff)
        summary["change_bbox"] = [
            int(rows.min()),
            int(cols.min()),
            int(rows.max()) + 1,
            int(cols.max()) + 1,
        ]
        if cells:
            summary["cells"] = [
                [int(r), int(c), int(previous[r, c]), int(current[r, c])]
                for r, c in zip(rows[:cells], cols[:cells])
            ]
    if (
        before_objects is not None
        and after_objects is not None
        and not before_objects.get("skipped")
        and not after_objects.get("skipped")
    ):
        appeared, disappeared, moved, background = _object_changes(before_objects, after_objects)
        summary["objects"] = {
            "appeared": appeared[:LISTED],
            "disappeared": disappeared[:LISTED],
            "moved": moved[:LISTED],
            "counts": {
                "appeared": len(appeared),
                "disappeared": len(disappeared),
                "moved": len(moved),
            },
            "background_reshaped": background,
        }
    return summary


class TransitionCache:
    """Host-side segmentation per board hash; a handful of recent boards is enough."""

    def __init__(self, *, component_limit=200, keep=16):
        self.component_limit, self.keep = component_limit, keep
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def objects(self, board):
        key = board_hash(board)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        objects = self.cache[key] = segment_host(board, limit=self.component_limit)
        while len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return objects
