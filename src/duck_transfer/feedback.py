"""Transition receipts, no-change repeat refusal and stuck notices for the Duck bridge.

Duck's action receipts said only ``board_changed``; the model had to diff frames in Python
to learn what moved. Every executed action now carries a compact ``transition`` (changed
cells, bounding box, objects that appeared, disappeared or moved). An action that already
produced no change on the exact same board is refused instead of executed (``GameSession``
keeps that memory in ``no_ops``), and a level that stalls gets a short stuck notice in the
next prompt.
"""

import time

from arc_infra.perception import board_hash
from arc_infra.transition import TransitionCache, summarize

__all__ = ["StuckTracker", "TransitionCache", "no_effect_here", "transition_summary"]

LISTED = 3


def transition_summary(cache, before, after, *, objects=True):
    """``transition`` receipt for one executed action (lists capped at ``LISTED``)."""
    before_objects = after_objects = None
    if objects:
        before_objects, after_objects = cache.objects(before), cache.objects(after)
    summary = summarize(before, after, before_objects=before_objects, after_objects=after_objects)
    summary.pop("note", None)
    listed = summary.get("objects")
    if listed:
        for key in ("appeared", "disappeared", "moved"):
            listed[key] = listed[key][:LISTED]
    return summary


def label(identity):
    name, x, y = identity
    from inference.agent.action_names import to_model_action

    return f"MOUSE(row={y}, col={x})" if name == "ACTION6" else to_model_action(name)


def no_effect_here(no_ops, levels, board):
    """Actions already tried on this exact board (same level) without any change."""
    key = board_hash(board)
    return [label(identity) for lv, digest, identity in no_ops if lv == levels and digest == key]


class StuckTracker:
    """Per-level counters behind the stuck notice."""

    def __init__(self, cfg, now=None):
        self.reset_level(now)

    def reset_level(self, now=None):
        self.started = time.monotonic() if now is None else now
        self.no_change = self.actions = self.turns = 0
        self.last_notice = None
        self.notices = 0

    def note_action(self, board_changed):
        self.actions += 1
        self.no_change = 0 if board_changed else self.no_change + 1

    def note_turn(self):
        self.turns += 1
