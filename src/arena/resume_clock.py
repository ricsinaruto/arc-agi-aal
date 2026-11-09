"""A shiftable ``time.monotonic`` for a resumed trial (``arena.checkpoint``).

A restored trial holds monotonic timestamps of the process that wrote the checkpoint (clock starts, deadlines, last
observation times). ``install()`` runs before any arena module binds ``time.monotonic`` (``arena/__init__``, resumed
processes only); ``set(value)`` then makes the clock read ``value`` now, so the restored timestamps stay valid and the
play clock continues from the checkpoint."""

import time

REAL = time.monotonic
SHIFT = [0.0]


def monotonic():
    return REAL() + SHIFT[0]


def install():
    time.monotonic = monotonic


def installed():
    return time.monotonic is monotonic


def set(value):  # noqa: A001
    SHIFT[0] = float(value) - REAL()
