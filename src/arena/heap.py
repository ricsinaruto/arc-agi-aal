"""v7.1 (6 October): bounded full-collection pauses in the long-lived arena process.

The fleet's heap grows all game (histories, receipts, logs); every gen-2 collection walks all of it while holding the
GIL (stress reproduction: 0.35 -> 1.1 s pauses within 20 minutes). Every ``FREEZE_SECONDS`` (120 s; 600 s left ~1M unfrozen objects and 0.7 s pauses) the young generations are
collected and everything alive is moved to the permanent generation (``gc.freeze``), so later full collections walk
only what was allocated since. Frozen objects are still freed by reference counting; only cycles among them are not.
The gen-2 pauses are recorded for the live progress (``heap``)."""

import gc
import time

FREEZE_SECONDS = 120


class HeapKeeper:
    def __init__(self, freeze_seconds=FREEZE_SECONDS):
        self.freeze_seconds = freeze_seconds
        self.next_freeze = freeze_seconds
        self.stats = {"gen2": 0, "gen2_ms": 0.0, "gen2_max_ms": 0.0, "freezes": 0, "frozen": 0, "freeze_ms": 0.0}
        self._start = None
        gc.callbacks.append(self._callback)

    def _callback(self, phase, info):
        if info.get("generation") != 2:
            return
        if phase == "start":
            self._start = time.perf_counter()
        elif self._start is not None:
            ms = (time.perf_counter() - self._start) * 1000
            self._start = None
            s = self.stats
            s["gen2"] += 1
            s["gen2_ms"] += ms
            s["gen2_max_ms"] = max(s["gen2_max_ms"], ms)

    def tick(self, elapsed):
        if not self.freeze_seconds or elapsed < self.next_freeze:
            return
        self.next_freeze = elapsed + self.freeze_seconds
        start = time.perf_counter()
        gc.collect(1)
        gc.freeze()
        s = self.stats
        s["freezes"] += 1
        s["frozen"] = gc.get_freeze_count()
        s["freeze_ms"] = round((time.perf_counter() - start) * 1000, 1)

    def summary(self):
        return {k: round(v, 1) if isinstance(v, float) else v for k, v in self.stats.items()}

    def close(self):
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)
