"""One actual gameplay clock, started only after all sessions are ready."""

import time


class Clock:
    def __init__(self, seconds, now=time.monotonic):
        self.seconds, self.now, self.start = seconds, now, None

    def release(self):
        if self.start is not None:
            raise RuntimeError("Gameplay clock already started")
        self.start = self.now()

    def remaining(self):
        return max(0, self.seconds - self.elapsed())

    def elapsed(self):
        return 0 if self.start is None else max(0, self.now() - self.start)

    def feedback(self):
        return {
            "solve_target_seconds": self.seconds,
            "elapsed_seconds": round(self.elapsed(), 3),
            "remaining_seconds": round(self.remaining(), 3),
        }
