"""Clocks: one wall clock per level bench (and per try), one active-time clock per try (``ActiveClock``).

``ActiveClock`` is duck-typed against ``arena.serving_core.clock.Clock`` (``start``, ``release``, ``elapsed``,
``remaining``). On ``phase.monotonic`` a train phase freezes both, so a try's timings exclude it.
"""

import time
from contextlib import contextmanager


class GlobalClock:
    """A wall clock, started by ``release``."""

    def __init__(self, seconds, now=time.monotonic):
        self.seconds, self.now, self.start = float(seconds), now, None

    def release(self):
        if self.start is not None:
            raise RuntimeError("Wall clock already started")
        self.start = self.now()

    def elapsed(self):
        return 0.0 if self.start is None else max(0.0, self.now() - self.start)

    def remaining(self):
        return max(0.0, self.seconds - self.elapsed())


class ActiveClock:
    """A try's clock: active seconds, inference and tool seconds, completion tokens of every completed request.

    ``budget_clock`` is the study's label (the agent's runtime status names it; "tokens" in every RL study). A level
    bench grants no level budget (``budget_remaining`` is None) and never parks a try (0 parked seconds):
    the try's token budget is the bench's own (level_bench), its wall the global clock's.
    """

    BUDGET_CLOCKS = ("eligible_wall", "inference", "busy", "normalized", "tokens")

    def __init__(self, global_clock, now=time.monotonic, *, budget_clock="eligible_wall"):
        if budget_clock not in self.BUDGET_CLOCKS:
            raise ValueError("Unknown arena budget clock")
        self.budget_clock = budget_clock
        self._completion_total = 0
        self._inference_total, self._inference_started = 0.0, None
        # tool execution: the Duck python tool, including the environment actions it issues
        self._tool_total, self._tool_started, self._tool_depth = 0.0, None, 0
        self.global_clock, self.now = global_clock, now
        self.start = None

    def release(self):
        if self.start is not None:
            raise RuntimeError("Game clock already started")
        self.start = self.now()

    def active_seconds(self):
        return 0.0 if self.start is None else max(0.0, self.now() - self.start)

    def parked_seconds(self):
        return 0.0

    def elapsed(self):
        return self.active_seconds()

    def _inference_wall(self):
        return min(self.global_clock.elapsed(), self.global_clock.seconds)

    def inference_seconds(self):
        total = self._inference_total
        if self._inference_started is not None:
            total += max(0.0, self._inference_wall() - self._inference_started)
        return total

    def tool_seconds(self):
        total = self._tool_total
        if self._tool_started is not None:
            total += max(0.0, self._inference_wall() - self._tool_started)
        return total

    def charge(self, prompt_tokens, completion_tokens, purpose="gameplay"):
        """One completed request's tokens."""
        self._completion_total += max(0, int(completion_tokens))

    def completion_tokens(self):
        return self._completion_total

    def budget_remaining(self):
        return None

    @contextmanager
    def tool(self, purpose="python"):
        """Measure one tool execution (re-entrant: nested calls charge once)."""
        if self.global_clock.start is None:
            raise RuntimeError("Tool time requires a released clock")
        if self._tool_depth:
            self._tool_depth += 1
            try:
                yield
            finally:
                self._tool_depth -= 1
            return
        self._tool_depth = 1
        self._tool_started = self._inference_wall()
        try:
            yield
        finally:
            self._tool_total += max(self._tool_started, self._inference_wall()) - self._tool_started
            self._tool_started = None
            self._tool_depth = 0

    @contextmanager
    def inference(self, purpose="gameplay"):
        """Measure the admitted HTTP generation, including failure/cancellation (tokenization, admission waits and
        tools are outside it). This measures request latency, not exclusive GPU execution time."""
        if self.global_clock.start is None or self._inference_started is not None:
            raise RuntimeError("Inference requires a released clock and one request per game")
        self._inference_started = self._inference_wall()
        try:
            yield
        finally:
            self._inference_total += max(self._inference_started, self._inference_wall()) - self._inference_started
            self._inference_started = None

    def remaining(self):
        return self.global_clock.remaining()
