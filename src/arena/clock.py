"""Clocks: one wall clock per trial and one active-time clock per game.

``ActiveClock`` is duck-typed against ``qwen_baselines.clock.Clock`` (``start``, ``release``,
``elapsed``, ``remaining``, ``feedback``) so the C/R bridges and request layers accept it. Its
``elapsed()`` counts active seconds only, so a parked game's level timings exclude the pause.
"""

import asyncio
import math
import time
from contextlib import contextmanager


class GlobalClock:
    """The trial's wall clock, released once every game is ready."""

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

    def feedback(self):
        return {
            "solve_target_seconds": self.seconds,
            "elapsed_seconds": round(self.elapsed(), 3),
            "remaining_seconds": round(self.remaining(), 3),
        }


class Normalization:
    """Equivalent seconds for one request under the normalized budget clock.

    cost = request_overhead_seconds
           + completion_tokens / rate_tokens_per_second * (1 + context_weight * prompt_tokens / context_reference)

    Defaults come from 1,131 phase-2 gameplay requests in four fixed-2x runs (25 September):
    a least-squares fit of request latency gave 0.00958 s per completion token (104 tok/s per
    stream at two streams) and 1.65 s per request. The same fit found no per-token slowdown
    with prompt size at two streams (the hybrid-attention model decodes 100K prompts as fast
    as 10K ones), so the context term is not a latency estimate: it charges the cache the
    request occupies, which is what keeps an audition pool out. It is a knob, recorded.
    """

    def __init__(self, rate_tokens_per_second=104.0, context_weight=0.25, context_reference=131072,
                 request_overhead_seconds=1.65):
        values = (rate_tokens_per_second, context_weight, context_reference, request_overhead_seconds)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Normalization parameters must be finite numbers")
        if rate_tokens_per_second <= 0 or context_reference <= 0 or context_weight < 0 or request_overhead_seconds < 0:
            raise ValueError("Normalization needs a positive rate and reference and non-negative weights")
        self.rate_tokens_per_second = float(rate_tokens_per_second)
        self.context_weight = float(context_weight)
        self.context_reference = float(context_reference)
        self.request_overhead_seconds = float(request_overhead_seconds)

    def cost(self, prompt_tokens, completion_tokens):
        prompt, completion = max(0, int(prompt_tokens)), max(0, int(completion_tokens))
        scale = 1.0 + self.context_weight * prompt / self.context_reference
        return self.request_overhead_seconds + completion / self.rate_tokens_per_second * scale

    def summary(self):
        return {"rate_tokens_per_second": self.rate_tokens_per_second, "context_weight": self.context_weight,
                "context_reference": self.context_reference,
                "request_overhead_seconds": self.request_overhead_seconds}


class ActiveClock:
    """Per-game clock counting active seconds; parked time is excluded.

    ``budget`` uses the selected ``budget_clock`` from ``budget_base``. The legacy
    eligible-wall clock is available for historical reproduction; ``inference``, ``busy`` and
    ``normalized`` reproduce the 23-28 September studies; new builds select ``tokens``.
    ``None`` means unbounded (opening phase).
    ``remaining()`` is the wall clock's; the budget only feeds the scheduler and ``feedback()``.

    Token clock ("It should definitely be token based to make it
    comparable with different number of running streams"): the budget unit is the completion
    tokens (reasoning included) of the game's completed requests whose purpose is in
    ``token_purposes`` (default: gameplay replies only; compaction summaries and notes/memory
    asides are harness overhead whose volume depends on the window, 2-4% of completion tokens at
    64K and 14% at 42K in the recorded c110 runs). A busy second bought ~23 tokens per stream in
    the opening and 64 / 40 / 30 after the switch at 6 / 14 / 21 focus games; a token is a token at
    any concurrency. Tokens are charged when a reply completes (``charge``), so a parked game
    accrues nothing; tool execution is not charged. ``budget_seconds()`` keeps its name for the
    scheduler and the records but returns tokens on this clock (``budget_unit``).
    """

    BUDGET_CLOCKS = ("eligible_wall", "inference", "busy", "normalized", "tokens")
    TOKEN_PURPOSES = ("gameplay",)

    def __init__(self, global_clock, now=time.monotonic, *, budget_clock="eligible_wall", normalization=None,
                 token_purposes=None):
        if budget_clock not in self.BUDGET_CLOCKS:
            raise ValueError("Unknown arena budget clock")
        self.budget_clock = budget_clock
        if token_purposes is not None and budget_clock != "tokens":
            raise ValueError("Token purposes apply to the token budget clock only")
        purposes = self.TOKEN_PURPOSES if token_purposes is None else tuple(token_purposes)
        if not purposes or any(type(p) is not str or not p for p in purposes):
            raise ValueError("token_purposes must be a non-empty list of request purposes")
        self.token_purposes = purposes if budget_clock == "tokens" else None
        self._budget_tokens = 0  # completion tokens of the counted purposes (token clock)
        # Normalized clock: budgets are charged in equivalent
        # seconds computed from the tokens a request generated, not from its latency, so
        # contention (the opening's five streams, an audition pool beside a focus game) does
        # not change what a budget buys or what T1 measures. Tool execution stays wall time.
        self.normalization = None
        if budget_clock == "normalized":
            self.normalization = Normalization(**(normalization or {}))
        elif normalization is not None:
            raise ValueError("Cost normalization applies to the normalized budget clock only")
        self.charges = []
        self._charged_total = 0.0
        self._completion_total = 0  # every completed request's completion tokens (hazard index)
        self.inference_intervals = []
        self._inference_total = 0.0
        self._inference_started = None
        # Tool execution (24 September ``busy`` clock): the Duck python tool, including the
        # environment actions it issues, measured on the same wall basis as inference.
        self.tool_intervals = []
        self._tool_total = 0.0
        self._tool_started = None
        self._tool_depth = 0
        self.global_clock, self.now = global_clock, now
        self.start = None
        self.parked = False
        self._active_before = 0.0  # active seconds accumulated before the current segment
        self._segment_started = None  # wall time the current active segment began
        self._parked_since = None
        self._parked_total = 0.0
        self.budget = None
        self.budget_base = 0.0
        self._event = asyncio.Event()
        self._event.set()

    @property
    def seconds(self):
        return self.global_clock.seconds

    def release(self):
        if self.start is not None:
            raise RuntimeError("Game clock already started")
        self.start = self.now()
        if not self.parked:
            self._segment_started = self.start
        else:
            self._parked_since = self.start

    def active_seconds(self):
        if self._segment_started is None:
            return self._active_before
        return self._active_before + max(0.0, self.now() - self._segment_started)

    def parked_seconds(self):
        if self._parked_since is None:
            return self._parked_total
        return self._parked_total + max(0.0, self.now() - self._parked_since)

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

    def busy_seconds(self):
        return self.inference_seconds() + self.tool_seconds()

    def charge(self, prompt_tokens, completion_tokens, purpose="gameplay"):
        """Charge one completed request's tokens (normalized clock); other clocks only record."""
        cost = None if self.normalization is None else self.normalization.cost(prompt_tokens, completion_tokens)
        self.charges.append({
            "wall_seconds": round(self._inference_wall(), 3), "purpose": purpose,
            "prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens),
            "cost_seconds": None if cost is None else round(cost, 3),
        })
        if cost is not None:
            self._charged_total += cost
        self._completion_total += max(0, int(completion_tokens))
        if self.token_purposes is not None and purpose in self.token_purposes:
            self._budget_tokens += max(0, int(completion_tokens))
        return cost

    def completion_tokens(self):
        return self._completion_total

    def budget_tokens(self):
        """Completion tokens of the counted purposes (the token clock's budget time; 0 on other clocks)."""
        return self._budget_tokens

    @property
    def budget_unit(self):
        return "completion_tokens" if self.budget_clock == "tokens" else "seconds"

    def normalized_seconds(self):
        """Charged equivalent seconds plus tool time; only the normalized clock budgets on it."""
        return self._charged_total + self.tool_seconds()

    def budget_seconds(self):
        if self.budget_clock == "inference":
            return self.inference_seconds()
        if self.budget_clock == "busy":
            return self.busy_seconds()
        if self.budget_clock == "normalized":
            return self.normalized_seconds()
        if self.budget_clock == "tokens":
            return float(self._budget_tokens)
        return self.active_seconds()

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
        status = "completed"
        try:
            yield
        except BaseException as exc:
            status = type(exc).__name__
            raise
        finally:
            ended = max(self._tool_started, self._inference_wall())
            self._tool_total += ended - self._tool_started
            self.tool_intervals.append({
                "started_wall_seconds": self._tool_started,
                "ended_wall_seconds": ended,
                "purpose": purpose, "status": status,
            })
            self._tool_started = None
            self._tool_depth = 0

    @contextmanager
    def inference(self, purpose="gameplay"):
        """Measure the admitted HTTP generation, including failure/cancellation.

        Tokenization, admission waits and tools are outside this scope. An already
        issued request keeps costing time while parked; cleanup past wall-end does not.
        This measures request latency, not exclusive GPU execution time.
        """
        if self.global_clock.start is None or self._inference_started is not None:
            raise RuntimeError("Inference requires a released clock and one request per game")
        self._inference_started = self._inference_wall()
        status = "completed"
        try:
            yield
        except BaseException as exc:
            status = type(exc).__name__
            raise
        finally:
            ended = max(self._inference_started, self._inference_wall())
            self._inference_total += ended - self._inference_started
            self.inference_intervals.append({
                "started_wall_seconds": self._inference_started,
                "ended_wall_seconds": ended,
                "purpose": purpose, "status": status,
            })
            self._inference_started = None

    def pause(self):
        if self.parked:
            return
        self.parked = True
        self._event.clear()
        if self._segment_started is not None:
            self._active_before += max(0.0, self.now() - self._segment_started)
            self._segment_started = None
        if self.start is not None:
            self._parked_since = self.now()

    def resume(self):
        if not self.parked:
            return
        self.parked = False
        self._event.set()
        if self._parked_since is not None:
            self._parked_total += max(0.0, self.now() - self._parked_since)
            self._parked_since = None
        if self.start is not None:
            self._segment_started = self.now()

    def grant(self, budget):
        """Set a budget in the selected clock counted from now; ``None`` = unbounded."""
        self.budget = None if budget is None else float(budget)
        self.budget_base = self.budget_seconds()

    def budget_spent(self):
        return self.budget_seconds() - self.budget_base

    def budget_remaining(self):
        if self.budget is None:
            return None
        return max(0.0, self.budget - self.budget_spent())

    def budget_exhausted(self):
        return self.budget is not None and self.budget_spent() >= self.budget

    def remaining(self):
        # The wall clock is the only hard deadline every agent, bridge and request layer
        # sees: a level budget never cuts a request or a step; the scheduler parks the game
        # between requests instead (``budget_exhausted``), and parking blocks at the gate.
        return self.global_clock.remaining()

    async def wait_active(self):
        await self._event.wait()

    def feedback(self):
        left = self.budget_remaining()
        if self.budget_clock == "tokens":
            budget = {"level_budget_tokens": None if self.budget is None else int(self.budget),
                      "level_budget_remaining_tokens": None if left is None else int(left)}
        else:
            budget = {"level_budget_seconds": None if self.budget is None else round(self.budget, 3),
                      "level_budget_remaining_seconds": None if left is None else round(left, 3)}
        return {
            **self.global_clock.feedback(),
            "game_active_seconds": round(self.active_seconds(), 3),
            "game_inference_seconds": round(self.inference_seconds(), 3),
            "game_tool_seconds": round(self.tool_seconds(), 3),
            "game_normalized_seconds": round(self.normalized_seconds(), 3),
            "level_budget_clock": self.budget_clock,
            **budget,
        }
