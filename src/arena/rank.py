"""Continuous token- and clear-based ranking of the games (``hero-v2-rank``).

"we need a better token (cost) and level clear-based continuous ranking of each game, and then
update the focus slots e.g. every 1 minute. The ranking should prioritize games that cleared levels recently
(according to token count), and if token counts are similar then games with deeper levels."

The state machine lives in ``arena.scheduler.Scheduler`` (``budget_policy="rank"``); this module holds the settings
and the score. Tuned in the calibrated simulator (``scripts/arena_analysis/sim_hero_v2.py``,
docs/validation-20260927/sim-hero-v2.md: random search on two held-out halves of the games, 9-hour objective).

- No opening and no level budgets: ``slots`` focus games from the release.
- Every ``tick_seconds`` of wall time every unfinished game is scored (``score``): the value of its next level times
  its recency-weighted clears per completion token, both on the game's own token clock::

      rate  = (prior_clears + sum_i exp(-age_i / H)) / (prior_tokens + H (1 - exp(-tokens / H)))
      value = (levels + 1) ** depth_power / (N (N + 1) / 2)

  ``age_i`` = the completion tokens the game has generated since its i-th clear, ``tokens`` = all its completion
  tokens, ``N`` = its level count (``default_levels`` until the game reports it). ``H`` = ``horizon_rel`` x the median
  completion tokens per cleared level over all games so far (floor ``horizon_floor_tokens``; ``horizon_tokens``
  before the first clear, or always when ``horizon_rel`` is None). A game that has never played scores the prior
  rate ``prior_clears / prior_tokens`` times its value.
- Free focus slots take the best waiting games (``rank_fill``). Then, while the best waiting game scores more than
  ``1 + hysteresis`` times the worst focus game that has generated ``tenure_tokens`` since its admission, the focus
  game parks (``rank``) and the waiting game is admitted (``rank``); a swap re-prefills the incoming game.
- Between ticks a slot freed by a finished game refills from the latest ranking.

The tick (29 September tick study, sim-hero-v2.md §8): 60 s (180 s is equivalent). A 600 s tick re-prefills less
(0.4-0.8% of phase-2 GPU time against 1.0-1.4%) but lets a stalled focus game hold its slot ~5 minutes longer, and
costs 0.4-0.8 points over 9 hours at 23-25 streams. The tenure, not the tick, is the knob for cache reuse: a game
becomes swappable at the first tick after its ``tenure_tokens``.
"""

import math
import statistics

DEFAULTS = {
    "slots": 25,
    "window": 65536,
    "tick_seconds": 60.0,
    "horizon_tokens": 40_000,
    "horizon_rel": 1.0,
    "horizon_floor_tokens": 2_000,
    "prior_clears": 0.25,
    "prior_tokens": 5_000,
    "depth_power": 1.5,
    "hysteresis": 0.05,
    "tenure_tokens": 10_000,
    "clears": "all",
    "default_levels": 8,
}
CLEARS = ("all", "last")


def settings(raw=None):
    """Validated ``rank`` settings."""
    merged = {**DEFAULTS, **(raw or {})}
    unknown = set(merged) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown rank settings: {sorted(unknown)}")

    def integer(key, low, high=math.inf):
        value = merged[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"rank.{key} must be an integer in [{low}, {high}]")

    def number(key, low, high=math.inf, low_open=False):
        value = merged[key]
        if (type(value) not in (int, float) or not math.isfinite(value) or value > high
                or (value <= low if low_open else value < low)):
            raise ValueError(f"rank.{key} must be a finite number {'>' if low_open else '>='} {low}")

    integer("slots", 1, 64)
    integer("window", 4096)
    number("tick_seconds", 1.0)
    integer("horizon_tokens", 1)
    if merged["horizon_rel"] is not None:
        number("horizon_rel", 0.0, low_open=True)
    integer("horizon_floor_tokens", 1)
    number("prior_clears", 0.0, low_open=True)
    number("prior_tokens", 0.0, low_open=True)
    number("depth_power", 0.0)
    number("hysteresis", 0.0)
    integer("tenure_tokens", 0)
    integer("default_levels", 1)
    if merged["clears"] not in CLEARS:
        raise ValueError(f"rank.clears must be one of {CLEARS}")
    return merged


def horizon(games, cfg):
    """The recency horizon H in completion tokens."""
    if cfg["horizon_rel"] is None:
        return float(cfg["horizon_tokens"])
    cleared = [t for g in games for t in g.level_tokens if t > 0]
    if not cleared:
        return float(cfg["horizon_tokens"])
    return max(cfg["horizon_rel"] * float(statistics.median(cleared)), float(cfg["horizon_floor_tokens"]))


def score(game, h, cfg):
    """(score, evidence) of one game; see the module docstring."""
    ages, acc = [], float(game.tokens_on_level())
    for tokens in reversed(game.level_tokens):
        ages.append(acc)
        acc += tokens
    if cfg["clears"] == "last":
        ages = ages[:1]
    spent = float(game.clock.completion_tokens())
    clears = cfg["prior_clears"] + sum(math.exp(-a / h) for a in ages)
    exposure = cfg["prior_tokens"] + h * (1.0 - math.exp(-spent / h))
    n = game.total_levels or cfg["default_levels"]
    value = (game.levels + 1) ** cfg["depth_power"] / (n * (n + 1) / 2)
    rate = clears / exposure
    return value * rate, {"levels": game.levels, "tokens": int(spent), "tokens_on_level": int(game.tokens_on_level()),
                          "rate_per_10k": round(rate * 1e4, 4), "value": round(value, 5)}
