"""Hazard-index slot allocation (variant ``fixed-2x-index``; budget audit, 26 September).

The arena's intent is to go deep on promising games and to open the search when a game is
stuck; the fixed-2x budgets do not achieve it (docs/validation-20260926/hazard-index-scheduler.md).
After the opening this policy has no level budgets: every game is scored by the expected score it
gains per token over the next horizon, and the slots go to the highest scores (with a minimum
quantum per admission and a hysteresis margin, since a readmitted context is prefilled again).

Everything with units is estimated online from the current run, so the rule carries to game sets
it was not fitted on (the 110 competition games). It is a Gamma-Poisson bandit over games in
relative token time:

- **Scale ``s``.** ``ln 2 x`` all completion tokens spent on levels in this run ``/`` level clears:
  the median time to a clear under a constant rate. Unfinished attempts count as exposure, so the
  opening's fast first clears do not bias it. At the switch at least two clears exist.
- **Clear rate.** A game that has spent ``t`` tokens on its level clears at
  ``rate x shape(t / s)`` per ``s`` tokens. There are three nested estimates:
  - the pooled rate: all clears over shape-weighted exposure, shrunk to one pseudo-clear at the
    shape's peak;
  - one rate per class (a first level, deeper levels), shrunk toward the pooled rate;
  - one rate per game: its own clears over its own exposure, shrunk toward its class rate.

  A game that clears fast gets a high rate (go deep). Tokens spent without a clear lower it (open
  the search).
- **Game value.** The level's weight in the game score, L / (N(N+1)/2), plus the later levels that
  fit in the time left (``s`` tokens per level), each discounted by the game's own chance of
  clearing a level within ``s``.

The one element taken from the recorded runs is ``shape``. It is dimensionless: the pooled clear
rate as a function of tokens spent on the level, in units of each run's own ``s`` (45 Flash xhigh
runs with clears in both classes). It sits at 0.45 of its peak below ``0.25s``, 0.7-1.0 up to
``3s``, and 0.15 beyond. The level-1 and deep curves agree where both are sampled, and the two
classes' scales are equal within noise (median ratio 0.91). The absolute scale varies 6x across
those runs (21-125K tokens for level 1), which is why nothing absolute is kept. ``shape: flat``
switches it off.
"""

import math

SHAPE_EDGES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
SHAPE_RECORDED = (0.45, 0.95, 1.0, 0.80, 0.75, 0.70, 0.15)
DEFAULTS = {"shape": "recorded", "prior_events": 1.0, "class_prior_events": 2.0, "game_prior_events": 1.0,
            "horizon_scale": 0.5, "quantum_scale": 0.4, "hysteresis": 0.25, "stream_tokens_per_second": 110.0,
            "default_levels": 7}


def settings(raw=None):
    merged = {**DEFAULTS, **(raw or {})}
    if set(merged) != set(DEFAULTS):
        raise ValueError(f"Unknown index settings: {sorted(set(merged) - set(DEFAULTS))}")
    if merged["shape"] not in ("recorded", "flat"):
        raise ValueError("shape must be 'recorded' or 'flat'")
    for key, value in merged.items():
        if key == "shape":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Index setting {key} must be a positive number")
    return merged


def shape_values(cfg):
    return SHAPE_RECORDED if cfg["shape"] == "recorded" else (1.0,) * len(SHAPE_RECORDED)


def shape_integral(values, u0, u1):
    """Integral of the piecewise-constant shape over relative time [u0, u1] (last band open)."""
    total = 0.0
    edges = (*SHAPE_EDGES, math.inf)
    for value, a, b in zip(values, edges, edges[1:]):
        lo, hi = max(a, u0), min(b, u1)
        if hi > lo:
            total += value * (hi - lo)
    return total


def attempts_of(game):
    """(class, tokens, cleared) for every level the game has played in this run."""
    rows = []
    for position, tokens in enumerate(game.level_tokens):
        if tokens > 0:  # a multi-level jump: the extra levels took no tokens of their own
            rows.append(("first" if position == 0 else "deep", tokens, True))
    if game.total_levels is None or game.levels < game.total_levels:
        rows.append(("first" if game.levels == 0 else "deep", game.tokens_on_level(), False))
    return rows


class RunModel:
    """Hazard model of the current run, rebuilt from the game records at every tick."""

    def __init__(self, games, cfg):
        self.cfg = cfg
        self.values = shape_values(cfg)
        per_game = {game.game_id: attempts_of(game) for game in games}
        rows = [row for rows in per_game.values() for row in rows]
        clears = sum(done for _, _, done in rows)
        spent = sum(tokens for _, tokens, _ in rows)
        self.scale = max(1000.0, math.log(2) * spent / max(1, clears))
        exposure = {"first": 0.0, "deep": 0.0}
        events = {"first": 0, "deep": 0}
        self.game_exposure, self.game_events = {}, {}
        for game_id, game_rows in per_game.items():
            self.game_exposure[game_id] = sum(self._exposure(tokens) for _, tokens, _ in game_rows)
            self.game_events[game_id] = sum(done for _, _, done in game_rows)
            for cls, tokens, done in game_rows:
                exposure[cls] += self._exposure(tokens)
                events[cls] += done
        prior = cfg["prior_events"]
        # One pseudo-clear per unit of s at the shape's peak (the scale's own definition).
        self.pooled = (sum(events.values()) + prior) / (sum(exposure.values()) + prior / math.log(2))
        weight = cfg["class_prior_events"]
        self.level = {cls: (events[cls] + weight) / (exposure[cls] + weight / self.pooled) for cls in events}
        self.events, self.exposure = events, exposure

    def _exposure(self, tokens):
        return shape_integral(self.values, 0.0, tokens / self.scale)

    def game_rate(self, game):
        cls = "first" if game.levels == 0 else "deep"
        weight = self.cfg["game_prior_events"]
        exposure = self.game_exposure.get(game.game_id)
        if exposure is None:
            exposure = sum(self._exposure(tokens) for _, tokens, _ in attempts_of(game))
            events = sum(done for _, _, done in attempts_of(game))
        else:
            events = self.game_events[game.game_id]
        return (events + weight) / (exposure + weight / self.level[cls]), cls

    def clear_probability(self, rate, spent, horizon):
        """P(clear within the next ``horizon`` tokens | no clear after ``spent`` tokens) at ``rate``."""
        return 1.0 - math.exp(-rate * shape_integral(self.values, spent / self.scale, (spent + horizon) / self.scale))

    def level_value(self, levels, total_levels, remaining_tokens, continuation):
        """Weight of the level being played plus the discounted later levels that fit in the time left."""
        total = int(total_levels or self.cfg["default_levels"])
        attempt = levels + 1
        if attempt > total:
            return 0.0
        weights = total * (total + 1) / 2
        fit = max(1, int(remaining_tokens // self.scale))
        last = min(total, attempt + fit - 1)
        return sum(k / weights * continuation ** (k - attempt) for k in range(attempt, last + 1))

    def quantum(self):
        return self.cfg["quantum_scale"] * self.scale

    def index(self, game, remaining_seconds):
        """Expected score share per 10K tokens over the horizon (the scheduler compares these)."""
        remaining_tokens = max(0.0, remaining_seconds) * self.cfg["stream_tokens_per_second"]
        horizon = min(self.cfg["horizon_scale"] * self.scale, remaining_tokens)
        if horizon <= 0:
            return 0.0, {}
        rate, cls = self.game_rate(game)
        spent = game.tokens_on_level()
        probability = self.clear_probability(rate, spent, horizon)
        continuation = self.clear_probability(rate, 0.0, self.scale)
        value = self.level_value(game.levels, game.total_levels, remaining_tokens, continuation)
        detail = {"class": cls, "rate": round(rate, 4), "spent_tokens": int(spent), "p_clear": round(probability, 4),
                  "continuation": round(continuation, 4), "value": round(value, 4)}
        return probability * value / horizon * 1e4, detail

    def summary(self):
        return {"scale_tokens": round(self.scale), "level": {k: round(v, 4) for k, v in self.level.items()},
                "pooled": round(self.pooled, 4), "events": dict(self.events),
                "exposure": {k: round(v, 2) for k, v in self.exposure.items()}}
