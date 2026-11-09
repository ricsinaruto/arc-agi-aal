"""Bayesian Gittins-style slot allocation (variant ``fixed-2x-gittins-broad``; 27 September).

Screened offline in the trace-driven simulator (``scripts/arena_analysis/schedule_sim.py``,
docs/validation-20260927/schedule-sim.md): with the recorded throughput, a clairvoyant scheduler
reaches ~11% on the 23 games and one that knows every level's true clear rate ~8.6%, while the
index and commit policies reach 4-5%. The gap is identifying the deep games: at the two-clear
switch 21 games have ~5K tokens each, inside the warm-up band where a level rarely clears. This
policy pairs a longer broad opening (the scheduler's ``opening_min_tokens``) with an index that
learns from every level a game plays.

Model. A game's clear hazard on level L after ``t`` tokens on it is
``H0_class(t) x m_g^p x e_L``: ``H0`` is the pooled hazard per token band (first level / deeper
levels; generic constants from the recorded Flash xhigh attempts over all public games, not per game),
``m_g ~ Gamma(k, k)`` is the game's factor shared by its levels (``p`` = 1 on the first level,
``beta`` deeper; recorded log first-level and deep factors correlate 0.56), and
``e_L ~ Gamma(ke, ke)`` is the level's own factor. The posterior of ``m_g`` is kept on a log grid
from this run only: every cleared level of the game and the censored current level (``e``
integrated out in closed form).

Index. The best expected score per token over plans "play up to ``j`` more levels, giving up on a
level once ``c`` tokens have been spent on it" (Monte Carlo over the posterior, common random
numbers across games): a reward rate with the option to abandon, so an uncertain game is worth a
bounded try and a game that clears quickly keeps its slot through its next, heavier levels. Levels
that cannot finish before the end of the run earn nothing. In the simulator the index is insensitive
to the absolute hazard scale (x0.5 and x2 within noise).

Options (27 September evening, ``fixed-2x-gittins-c110``; defaults keep the recorded model):
``shape: "within"`` uses the within-level hazard shape (inside one game-level the clear rate rises with
tokens; the pooled curve's fall after 40K is between-level heterogeneity, which the game and level factors
already carry) with its jointly fitted ``k``, ``ke`` and ``beta``. ``prior_scale: "run"`` sets the game
factor's prior mean by in-run empirical Bayes over every game's record in this run.
"""

import math

import numpy as np

EDGES = (0.0, 5e3, 10e3, 20e3, 40e3, 80e3, 160e3, math.inf)
# Clears per token by band, pooled over the recorded Flash xhigh attempts on all public games
# (2,829 attempts, 79 runs; one pseudo-clear per band at 0.3 per 10K).
H0_FIRST = (0.082e-4, 0.339e-4, 0.320e-4, 0.350e-4, 0.118e-4, 0.300e-4, 0.300e-4)
H0_DEEP = (0.079e-4, 0.225e-4, 0.323e-4, 0.287e-4, 0.138e-4, 0.086e-4, 0.130e-4)
# Within-level shape (shape "within", 27 September; scripts/arena_analysis/within_level_hazard.py,
# docs/validation-20260927/scheduler-c110.md). The pooled curve falls after 40K because easy levels clear
# early and leave the risk set; inside one (game, level) the clear rate rises with tokens. These bands, with
# k, ke and beta below, are this model fitted jointly by marginal likelihood (m_g on a grid, e_L in closed
# form) to the recorded Flash xhigh attempts on all 25 public games (2,653 attempts, 112 game-levels, 962
# clears): generic constants pooled over games, no per-game values. Relative to the 10-20K band (90% game
# bootstrap): first level 0.04 (0.01-0.07), 0.40 (0.31-0.47), 1, 2.1 (1.3-3.3), 1.35 (0.5-4.6);
# deeper levels 0.09 (0.05-0.15), 0.49 (0.38-0.59), 1, 1.6 (1.2-2.5), 1.4 (1.1-2.1), 1.6, 2.1.
H0_FIRST_WITHIN = (0.079e-4, 0.765e-4, 1.905e-4, 4.012e-4, 2.576e-4, 2.581e-4, 2.559e-4)
H0_DEEP_WITHIN = (0.054e-4, 0.276e-4, 0.568e-4, 0.912e-4, 0.789e-4, 0.927e-4, 1.190e-4)
WITHIN_FIT = {"k": 0.53, "ke": 0.67, "beta": 1.14}  # 90% bootstrap: k 0.38-1.2, ke 0.53-1.0, beta 0.47-1.44
SHAPES = {"pooled": (H0_FIRST, H0_DEEP), "within": (H0_FIRST_WITHIN, H0_DEEP_WITHIN)}
# fixed-2x-gittins-broad as recorded (kaggle-arena-duck-60m-r23-gittins-broad-v1/v2): frozen, pooled shape.
BROAD = {"model": "gittins", "k": 0.25, "ke": 1.5, "beta": 0.5,
         "caps": [10000, 20000, 40000, 80000, 160000], "samples": 256, "hysteresis": 0.15,
         "quantum_tokens": 6000, "stream_tokens_per_second": 115.0, "default_levels": 7, "seed": 0,
         "h0_scale": 1.0}
# A settings dict without "shape" (every recorded study) keeps the pooled shape.
DEFAULTS = {**BROAD, "shape": "pooled", "prior_scale": "fixed"}
# fixed-2x-gittins-c110 (110 clones of 20 games, one hour, two slots): the within-level shape with its jointly
# fitted k, ke and beta, the in-run prior scale, and a 4x hysteresis tuned in the simulator
# (docs/validation-20260927/scheduler-c110.md).
C110 = {**DEFAULTS, "shape": "within", **WITHIN_FIT, "prior_scale": "run", "hysteresis": 4.0,
        "quantum_tokens": 6000}
C110_OPENING_MAX_WALL_SECONDS = 1200  # the two-clear opening, capped (absolute, for 110 instances)


def settings(raw=None):
    merged = {**DEFAULTS, **(raw or {})}
    if set(merged) != set(DEFAULTS):
        raise ValueError(f"Unknown Gittins settings: {sorted(set(merged) - set(DEFAULTS))}")
    if merged["model"] != "gittins":
        raise ValueError("Gittins settings need model 'gittins'")
    if merged["shape"] not in SHAPES:
        raise ValueError(f"Gittins shape must be one of {sorted(SHAPES)}")
    if merged["prior_scale"] not in ("fixed", "run"):
        raise ValueError("Gittins prior_scale must be 'fixed' or 'run'")
    caps = merged["caps"]
    if not caps or any(type(c) not in (int, float) or not math.isfinite(c) or c <= 0 for c in caps):
        raise ValueError("caps must be positive token counts")
    merged["caps"] = sorted(float(c) for c in caps)
    for key in ("k", "ke", "beta", "samples", "quantum_tokens", "stream_tokens_per_second", "default_levels",
                "h0_scale"):
        value = merged[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Gittins setting {key} must be a positive number")
    if type(merged["hysteresis"]) not in (int, float) or merged["hysteresis"] < 0:
        raise ValueError("hysteresis must be non-negative")
    if type(merged["seed"]) is not int or type(merged["samples"]) is not int:
        raise ValueError("seed and samples must be integers")
    return merged


def _bands(tokens):
    lo, hi = np.array(EDGES[:-1]), np.array(EDGES[1:])
    return np.clip(np.minimum(tokens, hi) - lo, 0, None)


def cum_hazard(h, tokens):
    return float(_bands(tokens) @ h)


def inv_cum_hazard(h, target):
    """Smallest t with H(t) >= target (array); every band's hazard is positive."""
    edges = np.array(EDGES[:-1])
    knots = np.concatenate([[0.0], np.cumsum(np.diff(edges) * h[:-1])])
    band = np.searchsorted(knots, target, side="right") - 1
    return edges[band] + (target - knots[band]) / h[band]


class GittinsModel:
    """Rebuilt at every scheduler tick from the game records (the ``hazard_index.RunModel`` interface)."""

    def __init__(self, games, cfg):
        self.cfg = cfg
        scale = cfg["h0_scale"]
        first, deep = SHAPES[cfg.get("shape", "pooled")]
        self.h0 = {"first": np.array(first) * scale, "deep": np.array(deep) * scale}
        self.grid = np.exp(np.linspace(math.log(1e-3), math.log(50.0), 80))
        k = cfg["k"]
        self.log_prior = k * np.log(self.grid) - k * self.grid  # Gamma(k, k) on a log grid
        self.caps = np.array(cfg["caps"])
        self.scale = 1.0
        if cfg.get("prior_scale", "fixed") == "run":
            self.scale = self.run_scale(games)
            self.log_prior = k * np.log(self.grid) - k / self.scale * self.grid  # Gamma(k, k / scale)

    RUN_SCALES = np.exp(np.linspace(math.log(0.05), math.log(4.0), 45))

    def run_scale(self, games):
        """In-run empirical Bayes (prior_scale "run", fixed-2x-gittins-c110): the mean ``s`` of the game factor's
        Gamma(k, k / s) prior that maximises the marginal likelihood of every game's record in this run, with a
        N(0, 1) prior on log s. A roster harder than the public games the constants were fitted on (few early
        clears, long fruitless attempts) lowers the value of a fresh game; an easy one raises it. Uses only this
        run's clears and tokens."""
        k = self.cfg["k"]
        lls = [self.log_likelihood(g) for g in games
               if (g.level_tokens and any(t > 0 for t in g.level_tokens)) or g.tokens_on_level() > 0]
        if not lls:
            return 1.0
        ll = np.array(lls)  # games x grid
        s = self.RUN_SCALES[:, None]
        # log Gamma(k, k/s) density on the log grid (times m): k log(k/s) - lgamma(k) + k log m - (k/s) m
        lp = k * np.log(k / s) - math.lgamma(k) + k * np.log(self.grid)[None, :] - (k / s) * self.grid[None, :]
        step = math.log(self.grid[1] / self.grid[0])
        z = ll[None, :, :] + lp[:, None, :]  # scales x games x grid
        zmax = z.max(axis=2, keepdims=True)
        marginal = (zmax[..., 0] + np.log(np.exp(z - zmax).sum(axis=2) * step)).sum(axis=1)
        marginal -= 0.5 * np.log(self.RUN_SCALES) ** 2
        return float(self.RUN_SCALES[int(np.argmax(marginal))])

    def _h(self, level):
        return self.h0["first" if level == 1 else "deep"]

    def _power(self, level):
        return 1.0 if level == 1 else self.cfg["beta"]

    def posterior(self, game):
        """Normalised posterior weights of ``m_g`` on the grid."""
        lp = self.log_likelihood(game, self.log_prior)
        p = np.exp(lp - lp.max())
        return p / p.sum()

    def log_likelihood(self, game, base=None):
        """``base`` plus the log likelihood of the game's record in this run at every grid value of ``m_g``
        (``e`` integrated out); the terms are added in the recorded order so the posterior is bit-identical."""
        lp = np.zeros_like(self.grid) if base is None else base.copy()
        ke = self.cfg["ke"]
        for i, tokens in enumerate(game.level_tokens):
            if tokens <= 0:  # a multi-level jump: the extra levels took no tokens of their own
                continue
            mm = self.grid ** self._power(i + 1)
            lp += np.log(mm) + (ke + 1) * np.log(ke / (ke + mm * cum_hazard(self._h(i + 1), tokens)))
        level = game.levels + 1
        total = int(game.total_levels or self.cfg["default_levels"])
        if level <= total:
            mm = self.grid ** self._power(level)
            lp += ke * np.log(ke / (ke + mm * cum_hazard(self._h(level), game.tokens_on_level())))
        return lp

    def quantum(self):
        return self.cfg["quantum_tokens"]

    def index(self, game, remaining_seconds):
        """Best expected score share per 10K tokens over the plans (the scheduler compares these)."""
        cfg = self.cfg
        slot_tokens = max(0.0, remaining_seconds) * cfg["stream_tokens_per_second"]
        total = int(game.total_levels or cfg["default_levels"])
        if slot_tokens <= 0 or game.levels >= total:
            return 0.0, {}
        rng = np.random.default_rng(cfg["seed"])  # common random numbers: stable comparisons across games
        samples = cfg["samples"]
        p = self.posterior(game)
        m = rng.choice(self.grid, size=samples, p=p)
        weights = total * (total + 1) / 2
        caps = self.caps
        used = np.zeros((samples, len(caps)))
        gained = np.zeros((samples, len(caps)))
        alive = np.ones((samples, len(caps)), dtype=bool)
        best, plan = 0.0, None
        for level in range(game.levels + 1, total + 1):
            h = self._h(level)
            mm = m ** self._power(level)
            spent = game.tokens_on_level() if level == game.levels + 1 else 0.0
            h_spent = cum_hazard(h, spent)
            # The level's own factor given that it survived ``spent`` tokens.
            e = rng.gamma(cfg["ke"], 1.0 / (cfg["ke"] + mm * h_spent))
            tau = inv_cum_hazard(h, h_spent + rng.exponential(size=samples) / np.maximum(mm * e, 1e-12))
            room = np.maximum(caps[None, :] - spent, 0.0)
            need = (tau - spent)[:, None]
            clears = need <= room
            fits = used + np.where(clears, need, 0.0) <= slot_tokens
            gained += np.where(alive & clears & fits, level / weights, 0.0)
            used += np.where(alive, np.minimum(np.where(clears, need, room), np.maximum(slot_tokens - used, 0.0)), 0.0)
            alive &= clears & fits
            rate = gained.mean(axis=0) / np.maximum(used.mean(axis=0), 1.0)
            valid = caps > spent if level == game.levels + 1 else np.ones(len(caps), dtype=bool)
            if valid.any():
                j = int(np.argmax(np.where(valid, rate, -1.0)))
                if rate[j] > best:
                    best, plan = float(rate[j]), {"levels": level - game.levels, "cap_tokens": int(caps[j])}
            if not alive.any():
                break
        mean_m = float(p @ self.grid)
        detail = {"posterior_mean": round(mean_m, 4), "spent_tokens": int(game.tokens_on_level()), "plan": plan}
        return best * 1e4, detail

    def summary(self):
        return {"model": "gittins", "shape": self.cfg.get("shape", "pooled"), "k": self.cfg["k"], "beta": self.cfg["beta"],
                "ke": self.cfg["ke"]}
