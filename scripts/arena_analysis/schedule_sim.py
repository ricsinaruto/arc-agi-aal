"""Trace-driven scheduler simulator for the one-hour 23-game arena (27 September).

A one-hour run costs a Kaggle slot and has a standard deviation of ~1.4 points, so scheduling policies
are screened here first. The world is fitted to the recorded Flash xhigh level attempts
(``level_attempts.py`` output): a proportional-hazards model where a level clears when the tokens spent on
it cross a threshold drawn from ``m x h0_class(t)``. ``h0`` is the pooled clear hazard per token band
(first level / deeper levels) and ``m`` is drawn per (game, level) from its Gamma posterior given that
level's recorded exposure and clears (unreached levels use the game's deeper levels, else the prior).
Each simulated run draws fresh multipliers and thresholds, so run-to-run spread is included.

``World.fit(shape="within")`` (27 September evening, docs/validation-20260927/scheduler-c110.md) replaces the
pooled shape, which falls after 40K only because easy levels leave the risk set, with the within-level shape
fitted by ``within_level_hazard.py`` (hierarchical frailty model: per-game factor x per-level factor); it
matches the recorded one-hour runs much better (fixed-2x 6.55 simulated vs 6.65 recorded). ``--sweep`` runs a
JSON grid of policies in parallel on the 23-game, 110-game (23 repeated) or ``c110`` roster (110 cyclic clones
of 20 games), with the opening cap, a selective opening, other slot counts and throughputs, and the fixed-2x
baseline (``FixedTwoX``).

Throughput is fixed at the recorded rates: an opening in which all games advance together (~340 tok/s
in aggregate) until two games clear a level, then ``slots`` streams sharing ~230 tok/s. Readmitting a
parked game costs a few seconds of prefill. A cleared level scores ``level_score`` (the recorded mean is
~100) with weight equal to its index, as in the competition scorecard.

Policies see only in-run signals: levels, total levels, tokens per cleared level, tokens on the current
level. ``oracle`` sees the thresholds and is an upper bound for the budget.
"""

import argparse
import json
import math
from dataclasses import dataclass, field

import numpy as np

EDGES = np.array([0, 5e3, 10e3, 20e3, 40e3, 80e3, 160e3, np.inf])
GAMES23 = ["ar25", "bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20", "m0r0", "r11l", "re86",
           "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]
# The c110 research roster (27 September): 110 cyclic clones of the 25-game control roster minus ft09, lp85,
# ar25, vc33 and tu93, in the control's roster order (configs/arena_30m.json): the first 10 games get 6 copies.
C20 = ["tn36", "lf52", "cn04", "bp35", "wa30", "r11l", "sp80", "m0r0", "ka59", "sc25", "sk48", "dc22", "cd82", "g50t",
       "ls20", "re86", "s5i5", "sb26", "su15", "tr87"]
C110 = [C20[i % len(C20)] for i in range(110)]
AGG_RATE = {1: 127.0, 2: 230.0, 3: 245.0, 4: 255.0}
OPENING_RATE = 340.0


# The world's hazard inside the first (0-5K) band, in 1K sub-bands relative to the band's mean (28 September, c110
# calibration). Recorded clears never come in the first 1.7K tokens of a first level (sb26, the fastest game, clears
# at 1.7-2.3K), but a constant hazard over 0-5K puts a third of a fast game's clears under 1K, so the simulated
# opening's first clears came ~400-550 s early. Pooled clears / exposure per 1K over all recorded attempts (the
# 98 earlier runs and the c110 runs), normalised to keep H(5K) unchanged; 0 is floored at 0.02.
FINE_EDGES = np.array([0, 1e3, 2e3, 3e3, 4e3, 5e3, 10e3, 20e3, 40e3, 80e3, 160e3, np.inf])
FINE_START = {"first": np.array([0.02, 0.232, 1.211, 2.072, 1.853]),
              "deep": np.array([0.489, 0.588, 1.724, 1.059, 1.189])}
for _c, _r in FINE_START.items():
    FINE_START[_c] = _r * len(_r) / _r.sum()


def fine_hazard(h, cls):
    """``h`` on ``EDGES`` expanded to ``FINE_EDGES`` with the 0-5K band split by ``FINE_START``."""
    return np.concatenate([h[0] * FINE_START[cls], h[1:]])


def inv_cum_hazard_fine(h, target):
    knots = np.concatenate([[0.0], np.cumsum(np.diff(FINE_EDGES[:-1]) * h[:-1])])
    band = np.searchsorted(knots, target, side="right") - 1
    return float(FINE_EDGES[:-1][band] + (target - knots[band]) / h[band])


def band_exposure(tokens):
    """Tokens spent inside each hazard band by an attempt of ``tokens``."""
    lo, hi = EDGES[:-1], EDGES[1:]
    return np.clip(np.minimum(tokens, hi) - lo, 0, None)


def cum_hazard(h, t):
    return float(band_exposure(t) @ h)


def inv_cum_hazard(h, target):
    """Smallest t with H(t) >= target (``target`` scalar or array; every band's hazard is positive)."""
    knots = np.concatenate([[0.0], np.cumsum(np.diff(EDGES[:-1]) * h[:-1])])
    target = np.asarray(target, dtype=float)
    band = np.searchsorted(knots, target, side="right") - 1
    t = EDGES[:-1][band] + (target - knots[band]) / h[band]
    return float(t) if t.ndim == 0 else t


@dataclass
class World:
    h0: dict
    post: dict  # (game, level) -> (shape, rate)
    game_post: dict  # game -> (shape, rate) over its deeper levels
    total_levels: dict
    prior: tuple
    shape: str = "pooled"
    hier: dict = None  # shape "within": the within_level_hazard fit (k, ke, beta) and per-game posteriors

    @classmethod
    def fit(cls, rows, prior=(0.5, 0.5), shape="pooled"):
        """``shape="pooled"``: the 27 September world (pooled band hazard x a per-(game, level) Gamma posterior).

        ``shape="within"``: the within-level shape (``within_level_hazard.py``, hierarchical fit). A level's hazard
        is ``h0_class(t) x m_g^p x e_L``, ``m_g`` drawn from the game's posterior over all its recorded levels and
        ``e_L`` from the level's Gamma posterior given ``m_g``: the hazard inside a level rises with tokens as the
        recorded attempts do, and the between-level heterogeneity lives only in the multipliers (the pooled world
        draws its multipliers around a falling pooled shape, which counts that heterogeneity twice)."""
        if shape == "within":
            return cls._fit_within(rows, prior)
        rows = [r for r in rows if not (r["cleared"] and r["tokens"] < 500)]
        h0 = {}
        for c, sel in (("first", lambda r: r["level"] == 1), ("deep", lambda r: r["level"] > 1)):
            rs = [r for r in rows if sel(r)]
            exp = sum(band_exposure(r["tokens"]) for r in rs)
            ev = np.zeros(len(EDGES) - 1)
            for r in rs:
                if r["cleared"]:
                    ev[np.searchsorted(EDGES, r["tokens"], side="right") - 1] += 1
            # One pseudo-event per band at the deep-class mean, so sparse tails stay finite.
            h0[c] = (ev + 0.5) / (exp + 0.5 / 3e-5)
        post, deep = {}, {}
        total = {}
        for r in rows:
            total[r["game"]] = max(total.get(r["game"], 0), r["total_levels"] or 0)
            c = "first" if r["level"] == 1 else "deep"
            key = (r["game"], r["level"])
            n, e = post.get(key, (0, 0.0))
            post[key] = (n + r["cleared"], e + cum_hazard(h0[c], r["tokens"]))
        for (g, lv), (n, e) in post.items():
            if lv > 1:
                dn, de = deep.get(g, (0, 0.0))
                deep[g] = (dn + n, de + e)
        a, b = prior
        post = {k: (a + n, b + e) for k, (n, e) in post.items()}
        deep = {k: (a + n, b + e) for k, (n, e) in deep.items()}
        return cls(h0, post, deep, total, prior)

    @classmethod
    def _fit_within(cls, rows, prior):
        import within_level_hazard as wlh

        f = wlh.fit(rows, "hier")
        h0 = {"first": np.array(f["first"]), "deep": np.array(f["deep"])}
        total = {}
        for r in rows:
            total[r["game"]] = max(total.get(r["game"], 0), r["total_levels"] or 0)
        hier = {"k": f["k"], "ke": f["ke"], "beta": f["beta"], "grid": wlh.GRID,
                "games": wlh.game_posteriors(f["theta"], rows)}
        return cls(h0, {}, {}, total, prior, "within", hier)

    def _multipliers(self, base, rng):
        """One draw of every level's hazard multiplier for ``base``."""
        n = self.total_levels[base]
        if self.shape == "within":
            hier = self.hier
            post = hier["games"][base]
            m = rng.choice(hier["grid"], p=post["weights"])
            self._last_m = float(m)
            out = []
            for lv in range(1, n + 1):
                mm = m ** (1.0 if lv == 1 else hier["beta"])
                d, s = post["levels"].get(lv, (0.0, 0.0))
                out.append(mm * rng.gamma(hier["ke"] + d, 1.0 / (hier["ke"] + mm * s)))
            return out
        out = []
        for lv in range(1, n + 1):
            a, b = self.post.get((base, lv)) or self.game_post.get(base) or self.prior
            out.append(rng.gamma(a, 1.0 / b))
        return out

    def _population(self, base, rng):
        """A new game from the population (``population`` sampling, 28 September): ``m_g ~ Gamma(k, k)`` and
        ``e_L ~ Gamma(ke, ke)`` from the fitted hierarchy, with ``base``'s level count. No game's own record."""
        hier = self.hier
        m = rng.gamma(hier["k"], 1.0 / hier["k"])
        self._last_m = float(m)
        return [(m ** (1.0 if lv == 1 else hier["beta"])) * rng.gamma(hier["ke"], 1.0 / hier["ke"])
                for lv in range(1, self.total_levels[base] + 1)]

    def _tail(self, h, edges, tail):
        """``tail`` {"from": tokens, "scale": x}: the hazard on bands starting at or above ``from`` tokens x ``scale``
        (28 September depthwise calibration: long uninterrupted attempts clear less than the within shape says)."""
        if not tail:
            return h
        return np.where(edges[:-1] >= tail["from"], h * tail["scale"], h)

    def sample(self, games, rng, shared=False, efficiency=1.0, population=False, first_efficiency=1.0, tail=None):
        """Per game, the clear threshold (tokens) of every level; ``self.last_multipliers`` keeps the
        hazard multipliers drawn (for the ``known-rate`` bound).

        Repeated games (the 110-game rosters) are copies ``game#k``. With ``shared`` the copies of a game
        share one draw of the level multipliers (one true difficulty per game in a run) and differ only in
        their thresholds; otherwise every copy draws its own multipliers (27 September default).
        ``efficiency`` multiplies every level's clear hazard (a better model or harness, 28 September);
        ``population`` draws every game's multipliers from the fitted population instead of its record;
        ``first_efficiency`` multiplies the first level's hazard only (the depthwise level-0 turns' 32K window)."""
        out, mult, factor = {}, {}, {}
        seen, drawn, drawn_m = {}, {}, {}
        for base in games:
            seen[base] = seen.get(base, 0) + 1
            g = base if seen[base] == 1 else f"{base}#{seen[base]}"
            if not (shared and base in drawn):
                self._last_m = None
                src = base.split("~")[0]  # "sb26~2": a second, independent game drawn from sb26's record
                drawn[base] = self._population(src, rng) if population else self._multipliers(src, rng)
                drawn_m[base] = self._last_m
            ms = [m * efficiency * (first_efficiency if lv == 0 else 1.0) for lv, m in enumerate(drawn[base])]
            if getattr(self, "fine_start", False):
                th = [inv_cum_hazard_fine(self._tail(fine_hazard(self.h0["first" if lv == 1 else "deep"],
                                                                 "first" if lv == 1 else "deep"), FINE_EDGES, tail)
                                          * m, rng.exponential())
                      for lv, m in enumerate(ms, 1)]
            else:
                th = [inv_cum_hazard(self._tail(self.h0["first" if lv == 1 else "deep"], EDGES, tail) * m,
                                     rng.exponential())
                      for lv, m in enumerate(ms, 1)]
            out[g], mult[g], factor[g] = th, ms, (drawn_m[base], efficiency)
        self.last_multipliers = mult
        self.last_game_factor = factor  # (m_g, efficiency) per game: the ``known-game`` bound
        return out


@dataclass
class Game:
    gid: str
    index: int
    total_levels: int
    thresholds: list
    levels: int = 0
    tok: float = 0.0  # tokens on the current level
    level_tokens: list = field(default_factory=list)
    spent: float = 0.0
    admit_spent: float = 0.0
    active: bool = False
    finished: bool = False

    # interface of arena.scheduler.GameRecord used by hazard_index.RunModel
    @property
    def game_id(self):
        return self.gid

    def tokens_on_level(self):
        return self.tok

    def advance(self, dt):
        """Spend ``dt`` tokens; returns the levels cleared."""
        cleared = 0
        while dt > 0 and not self.finished:
            need = self.thresholds[self.levels] - self.tok
            if need > dt:
                self.tok += dt
                self.spent += dt
                return cleared
            self.spent += need
            dt -= need
            self.level_tokens.append(self.tok + need)
            self.levels += 1
            self.tok = 0.0
            cleared += 1
            if self.levels >= self.total_levels:
                self.finished = True
        return cleared

    def score(self, level_score):
        n = self.total_levels
        return level_score * sum(range(1, self.levels + 1)) / (n * (n + 1) / 2)


def simulate(policy, thresholds, *, wall=3600.0, slots=2, step=1000.0, readmit_seconds=3.0,
             level_score=100.0, switch_games=2, opening_tokens=0.0, opening_rate=OPENING_RATE,
             opening_max_wall=None, agg_rate=None, opening_step=None, opening_games=None, opening_share=None,
             opening_share_sd=0.0, share_rng=None, readmit_cost=None, switch_pause=0.0):
    """``opening_max_wall``: the opening also ends at this wall second (the scheduler's
    ``opening_max_wall_seconds``); ``agg_rate``: phase-2 tokens per second over all slots (default
    ``AGG_RATE[slots]``); ``opening_step``: tokens per game per opening round (default ``step / 4``);
    ``opening_games``: only the first N games (roster order) play in the opening, the rest wait for phase 2.
    ``opening_share`` (28 September, c110 calibration): {source game: relative token rate in the opening}. The
    real opening shares its requests about evenly, so a game with short replies (sb26: 0.6 of the mean) gets
    fewer tokens per wall second; ``opening_share_sd`` adds a per-game lognormal factor on top.
    ``readmit_cost`` (29 September, hero-v2 study): (fresh, played) whole-server seconds per admission, the prefill
    of a never-played game's first prompt and the re-prefill of a parked game's context; replaces
    ``readmit_seconds / slots``.
    ``switch_pause`` (29 September, sim-mtp3-kv4): a server restart at the opening switch (e.g. MTP-0 for the opening,
    MTP-3 for phase 2): that many wall seconds with no tokens after the switch; the focus games' first phase-2
    admissions then pay their re-prefill as usual (``readmit_cost``), which is a cold prefill after a restart."""
    games = [Game(g, i, len(th), th) for i, (g, th) in enumerate(thresholds.items())]
    opened = games if opening_games is None else games[:opening_games]
    t = 0.0
    ostep = step / 4 if opening_step is None else opening_step
    share = {}
    for g in opened:
        s = (opening_share or {}).get(g.gid.split("#")[0].split("~")[0], 1.0)
        if opening_share_sd and share_rng is not None:
            s *= math.exp(share_rng.normal(0.0, opening_share_sd) - opening_share_sd ** 2 / 2)
        share[g.gid] = s
    # Opening: every opened game advances together until ``switch_games`` games have cleared a level and
    # every opened game has spent ``opening_tokens``, or until ``opening_max_wall``.
    while t < wall and (opening_max_wall is None or t < opening_max_wall) and (
            sum(g.levels >= 1 for g in games) < switch_games
            or min((g.spent for g in opened if not g.finished), default=math.inf) < opening_tokens):
        live = [g for g in opened if not g.finished]
        if not live:
            break
        used = 0.0
        for g in live:
            g.advance(ostep * share[g.gid])
            used += ostep * share[g.gid]
        t += used / opening_rate
    switch = t
    cleared_at_switch = sum(g.levels >= 1 for g in games)
    played_at_switch = sum(g.spent > 0 for g in games)
    t += switch_pause
    policy.start(games, slots)
    rate = (AGG_RATE[slots] if agg_rate is None else agg_rate) / slots
    admissions = 0
    # Phase-2 cache accounting (29 September tick study): a focus game keeps its cache while it stays in focus (no
    # charge); every admission of a game that has played before re-prefills its context (``readmit_cost[1]``).
    readmits, fresh, reprefill_s, phase2_s = 0, 0, 0.0, wall - t
    while t < wall:
        remaining = wall - t
        chosen = policy.choose(games, remaining, rate)
        for g in games:
            if g in chosen and not g.active:
                g.active = True
                played = g.spent > 0
                g.admit_spent = g.spent
                admissions += 1
                readmits += played
                fresh += not played
                if readmit_cost is None:
                    t += readmit_seconds / slots
                else:
                    cost = readmit_cost[1] if played else readmit_cost[0]
                    t += cost
                    reprefill_s += cost if played else 0.0
            elif g not in chosen:
                g.active = False
        act = [g for g in games if g.active and not g.finished]
        if not act:
            break
        for g in act:
            g.advance(step)
        t += step / rate
    total = sum(g.score(level_score) for g in games) / len(games)
    stats = getattr(policy, "stats", lambda: None)()
    if readmit_cost is not None:
        stats = {**(stats or {}), "readmits": float(readmits), "fresh_admits": float(fresh),
                 "reprefill_seconds": reprefill_s, "reprefill_share": reprefill_s / max(phase2_s, 1.0),
                 "switch_seconds": float(switch), "cleared_at_switch": float(cleared_at_switch),
                 "played_at_switch": float(played_at_switch)}
    return {"score": total, "levels": sum(g.levels for g in games), "admissions": admissions, "switch": switch,
            "games": {g.gid: g.levels for g in games}, "played": sum(g.spent > 0 for g in games),
            "cleared1": sum(g.levels >= 1 for g in games), "stats": stats}


# ------------------------------------------------------------------------------------------ policies


class Policy:
    name = "policy"

    def start(self, games, slots):
        self.slots = slots

    def active(self, games):
        return [g for g in games if g.active and not g.finished]

    def waiting(self, games):
        return [g for g in games if not g.active and not g.finished]


class Oracle(Policy):
    """Knows every threshold: solves the knapsack over per-game level prefixes at the switch, then plays
    the chosen games in order of value per token (an upper bound on what any policy can reach)."""
    name = "oracle"

    def start(self, games, slots):
        super().start(games, slots)
        self.plan = None

    def choose(self, games, remaining, rate):
        if self.plan is None:
            budget = int(remaining * rate * self.slots / 1000)
            best = np.zeros(budget + 1)
            choice = []
            for g in games:
                opts = [(0, 0.0)]
                cost, value = g.thresholds[g.levels] - g.tok, 0.0
                n = g.total_levels
                for lv in range(g.levels + 1, n + 1):
                    if lv > g.levels + 1:
                        cost += g.thresholds[lv - 1]
                    if not math.isfinite(cost):
                        break
                    value += lv / (n * (n + 1) / 2)
                    opts.append((int(math.ceil(cost / 1000)), value))
                new = best.copy()
                arg = np.zeros(budget + 1, dtype=int)
                for j, (c, v) in enumerate(opts[1:], 1):
                    if c > budget:
                        break
                    cand = np.full(budget + 1, -1.0)
                    cand[c:] = best[:budget + 1 - c] + v
                    better = cand > new
                    new[better] = cand[better]
                    arg[better] = j
                choice.append((g, opts, arg))
                best = new
            b, plan = budget, {}
            for g, opts, arg in reversed(choice):
                j = arg[b]
                plan[g.gid] = g.levels + j
                b -= opts[j][0]
            self.plan = plan
        live = [g for g in games if not g.finished and g.levels < self.plan[g.gid]]
        keep = [g for g in live if g.active][: self.slots]
        rest = sorted((g for g in live if g not in keep), key=lambda g: g.thresholds[g.levels] - g.tok)
        return keep + rest[: self.slots - len(keep)]


class Index(Policy):
    """fixed-2x-index: hazard_index.RunModel with the scheduler's fill/one-swap/quantum/hysteresis tick."""
    name = "index"

    def __init__(self, **cfg):
        # c110 only (28 September): at most ``per_source_cap`` focus slots on clones of one source game.
        self.per_source_cap = cfg.pop("per_source_cap", None)
        custom = cfg.pop("custom_shape", None)  # {"first": [7], "deep": [7]}: a refitted within shape (held-out check)
        if cfg.get("model") == "gittins":  # the production arena.gittins model under the same tick
            from arena import gittins
            if custom is not None:
                name = "custom-" + "-".join(f"{x:.6g}" for x in custom["first"] + custom["deep"])
                gittins.SHAPES[name] = (tuple(custom["first"]), tuple(custom["deep"]))
                cfg = {**cfg, "shape": name}
            self.model_cls, self.cfg = gittins.GittinsModel, gittins.settings(cfg)
        else:
            from arena import hazard_index
            self.model_cls, self.cfg = hazard_index.RunModel, hazard_index.settings(cfg or None)

    def choose(self, games, remaining, rate):
        model = self.model_cls(games, self.cfg)
        live = [g for g in games if not g.finished]
        if self.cfg.get("model") == "gittins":
            # The Gittins index depends only on the game's own record and the time left: cache it per
            # (levels, tokens on the level to 1K, time left to 5 min) so 110-game, 9-hour runs stay fast.
            cache = self.__dict__.setdefault("_cache", {})
            sc = {}
            for g in live:
                key = (g.gid, g.levels, int(g.tok // 1000), int(remaining // 300), model.scale)
                if key not in cache:
                    cache[key] = model.index(g, remaining)[0]
                sc[g.gid] = cache[key]
        else:
            sc = {g.gid: model.index(g, remaining)[0] for g in live}
        act = self.active(games)
        wait = sorted(self.waiting(games), key=lambda g: (-sc[g.gid], g.index))
        chosen = list(act)
        cap = self.per_source_cap

        def src(g):
            return g.gid.split("#")[0]

        def room(g, held):
            return cap is None or sum(src(x) == src(g) for x in held) < cap

        while len(chosen) < self.slots and wait:
            g = wait.pop(0)
            if room(g, chosen):
                chosen.append(g)
        if cap is not None:
            wait = sorted(self.waiting(games), key=lambda g: (-sc[g.gid], g.index))
            wait = [g for g in wait if g not in chosen]
        if wait:
            elig = [g for g in chosen if g.spent - g.admit_spent >= model.quantum()]
            if elig:
                worst = min(elig, key=lambda g: (sc[g.gid], -g.index))
                rest = [g for g in chosen if g is not worst]
                cand = next((g for g in wait if room(g, rest)), None)
                if cand is not None and sc[cand.gid] > (1 + self.cfg["hysteresis"]) * sc[worst.gid]:
                    chosen[chosen.index(worst)] = cand
        return chosen


class Commit(Policy):
    """fixed-2x-commit without the lane: stall cap clamp(k x median cleared, lo, hi); fill by depth."""
    name = "commit"

    def __init__(self, multiple=2.5, lo=25e3, hi=60e3):
        self.multiple, self.lo, self.hi = multiple, lo, hi

    def cap(self, g):
        c = [x for x in g.level_tokens if x > 0]
        cap = self.multiple * float(np.median(c)) if c else 0
        return min(max(cap, self.lo), self.hi)

    def choose(self, games, remaining, rate):
        chosen = [g for g in self.active(games) if g.tok <= self.cap(g)]
        wait = [g for g in self.waiting(games) if g.tok <= self.cap(g)]
        wait.sort(key=lambda g: (-g.levels, g.tok, g.index))
        return chosen + wait[: self.slots - len(chosen)]


class FixedTwoX(Policy):
    """The fixed-2x baseline (reclaim-fixed-2x, ``arena.scheduler`` ``budget_policy="fixed-2x"``) on the token clock.

    The opening ends at the second level-1 clear (no minimum tokens) and keeps the two fastest first clears.
    A focus game playing level L gets 2 x L x T1 at every clear and every readmission (the remainder at the
    switch); an exhausted budget parks it while a backlog exists. A game without a clear is readmitted with
    what it had spent when first parked (floor 30 s). Backlog: fewest steady-state admissions, then most
    levels, then roster order. Budgets are busy seconds in the arena; here they are tokens at the phase-2
    ``tokens_per_busy_second`` (108 recorded), and an opening busy second buys ``1 / busy_factor`` of that
    (66 vs 108 tokens recorded: ``busy_factor`` 1.64), so T1 and level-0 grants earned in the opening weigh
    ``busy_factor`` times their tokens."""
    name = "fixed-2x"

    def __init__(self, multiple=2.0, busy_factor=108 / 66, floor_seconds=30.0, tokens_per_busy_second=108.0):
        self.multiple, self.busy_factor = multiple, busy_factor
        self.floor = floor_seconds * tokens_per_busy_second

    def _equiv(self, g, tokens):
        """Opening tokens at ``busy_factor``, phase-2 tokens at 1."""
        o = self.opening[g.gid]
        return min(tokens, o) * self.busy_factor + max(0.0, tokens - o)

    def _t1(self, g):
        if g.gid not in self.t1 and g.levels >= 1:
            self.t1[g.gid] = self._equiv(g, g.level_tokens[0])
        return self.t1.get(g.gid)

    def _level_budget(self, g):
        return self.multiple * (g.levels + 1) * self._t1(g)

    def _grant(self, g, budget):
        self.budget[g.gid] = (g.spent, budget, g.levels)

    def start(self, games, slots):
        super().start(games, slots)
        self.opening = {g.gid: g.spent for g in games}
        self.t1, self.budget, self.readmissions, self.level0 = {}, {}, {}, {}
        cleared = sorted((g for g in games if g.levels >= 1 and not g.finished),
                         key=lambda g: (self._t1(g), g.index))
        self.keep = cleared[:slots]
        for g in games:
            self.readmissions[g.gid] = 0
            if g.levels == 0:
                self.level0[g.gid] = max(self._equiv(g, g.spent), self.floor)
        for g in self.keep:
            self.readmissions[g.gid] = 1
            self._grant(g, max(self._level_budget(g) - self._equiv(g, g.tok), self.floor))
        self.kept_at_switch = len(self.keep)

    def stats(self):
        """Focus slots at the switch held by games that cleared in the opening (the rest fill from the backlog)."""
        return {"kept_at_switch": float(self.kept_at_switch), "backlog_at_switch": float(self.slots - self.kept_at_switch)}

    def choose(self, games, remaining, rate):
        if self.keep is not None:  # the opening switch: the kept games hold the slots
            act, self.keep = [g for g in self.keep if not g.finished], None
        else:
            act = self.active(games)
        wait = sorted((g for g in games if not g.finished and g not in act),
                      key=lambda g: (self.readmissions[g.gid], -g.levels, g.index))
        chosen = []
        for g in act:
            start, budget, level = self.budget[g.gid]
            if g.levels != level:  # a clear: the full budget of the new level
                self._grant(g, self._level_budget(g))
                start, budget, level = self.budget[g.gid]
            if g.spent - start >= budget and wait:
                if g.levels == 0 and g.gid not in self.level0:
                    self.level0[g.gid] = max(self._equiv(g, g.spent), self.floor)
                continue  # parked
            chosen.append(g)
        while len(chosen) < self.slots and wait:
            g = wait.pop(0)
            self.readmissions[g.gid] += 1
            self._grant(g, self._level_budget(g) if g.levels >= 1 else self.level0[g.gid])
            chosen.append(g)
        return chosen


class DepthWise(Policy):
    """fixed-2x-depthwise (``arena.depthwise`` + ``Scheduler(budget_policy="depthwise")``, 28 September) on the
    token clock.

    Run it with ``opening_max_wall=0`` (no opening): ``slots`` slots from t = 0. A game without a clear plays a
    level-0 turn of ``quantum`` completion tokens and rotates out if any game waits (else the turn renews). A clear in
    a level-0 turn keeps the slot as a depth game with the fixed-2x budget ``multiple x (k+1) x T1`` (T1 = the game's
    tokens at its level-1 clear over all turns; budgets on the busy clock are tokens at one per-stream rate), granted
    in full at every clear. An exhausted depth game parks only if a game waits; re-admission grants the full level
    budget again (``regrant="full"``, the recorded arms). Refill: ``backlog="depth-first"``: depth games by smallest
    T1, then never-played level-0 games (roster order), then level-0 games by fewest tokens; ``"fixed-2x"``: fewest
    admissions, most levels, roster order. A game parked in this tick does not refill its own slot. The first level's
    32K window enters through ``World.sample(first_efficiency=...)``. Games that arrive with a clear (a run with an
    opening before the depthwise rules) get T1 with opening tokens weighted by ``busy_factor``, as in fixed-2x.
    Parked clears (a clear landing after the park) do not occur on this synchronous token clock."""
    name = "depthwise"

    def __init__(self, quantum=18000.0, multiple=2.0, backlog="depth-first", regrant="full", busy_factor=1.0):
        if backlog not in ("depth-first", "fixed-2x") or regrant != "full":
            raise ValueError("depthwise: backlog depth-first|fixed-2x, regrant full")
        self.quantum, self.multiple, self.backlog, self.busy_factor = quantum, multiple, backlog, busy_factor

    def start(self, games, slots):
        super().start(games, slots)
        self.opening = {g.gid: g.spent for g in games}
        self.slot, self.adm, self.admit_tok, self.budget = {}, {g.gid: 0 for g in games}, {}, {}
        self.seen = {g.gid: g.levels for g in games}
        self.chosen, self.last_spent = [], {}
        self.tokens = {"level0": 0.0, "depth": 0.0}
        self.counts = {"level0_rotate": 0, "level0_clear": 0, "depth_park": 0, "depth_readmit": 0}

    def _t1(self, g):
        o = self.opening[g.gid]
        x = g.level_tokens[0]
        return min(x, o) * self.busy_factor + max(0.0, x - o)

    def _level_budget(self, g):
        return self.multiple * (g.levels + 1) * self._t1(g)

    def _key(self, g):
        if self.backlog == "fixed-2x":
            return (self.adm[g.gid], -g.levels, g.index)
        if g.levels >= 1:
            return (1, self._t1(g), g.index)
        if self.adm[g.gid] == 0:
            return (2, 0.0, g.index)
        return (3, g.spent, g.index)

    def stats(self):
        total = sum(self.tokens.values()) or 1.0
        return {"level0_share": self.tokens["level0"] / total, **{k: float(v) for k, v in self.counts.items()}}

    def choose(self, games, remaining, rate):
        for g in self.chosen:  # token accounting by slot since the last tick
            self.tokens[self.slot.get(g.gid, "level0")] += g.spent - self.last_spent.get(g.gid, g.spent)
        chosen = [g for g in self.chosen if not g.finished]
        held = set(id(g) for g in chosen)

        def waiting(exclude=()):
            return sorted((g for g in games if not g.finished and id(g) not in held and g not in exclude),
                          key=self._key)

        def admit(g):
            self.adm[g.gid] += 1
            self.admit_tok[g.gid] = g.spent
            if g.levels >= 1:
                self.slot[g.gid] = "depth"
                self.budget[g.gid] = (g.spent, self._level_budget(g))
                self.counts["depth_readmit"] += 1
            else:
                self.slot[g.gid] = "level0"
            chosen.append(g)
            held.add(id(g))

        def fill(exclude=()):
            while len(chosen) < self.slots:
                w = waiting(exclude)
                if not w:
                    return
                admit(w[0])

        def leave(g):
            chosen.remove(g)
            held.discard(id(g))

        for g in chosen:  # clears since the last tick
            if g.levels != self.seen[g.gid]:
                self.seen[g.gid] = g.levels
                if self.slot[g.gid] == "level0":
                    self.slot[g.gid] = "depth"
                    self.counts["level0_clear"] += 1
                self.budget[g.gid] = (g.spent, self._level_budget(g))
        fill()
        for g in [x for x in chosen if self.slot[x.gid] == "level0"]:
            if g.spent - self.admit_tok[g.gid] < self.quantum:
                continue
            if waiting():
                leave(g)
                self.counts["level0_rotate"] += 1
                fill(exclude=(g,))
            else:
                self.admit_tok[g.gid] = g.spent
        for g in [x for x in chosen if self.slot[x.gid] == "depth"]:
            start, budget = self.budget[g.gid]
            if g.spent - start >= budget and waiting():
                leave(g)
                self.counts["depth_park"] += 1
                fill(exclude=(g,))
        fill()
        self.chosen = chosen
        self.last_spent = {g.gid: g.spent for g in chosen}
        return list(chosen)


class Depth(Policy):
    """Posterior reward-rate index with commitment (the proposal of 27 September).

    Every game has an unknown multiplier ``m`` on the pooled clear hazard, shared by all its levels, with
    a Gamma(shape, rate) prior; its posterior uses all the game's in-run exposure (every level) and clears.
    A game's index is the best expected score per token over commitment horizons ``H`` (play it for H
    more tokens): E[sum of weights of the levels cleared within H] / H, by Monte Carlo over the posterior.
    Long horizons let a deep game with valuable later levels outrank a fresh game, and a game that just
    cleared a level keeps its rate (the posterior does not reset at a level boundary). Swaps need the
    waiting game to beat the worst active one by ``hysteresis`` after a minimum quantum.
    """
    name = "depth"

    def __init__(self, h0, prior=(0.4, 0.4), horizons=(10e3, 20e3, 40e3, 80e3, 160e3), samples=128,
                 hysteresis=0.25, quantum=8e3, seed=0, level_decay=1.0):
        self.h0, self.prior, self.horizons = h0, prior, np.array(horizons)
        self.samples, self.hysteresis, self.quantum = samples, hysteresis, quantum
        self.rng = np.random.default_rng(seed)
        self.level_decay = level_decay

    def posterior(self, g):
        a, b = self.prior
        n, e = 0, 0.0
        for i, x in enumerate(g.level_tokens):
            e += cum_hazard(self.h0["first" if i == 0 else "deep"], x)
            n += 1
        e += cum_hazard(self.h0["first" if g.levels == 0 else "deep"], g.tok)
        return a + n, b + e

    def value(self, g, remaining_tokens):
        a, b = self.posterior(g)
        m = self.rng.gamma(a, 1.0 / b, self.samples)
        n = g.total_levels
        w = n * (n + 1) / 2
        hor = np.minimum(self.horizons, max(remaining_tokens, 1.0))
        gained = np.zeros((self.samples, len(hor)))
        used = np.zeros(self.samples)
        for lv in range(g.levels + 1, n + 1):
            h = self.h0["first" if lv == 1 else "deep"]
            start = g.tok if lv == g.levels + 1 else 0.0
            base = cum_hazard(h, start)
            targets = self.rng.exponential(size=self.samples) / np.maximum(m, 1e-9) + base
            need = inv_cum_hazard(h, targets) - start
            used = used + need
            gained += (used[:, None] <= hor[None, :]) * (lv / w)
            if np.all(used > hor[-1]):
                break
        rates = gained.mean(axis=0) / hor
        return float(rates.max())

    def choose(self, games, remaining, rate):
        remaining_tokens = remaining * rate
        live = [g for g in games if not g.finished]
        sc = {g.gid: self.value(g, remaining_tokens) for g in live}
        act = self.active(games)
        wait = sorted(self.waiting(games), key=lambda g: (-sc[g.gid], g.index))
        chosen = list(act)
        while len(chosen) < self.slots and wait:
            chosen.append(wait.pop(0))
        if wait:
            elig = [g for g in chosen if g.spent - g.admit_spent >= self.quantum]
            if elig:
                worst = min(elig, key=lambda g: (sc[g.gid], -g.index))
                if sc[wait[0].gid] > (1 + self.hysteresis) * sc[worst.gid]:
                    chosen[chosen.index(worst)] = wait[0]
        return chosen


class Gittins(Policy):
    """Bayesian Gittins-style index over a hierarchical hazard model (27 September).

    A game's clear hazard on level L is ``h0_class(t) x m_g^b x e_L``: ``m_g ~ Gamma(k, k)`` is the game's
    factor shared by all its levels (``b`` = 1 on the first level, ``beta`` deeper), ``e_L ~ Gamma(ke, ke)``
    the level's own factor. The posterior of ``m_g`` is kept on a log grid from every level the game has
    played in this run (clears and the censored current level, ``e`` integrated out in closed form).

    The index is the best expected score per token over plans "play up to ``j`` more levels, giving up
    on a level once ``c`` tokens have been spent on it": a Gittins-style reward rate with the option to
    abandon, so an uncertain fresh game is worth a bounded try and a game that has cleared quickly keeps
    its slot through its next levels. Levels that cannot finish before the end of the run earn nothing.
    """
    name = "gittins"

    def __init__(self, h0, k=0.25, ke=1.5, beta=0.5, caps=(10e3, 20e3, 40e3, 80e3, 160e3), samples=256,
                 hysteresis=0.15, quantum=6e3, seed=0, grid=80, known=None, h0_scale=1.0, flat=False,
                 known_game=None):
        self.known = known  # {game: [multiplier per level]}: the known-rate bound (no learning needed)
        # {game: (m_g, efficiency)}: the known-game bound (28 September): the game factor is known, each level's
        # own factor e_L is not (drawn from Gamma(ke, ke) given the tokens the level has survived).
        self.known_game = known_game
        if flat:  # misspecification checks: no hazard shape, only the class means
            h0 = {c: np.full_like(v, float(np.mean(v))) for c, v in h0.items()}
        h0 = {c: v * h0_scale for c, v in h0.items()}
        self.h0, self.k, self.ke, self.beta = h0, k, ke, beta
        self.caps, self.samples = np.array(caps), samples
        self.hysteresis, self.quantum = hysteresis, quantum
        self.rng = np.random.default_rng(seed)
        self.grid = np.exp(np.linspace(np.log(1e-3), np.log(50.0), grid))
        # Gamma(k, k) prior density on the log grid (times m for the change of variable).
        self.log_prior = k * np.log(self.grid) - k * self.grid
        self.cache = {}

    def _power(self, lv):
        return 1.0 if lv == 1 else self.beta

    def log_post(self, g):
        key = (g.gid, g.levels, round(g.tok, -3))
        if key in self.cache:
            return self.cache[key]
        lp = self.log_prior.copy()
        ke = self.ke
        for i, x in enumerate(g.level_tokens):
            lv = i + 1
            H = cum_hazard(self.h0["first" if lv == 1 else "deep"], x)
            mm = self.grid ** self._power(lv)
            lp += np.log(mm) + (ke + 1) * np.log(ke / (ke + mm * H))
        lv = g.levels + 1
        if lv <= g.total_levels:
            H = cum_hazard(self.h0["first" if lv == 1 else "deep"], g.tok)
            mm = self.grid ** self._power(lv)
            lp += ke * np.log(ke / (ke + mm * H))
        lp -= lp.max()
        self.cache[key] = lp
        return lp

    def index(self, g, slot_tokens):
        S = self.samples
        eff = 1.0
        if self.known_game is not None:
            m0, eff = self.known_game[g.gid]
            m = np.full(S, m0)
        else:
            lp = self.log_post(g)
            p = np.exp(lp)
            p /= p.sum()
            m = self.rng.choice(self.grid, size=S, p=p)
        n = g.total_levels
        w = n * (n + 1) / 2
        caps = self.caps
        C = len(caps)
        # For every cap: tokens used and value gained, per sample, as levels are added.
        used = np.zeros((S, C))
        alive = np.ones((S, C), dtype=bool)
        gained = np.zeros((S, C))
        best = 0.0
        for lv in range(g.levels + 1, n + 1):
            h = self.h0["first" if lv == 1 else "deep"]
            mm = m ** self._power(lv) * eff
            spent = g.tok if lv == g.levels + 1 else 0.0
            Hs = cum_hazard(h, spent)
            if self.known is not None:
                mm, e = np.full(S, self.known[g.gid][lv - 1]), 1.0
            else:
                # The level's own factor, conditioned on surviving ``spent`` tokens.
                e = self.rng.gamma(self.ke, 1.0 / (self.ke + mm * Hs))
            target = Hs + self.rng.exponential(size=S) / np.maximum(mm * e, 1e-12)
            tau = inv_cum_hazard(h, target)  # absolute tokens on the level at the clear
            room = np.maximum(caps[None, :] - spent, 0.0)
            need = tau[:, None] - spent
            clears = need <= room
            cost = np.where(clears, need, room)
            ok = alive & (used + np.where(clears, need, 0) <= slot_tokens)
            gained += np.where(ok & clears, lv / w, 0.0)
            used += np.where(alive, np.minimum(cost, np.maximum(slot_tokens - used, 0)), 0.0)
            alive &= clears & ok
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = gained.mean(axis=0) / np.maximum(used.mean(axis=0), 1.0)
            valid = caps > spent if lv == g.levels + 1 else np.ones(C, dtype=bool)
            if np.any(valid):
                best = max(best, float(rate[valid].max()))
            if not alive.any():
                break
        return best

    def choose(self, games, remaining, rate):
        slot_tokens = remaining * rate
        live = [g for g in games if not g.finished]
        sc = {g.gid: self.index(g, slot_tokens) for g in live}
        act = self.active(games)
        wait = sorted(self.waiting(games), key=lambda g: (-sc[g.gid], g.index))
        chosen = list(act)
        while len(chosen) < self.slots and wait:
            chosen.append(wait.pop(0))
        if wait:
            elig = [g for g in chosen if g.spent - g.admit_spent >= self.quantum]
            if elig:
                worst = min(elig, key=lambda g: (sc[g.gid], -g.index))
                if sc[wait[0].gid] > (1 + self.hysteresis) * sc[worst.gid]:
                    chosen[chosen.index(worst)] = wait[0]
        return chosen


class RoundRobin(Policy):
    name = "round-robin"

    def __init__(self, quantum=20e3):
        self.quantum = quantum

    def choose(self, games, remaining, rate):
        chosen = [g for g in self.active(games) if g.spent - g.admit_spent < self.quantum]
        wait = sorted(self.waiting(games), key=lambda g: (g.spent, g.index))
        return chosen + wait[: self.slots - len(chosen)]


def rank_score(g, horizon, c0, t0, gamma, clears="all"):
    """The continuous ranking score (29 September, user; ``Rank`` and ``arena.rank``): the value of the game's next
    level times its recency-weighted clears per token, both on the game's own token clock.

    ``rate = (c0 + sum_i exp(-age_i / H)) / (t0 + H (1 - exp(-spent / H)))``: ``age_i`` is the tokens the game has
    spent since its i-th clear (so a clear 5K tokens ago counts ~1, one 3H ago ~0.05), the denominator is the same
    exponential weight over every token the game has spent, and ``c0 / t0`` is the prior rate of a game with no
    evidence (an unplayed game). A game that just cleared ranks high; one that has spent many tokens since its latest
    clear decays towards ``c0 / (t0 + H)``. ``value = (levels + 1)^gamma / (N (N + 1) / 2)``: the next level's share
    of the game's score (gamma 1), so between games with similar recency the deeper one ranks first.
    ``clears="last"`` counts only the latest clear."""
    ages, acc = [], g.tok
    for x in reversed(g.level_tokens):
        ages.append(acc)
        acc += x
    if clears == "last":
        ages = ages[:1]
    num = c0 + sum(math.exp(-a / horizon) for a in ages)
    den = t0 + horizon * (1.0 - math.exp(-g.spent / horizon))
    n = g.total_levels
    return (g.levels + 1) ** gamma / (n * (n + 1) / 2) * num / den


class Rank(Policy):
    """Continuous ranking with periodic refill (29 September, user: "a token (cost) and level clear-based continuous
    ranking of each game ... update the focus slots e.g. every 1 minute; prioritize games that cleared levels
    recently (according to token count), and if token counts are similar then games with deeper levels").

    Every ``tick`` wall seconds every live game is scored by ``rank_score``; free slots take the best waiting games,
    then the worst focus game that has held its slot for ``tenure`` tokens is swapped for the best waiting game while
    the waiting game's score is more than ``1 + hysteresis`` times its own (each swap re-prefills the incoming game,
    the simulator's ``readmit_cost``). No level budgets; with ``budget_multiple`` a focus game past
    ``budget_multiple x L x T1`` tokens on its level (token budgets, T1 = its tokens to the first clear) has its score
    multiplied by ``over_budget`` and may be swapped before its tenure. ``horizon_rel``: H = horizon_rel x the median
    tokens per cleared level over the run so far (``horizon`` until the first clear; floor 2K). Between ticks a slot
    freed by a finished game refills from the latest ranking."""
    name = "rank"

    def __init__(self, horizon=40e3, c0=0.5, t0=20e3, gamma=1.0, hysteresis=0.3, tenure=10e3, tick=60.0,
                 clears="all", horizon_rel=None, budget_multiple=None, over_budget=0.5):
        if clears not in ("all", "last"):
            raise ValueError("clears: all | last")
        self.horizon, self.c0, self.t0, self.gamma = horizon, c0, t0, gamma
        self.hysteresis, self.tenure, self.tick, self.clears = hysteresis, tenure, tick, clears
        self.horizon_rel, self.budget_multiple, self.over_budget = horizon_rel, budget_multiple, over_budget

    def start(self, games, slots):
        super().start(games, slots)
        self.last, self.order = None, []
        self.counts = {"ticks": 0, "swaps": 0, "fills": 0}
        # Tick study (29 September): the wall seconds between the first simulator step at which a swap would pass
        # (best waiting > (1 + hysteresis) x the worst focus game past its tenure) and the tick that makes it (the
        # first swap of each tick), and the tokens an outgoing game had spent since its latest clear (or, without a
        # clear, since its start). Recorded only; no decision depends on them.
        self.pending, self.delays, self.out_tok = None, [], []

    def stats(self):
        d = np.array(self.delays) if self.delays else np.zeros(1)
        return {**{k: float(v) for k, v in self.counts.items()}, "swap_delay_mean": float(d.mean()),
                "swap_delay_p90": float(np.percentile(d, 90)),
                "out_tokens_since_clear": float(np.mean(self.out_tok)) if self.out_tok else 0.0}

    def _would_swap(self, games, chosen, horizon):
        wait = self.waiting(games)
        elig = [g for g in chosen if g.spent - g.admit_spent >= self.tenure or self._over(g)]
        if not wait or not elig:
            return False
        best = max(self.score(g, horizon) for g in wait)
        return best > (1 + self.hysteresis) * min(self.score(g, horizon) for g in elig)

    def _horizon(self, games):
        if self.horizon_rel is None:
            return self.horizon
        cleared = [x for g in games for x in g.level_tokens if x > 0]
        return max(self.horizon_rel * float(np.median(cleared)), 2e3) if cleared else self.horizon

    def _over(self, g):
        return (self.budget_multiple is not None and g.levels >= 1
                and g.tok > self.budget_multiple * (g.levels + 1) * g.level_tokens[0])

    def score(self, g, horizon):
        s = rank_score(g, horizon, self.c0, self.t0, self.gamma, self.clears)
        return s * self.over_budget if self._over(g) else s

    def choose(self, games, remaining, rate):
        chosen = self.active(games)
        now = -remaining
        if self.pending is None and self._would_swap(games, chosen, self._horizon(games)):
            self.pending = now
        if self.last is not None and self.last - remaining < self.tick:
            if len(chosen) < self.slots:  # a finished game's slot: refill from the latest ranking
                wait = [g for g in self.order if not g.active and not g.finished]
                chosen += wait[: self.slots - len(chosen)]
            return chosen
        self.last = remaining
        self.counts["ticks"] += 1
        horizon = self._horizon(games)
        sc = {g.gid: self.score(g, horizon) for g in games if not g.finished}
        wait = sorted(self.waiting(games), key=lambda g: (-sc[g.gid], g.index))
        while len(chosen) < self.slots and wait:
            chosen.append(wait.pop(0))
            self.counts["fills"] += 1
        while wait:
            elig = [g for g in chosen if g.spent - g.admit_spent >= self.tenure or self._over(g)]
            if not elig:
                break
            worst = min(elig, key=lambda g: (sc[g.gid], -g.index))
            if sc[wait[0].gid] <= (1 + self.hysteresis) * sc[worst.gid]:
                break
            if self.pending is not None:
                self.delays.append(now - self.pending)
                self.pending = None
            self.out_tok.append(worst.tok)
            chosen[chosen.index(worst)] = wait.pop(0)
            self.counts["swaps"] += 1
        self.pending = None  # the tick has acted on (or rejected) every pending swap
        self.order = sorted((g for g in games if not g.finished), key=lambda g: (-sc[g.gid], g.index))
        return chosen


def load_rows(path, extra=(), exclude_runs=()):
    """Flash-Next xhigh attempts on the public games from ``path`` and the ``extra`` files (the c110 runs'
    ``c110_runs.py`` rows, 28 September), without the runs in ``exclude_runs``."""
    rows = []
    for p in (path, *extra):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    return [r for r in rows if r.get("model") == "Qwen/Qwen3.8-Flash-Next-NVFP4" and r.get("effort") == "xhigh"
            and not r["game"].startswith("q") and r.get("run") not in exclude_runs]


def make_policy(name, world, **kw):
    if name == "oracle":
        return Oracle()
    if name == "index":
        return Index(**kw)
    if name == "commit":
        return Commit(**kw)
    if name == "depth":
        return Depth(world.h0, **kw)
    if name == "gittins":
        return Gittins(world.h0, **kw)
    if name == "round-robin":
        return RoundRobin(**kw)
    if name == "rank":
        return Rank(**kw)
    if name == "fixed-2x":
        return FixedTwoX(**kw)
    if name == "depthwise":
        return DepthWise(**kw)
    raise ValueError(name)


ROSTERS = {"23": GAMES23, "110": [g for _ in range(4) for g in GAMES23] + GAMES23[:18], "c110": C110, "c20": C20}


def run_one(world, name, kw, sim_kw, runs, seed, games, shared=False):
    """``runs`` simulated runs of one policy (common seeds across policies)."""
    rng = np.random.default_rng(seed)
    share_rng = np.random.default_rng(seed + 7919)
    sim_kw = dict(sim_kw)
    wkw = {"efficiency": sim_kw.pop("efficiency", 1.0), "population": sim_kw.pop("population", False),
           "first_efficiency": sim_kw.pop("first_efficiency", 1.0), "tail": sim_kw.pop("tail", None)}
    shared = sim_kw.pop("shared", shared)
    # ``shuffle``: a fresh roster order every run (so a selective opening of the first N games is a random subset,
    # not one clone of each c110 source game).
    shuffle = sim_kw.pop("shuffle", False)
    order_rng = np.random.default_rng(seed + 104729)
    # ``roster_games``: this policy's own roster (a list of games; e.g. 110 clones of a held-out half).
    games = sim_kw.pop("roster_games", games)
    # ``bootstrap`` {"pool": [games], "k": 20, "n": 110, "per_roster": R} (28 September robustness check): every block
    # of R runs plays a roster of ``n`` cyclic clones of ``k`` games resampled with replacement from ``pool`` (a game
    # drawn twice is two independent games, ``g`` and ``g~2``). Block b uses the same roster in every policy.
    boot = sim_kw.pop("bootstrap", None)
    scores, levels, switches, per_game, played, cleared1, stats = [], [], [], {}, [], [], []
    for i in range(runs):
        if boot:
            brng = np.random.default_rng([seed, 99, i // boot["per_roster"]])
            picks = brng.choice(len(boot["pool"]), boot["k"], replace=True)
            seen_b, names = {}, []
            for j in picks:
                b = boot["pool"][j]
                seen_b[b] = seen_b.get(b, 0) + 1
                names.append(b if seen_b[b] == 1 else f"{b}~{seen_b[b]}")
            games = [names[j % len(names)] for j in range(boot.get("n", 110))]
        roster = [games[i] for i in order_rng.permutation(len(games))] if shuffle else games
        th = world.sample(roster, rng, shared=shared, **wkw)
        if name == "known-rate":
            pol = Gittins(world.h0, known=world.last_multipliers, **kw)
        elif name == "known-game":  # the game factor known, the level factors learned (within world only)
            pol = Gittins(world.h0, known_game=world.last_game_factor,
                          **{"k": world.hier["k"], "ke": world.hier["ke"], "beta": world.hier["beta"], **kw})
        else:
            pol = make_policy(name, world, **dict(kw))
        res = simulate(pol, th, share_rng=share_rng, **sim_kw)
        scores.append(res["score"])
        levels.append(res["levels"])
        switches.append(res["switch"])
        played.append(res["played"])
        cleared1.append(res["cleared1"])
        if res["stats"] is not None:
            stats.append(res["stats"])
        for g, lv in res["games"].items():
            per_game.setdefault(g.split("#")[0], []).append(lv)
    return {"mean": float(np.mean(scores)), "sd": float(np.std(scores)), "per_roster": (boot or {}).get("per_roster"),
            "se": float(np.std(scores) / math.sqrt(runs)), "levels": float(np.mean(levels)),
            "switch": float(np.mean(switches)), "scores": [round(s, 3) for s in scores],
            "played": played, "cleared1": cleared1,
            **({"stats": {k: round(float(np.mean([s[k] for s in stats])), 4) for k in stats[0]}} if stats else {}),
            "per_game_levels": {g: round(float(np.mean(v)), 2) for g, v in per_game.items()}}


def run(policies, world, runs, seed, slots=2, games=GAMES23, sim_kw=None, shared=False):
    out = {}
    for label, spec in policies.items():
        name, kw = spec[0], spec[1]
        skw = {"slots": slots, **(sim_kw or {}), **(spec[2] if len(spec) > 2 else {})}
        out[label] = run_one(world, name, kw, skw, runs, seed, games, shared)
        print(f"{label:28s} {out[label]['mean']:6.2f} ± {out[label]['se']:.2f} (sd {out[label]['sd']:.2f}) "
              f"levels {out[label]['levels']:.1f}", flush=True)
    return out


_WORLD = {}


def _sweep_job(item):
    label, name, kw, skw, runs, seed, games, shared, key = item
    return label, run_one(_WORLD[key], name, kw, skw, runs, seed, games, shared)


def sweep(config, attempts, jobs=8):
    """``config``: {"world": "pooled"|"within", "roster": "23"|"110"|"c110", "runs", "seed", "shared",
    "sim": {simulate kwargs for every policy}, "policies": {label: [policy, {policy kwargs}, {simulate kwargs}]}}.
    Policies run in parallel processes (fork) on one fitted world."""
    import multiprocessing as mp

    extra, exclude = tuple(config.get("extra_attempts", ())), tuple(config.get("exclude_runs", ()))
    key = (attempts, config.get("world", "pooled"), extra, exclude)
    if key not in _WORLD:
        _WORLD[key] = World.fit(load_rows(attempts, extra, exclude), shape=key[1])
    _WORLD[key].fine_start = bool(config.get("fine_start", False))
    games = ROSTERS[str(config.get("roster", "23"))]
    items = [(label, spec[0], spec[1], {**config.get("sim", {}), **(spec[2] if len(spec) > 2 else {})},
              config.get("runs", 20), config.get("seed", 0), games, config.get("shared", False), key)
             for label, spec in config["policies"].items()]
    out = {}
    with mp.get_context("fork").Pool(min(jobs, len(items))) as pool:
        for label, res in pool.imap_unordered(_sweep_job, items):
            out[label] = res
            print(f"{label:40s} {res['mean']:6.2f} ± {res['se']:.2f} (sd {res['sd']:.2f}) levels {res['levels']:6.1f} "
                  f"switch {res['switch']:6.0f}s", flush=True)
    return {label: out[label] for label in config["policies"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies", nargs="+", default=["oracle", "index", "commit", "round-robin", "depth"])
    parser.add_argument("--world", choices=("pooled", "within"), default="pooled")
    parser.add_argument("--sweep", help="JSON sweep config (see sweep()); overrides --policies/--world")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.sweep:
        res = sweep(json.loads(open(args.sweep).read()), args.attempts, args.jobs)
    else:
        world = World.fit(load_rows(args.attempts), shape=args.world)
        res = run({p: (p, {}) for p in args.policies}, world, args.runs, args.seed)
    if args.out:
        open(args.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
