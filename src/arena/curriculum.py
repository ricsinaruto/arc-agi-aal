"""Competence-frontier curriculum for the multi-game RL loop (envs arm, docs/envs-plan.md section 5).

A unit is one (game, level). The frontier is shaped by the run's reward (the logeq RHAE x token reward) and by
learning progress in RHAE per 10K tokens, not by solve rate. Each unit holds a discounted Beta posterior
of its clear rate within the cap, discounted moments of its clears' rewards, its RHAE/10K history and a pool of starts
(fresh level-1 layouts or restored level starts, from the probe recordings and the loop's in-run harvest,
``arena.harvest``). Each update the scheduler picks ``groups`` units, ``group_size`` tries each:

1. Split the groups over sources by the configured shares (largest remainder).
2. Inside a source, score each unit on one Thompson draw: the expected reward spread of a group,
   ``sqrt(p(1-p)(mu_c - r_fail)^2 + p sigma_c^2) x (1 - (1-p)^G)`` over the binary maximum (all-fail groups carry
   nothing; all-clear groups keep their spread among clears), plus ``progress`` x the relative change in RHAE/10K
   between the unit's recent and earlier observations (absolute learning progress), minus a stagnation penalty.
3. Pick the best unit per family, families best first; one family holds at most ``family_cap`` of the groups, and
   near-solved units (posterior mean above ``token_p``) at most ``token_share``.
4. Spend ``probe_share`` of the groups on units not yet observed in the run, or parked units due a re-probe.

Solve rate enters through the cap only. A unit whose failures are mostly budget-outs (``cap_out`` share) and whose
clear rate is under ``park_p`` moves up a cap tier (cap x ``tier_factor``, at most ``max_tier`` steps) before it parks;
a unit cleared at ``retire_p`` or better with its median clear inside ``master_cap`` of the cap and RHAE near the
baseline (``master_rhae``), tokens no longer falling, retires as mastered. Pure python with no I/O except
``state_dict`` / ``load_state``, so the same code runs in the loop, the CPU preflight and the offline backtest.
"""

from __future__ import annotations

import math
import random
import statistics

DEFAULTS = {
    "groups": 12,
    "group_size": 12,
    "half_life": 2.0,  # updates; older run evidence counts half after this many updates (the probe prior never decays)
    "prior_weight": 4.0,  # the probe prior counts as at most this many tries (built on another serving build)
    "log_clip": 8.0,  # reward: clear clip(log RHAE ratio + log token ratio, +-log 8), fail -log 16 (mra-v2 logeq)
    "log_fail": 16.0,
    "progress": 0.5,
    "stall_after": 3,  # observations without a better reward or RHAE/10K
    "stall_penalty": 0.15,
    "family_cap": 0.10,
    "token_p": 0.85,
    "token_share": 0.25,
    "probe_share": 0.10,
    "retire_p": 0.95,
    "master_cap": 0.5,  # mastered: median clear within this share of the cap ...
    "master_rhae": 0.9,  # ... at this RHAE or better
    "park_p": 0.10,
    "park_tries": 12,
    "reprobe_every": 4,
    "cap_out": 0.6,  # share of failures that are budget-outs: the cap, not the level, is the limit
    "tier_factor": 1.5,
    "max_tier": 2,
    # hero rebuild: "park" = tier up only on the park condition (no clears at park_p 0); "cap_out" = tier
    # up once park_tries tries show >= cap_out of the failures as budget-outs and a clear rate under tier_clear_max
    "tier_trigger": "park",
    "tier_clear_max": 0.5,
    "cap_ceiling": None,  # no cap (pool, running or tier) above this many tokens (131,072)
    # "logeq" = spread of the loop's log reward + LP on the group's RHAE/10K (ratio of sums);
    # "rhae_tok" = spread of the per-try z = RHAE x T_b / tokens (0 for a fail or a cut) + LP as the t-statistic of the
    # per-try mean z between the newer and older halves of the last observations
    "score": "logeq",
    "z_prior_mu": 1.1,  # a base clear's z (log-token sd 0.45 around T_b: E[exp(N(0, 0.45))])
    "z_prior_sigma": 0.5,
    "z_prior_n": 2.0,
    "z_hi": 6.0,  # z clipped here (as token_hi)
    # "rhae_own": z = RHAE x T_own / tokens with T_own the level's own median clear
    # tokens (no reference table, no clear rate in the score); a unit scores one Thompson draw of its mean observed
    # within-group z SD, shrunk to a prior: its game's units' mean SD, else every unit's (wide: own_prior_wide x the
    # SD across units, at least own_prior_floor), so unplayed games draw high often enough to be played; LP is
    # progress_z on the own-median z
    "own_prior_n": 1.0,
    "own_prior_wide": 2.0,
    "own_prior_floor": 0.2,
    "own_prior_mu0": 0.5,  # before any group anywhere
    "lp_window": 6,
    "repeat": True,  # a pool smaller than the update's groups: units take several groups
    "unit_share": 0.5,
    # level balance: equal groups per level depth, e.g. [1, 2, 3, 4] = L1 / L2 / L3 / L4+ each a quarter
    # of the update (a depth with no playable unit gives its groups to the others); None = one pool
    "depth_bins": None,
    # envs usefulness audit 24 Oct (hero v3-v6g, 84 groups): the long L1/L2 units (cn04-L2, wa30-L1/L2, p0ch-L2) took 34%
    # of tokens at 14-20 reward-sd x tries per Mtok vs 42-44 for L1 and L3+ overall; the lowest-yield 40% of tokens gave
    # 16% of the spread. ``token_cost`` > 0 scales a unit's spread score by (median over played units / its own expected
    # tokens per try) ^ token_cost, both from this run's tries only (reference free); 0 = off. ``deep_share``: plan_one keeps at least
    # this share of the last ``window`` groups at level >= ``deep_level`` when the source has such a unit (L3+ gave 42
    # per Mtok vs L2 30, 19 of 84 groups); 0 = off. All three are live (rl_async CURRICULUM_LIVE).
    "token_cost": 0.0,
    "deep_share": 0.0,
    "deep_level": 3,
    # v7 (one group at a time, plan_one): family and near-solved caps count over the last ``window`` groups
    "window": 12,
    # chains: up to ``chain_groups`` of each step's groups continue a game from a
    # cleared try's next-level start (``plan_chain``); one chain group per game per step, the family cap holds; the
    # deepest open level of a game gets ``chain_bonus`` on its score; a near-certain level (posterior mean >=
    # ``chain_sure_p`` with a flat token trend, progress_z < ``chain_flat``) takes at most ``chain_sure_max`` chain
    # groups per step. The caller owns budgets (``chain_budget`` tokens per game chain) and the stuck rule.
    "chain_groups": 0,
    "chain_bonus": 0.5,
    "chain_budget": 300000,
    "chain_stuck": 2,
    "chain_reopen": 4,
    "chain_sure_p": 0.95,
    "chain_flat": 0.34,
    "chain_sure_max": 1,
}


def usefulness(p, g, mix=1.0, token=0.3):
    """Solve-rate usefulness of one group (kept for comparison in the backtest): P(mixed) + token x P(all clear)."""
    p = min(1.0, max(0.0, p))
    return mix * (1.0 - p ** g - (1.0 - p) ** g) + token * p ** g


def spread(p, g, mu, sigma, fail):
    """Expected reward spread of a group of ``g`` tries: the std of a try's reward (clear at rate ``p`` with reward
    mean ``mu`` and std ``sigma``, else ``fail``) times the chance the group is not all-fail."""
    p = min(1.0, max(0.0, p))
    var = p * (1.0 - p) * (mu - fail) ** 2 + p * sigma ** 2
    return math.sqrt(max(0.0, var)) * (1.0 - (1.0 - p) ** g)


def spread_z(p, g, mu, sigma):
    """Expected within-group SD of the per-try z (clear at rate ``p`` with z mean ``mu`` and std ``sigma``, else 0), times
    the chance the group is not all-fail."""
    p = min(1.0, max(0.0, p))
    var = p * (sigma ** 2 + mu ** 2) - (p * mu) ** 2
    return math.sqrt(max(0.0, var) * (g - 1) / g) * (1.0 - (1.0 - p) ** g)


class Unit:
    __slots__ = ("key", "game", "level", "source", "family", "a", "b", "tokens", "starts", "used", "seen", "tries",
                 "best_r", "best_m", "stall", "state", "parked_at", "picked", "tok0", "won", "cap", "tier",
                 "rn", "rs", "rss", "rate", "outs", "fails", "rhae", "pa", "pb", "zn", "zs", "zss", "zrate", "sdz",
                 "tok_obs")
    OPTIONAL = {"pa": 0.0, "pb": 0.0, "zn": 0.0, "zs": 0.0, "zss": 0.0, "zrate": [], "sdz": [], "tok_obs": []}  # slots an older state may lack

    def __init__(self, key, game, level, source, family, clears=0, fails=0, tokens=None, prior_weight=4.0, cap=None):
        self.key, self.game, self.level, self.source, self.family = key, game, int(level), source, family
        n = clears + fails
        w = min(1.0, prior_weight / n) if n else 0.0
        self.pa, self.pb = w * clears, w * fails  # the probe prior's pseudo-counts: kept through decay (a multi-game
        # pool sees most units once in many updates; a decayed prior would turn every unseen unit into Beta(1, 1))
        self.a, self.b = 1.0 + self.pa, 1.0 + self.pb
        self.tokens = list(tokens or [])[-8:]
        self.tok_obs = []  # this run's tries, per try: clear tokens, the cap for a budget-out, None for another fail
        self.starts, self.used = [], 0
        self.seen = 0  # observations in this run (not the prior)
        self.tries, self.won, self.outs, self.fails = 0, 0, 0, 0  # this run's tries, clears, budget-outs, failures
        self.best_r, self.best_m, self.stall = None, None, 0
        self.state, self.parked_at, self.picked = "active", None, 0
        self.tok0 = statistics.median(self.tokens) if self.tokens else None  # token reference (T_b of the reward)
        self.cap, self.tier = cap, 0
        self.rn, self.rs, self.rss = 0.0, 0.0, 0.0  # discounted count, sum and sum of squares of clear rewards
        self.rate = []  # [(update, RHAE per 10K tokens of the observation)]
        self.rhae = []  # recent clears' RHAE
        self.zn, self.zs, self.zss = 0.0, 0.0, 0.0  # discounted count, sum and sum of squares of clears' z
        self.zrate = []  # [(update, mean z over the group's tries, its standard error)]
        self.sdz = []  # rhae_own: [(update, the group's z SD)]

    @property
    def mean(self):
        return self.a / (self.a + self.b)

    def median_tokens(self):
        return statistics.median(self.tokens) if self.tokens else None

    def clear_moments(self, prior_mu=0.0, prior_sigma=1.0, prior_n=1.0):
        """Mean and std of a clear's reward, shrunk towards the prior with ``prior_n`` pseudo-clears."""
        n = self.rn + prior_n
        mu = (self.rs + prior_n * prior_mu) / n
        var = (self.rss + prior_n * (prior_sigma ** 2 + prior_mu ** 2)) / n - mu ** 2
        return mu, math.sqrt(max(0.0, var))

    def z_moments(self, mu0, s0, n0):
        """Mean, std and effective count of a clear's z, shrunk towards (mu0, s0) with ``n0`` pseudo-clears."""
        n = self.zn + n0
        mu = (self.zs + n0 * mu0) / n
        var = (self.zss + n0 * (s0 ** 2 + mu0 ** 2)) / n - mu ** 2
        return mu, math.sqrt(max(0.0, var)), n

    def progress_z(self, window=6):
        """|t| of the per-try mean z between the newer and older halves of the last ``window`` observations, clipped to
        [0, 3] / 3 (a thin level's noise does not read as progress)."""
        obs = self.zrate[-int(window):]
        if len(obs) < 2:
            return 0.0
        h = len(obs) // 2
        old, new = obs[:h], obs[h:]
        mo, mn = statistics.mean(m for _, m, _ in old), statistics.mean(m for _, m, _ in new)
        se = math.sqrt(sum(e * e for _, _, e in old) / len(old) ** 2 + sum(e * e for _, _, e in new) / len(new) ** 2)
        return min(3.0, abs(mn - mo) / max(se, 1e-3)) / 3.0

    def progress(self):
        """Relative change of RHAE/10K between the newer and older halves of the observations (absolute LP)."""
        if len(self.rate) < 2:
            return 0.0
        vals = [v for _, v in self.rate[-6:]]
        h = len(vals) // 2
        old, new = statistics.mean(vals[:h]), statistics.mean(vals[h:])
        return abs(new - old) / max(old, new, 1e-6)


class Curriculum:
    def __init__(self, cfg=None, shares=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.shares = dict(shares or {})
        self.units = {}
        self.blocked = set()
        self.update = 0
        self.log = []
        self.made = {}  # plan_one: groups planned per source over the run (the mix holds by deficit)
        self.recent = []  # plan_one: the last ``window`` picks [unit key, family, near-solved]

    # ---- pool -----------------------------------------------------------------------------------------------
    @staticmethod
    def key(game, level):
        return f"{game}@L{int(level)}"

    def add_unit(self, game, level, source, family, clears=0, fails=0, tokens=None, cap=None):
        k = self.key(game, level)
        if k not in self.units:
            self.units[k] = Unit(k, game, level, source, family, clears, fails, tokens, self.cfg["prior_weight"],
                                 self.capped(cap))
        return self.units[k]

    def add_start(self, rec, create=True):
        """One start record (probe or harvest JSONL row): {game, level, source, family, start} with ``start`` either
        {"fresh": true, "seed": n} or {"attempt": path, "request": n}. A new unit inherits nothing but its source;
        ``create=False`` (the loop: a unit needs its training set) drops a start of an unknown unit."""
        u = self.units.get(self.key(rec["game"], rec["level"]))
        if u is None:
            if not create:
                return None
            u = self.add_unit(rec["game"], rec["level"], rec["source"], rec.get("family") or rec["game"])
        start = rec["start"]
        if start not in u.starts:
            u.starts.append(start)
        return u

    # ---- evidence -------------------------------------------------------------------------------------------
    @property
    def fail_reward(self):
        return -math.log(float(self.cfg["log_fail"]))

    def reward(self, u, cleared, tokens, rhae=None):
        """The loop's logeq reward of one try: clip(log(RHAE / 1) + log(T_b / tokens), +-log clip); fail -log fail.
        RHAE defaults to the baseline's 1; T_b is the unit's reference median tokens."""
        if not cleared or not tokens:
            return self.fail_reward
        lo = math.log(float(self.cfg["log_clip"]))
        t_b = u.tok0 or tokens
        term = math.log(max(1e-6, rhae if rhae is not None else 1.0)) + math.log(t_b / tokens)
        return max(self.fail_reward, min(lo, term))

    def decay(self):
        d = 0.5 ** (1.0 / float(self.cfg["half_life"]))
        for u in self.units.values():  # run evidence decays; the probe prior (pa, pb) stays
            u.a = 1.0 + u.pa + d * max(0.0, u.a - 1.0 - u.pa)
            u.b = 1.0 + u.pb + d * max(0.0, u.b - 1.0 - u.pb)
            u.rn, u.rs, u.rss = d * u.rn, d * u.rs, d * u.rss
            u.zn, u.zs, u.zss = d * u.zn, d * u.zs, d * u.zss

    def observe(self, game, level, outcomes):
        """``outcomes``: one played group, per try ``(cleared, tokens[, reward[, rhae[, budget_out]]])``; a missing
        reward is computed as the loop's (``reward``)."""
        u = self.units.get(self.key(game, level))
        if u is None or not outcomes:
            return None
        cfg = self.cfg
        rows = []
        for o in outcomes:
            o = tuple(o) + (None,) * (5 - len(o))
            ok, t, r, rh, out = bool(o[0]), o[1], o[2], o[3], o[4]
            if r is None:
                r = self.reward(u, ok, t, rh)
            rows.append((ok, t, r, rh, out))
        c = sum(1 for ok, *_ in rows if ok)
        u.a += c
        u.b += len(rows) - c
        u.tries += len(rows)
        u.won += c
        u.fails += len(rows) - c
        u.outs += sum(1 for ok, *_, out in rows if not ok and out is not False)  # unknown counts as a budget-out
        u.seen += 1
        u.tok_obs = (list(u.tok_obs) + [int(t) if ok and t else (int(u.cap) if not ok and out is not False and u.cap
                                                                   else None) for ok, t, _, _, out in rows])[-48:]
        self._pool_tok = None
        for ok, t, r, rh, _ in rows:
            if ok:
                u.rn, u.rs, u.rss = u.rn + 1, u.rs + r, u.rss + r * r
                if t:
                    u.tokens = (u.tokens + [int(t)])[-24:]
                u.rhae = (u.rhae + [1.0 if rh is None else float(rh)])[-24:]
        if u.tok0 is None:
            u.tok0 = u.median_tokens()
        zs = [self.z_value(u, ok, t, rh) for ok, t, _, rh, _ in rows]
        for (ok, *_), z in zip(rows, zs):
            if ok:
                u.zn, u.zs, u.zss = u.zn + 1, u.zs + z, u.zss + z * z
        se = statistics.pstdev(zs) / math.sqrt(len(zs)) if len(zs) > 1 else float(cfg["z_prior_sigma"])
        u.zrate = (list(u.zrate) + [(self.update, statistics.mean(zs), se)])[-24:]
        u.sdz = (list(u.sdz) + [(self.update, statistics.pstdev(zs) if len(zs) > 1 else 0.0)])[-24:]
        self._own = None
        toks = sum(int(t or 0) for _, t, *_ in rows) or 1
        m = 1e4 * sum((1.0 if rh is None else float(rh)) for ok, _, _, rh, _ in rows if ok) / toks
        u.rate.append((self.update, m))
        mean_r = statistics.mean(r for _, _, r, _, _ in rows)
        better = (u.best_r is None or mean_r > u.best_r + 0.1) or (u.best_m is None or m > 1.1 * u.best_m)
        u.stall = 0 if better else u.stall + 1
        u.best_r = mean_r if u.best_r is None else max(u.best_r, mean_r)
        u.best_m = m if u.best_m is None else max(u.best_m, m)
        med = u.median_tokens()
        mastered = (u.seen >= 2 and u.won >= cfg["retire_p"] * u.tries and u.stall >= 1
                    and (u.cap is None or (med is not None and med <= cfg["master_cap"] * u.cap))
                    and (not u.rhae or statistics.median(u.rhae) >= cfg["master_rhae"]))
        ceiling = cfg.get("cap_ceiling")
        room = u.cap and u.tier < cfg["max_tier"] and (not ceiling or int(u.cap) < int(ceiling))
        if mastered:
            u.state = "retired"
        elif (cfg.get("tier_trigger") == "cap_out" and room and u.tries >= cfg["park_tries"] and u.fails
              and u.outs >= cfg["cap_out"] * u.fails and u.won < float(cfg["tier_clear_max"]) * u.tries):
            self.tier_up(u)  # the cap, not the level, is the limit (whatever park_p is)
        elif u.tries >= cfg["park_tries"] and u.won <= cfg["park_p"] * u.tries:
            if room and u.fails and u.outs >= cfg["cap_out"] * u.fails:
                self.tier_up(u)
            else:
                u.state, u.parked_at = "parked", self.update
        elif u.state == "parked" and c > cfg["park_p"] * len(rows) * 2:
            u.state = "active"
        return u

    def z_value(self, u, cleared, tokens, rhae=None):
        """A try's z = RHAE x T_b / tokens (T_b the unit's reference median tokens), clipped to z_hi; 0 for a fail."""
        if not cleared or not tokens:
            return 0.0
        t_b = (u.median_tokens() if self.cfg.get("score") == "rhae_own" else u.tok0) or tokens
        return min(float(self.cfg["z_hi"]), max(0.0, (1.0 if rhae is None else float(rhae)) * t_b / float(tokens)))

    def own_prior(self):
        """rhae_own's prior: (global mean, global sd, {game: mean}) of the units' mean observed group z SD."""
        if getattr(self, "_own", None) is not None:
            return self._own
        cfg, w = self.cfg, int(self.cfg["lp_window"])
        means, games = [], {}
        for u in self.units.values():
            if u.sdz:
                m = statistics.mean(x for _, x in u.sdz[-w:])
                means.append(m)
                games.setdefault(u.game, []).append(m)
        mu = statistics.mean(means) if means else float(cfg["own_prior_mu0"])
        sd = max(float(cfg["own_prior_floor"]), float(cfg["own_prior_wide"]) * (statistics.pstdev(means) if len(means) > 1 else mu))
        self._own = (mu, sd, {g: statistics.mean(v) for g, v in games.items()})
        return self._own

    def capped(self, cap):
        ceiling = self.cfg.get("cap_ceiling")
        return int(min(int(ceiling), int(cap))) if ceiling and cap else cap

    def tier_up(self, u):
        """A higher cap tier (x tier_factor, at most cap_ceiling); the run's evidence restarts at half weight."""
        u.tier += 1
        u.cap = self.capped(int(round(u.cap * self.cfg["tier_factor"])))
        u.tries = u.won = u.outs = u.fails = 0
        u.a, u.b = 1.0 + 0.5 * (u.a - 1.0), 1.0 + 0.5 * (u.b - 1.0)
        u.pa, u.pb = 0.5 * u.pa, 0.5 * u.pb

    # ---- scheduling -----------------------------------------------------------------------------------------
    def predict(self, u, p=None):
        """Expected (try reward mean, group reward spread) at clear rate ``p`` (default the posterior mean)."""
        g, fail = int(self.cfg["group_size"]), self.fail_reward
        p = u.mean if p is None else p
        mu, sigma = u.clear_moments()
        return p * mu + (1 - p) * fail, spread(p, g, mu, sigma, fail)

    @staticmethod
    def own_tokens(u):
        """Expected tokens per try from the unit's own tries this run (clears, budget-outs at the cap, other fails at the
        median own clear); None before any."""
        obs = list(u.tok_obs or [])
        if not obs:
            return None
        clr = sorted(x for x in obs if x is not None)
        fill = clr[len(clr) // 2] if clr else (u.cap or None)
        vals = [x if x is not None else fill for x in obs]
        vals = [v for v in vals if v]
        return sum(vals) / len(vals) if vals else None

    def token_factor(self, u):
        """(median own expected tokens per try over the units played this run / the unit's own) ^ token_cost: reference
        free, from each level's own history only; 1 when off or the unit has no tries yet."""
        cost = float(self.cfg.get("token_cost") or 0.0)
        mine = self.own_tokens(u) if cost > 0 else None
        if not mine:
            return 1.0
        key = (self.update, sum(len(x.tok_obs or []) for x in self.units.values()))
        if getattr(self, "_pool_tok", None) is None or self._pool_tok[0] != key:
            pool = sorted(t for t in (self.own_tokens(x) for x in self.units.values()) if t)
            self._pool_tok = (key, pool[len(pool) // 2] if pool else None)
        mid = self._pool_tok[1]
        return (float(mid) / float(mine)) ** cost if mid else 1.0


    def score(self, u, rng):
        cfg = self.cfg
        if cfg.get("score") == "rhae_own":
            mu0, sd0, games = self.own_prior()
            obs = [x for _, x in u.sdz[-int(cfg["lp_window"]):]]
            n0 = float(cfg["own_prior_n"])
            mu = (sum(obs) + n0 * games.get(u.game, mu0)) / (len(obs) + n0)
            draw = max(0.0, rng.gauss(mu, sd0 / math.sqrt(len(obs) + n0)))
            tf = self.token_factor(u) if hasattr(self, "token_factor") else 1.0  # envs (token_cost; 1.0 when off)
            s = tf * draw / max(mu0, 1e-3) + cfg["progress"] * u.progress_z(cfg["lp_window"])
            if u.stall >= cfg["stall_after"]:
                s -= cfg["stall_penalty"]
            return s
        if cfg.get("score") == "rhae_tok":
            g = int(cfg["group_size"])
            p = rng.betavariate(u.a, u.b)
            mu, sd, n = u.z_moments(float(cfg["z_prior_mu"]), float(cfg["z_prior_sigma"]), float(cfg["z_prior_n"]))
            mu = max(0.0, rng.gauss(mu, sd / math.sqrt(n)))  # Thompson on the clear mean too: thin levels explore
            ref = spread_z(0.5, g, float(cfg["z_prior_mu"]), float(cfg["z_prior_sigma"]))
            s = spread_z(p, g, mu, sd) / ref * self.token_factor(u) + cfg["progress"] * u.progress_z(cfg["lp_window"])
            if u.stall >= cfg["stall_after"]:
                s -= cfg["stall_penalty"]
            return s
        _, sig = self.predict(u, rng.betavariate(u.a, u.b))
        top = (math.log(float(cfg["log_clip"])) - self.fail_reward) / 2.0  # a 50/50 group at the clip
        s = sig / top * self.token_factor(u) + cfg["progress"] * u.progress()
        if u.stall >= cfg["stall_after"]:
            s -= cfg["stall_penalty"]
        return s

    def quotas(self, n, sources, rng=None):
        """Groups per source by the shares; the remainder by largest fraction, or (``rng``: small per-depth plans)
        drawn in proportion to the fractions, so the mix holds on average over updates."""
        shares = {s: float(self.shares.get(s, 0.0)) for s in sources}
        total = sum(shares.values())
        if total <= 0:
            shares, total = {s: 1.0 for s in sources}, float(len(sources))
        raw = {s: n * v / total for s, v in shares.items()}
        out = {s: int(math.floor(v)) for s, v in raw.items()}
        left = n - sum(out.values())
        if rng is None:
            for s in sorted(raw, key=lambda s: -(raw[s] - out[s]))[:left]:
                out[s] += 1
            return out
        frac = {s: raw[s] - out[s] for s in raw}
        for _ in range(left):
            pool = [s for s in frac if frac[s] > 0]
            if not pool:
                break
            x, acc = rng.random() * sum(frac[s] for s in pool), 0.0
            for s in pool:
                acc += frac[s]
                if acc >= x:
                    out[s] += 1
                    frac[s] = 0.0
                    break
        return out

    def eligible(self, u, probe=False):
        if not u.starts or u.key in self.blocked:
            return False
        if probe:
            return u.seen == 0 and u.state == "active" or (
                u.state == "parked" and self.update - (u.parked_at or 0) >= self.cfg["reprobe_every"])
        return u.state == "active"

    def next_start(self, u):
        s = u.starts[u.used % len(u.starts)]  # no repeat until the unit's pool is spent
        u.used += 1
        return s

    def plan(self, update, seed=0, exclude=(), limit=None, groups=None, salt=""):
        """The groups for ``update``: [{unit, game, level, source, family, start, tries, why}]. ``exclude``: unit keys
        that cannot be played this round (no start available); ``limit``: {unit key: most groups it can supply}
        (starts left); ``groups``: plan this many groups instead of the configured number (a top-up after a unit
        could not supply its groups; ``salt`` varies its draws)."""
        cfg = self.cfg
        self.update = int(update)
        rng = random.Random(f"{seed}:{update}{salt}")
        n, g = int(cfg["groups"] if groups is None else groups), int(cfg["group_size"])
        self.blocked = set(exclude or ())
        limit = dict(limit or {})
        try:
            if not cfg.get("depth_bins"):
                return self._plan(n, g, rng, limit)
            return self._plan_depths(n, g, rng, limit)
        finally:
            self.blocked = set()

    def depth(self, u):
        bins = sorted(int(b) for b in self.cfg["depth_bins"])
        return max([b for b in bins if u.level >= b] or [bins[0]])

    def _plan_depths(self, n, g, rng, limit):
        """Equal groups per depth bin among the bins with a playable unit (remainder to random bins); inside a bin the
        usual plan (probes, source quotas, frontier score); family caps and the token cap count over the whole update;
        a bin that cannot fill its groups leaves them to a last pass over every unit."""
        base = set(self.blocked)
        units = list(self.units.values())
        bins = {}
        for u in units:
            if self.eligible(u) or self.eligible(u, True):
                bins.setdefault(self.depth(u), []).append(u.key)
        if not bins:
            return []
        order = sorted(bins)
        per = {b: n // len(order) for b in order}
        for b in rng.sample(order, n - sum(per.values())):
            per[b] += 1
        shared = {"fam_count": {}, "tokens_used": 0, "taken": set(), "n": n}
        picks = []
        for b in order:
            keys = set(bins[b])
            self.blocked = base | {k for k in self.units if k not in keys}
            picks += self._plan(per[b], g, rng, limit, shared=shared, log=False)
        if len(picks) < n:
            self.blocked = base | {k for k in shared["taken"] if k in limit and limit[k] <= 0}
            picks += self._plan(n - len(picks), g, rng, limit, shared=shared, log=False)
        self.blocked = base
        self.log.append({"update": self.update, "picks": [(p["unit"], p["why"]) for p in picks],
                         "depths": {str(b): per[b] for b in order}})
        return picks

    def _plan(self, n, g, rng, limit, shared=None, log=True):
        cfg = self.cfg
        total = int(shared["n"]) if shared else n  # caps count over the whole update
        fam_cap = max(1, int(math.floor(cfg["family_cap"] * total + 1e-9)))
        tok_cap = int(math.floor(cfg["token_share"] * total + 1e-9))
        n_probe = int(round(cfg["probe_share"] * n)) if not shared else int(rng.random() < cfg["probe_share"] * n)
        units = list(self.units.values())
        sources = sorted({u.source for u in units if self.eligible(u) or self.eligible(u, True)})
        picks = []
        if shared is None:
            shared = {"fam_count": {}, "tokens_used": 0, "taken": set(), "n": n}
        fam_count, taken = shared["fam_count"], shared["taken"]
        tokens_used = shared["tokens_used"]

        def take(u, why):
            nonlocal tokens_used
            shared["tokens_used"] = tokens_used + int(u.mean > cfg["token_p"])
            if u.key in limit:
                limit[u.key] -= 1
                if limit[u.key] <= 0:
                    self.blocked.add(u.key)  # its starts are spent for this round
            picks.append({"unit": u.key, "game": u.game, "level": u.level, "source": u.source, "family": u.family,
                          "start": self.next_start(u), "tries": g, "why": why, "p": round(u.mean, 3),
                          "cap": u.cap, "tier": u.tier})
            fam_count[u.family] = fam_count.get(u.family, 0) + 1
            tokens_used += int(u.mean > cfg["token_p"])
            taken.add(u.key)
            u.picked += 1

        probes = [u for u in units if self.eligible(u, True)]
        rng.shuffle(probes)
        for u in probes:
            if len(picks) >= n_probe:
                break
            if fam_count.get(u.family, 0) < fam_cap:
                take(u, "probe")
        quota = self.quotas(n - len(picks), sources, rng if total != n else None)
        for src in sources:
            scored = [(self.score(u, rng), u) for u in units
                      if u.source == src and self.eligible(u) and u.key not in taken]
            best = {}
            for s, u in scored:
                if u.family not in best or s > best[u.family][0]:
                    best[u.family] = (s, u)
            ranked = sorted(best.values(), key=lambda t: -t[0])
            got, i = 0, 0
            while got < quota[src] and ranked:
                s, u = ranked[i % len(ranked)]
                ok = fam_count.get(u.family, 0) < fam_cap and (u.mean <= cfg["token_p"] or tokens_used < tok_cap) \
                    and u.key not in taken
                if ok:
                    take(u, "frontier")
                    got += 1
                i += 1
                if i >= len(ranked):
                    # a second pass over families: the next-best unit of each family (several levels per family)
                    rest = sorted(((s2, u2) for s2, u2 in scored if u2.key not in taken), key=lambda t: -t[0])
                    ranked = [t for t in rest if fam_count.get(t[1].family, 0) < fam_cap
                              and (t[1].mean <= cfg["token_p"] or tokens_used < tok_cap)]
                    i = 0
                    if not ranked:
                        break
        # unfilled quota (a thin source): fill from any source by score, caps kept
        if len(picks) < n:
            rest = sorted(((self.score(u, rng), u) for u in units if self.eligible(u) and u.key not in taken),
                          key=lambda t: -t[0])
            for s, u in rest:
                if len(picks) >= n:
                    break
                if fam_count.get(u.family, 0) < fam_cap and (u.mean <= cfg["token_p"] or tokens_used < tok_cap):
                    take(u, "fill")
        # a small pool (fewer eligible units or families than groups): units take several groups, each group a fresh
        # Thompson draw; a unit holds at most max(ceil(groups / units), unit_share x groups), a family at most that or
        # ceil(groups / families)
        live = [u for u in units if self.eligible(u)]
        if len(picks) < n and live and cfg["repeat"]:
            unit_cap = max(math.ceil(n / len(live)), math.ceil(cfg["unit_share"] * n))
            fam_cap2 = max(fam_cap, unit_cap, math.ceil(n / len({u.family for u in live})))
            count = {}
            for p_ in picks:
                count[p_["unit"]] = count.get(p_["unit"], 0) + 1
            while len(picks) < n:
                ok = [u for u in live if count.get(u.key, 0) < unit_cap and fam_count.get(u.family, 0) < fam_cap2
                      and u.key not in self.blocked]
                if not ok:
                    break
                capped = [u for u in ok if u.mean <= cfg["token_p"] or tokens_used < tok_cap] or ok
                s_, u = max(((self.score(u, rng), u) for u in capped), key=lambda t: t[0])
                take(u, "repeat")
                count[u.key] = count.get(u.key, 0) + 1
        if log:
            self.log.append({"update": self.update, "picks": [(p["unit"], p["why"]) for p in picks]})
        return picks

    # ---- v7: one group at a time ---------------------------------------------------------------------------
    def plan_one(self, update, seed=0, exclude=(), salt=""):
        """The next group for the v7 loop (groups made one at a time), or None when no unit can be played.

        - Source: the largest deficit against the shares over every group planned so far (``plan(groups=1)``'s
          largest remainder would give every group to the largest share); sources without a playable unit skip.
        - Unit: with chance ``probe_share`` a probe (unseen, or parked and due); else the best frontier score (spread +
          learning progress - stall), one draw each. Family cap and near-solved cap count over the last ``window``
          groups; when every unit of the source is over a cap, the best unit is taken anyway (never a stall).
        - ``exclude``: unit keys with no start available now (or inactive games). The start is the caller's.
        """
        cfg = self.cfg
        self.update = int(update)
        rng = random.Random(f"{seed}:{update}:{sum(self.made.values())}{salt}")
        blocked = set(exclude or ())
        live = [u for u in self.units.values() if u.key not in blocked and u.starts]
        play = [u for u in live if self.eligible(u) or self.eligible(u, True)]
        if not play:
            return None
        sources = sorted({u.source for u in play})
        total = sum(self.made.values()) + 1
        shares = {s: float(self.shares.get(s, 0.0)) for s in sources}
        if sum(shares.values()) <= 0:
            shares = {s: 1.0 for s in sources}
        norm = sum(shares.values())
        order = sorted(sources, key=lambda s: (-(total * shares[s] / norm - self.made.get(s, 0)), -shares[s], s))
        win = self.recent[-int(cfg["window"]):]
        fam_cap = max(1, int(math.floor(cfg["family_cap"] * int(cfg["window"]) + 1e-9)))
        tok_cap = int(math.floor(cfg["token_share"] * int(cfg["window"]) + 1e-9))
        fams, near = {}, sum(1 for x in win if x[2])
        for x in win:
            fams[x[1]] = fams.get(x[1], 0) + 1

        def ok(u):
            return fams.get(u.family, 0) < fam_cap and (u.mean <= cfg["token_p"] or near < tok_cap)

        deep_lv, deep_sh = int(cfg.get("deep_level") or 3), float(cfg.get("deep_share") or 0.0)
        deep_due = deep_sh > 0 and sum(1 for x in win if x[0] in self.units and self.units[x[0]].level >= deep_lv) < deep_sh * max(1, len(win))
        for src in order:
            mine = [u for u in play if u.source == src]
            if deep_due and any(u.level >= deep_lv and self.eligible(u) for u in mine):
                mine = [u for u in mine if u.level >= deep_lv]
            pick, why = None, None
            probes = [u for u in mine if self.eligible(u, True)]
            if probes and rng.random() < float(cfg["probe_share"]):
                rng.shuffle(probes)
                pick = next((u for u in probes if ok(u)), None)
                why = "probe" if pick is not None else None
            if pick is None:
                scored = sorted(((self.score(u, rng), u.key, u) for u in mine if self.eligible(u)), key=lambda t: (-t[0], t[1]))
                if not scored:
                    scored = [(0.0, u.key, u) for u in probes]
                pick = next((u for _, _, u in scored if ok(u)), None)
                why = "frontier"
                if pick is None and scored:
                    pick, why = scored[0][2], "over_cap"
            if pick is None:
                continue
            self.made[src] = self.made.get(src, 0) + 1
            self.recent = (self.recent + [[pick.key, pick.family, bool(pick.mean > cfg["token_p"])]])[-int(cfg["window"]):]
            pick.picked += 1
            return {"unit": pick.key, "game": pick.game, "level": pick.level, "source": pick.source,
                    "family": pick.family, "tries": int(cfg["group_size"]), "why": why, "p": round(pick.mean, 3),
                    "cap": pick.cap, "tier": pick.tier}
        return None

    def sure(self, u):
        """Near-certain: posterior clear mean >= chain_sure_p and no token trend (progress_z < chain_flat)."""
        cfg = self.cfg
        return u.mean >= float(cfg["chain_sure_p"]) and u.progress_z(cfg["lp_window"]) < float(cfg["chain_flat"])

    def plan_chain(self, update, cands, seed=0, sure_ok=True, salt=""):
        """A chain group: ``cands`` {unit key: {"deepest": bool}} are open chain levels (a fresh
        next-level start inside the game's budget), at most one per game (the caller drops games that already have a
        chain group this step). Picks the best rhae_tok score (+ ``chain_bonus`` on a game's deepest open level) under
        the family cap over the last ``window`` groups; a near-certain level only while ``sure_ok``. None when nothing
        qualifies (the group goes to ``plan_one``). Counts toward the source mix like any group."""
        cfg = self.cfg
        self.update = int(update)
        rng = random.Random(f"chain:{seed}:{update}:{sum(self.made.values())}{salt}")
        win = self.recent[-int(cfg["window"]):]
        fam_cap = max(1, int(math.floor(cfg["family_cap"] * int(cfg["window"]) + 1e-9)))
        fams = {}
        for x in win:
            fams[x[1]] = fams.get(x[1], 0) + 1
        scored = []
        for key, info in sorted(cands.items()):
            u = self.units.get(key)
            if u is None or not u.starts or not (self.eligible(u) or self.eligible(u, True)):
                continue
            if fams.get(u.family, 0) >= fam_cap:
                continue
            sure = self.sure(u)
            if sure and not sure_ok:
                continue
            s = self.score(u, rng) + (float(cfg["chain_bonus"]) if info.get("deepest") else 0.0)
            scored.append((s, key, u, sure))
        if not scored:
            return None
        s, _, u, sure = max(scored, key=lambda t: (t[0], t[1]))
        self.made[u.source] = self.made.get(u.source, 0) + 1
        self.recent = (self.recent + [[u.key, u.family, bool(u.mean > cfg["token_p"])]])[-int(cfg["window"]):]
        u.picked += 1
        return {"unit": u.key, "game": u.game, "level": u.level, "source": u.source, "family": u.family,
                "tries": int(cfg["group_size"]), "why": "chain", "p": round(u.mean, 3), "cap": u.cap, "tier": u.tier,
                "sure": bool(sure), "score": round(s, 3)}

    def unplan(self, pick):
        """A planned group that could not be made (its start was taken meanwhile): its source count and window entry
        are returned."""
        src = pick["source"]
        self.made[src] = max(0, self.made.get(src, 0) - 1)
        for i in range(len(self.recent) - 1, -1, -1):
            if self.recent[i][0] == pick["unit"]:
                del self.recent[i]
                break

    # ---- persistence ----------------------------------------------------------------------------------------
    def state_dict(self):
        return {"cfg": self.cfg, "shares": self.shares, "update": self.update, "made": dict(self.made),
                "recent": [list(x) for x in self.recent],
                "units": {k: {s: getattr(u, s) for s in Unit.__slots__} for k, u in self.units.items()}}

    @classmethod
    def load_state(cls, data):
        c = cls(data["cfg"], data["shares"])
        c.update = int(data["update"])
        c.made = {k: int(v) for k, v in (data.get("made") or {}).items()}
        c.recent = [list(x) for x in data.get("recent") or []]
        for k, d in data["units"].items():
            u = Unit(k, d["game"], d["level"], d["source"], d["family"])
            for s in Unit.__slots__:
                setattr(u, s, d[s] if s in d else Unit.OPTIONAL[s])
            c.units[k] = u
        return c

    def summary(self):
        by = {}
        for u in self.units.values():
            row = by.setdefault(u.source, {"units": 0, "active": 0, "parked": 0, "retired": 0, "seen": 0})
            row["units"] += 1
            row[u.state] += 1
            row["seen"] += int(u.seen > 0)
        return by
