"""Within-level (conditional) clear-hazard shape by token band (27 September).

The pooled clear rate per token band (``schedule_sim.World`` before this, ``arena.gittins.H0_FIRST``/
``H0_DEEP``) falls after 40K tokens, but within the same (game, level) the clear rate rises with tokens
spent: the falling pooled curve is selection (easy levels clear early and leave the risk set; hard ones
stay). A model that multiplies the pooled shape by a per-game or per-level factor counts that
heterogeneity twice. This script fits the shape *inside* a cell, with the between-cell heterogeneity
carried by frailties, by marginal likelihood:

- ``cell``: per class (first level / deeper levels), the hazard on (game, level) after ``t`` tokens is
  ``z x h0(t)``, ``z ~ Gamma(a, a)`` shared by every run's attempt on that cell (closed form).
- ``hier``: the ``arena.gittins`` model itself, fitted jointly: ``h0_class(t) x m_g^p x e_L`` with
  ``m_g ~ Gamma(k, k)`` per game (``p`` = 1 on level 1, ``beta`` deeper; integrated on a log grid) and
  ``e_L ~ Gamma(ke, ke)`` per (game, level) (closed form). Its ``h0`` shapes and ``k``, ``ke``, ``beta``
  are pooled over all public games: generic constants, no per-game values.

``h0`` is piecewise constant on the bands of ``EDGES`` with a weak random-walk prior on log h between
adjacent bands (SD 1), so sparse tails follow their neighbours. Uncertainty: a game-level (cluster)
bootstrap. A diagnostic repeats the observed/expected-clears table by token band inside cells.

    PYTHONPATH=src uv run python scripts/arena_analysis/within_level_hazard.py --attempts <level_attempts.jsonl> \
        [--boot 100] [--out fit.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schedule_sim import EDGES, band_exposure, load_rows  # noqa: E402

NB = len(EDGES) - 1
PRIOR_RATE = 3e-5  # the pseudo-clear's rate (0.3 per 10K), as in schedule_sim.World
GRID = np.exp(np.linspace(math.log(1e-4), math.log(100.0), 240))
LOG_STEP = float(np.log(GRID[1] / GRID[0]))


def cells_from_rows(rows):
    """Per (game, level): class, clears, band exposure and the clears per band."""
    cells = {}
    for r in rows:
        if r["cleared"] and r["tokens"] < 500:
            continue  # a level cleared in the same observation (or nearly) as the previous one
        key = (r["game"], r["level"])
        c = cells.setdefault(key, {"game": r["game"], "level": r["level"], "deep": r["level"] > 1,
                                   "d": 0, "x": np.zeros(NB), "ev": np.zeros(NB), "n": 0})
        c["n"] += 1
        c["x"] += band_exposure(r["tokens"])
        if r["cleared"]:
            c["d"] += 1
            c["ev"][np.searchsorted(EDGES, r["tokens"], side="right") - 1] += 1
    return list(cells.values())


class Data:
    def __init__(self, cells):
        self.cells = cells
        self.deep = np.array([c["deep"] for c in cells])
        self.d = np.array([c["d"] for c in cells], dtype=float)
        self.x = np.array([c["x"] for c in cells])
        self.ev = np.array([c["ev"] for c in cells])
        games = sorted({c["game"] for c in cells})
        self.games = games
        self.gi = np.array([games.index(c["game"]) for c in cells])
        self.ev_first = self.ev[~self.deep].sum(axis=0)
        self.ev_deep = self.ev[self.deep].sum(axis=0)


RW_SD = 1.0  # random-walk prior on log h between adjacent bands (a factor of e per band is one SD)


def _band_prior(h):
    """Smoothness prior: a Gaussian random walk on log h across bands, so a band without data (level 1
    past 80K) takes its neighbour's value instead of an arbitrary constant."""
    step = np.diff(np.log(h))
    return float(-0.5 * np.sum((step / RW_SD) ** 2))


def nll_cell(theta, data):
    """Per-class cell frailty: theta = log h_first (NB), log h_deep (NB), log a_first, log a_deep."""
    hf, hd = np.exp(theta[:NB]), np.exp(theta[NB:2 * NB])
    af, ad = np.exp(theta[2 * NB]), np.exp(theta[2 * NB + 1])
    h = np.where(data.deep[:, None], hd[None, :], hf[None, :])
    a = np.where(data.deep, ad, af)
    s = (data.x * h).sum(axis=1)
    ll = (gammaln(a + data.d) - gammaln(a) + a * np.log(a) - (a + data.d) * np.log(a + s)).sum()
    ll += data.ev_first @ np.log(hf) + data.ev_deep @ np.log(hd)
    ll += _band_prior(hf) + _band_prior(hd)
    return -ll


def hier_terms(theta, data):
    hf, hd = np.exp(theta[:NB]), np.exp(theta[NB:2 * NB])
    k, ke = np.exp(theta[2 * NB]), np.exp(theta[2 * NB + 1])
    beta = theta[2 * NB + 2]
    return hf, hd, k, ke, beta


def game_log_lik(theta, data):
    """[games x grid] log p(data of the game's cells | m_g) (without the band event terms) and the prior."""
    hf, hd, k, ke, beta = hier_terms(theta, data)
    h = np.where(data.deep[:, None], hd[None, :], hf[None, :])
    s = (data.x * h).sum(axis=1)  # cumulative baseline hazard per cell
    p = np.where(data.deep, beta, 1.0)
    mm = GRID[None, :] ** p[:, None]  # cells x grid
    per_cell = (data.d[:, None] * np.log(mm) + gammaln(ke + data.d)[:, None] - gammaln(ke)
                + ke * np.log(ke / (ke + mm * s[:, None])) - data.d[:, None] * np.log(ke + mm * s[:, None]))
    lg = np.zeros((len(data.games), len(GRID)))
    np.add.at(lg, data.gi, per_cell)
    log_prior = k * np.log(k) - gammaln(k) + k * np.log(GRID) - k * GRID + math.log(LOG_STEP)
    return lg, log_prior


def nll_hier(theta, data):
    hf, hd, k, ke, beta = hier_terms(theta, data)
    if not 0.0 < beta < 2.0:
        return 1e12
    lg, log_prior = game_log_lik(theta, data)
    z = lg + log_prior[None, :]
    zmax = z.max(axis=1, keepdims=True)
    ll = float((zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))).sum())
    ll += data.ev_first @ np.log(hf) + data.ev_deep @ np.log(hd)
    ll += _band_prior(hf) + _band_prior(hd)
    # Weak priors so k and ke stay inside the index's grid range in sparse bootstrap samples.
    ll += -0.5 * (math.log(k) / 2.0) ** 2 - 0.5 * (math.log(ke) / 2.0) ** 2
    return -ll


def pooled_shape(data):
    """The pooled (marginal) rate per band: what schedule_sim.World / H0_FIRST/H0_DEEP use."""
    out = {}
    for name, sel in (("first", ~data.deep), ("deep", data.deep)):
        out[name] = (data.ev[sel].sum(axis=0) + 0.5) / (data.x[sel].sum(axis=0) + 0.5 / PRIOR_RATE)
    return out


def fit(rows, model="hier", start=None):
    data = Data(cells_from_rows(rows))
    pooled = pooled_shape(data)
    if model == "cell":
        theta0 = np.concatenate([np.log(pooled["first"]), np.log(pooled["deep"]), [0.0, 0.0]])
        fun = nll_cell
    else:
        theta0 = np.concatenate([np.log(pooled["first"]), np.log(pooled["deep"]), [math.log(0.25), math.log(1.5), 0.5]])
        fun = nll_hier
    if start is not None:
        theta0 = np.array(start, dtype=float)
    res = minimize(fun, theta0, args=(data,), method="L-BFGS-B", options={"maxiter": 4000, "maxfun": 40000})
    t = res.x
    out = {"model": model, "nll": float(res.fun), "converged": bool(res.success), "theta": t.tolist(),
           "first": np.exp(t[:NB]).tolist(), "deep": np.exp(t[NB:2 * NB]).tolist(),
           "pooled_first": pooled["first"].tolist(), "pooled_deep": pooled["deep"].tolist(),
           "cells": len(data.cells), "games": len(data.games),
           "attempts": int(sum(c["n"] for c in data.cells)), "clears": int(data.d.sum())}
    if model == "cell":
        out["a_first"], out["a_deep"] = float(np.exp(t[2 * NB])), float(np.exp(t[2 * NB + 1]))
    else:
        out["k"], out["ke"], out["beta"] = float(np.exp(t[2 * NB])), float(np.exp(t[2 * NB + 1])), float(t[2 * NB + 2])
    return out


def game_posteriors(theta, rows):
    """Per game: normalised posterior weights of m_g on ``GRID`` and per level (D, cumulative baseline hazard),
    under the ``hier`` fit (the corrected simulator world)."""
    data = Data(cells_from_rows(rows))
    hf, hd, k, ke, beta = hier_terms(np.asarray(theta), data)
    lg, log_prior = game_log_lik(np.asarray(theta), data)
    z = lg + log_prior[None, :]
    w = np.exp(z - z.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    h = np.where(data.deep[:, None], hd[None, :], hf[None, :])
    s = (data.x * h).sum(axis=1)
    post = {g: {"weights": w[i], "levels": {}} for i, g in enumerate(data.games)}
    for c, d, si in zip(data.cells, data.d, s):
        post[c["game"]]["levels"][c["level"]] = (float(d), float(si))
    return post


def observed_expected(rows, min_attempts=6, min_clears=2):
    """Observed / expected clears per token band inside cells, each cell at its own constant rate."""
    cells = cells_from_rows(rows)
    obs, exp = np.zeros(NB), np.zeros(NB)
    used = 0
    for c in cells:
        if c["n"] < min_attempts or c["d"] < min_clears:
            continue
        used += 1
        rate = c["d"] / c["x"].sum()
        obs += c["ev"]
        exp += rate * c["x"]
    return used, obs, exp


def boot(rows, model, reps, seed, start):
    rng = np.random.default_rng(seed)
    games = sorted({r["game"] for r in rows})
    by_game = {g: [r for r in rows if r["game"] == g] for g in games}
    draws = []
    for _ in range(reps):
        pick = rng.choice(len(games), size=len(games), replace=True)
        rs = []
        for j, i in enumerate(pick):
            # Relabel resampled games so duplicates are separate games (and cells).
            rs.extend({**r, "game": f"{games[i]}~{j}"} for r in by_game[games[i]])
        draws.append(fit(rs, model, start=start))
    return draws


def fmt(v):
    return " ".join(f"{x * 1e4:6.3f}" for x in v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--boot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = load_rows(args.attempts)
    labels = ["0-5K", "5-10K", "10-20K", "20-40K", "40-80K", "80-160K", "160K+"]
    result = {}
    print("clears per 10K tokens by band:", " ".join(f"{b:>6s}" for b in labels))
    for model in ("cell", "hier"):
        f = fit(rows, model)
        result[model] = f
        if args.boot:
            draws = boot(rows, model, args.boot, args.seed, f["theta"])
            for key in ("first", "deep"):
                arr = np.array([d[key] for d in draws])
                f[f"{key}_ci"] = np.percentile(arr, [5, 95], axis=0).tolist()
                f[f"{key}_rel_ci"] = np.percentile(arr / arr[:, 2:3], [5, 95], axis=0).tolist()
            scal = ["a_first", "a_deep"] if model == "cell" else ["k", "ke", "beta"]
            for key in scal:
                f[f"{key}_ci"] = np.percentile([d[key] for d in draws], [5, 95]).tolist()
        print(f"\n[{model}] cells {f['cells']} games {f['games']} attempts {f['attempts']} clears {f['clears']} "
              f"converged {f['converged']} nll {f['nll']:.1f}")
        print("  pooled first ", fmt(f["pooled_first"]))
        print("  within first ", fmt(f["first"]))
        if "first_ci" in f:
            print("     5%        ", fmt(f["first_ci"][0]))
            print("    95%        ", fmt(f["first_ci"][1]))
        print("  pooled deep  ", fmt(f["pooled_deep"]))
        print("  within deep  ", fmt(f["deep"]))
        if "deep_ci" in f:
            print("     5%        ", fmt(f["deep_ci"][0]))
            print("    95%        ", fmt(f["deep_ci"][1]))
        extra = {k: v for k, v in f.items() if k in ("a_first", "a_deep", "k", "ke", "beta") or k.endswith("_ci")
                 and not k.startswith(("first", "deep"))}
        print("  ", json.dumps({k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v])
                                for k, v in extra.items()}))
    used, obs, exp = observed_expected(rows)
    print(f"\nobserved/expected clears inside {used} cells (>= 6 attempts, >= 2 clears):",
          " ".join(f"{b}:{o / e:.2f}" if e > 0 else f"{b}:-" for b, o, e in zip(labels, obs, exp)))
    result["observed_expected"] = {"cells": used, "observed": obs.tolist(), "expected": exp.tolist()}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
