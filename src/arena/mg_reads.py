"""Per-game normalised held-out reads for the multi-game loop (envs, 22 Oct).

A pooled RHAE/10K over many sets is dominated by the cheap levels (sc25 22 Oct: L2 ~3.5 vs L1 ~0.24, L3 ~0.54), so a
multi-game read compares each set with the base on the same set and pools the log ratios with equal weight per game:

- per set, score = (sum RHAE + lam x rbar) / (sum tokens + lam x tbar) for the policy's tries and for the base's, with
  lam pseudo-tries at the base's pooled RHAE per try (rbar) and the set's base tokens per try (tbar), so a thin or
  clear-less set shrinks towards a ratio of 1 instead of 0 or infinity;
- per game, the mean log ratio over its sets; overall, the mean over games (``ratio`` = exp of it);
- 95% CI by a two-level bootstrap: games resampled, and inside each drawn set both arms' tries resampled;
- the same summary per level stratum (L1, L2, L3+) and per source, plus the mean clear-rate change per game.

A try is ``(rhae, tokens, cleared)``: the level's RHAE on a clear (0 on a fail) and the try's completion tokens.
"""

from __future__ import annotations

import math
import random
import statistics


def tries(rows, clear_rhae, clear_tokens):
    """Loop result rows -> [(rhae, tokens, cleared)]."""
    out = []
    for r in rows:
        ok = bool(clear_tokens(r))
        out.append((float(clear_rhae(r) or 0.0) if ok else 0.0, int(r.get("completion_tokens") or 0), ok))
    return out


def _score(ts, rbar, tbar, lam):
    return (sum(t[0] for t in ts) + lam * rbar) / max(1e-9, sum(t[1] for t in ts) + lam * tbar)


def _set_logs(cur, base, rbar, lam):
    out = {}
    for name, b in base.items():
        c = cur.get(name)
        if not c or not b:
            continue
        tbar = statistics.mean(t[1] for t in b) or 1.0
        out[name] = (math.log(_score(c, rbar, tbar, lam) / _score(b, rbar, tbar, lam)),
                     sum(t[2] for t in c) / len(c) - sum(t[2] for t in b) / len(b))
    return out


def _pool(logs, meta, names):
    per = {}
    for n in names:
        if n in logs:
            per.setdefault(meta[n]["game"], []).append(logs[n])
    if not per:
        return None
    g = [statistics.mean(x[0] for x in v) for v in per.values()]
    d = [statistics.mean(x[1] for x in v) for v in per.values()]
    return statistics.mean(g), statistics.mean(d), len(per)


def stratum(meta, name):
    lv = int(meta[name].get("level") or 1)
    return "L1" if lv == 1 else ("L2" if lv == 2 else "L3+")


def norm_read(cur, base, meta, lam=1.0, n_boot=1000, seed=0):
    """``cur`` / ``base``: {set: [(rhae, tokens, cleared)]}; ``meta``: {set: {game, level, source}}."""
    names = sorted(n for n in base if n in cur and cur[n] and base[n] and n in meta)
    if not names:
        return None
    every = [t for n in names for t in base[n]]
    rbar = statistics.mean(t[0] for t in every) or 1e-3
    logs = _set_logs(cur, base, rbar, lam)
    groups = {"all": names}
    for n in names:
        groups.setdefault(f"level:{stratum(meta, n)}", []).append(n)
        groups.setdefault(f"source:{meta[n].get('source') or 'sets'}", []).append(n)
    point = {k: _pool(logs, meta, v) for k, v in groups.items()}
    by_game = {}
    for n in names:
        by_game.setdefault(meta[n]["game"], []).append(n)
    games = sorted(by_game)
    rng = random.Random(seed)
    boots = {k: [] for k in groups}
    for _ in range(int(n_boot)):
        draw = [rng.choice(games) for _ in games]
        c2, b2, m2 = {}, {}, {}
        for i, g in enumerate(draw):
            for n in by_game[g]:
                key = f"{n}#{i}"
                c2[key] = [rng.choice(cur[n]) for _ in cur[n]]
                b2[key] = [rng.choice(base[n]) for _ in base[n]]
                m2[key] = {**meta[n], "game": f"{g}#{i}"}
        lg = _set_logs(c2, b2, rbar, lam)
        for k, v in groups.items():
            got = _pool(lg, m2, [f"{n}#{i}" for i, g in enumerate(draw) for n in by_game[g] if n in v])
            if got is not None:
                boots[k].append(got[0])
    out = {}
    for k, p in point.items():
        if p is None:
            continue
        b = sorted(boots[k])
        ci = [round(math.exp(b[int(0.025 * (len(b) - 1))]), 3), round(math.exp(b[int(0.975 * (len(b) - 1))]), 3)] if b else None
        out[k] = {"ratio": round(math.exp(p[0]), 3), "ci": ci, "clear_delta": round(p[1], 3), "games": p[2],
                  "sets": len(groups[k])}
    return out


def compact(ts):
    """A set's tries as stored in mg_state: [[rhae, tokens, cleared]]."""
    return [[round(float(r), 4), int(t), int(bool(c))] for r, t, c in ts]


def expand(rows):
    return [(float(r), int(t), bool(c)) for r, t, c in rows]
