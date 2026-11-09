"""Gen 0's selection policy (docs/rsi.md 1 and 3). The proposer owns a copy (``meta/policy.py``) and may rewrite it;
the root runs that copy in the proposer's sandbox through ``decide`` (stdlib only, JSON in, JSON out) and falls back
to this file when it fails.

Pool score: per (candidate, game) the official RHAE game score of the attempt (reference solves as the baselines).
Candidates play different random minibatches, so a candidate's standing is its effect in an additive model
score = mu + candidate + game, fitted by ridge-penalised alternating least squares (candidates with few games shrink
toward the mean). Parent choice (HGM-style): Thompson sampling of each candidate's clade productivity, where every
evaluated clade member adds q = P(member beats gen 0) to successes and 1 - q to failures."""

import math
import random

POLICY_DEFAULTS = {
    "minibatch": 15,          # pool games per candidate evaluation (random, unplayed by that candidate)
    "promote_top": 3,         # candidates in the top this many by effect get another minibatch, up to the full pool
    "parent_rule": "hgm",     # hgm | best | uniform
    "practice_per_eval": 2,   # practice games played with each evaluation (full traces for the proposer)
    "ridge": 2.0,             # game-equivalents of shrinkage toward the mean effect
    "anchor_games": 30,       # gen 0 is promoted until it has this many pool games (the comparison anchor)
    "paired": False,          # owner switch (third audit): every candidate's minibatches follow one fixed game order,
                              # so candidates are compared on the same games (paired); False: a random draw each
}
POLICY_RANGES = {"minibatch": (1, 1000), "promote_top": (0, 1000), "practice_per_eval": (0, 100),
                 "ridge": (0.0, 1000.0), "anchor_games": (0, 100000)}
PARENT_RULES = ("hgm", "best", "uniform")


def check_policy(policy):
    merged = {**POLICY_DEFAULTS, **(policy or {})}
    if set(merged) - set(POLICY_DEFAULTS):
        raise ValueError(f"policy keys: {sorted(POLICY_DEFAULTS)}")
    if not isinstance(merged["paired"], bool):
        raise ValueError("policy.paired must be true or false")
    for key, (low, high) in POLICY_RANGES.items():
        value = merged[key]
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f"policy.{key} must be in [{low}, {high}]")
        if type(POLICY_DEFAULTS[key]) is int:
            if int(value) != value:
                raise ValueError(f"policy.{key} must be an integer")
            merged[key] = int(value)
    if merged["parent_rule"] not in PARENT_RULES:
        raise ValueError(f"policy.parent_rule must be one of {PARENT_RULES}")
    return merged


def effects(candidates, ridge=2.0, iterations=50):
    """``({candidate: effect}, {game: effect}, mu)`` from every pool score of every candidate."""
    candidates = {c: v for c, v in candidates.items() if v.get("status") != "failed"}  # failed: no evidence
    rows = [(c, g, s) for c, item in candidates.items() for g, s in (item.get("pool") or {}).items()]
    if not rows:
        return {}, {}, 0.0
    mu = sum(s for _, _, s in rows) / len(rows)
    a = {c: 0.0 for c, _, _ in rows}
    b = {g: 0.0 for _, g, _ in rows}
    for _ in range(iterations):
        for g in b:
            pairs = [(c, s) for c, gg, s in rows if gg == g]
            b[g] = sum(s - mu - a[c] for c, s in pairs) / len(pairs)
        for c in a:
            pairs = [(g, s) for cc, g, s in rows if cc == c]
            a[c] = sum(s - mu - b[g] for g, s in pairs) / (len(pairs) + ridge)
    return a, b, mu


def standing(candidates, ridge=2.0):
    """Per candidate: effect, games, residual sd and a standard error (sd / sqrt(n))."""
    candidates = {c: v for c, v in candidates.items() if v.get("status") != "failed"}
    a, b, mu = effects(candidates, ridge)
    out = {}
    residuals = [s - mu - a[c] - b[g] for c, item in candidates.items() for g, s in (item.get("pool") or {}).items()]
    sd = math.sqrt(sum(r * r for r in residuals) / max(1, len(residuals) - 1)) if len(residuals) > 1 else 20.0
    for c, item in candidates.items():
        n = len(item.get("pool") or {})
        out[c] = {"effect": round(a.get(c, 0.0), 3), "games": n, "se": round(sd / math.sqrt(n), 3) if n else None,
                  "raw_mean": round(sum(item["pool"].values()) / n, 3) if n else None}
    return out


def descendants(candidates, root):
    children = {}
    for c, item in candidates.items():
        if isinstance(item.get("parent"), str):
            children.setdefault(item["parent"], []).append(c)
    out, stack, seen = [], [root], set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        stack += children.get(node, [])
    return out


def choose_parent(candidates, policy, rng, root="c0000"):
    """A parent among the evaluated, not failed candidates (the policy's rule)."""
    ready = [c for c, item in candidates.items() if item.get("pool") and item.get("status") != "failed"]
    if not ready:
        return root
    stand = standing({c: candidates[c] for c in candidates if candidates[c].get("pool")}, policy["ridge"])
    if policy["parent_rule"] == "uniform":
        return rng.choice(sorted(ready))
    if policy["parent_rule"] == "best":
        return max(sorted(ready), key=lambda c: stand[c]["effect"])
    base = stand.get(root, {"effect": 0.0})["effect"]
    best, pick = -1.0, root
    for c in sorted(ready):
        wins = losses = 0.0
        for d in descendants(candidates, c):
            if d not in stand or not stand[d]["games"]:
                continue
            se = stand[d]["se"] or 20.0
            q = 0.5 * (1 + math.erf((stand[d]["effect"] - base) / (se * math.sqrt(2))))
            wins, losses = wins + q, losses + 1 - q
        draw = rng.betavariate(1 + wins, 1 + losses)
        if draw > best:
            best, pick = draw, c
    return pick


def played(item):
    """Pool games a candidate has played: scored ones and the proposer's added games (levels only, no score)."""
    return set(item.get("pool") or {}) | set(item.get("pool_levels") or {})


PAIRED_SEED = 20261012


def minibatch(pool_games, played, size, rng, paired=False):
    """The next ``size`` pool games a candidate has not played: a random draw, or with ``paired`` the next ones in a
    fixed order of the pool (the same for every candidate; games added later go to the end)."""
    if paired:
        order = sorted(pool_games, key=lambda g: (random.Random(f"{PAIRED_SEED}:{g}").random(), g))
        return [g for g in order if g not in set(played)][:size]
    left = sorted(set(pool_games) - set(played))
    rng.shuffle(left)
    return left[:size]


def promotions(candidates, policy, pool_size, busy, root="c0000"):
    """Candidates owed another minibatch: gen 0 until it has ``anchor_games``; others when they are in the top
    ``promote_top`` by effect and above gen 0; never past the full pool or while one of theirs is queued."""
    stand = standing({c: v for c, v in candidates.items() if v.get("pool")}, policy["ridge"])
    done = [c for c in stand if candidates[c].get("status") == "done" and c not in busy]
    out = []
    if root in done and stand[root]["games"] < min(policy["anchor_games"], pool_size) and len(
            played(candidates[root])) < pool_size:
        out.append(root)
    base = stand.get(root, {"effect": 0.0})["effect"]
    ranked = sorted((c for c in stand if c != root and candidates[c].get("status") == "done"),
                    key=lambda c: -stand[c]["effect"])
    out += [c for c in ranked[:policy["promote_top"]] if c in done and stand[c]["effect"] > base
            and len(played(candidates[c])) < pool_size]
    return out


def practice_ranking(practice, games, rng):
    """Practice games by recent gain in clear rate (levels cleared / levels): unplayed first, then the newest three
    attempts' rate minus the earlier ones' (docs/rsi.md 3)."""
    def gain(game):
        rates = [r["levels"] / max(1, r["total_levels"]) for r in practice.get(game, [])]
        if not rates:
            return (1, rng.random())
        recent, earlier = rates[-3:], rates[:-3]
        return (0, (sum(recent) / len(recent)) - (sum(earlier) / len(earlier) if earlier else 0.0) + 1e-3 * rng.random())
    return sorted(games, key=gain, reverse=True)


def rng_state(rng):
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def rng_restore(state):
    rng = random.Random()
    rng.setstate((state[0], tuple(state[1]), state[2]))
    return rng


def decide(view):
    """One scheduling decision; the loop calls it after every submission and every finished smoke or pool game.

    ``view``: {"candidates": {id: {"status" (new | evaluating | done | failed), "parent", "pool": {game: mean score},
    "pool_runs": {game: [score, ...]}, "pool_levels": {game: levels}, "live": bool (its serving/ is the running stack),
    "stack_reverted": bool (its stack failed probation: never the held-out pick), "queued": int (pool games queued or
    running)}}, "pool_games": [...], "base_pool_games": [...] (the bundled ones),
    "practice": {game: [{"levels", "total_levels", "candidate"}]}, "practice_games": [...], "policy": {...} (loop
    overrides of POLICY_DEFAULTS, normally none), "seed": int}.

    Returns {"parent": id (suggested next fork), "evaluate": [{"candidate", "games"}] (pool games to queue, live
    candidates only, up to 200 per item; a game already played is played again and its scores averaged),
    "practice": [{"candidate", "games"}] (up to 20 per item), optional "cancel": [id] (drop that candidate's queued
    pool games: early stopping), optional "best": id (the candidate held-out evaluations should use; honoured when it
    has at least 15 bundled-pool scores and runs on the live stack, else gen 0's standing over the bundled games
    decides)}. Gen 0 sets neither "cancel" nor "best"."""
    policy = check_policy(view.get("policy"))
    rng = random.Random(view.get("seed", 0))
    cands = view["candidates"]
    pool = view["pool_games"]
    out = {"parent": choose_parent(cands, policy, rng), "evaluate": [], "practice": []}
    busy = {c for c, v in cands.items() if v.get("queued")}
    for c, v in sorted(cands.items()):
        if v.get("status") == "new" and v.get("live") and c not in busy:
            out["evaluate"].append({"candidate": c, "games": minibatch(pool, [], policy["minibatch"], rng,
                                                                       policy["paired"])})
            ranked = practice_ranking(view.get("practice") or {}, view.get("practice_games") or [], rng)
            if ranked and policy["practice_per_eval"]:
                out["practice"].append({"candidate": c, "games": ranked[:policy["practice_per_eval"]]})
    live = {c: v for c, v in cands.items() if v.get("live")}
    for c in promotions(live, policy, len(pool), busy):
        out["evaluate"].append({"candidate": c, "games": minibatch(pool, played(cands[c]), policy["minibatch"], rng,
                                                                   policy["paired"])})
    return out


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(decide(json.load(sys.stdin))))
