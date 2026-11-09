"""Multi-game training pool -> loop sets (envs, 22 Oct; docs/envs-plan.md sections 5-7).

Input: the env registry (``data/envs/registry.json`` on ``envs``: game id, source, family, split, solution lengths) and
the difficulty tables (``probe-table-*.json``: per game and level the probes' reached / clears / clear tokens / median
actions), plus optional ``extra`` rows of the same shape (e.g. RL screens for the public games).

Output (``build``): ``sets`` for ``mg.sets`` (one training set per (game, level) unit, ``source`` = the mix group,
``family``, ``cap``, ``ref`` {tokens, actions, baseline} for the reward, ``prior`` {tries, clears, median_to_clear} for the
curriculum), ``shares`` for ``mg.curriculum.shares``, ``points`` (one level-1 point per game for ``rl1_points.json``;
``board_hash`` filled from ``hashes``) and a ``report``.

Rules:
- training splits only (``train``, ``public_train``, ``dev_b``: D-decisions train DEV-B); the
  held-out 10 and every ``excluded`` or ``he`` game are refused (a held-out game in ``sets`` raises);
- per game, levels 1..k where k is the deepest level whose kind is band or near (``kinds``), walking up from L1 and
  stopping at the first level that is low or thin (a harvest chain needs every level below the target to clear);
  easy levels inside the chain stay as feeders (the curriculum caps near-solved units at ``token_share``);
- cap = clamp(cap_factor x p80 clear tokens, cap_min, cap_max) (envs-plan D6: p80; x1.25 as the screen bank);
- reward refs: tokens = median clear tokens, actions = median clear actions (v7.1 probes: v8.5p harness tokens
  geo 1.035x, Spearman 0.91 on 40 L1 games, 22 Oct), baseline = ceil(baseline_factor x solution length) (plan 7.4:
  solver optimum x 1.3) or the median actions when the game ships no solution; ``measured`` True. A level no base try
  cleared gets ``measured: False`` and no tokens/actions (never the cap: a 64K placeholder T_b saturated RL's token
  term, 24 Oct); ``apply_refs`` fills such levels from a base-clears table (``data/envs/mg-refs-*.json``).
"""

from __future__ import annotations

import math
import statistics

HELD_OUT_10 = ("tr87", "r11l", "m0r0", "dc22", "s5i5", "su15", "sp80", "g50t", "lf52", "bp35")
TRAIN_SPLITS = ("train", "public_train", "dev_b")
# the mix (21 Oct D-decisions): public-15 30 / generated 25 / PuzzleScript 20 / sonpham+interactive 15 /
# Witness 5 (sums to 95: the curriculum's quotas normalise; the open 5 is a judgement call)
GROUP = {"public": "public", "generated": "generated", "writer": "generated", "puzzlescript": "puzzlescript",
         "sonpham": "sonpham_interactive", "interactive": "sonpham_interactive", "witness": "witness"}
# public 50%; the others' 25/20/15/5 with the open 5% to generated, scaled to the other half
SHARES = {"public": 0.50, "generated": 0.2143, "puzzlescript": 0.1429, "sonpham_interactive": 0.1071, "witness": 0.0357}


def kind(reached, clears, min_reached=3):
    """The M1 band rule: within-cap clear rate (0.8 x overall at a p80 cap) 0.30-0.85 is band; overall >= 0.95 easy."""
    if reached < min_reached:
        return "thin"
    rate = clears / reached
    if rate >= 0.95:
        return "easy"
    within = 0.8 * rate
    if 0.30 <= within <= 0.85:
        return "band"
    return "near" if within >= 0.2 else "low"


def merge_tables(tables, ids):
    """{(game_id, level): {reached, clears, tokens, actions}} over every table row; short ids (4 chars) resolved."""
    by4 = {g[:4]: g for g in ids}
    out = {}
    for t in tables:
        for r in t["levels"]:
            gid = r["game_id"] if r["game_id"] in ids else by4.get(r["game_id"][:4])
            if gid is None:
                continue
            o = out.setdefault((gid, int(r["level"])), {"reached": 0, "clears": 0, "tokens": [], "actions": [],
                                                         "medians": [], "p80s": [], "cap": None})
            o["reached"] += int(r["reached"])
            o["clears"] += int(r["clears"])
            o["tokens"] += [int(x) for x in r.get("tokens") or []]
            if not r.get("tokens") and r.get("median_tokens"):  # a screen row: medians only (RL screens)
                o["medians"] += [float(r["median_tokens"])] * max(1, int(r["clears"]))
                if r.get("p80_tokens"):
                    o["p80s"] += [float(r["p80_tokens"])] * max(1, int(r["clears"]))
            if r.get("cap"):
                o["cap"] = int(r["cap"])
            if r.get("median_actions") is not None:
                o["actions"] += [float(r["median_actions"])] * max(1, int(r["clears"]))
    return out


def p80(xs):
    xs = sorted(xs)
    return xs[int(0.8 * (len(xs) - 1))] if xs else None


def build(registry, tables, extra=(), hashes=None, kinds=("band", "near"), cap_factor=1.25, cap_min=4000,
          cap_max=64000, baseline_factor=1.3, max_level=None, sources=None, human=None, admit="frontier",
          cap_nodata=None, games_with_rows=True, always=()):
    """``human``: {game_id: [baseline actions per level]} (public games' metadata): used as is, no factor.
    ``admit``: "frontier" (levels 1..deepest band/near level, a low or thin level ends the chain) or "reached" (owner
    22 Oct: solve rate never selects; every level some probe reached, contiguous from L1, the curriculum's reward
    score decides; a level no probe cleared takes cap_max and the probes' censored tokens as its references) or "all"
    (every level of every game with a probe row, plus ``always`` codes; a level no try reached has
    prior 0/0 and gets starts from the harvest or a set_level start). ``cap_nodata``: the cap of a level no probe
    cleared (default cap_max); cleared levels keep p80 x cap_factor up to cap_max."""
    games = {g["game_id"]: g for g in registry["games"]}
    held = set(registry.get("held_out") or HELD_OUT_10) | set(HELD_OUT_10)
    rows = merge_tables(list(tables) + [{"levels": list(extra)}] if extra else list(tables), set(games))
    sets, points, report = [], [], {"games": 0, "units": 0, "by_group": {}, "by_kind": {}, "no_solution": 0,
                                    "skipped_held_out": 0}
    for gid in sorted(games):
        g = games[gid]
        code = gid[:4]
        if g.get("split") not in TRAIN_SPLITS or g.get("excluded") or code in held:
            report["skipped_held_out"] += code in held
            continue
        group = GROUP.get(g["source"])
        if group is None or (sources and group not in sources):
            continue
        chain = []
        if admit == "all" and code not in always and not any(rows.get((gid, lv)) for lv in (1, 2)):
            continue
        for level in range(1, int(g.get("levels") or 0) + 1):
            if max_level and level > max_level:
                break
            o = rows.get((gid, level))
            k = kind(o["reached"], o["clears"]) if o else "thin"
            if admit == "all":
                o = o or {"reached": 0, "clears": 0, "tokens": [], "medians": [], "p80s": [], "actions": [], "cap": None}
            elif admit == "reached":
                if not o or not o["reached"]:
                    break
            elif k in ("low", "thin"):
                break
            chain.append((level, k, o))
        while admit == "frontier" and chain and chain[-1][1] not in kinds:  # ends at its deepest frontier level
            chain.pop()
        if not chain:
            continue
        report["games"] += 1
        sol = g.get("solution_lengths") or []
        for level, k, o in chain:
            med_t = statistics.median(o["tokens"] or o["medians"]) if (o["tokens"] or o["medians"]) else None
            nodata = cap_nodata or cap_max
            measured = med_t is not None
            if med_t is None and admit in ("reached", "all"):  # no probe cleared it: the cap sizes it, no reward ref
                med_t, o = nodata, {**o, "p80s": [nodata / cap_factor]}
            med_a = statistics.median(o["actions"]) if o["actions"] and measured else None
            if not med_t:
                continue
            hb = (human or {}).get(gid) or []
            if level - 1 < len(hb) and hb[level - 1]:
                base = int(hb[level - 1])
            else:
                base = math.ceil(baseline_factor * sol[level - 1]) if level - 1 < len(sol) and sol[level - 1] else None
            report["no_solution"] += base is None
            hi = p80(o["tokens"]) or (statistics.median(o["p80s"]) if o["p80s"] else None) or med_t
            cap = o["cap"] or int(min(cap_max, max(cap_min, round(cap_factor * hi, -3))))
            sets.append({"name": f"{code}-L{level}", "game": code, "level": level, "role": "train",
                         **({"fresh": True} if level == 1 else {}), "cap": cap, "source": group,
                         "family": g.get("family") or code,
                         "ref": {"tokens": int(med_t) if measured else None, "actions": med_a,
                                 "baseline": base or med_a, "measured": measured,
                                 **({"n": max(1, int(o["clears"]))} if measured else {})},
                         "prior": {"tries": o["reached"], "clears": o["clears"], "median_to_clear": int(med_t)}})
            report["units"] += 1
            report["by_group"][group] = report["by_group"].get(group, 0) + 1
            report["by_kind"][k] = report["by_kind"].get(k, 0) + 1
        baselines = ([int(x) if x else None for x in (human or {}).get(gid) or []]
                     or [math.ceil(baseline_factor * x) if x else None for x in sol] or None)
        points.append({"id": f"{code}-game", "game": gid, "split": "dev", "level_index": 0, "level": 1,
                       "through": True, "total_levels": int(g.get("levels") or len(sol)), "baselines": baselines,
                       "baseline_actions": None, "prefix_actions": 0, "prefix": [],
                       "board_hash": (hashes or {}).get(gid)})
    bad = [s["name"] for s in sets if s["game"] in held]
    if bad:
        raise ValueError(f"held-out games in the training sets: {bad}")
    report["families"] = len({s["family"] for s in sets})
    report["no_board_hash"] = sum(1 for p in points if not p["board_hash"])
    present = {s["source"] for s in sets}
    return {"sets": sets, "points": points, "report": report,
            "shares": {k: v for k, v in SHARES.items() if k in present}}


PLACEHOLDER_TOKENS = (64000,)  # cap_nodata values written as ref.tokens for never-cleared levels before 24 Oct


def placeholder_refs(sets):
    """Names of training sets whose reward ref is neither a measured base clear (``measured: True`` with ``n``) nor a
    marked derivation (``derived: True`` with its ``basis``), e.g. a known no-data cap as tokens with no clears."""
    bad = []
    for s in sets:
        if s.get("role", "train") != "train":
            continue
        ref, prior = s.get("ref") or {}, s.get("prior") or {}
        if ref.get("measured") is True:
            ok = bool(ref.get("tokens")) and bool(ref.get("n"))
        elif ref.get("derived") is True:
            ok = bool(ref.get("tokens")) and bool(ref.get("basis")) and ref.get("measured") is False
        elif ref.get("measured") is False:
            ok = ref.get("tokens") is None and ref.get("actions") is None
        else:
            ok = ref.get("tokens") is None or (int(prior.get("clears") or 0) > 0 and ref["tokens"] not in PLACEHOLDER_TOKENS)
        if not ok:
            bad.append(s["name"])
    return bad


def apply_refs(sets, table):
    """Reward refs from a base-clears table (``data/envs/mg-refs-*.json``: ``refs`` {set: {tokens, actions, n, src}} =
    medians over base (version-0) clears of the level; ``derived`` {set: {tokens, actions, basis}} for levels no base
    try cleared: actions = optimal length x the base's median actions/optimal, tokens = the base's median tokens to
    clear, both over the game's (else family's) measured levels; ``no_base`` {set: evidence}). A set the table measured takes its tokens/actions (``measured: True``, ``n``,
    ``src``); an env-table ref with clears behind it is kept and marked measured; else a derived ref (``derived:
    True``, ``measured: False``, ``basis``); else ``measured: False`` with no tokens/actions. RL's order: screen >
    measured > derived > its v0-group fallback. Returns (sets, report)."""
    refs, der = table.get("refs") or {}, table.get("derived") or {}
    out, rep = [], {"from_table": [], "derived": [], "kept": 0, "unmeasured": []}
    for s in sets:
        s = dict(s)
        if s.get("role", "train") == "train":
            ref, prior, m = dict(s.get("ref") or {}), s.get("prior") or {}, refs.get(s["name"])
            real = (ref.get("measured") is True or (ref.get("measured") is None and ref.get("tokens")
                    and int(prior.get("clears") or 0) > 0 and ref["tokens"] not in PLACEHOLDER_TOKENS))
            if real:
                ref.setdefault("n", int(prior.get("clears") or 0))
                ref.setdefault("src", "env-table")
                ref["measured"] = True
                rep["kept"] += 1
            elif m:
                ref.update(tokens=int(m["tokens"]), actions=m.get("actions"), n=int(m["n"]), src=m.get("src"),
                           measured=True)
                ref["baseline"] = ref.get("baseline") or m.get("actions")
                rep["from_table"].append(s["name"])
            elif s["name"] in der:
                d = der[s["name"]]
                ref.update(tokens=int(d["tokens"]), actions=d["actions"], basis=d["basis"], derived=True, measured=False)
                rep["derived"].append(s["name"])
            else:
                ref.update(tokens=None, actions=None, measured=False)
                rep["unmeasured"].append(s["name"])
            s["ref"] = ref
        out.append(s)
    bad = placeholder_refs(out)
    if bad:
        raise ValueError(f"placeholder refs left: {bad[:10]}")
    return out, rep
