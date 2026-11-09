"""Scheduling analysis of a finished fixed-2x arena run on the token clock (hero-v2 2 h, 29 September; CPU only).

Writes the numbers behind docs/validation-20260927/hero-v2-analysis/scheduling.md:

- ``tokens``: completion tokens by phase and by the level being played, tokens per cleared level by level index,
  score points per million tokens by allocation category;
- ``budgets``: parks by reason and level, level-budget parks at level >= 1 and what the next tenure did, level-0
  tenures, regrants, the budgets against the tokens levels actually took (sb26), deep games' slot share and waits;
- ``opening``: tokens / requests / time per game before the switch, L1 clears in the opening vs phase 2;
- ``rank``: ``arena.rank.score`` evaluated on the recorded trajectories every 60 s (which parked games rank would
  have promoted, which focus games it would have demoted), and the assumption checks (clears per token, recency:
  the clear hazard against the tokens already spent on a level, successive levels' token costs).

    PYTHONPATH=src uv run python scripts/arena_analysis/hero_v2_scheduling.py <trial dir> --config <comparison.json>
        [--out result.json]

A clear of level k is credited to the category that played level k - 1 when it happened; a game's score is
sum_{k <= levels} k / (N (N + 1) / 2) (N = its level count), the arena score the mean over the games.
"""

import argparse
import bisect
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HOLDOUT = ("sb26", "re86", "s5i5", "su15", "dc22", "cd82", "g50t", "ls20", "tr87")
WALL = 7200.0


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(q * len(values)))]


def r(x, n=1):
    return None if x is None else round(x, n)


class Game:
    def __init__(self, raw, names, switch):
        self.raw = raw
        self.id = raw["game_id"]
        self.index = raw["index"]
        self.kind = names[self.index % len(names)][:4]
        self.holdout = self.kind in HOLDOUT
        self.n = raw["total_levels"] or 8
        self.levels = raw["levels"]
        self.t1 = raw["t1_seconds"]  # tokens on the token clock
        self.clears = raw["time_to_level"]  # level, wall_seconds, budget_seconds (= gameplay tokens at the clear)
        self.clear_walls = [c["wall_seconds"] for c in self.clears]
        self.charges = raw["charges"]
        self.admissions = raw["admissions"]
        self.switch = switch

    def level_at(self, wall):
        """Levels reached strictly before ``wall`` (a reply is charged before its action reports the clear)."""
        return bisect.bisect_left(self.clear_walls, wall)

    def points(self, level):
        return 100.0 * level / (self.n * (self.n + 1) / 2) / 110.0

    def spans(self):
        """(start, stop, reason) of active spans."""
        out, since, reason = [], None, None
        for e in self.admissions:
            if e["event"] == "admit":
                since, reason = e["wall_seconds"], e["reason"]
            elif e["event"] in ("park", "finish") and since is not None:
                out.append((since, e["wall_seconds"], reason, e["reason"]))
                since = None
        if since is not None:
            out.append((since, WALL, reason, "wall_end"))
        return out

    def gameplay_tokens_at(self, wall):
        return sum(c["completion_tokens"] for c in self.charges if c["purpose"] == "gameplay" and c["wall_seconds"] <= wall)

    def level_token_marks(self):
        """Gameplay tokens at each clear (the budget clock)."""
        return [c["budget_seconds"] for c in self.clears]


def load(trial_dir, config):
    trial_dir = Path(trial_dir)
    report = json.loads((trial_dir / "arena.json").read_text())
    names = json.loads(Path(config).read_text())["games"]
    switch = report["scheduler"]["switch_wall_seconds"]
    games = [Game(g, names, switch) for g in report["games"]]
    return report, games, switch


def category(phase, level):
    if phase == "opening":
        return "opening (all games)"
    return {0: "phase 2, level 0", 1: "phase 2, playing L2", 2: "phase 2, playing L3"}.get(level, "phase 2, playing L4+")


# ---------------------------------------------------------------------------------------------- 1. tokens

def tokens(games, switch):
    by = defaultdict(lambda: defaultdict(int))
    compaction = defaultdict(int)
    for g in games:
        for c in g.charges:
            phase = "opening" if c["wall_seconds"] <= switch else "phase2"
            if c["purpose"] != "gameplay":
                compaction[phase] += c["completion_tokens"]
                continue
            lvl = g.level_at(c["wall_seconds"])
            by[phase][min(lvl, 4)] += c["completion_tokens"]
    total = sum(sum(v.values()) for v in by.values()) + sum(compaction.values())
    table = {}
    for phase in ("opening", "phase2"):
        table[phase] = {f"level {k}{'+' if k == 4 else ''}": v for k, v in sorted(by[phase].items())}
        table[phase]["compaction"] = compaction[phase]
    # tokens per cleared level by level index: gameplay tokens on level k-1 for every clear of level k, and the
    # tokens still being spent on uncleared levels at the end (censored)
    per_level, censored = defaultdict(list), defaultdict(list)
    for g in games:
        marks = [0] + g.level_token_marks()
        for k in range(1, len(marks)):
            per_level[k].append(marks[k] - marks[k - 1])
        spent = g.raw["budget_seconds"] - marks[-1]
        if g.levels < g.n:
            censored[g.levels + 1].append(spent)
    per_level_table = {}
    for k in sorted(set(per_level) | set(censored)):
        c, u = per_level.get(k, []), censored.get(k, [])
        spent = sum(c) + sum(u)
        per_level_table[f"L{k}"] = {"clears": len(c), "median_tokens_to_clear": quantile(c, 0.5),
                                    "mean_tokens_to_clear": r(statistics.mean(c), 0) if c else None,
                                    "uncleared_attempts": len(u), "uncleared_tokens": sum(u),
                                    "tokens_per_clear_incl_failed": r(spent / len(c), 0) if c else None}
    # score per million tokens by allocation category
    cat_tokens, cat_points, cat_clears = defaultdict(int), defaultdict(float), defaultdict(int)
    for g in games:
        for c in g.charges:
            if c["purpose"] != "gameplay":
                continue
            phase = "opening" if c["wall_seconds"] <= switch else "phase2"
            cat_tokens[category(phase, g.level_at(c["wall_seconds"]))] += c["completion_tokens"]
        for c in g.clears:
            phase = "opening" if c["wall_seconds"] <= switch else "phase2"
            cat = category(phase, c["level"] - 1)
            cat_points[cat] += g.points(c["level"])
            cat_clears[cat] += 1
    categories = {}
    for cat in sorted(cat_tokens):
        categories[cat] = {"tokens": cat_tokens[cat], "clears": cat_clears[cat], "points": r(cat_points[cat], 3),
                           "points_per_million_tokens": r(cat_points[cat] / cat_tokens[cat] * 1e6, 3),
                           "tokens_per_clear": r(cat_tokens[cat] / cat_clears[cat], 0) if cat_clears[cat] else None}
    return {"total_completion_tokens": total, "by_phase_and_level": table, "per_level": per_level_table,
            "categories": categories}


# ---------------------------------------------------------------------------------------------- 2. budgets

def tenures(games, switch):
    """Phase-2 tenures: game, start/stop, reason in/out, levels at start, clears in it, gameplay tokens in it."""
    out = []
    for g in games:
        for start, stop, reason_in, reason_out in g.spans():
            s0 = max(start, switch)
            if stop <= s0 + 1.0:  # parked at the switch (the park is stamped a few ms after it)
                continue
            if start < switch:
                reason_in = "opening_keep"
            lv0 = sum(1 for w in g.clear_walls if w <= s0)
            tok = sum(c["completion_tokens"] for c in g.charges
                      if c["purpose"] == "gameplay" and s0 < c["wall_seconds"] <= stop + 1.0)
            clears = [c for c in g.clears if s0 < c["wall_seconds"] <= stop + 2.0]
            out.append({"game": g.id, "kind": g.kind, "start": s0, "stop": stop, "seconds": stop - s0,
                        "reason_in": reason_in, "reason_out": reason_out, "levels": lv0, "tokens": tok,
                        "clears": len(clears), "first_clear_tokens": None, "t1": g.t1})
    return out


def budgets(games, switch, report):
    parks = Counter()
    noop = Counter()
    events = report["scheduler"]["events"]
    for i, e in enumerate(events):
        if e["event"] != "park":
            continue
        parks[(e["reason"], "L0" if e["levels"] == 0 else "L1+")] += 1
        nxt = events[i + 1] if i + 1 < len(events) else None
        if (nxt and nxt["event"] == "admit" and nxt["game"] == e["game"]
                and abs(nxt["wall_seconds"] - e["wall_seconds"]) < 0.01):
            noop[(e["reason"], "L0" if e["levels"] == 0 else "L1+")] += 1
    ten = tenures(games, switch)
    # level-0 tenures in phase 2 (not the opening keeps)
    l0 = [t for t in ten if t["levels"] == 0]
    l0_total_slot = sum(t["seconds"] for t in l0)
    all_slot = sum(t["seconds"] for t in ten)
    conv = [t for t in l0 if t["clears"] > 0]
    level0 = {"tenures": len(l0), "median_seconds": r(quantile([t["seconds"] for t in l0], 0.5)),
              "median_tokens": quantile([t["tokens"] for t in l0], 0.5),
              "converted_to_L1": len(conv), "conversion_share": r(len(conv) / len(l0), 3),
              "slot_seconds": r(l0_total_slot, 0), "share_of_phase2_slot_seconds": r(l0_total_slot / all_slot, 3),
              "tokens": sum(t["tokens"] for t in l0),
              "tokens_per_L1": r(sum(t["tokens"] for t in l0) / max(1, sum(t["clears"] > 0 for t in l0)), 0)}
    # level-0 grants: the re-admission grant against the tokens the tenure actually used
    grants = []
    for g in games:
        for e in g.admissions:
            if e["event"] == "admit" and e["reason"] == "regrant" and e["levels"] == 0:
                grants.append(e["budget"])
    level0["grant_tokens"] = {"median": quantile(grants, 0.5), "p10": quantile(grants, 0.1),
                              "p90": quantile(grants, 0.9)}
    # level >= 1 budget parks: budget, the level's tokens so far, what happened next
    deep_parks = []
    for g in games:
        adm = g.admissions
        for i, e in enumerate(adm):
            if e["event"] != "park" or e["reason"] != "level_budget" or e["levels"] == 0:
                continue
            level = e["levels"]
            level_start_tokens = g.level_token_marks()[level - 1]
            nxt = next((a for a in adm[i + 1:] if a["event"] == "admit"), None)
            clear = next((c for c in g.clears if c["level"] == level + 1), None)
            row = {"game": g.id, "kind": g.kind, "level": level, "wall": e["wall_seconds"], "budget": e["budget"],
                   "t1": g.t1, "tokens_on_level": e["budget_seconds"] - level_start_tokens,
                   "readmitted_wall": nxt["wall_seconds"] if nxt else None,
                   "parked_seconds": r((nxt["wall_seconds"] if nxt else WALL) - e["wall_seconds"], 0),
                   "noop": bool(nxt and abs(nxt["wall_seconds"] - e["wall_seconds"]) < 0.01)}
            if clear and nxt and clear["wall_seconds"] >= nxt["wall_seconds"]:
                row["cleared_after_readmit_tokens"] = clear["budget_seconds"] - nxt["budget_seconds"]
                row["cleared_after_readmit_seconds"] = r(clear["wall_seconds"] - nxt["wall_seconds"], 0)
            elif clear and clear["wall_seconds"] < e["wall_seconds"] + 5:
                row["cleared_in_flight"] = True
            if nxt:  # the tenure after the re-admission: its tokens and clears
                after = next((a for a in adm[adm.index(nxt) + 1:] if a["event"] in ("park", "finish")), None)
                stop = after["wall_seconds"] if after else WALL
                row["next_tenure_tokens"] = (after["budget_seconds"] if after else g.raw["budget_seconds"]) - nxt["budget_seconds"]
                row["next_tenure_clears"] = sum(1 for c in g.clears if nxt["wall_seconds"] < c["wall_seconds"] <= stop + 2)
                row["next_tenure_points"] = r(sum(g.points(c["level"]) for c in g.clears
                                                if nxt["wall_seconds"] < c["wall_seconds"] <= stop + 2), 4)
            deep_parks.append(row)
    # tenure outcomes by entry reason and level
    outcome = defaultdict(lambda: {"tenures": 0, "with_clear": 0, "clears": 0, "tokens": 0, "seconds": 0.0})
    for t in ten:
        key = f"{t['reason_in']} @ {'L0' if t['levels'] == 0 else 'L1' if t['levels'] == 1 else 'L2+'}"
        o = outcome[key]
        o["tenures"] += 1
        o["with_clear"] += t["clears"] > 0
        o["clears"] += t["clears"]
        o["tokens"] += t["tokens"]
        o["seconds"] += t["seconds"]
    for o in outcome.values():
        o["seconds"] = round(o["seconds"])
        o["tokens_per_clear"] = round(o["tokens"] / o["clears"]) if o["clears"] else None
    # budgets against what levels took: per game kind, budget at each level vs tokens to clear it
    ratio = []  # tokens to clear level k (k >= 2) / T1, and the budget multiple 2k
    for g in games:
        marks = [0] + g.level_token_marks()
        for k in range(2, len(marks)):
            ratio.append({"kind": g.kind, "level": k, "tokens": marks[k] - marks[k - 1], "t1": g.t1,
                          "over_t1": r((marks[k] - marks[k - 1]) / g.t1, 2), "budget_multiple": 2 * k})
    regrant_summary = {}
    for label, rows in (("noop (readmitted at once)", [x for x in deep_parks if x["noop"]]),
                        ("readmitted later", [x for x in deep_parks if not x["noop"] and x["readmitted_wall"]]),
                        ("never readmitted", [x for x in deep_parks if not x["readmitted_wall"]])):
        tok = sum(x.get("next_tenure_tokens", 0) for x in rows)
        pts = sum(x.get("next_tenure_points") or 0 for x in rows)
        regrant_summary[label] = {"parks": len(rows), "next_tenure_tokens": tok,
                                  "next_tenure_clears": sum(x.get("next_tenure_clears", 0) for x in rows),
                                  "points": r(pts, 3), "points_per_million_tokens": r(pts / tok * 1e6, 3) if tok else None,
                                  "parked_seconds": sum(x["parked_seconds"] for x in rows)}
    completions = sorted(c["completion_tokens"] for g in games for c in g.charges if c["purpose"] == "gameplay")
    cap = report.get("completion_cap", 16384)
    completion = {"replies": len(completions), "median": quantile(completions, 0.5), "p90": quantile(completions, 0.9),
                  "p99": quantile(completions, 0.99), "max": completions[-1],
                  "share_over_8k": r(sum(c > 8192 for c in completions) / len(completions), 4),
                  "at_cap": sum(c >= cap for c in completions),
                  "length_truncations": sum(g.raw.get("length_truncations", 0) for g in games)}
    return {"parks": {f"{a} {b}": v for (a, b), v in sorted(parks.items())}, "regrants_after_level_budget_parks":
            regrant_summary, "completion_tokens_per_reply": completion,
            "noop_parks": {f"{a} {b}": v for (a, b), v in sorted(noop.items())},
            "level0": level0, "deep_parks": deep_parks, "tenure_outcomes": dict(outcome),
            "level_cost_over_t1": ratio, "tenures": ten}


def sb26(games, switch, report):
    rows = []
    for g in games:
        if g.kind != "sb26":
            continue
        marks = g.level_token_marks()
        spent_l1 = g.raw["budget_seconds"] - marks[0]
        parks = [e for e in g.admissions if e["event"] == "park"]
        admits = [e for e in g.admissions if e["event"] == "admit"]
        active = sum(stop - max(start, switch) for start, stop, *_ in g.spans() if stop > switch)
        rows.append({"game": g.id, "t1": g.t1, "l1_wall": g.clear_walls[0], "l2_budget": 4 * g.t1,
                     "tokens_on_L1": spent_l1, "over_budget": r(spent_l1 / (4 * g.t1), 2),
                     "parks": len(parks), "admissions": len(admits), "phase2_active_seconds": r(active, 0),
                     "compactions": g.raw["compactions"], "max_prompt": g.raw["max_prompt_tokens"],
                     "actions": g.raw["actions"]})
    return rows


def deep_games(games, switch, report, gate_waits):
    """Games with >= 3 levels: phase-2 active share, parks, parked time, gate waits (``gate_waits``: game -> the
    gate's own FIFO wait of each attempt admitted after the switch, ``gate_replay.recorded_attempts``;
    ``eligible_wait_seconds`` would also count the opening queue of a request queued at the switch and admitted at
    the game's re-admission)."""
    out = []
    for g in games:
        if g.levels < 3:
            continue
        active = sum(stop - max(start, switch) for start, stop, *_ in g.spans() if stop > switch)
        att = [a for a in g.raw["admission_timing"]["attempts"] if a.get("admitted_wall_seconds", 0) > switch]
        waits = gate_waits.get(g.id, [])
        first = next((e["wall_seconds"] for e in g.admissions
                      if e["event"] == "admit" and e["wall_seconds"] > switch + 1.0), None)
        kept = any(e["event"] == "grant" and e["reason"] == "opening_switch" for e in g.admissions)
        parks = [e for e in g.admissions if e["event"] == "park" and e["wall_seconds"] > switch]
        out.append({"game": g.id, "kind": g.kind, "levels": g.levels, "t1": g.t1,
                    "phase2_active_share": r(active / (WALL - switch), 3), "phase2_parks": len(parks),
                    "kept_at_switch": kept, "first_phase2_admission": r(first, 0),
                    "park_reasons": dict(Counter(e["reason"] for e in parks)),
                    "mean_gate_wait": r(statistics.mean(waits), 2) if waits else None,
                    "max_gate_wait": r(max(waits), 1) if waits else None, "requests": len(att),
                    "max_prompt": g.raw["max_prompt_tokens"], "compactions": g.raw["compactions"],
                    "clear_walls": [r(w, 0) for w in g.clear_walls]})
    return out


def slot_time(games, switch, step=1.0):
    """Phase-2 slot seconds and gameplay tokens by the level being played; the backlog cycle of the switch-parked
    games; how long level >= 1 games waited parked while level-0 games held focus slots."""
    spans = {g.id: [(max(a, switch), b) for a, b, *_ in g.spans() if b > switch + 1.0] for g in games}
    seconds = defaultdict(float)
    t = switch
    blocked = 0.0  # seconds x games: a parked level >= 1 game waits while a level-0 game holds a slot
    blocked_by_kind = defaultdict(float)
    while t < WALL:
        focus_l0, parked_deep = 0, []
        for g in games:
            active = any(a <= t < b for a, b in spans[g.id])
            level = g.level_at(t)
            if active:
                seconds[min(level, 4)] += step
                focus_l0 += level == 0
            elif level >= 1 and level < g.n:
                parked_deep.append(g)
        n = min(focus_l0, len(parked_deep))
        blocked += n * step
        for g in parked_deep:
            blocked_by_kind[g.kind] += step * n / len(parked_deep) if parked_deep else 0.0
        t += step
    total = sum(seconds.values())
    tok = defaultdict(int)
    for g in games:
        for c in g.charges:
            if c["purpose"] == "gameplay" and c["wall_seconds"] > switch:
                tok[min(g.level_at(c["wall_seconds"]), 4)] += c["completion_tokens"]
    by_level = {f"level {k}{'+' if k == 4 else ''}": {
        "slot_seconds": round(v), "share": r(v / total, 3), "tokens": tok[k],
        "token_share": r(tok[k] / sum(tok.values()), 3), "tokens_per_slot_second": r(tok[k] / v, 1)}
        for k, v in sorted(seconds.items())}
    # first phase-2 admission of every game parked at the switch (the level-0 audition cycle)
    first = []
    for g in games:
        adm = [e for e in g.admissions if e["event"] == "admit" and e["wall_seconds"] > switch + 1.0]
        parked_at_switch = any(e["event"] == "park" and e["reason"] == "opening_switch" for e in g.admissions)
        if parked_at_switch:
            first.append((adm[0]["wall_seconds"] if adm else None, g))
    admitted = sorted(w for w, _ in first if w is not None)
    # phase-2 L1 clears: wall from the switch to the clear, and gameplay tokens from the tenure start to the clear
    l1 = []
    for g in games:
        if not g.clears or g.clear_walls[0] <= switch:
            continue
        wall = g.clear_walls[0]
        start = max((a for a, b in spans[g.id] if a <= wall + 2.0), default=switch)  # a clear while parked
        before = g.gameplay_tokens_at(start)
        l1.append({"game": g.id, "kind": g.kind, "clear_wall": r(wall, 0), "tenure_start": r(start, 0),
                   "tokens_in_tenure_to_clear": g.clears[0]["budget_seconds"] - before,
                   "tokens_total_to_clear": g.clears[0]["budget_seconds"], "final_levels": g.levels})
    l1.sort(key=lambda x: x["clear_wall"])
    return {"by_level": by_level,
            "switch_parked_games": len(first), "first_readmission": {
                "admitted": len(admitted), "never": sum(1 for w, _ in first if w is None),
                "median_wall": r(quantile(admitted, 0.5), 0), "p90_wall": r(quantile(admitted, 0.9), 0),
                "last_wall": r(admitted[-1], 0) if admitted else None},
            "deep_parked_while_level0_focus_game_seconds": round(blocked),
            "deep_parked_while_level0_by_kind": {k: round(v) for k, v in sorted(blocked_by_kind.items(), key=lambda kv: -kv[1])},
            "phase2_l1_clears": l1}


# ---------------------------------------------------------------------------------------------- 3. opening

def opening(games, switch, report):
    tok, req, clears = [], [], []
    kinds = defaultdict(lambda: {"games": 0, "tokens": 0, "l1_opening": 0, "l1_phase2": 0, "l1_never": 0})
    for g in games:
        t = sum(c["completion_tokens"] for c in g.charges if c["wall_seconds"] <= switch)
        n = sum(1 for c in g.charges if c["wall_seconds"] <= switch)
        tok.append(t)
        req.append(n)
        k = kinds[g.kind]
        k["games"] += 1
        k["tokens"] += t
        if g.clears and g.clear_walls[0] <= switch:
            k["l1_opening"] += 1
            clears.append(g.clear_walls[0])
        elif g.clears:
            k["l1_phase2"] += 1
        else:
            k["l1_never"] += 1
    waits = [a["eligible_wait_seconds"] for g in games for a in g.raw["admission_timing"]["attempts"]
             if a.get("admitted_phase") == "opening"]
    l1_p2 = [g for g in games if g.clears and g.clear_walls[0] > switch]
    l1_p2_soon = [g for g in l1_p2 if g.clear_walls[0] <= switch + 30]
    return {"switch_seconds": switch, "tokens_per_game": {"mean": r(statistics.mean(tok), 0),
                                                          "median": quantile(tok, 0.5), "p10": quantile(tok, 0.1),
                                                          "p90": quantile(tok, 0.9), "total": sum(tok)},
            "requests_per_game": {"mean": r(statistics.mean(req), 1), "median": quantile(req, 0.5)},
            "gate_wait_per_request": {"mean": r(statistics.mean(waits), 1), "p50": r(quantile(waits, 0.5), 1),
                                      "p90": r(quantile(waits, 0.9), 1)},
            "l1_in_opening": len(clears), "l1_in_phase2": len(l1_p2),
            "l1_in_phase2_within_30s_of_switch": len(l1_p2_soon),
            "by_kind": dict(sorted(kinds.items()))}


# ---------------------------------------------------------------------------------------------- 5. rank

class Snapshot:
    """The attributes ``arena.rank.score`` reads, from a game's recorded trajectory at one wall time."""

    def __init__(self, g, wall):
        self.levels = g.level_at(wall + 1e-6)
        self.total_levels = g.n
        marks = [c["budget_seconds"] for c in g.clears[: self.levels]]
        gameplay = [c for c in g.charges if c["wall_seconds"] <= wall]
        self._tokens = sum(c["completion_tokens"] for c in gameplay)  # the rank clock counts all completion tokens
        gp = sum(c["completion_tokens"] for c in gameplay if c["purpose"] == "gameplay")
        prev = 0
        self.level_tokens = []
        for m in marks:
            self.level_tokens.append(m - prev)
            prev = m
        self._on_level = gp - prev

        class _Clock:
            def __init__(s, t):
                s.t = t

            def completion_tokens(s):
                return s.t
        self.clock = _Clock(self._tokens)

    def tokens_on_level(self):
        return self._on_level


def rank_counterfactual(games, switch, report, tick=60.0):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from arena import rank

    cfg = rank.settings({"slots": 23})
    spans = {g.id: g.spans() for g in games}

    def is_focus(g, wall):
        return any(a <= wall < b for a, b, *_ in spans[g.id])

    promote, demote = Counter(), Counter()
    promote_level, demote_level = Counter(), Counter()
    rows = []
    t = switch + tick
    while t < WALL:
        snaps = {g.id: Snapshot(g, t) for g in games}
        h = rank.horizon(list(snaps.values()), cfg)
        scores = {gid: rank.score(s, h, cfg)[0] for gid, s in snaps.items()}
        order = sorted(games, key=lambda g: (-scores[g.id], g.index))
        top = {g.id for g in order[:23]}
        focus = {g.id for g in games if is_focus(g, t)}
        ins = [g for g in games if g.id in top and g.id not in focus]
        outs = [g for g in games if g.id in focus and g.id not in top]
        for g in ins:
            promote[g.id] += 1
            promote_level[snaps[g.id].levels] += 1
        for g in outs:
            demote[g.id] += 1
            demote_level[snaps[g.id].levels] += 1
        rows.append({"wall": round(t), "horizon": round(h), "overlap": len(top & focus),
                     "top_level0": sum(1 for gid in top if snaps[gid].levels == 0),
                     "focus_level0": sum(1 for gid in focus if snaps[gid].levels == 0),
                     "promote": [(g.id, g.kind, snaps[g.id].levels) for g in ins][:12],
                     "demote": [(g.id, g.kind, snaps[g.id].levels) for g in outs][:12]})
        t += tick
    by_id = {g.id: g for g in games}

    def named(counter):
        return [{"game": gid, "kind": by_id[gid].kind, "ticks": n, "final_levels": by_id[gid].levels}
                for gid, n in counter.most_common(20)]
    return {"ticks": len(rows), "mean_overlap_of_top23_with_focus": r(statistics.mean(x["overlap"] for x in rows), 1),
            "mean_level0_in_top23": r(statistics.mean(x["top_level0"] for x in rows), 1),
            "mean_level0_in_focus": r(statistics.mean(x["focus_level0"] for x in rows), 1),
            "promote_ticks_by_level": dict(sorted(promote_level.items())),
            "demote_ticks_by_level": dict(sorted(demote_level.items())),
            "most_promoted": named(promote), "most_demoted": named(demote),
            "timeline": rows[:: 10]}


def hazard(games, switch):
    """Clear hazard against the gameplay tokens already spent on the level (all levels; level 0 includes the
    opening): P(clear within the next 10K tokens | x tokens spent, not yet cleared), from each level attempt."""
    attempts = []  # (level played, tokens spent in total on it, cleared)
    for g in games:
        marks = [0] + g.level_token_marks()
        for k in range(1, len(marks)):
            attempts.append((k - 1, marks[k] - marks[k - 1], True))
        if g.levels < g.n:
            attempts.append((g.levels, g.raw["budget_seconds"] - marks[-1], False))
    bins = [0, 5_000, 10_000, 20_000, 40_000, 80_000]
    out = {}
    for label, sel in (("level 0", lambda a: a[0] == 0), ("level >= 1", lambda a: a[0] >= 1)):
        rows = []
        pool = [a for a in attempts if sel(a)]
        for lo, hi in zip(bins, bins[1:] + [math.inf]):
            at_risk = [a for a in pool if a[1] > lo]
            cleared = [a for a in at_risk if a[2] and a[1] <= hi]
            exposure = sum(min(a[1], hi) - lo for a in at_risk)  # tokens spent inside the bin (censoring included)
            rows.append({"from_tokens": lo, "to_tokens": hi if hi < math.inf else None, "at_risk": len(at_risk),
                         "clears": len(cleared),
                         "clears_per_10k_tokens": r(len(cleared) / exposure * 1e4, 3) if exposure > 0 else None})
        out[label] = rows
    # depth persistence: P(clear level k + 1 | cleared level k), and T1 against the levels after the first
    reach = Counter()
    for g in games:
        for k in range(g.levels + 1):
            reach[k] += 1
    persistence = {f"P(L{k + 1} | L{k})": r(reach[k + 1] / reach[k], 2) for k in range(0, 5) if reach[k]}
    t1_rows = [(g.t1, g.levels - 1) for g in games if g.t1]
    ranks = lambda xs: [sorted(xs).index(x) for x in xs]  # noqa: E731
    t1_spearman = statistics.correlation(ranks([a for a, _ in t1_rows]), ranks([b for _, b in t1_rows]))
    # successive levels: does a fast level predict a fast next level?
    pairs = []
    for g in games:
        marks = [0] + g.level_token_marks()
        costs = [marks[k] - marks[k - 1] for k in range(1, len(marks))]
        for a, b in zip(costs, costs[1:]):
            pairs.append((a, b))
    lx = [math.log(a) for a, b in pairs if a > 0 and b > 0]
    ly = [math.log(b) for a, b in pairs if a > 0 and b > 0]
    corr = statistics.correlation(lx, ly) if len(lx) > 2 else None
    ratios = [b / a for a, b in pairs if a > 0]
    # did the game that cleared most recently (fewest tokens since its last clear) clear next?
    return {"hazard": out, "depth_persistence": persistence, "t1_vs_later_levels_spearman": r(t1_spearman, 2),
            "successive_levels": {"pairs": len(pairs), "log_correlation": r(corr, 2),
                                                 "median_next_over_prev": r(quantile(ratios, 0.5), 2),
                                                 "p25": r(quantile(ratios, 0.25), 2), "p75": r(quantile(ratios, 0.75), 2)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_dir")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report, games, switch = load(args.trial_dir, args.config)
    import gate_replay

    gate_report, budget = gate_replay.load_run(args.trial_dir)
    gate_waits = defaultdict(list)
    for row in gate_replay.recorded_attempts(gate_report, budget):
        if row["arrival"] >= switch:  # queued in phase 2 (not the opening queue at the switch)
            gate_waits[row["game"]].append(row["admitted"] - row["arrival"])
    result = {"switch_seconds": switch, "score": report["mean_score"], "levels": report["levels"],
              "tokens": tokens(games, switch), "budgets": budgets(games, switch, report),
              "sb26": sb26(games, switch, report), "deep_games": deep_games(games, switch, report, gate_waits),
              "opening": opening(games, switch, report), "slots": slot_time(games, switch),
              "rank": rank_counterfactual(games, switch, report),
              "rank_assumptions": hazard(games, switch)}
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
